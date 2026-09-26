//! Spikes as samples: a spiking network that performs Bayesian inference by firing.
//!
//! # What the mechanism is
//!
//! A spiking network is stochastic almost everywhere — channel noise, unreliable release, jittered
//! arrival times — and the usual engineering response is to treat that as a defect to be averaged
//! away. One line of theory treats it as the computation. In **neural sampling** (Buesing, Bill,
//! Nessler & Maass, *Neural Dynamics as Sampling: A Model for Stochastic Computation in Recurrent
//! Networks of Spiking Neurons*, `PLoS` Computational Biology 7(11):e1002211, 2011) the network's
//! trajectory through its own state space **is** a Markov chain Monte Carlo run over a probability
//! distribution, and the spike train is the sequence of samples. The network is not computing an
//! answer and then reporting noise around it; the noise is the answer, spread over time.
//!
//! The variable being sampled is not the membrane potential. It is a binary `z_k` per neuron, with
//! `z_k = 1` meaning *neuron k fired within the last `tau` ticks*. The refractory period is not a
//! nuisance here either — it is what gives `z_k` a duration, and therefore what turns a train of
//! instantaneous events into a state that can be read.
//!
//! The condition that makes this work is the paper's **neural computability condition (NCC)**: the
//! membrane potential must equal the conditional log-odds of the neuron's own variable,
//!
//! ```text
//! u_k(t) = log [ p(z_k = 1 | z_\k(t)) / p(z_k = 0 | z_\k(t)) ]
//! ```
//!
//! For a Boltzmann distribution `p(z) ∝ exp(Σ b_k z_k + ½ Σ_{k≠j} W_kj z_k z_j)` that log-odds is
//! exactly `b_k + Σ_j W_kj z_j` — a **weighted sum of the inputs a neuron already has**. That is
//! the whole content of the theory: a linear synaptic integrator is a log-odds computer, provided
//! the target distribution has no interactions above second order. [`PairwiseFit`] measures that
//! proviso, and [`PairwiseFit::residual`] is nonzero exactly when it fails.
//!
//! # What it buys
//!
//! A **posterior, not a point estimate**, from the same hardware and the same spikes. Read the
//! train for 10 ms and you have a coarse posterior; read for 100 ms and it sharpens; there is no
//! separate uncertainty pathway to build, and no forward pass to re-run. It is *anytime*
//! inference, which is what a robot that must act before it is certain actually needs.
//!
//! It also gives a principled account of what a firing rate means. A rate is a marginal
//! probability. A correlation between two cells is a joint. Explaining-away between competing
//! hypotheses is the negative coupling `W_kj` that the same synapse would carry anyway.
//!
//! # What it costs
//!
//! **Correlated samples.** A chain's successive states are not independent, so `n` ticks of
//! observation are worth fewer than `n` samples — often far fewer. Reporting `sd / sqrt(n)` as an
//! error bar on a quantity read off a chain is the single most common way this family of methods
//! produces a confident wrong answer, and it is not a subtle failure: for the correlated chain in
//! `an_error_bar_built_on_the_raw_count_covers_a_third_of_the_time_it_claims_ninety_five`, a
//! nominal 95% interval covers the truth **35%** of the time. [`Estimate`] therefore carries both
//! bars, and [`Estimate::sem`] — the one built on [`Estimate::ess`] — is the one to report.
//!
//! **Update order matters.** [`Scan::Random`] samples the target exactly; [`Scan::Parallel`], in
//! which every neuron updates every tick as real neurons do, **does not** once the units are
//! coupled. This implementation measures the gap rather than asserting either way: see
//! `parallel_updates_are_exact_without_coupling_and_wrong_with_it`, where the same model, the same
//! seed and the same million ticks at `tau = 3` give a total-variation distance of **0.0031**
//! under random scan and **0.092** under parallel — a factor of 30. The irony is worth the
//! lesson: the biologically honest update is the statistically dishonest one, and it is also the
//! only one whose spike train can be decoded with
//! a fixed window ([`Recording::reconstruct`] refuses for the other).
//!
//! # The adjacent formalism
//!
//! [`Hypervector`] is hyperdimensional / vector-symbolic computing (Kanerva, *Hyperdimensional
//! Computing: An Introduction to Computing in Distributed Representation with High-Dimensional
//! Random Vectors*, Cognitive Computation 1:139–159, 2009). It is included here because it answers
//! the same question with the opposite trade: where neural sampling spends **time** to represent a
//! distribution, hyperdimensional computing spends **dimension** to represent structure, and both
//! degrade gracefully rather than failing. Binding is exactly invertible; bundling is lossy by a
//! known amount; and the loss has a closed form, [`bundle_similarity`], which this module checks
//! against measurement rather than quoting.
//!
//! # Units
//!
//! Biases, couplings, membrane potentials and log-odds in this module are in **nats** — they are
//! logarithms of probability ratios, which is dimensionless, and the natural-log convention is
//! what makes `exp(u)` the odds directly. They are *not* volts: the mapping from a model's `u` to a
//! real membrane potential is a per-neuron affine calibration this module does not attempt, and
//! writing volts here would be a unit claim nothing supports.
//!
//! Where a quantity is physical it is SI: [`population_log_odds`] takes firing rates in **hertz**
//! and a window in **seconds**, and multiplies them at the boundary to get the dimensionless
//! Poisson mean the likelihood needs. `tau` is in **ticks**, because it is a count of simulator
//! steps and rounding it from seconds is the caller's decision to make and record.
//!
//! # What is verified here, and what is not
//!
//! The strongest check in this crate lives in this module: the sampler's empirical distribution is
//! compared to the **exactly enumerated** Boltzmann distribution by total-variation distance, on a
//! system small enough to write down. There is no tolerance-shopping in that comparison — the
//! companion test drops the `−ln tau` correction from the firing probability and measures the
//! distance move from 0.0038 to 0.359, a factor of 93, so the threshold is known to discriminate.
//!
//! What is **not** re-derived here is the paper's general proof. Buesing et al. state the
//! discrete-time theorem (their Theorem 1) for a fixed-order serial sweep, which is not the
//! [`Scan::Random`] update this module samples with; the paper's remark that any mixture of its
//! per-unit operators keeps the target invariant is what covers the random scan. The `−ln tau`
//! correction is derived from scratch in [`NeuralSampler`]'s documentation for the single-unit
//! case, where it is elementary, and confirmed numerically for the coupled case. That is an
//! empirical confirmation of a published result, not a proof, and it is described that way.
//!
//! # Quickstart
//!
//! Three units, one coupling, and the network's own spikes answering a question about the
//! posterior — with an error bar that knows the samples came from a chain.
//!
//! ```
//! use ferromorphic::bayes::{Boltzmann, NeuralSampler, Scan};
//!
//! // p(z) favours units 0 and 1 being ON TOGETHER (a positive coupling) and unit 2 being off.
//! let bias = [0.0, 0.0, -1.0];
//! let mut w = vec![0.0; 9];
//! w[0 * 3 + 1] = 2.0;
//! w[1 * 3 + 0] = 2.0;
//! let model = Boltzmann::new(&bias, &w)?;
//!
//! // What the answer IS, by enumeration — available here only because the system is tiny.
//! let exact = model.exact_marginals()?;
//!
//! // What the network SAYS, by firing: a 3-tick refractory window, 200 000 ticks.
//! let mut net = NeuralSampler::new(model, 3, Scan::Random)?;
//! let mut rng = ferromorphic::Rng::new(1);
//! for _ in 0..20_000 {
//!     net.step(&mut rng); // burn in
//! }
//! let run = net.record(&mut rng, 200_000)?;
//!
//! for k in 0..3 {
//!     let e = run.marginal(k)?;
//!     let (lo, hi) = e.ci95();
//!     assert!(lo <= exact[k] && exact[k] <= hi, "unit {k}");
//!     // The samples are correlated, so the honest bar is several times the naive one.
//!     assert!(e.ess < e.n as f64);
//!     assert!(e.understatement().unwrap() > 1.5);
//! }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # The sibling
//!
//! This is the bridge to `ferrotherm`, the Institute's thermodynamic-computing crate, which samples
//! from Boltzmann distributions in physical hardware for a living. The same [`Boltzmann`] energy,
//! the same total-variation check, a different substrate. A spiking network that samples and a
//! thermodynamic device that samples are doing one thing, and the interesting question is which one
//! does it for fewer joules — which is [`crate::ledger`]'s question, not this module's.

use crate::rng::Rng;
use crate::spike::{Spike, Train};

/// The widest state this module will represent.
///
/// A configuration is packed into a `u64` bitmask, one bit per unit, so 64 is the representational
/// ceiling. It is a limit of this module's state encoding, not of neural sampling: a network of
/// millions of sampling neurons is the point of the theory, and [`crate::sim`] is where one runs.
/// What lives here is the part small enough to check against an exact answer.
pub const MAX_UNITS: usize = 64;

/// The widest state this module will **enumerate**.
///
/// Enumeration allocates `2^units` `f64`s: 8 MiB at 20 units, 8 GiB at 30. The cap is set where
/// the allocation is still obviously affordable, because every use of enumeration here is a
/// verification path and a verification path that runs out of memory teaches nothing.
pub const MAX_ENUMERABLE_UNITS: usize = 20;

/// The dimension Kanerva's 2009 exposition uses throughout, and the one its stated distance
/// figures (mean 5000 bits, standard deviation 50 bits between two random vectors) refer to.
pub const KANERVA_DIM: usize = 10_000;

/// Everything this module refuses, with the number that made it refuse.
///
/// Each variant names the quantity and its value. The alternative — returning a plausible number
/// from a malformed model — is how a sampler ends up reporting a posterior for a distribution
/// nobody specified.
#[derive(Debug, Clone, PartialEq)]
pub enum BayesError {
    /// A supplied number was not finite.
    NonFinite {
        /// Which array or scalar, for example `"bias"` or `"window_s"`.
        what: &'static str,
        /// Position within that array, or `0` when `what` names a scalar.
        index: usize,
    },
    /// A slice had the wrong length for the shape it was used at.
    ShapeMismatch {
        /// Which array, for example `"coupling"`.
        what: &'static str,
        /// The length supplied.
        got: usize,
        /// The length required.
        want: usize,
    },
    /// A scalar fell outside the range the mechanism is defined on.
    OutOfRange {
        /// Which quantity.
        what: &'static str,
        /// The value supplied.
        value: f64,
        /// Lowest acceptable value, inclusive.
        low: f64,
        /// Highest acceptable value, inclusive.
        high: f64,
    },
    /// An array that must be non-empty was empty.
    Empty {
        /// Which array.
        what: &'static str,
    },
    /// A count exceeded the limit this module can represent or enumerate.
    TooLarge {
        /// Which count, for example `"units"` or `"dimension"`.
        what: &'static str,
        /// The value supplied.
        got: usize,
        /// The largest accepted value.
        limit: usize,
    },
    /// The coupling matrix disagreed with its own transpose.
    ///
    /// A Boltzmann distribution is defined by a **symmetric** interaction: `W_kj` and `W_jk` are
    /// two names for the energy of one pair, and a matrix whose halves disagree does not define an
    /// energy at all. The sampler would still run and would still produce a stationary
    /// distribution — just not one anybody wrote down.
    Asymmetric {
        /// Row index.
        i: usize,
        /// Column index.
        j: usize,
        /// The entry at `(i, j)`.
        w_ij: f64,
        /// The entry at `(j, i)`, which differs.
        w_ji: f64,
    },
    /// A unit was coupled to itself.
    ///
    /// `W_kk` has no meaning in a binary pairwise model: `z_k² = z_k`, so a self-coupling is a bias
    /// wearing a coupling's name, and silently folding it into `b_k` would make two different
    /// models print the same parameters.
    SelfCoupling {
        /// The unit.
        unit: usize,
        /// The diagonal entry, which was not zero.
        value: f64,
    },
    /// A series was too short for the estimator asked of it.
    TooShort {
        /// How many samples were supplied.
        got: usize,
        /// The fewest the estimator is defined for.
        want: usize,
    },
    /// Every sample in a series was identical, so it has no sample variance and no error bar.
    ///
    /// Returned rather than reporting `sem = 0`. A unit that never fired across a whole run has an
    /// empirical marginal of exactly zero and an empirical error bar of exactly zero, and the
    /// second number is a statement about the sample, not about the posterior.
    NoVariation {
        /// Length of the constant series.
        n: usize,
        /// The value every sample took.
        value: f64,
    },
    /// Two hypervectors of different dimension met in an operation that needs them aligned.
    DimensionMismatch {
        /// Dimension of the left operand, in bits.
        a: usize,
        /// Dimension of the right operand, in bits.
        b: usize,
    },
    /// A supplied distribution did not sum to one.
    NotNormalised {
        /// The mass it did sum to.
        mass: f64,
    },
}

impl core::fmt::Display for BayesError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NonFinite { what, index } => write!(f, "{what}[{index}] is not a finite number"),
            Self::ShapeMismatch { what, got, want } => {
                write!(f, "{what} has {got} entries where {want} are required")
            }
            Self::OutOfRange { what, value, low, high } => {
                write!(f, "{what} = {value} is outside [{low}, {high}]")
            }
            Self::Empty { what } => write!(f, "{what} is empty"),
            Self::TooLarge { what, got, limit } => {
                write!(f, "{what} = {got} exceeds the limit of {limit}")
            }
            Self::Asymmetric { i, j, w_ij, w_ji } => write!(
                f,
                "coupling is not symmetric: W[{i}][{j}] = {w_ij} but W[{j}][{i}] = {w_ji}; a \
                 Boltzmann energy needs one value per pair"
            ),
            Self::SelfCoupling { unit, value } => write!(
                f,
                "W[{unit}][{unit}] = {value}; a binary unit has no self-coupling, put it in the bias"
            ),
            Self::TooShort { got, want } => {
                write!(f, "{got} samples where the estimator needs at least {want}")
            }
            Self::NoVariation { n, value } => write!(
                f,
                "all {n} samples equal {value}; there is no sample variance and therefore no \
                 error bar to report"
            ),
            Self::DimensionMismatch { a, b } => {
                write!(f, "hypervectors of {a} and {b} bits cannot be combined")
            }
            Self::NotNormalised { mass } => {
                write!(f, "the supplied distribution sums to {mass}, not 1")
            }
        }
    }
}

impl std::error::Error for BayesError {}

/// The logistic function `1 / (1 + exp(-x))`, computed without overflowing either tail.
///
/// This is the map from a log-odds in nats to a probability in `[0, 1]`, and it is public because
/// every readout in this module ends in it. The branch is not an optimisation: `exp(-x)` overflows
/// to infinity below `x ≈ -745`, and the naive form then returns `0.0` where the correct answer is
/// a denormal — small, but the difference between "impossible" and "very unlikely" survives into a
/// log-likelihood sum where it matters.
#[must_use]
pub fn logistic(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

fn finite(xs: &[f64], what: &'static str) -> Result<(), BayesError> {
    for (i, x) in xs.iter().enumerate() {
        if !x.is_finite() {
            return Err(BayesError::NonFinite { what, index: i });
        }
    }
    Ok(())
}

fn finite_scalar(x: f64, what: &'static str) -> Result<(), BayesError> {
    if x.is_finite() { Ok(()) } else { Err(BayesError::NonFinite { what, index: 0 }) }
}

// ---------------------------------------------------------------------------------------------
// The target distribution
// ---------------------------------------------------------------------------------------------

/// A Boltzmann distribution over binary units: the target the network is asked to sample.
///
/// `p(z) ∝ exp(E(z))` with
///
/// ```text
/// E(z) = Σ_k b_k z_k + ½ Σ_{k≠j} W_kj z_k z_j        (nats)
/// ```
///
/// The sign convention is the machine-learning one — `E` is a **log-probability**, so a larger `E`
/// is a more likely state — rather than the physicist's energy, which carries the opposite sign.
/// The factor of `½` is there because the double sum visits each pair twice; with `W` symmetric and
/// zero on the diagonal, `E` is what a Boltzmann machine's textbook definition gives.
///
/// The invariants, all enforced in [`Boltzmann::new`]: `W` is symmetric, `W_kk = 0`, everything is
/// finite, and `units` is at most [`MAX_UNITS`].
#[derive(Debug, Clone, PartialEq)]
pub struct Boltzmann {
    units: usize,
    bias: Vec<f64>,
    coupling: Vec<f64>,
}

impl Boltzmann {
    /// Build from a bias vector (nats) and a row-major `units × units` coupling matrix (nats).
    ///
    /// # Errors
    ///
    /// [`BayesError::Empty`] for no units, [`BayesError::TooLarge`] past [`MAX_UNITS`],
    /// [`BayesError::ShapeMismatch`] for a coupling matrix of the wrong length,
    /// [`BayesError::NonFinite`] for any non-finite entry, [`BayesError::SelfCoupling`] for a
    /// nonzero diagonal, and [`BayesError::Asymmetric`] naming the first pair whose two entries
    /// disagree.
    pub fn new(bias: &[f64], coupling: &[f64]) -> Result<Self, BayesError> {
        let units = bias.len();
        if units == 0 {
            return Err(BayesError::Empty { what: "bias" });
        }
        if units > MAX_UNITS {
            return Err(BayesError::TooLarge { what: "units", got: units, limit: MAX_UNITS });
        }
        if coupling.len() != units * units {
            return Err(BayesError::ShapeMismatch {
                what: "coupling",
                got: coupling.len(),
                want: units * units,
            });
        }
        finite(bias, "bias")?;
        finite(coupling, "coupling")?;
        for k in 0..units {
            let d = coupling[k * units + k];
            if d != 0.0 {
                return Err(BayesError::SelfCoupling { unit: k, value: d });
            }
            for j in (k + 1)..units {
                let (a, b) = (coupling[k * units + j], coupling[j * units + k]);
                if a != b {
                    return Err(BayesError::Asymmetric { i: k, j, w_ij: a, w_ji: b });
                }
            }
        }
        Ok(Self { units, bias: bias.to_vec(), coupling: coupling.to_vec() })
    }

    /// An uncoupled model: independent units with the given biases.
    ///
    /// Its exact distribution is the product `∏_k logistic(b_k)^{z_k} (1 - logistic(b_k))^{1-z_k}`,
    /// which is why it is the reference case for
    /// `zero_coupling_samples_the_exact_product_of_independent_bits`.
    ///
    /// # Errors
    ///
    /// As [`Boltzmann::new`], for the bias alone.
    pub fn independent(bias: &[f64]) -> Result<Self, BayesError> {
        Self::new(bias, &vec![0.0; bias.len() * bias.len()])
    }

    /// How many binary units the distribution is over.
    #[must_use]
    pub fn units(&self) -> usize {
        self.units
    }

    /// The bias vector, nats, one entry per unit.
    #[must_use]
    pub fn bias(&self) -> &[f64] {
        &self.bias
    }

    /// The coupling matrix, nats, row-major `units × units`, symmetric with a zero diagonal.
    #[must_use]
    pub fn coupling(&self) -> &[f64] {
        &self.coupling
    }

    /// How many configurations there are: `2^units`, or `None` past [`MAX_ENUMERABLE_UNITS`].
    #[must_use]
    pub fn configurations(&self) -> Option<usize> {
        (self.units <= MAX_ENUMERABLE_UNITS).then(|| 1usize << self.units)
    }

    fn energy_of(&self, z: u64) -> f64 {
        let mut e = 0.0;
        for k in 0..self.units {
            if z >> k & 1 == 1 {
                e += self.bias[k];
                for j in 0..self.units {
                    if z >> j & 1 == 1 {
                        e += 0.5 * self.coupling[k * self.units + j];
                    }
                }
            }
        }
        e
    }

    /// The log-unnormalised probability `E(z)` of a configuration, in nats.
    ///
    /// `z` is a bitmask, bit `k` for unit `k`. Returns `None` if any bit at or above `units` is
    /// set, rather than masking them away: a stray high bit means the caller's state encoding and
    /// this model's disagree, and silently ignoring it would make the two disagree forever.
    ///
    /// The sum is reported as computed, so a model whose finite biases and couplings add past
    /// `f64::MAX` gives `Some(±inf)` here. [`Boltzmann::exact`] refuses such a model rather than
    /// turning the infinity into a `NaN` distribution.
    #[must_use]
    pub fn energy(&self, z: u64) -> Option<f64> {
        (self.units == MAX_UNITS || z >> self.units == 0).then(|| self.energy_of(z))
    }

    fn membrane_of(&self, z: u64, k: usize) -> f64 {
        let mut u = self.bias[k];
        for j in 0..self.units {
            if z >> j & 1 == 1 {
                u += self.coupling[k * self.units + j];
            }
        }
        u
    }

    /// The membrane potential of unit `k` in configuration `z`: `b_k + Σ_j W_kj z_j`, in nats.
    ///
    /// **This is the neural computability condition's left-hand side**, and the claim the theory
    /// makes is that it equals the conditional log-odds of `z_k`. Bit `k` of `z` is read but cannot
    /// matter, because `W_kk = 0` is an enforced invariant — which is exactly why a self-coupling
    /// is refused rather than folded into the bias.
    ///
    /// Returns `None` for a `z` with bits set above `units`, as [`Boltzmann::energy`].
    #[must_use]
    pub fn membrane(&self, z: u64, k: usize) -> Option<f64> {
        if k >= self.units || (self.units != MAX_UNITS && z >> self.units != 0) {
            return None;
        }
        Some(self.membrane_of(z, k))
    }

    /// The exact normalised distribution over all `2^units` configurations, indexed by bitmask.
    ///
    /// Computed by subtracting the largest energy before exponentiating, so a model with couplings
    /// of a few hundred nats normalises rather than returning a vector of infinities and `NaN`.
    /// That trick handles large **finite** energies and nothing else: [`Boltzmann::new`] checks
    /// that its inputs are finite, which does not stop `E(z)` itself from overflowing when the
    /// inputs are near `f64::MAX`, and `inf − inf` is `NaN`. An energy that is not finite is
    /// therefore refused here rather than normalised into a vector a caller would compare with
    /// `<` and silently get `false` from.
    ///
    /// # Errors
    ///
    /// [`BayesError::TooLarge`] past [`MAX_ENUMERABLE_UNITS`], and [`BayesError::NonFinite`]
    /// naming the first configuration whose energy overflowed.
    pub fn exact(&self) -> Result<Vec<f64>, BayesError> {
        let n = self.configurations().ok_or(BayesError::TooLarge {
            what: "units to enumerate",
            got: self.units,
            limit: MAX_ENUMERABLE_UNITS,
        })?;
        let mut e = Vec::with_capacity(n);
        let mut top = f64::NEG_INFINITY;
        for z in 0..n {
            let v = self.energy_of(z as u64);
            if !v.is_finite() {
                return Err(BayesError::NonFinite { what: "energy", index: z });
            }
            top = top.max(v);
            e.push(v);
        }
        let mut sum = 0.0;
        for v in &mut e {
            *v = (*v - top).exp();
            sum += *v;
        }
        for v in &mut e {
            *v /= sum;
        }
        Ok(e)
    }

    /// The exact marginal `p(z_k = 1)` for each unit, from full enumeration.
    ///
    /// # Errors
    ///
    /// As [`Boltzmann::exact`].
    pub fn exact_marginals(&self) -> Result<Vec<f64>, BayesError> {
        let p = self.exact()?;
        let mut m = vec![0.0; self.units];
        for (z, &pz) in p.iter().enumerate() {
            for (k, mk) in m.iter_mut().enumerate() {
                if z >> k & 1 == 1 {
                    *mk += pz;
                }
            }
        }
        Ok(m)
    }
}

/// An arbitrary distribution over binary configurations, held as log-weights.
///
/// This exists so the neural computability condition can be **falsified**, not merely restated. A
/// [`Boltzmann`] satisfies the NCC by algebra, so checking it against one proves only that the
/// algebra was typed correctly. A `Target` can carry interactions of any order, and
/// [`PairwiseFit::residual`] then measures how far its conditional log-odds is from anything a
/// linear synaptic integrator could produce.
///
/// The invariant, and it holds for **both** constructors: every log-weight is finite, so every
/// conditional log-odds this type reports is a difference of two finite numbers.
#[derive(Debug, Clone, PartialEq)]
pub struct Target {
    units: usize,
    log_w: Vec<f64>,
}

impl Target {
    /// Build from `2^units` log-weights, indexed by configuration bitmask.
    ///
    /// The weights need not be normalised: every quantity this type computes is a difference of two
    /// log-weights, so an additive constant cancels.
    ///
    /// # Errors
    ///
    /// [`BayesError::Empty`], [`BayesError::TooLarge`] past [`MAX_ENUMERABLE_UNITS`],
    /// [`BayesError::ShapeMismatch`] for a wrong length, [`BayesError::NonFinite`] for any entry.
    pub fn from_log_weights(units: usize, log_w: &[f64]) -> Result<Self, BayesError> {
        if units == 0 {
            return Err(BayesError::Empty { what: "units" });
        }
        if units > MAX_ENUMERABLE_UNITS {
            return Err(BayesError::TooLarge {
                what: "units",
                got: units,
                limit: MAX_ENUMERABLE_UNITS,
            });
        }
        let want = 1usize << units;
        if log_w.len() != want {
            return Err(BayesError::ShapeMismatch { what: "log_w", got: log_w.len(), want });
        }
        finite(log_w, "log_w")?;
        Ok(Self { units, log_w: log_w.to_vec() })
    }

    /// The target a [`Boltzmann`] defines, enumerated.
    ///
    /// Runs the same finiteness gate as [`Target::from_log_weights`], so the two constructors
    /// leave the type with one invariant and not two: a model whose energies overflow to `±inf` is
    /// refused here exactly as its log-weights would be if they were passed in by hand.
    ///
    /// # Errors
    ///
    /// [`BayesError::TooLarge`] past [`MAX_ENUMERABLE_UNITS`], and [`BayesError::NonFinite`]
    /// naming the first configuration whose energy overflowed.
    pub fn of_boltzmann(b: &Boltzmann) -> Result<Self, BayesError> {
        let n = b.configurations().ok_or(BayesError::TooLarge {
            what: "units to enumerate",
            got: b.units(),
            limit: MAX_ENUMERABLE_UNITS,
        })?;
        let log_w: Vec<f64> = (0..n).map(|z| b.energy_of(z as u64)).collect();
        finite(&log_w, "log_w")?;
        Ok(Self { units: b.units(), log_w })
    }

    /// How many binary units the target is over.
    #[must_use]
    pub fn units(&self) -> usize {
        self.units
    }

    /// The conditional log-odds of unit `k` given the other units' values in `z`, in nats.
    ///
    /// `log p(z_k = 1 | z_\k) − log p(z_k = 0 | z_\k)`, which is a difference of two log-weights
    /// and therefore needs no normalising constant. Bit `k` of `z` is ignored by construction —
    /// both weights are read with it forced.
    ///
    /// # Errors
    ///
    /// [`BayesError::OutOfRange`] if `k` is not a unit or `z` has bits set above `units`.
    pub fn conditional_log_odds(&self, z: u64, k: usize) -> Result<f64, BayesError> {
        if k >= self.units {
            return Err(BayesError::OutOfRange {
                what: "unit index",
                value: k as f64,
                low: 0.0,
                high: (self.units - 1) as f64,
            });
        }
        if z >> self.units != 0 {
            return Err(BayesError::OutOfRange {
                what: "configuration",
                value: z as f64,
                low: 0.0,
                high: ((1u64 << self.units) - 1) as f64,
            });
        }
        let off = z & !(1u64 << k);
        let on = off | (1u64 << k);
        Ok(self.log_w[on as usize] - self.log_w[off as usize])
    }
}

/// What a linear synaptic integrator can and cannot represent of a target distribution.
///
/// Built by reading the target's own conditional log-odds at chosen configurations: the bias is the
/// log-odds against an all-zero background, and each coupling is the change produced by switching
/// one other unit on. Those `units + units²` readings **determine** the only pairwise model that
/// could possibly match, and [`PairwiseFit::residual`] is then the largest disagreement between
/// that model and the target over every configuration.
///
/// The residual is zero exactly when the target is a pairwise Boltzmann distribution, and equals
/// the size of the missing term when it is not — see
/// `the_computability_condition_fails_for_a_third_order_target`, where a deliberately planted
/// three-way interaction of 0.8 nats is recovered as a residual of 0.8 to floating-point noise.
///
/// **There is no symmetry defect to report.** Expanding the two readings,
///
/// ```text
/// W_kj = [w(e_k+e_j) − w(e_j)] − [w(e_k) − w(0)]
/// W_jk = [w(e_j+e_k) − w(e_k)] − [w(e_j) − w(0)]
/// ```
///
/// are the same four log-weights with the same signs, so `W_kj − W_jk` is identically zero for
/// **any** target, third-order terms included. A field reporting it would be a check that no input
/// could move — the symmetry is a property of where the fit reads the conditionals, and
/// `the_fitted_coupling_is_symmetric_because_of_where_the_fit_reads_it` is what defends it.
#[derive(Debug, Clone, PartialEq)]
pub struct PairwiseFit {
    /// Bias implied by the target's conditionals, nats, one per unit.
    pub bias: Vec<f64>,
    /// Coupling implied by the target's conditionals, nats, row-major `units × units`.
    pub coupling: Vec<f64>,
    /// Largest `|target log-odds − fitted log-odds|` over every unit and every configuration, nats.
    ///
    /// **Zero iff the neural computability condition is satisfiable by a linear integrator.** A
    /// residual of `r` nats means some conditional the network must compute is off by `r`, and a
    /// tenth of a nat is already a 10% error in an odds ratio.
    pub residual: f64,
}

impl PairwiseFit {
    /// Fit a pairwise model to a target's conditional log-odds and measure what is left over.
    ///
    /// Cost is `units · 2^units` evaluations, which is why the target is capped at
    /// [`MAX_ENUMERABLE_UNITS`].
    ///
    /// # Errors
    ///
    /// Propagates [`Target::conditional_log_odds`], which cannot fail for indices this method
    /// generates; the signature keeps the error path visible rather than unwrapping inside.
    pub fn of(t: &Target) -> Result<Self, BayesError> {
        let n = t.units;
        let mut bias = vec![0.0; n];
        let mut coupling = vec![0.0; n * n];
        for k in 0..n {
            bias[k] = t.conditional_log_odds(0, k)?;
            for j in 0..n {
                if j != k {
                    coupling[k * n + j] = t.conditional_log_odds(1u64 << j, k)? - bias[k];
                }
            }
        }
        let mut residual = 0.0f64;
        for z in 0..(1u64 << n) {
            for k in 0..n {
                let mut fitted = bias[k];
                for j in 0..n {
                    if j != k && z >> j & 1 == 1 {
                        fitted += coupling[k * n + j];
                    }
                }
                residual = residual.max((t.conditional_log_odds(z, k)? - fitted).abs());
            }
        }
        Ok(Self { bias, coupling, residual })
    }
}

// ---------------------------------------------------------------------------------------------
// The sampler
// ---------------------------------------------------------------------------------------------

/// Which units update on a tick, and therefore whether the chain samples the target exactly.
///
/// This is not a performance knob. It decides correctness, and the two options trade correctness
/// against biological honesty in opposite directions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scan {
    /// One unit, drawn uniformly, updates per tick.
    ///
    /// **This is the mode that samples the target.** It is a random-scan form of the serial update.
    /// Buesing et al. prove their discrete-time theorem (Theorem 1) for a fixed-order sequential
    /// sweep `T = T_K∘…∘T_1`, "the neurons are updated sequentially in the same order", and note
    /// that because each `T_k` leaves `p` invariant, "any composition or mixture of these operators
    /// also leaves `p` invariant". One uniformly drawn unit per tick is such a mixture, so its
    /// invariance follows from that remark; its convergence to the target is confirmed here
    /// numerically rather than taken from the paper's proof.
    /// `the_sampled_distribution_matches_the_exact_boltzmann_distribution` measures a
    /// total-variation distance of 0.0014 to 0.0040 against the enumerated answer, over refractory
    /// windows of 1, 2, 5 and 20 ticks and 1.8 million post-burn-in samples each.
    ///
    /// This paragraph used to call this mode "the serial update Buesing et al. state their
    /// discrete-time theorem for". The theorem's operator is the fixed-order sweep above, which
    /// this module does not implement; the paper's "different order" at each time step re-orders a
    /// full sweep, which is not one unit drawn per tick either. Nothing in the sampler changed.
    ///
    /// Its cost is that a unit's refractory countdown advances only on the ticks where that unit is
    /// selected, so `z_k` stays high for a random number of ticks — negative binomial, mean
    /// `tau · units` — and a fixed-window reconstruction of the state from the spike train alone is
    /// therefore wrong. [`Recording::reconstruct`] refuses rather than returning it.
    Random,
    /// Every unit updates every tick, from a snapshot of the state at the start of the tick.
    ///
    /// **This does not sample a coupled target**, and the error is not small: the same model, seed
    /// and budget that give 0.0031 under [`Scan::Random`] give 0.092 under this one at `tau = 3`
    /// — a tenth of the maximum a total-variation distance can be. With zero coupling it *is*
    /// exact, because there is then nothing for the simultaneous
    /// updates to disagree about — the pair of measurements is
    /// `parallel_updates_are_exact_without_coupling_and_wrong_with_it`.
    ///
    /// It is here because it is what a network of real neurons does — every cell integrates on
    /// every millisecond — and because it is the only mode whose spike train decodes with a fixed
    /// window: a spike sets `z_k = 1` for exactly `tau` ticks, so
    /// [`Recording::reconstruct`] returns the state exactly.
    Parallel,
}

/// A network of sampling neurons: a Markov chain whose spike train is a sequence of samples.
///
/// # State
///
/// Each unit carries a refractory counter `ζ_k ∈ {0, …, tau}`, and the sampled variable is
/// `z_k = [ζ_k ≥ 1]` — *fired within the refractory window*. A spike sets `ζ_k = tau`; each
/// subsequent update of that unit counts it down; at `ζ_k ≤ 1` the unit decides again.
///
/// # The firing probability, derived
///
/// A unit whose membrane potential is `u` must spend a fraction `logistic(u)` of its updates with
/// `z_k = 1`. One spike buys `tau` updates of `z_k = 1`; after that the unit re-decides each
/// update, so the number of `z_k = 0` updates before the next spike is geometric with success
/// probability `q`, mean `(1 − q)/q`. Setting the duty cycle equal to the target,
///
/// ```text
/// tau·q / (tau·q + (1 − q))  =  e^u / (1 + e^u)
/// ```
///
/// and solving gives `q = e^u / (tau + e^u) = logistic(u − ln tau)`. **The `−ln tau` is the whole
/// correction**, and it is not cosmetic: dropping it moves the sampled distribution from a
/// total-variation distance of 0.0038 off the target to 0.359 off it — a factor of 93 — measured
/// in `dropping_the_log_tau_correction_moves_the_distribution_a_hundredfold`.
///
/// The derivation above is elementary for one unit. For coupled units it is Theorem 1 of Buesing et
/// al. (2011), proved in their Methods by Lemmata 1–3 (each `T_k` leaves `p` invariant; the
/// fixed-order composition `T = T_K∘…∘T_1` is irreducible and aperiodic). The theorem is stated
/// for that fixed-order sweep, not for [`Scan::Random`]; see there for why the random scan still
/// samples `p`. This implementation confirms it numerically against the enumerated distribution
/// rather than reproving it, and says so.
///
/// This doc used to cite "Theorem 3". The paper's Methods read "The following Lemmata 1 – 3
/// provide a proof of Theorem 1", and this review did not locate a Theorem 2 or a Theorem 3
/// anywhere in its full text (Europe PMC, PMCID PMC3207943, PMID 22096452).
///
/// # Provenance of the shape
///
/// Buesing, Bill, Nessler & Maass, `PLoS` Computational Biology 7(11):e1002211, 2011, "Neural
/// sampling in discrete time" and the absolute-refractory mechanism. The continuous-time version in
/// the same paper uses an instantaneous rate `(1/tau)·exp(u)`, which this module does not
/// implement: a rate that grows without bound is unusable in a fixed-step simulator, and the
/// discrete-time form is the one with an exactly checkable stationary distribution.
#[derive(Debug, Clone, PartialEq)]
pub struct NeuralSampler {
    model: Boltzmann,
    tau: u32,
    scan: Scan,
    zeta: Vec<u32>,
    log_tau: f64,
}

impl NeuralSampler {
    /// The largest refractory window this module accepts, in ticks.
    ///
    /// A window of a million ticks is not a modelling choice, it is a mistake in unit conversion —
    /// and it would make the chain mix so slowly that no run could reach its stationary
    /// distribution. The bound is a guard against that, not a claim about biology.
    pub const MAX_TAU: u32 = 1 << 20;

    /// Build a sampler for `model` with a refractory window of `tau` **ticks**.
    ///
    /// Starts from the all-quiet state (`ζ = 0`, `z = 0`), which is a valid but generally
    /// improbable configuration; a run therefore needs a burn-in, which [`NeuralSampler::run`]
    /// takes as an explicit argument rather than choosing for the caller.
    ///
    /// # Errors
    ///
    /// [`BayesError::OutOfRange`] for `tau == 0` or `tau > MAX_TAU`. A window of zero ticks would
    /// leave `z_k` permanently zero and sample nothing.
    pub fn new(model: Boltzmann, tau: u32, scan: Scan) -> Result<Self, BayesError> {
        if tau == 0 || tau > Self::MAX_TAU {
            return Err(BayesError::OutOfRange {
                what: "tau (ticks)",
                value: f64::from(tau),
                low: 1.0,
                high: f64::from(Self::MAX_TAU),
            });
        }
        let units = model.units();
        Ok(Self { model, tau, scan, zeta: vec![0; units], log_tau: f64::from(tau).ln() })
    }

    /// The target distribution this chain is sampling.
    #[must_use]
    pub fn model(&self) -> &Boltzmann {
        &self.model
    }

    /// The refractory window, in ticks: how long one spike holds `z_k = 1`.
    #[must_use]
    pub fn tau(&self) -> u32 {
        self.tau
    }

    /// Which update order this chain uses.
    #[must_use]
    pub fn scan(&self) -> Scan {
        self.scan
    }

    /// The current configuration as a bitmask, bit `k` set iff `z_k = 1`.
    #[must_use]
    pub fn state(&self) -> u64 {
        let mut z = 0u64;
        for (k, &c) in self.zeta.iter().enumerate() {
            if c >= 1 {
                z |= 1u64 << k;
            }
        }
        z
    }

    /// Return every unit to quiescence, discarding the chain's history.
    ///
    /// A [`Recording`] started from this state can be reconstructed from its spike train alone;
    /// one started mid-refractory cannot, because the spike that set the counter is not in the
    /// recording. [`NeuralSampler::record`] checks for it.
    pub fn reset(&mut self) {
        self.zeta.fill(0);
    }

    fn update(&mut self, k: usize, z_snapshot: u64, rng: &mut Rng) -> bool {
        if self.zeta[k] >= 2 {
            self.zeta[k] -= 1;
            return false;
        }
        let u = self.model.membrane_of(z_snapshot, k);
        let q = logistic(u - self.log_tau);
        if rng.next_f64() < q {
            self.zeta[k] = self.tau;
            true
        } else {
            self.zeta[k] = 0;
            false
        }
    }

    /// Advance one tick. Returns a bitmask of the units that spiked on this tick.
    ///
    /// Under [`Scan::Random`] at most one bit can be set; under [`Scan::Parallel`] any subset can.
    /// The returned mask is the spike train; the state afterwards is [`NeuralSampler::state`], and
    /// the two are different objects — a unit can be `z_k = 1` on a tick it did not spike on,
    /// which is the entire reason the refractory window exists.
    pub fn step(&mut self, rng: &mut Rng) -> u64 {
        match self.scan {
            Scan::Random => {
                let k = rng.below(self.model.units() as u32) as usize;
                let z = self.state();
                u64::from(self.update(k, z, rng)) << k
            }
            Scan::Parallel => {
                let z = self.state();
                let mut fired = 0u64;
                for k in 0..self.model.units() {
                    if self.update(k, z, rng) {
                        fired |= 1u64 << k;
                    }
                }
                fired
            }
        }
    }

    /// Run for `steps` ticks, discarding the first `burn_in`, and histogram the configurations.
    ///
    /// The histogram is over the state **after** each tick, one observation per tick. That makes
    /// successive observations strongly dependent — under [`Scan::Random`] only one unit can even
    /// have changed — which is why [`Estimate`] and not a raw count is what any number read off
    /// this run should carry.
    ///
    /// # Errors
    ///
    /// [`BayesError::TooLarge`] if the model has more units than [`MAX_ENUMERABLE_UNITS`], since
    /// the histogram has one bin per configuration; [`BayesError::TooShort`] if `burn_in` is not
    /// smaller than `steps`, because a run with nothing left after burn-in has no samples and
    /// returning an empty histogram would let a caller divide by zero downstream.
    pub fn run(
        &mut self,
        rng: &mut Rng,
        steps: u64,
        burn_in: u64,
    ) -> Result<Histogram, BayesError> {
        if burn_in >= steps {
            return Err(BayesError::TooShort {
                got: steps as usize,
                want: burn_in as usize + 1,
            });
        }
        let mut h = Histogram::new(self.model.units())?;
        for t in 0..steps {
            self.step(rng);
            if t >= burn_in {
                h.observe(self.state());
            }
        }
        Ok(h)
    }

    /// Run for `steps` ticks, keeping the full state trace and the spike train.
    ///
    /// Memory is 8 bytes per tick for the trace plus 16 bytes per spike, so a million ticks of a
    /// busy network is tens of megabytes. Use [`NeuralSampler::run`] when only the distribution is
    /// wanted.
    ///
    /// # Errors
    ///
    /// [`BayesError::TooShort`] for `steps == 0`.
    pub fn record(&mut self, rng: &mut Rng, steps: u64) -> Result<Recording, BayesError> {
        if steps == 0 {
            return Err(BayesError::TooShort { got: 0, want: 1 });
        }
        let from_reset = self.zeta.iter().all(|&c| c == 0);
        let mut states = Vec::with_capacity(steps as usize);
        let mut train = Train::new();
        for t in 0..steps {
            let mut fired = self.step(rng);
            while fired != 0 {
                let k = fired.trailing_zeros();
                train.push(Spike { t, source: k });
                fired &= fired - 1;
            }
            states.push(self.state());
        }
        Ok(Recording {
            units: self.model.units(),
            tau: self.tau,
            scan: self.scan,
            from_reset,
            states,
            train,
        })
    }
}

/// Counts of configurations, one bin per bitmask.
///
/// The bins are indexed by the configuration itself, so bin `0b0101` is the count of ticks on which
/// units 0 and 2 were high and the rest were low. That indexing is what makes
/// [`Histogram::total_variation`] a direct comparison against [`Boltzmann::exact`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Histogram {
    units: usize,
    counts: Vec<u64>,
    samples: u64,
}

impl Histogram {
    /// An empty histogram over `units` binary units.
    ///
    /// # Errors
    ///
    /// [`BayesError::Empty`] for zero units, [`BayesError::TooLarge`] past
    /// [`MAX_ENUMERABLE_UNITS`] — there is one bin per configuration.
    pub fn new(units: usize) -> Result<Self, BayesError> {
        if units == 0 {
            return Err(BayesError::Empty { what: "units" });
        }
        if units > MAX_ENUMERABLE_UNITS {
            return Err(BayesError::TooLarge {
                what: "units",
                got: units,
                limit: MAX_ENUMERABLE_UNITS,
            });
        }
        Ok(Self { units, counts: vec![0; 1usize << units], samples: 0 })
    }

    /// Record one configuration. Bits at or above `units` are ignored by the caller's contract;
    /// a mask with them set is dropped rather than aliased onto a real bin.
    ///
    /// Dropping rather than masking is deliberate: an aliased observation would inflate a bin that
    /// belongs to a different state, and the histogram would still sum to something plausible.
    /// [`Histogram::samples`] counts only what was recorded, so a dropped observation shows up as
    /// a sample count below the number of ticks.
    pub fn observe(&mut self, z: u64) {
        if self.units < MAX_UNITS && z >> self.units != 0 {
            return;
        }
        self.counts[z as usize] += 1;
        self.samples += 1;
    }

    /// How many units the histogram is over.
    #[must_use]
    pub fn units(&self) -> usize {
        self.units
    }

    /// How many observations were recorded.
    #[must_use]
    pub fn samples(&self) -> u64 {
        self.samples
    }

    /// Raw counts, indexed by configuration bitmask.
    #[must_use]
    pub fn counts(&self) -> &[u64] {
        &self.counts
    }

    /// The empirical distribution, indexed by configuration bitmask.
    ///
    /// Returns `None` with no samples: an empty histogram has no distribution, and a vector of
    /// zeros would compare favourably against nothing and badly against everything.
    #[must_use]
    pub fn probabilities(&self) -> Option<Vec<f64>> {
        (self.samples > 0)
            .then(|| self.counts.iter().map(|&c| c as f64 / self.samples as f64).collect())
    }

    /// The empirical marginal `p(z_k = 1)` for each unit.
    ///
    /// Returns `None` with no samples, as [`Histogram::probabilities`].
    #[must_use]
    pub fn marginals(&self) -> Option<Vec<f64>> {
        let p = self.probabilities()?;
        let mut m = vec![0.0; self.units];
        for (z, &pz) in p.iter().enumerate() {
            for (k, mk) in m.iter_mut().enumerate() {
                if z >> k & 1 == 1 {
                    *mk += pz;
                }
            }
        }
        Some(m)
    }

    /// Total-variation distance to a reference distribution: `½ Σ |p_i − q_i|`, in `[0, 1]`.
    ///
    /// **This is the strongest verification available in this crate.** It is not a summary
    /// statistic that a wrong sampler could match by accident — it bounds, over *every* event, the
    /// difference between the probability the chain assigns and the probability the model does. A
    /// value of 0.001 means no statement about this system can be wrong by more than a tenth of a
    /// percent; a value of 0.36 means a third of the probability mass is in the wrong place.
    ///
    /// # Errors
    ///
    /// [`BayesError::ShapeMismatch`] for a reference of the wrong length,
    /// [`BayesError::NonFinite`] for a non-finite entry, [`BayesError::NotNormalised`] if the
    /// reference misses unit mass by more than `1e-9`, and [`BayesError::Empty`] with no samples.
    pub fn total_variation(&self, reference: &[f64]) -> Result<f64, BayesError> {
        if reference.len() != self.counts.len() {
            return Err(BayesError::ShapeMismatch {
                what: "reference distribution",
                got: reference.len(),
                want: self.counts.len(),
            });
        }
        finite(reference, "reference distribution")?;
        let mass: f64 = reference.iter().sum();
        if (mass - 1.0).abs() > 1e-9 {
            return Err(BayesError::NotNormalised { mass });
        }
        let p = self.probabilities().ok_or(BayesError::Empty { what: "histogram" })?;
        Ok(0.5 * p.iter().zip(reference).map(|(a, b)| (a - b).abs()).sum::<f64>())
    }
}

/// A run kept in full: every configuration, every spike.
///
/// Held so that the two readouts can be compared against each other — the state trace the simulator
/// knows, and the state a downstream decoder could recover from the spikes alone.
#[derive(Debug, Clone, PartialEq)]
pub struct Recording {
    units: usize,
    tau: u32,
    scan: Scan,
    from_reset: bool,
    states: Vec<u64>,
    train: Train,
}

impl Recording {
    /// The configuration after each tick, as bitmasks.
    #[must_use]
    pub fn states(&self) -> &[u64] {
        &self.states
    }

    /// The spikes, with `t` the tick index within this recording.
    #[must_use]
    pub fn train(&self) -> &Train {
        &self.train
    }

    /// How many units the recording is over.
    #[must_use]
    pub fn units(&self) -> usize {
        self.units
    }

    /// The `z_k(t)` series for one unit as `0.0`/`1.0`, taken from the state trace.
    ///
    /// Returns `None` for a unit index out of range. This is the series [`Recording::marginal`]
    /// estimates, and feeding it to [`estimate`] directly is the supported way to put an error bar
    /// on any function of the state.
    #[must_use]
    pub fn indicator(&self, k: usize) -> Option<Vec<f64>> {
        (k < self.units)
            .then(|| self.states.iter().map(|&z| f64::from(u8::from(z >> k & 1 == 1))).collect())
    }

    /// The state of unit `k` recovered from the **spike train alone**, with a fixed `tau`-tick
    /// window: `z_k(t) = 1` iff unit `k` spiked at some tick in `(t − tau, t]`.
    ///
    /// Returns `None` unless the recording was made under [`Scan::Parallel`] **and** started from
    /// a reset sampler. Those are the two conditions under which the window is the state:
    ///
    /// - Under [`Scan::Parallel`] a unit's counter decrements on every tick, so one spike holds
    ///   `z_k = 1` for exactly `tau` ticks and the window is exact.
    /// - Under [`Scan::Random`] the counter decrements only when that unit is drawn, so the high
    ///   period is negative binomial with mean `tau · units` ticks. A `tau`-tick window would report
    ///   a state the network was never in — and it would look entirely reasonable in a raster.
    /// - Starting mid-refractory means the spike that set the counter is outside the recording, so
    ///   the first few ticks would be reconstructed as low when they were high.
    ///
    /// Refusing is the point. A decoder that returned the wrong window here would misread every
    /// marginal in the network by a factor of roughly `units`.
    #[must_use]
    pub fn reconstruct(&self, k: usize) -> Option<Vec<f64>> {
        if k >= self.units || self.scan != Scan::Parallel || !self.from_reset {
            return None;
        }
        let n = self.states.len();
        let mut out = vec![0.0; n];
        for s in self.train.spikes() {
            if s.source as usize != k {
                continue;
            }
            let start = s.t as usize;
            let end = (start + self.tau as usize).min(n);
            for v in &mut out[start..end] {
                *v = 1.0;
            }
        }
        Some(out)
    }

    /// The marginal `p(z_k = 1)` with a chain-aware error bar.
    ///
    /// # Errors
    ///
    /// [`BayesError::OutOfRange`] for a unit index past the end, and whatever [`estimate`] returns
    /// — in particular [`BayesError::NoVariation`] for a unit that never fired or never stopped,
    /// which has no error bar this method is willing to invent.
    pub fn marginal(&self, k: usize) -> Result<Estimate, BayesError> {
        let x = self.indicator(k).ok_or(BayesError::OutOfRange {
            what: "unit index",
            value: k as f64,
            low: 0.0,
            high: (self.units.saturating_sub(1)) as f64,
        })?;
        estimate(&x)
    }

    /// Histogram the recorded configurations.
    ///
    /// # Errors
    ///
    /// As [`Histogram::new`].
    pub fn histogram(&self) -> Result<Histogram, BayesError> {
        let mut h = Histogram::new(self.units)?;
        for &z in &self.states {
            h.observe(z);
        }
        Ok(h)
    }
}

// ---------------------------------------------------------------------------------------------
// Error bars that know they came from a chain
// ---------------------------------------------------------------------------------------------

/// A mean read off a correlated series, with both the honest error bar and the tempting one.
///
/// Successive samples from a Markov chain are dependent, so the variance of their mean is not
/// `σ²/n`. It is `σ²·τ_int/n`, where `τ_int` is the **integrated autocorrelation time** — the number
/// of consecutive samples that carry, between them, the information of one independent draw. The
/// effective sample size is `n / τ_int`, and the error bar that matters is built on it.
///
/// Both bars are carried because the wrong one is what a reader will otherwise compute. [`sem`] is
/// the one to report; [`naive_sem`] is here so that [`understatement`] can say by how much a chain
/// would have lied. For an anti-correlated series `τ_int < 1`, the effective sample size is
/// *larger* than the raw count and [`understatement`] comes back below 1 — the naive bar was too
/// wide, not too narrow, and the type does not hide that either.
///
/// [`sem`]: Estimate::sem
/// [`naive_sem`]: Estimate::naive_sem
/// [`understatement`]: Estimate::understatement
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Estimate {
    /// Sample mean, in the units of the series.
    pub mean: f64,
    /// Sample standard deviation, in the units of the series, with the `n − 1` denominator.
    pub sd: f64,
    /// Number of samples the mean was taken over. **Not** the number of independent ones.
    pub n: usize,
    /// Integrated autocorrelation time, in samples: `(1+φ)/(1−φ)` for an `AR(1)` chain of
    /// coefficient `φ`, which is **below 1 for a negative `φ`**. See
    /// [`integrated_autocorrelation_time`]; `1.0` exactly, with [`Estimate::floored`] set, is the
    /// one value that is a fallback rather than a measurement.
    pub iact: f64,
    /// Effective sample size, `n / iact`.
    ///
    /// Larger than `n` when `iact < 1`, which is the honest answer for an anti-correlated series:
    /// its mean really is better determined than `n` independent draws would give.
    pub ess: f64,
    /// Standard error of the mean, `sd / sqrt(ess)`. **This is the one to report.**
    ///
    /// A **lower bound** rather than an estimate when [`Estimate::truncated_at_cap`] is set, and a
    /// naive bar wearing this field's name when [`Estimate::floored`] is set.
    pub sem: f64,
    /// Standard error the raw count would have given, `sd / sqrt(n)`.
    ///
    /// Present as a counterexample, not as an alternative. For the correlated chains in this
    /// module's tests it is between three and five times too small, and a 95% interval built on it
    /// covers about 35% of the time.
    pub naive_sem: f64,
    /// Highest lag the autocorrelation sum reached before truncating, in samples.
    ///
    /// Odd whenever anything was summed, because Geyer's pairs end on an odd lag.
    pub lags: usize,
    /// [`Iact::truncated_at_cap`], carried through: the correlation outlasted the lag window, so
    /// `iact` is a lower bound, `ess` an upper bound, and `sem` **still too narrow**.
    ///
    /// The cap is `min(max_lag, n/2)` and is not on this struct, which is exactly why the flag has
    /// to be: `lags` alone cannot be compared against a number the caller does not hold.
    pub truncated_at_cap: bool,
    /// [`Iact::floored`], carried through: the raw estimate was non-positive and `iact` is the
    /// fallback `1.0`, so `sem` is the naive bar under another name.
    pub floored: bool,
}

impl Estimate {
    /// Half-width of a 95% interval on the mean, using the effective sample size.
    ///
    /// `1.96` is the normal quantile; for an effective sample size in the tens the `t` quantile
    /// would be a little wider, and this does not apply that correction. Stated rather than hidden:
    /// at `ess = 20` the true two-sided 95% quantile is 2.09, so this interval is about 7% narrow.
    #[must_use]
    pub fn half_width95(&self) -> f64 {
        1.96 * self.sem
    }

    /// The 95% interval on the mean: `(mean − half_width95, mean + half_width95)`.
    #[must_use]
    pub fn ci95(&self) -> (f64, f64) {
        (self.mean - self.half_width95(), self.mean + self.half_width95())
    }

    /// The interval a raw-count error bar would have claimed, for comparison.
    ///
    /// The same `1.96` normal quantile as [`Estimate::half_width95`], applied to
    /// [`Estimate::naive_sem`] — so the two intervals differ only by the effective sample size and
    /// not by the quantile, which is what makes their ratio `sqrt(iact)`.
    #[must_use]
    pub fn naive_ci95(&self) -> (f64, f64) {
        let h = 1.96 * self.naive_sem;
        (self.mean - h, self.mean + h)
    }

    /// How many times too narrow the raw-count error bar is: `sem / naive_sem`, which is
    /// `sqrt(iact)`. Below 1 for an anti-correlated series, where the naive bar was too wide.
    ///
    /// Returns `None` when the naive bar is zero, which happens only for a series whose sample
    /// variance is zero — and [`estimate`] refuses those before they reach here.
    #[must_use]
    pub fn understatement(&self) -> Option<f64> {
        (self.naive_sem > 0.0).then(|| self.sem / self.naive_sem)
    }
}

/// An integrated autocorrelation time and the window it was measured over.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Iact {
    /// `−1 + 2 Σ_{k=0}^{K} Γ_k` over Geyer's initial positive sequence, in samples.
    ///
    /// Equivalently `1 + 2 Σ_{l=1}^{2K+1} ρ_l`. **It is not floored at 1**: an anti-correlated
    /// series has an integrated autocorrelation time genuinely below one — an `AR(1)` chain of
    /// coefficient `φ` has exactly `(1+φ)/(1−φ)`, which is `0.0526` at `φ = −0.9` — and reporting
    /// `1.0` there would be a claim about the estimator rather than about the chain. The one
    /// exception is [`Iact::floored`].
    pub value: f64,
    /// Highest lag included in the sum, in samples, which is `2K+1` and therefore **odd** whenever
    /// anything was included. `0` means even `Γ_0 = 1 + ρ_1` came out non-positive, which needs a
    /// series that alternates almost perfectly.
    pub lags: usize,
    /// Whether the sum reached the estimator's lag cap without turning negative.
    ///
    /// `true` means [`Iact::value`] is a **lower bound**: the correlation outlasted the window, so
    /// the real effective sample size is smaller than the one reported and the error bar is still
    /// too narrow, just less so.
    pub truncated_at_cap: bool,
    /// Whether the raw sum came out non-positive and [`Iact::value`] is the fallback `1.0`.
    ///
    /// Geyer's Theorem 3.1 makes every `Γ_k` strictly positive for a reversible chain, and the
    /// *infinite* sum then satisfies `2 Σ Γ_k > γ_0`, so the exact-arithmetic answer is positive.
    /// A **truncated** sum can fall short of `γ_0/2` on a strongly anti-correlated series —
    /// perfect alternation is the extreme, where every `Γ_k` is `1/n` — and `−1 + 2 Σ` is then
    /// negative. A negative time would make `n / value` a negative effective sample size and
    /// `sd / sqrt(ess)` a `NaN`, so `1.0` is reported instead: the independent-samples time, which
    /// is the conservative direction because the error bar it implies is the naive one and never
    /// narrower. `true` says the number is that fallback and not a measurement.
    pub floored: bool,
}

/// The default ceiling on the lag window, in samples.
///
/// This constant is a **cost** ceiling — it bounds the estimator at `O(n · 4096)` for a
/// pathologically slow chain — and [`Iact::truncated_at_cap`] reports when it binds. The separate
/// `n/2` ceiling inside the estimator is a **statistical** one and is not redundant with it:
/// `the_geyer_window_on_a_perfectly_alternating_series_is_hand_computable` is an eight-sample
/// series whose pairs stay positive to lag 7, so dropping the `n/2` clamp there changes both the
/// reported lag and the truncation flag.
///
/// For the chains anyone actually runs the `n/2` ceiling does not bind, and there is a reason: the
/// sample autocovariances of a mean-centred series satisfy `Σ_{l} (1 − |l|/n) c_l = 0` exactly, so
/// the estimates must go negative somewhere, and in practice they do so well before half the
/// series length.
pub const DEFAULT_MAX_LAG: usize = 4096;

/// Integrated autocorrelation time by Geyer's initial positive sequence.
///
/// Geyer, *Practical Markov Chain Monte Carlo*, Statistical Science 7(4):473–483, 1992. The
/// autocorrelation estimates `ρ_l` are noise beyond a few correlation times, and summing them all
/// adds variance without adding signal. Geyer's Theorem 3.1 is that for a reversible chain the
/// pairs
///
/// ```text
/// Γ_k = γ_{2k} + γ_{2k+1}        k = 0, 1, 2, …
/// ```
///
/// are strictly positive, strictly decreasing and strictly convex in exact arithmetic, so the first
/// pair that comes out non-positive marks where the estimates have become noise. The sum stops
/// there, and `σ² = −γ_0 + 2 Σ_{k=0}^{K} Γ_k` gives
///
/// ```text
/// τ_int = −1 + 2 Σ_{k=0}^{K} (ρ_{2k} + ρ_{2k+1})  =  1 + 2 Σ_{l=1}^{2K+1} ρ_l
/// ```
///
/// **The pairing starts at lag 0 and the theorem is about that pairing.** Pairing the other way —
/// `ρ_1 + ρ_2`, `ρ_3 + ρ_4`, … — gives the same infinite sum and a different *truncation*, and the
/// positivity theorem does not cover it: `Γ_0 = 1 + ρ_1 ≥ 0` always, so Geyer's sequence cannot
/// truncate before it has added anything, while the shifted one can and does. On a reversible chain
/// with one fast anti-correlated mode and one slow positive one — `ρ_l = 0.95·(−0.5)^l +
/// 0.05·(0.99)^l`, integrated time 10.267 — the shifted pairing's first pair is
/// `ρ_1 + ρ_2 = −0.139`, it truncates at once, and it calls a chain worth 10 samples per
/// independent draw an independent sequence. That is
/// `a_chain_with_a_fast_negative_mode_and_a_slow_positive_one_is_not_called_independent`.
///
/// `value` is therefore **not** floored at 1: an anti-correlated chain has an integrated time below
/// one, and the estimator tracks the `AR(1)` closed form `(1+φ)/(1−φ)` at `φ = −0.9` (`0.0526`) as
/// well as at `φ = 0.9` (`19`) — both in
/// `the_autocorrelation_time_of_an_ar1_chain_matches_its_closed_form`, where the bias runs about
/// 2.5% high at `φ = 0.9` and 5% low at `φ = −0.9` over 20 000 samples. The single exception is
/// [`Iact::floored`], for a truncated sum that comes out non-positive.
///
/// Lags are computed one at a time and the loop stops at the first non-positive pair, so the cost
/// is `O(n · lags)` rather than `O(n²)`.
///
/// # Errors
///
/// [`BayesError::TooShort`] for fewer than 8 samples — an autocorrelation from four pairs is not an
/// estimate — [`BayesError::NonFinite`] for a non-finite sample, and [`BayesError::NoVariation`]
/// for a constant series, which has no autocorrelation to measure.
pub fn integrated_autocorrelation_time(x: &[f64]) -> Result<Iact, BayesError> {
    integrated_autocorrelation_time_within(x, DEFAULT_MAX_LAG)
}

/// Integrated autocorrelation time with an explicit ceiling on the lag window.
///
/// The highest lag the sum may reach is `min(max_lag, n/2)`, and [`Iact::truncated_at_cap`] says
/// whether it stopped because the window ran out rather than because the autocorrelations turned
/// negative. A caller with a fixed time budget sets this; a caller who does not care uses
/// [`integrated_autocorrelation_time`] and its [`DEFAULT_MAX_LAG`].
///
/// **A truncated estimate is a lower bound**, so the effective sample size it implies is an upper
/// bound and the error bar built on it is still too narrow — less so than the raw-count bar, but
/// not enough. That is why the flag is on the struct rather than in a log line, and why
/// [`Estimate`] carries it too.
///
/// `max_lag == 1` is accepted and does real work: it admits `Γ_0 = 1 + ρ_1` and nothing else, so
/// the answer is `1 + 2ρ_1`, the lag-one correction on its own.
///
/// # Errors
///
/// As [`integrated_autocorrelation_time`], plus [`BayesError::OutOfRange`] for `max_lag == 0`,
/// which admits no pair at all and would make every series look uncorrelated.
pub fn integrated_autocorrelation_time_within(
    x: &[f64],
    max_lag: usize,
) -> Result<Iact, BayesError> {
    const MIN_SAMPLES: usize = 8;
    if max_lag == 0 {
        return Err(BayesError::OutOfRange {
            what: "max_lag",
            value: 0.0,
            low: 1.0,
            high: f64::INFINITY,
        });
    }
    if x.len() < MIN_SAMPLES {
        return Err(BayesError::TooShort { got: x.len(), want: MIN_SAMPLES });
    }
    finite(x, "series")?;
    let n = x.len();
    let mean = x.iter().sum::<f64>() / n as f64;
    let c0 = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n as f64;
    if c0 <= 0.0 {
        return Err(BayesError::NoVariation { n, value: x[0] });
    }
    let rho = |lag: usize| -> f64 {
        let mut s = 0.0;
        for t in 0..(n - lag) {
            s += (x[t] - mean) * (x[t + lag] - mean);
        }
        s / (n as f64) / c0
    };
    // n/2 because an autocorrelation at lag n/2 is an average of n/2 products and is already
    // mostly noise; beyond it the estimate is not informative at any n. `max_lag` is the caller's
    // cost ceiling on top of that. Neither bound is what makes the loop terminate — `rho(lag)` is
    // an empty sum and therefore 0 for `lag >= n`, so the pair would be non-positive — but the
    // `n/2` bound does bind, and visibly: see
    // `the_geyer_window_on_a_perfectly_alternating_series_is_hand_computable`.
    let cap = max_lag.min(n / 2);
    // Geyer's pairing, Theorem 3.1: Γ_k = ρ_{2k} + ρ_{2k+1}, k = 0, 1, 2, …, starting at LAG ZERO.
    let mut sum = 0.0;
    let mut lags = 0usize;
    let mut k = 0usize;
    let mut truncated_at_cap = true;
    loop {
        // The pair spans lags `2k` and `2k+1`; the higher of the two is what has to fit the cap.
        let hi = 2 * k + 1;
        if hi > cap {
            break;
        }
        let g = rho(2 * k) + rho(hi);
        if g <= 0.0 {
            truncated_at_cap = false;
            break;
        }
        sum += g;
        lags = hi;
        k += 1;
    }
    let raw = -1.0 + 2.0 * sum;
    let floored = !(raw > 0.0);
    Ok(Iact { value: if floored { 1.0 } else { raw }, lags, truncated_at_cap, floored })
}

/// Mean and chain-aware error bar for a series of samples from a Markov chain.
///
/// Uses [`DEFAULT_MAX_LAG`]; [`estimate_within`] takes the ceiling explicitly.
///
/// # Errors
///
/// As [`integrated_autocorrelation_time`]: [`BayesError::TooShort`] under 8 samples,
/// [`BayesError::NonFinite`], and [`BayesError::NoVariation`] for a constant series — refused
/// rather than reported with a zero error bar, because a marginal of exactly 0 with an uncertainty
/// of exactly 0 is a claim the sample cannot support.
pub fn estimate(x: &[f64]) -> Result<Estimate, BayesError> {
    estimate_within(x, DEFAULT_MAX_LAG)
}

/// Mean and chain-aware error bar, with an explicit ceiling on the lag window.
///
/// Every flag [`integrated_autocorrelation_time_within`] sets is carried onto the returned
/// [`Estimate`] — [`Estimate::truncated_at_cap`] and [`Estimate::floored`] — because the type that
/// carries the error bar is the one that has to say the bar is a lower bound.
///
/// # Errors
///
/// As [`integrated_autocorrelation_time_within`], including [`BayesError::OutOfRange`] for
/// `max_lag == 0`.
pub fn estimate_within(x: &[f64], max_lag: usize) -> Result<Estimate, BayesError> {
    let iact = integrated_autocorrelation_time_within(x, max_lag)?;
    let n = x.len();
    let mean = x.iter().sum::<f64>() / n as f64;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (n as f64 - 1.0);
    let sd = var.sqrt();
    let ess = n as f64 / iact.value;
    Ok(Estimate {
        mean,
        sd,
        n,
        iact: iact.value,
        ess,
        sem: sd / ess.sqrt(),
        naive_sem: sd / (n as f64).sqrt(),
        lags: iact.lags,
        truncated_at_cap: iact.truncated_at_cap,
        floored: iact.floored,
    })
}

// ---------------------------------------------------------------------------------------------
// Bayesian confidence from population activity
// ---------------------------------------------------------------------------------------------

/// Which of two hypotheses a readout favours.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Hypothesis {
    /// The first hypothesis, whose rates were passed as `rate_a`.
    A,
    /// The second hypothesis, whose rates were passed as `rate_b`.
    B,
}

/// A posterior over two hypotheses, read off a population's spike counts.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Confidence {
    /// Log posterior odds of `A` against `B`, in nats. Positive favours `A`.
    pub log_odds: f64,
    /// Posterior probability of `A`, in `[0, 1]`: `logistic(log_odds)`.
    pub probability_a: f64,
    /// Which hypothesis the posterior favours; `A` on an exact tie, arbitrarily and documentedly.
    pub favours: Hypothesis,
}

impl Confidence {
    /// Wrap a log posterior odds in nats.
    #[must_use]
    pub fn from_log_odds(log_odds: f64) -> Self {
        Self {
            log_odds,
            probability_a: logistic(log_odds),
            favours: if log_odds >= 0.0 { Hypothesis::A } else { Hypothesis::B },
        }
    }
}

fn check_rates(rates: &[f64], what: &'static str) -> Result<(), BayesError> {
    if rates.is_empty() {
        return Err(BayesError::Empty { what });
    }
    for (i, &r) in rates.iter().enumerate() {
        if !r.is_finite() {
            return Err(BayesError::NonFinite { what, index: i });
        }
        if !(r > 0.0) {
            return Err(BayesError::OutOfRange {
                what,
                value: r,
                low: f64::MIN_POSITIVE,
                high: f64::INFINITY,
            });
        }
    }
    Ok(())
}

/// Log posterior odds of hypothesis `A` against `B`, given a population's spike counts.
///
/// The likelihood is independent Poisson per neuron — Ma, Beck, Latham & Pouget, *Bayesian
/// inference with probabilistic population codes*, Nature Neuroscience 9:1432–1438, 2006. Their
/// central result is that for this noise model the log-likelihood is **linear in the spike
/// counts**, so a Bayesian readout is a weighted sum a downstream neuron can already compute:
///
/// ```text
/// log p(n|A) − log p(n|B) = Σ_i [ n_i · ln(λ_i^A / λ_i^B) − T·(λ_i^A − λ_i^B) ]
/// ```
///
/// The first term is the weighted sum; the second is a count-independent offset, the price of
/// having expected different totals under the two hypotheses. **Dropping that offset is the usual
/// error**, and it does not change which hypothesis wins on a symmetric population — it changes the
/// *confidence*, which is the only thing this function is for.
///
/// Rates are in **hertz** and `window_s` in **seconds**; they meet at `λ·T`, the dimensionless
/// Poisson mean. `prior_log_odds` is in nats and is added, which is Bayes' rule in log form.
///
/// # Errors
///
/// [`BayesError::Empty`] for an empty population, [`BayesError::ShapeMismatch`] if the three
/// slices disagree in length, [`BayesError::NonFinite`] for a non-finite rate or prior, and
/// [`BayesError::OutOfRange`] for a rate that is not strictly positive or a non-positive window.
///
/// A rate of zero is refused rather than accepted: it makes one likelihood exactly zero, so a
/// single spike from that neuron would drive the posterior to certainty. That is a modelling
/// artefact — no population has a cell that literally never fires — and infinite confidence from
/// one spike is the worst failure mode a confidence readout has.
pub fn population_log_odds(
    counts: &[u64],
    rate_a: &[f64],
    rate_b: &[f64],
    window_s: f64,
    prior_log_odds: f64,
) -> Result<f64, BayesError> {
    check_rates(rate_a, "rate_a")?;
    check_rates(rate_b, "rate_b")?;
    if rate_b.len() != rate_a.len() {
        return Err(BayesError::ShapeMismatch {
            what: "rate_b",
            got: rate_b.len(),
            want: rate_a.len(),
        });
    }
    if counts.len() != rate_a.len() {
        return Err(BayesError::ShapeMismatch {
            what: "counts",
            got: counts.len(),
            want: rate_a.len(),
        });
    }
    finite_scalar(window_s, "window_s")?;
    finite_scalar(prior_log_odds, "prior_log_odds")?;
    if !(window_s > 0.0) {
        return Err(BayesError::OutOfRange {
            what: "window_s",
            value: window_s,
            low: f64::MIN_POSITIVE,
            high: f64::INFINITY,
        });
    }
    let mut llr = prior_log_odds;
    for i in 0..rate_a.len() {
        llr += counts[i] as f64 * (rate_a[i] / rate_b[i]).ln()
            - window_s * (rate_a[i] - rate_b[i]);
    }
    Ok(llr)
}

/// The Kullback-Leibler divergence `KL(Poisson(λ_p·T) ‖ Poisson(λ_q·T))` summed over a population,
/// in nats.
///
/// For one Poisson pair with means `a` and `b`, `KL = a·ln(a/b) + b − a`. Summing over independent
/// neurons sums the divergences.
///
/// **This is the closed form the log-odds readout is checked against.** The expected log-likelihood
/// ratio under the hypothesis that generated the data is exactly this divergence, so averaging
/// [`population_log_odds`] over many simulated trials must converge to it — which is
/// `the_mean_log_odds_under_a_hypothesis_is_the_divergence_between_the_likelihoods`, and it tests
/// the offset term that a wrong implementation would drop.
///
/// # Errors
///
/// As [`population_log_odds`], for the rates and window.
pub fn poisson_kl(rate_p: &[f64], rate_q: &[f64], window_s: f64) -> Result<f64, BayesError> {
    check_rates(rate_p, "rate_p")?;
    check_rates(rate_q, "rate_q")?;
    if rate_q.len() != rate_p.len() {
        return Err(BayesError::ShapeMismatch {
            what: "rate_q",
            got: rate_q.len(),
            want: rate_p.len(),
        });
    }
    finite_scalar(window_s, "window_s")?;
    if !(window_s > 0.0) {
        return Err(BayesError::OutOfRange {
            what: "window_s",
            value: window_s,
            low: f64::MIN_POSITIVE,
            high: f64::INFINITY,
        });
    }
    let mut kl = 0.0;
    for i in 0..rate_p.len() {
        let (a, b) = (rate_p[i] * window_s, rate_q[i] * window_s);
        kl += a * (a / b).ln() + b - a;
    }
    Ok(kl)
}

/// Draw one trial's spike counts from a population of independent Poisson neurons.
///
/// Uses [`crate::coding::poisson_count`], so the draw is the crate's one Poisson sampler and is
/// deterministic in the seed. Rates in **hertz**, window in **seconds**.
///
/// # Errors
///
/// As [`population_log_odds`] for the rates and window, and [`BayesError::OutOfRange`] naming the
/// neuron whose mean `λ·T` exceeded what the Poisson sampler will draw.
pub fn sample_population(
    rng: &mut Rng,
    rates: &[f64],
    window_s: f64,
) -> Result<Vec<u64>, BayesError> {
    check_rates(rates, "rates")?;
    finite_scalar(window_s, "window_s")?;
    if !(window_s > 0.0) {
        return Err(BayesError::OutOfRange {
            what: "window_s",
            value: window_s,
            low: f64::MIN_POSITIVE,
            high: f64::INFINITY,
        });
    }
    let mut out = Vec::with_capacity(rates.len());
    for &r in rates {
        let lambda = r * window_s;
        let c = crate::coding::poisson_count(rng, lambda).ok_or(BayesError::OutOfRange {
            what: "expected count for a neuron",
            value: lambda,
            low: 0.0,
            high: f64::INFINITY,
        })?;
        out.push(c);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------------------------
// Hyperdimensional / vector-symbolic computing
// ---------------------------------------------------------------------------------------------

/// The largest hypervector dimension this module will allocate, in bits.
///
/// 16 Mibit is 2 MiB per vector. The bound exists so that a mistyped dimension fails immediately
/// rather than after a multi-gigabyte allocation; nothing in the theory stops at this size.
pub const MAX_DIM: usize = 1 << 24;

/// A binary hypervector: `dim` bits, packed into `u64` words, tail bits held at zero.
///
/// Kanerva, *Hyperdimensional Computing*, Cognitive Computation 1:139–159, 2009. The whole
/// argument rests on one fact about high-dimensional spaces: two random `D`-bit vectors are almost
/// exactly `D/2` bits apart, with standard deviation `√D/2`, so at `D = 10 000` **essentially the
/// entire space sits within five standard deviations of orthogonal to any given vector**. That
/// leaves a vast supply of mutually near-orthogonal symbols, and it means "closer than random" is a
/// decision that can be made with confidence from a single comparison.
///
/// Three operations, and each does exactly one thing to similarity:
///
/// | operation | implementation | effect on similarity |
/// |---|---|---|
/// | **bind** (`⊗`) | bitwise `XOR` | destroys it — the result is near-orthogonal to both operands |
/// | **bundle** (`+`) | bitwise majority | preserves it — the result is near *all* its components |
/// | **permute** (`ρ`) | cyclic shift | destroys it, invertibly, and is order-sensitive |
///
/// Binding is its own inverse, exactly, with no error: `(a ⊗ b) ⊗ b = a` bit for bit. That is what
/// makes role-filler structure work — bind a role to a filler, bundle the pairs, and unbind by the
/// role to recover the filler approximately, then clean up against a [`Codebook`].
///
/// Bundling is where the loss lives, and the loss has a closed form: see [`bundle_similarity`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Hypervector {
    dim: usize,
    words: Vec<u64>,
}

fn check_dim(dim: usize) -> Result<usize, BayesError> {
    if dim == 0 {
        return Err(BayesError::Empty { what: "dimension" });
    }
    if dim > MAX_DIM {
        return Err(BayesError::TooLarge { what: "dimension", got: dim, limit: MAX_DIM });
    }
    Ok(dim.div_ceil(64))
}

impl Hypervector {
    /// The all-zero vector of `dim` bits.
    ///
    /// # Errors
    ///
    /// [`BayesError::Empty`] for `dim == 0`, [`BayesError::TooLarge`] past [`MAX_DIM`].
    pub fn zeros(dim: usize) -> Result<Self, BayesError> {
        let words = check_dim(dim)?;
        Ok(Self { dim, words: vec![0; words] })
    }

    /// A uniformly random vector of `dim` bits.
    ///
    /// Bits past `dim` within the last word are forced to zero, which every other operation then
    /// preserves; without that, [`Hypervector::hamming`] would count padding as signal for any
    /// `dim` that is not a multiple of 64 — and Kanerva's own 10 000 is not.
    ///
    /// # Errors
    ///
    /// As [`Hypervector::zeros`].
    pub fn random(rng: &mut Rng, dim: usize) -> Result<Self, BayesError> {
        let nwords = check_dim(dim)?;
        let mut words = Vec::with_capacity(nwords);
        for _ in 0..nwords {
            words.push((u64::from(rng.next_u32()) << 32) | u64::from(rng.next_u32()));
        }
        let rem = dim % 64;
        if rem != 0 {
            let last = nwords - 1;
            words[last] &= (1u64 << rem) - 1;
        }
        Ok(Self { dim, words })
    }

    /// Dimension in bits.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The packed words, least-significant bit of word 0 being bit 0.
    #[must_use]
    pub fn words(&self) -> &[u64] {
        &self.words
    }

    /// Bit `index`, or `None` past the end.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<bool> {
        (index < self.dim).then(|| self.words[index / 64] >> (index % 64) & 1 == 1)
    }

    /// Set bit `index`.
    ///
    /// # Errors
    ///
    /// [`BayesError::OutOfRange`] past the end. Silently ignoring an out-of-range write would let a
    /// caller build a vector missing exactly the bits it thought it had set.
    pub fn set(&mut self, index: usize, value: bool) -> Result<(), BayesError> {
        if index >= self.dim {
            return Err(BayesError::OutOfRange {
                what: "bit index",
                value: index as f64,
                low: 0.0,
                high: (self.dim - 1) as f64,
            });
        }
        let (w, b) = (index / 64, index % 64);
        if value {
            self.words[w] |= 1u64 << b;
        } else {
            self.words[w] &= !(1u64 << b);
        }
        Ok(())
    }

    /// How many bits this vector's population is: the count of ones.
    #[must_use]
    pub fn ones(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// Bind: bitwise `XOR`. **Its own inverse, exactly.**
    ///
    /// `a.bind(&b).bind(&b) == a` bit for bit — no tolerance, no cleanup, no dimension-dependent
    /// error. The result is near-orthogonal to both operands, which is what makes a bound pair a
    /// new symbol rather than a blend of two old ones.
    ///
    /// `XOR` also preserves Hamming distance: `d(a⊗c, b⊗c) = d(a, b)` for any `c`. That is why a
    /// whole structure can be bound by one key and unbound later with every internal similarity
    /// intact.
    ///
    /// # Errors
    ///
    /// [`BayesError::DimensionMismatch`] if the two vectors are different sizes.
    pub fn bind(&self, other: &Self) -> Result<Self, BayesError> {
        if self.dim != other.dim {
            return Err(BayesError::DimensionMismatch { a: self.dim, b: other.dim });
        }
        let words = self.words.iter().zip(&other.words).map(|(a, b)| a ^ b).collect();
        Ok(Self { dim: self.dim, words })
    }

    /// Unbind: the same operation as [`Hypervector::bind`], named for the direction of use.
    ///
    /// Provided as a distinct name because a reader of `x.unbind(&role)` can tell what is intended,
    /// and because in other vector-symbolic algebras — holographic reduced representations, for
    /// instance — binding and unbinding are genuinely different operations. Here they are the same
    /// one, and pretending otherwise would misrepresent the algebra.
    ///
    /// # Errors
    ///
    /// As [`Hypervector::bind`].
    pub fn unbind(&self, other: &Self) -> Result<Self, BayesError> {
        self.bind(other)
    }

    /// Permute: cyclic shift by `shift` positions, wrapping at `dim`.
    ///
    /// The result is near-orthogonal to the input for any nonzero shift, and `permute(-s)` undoes
    /// `permute(s)` exactly. This is how order enters an otherwise commutative algebra: bundling is
    /// a set operation, so `a + b` cannot represent a sequence, but `ρ(a) + ρ²(b)` can.
    ///
    /// A shift of zero is the identity, and a shift of `dim` is too, which is why the shift is
    /// reduced modulo `dim` rather than rejected.
    #[must_use]
    pub fn permute(&self, shift: i64) -> Self {
        let d = self.dim as i64;
        let s = shift.rem_euclid(d);
        let mut out = Self { dim: self.dim, words: vec![0; self.words.len()] };
        if s == 0 {
            out.words.copy_from_slice(&self.words);
            return out;
        }
        for i in 0..self.dim {
            if self.words[i / 64] >> (i % 64) & 1 == 1 {
                let j = ((i as i64 + s) % d) as usize;
                out.words[j / 64] |= 1u64 << (j % 64);
            }
        }
        out
    }

    /// Hamming distance in bits: how many positions differ.
    ///
    /// # Errors
    ///
    /// [`BayesError::DimensionMismatch`].
    pub fn hamming(&self, other: &Self) -> Result<usize, BayesError> {
        if self.dim != other.dim {
            return Err(BayesError::DimensionMismatch { a: self.dim, b: other.dim });
        }
        Ok(self
            .words
            .iter()
            .zip(&other.words)
            .map(|(a, b)| (a ^ b).count_ones() as usize)
            .sum())
    }

    /// Normalised similarity in `[-1, 1]`: `1 − 2·hamming/dim`.
    ///
    /// `1.0` for identical vectors, `−1.0` for complementary ones, and `0.0 ± 1/√dim` for
    /// independent ones — that last spread is [`random_similarity_sd`], and it is the yardstick
    /// every other number in this section is measured against. The scale is the cosine similarity
    /// of the same vectors read as `±1`, which is why it is called a similarity and not a distance.
    ///
    /// # Errors
    ///
    /// [`BayesError::DimensionMismatch`].
    pub fn similarity(&self, other: &Self) -> Result<f64, BayesError> {
        let h = self.hamming(other)?;
        Ok(1.0 - 2.0 * (h as f64) / (self.dim as f64))
    }

    /// Bundle: bitwise majority over `parts`, with a fair coin on ties.
    ///
    /// The result is similar to every component at once — that is the operation's whole purpose,
    /// and it is what a set, a superposition or an unordered record is represented by. The
    /// similarity it achieves is not a free parameter: [`bundle_similarity`] gives it in closed
    /// form as a function of the number of components only, and this implementation is checked
    /// against that form at 8192 bits over `k = 2, 3, 5, 9, 25`, where the worst disagreement is
    /// 6.1e-4 (at `k = 25`) against an asserted bound of 1e-3.
    ///
    /// Ties are only possible for an even number of components, and `rng` is drawn from **only**
    /// on a tie — so bundling an odd number of vectors consumes no randomness and is a pure
    /// function of its inputs. Kanerva's own exposition breaks ties by adding a fixed random
    /// vector; a fair coin per bit is the same in distribution and does not require the caller to
    /// carry one around.
    ///
    /// # Errors
    ///
    /// [`BayesError::Empty`] for no components — a bundle of nothing has no dimension to be — and
    /// [`BayesError::DimensionMismatch`] if the components disagree.
    pub fn bundle(parts: &[Self], rng: &mut Rng) -> Result<Self, BayesError> {
        let first = parts.first().ok_or(BayesError::Empty { what: "bundle components" })?;
        let dim = first.dim;
        for p in parts {
            if p.dim != dim {
                return Err(BayesError::DimensionMismatch { a: dim, b: p.dim });
            }
        }
        let mut count = vec![0u32; dim];
        for p in parts {
            for (wi, &w) in p.words.iter().enumerate() {
                let mut w = w;
                while w != 0 {
                    let idx = wi * 64 + w.trailing_zeros() as usize;
                    if idx < dim {
                        count[idx] += 1;
                    }
                    w &= w - 1;
                }
            }
        }
        let k = parts.len() as u32;
        let mut out = Self::zeros(dim)?;
        for i in 0..dim {
            let twice = 2 * count[i];
            let one = if twice > k {
                true
            } else if twice == k {
                rng.next_f64() < 0.5
            } else {
                false
            };
            if one {
                out.words[i / 64] |= 1u64 << (i % 64);
            }
        }
        Ok(out)
    }
}

/// The expected similarity between a majority bundle of `k` random binary hypervectors and one of
/// its own components, in closed form.
///
/// Let `j` be the number of the *other* `k − 1` components that disagree with the chosen one at a
/// given bit; `j ~ Binomial(k − 1, ½)`. The bundle's bit differs from the chosen component's when
/// the disagreeing votes carry it, so
///
/// ```text
/// d(k) = P(j > k/2) + ½·P(j = k/2)          (the ½ is the fair coin on an exact tie)
/// sim(k) = 1 − 2·d(k)
/// ```
///
/// which gives 1, ½, ½, ⅜, ⅜, 0.3125, 0.3125 for `k = 1…7` and approaches `√(2/(πk))` from above —
/// the ratio is 1.0047% high at `k = 25`, 0.248% at `k = 101`, 0.0250% at `k = 1001` and 0.0025%
/// at `k = 10 001`. For odd `k = 2m+1` the value is exactly `C(2m, m) / 2^{2m}`, so
/// `sim(25) = 2 704 156 / 16 777 216`. Two consequences worth knowing:
///
/// - **An even bundle is worth exactly the odd one below it.** `sim(2j) = sim(2j+1)` identically,
///   so bundling a fourth vector into a bundle of three buys nothing at all. The proof is
///   `C(2m, m) = 2·C(2m−1, m)`, and the test is
///   `an_even_bundle_is_worth_exactly_the_odd_one_below_it`.
/// - **The signal falls as `1/√k` while the noise floor stays at `1/√dim`**, so the number of items
///   a bundle can hold grows linearly in the dimension. That is [`bundle_z_score`].
///
/// Computed from the binomial mass anchored at its mode, so it neither underflows nor needs a
/// factorial: at `k = 10 000` the direct form `2^-(k-1)` would be zero in `f64`.
///
/// Returns `None` for `k == 0` — a bundle of nothing has no similarity to anything — or for `k`
/// past 65 536, where the `O(k)` array is no longer a sensible thing to allocate for a constant.
#[must_use]
pub fn bundle_similarity(k: usize) -> Option<f64> {
    const MAX_K: usize = 1 << 16;
    if k == 0 || k > MAX_K {
        return None;
    }
    let n = k - 1;
    let mut r = vec![0.0f64; n + 1];
    let mode = n / 2;
    r[mode] = 1.0;
    for j in mode..n {
        r[j + 1] = r[j] * ((n - j) as f64) / ((j + 1) as f64);
    }
    for j in (1..=mode).rev() {
        r[j - 1] = r[j] * (j as f64) / ((n - j + 1) as f64);
    }
    let total: f64 = r.iter().sum();
    let half = k as f64 / 2.0;
    let mut d = 0.0;
    for (j, &rj) in r.iter().enumerate() {
        let jf = j as f64;
        if jf > half {
            d += rj;
        } else if jf == half {
            d += 0.5 * rj;
        }
    }
    Some(1.0 - 2.0 * (d / total))
}

/// The standard deviation of the similarity between two **independent** hypervectors: `1/√dim`.
///
/// Hamming distance between random vectors is `Binomial(dim, ½)`, standard deviation `√dim/2`;
/// similarity is `1 − 2h/dim`, so its standard deviation is `1/√dim`. At Kanerva's `dim = 10 000`
/// that is 0.01 in similarity, equivalently 50 bits in distance out of 5000 — the figures his 2009
/// paper prints.
///
/// Returns `None` for `dim == 0`.
#[must_use]
pub fn random_similarity_sd(dim: usize) -> Option<f64> {
    (dim > 0).then(|| 1.0 / (dim as f64).sqrt())
}

/// How many standard deviations of noise separate a bundle from its own components:
/// `bundle_similarity(k) · √dim`.
///
/// **This is the capacity result, as a number you can act on.** Signal falls as `√(2/(πk))`, the
/// noise floor of an unrelated vector is `1/√dim`, so the detectability is `√(2·dim/(π·k))` — it
/// grows with dimension and falls with the number of items bundled, and recovery survives while
/// this stays above roughly 4.
///
/// Measured in `recovery_from_a_bundle_degrades_where_the_capacity_bound_says_it_does`, over
/// twelve independently drawn codebooks: at `dim = 1024` against a 512-item codebook, `k = 5`
/// scores 12.0 and recovers everything, `k = 25` scores 5.2 and recovers 0.983 — **not**
/// everything — and `k = 125` scores 2.3 and recovers 0.797.
///
/// Returns `None` when [`bundle_similarity`] does, or for `dim == 0`.
#[must_use]
pub fn bundle_z_score(dim: usize, k: usize) -> Option<f64> {
    let s = bundle_similarity(k)?;
    (dim > 0).then(|| s * (dim as f64).sqrt())
}

/// A named set of hypervectors, and the nearest-neighbour search that cleans up a noisy one.
///
/// The cleanup memory is the part of a vector-symbolic system that makes it usable: unbinding and
/// bundling both return something *near* the answer, and the codebook is what turns near into
/// exact. Kanerva calls the equivalent structure an item memory.
#[derive(Debug, Clone, PartialEq)]
pub struct Codebook {
    dim: usize,
    names: Vec<String>,
    vectors: Vec<Hypervector>,
}

impl Codebook {
    /// An empty codebook over `dim`-bit vectors.
    ///
    /// # Errors
    ///
    /// As [`Hypervector::zeros`].
    pub fn new(dim: usize) -> Result<Self, BayesError> {
        check_dim(dim)?;
        Ok(Self { dim, names: Vec::new(), vectors: Vec::new() })
    }

    /// A codebook of `count` independent random vectors, named `"0"`, `"1"`, …
    ///
    /// # Errors
    ///
    /// As [`Hypervector::random`], and [`BayesError::Empty`] for `count == 0`.
    pub fn random(rng: &mut Rng, dim: usize, count: usize) -> Result<Self, BayesError> {
        if count == 0 {
            return Err(BayesError::Empty { what: "codebook entries" });
        }
        let mut cb = Self::new(dim)?;
        for i in 0..count {
            cb.add(&i.to_string(), Hypervector::random(rng, dim)?)?;
        }
        Ok(cb)
    }

    /// Add a named vector.
    ///
    /// # Errors
    ///
    /// [`BayesError::DimensionMismatch`] if it does not match the codebook's dimension.
    pub fn add(&mut self, name: &str, v: Hypervector) -> Result<(), BayesError> {
        if v.dim != self.dim {
            return Err(BayesError::DimensionMismatch { a: self.dim, b: v.dim });
        }
        self.names.push(name.to_string());
        self.vectors.push(v);
        Ok(())
    }

    /// How many entries the codebook holds.
    #[must_use]
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// Whether the codebook holds nothing.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// The stored vectors, in insertion order.
    #[must_use]
    pub fn vectors(&self) -> &[Hypervector] {
        &self.vectors
    }

    /// The name of entry `i`, or `None` past the end.
    #[must_use]
    pub fn name(&self, i: usize) -> Option<&str> {
        self.names.get(i).map(String::as_str)
    }

    /// Every entry's index and similarity to `probe`, sorted most similar first.
    ///
    /// Ties are broken by index, so the order is deterministic for identical entries.
    ///
    /// # Errors
    ///
    /// [`BayesError::DimensionMismatch`], or [`BayesError::Empty`] for an empty codebook.
    pub fn rank(&self, probe: &Hypervector) -> Result<Vec<(usize, f64)>, BayesError> {
        if self.vectors.is_empty() {
            return Err(BayesError::Empty { what: "codebook" });
        }
        let mut out = Vec::with_capacity(self.vectors.len());
        for (i, v) in self.vectors.iter().enumerate() {
            out.push((i, probe.similarity(v)?));
        }
        out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal).then(a.0.cmp(&b.0)));
        Ok(out)
    }

    /// The nearest entry and its similarity.
    ///
    /// # Errors
    ///
    /// As [`Codebook::rank`].
    pub fn nearest(&self, probe: &Hypervector) -> Result<(usize, f64), BayesError> {
        Ok(self.rank(probe)?[0])
    }

    /// The nearest entry, but only if it clears `threshold` similarity.
    ///
    /// `Ok(None)` means *nothing in this memory resembles the probe*, which is a different answer
    /// from "the nearest one is entry 3", and a cleanup memory that cannot say it will confidently
    /// name a symbol for a vector carrying no signal at all. A threshold of `4/√dim` is four
    /// standard deviations of the random-similarity noise floor ([`random_similarity_sd`]) and is
    /// a defensible default.
    ///
    /// # Errors
    ///
    /// As [`Codebook::rank`], and [`BayesError::NonFinite`] for a non-finite threshold.
    pub fn cleanup(
        &self,
        probe: &Hypervector,
        threshold: f64,
    ) -> Result<Option<(usize, f64)>, BayesError> {
        finite_scalar(threshold, "threshold")?;
        let (i, s) = self.nearest(probe)?;
        Ok((s >= threshold).then_some((i, s)))
    }
}

#[cfg(test)]
mod tests {
    use super::{
        BayesError, Boltzmann, Codebook, Confidence, DEFAULT_MAX_LAG, Estimate, Histogram,
        Hypervector, Hypothesis, KANERVA_DIM, MAX_UNITS, NeuralSampler, PairwiseFit, Scan, Target,
        bundle_similarity, bundle_z_score, estimate, estimate_within,
        integrated_autocorrelation_time, integrated_autocorrelation_time_within, logistic,
        poisson_kl, population_log_odds, random_similarity_sd, sample_population,
    };
    use crate::rng::Rng;

    /// A coupled four-unit model used everywhere below. Four units because 16 configurations is
    /// small enough to enumerate exactly and large enough that a wrong sampler cannot match the
    /// marginals by accident; the couplings are of both signs so that a sign error anywhere shows.
    fn coupled() -> Boltzmann {
        let bias = [0.6, -0.4, 0.2, -1.1];
        let mut w = vec![0.0; 16];
        for (i, j, v) in [(0usize, 1usize, 1.2f64), (1, 2, -0.9), (2, 3, 0.7), (0, 3, -1.4)] {
            w[i * 4 + j] = v;
            w[j * 4 + i] = v;
        }
        Boltzmann::new(&bias, &w).expect("valid model")
    }

    /// Box-Muller, test-only: the crate ships no Gaussian sampler because nothing in it needs one,
    /// and the `AR(1)` chains below are a property of the estimator rather than of the library.
    fn gaussian(rng: &mut Rng) -> f64 {
        let mut u = rng.next_f64();
        if u <= 0.0 {
            u = f64::MIN_POSITIVE;
        }
        let v = rng.next_f64();
        (-2.0 * u.ln()).sqrt() * (2.0 * core::f64::consts::PI * v).cos()
    }

    /// An `AR(1)` chain with unit stationary variance and zero mean: `x_t = φ x_{t-1} + √(1−φ²) ε`.
    /// Its integrated autocorrelation time is `(1+φ)/(1−φ)` exactly, which is what makes it the
    /// closed form the effective-sample-size machinery is checked against.
    /// The sample autocorrelation at one lag, with the same `1/n` normalisation the estimator
    /// uses. Test-local and deliberately independent of the estimator's own inner closure, so that
    /// a test can look at the pairs the estimator is deciding on without going through it.
    fn sample_rho(x: &[f64], lag: usize) -> f64 {
        let n = x.len();
        let mean = x.iter().sum::<f64>() / n as f64;
        let c0 = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n as f64;
        let mut s = 0.0;
        for t in 0..(n - lag) {
            s += (x[t] - mean) * (x[t + lag] - mean);
        }
        s / (n as f64) / c0
    }

    /// A reversible chain with a FAST ANTI-CORRELATED mode and a SLOW POSITIVE one: the sum of two
    /// independent `AR(1)` processes with coefficients `−0.5` and `0.99`, carrying 0.95 and 0.05 of
    /// the variance. Its autocorrelation is `ρ_l = 0.95·(−0.5)^l + 0.05·(0.99)^l` and its
    /// integrated autocorrelation time is [`TWO_MODE_TAU`]. This is the shape a Gibbs-type sampler
    /// with one strongly coupled pair has, and it is the family in which the two candidate pair
    /// truncations disagree completely.
    fn two_mode(rng: &mut Rng, n: usize, burn: usize) -> Vec<f64> {
        let (p1, p2) = (-0.5f64, 0.99f64);
        let (a, b) = (0.95f64.sqrt(), 0.05f64.sqrt());
        let (s1, s2) = ((1.0 - p1 * p1).sqrt(), (1.0 - p2 * p2).sqrt());
        let (mut y, mut w) = (0.0, 0.0);
        for _ in 0..burn {
            y = p1 * y + s1 * gaussian(rng);
            w = p2 * w + s2 * gaussian(rng);
        }
        (0..n)
            .map(|_| {
                y = p1 * y + s1 * gaussian(rng);
                w = p2 * w + s2 * gaussian(rng);
                a * y + b * w
            })
            .collect()
    }

    /// `1 + 2 Σ_{l≥1} ρ_l` for [`two_mode`], written out rather than computed by the code under
    /// test: `1 + 2[0.95·(−0.5)/(1 + 0.5) + 0.05·0.99/(1 − 0.99)] = 1 + 2(−19/60 + 99/20)`.
    const TWO_MODE_TAU: f64 = 10.266_666_666_666_667;

    fn ar1(rng: &mut Rng, phi: f64, n: usize, burn: usize) -> Vec<f64> {
        let s = (1.0 - phi * phi).sqrt();
        let mut x = 0.0;
        for _ in 0..burn {
            x = phi * x + s * gaussian(rng);
        }
        (0..n)
            .map(|_| {
                x = phi * x + s * gaussian(rng);
                x
            })
            .collect()
    }

    // ---------------------------------------------------------------------------------------
    // The target distribution and the neural computability condition
    // ---------------------------------------------------------------------------------------

    #[test]
    fn a_model_refuses_an_asymmetric_coupling_a_self_coupling_and_a_nan() {
        let bias = [0.0, 0.0];
        let mut w = vec![0.0, 1.0, 2.0, 0.0];
        assert!(matches!(
            Boltzmann::new(&bias, &w),
            Err(BayesError::Asymmetric { i: 0, j: 1, .. })
        ));
        w = vec![0.5, 0.0, 0.0, 0.0];
        assert!(matches!(
            Boltzmann::new(&bias, &w),
            Err(BayesError::SelfCoupling { unit: 0, .. })
        ));
        assert!(matches!(
            Boltzmann::new(&[f64::NAN, 0.0], &[0.0; 4]),
            Err(BayesError::NonFinite { what: "bias", index: 0 })
        ));
        assert!(matches!(Boltzmann::new(&[], &[]), Err(BayesError::Empty { .. })));
        assert!(matches!(
            Boltzmann::new(&bias, &[0.0; 3]),
            Err(BayesError::ShapeMismatch { got: 3, want: 4, .. })
        ));
    }

    /// The uncoupled case has an exact answer with no enumeration in it at all: the marginals are
    /// the logistic of the biases, and the joint is their product. Checking enumeration against
    /// that catches a wrong factor of ½, a transposed index, or a sign flip in the energy.
    #[test]
    fn the_enumerated_distribution_of_an_uncoupled_model_is_the_product_of_logistics() {
        let bias = [0.6, -0.4, 0.2, -1.1];
        let m = Boltzmann::independent(&bias).expect("valid");
        let exact = m.exact().expect("small enough");
        for z in 0..16u64 {
            let mut want = 1.0;
            for (k, &b) in bias.iter().enumerate() {
                let p = logistic(b);
                want *= if z >> k & 1 == 1 { p } else { 1.0 - p };
            }
            assert!((exact[z as usize] - want).abs() < 1e-14, "state {z}: {} vs {want}", exact[z as usize]);
        }
        for (k, &b) in bias.iter().enumerate() {
            let got = m.exact_marginals().expect("small enough")[k];
            assert!((got - logistic(b)).abs() < 1e-14, "unit {k}");
        }

        // Energies large enough to overflow `exp` on their own still normalise, because the
        // largest is subtracted first. Without that, every entry here is `inf` and the whole
        // distribution comes back `NaN` — which no total-variation check would survive, but which
        // a caller would meet only on a strongly coupled model.
        let big = Boltzmann::new(&[800.0, 800.0], &[0.0, 900.0, 900.0, 0.0]).expect("valid");
        let p = big.exact().expect("small");
        assert!(p.iter().all(|v| v.is_finite()), "enumeration overflowed: {p:?}");
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!((p[3] - 1.0).abs() < 1e-300, "the dominant state should carry all the mass");
    }

    /// The coupling term, checked by hand on a model small enough to write the energy down.
    /// A two-unit model with bias `(b0, b1)` and coupling `w` has unnormalised weights
    /// `1, e^{b0}, e^{b1}, e^{b0+b1+w}` — the `w` appears once, which is what the ½ in front of the
    /// double sum is for.
    #[test]
    fn the_coupling_enters_the_energy_exactly_once_per_pair() {
        let (b0, b1, w) = (0.3, -0.7, 1.9);
        let m = Boltzmann::new(&[b0, b1], &[0.0, w, w, 0.0]).expect("valid");
        let raw = [0.0, b0, b1, b0 + b1 + w];
        for (z, &r) in raw.iter().enumerate() {
            assert!((m.energy(z as u64).expect("in range") - r).abs() < 1e-15, "state {z}");
        }
        let exact = m.exact().expect("small");
        let norm: f64 = raw.iter().map(|r| r.exp()).sum();
        for (z, &r) in raw.iter().enumerate() {
            assert!((exact[z] - r.exp() / norm).abs() < 1e-14, "state {z}");
        }
    }

    /// ⭐ A model whose inputs are all finite can still have energies that are not: `Boltzmann::new`
    /// checks the bias and coupling, and nothing stopped `E(z) = Σ b_k z_k + …` from overflowing to
    /// `+inf` on the way. `exact()` then computed `exp(inf − inf)` and returned `Ok` on a vector of
    /// `NaN`, which `<` and `>` compare `false` against silently — and `exact_marginals()` is the
    /// ground truth every sampler test in this module compares itself to.
    ///
    /// The same model also had to be refused by `Target::of_boltzmann`, which built its log-weights
    /// straight into the struct and skipped the finiteness gate that `Target::from_log_weights`
    /// runs on the identical numbers. Two constructors, one type, two invariants.
    #[test]
    fn an_energy_that_overflows_is_refused_rather_than_normalised_into_nan() {
        let huge = Boltzmann::new(&[1e308, 1e308], &[0.0; 4]).expect("finite inputs are accepted");
        assert_eq!(huge.energy(0b11), Some(f64::INFINITY), "the energy really does overflow");
        assert!(matches!(
            huge.exact(),
            Err(BayesError::NonFinite { what: "energy", index: 3 })
        ));
        assert!(matches!(huge.exact_marginals(), Err(BayesError::NonFinite { .. })));
        assert!(
            matches!(Target::of_boltzmann(&huge), Err(BayesError::NonFinite { what: "log_w", index: 3 })),
            "of_boltzmann accepted weights that from_log_weights refuses"
        );
        // …and the two constructors now agree on the identical numbers.
        let by_hand = [0.0, 1e308, 1e308, f64::INFINITY];
        assert!(matches!(
            Target::from_log_weights(2, &by_hand),
            Err(BayesError::NonFinite { what: "log_w", index: 3 })
        ));
        // The negative tail overflows too, and is caught at the first configuration that reaches it.
        let low = Boltzmann::new(&[-1e308, -1e308], &[0.0; 4]).expect("valid");
        assert!(matches!(low.exact(), Err(BayesError::NonFinite { what: "energy", .. })));

        // The regime the max-subtraction IS for — large but finite — still normalises, so this is
        // a refusal of the impossible case and not a retreat from the hard one.
        let big = Boltzmann::new(&[800.0, 800.0], &[0.0, 900.0, 900.0, 0.0]).expect("valid");
        let p = big.exact().expect("finite energies");
        assert!(p.iter().all(|v| v.is_finite()) && (p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(Target::of_boltzmann(&big).is_ok());
    }

    /// A state with bits set above the model's width is refused, not masked. A masked read would
    /// silently alias unit 4 onto unit 0 and produce an energy for a state that does not exist.
    #[test]
    fn a_state_wider_than_the_model_is_refused_rather_than_masked() {
        let m = coupled();
        assert!(m.energy(0b1111).is_some());
        assert!(m.energy(0b1_0000).is_none());
        assert!(m.membrane(0b1_0000, 0).is_none());
        assert!(m.membrane(0b0001, 4).is_none());
    }

    /// ⭐ The neural computability condition, verified rather than restated: the membrane
    /// potential — a plain weighted sum of the unit's inputs — is compared against the conditional
    /// log-odds computed by an independent route, differencing two enumerated log-weights.
    #[test]
    fn the_membrane_potential_is_the_exact_conditional_log_odds() {
        let m = coupled();
        let t = Target::of_boltzmann(&m).expect("small");
        let mut worst = 0.0f64;
        for z in 0..16u64 {
            for k in 0..4 {
                let linear = m.membrane(z, k).expect("in range");
                let exact = t.conditional_log_odds(z, k).expect("in range");
                worst = worst.max((linear - exact).abs());
            }
        }
        assert!(worst < 1e-13, "worst NCC residual {worst} nats");
        // And the same thing said through the fit, which also recovers the parameters.
        let fit = PairwiseFit::of(&t).expect("small");
        assert!(fit.residual < 1e-13, "residual {}", fit.residual);
        for k in 0..4 {
            assert!((fit.bias[k] - m.bias()[k]).abs() < 1e-13, "bias {k}");
            for j in 0..4 {
                assert!(
                    (fit.coupling[k * 4 + j] - m.coupling()[k * 4 + j]).abs() < 1e-13,
                    "coupling {k},{j}"
                );
            }
        }
    }

    /// ⭐ …and it FAILS where the theory says it must. A three-way interaction of 0.8 nats is
    /// planted in the target; no linear integrator can produce its conditionals, and the residual
    /// comes back as exactly 0.8. Without this test the one above proves only that the algebra was
    /// typed twice the same way.
    #[test]
    fn the_computability_condition_fails_for_a_third_order_target() {
        let m = coupled();
        let base = Target::of_boltzmann(&m).expect("small");
        let planted = 0.8;
        let log_w: Vec<f64> = (0..16u64)
            .map(|z| {
                let e = m.energy(z).expect("in range");
                if z & 0b0111 == 0b0111 { e + planted } else { e }
            })
            .collect();
        let t = Target::from_log_weights(4, &log_w).expect("valid");
        let fit = PairwiseFit::of(&t).expect("small");
        assert!(
            (fit.residual - planted).abs() < 1e-12,
            "residual {} should equal the planted {planted}",
            fit.residual
        );
        // The pairwise part is still recovered exactly — the third-order term vanishes at every
        // background with fewer than two other units on, which is where the fit reads it.
        let base_fit = PairwiseFit::of(&base).expect("small");
        for k in 0..4 {
            assert!((fit.bias[k] - base_fit.bias[k]).abs() < 1e-13, "bias {k}");
        }
    }

    /// ⭐ The residual is a **magnitude**: a target the fit over-predicts everywhere is exactly
    /// as far from pairwise as one it under-predicts by the same amount.
    ///
    /// Why the suite could not see this. Both planted-interaction fixtures in this module plant a
    /// **positive** third-order term — 0.8 nats above and 1.7 nats in
    /// `the_fitted_coupling_is_symmetric_because_of_where_the_fit_reads_it` — so at every
    /// configuration where the fit is wrong the target sits *above* the fitted value. The largest
    /// signed difference and the largest absolute difference are then the same number, and the
    /// `.abs()` in the residual sweep is doing nothing either fixture can observe. Here the term is
    /// planted with the opposite sign: every nonzero disagreement is negative, the largest signed
    /// difference over the whole sweep is exactly zero — the value at the empty configuration,
    /// where the fit reads the target and is exact by construction — and a residual that reported
    /// the signed maximum would call this target pairwise while it is 0.8 nats from being one.
    ///
    /// The fit itself is unmoved by the sign, and that is asserted too: the third-order term
    /// vanishes at every background with fewer than two other units on, which is where the bias and
    /// the couplings are read, so both targets get the identical pairwise part.
    #[test]
    fn a_target_the_fit_over_predicts_is_as_far_from_pairwise_as_one_it_under_predicts() {
        let m = coupled();
        let planted = 0.8;
        let signed_target = |sign: f64| {
            let log_w: Vec<f64> = (0..16u64)
                .map(|z| {
                    let e = m.energy(z).expect("in range");
                    if z & 0b0111 == 0b0111 { e + sign * planted } else { e }
                })
                .collect();
            Target::from_log_weights(4, &log_w).expect("valid")
        };
        let under = PairwiseFit::of(&signed_target(1.0)).expect("small");
        let over_target = signed_target(-1.0);
        let over = PairwiseFit::of(&over_target).expect("small");
        let (ru, ro) = (under.residual, over.residual);
        assert!((ru - planted).abs() < 1e-12, "a +{planted} nat term reads as a residual of {ru}");
        assert!((ro - planted).abs() < 1e-12, "a -{planted} nat term reads as a residual of {ro}");
        for k in 0..4 {
            let (bu, bo) = (under.bias[k], over.bias[k]);
            assert!((bu - bo).abs() < 1e-13, "bias {k}: {bu} under-predicting, {bo} over-predicting");
            for j in 0..4 {
                let (cu, co) = (under.coupling[k * 4 + j], over.coupling[k * 4 + j]);
                assert!((cu - co).abs() < 1e-13, "coupling {k},{j}: {cu} against {co}");
            }
        }

        // The quantity a sweep without the absolute value would report, spelled out over the same
        // configurations the fit sweeps. Its maximum is 0 and its minimum is the planted term, so
        // the two are not interchangeable on this target even though they are on the other one.
        let mut largest_signed = f64::NEG_INFINITY;
        let mut most_negative = f64::INFINITY;
        for z in 0..16u64 {
            for k in 0..4 {
                let mut fitted = over.bias[k];
                for j in 0..4 {
                    if j != k && z >> j & 1 == 1 {
                        fitted += over.coupling[k * 4 + j];
                    }
                }
                let d = over_target.conditional_log_odds(z, k).expect("in range") - fitted;
                largest_signed = largest_signed.max(d);
                most_negative = most_negative.min(d);
            }
        }
        assert!(
            largest_signed.abs() < 1e-12,
            "the signed sweep peaks at {largest_signed} nats, so it is not the zero the argument needs"
        );
        assert!(
            (most_negative + planted).abs() < 1e-12,
            "the planted term is not in the sweep: the worst signed disagreement is {most_negative}"
        );
    }

    /// The fitted coupling is symmetric, and it is symmetric for a reason that has nothing to do
    /// with the target being pairwise: `W_kj` and `W_jk` expand to the same four log-weights with
    /// the same signs, so their difference is identically zero for **any** target. A field
    /// reporting that difference would be a check no input could move — which is what
    /// `PairwiseFit::symmetry_defect` was, asserted at `< 1e-13` on a model where it could not be
    /// anything else.
    ///
    /// What can move is *where the fit reads the conditionals*. Asserted on a third-order target,
    /// because that is the case where a fit reading the pair at any background other than the
    /// empty one would come back asymmetric, and on the pairwise one, where symmetry also has to
    /// agree with the model's own matrix.
    #[test]
    fn the_fitted_coupling_is_symmetric_because_of_where_the_fit_reads_it() {
        let m = coupled();
        let pairwise = Target::of_boltzmann(&m).expect("small");
        let third: Vec<f64> = (0..16u64)
            .map(|z| m.energy(z).expect("in range") + if z & 0b1011 == 0b1011 { 1.7 } else { 0.0 })
            .collect();
        let third = Target::from_log_weights(4, &third).expect("valid");
        assert!(PairwiseFit::of(&third).expect("small").residual > 1.6, "the planted term is not there");

        for (name, t) in [("pairwise", &pairwise), ("third-order", &third)] {
            let fit = PairwiseFit::of(t).expect("small");
            let mut worst = 0.0f64;
            for k in 0..4 {
                for j in (k + 1)..4 {
                    worst = worst.max((fit.coupling[k * 4 + j] - fit.coupling[j * 4 + k]).abs());
                }
                assert!(fit.coupling[k * 4 + k] == 0.0, "{name}: self-coupling {k} is not zero");
            }
            assert!(worst < 1e-13, "{name}: the fitted coupling is asymmetric by {worst} nats");
        }
        // And on the pairwise target the symmetric matrix is the model's own.
        let fit = PairwiseFit::of(&pairwise).expect("small");
        for k in 0..4 {
            for j in 0..4 {
                assert!(
                    (fit.coupling[k * 4 + j] - m.coupling()[k * 4 + j]).abs() < 1e-13,
                    "coupling {k},{j}"
                );
            }
        }
    }

    #[test]
    fn a_target_refuses_a_wrong_length_a_nan_and_too_many_units() {
        assert!(matches!(
            Target::from_log_weights(3, &[0.0; 7]),
            Err(BayesError::ShapeMismatch { got: 7, want: 8, .. })
        ));
        assert!(matches!(
            Target::from_log_weights(2, &[0.0, f64::INFINITY, 0.0, 0.0]),
            Err(BayesError::NonFinite { index: 1, .. })
        ));
        assert!(matches!(
            Target::from_log_weights(21, &[]),
            Err(BayesError::TooLarge { got: 21, limit: 20, .. })
        ));
        let t = Target::from_log_weights(2, &[0.0; 4]).expect("valid");
        assert!(t.conditional_log_odds(0, 2).is_err());
        assert!(t.conditional_log_odds(4, 0).is_err());
        assert_eq!(t.units(), 2);
    }

    // ---------------------------------------------------------------------------------------
    // The sampler, against the exactly enumerated answer
    // ---------------------------------------------------------------------------------------

    /// ⭐⭐ THE CHECK. The chain's empirical distribution over all 16 configurations is compared
    /// to the exactly enumerated Boltzmann distribution by total-variation distance, across four
    /// refractory windows. Total variation bounds the error of **every** statement about the
    /// system at once, so there is no summary statistic a wrong sampler could match its way past.
    ///
    /// The threshold is 0.02 and the measurements come in between 0.0014 (`tau = 1`) and 0.0040
    /// (`tau = 20`). That gap is not slack, but it is not symmetric either: 0.02 sits a factor of
    /// 5 above the worst honest measurement and a factor of 18 below the 0.359 the companion
    /// mutation test lands at — about a decade above the defect's floor and half a decade above
    /// the sampler's.
    #[test]
    fn the_sampled_distribution_matches_the_exact_boltzmann_distribution() {
        let m = coupled();
        let exact = m.exact().expect("small");
        for (tau, seed) in [(1u32, 7u64), (2, 11), (5, 13), (20, 17)] {
            let mut s = NeuralSampler::new(m.clone(), tau, Scan::Random).expect("valid");
            let mut rng = Rng::new(seed);
            let h = s.run(&mut rng, 2_000_000, 200_000).expect("runs");
            let tv = h.total_variation(&exact).expect("same width");
            assert!(tv < 0.02, "tau = {tau}: total variation {tv} from the exact distribution");
            // And the marginals separately, so a failure says which unit moved.
            let want = m.exact_marginals().expect("small");
            let got = h.marginals().expect("has samples");
            for k in 0..4 {
                assert!((got[k] - want[k]).abs() < 0.01, "tau {tau} unit {k}: {} vs {}", got[k], want[k]);
            }
        }
    }

    /// ⭐ The anti-vacuity test for the one above. The `−ln tau` correction in the firing
    /// probability is dropped — the single most plausible thing to get wrong in this sampler,
    /// since without it the update is exactly Gibbs sampling and looks right — and the same
    /// comparison is run. The distance moves from 0.0038 to 0.359, a factor of 93, so the 0.02
    /// threshold is known to discriminate rather than assumed to.
    #[test]
    fn dropping_the_log_tau_correction_moves_the_distribution_a_hundredfold() {
        let m = coupled();
        let exact = m.exact().expect("small");
        let tau = 3u32;

        let mut good = NeuralSampler::new(m.clone(), tau, Scan::Random).expect("valid");
        let mut rng = Rng::new(7);
        let tv_good = good
            .run(&mut rng, 1_000_000, 100_000)
            .expect("runs")
            .total_variation(&exact)
            .expect("same width");

        // The mutant, written out by hand against the public API: identical in every respect but
        // the missing `- ln(tau)`.
        let mut rng = Rng::new(7);
        let mut zeta = [0u32; 4];
        let mut h = Histogram::new(4).expect("valid");
        for t in 0..1_000_000u64 {
            let k = rng.below(4) as usize;
            let mut z = 0u64;
            for (j, &c) in zeta.iter().enumerate() {
                if c >= 1 {
                    z |= 1u64 << j;
                }
            }
            if zeta[k] >= 2 {
                zeta[k] -= 1;
            } else {
                let q = logistic(m.membrane(z, k).expect("in range"));
                zeta[k] = if rng.next_f64() < q { tau } else { 0 };
            }
            if t >= 100_000 {
                let mut z = 0u64;
                for (j, &c) in zeta.iter().enumerate() {
                    if c >= 1 {
                        z |= 1u64 << j;
                    }
                }
                h.observe(z);
            }
        }
        let tv_bad = h.total_variation(&exact).expect("same width");

        assert!(tv_good < 0.02, "correct sampler {tv_good}");
        // Two-sided, because the doc prints 0.359 for this mutant and a one-sided `> 0.2` would
        // let that number drift anywhere above the floor without a test noticing.
        assert!(
            tv_bad > 0.34 && tv_bad < 0.38,
            "the mutant's distance moved off the documented 0.359: {tv_bad}"
        );
        assert!(
            tv_good > 0.003 && tv_good < 0.005,
            "the correct sampler moved off the documented 0.0038: {tv_good}"
        );
        assert!(
            tv_bad > 50.0 * tv_good,
            "the threshold does not discriminate: good {tv_good}, mutant {tv_bad}"
        );
    }

    /// Requirement (e): with every coupling zero the chain must reproduce the product of
    /// independent bits, which is an exact answer written in closed form rather than enumerated.
    /// This isolates the single-unit dynamics — the `−ln tau` derivation — from the interaction
    /// term entirely.
    #[test]
    fn zero_coupling_samples_the_exact_product_of_independent_bits() {
        let bias = [0.6, -0.4, 0.2, -1.1];
        let m = Boltzmann::independent(&bias).expect("valid");
        let product: Vec<f64> = (0..16u64)
            .map(|z| {
                let mut p = 1.0;
                for (k, &b) in bias.iter().enumerate() {
                    let q = logistic(b);
                    p *= if z >> k & 1 == 1 { q } else { 1.0 - q };
                }
                p
            })
            .collect();
        for (tau, seed) in [(1u32, 21u64), (4, 23)] {
            let mut s = NeuralSampler::new(m.clone(), tau, Scan::Random).expect("valid");
            let mut rng = Rng::new(seed);
            let h = s.run(&mut rng, 1_500_000, 150_000).expect("runs");
            let tv = h.total_variation(&product).expect("same width");
            assert!(tv < 0.02, "tau {tau}: {tv} from the exact product distribution");
            let marg = h.marginals().expect("has samples");
            for (k, &b) in bias.iter().enumerate() {
                assert!(
                    (marg[k] - logistic(b)).abs() < 0.006,
                    "tau {tau} unit {k}: {} vs logistic({b}) = {}",
                    marg[k],
                    logistic(b)
                );
            }
        }
    }

    /// The update order decides correctness, and this measures by how much. Parallel updates are
    /// exact with no coupling — there is nothing for simultaneous decisions to disagree about —
    /// and badly wrong with it, on the same model, the same budget and the same seed as the random
    /// scan it is compared against.
    #[test]
    fn parallel_updates_are_exact_without_coupling_and_wrong_with_it() {
        let steps = 1_000_000u64;
        let burn = 100_000u64;

        let free = Boltzmann::independent(&[0.6, -0.4, 0.2, -1.1]).expect("valid");
        let exact_free = free.exact().expect("small");
        let mut s = NeuralSampler::new(free, 3, Scan::Parallel).expect("valid");
        let tv_free = s
            .run(&mut Rng::new(31), steps, burn)
            .expect("runs")
            .total_variation(&exact_free)
            .expect("same width");
        assert!(tv_free < 0.02, "uncoupled parallel should be exact, got {tv_free}");

        let m = coupled();
        let exact = m.exact().expect("small");
        let mut par = NeuralSampler::new(m.clone(), 3, Scan::Parallel).expect("valid");
        let tv_par = par
            .run(&mut Rng::new(31), steps, burn)
            .expect("runs")
            .total_variation(&exact)
            .expect("same width");
        let mut ser = NeuralSampler::new(m, 3, Scan::Random).expect("valid");
        let tv_ser = ser
            .run(&mut Rng::new(31), steps, burn)
            .expect("runs")
            .total_variation(&exact)
            .expect("same width");

        assert!(tv_ser < 0.02, "random scan {tv_ser}");
        assert!(tv_par > 0.05, "coupled parallel should be visibly wrong, got {tv_par}");
        // Two-sided on both, because the two module docs print 0.0031 and 0.092 for exactly this
        // model, seed and budget.
        assert!(tv_par < 0.12, "parallel moved off the documented 0.092: {tv_par}");
        assert!(tv_ser > 0.002 && tv_ser < 0.004, "random scan moved off the documented 0.0031: {tv_ser}");
        assert!(tv_par > 20.0 * tv_ser, "serial {tv_ser}, parallel {tv_par}");
    }

    /// A fixed-window decode of the spike train is the state under parallel updates and is not
    /// under random scan. The first is checked bit for bit against the simulator's own trace; the
    /// second is refused, which is the behaviour, not an unimplemented path.
    #[test]
    fn a_fixed_window_decode_is_exact_for_parallel_and_refused_for_random_scan() {
        let m = coupled();
        for tau in [1u32, 4, 9] {
            let mut s = NeuralSampler::new(m.clone(), tau, Scan::Parallel).expect("valid");
            let rec = s.record(&mut Rng::new(41 + u64::from(tau)), 20_000).expect("runs");
            for k in 0..4 {
                let from_state = rec.indicator(k).expect("unit exists");
                let from_spikes = rec.reconstruct(k).expect("parallel, from reset");
                assert_eq!(from_state, from_spikes, "tau {tau} unit {k}");
                // Not vacuous: the unit must actually have been high some of the time and low
                // some of the time, or two all-zero vectors would compare equal.
                let high = from_state.iter().sum::<f64>();
                assert!(high > 100.0 && high < 19_900.0, "tau {tau} unit {k}: {high} high ticks");
            }
        }
        let mut s = NeuralSampler::new(m.clone(), 4, Scan::Random).expect("valid");
        let rec = s.record(&mut Rng::new(43), 5_000).expect("runs");
        assert!(rec.reconstruct(0).is_none(), "random scan must refuse a fixed-window decode");
        assert!(rec.indicator(0).is_some());
        assert!(rec.indicator(4).is_none());

        // …and refused again when the chain did not start from rest, because the spike that set
        // the counter is then outside the recording.
        let mut s = NeuralSampler::new(m, 6, Scan::Parallel).expect("valid");
        let mut rng = Rng::new(47);
        let mut fired = 0;
        while fired == 0 {
            fired = s.step(&mut rng);
        }
        let mid = s.record(&mut rng, 100).expect("runs");
        assert!(mid.reconstruct(0).is_none(), "a mid-refractory start must refuse");
    }

    /// The accessors, `reset`, `Recording::histogram` and the error messages — six public items
    /// that no other test in this module touches, and one of them carries a documented guarantee.
    ///
    /// `reset` is the one that matters: [`super::Recording::reconstruct`] refuses a recording that
    /// began mid-refractory, and `reset`'s doc says the way back is to return the sampler to
    /// quiescence. Nothing asserted that it does. A `reset` that forgot to clear the counters would
    /// leave every later recording undecodable, and the only symptom is a `None` that the caller
    /// has already been told to expect.
    #[test]
    fn reset_restores_the_decodable_state_and_the_accessors_report_the_model_they_were_given() {
        let m = coupled();
        let mut s = NeuralSampler::new(m.clone(), 5, Scan::Parallel).expect("valid");
        assert_eq!(s.tau(), 5);
        assert_eq!(s.scan(), Scan::Parallel);
        assert_eq!(s.model(), &m, "the sampler reported a different model from the one it was given");
        assert_eq!(s.state(), 0, "a fresh sampler is quiescent");

        // Step into the middle of a refractory period: a recording from here cannot be decoded.
        let mut rng = Rng::new(2_026);
        let mut fired = 0;
        while fired == 0 {
            fired = s.step(&mut rng);
        }
        assert_ne!(s.state(), 0, "the chain did not leave quiescence");
        let mid = s.record(&mut rng, 200).expect("runs");
        assert!(mid.reconstruct(0).is_none(), "a mid-refractory recording must refuse");

        // …and `reset` is what makes it decodable again, bit for bit against the trace.
        s.reset();
        assert_eq!(s.state(), 0, "reset left a unit high");
        let rec = s.record(&mut rng, 5_000).expect("runs");
        assert_eq!(rec.units(), 4);
        for k in 0..4 {
            let from_spikes = rec.reconstruct(k).expect("reset, parallel");
            assert_eq!(from_spikes, rec.indicator(k).expect("unit exists"), "unit {k}");
        }

        // `Recording::histogram` is the same tally as the trace, one observation per tick.
        let h = rec.histogram().expect("four units");
        assert_eq!(h.units(), 4);
        assert_eq!(h.samples(), rec.states().len() as u64);
        assert_eq!(h.counts().iter().sum::<u64>(), 5_000);
        let marg = h.marginals().expect("has samples");
        for k in 0..4 {
            let mean = rec.indicator(k).expect("unit").iter().sum::<f64>() / 5_000.0;
            assert!((marg[k] - mean).abs() < 1e-12, "unit {k}: histogram {} vs trace {mean}", marg[k]);
        }
    }

    /// The enumeration ceiling and the representation ceiling are different numbers, and both are
    /// public. A 21-unit model is legal — [`super::MAX_UNITS`] is 64 — and simply cannot be
    /// enumerated; a 65-unit one cannot be represented at all.
    #[test]
    fn the_enumeration_ceiling_and_the_representation_ceiling_are_enforced_separately() {
        let wide = Boltzmann::independent(&[0.1; 21]).expect("21 units is within MAX_UNITS");
        assert_eq!(wide.units(), 21);
        assert_eq!(wide.configurations(), None, "21 units must not offer an enumeration");
        assert!(matches!(
            wide.exact(),
            Err(BayesError::TooLarge { got: 21, limit: 20, .. })
        ));
        assert!(matches!(Target::of_boltzmann(&wide), Err(BayesError::TooLarge { .. })));
        assert!(wide.membrane(0b101, 20).is_some(), "the model itself still works at 21 units");

        let edge = Boltzmann::independent(&[0.1; 20]).expect("valid");
        assert_eq!(edge.configurations(), Some(1 << 20), "20 units is the last enumerable width");
        let tiny = Boltzmann::independent(&[0.1, -0.2, 0.3]).expect("valid");
        assert_eq!(tiny.configurations(), Some(8));
        assert_eq!(tiny.exact().expect("enumerable").len(), 8);

        let full = Boltzmann::independent(&[0.0; MAX_UNITS]).expect("64 units is the ceiling");
        assert_eq!(full.units(), MAX_UNITS);
        assert!(full.energy(u64::MAX).is_some(), "at 64 units every bit pattern is a state");
        assert_eq!(full.configurations(), None);
        assert!(matches!(
            Boltzmann::independent(&[0.0; MAX_UNITS + 1]),
            Err(BayesError::TooLarge { got: 65, limit: 64, .. })
        ));
    }

    /// Every refusal names the quantity and the number that made it refuse — which is the whole
    /// claim [`super::BayesError`]'s doc makes, and a `Display` that printed only the variant name
    /// would satisfy nothing else in this file.
    #[test]
    fn every_error_message_names_the_number_that_made_it_refuse() {
        let cases: [(BayesError, &[&str]); 6] = [
            (BayesError::NonFinite { what: "bias", index: 3 }, &["bias", "3"]),
            (BayesError::Asymmetric { i: 1, j: 2, w_ij: 0.5, w_ji: -0.25 }, &["1", "2", "0.5", "-0.25"]),
            (BayesError::SelfCoupling { unit: 7, value: 1.5 }, &["7", "1.5"]),
            (BayesError::NoVariation { n: 64, value: 2.5 }, &["64", "2.5"]),
            (BayesError::TooLarge { what: "units", got: 21, limit: 20 }, &["units", "21", "20"]),
            (BayesError::NotNormalised { mass: 1.75 }, &["1.75"]),
        ];
        for (e, needles) in cases {
            let text = e.to_string();
            for needle in needles {
                assert!(text.contains(needle), "{e:?} printed {text:?}, missing {needle:?}");
            }
        }
        // …and the real ones, from the call that produced them.
        let text = Boltzmann::new(&[0.0, 0.0], &[0.0, 1.0, 2.0, 0.0]).expect_err("asymmetric").to_string();
        assert!(text.contains('1') && text.contains('2'), "{text}");
        let text = estimate(&[2.5; 64]).expect_err("constant").to_string();
        assert!(text.contains("64") && text.contains("2.5"), "{text}");
    }

    #[test]
    fn a_sampler_refuses_a_zero_window_and_a_burn_in_that_eats_the_run() {
        let m = coupled();
        assert!(matches!(
            NeuralSampler::new(m.clone(), 0, Scan::Random),
            Err(BayesError::OutOfRange { what: "tau (ticks)", .. })
        ));
        assert!(NeuralSampler::new(m.clone(), NeuralSampler::MAX_TAU + 1, Scan::Random).is_err());
        let mut s = NeuralSampler::new(m, 2, Scan::Random).expect("valid");
        assert!(matches!(
            s.run(&mut Rng::new(1), 100, 100),
            Err(BayesError::TooShort { got: 100, want: 101 })
        ));
        assert!(s.record(&mut Rng::new(1), 0).is_err());
    }

    /// ⭐ `NeuralSampler::run` keeps the ticks *after* the burn-in and no others — both how many
    /// and which ones.
    ///
    /// Why the suite could not see this. Every other call to `run` in this module burns in about a
    /// tenth of a one-to-two-million-tick run, and then reads the histogram only through
    /// `Histogram::total_variation`, which divides by the sample count and so cannot see how many
    /// samples there are. A transient a ten-thousandth of the run long moves that distance by far
    /// less than its 0.02 threshold, so keeping the burn-in instead of discarding it passed every
    /// one of those tests unchanged. Nothing in this module read `samples()` off a `run` at all.
    ///
    /// Both halves are pinned. The kept window has exactly `steps - burn_in` observations; and it
    /// is the **tail**, reproduced here by stepping the burn-in by hand off the same seed and
    /// running the remainder with no burn-in, which must give bin-for-bin identical counts. The
    /// second half is what separates "discards the first `burn_in`" from "discards some `burn_in`
    /// of them".
    ///
    /// The window that is dropped is a different chain by construction and not by luck: the sampler
    /// starts from rest, a random scan can raise at most one unit per tick, so across the three
    /// discarded ticks at most `1 + 2 + 3 = 6` of the nine unit-tick slots can be high. The
    /// stationary occupancy of this model is `logistic(4) = 0.98201`, and the kept window measures
    /// it.
    #[test]
    fn a_run_keeps_the_ticks_after_the_burn_in_and_no_others() {
        let m = Boltzmann::independent(&[4.0, 4.0, 4.0]).expect("valid");
        let (steps, burn_in) = (1_003u64, 3u64);
        let occupied = |h: &Histogram| -> u64 {
            h.counts().iter().enumerate().map(|(z, &n)| n * u64::from((z as u64).count_ones())).sum()
        };

        let mut a = NeuralSampler::new(m.clone(), 1, Scan::Random).expect("valid");
        let mut ra = Rng::new(31);
        let kept = a.run(&mut ra, steps, burn_in).expect("runs");
        assert_eq!(
            kept.samples(),
            steps - burn_in,
            "a run of {steps} ticks with a burn-in of {burn_in} kept the wrong number of them"
        );
        assert_eq!(kept.counts().iter().sum::<u64>(), steps - burn_in, "the bins and the count disagree");

        let mut b = NeuralSampler::new(m.clone(), 1, Scan::Random).expect("valid");
        let mut rb = Rng::new(31);
        for _ in 0..burn_in {
            b.step(&mut rb);
        }
        let tail = b.run(&mut rb, steps - burn_in, 0).expect("runs");
        assert_eq!(kept.counts(), tail.counts(), "the window kept is not the tail of the run");

        let mut c = NeuralSampler::new(m, 1, Scan::Random).expect("valid");
        let mut rc = Rng::new(31);
        let dropped = c.run(&mut rc, burn_in, 0).expect("runs");
        let transient = occupied(&dropped);
        assert!(
            transient <= 6,
            "the three discarded ticks hold {transient} high unit-ticks; from rest at most 6 are reachable"
        );
        let after = occupied(&kept) as f64 / (3 * (steps - burn_in)) as f64;
        assert!(
            after > 0.95,
            "the kept window sits at an occupancy of {after}, nowhere near the stationary 0.98201"
        );
    }

    /// ⭐ A random-scan spike is stamped with the unit that fired, not with a constant.
    ///
    /// Why the suite could not see this. Nothing downstream reads a random-scan spike's *identity*.
    /// `Recording::reconstruct` refuses for `Scan::Random` by design, the histogram, every
    /// marginal and the total-variation check are all built from the state trace rather than from
    /// the train, and `the_chain_is_reproducible_from_its_seed` compares two trains that would
    /// carry the same mistake in both. So the shift that turns "the selected unit fired" into a
    /// bitmask was asserted nowhere, and reporting every random-scan spike as unit zero's changed
    /// no test in this module.
    ///
    /// What is pinned is an exact invariant rather than a statistic. Under a random scan exactly
    /// one unit updates per tick; a unit that spikes has its counter set to `tau >= 1`, so its
    /// state bit is high immediately after the tick; and the set of bits that went from low to high
    /// across the tick is therefore either empty — the unit was already high and re-fired — or
    /// exactly the spiking unit's, never any other unit's. The second case is the discriminating
    /// one, so the run is checked to contain enough of it to be a test at all.
    #[test]
    fn a_random_scan_spike_is_stamped_with_the_unit_that_fired_not_with_unit_zero() {
        let mut s = NeuralSampler::new(coupled(), 3, Scan::Random).expect("valid");
        let rec = s.record(&mut Rng::new(1_234), 20_000).expect("runs");
        let states = rec.states();
        let mut seen = [0u32; 4];
        let mut raised_a_bit = 0u32;
        for sp in rec.train().spikes() {
            let (t, k) = (sp.t as usize, sp.source as usize);
            assert!(k < 4, "the spike at tick {t} names unit {k} of a four-unit model");
            seen[k] += 1;
            assert!(
                states[t] >> k & 1 == 1,
                "unit {k} is recorded as spiking at tick {t} but is not high after it"
            );
            let previous = if t == 0 { 0 } else { states[t - 1] };
            let newly = states[t] & !previous;
            assert!(
                newly == 0 || newly == 1u64 << k,
                "tick {t} raised {newly:#06b} while the spike it recorded belongs to unit {k}"
            );
            if newly != 0 {
                raised_a_bit += 1;
            }
        }
        assert!(
            raised_a_bit > 500,
            "only {raised_a_bit} spikes took a unit from low to high; the invariant above is vacuous below that"
        );
        let total = rec.train().len();
        for (k, &c) in seen.iter().enumerate() {
            assert!(c > 0, "unit {k} is never named as a source across {total} spikes");
        }
    }

    /// Same seed, same spikes. The property the whole crate rests on, asserted for this module's
    /// chain because a sampler that drifted would make every measurement above unrepeatable.
    #[test]
    fn the_chain_is_reproducible_from_its_seed() {
        let m = coupled();
        let mut a = NeuralSampler::new(m.clone(), 3, Scan::Random).expect("valid");
        let mut b = NeuralSampler::new(m, 3, Scan::Random).expect("valid");
        let ra = a.record(&mut Rng::new(99), 5_000).expect("runs");
        let rb = b.record(&mut Rng::new(99), 5_000).expect("runs");
        assert_eq!(ra.states(), rb.states());
        assert_eq!(ra.train(), rb.train());
        assert!(ra.train().len() > 500, "only {} spikes, too few to be a check", ra.train().len());
    }

    #[test]
    fn a_histogram_refuses_a_reference_that_is_the_wrong_width_or_unnormalised() {
        let mut h = Histogram::new(2).expect("valid");
        assert!(h.probabilities().is_none());
        assert!(h.total_variation(&[0.25; 4]).is_err());
        h.observe(0);
        h.observe(3);
        assert_eq!(h.samples(), 2);
        h.observe(4); // out of range: dropped, not aliased onto state 0
        assert_eq!(h.samples(), 2, "an out-of-range observation was folded into a real bin");
        assert_eq!(h.counts()[0], 1);
        assert!(matches!(
            h.total_variation(&[0.25; 8]),
            Err(BayesError::ShapeMismatch { got: 8, want: 4, .. })
        ));
        assert!(matches!(
            h.total_variation(&[0.5, 0.5, 0.5, 0.5]),
            Err(BayesError::NotNormalised { .. })
        ));
        assert!((h.total_variation(&[0.5, 0.0, 0.0, 0.5]).expect("valid")).abs() < 1e-15);
        // A NONZERO case with an exact answer, because a distance that is only ever asserted to
        // be small would survive losing its factor of ½: the empirical distribution here is
        // (½, 0, 0, ½) against a uniform reference, so the distance is ½·(4 × ¼) = ½ exactly.
        let tv = h.total_variation(&[0.25; 4]).expect("valid");
        assert!((tv - 0.5).abs() < 1e-15, "total variation {tv} should be exactly 0.5");
        assert!(Histogram::new(0).is_err());
        assert!(Histogram::new(21).is_err());
    }

    // ---------------------------------------------------------------------------------------
    // Error bars: the closed form, and the failure the naive bar produces
    // ---------------------------------------------------------------------------------------

    /// ⭐ The integrated autocorrelation time of an `AR(1)` chain is `(1+φ)/(1−φ)` exactly. The
    /// estimator is checked against that at five coefficients spanning **both signs** — 0.0526 at
    /// `φ = −0.9`, 1 at `φ = 0`, 19 at `φ = 0.9`.
    ///
    /// The negative half is not decoration. Every `ρ_l` of a non-negative `AR(1)` is positive, so
    /// that is the one family in which Geyer's pairing `ρ_0+ρ_1, ρ_2+ρ_3, …` and the shifted
    /// pairing `ρ_1+ρ_2, ρ_3+ρ_4, …` **cannot** disagree about where to stop: a sweep over
    /// `φ ≥ 0` alone is a sweep pinned to the one family where the truncation rule cannot break.
    /// At `φ = −0.5` the shifted pairing's first term is `ρ_1 + ρ_2 = −0.5 + 0.25 = −0.25`, it
    /// truncates before adding anything, and it returns 1.0 where the closed form is 1/3.
    #[test]
    fn the_autocorrelation_time_of_an_ar1_chain_matches_its_closed_form() {
        for (phi, n, seed) in [
            (-0.9f64, 50_000usize, 400u64),
            (-0.5, 20_000, 500),
            (0.0, 20_000, 100),
            (0.5, 20_000, 200),
            (0.9, 20_000, 300),
        ] {
            let want = (1.0 + phi) / (1.0 - phi);
            let mut got = 0.0;
            let reps = 24u32;
            for r in 0..reps {
                let mut rng = Rng::new(seed + u64::from(r));
                let x = ar1(&mut rng, phi, n, 2_000);
                let i = integrated_autocorrelation_time(&x).expect("valid");
                assert!(!i.floored, "phi = {phi} rep {r}: value {} is the fallback, not a measurement", i.value);
                assert!(!i.truncated_at_cap, "phi = {phi} rep {r}: the lag window ran out at {}", i.lags);
                assert!(i.lags % 2 == 1, "phi = {phi} rep {r}: Geyer's pairs end on an odd lag, got {}", i.lags);
                got += i.value;
            }
            got /= f64::from(reps);
            let rel = (got - want).abs() / want;
            assert!(rel < 0.08, "phi = {phi}: iact {got}, closed form {want} ({:.1}% off)", rel * 100.0);
        }

        // The slowest coefficient the sweep this test replaces covered, at eight reps rather than
        // twenty-four because the estimator costs `O(n * lags)` and this chain's window runs past
        // lag 1000. The closed form is 199, and the initial-positive-sequence estimator's upward
        // bias is largest here — which is why the band is two-sided and not just an upper one.
        let mut got = 0.0;
        for r in 0..8u32 {
            let mut rng = Rng::new(700 + u64::from(r));
            let x = ar1(&mut rng, 0.99, 20_000, 2_000);
            let i = integrated_autocorrelation_time(&x).expect("valid");
            assert!(!i.floored && !i.truncated_at_cap, "phi = 0.99 rep {r}: {i:?}");
            assert!(i.lags > 100, "phi = 0.99 rep {r}: the sum stopped at lag {}", i.lags);
            got += i.value;
        }
        got /= 8.0;
        assert!(
            got > 170.0 && got < 240.0,
            "phi = 0.99: iact {got}, closed form 199"
        );
    }

    /// ⭐⭐ The failure Geyer's pairing exists to prevent, on a chain built to produce it and
    /// measured against a closed form written out as a literal.
    ///
    /// [`two_mode`] is a reversible chain with one fast anti-correlated mode and one slow positive
    /// one, `ρ_l = 0.95·(−0.5)^l + 0.05·(0.99)^l`, integrated time `TWO_MODE_TAU = 10.2667`. Its
    /// first *shifted* pair is `ρ_1 + ρ_2 = −0.4255 + 0.2865 = −0.1390`, negative — so the pairing
    /// this module used to implement truncates before adding a single term and reports exactly
    /// 1.0: *these samples are independent, the naive bar was fine*, for a chain worth ten ticks
    /// per independent draw, with `truncated_at_cap` false so there is no flag either. The
    /// reported standard error is then 3.2× too narrow and nothing says so.
    ///
    /// Geyer's first pair is `ρ_0 + ρ_1 = 1 − 0.4255 = 0.5745`, positive — as Theorem 3.1 says it
    /// must be for a reversible chain, `Γ_0 = 1 + ρ_1 ≥ 0` being unconditional.
    #[test]
    fn a_chain_with_a_fast_negative_mode_and_a_slow_positive_one_is_not_called_independent() {
        // The closed form, from the two coefficients, before any sample is drawn.
        let rho1: f64 = 0.95 * (-0.5) + 0.05 * 0.99;
        let rho2: f64 = 0.95 * 0.25 + 0.05 * 0.99 * 0.99;
        assert!((rho1 - (-0.4255)).abs() < 1e-12, "rho(1) = {rho1}");
        assert!((rho2 - 0.286_505).abs() < 1e-12, "rho(2) = {rho2}");
        assert!(rho1 + rho2 < -0.13, "the shifted pairing's first term is not negative: {}", rho1 + rho2);
        assert!(1.0 + rho1 > 0.57, "Geyer's first pair is not positive: {}", 1.0 + rho1);
        assert!(
            (TWO_MODE_TAU - (1.0 + 2.0 * (0.95 * (-1.0 / 3.0) + 0.05 * 99.0))).abs() < 1e-12,
            "the stated integrated time is not the one the two modes give"
        );

        let mut total = 0.0;
        let seeds = [1u64, 2, 3];
        for seed in seeds {
            let mut rng = Rng::new(seed);
            let x = two_mode(&mut rng, 200_000, 5_000);

            // The trap is live in THIS series, not only in the closed form: the first shifted pair
            // really is negative here, so an estimator paired that way would stop at once.
            let (r1, r2) = (sample_rho(&x, 1), sample_rho(&x, 2));
            assert!(r1 + r2 < 0.0, "seed {seed}: shifted first pair {} is not negative", r1 + r2);
            assert!(1.0 + r1 > 0.0, "seed {seed}: Geyer's first pair {} is not positive", 1.0 + r1);

            let i = integrated_autocorrelation_time(&x).expect("valid");
            assert!(!i.floored && !i.truncated_at_cap, "seed {seed}: {i:?}");
            assert!(
                i.value > 5.0,
                "seed {seed}: iact {} — a chain of integrated time {TWO_MODE_TAU} was called all but independent",
                i.value
            );
            assert!(i.lags > 100, "seed {seed}: the sum stopped at lag {}", i.lags);
            total += i.value;

            // …and the error bar the estimate carries is several times the naive one, which is the
            // whole consequence: the naive bar on this chain is sqrt(10.27) = 3.2 times too narrow.
            let e = estimate(&x).expect("valid");
            assert!(
                e.understatement().expect("nonzero") > 2.2,
                "seed {seed}: the naive bar was only {}x too narrow",
                e.understatement().expect("nonzero")
            );
        }
        let mean = total / seeds.len() as f64;
        let rel = (mean - TWO_MODE_TAU).abs() / TWO_MODE_TAU;
        assert!(rel < 0.2, "mean iact {mean}, closed form {TWO_MODE_TAU} ({:.1}% off)", rel * 100.0);
    }

    /// The effective sample size is smaller than the raw count for a correlated chain and about
    /// equal to it for an independent one. Both halves matter: an estimator that always returned
    /// `n/2` would pass the first assertion alone.
    #[test]
    fn the_effective_sample_size_is_below_the_raw_count_only_when_the_chain_is_correlated() {
        let mut rng = Rng::new(555);
        let indep = ar1(&mut rng, 0.0, 20_000, 100);
        let e = estimate(&indep).expect("valid");
        assert!(e.ess > 0.8 * e.n as f64, "independent chain lost samples: ess {} of {}", e.ess, e.n);
        // Two-sided, because `ess <= n` is no longer a construction guarantee: the estimator is not
        // floored at `iact = 1`, so an independent chain can land a little either side of it and a
        // one-sided assertion would no longer be measuring the estimator at all.
        assert!(e.ess < 1.25 * e.n as f64, "ess {} far exceeds n {}", e.ess, e.n);
        assert!((e.iact - 1.0).abs() < 0.1, "an independent chain should give iact ~ 1, got {}", e.iact);
        assert!(!e.floored, "an independent chain should not need the fallback");

        let corr = ar1(&mut rng, 0.9, 20_000, 2_000);
        let c = estimate(&corr).expect("valid");
        assert!(c.ess < 0.1 * c.n as f64, "correlated chain kept too much: ess {} of {}", c.ess, c.n);
        let under = c.understatement().expect("nonzero naive bar");
        // sem / naive_sem is sqrt(iact); at iact ≈ 19 that is ≈ 4.4.
        assert!((under - c.iact.sqrt()).abs() < 1e-12, "understatement {under} vs sqrt(iact)");
        assert!(under > 3.5, "the naive bar is only {under}x too narrow");
        assert!(c.sem > c.naive_sem);
        let (lo, hi) = c.ci95();
        let (nlo, nhi) = c.naive_ci95();
        assert!(hi - lo > nhi - nlo);
    }

    /// ⭐ The coverage failure, measured rather than asserted. 400 independent `AR(1)` chains of
    /// 2000 samples each, true mean zero. The naive 95% interval covers about a third of the time;
    /// the closed-form prediction is `2Φ(1.96/√19) − 1 = 2Φ(0.4497) − 1 = 0.347`, and the
    /// effective-sample-size interval recovers its nominal coverage.
    ///
    /// This is the test that makes [`Estimate::naive_sem`] worth carrying: an error bar that is
    /// 4.4× too narrow is not a rounding difference, it is a 95% claim that is wrong two times in
    /// three.
    #[test]
    fn an_error_bar_built_on_the_raw_count_covers_a_third_of_the_time_it_claims_ninety_five() {
        let phi = 0.9;
        let n = 2_000;
        let reps = 400u32;
        let (mut naive_hits, mut ess_hits) = (0u32, 0u32);
        for r in 0..reps {
            let mut rng = Rng::new(9_000 + u64::from(r));
            let x = ar1(&mut rng, phi, n, 500);
            let e = estimate(&x).expect("valid");
            if e.mean.abs() < 1.96 * e.naive_sem {
                naive_hits += 1;
            }
            if e.mean.abs() < e.half_width95() {
                ess_hits += 1;
            }
        }
        let naive = f64::from(naive_hits) / f64::from(reps);
        let ess = f64::from(ess_hits) / f64::from(reps);
        // The closed form for the naive interval's coverage at iact = 19.
        let predicted = 0.347;
        assert!(
            (naive - predicted).abs() < 0.06,
            "naive coverage {naive}, closed form predicts {predicted}"
        );
        assert!(ess > 0.88 && ess < 0.99, "ess-based coverage {ess}, nominal 0.95");
        assert!(ess - naive > 0.4, "the two bars barely differ: {naive} vs {ess}");
    }

    /// Closing the loop: the marginals of a real sampled chain, read with chain-aware error bars,
    /// bracket the exactly enumerated marginals — and the naive bars, on the same data, are
    /// several times too narrow to be trusted for it.
    #[test]
    fn a_marginal_read_off_a_sampled_chain_brackets_the_exact_answer() {
        let m = coupled();
        let want = m.exact_marginals().expect("small");
        let mut s = NeuralSampler::new(m, 3, Scan::Random).expect("valid");
        let mut rng = Rng::new(77);
        for _ in 0..50_000 {
            s.step(&mut rng);
        }
        let rec = s.record(&mut rng, 400_000).expect("runs");
        let mut worst_understatement = 0.0f64;
        for k in 0..4 {
            let e: Estimate = rec.marginal(k).expect("unit fires and rests");
            let (lo, hi) = e.ci95();
            assert!(
                lo <= want[k] && want[k] <= hi,
                "unit {k}: exact {} outside [{lo}, {hi}] (mean {}, sem {})",
                want[k],
                e.mean,
                e.sem
            );
            assert!(e.ess < e.n as f64, "unit {k} claimed {} independent samples of {}", e.ess, e.n);
            worst_understatement = worst_understatement.max(e.understatement().expect("nonzero"));
        }
        // The truncation flag on a series THIS MODULE'S OWN SAMPLER produced, not only on a
        // synthetic one: an eight-lag window on a chain whose correlation runs past lag 20 is
        // truncated, the bar built on it is narrower than the honest one, and the `Estimate`
        // carries the flag that says so.
        let x = rec.indicator(0).expect("unit exists");
        let tight = estimate_within(&x, 8).expect("valid");
        let honest = estimate(&x).expect("valid");
        assert!(tight.truncated_at_cap, "a real chain truncated at 8 lags without saying so");
        assert!(!honest.truncated_at_cap, "the default window should suffice here");
        assert!(tight.iact < honest.iact && tight.sem < honest.sem, "{tight:?} vs {honest:?}");

        assert!(
            worst_understatement > 2.0,
            "the naive bar was only {worst_understatement}x too narrow; this chain is barely correlated"
        );
    }

    /// The estimator's arithmetic on a series whose answer can be written down: `1..=8` has mean
    /// 4.5 and `Σ(x−m)² = 42`, so the `n−1` sample standard deviation is `√6 = 2.449…` and the
    /// `n` one would be `√5.25 = 2.291…`. Pinned because that choice is documented on
    /// [`super::Estimate::sd`] and is otherwise immaterial at every sample size this module's own
    /// tests use — a mutation audit found it to be the one change to this file that nothing else
    /// here could detect.
    #[test]
    fn the_mean_and_deviation_are_the_hand_computed_ones_with_the_stated_denominator() {
        let x: Vec<f64> = (1..=8).map(f64::from).collect();
        let e = estimate(&x).expect("valid");
        assert_eq!(e.n, 8);
        assert!((e.mean - 4.5).abs() < 1e-15, "mean {}", e.mean);
        assert!((e.sd - 6.0f64.sqrt()).abs() < 1e-14, "sd {} should be sqrt(6)", e.sd);
        assert!((e.naive_sem - 6.0f64.sqrt() / 8.0f64.sqrt()).abs() < 1e-14);
        assert!((e.sem - e.sd / e.ess.sqrt()).abs() < 1e-15);
        assert!((e.half_width95() - 1.96 * e.sem).abs() < 1e-15);
        let (lo, hi) = e.ci95();
        assert!((hi - lo - 2.0 * e.half_width95()).abs() < 1e-15);
        // The naive interval uses the SAME 1.96 normal quantile, pinned as a literal here because
        // every other use of `naive_ci95` in this module only ever compares its width against the
        // honest one — a comparison a narrower quantile would satisfy more easily, not less.
        let (nlo, nhi) = e.naive_ci95();
        assert!((nhi - nlo - 2.0 * 1.96 * e.naive_sem).abs() < 1e-15, "naive interval width");
        assert!((nlo - (e.mean - 1.96 * e.naive_sem)).abs() < 1e-15);
        assert!(
            ((nhi - nlo) / (hi - lo) - 1.0 / e.iact.sqrt()).abs() < 1e-14,
            "the two intervals must differ only by sqrt(iact)"
        );
    }

    /// ⭐ `Estimate::understatement` refuses a zero naive bar instead of dividing by it, and the
    /// guard is strict rather than merely non-negative.
    ///
    /// Why the suite could not see this. Every `Estimate` anywhere in this module arrives from
    /// `estimate`, and `estimate` refuses a constant series with `BayesError::NoVariation` before
    /// a zero `naive_sem` can reach the accessor — so the zero branch of the guard was never
    /// taken by any test, and loosening it from `> 0.0` to `>= 0.0` changed nothing any of them
    /// could observe. The struct's fields are public, which is the only route to the case, so the
    /// case is built here by hand.
    ///
    /// The three values are chosen to separate the two comparisons: at `naive_sem == 0.0` with a
    /// zero honest bar the loosened guard would return `Some(0.0 / 0.0)`, which is `Some(NaN)`; at
    /// `naive_sem == 0.0` with a nonzero one it would return `Some(+inf)`; and at the smallest
    /// positive `f64` the strict guard must still admit the ratio, which is `1.0` exactly because
    /// both fields hold the same value. A `NaN` bar is refused by both forms and is asserted so
    /// the refusal is not mistaken for a comparison this test moves.
    #[test]
    fn the_understatement_refuses_a_zero_naive_bar_rather_than_dividing_by_it() {
        let degenerate = Estimate {
            mean: 2.5,
            sd: 0.0,
            n: 64,
            iact: 1.0,
            ess: 64.0,
            sem: 0.0,
            naive_sem: 0.0,
            lags: 1,
            truncated_at_cap: false,
            floored: true,
        };
        assert_eq!(
            degenerate.understatement(),
            None,
            "a zero naive bar over a zero honest one is not a ratio, it is 0/0"
        );
        let infinite = Estimate { sem: 0.25, ..degenerate };
        assert_eq!(
            infinite.understatement(),
            None,
            "a zero naive bar under a 0.25 honest one would report an infinite understatement"
        );
        let smallest = Estimate { sem: f64::MIN_POSITIVE, naive_sem: f64::MIN_POSITIVE, ..degenerate };
        assert_eq!(
            smallest.understatement(),
            Some(1.0),
            "the guard is strict, so the smallest positive naive bar is still a bar"
        );
        assert_eq!(Estimate { naive_sem: f64::NAN, ..degenerate }.understatement(), None);
        // And the reason no measured `Estimate` reaches the zero case: the series that would make
        // one is refused a step earlier, with the count and the value that made it refuse.
        assert!(matches!(estimate(&[2.5; 64]), Err(BayesError::NoVariation { n: 64, value: 2.5 })));
    }

    #[test]
    fn the_estimator_refuses_a_short_series_a_constant_one_and_a_nan() {
        assert!(matches!(estimate(&[1.0; 4]), Err(BayesError::TooShort { got: 4, want: 8 })));
        assert!(matches!(
            estimate_within(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 0),
            Err(BayesError::OutOfRange { what: "max_lag", .. })
        ));
        assert!(matches!(
            estimate(&[2.5; 64]),
            Err(BayesError::NoVariation { n: 64, value: 2.5 })
        ));
        let mut x = vec![0.0; 64];
        x[9] = f64::NAN;
        assert!(matches!(estimate(&x), Err(BayesError::NonFinite { index: 9, .. })));
        // A unit that never fires has no error bar, and the refusal names why.
        let m = Boltzmann::independent(&[-40.0, 2.0]).expect("valid");
        let mut s = NeuralSampler::new(m, 1, Scan::Random).expect("valid");
        let rec = s.record(&mut Rng::new(3), 2_000).expect("runs");
        assert!(matches!(rec.marginal(0), Err(BayesError::NoVariation { .. })));
        assert!(rec.marginal(9).is_err());
    }

    /// ⭐ An anti-correlated chain has an integrated autocorrelation time **below one**, and this
    /// estimator reports it rather than a floor.
    ///
    /// The version of this test that shipped before asserted `iact < 1.2` for `φ < 0` and passed
    /// because the estimator truncated at once and returned exactly 1.0 for every one of them — it
    /// asserted the wrong answer was the right one, and the type's doc presented the 1.0 as a
    /// property rather than as early truncation. The closed forms are 0.0526 at `φ = −0.9`,
    /// 0.3333 at `φ = −0.5` and 0.6667 at `φ = −0.2`, and those are what is asserted now.
    ///
    /// The consequence carries through to [`super::Estimate`]: `ess > n` and `understatement < 1`,
    /// because the naive bar on an anti-correlated chain is too **wide**, not too narrow.
    #[test]
    fn an_anticorrelated_chain_reports_the_time_below_one_that_its_closed_form_gives() {
        for (phi, seed) in [(-0.9f64, 611u64), (-0.5, 612), (-0.2, 613)] {
            let want = (1.0 + phi) / (1.0 - phi);
            let reps = 8u32;
            let mut got = 0.0;
            for r in 0..reps {
                let mut rng = Rng::new(seed + u64::from(r));
                let x = ar1(&mut rng, phi, 50_000, 2_000);
                let i = integrated_autocorrelation_time(&x).expect("valid");
                assert!(i.value > 0.0 && i.value.is_finite(), "phi {phi}: iact {}", i.value);
                assert!(!i.floored, "phi {phi} rep {r}: floored to the fallback");
                assert!(i.value < 0.9, "phi {phi} rep {r}: iact {} is not below one", i.value);
                got += i.value;
            }
            got /= f64::from(reps);
            let rel = (got - want).abs() / want;
            assert!(rel < 0.15, "phi {phi}: iact {got}, closed form {want} ({:.1}% off)", rel * 100.0);
        }
        // What that means for the error bar, on one of those chains.
        let mut rng = Rng::new(614);
        let x = ar1(&mut rng, -0.5, 50_000, 2_000);
        let e = estimate(&x).expect("valid");
        assert!(e.iact < 0.9 && e.iact > 0.0, "iact {}", e.iact);
        assert!(e.ess > e.n as f64, "ess {} should exceed n {} for an anti-correlated chain", e.ess, e.n);
        assert!(e.sem < e.naive_sem, "the honest bar {} is not narrower than the naive {}", e.sem, e.naive_sem);
        let under = e.understatement().expect("nonzero");
        assert!(under < 1.0, "understatement {under} should be below 1 when the naive bar was too wide");
        assert!((under - e.iact.sqrt()).abs() < 1e-12, "understatement {under} vs sqrt(iact)");
    }

    /// ⭐ Geyer's window on a series whose every autocorrelation can be written down by hand, which
    /// pins three separate mechanisms at once: the **pairing**, the **`n/2` clamp** and the
    /// **floor**.
    ///
    /// For `x = [1,0,1,0,1,0,1,0]` the mean is ½, `c(0) = ¼`, and with the estimator's `1/n`
    /// normalisation `ρ_l = (−1)^l (8 − l)/8` exactly. So:
    ///
    /// - Geyer's pairs are `Γ_k = ρ_{2k} + ρ_{2k+1} = (8−2k)/8 − (7−2k)/8 = 1/8` — **all four of
    ///   them positive**, which is the theorem's guarantee and which the shifted pairing does not
    ///   get: `ρ_1 + ρ_2 = −7/8 + 6/8 = −1/8`, negative at the first term.
    /// - `cap = min(4096, n/2) = 4`, so only `Γ_0` and `Γ_1` fit and the window truncates at lag 3
    ///   with the flag set. Without the `n/2` clamp the loop would run to lag 7 and the flag would
    ///   come back false — which is what makes that clamp a live mechanism rather than dead code.
    /// - The truncated sum is `2 × (1/8 + 1/8) = 1/2`, so the raw value is `−1 + 1/2 = −1/2`. A
    ///   negative time would give a negative effective sample size and a `NaN` standard error, so
    ///   `1.0` is reported and [`super::Iact::floored`] says the number is the fallback.
    #[test]
    fn the_geyer_window_on_a_perfectly_alternating_series_is_hand_computable() {
        let x: Vec<f64> = (0..8).map(|i| f64::from(u8::from(i % 2 == 0))).collect();
        // The autocorrelations, against the hand-derived form.
        for lag in 0..8usize {
            let want = if lag % 2 == 0 { 1.0 } else { -1.0 } * (8.0 - lag as f64) / 8.0;
            assert!((sample_rho(&x, lag) - want).abs() < 1e-15, "rho({lag})");
        }
        assert!((sample_rho(&x, 1) + sample_rho(&x, 2) - (-0.125)).abs() < 1e-15);
        for k in 0..4usize {
            let g = sample_rho(&x, 2 * k) + sample_rho(&x, 2 * k + 1);
            assert!((g - 0.125).abs() < 1e-15, "Geyer pair {k} = {g}, not 1/8");
        }

        let i = integrated_autocorrelation_time(&x).expect("eight samples, varying");
        assert_eq!(i.lags, 3, "the n/2 clamp should have stopped the sum at lag 3, not {}", i.lags);
        assert!(i.truncated_at_cap, "the window ran out; the flag must say so");
        assert!(i.floored, "the raw sum is -1/2 and must be reported as floored");
        assert!((i.value - 1.0).abs() < 1e-15, "the fallback is 1.0, got {}", i.value);

        // The floor is what keeps the error bar real rather than NaN.
        let e = estimate(&x).expect("valid");
        assert!(e.floored && e.truncated_at_cap, "the flags must reach the estimate: {e:?}");
        assert!((e.iact - 1.0).abs() < 1e-15);
        assert!(e.sem.is_finite() && e.sem > 0.0, "sem {}", e.sem);
        assert!((e.sem - e.naive_sem).abs() < 1e-15, "a floored estimate is the naive bar exactly");
    }

    /// ⭐ A chain correlated past the lag window reports it, the number it reports is a LOWER
    /// bound — the whole content of [`super::Iact::truncated_at_cap`] — and **the flag reaches the
    /// type that carries the error bar**. It did not: `estimate` computed the `Iact`, kept its
    /// `value` and `lags`, and dropped the flag, so `Estimate::sem` was a lower bound with nothing
    /// on the struct saying so and no cap on the struct to compare `lags` against.
    ///
    /// Checked by giving the same `phi = 0.9` chain a window far too short and comparing against
    /// the full estimate: the short window returns a smaller `iact`, a larger `ess` and a narrower
    /// `sem`, and every one of those is flagged.
    ///
    /// Worth recording why the flag needs an explicit window to be reachable on a well-behaved
    /// chain: the sample autocovariances of a mean-centred series sum to exactly `-c(0)/2`, so they
    /// must turn negative somewhere, and for the chains anyone actually runs they do so long before
    /// `n/2`. A flag that only the default path could set would be a field that never fires.
    #[test]
    fn a_lag_window_too_short_for_the_chain_reports_a_lower_bound() {
        let mut rng = Rng::new(15);
        let x = ar1(&mut rng, 0.9, 20_000, 2_000);
        let full = integrated_autocorrelation_time(&x).expect("valid");
        // The default path really is `DEFAULT_MAX_LAG` and not some other number compiled in.
        assert_eq!(
            full,
            integrated_autocorrelation_time_within(&x, DEFAULT_MAX_LAG).expect("valid"),
            "the default estimator does not use DEFAULT_MAX_LAG"
        );
        assert!(!full.truncated_at_cap, "the default window should suffice at phi = 0.9");
        assert!(full.lags > 8, "the full sum only reached lag {}", full.lags);

        let short = integrated_autocorrelation_time_within(&x, 8).expect("valid");
        assert!(short.truncated_at_cap, "an 8-lag window must report that it ran out");
        assert!(short.lags <= 8);
        assert!(
            short.value < full.value,
            "the truncated estimate {} is not below the full one {}",
            short.value,
            full.value
        );
        // …and the truncated number is exactly the partial sum it claims to be, `1 + 2 Σ_{l=1}^{7} ρ_l`,
        // rather than merely "smaller". A lag dropped from either end of the window moves this.
        let partial: f64 = (1..=7).map(|l| sample_rho(&x, l)).sum();
        assert!(
            (short.value - (1.0 + 2.0 * partial)).abs() < 1e-12,
            "an 8-lag window gave {} where lags 1..=7 sum to {}",
            short.value,
            1.0 + 2.0 * partial
        );
        assert_eq!(short.lags, 7, "an 8-lag cap admits Geyer pairs up to lag 7");
        assert!(matches!(
            integrated_autocorrelation_time_within(&x, 0),
            Err(BayesError::OutOfRange { what: "max_lag", .. })
        ));

        // The flag on the ESTIMATE, which is the object a caller reports from.
        let e_short = estimate_within(&x, 8).expect("valid");
        let e_full = estimate(&x).expect("valid");
        assert!(e_short.truncated_at_cap, "the estimate dropped the truncation flag");
        assert!(!e_full.truncated_at_cap, "the full estimate should not claim truncation");
        assert!(!e_short.floored && !e_full.floored);
        assert_eq!(e_short.lags, short.lags);
        assert!((e_short.iact - short.value).abs() < 1e-15);
        assert!(
            e_short.sem < e_full.sem,
            "the truncated bar {} is not narrower than the honest one {} — nothing to flag",
            e_short.sem,
            e_full.sem
        );
        assert!(e_short.ess > e_full.ess);

        // `max_lag == 1` is refused's neighbour and is NOT inert: it admits Geyer's first pair
        // `Γ_0 = 1 + ρ_1` and nothing else, so the answer is exactly `1 + 2ρ_1`.
        let one = integrated_autocorrelation_time_within(&x, 1).expect("valid");
        assert_eq!(one.lags, 1, "max_lag = 1 admitted no pair at all");
        assert!(one.truncated_at_cap);
        let want = 1.0 + 2.0 * sample_rho(&x, 1);
        assert!((one.value - want).abs() < 1e-12, "max_lag = 1 gave {} not 1 + 2 rho(1) = {want}", one.value);
        assert!(one.value > 2.7 && one.value < 2.8, "1 + 2 rho(1) = {} on this chain", one.value);
    }

    // ---------------------------------------------------------------------------------------
    // Bayesian confidence from population activity
    // ---------------------------------------------------------------------------------------

    /// The readout against Bayes' rule computed by hand on a two-neuron population, including the
    /// count-independent offset a wrong implementation drops.
    #[test]
    fn the_population_log_odds_is_the_hand_computed_bayes_rule() {
        let counts = [3u64, 11];
        let a = [20.0, 50.0];
        let b = [35.0, 30.0];
        let t = 0.2;
        let prior = 0.4;
        let want = prior
            + 3.0 * (20.0f64 / 35.0).ln()
            - t * (20.0 - 35.0)
            + 11.0 * (50.0f64 / 30.0).ln()
            - t * (50.0 - 30.0);
        let got = population_log_odds(&counts, &a, &b, t, prior).expect("valid");
        assert!((got - want).abs() < 1e-13, "{got} vs {want}");

        let c = Confidence::from_log_odds(got);
        assert!((c.probability_a - logistic(want)).abs() < 1e-15);
        assert_eq!(c.favours, if want >= 0.0 { Hypothesis::A } else { Hypothesis::B });
        // The documented tie-break, which the assertion above can never reach because `want` is
        // never zero: an exact tie favours `A`, arbitrarily and documentedly, and a `>` in place of
        // the `>=` would move it to `B` with nothing to notice.
        let tie = Confidence::from_log_odds(0.0);
        assert_eq!(tie.favours, Hypothesis::A, "an exact tie must favour A");
        assert!((tie.probability_a - 0.5).abs() < 1e-16);
        assert_eq!(Confidence::from_log_odds(-0.0).favours, Hypothesis::A, "negative zero is a tie");
        assert_eq!(Confidence::from_log_odds(-1e-300).favours, Hypothesis::B);
        // Identical hypotheses carry no evidence whatever the counts, which is the offset term
        // doing its job: the two ln-ratios are zero and the two offsets cancel.
        assert!(population_log_odds(&counts, &a, &a, t, 0.0).expect("valid").abs() < 1e-13);
    }

    /// ⭐ The expected log-likelihood ratio under the hypothesis that generated the data is the
    /// Kullback-Leibler divergence between the two likelihoods, `Σ [a ln(a/b) + b − a]` with
    /// `a = λ^A T`. Averaging the readout over simulated trials must converge to it — and it is the
    /// offset term, the one that is easy to omit, that the check binds: dropping it shifts every
    /// trial's log-odds by `T Σ(λ^A − λ^B)` = 3.0 nats here, which is 6σ of the mean over this
    /// many trials.
    #[test]
    fn the_mean_log_odds_under_a_hypothesis_is_the_divergence_between_the_likelihoods() {
        // All three rates higher under A, so the asymmetry of the divergence does not cancel
        // across the population and the count-independent offset is a full 12 nats per trial.
        let a = [40.0, 60.0, 20.0];
        let b = [20.0, 30.0, 10.0];
        let t = 0.2;
        let trials = 40_000;
        let mut rng = Rng::new(808);
        let (mut sa, mut sb) = (0.0, 0.0);
        for _ in 0..trials {
            let na = sample_population(&mut rng, &a, t).expect("valid");
            sa += population_log_odds(&na, &a, &b, t, 0.0).expect("valid");
            let nb = sample_population(&mut rng, &b, t).expect("valid");
            sb += population_log_odds(&nb, &a, &b, t, 0.0).expect("valid");
        }
        let mean_a = sa / f64::from(trials);
        let mean_b = sb / f64::from(trials);
        let kl_ab = poisson_kl(&a, &b, t).expect("valid");
        let kl_ba = poisson_kl(&b, &a, t).expect("valid");
        assert!(kl_ab > 0.5 && kl_ba > 0.5, "the two hypotheses are too close to test: {kl_ab}, {kl_ba}");
        // 0.08 is about five standard errors of the mean at this trial count; the offset term a
        // wrong implementation drops is worth 12 nats, which is 150 times wider.
        assert!((mean_a - kl_ab).abs() < 0.08, "under A: mean {mean_a}, KL {kl_ab}");
        assert!((mean_b + kl_ba).abs() < 0.08, "under B: mean {mean_b}, -KL {}", -kl_ba);
        let offset = t * (a.iter().sum::<f64>() - b.iter().sum::<f64>());
        assert!(offset > 10.0, "the offset this test is meant to bind is only {offset} nats");
        // The divergence is asymmetric, so this is not a check that both sides are the same number.
        assert!((kl_ab - kl_ba).abs() > 0.1, "KL happened to be symmetric here; pick worse rates");
        // A hypothesis against itself carries no information.
        assert!(poisson_kl(&a, &a, t).expect("valid").abs() < 1e-13);
    }

    /// Calibration, which is the property a confidence readout is actually for. With the truth
    /// drawn from the prior, the average posterior must equal the prior — the law of total
    /// probability, exact — and within each confidence band the empirical frequency must match the
    /// band. A decoder that dropped the offset term would pass neither.
    #[test]
    fn the_posterior_is_calibrated_against_the_frequency_it_reports() {
        let a = [20.0, 30.0, 12.0];
        let b = [26.0, 24.0, 15.0];
        let t = 0.12;
        let trials = 40_000;
        let mut rng = Rng::new(909);
        let mut sum_post = 0.0;
        let mut bins = [(0u32, 0u32); 5]; // (trials, times A was true)
        for _ in 0..trials {
            let truth_a = rng.next_f64() < 0.5;
            let counts =
                sample_population(&mut rng, if truth_a { &a } else { &b }, t).expect("valid");
            let lo = population_log_odds(&counts, &a, &b, t, 0.0).expect("valid");
            let p = Confidence::from_log_odds(lo).probability_a;
            sum_post += p;
            let bin = ((p * 5.0) as usize).min(4);
            bins[bin].0 += 1;
            if truth_a {
                bins[bin].1 += 1;
            }
        }
        let mean_post = sum_post / f64::from(trials);
        assert!((mean_post - 0.5).abs() < 0.015, "mean posterior {mean_post}, prior 0.5");
        let mut used = 0;
        for (i, &(n, hits)) in bins.iter().enumerate() {
            if n < 500 {
                continue;
            }
            used += 1;
            let freq = f64::from(hits) / f64::from(n);
            let centre = (i as f64 + 0.5) / 5.0;
            assert!(
                (freq - centre).abs() < 0.08,
                "band {i}: reported ~{centre}, observed {freq} over {n} trials"
            );
        }
        assert!(used >= 4, "only {used} confidence bands were populated; the test saw no spread");
    }

    #[test]
    fn a_zero_rate_is_refused_rather_than_making_one_spike_infinitely_convincing() {
        let counts = [1u64, 0];
        let a = [10.0, 10.0];
        let zero = [0.0, 10.0];
        assert!(matches!(
            population_log_odds(&counts, &a, &zero, 0.1, 0.0),
            Err(BayesError::OutOfRange { what: "rate_b", .. })
        ));
        assert!(matches!(
            population_log_odds(&counts, &[-1.0, 10.0], &a, 0.1, 0.0),
            Err(BayesError::OutOfRange { what: "rate_a", .. })
        ));
        assert!(matches!(
            population_log_odds(&counts, &a, &a, 0.0, 0.0),
            Err(BayesError::OutOfRange { what: "window_s", .. })
        ));
        assert!(matches!(
            population_log_odds(&[1, 2, 3], &a, &a, 0.1, 0.0),
            Err(BayesError::ShapeMismatch { what: "counts", got: 3, want: 2 })
        ));
        assert!(matches!(
            population_log_odds(&counts, &a, &[1.0], 0.1, 0.0),
            Err(BayesError::ShapeMismatch { what: "rate_b", .. })
        ));
        assert!(population_log_odds(&counts, &a, &a, 0.1, f64::NAN).is_err());
        assert!(population_log_odds(&[], &[], &[], 0.1, 0.0).is_err());
        assert!(poisson_kl(&a, &[1.0], 0.1).is_err());
        assert!(sample_population(&mut Rng::new(1), &a, -1.0).is_err());
    }

    /// ⭐ A non-finite firing rate is refused as **non-finite**, which is what the `# Errors`
    /// section of `population_log_odds` promises — and an infinite one is refused at all.
    ///
    /// Why the suite could not see this. The zero-rate refusal test above passes a zero rate and a
    /// negative one, and both land in the neighbouring `OutOfRange` arm; no test in this module
    /// passed a `NaN` or an infinity as a rate at all. Deleting the finiteness check is then
    /// invisible for a `NaN`, because `!(NaN > 0.0)` is true and the positivity arm catches it
    /// under a different name — the call still fails, so a test that only asked `is_err()` would
    /// still pass either way. It is **not** invisible for `+inf`: `!(inf > 0.0)` is false, so an
    /// infinite rate sails through both arms and the call returns `Ok` with a log-odds of `NaN`,
    /// which is a confidence readout handing back a number that compares false against everything.
    ///
    /// So the variant and the index are asserted rather than merely the failure, over all three
    /// non-finite values and over every slice that shares the gate: `rate_a` and `rate_b` of
    /// `population_log_odds`, `rate_p` and `rate_q` of `poisson_kl`, and `rates` of
    /// `sample_population`.
    #[test]
    fn a_non_finite_rate_is_refused_as_non_finite_rather_than_as_out_of_range() {
        let ok = [10.0, 10.0];
        let counts = [1u64, 0];
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let second = [10.0, bad];
            let first = [bad, 10.0];
            assert!(
                matches!(
                    population_log_odds(&counts, &second, &ok, 0.1, 0.0),
                    Err(BayesError::NonFinite { what: "rate_a", index: 1 })
                ),
                "a rate of {bad} in rate_a[1] is not reported as non-finite"
            );
            assert!(
                matches!(
                    population_log_odds(&counts, &ok, &first, 0.1, 0.0),
                    Err(BayesError::NonFinite { what: "rate_b", index: 0 })
                ),
                "a rate of {bad} in rate_b[0] is not reported as non-finite"
            );
            assert!(
                matches!(
                    poisson_kl(&second, &ok, 0.1),
                    Err(BayesError::NonFinite { what: "rate_p", index: 1 })
                ),
                "a rate of {bad} in rate_p[1] is not reported as non-finite"
            );
            assert!(
                matches!(
                    poisson_kl(&ok, &first, 0.1),
                    Err(BayesError::NonFinite { what: "rate_q", index: 0 })
                ),
                "a rate of {bad} in rate_q[0] is not reported as non-finite"
            );
            assert!(
                matches!(
                    sample_population(&mut Rng::new(1), &second, 0.1),
                    Err(BayesError::NonFinite { what: "rates", index: 1 })
                ),
                "a rate of {bad} in rates[1] is not reported as non-finite"
            );
        }
        // The message names the neuron, which is the whole reason the index is carried.
        let text = population_log_odds(&counts, &[10.0, f64::INFINITY], &ok, 0.1, 0.0)
            .expect_err("an infinite rate")
            .to_string();
        assert!(text.contains("rate_a[1]"), "{text}");
    }

    /// `logistic` must not return exactly zero for a large negative argument: the difference
    /// between "impossible" and "very unlikely" is what a log-likelihood sum is made of.
    #[test]
    fn the_logistic_survives_both_tails() {
        assert!((logistic(0.0) - 0.5).abs() < 1e-16);
        assert!(logistic(-700.0) > 0.0, "underflowed to zero");
        assert!(logistic(-700.0) < 1e-300, "the tail is the wrong size");
        // The two tails are NOT symmetric in `f64`, and that asymmetry is the reason for the
        // branch. `1 - 1e-304` is not representable, so `logistic(700.0)` is exactly 1.0 and
        // nothing can be done about it; `1e-304` IS representable, so the small-probability side
        // keeps its resolution — which is the side a log-likelihood sum reads.
        assert!((logistic(700.0) - 1.0).abs() < f64::EPSILON);
        assert!((logistic(800.0) - 1.0).abs() < 1e-15);
        // The naive single-branch form would return 0.0 here; this one does not.
        assert!(logistic(-745.0) > 0.0 && 1.0 / (1.0 + 745.0f64.exp()) == 0.0);
        for x in [-30.0f64, -3.0, 0.7, 25.0] {
            assert!((logistic(x) + logistic(-x) - 1.0).abs() < 1e-15, "symmetry at {x}");
        }
    }

    // ---------------------------------------------------------------------------------------
    // Hyperdimensional computing
    // ---------------------------------------------------------------------------------------

    /// Requirement (c), first half: binding is invertible **exactly**, with no tolerance anywhere.
    /// Checked over many random pairs at a dimension that is not a multiple of 64, because the
    /// tail-bit masking is exactly where an off-by-one would hide.
    #[test]
    fn binding_is_exactly_invertible_and_distance_preserving() {
        let mut rng = Rng::new(1234);
        for dim in [1usize, 63, 64, 65, 1000, KANERVA_DIM] {
            for _ in 0..8 {
                let a = Hypervector::random(&mut rng, dim).expect("valid");
                let b = Hypervector::random(&mut rng, dim).expect("valid");
                let c = Hypervector::random(&mut rng, dim).expect("valid");
                assert_eq!(a.bind(&b).expect("same dim").unbind(&b).expect("same dim"), a);
                assert_eq!(a.bind(&b).expect("s"), b.bind(&a).expect("s"), "bind must commute");
                // XOR preserves Hamming distance under a common key, which is what lets a whole
                // structure be bound and unbound with its internal similarities intact.
                let da = a.hamming(&b).expect("s");
                let db = a.bind(&c).expect("s").hamming(&b.bind(&c).expect("s")).expect("s");
                assert_eq!(da, db, "dim {dim}: binding moved the distance");
                // The tail bits stay zero, structurally rather than statistically: padding that
                // survived would be counted as signal by every Hamming distance at any dimension
                // that is not a multiple of 64, and would show up only as a small similarity bias.
                let rem = dim % 64;
                if rem != 0 {
                    for v in [&a, &b, &c, &a.bind(&b).expect("s"), &a.permute(3)] {
                        let last = v.words()[v.words().len() - 1];
                        assert_eq!(last >> rem, 0, "dim {dim}: padding bits survived");
                    }
                }
                assert!(a.ones() <= dim && b.ones() <= dim);
            }
        }
    }

    /// A bound pair is a new symbol, not a blend: it sits at the random-similarity noise floor
    /// from both of its operands. Measured against `random_similarity_sd`, not against a
    /// hand-picked tolerance.
    #[test]
    fn a_bound_pair_is_near_orthogonal_to_both_operands() {
        let dim = KANERVA_DIM;
        let sd = random_similarity_sd(dim).expect("positive");
        let mut rng = Rng::new(4321);
        for _ in 0..32 {
            let a = Hypervector::random(&mut rng, dim).expect("valid");
            let b = Hypervector::random(&mut rng, dim).expect("valid");
            let ab = a.bind(&b).expect("s");
            assert!(ab.similarity(&a).expect("s").abs() < 5.0 * sd);
            assert!(ab.similarity(&b).expect("s").abs() < 5.0 * sd);
        }
    }

    /// Permutation is invertible and order-sensitive: `ρ(a) + ρ²(b)` is a sequence where `a + b` is
    /// only a set. Checked at a dimension that is not a multiple of 64 so the wrap is exercised.
    #[test]
    fn permutation_is_invertible_and_destroys_similarity() {
        let dim = 1_001;
        let sd = random_similarity_sd(dim).expect("positive");
        let mut rng = Rng::new(5_551);
        for _ in 0..16 {
            let a = Hypervector::random(&mut rng, dim).expect("valid");
            for s in [1i64, -1, 7, 500, dim as i64] {
                assert_eq!(a.permute(s).permute(-s), a, "shift {s} did not round-trip");
            }
            assert_eq!(a.permute(0), a);
            assert_eq!(a.permute(dim as i64), a, "a full turn is the identity");
            // The DIRECTION, pinned. A sign flip inside `permute` survives every round-trip test
            // ever written, because both halves of the round trip flip together; only a claim
            // about where a single bit lands can catch it.
            let p = a.permute(1);
            for i in 0..dim {
                assert_eq!(p.get((i + 1) % dim), a.get(i), "bit {i} moved the wrong way");
            }
            assert!(a.permute(1).similarity(&a).expect("s").abs() < 6.0 * sd);
            assert_eq!(a.permute(3).ones(), a.ones(), "permutation must preserve the population");
        }
    }

    /// ⭐ Requirement (c), second half, and (d)'s foundation: the bundle's similarity to its own
    /// components is compared against the **exact binomial closed form**, not against a previous
    /// run. Five bundle sizes, with the margin over an unrelated vector measured in units of the
    /// noise floor rather than asserted loosely.
    #[test]
    fn the_bundle_similarity_matches_the_binomial_closed_form() {
        let dim = 8_192;
        let sd = random_similarity_sd(dim).expect("positive");
        let mut rng = Rng::new(2468);
        for k in [2usize, 3, 5, 9, 25] {
            let want = bundle_similarity(k).expect("small k");
            let (mut acc, mut n) = (0.0, 0u32);
            let mut worst_margin = f64::INFINITY;
            for _ in 0..24 {
                let parts: Vec<Hypervector> =
                    (0..k).map(|_| Hypervector::random(&mut rng, dim).expect("valid")).collect();
                let bundle = Hypervector::bundle(&parts, &mut rng).expect("valid");
                let outsider = Hypervector::random(&mut rng, dim).expect("valid");
                let out_sim = bundle.similarity(&outsider).expect("s");
                for p in &parts {
                    let s = p.similarity(&bundle).expect("s");
                    acc += s;
                    n += 1;
                    worst_margin = worst_margin.min((s - out_sim) / sd);
                }
            }
            let got = acc / f64::from(n);
            // 1e-3, not the 5e-3 that shipped: the worst disagreement over these five bundle sizes
            // at this dimension and seed is 6.01e-4 (at k = 25), so a 5e-3 bound left a factor of
            // eight of unclaimed room and the doc claimed 5e-4, which is below what is measured.
            assert!((got - want).abs() < 1e-3, "k = {k}: measured {got}, closed form {want}");
            // Requirement (c): components stay closer than random, with the margin measured.
            assert!(
                worst_margin > 6.0,
                "k = {k}: the worst component sat only {worst_margin} sigma above an outsider"
            );
        }
    }

    /// The closed form's own identities, checked exactly. `sim(1) = 1` (a bundle of one is that
    /// one), and `sim(2j) = sim(2j+1)` for every `j` — adding an even member to a majority buys
    /// nothing, because `C(2m, m) = 2·C(2m−1, m)`. The second is a fact about the formula that no
    /// simulation would ever reveal, and it exercises both parities of the tie-handling branch.
    #[test]
    fn an_even_bundle_is_worth_exactly_the_odd_one_below_it() {
        // The first seven values, as literals from the binomial rather than read back off the
        // function. For odd `k = 2m+1` the disagreeing votes carry the bit exactly when more than
        // half of `Bin(2m, ½)` do, so `sim(2m+1) = C(2m, m) / 2^{2m}`: 1, 1/2, 3/8, 5/16.
        let want = [1.0, 0.5, 0.5, 0.375, 0.375, 0.3125, 0.3125];
        for (i, &w) in want.iter().enumerate() {
            let got = bundle_similarity(i + 1).expect("small k");
            assert!((got - w).abs() < 1e-15, "sim({}) = {got}, closed form {w}", i + 1);
        }
        assert!((bundle_similarity(7).expect("k=7") - 20.0 / 64.0).abs() < 1e-15, "C(6,3)/2^6");
        assert!(
            (bundle_similarity(25).expect("k=25") - 2_704_156.0 / 16_777_216.0).abs() < 1e-15,
            "sim(25) should be C(24,12)/2^24"
        );
        for j in 1..200usize {
            let even = bundle_similarity(2 * j).expect("valid");
            let odd = bundle_similarity(2 * j + 1).expect("valid");
            assert!((even - odd).abs() < 1e-12, "j = {j}: sim({}) = {even}, sim({}) = {odd}", 2 * j, 2 * j + 1);
        }
        // Monotone decreasing over the odd sizes, and asymptotic to sqrt(2/(pi k)).
        let mut last = f64::INFINITY;
        for k in (1..400usize).step_by(2) {
            let s = bundle_similarity(k).expect("valid");
            assert!(s < last, "k = {k} did not decrease");
            last = s;
        }
        // The approach to `sqrt(2/(pi k))`, from ABOVE and at the rates the doc prints. Two-sided
        // per `k`, because a one-sided "within 1%" band starting at k = 101 is what let the doc
        // claim 0.3% at k = 25 when the true figure is 1.0047% — the k the claim was about was the
        // one k the test did not visit.
        for (k, lo, hi) in [
            (25usize, 0.0100f64, 0.0101f64),
            (101, 0.00247, 0.00249),
            (1_001, 0.000249, 0.000251),
            (10_001, 0.0000249, 0.0000251),
        ] {
            let s = bundle_similarity(k).expect("valid");
            let asym = (2.0 / (core::f64::consts::PI * k as f64)).sqrt();
            let excess = s / asym - 1.0;
            assert!(
                excess > lo && excess < hi,
                "k = {k}: {s} is {:.6}% above the asymptote {asym}, expected {:.4}%..{:.4}%",
                excess * 100.0,
                lo * 100.0,
                hi * 100.0
            );
        }
        assert!(bundle_similarity(0).is_none());
        assert!(bundle_similarity((1 << 16) + 1).is_none());
    }

    /// ⭐ Requirement (d): recovery degrades where the capacity bound says it does. At 1024 bits
    /// against a 512-item cleanup memory, `k = 5` scores 12 standard deviations and recovers
    /// everything, `k = 25` scores 5.2 and recovers 0.983, `k = 125` scores 2.3 and recovers 0.797.
    /// The assertions are tied to the `z`-score, so this is a check of the bound rather than a
    /// recording of three numbers.
    ///
    /// **Each rep draws its own codebook.** The version that shipped drew one codebook outside the
    /// loop and bundled `cb.vectors()[..k]` twelve times; all three `k` are odd, so
    /// [`super::Hypervector::bundle`] never reaches a tie, never touches the stream, and is a pure
    /// function of a slice that does not change — the twelve reps were twelve copies of one
    /// measurement, and the three headline numbers were each a single draw averaged with itself.
    /// `distinct` below is asserted directly so that regression cannot come back silently.
    #[test]
    fn recovery_from_a_bundle_degrades_where_the_capacity_bound_says_it_does() {
        let dim = 1_024;
        let entries = 512;
        let mut results = Vec::new();
        for k in [5usize, 25, 125] {
            let z = bundle_z_score(dim, k).expect("valid");
            let reps = 12;
            let mut rng = Rng::new(31_337);
            let mut per_rep: Vec<f64> = Vec::with_capacity(reps);
            for _ in 0..reps {
                let cb = Codebook::random(&mut rng, dim, entries).expect("valid");
                let parts: Vec<Hypervector> = cb.vectors()[..k].to_vec();
                let bundle = Hypervector::bundle(&parts, &mut rng).expect("valid");
                let ranked = cb.rank(&bundle).expect("non-empty");
                let hits = ranked[..k].iter().filter(|(i, _)| *i < k).count();
                per_rep.push(hits as f64 / k as f64);
            }
            let recall = per_rep.iter().sum::<f64>() / per_rep.len() as f64;
            let distinct = {
                let mut v = per_rep.clone();
                v.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
                v.dedup();
                v.len()
            };
            // Below the clear regime the reps MUST disagree with one another: a `distinct` of 1
            // there is the frozen-fixture bug, whatever the mean happens to be.
            if z < 8.0 {
                assert!(
                    distinct > 1,
                    "k = {k}: all {reps} reps returned {recall} — the reps are copies of one draw"
                );
            }
            results.push((k, z, recall));
        }
        // The signal has to beat the LOUDEST of the distractors, not a typical one. The maximum
        // of `M` standard normals concentrates at `sqrt(2 ln M)` — 3.53 for a 512-item memory — so
        // that, not zero, is the floor a bundle's z-score is measured against.
        let floor = (2.0 * (entries as f64).ln()).sqrt();
        assert!((floor - 3.53).abs() < 0.02, "the extreme-value floor moved: {floor}");
        let (mut clear, mut marginal, mut lost) = (0, 0, 0);
        for &(k, z, recall) in &results {
            if z > floor + 5.0 {
                clear += 1;
                assert!(recall > 0.99, "k = {k} scores {z} sigma but recalled only {recall}");
            } else if z > floor + 1.0 {
                marginal += 1;
                assert!(recall > 0.80, "k = {k} scores {z} sigma but recalled only {recall}");
                // The point of the marginal band: it does NOT recover everything. `recall` is
                // `hits/k` with `hits <= k`, so the `recall < 1.0 + 1e-12` that shipped here was
                // unfailable by construction and this is the claim it was reaching for.
                assert!(
                    recall < 1.0,
                    "k = {k} at {z} sigma recovered everything; the marginal band is not marginal"
                );
            } else if z < floor - 1.0 {
                lost += 1;
                assert!(recall < 0.95, "k = {k} scores only {z} sigma yet recalled {recall}");
            }
        }
        assert!(
            clear == 1 && marginal == 1 && lost == 1,
            "the three regimes were not all exercised: {clear}/{marginal}/{lost} from {results:?}"
        );
        // Monotone in k, and the spread is real rather than three equal numbers.
        assert!(results[0].2 >= results[1].2 && results[1].2 > results[2].2, "{results:?}");
        assert!(results[0].2 - results[2].2 > 0.1, "recovery barely moved: {results:?}");
        // Raising the dimension raises the capacity at the same k, which is the bound's content.
        let wide = bundle_z_score(4 * dim, 125).expect("valid");
        assert!((wide / results[2].1 - 2.0).abs() < 1e-12, "z must scale as sqrt(dim)");
    }

    /// Kanerva's own figures for `dim = 10 000`, which the 2009 paper prints: two random vectors
    /// are 5000 bits apart with a standard deviation of 50 bits, so the normalised similarity is
    /// `0 ± 0.01`. This is the yardstick every other similarity in the section is measured in.
    #[test]
    fn random_hypervectors_sit_half_a_dimension_apart_with_the_deviation_kanerva_prints() {
        let dim = KANERVA_DIM;
        let mut rng = Rng::new(10_000);
        let pairs = 600;
        let (mut s1, mut s2) = (0.0f64, 0.0f64);
        let mut worst_bits = 0usize;
        for _ in 0..pairs {
            let a = Hypervector::random(&mut rng, dim).expect("valid");
            let b = Hypervector::random(&mut rng, dim).expect("valid");
            let h = a.hamming(&b).expect("s");
            worst_bits = worst_bits.max(h.abs_diff(dim / 2));
            let s = a.similarity(&b).expect("s");
            s1 += s;
            s2 += s * s;
        }
        let mean = s1 / f64::from(pairs);
        let sd = (s2 / f64::from(pairs) - mean * mean).sqrt();
        let want_sd = random_similarity_sd(dim).expect("positive");
        assert!((want_sd - 0.01).abs() < 1e-12, "the closed form should be 0.01 at 10 000 bits");
        assert!(mean.abs() < 4.0 * want_sd / (f64::from(pairs)).sqrt(), "mean similarity {mean}");
        assert!((sd - want_sd).abs() < 0.1 * want_sd, "measured sd {sd}, closed form {want_sd}");
        // Kanerva's five-sigma statement: 50 bits is one sigma, so 600 pairs should all land
        // inside 250 bits of 5000.
        assert!(worst_bits < 250, "a pair was {worst_bits} bits from orthogonal");
    }

    /// A role-filler record, which is what the algebra is for: bind each role to its filler, bundle
    /// the pairs, then unbind by a role and clean up. This exercises bind, bundle and the cleanup
    /// memory in the one composition they exist to support.
    #[test]
    fn a_bundled_record_gives_its_fillers_back_through_the_cleanup_memory() {
        let dim = 4_096;
        let mut rng = Rng::new(60_606);
        let roles = Codebook::random(&mut rng, dim, 4).expect("valid");
        let fillers = Codebook::random(&mut rng, dim, 64).expect("valid");
        let assignment = [7usize, 19, 40, 61];
        let pairs: Vec<Hypervector> = (0..4)
            .map(|i| roles.vectors()[i].bind(&fillers.vectors()[assignment[i]]).expect("s"))
            .collect();
        let record = Hypervector::bundle(&pairs, &mut rng).expect("valid");
        let threshold = 4.0 * random_similarity_sd(dim).expect("positive");
        for i in 0..4 {
            let noisy = record.unbind(&roles.vectors()[i]).expect("s");
            let (found, sim) = fillers.cleanup(&noisy, threshold).expect("valid").expect("above floor");
            assert_eq!(found, assignment[i], "role {i} cleaned up to the wrong filler");
            assert!(sim > threshold);
        }
        // A role that was never in the record retrieves nothing above the floor, which is the
        // answer a cleanup memory has to be able to give.
        let stranger = Hypervector::random(&mut rng, dim).expect("valid");
        let noise = record.unbind(&stranger).expect("s");
        assert!(
            fillers.cleanup(&noise, threshold).expect("valid").is_none(),
            "an absent role retrieved a filler"
        );
    }

    #[test]
    fn hypervector_operations_refuse_a_mismatch_an_empty_bundle_and_an_impossible_dimension() {
        let mut rng = Rng::new(77_777);
        let a = Hypervector::random(&mut rng, 128).expect("valid");
        let b = Hypervector::random(&mut rng, 256).expect("valid");
        assert!(matches!(
            a.bind(&b),
            Err(BayesError::DimensionMismatch { a: 128, b: 256 })
        ));
        assert!(a.similarity(&b).is_err());
        assert!(a.hamming(&b).is_err());
        assert!(matches!(Hypervector::zeros(0), Err(BayesError::Empty { .. })));
        assert!(matches!(
            Hypervector::random(&mut rng, (1 << 24) + 1),
            Err(BayesError::TooLarge { .. })
        ));
        assert!(matches!(
            Hypervector::bundle(&[], &mut rng),
            Err(BayesError::Empty { what: "bundle components" })
        ));
        assert!(Hypervector::bundle(&[a.clone(), b.clone()], &mut rng).is_err());
        let mut z = Hypervector::zeros(10).expect("valid");
        assert!(z.set(10, true).is_err());
        assert!(z.get(10).is_none());
        z.set(3, true).expect("in range");
        assert_eq!(z.get(3), Some(true));
        assert_eq!(z.ones(), 1);
        z.set(3, false).expect("in range");
        assert_eq!(z.ones(), 0);
        let mut cb = Codebook::new(128).expect("valid");
        assert!(cb.is_empty());
        assert!(cb.rank(&a).is_err());
        assert!(cb.add("wrong", b).is_err());
        cb.add("right", a.clone()).expect("same dim");
        assert_eq!(cb.len(), 1);
        assert_eq!(cb.name(0), Some("right"));
        assert!(cb.name(1).is_none());
        assert!((cb.nearest(&a).expect("valid").1 - 1.0).abs() < 1e-15);
        assert!(cb.cleanup(&a, f64::NAN).is_err());
        assert!(Codebook::random(&mut rng, 64, 0).is_err());
    }

    /// The two documented edges of the cleanup memory, neither of which any other test reaches:
    /// [`super::Codebook::rank`] breaks ties **by index**, and [`super::Codebook::cleanup`] accepts
    /// a similarity **equal** to the threshold.
    ///
    /// A genuine tie needs two entries that are equal, which no other test in this module builds —
    /// every codebook here is independent random vectors, and at 128 bits two of those tie with
    /// probability `2^-128`. Reversing the tie-break comparator, or turning the `>=` into a `>`,
    /// survives every other assertion in the file.
    #[test]
    fn the_cleanup_memory_breaks_ties_by_index_and_accepts_the_threshold_exactly() {
        let mut rng = Rng::new(9_191);
        let dim = 128;
        let v = Hypervector::random(&mut rng, dim).expect("valid");
        let w = Hypervector::random(&mut rng, dim).expect("valid");
        let mut cb = Codebook::new(dim).expect("valid");
        cb.add("first", w.clone()).expect("same dim");
        cb.add("tie-a", v.clone()).expect("same dim");
        cb.add("tie-b", v.clone()).expect("same dim");
        cb.add("last", w).expect("same dim");

        let ranked = cb.rank(&v).expect("non-empty");
        assert!((ranked[0].1 - 1.0).abs() < 1e-15 && (ranked[1].1 - 1.0).abs() < 1e-15, "{ranked:?}");
        assert!((ranked[0].1 - ranked[1].1).abs() < 1e-15, "the top two are not a genuine tie");
        assert_eq!(ranked[0].0, 1, "the tie must resolve to the LOWER index");
        assert_eq!(ranked[1].0, 2);
        assert_eq!(cb.nearest(&v).expect("non-empty").0, 1);
        // The rest of the ranking is still sorted, so the tie-break did not disturb the order.
        assert!(ranked[2].1 <= ranked[1].1 && ranked[3].1 <= ranked[2].1, "{ranked:?}");
        assert_eq!(ranked.len(), 4);

        // `cleanup` keeps a similarity EQUAL to the threshold. The probe is an entry of the
        // codebook, so the best similarity is exactly 1.0 and the boundary is hit exactly rather
        // than approached: `>` in place of `>=` turns the answer into `None`.
        let (i, s) = cb.nearest(&v).expect("non-empty");
        assert!((s - 1.0).abs() < 1e-15, "the boundary case needs an exact similarity, got {s}");
        assert_eq!(cb.cleanup(&v, s).expect("valid"), Some((i, s)), "s >= threshold must accept s == threshold");
        let next = s - 2.0 / dim as f64;
        assert!(cb.cleanup(&v, next).expect("valid").is_some());
        assert!(
            cb.cleanup(&v, s + 1e-12).expect("valid").is_none(),
            "a threshold above the best similarity must return None"
        );
    }

    /// ⭐⭐ The tie-break coin is **fair**, so an even bundle is not pulled toward all-ones.
    ///
    /// Why the suite could not see this. Every measurement this module makes on a bundle is a
    /// *similarity to a component*, and that quantity does not move with the coin's bias at all.
    /// At a tied bit exactly half of the components carry a 1, so the chosen component's own bit is
    /// a 1 at half the tied positions whatever the coin does; the bundle agrees with it at
    /// probability one half either way, and `bundle_similarity`, the closed-form check and the
    /// capacity sweep are all unmoved. `an_odd_bundle_consumes_no_randomness_and_an_even_one_does`
    /// only asserts that the stream was touched and that two seeds give different answers, both of
    /// which a nine-to-one coin still does. Nothing looked at the bundle's own **density**, which
    /// is the one thing the bias moves.
    ///
    /// The fixture makes every bit a tie and nothing else: a vector and its exact complement
    /// disagree at all `dim` positions, so the count at every bit is exactly one of two and the
    /// bundle is `dim` coin flips with no majority anywhere. That the fixture really is degenerate
    /// is asserted rather than assumed — the Hamming distance between the pair must be `dim`.
    ///
    /// The bound is arithmetic. Pooled over eight seeds the test watches 80 000 tied bits, and the
    /// standard deviation of a fair proportion over that many is `sqrt(0.25 / 80_000) = 0.001768`,
    /// so the asserted 0.01 is 5.7 of them. This implementation measures 0.50057, which is 0.32 of
    /// one; a coin weighted nine to one would measure about 0.9, which is 226 of them.
    #[test]
    fn the_tie_break_coin_is_fair_so_an_even_bundle_is_not_pulled_toward_all_ones() {
        let dim = 10_000;
        let (mut ones, mut tied_bits) = (0usize, 0usize);
        for seed in 1..=8u64 {
            let mut rng = Rng::new(seed);
            let v = Hypervector::random(&mut rng, dim).expect("valid");
            let mut complement = Hypervector::zeros(dim).expect("valid");
            for i in 0..dim {
                complement.set(i, !v.get(i).expect("in range")).expect("in range");
            }
            assert_eq!(
                v.hamming(&complement).expect("aligned"),
                dim,
                "seed {seed}: the pair is not a complement, so not every bit is a tie"
            );
            let bundled = Hypervector::bundle(&[v, complement], &mut rng).expect("valid");
            ones += bundled.ones();
            tied_bits += dim;
        }
        let p = ones as f64 / tied_bits as f64;
        assert!(
            (p - 0.5).abs() < 0.01,
            "the tie-break coin set {p} of {tied_bits} tied bits; a fair coin sets about half"
        );
    }

    /// Bundling an odd number of vectors draws no randomness and is therefore a pure function of
    /// its inputs; bundling an even number does draw, and two different coins give two different
    /// answers. Both halves are asserted, because "deterministic" and "seeded" are different
    /// claims and only one of them is true here.
    #[test]
    fn an_odd_bundle_consumes_no_randomness_and_an_even_one_does() {
        let dim = 512;
        let mut rng = Rng::new(4_242);
        let parts: Vec<Hypervector> =
            (0..5).map(|_| Hypervector::random(&mut rng, dim).expect("valid")).collect();
        let mut r1 = Rng::new(1);
        let mut r2 = Rng::new(2);
        assert_eq!(
            Hypervector::bundle(&parts, &mut r1).expect("valid"),
            Hypervector::bundle(&parts, &mut r2).expect("valid"),
            "an odd bundle must not depend on the stream"
        );
        assert_eq!(r1, Rng::new(1), "an odd bundle drew from the stream");

        let even: Vec<Hypervector> = parts[..4].to_vec();
        let mut r3 = Rng::new(1);
        let mut r4 = Rng::new(2);
        let b3 = Hypervector::bundle(&even, &mut r3).expect("valid");
        let b4 = Hypervector::bundle(&even, &mut r4).expect("valid");
        assert_ne!(r3, Rng::new(1), "an even bundle must break its ties from the stream");
        assert_ne!(b3, b4, "two coins gave the identical tie-break on 512 bits");
        // Same seed, same bundle.
        assert_eq!(b3, Hypervector::bundle(&even, &mut Rng::new(1)).expect("valid"));
    }

    /// The fit writes each coupling into the ROW of the unit whose conditional it read, and the
    /// slot is not a relabelling: the transposed write stores a different number.
    ///
    /// ⛔ THE HOLE: `W_kj` and `W_jk` are the same four log-weights with the same signs, so the
    /// record argued a transposed write differs "by at most a couple of ulps". That is an ulp of
    /// the LARGEST log-weight, and `Target::from_log_weights` gates length, unit count and
    /// finiteness and nothing else — it does not bound the magnitude. Every fixture in this module
    /// carries log-weights of order 1, where that ulp is 1e-16 and the tightest assertion on
    /// `coupling` is 1e-13, so nothing pinned which slot the value lands in.
    #[test]
    fn the_fit_writes_each_coupling_into_the_row_of_the_unit_whose_conditional_it_read() {
        // Ordinary magnitudes, and every quantity below is exact in binary64.
        let e = 2f64.powi(-53);
        let t = Target::from_log_weights(2, &[0.0, e, 3.0 * e, 1.0 + 2.0 * e]).expect("valid");
        let fit = PairwiseFit::of(&t).expect("small");
        // Equality, not a tolerance: the right-hand sides are the same two calls the fit makes, in
        // the same order, so anything but the identical f64 means the value landed elsewhere.
        assert_eq!(
            fit.coupling[1],
            t.conditional_log_odds(0b10, 0).expect("in range") - fit.bias[0],
            "row 0, column 1 does not hold the coupling read at unit 0"
        );
        assert_eq!(
            fit.coupling[2],
            t.conditional_log_odds(0b01, 1).expect("in range") - fit.bias[1],
            "row 1, column 0 does not hold the coupling read at unit 1"
        );
        // Measured: the two slots hold 1 − 2^-52 = 0.9999999999999998 and 1 − 3·2^-53 =
        // 0.9999999999999997, which differ by 2^-53 = 1.1102230246251565e-16. They differ because
        // (1 + 2^-52) − 2^-53 is a tie that rounds to 1.0 while (1 + 2^-52) − 3·2^-53 is exact:
        // the four log-weights are associated in a different order, so the sum is a different f64.
        assert_eq!(fit.coupling[1], 1.0 - 2.0 * e, "row 0, column 1");
        assert_eq!(fit.coupling[2], 1.0 - 3.0 * e, "row 1, column 0");
        // And the gap scales with the log-weights, so 1e-16 is a property of the fixtures and not
        // of the arithmetic. At 2^53 + 2 — finite, so the constructor accepts it — the same two
        // slots are measured a whole nat apart.
        let big = Target::from_log_weights(2, &[0.0, 1.0, 3.0, 9007199254740994.0]).expect("valid");
        let big = PairwiseFit::of(&big).expect("small");
        assert_eq!(big.coupling[1], 9007199254740990.0, "row 0, column 1 at magnitude 2^53");
        assert_eq!(big.coupling[2], 9007199254740989.0, "row 1, column 0 at magnitude 2^53");
    }

    /// The fit SKIPS the diagonal rather than writing a zero into it: at `j == k` the value it
    /// would write is `x − x`, which is `+0.0` only while `x` is finite.
    ///
    /// ⛔ THE HOLE: `Target`'s invariant is that every LOG-WEIGHT is finite, not that a DIFFERENCE
    /// of two of them is. `conditional_log_odds` returns `log_w[on] − log_w[off]`, which overflows
    /// to `±inf` for two finite weights of opposite sign near `f64::MAX`; the bias is such a
    /// difference, and `inf − inf` is NaN. Every fixture in this module holds log-weights of order
    /// 1, where the subtraction is exact and a filled-in diagonal writes back the `+0.0` that
    /// `vec![0.0; n * n]` already put there — which is why
    /// `the_fitted_coupling_is_symmetric_because_of_where_the_fit_reads_it` makes this very
    /// assertion and stays green. `coupling` is a `pub` field, so the entry is an observable
    /// whether or not the residual loop reads it.
    #[test]
    fn the_fit_leaves_the_diagonal_of_the_coupling_alone_rather_than_writing_a_difference_in() {
        // Both log-weights are finite, so the constructor accepts; their difference is not.
        let t = Target::from_log_weights(1, &[-1e308, 1e308]).expect("valid");
        let fit = PairwiseFit::of(&t).expect("small");
        // Measured: bias[0] = 1e308 − (−1e308) = 2e308, past f64::MAX, so +inf. The diagonal a
        // filled-in write would store is conditional_log_odds(1, 0) − bias[0] = inf − inf = NaN.
        assert_eq!(fit.bias[0], f64::INFINITY, "this fixture no longer overflows the bias");
        let diagonal = fit.coupling[0];
        assert_eq!(diagonal, 0.0, "the fit wrote a self-coupling of {diagonal} nats");
        assert!(fit.coupling.iter().all(|c| c.is_finite()), "the fitted coupling is not finite");
    }
}
