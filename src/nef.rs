//! The Neural Engineering Framework: representing a vector in a population of spiking neurons,
//! computing a function of it with a weight matrix, and closing the loop into a dynamical system —
//! with the gain, the bias, the decoder and the dynamics each checked against its closed form.
//!
//! # What the framework is
//!
//! Eliasmith and Anderson, *Neural Engineering: Computation, Representation, and Dynamics in
//! Neurobiological Systems*, MIT Press, 2003, state three principles, and every one of them is a
//! formula this module implements literally:
//!
//! 1. **Representation.** A population of `N` neurons represents a vector `x ∈ R^d`. Neuron `i` has
//!    a unit *encoder* `e_i`, a *gain* `α_i` and a *bias* `J_i^bias`; its input current is
//!    `J_i(x) = α_i · e_i·x / r + J_i^bias` (with `r` the radius of the represented ball), and its
//!    firing rate is `a_i(x) = G(J_i(x))` for the neuron's rate curve `G`. For a leaky
//!    integrate-and-fire cell with membrane constant `τ_rc` and refractory period `τ_ref`,
//!    normalised so that the threshold current is `1`,
//!    `G(J) = 1 / (τ_ref + τ_rc · ln(1 + 1/(J − 1)))` for `J > 1` and `0` otherwise. The vector is
//!    read back by *decoders* `d_i`: `x̂ = Σ_i a_i(x) · d_i`, least squares over sample points.
//! 2. **Transformation.** To compute `f(x)`, solve the same least squares against `f` instead of
//!    the identity. A connection from one population to another is then the outer product
//!    `W_ji = α_j · e_j · T · d_i / r_j` — the full weight matrix is a rank-`d` factorisation, which
//!    is the reason the framework maps onto neuromorphic hardware at all: `N_pre × N_post`
//!    synapses become `N_pre × d + d × N_post`, and [`connection_traffic`] says when that saves
//!    energy and when it does not.
//! 3. **Dynamics.** A linear system `ẋ = A x + B u` implemented through a first-order synapse of
//!    time constant `τ` needs the recurrent connection to compute `A' = τ A + I` and the input
//!    connection `B' = τ B`. That substitution is [`dynamics_transform`]; the integrator and the
//!    oscillator in the tests are built from it and compared with the ideal system they stand in
//!    for.
//!
//! Stewart, *A technical overview of the Neural Engineering Framework*, University of Waterloo
//! technical report, 2012, is the short derivation; Bekolay et al., *Nengo: a Python tool for
//! building large-scale functional brain models*, Frontiers in Neuroinformatics 7:48, 2014, is the
//! software the field runs, and its gain-and-bias solution is the one reproduced here:
//! `x = 1 / (1 − exp((τ_ref − 1/rate_max) / τ_rc))`, `α = (x − 1) / (1 − intercept)`,
//! `J^bias = 1 − α · intercept`, so that the tuning curve is exactly zero at the intercept and
//! exactly `rate_max` at the edge of the radius.
//!
//! # Why it is in this crate
//!
//! The NEF is how a spiking network is *engineered* rather than trained: the weights are solved,
//! not learned, and every population has a stated representational error. It is the compiler
//! behind the largest functional spiking models built (Spaun) and behind Nengo's deployments to
//! `Loihi`, `SpiNNaker` and FPGAs. What it charges for is the question this crate keeps asking, so
//! the module carries the answer in its own units: a factorised connection moves
//! `spikes · d + d · N_post` weights a tick against a full matrix's `spikes · N_post`, and the two
//! cross at a spike count of `d · N_post / (N_post − d)` — below `d` spikes a tick the factorisation
//! is *more* traffic, not less.
//!
//! # The closed forms this module is checked against
//!
//! - `G(J)` equals `1 / Lif::isi(I)` for this crate's own [`crate::neuron::Lif`] with
//!   `v_reset = v_rest` and `I = J · (v_th − v_rest) / r_m`, to 1e-12 across the current range.
//! - The gain-and-bias solution puts the tuning curve at exactly zero one step below the intercept
//!   and at `rate_max` to 1e-9 at `x = r`.
//! - The full weight matrix equals the factorised product to rounding, for a random transform.
//! - `dynamics_transform` against literals.
//! - A spiking population's measured rates match `G(J)` to the interval quantisation, and the
//!   decoded value from spike counts matches the rate-mode decode.
//! - Decoding error falls with `N`, monotonically, across an eightfold sweep.
//! - The integrator holds a value and the oscillator holds its frequency, against the ideal system.
//! - **A spiking recurrent network** ([`SpikingLoop`]) running the Legendre Memory Unit's matrices
//!   tracks the ideal unit's state and reproduces its delayed output, and tracks it BETTER with
//!   more neurons — the spiking LMU, which is how the LMU was first built (Voelker and Eliasmith,
//!   *Improving spiking dynamical networks: accurate delays, higher-order synapses, and time
//!   cells*, Neural Computation 30(3):569–609, 2018). The errors are measured, not derived: with
//!   1500 neurons the spiking network's state error equals the rate-mode network's, 1.7%. That
//!   needs EXACT spike timing ([`crate::neuron::Lif::step_exact`], the loop's default). Stepped
//!   tick by tick at 1 ms, a 300 Hz cell's interval of 3.3 ticks is rounded up to 4, every rate is
//!   biased low, and the same network's error is 47% however many neurons it has.
//! - **PES** ([`Pes`]): with `κ` normalised per neuron, as Nengo normalises it, on a fixed input
//!   every update multiplies the decoding error by exactly `1 − κ|a|²/n`; the rule is stable iff
//!   `κ < 2n/|a|²`, at which the error neither shrinks nor grows but alternates; and what it learns
//!   over a sample set can approach, and never beat, the least-squares decoders' error on that set.
//!   The published rules carry no `1/n`, and on their `κ` the same limit is `2/|a|²`; [`Pes`] says
//!   which paper wrote which form.
//!
//! # What this module has NOT reproduced
//!
//! - Spaun, or any model beyond one or two populations. The framework composes; this module gives
//!   the pieces and checks each, and stops there.
//! - Nengo's exact random-number stream: encoders, intercepts and maximum rates are drawn from
//!   this crate's generator, so a population built from [`EnsembleSpec::default_for`] is
//!   *statistically* the same population as a Nengo 2.x model with default parameters (with its
//!   intercepts drawn from `(−0.999, 0.999)` in place of `(−1, 1)`), and not the same neurons.
//!   Nengo ≥ 3.0 draws intercepts from `Uniform(−1, 0.9)`, and Nengo ≥ 3.1 draws encoders from
//!   `ScatteredHypersphere` — a quasi-random set with a random rotation, not independent Gaussian
//!   draws — so against current Nengo the encoder distribution differs even when the intercept
//!   range is matched. A scattered-hypersphere sampler is not here. The same Nengo change moved
//!   its default evaluation points to that sampler; [`Ensemble::sample_points`] draws them
//!   independently and uniformly in the ball.
//!
//!   This item used to say "the same seed as a Nengo model", with no version. That holds only
//!   against Nengo 2.x. In `nengo/ensemble.py` the encoder default is
//!   `UniformHypersphere(surface=True)` at v2.0.0, v2.8.0 and v3.0.0 — which `nengo/dists.py`
//!   (v2.8.0) samples as `randn`, normalised, the distribution [`Ensemble::new`] draws from, `±1`
//!   in one dimension — and `ScatteredHypersphere(surface=True)` from v3.1.0 on; Nengo's
//!   `CHANGES.rst`, 3.1.0: "The `encoders` and `eval_points` of `Ensemble` are now sampled from
//!   `ScatteredHypersphere` by default. (#1611)". The intercept change is at
//!   [`EnsembleSpec::default_for`].
//! - Exact timing for cells other than [`Lif`], or for the feedforward [`SpikingEnsemble::step`],
//!   which keeps the tick-based step it has always had; [`SpikingEnsemble::step_exact`] is the
//!   exact one.
//! - Higher-order synapse corrections (the subject of the Voelker–Eliasmith paper): the loop here
//!   uses the first-order mapping `A′ = τA + I`, whose error grows as the dynamics get fast
//!   compared with `τ`.
//! - PES on SPIKING activities with a filtered error, as Nengo runs it; [`Pes`] here takes rates.
//!   The voja and BCM rules of the same family are not here.

use core::fmt;

use crate::neuron::{Lif, Neuron};
use crate::reservoir::{ReservoirError, cholesky};
use crate::rng::Rng;

/// What went wrong, named rather than guessed around.
#[derive(Debug, Clone, PartialEq)]
pub enum NefError {
    /// A count of zero where at least one is needed.
    Empty {
        /// What was empty: `"neurons"`, `"dimension"`, `"samples"`.
        what: &'static str,
    },
    /// A vector or matrix of the wrong size.
    Dimension {
        /// Which object.
        what: &'static str,
        /// Length supplied.
        got: usize,
        /// Length required.
        want: usize,
    },
    /// A `NaN` or infinity where a number was needed.
    NonFinite {
        /// Which quantity.
        what: &'static str,
        /// Position in the offending array, `0` for a scalar.
        index: usize,
    },
    /// A parameter outside the range where the formula it feeds is defined.
    OutOfRange {
        /// Which parameter.
        what: &'static str,
        /// Value supplied.
        value: f64,
        /// Lowest admissible value.
        low: f64,
        /// Highest admissible value.
        high: f64,
    },
    /// The least-squares system could not be solved; carries the solver's reason.
    Solve(ReservoirError),
}

impl fmt::Display for NefError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty { what } => write!(f, "{what} is empty"),
            Self::Dimension { what, got, want } => write!(f, "{what} has {got} entries, needs {want}"),
            Self::NonFinite { what, index } => write!(f, "{what} is not finite at {index}"),
            Self::OutOfRange { what, value, low, high } => {
                write!(f, "{what} = {value} is outside [{low}, {high}]")
            }
            Self::Solve(e) => write!(f, "decoder solve failed: {e}"),
        }
    }
}

impl std::error::Error for NefError {}

impl From<ReservoirError> for NefError {
    fn from(e: ReservoirError) -> Self {
        Self::Solve(e)
    }
}

fn finite(what: &'static str, v: &[f64]) -> Result<(), NefError> {
    if let Some(i) = v.iter().position(|x| !x.is_finite()) {
        return Err(NefError::NonFinite { what, index: i });
    }
    Ok(())
}

fn len(what: &'static str, got: usize, want: usize) -> Result<(), NefError> {
    if got == want { Ok(()) } else { Err(NefError::Dimension { what, got, want }) }
}

fn in_range(what: &'static str, value: f64, low: f64, high: f64) -> Result<(), NefError> {
    if value.is_finite() && value >= low && value <= high {
        Ok(())
    } else {
        Err(NefError::OutOfRange { what, value, low, high })
    }
}

// ---------------------------------------------------------------------------------------------
// The rate curve
// ---------------------------------------------------------------------------------------------

/// The normalised leaky integrate-and-fire rate curve the framework is stated in.
///
/// Currents are dimensionless, with the threshold at `J = 1`; the mapping to amperes for a
/// [`crate::neuron::Lif`] is `I = J · (v_th − v_rest) / r_m`, which [`SpikingEnsemble`] applies.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LifRate {
    /// Membrane time constant, seconds.
    pub tau_rc: f64,
    /// Absolute refractory period, seconds. Bounds the rate at `1 / tau_ref`.
    pub tau_ref: f64,
}

impl Default for LifRate {
    /// Nengo's defaults: 20 ms membrane, 2 ms refractory — the same as [`Lif::default`].
    fn default() -> Self {
        Self { tau_rc: 20e-3, tau_ref: 2e-3 }
    }
}

impl LifRate {
    /// Build from time constants.
    ///
    /// # Errors
    ///
    /// [`NefError::OutOfRange`] for a non-positive or non-finite constant.
    pub fn new(tau_rc: f64, tau_ref: f64) -> Result<Self, NefError> {
        in_range("tau_rc", tau_rc, f64::MIN_POSITIVE, f64::INFINITY)?;
        in_range("tau_ref", tau_ref, f64::MIN_POSITIVE, f64::INFINITY)?;
        Ok(Self { tau_rc, tau_ref })
    }

    /// The rate curve of this crate's [`Lif`], with the threshold current normalised to `1`.
    ///
    /// # Errors
    ///
    /// [`NefError::OutOfRange`] if `lif.v_reset != lif.v_rest`: the framework's curve assumes a
    /// reset to rest, and a cell that resets elsewhere has a different interval
    /// (`τ · ln((v_∞ − v_reset)/(v_∞ − v_th))` rather than `τ · ln(J/(J − 1))`).
    pub fn of_lif(lif: &Lif) -> Result<Self, NefError> {
        if lif.v_reset != lif.v_rest {
            return Err(NefError::OutOfRange {
                what: "v_reset − v_rest (the NEF curve needs a reset to rest)",
                value: lif.v_reset - lif.v_rest,
                low: 0.0,
                high: 0.0,
            });
        }
        Self::new(lif.tau_m, lif.t_ref)
    }

    /// `G(J)`: hertz, `0` at and below the threshold current `J = 1`.
    #[must_use]
    pub fn rate(&self, j: f64) -> f64 {
        if !(j > 1.0) {
            return 0.0;
        }
        1.0 / (self.tau_ref + self.tau_rc * (1.0 + 1.0 / (j - 1.0)).ln())
    }

    /// The current that produces `rate` hertz: `1 / (1 − exp((τ_ref − 1/rate) / τ_rc))`.
    ///
    /// `None` for a rate at or below zero, or at or above the refractory bound `1/τ_ref`.
    ///
    /// ⚠ At very low rates the answer is not representable: a 1 Hz cell at the default constants
    /// needs `J − 1 = 2e-22`, which rounds to exactly `1.0`, and [`LifRate::rate`] of that is `0`.
    /// The round trip holds to 1e-9 from about 5 Hz upward and the test says so; below that the
    /// tuning curve's foot is a property of `f64`, not of the neuron.
    #[must_use]
    pub fn current_for_rate(&self, rate: f64) -> Option<f64> {
        if !(rate > 0.0) || rate >= 1.0 / self.tau_ref {
            return None;
        }
        Some(1.0 / (1.0 - ((self.tau_ref - 1.0 / rate) / self.tau_rc).exp()))
    }

    /// The highest rate this curve can produce, `1 / τ_ref`, approached as `J → ∞`.
    #[must_use]
    pub fn max_rate_bound(&self) -> f64 {
        1.0 / self.tau_ref
    }

    /// Gain and bias so that the tuning curve is zero at `intercept` (in units of the radius)
    /// and `max_rate` at the edge of the radius, along the encoder.
    ///
    /// Nengo's `LIF.gain_bias`: `x = current_for_rate(max_rate)`, `gain = (x − 1)/(1 − intercept)`,
    /// `bias = 1 − gain · intercept`.
    ///
    /// # Errors
    ///
    /// [`NefError::OutOfRange`] for an intercept outside `(−1, 1)` or a maximum rate outside
    /// `(0, 1/τ_ref)`.
    pub fn gain_bias(&self, max_rate: f64, intercept: f64) -> Result<(f64, f64), NefError> {
        if !intercept.is_finite() || intercept <= -1.0 || intercept >= 1.0 {
            return Err(NefError::OutOfRange { what: "intercept", value: intercept, low: -1.0, high: 1.0 });
        }
        let x = self.current_for_rate(max_rate).ok_or(NefError::OutOfRange {
            what: "max_rate",
            value: max_rate,
            low: 0.0,
            high: self.max_rate_bound(),
        })?;
        let gain = (x - 1.0) / (1.0 - intercept);
        let bias = 1.0 - gain * intercept;
        Ok((gain, bias))
    }
}

// ---------------------------------------------------------------------------------------------
// Ensembles
// ---------------------------------------------------------------------------------------------

/// How to build an [`Ensemble`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EnsembleSpec {
    /// Neurons. At least one.
    pub n: usize,
    /// Represented dimension. At least one.
    pub dim: usize,
    /// Radius of the represented ball: the population is tuned for `|x| ≤ radius`.
    pub radius: f64,
    /// Maximum firing rates are drawn uniformly from this range, hertz.
    pub max_rate: (f64, f64),
    /// Intercepts are drawn uniformly from this range, in units of the radius, inside `(−1, 1)`.
    pub intercept: (f64, f64),
    /// The rate curve.
    pub neuron: LifRate,
    /// Seed for encoders, intercepts and maximum rates.
    pub seed: u64,
}

impl EnsembleSpec {
    /// Nengo 2.x's defaults for a population of `n` neurons in `dim` dimensions: radius 1, maximum
    /// rates uniform in 200–400 Hz, intercepts uniform in `(−1, 1)` — drawn here from
    /// `(−0.999, 0.999)`, because [`Ensemble::new`] accepts only the open interval `(−1, 1)` (at
    /// an intercept of 1 the gain `(x − 1)/(1 − intercept)` is infinite). Nengo ≥ 3.0 uses
    /// `Uniform(−1, 0.9)`; to follow it, set `intercept` to `(-0.999, 0.9)`.
    ///
    /// This used to say "Nengo's defaults", with no version, and the intercepts are Nengo 2.x's
    /// only. `nengo/ensemble.py` has `default=Uniform(-1.0, 1.0)` at v2.0.0 and v2.8.0 and
    /// `default=Uniform(-1.0, 0.9)` at v3.0.0, v3.1.0, v3.2.0 and v4.0.0. The change is Nengo
    /// commit `2579f0e68e`, "Change default intercept range to -1, 0.9" (2019), and `CHANGES.rst`
    /// gives the reason under 3.0.0 (November 18, 2019): "The default `intercepts` value has been
    /// changed to `Uniform(-1, 0.9)` to avoid high gains when intercepts are close to 1. (#1534,
    /// #1561)". The maximum rates, `Uniform(200, 400)`, are the same in every release named. The
    /// drawn values did not change: the label did. The encoders are Nengo 2.x's too, and not
    /// current Nengo's; [`crate::nef`] says how they differ.
    #[must_use]
    pub fn default_for(n: usize, dim: usize, seed: u64) -> Self {
        Self {
            n,
            dim,
            radius: 1.0,
            max_rate: (200.0, 400.0),
            intercept: (-0.999, 0.999),
            neuron: LifRate::default(),
            seed,
        }
    }
}

/// A population of rate neurons representing a `dim`-dimensional vector.
#[derive(Debug, Clone, PartialEq)]
pub struct Ensemble {
    /// Represented dimension.
    pub dim: usize,
    /// Radius of the represented ball.
    pub radius: f64,
    /// The rate curve every neuron shares.
    pub neuron: LifRate,
    /// Unit encoders, row-major `n × dim`.
    pub encoders: Vec<f64>,
    /// Gains, one per neuron.
    pub gains: Vec<f64>,
    /// Bias currents, one per neuron.
    pub biases: Vec<f64>,
}

/// A standard normal draw by Box-Muller on the crate's generator.
fn normal(rng: &mut Rng) -> f64 {
    let u1 = rng.next_f64().max(1e-300);
    let u2 = rng.next_f64();
    (-2.0 * u1.ln()).sqrt() * (core::f64::consts::TAU * u2).cos()
}

impl Ensemble {
    /// Build from a spec: random unit encoders, uniform intercepts and maximum rates.
    ///
    /// In one dimension the encoders are `±1` with equal probability, which is what "a random
    /// unit vector" is on a line.
    ///
    /// # Errors
    ///
    /// [`NefError::Empty`] for zero neurons or dimension, [`NefError::OutOfRange`] for a
    /// non-positive radius, an intercept range outside `(−1, 1)` or a rate range outside
    /// `(0, 1/τ_ref)`, and anything [`LifRate::gain_bias`] refuses.
    pub fn new(spec: &EnsembleSpec) -> Result<Self, NefError> {
        if spec.n == 0 {
            return Err(NefError::Empty { what: "neurons" });
        }
        if spec.dim == 0 {
            return Err(NefError::Empty { what: "dimension" });
        }
        in_range("radius", spec.radius, f64::MIN_POSITIVE, f64::INFINITY)?;
        let (ilo, ihi) = spec.intercept;
        if !(ilo > -1.0) || !(ihi < 1.0) || !(ilo <= ihi) {
            return Err(NefError::OutOfRange { what: "intercept range", value: ilo, low: -1.0, high: 1.0 });
        }
        let (rlo, rhi) = spec.max_rate;
        if !(rlo > 0.0) || !(rhi < spec.neuron.max_rate_bound()) || !(rlo <= rhi) {
            return Err(NefError::OutOfRange {
                what: "max_rate range",
                value: rlo,
                low: 0.0,
                high: spec.neuron.max_rate_bound(),
            });
        }
        let mut rng = Rng::new(spec.seed);
        let mut encoders = Vec::with_capacity(spec.n * spec.dim);
        let mut gains = Vec::with_capacity(spec.n);
        let mut biases = Vec::with_capacity(spec.n);
        for _ in 0..spec.n {
            if spec.dim == 1 {
                encoders.push(if rng.next_u32() & 1 == 1 { 1.0 } else { -1.0 });
            } else {
                let mut e: Vec<f64> = (0..spec.dim).map(|_| normal(&mut rng)).collect();
                let mut norm = e.iter().map(|x| x * x).sum::<f64>().sqrt();
                if norm == 0.0 {
                    e[0] = 1.0;
                    norm = 1.0;
                }
                encoders.extend(e.iter().map(|x| x / norm));
            }
            let intercept = ilo + (ihi - ilo) * rng.next_f64();
            let max_rate = rlo + (rhi - rlo) * rng.next_f64();
            let (g, b) = spec.neuron.gain_bias(max_rate, intercept)?;
            gains.push(g);
            biases.push(b);
        }
        Ok(Self { dim: spec.dim, radius: spec.radius, neuron: spec.neuron, encoders, gains, biases })
    }

    /// Build from explicit parts, for a population whose tuning is chosen rather than drawn.
    ///
    /// # Errors
    ///
    /// [`NefError::Empty`] for no neurons or dimension, [`NefError::Dimension`] if the arrays
    /// disagree, [`NefError::NonFinite`] for a non-finite entry, [`NefError::OutOfRange`] for a
    /// non-positive radius.
    pub fn from_parts(
        dim: usize,
        radius: f64,
        neuron: LifRate,
        encoders: Vec<f64>,
        gains: Vec<f64>,
        biases: Vec<f64>,
    ) -> Result<Self, NefError> {
        if dim == 0 {
            return Err(NefError::Empty { what: "dimension" });
        }
        let n = gains.len();
        if n == 0 {
            return Err(NefError::Empty { what: "neurons" });
        }
        in_range("radius", radius, f64::MIN_POSITIVE, f64::INFINITY)?;
        len("encoders", encoders.len(), n * dim)?;
        len("biases", biases.len(), n)?;
        finite("encoders", &encoders)?;
        finite("gains", &gains)?;
        finite("biases", &biases)?;
        Ok(Self { dim, radius, neuron, encoders, gains, biases })
    }

    /// Neurons in the population.
    #[must_use]
    pub fn n(&self) -> usize {
        self.gains.len()
    }

    /// Encoder of neuron `i`.
    #[must_use]
    pub fn encoder(&self, i: usize) -> &[f64] {
        &self.encoders[i * self.dim..(i + 1) * self.dim]
    }

    /// Input current of neuron `i` at `x`: `gain · (e·x)/radius + bias`.
    ///
    /// # Errors
    ///
    /// [`NefError::Dimension`] for an `x` of the wrong length, [`NefError::NonFinite`] for a
    /// non-finite one.
    pub fn current(&self, i: usize, x: &[f64]) -> Result<f64, NefError> {
        len("x", x.len(), self.dim)?;
        finite("x", x)?;
        let dot: f64 = self.encoder(i).iter().zip(x).map(|(e, v)| e * v).sum();
        Ok(self.gains[i] * dot / self.radius + self.biases[i])
    }

    /// Every neuron's rate at `x`, hertz.
    ///
    /// # Errors
    ///
    /// As [`Ensemble::current`].
    pub fn rates(&self, x: &[f64]) -> Result<Vec<f64>, NefError> {
        len("x", x.len(), self.dim)?;
        finite("x", x)?;
        Ok((0..self.n())
            .map(|i| {
                let dot: f64 = self.encoder(i).iter().zip(x).map(|(e, v)| e * v).sum();
                self.neuron.rate(self.gains[i] * dot / self.radius + self.biases[i])
            })
            .collect())
    }

    /// `count` points drawn uniformly from the represented ball, for solving decoders.
    ///
    /// One dimension: uniform on `[−r, r]`. Higher: a Gaussian direction scaled by `r · u^(1/dim)`,
    /// which is uniform in the ball.
    #[must_use]
    pub fn sample_points(&self, count: usize, rng: &mut Rng) -> Vec<Vec<f64>> {
        (0..count)
            .map(|_| {
                if self.dim == 1 {
                    return vec![self.radius * (2.0 * rng.next_f64() - 1.0)];
                }
                let mut v: Vec<f64> = (0..self.dim).map(|_| normal(rng)).collect();
                let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-300);
                let scale = self.radius * rng.next_f64().powf(1.0 / self.dim as f64) / norm;
                for x in &mut v {
                    *x *= scale;
                }
                v
            })
            .collect()
    }

    /// Solve least-squares decoders for the function whose values at `samples` are `targets`,
    /// with ridge regularisation.
    ///
    /// Minimises `Σ_s |f(x_s) − Σ_i a_i(x_s) d_i|² + λ · (tr Γ / N) · Σ_i |d_i|²`, where `Γ` is the
    /// Gram matrix of the activities: `(Γ + λ·(tr Γ/N)·I) D = Υ`. The regularisation is stated
    /// relative to the mean diagonal of `Γ` so that `λ` means the same thing at every population
    /// size and sample count; `0.01` to `0.1` is the useful range, and `0` is the plain least
    /// squares, which is singular whenever two neurons share a tuning curve.
    ///
    /// # Errors
    ///
    /// [`NefError::Empty`] for no samples, [`NefError::Dimension`] if a sample or target has the
    /// wrong length or the two lists differ in count, [`NefError::NonFinite`] for a non-finite
    /// entry or a negative `lambda`, [`NefError::Solve`] if the regularised system is not positive
    /// definite.
    pub fn decoders(
        &self,
        samples: &[Vec<f64>],
        targets: &[Vec<f64>],
        lambda: f64,
    ) -> Result<Decoders, NefError> {
        if samples.is_empty() {
            return Err(NefError::Empty { what: "samples" });
        }
        len("targets", targets.len(), samples.len())?;
        if !(lambda >= 0.0) || !lambda.is_finite() {
            return Err(NefError::NonFinite { what: "lambda", index: 0 });
        }
        let out_dim = targets[0].len();
        if out_dim == 0 {
            return Err(NefError::Empty { what: "target dimension" });
        }
        let n = self.n();
        let s = samples.len() as f64;
        let mut gamma = vec![0.0f64; n * n];
        let mut upsilon = vec![0.0f64; n * out_dim];
        for (x, y) in samples.iter().zip(targets) {
            len("target", y.len(), out_dim)?;
            finite("target", y)?;
            let a = self.rates(x)?;
            for i in 0..n {
                if a[i] == 0.0 {
                    continue;
                }
                for k in i..n {
                    gamma[i * n + k] += a[i] * a[k] / s;
                }
                for (j, &yj) in y.iter().enumerate() {
                    upsilon[i * out_dim + j] += a[i] * yj / s;
                }
            }
        }
        // Fill the lower triangle; `cholesky` reads it.
        for i in 0..n {
            for k in i + 1..n {
                gamma[k * n + i] = gamma[i * n + k];
            }
        }
        let trace: f64 = (0..n).map(|i| gamma[i * n + i]).sum();
        let ridge = lambda * trace / n as f64;
        for i in 0..n {
            gamma[i * n + i] += ridge;
        }
        let chol = cholesky(&gamma, n, 1e-14)?;
        let mut d = vec![0.0f64; n * out_dim];
        let mut rhs = vec![0.0f64; n];
        for j in 0..out_dim {
            for i in 0..n {
                rhs[i] = upsilon[i * out_dim + j];
            }
            let col = chol.solve(&rhs)?;
            for i in 0..n {
                d[i * out_dim + j] = col[i];
            }
        }
        Ok(Decoders { d, n, out_dim })
    }

    /// Decoders for the identity over `count` uniform samples.
    ///
    /// # Errors
    ///
    /// As [`Ensemble::decoders`].
    pub fn identity_decoders(&self, count: usize, lambda: f64, rng: &mut Rng) -> Result<Decoders, NefError> {
        let samples = self.sample_points(count, rng);
        let targets = samples.clone();
        self.decoders(&samples, &targets, lambda)
    }

    /// `x̂ = Σ_i a_i · d_i` for a rate vector.
    ///
    /// # Errors
    ///
    /// [`NefError::Dimension`] if `rates` or the decoders do not match this population.
    pub fn decode(&self, decoders: &Decoders, rates: &[f64]) -> Result<Vec<f64>, NefError> {
        len("rates", rates.len(), self.n())?;
        len("decoders", decoders.n, self.n())?;
        let mut out = vec![0.0f64; decoders.out_dim];
        for (i, &a) in rates.iter().enumerate() {
            if a == 0.0 {
                continue;
            }
            for (j, o) in out.iter_mut().enumerate() {
                *o += a * decoders.d[i * decoders.out_dim + j];
            }
        }
        Ok(out)
    }

    /// Root-mean-square decoding error over `samples` against `targets`.
    ///
    /// # Errors
    ///
    /// As [`Ensemble::decode`], plus [`NefError::Empty`] for no samples.
    pub fn rmse(&self, decoders: &Decoders, samples: &[Vec<f64>], targets: &[Vec<f64>]) -> Result<f64, NefError> {
        if samples.is_empty() {
            return Err(NefError::Empty { what: "samples" });
        }
        len("targets", targets.len(), samples.len())?;
        let mut acc = 0.0;
        let mut count = 0usize;
        for (x, y) in samples.iter().zip(targets) {
            let x_hat = self.decode(decoders, &self.rates(x)?)?;
            len("target", y.len(), x_hat.len())?;
            for (a, b) in x_hat.iter().zip(y) {
                acc += (a - b) * (a - b);
                count += 1;
            }
        }
        Ok((acc / count as f64).sqrt())
    }
}

/// Least-squares decoders: row-major `n × out_dim`.
#[derive(Debug, Clone, PartialEq)]
pub struct Decoders {
    /// `d[i * out_dim + j]` is neuron `i`'s weight on output `j`.
    pub d: Vec<f64>,
    /// Neurons.
    pub n: usize,
    /// Output dimension.
    pub out_dim: usize,
}

impl Decoders {
    /// The decoders of `T · f(x)` for a linear `transform` (row-major `rows × out_dim`), which
    /// are `d_i ↦ T d_i` — linearity of the least squares in its target.
    ///
    /// # Errors
    ///
    /// [`NefError::Dimension`] if `transform` is not `rows × out_dim`, or `rows` is zero.
    pub fn transformed(&self, transform: &[f64], rows: usize) -> Result<Decoders, NefError> {
        if rows == 0 {
            return Err(NefError::Empty { what: "transform rows" });
        }
        len("transform", transform.len(), rows * self.out_dim)?;
        let mut d = vec![0.0f64; self.n * rows];
        for i in 0..self.n {
            for r in 0..rows {
                let mut acc = 0.0;
                for j in 0..self.out_dim {
                    acc += transform[r * self.out_dim + j] * self.d[i * self.out_dim + j];
                }
                d[i * rows + r] = acc;
            }
        }
        Ok(Decoders { d, n: self.n, out_dim: rows })
    }
}

// ---------------------------------------------------------------------------------------------
// Connections
// ---------------------------------------------------------------------------------------------

/// The full weight matrix of a connection, row-major `n_post × n_pre`:
/// `W_ji = gain_j · (e_j · T · d_i) / radius_post`, with `transform` row-major
/// `post.dim × pre_decoders.out_dim`.
///
/// This is the rank-`d` product a neuromorphic mapping avoids materialising; it is computed here
/// so that the factorised path can be checked against it, and so that a full-matrix deployment can
/// be priced.
///
/// # Errors
///
/// [`NefError::Dimension`] if `transform` is not `post.dim × pre_decoders.out_dim`.
pub fn full_weights(pre_decoders: &Decoders, transform: &[f64], post: &Ensemble) -> Result<Vec<f64>, NefError> {
    let td = pre_decoders.transformed(transform, post.dim)?;
    let (n_pre, n_post) = (pre_decoders.n, post.n());
    let mut w = vec![0.0f64; n_post * n_pre];
    for j in 0..n_post {
        let e = post.encoder(j);
        for i in 0..n_pre {
            let mut acc = 0.0;
            for k in 0..post.dim {
                acc += e[k] * td.d[i * post.dim + k];
            }
            w[j * n_pre + i] = post.gains[j] * acc / post.radius;
        }
    }
    Ok(w)
}

/// Weight traffic per tick for one connection, full matrix against factorised, in weights moved.
///
/// A full `n_post × n_pre` matrix moves `spikes · n_post` weights: every presynaptic spike touches
/// one row. The factorised form moves `spikes · d` decoder entries to form the `d`-vector and then
/// `d · n_post` encoder entries to deliver it — and that second term is paid **every tick**,
/// whether or not anything fired. Returns `(full, factorised)`; the two cross at
/// `spikes = d · n_post / (n_post − d)`, so below about `d` spikes a tick the factorisation is more
/// traffic, and above it less, up to `n_post / d` less at saturation.
#[must_use]
pub fn connection_traffic(spikes_per_tick: u64, n_post: u64, dim: u64) -> (u64, u64) {
    let full = spikes_per_tick.saturating_mul(n_post);
    let factorised = spikes_per_tick.saturating_mul(dim).saturating_add(dim.saturating_mul(n_post));
    (full, factorised)
}

/// The spike count per tick above which the factorised connection moves fewer weights than the
/// full matrix: `d · n_post / (n_post − d)`. `None` when `dim >= n_post`, where it never does.
#[must_use]
pub fn factorisation_break_even(n_post: u64, dim: u64) -> Option<f64> {
    if dim >= n_post || n_post == 0 {
        return None;
    }
    Some(dim as f64 * n_post as f64 / (n_post - dim) as f64)
}

// ---------------------------------------------------------------------------------------------
// Dynamics
// ---------------------------------------------------------------------------------------------

/// A first-order low-pass synapse, `τ ẏ = −y + x`, stepped by exponential Euler, which is exact
/// for an input constant over the step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Lowpass {
    /// Time constant, seconds.
    pub tau: f64,
    /// Current output.
    pub y: f64,
}

impl Lowpass {
    /// A synapse of time constant `tau`, starting at zero.
    ///
    /// # Errors
    ///
    /// [`NefError::OutOfRange`] for a non-positive or non-finite `tau`.
    pub fn new(tau: f64) -> Result<Self, NefError> {
        in_range("tau", tau, f64::MIN_POSITIVE, f64::INFINITY)?;
        Ok(Self { tau, y: 0.0 })
    }

    /// Advance by `dt` under input `x` and return the new output.
    #[must_use]
    pub fn step(&mut self, dt: f64, x: f64) -> f64 {
        let decay = (-dt / self.tau).exp();
        self.y = x + (self.y - x) * decay;
        self.y
    }
}

/// Principle 3: the recurrent and input matrices that implement `ẋ = A x + B u` through a synapse
/// of time constant `tau`: `A' = τ A + I`, `B' = τ B`. `a` is row-major `dim × dim`, `b` row-major
/// `dim × k`.
///
/// # Errors
///
/// [`NefError::Dimension`] if `a` is not `dim × dim` or `b` not `dim × k` for the `k` implied by
/// its length, [`NefError::OutOfRange`] for a non-positive `tau`, [`NefError::Empty`] for
/// `dim = 0`.
pub fn dynamics_transform(a: &[f64], b: &[f64], dim: usize, tau: f64) -> Result<(Vec<f64>, Vec<f64>), NefError> {
    if dim == 0 {
        return Err(NefError::Empty { what: "dimension" });
    }
    len("A", a.len(), dim * dim)?;
    if b.is_empty() || !b.len().is_multiple_of(dim) {
        return Err(NefError::Dimension { what: "B", got: b.len(), want: dim });
    }
    in_range("tau", tau, f64::MIN_POSITIVE, f64::INFINITY)?;
    finite("A", a)?;
    finite("B", b)?;
    let mut a2 = a.iter().map(|x| tau * x).collect::<Vec<f64>>();
    for i in 0..dim {
        a2[i * dim + i] += 1.0;
    }
    let b2 = b.iter().map(|x| tau * x).collect();
    Ok((a2, b2))
}

/// A population closed on itself through a synapse: `ẋ = A x + B u` in rate mode.
///
/// Each tick: the synapse filters `A' x̂ + B' u`, the population fires at the rates that filtered
/// value drives, and `x̂` is decoded from those rates by the identity decoders. That is the
/// standard rate-mode execution of an NEF network and it is what the integrator and oscillator
/// tests compare with the ideal system.
#[derive(Debug, Clone, PartialEq)]
pub struct RateLoop {
    /// The population.
    pub ensemble: Ensemble,
    /// Identity decoders for it.
    pub decoders: Decoders,
    /// `A' = τ A + I`, row-major `dim × dim`.
    pub a_prime: Vec<f64>,
    /// `B' = τ B`, row-major `dim × k`.
    pub b_prime: Vec<f64>,
    /// Input dimension `k`.
    pub input_dim: usize,
    /// One synapse per represented dimension.
    pub synapses: Vec<Lowpass>,
    /// The current decoded state.
    pub x_hat: Vec<f64>,
}

impl RateLoop {
    /// Build for the system `ẋ = A x + B u` through synapses of time constant `tau`.
    ///
    /// # Errors
    ///
    /// As [`dynamics_transform`], plus [`NefError::Dimension`] if the decoders do not decode
    /// `ensemble.dim` outputs.
    pub fn new(
        ensemble: Ensemble,
        decoders: Decoders,
        a: &[f64],
        b: &[f64],
        tau: f64,
    ) -> Result<Self, NefError> {
        let dim = ensemble.dim;
        len("decoder outputs", decoders.out_dim, dim)?;
        len("decoders", decoders.n, ensemble.n())?;
        let (a_prime, b_prime) = dynamics_transform(a, b, dim, tau)?;
        let input_dim = b.len() / dim;
        let synapses = (0..dim).map(|_| Lowpass::new(tau)).collect::<Result<Vec<_>, _>>()?;
        Ok(Self { ensemble, decoders, a_prime, b_prime, input_dim, synapses, x_hat: vec![0.0; dim] })
    }

    /// Advance one tick under input `u` and return the decoded state.
    ///
    /// # Errors
    ///
    /// [`NefError::Dimension`] for a `u` of the wrong length, [`NefError::NonFinite`] for a
    /// non-finite one.
    pub fn step(&mut self, dt: f64, u: &[f64]) -> Result<&[f64], NefError> {
        len("u", u.len(), self.input_dim)?;
        finite("u", u)?;
        let dim = self.ensemble.dim;
        let mut drive = vec![0.0f64; dim];
        for i in 0..dim {
            let mut acc = 0.0;
            for j in 0..dim {
                acc += self.a_prime[i * dim + j] * self.x_hat[j];
            }
            for (j, &uj) in u.iter().enumerate() {
                acc += self.b_prime[i * self.input_dim + j] * uj;
            }
            drive[i] = self.synapses[i].step(dt, acc);
        }
        let rates = self.ensemble.rates(&drive)?;
        self.x_hat = self.ensemble.decode(&self.decoders, &rates)?;
        Ok(&self.x_hat)
    }

    /// Set the represented state directly — a kick, or an initial condition.
    ///
    /// # Errors
    ///
    /// [`NefError::Dimension`], [`NefError::NonFinite`].
    pub fn set_state(&mut self, x: &[f64]) -> Result<(), NefError> {
        len("x", x.len(), self.ensemble.dim)?;
        finite("x", x)?;
        self.x_hat.copy_from_slice(x);
        for (s, &v) in self.synapses.iter_mut().zip(x) {
            s.y = v;
        }
        Ok(())
    }
}

/// A SPIKING population closed on itself through a synapse: `ẋ = A x + B u` carried by spikes.
///
/// Each tick the neurons that fired are decoded — a spike of neuron `i` is an impulse of weight
/// `d_i / dt` — the recurrent and input terms `A′ p + B′ u` pass through the synapse, and the
/// filtered value is the current that drives the population on the next tick. The readout
/// `x_hat` is the same decoded spike train through a synapse of its own.
#[derive(Debug, Clone, PartialEq)]
pub struct SpikingLoop {
    /// The spiking population.
    pub population: SpikingEnsemble,
    /// Identity decoders for it, in units per hertz.
    pub decoders: Decoders,
    /// `A′ = τ A + I`, row-major `dim × dim`.
    pub a_prime: Vec<f64>,
    /// `B′ = τ B`, row-major `dim × k`.
    pub b_prime: Vec<f64>,
    /// Input dimension `k`.
    pub input_dim: usize,
    /// The recurrent synapses, one per dimension; their outputs are the represented state that
    /// drives the population.
    pub synapses: Vec<Lowpass>,
    /// The readout synapses, one per dimension.
    pub readout: Vec<Lowpass>,
    /// The decoded state: the readout synapses' outputs.
    pub x_hat: Vec<f64>,
    /// Spikes emitted since construction: what the computation cost.
    pub spikes: u64,
    /// Whether the cells are stepped with exact spike timing ([`Lif::step_exact`]). On by
    /// default; turned off, every interspike interval is rounded up to whole ticks and the loop
    /// needs a tick far below its shortest interval to work at all.
    pub exact_timing: bool,
}

impl SpikingLoop {
    /// Build for `ẋ = A x + B u` through synapses of time constant `tau`.
    ///
    /// # Errors
    ///
    /// As [`RateLoop::new`].
    pub fn new(population: SpikingEnsemble, decoders: Decoders, a: &[f64], b: &[f64], tau: f64) -> Result<Self, NefError> {
        let dim = population.ensemble.dim;
        len("decoder outputs", decoders.out_dim, dim)?;
        len("decoders", decoders.n, population.ensemble.n())?;
        let (a_prime, b_prime) = dynamics_transform(a, b, dim, tau)?;
        let input_dim = b.len() / dim;
        let bank = || (0..dim).map(|_| Lowpass::new(tau)).collect::<Result<Vec<_>, _>>();
        Ok(Self { population, decoders, a_prime, b_prime, input_dim, synapses: bank()?, readout: bank()?, x_hat: vec![0.0; dim], spikes: 0, exact_timing: true })
    }

    /// Advance one tick of `dt` under input `u`; returns the decoded state.
    ///
    /// # Errors
    ///
    /// [`NefError::Dimension`] for a `u` of the wrong length, [`NefError::NonFinite`] for a
    /// non-finite one, [`NefError::OutOfRange`] for a non-positive `dt`.
    pub fn step(&mut self, dt: f64, u: &[f64]) -> Result<&[f64], NefError> {
        len("u", u.len(), self.input_dim)?;
        finite("u", u)?;
        in_range("dt", dt, f64::MIN_POSITIVE, f64::MAX)?;
        let dim = self.population.ensemble.dim;
        let state: Vec<f64> = self.synapses.iter().map(|s| s.y).collect();
        let fired: Vec<u32> = if self.exact_timing {
            self.population.step_exact(dt, &state)?
        } else {
            self.population.step(dt, &state)?.into_iter().map(u32::from).collect()
        };
        let mut pulse = vec![0.0f64; dim];
        for (i, &count) in fired.iter().enumerate().filter(|(_, c)| **c > 0) {
            self.spikes += u64::from(count);
            for (j, p) in pulse.iter_mut().enumerate() {
                *p += f64::from(count) * self.decoders.d[i * dim + j] / dt;
            }
        }
        for i in 0..dim {
            let mut acc = 0.0;
            for j in 0..dim {
                acc += self.a_prime[i * dim + j] * pulse[j];
            }
            for (j, &uj) in u.iter().enumerate() {
                acc += self.b_prime[i * self.input_dim + j] * uj;
            }
            let _ = self.synapses[i].step(dt, acc);
            self.x_hat[i] = self.readout[i].step(dt, pulse[i]);
        }
        Ok(&self.x_hat)
    }
}

// ---------------------------------------------------------------------------------------------
// Learning the decoders
// ---------------------------------------------------------------------------------------------

/// The PES (Prescribed Error Sensitivity) rule, written with Nengo's per-neuron normalisation:
/// `Δd_i = −(κ/n) a_i E`, with `E = x̂ − target` the decoded error broadcast to the population. It
/// is Widrow–Hoff least-mean-squares on the decoders — local to the neuron, given a broadcast
/// error, which is the form a chip's learning engine can run.
///
/// The rule is `MacNeil` and Eliasmith, *Fine-tuning and the stability of recurrent neural
/// networks*, `PLoS` ONE 6(9):e22885 (2011), doi:10.1371/journal.pone.0022885, Eq. 16,
/// "Converting this into standard delta rule form, and including the learning rate parameter":
/// `Δd_i = κ v_c a_i`, with `v_c` the corrective-saccade signal. The name is Bekolay,
/// Kolbeck and Eliasmith, *Simultaneous unsupervised and supervised learning of cognitive functions
/// in biologically plausible spiking neural networks*, `CogSci` (2013),
/// <https://compneuro.uwaterloo.ca/files/publications/bekolay.2013.pdf>, Eq. 6, `Δd_i = κ E a_i`:
/// "`MacNeil` and Eliasmith (2011) proposed a learning rule that minimizes the error … We will refer
/// to Equation (6) as the Prescribed Error Sensitivity (PES) rule." Neither equation has a `1/n`.
/// The `1/n` is Nengo's: `SimPES` in `nengo/builder/learning_rules.py` (v4.0.0) computes
/// `alpha = -self.learning_rate * dt / n_neurons`. This rule is one update, not one time step, so
/// `dt` is folded into `κ`: the `κ` here is Nengo's `learning_rate × dt`, and it is the papers' `κ`
/// times `n`. The error here is `x̂ − target`, which is why the step carries a minus, as Nengo's
/// negative `alpha` does.
///
/// This used to credit `Δd_i = −(κ/n) a_i E`, and the name "Prescribed Error Sensitivity", to both
/// papers. The `1/n` is in neither, and this review did not locate the name, or "PES", in the 2011
/// paper's full text (Europe PMC, PMC3181247); the name is the 2013 paper's. Nothing numeric
/// changed: [`Pes::update`], [`Pes::contraction`] and [`Pes::stability_limit`] agree with one
/// another under the normalisation stated here, and only the stability limit's units needed saying
/// — see [`Pes::kappa`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pes {
    /// Learning rate `κ`, normalised per neuron as in Nengo's `SimPES`. With that normalisation,
    /// on a fixed input, the rule is stable iff `κ < 2n/|a|²`, which with rates in hertz is small:
    /// fifty neurons at a hundred hertz put it at `2e-4`. On the papers' unnormalised `κ`
    /// (Bekolay et al. 2013 Eq. 6; `MacNeil` and Eliasmith 2011 Eq. 16) the same bound is
    /// `κ < 2/|a|²` per update. This used to state `2n/|a|²` without saying it belongs to the
    /// normalisation and not to either paper.
    pub kappa: f64,
}

/// What one PES update saw and did.
#[derive(Debug, Clone, PartialEq)]
pub struct PesStep {
    /// The decoded error `x̂ − target` BEFORE the update, one entry per output dimension.
    pub error: Vec<f64>,
    /// Decoder entries written: active neurons × output dimensions. A silent neuron's decoders do
    /// not move, and are not touched.
    pub touched: usize,
}

impl Pes {
    /// Build.
    ///
    /// # Errors
    ///
    /// [`NefError::OutOfRange`] for a non-positive or non-finite `kappa`.
    pub fn new(kappa: f64) -> Result<Self, NefError> {
        in_range("kappa", kappa, f64::MIN_POSITIVE, f64::MAX)?;
        Ok(Self { kappa })
    }

    /// The factor `1 − κ|a|²/n` the decoding error is multiplied by when the same activities are
    /// presented again, on this rule's per-neuron `κ` (on the papers' `κ` it is `1 − κ|a|²`).
    /// Inside `(−1, 1)` the rule converges on that input.
    #[must_use]
    pub fn contraction(&self, rates: &[f64]) -> f64 {
        let energy: f64 = rates.iter().map(|a| a * a).sum();
        1.0 - self.kappa * energy / rates.len().max(1) as f64
    }

    /// The learning rate at which the rule stops converging on these activities, `2n/|a|²`, in the
    /// per-neuron units of [`Pes::kappa`]; divide by `n` for the papers' `2/|a|²`. `None` for a
    /// silent population, which no rate can destabilise — or teach.
    #[must_use]
    pub fn stability_limit(rates: &[f64]) -> Option<f64> {
        let energy: f64 = rates.iter().map(|a| a * a).sum();
        if energy > 0.0 && energy.is_finite() { Some(2.0 * rates.len() as f64 / energy) } else { None }
    }

    /// One update of `decoders` from the activities `rates` toward `target`.
    ///
    /// # Errors
    ///
    /// [`NefError::Dimension`] if `rates` does not match the decoders' population or `target`
    /// their output dimension; [`NefError::NonFinite`] for a bad rate or target.
    pub fn update(&self, decoders: &mut Decoders, rates: &[f64], target: &[f64]) -> Result<PesStep, NefError> {
        len("rates", rates.len(), decoders.n)?;
        len("target", target.len(), decoders.out_dim)?;
        finite("rates", rates)?;
        finite("target", target)?;
        let out = decoders.out_dim;
        let mut error = vec![0.0f64; out];
        for (i, &a) in rates.iter().enumerate() {
            if a != 0.0 {
                for (j, e) in error.iter_mut().enumerate() {
                    *e += a * decoders.d[i * out + j];
                }
            }
        }
        for (e, t) in error.iter_mut().zip(target) {
            *e -= t;
        }
        let scale = self.kappa / decoders.n as f64;
        let mut touched = 0;
        for (i, &a) in rates.iter().enumerate() {
            if a != 0.0 {
                for (j, e) in error.iter().enumerate() {
                    decoders.d[i * out + j] -= scale * a * e;
                }
                touched += out;
            }
        }
        Ok(PesStep { error, touched })
    }
}

// ---------------------------------------------------------------------------------------------
// Spiking
// ---------------------------------------------------------------------------------------------

/// The same population as spiking cells: this crate's [`Lif`], one per neuron, driven by the
/// framework's normalised current converted to amperes.
#[derive(Debug, Clone, PartialEq)]
pub struct SpikingEnsemble {
    /// The tuning.
    pub ensemble: Ensemble,
    /// One cell per neuron.
    pub cells: Vec<Lif>,
    /// Spikes emitted per neuron since the last reset.
    pub counts: Vec<u64>,
    /// Ticks stepped since the last reset.
    pub ticks: u64,
    /// Amperes per unit of normalised current: `(v_th − v_rest) / r_m`.
    pub amps_per_unit: f64,
}

impl SpikingEnsemble {
    /// Build from an ensemble and a prototype cell.
    ///
    /// # Errors
    ///
    /// [`NefError::OutOfRange`] if the cell's time constants do not match the ensemble's rate
    /// curve, or if `v_reset != v_rest`, or `r_m ≤ 0`, or `v_th ≤ v_rest`.
    pub fn new(ensemble: Ensemble, proto: Lif) -> Result<Self, NefError> {
        let curve = LifRate::of_lif(&proto)?;
        if curve != ensemble.neuron {
            return Err(NefError::OutOfRange {
                what: "cell time constants (must match the ensemble's rate curve)",
                value: proto.tau_m,
                low: ensemble.neuron.tau_rc,
                high: ensemble.neuron.tau_rc,
            });
        }
        in_range("r_m", proto.r_m, f64::MIN_POSITIVE, f64::INFINITY)?;
        in_range("v_th − v_rest", proto.v_th - proto.v_rest, f64::MIN_POSITIVE, f64::INFINITY)?;
        let amps_per_unit = (proto.v_th - proto.v_rest) / proto.r_m;
        let n = ensemble.n();
        Ok(Self { ensemble, cells: vec![proto; n], counts: vec![0; n], ticks: 0, amps_per_unit })
    }

    /// Advance every cell one tick under the represented `x`, returning which fired.
    ///
    /// # Errors
    ///
    /// As [`Ensemble::current`].
    pub fn step(&mut self, dt: f64, x: &[f64]) -> Result<Vec<bool>, NefError> {
        len("x", x.len(), self.ensemble.dim)?;
        finite("x", x)?;
        let mut fired = vec![false; self.cells.len()];
        for (i, cell) in self.cells.iter_mut().enumerate() {
            let j = self.ensemble.current(i, x)?;
            let amps = j * self.amps_per_unit;
            if cell.step(dt, amps) {
                fired[i] = true;
                self.counts[i] += 1;
            }
        }
        self.ticks += 1;
        Ok(fired)
    }

    /// The same tick with EXACT spike timing ([`Lif::step_exact`]): the number of spikes each cell
    /// emitted, which can exceed one. The count over a run no longer depends on the tick.
    ///
    /// # Errors
    ///
    /// As [`SpikingEnsemble::step`], plus [`NefError::OutOfRange`] for a `dt` that is not positive
    /// and finite.
    pub fn step_exact(&mut self, dt: f64, x: &[f64]) -> Result<Vec<u32>, NefError> {
        len("x", x.len(), self.ensemble.dim)?;
        finite("x", x)?;
        in_range("dt", dt, f64::MIN_POSITIVE, f64::MAX)?;
        let mut fired = vec![0u32; self.cells.len()];
        for (i, cell) in self.cells.iter_mut().enumerate() {
            let j = self.ensemble.current(i, x)?;
            let spikes = cell.step_exact(dt, j * self.amps_per_unit).unwrap_or(0);
            fired[i] = spikes;
            self.counts[i] += u64::from(spikes);
        }
        self.ticks += 1;
        Ok(fired)
    }

    /// Measured rates, hertz, over the ticks since the last reset. `None` before any tick.
    #[must_use]
    pub fn measured_rates(&self, dt: f64) -> Option<Vec<f64>> {
        if self.ticks == 0 {
            return None;
        }
        let t = self.ticks as f64 * dt;
        Some(self.counts.iter().map(|&c| c as f64 / t).collect())
    }

    /// Return every cell to rest and zero the counters.
    pub fn reset(&mut self) {
        for c in &mut self.cells {
            c.reset();
        }
        self.counts.iter_mut().for_each(|c| *c = 0);
        self.ticks = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Decoders, Ensemble, EnsembleSpec, LifRate, Lowpass, NefError, Pes, RateLoop, SpikingEnsemble, SpikingLoop,
        connection_traffic, dynamics_transform, factorisation_break_even, full_weights, normal,
    };
    use crate::neuron::Lif;
    use crate::rng::Rng;

    // ---- the rate curve ----

    /// `G(J)` is `1 / Lif::isi` for this crate's own cell, with the current converted the way the
    /// module doc says. Two derivations of the same interval, agreeing to 1e-12.
    #[test]
    fn the_rate_curve_is_the_reciprocal_of_this_crates_own_isi() {
        let lif = Lif::default();
        let curve = LifRate::of_lif(&lif).unwrap();
        let amps_per_unit = (lif.v_th - lif.v_rest) / lif.r_m;
        for j in [1.0001, 1.01, 1.1, 1.5, 2.0, 5.0, 20.0, 200.0] {
            let want = 1.0 / lif.isi(j * amps_per_unit).expect("above threshold");
            let got = curve.rate(j);
            assert!((got - want).abs() <= 1e-12 * want, "J {j}: {got} vs 1/isi {want}");
        }
        assert_eq!(curve.rate(1.0), 0.0);
        assert_eq!(curve.rate(0.5), 0.0);
        assert_eq!(curve.rate(f64::NAN), 0.0);
        assert_eq!(lif.isi(0.9 * amps_per_unit), None, "sub-threshold agrees too");
        // A cell that resets somewhere other than rest has a different curve, and is refused.
        let other = Lif { v_reset: -70e-3, ..Lif::default() };
        assert!(matches!(LifRate::of_lif(&other), Err(NefError::OutOfRange { .. })));
    }

    /// `current_for_rate` inverts `rate` to 1e-12, and refuses beyond the refractory bound.
    #[test]
    fn the_current_for_a_rate_round_trips() {
        let curve = LifRate::default();
        for r in [5.0, 10.0, 100.0, 250.0, 400.0, 499.0] {
            let j = curve.current_for_rate(r).unwrap();
            assert!((curve.rate(j) - r).abs() < 1e-9, "rate {r}: J {j} gives {}", curve.rate(j));
        }
        // The foot of the curve is below f64's resolution: 1 Hz needs J − 1 = 2e-22.
        assert_eq!(curve.current_for_rate(1.0), Some(1.0));
        assert_eq!(curve.rate(1.0), 0.0);
        assert_eq!(curve.max_rate_bound(), 500.0);
        assert_eq!(curve.current_for_rate(500.0), None);
        assert_eq!(curve.current_for_rate(0.0), None);
        assert_eq!(curve.current_for_rate(-1.0), None);
    }

    /// Nengo's gain-and-bias solution, checked at the two points it is defined by: the tuning
    /// curve is exactly zero just below the intercept and `max_rate` to 1e-9 at the edge.
    #[test]
    fn gain_and_bias_put_the_tuning_curve_where_nengo_says() {
        let curve = LifRate::default();
        for (max_rate, intercept) in [(200.0, -0.5), (300.0, 0.0), (400.0, 0.7), (50.0, -0.95)] {
            let (g, b) = curve.gain_bias(max_rate, intercept).unwrap();
            let ens = Ensemble::from_parts(1, 1.0, curve, vec![1.0], vec![g], vec![b]).unwrap();
            assert_eq!(ens.rates(&[intercept - 1e-9]).unwrap()[0], 0.0, "below the intercept");
            assert!(ens.rates(&[intercept]).unwrap()[0] < 2.0, "at the intercept, within rounding of J = 1");
            let at_edge = ens.rates(&[1.0]).unwrap()[0];
            assert!((at_edge - max_rate).abs() < 1e-9, "max rate {max_rate}: {at_edge}");
            assert!(ens.rates(&[0.5 * (intercept + 1.0)]).unwrap()[0] < max_rate);
            // The literal formula, retyped.
            let x = 1.0 / (1.0 - ((2e-3 - 1.0 / max_rate) / 20e-3).exp());
            assert!((g - (x - 1.0) / (1.0 - intercept)).abs() < 1e-12);
            assert!((b - (1.0 - g * intercept)).abs() < 1e-12);
        }
        assert!(matches!(curve.gain_bias(300.0, 1.0), Err(NefError::OutOfRange { what: "intercept", .. })));
        assert!(matches!(curve.gain_bias(600.0, 0.0), Err(NefError::OutOfRange { what: "max_rate", .. })));
    }

    // ---- representation ----

    /// Decoding the identity: the error falls with `N` across an eightfold sweep, monotonically,
    /// and a population of 800 represents a scalar to better than 1%. The samples used to measure
    /// are not the samples used to solve.
    #[test]
    fn decoding_error_falls_with_population_size() {
        let mut rng = Rng::new(11);
        let mut last = f64::INFINITY;
        let mut errors = Vec::new();
        for n in [50usize, 100, 200, 400, 800] {
            let ens = Ensemble::new(&EnsembleSpec::default_for(n, 1, 7)).unwrap();
            let dec = ens.identity_decoders(500, 0.01, &mut rng).unwrap();
            let test = ens.sample_points(500, &mut rng);
            let rmse = ens.rmse(&dec, &test, &test).unwrap();
            assert!(rmse < last, "N {n}: rmse {rmse} did not fall below {last}");
            last = rmse;
            errors.push((n, rmse));
        }
        println!("nef identity rmse by N: {errors:?}");
        assert!(last < 0.01, "800 neurons decode a scalar to {last}");
        assert!(errors[0].1 / last > 3.0, "the sweep barely moved: {errors:?}");
    }

    /// Transformation: decoders for `x²` decode `x²` and not `x`, and a two-dimensional population
    /// decodes a product of its coordinates.
    #[test]
    fn decoders_compute_the_function_they_were_solved_for() {
        let mut rng = Rng::new(12);
        let ens = Ensemble::new(&EnsembleSpec::default_for(400, 1, 3)).unwrap();
        let samples = ens.sample_points(600, &mut rng);
        let squares: Vec<Vec<f64>> = samples.iter().map(|x| vec![x[0] * x[0]]).collect();
        let dec = ens.decoders(&samples, &squares, 0.01).unwrap();
        let test = ens.sample_points(300, &mut rng);
        let test_sq: Vec<Vec<f64>> = test.iter().map(|x| vec![x[0] * x[0]]).collect();
        let err_sq = ens.rmse(&dec, &test, &test_sq).unwrap();
        let err_as_identity = ens.rmse(&dec, &test, &test).unwrap();
        assert!(err_sq < 0.03, "x² decoded at rmse {err_sq}");
        assert!(err_as_identity > 0.2, "the square decoders also fit x, which cannot be: {err_as_identity}");

        let ens2 = Ensemble::new(&EnsembleSpec::default_for(1000, 2, 5)).unwrap();
        let s2 = ens2.sample_points(1500, &mut rng);
        let prod: Vec<Vec<f64>> = s2.iter().map(|x| vec![x[0] * x[1]]).collect();
        let dec2 = ens2.decoders(&s2, &prod, 0.01).unwrap();
        let t2 = ens2.sample_points(300, &mut rng);
        let tp: Vec<Vec<f64>> = t2.iter().map(|x| vec![x[0] * x[1]]).collect();
        let err = ens2.rmse(&dec2, &t2, &tp).unwrap();
        assert!(err < 0.05, "x·y decoded at rmse {err}");
    }

    /// The factorised connection equals the full matrix: `W a` and `gain · e · T · (D a) / r`
    /// agree to rounding for a random transform, and the transformed decoders are `T d_i`.
    #[test]
    fn the_full_weight_matrix_is_the_factorised_product() {
        let mut rng = Rng::new(13);
        let pre = Ensemble::new(&EnsembleSpec::default_for(60, 2, 21)).unwrap();
        // ⛔ A post-population of radius 2, because at radius 1 the `/ radius` in the weights is
        // invisible: dropping it survived the first mutation sweep.
        let mut post_spec = EnsembleSpec::default_for(40, 3, 22);
        post_spec.radius = 2.0;
        let post = Ensemble::new(&post_spec).unwrap();
        assert_eq!(post.radius, 2.0);
        let dec = pre.identity_decoders(300, 0.05, &mut rng).unwrap();
        // T is 3 x 2.
        let t = [1.0, 0.5, -2.0, 0.25, 0.0, 3.0];
        let w = full_weights(&dec, &t, &post).unwrap();
        assert_eq!(w.len(), 40 * 60);
        let x = [0.3, -0.6];
        let a = pre.rates(&x).unwrap();
        let x_hat = pre.decode(&dec, &a).unwrap();
        let tx: Vec<f64> = (0..3).map(|r| t[r * 2] * x_hat[0] + t[r * 2 + 1] * x_hat[1]).collect();
        for j in 0..40 {
            let via_w: f64 = (0..60).map(|i| w[j * 60 + i] * a[i]).sum();
            let e = post.encoder(j);
            let via_factors = post.gains[j] * (e[0] * tx[0] + e[1] * tx[1] + e[2] * tx[2]) / post.radius;
            assert!((via_w - via_factors).abs() <= 1e-9 * via_factors.abs().max(1.0), "neuron {j}: {via_w} vs {via_factors}");
        }
        let td = dec.transformed(&t, 3).unwrap();
        assert_eq!(td.out_dim, 3);
        assert!((td.d[5 * 3 + 1] - (t[2] * dec.d[5 * 2] + t[3] * dec.d[5 * 2 + 1])).abs() < 1e-15);
        assert!(matches!(dec.transformed(&t, 4), Err(NefError::Dimension { .. })));
    }

    /// Traffic: full against factorised per tick, and the break-even, against literals.
    #[test]
    fn factorised_traffic_crosses_the_full_matrix_where_the_closed_form_says() {
        assert_eq!(connection_traffic(10, 1000, 4), (10_000, 40 + 4_000));
        assert_eq!(connection_traffic(0, 1000, 4), (0, 4_000), "the encoder pass is paid on a silent tick");
        assert_eq!(connection_traffic(1000, 1000, 4), (1_000_000, 8_000));
        let be = factorisation_break_even(1000, 4).unwrap();
        assert!((be - 4000.0 / 996.0).abs() < 1e-12);
        let (f, g) = connection_traffic(be.ceil() as u64, 1000, 4);
        assert!(g < f);
        let (f, g) = connection_traffic(be.floor() as u64, 1000, 4);
        assert!(g >= f);
        assert_eq!(factorisation_break_even(4, 4), None);
        assert_eq!(factorisation_break_even(0, 1), None);
    }

    // ---- dynamics ----

    /// Principle 3 against literals, and the synapse against its exponential.
    #[test]
    fn the_dynamics_transform_and_the_synapse_are_their_formulas() {
        let a = [0.0, 1.0, -1.0, 0.0];
        let b = [2.0, 0.0];
        let (a2, b2) = dynamics_transform(&a, &b, 2, 0.1).unwrap();
        assert_eq!(a2, vec![1.0, 0.1, -0.1, 1.0]);
        assert_eq!(b2, vec![0.2, 0.0]);
        assert!(matches!(dynamics_transform(&a, &b, 3, 0.1), Err(NefError::Dimension { what: "A", .. })));
        assert!(matches!(dynamics_transform(&a, &[1.0, 2.0, 3.0], 2, 0.1), Err(NefError::Dimension { what: "B", .. })));
        assert!(matches!(dynamics_transform(&a, &b, 2, 0.0), Err(NefError::OutOfRange { .. })));
        let mut s = Lowpass::new(0.05).unwrap();
        let mut y = 0.0;
        for _ in 0..50 {
            y = s.step(1e-3, 1.0);
        }
        assert!((y - (1.0 - (-1.0f64).exp())).abs() < 1e-12, "after one time constant: {y}");
    }

    /// An integrator built by principle 3: a unit input for one second leaves the state at one,
    /// and it holds there for another second. The tolerances are the representational error of
    /// 500 neurons at radius 1.5, measured and stated, not chosen to pass.
    #[test]
    fn a_neural_integrator_integrates_and_holds() {
        let mut rng = Rng::new(14);
        let mut spec = EnsembleSpec::default_for(500, 1, 31);
        spec.radius = 1.5;
        let ens = Ensemble::new(&spec).unwrap();
        let dec = ens.identity_decoders(800, 0.02, &mut rng).unwrap();
        let tau = 0.1;
        let mut lp = RateLoop::new(ens, dec, &[0.0], &[1.0], tau).unwrap();
        let dt = 1e-3;
        for _ in 0..1000 {
            lp.step(dt, &[1.0]).unwrap();
        }
        let after_ramp = lp.x_hat[0];
        assert!((after_ramp - 1.0).abs() < 0.05, "integrated a unit input for 1 s to {after_ramp}");
        for _ in 0..1000 {
            lp.step(dt, &[0.0]).unwrap();
        }
        let held = lp.x_hat[0];
        assert!((held - after_ramp).abs() < 0.05, "drifted from {after_ramp} to {held} in 1 s");
        // And an integrator without the +I is a leaky filter that forgets: the control that shows
        // principle 3 is doing something.
        let ens = Ensemble::new(&spec).unwrap();
        let dec = ens.identity_decoders(800, 0.02, &mut rng).unwrap();
        let mut leaky = RateLoop::new(ens, dec, &[-1.0 / tau], &[1.0], tau).unwrap();
        for _ in 0..1000 {
            leaky.step(dt, &[1.0]).unwrap();
        }
        let forgot = leaky.x_hat[0];
        for _ in 0..1000 {
            leaky.step(dt, &[0.0]).unwrap();
        }
        assert!(leaky.x_hat[0].abs() < 0.2 * forgot.abs() + 0.02, "A = −1/τ should forget: {}", leaky.x_hat[0]);
    }

    /// A two-dimensional oscillator at 1 Hz: after a kick, the decoded state goes round at the
    /// frequency the matrix sets, measured by zero crossings, and does not collapse.
    #[test]
    fn a_neural_oscillator_holds_its_frequency() {
        let mut rng = Rng::new(15);
        let mut spec = EnsembleSpec::default_for(1200, 2, 41);
        spec.radius = 1.2;
        let ens = Ensemble::new(&spec).unwrap();
        let dec = ens.identity_decoders(2000, 0.02, &mut rng).unwrap();
        let omega = core::f64::consts::TAU;
        let a = [0.0, omega, -omega, 0.0];
        let mut lp = RateLoop::new(ens, dec, &a, &[0.0, 0.0], 0.1).unwrap();
        lp.set_state(&[0.8, 0.0]).unwrap();
        let dt = 1e-3;
        let mut trace = Vec::with_capacity(4000);
        for _ in 0..4000 {
            trace.push(lp.step(dt, &[0.0]).unwrap()[0]);
        }
        let mut ups = Vec::new();
        for i in 1..trace.len() {
            if trace[i - 1] <= 0.0 && trace[i] > 0.0 {
                ups.push(i as f64 * dt);
            }
        }
        assert!(ups.len() >= 3, "fewer than three cycles in 4 s: {ups:?}");
        let period = (ups[ups.len() - 1] - ups[0]) / (ups.len() - 1) as f64;
        assert!((period - 1.0).abs() < 0.05, "period {period} s against 1 s");
        let late = trace[3000..].iter().fold(0.0f64, |m, v| m.max(v.abs()));
        assert!(late > 0.3, "the oscillation collapsed to amplitude {late}");
        assert!(matches!(lp.step(dt, &[1.0, 1.0]), Err(NefError::Dimension { what: "u", got: 2, want: 1 })));
    }

    // ---- spiking ----

    /// The spiking population fires at the rates the curve says, to within the interval
    /// quantisation of a 2 s window, and its spike-count decode lands on the rate-mode decode.
    #[test]
    fn spiking_cells_fire_at_the_rate_curve_and_decode_the_same_value() {
        let mut rng = Rng::new(16);
        let ens = Ensemble::new(&EnsembleSpec::default_for(200, 1, 51)).unwrap();
        let dec = ens.identity_decoders(400, 0.01, &mut rng).unwrap();
        let mut sp = SpikingEnsemble::new(ens.clone(), Lif::default()).unwrap();
        let x = [0.4];
        let dt = 1e-4;
        let seconds = 2.0;
        for _ in 0..(seconds / dt) as usize {
            sp.step(dt, &x).unwrap();
        }
        let measured = sp.measured_rates(dt).unwrap();
        let predicted = ens.rates(&x).unwrap();
        let mut worst = 0.0f64;
        for (i, (m, p)) in measured.iter().zip(&predicted).enumerate() {
            // One spike per window of quantisation plus the tick discretisation of the interval.
            let tol = 1.0 / seconds + p * dt / (1.0 / p.max(1.0)) + 1.0;
            assert!((m - p).abs() <= tol, "neuron {i}: measured {m} Hz, predicted {p} Hz (tol {tol})");
            worst = worst.max((m - p).abs());
        }
        assert!(worst > 0.0, "every rate matched exactly, which a 2 s count cannot do");
        let x_rate = ens.decode(&dec, &predicted).unwrap()[0];
        let x_spike = ens.decode(&dec, &measured).unwrap()[0];
        assert!((x_rate - 0.4).abs() < 0.02, "rate decode {x_rate}");
        assert!((x_spike - x_rate).abs() < 0.02, "spike decode {x_spike} vs rate decode {x_rate}");
        sp.reset();
        assert_eq!(sp.measured_rates(dt), None);
        assert!(sp.counts.iter().all(|&c| c == 0));
        // With EXACT spike timing the only error left is the count's own quantisation — one spike
        // in the window, 0.5 Hz — at a tick a hundred times coarser, 10 ms, where several spikes
        // fall in one tick and the tick-based step could not fire faster than 100 Hz at all.
        let coarse = 10e-3;
        for _ in 0..(seconds / coarse) as usize {
            let fired = sp.step_exact(coarse, &x).unwrap();
            assert_eq!(fired.len(), 200);
        }
        let exact = sp.measured_rates(coarse).unwrap();
        assert!(predicted.iter().any(|p| *p > 150.0), "no cell is fast enough to need more than one spike a tick");
        for (i, (m, p)) in exact.iter().zip(&predicted).enumerate() {
            assert!((m - p).abs() <= 1.0 / seconds + 1e-9, "neuron {i}: {m} Hz against {p} Hz with exact timing");
        }
        assert!(matches!(sp.step_exact(0.0, &x), Err(NefError::OutOfRange { what: "dt", .. })));
        assert!(matches!(sp.step_exact(coarse, &[0.1, 0.2]), Err(NefError::Dimension { what: "x", .. })));
        // A prototype whose constants do not match the tuning is refused.
        let other = Lif { tau_m: 10e-3, ..Lif::default() };
        assert!(matches!(SpikingEnsemble::new(ens, other), Err(NefError::OutOfRange { .. })));
    }

    /// The spiking LMU: a spiking population running the Legendre Memory Unit's `A` and `B` holds
    /// the last `θ` seconds of its input — better with more neurons, down to the rate-mode floor,
    /// PROVIDED its spikes are timed exactly rather than rounded to the tick.
    #[test]
    fn a_spiking_population_runs_the_legendre_memory_unit() {
        use crate::resonate::Lmu;
        let (order, theta, tau) = (3usize, 0.5, 0.05);
        let ideal = Lmu::new(order, theta).unwrap();
        // Returns (relative error of the represented state, RMS error of the delayed readout Σ m_j
        // for the network, the same for the ideal unit, spikes, relative error of the readout).
        let run = |n: usize, dt: f64, exact: bool| {
            let mut rng = Rng::new(27);
            let mut spec = EnsembleSpec::default_for(n, order, 91);
            spec.radius = 1.5;
            let ens = Ensemble::new(&spec).unwrap();
            let dec = ens.identity_decoders(3000, 0.05, &mut rng).unwrap();
            let pop = SpikingEnsemble::new(ens, Lif::default()).unwrap();
            let mut net = SpikingLoop::new(pop, dec, &ideal.a, &ideal.b, tau).unwrap();
            assert!(net.exact_timing, "exact spike timing is the default");
            net.exact_timing = exact;
            let mut reference = Lmu::new(order, theta).unwrap();
            let (mut err, mut power, mut read, mut read_ideal, mut count) = (0.0, 0.0, 0.0, 0.0, 0.0);
            // The READOUT is the decoded spikes through one more synapse, so it trails the state by
            // about τ; it is compared with the reference as it was τ ago.
            let lag = (tau / dt) as usize;
            let mut past_states: Vec<Vec<f64>> = Vec::new();
            let mut readout_err = 0.0;
            for k in 0..(4.0 / dt) as usize {
                let t = k as f64 * dt;
                let u = 0.8 * (core::f64::consts::TAU * t).sin();
                reference.step(dt, u).unwrap();
                past_states.push(reference.m.clone());
                let x_hat = net.step(dt, &[u]).unwrap().to_vec();
                if t >= 1.0 {
                    for j in 0..order {
                        readout_err += (x_hat[j] - past_states[k - lag][j]).powi(2);
                    }
                    let past = 0.8 * (core::f64::consts::TAU * (t - theta)).sin();
                    let state: Vec<f64> = net.synapses.iter().map(|s| s.y).collect();
                    for j in 0..order {
                        err += (state[j] - reference.m[j]).powi(2);
                        power += reference.m[j].powi(2);
                    }
                    read += (state.iter().sum::<f64>() - past).powi(2);
                    read_ideal += (reference.m.iter().sum::<f64>() - past).powi(2);
                    count += 1.0;
                }
            }
            ((err / power).sqrt(), (read / count).sqrt(), (read_ideal / count).sqrt(), net.spikes, (readout_err / power).sqrt())
        };
        // Every number below is MEASURED at a 1 ms tick; the bounds leave each a factor of about 1.5.
        let (few, _, _, _, _) = run(60, 1e-3, true);
        let (some, read, read_ideal, spikes, readout) = run(300, 1e-3, true);
        let (many, _, _, _, _) = run(1500, 1e-3, true);
        // More neurons, less error — all the way down: 0.267 → 0.087 → 0.0172, and the last is the
        // error of the same network in RATE mode (0.0173), which is the floor the mapping sets.
        assert!(some < 0.5 * few && many < 0.5 * some, "{few} → {some} → {many}");
        assert!(some < 0.13 && many < 0.026, "{some} {many}");
        // An order-3 unit is itself only an approximation of a delay: 0.045 RMS on an amplitude of
        // 0.8. The spiking network reads the past back through that plus its own noise.
        assert!(read_ideal > 0.02 && read_ideal < 0.06, "{read_ideal}");
        assert!(read < 0.15, "the delayed input is read back with RMS error {read}");
        assert!(spikes > 50_000, "the population barely fired: {spikes} spikes");
        assert!(readout < 0.25, "the readout is {readout} of the state's size away from it");
        // ⛔ The same network WITHOUT exact spike timing, at the same tick: five times worse
        // (measured 0.467), with 11% of its spikes missing. A cell firing at 300 Hz has an interval
        // of 3.3 ticks, which a tick-based step rounds up to 4, so every rate is biased low — and
        // the bias is systematic, so more neurons do not remove it. The first draft of this test
        // blamed the mapping; rate mode, which has no ticks to round, showed the mapping was fine,
        // and `Lif::step_exact` is the repair.
        let (ticked, _, _, ticked_spikes, _) = run(300, 1e-3, false);
        assert!(ticked > 4.0 * some, "tick-based at 1 ms: {ticked} against {some}");
        assert!((ticked_spikes as f64) < 0.92 * spikes as f64, "{ticked_spikes} against {spikes}");
        // With exact timing the tick hardly matters: a TENTH of the tick changes the spike count
        // by a part in a thousand.
        // And a FIVE-millisecond tick, in which the faster cells fire twice, still works (measured
        // 0.123) and still counts every spike — a burst decoded or counted as one spike would not.
        let (burst, _, _, burst_spikes, _) = run(300, 5e-3, true);
        assert!(burst < 0.19, "at a 5 ms tick: {burst}");
        assert!((burst_spikes as f64 / spikes as f64 - 1.0).abs() < 0.01, "{burst_spikes} against {spikes}");
        let (fine, _, _, fine_spikes, _) = run(300, 1e-4, true);
        assert!((fine_spikes as f64 / spikes as f64 - 1.0).abs() < 0.01, "{fine_spikes} against {spikes}");
        assert!(fine < 0.13, "{fine}");
        // Silence in, (near) silence out, and the refusals.
        let mut rng = Rng::new(2);
        let ens = Ensemble::new(&EnsembleSpec::default_for(200, 1, 5)).unwrap();
        let dec = ens.identity_decoders(500, 0.05, &mut rng).unwrap();
        let pop = SpikingEnsemble::new(ens, Lif::default()).unwrap();
        let mut still = SpikingLoop::new(pop.clone(), dec.clone(), &[-10.0], &[10.0], 0.05).unwrap();
        for _ in 0..2000 {
            still.step(1e-3, &[0.0]).unwrap();
        }
        assert!(still.x_hat[0].abs() < 0.1, "a leaky loop with no input sits at {}", still.x_hat[0]);
        assert!(matches!(still.step(1e-3, &[0.0, 0.0]), Err(NefError::Dimension { what: "u", .. })));
        assert!(matches!(still.step(1e-3, &[f64::NAN]), Err(NefError::NonFinite { what: "u", .. })));
        assert!(matches!(still.step(0.0, &[0.0]), Err(NefError::OutOfRange { what: "dt", .. })));
        let two = Decoders { d: vec![0.0; 400], n: 200, out_dim: 2 };
        assert!(matches!(SpikingLoop::new(pop, two, &[-10.0], &[10.0], 0.05), Err(NefError::Dimension { what: "decoder outputs", .. })));
    }

    /// PES on a fixed input is a geometric sequence with a ratio you can compute beforehand, and
    /// its stability limit is the learning rate at which that ratio is −1.
    #[test]
    fn pes_contracts_the_error_by_exactly_its_closed_form() {
        let ens = Ensemble::new(&EnsembleSpec::default_for(40, 2, 31)).unwrap();
        let rates = ens.rates(&[0.3, -0.5]).unwrap();
        let active = rates.iter().filter(|a| **a != 0.0).count();
        assert!(active > 5 && active < 40, "{active} of 40 active: the silent-neuron path needs both kinds");
        let limit = Pes::stability_limit(&rates).unwrap();
        assert!((limit - 80.0 / rates.iter().map(|a| a * a).sum::<f64>()).abs() < 1e-18);
        let pes = Pes::new(0.25 * limit).unwrap();
        assert!((pes.contraction(&rates) - 0.5).abs() < 1e-12, "a quarter of the limit halves the error");
        let mut dec = Decoders { d: vec![0.0; 80], n: 40, out_dim: 2 };
        let target = [0.7, -0.2];
        for k in 0..20 {
            let step = pes.update(&mut dec, &rates, &target).unwrap();
            // From zero decoders the first error is −target, and each update halves it.
            for j in 0..2 {
                let want = -target[j] * 0.5f64.powi(k);
                assert!((step.error[j] - want).abs() < 1e-12 * target[j].abs(), "update {k}, output {j}: {} vs {want}", step.error[j]);
            }
            assert_eq!(step.touched, 2 * active);
        }
        // A silent neuron's decoders were never written.
        for (i, &a) in rates.iter().enumerate() {
            if a == 0.0 {
                assert_eq!(dec.d[2 * i..2 * i + 2], [0.0, 0.0]);
            }
        }
        // AT the limit the error alternates at constant size; past it, it grows.
        let edge = Pes::new(limit).unwrap();
        assert!((edge.contraction(&rates) + 1.0).abs() < 1e-12);
        let mut dec = Decoders { d: vec![0.0; 80], n: 40, out_dim: 2 };
        let errors: Vec<f64> = (0..6).map(|_| edge.update(&mut dec, &rates, &target).unwrap().error[0]).collect();
        for pair in errors.windows(2) {
            assert!((pair[0] + pair[1]).abs() < 1e-9, "{errors:?}");
        }
        let over = Pes::new(1.5 * limit).unwrap();
        let mut dec = Decoders { d: vec![0.0; 80], n: 40, out_dim: 2 };
        let first = over.update(&mut dec, &rates, &target).unwrap().error[0].abs();
        let mut last = first;
        for _ in 0..5 {
            last = over.update(&mut dec, &rates, &target).unwrap().error[0].abs();
        }
        assert!((last / first - 2.0f64.powi(5)).abs() < 1e-6, "past the limit the error doubles each update: {first} → {last}");
        assert_eq!(Pes::stability_limit(&[0.0, 0.0]), None);
        assert_eq!(Pes::stability_limit(&[]), None);
    }

    /// The published PES carries no `1/n` — `Δd_i = κ E a_i` in Bekolay, Kolbeck and Eliasmith
    /// 2013 Eq. 6, `Δd_i = κ v_c a_i` in `MacNeil` and Eliasmith 2011 Eq. 16 — and this rule's `κ` is
    /// normalised per neuron, as Nengo's is. So a [`Pes`] built with `κ = n·κ_p` writes exactly the
    /// papers' step on `κ_p` (with the papers' error read as `target − x̂`, the sign that descends;
    /// here `E = x̂ − target` and the minus is in the rule), its contraction is the papers'
    /// `1 − κ_p|a|²`, and the papers' stability limit is [`Pes::stability_limit`] over `n`,
    /// `2/|a|²`. The rates, decoders, targets and `κ_p` are binary fractions, so every product in
    /// the step and the contraction is exact; the limit `8/21` is not, but `n = 4` is a power of
    /// two, so dividing it by `n` lands exactly on the nearest double to `2/21`, and every
    /// comparison is an equality.
    #[test]
    fn the_papers_unnormalised_rate_is_this_rules_kappa_over_n() {
        let rates = [2.0, 0.0, 4.0, 1.0];
        let n = rates.len() as f64;
        let energy: f64 = rates.iter().map(|a| a * a).sum();
        assert_eq!(energy, 21.0);
        let kappa_paper = 0.03125;
        let pes = Pes::new(n * kappa_paper).unwrap();
        let mut dec = Decoders { d: vec![0.5, -0.25, 0.0, 0.75, 0.125, 0.0, -0.5, 0.25], n: 4, out_dim: 2 };
        let before = dec.d.clone();
        let target = [0.5, -1.0];
        let step = pes.update(&mut dec, &rates, &target).unwrap();
        // x̂ = Σ a_i d_i = (1, −0.25), so the error x̂ − target is (0.5, 0.75).
        assert_eq!(step.error, vec![0.5, 0.75]);
        for i in 0..4 {
            for j in 0..2 {
                let paper = before[i * 2 + j] - kappa_paper * step.error[j] * rates[i];
                assert_eq!(dec.d[i * 2 + j], paper, "neuron {i}, output {j}");
            }
        }
        assert_eq!(pes.contraction(&rates), 1.0 - kappa_paper * energy);
        assert_eq!(Pes::stability_limit(&rates).unwrap() / n, 2.0 / energy);
    }

    /// Over a sample set PES approaches the least-squares decoders' error from above: least
    /// squares is the optimum on that set, so PES can match it and cannot beat it.
    #[test]
    fn pes_learns_toward_the_least_squares_optimum_and_not_past_it() {
        let ens = Ensemble::new(&EnsembleSpec::default_for(60, 1, 77)).unwrap();
        let mut rng = Rng::new(3);
        let samples = ens.sample_points(200, &mut rng);
        let targets: Vec<Vec<f64>> = samples.iter().map(|x| vec![x[0] * x[0]]).collect();
        let optimum = ens.decoders(&samples, &targets, 0.0).unwrap();
        let floor = ens.rmse(&optimum, &samples, &targets).unwrap();
        let all_rates: Vec<Vec<f64>> = samples.iter().map(|x| ens.rates(x).unwrap()).collect();
        let tightest = all_rates.iter().filter_map(|a| Pes::stability_limit(a)).fold(f64::INFINITY, f64::min);
        let pes = Pes::new(0.5 * tightest).unwrap();
        let mut learned = Decoders { d: vec![0.0; 60], n: 60, out_dim: 1 };
        let untrained = ens.rmse(&learned, &samples, &targets).unwrap();
        let mut at = Vec::new();
        for sweep in 1..=400 {
            for (a, t) in all_rates.iter().zip(&targets) {
                pes.update(&mut learned, a, t).unwrap();
            }
            if sweep == 100 || sweep == 400 {
                at.push(ens.rmse(&learned, &samples, &targets).unwrap());
            }
        }
        let (early, late) = (at[0], at[1]);
        assert!(late >= floor * (1.0 - 1e-9), "PES beat the least-squares optimum: {late} < {floor}");
        assert!(early < 0.05 * untrained, "a hundred sweeps took the error from {untrained} only to {early}");
        // It keeps closing on the optimum and does so SLOWLY — least-mean-squares converges along
        // each direction of the activity covariance at that direction's own rate, and the weak
        // ones take long. Measured here: 3.6× the optimum after 400 sweeps. The first draft of this
        // test asserted "within 3×", a number with nothing behind it, and failed on it.
        assert!(late < early, "sweeps 100 → 400 did not improve: {early} → {late}");
        assert!(late > 1.5 * floor, "PES is already at the optimum ({late} vs {floor}): the slow-direction remark above is stale");
    }

    /// Every refusal names the problem.
    #[test]
    fn the_refusals_name_the_problem() {
        assert!(matches!(Pes::new(0.0), Err(NefError::OutOfRange { what: "kappa", .. })));
        assert!(matches!(Pes::new(f64::NAN), Err(NefError::OutOfRange { what: "kappa", .. })));
        let pes = Pes::new(1e-6).unwrap();
        let mut two = Decoders { d: vec![0.0; 4], n: 2, out_dim: 2 };
        assert!(matches!(pes.update(&mut two, &[1.0], &[0.0, 0.0]), Err(NefError::Dimension { what: "rates", .. })));
        assert!(matches!(pes.update(&mut two, &[1.0, 1.0], &[0.0]), Err(NefError::Dimension { what: "target", .. })));
        assert!(matches!(pes.update(&mut two, &[1.0, f64::NAN], &[0.0, 0.0]), Err(NefError::NonFinite { what: "rates", .. })));
        assert!(matches!(pes.update(&mut two, &[1.0, 1.0], &[0.0, f64::INFINITY]), Err(NefError::NonFinite { what: "target", .. })));
        assert_eq!(two.d, vec![0.0; 4], "a refused update wrote nothing");
        let spec = EnsembleSpec::default_for(0, 1, 1);
        assert!(matches!(Ensemble::new(&spec), Err(NefError::Empty { what: "neurons" })));
        let spec = EnsembleSpec::default_for(10, 0, 1);
        assert!(matches!(Ensemble::new(&spec), Err(NefError::Empty { what: "dimension" })));
        let mut spec = EnsembleSpec::default_for(10, 1, 1);
        spec.radius = 0.0;
        assert!(matches!(Ensemble::new(&spec), Err(NefError::OutOfRange { what: "radius", .. })));
        let mut spec = EnsembleSpec::default_for(10, 1, 1);
        spec.intercept = (-1.0, 0.5);
        assert!(matches!(Ensemble::new(&spec), Err(NefError::OutOfRange { what: "intercept range", .. })));
        let mut spec = EnsembleSpec::default_for(10, 1, 1);
        spec.max_rate = (100.0, 600.0);
        assert!(matches!(Ensemble::new(&spec), Err(NefError::OutOfRange { what: "max_rate range", .. })));
        let ens = Ensemble::new(&EnsembleSpec::default_for(10, 2, 1)).unwrap();
        assert!(matches!(ens.rates(&[1.0]), Err(NefError::Dimension { what: "x", got: 1, want: 2 })));
        assert!(matches!(ens.rates(&[1.0, f64::NAN]), Err(NefError::NonFinite { what: "x", index: 1 })));
        assert!(matches!(ens.decoders(&[], &[], 0.1), Err(NefError::Empty { what: "samples" })));
        assert!(matches!(ens.decoders(&[vec![0.0, 0.0]], &[vec![0.0]], -1.0), Err(NefError::NonFinite { what: "lambda", .. })));
        let wrong = Decoders { d: vec![0.0; 5], n: 5, out_dim: 1 };
        assert!(matches!(ens.decode(&wrong, &[0.0; 10]), Err(NefError::Dimension { what: "decoders", .. })));
        assert!(matches!(Lowpass::new(-1.0), Err(NefError::OutOfRange { .. })));
        assert!(matches!(
            Ensemble::from_parts(1, 1.0, LifRate::default(), vec![1.0], vec![1.0], vec![1.0, 2.0]),
            Err(NefError::Dimension { what: "biases", .. })
        ));
        // Plain least squares on a population with two identical neurons is singular, and says so.
        let curve = LifRate::default();
        let (g, b) = curve.gain_bias(300.0, 0.0).unwrap();
        let twins = Ensemble::from_parts(1, 1.0, curve, vec![1.0, 1.0], vec![g, g], vec![b, b]).unwrap();
        let mut rng = Rng::new(1);
        assert!(matches!(twins.identity_decoders(50, 0.0, &mut rng), Err(NefError::Solve(_))));
        assert!(twins.identity_decoders(50, 0.1, &mut rng).is_ok(), "the ridge makes it solvable");
        for e in [
            NefError::Empty { what: "x" },
            NefError::Dimension { what: "y", got: 1, want: 2 },
            NefError::NonFinite { what: "z", index: 3 },
            NefError::OutOfRange { what: "w", value: 9.0, low: 0.0, high: 1.0 },
        ] {
            assert!(!e.to_string().is_empty());
        }
    }


    /// The second mutation sweep accepted a maximum rate AT the refractory bound: nothing had asked
    /// for one. The bound itself is a rate no current reaches, so the spec refuses it by name.
    #[test]
    fn a_maximum_rate_at_the_refractory_bound_is_refused_by_the_spec() {
        let mut spec = EnsembleSpec::default_for(20, 1, 5);
        let bound = spec.neuron.max_rate_bound();
        spec.max_rate = (0.5 * bound, bound);
        assert!(matches!(Ensemble::new(&spec), Err(NefError::OutOfRange { what: "max_rate range", .. })));
        spec.max_rate = (0.5 * bound, 0.99 * bound);
        assert!(Ensemble::new(&spec).is_ok(), "just inside the bound is a legal population");
    }

    // ---- the guards (mutation repair, 0.20.0) ----

    /// Every time constant of a rate curve must be strictly positive AND finite, and so must
    /// every other parameter this module range-checks against an infinite upper bound.
    ///
    /// Pins the two `in_range` calls in [`LifRate::new`] and the `is_finite()` clause of
    /// `in_range` itself. Why the suite could not see it: the only refusals asserted anywhere
    /// were a zero radius, a zero `kappa`, a negative `tau` and a zero `dt`, and every one of
    /// those is already refused by the `value >= low` clause alone — so an `in_range` that had
    /// lost its finiteness test still refused every value the suite ever handed it, and the two
    /// constants of `LifRate::new` were never given a bad one at all. An INFINITY is the value
    /// that separates the clauses: where `high` is `f64::INFINITY` it satisfies
    /// `value >= low && value <= high` and only `is_finite()` stops it.
    #[test]
    fn a_time_constant_must_be_strictly_positive_and_finite() {
        for bad in [0.0, -1.0, -20e-3, f64::NAN, f64::INFINITY] {
            assert!(
                matches!(LifRate::new(bad, 2e-3), Err(NefError::OutOfRange { what: "tau_rc", .. })),
                "tau_rc = {bad} was accepted"
            );
            assert!(
                matches!(LifRate::new(20e-3, bad), Err(NefError::OutOfRange { what: "tau_ref", .. })),
                "tau_ref = {bad} was accepted"
            );
        }
        assert!(LifRate::new(f64::MIN_POSITIVE, f64::MIN_POSITIVE).is_ok(), "the smallest normal is legal");
        assert!(
            matches!(
                LifRate::of_lif(&Lif { tau_m: f64::INFINITY, ..Lif::default() }),
                Err(NefError::OutOfRange { what: "tau_rc", .. })
            ),
            "of_lif goes through the same gate"
        );
        // The same clause is the only thing guarding these four.
        assert!(matches!(Lowpass::new(f64::INFINITY), Err(NefError::OutOfRange { what: "tau", .. })));
        let mut spec = EnsembleSpec::default_for(4, 1, 1);
        spec.radius = f64::INFINITY;
        assert!(matches!(Ensemble::new(&spec), Err(NefError::OutOfRange { what: "radius", .. })));
        assert!(matches!(
            Ensemble::from_parts(1, f64::INFINITY, LifRate::default(), vec![1.0], vec![1.0], vec![0.0]),
            Err(NefError::OutOfRange { what: "radius", .. })
        ));
        let ens = Ensemble::new(&EnsembleSpec::default_for(4, 1, 1)).unwrap();
        assert!(matches!(
            SpikingEnsemble::new(ens, Lif { r_m: f64::INFINITY, ..Lif::default() }),
            Err(NefError::OutOfRange { what: "r_m", .. })
        ));
    }

    /// A tuning range given the wrong way round is refused, and a degenerate one is not.
    ///
    /// Pins the `rlo <= rhi` and `ilo <= ihi` clauses of [`Ensemble::new`]. Why the suite could
    /// not see it: an inverted range is not a value any fixture supplies, and without the clause
    /// the draw `rlo + (rhi − rlo)·u` walks DOWNWARD from `rlo`, so a population asked for
    /// `[400, 200]` gets maximum rates below 200 — outside both numbers the caller named, with no
    /// error and nothing in the suite reading a population's rates against the range it asked for.
    /// The second half is the other side of the same clause: `<=`, not `<`, so a population every
    /// cell of which has the same maximum rate is still legal, and is built.
    #[test]
    fn an_inverted_tuning_range_is_refused_and_a_degenerate_one_is_not() {
        let mut spec = EnsembleSpec::default_for(50, 1, 3);
        spec.max_rate = (400.0, 200.0);
        assert!(matches!(Ensemble::new(&spec), Err(NefError::OutOfRange { what: "max_rate range", .. })));
        let mut spec = EnsembleSpec::default_for(50, 1, 3);
        spec.intercept = (0.5, -0.5);
        assert!(matches!(Ensemble::new(&spec), Err(NefError::OutOfRange { what: "intercept range", .. })));
        let mut spec = EnsembleSpec::default_for(50, 1, 3);
        spec.max_rate = (300.0, 300.0);
        spec.intercept = (0.25, 0.25);
        let ens = Ensemble::new(&spec).unwrap();
        for i in 0..ens.n() {
            let x: Vec<f64> = ens.encoder(i).iter().map(|c| c * ens.radius).collect();
            let top = ens.rates(&x).unwrap()[i];
            assert!((top - 300.0).abs() < 1e-9, "neuron {i} tops out at {top} Hz, not 300");
            assert_eq!(ens.rates(&[0.25 * ens.radius * ens.encoder(i)[0]]).unwrap()[i], 0.0, "silent at its intercept");
        }
    }

    /// A population built from explicit parts refuses a non-finite encoder, gain or bias.
    ///
    /// Pins all three `finite` calls in [`Ensemble::from_parts`]. Why the suite could not see the
    /// middle one: the only `from_parts` refusal asserted anywhere is a `biases` LENGTH, and every
    /// other call site builds its gains from [`LifRate::gain_bias`], which cannot return a
    /// non-finite one for an intercept and rate it has itself accepted. A NaN gain makes every
    /// current NaN and every rate silently zero, which no assertion in the suite distinguishes
    /// from a population that is merely quiet.
    #[test]
    fn a_population_built_from_parts_refuses_a_non_finite_entry() {
        let curve = LifRate::default();
        let build = |e: Vec<f64>, g: Vec<f64>, b: Vec<f64>| Ensemble::from_parts(2, 1.0, curve, e, g, b);
        assert!(build(vec![1.0, 0.0, 0.0, 1.0], vec![1.0, 1.0], vec![0.0, 0.0]).is_ok());
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(
                matches!(
                    build(vec![1.0, 0.0, bad, 1.0], vec![1.0, 1.0], vec![0.0, 0.0]),
                    Err(NefError::NonFinite { what: "encoders", index: 2 })
                ),
                "encoder {bad} was accepted"
            );
            assert!(
                matches!(
                    build(vec![1.0, 0.0, 0.0, 1.0], vec![1.0, bad], vec![0.0, 0.0]),
                    Err(NefError::NonFinite { what: "gains", index: 1 })
                ),
                "gain {bad} was accepted"
            );
            assert!(
                matches!(
                    build(vec![1.0, 0.0, 0.0, 1.0], vec![1.0, 1.0], vec![bad, 0.0]),
                    Err(NefError::NonFinite { what: "biases", index: 0 })
                ),
                "bias {bad} was accepted"
            );
        }
    }

    /// A spiking population refuses a cell whose threshold does not sit strictly above rest.
    ///
    /// Pins the `v_th − v_rest` guard in [`SpikingEnsemble::new`]. Why the suite could not see it:
    /// the only prototypes ever offered are [`Lif::default`] and one with a mismatched `tau_m`,
    /// and the guard's job is to keep `amps_per_unit` — the volts-per-unit-current scale — away
    /// from zero. A cell with `v_th = v_rest` is accepted without it, gets an `amps_per_unit` of
    /// exactly zero, and then never fires whatever it is asked to represent.
    #[test]
    fn a_cell_whose_threshold_is_not_above_rest_is_refused() {
        let ens = Ensemble::new(&EnsembleSpec::default_for(8, 1, 1)).unwrap();
        for v_th in [-65e-3, -70e-3] {
            let proto = Lif { v_th, v_rest: -65e-3, v_reset: -65e-3, ..Lif::default() };
            assert!(
                matches!(
                    SpikingEnsemble::new(ens.clone(), proto),
                    Err(NefError::OutOfRange { what: "v_th − v_rest", .. })
                ),
                "a threshold at {v_th} V against a rest of -0.065 V was accepted"
            );
        }
        assert!(matches!(
            SpikingEnsemble::new(ens.clone(), Lif { r_m: 0.0, ..Lif::default() }),
            Err(NefError::OutOfRange { what: "r_m", .. })
        ));
        let ok = SpikingEnsemble::new(ens, Lif::default()).unwrap();
        assert_eq!(ok.amps_per_unit, (Lif::default().v_th - Lif::default().v_rest) / Lif::default().r_m);
        assert!(ok.amps_per_unit > 0.0);
    }

    // ---- what the population is made of ----

    /// [`Ensemble::encoder`] returns neuron `i`'s own ROW of the row-major encoder matrix.
    ///
    /// Pins the row stride. Why the suite could not see it: `encoder(i)` is only ever read to
    /// rebuild something the same population computed from the same rows — the factorised weight
    /// check reads `post.encoder(j)` and compares it with `full_weights`, which reads it too, so
    /// a shared stride error cancels — and for a one-dimensional population the two strides are
    /// the same expression. Read at a one-element stride, row `i` is a window sliding across two
    /// adjacent neurons' encoders, which is no longer a unit vector.
    #[test]
    fn an_encoder_is_the_neurons_own_row_of_the_encoder_matrix() {
        let curve = LifRate::default();
        let (g, b) = curve.gain_bias(300.0, 0.0).unwrap();
        let rows = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.6, 0.0, 0.8];
        let ens = Ensemble::from_parts(3, 1.0, curve, rows, vec![g; 3], vec![b; 3]).unwrap();
        assert_eq!(ens.encoder(0).to_vec(), vec![1.0, 0.0, 0.0]);
        assert_eq!(ens.encoder(1).to_vec(), vec![0.0, 1.0, 0.0]);
        assert_eq!(ens.encoder(2).to_vec(), vec![0.6, 0.0, 0.8]);
        // Neuron 1's preferred direction is the second axis, so that is where its current peaks.
        assert!(ens.current(1, &[0.0, 1.0, 0.0]).unwrap() > ens.current(1, &[1.0, 0.0, 0.0]).unwrap());
        // And a drawn population's rows are unit vectors ONE ROW AT A TIME.
        let drawn = Ensemble::new(&EnsembleSpec::default_for(50, 4, 9)).unwrap();
        for i in 0..drawn.n() {
            let norm = drawn.encoder(i).iter().map(|x| x * x).sum::<f64>().sqrt();
            assert!((norm - 1.0).abs() < 1e-15, "neuron {i}'s encoder has norm {norm}");
        }
    }

    /// The Box-Muller draw behind the encoders and the sample points is a STANDARD normal:
    /// mean zero, variance one.
    ///
    /// Pins `normal`'s own documented distribution. Why the suite could not see it: both callers
    /// divide the drawn vector by its own norm, so a constant scale on the draw cancels to within
    /// rounding. Measured here over two hundred thousand draws per dimension, halving the variance
    /// leaves 54.5% of the normalised coordinates BIT-IDENTICAL and moves the rest by at most 6
    /// ulps, the largest absolute difference at any dimension from 2 to 4 being 3.33e-16. Chasing
    /// the factor through the encoders would therefore pin floating-point noise, which is a change
    /// detector; the function's own property is where a factor of two is a factor of two.
    ///
    /// The bounds are the sample size: for `N = 100_000` standard normals the standard error of
    /// the mean is `1/sqrt(N) = 3.2e-3` and of the variance `sqrt(2/N) = 4.5e-3`, so 0.05 and 0.10
    /// are about 15 and 22 standard errors — loose enough for any generator, and five times inside
    /// the variance of 1/2 that a draw of `sqrt(−ln u)` rather than `sqrt(−2 ln u)` would have.
    #[test]
    fn the_standard_normal_draw_has_mean_zero_and_variance_one() {
        const N: usize = 100_000;
        for seed in [1u64, 7, 99] {
            let mut rng = Rng::new(seed);
            let (mut sum, mut sum_sq) = (0.0f64, 0.0f64);
            for _ in 0..N {
                let z = normal(&mut rng);
                sum += z;
                sum_sq += z * z;
            }
            let mean = sum / N as f64;
            let var = sum_sq / N as f64 - mean * mean;
            assert!(mean.abs() < 0.05, "seed {seed}: mean {mean}");
            assert!((var - 1.0).abs() < 0.10, "seed {seed}: variance {var}");
        }
    }

    /// The maximum rates of a population are DRAWN from across the range the spec names, not
    /// handed out at one end of it.
    ///
    /// A cell's maximum rate is its rate at its own preferred direction on the represented ball,
    /// so the whole distribution is readable through the public surface. Why the suite could not
    /// see it: every existing assertion asks whether a rate lies INSIDE the range, and the top of
    /// a range is inside it — giving every neuron `rhi` passes all of them while collapsing the
    /// population's heterogeneity, which is the thing the representation's error depends on.
    #[test]
    fn a_populations_maximum_rates_are_drawn_from_across_the_range() {
        let spec = EnsembleSpec::default_for(200, 1, 7);
        let (rlo, rhi) = spec.max_rate;
        let width = rhi - rlo;
        let ens = Ensemble::new(&spec).unwrap();
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for i in 0..ens.n() {
            let x: Vec<f64> = ens.encoder(i).iter().map(|c| c * ens.radius).collect();
            let top = ens.rates(&x).unwrap()[i];
            assert!(top >= rlo - 1e-6 && top <= rhi + 1e-6, "neuron {i} tops out at {top} Hz, outside {rlo}..{rhi}");
            lo = lo.min(top);
            hi = hi.max(top);
        }
        // 200 independent uniform draws: the chance that none lands in the bottom quarter of the
        // range is 0.75^200 = 1e-25, and the same at the top.
        assert!(lo < rlo + 0.25 * width, "no cell near the bottom of the range: lowest maximum rate {lo}");
        assert!(hi > rhi - 0.25 * width, "no cell near the top of the range: highest maximum rate {hi}");
        assert!(hi - lo > 0.5 * width, "the drawn maxima span {} Hz of {width}", hi - lo);
    }

    /// [`Ensemble::sample_points`] draws UNIFORMLY from the represented ball — the right radius
    /// and the right radial law.
    ///
    /// Why the suite could not see it: the sample points are only ever used as inputs to a least
    /// squares and then to the error it is measured by, and both are computed from the same
    /// points, so a population trained and tested on points that all hug the centre still reports
    /// a small error. Nothing read where the points actually were.
    #[test]
    fn sample_points_are_uniform_in_the_represented_ball() {
        let mut spec = EnsembleSpec::default_for(4, 3, 1);
        spec.radius = 2.5;
        let ens = Ensemble::new(&spec).unwrap();
        let mut rng = Rng::new(5);
        const N: usize = 20_000;
        let mut sum = 0.0f64;
        let mut max = 0.0f64;
        for p in ens.sample_points(N, &mut rng) {
            let r = p.iter().map(|v| v * v).sum::<f64>().sqrt();
            // The norm is recomputed from the rounded coordinates, so it can sit a few ulps
            // (4.4e-16 relative) above the scale that produced it; 1e-12 is far inside that.
            assert!(r <= ens.radius * (1.0 + 1e-12), "a sample at |x| = {r} is outside the ball of radius 2.5");
            sum += r;
            max = max.max(r);
        }
        let mean = sum / N as f64;
        // Uniform in a d-ball of radius R: |x|/R has density d·u^(d−1), so E|x| = R·d/(d+1) =
        // 1.875 and sd|x| = R·sqrt(d/(d+2) − (d/(d+1))²) = 0.484. Over N = 20_000 the standard
        // error of the mean is 3.4e-3, so 0.025 is about seven of them. Drawing the radius without
        // the u^(1/d) correction puts the mean at R/2 = 1.25; dropping the radius puts it at 0.75.
        assert!((mean - 1.875).abs() < 0.025, "mean |x| = {mean}, not R·d/(d+1) = 1.875");
        assert!(max > 0.99 * ens.radius, "no sample reached the surface: the largest was {max}");
    }

    /// [`EnsembleSpec::default_for`] is the spec the documentation states, radius included.
    ///
    /// Why the suite could not see the radius: every test that depends on a particular radius sets
    /// it, and every test that does not uses the radius only as the unit its own sample points and
    /// tolerances are already expressed in — so doubling the default rescales the whole fixture
    /// and changes nothing any assertion reads. The radius is the unit of the represented ball and
    /// of the intercepts, so it has to be pinned where it is written down.
    #[test]
    fn the_default_spec_is_the_one_the_documentation_states() {
        let spec = EnsembleSpec::default_for(64, 3, 9);
        assert_eq!(spec.n, 64);
        assert_eq!(spec.dim, 3);
        assert_eq!(spec.seed, 9);
        assert_eq!(spec.radius, 1.0, "Nengo's default radius is one");
        assert_eq!(spec.max_rate, (200.0, 400.0));
        // Nengo 2.x's `Uniform(-1.0, 1.0)`, inset to the open interval `Ensemble::new` accepts. It
        // is NOT current Nengo's: v3.0.0 onward ship `Uniform(-1.0, 0.9)`, and the doc says so.
        assert_eq!(spec.intercept, (-0.999, 0.999));
        assert_eq!(spec.neuron, LifRate::new(20e-3, 2e-3).unwrap());
        // And the default population really does represent the UNIT ball: its own sample points
        // land inside it, and it tops out at the edge of it.
        let ens = Ensemble::new(&EnsembleSpec::default_for(40, 1, 9)).unwrap();
        assert_eq!(ens.radius, 1.0);
        let mut rng = Rng::new(2);
        for p in ens.sample_points(300, &mut rng) {
            assert!(p[0].abs() <= 1.0, "a default population samples {} , outside the unit ball", p[0]);
        }
        for i in 0..ens.n() {
            let top = ens.rates(&[ens.encoder(i)[0]]).unwrap()[i];
            assert!(top >= 200.0 - 1e-6, "neuron {i} reaches only {top} Hz at x = ±1");
        }
    }

    /// The way [`EnsembleSpec::default_for`] says to follow Nengo ≥ 3.0 works. Nengo's default
    /// intercepts are `Uniform(-1.0, 0.9)` in `nengo/ensemble.py` from v3.0.0 on, and the doc says
    /// to set `intercept` to `(-0.999, 0.9)`. That spec builds; every neuron's intercept, read back
    /// from its gain and bias as `(1 − J^bias)/α`, lies in that range; and the reason Nengo's
    /// `CHANGES.rst` gives for the change, "to avoid high gains when intercepts are close to 1",
    /// shows here too. With the same seed the two specs make the same draws, so each neuron's
    /// intercept can only move down, and `α = (x − 1)/(1 − intercept)` with it: no neuron's gain
    /// rises, the largest falls, and none exceeds `(x(400 Hz) − 1)/(1 − 0.9)`.
    #[test]
    fn nengo_3_intercepts_are_one_field_away_and_lower_the_gains() {
        let old = EnsembleSpec::default_for(200, 2, 13);
        let mut new = old;
        new.intercept = (-0.999, 0.9);
        let a = Ensemble::new(&old).unwrap();
        let b = Ensemble::new(&new).unwrap();
        assert_eq!(a.encoders, b.encoders, "the intercept range must not change the encoder draws");
        let x_top = old.neuron.current_for_rate(400.0).unwrap();
        let cap = (x_top - 1.0) / (1.0 - 0.9);
        let mut hi = [0.0f64; 2];
        for i in 0..200 {
            let c = (1.0 - b.biases[i]) / b.gains[i];
            assert!((-0.999 - 1e-12..=0.9 + 1e-12).contains(&c), "neuron {i}: intercept {c}");
            assert!(b.gains[i] <= a.gains[i], "neuron {i}: gain rose from {} to {}", a.gains[i], b.gains[i]);
            assert!(b.gains[i] <= cap * (1.0 + 1e-12), "neuron {i}: gain {} above {cap}", b.gains[i]);
            hi[0] = hi[0].max(a.gains[i]);
            hi[1] = hi[1].max(b.gains[i]);
        }
        assert!(hi[1] < hi[0], "the largest gain did not fall: {} then {}", hi[0], hi[1]);
    }

    // ---- what the numbers mean ----

    /// [`Ensemble::rmse`] is a ROOT mean square, averaged over every scalar compared and not over
    /// the samples they came in.
    ///
    /// Zero decoders decode zero whatever the rates are, so the error is exactly the target and
    /// the answer can be written down: four scalars, `3² + 4² + 0 + 0 = 25`, over four of them is
    /// 6.25, whose root is 2.5. A mean square would report 6.25 and a mean over the two SAMPLES
    /// would report sqrt(12.5) = 3.5355. Why the suite could not see either: every use of `rmse`
    /// is an inequality against a number below one, where squaring makes the value SMALLER and
    /// every `rmse < bound` still passes, and every comparison between two of them is a ratio, in
    /// which a missing root and a constant factor both survive.
    #[test]
    fn the_decoding_error_is_a_root_mean_square_over_every_scalar_compared() {
        let curve = LifRate::default();
        let (g, b) = curve.gain_bias(300.0, 0.0).unwrap();
        let ens = Ensemble::from_parts(1, 1.0, curve, vec![1.0], vec![g], vec![b]).unwrap();
        let zero = Decoders { d: vec![0.0, 0.0], n: 1, out_dim: 2 };
        let samples = vec![vec![0.3], vec![0.7]];
        let targets = vec![vec![3.0, 4.0], vec![0.0, 0.0]];
        assert_eq!(ens.decode(&zero, &ens.rates(&samples[0]).unwrap()).unwrap(), vec![0.0, 0.0]);
        assert_eq!(ens.rmse(&zero, &samples, &targets).unwrap(), 2.5);
        // Doubling every error doubles the root mean square, which a mean square would quadruple.
        let doubled: Vec<Vec<f64>> = targets.iter().map(|t| t.iter().map(|v| 2.0 * v).collect()).collect();
        assert_eq!(ens.rmse(&zero, &samples, &doubled).unwrap(), 5.0);
    }

    /// [`Pes::contraction`] of a population with no neurons is one: nothing is learned and nothing
    /// is forgotten.
    ///
    /// Pins the `.max(1)` on the divisor. Why the suite could not see it: the rule is only ever
    /// asked about populations of 40 and 60 neurons, and an empty one divides zero energy by zero
    /// neurons, which is NaN — and `assert!(x < bound)` on a NaN is the one comparison that is
    /// false for every bound, so the defect only shows where the value is read for equality.
    /// [`Pes::stability_limit`] already answers `None` for the same population.
    #[test]
    fn the_contraction_of_a_population_with_no_neurons_is_one() {
        let pes = Pes::new(1e-3).unwrap();
        assert_eq!(pes.contraction(&[]), 1.0);
        assert_eq!(pes.contraction(&[0.0, 0.0]), 1.0, "a silent population contracts nothing either");
        assert_eq!(Pes::stability_limit(&[]), None);
        // And for a population that does fire it is the closed form, retyped.
        let rates = [3.0, 4.0];
        assert_eq!(pes.contraction(&rates), 1.0 - 1e-3 * 25.0 / 2.0);
    }

    // ---- the loops ----

    /// The rate loop applies `A′` ROW by row: `ẋ_i` is set by row `i` of the matrix.
    ///
    /// The system here is `ẋ₀ = 0`, `ẋ₁ = 2x₀` — a value held on the first coordinate and ramped
    /// onto the second. Transposed it becomes `ẋ₀ = 2x₁`, `ẋ₁ = 0`, which holds the first
    /// coordinate and never moves the second at all. Why the suite could not see it: the only
    /// two-dimensional system it runs is the oscillator `[[0, ω], [−ω, 0]]`, whose transpose is
    /// the SAME oscillator run backwards — and the assertions on it are the period between zero
    /// crossings and the surviving amplitude, both of which are invariant under time reversal.
    #[test]
    fn the_rate_loop_applies_the_recurrent_matrix_row_by_row() {
        let mut rng = Rng::new(19);
        let mut spec = EnsembleSpec::default_for(500, 2, 63);
        spec.radius = 2.0;
        let ens = Ensemble::new(&spec).unwrap();
        let dec = ens.identity_decoders(1500, 0.02, &mut rng).unwrap();
        let a = [0.0, 0.0, 2.0, 0.0];
        let mut lp = RateLoop::new(ens, dec, &a, &[0.0, 0.0], 0.1).unwrap();
        assert_eq!(lp.a_prime, vec![1.0, 0.0, 0.2, 1.0]);
        lp.set_state(&[1.0, 0.0]).unwrap();
        for _ in 0..500 {
            lp.step(1e-3, &[0.0]).unwrap();
        }
        let held = lp.x_hat[0];
        let ramped = lp.x_hat[1];
        assert!((held - 1.0).abs() < 0.15, "the first coordinate is held by row 0 of A′: {held}");
        assert!((ramped - 1.0).abs() < 0.2, "row 1 ramps the second coordinate to 2·1·0.5 s = 1: {ramped}");
    }

    /// The spiking loop's readout filters the DECODED SPIKES, not the recurrent state.
    ///
    /// Two pins. First: a population whose decoders are all zero decodes nothing, so the readout
    /// stays exactly zero however hard the recurrent synapse is driven — and it is driven here,
    /// to within a fifth of its input. Second: after one tick from rest the readout is the pulse
    /// through one low-pass, computed with the same operations in the same order, so the
    /// comparison is exact. Why the suite could not see it: in a converged loop the recurrent
    /// synapse's output and the decoded pulse both approximate the represented state, so filtering
    /// the state instead of the spikes changes the readout by about one extra time constant of
    /// lag — and the only assertion on the readout compares it with the reference delayed by
    /// exactly one time constant, which that extra lag flatters rather than breaks.
    #[test]
    fn the_spiking_loops_readout_filters_the_decoded_spikes() {
        let mut rng = Rng::new(23);
        let ens = Ensemble::new(&EnsembleSpec::default_for(120, 1, 61)).unwrap();
        let dec = ens.identity_decoders(400, 0.02, &mut rng).unwrap();
        let pop = SpikingEnsemble::new(ens, Lif::default()).unwrap();
        let (tau, dt) = (0.05, 1e-3);

        let silent = Decoders { d: vec![0.0; 120], n: 120, out_dim: 1 };
        let mut mute = SpikingLoop::new(pop.clone(), silent, &[0.0], &[20.0], tau).unwrap();
        for _ in 0..400 {
            mute.step(dt, &[1.0]).unwrap();
        }
        assert!(mute.spikes > 1_000, "the population did not fire at all: {} spikes", mute.spikes);
        assert!(mute.synapses[0].y > 0.5, "the recurrent synapse was not driven: {}", mute.synapses[0].y);
        assert_eq!(mute.x_hat[0], 0.0, "zero decoders decode nothing, so the readout has nothing to filter");

        let mut net = SpikingLoop::new(pop, dec.clone(), &[3.0], &[100.0], tau).unwrap();
        net.synapses[0].y = 0.6;
        net.step(dt, &[1.0]).unwrap();
        let mut pulse = 0.0f64;
        for (i, &c) in net.population.counts.iter().enumerate() {
            if c > 0 {
                pulse += c as f64 * dec.d[i] / dt;
            }
        }
        assert!(pulse.abs() > 0.05, "nothing was decoded on the first tick: {pulse}");
        let decay = (-dt / tau).exp();
        assert_eq!(net.x_hat[0], pulse + (0.0 - pulse) * decay);
    }

    /// [`SpikingEnsemble::reset`] returns every membrane to rest and clears every refractory
    /// countdown, not just the spike counters.
    ///
    /// Why the suite could not see it: the one `reset` in the suite is followed by assertions on
    /// `counts` and `measured_rates`, which the counter half of the method already satisfies, and
    /// then by a run long enough (2 s) for the leftover membrane state to wash out of the rates it
    /// measures. A reset that leaves the cells charged makes the next run's first interspike
    /// interval depend on the run before it.
    #[test]
    fn a_reset_returns_every_membrane_to_rest() {
        let ens = Ensemble::new(&EnsembleSpec::default_for(40, 1, 1)).unwrap();
        let mut sp = SpikingEnsemble::new(ens, Lif::default()).unwrap();
        for _ in 0..50 {
            sp.step(1e-3, &[0.9]).unwrap();
        }
        assert!(sp.cells.iter().any(|c| c.v != c.v_rest), "no cell charged: the reset would have nothing to undo");
        assert!(sp.cells.iter().any(|c| c.refractory > 0.0), "no cell is refractory");
        sp.reset();
        for (i, c) in sp.cells.iter().enumerate() {
            assert_eq!(c.v, c.v_rest, "cell {i} was left at {} V", c.v);
            assert_eq!(c.refractory, 0.0, "cell {i} was left refractory");
        }
        assert!(sp.counts.iter().all(|&c| c == 0));
        assert_eq!(sp.ticks, 0);
        // A reset population is a fresh one: the same drive from here reproduces the same spikes.
        let mut counts = Vec::new();
        for _ in 0..2 {
            for _ in 0..50 {
                sp.step(1e-3, &[0.9]).unwrap();
            }
            counts.push(sp.counts.clone());
            sp.reset();
        }
        assert_eq!(counts[0], counts[1], "the second run after a reset differed from the first");
    }
}
