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
//!
//! # What this module has NOT reproduced
//!
//! - Spaun, or any model beyond one or two populations. The framework composes; this module gives
//!   the pieces and checks each, and stops there.
//! - Nengo's exact random-number stream: encoders, intercepts and maximum rates are drawn from
//!   this crate's generator, so a population built here with the same seed as a Nengo model is
//!   *statistically* the same population and not the same neurons.
//! - Learning rules on the decoders (PES). Those belong beside [`crate::plasticity`] when they
//!   come.

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
    /// Nengo's defaults for a population of `n` neurons in `dim` dimensions: radius 1, maximum
    /// rates 200–400 Hz, intercepts uniform in `(−1, 1)`.
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
        Decoders, Ensemble, EnsembleSpec, LifRate, Lowpass, NefError, RateLoop, SpikingEnsemble,
        connection_traffic, dynamics_transform, factorisation_break_even, full_weights,
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
        // A prototype whose constants do not match the tuning is refused.
        let other = Lif { tau_m: 10e-3, ..Lif::default() };
        assert!(matches!(SpikingEnsemble::new(ens, other), Err(NefError::OutOfRange { .. })));
    }

    /// Every refusal names the problem.
    #[test]
    fn the_refusals_name_the_problem() {
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
}
