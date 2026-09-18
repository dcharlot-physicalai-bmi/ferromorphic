//! Predictive coding: inference as the relaxation of prediction-error neurons, learning as the
//! product of an error and the activity next to it — checked against the Bayesian posterior it
//! computes and against the backpropagated gradient it approximates.
//!
//! # What the mechanism is
//!
//! A predictive coding network (Rao and Ballard, *Predictive coding in the visual cortex: a
//! functional interpretation of some extra-classical receptive-field effects*, Nature Neuroscience
//! 2(1):79–87, 1999) holds two kinds of unit at every level: **value** units that carry the current
//! estimate of a cause, and **error** units that carry the difference between a value and the
//! prediction of it from the level above. Inference is a relaxation: every value unit moves to
//! reduce the errors it takes part in, which is gradient descent on the free energy
//! `F = ½ Σ (error)² / variance`. Learning is local: a weight changes by the product of the error
//! on its output side and the activity on its input side, and that product IS `−∂F/∂W`.
//!
//! Three models are here, from the one with a complete closed form to the one that is only
//! approximately something else:
//!
//! - [`LinearGaussian`] — one layer, `y = W x + noise`, Gaussian prior on `x`. The relaxation's
//!   fixed point is the posterior mean, which is a linear solve.
//! - [`TutorialNeuron`] — the scalar non-linear example of Bogacz (*A tutorial on the free-energy
//!   framework for modelling perception and learning*, Journal of Mathematical Psychology
//!   76:198–211, 2017), with error units that compute their own division by the variance.
//! - [`Network`] — a multi-layer network with `tanh` units, whose weight updates converge on the
//!   backpropagated gradient as the output is clamped more WEAKLY (Whittington and Bogacz, *An
//!   approximation of the error backpropagation algorithm in a predictive coding network with
//!   local Hebbian synaptic plasticity*, Neural Computation 29(5):1229–1262, 2017).
//!
//! # Why it is in a neuromorphic crate
//!
//! Backpropagation needs a second, global pass that carries derivatives backwards through the
//! same weights. Predictive coding gets the same gradient from quantities that are present AT the
//! synapse — the error unit on one side, the value unit on the other — after a relaxation that is
//! exactly the kind of analogue settling a physical substrate does for free. What it does not
//! remove is the symmetric feedback weight: the value units still read the errors above them
//! through `Wᵀ`. That is stated here because it is the part a chip still has to arrange.
//!
//! # The closed forms this module is checked against
//!
//! - **Posterior mean.** The fixed point of the relaxation is the solution of
//!   `(WᵀW/σ_y² + I/σ_p²) x = Wᵀy/σ_y² + μ/σ_p²` ([`LinearGaussian::map`], by Cholesky), and for
//!   one latent and one observation with `w = 1` it is the precision-weighted average
//!   `(σ_p² y + σ_y² μ)/(σ_p² + σ_y²)`.
//! - **Discrete convergence.** For a scalar model, `k` Euler steps of `dt` leave exactly
//!   `(1 − dt·H)^k` of the initial error, `H = w²/σ_y² + 1/σ_p²`.
//! - **Descent.** Euler at [`LinearGaussian::stable_dt`] never raises the free energy, and a step
//!   far past `2/L` does — so the monitor that reports it can fire.
//! - **The learning rule is the gradient.** [`LinearGaussian::weight_gradient`] equals `−∂F/∂W` by
//!   central differences.
//! - **The tutorial's number.** With `v_p = 3`, `Σ_p = Σ_u = 1`, `u = 2` and `g(v) = v²` the
//!   posterior mode is the real root of `2φ³ − 3φ − 3 = 0`, `φ = 1.56747` — the tutorial's "about
//!   1.6" — and the error units settle at `(φ − v_p)/Σ_p` and `(u − φ²)/Σ_u`. (The first draft of
//!   this doc had `1.5676`, from a hand bisection; the test's bisection corrected it.)
//! - **Backpropagation in the limit — and WHICH limit.** The first draft of this module claimed
//!   the local updates approach the backpropagated gradient as the output error `δ` shrinks. They
//!   do not, and the test said so: the mismatch fell 1.01-fold for a tenfold smaller `δ`. The
//!   reason is in the one-unit chain, which [`LinearGaussian`] solves: with output variance `Σ`
//!   and generative weight `w` the relaxed output error is `δ/(Σ + w²)`, not `δ/Σ` — the hidden
//!   unit moves to absorb a FIXED FRACTION of the error however small the error is. The limit
//!   that works is a weakly clamped output, `Σ → ∞`: there `Σ ×` the local updates approach the
//!   backpropagated gradient at first order in `1/Σ`, falling tenfold when `Σ` grows tenfold.
//!
//! # What this module has NOT reproduced
//!
//! - A spiking implementation. The units here are rate units with continuous dynamics.
//! - Rao and Ballard's receptive fields or end-stopping results, which need natural images.
//! - Any claim that this removes weight transport — see above; it does not.

use core::fmt;

use crate::reservoir::{ReservoirError, cholesky};
use crate::rng::Rng;

/// The most relaxation steps one call will take; a request past it is refused.
pub const MAX_STEPS: u64 = 10_000_000;

/// What went wrong, named rather than guessed around.
#[derive(Debug, Clone, PartialEq)]
pub enum PredictiveError {
    /// A count of zero where at least one is needed.
    Empty {
        /// What was empty.
        what: &'static str,
    },
    /// Two lengths that had to agree.
    Dimension {
        /// Which array.
        what: &'static str,
        /// Length supplied.
        got: usize,
        /// Length required.
        want: usize,
    },
    /// A parameter outside its range.
    OutOfRange {
        /// Which parameter.
        what: &'static str,
        /// Value supplied.
        value: f64,
        /// Lowest admissible.
        low: f64,
        /// Highest admissible.
        high: f64,
    },
    /// A `NaN` or infinity.
    NonFinite {
        /// Which quantity.
        what: &'static str,
        /// Position in the offending array, `0` for a scalar.
        index: usize,
    },
    /// The posterior's linear solve failed.
    Solve(ReservoirError),
}

impl fmt::Display for PredictiveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty { what } => write!(f, "{what} is empty"),
            Self::Dimension { what, got, want } => write!(f, "{what} has length {got}, expected {want}"),
            Self::OutOfRange { what, value, low, high } => {
                write!(f, "{what} = {value} is outside [{low}, {high}]")
            }
            Self::NonFinite { what, index } => write!(f, "{what} is not finite at {index}"),
            Self::Solve(e) => write!(f, "posterior solve: {e}"),
        }
    }
}

impl std::error::Error for PredictiveError {}

impl From<ReservoirError> for PredictiveError {
    fn from(e: ReservoirError) -> Self {
        Self::Solve(e)
    }
}

fn finite_all(what: &'static str, v: &[f64]) -> Result<(), PredictiveError> {
    if let Some(i) = v.iter().position(|x| !x.is_finite()) {
        return Err(PredictiveError::NonFinite { what, index: i });
    }
    Ok(())
}

fn dims(what: &'static str, got: usize, want: usize) -> Result<(), PredictiveError> {
    if got == want { Ok(()) } else { Err(PredictiveError::Dimension { what, got, want }) }
}

fn positive(what: &'static str, v: f64) -> Result<f64, PredictiveError> {
    if v.is_finite() && v > 0.0 {
        Ok(v)
    } else {
        Err(PredictiveError::OutOfRange { what, value: v, low: f64::MIN_POSITIVE, high: f64::INFINITY })
    }
}

fn step_count(steps: u64) -> Result<u64, PredictiveError> {
    if steps > MAX_STEPS {
        return Err(PredictiveError::OutOfRange { what: "steps", value: steps as f64, low: 0.0, high: MAX_STEPS as f64 });
    }
    Ok(steps)
}

// ---------------------------------------------------------------------------------------------
// One linear Gaussian layer
// ---------------------------------------------------------------------------------------------

/// The generative model `y = W x + noise`, `noise ~ N(0, σ_y² I)`, `x ~ N(μ, σ_p² I)`.
#[derive(Debug, Clone, PartialEq)]
pub struct LinearGaussian {
    /// Observations.
    pub n_obs: usize,
    /// Latent causes.
    pub n_lat: usize,
    /// Generative weights, row-major `n_obs × n_lat`.
    pub w: Vec<f64>,
    /// Observation noise variance `σ_y²`.
    pub var_obs: f64,
    /// Prior variance `σ_p²`.
    pub var_prior: f64,
    /// Prior mean `μ`, length `n_lat`.
    pub prior: Vec<f64>,
}

/// What a relaxation did.
#[derive(Debug, Clone, PartialEq)]
pub struct Relaxation {
    /// The latent estimate it ended on.
    pub x: Vec<f64>,
    /// Free energy there.
    pub free_energy: f64,
    /// The largest single-step rise in free energy; non-positive when every step descended.
    pub worst_increase: f64,
    /// Steps taken.
    pub steps: u64,
}

impl LinearGaussian {
    /// Build.
    ///
    /// # Errors
    ///
    /// [`PredictiveError::Empty`] for a zero dimension, [`PredictiveError::Dimension`] for a wrong
    /// length, [`PredictiveError::NonFinite`] for a bad entry, [`PredictiveError::OutOfRange`] for
    /// a non-positive variance.
    pub fn new(n_obs: usize, n_lat: usize, w: Vec<f64>, var_obs: f64, var_prior: f64, prior: Vec<f64>) -> Result<Self, PredictiveError> {
        if n_obs == 0 {
            return Err(PredictiveError::Empty { what: "observations" });
        }
        if n_lat == 0 {
            return Err(PredictiveError::Empty { what: "latents" });
        }
        dims("w", w.len(), n_obs * n_lat)?;
        dims("prior", prior.len(), n_lat)?;
        finite_all("w", &w)?;
        finite_all("prior", &prior)?;
        let var_obs = positive("var_obs", var_obs)?;
        let var_prior = positive("var_prior", var_prior)?;
        Ok(Self { n_obs, n_lat, w, var_obs, var_prior, prior })
    }

    fn check(&self, x: &[f64], y: &[f64]) -> Result<(), PredictiveError> {
        dims("x", x.len(), self.n_lat)?;
        dims("y", y.len(), self.n_obs)?;
        finite_all("x", x)?;
        finite_all("y", y)
    }

    /// The two error populations at `x` given `y`: `ε_y = (y − Wx)/σ_y²` (length `n_obs`) and
    /// `ε_x = (x − μ)/σ_p²` (length `n_lat`) — each an error divided by its variance, which is
    /// what makes the relaxation weigh them by precision.
    ///
    /// # Errors
    ///
    /// [`PredictiveError::Dimension`] or [`PredictiveError::NonFinite`] for bad `x` or `y`.
    pub fn errors(&self, x: &[f64], y: &[f64]) -> Result<(Vec<f64>, Vec<f64>), PredictiveError> {
        self.check(x, y)?;
        let eps_y = (0..self.n_obs)
            .map(|i| {
                let pred: f64 = (0..self.n_lat).map(|j| self.w[i * self.n_lat + j] * x[j]).sum();
                (y[i] - pred) / self.var_obs
            })
            .collect();
        let eps_x = (0..self.n_lat).map(|j| (x[j] - self.prior[j]) / self.var_prior).collect();
        Ok((eps_y, eps_x))
    }

    /// The free energy `F = |y − Wx|²/(2σ_y²) + |x − μ|²/(2σ_p²)` — the negative log joint, up to
    /// a constant.
    ///
    /// # Errors
    ///
    /// As [`LinearGaussian::errors`].
    pub fn free_energy(&self, x: &[f64], y: &[f64]) -> Result<f64, PredictiveError> {
        let (eps_y, eps_x) = self.errors(x, y)?;
        let obs: f64 = eps_y.iter().map(|e| e * e).sum::<f64>() * self.var_obs;
        let pri: f64 = eps_x.iter().map(|e| e * e).sum::<f64>() * self.var_prior;
        Ok(0.5 * (obs + pri))
    }

    /// The posterior precision `H = WᵀW/σ_y² + I/σ_p²`, row-major `n_lat × n_lat`.
    #[must_use]
    pub fn precision(&self) -> Vec<f64> {
        let n = self.n_lat;
        let mut h = vec![0.0; n * n];
        for a in 0..n {
            for b in 0..n {
                let dot: f64 = (0..self.n_obs).map(|i| self.w[i * n + a] * self.w[i * n + b]).sum();
                h[a * n + b] = dot / self.var_obs;
            }
            h[a * n + a] += 1.0 / self.var_prior;
        }
        h
    }

    /// The posterior mean, by solving `H x = Wᵀy/σ_y² + μ/σ_p²` — the referee the relaxation is
    /// checked against, and what a digital machine would compute instead of relaxing.
    ///
    /// # Errors
    ///
    /// [`PredictiveError::Dimension`] or [`PredictiveError::NonFinite`] for a bad `y`, and
    /// [`PredictiveError::Solve`] if the factorisation fails.
    pub fn map(&self, y: &[f64]) -> Result<Vec<f64>, PredictiveError> {
        dims("y", y.len(), self.n_obs)?;
        finite_all("y", y)?;
        let n = self.n_lat;
        let rhs: Vec<f64> = (0..n)
            .map(|j| {
                let back: f64 = (0..self.n_obs).map(|i| self.w[i * n + j] * y[i]).sum();
                back / self.var_obs + self.prior[j] / self.var_prior
            })
            .collect();
        Ok(cholesky(&self.precision(), n, 1e-14)?.solve(&rhs)?)
    }

    /// A step at which Euler cannot raise the free energy: `1/L`, `L` the largest absolute row sum
    /// of the precision (Gershgorin), which bounds its largest eigenvalue.
    #[must_use]
    pub fn stable_dt(&self) -> f64 {
        let n = self.n_lat;
        let h = self.precision();
        let worst = (0..n).map(|a| h[a * n..(a + 1) * n].iter().map(|v| v.abs()).sum::<f64>()).fold(0.0, f64::max);
        1.0 / worst
    }

    /// One Euler step of the relaxation `dx/dt = Wᵀ ε_y − ε_x`, in place.
    ///
    /// # Errors
    ///
    /// As [`LinearGaussian::errors`], plus [`PredictiveError::OutOfRange`] for a non-positive `dt`.
    pub fn step(&self, x: &mut [f64], y: &[f64], dt: f64) -> Result<(), PredictiveError> {
        let dt = positive("dt", dt)?;
        let (eps_y, eps_x) = self.errors(x, y)?;
        for j in 0..self.n_lat {
            let back: f64 = (0..self.n_obs).map(|i| self.w[i * self.n_lat + j] * eps_y[i]).sum();
            x[j] += dt * (back - eps_x[j]);
        }
        Ok(())
    }

    /// Relax from `x0` for `steps` Euler steps of `dt`.
    ///
    /// # Errors
    ///
    /// As [`LinearGaussian::step`], plus [`PredictiveError::OutOfRange`] past [`MAX_STEPS`].
    pub fn relax(&self, x0: &[f64], y: &[f64], dt: f64, steps: u64) -> Result<Relaxation, PredictiveError> {
        let mut x = x0.to_vec();
        let mut last = self.free_energy(&x, y)?;
        let mut worst = f64::NEG_INFINITY;
        for _ in 0..step_count(steps)? {
            self.step(&mut x, y, dt)?;
            let f = self.free_energy(&x, y)?;
            worst = worst.max(f - last);
            last = f;
        }
        Ok(Relaxation { x, free_energy: last, worst_increase: worst, steps })
    }

    /// The local learning signal `ε_y xᵀ`, row-major `n_obs × n_lat`: the error unit on the
    /// output side times the value unit on the input side. It is `−∂F/∂W`.
    ///
    /// # Errors
    ///
    /// As [`LinearGaussian::errors`].
    pub fn weight_gradient(&self, x: &[f64], y: &[f64]) -> Result<Vec<f64>, PredictiveError> {
        let (eps_y, _) = self.errors(x, y)?;
        let mut g = vec![0.0; self.n_obs * self.n_lat];
        for i in 0..self.n_obs {
            for j in 0..self.n_lat {
                g[i * self.n_lat + j] = eps_y[i] * x[j];
            }
        }
        Ok(g)
    }
}

// ---------------------------------------------------------------------------------------------
// The tutorial's neuron
// ---------------------------------------------------------------------------------------------

/// The scalar model of Bogacz (2017, section 2): a cause `v` with prior `N(v_p, Σ_p)` is seen
/// through `u = g(v) + noise`, `g(v) = v²`, noise variance `Σ_u`. Three units — the estimate `φ`
/// and two error units — relax together, and the error units divide by their variance through
/// their own leak: `dε_p/dt = φ − v_p − Σ_p ε_p`, `dε_u/dt = u − g(φ) − Σ_u ε_u`,
/// `dφ/dt = ε_u g′(φ) − ε_p`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TutorialNeuron {
    /// Prior mean `v_p`.
    pub v_p: f64,
    /// Prior variance `Σ_p`.
    pub var_p: f64,
    /// Sensory noise variance `Σ_u`.
    pub var_u: f64,
    /// The estimate `φ`.
    pub phi: f64,
    /// Prior error unit `ε_p`.
    pub eps_p: f64,
    /// Sensory error unit `ε_u`.
    pub eps_u: f64,
}

impl TutorialNeuron {
    /// Build with the estimate at the prior mean and both error units silent.
    ///
    /// # Errors
    ///
    /// [`PredictiveError::NonFinite`] for a non-finite prior mean, [`PredictiveError::OutOfRange`]
    /// for a non-positive variance.
    pub fn new(v_p: f64, var_p: f64, var_u: f64) -> Result<Self, PredictiveError> {
        if !v_p.is_finite() {
            return Err(PredictiveError::NonFinite { what: "v_p", index: 0 });
        }
        let var_p = positive("var_p", var_p)?;
        let var_u = positive("var_u", var_u)?;
        Ok(Self { v_p, var_p, var_u, phi: v_p, eps_p: 0.0, eps_u: 0.0 })
    }

    /// The slope of the log posterior at `phi` for an observation `u`:
    /// `(v_p − φ)/Σ_p + 2φ (u − φ²)/Σ_u`. Zero at the posterior mode.
    #[must_use]
    pub fn posterior_slope(&self, phi: f64, u: f64) -> f64 {
        (self.v_p - phi) / self.var_p + 2.0 * phi * (u - phi * phi) / self.var_u
    }

    /// One Euler step of the three units under observation `u`.
    ///
    /// # Errors
    ///
    /// [`PredictiveError::OutOfRange`] for a non-positive `dt`, [`PredictiveError::NonFinite`] for
    /// a non-finite `u`.
    pub fn step(&mut self, dt: f64, u: f64) -> Result<(), PredictiveError> {
        let dt = positive("dt", dt)?;
        if !u.is_finite() {
            return Err(PredictiveError::NonFinite { what: "u", index: 0 });
        }
        let d_phi = self.eps_u * 2.0 * self.phi - self.eps_p;
        let d_p = self.phi - self.v_p - self.var_p * self.eps_p;
        let d_u = u - self.phi * self.phi - self.var_u * self.eps_u;
        self.phi += dt * d_phi;
        self.eps_p += dt * d_p;
        self.eps_u += dt * d_u;
        Ok(())
    }
}

// ---------------------------------------------------------------------------------------------
// A multi-layer network and its backpropagation referee
// ---------------------------------------------------------------------------------------------

/// A layered network with predictions `μ_l = W_l tanh(x_{l−1})` and free energy
/// `F = ½ Σ_hidden |x_l − μ_l|² + |x_L − μ_L|²/(2Σ)`. Layer `0` is clamped to the input; during
/// learning the last layer is clamped to the target and the hidden layers relax.
#[derive(Debug, Clone, PartialEq)]
pub struct Network {
    /// Units per layer, input first; at least two layers.
    pub sizes: Vec<usize>,
    /// `w[l]` maps layer `l` to layer `l + 1`, row-major `sizes[l+1] × sizes[l]`.
    pub w: Vec<Vec<f64>>,
    /// Output variance `Σ`: how weakly the target is clamped. The hidden layers have variance 1.
    pub var_out: f64,
}

impl Network {
    /// Build with weights drawn uniformly from `±1/√fan_in`.
    ///
    /// # Errors
    ///
    /// [`PredictiveError::Empty`] for fewer than two layers or an empty layer,
    /// [`PredictiveError::OutOfRange`] for a non-positive output variance.
    pub fn random(sizes: &[usize], var_out: f64, rng: &mut Rng) -> Result<Self, PredictiveError> {
        if sizes.len() < 2 {
            return Err(PredictiveError::Empty { what: "layers (needs two)" });
        }
        if sizes.contains(&0) {
            return Err(PredictiveError::Empty { what: "a layer" });
        }
        let var_out = positive("var_out", var_out)?;
        let w = sizes
            .windows(2)
            .map(|p| {
                let bound = 1.0 / (p[0] as f64).sqrt();
                (0..p[0] * p[1]).map(|_| bound * (2.0 * rng.next_f64() - 1.0)).collect()
            })
            .collect();
        Ok(Self { sizes: sizes.to_vec(), w, var_out })
    }

    fn predict_layer(&self, l: usize, below: &[f64]) -> Vec<f64> {
        let (n_in, n_out) = (self.sizes[l], self.sizes[l + 1]);
        (0..n_out).map(|i| (0..n_in).map(|j| self.w[l][i * n_in + j] * below[j].tanh()).sum()).collect()
    }

    /// The feedforward pass: every layer at its own prediction, so every error is zero. Returns
    /// all layer values, input first.
    ///
    /// # Errors
    ///
    /// [`PredictiveError::Dimension`] or [`PredictiveError::NonFinite`] for a bad input.
    pub fn forward(&self, input: &[f64]) -> Result<Vec<Vec<f64>>, PredictiveError> {
        dims("input", input.len(), self.sizes[0])?;
        finite_all("input", input)?;
        let mut x = vec![input.to_vec()];
        for l in 0..self.w.len() {
            let next = self.predict_layer(l, &x[l]);
            x.push(next);
        }
        Ok(x)
    }

    /// The error units `ε_l = (x_l − μ_l)/variance` for `l = 1..`, indexed from zero (so
    /// `errors[0]` belongs to layer `1`); the variance is 1 except at the output.
    fn errors(&self, x: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let top = self.w.len() - 1;
        (0..self.w.len())
            .map(|l| {
                let var = if l == top { self.var_out } else { 1.0 };
                let mu = self.predict_layer(l, &x[l]);
                x[l + 1].iter().zip(&mu).map(|(v, m)| (v - m) / var).collect()
            })
            .collect()
    }

    /// The free energy of a full state.
    ///
    /// # Errors
    ///
    /// [`PredictiveError::Dimension`] if the state does not match the layer sizes.
    pub fn free_energy(&self, x: &[Vec<f64>]) -> Result<f64, PredictiveError> {
        self.check_state(x)?;
        let top = self.w.len() - 1;
        let eps = self.errors(x);
        // An error unit carries (x − μ)/variance, so its share of F is ½ ε² · variance.
        Ok(0.5 * eps.iter().enumerate().map(|(l, layer)| {
            let var = if l == top { self.var_out } else { 1.0 };
            var * layer.iter().map(|e| e * e).sum::<f64>()
        }).sum::<f64>())
    }

    fn check_state(&self, x: &[Vec<f64>]) -> Result<(), PredictiveError> {
        dims("state (layers)", x.len(), self.sizes.len())?;
        for (l, layer) in x.iter().enumerate() {
            dims("state (layer width)", layer.len(), self.sizes[l])?;
            finite_all("state", layer)?;
        }
        Ok(())
    }

    /// Clamp the input and the target, start the hidden layers at their feedforward values and
    /// relax them by `dx_l/dt = −ε_l + tanh′(x_l) ⊙ (W_{l+1}ᵀ ε_{l+1})` for `steps` Euler steps.
    /// Returns the relaxed state and the largest hidden-unit movement on the final step.
    ///
    /// # Errors
    ///
    /// As [`Network::forward`], plus [`PredictiveError::Dimension`] for a wrong target length and
    /// [`PredictiveError::OutOfRange`] for a bad `dt` or too many steps.
    pub fn relax(&self, input: &[f64], target: &[f64], dt: f64, steps: u64) -> Result<(Vec<Vec<f64>>, f64), PredictiveError> {
        let dt = positive("dt", dt)?;
        let mut x = self.forward(input)?;
        let last = self.sizes.len() - 1;
        dims("target", target.len(), self.sizes[last])?;
        finite_all("target", target)?;
        x[last] = target.to_vec();
        let mut moved = 0.0f64;
        for _ in 0..step_count(steps)? {
            let eps = self.errors(&x);
            moved = 0.0;
            for l in 1..last {
                let n_here = self.sizes[l];
                let n_above = self.sizes[l + 1];
                for j in 0..n_here {
                    let back: f64 = (0..n_above).map(|i| self.w[l][i * n_here + j] * eps[l][i]).sum();
                    let t = x[l][j].tanh();
                    let dx = dt * (-eps[l - 1][j] + (1.0 - t * t) * back);
                    x[l][j] += dx;
                    moved = moved.max(dx.abs());
                }
            }
        }
        Ok((x, moved))
    }

    /// The local weight updates `ε_{l+1} tanh(x_l)ᵀ` at a given state, one matrix per weight
    /// layer, shaped like [`Network::w`]. This is `−∂F/∂W`.
    ///
    /// # Errors
    ///
    /// [`PredictiveError::Dimension`] if the state does not match the layer sizes.
    pub fn local_updates(&self, x: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, PredictiveError> {
        self.check_state(x)?;
        let eps = self.errors(x);
        Ok((0..self.w.len())
            .map(|l| {
                let n_in = self.sizes[l];
                let mut g = vec![0.0; self.sizes[l + 1] * n_in];
                for (i, e) in eps[l].iter().enumerate() {
                    for j in 0..n_in {
                        g[i * n_in + j] = e * x[l][j].tanh();
                    }
                }
                g
            })
            .collect())
    }

    /// The backpropagated descent direction `−∂ℒ/∂W` of `ℒ = ½ |target − output|²`, the referee
    /// [`Network::local_updates`] is compared with. A global backward pass: `δ_L = target − ŷ`,
    /// `δ_l = tanh′(x̂_l) ⊙ W_{l+1}ᵀ δ_{l+1}`.
    ///
    /// # Errors
    ///
    /// As [`Network::forward`], plus [`PredictiveError::Dimension`] for a wrong target length.
    pub fn backprop(&self, input: &[f64], target: &[f64]) -> Result<Vec<Vec<f64>>, PredictiveError> {
        let x = self.forward(input)?;
        let last = self.sizes.len() - 1;
        dims("target", target.len(), self.sizes[last])?;
        finite_all("target", target)?;
        let mut delta: Vec<f64> = target.iter().zip(&x[last]).map(|(t, y)| t - y).collect();
        let mut grads = vec![Vec::new(); self.w.len()];
        for l in (0..self.w.len()).rev() {
            let n_in = self.sizes[l];
            let mut g = vec![0.0; self.sizes[l + 1] * n_in];
            for (i, d) in delta.iter().enumerate() {
                for j in 0..n_in {
                    g[i * n_in + j] = d * x[l][j].tanh();
                }
            }
            grads[l] = g;
            if l > 0 {
                delta = (0..n_in)
                    .map(|j| {
                        let back: f64 = delta.iter().enumerate().map(|(i, d)| self.w[l][i * n_in + j] * d).sum();
                        let t = x[l][j].tanh();
                        (1.0 - t * t) * back
                    })
                    .collect();
            }
        }
        Ok(grads)
    }
}

/// The relative mismatch `|a − b| / |b|` between two sets of weight matrices, Frobenius norms over
/// every layer at once. `None` when the shapes differ or `b` is all zero.
#[must_use]
pub fn relative_mismatch(a: &[Vec<f64>], b: &[Vec<f64>]) -> Option<f64> {
    if a.len() != b.len() || a.iter().zip(b).any(|(p, q)| p.len() != q.len()) {
        return None;
    }
    let diff: f64 = a.iter().flatten().zip(b.iter().flatten()).map(|(p, q)| (p - q) * (p - q)).sum();
    let norm: f64 = b.iter().flatten().map(|q| q * q).sum();
    if norm == 0.0 { None } else { Some((diff / norm).sqrt()) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> LinearGaussian {
        // Three observations of two causes.
        LinearGaussian::new(3, 2, vec![1.0, 0.5, -0.5, 1.0, 0.25, 0.75], 0.5, 2.0, vec![0.3, -0.2]).unwrap()
    }

    #[test]
    fn the_relaxation_settles_on_the_posterior_mean() {
        let m = model();
        let y = [1.2, -0.4, 0.9];
        let want = m.map(&y).unwrap();
        // The referee satisfies the normal equations it claims to solve.
        let h = m.precision();
        for a in 0..2 {
            let lhs: f64 = (0..2).map(|b| h[a * 2 + b] * want[b]).sum();
            let rhs: f64 = (0..3).map(|i| m.w[i * 2 + a] * y[i]).sum::<f64>() / 0.5 + m.prior[a] / 2.0;
            assert!((lhs - rhs).abs() < 1e-12);
        }
        let run = m.relax(&[0.0, 0.0], &y, m.stable_dt(), 400).unwrap();
        for (got, w) in run.x.iter().zip(&want) {
            assert!((got - w).abs() < 1e-12, "relaxed to {got}, posterior mean {w}");
        }
        // At the fixed point the value units are still: Wᵀ ε_y = ε_x.
        let (eps_y, eps_x) = m.errors(&run.x, &y).unwrap();
        for j in 0..2 {
            let back: f64 = (0..3).map(|i| m.w[i * 2 + j] * eps_y[i]).sum();
            assert!((back - eps_x[j]).abs() < 1e-11);
        }
        assert!(eps_y.iter().any(|e| e.abs() > 0.1), "the errors are not all zero at the posterior: that is the point");
    }

    #[test]
    fn a_scalar_posterior_is_the_precision_weighted_average() {
        let m = LinearGaussian::new(1, 1, vec![1.0], 0.5, 2.0, vec![1.0]).unwrap();
        let want = (2.0 * 3.0 + 0.5 * 1.0) / (2.0 + 0.5);
        assert!((m.map(&[3.0]).unwrap()[0] - want).abs() < 1e-15);
        // k Euler steps leave exactly (1 − dt·H)^k of the initial error; H = 1/0.5 + 1/2 = 2.5.
        assert_eq!(m.precision(), vec![2.5]);
        assert_eq!(m.stable_dt(), 0.4);
        let dt = 0.1;
        let mut x = [0.0];
        for _ in 0..7 {
            m.step(&mut x, &[3.0], dt).unwrap();
        }
        let left = (x[0] - want) / (0.0 - want);
        assert!((left - 0.75f64.powi(7)).abs() < 1e-14, "{left} of the error is left, (1 − 0.25)^7 = {}", 0.75f64.powi(7));
        // A generative weight that is not 1: (σ_p² w y + σ_y² μ)/(w² σ_p² + σ_y²).
        let scaled = LinearGaussian::new(1, 1, vec![2.0], 0.5, 2.0, vec![1.0]).unwrap();
        assert!((scaled.map(&[3.0]).unwrap()[0] - (2.0 * 2.0 * 3.0 + 0.5) / (4.0 * 2.0 + 0.5)).abs() < 1e-15);
    }

    #[test]
    fn euler_at_the_stable_step_descends_and_the_monitor_can_fire() {
        let m = model();
        let y = [1.2, -0.4, 0.9];
        let start = m.free_energy(&[2.0, -2.0], &y).unwrap();
        let run = m.relax(&[2.0, -2.0], &y, m.stable_dt(), 300).unwrap();
        // F is a sum of five squares of size ≤ start; two evaluations of a settled state differ by
        // at most a few ulps of it.
        assert!(run.worst_increase <= 8.0 * f64::EPSILON * start, "free energy rose by {}", run.worst_increase);
        assert!(run.free_energy < start);
        assert_eq!(run.steps, 300);
        // The minimum of F is at the posterior mean, and nowhere lower.
        let at_map = m.free_energy(&m.map(&y).unwrap(), &y).unwrap();
        assert!((run.free_energy - at_map).abs() < 1e-12);
        let rough = m.relax(&[2.0, -2.0], &y, 10.0 * m.stable_dt(), 20).unwrap();
        assert!(rough.worst_increase > 1.0, "a step of 10/L never raised the free energy ({})", rough.worst_increase);
    }

    #[test]
    fn the_local_learning_signal_is_the_gradient_of_the_free_energy() {
        let m = model();
        let y = [1.2, -0.4, 0.9];
        let x = [0.7, -0.3];
        let g = m.weight_gradient(&x, &y).unwrap();
        for k in 0..6 {
            let h = 1e-6;
            let (mut up, mut down) = (m.clone(), m.clone());
            up.w[k] += h;
            down.w[k] -= h;
            let slope = (up.free_energy(&x, &y).unwrap() - down.free_energy(&x, &y).unwrap()) / (2.0 * h);
            assert!((g[k] + slope).abs() < 1e-8, "weight {k}: signal {}, −∂F/∂W {}", g[k], -slope);
        }
        assert!(g.iter().all(|v| v.abs() > 1e-3), "a zero signal would match any gradient");
    }

    #[test]
    fn the_tutorial_neuron_finds_the_tutorials_answer() {
        let mut cell = TutorialNeuron::new(3.0, 1.0, 1.0).unwrap();
        assert_eq!(cell.phi, 3.0);
        for _ in 0..20_000 {
            cell.step(1e-3, 2.0).unwrap();
        }
        // The posterior mode is the real root of 2φ³ − 3φ − 3 = 0, bracketed and bisected here.
        let (mut lo, mut hi) = (1.0f64, 2.0f64);
        assert!(cell.posterior_slope(lo, 2.0) > 0.0 && cell.posterior_slope(hi, 2.0) < 0.0);
        for _ in 0..80 {
            let mid = 0.5 * (lo + hi);
            if cell.posterior_slope(mid, 2.0) > 0.0 { lo = mid } else { hi = mid }
        }
        assert!((2.0 * lo.powi(3) - 3.0 * lo - 3.0).abs() < 1e-12, "the slope's root is the cubic's root");
        assert!((lo - 1.56747).abs() < 1e-5, "the tutorial's 'about 1.6' is {lo}");
        assert!((cell.phi - lo).abs() < 1e-6, "φ settled at {}, the mode is {lo}", cell.phi);
        // The error units have divided by their variances on their own.
        assert!((cell.eps_p - (cell.phi - 3.0) / 1.0).abs() < 1e-6);
        assert!((cell.eps_u - (2.0 - cell.phi * cell.phi) / 1.0).abs() < 1e-6);
        // With variances that are not 1 the same holds, which is what the leak term is for.
        let mut other = TutorialNeuron::new(3.0, 2.0, 0.5).unwrap();
        for _ in 0..40_000 {
            other.step(1e-3, 2.0).unwrap();
        }
        assert!(other.posterior_slope(other.phi, 2.0).abs() < 1e-6, "slope {} at φ = {}", other.posterior_slope(other.phi, 2.0), other.phi);
        assert!((other.eps_p - (other.phi - 3.0) / 2.0).abs() < 1e-6);
        assert!((other.eps_u - (2.0 - other.phi * other.phi) / 0.5).abs() < 1e-6);
    }

    #[test]
    fn local_updates_converge_on_backpropagation_as_the_clamp_weakens() {
        let mut rng = Rng::new(17);
        let base = Network::random(&[4, 6, 5, 3], 1.0, &mut rng).unwrap();
        let input = [0.5, -1.0, 0.25, 0.8];
        let out = base.forward(&input).unwrap().pop().unwrap();
        let mismatch = |var_out: f64, delta: f64| {
            let net = Network { var_out, ..base.clone() };
            let target: Vec<f64> = out.iter().zip(&[0.6, -0.3, 0.74]).map(|(o, d)| o + delta * d).collect();
            let (x, moved) = net.relax(&input, &target, 0.2, 4000).unwrap();
            assert!(moved < 1e-15, "the relaxation is still moving by {moved}");
            let scaled: Vec<Vec<f64>> = net.local_updates(&x).unwrap().iter().map(|g| g.iter().map(|v| v * var_out).collect()).collect();
            relative_mismatch(&scaled, &net.backprop(&input, &target).unwrap()).unwrap()
        };
        // First order in 1/Σ: a tenfold weaker clamp, a tenfold smaller mismatch.
        let (coarse, fine) = (mismatch(100.0, 0.5), mismatch(1000.0, 0.5));
        let ratio = coarse / fine;
        assert!((ratio - 10.0).abs() < 0.5, "the mismatch fell {ratio}-fold for a tenfold weaker clamp ({coarse} → {fine})");
        assert!(fine > 1e-6, "a mismatch of {fine} would mean the two are the same computation");
        // And NOT in the output error: at Σ = 1 the mismatch does not move when δ falls tenfold,
        // which is what the first draft of this module got wrong.
        let (big, small) = (mismatch(1.0, 1e-2), mismatch(1.0, 1e-3));
        assert!(big > 0.1 && (big / small - 1.0).abs() < 0.05, "at Σ = 1: {big} at δ = 1e-2, {small} at δ = 1e-3");
        // Why: the one-unit chain. Prior N(μ, 1) on the hidden unit, output variance Σ, weight w,
        // target δ past the prediction: the relaxed output error is δ/(Σ + w²), whatever δ is.
        let (w, sigma, mu, delta) = (1.5, 2.0, 0.4, 1e-3);
        let chain = LinearGaussian::new(1, 1, vec![w], sigma, 1.0, vec![mu]).unwrap();
        let y = [w * mu + delta];
        let (eps_y, _) = chain.errors(&chain.map(&y).unwrap(), &y).unwrap();
        assert!((eps_y[0] / (delta / (sigma + w * w)) - 1.0).abs() < 1e-9, "{} vs {}", eps_y[0], delta / (sigma + w * w));
    }

    #[test]
    fn the_network_updates_are_the_gradient_of_its_free_energy() {
        let mut rng = Rng::new(23);
        let net = Network::random(&[3, 4, 2], 2.5, &mut rng).unwrap();
        let x = vec![vec![0.4, -0.7, 0.2], vec![0.3, -0.1, 0.5, 0.9], vec![0.2, -0.6]];
        let g = net.local_updates(&x).unwrap();
        for l in 0..2 {
            for k in 0..net.w[l].len() {
                let h = 1e-6;
                let (mut up, mut down) = (net.clone(), net.clone());
                up.w[l][k] += h;
                down.w[l][k] -= h;
                let slope = (up.free_energy(&x).unwrap() - down.free_energy(&x).unwrap()) / (2.0 * h);
                assert!((g[l][k] + slope).abs() < 1e-8, "layer {l} weight {k}: {} vs {}", g[l][k], -slope);
            }
        }
        // A feedforward state has no error anywhere, so nothing to learn from.
        let ff = net.forward(&[0.4, -0.7, 0.2]).unwrap();
        assert!(net.free_energy(&ff).unwrap() < 1e-30);
        assert!(net.local_updates(&ff).unwrap().iter().flatten().all(|v| v.abs() < 1e-15));
    }

    #[test]
    fn bad_arguments_are_refused() {
        assert!(matches!(LinearGaussian::new(0, 1, vec![], 1.0, 1.0, vec![0.0]), Err(PredictiveError::Empty { what: "observations" })));
        assert!(matches!(LinearGaussian::new(1, 0, vec![], 1.0, 1.0, vec![]), Err(PredictiveError::Empty { what: "latents" })));
        assert!(matches!(LinearGaussian::new(2, 2, vec![1.0; 3], 1.0, 1.0, vec![0.0; 2]), Err(PredictiveError::Dimension { what: "w", .. })));
        assert!(matches!(LinearGaussian::new(2, 2, vec![1.0; 4], 1.0, 1.0, vec![0.0; 1]), Err(PredictiveError::Dimension { what: "prior", .. })));
        assert!(matches!(LinearGaussian::new(1, 1, vec![f64::NAN], 1.0, 1.0, vec![0.0]), Err(PredictiveError::NonFinite { what: "w", .. })));
        assert!(matches!(LinearGaussian::new(1, 1, vec![1.0], 0.0, 1.0, vec![0.0]), Err(PredictiveError::OutOfRange { what: "var_obs", .. })));
        assert!(matches!(LinearGaussian::new(1, 1, vec![1.0], 1.0, -1.0, vec![0.0]), Err(PredictiveError::OutOfRange { what: "var_prior", .. })));
        let m = model();
        assert!(matches!(m.errors(&[0.0], &[0.0; 3]), Err(PredictiveError::Dimension { what: "x", .. })));
        assert!(matches!(m.errors(&[0.0; 2], &[0.0; 2]), Err(PredictiveError::Dimension { what: "y", .. })));
        assert!(matches!(m.map(&[0.0, f64::NAN, 0.0]), Err(PredictiveError::NonFinite { what: "y", .. })));
        assert!(matches!(m.step(&mut [0.0, 0.0], &[0.0; 3], 0.0), Err(PredictiveError::OutOfRange { what: "dt", .. })));
        assert!(matches!(m.relax(&[0.0, 0.0], &[0.0; 3], 0.1, MAX_STEPS + 1), Err(PredictiveError::OutOfRange { what: "steps", .. })));
        assert!(matches!(TutorialNeuron::new(f64::NAN, 1.0, 1.0), Err(PredictiveError::NonFinite { .. })));
        assert!(matches!(TutorialNeuron::new(1.0, 0.0, 1.0), Err(PredictiveError::OutOfRange { what: "var_p", .. })));
        assert!(matches!(TutorialNeuron::new(1.0, 1.0, 0.0), Err(PredictiveError::OutOfRange { what: "var_u", .. })));
        let mut cell = TutorialNeuron::new(1.0, 1.0, 1.0).unwrap();
        assert!(matches!(cell.step(1e-3, f64::INFINITY), Err(PredictiveError::NonFinite { what: "u", .. })));
        assert!(matches!(cell.step(0.0, 1.0), Err(PredictiveError::OutOfRange { what: "dt", .. })));
        let mut rng = Rng::new(1);
        assert!(matches!(Network::random(&[3], 1.0, &mut rng), Err(PredictiveError::Empty { .. })));
        assert!(matches!(Network::random(&[3, 0, 2], 1.0, &mut rng), Err(PredictiveError::Empty { what: "a layer" })));
        assert!(matches!(Network::random(&[3, 2], 0.0, &mut rng), Err(PredictiveError::OutOfRange { what: "var_out", .. })));
        let net = Network::random(&[2, 3, 1], 1.0, &mut rng).unwrap();
        assert_eq!(net.w[0].len(), 6);
        assert!(net.w[0].iter().all(|v| v.abs() <= 1.0 / 2.0f64.sqrt()) && net.w[1].iter().all(|v| v.abs() <= 1.0 / 3.0f64.sqrt()));
        assert!(matches!(net.forward(&[0.0]), Err(PredictiveError::Dimension { what: "input", .. })));
        assert!(matches!(net.relax(&[0.0, 0.0], &[0.0, 0.0], 0.1, 10), Err(PredictiveError::Dimension { what: "target", .. })));
        assert!(matches!(net.backprop(&[0.0, 0.0], &[0.0, 0.0]), Err(PredictiveError::Dimension { what: "target", .. })));
        assert!(matches!(net.free_energy(&[vec![0.0; 2]]), Err(PredictiveError::Dimension { .. })));
        assert!(matches!(net.local_updates(&[vec![0.0; 2], vec![0.0; 2], vec![0.0]]), Err(PredictiveError::Dimension { .. })));
        assert_eq!(relative_mismatch(&[vec![1.0]], &[vec![1.0, 2.0]]), None);
        assert_eq!(relative_mismatch(&[vec![1.0]], &[vec![0.0]]), None);
        assert_eq!(relative_mismatch(&[vec![3.0, 0.0]], &[vec![0.0, 4.0]]), Some(1.25));
    }
}
