//! Deep continuous local learning (DECOLLE): every spiking layer learns from its OWN loss, read
//! out through a fixed random matrix, with no gradient passing between layers or through time.
//!
//! # What the mechanism is
//!
//! Kaiser, Mostafa and Neftci (*Synaptic plasticity dynamics for deep continuous local learning
//! (DECOLLE)*, Frontiers in Neuroscience 14:424 (2020), doi:10.3389/fnins.2020.00424) attach to
//! each layer of a spiking network a small linear readout `Y = G S` whose weights `G` are random
//! and never trained, and ask each layer to make ITS readout match the target at every time step.
//! The layer's neurons are leaky integrators written so that everything the gradient needs is
//! already a state variable. This module runs them in the UNNORMALISED form below, which is not
//! the form that journal paper prints; the next section says whose form it is and how the two
//! convert:
//!
//! ```text
//! U_i[t] = Σ_j W_ij P_j[t] − ρ R_i[t] + b_i          S_i[t] = Θ(U_i[t])
//! P_j[t+1] = α P_j[t] + Q_j[t]        Q_j[t+1] = β Q_j[t] + S_j^in[t]        R_i[t+1] = γ R_i[t] + S_i[t]
//! ```
//!
//! `P` is the input spike train filtered twice (synapse, then membrane), `R` the neuron's own
//! spikes filtered once (refractoriness). With the loss `L = ½ |G S − Ŷ|²` taken at the current
//! step and the spike's derivative replaced by a surrogate `σ′(U)`, the weight update is
//!
//! ```text
//! ∂L/∂W_ij = (Gᵀ (Y − Ŷ))_i · σ′(U_i) · P_j
//! ```
//!
//! — three factors, all available at the synapse at time `t`: an error broadcast through `Gᵀ`, the
//! postsynaptic slope, and the presynaptic trace. Nothing is stored over time (the trace `P` IS
//! the eligibility), and nothing comes from the layer above. As in the paper, the dependence of
//! `U` on the refractory trace `R` is left out of the gradient.
//!
//! # Whose equations these are
//!
//! Until this correction the module presented the recursions above as the cited paper's, and they
//! are not; the code has not changed, only what it says it follows. The journal's Eq. (4) reads
//!
//! ```text
//! P_j[t+Δt] = α P_j[t] + (1 − α) Q_j[t]    Q_j[t+Δt] = β Q_j[t] + (1 − β) S_j^{l−1}[t]    R_i[t+Δt] = γ R_i[t] + (1 − γ) S_i[t]
//! ```
//!
//! with every trace normalised to unit gain. The form used here is the authors' earlier one: the
//! preprint (Kaiser, Mostafa and Neftci, *Synaptic plasticity dynamics for deep continuous local
//! learning (DECOLLE)*, `arXiv`:1811.10766v3 (2019), Eq. (1)) writes
//! `P_j^l[n+1] = α P_j^l[n] + Q_j^l[n]` and `Q_j^l[n+1] = β Q_j^l[n] + S_j^{l−1}[n]`, and the
//! authors' code (`nmi-lab/decolle-public`, `decolle/base_model.py`) keeps it as the class
//! `LIFLayerNonorm`, whose `Q = self.beta * state.Q + Sin_t*self.gain` and
//! `P = self.alpha * state.P + state.Q` drop the factors their default `LIFLayer` carries. The
//! preprint typesets its `R` line with the layer's input `S_j^{l−1}`, while its text calls `R`
//! the state that "resets and inhibits the neuron after it has emitted a spike"; `R` here filters
//! the neuron's own spikes, as the journal's Eq. (4) and both of those classes do.
//!
//! The two forms are one model in two coordinate systems. Each trace is linear in its input and
//! starts from zero, so the journal's `Q`, `P` and `R` are the ones here times `(1 − β)`,
//! `(1 − α)(1 − β)` and `(1 − γ)`, and a layer here with `W = (1 − α)(1 − β) W_paper` and
//! `ρ = (1 − γ) ρ_paper` has the journal layer's potentials, output and readout at every step.
//! The update keeps its three factors and only its scale moves,
//! `∂L/∂W_paper = (1 − α)(1 − β) ∂L/∂W`, so a descent step of rate `η` in the journal's
//! coordinates is a weight step here of rate `((1 − α)(1 − β))² η`; the biases, which neither
//! form rescales, keep the rate `η`.
//! [`Layer::apply`] takes one rate for both, so following a journal-normalised run exactly means
//! applying the bias part of the update on its own. At `α = 0.9` and `β = 0.8` the factor is
//! 0.02: a weight here is a fiftieth of the journal weight that gives the same potential, and
//! the weight rate is 1/2500 of the journal's. A weight, refractory weight or learning rate taken
//! from that paper or from `LIFLayer` has to be converted before it is used here.
//!
//! # Why it is in a neuromorphic crate
//!
//! It is the learning rule built for the hardware: memory that does not grow with the length of
//! the sequence, no backward pass, no weight transport (`G` is fixed and local). It sits between
//! [`crate::eprop`], which keeps a per-synapse eligibility trace and a global error, and
//! [`crate::alignment`], which keeps the global error and drops the time.
//!
//! # The closed forms this module is checked against
//!
//! - **The traces.** One input spike at step 0 gives `Q[t] = β^{t−1}` and
//!   `P[t] = (α^{t−1} − β^{t−1})/(α − β)`; a neuron's own spike enters `R` as `γ^{t−1}`. The
//!   journal's Eq. (4) would put `(1 − β)`, `(1 − α)(1 − β)` and `(1 − γ)` in front of these.
//! - **The journal's normalisation is a change of coordinates.** Eq. (4), run as a recursion of
//!   its own beside a layer here converted as above, gives the traces here times those factors,
//!   the same potentials and output, a weight gradient `(1 − α)(1 − β)` times this one, and, with
//!   the rates converted, the same descent step: bit for bit at `α = 3/4` and `β = γ = 1/2`,
//!   where every factor is a power of two, and to 1e-12 at `α = 0.9`, `β = 0.8`, `γ = 0.7`.
//! - **The update is the gradient of the layer's own loss.** With graded output `S = σ(U)`
//!   ([`Output::Graded`]) the loss is differentiable and the update — weights and biases — is
//!   checked against central finite differences. With spiking output the same three factors are
//!   used with `S = Θ(U)` in the error, which is the surrogate-gradient step: given the same
//!   error vector the two modes return identical updates.
//! - **Locality.** A layer's update is unchanged, bit for bit, by any change to the layers above
//!   it — their weights, their readouts, even their removal.
//! - **It learns.** On a two-class spike-pattern task every layer's own loss falls and the last
//!   layer's readout classifies held-out patterns; the figures are measured, and labelled.
//!
//! # What this module has NOT reproduced
//!
//! - The paper's benchmarks (N-MNIST, DVS-Gesture), convolutional layers, its regularisers, and
//!   its smooth-L1 loss; the loss here is the squared error the derivation above uses.
//! - The `R` term of the gradient, which the paper also drops.
//! - The journal's normalised traces as a mode of the layer. Only the unnormalised form runs
//!   here; the conversion above is checked by a test, not offered as an option.

use crate::rng::Rng;
use core::fmt;

/// The most weights one layer may hold.
pub const MAX_WEIGHTS: usize = 1 << 24;

/// What went wrong, named rather than guessed around.
#[derive(Debug, Clone, PartialEq)]
pub enum DecolleError {
    /// A layer of zero width or one past [`MAX_WEIGHTS`], or a network of no layers.
    BadShape,
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
    /// A vector of the wrong length.
    Shape {
        /// Which vector.
        what: &'static str,
        /// Length supplied.
        got: usize,
        /// Length required.
        want: usize,
    },
    /// A `NaN` or infinity.
    NonFinite {
        /// Which quantity.
        what: &'static str,
    },
}

impl fmt::Display for DecolleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadShape => f.write_str("every layer needs at least one input, one neuron and one readout unit"),
            Self::OutOfRange { what, value, low, high } => write!(f, "{what} = {value} is outside [{low}, {high}]"),
            Self::Shape { what, got, want } => write!(f, "{what} has {got} entries, not {want}"),
            Self::NonFinite { what } => write!(f, "{what} is not finite"),
        }
    }
}

impl std::error::Error for DecolleError {}

fn checked(what: &'static str, v: &[f64], want: usize) -> Result<(), DecolleError> {
    if v.len() != want {
        return Err(DecolleError::Shape { what, got: v.len(), want });
    }
    if v.iter().all(|x| x.is_finite()) { Ok(()) } else { Err(DecolleError::NonFinite { what }) }
}

/// What a neuron emits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Output {
    /// `Θ(U)`: a spike when the potential is at or above zero. The learning rule then uses the
    /// surrogate slope in place of the spike's derivative.
    Spiking,
    /// `σ(U)`: the sigmoid whose slope the surrogate is. The rule is then the exact gradient.
    Graded,
}

/// The decay constants of a layer, each in `[0, 1)`, and its refractory weight.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Dynamics {
    /// Membrane trace decay `α`.
    pub alpha: f64,
    /// Synaptic trace decay `β`.
    pub beta: f64,
    /// Refractory trace decay `γ`.
    pub gamma: f64,
    /// Refractory weight `ρ`, non-negative.
    pub rho: f64,
    /// Steepness `k` of the surrogate sigmoid `σ(U) = 1/(1 + e^{−kU})`, positive.
    pub steepness: f64,
}

impl Dynamics {
    fn validated(self) -> Result<Self, DecolleError> {
        for (what, v) in [("alpha", self.alpha), ("beta", self.beta), ("gamma", self.gamma)] {
            if !(v >= 0.0) || !(v < 1.0) {
                return Err(DecolleError::OutOfRange { what, value: v, low: 0.0, high: 1.0 });
            }
        }
        if !(self.rho >= 0.0) || !self.rho.is_finite() {
            return Err(DecolleError::OutOfRange { what: "rho", value: self.rho, low: 0.0, high: f64::INFINITY });
        }
        if !(self.steepness > 0.0) || !self.steepness.is_finite() {
            return Err(DecolleError::OutOfRange { what: "steepness", value: self.steepness, low: f64::MIN_POSITIVE, high: f64::INFINITY });
        }
        Ok(self)
    }
}

/// One layer's update: `∂L/∂W` and `∂L/∂b` of its own loss.
#[derive(Debug, Clone, PartialEq)]
pub struct Update {
    /// Row-major, `n × n_in`.
    pub w: Vec<f64>,
    /// `n` long.
    pub b: Vec<f64>,
}

/// One DECOLLE layer: its neurons, their traces, and the fixed random readout they learn through.
#[derive(Debug, Clone, PartialEq)]
pub struct Layer {
    /// Inputs.
    pub n_in: usize,
    /// Neurons.
    pub n: usize,
    /// Readout units (the width of the target).
    pub n_out: usize,
    /// The decay constants.
    pub dynamics: Dynamics,
    /// What the neurons emit.
    pub output: Output,
    /// Trained weights, row-major `n × n_in`.
    pub w: Vec<f64>,
    /// Trained biases; a neuron spikes when `U ≥ 0`, so `−b` is its threshold.
    pub b: Vec<f64>,
    /// The FIXED readout, row-major `n_out × n`.
    pub g: Vec<f64>,
    /// Input trace `P`.
    pub p: Vec<f64>,
    /// Input trace `Q`.
    pub q: Vec<f64>,
    /// Refractory trace `R`.
    pub r: Vec<f64>,
    /// Potentials `U` at the last step.
    pub u: Vec<f64>,
    /// Output `S` at the last step.
    pub s: Vec<f64>,
}

impl Layer {
    /// A random layer at rest: weights uniform in `±1/√n_in` scaled by `weight_scale`, biases
    /// `−1` (a unit threshold), readout uniform in `±1/√n`.
    ///
    /// # Errors
    ///
    /// [`DecolleError::BadShape`] or [`DecolleError::OutOfRange`].
    pub fn random(n_in: usize, n: usize, n_out: usize, dynamics: Dynamics, output: Output, weight_scale: f64, rng: &mut Rng) -> Result<Self, DecolleError> {
        if n_in == 0 || n == 0 || n_out == 0 || n_in.saturating_mul(n) > MAX_WEIGHTS || n_out.saturating_mul(n) > MAX_WEIGHTS {
            return Err(DecolleError::BadShape);
        }
        let dynamics = dynamics.validated()?;
        if !(weight_scale > 0.0) || !weight_scale.is_finite() {
            return Err(DecolleError::OutOfRange { what: "weight_scale", value: weight_scale, low: f64::MIN_POSITIVE, high: f64::INFINITY });
        }
        let mut draw = |count: usize, scale: f64| -> Vec<f64> { (0..count).map(|_| scale * (2.0 * rng.next_f64() - 1.0)).collect() };
        let w = draw(n * n_in, weight_scale / (n_in as f64).sqrt());
        let g = draw(n_out * n, 1.0 / (n as f64).sqrt());
        Ok(Self { n_in, n, n_out, dynamics, output, w, b: vec![-1.0; n], g, p: vec![0.0; n_in], q: vec![0.0; n_in], r: vec![0.0; n], u: vec![0.0; n], s: vec![0.0; n] })
    }

    /// Forget every trace.
    pub fn reset(&mut self) {
        for v in [&mut self.p, &mut self.q, &mut self.r, &mut self.u, &mut self.s] {
            v.iter_mut().for_each(|x| *x = 0.0);
        }
    }

    fn sigmoid(&self, u: f64) -> f64 {
        1.0 / (1.0 + (-self.dynamics.steepness * u).exp())
    }

    /// The surrogate slope `σ′(U) = k σ(U) (1 − σ(U))`.
    #[must_use]
    pub fn slope(&self, u: f64) -> f64 {
        let s = self.sigmoid(u);
        self.dynamics.steepness * s * (1.0 - s)
    }

    /// One time step: compute `U` and `S` from the traces as they stand, then advance the traces
    /// with this step's input and output. Returns `S`.
    ///
    /// # Errors
    ///
    /// [`DecolleError::Shape`] or [`DecolleError::NonFinite`] for a bad input.
    pub fn step(&mut self, input: &[f64]) -> Result<&[f64], DecolleError> {
        checked("input", input, self.n_in)?;
        let d = self.dynamics;
        for i in 0..self.n {
            let drive: f64 = self.w[i * self.n_in..(i + 1) * self.n_in].iter().zip(&self.p).map(|(w, p)| w * p).sum();
            self.u[i] = drive - d.rho * self.r[i] + self.b[i];
            self.s[i] = match self.output {
                Output::Spiking => f64::from(u8::from(self.u[i] >= 0.0)),
                Output::Graded => self.sigmoid(self.u[i]),
            };
        }
        for j in 0..self.n_in {
            self.p[j] = d.alpha * self.p[j] + self.q[j];
            self.q[j] = d.beta * self.q[j] + input[j];
        }
        for i in 0..self.n {
            self.r[i] = d.gamma * self.r[i] + self.s[i];
        }
        Ok(&self.s)
    }

    /// The layer's readout `Y = G S` at the last step.
    #[must_use]
    pub fn readout(&self) -> Vec<f64> {
        (0..self.n_out).map(|k| self.g[k * self.n..(k + 1) * self.n].iter().zip(&self.s).map(|(g, s)| g * s).sum()).collect()
    }

    /// The layer's own loss at the last step, `½ |Y − Ŷ|²`.
    ///
    /// # Errors
    ///
    /// [`DecolleError::Shape`] or [`DecolleError::NonFinite`] for a bad target.
    pub fn local_loss(&self, target: &[f64]) -> Result<f64, DecolleError> {
        checked("target", target, self.n_out)?;
        Ok(0.5 * self.readout().iter().zip(target).map(|(y, t)| (y - t) * (y - t)).sum::<f64>())
    }

    /// The update for a given readout error `Y − Ŷ`: `(Gᵀ e)_i σ′(U_i) P_j`, with `P` the trace
    /// that produced the last step's `U` — which the caller passes, since [`Layer::step`] has
    /// already advanced it.
    ///
    /// # Errors
    ///
    /// [`DecolleError::Shape`] or [`DecolleError::NonFinite`].
    pub fn update_from_error(&self, error: &[f64], trace: &[f64]) -> Result<Update, DecolleError> {
        checked("error", error, self.n_out)?;
        checked("trace", trace, self.n_in)?;
        let b: Vec<f64> = (0..self.n)
            .map(|i| {
                let broadcast: f64 = (0..self.n_out).map(|k| self.g[k * self.n + i] * error[k]).sum();
                broadcast * self.slope(self.u[i])
            })
            .collect();
        let w = b.iter().flat_map(|d| trace.iter().map(move |p| d * p)).collect();
        Ok(Update { w, b })
    }

    /// Move the weights against an update.
    ///
    /// # Errors
    ///
    /// [`DecolleError::Shape`] if the update is not this layer's; [`DecolleError::NonFinite`] for
    /// a non-finite rate.
    pub fn apply(&mut self, update: &Update, rate: f64) -> Result<(), DecolleError> {
        if !rate.is_finite() {
            return Err(DecolleError::NonFinite { what: "rate" });
        }
        checked("update.w", &update.w, self.w.len())?;
        checked("update.b", &update.b, self.b.len())?;
        self.w.iter_mut().zip(&update.w).for_each(|(w, g)| *w -= rate * g);
        self.b.iter_mut().zip(&update.b).for_each(|(b, g)| *b -= rate * g);
        Ok(())
    }
}

/// A stack of layers, each fed the spikes of the one below and each learning from its own loss.
#[derive(Debug, Clone, PartialEq)]
pub struct Network {
    /// The layers, input side first.
    pub layers: Vec<Layer>,
}

impl Network {
    /// A random network: `sizes` is the input width followed by each layer's width, and every
    /// layer reads out to `n_out` units.
    ///
    /// # Errors
    ///
    /// As [`Layer::random`]; [`DecolleError::BadShape`] for fewer than two sizes.
    pub fn random(sizes: &[usize], n_out: usize, dynamics: Dynamics, output: Output, weight_scale: f64, seed: u64) -> Result<Self, DecolleError> {
        if sizes.len() < 2 {
            return Err(DecolleError::BadShape);
        }
        let mut rng = Rng::new(seed);
        let layers = sizes.windows(2).map(|p| Layer::random(p[0], p[1], n_out, dynamics, output, weight_scale, &mut rng)).collect::<Result<_, _>>()?;
        Ok(Self { layers })
    }

    /// Forget every trace in every layer.
    pub fn reset(&mut self) {
        self.layers.iter_mut().for_each(Layer::reset);
    }

    /// One time step through the stack. Returns, for each layer, the input trace `P` that
    /// produced its potentials — what its update needs.
    ///
    /// # Errors
    ///
    /// As [`Layer::step`].
    pub fn step(&mut self, input: &[f64]) -> Result<Vec<Vec<f64>>, DecolleError> {
        let mut traces = Vec::with_capacity(self.layers.len());
        let mut signal = input.to_vec();
        for layer in &mut self.layers {
            traces.push(layer.p.clone());
            signal = layer.step(&signal)?.to_vec();
        }
        Ok(traces)
    }

    /// Every layer's update for `target` at the last step, from the traces [`Network::step`]
    /// returned.
    ///
    /// # Errors
    ///
    /// As [`Layer::update_from_error`].
    pub fn updates(&self, target: &[f64], traces: &[Vec<f64>]) -> Result<Vec<Update>, DecolleError> {
        if traces.len() != self.layers.len() {
            return Err(DecolleError::Shape { what: "traces", got: traces.len(), want: self.layers.len() });
        }
        self.layers
            .iter()
            .zip(traces)
            .map(|(layer, trace)| {
                checked("target", target, layer.n_out)?;
                let error: Vec<f64> = layer.readout().iter().zip(target).map(|(y, t)| y - t).collect();
                layer.update_from_error(&error, trace)
            })
            .collect()
    }

    /// Step, then let every layer learn from its own loss. Returns each layer's loss at this step
    /// (before the weights moved).
    ///
    /// # Errors
    ///
    /// As [`Network::step`], [`Network::updates`] and [`Layer::apply`].
    pub fn train_step(&mut self, input: &[f64], target: &[f64], rate: f64) -> Result<Vec<f64>, DecolleError> {
        let traces = self.step(input)?;
        let updates = self.updates(target, &traces)?;
        let losses = self.layers.iter().map(|l| l.local_loss(target)).collect::<Result<Vec<_>, _>>()?;
        for (layer, update) in self.layers.iter_mut().zip(&updates) {
            layer.apply(update, rate)?;
        }
        Ok(losses)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DYNAMICS: Dynamics = Dynamics { alpha: 0.9, beta: 0.8, gamma: 0.7, rho: 0.5, steepness: 4.0 };

    #[test]
    fn the_traces_are_the_double_and_single_exponential_filters() {
        let mut rng = Rng::new(1);
        let mut layer = Layer::random(2, 1, 1, DYNAMICS, Output::Spiking, 1.0, &mut rng).unwrap();
        // Make the neuron fire at step 0 only: a bias above zero, then far below.
        layer.b[0] = 1.0;
        let mut seen = Vec::new();
        for t in 0..12 {
            let s = layer.step(&[f64::from(u8::from(t == 0)), 0.0]).unwrap()[0];
            if t == 0 {
                assert_eq!(s, 1.0);
                layer.b[0] = -100.0;
            } else {
                assert_eq!(s, 0.0);
            }
            seen.push((layer.q[0], layer.p[0], layer.r[0], layer.p[1]));
        }
        // After step t the traces hold their values for step t + 1.
        for (t, &(q, p, r, idle)) in seen.iter().enumerate() {
            let k = t as i32; // the traces' time index, minus one
            assert!((q - 0.8f64.powi(k)).abs() < 1e-15, "Q[{}] = {q}", t + 1);
            assert!((p - (0.9f64.powi(k) - 0.8f64.powi(k)) / (0.9 - 0.8)).abs() < 1e-14, "P[{}] = {p}", t + 1);
            assert!((r - 0.7f64.powi(k)).abs() < 1e-15, "R[{}] = {r}", t + 1);
            assert_eq!(idle, 0.0);
        }
        layer.reset();
        assert!(layer.p.iter().chain(&layer.q).chain(&layer.r).chain(&layer.u).chain(&layer.s).all(|&x| x == 0.0));
    }

    #[test]
    fn the_journal_normalisation_is_this_layer_reparametrised() {
        // Referee: Eq. (4) of Kaiser, Mostafa and Neftci, Frontiers in Neuroscience 14:424 (2020),
        // with its (1 − α), (1 − β), (1 − γ) factors, run as its own recursion on the journal's
        // weights, beside a layer here whose weights and refractory weight are the journal's times
        // (1 − α)(1 − β) and (1 − γ). At the first dynamics every factor is a power of two, so the
        // scaling commutes with rounding and the two must agree BIT FOR BIT; the second is the
        // decimal set the other tests use.
        let binary = Dynamics { alpha: 0.75, beta: 0.5, gamma: 0.5, rho: 0.0, steepness: 4.0 };
        let (n_in, n, rho_paper, eta, target) = (6usize, 4usize, 2.0, 0.25, [0.25, -0.5]);
        for (dynamics, tol) in [(binary, 0.0), (DYNAMICS, 1e-12)] {
            let (al, be, ga) = (dynamics.alpha, dynamics.beta, dynamics.gamma);
            let (c, cr) = ((1.0 - al) * (1.0 - be), 1.0 - ga);
            let close = |x: f64, y: f64| (x - y).abs() <= tol * (1.0 + y.abs());
            for output in [Output::Graded, Output::Spiking] {
                let mut rng = Rng::new(21);
                let mut layer = Layer::random(n_in, n, 2, Dynamics { rho: cr * rho_paper, ..dynamics }, output, 1.0, &mut rng).unwrap();
                let w_paper: Vec<f64> = (0..n * n_in).map(|_| 1.5 * (2.0 * rng.next_f64() - 1.0)).collect();
                layer.w = w_paper.iter().map(|w| c * w).collect();
                layer.b = (0..n).map(|_| rng.next_f64() - 0.5).collect();
                let sigmoid = |u: f64| 1.0 / (1.0 + (-dynamics.steepness * u).exp());
                let (mut pj, mut qj, mut ri) = (vec![0.0; n_in], vec![0.0; n_in], vec![0.0; n]);
                let (mut fired, mut quiet) = (0, 0);
                for t in 0..40 {
                    let input: Vec<f64> = (0..n_in).map(|_| f64::from(u8::from(rng.next_f64() < 0.4))).collect();
                    // The journal's step, from its own traces.
                    let u: Vec<f64> = (0..n).map(|i| (0..n_in).map(|j| w_paper[i * n_in + j] * pj[j]).sum::<f64>() - rho_paper * ri[i] + layer.b[i]).collect();
                    let s: Vec<f64> = u
                        .iter()
                        .map(|&u| match output {
                            Output::Spiking => f64::from(u8::from(u >= 0.0)),
                            Output::Graded => sigmoid(u),
                        })
                        .collect();
                    let trace_paper = pj.clone();
                    for j in 0..n_in {
                        pj[j] = al * pj[j] + (1.0 - al) * qj[j];
                        qj[j] = be * qj[j] + (1.0 - be) * input[j];
                    }
                    for i in 0..n {
                        ri[i] = ga * ri[i] + (1.0 - ga) * s[i];
                    }
                    // This module's step, on the converted parameters.
                    let trace = layer.p.clone();
                    layer.step(&input).unwrap();
                    for i in 0..n {
                        assert!(close(layer.u[i], u[i]), "{output:?} at α = {al}, t = {t}: U[{i}] = {} here, {} by Eq. (4)", layer.u[i], u[i]);
                        assert!(output == Output::Graded || u[i].abs() > 1e-9, "U[{i}] = {} sits on the threshold, so the spike comparison is a coin flip", u[i]);
                        assert!(close(layer.s[i], s[i]), "{output:?} at α = {al}, t = {t}: S[{i}] = {} here, {} by Eq. (4)", layer.s[i], s[i]);
                        assert!(close(cr * layer.r[i], ri[i]), "R[{i}]: (1 − γ) · {} against {}", layer.r[i], ri[i]);
                        fired += usize::from(s[i] >= 0.5);
                        quiet += usize::from(s[i] < 0.5);
                    }
                    for j in 0..n_in {
                        assert!(close(c * layer.p[j], pj[j]), "P[{j}]: (1 − α)(1 − β) · {} against {}", layer.p[j], pj[j]);
                        assert!(close((1.0 - be) * layer.q[j], qj[j]), "Q[{j}]: (1 − β) · {} against {}", layer.q[j], qj[j]);
                    }
                    if t < 39 {
                        continue;
                    }
                    // The readout, the gradient in the journal's coordinates, and one descent step.
                    let y: Vec<f64> = (0..2).map(|k| (0..n).map(|i| layer.g[k * n + i] * s[i]).sum::<f64>()).collect();
                    assert!(y.iter().zip(layer.readout()).all(|(&y, here)| close(here, y)), "the readouts differ");
                    let error: Vec<f64> = y.iter().zip(&target).map(|(y, t)| y - t).collect();
                    let update = layer.update_from_error(&error, &trace).unwrap();
                    assert!(update.w.iter().any(|g| g.abs() > 1e-6), "the gradient checked was all but zero");
                    let mut stepped = layer.clone();
                    stepped.apply(&Update { w: update.w.clone(), b: vec![0.0; n] }, c * c * eta).unwrap();
                    stepped.apply(&Update { w: vec![0.0; n * n_in], b: update.b.clone() }, eta).unwrap();
                    for i in 0..n {
                        let sg = sigmoid(u[i]);
                        let delta = (0..2).map(|k| layer.g[k * n + i] * error[k]).sum::<f64>() * (dynamics.steepness * sg * (1.0 - sg));
                        assert!(close(update.b[i], delta), "∂L/∂b[{i}]: {} against {delta}", update.b[i]);
                        assert!(close(stepped.b[i], layer.b[i] - eta * delta), "b[{i}] after the step");
                        for j in 0..n_in {
                            let (k, g_paper) = (i * n_in + j, delta * trace_paper[j]);
                            assert!(close(c * update.w[k], g_paper), "∂L/∂W_paper[{k}]: (1 − α)(1 − β) · {} against {g_paper}", update.w[k]);
                            assert!(close(stepped.w[k], c * (w_paper[k] - eta * g_paper)), "W[{k}] after the step");
                        }
                    }
                }
                assert!(fired > 0 && quiet > 0 && layer.r.iter().any(|&r| r > 0.1), "{output:?} at α = {al}: {fired} fired, {quiet} quiet");
            }
        }
    }

    /// A graded layer driven for a few steps, its traces warm and its refractory trace non-zero.
    fn warm(output: Output) -> (Layer, Vec<f64>) {
        let mut rng = Rng::new(5);
        let mut layer = Layer::random(6, 5, 3, DYNAMICS, output, 3.0, &mut rng).unwrap();
        layer.b = (0..5).map(|_| rng.next_f64() - 0.8).collect();
        let mut trace = Vec::new();
        for _ in 0..8 {
            let input: Vec<f64> = (0..6).map(|_| f64::from(u8::from(rng.next_f64() < 0.4))).collect();
            trace = layer.p.clone();
            layer.step(&input).unwrap();
        }
        (layer, trace)
    }

    #[test]
    fn with_graded_output_the_update_is_the_gradient_of_the_layers_own_loss() {
        let (layer, trace) = warm(Output::Graded);
        assert!(layer.r.iter().any(|&r| r > 0.1) && trace.iter().any(|&p| p > 0.1));
        let target = [0.3, -0.2, 0.9];
        let error: Vec<f64> = layer.readout().iter().zip(&target).map(|(y, t)| y - t).collect();
        let update = layer.update_from_error(&error, &trace).unwrap();
        // Referee: recompute the last step's loss from the trace and refractory state that
        // produced it, with one parameter nudged. R before the step is (R_now − S)/γ.
        let loss = |w: &[f64], b: &[f64]| {
            let s: Vec<f64> = (0..5)
                .map(|i| {
                    let r_before = (layer.r[i] - layer.s[i]) / 0.7;
                    let u = (0..6).map(|j| w[i * 6 + j] * trace[j]).sum::<f64>() - 0.5 * r_before + b[i];
                    1.0 / (1.0 + (-4.0 * u).exp())
                })
                .collect();
            (0..3).map(|k| (0..5).map(|i| layer.g[k * 5 + i] * s[i]).sum::<f64>() - target[k]).map(|e| 0.5 * e * e).sum::<f64>()
        };
        assert!((loss(&layer.w, &layer.b) - layer.local_loss(&target).unwrap()).abs() < 1e-14, "the referee does not reproduce the step");
        let h = 1e-6;
        for k in 0..30 {
            let (mut up, mut down) = (layer.w.clone(), layer.w.clone());
            up[k] += h;
            down[k] -= h;
            let fd = (loss(&up, &layer.b) - loss(&down, &layer.b)) / (2.0 * h);
            assert!((update.w[k] - fd).abs() < 1e-9, "w[{k}]: {} against {fd}", update.w[k]);
        }
        for k in 0..5 {
            let (mut up, mut down) = (layer.b.clone(), layer.b.clone());
            up[k] += h;
            down[k] -= h;
            let fd = (loss(&layer.w, &up) - loss(&layer.w, &down)) / (2.0 * h);
            assert!((update.b[k] - fd).abs() < 1e-9, "b[{k}]: {} against {fd}", update.b[k]);
        }
        assert!(update.w.iter().any(|g| g.abs() > 1e-3), "the gradient checked was all but zero");
    }

    #[test]
    fn a_spiking_layer_uses_the_same_three_factors_with_spikes_in_the_error() {
        // A warm layer, then one more step with the biases set so that three of five fire.
        let (mut spiking, _) = warm(Output::Spiking);
        spiking.b = vec![20.0, -20.0, 20.0, -20.0, 20.0];
        let trace = spiking.p.clone();
        let s = spiking.step(&[1.0, 0.0, 1.0, 0.0, 0.0, 1.0]).unwrap().to_vec();
        assert_eq!(s, vec![1.0, 0.0, 1.0, 0.0, 1.0]);
        // Bring the potentials back into the surrogate's range, where its slope is not rounding.
        spiking.u = vec![0.3, -0.4, 1.2, -0.1, 0.05];
        // By hand, for one synapse: (Σ_k G_ki e_k) · kσ(1 − σ) · P_j.
        let error = [0.5, -1.0, 0.25];
        let update = spiking.update_from_error(&error, &trace).unwrap();
        let (i, j) = (2usize, 3usize);
        let broadcast: f64 = (0..3).map(|k| spiking.g[k * 5 + i] * error[k]).sum();
        let sig = 1.0 / (1.0 + (-4.0 * spiking.u[i]).exp());
        assert!((update.w[i * 6 + j] - broadcast * 4.0 * sig * (1.0 - sig) * trace[j]).abs() < 1e-15);
        assert!((update.b[i] - broadcast * 4.0 * sig * (1.0 - sig)).abs() < 1e-15);
        assert!((spiking.slope(0.0) - 1.0).abs() < 1e-15, "the surrogate's peak slope is k/4");
        // A graded layer in the same state returns the same update for the same error: the mode
        // changes what is emitted, not the rule.
        let mut graded = spiking.clone();
        graded.output = Output::Graded;
        assert_eq!(graded.update_from_error(&error, &trace).unwrap(), update);
    }

    fn pattern(rng: &mut Rng, class: usize) -> Vec<f64> {
        // Twelve inputs; class 0 drives the first six at 0.5 a step, class 1 the last six.
        (0..12).map(|j| f64::from(u8::from(rng.next_f64() < if (j < 6) == (class == 0) { 0.5 } else { 0.05 }))).collect()
    }

    #[test]
    fn the_potential_is_built_from_the_trace_the_step_hands_back_and_the_update_uses_that_trace() {
        // The trace a step returns is the one that produced that step's potentials, which is an
        // identity anyone can check: U = W P − ρ R_before + b. A stack that handed back the trace
        // as it stands AFTER the step would fail it by one filter step.
        let mut net = Network::random(&[12, 10, 8], 2, DYNAMICS, Output::Spiking, 3.0, 3).unwrap();
        let mut rng = Rng::new(8);
        let mut traces = Vec::new();
        for _ in 0..10 {
            traces = net.step(&pattern(&mut rng, 0)).unwrap();
        }
        for (layer, trace) in net.layers.iter().zip(&traces) {
            assert!(trace.iter().any(|&p| p > 0.1), "the identity is vacuous on an all-zero trace");
            for i in 0..layer.n {
                let drive: f64 = layer.w[i * layer.n_in..(i + 1) * layer.n_in].iter().zip(trace).map(|(w, p)| w * p).sum();
                let r_before = (layer.r[i] - layer.s[i]) / layer.dynamics.gamma;
                assert!((layer.u[i] - (drive - layer.dynamics.rho * r_before + layer.b[i])).abs() < 1e-12, "U[{i}] is not what that trace gives");
            }
        }
        // And the update is computed from the trace it is HANDED, not from whatever the layer
        // happens to be holding: change the trace and the update changes with it.
        let whole = net.updates(&[1.0, 0.0], &traces).unwrap();
        let mut nudged = traces.clone();
        nudged[0][0] += 1.0;
        let other = net.updates(&[1.0, 0.0], &nudged).unwrap();
        assert_ne!(other[0], whole[0]);
        assert_eq!(other[1], whole[1], "nudging one layer's trace must not disturb another's");
    }

    #[test]
    fn a_neuron_at_exactly_zero_has_reached_threshold() {
        // Θ(0) = 1: the potential is measured against zero, and `≥` is the comparison. A model
        // that waited for a strictly positive potential would differ only on this measure-zero
        // set — and a layer at rest with a zero bias sits exactly on it.
        let mut rng = Rng::new(2);
        let mut layer = Layer::random(3, 2, 1, DYNAMICS, Output::Spiking, 1.0, &mut rng).unwrap();
        layer.b = vec![0.0, -1e-300];
        let s = layer.step(&[0.0, 0.0, 0.0]).unwrap().to_vec();
        assert_eq!((layer.u[0], layer.u[1]), (0.0, -1e-300));
        assert_eq!(s, [1.0, 0.0], "a neuron at exactly zero spiked {s:?}");
    }

    #[test]
    fn a_layers_update_does_not_depend_on_anything_above_it() {
        let mut net = Network::random(&[12, 10, 8, 6], 2, DYNAMICS, Output::Spiking, 3.0, 3).unwrap();
        let mut rng = Rng::new(8);
        let mut traces = Vec::new();
        for _ in 0..10 {
            traces = net.step(&pattern(&mut rng, 0)).unwrap();
        }
        let whole = net.updates(&[1.0, 0.0], &traces).unwrap();
        assert!(whole.iter().all(|u| u.w.iter().any(|&g| g != 0.0)), "an all-zero update would make this vacuous");
        // Scramble everything above the first layer, then cut it off entirely.
        let mut scrambled = net.clone();
        for layer in &mut scrambled.layers[1..] {
            layer.w.iter_mut().for_each(|w| *w = -*w + 0.3);
            layer.g.iter_mut().for_each(|g| *g *= 2.0);
        }
        assert_eq!(scrambled.updates(&[1.0, 0.0], &traces).unwrap()[0], whole[0]);
        assert_ne!(scrambled.updates(&[1.0, 0.0], &traces).unwrap()[2], whole[2]);
        let alone = Network { layers: vec![net.layers[0].clone()] };
        assert_eq!(alone.updates(&[1.0, 0.0], &traces[..1]).unwrap()[0], whole[0]);
    }

    #[test]
    fn every_layer_learns_from_its_own_loss() {
        let mut net = Network::random(&[12, 24, 16], 2, DYNAMICS, Output::Spiking, 3.0, 11).unwrap();
        let mut rng = Rng::new(12);
        let run = |net: &mut Network, rng: &mut Rng, rate: f64| {
            // One pattern presentation of 40 steps; returns each layer's mean loss over the last
            // 20 and whether the last layer's summed readout picked the class.
            let class = rng.below(2) as usize;
            let target = [f64::from(u8::from(class == 0)), f64::from(u8::from(class == 1))];
            net.reset();
            let mut loss = vec![0.0; net.layers.len()];
            let mut votes = [0.0f64; 2];
            for t in 0..40 {
                let l = net.train_step(&pattern(rng, class), &target, if t >= 10 { rate } else { 0.0 }).unwrap();
                if t >= 20 {
                    loss.iter_mut().zip(&l).for_each(|(s, l)| *s += l / 20.0);
                    let y = net.layers.last().unwrap().readout();
                    votes[0] += y[0];
                    votes[1] += y[1];
                }
            }
            (loss, usize::from(votes[1] > votes[0]) == class)
        };
        let evaluate = |net: &mut Network, rng: &mut Rng| {
            let (mut loss, mut right) = (vec![0.0; 2], 0);
            for _ in 0..100 {
                let (l, ok) = run(net, rng, 0.0);
                loss.iter_mut().zip(&l).for_each(|(s, l)| *s += l / 100.0);
                right += usize::from(ok);
            }
            (loss, right)
        };
        let (before, right_before) = evaluate(&mut net, &mut rng);
        for _ in 0..300 {
            run(&mut net, &mut rng, 0.02);
        }
        let (after, right_after) = evaluate(&mut net, &mut rng);
        // MEASURED at this seed.
        for l in 0..2 {
            assert!(after[l] < 0.5 * before[l], "layer {l}: own loss {} → {}", before[l], after[l]);
        }
        assert!(right_after >= 95 && right_after > right_before, "held-out accuracy {right_before} → {right_after} of 100");
    }

    #[test]
    fn bad_shapes_and_values_are_refused() {
        let mut rng = Rng::new(1);
        assert_eq!(Layer::random(0, 3, 2, DYNAMICS, Output::Spiking, 1.0, &mut rng), Err(DecolleError::BadShape));
        assert_eq!(Layer::random(3, 3, 0, DYNAMICS, Output::Spiking, 1.0, &mut rng), Err(DecolleError::BadShape));
        assert_eq!(Layer::random(1 << 13, 1 << 13, 1, DYNAMICS, Output::Spiking, 1.0, &mut rng), Err(DecolleError::BadShape));
        for bad in [Dynamics { alpha: 1.0, ..DYNAMICS }, Dynamics { beta: -0.1, ..DYNAMICS }, Dynamics { gamma: f64::NAN, ..DYNAMICS }, Dynamics { rho: -1.0, ..DYNAMICS }, Dynamics { steepness: 0.0, ..DYNAMICS }] {
            assert!(matches!(Layer::random(3, 3, 2, bad, Output::Spiking, 1.0, &mut rng), Err(DecolleError::OutOfRange { .. })), "{bad:?}");
        }
        assert!(Layer::random(3, 3, 2, DYNAMICS, Output::Spiking, 0.0, &mut rng).is_err());
        assert_eq!(Network::random(&[3], 2, DYNAMICS, Output::Spiking, 1.0, 1), Err(DecolleError::BadShape));
        let mut net = Network::random(&[3, 4], 2, DYNAMICS, Output::Spiking, 1.0, 1).unwrap();
        assert_eq!(net.step(&[1.0, 0.0]), Err(DecolleError::Shape { what: "input", got: 2, want: 3 }));
        assert_eq!(net.step(&[1.0, f64::NAN, 0.0]), Err(DecolleError::NonFinite { what: "input" }));
        let traces = net.step(&[1.0, 0.0, 1.0]).unwrap();
        assert_eq!(net.updates(&[1.0], &traces), Err(DecolleError::Shape { what: "target", got: 1, want: 2 }));
        assert_eq!(net.updates(&[1.0, 0.0], &[]), Err(DecolleError::Shape { what: "traces", got: 0, want: 1 }));
        assert!(net.layers[0].local_loss(&[f64::INFINITY, 0.0]).is_err());
        assert!(net.layers[0].update_from_error(&[0.0, 0.0], &[0.0]).is_err());
        // A warm layer, so that the update below is not all zeros and a step is visible.
        for _ in 0..6 {
            net.step(&[1.0, 0.0, 1.0]).unwrap();
        }
        let traces = net.step(&[1.0, 0.0, 1.0]).unwrap();
        let update = net.updates(&[1.0, 0.0], &traces).unwrap().remove(0);
        assert!(update.w.iter().any(|&g| g != 0.0) && update.b.iter().any(|&g| g != 0.0), "the update is all zeros, so nothing below could move");
        let before = net.clone();
        assert_eq!(net.layers[0].apply(&update, f64::NAN), Err(DecolleError::NonFinite { what: "rate" }));
        let mut short = update.clone();
        short.b.pop();
        assert!(net.layers[0].apply(&short, 0.1).is_err());
        assert_eq!(net, before, "a refused update must not move the layer");
        // And an accepted one moves every parameter by exactly `-rate * gradient`, biases too.
        net.layers[0].apply(&update, 0.5).unwrap();
        let mut moved = 0;
        for (k, g) in update.w.iter().enumerate() {
            assert!((net.layers[0].w[k] - (before.layers[0].w[k] - 0.5 * g)).abs() < 1e-18);
            moved += usize::from(net.layers[0].w[k] != before.layers[0].w[k]);
        }
        for (k, g) in update.b.iter().enumerate() {
            assert!((net.layers[0].b[k] - (before.layers[0].b[k] - 0.5 * g)).abs() < 1e-18);
            moved += usize::from(net.layers[0].b[k] != before.layers[0].b[k]);
        }
        assert!(moved >= 4, "only {moved} parameters moved");
        assert!(DecolleError::BadShape.to_string().contains("at least one"));
    }
}
