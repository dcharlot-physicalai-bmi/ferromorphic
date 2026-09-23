//! Learning with spike TIMES: networks in which every neuron fires at most once, the time of that
//! spike has a closed form, and so does its gradient — exact backpropagation through spikes, with
//! no surrogate, checked against finite differences of the spike times themselves.
//!
//! # What the mechanism is
//!
//! In time-to-first-spike coding a value is WHEN a neuron fires: earlier is stronger, and a
//! classifier's answer is whichever output fires first. The difficulty of training spiking
//! networks — the spike is a step, and a step has no derivative — goes away, because the quantity
//! being learned is the spike TIME, and the spike time is a smooth function of the weights and
//! the input times as long as the set of inputs that caused it does not change. Two neuron models
//! make that function explicit:
//!
//! - **The non-leaky integrator with exponential synapses** (Mostafa, *Supervised learning based on
//!   temporal coding in spiking neural networks*, IEEE Transactions on Neural Networks and Learning
//!   Systems 29(7):3227–3235, 2018). The membrane is `V(t) = Σ_{t_i<t} w_i (1 − e^{−(t−t_i)/τ})`;
//!   setting it to the threshold gives `e^{t/τ} = Σ_C w_i e^{t_i/τ} / (Σ_C w_i − θ)` over the CAUSAL
//!   set `C` of inputs that arrived before the output spike — linear in the variables `e^{t/τ}`.
//! - **The leaky integrator whose membrane is twice as slow as its synapse**, `τ_m = 2τ_s` (Göltz,
//!   Kriener, Baumbach, Billaudelle, Breitwieser, Cramer, Dold, Kungl, Senn, Schemmel, Meier and
//!   Petrovici, *Fast and energy-efficient neuromorphic deep learning with first-spike times*,
//!   Nature Machine Intelligence 3(9):823–835, 2021). With `V(t) = Σ w_i (e^{−s_i/τ_m} − e^{−s_i/τ_s})`
//!   and `x = e^{−t/τ_m}` the threshold condition is the quadratic `a₁x² − a₂x + θ = 0`, whose
//!   larger root is the first upward crossing.
//!
//! For either, the implicit function theorem gives the gradient of the spike time from the
//! membrane's slope at the crossing: `∂t*/∂w_p = −κ(t* − t_p)/V̇(t*)` and
//! `∂t*/∂t_p = w_p κ′(t* − t_p)/V̇(t*)` for every input `p` in the causal set, zero otherwise
//! ([`gradients`]). Chained through layers that is backpropagation in spike times, exact — the
//! same quantity `EventProp` computes by an adjoint system (Wunderlich and Pehle, *Event-based
//! backpropagation can compute exact gradients for spiking neural networks*, Scientific Reports
//! 11:12829, 2021) for neurons that fire more than once.
//!
//! # Why it is in a neuromorphic crate
//!
//! One spike per neuron per inference is the cheapest code a spiking chip can run, and the
//! Göltz et al. networks were trained this way and run on the `BrainScaleS-2` analog system. What
//! the computation costs is a count: [`Network::forward`] reports the spikes, which can never
//! exceed the neurons.
//!
//! # The closed forms this module is checked against
//!
//! - **The spike time.** Each model's closed form equals the first upward threshold crossing found
//!   by brute force — a scan of the explicit membrane sum, then bisection — on random inputs with
//!   weights of both signs, including inputs that arrive after the spike and must not count.
//!   For the non-leaky model it also equals Mostafa's expression literally.
//! - **The gradient.** [`gradients`] equals central finite differences of the closed-form spike
//!   time in every weight and every input time; inputs outside the causal set have exactly zero
//!   gradient; and for the non-leaky model the weight gradient equals Mostafa's
//!   `τ (z_p − z_out)/(z_out (Σ_C w − θ))`.
//! - **The network.** The gradient of the loss with respect to every weight of a two-layer network,
//!   by backpropagation through spike times, equals finite differences of the loss.
//! - **Learning.** Gradient descent on first-spike times solves XOR — the task of Mostafa's paper —
//!   for both neuron models, deterministically by seed, with one spike or none per neuron.
//!
//! # What this module has NOT reproduced
//!
//! - The `MNIST` results of either paper, or anything on the `BrainScaleS-2` hardware.
//! - `EventProp` itself: neurons here fire at most once, which is what makes the closed forms
//!   possible; the adjoint method handles repeated spikes and is not implemented.
//! - The `τ_m = τ_s` case of Göltz et al., whose spike time needs the Lambert W function.
//! - Robust training. When the causal set changes the spike time is continuous but its gradient
//!   jumps, and a neuron that stops firing has no gradient at all; the remedy here is Mostafa's — a
//!   uniform push on the weights of a silent neuron — and it is a heuristic.

use core::fmt;

use crate::rng::Rng;

/// What went wrong, named rather than guessed around.
#[derive(Debug, Clone, PartialEq)]
pub enum TtfsError {
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
}

impl fmt::Display for TtfsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty { what } => write!(f, "{what} is empty"),
            Self::Dimension { what, got, want } => write!(f, "{what} has length {got}, expected {want}"),
            Self::OutOfRange { what, value, low, high } => {
                write!(f, "{what} = {value} is outside [{low}, {high}]")
            }
            Self::NonFinite { what, index } => write!(f, "{what} is not finite at {index}"),
        }
    }
}

impl std::error::Error for TtfsError {}

fn positive(what: &'static str, v: f64) -> Result<f64, TtfsError> {
    if v.is_finite() && v > 0.0 {
        Ok(v)
    } else {
        Err(TtfsError::OutOfRange { what, value: v, low: f64::MIN_POSITIVE, high: f64::INFINITY })
    }
}

// ---------------------------------------------------------------------------------------------
// One neuron
// ---------------------------------------------------------------------------------------------

/// The neuron model: which post-synaptic potential an input spike leaves.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Kernel {
    /// Non-leaky integration of an exponentially decaying current: `κ(s) = 1 − e^{−s/τ}`.
    NonLeaky {
        /// Synaptic time constant `τ`, seconds.
        tau: f64,
    },
    /// A leaky membrane twice as slow as its synapse: `κ(s) = e^{−s/2τ_s} − e^{−s/τ_s}`.
    Lif {
        /// Synaptic time constant `τ_s`, seconds; the membrane's is `2τ_s`.
        tau_s: f64,
    },
}

/// A first spike.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Spike {
    /// When, seconds.
    pub time: f64,
    /// The membrane's slope `V̇` at the crossing, threshold units per second; positive.
    pub slope: f64,
}

impl Kernel {
    /// The longest time constant of the model, seconds: the scale spike times are measured in.
    #[must_use]
    pub fn tau(&self) -> f64 {
        match *self {
            Self::NonLeaky { tau } => tau,
            Self::Lif { tau_s } => 2.0 * tau_s,
        }
    }

    /// The largest value `κ` reaches: `1` for the non-leaky model (approached, never passed) and
    /// `¼` for the leaky one, at `s = τ_m ln 2`. A single input of weight `w` can fire a neuron
    /// only if `w · peak > θ`.
    #[must_use]
    pub fn peak(&self) -> f64 {
        match self {
            Self::NonLeaky { .. } => 1.0,
            Self::Lif { .. } => 0.25,
        }
    }

    /// The potential `κ(s)` an input leaves `s` seconds after it arrives; zero before.
    #[must_use]
    pub fn psp(&self, s: f64) -> f64 {
        if !(s > 0.0) {
            return 0.0;
        }
        match *self {
            Self::NonLeaky { tau } => -(-s / tau).exp_m1(),
            Self::Lif { tau_s } => (-s / (2.0 * tau_s)).exp() - (-s / tau_s).exp(),
        }
    }

    /// Its slope `κ′(s)`; zero before the input arrives.
    #[must_use]
    pub fn psp_slope(&self, s: f64) -> f64 {
        if !(s > 0.0) {
            return 0.0;
        }
        match *self {
            Self::NonLeaky { tau } => (-s / tau).exp() / tau,
            Self::Lif { tau_s } => -(-s / (2.0 * tau_s)).exp() / (2.0 * tau_s) + (-s / tau_s).exp() / tau_s,
        }
    }

    /// The membrane `Σ_i w_i κ(at − t_i)` at time `at`, from its definition. Silent inputs
    /// (`None`) contribute nothing.
    #[must_use]
    pub fn potential(&self, weights: &[f64], times: &[Option<f64>], at: f64) -> f64 {
        weights.iter().zip(times).map(|(w, t)| t.map_or(0.0, |t| w * self.psp(at - t))).sum()
    }

    /// The time of the FIRST upward crossing of `theta`, in closed form, or `None` if the neuron
    /// never fires. For each prefix of the inputs in order of arrival — a candidate causal set —
    /// the model's equation is solved, and the first solution that falls between that prefix's
    /// last input and the next input's arrival, on a rising membrane, is the spike.
    ///
    /// # Errors
    ///
    /// [`TtfsError::Dimension`] if `weights` and `times` differ in length,
    /// [`TtfsError::NonFinite`] for a bad weight or time, [`TtfsError::OutOfRange`] for a
    /// non-positive threshold or time constant.
    pub fn first_spike(&self, weights: &[f64], times: &[Option<f64>], theta: f64) -> Result<Option<Spike>, TtfsError> {
        if weights.len() != times.len() {
            return Err(TtfsError::Dimension { what: "times", got: times.len(), want: weights.len() });
        }
        if let Some(i) = weights.iter().position(|w| !w.is_finite()) {
            return Err(TtfsError::NonFinite { what: "weights", index: i });
        }
        if let Some(i) = times.iter().position(|t| t.is_some_and(|t| !t.is_finite())) {
            return Err(TtfsError::NonFinite { what: "times", index: i });
        }
        let theta = positive("theta", theta)?;
        positive("tau", self.tau())?;
        let mut order: Vec<(f64, f64)> = times.iter().zip(weights).filter_map(|(t, w)| t.map(|t| (t, *w))).collect();
        order.sort_by(|a, b| a.0.total_cmp(&b.0));
        for k in 1..=order.len() {
            let last = order[k - 1].0;
            let next = order.get(k).map_or(f64::INFINITY, |p| p.0);
            if !(next > last) {
                continue;
            }
            // Times are taken relative to the prefix's last arrival, so the exponentials stay ≤ 1.
            let candidate = match *self {
                Self::NonLeaky { tau } => {
                    let sum: f64 = order[..k].iter().map(|p| p.1).sum();
                    let weighted: f64 = order[..k].iter().map(|p| p.1 * ((p.0 - last) / tau).exp()).sum();
                    // S − e^{−t/τ} A = θ rises iff A > 0, and has a root after `last` iff S > θ.
                    if sum > theta && weighted > 0.0 { Some(last + tau * (weighted / (sum - theta)).ln()) } else { None }
                }
                Self::Lif { tau_s } => {
                    let tau_m = 2.0 * tau_s;
                    let a2: f64 = order[..k].iter().map(|p| p.1 * ((p.0 - last) / tau_m).exp()).sum();
                    let a1: f64 = order[..k].iter().map(|p| p.1 * ((p.0 - last) / tau_s).exp()).sum();
                    // V = a₂x − a₁x² with x = e^{−(t − last)/τ_m} ∈ (0, 1]; rising iff a₂ − 2a₁x < 0.
                    let disc = a2 * a2 - 4.0 * a1 * theta;
                    if a1 > 0.0 && disc >= 0.0 {
                        let x = (a2 + disc.sqrt()) / (2.0 * a1);
                        if x > 0.0 && x <= 1.0 && a2 - 2.0 * a1 * x < 0.0 { Some(last - tau_m * x.ln()) } else { None }
                    } else {
                        None
                    }
                }
            };
            if let Some(t) = candidate.filter(|t| *t >= last && *t <= next) {
                let slope: f64 = order[..k].iter().map(|p| p.1 * self.psp_slope(t - p.0)).sum();
                if slope > 0.0 {
                    return Ok(Some(Spike { time: t, slope }));
                }
            }
        }
        Ok(None)
    }
}

/// The gradient of a first-spike time: `(∂t*/∂w_p, ∂t*/∂t_p)` for every input `p`, by the implicit
/// function theorem at the crossing. Inputs that are silent, or arrive at or after the spike, get
/// exactly zero.
#[must_use]
pub fn gradients(kernel: &Kernel, weights: &[f64], times: &[Option<f64>], spike: &Spike) -> (Vec<f64>, Vec<f64>) {
    let mut d_weight = vec![0.0; weights.len()];
    let mut d_time = vec![0.0; weights.len()];
    for (p, (w, t)) in weights.iter().zip(times).enumerate() {
        if let Some(t) = t.filter(|t| *t < spike.time) {
            let s = spike.time - t;
            d_weight[p] = -kernel.psp(s) / spike.slope;
            d_time[p] = w * kernel.psp_slope(s) / spike.slope;
        }
    }
    (d_weight, d_time)
}

// ---------------------------------------------------------------------------------------------
// Layers and networks
// ---------------------------------------------------------------------------------------------

/// A fully connected layer of first-spike neurons.
#[derive(Debug, Clone, PartialEq)]
pub struct Layer {
    /// Inputs.
    pub n_in: usize,
    /// Neurons.
    pub n_out: usize,
    /// Weights, row-major `n_out × n_in`, in units of the threshold.
    pub w: Vec<f64>,
}

/// A feedforward network of first-spike neurons sharing one model and one threshold.
#[derive(Debug, Clone, PartialEq)]
pub struct Network {
    /// The neuron model.
    pub kernel: Kernel,
    /// Threshold, in the units of the weights.
    pub theta: f64,
    /// The layers, input side first.
    pub layers: Vec<Layer>,
}

/// What one input did to the network.
#[derive(Debug, Clone, PartialEq)]
pub struct Forward {
    /// Spike of every neuron of every layer; `None` for a neuron that never fired.
    pub spikes: Vec<Vec<Option<Spike>>>,
    /// Spikes emitted, inputs not counted: at most one per neuron.
    pub spike_count: usize,
}

impl Forward {
    /// The output neuron that fired first, lowest index on a tie; `None` if none fired.
    #[must_use]
    pub fn winner(&self) -> Option<usize> {
        let out = self.spikes.last()?;
        let mut best: Option<(usize, f64)> = None;
        for (j, s) in out.iter().enumerate() {
            if let Some(s) = s
                && best.is_none_or(|b| s.time < b.1)
            {
                best = Some((j, s.time));
            }
        }
        best.map(|b| b.0)
    }
}

/// The loss of one example and its gradient with respect to every weight.
#[derive(Debug, Clone, PartialEq)]
pub struct Backward {
    /// `t_c/τ + ln Σ_j e^{−t_j/τ}` over the output neurons that fired — the cross-entropy of a
    /// softmax over NEGATIVE spike times, so that firing first is being most probable. `None`
    /// when the target neuron did not fire: there is then no finite loss and no gradient.
    pub loss: Option<f64>,
    /// `∂loss/∂w`, one matrix per layer shaped like [`Layer::w`]; zeros when `loss` is `None`.
    pub gradient: Vec<Vec<f64>>,
    /// `(layer, neuron)` of every neuron that did not fire.
    pub silent: Vec<(usize, usize)>,
}

impl Network {
    /// A network with weights drawn uniformly from `[0, 4 · scale · θ/(n_in · peak)]` — positive,
    /// and scaled by the model's [`Kernel::peak`], so that at `scale = 1` the inputs of a layer
    /// arriving together drive a neuron to about twice its threshold whichever model it is — for
    /// the given layer sizes, input size first.
    ///
    /// # Errors
    ///
    /// [`TtfsError::Empty`] for fewer than two sizes or a size of zero,
    /// [`TtfsError::OutOfRange`] for a non-positive threshold, scale or time constant.
    pub fn random(kernel: Kernel, theta: f64, sizes: &[usize], scale: f64, rng: &mut Rng) -> Result<Self, TtfsError> {
        if sizes.len() < 2 || sizes.contains(&0) {
            return Err(TtfsError::Empty { what: "layer sizes (needs two, none zero)" });
        }
        let theta = positive("theta", theta)?;
        let scale = positive("scale", scale)?;
        positive("tau", kernel.tau())?;
        let layers = sizes
            .windows(2)
            .map(|p| {
                let top = 4.0 * scale * theta / (p[0] as f64 * kernel.peak());
                Layer { n_in: p[0], n_out: p[1], w: (0..p[0] * p[1]).map(|_| top * rng.next_f64()).collect() }
            })
            .collect();
        Ok(Self { kernel, theta, layers })
    }

    /// Propagate input spike times (`None` for a silent input) through the network.
    ///
    /// # Errors
    ///
    /// [`TtfsError::Dimension`] for an input of the wrong length, and as [`Kernel::first_spike`].
    pub fn forward(&self, input: &[Option<f64>]) -> Result<Forward, TtfsError> {
        let first = self.layers.first().ok_or(TtfsError::Empty { what: "layers" })?;
        if input.len() != first.n_in {
            return Err(TtfsError::Dimension { what: "input", got: input.len(), want: first.n_in });
        }
        let mut times = input.to_vec();
        let mut spikes = Vec::with_capacity(self.layers.len());
        let mut spike_count = 0;
        for layer in &self.layers {
            let mut out = Vec::with_capacity(layer.n_out);
            for j in 0..layer.n_out {
                out.push(self.kernel.first_spike(&layer.w[j * layer.n_in..(j + 1) * layer.n_in], &times, self.theta)?);
            }
            spike_count += out.iter().flatten().count();
            times = out.iter().map(|s| s.map(|s| s.time)).collect();
            spikes.push(out);
        }
        Ok(Forward { spikes, spike_count })
    }

    /// The loss of one example and its exact gradient, by backpropagation through spike times.
    ///
    /// # Errors
    ///
    /// As [`Network::forward`], plus [`TtfsError::OutOfRange`] for a `target` past the outputs.
    pub fn backward(&self, input: &[Option<f64>], target: usize) -> Result<Backward, TtfsError> {
        let fwd = self.forward(input)?;
        let n_layers = self.layers.len();
        let outputs = self.layers[n_layers - 1].n_out;
        if target >= outputs {
            return Err(TtfsError::OutOfRange { what: "target", value: target as f64, low: 0.0, high: (outputs - 1) as f64 });
        }
        let mut gradient: Vec<Vec<f64>> = self.layers.iter().map(|l| vec![0.0; l.w.len()]).collect();
        let silent: Vec<(usize, usize)> =
            fwd.spikes.iter().enumerate().flat_map(|(l, layer)| layer.iter().enumerate().filter(|(_, s)| s.is_none()).map(move |(j, _)| (l, j))).collect();
        let tau = self.kernel.tau();
        let out = &fwd.spikes[n_layers - 1];
        let Some(hit) = out[target] else {
            return Ok(Backward { loss: None, gradient, silent });
        };
        // Softmax over −t/τ, shifted by the earliest spike so that the largest exponent is zero.
        let earliest = out.iter().flatten().map(|s| s.time).fold(f64::INFINITY, f64::min);
        let expo: Vec<f64> = out.iter().map(|s| s.map_or(0.0, |s| (-(s.time - earliest) / tau).exp())).collect();
        let total: f64 = expo.iter().sum();
        let loss = (hit.time - earliest) / tau + total.ln();
        // ∂loss/∂t_j = (δ_jc − p_j)/τ.
        let mut d_time: Vec<f64> = (0..outputs).map(|j| (f64::from(u8::from(j == target)) - expo[j] / total) / tau).collect();
        for l in (0..n_layers).rev() {
            let layer = &self.layers[l];
            let below: Vec<Option<f64>> = if l == 0 { input.to_vec() } else { fwd.spikes[l - 1].iter().map(|s| s.map(|s| s.time)).collect() };
            let mut d_below = vec![0.0; layer.n_in];
            for j in 0..layer.n_out {
                let Some(spike) = fwd.spikes[l][j] else { continue };
                if d_time[j] == 0.0 {
                    continue;
                }
                let row = &layer.w[j * layer.n_in..(j + 1) * layer.n_in];
                let (dw, dt) = gradients(&self.kernel, row, &below, &spike);
                for p in 0..layer.n_in {
                    gradient[l][j * layer.n_in + p] = d_time[j] * dw[p];
                    d_below[p] += d_time[j] * dt[p];
                }
            }
            d_time = d_below;
        }
        Ok(Backward { loss: Some(loss), gradient, silent })
    }

    /// One step of gradient descent on one example: weights move by `−rate ·` the gradient, with
    /// each layer's gradient rescaled to a largest entry of `clip` if it exceeds that, and every
    /// weight of every silent neuron is raised by `boost · θ/n_in` — Mostafa's remedy for a neuron
    /// with no gradient. Returns what [`Network::backward`] found BEFORE the step.
    ///
    /// # Errors
    ///
    /// As [`Network::backward`], plus [`TtfsError::OutOfRange`] for a non-positive `rate` or
    /// `clip` or a negative `boost`.
    pub fn train(&mut self, input: &[Option<f64>], target: usize, rate: f64, clip: f64, boost: f64) -> Result<Backward, TtfsError> {
        let rate = positive("rate", rate)?;
        let clip = positive("clip", clip)?;
        if !(boost >= 0.0) || !boost.is_finite() {
            return Err(TtfsError::OutOfRange { what: "boost", value: boost, low: 0.0, high: f64::INFINITY });
        }
        let back = self.backward(input, target)?;
        for (layer, g) in self.layers.iter_mut().zip(&back.gradient) {
            let largest = g.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            let shrink = if largest > clip { clip / largest } else { 1.0 };
            for (w, gi) in layer.w.iter_mut().zip(g) {
                *w -= rate * shrink * gi;
            }
        }
        for &(l, j) in &back.silent {
            let layer = &mut self.layers[l];
            let push = boost * self.theta / layer.n_in as f64;
            for w in &mut layer.w[j * layer.n_in..(j + 1) * layer.n_in] {
                *w += push;
            }
        }
        Ok(back)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MODELS: [Kernel; 2] = [Kernel::NonLeaky { tau: 5e-3 }, Kernel::Lif { tau_s: 5e-3 }];

    /// The first upward crossing of `theta` by the explicit membrane sum: a scan, then bisection.
    fn brute_force(kernel: &Kernel, w: &[f64], t: &[Option<f64>], theta: f64) -> Option<f64> {
        let start = t.iter().flatten().copied().fold(f64::INFINITY, f64::min);
        let end = t.iter().flatten().copied().fold(f64::NEG_INFINITY, f64::max) + 12.0 * kernel.tau();
        let steps = 400_000;
        let at = |k: usize| start + (end - start) * k as f64 / steps as f64;
        let mut below = kernel.potential(w, t, at(0)) - theta;
        for k in 1..=steps {
            let now = kernel.potential(w, t, at(k)) - theta;
            if below < 0.0 && now >= 0.0 {
                let (mut lo, mut hi) = (at(k - 1), at(k));
                for _ in 0..100 {
                    let mid = 0.5 * (lo + hi);
                    if kernel.potential(w, t, mid) < theta { lo = mid } else { hi = mid }
                }
                return Some(hi);
            }
            below = now;
        }
        None
    }

    #[test]
    fn the_closed_form_spike_time_is_the_first_upward_crossing() {
        let mut rng = Rng::new(71);
        let (mut fired, mut silent) = (0, 0);
        for kernel in &MODELS {
            for _ in 0..40 {
                let n = 6;
                // Weights of both signs, arrivals spread over a few time constants, one silent input.
                let w: Vec<f64> = (0..n).map(|_| 3.0 * rng.next_f64() - 0.8).collect();
                let mut t: Vec<Option<f64>> = (0..n).map(|_| Some(20e-3 * rng.next_f64())).collect();
                t[2] = None;
                let closed = kernel.first_spike(&w, &t, 1.0).unwrap();
                let brute = brute_force(kernel, &w, &t, 1.0);
                match (closed, brute) {
                    (Some(s), Some(b)) => {
                        assert!((s.time - b).abs() < 1e-12, "{kernel:?}: closed form {}, brute force {b}", s.time);
                        assert!((kernel.potential(&w, &t, s.time) - 1.0).abs() < 1e-12, "the membrane at the spike is the threshold");
                        assert!(s.slope > 0.0);
                        fired += 1;
                    }
                    (None, None) => silent += 1,
                    other => panic!("{kernel:?}: the two disagree on whether it fires: {other:?}"),
                }
            }
        }
        assert!(fired > 30 && silent > 3, "{fired} fired and {silent} stayed silent: both cases need covering");
    }

    #[test]
    fn the_non_leaky_spike_time_is_mostafas_expression() {
        let kernel = Kernel::NonLeaky { tau: 1.0 };
        // Three inputs at t = 0, 0.5, 3; the third arrives too late to matter.
        let (w, t) = ([0.9, 0.8, 5.0], [Some(0.0), Some(0.5), Some(3.0)]);
        let s = kernel.first_spike(&w, &t, 1.0).unwrap().unwrap();
        // e^{t} = (0.9·1 + 0.8·e^{0.5})/(1.7 − 1).
        let z_out = (0.9 + 0.8 * 0.5f64.exp()) / 0.7;
        assert!((s.time - z_out.ln()).abs() < 1e-15 && s.time < 3.0);
        let (dw, dt) = gradients(&kernel, &w, &t, &s);
        // ∂t/∂w_p = (z_p − z_out)/(z_out (Σ_C w − 1)); ∂t/∂t_p = w_p z_p/(z_out (Σ_C w − 1)).
        assert!((dw[0] - (1.0 - z_out) / (z_out * 0.7)).abs() < 1e-14);
        assert!((dw[1] - (0.5f64.exp() - z_out) / (z_out * 0.7)).abs() < 1e-14);
        assert!((dt[1] - 0.8 * 0.5f64.exp() / (z_out * 0.7)).abs() < 1e-14);
        assert_eq!((dw[2], dt[2]), (0.0, 0.0), "an input after the spike did not cause it");
        assert!(dw[0] < 0.0 && dw[1] < 0.0, "more weight, earlier spike");
        // With only the first input the weights never reach the threshold: silent.
        assert_eq!(kernel.first_spike(&[0.9], &[Some(0.0)], 1.0).unwrap(), None);
        // The causal set is found, not assumed: a strong late input fires the cell that the early
        // ones could not, and the spike comes after it.
        let late = kernel.first_spike(&[0.3, 0.3, 5.0], &t, 1.0).unwrap().unwrap();
        assert!(late.time > 3.0);
        assert!((late.time - ((0.3 + 0.3 * 0.5f64.exp() + 5.0 * 3.0f64.exp()) / 4.6).ln()).abs() < 1e-13);
    }

    #[test]
    fn the_gradient_is_the_finite_difference_of_the_spike_time() {
        let mut rng = Rng::new(83);
        let mut checked = 0;
        for kernel in &MODELS {
            for _ in 0..25 {
                let n = 5;
                let w: Vec<f64> = (0..n).map(|_| 0.4 + 1.2 * rng.next_f64()).collect();
                let t: Vec<Option<f64>> = (0..n).map(|_| Some(8e-3 * rng.next_f64())).collect();
                let Some(s) = kernel.first_spike(&w, &t, 1.0).unwrap() else { continue };
                // Skip the rare case in which an input arrives within the perturbation of the spike:
                // there the causal set changes and the gradient jumps.
                if t.iter().flatten().any(|ti| (ti - s.time).abs() < 1e-5) {
                    continue;
                }
                let (dw, dt) = gradients(kernel, &w, &t, &s);
                let time_of = |w: &[f64], t: &[Option<f64>]| kernel.first_spike(w, t, 1.0).unwrap().unwrap().time;
                for p in 0..n {
                    let (hw, ht) = (1e-6, 1e-8);
                    let (mut up, mut down) = (w.clone(), w.clone());
                    up[p] += hw;
                    down[p] -= hw;
                    let fd_w = (time_of(&up, &t) - time_of(&down, &t)) / (2.0 * hw);
                    assert!((dw[p] - fd_w).abs() < 1e-7 * (1.0 + fd_w.abs()), "{kernel:?} ∂t/∂w_{p}: {} vs {fd_w}", dw[p]);
                    let (mut later, mut sooner) = (t.clone(), t.clone());
                    later[p] = Some(t[p].unwrap() + ht);
                    sooner[p] = Some(t[p].unwrap() - ht);
                    let fd_t = (time_of(&w, &later) - time_of(&w, &sooner)) / (2.0 * ht);
                    assert!((dt[p] - fd_t).abs() < 1e-5 * (1.0 + fd_t.abs()), "{kernel:?} ∂t/∂t_{p}: {} vs {fd_t}", dt[p]);
                    checked += 1;
                }
                // Delaying EVERY input by the same amount delays the spike by that amount.
                let causal_sum: f64 = dt.iter().sum();
                assert!((causal_sum - 1.0).abs() < 1e-9 || t.iter().flatten().any(|ti| *ti >= s.time), "Σ ∂t/∂t_p = {causal_sum}");
            }
        }
        assert!(checked > 150, "only {checked} partial derivatives were checked");
    }

    #[test]
    fn backpropagation_through_spike_times_is_the_gradient_of_the_loss() {
        let mut rng = Rng::new(97);
        for kernel in &MODELS {
            let net = Network::random(*kernel, 1.0, &[4, 5, 3], 1.5, &mut rng).unwrap();
            // Inputs close together compared with τ, so that the hidden neurons — which fire early
            // with weights this large — still have all three live inputs in their causal sets.
            let input = [Some(0.0), Some(0.2e-3), None, Some(0.5e-3)];
            let back = net.backward(&input, 1).unwrap();
            let loss = back.loss.expect("a fresh network with positive weights fires");
            // EVERY output fired, so the softmax has three terms and the loss is not the trivial
            // zero of a lone winner — which is what the first draft of this fixture produced.
            assert!(back.silent.iter().all(|(l, _)| *l == 0), "an output neuron was silent: {:?}", back.silent);
            assert!(loss > 0.0, "{loss}");
            let mut compared = 0;
            for l in 0..2 {
                for k in 0..net.layers[l].w.len() {
                    let h = 1e-6;
                    let (mut up, mut down) = (net.clone(), net.clone());
                    up.layers[l].w[k] += h;
                    down.layers[l].w[k] -= h;
                    let (Some(a), Some(b)) = (up.backward(&input, 1).unwrap().loss, down.backward(&input, 1).unwrap().loss) else { continue };
                    let fd = (a - b) / (2.0 * h);
                    let g = back.gradient[l][k];
                    assert!((g - fd).abs() < 1e-5 * (1.0 + fd.abs()), "{kernel:?} layer {l} weight {k}: {g} vs {fd}");
                    compared += 1;
                }
            }
            assert_eq!(compared, 20 + 15);
            // The weights from the silent input have no gradient at all.
            for j in 0..5 {
                assert_eq!(back.gradient[0][j * 4 + 2], 0.0);
            }
            assert!(back.gradient[0].iter().filter(|g| **g != 0.0).count() >= 10, "the first layer's gradient is all but empty");
        }
    }

    #[test]
    fn gradient_descent_on_spike_times_solves_xor() {
        // Mostafa's encoding: a value is early (0) or late (1), beside a reference that is always
        // early; the answer is whichever output fires first.
        let (early, late) = (0.0, 6e-3);
        let cases: Vec<([Option<f64>; 3], usize)> = (0..4)
            .map(|c| {
                let (a, b) = (c & 1, c >> 1);
                ([Some(if a == 1 { late } else { early }), Some(if b == 1 { late } else { early }), Some(early)], a ^ b)
            })
            .collect();
        for kernel in &MODELS {
            let mut rng = Rng::new(5);
            let mut net = Network::random(*kernel, 1.0, &[3, 6, 2], 1.0, &mut rng).unwrap();
            let correct = |net: &Network| cases.iter().filter(|(x, y)| net.forward(x).unwrap().winner() == Some(*y)).count();
            assert!(correct(&net) < 4, "{kernel:?}: the untrained network already solves it");
            let mut solved_at = None;
            for epoch in 0..3000 {
                for (x, y) in &cases {
                    net.train(x, *y, 0.05, 1.0 / kernel.tau(), 0.2).unwrap();
                }
                if correct(&net) == 4 {
                    solved_at = Some(epoch);
                    break;
                }
            }
            assert!(solved_at.is_some(), "{kernel:?}: XOR unsolved after 3000 epochs ({} of 4)", correct(&net));
            // One spike or none per neuron: at most 8 spikes an inference, inputs not counted.
            for (x, _) in &cases {
                let f = net.forward(x).unwrap();
                assert!(f.spike_count <= 8 && f.spike_count >= 2);
            }
        }
    }

    #[test]
    fn bad_arguments_are_refused() {
        let k = Kernel::Lif { tau_s: 5e-3 };
        assert_eq!(k.tau(), 10e-3);
        assert_eq!((k.psp(0.0), k.psp(-1.0), k.psp_slope(-1.0)), (0.0, 0.0, 0.0));
        // The LIF kernel peaks at s = τ_m ln 2 with value ¼, where its slope is zero.
        let peak = 10e-3 * 2f64.ln();
        assert!((k.psp(peak) - 0.25).abs() < 1e-15 && k.psp_slope(peak).abs() < 1e-12);
        assert!((Kernel::NonLeaky { tau: 2.0 }.psp(2.0) - (1.0 - (-1.0f64).exp())).abs() < 1e-15);
        assert!(matches!(k.first_spike(&[1.0], &[], 1.0), Err(TtfsError::Dimension { what: "times", .. })));
        assert!(matches!(k.first_spike(&[f64::NAN], &[Some(0.0)], 1.0), Err(TtfsError::NonFinite { what: "weights", .. })));
        assert!(matches!(k.first_spike(&[1.0], &[Some(f64::INFINITY)], 1.0), Err(TtfsError::NonFinite { what: "times", .. })));
        assert!(matches!(k.first_spike(&[1.0], &[Some(0.0)], 0.0), Err(TtfsError::OutOfRange { what: "theta", .. })));
        assert!(matches!(Kernel::Lif { tau_s: 0.0 }.first_spike(&[1.0], &[Some(0.0)], 1.0), Err(TtfsError::OutOfRange { what: "tau", .. })));
        assert_eq!(k.first_spike(&[], &[], 1.0).unwrap(), None);
        assert_eq!(k.first_spike(&[9.0], &[None], 1.0).unwrap(), None, "a silent input drives nothing");
        // One input of weight 5 on the LIF: V peaks at 5/4 > 1, so it fires, on the way up.
        let s = k.first_spike(&[5.0], &[Some(1e-3)], 1.0).unwrap().unwrap();
        assert!(s.time > 1e-3 && s.time < 1e-3 + peak);
        assert_eq!(k.first_spike(&[3.9], &[Some(1e-3)], 1.0).unwrap(), None, "a peak of 0.975 never reaches threshold");
        let mut rng = Rng::new(1);
        assert!(matches!(Network::random(k, 1.0, &[3], 1.0, &mut rng), Err(TtfsError::Empty { .. })));
        assert!(matches!(Network::random(k, 1.0, &[3, 0], 1.0, &mut rng), Err(TtfsError::Empty { .. })));
        assert!(matches!(Network::random(k, 0.0, &[3, 2], 1.0, &mut rng), Err(TtfsError::OutOfRange { what: "theta", .. })));
        assert!(matches!(Network::random(k, 1.0, &[3, 2], 0.0, &mut rng), Err(TtfsError::OutOfRange { what: "scale", .. })));
        let mut net = Network::random(k, 1.0, &[3, 2], 1.0, &mut rng).unwrap();
        assert!(net.layers[0].w.iter().all(|w| (0.0..=16.0 / 3.0).contains(w)), "4·θ/(3 inputs · ¼)");
        assert_eq!((Kernel::NonLeaky { tau: 1.0 }.peak(), k.peak()), (1.0, 0.25));
        assert!(matches!(net.forward(&[Some(0.0)]), Err(TtfsError::Dimension { what: "input", .. })));
        assert!(matches!(net.backward(&[Some(0.0); 3], 2), Err(TtfsError::OutOfRange { what: "target", .. })));
        assert!(matches!(net.train(&[Some(0.0); 3], 0, 0.0, 1.0, 0.1), Err(TtfsError::OutOfRange { what: "rate", .. })));
        assert!(matches!(net.train(&[Some(0.0); 3], 0, 0.1, 0.0, 0.1), Err(TtfsError::OutOfRange { what: "clip", .. })));
        assert!(matches!(net.train(&[Some(0.0); 3], 0, 0.1, 1.0, -0.1), Err(TtfsError::OutOfRange { what: "boost", .. })));
        // The clip bounds the step: with a clip far below the gradient's largest entry, no weight
        // moves by more than rate · clip, and the largest one moves by exactly that.
        let mut clipped = Network::random(k, 1.0, &[3, 2], 1.0, &mut rng).unwrap();
        let start = clipped.clone();
        let back = clipped.train(&[Some(0.0), Some(1e-3), Some(2e-3)], 0, 0.5, 1e-4, 0.0).unwrap();
        let largest = back.gradient[0].iter().fold(0.0f64, |m, g| m.max(g.abs()));
        assert!(largest > 100.0 * 1e-4, "the unclipped gradient {largest} is not far above the clip");
        let moved = clipped.layers[0].w.iter().zip(&start.layers[0].w).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
        assert!((moved - 0.5e-4).abs() < 1e-15, "the largest step was {moved}, rate · clip is 5e-5");
        // A network that cannot fire has no loss and no gradient, says which neurons were silent,
        // and the boost raises exactly their weights.
        net.layers[0].w = vec![0.0; 6];
        let before = net.clone();
        let back = net.train(&[Some(0.0); 3], 0, 0.1, 1.0, 0.3).unwrap();
        assert_eq!((back.loss, back.silent.clone()), (None, vec![(0, 0), (0, 1)]));
        assert!(back.gradient.iter().flatten().all(|g| *g == 0.0));
        assert!(net.layers[0].w.iter().all(|w| (*w - 0.1).abs() < 1e-15), "0.3 · θ / 3 inputs");
        assert_eq!(before.forward(&[Some(0.0); 3]).unwrap().winner(), None);
        assert_eq!(before.forward(&[Some(0.0); 3]).unwrap().spike_count, 0);
    }

    /// The final `slope > 0.0` test is a real gate, not a restatement of each model's own rising
    /// condition, because the two are computed from DIFFERENT inputs when the crossing lands
    /// exactly on the prefix's last arrival.
    ///
    /// The hole this fills: the suite's crossings all land strictly inside their bracket, where
    /// every `s = t − t_p` is positive and the recomputed sum really does equal
    /// `−(x/τ_m)(a₂ − 2a₁x)`. The candidate filter is `t >= last`, which admits `t == last`, and
    /// there [`Kernel::psp_slope`] returns `0.0` from its own `!(s > 0.0)` guard — so the last
    /// input's term, the dominant one, is dropped from the recomputed slope while the quadratic's
    /// rising test counts it at full weight. `t == last` is exactly `x == 1.0`, which the formula
    /// produces whenever `4a₁θ` is below half an ulp of `a₂²` and `a₂` rounds equal to `a₁`.
    ///
    /// Measured on the inputs below: `a₂ = a₁ = disc = 1.0` exactly, `x = 1.0`, the rising test
    /// `a₂ − 2a₁x = −1.0 < 0` passes, and the recomputed slope is
    /// `−6.177_801_170_635_097e-19` — NEGATIVE. Accepting a non-zero slope instead of a positive
    /// one therefore reports a spike on a FALLING membrane, violating [`Spike::slope`]'s own
    /// "positive" and flipping the sign of every gradient [`gradients`] divides by it.
    #[test]
    fn a_crossing_that_lands_on_its_last_input_is_rejected_by_the_slope_and_not_by_the_model() {
        let k = Kernel::Lif { tau_s: 1.0 };
        let w = [1e-17, 1.0];
        let t = [Some(0.0), Some(3.0)];
        let theta = 1e-17;
        // The prefix {both inputs}, measured from last = 3.0: the 1e-17 term is below half an ulp
        // of 1.0 in every coefficient, so a2, a1 and the discriminant all round to exactly one
        // and the root is exactly one. Spelled here with names of its own, because repeating the
        // module's own lines verbatim would give the mutation harness two copies of text it
        // mutates by exact match — which it reports as NOT-APPLIED, not as a catch.
        let quad_x2 = w[0] * (-1.5f64).exp() + w[1];
        let quad_x1 = w[0] * (-3.0f64).exp() + w[1];
        assert_eq!((quad_x2, quad_x1), (1.0, 1.0));
        let d = quad_x2 * quad_x2 - 4.0 * quad_x1 * theta;
        assert_eq!(d, 1.0);
        let root = (quad_x2 + d.sqrt()) / (2.0 * quad_x1);
        assert_eq!(root, 1.0);
        // The model's own condition says RISING; the honest slope at that instant says falling.
        assert!(quad_x2 - 2.0 * quad_x1 * root < 0.0, "the quadratic's rising test passes");
        let slope = w[0] * k.psp_slope(3.0) + w[1] * k.psp_slope(0.0);
        assert_eq!(slope, -6.177_801_170_635_097e-19);
        assert!(slope < 0.0, "the recomputed slope is negative: {slope}");
        assert_eq!(k.first_spike(&w, &t, theta).unwrap(), None, "a falling membrane is not a spike");
    }


    /// Each error's text names the quantity that was wrong and says which way round it was wrong:
    /// the dimension message reports the length SUPPLIED first and the length REQUIRED second, and
    /// the empty message says the count was empty.
    ///
    /// The hole this fills: every check in the suite matches on the VARIANT and its `what` field —
    /// `Err(TtfsError::Dimension { what: "times", .. })` — and this review did not locate a test
    /// that renders one. The message is the only part of an error a caller at a terminal reads, so
    /// a `Display` that swapped the two lengths, or called an empty count full, was invisible: the
    /// swap is not even detectable from the fields, since both are `usize` and both are printed.
    #[test]
    fn each_error_says_what_was_wrong_and_which_way_round_it_was_wrong() {
        let k = Kernel::Lif { tau_s: 5e-3 };
        // Two weights, one time: the array named in the message is the one that was short.
        let dimension = k.first_spike(&[1.0, 2.0], &[Some(0.0)], 1.0).unwrap_err();
        assert_eq!(dimension.to_string(), "times has length 1, expected 2");
        let mut rng = Rng::new(3);
        let empty = Network::random(k, 1.0, &[4], 1.0, &mut rng).unwrap_err();
        assert_eq!(empty.to_string(), "layer sizes (needs two, none zero) is empty");
        let low = f64::MIN_POSITIVE;
        let range = k.first_spike(&[1.0], &[Some(0.0)], -2.0).unwrap_err();
        assert_eq!(range.to_string(), format!("theta = -2 is outside [{low}, inf]"));
        let non_finite = k.first_spike(&[1.0, f64::NAN], &[Some(0.0), Some(1.0)], 1.0).unwrap_err();
        assert_eq!(non_finite.to_string(), "weights is not finite at 1");
    }

    /// No infinity passes a guard as a positive number: not a time constant, not a threshold, not a
    /// weight, not a learning rate, a clip or a boost. A refused call leaves the network untouched.
    ///
    /// The hole this fills: every rejection the suite asks for is a `NaN` or a zero, and both of
    /// those are refused by the comparisons alone — `0.0 > 0.0` is false and every comparison with
    /// `NaN` is false. Infinity is the one value that passes `v > 0.0` and is not `NaN`, so the
    /// finiteness half of each guard was never read. What it prevents: an infinite `tau` makes
    /// every `Kernel::psp` exactly zero, so the neuron is silent for ever and `first_spike` reports
    /// a perfectly ordinary `Ok(None)`; an infinite weight makes the non-leaky ratio `inf/inf`,
    /// whose `NaN` the candidate filter drops, so that too reports `Ok(None)`; and an infinite
    /// boost writes infinities into every weight of every silent neuron and returns `Ok`.
    #[test]
    fn no_infinity_passes_a_guard_as_a_positive_number() {
        let k = Kernel::Lif { tau_s: 5e-3 };
        let inf = f64::INFINITY;
        let slow = Kernel::NonLeaky { tau: inf };
        assert!(matches!(slow.first_spike(&[1.0], &[Some(0.0)], 1.0), Err(TtfsError::OutOfRange { what: "tau", .. })));
        assert_eq!(slow.psp(1.0), 0.0, "which is why nothing downstream would have noticed");
        let slow_leaky = Kernel::Lif { tau_s: inf };
        assert!(matches!(slow_leaky.first_spike(&[1.0], &[Some(0.0)], 1.0), Err(TtfsError::OutOfRange { what: "tau", .. })));
        assert!(matches!(k.first_spike(&[1.0], &[Some(0.0)], inf), Err(TtfsError::OutOfRange { what: "theta", .. })));
        assert!(matches!(k.first_spike(&[inf], &[Some(0.0)], 1.0), Err(TtfsError::NonFinite { what: "weights", index: 0 })));
        assert!(matches!(k.first_spike(&[1.0, -inf], &[Some(0.0), Some(1e-3)], 1.0), Err(TtfsError::NonFinite { what: "weights", index: 1 })));
        let mut rng = Rng::new(11);
        assert!(matches!(Network::random(k, inf, &[3, 2], 1.0, &mut rng), Err(TtfsError::OutOfRange { what: "theta", .. })));
        assert!(matches!(Network::random(k, 1.0, &[3, 2], inf, &mut rng), Err(TtfsError::OutOfRange { what: "scale", .. })));
        let mut net = Network::random(k, 1.0, &[3, 2], 1.0, &mut rng).unwrap();
        let before = net.clone();
        let x = [Some(0.0), Some(1e-3), Some(2e-3)];
        assert!(matches!(net.train(&x, 0, inf, 1.0, 0.1), Err(TtfsError::OutOfRange { what: "rate", .. })));
        assert!(matches!(net.train(&x, 0, 0.1, inf, 0.1), Err(TtfsError::OutOfRange { what: "clip", .. })));
        assert!(matches!(net.train(&x, 0, 0.1, 1.0, inf), Err(TtfsError::OutOfRange { what: "boost", .. })));
        assert_eq!(net, before, "a refused step moved nothing");
    }

    /// `Kernel::tau` is the model's own longest time constant: the synaptic one for the non-leaky
    /// model, where there is only one, and the MEMBRANE's — twice the synaptic one — for the leaky
    /// model, where there are two.
    ///
    /// The hole this fills: the suite reads `tau()` for the leaky model only, where the factor of
    /// two is the model's definition. For the non-leaky model the value re-enters `first_spike`
    /// only through a positivity check, and `backward` only as the softmax temperature — where a
    /// factor of two rescales the loss and its gradient TOGETHER, so the finite-difference test,
    /// which compares those two to each other, cannot see it either. Reporting `2 tau` there would
    /// halve every published spike time expressed in time constants.
    #[test]
    fn the_non_leaky_models_time_constant_is_the_one_its_own_kernel_decays_with() {
        let k = Kernel::NonLeaky { tau: 5e-3 };
        assert_eq!(k.tau(), 5e-3);
        // κ(τ) = 1 − e^{−1}, one time constant being where the kernel has risen by that much. The
        // right-hand side is the module's own expression at s = τ, which is how this stays exact.
        assert_eq!(k.psp(k.tau()), -(-1.0f64).exp_m1());
        let leaky = Kernel::Lif { tau_s: 5e-3 };
        assert_eq!(leaky.tau(), 2.0 * 5e-3);
        // For the leaky model the same quantity is the MEMBRANE constant, and the kernel peaks at
        // τ ln 2 — which is inside one τ for that model and outside it for the non-leaky one.
        // For the leaky model that same quantity is the MEMBRANE constant, the one its kernel
        // peaks at τ ln 2 of.
        assert!((leaky.psp(leaky.tau() * core::f64::consts::LN_2) - leaky.peak()).abs() < 1e-16);
    }

    /// A root is admissible only inside the window of the inputs that produced it: at or after that
    /// prefix's last arrival, and at or before the next one. Outside it the equation describes a
    /// membrane that does not exist — one missing an input that had already arrived, or carrying one
    /// that had not — and its root is a spike time no simulation would reproduce.
    ///
    /// The hole this fills: the suite's crossings are all interior. Its random inputs are spread
    /// over a few time constants with weights of order one, so no two arrive at the same instant
    /// (which is what empties a window) and the threshold is never so far below the weights that a
    /// prefix's root lands before its own last input. Both ends of the window were therefore
    /// unread, and both fail with the same symptom: a spike reported at a time when the membrane is
    /// nowhere near θ. The fixtures below are extreme on purpose — the guard is a NUMERICAL
    /// admissibility guard, and the regimes that reach it are the ones where the threshold or an
    /// early input falls below the last place of a later one.
    #[test]
    fn a_root_outside_the_window_of_the_inputs_that_produced_it_is_not_a_spike() {
        let k = Kernel::NonLeaky { tau: 1.0 };
        // (a) TWO INPUTS AT THE SAME INSTANT. The prefix that ends at the first of them has an
        // EMPTY window — its last arrival and the next one are the same instant — so it is not
        // solved at all. Solved anyway, it answers exactly that instant, because both θ and the
        // early input fall below half an ulp of 1e17:
        let w = [1.0, 1e17, -2e17];
        let t = [Some(0.0), Some(1.0), Some(1.0)];
        let sum = 1.0 + 1e17;
        let weighted = (-1.0f64).exp() + 1e17; // the first input's weight is one
        assert_eq!((sum, weighted), (1e17, 1e17));
        assert_eq!(1.0 + (weighted / (sum - 1.0)).ln(), 1.0, "the root of that prefix is its own last arrival");
        // At that instant the pair has not yet contributed anything, so the membrane is the first
        // input's alone: 63% of the threshold, not the threshold.
        assert_eq!(k.potential(&w, &t, 1.0), -(-1.0f64).exp_m1());
        assert!(k.potential(&w, &t, 1.0) < 1.0);
        // And it never reaches θ: below 1 − e^{−1} before the pair, driven down by their net −1e17
        // after it. The scan this module checks its closed form against agrees.
        assert_eq!(k.first_spike(&w, &t, 1.0).unwrap(), None);
        assert_eq!(brute_force(&k, &w, &t, 1.0), None);

        // (b) A ROOT BEFORE THE PREFIX'S LAST INPUT. A modest early input, a dominating late one,
        // and a threshold far below the late one's last place: the equation of BOTH inputs puts its
        // crossing 6.5e-10 s before the second input arrived.
        let w = [1.5e8, 2e17];
        let t = [Some(0.0), Some(2.0)];
        let theta = 1e-17;
        let sum = 1.5e8 + 2e17;
        let weighted = 1.5e8 * (-2.0f64).exp() + 2e17;
        let root = 2.0 + (weighted / (sum - theta)).ln();
        assert_eq!(root, 1.999_999_999_351_501_4);
        assert!(root < 2.0);
        // There the second input has not arrived, so the membrane is the first one's alone — 1.3e8
        // threshold units, twenty-five orders of magnitude from θ. Accepting the root would report
        // that as a threshold crossing.
        assert_eq!(k.potential(&w, &t, root), 1.5e8 * -(-root).exp_m1());
        assert!(k.potential(&w, &t, root) > 1e8);
        assert_eq!(k.first_spike(&w, &t, theta).unwrap(), None);
        // What is NOT claimed: that `None` is the whole truth here. The membrane does pass a
        // threshold this small, at about 6.7e-26 s, and the closed form cannot represent it — the
        // ratio rounds to exactly one, the root to exactly the arrival, and κ′ is zero there by its
        // own convention. What this fixture pins is the window: the answer is "no spike I can
        // place", not a spike two seconds later.
        assert_eq!(1.5e8 / (1.5e8 - theta), 1.0);
    }

    /// The larger root of the leaky model's quadratic is a threshold crossing only when the membrane
    /// is RISING there. At a tangency it is the peak touching θ from below, and the quadratic alone
    /// cannot tell the two apart — both are `a₁x² − a₂x + θ = 0` — so the rising test is the only
    /// thing between a neuron that fires and one that grazes the threshold and does not.
    ///
    /// The hole this fills: the recomputed slope is `(x/τ_m)·√disc` exactly, so it agrees with the
    /// rising test everywhere except where the discriminant is zero, and there both are zero: the
    /// final `slope > 0.0` gate catches a tangency only when the rounding goes its way, and the
    /// suite has no tangency in it at all — its nearest fixture is a peak of 0.975 that misses the
    /// threshold outright. Here the rounding goes the other way. Measured: the recomputed slope is
    /// `+1.16e-9`, so the slope gate passes it and the rising test is what rejects it.
    #[test]
    fn the_larger_root_at_a_tangency_is_the_peak_touching_the_threshold_and_not_a_crossing() {
        let k = Kernel::Lif { tau_s: 0.5 };
        // A pair of inputs at t = 1 whose weights sum to exactly 1, and an early input of −1e-8
        // which is below half an ulp of 1e8 and so vanishes from both coefficients.
        let w = [1e8, 1.0 - 1e8, -1e-8];
        let t = [Some(1.0), Some(1.0), Some(0.0)];
        let theta = 0.25;
        let x2 = -1e-8 * (-1.0f64).exp() + 1e8 + (1.0 - 1e8);
        let x1 = -1e-8 * (-2.0f64).exp() + 1e8 + (1.0 - 1e8);
        assert_eq!((x2, x1), (1.0, 1.0));
        assert_eq!(x2 * x2 - 4.0 * x1 * theta, 0.0, "a double root: the quadratic is tangent to θ");
        // x = ½ is e^{−(t−last)/τ_m} at the kernel's peak, τ_m ln 2 after the arrival.
        let root_t = 1.0 - 2.0 * 0.5 * 0.5f64.ln();
        assert_eq!(root_t, 1.693_147_180_559_945_4);
        // The membrane there is below θ, by the early input's own contribution, and that is its
        // largest value anywhere.
        assert!(k.potential(&w, &t, root_t) < theta);
        assert_eq!(k.first_spike(&w, &t, theta).unwrap(), None);
        // Measured by the scan this module checks its closed form against: over the whole run the
        // membrane's largest value is 0.249_999_998_5, so there is no crossing to find anywhere.
        assert_eq!(brute_force(&k, &w, &t, theta), None);
        // Measured on this fixture: the slope the module recomputes at that instant is positive, so
        // the slope gate would accept it. The sum is written in the module's order — earliest
        // arrival first — because that order is what makes it this number.
        let at_peak = k.psp_slope(root_t - 1.0);
        let slope = (-1e-8 * k.psp_slope(root_t)) + 1e8 * at_peak + (1.0 - 1e8) * at_peak;
        assert_eq!(slope, 1.162_720_734_162_996_6e-9);
        assert!(slope > 0.0);
    }

    /// A tie between two outputs is won by the LOWEST index, and the tie is reachable: two output
    /// neurons with identical weight rows see the same inputs in the same order, so their spike
    /// times are the same f64 bit for bit.
    ///
    /// The hole this fills: the suite reads `Forward::winner` only through XOR, where the two
    /// outputs are trained apart and a tie never arises. The convention is written on the method
    /// and was unread, so the comparison could be relaxed to make the HIGHEST index win with every
    /// test still green — and a convention that is documented and unread is one that will change.
    #[test]
    fn two_outputs_that_fire_at_the_same_instant_are_won_by_the_lower_index() {
        let kernel = Kernel::NonLeaky { tau: 5e-3 };
        let net = Network { kernel, theta: 1.0, layers: vec![Layer { n_in: 2, n_out: 3, w: vec![2.0, 0.5, 2.0, 0.5, 0.1, 0.1] }] };
        let f = net.forward(&[Some(0.0), Some(1e-3)]).unwrap();
        let (first, second) = (f.spikes[0][0].expect("row 0 fires"), f.spikes[0][1].expect("row 1 fires"));
        assert_eq!(first.time, second.time, "identical rows, identical arithmetic, identical time");
        assert_eq!(f.spikes[0][2], None, "0.1 + 0.1 never reaches a threshold of 1");
        assert_eq!(f.winner(), Some(0));
        // And the tie is broken by the index alone, not by anything else the spike carries: here
        // the later-indexed neuron is given the larger slope and still loses.
        let hand = Forward {
            spikes: vec![vec![Some(Spike { time: 3e-3, slope: 1.0 }), Some(Spike { time: 3e-3, slope: 9.0 }), None]],
            spike_count: 2,
        };
        assert_eq!(hand.winner(), Some(0));
    }

    /// A network fires its neurons at the threshold it was BUILT with. `Network::theta` is public,
    /// nothing constrains it, and a neuron whose weights reach 1.5 must stay silent in a network of
    /// threshold 2.
    ///
    /// The hole this fills: every network in the suite is built with θ = 1 — four of them, all
    /// `Network::random(kernel, 1.0, ..)` — so `forward` passing the literal 1.0 to the neuron model
    /// instead of `self.theta` is the same arithmetic in every test. The defect is invisible until
    /// somebody scales their weights and threshold together, which is the ordinary way to move a
    /// trained network onto hardware.
    #[test]
    fn the_network_fires_its_neurons_at_the_threshold_it_was_built_with() {
        let kernel = Kernel::NonLeaky { tau: 5e-3 };
        let net = Network { kernel, theta: 2.0, layers: vec![Layer { n_in: 1, n_out: 2, w: vec![3.0, 1.5] }] };
        let f = net.forward(&[Some(0.0)]).unwrap();
        let own = kernel.first_spike(&[3.0], &[Some(0.0)], 2.0).unwrap().expect("3 passes 2");
        assert_eq!(f.spikes[0][0].expect("the first neuron fires").time, own.time);
        assert_eq!(own.time, 5e-3 * 3.0f64.ln(), "τ ln(w/(w − θ)) = τ ln 3");
        assert_eq!(f.spikes[0][1], None, "1.5 never reaches a threshold of 2");
        assert_eq!(f.spike_count, 1);
        // At the literal threshold of one both neurons fire, and the first fires sooner: every
        // reading in this fixture differs between the network's threshold and that literal.
        assert!(kernel.first_spike(&[1.5], &[Some(0.0)], 1.0).unwrap().is_some());
        assert!(kernel.first_spike(&[3.0], &[Some(0.0)], 1.0).unwrap().expect("3 passes 1").time < own.time);
    }

    /// `Forward::spike_count` counts the spikes of EVERY layer, which is what makes it the cost of
    /// one inference; on anything deeper than a perceptron the hidden layers are most of it.
    ///
    /// The hole this fills: the only place the count is read against a network that fires is XOR's
    /// `spike_count <= 8 && spike_count >= 2`, a window wide enough to hold both the whole network's
    /// spikes and the output layer's alone — a [3, 6, 2] network whose two outputs both fire sits
    /// inside it either way. Accumulating over layers and overwriting per layer are then the same
    /// test, and the energy claim this crate exists to make is the count.
    #[test]
    fn the_spike_count_is_every_layers_spikes_and_not_the_last_layers() {
        let mut rng = Rng::new(29);
        let net = Network::random(Kernel::NonLeaky { tau: 5e-3 }, 1.0, &[3, 5, 2], 1.5, &mut rng).unwrap();
        let f = net.forward(&[Some(0.0), Some(0.3e-3), Some(0.6e-3)]).unwrap();
        let hidden = f.spikes[0].iter().flatten().count();
        let out = f.spikes[1].iter().flatten().count();
        assert_eq!((hidden, out), (5, 2), "measured: every hidden neuron and every output fires");
        assert_eq!(f.spike_count, hidden + out);
        assert!(f.spike_count > out, "the hidden layer's spikes are what a last-layer count drops");
    }

    /// A silent output is not in the softmax at all. When the target is the only neuron that fired,
    /// the loss is exactly zero — the cross-entropy of a certainty — and so is every gradient.
    ///
    /// The hole this fills: every network the suite differentiates has ALL of its outputs firing,
    /// which `backpropagation_through_spike_times_is_the_gradient_of_the_loss` asserts outright, so
    /// the value a silent output contributes is never read. Giving it `1.0` puts a neuron that never
    /// fired into the sum with the weight of one that fired FIRST, since the shift makes `exp(0)`
    /// the largest term any neuron can have: this loss becomes ln 2, and the target acquires a
    /// gradient computed against a spike that does not exist.
    #[test]
    fn a_silent_output_is_not_in_the_softmax_at_all() {
        let kernel = Kernel::NonLeaky { tau: 5e-3 };
        let net = Network { kernel, theta: 1.0, layers: vec![Layer { n_in: 1, n_out: 2, w: vec![3.0, 0.0] }] };
        let back = net.backward(&[Some(0.0)], 0).unwrap();
        assert_eq!(back.silent, vec![(0, 1)], "the second output has no weight at all");
        assert_eq!(back.loss, Some(0.0), "the only output that fired is the target: nothing to learn");
        let gradient = back.gradient[0].clone();
        assert_eq!(gradient.len(), 2, "one weight per output, so the check below reads two numbers");
        assert!(gradient.iter().all(|g| *g == 0.0), "measured gradient {gradient:?}");
    }

    /// The clip bounds the step by the largest gradient entry in MAGNITUDE. A gradient whose
    /// largest entry is negative is reachable and not rare — the target neuron's whole row is
    /// negative by construction, since `∂t/∂w` is negative for every causal input and the target's
    /// own `∂loss/∂t` is positive — and measuring it signed lets that row move by the ratio of the
    /// two maxima, or by the whole unclipped gradient when every entry is negative and the fold
    /// returns its own starting zero.
    ///
    /// The hole this fills: the suite has one clip fixture, a random network at seed 1, and
    /// measured on it the largest entry of the first layer's gradient is POSITIVE
    /// (`+0.014_957_297_621_445_704`, which is also its largest magnitude). A fold over signed
    /// values and a fold over magnitudes return the same number there, so the two cannot be told
    /// apart. Here the target is the LATER of the two outputs, whose row carries both the larger
    /// `−∂t/∂w` and the shallower membrane, and the largest magnitude is negative.
    #[test]
    fn the_clip_bounds_the_largest_gradient_by_magnitude_and_not_by_sign() {
        let kernel = Kernel::NonLeaky { tau: 5e-3 };
        let mut net = Network { kernel, theta: 1.0, layers: vec![Layer { n_in: 2, n_out: 2, w: vec![2.0, 2.0, 1.2, 1.2] }] };
        let start = net.clone();
        let (rate, clip) = (0.5, 1e-3);
        let back = net.train(&[Some(0.0), Some(0.0)], 1, rate, clip, 0.0).unwrap();
        let by_magnitude = back.gradient[0].iter().fold(0.0f64, |m, g| m.max(g.abs()));
        let by_sign = back.gradient[0].iter().fold(0.0f64, |m, g| m.max(*g));
        assert_eq!(by_magnitude, 0.167_410_714_285_714_36, "measured: the target's row, negative");
        assert_eq!(by_sign, 0.046_874_999_999_999_98, "measured: the other row, positive and smaller");
        assert!(by_magnitude > clip, "the fixture is above the clip, so the clip is what decides");
        let moved = net.layers[0].w.iter().zip(&start.layers[0].w).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
        // The largest step is rate · clip. The tolerance is the ulp of the weights it is stored
        // into — all below 4 in magnitude, so 4·2^−52 — not a number chosen to fit.
        assert!((moved - rate * clip).abs() <= 4.0 * f64::EPSILON, "largest step {moved}, rate · clip {}", rate * clip);
        // Measuring the maximum signed instead would scale every step by the ratio of the two,
        // which is 3.57 here and unbounded in general.
        assert!(rate * (clip / by_sign) * by_magnitude > 3.0 * rate * clip);
    }

    /// The push a silent neuron's weights get is a fraction of the THRESHOLD: `boost · θ / n_in`, so
    /// that `boost = 1` is one threshold's worth of drive spread over the inputs whatever θ is.
    ///
    /// The hole this fills: every network in the suite has θ = 1, where multiplying by θ and not
    /// multiplying by it are the same arithmetic. At θ = 2 they differ by a factor of two, and an
    /// unscaled push is half the drive Mostafa's remedy asks for — a silent neuron stays silent for
    /// twice as many epochs, which a test that only asks whether XOR is eventually solved cannot
    /// see, and which vanishes altogether as θ shrinks.
    #[test]
    fn the_silent_neurons_boost_is_a_fraction_of_the_threshold_and_not_of_one() {
        let mut net = Network {
            kernel: Kernel::Lif { tau_s: 5e-3 },
            theta: 2.0,
            layers: vec![Layer { n_in: 3, n_out: 2, w: vec![0.0; 6] }],
        };
        let back = net.train(&[Some(0.0), Some(1e-3), Some(2e-3)], 1, 0.1, 1.0, 0.3).unwrap();
        assert_eq!((back.loss, back.silent.clone()), (None, vec![(0, 0), (0, 1)]));
        let push = 0.3 * 2.0 / 3.0;
        let raised = &net.layers[0].w;
        assert_eq!(raised.len(), 6, "two silent neurons of three inputs, so the check below reads six");
        assert!(raised.iter().all(|w| *w == push), "0.3 · θ / 3 inputs, measured {raised:?}");
        assert!(push > 0.3 / 3.0, "and it is θ times the unscaled push, not the unscaled push");
    }

    /// `Network::train` reports the step it took: the loss and gradient of the network that produced
    /// the step, not of the network the step left behind.
    ///
    /// The hole this fills: nothing in the suite reads what `train` returns except
    /// `bad_arguments_are_refused`, and there the network's weights are all zero — every neuron
    /// silent, no loss, no gradient, and a boost that raises both neurons equally — so the value
    /// before the step and the value after it are the same `Backward`. A training loop that logged
    /// the returned loss would be logging a curve one step ahead of the weights that produced it,
    /// and recomputing it also doubles the cost of every step.
    #[test]
    fn train_reports_the_step_it_took_and_not_the_one_it_would_take_next() {
        let mut rng = Rng::new(37);
        let mut net = Network::random(Kernel::Lif { tau_s: 5e-3 }, 1.0, &[3, 4, 2], 1.5, &mut rng).unwrap();
        let x = [Some(0.0), Some(0.4e-3), Some(0.9e-3)];
        let before = net.clone();
        let expected = before.backward(&x, 0).unwrap();
        let returned = net.train(&x, 0, 0.5, 10.0, 0.0).unwrap();
        assert_eq!(returned, expected, "the loss, the gradient and the silent list, as they were");
        let after = net.backward(&x, 0).unwrap();
        assert_ne!(after.loss, expected.loss, "the step moved the loss, so the two readings differ");
        let (a, b) = (after.loss.expect("still fires"), expected.loss.expect("fired"));
        assert!(a < b, "measured: {b} before the step, {a} after it");
    }
}
