//! `EventProp`: exact gradients of a spiking network's loss, computed backward in time and only at
//! the spikes.
//!
//! # What the mechanism is
//!
//! Wunderlich and Pehle (*Event-based backpropagation can compute exact gradients for spiking
//! neural networks*, Scientific Reports 11:12829, 2021) treat a network of leaky integrate-and-fire
//! neurons with exponential current synapses,
//!
//! ```text
//! τ_mem V̇ = −V + I        τ_syn İ = −I
//! V_n reaches ϑ  ⇒  V_n ← 0  and  I_m ← I_m + W_mn for every m
//! ```
//!
//! as a hybrid system — smooth flow between spikes, jumps at them — and derive its adjoint. The
//! adjoint variables obey the mirror-image flow in reversed time,
//! `τ_mem λ_V′ = −λ_V`, `τ_syn λ_I′ = −λ_I + λ_V`, and jump only where the forward pass spiked:
//!
//! ```text
//! λ_V,n⁻ = λ_V,n⁺ + [ ϑ λ_V,n⁺ + Σ_m W_mn (λ_V,m − λ_I,m) + ∂L/∂t_k ] / (τ_mem V̇_n⁻)
//! ∂L/∂W_mn = −τ_syn Σ_{spikes of n} λ_I,m(t_spike)
//! ```
//!
//! The spike is not smoothed and no surrogate is used: the gradient is exact, because a spike
//! TIME is a differentiable function of the weights (the implicit function theorem, applied at
//! `V(t_k) = ϑ`) even though the spike is not. The backward pass needs, from the forward one, only
//! the spike times and the slope `V̇⁻` at each — memory proportional to the number of spikes, not
//! to the number of time steps — and does work only at those spikes.
//!
//! [`crate::ttfs`] is the special case in which every neuron fires at most once and the spike
//! time has a closed form. Here neurons fire any number of times, reset, and may be recurrently
//! connected; the spike time is found by bracketing and bisection to the last bit.
//!
//! # Why it is in a neuromorphic crate
//!
//! It is the training algorithm shaped like the hardware: event-driven forward, event-driven
//! backward, sparse in time. It is also the referee for every approximate rule in this crate —
//! [`crate::surrogate`], [`crate::eprop`], [`crate::decolle`] — on any task where a loss on spike
//! times can be written down.
//!
//! # The closed forms this module is checked against
//!
//! - **The flow between events**, `I(s) = I₀ e^{−s/τ_syn}` and
//!   `V(s) = V₀ e^{−s/τ_mem} + I₀ τ_syn/(τ_syn − τ_mem) (e^{−s/τ_syn} − e^{−s/τ_mem})`, against the
//!   differential equation it claims to solve.
//! - **The spike time.** With `τ_mem = 2 τ_syn` a neuron driven by one input spike of weight `w`
//!   crosses threshold at `t = −τ_mem ln x`, `x = (1 + √(1 − 4ϑ/w))/2`; the bisection is checked
//!   against that, and the `EventProp` gradient of `L = t` against its derivative
//!   `dt/dw = −τ_mem ϑ / (x w² √(1 − 4ϑ/w))`.
//! - **The gradient is exact**: on a recurrent network whose neurons each fire several times,
//!   every input and recurrent weight against central finite differences of the loss — for a
//!   loss that weights EVERY spike time, so that no path is left unexercised, and with the
//!   perturbed runs required to have the same spikes in the same order (the condition under which
//!   a finite difference of spike times means anything).
//! - **It learns**: gradient descent on a first-spike-time loss moves an output neuron's spike
//!   onto its target. Measured, and labelled.
//!
//! # What this module has NOT reproduced
//!
//! - Losses on the membrane potential (the paper's `l_V`); the loss here is any differentiable
//!   function of the spike times, supplied as `∂L/∂t_k` per spike.
//! - The paper's benchmarks (Yin-Yang, MNIST), and refractory periods.
//! - Anything at a spike's creation or loss. When a weight change makes a neuron graze threshold
//!   the spike time's derivative diverges (`V̇⁻ → 0`) and then the spike is gone; the loss is
//!   discontinuous there and no gradient method sees across it. [`Network::backward`] returns what
//!   the formulas give and [`Event::slope`] says how close to grazing each spike was.

use core::fmt;

/// The most neurons a network may have (the recurrent matrix is dense).
pub const MAX_NEURONS: usize = 4096;

/// What went wrong, named rather than guessed around.
#[derive(Debug, Clone, PartialEq)]
pub enum EventPropError {
    /// No neurons, no inputs, or more than [`MAX_NEURONS`].
    BadShape,
    /// A parameter outside its range, or equal time constants (the closed forms divide by their
    /// difference).
    OutOfRange {
        /// Which parameter.
        what: &'static str,
        /// Value supplied.
        value: f64,
    },
    /// A weight, time or coefficient that is `NaN` or infinite.
    NonFinite {
        /// Which quantity.
        what: &'static str,
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
    /// A neuron connected to itself. The jump condition used here assumes a spike does not change
    /// the spiking neuron's own current.
    SelfConnection {
        /// The neuron.
        neuron: usize,
    },
    /// Input spikes out of time order, before zero, or from an input that does not exist.
    BadInput {
        /// Index into the input list.
        index: usize,
    },
    /// The run emitted more events than allowed — a recurrent network exciting itself without end.
    TooManyEvents {
        /// The limit that was hit.
        limit: usize,
    },
    /// A neuron whose first spike the loss needs did not fire.
    Silent {
        /// The neuron.
        neuron: usize,
    },
}

impl fmt::Display for EventPropError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadShape => f.write_str("a network needs at least one input and between one and MAX_NEURONS neurons"),
            Self::OutOfRange { what, value } => write!(f, "{what} = {value} is out of range"),
            Self::NonFinite { what } => write!(f, "{what} is not finite"),
            Self::Shape { what, got, want } => write!(f, "{what} has {got} entries, not {want}"),
            Self::SelfConnection { neuron } => write!(f, "neuron {neuron} is connected to itself"),
            Self::BadInput { index } => write!(f, "input spike {index} is out of order, negative, or from an input that does not exist"),
            Self::TooManyEvents { limit } => write!(f, "more than {limit} events: the network is exciting itself without end"),
            Self::Silent { neuron } => write!(f, "neuron {neuron} did not fire, so a loss on its first spike is undefined"),
        }
    }
}

impl std::error::Error for EventPropError {}

/// What an event was.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    /// A spike arriving on this input.
    Input(usize),
    /// A spike of this neuron.
    Neuron(usize),
}

/// One event of a forward pass.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Event {
    /// When, seconds.
    pub time: f64,
    /// What.
    pub source: Source,
    /// For a neuron's spike, `τ_mem V̇⁻ = I − ϑ` at the crossing — positive, and small when the
    /// neuron only grazed threshold. Zero for an input spike.
    pub slope: f64,
}

/// A forward pass: every event in time order.
#[derive(Debug, Clone, PartialEq)]
pub struct Record {
    /// The events.
    pub events: Vec<Event>,
    /// When the run ended, seconds.
    pub t_end: f64,
}

impl Record {
    /// The spike times of neuron `n`, in order.
    #[must_use]
    pub fn spikes_of(&self, n: usize) -> Vec<f64> {
        self.events.iter().filter(|e| e.source == Source::Neuron(n)).map(|e| e.time).collect()
    }

    /// The order of events with the times left out: what two runs must share for a finite
    /// difference between them to mean anything.
    #[must_use]
    pub fn order(&self) -> Vec<Source> {
        self.events.iter().map(|e| e.source).collect()
    }
}

/// `∂L/∂W` for both weight matrices.
#[derive(Debug, Clone, PartialEq)]
pub struct Gradient {
    /// Row-major `n × n_in`.
    pub w_in: Vec<f64>,
    /// Row-major `n × n`.
    pub w: Vec<f64>,
}

/// A network of LIF neurons with exponential current synapses, simulated event by event.
#[derive(Debug, Clone, PartialEq)]
pub struct Network {
    /// Neurons.
    pub n: usize,
    /// Inputs.
    pub n_in: usize,
    /// Membrane time constant, seconds.
    pub tau_mem: f64,
    /// Synaptic time constant, seconds.
    pub tau_syn: f64,
    /// Threshold `ϑ`. The reset is to zero.
    pub theta: f64,
    /// Input weights, row-major `n × n_in`: `w_in[m * n_in + j]` is from input `j` to neuron `m`.
    pub w_in: Vec<f64>,
    /// Recurrent weights, row-major `n × n`: `w[m * n + k]` is from neuron `k` to neuron `m`. The
    /// diagonal must be zero.
    pub w: Vec<f64>,
}

impl Network {
    /// A network with every weight zero.
    ///
    /// # Errors
    ///
    /// [`EventPropError::BadShape`] or [`EventPropError::OutOfRange`].
    pub fn new(n_in: usize, n: usize, tau_mem: f64, tau_syn: f64, theta: f64) -> Result<Self, EventPropError> {
        if n_in == 0 || n == 0 || n > MAX_NEURONS || n_in > MAX_NEURONS {
            return Err(EventPropError::BadShape);
        }
        for (what, value) in [("tau_mem", tau_mem), ("tau_syn", tau_syn), ("theta", theta)] {
            if !(value > 0.0) || !value.is_finite() {
                return Err(EventPropError::OutOfRange { what, value });
            }
        }
        if tau_mem == tau_syn {
            return Err(EventPropError::OutOfRange { what: "tau_mem - tau_syn", value: 0.0 });
        }
        Ok(Self { n, n_in, tau_mem, tau_syn, theta, w_in: vec![0.0; n * n_in], w: vec![0.0; n * n] })
    }

    fn validated(&self) -> Result<(), EventPropError> {
        if self.w_in.len() != self.n * self.n_in {
            return Err(EventPropError::Shape { what: "w_in", got: self.w_in.len(), want: self.n * self.n_in });
        }
        if self.w.len() != self.n * self.n {
            return Err(EventPropError::Shape { what: "w", got: self.w.len(), want: self.n * self.n });
        }
        if !self.w_in.iter().chain(&self.w).all(|w| w.is_finite()) {
            return Err(EventPropError::NonFinite { what: "weight" });
        }
        if let Some(neuron) = (0..self.n).find(|&k| self.w[k * self.n + k] != 0.0) {
            return Err(EventPropError::SelfConnection { neuron });
        }
        Ok(())
    }

    /// The state `(V, I)` a time `s` after `(v0, i0)` with no event between: the closed form of
    /// the module documentation.
    #[must_use]
    pub fn flow(&self, v0: f64, i0: f64, s: f64) -> (f64, f64) {
        let (em, es) = ((-s / self.tau_mem).exp(), (-s / self.tau_syn).exp());
        let k = self.tau_syn / (self.tau_syn - self.tau_mem);
        (v0 * em + i0 * k * (es - em), i0 * es)
    }

    /// How long until a neuron at `(v0, i0)` reaches threshold if nothing else happens, or `None`
    /// if it never does.
    ///
    /// `V(s)` is a sum of two exponentials and so has at most one stationary point. If the
    /// potential is not rising it never will rise above zero again; if it is, it rises to a single
    /// maximum at a time known in closed form, and the crossing — if the maximum clears `ϑ` — lies
    /// before it, where `V` is monotone and bisection cannot miss.
    #[must_use]
    pub fn time_to_threshold(&self, v0: f64, i0: f64) -> Option<f64> {
        if v0 >= self.theta {
            return Some(0.0);
        }
        if !(i0 > v0) {
            return None;
        }
        let k = self.tau_syn / (self.tau_syn - self.tau_mem);
        let (a, b) = (v0 - i0 * k, i0 * k);
        // Stationary point: (A/τ_mem) e^{−s/τ_mem} + (B/τ_syn) e^{−s/τ_syn} = 0.
        let q = -(b * self.tau_mem) / (a * self.tau_syn);
        if !(q > 0.0) || !q.is_finite() {
            return None;
        }
        let peak = q.ln() / (1.0 / self.tau_syn - 1.0 / self.tau_mem);
        if !(peak > 0.0) || !peak.is_finite() || self.flow(v0, i0, peak).0 < self.theta {
            return None;
        }
        let (mut lo, mut hi) = (0.0f64, peak);
        loop {
            let mid = 0.5 * (lo + hi);
            if mid <= lo || mid >= hi {
                return Some(hi);
            }
            if self.flow(v0, i0, mid).0 >= self.theta { hi = mid } else { lo = mid }
        }
    }

    /// Run the network on input spikes `(time, input)` — in time order — until `t_end` or until
    /// `max_events` events have happened.
    ///
    /// # Errors
    ///
    /// [`EventPropError::BadInput`], [`EventPropError::TooManyEvents`], and whatever is wrong with
    /// the weights.
    pub fn forward(&self, inputs: &[(f64, usize)], t_end: f64, max_events: usize) -> Result<Record, EventPropError> {
        self.validated()?;
        if !(t_end > 0.0) || !t_end.is_finite() {
            return Err(EventPropError::OutOfRange { what: "t_end", value: t_end });
        }
        let mut last = 0.0;
        for (index, &(time, input)) in inputs.iter().enumerate() {
            if !(time >= last) || !time.is_finite() || input >= self.n_in {
                return Err(EventPropError::BadInput { index });
            }
            last = time;
        }
        let (mut v, mut i) = (vec![0.0; self.n], vec![0.0; self.n]);
        let (mut t, mut next_input) = (0.0f64, 0usize);
        let mut events = Vec::new();
        loop {
            let spike = (0..self.n).filter_map(|k| self.time_to_threshold(v[k], i[k]).map(|s| (s, k))).min_by(|a, b| a.0.total_cmp(&b.0));
            let input = inputs.get(next_input).filter(|(time, _)| *time <= t_end);
            let spike = spike.filter(|(s, _)| t + s <= t_end);
            let (advance, what) = match (input, spike) {
                (Some(&(time, j)), Some((s, _))) if time <= t + s => (time - t, Source::Input(j)),
                (_, Some((s, k))) => (s, Source::Neuron(k)),
                (Some(&(time, j)), None) => (time - t, Source::Input(j)),
                (None, None) => break,
            };
            if events.len() >= max_events {
                return Err(EventPropError::TooManyEvents { limit: max_events });
            }
            for k in 0..self.n {
                (v[k], i[k]) = self.flow(v[k], i[k], advance);
            }
            t += advance;
            match what {
                Source::Input(j) => {
                    for m in 0..self.n {
                        i[m] += self.w_in[m * self.n_in + j];
                    }
                    next_input += 1;
                    events.push(Event { time: t, source: what, slope: 0.0 });
                }
                Source::Neuron(k) => {
                    events.push(Event { time: t, source: what, slope: i[k] - self.theta });
                    v[k] = 0.0;
                    for m in 0..self.n {
                        i[m] += self.w[m * self.n + k];
                    }
                }
            }
        }
        Ok(Record { events, t_end })
    }

    /// The gradient of a loss on spike times: `dl_dt[k]` is `∂L/∂t_k` for event `k` of `record`
    /// (ignored for input events, whose times are given).
    ///
    /// # Errors
    ///
    /// [`EventPropError::Shape`] or [`EventPropError::NonFinite`] for a bad `dl_dt`, and whatever
    /// is wrong with the weights.
    pub fn backward(&self, record: &Record, dl_dt: &[f64]) -> Result<Gradient, EventPropError> {
        self.validated()?;
        if dl_dt.len() != record.events.len() {
            return Err(EventPropError::Shape { what: "dl_dt", got: dl_dt.len(), want: record.events.len() });
        }
        if !dl_dt.iter().all(|d| d.is_finite()) {
            return Err(EventPropError::NonFinite { what: "dl_dt" });
        }
        let n = self.n;
        let (mut lv, mut li) = (vec![0.0; n], vec![0.0; n]);
        let mut gradient = Gradient { w_in: vec![0.0; n * self.n_in], w: vec![0.0; n * n] };
        let mut t = record.t_end;
        let ratio = self.tau_mem / (self.tau_mem - self.tau_syn);
        for (event, &dl) in record.events.iter().zip(dl_dt).rev() {
            // Carry every adjoint variable back from `t` to the event.
            let gap = t - event.time;
            let (em, es) = ((-gap / self.tau_mem).exp(), (-gap / self.tau_syn).exp());
            for m in 0..n {
                li[m] = li[m] * es + lv[m] * ratio * (em - es);
                lv[m] *= em;
            }
            t = event.time;
            match event.source {
                Source::Input(j) => {
                    for m in 0..n {
                        gradient.w_in[m * self.n_in + j] -= self.tau_syn * li[m];
                    }
                }
                Source::Neuron(k) => {
                    let mut through = 0.0;
                    for m in 0..n {
                        gradient.w[m * n + k] -= self.tau_syn * li[m];
                        through += self.w[m * n + k] * (lv[m] - li[m]);
                    }
                    lv[k] += (self.theta * lv[k] + through + dl) / event.slope;
                }
            }
        }
        Ok(gradient)
    }
}

/// The loss `½ Σ (t_first − target)²` over the first spikes of `outputs`, and its `∂L/∂t_k` for
/// every event of `record`.
///
/// # Errors
///
/// [`EventPropError::Silent`] if an output neuron did not fire; [`EventPropError::Shape`] if
/// `targets` and `outputs` differ in length.
pub fn first_spike_loss(record: &Record, outputs: &[usize], targets: &[f64]) -> Result<(f64, Vec<f64>), EventPropError> {
    if targets.len() != outputs.len() {
        return Err(EventPropError::Shape { what: "targets", got: targets.len(), want: outputs.len() });
    }
    let mut dl_dt = vec![0.0; record.events.len()];
    let mut loss = 0.0;
    for (&neuron, &target) in outputs.iter().zip(targets) {
        let first = record.events.iter().position(|e| e.source == Source::Neuron(neuron)).ok_or(EventPropError::Silent { neuron })?;
        let miss = record.events[first].time - target;
        loss += 0.5 * miss * miss;
        dl_dt[first] += miss;
    }
    Ok((loss, dl_dt))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    #[test]
    fn the_flow_between_events_solves_the_membrane_and_synapse_equations() {
        let net = Network::new(1, 1, 20e-3, 5e-3, 1.0).unwrap();
        let (v0, i0) = (0.3, 2.5);
        assert_eq!(net.flow(v0, i0, 0.0), (v0, i0));
        for s in [1e-3, 4e-3, 15e-3, 60e-3] {
            let h = 1e-7;
            let ((v_up, i_up), (v_down, i_down), (v, i)) = (net.flow(v0, i0, s + h), net.flow(v0, i0, s - h), net.flow(v0, i0, s));
            // τ_mem V̇ = −V + I and τ_syn İ = −I, with the derivatives taken numerically.
            assert!((20e-3 * (v_up - v_down) / (2.0 * h) - (i - v)).abs() < 1e-7, "membrane equation at s = {s}");
            assert!((5e-3 * (i_up - i_down) / (2.0 * h) + i).abs() < 1e-7, "synapse equation at s = {s}");
        }
        // And flows compose: 3 ms then 4 ms is 7 ms.
        let (v1, i1) = net.flow(v0, i0, 3e-3);
        let (two_steps, one_step) = (net.flow(v1, i1, 4e-3), net.flow(v0, i0, 7e-3));
        assert!((two_steps.0 - one_step.0).abs() < 1e-15 && (two_steps.1 - one_step.1).abs() < 1e-15);
    }

    #[test]
    fn one_input_spike_gives_the_closed_form_spike_time_and_its_derivative() {
        // τ_mem = 2 τ_syn: V = w (x − x²) with x = e^{−t/τ_mem}, so the first crossing of ϑ is at
        // x = (1 + √(1 − 4ϑ/w))/2.
        let mut net = Network::new(1, 1, 20e-3, 10e-3, 1.0).unwrap();
        for w in [4.5, 6.0, 20.0] {
            net.w_in[0] = w;
            let root = (1.0f64 - 4.0 / w).sqrt();
            let x = 0.5 * (1.0 + root);
            let want = 5e-3 - 20e-3 * x.ln();
            let record = net.forward(&[(5e-3, 0)], 0.2, 100).unwrap();
            let spikes = record.spikes_of(0);
            assert!(!spikes.is_empty() && (spikes[0] - want).abs() < 1e-15, "w = {w}: {spikes:?} against {want}");
            // L = (the first spike's time): ∂L/∂t_k is one there and zero elsewhere.
            let first = record.events.iter().position(|e| e.source == Source::Neuron(0)).unwrap();
            let mut dl_dt = vec![0.0; record.events.len()];
            dl_dt[first] = 1.0;
            let gradient = net.backward(&record, &dl_dt).unwrap();
            let dt_dw = -20e-3 / (x * w * w * root);
            assert!((gradient.w_in[0] - dt_dw).abs() < 1e-12 * dt_dw.abs(), "w = {w}: {} against {dt_dw}", gradient.w_in[0]);
            // The slope recorded is τ_mem V̇⁻ = I − ϑ = w x² − ϑ.
            assert!((record.events[first].slope - (w * x * x - 1.0)).abs() < 1e-12);
        }
        // The bisection lands on the crossing to the LAST BIT: at the time it returns the
        // potential is at or above threshold, and one float earlier it is below.
        net.w_in[0] = 6.0;
        let s = net.time_to_threshold(0.0, 6.0).unwrap();
        assert!(net.flow(0.0, 6.0, s).0 >= net.theta, "V at the returned time is below threshold");
        let earlier = f64::from_bits(s.to_bits() - 1);
        assert!(net.flow(0.0, 6.0, earlier).0 < net.theta, "one float earlier is already over threshold");
        // At w = 4ϑ the potential only touches threshold; below it the neuron is silent.
        net.w_in[0] = 3.99;
        assert!(net.forward(&[(5e-3, 0)], 0.2, 100).unwrap().spikes_of(0).is_empty());
        assert_eq!(net.time_to_threshold(1.0, 0.0), Some(0.0));
        assert_eq!(net.time_to_threshold(0.5, 0.2), None, "a falling potential never reaches threshold");
        assert_eq!(net.time_to_threshold(-0.5, 0.0), None, "nor does one relaxing up to zero");
    }

    #[test]
    fn an_input_arriving_at_the_instant_of_a_spike_is_delivered_first() {
        // Simultaneity has to be resolved one way and stated. The closed form gives the spike
        // time exactly, so an input can be placed exactly on it: the input is taken first, and
        // the spike — whose crossing has already happened — follows at the same instant.
        let mut net = Network::new(2, 2, 20e-3, 10e-3, 1.0).unwrap();
        net.w_in[0] = 6.0; // input 0 → neuron 0
        net.w_in[3] = 0.5; // row 1, column 1: input 1 → neuron 1, small and late
        let alone = net.forward(&[(0.0, 0)], 0.2, 100).unwrap();
        // The instant to aim at is the one the solver itself lands on — the closed form agrees
        // with it to a float or two, and "the same instant" here has to mean the same bits.
        let when = alone.spikes_of(0)[0];
        let x = 0.5 * (1.0 + (1.0f64 - 4.0 / 6.0).sqrt());
        assert!((when - -20e-3 * x.ln()).abs() < 1e-17);
        let together = net.forward(&[(0.0, 0), (when, 1)], 0.2, 100).unwrap();
        let at_the_instant: Vec<Source> = together.events.iter().filter(|e| e.time == when).map(|e| e.source).collect();
        assert_eq!(at_the_instant, vec![Source::Input(1), Source::Neuron(0)], "{:?}", together.events);
    }

    #[test]
    fn a_run_is_a_prefix_of_a_longer_one_and_holds_nothing_past_its_end() {
        let (net, inputs) = busy();
        let long = net.forward(&inputs, 0.1, 10_000).unwrap();
        for end in [0.01, 0.02, 0.04] {
            let short = net.forward(&inputs, end, 10_000).unwrap();
            assert!(short.events.iter().all(|e| e.time <= end), "an event past {end}");
            assert_eq!(short.events[..], long.events[..short.events.len()], "the shorter run is not a prefix at {end}");
            let past = long.events.iter().filter(|e| e.time > end).count();
            assert!(past > 5, "only {past} events happen after {end}, so the cut-off is not being tested");
            assert_eq!(short.t_end, end);
        }
    }

    /// A recurrent network that fires plenty, with its input.
    fn busy() -> (Network, Vec<(f64, usize)>) {
        let mut rng = Rng::new(17);
        let mut net = Network::new(4, 6, 20e-3, 5e-3, 1.0).unwrap();
        net.w_in.iter_mut().for_each(|w| *w = 0.6 + 1.2 * rng.next_f64());
        for m in 0..6 {
            for k in 0..6 {
                if m != k {
                    net.w[m * 6 + k] = 0.5 * (2.0 * rng.next_f64() - 1.0);
                }
            }
        }
        let mut inputs: Vec<(f64, usize)> = (0..24).map(|_| (0.06 * rng.next_f64(), rng.below(4) as usize)).collect();
        inputs.sort_by(|a, b| a.0.total_cmp(&b.0));
        (net, inputs)
    }

    #[test]
    fn the_gradient_is_exact_on_a_recurrent_network_that_fires_many_times() {
        let (net, inputs) = busy();
        let record = net.forward(&inputs, 0.1, 10_000).unwrap();
        let counts: Vec<usize> = (0..6).map(|k| record.spikes_of(k).len()).collect();
        assert!(counts.iter().all(|&c| c >= 2) && counts.iter().any(|&c| c >= 4), "spike counts {counts:?}: the fixture must fire repeatedly");
        assert!(record.events.iter().all(|e| e.slope > 0.05 || matches!(e.source, Source::Input(_))), "a spike grazes threshold, where no finite difference is meaningful");
        // A loss that weights EVERY spike time, so every path through the network carries gradient.
        let mut rng = Rng::new(5);
        let coefficients: Vec<f64> = record.events.iter().map(|_| 2.0 * rng.next_f64() - 1.0).collect();
        let loss = |net: &Network| {
            let r = net.forward(&inputs, 0.1, 10_000).unwrap();
            assert_eq!(r.order(), record.order(), "the perturbation changed which spikes happen");
            r.events.iter().zip(&coefficients).filter(|(e, _)| matches!(e.source, Source::Neuron(_))).map(|(e, c)| c * e.time).sum::<f64>()
        };
        let gradient = net.backward(&record, &coefficients).unwrap();
        let h = 1e-7;
        let (mut checked, mut largest) = (0, 0.0f64);
        for k in 0..net.w_in.len() {
            let (mut up, mut down) = (net.clone(), net.clone());
            up.w_in[k] += h;
            down.w_in[k] -= h;
            let fd = (loss(&up) - loss(&down)) / (2.0 * h);
            assert!((gradient.w_in[k] - fd).abs() < 1e-6 * fd.abs().max(1e-3), "w_in[{k}]: {} against {fd}", gradient.w_in[k]);
            largest = largest.max(fd.abs());
            checked += 1;
        }
        for k in 0..net.w.len() {
            if k / 6 == k % 6 {
                continue;
            }
            let (mut up, mut down) = (net.clone(), net.clone());
            up.w[k] += h;
            down.w[k] -= h;
            let fd = (loss(&up) - loss(&down)) / (2.0 * h);
            assert!((gradient.w[k] - fd).abs() < 1e-6 * fd.abs().max(1e-3), "w[{k}]: {} against {fd}", gradient.w[k]);
            largest = largest.max(fd.abs());
            checked += 1;
        }
        assert_eq!(checked, 24 + 30);
        assert!(largest > 1e-3, "every derivative checked was all but zero: {largest}");
    }

    #[test]
    fn descending_the_gradient_moves_a_spike_onto_its_target() {
        // Four inputs → five hidden → one output, as blocks of the recurrent matrix.
        let mut rng = Rng::new(3);
        let mut net = Network::new(4, 6, 20e-3, 5e-3, 1.0).unwrap();
        for m in 0..5 {
            for j in 0..4 {
                net.w_in[m * 4 + j] = 2.5 + rng.next_f64();
            }
            net.w[5 * 6 + m] = 2.0 + rng.next_f64();
        }
        let inputs = [(1e-3, 0), (3e-3, 1), (4e-3, 2), (6e-3, 3)];
        let first = |net: &Network| net.forward(&inputs, 0.1, 1_000).unwrap().spikes_of(5)[0];
        let before = first(&net);
        // Ask for the output's first spike 4 ms later than it comes untrained.
        let target = before + 4e-3;
        // A fifth of Polyak's step, η = L/(5|g|²): to first order it takes a tenth off the miss,
        // whatever the gradient's scale — so each step's ratio is itself a check on the gradient.
        let mut misses = vec![before - target];
        for _ in 0..120 {
            let record = net.forward(&inputs, 0.1, 1_000).unwrap();
            let (loss, dl_dt) = first_spike_loss(&record, &[5], &[target]).unwrap();
            let gradient = net.backward(&record, &dl_dt).unwrap();
            // Only the synapses that exist learn: input → hidden, hidden → output.
            let mut norm2 = 0.0;
            for m in 0..5 {
                norm2 += gradient.w[5 * 6 + m].powi(2) + (0..4).map(|j| gradient.w_in[m * 4 + j].powi(2)).sum::<f64>();
            }
            let rate = 0.2 * loss / norm2;
            for m in 0..5 {
                for j in 0..4 {
                    net.w_in[m * 4 + j] -= rate * gradient.w_in[m * 4 + j];
                }
                net.w[5 * 6 + m] -= rate * gradient.w[5 * 6 + m];
            }
            misses.push(first(&net) - target);
        }
        let after = first(&net);
        assert!((misses[1] / misses[0] - 0.9).abs() < 0.03, "the first step took the miss {} → {}", misses[0], misses[1]);
        // MEASURED: every step takes between 3% and 20% off (the prediction is first order, and
        // the spike time is not linear in the weights), and 120 of them take the miss from 4 ms
        // to under a microsecond.
        for pair in misses.windows(2) {
            let ratio = pair[1] / pair[0];
            assert!(ratio > 0.8 && ratio < 0.97, "a step took the miss {} → {}", pair[0], pair[1]);
        }
        assert!((after - target).abs() < 1e-6, "the output's first spike went {before} → {after} for a target of {target}");
    }

    #[test]
    fn the_first_spike_loss_is_half_the_squared_miss_and_refuses_a_silent_neuron() {
        let (net, inputs) = busy();
        let record = net.forward(&inputs, 0.1, 10_000).unwrap();
        let (t0, t3) = (record.spikes_of(0)[0], record.spikes_of(3)[0]);
        let (loss, dl_dt) = first_spike_loss(&record, &[0, 3], &[t0 - 2e-3, t3 + 1e-3]).unwrap();
        assert!((loss - 0.5 * (4e-6 + 1e-6)).abs() < 1e-18);
        let nonzero: Vec<(Source, f64)> = record.events.iter().zip(&dl_dt).filter(|(_, d)| **d != 0.0).map(|(e, d)| (e.source, *d)).collect();
        assert_eq!(nonzero.len(), 2);
        for (source, d) in nonzero {
            let want = if source == Source::Neuron(0) { 2e-3 } else { -1e-3 };
            assert!(matches!(source, Source::Neuron(0 | 3)) && (d - want).abs() < 1e-15);
        }
        assert_eq!(first_spike_loss(&record, &[0], &[0.0, 1.0]), Err(EventPropError::Shape { what: "targets", got: 2, want: 1 }));
        let quiet = Network::new(4, 6, 20e-3, 5e-3, 1.0).unwrap().forward(&inputs, 0.1, 100).unwrap();
        assert_eq!(quiet.events.len(), 24, "with zero weights only the inputs happen");
        assert_eq!(first_spike_loss(&quiet, &[2], &[0.01]), Err(EventPropError::Silent { neuron: 2 }));
    }

    #[test]
    fn a_spike_resets_the_potential_and_feeds_the_others() {
        // Neuron 0 is driven; neuron 1 hears only neuron 0. Until neuron 0 fires, neuron 1 is at
        // rest, and its current jumps by exactly the weight when it does.
        let mut net = Network::new(1, 2, 20e-3, 5e-3, 1.0).unwrap();
        net.w_in[0] = 12.0;
        net.w[2] = 9.0; // from neuron 0 to neuron 1
        let record = net.forward(&[(0.0, 0)], 0.1, 100).unwrap();
        let first = record.spikes_of(0)[0];
        let mine = record.spikes_of(1);
        assert!(!mine.is_empty() && mine[0] > first, "{mine:?} against {first}");
        // Neuron 1's first spike, from rest with I = 9 at `first`: the solver's own crossing time.
        let lone = net.time_to_threshold(0.0, 9.0).unwrap();
        let second = record.spikes_of(0).get(1).copied().unwrap_or(f64::INFINITY);
        assert!(first + lone < second, "the fixture needs neuron 1 to fire before neuron 0 fires again");
        assert!((mine[0] - (first + lone)).abs() < 1e-15);
        // And neuron 0 restarts from zero with the current it had: its second spike follows the
        // first by the crossing time from (0, I(first)).
        let i_at_first = net.flow(0.0, 12.0, first).1;
        assert!((second - first - net.time_to_threshold(0.0, i_at_first).unwrap()).abs() < 1e-15);
    }

    #[test]
    fn bad_networks_inputs_and_runaways_are_refused() {
        assert_eq!(Network::new(0, 3, 20e-3, 5e-3, 1.0), Err(EventPropError::BadShape));
        assert_eq!(Network::new(3, MAX_NEURONS + 1, 20e-3, 5e-3, 1.0), Err(EventPropError::BadShape));
        assert!(matches!(Network::new(1, 1, 0.0, 5e-3, 1.0), Err(EventPropError::OutOfRange { what: "tau_mem", .. })));
        assert!(matches!(Network::new(1, 1, 20e-3, 5e-3, -1.0), Err(EventPropError::OutOfRange { what: "theta", .. })));
        assert!(matches!(Network::new(1, 1, 5e-3, 5e-3, 1.0), Err(EventPropError::OutOfRange { what: "tau_mem - tau_syn", .. })));
        let mut net = Network::new(2, 2, 20e-3, 5e-3, 1.0).unwrap();
        assert_eq!(net.forward(&[(2e-3, 0), (1e-3, 1)], 0.1, 100), Err(EventPropError::BadInput { index: 1 }));
        assert_eq!(net.forward(&[(1e-3, 2)], 0.1, 100), Err(EventPropError::BadInput { index: 0 }));
        assert_eq!(net.forward(&[(-1e-3, 0)], 0.1, 100), Err(EventPropError::BadInput { index: 0 }));
        assert!(net.forward(&[], 0.0, 100).is_err() && net.forward(&[], f64::NAN, 100).is_err());
        net.w[0] = 0.5;
        assert_eq!(net.forward(&[], 0.1, 100), Err(EventPropError::SelfConnection { neuron: 0 }));
        net.w[0] = 0.0;
        net.w_in[1] = f64::NAN;
        assert_eq!(net.forward(&[], 0.1, 100), Err(EventPropError::NonFinite { what: "weight" }));
        net.w_in[1] = 0.0;
        net.w_in.pop();
        assert!(matches!(net.forward(&[], 0.1, 100), Err(EventPropError::Shape { what: "w_in", .. })));
        // The limit is enforced exactly: a run of `k` events is refused at a limit of `k − 1` and
        // allowed at `k`. (Checked before the runaway below, so that a build which does not
        // enforce it at all fails here rather than spinning.)
        let (busy, inputs) = busy();
        let record = busy.forward(&inputs, 0.1, 10_000).unwrap();
        assert_eq!(busy.forward(&inputs, 0.1, record.events.len() - 1), Err(EventPropError::TooManyEvents { limit: record.events.len() - 1 }));
        assert_eq!(busy.forward(&inputs, 0.1, record.events.len()).unwrap(), record);
        // Two neurons that excite each other past threshold never stop on their own: each spike
        // raises the other's current by 20, so the intervals shrink faster than they accumulate
        // and the run never reaches its end time. Only the limit stops it — which is why the two
        // checks above, which a build that had lost the limit fails in milliseconds, come first.
        let mut runaway = Network::new(1, 2, 20e-3, 5e-3, 1.0).unwrap();
        runaway.w_in[0] = 20.0;
        runaway.w[1] = 20.0;
        runaway.w[2] = 20.0;
        assert_eq!(runaway.forward(&[(0.0, 0)], 0.05, 500), Err(EventPropError::TooManyEvents { limit: 500 }));
        assert_eq!(busy.backward(&record, &[0.0]), Err(EventPropError::Shape { what: "dl_dt", got: 1, want: record.events.len() }));
        let mut bad = vec![0.0; record.events.len()];
        bad[1] = f64::INFINITY;
        assert_eq!(busy.backward(&record, &bad), Err(EventPropError::NonFinite { what: "dl_dt" }));
        assert!(EventPropError::Silent { neuron: 2 }.to_string().contains("did not fire"));
    }

    /// The `i0 > v0` guard in [`Network::time_to_threshold`] is load-bearing, not a fast path: it
    /// is the only thing keeping the stationary-point search out of the regime where `flow` cannot
    /// evaluate `V` to a useful number of bits. With `tau_syn = tau_mem (1 + 2^-49)` the factor
    /// `k = tau_syn / (tau_syn - tau_mem)` is 5.6e14, `es - em` cancels to a handful of bits, and
    /// the arithmetic reports both a positive `peak` (the true stationary point is at a NEGATIVE
    /// time, because `V` falls monotonically from `v0` toward zero when `i0 <= v0`) and a
    /// `V(peak)` ABOVE `v0`, which no real trajectory does from a non-rising start.
    ///
    /// Why the suite could not see it: every fixture in this module, and the 86,400-start survey
    /// the entry's recorded argument rests on, uses well-separated time constants (20 ms against
    /// 5 ms or 10 ms), where `k` is order one and the subtraction does not cancel. Nothing here
    /// exercised a near-degenerate tau pair, which [`Network::new`] accepts: it refuses only
    /// `tau_mem == tau_syn` exactly.
    #[test]
    fn the_non_rising_guard_is_what_keeps_the_peak_search_out_of_the_degenerate_tau_regime() {
        let tau_mem = 20e-3;
        let tau_syn = tau_mem * (1.0 + 2.0_f64.powi(-49));
        // MEASURED on this machine: the cancelled evaluation of V at the computed stationary
        // point, 100.591..., against a start of 100.0 that the true V never rises above.
        let net = Network::new(1, 1, tau_mem, tau_syn, 100.5).unwrap();
        let v0 = 100.0;
        let i0 = v0 * (1.0 - 2.0_f64.powi(-49));
        assert!(i0 < v0, "the fixture needs a non-rising start");
        // The degeneracy is real, not a typo: k is 5.6e14 and the two exponentials agree to 14
        // digits at the reported peak.
        let k = tau_syn / (tau_syn - tau_mem);
        assert!(k > 5e14, "k = {k}, so this tau pair is not degenerate enough to cancel");
        // What the unguarded path would compute, spelled out so the failure is readable.
        let a = v0 - i0 * k;
        let q = -(i0 * k * tau_mem) / (a * tau_syn);
        let peak = q.ln() / (1.0 / tau_syn - 1.0 / tau_mem);
        let at_peak = net.flow(v0, i0, peak).0;
        assert!(peak > 0.0, "the cancelled peak is {peak}, so this fixture no longer bites");
        assert!(at_peak > net.theta, "the cancelled V(peak) is {at_peak}, below theta = {}", net.theta);
        // `V` is bounded by `v0` on `[0, inf)` from a non-rising start: `tau_mem V' = I - V`, and
        // `I - V` starts non-positive and V decays toward I which decays toward zero. 100.0 is
        // under a threshold of 100.5, so the honest answer is None whatever the arithmetic does at
        // the stationary point.
        assert_eq!(
            net.time_to_threshold(v0, i0),
            None,
            "reported a crossing of theta = {} from V = {v0}, I = {i0}, which never rises",
            net.theta
        );
    }

}
