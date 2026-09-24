//! Inhibitory spike-timing-dependent plasticity: the rule by which inhibition learns to balance
//! excitation, and the target firing rate it sets in closed form.
//!
//! # The rule
//!
//! Vogels, Sprekeler, Zenke, Clopath and Gerstner, *Inhibitory plasticity balances excitation and
//! inhibition in sensory pathways and memory networks*, Science 334:1569–1573, 2011. Each synapse
//! from an inhibitory neuron `j` onto a neuron `i` keeps two traces, each jumping by one at its
//! neuron's spike and decaying with `τ_STDP`, and changes its weight at every spike of either side:
//!
//! ```text
//! presynaptic spike:   w ← w + η (x_post − α)
//! postsynaptic spike:  w ← w + η x_pre
//! α = 2 ρ₀ τ_STDP
//! ```
//!
//! Near-coincident spikes in either order strengthen the synapse; every presynaptic spike also
//! weakens it by `ηα`. The paper's values are `τ_STDP = 20 ms` and a target `ρ₀ = 5 Hz`, so `α = 0.2`.
//!
//! # Why the target is `ρ₀`, exactly
//!
//! For presynaptic and postsynaptic trains that are independent and stationary at rates `ν_pre` and
//! `ν_post`, each trace's expectation is its rate times `τ`, so the expected drift is
//!
//! ```text
//! E[dw/dt] = η ν_pre (ν_post τ − α) + η ν_post (ν_pre τ) = η ν_pre (2 ν_post τ − α)
//! ```
//!
//! — [`Vogels::expected_drift`] — which vanishes at `ν_post = α/(2τ) = ρ₀` whatever the presynaptic
//! rate. Inhibition is strengthened while the neuron fires above `ρ₀` and weakened below it, so the
//! rule drives the postsynaptic rate towards `ρ₀`: the tests measure the drift against that formula
//! on independent Poisson trains.
//!
//! ⚠ **An inhibitory synapse is not independent of the neuron it inhibits, and the rate it settles
//! at is above `ρ₀`.** Each presynaptic spike lowers the postsynaptic trace that the depression at
//! the next presynaptic spike reads, so depression is undercounted. The tests let a leaky
//! integrate-and-fire neuron, overdriven by excitation, learn its inhibition: with twenty
//! inhibitory inputs a 5 Hz target settles at 6.5 Hz, with a hundred at 5.6 Hz, and the offset does
//! not move with the learning rate: it shrinks as each input's share of the inhibition does.

use core::fmt;

/// Why the rule could not be built or applied.
#[derive(Debug, Clone, PartialEq)]
pub enum IstdpError {
    /// A parameter that must be finite and positive was not.
    NotPositive {
        /// Which.
        what: &'static str,
        /// Its value.
        value: f64,
    },
    /// A value that must be finite was not.
    NonFinite {
        /// Which.
        what: &'static str,
        /// Its value.
        value: f64,
    },
}

impl fmt::Display for IstdpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotPositive { what, value } => write!(f, "{what} = {value} must be finite and positive"),
            Self::NonFinite { what, value } => write!(f, "{what} = {value} is not finite"),
        }
    }
}

impl std::error::Error for IstdpError {}

fn positive(what: &'static str, value: f64) -> Result<f64, IstdpError> {
    if value.is_finite() && value > 0.0 { Ok(value) } else { Err(IstdpError::NotPositive { what, value }) }
}

/// The Vogels et al. rule.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vogels {
    /// The learning rate `η`, weight units per unit of trace.
    pub eta: f64,
    /// The trace time constant `τ_STDP`, seconds.
    pub tau: f64,
    /// The target postsynaptic rate `ρ₀`, hertz.
    pub rho0: f64,
}

impl Vogels {
    /// The paper's `τ_STDP = 20 ms` and `ρ₀ = 5 Hz`, with learning rate `eta`.
    ///
    /// # Errors
    ///
    /// [`IstdpError::NotPositive`] for an `eta` that is not.
    pub fn science_2011(eta: f64) -> Result<Self, IstdpError> {
        Self::new(eta, 0.020, 5.0)
    }

    /// A rule with learning rate `eta`, trace time constant `tau` seconds and target `rho0` hertz.
    ///
    /// # Errors
    ///
    /// [`IstdpError::NotPositive`] for any of the three that is not finite and positive.
    pub fn new(eta: f64, tau: f64, rho0: f64) -> Result<Self, IstdpError> {
        Ok(Self { eta: positive("eta", eta)?, tau: positive("tau", tau)?, rho0: positive("rho0", rho0)? })
    }

    /// The depression per presynaptic spike, in trace units: `α = 2ρ₀τ`.
    #[must_use]
    pub fn alpha(&self) -> f64 {
        2.0 * self.rho0 * self.tau
    }

    /// The weight after a presynaptic spike, given the postsynaptic trace: `w + η(x_post − α)`,
    /// floored at zero — an inhibitory weight does not turn excitatory.
    #[must_use]
    pub fn on_pre(&self, w: f64, x_post: f64) -> f64 {
        (w + self.eta * (x_post - self.alpha())).max(0.0)
    }

    /// The weight after a postsynaptic spike, given the presynaptic trace: `w + η x_pre`.
    #[must_use]
    pub fn on_post(&self, w: f64, x_pre: f64) -> f64 {
        w + self.eta * x_pre
    }

    /// `E[dw/dt] = η ν_pre (2ν_post τ − α)` for independent stationary trains, per second.
    ///
    /// # Errors
    ///
    /// [`IstdpError::NonFinite`] for a rate that is not finite.
    pub fn expected_drift(&self, nu_pre: f64, nu_post: f64) -> Result<f64, IstdpError> {
        for (what, value) in [("nu_pre", nu_pre), ("nu_post", nu_post)] {
            if !value.is_finite() {
                return Err(IstdpError::NonFinite { what, value });
            }
        }
        Ok(self.eta * nu_pre * (2.0 * nu_post * self.tau - self.alpha()))
    }

    /// The postsynaptic rate at which the expected drift vanishes: `α/(2τ)`, which is `ρ₀`.
    #[must_use]
    pub fn fixed_rate(&self) -> f64 {
        self.alpha() / (2.0 * self.tau)
    }
}

/// A trace that jumps by one at each spike and decays with time constant `tau`, advanced exactly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Trace {
    /// The value.
    pub value: f64,
    /// Seconds.
    pub tau: f64,
}

impl Trace {
    /// A trace at zero.
    ///
    /// # Errors
    ///
    /// [`IstdpError::NotPositive`] for a `tau` that is not finite and positive.
    pub fn new(tau: f64) -> Result<Self, IstdpError> {
        Ok(Self { value: 0.0, tau: positive("tau", tau)? })
    }

    /// Decay for `dt` seconds: `x ← x e^{−dt/τ}`, exact for any `dt`.
    pub fn decay(&mut self, dt: f64) {
        self.value *= (-dt / self.tau).exp();
    }

    /// A spike: `x ← x + 1`.
    pub fn spike(&mut self) {
        self.value += 1.0;
    }
}


#[cfg(test)]
mod tests {
    use super::{IstdpError, Trace, Vogels};
    use crate::rng::Rng;

    /// The paper's parameters give `α = 0.2`; the updates are the rule's two lines; the fixed rate is
    /// the target.
    #[test]
    fn the_rule_is_the_papers() {
        let v = Vogels::science_2011(1e-3).unwrap();
        assert_eq!((v.tau, v.rho0), (0.020, 5.0));
        assert!((v.alpha() - 0.2).abs() < 1e-16);
        assert!((v.fixed_rate() - 5.0).abs() < 1e-14);
        assert!((v.on_pre(1.0, 0.5) - (1.0 + 1e-3 * 0.3)).abs() < 1e-16);
        assert_eq!(v.on_pre(0.0, 0.0), 0.0, "an inhibitory weight is floored at zero");
        assert!((v.on_post(1.0, 0.7) - 1.0007).abs() < 1e-15);
        let w = Vogels::new(0.5, 0.01, 8.0).unwrap();
        assert!((w.alpha() - 0.16).abs() < 1e-16 && (w.fixed_rate() - 8.0).abs() < 1e-14);
        assert!((w.expected_drift(10.0, 3.0).unwrap() - 0.5 * 10.0 * (2.0 * 3.0 * 0.01 - 0.16)).abs() < 1e-15);
        assert!(w.expected_drift(10.0, 8.0).unwrap().abs() < 1e-15, "no drift at the target");
    }

    /// A trace advanced exactly: decay composes, `e^{−a/τ}e^{−b/τ} = e^{−(a+b)/τ}`, and a spike adds one.
    #[test]
    fn a_trace_decays_exactly() {
        let mut t = Trace::new(0.02).unwrap();
        t.spike();
        t.decay(0.01);
        t.decay(0.03);
        assert!((t.value - (-2.0_f64).exp()).abs() < 1e-16);
        t.spike();
        assert!((t.value - (1.0 + (-2.0_f64).exp())).abs() < 1e-15);
    }

    /// Independent Poisson trains drift as `η ν_pre (2ν_post τ − α)`: measured over 2 000 s against
    /// the formula, below, at and above the target rate.
    #[test]
    fn the_drift_of_independent_trains_is_the_formula() {
        let v = Vogels::science_2011(1.0).unwrap();
        let mut rng = Rng::new(2011);
        let nu_pre = 10.0;
        for nu_post in [2.0, 5.0, 12.0] {
            let (blocks, block_s) = (100, 20.0);
            let mut per_block = Vec::with_capacity(blocks);
            let (mut pre, mut post) = (Trace::new(v.tau).unwrap(), Trace::new(v.tau).unwrap());
            let next = |rate: f64, rng: &mut Rng| -((1.0 - rng.next_f64()).ln()) / rate;
            let (mut t_pre, mut t_post, mut now) = (next(nu_pre, &mut rng), next(nu_post, &mut rng), 0.0);
            for b in 0..blocks {
                let end = (b + 1) as f64 * block_s;
                let mut dw = 0.0;
                loop {
                    let t = t_pre.min(t_post);
                    if t >= end {
                        break;
                    }
                    pre.decay(t - now);
                    post.decay(t - now);
                    now = t;
                    if t_pre <= t_post {
                        dw += v.eta * (post.value - v.alpha());
                        pre.spike();
                        t_pre += next(nu_pre, &mut rng);
                    } else {
                        dw += v.eta * pre.value;
                        post.spike();
                        t_post += next(nu_post, &mut rng);
                    }
                }
                per_block.push(dw / block_s);
            }
            let mean = per_block.iter().sum::<f64>() / blocks as f64;
            let var = per_block.iter().map(|d| (d - mean) * (d - mean)).sum::<f64>() / (blocks - 1) as f64;
            let se = (var / blocks as f64).sqrt();
            let want = v.expected_drift(nu_pre, nu_post).unwrap();
            assert!((mean - want).abs() < 4.0 * se, "ν_post = {nu_post}: {mean} against {want} ± {se}");
        }
    }

    /// A neuron overdriven by excitation learns its inhibition down to the target — and settles just
    /// ABOVE it, because an inhibitory synapse is not independent of the neuron it inhibits.
    ///
    /// A leaky integrate-and-fire neuron with a constant excitatory drive that alone would make it
    /// fire at about 50 Hz receives `n` Poisson inhibitory inputs at 10 Hz each, all plastic and
    /// starting at zero; after 400 s of learning its rate is measured over the next 100 s. The drift
    /// formula's fixed point assumes the postsynaptic trace a presynaptic spike reads is independent
    /// of that spike, and here it is not: each inhibitory spike lowers the rate the next depression
    /// reads, so depression undercounts and the rate settles above `ρ₀`. The bias is a property of
    /// the coupling, not of the learning rate, and it shrinks as each input's share of the
    /// inhibition does. Measured: 6.53 Hz for a 5 Hz target with twenty inputs, 5.61 Hz with a
    /// hundred (5.60 Hz at a quarter of the learning rate), 12.79 Hz for a 12 Hz target with a
    /// hundred.
    #[test]
    fn an_overdriven_neuron_settles_just_above_the_target() {
        let mut rates = Vec::new();
        for (rho0, n) in [(5.0, 20), (5.0, 100), (12.0, 100)] {
            let rule = Vogels::new(if n == 20 { 2e-3 } else { 4e-4 }, 0.020, rho0).unwrap();
            let mut rng = Rng::new(7);
            let (dt, tau_m, tau_i) = (1e-4, 0.020, 0.010);
            let (v_rest, v_th, drive) = (0.0, 1.0, 1.35);
            let mut w = vec![0.0; n];
            let mut g = vec![0.0; n];
            let mut pre: Vec<Trace> = (0..n).map(|_| Trace::new(rule.tau).unwrap()).collect();
            let mut post = Trace::new(rule.tau).unwrap();
            let mut v = v_rest;
            let (learn, measure) = (4_000_000usize, 1_000_000usize);
            let mut spikes = 0;
            for step in 0..learn + measure {
                let inhibition: f64 = w.iter().zip(&g).map(|(w, g)| w * g).sum();
                v += dt / tau_m * (v_rest - v + drive - inhibition);
                for gi in &mut g {
                    *gi *= (-dt / tau_i).exp();
                }
                post.decay(dt);
                for j in 0..n {
                    pre[j].decay(dt);
                    if rng.next_f64() < 10.0 * dt {
                        g[j] += 1.0;
                        w[j] = rule.on_pre(w[j], post.value);
                        pre[j].spike();
                    }
                }
                if v >= v_th {
                    v = v_rest;
                    post.spike();
                    for j in 0..n {
                        w[j] = rule.on_post(w[j], pre[j].value);
                    }
                    if step >= learn {
                        spikes += 1;
                    }
                }
            }
            rates.push(spikes as f64 / (measure as f64 * dt));
        }
        let (few, many, fast) = (rates[0], rates[1], rates[2]);
        assert!(few > many && many > 5.0, "the bias is upward and shrinks with more inputs: {rates:?}");
        assert!(many < 5.0 * 1.15 && fast > 12.0 && fast < 12.0 * 1.1, "{rates:?}");
        assert!(few > 5.0 * 1.2, "with twenty inputs the bias is large: {few}");
    }

    /// Every refusal.
    #[test]
    fn every_refusal_names_what_it_refused() {
        assert_eq!(Vogels::new(0.0, 0.02, 5.0).unwrap_err().to_string(), "eta = 0 must be finite and positive");
        assert_eq!(Vogels::new(1.0, f64::INFINITY, 5.0).unwrap_err().to_string(), "tau = inf must be finite and positive");
        assert_eq!(Vogels::new(1.0, 0.02, -5.0).unwrap_err().to_string(), "rho0 = -5 must be finite and positive");
        assert_eq!(Vogels::science_2011(f64::NAN).unwrap_err().to_string(), "eta = NaN must be finite and positive");
        let v = Vogels::science_2011(1.0).unwrap();
        assert_eq!(v.expected_drift(f64::NAN, 1.0).unwrap_err().to_string(), "nu_pre = NaN is not finite");
        assert_eq!(v.expected_drift(1.0, f64::INFINITY), Err(IstdpError::NonFinite { what: "nu_post", value: f64::INFINITY }));
        assert_eq!(Trace::new(0.0).unwrap_err().to_string(), "tau = 0 must be finite and positive");
    }
}
