//! The Spike Response Model: a neuron written as two kernels and a threshold, and the escape
//! noise that turns its hard threshold into a firing probability.
//!
//! # What the mechanism is
//!
//! Gerstner's Spike Response Model (Gerstner and Kistler, *Spiking Neuron Models*, Cambridge
//! University Press, 2002, chapter 4) replaces the differential equation of a leaky integrator
//! with the sum of its own responses:
//!
//! ```text
//! u(t) = Σ_k η(t − t̂_k)  +  Σ_j w_j Σ_f ε(t − t_j^f)
//! ```
//!
//! `ε` is the postsynaptic potential one input spike of unit weight leaves behind, and `η` is the
//! after-potential the neuron's OWN spike leaves behind — a reset, and whatever relative
//! refractoriness follows it. Nothing is integrated: the potential at any time is a sum over the
//! spikes that have happened, which is why this is the form that time-coded learning rules are
//! written in ([`crate::ttfs`], [`crate::eprop`]'s tempotron and `SpikeProp`,
//! [`crate::surrogate`]'s `SLAYER` kernel all differentiate an `ε`).
//!
//! **SRM₀** is the simplification that keeps only the LAST own spike, `u(t) = η(t − t̂) + Σ …`,
//! on the argument that a reset erases what came before it. This module implements both, and
//! measures exactly where that argument fails — see below.
//!
//! **Escape noise** replaces "fires when `u` reaches `θ`" with a hazard rate that rises smoothly
//! through the threshold, `ρ(t) = ρ₀ exp((u(t) − θ)/Δu)`. The neuron becomes a point process; the
//! probability of surviving from `t̂` to `t` without firing is `S = exp(−∫ρ)`, and the interval
//! density is `ρ S`. As `Δu → 0` the hazard becomes a step and the deterministic neuron comes
//! back.
//!
//! # Why it is in a neuromorphic crate
//!
//! It is the formalism the rest of the crate's time-coded learning is written in, and it is the
//! bridge between a simulated membrane and a probabilistic one: the same neuron, as an ODE for a
//! simulator and as a kernel sum for a gradient. The escape-noise form is also how a stochastic
//! spiking chip is modelled — the hazard is what a noisy comparator does.
//!
//! # The closed forms this module is checked against
//!
//! - **The kernel is the membrane's own impulse response.** For `τ_m ≠ τ_s`,
//!   `ε(s) = τ_s/(τ_s − τ_m) · (e^{−s/τ_s} − e^{−s/τ_m})`, which is checked against
//!   [`crate::eventprop::Network::flow`] — an independent implementation, written for a different
//!   purpose, that integrates the same two equations. For `τ_s = 0` it is `e^{−s/τ_m}`, and the
//!   double-exponential form converges on it as `τ_s → 0`.
//! - **Its peak.** `ε` rises to a single maximum at `t* = τ_m τ_s/(τ_m − τ_s) · ln(τ_m/τ_s)`, where
//!   its value is `(τ_s/τ_m)^{τ_m/(τ_m−τ_s)}`; checked against the derivative's root and against a
//!   scan. That height is PROPORTIONAL to `τ_s`, because a synapse that adds `w` to a current
//!   decaying with `τ_s` delivers a charge of `w τ_s` — so the double exponential does NOT tend
//!   to the delta kernel as `τ_s → 0`, it tends to zero. `ε · τ_m/τ_s` is the one that tends to
//!   `e^{−s/τ_m}`, and that convergence is what is checked.
//! - **The full SRM IS the linear neuron with reset by subtraction**, exactly: a direct
//!   integration of `τ_m u̇ = −u + I`, `τ_s İ = −I`, subtracting `θ` at each spike, agrees with
//!   the kernel sum to a tolerance set by the integrator, not by the model.
//! - **SRM₀ is exact for delta synapses and NOT otherwise.** With `τ_s = 0`, resetting to zero
//!   erases everything before the last spike and SRM₀ is exact. With `τ_s > 0` the synaptic
//!   current is NOT reset by a spike, so input that arrived before `t̂` keeps flowing in
//!   afterwards; SRM₀ drops exactly `ε(t − t̂) · Σ_{f < t̂} w e^{−(t̂ − t_f)/τ_s}`, and
//!   [`Srm::carried_current`] is that term. Adding it back closes the gap to rounding — which is
//!   how this module knows it has named the right term rather than a plausible one.
//! - **Escape noise.** At a constant potential the interval density is `ρ e^{−ρt}`: the survivor
//!   function is `e^{−ρt}`, the density integrates to one, the mean interval is `1/ρ`, and drawn
//!   intervals match that mean and that survivor. The hazard's slope is `1/Δu` per volt — a decade
//!   of rate per `Δu · ln 10` of potential.
//! - **The deterministic limit.** As `Δu → 0` the survivor function at a fixed sub-threshold
//!   potential goes to one and at a supra-threshold one to zero, geometrically in `1/Δu`.
//!
//! # What this module has NOT reproduced
//!
//! - Simulation. There is no spike-time solver here: [`crate::eventprop`] has one, and this
//!   module is checked against it rather than repeating it. `own_spikes` is supplied by the
//!   caller.
//! - Adaptation kernels beyond the exponential reset (Gerstner's `η` may be any shape; this one
//!   is `−amplitude · e^{−s/τ_m}` with an absolute refractory period in front), spike-triggered
//!   current kernels, and the Generalized Linear Model's maximum-likelihood fitting.
//! - The population equation and its integral form, which is what SRM was built to support;
//!   [`crate::meanfield`] carries the population side of this crate.

use core::fmt;

/// The most spikes a kernel sum will be taken over.
pub const MAX_SPIKES: usize = 1 << 20;

/// What went wrong, named rather than guessed around.
#[derive(Debug, Clone, PartialEq)]
pub enum SrmError {
    /// A parameter outside its range.
    OutOfRange {
        /// Which parameter.
        what: &'static str,
        /// Value supplied.
        value: f64,
    },
    /// A `NaN` or infinity.
    NonFinite {
        /// Which quantity.
        what: &'static str,
    },
    /// More spikes than [`MAX_SPIKES`].
    TooManySpikes {
        /// How many were supplied.
        got: usize,
    },
    /// Spike times out of order.
    Unsorted {
        /// Which entry is not after the one before it.
        index: usize,
    },
}

impl fmt::Display for SrmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfRange { what, value } => write!(f, "{what} = {value} is out of range"),
            Self::NonFinite { what } => write!(f, "{what} is not finite"),
            Self::TooManySpikes { got } => write!(f, "{got} spikes is more than this module will sum over"),
            Self::Unsorted { index } => write!(f, "spike {index} is not after the one before it"),
        }
    }
}

impl std::error::Error for SrmError {}

fn sorted(what: &'static str, times: &[f64]) -> Result<(), SrmError> {
    if times.len() > MAX_SPIKES {
        return Err(SrmError::TooManySpikes { got: times.len() });
    }
    for (i, t) in times.iter().enumerate() {
        if !t.is_finite() {
            return Err(SrmError::NonFinite { what });
        }
        if i > 0 && *t < times[i - 1] {
            return Err(SrmError::Unsorted { index: i });
        }
    }
    Ok(())
}

/// The two response kernels of a leaky integrator with exponential current synapses.
///
/// `τ_s = 0` is the delta synapse: a presynaptic spike displaces the membrane at once.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Kernel {
    /// Membrane time constant, seconds.
    pub tau_m: f64,
    /// Synaptic time constant, seconds; zero for a delta synapse.
    pub tau_s: f64,
}

impl Kernel {
    /// Build.
    ///
    /// # Errors
    ///
    /// [`SrmError::OutOfRange`] for a non-positive `tau_m`, a negative `tau_s`, or the degenerate
    /// `tau_s == tau_m`, where the closed form divides by their difference.
    pub fn new(tau_m: f64, tau_s: f64) -> Result<Self, SrmError> {
        if !(tau_m > 0.0) || !tau_m.is_finite() {
            return Err(SrmError::OutOfRange { what: "tau_m", value: tau_m });
        }
        if !(tau_s >= 0.0) || !tau_s.is_finite() {
            return Err(SrmError::OutOfRange { what: "tau_s", value: tau_s });
        }
        if tau_s == tau_m {
            return Err(SrmError::OutOfRange { what: "tau_s - tau_m", value: 0.0 });
        }
        Ok(Self { tau_m, tau_s })
    }

    /// Whether this is a delta synapse.
    #[must_use]
    pub fn is_delta(&self) -> bool {
        self.tau_s == 0.0
    }

    /// The postsynaptic potential `s` seconds after one input spike of unit weight; zero before
    /// it, and zero exactly at it for a synapse with a time constant (the membrane has not moved
    /// yet).
    #[must_use]
    pub fn epsilon(&self, s: f64) -> f64 {
        if !(s >= 0.0) {
            return 0.0;
        }
        if self.is_delta() {
            (-s / self.tau_m).exp()
        } else {
            self.tau_s / (self.tau_s - self.tau_m) * ((-s / self.tau_s).exp() - (-s / self.tau_m).exp())
        }
    }

    /// When `ε` peaks: `τ_m τ_s/(τ_m − τ_s) · ln(τ_m/τ_s)`, and zero for a delta synapse.
    #[must_use]
    pub fn peak_time(&self) -> f64 {
        if self.is_delta() {
            0.0
        } else {
            self.tau_m * self.tau_s / (self.tau_m - self.tau_s) * (self.tau_m / self.tau_s).ln()
        }
    }

    /// The value of `ε` at its peak: `(τ_s/τ_m)^{τ_m/(τ_m−τ_s)}`, and one for a delta synapse.
    ///
    /// Note what that says about SIZE. A current synapse that adds `w` to a current decaying with
    /// `τ_s` delivers a charge of `w τ_s`, so its potential peaks at a height PROPORTIONAL TO
    /// `τ_s` — it is not a normalised kernel, and a shorter synapse is a smaller one. The delta
    /// kernel is the charge-normalised limit: `ε · τ_m/τ_s → e^{−s/τ_m}` as `τ_s → 0`, and
    /// [`Kernel::charge_normalised_peak`] is that limit's peak, which does tend to one.
    #[must_use]
    pub fn peak(&self) -> f64 {
        if self.is_delta() {
            1.0
        } else {
            (self.tau_s / self.tau_m).powf(self.tau_m / (self.tau_m - self.tau_s))
        }
    }

    /// The peak of `ε · τ_m/τ_s`, the kernel scaled so that a vanishing synapse becomes the delta
    /// kernel: `(τ_s/τ_m)^{τ_s/(τ_m−τ_s)}`, which tends to one.
    #[must_use]
    pub fn charge_normalised_peak(&self) -> f64 {
        if self.is_delta() {
            1.0
        } else {
            (self.tau_s / self.tau_m).powf(self.tau_s / (self.tau_m - self.tau_s))
        }
    }

    /// The after-potential `s` seconds after the neuron's own spike: `−amplitude · e^{−s/τ_m}`,
    /// and zero before it.
    #[must_use]
    pub fn eta(&self, s: f64, amplitude: f64) -> f64 {
        if s >= 0.0 { -amplitude * (-s / self.tau_m).exp() } else { 0.0 }
    }
}

/// A Spike Response Model neuron: kernels, a threshold, and the size of its own reset.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Srm {
    /// The response kernels.
    pub kernel: Kernel,
    /// Firing threshold.
    pub theta: f64,
    /// How far the neuron's own spike pushes its potential down. `theta` reproduces a reset by
    /// subtraction.
    pub reset: f64,
}

impl Srm {
    /// A neuron whose own spike subtracts exactly its threshold — the linear reset for which the
    /// kernel sum is exact.
    ///
    /// # Errors
    ///
    /// [`SrmError::OutOfRange`] for a non-positive threshold.
    pub fn subtracting(kernel: Kernel, theta: f64) -> Result<Self, SrmError> {
        if !(theta > 0.0) || !theta.is_finite() {
            return Err(SrmError::OutOfRange { what: "theta", value: theta });
        }
        Ok(Self { kernel, theta, reset: theta })
    }

    /// The full kernel sum at time `t`: every own spike's `η` and every input's `ε`.
    ///
    /// `inputs` is `(time, weight)` in non-decreasing time order; `own` likewise.
    ///
    /// # Errors
    ///
    /// [`SrmError::Unsorted`], [`SrmError::NonFinite`] or [`SrmError::TooManySpikes`].
    pub fn potential(&self, t: f64, inputs: &[(f64, f64)], own: &[f64]) -> Result<f64, SrmError> {
        let times: Vec<f64> = inputs.iter().map(|(t, _)| *t).collect();
        sorted("input", &times)?;
        sorted("own", own)?;
        if !t.is_finite() || inputs.iter().any(|(_, w)| !w.is_finite()) {
            return Err(SrmError::NonFinite { what: "t" });
        }
        let drive: f64 = inputs.iter().map(|(f, w)| w * self.kernel.epsilon(t - f)).sum();
        let after: f64 = own.iter().map(|f| self.kernel.eta(t - f, self.reset)).sum();
        Ok(drive + after)
    }

    /// SRM₀ at time `t`: only the LAST own spike's `η`, and only the inputs that arrived after it.
    ///
    /// # Errors
    ///
    /// As [`Srm::potential`].
    pub fn potential_srm0(&self, t: f64, inputs: &[(f64, f64)], last: Option<f64>) -> Result<f64, SrmError> {
        let times: Vec<f64> = inputs.iter().map(|(t, _)| *t).collect();
        sorted("input", &times)?;
        if !t.is_finite() || inputs.iter().any(|(_, w)| !w.is_finite()) {
            return Err(SrmError::NonFinite { what: "t" });
        }
        let t_hat = match last {
            Some(t_hat) if t_hat.is_finite() => t_hat,
            Some(_) => return Err(SrmError::NonFinite { what: "last" }),
            None => f64::NEG_INFINITY,
        };
        let drive: f64 = inputs.iter().filter(|(f, _)| *f >= t_hat).map(|(f, w)| w * self.kernel.epsilon(t - f)).sum();
        let after = if last.is_some() { self.kernel.eta(t - t_hat, self.reset) } else { 0.0 };
        Ok(drive + after)
    }

    /// The synaptic current standing at `t_hat` — what a reset does NOT erase, and exactly what
    /// SRM₀ leaves out: `Σ_{f < t̂} w e^{−(t̂ − t_f)/τ_s}`. Zero for a delta synapse, which is why
    /// SRM₀ is exact there.
    ///
    /// # Errors
    ///
    /// As [`Srm::potential`].
    pub fn carried_current(&self, t_hat: f64, inputs: &[(f64, f64)]) -> Result<f64, SrmError> {
        let times: Vec<f64> = inputs.iter().map(|(t, _)| *t).collect();
        sorted("input", &times)?;
        if !t_hat.is_finite() {
            return Err(SrmError::NonFinite { what: "t_hat" });
        }
        if self.kernel.is_delta() {
            return Ok(0.0);
        }
        Ok(inputs.iter().filter(|(f, _)| *f < t_hat).map(|(f, w)| w * (-(t_hat - f) / self.kernel.tau_s).exp()).sum())
    }
}

/// Escape noise: a hazard rate that rises exponentially through the threshold.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Escape {
    /// The hazard exactly at threshold, per second.
    pub rho0: f64,
    /// How many volts of potential multiply the hazard by `e`.
    pub delta_u: f64,
}

impl Escape {
    /// Build.
    ///
    /// # Errors
    ///
    /// [`SrmError::OutOfRange`] for a non-positive `rho0` or `delta_u`.
    pub fn new(rho0: f64, delta_u: f64) -> Result<Self, SrmError> {
        if !(rho0 > 0.0) || !rho0.is_finite() {
            return Err(SrmError::OutOfRange { what: "rho0", value: rho0 });
        }
        if !(delta_u > 0.0) || !delta_u.is_finite() {
            return Err(SrmError::OutOfRange { what: "delta_u", value: delta_u });
        }
        Ok(Self { rho0, delta_u })
    }

    /// `ρ(u) = ρ₀ exp((u − θ)/Δu)`, per second.
    ///
    /// A potential far above threshold saturates at infinity — `ρ₀` is finite by construction and
    /// `exp` overflows to infinity, not to a `NaN` — and one far below underflows to zero. A
    /// non-finite potential propagates rather than being mapped to anything: it can only have
    /// come from a caller's own arithmetic, since [`Srm::potential`] refuses non-finite inputs,
    /// and turning it into "certain to fire" would hide it.
    #[must_use]
    pub fn hazard(&self, u: f64, theta: f64) -> f64 {
        self.rho0 * ((u - theta) / self.delta_u).exp()
    }

    /// The potential at which the hazard is `rate`: the inverse of [`Escape::hazard`]. `None` for
    /// a non-positive rate.
    #[must_use]
    pub fn potential_for(&self, rate: f64, theta: f64) -> Option<f64> {
        if rate > 0.0 && rate.is_finite() { Some(theta + self.delta_u * (rate / self.rho0).ln()) } else { None }
    }

    /// The probability of surviving `t` seconds at a CONSTANT hazard: `e^{−ρt}`.
    #[must_use]
    pub fn survivor_at(rate: f64, t: f64) -> f64 {
        if t <= 0.0 { 1.0 } else { (-rate * t).exp() }
    }

    /// The interval density at a constant hazard: `ρ e^{−ρt}`.
    #[must_use]
    pub fn interval_density_at(rate: f64, t: f64) -> f64 {
        if t < 0.0 { 0.0 } else { rate * Self::survivor_at(rate, t) }
    }

    /// The survivor function for a potential that varies, `exp(−∫₀ᵗ ρ(u(s)) ds)`, by Simpson's
    /// rule over `panels` panels (rounded up to an even number).
    ///
    /// # Errors
    ///
    /// [`SrmError::OutOfRange`] for a non-positive `t` or fewer than two panels.
    pub fn survivor(&self, u: impl Fn(f64) -> f64, theta: f64, t: f64, panels: usize) -> Result<f64, SrmError> {
        if !(t > 0.0) || !t.is_finite() {
            return Err(SrmError::OutOfRange { what: "t", value: t });
        }
        if panels < 2 {
            return Err(SrmError::OutOfRange { what: "panels", value: panels as f64 });
        }
        let n = panels + panels % 2;
        let h = t / n as f64;
        let mut acc = self.hazard(u(0.0), theta) + self.hazard(u(t), theta);
        for k in 1..n {
            acc += self.hazard(u(h * k as f64), theta) * if k % 2 == 1 { 4.0 } else { 2.0 };
        }
        Ok((-acc * h / 3.0).exp())
    }

    /// An interval drawn at a constant hazard, by inverting the survivor function:
    /// `−ln(1 − U)/ρ`. `None` for a non-positive rate.
    #[must_use]
    pub fn draw_interval(rate: f64, uniform: f64) -> Option<f64> {
        if !(rate > 0.0) || !rate.is_finite() || !(0.0..1.0).contains(&uniform) {
            return None;
        }
        Some(-(1.0 - uniform).ln() / rate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eventprop;
    use crate::rng::Rng;

    #[test]
    fn the_kernel_is_the_membranes_own_impulse_response() {
        // Referee: `eventprop`, which integrates the same two equations for a different purpose
        // and shares no code with this module.
        let (tau_m, tau_s) = (20e-3, 5e-3);
        let k = Kernel::new(tau_m, tau_s).unwrap();
        let net = eventprop::Network::new(1, 1, tau_m, tau_s, 1.0).unwrap();
        let mut largest = 0.0f64;
        for step in 0..=80 {
            let s = f64::from(step) * 1e-3;
            let (v, _) = net.flow(0.0, 1.0, s);
            assert!((k.epsilon(s) - v).abs() < 1e-15, "at {s}: {} against {v}", k.epsilon(s));
            largest = largest.max(v);
        }
        assert!(largest > 0.1, "the kernel never rose far enough to compare anything: {largest}");
        assert_eq!(k.epsilon(-1e-9), 0.0);
        assert_eq!(k.epsilon(0.0), 0.0, "a synapse with a time constant moves no membrane at the instant it arrives");

        // Its peak, against the derivative's root and against a scan.
        let t_peak = k.peak_time();
        assert!((t_peak - tau_m * tau_s / (tau_m - tau_s) * (tau_m / tau_s).ln()).abs() < 1e-18);
        assert!((k.epsilon(t_peak) - k.peak()).abs() < 1e-15);
        let scan = (0..200_000).map(|i| k.epsilon(f64::from(i) * 1e-6)).fold(0.0f64, f64::max);
        assert!((k.peak() - scan).abs() < 1e-9 && k.peak() > 0.1, "peak {} against a scan's {scan}", k.peak());
        // The height is proportional to tau_s: half the synapse, half the potential, to the
        // extent that the exponent's own tau_s dependence allows.
        let shorter = Kernel::new(tau_m, tau_s / 2.0).unwrap();
        assert!(shorter.peak() < 0.62 * k.peak() && shorter.peak() > 0.4 * k.peak(), "{} against {}", shorter.peak(), k.peak());
        let h = 1e-7;
        assert!((k.epsilon(t_peak + h) - k.epsilon(t_peak - h)).abs() / (2.0 * h) < 1e-3, "the peak is not stationary");

        // The delta synapse, and the double exponential converging on it.
        let delta = Kernel::new(tau_m, 0.0).unwrap();
        assert!(delta.is_delta() && (delta.epsilon(0.0) - 1.0).abs() < 1e-18 && delta.peak() == 1.0 && delta.peak_time() == 0.0);
        assert!((delta.epsilon(tau_m) - core::f64::consts::E.recip()).abs() < 1e-15);
        // The charge-normalised kernel is the one that converges; the raw one shrinks to nothing.
        let mut gap = f64::INFINITY;
        let mut raw = 0.0f64;
        for tau_s in [1e-3, 1e-4, 1e-5, 1e-6] {
            let near = Kernel::new(tau_m, tau_s).unwrap();
            let scale = tau_m / tau_s;
            let at = |i: i32| f64::from(i) * 2e-3;
            let now = (1..=40).map(|i| (scale * near.epsilon(at(i)) - delta.epsilon(at(i))).abs()).fold(0.0f64, f64::max);
            assert!(now < gap, "tau_s = {tau_s} did not close the gap: {now} against {gap}");
            gap = now;
            raw = (1..=40).map(|i| near.epsilon(at(i))).fold(0.0f64, f64::max);
            // 1 − (τ_s/τ_m)^a with a = τ_s/(τ_m−τ_s) is a·ln(τ_m/τ_s) to first order — the log
            // factor is why a bound of "a few τ_s/τ_m" would be wrong.
            let first_order = tau_s / (tau_m - tau_s) * (tau_m / tau_s).ln();
            assert!((near.charge_normalised_peak() - 1.0).abs() < 1.5 * first_order, "at tau_s = {tau_s}: {}", near.charge_normalised_peak());
        }
        assert!(gap < 1e-4, "at tau_s = 1 µs the scaled kernels still differ by {gap}");
        assert!(raw < 1e-4, "the RAW kernel should have shrunk to nothing, not converged: {raw}");
        assert_eq!(delta.charge_normalised_peak(), 1.0);

        assert_eq!(k.eta(1e-3, 2.0), -2.0 * (-1e-3f64 / tau_m).exp());
        assert_eq!(k.eta(-1e-9, 2.0), 0.0);
        assert!(Kernel::new(0.0, 5e-3).is_err() && Kernel::new(20e-3, -1e-3).is_err() && Kernel::new(5e-3, 5e-3).is_err());
        assert!(Kernel::new(f64::NAN, 1e-3).is_err() && Kernel::new(1e-3, f64::INFINITY).is_err());
    }

    /// How an own spike changes the potential in the reference integration.
    #[derive(Clone, Copy, PartialEq)]
    enum Reset {
        /// Subtract `reset` — the linear reset the full kernel sum is exact for.
        Subtract,
        /// Set the potential to zero, leaving the synaptic current alone. This is what a real
        /// LIF does, and what SRM₀ assumes erases the past.
        ToZero,
    }

    /// A direct integration of `τ_m u̇ = −u + I`, `τ_s İ = −I` with `I += w` at each input and
    /// the given reset at each own spike, advanced EXACTLY from one event to the next (the flow
    /// is a closed form between events, so this has no discretisation error at all, and the only
    /// thing it shares with the kernel sum is the pair of differential equations).
    fn integrate(srm: &Srm, inputs: &[(f64, f64)], own: &[f64], t_end: f64, reset: Reset) -> f64 {
        let mut events: Vec<(f64, Option<f64>)> = inputs.iter().map(|(t, w)| (*t, Some(*w))).collect();
        events.extend(own.iter().map(|t| (*t, None)));
        events.sort_by(|a, b| a.0.total_cmp(&b.0).then(b.1.is_some().cmp(&a.1.is_some())));
        let (mut u, mut i, mut t) = (0.0f64, 0.0f64, 0.0f64);
        let delta = srm.kernel.is_delta();
        let advance = |u: &mut f64, i: &mut f64, gap: f64| {
            let em = (-gap / srm.kernel.tau_m).exp();
            if delta {
                *u *= em;
                return;
            }
            let es = (-gap / srm.kernel.tau_s).exp();
            let c = srm.kernel.tau_s / (srm.kernel.tau_s - srm.kernel.tau_m);
            *u = *u * em + *i * c * (es - em);
            *i *= es;
        };
        for (when, what) in events {
            if when > t_end {
                break;
            }
            advance(&mut u, &mut i, when - t);
            t = when;
            match what {
                // A delta synapse displaces the membrane; a current synapse charges the current.
                Some(w) => {
                    if delta {
                        u += w;
                    } else {
                        i += w;
                    }
                }
                None => match reset {
                    Reset::Subtract => u -= srm.reset,
                    Reset::ToZero => u = 0.0,
                },
            }
        }
        advance(&mut u, &mut i, t_end - t);
        u
    }

    #[test]
    fn the_full_kernel_sum_is_the_linear_neuron_with_reset_by_subtraction() {
        let k = Kernel::new(20e-3, 5e-3).unwrap();
        let srm = Srm::subtracting(k, 1.0).unwrap();
        let inputs = [(2e-3, 0.7), (5e-3, -0.3), (5e-3, 1.1), (11e-3, 0.5), (23e-3, 0.9)];
        let own = [9e-3, 17e-3];
        let mut apart = 0.0f64;
        for end in [7e-3, 13e-3, 20e-3, 40e-3] {
            let sum = srm.potential(end, &inputs, &own).unwrap();
            let fine = integrate(&srm, &inputs, &own, end, Reset::Subtract);
            assert!((sum - fine).abs() < 1e-14, "at {end}: kernel sum {sum} against integration {fine}");
            apart = apart.max(sum.abs());
        }
        assert!(apart > 0.1, "the potential never left zero, so this compared nothing");
        // Linear in the weights, which is the property that makes the sum legal at all.
        let doubled: Vec<(f64, f64)> = inputs.iter().map(|(t, w)| (*t, 2.0 * w)).collect();
        let (a, b) = (srm.potential(13e-3, &inputs, &[]).unwrap(), srm.potential(13e-3, &doubled, &[]).unwrap());
        assert!((b - 2.0 * a).abs() < 1e-15 && a.abs() > 0.1);
        // And each own spike subtracts exactly `reset`, decayed.
        let with = srm.potential(20e-3, &inputs, &own).unwrap();
        let without = srm.potential(20e-3, &inputs, &[]).unwrap();
        let expected: f64 = own.iter().map(|f| -(-(20e-3 - f) / k.tau_m).exp()).sum();
        assert!((with - without - expected).abs() < 1e-15);
    }

    #[test]
    fn srm0_is_exact_for_a_delta_synapse_and_short_by_a_named_term_otherwise() {
        // The reference throughout: the ODE, reset TO ZERO at the own spike, integrated exactly.
        // SRM₀ claims to reproduce it from the last spike onward.
        // Two of the five inputs arrive AFTER the own spike, so there is something for SRM₀ to
        // be right about, and three before it, so there is something for it to drop.
        let inputs = [(2e-3, 0.7), (5e-3, 1.1), (11e-3, 0.5), (15e-3, 0.8), (23e-3, 0.9)];
        let t_hat = 13e-3;

        // Delta synapses: a reset to zero erases everything, and SRM₀ is the whole story. The
        // reference here is the delta-synapse membrane summed directly, since the two-equation
        // integrator above has no τ_s to run with.
        let delta = Srm { kernel: Kernel::new(20e-3, 0.0).unwrap(), theta: 1.0, reset: 1.0 };
        // Exactly zero, not merely small: a delta synapse holds no current between spikes, and
        // the raw sum of the three earlier weights — what a model that forgot to special-case it
        // would return — is 2.3.
        assert_eq!(delta.carried_current(t_hat, &inputs).unwrap(), 0.0);
        assert!((inputs.iter().filter(|(f, _)| *f < t_hat).map(|(_, w)| w).sum::<f64>() - 2.3).abs() < 1e-15);
        for t in [18e-3, 30e-3] {
            let exact = integrate(&delta, &inputs, &[t_hat], t, Reset::ToZero);
            let srm0 = delta.potential_srm0(t, &inputs, Some(t_hat)).unwrap() - delta.kernel.eta(t - t_hat, delta.reset);
            assert!((srm0 - exact).abs() < 1e-15, "delta synapse at {t}: {srm0} against {exact}");
            assert!(exact.abs() > 1e-2, "the comparison at {t} is between two near-zeros");
        }
        // Before the next input arrives there is nothing at all left: that is what "erased" means.
        assert_eq!(integrate(&delta, &inputs, &[t_hat], 14e-3, Reset::ToZero), 0.0);

        // With a synaptic time constant the current standing at t̂ keeps flowing, and SRM₀ is
        // short by exactly that current's own PSP.
        let srm = Srm { kernel: Kernel::new(20e-3, 5e-3).unwrap(), theta: 1.0, reset: 1.0 };
        let carried = srm.carried_current(t_hat, &inputs).unwrap();
        let by_hand: f64 = [(2e-3, 0.7), (5e-3, 1.1), (11e-3, 0.5)].iter().map(|(f, w)| w * (-(t_hat - f) / 5e-3).exp()).sum();
        assert_eq!(inputs.iter().filter(|(f, _)| *f < t_hat).count(), 3);
        assert!((carried - by_hand).abs() < 1e-18 && carried > 0.01, "carried {carried}");
        let mut worst = 0.0f64;
        for t in [14e-3, 18e-3, 30e-3] {
            let truth = integrate(&srm, &inputs, &[t_hat], t, Reset::ToZero);
            let srm0 = srm.potential_srm0(t, &inputs, Some(t_hat)).unwrap() - srm.kernel.eta(t - t_hat, srm.reset);
            let missing = truth - srm0;
            assert!(missing.abs() > 1e-3, "at {t} SRM₀ is short by only {missing}, too little to be evidence");
            // And what is missing is the named term, to rounding — not merely of its order.
            assert!((missing - carried * srm.kernel.epsilon(t - t_hat)).abs() < 1e-14, "at {t}: short by {missing}, named term {}", carried * srm.kernel.epsilon(t - t_hat));
            worst = worst.max(missing.abs() / truth.abs().max(1e-12));
        }
        assert!(worst > 0.05, "SRM₀'s largest relative error was only {worst}");

        // With no own spike at all, SRM₀ IS the full sum — no after-potential is applied from a
        // spike that never happened. `None` is not "a spike at minus infinity" only because the
        // after-potential is skipped; the kernel there would be exp(-inf) = 0, so the difference
        // only shows where the reset is NOT exponentially small, which is what this checks.
        let t = 30e-3;
        let none = srm.potential_srm0(t, &inputs, None).unwrap();
        assert_eq!(none, srm.potential(t, &inputs, &[]).unwrap());
        assert!(none.abs() > 1e-2, "the comparison is between two near-zeros");
        let flat = Srm { kernel: srm.kernel, theta: 1.0, reset: 2.0 };
        let mut wide = flat;
        wide.kernel.tau_m = 1e12; // an after-potential that does not decay over this window
        assert_eq!(wide.potential_srm0(t, &inputs, None).unwrap(), wide.potential(t, &inputs, &[]).unwrap());
        assert!((wide.potential_srm0(t, &inputs, Some(t_hat)).unwrap() - wide.potential_srm0(t, &inputs, None).unwrap()).abs() > 1.0, "the two must differ by the whole reset");
        // And it drops the inputs before t̂, which is the other half of the simplification.
        let after_only: f64 = inputs.iter().filter(|(f, _)| *f >= t_hat).map(|(f, w)| w * srm.kernel.epsilon(t - f)).sum();
        assert!((srm.potential_srm0(t, &inputs, Some(t_hat)).unwrap() - (after_only + srm.kernel.eta(t - t_hat, 1.0))).abs() < 1e-18);
    }

    #[test]
    fn bad_spike_lists_are_refused() {
        let srm = Srm::subtracting(Kernel::new(20e-3, 5e-3).unwrap(), 1.0).unwrap();
        assert_eq!(srm.potential(0.0, &[(2e-3, 1.0), (1e-3, 1.0)], &[]), Err(SrmError::Unsorted { index: 1 }));
        assert_eq!(srm.potential(0.0, &[], &[2e-3, 1e-3]), Err(SrmError::Unsorted { index: 1 }));
        assert_eq!(srm.potential(0.0, &[(f64::NAN, 1.0)], &[]), Err(SrmError::NonFinite { what: "input" }));
        assert_eq!(srm.potential(0.0, &[(1e-3, f64::NAN)], &[]), Err(SrmError::NonFinite { what: "t" }));
        assert_eq!(srm.potential(f64::NAN, &[], &[]), Err(SrmError::NonFinite { what: "t" }));
        assert!(srm.potential_srm0(0.0, &[(2e-3, 1.0), (1e-3, 1.0)], None).is_err());
        assert_eq!(srm.potential_srm0(0.0, &[], Some(f64::NAN)), Err(SrmError::NonFinite { what: "last" }));
        assert!(srm.carried_current(f64::NAN, &[]).is_err() && srm.carried_current(0.0, &[(2e-3, 1.0), (1e-3, 1.0)]).is_err());
        let many = vec![0.0; MAX_SPIKES + 1];
        assert_eq!(srm.potential(0.0, &[], &many), Err(SrmError::TooManySpikes { got: MAX_SPIKES + 1 }));
        assert!(Srm::subtracting(Kernel::new(20e-3, 5e-3).unwrap(), 0.0).is_err());
        assert!(SrmError::Unsorted { index: 3 }.to_string().contains("spike 3"));
        assert!(SrmError::TooManySpikes { got: 9 }.to_string().contains('9'));
    }

    #[test]
    fn escape_noise_is_an_exponential_hazard_and_a_geometric_survivor() {
        let e = Escape::new(10.0, 2e-3).unwrap();
        assert_eq!(e.hazard(1.0, 1.0), 10.0, "at threshold the hazard is rho0");
        assert!((e.hazard(1.0 + 2e-3, 1.0) - 10.0 * core::f64::consts::E).abs() < 1e-12);
        // A decade of rate per delta_u * ln 10 of potential.
        let decade = 2e-3 * 10f64.ln();
        assert!((e.hazard(1.0 + decade, 1.0) - 100.0).abs() < 1e-9);
        // A potential far enough above threshold overflows the exponential; the hazard saturates
        // at infinity rather than becoming a NaN, which would compare false against everything
        // and make the survivor below silently meaningless.
        assert_eq!(e.hazard(f64::INFINITY, 1.0), f64::INFINITY);
        let huge = e.hazard(1.0 + 2e-3 * 1000.0, 1.0);
        assert!(huge.is_infinite() && !huge.is_nan(), "a hazard of {huge}");
        assert_eq!(Escape::survivor_at(huge, 1e-3), 0.0);
        assert_eq!(e.hazard(-f64::INFINITY, 1.0), 0.0);
        assert!(e.hazard(f64::NAN, 1.0).is_nan(), "a non-finite potential must propagate, not become a rate");
        assert_eq!(e.hazard(-1e300, 1.0), 0.0);
        assert!((e.potential_for(100.0, 1.0).unwrap() - (1.0 + decade)).abs() < 1e-15);
        // The round trip subtracts two numbers of order θ to get a difference of order Δu, so it
        // loses θ/Δu of its precision — 500 here, which is what this tolerance is.
        assert!((e.hazard(e.potential_for(37.0, 1.0).unwrap(), 1.0) - 37.0).abs() < 37.0 * 1e-13 / 2e-3);
        assert!(e.potential_for(0.0, 1.0).is_none() && e.potential_for(f64::INFINITY, 1.0).is_none());

        // At a constant hazard the interval is exponential: the survivor, the density's integral,
        // and the mean.
        let rate = 40.0;
        assert_eq!(Escape::survivor_at(rate, 0.0), 1.0);
        assert_eq!(Escape::survivor_at(rate, -1.0), 1.0);
        assert!((Escape::survivor_at(rate, 1.0 / rate) - core::f64::consts::E.recip()).abs() < 1e-15);
        let (mut mass, mut mean) = (0.0, 0.0);
        let (h, panels) = (1.0 / 200_000.0, 200_000);
        for k in 0..=panels {
            let t = h * f64::from(k);
            let w = if k == 0 || k == panels { 1.0 } else if k % 2 == 1 { 4.0 } else { 2.0 };
            mass += w * Escape::interval_density_at(rate, t);
            mean += w * t * Escape::interval_density_at(rate, t);
        }
        assert!((mass * h / 3.0 - 1.0).abs() < 1e-9, "the density integrates to {}", mass * h / 3.0);
        assert!((mean * h / 3.0 - 1.0 / rate).abs() < 1e-9, "the mean interval is {}", mean * h / 3.0);
        assert_eq!(Escape::interval_density_at(rate, -1.0), 0.0);

        // The varying-potential survivor reduces to the constant one.
        let flat = e.survivor(|_| 1.0, 1.0, 0.25, 2_000).unwrap();
        assert!((flat - Escape::survivor_at(10.0, 0.25)).abs() < 1e-12, "{flat}");
        // And a potential that rises makes survival less likely than the best moment alone.
        let rising = e.survivor(|s| 1.0 + 0.004 * s / 0.25, 1.0, 0.25, 2_000).unwrap();
        assert!(rising < flat && rising > Escape::survivor_at(e.hazard(1.004, 1.0), 0.25));

        // Drawn intervals: the mean and the empirical survivor.
        let mut rng = Rng::new(7);
        let n = 100_000;
        let draws: Vec<f64> = (0..n).map(|_| Escape::draw_interval(rate, rng.next_f64()).unwrap()).collect();
        let mean = draws.iter().sum::<f64>() / n as f64;
        assert!((mean * rate - 1.0).abs() < 0.02, "mean interval {mean} for a rate of {rate}");
        for t in [0.005, 0.02, 0.05] {
            let survived = draws.iter().filter(|d| **d > t).count() as f64 / n as f64;
            assert!((survived - Escape::survivor_at(rate, t)).abs() < 0.01, "survivor at {t}: {survived} against {}", Escape::survivor_at(rate, t));
        }
        assert!(Escape::draw_interval(rate, 1.0).is_none() && Escape::draw_interval(rate, -0.1).is_none() && Escape::draw_interval(0.0, 0.5).is_none());
        assert_eq!(Escape::draw_interval(rate, 0.0), Some(0.0));
        assert!(e.survivor(|_| 1.0, 1.0, 0.0, 100).is_err() && e.survivor(|_| 1.0, 1.0, 0.25, 1).is_err());
        // Simpson's rule needs an even number of panels; an odd count is rounded up, so asking
        // for 101 gives the 102-panel answer and not a weighting that is wrong at both ends.
        let varying = |s: f64| 1.0 + 0.004 * s / 0.25;
        let odd = e.survivor(varying, 1.0, 0.25, 101).unwrap();
        let up = e.survivor(varying, 1.0, 0.25, 102).unwrap();
        assert_eq!(odd, up);
        assert!((odd - e.survivor(varying, 1.0, 0.25, 100).unwrap()).abs() > 0.0, "100 and 102 panels must not agree exactly, or this proves nothing");
        assert!(Escape::new(0.0, 1e-3).is_err() && Escape::new(1.0, 0.0).is_err() && Escape::new(f64::NAN, 1e-3).is_err());
    }

    #[test]
    fn a_vanishing_delta_u_brings_the_deterministic_threshold_back() {
        // Below threshold survival goes to certainty, above it to impossibility, and the approach
        // is geometric in 1/delta_u: each halving of delta_u squares the hazard ratio.
        let (theta, below, above, window) = (1.0, 0.99, 1.01, 0.1);
        let (mut last_below, mut last_above) = (0.0, 1.0);
        for k in 0..6 {
            let e = Escape::new(10.0, 4e-3 / f64::from(1 << k)).unwrap();
            let s_below = Escape::survivor_at(e.hazard(below, theta), window);
            let s_above = Escape::survivor_at(e.hazard(above, theta), window);
            // Non-strict: both saturate — at one and at zero — which is the limit being claimed.
            assert!(s_below >= last_below && s_above <= last_above, "delta_u = {}: {s_below}, {s_above}", e.delta_u);
            assert!(k > 0 || (s_below < 0.99 && s_above > 1e-12), "the first step is already at the limit, so nothing is being watched");
            last_below = s_below;
            last_above = s_above;
        }
        assert!(last_below > 0.999 && last_above < 1e-12, "the limit was not reached: {last_below}, {last_above}");
        // At threshold itself the hazard is rho0 whatever delta_u is — the one potential the
        // limit says nothing about.
        for k in 0..6 {
            let e = Escape::new(10.0, 4e-3 / f64::from(1 << k)).unwrap();
            assert_eq!(e.hazard(theta, theta), 10.0);
        }
    }

    /// A neuron that has never fired applies no after-potential — not even one that happens to
    /// vanish.
    ///
    /// The hole: `srm0_is_exact_for_delta_synapses_and_not_otherwise` already asserts
    /// `potential_srm0(t, inputs, None) == potential(t, inputs, &[])`, and widens `tau_m` to
    /// `1e12` so that an applied after-potential could not hide inside an exponential. But what
    /// makes the conditional invisible there is that `eta(+inf, reset)` is
    /// `-reset * exp(-inf/tau_m)`, i.e. `-reset * 0.0`, which is `-0.0`, and `x + (-0.0) == x`.
    /// That holds for every FINITE reset and for no other: `inf * 0.0` is a `NaN`. `Srm::reset`
    /// is a `pub` field on a struct this module's own tests build by literal and by assignment,
    /// and `potential_srm0` validates `t`, the weights and `last` — nothing else. So the identity
    /// being pinned is over ALL resets: with `last: None` the reset is not an argument to any
    /// arithmetic, so a reset that would poison the sum cannot reach it.
    #[test]
    fn a_neuron_that_never_fired_applies_no_after_potential_whatever_its_reset() {
        let inputs = [(1e-3, 2.0), (4e-3, -0.5)];
        let t = 5e-3;
        let kernel = Kernel::new(20e-3, 0.0).unwrap();
        // The same operations in the same order as the drive the module sums, so this is an
        // equality and not a tolerance. Measured: 2 e^{-0.2} - 0.5 e^{-0.05} = 1.1618467939056067,
        // which is of order one — so a NaN or an infinity landing on it could not hide.
        let drive: f64 = inputs.iter().map(|(f, w)| w * (-(t - f) / kernel.tau_m).exp()).sum();
        assert!((drive - 1.161_846_79).abs() < 1e-8, "the fixture's drive is {drive}");
        for reset in [1.0, 0.0, -3.0, 1e300, f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
            let srm = Srm { kernel, theta: 1.0, reset };
            let none = srm.potential_srm0(t, &inputs, None).unwrap();
            assert_eq!(none, drive, "a reset of {reset} reached the sum");
            assert_eq!(none, srm.potential(t, &inputs, &[]).unwrap(), "SRM₀ with no own spike is not the full sum over no own spikes, at a reset of {reset}");
        }
        // Why the finite half of that loop cannot see it, stated so the non-finite half reads as
        // the measurement it is: at a finite reset the after-potential a spike at minus infinity
        // WOULD contribute is exactly -0.0, and the drive is a `sum()` folded from +0.0, so it
        // can never itself be -0.0 for the addition to expose.
        assert_eq!(kernel.eta(f64::INFINITY, 1e300), -0.0);
        assert!(kernel.eta(f64::INFINITY, 1e300).is_sign_negative());
        assert!(kernel.eta(f64::INFINITY, f64::INFINITY).is_nan(), "an infinite reset no longer poisons the after-potential, and this test has stopped measuring anything");
    }

    /// A delta synapse carries no current, and the early return that says so is load-bearing at
    /// two edges the rest of the suite cannot reach.
    ///
    /// The hole: `srm0_is_exact_for_delta_synapses_and_not_otherwise` asserts
    /// `carried_current(t_hat, &inputs) == 0.0` for `tau_s = 0.0` with weights of order one, and
    /// the argument that the early return is redundant there is sound — every term is
    /// `w * exp(-(t_hat - f)/0)` with a strictly positive numerator, `w * exp(-inf)`, `w * 0`.
    /// It stops being sound twice. (a) `Kernel::new` ACCEPTS a `tau_s` of `-0.0` — its guard is
    /// `!(tau_s >= 0.0) || !tau_s.is_finite()`, and `-0.0 >= 0.0` is true and `-0.0` is finite —
    /// while `is_delta` is `tau_s == 0.0`, which `-0.0` also satisfies. Dividing the strictly
    /// NEGATIVE numerator by a negative zero gives `+inf`, and `exp(+inf)` is `+inf`. (b)
    /// `carried_current` is the one entry point in this module that does not check the weights;
    /// `potential` and `potential_srm0` both refuse a non-finite one. `inf * exp(-inf)` is
    /// `inf * 0.0`, a `NaN`.
    #[test]
    fn a_delta_synapse_carries_no_current_at_a_negative_zero_or_an_infinite_weight() {
        let inputs = [(2e-3, 0.7), (5e-3, 1.1), (11e-3, 0.5), (15e-3, 0.8)];
        let t_hat = 13e-3;

        let signed = Kernel::new(20e-3, -0.0).expect("the constructor accepts a tau_s of -0.0");
        assert!(signed.tau_s.is_sign_negative() && signed.is_delta(), "the fixture is no longer a negative-zero delta kernel");
        let srm = Srm::subtracting(signed, 1.0).unwrap();
        assert_eq!(srm.carried_current(t_hat, &inputs).unwrap(), 0.0);
        // The term the sum would otherwise take, so the line above reads as a measurement of the
        // early return rather than of the fixture's weights.
        assert_eq!((-(t_hat - 2e-3) / signed.tau_s).exp(), f64::INFINITY, "a negative zero no longer flips the sign of the exponent");

        let plain = Kernel::new(20e-3, 0.0).unwrap();
        let srm = Srm::subtracting(plain, 1.0).unwrap();
        assert_eq!(srm.carried_current(t_hat, &[(2e-3, f64::INFINITY)]).unwrap(), 0.0);
        let term = f64::INFINITY * (-(t_hat - 2e-3) / plain.tau_s).exp();
        assert!(term.is_nan(), "an infinite weight no longer makes the dropped term a NaN, and this half of the test has stopped measuring anything");
    }
}
