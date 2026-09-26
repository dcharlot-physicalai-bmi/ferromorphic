//! The parametric LIF neuron: a membrane time constant that is learned by gradient descent along
//! with the weights, as `SpikingJelly` defines it — and its reverse-mode gradient shown exact.
//!
//! # The model
//!
//! Fang, Yu, Chen, Masquelier, Huang and Tian, *Incorporating Learnable Membrane Time Constant to
//! Enhance Learning of Spiking Neural Networks*, ICCV 2021 (arXiv:2007.05785), make the leak of a
//! discrete-time LIF a parameter: `1/τ = sigmoid(w)`, so that `τ` stays above one whatever `w`
//! gradient descent reaches. `SpikingJelly`'s `ParametricLIFNode` (`functional.plif_step`, read at
//! commit `4c98a4a3f3` of 2026-09-24) is the reference implementation, and this module follows it
//! line for line, with `k = sigmoid(w)`:
//!
//! ```text
//! decay_input:      H[t] = V[t−1] + k (X[t] − (V[t−1] − V_reset))
//! not decay_input:  H[t] = V[t−1] − k (V[t−1] − V_reset) + X[t]
//! S[t] = Θ(H[t] − V_th)                        Θ(0) = 1: reaching threshold fires
//! hard reset:       V[t] = H[t](1 − S[t]) + V_reset S[t]
//! soft reset:       V[t] = H[t] − V_th S[t]        (V_reset taken as 0 in the charge)
//! w = −ln(τ₀ − 1)   for an initial τ₀ > 1, so that sigmoid(w) = 1/τ₀ exactly
//! ```
//!
//! # The gradient, and why it can be trusted
//!
//! [`Plif::backward`] is backpropagation through time: `∂S/∂H` is the surrogate's derivative, the
//! reset contributes `(V_reset − H)∂S/∂H` (hard) or `−V_th ∂S/∂H` (soft) unless `detach_reset`
//! drops it as `SpikingJelly` does, and `w` collects `∂H/∂k · k(1 − k)` at every step. A surrogate
//! gradient is the gradient of no network that was run, so it cannot be checked against finite
//! differences as it stands. Run forward with [`SpikeFn::Smooth`] instead — the surrogate's own
//! antiderivative in place of the step — and the same code is the EXACT gradient of that network:
//! the tests compare it with central differences for every combination of reset and input decay.
//! Then a leak is learned: from `τ = 2`, gradient descent on `w` alone recovers `τ = 5` from the
//! membrane trace it produced.

use core::fmt;

use crate::surrogate::{SigmoidDeriv, SpikeFn, Surrogate};

/// Why a PLIF neuron could not be built or run.
#[derive(Debug, Clone, PartialEq)]
pub enum PlifError {
    /// An initial time constant at or below one, which no `w` can reach: `sigmoid(w) < 1` always.
    Tau {
        /// The time constant asked for.
        tau: f64,
    },
    /// A value that must be finite was not.
    NonFinite {
        /// Which.
        what: &'static str,
        /// Its value.
        value: f64,
    },
    /// Inputs that are not a whole number of steps of the population.
    Steps {
        /// Entries in `x`.
        got: usize,
        /// Neurons per step.
        n: usize,
    },
    /// Arrays whose lengths disagree.
    Length {
        /// Which array.
        what: &'static str,
        /// Its length.
        got: usize,
        /// The length it must have.
        want: usize,
    },
}

impl fmt::Display for PlifError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Tau { tau } => write!(f, "a time constant of {tau} steps is not above one, which 1/sigmoid(w) always is"),
            Self::NonFinite { what, value } => write!(f, "{what} = {value} is not finite"),
            Self::Steps { got, n } => write!(f, "{got} inputs are not a whole number of steps of {n} neurons"),
            Self::Length { what, got, want } => write!(f, "{what} has {got} entries where {want} are needed"),
        }
    }
}

impl std::error::Error for PlifError {}

fn finite(what: &'static str, value: f64) -> Result<f64, PlifError> {
    if value.is_finite() { Ok(value) } else { Err(PlifError::NonFinite { what, value }) }
}

/// How the membrane is reset after a spike.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Reset {
    /// `V ← V_reset` — `SpikingJelly`'s default, with `V_reset = 0`.
    Hard {
        /// The potential reset to.
        v_reset: f64,
    },
    /// `V ← V − V_th`, keeping the overshoot — `SpikingJelly`'s `v_reset = None`.
    Soft,
}

/// A population of parametric LIF neurons sharing one learnable leak.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Plif {
    /// The learnable parameter; the leak is `sigmoid(w)` per step.
    pub w: f64,
    /// The threshold.
    pub v_threshold: f64,
    /// The reset.
    pub reset: Reset,
    /// Whether the input is leaked as well as the membrane (`SpikingJelly`'s default, true).
    pub decay_input: bool,
    /// Whether the gradient ignores the reset's dependence on the spike.
    pub detach_reset: bool,
}

/// One forward run: every step's charged potential, spike and post-reset potential, `[t][neuron]`
/// flattened step-major.
#[derive(Debug, Clone, PartialEq)]
pub struct PlifTrace {
    /// Neurons in the population.
    pub n: usize,
    /// The potential the neurons start from.
    pub v0: Vec<f64>,
    /// `H[t]`, before the spike.
    pub h: Vec<f64>,
    /// `S[t]`: 0 or 1 under [`SpikeFn::Heaviside`], the surrogate's antiderivative under
    /// [`SpikeFn::Smooth`].
    pub s: Vec<f64>,
    /// `V[t]`, after the reset.
    pub v: Vec<f64>,
}

/// The gradient of a loss with respect to the leak parameter and every input.
#[derive(Debug, Clone, PartialEq)]
pub struct PlifGrad {
    /// `∂L/∂w`.
    pub w: f64,
    /// `∂L/∂X[t][i]`, flattened step-major.
    pub x: Vec<f64>,
    /// `∂L/∂V₀[i]`.
    pub v0: Vec<f64>,
}

impl Plif {
    /// `SpikingJelly`'s defaults with an initial time constant `tau`: `w = −ln(τ − 1)`, threshold 1,
    /// hard reset to 0, input decayed, reset not detached.
    ///
    /// # Errors
    ///
    /// [`PlifError::Tau`] for a `tau` that is not finite and above one.
    pub fn with_tau(tau: f64) -> Result<Self, PlifError> {
        if !(tau.is_finite() && tau > 1.0) {
            return Err(PlifError::Tau { tau });
        }
        Ok(Self { w: -(tau - 1.0).ln(), v_threshold: 1.0, reset: Reset::Hard { v_reset: 0.0 }, decay_input: true, detach_reset: false })
    }

    /// The leak per step, `k = sigmoid(w) = 1/τ`.
    #[must_use]
    pub fn leak(&self) -> f64 {
        SigmoidDeriv::logistic(self.w)
    }

    /// The membrane time constant in steps, `1/sigmoid(w)`.
    #[must_use]
    pub fn tau(&self) -> f64 {
        1.0 / self.leak()
    }

    fn v_reset(&self) -> f64 {
        match self.reset {
            Reset::Hard { v_reset } => v_reset,
            Reset::Soft => 0.0,
        }
    }

    fn check(&self) -> Result<(), PlifError> {
        finite("w", self.w)?;
        finite("v_threshold", self.v_threshold)?;
        finite("v_reset", self.v_reset())?;
        Ok(())
    }

    /// Run `x.len() / n` steps of the `n = v0.len()` neurons from `v0`, spikes formed by `spike_fn`
    /// with `sur`.
    ///
    /// # Errors
    ///
    /// [`PlifError::Length`] for an empty population; [`PlifError::Steps`] when `x` is not a whole
    /// number of steps;
    /// [`PlifError::NonFinite`] for a parameter, input or start that is not finite.
    pub fn forward(&self, x: &[f64], v0: &[f64], spike_fn: SpikeFn, sur: &dyn Surrogate) -> Result<PlifTrace, PlifError> {
        self.check()?;
        let n = v0.len();
        if n == 0 {
            return Err(PlifError::Length { what: "v0", got: 0, want: 1 });
        }
        if !x.len().is_multiple_of(n) {
            return Err(PlifError::Steps { got: x.len(), n });
        }
        for &value in x {
            finite("x", value)?;
        }
        for &value in v0 {
            finite("v0", value)?;
        }
        let (k, vr, th) = (self.leak(), self.v_reset(), self.v_threshold);
        let mut v = v0.to_vec();
        let mut trace = PlifTrace { n, v0: v0.to_vec(), h: Vec::with_capacity(x.len()), s: Vec::with_capacity(x.len()), v: Vec::with_capacity(x.len()) };
        for step in x.chunks_exact(n) {
            for (i, &xi) in step.iter().enumerate() {
                let h = if self.decay_input { v[i] + k * (xi - (v[i] - vr)) } else { v[i] - k * (v[i] - vr) + xi };
                let s = match spike_fn {
                    SpikeFn::Heaviside => sur.forward(h - th),
                    SpikeFn::Smooth => sur.antiderivative(h - th),
                };
                v[i] = match self.reset {
                    Reset::Hard { v_reset } => h * (1.0 - s) + v_reset * s,
                    Reset::Soft => h - th * s,
                };
                trace.h.push(h);
                trace.s.push(s);
                trace.v.push(v[i]);
            }
        }
        Ok(trace)
    }

    /// Backpropagation through time of a loss whose gradient with respect to every spike is
    /// `dl_ds` and with respect to every charged potential is `dl_dh`, both `[t][neuron]` like the
    /// trace.
    ///
    /// # Errors
    ///
    /// [`PlifError::Length`] when either gradient is not the trace's length; [`PlifError::NonFinite`]
    /// for a gradient entry that is not finite.
    pub fn backward(&self, x: &[f64], trace: &PlifTrace, dl_ds: &[f64], dl_dh: &[f64], sur: &dyn Surrogate) -> Result<PlifGrad, PlifError> {
        let len = trace.h.len();
        for (what, got) in [("x", x.len()), ("dl_ds", dl_ds.len()), ("dl_dh", dl_dh.len())] {
            if got != len {
                return Err(PlifError::Length { what, got, want: len });
            }
        }
        for &g in dl_ds.iter().chain(dl_dh) {
            finite("a loss gradient", g)?;
        }
        let (k, vr, th, n) = (self.leak(), self.v_reset(), self.v_threshold, trace.n);
        let dk_dw = k * (1.0 - k);
        let mut grad = PlifGrad { w: 0.0, x: vec![0.0; len], v0: vec![0.0; n] };
        // `dv[i]`: the gradient reaching `V[t][i]` from everything after step `t`.
        let mut dv = vec![0.0; n];
        for t in (0..len / n).rev() {
            for i in 0..n {
                let at = t * n + i;
                let h = trace.h[at];
                let g = sur.backward(h - th);
                let s = trace.s[at];
                let reset_path = if self.detach_reset { 0.0 } else { 1.0 };
                let dv_dh = match self.reset {
                    Reset::Hard { v_reset } => (1.0 - s) + reset_path * (v_reset - h) * g,
                    Reset::Soft => 1.0 - reset_path * th * g,
                };
                let dh = dl_dh[at] + dl_ds[at] * g + dv[i] * dv_dh;
                let v_prev = if t == 0 { trace.v0[i] } else { trace.v[at - n] };
                let (dh_dk, dh_dx) = if self.decay_input { (x[at] - (v_prev - vr), k) } else { (-(v_prev - vr), 1.0) };
                grad.w += dh * dh_dk * dk_dw;
                grad.x[at] = dh * dh_dx;
                dv[i] = dh * (1.0 - k);
            }
        }
        grad.v0 = dv;
        Ok(grad)
    }
}


#[cfg(test)]
mod tests {
    use super::{Plif, PlifError, Reset};
    use crate::surrogate::{SigmoidDeriv, SpikeFn, Surrogate};

    /// `w = −ln(τ − 1)`, `SpikingJelly`'s `init_w`, gives `sigmoid(w) = 1/τ` in exact arithmetic:
    /// at the two initial values the paper trains from, `τ₀ = 2` and `τ₀ = 16`, and at three other
    /// values above one. One or less is refused, as `SpikingJelly` asserts.
    ///
    /// Corrected. This comment used to say the loop covered "the time constants the paper starts
    /// from" while it ran over 2, 1.5, 4 and 10, and 16 was not tested. Fang et al. (ICCV 2021,
    /// read here as arXiv:2007.05785v5) start PLIF from `τ₀ = 2` and `τ₀ = 16` only: the PLIF rows
    /// of Table 4 are "PLIF(τ₀=2)" and "PLIF(τ₀=16)", each beside LIF at the same `τ`; the legend
    /// of Fig. 7 has the same two, "PLIF, τ₀ = 2" and "PLIF, τ₀ = 16"; and the supplement says "We
    /// set τ₀ = 2 for all PLIF neurons." The paper defines `k(a) = 1/(1 + exp(−a))` and
    /// `τ = 1/k(a)`, and this review did not locate the inverse `w = −ln(τ₀ − 1)` in it. That
    /// inverse is `SpikingJelly`'s `init_w = -math.log(init_tau - 1.0)` in
    /// `activation_based/neuron/plif.py`, with a default `init_tau` of 2.
    ///
    /// In `f64`, on the machine this was measured on, 2, 16, 1.5 and 4 come back to the bit, while
    /// 10 does not: `1/k` is `10.000000000000002`, 1.8e-15 above it. Hence a tolerance for the
    /// loop, and equality only at `τ = 2`, where `w = −ln 1 = 0` and `sigmoid(0) = 1/2` whatever
    /// the platform's `exp` and `ln` round to.
    #[test]
    fn the_initial_tau_is_exact() {
        // The paper's two initial values, then three other values above one.
        for tau in [2.0, 16.0, 1.5, 4.0, 10.0] {
            let p = Plif::with_tau(tau).unwrap();
            assert!((p.leak() - 1.0 / tau).abs() < 1e-15 && (p.tau() - tau).abs() < 1e-13, "{tau}");
        }
        let p = Plif::with_tau(2.0).unwrap();
        assert_eq!((p.leak(), p.tau()), (0.5, 2.0));
        assert_eq!((p.w, p.v_threshold, p.reset, p.decay_input, p.detach_reset), (0.0, 1.0, Reset::Hard { v_reset: 0.0 }, true, false));
        for tau in [1.0, 0.5, f64::NAN, f64::INFINITY] {
            assert!(matches!(Plif::with_tau(tau), Err(PlifError::Tau { .. })), "{tau}");
        }
        assert_eq!(Plif::with_tau(1.0).unwrap_err().to_string(), "a time constant of 1 steps is not above one, which 1/sigmoid(w) always is");
    }

    /// Below threshold the membrane is the geometric relaxation the recursion defines:
    /// `V[t] = V_r + (V₀ − V_r)(1 − k)^t + X(1 − (1 − k)^t)` with the input decayed, and
    /// `V_r + (V₀ − V_r)(1 − k)^t + (X/k)(1 − (1 − k)^t)` without.
    #[test]
    fn below_threshold_the_membrane_is_geometric() {
        let sur = SigmoidDeriv::default();
        for decay_input in [true, false] {
            let p = Plif { w: -1.0, v_threshold: 10.0, reset: Reset::Hard { v_reset: -0.2 }, decay_input, detach_reset: false };
            let k = p.leak();
            let (x, v0, vr) = (0.3, 0.8, -0.2);
            let trace = p.forward(&[x; 30], &[v0], SpikeFn::Heaviside, &sur).unwrap();
            assert!(trace.s.iter().all(|&s| s == 0.0));
            for (t, &v) in trace.v.iter().enumerate() {
                let q = (1.0 - k).powi(t as i32 + 1);
                let drive = if decay_input { x } else { x / k };
                let want = vr + (v0 - vr) * q + drive * (1.0 - q);
                assert!((v - want).abs() < 1e-14, "decay_input {decay_input}, step {t}: {v} against {want}");
            }
        }
    }

    /// Spikes and resets as `SpikingJelly` writes them: reaching the threshold fires, a hard reset
    /// lands on `V_reset`, a soft one keeps the overshoot.
    #[test]
    fn spikes_and_resets_follow_the_recursion() {
        let sur = SigmoidDeriv::default();
        let hard = Plif { w: 0.0, v_threshold: 1.0, reset: Reset::Hard { v_reset: -0.5 }, decay_input: true, detach_reset: false };
        // k = 1/2: from V₀ = 0 under X = 2, H = 0 + (2 − (0 + 0.5))/2 = 0.75, then 0.75 + (2 − 1.25)/2 = 1.125.
        let t = hard.forward(&[2.0, 2.0, 2.0], &[0.0], SpikeFn::Heaviside, &sur).unwrap();
        assert_eq!((t.h[0], t.s[0], t.v[0]), (0.75, 0.0, 0.75));
        assert_eq!((t.h[1], t.s[1], t.v[1]), (1.125, 1.0, -0.5));
        let soft = Plif { reset: Reset::Soft, ..hard };
        let t = soft.forward(&[2.0, 2.0], &[0.0], SpikeFn::Heaviside, &sur).unwrap();
        // Soft: the charge leaks towards 0, not towards V_reset: H = 1, which fires, and V = 0.
        assert_eq!((t.h[0], t.s[0], t.v[0]), (1.0, 1.0, 0.0), "reaching the threshold exactly fires");
        // A soft reset subtracts the threshold, not one: at V_th = 0.75, H = 1 keeps 0.25.
        let t = Plif { v_threshold: 0.75, ..soft }.forward(&[2.0], &[0.0], SpikeFn::Heaviside, &sur).unwrap();
        assert_eq!(t.v[0], 0.25);
        let not_decayed = Plif { decay_input: false, ..hard };
        let t = not_decayed.forward(&[0.5], &[0.0], SpikeFn::Heaviside, &sur).unwrap();
        assert_eq!(t.h[0], 0.0 - 0.5 * (0.0 + 0.5) + 0.5);
        // Two neurons, step-major.
        let t = hard.forward(&[2.0, 0.0, 2.0, 0.0], &[0.0, 0.0], SpikeFn::Heaviside, &sur).unwrap();
        assert_eq!((t.n, t.h.len(), t.h[1], t.h[2]), (2, 4, -0.25, 1.125));
    }

    /// With the smooth spike, `backward` is the exact gradient: every combination of reset, input
    /// decay and a loss on both spikes and potentials, against central differences in `w`, in every
    /// input and in the starting potential.
    #[test]
    fn the_smooth_gradient_is_exact() {
        let sur = SigmoidDeriv::default();
        let x = [0.9, 1.4, -0.3, 2.2, 0.7, 1.9, 0.1, 1.3];
        let v0 = [0.2, -0.1];
        let (cs, ch) = ([0.7, -1.1, 0.4, 0.9, -0.6, 1.3, 0.8, -0.2], [0.3, 0.5, -0.4, 0.2, 0.9, -0.7, 0.1, 0.6]);
        for reset in [Reset::Hard { v_reset: -0.3 }, Reset::Soft] {
            for decay_input in [true, false] {
                let p = Plif { w: 0.4, v_threshold: 1.3, reset, decay_input, detach_reset: false };
                let loss = |p: &Plif, x: &[f64], v0: &[f64]| {
                    let t = p.forward(x, v0, SpikeFn::Smooth, &sur).unwrap();
                    t.s.iter().zip(&cs).map(|(s, c)| s * c).sum::<f64>() + t.h.iter().zip(&ch).map(|(h, c)| h * c).sum::<f64>()
                };
                let trace = p.forward(&x, &v0, SpikeFn::Smooth, &sur).unwrap();
                let g = p.backward(&x, &trace, &cs, &ch, &sur).unwrap();
                let e = 1e-6;
                let fd_w = (loss(&Plif { w: p.w + e, ..p }, &x, &v0) - loss(&Plif { w: p.w - e, ..p }, &x, &v0)) / (2.0 * e);
                assert!((g.w - fd_w).abs() < 1e-7, "{reset:?} {decay_input}: dw {} against {fd_w}", g.w);
                for j in 0..x.len() {
                    let (mut a, mut b) = (x, x);
                    a[j] += e;
                    b[j] -= e;
                    let fd = (loss(&p, &a, &v0) - loss(&p, &b, &v0)) / (2.0 * e);
                    assert!((g.x[j] - fd).abs() < 1e-7, "{reset:?} {decay_input}: dx[{j}] {} against {fd}", g.x[j]);
                }
                for j in 0..2 {
                    let (mut a, mut b) = (v0, v0);
                    a[j] += e;
                    b[j] -= e;
                    let fd = (loss(&p, &x, &a) - loss(&p, &x, &b)) / (2.0 * e);
                    assert!((g.v0[j] - fd).abs() < 1e-7, "{reset:?} {decay_input}: dv0[{j}] {} against {fd}", g.v0[j]);
                }
            }
        }
    }

    /// `detach_reset` drops exactly the reset's dependence on the spike: through a hard reset the
    /// one-step gradient of `V` in `H` is `1 − S` instead of `1 − S + (V_r − H)∂S/∂H`, and through a
    /// soft one `1` instead of `1 − V_th ∂S/∂H`.
    #[test]
    fn detach_reset_drops_the_reset_path() {
        let sur = SigmoidDeriv::default();
        for reset in [Reset::Hard { v_reset: -0.3 }, Reset::Soft] {
            let p = Plif { w: 0.4, v_threshold: 1.0, reset, decay_input: true, detach_reset: false };
            let d = Plif { detach_reset: true, ..p };
            let x = [1.1, 0.0];
            let trace = p.forward(&x, &[0.3], SpikeFn::Smooth, &sur).unwrap();
            // A loss on the second step's charge only: it reaches step one's H through V[0].
            let (ds, dh) = ([0.0, 0.0], [0.0, 1.0]);
            let (gp, gd) = (p.backward(&x, &trace, &ds, &dh, &sur).unwrap(), d.backward(&x, &trace, &ds, &dh, &sur).unwrap());
            let (h, s, g) = (trace.h[0], trace.s[0], sur.backward(trace.h[0] - 1.0));
            let k = p.leak();
            let (full, detached) = match reset {
                Reset::Hard { v_reset } => ((1.0 - s) + (v_reset - h) * g, 1.0 - s),
                Reset::Soft => (1.0 - g, 1.0),
            };
            assert!((gp.x[0] - (1.0 - k) * full * k).abs() < 1e-15, "{reset:?}");
            assert!((gd.x[0] - (1.0 - k) * detached * k).abs() < 1e-15, "{reset:?}");
            assert!(gp.x[0] != gd.x[0]);
        }
    }

    /// A leak learned: from `τ = 2`, gradient descent on `w` alone, against the squared error of the
    /// membrane trace, recovers the `τ = 5` that produced it.
    #[test]
    fn gradient_descent_recovers_a_time_constant() {
        let sur = SigmoidDeriv::default();
        let x: Vec<f64> = (0..40).map(|t| 0.5 + 0.4 * (f64::from(t) * 0.7).sin()).collect();
        let target = Plif { v_threshold: 100.0, ..Plif::with_tau(5.0).unwrap() };
        let want = target.forward(&x, &[0.0], SpikeFn::Heaviside, &sur).unwrap().h;
        let mut p = Plif { v_threshold: 100.0, ..Plif::with_tau(2.0).unwrap() };
        for _ in 0..3000 {
            let t = p.forward(&x, &[0.0], SpikeFn::Heaviside, &sur).unwrap();
            let dh: Vec<f64> = t.h.iter().zip(&want).map(|(h, w)| 2.0 * (h - w)).collect();
            let g = p.backward(&x, &t, &vec![0.0; 40], &dh, &sur).unwrap();
            p.w -= 0.5 * g.w;
        }
        assert!((p.tau() - 5.0).abs() < 1e-6, "τ = {}", p.tau());
    }

    /// Every refusal.
    #[test]
    fn every_refusal_names_what_it_refused() {
        let sur = SigmoidDeriv::default();
        let p = Plif::with_tau(2.0).unwrap();
        let fwd = |p: Plif, x: &[f64], v0: &[f64]| p.forward(x, v0, SpikeFn::Heaviside, &sur).unwrap_err().to_string();
        assert_eq!(fwd(p, &[1.0, 2.0, 3.0], &[0.0, 0.0]), "3 inputs are not a whole number of steps of 2 neurons");
        assert_eq!(fwd(p, &[1.0], &[]), "v0 has 0 entries where 1 are needed");
        assert_eq!(fwd(p, &[f64::NAN], &[0.0]), "x = NaN is not finite");
        assert_eq!(fwd(p, &[1.0], &[f64::INFINITY]), "v0 = inf is not finite");
        assert_eq!(fwd(Plif { w: f64::NAN, ..p }, &[1.0], &[0.0]), "w = NaN is not finite");
        assert_eq!(fwd(Plif { v_threshold: f64::NAN, ..p }, &[1.0], &[0.0]), "v_threshold = NaN is not finite");
        assert_eq!(fwd(Plif { reset: Reset::Hard { v_reset: f64::NEG_INFINITY }, ..p }, &[1.0], &[0.0]), "v_reset = -inf is not finite");
        let t = p.forward(&[1.0, 0.5], &[0.0], SpikeFn::Heaviside, &sur).unwrap();
        let bwd = |x: &[f64], ds: &[f64], dh: &[f64]| p.backward(x, &t, ds, dh, &sur).unwrap_err().to_string();
        assert_eq!(bwd(&[1.0], &[0.0, 0.0], &[0.0, 0.0]), "x has 1 entries where 2 are needed");
        assert_eq!(bwd(&[1.0, 0.5], &[0.0], &[0.0, 0.0]), "dl_ds has 1 entries where 2 are needed");
        assert_eq!(bwd(&[1.0, 0.5], &[0.0, 0.0], &[0.0]), "dl_dh has 1 entries where 2 are needed");
        assert_eq!(bwd(&[1.0, 0.5], &[0.0, f64::NAN], &[0.0, 0.0]), "a loss gradient = NaN is not finite");
        assert_eq!(bwd(&[1.0, 0.5], &[0.0, 0.0], &[f64::INFINITY, 0.0]), "a loss gradient = inf is not finite");
    }
}
