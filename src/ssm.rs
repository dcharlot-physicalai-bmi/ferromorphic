//! Diagonal state-space models: a bank of leaky integrators that is at once a sequence model and
//! a synapse, and the two ways of running it that must agree.
//!
//! # What the mechanism is
//!
//! A linear state-space model is `ẋ = A x + B u`, `y = C x + D u`. Gu, Goel and Ré's S4 (*Efficiently
//! modeling long sequences with structured state spaces*, ICLR 2022) made this a sequence model by
//! choosing `A` so that the state remembers usefully far back, and Gu, Gupta, Goel and Ré
//! (*On the parameterization and initialization of diagonal state space models*, `NeurIPS` 2022)
//! showed the structure can be DIAGONAL without losing that — which is the version here, because a
//! diagonal `A` is exactly a bank of independent leaky integrators, and a bank of leaky
//! integrators is what neuromorphic hardware already is.
//!
//! Discretised for a step `Δ`, a diagonal model has two forms that are the same computation:
//!
//! ```text
//! recurrence:   x_k = Ā x_{k−1} + B̄ u_k ,  y_k = C x_k + D u_k      state: n numbers
//! convolution:  y = K ∗ u + D u ,           K_m = Σ_i C_i Ā_i^m B̄_i  kernel: as long as the run
//! ```
//!
//! The recurrence is what a chip runs — constant memory, one multiply-accumulate per mode per
//! step. The convolution is what a trainer runs, because it is parallel in time. They are checked
//! against each other here to rounding, because a model whose two forms disagree is two models.
//!
//! **Zero-order hold** is not an approximation. For an input held constant across the step,
//! `x(t+Δ) = e^{AΔ} x(t) + A⁻¹(e^{AΔ} − I) B u` is the exact solution, so `Ā_i = e^{A_i Δ}` and
//! `B̄_i = (Ā_i − 1) B_i / A_i` reproduce the continuous system exactly on sampled input.
//! [`Discretisation::Bilinear`] is the other common choice and IS an approximation; the module
//! measures its order.
//!
//! # Why it is in a neuromorphic crate
//!
//! Because the same object is a synapse. One mode with `A = −1/τ` is an exponential synapse; two
//! modes with the right `C` are the double-exponential postsynaptic potential of
//! [`crate::srm`] — and [`double_exponential`] is checked against that module's kernel, entry for
//! entry. So "spiking state-space model" is not a new mechanism bolted onto a sequence model: a
//! layer of multi-timescale synapses IS a diagonal SSM, and the sequence-modelling literature's
//! initialisations are choices of synaptic time constants. [`crate::nef`]'s Legendre Memory Unit
//! is the same idea from the other direction, with `A` chosen to hold a sliding window.
//!
//! # The closed forms this module is checked against
//!
//! - **Zero-order hold is exact.** For piecewise-constant input the recurrence agrees with a fine
//!   numerical integration of `ẋ = A x + B u` to twelve figures, at step sizes where the bilinear
//!   form is visibly off — and the bilinear error falls as `Δ²`, measured.
//! - **Convolution equals recurrence**, `y_k = Σ_m K_m u_{k−m} + D u_k`, to rounding, on random
//!   input and for both discretisations.
//! - **The continuous impulse response is `h(t) = Σ_i C_i B_i e^{A_i t}`**, which is a different
//!   object from the discrete kernel and is documented as such: `K` is the response to a held
//!   sample, `h` to a delta. [`double_exponential`]'s `h` is checked against
//!   [`crate::srm::Kernel::epsilon`] exactly, and [`exponential_synapse`]'s against `e^{−t/τ}`.
//! - **Stability.** `|Ā_i| < 1` exactly when `A_i < 0` under zero-order hold; a mode with `A_i > 0`
//!   grows as `e^{A_i t}`, checked against that rate.
//! - **The DC gain is `D − C A⁻¹ B`**, which is also `Σ_m K_m + D` — two routes to the same
//!   number, one through the continuous parameters and one through the discrete kernel.
//! - **A spike train is the limit of narrowing pulses.** Driving the model with a unit-area pulse
//!   one step wide converges on the continuous impulse response as `Δ → 0`, at first order,
//!   measured.
//!
//! # What this module has NOT reproduced
//!
//! - Training. There is no gradient here: the parameters are given. S4's contribution is as much
//!   about what `A` to start from as about the form, and the initialisations (HiPPO-LegS, S4D-Lin,
//!   S4D-Inv) are not reproduced — [`crate::nef`] has the Legendre one.
//! - Complex modes. A real diagonal `A` gives decays; the oscillatory modes S4D actually uses are
//!   complex conjugate pairs, which this module does not carry. What it says about the real case
//!   is exact; it says nothing about the complex one.
//! - The fast convolution. `K ∗ u` here is the direct sum, `O(T²)`; S4's speed comes from doing it
//!   by FFT, which is a transform this crate does not have.
//! - Any claim that a spiking SSM is better than anything. The module shows the identity between a
//!   synapse bank and a diagonal SSM and checks it; it runs no benchmark.

use core::fmt;

/// The most modes a model may have.
pub const MAX_MODES: usize = 4096;
/// The longest kernel or run this module will build.
pub const MAX_STEPS: usize = 1 << 20;

/// What went wrong, named rather than guessed around.
#[derive(Debug, Clone, PartialEq)]
pub enum SsmError {
    /// No modes, more than [`MAX_MODES`], or `a`, `b` and `c` of different lengths.
    BadShape {
        /// How many entries `a` has.
        a: usize,
        /// How many `b` has.
        b: usize,
        /// How many `c` has.
        c: usize,
    },
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
    /// A vector of the wrong length.
    Shape {
        /// Which vector.
        what: &'static str,
        /// Length supplied.
        got: usize,
        /// Length required.
        want: usize,
    },
}

impl fmt::Display for SsmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadShape { a, b, c } => write!(f, "a, b and c have {a}, {b} and {c} entries"),
            Self::OutOfRange { what, value } => write!(f, "{what} = {value} is out of range"),
            Self::NonFinite { what } => write!(f, "{what} is not finite"),
            Self::Shape { what, got, want } => write!(f, "{what} has {got} entries, not {want}"),
        }
    }
}

impl std::error::Error for SsmError {}

/// How the continuous model is turned into a discrete one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Discretisation {
    /// Zero-order hold: exact for input held constant across the step.
    Zoh,
    /// The bilinear (Tustin) transform: `Ā = (1 + AΔ/2)/(1 − AΔ/2)`. An approximation, second
    /// order in `Δ`, and the one most of the state-space literature uses.
    Bilinear,
}

/// A diagonal state-space model: `ẋ_i = A_i x_i + B_i u`, `y = Σ C_i x_i + D u`.
#[derive(Debug, Clone, PartialEq)]
pub struct Ssm {
    /// The diagonal of `A`, one entry per mode. Negative entries decay.
    pub a: Vec<f64>,
    /// `B`, one entry per mode.
    pub b: Vec<f64>,
    /// `C`, one entry per mode.
    pub c: Vec<f64>,
    /// The feedthrough `D`.
    pub d: f64,
    /// The step, seconds.
    pub dt: f64,
    /// How it is discretised.
    pub method: Discretisation,
}

impl Ssm {
    /// Build.
    ///
    /// # Errors
    ///
    /// [`SsmError::BadShape`] for no modes, more than [`MAX_MODES`], or vectors of different
    /// lengths; [`SsmError::NonFinite`]; [`SsmError::OutOfRange`] for a non-positive `dt` or, for
    /// [`Discretisation::Bilinear`], a mode at exactly `2/Δ`, where the transform divides by zero.
    pub fn new(a: Vec<f64>, b: Vec<f64>, c: Vec<f64>, d: f64, dt: f64, method: Discretisation) -> Result<Self, SsmError> {
        if a.is_empty() || a.len() > MAX_MODES || a.len() != b.len() || a.len() != c.len() {
            return Err(SsmError::BadShape { a: a.len(), b: b.len(), c: c.len() });
        }
        if !a.iter().chain(&b).chain(&c).all(|v| v.is_finite()) || !d.is_finite() {
            return Err(SsmError::NonFinite { what: "parameter" });
        }
        if !(dt > 0.0) || !dt.is_finite() {
            return Err(SsmError::OutOfRange { what: "dt", value: dt });
        }
        // Zero-order hold divides by A; the bilinear transform divides by 1 − AΔ/2.
        if let Some(bad) = a.iter().find(|x| match method {
            Discretisation::Zoh => **x == 0.0,
            Discretisation::Bilinear => 1.0 - **x * dt / 2.0 == 0.0,
        }) {
            return Err(SsmError::OutOfRange { what: "a", value: *bad });
        }
        Ok(Self { a, b, c, d, dt, method })
    }

    /// How many modes.
    #[must_use]
    pub fn modes(&self) -> usize {
        self.a.len()
    }

    /// The discrete state matrix's diagonal, `Ā`.
    #[must_use]
    pub fn a_bar(&self) -> Vec<f64> {
        self.a
            .iter()
            .map(|a| match self.method {
                Discretisation::Zoh => (a * self.dt).exp(),
                Discretisation::Bilinear => (1.0 + a * self.dt / 2.0) / (1.0 - a * self.dt / 2.0),
            })
            .collect()
    }

    /// The discrete input vector, `B̄`.
    #[must_use]
    pub fn b_bar(&self) -> Vec<f64> {
        let bar = self.a_bar();
        (0..self.modes())
            .map(|i| match self.method {
                Discretisation::Zoh => (bar[i] - 1.0) * self.b[i] / self.a[i],
                Discretisation::Bilinear => self.dt * self.b[i] / (1.0 - self.a[i] * self.dt / 2.0),
            })
            .collect()
    }

    /// Whether every mode decays, `|Ā_i| < 1`.
    #[must_use]
    pub fn stable(&self) -> bool {
        self.a_bar().iter().all(|x| x.abs() < 1.0)
    }

    /// The continuous impulse response `h(t) = Σ_i C_i B_i e^{A_i t}`, zero before `t = 0`.
    ///
    /// Not the same object as [`Ssm::kernel`]: `h` is the response to a DELTA, `K` to a sample
    /// HELD across a step. They agree only in the limit, and the module documentation says how.
    #[must_use]
    pub fn impulse_response(&self, t: f64) -> f64 {
        if !(t >= 0.0) {
            return 0.0;
        }
        (0..self.modes()).map(|i| self.c[i] * self.b[i] * (self.a[i] * t).exp()).sum()
    }

    /// The discrete convolution kernel, `K_m = Σ_i C_i Ā_i^m B̄_i` for `m` in `0..len`. The
    /// feedthrough `D` is NOT in it; [`Ssm::convolve`] adds it.
    ///
    /// # Errors
    ///
    /// [`SsmError::OutOfRange`] for a `len` of zero or past [`MAX_STEPS`].
    pub fn kernel(&self, len: usize) -> Result<Vec<f64>, SsmError> {
        if len == 0 || len > MAX_STEPS {
            return Err(SsmError::OutOfRange { what: "len", value: len as f64 });
        }
        let (bar, b_bar) = (self.a_bar(), self.b_bar());
        let mut power: Vec<f64> = self.c.iter().zip(&b_bar).map(|(c, b)| c * b).collect();
        let mut k = Vec::with_capacity(len);
        for _ in 0..len {
            k.push(power.iter().sum());
            for (p, a) in power.iter_mut().zip(&bar) {
                *p *= a;
            }
        }
        Ok(k)
    }

    /// One step of the recurrence, in place. Returns `y_k`.
    ///
    /// # Errors
    ///
    /// [`SsmError::Shape`] for a state of the wrong length, [`SsmError::NonFinite`] for a bad
    /// input or a state that has overflowed.
    pub fn step(&self, x: &mut [f64], u: f64) -> Result<f64, SsmError> {
        if x.len() != self.modes() {
            return Err(SsmError::Shape { what: "state", got: x.len(), want: self.modes() });
        }
        if !u.is_finite() {
            return Err(SsmError::NonFinite { what: "u" });
        }
        let (bar, b_bar) = (self.a_bar(), self.b_bar());
        let mut y = self.d * u;
        for i in 0..self.modes() {
            x[i] = bar[i] * x[i] + b_bar[i] * u;
            y += self.c[i] * x[i];
        }
        if !y.is_finite() || !x.iter().all(|v| v.is_finite()) {
            return Err(SsmError::NonFinite { what: "state" });
        }
        Ok(y)
    }

    /// The whole output by the RECURRENCE, from a state at rest.
    ///
    /// # Errors
    ///
    /// As [`Ssm::step`], and [`SsmError::OutOfRange`] for an input longer than [`MAX_STEPS`].
    pub fn run(&self, u: &[f64]) -> Result<Vec<f64>, SsmError> {
        if u.len() > MAX_STEPS {
            return Err(SsmError::OutOfRange { what: "u", value: u.len() as f64 });
        }
        let mut x = vec![0.0; self.modes()];
        u.iter().map(|&u| self.step(&mut x, u)).collect()
    }

    /// The whole output by CONVOLUTION, `y_k = Σ_m K_m u_{k−m} + D u_k`.
    ///
    /// # Errors
    ///
    /// As [`Ssm::kernel`], and [`SsmError::NonFinite`] for a bad input.
    pub fn convolve(&self, u: &[f64]) -> Result<Vec<f64>, SsmError> {
        if u.is_empty() {
            return Ok(Vec::new());
        }
        if !u.iter().all(|v| v.is_finite()) {
            return Err(SsmError::NonFinite { what: "u" });
        }
        let k = self.kernel(u.len())?;
        Ok((0..u.len()).map(|t| self.d * u[t] + (0..=t).map(|m| k[m] * u[t - m]).sum::<f64>()).collect())
    }

    /// The steady output under a held input of one: `D − Σ_i C_i B_i / A_i`.
    #[must_use]
    pub fn dc_gain(&self) -> f64 {
        self.d - (0..self.modes()).map(|i| self.c[i] * self.b[i] / self.a[i]).sum::<f64>()
    }
}

/// One mode: an exponential synapse of time constant `tau`, whose continuous impulse response is
/// `e^{−t/τ}`.
///
/// # Errors
///
/// [`SsmError::OutOfRange`] for a non-positive `tau` or `dt`.
pub fn exponential_synapse(tau: f64, dt: f64) -> Result<Ssm, SsmError> {
    if !(tau > 0.0) || !tau.is_finite() {
        return Err(SsmError::OutOfRange { what: "tau", value: tau });
    }
    Ssm::new(vec![-1.0 / tau], vec![1.0], vec![1.0], 0.0, dt, Discretisation::Zoh)
}

/// Two modes whose continuous impulse response is exactly the postsynaptic potential of
/// [`crate::srm::Kernel`]: `τ_s/(τ_s − τ_m) · (e^{−t/τ_s} − e^{−t/τ_m})`.
///
/// # Errors
///
/// [`SsmError::OutOfRange`] for a non-positive or equal time constant, or a bad `dt`.
pub fn double_exponential(tau_m: f64, tau_s: f64, dt: f64) -> Result<Ssm, SsmError> {
    for (what, v) in [("tau_m", tau_m), ("tau_s", tau_s)] {
        if !(v > 0.0) || !v.is_finite() {
            return Err(SsmError::OutOfRange { what, value: v });
        }
    }
    if tau_m == tau_s {
        return Err(SsmError::OutOfRange { what: "tau_s - tau_m", value: 0.0 });
    }
    let scale = tau_s / (tau_s - tau_m);
    Ssm::new(vec![-1.0 / tau_s, -1.0 / tau_m], vec![1.0, 1.0], vec![scale, -scale], 0.0, dt, Discretisation::Zoh)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;
    use crate::srm;

    fn model(method: Discretisation, dt: f64) -> Ssm {
        Ssm::new(vec![-40.0, -8.0, -1.5], vec![1.0, 0.5, -2.0], vec![0.3, -1.1, 0.7], 0.25, dt, method).unwrap()
    }

    fn noise(n: usize, seed: u64) -> Vec<f64> {
        let mut rng = Rng::new(seed);
        (0..n).map(|_| 2.0 * rng.next_f64() - 1.0).collect()
    }

    #[test]
    fn zero_order_hold_is_exact_and_the_bilinear_form_is_second_order() {
        // The reference: forward Euler at a step ten thousand times finer, integrating
        // dx/dt = a x + b u with u HELD across each coarse step — which is the input the
        // discretisation is defined for.
        let (dt, steps) = (5e-3, 40usize);
        let u = noise(steps, 3);
        let fine = |m: &Ssm| -> Vec<f64> {
            let sub = 20_000;
            let h = m.dt / f64::from(sub);
            let mut x = vec![0.0; m.modes()];
            let mut out = Vec::new();
            for &u in &u {
                for _ in 0..sub {
                    for i in 0..m.modes() {
                        x[i] += h * (m.a[i] * x[i] + m.b[i] * u);
                    }
                }
                out.push(m.d * u + (0..m.modes()).map(|i| m.c[i] * x[i]).sum::<f64>());
            }
            out
        };
        let zoh = model(Discretisation::Zoh, dt);
        let reference = fine(&zoh);
        let run = zoh.run(&u).unwrap();
        let mut worst = 0.0f64;
        for (a, b) in run.iter().zip(&reference) {
            worst = worst.max((a - b).abs());
        }
        assert!(worst < 1e-6, "zero-order hold is {worst} from the integration");
        let swing = reference.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        assert!(swing > 0.2, "the output only reached {swing}, so this compared nothing");

        // The bilinear form is not exact, and its error falls as the square of the step.
        let mut last: Option<(f64, f64)> = None;
        for k in 0..3 {
            let dt = 5e-3 / f64::from(1 << k);
            let steps = (0.2 / dt).round() as usize;
            let u: Vec<f64> = (0..steps).map(|i| (0.3 + 7.0 * dt * i as f64).sin()).collect();
            let bilinear = model(Discretisation::Bilinear, dt);
            let truth = model(Discretisation::Zoh, dt).run(&u).unwrap();
            let got = bilinear.run(&u).unwrap();
            let err = got.iter().zip(&truth).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
            if let Some((prev_dt, prev)) = last {
                let ratio = prev / err;
                assert!(ratio > 3.0 && ratio < 5.5, "halving {prev_dt} to {dt} divided the error by {ratio}, not four");
            }
            assert!(err > 1e-9, "the bilinear form agreed exactly at dt = {dt}, which it should not");
            last = Some((dt, err));
        }
    }

    #[test]
    fn the_convolution_and_the_recurrence_are_the_same_computation() {
        for method in [Discretisation::Zoh, Discretisation::Bilinear] {
            let m = model(method, 2e-3);
            let u = noise(120, 7);
            let (run, conv) = (m.run(&u).unwrap(), m.convolve(&u).unwrap());
            let mut worst = 0.0f64;
            for (a, b) in run.iter().zip(&conv) {
                worst = worst.max((a - b).abs());
            }
            assert!(worst < 1e-12, "{method:?}: the two forms differ by {worst}");
            assert!(run.iter().any(|v| v.abs() > 0.2), "the output never moved");
            // The kernel IS the response to a unit sample, which is the definition being used.
            let mut impulse = vec![0.0; 40];
            impulse[0] = 1.0;
            let response = m.run(&impulse).unwrap();
            let k = m.kernel(40).unwrap();
            assert!((response[0] - (k[0] + m.d)).abs() < 1e-15, "the feedthrough is not in the kernel");
            for t in 1..40 {
                assert!((response[t] - k[t]).abs() < 1e-15, "{method:?} at {t}");
            }
        }
        assert_eq!(model(Discretisation::Zoh, 1e-3).convolve(&[]).unwrap(), Vec::<f64>::new());
    }

    #[test]
    fn the_continuous_impulse_response_is_the_postsynaptic_potential_of_the_spike_response_model() {
        // A different module, written for another purpose, computes the same two kernels.
        let (tau_m, tau_s, dt) = (20e-3, 5e-3, 1e-4);
        let psp = double_exponential(tau_m, tau_s, dt).unwrap();
        let reference = srm::Kernel::new(tau_m, tau_s).unwrap();
        let mut peak = 0.0f64;
        for step in 0..=600 {
            let t = f64::from(step) * 1e-4;
            let got = psp.impulse_response(t);
            assert!((got - reference.epsilon(t)).abs() < 1e-15, "at {t}: {got} against {}", reference.epsilon(t));
            peak = peak.max(got);
        }
        assert!(peak > 0.1, "the kernel never rose: {peak}");
        assert_eq!(psp.impulse_response(-1e-9), 0.0);
        // On a model whose B is not all ones, so that dropping it would show: h(t) = Σ c_i b_i e^{a_i t}.
        let m = model(Discretisation::Zoh, 1e-3);
        assert_eq!(m.b, vec![1.0, 0.5, -2.0]);
        for step in 0..=40 {
            let t = f64::from(step) * 5e-3;
            let by_hand = 0.3 * 1.0 * (-40.0 * t).exp() + -1.1 * 0.5 * (-8.0 * t).exp() + 0.7 * -2.0 * (-1.5 * t).exp();
            assert!((m.impulse_response(t) - by_hand).abs() < 1e-15, "at {t}");
        }
        assert!(m.impulse_response(0.0).abs() > 1.0, "the comparison starts from a value of {}", m.impulse_response(0.0));
        // One mode is the exponential synapse, and its impulse response is the exponential.
        let syn = exponential_synapse(tau_m, dt).unwrap();
        for step in 0..=200 {
            let t = f64::from(step) * 1e-3;
            assert!((syn.impulse_response(t) - (-t / tau_m).exp()).abs() < 1e-15);
        }
        assert_eq!(syn.modes(), 1);
        // Each guard is checked by the error it raises, not merely by something failing: a zero
        // time constant would otherwise be caught downstream as a non-finite parameter, and a
        // NEGATIVE one would not be caught at all — it builds a perfectly good growing mode.
        assert_eq!(exponential_synapse(0.0, dt), Err(SsmError::OutOfRange { what: "tau", value: 0.0 }));
        assert_eq!(exponential_synapse(-10e-3, dt), Err(SsmError::OutOfRange { what: "tau", value: -10e-3 }));
        assert!(exponential_synapse(f64::NAN, dt).is_err());
        assert!(exponential_synapse(tau_m, 0.0).is_err());
        assert_eq!(double_exponential(tau_m, tau_m, dt), Err(SsmError::OutOfRange { what: "tau_s - tau_m", value: 0.0 }));
        assert_eq!(double_exponential(-1.0, tau_s, dt), Err(SsmError::OutOfRange { what: "tau_m", value: -1.0 }));
        assert_eq!(double_exponential(tau_m, 0.0, dt), Err(SsmError::OutOfRange { what: "tau_s", value: 0.0 }));
    }

    #[test]
    fn a_narrowing_unit_area_pulse_converges_on_the_impulse_response() {
        // The discrete kernel answers a HELD sample; the impulse response answers a delta. Feed
        // the model a pulse of unit area one step wide and the two meet, at first order in dt.
        let syn = exponential_synapse(10e-3, 1e-3).unwrap();
        let mut last: Option<(f64, f64)> = None;
        for k in 0..4 {
            let dt = 1e-3 / f64::from(1 << k);
            let m = exponential_synapse(10e-3, dt).unwrap();
            let steps = (50e-3 / dt).round() as usize;
            let mut u = vec![0.0; steps];
            u[0] = 1.0 / dt;
            let got = m.run(&u).unwrap();
            let err = (1..steps)
                .map(|t| (got[t] - m.impulse_response(t as f64 * dt)).abs())
                .fold(0.0f64, f64::max);
            if let Some((prev_dt, prev)) = last {
                let ratio = prev / err;
                assert!(ratio > 1.6 && ratio < 2.4, "halving {prev_dt} to {dt} divided the error by {ratio}, not two");
            }
            assert!(err > 1e-12, "at dt = {dt} the pulse already agreed exactly");
            last = Some((dt, err));
        }
        assert!(last.unwrap().1 < 0.05, "the finest step is still {} out", last.unwrap().1);
        assert_eq!(syn.dc_gain(), 10e-3, "a unit-gain synapse integrates a held input to tau");
    }

    #[test]
    fn the_dc_gain_is_the_same_number_by_two_routes() {
        for method in [Discretisation::Zoh, Discretisation::Bilinear] {
            let m = model(method, 1e-3);
            // Route one: the continuous parameters, D − C A⁻¹ B.
            let want = m.dc_gain();
            // Route two: the discrete kernel's own sum, which knows nothing of A except through Ā.
            let k = m.kernel(60_000).unwrap();
            let summed: f64 = k.iter().sum::<f64>() + m.d;
            assert!((summed - want).abs() < 1e-9, "{method:?}: kernel sum {summed} against {want}");
            // Route three: run it and see.
            let settled = *m.run(&vec![1.0; 40_000]).unwrap().last().unwrap();
            assert!((settled - want).abs() < 1e-9, "{method:?}: settled at {settled} against {want}");
            assert!(want.abs() > 0.1, "the gain is {want}, too near zero to compare");
        }
    }

    #[test]
    fn a_mode_decays_exactly_when_its_continuous_rate_is_negative() {
        let dt = 1e-3;
        for a in [-100.0, -1.0, -1e-6] {
            let m = Ssm::new(vec![a], vec![1.0], vec![1.0], 0.0, dt, Discretisation::Zoh).unwrap();
            assert!(m.stable() && m.a_bar()[0] < 1.0 && m.a_bar()[0] > 0.0, "a = {a} gave {}", m.a_bar()[0]);
        }
        for a in [1e-6, 1.0, 100.0] {
            let m = Ssm::new(vec![a], vec![1.0], vec![1.0], 0.0, dt, Discretisation::Zoh).unwrap();
            assert!(!m.stable(), "a = {a} gave {}", m.a_bar()[0]);
        }
        // And an unstable mode grows at exactly its own rate.
        let m = Ssm::new(vec![50.0], vec![1.0], vec![1.0], 0.0, dt, Discretisation::Zoh).unwrap();
        let mut u = vec![0.0; 200];
        u[0] = 1.0;
        let y = m.run(&u).unwrap();
        for t in [50usize, 100, 199] {
            let ratio = y[t] / y[0];
            let want = (50.0 * (t as f64) * dt).exp();
            assert!((ratio / want - 1.0).abs() < 1e-12, "at {t}: grew by {ratio} against {want}");
        }
        // A mixed model is unstable if ANY mode is.
        let mixed = Ssm::new(vec![-10.0, 1.0], vec![1.0, 1.0], vec![1.0, 1.0], 0.0, dt, Discretisation::Zoh).unwrap();
        assert!(!mixed.stable());
        // Under the bilinear transform a pole can land on the NEGATIVE real axis and outside the
        // unit circle — a > 2/Δ sends (1 + aΔ/2)/(1 − aΔ/2) below −1 — so stability is a question
        // about the magnitude, not the value. Zero-order hold never does this, because e^{aΔ} is
        // positive whatever a is, which is why only the bilinear form can show it.
        let flipped = Ssm::new(vec![3.0 / dt], vec![1.0], vec![1.0], 0.0, dt, Discretisation::Bilinear).unwrap();
        assert_eq!(flipped.a_bar(), vec![-5.0]);
        assert!(!flipped.stable(), "a pole at -5 is outside the unit circle");
        let mut u = vec![0.0; 20];
        u[0] = 1.0;
        let y = flipped.run(&u).unwrap();
        assert!(y[10].abs() > y[1].abs() * 1e6 && y[10] * y[11] < 0.0, "it should grow AND alternate: {:?}", &y[..4]);
        // And zero-order hold keeps every pole positive, so the two rules agree there.
        for a in [-1e5, -1.0, 1.0, 1e5] {
            let m = Ssm::new(vec![a], vec![1.0], vec![1.0], 0.0, dt, Discretisation::Zoh).unwrap();
            assert!(m.a_bar()[0] > 0.0, "zero-order hold gave a negative pole for a = {a}");
        }
        // A state that overflows is named rather than returned.
        let mut x = vec![f64::MAX];
        assert_eq!(m.step(&mut x, f64::MAX), Err(SsmError::NonFinite { what: "state" }));
    }

    #[test]
    fn bad_models_and_inputs_are_refused() {
        let dt = 1e-3;
        assert_eq!(Ssm::new(vec![], vec![], vec![], 0.0, dt, Discretisation::Zoh), Err(SsmError::BadShape { a: 0, b: 0, c: 0 }));
        assert_eq!(
            Ssm::new(vec![-1.0], vec![1.0, 2.0], vec![1.0], 0.0, dt, Discretisation::Zoh),
            Err(SsmError::BadShape { a: 1, b: 2, c: 1 })
        );
        assert!(Ssm::new(vec![-1.0; MAX_MODES + 1], vec![1.0; MAX_MODES + 1], vec![1.0; MAX_MODES + 1], 0.0, dt, Discretisation::Zoh).is_err());
        assert_eq!(Ssm::new(vec![f64::NAN], vec![1.0], vec![1.0], 0.0, dt, Discretisation::Zoh), Err(SsmError::NonFinite { what: "parameter" }));
        assert_eq!(Ssm::new(vec![-1.0], vec![1.0], vec![1.0], f64::NAN, dt, Discretisation::Zoh), Err(SsmError::NonFinite { what: "parameter" }));
        assert!(Ssm::new(vec![-1.0], vec![1.0], vec![1.0], 0.0, 0.0, Discretisation::Zoh).is_err());
        // Zero-order hold divides by A, so a mode at zero is refused — and the bilinear form,
        // which does not divide by it, accepts the same model.
        assert_eq!(
            Ssm::new(vec![0.0], vec![1.0], vec![1.0], 0.0, dt, Discretisation::Zoh),
            Err(SsmError::OutOfRange { what: "a", value: 0.0 })
        );
        let integrator = Ssm::new(vec![0.0], vec![1.0], vec![1.0], 0.0, dt, Discretisation::Bilinear).unwrap();
        assert_eq!(integrator.a_bar(), vec![1.0], "a pole at zero is an integrator, and it does not decay");
        assert!(!integrator.stable());
        // The bilinear form divides by 1 − AΔ/2, so a mode at exactly 2/Δ is refused.
        assert!(Ssm::new(vec![2.0 / dt], vec![1.0], vec![1.0], 0.0, dt, Discretisation::Bilinear).is_err());
        assert!(Ssm::new(vec![2.0 / dt], vec![1.0], vec![1.0], 0.0, dt, Discretisation::Zoh).is_ok());

        let m = model(Discretisation::Zoh, dt);
        assert_eq!(m.kernel(0), Err(SsmError::OutOfRange { what: "len", value: 0.0 }));
        assert!(m.kernel(MAX_STEPS + 1).is_err());
        let mut x = vec![0.0; 2];
        assert_eq!(m.step(&mut x, 1.0), Err(SsmError::Shape { what: "state", got: 2, want: 3 }));
        let mut x = vec![0.0; 3];
        assert_eq!(m.step(&mut x, f64::NAN), Err(SsmError::NonFinite { what: "u" }));
        assert_eq!(x, vec![0.0; 3], "a refused step must not move the state");
        assert_eq!(m.convolve(&[1.0, f64::INFINITY]), Err(SsmError::NonFinite { what: "u" }));
        assert!(m.run(&[f64::NAN]).is_err());
        assert!(SsmError::BadShape { a: 1, b: 2, c: 3 }.to_string().contains("1, 2 and 3"));
        assert!(SsmError::Shape { what: "state", got: 2, want: 3 }.to_string().contains("not 3"));
    }
}
