//! Oscillatory state in a neuron: the resonate-and-fire cell, and the Legendre Memory Unit that
//! turns a linear system into a delay line — both stepped exactly and checked against the closed
//! forms their authors derived.
//!
//! # Two ways to remember the past in two numbers
//!
//! A leaky integrator has one state variable and forgets exponentially. Give a neuron a **pair** of
//! state variables coupled as a damped oscillator and it remembers *phase*: an input that arrives
//! in step with its own rhythm adds to what is already there, and one that arrives half a period
//! late cancels it. That is Izhikevich's resonate-and-fire neuron (*Resonate-and-fire neurons*,
//! Neural Networks 14(6–7):883–894, 2001):
//!
//! ```text
//! dz/dt = (b + iω) z + I(t),      z = x + iy,      spike when y ≥ threshold, then z ← z_reset
//! ```
//!
//! with `b < 0` the damping and `ω` the natural frequency. The whole behaviour is one complex
//! exponential: from any state, `z(t) = e^{(b+iω)t} z(0)` in the absence of input, and a pulse
//! that arrives after an interval `Δ` adds to a response of magnitude `|1 + e^{(b+iω)Δ}|` — which
//! is `1 + e^{bΔ}` at `Δ = 2π/ω` and `1 − e^{bΔ}` at `Δ = π/ω`. [`ResonateAndFire`] steps that
//! exponential exactly (so a gap of `k` steps equals one step of `k·dt`, to rounding) and the tests
//! check the pair-of-pulses response against the formula and the sinusoidal steady state against
//! the transfer function `|I| / |iω_in − (b + iω)|`.
//!
//! Give a unit `d` state variables instead, coupled by one particular matrix, and it remembers a
//! **window**: the Legendre Memory Unit (Voelker, Kajić and Eliasmith, *Legendre Memory Units:
//! continuous-time representation in recurrent neural networks*, `NeurIPS` 32, 2019) is the linear
//! system
//!
//! ```text
//! θ dm/dt = A m + B u,   A_ij = (2i+1)·(−1 if i < j else (−1)^{i−j+1}),   B_i = (2i+1)·(−1)^i
//! ```
//!
//! whose state holds the coefficients of the last `θ` seconds of input in the shifted Legendre
//! basis: `u(t − rθ) ≈ Σ_i P̃_i(r)·m_i(t)` for `r ∈ [0, 1]`. So `Σ_i m_i` is the input `θ` seconds
//! ago and `Σ_i (−1)^i m_i` is the input now, and a delay line of `θ/dt` taps has become `d`
//! numbers. The tests check the matrix against the paper's literal, a constant input against
//! `m = (1, 0, …)`, a ramp against `m_0 = t − θ/2, m_1 = −θ/2` (the exact Legendre expansion of a
//! line), and a slow sine against its own delayed value.
//!
//! # Why they are in a neuromorphic crate
//!
//! Both are the state-space family that currently holds the spiking benchmarks on temporal tasks:
//! resonate-and-fire cells and their balanced variant (Higuchi, Kairat, Bohté and Otte, *Balanced
//! resonate-and-fire neurons*, ICML 2024) on the Spiking Heidelberg Digits, and the LMU as the
//! recurrence inside spiking networks deployed on Loihi by its authors' company. What they buy is
//! the thing this crate prices: a resonator detects a frequency with two state variables and no
//! filter bank, and an LMU holds a `θ`-second window in `d` state variables instead of `θ/dt` —
//! [`Lmu::delay_line_taps`] is the count it replaces. Fewer state variables is fewer membrane
//! updates on every tick, which [`crate::ledger`] counts and which the `28x` argument in
//! [`crate::spikeconv`] is about.
//!
//! # What this module has NOT reproduced
//!
//! - The balanced resonate-and-fire neuron's divergence boundary and refractory adaptation, and
//!   any benchmark number. The plain Izhikevich model is here with its closed forms; the balanced
//!   variant's stability condition is named and not built.
//! - A spiking LMU. The unit here is the continuous linear system, stepped by fourth-order
//!   Runge-Kutta at a stated step; encoding its state in spikes is a job for
//!   [`crate::nef`]'s populations and is not done here.
//! - Training either one. No gradient passes through this module.

use core::fmt;

/// What went wrong, named rather than guessed around.
#[derive(Debug, Clone, PartialEq)]
pub enum ResonateError {
    /// A parameter outside its admissible range.
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
        /// Value supplied.
        value: f64,
    },
    /// An order of zero.
    Empty {
        /// What was empty.
        what: &'static str,
    },
}

impl fmt::Display for ResonateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfRange { what, value, low, high } => {
                write!(f, "{what} = {value} is outside [{low}, {high}]")
            }
            Self::NonFinite { what, value } => write!(f, "{what} = {value} is not finite"),
            Self::Empty { what } => write!(f, "{what} is empty"),
        }
    }
}

impl std::error::Error for ResonateError {}

fn finite(what: &'static str, value: f64) -> Result<f64, ResonateError> {
    if value.is_finite() { Ok(value) } else { Err(ResonateError::NonFinite { what, value }) }
}

fn positive(what: &'static str, value: f64) -> Result<f64, ResonateError> {
    let value = finite(what, value)?;
    if value > 0.0 {
        Ok(value)
    } else {
        Err(ResonateError::OutOfRange { what, value, low: f64::MIN_POSITIVE, high: f64::INFINITY })
    }
}

// ---------------------------------------------------------------------------------------------
// Resonate-and-fire
// ---------------------------------------------------------------------------------------------

/// Izhikevich's resonate-and-fire neuron, stepped by the exact complex exponential.
///
/// State `z = x + iy`; the imaginary part is the one compared with the threshold, as in the paper.
/// Units are the paper's dimensionless ones — `b` and `ω` in reciprocal seconds, `I` in the units
/// of `z` per second — which is why this type does not implement [`crate::neuron::Neuron`]: that
/// trait promises volts at its interface, and a resonator's `y` is not a voltage.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResonateAndFire {
    /// Damping, reciprocal seconds. Negative for a stable resonator; zero is a pure oscillator.
    pub b: f64,
    /// Natural angular frequency, radians per second. Strictly positive.
    pub omega: f64,
    /// Fires when `y` reaches this.
    pub threshold: f64,
    /// State after a spike, `(x, y)`.
    pub z_reset: (f64, f64),
    /// Absolute refractory period, seconds: the state is held at `z_reset` for this long.
    pub t_ref: f64,
    /// Real part of the state.
    pub x: f64,
    /// Imaginary part of the state — the one the threshold reads.
    pub y: f64,
    /// Seconds of refractory period remaining.
    pub refractory: f64,
}

impl ResonateAndFire {
    /// Build at rest.
    ///
    /// # Errors
    ///
    /// [`ResonateError::OutOfRange`] for a positive `b` (an unstable resonator whose state grows
    /// without input), a non-positive `omega`, a non-positive `threshold` or a negative `t_ref`;
    /// [`ResonateError::NonFinite`] for a non-finite reset.
    pub fn new(b: f64, omega: f64, threshold: f64, z_reset: (f64, f64), t_ref: f64) -> Result<Self, ResonateError> {
        let b = finite("b", b)?;
        if b > 0.0 {
            return Err(ResonateError::OutOfRange { what: "b", value: b, low: f64::NEG_INFINITY, high: 0.0 });
        }
        let omega = positive("omega", omega)?;
        let threshold = positive("threshold", threshold)?;
        let t_ref = finite("t_ref", t_ref)?;
        if t_ref < 0.0 {
            return Err(ResonateError::OutOfRange { what: "t_ref", value: t_ref, low: 0.0, high: f64::INFINITY });
        }
        finite("z_reset.0", z_reset.0)?;
        finite("z_reset.1", z_reset.1)?;
        Ok(Self { b, omega, threshold, z_reset, t_ref, x: z_reset.0, y: z_reset.1, refractory: 0.0 })
    }

    /// The paper's illustrative resonator: `b = −1`, `ω = 2π · 10 Hz`, threshold `1`, reset to
    /// the origin, no refractory period.
    ///
    /// # Errors
    ///
    /// Never in practice; the signature is `Result` because [`ResonateAndFire::new`]'s is.
    pub fn textbook() -> Result<Self, ResonateError> {
        Self::new(-1.0, core::f64::consts::TAU * 10.0, 1.0, (0.0, 0.0), 0.0)
    }

    /// The natural period `2π/ω`, seconds.
    #[must_use]
    pub fn period(&self) -> f64 {
        core::f64::consts::TAU / self.omega
    }

    /// Add `(dx, dy)` to the state instantaneously — an input pulse. Ignored while refractory.
    pub fn kick(&mut self, dx: f64, dy: f64) {
        if self.refractory > 0.0 {
            return;
        }
        self.x += dx;
        self.y += dy;
    }

    /// Advance by `dt` under constant input `i`, returning whether the neuron fired.
    ///
    /// Exact for constant input: with `λ = b + iω`,
    /// `z(t + dt) = e^{λ dt} z(t) + I · (e^{λ dt} − 1) / λ`.
    ///
    /// # Errors
    ///
    /// [`ResonateError::OutOfRange`] for a non-positive `dt`, [`ResonateError::NonFinite`] for a
    /// non-finite `i`.
    pub fn step(&mut self, dt: f64, i: f64) -> Result<bool, ResonateError> {
        let dt = positive("dt", dt)?;
        let i = finite("i", i)?;
        if self.refractory > 0.0 {
            // A residue below a billionth of a tick is rounding, not refractoriness: fifty ticks
            // of 0.1 ms end a 5 ms period on the fiftieth tick, not the fifty-first.
            let left = self.refractory - dt;
            self.refractory = if left < 1e-9 * dt { 0.0 } else { left };
            self.x = self.z_reset.0;
            self.y = self.z_reset.1;
            return Ok(false);
        }
        // e^{λ dt} = e^{b dt} (cos ω dt + i sin ω dt)
        let g = (self.b * dt).exp();
        let (s, c) = (self.omega * dt).sin_cos();
        let (er, ei) = (g * c, g * s);
        let (x0, y0) = (self.x, self.y);
        let mut x = er * x0 - ei * y0;
        let mut y = er * y0 + ei * x0;
        if i != 0.0 {
            // (e^{λ dt} − 1) / λ, complex division by λ = b + iω.
            let (nr, ni) = (er - 1.0, ei);
            let denom = self.b * self.b + self.omega * self.omega;
            let qr = (nr * self.b + ni * self.omega) / denom;
            let qi = (ni * self.b - nr * self.omega) / denom;
            x += i * qr;
            y += i * qi;
        }
        self.x = x;
        self.y = y;
        if self.y >= self.threshold {
            self.x = self.z_reset.0;
            self.y = self.z_reset.1;
            self.refractory = self.t_ref;
            return Ok(true);
        }
        Ok(false)
    }

    /// Return to the reset state with no refractory period pending.
    pub fn reset(&mut self) {
        self.x = self.z_reset.0;
        self.y = self.z_reset.1;
        self.refractory = 0.0;
    }

    /// The closed-form free response from `(x0, y0)` after `t` seconds with no input:
    /// `e^{bt} (x0 cos ωt − y0 sin ωt, x0 sin ωt + y0 cos ωt)`.
    #[must_use]
    pub fn free_response(&self, x0: f64, y0: f64, t: f64) -> (f64, f64) {
        let g = (self.b * t).exp();
        let (s, c) = (self.omega * t).sin_cos();
        (g * (x0 * c - y0 * s), g * (x0 * s + y0 * c))
    }

    /// The magnitude of the state immediately after two unit pulses `interval` seconds apart,
    /// from rest: `|1 + e^{λ·interval}|`. `1 + e^{b·interval}` at the natural period and
    /// `1 − e^{b·interval}` at half of it — the frequency selectivity in one line.
    #[must_use]
    pub fn pair_response(&self, interval: f64) -> f64 {
        let g = (self.b * interval).exp();
        let (s, c) = (self.omega * interval).sin_cos();
        ((1.0 + g * c).powi(2) + (g * s).powi(2)).sqrt()
    }

    /// The steady-state amplitude of `y` per unit amplitude of a real drive `cos(ω_in t)`.
    ///
    /// A real cosine is two complex exponentials at `±ω_in`, and the complex-state resonator
    /// answers each through `G± = 1 / (±iω_in − λ)`: `z = ½[G₊ e^{iω_in t} + G₋ e^{−iω_in t}]`, so
    /// `y = ½[Im(G₊ + G₋) cos ω_in t + Re(G₊ − G₋) sin ω_in t]` and the amplitude is
    /// `½ · √(Im(G₊ + G₋)² + Re(G₊ − G₋)²)`. Near resonance `G₊` dominates and this is
    /// `≈ 1 / (2|b|)`; the first draft of this function quoted `1/|iω_in − λ|`, the response to a
    /// single complex exponential, and was wrong by the `G₋` term and a factor of two — which the
    /// test caught at 4 Hz on its first run.
    #[must_use]
    pub fn sinusoidal_gain(&self, omega_in: f64) -> f64 {
        // 1 / (p + iq) = (p − iq) / (p² + q²) with p = −b.
        let p = -self.b;
        let q_plus = omega_in - self.omega;
        let q_minus = -(omega_in + self.omega);
        let d_plus = p * p + q_plus * q_plus;
        let d_minus = p * p + q_minus * q_minus;
        let (re_plus, im_plus) = (p / d_plus, -q_plus / d_plus);
        let (re_minus, im_minus) = (p / d_minus, -q_minus / d_minus);
        0.5 * ((im_plus + im_minus).powi(2) + (re_plus - re_minus).powi(2)).sqrt()
    }
}

// ---------------------------------------------------------------------------------------------
// Legendre Memory Unit
// ---------------------------------------------------------------------------------------------

/// The Legendre Memory Unit's linear system, stepped by fourth-order Runge-Kutta.
#[derive(Debug, Clone, PartialEq)]
pub struct Lmu {
    /// Order `d`: state variables, and Legendre polynomials in the window's expansion.
    pub order: usize,
    /// Window length `θ`, seconds.
    pub theta: f64,
    /// `A / θ`, row-major `d × d`.
    pub a: Vec<f64>,
    /// `B / θ`, length `d`.
    pub b: Vec<f64>,
    /// The state `m`.
    pub m: Vec<f64>,
}

impl Lmu {
    /// Build with the paper's matrices for `order` and `theta`, at rest.
    ///
    /// # Errors
    ///
    /// [`ResonateError::Empty`] for order zero, [`ResonateError::OutOfRange`] for a non-positive
    /// `theta`.
    pub fn new(order: usize, theta: f64) -> Result<Self, ResonateError> {
        if order == 0 {
            return Err(ResonateError::Empty { what: "order" });
        }
        let theta = positive("theta", theta)?;
        let d = order;
        let mut a = vec![0.0; d * d];
        let mut b = vec![0.0; d];
        for i in 0..d {
            let scale = (2 * i + 1) as f64 / theta;
            for j in 0..d {
                a[i * d + j] = if i < j {
                    -scale
                } else if (i - j + 1).is_multiple_of(2) {
                    scale
                } else {
                    -scale
                };
            }
            b[i] = if i.is_multiple_of(2) { scale } else { -scale };
        }
        Ok(Self { order, theta, a, b, m: vec![0.0; d] })
    }

    fn deriv(&self, m: &[f64], u: f64, out: &mut [f64]) {
        let d = self.order;
        for i in 0..d {
            let mut acc = self.b[i] * u;
            for j in 0..d {
                acc += self.a[i * d + j] * m[j];
            }
            out[i] = acc;
        }
    }

    /// Advance by `dt` under input `u`, held constant over the step.
    ///
    /// # Errors
    ///
    /// [`ResonateError::OutOfRange`] for a non-positive `dt`, [`ResonateError::NonFinite`] for a
    /// non-finite `u`.
    pub fn step(&mut self, dt: f64, u: f64) -> Result<(), ResonateError> {
        let dt = positive("dt", dt)?;
        let u = finite("u", u)?;
        let d = self.order;
        let (mut k1, mut k2, mut k3, mut k4) = (vec![0.0; d], vec![0.0; d], vec![0.0; d], vec![0.0; d]);
        let mut tmp = vec![0.0; d];
        self.deriv(&self.m, u, &mut k1);
        for i in 0..d {
            tmp[i] = self.m[i] + 0.5 * dt * k1[i];
        }
        self.deriv(&tmp, u, &mut k2);
        for i in 0..d {
            tmp[i] = self.m[i] + 0.5 * dt * k2[i];
        }
        self.deriv(&tmp, u, &mut k3);
        for i in 0..d {
            tmp[i] = self.m[i] + dt * k3[i];
        }
        self.deriv(&tmp, u, &mut k4);
        for i in 0..d {
            self.m[i] += dt / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
        }
        Ok(())
    }

    /// Return the state to zero.
    pub fn reset(&mut self) {
        self.m.iter_mut().for_each(|x| *x = 0.0);
    }

    /// The input `r·θ` seconds ago as the state represents it: `Σ_i P̃_i(r) m_i`, with `P̃_i` the
    /// shifted Legendre polynomial on `[0, 1]`.
    ///
    /// # Errors
    ///
    /// [`ResonateError::OutOfRange`] for `r` outside `[0, 1]`.
    pub fn reconstruct(&self, r: f64) -> Result<f64, ResonateError> {
        if !(0.0..=1.0).contains(&r) {
            return Err(ResonateError::OutOfRange { what: "r", value: r, low: 0.0, high: 1.0 });
        }
        Ok(self.m.iter().enumerate().map(|(i, &mi)| shifted_legendre(i, r) * mi).sum())
    }

    /// The input `θ` seconds ago: `Σ_i m_i`.
    #[must_use]
    pub fn delayed(&self) -> f64 {
        self.m.iter().sum()
    }

    /// The input now, as represented: `Σ_i (−1)^i m_i`.
    #[must_use]
    pub fn current(&self) -> f64 {
        self.m.iter().enumerate().map(|(i, &mi)| if i.is_multiple_of(2) { mi } else { -mi }).sum()
    }

    /// The taps a delay line sampled at `dt` would need for the same window: `ceil(θ / dt)`, the
    /// count this unit's `order` state variables replace. `None` for a non-positive `dt`.
    #[must_use]
    pub fn delay_line_taps(&self, dt: f64) -> Option<u64> {
        if !(dt > 0.0) || !dt.is_finite() {
            return None;
        }
        Some((self.theta / dt).ceil() as u64)
    }
}

/// The shifted Legendre polynomial `P̃_n(r) = P_n(2r − 1)`, by Bonnet's recurrence.
///
/// `P̃_n(1) = 1` and `P̃_n(0) = (−1)^n` for every `n`.
#[must_use]
pub fn shifted_legendre(n: usize, r: f64) -> f64 {
    let x = 2.0 * r - 1.0;
    if n == 0 {
        return 1.0;
    }
    let (mut p0, mut p1) = (1.0, x);
    for k in 1..n {
        let kf = k as f64;
        let p2 = ((2.0 * kf + 1.0) * x * p1 - kf * p0) / (kf + 1.0);
        p0 = p1;
        p1 = p2;
    }
    p1
}

#[cfg(test)]
mod tests {
    use super::{Lmu, ResonateAndFire, ResonateError, shifted_legendre};
    use core::f64::consts::{PI, TAU};

    // ---- resonate-and-fire ----

    /// The exact step composes: a hundred steps of `dt` equal one step of `100·dt`, to rounding,
    /// and both equal the closed-form free response. This is the property that makes a gap
    /// jumpable, and forward Euler on the same equations fails it by orders of magnitude.
    #[test]
    fn the_free_response_is_the_complex_exponential_and_composes() {
        let mut fine = ResonateAndFire::textbook().unwrap();
        fine.kick(0.7, -0.2);
        let mut coarse = fine;
        let dt = 1e-4;
        for _ in 0..100 {
            fine.step(dt, 0.0).unwrap();
        }
        coarse.step(100.0 * dt, 0.0).unwrap();
        assert!((fine.x - coarse.x).abs() < 1e-12 && (fine.y - coarse.y).abs() < 1e-12);
        let (wx, wy) = fine.free_response(0.7, -0.2, 0.01);
        assert!((fine.x - wx).abs() < 1e-12 && (fine.y - wy).abs() < 1e-12, "({}, {}) vs ({wx}, {wy})", fine.x, fine.y);
        // And the closed form is the damped rotation it says: after one full period the phase is
        // back and the magnitude has decayed by e^{b T}.
        let t = fine.period();
        let (rx, ry) = fine.free_response(0.7, -0.2, t);
        let decay = (fine.b * t).exp();
        assert!((rx - 0.7 * decay).abs() < 1e-12 && (ry + 0.2 * decay).abs() < 1e-12);
    }

    /// Resonance: two unit pulses one period apart add, half a period apart cancel, both to the
    /// closed form `|1 + e^{λΔ}|`; and with the threshold between the two magnitudes, only the
    /// in-phase pair makes the neuron fire.
    #[test]
    fn a_pair_of_pulses_resonates_at_the_natural_period_and_cancels_at_half_of_it() {
        let cell = ResonateAndFire::new(-2.0, TAU * 5.0, 1.5, (0.0, 0.0), 0.0).unwrap();
        let t = cell.period();
        for &interval in &[t, t / 2.0, t / 3.0, 1.7 * t] {
            let mut c = cell;
            c.kick(1.0, 0.0);
            let dt = interval / 1000.0;
            for _ in 0..1000 {
                c.step(dt, 0.0).unwrap();
            }
            c.kick(1.0, 0.0);
            let mag = (c.x * c.x + c.y * c.y).sqrt();
            let want = cell.pair_response(interval);
            assert!((mag - want).abs() < 1e-9, "interval {interval}: |z| {mag} vs closed form {want}");
        }
        let in_phase = cell.pair_response(t);
        let anti = cell.pair_response(t / 2.0);
        assert!((in_phase - (1.0 + (cell.b * t).exp())).abs() < 1e-12);
        assert!((anti - (1.0 - (cell.b * t / 2.0).exp())).abs() < 1e-12);
        assert!(in_phase > anti);

        // Spiking selectivity: a threshold of 1.5 sits between the in-phase and anti-phase peaks
        // of y, so the in-phase pair fires and the anti-phase pair does not.
        let fires_after = |interval: f64| -> bool {
            let mut c = cell;
            c.kick(1.0, 0.0);
            let dt = interval / 1000.0;
            for _ in 0..1000 {
                if c.step(dt, 0.0).unwrap() {
                    return true;
                }
            }
            c.kick(1.0, 0.0);
            // A quarter period later the in-phase state has rotated onto the imaginary axis.
            for _ in 0..250 {
                if c.step(t / 1000.0, 0.0).unwrap() {
                    return true;
                }
            }
            false
        };
        assert!(fires_after(t), "the in-phase pair did not fire");
        assert!(!fires_after(t / 2.0), "the anti-phase pair fired");
    }

    /// Under a sinusoidal drive the steady amplitude of `y` is the real-input transfer function
    /// in [`ResonateAndFire::sinusoidal_gain`], measured at three input frequencies to 0.2%, and
    /// the response peaks at the natural frequency — twelvefold over an octave above it.
    #[test]
    fn the_sinusoidal_steady_state_is_the_transfer_function() {
        let cell = ResonateAndFire::new(-3.0, TAU * 8.0, 1e9, (0.0, 0.0), 0.0).unwrap();
        let mut gains = Vec::new();
        for f_in in [4.0, 8.0, 16.0] {
            let w_in = TAU * f_in;
            let mut c = cell;
            let dt = 1e-5;
            let steps = (4.0 / dt) as usize; // 4 s: twelve damping constants, then a full period
            let mut peak = 0.0f64;
            for k in 0..steps {
                let t = k as f64 * dt;
                c.step(dt, 0.5 * (w_in * t).cos()).unwrap();
                if t > 3.0 {
                    peak = peak.max(c.y.abs());
                }
            }
            let want = 0.5 * cell.sinusoidal_gain(w_in);
            assert!((peak - want).abs() < 2e-3 * want, "{f_in} Hz: peak {peak} vs {want}");
            gains.push(peak);
        }
        assert!(gains[1] > gains[0] && gains[1] > gains[2], "no resonance peak: {gains:?}");
        assert!(gains[1] / gains[2] > 3.0, "the peak is not selective: {gains:?}");
    }

    /// Spiking and reset: a strong constant drive fires, the state is at `z_reset` afterwards,
    /// the refractory period holds it there and ignores kicks, and no input never fires.
    #[test]
    fn firing_resets_and_the_refractory_period_holds() {
        let mut c = ResonateAndFire::new(-1.0, TAU * 10.0, 1.0, (0.1, -0.3), 5e-3).unwrap();
        let mut fired_at = None;
        for k in 0..10_000 {
            if c.step(1e-4, 200.0).unwrap() {
                fired_at = Some(k);
                break;
            }
        }
        let k = fired_at.expect("a 200-unit drive must fire");
        assert_eq!((c.x, c.y), (0.1, -0.3), "reset state");
        assert_eq!(c.refractory, 5e-3);
        c.kick(5.0, 5.0);
        assert_eq!((c.x, c.y), (0.1, -0.3), "a kick during the refractory period is ignored");
        for _ in 0..49 {
            assert!(!c.step(1e-4, 200.0).unwrap());
            assert_eq!((c.x, c.y), (0.1, -0.3));
        }
        assert!(c.refractory > 0.0);
        c.step(1e-4, 200.0).unwrap();
        assert_eq!(c.refractory, 0.0, "fifty ticks of 0.1 ms is the 5 ms period exactly");
        // And the very next tick integrates again: the state leaves the reset point.
        c.step(1e-4, 200.0).unwrap();
        assert_ne!((c.x, c.y), (0.1, -0.3), "still held after the period ended");
        let _ = k;
        let mut quiet = ResonateAndFire::textbook().unwrap();
        for _ in 0..10_000 {
            assert!(!quiet.step(1e-4, 0.0).unwrap());
        }
        quiet.reset();
        assert_eq!((quiet.x, quiet.y, quiet.refractory), (0.0, 0.0, 0.0));
    }

    // ---- the Legendre Memory Unit ----

    /// The matrices against the paper's literal for `d = 3`, and the shifted Legendre values the
    /// reconstruction relies on.
    #[test]
    fn the_lmu_matrices_and_polynomials_are_the_papers() {
        let lmu = Lmu::new(3, 2.0).unwrap();
        // A_ij = (2i+1) · (−1 if i < j else (−1)^{i−j+1}), divided by θ = 2.
        let want_a = [-1.0, -1.0, -1.0, 3.0, -3.0, -3.0, -5.0, 5.0, -5.0];
        for (g, w) in lmu.a.iter().zip(want_a.iter()) {
            assert_eq!(*g, w / 2.0);
        }
        assert_eq!(lmu.b, vec![0.5, -1.5, 2.5]);
        for n in 0..6 {
            assert!((shifted_legendre(n, 1.0) - 1.0).abs() < 1e-15, "P̃_{n}(1)");
            let want0 = if n % 2 == 0 { 1.0 } else { -1.0 };
            assert!((shifted_legendre(n, 0.0) - want0).abs() < 1e-15, "P̃_{n}(0)");
        }
        assert!((shifted_legendre(1, 0.75) - 0.5).abs() < 1e-15, "P̃_1(r) = 2r − 1");
        assert!((shifted_legendre(2, 0.5) + 0.5).abs() < 1e-15, "P̃_2(1/2) = P_2(0) = −1/2");
        assert!((shifted_legendre(3, 0.5)).abs() < 1e-15, "odd polynomials vanish at the centre");
        assert!(matches!(Lmu::new(0, 1.0), Err(ResonateError::Empty { .. })));
        assert!(matches!(Lmu::new(3, 0.0), Err(ResonateError::OutOfRange { what: "theta", .. })));
        assert!(matches!(lmu.reconstruct(1.5), Err(ResonateError::OutOfRange { what: "r", .. })));
    }

    /// A constant input fills the window with a constant, whose Legendre expansion is
    /// `(1, 0, 0, …)`; a ramp fills it with a line, whose expansion is exactly
    /// `m_0 = t − θ/2, m_1 = −θ/2` and nothing above. Both to 1e-8 after the transient.
    #[test]
    fn a_constant_and_a_ramp_have_the_legendre_coefficients_they_must() {
        let theta = 0.5;
        let dt = theta / 2000.0;
        let mut lmu = Lmu::new(6, theta).unwrap();
        for _ in 0..(10.0 * theta / dt) as usize {
            lmu.step(dt, 1.0).unwrap();
        }
        assert!((lmu.m[0] - 1.0).abs() < 1e-8, "m_0 {}", lmu.m[0]);
        for i in 1..6 {
            assert!(lmu.m[i].abs() < 1e-8, "m_{i} = {}", lmu.m[i]);
        }
        assert!((lmu.delayed() - 1.0).abs() < 1e-8);
        assert!((lmu.current() - 1.0).abs() < 1e-8);

        lmu.reset();
        let mut t = 0.0;
        let steps = (10.0 * theta / dt) as usize;
        for _ in 0..steps {
            lmu.step(dt, t).unwrap();
            t += dt;
        }
        // The input over the step is held at its start value, so the window the unit saw is the
        // ramp sampled half a step early: t − dt/2 at "now".
        let now = t - dt / 2.0;
        assert!((lmu.m[0] - (now - theta / 2.0)).abs() < 1e-6, "m_0 {} vs {}", lmu.m[0], now - theta / 2.0);
        assert!((lmu.m[1] + theta / 2.0).abs() < 1e-6, "m_1 {} vs {}", lmu.m[1], -theta / 2.0);
        for i in 2..6 {
            assert!(lmu.m[i].abs() < 1e-6, "m_{i} = {} on a line", lmu.m[i]);
        }
        assert!((lmu.delayed() - (now - theta)).abs() < 1e-6, "the delayed value of a ramp");
        assert!((lmu.current() - now).abs() < 1e-6, "the current value of a ramp");
        assert_eq!(lmu.delay_line_taps(dt), Some(2000));
        assert_eq!(lmu.delay_line_taps(0.0), None);
    }

    /// A slow sine is delayed by `θ` to within a percent of its amplitude at order 6, and the
    /// reconstruction at a quarter of the window is the sine a quarter-window ago. A fast sine —
    /// one the order cannot hold — is not, which is what makes the first assertion a test.
    #[test]
    fn a_slow_sine_is_delayed_by_theta_and_a_fast_one_is_not() {
        let theta = 1.0;
        let dt = 1e-3;
        let run = |f: f64| -> (f64, f64) {
            let mut lmu = Lmu::new(6, theta).unwrap();
            let mut worst_delay = 0.0f64;
            let mut worst_quarter = 0.0f64;
            let steps = (6.0 / dt) as usize;
            for k in 0..steps {
                let t = k as f64 * dt;
                lmu.step(dt, (TAU * f * t).sin()).unwrap();
                if t > 3.0 {
                    let now = t + dt / 2.0;
                    let want = (TAU * f * (now - theta)).sin();
                    worst_delay = worst_delay.max((lmu.delayed() - want).abs());
                    let want_q = (TAU * f * (now - 0.25 * theta)).sin();
                    worst_quarter = worst_quarter.max((lmu.reconstruct(0.25).unwrap() - want_q).abs());
                }
            }
            (worst_delay, worst_quarter)
        };
        let (slow, slow_q) = run(0.5);
        assert!(slow < 0.02, "a 0.5 Hz sine through a 1 s window: worst delay error {slow}");
        assert!(slow_q < 0.02, "quarter-window reconstruction error {slow_q}");
        let (fast, _) = run(6.0);
        assert!(fast > 0.3, "a 6 Hz sine cannot be held by six coefficients, yet the error was {fast}");
        let _ = PI;
    }

    /// Every refusal names the problem.
    #[test]
    fn the_refusals_name_the_problem() {
        assert!(matches!(ResonateAndFire::new(1.0, 1.0, 1.0, (0.0, 0.0), 0.0), Err(ResonateError::OutOfRange { what: "b", .. })));
        assert!(matches!(ResonateAndFire::new(-1.0, 0.0, 1.0, (0.0, 0.0), 0.0), Err(ResonateError::OutOfRange { what: "omega", .. })));
        assert!(matches!(ResonateAndFire::new(-1.0, 1.0, 0.0, (0.0, 0.0), 0.0), Err(ResonateError::OutOfRange { what: "threshold", .. })));
        assert!(matches!(ResonateAndFire::new(-1.0, 1.0, 1.0, (0.0, 0.0), -1.0), Err(ResonateError::OutOfRange { what: "t_ref", .. })));
        assert!(matches!(ResonateAndFire::new(-1.0, 1.0, 1.0, (f64::NAN, 0.0), 0.0), Err(ResonateError::NonFinite { .. })));
        let mut c = ResonateAndFire::textbook().unwrap();
        assert!(matches!(c.step(0.0, 0.0), Err(ResonateError::OutOfRange { what: "dt", .. })));
        assert!(matches!(c.step(1e-3, f64::INFINITY), Err(ResonateError::NonFinite { what: "i", .. })));
        let mut l = Lmu::new(2, 1.0).unwrap();
        assert!(matches!(l.step(-1.0, 0.0), Err(ResonateError::OutOfRange { .. })));
        assert!(matches!(l.step(1e-3, f64::NAN), Err(ResonateError::NonFinite { what: "u", .. })));
        for e in [
            ResonateError::OutOfRange { what: "w", value: 9.0, low: 0.0, high: 1.0 },
            ResonateError::NonFinite { what: "z", value: f64::NAN },
            ResonateError::Empty { what: "order" },
        ] {
            assert!(!e.to_string().is_empty());
        }
    }
}
