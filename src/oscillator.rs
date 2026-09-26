//! Coupled phase oscillators: synchronisation with a threshold you can write down, locomotion as
//! a travelling wave of phase lags, and combinatorial search as the relaxation of an oscillator
//! network — each checked against its closed form.
//!
//! # What the mechanism is
//!
//! A limit-cycle oscillator that is weakly coupled to others is described, to first order, by its
//! **phase** alone (Kuramoto, *Chemical Oscillations, Waves, and Turbulence*, Springer, 1984). The
//! population model `dθ_i/dt = ω_i + (K/N) Σ_j sin(θ_j − θ_i)` is the one such model with a complete
//! theory: the order parameter `r e^{iψ} = (1/N) Σ_j e^{iθ_j}` turns the `N²` pairwise sum into
//! the mean-field term `K r sin(ψ − θ_i)`, which is an identity and not an approximation
//! ([`Kuramoto::velocity`] against [`Kuramoto::velocity_pairwise`]).
//!
//! Three uses of the same equation are here:
//!
//! - **Synchronisation** ([`Kuramoto`]). Two oscillators lock iff their detuning is inside the
//!   coupling (Adler, *A study of locking phenomena in oscillators*, Proceedings of the IRE
//!   34(6):351–357, 1946), and a population with Lorentzian frequencies synchronises above
//!   `K_c = 2γ` with `r = √(1 − K_c/K)` (Kuramoto 1984; Ott and Antonsen, *Low dimensional
//!   behavior of large systems of globally coupled oscillators*, Chaos 18:037113, 2008).
//! - **Locomotion** ([`CpgChain`]). A chain of oscillators with nearest-neighbour coupling and a
//!   built-in phase lag locks into a travelling wave — the lamprey's swimming pattern and the
//!   controller of the salamander robot (Ijspeert, Crespi, Ryczko and Cabelguen, *From swimming to
//!   walking with a salamander robot driven by a spinal cord model*, Science 315(5817):1416–1420,
//!   2007), including its critically damped amplitude equation ([`Amplitude`]).
//! - **Search** ([`Oim`]). With a second-harmonic injection `−K_s sin 2θ` the phases binarise to
//!   `{0, π}` and the network's Lyapunov function becomes an Ising Hamiltonian (Wang and
//!   Roychowdhury, *OIM: oscillator-based Ising machines for solving combinatorial optimisation
//!   problems*, UCNC 2019, LNCS 11493:232–256): the oscillators relax toward a maximum cut.
//!
//! # Why it is in a neuromorphic crate
//!
//! Oscillator networks are a physical computing substrate in their own right — ring-oscillator
//! CMOS, spin-torque and vanadium-dioxide oscillators have all been built as Ising machines — and
//! the central pattern generator is the oldest neuromorphic controller there is: a spinal circuit
//! that turns a tonic drive into a rhythm, which is how the crate's `control::Matsuoka` cell is
//! used. This module is the phase reduction of those circuits, where the theory is exact.
//!
//! # The closed forms this module is checked against
//!
//! - **Mean field is an identity.** The `O(N)` velocity equals the `O(N²)` pairwise sum to
//!   rounding.
//! - **Adler.** Two oscillators with detuning `Δω` and coupling `K` obey `dφ/dt = Δω − K sin φ`:
//!   they lock at `φ* = asin(Δω/K)` iff `|Δω| ≤ K` and both then turn at the MEAN frequency;
//!   outside the lock range the phase slips with period exactly `2π/√(Δω² − K²)`.
//! - **Identical oscillators** close on each other at rate `K`: a small spread decays as
//!   `e^{−Kt}`.
//! - **Lorentzian population.** `K_c = 2γ` and `r∞ = √(1 − 2γ/K)`; the Ott–Antonsen reduction
//!   `dr/dt = −γr + (K/2) r (1 − r²)` is logistic in `r²` and [`ott_antonsen_r`] is its solution,
//!   checked against the differential equation it claims to solve. A finite population of `N`
//!   quantile-sampled oscillators is held to that `r∞` within the finite-size scale `2/√N`.
//! - **OIM.** The velocity is minus the gradient of [`Oim::energy`] (checked by central
//!   differences); at binary phases the energy is `K(W − 2·cut) − K_s N/2` exactly; explicit Euler
//!   at [`Oim::stable_dt`] never raises the energy (the descent lemma, with the Gershgorin bound on
//!   the Hessian as the Lipschitz constant); and on small graphs the best of a few restarts equals
//!   the brute-force maximum cut of [`crate::optimise::Qubo::max_cut`].
//! - **CPG chain.** The locked state has every neighbouring lag equal to the built-in lag and
//!   every oscillator turning at exactly `ω`; the slowest perturbation mode of an `n`-chain decays
//!   at `w (2 − 2cos(π/n))`, the smallest non-zero eigenvalue of the path Laplacian.
//! - **Amplitude.** `r̈ = a (a/4 (R − r) − ṙ)` is critically damped:
//!   `r(t) = R − (R − r₀)(1 + at/2) e^{−at/2}` from rest, and [`Amplitude::step`] is exact for a
//!   constant target whatever the step.
//!
//! # What this module has NOT reproduced
//!
//! - Any hardware oscillator's waveform, noise or coupling non-ideality. The phase model is the
//!   weak-coupling limit and says nothing about amplitude death or harmonic content.
//! - The benchmark results of the OIM paper (all 54 G-set MAX-CUT graphs, G1–G54, of 800 to 3000
//!   vertices). The check here is against brute force, which stops at
//!   [`crate::optimise::MAX_BRUTE_FORCE`] vertices. ⚠ CORRECTED against the paper: this line used
//!   to say "G-set graphs of 800–2000 vertices", which drops the largest size the paper ran. Its
//!   Sec. 4.2 says "we have run simulations on all the problems in a widely used set of MAX-CUT
//!   benchmarks known as the G-set" and "Problem sizes range from 800 to 3000"; footnote 8 gives
//!   "G48∼50 are of size 3000", and Table 1 has rows for G48, G49 and G50 (Wang and Roychowdhury,
//!   *OIM: oscillator-based Ising machines for solving combinatorial optimisation problems*, UCNC
//!   2019, LNCS 11493:232–256, doi:10.1007/978-3-030-19311-9_19, arXiv:1903.07163).
//! - The salamander's limb oscillators and its swim–walk transition by drive saturation; the chain
//!   here is the body axis alone.

use core::f64::consts::{PI, TAU};
use core::fmt;

use crate::rng::Rng;

/// The most steps one `run` call will take; a request past it is refused rather than started.
pub const MAX_STEPS: u64 = 100_000_000;

/// What went wrong, named rather than guessed around.
#[derive(Debug, Clone, PartialEq)]
pub enum OscillatorError {
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
    /// An index past the network.
    Index {
        /// Which index.
        what: &'static str,
        /// The value.
        index: usize,
        /// The count it had to be below.
        count: usize,
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

impl fmt::Display for OscillatorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty { what } => write!(f, "{what} is empty"),
            Self::Dimension { what, got, want } => write!(f, "{what} has length {got}, expected {want}"),
            Self::Index { what, index, count } => write!(f, "{what} {index} is past the {count} available"),
            Self::OutOfRange { what, value, low, high } => {
                write!(f, "{what} = {value} is outside [{low}, {high}]")
            }
            Self::NonFinite { what, index } => write!(f, "{what} is not finite at {index}"),
        }
    }
}

impl std::error::Error for OscillatorError {}

fn finite_all(what: &'static str, v: &[f64]) -> Result<(), OscillatorError> {
    if let Some(i) = v.iter().position(|x| !x.is_finite()) {
        return Err(OscillatorError::NonFinite { what, index: i });
    }
    Ok(())
}

fn positive(what: &'static str, v: f64) -> Result<f64, OscillatorError> {
    if v.is_finite() && v > 0.0 {
        Ok(v)
    } else {
        Err(OscillatorError::OutOfRange { what, value: v, low: f64::MIN_POSITIVE, high: f64::INFINITY })
    }
}

fn non_negative(what: &'static str, v: f64) -> Result<f64, OscillatorError> {
    if v.is_finite() && v >= 0.0 {
        Ok(v)
    } else {
        Err(OscillatorError::OutOfRange { what, value: v, low: 0.0, high: f64::INFINITY })
    }
}

fn step_count(steps: u64) -> Result<u64, OscillatorError> {
    if steps > MAX_STEPS {
        return Err(OscillatorError::OutOfRange { what: "steps", value: steps as f64, low: 0.0, high: MAX_STEPS as f64 });
    }
    Ok(steps)
}

/// One classical Runge–Kutta step of `dθ/dt = f(θ)`, in place.
fn rk4(theta: &mut [f64], dt: f64, f: impl Fn(&[f64], &mut [f64])) {
    let n = theta.len();
    let (mut k1, mut k2, mut k3, mut k4) = (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    let mut probe = vec![0.0; n];
    f(theta, &mut k1);
    for i in 0..n {
        probe[i] = theta[i] + 0.5 * dt * k1[i];
    }
    f(&probe, &mut k2);
    for i in 0..n {
        probe[i] = theta[i] + 0.5 * dt * k2[i];
    }
    f(&probe, &mut k3);
    for i in 0..n {
        probe[i] = theta[i] + dt * k3[i];
    }
    f(&probe, &mut k4);
    for i in 0..n {
        theta[i] += dt / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
    }
}

/// Wrap an angle into `(−π, π]`, radians.
#[must_use]
pub fn wrap_pi(x: f64) -> f64 {
    let r = x.rem_euclid(TAU);
    if r > PI { r - TAU } else { r }
}

// ---------------------------------------------------------------------------------------------
// Adler's equation
// ---------------------------------------------------------------------------------------------

/// The locked phase difference `φ* = asin(Δω / K)` of `dφ/dt = Δω − K sin φ`, radians, or `None`
/// outside the lock range `|Δω| ≤ K` (and for a non-positive or non-finite `K`).
///
/// The stable branch is the one with `cos φ* ≥ 0`, which is what `asin` returns.
#[must_use]
pub fn locked_phase(delta_omega: f64, k: f64) -> Option<f64> {
    if !(k > 0.0) || !k.is_finite() || !delta_omega.is_finite() || delta_omega.abs() > k {
        return None;
    }
    Some((delta_omega / k).asin())
}

/// The phase-slip angular frequency `√(Δω² − K²)` of `dφ/dt = Δω − K sin φ`, rad/s: the phase
/// difference advances by `2π` every `2π` over this. `Some(0.0)` inside the lock range, `None`
/// for a negative or non-finite argument.
#[must_use]
pub fn beat_frequency(delta_omega: f64, k: f64) -> Option<f64> {
    if !(k >= 0.0) || !k.is_finite() || !delta_omega.is_finite() {
        return None;
    }
    let d = delta_omega * delta_omega - k * k;
    Some(if d > 0.0 { d.sqrt() } else { 0.0 })
}

// ---------------------------------------------------------------------------------------------
// The Kuramoto model
// ---------------------------------------------------------------------------------------------

/// A globally coupled population `dθ_i/dt = ω_i + (K/N) Σ_j sin(θ_j − θ_i)`.
#[derive(Debug, Clone, PartialEq)]
pub struct Kuramoto {
    /// Natural frequencies, rad/s.
    pub omega: Vec<f64>,
    /// Coupling `K`, rad/s. The sum is divided by `N`, so `K` is the locking range of a pair.
    pub k: f64,
    /// Phases, radians, NOT wrapped — a difference of `2π` is a slip that happened.
    pub theta: Vec<f64>,
}

impl Kuramoto {
    /// Build from frequencies, coupling and initial phases.
    ///
    /// # Errors
    ///
    /// [`OscillatorError::Empty`] for no oscillators, [`OscillatorError::Dimension`] when the two
    /// arrays disagree, [`OscillatorError::NonFinite`] for a bad entry and
    /// [`OscillatorError::OutOfRange`] for a negative or non-finite `K`.
    pub fn new(omega: Vec<f64>, k: f64, theta: Vec<f64>) -> Result<Self, OscillatorError> {
        if omega.is_empty() {
            return Err(OscillatorError::Empty { what: "oscillators" });
        }
        if theta.len() != omega.len() {
            return Err(OscillatorError::Dimension { what: "theta", got: theta.len(), want: omega.len() });
        }
        finite_all("omega", &omega)?;
        finite_all("theta", &theta)?;
        let k = non_negative("k", k)?;
        Ok(Self { omega, k, theta })
    }

    /// The order parameter `(r, ψ)`: coherence in `[0, 1]` and mean phase in `(−π, π]`.
    #[must_use]
    pub fn order_parameter(&self) -> (f64, f64) {
        let (c, s) = mean_field(&self.theta);
        (c.hypot(s), s.atan2(c))
    }

    fn derivative(&self, theta: &[f64], out: &mut [f64]) {
        let (c, s) = mean_field(theta);
        for i in 0..theta.len() {
            out[i] = self.omega[i] + self.k * (s * theta[i].cos() - c * theta[i].sin());
        }
    }

    /// The instantaneous phase velocities by the mean field, rad/s — `N` sine–cosine pairs.
    #[must_use]
    pub fn velocity(&self) -> Vec<f64> {
        let mut out = vec![0.0; self.theta.len()];
        self.derivative(&self.theta, &mut out);
        out
    }

    /// The same velocities by the literal pairwise sum — `N²` sines. The reference the mean field
    /// is checked against; nothing else calls it.
    #[must_use]
    pub fn velocity_pairwise(&self) -> Vec<f64> {
        let n = self.theta.len();
        (0..n)
            .map(|i| {
                let sum: f64 = (0..n).map(|j| (self.theta[j] - self.theta[i]).sin()).sum();
                self.omega[i] + self.k / n as f64 * sum
            })
            .collect()
    }

    /// Advance by `dt` seconds with one classical Runge–Kutta step.
    ///
    /// # Errors
    ///
    /// [`OscillatorError::OutOfRange`] for a non-positive or non-finite `dt`.
    pub fn step(&mut self, dt: f64) -> Result<(), OscillatorError> {
        let dt = positive("dt", dt)?;
        let mut theta = core::mem::take(&mut self.theta);
        rk4(&mut theta, dt, |at, out| self.derivative(at, out));
        self.theta = theta;
        Ok(())
    }

    /// Take `steps` steps of `dt`.
    ///
    /// # Errors
    ///
    /// As [`Kuramoto::step`], plus [`OscillatorError::OutOfRange`] for more than [`MAX_STEPS`].
    pub fn run(&mut self, dt: f64, steps: u64) -> Result<(), OscillatorError> {
        for _ in 0..step_count(steps)? {
            self.step(dt)?;
        }
        Ok(())
    }
}

fn mean_field(theta: &[f64]) -> (f64, f64) {
    let n = theta.len() as f64;
    let (mut c, mut s) = (0.0, 0.0);
    for &t in theta {
        c += t.cos();
        s += t.sin();
    }
    (c / n, s / n)
}

/// `n` frequencies at the quantiles of a Lorentzian of centre `centre` and half-width `gamma`,
/// rad/s: `ω_i = centre + γ tan(π (i + ½)/n − π/2)`. Deterministic, symmetric about the centre,
/// and with its largest member at about `2γn/π` — which is what bounds a usable time step.
///
/// # Errors
///
/// [`OscillatorError::Empty`] for `n = 0`, [`OscillatorError::OutOfRange`] for a non-positive
/// `gamma` and [`OscillatorError::NonFinite`] for a non-finite centre.
pub fn lorentzian_quantiles(n: usize, centre: f64, gamma: f64) -> Result<Vec<f64>, OscillatorError> {
    if n == 0 {
        return Err(OscillatorError::Empty { what: "oscillators" });
    }
    if !centre.is_finite() {
        return Err(OscillatorError::NonFinite { what: "centre", index: 0 });
    }
    let gamma = positive("gamma", gamma)?;
    Ok((0..n).map(|i| centre + gamma * (PI * (i as f64 + 0.5) / n as f64 - 0.5 * PI).tan()).collect())
}

/// The critical coupling `K_c = 2γ` of a Lorentzian population of half-width `gamma`, rad/s.
#[must_use]
pub fn lorentzian_critical_coupling(gamma: f64) -> f64 {
    2.0 * gamma
}

/// The coherence `r(t)` of an infinite Lorentzian population on the Ott–Antonsen manifold,
/// starting from `r0`: the solution of `dr/dt = −γr + (K/2) r (1 − r²)`, which is logistic in
/// `u = r²` with rate `K − 2γ` and carrying capacity `1 − 2γ/K`.
///
/// `None` for `r0` outside `[0, 1]`, a negative `t`, a non-positive `K` or `γ`, or anything
/// non-finite.
#[must_use]
pub fn ott_antonsen_r(r0: f64, k: f64, gamma: f64, t: f64) -> Option<f64> {
    let ok = (0.0..=1.0).contains(&r0) && k > 0.0 && k.is_finite() && gamma > 0.0 && gamma.is_finite() && t >= 0.0 && t.is_finite();
    if !ok {
        return None;
    }
    if r0 == 0.0 {
        return Some(0.0);
    }
    let u0 = r0 * r0;
    let rate = k - 2.0 * gamma;
    let u = if rate == 0.0 {
        u0 / (1.0 + k * u0 * t)
    } else {
        let cap = rate / k;
        cap / (1.0 + (cap / u0 - 1.0) * (-rate * t).exp())
    };
    Some(u.sqrt())
}

// ---------------------------------------------------------------------------------------------
// A central pattern generator: the chain and its amplitude
// ---------------------------------------------------------------------------------------------

/// The critically damped amplitude equation `r̈ = a (a/4 (R − r) − ṙ)` of the salamander model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Amplitude {
    /// Convergence constant `a`, 1/s; the amplitude closes on its target at rate `a/2`.
    pub a: f64,
    /// Amplitude `r`, in the units of the target.
    pub r: f64,
    /// Its rate of change, units per second.
    pub r_dot: f64,
}

impl Amplitude {
    /// Build at rest at `r`.
    ///
    /// # Errors
    ///
    /// [`OscillatorError::OutOfRange`] for a non-positive `a`, [`OscillatorError::NonFinite`] for a
    /// non-finite `r`.
    pub fn new(a: f64, r: f64) -> Result<Self, OscillatorError> {
        let a = positive("a", a)?;
        if !r.is_finite() {
            return Err(OscillatorError::NonFinite { what: "r", index: 0 });
        }
        Ok(Self { a, r, r_dot: 0.0 })
    }

    /// Advance by `dt` seconds toward a constant `target` — EXACT for a constant target, because
    /// the equation is linear: with `e = r − R` and `h = a/2`,
    /// `e(t) = (e₀ + (ė₀ + h e₀) t) e^{−ht}`.
    ///
    /// # Errors
    ///
    /// [`OscillatorError::OutOfRange`] for a non-positive `dt`, [`OscillatorError::NonFinite`] for a
    /// non-finite target.
    pub fn step(&mut self, dt: f64, target: f64) -> Result<f64, OscillatorError> {
        let dt = positive("dt", dt)?;
        if !target.is_finite() {
            return Err(OscillatorError::NonFinite { what: "target", index: 0 });
        }
        let h = 0.5 * self.a;
        let e0 = self.r - target;
        let slope = self.r_dot + h * e0;
        let decay = (-h * dt).exp();
        let e = (e0 + slope * dt) * decay;
        self.r_dot = (slope - h * (e0 + slope * dt)) * decay;
        self.r = target + e;
        Ok(self.r)
    }
}

/// A chain of `n` phase oscillators with nearest-neighbour coupling and a built-in lag:
/// `dθ_i/dt = ω + w sin(θ_{i−1} − θ_i − φ) + w sin(θ_{i+1} − θ_i + φ)`. Locked, each segment
/// trails the one before it by `φ` and the body carries `(n − 1) φ / 2π` wavelengths.
#[derive(Debug, Clone, PartialEq)]
pub struct CpgChain {
    /// Common frequency `ω`, rad/s.
    pub omega: f64,
    /// Coupling weight `w`, rad/s.
    pub w: f64,
    /// Built-in lag `φ` between neighbours, radians; positive sends the wave from segment `0`
    /// toward segment `n − 1`.
    pub lag: f64,
    /// Phases, radians, not wrapped.
    pub theta: Vec<f64>,
}

impl CpgChain {
    /// Build from initial phases.
    ///
    /// # Errors
    ///
    /// [`OscillatorError::Empty`] for fewer than two segments, [`OscillatorError::NonFinite`] for a
    /// bad frequency, lag or phase, [`OscillatorError::OutOfRange`] for a non-positive weight.
    pub fn new(omega: f64, w: f64, lag: f64, theta: Vec<f64>) -> Result<Self, OscillatorError> {
        if theta.len() < 2 {
            return Err(OscillatorError::Empty { what: "segments (needs two)" });
        }
        if !omega.is_finite() {
            return Err(OscillatorError::NonFinite { what: "omega", index: 0 });
        }
        if !lag.is_finite() {
            return Err(OscillatorError::NonFinite { what: "lag", index: 0 });
        }
        finite_all("theta", &theta)?;
        let w = positive("w", w)?;
        Ok(Self { omega, w, lag, theta })
    }

    fn derivative(&self, theta: &[f64], out: &mut [f64]) {
        let n = theta.len();
        for i in 0..n {
            let mut v = self.omega;
            if i > 0 {
                v += self.w * (theta[i - 1] - theta[i] - self.lag).sin();
            }
            if i + 1 < n {
                v += self.w * (theta[i + 1] - theta[i] + self.lag).sin();
            }
            out[i] = v;
        }
    }

    /// The instantaneous phase velocities, rad/s.
    #[must_use]
    pub fn velocity(&self) -> Vec<f64> {
        let mut out = vec![0.0; self.theta.len()];
        self.derivative(&self.theta, &mut out);
        out
    }

    /// The lag errors `θ_{i−1} − θ_i − φ` for `i = 1..n`, wrapped into `(−π, π]`: all zero when
    /// the wave is locked.
    #[must_use]
    pub fn lag_errors(&self) -> Vec<f64> {
        self.theta.windows(2).map(|p| wrap_pi(p[0] - p[1] - self.lag)).collect()
    }

    /// The decay rate of the slowest perturbation of the locked wave, 1/s:
    /// `w (2 − 2cos(π/n))`, the smallest non-zero eigenvalue of the path Laplacian times `w`.
    #[must_use]
    pub fn slowest_rate(&self) -> f64 {
        self.w * (2.0 - 2.0 * (PI / self.theta.len() as f64).cos())
    }

    /// Advance by `dt` seconds with one classical Runge–Kutta step.
    ///
    /// # Errors
    ///
    /// [`OscillatorError::OutOfRange`] for a non-positive or non-finite `dt`.
    pub fn step(&mut self, dt: f64) -> Result<(), OscillatorError> {
        let dt = positive("dt", dt)?;
        let mut theta = core::mem::take(&mut self.theta);
        rk4(&mut theta, dt, |at, out| self.derivative(at, out));
        self.theta = theta;
        Ok(())
    }

    /// Take `steps` steps of `dt`.
    ///
    /// # Errors
    ///
    /// As [`CpgChain::step`], plus [`OscillatorError::OutOfRange`] for more than [`MAX_STEPS`].
    pub fn run(&mut self, dt: f64, steps: u64) -> Result<(), OscillatorError> {
        for _ in 0..step_count(steps)? {
            self.step(dt)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------------------------
// The oscillator Ising machine
// ---------------------------------------------------------------------------------------------

/// An oscillator Ising machine for MAXIMUM CUT: `dθ_i/dt = K Σ_j w_ij sin(θ_i − θ_j) − K_s sin 2θ_i`,
/// the gradient flow of [`Oim::energy`]. An edge pushes its two ends toward antiphase; the second
/// harmonic pushes every phase toward `0` or `π`.
#[derive(Debug, Clone, PartialEq)]
pub struct Oim {
    /// Vertices.
    pub n: usize,
    /// Symmetric edge weights, row-major `n × n`, zero diagonal, non-negative.
    pub w: Vec<f64>,
    /// Coupling gain `K`, rad/s per unit weight.
    pub k: f64,
    /// Second-harmonic injection `K_s`, rad/s.
    pub k_s: f64,
    /// Phases, radians.
    pub theta: Vec<f64>,
}

impl Oim {
    /// Build from an edge list `(u, v, weight)`; a repeated edge adds its weight. Phases start at
    /// zero — call [`Oim::randomise`] before relaxing, because all-zero is an equilibrium.
    ///
    /// # Errors
    ///
    /// [`OscillatorError::Empty`] for no vertices, [`OscillatorError::Index`] for an endpoint past
    /// the graph, [`OscillatorError::OutOfRange`] for a self-loop, a negative weight or a
    /// non-positive `K`, a negative `K_s`.
    pub fn max_cut(n: usize, edges: &[(usize, usize, f64)], k: f64, k_s: f64) -> Result<Self, OscillatorError> {
        if n == 0 {
            return Err(OscillatorError::Empty { what: "vertices" });
        }
        let k = positive("k", k)?;
        let k_s = non_negative("k_s", k_s)?;
        let mut w = vec![0.0; n * n];
        for &(u, v, weight) in edges {
            if u >= n {
                return Err(OscillatorError::Index { what: "vertex", index: u, count: n });
            }
            if v >= n {
                return Err(OscillatorError::Index { what: "vertex", index: v, count: n });
            }
            if u == v {
                return Err(OscillatorError::OutOfRange { what: "self-loop at vertex", value: u as f64, low: 0.0, high: 0.0 });
            }
            let weight = non_negative("weight", weight)?;
            w[u * n + v] += weight;
            w[v * n + u] += weight;
        }
        Ok(Self { n, w, k, k_s, theta: vec![0.0; n] })
    }

    /// Draw every phase uniformly on the circle.
    pub fn randomise(&mut self, rng: &mut Rng) {
        for t in &mut self.theta {
            *t = TAU * rng.next_f64();
        }
    }

    /// The Lyapunov function `E = K Σ_{i<j} w_ij cos(θ_i − θ_j) − (K_s/2) Σ_i cos 2θ_i`, rad/s.
    /// At binary phases it is `K (W − 2·cut) − K_s N/2`, with `W` the total weight.
    #[must_use]
    pub fn energy(&self) -> f64 {
        let n = self.n;
        let mut e = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                let wij = self.w[i * n + j];
                if wij != 0.0 {
                    e += self.k * wij * (self.theta[i] - self.theta[j]).cos();
                }
            }
            e -= 0.5 * self.k_s * (2.0 * self.theta[i]).cos();
        }
        e
    }

    /// The phase velocities `−∂E/∂θ_i`, rad/s.
    #[must_use]
    pub fn velocity(&self) -> Vec<f64> {
        let n = self.n;
        (0..n)
            .map(|i| {
                let mut v = -self.k_s * (2.0 * self.theta[i]).sin();
                for j in 0..n {
                    let wij = self.w[i * n + j];
                    if wij != 0.0 {
                        v += self.k * wij * (self.theta[i] - self.theta[j]).sin();
                    }
                }
                v
            })
            .collect()
    }

    /// A step at which explicit Euler cannot raise the energy: `1/L`, with
    /// `L = max_i (2K Σ_j w_ij + 2K_s)` the Gershgorin bound on the Hessian of the energy. The
    /// descent lemma gives `E(θ − h∇E) ≤ E − h (1 − Lh/2) |∇E|²`, which is a decrease for any
    /// `h < 2/L`.
    #[must_use]
    pub fn stable_dt(&self) -> f64 {
        let n = self.n;
        let worst = (0..n).map(|i| self.w[i * n..(i + 1) * n].iter().sum::<f64>()).fold(0.0, f64::max);
        1.0 / (2.0 * self.k * worst + 2.0 * self.k_s).max(f64::MIN_POSITIVE)
    }

    /// One explicit Euler step of `dt` seconds down the energy; returns the energy afterwards.
    ///
    /// # Errors
    ///
    /// [`OscillatorError::OutOfRange`] for a non-positive or non-finite `dt`.
    pub fn step(&mut self, dt: f64) -> Result<f64, OscillatorError> {
        let dt = positive("dt", dt)?;
        let v = self.velocity();
        for (t, vi) in self.theta.iter_mut().zip(&v) {
            *t += dt * vi;
        }
        Ok(self.energy())
    }

    /// Relax for `steps` Euler steps of `dt`; returns the LARGEST single-step energy increase seen
    /// (non-positive when the run descended throughout).
    ///
    /// # Errors
    ///
    /// As [`Oim::step`], plus [`OscillatorError::OutOfRange`] for more than [`MAX_STEPS`].
    pub fn relax(&mut self, dt: f64, steps: u64) -> Result<f64, OscillatorError> {
        let mut last = self.energy();
        let mut worst = f64::NEG_INFINITY;
        for _ in 0..step_count(steps)? {
            let e = self.step(dt)?;
            worst = worst.max(e - last);
            last = e;
        }
        Ok(worst)
    }

    /// The spin read from each phase: `+1` when `cos θ ≥ 0`, else `−1`.
    #[must_use]
    pub fn spins(&self) -> Vec<i8> {
        self.theta.iter().map(|t| if t.cos() >= 0.0 { 1 } else { -1 }).collect()
    }

    /// The weight of the cut the spins define.
    #[must_use]
    pub fn cut(&self) -> f64 {
        let s = self.spins();
        let n = self.n;
        let mut cut = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                if s[i] != s[j] {
                    cut += self.w[i * n + j];
                }
            }
        }
        cut
    }

    /// How binary the phases are: the smallest `|cos θ_i|`, which is `1` when every oscillator
    /// sits at `0` or `π`.
    #[must_use]
    pub fn binarisation(&self) -> f64 {
        self.theta.iter().map(|t| t.cos().abs()).fold(f64::INFINITY, f64::min)
    }

    /// Snap every phase to the spin it reads as: `0` for `+1`, `π` for `−1`.
    pub fn snap(&mut self) {
        for t in &mut self.theta {
            *t = if t.cos() >= 0.0 { 0.0 } else { PI };
        }
    }
}

/// The best cut over `restarts` relaxations from random phases, each of `steps` Euler steps at
/// [`Oim::stable_dt`]: `(cut, spins)`.
///
/// # Errors
///
/// [`OscillatorError::Empty`] for zero restarts, and as [`Oim::relax`].
pub fn best_cut(oim: &mut Oim, restarts: usize, steps: u64, rng: &mut Rng) -> Result<(f64, Vec<i8>), OscillatorError> {
    if restarts == 0 {
        return Err(OscillatorError::Empty { what: "restarts" });
    }
    let dt = oim.stable_dt();
    let mut best = (f64::NEG_INFINITY, Vec::new());
    for _ in 0..restarts {
        oim.randomise(rng);
        oim.relax(dt, steps)?;
        let cut = oim.cut();
        if cut > best.0 {
            best = (cut, oim.spins());
        }
    }
    Ok(best)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimise::Qubo;

    #[test]
    fn the_mean_field_is_the_pairwise_sum() {
        let mut rng = Rng::new(5);
        let n = 50;
        let omega: Vec<f64> = (0..n).map(|_| 4.0 * rng.next_f64() - 2.0).collect();
        let theta: Vec<f64> = (0..n).map(|_| TAU * rng.next_f64()).collect();
        let k = Kuramoto::new(omega, 1.7, theta).unwrap();
        let (fast, slow) = (k.velocity(), k.velocity_pairwise());
        let worst = fast.iter().zip(&slow).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        // N terms of size ≤ 1, each good to an ulp, times K/N.
        assert!(worst < 1.7 * 50.0 * f64::EPSILON, "mean field and pairwise differ by {worst}");
        // And the comparison is not between two zeros.
        assert!(slow.iter().zip(&k.omega).any(|(v, w)| (v - w).abs() > 0.05));
    }

    #[test]
    fn two_oscillators_lock_where_adler_says() {
        let mut pair = Kuramoto::new(vec![1.0, 1.6], 1.0, vec![0.3, 2.0]).unwrap();
        pair.run(1e-2, 6000).unwrap();
        let phi = wrap_pi(pair.theta[1] - pair.theta[0]);
        let want = locked_phase(0.6, 1.0).unwrap();
        assert!((want - 0.6f64.asin()).abs() < 1e-15);
        // The approach rate is K cos φ* = 0.8 1/s, so 60 s leaves e^{−48} of the initial error.
        assert!((phi - want).abs() < 1e-9, "locked at {phi}, Adler says {want}");
        // Locked, both turn at the MEAN frequency, not at either natural one.
        let v = pair.velocity();
        assert!((v[0] - 1.3).abs() < 1e-9 && (v[1] - 1.3).abs() < 1e-9, "{v:?}");
        // The lock range closes at |Δω| = K, where the phase is a quarter turn.
        assert_eq!(locked_phase(1.0, 1.0), Some(core::f64::consts::FRAC_PI_2));
        assert_eq!(locked_phase(-1.0, 1.0), Some(-core::f64::consts::FRAC_PI_2));
        assert_eq!(locked_phase(1.0001, 1.0), None);
        assert_eq!(locked_phase(0.5, 0.0), None);
        assert_eq!(locked_phase(0.5, -1.0), None);
        assert_eq!(locked_phase(f64::NAN, 1.0), None);
        assert_eq!(beat_frequency(0.5, 1.0), Some(0.0));
        assert_eq!(beat_frequency(5.0, 3.0), Some(4.0));
        assert_eq!(beat_frequency(5.0, -3.0), None);
    }

    #[test]
    fn outside_the_lock_range_the_phase_slips_at_the_adler_beat() {
        // Δω = 1.0, K = 0.6: the slip period is 2π/√(1 − 0.36) = 2π/0.8.
        let mut pair = Kuramoto::new(vec![0.0, 1.0], 0.6, vec![0.0, 0.0]).unwrap();
        let dt = 1e-3;
        let mut crossings = Vec::new();
        let mut level = TAU;
        let mut last = 0.0;
        for step in 0..40_000u32 {
            pair.step(dt).unwrap();
            let phi = pair.theta[1] - pair.theta[0];
            if phi >= level {
                let frac = (level - last) / (phi - last);
                crossings.push((f64::from(step) + frac) * dt);
                level += TAU;
            }
            last = phi;
        }
        assert!(crossings.len() >= 4, "only {} slips in 40 s", crossings.len());
        let period = (crossings[3] - crossings[0]) / 3.0;
        let want = TAU / beat_frequency(1.0, 0.6).unwrap();
        assert!((want - TAU / 0.8).abs() < 1e-12);
        // Linear interpolation of a crossing is good to dt²·|φ̈|/(8|φ̇|) ≤ 1e-6·(0.6·1.6)/(8·0.4).
        assert!((period - want).abs() < 1e-6, "slip period {period}, Adler says {want}");
    }

    #[test]
    fn identical_oscillators_close_at_rate_k() {
        let n = 40;
        let mut rng = Rng::new(9);
        let mut delta: Vec<f64> = (0..n).map(|_| 0.02 * (rng.next_f64() - 0.5)).collect();
        let mean = delta.iter().sum::<f64>() / n as f64;
        for d in &mut delta {
            *d -= mean;
        }
        let spread = |v: &[f64]| {
            let m = v.iter().sum::<f64>() / v.len() as f64;
            (v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / v.len() as f64).sqrt()
        };
        let before = spread(&delta);
        let mut pop = Kuramoto::new(vec![3.0; n], 2.0, delta).unwrap();
        pop.run(1e-3, 1000).unwrap();
        let ratio = spread(&pop.theta) / before;
        // Linearised about the synchronised state every deviation obeys δ' = −Kδ; the first
        // correction is cubic, relative size δ² = 1e-4.
        assert!((ratio / (-2.0f64).exp() - 1.0).abs() < 1e-3, "spread fell by {ratio}, e^-2 is {}", (-2.0f64).exp());
        let (r, _) = pop.order_parameter();
        assert!(r > 0.999_99);
        // The order parameter on states whose answer is known: a quarter-turn pair has r = √2/2 at
        // ψ = π/4, an antiphase pair has no coherence at all, and a lone phase reports itself.
        let quarter = Kuramoto::new(vec![0.0; 2], 0.0, vec![0.0, core::f64::consts::FRAC_PI_2]).unwrap();
        let (r, psi) = quarter.order_parameter();
        assert!((r - core::f64::consts::FRAC_1_SQRT_2).abs() < 1e-15 && (psi - core::f64::consts::FRAC_PI_4).abs() < 1e-15);
        assert!(Kuramoto::new(vec![0.0; 2], 0.0, vec![0.3, 0.3 + PI]).unwrap().order_parameter().0 < 1e-15);
        let lone = Kuramoto::new(vec![0.0], 0.0, vec![-2.0]).unwrap().order_parameter();
        assert!((lone.0 - 1.0).abs() < 1e-15 && (lone.1 + 2.0).abs() < 1e-15);
    }

    #[test]
    fn the_ott_antonsen_solution_solves_its_equation() {
        let (k, gamma) = (3.0, 0.5);
        let rhs = |r: f64| -gamma * r + 0.5 * k * r * (1.0 - r * r);
        for &(r0, t) in &[(0.05, 0.5), (0.05, 3.0), (0.9, 1.0), (0.3, 0.0)] {
            let h = 1e-5;
            let t_lo = if t == 0.0 { 0.0 } else { t - h };
            let slope = (ott_antonsen_r(r0, k, gamma, t + h).unwrap() - ott_antonsen_r(r0, k, gamma, t_lo).unwrap()) / (t + h - t_lo);
            let r = ott_antonsen_r(r0, k, gamma, 0.5 * (t + h + t_lo)).unwrap();
            assert!((slope - rhs(r)).abs() < 1e-8, "at r0 = {r0}, t = {t}: slope {slope}, equation {}", rhs(r));
        }
        assert_eq!(ott_antonsen_r(0.3, k, gamma, 0.0), Some(0.3));
        let settled = ott_antonsen_r(0.05, k, gamma, 200.0).unwrap();
        assert!((settled - (1.0f64 - 1.0 / 3.0).sqrt()).abs() < 1e-12);
        assert_eq!(lorentzian_critical_coupling(0.5), 1.0);
        // Below threshold it decays to nothing, and AT threshold algebraically: r² = r0²/(1 + K r0² t).
        assert!(ott_antonsen_r(0.5, 0.5, 0.5, 100.0).unwrap() < 1e-10);
        // (K = 3 and not 1, so that the K in that formula is visible: at K = 1 dropping it
        // survived this module's own mutation sweep.)
        let at = ott_antonsen_r(0.5, 3.0, 1.5, 4.0).unwrap();
        assert!((at - (0.25f64 / (1.0 + 3.0 * 0.25 * 4.0)).sqrt()).abs() < 1e-15, "{at}");
        assert_eq!(at, 0.25);
        assert_eq!(ott_antonsen_r(0.0, k, gamma, 5.0), Some(0.0));
        for bad in [ott_antonsen_r(1.5, k, gamma, 1.0), ott_antonsen_r(0.5, 0.0, gamma, 1.0), ott_antonsen_r(0.5, k, 0.0, 1.0), ott_antonsen_r(0.5, k, gamma, -1.0)] {
            assert_eq!(bad, None);
        }
    }

    #[test]
    fn a_lorentzian_population_synchronises_above_two_gamma() {
        let n = 400;
        let gamma = 0.5;
        let omega = lorentzian_quantiles(n, 0.0, gamma).unwrap();
        assert!((omega[n / 2 - 1] + omega[n / 2]).abs() < 1e-12, "the quantiles are symmetric");
        // The fastest member turns at 2γn/π = 127 rad/s, so dt = 2e-3 keeps ω·dt at a quarter.
        assert!((omega[n - 1] - 2.0 * gamma * n as f64 / PI).abs() < 0.01 * omega[n - 1]);
        let mut rng = Rng::new(11);
        let average_r = |k: f64, rng: &mut Rng| {
            let theta: Vec<f64> = (0..n).map(|_| TAU * rng.next_f64()).collect();
            let mut pop = Kuramoto::new(omega.clone(), k, theta).unwrap();
            pop.run(2e-3, 20_000).unwrap();
            let mut acc = 0.0;
            for _ in 0..5_000 {
                pop.step(2e-3).unwrap();
                acc += pop.order_parameter().0;
            }
            acc / 5_000.0
        };
        let finite_size = 2.0 / (n as f64).sqrt();
        let above = average_r(2.0, &mut rng);
        let want = (1.0f64 - 2.0 * gamma / 2.0).sqrt();
        assert!((above - want).abs() < finite_size, "r = {above} at K = 2, theory {want} ± {finite_size}");
        let below = average_r(0.5, &mut rng);
        assert!(below < finite_size, "r = {below} at half the critical coupling");
    }

    fn ring(n: usize) -> Vec<(usize, usize, f64)> {
        (0..n).map(|i| (i, (i + 1) % n, 1.0)).collect()
    }

    #[test]
    fn the_oim_velocity_is_minus_the_gradient_of_its_energy() {
        let mut rng = Rng::new(21);
        let edges = [(0, 1, 1.0), (1, 2, 2.5), (2, 3, 0.5), (3, 0, 1.0), (0, 2, 1.5)];
        let mut oim = Oim::max_cut(4, &edges, 0.8, 0.6).unwrap();
        oim.randomise(&mut rng);
        let v = oim.velocity();
        for i in 0..4 {
            let h = 1e-6;
            let mut up = oim.clone();
            up.theta[i] += h;
            let mut down = oim.clone();
            down.theta[i] -= h;
            let grad = (up.energy() - down.energy()) / (2.0 * h);
            assert!((v[i] + grad).abs() < 1e-8, "vertex {i}: velocity {}, −gradient {}", v[i], -grad);
        }
        // At binary phases the energy is the Ising energy: K (W − 2·cut) − K_s N/2.
        oim.theta = vec![0.0, PI, 0.0, PI];
        let total = 6.5;
        let cut = 1.0 + 2.5 + 0.5 + 1.0;
        assert_eq!(oim.cut(), cut);
        assert!((oim.energy() - (0.8 * (total - 2.0 * cut) - 0.6 * 4.0 / 2.0)).abs() < 1e-12);
        assert_eq!(oim.binarisation(), 1.0);
        assert_eq!(oim.spins(), vec![1, -1, 1, -1]);
    }

    #[test]
    fn euler_at_the_stable_step_never_raises_the_energy() {
        let mut rng = Rng::new(33);
        let mut oim = Oim::max_cut(8, &ring(8), 1.0, 1.0).unwrap();
        // L = 2·1·2 + 2·1 = 6.
        assert!((oim.stable_dt() - 1.0 / 6.0).abs() < 1e-15);
        oim.randomise(&mut rng);
        let start = oim.energy();
        let worst = oim.relax(oim.stable_dt(), 2000).unwrap();
        // The lemma is about exact arithmetic. The energy is a sum of 8 edge terms and 8 injection
        // terms of size ≤ 1, each good to an ulp, so two evaluations of the SAME converged state
        // may differ by 16ε — and that, not zero, is the floor a settled run reports.
        let rounding = 16.0 * f64::EPSILON;
        assert!(worst <= rounding, "the energy rose by {worst} in one step");
        assert!(oim.energy() < start - 1.0, "and it did descend: {start} → {}", oim.energy());
        // The monitor can fire: far past 2/L the same flow overshoots and the energy rises.
        oim.randomise(&mut rng);
        let rough = oim.relax(10.0 * oim.stable_dt(), 200).unwrap();
        assert!(rough > 1e-3, "a step of 10/L never raised the energy ({rough}); the monitor is untested");
    }

    #[test]
    fn the_oim_finds_the_maximum_cut_of_small_graphs() {
        let mut rng = Rng::new(7);
        // An even ring is bipartite: every edge can be cut.
        let mut even = Oim::max_cut(8, &ring(8), 1.0, 0.5).unwrap();
        let (cut, spins) = best_cut(&mut even, 10, 3000, &mut rng).unwrap();
        assert_eq!(cut, 8.0);
        assert!(spins.windows(2).all(|p| p[0] != p[1]));
        // A random graph against brute force.
        let n = 12;
        let mut edges = Vec::new();
        for u in 0..n {
            for v in (u + 1)..n {
                if rng.next_f64() < 0.4 {
                    edges.push((u, v));
                }
            }
        }
        let (_, best_energy) = Qubo::max_cut(n, &edges).unwrap().brute_force().unwrap();
        let weighted: Vec<(usize, usize, f64)> = edges.iter().map(|&(u, v)| (u, v, 1.0)).collect();
        let mut oim = Oim::max_cut(n, &weighted, 1.0, 0.5).unwrap();
        let (cut, _) = best_cut(&mut oim, 20, 3000, &mut rng).unwrap();
        assert!(cut <= -best_energy, "a cut of {cut} beats the brute-force optimum {}", -best_energy);
        assert_eq!(cut, -best_energy, "the best of 20 relaxations fell short of the optimum");
        // After relaxing with the second harmonic on, the phases are binary.
        assert!(oim.binarisation() > 0.99, "least binary phase has |cos θ| = {}", oim.binarisation());
        // Snapping keeps the cut and lands on the Ising energy exactly.
        let (before, spins) = (oim.cut(), oim.spins());
        oim.snap();
        assert_eq!((oim.cut(), oim.spins()), (before, spins));
        assert_eq!(oim.binarisation(), 1.0);
        assert!(oim.theta.iter().all(|t| *t == 0.0 || *t == PI));
        let total = edges.len() as f64;
        assert!((oim.energy() - (total - 2.0 * before - 0.5 * n as f64 / 2.0)).abs() < 1e-12);
    }

    #[test]
    fn the_chain_locks_into_a_travelling_wave() {
        let n = 10;
        let lag = TAU / n as f64;
        let mut rng = Rng::new(3);
        let theta: Vec<f64> = (0..n).map(|_| 0.8 * (rng.next_f64() - 0.5)).collect();
        let mut chain = CpgChain::new(TAU, 4.0, lag, theta).unwrap();
        // Slowest mode: 4·(2 − 2cos(π/10)) = 0.3915 1/s; 100 s leaves e^{−39}.
        assert!((chain.slowest_rate() - 4.0 * (2.0 - 2.0 * (PI / 10.0).cos())).abs() < 1e-15);
        chain.run(1e-2, 10_000).unwrap();
        let worst = chain.lag_errors().iter().fold(0.0f64, |a, e| a.max(e.abs()));
        assert!(worst < 1e-9, "a neighbouring lag is off by {worst}");
        for v in chain.velocity() {
            assert!((v - TAU).abs() < 1e-9, "a locked segment turns at {v}, not ω");
        }
        // Head to tail the body carries (n − 1)/n of a wavelength.
        let span = chain.theta[0] - chain.theta[n - 1];
        assert!((span - 9.0 * lag).abs() < 1e-8);
    }

    #[test]
    fn the_slowest_mode_decays_at_the_path_laplacian_rate() {
        let n = 8;
        let lag = 0.4;
        // The k = 1 eigenvector of the path Laplacian, on top of the locked wave.
        let mode: Vec<f64> = (0..n).map(|i| (PI * (i as f64 + 0.5) / n as f64).cos()).collect();
        let eps = 1e-3;
        let theta: Vec<f64> = (0..n).map(|i| -lag * i as f64 + eps * mode[i]).collect();
        let mut chain = CpgChain::new(5.0, 2.0, lag, theta).unwrap();
        let project = |c: &CpgChain, t: f64| {
            let dev: Vec<f64> = (0..n).map(|i| c.theta[i] - 5.0 * t + lag * i as f64).collect();
            dev.iter().zip(&mode).map(|(d, m)| d * m).sum::<f64>() / mode.iter().map(|m| m * m).sum::<f64>()
        };
        let t = 2.0;
        chain.run(1e-3, 2000).unwrap();
        let ratio = project(&chain, t) / eps;
        let want = (-chain.slowest_rate() * t).exp();
        // The sine's cubic term is relative ε² = 1e-6; RK4 at dt = 1e-3 is far below that.
        assert!((ratio / want - 1.0).abs() < 1e-5, "the mode fell to {ratio}, the Laplacian says {want}");
    }

    #[test]
    fn the_amplitude_is_critically_damped_and_its_step_is_exact() {
        let (a, r0, target) = (20.0, 0.2, 1.0);
        let closed = |t: f64| target - (target - r0) * (1.0 + 0.5 * a * t) * (-0.5 * a * t).exp();
        let mut one = Amplitude::new(a, r0).unwrap();
        one.step(0.3, target).unwrap();
        assert!((one.r - closed(0.3)).abs() < 1e-15);
        // Three hundred small steps land where one large one does: the update composes.
        let mut many = Amplitude::new(a, r0).unwrap();
        for _ in 0..300 {
            many.step(1e-3, target).unwrap();
        }
        assert!((many.r - closed(0.3)).abs() < 1e-13, "{} vs {}", many.r, closed(0.3));
        assert!((many.r_dot - one.r_dot).abs() < 1e-12);
        // Critically damped: it never overshoots the target.
        let mut probe = Amplitude::new(a, r0).unwrap();
        for _ in 0..2000 {
            assert!(probe.step(1e-3, target).unwrap() <= target);
        }
        assert!((probe.r - target).abs() < 1e-6);
    }

    #[test]
    fn bad_arguments_are_refused() {
        assert!(matches!(Kuramoto::new(vec![], 1.0, vec![]), Err(OscillatorError::Empty { .. })));
        assert!(matches!(Kuramoto::new(vec![1.0], 1.0, vec![]), Err(OscillatorError::Dimension { .. })));
        assert!(matches!(Kuramoto::new(vec![f64::NAN], 1.0, vec![0.0]), Err(OscillatorError::NonFinite { what: "omega", .. })));
        assert!(matches!(Kuramoto::new(vec![1.0], 1.0, vec![f64::INFINITY]), Err(OscillatorError::NonFinite { what: "theta", .. })));
        assert!(matches!(Kuramoto::new(vec![1.0], -0.1, vec![0.0]), Err(OscillatorError::OutOfRange { what: "k", .. })));
        let mut k = Kuramoto::new(vec![1.0], 0.0, vec![0.0]).unwrap();
        assert!(matches!(k.step(0.0), Err(OscillatorError::OutOfRange { what: "dt", .. })));
        assert!(matches!(k.run(1e-3, MAX_STEPS + 1), Err(OscillatorError::OutOfRange { what: "steps", .. })));
        // Uncoupled, an oscillator turns at its own frequency, exactly.
        k.run(0.5, 4).unwrap();
        assert_eq!(k.theta[0], 2.0);
        assert!(matches!(lorentzian_quantiles(0, 0.0, 1.0), Err(OscillatorError::Empty { .. })));
        assert!(matches!(lorentzian_quantiles(4, 0.0, 0.0), Err(OscillatorError::OutOfRange { .. })));
        assert!(matches!(lorentzian_quantiles(4, f64::NAN, 1.0), Err(OscillatorError::NonFinite { .. })));
        assert!(matches!(CpgChain::new(1.0, 1.0, 0.1, vec![0.0]), Err(OscillatorError::Empty { .. })));
        assert!(matches!(CpgChain::new(1.0, 0.0, 0.1, vec![0.0, 0.0]), Err(OscillatorError::OutOfRange { what: "w", .. })));
        assert!(matches!(CpgChain::new(f64::NAN, 1.0, 0.1, vec![0.0, 0.0]), Err(OscillatorError::NonFinite { what: "omega", .. })));
        assert!(matches!(CpgChain::new(1.0, 1.0, f64::NAN, vec![0.0, 0.0]), Err(OscillatorError::NonFinite { what: "lag", .. })));
        assert!(matches!(Amplitude::new(0.0, 1.0), Err(OscillatorError::OutOfRange { what: "a", .. })));
        assert!(matches!(Amplitude::new(1.0, f64::NAN), Err(OscillatorError::NonFinite { .. })));
        let mut amp = Amplitude::new(1.0, 0.0).unwrap();
        assert!(matches!(amp.step(1e-3, f64::NAN), Err(OscillatorError::NonFinite { what: "target", .. })));
        assert!(matches!(amp.step(-1.0, 1.0), Err(OscillatorError::OutOfRange { what: "dt", .. })));
        assert!(matches!(Oim::max_cut(0, &[], 1.0, 1.0), Err(OscillatorError::Empty { .. })));
        assert!(matches!(Oim::max_cut(3, &[(0, 3, 1.0)], 1.0, 1.0), Err(OscillatorError::Index { index: 3, .. })));
        assert!(matches!(Oim::max_cut(3, &[(3, 0, 1.0)], 1.0, 1.0), Err(OscillatorError::Index { index: 3, .. })));
        assert!(matches!(Oim::max_cut(3, &[(1, 1, 1.0)], 1.0, 1.0), Err(OscillatorError::OutOfRange { .. })));
        assert!(matches!(Oim::max_cut(3, &[(0, 1, -1.0)], 1.0, 1.0), Err(OscillatorError::OutOfRange { what: "weight", .. })));
        assert!(matches!(Oim::max_cut(3, &[], 0.0, 1.0), Err(OscillatorError::OutOfRange { what: "k", .. })));
        assert!(matches!(Oim::max_cut(3, &[], 1.0, -1.0), Err(OscillatorError::OutOfRange { what: "k_s", .. })));
        // A repeated edge adds its weight, on both sides of the diagonal.
        let twice = Oim::max_cut(2, &[(0, 1, 1.0), (1, 0, 2.0)], 1.0, 0.0).unwrap();
        assert_eq!(twice.w, vec![0.0, 3.0, 3.0, 0.0]);
        let mut empty = Oim::max_cut(2, &[], 1.0, 0.0).unwrap();
        assert!(matches!(best_cut(&mut empty, 0, 10, &mut Rng::new(1)), Err(OscillatorError::Empty { .. })));
        assert_eq!(wrap_pi(3.0 * PI), PI);
        assert_eq!(wrap_pi(-PI), PI);
        assert_eq!(wrap_pi(0.5), 0.5);
        assert!((wrap_pi(-0.5) + 0.5).abs() < 1e-15);
    }

    /// The ceiling on one `run` is the hundred million its constant documents, and a run of
    /// EXACTLY that many steps is started rather than refused.
    ///
    /// Why the suite could not see either half: its only ceiling check spells the bound
    /// `MAX_STEPS + 1`, so a constant a tenth the size moves the test along with it and the
    /// refusal still fires; and nothing asked what happens AT the ceiling, where `steps >=
    /// MAX_STEPS` refuses a run the documentation admits. The run at the ceiling is asked for
    /// with a `dt` that cannot be taken, so the answer costs one rejected step rather than 1e8
    /// taken ones: a refusal names `steps`, an admission names `dt`.
    #[test]
    fn the_step_ceiling_is_the_hundred_million_it_documents_and_a_run_of_that_many_is_admitted() {
        const { assert!(MAX_STEPS == 100_000_000) };
        let mut k = Kuramoto::new(vec![1.0], 0.0, vec![0.0]).unwrap();
        assert!(matches!(k.run(0.0, MAX_STEPS), Err(OscillatorError::OutOfRange { what: "dt", .. })));
        assert!(matches!(k.run(1e-3, MAX_STEPS + 1), Err(OscillatorError::OutOfRange { what: "steps", .. })));
    }

    /// A non-finite entry is reported at the position it occupies, not always at position zero.
    ///
    /// Why the suite could not see it: every refusal check matches with `..` and never reads the
    /// index, and each one passes a ONE-element array, where position zero is the right answer
    /// whatever the code does.
    #[test]
    fn a_non_finite_entry_is_reported_at_the_position_it_occupies() {
        assert_eq!(
            Kuramoto::new(vec![1.0, 2.0, f64::NAN], 1.0, vec![0.0; 3]),
            Err(OscillatorError::NonFinite { what: "omega", index: 2 })
        );
        assert_eq!(
            Kuramoto::new(vec![1.0; 3], 1.0, vec![0.0, f64::INFINITY, 0.0]),
            Err(OscillatorError::NonFinite { what: "theta", index: 1 })
        );
        assert_eq!(
            CpgChain::new(1.0, 1.0, 0.1, vec![0.0, 0.0, f64::NEG_INFINITY]),
            Err(OscillatorError::NonFinite { what: "theta", index: 2 })
        );
    }

    /// An infinity is refused everywhere the positivity guard stands: a step, a chain weight, an
    /// amplitude rate, a half-width and a coupling gain.
    ///
    /// Why the suite could not see it: `bad_arguments_are_refused` probes each of those sites with
    /// zero or with a negative number, and `v > 0.0` alone still rejects both. Only an infinity
    /// separates the guard from its finiteness clause, and an accepted infinite `dt` turns a state
    /// into NaN while returning `Ok`.
    #[test]
    fn an_infinite_step_rate_or_coupling_is_refused() {
        let mut k = Kuramoto::new(vec![1.0], 1.0, vec![0.25]).unwrap();
        assert!(matches!(k.step(f64::INFINITY), Err(OscillatorError::OutOfRange { what: "dt", .. })));
        assert_eq!(k.theta, vec![0.25], "a refused step must leave the state where it was");
        assert!(matches!(
            CpgChain::new(1.0, f64::INFINITY, 0.1, vec![0.0, 0.0]),
            Err(OscillatorError::OutOfRange { what: "w", .. })
        ));
        assert!(matches!(Amplitude::new(f64::INFINITY, 1.0), Err(OscillatorError::OutOfRange { what: "a", .. })));
        assert!(matches!(
            lorentzian_quantiles(4, 0.0, f64::INFINITY),
            Err(OscillatorError::OutOfRange { what: "gamma", .. })
        ));
        assert!(matches!(Oim::max_cut(2, &[], f64::INFINITY, 1.0), Err(OscillatorError::OutOfRange { what: "k", .. })));
        let mut amp = Amplitude::new(1.0, 0.0).unwrap();
        assert!(matches!(amp.step(f64::INFINITY, 1.0), Err(OscillatorError::OutOfRange { what: "dt", .. })));
    }

    /// The lock range is the CLOSED interval `|Δω| ≤ K` about zero detuning — both signs of it —
    /// and a zero coupling is outside it rather than an `asin(0/0)`.
    ///
    /// Why the suite could not see it: its out-of-range probes are `locked_phase(1.0001, 1.0)` and
    /// `locked_phase(0.5, 0.0)`. A one-sided `Δω > K` still refuses the first (its detuning is
    /// positive) and still refuses the second (`0.5 > 0.0`), so neither reads the absolute value;
    /// and a guard that admits `K = 0` is only visible at the one detuning that is not itself
    /// outside a zero range, `Δω = 0`, where `0/0` reaches `asin` as a NaN.
    #[test]
    fn the_lock_range_is_closed_and_symmetric_and_a_zero_coupling_is_outside_it() {
        assert_eq!(locked_phase(-1.5, 1.0), None);
        assert_eq!(locked_phase(1.5, 1.0), None);
        assert_eq!(locked_phase(0.0, 0.0), None);
        assert_eq!(locked_phase(-0.0, 0.0), None);
        assert_eq!(locked_phase(-0.6, 1.0), Some((-0.6f64).asin()));
        assert_eq!(locked_phase(0.6, 1.0), Some(0.6f64.asin()));
    }

    /// A phase array LONGER than the frequencies is a dimension error, not an accepted population.
    ///
    /// Why the suite could not see it: its only dimension probe is an empty phase array against one
    /// frequency, which a `theta.len() < omega.len()` test refuses just as well. Accepted, the
    /// extra phases would be carried in the state and summed into every mean field while no
    /// velocity was ever written for them.
    #[test]
    fn a_phase_array_longer_than_the_frequencies_is_refused() {
        assert_eq!(
            Kuramoto::new(vec![1.0], 1.0, vec![0.0, 0.0]),
            Err(OscillatorError::Dimension { what: "theta", got: 2, want: 1 })
        );
    }

    /// Full coherence is INSIDE the Ott–Antonsen domain: `r0 = 1` is answered, not refused.
    ///
    /// Why the suite could not see it: its domain probes are `r0 = 1.5` (outside either way) and
    /// starts of 0.05, 0.3, 0.5 and 0.9, all strictly below one. The endpoint the documented range
    /// `[0, 1]` includes is the only value a half-open range would drop.
    #[test]
    fn full_coherence_is_an_admissible_start_for_the_ott_antonsen_solution() {
        let (k, gamma) = (3.0, 0.5);
        assert_eq!(ott_antonsen_r(1.0, k, gamma, 0.0), Some(1.0));
        let later = ott_antonsen_r(1.0, k, gamma, 200.0).unwrap();
        assert!((later - (1.0 - 2.0 * gamma / k).sqrt()).abs() < 1e-12, "a fully coherent start settled at {later}");
        assert_eq!(ott_antonsen_r(1.000_000_1, k, gamma, 0.0), None);
    }

    /// An incoherent start stays at zero however long the run, rather than becoming a NaN once the
    /// logistic exponential underflows.
    ///
    /// Why the suite could not see it: its `r0 = 0` probe is at `t = 5`, where `e^{−2t}` is still
    /// 4.5e−5, so the infinity from `cap/u0` merely swamps it and the answer is right by accident.
    /// The `0 · ∞` needs `e^{−rate·t}` to reach EXACTLY zero, which at this rate takes `t ≳ 355`.
    #[test]
    fn an_incoherent_start_stays_incoherent_however_long_the_run() {
        assert_eq!((-2.0f64 * 400.0).exp(), 0.0, "this test only bites if the exponential underflows");
        assert_eq!(ott_antonsen_r(0.0, 3.0, 0.5, 400.0), Some(0.0));
    }

    /// The chain reports one lag error per neighbouring PAIR, each wrapped into the half-open turn
    /// `(−π, π]`.
    ///
    /// Why the suite could not see it: the locked-wave test reads the errors only through
    /// `max |e|` after the chain has converged, where every error is within 1e−9 of zero. A window
    /// short by one pair still reports those zeros, and a wrap that never happens is invisible
    /// because nothing near zero needs wrapping. This fixture is a chain that is NOT locked, with
    /// raw differences of 5.6 and −12.5 rad.
    #[test]
    fn every_neighbouring_pair_reports_a_lag_error_wrapped_into_one_turn() {
        let lag = 0.4;
        let chain = CpgChain::new(1.0, 1.0, lag, vec![0.5, -5.5, -11.5, 0.6, 3.0]).unwrap();
        let errs = chain.lag_errors();
        assert_eq!(errs.len(), chain.theta.len() - 1);
        for e in &errs {
            assert!(*e > -PI && *e <= PI, "a lag error escaped the turn: {e}");
        }
        let raw = 0.5 - (-5.5) - lag;
        assert!(raw > PI, "the fixture must need wrapping: {raw}");
        assert_eq!(errs[0], raw - TAU);
    }

    /// A new machine starts with every phase at zero — the equilibrium its own constructor
    /// documents and tells the caller to randomise away from.
    ///
    /// Why the suite could not see it: every fixture calls `randomise` before reading anything, and
    /// all-π is an equilibrium of the same flow with the same energy as all-zero: the edge terms
    /// read `cos(π − π)` and the injection `cos 2π`. Only the phases themselves, and the spins they
    /// read as, tell the two starts apart.
    #[test]
    fn a_new_machine_starts_at_phase_zero() {
        let oim = Oim::max_cut(4, &ring(4), 1.0, 0.5).unwrap();
        assert_eq!(oim.theta, vec![0.0; 4]);
        assert_eq!(oim.spins(), vec![1; 4]);
        assert_eq!(oim.cut(), 0.0);
    }

    /// `randomise` draws from the WHOLE circle, not from half of it.
    ///
    /// Why the suite could not see it: a machine randomised on `[0, π)` still relaxes to the same
    /// cuts — the second harmonic pulls every phase to `0` or `π` from either half — so the
    /// max-cut and descent tests are blind to it. What changes is the SUPPORT of the draw, which
    /// nothing looked at.
    #[test]
    fn randomise_draws_from_the_whole_circle() {
        let mut oim = Oim::max_cut(64, &[], 1.0, 0.0).unwrap();
        oim.randomise(&mut Rng::new(4));
        assert!(oim.theta.iter().all(|t| (0.0..TAU).contains(t)), "a phase left the turn: {:?}", oim.theta);
        let above = oim.theta.iter().filter(|t| **t > PI).count();
        // 64 fair draws: the count above π has mean 32 and standard deviation 4, so this window is
        // three standard deviations either side and the seed is fixed. Measured at this seed: 38 of 64.
        assert!((20..=44).contains(&above), "{above} of 64 phases fell above π");
    }

    /// The energy reads the entry at row `i`, column `j` — the upper triangle its own formula
    /// `Σ_{i<j} w_ij` names — and not the transpose.
    ///
    /// Why the suite could not see it: every fixture reaches `energy` through `Oim::max_cut`, which
    /// writes `w[u][v]` and `w[v][u]` together, and a symmetric matrix IS its own transpose. The
    /// `w` field is public, so an asymmetric matrix is reachable from outside the module without
    /// touching the constructor, and it is the only state in which the two readings differ.
    #[test]
    fn the_energy_reads_the_upper_triangle_of_the_weights_and_not_its_transpose() {
        let mut oim = Oim::max_cut(2, &[(0, 1, 1.0)], 1.0, 0.0).unwrap();
        oim.w = vec![0.0, 2.0, 0.5, 0.0];
        oim.theta = vec![0.0, PI];
        // One pair, i < j: E = K w_01 cos(θ0 − θ1) = 1 · 2 · (−1), with no injection at K_s = 0.
        assert_eq!(oim.energy(), -2.0);
    }

    /// The Gershgorin row sum covers every column of its row, the last one included.
    ///
    /// Why the suite could not see it: its only fixture is a ring, where the last column of row `i`
    /// is empty for every `i` but `0` and `n − 1`, and row `n − 1` carries the same mass in
    /// columns that are not last. Dropping the last column therefore leaves the MAXIMUM row sum
    /// untouched. A star at vertex 0 puts the largest row's mass in that column.
    #[test]
    fn the_gershgorin_row_sum_includes_the_last_column() {
        let oim = Oim::max_cut(3, &[(0, 1, 3.0), (0, 2, 3.0)], 1.0, 0.0).unwrap();
        // Row 0 sums to 6 and rows 1 and 2 to 3, so L = 2K·6 + 2K_s = 12; without w[0][2] every
        // row would sum to 3 and the step would double.
        assert_eq!(oim.stable_dt(), 1.0 / 12.0);
    }

    /// An uncoupled machine is given a FINITE stable step, and a step of it is a step.
    ///
    /// Why the suite could not see it: the only machine it measures `stable_dt` on is a coupled
    /// ring with `K_s = 1`, whose Lipschitz bound is 6. The floor under the divisor is reached only
    /// when a machine has no edges AND no injection, where `1/0` is an infinity that turns the
    /// first step into `∞ · 0 = NaN`.
    #[test]
    fn an_uncoupled_machine_is_given_a_finite_stable_step() {
        let mut oim = Oim::max_cut(3, &[], 1.0, 0.0).unwrap();
        let dt = oim.stable_dt();
        assert!(dt.is_finite(), "an uncoupled machine was handed dt = {dt}");
        assert_eq!(dt, 1.0 / f64::MIN_POSITIVE);
        oim.theta = vec![0.25, 0.5, 0.75];
        oim.step(dt).unwrap();
        assert_eq!(oim.theta, vec![0.25, 0.5, 0.75], "a machine with no forces moved");
    }

    /// Binarisation reports the LEAST binary oscillator: the smallest `|cos θ|` in the machine.
    ///
    /// Why the suite could not see it: it reads `binarisation` only on states where every phase is
    /// already binary (`= 1`) or every phase is nearly so (`> 0.99`), and on those the smallest and
    /// the largest `|cos θ|` agree to within the tolerance. A mixed machine separates them.
    #[test]
    fn binarisation_reports_the_least_binary_oscillator() {
        let mut oim = Oim::max_cut(3, &ring(3), 1.0, 0.5).unwrap();
        oim.theta = vec![0.0, 1.0, PI];
        assert_eq!(oim.binarisation(), 1.0f64.cos());
        assert!(oim.binarisation() < 1.0, "a mixed machine read as fully binary");
    }

    /// A run that only descended reports the fall it saw, not zero.
    ///
    /// Why the suite could not see it: the descent test asserts `worst <= 16ε` and the overshoot
    /// test asserts `rough > 1e-3`, and a monitor that starts its maximum at zero satisfies both —
    /// zero is below the first bound and the second run really does rise. What is lost is the
    /// SIGN of a run that never rose, which is the sentence the return value's documentation
    /// makes. One step of a descending run is the whole answer, with no maximum to hide it.
    #[test]
    fn a_run_that_only_descended_reports_a_negative_worst_rise() {
        let mut oim = Oim::max_cut(6, &ring(6), 1.0, 0.5).unwrap();
        oim.randomise(&mut Rng::new(12));
        let dt = oim.stable_dt();
        let mut probe = oim.clone();
        let before = probe.energy();
        let fall = probe.step(dt).unwrap() - before;
        assert!(fall < -1e-6, "the fixture did not descend: {fall}");
        assert_eq!(oim.relax(dt, 1).unwrap(), fall);
        assert_eq!(oim.theta, probe.theta);
        let worst = oim.relax(dt, 3).unwrap();
        assert!(worst < 0.0, "three descending steps reported a worst rise of {worst}");
    }
}
