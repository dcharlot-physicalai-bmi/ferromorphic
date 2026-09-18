//! Mean-field theory: what a spiking network does in aggregate, in closed form.
//!
//! # The lesson
//!
//! Simulating a million neurons to learn the population firing rate is the expensive way to get an
//! answer theory gives exactly. A cortical neuron receives thousands of synapses, each delivering a
//! tiny voltage step at an irregular time. Add enough of them and the *central limit theorem*
//! applies: the summed input stops looking like a train of discrete kicks and starts looking like a
//! constant drift plus Gaussian noise. That substitution is the **diffusion approximation**, and it
//! turns a network of coupled point processes into a single Ornstein-Uhlenbeck process with an
//! absorbing boundary — an object with a first-passage time that is a one-dimensional integral.
//!
//! What it buys: a firing rate for 10<sup>6</sup> neurons in microseconds, an exact statement of
//! *why* a parameter change moved the rate, and — the reason this module sits in a simulation
//! library — a reference every other module can be checked against without running a simulation at
//! all. [`SiegertInput::rate`] is a closed form; [`simulate_diffusion`] is the simulation; the test
//! `the_siegert_rate_matches_a_direct_simulation_of_the_same_lif` runs both and compares them.
//!
//! What it costs, and this is not a footnote:
//!
//! - **It assumes the input is white noise.** Real synaptic current has a time constant, and
//!   Brunel & Sergi (J. Theor. Biol. 195:87, 1998) and Fourcaud & Brunel (Neural Comput.
//!   14:2057, 2002) show the correction is `O(sqrt(tau_syn / tau_m))` — first order in a square
//!   root, so a 5% time-constant ratio is a 22% correction. **This module implements the white-noise
//!   limit only.** This implementation did not locate a form of the coloured-noise correction it
//!   could check against a closed form, so it is absent rather than approximate.
//! - **It assumes neurons are independent.** They are not: two neurons sharing 10% of their inputs
//!   have correlated drive, and correlations change the population rate. The independence assumption
//!   is what makes the self-consistent equation one-dimensional.
//! - **It is a *stationary* theory.** [`SiegertInput::rate`] answers "what rate does this input
//!   sustain", not "how fast does the population get there". [`RefractoryDensity`] is the answer to
//!   the second question for a renewal population, and it is a different object.
//!
//! # Units
//!
//! SI throughout: seconds, volts, hertz. Synaptic efficacy `j` is **volts per spike** — the
//! membrane displacement one presynaptic spike produces, which is what [`crate::neuron::Neuron::bump`]
//! takes. The dimensionless combinations the papers work in (`(v_th - mu) / sigma`, `g * gamma`,
//! `nu_ext / nu_thr`) are exposed by name so a reader can compare a figure axis directly:
//! [`SiegertInput::threshold_distance`], [`BrunelNetwork::balance_index`],
//! [`BrunelNetwork::nu_ext_ratio`].
//!
//! # The noise convention, which is the easiest thing here to get wrong
//!
//! Every formula below is written for
//!
//! ```text
//! tau_m dV/dt = -(V - mu) + sigma sqrt(tau_m) eta(t),   <eta(t) eta(t')> = delta(t - t')
//! ```
//!
//! which is Brunel's own convention (J. Comput. Neurosci. 8:183-208, 2000, section 2). Note what
//! `sigma` is **not**: the stationary standard deviation of the free membrane potential under this
//! equation is `sigma / sqrt(2)`, not `sigma`. Papers differ on this factor and a `sqrt(2)` in the
//! wrong place moves a predicted rate by tens of percent while leaving every plot looking sane.
//! [`SiegertInput::free_membrane_sd`] returns the other one, explicitly, so the two never have to be
//! inferred from context.
//!
//! # What is in here
//!
//! | Mechanism | Source | Checked against |
//! |---|---|---|
//! | Siegert stationary rate | Brunel 2000 eq. 22; Ricciardi 1977 | a direct `Lif` simulation, and `Lif::isi` in the zero-noise limit |
//! | Balanced network | van Vreeswijk & Sompolinsky, Science 274:1724, 1996 | `sigma` exactly independent of `K`; the linear balance equations solved to machine precision |
//! | Phase diagram | Brunel 2000 | the `g * gamma = 1` boundary, exactly |
//! | Refractory density | Gerstner, Neural Comput. 12:43, 2000 | stationary activity equals `1 / mean ISI` to 1e-13 |
//! | Avalanches and criticality | Beggs & Plenz, J. Neurosci. 23:11167, 2003 | the Borel distribution, whose `-3/2` tail is exact |
//!
//! # Two results in fifteen lines
//!
//! ```
//! use ferromorphic::meanfield::{BranchingProcess, SiegertInput};
//! use ferromorphic::neuron::Lif;
//!
//! // Brunel's cell: 20 ms membrane, 20 mV threshold, 10 mV reset, 2 ms dead time.
//! let lif = Lif { tau_m: 20e-3, v_rest: 0.0, v_th: 20e-3, v_reset: 10e-3,
//!                 r_m: 10e6, t_ref: 2e-3, v: 10e-3, refractory: 0.0 };
//!
//! // Drive it to a mean 5 mV BELOW threshold. Deterministically it never fires at all.
//! assert!(lif.isi(15e-3 / lif.r_m).is_none());
//!
//! // Add 5 mV of input noise and the same cell fires at 9.5 Hz — in closed form, no simulation.
//! let input = SiegertInput::from_lif(&lif, 15e-3, 5e-3)?;
//! assert!(!input.is_mean_driven());
//! assert!((input.rate().unwrap() - 9.4608).abs() < 1e-3);
//!
//! // And the -3/2 avalanche exponent is derived rather than fitted: at criticality the total
//! // progeny is Borel-distributed, whose tail is exactly `k^(-3/2) / sqrt(2 pi)`.
//! let critical = BranchingProcess::new(1.0)?;
//! let tail = 1000f64.powf(-1.5) / (2.0 * std::f64::consts::PI).sqrt();
//! assert!((critical.borel_pmf(1000).unwrap() / tail - 1.0).abs() < 1e-3);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # Determinism
//!
//! Everything analytic here is a pure function of its arguments. Everything stochastic takes a
//! [`Rng`] and is reproducible from its seed, on every target.

use crate::neuron::{Lif, Neuron};
use crate::rng::Rng;
use crate::spike::{Spike, Train};
use crate::surrogate::erf;
use core::f64::consts::PI;

/// What went wrong, named, with the number that made it wrong.
///
/// Mean-field formulas fail quietly. A `NaN` threshold produces a `NaN` rate, which produces a
/// `NaN` self-consistent solution, which a bisection reports as a perfectly ordinary midpoint — so
/// every entry point here checks its arguments and says which one it rejected.
#[derive(Debug, Clone, PartialEq)]
pub enum MeanFieldError {
    /// A supplied number was not finite.
    NonFinite {
        /// Which parameter, for example `"sigma"` or `"hazard"`.
        what: &'static str,
        /// Index within an array, or `0` when `what` names a scalar.
        index: usize,
    },
    /// A scalar fell outside the range the mechanism is defined on.
    OutOfRange {
        /// Which parameter, for example `"tau_m"`.
        what: &'static str,
        /// The value supplied.
        value: f64,
        /// Lowest acceptable value, inclusive.
        low: f64,
        /// Highest acceptable value, inclusive.
        high: f64,
    },
    /// Two parameters were supplied in an order the mechanism forbids.
    Ordering {
        /// The parameter that must be the larger, for example `"v_th"`.
        larger: &'static str,
        /// The parameter that must be the smaller, for example `"v_reset"`.
        smaller: &'static str,
        /// The value supplied for `larger`.
        larger_value: f64,
        /// The value supplied for `smaller`.
        smaller_value: f64,
    },
    /// An array that must be non-empty was empty.
    Empty {
        /// Which array.
        what: &'static str,
    },
    /// Fewer samples than the estimator needs to return anything.
    TooFewSamples {
        /// Which estimator, for example `"power-law exponent"`.
        what: &'static str,
        /// How many usable samples were supplied.
        got: usize,
        /// How many are required.
        need: usize,
    },
    /// A self-consistent rate was asked for on a neuron with no absolute refractory period.
    ///
    /// The bracket `[0, 1 / t_ref]` is what guarantees the bisection contains a root, because the
    /// transfer function cannot exceed `1 / t_ref`. Without a refractory period the rate is
    /// unbounded above and there is no bracket to search, so this refuses rather than picking an
    /// arbitrary ceiling and reporting whatever root happens to lie under it.
    NoRefractoryPeriod,
    /// The linear balance equations have no solution with both rates positive.
    ///
    /// This is van Vreeswijk & Sompolinsky's existence condition, not a numerical failure: for some
    /// connectivity matrices the only way to cancel the `O(sqrt(K))` drive is with a negative firing
    /// rate, and the network then has no balanced state at all — it saturates or falls silent.
    NoBalancedState {
        /// Excitatory rate the linear system produced, hertz. Negative or non-finite.
        nu_exc: f64,
        /// Inhibitory rate the linear system produced, hertz.
        nu_inh: f64,
    },
    /// The balance matrix is singular to within the scale of its own entries, so the balanced rates
    /// are not determined.
    Degenerate {
        /// The determinant, in the units of the weight products.
        determinant: f64,
        /// The largest magnitude among the matrix entries, which the determinant is judged against.
        scale: f64,
    },
}

impl core::fmt::Display for MeanFieldError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NonFinite { what, index } => write!(f, "{what}[{index}] is not a finite number"),
            Self::OutOfRange { what, value, low, high } => {
                write!(f, "{what} = {value} is outside [{low}, {high}]")
            }
            Self::Ordering { larger, smaller, larger_value, smaller_value } => write!(
                f,
                "{larger} = {larger_value} must exceed {smaller} = {smaller_value}"
            ),
            Self::Empty { what } => write!(f, "{what} is empty"),
            Self::TooFewSamples { what, got, need } => {
                write!(f, "{what} needs {need} samples and was given {got}")
            }
            Self::NoRefractoryPeriod => f.write_str(
                "a self-consistent rate needs a positive absolute refractory period to bracket the \
                 search; without one the transfer function has no upper bound",
            ),
            Self::NoBalancedState { nu_exc, nu_inh } => write!(
                f,
                "the balance equations give nu_exc = {nu_exc} Hz and nu_inh = {nu_inh} Hz; a \
                 balanced state needs both positive"
            ),
            Self::Degenerate { determinant, scale } => write!(
                f,
                "the balance matrix has determinant {determinant} against an entry scale of \
                 {scale}; the balanced rates are not determined"
            ),
        }
    }
}

/// As elsewhere in this crate: an error that cannot cross a `Box<dyn Error>` boundary forces every
/// caller to write a conversion, and the ones who do not write it reach for `.unwrap()`.
impl std::error::Error for MeanFieldError {}

fn finite(what: &'static str, value: f64) -> Result<f64, MeanFieldError> {
    if value.is_finite() { Ok(value) } else { Err(MeanFieldError::NonFinite { what, index: 0 }) }
}

// ---------------------------------------------------------------------------------------------
// Special functions
// ---------------------------------------------------------------------------------------------

/// The **scaled** complementary error function, `erfcx(x) = exp(x^2) * erfc(x)`.
///
/// The Siegert integrand is `exp(u^2) * (1 + erf(u))`, which is `erfcx(-u)`. Writing it that way is
/// not cosmetic: `exp(u^2)` overflows `f64` at `u = 26.7` while the *product* stays in `(0, 1]` for
/// every `u <= 0`, so the scaled form is finite over the whole supra-threshold half of the problem
/// where the naive form is not.
///
/// For `x >= 2` this uses the continued fraction
/// `erfcx(x) = 1/sqrt(pi) * 1/(x + (1/2)/(x + 1/(x + (3/2)/(x + ...))))`
/// (Abramowitz & Stegun 7.1.14), evaluated by backward recurrence. Below 2 the continued fraction
/// converges slowly and `exp(x^2) * (1 - erf(x))` is used instead with [`crate::surrogate::erf`];
/// the crossover is placed at 2 because that is where the two paths are each at their best — at
/// `x = 2` they agree to `1.3e-14` absolute, which is the check
/// `erfcx_is_continuous_across_its_crossover` makes.
///
/// For `x < 0` the reflection `erfcx(x) = 2 exp(x^2) - erfcx(-x)` is exact and returns
/// [`f64::INFINITY`] once `exp(x^2)` overflows, at about `x = -26.7`.
///
/// Accurate to about `1e-13` relative over `[0, 6]`, checked against published values of
/// `erfcx(1)`, `erfcx(2)` and `erfcx(3)` and against the asymptotic series at `x = 100`.
#[must_use]
pub fn erfcx(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    if x >= 2.0 {
        // Backward recurrence on the continued fraction. 200 levels is far past convergence at
        // x = 2 (the value is already stable at 50) and the tail costs one divide each.
        let mut t = 0.0f64;
        for k in (1..=200u32).rev() {
            t = (f64::from(k) * 0.5) / (x + t);
        }
        return 1.0 / ((x + t) * PI.sqrt());
    }
    if x >= 0.0 {
        return (x * x).exp() * (1.0 - erf(x));
    }
    let e = (x * x).exp();
    if !e.is_finite() {
        return f64::INFINITY;
    }
    2.0 * e - erfcx(-x)
}

/// Number of Gauss-Legendre nodes per quadrature panel. Exact for polynomials up to degree 47,
/// which is what lets a panel of an analytic function be integrated to rounding.
const GL_N: usize = 24;

/// Gauss-Legendre nodes and weights on `[-1, 1]`, computed rather than transcribed.
///
/// Newton's method on the Legendre polynomial, with the standard Chebyshev-like starting guess. The
/// alternative — a table of 24 constants copied from a reference — cannot be checked by the crate
/// that uses it; this can, and `gauss_legendre_integrates_polynomials_exactly` does, by integrating
/// `x^k` for every `k` up to 47 and comparing against `2 / (k + 1)`.
fn gauss_legendre() -> ([f64; GL_N], [f64; GL_N]) {
    let mut xs = [0.0f64; GL_N];
    let mut ws = [0.0f64; GL_N];
    let n = GL_N as f64;
    for i in 0..GL_N {
        let mut z = (PI * (i as f64 + 0.75) / (n + 0.5)).cos();
        let mut dp = 1.0f64;
        for _ in 0..100 {
            let (mut p0, mut p1) = (1.0f64, 0.0f64);
            for j in 0..GL_N {
                let p2 = p1;
                p1 = p0;
                let jf = j as f64;
                p0 = ((2.0 * jf + 1.0) * z * p1 - jf * p2) / (jf + 1.0);
            }
            dp = n * (z * p0 - p1) / (z * z - 1.0);
            let step = p0 / dp;
            z -= step;
            if step.abs() < 1e-16 {
                break;
            }
        }
        xs[i] = z;
        ws[i] = 2.0 / ((1.0 - z * z) * dp * dp);
    }
    (xs, ws)
}

fn gl_panel<F: Fn(f64) -> f64>(a: f64, b: f64, xs: &[f64; GL_N], ws: &[f64; GL_N], f: F) -> f64 {
    let c = 0.5 * (a + b);
    let h = 0.5 * (b - a);
    let mut s = 0.0;
    for k in 0..GL_N {
        s += ws[k] * f(c + h * xs[k]);
    }
    s * h
}

/// The Siegert integral `integral from p to q of erfcx(w) dw`, the whole content of the
/// first-passage time.
///
/// `p = (mu - v_th) / sigma` and `q = (mu - v_reset) / sigma`, so `q > p` always. The two halves of
/// the axis are integrated differently because they are different problems:
///
/// - **`w > 0` (supra-threshold, the mean carries the neuron over).** `erfcx` is bounded by 1 and
///   decays as `1 / (w sqrt(pi))`, so the integral grows only logarithmically. Panels widen
///   geometrically (`width = max(0.5, w/2)`), which keeps the integrand's variation per panel
///   bounded and reaches `w = 1e9` in about 50 panels.
/// - **`w < 0` (sub-threshold, only noise fires the neuron).** The integrand is
///   `2 exp(s^2) - erfcx(s)` with `s = -w`, concentrated at the far end with width `1/(2s)`.
///   Substituting `x = s^2` turns that into `exp(x)` on a uniform grid, which uniform panels
///   integrate to rounding.
///
/// `None` for non-finite arguments or `q < p`. Returns [`f64::INFINITY`] where the true value
/// exceeds [`f64::MAX`], which happens below about `p = -26.5`; the corresponding firing rate there
/// is under `1e-290` Hz.
#[must_use]
pub fn siegert_integral(p: f64, q: f64) -> Option<f64> {
    if !p.is_finite() || !q.is_finite() || q < p {
        return None;
    }
    if q == p {
        return Some(0.0);
    }
    let (xs, ws) = gauss_legendre();
    let mut total = 0.0f64;
    if q > 0.0 {
        let (mut a, b) = (p.max(0.0), q);
        while a < b {
            let next = (a + (0.5f64).max(0.5 * a)).min(b);
            total += gl_panel(a, next, &xs, &ws, erfcx);
            a = next;
        }
    }
    if p < 0.0 {
        let (lo, hi) = (-q.min(0.0), -p); // 0 <= lo < hi, integrating in s = -w
        if hi > 26.7 {
            return Some(f64::INFINITY);
        }
        let split = hi.min(1.0).max(lo);
        let mut s = lo;
        while s < split {
            let next = (s + 0.5).min(split);
            total += gl_panel(s, next, &xs, &ws, |s| 2.0 * (s * s).exp() - erfcx(s));
            s = next;
        }
        if hi > split {
            let g = |x: f64| {
                let s = x.sqrt();
                (2.0 * x.exp() - erfcx(s)) / (2.0 * s)
            };
            let (mut x, x1) = (split * split, hi * hi);
            while x < x1 {
                let next = (x + 2.0).min(x1);
                total += gl_panel(x, next, &xs, &ws, g);
                x = next;
            }
        }
    }
    Some(total)
}

// ---------------------------------------------------------------------------------------------
// The diffusion approximation and the Siegert formula
// ---------------------------------------------------------------------------------------------

/// An integrate-and-fire neuron under stationary Gaussian white-noise input, and its exact firing
/// rate.
///
/// # The formula
///
/// For `tau_m dV/dt = -(V - mu) + sigma sqrt(tau_m) eta(t)` with an absorbing boundary at `v_th`,
/// reinjection at `v_reset` and a dead time `t_ref`, the mean interval between spikes is
///
/// ```text
/// T = t_ref + tau_m sqrt(pi) * integral from (v_reset-mu)/sigma to (v_th-mu)/sigma of
///             exp(u^2) (1 + erf(u)) du
/// ```
///
/// Brunel, *Dynamics of sparsely connected networks of excitatory and inhibitory spiking neurons*,
/// J. Comput. Neurosci. 8:183-208, 2000, eq. (22); the first-passage result itself is older —
/// Siegert, Phys. Rev. 81:617, 1951, and Ricciardi, *Diffusion Processes and Related Topics in
/// Biology*, 1977. The integral is [`siegert_integral`], after the substitution `w = -u` that turns
/// `exp(u^2)(1 + erf(u))` into `erfcx(w)`.
///
/// # Why the rate is not just `Lif::isi`
///
/// Set `sigma = 0` and it is: [`SiegertInput::mean_interval`] returns exactly what
/// [`Lif::isi`] returns, bit for bit, and the test `the_zero_noise_limit_is_exactly_lif_isi`
/// asserts that with no tolerance at all. Noise changes the answer in both directions and the
/// difference is not small:
///
/// - **Above threshold** noise *speeds the neuron up* slightly, because the trajectory that reaches
///   threshold early is not compensated by the one that reaches it late — the boundary is absorbing,
///   so late trajectories keep integrating while early ones have already fired.
/// - **Below threshold** noise is the *only* thing that fires the neuron. `Lif::isi` returns `None`
///   there, correctly, and this returns a finite rate. At `mu = 15 mV` against a 20 mV threshold
///   with `sigma = 5 mV`, that rate is 9.5 Hz — the difference between a silent network and a
///   cortical one.
///
/// That second case is the whole reason the formula exists, and it is why cortex can run in the
/// **fluctuation-driven** regime: mean input below threshold, firing carried entirely by variance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SiegertInput {
    /// Membrane time constant, seconds. Strictly positive.
    pub tau_m: f64,
    /// Absolute refractory period, seconds. Zero or more; it adds to the interval unconditionally
    /// and so caps the rate at `1 / t_ref`.
    pub t_ref: f64,
    /// Firing threshold, volts. Must exceed `v_reset`.
    pub v_th: f64,
    /// Post-spike potential, volts.
    pub v_reset: f64,
    /// Asymptotic mean membrane potential in the **absence of the threshold**, volts. This is where
    /// the free membrane would settle, not where the firing membrane sits.
    pub mu: f64,
    /// Input noise amplitude, volts, in Brunel's convention: the coefficient of
    /// `sqrt(tau_m) eta(t)` in the membrane equation. The stationary standard deviation of the free
    /// membrane potential is `sigma / sqrt(2)`, not `sigma` — see [`Self::free_membrane_sd`].
    /// Zero or more; zero is the deterministic case.
    pub sigma: f64,
}

impl SiegertInput {
    /// Build from explicit parameters, rejecting anything the formula is not defined on.
    ///
    /// # Errors
    ///
    /// [`MeanFieldError::NonFinite`] for any non-finite argument; [`MeanFieldError::OutOfRange`]
    /// for `tau_m <= 0`, `t_ref < 0` or `sigma < 0`; [`MeanFieldError::Ordering`] when
    /// `v_th <= v_reset`, which would make the integration interval empty or reversed and the
    /// "interval" a negative number.
    pub fn new(
        tau_m: f64,
        t_ref: f64,
        v_th: f64,
        v_reset: f64,
        mu: f64,
        sigma: f64,
    ) -> Result<Self, MeanFieldError> {
        let tau_m = finite("tau_m", tau_m)?;
        let t_ref = finite("t_ref", t_ref)?;
        let v_th = finite("v_th", v_th)?;
        let v_reset = finite("v_reset", v_reset)?;
        let mu = finite("mu", mu)?;
        let sigma = finite("sigma", sigma)?;
        if tau_m <= 0.0 {
            return Err(MeanFieldError::OutOfRange {
                what: "tau_m",
                value: tau_m,
                low: f64::MIN_POSITIVE,
                high: f64::MAX,
            });
        }
        if t_ref < 0.0 {
            return Err(MeanFieldError::OutOfRange {
                what: "t_ref",
                value: t_ref,
                low: 0.0,
                high: f64::MAX,
            });
        }
        if sigma < 0.0 {
            return Err(MeanFieldError::OutOfRange {
                what: "sigma",
                value: sigma,
                low: 0.0,
                high: f64::MAX,
            });
        }
        if v_th <= v_reset {
            return Err(MeanFieldError::Ordering {
                larger: "v_th",
                smaller: "v_reset",
                larger_value: v_th,
                smaller_value: v_reset,
            });
        }
        Ok(Self { tau_m, t_ref, v_th, v_reset, mu, sigma })
    }

    /// Take the membrane parameters from an existing [`Lif`] and supply the drive in volts.
    ///
    /// `mu` is the asymptotic mean potential in absolute volts, on the same scale as `lif.v_th`;
    /// it is **not** an offset from rest. Passing `lif.v_rest` gives an undriven neuron.
    ///
    /// # Errors
    ///
    /// As [`Self::new`].
    pub fn from_lif(lif: &Lif, mu: f64, sigma: f64) -> Result<Self, MeanFieldError> {
        Self::new(lif.tau_m, lif.t_ref, lif.v_th, lif.v_reset, mu, sigma)
    }

    /// Take the drive as a **current** rather than a voltage, through the neuron's own resistance.
    ///
    /// `i_mean` is amperes. `i_noise` is the white-noise amplitude density in **amperes per root
    /// hertz** (equivalently A·s^(1/2)), which is what a current noise spectral density is quoted
    /// in; it is not an r.m.s. current, because white noise has no finite r.m.s. without a
    /// bandwidth. The conversion is `mu = v_rest + r_m i_mean` and
    /// `sigma = r_m i_noise / sqrt(tau_m)`, the second of which is where the `sqrt(tau_m)` in the
    /// membrane equation goes.
    ///
    /// # Errors
    ///
    /// As [`Self::new`], plus [`MeanFieldError::NonFinite`] for a non-finite current.
    pub fn from_current(lif: &Lif, i_mean: f64, i_noise: f64) -> Result<Self, MeanFieldError> {
        let i_mean = finite("i_mean", i_mean)?;
        let i_noise = finite("i_noise", i_noise)?;
        if i_noise < 0.0 {
            return Err(MeanFieldError::OutOfRange {
                what: "i_noise",
                value: i_noise,
                low: 0.0,
                high: f64::MAX,
            });
        }
        let mu = lif.v_rest + lif.r_m * i_mean;
        let sigma = lif.r_m * i_noise / lif.tau_m.sqrt();
        Self::new(lif.tau_m, lif.t_ref, lif.v_th, lif.v_reset, mu, sigma)
    }

    /// Stationary standard deviation of the free membrane potential, volts: `sigma / sqrt(2)`.
    ///
    /// Here because the factor is the single most common way a mean-field calculation comes out
    /// wrong by tens of percent while every intermediate quantity still looks plausible. If you
    /// measured a membrane's fluctuation with the threshold removed, **this** is the number you
    /// measured, and `sigma` is `sqrt(2)` times it.
    #[must_use]
    pub fn free_membrane_sd(&self) -> f64 {
        self.sigma / core::f64::consts::SQRT_2
    }

    /// `(v_th - mu) / sigma`: how many noise units the mean sits below threshold.
    ///
    /// The x-axis of every figure in this literature. Positive is fluctuation-driven (the mean alone
    /// never fires the neuron), negative is mean-driven, and the crossover at zero is where a
    /// cortical network is usually claimed to sit. `None` when `sigma == 0`, where the quantity is
    /// a division by zero rather than a large number.
    #[must_use]
    pub fn threshold_distance(&self) -> Option<f64> {
        if self.sigma <= 0.0 { None } else { Some((self.v_th - self.mu) / self.sigma) }
    }

    /// Whether the mean input alone reaches threshold.
    ///
    /// The regime boundary, and the one place where `Lif::isi` and this formula agree on the
    /// question and disagree on the answer: below it `Lif::isi` says "never" and this says "rarely".
    #[must_use]
    pub fn is_mean_driven(&self) -> bool {
        self.mu > self.v_th
    }

    /// Mean interval between spikes, seconds.
    ///
    /// `None` only in the one case where there is no interval: `sigma == 0` and `mu <= v_th`, a
    /// deterministic neuron that never reaches threshold. That is the same refusal
    /// [`Lif::isi`] makes, for the same reason — "fires rarely" and "does not fire" are different
    /// statements and a rate-coded readout cannot tell them apart later.
    ///
    /// [`f64::INFINITY`] when the integral exceeds [`f64::MAX`], which happens below about
    /// `(mu - v_th) / sigma = -26.5`. The true interval there is past `1e290` seconds.
    #[must_use]
    pub fn mean_interval(&self) -> Option<f64> {
        if self.sigma == 0.0 {
            // The zero-noise limit, written with exactly the expression `Lif::isi` uses so the two
            // agree bit for bit rather than to a tolerance. This is a branch and not a limit of the
            // quadrature, so `the_siegert_interval_converges_to_the_deterministic_one` checks the
            // quadrature approaches it as sigma shrinks — an exact equality on a special case is
            // not evidence that the general case is right.
            if self.mu <= self.v_th {
                return None;
            }
            let t = self.tau_m * ((self.mu - self.v_reset) / (self.mu - self.v_th)).ln();
            return Some(t + self.t_ref);
        }
        let p = (self.mu - self.v_th) / self.sigma;
        let q = (self.mu - self.v_reset) / self.sigma;
        let j = siegert_integral(p, q)?;
        Some(self.t_ref + self.tau_m * PI.sqrt() * j)
    }

    /// Stationary firing rate, hertz. `None` under the same condition as [`Self::mean_interval`].
    ///
    /// Exactly `0.0` where the interval is infinite, which is the correct limit and is reported as
    /// a rate rather than a refusal because the neuron does have a rate — it is just below anything
    /// representable.
    #[must_use]
    pub fn rate(&self) -> Option<f64> {
        self.mean_interval().map(|t| 1.0 / t)
    }
}

/// Simulate the very neuron [`SiegertInput`] describes, with a [`Lif`] and real pseudorandom noise.
///
/// This is the reference the closed form is checked against, and it is written to be an **exact**
/// sampler of the underlying Ornstein-Uhlenbeck process rather than an Euler-Maruyama
/// approximation of it. Two things make that work:
///
/// 1. [`Lif::step`] integrates by exponential Euler, which is the exact solution over a step of
///    constant input — so the deterministic part carries no discretisation error at all.
/// 2. The noise is injected with [`Neuron::bump`] **before** the step and pre-scaled by
///    `exp(dt / tau_m)`, which exactly cancels the decay the step then applies to it. The result is
///    the exact `V(t + dt) = mu + (V(t) - mu) e^(-dt/tau) + N(0, sigma^2 (1 - e^(-2dt/tau)) / 2)`
///    recursion, with the threshold tested on the true `V(t + dt)`.
///
/// # What error remains, and its sign
///
/// One error survives and it cannot be removed by a better integrator: **a trajectory that crosses
/// threshold between two ticks and comes back down is not seen**. Every missed crossing is a
/// missing spike, so the simulated rate is biased **low**, by `O(sqrt(dt))` (Gobet, Stochastic
/// Process. Appl. 87:167, 2000, for the weak error of a killed diffusion). Measured here at
/// `dt / tau_m = 5e-4`: 0.6% in the mean-driven regime and 6% at `CV = 1`. Halving `dt` is the only
/// fix and it costs twice the run.
///
/// # Errors
///
/// [`MeanFieldError::OutOfRange`] for `dt <= 0`; [`MeanFieldError::NonFinite`] for a non-finite
/// `dt`.
///
/// # Panics
///
/// Never. The train is built forward in tick order, which is what [`Train::push`] requires.
pub fn simulate_diffusion(
    input: &SiegertInput,
    dt: f64,
    ticks: u64,
    rng: &mut Rng,
) -> Result<Train, MeanFieldError> {
    let dt = finite("dt", dt)?;
    if dt <= 0.0 {
        return Err(MeanFieldError::OutOfRange {
            what: "dt",
            value: dt,
            low: f64::MIN_POSITIVE,
            high: f64::MAX,
        });
    }
    // r_m = 1 and v_rest = mu, with zero input current: the neuron's own asymptote is the drive.
    let mut n = Lif {
        tau_m: input.tau_m,
        v_rest: input.mu,
        v_th: input.v_th,
        v_reset: input.v_reset,
        r_m: 1.0,
        t_ref: input.t_ref,
        v: input.v_reset,
        refractory: 0.0,
    };
    let step_sd = ou_step_sd(input.sigma, dt, input.tau_m);
    let mut normal = BoxMuller::new();
    let mut train = Train::new();
    for k in 0..ticks {
        n.bump(normal.draw(rng) * step_sd);
        if n.step(dt, 0.0) {
            train.push(Spike { t: k, source: 0 });
        }
    }
    Ok(train)
}

/// Standard deviation of the noise [`simulate_diffusion`] injects per tick, volts.
///
/// The exact Ornstein-Uhlenbeck increment is `N(0, sigma^2 (1 - exp(-2 dt/tau)) / 2)`, and the
/// `exp(dt/tau)` factor pre-compensates the decay that [`Lif::step`] applies to a displacement made
/// before it. Both simulators here go through this one function so that a change to the convention
/// cannot reach one of them and not the other — and so that
/// `the_free_membrane_has_the_exact_ornstein_uhlenbeck_variance_and_autocorrelation`, which measures
/// the resulting variance against `sigma^2 / 2`, constrains the spiking simulator too.
fn ou_step_sd(sigma: f64, dt: f64, tau_m: f64) -> f64 {
    let ratio = dt / tau_m;
    sigma * ((1.0 - (-2.0 * ratio).exp()) / 2.0).sqrt() * ratio.exp()
}

/// The same membrane with the **threshold removed**: the free Ornstein-Uhlenbeck process, sampled.
///
/// Returns the potential in volts at every tick, started at `mu` so there is no relaxation
/// transient. Two exact facts make this the instrument for checking a noise convention, which is
/// the single easiest thing in this literature to get wrong by a factor of `sqrt(2)`:
///
/// - the stationary variance is `sigma^2 / 2` — **not** `sigma^2`;
/// - the autocorrelation at lag `k` ticks is exactly `exp(-k dt / tau_m)`, with no dependence on
///   `sigma` at all.
///
/// If you were handed a `sigma` and are not sure which convention it is in, run this, take the
/// variance, and compare. It shares [`ou_step_sd`] with [`simulate_diffusion`], so a measurement
/// here is a measurement of the spiking simulator's noise as well.
///
/// # Errors
///
/// [`MeanFieldError::OutOfRange`] for `dt <= 0`; [`MeanFieldError::NonFinite`] for a non-finite
/// `dt`.
pub fn simulate_free_membrane(
    input: &SiegertInput,
    dt: f64,
    ticks: u64,
    rng: &mut Rng,
) -> Result<Vec<f64>, MeanFieldError> {
    let dt = finite("dt", dt)?;
    if dt <= 0.0 {
        return Err(MeanFieldError::OutOfRange {
            what: "dt",
            value: dt,
            low: f64::MIN_POSITIVE,
            high: f64::MAX,
        });
    }
    // An infinite threshold is what "free" means here, and it goes through the same `Lif::step`
    // the spiking simulator uses rather than a re-derivation of the same recursion.
    let mut n = Lif {
        tau_m: input.tau_m,
        v_rest: input.mu,
        v_th: f64::INFINITY,
        v_reset: input.v_reset,
        r_m: 1.0,
        t_ref: 0.0,
        v: input.mu,
        refractory: 0.0,
    };
    let step_sd = ou_step_sd(input.sigma, dt, input.tau_m);
    let mut normal = BoxMuller::new();
    let mut out = Vec::with_capacity(ticks as usize);
    for _ in 0..ticks {
        n.bump(normal.draw(rng) * step_sd);
        n.step(dt, 0.0);
        out.push(n.potential());
    }
    Ok(out)
}

/// Box-Muller normal deviates from [`Rng`], with the second of each pair cached.
///
/// Private because the crate's public randomness is [`Rng`] and adding a second entry point would
/// invite two conventions for "a seeded normal". Box-Muller rather than the ziggurat because it is
/// twelve lines that can be read against the textbook, and the cost is two transcendentals per
/// pair, which is nothing beside the network step it feeds.
struct BoxMuller {
    spare: Option<f64>,
}

impl BoxMuller {
    fn new() -> Self {
        Self { spare: None }
    }

    fn draw(&mut self, rng: &mut Rng) -> f64 {
        if let Some(s) = self.spare.take() {
            return s;
        }
        // `1 - u` rather than `u`: Rng::next_f64 can return exactly 0.0, and ln(0) is -inf.
        let u1 = 1.0 - rng.next_f64();
        let u2 = rng.next_f64();
        let m = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * PI * u2;
        self.spare = Some(m * theta.sin());
        m * theta.cos()
    }
}

// ---------------------------------------------------------------------------------------------
// Balanced networks
// ---------------------------------------------------------------------------------------------

/// The shot-noise drive one neuron receives in a sparse recurrent network, and the diffusion
/// approximation of it.
///
/// Each neuron gets `c_exc` recurrent excitatory synapses of efficacy `j` volts, `c_inh` inhibitory
/// ones of efficacy `-g j`, and `c_exc` external excitatory ones also of efficacy `j`. Recurrent
/// sources fire at `nu`, external ones at `nu_ext`, all Poisson and independent. Summing the first
/// two moments of that compound Poisson drive gives
///
/// ```text
/// mu     = tau_m j (c_exc (nu + nu_ext) - g c_inh nu)
/// sigma^2 = tau_m j^2 (c_exc (nu + nu_ext) + g^2 c_inh nu)
/// ```
///
/// (Brunel 2000 eqs. 5-6.) Note the asymmetry that is the whole subject: inhibition **subtracts**
/// from the mean and **adds** to the variance. That is what makes balance possible and what makes
/// it interesting — cancelling the mean does not cancel the noise, it doubles it.
///
/// `mu` here is an offset **above the resting potential**, because that is the quantity the input
/// produces; add `v_rest` to put it on the same scale as `v_th`. [`Self::siegert`] does that for
/// you and is the safer route.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BalancedInput {
    /// Membrane time constant, seconds. Appears in both moments because the membrane integrates the
    /// drive over its own window and nothing longer.
    pub tau_m: f64,
    /// Number of recurrent excitatory synapses onto one neuron, and also the number of external
    /// ones, following Brunel's convention that the external drive matches the recurrent fan-in.
    pub c_exc: f64,
    /// Number of recurrent inhibitory synapses onto one neuron.
    pub c_inh: f64,
    /// Ratio of inhibitory to excitatory efficacy, dimensionless and non-negative. The inhibitory
    /// synapse is `-g * j` volts. Brunel's `g`.
    pub g: f64,
    /// Excitatory synaptic efficacy, **volts per presynaptic spike** — the membrane displacement one
    /// spike causes, which is the argument [`Neuron::bump`] takes.
    pub j: f64,
    /// Recurrent firing rate of every neuron in the network, hertz.
    pub nu: f64,
    /// Firing rate of each external synapse, hertz.
    pub nu_ext: f64,
}

impl BalancedInput {
    /// `c_inh / c_exc`, the connectivity ratio Brunel calls `gamma`. Typically 0.25, because cortex
    /// is about 80% excitatory.
    ///
    /// `None` when `c_exc == 0`, where the ratio is a division by zero and the network has no
    /// excitatory drive to balance.
    #[must_use]
    pub fn gamma(&self) -> Option<f64> {
        if self.c_exc == 0.0 { None } else { Some(self.c_inh / self.c_exc) }
    }

    /// `g * gamma`, the number that decides everything.
    ///
    /// Below 1 the recurrent loop is net excitatory and the network runs away to saturation; above 1
    /// it is net inhibitory and the recurrent term is a negative feedback on the rate; **at exactly
    /// 1 the recurrent contribution to `mu` vanishes identically**, whatever `nu` is, and the mean
    /// drive is the external drive alone. That last statement is an algebraic identity, not an
    /// approximation, and `the_balance_line_cancels_the_recurrent_mean_exactly` asserts it as one.
    ///
    /// `None` when `c_exc == 0`.
    #[must_use]
    pub fn balance_index(&self) -> Option<f64> {
        self.gamma().map(|gam| self.g * gam)
    }

    /// Mean drive, **volts above rest**.
    #[must_use]
    pub fn mu(&self) -> f64 {
        self.tau_m
            * self.j
            * (self.c_exc * (self.nu + self.nu_ext) - self.g * self.c_inh * self.nu)
    }

    /// Input noise amplitude in Brunel's convention, volts. Always non-negative.
    #[must_use]
    pub fn sigma(&self) -> f64 {
        let var = self.tau_m
            * self.j
            * self.j
            * (self.c_exc * (self.nu + self.nu_ext) + self.g * self.g * self.c_inh * self.nu);
        if var <= 0.0 { 0.0 } else { var.sqrt() }
    }

    /// The [`SiegertInput`] this drive produces for a given membrane, with `mu` put on the
    /// absolute voltage scale by adding `lif.v_rest`.
    ///
    /// # Errors
    ///
    /// As [`SiegertInput::new`]. The `tau_m` used is this struct's, not the neuron's, so that a
    /// mismatch between the two is visible as a wrong rate rather than silently resolved.
    pub fn siegert(&self, lif: &Lif) -> Result<SiegertInput, MeanFieldError> {
        SiegertInput::new(
            self.tau_m,
            lif.t_ref,
            lif.v_th,
            lif.v_reset,
            lif.v_rest + self.mu(),
            self.sigma(),
        )
    }

    /// Run the **microscopic** drive — actual Poisson spike counts, actual voltage steps — through a
    /// [`Lif`] and return the spike train.
    ///
    /// This is the thing the diffusion approximation approximates. Each tick draws
    /// `Poisson(c_exc (nu + nu_ext) dt)` excitatory arrivals and `Poisson(c_inh nu dt)` inhibitory
    /// ones and displaces the membrane by `j (n_exc - g n_inh)`. Nothing here is Gaussian; the
    /// central limit theorem is what makes it look Gaussian, and it only does so when the expected
    /// counts per membrane time constant are large.
    ///
    /// Use it to see the balance argument work: with `g * gamma == 1` and a sub-threshold external
    /// drive the resulting train has a coefficient of variation near 1, and with `g == 0` the same
    /// excitatory drive produces a near-metronomic one.
    ///
    /// # Errors
    ///
    /// [`MeanFieldError::OutOfRange`] for `dt <= 0` or a negative rate or count;
    /// [`MeanFieldError::NonFinite`] for a non-finite parameter.
    ///
    /// # Panics
    ///
    /// Never. Spikes are pushed in tick order, which is [`Train::push`]'s requirement.
    pub fn simulate(
        &self,
        lif: &Lif,
        dt: f64,
        ticks: u64,
        rng: &mut Rng,
    ) -> Result<Train, MeanFieldError> {
        let dt = finite("dt", dt)?;
        if dt <= 0.0 {
            return Err(MeanFieldError::OutOfRange {
                what: "dt",
                value: dt,
                low: f64::MIN_POSITIVE,
                high: f64::MAX,
            });
        }
        for (what, v) in [
            ("c_exc", self.c_exc),
            ("c_inh", self.c_inh),
            ("j", self.j),
            ("g", self.g),
            ("nu", self.nu),
            ("nu_ext", self.nu_ext),
        ] {
            let v = finite(what, v)?;
            if v < 0.0 {
                return Err(MeanFieldError::OutOfRange {
                    what,
                    value: v,
                    low: 0.0,
                    high: f64::MAX,
                });
            }
        }
        let lambda_exc = self.c_exc * (self.nu + self.nu_ext) * dt;
        let lambda_inh = self.c_inh * self.nu * dt;
        // Checked here rather than left to `poisson_count`'s own refusal inside the loop, where an
        // `unwrap_or(0)` would turn "this mean is past what a u64 count can hold" into "no input
        // arrived", and the run would look like a silent network rather than like a rejected
        // parameter. 1e12 arrivals in one tick is already past any physical fan-in.
        for (what, lambda) in [("c_exc * (nu + nu_ext) * dt", lambda_exc), ("c_inh * nu * dt", lambda_inh)]
        {
            if !lambda.is_finite() || lambda > 1e12 {
                return Err(MeanFieldError::OutOfRange {
                    what,
                    value: lambda,
                    low: 0.0,
                    high: 1e12,
                });
            }
        }
        let mut n = *lif;
        let mut train = Train::new();
        for k in 0..ticks {
            let ne = crate::coding::poisson_count(rng, lambda_exc).unwrap_or(0);
            let ni = crate::coding::poisson_count(rng, lambda_inh).unwrap_or(0);
            n.bump(self.j * (ne as f64 - self.g * ni as f64));
            if n.step(dt, 0.0) {
                train.push(Spike { t: k, source: 0 });
            }
        }
        Ok(train)
    }
}

/// The two-population balance equations of van Vreeswijk & Sompolinsky, Science 274:1724, 1996.
///
/// # The argument
///
/// Let every neuron receive `K` synapses of each kind, with efficacies scaled as `J / sqrt(K)`. Then
/// the **mean** input is `sqrt(K) * (J_EE nu_E - J_EI nu_I + J_E0 nu_0)` — it grows without bound —
/// while the **variance** is `J^2 K / K = J^2`, which does not. For firing rates to stay finite as
/// `K` grows, the bracket must vanish as `1 / sqrt(K)`, and then the residual input is a mean of
/// order 1 riding on a fluctuation of order 1. Irregular firing is not imposed; it is what is left
/// when the large terms cancel.
///
/// The consequence that made the paper famous: to leading order in `K` the rates solve a **linear**
/// system, so they do not depend on the neuron model at all. Replace the integrate-and-fire cell
/// with anything monotone and the balanced rates are unchanged. [`Self::balanced_rates`] solves that
/// system, and `sigma` is exactly independent of `K` — `the_fluctuation_is_exactly_independent_of_k`
/// asserts it at four decades of `K`.
///
/// # Units
///
/// The `j_*` are volts per spike at `K = 1`, i.e. the `J_0` before the `1 / sqrt(K)` scaling. Rates
/// are hertz. `tau_m` cancels out of the balance condition entirely and so does not appear.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BalanceMatrix {
    /// Excitatory-to-excitatory efficacy, volts per spike at `K = 1`. Positive.
    pub j_ee: f64,
    /// Inhibitory-to-excitatory efficacy **magnitude**, volts per spike at `K = 1`. Positive; the
    /// sign is carried by the equations, not by this field.
    pub j_ei: f64,
    /// Excitatory-to-inhibitory efficacy, volts per spike at `K = 1`. Positive.
    pub j_ie: f64,
    /// Inhibitory-to-inhibitory efficacy magnitude, volts per spike at `K = 1`. Positive.
    pub j_ii: f64,
    /// External-to-excitatory efficacy, volts per spike at `K = 1`. Positive.
    pub j_e0: f64,
    /// External-to-inhibitory efficacy, volts per spike at `K = 1`. Positive.
    pub j_i0: f64,
}

impl BalanceMatrix {
    /// The rates `(nu_exc, nu_inh)` in hertz at which the `O(sqrt(K))` drive cancels for both
    /// populations.
    ///
    /// Solves
    ///
    /// ```text
    /// j_ee nu_E - j_ei nu_I + j_e0 nu_0 = 0
    /// j_ie nu_E - j_ii nu_I + j_i0 nu_0 = 0
    /// ```
    ///
    /// by Cramer's rule. **No neuron model appears**, which is the point.
    ///
    /// # Errors
    ///
    /// [`MeanFieldError::NonFinite`] for a non-finite argument or field.
    /// [`MeanFieldError::Degenerate`] when the determinant is below `1e-12` of the largest entry
    /// product, where the two equations say the same thing and the rates are a ratio of two
    /// roundings. [`MeanFieldError::NoBalancedState`] when the solution has a non-positive rate:
    /// that is van Vreeswijk & Sompolinsky's existence condition failing, and it means the network
    /// has no balanced state rather than that the arithmetic went wrong.
    pub fn balanced_rates(&self, nu_ext: f64) -> Result<(f64, f64), MeanFieldError> {
        let nu_ext = finite("nu_ext", nu_ext)?;
        for (what, v) in [
            ("j_ee", self.j_ee),
            ("j_ei", self.j_ei),
            ("j_ie", self.j_ie),
            ("j_ii", self.j_ii),
            ("j_e0", self.j_e0),
            ("j_i0", self.j_i0),
        ] {
            finite(what, v)?;
        }
        // [ j_ee  -j_ei ] [nu_E]   [ -j_e0 nu_0 ]
        // [ j_ie  -j_ii ] [nu_I] = [ -j_i0 nu_0 ]
        let det = -self.j_ee * self.j_ii + self.j_ei * self.j_ie;
        let scale = (self.j_ee * self.j_ii).abs().max((self.j_ei * self.j_ie).abs()).max(1e-300);
        if det.abs() <= 1e-12 * scale {
            return Err(MeanFieldError::Degenerate { determinant: det, scale });
        }
        let (b0, b1) = (-self.j_e0 * nu_ext, -self.j_i0 * nu_ext);
        let nu_exc = (b0 * -self.j_ii - -self.j_ei * b1) / det;
        let nu_inh = (self.j_ee * b1 - self.j_ie * b0) / det;
        if !(nu_exc > 0.0) || !(nu_inh > 0.0) {
            return Err(MeanFieldError::NoBalancedState { nu_exc, nu_inh });
        }
        Ok((nu_exc, nu_inh))
    }

    /// How far a pair of rates is from balance, in volts per spike per second, for each population.
    ///
    /// Zero for both entries exactly at the balanced solution. This is the quantity that is
    /// multiplied by `sqrt(K)` to give the mean drive, so a residual of `r` produces a mean input of
    /// `tau_m sqrt(K) r` — which is why it has to go to zero as `K` grows rather than merely be
    /// small.
    #[must_use]
    pub fn residual(&self, nu_exc: f64, nu_inh: f64, nu_ext: f64) -> (f64, f64) {
        (
            self.j_ee * nu_exc - self.j_ei * nu_inh + self.j_e0 * nu_ext,
            self.j_ie * nu_exc - self.j_ii * nu_inh + self.j_i0 * nu_ext,
        )
    }

    /// Mean drive to the excitatory population at finite `K`, volts: `tau_m sqrt(K) * residual_E`.
    ///
    /// `None` for `k == 0` or a non-finite `tau_m`. This is the term that diverges: at `K = 10 000`
    /// an unbalanced residual of one millivolt-per-spike-per-second becomes a 100x larger drive than
    /// at `K = 1`, which is why a network that is merely *approximately* balanced at small `K` is
    /// saturated at large `K`.
    #[must_use]
    pub fn mean_drive_exc(&self, nu_exc: f64, nu_inh: f64, nu_ext: f64, k: u32, tau_m: f64) -> Option<f64> {
        if k == 0 || !tau_m.is_finite() {
            return None;
        }
        let (r, _) = self.residual(nu_exc, nu_inh, nu_ext);
        Some(tau_m * f64::from(k).sqrt() * r)
    }

    /// Input noise amplitude to the excitatory population, volts, in Brunel's convention.
    ///
    /// `sigma^2 = tau_m (j_ee^2 nu_E + j_ei^2 nu_I + j_e0^2 nu_0)`. **`K` does not appear**: the
    /// `K` synapses each carry `J / sqrt(K)`, and `K * (J / sqrt(K))^2 = J^2`. That exact
    /// cancellation is the reason a balanced network's fluctuation survives the large-`K` limit
    /// while its mean does not, and it is asserted rather than described in
    /// `the_fluctuation_is_exactly_independent_of_k`.
    ///
    /// `None` for a non-finite argument or a negative rate.
    #[must_use]
    pub fn fluctuation_exc(&self, nu_exc: f64, nu_inh: f64, nu_ext: f64, tau_m: f64) -> Option<f64> {
        if !nu_exc.is_finite() || !nu_inh.is_finite() || !nu_ext.is_finite() || !tau_m.is_finite() {
            return None;
        }
        if nu_exc < 0.0 || nu_inh < 0.0 || nu_ext < 0.0 || tau_m < 0.0 {
            return None;
        }
        let var = tau_m
            * (self.j_ee * self.j_ee * nu_exc
                + self.j_ei * self.j_ei * nu_inh
                + self.j_e0 * self.j_e0 * nu_ext);
        Some(var.sqrt())
    }
}

// ---------------------------------------------------------------------------------------------
// Brunel's phase diagram
// ---------------------------------------------------------------------------------------------

/// The dynamical states of a sparse random network of excitatory and inhibitory
/// integrate-and-fire neurons.
///
/// Brunel 2000 Fig. 1 and Fig. 8. The two axes of the classification are independent: *synchrony*
/// asks whether neurons fire together, *regularity* asks whether each one fires like a clock. All
/// four combinations occur in the same network at different `(g, nu_ext)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Regime {
    /// Neurons fire together and each one fires like a clock. Excitation dominates, rates approach
    /// the refractory ceiling, and the population oscillates at close to the single-neuron rate.
    SynchronousRegular,
    /// Neurons fire together but each one irregularly, so the population rate oscillates while no
    /// individual cell has a reliable phase. The fast variant, at a frequency set by the synaptic
    /// delay rather than by any membrane constant, is the one that resembles cortical gamma.
    SynchronousIrregular,
    /// The state usually claimed for cortex: the population rate is flat in time and every neuron
    /// fires with a coefficient of variation near 1. Requires inhibition-dominated recurrence.
    AsynchronousIrregular,
    /// Every neuron a metronome at its own phase, so the population rate is flat while no
    /// individual train is.
    ///
    /// **Not one of the three labels Brunel's phase diagram carries.** It is here because
    /// [`classify`] takes two independent binary measurements and a two-by-two grid has four cells;
    /// reporting this corner as one of the other three would be inventing a classification the
    /// measurements do not support. It arises with heterogeneous drive, where each cell is
    /// mean-driven at a rate of its own.
    AsynchronousRegular,
    /// Neither the external drive nor the recurrence sustains firing. Not a dynamical state so much
    /// as its absence, and it is reported separately because a near-zero rate makes both synchrony
    /// and regularity undefined rather than small.
    NearlySilent,
}

/// Population rate below which [`classify`] reports [`Regime::NearlySilent`], hertz.
///
/// **This crate's convention, not a published boundary.** Brunel classified by eye from rasters. One
/// hertz is chosen because below it a 1-second window holds too few intervals to estimate a
/// coefficient of variation, so the other two axes of the classification stop meaning anything.
pub const SILENCE_RATE_HZ: f64 = 1.0;

/// Coefficient of variation above which [`classify`] calls a train irregular.
///
/// **This crate's convention.** 0.5 sits between a driven integrate-and-fire neuron's near-zero CV
/// and a Poisson process's 1.0, and it is the value most often used informally in the literature;
/// this implementation did not locate a principled derivation of any particular cut.
pub const IRREGULARITY_CV: f64 = 0.5;

/// Synchrony index above which [`classify`] calls a population synchronous.
///
/// **This crate's convention.** For `N` independent neurons [`synchrony`] returns about
/// `1 / sqrt(N)`, so 0.3 is well clear of the asynchronous value for any `N` above about 20 and
/// well below the 1.0 of a perfectly locked population.
pub const SYNCHRONY_INDEX: f64 = 0.3;

/// Put a measured population into one of the four boxes the two measurements define.
///
/// Takes the three numbers a simulation can actually produce: the mean single-neuron rate in hertz,
/// the mean coefficient of variation of the intervals ([`Train::cv`]), and the synchrony index
/// ([`synchrony`]). The thresholds are [`SILENCE_RATE_HZ`], [`IRREGULARITY_CV`] and
/// [`SYNCHRONY_INDEX`], and **they are this crate's conventions rather than the paper's** — Brunel
/// classified by looking at rasters, and this implementation did not locate published numerical cuts
/// for the boundaries.
///
/// `None` for a non-finite argument or a negative rate.
#[must_use]
pub fn classify(rate_hz: f64, cv: f64, synchrony_index: f64) -> Option<Regime> {
    if !rate_hz.is_finite() || !cv.is_finite() || !synchrony_index.is_finite() || rate_hz < 0.0 {
        return None;
    }
    if rate_hz < SILENCE_RATE_HZ {
        return Some(Regime::NearlySilent);
    }
    let irregular = cv > IRREGULARITY_CV;
    let synchronous = synchrony_index > SYNCHRONY_INDEX;
    Some(match (synchronous, irregular) {
        (true, true) => Regime::SynchronousIrregular,
        (true, false) => Regime::SynchronousRegular,
        (false, true) => Regime::AsynchronousIrregular,
        (false, false) => Regime::AsynchronousRegular,
    })
}

/// Golomb and Rinzel's synchrony index, computed from binned spike counts.
///
/// ```text
/// chi^2 = Var_t( population mean count ) / mean_i Var_t( count of neuron i )
/// ```
///
/// Golomb & Rinzel, Phys. Rev. E 48:4810, 1993; used in this form for spike counts by Brunel 2000
/// and Hansel & Sompolinsky. Two exact anchors make it readable as a number rather than a score:
///
/// - **Perfectly synchronous** (every neuron fires in exactly the same bins): the population mean
///   *is* each neuron's count, numerator equals denominator, `chi = 1` exactly.
/// - **Independent** neurons: the variance of a mean of `N` independent variables is `1 / N` of
///   theirs, so `chi = 1 / sqrt(N)` in expectation. At `N = 400` that is 0.05.
///
/// `sources` is the number of neurons the train could have come from, including silent ones —
/// passing the number that actually fired would inflate the index by dropping the zero-variance
/// denominators. `None` when there are fewer than two bins, no sources, or a zero denominator (every
/// neuron constant, so there is no variability to share).
///
/// # Panics
///
/// Never. `bin_ticks == 0` returns `None` before any division.
#[must_use]
pub fn synchrony(train: &Train, sources: u32, ticks: u64, bin_ticks: u64) -> Option<f64> {
    if sources == 0 || bin_ticks == 0 || ticks == 0 {
        return None;
    }
    let bins = (ticks / bin_ticks) as usize;
    if bins < 2 {
        return None;
    }
    let n = sources as usize;
    let mut counts = vec![0.0f64; n * bins];
    for s in train.spikes() {
        let b = (s.t / bin_ticks) as usize;
        let src = s.source as usize;
        if b < bins && src < n {
            counts[src * bins + b] += 1.0;
        }
    }
    let mut pop = vec![0.0f64; bins];
    for i in 0..n {
        for b in 0..bins {
            pop[b] += counts[i * bins + b];
        }
    }
    let inv_n = 1.0 / n as f64;
    for p in &mut pop {
        *p *= inv_n;
    }
    let var = |v: &[f64]| {
        let m = v.iter().sum::<f64>() / v.len() as f64;
        v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (v.len() as f64 - 1.0)
    };
    let num = var(&pop);
    let mut den = 0.0f64;
    for i in 0..n {
        den += var(&counts[i * bins..(i + 1) * bins]);
    }
    den *= inv_n;
    if !(den > 0.0) {
        return None;
    }
    Some((num / den).sqrt())
}

/// Brunel's sparse random network, as a set of parameters with closed-form consequences.
///
/// The network of the 2000 paper: `n_exc` excitatory and `n_inh` inhibitory integrate-and-fire
/// neurons, each drawing `c_exc` and `c_inh` presynaptic partners at random, plus `c_exc` external
/// Poisson inputs. Only the fan-in matters for the mean-field treatment, so the population sizes do
/// not appear — which is itself a result worth noticing: in this theory a network of 12 500 neurons
/// and one of 12 500 000 behave identically at the same connection probability.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BrunelNetwork {
    /// The single-neuron model, supplying `tau_m`, `v_th`, `v_reset`, `v_rest` and `t_ref`.
    pub neuron: Lif,
    /// Excitatory synapses per neuron, and also external synapses per neuron.
    pub c_exc: f64,
    /// Inhibitory synapses per neuron.
    pub c_inh: f64,
    /// Excitatory efficacy, volts per spike. Brunel's `J`, typically 0.1 mV.
    pub j: f64,
    /// Relative strength of inhibition. The inhibitory efficacy is `-g * j`.
    pub g: f64,
    /// External drive as a multiple of [`Self::nu_thr`]. Brunel's `nu_ext / nu_thr`, the horizontal
    /// axis of the phase diagram; 1.0 means the external input alone just reaches threshold.
    pub nu_ext_ratio: f64,
    /// Synaptic transmission delay, seconds. It does not enter the stationary rate at all, and it
    /// sets the frequency of the fast oscillation — see [`Self::fast_oscillation_band`].
    pub delay: f64,
}

impl BrunelNetwork {
    /// `c_inh / c_exc`. `None` when `c_exc == 0`.
    #[must_use]
    pub fn gamma(&self) -> Option<f64> {
        if self.c_exc == 0.0 { None } else { Some(self.c_inh / self.c_exc) }
    }

    /// `g * gamma`: below 1 excitation dominates the recurrent loop, above 1 inhibition does, and
    /// at exactly 1 the recurrent mean cancels identically. `None` when `c_exc == 0`.
    #[must_use]
    pub fn balance_index(&self) -> Option<f64> {
        self.gamma().map(|gam| self.g * gam)
    }

    /// The external rate per synapse at which the external drive alone brings the mean membrane
    /// potential exactly to threshold, hertz.
    ///
    /// `nu_thr = (v_th - v_rest) / (c_exc j tau_m)`. Brunel's normalising rate. With his parameters
    /// — 20 mV of threshold, 1000 synapses, 0.1 mV each, 20 ms — it is 10 Hz.
    ///
    /// `None` when the denominator is zero or the threshold is not above rest.
    #[must_use]
    pub fn nu_thr(&self) -> Option<f64> {
        let d = self.c_exc * self.j * self.neuron.tau_m;
        let num = self.neuron.v_th - self.neuron.v_rest;
        if d <= 0.0 || num <= 0.0 { None } else { Some(num / d) }
    }

    /// The drive one neuron receives when the whole network fires at `nu` hertz.
    ///
    /// `None` when [`Self::nu_thr`] is undefined or `nu` is not finite and non-negative.
    #[must_use]
    pub fn input(&self, nu: f64) -> Option<BalancedInput> {
        if !nu.is_finite() || nu < 0.0 {
            return None;
        }
        let nu_ext = self.nu_ext_ratio * self.nu_thr()?;
        Some(BalancedInput {
            tau_m: self.neuron.tau_m,
            c_exc: self.c_exc,
            c_inh: self.c_inh,
            g: self.g,
            j: self.j,
            nu,
            nu_ext,
        })
    }

    /// The transfer function `Phi(nu)`: the rate one neuron fires at when the network fires at
    /// `nu`. Hertz. `None` when the input is undefined or the Siegert formula refuses.
    #[must_use]
    pub fn transfer(&self, nu: f64) -> Option<f64> {
        self.input(nu)?.siegert(&self.neuron).ok()?.rate()
    }

    /// The self-consistent rate: the `nu` at which `Phi(nu) == nu`, hertz.
    ///
    /// Found by bisection on `[0, 1 / t_ref]`, which is guaranteed to bracket a root because
    /// `Phi(0) >= 0` and `Phi` can never exceed `1 / t_ref` — a neuron with a dead time cannot fire
    /// faster than its dead time allows, whatever the drive. That is why a refractory period is
    /// required rather than optional here.
    ///
    /// **Uniqueness is not claimed.** When `g * gamma < 1` the transfer function is increasing and
    /// can cross the diagonal three times; the root returned is then the one bisection converges to
    /// from this bracket, which is the lowest one at which `Phi - nu` changes sign downward. In the
    /// inhibition-dominated case (`g * gamma > 1`) the recurrent term is negative feedback and the
    /// root is unique in practice, though this implementation does not prove it.
    ///
    /// # Errors
    ///
    /// [`MeanFieldError::NoRefractoryPeriod`] when `t_ref <= 0`, because the bracket does not exist.
    /// [`MeanFieldError::OutOfRange`] when [`Self::nu_thr`] is undefined, which means the network's
    /// parameters do not define an external drive scale.
    pub fn self_consistent_rate(&self) -> Result<f64, MeanFieldError> {
        if !(self.neuron.t_ref > 0.0) {
            return Err(MeanFieldError::NoRefractoryPeriod);
        }
        if self.nu_thr().is_none() {
            return Err(MeanFieldError::OutOfRange {
                what: "c_exc * j * tau_m",
                value: self.c_exc * self.j * self.neuron.tau_m,
                low: f64::MIN_POSITIVE,
                high: f64::MAX,
            });
        }
        let f = |nu: f64| self.transfer(nu).map_or(-nu, |phi| phi - nu);
        let (mut lo, mut hi) = (0.0f64, 1.0 / self.neuron.t_ref);
        if f(lo) <= 0.0 {
            return Ok(0.0);
        }
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            if mid == lo || mid == hi {
                break;
            }
            if f(mid) > 0.0 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        Ok(0.5 * (lo + hi))
    }

    /// What the **closed-form** boundaries decide about the regime, or `None` where they do not.
    ///
    /// Two boundaries here have exact expressions and the third does not:
    ///
    /// - `g * gamma < 1`: the recurrent loop is net excitatory, the rate runs to near the refractory
    ///   ceiling, and the state is [`Regime::SynchronousRegular`]. Exact.
    /// - The self-consistent rate falls below [`SILENCE_RATE_HZ`]: [`Regime::NearlySilent`].
    ///   Computed, not guessed.
    /// - **Everything else returns `None`.** Separating [`Regime::AsynchronousIrregular`] from
    ///   [`Regime::SynchronousIrregular`] requires the linear stability of the asynchronous state
    ///   under delayed interaction, which Brunel & Hakim (Neural Comput. 11:1621, 1999) derive as a
    ///   transcendental condition on the transfer function's complex susceptibility. **This
    ///   implementation did not locate a form of that boundary it could reduce to a closed form and
    ///   check**, so it declines rather than transcribing a curve off a figure. Use [`classify`] on
    ///   a measured [`synchrony`] to decide it empirically, which is how the paper decided it.
    ///
    /// # Errors
    ///
    /// As [`Self::self_consistent_rate`].
    pub fn predicted_regime(&self) -> Result<Option<Regime>, MeanFieldError> {
        let nu = self.self_consistent_rate()?;
        if nu < SILENCE_RATE_HZ {
            return Ok(Some(Regime::NearlySilent));
        }
        match self.balance_index() {
            Some(b) if b < 1.0 => Ok(Some(Regime::SynchronousRegular)),
            _ => Ok(None),
        }
    }

    /// The frequency band the fast synchronous-irregular oscillation is reported in, hertz.
    ///
    /// `(1 / (4 delay), 1 / (2 delay))`. In the synchronous-irregular state the population rate
    /// oscillates at a frequency set by the **transmission delay** rather than by any membrane or
    /// synaptic time constant, which is what lets it exceed every individual neuron's firing rate by
    /// an order of magnitude — a 100 Hz population rhythm carried by cells firing at 10 Hz.
    ///
    /// The band is the range reported in Brunel & Hakim (1999) and Brunel (2000), where the period
    /// is given as between two and four times the delay depending on where in the region the network
    /// sits. **This implementation did not locate a closed form pinning the frequency inside that
    /// band**, and returning a single number would be inventing precision the sources do not have.
    ///
    /// `None` for a non-positive or non-finite delay.
    #[must_use]
    pub fn fast_oscillation_band(&self) -> Option<(f64, f64)> {
        if !self.delay.is_finite() || self.delay <= 0.0 {
            return None;
        }
        Some((1.0 / (4.0 * self.delay), 1.0 / (2.0 * self.delay)))
    }
}

// ---------------------------------------------------------------------------------------------
// Refractory density
// ---------------------------------------------------------------------------------------------

/// Population dynamics by **age**: how many neurons last fired how long ago.
///
/// # Why age and not voltage
///
/// The obvious population equation tracks the distribution of membrane potentials and is a partial
/// differential equation with a moving absorbing boundary. The refractory-density formulation tracks
/// the distribution of *time since last spike* instead, and for a **renewal** population — one where
/// a spike erases all memory of what came before — that distribution is closed under a single
/// hazard function `h(a)`, the instantaneous probability per unit time of firing at age `a`. See
/// Gerstner, *Population dynamics of spiking neurons*, Neural Comput. 12:43-89, 2000, and Gerstner &
/// Kistler, *Spiking Neuron Models*, ch. 6.
///
/// The renewal assumption is what a leaky integrate-and-fire neuron with reset satisfies exactly
/// under **constant** input and violates under time-varying input, because then the potential at a
/// given age depends on when in the stimulus the last spike happened.
///
/// # The discrete scheme, and its two exact properties
///
/// Age is binned at `dt`. Each step, the mass at age `k` fires with probability
/// `min(h_k dt, 1)`; the survivors advance one bin; everyone who fired returns to age 0. The last
/// bin is **absorbing** — mass that reaches it stays there and keeps firing at `h_last` — which
/// makes the scheme exact whenever the hazard is constant in the tail, and approximate otherwise.
///
/// 1. **Mass is conserved**, to floating-point summation error, for any hazard. The step adds
///    nothing and removes nothing.
/// 2. **The stationary activity equals `1 / mean_interval()`**, where [`Self::mean_interval`]
///    computes the mean from the hazard by a completely different route — a sum of survival
///    products. The two agreeing to 1e-13 is the module's tightest check on this object.
#[derive(Debug, Clone, PartialEq)]
pub struct RefractoryDensity {
    dt: f64,
    hazard: Vec<f64>,
    density: Vec<f64>,
    activity: f64,
    steps: u64,
}

impl RefractoryDensity {
    /// Build from a time step and a per-age hazard in hertz.
    ///
    /// `hazard[k]` is the firing rate of a neuron whose last spike was `k * dt` ago. The last entry
    /// is the asymptotic hazard and governs the absorbing bin, so choose the array long enough that
    /// the hazard has flattened by its end. The population starts **fully synchronised** at age 0,
    /// which is the initial condition that shows the relaxation transient; call
    /// [`Self::set_density`] for any other.
    ///
    /// # Errors
    ///
    /// [`MeanFieldError::Empty`] for an empty hazard; [`MeanFieldError::NonFinite`] for a non-finite
    /// entry or `dt`; [`MeanFieldError::OutOfRange`] for `dt <= 0` or a negative hazard.
    pub fn new(dt: f64, hazard: Vec<f64>) -> Result<Self, MeanFieldError> {
        let dt = finite("dt", dt)?;
        if dt <= 0.0 {
            return Err(MeanFieldError::OutOfRange {
                what: "dt",
                value: dt,
                low: f64::MIN_POSITIVE,
                high: f64::MAX,
            });
        }
        if hazard.is_empty() {
            return Err(MeanFieldError::Empty { what: "hazard" });
        }
        for (i, &h) in hazard.iter().enumerate() {
            if !h.is_finite() {
                return Err(MeanFieldError::NonFinite { what: "hazard", index: i });
            }
            if h < 0.0 {
                return Err(MeanFieldError::OutOfRange {
                    what: "hazard",
                    value: h,
                    low: 0.0,
                    high: f64::MAX,
                });
            }
        }
        let mut density = vec![0.0; hazard.len()];
        density[0] = 1.0;
        Ok(Self { dt, hazard, density, activity: 0.0, steps: 0 })
    }

    /// The renewal process with a dead time and then a constant hazard: `h(a) = 0` for
    /// `a < t_ref`, `h(a) = hazard` after.
    ///
    /// Its interval distribution is an exponential shifted by `t_ref`, so both the mean interval and
    /// the coefficient of variation are known in closed form — which is what makes it the process
    /// this module's dynamics are verified against. It is also the exact statistics of a perfect
    /// integrator driven by Poisson input with a dead time, so it is not merely a test fixture.
    ///
    /// `bins` sets the age window; mass beyond it lands in the absorbing bin, which here carries the
    /// same constant hazard and so costs nothing in accuracy.
    ///
    /// # Errors
    ///
    /// As [`Self::new`], plus [`MeanFieldError::OutOfRange`] for `t_ref < 0` or `bins == 0`.
    pub fn dead_time_renewal(
        dt: f64,
        t_ref: f64,
        hazard: f64,
        bins: usize,
    ) -> Result<Self, MeanFieldError> {
        let t_ref = finite("t_ref", t_ref)?;
        let hazard = finite("hazard", hazard)?;
        if t_ref < 0.0 {
            return Err(MeanFieldError::OutOfRange {
                what: "t_ref",
                value: t_ref,
                low: 0.0,
                high: f64::MAX,
            });
        }
        if bins == 0 {
            return Err(MeanFieldError::OutOfRange {
                what: "bins",
                value: 0.0,
                low: 1.0,
                high: f64::MAX,
            });
        }
        let r = (t_ref / dt).round() as usize;
        let h = (0..bins).map(|k| if k < r { 0.0 } else { hazard }).collect();
        Self::new(dt, h)
    }

    /// Advance one step and return the population activity, hertz.
    ///
    /// The activity is the total firing flux divided by `dt` — the fraction of the population that
    /// fired this step, per second.
    pub fn step(&mut self) -> f64 {
        let k = self.density.len();
        let mut fired = 0.0f64;
        let mut survivors = vec![0.0f64; k];
        for i in 0..k {
            // `min(_, 1)` rather than a clamp with two bounds: the hazard is already checked
            // non-negative at construction, and a clamp whose lower bound could exceed its upper
            // one is a panic waiting for a parameter change.
            let p = (self.hazard[i] * self.dt).min(1.0);
            let f = self.density[i] * p;
            fired += f;
            survivors[i] = self.density[i] - f;
        }
        let mut next = vec![0.0f64; k];
        next[1..k].copy_from_slice(&survivors[..k - 1]);
        // The last bin is absorbing: survivors there stay there rather than falling off the end,
        // which is what makes the scheme exact for a hazard that has flattened.
        next[k - 1] += survivors[k - 1];
        next[0] += fired;
        self.density = next;
        self.activity = fired / self.dt;
        self.steps += 1;
        self.activity
    }

    /// Run `n` steps and return the activity after the last one, hertz.
    pub fn run(&mut self, n: u64) -> f64 {
        for _ in 0..n {
            self.step();
        }
        self.activity
    }

    /// The most recent population activity, hertz. Zero before the first [`Self::step`].
    #[must_use]
    pub fn activity(&self) -> f64 {
        self.activity
    }

    /// The age density: `density()[k]` is the fraction of the population whose last spike was
    /// `k * dt` ago. Sums to 1.
    #[must_use]
    pub fn density(&self) -> &[f64] {
        &self.density
    }

    /// Total density. Exactly 1 in exact arithmetic, and within a few units in the last place of 1
    /// after any number of steps.
    #[must_use]
    pub fn mass(&self) -> f64 {
        self.density.iter().sum()
    }

    /// Fraction of the population sitting in the absorbing last bin.
    ///
    /// The diagnostic for whether the age window is long enough: if this is not small, neurons are
    /// spending appreciable time at ages the hazard array does not resolve, and any age-dependent
    /// conclusion drawn from the density is about a bin rather than about an age.
    #[must_use]
    pub fn tail_mass(&self) -> f64 {
        self.density[self.density.len() - 1]
    }

    /// How many steps have been taken.
    #[must_use]
    pub fn steps(&self) -> u64 {
        self.steps
    }

    /// Replace the density, normalising it to sum to 1.
    ///
    /// # Errors
    ///
    /// [`MeanFieldError::Empty`] on a length mismatch reported as an empty-shape refusal;
    /// [`MeanFieldError::NonFinite`] for a non-finite entry; [`MeanFieldError::OutOfRange`] for a
    /// negative entry or a total of zero, which has no normalisation.
    pub fn set_density(&mut self, d: &[f64]) -> Result<(), MeanFieldError> {
        if d.len() != self.density.len() {
            return Err(MeanFieldError::Empty { what: "density (length must match the hazard)" });
        }
        let mut total = 0.0;
        for (i, &x) in d.iter().enumerate() {
            if !x.is_finite() {
                return Err(MeanFieldError::NonFinite { what: "density", index: i });
            }
            if x < 0.0 {
                return Err(MeanFieldError::OutOfRange {
                    what: "density",
                    value: x,
                    low: 0.0,
                    high: f64::MAX,
                });
            }
            total += x;
        }
        if total <= 0.0 {
            return Err(MeanFieldError::OutOfRange {
                what: "density total",
                value: total,
                low: f64::MIN_POSITIVE,
                high: f64::MAX,
            });
        }
        for (slot, &x) in self.density.iter_mut().zip(d) {
            *slot = x / total;
        }
        Ok(())
    }

    /// The mean interval between spikes implied by the hazard, seconds, in closed form.
    ///
    /// `E[N] = sum over k >= 0 of P(N > k)`, where `P(N > k)` is the product of the survival
    /// probabilities of ages `0 .. k-1`, plus the geometric tail from the absorbing last bin. This
    /// is computed **from the hazard array alone**, with no iteration of the density, which is what
    /// makes the agreement with the stationary [`Self::activity`] a real check rather than a
    /// restatement.
    ///
    /// `None` when the last bin's hazard is zero: mass that reaches it never leaves, so a fraction
    /// of the population never fires and the mean interval is infinite rather than large.
    #[must_use]
    pub fn mean_interval(&self) -> Option<f64> {
        let k = self.hazard.len();
        let p_last = (self.hazard[k - 1] * self.dt).min(1.0);
        if p_last <= 0.0 {
            return None;
        }
        let mut survival = 1.0f64;
        let mut sum = 0.0f64;
        for i in 0..k {
            sum += survival; // the k = i term of sum P(N > k)
            survival *= 1.0 - (self.hazard[i] * self.dt).min(1.0);
        }
        // `survival` is now the product over ages 0 .. k-1. The absorbing bin re-uses its own
        // survival factor forever, so the remaining terms are geometric with ratio (1 - p_last).
        // P(N > k-1) is the product over 0..k-2, which is `survival / (1 - p_last)` when that is
        // defined; computing it forward avoids the division.
        let mut s_km1 = 1.0f64;
        for i in 0..k - 1 {
            s_km1 *= 1.0 - (self.hazard[i] * self.dt).min(1.0);
        }
        let tail = s_km1 * (1.0 - p_last) / p_last;
        Some(self.dt * (sum + tail))
    }

    /// The coefficient of variation of the intervals implied by the hazard, in closed form.
    ///
    /// From `E[N^2] = sum over k >= 0 of (2k + 1) P(N > k)` with the same geometric tail. For the
    /// dead-time-plus-constant-hazard process this reduces to `sqrt(1 - p) / (p R + 1)` with
    /// `p = h dt` and `R` the dead time in bins, which tends to the continuous-time
    /// `1 / (1 + h t_ref)` as `dt` shrinks — the classic statement that a dead time is what turns a
    /// Poisson train (CV 1) into a more regular one.
    ///
    /// `None` under the same condition as [`Self::mean_interval`].
    #[must_use]
    pub fn interval_cv(&self) -> Option<f64> {
        let k = self.hazard.len();
        let p_last = (self.hazard[k - 1] * self.dt).min(1.0);
        if p_last <= 0.0 {
            return None;
        }
        let mut survival = 1.0f64;
        let (mut m1, mut m2) = (0.0f64, 0.0f64);
        for i in 0..k {
            m1 += survival;
            m2 += (2.0 * i as f64 + 1.0) * survival;
            survival *= 1.0 - (self.hazard[i] * self.dt).min(1.0);
        }
        let mut s_km1 = 1.0f64;
        for i in 0..k - 1 {
            s_km1 *= 1.0 - (self.hazard[i] * self.dt).min(1.0);
        }
        let q = 1.0 - p_last;
        // sum over i >= 1 of (2 (k - 1 + i) + 1) q^i = (2k - 1) q/p + 2 q/p^2
        let n = k as f64;
        m1 += s_km1 * q / p_last;
        m2 += s_km1 * ((2.0 * n - 1.0) * q / p_last + 2.0 * q / (p_last * p_last));
        if m1 <= 0.0 {
            return None;
        }
        let var = m2 - m1 * m1;
        if var < 0.0 {
            return Some(0.0);
        }
        Some(var.sqrt() / m1)
    }
}

// ---------------------------------------------------------------------------------------------
// Criticality
// ---------------------------------------------------------------------------------------------

/// A Galton-Watson branching process with Poisson offspring — the standard null model for a
/// neuronal avalanche.
///
/// # What an avalanche is
///
/// Beggs & Plenz (J. Neurosci. 23:11167-11177, 2003) recorded spontaneous activity in cortical slice
/// cultures and cut it at silent frames: a burst bounded by silence on both sides is an *avalanche*,
/// and its *size* is the number of events it contains. They found the size distribution followed a
/// power law with exponent `-3/2` over more than two decades, and the exponent is the signature: it
/// is what a **critical** branching process gives, and critical branching is the boundary between
/// activity that dies out and activity that explodes.
///
/// # Why `-3/2` is not a fitted constant
///
/// For a branching process whose offspring count is Poisson with mean `m`, the **total progeny**
/// starting from one individual has the Borel distribution
///
/// ```text
/// P(S = k) = exp(-m k) (m k)^(k-1) / k!
/// ```
///
/// exactly (Borel, C. R. Acad. Sci. 214:452, 1942; Otter, Ann. Math. Statist. 20:206, 1949). Put
/// `m = 1` and apply Stirling: `k! = sqrt(2 pi k) (k/e)^k (1 + 1/(12k) + ...)`, so
///
/// ```text
/// P(S = k) = k^(-3/2) / sqrt(2 pi) * (1 - 1/(12k) + O(1/k^2))
/// ```
///
/// The exponent is `-3/2` **and the prefactor is `1 / sqrt(2 pi)`**, both derived rather than
/// measured. `the_critical_borel_tail_is_exactly_the_minus_three_halves_power_law` checks the
/// identity including the `1/(12k)` correction, which is a far sharper test than fitting a slope.
///
/// # What criticality does and does not explain
///
/// A power-law size distribution is consistent with critical branching and is **not** evidence for
/// it: several non-critical mechanisms produce one, and Touboul & Destexhe (Phys. Rev. E 95:012413,
/// 2017) show that a fit within the usual bands passes on processes that are not critical. The
/// branching parameter is the more direct measurement, which is why [`branching_parameter`] and
/// [`multistep_regression`] are here beside the exponent fit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BranchingProcess {
    /// Mean number of offspring per individual — the **branching parameter**. Below 1 subcritical
    /// (activity dies), 1 critical, above 1 supercritical (activity can persist forever).
    pub m: f64,
}

impl BranchingProcess {
    /// Build, rejecting a negative or non-finite mean.
    ///
    /// # Errors
    ///
    /// [`MeanFieldError::NonFinite`] or [`MeanFieldError::OutOfRange`] for `m < 0`.
    pub fn new(m: f64) -> Result<Self, MeanFieldError> {
        let m = finite("m", m)?;
        if m < 0.0 {
            return Err(MeanFieldError::OutOfRange {
                what: "m",
                value: m,
                low: 0.0,
                high: f64::MAX,
            });
        }
        Ok(Self { m })
    }

    /// The Borel probability of a total progeny of exactly `k`, computed in logs.
    ///
    /// Valid for `m <= 1`, where the process is certain to die out and the distribution is proper.
    /// For `m > 1` it is **defective** — it sums to the extinction probability rather than to 1,
    /// because a supercritical avalanche may never end — and this returns `None` rather than a
    /// number that would be read as a probability.
    ///
    /// `None` also for `k == 0` (an avalanche contains at least its trigger) and for `m == 0`
    /// except at `k == 1`.
    #[must_use]
    pub fn borel_pmf(&self, k: u64) -> Option<f64> {
        if k == 0 || self.m > 1.0 {
            return None;
        }
        if self.m == 0.0 {
            return Some(if k == 1 { 1.0 } else { 0.0 });
        }
        let kf = k as f64;
        let ln_fact: f64 = (1..=k).map(|i| (i as f64).ln()).sum();
        Some((-self.m * kf + (kf - 1.0) * (self.m * kf).ln() - ln_fact).exp())
    }

    /// Mean total progeny, `1 / (1 - m)`.
    ///
    /// `None` for `m >= 1`, where the mean diverges. The divergence is the point: at criticality the
    /// *average* avalanche has no size, which is why the distribution has to be reported rather than
    /// summarised.
    #[must_use]
    pub fn mean_size(&self) -> Option<f64> {
        if self.m >= 1.0 { None } else { Some(1.0 / (1.0 - self.m)) }
    }

    /// Probability that the process eventually dies out.
    ///
    /// 1 for `m <= 1`. For `m > 1` it is the root in `(0, 1)` of `q = exp(m (q - 1))`, found by
    /// fixed-point iteration, which converges monotonically from `q = 0` because the map is
    /// increasing and contracting on `[0, q*]`.
    #[must_use]
    pub fn extinction_probability(&self) -> f64 {
        if self.m <= 1.0 {
            return 1.0;
        }
        let mut q = 0.0f64;
        for _ in 0..400 {
            let next = (self.m * (q - 1.0)).exp();
            if (next - q).abs() < 1e-16 {
                return next;
            }
            q = next;
        }
        q
    }

    /// Draw one avalanche: generation sizes starting from a single triggering event.
    ///
    /// `cap` bounds the total size; an avalanche that reaches it is returned with
    /// [`Avalanche::complete`] false, meaning its last generation's offspring were **not** observed.
    /// That flag is not cosmetic — [`branching_parameter`] has to exclude an unobserved transition
    /// from its denominator, and treating a truncated avalanche as finished biases the estimate
    /// downward.
    ///
    /// A Poisson draw that refuses — which needs a mean past `9e18`, reachable only from a huge `m`
    /// and a huge `cap` together — is treated as zero offspring and ends the avalanche. That is
    /// recorded here rather than left silent because it makes an avalanche look extinct when it was
    /// merely unrepresentable.
    ///
    /// # Panics
    ///
    /// Never.
    pub fn avalanche(&self, rng: &mut Rng, cap: u64) -> Avalanche {
        let mut generations = vec![1u64];
        let mut current = 1u64;
        let mut total = 1u64;
        loop {
            let lambda = self.m * current as f64;
            let next = crate::coding::poisson_count(rng, lambda).unwrap_or(0);
            if next == 0 {
                return Avalanche { generations, complete: true };
            }
            total += next;
            generations.push(next);
            current = next;
            if total >= cap {
                return Avalanche { generations, complete: false };
            }
        }
    }
}

/// One avalanche, as generation sizes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Avalanche {
    /// Number of events in each successive generation, starting with the trigger. Never empty, and
    /// `generations[0]` is the number of triggering events — 1 for [`BranchingProcess::avalanche`].
    pub generations: Vec<u64>,
    /// Whether the avalanche was observed to end. `false` means it hit a size or duration cap, so
    /// the offspring of the last recorded generation are unknown rather than zero.
    pub complete: bool,
}

impl Avalanche {
    /// Total number of events. The quantity whose distribution carries the `-3/2` exponent.
    #[must_use]
    pub fn size(&self) -> u64 {
        self.generations.iter().sum()
    }

    /// Number of generations — the avalanche's duration, in frames.
    #[must_use]
    pub fn duration(&self) -> usize {
        self.generations.len()
    }
}

/// The branching parameter estimated from observed avalanches: total offspring over total parents.
///
/// For a Galton-Watson process every event except the trigger is somebody's offspring, so the
/// numerator is `size - generations[0]` and, for an avalanche observed to completion, the
/// denominator is the whole size — **including the final generation, whose offspring count is
/// zero**. Omitting that last transition is the classic way this estimator comes out too high: it
/// conditions on survival. A subcritical process at `m = 0.7` estimates to 1.06 with the omission
/// and to 0.698 without it, and the wrong one looks like a critical network.
///
/// For a truncated avalanche the last generation's offspring were not observed, so it is removed
/// from the denominator and its own events still count as offspring in the numerator.
///
/// This is the ratio estimator of Harris (*The Theory of Branching Processes*, 1963, ch. 1); it is
/// consistent and it is exactly unbiased in the ratio-of-sums sense. `None` for an empty slice or a
/// zero denominator.
#[must_use]
pub fn branching_parameter(avalanches: &[Avalanche]) -> Option<f64> {
    if avalanches.is_empty() {
        return None;
    }
    let (mut offspring, mut parents) = (0u64, 0u64);
    for a in avalanches {
        let size = a.size();
        offspring += size - a.generations[0];
        parents += if a.complete { size } else { size - a.generations[a.generations.len() - 1] };
    }
    if parents == 0 {
        return None;
    }
    Some(offspring as f64 / parents as f64)
}

/// Beggs and Plenz's own estimator: the ratio of the mean second-generation size to the mean first.
///
/// Their `sigma` — the "average number of descendants from one ancestor" — read off the first two
/// frames of each avalanche (J. Neurosci. 23:11167, 2003, and Beggs & Plenz, J. Neurosci. 24:5216,
/// 2004). Avalanches with only one generation contribute a zero to the numerator and must be
/// included; dropping them is the same survival-conditioning error [`branching_parameter`] warns
/// about, and it is why this function takes every avalanche rather than the ones that lasted.
///
/// It uses two frames where [`branching_parameter`] uses all of them, so it is noisier by roughly
/// the square root of the mean duration — it is here because it is what the founding paper reported,
/// not because it is the better estimator.
///
/// `None` for an empty slice or a zero first-generation total.
#[must_use]
pub fn beggs_plenz_ratio(avalanches: &[Avalanche]) -> Option<f64> {
    if avalanches.is_empty() {
        return None;
    }
    let (mut first, mut second) = (0u64, 0u64);
    for a in avalanches {
        first += a.generations[0];
        second += a.generations.get(1).copied().unwrap_or(0);
    }
    if first == 0 {
        return None;
    }
    Some(second as f64 / first as f64)
}

/// Maximum-likelihood exponent of a **continuous** power law `p(x) ~ x^-alpha` above `x_min`.
///
/// `alpha = 1 + n / sum(ln(x_i / x_min))` — Clauset, Shalizi & Newman, SIAM Review 51:661-703, 2009,
/// eq. (3.1); the result itself is older (Hill, Ann. Statist. 3:1163, 1975). The estimator's own
/// sampling distribution is known, which is what makes it testable: `alpha - 1` is inverse-gamma, so
/// the standard error is `(alpha - 1) / sqrt(n)`. At `n = 200 000` and `alpha = 2.5` that is 0.0034,
/// and `the_power_law_estimator_recovers_a_known_exponent` asserts recovery inside six of them.
///
/// **A fitted exponent is not evidence of a power law.** This function reports the best exponent
/// *assuming* the tail is one, and it will report a confident number for a log-normal. Clauset et
/// al.'s goodness-of-fit test is the companion this implementation does not provide.
///
/// # Errors
///
/// [`MeanFieldError::OutOfRange`] for a non-positive or non-finite `x_min`;
/// [`MeanFieldError::NonFinite`] for a non-finite sample; [`MeanFieldError::TooFewSamples`] when
/// fewer than two samples are at or above `x_min`, or when every one of them equals `x_min` exactly
/// and the log sum is zero.
pub fn power_law_exponent(samples: &[f64], x_min: f64) -> Result<f64, MeanFieldError> {
    let x_min = finite("x_min", x_min)?;
    if x_min <= 0.0 {
        return Err(MeanFieldError::OutOfRange {
            what: "x_min",
            value: x_min,
            low: f64::MIN_POSITIVE,
            high: f64::MAX,
        });
    }
    let mut n = 0usize;
    let mut s = 0.0f64;
    for (i, &x) in samples.iter().enumerate() {
        if !x.is_finite() {
            return Err(MeanFieldError::NonFinite { what: "samples", index: i });
        }
        if x >= x_min {
            n += 1;
            s += (x / x_min).ln();
        }
    }
    if n < 2 || s <= 0.0 {
        return Err(MeanFieldError::TooFewSamples {
            what: "power-law exponent",
            got: n,
            need: 2,
        });
    }
    Ok(1.0 + n as f64 / s)
}

/// The same estimator for **integer** data, with Clauset et al.'s continuity correction.
///
/// `alpha = 1 + n / sum(ln(x_i / (x_min - 1/2)))`. Avalanche sizes are counts, and which estimator
/// you use matters most exactly where the data is richest — at a small lower cut. Measured on
/// 200 000 critical avalanches from [`BranchingProcess`], against the derived 1.5:
///
/// ```text
/// x_min     1      2      4      8     16
/// corrected 1.447  1.492  1.504  1.510  1.516
/// continuous 1.648  1.573  1.540  1.528  1.524
/// ```
///
/// The continuous estimator biases **high** at a small cut and the correction biases **low**, and
/// the two converge as `x_min` rises. Clauset et al. report the approximation as good for `x_min`
/// above about 6; below it, `x_min - 1/2` makes the logarithm of a size-1 avalanche positive, which
/// is the visible symptom. The table is reproduced by
/// `the_avalanche_size_exponent_is_three_halves_at_criticality_and_not_otherwise`.
///
/// # Errors
///
/// As [`power_law_exponent`], plus [`MeanFieldError::OutOfRange`] for `x_min == 0`, which has no
/// continuity correction.
pub fn power_law_exponent_discrete(sizes: &[u64], x_min: u64) -> Result<f64, MeanFieldError> {
    if x_min == 0 {
        return Err(MeanFieldError::OutOfRange {
            what: "x_min",
            value: 0.0,
            low: 1.0,
            high: f64::MAX,
        });
    }
    let shifted = x_min as f64 - 0.5;
    let mut n = 0usize;
    let mut s = 0.0f64;
    for &x in sizes {
        if x >= x_min {
            n += 1;
            s += (x as f64 / shifted).ln();
        }
    }
    if n < 2 || s <= 0.0 {
        return Err(MeanFieldError::TooFewSamples {
            what: "discrete power-law exponent",
            got: n,
            need: 2,
        });
    }
    Ok(1.0 + n as f64 / s)
}

/// The multistep-regression estimate of the branching parameter from a population activity series.
///
/// Wilting & Priesemann, *Inferring collective dynamical states from widely unobserved systems*,
/// Nat. Commun. 9:2325, 2018. The idea and the reason it matters:
///
/// For a branching process with drive, `E[A(t + k) | A(t)] = m^k A(t) + constant`, so the
/// autocorrelation is **exactly** `r_k = m^k` — an identity, not an approximation. Taking logs makes
/// `ln r_k` linear in `k` with slope `ln m`, and this fits that line by least squares.
///
/// The reason to do it that way rather than read `r_1` off directly is **subsampling**. If you
/// record a fraction `p` of the neurons, the recorded activity has an extra variance term that
/// inflates the `k = 0` denominator but affects no other lag, so `r_1` is biased toward zero while
/// the *slope* over `k >= 1` is untouched. Measured here at `p = 0.05` on a process with `m = 0.95`:
/// the lag-1 correlation reports 0.34 and the regression reports 0.951. An electrode array sees a
/// few hundred of 10^8 neurons, so this is the difference between concluding a cortex is far from
/// critical and concluding it is near it.
///
/// `k_max` is the largest lag to include and `floor` is the smallest `r_k` to trust; lags are taken
/// in order and the fit stops at the first one at or below `floor`, because beyond that `ln r_k` is
/// the logarithm of noise. `None` when fewer than two usable lags survive, which is a refusal rather
/// than a fit to one point — it happens for small `m` under heavy subsampling, where the correlation
/// is gone by the second lag.
///
/// # Panics
///
/// Never. `k_max` is clamped below the series length before any indexing.
#[must_use]
pub fn multistep_regression(activity: &[f64], k_max: usize, floor: f64) -> Option<f64> {
    let t = activity.len();
    if t < 4 || k_max == 0 || !floor.is_finite() {
        return None;
    }
    if activity.iter().any(|x| !x.is_finite()) {
        return None;
    }
    let mean = activity.iter().sum::<f64>() / t as f64;
    let var = activity.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / t as f64;
    if !(var > 0.0) {
        return None;
    }
    let k_max = k_max.min(t - 2);
    let (mut sx, mut sy, mut sxx, mut sxy, mut n) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for k in 1..=k_max {
        let c = (0..t - k).map(|i| (activity[i] - mean) * (activity[i + k] - mean)).sum::<f64>()
            / (t - k) as f64;
        let r = c / var;
        if !(r > floor) {
            break;
        }
        let (x, y) = (k as f64, r.ln());
        sx += x;
        sy += y;
        sxx += x * x;
        sxy += x * y;
        n += 1.0;
    }
    if n < 2.0 {
        return None;
    }
    let denom = n * sxx - sx * sx;
    if denom == 0.0 {
        return None;
    }
    Some(((n * sxy - sx * sy) / denom).exp())
}

#[cfg(test)]
mod tests {
    use super::{
        Avalanche, BalanceMatrix, BalancedInput, BranchingProcess, BrunelNetwork, MeanFieldError,
        Regime, RefractoryDensity, SiegertInput, beggs_plenz_ratio, branching_parameter, classify,
        erfcx, gauss_legendre, multistep_regression, power_law_exponent,
        power_law_exponent_discrete, siegert_integral, simulate_diffusion,
        simulate_free_membrane, synchrony,
    };
    use crate::coding::poisson_count;
    use crate::neuron::Lif;
    use crate::rng::Rng;
    use crate::surrogate::erf;
    use crate::spike::{Spike, Train};
    use core::f64::consts::PI;

    /// Brunel's own single-neuron parameters, in SI, with rest at 0 V so that his millivolt figures
    /// read directly: 20 ms membrane, 20 mV threshold, 10 mV reset, 2 ms refractory.
    fn brunel_lif() -> Lif {
        Lif {
            tau_m: 20e-3,
            v_rest: 0.0,
            v_th: 20e-3,
            v_reset: 10e-3,
            r_m: 10e6,
            t_ref: 2e-3,
            v: 10e-3,
            refractory: 0.0,
        }
    }

    // ---------------------------------------------------------------------------------------
    // Special functions
    // ---------------------------------------------------------------------------------------

    /// Against published values. `erfcx(1)`, `erfcx(2)` and `erfcx(3)` are `exp(x^2) erfc(x)` with
    /// `erfc` from Abramowitz & Stegun table 7.1; they are also the values every `erfcx`
    /// implementation is checked against.
    #[test]
    fn erfcx_matches_published_values() {
        assert!((erfcx(0.0) - 1.0).abs() < 1e-15, "erfcx(0) = {}", erfcx(0.0));
        for &(x, want) in &[
            (1.0, 0.427_583_576_155_807),
            (2.0, 0.255_395_676_310_518_9),
            (3.0, 0.179_001_151_181_39),
            (0.5, 0.615_690_344_192_130_9),
        ] {
            let got = erfcx(x);
            assert!((got - want).abs() / want < 1e-11, "erfcx({x}) = {got}, want {want}");
        }
        // The reflection is exact arithmetic, not an approximation: erfcx(-1) = 2 e - erfcx(1).
        let want = 2.0 * core::f64::consts::E - erfcx(1.0);
        assert!((erfcx(-1.0) - want).abs() < 1e-14, "erfcx(-1) = {}", erfcx(-1.0));
    }

    /// The two evaluation paths meet at `x = 2`. If they ever stop meeting, one of them has drifted
    /// and every Siegert rate in the crate moves with it — silently, because the discontinuity is
    /// far smaller than any tolerance a rate test would use.
    #[test]
    fn erfcx_is_continuous_across_its_crossover() {
        // Both paths at exactly the same argument. Comparing `erfcx(2 - eps)` against
        // `erfcx(2 + eps)` instead would be comparing two different values of a function whose
        // slope at 2 is -0.107, and the first draft of this test did exactly that and failed by
        // 2.1e-10, which is the slope times the gap and not a discontinuity at all.
        let x = 2.0f64;
        let continued_fraction = erfcx(x); // erfcx dispatches to the continued fraction at x >= 2
        let series = (x * x).exp() * (1.0 - erf(x));
        assert!(
            (continued_fraction - series).abs() < 1e-13,
            "at x = 2 the continued fraction gives {continued_fraction} and the series {series}"
        );
        // And the function is smooth across the branch: the change over a 1e-12 step must be the
        // derivative times the step, `erfcx'(x) = 2 x erfcx(x) - 2/sqrt(pi)`, which is -0.1068 here.
        // h = 1e-5 rather than something smaller: the two evaluations straddle the branch, so their
        // difference has to stay well above the 1e-14 the two paths disagree by, and a central
        // difference at 1e-12 is 95% cancellation noise. The first draft used 1e-12 and reported a
        // slope 11% off, which was the test measuring its own rounding.
        let h = 1e-5f64;
        let slope = (erfcx(x + h) - erfcx(x - h)) / (2.0 * h);
        let want = 2.0 * x * erfcx(x) - 2.0 / PI.sqrt();
        assert!((slope - want).abs() < 1e-8, "numerical slope {slope} vs exact {want}");
    }

    /// The far tail against the asymptotic series, which is independent of both evaluation paths.
    /// At `x = 100` the three-term series is itself good to 1.9e-12, so the comparison is a real
    /// constraint on `erfcx` rather than on the series.
    #[test]
    fn erfcx_matches_its_asymptotic_series_in_the_far_tail() {
        for &x in &[100.0f64, 1000.0] {
            let want = (1.0 - 1.0 / (2.0 * x * x) + 3.0 / (4.0 * x.powi(4))) / (x * PI.sqrt());
            let got = erfcx(x);
            assert!((got - want).abs() / want < 1e-11, "erfcx({x}) = {got}, asymptotic {want}");
        }
    }

    /// 24-point Gauss-Legendre is exact for polynomials to degree 47. This is the whole quadrature's
    /// foundation and it has an exact answer for every `k`, so there is no tolerance to choose
    /// loosely: `integral of x^k over [-1,1]` is `2/(k+1)` for even `k` and zero for odd.
    #[test]
    fn gauss_legendre_integrates_polynomials_exactly() {
        let (xs, ws) = gauss_legendre();
        for k in 0..=47i32 {
            let got: f64 = xs.iter().zip(&ws).map(|(x, w)| w * x.powi(k)).sum();
            let want = if k % 2 == 1 { 0.0 } else { 2.0 / (f64::from(k) + 1.0) };
            assert!(
                (got - want).abs() < 1e-14,
                "degree {k}: quadrature {got}, exact {want}"
            );
        }
        // And it is NOT exact at degree 48, which is what makes the previous assertion a
        // measurement of the rule's order rather than a statement that any sum works.
        let k = 48i32;
        let got: f64 = xs.iter().zip(&ws).map(|(x, w)| w * x.powi(k)).sum();
        let want = 2.0 / (f64::from(k) + 1.0);
        assert!((got - want).abs() > 1e-16, "degree 48 was exact: {got} vs {want}");
    }

    /// The Siegert integral against the Taylor series of its own integrand, summed to 24 terms so
    /// the reference is exact to `1e-17` over the interval used. The recurrence
    /// `(n+1) c_(n+1) = 2 c_(n-1)` comes from `erfcx'(w) = 2 w erfcx(w) - 2/sqrt(pi)`, which is the
    /// defining differential equation — so this compares the quadrature against calculus rather
    /// than against another quadrature.
    #[test]
    fn the_siegert_integral_matches_the_taylor_series_of_its_integrand() {
        const N: usize = 24;
        let mut c = [0.0f64; N];
        c[0] = 1.0;
        c[1] = -2.0 / PI.sqrt();
        for n in 1..N - 1 {
            c[n + 1] = 2.0 * c[n - 1] / (n as f64 + 1.0);
        }
        let anti = |w: f64| {
            let mut s = 0.0;
            let mut p = w;
            for n in 0..N {
                s += c[n] * p / (n as f64 + 1.0);
                p *= w;
            }
            s
        };
        for &(p, q) in &[(-0.3f64, 0.4f64), (-0.05, 0.05), (0.1, 0.5), (-0.6, -0.2)] {
            let want = anti(q) - anti(p);
            let got = siegert_integral(p, q).unwrap();
            assert!(
                (got - want).abs() / want.abs() < 1e-13,
                "integral({p}, {q}) = {got}, series {want}"
            );
        }
    }

    /// In the far supra-threshold tail the integrand is `1/(w sqrt(pi))` to three terms, and its
    /// antiderivative is elementary. This pins the geometric-panel half of the quadrature, which the
    /// Taylor test near zero cannot reach.
    #[test]
    fn the_siegert_integral_matches_the_large_argument_antiderivative() {
        let f = |w: f64| (w.ln() + 0.25 / (w * w) - 3.0 / (16.0 * w.powi(4))) / PI.sqrt();
        for &(p, q) in &[(1e3f64, 2e3f64), (1e6, 3e6), (500.0, 5e5)] {
            let want = f(q) - f(p);
            let got = siegert_integral(p, q).unwrap();
            assert!(
                (got - want).abs() / want < 1e-12,
                "integral({p}, {q}) = {got}, asymptotic {want}"
            );
        }
    }

    #[test]
    fn the_siegert_integral_refuses_a_reversed_or_non_finite_interval() {
        assert!(siegert_integral(1.0, 0.0).is_none());
        assert!(siegert_integral(f64::NAN, 1.0).is_none());
        assert!(siegert_integral(0.0, f64::INFINITY).is_none());
        assert_eq!(siegert_integral(2.0, 2.0), Some(0.0));
        // Past the overflow point the answer is infinite, not wrong and not a panic.
        assert_eq!(siegert_integral(-30.0, -29.0), Some(f64::INFINITY));
    }

    // ---------------------------------------------------------------------------------------
    // The Siegert rate
    // ---------------------------------------------------------------------------------------

    /// ⭐ The module's central check: the closed-form rate against a direct simulation of the same
    /// `Lif` under the same noisy input.
    ///
    /// Six operating points from deeply mean-driven to `CV > 1`. Each one asserts three things:
    ///
    /// 1. enough intervals were collected that the Monte-Carlo standard error is under 2%, so the
    ///    comparison that follows is a measurement rather than a coin toss;
    /// 2. the simulated interval is within 8% of the closed form;
    /// 3. the simulated interval is **not shorter** than the closed form by more than the sampling
    ///    error — the discretisation bias has a known sign, and a simulated rate that came out
    ///    higher than theory would mean something other than missed crossings is happening.
    #[test]
    fn the_siegert_rate_matches_a_direct_simulation_of_the_same_lif() {
        let lif = brunel_lif();
        let dt = 1e-5;
        for &(mu_mv, sigma_mv, seconds) in &[
            (30.0f64, 1.0f64, 150.0f64),
            (25.0, 2.0, 150.0),
            (22.0, 5.0, 200.0),
            (20.0, 4.0, 300.0),
            (15.0, 5.0, 600.0),
            (10.0, 8.0, 600.0),
        ] {
            let input =
                SiegertInput::from_lif(&lif, mu_mv * 1e-3, sigma_mv * 1e-3).expect("valid");
            let theory = input.mean_interval().expect("fires");
            let ticks = (seconds / dt) as u64;
            let mut rng = Rng::new(0x5163_4552_7400 + mu_mv as u64);
            let train = simulate_diffusion(&input, dt, ticks, &mut rng).expect("valid");
            let intervals = train.intervals(0, dt);
            assert!(intervals.len() > 500, "{mu_mv} mV: only {} intervals", intervals.len());
            let n = intervals.len() as f64;
            let mean = intervals.iter().sum::<f64>() / n;
            let cv = train.cv(0, dt).expect("many intervals");
            let sem = cv / n.sqrt();
            assert!(sem < 0.02, "{mu_mv} mV: standard error {sem} is too large to conclude from");
            let rel = (mean - theory) / theory;
            assert!(
                rel.abs() < 0.08,
                "mu = {mu_mv} mV, sigma = {sigma_mv} mV: simulated ISI {mean} s vs Siegert \
                 {theory} s ({:.2}%, cv {cv:.3}, n {n})",
                rel * 100.0
            );
            assert!(
                rel > -3.0 * sem,
                "mu = {mu_mv} mV: the simulation fired FASTER than theory by {:.2}%, which missed \
                 threshold crossings cannot explain",
                -rel * 100.0
            );
        }
    }

    /// The half of the ⭐ test that makes it non-vacuous: below threshold the deterministic formula
    /// says the neuron never fires, and the simulation fires at 9 Hz. If `SiegertInput::rate` were
    /// quietly returning the deterministic answer, or the noise were not reaching the membrane, this
    /// would fail immediately.
    #[test]
    fn noise_alone_fires_a_subthreshold_neuron_and_the_deterministic_formula_cannot_say_so() {
        let lif = brunel_lif();
        // v_inf = 15 mV against a 20 mV threshold. Lif::isi refuses, correctly.
        let i = (15e-3 - lif.v_rest) / lif.r_m;
        assert!(lif.isi(i).is_none(), "the drive must be genuinely sub-threshold");
        let input = SiegertInput::from_lif(&lif, 15e-3, 5e-3).unwrap();
        assert!(!input.is_mean_driven());
        let theory = input.rate().expect("noise fires it");
        assert!((8.0..11.0).contains(&theory), "Siegert rate {theory} Hz");
        let dt = 1e-5;
        let mut rng = Rng::new(31);
        let train = simulate_diffusion(&input, dt, (200.0 / dt) as u64, &mut rng).unwrap();
        let measured = train.rate(0, (200.0 / dt) as u64, dt).unwrap();
        assert!(measured > 5.0, "the simulation fired at {measured} Hz, so the noise is not arriving");
        assert!((measured - theory).abs() / theory < 0.08, "{measured} Hz vs {theory} Hz");
    }

    /// The Siegert prediction must be *better* than the deterministic one where the two differ, or
    /// the extra machinery is not earning its place. At 22 mV mean and 5 mV noise the deterministic
    /// rate is 26.4 Hz, the Siegert rate is 35.3 Hz, and the simulation lands next to the second.
    #[test]
    fn the_siegert_rate_beats_the_deterministic_one_where_noise_matters() {
        let lif = brunel_lif();
        let mu = 22e-3;
        let input = SiegertInput::from_lif(&lif, mu, 5e-3).unwrap();
        let siegert = input.rate().unwrap();
        let deterministic = SiegertInput::from_lif(&lif, mu, 0.0).unwrap().rate().unwrap();
        assert!(
            (siegert - deterministic).abs() / deterministic > 0.25,
            "the two predictions are too close for this test to discriminate: {siegert} vs \
             {deterministic}"
        );
        let dt = 1e-5;
        let mut rng = Rng::new(97);
        let train = simulate_diffusion(&input, dt, (300.0 / dt) as u64, &mut rng).unwrap();
        let intervals = train.intervals(0, dt);
        let measured = intervals.len() as f64 / intervals.iter().sum::<f64>();
        assert!(
            (measured - siegert).abs() * 3.0 < (measured - deterministic).abs(),
            "simulation {measured} Hz sits nearer the deterministic {deterministic} Hz than the \
             Siegert {siegert} Hz"
        );
    }

    /// The zero-noise limit reduces to `Lif::isi` **exactly**, with no tolerance, because it is the
    /// same expression rather than a limit of the quadrature.
    #[test]
    fn the_zero_noise_limit_is_exactly_lif_isi() {
        let lif = brunel_lif();
        for &na in &[3.0f64, 5.0, 10.0, 50.0] {
            let i = na * 1e-9;
            let want = lif.isi(i).expect("supra-threshold");
            let mu = lif.v_inf(i);
            let got = SiegertInput::from_lif(&lif, mu, 0.0).unwrap().mean_interval().unwrap();
            assert_eq!(got, want, "{na} nA: {got} != {want}");
            // And through the current constructor, which does the conversion itself.
            let via_current =
                SiegertInput::from_current(&lif, i, 0.0).unwrap().mean_interval().unwrap();
            assert_eq!(via_current, want, "{na} nA via from_current: {via_current} != {want}");
        }
        // Sub-threshold: both refuse, and refuse rather than returning a very large interval.
        let i = 1e-10;
        assert!(lif.isi(i).is_none());
        assert!(SiegertInput::from_current(&lif, i, 0.0).unwrap().mean_interval().is_none());
    }

    /// The exact-equality test above is a special case in the code, so on its own it proves nothing
    /// about the quadrature. This one closes that hole: the general branch must **converge** to the
    /// special one, and it must do so at the right rate. The leading correction to the interval is
    /// `O(sigma^2)`, so the relative error must fall by a factor near 100 for every decade of
    /// `sigma` — measured here at 99 to 101, which a formula with a wrong power or a wrong
    /// coefficient cannot produce.
    #[test]
    fn the_siegert_interval_converges_to_the_deterministic_one_as_sigma_squared() {
        let lif = brunel_lif();
        let mu = 25e-3;
        let exact = SiegertInput::from_lif(&lif, mu, 0.0).unwrap().mean_interval().unwrap();
        let err = |sigma: f64| {
            let t = SiegertInput::from_lif(&lif, mu, sigma).unwrap().mean_interval().unwrap();
            ((t - exact) / exact).abs()
        };
        let mut previous = err(1e-3);
        assert!(previous > 1e-4, "sigma = 1 mV must move the answer measurably: {previous}");
        for &sigma in &[1e-4f64, 1e-5, 1e-6] {
            let e = err(sigma);
            let ratio = previous / e;
            assert!(
                (95.0..105.0).contains(&ratio),
                "sigma = {sigma}: the error fell by {ratio}x, not the ~100x that an O(sigma^2) \
                 correction requires"
            );
            previous = e;
        }
    }

    /// Scale every voltage and the noise by the same factor and nothing changes: the formula depends
    /// only on `(v_th - mu)/sigma` and `(v_reset - mu)/sigma`. A stray absolute voltage anywhere in
    /// the implementation — a hard-coded millivolt, a rest potential that leaked in — breaks this
    /// and breaks nothing else visibly.
    #[test]
    fn the_siegert_interval_is_invariant_under_rescaling_every_voltage() {
        let base = SiegertInput::new(20e-3, 2e-3, 20e-3, 10e-3, 16e-3, 4e-3).unwrap();
        let want = base.mean_interval().unwrap();
        for &k in &[0.5f64, 3.0, 1000.0] {
            let scaled =
                SiegertInput::new(20e-3, 2e-3, 20e-3 * k, 10e-3 * k, 16e-3 * k, 4e-3 * k).unwrap();
            let got = scaled.mean_interval().unwrap();
            assert!((got - want).abs() / want < 1e-12, "scale {k}: {got} vs {want}");
        }
    }

    /// The refractory period is an additive dead time, so it caps the rate at exactly `1 / t_ref`
    /// however hard the neuron is driven — and the cap must be approached from below, never crossed.
    #[test]
    fn the_refractory_period_caps_the_siegert_rate() {
        let lif = brunel_lif();
        for &mu in &[1.0f64, 100.0, 10_000.0] {
            let r = SiegertInput::from_lif(&lif, mu, 1e-3).unwrap().rate().unwrap();
            assert!(r < 1.0 / lif.t_ref, "{mu} V drive gave {r} Hz, past the {} Hz cap", 1.0 / lif.t_ref);
        }
        let huge = SiegertInput::from_lif(&lif, 1e6, 1e-3).unwrap().rate().unwrap();
        assert!(
            (huge - 1.0 / lif.t_ref).abs() / (1.0 / lif.t_ref) < 1e-6,
            "an unbounded drive gave {huge} Hz against a {} Hz cap",
            1.0 / lif.t_ref
        );
    }

    #[test]
    fn a_deeply_subthreshold_neuron_has_a_rate_of_exactly_zero_rather_than_a_nan() {
        let lif = brunel_lif();
        let input = SiegertInput::from_lif(&lif, -10e-3, 0.2e-3).unwrap();
        assert_eq!(input.mean_interval(), Some(f64::INFINITY));
        assert_eq!(input.rate(), Some(0.0));
    }

    /// ⭐ The noise convention itself, measured. The free membrane's stationary variance must be
    /// `sigma^2 / 2` and its lag-1 autocorrelation `exp(-dt / tau_m)`, and both must hold at coarse
    /// time steps as well as fine ones — because exponential Euler is exact at any step and the
    /// noise is pre-compensated to match it.
    ///
    /// This is the check that constrains `ou_step_sd`, and therefore the spiking simulator's noise,
    /// without going through a firing rate. Dropping the `exp(dt/tau)` compensation moves the
    /// variance by `exp(-2 dt/tau)` — 33% at `dt = 0.2 tau` — which is invisible in a rate
    /// comparison at fine `dt` and glaring here.
    #[test]
    fn the_free_membrane_has_the_exact_ornstein_uhlenbeck_variance_and_autocorrelation() {
        let sigma = 4e-3;
        let input = SiegertInput::new(20e-3, 0.0, 20e-3, 0.0, 5e-3, sigma).unwrap();
        for &r in &[0.2f64, 0.5, 1.0] {
            let dt = r * input.tau_m;
            let n = 2_000_000u64;
            let mut rng = Rng::new(0x0_0000_4F55 + (r * 10.0) as u64);
            let v = simulate_free_membrane(&input, dt, n, &mut rng).unwrap();
            let mean = v.iter().sum::<f64>() / v.len() as f64;
            let var = v.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / v.len() as f64;
            let want_var = sigma * sigma / 2.0;
            assert!(
                (var - want_var).abs() / want_var < 0.03,
                "dt = {r} tau: variance {var} V^2 vs sigma^2/2 = {want_var} V^2"
            );
            assert!(
                (mean - input.mu).abs() < 0.02 * sigma,
                "dt = {r} tau: the free membrane settled at {mean} V, not {} V",
                input.mu
            );
            let cov = (0..v.len() - 1).map(|i| (v[i] - mean) * (v[i + 1] - mean)).sum::<f64>()
                / (v.len() - 1) as f64;
            let want_rho = (-r).exp();
            assert!(
                (cov / var - want_rho).abs() < 0.01,
                "dt = {r} tau: lag-1 correlation {} vs exp(-dt/tau) = {want_rho}",
                cov / var
            );
        }
        // It starts AT `mu`, so there is no relaxation transient to discard. Measured as an
        // ensemble: the mean of the first sample over 4000 independent seeds must be `mu`, because
        // `E[V(dt)] = mu + (V(0) - mu) exp(-dt/tau)` and the second term is zero only when
        // `V(0) == mu`. Starting from `v_reset` instead would put that mean at 1.97 mV, which is 84
        // standard errors away.
        let dt = 0.5 * input.tau_m;
        let mut first = 0.0f64;
        let seeds = 4000u64;
        for seed in 0..seeds {
            let mut rng = Rng::new(seed);
            first += simulate_free_membrane(&input, dt, 1, &mut rng).unwrap()[0];
        }
        first /= seeds as f64;
        let step_noise = sigma * ((1.0 - (-1.0f64).exp()) / 2.0).sqrt();
        let sem = step_noise / (seeds as f64).sqrt();
        assert!(
            (first - input.mu).abs() < 4.0 * sem,
            "the first sample averaged {first} V, not the starting value {} V (s.e. {sem})",
            input.mu
        );
        assert!(simulate_free_membrane(&input, 0.0, 10, &mut Rng::new(1)).is_err());
        assert!(simulate_free_membrane(&input, f64::NAN, 10, &mut Rng::new(1)).is_err());
    }

    #[test]
    fn the_free_membrane_standard_deviation_is_sigma_over_root_two() {
        let input = SiegertInput::new(20e-3, 0.0, 20e-3, 0.0, 0.0, 4e-3).unwrap();
        assert!((input.free_membrane_sd() * core::f64::consts::SQRT_2 - 4e-3).abs() < 1e-18);
        assert!((input.threshold_distance().unwrap() - 5.0).abs() < 1e-12);
        let noiseless = SiegertInput::new(20e-3, 0.0, 20e-3, 0.0, 0.0, 0.0).unwrap();
        assert!(noiseless.threshold_distance().is_none());
    }

    #[test]
    fn the_constructor_refuses_every_parameter_the_formula_is_undefined_on() {
        assert!(matches!(
            SiegertInput::new(f64::NAN, 0.0, 1.0, 0.0, 0.0, 1.0),
            Err(MeanFieldError::NonFinite { what: "tau_m", .. })
        ));
        assert!(matches!(
            SiegertInput::new(0.0, 0.0, 1.0, 0.0, 0.0, 1.0),
            Err(MeanFieldError::OutOfRange { what: "tau_m", .. })
        ));
        assert!(matches!(
            SiegertInput::new(1.0, -1.0, 1.0, 0.0, 0.0, 1.0),
            Err(MeanFieldError::OutOfRange { what: "t_ref", .. })
        ));
        assert!(matches!(
            SiegertInput::new(1.0, 0.0, 1.0, 0.0, 0.0, -1.0),
            Err(MeanFieldError::OutOfRange { what: "sigma", .. })
        ));
        assert!(matches!(
            SiegertInput::new(1.0, 0.0, 0.0, 1.0, 0.0, 1.0),
            Err(MeanFieldError::Ordering { larger: "v_th", .. })
        ));
        let lif = brunel_lif();
        assert!(SiegertInput::from_current(&lif, 1e-9, -1.0).is_err());
        assert!(simulate_diffusion(
            &SiegertInput::from_lif(&lif, 0.0, 0.0).unwrap(),
            0.0,
            10,
            &mut Rng::new(1)
        )
        .is_err());
    }

    // ---------------------------------------------------------------------------------------
    // Balanced networks
    // ---------------------------------------------------------------------------------------

    /// At `g * gamma == 1` the recurrent contribution to the mean cancels **identically**, for every
    /// recurrent rate. This is the balance condition, and it is an algebraic identity rather than a
    /// numerical coincidence, so it is asserted at rates spanning four decades.
    #[test]
    fn the_balance_line_cancels_the_recurrent_mean_exactly() {
        let base = BalancedInput {
            tau_m: 20e-3,
            c_exc: 1000.0,
            c_inh: 250.0,
            g: 4.0,
            j: 0.1e-3,
            nu: 0.0,
            nu_ext: 9.0,
        };
        assert!((base.balance_index().unwrap() - 1.0).abs() < 1e-15);
        let external_only = base.mu();
        for &nu in &[0.0f64, 1.0, 37.0, 500.0, 10_000.0] {
            let b = BalancedInput { nu, ..base };
            assert!(
                (b.mu() - external_only).abs() < 1e-15,
                "at nu = {nu} Hz the mean moved to {} V from {external_only} V",
                b.mu()
            );
            // The variance does NOT cancel, and that asymmetry is the whole mechanism.
            assert!(
                nu == 0.0 || b.sigma() > base.sigma() * 1.05,
                "at nu = {nu} Hz the fluctuation failed to grow: {} vs {}",
                b.sigma(),
                base.sigma()
            );
        }
        // Off the line the mean does move, so the test above is not asserting a constant function.
        let off = BalancedInput { g: 5.0, nu: 100.0, ..base };
        assert!((off.mu() - external_only).abs() > 1e-4, "off-balance mean did not move");
    }

    /// The van Vreeswijk-Sompolinsky scaling: with `J = J_0 / sqrt(K)` the fluctuation is **exactly**
    /// independent of `K` while the unbalanced mean grows as `sqrt(K)`. Four decades of `K`, and the
    /// fluctuation is required to be equal to 1e-12 relative rather than merely similar.
    #[test]
    fn the_fluctuation_is_exactly_independent_of_k_while_the_mean_grows_as_root_k() {
        let (j0, gamma, g, tau) = (1e-3f64, 0.25f64, 5.0f64, 20e-3f64);
        let (nu_e, nu_0) = (10.0f64, 8.0f64);
        let mut reference_sigma = None;
        let mut reference_mu = None;
        for &k in &[1u32, 100, 10_000, 1_000_000] {
            let kf = f64::from(k);
            // The microscopic form: K synapses each carrying J_0 / sqrt(K).
            let j = j0 / kf.sqrt();
            let b = BalancedInput {
                tau_m: tau,
                c_exc: kf,
                c_inh: gamma * kf,
                g,
                j,
                nu: nu_e,
                nu_ext: nu_0,
            };
            let sigma = b.sigma();
            let mu = b.mu();
            match reference_sigma {
                None => {
                    reference_sigma = Some(sigma);
                    reference_mu = Some(mu);
                }
                Some(s) => {
                    assert!(
                        (sigma - s).abs() / s < 1e-12,
                        "K = {k}: sigma moved from {s} to {sigma}"
                    );
                    let want = reference_mu.unwrap() * kf.sqrt();
                    assert!(
                        (mu - want).abs() / want.abs() < 1e-12,
                        "K = {k}: mean {mu} is not sqrt(K) times {}",
                        reference_mu.unwrap()
                    );
                }
            }
        }
    }

    /// The two-population balance equations, solved and then checked by substitution. The residual
    /// is required to vanish to machine precision relative to the drive, which is a statement about
    /// the solve rather than about the model.
    #[test]
    fn the_balanced_rates_solve_their_own_equations_to_machine_precision() {
        let m = BalanceMatrix {
            j_ee: 1.0e-3,
            j_ei: 2.0e-3,
            j_ie: 1.0e-3,
            j_ii: 1.8e-3,
            j_e0: 1.0e-3,
            j_i0: 0.8e-3,
        };
        let nu_0 = 10.0;
        let (nu_e, nu_i) = m.balanced_rates(nu_0).expect("a balanced state exists here");
        assert!(nu_e > 0.0 && nu_i > 0.0);
        let (r_e, r_i) = m.residual(nu_e, nu_i, nu_0);
        let scale = m.j_e0 * nu_0;
        assert!(r_e.abs() / scale < 1e-13, "excitatory residual {r_e}");
        assert!(r_i.abs() / scale < 1e-13, "inhibitory residual {r_i}");
        // The rates are linear in the drive — a property of the balance equations, not of the
        // neuron model, which is the striking claim of the 1996 paper.
        let (e2, i2) = m.balanced_rates(2.0 * nu_0).unwrap();
        assert!((e2 - 2.0 * nu_e).abs() / nu_e < 1e-12, "{e2} vs {}", 2.0 * nu_e);
        assert!((i2 - 2.0 * nu_i).abs() / nu_i < 1e-12, "{i2} vs {}", 2.0 * nu_i);
        // And the residual is NOT zero away from the solution, so the check above is not vacuous.
        let (bad, _) = m.residual(nu_e * 1.01, nu_i, nu_0);
        assert!(bad.abs() / scale > 1e-4, "perturbing the rate left the residual at {bad}");
    }

    /// Existence is a condition, not a guarantee: for some connectivity the only cancellation has a
    /// negative rate in it, and the honest answer is that the network has no balanced state.
    #[test]
    fn a_connectivity_with_no_balanced_state_is_refused_rather_than_reported() {
        // The external drive strongly favours the excitatory population over the inhibitory one, so
        // the only way to cancel the excitatory drive is with more inhibitory firing than the
        // inhibitory population's own balance permits. The solution is negative in both rates.
        let m = BalanceMatrix {
            j_ee: 1.0e-3,
            j_ei: 1.0e-3,
            j_ie: 1.0e-3,
            j_ii: 2.0e-3,
            j_e0: 1.0e-3,
            j_i0: 0.1e-3,
        };
        match m.balanced_rates(10.0) {
            Err(MeanFieldError::NoBalancedState { nu_exc, nu_inh }) => {
                assert!(nu_exc < 0.0 && nu_inh < 0.0, "{nu_exc}, {nu_inh}");
            }
            other => panic!("expected a refusal, got {other:?}"),
        }
        // Raising the external drive onto the inhibitory population restores the balanced state,
        // so the refusal is a property of the connectivity and not of the solver.
        let fixed = BalanceMatrix { j_i0: 3.0e-3, ..m };
        let (e, i) = fixed.balanced_rates(10.0).expect("a balanced state exists here");
        assert!(e > 0.0 && i > 0.0, "{e}, {i}");
        // A singular matrix is a different refusal with a different name.
        let singular = BalanceMatrix {
            j_ee: 1.0e-3,
            j_ei: 2.0e-3,
            j_ie: 2.0e-3,
            j_ii: 4.0e-3,
            j_e0: 1.0e-3,
            j_i0: 1.0e-3,
        };
        assert!(matches!(
            singular.balanced_rates(10.0),
            Err(MeanFieldError::Degenerate { .. })
        ));
    }

    /// ⭐ The consequence of balance that can be measured with `Train::cv`: a balanced network's
    /// neuron fires irregularly and an unbalanced one fires like a clock, under the **same**
    /// microscopic Poisson input machinery and the same excitatory drive.
    #[test]
    fn a_balanced_drive_gives_irregular_firing_and_an_unbalanced_one_does_not() {
        let lif = brunel_lif();
        let dt = 1e-4;
        let ticks = (400.0 / dt) as u64;
        let balanced = BalancedInput {
            tau_m: lif.tau_m,
            c_exc: 1000.0,
            c_inh: 250.0,
            g: 4.0,
            j: 0.1e-3,
            nu: 10.0,
            nu_ext: 9.0,
        };
        // Same excitatory input, inhibition switched off. That is what "unbalanced" means here.
        let unbalanced = BalancedInput { g: 0.0, ..balanced };

        let mu_b = lif.v_rest + balanced.mu();
        let mu_u = lif.v_rest + unbalanced.mu();
        assert!(mu_b < lif.v_th, "the balanced mean must sit below threshold: {mu_b} V");
        assert!(mu_u > lif.v_th, "the unbalanced mean must sit above it: {mu_u} V");

        let mut rng = Rng::new(0xBA1A_4CED);
        let tb = balanced.simulate(&lif, dt, ticks, &mut rng).unwrap();
        let tu = unbalanced.simulate(&lif, dt, ticks, &mut rng).unwrap();
        let cv_b = tb.cv(0, dt).expect("many intervals");
        let cv_u = tu.cv(0, dt).expect("many intervals");
        assert!(tb.len() > 200 && tu.len() > 200, "{} and {} spikes", tb.len(), tu.len());
        assert!(cv_b > 0.6, "the balanced drive gave CV {cv_b}, which is not irregular firing");
        assert!(cv_u < 0.25, "the unbalanced drive gave CV {cv_u}, which is not regular firing");
        assert!(cv_b > 3.0 * cv_u, "CV {cv_b} vs {cv_u} does not separate the regimes");

        // And the diffusion approximation predicts the balanced rate from the same parameters,
        // which is the link between the microscopic drive and the closed form.
        let predicted = balanced.siegert(&lif).unwrap().rate().unwrap();
        let measured = tb.rate(0, ticks, dt).unwrap();
        assert!(
            (measured - predicted).abs() / predicted < 0.15,
            "shot-noise simulation {measured} Hz vs diffusion approximation {predicted} Hz"
        );
    }

    /// The `BalanceMatrix` moments, which the `BalancedInput` scaling test does not reach. Three
    /// separate claims: the mean drive vanishes exactly at the balanced rates for every `K`, it
    /// grows as `sqrt(K)` away from them, and the fluctuation contains no `K` and adds its three
    /// inputs in quadrature rather than linearly.
    #[test]
    fn the_balance_matrix_moments_grow_as_root_k_and_add_in_quadrature() {
        let m = BalanceMatrix {
            j_ee: 1.0e-3,
            j_ei: 2.0e-3,
            j_ie: 1.0e-3,
            j_ii: 1.8e-3,
            j_e0: 1.0e-3,
            j_i0: 0.8e-3,
        };
        let (nu_0, tau) = (10.0f64, 20e-3f64);
        let (be, bi) = m.balanced_rates(nu_0).unwrap();
        for &k in &[1u32, 100, 1_000_000] {
            let d = m.mean_drive_exc(be, bi, nu_0, k, tau).unwrap();
            // Against the scale of the term being cancelled, which itself carries the sqrt(K):
            // the residual is a rounding of a 2 mV drive, not a small absolute number that a
            // fixed tolerance would happen to admit at one K and not another.
            let scale = tau * f64::from(k).sqrt() * m.j_e0 * nu_0;
            assert!(
                d.abs() / scale < 1e-13,
                "K = {k}: the balanced mean drive is {d} V against a {scale} V drive"
            );
        }
        // Away from balance the same quantity grows as sqrt(K), which is the divergence the
        // balance condition exists to cancel.
        let (nu_e, nu_i) = (be * 1.2, bi);
        let d1 = m.mean_drive_exc(nu_e, nu_i, nu_0, 1, tau).unwrap();
        assert!(d1.abs() > 1e-6, "the off-balance reference drive is only {d1} V");
        for &k in &[4u32, 100, 10_000] {
            let d = m.mean_drive_exc(nu_e, nu_i, nu_0, k, tau).unwrap();
            let want = d1 * f64::from(k).sqrt();
            assert!((d - want).abs() / want.abs() < 1e-12, "K = {k}: {d} V vs {want} V");
        }
        // Quadrature: doubling every rate multiplies sigma by exactly sqrt(2).
        let s = m.fluctuation_exc(nu_e, nu_i, nu_0, tau).unwrap();
        let doubled = m.fluctuation_exc(2.0 * nu_e, 2.0 * nu_i, 2.0 * nu_0, tau).unwrap();
        assert!(
            (doubled - s * core::f64::consts::SQRT_2).abs() / s < 1e-12,
            "doubling the rates gave {doubled} against {} for a quadrature sum",
            s * core::f64::consts::SQRT_2
        );
        // Every one of the three inputs contributes, and the inhibitory one contributes most here
        // because its efficacy is twice the others'. Dropping any term is visible.
        for (label, without) in [
            ("excitatory", m.fluctuation_exc(0.0, nu_i, nu_0, tau).unwrap()),
            ("inhibitory", m.fluctuation_exc(nu_e, 0.0, nu_0, tau).unwrap()),
            ("external", m.fluctuation_exc(nu_e, nu_i, 0.0, tau).unwrap()),
        ] {
            assert!(
                without < 0.95 * s,
                "removing the {label} input left sigma at {without} against {s}"
            );
        }
        assert!(m.mean_drive_exc(nu_e, nu_i, nu_0, 0, tau).is_none());
        assert!(m.mean_drive_exc(nu_e, nu_i, nu_0, 4, f64::NAN).is_none());
        assert!(m.fluctuation_exc(-1.0, nu_i, nu_0, tau).is_none());
        assert!(m.fluctuation_exc(f64::NAN, nu_i, nu_0, tau).is_none());
        assert!(m.balanced_rates(f64::NAN).is_err());
    }

    /// A network with no excitatory fan-in has no connectivity ratio, no threshold drive and no
    /// transfer function — and every one of those is `None` rather than a zero that would read as
    /// "perfectly inhibition-dominated".
    #[test]
    fn a_network_with_no_excitatory_fan_in_refuses_every_ratio_it_cannot_form() {
        let b = BalancedInput {
            tau_m: 20e-3,
            c_exc: 0.0,
            c_inh: 250.0,
            g: 4.0,
            j: 1e-4,
            nu: 10.0,
            nu_ext: 10.0,
        };
        assert!(b.gamma().is_none());
        assert!(b.balance_index().is_none());
        let mut net = brunel_net(4.0, 1.0);
        net.c_exc = 0.0;
        assert!(net.gamma().is_none());
        assert!(net.balance_index().is_none());
        assert!(net.nu_thr().is_none());
        assert!(net.input(1.0).is_none());
        assert!(net.transfer(1.0).is_none());
        assert!(matches!(
            net.self_consistent_rate(),
            Err(MeanFieldError::OutOfRange { what: "c_exc * j * tau_m", .. })
        ));
        let ok = brunel_net(4.0, 1.0);
        assert!(ok.input(-1.0).is_none());
        assert!(ok.input(f64::NAN).is_none());
        // A threshold at or below rest is also not a drive scale.
        let mut sunk = brunel_net(4.0, 1.0);
        sunk.neuron.v_rest = sunk.neuron.v_th;
        assert!(sunk.nu_thr().is_none());
        // And the microscopic simulator refuses the same nonsense the moments do.
        let lif = brunel_lif();
        let mut rng = Rng::new(1);
        assert!(BalancedInput { nu: -1.0, ..b }.simulate(&lif, 1e-4, 10, &mut rng).is_err());
        assert!(b.simulate(&lif, 0.0, 10, &mut rng).is_err());
        assert!(BalancedInput { j: f64::NAN, ..b }.simulate(&lif, 1e-4, 10, &mut rng).is_err());
    }

    // ---------------------------------------------------------------------------------------
    // Brunel's phase diagram
    // ---------------------------------------------------------------------------------------

    fn brunel_net(g: f64, nu_ext_ratio: f64) -> BrunelNetwork {
        BrunelNetwork {
            neuron: brunel_lif(),
            c_exc: 1000.0,
            c_inh: 250.0,
            j: 0.1e-3,
            g,
            nu_ext_ratio,
            delay: 1.5e-3,
        }
    }

    /// `nu_thr` with Brunel's parameters is 10 Hz, and it is a closed form: 20 mV of threshold over
    /// 1000 synapses of 0.1 mV each integrated over 20 ms.
    #[test]
    fn the_threshold_drive_rate_is_the_published_ten_hertz() {
        let net = brunel_net(4.0, 1.0);
        let nu_thr = net.nu_thr().unwrap();
        assert!((nu_thr - 10.0).abs() < 1e-12, "nu_thr = {nu_thr} Hz");
        // At exactly nu_ext = nu_thr the external drive alone puts the mean exactly at threshold,
        // which is what the definition says and is worth checking rather than assuming.
        let b = net.input(0.0).unwrap();
        assert!((net.neuron.v_rest + b.mu() - net.neuron.v_th).abs() < 1e-15);
    }

    /// The self-consistent rate is a fixed point, and a fixed point can be checked by substitution.
    /// Across the phase diagram the residual `Phi(nu) - nu` must vanish, and it does to 1e-10 Hz.
    #[test]
    fn the_self_consistent_rate_is_a_fixed_point_of_the_transfer_function() {
        for &g in &[3.0f64, 4.0, 5.0, 6.0] {
            for &x in &[0.9f64, 1.0, 2.0, 4.0] {
                let net = brunel_net(g, x);
                let nu = net.self_consistent_rate().unwrap();
                let phi = net.transfer(nu).unwrap();
                assert!(
                    (phi - nu).abs() < 1e-9 * nu.max(1.0),
                    "g = {g}, x = {x}: Phi({nu}) = {phi}"
                );
            }
        }
    }

    /// On the balance line the mean drive is exactly the external drive, whatever the recurrent rate
    /// turns out to be: `mu = x * (v_th - v_rest)`. That is a closed form with no free parameter and
    /// it is the sharpest statement the phase diagram contains.
    #[test]
    fn on_the_balance_line_the_mean_drive_is_exactly_the_external_drive() {
        for &x in &[0.9f64, 1.0, 2.0, 4.0] {
            let net = brunel_net(4.0, x);
            let nu = net.self_consistent_rate().unwrap();
            let mu = net.input(nu).unwrap().mu();
            let want = x * (net.neuron.v_th - net.neuron.v_rest);
            assert!((mu - want).abs() / want < 1e-12, "x = {x}: mu = {mu} V, want {want} V");
        }
    }

    /// Inhibition-dominated recurrence is negative feedback on the rate: raise `g` and the network
    /// slows down, monotonically. Excitation-dominated recurrence does the opposite and runs to near
    /// the refractory ceiling.
    #[test]
    fn increasing_inhibition_lowers_the_rate_and_excitation_dominance_saturates_it() {
        let ceiling = 1.0 / brunel_lif().t_ref;
        let mut previous = f64::INFINITY;
        for &g in &[4.5f64, 5.0, 5.5, 6.0, 7.0, 8.0] {
            let nu = brunel_net(g, 2.0).self_consistent_rate().unwrap();
            assert!(nu < previous, "g = {g} raised the rate to {nu} Hz from {previous} Hz");
            previous = nu;
        }
        assert!(previous > 1.0, "the sweep ended at {previous} Hz, so it was pinned at silence");
        for &g in &[0.0f64, 2.0, 3.0] {
            let nu = brunel_net(g, 2.0).self_consistent_rate().unwrap();
            assert!(
                nu > 0.5 * ceiling,
                "g = {g} is excitation dominated and gave only {nu} Hz against a {ceiling} Hz cap"
            );
        }
    }

    /// The regime prediction decides exactly the two boundaries it has closed forms for and refuses
    /// the third. The refusal is the point: an implementation that returned
    /// `AsynchronousIrregular` for everything inhibition-dominated would pass a looser test and
    /// would be claiming a boundary it does not have.
    #[test]
    fn the_predicted_regime_decides_only_the_boundaries_that_have_closed_forms() {
        assert_eq!(
            brunel_net(2.0, 2.0).predicted_regime().unwrap(),
            Some(Regime::SynchronousRegular),
            "g * gamma < 1 is excitation dominated"
        );
        assert_eq!(
            brunel_net(6.0, 0.8).predicted_regime().unwrap(),
            Some(Regime::NearlySilent),
            "below the threshold drive with dominant inhibition"
        );
        assert_eq!(
            brunel_net(5.0, 2.0).predicted_regime().unwrap(),
            None,
            "the asynchronous-to-synchronous boundary has no closed form here"
        );
        // A neuron without a refractory period has no bracket, and that is an error rather than a
        // guess at a ceiling.
        let mut net = brunel_net(5.0, 2.0);
        net.neuron.t_ref = 0.0;
        assert!(matches!(
            net.self_consistent_rate(),
            Err(MeanFieldError::NoRefractoryPeriod)
        ));
    }

    /// The claim being tested is not that the band has a particular value but that it depends on
    /// **nothing except the delay**: change the synaptic strength, the inhibition ratio and the
    /// fan-in enough to move the firing rate by a factor of three, and the band must not move at
    /// all. That is what "the frequency is set by the transmission delay" means, and it is a
    /// statement an implementation that quietly folded in `tau_m` would fail.
    #[test]
    fn the_fast_oscillation_band_depends_on_the_delay_and_on_nothing_else() {
        let net = brunel_net(5.0, 2.0);
        let (lo, hi) = net.fast_oscillation_band().unwrap();
        assert!((lo - 1.0 / (4.0 * 1.5e-3)).abs() < 1e-9, "{lo}");
        assert!((hi - 1.0 / (2.0 * 1.5e-3)).abs() < 1e-9, "{hi}");
        let base_rate = net.self_consistent_rate().unwrap();

        let mut moved = net;
        moved.g = 8.0;
        moved.j = 0.2e-3;
        moved.neuron.tau_m = 10e-3;
        let moved_rate = moved.self_consistent_rate().unwrap();
        assert!(
            moved_rate < base_rate / 2.0,
            "the perturbation left the rate at {moved_rate} Hz against {base_rate} Hz, so the              invariance below is not being tested against anything"
        );
        assert_eq!(moved.fast_oscillation_band().unwrap(), (lo, hi));

        // Halving the delay doubles both edges, exactly.
        let mut faster = net;
        faster.delay = 0.75e-3;
        let (lo2, hi2) = faster.fast_oscillation_band().unwrap();
        assert!((lo2 - 2.0 * lo).abs() < 1e-9 && (hi2 - 2.0 * hi).abs() < 1e-9, "{lo2}, {hi2}");

        // In the inhibition-dominated regime the population rhythm outruns the neurons carrying it,
        // which is the property that makes a 100 Hz cortical rhythm possible at 10 Hz firing.
        assert!(lo > 4.0 * base_rate, "{lo} Hz against {base_rate} Hz firing");
        assert!(lo > 4.0 * moved_rate, "{lo} Hz against {moved_rate} Hz firing");

        let mut slow = net;
        slow.delay = 0.0;
        assert!(slow.fast_oscillation_band().is_none());
        slow.delay = f64::NAN;
        assert!(slow.fast_oscillation_band().is_none());
    }

    /// The synchrony index has two exact anchors and this asserts both. Perfectly locked neurons
    /// give exactly 1; `N` independent Poisson neurons give `1 / sqrt(N)` in expectation, which at
    /// `N = 400` is 0.05.
    #[test]
    fn the_synchrony_index_is_one_for_a_locked_population_and_one_over_root_n_for_an_independent_one()
     {
        let bins = 200u64;
        let bin_ticks = 10u64;
        let ticks = bins * bin_ticks;
        let n = 400u32;

        // Locked: every neuron fires in exactly the same ticks, at an irregular set of them so the
        // population count actually varies.
        let mut rng = Rng::new(5);
        let mut spikes = Vec::new();
        for b in 0..bins {
            let k = rng.below(4); // 0..3 spikes in this bin, same for everyone
            for j in 0..k {
                for src in 0..n {
                    spikes.push(Spike { t: b * bin_ticks + u64::from(j), source: src });
                }
            }
        }
        let locked = Train::from_spikes(spikes);
        let chi = synchrony(&locked, n, ticks, bin_ticks).unwrap();
        assert!((chi - 1.0).abs() < 1e-12, "a locked population gave chi = {chi}");

        // Independent: each neuron draws its own Poisson counts.
        let mut spikes = Vec::new();
        for src in 0..n {
            for b in 0..bins {
                let k = poisson_count(&mut rng, 1.5).unwrap();
                for j in 0..k.min(bin_ticks) {
                    spikes.push(Spike { t: b * bin_ticks + j, source: src });
                }
            }
        }
        let independent = Train::from_spikes(spikes);
        let chi = synchrony(&independent, n, ticks, bin_ticks).unwrap();
        let want = 1.0 / f64::from(n).sqrt();
        assert!(
            (chi - want).abs() < 0.4 * want,
            "an independent population gave chi = {chi}, want about {want}"
        );
        assert!(synchrony(&independent, 0, ticks, bin_ticks).is_none());
        assert!(synchrony(&independent, n, ticks, 0).is_none());
    }

    #[test]
    fn the_regime_classifier_separates_the_four_boxes_and_refuses_nonsense() {
        assert_eq!(classify(0.2, 1.0, 0.05), Some(Regime::NearlySilent));
        assert_eq!(classify(10.0, 0.9, 0.05), Some(Regime::AsynchronousIrregular));
        assert_eq!(classify(10.0, 0.9, 0.7), Some(Regime::SynchronousIrregular));
        assert_eq!(classify(200.0, 0.1, 0.7), Some(Regime::SynchronousRegular));
        assert_eq!(classify(200.0, 0.1, 0.05), Some(Regime::AsynchronousRegular));
        assert!(classify(f64::NAN, 0.5, 0.5).is_none());
        assert!(classify(-1.0, 0.5, 0.5).is_none());
        // Silence wins over the other two axes, because below a hertz there are too few intervals
        // for either of them to have been measured.
        assert_eq!(classify(0.99, 0.0, 1.0), Some(Regime::NearlySilent));
        assert_eq!(classify(1.01, 0.0, 1.0), Some(Regime::SynchronousRegular));
    }

    // ---------------------------------------------------------------------------------------
    // Refractory density
    // ---------------------------------------------------------------------------------------

    /// ⭐ The stationary population activity against the mean interval computed from the hazard by a
    /// different route. `1 / (t_ref + 1 / h)` for a dead time plus a constant hazard, exactly.
    #[test]
    fn the_stationary_activity_is_exactly_one_over_the_mean_interval() {
        let dt = 1e-4;
        for &(t_ref, h) in &[(2e-3f64, 40.0f64), (0.0, 25.0), (5e-3, 100.0), (1e-3, 5.0)] {
            let mut rd = RefractoryDensity::dead_time_renewal(dt, t_ref, h, 4000).unwrap();
            let a = rd.run(300_000);
            let closed = rd.mean_interval().unwrap();
            assert!(
                (a - 1.0 / closed).abs() / (1.0 / closed) < 1e-10,
                "t_ref = {t_ref}, h = {h}: activity {a} Hz vs 1/{closed} s"
            );
            // And the mean interval itself against the analytic dead-time-plus-exponential value,
            // which involves both parameters and so cannot be satisfied by getting one of them right.
            let p = h * dt;
            let want = dt * ((t_ref / dt).round() + 1.0 / p);
            assert!(
                (closed - want).abs() / want < 1e-12,
                "t_ref = {t_ref}, h = {h}: mean interval {closed} s vs {want} s"
            );
        }
    }

    /// The coefficient of variation of the same process, in closed form:
    /// `sqrt(1 - p) / (p R + 1)`. Two independent routes to it — the moment sum over the hazard, and
    /// the geometric-with-dead-time formula — and they must agree.
    #[test]
    fn the_interval_cv_matches_the_dead_time_geometric_formula() {
        let dt = 1e-4;
        for &(t_ref, h) in &[(0.0f64, 40.0f64), (2e-3, 40.0), (10e-3, 40.0), (2e-3, 200.0)] {
            let rd = RefractoryDensity::dead_time_renewal(dt, t_ref, h, 6000).unwrap();
            let got = rd.interval_cv().unwrap();
            let p = h * dt;
            let r = (t_ref / dt).round();
            let want = (1.0 - p).sqrt() / (p * r + 1.0);
            assert!((got - want).abs() / want < 1e-9, "t_ref = {t_ref}, h = {h}: {got} vs {want}");
        }
        // No dead time is a geometric interval, whose CV tends to 1 as dt shrinks. That limit is
        // what makes the formula recognisable as "Poisson".
        let rd = RefractoryDensity::dead_time_renewal(1e-6, 0.0, 40.0, 200_000).unwrap();
        let cv = rd.interval_cv().unwrap();
        assert!((cv - 1.0).abs() < 1e-4, "a memoryless process gave CV {cv}");
    }

    /// Mass is conserved by construction, and this asserts it over a long run rather than assuming
    /// it from the arithmetic. A scheme that lost a part in 1e6 per step would look perfectly stable
    /// and would report an activity 10% low after 100 000 steps.
    #[test]
    fn the_refractory_density_conserves_mass() {
        let dt = 1e-4;
        let hazard: Vec<f64> = (0..500).map(|k| if k < 20 { 0.0 } else { 3.0 * k as f64 }).collect();
        let mut rd = RefractoryDensity::new(dt, hazard).unwrap();
        for step in 0..200_000u32 {
            rd.step();
            if step % 20_000 == 0 {
                assert!(
                    (rd.mass() - 1.0).abs() < 1e-12,
                    "step {step}: mass drifted to {}",
                    rd.mass()
                );
            }
        }
        assert!((rd.mass() - 1.0).abs() < 1e-12, "final mass {}", rd.mass());
    }

    /// A fully synchronised population cannot fire before the dead time has run, and must fire the
    /// instant it has. The first non-zero activity is at step `R` exactly — an integer, so there is
    /// no tolerance to loosen.
    #[test]
    fn a_synchronised_population_fires_first_exactly_when_the_dead_time_ends() {
        let dt = 1e-4;
        let r = 37usize;
        let mut rd = RefractoryDensity::dead_time_renewal(dt, r as f64 * dt, 50.0, 500).unwrap();
        for step in 0..r {
            let a = rd.step();
            assert_eq!(a, 0.0, "step {step} fired during the dead time");
        }
        let a = rd.step();
        assert!(a > 0.0, "the population stayed silent past the dead time");
        // And the transient rings: the activity dips again before the population decorrelates.
        let mut minimum = f64::INFINITY;
        for _ in 0..r {
            minimum = minimum.min(rd.step());
        }
        assert!(minimum < a, "the activity never dipped after its first peak: min {minimum} vs {a}");
    }

    #[test]
    fn a_population_that_never_fires_has_no_mean_interval_rather_than_a_large_one() {
        let rd = RefractoryDensity::new(1e-4, vec![0.0; 100]).unwrap();
        assert!(rd.mean_interval().is_none());
        assert!(rd.interval_cv().is_none());
        assert!((rd.mass() - 1.0).abs() < 1e-15);
        assert_eq!(rd.tail_mass(), 0.0);
        assert_eq!(rd.steps(), 0);
    }

    #[test]
    fn the_refractory_density_refuses_a_malformed_hazard() {
        assert!(matches!(
            RefractoryDensity::new(1e-4, vec![]),
            Err(MeanFieldError::Empty { .. })
        ));
        assert!(matches!(
            RefractoryDensity::new(1e-4, vec![1.0, f64::NAN]),
            Err(MeanFieldError::NonFinite { what: "hazard", index: 1 })
        ));
        assert!(matches!(
            RefractoryDensity::new(1e-4, vec![-1.0]),
            Err(MeanFieldError::OutOfRange { what: "hazard", .. })
        ));
        assert!(RefractoryDensity::new(0.0, vec![1.0]).is_err());
        assert!(RefractoryDensity::dead_time_renewal(1e-4, -1.0, 1.0, 10).is_err());
        assert!(RefractoryDensity::dead_time_renewal(1e-4, 0.0, 1.0, 0).is_err());
        let mut rd = RefractoryDensity::new(1e-4, vec![1.0, 2.0]).unwrap();
        assert!(rd.set_density(&[1.0]).is_err());
        assert!(rd.set_density(&[0.0, 0.0]).is_err());
        assert!(rd.set_density(&[1.0, f64::NAN]).is_err());
        assert!(rd.set_density(&[1.0, 3.0]).is_ok());
        assert!((rd.density()[1] - 0.75).abs() < 1e-15);
    }

    /// A hazard that grows with age gives a MORE regular train than a flat one, which is the whole
    /// reason relative refractoriness exists. Checked against the two closed-form CVs rather than
    /// against a simulation.
    #[test]
    fn a_rising_hazard_is_more_regular_than_a_flat_one_at_the_same_rate() {
        let dt = 1e-4;
        let flat = RefractoryDensity::new(dt, vec![50.0; 2000]).unwrap();
        // Ramp chosen so the mean interval matches the flat case to within 1%, so the comparison is
        // about shape and not about rate.
        let ramp: Vec<f64> =
            (0..2000).map(|k| (2.0 * 50.0 * k as f64 * dt / (1.0 / 50.0)).min(5000.0)).collect();
        let ramped = RefractoryDensity::new(dt, ramp).unwrap();
        let (mf, mr) = (flat.mean_interval().unwrap(), ramped.mean_interval().unwrap());
        assert!((mf - mr).abs() / mf < 0.25, "the two processes differ in rate: {mf} vs {mr}");
        let (cf, cr) = (flat.interval_cv().unwrap(), ramped.interval_cv().unwrap());
        assert!(cr < 0.8 * cf, "a rising hazard gave CV {cr} against a flat {cf}");
    }

    /// A hazard above `1 / dt` cannot fire more than the whole bin, and the scheme has to say so
    /// rather than produce a firing probability above 1 and a negative density behind it. The
    /// activity saturates at exactly `1 / dt` and the density stays non-negative.
    #[test]
    fn a_hazard_faster_than_the_time_step_saturates_instead_of_going_negative() {
        let dt = 1e-4;
        // 5e5 Hz against a 1e-4 s step is p = 50 before clamping.
        let mut rd = RefractoryDensity::new(dt, vec![0.0, 0.0, 5e5, 5e5]).unwrap();
        for step in 0..100 {
            let a = rd.step();
            assert!(a >= 0.0 && a <= 1.0 / dt + 1e-9, "step {step}: activity {a} Hz");
            assert!(
                rd.density().iter().all(|&x| x >= 0.0),
                "step {step}: density went negative: {:?}",
                rd.density()
            );
            assert!((rd.mass() - 1.0).abs() < 1e-12, "step {step}: mass {}", rd.mass());
        }
        // Two silent bins then certain firing is a perfectly regular train: mean interval 3 bins,
        // CV zero. Both are exact.
        let want = 3.0 * dt;
        let got = rd.mean_interval().unwrap();
        assert!((got - want).abs() < 1e-18, "mean interval {got} s vs {want} s");
        assert!(rd.interval_cv().unwrap() < 1e-12, "cv {}", rd.interval_cv().unwrap());
    }

    #[test]
    fn the_refractory_density_reports_its_own_progress_and_how_much_ran_off_the_end() {
        let dt = 1e-4;
        let mut short = RefractoryDensity::dead_time_renewal(dt, 2e-3, 40.0, 60).unwrap();
        assert_eq!(short.steps(), 0);
        assert_eq!(short.activity(), 0.0);
        assert_eq!(short.density().len(), 60);
        let a = short.run(20_000);
        assert_eq!(short.steps(), 20_000);
        assert_eq!(a, short.activity());
        // A 6 ms age window against a 27 ms mean interval parks most of the population in the
        // absorbing bin, and `tail_mass` is what says the window is too short to read ages from.
        assert!(short.tail_mass() > 0.5, "tail mass {}", short.tail_mass());
        let mut long = RefractoryDensity::dead_time_renewal(dt, 2e-3, 40.0, 4000).unwrap();
        long.run(200_000);
        assert!(long.tail_mass() < 1e-6, "tail mass {}", long.tail_mass());
        // Both report the same stationary activity anyway, because the hazard is flat in the tail —
        // which is the condition under which the absorbing bin is exact rather than an error.
        assert!(
            (short.activity() - long.activity()).abs() / long.activity() < 1e-9,
            "{} Hz vs {} Hz",
            short.activity(),
            long.activity()
        );
    }

    /// Every error variant names the thing that was wrong, so a caller reading the message knows
    /// which argument to change. A `Display` that fell back to the debug format would still print
    /// something and would still pass a test that only checked it was non-empty.
    #[test]
    fn every_error_variant_names_what_was_wrong() {
        let cases: [(MeanFieldError, &str); 8] = [
            (MeanFieldError::NonFinite { what: "sigma", index: 3 }, "sigma"),
            (
                MeanFieldError::OutOfRange { what: "tau_m", value: -1.0, low: 0.0, high: 1.0 },
                "tau_m",
            ),
            (
                MeanFieldError::Ordering {
                    larger: "v_th",
                    smaller: "v_reset",
                    larger_value: 0.0,
                    smaller_value: 1.0,
                },
                "v_reset",
            ),
            (MeanFieldError::Empty { what: "hazard" }, "hazard"),
            (
                MeanFieldError::TooFewSamples { what: "power-law exponent", got: 1, need: 2 },
                "power-law exponent",
            ),
            (MeanFieldError::NoRefractoryPeriod, "refractory"),
            (MeanFieldError::NoBalancedState { nu_exc: -1.0, nu_inh: 2.0 }, "balance"),
            (MeanFieldError::Degenerate { determinant: 0.0, scale: 1.0 }, "determinant"),
        ];
        for (e, needle) in cases {
            let text = e.to_string();
            assert!(text.contains(needle), "{text:?} does not mention {needle}");
            let _: &dyn std::error::Error = &e;
        }
    }

    // ---------------------------------------------------------------------------------------
    // Criticality
    // ---------------------------------------------------------------------------------------

    /// ⭐⭐ The `-3/2` exponent is not fitted here, it is derived. The Borel probability times
    /// `k^(3/2) sqrt(2 pi)` must equal `1 - 1/(12k)` to the next Stirling order, and it does to
    /// 3.5e-7 at `k = 100`. Any error in the pmf, the logarithm, the factorial or the exponent shows
    /// up here immediately and at full strength.
    #[test]
    fn the_critical_borel_tail_is_exactly_the_minus_three_halves_power_law() {
        let p = BranchingProcess::new(1.0).unwrap();
        for &k in &[100u64, 1000, 10_000] {
            let got = p.borel_pmf(k).unwrap() * (k as f64).powf(1.5) * (2.0 * PI).sqrt();
            let want = 1.0 - 1.0 / (12.0 * k as f64);
            assert!(
                (got - want).abs() < 1e-6,
                "k = {k}: P(k) k^1.5 sqrt(2 pi) = {got}, Stirling says {want}"
            );
        }
        // Small k must NOT satisfy the asymptotic form, or the test above would be satisfied by any
        // smooth function near 1.
        let got = p.borel_pmf(2).unwrap() * 2.0f64.powf(1.5) * (2.0 * PI).sqrt();
        assert!((got - 1.0).abs() > 1e-3, "k = 2 already matched the asymptote at {got}");
    }

    /// The Borel distribution is a probability distribution with a known mean. Both are exact and
    /// both involve the whole pmf, so getting either one right by accident is not available.
    #[test]
    fn the_borel_distribution_normalises_and_has_the_published_mean() {
        for &m in &[0.3f64, 0.5, 0.8] {
            let p = BranchingProcess::new(m).unwrap();
            let (mut total, mut mean) = (0.0f64, 0.0f64);
            for k in 1..=2000u64 {
                let pk = p.borel_pmf(k).unwrap();
                total += pk;
                mean += k as f64 * pk;
            }
            assert!((total - 1.0).abs() < 1e-9, "m = {m}: the pmf sums to {total}");
            let want = p.mean_size().unwrap();
            assert!((mean - want).abs() / want < 1e-9, "m = {m}: mean {mean} vs 1/(1-m) = {want}");
            assert!((want - 1.0 / (1.0 - m)).abs() < 1e-15);
        }
        // Above criticality the distribution is defective and there is no mean, so both refuse.
        let super_critical = BranchingProcess::new(1.3).unwrap();
        assert!(super_critical.borel_pmf(5).is_none());
        assert!(super_critical.mean_size().is_none());
        assert!(BranchingProcess::new(1.0).unwrap().mean_size().is_none());
        assert!(BranchingProcess::new(0.5).unwrap().borel_pmf(0).is_none());
    }

    /// The extinction probability solves `q = exp(m (q - 1))`, which can be checked by substitution
    /// rather than against a table.
    #[test]
    fn the_extinction_probability_solves_its_own_fixed_point_equation() {
        for &m in &[1.1f64, 1.5, 2.0, 5.0] {
            let q = BranchingProcess::new(m).unwrap().extinction_probability();
            assert!(q > 0.0 && q < 1.0, "m = {m}: q = {q}");
            let residual = (m * (q - 1.0)).exp() - q;
            assert!(residual.abs() < 1e-12, "m = {m}: q = {q} leaves residual {residual}");
        }
        for &m in &[0.0f64, 0.5, 1.0] {
            assert_eq!(BranchingProcess::new(m).unwrap().extinction_probability(), 1.0);
        }
    }

    /// ⭐ The branching-parameter estimator on simulated avalanches, at four known values. A
    /// critical process must estimate to 1, and sub- and supercritical ones to their own `m`, within
    /// 1% — which is far tighter than the difference between the four, so the estimator is being
    /// asked to identify the process rather than to be in the right ballpark.
    #[test]
    fn the_branching_parameter_recovers_the_process_that_generated_the_avalanches() {
        for &m in &[0.7f64, 0.9, 1.0, 1.05] {
            let p = BranchingProcess::new(m).unwrap();
            let mut rng = Rng::new(0xA5A5_0000 + (m * 100.0) as u64);
            let avalanches: Vec<_> = (0..20_000).map(|_| p.avalanche(&mut rng, 20_000)).collect();
            let got = branching_parameter(&avalanches).unwrap();
            assert!((got - m).abs() < 0.01, "m = {m}: estimated {got}");
            // Beggs and Plenz's two-frame ratio is noisier but unbiased, so it needs a wider band.
            let bp = beggs_plenz_ratio(&avalanches).unwrap();
            assert!((bp - m).abs() < 0.05, "m = {m}: Beggs-Plenz ratio {bp}");
            // The mean size against 1/(1-m), where that exists. This is the second, independent
            // consequence of the same simulated process.
            if let Some(want) = p.mean_size() {
                let sizes: f64 =
                    avalanches.iter().map(|a| a.size() as f64).sum::<f64>() / avalanches.len() as f64;
                assert!((sizes - want) / want < 0.05, "m = {m}: mean size {sizes} vs {want}");
            }
        }
    }

    /// Dropping the terminal transition from the denominator is the classic bias in this estimator,
    /// and it turns a subcritical process into a critical-looking one. Here it is demonstrated
    /// rather than described: the same avalanches, estimated with and without their last generation,
    /// give 0.70 and 1.06.
    #[test]
    fn omitting_the_terminal_generation_would_make_a_subcritical_process_look_critical() {
        let m = 0.7;
        let p = BranchingProcess::new(m).unwrap();
        let mut rng = Rng::new(1234);
        let avalanches: Vec<_> = (0..20_000).map(|_| p.avalanche(&mut rng, 20_000)).collect();
        let correct = branching_parameter(&avalanches).unwrap();
        // The biased form: mark every avalanche as truncated, which removes the final zero
        // transition from the denominator exactly as conditioning on survival does.
        let biased_input: Vec<Avalanche> = avalanches
            .iter()
            .map(|a| Avalanche { generations: a.generations.clone(), complete: false })
            .collect();
        let biased = branching_parameter(&biased_input).unwrap();
        assert!((correct - m).abs() < 0.01, "the correct estimate is {correct}");
        assert!(biased > 1.0, "the biased estimate is {biased}, which does not reach criticality");
        assert!(biased - correct > 0.3, "the two estimates differ by only {}", biased - correct);
    }

    /// The power-law estimator calibrated on data whose exponent is known exactly, generated by
    /// inverse transform from a Pareto. The tolerance is derived, not chosen: the standard error of
    /// the maximum-likelihood estimate is `(alpha - 1) / sqrt(n)`, which at `n = 200 000` and
    /// `alpha = 2.5` is 0.00335, and the band here is six of them.
    #[test]
    fn the_power_law_estimator_recovers_a_known_exponent() {
        let mut rng = Rng::new(808);
        for &alpha in &[2.5f64, 3.5] {
            let n = 200_000usize;
            let x_min = 1.0f64;
            let samples: Vec<f64> = (0..n)
                .map(|_| {
                    let u = 1.0 - rng.next_f64();
                    x_min * u.powf(-1.0 / (alpha - 1.0))
                })
                .collect();
            let got = power_law_exponent(&samples, x_min).unwrap();
            let se = (alpha - 1.0) / (n as f64).sqrt();
            assert!(
                (got - alpha).abs() < 6.0 * se,
                "alpha = {alpha}: estimated {got}, standard error {se}"
            );
        }
        assert!(power_law_exponent(&[1.0, 2.0], 0.0).is_err());
        assert!(power_law_exponent(&[1.0], 1.0).is_err());
        assert!(power_law_exponent(&[1.0, f64::NAN], 1.0).is_err());
        assert!(power_law_exponent_discrete(&[1, 2, 3], 0).is_err());
        assert!(power_law_exponent_discrete(&[1, 2, 3], 100).is_err());
    }

    /// ⭐ Avalanche sizes from a critical branching process, fitted, against the `-3/2` the theory
    /// requires. The band `[1.45, 1.55]` is stated: the measured value at `x_min = 8` is 1.50, and
    /// the subcritical and supercritical processes must land clearly outside it, which is what makes
    /// the exponent a measurement of criticality rather than a number that always comes out near
    /// 1.5.
    #[test]
    fn the_avalanche_size_exponent_is_three_halves_at_criticality_and_not_otherwise() {
        let sizes_for = |m: f64, seed: u64| -> Vec<u64> {
            let p = BranchingProcess::new(m).unwrap();
            let mut rng = Rng::new(seed);
            (0..200_000).map(|_| p.avalanche(&mut rng, 20_000).size()).collect()
        };
        let critical = sizes_for(1.0, 11);
        let alpha = power_law_exponent_discrete(&critical, 8).unwrap();
        assert!(
            (1.45..=1.55).contains(&alpha),
            "critical avalanches fitted to alpha = {alpha}, not the derived 1.5"
        );
        // x_min = 1 is where Clauset et al.'s continuity correction is weakest, and the fit is
        // visibly worse. Pinning it here keeps the doc's claim honest.
        let at_one = power_law_exponent_discrete(&critical, 1).unwrap();
        assert!(at_one < 1.47, "x_min = 1 gave {at_one}, so the correction's weakness is not real");
        // And the two estimators fail in OPPOSITE directions at a small cut, converging as it
        // rises. This is the table in `power_law_exponent_discrete`'s doc, measured here.
        let counts: Vec<f64> = critical.iter().map(|&x| x as f64).collect();
        let continuous_at_one = power_law_exponent(&counts, 1.0).unwrap();
        let continuous_at_eight = power_law_exponent(&counts, 8.0).unwrap();
        assert!(
            continuous_at_one > 1.6,
            "the uncorrected estimator at x_min = 1 gave {continuous_at_one}, not the high bias the              doc claims"
        );
        assert!(
            (continuous_at_eight - alpha).abs() < 0.05,
            "at x_min = 8 the two estimators differ by {}, so they have not converged",
            (continuous_at_eight - alpha).abs()
        );

        let sub = power_law_exponent_discrete(&sizes_for(0.9, 12), 8).unwrap();
        assert!(sub > 1.6, "subcritical avalanches fitted to {sub}, inside the critical band");
        let sup = power_law_exponent_discrete(&sizes_for(1.1, 13), 8).unwrap();
        assert!(sup < 1.4, "supercritical avalanches fitted to {sup}, inside the critical band");
    }

    /// ⭐⭐ Wilting and Priesemann's result, reproduced: under heavy subsampling the lag-1
    /// correlation collapses and the multistep regression does not. The generating process has an
    /// autocorrelation of exactly `m^k`, so the regression is being checked against an identity.
    #[test]
    fn multistep_regression_survives_subsampling_where_the_lag_one_correlation_does_not() {
        for &m in &[0.8f64, 0.95] {
            let h = 20.0 * (1.0 - m);
            let mut rng = Rng::new(0x5757_0000 + (m * 100.0) as u64);
            let mut n = 20u64;
            let mut full = Vec::with_capacity(400_000);
            for _ in 0..400_000 {
                n = poisson_count(&mut rng, m * n as f64 + h).unwrap();
                full.push(n as f64);
            }
            let m_full = multistep_regression(&full, 20, 0.02).unwrap();
            assert!((m_full - m).abs() < 0.02, "m = {m}: fully sampled estimate {m_full}");

            // Record one neuron in twenty.
            let sub: Vec<f64> = full
                .iter()
                .map(|&x| {
                    let mut c = 0u32;
                    for _ in 0..(x as u32) {
                        if rng.next_f64() < 0.05 {
                            c += 1;
                        }
                    }
                    f64::from(c)
                })
                .collect();
            let m_sub = multistep_regression(&sub, 20, 0.02).unwrap();
            assert!((m_sub - m).abs() < 0.02, "m = {m}: subsampled estimate {m_sub}");

            // The naive lag-1 correlation on the same subsampled series.
            let t = sub.len();
            let mean = sub.iter().sum::<f64>() / t as f64;
            let var = sub.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / t as f64;
            let r1 = (0..t - 1).map(|i| (sub[i] - mean) * (sub[i + 1] - mean)).sum::<f64>()
                / (t - 1) as f64
                / var;
            assert!(
                r1 < m - 0.3,
                "m = {m}: the lag-1 correlation on subsampled data was {r1}, which is not the \
                 collapse the method exists to fix"
            );
        }
    }

    #[test]
    fn multistep_regression_refuses_rather_than_fitting_one_point() {
        assert!(multistep_regression(&[1.0, 2.0], 5, 0.02).is_none());
        assert!(multistep_regression(&[1.0; 100], 5, 0.02).is_none(), "zero variance");
        assert!(multistep_regression(&[1.0, 2.0, 3.0, f64::NAN], 5, 0.02).is_none());
        assert!(multistep_regression(&[1.0, 2.0, 1.0, 2.0, 1.0, 2.0], 0, 0.02).is_none());
        // White noise has no correlation at any lag, so there is nothing to regress and the answer
        // is a refusal rather than a number near zero.
        let mut rng = Rng::new(3);
        let noise: Vec<f64> = (0..10_000).map(|_| rng.next_f64()).collect();
        assert!(multistep_regression(&noise, 20, 0.05).is_none());
    }

    #[test]
    fn avalanche_bookkeeping_is_what_it_says() {
        let a = Avalanche { generations: vec![1, 3, 2], complete: true };
        assert_eq!(a.size(), 6);
        assert_eq!(a.duration(), 3);
        assert!(branching_parameter(&[]).is_none());
        assert!(beggs_plenz_ratio(&[]).is_none());
        // A single-generation avalanche contributes a zero to the ratio, not nothing.
        let singles = vec![Avalanche { generations: vec![1], complete: true }; 4];
        assert_eq!(beggs_plenz_ratio(&singles), Some(0.0));
        assert_eq!(branching_parameter(&singles), Some(0.0));
        assert!(BranchingProcess::new(-1.0).is_err());
        assert!(BranchingProcess::new(f64::NAN).is_err());
        // A truncated avalanche drops its last generation from the denominator.
        let truncated = vec![Avalanche { generations: vec![1, 2, 4], complete: false }];
        assert_eq!(branching_parameter(&truncated), Some(6.0 / 3.0));
    }

    /// A supercritical process produces both kinds of avalanche and labels them correctly: the ones
    /// that hit the cap are incomplete and are at least as large as it, the ones that ended are
    /// smaller. The flag is what [`branching_parameter`] depends on, so a simulator that set it
    /// wrong would bias every estimate downward with nothing else looking different.
    #[test]
    fn a_capped_avalanche_reports_itself_incomplete() {
        let p = BranchingProcess::new(1.5).unwrap();
        let mut rng = Rng::new(0xCA9E);
        let (mut truncated, mut complete) = (0u32, 0u32);
        for _ in 0..3_000 {
            let a = p.avalanche(&mut rng, 500);
            if a.complete {
                complete += 1;
                assert!(a.size() < 500, "a completed avalanche reached {}", a.size());
                assert!(a.generations[a.generations.len() - 1] > 0);
            } else {
                truncated += 1;
                assert!(a.size() >= 500, "a truncated avalanche only reached {}", a.size());
            }
        }
        assert!(truncated > 100 && complete > 100, "{truncated} truncated, {complete} complete");
        // Extinction probability at m = 1.5 is about 0.417, and the completed fraction must be
        // near it: a supercritical avalanche either dies early or runs to the cap.
        let q = p.extinction_probability();
        let measured = f64::from(complete) / 3_000.0;
        assert!((measured - q).abs() < 0.03, "completed fraction {measured} vs extinction {q}");
    }
}
