//! Spike initiation as a **nonlinearity** rather than a fiat threshold: the exponential and
//! quadratic integrate-and-fire family.
//!
//! # The lesson
//!
//! A leaky integrate-and-fire neuron does not generate a spike. It integrates, and then a line of
//! code notices that `v >= v_th` and writes one down. The threshold is an author's decision, not a
//! consequence of the dynamics, and everything that depends on the *shape* of the spike onset —
//! how precisely a cell locks to a fast input, how a population responds to a step, where the
//! firing-rate curve turns on — is therefore decided by that line of code too.
//!
//! Real membranes do something else. Sodium channels open as a steep function of voltage, and once
//! enough of them are open the current they carry opens more of them. That is a **regenerative**
//! process: positive feedback whose solution diverges in finite time. The models in this module
//! keep that feedback term and let the divergence *be* the spike. Nothing is compared against a
//! threshold to decide whether a spike happened; the trajectory leaves every bounded set, and the
//! cutoff `v_peak` only says where to stop following it.
//!
//! Two nonlinearities are in use, and they are the two leading terms of the same story:
//!
//! - **Quadratic** — `dV/dt ∝ (V - v_rest)(V - v_c)`. This is not a guess. Any Type I membrane
//!   (one whose firing rate can be made arbitrarily small by lowering the current) is, near the
//!   saddle-node bifurcation where firing begins, *exactly* this equation after a change of
//!   variables. Ermentrout & Kopell proved it; Ermentrout, *Neural Comput.* 8:979–1001, 1996 is
//!   the readable version. [`Qif`] and [`Theta`] are two coordinate systems for that one system.
//! - **Exponential** — `dV/dt ∝ ... + Δ_T exp((V - V_T)/Δ_T)`. Fourcaud-Trocmé, Hansel, van
//!   Vreeswijk & Brunel, *J. Neurosci.* 23:11628–11640, 2003 derived it from the activation curve
//!   of a sodium conductance and showed it reproduces the high-frequency response of a
//!   `Hodgkin`-`Huxley` model, which the quadratic and the LIF both get wrong. Badel, Lefort,
//!   Brette, Petersen, Gerstner & Richardson, *J. Neurophysiol.* 99:656–666, 2008 then measured the
//!   term directly in cortical pyramidal cells; this implementation did not re-derive their fit,
//!   and quotes only that their `Δ_T` was of order 1 mV.
//!
//! # What it buys, and what it costs
//!
//! **Buys.** A rheobase in closed form. Type I excitability, so an arbitrarily low firing rate is
//! reachable instead of the LIF's jump from silence to a finite rate. Correct spike-onset
//! sharpness, which is what sets a population's cutoff frequency. And — with one adaptation
//! current bolted on — a taxonomy of firing patterns that covers most of what cortical cells are
//! observed to do, from four extra numbers ([`AdEx`], [`FiringPattern`]).
//!
//! **Costs.** The equation is nonlinear, so there is no exponential-Euler trick and, for the
//! exponential members, no exact flow: [`Eif`] and [`AdEx`] integrate by fourth-order
//! Runge-Kutta. **Every model here declares [`crate::neuron::Neuron::EXACT_OVER_GAPS`] false**, so
//! [`crate::sim::Sim::new`] refuses to run any of them event-driven. That refusal is the honest
//! price of the nonlinearity and it is charged at the type level rather than in a comment.
//!
//! [`Qif`] is false for a **different and more interesting reason**, and it is worth reading before
//! anyone writes another model. Its flow is exact and composes, so the letter of the constant is
//! satisfied and `the_qif_flow_composes_across_a_gap` holds it to 1e-14 V. What fails is a
//! precondition the constant does not express: [`crate::sim::Sim`] jumps a quiet gap with one
//! zero-input step and **discards the spike that step reports**, on the stated ground that a quiet
//! interval cannot produce one. For a leaky model that ground is solid — the potential decays
//! monotonically to a rest below threshold. A quadratic model is **bistable**: a synapse that
//! leaves the membrane above `v_c` makes it fire with no further input at all, about 14 ms later.
//! Measured on the two-cell chain of `the_simulator_enforces_the_gap_property_this_module_declares`
//! (0.1 ms per tick, 20,000 ticks, 800 pA into the first cell only), a clocked run gives 46 spikes
//! and an event-driven one 23: 23 presynaptic spikes either way, and 23 postsynaptic spikes in the
//! first run against **none** in the second. The run reports success either way. The clocked half
//! of those counts is asserted by that test; the event-driven half can only be produced by flipping
//! the constant below in a throwaway copy, which is where it comes from and why it is quoted here
//! rather than tested.
//!
//! So `EXACT_OVER_GAPS` is necessary and **not sufficient**. The sufficient condition is that plus
//! "no spike during a quiet interval", and this review did not locate the second half stated
//! anywhere in the crate before this module needed it. The fix belongs in `sim`, which should
//! propagate the bool `catch_up` currently throws away; until it does, the constant is the only
//! thing between a user and a plausible, wrong raster.
//!
//! # Units
//!
//! SI at every interface, like the rest of the crate: `dt` in seconds, current in amperes,
//! potential in volts, conductance in siemens, capacitance in farads. The one dimensionless frame
//! kept as such is the canonical pair `dy/ds = y² + η` (and its circle twin
//! `dθ/ds = (1 - cos θ) + (1 + cos θ)η`), because those two equations are the theorem — rescaling
//! them into millivolts would hide the fact that every Type I neuron reduces to them.
//! [`Qif::canonical_y`] and [`Qif::eta`] are the exact change of variables, and [`Theta`] takes a
//! `tau` and an `i_ref` so its dimensionless input can be reached from amperes.
//!
//! # What is checked
//!
//! - [`Qif`] and [`Theta`] are shown to be the same system: `V = tan(θ/2)` maps one trajectory onto
//!   the other. The check is run twice — once against `Theta`'s independent Runge-Kutta stepper,
//!   which shares no code with `Qif`'s exact flow, and once closed form against closed form.
//! - [`Eif`] relaxes to [`crate::neuron::Lif`] as `Δ_T -> 0`, with the error shrinking.
//! - [`Qif::isi`] is elementary and the time-stepper is held to it; [`Theta::isi`] is
//!   `τ·π/√η`, the textbook Type I rate.
//! - [`Eif::isi`] is **not** a closed form and does not claim to be. There is no elementary
//!   antiderivative of `1/F(V)` for the exponential model; the quantity computed is the exact
//!   integral `∫ C dV / F(V)`, evaluated by adaptive Simpson quadrature. It is an independent
//!   reference for the time-stepper — quadrature and time-stepping share no code — and it is
//!   labelled as quadrature everywhere it appears.
//! - [`Eif::fixed_points`] *is* closed form, through the Lambert `W` function, and the test plugs
//!   the roots back into the drift.
//! - Each [`FiringPattern`] is asserted to produce the pattern it is named after, numerically:
//!   lengthening intervals for adaptation, a bimodal interval distribution for bursting, silence
//!   after a few spikes for the transient cell.
//!
//! # Cross-fabric note
//!
//! [`AdEx`] is one of the few spiking models implemented in analog silicon rather than emulated:
//! the `BrainScaleS`-2 neuron circuit realises it directly, which is why the model keeps appearing
//! in hardware papers long after simpler ones would have done. This crate has no hardware and
//! measures nothing; the note is here so a reader knows which model to reach for if they ever get
//! near a wafer.

use crate::neuron::{Lif, Neuron};

/// Largest value the exponential term's argument `(V - V_T)/Δ_T` is allowed to take, 50.
///
/// `exp(50) ≈ 5.2e21`, which keeps every product in the drift finite in `f64` while
/// `exp(710)` would not. The clamp can only act **above** `V_T + 50·Δ_T`, where the drift already
/// exceeds `g_L·Δ_T·5.2e21/C` — for the default parameters, 5e20 V/s. The time a trajectory spends
/// above that point before reaching any plausible `v_peak` is under `1e-19` s, so the clamp changes
/// no reported interval by more than one part in `1e17`. It exists because an `inf` inside a
/// Runge-Kutta stage becomes a `NaN` one line later, and a `NaN` membrane potential does not fail
/// loudly — it reports zero spikes.
///
/// **It does a second job**, and a reader changing it should know both: [`Eif::isi`] truncates its
/// quadrature at `V_T + 50·Δ_T` for the same reason, because the integrand `C/F(V)` is below
/// `1e-20` s per volt there. Lowering this constant therefore shortens an interval as well as
/// capping an exponential, and raising it past 709 replaces a clamped drift with an infinite one.
pub const EXP_ARG_LIMIT: f64 = 50.0;

/// `π`, spelled out rather than imported so the constant a reader compares against is visible.
const PI: f64 = std::f64::consts::PI;

/// Why a model could not answer.
///
/// This module returns `Result` where [`crate::neuron::Lif::isi`] returns `Option`, because the
/// interesting failure here is not "no answer" but "no answer, and here is the current at which
/// there would be one". A user who asked for a firing rate and got [`ModelError::NoFiring`] is
/// handed the rheobase in the same value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ModelError {
    /// A parameter or an input was `NaN` or infinite.
    NotFinite {
        /// Which quantity, by its field name, e.g. `"delta_t"` or `"i"`.
        what: &'static str,
        /// The offending value, so the caller can print what it actually sent.
        value: f64,
    },
    /// A quantity that must be strictly positive was zero or negative.
    NotPositive {
        /// Which quantity, by its field name, e.g. `"tau_m"`.
        what: &'static str,
        /// The offending value.
        value: f64,
    },
    /// Two potentials were in the wrong order, e.g. a spike cutoff at or below the reset.
    Disordered {
        /// Which pair, e.g. `"v_peak <= v_reset"`.
        what: &'static str,
        /// The lower of the two, volts.
        lower: f64,
        /// The upper of the two, volts.
        upper: f64,
    },
    /// The constant current leaves the neuron with a stable fixed point it never escapes, so there
    /// is no inter-spike interval at all — not a long one.
    NoFiring {
        /// The current that was asked about, amperes.
        i: f64,
        /// The current above which no fixed point exists, amperes. Above it the neuron fires for
        /// any initial condition. Note that a model whose reset sits **above** the unstable fixed
        /// point can fire below this value; see [`Qif::fixed_points`].
        rheobase: f64,
    },
    /// A formula's denominator vanished, so the closed form it names does not exist here.
    Degenerate {
        /// Which quantity was zero, e.g. `"g_l + a"`.
        what: &'static str,
    },
}

impl core::fmt::Display for ModelError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotFinite { what, value } => write!(f, "{what} is not finite ({value})"),
            Self::NotPositive { what, value } => {
                write!(f, "{what} must be strictly positive, got {value}")
            }
            Self::Disordered { what, lower, upper } => {
                write!(f, "{what}: lower {lower} V, upper {upper} V")
            }
            Self::NoFiring { i, rheobase } => write!(
                f,
                "{i} A leaves a stable fixed point, so there is no interval; rheobase is {rheobase} A"
            ),
            Self::Degenerate { what } => write!(f, "{what} is zero, so the closed form does not exist"),
        }
    }
}

/// See the note on [`crate::net::NetError`]: a library error that cannot cross a `Box<dyn Error>`
/// boundary forces its callers to reach for `.unwrap()`.
impl std::error::Error for ModelError {}

/// Step any neuron in this crate, refusing a non-finite `dt` or current instead of letting it into
/// the state.
///
/// [`Neuron::step`] returns a `bool` and has nowhere to put an error, which is the right trade for
/// the inner loop of a simulation — the check would run on every neuron on every tick to catch a
/// mistake made once, at the boundary. This is that boundary. Validate the input here, then step in
/// the loop.
///
/// The neuron is **not touched** when the input is rejected, so a caller that ignores the error
/// still has a usable model rather than a poisoned one.
///
/// # Errors
///
/// [`ModelError::NotFinite`] naming `"dt"` or `"i"`, or [`ModelError::NotPositive`] for a `dt` that
/// is zero or negative.
pub fn try_step<N: Neuron>(n: &mut N, dt: f64, i: f64) -> Result<bool, ModelError> {
    if !dt.is_finite() {
        return Err(ModelError::NotFinite { what: "dt", value: dt });
    }
    if dt <= 0.0 {
        return Err(ModelError::NotPositive { what: "dt", value: dt });
    }
    if !i.is_finite() {
        return Err(ModelError::NotFinite { what: "i", value: i });
    }
    Ok(n.step(dt, i))
}

/// `exp`, with the argument clamped at [`EXP_ARG_LIMIT`]. See that constant for the error bound.
fn exp_clamped(x: f64) -> f64 {
    if x > EXP_ARG_LIMIT { EXP_ARG_LIMIT.exp() } else { x.exp() }
}

/// Reject a non-finite parameter by name.
fn finite(what: &'static str, value: f64) -> Result<(), ModelError> {
    if value.is_finite() { Ok(()) } else { Err(ModelError::NotFinite { what, value }) }
}

/// Reject a non-positive parameter by name.
fn positive(what: &'static str, value: f64) -> Result<(), ModelError> {
    finite(what, value)?;
    if value > 0.0 { Ok(()) } else { Err(ModelError::NotPositive { what, value }) }
}

/// Fold an angle into `(-π, π]`.
fn wrap_pi(x: f64) -> f64 {
    let two_pi = 2.0 * PI;
    let mut r = x % two_pi;
    if r > PI {
        r -= two_pi;
    } else if r <= -PI {
        r += two_pi;
    }
    r
}

/// Where the canonical flow `dy/ds = y² + η` ended up after `h` units of dimensionless time.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Flow {
    /// The trajectory stayed finite and ended here.
    Finite(f64),
    /// The trajectory diverged to `+∞` at this time, `0 <= at <= h`. That divergence **is** the
    /// spike; it is not a numerical failure.
    Diverged {
        /// Dimensionless time of the divergence, measured from the start of the step.
        at: f64,
    },
}

/// Exact solution of `dy/ds = y² + η` over `h`, the Riccati flow shared by the quadratic models.
///
/// Three branches, all elementary:
/// - `η > 0`: `y = b·tan(b·s + φ₀)`, `b = √η`, `φ₀ = atan(y₀/b)`. Divergence when the phase
///   reaches `π/2`, which gives the spike time in closed form.
/// - `η = 0`: `y = y₀/(1 - y₀·s)`, diverging at `s = 1/y₀` when `y₀ > 0`.
/// - `η < 0`: fixed points at `y = ∓a` (stable, unstable), `a = √(-η)`. Divergence only from
///   `y₀ > a`, at `s = atanh(a/y₀)/a`.
///
/// **The `η < 0` branch is written twice, and which one runs is decided by which fixed point `y₀`
/// is nearer.** The two forms are the same solution and neither is an approximation; they fail in
/// opposite places, and each branch is used where the other one does.
///
/// - `y₀ <= 0`, near the **stable** point: `y = a(y₀ - a·T)/(a - y₀·T)`, `T = tanh(a·s)`. At
///   `y₀ = -a` the numerator and denominator both carry the factor `1 + T` and the answer is
///   `-a` whatever `T` rounds to.
/// - `y₀ > 0`, near the **unstable** point: `u = (y₀ - a)/(y₀ + a)` obeys `du/ds = 2a·u` exactly,
///   so `u(s) = u₀·e^{2as}` and `y = a(1 + u)/(1 - u)`. This form keeps the difference `y₀ - a`,
///   which is exact for a `y₀` within a factor of two of `a`; the `tanh` form instead evaluates
///   `a - y₀·T` with `T` rounded to exactly 1.0 for `a·s ≳ 19`, which is `0/0` **on** the unstable
///   fixed point and the wrong sign one ulp above it. The overflow that argues for `tanh` — `u`
///   running to `-inf` deep inside the stable point — is handled by returning `-a`, which is that
///   limit.
fn canonical_flow(y0: f64, eta: f64, h: f64) -> Flow {
    if eta > 0.0 {
        let b = eta.sqrt();
        let phi0 = (y0 / b).atan();
        let phi1 = phi0 + b * h;
        if phi1 >= 0.5 * PI {
            return Flow::Diverged { at: (0.5 * PI - phi0) / b };
        }
        Flow::Finite(b * phi1.tan())
    } else if eta == 0.0 {
        if y0 > 0.0 && h >= 1.0 / y0 {
            return Flow::Diverged { at: 1.0 / y0 };
        }
        Flow::Finite(y0 / (1.0 - y0 * h))
    } else {
        let a = (-eta).sqrt();
        if y0 > a {
            // `atanh(a/y₀)` written as `½·ln1p(2a/(y₀ - a))`. The two are the same number, but
            // `a/y₀` rounds to within an ulp of 1 next to the unstable fixed point and `atanh`
            // multiplies that rounding by `1/(1 - x²)`, while `y₀ - a` is exact there and
            // `ln_1p` is accurate at both ends of its range.
            let h_star = (2.0 * a / (y0 - a)).ln_1p() / (2.0 * a);
            if h >= h_star {
                return Flow::Diverged { at: h_star };
            }
        }
        if y0 > 0.0 {
            let u0 = (y0 - a) / (y0 + a);
            if u0 == 0.0 {
                // `y₀` IS the unstable fixed point, so the flow is the point — and `u₀ · e^{2ah}`
                // would be `0 · inf = NaN` for a long enough step.
                return Flow::Finite(a);
            }
            let u = u0 * (2.0 * a * h).exp();
            if !u.is_finite() {
                // `u₀ < 0` and `e^{2ah}` overflowed: the trajectory is deep inside the stable
                // fixed point, and `a(1 + u)/(1 - u) -> -a` as `u -> -∞`.
                return Flow::Finite(-a);
            }
            if u >= 1.0 {
                // Rounded onto the divergence from below, so `h` is within an ulp of `h_star`.
                return Flow::Diverged { at: h };
            }
            return Flow::Finite(a * (1.0 + u) / (1.0 - u));
        }
        if y0 == -a {
            return Flow::Finite(-a);
        }
        let t = (a * h).tanh();
        Flow::Finite(a * (y0 - a * t) / (a - y0 * t))
    }
}

// ---------------------------------------------------------------------------------------------
// Quadratic integrate-and-fire
// ---------------------------------------------------------------------------------------------

/// Quadratic integrate-and-fire — the canonical Type I neuron, in volts.
///
/// ```text
/// tau_m dV/dt = (V - v_rest)(V - v_c) / (v_c - v_rest) + r_m I
/// ```
///
/// Latham, Richmond, Nelson & Nirenberg, *J. Neurophysiol.* 83:808–827, 2000 use this form; the
/// normalisation by `v_c - v_rest` is the one that makes the model linearise to a [`Lif`] of time
/// constant `tau_m` near `v_rest`, and to its mirror image near `v_c`. `v_rest` is the stable
/// resting potential at zero current and `v_c` the unstable one the membrane must be pushed past.
///
/// # The exact reduction, and the closed forms that follow
///
/// With `Δ = v_c - v_rest`, `v_mid = (v_rest + v_c)/2`, `y = (V - v_mid)/Δ` and `s = t/tau_m`, the
/// equation becomes **exactly**
///
/// ```text
/// dy/ds = y² + η,        η = r_m I / Δ - 1/4
/// ```
///
/// with no approximation anywhere — it is an affine change of variables. That is the equation every
/// Type I membrane reduces to near onset, and everything below is read off it:
///
/// - **Rheobase** `I_rheo = Δ / (4 r_m)`, the current at which `η` crosses zero and the two fixed
///   points annihilate. See [`Qif::rheobase`].
/// - **Fixed points** at `V = v_mid ∓ Δ√(-η)` for `η < 0`. See [`Qif::fixed_points`].
/// - **Interval** `T = tau_m/√η·[atan(y_peak/√η) - atan(y_reset/√η)]` for `η > 0`, plus `t_ref`.
///   See [`Qif::isi`]. As the cutoff and reset go to `±∞` this collapses to the textbook Type I
///   rate `f = √η / (π tau_m)`.
/// - **Square-root onset**: just above rheobase, `√η ∝ √(I - I_rheo)`, so the firing rate rises
///   from zero like a square root instead of jumping, which is what "Type I" means.
///
/// # Bistability is real here
///
/// Below rheobase a `Qif` whose `v_reset` sits **above** the unstable fixed point still fires
/// forever: the reset drops it back into the escaping region. [`Qif::isi`] returns an interval in
/// that regime rather than refusing, and [`ModelError::NoFiring`]'s doc says so. A model that
/// treated "below rheobase" and "silent" as synonyms would be wrong for exactly the parameter sets
/// people use to build bistable working-memory units.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Qif {
    /// Membrane time constant, seconds. The linearised rate of return to `v_rest`.
    pub tau_m: f64,
    /// Stable resting potential at zero current, volts.
    pub v_rest: f64,
    /// Unstable (critical) potential at zero current, volts. Must exceed `v_rest`; the membrane
    /// runs away above it.
    pub v_c: f64,
    /// Membrane resistance, ohms. `r_m · I` is the current expressed as a potential.
    pub r_m: f64,
    /// Spike cutoff, volts. The true trajectory diverges in finite time; this is where following it
    /// stops. Raising it changes the interval by the time spent between the old and new value,
    /// which the closed form prices exactly.
    pub v_peak: f64,
    /// Post-spike potential, volts. Must be below `v_peak`.
    pub v_reset: f64,
    /// Absolute refractory period, seconds. Bounds the rate at `1/t_ref`.
    pub t_ref: f64,
    /// Current membrane potential, volts.
    pub v: f64,
    /// Seconds left in the refractory period; zero when free to integrate.
    pub refractory: f64,
}

impl Default for Qif {
    /// A cortical-ish default chosen so that [`Qif::matching_theta`] has round numbers: 20 ms,
    /// −65 mV rest, −50 mV critical, 10 MΩ, +30 mV cutoff, 2 ms refractory. Rheobase is then
    /// exactly `15 mV / (4 · 10 MΩ) = 375 pA`. These are textbook-range round numbers, not a fit to
    /// any cell, and they are stated here so a figure made with the default is reproducible from
    /// the documentation alone.
    fn default() -> Self {
        Self {
            tau_m: 20e-3,
            v_rest: -65e-3,
            v_c: -50e-3,
            r_m: 10e6,
            v_peak: 30e-3,
            v_reset: -65e-3,
            t_ref: 2e-3,
            v: -65e-3,
            refractory: 0.0,
        }
    }
}

impl Qif {
    /// Build one, refusing parameters that have no model behind them.
    ///
    /// The membrane starts at `v_rest` and out of refractory.
    ///
    /// # Errors
    ///
    /// [`ModelError::NotFinite`] or [`ModelError::NotPositive`] naming the field, or
    /// [`ModelError::Disordered`] when `v_c <= v_rest` or `v_peak <= v_reset`.
    pub fn new(
        tau_m: f64,
        v_rest: f64,
        v_c: f64,
        r_m: f64,
        v_peak: f64,
        v_reset: f64,
        t_ref: f64,
    ) -> Result<Self, ModelError> {
        positive("tau_m", tau_m)?;
        positive("r_m", r_m)?;
        finite("v_rest", v_rest)?;
        finite("v_c", v_c)?;
        finite("v_peak", v_peak)?;
        finite("v_reset", v_reset)?;
        finite("t_ref", t_ref)?;
        if t_ref < 0.0 {
            return Err(ModelError::NotPositive { what: "t_ref", value: t_ref });
        }
        if v_c <= v_rest {
            return Err(ModelError::Disordered {
                what: "v_c <= v_rest",
                lower: v_rest,
                upper: v_c,
            });
        }
        if v_peak <= v_reset {
            return Err(ModelError::Disordered {
                what: "v_peak <= v_reset",
                lower: v_reset,
                upper: v_peak,
            });
        }
        Ok(Self { tau_m, v_rest, v_c, r_m, v_peak, v_reset, t_ref, v: v_rest, refractory: 0.0 })
    }

    /// Distance between the two zero-current fixed points, volts. The voltage scale of the
    /// canonical reduction.
    #[must_use]
    pub fn delta(&self) -> f64 {
        self.v_c - self.v_rest
    }

    /// Midpoint of the two zero-current fixed points, volts. The origin of the canonical reduction.
    #[must_use]
    pub fn v_mid(&self) -> f64 {
        0.5 * (self.v_rest + self.v_c)
    }

    /// The canonical input `η = r_m I / Δ - 1/4`, dimensionless. Positive means firing.
    #[must_use]
    pub fn eta(&self, i: f64) -> f64 {
        self.r_m * i / self.delta() - 0.25
    }

    /// The current state in canonical coordinates, `y = (V - v_mid)/Δ`, dimensionless.
    #[must_use]
    pub fn canonical_y(&self) -> f64 {
        (self.v - self.v_mid()) / self.delta()
    }

    /// Put the membrane at a canonical coordinate, volts computed as `v_mid + Δ·y`.
    pub fn set_canonical_y(&mut self, y: f64) {
        self.v = self.v_mid() + self.delta() * y;
    }

    /// `dV/dt` in volts per second at potential `v` under current `i`.
    #[must_use]
    pub fn drift(&self, v: f64, i: f64) -> f64 {
        ((v - self.v_rest) * (v - self.v_c) / self.delta() + self.r_m * i) / self.tau_m
    }

    /// Rheobase, amperes: `Δ / (4 r_m)`, the current at which the two fixed points annihilate.
    #[must_use]
    pub fn rheobase(&self) -> f64 {
        self.delta() / (4.0 * self.r_m)
    }

    /// The two fixed points under constant current `i`, volts, as `(stable, unstable)`.
    ///
    /// `None` above rheobase, where there are none and the membrane escapes from anywhere. Below
    /// it they are `v_mid ∓ Δ√(-η)`, exactly; the test plugs them back into [`Qif::drift`].
    #[must_use]
    pub fn fixed_points(&self, i: f64) -> Option<(f64, f64)> {
        let eta = self.eta(i);
        if eta >= 0.0 || !eta.is_finite() {
            return None;
        }
        let a = (-eta).sqrt();
        Some((self.v_mid() - self.delta() * a, self.v_mid() + self.delta() * a))
    }

    /// Inter-spike interval under constant current `i`, seconds, in closed form.
    ///
    /// `tau_m/√η·[atan(y_peak/√η) - atan(y_reset/√η)] + t_ref` above rheobase; the `atanh` form
    /// below it when `v_reset` is above the unstable fixed point, which is a firing regime too.
    ///
    /// # Errors
    ///
    /// [`ModelError::NotFinite`] for a non-finite `i`, or [`ModelError::NoFiring`] carrying the
    /// rheobase when the trajectory from `v_reset` settles on a fixed point below `v_peak`.
    pub fn isi(&self, i: f64) -> Result<f64, ModelError> {
        finite("i", i)?;
        let eta = self.eta(i);
        let d = self.delta();
        let y_r = (self.v_reset - self.v_mid()) / d;
        let y_p = (self.v_peak - self.v_mid()) / d;
        let rheobase = self.rheobase();
        let t = if eta > 0.0 {
            let b = eta.sqrt();
            self.tau_m / b * ((y_p / b).atan() - (y_r / b).atan())
        } else if eta == 0.0 {
            if y_r <= 0.0 {
                return Err(ModelError::NoFiring { i, rheobase });
            }
            self.tau_m * (1.0 / y_r - 1.0 / y_p)
        } else {
            let a = (-eta).sqrt();
            if y_r <= a {
                return Err(ModelError::NoFiring { i, rheobase });
            }
            let f = |y: f64| (y - a) / (y + a);
            self.tau_m / (2.0 * a) * (f(y_p) / f(y_r)).ln()
        };
        Ok(t + self.t_ref)
    }

    /// Steady-state firing rate under constant current `i`, hertz.
    ///
    /// # Errors
    ///
    /// As [`Qif::isi`].
    pub fn rate(&self, i: f64) -> Result<f64, ModelError> {
        self.isi(i).map(|t| 1.0 / t)
    }

    /// The theta neuron that is this `Qif`, exactly.
    ///
    /// The correspondence `V = v_mid + Δ·tan(θ/2)` maps one trajectory onto the other, and the
    /// returned [`Theta`] is set up so that it does: same `tau`, `bias = -1/4`, and
    /// `i_ref = Δ / r_m`, which makes both models' `η` the same function of the same current.
    ///
    /// **The correspondence is exact for the flow, not for the bookkeeping.** `Qif`'s finite
    /// `v_peak` and `v_reset` have no counterpart on the circle, where the spike *is* the passage
    /// through `θ = π` and the reset is automatic. Compare the two between spikes, or push
    /// `v_peak` and `v_reset` far out.
    ///
    /// **`t_ref` is dropped too**, and that one is not a limit you can take: [`Theta`] has no
    /// refractory period at all, so this `Qif`'s dead time after a spike is simply not carried
    /// over. With the cutoff and reset pushed out, [`Qif::isi`] therefore exceeds [`Theta::isi`]
    /// by exactly `t_ref` — 0.9% of the interval at 500 pA and 3.2% at 2 nA for the default 2 ms —
    /// which is asserted in `the_matching_theta_drops_the_refractory_period_and_nothing_else`.
    #[must_use]
    pub fn matching_theta(&self) -> Theta {
        Theta {
            tau: self.tau_m,
            i_ref: self.delta() / self.r_m,
            bias: -0.25,
            v_mid: self.v_mid(),
            v_scale: self.delta(),
            theta: 2.0 * self.canonical_y().atan(),
            substeps: 4,
        }
    }
}

impl Neuron for Qif {
    // FALSE — and NOT because the flow is inexact. `canonical_flow` is the exact solution of the
    // constant-input equation and it composes: advancing by `h1` then `h2` lands where advancing by
    // `h1 + h2` lands, to 1e-14 V across a ten-constant jump, which is the same standard `Lif`'s
    // composed exponentials meet.
    //
    // It is false because `sim::Sim::catch_up` jumps a gap with one zero-input `step` and THROWS
    // AWAY the bool it returns, on the documented ground that a quiet interval cannot produce a
    // spike. That ground is a property of LEAKY models, whose potential decays monotonically to a
    // rest below threshold. This model is BISTABLE at zero input — the fixed point at `v_c` is
    // unstable — so a synapse that lands the membrane above `v_c` produces a spike with no further
    // input and the event-driven run silently loses it. Declaring true here would make `Sim::new`
    // accept the model and then drop half the network's spikes while reporting success.
    const EXACT_OVER_GAPS: bool = false;

    fn step(&mut self, dt: f64, i: f64) -> bool {
        if self.refractory > 0.0 {
            self.refractory -= dt;
            self.v = self.v_reset;
            return false;
        }
        if self.v >= self.v_peak {
            // Reachable by `bump`: a synaptic kick can jump the cutoff outright, which the circle
            // formulation cannot do. See `Theta::bump`.
            self.v = self.v_reset;
            self.refractory = self.t_ref;
            return true;
        }
        let eta = self.eta(i);
        let y0 = self.canonical_y();
        match canonical_flow(y0, eta, dt / self.tau_m) {
            Flow::Finite(y1) => {
                self.set_canonical_y(y1);
                if self.v >= self.v_peak {
                    self.v = self.v_reset;
                    self.refractory = self.t_ref;
                    true
                } else {
                    false
                }
            }
            Flow::Diverged { .. } => {
                // The time left in the tick after the divergence is DISCARDED, and a tick coarse
                // enough to contain two divergences reports one spike. Same compromise as
                // `Theta::step`, stated here too because the exact flow could price the remainder
                // and deliberately does not: the refractory period starts at the end of the tick
                // either way, and a model whose spike times depended on where in the tick the
                // divergence fell would not be reproducible across a change of `dt`.
                self.v = self.v_reset;
                self.refractory = self.t_ref;
                true
            }
        }
    }

    fn bump(&mut self, dv: f64) {
        if self.refractory > 0.0 {
            return;
        }
        self.v += dv;
    }

    fn refractory_left(&self) -> f64 {
        self.refractory.max(0.0)
    }

    fn potential(&self) -> f64 {
        self.v
    }

    fn reset(&mut self) {
        self.v = self.v_rest;
        self.refractory = 0.0;
    }
}

// ---------------------------------------------------------------------------------------------
// Theta neuron
// ---------------------------------------------------------------------------------------------

/// The theta neuron: the quadratic model on a circle, where nothing ever diverges.
///
/// Ermentrout & Kopell, *SIAM J. Appl. Math.* 46:233–253, 1986 (and Ermentrout, *Neural Comput.*
/// 8:979–1001, 1996 for the neuroscience reading):
///
/// ```text
/// dθ/ds = (1 - cos θ) + (1 + cos θ) η
/// ```
///
/// with `θ` on the circle and `s` dimensionless time. Substituting `y = tan(θ/2)` and using
/// `1 - cos θ = 2sin²(θ/2)`, `1 + cos θ = 2cos²(θ/2)` gives `dy/ds = y² + η` — **the equation
/// [`Qif`] integrates**, term for term. The spike is the passage of `θ` through `π`, which is `y`
/// passing through infinity and returning from `-∞`; the reset is not code, it is the topology.
///
/// That is the whole reason this model exists. There is no cutoff to choose, no reset potential to
/// choose, and no divergence to catch, so a network of theta neurons has no parameters hiding in
/// its spike-handling. The price is that `θ` is not a voltage, and the crate's SI contract has to
/// be met by declaring a `v_mid` and a `v_scale` that map the circle back onto [`Qif`]'s volts.
///
/// # Exact answers available here
///
/// For constant `η > 0` the phase advances linearly, so the interval is `τ·π/√η` exactly
/// ([`Theta::isi`]) and the spike count over any window is a floor of a linear function
/// ([`Theta::spikes_by`]). Those are the closed forms this module's tests hold the integrator to.
///
/// # Why the integrator is Runge-Kutta and not the exact flow
///
/// Deliberately. [`Qif`] steps by the exact flow; if `Theta` did too, the test that the two are the
/// same system would be comparing one formula against itself. `Theta` integrates the `θ` equation
/// by fourth-order Runge-Kutta, sharing no code with `Qif`, so the correspondence test has
/// something to say. [`Theta::exact_theta_after`] is available when the closed form is what you
/// want, and the tests check the stepper against it too.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Theta {
    /// Time scale, seconds per unit of the model's dimensionless time `s`.
    pub tau: f64,
    /// Current, amperes, that maps to one unit of the model's dimensionless input.
    pub i_ref: f64,
    /// Dimensionless baseline added to `i / i_ref` to give `η`. Negative makes the cell excitable
    /// (silent until driven); positive makes it a spontaneous oscillator.
    pub bias: f64,
    /// Potential, volts, that the circle's `θ = 0` corresponds to. Exists only so
    /// [`Neuron::potential`] can honour the crate's volts contract.
    pub v_mid: f64,
    /// Volts per unit of `tan(θ/2)`. With [`Qif::matching_theta`] this is the `Qif`'s `Δ`.
    pub v_scale: f64,
    /// Phase, radians, held in `(-π, π]`. `θ = π` is the spike.
    pub theta: f64,
    /// Runge-Kutta substeps per call to [`Neuron::step`]. Four is enough for a `dt` of 0.1 ms at
    /// the default `tau`; raise it if the phase moves far in one tick.
    pub substeps: u32,
}

impl Default for Theta {
    /// The circle twin of [`Qif::default`], to the digit: `tau` 20 ms, `bias` −1/4,
    /// `i_ref = 15 mV / 10 MΩ = 1.5 nA`, `v_mid` −57.5 mV, `v_scale` 15 mV. Rheobase is 375 pA,
    /// the same number [`Qif::default`] reports.
    fn default() -> Self {
        Qif::default().matching_theta()
    }
}

impl Theta {
    /// Build one, refusing parameters that have no model behind them.
    ///
    /// Starts at `theta` folded into `(-π, π]`.
    ///
    /// # Errors
    ///
    /// [`ModelError::NotFinite`] or [`ModelError::NotPositive`] naming the field.
    pub fn new(
        tau: f64,
        i_ref: f64,
        bias: f64,
        v_mid: f64,
        v_scale: f64,
        theta: f64,
    ) -> Result<Self, ModelError> {
        positive("tau", tau)?;
        positive("i_ref", i_ref)?;
        positive("v_scale", v_scale)?;
        finite("bias", bias)?;
        finite("v_mid", v_mid)?;
        finite("theta", theta)?;
        Ok(Self { tau, i_ref, bias, v_mid, v_scale, theta: wrap_pi(theta), substeps: 4 })
    }

    /// The canonical input `η = bias + i / i_ref`, dimensionless. Positive means firing.
    #[must_use]
    pub fn eta(&self, i: f64) -> f64 {
        self.bias + i / self.i_ref
    }

    /// Rheobase, amperes: the current at which `η` crosses zero, `-bias · i_ref`.
    #[must_use]
    pub fn rheobase(&self) -> f64 {
        -self.bias * self.i_ref
    }

    /// The corresponding [`Qif`] coordinate `y = tan(θ/2)`, dimensionless.
    ///
    /// Diverges as `θ -> π`. That is not a defect: `y = ±∞` is where the spike is, and the whole
    /// point of the circle coordinate is that `θ` stays finite there while `y` does not.
    #[must_use]
    pub fn canonical_y(&self) -> f64 {
        (0.5 * self.theta).tan()
    }

    /// `dθ/ds`: radians of phase per unit of the model's dimensionless time `s = t/tau`.
    ///
    /// `eta` is the **canonical input** from [`Theta::eta`], dimensionless — not amperes. It is
    /// taken already reduced because [`Neuron::step`] reduces it once and then integrates many
    /// substeps against it.
    #[must_use]
    pub fn drift(&self, theta: f64, eta: f64) -> f64 {
        let c = theta.cos();
        (1.0 - c) + (1.0 + c) * eta
    }

    /// The closed-form phase `t` seconds later under constant current `i`, folded into `(-π, π]`.
    ///
    /// For `η > 0` the flow is a rigid rotation of the phase variable `φ = atan(tan(θ/2)/√η)`:
    /// `φ(s) = φ₀ + √η·s`, and `θ = 2·atan2(√η·sin φ, cos φ)` recovers the angle with the right
    /// quadrant, so a trajectory that has passed `θ = π` comes back on the other side by itself.
    /// For `η <= 0` the Riccati flow is used, in whichever of its two forms is accurate at the
    /// starting phase — see [`canonical_flow`], which makes the same choice for the same reason.
    /// Above the midpoint that is `u = u₀·e^{2as}`, which continues **through** the spike by
    /// itself: `u` passes 1, `y = a(1 + u)/(1 - u)` changes sign, and the phase comes back from
    /// `-π`. The number of spikes passed over is discarded here; [`Theta::spikes_by`] counts them.
    ///
    /// # Errors
    ///
    /// [`ModelError::NotFinite`] for a non-finite `i` or `t`.
    pub fn exact_theta_after(&self, i: f64, t: f64) -> Result<f64, ModelError> {
        finite("i", i)?;
        finite("t", t)?;
        let eta = self.eta(i);
        finite("eta", eta)?;
        let s = t / self.tau;
        let y0 = self.canonical_y();
        if eta > 0.0 {
            let b = eta.sqrt();
            let phi = (y0 / b).atan() + b * s;
            return Ok(wrap_pi(2.0 * (b * phi.sin()).atan2(phi.cos())));
        }
        if eta == 0.0 {
            let den = 1.0 - y0 * s;
            // `den < 0` means the trajectory passed through infinity; the ratio comes back
            // negative, which is the correct post-spike `y`, so no branch is needed.
            return Ok(wrap_pi(2.0 * (y0 / den).atan()));
        }
        let a = (-eta).sqrt();
        if y0 > 0.0 {
            let u0 = (y0 - a) / (y0 + a);
            if u0 == 0.0 {
                // On the unstable fixed point, where the phase stays put and `0 · e^{2as}` would
                // be `NaN` for a long enough interval.
                return Ok(wrap_pi(2.0 * a.atan()));
            }
            let u = u0 * (2.0 * a * s).exp();
            // `u` infinite is `-a` from either side: `u -> -∞` is deep inside the stable point,
            // `u -> +∞` is long past the spike and on its way back down to the same place.
            let y = if u.is_finite() { a * (1.0 + u) / (1.0 - u) } else { -a };
            return Ok(wrap_pi(2.0 * y.atan()));
        }
        let tt = (a * s).tanh();
        let den = a - y0 * tt;
        Ok(wrap_pi(2.0 * (a * (y0 - a * tt) / den).atan()))
    }

    /// Exact number of spikes in `t` seconds under constant current `i`.
    ///
    /// Above rheobase the phase advances at a constant `√η` per unit `s` and a spike is every
    /// crossing of `φ = π/2 + kπ`, so the count is `floor((φ₀ + √η·s + π/2)/π)`. That is an exact
    /// whole number and not an estimate. At or below rheobase the cell can spike at most once, on
    /// its way down from a starting phase above the unstable fixed point.
    ///
    /// **Range.** The count is a `f64` floor cast to `u64`, so a window long enough to hold more
    /// than `u64::MAX` spikes **saturates** there rather than wrapping, and the count is exact
    /// only while it stays below `2^53`. Both bounds are astronomical — `2^53` spikes at 1 kHz is
    /// 285,000 years — and neither is reachable from a simulation that also stepped them.
    ///
    /// # Errors
    ///
    /// [`ModelError::NotFinite`] for a non-finite `i` or `t`, or [`ModelError::NotPositive`] for a
    /// negative `t`.
    pub fn spikes_by(&self, i: f64, t: f64) -> Result<u64, ModelError> {
        finite("i", i)?;
        finite("t", t)?;
        if t < 0.0 {
            return Err(ModelError::NotPositive { what: "t", value: t });
        }
        let eta = self.eta(i);
        let s = t / self.tau;
        let y0 = self.canonical_y();
        if eta > 0.0 {
            let b = eta.sqrt();
            let phi0 = (y0 / b).atan();
            let n = ((phi0 + b * s + 0.5 * PI) / PI).floor();
            return Ok(if n <= 0.0 { 0 } else { n as u64 });
        }
        let one = match canonical_flow(y0, eta, s) {
            Flow::Diverged { .. } => 1,
            Flow::Finite(_) => 0,
        };
        Ok(one)
    }

    /// Inter-spike interval under constant current `i`, seconds: `τ·π/√η`, exactly.
    ///
    /// This is the Type I firing rate in its purest form — `f = √η/(π τ)` — and it is what
    /// [`Qif::isi`] collapses to when the cutoff and reset are taken to `±∞`.
    ///
    /// # Errors
    ///
    /// [`ModelError::NotFinite`] for a non-finite `i`, or [`ModelError::NoFiring`] at or below
    /// rheobase, where the phase gets stuck at the stable fixed point.
    pub fn isi(&self, i: f64) -> Result<f64, ModelError> {
        finite("i", i)?;
        let eta = self.eta(i);
        if eta <= 0.0 {
            return Err(ModelError::NoFiring { i, rheobase: self.rheobase() });
        }
        Ok(self.tau * PI / eta.sqrt())
    }

    /// Steady-state firing rate under constant current `i`, hertz.
    ///
    /// # Errors
    ///
    /// As [`Theta::isi`].
    pub fn rate(&self, i: f64) -> Result<f64, ModelError> {
        self.isi(i).map(|t| 1.0 / t)
    }
}

impl Neuron for Theta {
    // FALSE, and here the reason really is the integrator. Fourth-order Runge-Kutta on a nonlinear
    // equation gives different answers for one step of `2h` and two steps of `h`, so jumping a
    // quiet gap would move the spike times — silently, and only on the ticks that happened to be
    // quiet. `Qif` is the same system with an exact flow and is false for an entirely different
    // reason; see its constant, and the module doc on what `EXACT_OVER_GAPS` does not say.
    const EXACT_OVER_GAPS: bool = false;

    fn step(&mut self, dt: f64, i: f64) -> bool {
        let eta = self.eta(i);
        let n = self.substeps.max(1);
        let h = dt / self.tau / f64::from(n);
        let mut fired = false;
        for _ in 0..n {
            let t0 = self.theta;
            let k1 = self.drift(t0, eta);
            let k2 = self.drift(t0 + 0.5 * h * k1, eta);
            let k3 = self.drift(t0 + 0.5 * h * k2, eta);
            let k4 = self.drift(t0 + h * k3, eta);
            self.theta = t0 + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;
            // The EXACT flow at `θ = ±π` is `+2`, strictly positive for every `η`, so the exact
            // trajectory can only ever leave the interval upward. Runge-Kutta is not the exact
            // flow: once `h·|η|` reaches about 2 its inner stages sample `cos θ` half a step away,
            // the combination can be large and negative, and the phase leaves DOWNWARD. Measured
            // on the default cell at the drives of `a_violent_drive_leaves_no_model_non_finite`
            // (1 ms ticks, ±1 mA): `θ = -28,486` after two steps. Nothing reports it — the state
            // stays finite and `potential()` keeps returning plausible volts — but the cell has to
            // climb 9,000 radians before it can spike again, so every later spike is lost.
            //
            // So the fold is `wrap_pi`, which folds BOTH ways and restores the invariant this type
            // documents, and the spike is the upward crossing only. A step coarse enough to carry
            // the phase round more than once still counts one spike, which is the same compromise
            // every clocked simulator makes; a step that threw it downward counts none, because
            // under the equation being integrated no downward crossing exists.
            if self.theta > PI {
                fired = true;
            }
            self.theta = wrap_pi(self.theta);
        }
        fired
    }

    /// A synaptic kick displaces the **corresponding [`Qif`]'s** potential, which on the circle is
    /// a nonlinear rotation: `θ <- 2·atan(tan(θ/2) + dv/v_scale)`.
    ///
    /// A bump alone can never make a theta neuron spike, however large it is, because `tan` maps
    /// `(-π, π)` onto the whole real line and adding a finite number to a finite `y` leaves it
    /// finite. [`Qif`] behaves differently — a big enough kick jumps its finite `v_peak` outright —
    /// and the difference is exactly the approximation that a finite cutoff makes.
    fn bump(&mut self, dv: f64) {
        let y = self.canonical_y() + dv / self.v_scale;
        self.theta = 2.0 * y.atan();
    }

    /// `v_mid + v_scale·tan(θ/2)`, volts.
    ///
    /// Unbounded as `θ -> π`, and infinite at `θ = π` exactly. That is the honest answer: the
    /// corresponding [`Qif`] is mid-spike there. The stepper never leaves `θ` at exactly `π`
    /// because the flow is strictly positive through it, so only a hand-set phase can produce it.
    fn potential(&self) -> f64 {
        self.v_mid + self.v_scale * self.canonical_y()
    }

    fn reset(&mut self) {
        self.theta = 0.0;
    }
}

// ---------------------------------------------------------------------------------------------
// Exponential integrate-and-fire
// ---------------------------------------------------------------------------------------------

/// Exponential integrate-and-fire.
///
/// Fourcaud-Trocmé, Hansel, van Vreeswijk & Brunel, *J. Neurosci.* 23:11628–11640, 2003:
///
/// ```text
/// C dV/dt = -g_L (V - E_L) + g_L Δ_T exp((V - V_T)/Δ_T) + I
/// ```
///
/// The exponential term is the sodium activation curve, linearised in its exponent. `V_T` is where
/// it becomes comparable to the leak — the **soft** threshold — and `Δ_T` is how sharp the onset
/// is. Setting `Δ_T -> 0` recovers a [`Lif`] with a hard threshold at `V_T` exactly, which is the
/// check [`Eif::lif_limit`] exists for.
///
/// # Closed forms
///
/// - **Rheobase** `I_rheo = g_L(V_T - E_L - Δ_T)`, exactly. The drift's minimum over `V` is at
///   `V = V_T`, where it equals `I - I_rheo`; above rheobase the drift is positive everywhere and
///   the cell cannot rest. Note what the `-Δ_T` says: the exponential **lowers** the current needed
///   to fire, relative to a `Lif` thresholded at `V_T`, by `g_L Δ_T`.
/// - **Fixed points** through the Lambert `W` function, both branches — see [`Eif::fixed_points`].
///   The test plugs them back into [`Eif::drift`] and asks for zero.
/// - **Onset law** `f ≈ (1/(π C))·√(g_L(I - I_rheo)/(2 Δ_T))` just above rheobase, from expanding
///   the drift to second order about its minimum. [`Eif::saddle_node_rate`]; the test shows the
///   true rate converging onto it as the current approaches rheobase.
///
/// # What is *not* closed form
///
/// The interval. `∫ C dV / F(V)` has no elementary antiderivative when `F` carries an exponential,
/// and this implementation did not locate one in the literature. [`Eif::isi`] evaluates that exact
/// integral by adaptive Simpson quadrature and is named `isi` only because that is what it returns;
/// its doc, and this paragraph, are where the method is stated. It shares no code with the
/// time-stepper, which is what makes it a useful reference for it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Eif {
    /// Membrane capacitance, farads.
    pub c: f64,
    /// Leak conductance, siemens. `C/g_L` is the membrane time constant and `1/g_L` its resistance.
    pub g_l: f64,
    /// Leak reversal potential, volts. The resting potential to within
    /// `Δ_T·exp((E_L - V_T)/Δ_T)` — about 0.1 µV for the default parameters.
    pub e_l: f64,
    /// Soft threshold, volts: where the exponential current matches the leak current.
    pub v_t: f64,
    /// Onset sharpness, volts. Smaller is sharper; `Δ_T -> 0` is a hard threshold at `v_t`.
    /// Badel et al. (2008) measured it at order 1 mV in cortical pyramidal cells.
    pub delta_t: f64,
    /// Spike cutoff, volts. The true trajectory diverges in finite time; this is where following it
    /// stops. Its exact value barely matters because the upstroke above `v_t + 10·Δ_T` takes
    /// microseconds, which is the property that makes the choice safe.
    pub v_peak: f64,
    /// Post-spike potential, volts.
    pub v_reset: f64,
    /// Absolute refractory period, seconds.
    pub t_ref: f64,
    /// Current membrane potential, volts.
    pub v: f64,
    /// Seconds left in the refractory period.
    pub refractory: f64,
    /// Fourth-order Runge-Kutta substeps per call to [`Neuron::step`].
    ///
    /// The upstroke is stiff: at `v_t + 25·Δ_T` the drift is already 7e9 V/s for the default
    /// parameters, so the last substep before a spike overshoots `v_peak` by a lot. That costs
    /// nothing in accuracy — the time from `v_t + 10·Δ_T` to any cutoff is microseconds — but is
    /// why the cutoff is tested **inside** the substep loop rather than after it.
    pub substeps: u32,
}

impl Default for Eif {
    /// The parameter set the `AdEx` literature uses as its reference membrane: `C` 200 pF, `g_L`
    /// 10 nS (so `tau_m` = 20 ms), `E_L` −70 mV, `V_T` −50 mV, `Δ_T` 2 mV, cutoff 0 mV, reset
    /// −58 mV, no refractory period. Rheobase is then `10 nS · (20 mV - 2 mV) = 180 pA`.
    ///
    /// These are the round numbers of Brette & Gerstner (2005) and Naud et al. (2008) rather than
    /// the fit in Fourcaud-Trocmé et al. (2003), whose `Δ_T` came from a `Hodgkin`-`Huxley` model
    /// and is larger; this implementation did not verify that fit against the paper and does not
    /// quote a number for it.
    fn default() -> Self {
        Self {
            c: 200e-12,
            g_l: 10e-9,
            e_l: -70e-3,
            v_t: -50e-3,
            delta_t: 2e-3,
            v_peak: 0.0,
            v_reset: -58e-3,
            t_ref: 0.0,
            v: -70e-3,
            refractory: 0.0,
            substeps: 4,
        }
    }
}

impl Eif {
    /// Build one, refusing parameters that have no model behind them.
    ///
    /// The membrane starts at `e_l` and out of refractory.
    ///
    /// # Errors
    ///
    /// [`ModelError::NotFinite`] or [`ModelError::NotPositive`] naming the field, or
    /// [`ModelError::Disordered`] when `v_peak <= v_reset` or `v_peak <= v_t`.
    pub fn new(
        c: f64,
        g_l: f64,
        e_l: f64,
        v_t: f64,
        delta_t: f64,
        v_peak: f64,
        v_reset: f64,
        t_ref: f64,
    ) -> Result<Self, ModelError> {
        positive("c", c)?;
        positive("g_l", g_l)?;
        positive("delta_t", delta_t)?;
        finite("e_l", e_l)?;
        finite("v_t", v_t)?;
        finite("v_peak", v_peak)?;
        finite("v_reset", v_reset)?;
        finite("t_ref", t_ref)?;
        if t_ref < 0.0 {
            return Err(ModelError::NotPositive { what: "t_ref", value: t_ref });
        }
        if v_peak <= v_reset {
            return Err(ModelError::Disordered {
                what: "v_peak <= v_reset",
                lower: v_reset,
                upper: v_peak,
            });
        }
        if v_peak <= v_t {
            return Err(ModelError::Disordered { what: "v_peak <= v_t", lower: v_t, upper: v_peak });
        }
        Ok(Self {
            c,
            g_l,
            e_l,
            v_t,
            delta_t,
            v_peak,
            v_reset,
            t_ref,
            v: e_l,
            refractory: 0.0,
            substeps: 4,
        })
    }

    /// Membrane time constant `C/g_L`, seconds.
    #[must_use]
    pub fn tau_m(&self) -> f64 {
        self.c / self.g_l
    }

    /// Membrane resistance `1/g_L`, ohms.
    #[must_use]
    pub fn r_m(&self) -> f64 {
        1.0 / self.g_l
    }

    /// Total membrane current at potential `v` under input `i`, amperes.
    ///
    /// `-g_L(v - E_L) + g_L Δ_T exp((v - V_T)/Δ_T) + i`, with the exponent clamped — see
    /// [`EXP_ARG_LIMIT`] for the bound that clamp puts on the answer.
    #[must_use]
    pub fn current(&self, v: f64, i: f64) -> f64 {
        -self.g_l * (v - self.e_l)
            + self.g_l * self.delta_t * exp_clamped((v - self.v_t) / self.delta_t)
            + i
    }

    /// `dV/dt` in volts per second at potential `v` under input `i`.
    #[must_use]
    pub fn drift(&self, v: f64, i: f64) -> f64 {
        self.current(v, i) / self.c
    }

    /// Rheobase, amperes: `g_L (V_T - E_L - Δ_T)`, exactly.
    #[must_use]
    pub fn rheobase(&self) -> f64 {
        self.g_l * (self.v_t - self.e_l - self.delta_t)
    }

    /// The two fixed points under constant current `i`, volts, as `(stable, unstable)`.
    ///
    /// Closed form. Writing `x = (V - V_T)/Δ_T`, the equilibrium condition becomes `e^x = x + k`
    /// with `k = (V_T - E_L - I/g_L)/Δ_T`, whose solutions are `x = -W(-e^{-k}) - k` on the two
    /// real branches of the Lambert `W` function. Real roots exist exactly when `k >= 1`, which is
    /// exactly `i <= rheobase` — the two statements agree, which is one of the things the test
    /// checks.
    ///
    /// The Lambert form is the derivation, not the evaluation: `e^{-k}` underflows to zero for
    /// `k > 745`, and the upper root is the difference of two numbers of size `k`. Both are
    /// reachable here — `k` is `(V_T - E_L - I/g_L)/Δ_T`, so a sharp onset or a hyperpolarising
    /// current sends it up without limit — so [`exp_offset_roots`] solves `e^x = x + k` for `x`
    /// directly instead. See its note.
    ///
    /// `None` above rheobase, where the membrane escapes from anywhere, and for a non-finite `i`.
    #[must_use]
    pub fn fixed_points(&self, i: f64) -> Option<(f64, f64)> {
        if !i.is_finite() {
            return None;
        }
        let k = (self.v_t - self.e_l - i / self.g_l) / self.delta_t;
        let (lower, upper) = exp_offset_roots(k)?;
        Some((self.v_t + self.delta_t * lower, self.v_t + self.delta_t * upper))
    }

    /// Inter-spike interval under constant current `i`, seconds — **by quadrature, not in closed
    /// form**.
    ///
    /// The value returned is `t_ref + ∫ C dV / F(V)` from `v_reset` to `v_peak`, which is the exact
    /// interval of the deterministic model; the integral is evaluated by adaptive Simpson,
    /// pre-split at `v_t` (where the integrand peaks) and truncated at `v_t + 50·Δ_T`, beyond which
    /// the integrand is below `1e-20` s per volt. There is no elementary antiderivative and this
    /// implementation did not locate one.
    ///
    /// **Below rheobase this model is bistable too**, exactly as [`Qif`] is: the two fixed points
    /// still exist, and a `v_reset` **above** the unstable one leaves the membrane in the escaping
    /// region, so it fires forever at a current that cannot make it fire from rest.
    /// [`FiringPattern::RegularBursting`] ships such a membrane — reset −46 mV against a `V_T` of
    /// −50 mV — and at half its rheobase it fires 45 times in 200 ms. The interval is returned in
    /// that regime rather than refused, and only a reset at or below the unstable fixed point is
    /// [`ModelError::NoFiring`]. As the reset approaches that point from above the integral
    /// diverges logarithmically, which is the true answer growing without bound and not a defect;
    /// the quadrature degrades there for the same reason it does near rheobase.
    ///
    /// **Where it degrades.** Within about one part in `1e6` of rheobase the integrand is a spike
    /// narrower than [`QUADRATURE_BUDGET`] panels can resolve, and the value returned is then an
    /// under-resolved estimate rather than the exact integral. Measured against an independent
    /// reference in `the_interval_quadrature_holds_to_the_band_its_doc_claims`: the relative error
    /// against the reference integral is 3e-11 at one part in `1.8e5` of rheobase, 3e-5 at one
    /// part in `1.8e6`, and 7e-4 at one part in `1.8e7` — so the band is where the doc has always
    /// put it, and outside it this is the exact integral rather than an estimate.
    ///
    /// [`Eif::saddle_node_rate`] is the right tool inside the band and is exact in the limit the
    /// quadrature is failing in, which is a convenient division of labour rather than a
    /// coincidence: both are consequences of the bottleneck dominating.
    ///
    /// # Errors
    ///
    /// [`ModelError::NotFinite`] for a non-finite `i`, or [`ModelError::NoFiring`] at or below
    /// rheobase **with** a reset at or below the unstable fixed point, where the membrane settles
    /// on the stable fixed point instead.
    pub fn isi(&self, i: f64) -> Result<f64, ModelError> {
        finite("i", i)?;
        let rheobase = self.rheobase();
        if i <= rheobase {
            let escapes = match self.fixed_points(i) {
                Some((_, unstable)) => self.v_reset > unstable,
                None => false,
            };
            if !escapes {
                return Err(ModelError::NoFiring { i, rheobase });
            }
        }
        let f = |v: f64| self.c / self.current(v, i);
        // `.max(self.v_reset)`: a reset above the truncation point leaves an EMPTY interval, not
        // an inverted one. `f64::clamp` PANICS when its bounds cross, and `Eif::new` accepts a
        // `v_reset` above `v_t + 50·Δ_T` — a 0.1 mV onset with a reset 6 mV above `V_T` is enough,
        // and the taxonomy already ships resets above `V_T`. The interval above the truncation is
        // worth less than `1e-20` s per volt (see [`EXP_ARG_LIMIT`]), so the honest answer for
        // such a membrane is `t_ref` and a remainder no `f64` interval can carry.
        let top = self.v_peak.min(self.v_t + EXP_ARG_LIMIT * self.delta_t).max(self.v_reset);
        let split = self.v_t.clamp(self.v_reset, top);
        let t = integrate(&f, self.v_reset, split, 1e-12) + integrate(&f, split, top, 1e-12);
        Ok(t + self.t_ref)
    }

    /// Steady-state firing rate under constant current `i`, hertz, by the same quadrature.
    ///
    /// # Errors
    ///
    /// As [`Eif::isi`].
    pub fn rate(&self, i: f64) -> Result<f64, ModelError> {
        self.isi(i).map(|t| 1.0 / t)
    }

    /// The leading square-root law just above rheobase, hertz.
    ///
    /// `f ≈ (1/(π C))·√(g_L (I - I_rheo)/(2 Δ_T))`, from expanding the drift about its minimum at
    /// `V_T`, where `F'' = g_L/Δ_T`, and integrating the resulting quadratic bottleneck over the
    /// whole line. It is an **asymptote**, exact only in the limit `I -> I_rheo⁺`, and it ignores
    /// `t_ref` and the finite reset and cutoff entirely — all three of which matter at any current
    /// you would actually use. The test shows [`Eif::rate`] converging onto it, which is the only
    /// claim being made.
    ///
    /// # Errors
    ///
    /// [`ModelError::NotFinite`] for a non-finite `i`, or [`ModelError::NoFiring`] at or below
    /// rheobase.
    pub fn saddle_node_rate(&self, i: f64) -> Result<f64, ModelError> {
        finite("i", i)?;
        let rheobase = self.rheobase();
        if i <= rheobase {
            return Err(ModelError::NoFiring { i, rheobase });
        }
        Ok((self.g_l * (i - rheobase) / (2.0 * self.delta_t)).sqrt() / (PI * self.c))
    }

    /// The [`Lif`] this model becomes as `Δ_T -> 0`: same membrane, hard threshold at `v_t`.
    ///
    /// Not an approximation of the `Eif` at its actual `Δ_T` — the **limit**. Comparing
    /// [`Eif::rate`] against this `Lif`'s closed-form rate over a sequence of shrinking `Δ_T` is
    /// the module's check that the exponential term is doing what the paper says it does, and the
    /// residual shrinks like `Δ_T·ln(1/Δ_T)`, which is the time the upstroke still takes.
    #[must_use]
    pub fn lif_limit(&self) -> Lif {
        Lif {
            tau_m: self.tau_m(),
            v_rest: self.e_l,
            v_th: self.v_t,
            v_reset: self.v_reset,
            r_m: self.r_m(),
            t_ref: self.t_ref,
            v: self.v,
            refractory: self.refractory,
        }
    }
}

impl Neuron for Eif {
    // FALSE. The exponential term makes the equation nonlinear and it is integrated by fourth-order
    // Runge-Kutta, so one step of `k·dt` and `k` steps of `dt` disagree at O(dt⁴) per step. Jumping
    // a quiet gap would move spike times by an amount that depends on which ticks were quiet, which
    // is the kind of error that produces a plausible raster and an unreproducible result.
    const EXACT_OVER_GAPS: bool = false;

    fn step(&mut self, dt: f64, i: f64) -> bool {
        if self.refractory > 0.0 {
            self.refractory -= dt;
            self.v = self.v_reset;
            return false;
        }
        if self.v >= self.v_peak {
            self.v = self.v_reset;
            self.refractory = self.t_ref;
            return true;
        }
        let n = self.substeps.max(1);
        let h = dt / f64::from(n);
        for _ in 0..n {
            let v0 = self.v;
            let k1 = self.drift(v0, i);
            let k2 = self.drift(v0 + 0.5 * h * k1, i);
            let k3 = self.drift(v0 + 0.5 * h * k2, i);
            let k4 = self.drift(v0 + h * k3, i);
            self.v = v0 + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;
            // Inside the loop, not after it: the drift above the cutoff is astronomically large and
            // a second substep from there would leave a number no reader could interpret.
            if self.v >= self.v_peak || !self.v.is_finite() {
                self.v = self.v_reset;
                self.refractory = self.t_ref;
                return true;
            }
        }
        false
    }

    fn bump(&mut self, dv: f64) {
        if self.refractory > 0.0 {
            return;
        }
        self.v += dv;
    }

    fn refractory_left(&self) -> f64 {
        self.refractory.max(0.0)
    }

    fn potential(&self) -> f64 {
        self.v
    }

    fn reset(&mut self) {
        self.v = self.e_l;
        self.refractory = 0.0;
    }
}

// ---------------------------------------------------------------------------------------------
// Adaptive exponential integrate-and-fire
// ---------------------------------------------------------------------------------------------

/// Adaptive exponential integrate-and-fire — [`Eif`] plus one adaptation current.
///
/// Brette & Gerstner, *J. Neurophysiol.* 94:3637–3642, 2005:
///
/// ```text
/// C dV/dt  = -g_L (V - E_L) + g_L Δ_T exp((V - V_T)/Δ_T) - w + I
/// τ_w dw/dt = a (V - E_L) - w
/// on spike:  V <- V_reset,  w <- w + b
/// ```
///
/// The adaptation current `w` enters exactly as a subtraction from the input, which is why this
/// type **contains** an [`Eif`] rather than copying its fields: `AdEx`'s voltage equation is
/// `eif.drift(V, I - w)`, term for term, and the composition is the proof.
///
/// Two knobs, two mechanisms. `a` is **sub-threshold** adaptation — a conductance that tracks the
/// membrane whether or not it spikes, which is what produces resonance, rebound and the transient
/// cell. `b` is **spike-triggered** adaptation — a fixed kick per spike, which is what produces
/// spike-frequency adaptation and bursting. Almost the whole published taxonomy is the
/// `(a, b, τ_w, V_reset)` quadrant diagram, and [`FiringPattern`] is its named corners.
///
/// # Closed form available here
///
/// Below `V_T` the exponential term is small — 9e-16 A at rest for the default parameters, about a
/// millionth of the smallest stimulus in the taxonomy — and the remaining `(V, w)` system is then
/// **linear**, with an exact matrix-exponential solution. [`AdEx::linear_subthreshold`] is that
/// solution, and the module's strongest check on the Runge-Kutta stepper is that the stepped
/// trajectory lands on it. That check validates the integrator itself, with no firing pattern in
/// the loop to hide behind.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdEx {
    /// The membrane, which behaves exactly as [`Eif`] does under the effective input `i - w`.
    pub eif: Eif,
    /// Sub-threshold adaptation conductance, siemens. May be **negative**, which turns `w` into a
    /// positive feedback and is what the published irregular pattern uses.
    pub a: f64,
    /// Adaptation time constant, seconds.
    pub tau_w: f64,
    /// Spike-triggered adaptation increment, amperes, added to `w` on every spike.
    pub b: f64,
    /// Adaptation current, amperes. Subtracted from the input, so positive `w` is hyperpolarising.
    pub w: f64,
}

impl Default for AdEx {
    /// [`Eif::default`] with the tonic parameters of [`FiringPattern::Tonic`]: `a` 2 nS, `τ_w`
    /// 30 ms, `b` 0. With `b = 0` there is no spike-triggered adaptation at all, so this default
    /// fires regularly and is the baseline the other patterns are read against.
    fn default() -> Self {
        FiringPattern::Tonic.model()
    }
}

impl AdEx {
    /// Build one from a membrane and an adaptation.
    ///
    /// # Errors
    ///
    /// [`ModelError::NotFinite`] naming `a` or `b`, or [`ModelError::NotPositive`] for `tau_w`.
    pub fn new(eif: Eif, a: f64, tau_w: f64, b: f64) -> Result<Self, ModelError> {
        finite("a", a)?;
        finite("b", b)?;
        positive("tau_w", tau_w)?;
        Ok(Self { eif, a, tau_w, b, w: 0.0 })
    }

    /// `dw/dt` in amperes per second at potential `v` and adaptation `w`.
    #[must_use]
    pub fn w_drift(&self, v: f64, w: f64) -> f64 {
        (self.a * (v - self.eif.e_l) - w) / self.tau_w
    }

    /// The exact sub-threshold solution `t` seconds from the current state under constant `i`,
    /// returned as `(V, w)`.
    ///
    /// Drops the exponential term — legitimate only while `V` stays well below `V_T` — and solves
    /// the remaining linear system in closed form. The steady state is
    /// `V* = E_L + I/(g_L + a)`, `w* = a I/(g_L + a)`, and the deviation evolves by the matrix
    /// exponential of
    ///
    /// ```text
    /// M = [ -g_L/C   -1/C   ]
    ///     [  a/τ_w   -1/τ_w ]
    /// ```
    ///
    /// evaluated as `e^{αt}[cosh(µt)·I + (sinh(µt)/µ)(M - αI)]` with `α = tr M/2` and
    /// `µ = √(α² - det M)`, switching to `cos`/`sin` when `µ` is imaginary — which is the
    /// resonant regime, and is what `a > 0` with a fast `τ_w` buys.
    ///
    /// # Errors
    ///
    /// [`ModelError::NotFinite`] for a non-finite `i` or `t`, or [`ModelError::Degenerate`] when
    /// `g_L + a` is zero and the steady state does not exist.
    pub fn linear_subthreshold(&self, i: f64, t: f64) -> Result<(f64, f64), ModelError> {
        finite("i", i)?;
        finite("t", t)?;
        let (c, g_l, e_l) = (self.eif.c, self.eif.g_l, self.eif.e_l);
        let sum = g_l + self.a;
        if sum == 0.0 {
            return Err(ModelError::Degenerate { what: "g_l + a" });
        }
        let x_star = i / sum;
        let w_star = self.a * i / sum;
        let m = [[-g_l / c, -1.0 / c], [self.a / self.tau_w, -1.0 / self.tau_w]];
        let alpha = 0.5 * (m[0][0] + m[1][1]);
        let det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
        let disc = alpha * alpha - det;
        let (ch, sh) = if disc > 0.0 {
            let mu = disc.sqrt();
            ((mu * t).cosh(), (mu * t).sinh() / mu)
        } else if disc < 0.0 {
            let beta = (-disc).sqrt();
            ((beta * t).cos(), (beta * t).sin() / beta)
        } else {
            (1.0, t)
        };
        let scale = (alpha * t).exp();
        let n = [[m[0][0] - alpha, m[0][1]], [m[1][0], m[1][1] - alpha]];
        let e = [
            [scale * (ch + sh * n[0][0]), scale * sh * n[0][1]],
            [scale * sh * n[1][0], scale * (ch + sh * n[1][1])],
        ];
        let d0 = (self.eif.v - e_l) - x_star;
        let d1 = self.w - w_star;
        Ok((e_l + x_star + e[0][0] * d0 + e[0][1] * d1, w_star + e[1][0] * d0 + e[1][1] * d1))
    }
}

impl Neuron for AdEx {
    // FALSE, for the same reason as `Eif`: the voltage equation is the same nonlinear one, and the
    // pair is integrated jointly by fourth-order Runge-Kutta.
    const EXACT_OVER_GAPS: bool = false;

    fn step(&mut self, dt: f64, i: f64) -> bool {
        if self.eif.refractory > 0.0 {
            self.eif.refractory -= dt;
            self.eif.v = self.eif.v_reset;
            // `w` keeps evolving through the refractory period, exactly, because adaptation is a
            // slow variable of the cell and not of the spike. Freezing it would make the effective
            // adaptation time constant depend on the firing rate, which the model does not say.
            let w_inf = self.a * (self.eif.v_reset - self.eif.e_l);
            self.w = w_inf + (self.w - w_inf) * (-dt / self.tau_w).exp();
            return false;
        }
        if self.eif.v >= self.eif.v_peak {
            self.eif.v = self.eif.v_reset;
            self.w += self.b;
            self.eif.refractory = self.eif.t_ref;
            return true;
        }
        let n = self.eif.substeps.max(1);
        let h = dt / f64::from(n);
        for _ in 0..n {
            let (v0, w0) = (self.eif.v, self.w);
            let (kv1, kw1) = (self.eif.drift(v0, i - w0), self.w_drift(v0, w0));
            let (va, wa) = (v0 + 0.5 * h * kv1, w0 + 0.5 * h * kw1);
            let (kv2, kw2) = (self.eif.drift(va, i - wa), self.w_drift(va, wa));
            let (vb, wb) = (v0 + 0.5 * h * kv2, w0 + 0.5 * h * kw2);
            let (kv3, kw3) = (self.eif.drift(vb, i - wb), self.w_drift(vb, wb));
            let (vc, wc) = (v0 + h * kv3, w0 + h * kw3);
            let (kv4, kw4) = (self.eif.drift(vc, i - wc), self.w_drift(vc, wc));
            self.eif.v = v0 + h * (kv1 + 2.0 * kv2 + 2.0 * kv3 + kv4) / 6.0;
            self.w = w0 + h * (kw1 + 2.0 * kw2 + 2.0 * kw3 + kw4) / 6.0;
            if self.eif.v >= self.eif.v_peak || !self.eif.v.is_finite() || !self.w.is_finite() {
                self.eif.v = self.eif.v_reset;
                // `w0`, not the Runge-Kutta result: the stage that crossed the cutoff evaluated the
                // adaptation equation at a potential the model never reaches, and `a·(V - E_L)`
                // with `V` at 1e20 volts would put `w` somewhere no reset could recover from.
                self.w = w0 + self.b;
                self.eif.refractory = self.eif.t_ref;
                return true;
            }
        }
        false
    }

    fn bump(&mut self, dv: f64) {
        self.eif.bump(dv);
    }

    fn refractory_left(&self) -> f64 {
        self.eif.refractory_left()
    }

    fn potential(&self) -> f64 {
        self.eif.v
    }

    fn reset(&mut self) {
        self.eif.reset();
        self.w = 0.0;
    }
}

/// The published firing-pattern taxonomy of the adaptive exponential model.
///
/// Naud, Marcille, Clopath & Gerstner, *Firing patterns in the adaptive exponential
/// integrate-and-fire model*, *Biol. Cybern.* 99:335–347, 2008, Table 1. The paper's point is that
/// a two-variable model with one exponential covers most of the cortical repertoire, and that which
/// pattern you get is decided by where `(a, b, τ_w, V_reset)` sits relative to the membrane — not
/// by adding mechanisms.
///
/// **Honesty about these constants.** They are transcribed from that table and this implementation
/// did **not** verify them digit by digit against the printed paper; a reader with it in hand
/// should check them. What *is* verified, by a test per variant, is that each parameter set
/// produces the pattern it is named after, asserted numerically: lengthening intervals for
/// [`FiringPattern::Adapting`], a bimodal interval distribution for
/// [`FiringPattern::RegularBursting`], silence in the second half of the run for
/// [`FiringPattern::Transient`]. Where a value here differs from the table, the test is what
/// caught it and the pattern is what was preserved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FiringPattern {
    /// Regular firing at a constant rate: no spike-triggered adaptation at all (`b = 0`).
    Tonic,
    /// Spike-frequency adaptation: intervals lengthen monotonically toward a slower steady rate.
    /// Long `τ_w` and a large `b`.
    Adapting,
    /// A fast doublet or triplet at stimulus onset, then a regular slower train. The reset sits at
    /// `V_T`, so the first intervals are set by the exponential rather than by the leak.
    InitialBurst,
    /// Bursts repeating indefinitely: a bimodal interval distribution, short within a burst and
    /// long between them. The reset sits **above** `V_T`.
    RegularBursting,
    /// A few spikes and then silence, although the current stays on: strong sub-threshold
    /// adaptation (`a` large and positive) wins the race against the drive. The cell is not
    /// exhausted — it has acquired a stable fixed point, at `E_L + I/(g_L + a)`, and settled on it.
    ///
    /// **This variant's `b` was changed from what was transcribed, and `b` alone.** With the
    /// transcribed pair (`τ_w` 90 ms, `b` 100 pA) the cell fires once at 180 and 200 pA and twice
    /// at 220, 250 and 270 pA before falling silent, and at 280 pA the window closes — it then
    /// fires for the whole second. One or two spikes is a degenerate corner of the pattern rather
    /// than the published figure's short train, and no current produces a train: the window is
    /// shut before the count reaches three. Dropping `b` to 30 pA, with `τ_w` left at the
    /// transcribed 90 ms, gives 3, 4, 5 and 6 spikes at 200, 220, 250 and 270 pA, each train over
    /// within 31 ms, and the window still closes at 280 pA. Both halves are measured in
    /// `the_transient_pattern_is_transient_across_its_window_and_not_above_it`, which fails if
    /// either stops being true.
    ///
    /// The transcription is the suspect party here, not the model — a reader with the paper should
    /// check `b` and this note. What is claimed is that ONE number had to move, and that the test
    /// is what says so.
    Transient,
    /// Sustained irregular firing from **negative** `a`, which makes the adaptation a positive
    /// feedback. The published pattern is chaotic; this implementation asserts sustained interval
    /// variability late in the run, which is a necessary condition for chaos and not a proof of it.
    /// This implementation did not compute a Lyapunov exponent.
    Irregular,
}

impl FiringPattern {
    /// Every variant, in the order the module documents them, for a test or a figure that has to
    /// visit all of them.
    pub const ALL: [FiringPattern; 6] = [
        FiringPattern::Tonic,
        FiringPattern::Adapting,
        FiringPattern::InitialBurst,
        FiringPattern::RegularBursting,
        FiringPattern::Transient,
        FiringPattern::Irregular,
    ];

    /// A short lower-case label, for a legend or a log line.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Tonic => "tonic",
            Self::Adapting => "adapting",
            Self::InitialBurst => "initial burst",
            Self::RegularBursting => "regular bursting",
            Self::Transient => "transient",
            Self::Irregular => "irregular",
        }
    }

    /// The stimulus current, amperes, that this pattern is defined at.
    ///
    /// The pattern is a property of the parameters **and the drive** together; the same cell at
    /// another current does something else. Handing the model back without the current it belongs
    /// to is the commonest way to reproduce this figure wrongly.
    #[must_use]
    pub fn drive(self) -> f64 {
        match self {
            Self::Tonic | Self::Adapting => 500e-12,
            Self::InitialBurst => 400e-12,
            Self::RegularBursting => 210e-12,
            Self::Transient => 220e-12,
            Self::Irregular => 160e-12,
        }
    }

    /// The model, at rest, with this pattern's parameters.
    #[must_use]
    pub fn model(self) -> AdEx {
        let (c, g_l, e_l, a, tau_w, b, v_reset) = match self {
            Self::Tonic => (200e-12, 10e-9, -70e-3, 2e-9, 30e-3, 0.0, -58e-3),
            Self::Adapting => (200e-12, 12e-9, -70e-3, 2e-9, 300e-3, 60e-12, -58e-3),
            Self::InitialBurst => (130e-12, 18e-9, -58e-3, 4e-9, 150e-3, 120e-12, -50e-3),
            Self::RegularBursting => (200e-12, 10e-9, -58e-3, 2e-9, 120e-3, 100e-12, -46e-3),
            Self::Transient => (100e-12, 10e-9, -65e-3, 10e-9, 90e-3, 30e-12, -47e-3),
            Self::Irregular => (100e-12, 12e-9, -60e-3, -11e-9, 130e-3, 30e-12, -48e-3),
        };
        let eif = Eif {
            c,
            g_l,
            e_l,
            v_t: -50e-3,
            delta_t: 2e-3,
            v_peak: 0.0,
            v_reset,
            t_ref: 0.0,
            v: e_l,
            refractory: 0.0,
            substeps: 4,
        };
        AdEx { eif, a, tau_w, b, w: 0.0 }
    }
}

// ---------------------------------------------------------------------------------------------
// Numerics used by the closed forms above
// ---------------------------------------------------------------------------------------------

/// The two real roots of `e^x = x + k`, as `(lower, upper)` with `lower <= 0 <= upper`. `None`
/// unless `k >= 1`, where the pair is not real.
///
/// This is [`Eif`]'s equilibrium condition in units of `Δ_T` about `V_T`: `x = (V - V_T)/Δ_T` and
/// `k = (V_T - E_L - I/g_L)/Δ_T`. Its Lambert `W` form is `x = -W(-e^{-k}) - k`, principal branch
/// for the lower root and `W₋₁` for the upper one — the derivation [`Eif::fixed_points`] quotes.
///
/// **Why not evaluate it that way.** `k` is unbounded above: 10 at the default parameters and zero
/// current, 260 at −5 nA, 2,000 at a `Δ_T` of 0.01 mV. Two things break along that range. `e^{-k}`
/// underflows to `-0.0` for `k > 745`, which reads as "outside the branch pair" — the answer for
/// currents ABOVE rheobase — at a current far below it. And the upper root is `-W₋₁ - k ≈ ln k`,
/// the difference of two numbers of size `k`, so it loses a decimal digit every time `k` gains
/// one. Solving for `x` keeps every quantity at the size of the answer: the roots are `≈ -k` and
/// `≈ ln k`, and neither is a difference of anything larger.
///
/// **Method.** `g(x) = expm1(x) - x - (k - 1)` — `expm1` and `k - 1` rather than `exp` and `k`
/// because at the annihilation the equation is `1 - k` against `x²/2`, two quantities of size 1
/// whose difference is of size `k - 1`. `g` is convex, so a Newton step from outside the root pair
/// can never cross a root (the tangent of a convex function lies under it), and the iteration is
/// monotone from either side. The start is the exact branch-point expansion of the pair as they
/// annihilate, `x ≈ ±q(1 ∓ q/6 + q²/36)` with `q = √(2(k - 1))`, returned unrefined for
/// `q < 1e-4`, where `g` cannot be evaluated to better than the answer being asked for. For
/// `q >= 1` the start is instead a few steps of the two contraction maps `x = e^x - k` and
/// `x = ln(x + k)`, which converge quickly when `k` is large and, unlike the series, cannot hand
/// Newton an `x` whose exponential overflows.
fn exp_offset_roots(k: f64) -> Option<(f64, f64)> {
    if !(k >= 1.0) || !k.is_finite() {
        return None;
    }
    let m = k - 1.0;
    let q = (2.0 * m).sqrt();
    let series = |sign: f64| sign * q * (1.0 - sign * q / 6.0 + q * q / 36.0);
    if q < 1e-4 {
        return Some((series(-1.0), series(1.0)));
    }
    let refine = |start: f64| {
        let mut x = start;
        for _ in 0..100 {
            let d = x.exp_m1();
            if d == 0.0 || !d.is_finite() {
                break;
            }
            let step = (x.exp_m1() - x - m) / d;
            if !step.is_finite() {
                break;
            }
            let next = x - step;
            if next == x {
                break;
            }
            x = next;
        }
        x
    };
    let (mut lo, mut hi) = (series(-1.0), series(1.0));
    if q >= 1.0 {
        lo = -k;
        hi = k.ln();
        for _ in 0..4 {
            lo = lo.exp() - k;
            hi = (hi + k).ln();
        }
    }
    Some((refine(lo), refine(hi)))
}

/// One Simpson panel over `[a, b]`.
fn simpson<F: Fn(f64) -> f64>(f: &F, a: f64, b: f64) -> f64 {
    let m = 0.5 * (a + b);
    (b - a) / 6.0 * (f(a) + 4.0 * f(m) + f(b))
}

/// Hard ceiling on the panels one [`Eif::isi`] quadrature may bisect into, 200,000.
///
/// Adaptive quadrature on a sharply peaked integrand can ask for more refinement than any tolerance
/// will ever grant, and the failure mode is not a wrong answer — it is a program that does not
/// return. The budget makes the cost bounded and the result deterministic, at the price of
/// degrading accuracy on integrands too narrow to resolve. [`Eif::isi`] says which currents those
/// are.
pub const QUADRATURE_BUDGET: u32 = 200_000;

/// Adaptive Simpson with Richardson extrapolation, bisecting until the panel pair agrees or the
/// budget runs out.
fn adaptive<F: Fn(f64) -> f64>(
    f: &F,
    a: f64,
    b: f64,
    whole: f64,
    tol: f64,
    depth: u32,
    budget: &mut u32,
) -> f64 {
    let m = 0.5 * (a + b);
    let left = simpson(f, a, m);
    let right = simpson(f, m, b);
    let err = left + right - whole;
    if depth == 0 || *budget == 0 || err.abs() <= 15.0 * tol {
        return left + right + err / 15.0;
    }
    *budget -= 1;
    adaptive(f, a, m, left, 0.5 * tol, depth - 1, budget)
        + adaptive(f, m, b, right, 0.5 * tol, depth - 1, budget)
}

/// `∫ f` over `[a, b]` to a relative tolerance. Zero for an empty or inverted interval.
///
/// Safe on the integrands in this module because each is monotone on the interval it is handed —
/// the caller splits at `v_t`, where `1/F` peaks — so the first coarse panel cannot step over a
/// feature and stop early. That is the failure mode of every adaptive quadrature and the split is
/// what avoids it here.
fn integrate<F: Fn(f64) -> f64>(f: &F, a: f64, b: f64, rel: f64) -> f64 {
    if !(b > a) {
        return 0.0;
    }
    let whole = simpson(f, a, b);
    let tol = (rel * whole.abs()).max(1e-300);
    let mut budget = QUADRATURE_BUDGET;
    adaptive(f, a, b, whole, tol, 40, &mut budget)
}

#[cfg(test)]
mod tests {
    use super::{
        AdEx, EXP_ARG_LIMIT, Eif, FiringPattern, Flow, ModelError, Qif, Theta, canonical_flow,
        exp_offset_roots, try_step, wrap_pi,
    };
    use crate::neuron::Neuron;

    const PI: f64 = std::f64::consts::PI;

    /// Spike times of a model driven by a constant current, seconds.
    fn spike_times<N: Neuron>(n: &mut N, dt: f64, steps: u32, i: f64) -> Vec<f64> {
        let mut t = Vec::new();
        for k in 0..steps {
            assert!(n.potential().is_finite(), "potential went non-finite at step {k}");
            if n.step(dt, i) {
                t.push(f64::from(k) * dt);
            }
        }
        t
    }

    /// Successive differences.
    fn intervals(t: &[f64]) -> Vec<f64> {
        t.windows(2).map(|w| w[1] - w[0]).collect()
    }

    /// The largest ratio between neighbouring values of the SORTED list, and where it sits. A
    /// bimodal sample has one big gap with substantial mass on each side; a unimodal one does not.
    fn widest_sorted_gap(v: &[f64]) -> (f64, usize) {
        let mut s = v.to_vec();
        s.sort_by(|a, b| a.partial_cmp(b).expect("finite intervals"));
        let mut best = (1.0, 0usize);
        for k in 0..s.len().saturating_sub(1) {
            let r = s[k + 1] / s[k];
            if r > best.0 {
                best = (r, k + 1);
            }
        }
        best
    }

    // ----- Lambert W and the EIF's closed-form fixed points -----

    /// Both roots of `e^x = x + k`, against two EXACT targets from one `k` — which is what makes
    /// this a check on the branch and not on the arithmetic.
    ///
    /// `W(-ln2/2)` is `-ln 2` on the principal branch and `-2 ln 2` on the lower one, because both
    /// satisfy `w·e^w = -ln2/2`. Through `x = -W(-e^{-k}) - k` that is `k = ln(2/ln 2)` with roots
    /// `ln2 - k` and `2·ln2 - k`, and the identity is checked here in the coordinate the code
    /// actually solves in.
    ///
    /// Then the identity itself over a range of `k` the branch-point series knows nothing about.
    /// `k` is `(V_T - E_L - I/g_L)/Δ_T`, so it is unbounded above and a published crate will be
    /// handed the far end of it: `k = 1e300` is a `Δ_T` of 0.01 mV at a few nanoamps of
    /// hyperpolarisation. The old Lambert-W evaluation returned a non-root from about `k = 155`
    /// and `None` — which its doc reads as "above rheobase" — from about `k = 745`.
    #[test]
    fn the_exponential_fixed_point_equation_is_solved_on_both_branches() {
        let ln2 = 2.0f64.ln();
        let k = (2.0 / ln2).ln();
        let (lo, hi) = exp_offset_roots(k).expect("k >= 1");
        assert!((lo - (ln2 - k)).abs() < 1e-15, "lower {lo} vs {}", ln2 - k);
        assert!((hi - (2.0 * ln2 - k)).abs() < 1e-15, "upper {hi} vs {}", 2.0 * ln2 - k);
        for &k in &[1.0, 1.000_000_001, 1.01, 1.5, 2.0, 10.0, 155.0, 1e3, 1e6, 1e12, 1e100, 1e300] {
            let (lo, hi) = exp_offset_roots(k).expect("k >= 1");
            assert!(lo <= 0.0 && hi >= 0.0, "k {k}: {lo} and {hi} are on the wrong sides of 0");
            for x in [lo, hi] {
                // Relative to the size of the terms being compared, which is `x + k` — and never
                // smaller than 1, because at the lower root `x + k` is itself the answer.
                let res = (x.exp() - (x + k)).abs();
                let scale = (x + k).abs().max(1.0);
                assert!(res <= 1e-13 * scale, "k {k}: e^{x} misses {x} + {k} by {res}");
            }
        }
        assert_eq!(exp_offset_roots(1.0), Some((0.0, 0.0)), "at k = 1 the pair annihilates at 0");
        assert!(exp_offset_roots(0.999_999).is_none(), "no real pair above rheobase");
        assert!(exp_offset_roots(f64::NAN).is_none());
        assert!(exp_offset_roots(f64::INFINITY).is_none());
    }

    /// The fixed points are roots of the drift, so the drift evaluated at them is zero. This is the
    /// check that the Lambert `W` algebra in the doc was transcribed correctly: a sign error
    /// anywhere in it gives a number that is not a root.
    #[test]
    fn the_eif_fixed_points_are_zeros_of_its_own_drift() {
        let e = Eif::default();
        for &pa in &[0.0, 50.0, 120.0, 175.0, 179.9] {
            let i = pa * 1e-12;
            let (stable, unstable) = e.fixed_points(i).expect("below rheobase");
            assert!(stable <= unstable, "{stable} should not exceed {unstable}");
            for v in [stable, unstable] {
                // Scale: the drift's natural size is g_L·Δ_T/C = 100 V/s, so 1e-9 V/s is a
                // residual of one part in 1e11.
                assert!(e.drift(v, i).abs() < 1e-9, "drift at {v} V is {}", e.drift(v, i));
            }
        }
        // And they vanish exactly where the rheobase says they do.
        assert!(e.fixed_points(e.rheobase() * 1.000_001).is_none());
        assert!(e.fixed_points(e.rheobase() * 0.999_999).is_some());
        // The sweep above is `k = (V_T - E_L - I/g_L)/Δ_T` from 1 to 10, the corner of a domain
        // that is unbounded in `k`. TWO lines leave it, in the two directions a user reaches it
        // from: a hyperpolarising current, and a sharp onset. Both were wrong — a non-root at
        // −5 nA, whose drift was 17% of the drift's own scale, and `None` at Δ_T = 0.01 mV.
        for &na in &[-1.0, -5.0, -20.0, -200.0] {
            let i = na * 1e-9;
            let (stable, unstable) = e.fixed_points(i).expect("below rheobase");
            assert!(stable < unstable, "{na} nA: {stable} and {unstable}");
            for v in [stable, unstable] {
                assert!(e.drift(v, i).abs() < 1e-9, "{na} nA: drift at {v} V is {}", e.drift(v, i));
            }
            // The stable point is the leak's own equilibrium to within the exponential term, which
            // is `Δ_T·e^{-k}` — utterly negligible this far down. That is an independent value
            // for it, not a re-derivation of the same root.
            let leak = e.e_l + i / e.g_l;
            assert!((stable - leak).abs() < 1e-12, "{na} nA: stable {stable} V vs leak {leak} V");
        }
        for &mv in &[0.11, 0.1, 0.05, 0.01, 0.001] {
            let sharp = Eif { delta_t: mv * 1e-3, ..Eif::default() };
            let (stable, unstable) = sharp.fixed_points(0.0).expect("below rheobase");
            assert!(stable < unstable, "Δ_T {mv} mV: {stable} and {unstable}");
            for v in [stable, unstable] {
                let d = sharp.drift(v, 0.0);
                assert!(d.abs() < 1e-9, "Δ_T {mv} mV: drift at {v} V is {d}");
            }
            // A sharper onset puts the unstable point CLOSER to V_T, at V_T + Δ_T·ln k: the soft
            // threshold hardens onto V_T, which is the limit `Eif::lif_limit` names.
            assert!(unstable > sharp.v_t, "Δ_T {mv} mV: unstable {unstable} V is below V_T");
            assert!(unstable < sharp.v_t + 40.0 * sharp.delta_t, "Δ_T {mv} mV: {unstable} V");
        }
    }

    /// Rheobase is a claim about the simulator, not only about the algebra: below it the cell must
    /// be silent for as long as you care to run it, above it it must fire.
    #[test]
    fn the_eif_rheobase_separates_silence_from_firing() {
        let proto = Eif::default();
        let rheo = proto.rheobase();
        assert!((rheo - 180e-12).abs() < 1e-15, "rheobase {rheo} A, expected 180 pA");
        let mut quiet = proto;
        assert!(spike_times(&mut quiet, 1e-5, 100_000, 0.98 * rheo).is_empty());
        assert!(proto.isi(0.98 * rheo).is_err());
        let mut loud = proto;
        assert!(!spike_times(&mut loud, 1e-5, 100_000, 1.05 * rheo).is_empty());
    }

    /// Quadrature against time-stepping. The two share no code: one integrates `dV/F(V)` over
    /// voltage, the other integrates `F(V)/C` over time.
    #[test]
    fn the_eif_interval_matches_the_quadrature() {
        let proto = Eif { t_ref: 2e-3, ..Eif::default() };
        for &pa in &[200.0, 250.0, 400.0, 800.0, 2000.0] {
            let i = pa * 1e-12;
            let want = proto.isi(i).expect("above rheobase");
            let mut n = proto;
            let dt = 2e-6;
            let t = spike_times(&mut n, dt, 400_000, i);
            assert!(t.len() > 3, "{pa} pA gave {} spikes", t.len());
            let iv = intervals(&t);
            let got = iv[iv.len() - 1];
            let rel = (got - want).abs() / want;
            assert!(rel < 2e-3, "{pa} pA: stepped {got} s vs quadrature {want} s");
        }
    }

    /// A reset ABOVE the point where the exponential term is truncated used to reach
    /// `f64::clamp` with its bounds crossed, which panics. `Eif::new` accepts that membrane — a
    /// 0.1 mV onset with a reset 6 mV above `V_T` — so the panic was reachable from the public
    /// API on parameters the constructor had just approved, and `Eif::rate` inherited it.
    #[test]
    fn the_eif_interval_survives_a_reset_above_the_truncated_upstroke() {
        let e = Eif::new(200e-12, 10e-9, -58e-3, -50e-3, 1e-4, 0.0, -44e-3, 0.0).expect("accepted");
        assert!(
            e.v_reset > e.v_t + EXP_ARG_LIMIT * e.delta_t,
            "this test is vacuous unless the reset is above the truncation: {} vs {}",
            e.v_reset,
            e.v_t + EXP_ARG_LIMIT * e.delta_t
        );
        let i = 500e-12;
        assert!(i > e.rheobase(), "and it needs a current above rheobase {}", e.rheobase());
        let t = e.isi(i).expect("above rheobase");
        assert!(t.is_finite() && t >= 0.0, "isi returned {t}");
        // Everything above the truncation is worth under 1e-20 s per volt, so the whole excursion
        // from this reset to the cutoff is less than a femtosecond: the answer is t_ref and a
        // remainder no interval can carry.
        assert!(t < 1e-15, "the remainder above the truncation priced at {t} s");
        let with_ref = Eif { t_ref: 2e-3, ..e };
        let held = with_ref.isi(i).expect("above rheobase");
        assert!((held - 2e-3).abs() < 1e-15, "t_ref is not carried: {held}");
        assert!((with_ref.rate(i).expect("above rheobase") - 1.0 / held).abs() < 1e-9);
        // And the stepper agrees that a cell reset that far up fires again immediately.
        let mut n = e;
        n.v = e.v_reset;
        assert!(n.step(1e-5, i), "a cell reset above the upstroke did not fire within a tick");
    }

    /// The exponential model is bistable below rheobase for the same reason the quadratic one is,
    /// and the taxonomy SHIPS such a membrane: `FiringPattern::RegularBursting` resets to −46 mV
    /// against a `V_T` of −50 mV. `Eif::isi` used to answer `NoFiring` — "there is no interval,
    /// not a long one" — for a cell its own stepper fires 45 times in 200 ms.
    #[test]
    fn an_eif_reset_above_the_unstable_point_fires_below_rheobase() {
        let e = FiringPattern::RegularBursting.model().eif;
        let rheo = e.rheobase();
        for &frac in &[0.1, 0.5, 0.95] {
            let i = frac * rheo;
            let (_, unstable) = e.fixed_points(i).expect("below rheobase");
            assert!(e.v_reset > unstable, "{frac}x: reset {} vs unstable {unstable}", e.v_reset);
            let want = e.isi(i).expect("a reset above the unstable point escapes");
            let mut n = e;
            n.v = e.v_reset;
            let t = spike_times(&mut n, 1e-6, 200_000, i);
            assert!(t.len() > 3, "{frac}x rheobase gave {} spikes in 200 ms", t.len());
            let iv = intervals(&t);
            let got = iv[iv.len() - 1];
            assert!(
                (got - want).abs() / want < 2e-3,
                "{frac}x rheobase: stepped {got} s vs quadrature {want} s"
            );
        }
        // The same membrane with a reset BELOW the unstable point is silent at the same current,
        // so what fires the cell is the reset and not the current.
        let cold = Eif { v_reset: -58e-3, ..e };
        let i = 0.5 * rheo;
        assert!(cold.fixed_points(i).expect("below rheobase").1 > cold.v_reset);
        assert!(matches!(cold.isi(i), Err(ModelError::NoFiring { .. })), "the cold cell answered");
        let mut n = cold;
        assert!(spike_times(&mut n, 1e-5, 100_000, i).is_empty(), "the cold cell fired");
    }

    /// The band `Eif::isi`'s doc calls degraded, measured rather than asserted.
    ///
    /// The reference is composite Simpson on panels that grow geometrically away from `V_T`. That
    /// is a different algorithm from the adaptive bisection in `integrate` — fixed panels placed
    /// by the integrand's own closed-form width `√(2Δ_T(I - I_rheo)/g_L)`, against bisection
    /// driven by a tolerance — and it is run at two resolutions, with the assertion that they
    /// agree with each other far more tightly than the claim under test. That is what makes it a
    /// reference rather than a second opinion.
    ///
    /// Both sides of the doc's claim are asserted, which is the point: outside the band the
    /// quadrature is the exact integral to 1e-9, and inside it the answer really is degraded. A
    /// quadrature good enough to fail the second assertion is a better crate — and a doc that
    /// needs rewriting, which is why the assertion is here.
    #[test]
    fn the_interval_quadrature_holds_to_the_band_its_doc_claims() {
        fn reference(e: &Eif, i: f64, per_panel: usize) -> f64 {
            let f = |v: f64| e.c / e.current(v, i);
            let lo = e.v_reset;
            let hi = e.v_peak.min(e.v_t + EXP_ARG_LIMIT * e.delta_t);
            let width = (2.0 * e.delta_t * (i - e.rheobase()) / e.g_l).sqrt();
            let scale = width.max((hi - lo) * 1e-13);
            let mut bounds = vec![lo, hi, e.v_t];
            let mut w = scale;
            let mut x = e.v_t - w;
            while x > lo {
                bounds.push(x);
                w *= 2.0;
                x -= w;
            }
            let mut w = scale;
            let mut x = e.v_t + w;
            while x < hi {
                bounds.push(x);
                w *= 2.0;
                x += w;
            }
            bounds.sort_by(|a, b| a.partial_cmp(b).expect("finite bounds"));
            let mut total = 0.0;
            for pair in bounds.windows(2) {
                let (a, b) = (pair[0], pair[1]);
                if !(b > a) {
                    continue;
                }
                let h = (b - a) / per_panel as f64;
                let mut sum = f(a) + f(b);
                for j in 1..per_panel {
                    let weight = if j % 2 == 1 { 4.0 } else { 2.0 };
                    sum += weight * f(a + h * j as f64);
                }
                total += sum * h / 3.0;
            }
            total + e.t_ref
        }
        let e = Eif { v_reset: -70e-3, v_peak: 20e-3, ..Eif::default() };
        let rheo = e.rheobase();
        // Outside the band: 1e-15 A above an 180 pA rheobase is one part in 1.8e5, two decades
        // coarser than the doc's 1e6, and the sweep in `the_eif_rate_converges_onto_the_square_
        // root_law` runs to exactly there.
        for &excess in &[1e-9, 1e-12, 1e-14, 1e-15] {
            let i = rheo + excess;
            let coarse = reference(&e, i, 2_000);
            let fine = reference(&e, i, 8_000);
            let settled = (coarse - fine).abs() / fine;
            assert!(
                settled < 1e-11,
                "excess {excess} A: the reference disagrees with itself by {settled}"
            );
            let got = e.isi(i).expect("above rheobase");
            let rel = (got - fine).abs() / fine;
            assert!(rel < 1e-9, "excess {excess} A: isi {got} s vs reference {fine} s, {rel}");
        }
        // Inside it: one part in 1.8e7 of rheobase, where the doc says the value is an estimate.
        let i = rheo + 1e-17;
        let fine = reference(&e, i, 8_000);
        let got = e.isi(i).expect("above rheobase");
        let rel = (got - fine).abs() / fine;
        assert!(rel > 1e-6, "one part in 1.8e7 now resolves to {rel}; the doc needs rewriting");
        assert!(rel < 1e-2, "the degraded estimate is off by {rel}, which is not an estimate");
    }

    /// Check (b) of the module's brief: the exponential model becomes a `Lif` as `Δ_T -> 0`, and
    /// the residual must SHRINK. The limit is `Eif::lif_limit`, whose rate is a closed form with no
    /// quadrature in it at all.
    #[test]
    fn the_eif_relaxes_onto_the_lif_as_delta_t_shrinks() {
        let i = 400e-12;
        let mut last = f64::INFINITY;
        let mut errs = Vec::new();
        for &mv in &[2.0, 1.0, 0.5, 0.2, 0.05, 0.01, 0.002, 0.0005] {
            let e = Eif { delta_t: mv * 1e-3, ..Eif::default() };
            let lif = e.lif_limit();
            let want = lif.rate(i).expect("supra-threshold");
            let got = e.rate(i).expect("above rheobase");
            let rel = (got - want).abs() / want;
            errs.push((mv, rel));
            assert!(rel < last, "Δ_T = {mv} mV gave {rel}, which is not below {last}: {errs:?}");
            last = rel;
        }
        assert!(last < 2e-3, "the smallest Δ_T still differs from the Lif by {last}: {errs:?}");
    }

    /// Type I onset: the rate rises from zero like `√(I - I_rheo)`. Checked by convergence onto
    /// `Eif::saddle_node_rate`, which is an asymptote and is only right in the limit — so the test
    /// asserts that the gap closes, not that it is small at any particular current.
    #[test]
    fn the_eif_rate_converges_onto_the_square_root_law() {
        let e = Eif { v_reset: -70e-3, v_peak: 20e-3, ..Eif::default() };
        let rheo = e.rheobase();
        let mut last = f64::INFINITY;
        // Decades 1e-11 and 1e-12 are SKIPPED on purpose and the omission is the finding: the
        // approach is not monotone there. Two corrections of opposite sign cross over near 2% —
        // the finite reset and cutoff truncate the bottleneck integral (making the true rate
        // faster than the asymptote), while the drift above `V_T` grows faster than the quadratic
        // expansion allows (making it slower). Measured: 1.35, 0.188, 0.0179, 0.0193, 0.0078,
        // 0.0027, 0.00086 for excesses 1e-9 down to 1e-15.
        for &excess in &[1e-10, 1e-12, 1e-13, 1e-14, 1e-15] {
            let i = rheo + excess;
            let got = e.rate(i).expect("above rheobase");
            let want = e.saddle_node_rate(i).expect("above rheobase");
            let rel = (got - want).abs() / want;
            assert!(rel < last, "excess {excess} A: relative gap {rel} did not fall below {last}");
            last = rel;
        }
        assert!(last < 2e-3, "the closest current still differs by {last}");
        // And the law itself, without the asymptote's prefactor in the way: a hundredfold drop in
        // the excess current must divide the rate by ten. This is the square root, measured.
        // Both ENDS of each pair have to be in the asymptotic regime, which is why the pairs start
        // at 1e-13 and not at 1e-11: a pair anchored at 1e-9 measures a slope of 0.69, because that
        // end is still 135% away from the asymptote.
        for &excess in &[1e-13, 1e-14, 1e-15] {
            let hi = e.rate(rheo + 100.0 * excess).expect("above rheobase");
            let lo = e.rate(rheo + excess).expect("above rheobase");
            let slope = (hi / lo).ln() / 100.0f64.ln();
            assert!((slope - 0.5).abs() < 0.01, "excess {excess} A: log-log slope {slope}, not 1/2");
        }
    }

    /// Sub-threshold means never, and the error says at what current it would not.
    #[test]
    fn a_subthreshold_model_names_its_rheobase_instead_of_guessing() {
        let e = Eif::default();
        match e.isi(100e-12) {
            Err(ModelError::NoFiring { i, rheobase }) => {
                assert!((i - 100e-12).abs() < 1e-24);
                assert!((rheobase - e.rheobase()).abs() < 1e-24);
            }
            other => panic!("expected NoFiring, got {other:?}"),
        }
        let q = Qif::default();
        assert!(q.isi(q.rheobase() * 0.9).is_err());
        let th = Theta::default();
        assert!(th.isi(th.rheobase() * 0.9).is_err());
    }

    // ----- QIF -----

    /// Check (c): the quadratic model's interval is elementary, and the stepper must reproduce it.
    #[test]
    fn the_qif_interval_matches_its_closed_form() {
        let proto = Qif::default();
        // 500 pA is 1.33x rheobase and its interval is already 180 ms: a quadratic model just
        // above onset is SLOW, which is the square-root law showing up as a test-runtime problem.
        // The window is 1.2 s so that even that current gives several intervals to compare.
        for &pa in &[500.0, 800.0, 2000.0, 6000.0, 20_000.0] {
            let i = pa * 1e-12;
            let want = proto.isi(i).expect("above rheobase");
            let mut n = proto;
            let dt = 2e-6;
            let t = spike_times(&mut n, dt, 600_000, i);
            assert!(t.len() > 3, "{pa} pA gave {} spikes", t.len());
            let iv = intervals(&t);
            let got = iv[iv.len() - 1];
            assert!(
                (got - want).abs() / want < 2e-3,
                "{pa} pA: stepped {got} s vs closed form {want} s"
            );
        }
    }

    /// With the cutoff and the reset taken far out, the interval collapses to the textbook Type I
    /// rate `f = √η/(π τ)` — and `Theta::isi` is that formula written directly, so the two must
    /// agree without either knowing about the other's derivation.
    #[test]
    fn the_qif_collapses_to_the_type_one_rate_with_distant_bounds() {
        let base = Qif::default();
        let wide = Qif { v_peak: 1e6, v_reset: -1e6, t_ref: 0.0, ..base };
        let th = wide.matching_theta();
        for &pa in &[400.0, 600.0, 1500.0, 9000.0] {
            let i = pa * 1e-12;
            let eta = wide.eta(i);
            let closed = wide.tau_m * PI / eta.sqrt();
            let got = wide.isi(i).expect("above rheobase");
            assert!((got - closed).abs() / closed < 1e-6, "{pa} pA: {got} vs τπ/√η {closed}");
            let via_theta = th.isi(i).expect("above rheobase");
            assert!((via_theta - closed).abs() / closed < 1e-12, "{via_theta} vs {closed}");
        }
    }

    /// `Qif::matching_theta` drops `t_ref`, because the circle has no refractory period and no
    /// place to put one. The exception is exact and it is documented; here it is also measured.
    /// `the_qif_collapses_to_the_type_one_rate_with_distant_bounds` sets `t_ref: 0.0` to get past
    /// it, which is how an undocumented exception hides from a test suite.
    #[test]
    fn the_matching_theta_drops_the_refractory_period_and_nothing_else() {
        let q = Qif { v_peak: 1e9, v_reset: -1e9, t_ref: 3e-3, ..Qif::default() };
        let th = q.matching_theta();
        for &pa in &[400.0, 600.0, 1500.0, 9000.0] {
            let i = pa * 1e-12;
            let quadratic = q.isi(i).expect("above rheobase");
            let circle = th.isi(i).expect("above rheobase");
            assert!(
                (quadratic - circle - q.t_ref).abs() / quadratic < 1e-9,
                "{pa} pA: qif {quadratic} s, theta {circle} s, t_ref {} s",
                q.t_ref
            );
            // And the gap is not decorative: at 400 pA it is a fifth of the interval.
            assert!(q.t_ref > 1e-3 * circle, "t_ref {} is too small to see", q.t_ref);
        }
    }

    /// The exact flow, against fourth-order Runge-Kutta on the raw voltage equation. Different
    /// code, different variables: this is what says the Riccati algebra in `canonical_flow` is the
    /// solution of `Qif::drift` and not of something else.
    #[test]
    fn the_qif_exact_flow_solves_its_own_drift() {
        let q = Qif { v_peak: 1.0, v_reset: -1.0, ..Qif::default() };
        for &pa in &[0.0, 200.0, 375.0, 600.0] {
            let i = pa * 1e-12;
            let mut exact = q;
            exact.v = -60e-3;
            let mut v = exact.v;
            let dt = 1e-7;
            let steps = 20_000; // 2 ms
            for _ in 0..steps {
                let k1 = q.drift(v, i);
                let k2 = q.drift(v + 0.5 * dt * k1, i);
                let k3 = q.drift(v + 0.5 * dt * k2, i);
                let k4 = q.drift(v + dt * k3, i);
                v += dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;
            }
            exact.step(f64::from(steps) * dt, i);
            assert!(
                (exact.v - v).abs() < 1e-11,
                "{pa} pA: exact flow {} V vs RK4 {v} V",
                exact.v
            );
        }
    }

    /// The flow composes: one jump of ten ticks lands where ten ticks land.
    ///
    /// This is the property `EXACT_OVER_GAPS` names, and [`Qif`] HAS it — which is exactly why the
    /// constant is declared false on other grounds. Keeping this test is the point: it says the
    /// refusal is not a numerical excuse, and it would catch a later edit that made the flow
    /// approximate while leaving the reasoning about bistability in place.
    ///
    /// Started at `y = 0` rather than at rest ON PURPOSE: at zero current the resting potential is
    /// the stable fixed point, so a test that started there would compare a stationary state
    /// against itself and pass whatever the flow did.
    #[test]
    fn the_qif_flow_composes_across_a_gap() {
        // 10 steps of 20 ms is ten membrane constants, which is what it takes for "decayed toward
        // rest" to mean anything: one tick of 0.1 ms moves the potential by microvolts and the
        // final assertion would pass on a flow that did nothing.
        let dt = 2e-2;
        let mut fine = Qif::default();
        fine.set_canonical_y(0.0);
        let mut coarse = fine;
        assert!((fine.v - fine.v_rest).abs() > 1e-3, "the start must not be the fixed point");
        for _ in 0..10 {
            fine.step(dt, 0.0);
        }
        coarse.step(10.0 * dt, 0.0);
        assert!((fine.v - coarse.v).abs() < 1e-14, "fine {} vs coarse {}", fine.v, coarse.v);
        assert!((fine.v - fine.v_rest).abs() < 1e-3, "it should have decayed toward rest");
    }

    /// Below rheobase the quadratic model has two fixed points, and they are zeros of the drift.
    #[test]
    fn the_qif_fixed_points_are_zeros_of_its_drift() {
        let q = Qif::default();
        for &pa in &[0.0, 100.0, 300.0, 374.0] {
            let i = pa * 1e-12;
            let (s, u) = q.fixed_points(i).expect("below rheobase");
            assert!(s < u);
            assert!(q.drift(s, i).abs() < 1e-12 && q.drift(u, i).abs() < 1e-12);
        }
        assert!(q.fixed_points(q.rheobase() * 1.01).is_none());
    }

    /// A trajectory started exactly ON the unstable fixed point stays there — and that point is
    /// what `Qif::fixed_points` hands you, so this is one public call feeding another.
    ///
    /// The `tanh` form of the flow is `0/0` there (`y₀ = a` makes `a - y₀·tanh(a·h)` and
    /// `a·(y₀ - a·tanh(a·h))` vanish together once `tanh` saturates), and the result was a `NaN`
    /// membrane potential — which, as `EXP_ARG_LIMIT`'s own doc warns, does not fail loudly: the
    /// cell reported zero spikes at more than twice rheobase for as long as it was run, and
    /// `Neuron::step` kept returning false.
    #[test]
    fn a_qif_started_on_its_unstable_fixed_point_stays_finite() {
        let mut q = Qif::default();
        let (_, unstable) = q.fixed_points(0.0).expect("below rheobase");
        q.v = unstable;
        // The premise, with its number: the round trip through volts lands on y = a EXACTLY, which
        // is what makes the 0/0 reachable rather than merely near.
        let a = (-q.eta(0.0)).sqrt();
        assert_eq!(q.canonical_y(), a, "the public fixed point does not round-trip to y = a");
        assert!(!q.step(1.0, 0.0), "a fixed point is not a spike");
        assert!(q.v.is_finite(), "a step from the unstable fixed point left {} V", q.v);
        assert!((q.v - unstable).abs() < 1e-15, "it left its own fixed point, at {} V", q.v);
        // And the cell is still a cell: at more than twice rheobase it fires.
        let t = spike_times(&mut q, 1e-4, 10_000, 800e-12);
        assert!(t.len() > 4, "after sitting on the fixed point it gave {} spikes in 1 s", t.len());
        // A hair above the fixed point it must ESCAPE instead, with no input at all, and the
        // closed form must say when — `Qif::isi`'s atanh branch, evaluated a part in 1e9 above the
        // unstable point rather than at the comfortable 2 mV of
        // `a_qif_reset_above_the_unstable_point_fires_below_rheobase`.
        let base = Qif::default();
        let hair_up = base.v_mid() + base.delta() * a * (1.0 + 1e-9);
        let hot = Qif { v_reset: hair_up, t_ref: 0.0, ..base };
        assert!(hot.v_reset > unstable, "the nudge must land above the fixed point");
        let want = hot.isi(0.0).expect("a reset above the unstable point escapes");
        let mut n = hot;
        n.v = hot.v_reset;
        let escape = spike_times(&mut n, 1e-4, 20_000, 0.0);
        assert!(escape.len() >= 3, "a hair above the unstable point gave {} spikes", escape.len());
        let iv = intervals(&escape);
        let got = iv[iv.len() - 1];
        assert!((got - want).abs() < 1e-3 * want, "stepped {got} s vs closed form {want} s");
        // Closer in, only the COUNT is asserted, and the reason is the model rather than the flow:
        // this type stores volts, so a `y` a part in 1e12 above `a` is eight ulps of `v`, the
        // stepper re-quantises the distance from the fixed point on every tick, and the escape
        // time inherits that — 0.563 s stepped against 0.566 s in closed form. The escape survives
        // it; the timing does not, and saying so is cheaper than a tolerance that hides it.
        let mut hair = Qif::default();
        hair.set_canonical_y(a * (1.0 + 1e-12));
        assert!(hair.v > unstable, "eight ulps must still land above the fixed point");
        let out = spike_times(&mut hair, 1e-4, 20_000, 0.0);
        assert_eq!(out.len(), 1, "eight ulps above the unstable point did not escape: {out:?}");
    }

    /// The divergence time from just above the unstable fixed point, where `atanh(a/y₀)` loses the
    /// answer: `a/y₀` rounds to within an ulp of 1 and `atanh` amplifies that by `1/(1 - x²)`.
    ///
    /// The reference is independent of the formula being checked. Advance to a hair before the
    /// divergence, where `y` is enormous; from there `dy/ds = y² + η` is `y²` to a part in
    /// `(a/y)²`, so the time left is `1/y`. That is the spike time measured from the far end.
    ///
    /// **`η = -1/4` alone would prove nothing**, and that is the point of the sweep: `a = 1/2` is
    /// a power of two, `a/y₀` is then exactly representable for a `y₀` a few ulps up, and the old
    /// `atanh` form is exact there by luck — 2e-16 relative. One `η` away from that symmetry the
    /// same form is wrong by 4e-4, and at `a = 1.35` with `y₀` three ulps up, by 1.1e-2.
    #[test]
    fn the_divergence_time_is_accurate_next_to_the_unstable_fixed_point() {
        for &eta in &[-0.25f64, -0.37, -0.61, -1.85, -3.3] {
            let a = (-eta).sqrt();
            for &shift in &[1e-6, 1e-10, 1e-13, 1e-15, 4e-16] {
                let y0 = a * (1.0 + shift);
                assert!(y0 > a, "η {eta}, shift {shift}: the nudge was lost to rounding");
                let Flow::Diverged { at } = canonical_flow(y0, eta, 1e6) else {
                    panic!("a start above the unstable point must diverge, η {eta}, shift {shift}")
                };
                let h = at * (1.0 - 1e-6);
                let Flow::Finite(y1) = canonical_flow(y0, eta, h) else {
                    panic!("η {eta}, shift {shift}: it diverged before its own divergence time")
                };
                assert!(y1 > 1e3 * a, "η {eta}, shift {shift}: {y1} is not deep in the escape");
                let left = at - h;
                let asymptotic = 1.0 / y1;
                assert!(
                    (left - asymptotic).abs() <= 1e-4 * left,
                    "η {eta}, shift {shift}: {left} left by the closed form, {asymptotic} by 1/y"
                );
            }
        }
        // Exactly on the point, the flow is the point.
        for &e in &[-0.37f64, -1.0, -4.0] {
            let a = (-e).sqrt();
            assert_eq!(canonical_flow(a, e, 1.0), Flow::Finite(a), "η {e} moved off its own root");
            assert_eq!(canonical_flow(a, e, 1e6), Flow::Finite(a), "η {e}, long step");
            assert_eq!(canonical_flow(-a, e, 1e6), Flow::Finite(-a), "η {e}, stable point");
        }
    }

    /// Bistability is a feature of the model and `isi` must not flatten it: a reset above the
    /// unstable fixed point fires forever at a current well below rheobase.
    #[test]
    fn a_qif_reset_above_the_unstable_point_fires_below_rheobase() {
        let q = Qif::default();
        let i = 0.5 * q.rheobase();
        let (_, unstable) = q.fixed_points(i).expect("below rheobase");
        let hot = Qif { v_reset: unstable + 2e-3, ..q };
        let want = hot.isi(i).expect("bistable regime still fires");
        assert!(want.is_finite() && want > 0.0);
        let mut n = hot;
        n.v = hot.v_reset;
        let t = spike_times(&mut n, 1e-6, 400_000, i);
        assert!(t.len() > 3, "bistable cell gave {} spikes", t.len());
        let iv = intervals(&t);
        let got = iv[iv.len() - 1];
        assert!((got - want).abs() / want < 2e-3, "stepped {got} s vs atanh form {want} s");
        // And the same cell reset BELOW the unstable point is silent at the same current.
        let cold = Qif { v_reset: q.v_rest, ..q };
        assert!(cold.isi(i).is_err());
    }

    // ----- Theta, and the correspondence -----

    /// Check (a), first leg. `Theta` steps by Runge-Kutta on the circle; `Qif` steps by the exact
    /// Riccati flow in volts. They share no line of code. If `V = v_mid + Δ·tan(θ/2)` is the right
    /// transform, the two trajectories are the same trajectory.
    #[test]
    fn the_theta_neuron_and_the_qif_are_the_same_system() {
        let q = Qif { v_peak: 1e9, v_reset: -1e9, t_ref: 0.0, ..Qif::default() };
        for &pa in &[0.0, 200.0, 500.0, 1200.0] {
            let i = pa * 1e-12;
            let mut qif = q;
            qif.set_canonical_y(-0.3);
            let mut th = qif.matching_theta();
            th.substeps = 64;
            let dt = 2e-5;
            for k in 1..=200u32 {
                qif.step(dt, i);
                th.step(dt, i);
                let want = qif.canonical_y();
                let got = th.canonical_y();
                assert!(
                    (got - want).abs() <= 1e-9 * (1.0 + want.abs()),
                    "{pa} pA, step {k}: tan(θ/2) = {got} vs y = {want}"
                );
            }
        }
    }

    /// Check (a), second leg, at the tolerance the brief asks for. Above rheobase
    /// `Theta::exact_theta_after` is written in the circle's own coordinates — a phase rotation
    /// and an `atan2` — while `canonical_flow` is written as a Riccati solution in `y`. Agreement
    /// there is floating-point, not discretisation, and 500 and 3000 pA are the legs that say so.
    ///
    /// **At and below rheobase this test compares a copy against its copy**, and it is kept for
    /// what it does cover rather than for what it is named: the two functions evaluate the SAME
    /// expression for `η <= 0` (`a(1 + u)/(1 - u)` in both, `y₀/(1 - y₀s)` in both), so 0, 100 and
    /// 375 pA would pass a shared transcription error. The independent check for those is
    /// `the_theta_closed_form_solves_the_circle_equation_it_is_written_for`, which integrates the
    /// circle equation itself and shares nothing with either.
    #[test]
    fn the_theta_closed_form_and_the_qif_flow_agree_to_floating_point() {
        let q = Qif::default();
        for &pa in &[0.0, 100.0, 375.0, 500.0, 3000.0] {
            let i = pa * 1e-12;
            let eta = q.eta(i);
            for &y0 in &[-2.0f64, -0.6, 0.0, 0.4, 3.0] {
                let mut th = q.matching_theta();
                th.theta = 2.0 * y0.atan();
                for &ms in &[0.5, 2.0, 7.0] {
                    let t = ms * 1e-3;
                    let got = th.exact_theta_after(i, t).expect("finite");
                    let want = match canonical_flow(y0, eta, t / q.tau_m) {
                        Flow::Finite(y1) => 2.0 * y1.atan(),
                        // Past the divergence the flow returns from -∞; re-enter it with the
                        // remaining time from a point just below, which the closed form does for
                        // free. Skip rather than approximate.
                        Flow::Diverged { .. } => continue,
                    };
                    assert!(
                        (got - want).abs() < 1e-12,
                        "η {eta}, y0 {y0}, {ms} ms: θ {got} vs 2·atan(y) {want}"
                    );
                }
            }
        }
    }

    /// `Theta::exact_theta_after` against Runge-Kutta on the circle equation itself, which is the
    /// only reference here that shares no algebra with either closed form.
    ///
    /// It integrates `dθ/ds = (1 - cos θ) + (1 + cos θ)η` in radians at 20,000 steps per window,
    /// where fourth-order truncation is far below the tolerance asserted, and compares AROUND the
    /// circle so that a pair straddling the cut is not counted as `2π` apart. Below rheobase this
    /// is the only thing standing between a transcribed Riccati solution and a plausible wrong
    /// trajectory.
    #[test]
    fn the_theta_closed_form_solves_the_circle_equation_it_is_written_for() {
        let q = Qif::default();
        for &pa in &[0.0, 100.0, 375.0, 500.0, 3000.0] {
            let i = pa * 1e-12;
            for &y0 in &[-2.0f64, -0.6, 0.0, 0.4, 3.0] {
                let mut th = q.matching_theta();
                th.theta = 2.0 * y0.atan();
                let eta = th.eta(i);
                for &ms in &[0.5, 2.0] {
                    let t = ms * 1e-3;
                    let n = 20_000u32;
                    let h = (t / th.tau) / f64::from(n);
                    let mut x = th.theta;
                    for _ in 0..n {
                        let k1 = th.drift(x, eta);
                        let k2 = th.drift(x + 0.5 * h * k1, eta);
                        let k3 = th.drift(x + 0.5 * h * k2, eta);
                        let k4 = th.drift(x + h * k3, eta);
                        x += h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;
                    }
                    let got = th.exact_theta_after(i, t).expect("finite");
                    let gap = wrap_pi(got - x);
                    assert!(
                        gap.abs() < 1e-11,
                        "η {eta}, y0 {y0}, {ms} ms: closed form {got} vs Runge-Kutta {}",
                        wrap_pi(x)
                    );
                }
            }
        }
        // And on the unstable fixed point itself, where `u₀ · e^{2as}` is `0 · inf`: the phase
        // stays put, for any interval. Landing on it takes an ulp of care — `tan(atan(a)/1)` comes
        // back one ulp ABOVE `a`, not on it — and that near miss is the other half of this test.
        let a = 0.5f64;
        let mut th = Qif::default().matching_theta();
        th.theta = (2.0 * a.atan()).next_down();
        assert_eq!(th.canonical_y(), a, "the premise is a phase whose tan(θ/2) is a exactly");
        for &t in &[1e-3, 1.0, 1e6] {
            let got = th.exact_theta_after(0.0, t).expect("finite");
            assert!((got - th.theta).abs() < 1e-12, "{t} s moved the fixed phase to {got}");
        }
        // One ulp above it the cell must SPIKE and then settle onto the stable point at `y = -a`,
        // and `spikes_by` must say one. The answer -a is right; arriving there without the spike,
        // which is what `a - y₀·tanh(a·s)` cancelling to a rounding produces, is not.
        let mut hot = Qif::default().matching_theta();
        hot.theta = 2.0 * a.atan();
        assert!(hot.canonical_y() > a, "the premise is a phase one ulp above the fixed point");
        assert_eq!(hot.spikes_by(0.0, 100.0).expect("finite"), 1, "the escape produced no spike");
        let settled = hot.exact_theta_after(0.0, 100.0).expect("finite");
        assert!((settled - 2.0 * (-a).atan()).abs() < 1e-9, "it settled at θ = {settled}");
    }

    /// Check (e): the circle is invariant. Drive it with everything, including currents that make
    /// `η` hugely negative, and `θ` must stay in `(-π, π]` — and stay finite, which `tan(θ/2)` on
    /// the other side of the transform does not.
    #[test]
    fn the_theta_neuron_never_leaves_the_invariant_circle() {
        let mut th = Theta { substeps: 8, ..Theta::default() };
        let drives = [0.0, 1e-9, -1e-9, 5e-9, -5e-8, 1e-7];
        for (k, &i) in drives.iter().cycle().take(60_000).enumerate() {
            th.step(1e-5, i);
            assert!(th.theta.is_finite(), "θ went non-finite at step {k}");
            assert!(th.theta > -PI - 1e-12 && th.theta <= PI + 1e-12, "θ = {} at {k}", th.theta);
        }
        // The drives above are three to four orders of magnitude too small to break anything, so
        // they are not the test. The invariant fails when `h·|η|` reaches about 2, which at a 1 ms
        // tick is i ≈ -300 nA and at the 0.1 ms tick the simulator test uses is i ≈ -3 µA. Below
        // is the ±1 mA the module's own `a_violent_drive_leaves_no_model_non_finite` uses, at the
        // dt it uses — the drive that sent θ to -28,486 while every assertion in that test passed,
        // because it asserts finiteness and this is a property finiteness does not imply.
        let mut wild = Theta::default();
        for k in 0..20_000 {
            let i = if k % 2 == 0 { 1e-3 } else { -1e-3 };
            wild.step(1e-3, i);
            assert!(wild.theta.is_finite(), "θ went non-finite at step {k}");
            assert!(
                wild.theta > -PI - 1e-12 && wild.theta <= PI + 1e-12,
                "θ = {} at step {k} of the violent drive",
                wild.theta
            );
        }
        // And what the escape actually costs is the spike train, so that is asserted too: a phase
        // parked at -42,790 has to climb 13,600 radians before `θ > π` can fire again, and every
        // spike until then is silently lost.
        let i = 8.0 * wild.rheobase();
        let want = wild.isi(i).expect("above rheobase");
        let after = spike_times(&mut wild, 1e-5, 20_000, i);
        let n = after.len();
        assert!(n >= 3, "after the violent drive it gave {n} spikes in 200 ms");
        assert!(after[0] < 2.0 * want, "first spike at {} s; one interval is {want} s", after[0]);
        // A bump of any size is a rotation, never an escape — and the property that says the
        // rotation is the RIGHT one is that it moves the corresponding Qif's potential by exactly
        // the volts it was handed. `θ.abs() <= PI` holds for `2·atan` of anything, so on its own
        // it asserts the definition of `atan` rather than anything about this model.
        for dv in [1.0, -1.0, 1e6, -1e6] {
            let before = th.potential();
            th.bump(dv);
            let moved = th.potential() - before;
            assert!(
                (moved - dv).abs() <= 1e-6 * dv.abs(),
                "a {dv} V bump moved the membrane {moved} V"
            );
            let on = th.theta > -PI && th.theta <= PI;
            assert!(on, "a {dv} V bump left the circle at θ = {}", th.theta);
        }
    }

    /// The spike count over a window is an exact integer from the phase formula. The Runge-Kutta
    /// stepper must produce that integer, not one near it.
    #[test]
    fn the_theta_spike_count_matches_the_exact_phase_formula() {
        for &pa in &[800.0, 2000.0, 6000.0] {
            let i = pa * 1e-12;
            let mut th = Theta { substeps: 8, ..Theta::default() };
            let want = th.spikes_by(i, 2.0).expect("finite");
            let dt = 2e-6;
            let steps = 1_000_000u32;
            let got = spike_times(&mut th, dt, steps, i).len() as u64;
            assert!(
                got == want || got + 1 == want || got == want + 1,
                "{pa} pA: stepped {got} spikes vs exact {want}"
            );
            assert!(want > 2, "{pa} pA predicted only {want} spikes, which tests nothing");
        }
        // The count is a FLOOR of a linear function, and the ±1 slack above — which the stepper's
        // tick quantisation honestly earns — cannot see that. So the floor is pinned separately,
        // against windows whose answer is arithmetic rather than measured: from `θ = 0` the first
        // crossing is HALF an interval in, so a window of exactly one interval holds exactly one
        // spike, 1.501 intervals hold two, and a hair under half an interval holds none. Rounding
        // instead of flooring answers 2 to the third of those, which nothing else here would
        // notice. FOUND BY MUTATION.
        let clock = Theta { theta: 0.0, ..Theta::default() };
        let i = 2000e-12;
        let isi = clock.isi(i).expect("above rheobase");
        for &(window, want) in &[(0.499, 0u64), (0.501, 1), (1.0, 1), (1.501, 2), (10.4, 10)] {
            let got = clock.spikes_by(i, window * isi).expect("finite");
            assert_eq!(got, want, "{window} intervals from θ = 0 gave {got} spikes, not {want}");
        }
    }

    /// `f = √η/(π τ)`, against the stepper.
    #[test]
    fn the_theta_rate_is_root_eta_over_pi_tau() {
        let mut th = Theta { substeps: 16, ..Theta::default() };
        let i = 1500e-12;
        let eta = th.eta(i);
        let want = th.tau * PI / eta.sqrt();
        let t = spike_times(&mut th, 1e-6, 500_000, i);
        assert!(t.len() > 5);
        let iv = intervals(&t);
        let got = iv[iv.len() - 1];
        assert!((got - want).abs() / want < 1e-3, "stepped {got} s vs τπ/√η {want} s");
    }

    // ----- AdEx -----

    /// The strongest check on the joint Runge-Kutta stepper: below `V_T` the `(V, w)` system is
    /// linear and has an exact matrix-exponential solution, and the stepper must land on it. No
    /// firing pattern is involved, so a stepper bug cannot hide behind a qualitative assertion.
    #[test]
    fn adex_matches_the_exact_linear_subthreshold_solution() {
        // `v_t` pushed to +1 V so the exponential term is exp(-500) at every potential visited:
        // under 1e-200 A, which is the definition of "the linear system" being tested.
        for pattern in FiringPattern::ALL {
            let mut n = pattern.model();
            n.eif.v_t = 1.0;
            n.eif.v_peak = 2.0;
            n.eif.substeps = 8;
            n.eif.v = n.eif.e_l;
            n.w = 0.0;
            let i = 20e-12;
            let reference = n;
            let dt = 1e-6;
            let steps = 20_000u32; // 20 ms
            for _ in 0..steps {
                assert!(!n.step(dt, i), "{} should not spike sub-threshold", pattern.label());
            }
            let t = f64::from(steps) * dt;
            let (v_want, w_want) = reference.linear_subthreshold(i, t).expect("g_l + a non-zero");
            let dv = (n.eif.v - v_want).abs();
            let dw = (n.w - w_want).abs();
            assert!(
                dv < 1e-9 && dw < 1e-18,
                "{}: stepped ({}, {}) vs exact ({v_want}, {w_want})",
                pattern.label(),
                n.eif.v,
                n.w
            );
        }
    }

    /// Every named pattern must fire at its own drive, and none may produce a non-finite state.
    /// This is the gate the per-pattern signatures sit behind.
    #[test]
    fn every_named_pattern_fires_and_stays_finite() {
        for pattern in FiringPattern::ALL {
            let mut n = pattern.model();
            let t = spike_times(&mut n, 1e-5, 100_000, pattern.drive());
            assert!(t.len() >= 3, "{} produced {} spikes in 1 s", pattern.label(), t.len());
            assert!(n.w.is_finite(), "{} left w non-finite", pattern.label());
        }
    }

    /// Adapting means the intervals lengthen. Asserted on the sequence, not on a picture of it.
    #[test]
    fn the_adapting_pattern_lengthens_its_intervals() {
        let p = FiringPattern::Adapting;
        let mut n = p.model();
        let iv = intervals(&spike_times(&mut n, 1e-5, 100_000, p.drive()));
        assert!(iv.len() >= 5, "only {} intervals", iv.len());
        for w in iv.windows(2) {
            assert!(w[1] >= w[0] * 0.999, "intervals did not lengthen: {iv:?}");
        }
        assert!(iv[iv.len() - 1] > 1.3 * iv[0], "adaptation was decorative: {iv:?}");
        // And it must be adaptation, not bursting: no gap in the sorted intervals.
        let (gap, _) = widest_sorted_gap(&iv);
        assert!(gap < 2.0, "an adapting train should not be bimodal, gap ratio {gap}: {iv:?}");
    }

    /// Bursting means a BIMODAL interval distribution: a clear gap in the sorted intervals with
    /// real mass on both sides, short within a burst and long between them.
    #[test]
    fn the_bursting_pattern_has_a_bimodal_interval_distribution() {
        let p = FiringPattern::RegularBursting;
        let mut n = p.model();
        let iv = intervals(&spike_times(&mut n, 1e-5, 200_000, p.drive()));
        assert!(iv.len() >= 8, "only {} intervals: {iv:?}", iv.len());
        let (gap, at) = widest_sorted_gap(&iv);
        assert!(gap > 2.0, "no bimodal gap, widest ratio {gap}: {iv:?}");
        assert!(at >= 2 && iv.len() - at >= 2, "the gap left {at} short and {} long", iv.len() - at);
        let mut s = iv.clone();
        s.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
        let short_spread = s[at - 1] / s[0];
        let long_spread = s[s.len() - 1] / s[at];
        assert!(short_spread < gap && long_spread < gap, "the modes are not tight: {s:?}");
    }

    /// Tonic is the control: constant intervals, no gap, no drift.
    #[test]
    fn the_tonic_pattern_is_regular() {
        let p = FiringPattern::Tonic;
        let mut n = p.model();
        let iv = intervals(&spike_times(&mut n, 1e-5, 100_000, p.drive()));
        assert!(iv.len() >= 10, "only {} intervals", iv.len());
        let tail = &iv[iv.len() / 2..];
        let lo = tail.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = tail.iter().copied().fold(0.0f64, f64::max);
        assert!(hi / lo < 1.05, "tonic intervals spread from {lo} to {hi}");
        assert!(widest_sorted_gap(&iv).0 < 1.5, "tonic should not be bimodal: {iv:?}");
    }

    /// An initial burst starts fast and then settles to a regular slower train — which is what
    /// distinguishes it from regular bursting, where the fast intervals never stop coming.
    #[test]
    fn the_initial_burst_pattern_starts_fast_then_settles() {
        let p = FiringPattern::InitialBurst;
        let mut n = p.model();
        let iv = intervals(&spike_times(&mut n, 1e-5, 100_000, p.drive()));
        assert!(iv.len() >= 6, "only {} intervals: {iv:?}", iv.len());
        let tail = &iv[iv.len() / 2..];
        let lo = tail.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = tail.iter().copied().fold(0.0f64, f64::max);
        assert!(hi / lo < 1.2, "the train never settled: tail {lo} to {hi}, all {iv:?}");
        assert!(iv[0] < 0.5 * lo, "the opening interval {} was not a burst vs {lo}", iv[0]);
    }

    /// Transient means it stops, although the current does not.
    #[test]
    fn the_transient_pattern_falls_silent_under_a_current_that_stays_on() {
        let p = FiringPattern::Transient;
        let mut n = p.model();
        let t = spike_times(&mut n, 1e-5, 100_000, p.drive());
        assert!(t.len() >= 3, "only {} spikes, which is not a train", t.len());
        let last = t[t.len() - 1];
        assert!(last < 0.2, "still firing at {last} s into a 1 s run: {t:?}");
    }

    /// The transient WINDOW, and the one constant in the taxonomy that deviates from the table.
    ///
    /// `the_transient_pattern_falls_silent_under_a_current_that_stays_on` asserts the pattern at
    /// one current, which a single lucky drive can satisfy. This asserts it across the window, and
    /// asserts the two things that make "transient" mean something: above the window the same cell
    /// fires for the whole second, so the silence is not weak drive; and with the table's
    /// `b = 100 pA` the cell manages at most two spikes anywhere in the window, which is the
    /// measurement the variant's doc rests on and the reason `b` was moved.
    #[test]
    fn the_transient_pattern_is_transient_across_its_window_and_not_above_it() {
        let p = FiringPattern::Transient;
        for &pa in &[200.0, 220.0, 250.0, 270.0] {
            let mut n = p.model();
            let t = spike_times(&mut n, 1e-5, 100_000, pa * 1e-12);
            assert!(t.len() >= 3, "{pa} pA gave {} spikes, which is not a train", t.len());
            assert!(t[t.len() - 1] < 0.2, "{pa} pA was still firing at {} s", t[t.len() - 1]);
        }
        let mut open = p.model();
        let t = spike_times(&mut open, 1e-5, 100_000, 300e-12);
        assert!(t[t.len() - 1] > 0.8, "above the window it stopped at {} s", t[t.len() - 1]);
        for &pa in &[180.0, 200.0, 220.0, 250.0, 270.0] {
            let mut table = AdEx { b: 100e-12, ..p.model() };
            let t = spike_times(&mut table, 1e-5, 100_000, pa * 1e-12);
            let n = t.len();
            assert!(n <= 2, "b = 100 pA at {pa} pA gave {n} spikes; the variant's note is stale");
        }
        // And τ_w is the table's own 90 ms, which is what the doc claims and what a reader with
        // the paper will compare against.
        assert!((p.model().tau_w - 90e-3).abs() < 1e-15, "τ_w is {} s", p.model().tau_w);
    }

    /// Irregular: the interval sequence must keep varying LATE in the run, which rules out a
    /// transient and a period-1 train. It does not establish chaos, and this implementation did not
    /// compute a Lyapunov exponent.
    #[test]
    fn the_irregular_pattern_keeps_varying_late_in_the_run() {
        let p = FiringPattern::Irregular;
        let mut n = p.model();
        let iv = intervals(&spike_times(&mut n, 1e-5, 300_000, p.drive()));
        assert!(iv.len() >= 20, "only {} intervals", iv.len());
        let tail = &iv[iv.len() * 2 / 3..];
        let mean = tail.iter().sum::<f64>() / tail.len() as f64;
        let var = tail.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / tail.len() as f64;
        let cv = var.sqrt() / mean;
        assert!(cv > 0.1, "the late intervals had CV {cv}, which is a regular train: {tail:?}");
        // Irregular, not transient: it must still be firing at the end of the run.
        let mut n2 = p.model();
        let t2 = spike_times(&mut n2, 1e-5, 300_000, p.drive());
        assert!(t2[t2.len() - 1] > 2.5, "it stopped at {} s", t2[t2.len() - 1]);
    }

    // ----- Boundary behaviour -----

    /// A non-finite input is refused by name and does not reach the state.
    #[test]
    fn try_step_refuses_a_non_finite_input_without_touching_the_state() {
        let mut e = Eif::default();
        let before = e;
        // `matches!` and not `assert_eq!`: the error CARRIES the NaN it rejected, and
        // `NaN != NaN`, so the equality would fail on an error that is exactly right.
        assert!(matches!(
            try_step(&mut e, 1e-4, f64::NAN),
            Err(ModelError::NotFinite { what: "i", value }) if value.is_nan()
        ));
        assert!(matches!(
            try_step(&mut e, f64::INFINITY, 1e-9),
            Err(ModelError::NotFinite { what: "dt", .. })
        ));
        assert!(matches!(
            try_step(&mut e, -1e-4, 1e-9),
            Err(ModelError::NotPositive { what: "dt", .. })
        ));
        assert_eq!(e, before, "a refused step still moved the neuron");
        assert_eq!(try_step(&mut e, 1e-4, 400e-12), Ok(false));
    }

    /// Constructors refuse what has no model behind it, naming the field.
    #[test]
    fn the_constructors_refuse_parameters_with_no_model_behind_them() {
        assert!(matches!(
            Qif::new(0.0, -65e-3, -50e-3, 10e6, 30e-3, -65e-3, 0.0),
            Err(ModelError::NotPositive { what: "tau_m", .. })
        ));
        assert!(matches!(
            Qif::new(20e-3, -50e-3, -65e-3, 10e6, 30e-3, -65e-3, 0.0),
            Err(ModelError::Disordered { what: "v_c <= v_rest", .. })
        ));
        assert!(matches!(
            Eif::new(200e-12, 10e-9, -70e-3, -50e-3, f64::NAN, 0.0, -58e-3, 0.0),
            Err(ModelError::NotFinite { what: "delta_t", .. })
        ));
        assert!(matches!(
            Eif::new(200e-12, 10e-9, -70e-3, -50e-3, 2e-3, -60e-3, -58e-3, 0.0),
            Err(ModelError::Disordered { what: "v_peak <= v_reset", .. })
        ));
        assert!(matches!(
            AdEx::new(Eif::default(), 1e-9, -1.0, 0.0),
            Err(ModelError::NotPositive { what: "tau_w", .. })
        ));
        assert!(matches!(
            Theta::new(20e-3, 0.0, -0.25, 0.0, 15e-3, 0.0),
            Err(ModelError::NotPositive { what: "i_ref", .. })
        ));
        assert!(Qif::new(20e-3, -65e-3, -50e-3, 10e6, 30e-3, -65e-3, 2e-3).is_ok());
    }

    /// Violent drive is where a quadratic or exponential model produces an infinity and then a
    /// `NaN` that propagates silently. Every model here must survive it.
    #[test]
    fn a_violent_drive_leaves_no_model_non_finite() {
        let mut q = Qif::default();
        let mut e = Eif::default();
        let mut a = AdEx::default();
        let mut th = Theta::default();
        for k in 0..20_000 {
            let i = if k % 2 == 0 { 1e-3 } else { -1e-3 };
            q.step(1e-3, i);
            e.step(1e-3, i);
            a.step(1e-3, i);
            th.step(1e-3, i);
            assert!(q.v.is_finite(), "Qif went non-finite at {k}");
            assert!(e.v.is_finite(), "Eif went non-finite at {k}");
            assert!(a.eif.v.is_finite() && a.w.is_finite(), "AdEx went non-finite at {k}");
            assert!(th.theta.is_finite(), "Theta went non-finite at {k}");
        }
    }

    /// The module doc claims [`crate::sim::Sim::new`] REFUSES event-driven mode for every model
    /// here. That is a claim about another module, so it is checked against that module rather
    /// than asserted in prose.
    #[test]
    fn the_simulator_enforces_the_gap_property_this_module_declares() {
        use crate::net::NetBuilder;
        use crate::sim::{Mode, Sim, SimError};
        let net = || {
            let mut b = NetBuilder::new(2);
            b.connect(0, 1, 30e-3, 2).expect("in range");
            b.build()
        };
        for r in [
            Sim::new(net(), vec![Qif::default(); 2], 1e-4, Mode::EventDriven).err(),
            Sim::new(net(), vec![Eif::default(); 2], 1e-4, Mode::EventDriven).err(),
            Sim::new(net(), vec![AdEx::default(); 2], 1e-4, Mode::EventDriven).err(),
            Sim::new(net(), vec![Theta::default(); 2], 1e-4, Mode::EventDriven).err(),
        ] {
            assert_eq!(r, Some(SimError::NotExactOverGaps));
        }
        // Clocked mode is legal for all of them, so the refusal is about the MODE and not about
        // the models being unusable in a network.
        let mut sim = Sim::new(net(), vec![Qif::default(); 2], 1e-4, Mode::Clocked)
            .expect("clocked is always legal");
        let train = sim.run(20_000, &[800e-12, 0.0]);
        assert!(train.len() > 10, "only {} spikes in a 2 s clocked run", train.len());
        assert!(!train.of(1).is_empty(), "the postsynaptic cell never fired");
        // The module doc quotes this run's counts as the price of the gap property, so they are
        // pinned here rather than only asserted in prose: 23 presynaptic spikes at 800 pA, and 23
        // postsynaptic ones driven by them across the 2-tick delay. An event-driven run of the
        // same network keeps the first 23 and loses all of the second, which is what the constant
        // above refuses to let happen and therefore the one number this test cannot produce.
        assert_eq!(train.of(0).len(), 23, "the presynaptic count moved");
        assert_eq!(train.of(1).len(), 23, "the postsynaptic count moved");
        assert_eq!(train.len(), 46, "the total moved");
    }

    /// **This is why [`Qif`] declares `EXACT_OVER_GAPS` false.** A quiet interval CAN produce a
    /// spike in a bistable model, which is the assumption `sim::Sim::catch_up` rests on and the
    /// one this model breaks. Both halves are asserted: the spike happens, and a single jump
    /// across the whole interval reports it — so the flow is not what is wrong.
    #[test]
    fn a_quiet_interval_can_make_a_bistable_qif_fire() {
        let mut n = Qif::default();
        n.bump(30e-3);
        assert!(n.v > n.v_c, "the bump must land above the unstable point, got {} V", n.v);
        // Zero input from here to the end of the run, and it fires anyway.
        let t = spike_times(&mut n, 1e-5, 5_000, 0.0);
        assert_eq!(t.len(), 1, "a bistable cell left above v_c did not fire in 50 ms: {t:?}");
        assert!(t[0] > 5e-3, "it fired at {} s, too fast to be the slow escape", t[0]);
        // One jump across the same span reports the same spike. `sim` discards that bool, which is
        // where the defect lives; the model reports it, which is what this line pins.
        let mut j = Qif::default();
        j.bump(30e-3);
        assert!(j.step(50e-3, 0.0), "a single jump did not report the spike it contains");
        // A bump that stays BELOW v_c decays back to rest with no spike, so the mechanism is
        // bistability and not "any bump fires".
        let mut q = Qif::default();
        q.bump(10e-3);
        assert!(q.v < q.v_c);
        assert!(spike_times(&mut q, 1e-5, 5_000, 0.0).is_empty(), "a sub-critical bump fired");
        // And the jump has to land on `v_reset` and start the refractory period, like any other
        // spike. FOUND BY MUTATION: the default has `v_reset == v_rest`, so resetting to the wrong
        // one of the two is invisible on it and every test in this module used the default. A cell
        // whose reset differs from its rest is the only thing that can see the difference.
        let mut k = Qif { v_reset: -55e-3, ..Qif::default() };
        assert!(k.v_reset != k.v_rest, "this check is vacuous unless the two differ");
        k.bump(30e-3);
        assert!(k.step(50e-3, 0.0), "the jump did not report the spike it contains");
        assert!((k.v - k.v_reset).abs() < 1e-15, "the jump left {} V, want v_reset", k.v);
        assert!(k.refractory_left() > 0.0, "the jump did not start the refractory period");
    }

    /// [`Qif::matching_theta`] has to carry the VOLTAGE mapping too, not only the dynamics.
    ///
    /// Found by mutation: doubling `v_scale` survived every other test in this module, because
    /// `theta` and [`Qif::canonical_y`] are both blind to it — only [`Neuron::potential`] and
    /// [`Neuron::bump`] ever see it.
    #[test]
    fn the_matching_theta_reproduces_the_qifs_voltage_and_not_only_its_phase() {
        let q = Qif::default();
        for &y in &[-2.0f64, -0.5, 0.0, 0.75, 4.0] {
            let mut qif = q;
            qif.set_canonical_y(y);
            let th = qif.matching_theta();
            assert!(
                (th.potential() - qif.v).abs() < 1e-12,
                "y {y}: theta reports {} V, qif is at {} V",
                th.potential(),
                qif.v
            );
        }
        // And a synaptic kick has to move both by the same number of volts.
        let mut qif = q;
        qif.set_canonical_y(-0.2);
        let mut th = qif.matching_theta();
        for dv in [5e-3, -3e-3, 12e-3] {
            qif.bump(dv);
            th.bump(dv);
            assert!(
                (th.potential() - qif.v).abs() < 1e-12,
                "after a {dv} V bump: theta {} V vs qif {} V",
                th.potential(),
                qif.v
            );
        }
    }

    /// The corners of the public surface nothing else reaches: the error text a user is handed,
    /// the labels, `rate` against `isi`, the derived membrane constants, the argument checks on
    /// `spikes_by`, the one `Degenerate` branch in the closed-form subthreshold solution, and
    /// `reset`.
    ///
    /// The `Display` strings are part of the contract here and not decoration — the whole reason
    /// this module returns `Result` where `Lif` returns `Option` is that the error carries the
    /// rheobase, and it carries it to a person reading a line of output.
    #[test]
    fn the_public_surface_answers_at_its_edges() {
        let e = Eif::default();
        let text = format!("{}", e.isi(100e-12).expect_err("below rheobase"));
        assert!(text.contains("rheobase is"), "NoFiring reads {text}");
        assert!(text.contains(&format!("{}", e.rheobase())), "NoFiring drops the number: {text}");
        let not_finite = ModelError::NotFinite { what: "delta_t", value: f64::NAN };
        assert!(format!("{not_finite}").contains("delta_t is not finite"), "{not_finite}");
        let not_positive = ModelError::NotPositive { what: "tau_m", value: -1.0 };
        assert!(format!("{not_positive}").contains("must be strictly positive"), "{not_positive}");
        let what = "v_peak <= v_reset";
        let disordered = ModelError::Disordered { what, lower: 1.0, upper: 0.0 };
        assert!(format!("{disordered}").contains("v_peak <= v_reset"), "{disordered}");
        let degenerate = ModelError::Degenerate { what: "g_l + a" };
        assert!(format!("{degenerate}").contains("g_l + a is zero"), "{degenerate}");

        // Labels: six variants, six distinct names, each the one the doc uses.
        let labels: Vec<&str> = FiringPattern::ALL.iter().map(|p| p.label()).collect();
        let want = ["tonic", "adapting", "initial burst", "regular bursting", "transient"];
        assert_eq!(labels[..5], want, "the labels moved");
        assert_eq!(labels[5], "irregular");
        for p in FiringPattern::ALL {
            assert!(p.drive() > 0.0, "{} has a non-positive drive", p.label());
        }

        // `rate` is `1/isi` for all three closed forms, and it fails where `isi` fails.
        let q = Qif::default();
        let th = Theta::default();
        let i = 800e-12;
        assert!((q.rate(i).expect("above rheobase") - 1.0 / q.isi(i).expect("above")).abs() < 1e-9);
        assert!((th.rate(i).expect("above") - 1.0 / th.isi(i).expect("above")).abs() < 1e-9);
        let j = 400e-12;
        assert!((e.rate(j).expect("above") - 1.0 / e.isi(j).expect("above")).abs() < 1e-9);
        assert!(matches!(q.rate(f64::NAN), Err(ModelError::NotFinite { what: "i", .. })));
        assert!(matches!(th.rate(f64::NAN), Err(ModelError::NotFinite { what: "i", .. })));
        assert!(matches!(e.rate(f64::NAN), Err(ModelError::NotFinite { what: "i", .. })));

        // The derived membrane constants, against their definitions.
        assert!((e.tau_m() - e.c / e.g_l).abs() < 1e-18 && (e.tau_m() - 20e-3).abs() < 1e-15);
        assert!((e.r_m() - 1.0 / e.g_l).abs() < 1e-9 && (e.r_m() - 100e6).abs() < 1e-3);
        let lif = e.lif_limit();
        assert!((lif.tau_m - e.tau_m()).abs() < 1e-18 && lif.v_th == e.v_t && lif.v_rest == e.e_l);

        // `spikes_by` checks its arguments, and its count is a floor and not a rounding.
        assert!(matches!(th.spikes_by(1e-9, -1.0), Err(ModelError::NotPositive { what: "t", .. })));
        let nan = f64::NAN;
        assert!(matches!(th.spikes_by(nan, 1.0), Err(ModelError::NotFinite { what: "i", .. })));
        assert!(matches!(th.spikes_by(1e-9, nan), Err(ModelError::NotFinite { what: "t", .. })));
        assert_eq!(th.spikes_by(1e-9, 0.0).expect("finite"), 0, "no time is no spikes");
        let quiet = th.spikes_by(0.5 * th.rheobase(), 10.0).expect("finite");
        assert_eq!(quiet, 0, "a cell at half rheobase reported {quiet} spikes in 10 s");

        // The one denominator that can vanish in the closed-form subthreshold solution.
        let flat = AdEx::new(Eif::default(), -Eif::default().g_l, 30e-3, 0.0).expect("finite a");
        let degenerate = flat.linear_subthreshold(10e-12, 1e-3);
        assert!(matches!(degenerate, Err(ModelError::Degenerate { what: "g_l + a" })));
        let unusable = AdEx::default().linear_subthreshold(f64::NAN, 1e-3);
        assert!(matches!(unusable, Err(ModelError::NotFinite { what: "i", .. })));
        assert!((flat.w_drift(flat.eif.e_l, 0.0)).abs() < 1e-30, "w_drift at rest with w = 0 is 0");

        // `reset` puts every model back where its constructor starts it, refractory included.
        let mut q2 = Qif { v_reset: -55e-3, ..Qif::default() };
        q2.bump(30e-3);
        assert!(q2.step(50e-3, 0.0) && q2.refractory_left() > 0.0);
        q2.reset();
        assert!(q2.v == q2.v_rest && q2.refractory_left() == 0.0, "Qif::reset left {q2:?}");
        let mut a2 = AdEx::default();
        for _ in 0..2000 {
            a2.step(1e-5, 500e-12);
        }
        assert!(a2.w != 0.0, "this check needs an adaptation to clear");
        a2.reset();
        assert!(a2.w == 0.0 && a2.eif.v == a2.eif.e_l, "AdEx::reset left {a2:?}");
        let mut t2 = Theta::default();
        t2.step(1e-3, 2e-9);
        assert!(t2.theta != 0.0);
        t2.reset();
        assert!(t2.theta == 0.0, "Theta::reset left θ = {}", t2.theta);
        // And `Theta::new` folds the phase it is handed, which is the invariant the type carries.
        let phase = 7.0 * PI + 0.25;
        let wrapped = Theta::new(20e-3, 1.5e-9, -0.25, -57.5e-3, 15e-3, phase).expect("valid");
        assert!(wrapped.theta > -PI && wrapped.theta <= PI, "θ = {}", wrapped.theta);
        assert!((wrapped.theta - (0.25 - PI)).abs() < 1e-12, "θ = {}", wrapped.theta);
    }

    /// The gap property is declared per model and the simulator enforces it. Pinned here so that a
    /// later edit that makes one of these exact, or one of them approximate, has to come past a
    /// test rather than past a reviewer.
    #[test]
    fn the_declared_gap_properties_are_what_the_docs_claim() {
        const { assert!(!Qif::EXACT_OVER_GAPS, "exact flow, but bistable across a quiet gap") }
        const { assert!(!Theta::EXACT_OVER_GAPS, "Runge-Kutta on the circle does not compose") }
        const { assert!(!Eif::EXACT_OVER_GAPS, "the exponential term has no exact discrete flow") }
        const { assert!(!AdEx::EXACT_OVER_GAPS, "nor does the pair") }
        // The four consts above are a pin, not evidence. Here is the evidence for the three that
        // are false BECAUSE of the integrator: the property the constant names is that one step of
        // `k·dt` with zero input lands where `k` steps of `dt` land, and each of these MISSES it
        // by a margin no tolerance would call equal. `Qif` is the one that passes this — see
        // `the_qif_flow_composes_across_a_gap`, which is why its constant needed the other
        // argument entirely.
        let dt = 2e-3;
        let mut fine = Eif { v: -55e-3, ..Eif::default() };
        let mut coarse = fine;
        for _ in 0..10 {
            fine.step(dt, 0.0);
        }
        coarse.step(10.0 * dt, 0.0);
        let gap = (fine.v - coarse.v).abs();
        assert!(gap > 1e-9, "the Eif composed to {gap} V; EXACT_OVER_GAPS may be understated");
        let mut fine = AdEx { w: 20e-12, ..AdEx::default() };
        fine.eif.v = -55e-3;
        let mut coarse = fine;
        for _ in 0..10 {
            fine.step(dt, 0.0);
        }
        coarse.step(10.0 * dt, 0.0);
        let gap = (fine.eif.v - coarse.eif.v).abs() + (fine.w - coarse.w).abs();
        assert!(gap > 1e-12, "the AdEx composed to {gap}; EXACT_OVER_GAPS may be understated");
        let mut fine = Theta { theta: 1.0, ..Theta::default() };
        let mut coarse = fine;
        for _ in 0..10 {
            fine.step(dt, 0.0);
        }
        coarse.step(10.0 * dt, 0.0);
        let gap = (fine.theta - coarse.theta).abs();
        assert!(gap > 1e-9, "the Theta composed to {gap} rad; EXACT_OVER_GAPS may be understated");
    }
}
