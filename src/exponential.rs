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
//! Measured on a two-cell chain at 0.1 ms per tick, a clocked run gives 46 spikes and an
//! event-driven one 23 — the postsynaptic cell fires six times in the first and never in the
//! second, and the run reports success either way.
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
/// - `η < 0`: `y = a(y₀ - a·T)/(a - y₀·T)` with `a = √(-η)`, `T = tanh(a·s)`. Fixed points at
///   `y = ∓a` (stable, unstable). Divergence only from `y₀ > a`, at `s = atanh(a/y₀)/a`.
///
/// Written in the `tanh`/phase forms rather than as `w₀·exp(2ah)` because the exponential form
/// overflows to `inf` near the stable fixed point and then produces `inf/inf = NaN` at the very
/// place where the answer is simply `-a`.
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
            let h_star = (a / y0).atanh() / a;
            if h >= h_star {
                return Flow::Diverged { at: h_star };
            }
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
    /// For `η <= 0` the Riccati flow is used in its `tanh` form. The number of spikes passed over
    /// is discarded here; [`Theta::spikes_by`] counts them.
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
            // The flow at `θ = ±π` is `+2`, strictly positive for every `η`, so the phase can only
            // ever leave the interval upward. `while` rather than `if` because an absurdly coarse
            // step can carry it round more than once; the extra spikes are counted as one, which
            // is the same compromise every clocked simulator makes.
            while self.theta > PI {
                self.theta -= 2.0 * PI;
                fired = true;
            }
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
    /// `None` above rheobase, where the membrane escapes from anywhere.
    #[must_use]
    pub fn fixed_points(&self, i: f64) -> Option<(f64, f64)> {
        if !i.is_finite() {
            return None;
        }
        let k = (self.v_t - self.e_l - i / self.g_l) / self.delta_t;
        if !(k >= 1.0) {
            return None;
        }
        let y = -(-k).exp();
        let w0 = lambert_w(y, false)?;
        let wm1 = lambert_w(y, true)?;
        let stable = self.v_t + self.delta_t * (-w0 - k);
        let unstable = self.v_t + self.delta_t * (-wm1 - k);
        Some((stable, unstable))
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
    /// **Where it degrades.** Within about one part in `1e6` of rheobase the integrand is a spike
    /// narrower than [`QUADRATURE_BUDGET`] panels can resolve, and the value returned is then an
    /// under-resolved estimate rather than the exact integral. [`Eif::saddle_node_rate`] is the
    /// right tool in that regime and is exact in the limit the quadrature is failing in, which is
    /// a convenient division of labour rather than a coincidence: both are consequences of the
    /// bottleneck dominating.
    ///
    /// # Errors
    ///
    /// [`ModelError::NotFinite`] for a non-finite `i`, or [`ModelError::NoFiring`] at or below
    /// rheobase, where the membrane settles on the stable fixed point instead.
    pub fn isi(&self, i: f64) -> Result<f64, ModelError> {
        finite("i", i)?;
        let rheobase = self.rheobase();
        if i <= rheobase {
            return Err(ModelError::NoFiring { i, rheobase });
        }
        let f = |v: f64| self.c / self.current(v, i);
        let top = self.v_peak.min(self.v_t + EXP_ARG_LIMIT * self.delta_t);
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
    /// **This variant's `τ_w` and `b` were changed from what was transcribed.** The transcribed
    /// pair (90 ms, 100 pA) fires **once** at every current in the transient window and then stops,
    /// which is a degenerate corner of the pattern rather than the published figure's short train;
    /// the window itself closes at 280 pA, above which the cell fires forever. 400 ms and 30 pA at
    /// 220 pA give five spikes and then silence. The transcription is the suspect party here, not
    /// the model — a reader with the paper should replace these three numbers and this note.
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
            Self::Transient => (100e-12, 10e-9, -65e-3, 10e-9, 400e-3, 30e-12, -47e-3),
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

/// Lambert `W` on its two real branches, restricted to `y ∈ [-1/e, 0)`, which is the only range
/// [`Eif::fixed_points`] needs.
///
/// Halley iteration from the branch-point series `W ≈ -1 ± p - p²/3 ± 11p³/72` with
/// `p = √(2(e·y + 1))`. Near the branch point the series is already accurate to `O(p⁴)`, so for
/// `p < 1e-4` it is returned unrefined — which also sidesteps Halley's `2(w+1)` denominator
/// vanishing at `w = -1`. `None` outside the range, where the branches are not both real.
fn lambert_w(y: f64, lower_branch: bool) -> Option<f64> {
    let e_inv = -(-1.0f64).exp();
    if !y.is_finite() || y < e_inv || y >= 0.0 {
        return None;
    }
    let p = (2.0 * (std::f64::consts::E * y + 1.0)).max(0.0).sqrt();
    let series = if lower_branch {
        -1.0 - p - p * p / 3.0 - 11.0 * p * p * p / 72.0
    } else {
        -1.0 + p - p * p / 3.0 + 11.0 * p * p * p / 72.0
    };
    if p < 1e-4 {
        return Some(series);
    }
    let mut w = series;
    for _ in 0..80 {
        let ew = w.exp();
        let f = w * ew - y;
        if f == 0.0 {
            break;
        }
        let d = ew * (w + 1.0) - (w + 2.0) * f / (2.0 * w + 2.0);
        if d == 0.0 || !d.is_finite() {
            break;
        }
        let step = f / d;
        w -= step;
        if step.abs() <= 1e-16 * w.abs() {
            break;
        }
    }
    Some(w)
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
        AdEx, Eif, FiringPattern, Flow, ModelError, Qif, Theta, canonical_flow, lambert_w, try_step,
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

    /// `W(-ln2/2)` has two exact values: `-ln 2` on the principal branch and `-2 ln 2` on the
    /// lower one, because both satisfy `w·e^w = -ln2/2`. Two exact targets from one input, which
    /// is what makes this a real check on the branch selection rather than on the arithmetic.
    #[test]
    fn lambert_w_hits_both_of_its_exact_branches() {
        let ln2 = 2.0f64.ln();
        let y = -ln2 / 2.0;
        let w0 = lambert_w(y, false).expect("in range");
        let wm1 = lambert_w(y, true).expect("in range");
        assert!((w0 + ln2).abs() < 1e-14, "principal branch {w0} vs {}", -ln2);
        assert!((wm1 + 2.0 * ln2).abs() < 1e-13, "lower branch {wm1} vs {}", -2.0 * ln2);
        // At the branch point both branches are -1 exactly.
        let bp = -(-1.0f64).exp();
        assert!((lambert_w(bp, false).expect("in range") + 1.0).abs() < 1e-7);
        assert!(lambert_w(0.0, false).is_none(), "0 is outside the branch-pair range");
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

    /// Check (a), second leg, at the tolerance the brief asks for. `Theta::exact_theta_after` is
    /// written in the circle's own coordinates — a phase rotation and an `atan2` — while
    /// `canonical_flow` is written as a Riccati solution in `y`. Agreement here is floating-point,
    /// not discretisation.
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
        // A bump of any size is a rotation, never an escape.
        for dv in [1.0, -1.0, 1e6, -1e6] {
            th.bump(dv);
            assert!(th.theta.abs() <= PI, "a {dv} V bump left the circle at θ = {}", th.theta);
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

    /// The gap property is declared per model and the simulator enforces it. Pinned here so that a
    /// later edit that makes one of these exact, or one of them approximate, has to come past a
    /// test rather than past a reviewer.
    #[test]
    fn the_declared_gap_properties_are_what_the_docs_claim() {
        const { assert!(!Qif::EXACT_OVER_GAPS, "exact flow, but bistable across a quiet gap") }
        const { assert!(!Theta::EXACT_OVER_GAPS, "Runge-Kutta on the circle does not compose") }
        const { assert!(!Eif::EXACT_OVER_GAPS, "the exponential term has no exact discrete flow") }
        const { assert!(!AdEx::EXACT_OVER_GAPS, "nor does the pair") }
    }
}
