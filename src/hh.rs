//! Hodgkin-Huxley: the action potential derived rather than imposed.
//!
//! Every other neuron model in this crate *asserts* a spike. [`crate::neuron::Lif`] watches its
//! membrane cross a threshold you supplied and writes a reset value you supplied; the spike is a
//! bookkeeping event, and "threshold" and "refractory period" are parameters. This module contains
//! the model where none of that is true. Hodgkin and Huxley wrote down four coupled ordinary
//! differential equations fitted to voltage-clamp measurements of the squid giant axon, and the
//! action potential — its threshold, its 100 mV overshoot, its width, its after-hyperpolarisation
//! and its refractory period — comes out as a *consequence*. Nothing in the equations below knows
//! what a spike is.
//!
//! **Primary source:** A. L. Hodgkin and A. F. Huxley, "A quantitative description of membrane
//! current and its application to conduction and excitation in nerve", *Journal of Physiology*
//! 117:500-544, 1952. Nobel Prize in Physiology or Medicine, 1963.
//!
//! # The mechanism, in one paragraph
//!
//! A patch of membrane is a capacitor with resistors across it. The capacitor is the lipid bilayer;
//! the resistors are ion channels, each selective for one ion and each pulling the membrane toward
//! that ion's Nernst reversal potential. What Hodgkin and Huxley discovered is that two of those
//! conductances **depend on the voltage across them, and on how long it has been there**. Sodium
//! conductance rises steeply and fast with depolarisation (activation, the `m` gate), then falls
//! again if the depolarisation persists (inactivation, the `h` gate). Potassium conductance rises
//! more slowly and does not inactivate (the `n` gate). Positive feedback — depolarisation opens
//! sodium channels, sodium influx depolarises further — is what makes the upstroke explosive and
//! what creates an apparent threshold out of a system with no threshold in it. The two slow
//! processes, sodium inactivation and potassium activation, are what terminate the spike and what
//! make the membrane refractory afterwards. The whole model is:
//!
//! ```text
//! C dV/dt = I_ext - gNa·m³h·(V - ENa) - gK·n⁴·(V - EK) - gL·(V - EL)
//! dm/dt = alpha_m(V)·(1-m) - beta_m(V)·m       (and likewise h and n)
//! ```
//!
//! The exponents 3 and 4 are not derived from anything, and the paper does not claim they are the
//! smallest that would do. It reaches them from curve shape: a first-order variable cannot reproduce
//! the inflexion at the start of the potassium rise, for which "a third- or fourth-order equation is
//! needed", and supposing the conductance proportional to the fourth power of a first-order variable
//! is the "useful simplification" that produces it — `(1 - e^-t)^4` rises with an inflexion while
//! `e^-4t` falls as a simple exponential. Sodium gets the cube by "a similar assumption using a cube
//! instead of a fourth power", plus an inactivation term. Minimality is not argued in the paper and
//! is not claimed here.
//!
//! # What it buys, and what it costs
//!
//! **Buys:** the spike shape is a prediction, not a parameter. Threshold emerges, and sharply: a
//! 0.5 ms stimulus 1% below the critical amplitude leaves the membrane at -57.7 mV and 1% above it
//! reaches +34.8 mV, a 92 mV swing across a 2% change in the input. Refractoriness emerges —
//! [`HodgkinHuxley`] has no refractory field at all, and
//! [`crate::neuron::Neuron::refractory_left`] returns `0.0` for it, yet
//! an identical second stimulus 10 ms later fails while the same stimulus 12 ms later succeeds.
//! Class 2 excitability emerges: the firing rate does not rise from zero at the repetitive-firing
//! threshold, it switches on at a measured 52 Hz, which no integrate-and-fire model reproduces.
//! Depolarisation block emerges: drive it above about 155 µA/cm² and the limit cycle collapses onto
//! a depolarised fixed point and the cell falls silent.
//!
//! **Costs:** four state variables instead of one, about ten transcendental function evaluations per
//! step against [`crate::neuron::Lif`]'s one, and a stiff system that needs a step of order 25 µs
//! where a leaky integrator is content with 1 ms. That is a few hundred times the arithmetic per
//! millisecond of simulated time — an operation count, not a wall-clock benchmark, and this crate
//! has not measured the wall clock.
//!
//! **That cost is why the neuromorphic hardware runs something else.** `TrueNorth`'s neuron is a
//! fixed-function integrate-and-fire; Loihi's is a leaky integrate-and-fire with configurable
//! traces. This review did not locate a fixed-function neuromorphic ASIC that implements
//! Hodgkin-Huxley. The programmable machines are a real exception and worth stating rather than
//! eliding: `SpiNNaker` is general-purpose ARM cores, so its software stack can and does run
//! conductance-based cell models — at a cost in neurons per core, which is the same trade in a
//! different currency. The model that explains the biology is not the model the fixed silicon runs,
//! and a student who understands why is most of the way to understanding what neuromorphic
//! engineering actually chose to keep.
//!
//! # Sign convention — read this before comparing anything to the paper
//!
//! **This module uses the modern convention**: `V` is the membrane potential, inside minus outside,
//! in millivolts, resting near -65 mV, with depolarisation POSITIVE. The 1952 paper uses the
//! opposite: its `V` is the *displacement from the resting potential* with depolarisation
//! **negative**. Every textbook reproduction of the model has quietly performed the substitution
//!
//! ```text
//! V_paper = -(V_modern + 65)
//! ```
//!
//! and this module performs it in the open: [`rates`] gives the modern rate functions, [`rates_1952`]
//! transcribes the six equations exactly as the paper prints them in its own frame, and the test
//! `the_paper_s_own_rate_functions_transform_into_the_modern_ones` checks that the substitution maps
//! one onto the other across 641 voltages — bit for bit, as it turns out, because on that grid the
//! substitution is exact in floating point rather than merely accurate.
//!
//! **What that test can and cannot show.** It pins every constant in one frame against a
//! differently-written constant in the other, so a transposed digit on either side moves one of them
//! and fails; that is worth having. What it cannot do is tell a transcription of the paper from a
//! transcription of somebody's copy of the paper, because two transcriptions of the same wrong
//! source agree with each other perfectly. The check that does that is the citation on
//! [`rates_1952`] — six equation numbers and a page, verifiable against the source in an afternoon —
//! and this module's citation named the wrong equations until an audit went and read the paper. The
//! same test also pins the sign convention itself, by requiring the paper's `V = 0`, `V = -115` and
//! `V = +12` to be the modern -65, +50 and -77 mV.
//!
//! # Units: the paper's frame inside, SI at the boundary
//!
//! Inside [`HodgkinHuxley`] everything is in the paper's units — millivolts, milliseconds,
//! µA/cm², mS/cm², µF/cm² — because they are what a reader can check line by line against the
//! source, and because those units are dimensionally self-consistent (µF·mV/ms is exactly µA, and
//! mS·mV is exactly µA). [`HodgkinHuxley::advance`] is the entry point in that frame.
//!
//! [`crate::neuron::Neuron::step`] is the SI boundary: seconds, amperes, volts, exactly like every
//! other model in the crate. Converting a current density into an ampere needs a membrane area, so
//! [`HodgkinHuxley::area_cm2`] exists and its default is a stated convention rather than a
//! measurement — see its doc.
//!
//! # What "verified" means here
//!
//! There is no closed-form solution of the Hodgkin-Huxley equations; there is not even a closed-form
//! expression for the threshold. So this module is checked against limits, invariants, exactly
//! computable special cases and published figures:
//!
//! - the paper's own rate functions, under the sign substitution (exact, to 1e-12);
//! - the removable singularities in `alpha_m` and `alpha_n` against their L'Hôpital limits;
//! - the resting potential as a genuine fixed point: the default cell drifts 2.8e-4 mV in 500 ms,
//!   which is the distance between the -65.0 mV it is set to and the -64.99972 mV where the three
//!   currents actually cancel;
//! - `V` confined to `[EK, ENa]` under zero input **for any step size**, which is a provable
//!   property of the integrator used here and not an empirical observation;
//! - gating variables confined to `[0,1]` under violent drive, likewise provable;
//! - spike peak and width against the published range, with the measured values in the doc;
//! - the refractory period emerging from `h` and `n` with nothing imposed, and the threshold
//!   emerging as a 92 mV response gap across a 2% change in stimulus;
//! - convergence in the step size at a **measured** rate of 3.93x, 4.02x and 4.16x over the four
//!   halvings from `dt = 0.05 ms` to `dt = 0.003125 ms` — near second order, not exactly it, and
//!   stated as the three numbers rather than as a range that excluded one of them — with a stated
//!   bound: below `dt = 0.0125 ms` the spike time moves by under 2 µs in total;
//! - an independent integrator ([`Integrator::Rk4`]) agreeing on the spike time.
//!
//! # Quickstart
//!
//! ```
//! use ferromorphic::hh::HodgkinHuxley;
//!
//! let mut cell = HodgkinHuxley::default();
//! // 10 µA/cm² is above rheobase. Step in 25 µs and watch for the spike.
//! let mut peak: f64 = cell.v;
//! for _ in 0..400 {
//!     cell.advance(0.025, 10.0)?;
//!     peak = peak.max(cell.v);
//! }
//! assert!(peak > 30.0, "peak was {peak} mV");
//!
//! // Rest is not "nothing happening": two large opposed currents cancel to a millionth of an amp
//! // per square centimetre, and the whole model is about what happens when they stop cancelling.
//! let c = HodgkinHuxley::default().currents();
//! assert!(c.i_na < 0.0, "sodium is INWARD at rest");
//! assert!(c.i_k > 0.0, "potassium is OUTWARD at rest");
//! assert!(c.i_ion.abs() < 1e-3, "and at rest they cancel: {}", c.i_ion);
//! # Ok::<(), ferromorphic::hh::HhError>(())
//! ```

use crate::neuron::Neuron;
use core::fmt;

/// `x / (1 - exp(-x))`, with the removable singularity at `x = 0` replaced by its limit.
///
/// Both `alpha_m` and `alpha_n` are written in the literature as a ratio that is `0/0` at one
/// particular voltage — -40 mV and -55 mV respectively — and a direct transcription returns `NaN`
/// there. The function is perfectly smooth; only the expression is singular. By L'Hôpital the limit
/// at `x = 0` is 1, and the Taylor expansion is `1 + x/2 + x²/12 - x⁴/720 + …`, so the series branch
/// below is accurate to well past f64 precision for `|x| < 1e-8` while the `exp_m1` branch is
/// accurate everywhere else.
///
/// Silently returning `NaN` at exactly one voltage is a classic and nasty failure: a simulation
/// finds the singular voltage roughly once in every few million steps, produces `NaN`, and reports
/// zero spikes for the rest of the run.
fn exprel_recip(x: f64) -> f64 {
    if x.abs() < 1e-8 {
        1.0 + 0.5 * x + x * x / 12.0
    } else {
        x / -((-x).exp_m1())
    }
}

/// How far outside `[0,1]` a gating variable may sit and still be called an occupancy.
///
/// 1e-9 is about seven orders of magnitude above the rounding a single exponential update can
/// produce and eight below any excursion a failing integrator produces — forward Euler's first
/// illegal `m` in this module's own demonstration is 4.3, not 1.000000001. The number is a property
/// of the guard rather than of the model, so it is pinned directly by
/// `the_state_guard_is_a_tolerance_and_this_is_exactly_where_it_sits` and not left to be inferred
/// from a run.
const GATE_SLACK: f64 = 1e-9;

/// Whether `x` is a legal gating occupancy: inside `[0,1]` up to [`GATE_SLACK`].
///
/// `false` for `NaN`, because every comparison against `NaN` is false and this is written as a range
/// containment rather than as a pair of negated comparisons. Shared by both models so their guards
/// cannot drift apart; they had, by eight orders of magnitude.
fn gate_is_legal(x: f64) -> bool {
    (-GATE_SLACK..=1.0 + GATE_SLACK).contains(&x)
}

/// The spike detector, shared by both models: an **upward crossing** of `level`, re-armed below
/// `reset`.
///
/// `prev` is the potential before the substep and `now` the potential after it, and both are needed.
/// A level test on `now` alone reports a spike for a membrane that was already above `level` when
/// the caller handed it over — [`HodgkinHuxley::at`] above the level, a large
/// [`crate::neuron::Neuron::bump`], a state reconstructed field by field — while the potential is on
/// its way **down**: `HodgkinHuxley::at(10.0)` reported `fired = true` on its first 1 µs step, on
/// which the membrane fell from +10 to +7.68 mV, and `at(1.0)` reported one while falling clean
/// through the level to -0.91 mV. A trajectory that starts above the level reports nothing until it
/// has fallen below `reset` and come back up through `level`, which is what `armed` being public is
/// for.
fn detect_crossing(armed: &mut bool, prev: f64, now: f64, level: f64, reset: f64) -> bool {
    if *armed && prev < level && now >= level {
        *armed = false;
        return true;
    }
    if !*armed && now <= reset {
        *armed = true;
    }
    false
}

/// Bisect `f` for a sign change, to machine precision, on the `[-90, -40]` mV resting bracket.
///
/// Shared by [`HodgkinHuxley::rest_potential_mv`] and [`ReducedHh::rest_potential_mv`] so that the
/// two cannot say different things: the reduced copy of this loop was missing the finiteness guard,
/// and since every comparison against `NaN` is false, a cell with one `NaN` parameter ran the
/// bisection on `NaN`, took the same branch two hundred times and returned the bracket endpoint
/// **-40.0 mV as a resting potential** — which [`ReducedHh::default`] and
/// [`crate::neuron::Neuron::reset`] would then have sat a cell at.
///
/// `None` unless both endpoints are finite and their values straddle zero: a root this bracket
/// cannot see is reported as absent rather than as an endpoint.
///
/// `f(lo)` is carried rather than recomputed, and the loop stops when the bracket reaches one ulp,
/// where every further iteration is a fixed point of the update and cannot move the answer. Both are
/// exact: the returned root is bit-identical to the 200-iteration, 400-evaluation version, at about
/// a seventh of the transcendental evaluations — which `ReducedHh::default` pays on construction and
/// `Neuron::reset` pays again.
fn bisect_rest(f: impl Fn(f64) -> f64) -> Option<f64> {
    let (mut lo, mut hi) = (-90.0_f64, -40.0_f64);
    let (mut flo, fhi) = (f(lo), f(hi));
    if !flo.is_finite() || !fhi.is_finite() || flo * fhi > 0.0 {
        return None;
    }
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if mid == lo || mid == hi {
            break;
        }
        let fmid = f(mid);
        if flo * fmid <= 0.0 {
            hi = mid;
        } else {
            lo = mid;
            flo = fmid;
        }
    }
    Some(0.5 * (lo + hi))
}

/// The six voltage-dependent transition rates, all in **reciprocal milliseconds**.
///
/// `alpha_x` is the opening rate of gate `x` and `beta_x` its closing rate, both functions of the
/// membrane potential alone. Every one is non-negative for every finite voltage, which is what makes
/// `alpha/(alpha+beta)` a number in `[0,1]` and therefore what makes the gating variables
/// probabilities.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rates {
    /// Sodium activation opening rate, 1/ms. Steep and fast: about 1 /ms at -40 mV, about 9 /ms at
    /// +50 mV, so the `m` gate's time constant falls to roughly 0.1 ms during the upstroke.
    pub alpha_m: f64,
    /// Sodium activation closing rate, 1/ms. 4.0 /ms at rest, falling exponentially with
    /// depolarisation.
    pub beta_m: f64,
    /// Sodium inactivation opening rate, 1/ms — the rate at which inactivation is *removed*. 0.07
    /// /ms at rest, which is 50 times slower than `alpha_m` and is the whole reason the spike has a
    /// shape.
    pub alpha_h: f64,
    /// Sodium inactivation closing rate, 1/ms — the rate at which the channel inactivates.
    pub beta_h: f64,
    /// Potassium activation opening rate, 1/ms. About 0.058 /ms at rest, an order of magnitude below
    /// `alpha_m`, which is why potassium arrives late and repolarises rather than competing with the
    /// upstroke.
    pub alpha_n: f64,
    /// Potassium activation closing rate, 1/ms.
    pub beta_n: f64,
}

impl Rates {
    /// The steady-state gate values `alpha/(alpha+beta)` this voltage would reach if held forever.
    ///
    /// Each component is in `[0,1]` by construction, since both rates are non-negative.
    #[must_use]
    pub fn steady_state(&self) -> Gates {
        Gates {
            m: self.alpha_m / (self.alpha_m + self.beta_m),
            h: self.alpha_h / (self.alpha_h + self.beta_h),
            n: self.alpha_n / (self.alpha_n + self.beta_n),
        }
    }

    /// The gate time constants `1/(alpha+beta)`, in **milliseconds**.
    ///
    /// The separation between them is the model's engine: at rest this implementation computes
    /// `tau_m = 0.2368 ms`, `tau_h = 8.5160 ms` and `tau_n = 5.4586 ms`, so sodium activation is
    /// effectively instantaneous on the timescale over which the other two move — a factor of
    /// **23.1 against `tau_n` and 36.0 against `tau_h`** at rest, and more when depolarised, since
    /// `tau_m` never exceeds 0.501 ms at any voltage. That separation is what [`ReducedHh`] exploits,
    /// and `the_gate_time_constants_are_separated_and_peak_where_this_implementation_says` is where
    /// every number in this paragraph and in [`Taus`] is measured rather than remembered.
    #[must_use]
    pub fn time_constants_ms(&self) -> Taus {
        Taus {
            tau_m: 1.0 / (self.alpha_m + self.beta_m),
            tau_h: 1.0 / (self.alpha_h + self.beta_h),
            tau_n: 1.0 / (self.alpha_n + self.beta_n),
        }
    }
}

/// The three gating variables, each a dimensionless occupancy in `[0,1]`.
///
/// Read them as probabilities that an independent subunit is in its permissive state: the channel
/// conducts when all of its subunits are permissive, which is where `m³h` and `n⁴` come from.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Gates {
    /// Sodium activation, `[0,1]`. 0.0529 at rest; this implementation measures a maximum of 0.994
    /// during a spike, so the sodium conductance never approaches its nominal `g_na`.
    pub m: f64,
    /// Sodium inactivation, `[0,1]`, where **1 means not inactivated**. 0.5961 at rest, falling to a
    /// measured 0.077 during a spike; its slow recovery is half of the refractory period.
    pub h: f64,
    /// Potassium activation, `[0,1]`. 0.3177 at rest, rising to a measured 0.770 during
    /// repolarisation.
    pub n: f64,
}

/// Gate time constants, all in **milliseconds**.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Taus {
    /// Time constant of sodium activation, ms. Sub-millisecond at every voltage: 0.237 ms at rest,
    /// 0.111 ms at +50 mV.
    pub tau_m: f64,
    /// Time constant of sodium inactivation, ms. Swept at 0.1 µV resolution this implementation
    /// measures the peak at **8.582 ms, at -66.81 mV** — just below rest, not at the -50 mV a
    /// hand-drawn figure suggests, where it is already down to 4.641 ms.
    pub tau_h: f64,
    /// Time constant of potassium activation, ms. Same sweep: the peak is **5.792 ms at -77.17 mV**,
    /// and at -55 mV — which is `alpha_n`'s singular voltage, not its slowest one — it is 4.755 ms.
    pub tau_n: f64,
}

/// The instantaneous ionic conductances of the patch, in **mS/cm²**.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Conductances {
    /// Sodium conductance `gNa·m³h`, mS/cm². Ranges over three and a half orders of magnitude within
    /// one spike: 0.0106 at rest, and a measured 33.2 at its peak, 0.11 ms after the voltage peak.
    pub g_na: f64,
    /// Potassium conductance `gK·n⁴`, mS/cm². 0.367 at rest, and a measured 12.7 at its peak, which
    /// falls **1.48 ms after** the sodium peak. That delay is the spike.
    pub g_k: f64,
    /// Leak conductance, mS/cm². Constant — it is the one conductance in the model that does not
    /// gate, and it is what sets the membrane's behaviour when the others are shut.
    pub g_leak: f64,
    /// Their sum, mS/cm². Divided into `c_m` it gives the instantaneous membrane time constant, which
    /// this implementation measures swinging from 1.477 ms at rest to 0.027 ms at the peak of the
    /// sodium conductance — a factor of 55. See [`HodgkinHuxley::membrane_time_constant_ms`].
    pub total: f64,
}

/// The instantaneous ionic currents of the patch, in **µA/cm²**.
///
/// **Sign convention: positive is OUTWARD.** Each current is written `g·(V - E)`, so it is positive
/// when the membrane sits above that ion's reversal potential and the ion therefore carries positive
/// charge out of the cell. A depolarising current is negative. The membrane equation is
/// `C dV/dt = I_ext - i_ion`, so a negative `i_ion` drives `V` up.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Currents {
    /// Sodium current, µA/cm². **Inward (negative) at all physiological voltages** because `ENa` is
    /// +50 mV, and enormous during the upstroke — this is the current that charges the membrane.
    pub i_na: f64,
    /// Potassium current, µA/cm². Outward (positive) above `EK` = -77 mV, which is everywhere the
    /// membrane normally goes. This is the current that repolarises.
    pub i_k: f64,
    /// Leak current, µA/cm². Small and outward at rest, and the term that makes the resting
    /// potential a stable fixed point rather than a mere crossing.
    pub i_leak: f64,
    /// Their sum, µA/cm². Zero at the resting potential, by the definition of "resting".
    pub i_ion: f64,
}

/// Which numerical integrator [`HodgkinHuxley::advance`] uses.
///
/// This is exposed, rather than hidden as an implementation detail, because **the choice of
/// integrator changes which properties the model has**, and that is a lesson rather than a
/// configuration option. See the variant docs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Integrator {
    /// Exponential Euler, also called Rush-Larsen after Rush and Larsen, *IEEE Transactions on
    /// Biomedical Engineering* 25:389-392, 1978. **The default, and the only variant with
    /// guarantees.**
    ///
    /// Each gate is advanced by freezing the voltage over the step and solving its linear equation
    /// exactly: `x ← x_inf + (x - x_inf)·exp(-dt/tau)`. Because `x_inf ∈ [0,1]` and
    /// `exp(-dt/tau) ∈ (0,1]`, the update is a convex combination of two numbers in `[0,1]`, so
    /// **a gate that starts inside the unit interval provably cannot leave it, at any step size**.
    /// The gate fields are public and a caller can write an illegal value into one directly; that is
    /// what [`HodgkinHuxley::advance`]'s state check is for. The voltage is advanced
    /// the same way, by freezing the conductances: `V ← V_inf + (V - V_inf)·exp(-dt·g_total/C)`,
    /// where `V_inf` is a conductance-weighted average of the reversal potentials plus `I/g_total`.
    /// With zero input that is a convex combination of `ENa`, `EK` and `EL`, so **`V` provably
    /// cannot leave `[EK, ENa]` at any step size**. It is unconditionally stable, which is the right
    /// trade for a stiff system whose stiffness varies 55-fold within one spike.
    ///
    /// The scheme is formally first order. **Measured on this problem it behaves as second order**:
    /// the four halvings from `dt = 0.05 ms` to `dt = 0.003125 ms` shrink the successive changes in
    /// the interpolated spike time by 3.93x, 4.02x and 4.16x, which is what
    /// `halving_the_step_converges_at_the_measured_rate` records and pins to 1%. This module reports
    /// the
    /// measurement rather than the textbook order, because the ordering used here — gates at the old
    /// voltage, then the voltage with the new gates — is a splitting whose order this implementation
    /// has not derived.
    ExponentialEuler,
    /// Forward Euler on all four variables. **Present as a demonstration of failure**, and the test
    /// `forward_euler_leaves_the_unit_interval_where_the_exponential_update_cannot` is that
    /// demonstration: at +50 mV the `m` gate's time constant is 0.111 ms, so a single 0.5 ms forward
    /// step drives `m` to a measured **4.315** — an occupancy probability above four — after which
    /// `m³h` is nonsense and the membrane reaches `NaN` within a few more steps.
    /// Accurate enough below about 10 µs, which is why so much published code gets away with it.
    ForwardEuler,
    /// Classical fourth-order Runge-Kutta. Fourth-order accurate and used here as an **independent**
    /// check on the default: two integrators sharing no update rule that agree on a spike time to a
    /// measured **7 nanoseconds** are evidence about the equations rather than about either
    /// integrator.
    ///
    /// It carries no invariant — the gates can leave `[0,1]` under a large step exactly as forward
    /// Euler's can — and it costs four rate evaluations per step against the default's one.
    Rk4,
}

/// What [`HodgkinHuxley::advance`] refuses, and why.
///
/// Every variant names the quantity that was wrong rather than reporting a generic failure, because
/// the caller is usually a sweep and the sweep's job is to print which parameter broke.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HhError {
    /// The time step was `NaN` or an infinity. Refused at the boundary: a non-finite step would
    /// turn the whole state non-finite in one call and the run would then report zero spikes
    /// forever without anything having failed loudly.
    NonFiniteStep,
    /// The time step was zero or negative. Zero is refused rather than treated as a no-op because a
    /// loop that steps by zero never terminates and a caller who passed it wants to know.
    NonPositiveStep,
    /// The injected current was `NaN` or an infinity.
    NonFiniteCurrent,
    /// The state left its legal region during the step: a gate outside `[0,1]` by more than
    /// `GATE_SLACK`, or a non-finite potential.
    ///
    /// [`Integrator::ExponentialEuler`] cannot **produce** this from a legal state at any step size,
    /// which is the point of that variant's guarantees, and the other two can at a large step. It is
    /// not the same as being unreachable under the default: every state field is public, so a gate
    /// written outside `[0,1]` by hand is reported here rather than integrated; and a cell whose
    /// three conductances sum to zero has a `v_inf` of `0/0`, which is a non-finite potential the
    /// exponential update produces from a state that was perfectly legal.
    Diverged,
}

impl fmt::Display for HhError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::NonFiniteStep => "time step was not finite",
            Self::NonPositiveStep => "time step was zero or negative",
            Self::NonFiniteCurrent => "injected current was not finite",
            Self::Diverged => {
                "state left its legal region: a gating variable outside [0,1] or a non-finite \
                 potential — reduce the step, or use Integrator::ExponentialEuler, which cannot"
            }
        };
        f.write_str(s)
    }
}

impl std::error::Error for HhError {}

/// The measured shape of one action potential, all in the model's own frame.
///
/// Produced by [`HodgkinHuxley::spike_shape`]. Every field is a measurement off a simulated
/// trajectory at the step size you passed, not a closed form — the Hodgkin-Huxley equations have
/// none — so treat the numbers as accurate to about the step size.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpikeShape {
    /// Highest membrane potential reached, mV. The published squid overshoot is about +40 mV above a
    /// -65 mV rest; this implementation measures **+40.27 mV** under a sustained 10 µA/cm² and
    /// +40.32 mV after a brief 30 µA/cm² pulse.
    pub peak_mv: f64,
    /// Time of `peak_mv` relative to the start of the run, ms — the time of the **sample** that was
    /// the peak, which is the end of the step that produced it and is therefore a multiple of
    /// `dt_ms`. It was reported one step early until an audit checked it against a hand-run replay.
    pub peak_time_ms: f64,
    /// Time the potential spent continuously above `level_mv` on this spike, ms. The conventional
    /// "spike width" for squid at 6.3 °C is of order 1-2 ms; this implementation measures **1.165 ms**
    /// above 0 mV under a sustained 10 µA/cm², with a maximum `dV/dt` on the upstroke of 310 mV/ms.
    pub width_ms: f64,
    /// The level the width was measured at, mV, carried so a figure can state its own definition.
    /// A spike width without its level is not a number.
    pub level_mv: f64,
    /// Time of the first sample at or above `level_mv`, ms. The crossing itself lies somewhere in
    /// the `dt_ms` before it, so this is an upper bound on the crossing time and never an earlier
    /// one; `width_ms` is a difference of two such times and is unaffected by the convention.
    pub upstroke_time_ms: f64,
    /// The lowest potential reached after the spike fell back through `level_mv`, mV — the
    /// after-hyperpolarisation. Measured at -75.1 mV under a sustained 10 µA/cm² and -76.2 mV after
    /// a brief pulse. It goes below rest because `n` is still elevated, holding potassium
    /// conductance high; this is the *other* half of the refractory period, and it is why the
    /// membrane is harder to fire than usual even after `h` has partly recovered.
    ///
    /// `None` when the run ended on the downstroke, so that no sample after the spike exists. That
    /// is a refusal rather than a fallback: reporting the peak, or the last sample, would be a
    /// number in the right units that measured something else.
    pub after_hyperpolarisation_mv: Option<f64>,
}

/// The four-variable squid giant axon model of Hodgkin and Huxley (1952).
///
/// State is `(v, m, h, n)`; everything else is a parameter. The defaults are the paper's own
/// maximal conductances and the modern reversal potentials, at the paper's 6.3 °C. **There is no
/// threshold field and no refractory field** — both of those behaviours emerge, and the module doc
/// explains how.
///
/// # Temperature
///
/// The 1952 measurements were made at 6.3 °C and the rate functions here are the 6.3 °C ones. Warmer
/// preparations are usually modelled by multiplying every rate by `Q10^((T-6.3)/10)` with `Q10 = 3`,
/// which at mammalian body temperature is a factor of about 24 and turns the 1 ms spike into a
/// 0.1 ms one.
///
/// **The paper states that scaling itself**, on the page that carries the constants this module
/// transcribed: "The expressions for the α's and β's are appropriate to a temperature of 6.3 °C; for
/// other temperatures they must be scaled with a `Q10` of 3. The constants in eqn. (26) are taken as
/// independent of temperature." So the authoritative set is `Q10 = 3` on all six rates and none on
/// the conductances or the capacitance, and an earlier version of this doc was wrong to say no such
/// set could be located.
///
/// **It is still not implemented here**, and the reason is a choice rather than an absence: a
/// temperature multiplier applied inside the rate functions is invisible at the call site, and it is
/// exactly the kind of hidden factor that makes two people's "Hodgkin-Huxley" disagree by an order
/// of magnitude in spike width with both of them reading the same source. Scale the rates yourself,
/// in the open — multiply all six by `3f64.powf((t_celsius - 6.3) / 10.0)` and leave `g_na`, `g_k`,
/// `g_leak` and `c_m` alone — if you need a temperature other than the paper's.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HodgkinHuxley {
    /// Specific membrane capacitance, µF/cm². The paper's value is 1.0, and it is the one parameter
    /// of the model that is very nearly universal across cell types and species.
    pub c_m: f64,
    /// Maximal sodium conductance, mS/cm². The paper's 120 — the value the conductance would reach
    /// if every channel were open, which never happens; the peak of `m³h` in a real spike is
    /// about 0.2.
    pub g_na: f64,
    /// Maximal potassium conductance, mS/cm². The paper's 36.
    pub g_k: f64,
    /// Leak conductance, mS/cm². The paper's 0.3, and the only conductance here that does not gate.
    pub g_leak: f64,
    /// Sodium reversal potential, mV. +50 in the modern convention. The paper writes it as a
    /// displacement from rest in its own frame, where depolarisation is **negative**, so the paper's
    /// number is `V_Na = -115` and the module's substitution `V_paper = -(V_modern + 65)` takes
    /// `+50` to `-(50 + 65) = -115`. It is 115 mV *above* rest; the sign is the paper's convention,
    /// not its physics, and reading it as `+115` inverts the one thing this module's sign section
    /// exists to get right.
    pub e_na: f64,
    /// Potassium reversal potential, mV. -77 in the modern convention, 12 mV **below** rest, which
    /// in the paper's depolarisation-negative frame is written `V_K = +12`: `-(-77 + 65) = +12`.
    pub e_k: f64,
    /// Leak reversal potential, mV. -54.4 is the value in general circulation, and it is not an
    /// independent measurement: it is chosen so that the three currents cancel at -65 mV and the
    /// resting potential comes out where the experiment put it. This implementation measures the
    /// resulting root at -65.00 mV; see [`HodgkinHuxley::rest_potential_mv`].
    pub e_leak: f64,
    /// Membrane potential, mV, inside minus outside, depolarisation positive. See the module doc on
    /// the sign convention before comparing it against the 1952 paper.
    pub v: f64,
    /// Sodium activation gate, `[0,1]`.
    pub m: f64,
    /// Sodium inactivation gate, `[0,1]`, where 1 means **not** inactivated. It sits at 0.60 at
    /// rest, so the cell rests with 40% of its sodium already unavailable.
    pub h: f64,
    /// Potassium activation gate, `[0,1]`. Raised to the fourth power in the conductance, which is
    /// what gives potassium its delayed, sigmoid turn-on.
    pub n: f64,
    /// Membrane area, cm², used only to convert between the model's µA/cm² and the `Neuron` trait's
    /// amperes. The default 1e-4 cm² (10,000 µm², a sphere of 56 µm diameter) is a **convention
    /// chosen for arithmetic**, not a measurement of anything: it makes 1 µA/cm² exactly 0.1 nA, so
    /// the rheobase of about 6 µA/cm² is 0.6 nA, which lands in the same range as the currents the
    /// rest of this crate's models take. The squid giant axon is orders of magnitude larger. Change
    /// it and every ampere in and out of this model rescales; nothing else moves.
    pub area_cm2: f64,
    /// Which integrator [`HodgkinHuxley::advance`] uses. See [`Integrator`] — the choice changes
    /// which guarantees hold.
    pub integrator: Integrator,
    /// How many equal internal substeps each call to [`HodgkinHuxley::advance`] or
    /// [`crate::neuron::Neuron::step`] is split into, minimum 1 in effect.
    ///
    /// Default 4. The crate's simulator commonly runs at `dt = 0.1 ms`, and this model wants
    /// something nearer 25 µs. The sensitive quantity is timing, not amplitude: at `dt = 0.1 ms` with
    /// one substep this implementation measures the 0 mV crossing landing **24.5 µs late** while the
    /// peak is only 0.03 mV low, and four substeps bring the crossing error to **1.7 µs**. Four
    /// substeps buy a fourteenfold improvement in spike timing without the caller having to change
    /// the network's time base.
    pub substeps: u32,
    /// The level, mV, whose upward crossing is reported as a spike by
    /// [`crate::neuron::Neuron::step`]. Default 0 mV, the convention used by `NEURON` and `Brian`.
    ///
    /// **This is a reporting device, not part of the model.** The model has no threshold; something
    /// has to decide when to tell the network a spike happened, and a fixed level partway up a
    /// 100 mV upstroke is the standard choice because the upstroke is so steep that the crossing
    /// time is insensitive to where exactly the level sits.
    ///
    /// It is a **crossing** and not a level: the potential must have been below this value before
    /// the substep and at or above it after. A cell handed over already above the level therefore
    /// reports nothing until it has come back down through `detect_reset` — see `detect_crossing`,
    /// which is where a level test was found reporting a spike for a membrane on its way down.
    pub v_detect: f64,
    /// The level, mV, the potential must fall back below before another spike can be reported.
    /// Default -20 mV.
    ///
    /// This is what makes the detector report one spike per excursion rather than one per crossing:
    /// without it, a trajectory that dips a hair below `v_detect` and comes back — a shrinking limit
    /// cycle near depolarisation block, a noisy synaptic drive — reports a second spike for the same
    /// action potential. The default cell does not need it: at 10 µA/cm² this implementation counts
    /// **21 spikes in 300 ms with `detect_reset = -20` and the same 21 with `detect_reset = 0`**,
    /// because a squid upstroke crosses 0 mV exactly once on the way up. It is insurance, and its
    /// cost is that a membrane parked above `v_detect` never re-arms.
    pub detect_reset: f64,
    /// Whether the detector is ready to report the next upward crossing. False between the crossing
    /// and the fall back below `detect_reset`. Public because a caller reconstructing state has to
    /// be able to set it; it is not part of the differential equations.
    ///
    /// `true` in a cell that is above `v_detect` is not a contradiction and does not mean the next
    /// tick reports a spike: the crossing test needs a step that *enters* the level from below.
    pub armed: bool,
}

impl Default for HodgkinHuxley {
    /// The squid giant axon at 6.3 °C, sitting exactly at its resting potential.
    ///
    /// `c_m = 1 µF/cm²`, `gNa = 120`, `gK = 36`, `gL = 0.3 mS/cm²`, `ENa = +50`, `EK = -77`,
    /// `EL = -54.4 mV`, `V = -65 mV` with all three gates at their steady-state values there
    /// (`m = 0.0529`, `h = 0.5961`, `n = 0.3177`). Those gate values are computed, not transcribed,
    /// so they cannot drift out of step with the rate functions.
    fn default() -> Self {
        let g = steady_state_gates(-65.0);
        Self {
            c_m: 1.0,
            g_na: 120.0,
            g_k: 36.0,
            g_leak: 0.3,
            e_na: 50.0,
            e_k: -77.0,
            e_leak: -54.4,
            v: -65.0,
            m: g.m,
            h: g.h,
            n: g.n,
            area_cm2: 1e-4,
            integrator: Integrator::ExponentialEuler,
            substeps: 4,
            v_detect: 0.0,
            detect_reset: -20.0,
            armed: true,
        }
    }
}

/// The six rate functions in the **modern** sign convention: `v_mv` is the membrane potential,
/// depolarisation positive, resting near -65 mV. Returns rates in 1/ms.
///
/// These are the forms printed in every modern textbook (for example Dayan and Abbott, *Theoretical
/// Neuroscience*, MIT Press 2001, §5.6). They are not independent of [`rates_1952`]: they are that
/// function's six equations under the substitution `V_paper = -(V_modern + 65)`, and the module's
/// provenance test checks exactly that.
///
/// The two removable singularities — `alpha_m` at -40 mV and `alpha_n` at -55 mV — are handled by
/// `exprel_recip`'s series branch, so this function is finite at every finite input.
#[must_use]
pub fn rates(v_mv: f64) -> Rates {
    Rates {
        // 0.1·(V+40) / (1 - exp(-(V+40)/10)) rewritten as x/(1-exp(-x)) with x = (V+40)/10, which
        // is the same expression with the singularity in a form that can be expanded.
        alpha_m: exprel_recip((v_mv + 40.0) / 10.0),
        beta_m: 4.0 * (-(v_mv + 65.0) / 18.0).exp(),
        alpha_h: 0.07 * (-(v_mv + 65.0) / 20.0).exp(),
        beta_h: 1.0 / (1.0 + (-(v_mv + 35.0) / 10.0).exp()),
        // 0.01·(V+55) / (1 - exp(-(V+55)/10)) = 0.1·x/(1-exp(-x)) with x = (V+55)/10.
        alpha_n: 0.1 * exprel_recip((v_mv + 55.0) / 10.0),
        beta_n: 0.125 * (-(v_mv + 65.0) / 80.0).exp(),
    }
}

/// The six rate functions **exactly as the 1952 paper prints them**, in the paper's own frame.
///
/// `v_displacement_mv` is the paper's `V`: the displacement of the membrane potential from its
/// resting value, with **depolarisation NEGATIVE**. So the paper's `V = 0` is the modern -65 mV, and
/// the paper's `V = -115` is the modern +50 mV.
///
/// From Hodgkin and Huxley, *J. Physiol.* 117:500-544, 1952. The six rate functions are equations
/// **(12), (13), (20), (21), (23) and (24)**, each first given where its curve is fitted in Part II
/// and all six restated together under eqn. (26) in the summary of equations that opens Part III,
/// pp. 518-519. One per line below, with its own number:
///
/// ```text
/// (12) alpha_n = 0.01(V+10) / (exp((V+10)/10) - 1)   (13) beta_n = 0.125·exp(V/80)
/// (20) alpha_m = 0.1(V+25)  / (exp((V+25)/10) - 1)   (21) beta_m = 4·exp(V/18)
/// (23) alpha_h = 0.07·exp(V/20)                      (24) beta_h = 1 / (exp((V+30)/10) + 1)
/// ```
///
/// An earlier version of this citation read "(12), (13), (16), (17), (20) and (21)", which is wrong
/// in the one place the module declares itself a receipt: (16) is `dh/dt = alpha_h(1-h) - beta_h·h`
/// and (17) is `m = m_inf - (m_inf - m_0)·exp(-t/tau_m)`. Neither is a rate function. The transcribed
/// formulas were right and only the numbers were wrong, which is the failure a receipt is supposed
/// to make impossible and the reason this doc now names each equation on its own line.
///
/// The sign convention is the paper's own, stated on its p. 505: "V is the displacement of the
/// membrane potential from its resting value (depolarization negative)". That is why the paper's
/// sodium reversal potential is `V_Na = -115` and its potassium reversal potential is `V_K = +12`.
///
/// This function exists to be *checked against*, not to be used: it is the receipt that the
/// constants in [`rates`] came from the primary source rather than from a copy of a copy. Use
/// [`rates`] for anything else, and note that `V` here runs in the opposite direction from every
/// other voltage in this crate, which is precisely the trap it is here to expose.
#[must_use]
pub fn rates_1952(v_displacement_mv: f64) -> Rates {
    let v = v_displacement_mv;
    Rates {
        // y/(exp(y)-1) is exprel_recip(-y): both are the same smooth function, and routing through
        // one implementation means both conventions inherit the same singularity treatment.
        alpha_m: exprel_recip(-(v + 25.0) / 10.0),
        beta_m: 4.0 * (v / 18.0).exp(),
        alpha_h: 0.07 * (v / 20.0).exp(),
        beta_h: 1.0 / (((v + 30.0) / 10.0).exp() + 1.0),
        alpha_n: 0.1 * exprel_recip(-(v + 10.0) / 10.0),
        beta_n: 0.125 * (v / 80.0).exp(),
    }
}

/// The gate values a membrane clamped at `v_mv` forever would settle to, each in `[0,1]`.
///
/// `m_inf` and `n_inf` rise with voltage, `h_inf` falls; all three are sigmoid. Their half-points, as
/// located by bisection in this implementation, are **-40.02 mV for `m`, -62.31 mV for `h` and
/// -53.41 mV for `n`** — note how close `h`'s is to the -65 mV resting potential, which is why the
/// cell rests with 40% of its sodium already inactivated and has so little margin.
#[must_use]
pub fn steady_state_gates(v_mv: f64) -> Gates {
    rates(v_mv).steady_state()
}

impl HodgkinHuxley {
    /// A cell placed at `v_mv` with all three gates at their steady-state values there.
    ///
    /// This is the right way to initialise a voltage-clamp experiment; initialising the gates at
    /// their resting values and then jumping the voltage is a different experiment, and it is the
    /// one the 1952 paper actually performed.
    #[must_use]
    pub fn at(v_mv: f64) -> Self {
        let g = steady_state_gates(v_mv);
        Self { v: v_mv, m: g.m, h: g.h, n: g.n, ..Self::default() }
    }

    /// The resting potential: the voltage at which the three ionic currents cancel with the gates at
    /// their steady state, mV.
    ///
    /// Bisected to machine precision on `[-90, -40]` mV by `bisect_rest`. Returns `None` if the
    /// total ionic current is non-finite at either end of that bracket or has the same sign at both,
    /// which happens for parameter sets whose fixed point lies outside it — a real answer that this
    /// method cannot see is reported as `None` rather than as a bracket endpoint.
    ///
    /// For the default parameters this returns -65.00 mV, which is the point of `e_leak = -54.4`.
    #[must_use]
    pub fn rest_potential_mv(&self) -> Option<f64> {
        bisect_rest(|v| self.i_ion_at(v, steady_state_gates(v)))
    }

    /// A copy of this cell moved to its own resting potential with the gates settled there.
    ///
    /// `None` when [`HodgkinHuxley::rest_potential_mv`] cannot bracket a root. Use it after changing
    /// conductances or reversal potentials: a modified cell left at -65 mV is not at rest, and it
    /// will drift for tens of milliseconds at the start of a run, which looks exactly like a slow
    /// spontaneous depolarisation.
    #[must_use]
    pub fn settled(&self) -> Option<Self> {
        let v = self.rest_potential_mv()?;
        let g = steady_state_gates(v);
        Some(Self { v, m: g.m, h: g.h, n: g.n, armed: true, ..*self })
    }

    /// The instantaneous conductances at the current state, mS/cm².
    #[must_use]
    pub fn conductances(&self) -> Conductances {
        let g_na = self.g_na * self.m * self.m * self.m * self.h;
        let g_k = self.g_k * self.n * self.n * self.n * self.n;
        Conductances { g_na, g_k, g_leak: self.g_leak, total: g_na + g_k + self.g_leak }
    }

    /// The instantaneous ionic currents at the current state, µA/cm², positive outward.
    ///
    /// # The separation, and a correction to how it is usually taught
    ///
    /// The famous figure in the 1952 paper plots **conductances**, and those separate cleanly: this
    /// implementation measures `g_na` peaking 1.48 ms before `g_k`, at 1.58 ms and 3.06 ms after a
    /// brief stimulus. The **currents** do not. Sampled through a freely running action potential
    /// they peak within 0.003 ms of each other, both near 800 µA/cm², because as `V` climbs toward
    /// `ENa` the sodium driving force collapses at the very moment the sodium conductance is
    /// largest. Under voltage clamp — the 1952 experiment — `V` is held, the driving force is
    /// constant, and the currents separate; under current clamp they do not. A teaching figure that
    /// shows separated *currents* from a current-clamp simulation is showing something this
    /// implementation did not reproduce.
    ///
    /// What survives under current clamp is the **dominance ratio**, and it moves by three orders of
    /// magnitude: at maximum `dV/dt` this implementation measures `-i_na / i_k = 4.7`, and at the
    /// trough of the after-hyperpolarisation, 0.0020. That is the checkable version of "sodium
    /// first, potassium second".
    #[must_use]
    pub fn currents(&self) -> Currents {
        let c = self.conductances();
        let i_na = c.g_na * (self.v - self.e_na);
        let i_k = c.g_k * (self.v - self.e_k);
        let i_leak = c.g_leak * (self.v - self.e_leak);
        Currents { i_na, i_k, i_leak, i_ion: i_na + i_k + i_leak }
    }

    /// The instantaneous membrane time constant `C / g_total`, ms.
    ///
    /// This implementation measures 1.4766 ms at rest and 0.02707 ms at the peak of the sodium
    /// conductance, a swing of 54.5. That swing **is** the stiffness of the system, and it is why a
    /// fixed step chosen from the resting time constant integrates the upstroke wrongly.
    ///
    /// Positive whenever `g_leak > 0`, since the gated conductances cannot be negative — but nothing
    /// in the type enforces `g_leak > 0`, every conductance is a public field, and a cell with all
    /// three set to zero has a total conductance of zero and returns `+inf` here. That cell is not a
    /// membrane with an infinitely slow time constant; it is a membrane with no ionic path at all,
    /// and [`crate::neuron::Neuron::step`] refuses to advance it.
    #[must_use]
    pub fn membrane_time_constant_ms(&self) -> f64 {
        self.c_m / self.conductances().total
    }

    /// The gate values this cell's present voltage would settle to.
    #[must_use]
    pub fn steady_state_here(&self) -> Gates {
        steady_state_gates(self.v)
    }

    fn i_ion_at(&self, v: f64, g: Gates) -> f64 {
        self.g_na * g.m * g.m * g.m * g.h * (v - self.e_na)
            + self.g_k * g.n * g.n * g.n * g.n * (v - self.e_k)
            + self.g_leak * (v - self.e_leak)
    }

    /// `(dv/dt, dm/dt, dh/dt, dn/dt)` in mV/ms and 1/ms, at an arbitrary state.
    fn derivatives(&self, v: f64, g: Gates, i_ext: f64) -> [f64; 4] {
        let r = rates(v);
        [
            (i_ext - self.i_ion_at(v, g)) / self.c_m,
            r.alpha_m * (1.0 - g.m) - r.beta_m * g.m,
            r.alpha_h * (1.0 - g.h) - r.beta_h * g.h,
            r.alpha_n * (1.0 - g.n) - r.beta_n * g.n,
        ]
    }

    /// One substep of length `h_ms`. Returns whether a spike was reported during it.
    ///
    /// The potential before the substep is kept so the detector can test a **crossing** rather than
    /// a level; see [`detect_crossing`] for why that distinction is not cosmetic.
    fn sub_step(&mut self, h_ms: f64, i_ext: f64) -> bool {
        let prev = self.v;
        match self.integrator {
            Integrator::ExponentialEuler => {
                // Gates first, at the OLD voltage; then the voltage, with the NEW gates. Both halves
                // are exact for their own linear equation with the other variable frozen, which is
                // what buys the invariants documented on the variant. The ordering is an O(h)
                // choice and is stated so a reader comparing against another implementation knows
                // which one this is.
                let r = rates(self.v);
                let inf = r.steady_state();
                self.m = inf.m + (self.m - inf.m) * (-h_ms * (r.alpha_m + r.beta_m)).exp();
                self.h = inf.h + (self.h - inf.h) * (-h_ms * (r.alpha_h + r.beta_h)).exp();
                self.n = inf.n + (self.n - inf.n) * (-h_ms * (r.alpha_n + r.beta_n)).exp();
                let c = self.conductances();
                let v_inf = (c.g_na * self.e_na
                    + c.g_k * self.e_k
                    + c.g_leak * self.e_leak
                    + i_ext)
                    / c.total;
                self.v = v_inf + (self.v - v_inf) * (-h_ms * c.total / self.c_m).exp();
            }
            Integrator::ForwardEuler => {
                let d = self.derivatives(self.v, Gates { m: self.m, h: self.h, n: self.n }, i_ext);
                self.v += h_ms * d[0];
                self.m += h_ms * d[1];
                self.h += h_ms * d[2];
                self.n += h_ms * d[3];
            }
            Integrator::Rk4 => {
                let y0 = [self.v, self.m, self.h, self.n];
                let at = |y: [f64; 4]| Gates { m: y[1], h: y[2], n: y[3] };
                let k1 = self.derivatives(y0[0], at(y0), i_ext);
                let y1 = shift(y0, k1, 0.5 * h_ms);
                let k2 = self.derivatives(y1[0], at(y1), i_ext);
                let y2 = shift(y0, k2, 0.5 * h_ms);
                let k3 = self.derivatives(y2[0], at(y2), i_ext);
                let y3 = shift(y0, k3, h_ms);
                let k4 = self.derivatives(y3[0], at(y3), i_ext);
                let mut y = y0;
                for j in 0..4 {
                    y[j] += h_ms / 6.0 * (k1[j] + 2.0 * k2[j] + 2.0 * k3[j] + k4[j]);
                }
                self.v = y[0];
                self.m = y[1];
                self.h = y[2];
                self.n = y[3];
            }
        }
        detect_crossing(&mut self.armed, prev, self.v, self.v_detect, self.detect_reset)
    }

    /// Whether the state is one [`HodgkinHuxley::advance`] will return `Ok` for: a finite potential
    /// and three gates inside `[0,1]` up to [`GATE_SLACK`].
    fn state_is_legal(&self) -> bool {
        self.v.is_finite()
            && gate_is_legal(self.m)
            && gate_is_legal(self.h)
            && gate_is_legal(self.n)
    }

    /// Advance by `dt_ms` **milliseconds** under an injected current density of `i_ua_cm2`
    /// **µA/cm²**, in the paper's own frame. Returns whether a spike was reported.
    ///
    /// This is the checked entry point. The step is divided into [`HodgkinHuxley::substeps`] equal
    /// substeps, the inputs are validated before anything moves, and the state is validated
    /// afterwards, so a caller who gets `Ok` has a state that is finite and whose gates are
    /// occupancies. [`crate::neuron::Neuron::step`] cannot do the second of those, because the trait
    /// returns a `bool` and has nowhere to put an error.
    ///
    /// # Errors
    ///
    /// [`HhError::NonFiniteStep`] or [`HhError::NonPositiveStep`] for a `dt_ms` that is not a
    /// positive finite number; [`HhError::NonFiniteCurrent`] for a non-finite current;
    /// [`HhError::Diverged`] if the step left a gate outside `[0,1]` by more than `GATE_SLACK` or
    /// the potential non-finite. [`Integrator::ExponentialEuler`] cannot produce either from a legal
    /// state at any step size and the other two can — but see [`HhError::Diverged`] for the two ways
    /// the default integrator reaches it anyway, both of them through public fields.
    pub fn advance(&mut self, dt_ms: f64, i_ua_cm2: f64) -> Result<bool, HhError> {
        if !dt_ms.is_finite() {
            return Err(HhError::NonFiniteStep);
        }
        if dt_ms <= 0.0 {
            return Err(HhError::NonPositiveStep);
        }
        if !i_ua_cm2.is_finite() {
            return Err(HhError::NonFiniteCurrent);
        }
        let n = self.substeps.max(1);
        let h = dt_ms / f64::from(n);
        let mut fired = false;
        for _ in 0..n {
            fired |= self.sub_step(h, i_ua_cm2);
        }
        if self.state_is_legal() { Ok(fired) } else { Err(HhError::Diverged) }
    }

    /// Whether a sustained `i_ua_cm2` makes this cell report at least one spike within `probe_ms`.
    ///
    /// A simulation, not a bifurcation analysis: it answers for the stimulus it was given, over the
    /// window it was given. `false` from a short window is not a statement about the cell.
    ///
    /// **A run that leaves the legal state region also answers `false`**, so this `bool` carries two
    /// different facts. [`HodgkinHuxley::fires_checked`] separates them, and every search over
    /// currents in this module uses that one instead.
    #[must_use]
    pub fn fires(&self, i_ua_cm2: f64, dt_ms: f64, probe_ms: f64) -> bool {
        self.fires_checked(i_ua_cm2, dt_ms, probe_ms).unwrap_or(false)
    }

    /// [`HodgkinHuxley::fires`] with the divergence separated from the verdict.
    ///
    /// The window is run **to its end even after a spike is seen**, which is the expensive choice and
    /// the correct one: a spike reported on a trajectory that later leaves the legal region is not
    /// evidence that this cell fires, and the state at the moment of such a spike can be perfectly
    /// legal. Measured, on the integrator that has no invariants:
    /// `Integrator::ForwardEuler` at `dt = 0.5 ms` reports a 0 mV crossing at 0.5214 µA/cm² — the `m`
    /// gate is oscillating with a growing amplitude, `m = 0.064` and `v = +32.9 mV` are both inside
    /// their legal ranges at that instant — and the state is illegal one step later. Folded into a
    /// `bool`, that made [`HodgkinHuxley::rheobase_ua_cm2`] report **0.5214 µA/cm² against the
    /// true 2.2493**, a 4.3x error with no error and no warning.
    ///
    /// # Errors
    ///
    /// Whatever [`HodgkinHuxley::advance`] returned: [`HhError::Diverged`] for a run that left the
    /// legal region, or [`HhError::NonFiniteStep`], [`HhError::NonPositiveStep`] or
    /// [`HhError::NonFiniteCurrent`] for a step or current that is not finite and positive. Those
    /// three are checked once here rather than only inside the loop, so that a `dt_ms` of zero is
    /// named instead of turning into a step count of `u64::MAX`.
    pub fn fires_checked(
        &self,
        i_ua_cm2: f64,
        dt_ms: f64,
        probe_ms: f64,
    ) -> Result<bool, HhError> {
        if !dt_ms.is_finite() {
            return Err(HhError::NonFiniteStep);
        }
        if dt_ms <= 0.0 {
            return Err(HhError::NonPositiveStep);
        }
        if !i_ua_cm2.is_finite() {
            return Err(HhError::NonFiniteCurrent);
        }
        let mut c = *self;
        let steps = (probe_ms / dt_ms).ceil().max(0.0) as u64;
        let mut fired = false;
        for _ in 0..steps {
            fired |= c.advance(dt_ms, i_ua_cm2)?;
        }
        Ok(fired)
    }

    /// Steady firing rate under a sustained current, in **hertz**, or `None` if the cell does not
    /// fire at least twice in the measuring window.
    ///
    /// `settle_ms` is discarded before counting, so the first interval — which starts from rest
    /// rather than from the limit cycle — does not enter the average. The rate is taken between the
    /// first and last spike in the window and divided by the number of intervals, never by the
    /// window length, because a window that does not contain a whole number of periods would
    /// otherwise bias the answer downward.
    ///
    /// **`None` means "this measurement did not observe two spikes", not "the rate is small".** A
    /// sub-rheobase Hodgkin-Huxley cell has no firing rate at all, and the Class 2 onset means that
    /// the rates just above rheobase are not small either — they start near 50 Hz. There is no
    /// continuum between the two answers to interpolate across.
    ///
    /// `None` **also** when the run left its legal state region, exactly as
    /// [`HodgkinHuxley::voltage_range_mv`] does and for the same reason: there is nowhere in an
    /// `Option<f64>` to put the difference. It cannot happen under the default integrator, which has
    /// no divergence to report; under the other two at a coarse step it can, and a caller who needs
    /// to tell the two `None`s apart should step the cell through [`HodgkinHuxley::advance`] or ask
    /// [`HodgkinHuxley::fires_checked`] first.
    #[must_use]
    pub fn firing_rate_hz(
        &self,
        i_ua_cm2: f64,
        dt_ms: f64,
        settle_ms: f64,
        window_ms: f64,
    ) -> Option<f64> {
        let mut c = *self;
        let settle = (settle_ms / dt_ms).ceil().max(0.0) as u64;
        for _ in 0..settle {
            c.advance(dt_ms, i_ua_cm2).ok()?;
        }
        let steps = (window_ms / dt_ms).ceil().max(0.0) as u64;
        let mut first = None;
        let mut last = 0.0;
        let mut count = 0u32;
        for k in 0..steps {
            if c.advance(dt_ms, i_ua_cm2).ok()? {
                let t = k as f64 * dt_ms;
                if first.is_none() {
                    first = Some(t);
                }
                last = t;
                count += 1;
            }
        }
        if count < 2 {
            return None;
        }
        let span_ms = last - first?;
        if span_ms <= 0.0 {
            return None;
        }
        Some(1000.0 * f64::from(count - 1) / span_ms)
    }

    /// The smallest sustained current that makes this cell emit **at least one** spike, µA/cm², or
    /// `None` if nothing in `(0, i_max]` does within `probe_ms`.
    ///
    /// **This is not the current at which repetitive firing begins, and the gap between the two is
    /// large.** A step of current from rest produces a single onset spike well below the threshold
    /// for a sustained train, and the cell then sits quiet at a depolarised fixed point. For the
    /// default cell this implementation measures **2.24 µA/cm²** for the onset spike and
    /// **between 6.2 and 6.3 µA/cm²** for repetitive firing, which is
    /// [`HodgkinHuxley::repetitive_onset_ua_cm2`]. Published values near 6.3 for "the rheobase of the
    /// Hodgkin-Huxley model" refer to the second quantity. Quoting one number for both is a factor
    /// of 2.8 error in whichever direction you did not mean.
    ///
    /// Found by scanning upward in `coarse` steps to the first current that fires, then bisecting 40
    /// times inside that bracket. The scan matters: **the set of currents that make this cell fire
    /// repetitively is an interval, not a ray** — see [`HodgkinHuxley::voltage_range_mv`] — so a
    /// plain bisection over a wide range could converge on the upper boundary and report it as the
    /// threshold, producing a plausible number and no error.
    ///
    /// The answer is a simulation result at the step and window you supplied, not a bifurcation
    /// point.
    ///
    /// **`None` also when any probe leaves the legal state region**, because a diverging integrator
    /// has no verdict to give and does not get to guess: every probe here goes through
    /// [`HodgkinHuxley::fires_checked`]. That case is not hypothetical —
    /// `Integrator::ForwardEuler` with one substep at `dt = 0.5 ms` used to answer this question with
    /// 0.5214 µA/cm² against the default integrator's 2.2493, from a 0 mV crossing produced by its
    /// own instability.
    #[must_use]
    pub fn rheobase_ua_cm2(
        &self,
        dt_ms: f64,
        probe_ms: f64,
        i_max: f64,
        coarse: f64,
    ) -> Option<f64> {
        if !(coarse > 0.0) || !(i_max > 0.0) {
            return None;
        }
        let mut lo = 0.0_f64;
        let mut hi = None;
        let mut i = coarse;
        while i <= i_max {
            if self.fires_checked(i, dt_ms, probe_ms).ok()? {
                hi = Some(i);
                break;
            }
            lo = i;
            i += coarse;
        }
        let mut hi = hi?;
        for _ in 0..40 {
            let mid = 0.5 * (lo + hi);
            if self.fires_checked(mid, dt_ms, probe_ms).ok()? {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        Some(0.5 * (lo + hi))
    }

    /// The smallest sustained current for which [`HodgkinHuxley::firing_rate_hz`] reports a rate at
    /// all, µA/cm², or `None` if nothing in `(0, i_max]` fires twice.
    ///
    /// This is the Class 2 onset, and the point of measuring it is what happens to the **rate** at
    /// it. For an integrate-and-fire neuron the rate rises continuously from zero, so the onset
    /// current and the onset rate are the same statement. Here they are not: this implementation
    /// finds no rate at 6.2 µA/cm² and **52.3 Hz** at 6.3. The firing rate of this model is
    /// discontinuous at threshold, jumping from undefined to about 52 Hz, which is the signature of
    /// a subcritical Hopf bifurcation and is what "Class 2 excitability" means (Hodgkin, *J.
    /// Physiol.* 107:165-181, 1948; the modern classification is in Izhikevich, *Dynamical Systems
    /// in Neuroscience*, MIT Press 2007, ch. 7).
    ///
    /// The returned value is resolved to `coarse / 2^40`, but its accuracy is set by `window_ms`:
    /// arbitrarily close to threshold the cell fires arbitrarily slowly to *start*, so a longer
    /// window finds a slightly lower onset. State the window beside the number.
    #[must_use]
    pub fn repetitive_onset_ua_cm2(
        &self,
        dt_ms: f64,
        settle_ms: f64,
        window_ms: f64,
        i_max: f64,
        coarse: f64,
    ) -> Option<f64> {
        if !(coarse > 0.0) || !(i_max > 0.0) {
            return None;
        }
        let fires_twice =
            |i: f64| self.firing_rate_hz(i, dt_ms, settle_ms, window_ms).is_some();
        let mut lo = 0.0_f64;
        let mut hi = None;
        let mut i = coarse;
        while i <= i_max {
            if fires_twice(i) {
                hi = Some(i);
                break;
            }
            lo = i;
            i += coarse;
        }
        let mut hi = hi?;
        for _ in 0..40 {
            let mid = 0.5 * (lo + hi);
            if fires_twice(mid) {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        Some(0.5 * (lo + hi))
    }

    /// The `(minimum, maximum)` membrane potential over `window_ms` after discarding `settle_ms`,
    /// mV, or `None` if the run diverged or the window was empty.
    ///
    /// This exists because **the spike detector and the cell disagree about where firing stops**,
    /// and only one of them is physics. As the injected current rises, the limit cycle's amplitude
    /// shrinks continuously: with 5 s of settling discarded this implementation measures the swing
    /// falling from 105.33 mV at 10 µA/cm² to 69.46 at 60, 40.47 at 100, 8.216 at 150, 4.704 at 153
    /// and 2.750 at 154 — every one of those a converged limit cycle, unchanged to six figures
    /// between 5 s and 20 s of settling. At **155 µA/cm² it collapses**: 6.4e-5 mV after 5 s and
    /// 3.3e-10 mV after 20 s, a number still falling because it is a decaying spiral onto a
    /// depolarised fixed point at -43.03 mV. That is the genuine depolarisation block, and the
    /// bracket it puts the bifurcation in — between 154 and 155 — agrees with the classical figure
    /// of about 154 µA/cm² for the upper limit of repetitive firing in this parameter set (Rinzel
    /// and Miller, *Mathematical Biosciences* 49:27-59, 1980). At 154.5 this implementation cannot
    /// decide: 20 s of settling leaves 0.62 mV of swing and still shrinking, which is what a slow
    /// spiral next to a bifurcation looks like from inside a finite window.
    ///
    /// **`settle_ms` is load-bearing near the block and this doc used to get it wrong.** An earlier
    /// version quoted "0.007 mV at 160 — the last being a fixed point" from a 300 ms settle. At
    /// 160 µA/cm² the converged answer is 4.3e-11 mV; 0.007 was a transient that had not finished
    /// decaying, and reading a bracket off it put the collapse between 154 and 160 instead of
    /// between 154 and 155. Near a bifurcation, settle for seconds and check the number twice at
    /// different settles before calling anything a fixed point.
    ///
    /// But [`HodgkinHuxley::firing_rate_hz`] stops reporting anything above about 62 µA/cm², because
    /// past there the oscillation no longer rises to `v_detect` and fall back to `detect_reset`. The
    /// cell is still oscillating over tens of millivolts; the detector cannot see it. **The upper
    /// end of any firing band this crate reports is a property of the detector, not of the
    /// membrane**, and this method is how you tell the two apart.
    #[must_use]
    pub fn voltage_range_mv(
        &self,
        i_ua_cm2: f64,
        dt_ms: f64,
        settle_ms: f64,
        window_ms: f64,
    ) -> Option<(f64, f64)> {
        let mut c = *self;
        let settle = (settle_ms / dt_ms).ceil().max(0.0) as u64;
        for _ in 0..settle {
            c.advance(dt_ms, i_ua_cm2).ok()?;
        }
        let steps = (window_ms / dt_ms).ceil().max(0.0) as u64;
        if steps == 0 {
            return None;
        }
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for _ in 0..steps {
            c.advance(dt_ms, i_ua_cm2).ok()?;
            lo = lo.min(c.v);
            hi = hi.max(c.v);
        }
        Some((lo, hi))
    }

    /// Measure one action potential: peak, width at `level_mv`, and after-hyperpolarisation.
    ///
    /// The cell is run from its current state under a sustained `i_ua_cm2` for `duration_ms`. The
    /// first continuous excursion above `level_mv` is the spike that gets measured; the
    /// after-hyperpolarisation is the minimum potential between the end of that excursion and the
    /// end of the run, so `duration_ms` has to be long enough to contain the trough — 20 ms is
    /// comfortable at 10 µA/cm², where the interval between spikes is about 15 ms. A run that ends
    /// on the downstroke reports `None` for it rather than a number measured off the wrong part of
    /// the trace.
    ///
    /// `None` if the potential never crosses `level_mv`, or if the run diverges.
    #[must_use]
    pub fn spike_shape(
        &self,
        i_ua_cm2: f64,
        dt_ms: f64,
        duration_ms: f64,
        level_mv: f64,
    ) -> Option<SpikeShape> {
        let mut c = *self;
        let steps = (duration_ms / dt_ms).ceil().max(0.0) as u64;
        let mut up = None;
        let mut down = None;
        let mut peak = f64::NEG_INFINITY;
        let mut peak_t = 0.0;
        let mut ahp = f64::INFINITY;
        for k in 0..steps {
            // The time of the state AFTER this step, which is the state every test below reads.
            // `k * dt_ms` is the time of `prev`, and using it reported the peak one step early.
            let t = (k + 1) as f64 * dt_ms;
            let prev = c.v;
            c.advance(dt_ms, i_ua_cm2).ok()?;
            if up.is_none() && prev < level_mv && c.v >= level_mv {
                up = Some(t);
            }
            if up.is_some() && down.is_none() {
                if c.v > peak {
                    peak = c.v;
                    peak_t = t;
                }
                if prev >= level_mv && c.v < level_mv {
                    down = Some(t);
                }
            }
            if down.is_some() && c.v < ahp {
                ahp = c.v;
            }
        }
        let upstroke_time_ms = up?;
        let width_ms = down? - upstroke_time_ms;
        Some(SpikeShape {
            peak_mv: peak,
            peak_time_ms: peak_t,
            width_ms,
            level_mv,
            upstroke_time_ms,
            after_hyperpolarisation_mv: if ahp.is_finite() { Some(ahp) } else { None },
        })
    }
}

/// `y + s·k`, componentwise — the Runge-Kutta stage shift.
fn shift(y: [f64; 4], k: [f64; 4], s: f64) -> [f64; 4] {
    [y[0] + s * k[0], y[1] + s * k[1], y[2] + s * k[2], y[3] + s * k[3]]
}

impl Neuron for HodgkinHuxley {
    /// **False**, and it is not close. All four variables are coupled and nonlinear, none of them is
    /// stationary under zero input — a cell displaced from rest relaxes back over tens of
    /// milliseconds — and the exponential updates are exact only with the *other* variables frozen.
    /// One step of `k·dt` and `k` steps of `dt` give different answers, so [`crate::sim::Sim`]
    /// refuses to run this model event-driven rather than letting quiet ticks change spike times.
    const EXACT_OVER_GAPS: bool = false;

    /// `dt` in **seconds**, `i` in **amperes**: the SI boundary. Both are converted here into the
    /// paper's milliseconds and µA/cm², the latter through [`HodgkinHuxley::area_cm2`].
    ///
    /// Unlike [`HodgkinHuxley::advance`] this cannot report a problem, so it defends the state and
    /// documents the rest. A non-finite or non-positive `dt`, or a non-finite `i`, leaves the state
    /// **untouched** and returns `false`. So does a step whose *result* would be illegal — a gate
    /// outside `[0,1]` by more than `GATE_SLACK`, or a non-finite potential: the step is rolled
    /// back rather than written, because a `NaN` membrane potential silently poisons every spike
    /// time downstream of it and a frozen cell does not.
    ///
    /// That second case is not only about the non-default integrators. Every conductance is a public
    /// field, and `g_na = g_k = g_leak = 0` makes the exponential update's `v_inf` a `0/0`: this
    /// method used to write `NaN` into the membrane there and return `false`, with the doc claiming
    /// it did not need to check. [`HodgkinHuxley::advance`] answers the same state with
    /// [`HhError::Diverged`], which is where to go to find out *why* a cell stopped moving — a
    /// rolled-back step is indistinguishable from a quiet one through the trait's `bool`.
    fn step(&mut self, dt: f64, i: f64) -> bool {
        if !dt.is_finite() || dt <= 0.0 || !i.is_finite() {
            return false;
        }
        let before = *self;
        let dt_ms = dt * 1e3;
        let i_density = i * 1e6 / self.area_cm2;
        let n = self.substeps.max(1);
        let h = dt_ms / f64::from(n);
        let mut fired = false;
        for _ in 0..n {
            fired |= self.sub_step(h, i_density);
        }
        if self.state_is_legal() {
            fired
        } else {
            *self = before;
            false
        }
    }

    /// `dv` in volts, converted to the model's millivolts and added to the membrane.
    ///
    /// Applied unconditionally: there is no refractory flag to consult, because this model has no
    /// refractory period to impose. A synaptic input arriving during the falling phase of a spike is
    /// integrated, and whether it does anything is decided by `h` and `n` — which is the physical
    /// answer and the reason the crate's other models need a flag to approximate it.
    fn bump(&mut self, dv: f64) {
        self.v += dv * 1e3;
    }

    /// Always `0.0` — **and that is the finding, not a stub.**
    ///
    /// Hodgkin-Huxley has no absolute refractory period as a parameter. What it has is sodium
    /// inactivation that needs milliseconds to recover and potassium activation that needs
    /// milliseconds to decay, and between them they make a second spike impossible for a few
    /// milliseconds and harder for ten or so. The test
    /// `the_refractory_period_emerges_from_h_and_n` drives two pulses and shows the second fail with
    /// this method returning zero throughout.
    fn refractory_left(&self) -> f64 {
        0.0
    }

    /// Membrane potential in **volts**, converted from the model's millivolts.
    fn potential(&self) -> f64 {
        self.v * 1e-3
    }

    /// Back to -65 mV with the gates at their steady state there, and the spike detector re-armed.
    ///
    /// Note that -65 mV is the resting potential of the *default* parameters. A cell whose
    /// conductances have been changed should be reset through [`HodgkinHuxley::settled`], which
    /// solves for its own fixed point.
    fn reset(&mut self) {
        let g = steady_state_gates(-65.0);
        self.v = -65.0;
        self.m = g.m;
        self.h = g.h;
        self.n = g.n;
        self.armed = true;
    }
}

/// The two-variable reduction of Hodgkin-Huxley, after Rinzel.
///
/// **Source:** J. Rinzel, "Excitation dynamics: insights from simplified membrane models",
/// *Federation Proceedings* 44:2944-2946, 1985. The same reduction is reproduced in Keener and
/// Sneyd, *Mathematical Physiology*, Springer, and in Izhikevich, *Dynamical Systems in
/// Neuroscience*, MIT Press 2007.
///
/// # The two observations it rests on
///
/// 1. **`m` is fast.** At every voltage `tau_m` is under 0.5 ms while `tau_h` and `tau_n` are
///    several milliseconds, so sodium activation can be treated as instantaneous: `m = m_inf(V)`.
///    That removes one variable exactly in the limit of infinite separation, and approximately at
///    the real separation of about twenty.
/// 2. **`h` and `n` move together.** Along a trajectory through a spike, `h + n` stays roughly
///    constant, so `h` can be written as an affine function of `n`. The coefficients in circulation
///    are `h ≈ 0.89 - 1.1·n`, and they are carried here as [`ReducedHh::h_intercept`] and
///    [`ReducedHh::h_slope`] rather than baked in.
///
/// ⚠ **This implementation did not verify 0.89 and 1.1 against the 1985 Federation Proceedings
/// abstract**, which is a two-page conference abstract; they are the values reproduced in the
/// secondary literature. What this module does instead is *measure the approximation*: the test
/// `the_h_from_n_approximation_is_good_but_is_not_an_identity` runs the full model through a spike
/// and reports the largest residual `|h - (0.89 - 1.1n)|` along it, and asserts both that it is
/// small and that it is not zero. If the coefficients are wrong, that residual is the quantity that
/// says so. As measured: the largest residual along a full spike is **0.0648** and the smallest is
/// 0.00018, so the relation is good to about 6% of the unit interval and is emphatically not an
/// identity.
///
/// # What the reduction loses
///
/// - **The `m` transient.** The upstroke is too fast and the spike overshoots: this implementation
///   measures a peak of **+47.98 mV against the full model's +40.32** on the same brief stimulus,
///   and a width of 1.04 ms against 1.165 ms. A real `m` takes ~0.1 ms to follow a voltage jump and
///   this one takes none, so nothing limits the sodium influx at the top of the upstroke.
/// - **Independent `h`.** Any experiment that sets sodium inactivation and potassium activation
///   independently is outside the model's reach — including recovery from a hyperpolarising pulse,
///   where the full model's `h` de-inactivates *above* its resting value and produces anode-break
///   excitation. The resting point moves too, from -64.9997 mV to -65.098 mV, and the
///   repetitive-firing onset moves from between 6.2 and 6.3 µA/cm² to between 4.5 and 5.0 — a 25%
///   error in the threshold current, which is the largest single cost of the reduction.
/// - **Dimension.** A two-dimensional autonomous system cannot be chaotic; by the
///   Poincaré-Bendixson theorem its bounded trajectories can only approach fixed points or closed
///   orbits. The four-dimensional model under periodic stimulation can be, and is reported to be
///   (Aihara and Matsumoto, *J. Theor. Biol.* 109:249-269, 1984). Any such behaviour is structurally
///   invisible here.
///
/// # What the reduction buys
///
/// A phase plane. With two variables the whole dynamics is a picture: the `V`-nullcline is cubic,
/// the `n`-nullcline is a sigmoid, and where they cross and how they cross is the threshold, the
/// spike and the Class 2 onset. That picture is why this reduction is taught, and it is the reason
/// it is in an education-first library. It is not an efficiency measure — it saves two of four
/// variables and still needs the same fine time step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReducedHh {
    /// Specific membrane capacitance, µF/cm².
    pub c_m: f64,
    /// Maximal sodium conductance, mS/cm².
    pub g_na: f64,
    /// Maximal potassium conductance, mS/cm².
    pub g_k: f64,
    /// Leak conductance, mS/cm².
    pub g_leak: f64,
    /// Sodium reversal potential, mV.
    pub e_na: f64,
    /// Potassium reversal potential, mV.
    pub e_k: f64,
    /// Leak reversal potential, mV.
    pub e_leak: f64,
    /// Membrane potential, mV, modern convention.
    pub v: f64,
    /// Potassium activation, `[0,1]` — the only gating state left. Sodium activation is slaved to
    /// the voltage and sodium inactivation is slaved to this.
    pub n: f64,
    /// Intercept of `h ≈ h_intercept - h_slope·n`, dimensionless. 0.89 in the literature; see the
    /// type doc for what this implementation did and did not verify about it.
    pub h_intercept: f64,
    /// Slope of `h ≈ h_intercept - h_slope·n`, dimensionless. 1.1 in the literature.
    pub h_slope: f64,
    /// Membrane area, cm², for the SI boundary only. Same convention as
    /// [`HodgkinHuxley::area_cm2`].
    pub area_cm2: f64,
    /// Internal substeps per call, minimum 1 in effect. Default 4, for the same reason as
    /// [`HodgkinHuxley::substeps`].
    pub substeps: u32,
    /// Spike-reporting level, mV. Default 0.
    pub v_detect: f64,
    /// Re-arm level, mV. Default -20.
    pub detect_reset: f64,
    /// Whether the detector is ready to report the next upward crossing.
    pub armed: bool,
}

impl Default for ReducedHh {
    /// The same squid parameters as [`HodgkinHuxley::default`], placed at **its own** fixed point.
    ///
    /// That point is not -65 mV: slaving `h` to `n` moves `h` at rest from 0.5961 to 0.5422, which
    /// unbalances the currents slightly, and the reduced cell rests at a measured **-65.098 mV**
    /// instead. A 0.1 mV shift is small, and it is shown rather than hidden because it is the first
    /// observable consequence of the approximation — and because a reduced cell initialised at
    /// -65.0 mV would drift for tens of milliseconds at the start of every run.
    fn default() -> Self {
        let seed = Self {
            c_m: 1.0,
            g_na: 120.0,
            g_k: 36.0,
            g_leak: 0.3,
            e_na: 50.0,
            e_k: -77.0,
            e_leak: -54.4,
            v: -65.0,
            n: steady_state_gates(-65.0).n,
            h_intercept: 0.89,
            h_slope: 1.1,
            area_cm2: 1e-4,
            substeps: 4,
            v_detect: 0.0,
            detect_reset: -20.0,
            armed: true,
        };
        // If the bisection cannot bracket the fixed point for these parameters, the un-settled seed
        // is still a usable cell — it just starts with a drift, which `rest_potential_mv` reports.
        seed.settled().unwrap_or(seed)
    }
}

impl ReducedHh {
    /// Sodium inactivation implied by the present `n`, clamped into `[0,1]`.
    ///
    /// The clamp binds at the top for `n < (h_intercept - 1)/h_slope`, which for the default
    /// coefficients is never, since that is a negative number; and at the bottom for
    /// `n > h_intercept/h_slope = 0.809`, which the potassium gate does approach at the peak of a
    /// strong spike. Without the clamp `h` would go negative there and the sodium current would
    /// reverse direction — a sign error that a plot of the voltage would not obviously show.
    #[must_use]
    pub fn h(&self) -> f64 {
        (self.h_intercept - self.h_slope * self.n).clamp(0.0, 1.0)
    }

    /// Sodium activation implied by the present voltage: `m_inf(v)`, the instantaneous limit.
    #[must_use]
    pub fn m(&self) -> f64 {
        steady_state_gates(self.v).m
    }

    /// The instantaneous conductances, mS/cm².
    #[must_use]
    pub fn conductances(&self) -> Conductances {
        let m = self.m();
        let g_na = self.g_na * m * m * m * self.h();
        let g_k = self.g_k * self.n * self.n * self.n * self.n;
        Conductances { g_na, g_k, g_leak: self.g_leak, total: g_na + g_k + self.g_leak }
    }

    /// The instantaneous ionic currents, µA/cm², positive outward.
    #[must_use]
    pub fn currents(&self) -> Currents {
        let c = self.conductances();
        let i_na = c.g_na * (self.v - self.e_na);
        let i_k = c.g_k * (self.v - self.e_k);
        let i_leak = c.g_leak * (self.v - self.e_leak);
        Currents { i_na, i_k, i_leak, i_ion: i_na + i_k + i_leak }
    }

    /// The reduced model's resting potential, mV, or `None` if no root lies in `[-90, -40]` mV.
    ///
    /// Literally the same bisection as [`HodgkinHuxley::rest_potential_mv`] — both call
    /// `bisect_rest` — differing only in the current-balance equation handed to it, where `h`
    /// comes from the affine relation rather than from `h_inf`. It was a second copy of the loop
    /// until an audit found the copy missing the finiteness guard, so that `ReducedHh { e_leak: NaN,
    /// .. }` answered `Some(-40.0)`, the bracket endpoint, and [`ReducedHh::settled`] sat the cell
    /// there.
    #[must_use]
    pub fn rest_potential_mv(&self) -> Option<f64> {
        bisect_rest(|v| {
            let g = steady_state_gates(v);
            let h = (self.h_intercept - self.h_slope * g.n).clamp(0.0, 1.0);
            self.g_na * g.m * g.m * g.m * h * (v - self.e_na)
                + self.g_k * g.n * g.n * g.n * g.n * (v - self.e_k)
                + self.g_leak * (v - self.e_leak)
        })
    }

    /// A copy of this cell at its own fixed point with `n` settled there.
    ///
    /// `None` when the bisection cannot bracket a root.
    #[must_use]
    pub fn settled(&self) -> Option<Self> {
        let v = self.rest_potential_mv()?;
        Some(Self { v, n: steady_state_gates(v).n, armed: true, ..*self })
    }

    /// One substep of length `h_ms`. Returns whether a spike was reported during it.
    ///
    /// The detector is [`detect_crossing`], the same function the full model uses rather than a
    /// second copy of it: the copy that used to live here inherited the level-versus-crossing defect
    /// and would have had to be fixed twice.
    fn sub_step(&mut self, h_ms: f64, i_ext: f64) -> bool {
        let prev = self.v;
        let r = rates(self.v);
        let n_inf = r.alpha_n / (r.alpha_n + r.beta_n);
        self.n = n_inf + (self.n - n_inf) * (-h_ms * (r.alpha_n + r.beta_n)).exp();
        let c = self.conductances();
        let v_inf =
            (c.g_na * self.e_na + c.g_k * self.e_k + c.g_leak * self.e_leak + i_ext) / c.total;
        self.v = v_inf + (self.v - v_inf) * (-h_ms * c.total / self.c_m).exp();
        detect_crossing(&mut self.armed, prev, self.v, self.v_detect, self.detect_reset)
    }

    /// Advance by `dt_ms` milliseconds under `i_ua_cm2` µA/cm². Returns whether a spike was
    /// reported.
    ///
    /// Integrated by the same exponential-Euler scheme as [`Integrator::ExponentialEuler`] and
    /// carrying the same two invariants: `n` cannot leave `[0,1]` and, under zero input, `v` cannot
    /// leave `[e_k, e_na]`, at any step size.
    ///
    /// # Errors
    ///
    /// [`HhError::NonFiniteStep`], [`HhError::NonPositiveStep`] or [`HhError::NonFiniteCurrent`] for
    /// inputs that are not a positive finite step and a finite current. [`HhError::Diverged`] if the
    /// step left `n` outside `[0,1]` by more than `GATE_SLACK` or the potential non-finite.
    ///
    /// The integrator cannot *produce* either, and an earlier version of this doc concluded from
    /// that that `Diverged` was unreachable. It is not: `n` is a public field, so
    /// `ReducedHh { n: 1.5, ..Default::default() }.advance(0.01, 0.0)` is `Err(Diverged)` — the
    /// guard is there for the state a caller hands over, exactly as in the full model. The tolerance
    /// is `GATE_SLACK` for the same reason it is there: this check used to be an exact
    /// `(0.0..=1.0)`, eight orders of magnitude stricter than the full model's, which is a
    /// difference no doc mentioned and no test would have caught.
    pub fn advance(&mut self, dt_ms: f64, i_ua_cm2: f64) -> Result<bool, HhError> {
        if !dt_ms.is_finite() {
            return Err(HhError::NonFiniteStep);
        }
        if dt_ms <= 0.0 {
            return Err(HhError::NonPositiveStep);
        }
        if !i_ua_cm2.is_finite() {
            return Err(HhError::NonFiniteCurrent);
        }
        let n = self.substeps.max(1);
        let h = dt_ms / f64::from(n);
        let mut fired = false;
        for _ in 0..n {
            fired |= self.sub_step(h, i_ua_cm2);
        }
        if self.v.is_finite() && gate_is_legal(self.n) {
            Ok(fired)
        } else {
            Err(HhError::Diverged)
        }
    }
}

impl Neuron for ReducedHh {
    /// False, for the same reasons as the full model: nonlinear, coupled, and not stationary under
    /// zero input.
    const EXACT_OVER_GAPS: bool = false;

    /// `dt` in seconds and `i` in amperes, converted at this boundary. A non-finite or non-positive
    /// `dt`, or a non-finite `i`, leaves the state untouched and returns `false` — and so does a step
    /// whose result would be illegal, which is rolled back rather than written, for the same reason
    /// and in the same cases as [`HodgkinHuxley`]'s: all three conductances are public and all three
    /// set to zero makes `v_inf` a `0/0`.
    fn step(&mut self, dt: f64, i: f64) -> bool {
        if !dt.is_finite() || dt <= 0.0 || !i.is_finite() {
            return false;
        }
        let before = *self;
        let n = self.substeps.max(1);
        let h = dt * 1e3 / f64::from(n);
        let i_density = i * 1e6 / self.area_cm2;
        let mut fired = false;
        for _ in 0..n {
            fired |= self.sub_step(h, i_density);
        }
        if self.v.is_finite() && gate_is_legal(self.n) {
            fired
        } else {
            *self = before;
            false
        }
    }

    /// `dv` in volts, added to the membrane in millivolts.
    fn bump(&mut self, dv: f64) {
        self.v += dv * 1e3;
    }

    /// Always `0.0`: refractoriness here is carried by `n` alone, and is therefore weaker and
    /// shorter than the full model's, which also has `h` to recover.
    fn refractory_left(&self) -> f64 {
        0.0
    }

    /// Membrane potential in volts.
    fn potential(&self) -> f64 {
        self.v * 1e-3
    }

    /// Back to this cell's own fixed point, or to -65 mV with `n` settled there if the fixed point
    /// cannot be bracketed.
    fn reset(&mut self) {
        if let Some(s) = self.settled() {
            *self = s;
        } else {
            self.v = -65.0;
            self.n = steady_state_gates(-65.0).n;
            self.armed = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        GATE_SLACK, HhError, HodgkinHuxley, Integrator, ReducedHh, Taus, exprel_recip,
        gate_is_legal, rates, rates_1952, steady_state_gates,
    };
    use crate::net::NetBuilder;
    use crate::neuron::Neuron;
    use crate::rng::Rng;
    use crate::sim::{Mode, Sim, SimError};

    /// Interpolated time of the first upward crossing of 0 mV, ms. Interpolating matters: the raw
    /// crossing tick is quantised to `dt`, so a convergence study on it measures the quantisation
    /// and not the integrator, and reports convergence long before the integrator has any.
    fn crossing_ms(dt: f64, substeps: u32, integ: Integrator, i: f64) -> Option<f64> {
        let mut x = HodgkinHuxley { substeps, integrator: integ, ..HodgkinHuxley::default() };
        let mut k = 0u64;
        loop {
            let prev = x.v;
            x.advance(dt, i).ok()?;
            let t = k as f64 * dt;
            if prev < 0.0 && x.v >= 0.0 {
                return Some(t + dt * (0.0 - prev) / (x.v - prev));
            }
            k += 1;
            if k as f64 * dt > 60.0 {
                return None;
            }
        }
    }

    /// Run a cell under a `amp` µA/cm² pulse lasting `pulse_ms`, then zero, for `total_ms`.
    /// Returns `(peak_mv, spikes, final_state)`.
    fn pulse(
        cell: HodgkinHuxley,
        amp: f64,
        pulse_ms: f64,
        total_ms: f64,
        dt: f64,
    ) -> (f64, u32, HodgkinHuxley) {
        let mut x = cell;
        let mut peak = f64::NEG_INFINITY;
        let mut spikes = 0u32;
        let steps = (total_ms / dt) as u64;
        for k in 0..steps {
            let t = k as f64 * dt;
            let i = if t < pulse_ms { amp } else { 0.0 };
            if x.advance(dt, i).expect("exponential Euler cannot diverge") {
                spikes += 1;
            }
            peak = peak.max(x.v);
        }
        (peak, spikes, x)
    }

    /// **The two transcriptions agree.** The rate functions in [`rates`] must be the 1952 paper's
    /// own six equations under the paper's sign convention, `V_paper = -(V_modern + 65)`, and this
    /// checks it at 641 voltages spanning everything the model ever visits.
    ///
    /// This is the one test in the module that checks the constants themselves rather than the
    /// behaviour they produce. Without it, a transposed digit in `beta_n`'s 80 would be caught by
    /// nothing else here: the spike would still look like a spike.
    ///
    /// **It is not a provenance receipt and this module used to call it one.** Two transcriptions of
    /// the same wrong source agree with each other perfectly, and the algebra that turns one into the
    /// other was done by hand here (see the comment in `rates_1952` about `y/(exp(y)-1)`), so a
    /// `rates_1952` derived FROM `rates` would pass identically. What backs provenance is the
    /// citation — and the citation named the wrong equations. What this test does pin, beyond the
    /// constants, is the sign convention: the paper's `V = 0`, `V = -115` and `V = +12` are the
    /// modern -65, +50 and -77 mV, which is the claim two of the reversal-potential docs stated
    /// backwards.
    #[test]
    fn the_paper_s_own_rate_functions_transform_into_the_modern_ones() {
        let mut worst = 0.0_f64;
        for k in 0..=640 {
            let v = -100.0 + f64::from(k) * 0.25;
            let modern = rates(v);
            let paper = rates_1952(-(v + 65.0));
            let pairs = [
                ("alpha_m", modern.alpha_m, paper.alpha_m),
                ("beta_m", modern.beta_m, paper.beta_m),
                ("alpha_h", modern.alpha_h, paper.alpha_h),
                ("beta_h", modern.beta_h, paper.beta_h),
                ("alpha_n", modern.alpha_n, paper.alpha_n),
                ("beta_n", modern.beta_n, paper.beta_n),
            ];
            for (name, a, b) in pairs {
                assert!(a.is_finite() && b.is_finite(), "{name} not finite at {v} mV");
                let rel = (a - b).abs() / a.abs().max(1e-300);
                worst = worst.max(rel);
                assert!(rel < 1e-12, "{name} at V = {v} mV: modern {a} vs paper {b}");
            }
        }
        // Stronger than the loop, and not a restatement of it: on this grid every voltage is a
        // multiple of 0.25 mV, so the substitution's arithmetic is exact in f64 and both sides hand
        // `exp` bit-identical arguments. The two forms therefore agree bit for bit, not merely to
        // 1e-12 — and a `worst` of 1e-16 would mean they had stopped being the same function and
        // started being two approximations of it. (The line this replaces asserted `worst < 1e-12`
        // under a comment saying it was not asserted, and duplicated the in-loop check.)
        assert_eq!(worst, 0.0, "worst relative disagreement {worst}");

        // **The sign convention, which is prose everywhere else in this module and was wrong twice.**
        // Depolarisation is NEGATIVE in the paper's frame, so the paper's sodium reversal potential
        // is -115 and its potassium reversal potential is +12, not the +115 and -12 the field docs
        // used to claim.
        let c = HodgkinHuxley::default();
        for (name, v_paper, v_modern) in
            [("rest", 0.0, -65.0), ("E_Na", -115.0, c.e_na), ("E_K", 12.0, c.e_k)]
        {
            assert_eq!(v_paper, -(v_modern + 65.0), "{name}: the substitution does not map them");
            let (paper, modern) = (rates_1952(v_paper), rates(v_modern));
            assert_eq!(
                (paper.alpha_m, paper.beta_h, paper.beta_n),
                (modern.alpha_m, modern.beta_h, modern.beta_n),
                "{name}: the two frames disagree at the reversal potential"
            );
        }
        // And the direction is not symmetric: reading the paper's E_Na as +115 lands 230 mV away.
        assert!((rates_1952(115.0).alpha_m - rates_1952(-115.0).alpha_m).abs() > 1.0);
    }

    /// The two removable singularities must equal their L'Hôpital limits, and the series branch must
    /// join the direct branch smoothly.
    ///
    /// `alpha_m` is `0.1(V+40)/(1 - exp(-(V+40)/10))`, which is `0/0` at exactly -40 mV. The limit is
    /// `0.1 / (1/10) = 1`. `alpha_n` is the same shape at -55 mV with limit 0.1. A direct
    /// transcription returns `NaN` at those two voltages and nowhere else, which is a bug that finds
    /// a simulation about once in a few million steps and then silences it.
    #[test]
    fn the_removable_singularities_equal_their_limits_and_the_branches_join() {
        assert_eq!(rates(-40.0).alpha_m, 1.0, "alpha_m at its singular voltage");
        assert_eq!(rates(-55.0).alpha_n, 0.1, "alpha_n at its singular voltage");
        assert_eq!(exprel_recip(0.0), 1.0);

        // Approach from both sides, against the Taylor series 1 + x/2 + x^2/12.
        for &d in &[-1e-3, -1e-5, -1e-7, 1e-7, 1e-5, 1e-3] {
            let v = -40.0 + d;
            let x = d / 10.0;
            let want = 1.0 + 0.5 * x + x * x / 12.0;
            let got = rates(v).alpha_m;
            assert!((got - want).abs() < 1e-12, "alpha_m({v}) = {got}, series says {want}");
        }
        // The series branch is used below 1e-8; just above it the direct `exp_m1` branch must give
        // the same answer, or the function has a step in it at the threshold.
        for &x in &[1e-8_f64, 2e-8, 1e-7, 1e-6] {
            let direct = x / -((-x).exp_m1());
            let series = 1.0 + 0.5 * x + x * x / 12.0;
            assert!(
                (direct - series).abs() < 1e-15,
                "branches disagree at x = {x}: {direct} vs {series}"
            );
            // ...and the function is on the accurate side of the seam. Written as
            // `exprel_recip(x)` vs `series`, the check above would compare the series against
            // itself for any threshold wide enough to swallow these four points, which is a test
            // that a later change could make unfalsifiable without touching the test.
            assert!((exprel_recip(x) - direct).abs() < 1e-15, "exprel_recip({x}) took the series");
        }
        // **An independent identity, because the check above recomputes the implementation.**
        // `1/(1-e^-x) + 1/(1-e^x) = 1` for every x, so `f(x) - f(-x) = x` EXACTLY, for both
        // branches and across the threshold between them. A mutation test found that without this,
        // dropping the `x/2` term from the series branch changed nothing any test could see: the
        // branch is reachable only for `|x| < 1e-8`, where every other assertion here is satisfied
        // by `1.0`. The identity is a property of the function, not a copy of the code.
        for &x in &[1e-12, 1e-10, 5e-9, 1e-8, 1e-7, 1e-3, 1.0, 5.0, -5.0, -1e-9] {
            let d = exprel_recip(x) - exprel_recip(-x);
            assert!((d - x).abs() < 1e-15, "f({x}) - f({}) = {d}, should be {x}", -x);
        }
        // And the slope at the origin is one half, from the same expansion — the term the mutant
        // dropped, measured rather than restated. The tolerance is set by cancellation, not by the
        // series: `f(x) - 1` for `x ~ 1e-9` subtracts two numbers that differ in the tenth digit, so
        // the quotient carries about `eps / 2x ~ 1e-7` of relative error however exact the series is.
        // The mutant's slope is `x/12 ~ 1e-10`, so 1e-6 separates them by five orders of magnitude.
        for &x in &[1e-9, 5e-9, -3e-9] {
            let slope = (exprel_recip(x) - 1.0) / x;
            assert!((slope - 0.5).abs() < 1e-6, "slope near 0 at x = {x} is {slope}, not 1/2");
        }
        // And nothing anywhere near the singularities is non-finite.
        for &v in &[-55.0, -40.0, -55.0 + 1e-14, -40.0 - 1e-14, -54.999_999_999, -40.000_000_001] {
            let r = rates(v);
            let all = [r.alpha_m, r.beta_m, r.alpha_h, r.beta_h, r.alpha_n, r.beta_n];
            assert!(all.iter().all(|x| x.is_finite()), "non-finite rate at {v} mV");
        }
    }

    /// The steady-state curves are the model's static picture, and they must be sigmoid and monotone
    /// in the directions the biology requires: activation rises with voltage, inactivation falls.
    /// An `h_inf` that rose with voltage would give a cell that fired harder the longer it was
    /// depolarised, which is the opposite of a neuron.
    #[test]
    fn the_steady_state_curves_are_sigmoid_and_monotone_in_the_right_directions() {
        let mut prev = steady_state_gates(-100.0);
        for k in 1..=1600 {
            let v = -100.0 + f64::from(k) * 0.1;
            let g = steady_state_gates(v);
            for (name, x) in [("m", g.m), ("h", g.h), ("n", g.n)] {
                assert!(x > 0.0 && x < 1.0, "{name}_inf = {x} at {v} mV is outside (0,1)");
            }
            assert!(g.m > prev.m, "m_inf not increasing at {v} mV");
            assert!(g.n > prev.n, "n_inf not increasing at {v} mV");
            assert!(g.h < prev.h, "h_inf not decreasing at {v} mV");
            prev = g;
        }
        // The saturating limits, which is what makes them sigmoid rather than merely monotone.
        let lo = steady_state_gates(-200.0);
        let hi = steady_state_gates(200.0);
        assert!(lo.m < 1e-8, "m_inf at -200 mV = {}", lo.m);
        assert!(lo.h > 1.0 - 1e-6, "h_inf at -200 mV = {}", lo.h);
        assert!(lo.n < 1e-3, "n_inf at -200 mV = {}", lo.n);
        assert!(hi.m > 1.0 - 1e-4, "m_inf at +200 mV = {}", hi.m);
        assert!(hi.h < 1e-6, "h_inf at +200 mV = {}", hi.h);
        assert!(hi.n > 0.99, "n_inf at +200 mV = {}", hi.n);
    }

    /// Where each curve crosses one half, against the values quoted in the docs. These are the
    /// numbers a reader would compare against a textbook figure, so they are pinned.
    #[test]
    fn the_half_activation_voltages_are_where_the_docs_say() {
        let half = |f: fn(f64) -> f64| {
            let (mut lo, mut hi) = (-120.0_f64, 60.0_f64);
            for _ in 0..200 {
                let mid = 0.5 * (lo + hi);
                if (f(lo) - 0.5) * (f(mid) - 0.5) <= 0.0 {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }
            0.5 * (lo + hi)
        };
        let vm = half(|v| steady_state_gates(v).m);
        let vh = half(|v| steady_state_gates(v).h);
        let vn = half(|v| steady_state_gates(v).n);
        assert!((vm - -40.02).abs() < 0.02, "m half-activation {vm}");
        assert!((vh - -62.31).abs() < 0.02, "h half-inactivation {vh}");
        assert!((vn - -53.41).abs() < 0.02, "n half-activation {vn}");
        // h is half-inactivated within 3 mV of rest: the cell rests with 40% of its sodium already
        // unavailable, which is the margin the whole model runs on.
        assert!((vh - -65.0).abs() < 3.0, "h half-point {vh} is far from rest");
    }

    /// **(a)** A stable resting potential that persists. The default cell is set to -65.0 mV, the
    /// currents actually cancel at -64.99972 mV, and over 500 ms of zero input it moves the 2.8e-4
    /// mV between the two and then stops. A model with the sign of a conductance wrong, or a leak
    /// that does not balance, drifts or oscillates here.
    #[test]
    fn the_resting_potential_is_a_fixed_point_that_persists_for_half_a_second() {
        let c = HodgkinHuxley::default();
        let root = c.rest_potential_mv().expect("the default bracket contains the root");
        assert!((root - -65.0).abs() < 5e-4, "rest solved to {root} mV");
        let i = c.currents();
        assert!(i.i_ion.abs() < 1e-3, "net ionic current at the default state is {}", i.i_ion);
        assert!(i.i_na < 0.0 && i.i_k > 0.0, "sodium must be inward and potassium outward at rest");

        let mut x = c;
        for _ in 0..50_000 {
            x.advance(0.01, 0.0).expect("no divergence at rest");
        }
        let moved = (x.v - c.v).abs();
        assert!(moved < 1e-3, "drifted {moved} mV in 500 ms");
        assert!((x.v - root).abs() < 1e-4, "did not settle onto the root: {} vs {root}", x.v);
        // And it settled BY MOVING: a cell that never moved at all would pass the line above while
        // being frozen by a broken integrator.
        assert!(moved > 1e-6, "did not move at all in 500 ms, which no integrator should do");
    }

    /// The default's gates must be the steady state of the default's voltage, computed rather than
    /// transcribed. A hand-copied 0.0529 that drifted out of step with the rate functions would put
    /// the cell slightly off its fixed point, and every figure produced from the default would open
    /// with a slow drift that looked like biology.
    #[test]
    fn the_default_gates_are_the_steady_state_of_the_default_voltage() {
        let c = HodgkinHuxley::default();
        let g = steady_state_gates(c.v);
        assert_eq!((c.m, c.h, c.n), (g.m, g.h, g.n));
        assert!((c.m - 0.052_932).abs() < 1e-5, "m at rest {}", c.m);
        assert!((c.h - 0.596_121).abs() < 1e-5, "h at rest {}", c.h);
        assert!((c.n - 0.317_677).abs() < 1e-5, "n at rest {}", c.n);

        // The rest of the defaults, pinned because the docs quote them and every measured figure in
        // this module was taken with them.
        assert_eq!((c.c_m, c.g_na, c.g_k, c.g_leak), (1.0, 120.0, 36.0, 0.3));
        assert_eq!((c.e_na, c.e_k, c.e_leak), (50.0, -77.0, -54.4));
        assert_eq!((c.v, c.area_cm2, c.substeps), (-65.0, 1e-4, 4));
        assert_eq!((c.v_detect, c.detect_reset, c.armed), (0.0, -20.0, true));
        assert_eq!(c.integrator, Integrator::ExponentialEuler);
    }

    /// The detection level is a **reporting convention and not part of the model**, and the doc on
    /// [`HodgkinHuxley::v_detect`] claims the upstroke is steep enough that it barely matters where
    /// the level sits. That claim is checked here rather than asserted. Measured: sweeping the level
    /// across 40 mV of a 105 mV upstroke, from -20 mV to +20 mV, moves the reported spike time from
    /// 1.818 ms to 1.967 ms — a total of 0.149 ms, an eighth of the spike's own width, against a
    /// stimulus level range of 40 mV.
    ///
    /// This test exists because a mutation that changed the default level from 0 to 5 mV survived
    /// the whole rest of the module. The survivor was not a gap in the physics — it was this
    /// property, unasserted.
    #[test]
    fn the_detection_level_is_a_convention_the_upstroke_makes_almost_irrelevant() {
        let first_spike_ms = |level: f64| {
            let mut x =
                HodgkinHuxley { v_detect: level, substeps: 1, ..HodgkinHuxley::default() };
            for k in 0..20_000u64 {
                if x.advance(0.001, 10.0).expect("no divergence") {
                    return k as f64 * 0.001;
                }
            }
            f64::NAN
        };
        let at_zero = first_spike_ms(0.0);
        assert!(at_zero.is_finite(), "no spike at the default level");
        assert!((at_zero - 1.901).abs() < 1e-3, "the default level reports {at_zero} ms");
        assert!((first_spike_ms(5.0) - at_zero).abs() < 0.02, "+5 mV moved the spike time");
        assert!((first_spike_ms(20.0) - at_zero).abs() < 0.07, "+20 mV moved the spike time");
        assert!((first_spike_ms(-20.0) - at_zero).abs() < 0.09, "-20 mV moved the spike time");
        let span = first_spike_ms(20.0) - first_spike_ms(-20.0);
        assert!(span < 0.16, "40 mV of detection level moved the spike time by {span} ms");
        // It does matter eventually: a level above the peak reports nothing at all, which is a
        // silence the caller has to be able to distinguish from a cell that did not fire. The peak
        // is +40.27 mV, so 41 mV is above everything this cell ever reaches.
        assert!(first_spike_ms(41.0).is_nan(), "a level above the peak reported a spike");
        assert!(first_spike_ms(40.0).is_finite(), "a level just below the peak reported nothing");
    }

    /// **(b)** Gating variables cannot leave `[0,1]`, under drive violent enough to break a naive
    /// integrator, at a step ten times coarser than anyone should use.
    ///
    /// This is a structural property of the exponential update, not a lucky parameter choice: the
    /// new value is `x_inf + (x - x_inf)·d` with `x_inf ∈ [0,1]` and `d ∈ (0,1]`, a convex
    /// combination of two numbers in the interval. The test drives ±500 µA/cm² at 0.5 ms — a step
    /// at which [`Integrator::ForwardEuler`] produces `m > 4` in a single call.
    #[test]
    fn gating_variables_cannot_leave_the_unit_interval_under_violent_drive() {
        let mut x = HodgkinHuxley { substeps: 1, ..HodgkinHuxley::default() };
        for k in 0..4_000u32 {
            let i = if k % 2 == 0 { 500.0 } else { -500.0 };
            x.advance(0.5, i).expect("the exponential integrator cannot diverge");
            for (name, g) in [("m", x.m), ("h", x.h), ("n", x.n)] {
                assert!((0.0..=1.0).contains(&g), "{name} = {g} left [0,1] at step {k}");
            }
            assert!(x.v.is_finite(), "v = {} at step {k}", x.v);
        }
        // The drive has to have actually done something, or the invariant held over nothing.
        assert!(x.m > 0.2 || x.n > 0.5, "the violent drive left the gates near rest: {x:?}");
    }

    /// **(b), the other half.** With no input the membrane cannot leave `[EK, ENa]`, at any step
    /// size, from any legal starting state — because `v_inf` is then a conductance-weighted average
    /// of the three reversal potentials, all of which lie in that band, and the update moves `v`
    /// toward it.
    ///
    /// Randomised over 200 states with the crate's seeded generator, so it is reproducible: same
    /// seed, same 200 states, every platform.
    #[test]
    fn the_potential_cannot_leave_the_reversal_potential_band_with_no_input() {
        let mut rng = Rng::new(0x48_48_31_39_35_32);
        let proto = HodgkinHuxley { substeps: 1, ..HodgkinHuxley::default() };
        for trial in 0..200 {
            let mut x = HodgkinHuxley {
                v: proto.e_k + rng.next_f64() * (proto.e_na - proto.e_k),
                m: rng.next_f64(),
                h: rng.next_f64(),
                n: rng.next_f64(),
                ..proto
            };
            let dt = 1e-4 + rng.next_f64() * 20.0;
            for step in 0..50 {
                x.advance(dt, 0.0).expect("no divergence with zero input");
                assert!(
                    x.v >= proto.e_k - 1e-9 && x.v <= proto.e_na + 1e-9,
                    "trial {trial} step {step}: v = {} outside [{}, {}] at dt = {dt}",
                    x.v,
                    proto.e_k,
                    proto.e_na
                );
            }
        }
    }

    /// **The integrator is the invariant.** Forward Euler at a step it cannot handle drives the
    /// sodium activation gate — a probability — to above four, and [`HodgkinHuxley::advance`] names
    /// it. The exponential update, same state, same step, stays legal. This is the failure mode the
    /// task of "keep `m` in `[0,1]`" is really about, and it is here as a demonstration rather than
    /// as a warning in prose.
    #[test]
    fn forward_euler_leaves_the_unit_interval_where_the_exponential_update_cannot() {
        let mut fe = HodgkinHuxley {
            integrator: Integrator::ForwardEuler,
            substeps: 1,
            v: 50.0,
            ..HodgkinHuxley::default()
        };
        let err = fe.advance(0.5, 0.0).expect_err("forward Euler must blow up here");
        assert_eq!(err, HhError::Diverged);
        assert!(fe.m > 4.0, "m only reached {}", fe.m);

        let mut ee = HodgkinHuxley { substeps: 1, v: 50.0, ..HodgkinHuxley::default() };
        ee.advance(0.5, 0.0).expect("the exponential update stays legal");
        assert!((0.0..=1.0).contains(&ee.m), "m = {}", ee.m);
        assert!(ee.m > 0.9, "m should have nearly saturated at +50 mV, got {}", ee.m);

        // Forward Euler is not broken, only step-limited: at 5 µs it tracks the exponential.
        let a = crossing_ms(0.005, 1, Integrator::ForwardEuler, 10.0).expect("fires");
        let b = crossing_ms(0.005, 1, Integrator::ExponentialEuler, 10.0).expect("fires");
        assert!((a - b).abs() < 0.01, "forward Euler {a} vs exponential {b} at dt = 5 µs");
    }

    /// **(c)** An action potential above rheobase, with a peak and a width in the published range.
    ///
    /// Squid giant axon at 6.3 °C overshoots to roughly +40 mV from a -65 mV rest and the spike is
    /// of order 1-2 ms wide. This implementation measures a peak of +40.27 mV, a width above 0 mV of
    /// 1.165 ms and an after-hyperpolarisation to -75.1 mV under a sustained 10 µA/cm². The
    /// assertions below are the published range, not the measured value, so that a change which
    /// moved the spike out of the biology fails while one that moved it within the biology does not.
    #[test]
    fn an_action_potential_has_a_peak_and_a_width_in_the_published_range() {
        let c = HodgkinHuxley::default();
        let s = c.spike_shape(10.0, 0.005, 20.0, 0.0).expect("10 µA/cm² is above rheobase");
        assert!((30.0..=50.0).contains(&s.peak_mv), "peak {} mV", s.peak_mv);
        assert!((0.5..=2.5).contains(&s.width_ms), "width {} ms", s.width_ms);
        let ahp = s.after_hyperpolarisation_mv.expect("20 ms contains the trough");
        assert!(ahp < -70.0, "AHP {ahp} mV");
        assert!(s.peak_time_ms > s.upstroke_time_ms, "peak before the upstroke crossing");
        assert_eq!(s.level_mv, 0.0);
        // The published figures, tightly, so the docs cannot drift away from the code.
        assert!((s.peak_mv - 40.27).abs() < 0.1, "peak {} vs the documented 40.27", s.peak_mv);
        assert!((s.width_ms - 1.165).abs() < 0.02, "width {} vs the documented 1.165", s.width_ms);

        // The total excursion is the ~100 mV the 1952 paper reports for the squid action potential.
        let excursion = s.peak_mv - ahp;
        assert!((100.0..=125.0).contains(&excursion), "excursion {excursion} mV");
        // Below rheobase there is no shape to measure, and that is a None rather than a flat line.
        assert!(c.spike_shape(1.0, 0.005, 20.0, 0.0).is_none());
    }

    /// The teaching claim, checked — and corrected. The 1952 figure separates **conductances**, and
    /// those do separate: `g_na` peaks 1.48 ms before `g_k`. The **currents** under current clamp do
    /// not, because the sodium driving force collapses exactly when the sodium conductance peaks;
    /// they crest within 0.01 ms of each other. What separates in the currents is which one
    /// dominates: sodium by 4.7x at maximum `dV/dt`, potassium by 500x at the trough.
    #[test]
    fn sodium_leads_the_upstroke_and_potassium_owns_the_trough() {
        let c = HodgkinHuxley::default();
        let dt = 0.001;
        let mut x = c;
        let (mut gna_pk, mut gna_t) = (0.0_f64, 0.0);
        let (mut gk_pk, mut gk_t) = (0.0_f64, 0.0);
        let (mut ina_pk, mut ina_t) = (0.0_f64, 0.0);
        let (mut ik_pk, mut ik_t) = (0.0_f64, 0.0);
        let (mut dvmax, mut ratio_up) = (f64::NEG_INFINITY, 0.0);
        let (mut vmin, mut ratio_dn) = (f64::INFINITY, 0.0);
        for s in 0..15_000u64 {
            let t = s as f64 * dt;
            let i = if t < 0.5 { 30.0 } else { 0.0 };
            let prev = x.v;
            x.advance(dt, i).expect("no divergence");
            let g = x.conductances();
            let cu = x.currents();
            if g.g_na > gna_pk {
                gna_pk = g.g_na;
                gna_t = t;
            }
            if g.g_k > gk_pk {
                gk_pk = g.g_k;
                gk_t = t;
            }
            if -cu.i_na > ina_pk {
                ina_pk = -cu.i_na;
                ina_t = t;
            }
            if cu.i_k > ik_pk {
                ik_pk = cu.i_k;
                ik_t = t;
            }
            if (x.v - prev) / dt > dvmax {
                dvmax = (x.v - prev) / dt;
                ratio_up = -cu.i_na / cu.i_k;
            }
            if t > 2.0 && x.v < vmin {
                vmin = x.v;
                ratio_dn = -cu.i_na / cu.i_k;
            }
        }
        assert!(gna_t < gk_t, "sodium conductance must peak first: {gna_t} vs {gk_t}");
        let sep = gk_t - gna_t;
        assert!((1.0..=2.0).contains(&sep), "conductance peak separation {sep} ms");
        assert!(gna_pk > 25.0 && gna_pk < 40.0, "peak g_na {gna_pk} mS/cm²");
        assert!(gk_pk > 9.0 && gk_pk < 16.0, "peak g_k {gk_pk} mS/cm²");
        // The correction: the current peaks do NOT separate.
        assert!(
            (ik_t - ina_t).abs() < 0.02,
            "the currents peaked {} ms apart, which this implementation did not reproduce",
            ik_t - ina_t
        );
        assert!(ina_pk > 500.0 && ik_pk > 500.0, "peak |INa| {ina_pk}, peak IK {ik_pk}");
        // What does separate: the dominance ratio, by three orders of magnitude.
        assert!(ratio_up > 3.0, "sodium/potassium at max dV/dt = {ratio_up}");
        assert!(ratio_dn < 0.01, "sodium/potassium at the trough = {ratio_dn}");
        assert!(dvmax > 200.0, "max dV/dt {dvmax} mV/ms");
    }

    /// **The threshold emerges, and it is sharp.** Nothing in the model contains a threshold; the
    /// positive feedback between depolarisation and sodium activation makes one. Bisect for the
    /// critical amplitude of a 0.5 ms pulse, then step 1% either side: 1% below leaves the membrane
    /// at -57.7 mV, 1% above reaches +34.8 mV. A 2% change in the stimulus moves the response by
    /// more than 90 mV. That is what "all-or-none" means, measured.
    #[test]
    fn the_threshold_is_all_or_none_and_nothing_in_the_model_imposes_it() {
        let c = HodgkinHuxley::default();
        let fires = |amp: f64| pulse(c, amp, 0.5, 10.0, 0.005).1 > 0;
        // **The bracket is measured, not assumed.** `rheobase_ua_cm2`'s own doc warns that bisecting
        // a wide range reports the wrong boundary when the firing set is an interval rather than a
        // ray, and this test used to bisect [0, 200] on exactly that assumption. Swept at
        // 0.5 µA/cm², the 0.5 ms pulse's firing set flips once — silent below 13.0, firing from 13.5
        // — and never flips back, so a bisection inside that one bracket is entitled to its answer.
        let mut flips = Vec::new();
        let mut firing = fires(0.0);
        let mut amp = 0.5;
        while amp <= 200.0 {
            let f = fires(amp);
            if f != firing {
                flips.push(amp);
                firing = f;
            }
            amp += 0.5;
        }
        assert_eq!(flips.len(), 1, "the pulse firing set is not an interval: flips at {flips:?}");
        assert!((flips[0] - 13.5).abs() < 1e-9, "the scan put the boundary at {}", flips[0]);
        assert!(firing, "the top of the range does not fire, so there is no bracket to bisect");
        let (mut lo, mut hi) = (flips[0] - 0.5, flips[0]);
        for _ in 0..40 {
            let mid = 0.5 * (lo + hi);
            if fires(mid) {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        assert!((hi - 13.279).abs() < 0.01, "critical amplitude {hi} µA/cm²");
        let below = pulse(c, hi * 0.99, 0.5, 10.0, 0.005).0;
        let above = pulse(c, hi * 1.01, 0.5, 10.0, 0.005).0;
        assert!(below < -50.0, "1% below threshold peaked at {below} mV");
        assert!(above > 25.0, "1% above threshold peaked at {above} mV");
        assert!(above - below > 80.0, "the response gap across 2% of stimulus is {}", above - below);
    }

    /// **(e)** The refractory period is not a parameter. [`Neuron::refractory_left`] returns zero at
    /// every instant of this test, and yet an identical second stimulus 11.5 ms after the first
    /// fails while the same stimulus at 12 ms succeeds. The mechanism is visible in the state: at
    /// 10 ms `h` is still 0.514 against its resting 0.596, and `n` is still 0.379 against 0.318 —
    /// sodium is under-available and potassium is over-open.
    #[test]
    fn the_refractory_period_emerges_from_h_and_n() {
        let c = HodgkinHuxley::default();
        let two_pulses = |gap: f64| {
            let mut x = c;
            let dt = 0.005;
            let (mut first, mut second) = (0u32, 0u32);
            for k in 0..(60.0 / dt) as u64 {
                let t = k as f64 * dt;
                let i = if t < 0.5 || (t >= gap && t < gap + 0.5) { 30.0 } else { 0.0 };
                let fired = x.advance(dt, i).expect("no divergence");
                assert_eq!(x.refractory_left(), 0.0, "this model imposes no refractory period");
                if fired {
                    if t < gap {
                        first += 1;
                    } else {
                        second += 1;
                    }
                }
            }
            (first, second)
        };
        assert_eq!(two_pulses(11.5), (1, 0), "the second pulse fired at a 11.5 ms gap");
        assert_eq!(two_pulses(12.0), (1, 1), "the second pulse failed at a 12 ms gap");
        assert_eq!(two_pulses(4.0), (1, 0), "the second pulse fired at a 4 ms gap");

        // And the reason, in the state variables, ten milliseconds after the first spike.
        let rest = steady_state_gates(-65.0);
        let (_, _, x) = pulse(c, 30.0, 0.5, 10.0, 0.005);
        assert!(x.h < rest.h - 0.05, "h recovered to {} (rest {})", x.h, rest.h);
        assert!(x.n > rest.n + 0.03, "n decayed to {} (rest {})", x.n, rest.n);
        assert!(x.v < -68.0, "and the membrane is still hyperpolarised, at {} mV", x.v);
    }

    /// **(f)** The step-size convergence check, on an interpolated crossing time rather than on the
    /// crossing tick — the tick is quantised to `dt`, so a study on it measures quantisation and
    /// reports convergence the integrator has not achieved.
    ///
    /// Measured: halving the step from 0.05 ms down to 0.003125 ms changes the 0 mV crossing by
    /// 4.772e-3, 1.214e-3, 3.017e-4 and 7.258e-5 ms — ratios of **3.933, 4.022 and 4.157**, which is
    /// near second order and is not exactly second order. The doc used to quote "3.8 to 4.0", a
    /// range containing none of the first and last of those; the band asserted below was 3.0 to 5.0
    /// and could not contradict it, so the three ratios are now pinned to 1% as well. **The stated
    /// bound: from `dt = 0.0125 ms` downward the crossing time moves by less than 2 µs in total.**
    #[test]
    fn halving_the_step_converges_at_the_measured_rate() {
        let steps = [0.05, 0.025, 0.0125, 0.00625, 0.003125];
        let times: Vec<f64> = steps
            .iter()
            .map(|&d| crossing_ms(d, 1, Integrator::ExponentialEuler, 10.0).expect("fires"))
            .collect();
        let deltas: Vec<f64> = times.windows(2).map(|w| (w[1] - w[0]).abs()).collect();
        let ratios: Vec<f64> = deltas.windows(2).map(|w| w[0] / w[1]).collect();
        for &ratio in &ratios {
            assert!((3.0..=5.0).contains(&ratio), "convergence ratio {ratio}, deltas {deltas:?}");
        }
        // The band above is the claim "near second order". These are the measured numbers, to 1%,
        // and they are what stops the doc drifting away from the code again: the quoted range used
        // to exclude two of the three.
        for (&got, want) in ratios.iter().zip([3.9325, 4.0224, 4.1568]) {
            assert!((got - want).abs() < 0.04, "ratio {got} vs the measured {want}, all {ratios:?}");
        }
        let total = (times[4] - times[2]).abs();
        assert!(total < 2e-3, "moved {total} ms below dt = 0.0125 ms");
        // The converged value itself, so a change that shifted every step equally still fails.
        assert!((times[4] - 1.9014).abs() < 1e-3, "converged crossing at {} ms", times[4]);
        // And the default four substeps really do buy what their doc claims at dt = 0.1 ms.
        let one = crossing_ms(0.1, 1, Integrator::ExponentialEuler, 10.0).expect("fires");
        let four = crossing_ms(0.1, 4, Integrator::ExponentialEuler, 10.0).expect("fires");
        let converged = crossing_ms(0.0005, 1, Integrator::ExponentialEuler, 10.0).expect("fires");
        let (e1, e4) = ((one - converged).abs(), (four - converged).abs());
        assert!(e1 > 8.0 * e4, "substeps bought only {}x, errors {e1} and {e4}", e1 / e4);
        assert!(e4 < 5e-3, "four substeps at dt = 0.1 ms still {e4} ms off");
    }

    /// Two integrators that share no update rule agree on the spike time to 7 nanoseconds. This is
    /// the strongest statement available about a system with no closed-form solution: an error in
    /// the equations would move both, but an error in the exponential scheme would not move the
    /// Runge-Kutta one.
    #[test]
    fn an_independent_integrator_agrees_on_the_spike_time() {
        let fine = 0.0015625;
        let rk = crossing_ms(fine, 1, Integrator::Rk4, 10.0).expect("fires");
        let ee = crossing_ms(fine, 1, Integrator::ExponentialEuler, 10.0).expect("fires");
        assert!((rk - ee).abs() < 1e-4, "rk4 {rk} vs exponential {ee}");
        // Fourth order against first: `Rk4` at a 6x coarser step still matches.
        let rk_coarse = crossing_ms(0.01, 1, Integrator::Rk4, 10.0).expect("fires");
        assert!((rk_coarse - ee).abs() < 1e-3, "rk4 at dt = 0.01 {rk_coarse} vs {ee}");
        // And they agree on the amplitude, not only the timing.
        let peak = |integ| {
            let mut x =
                HodgkinHuxley { substeps: 1, integrator: integ, ..HodgkinHuxley::default() };
            let mut p = f64::NEG_INFINITY;
            for _ in 0..4_000 {
                x.advance(fine, 10.0).expect("no divergence");
                p = p.max(x.v);
            }
            p
        };
        let (a, b) = (peak(Integrator::Rk4), peak(Integrator::ExponentialEuler));
        assert!((a - b).abs() < 0.01, "peaks {a} and {b}");
    }

    /// **Class 2 excitability**: the firing rate is discontinuous at threshold. At 6.2 µA/cm² this
    /// cell has no firing rate at all; at 6.3 it has one of 52.3 Hz. An integrate-and-fire neuron
    /// rises continuously from zero and cannot do this, and the difference is why a spiking network
    /// built out of `Lif` cannot reproduce a squid axon's input-output curve near threshold however
    /// carefully its parameters are fitted.
    #[test]
    fn the_firing_rate_is_discontinuous_at_threshold() {
        let c = HodgkinHuxley::default();
        assert!(c.firing_rate_hz(6.2, 0.01, 150.0, 300.0).is_none(), "6.2 µA/cm² fired twice");
        let onset = c.firing_rate_hz(6.3, 0.01, 150.0, 300.0).expect("6.3 µA/cm² fires");
        assert!(onset > 40.0, "onset rate {onset} Hz is not a Class 2 jump");
        assert!((onset - 52.3).abs() < 1.0, "onset rate {onset} Hz vs the documented 52.3");
        // Monotone above onset, over the range where the detector can still see the spikes.
        let mut last = onset;
        for &i in &[7.0, 8.0, 10.0, 20.0, 50.0] {
            let r = c.firing_rate_hz(i, 0.01, 150.0, 300.0).expect("above onset");
            assert!(r > last, "rate fell from {last} to {r} between currents");
            last = r;
        }
        assert!(last > 100.0, "the rate only reached {last} Hz at 50 µA/cm²");
    }

    /// The two thresholds that get quoted as one. A step of current makes this cell emit a single
    /// onset spike from 2.24 µA/cm², and a sustained train only from 6.26. Reporting either as "the
    /// rheobase" without saying which is a factor of 2.8.
    #[test]
    fn the_single_spike_threshold_and_the_repetitive_threshold_are_different_numbers() {
        let c = HodgkinHuxley::default();
        let one = c.rheobase_ua_cm2(0.01, 200.0, 40.0, 0.5).expect("something in range fires");
        assert!((one - 2.24).abs() < 0.05, "single-spike threshold {one} µA/cm²");
        assert!(!c.fires(2.2, 0.01, 200.0), "2.2 µA/cm² fired");
        assert!(c.fires(2.3, 0.01, 200.0), "2.3 µA/cm² did not fire");

        let many = c
            .repetitive_onset_ua_cm2(0.01, 150.0, 300.0, 20.0, 1.0)
            .expect("something in range fires twice");
        assert!((many - 6.26).abs() < 0.1, "repetitive onset {many} µA/cm²");
        assert!(many > 2.5 * one, "the two thresholds are only {}x apart", many / one);
    }

    /// **The detector is not the cell, and this is how you tell.** Above about 62 µA/cm² this crate
    /// reports no firing, because the limit cycle stops reaching the 0 mV detection level. The cell
    /// is still oscillating over 40 mV at 100 µA/cm². Genuine depolarisation block — a real fixed
    /// point — arrives near 155, in agreement with the classical figure of about 154 for this
    /// parameter set. An implementation that reported "firing stops at 62" would be reporting its
    /// own detector.
    #[test]
    fn the_reported_firing_band_ends_where_the_detector_stops_not_where_the_cell_does() {
        let c = HodgkinHuxley::default();
        let amp = |i: f64| {
            let (lo, hi) = c.voltage_range_mv(i, 0.01, 300.0, 60.0).expect("no divergence");
            hi - lo
        };
        assert!(amp(10.0) > 100.0, "swing at 10 µA/cm² is {}", amp(10.0));
        // Detector silent, cell emphatically not.
        assert!(c.firing_rate_hz(100.0, 0.01, 300.0, 60.0).is_none(), "the detector saw 100");
        let a100 = amp(100.0);
        assert!(a100 > 30.0, "the cell at 100 µA/cm² only swings {a100} mV");
        // The oscillation shrinks monotonically toward the block.
        let (a60, a150) = (amp(60.0), amp(150.0));
        assert!(a60 > a100 && a100 > a150, "amplitudes {a60}, {a100}, {a150} are not shrinking");
        assert!(a150 < 15.0, "swing at 150 µA/cm² is {a150} mV");
        // And past the bifurcation it is a point, not a small orbit.
        assert!(amp(200.0) < 1e-3, "at 200 µA/cm² the swing is {}, not a fixed point", amp(200.0));
        let (lo, hi) = c.voltage_range_mv(200.0, 0.01, 300.0, 60.0).expect("no divergence");
        assert!((-45.0..=-35.0).contains(&lo), "the blocked cell sits at {lo} mV");
        assert_eq!(lo, hi, "a fixed point has no range at all");

        // **A small swing after a short settle is not a fixed point.** This is the test that tells
        // the two apart, and it is here because the doc on `voltage_range_mv` quoted "0.007 mV at
        // 160 — the last being a fixed point" off a 300 ms settle, and read a bifurcation bracket of
        // 154-to-160 off a transient that had not finished decaying.
        let amp_after = |i: f64, settle: f64| {
            let (lo, hi) = c.voltage_range_mv(i, 0.01, settle, 60.0).expect("no divergence");
            hi - lo
        };
        // 154 µA/cm² is a converged limit cycle: the same swing after 2 s and after 5 s.
        let (a154_2s, a154_5s) = (amp_after(154.0, 2000.0), amp_after(154.0, 5000.0));
        assert!((a154_2s - a154_5s).abs() < 1e-3, "154 has not converged: {a154_2s} then {a154_5s}");
        assert!((a154_5s - 2.7498).abs() < 0.01, "the converged swing at 154 is {a154_5s} mV");
        // 155 is a decaying spiral onto the depolarised fixed point: four orders of magnitude
        // smaller after 5 s than after 2, and still falling.
        let (a155_2s, a155_5s) = (amp_after(155.0, 2000.0), amp_after(155.0, 5000.0));
        assert!(a155_5s * 100.0 < a155_2s, "155 is not collapsing: {a155_2s} then {a155_5s}");
        assert!(a155_5s < 1e-3, "the swing at 155 after 5 s is {a155_5s} mV");
        // So the collapse is between 154 and 155, which is where the classical figure puts it.
        assert!(a154_5s > 1000.0 * a155_5s, "154 {a154_5s} and 155 {a155_5s} are the same thing");
        // And the number that used to be quoted as a fixed point, shown to be a transient.
        let (a160_short, a160_long) = (amp_after(160.0, 300.0), amp_after(160.0, 3000.0));
        assert!((a160_short - 0.00706).abs() < 1e-4, "160 after 300 ms is {a160_short} mV");
        assert!(a160_long < 1e-9, "160 after 3 s is {a160_long} mV");
        assert!(a160_short > 1e6 * a160_long, "the 300 ms number was not a transient after all");
    }

    /// A sub-threshold cell has **no** firing rate. Not a small one, not zero — none, and `None` is
    /// the only honest way to return that. The same discipline as [`crate::neuron::Lif::rate`], and
    /// it matters more here because the Class 2 onset means there are no small rates to interpolate
    /// toward.
    #[test]
    fn a_subthreshold_cell_has_no_firing_rate_rather_than_a_small_one() {
        let c = HodgkinHuxley::default();
        assert!(c.firing_rate_hz(0.0, 0.01, 50.0, 200.0).is_none());
        assert!(c.firing_rate_hz(1.0, 0.01, 50.0, 200.0).is_none());
        assert!(!c.fires(1.0, 0.01, 200.0));
        assert!(c.spike_shape(0.0, 0.01, 50.0, 0.0).is_none());
        // Negative (hyperpolarising) current, likewise, and without divergence.
        assert!(c.firing_rate_hz(-20.0, 0.01, 50.0, 200.0).is_none());
        let (lo, hi) = c.voltage_range_mv(-20.0, 0.01, 200.0, 50.0).expect("no divergence");
        assert_eq!(lo, hi, "the hyperpolarised cell should settle to a point");
        assert!(hi < -65.0, "a hyperpolarising current did not hyperpolarise: {hi} mV");
        // And it settles BELOW `e_k`, at -121 mV. The `[e_k, e_na]` invariant is a **zero-input**
        // property and this is the case that shows why the qualifier is in its statement: with a
        // current injected, `v_inf` is the conductance-weighted average of the reversal potentials
        // PLUS `i_ext / g_total`, and that second term is unbounded.
        assert!(lo < c.e_k, "injected current did not push v past e_k: {lo} vs {}", c.e_k);
        assert!((lo - -121.07).abs() < 0.1, "settled at {lo} mV");
        assert!(lo.is_finite());
    }

    /// Every refusal names the quantity that was wrong, and no bad input reaches the state.
    #[test]
    fn non_finite_inputs_are_refused_by_name() {
        let mut c = HodgkinHuxley::default();
        let before = c;
        assert_eq!(c.advance(f64::NAN, 0.0), Err(HhError::NonFiniteStep));
        assert_eq!(c.advance(f64::INFINITY, 0.0), Err(HhError::NonFiniteStep));
        assert_eq!(c.advance(0.0, 0.0), Err(HhError::NonPositiveStep));
        assert_eq!(c.advance(-0.01, 0.0), Err(HhError::NonPositiveStep));
        assert_eq!(c.advance(0.01, f64::NAN), Err(HhError::NonFiniteCurrent));
        assert_eq!(c.advance(0.01, f64::NEG_INFINITY), Err(HhError::NonFiniteCurrent));
        assert_eq!(c, before, "a refused call moved the state");

        // The trait has nowhere to put an error, so it defends the state and reports no spike.
        assert!(!c.step(f64::NAN, 1e-9));
        assert!(!c.step(1e-5, f64::NAN));
        assert!(!c.step(-1e-5, 1e-9));
        assert_eq!(c, before, "the trait path let a non-finite input through");

        // The messages have to name the quantity, not merely fail.
        assert!(HhError::NonFiniteStep.to_string().contains("time step"));
        assert!(HhError::NonFiniteCurrent.to_string().contains("current"));
        assert!(HhError::Diverged.to_string().contains("[0,1]"));
        assert!(HhError::NonPositiveStep.to_string().contains("negative"));

        // And the reduced model refuses identically.
        let mut r = ReducedHh::default();
        assert_eq!(r.advance(f64::NAN, 0.0), Err(HhError::NonFiniteStep));
        assert_eq!(r.advance(0.0, 0.0), Err(HhError::NonPositiveStep));
        assert_eq!(r.advance(0.01, f64::INFINITY), Err(HhError::NonFiniteCurrent));
    }

    /// The SI boundary converts both directions and loses nothing: with the default 1e-4 cm² patch,
    /// 1 nA is exactly 10 µA/cm², and 2000 steps through the trait's seconds-and-amperes interface
    /// land on **bit-identical** state with 2000 steps through the paper's milliseconds-and-µA/cm²
    /// interface.
    #[test]
    fn the_si_boundary_converts_both_directions_without_loss() {
        let c = HodgkinHuxley::default();
        assert_eq!(1e-9 * 1e6 / c.area_cm2, 10.0, "1 nA is not 10 µA/cm² at the default area");

        let (mut paper, mut si) = (c, c);
        for _ in 0..2_000 {
            paper.advance(0.01, 10.0).expect("no divergence");
            si.step(1e-5, 1e-9);
        }
        assert_eq!(paper.v, si.v, "paper path {} vs SI path {}", paper.v, si.v);
        assert_eq!((paper.m, paper.h, paper.n), (si.m, si.h, si.n));
        assert!(paper.v.is_finite());

        // Volts in, volts out.
        let mut b = c;
        b.bump(7e-3);
        assert!((b.v - (c.v + 7.0)).abs() < 1e-12, "a 7 mV bump moved v to {}", b.v);
        // `potential()` against a number worked out away from the model, not against a copy of its
        // own body: -65 mV plus a 7 mV bump is -58 mV, which is -0.058 V. The line this replaces
        // compared `b.potential()` with `b.v * 1e-3`, which is `potential()`'s entire implementation,
        // so its left-hand side was exactly 0.0 for every possible `b` and its 1e-18 was decoration.
        assert!((b.potential() - -0.058).abs() < 1e-15, "a bumped cell reads {} V", b.potential());
        assert!((c.potential() - -0.065).abs() < 1e-15, "a resting cell reads {} V", c.potential());
        // 7 mV of synaptic input fires this cell from rest and 6.5 mV does not — the same emergent
        // threshold as the current-pulse test, reached through the synaptic interface.
        let fires_from_bump = |mv: f64| {
            let mut x = c;
            x.bump(mv * 1e-3);
            (0..4_000).any(|_| x.advance(0.005, 0.0).expect("no divergence"))
        };
        assert!(fires_from_bump(7.0), "a 7 mV synaptic input did not fire the cell");
        assert!(!fires_from_bump(6.5), "a 6.5 mV synaptic input fired the cell");
    }

    /// `EXACT_OVER_GAPS` is false, and the state says so: one 0.4 ms step and four 0.1 ms steps of
    /// the same zero-input interval end 4.3 mV apart. A model that declared this true would have its
    /// spike times quietly depend on which ticks happened to be quiet.
    #[test]
    fn exact_over_gaps_is_false_and_the_state_proves_it() {
        const { assert!(!HodgkinHuxley::EXACT_OVER_GAPS) };
        const { assert!(!ReducedHh::EXACT_OVER_GAPS) };
        let proto = HodgkinHuxley { v: -50.0, substeps: 1, ..HodgkinHuxley::default() };
        let mut one = proto;
        let mut four = proto;
        one.advance(0.4, 0.0).expect("no divergence");
        for _ in 0..4 {
            four.advance(0.1, 0.0).expect("no divergence");
        }
        let gap = (one.v - four.v).abs();
        assert!(gap > 1.0, "jumping the gap changed v by only {gap} mV");

        // And the reduction, whose declaration was asserted as a constant and demonstrated by
        // nothing. Same experiment, and it is worse: 72 mV apart.
        let rproto = ReducedHh { v: -50.0, substeps: 1, ..ReducedHh::default() };
        let (mut rone, mut rfour) = (rproto, rproto);
        rone.advance(0.4, 0.0).expect("no divergence");
        for _ in 0..4 {
            rfour.advance(0.1, 0.0).expect("no divergence");
        }
        let rgap = (rone.v - rfour.v).abs();
        assert!(rgap > 1.0, "jumping the gap changed the reduction's v by only {rgap} mV");
    }

    /// The crate's simulator enforces the declaration, and the model runs inside a network on the
    /// clocked path. A postsynaptic cell driven only through a 20 mV synapse must spike, or the
    /// `bump` boundary is wrong in a way no single-cell test would catch.
    #[test]
    fn the_simulator_refuses_event_driven_and_runs_it_clocked() {
        let mut b = NetBuilder::new(2);
        b.connect(0, 1, 20e-3, 1).expect("a legal synapse");
        let net = b.build();
        let err = Sim::new(
            net.clone(),
            vec![HodgkinHuxley::default(); 2],
            1e-4,
            Mode::EventDriven,
        )
        .expect_err("event-driven must be refused for a stiff nonlinear model");
        assert_eq!(err, SimError::NotExactOverGaps);

        let mut sim = Sim::new(net, vec![HodgkinHuxley::default(); 2], 1e-5, Mode::Clocked)
            .expect("clocked is legal");
        let train = sim.run(4_000, &[1e-9, 0.0]);
        assert!(!train.of(0).is_empty(), "the driven cell did not fire");
        assert!(!train.of(1).is_empty(), "the synaptically driven cell did not fire");
    }

    /// Same state in, same state out, twice — the crate's determinism rule, checked on the model
    /// with the most arithmetic in it.
    #[test]
    fn the_model_is_deterministic() {
        let run = || {
            let mut x = HodgkinHuxley::default();
            for k in 0..20_000u32 {
                let i = if k % 1000 < 100 { 30.0 } else { 0.0 };
                x.advance(0.01, i).expect("no divergence");
            }
            x
        };
        assert_eq!(run(), run());
    }

    /// `reset` returns the cell to rest with the gates consistent, and `settled` finds the fixed
    /// point of a cell whose parameters have been changed. A modified cell left at -65 mV drifts for
    /// tens of milliseconds at the start of a run, which looks exactly like slow biology.
    #[test]
    fn a_modified_cell_settles_to_its_own_resting_potential() {
        let mut c = HodgkinHuxley { v: 20.0, m: 0.9, h: 0.1, n: 0.8, armed: false, ..Default::default() };
        c.reset();
        assert_eq!(c, HodgkinHuxley::default());

        for (label, cell, want) in [
            ("double leak", HodgkinHuxley { g_leak: 0.6, ..Default::default() }, -63.09),
            ("EK at -90", HodgkinHuxley { e_k: -90.0, ..Default::default() }, -67.86),
        ] {
            let root = cell.rest_potential_mv().expect("bracketed");
            assert!((root - want).abs() < 0.01, "{label}: rest {root} mV, expected {want}");
            let mut s = cell.settled().expect("bracketed");
            let v0 = s.v;
            for _ in 0..10_000 {
                s.advance(0.01, 0.0).expect("no divergence");
            }
            assert!((s.v - v0).abs() < 1e-6, "{label}: settled cell drifted {} mV", s.v - v0);
        }
        // A cell with no root in the bracket refuses rather than returning a bracket endpoint.
        let impossible = HodgkinHuxley { e_leak: 1000.0, e_na: 1000.0, e_k: 1000.0, ..Default::default() };
        assert!(impossible.rest_potential_mv().is_none());
        assert!(impossible.settled().is_none());
    }

    /// The Rinzel reduction's central approximation, measured against the full model rather than
    /// asserted. Along a full action potential the residual `|h - (0.89 - 1.1n)|` reaches 0.0648 and
    /// falls to 0.00018: good to about 6% of the unit interval, and **not** an identity. Both bounds
    /// are asserted, because a residual of zero would mean the test was comparing the reduction with
    /// itself.
    #[test]
    fn the_h_from_n_approximation_is_good_but_is_not_an_identity() {
        let mut x = HodgkinHuxley::default();
        let (mut worst, mut best) = (0.0_f64, f64::INFINITY);
        for k in 0..5_000u64 {
            let t = k as f64 * 0.005;
            let i = if t < 0.5 { 30.0 } else { 0.0 };
            x.advance(0.005, i).expect("no divergence");
            let res = (x.h - (0.89 - 1.1 * x.n)).abs();
            worst = worst.max(res);
            best = best.min(res);
        }
        assert!(worst < 0.10, "the affine relation is off by {worst} somewhere on the spike");
        assert!(worst > 0.01, "a residual of {worst} means this test compared nothing");
        assert!(best < 0.01, "the relation is never better than {best}, so it is not a fit at all");
        assert!((worst - 0.0648).abs() < 0.005, "worst residual {worst} vs the documented 0.0648");
    }

    /// The reduced model must fire, must rest at its own fixed point, and must be **wrong in the
    /// ways its doc says**. A reduction whose costs cannot be measured is a reduction nobody can
    /// decide to accept.
    #[test]
    fn the_reduction_fires_and_pays_the_documented_price() {
        let r = ReducedHh::default();
        let full = HodgkinHuxley::default();

        // Its own fixed point, 0.1 mV from the full model's.
        let root = r.rest_potential_mv().expect("bracketed");
        assert!((root - -65.098).abs() < 0.01, "reduced rest {root} mV");
        assert_eq!(r.v, root, "the default is not at its own fixed point");
        let full_root = full.rest_potential_mv().expect("bracketed");
        assert!((root - full_root).abs() > 0.05, "the reduction moved rest by nothing at all");
        assert!((r.h() - 0.5422).abs() < 1e-3, "h at the reduced rest is {}", r.h());

        // It fires, and it overshoots: +47.98 mV against the full model's +40.32.
        let mut x = r;
        let (mut peak, mut spikes, mut up, mut down) = (f64::NEG_INFINITY, 0u32, None, None);
        for k in 0..15_000u64 {
            let t = k as f64 * 0.001;
            let i = if t < 0.5 { 30.0 } else { 0.0 };
            let prev = x.v;
            if x.advance(0.001, i).expect("no divergence") {
                spikes += 1;
            }
            if up.is_none() && prev < 0.0 && x.v >= 0.0 {
                up = Some(t);
            }
            if up.is_some() && down.is_none() && prev >= 0.0 && x.v < 0.0 {
                down = Some(t);
            }
            peak = peak.max(x.v);
            assert!((0.0..=1.0).contains(&x.n), "n = {} left [0,1]", x.n);
        }
        assert_eq!(spikes, 1, "the reduced model fired {spikes} times on one pulse");
        assert!((peak - 47.98).abs() < 0.2, "reduced peak {peak} mV");
        let width = down.expect("came back down") - up.expect("went up");
        assert!((width - 1.04).abs() < 0.05, "reduced width {width} ms");

        // The documented costs, as differences rather than as prose.
        let fullshape = full.spike_shape(30.0, 0.001, 20.0, 0.0).expect("fires");
        assert!(peak - fullshape.peak_mv > 5.0, "the reduction only overshot by {}", peak - fullshape.peak_mv);
        assert!(width < fullshape.width_ms, "the reduced spike is not narrower");

        // Its threshold moved too: repetitive firing from 5.0 µA/cm² where the full model needs 6.3.
        let reduced_fires = |i: f64| {
            let mut y = r;
            let mut n = 0u32;
            for _ in 0..40_000 {
                if y.advance(0.01, i).expect("no divergence") {
                    n += 1;
                }
            }
            n
        };
        assert!(reduced_fires(4.5) <= 1, "the reduced model fired repetitively at 4.5 µA/cm²");
        assert!(reduced_fires(5.0) > 5, "the reduced model did not fire at 5.0 µA/cm²");
        assert!(full.firing_rate_hz(5.0, 0.01, 150.0, 300.0).is_none(), "the full model fires at 5.0");
    }

    /// **Four public methods had no test at all, and two of them carried numbers that were wrong.**
    /// [`Rates::time_constants_ms`] is the first: mutating its `1.0/(alpha+beta)` to `2.0/(...)`
    /// passed the whole module. The peaks are swept at 1 µV here because the two documented ones
    /// were not merely imprecise — `tau_h` was said to peak "near 8-9 ms around -50 mV" and it peaks
    /// at 8.582 ms at -66.8 mV, where -50 mV is already down to 4.641; `tau_n` was said to peak
    /// "near 5.6 ms around -55 mV" and it peaks at 5.792 ms at -77.2 mV, with -55 mV, which is
    /// `alpha_n`'s singular voltage rather than its slowest one, at 4.755.
    #[test]
    fn the_gate_time_constants_are_separated_and_peak_where_this_implementation_says() {
        // **The meaning, measured independently of the expression.** A gate held at a fixed voltage
        // from x = 0 reaches `x_inf·(1 - 1/e)` after exactly one time constant, so integrating the
        // gate's own ODE by hand — not through any integrator in this module — and timing that
        // crossing measures `tau` without recomputing `1/(alpha+beta)`.
        let dt = 1e-5;
        for &v in &[-80.0, -65.0, -40.0, 0.0] {
            let r = rates(v);
            let inf = r.steady_state();
            let taus = r.time_constants_ms();
            for (name, alpha, beta, x_inf, tau) in [
                ("m", r.alpha_m, r.beta_m, inf.m, taus.tau_m),
                ("h", r.alpha_h, r.beta_h, inf.h, taus.tau_h),
                ("n", r.alpha_n, r.beta_n, inf.n, taus.tau_n),
            ] {
                let target = x_inf * (1.0 - (-1.0_f64).exp());
                let (mut x, mut t) = (0.0_f64, 0.0_f64);
                while x < target {
                    x += dt * (alpha * (1.0 - x) - beta * x);
                    t += dt;
                    assert!(t < 100.0, "{name} at {v} mV never reached 1 - 1/e of its steady state");
                }
                assert!(
                    (t - tau).abs() < 2e-3,
                    "{name} at {v} mV relaxed in {t} ms, tau_{name} says {tau} ms"
                );
            }
        }

        // At rest, the three numbers the `time_constants_ms` doc quotes, and the separation that is
        // the reason [`ReducedHh`] exists.
        let r = rates(-65.0).time_constants_ms();
        assert!((r.tau_m - 0.236_767).abs() < 1e-5, "tau_m at rest {}", r.tau_m);
        assert!((r.tau_h - 8.516_011).abs() < 1e-5, "tau_h at rest {}", r.tau_h);
        assert!((r.tau_n - 5.458_585).abs() < 1e-5, "tau_n at rest {}", r.tau_n);
        assert!((r.tau_n / r.tau_m - 23.054).abs() < 0.01, "tau_n/tau_m = {}", r.tau_n / r.tau_m);
        assert!((r.tau_h / r.tau_m - 35.968).abs() < 0.01, "tau_h/tau_m = {}", r.tau_h / r.tau_m);
        assert!((rates(50.0).time_constants_ms().tau_m - 0.111_015).abs() < 1e-5);

        // The peaks, swept at 1 µV over everything the model visits.
        let mut peak = Taus { tau_m: 0.0, tau_h: 0.0, tau_n: 0.0 };
        let mut arg = Taus { tau_m: 0.0, tau_h: 0.0, tau_n: 0.0 };
        for k in 0..=180_000 {
            let v = -120.0 + f64::from(k) * 0.001;
            let t = rates(v).time_constants_ms();
            if t.tau_m > peak.tau_m {
                peak.tau_m = t.tau_m;
                arg.tau_m = v;
            }
            if t.tau_h > peak.tau_h {
                peak.tau_h = t.tau_h;
                arg.tau_h = v;
            }
            if t.tau_n > peak.tau_n {
                peak.tau_n = t.tau_n;
                arg.tau_n = v;
            }
        }
        assert!((peak.tau_h - 8.5824).abs() < 1e-3, "tau_h peaks at {} ms", peak.tau_h);
        assert!((arg.tau_h - -66.81).abs() < 0.01, "tau_h peaks at {} mV", arg.tau_h);
        assert!((peak.tau_n - 5.7923).abs() < 1e-3, "tau_n peaks at {} ms", peak.tau_n);
        assert!((arg.tau_n - -77.17).abs() < 0.01, "tau_n peaks at {} mV", arg.tau_n);
        assert!((peak.tau_m - 0.5014).abs() < 1e-3, "tau_m peaks at {} ms", peak.tau_m);
        assert!((arg.tau_m - -38.84).abs() < 0.01, "tau_m peaks at {} mV", arg.tau_m);
        // `tau_m` stays sub-millisecond everywhere, which is the reduction's first premise.
        assert!(peak.tau_m < 1.0, "tau_m reaches {} ms somewhere", peak.tau_m);
        // The two voltages the docs used to name are nowhere near the peaks, by a factor this
        // assertion would have failed on before they were corrected.
        let at_50 = rates(-50.0).time_constants_ms().tau_h;
        let at_55 = rates(-55.0).time_constants_ms().tau_n;
        assert!((at_50 - 4.6406).abs() < 1e-3, "tau_h at -50 mV is {at_50} ms");
        assert!((at_55 - 4.7548).abs() < 1e-3, "tau_n at -55 mV is {at_55} ms");
        assert!(peak.tau_h > at_50 * 1.8, "tau_h at -50 mV is not far from its peak after all");
        assert!(peak.tau_n > at_55 * 1.2, "tau_n at -55 mV is not far from its peak after all");
    }

    /// [`HodgkinHuxley::membrane_time_constant_ms`] had no test either: doubling its numerator
    /// passed the module. The check that kills that is the one cell where `C/g` is exact.
    #[test]
    fn the_membrane_time_constant_is_capacitance_over_conductance_and_swings_54_fold() {
        // With both gated conductances switched off the membrane is a pure RC: tau = 1/0.3 ms, and a
        // displacement must decay by exactly 1/e over one tau under the model's own integrator.
        // Nothing about `m`, `h` or `n` enters, so this is the definition and not a restatement of
        // the expression.
        let rc =
            HodgkinHuxley { g_na: 0.0, g_k: 0.0, v: -44.4, substeps: 1, ..Default::default() };
        let tau = rc.membrane_time_constant_ms();
        assert!((tau - 1.0 / 0.3).abs() < 1e-12, "the pure-RC time constant is {tau} ms");
        let mut x = rc;
        x.advance(tau, 0.0).expect("no divergence");
        let left = (x.v - rc.e_leak) / (rc.v - rc.e_leak);
        assert!((left - (-1.0_f64).exp()).abs() < 1e-12, "one tau left {left} of the displacement");
        // Two taus leave 1/e².
        x.advance(tau, 0.0).expect("no divergence");
        let left2 = (x.v - rc.e_leak) / (rc.v - rc.e_leak);
        assert!((left2 - (-2.0_f64).exp()).abs() < 1e-12, "two taus left {left2}");

        // The documented numbers for the real cell, and the swing that is its stiffness.
        let c = HodgkinHuxley::default();
        assert!((c.membrane_time_constant_ms() - 1.4766).abs() < 1e-3);
        let mut y = c;
        let mut fastest = f64::INFINITY;
        for k in 0..20_000u64 {
            let i = if (k as f64) * 0.001 < 0.5 { 30.0 } else { 0.0 };
            y.advance(0.001, i).expect("no divergence");
            fastest = fastest.min(y.membrane_time_constant_ms());
        }
        assert!((fastest - 0.027_068).abs() < 1e-5, "the fastest is {fastest} ms");
        let swing = c.membrane_time_constant_ms() / fastest;
        assert!((swing - 54.55).abs() < 0.05, "the stiffness swing is {swing}, not 54.5");
        // A cell with no ionic path at all is +inf, which the doc now says instead of claiming the
        // result is always positive because `g_leak > 0` in a type that does not require it.
        let dead = HodgkinHuxley { g_na: 0.0, g_k: 0.0, g_leak: 0.0, ..Default::default() };
        assert_eq!(dead.membrane_time_constant_ms(), f64::INFINITY);
    }

    /// [`HodgkinHuxley::at`] is the doc's "right way to initialise a voltage-clamp experiment" and
    /// [`HodgkinHuxley::steady_state_here`] reports what it set; neither had a test, and mutating
    /// `at`'s gates to zero or shifting `steady_state_here` by 10 mV passed the module.
    #[test]
    fn a_voltage_clamped_cell_starts_at_the_steady_state_of_its_own_voltage() {
        for &v in &[-90.0, -65.0, -55.0, -40.0, 0.0, 30.0] {
            let cell = HodgkinHuxley::at(v);
            assert_eq!(cell.v, v, "at({v}) is not at {v}");
            // **The property, not the expression**: every gate derivative is zero there. Comparing
            // the gates against `steady_state_gates(v)` would be the code agreeing with itself.
            let r = rates(v);
            for (name, alpha, beta, x) in [
                ("m", r.alpha_m, r.beta_m, cell.m),
                ("h", r.alpha_h, r.beta_h, cell.h),
                ("n", r.alpha_n, r.beta_n, cell.n),
            ] {
                let d = alpha * (1.0 - x) - beta * x;
                assert!(d.abs() < 1e-12, "d{name}/dt = {d} at {v} mV, so {name} is not settled");
                assert!((0.0..=1.0).contains(&x), "{name} = {x} at {v} mV");
            }
            // And `steady_state_here` reports that state rather than a shifted one.
            let g = cell.steady_state_here();
            assert_eq!((g.m, g.h, g.n), (cell.m, cell.h, cell.n), "steady_state_here at {v} mV");
        }
        // At rest it is exactly the default cell, gates included.
        assert_eq!(HodgkinHuxley::at(-65.0), HodgkinHuxley::default());
        // The gates really do move with the voltage: sodium activation runs from 0.002 to 0.997
        // across the same sweep, so the zero-derivative check above is not being satisfied by a
        // constant.
        assert!((HodgkinHuxley::at(-90.0).m - 0.002_110).abs() < 1e-5);
        assert!((HodgkinHuxley::at(30.0).m - 0.997_095).abs() < 1e-5);
        assert!(HodgkinHuxley::at(30.0).h < 0.001, "h should be inactivated at +30 mV");
        // Everything else is the default: `at` changes the state, not the parameters.
        let a = HodgkinHuxley::at(-40.0);
        let d = HodgkinHuxley::default();
        assert_eq!((a.c_m, a.g_na, a.g_k, a.g_leak), (d.c_m, d.g_na, d.g_k, d.g_leak));
        assert_eq!(
            (a.area_cm2, a.substeps, a.v_detect, a.armed),
            (d.area_cm2, d.substeps, d.v_detect, d.armed)
        );
    }

    /// The reduction's invariants are the full model's: `n` cannot leave `[0,1]` and, with no input,
    /// `v` cannot leave the reversal-potential band — at any step size, because it uses the same
    /// exponential update.
    #[test]
    fn the_reduction_keeps_the_same_invariants() {
        let mut rng = Rng::new(0x52_69_6e_7a_65_6c);
        let proto = ReducedHh { substeps: 1, ..ReducedHh::default() };
        for trial in 0..100 {
            let mut x = ReducedHh {
                v: proto.e_k + rng.next_f64() * (proto.e_na - proto.e_k),
                n: rng.next_f64(),
                ..proto
            };
            let dt = 1e-4 + rng.next_f64() * 10.0;
            for step in 0..50 {
                x.advance(dt, 0.0).expect("no divergence with zero input");
                assert!((0.0..=1.0).contains(&x.n), "trial {trial} step {step}: n = {}", x.n);
                assert!(
                    x.v >= proto.e_k - 1e-9 && x.v <= proto.e_na + 1e-9,
                    "trial {trial} step {step}: v = {}",
                    x.v
                );
                assert!((0.0..=1.0).contains(&x.h()), "h from n is {}", x.h());
            }
        }
        // Violent drive, coarse step, same result.
        let mut x = ReducedHh { substeps: 1, ..ReducedHh::default() };
        for k in 0..2_000u32 {
            let i = if k % 2 == 0 { 500.0 } else { -500.0 };
            x.advance(0.5, i).expect("no divergence");
            assert!((0.0..=1.0).contains(&x.n), "n = {} at step {k}", x.n);
        }
    }

    /// **A spike is a crossing, and the detector used to test a level.** A cell handed over above
    /// `v_detect` — through [`HodgkinHuxley::at`], through a large `bump`, or field by field — was
    /// reported as spiking on its first step while its membrane was on the way *down*.
    #[test]
    fn the_detector_reports_a_crossing_and_not_a_level() {
        let mut high = HodgkinHuxley::at(10.0);
        let before = high.v;
        let fired = high.advance(0.001, 0.0).expect("no divergence");
        assert!(high.v < before, "this membrane should be falling: {before} -> {}", high.v);
        assert!(!fired, "a falling membrane reported a spike");
        assert!(high.armed, "and it is still waiting for a spike that has not happened");

        // The same through the synaptic boundary, which is how a network reaches it.
        let mut bumped = HodgkinHuxley::default();
        bumped.bump(80e-3);
        assert!((bumped.v - 15.0).abs() < 1e-12, "an 80 mV bump put v at {}", bumped.v);
        assert!(
            !bumped.advance(0.001, 0.0).expect("no divergence"),
            "the bump reported a spike by itself"
        );

        // It is not deaf afterwards. Let it fall back to rest and drive it, and the crossings it
        // does make are reported.
        let mut later = HodgkinHuxley::at(10.0);
        for _ in 0..20_000 {
            later.advance(0.005, 0.0).expect("no divergence");
        }
        assert!((later.v - -65.0).abs() < 0.2, "it did not return to rest: {} mV", later.v);
        let mut spikes = 0u32;
        for _ in 0..20_000 {
            if later.advance(0.005, 10.0).expect("no divergence") {
                spikes += 1;
            }
        }
        assert!(spikes > 3, "only {spikes} spikes on 100 ms of suprathreshold drive");

        // Genuine detection is untouched: 21 spikes in 300 ms at 10 µA/cm². And the hysteresis is
        // not what produces that count — `detect_reset` at the detection level gives the same 21,
        // because a squid upstroke crosses 0 mV exactly once, which is what the `detect_reset` doc
        // now claims and used to claim the opposite of.
        let count = |reset: f64| {
            let mut x = HodgkinHuxley { detect_reset: reset, ..HodgkinHuxley::default() };
            let mut n = 0u32;
            for _ in 0..30_000 {
                if x.advance(0.01, 10.0).expect("no divergence") {
                    n += 1;
                }
            }
            n
        };
        assert_eq!(count(-20.0), 21, "the default cell's spike count moved");
        assert_eq!(count(0.0), 21, "the hysteresis turns out to be load-bearing after all");

        // And the reduction shares the detector rather than carrying a second copy of it.
        let mut r = ReducedHh { v: 10.0, ..ReducedHh::default() };
        assert!(!r.advance(0.001, 0.0).expect("no divergence"), "the reduction reported a level");
    }

    /// **A diverged run has no verdict, and folding it into `false` cost a factor of 4.3.**
    /// `Integrator::ForwardEuler` at `dt = 0.5 ms` reports a 0 mV crossing at 0.5214 µA/cm² from a
    /// state that is still legal — its `m` gate is oscillating with a growing amplitude — and is
    /// illegal one step later. [`HodgkinHuxley::rheobase_ua_cm2`] used to believe it.
    #[test]
    fn a_run_that_diverges_has_no_verdict_about_whether_the_cell_fires() {
        let fe =
            HodgkinHuxley { integrator: Integrator::ForwardEuler, substeps: 1, ..Default::default() };
        assert_eq!(fe.fires_checked(0.5214, 0.5, 50.0), Err(HhError::Diverged));
        assert!(!fe.fires(0.5214, 0.5, 50.0), "a diverged run answered yes");
        assert_eq!(fe.rheobase_ua_cm2(0.5, 50.0, 40.0, 0.5), None, "forward Euler answered anyway");

        // The default integrator cannot diverge, so it answers — and answers 2.2493, four times the
        // number the unstable integrator produced.
        let c = HodgkinHuxley::default();
        let r = c.rheobase_ua_cm2(0.5, 50.0, 40.0, 0.5).expect("the default cannot diverge");
        assert!((r - 2.2493).abs() < 0.01, "rheobase {r} µA/cm²");
        assert!(r > 4.0 * 0.5214, "the two answers are not far apart after all: {r}");

        // `fires` and `fires_checked` agree wherever there is anything to agree about.
        assert_eq!(c.fires_checked(10.0, 0.01, 50.0), Ok(true));
        assert_eq!(c.fires_checked(1.0, 0.01, 50.0), Ok(false));
        assert!(c.fires(10.0, 0.01, 50.0) && !c.fires(1.0, 0.01, 50.0));

        // Bad inputs are named rather than turned into a step count.
        assert_eq!(c.fires_checked(10.0, 0.0, 50.0), Err(HhError::NonPositiveStep));
        assert_eq!(c.fires_checked(10.0, f64::NAN, 50.0), Err(HhError::NonFiniteStep));
        assert_eq!(c.fires_checked(f64::NAN, 0.01, 50.0), Err(HhError::NonFiniteCurrent));
        assert!(!c.fires(10.0, 0.0, 50.0), "a zero step reported firing");
    }

    /// **A `NaN` parameter must not come back as a resting potential.** Every comparison against
    /// `NaN` is false, so the reduced model's copy of the bisection — which was missing the
    /// finiteness guard the full model had — ran 200 iterations on `NaN`, took the same branch every
    /// time and returned the bracket endpoint. A cell then sat at -40.0 mV as though that were rest.
    #[test]
    fn a_non_finite_parameter_is_refused_rather_than_answered_with_a_bracket_endpoint() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let r = ReducedHh { e_leak: bad, ..Default::default() };
            assert_eq!(r.rest_potential_mv(), None, "the reduction answered for e_leak = {bad}");
            assert!(r.settled().is_none(), "and settled it");
            let f = HodgkinHuxley { e_leak: bad, ..Default::default() };
            assert_eq!(f.rest_potential_mv(), None, "the full model answered for e_leak = {bad}");
            assert!(f.settled().is_none());
        }
        // `reset` falls back to -65 mV rather than to the endpoint.
        let mut r = ReducedHh { e_leak: f64::NAN, v: 0.0, n: 0.9, armed: false, ..Default::default() };
        r.reset();
        assert_eq!(r.v, -65.0, "reset put the cell at {} mV", r.v);
        assert_eq!(r.n, steady_state_gates(-65.0).n);
        assert!(r.armed);
        // And the legal cells still answer, at the same roots as before the two loops became one.
        let full = HodgkinHuxley::default().rest_potential_mv().expect("bracketed");
        let red = ReducedHh::default().rest_potential_mv().expect("bracketed");
        assert!((full - -64.999_722_433_734_58).abs() < 1e-12, "full rest moved to {full}");
        assert!((red - -65.097_766_252_847_98).abs() < 1e-12, "reduced rest moved to {red}");
        assert_eq!(ReducedHh::default().v, red, "the default is not at its own fixed point");
    }

    /// **The reduction can diverge, its doc said it could not, and its guard was eight orders of
    /// magnitude stricter than the full model's.** `n` is a public field; the guard exists for the
    /// state a caller hands over, not for the one the integrator produces.
    #[test]
    fn the_reduced_model_can_diverge_and_its_guard_is_the_full_model_s() {
        let mut wild = ReducedHh { n: 1.5, ..Default::default() };
        assert_eq!(wild.advance(0.01, 0.0), Err(HhError::Diverged), "n = 1.5 was accepted");
        let mut negative = ReducedHh { n: -0.5, ..Default::default() };
        assert_eq!(negative.advance(0.01, 0.0), Err(HhError::Diverged), "n = -0.5 was accepted");

        // The tolerance, at a step short enough that the update cannot pull the gate back inside on
        // its own: 1e-10 outside is accepted and 1e-8 outside is refused, by BOTH models. The
        // reduced check used to be an exact `(0.0..=1.0)`, which refused all four of these.
        for &n in &[1.0 + 1e-10, -1e-10] {
            let mut r = ReducedHh { n, ..Default::default() };
            assert_eq!(r.advance(1e-11, 0.0), Ok(false), "the reduction refused n = {n}");
        }
        for &n in &[1.0 + 1e-8, -1e-8] {
            let mut r = ReducedHh { n, ..Default::default() };
            assert_eq!(r.advance(1e-11, 0.0), Err(HhError::Diverged), "the reduction took n = {n}");
        }
        let mut ok = HodgkinHuxley { m: 1.0 + 1e-10, substeps: 1, ..Default::default() };
        assert_eq!(ok.advance(1e-9, 0.0), Ok(false), "the full model refused m = 1 + 1e-10");
        let mut bad = HodgkinHuxley { m: 1.0 + 1e-8, substeps: 1, ..Default::default() };
        assert_eq!(bad.advance(1e-9, 0.0), Err(HhError::Diverged), "the full model took m = 1 + 1e-8");
    }

    /// **The only state guard's tolerance, pinned at both ends.** Widening `LO` from -1e-9 to -1e-1
    /// passed the whole module, so `advance`'s contract — "a caller who gets `Ok` has gates that are
    /// occupancies" — was pinned only up to a number that could move by eight orders of magnitude
    /// unobserved. The guard is private, this test is in the module, and there is no reason to check
    /// it through a simulation that might pull the value back inside before the check runs.
    #[test]
    fn the_state_guard_is_a_tolerance_and_this_is_exactly_where_it_sits() {
        assert_eq!(GATE_SLACK, 1e-9, "the documented slack");
        assert!(gate_is_legal(0.0) && gate_is_legal(1.0) && gate_is_legal(0.5));
        assert!(gate_is_legal(GATE_SLACK.mul_add(-1.0, 0.0)), "exactly -GATE_SLACK is legal");
        assert!(gate_is_legal(1.0 + GATE_SLACK), "exactly 1 + GATE_SLACK is legal");
        assert!(gate_is_legal(-1e-10) && gate_is_legal(1.0 + 1e-10), "well inside the slack");
        assert!(!gate_is_legal(-1e-8), "a hundredth of a part per million below zero is not a gate");
        assert!(!gate_is_legal(1.0 + 1e-8));
        assert!(!gate_is_legal(-2e-9) && !gate_is_legal(1.0 + 2e-9), "twice the slack");
        assert!(!gate_is_legal(f64::NAN), "NaN compares false against everything, including this");
        assert!(!gate_is_legal(f64::INFINITY) && !gate_is_legal(f64::NEG_INFINITY));

        // And every field is actually consulted, one at a time.
        let g = steady_state_gates(-65.0);
        let cell = |m: f64, h: f64, n: f64, v: f64| {
            HodgkinHuxley { m, h, n, v, ..HodgkinHuxley::default() }.state_is_legal()
        };
        assert!(cell(g.m, g.h, g.n, -65.0), "the default state is illegal");
        assert!(!cell(1.0 + 1e-8, g.h, g.n, -65.0), "m was not checked");
        assert!(!cell(g.m, -1e-8, g.n, -65.0), "h was not checked");
        assert!(!cell(g.m, g.h, 1.0 + 1e-8, -65.0), "n was not checked");
        assert!(!cell(g.m, g.h, g.n, f64::NAN), "v was not checked");
        assert!(!cell(g.m, g.h, g.n, f64::INFINITY));
        // A potential of ±10^6 mV is legal: the guard is about finiteness and occupancies, and the
        // `[e_k, e_na]` band is a zero-input property of the integrator rather than a state check.
        assert!(cell(g.m, g.h, g.n, 1e6));
    }

    /// **`Neuron::step` could write `NaN` into a membrane, and its doc said it did not need to
    /// check.** Every conductance is public; all three at zero makes the exponential update's
    /// `v_inf` a `0/0`. The trait has nowhere to put an error, so the step is rolled back.
    #[test]
    fn a_step_that_would_write_a_non_finite_state_is_rolled_back_rather_than_taken() {
        let dead = HodgkinHuxley { g_na: 0.0, g_k: 0.0, g_leak: 0.0, ..Default::default() };
        let mut x = dead;
        assert!(!x.step(1e-5, 0.0), "a rolled-back step is not a spike");
        assert!(x.v.is_finite(), "v = {} is the whole reason this guard exists", x.v);
        assert_eq!(x, dead, "the illegal step was written anyway");
        // The checked path says why, in a variant that names the quantity.
        let mut y = dead;
        assert_eq!(y.advance(0.01, 0.0), Err(HhError::Diverged));
        // The reduction has the same public fields and the same guard.
        let rdead = ReducedHh { g_na: 0.0, g_k: 0.0, g_leak: 0.0, ..Default::default() };
        let mut r = rdead;
        assert!(!r.step(1e-5, 0.0));
        assert_eq!(r, rdead, "the reduction wrote its illegal step");
        assert!(r.v.is_finite());
        // Forward Euler at a step it cannot handle, through the trait: frozen, not `NaN`.
        let fe = HodgkinHuxley {
            integrator: Integrator::ForwardEuler,
            substeps: 1,
            v: 50.0,
            ..Default::default()
        };
        let mut f = fe;
        assert!(!f.step(5e-4, 0.0), "forward Euler reported a spike on a step it cannot take");
        assert_eq!(f, fe, "and it wrote an m of 4.3 into the state");
        // The guard refuses illegal outcomes, not every outcome: a legal cell still moves.
        let mut live = HodgkinHuxley::default();
        assert!(!live.step(1e-5, 1e-9), "no spike is expected on the first tick");
        assert!(live.v > -65.0, "a legal step must still move the membrane, v = {}", live.v);
        assert!(live.v < -64.0, "and not by much in 10 µs, v = {}", live.v);
    }

    /// The times in a [`SpikeShape`] are the times of the samples they measured. They were reported
    /// one step early — `k * dt` is the time *before* the step whose result is being tested — which
    /// no assertion here could see, because `width_ms` is a difference of two of them and cancels.
    #[test]
    fn spike_shape_reports_the_times_of_the_samples_it_measured() {
        let c = HodgkinHuxley::default();
        let dt = 0.005;
        let s = c.spike_shape(10.0, dt, 20.0, 0.0).expect("10 µA/cm² is above rheobase");

        // Replay the same run by hand, counting the state after step k as being at (k+1)·dt.
        let mut x = c;
        let (mut peak, mut peak_t) = (f64::NEG_INFINITY, f64::NAN);
        let mut up_t = None;
        for k in 0..(20.0 / dt) as u64 {
            let prev = x.v;
            x.advance(dt, 10.0).expect("no divergence");
            let t = (k + 1) as f64 * dt;
            if up_t.is_none() && prev < 0.0 && x.v >= 0.0 {
                up_t = Some(t);
            }
            if x.v > peak {
                peak = x.v;
                peak_t = t;
            }
        }
        assert_eq!(s.peak_mv, peak, "the peak value itself");
        assert_eq!(s.peak_time_ms, peak_t, "the peak's time");
        assert_eq!(s.upstroke_time_ms, up_t.expect("it crossed"), "the upstroke's time");

        // Which lands one step later than it used to, and is a whole number of steps.
        assert!((s.peak_time_ms - 2.140).abs() < 1e-9, "peak at {} ms", s.peak_time_ms);
        assert!((s.upstroke_time_ms - 1.905).abs() < 1e-9, "upstroke at {} ms", s.upstroke_time_ms);
        let ticks = s.peak_time_ms / dt;
        assert!((ticks - ticks.round()).abs() < 1e-6, "{ticks} is not a whole number of steps");
        // The width is a difference of two such times and did not move when they did.
        assert!((s.width_ms - 1.165).abs() < 1e-9, "width {} ms", s.width_ms);
        // The peak is the top of the excursion, so the sample one step earlier is lower — which is
        // what makes "one step early" a wrong answer rather than a convention.
        let mut y = c;
        for _ in 0..(s.peak_time_ms / dt).round() as u64 - 1 {
            y.advance(dt, 10.0).expect("no divergence");
        }
        assert!(y.v < s.peak_mv, "the sample before the peak is {} mV", y.v);
    }
}
