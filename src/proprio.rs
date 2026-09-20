//! Proprioception: what a muscle tells the spinal cord about its own length, speed and force, and
//! what happens when that report comes back late — the power-law spindle and the fusimotor gains
//! that retune it, an intrafusal fibre with its gamma drive, the tendon organ, an exact
//! rate-to-spike encoder, and a reflex loop whose delay sets the gain at which it rings.
//!
//! # What the mechanism is
//!
//! Two receptors report the mechanical state of a muscle. **Muscle spindles** lie in parallel with
//! the fibres and respond to stretch: their primary (Ia) afferents fire with the muscle's length
//! and, much more strongly, with its lengthening VELOCITY — not linearly but as a power of it, so
//! that they stay sensitive to slow movement without saturating in fast. **Golgi tendon organs**
//! lie in series and report force. Prochazka and Gorassini (*Models of ensemble firing of muscle
//! spindle afferents recorded during normal locomotion in cats*, Journal of Physiology
//! 507(1):277–291, 1998) fitted a family of such models to Ia afferents recorded in walking cats and
//! found the fit dominated by a velocity term with an exponent of 0.5 to 0.6.
//!
//! A spindle is not a fixed sensor. Its intrafusal fibres have their own motor supply — the
//! **gamma (fusimotor) system** — and the nervous system turns the sensor's sensitivities up and
//! down with it: **dynamic** gamma drive raises the response to stretch VELOCITY, **static** gamma
//! drive raises the response to LENGTH. Downstream of the receptor a third gain, the **synaptic
//! gain** with which the afferent volley drives the motor pool, sets how much reflex a given
//! amount of afferent firing buys. These three are what "spindle gains" means in neuromorphic
//! motor control (Niu, Nandyala and Sanger, *Emulated muscle spindle and spiking afferents
//! validates VLSI neuromorphic hardware as a testbed for sensorimotor function and disease*,
//! Frontiers in Computational Neuroscience 8:141, 2014; Niu, Jalaleddini and colleagues, and
//! Jalaleddini, Niu and colleagues, *Neuromorphic meets neuromechanics*, parts I and II, Journal
//! of Neural Engineering 14(2):025001 and 025002, 2017), and they are here at two levels: as plain multipliers on the power-law receptor
//! ([`Fusimotor`]), and inside a fibre model where the drive acts on the mechanics
//! ([`IntrafusalFibre`]).
//!
//! The report travels at a finite speed. A stretch reflex is therefore a feedback loop with a
//! dead time — tens of milliseconds for a human ankle — and a loop with a dead time has a gain it
//! cannot exceed without oscillating. That ceiling, and the frequency it rings at, depend on the
//! delay alone ([`DelayedReflex`]); [`StretchReflex`] puts the spindle, its fusimotor gains and the
//! synaptic gain into that loop, so the ceiling becomes a statement about THEIR PRODUCT.
//!
//! # Why it is in a neuromorphic crate
//!
//! A robot limb with spiking proprioceptors is the standard neuromorphic control demonstration,
//! and its recurring problems are here in closed form: turning a rate model into spikes without
//! changing the rate ([`SpikeEncoder`]); knowing how much reflex gain a given conduction-plus-
//! computation delay can carry before the limb shakes; and the same loop read the other way —
//! as a model of a DISORDER. Too much spindle feedback gain is a sufficient explanation for the
//! involuntary oscillation of spasticity (clonus), and a gain that a prosthesis designer would
//! call "too stiff" is one a clinician would call hyperreflexia: [`StretchReflex`] is both, and
//! [`StretchReflex::critical_synaptic_gain`] is where one turns into the other. The gains are also
//! an energy knob: an event-driven afferent emits `γ_s · k_L` extra spikes per second per unit of
//! stretch, so halving the static gain halves the spikes a held posture costs.
//!
//! # The closed forms this module is checked against
//!
//! - **The spindle's power law.** Doubling the lengthening velocity multiplies the dynamic
//!   response by `2^p` — `1.516` at `p = 0.6`, not 2 — and the response is odd in velocity until
//!   the rate reaches zero, where the afferent falls silent, as spindles do in rapid shortening.
//! - **The fusimotor gains.** Under [`Fusimotor`] the dynamic response is `γ_d` times the
//!   undriven one and the static response `γ_s` times it, exactly; through [`SpikeEncoder`] a hold
//!   of `T` seconds at stretch `x` emits `⌊(baseline + γ_s k_L x) T⌋` spikes.
//! - **Reflex stiffness.** A limb `b ẋ = F − k x − g (r(t − τ) − r_rest)` settles, whatever the
//!   delay, at `x* = F / (k + g γ_s k_L)`: the reflex is a spring of stiffness `g γ_s k_L`, which
//!   is the "high gain is stiff, low gain is compliant" of the robotics literature as one line.
//! - **The clonus threshold.** With no passive spring that loop IS `x_{n+1} = x_n − hK x_{n−m}`
//!   with `K = g γ_s k_L / b`, so it rings when `g γ_s k_L / b` exceeds Levin and May's boundary
//!   (below). Checked 2% either side, by raising the synaptic gain and, separately, by raising
//!   only the static fusimotor gain — the loop cannot tell them apart. Above the boundary the
//!   linear loop diverges; the spindle's floor at zero rate stops this one, and what is left is a
//!   sustained bounded oscillation, which is what clonus is.
//! - **Fusimotor activation.** A Hill saturation `γ^p / (γ^p + γ_half^p)` — one half at
//!   `γ = γ_half` — and Vannucci, Falotico and Laschi's spike-driven form (*Proprioceptive
//!   feedback through a neuromorphic muscle spindle model*, Frontiers in Neuroscience 11:341,
//!   2017), in which each gamma spike moves the activation a fraction `r` of the way to one and
//!   it decays between spikes: under a regular train at rate `ν` the peak settles at
//!   `r / (1 − (1 − r) e^{−1/(ντ)})`, below one at every rate.
//! - **The intrafusal fibre.** The fibre of Mileusnic, Brown, Lan and Loeb (*Mathematical models
//!   of proprioceptors. I. Control and transduction in the muscle spindle*, Journal of
//!   Neurophysiology 96(4):1772–1788, 2006), in the first-order form Vannucci and colleagues give
//!   it: an elastic sensory region in series with a polar region that is a spring, a
//!   velocity-power damper whose coefficient `β = β₀ + β₁ f_d + β₂ f_s` the gamma drive raises,
//!   and an active force `Γ = Γ₁ f_d + Γ₂ f_s`. Held at length `L` its tension is
//!   `T* = [K_PR (L − L0_SR − L0_PR) + Γ] / (1 + K_PR / K_SR)` — two springs in series plus the
//!   active force — so at equilibrium the LENGTH sensitivity `K_SR K_PR / (K_SR + K_PR)` does not
//!   depend on the drive at all and static gamma drive appears as a BIAS `Γ₂ f_s K_SR / (K_SR +
//!   K_PR)`. (That is a property of the model worth knowing before calling `γ_s` a "gain": in
//!   [`Fusimotor`] it multiplies the slope, in the fibre it shifts the intercept.) The tension
//!   rate is zero at `T*`, positive below it and negative above; the rate returned satisfies the
//!   polar region's constitutive law exactly; and the implicit step agrees with a fine explicit
//!   integration of the same equation.
//! - **The tendon organ's dynamics.** Houk and Henneman (*Responses of Golgi tendon organs to active
//!   contractions of the soleus muscle of the cat*, Journal of Neurophysiology 30(3):466–481,
//!   1967) fitted a linear model whose response to a unit step of force is
//!   `K [1 + B e^{−bt} + C e^{−ct}]` — an overshoot of `K(B + C)` that relaxes at two rates to the
//!   static gain `K` — as summarised by Mileusnic and Loeb in *Proprioceptors and models of
//!   transduction* (Scholarpedia 10(5):12390, 2015). [`TendonDynamics`] steps that model exactly
//!   for a force held over the step: checked against the step response, against composition of
//!   steps, and against superposition.
//! - **The encoder is exact.** Over any run the spikes emitted number `⌊∫ rate dt⌋`, and at a
//!   constant rate the intervals are `1/rate`.
//! - **The delayed loop.** For `ẋ = −K x(t − τ)` the largest stable gain is `K_c = π/(2τ)`, and at
//!   it the loop rings with period `4τ` — 6.25 Hz for a 40 ms loop, which is where ankle clonus
//!   sits. The simulation is a discrete loop, `x_{n+1} = x_n − hK x_{n−m}`, whose own exact
//!   boundary is `hK_c = 2 sin(π/(2(2m + 1)))` with ringing period `(4m + 2) h` (Levin and May, *A
//!   note on difference-delay equations*, Theoretical Population Biology 9(2):178–187, 1976). The
//!   loop is checked against ITS boundary — decaying 2% below, growing 2% above — and the
//!   discrete boundary is checked to converge on the continuous one as `π/(2τ + h)`.
//!
//! # What this module has NOT reproduced
//!
//! - **The published spindle coefficients.** This review confirmed the exponent (0.5–0.6) and the
//!   82 impulses-per-second baseline of the cat hamstring model from open sources, and did not
//!   locate the velocity and length gains in a source it could read; they are therefore
//!   parameters of [`Spindle`], not constants of this crate. Supply them from the paper.
//! - **The intrafusal fibre's constants.** The STRUCTURE of [`IntrafusalFibre`] was read from
//!   Vannucci and colleagues' open-access text (their equations 1 to 11). The values — a column
//!   each for the bag1, bag2 and chain fibres — are Table 1 of Mileusnic and colleagues (2006),
//!   which this review did not locate in a source it could read; every one is a field the caller
//!   supplies, and the numbers in this module's tests are illustrative, not the published fit.
//!   One discrepancy is recorded rather than resolved silently: equation 8 of the open-access
//!   text raises the normalised force to the power `a`, while inverting its own equation 5 gives
//!   `1/a`. This module uses `1/a`, and `the_tension_rate_satisfies_the_constitutive_law` is the
//!   test that holds it to equation 5.
//! - The acceleration term of the original second-order fibre (dropped by Vannucci and
//!   colleagues, who measured its effect at under 1% in most cases — their figure, not checked
//!   here); the assembly of three fibres into Ia and II afferents beyond the partial-occlusion
//!   rule ([`occlusion`]); the initial burst and history dependence of real afferents.
//! - **The dynamic gain's effect on loop stability.** Velocity feedback through a delay makes the
//!   loop a NEUTRAL delay equation, whose boundary this module does not derive. [`StretchReflex`]
//!   carries the term and one step of it is checked against arithmetic; how `γ_d` moves the
//!   clonus threshold is not claimed.
//! - Fitted constants for the tendon organ's dynamics. [`TendonDynamics`] is the linear model's
//!   FORM, with its five parameters supplied by the caller: the source read for it (Mileusnic
//!   and Loeb's review, above) reports that they were fitted receptor by receptor and does not
//!   tabulate them.
//! - A muscle, or a limb with inertia. The reflex loops here are first order.

use core::f64::consts::PI;
use core::fmt;
use std::collections::VecDeque;

/// The velocity exponent of the cat hamstring Ia model (Prochazka and Gorassini, 1998).
pub const HAMSTRING_EXPONENT: f64 = 0.6;
/// The baseline rate of that model, impulses per second.
pub const HAMSTRING_BASELINE: f64 = 82.0;
/// The longest loop delay, in steps, this module will buffer.
pub const MAX_DELAY_STEPS: usize = 1 << 20;

/// What went wrong, named rather than guessed around.
#[derive(Debug, Clone, PartialEq)]
pub enum ProprioError {
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
    },
}

impl fmt::Display for ProprioError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfRange { what, value, low, high } => {
                write!(f, "{what} = {value} is outside [{low}, {high}]")
            }
            Self::NonFinite { what } => write!(f, "{what} is not finite"),
        }
    }
}

impl std::error::Error for ProprioError {}

fn finite(what: &'static str, v: f64) -> Result<f64, ProprioError> {
    if v.is_finite() { Ok(v) } else { Err(ProprioError::NonFinite { what }) }
}

fn positive(what: &'static str, v: f64) -> Result<f64, ProprioError> {
    if v.is_finite() && v > 0.0 {
        Ok(v)
    } else {
        Err(ProprioError::OutOfRange { what, value: v, low: f64::MIN_POSITIVE, high: f64::INFINITY })
    }
}

fn non_negative(what: &'static str, v: f64) -> Result<f64, ProprioError> {
    if v.is_finite() && v >= 0.0 {
        Ok(v)
    } else {
        Err(ProprioError::OutOfRange { what, value: v, low: 0.0, high: f64::INFINITY })
    }
}

// ---------------------------------------------------------------------------------------------
// Receptors
// ---------------------------------------------------------------------------------------------

/// A spindle afferent as a static function of stretch and velocity:
/// `rate = baseline + length_gain · stretch + velocity_gain · sign(v) |v|^exponent`, floored at
/// zero. Stretch and velocity are in whatever length unit the gains were fitted in.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Spindle {
    /// Rate at rest length and rest, impulses per second.
    pub baseline: f64,
    /// Impulses per second per unit of stretch.
    pub length_gain: f64,
    /// Impulses per second per (unit of velocity)^exponent.
    pub velocity_gain: f64,
    /// The power of velocity, in `(0, 1]`; below one is what keeps slow movement visible.
    pub exponent: f64,
}

impl Spindle {
    /// Build.
    ///
    /// # Errors
    ///
    /// [`ProprioError::OutOfRange`] for a negative baseline or gain, or an exponent outside
    /// `(0, 1]`.
    pub fn new(baseline: f64, length_gain: f64, velocity_gain: f64, exponent: f64) -> Result<Self, ProprioError> {
        let baseline = non_negative("baseline", baseline)?;
        let length_gain = non_negative("length_gain", length_gain)?;
        let velocity_gain = non_negative("velocity_gain", velocity_gain)?;
        if !(exponent > 0.0) || !(exponent <= 1.0) {
            return Err(ProprioError::OutOfRange { what: "exponent", value: exponent, low: f64::MIN_POSITIVE, high: 1.0 });
        }
        Ok(Self { baseline, length_gain, velocity_gain, exponent })
    }

    /// The cat hamstring FORM — exponent [`HAMSTRING_EXPONENT`], baseline [`HAMSTRING_BASELINE`] —
    /// with the two gains supplied by the caller, because this review did not locate them in a
    /// source it could read.
    ///
    /// # Errors
    ///
    /// As [`Spindle::new`].
    pub fn hamstring_form(length_gain: f64, velocity_gain: f64) -> Result<Self, ProprioError> {
        Self::new(HAMSTRING_BASELINE, length_gain, velocity_gain, HAMSTRING_EXPONENT)
    }

    /// The firing rate, impulses per second, never negative.
    ///
    /// # Errors
    ///
    /// [`ProprioError::NonFinite`] for a non-finite stretch or velocity.
    pub fn rate(&self, stretch: f64, velocity: f64) -> Result<f64, ProprioError> {
        let (stretch, velocity) = (finite("stretch", stretch)?, finite("velocity", velocity)?);
        let dynamic = self.velocity_gain * velocity.signum() * velocity.abs().powf(self.exponent);
        let dynamic = if velocity == 0.0 { 0.0 } else { dynamic };
        Ok((self.baseline + self.length_gain * stretch + dynamic).max(0.0))
    }
}

/// A Golgi tendon organ's static response: silent below a force threshold, linear above it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TendonOrgan {
    /// Impulses per second per newton.
    pub gain: f64,
    /// Force below which it is silent, newtons.
    pub threshold: f64,
}

impl TendonOrgan {
    /// Build.
    ///
    /// # Errors
    ///
    /// [`ProprioError::OutOfRange`] for a non-positive gain or a negative threshold.
    pub fn new(gain: f64, threshold: f64) -> Result<Self, ProprioError> {
        Ok(Self { gain: positive("gain", gain)?, threshold: non_negative("threshold", threshold)? })
    }

    /// The firing rate at `force` newtons, impulses per second.
    ///
    /// # Errors
    ///
    /// [`ProprioError::NonFinite`] for a non-finite force.
    pub fn rate(&self, force: f64) -> Result<f64, ProprioError> {
        Ok(self.gain * (finite("force", force)? - self.threshold).max(0.0))
    }
}

/// The Houk–Henneman linear model of a tendon organ: the rate is `K` times the force plus two
/// high-passed copies of it, so that a step of force `F` gives `K F [1 + B e^{−bt} + C e^{−ct}]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TendonDynamics {
    /// Static gain `K`, impulses per second per newton.
    pub k: f64,
    /// Size of the fast overshoot, as a multiple of the static response.
    pub b_gain: f64,
    /// Its decay rate `b`, 1/s.
    pub b_rate: f64,
    /// Size of the slow overshoot.
    pub c_gain: f64,
    /// Its decay rate `c`, 1/s.
    pub c_rate: f64,
    /// The two low-pass states the high-passed copies are formed from, newtons.
    pub state: [f64; 2],
}

impl TendonDynamics {
    /// Build at rest (no force history).
    ///
    /// # Errors
    ///
    /// [`ProprioError::OutOfRange`] for a non-positive gain `K` or decay rate, or a negative
    /// overshoot.
    pub fn new(k: f64, b_gain: f64, b_rate: f64, c_gain: f64, c_rate: f64) -> Result<Self, ProprioError> {
        Ok(Self {
            k: positive("k", k)?,
            b_gain: non_negative("b_gain", b_gain)?,
            b_rate: positive("b_rate", b_rate)?,
            c_gain: non_negative("c_gain", c_gain)?,
            c_rate: positive("c_rate", c_rate)?,
            state: [0.0; 2],
        })
    }

    /// Advance by `dt` with the force held at `force` over the step, and return the rate at the
    /// END of the step, impulses per second, floored at zero. Exact for a held force: each
    /// overshoot term is the force minus its own exponential low-pass.
    ///
    /// # Errors
    ///
    /// [`ProprioError::OutOfRange`] for a non-positive `dt`, [`ProprioError::NonFinite`] for a
    /// non-finite force.
    pub fn step(&mut self, dt: f64, force: f64) -> Result<f64, ProprioError> {
        let dt = positive("dt", dt)?;
        let force = finite("force", force)?;
        for (s, rate) in self.state.iter_mut().zip([self.b_rate, self.c_rate]) {
            *s = force + (*s - force) * (-rate * dt).exp();
        }
        let overshoot = self.b_gain * (force - self.state[0]) + self.c_gain * (force - self.state[1]);
        Ok((self.k * (force + overshoot)).max(0.0))
    }
}

/// Turns a rate into spikes without changing it: a phase accumulator that emits one spike each
/// time the integral of the rate passes a whole number.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct SpikeEncoder {
    /// The fraction of a spike accumulated so far, in `[0, 1)`.
    pub phase: f64,
    /// Spikes emitted since the start.
    pub spikes: u64,
}

impl SpikeEncoder {
    /// Advance by `dt` seconds at `rate` impulses per second; returns the spikes emitted in this
    /// step (more than one when `rate · dt` exceeds one).
    ///
    /// # Errors
    ///
    /// [`ProprioError::OutOfRange`] for a negative rate or a non-positive `dt`.
    pub fn step(&mut self, dt: f64, rate: f64) -> Result<u64, ProprioError> {
        let dt = positive("dt", dt)?;
        let rate = non_negative("rate", rate)?;
        let total = self.phase + rate * dt;
        let emitted = total.floor();
        self.phase = total - emitted;
        // `emitted` is a non-negative whole number; a rate·dt past 2⁶⁴ saturates rather than wraps.
        let emitted = emitted as u64;
        self.spikes = self.spikes.saturating_add(emitted);
        Ok(emitted)
    }
}

// ---------------------------------------------------------------------------------------------
// The delayed reflex loop
// ---------------------------------------------------------------------------------------------

/// The simplest reflex loop with a dead time: `ẋ = −K x(t − τ)`, stepped as
/// `x_{n+1} = x_n − hK x_{n−m}` with `τ = m h`.
#[derive(Debug, Clone, PartialEq)]
pub struct DelayedReflex {
    /// Loop gain `K`, 1/s.
    pub gain: f64,
    /// Time step `h`, seconds.
    pub dt: f64,
    /// The last `m + 1` values of `x`, oldest first; its length fixes the delay.
    pub history: VecDeque<f64>,
}

impl DelayedReflex {
    /// A loop that has been held at `x0` for at least its delay.
    ///
    /// # Errors
    ///
    /// [`ProprioError::OutOfRange`] for a negative gain, a non-positive `dt` or a delay past
    /// [`MAX_DELAY_STEPS`]; [`ProprioError::NonFinite`] for a non-finite `x0`.
    pub fn new(gain: f64, dt: f64, delay_steps: usize, x0: f64) -> Result<Self, ProprioError> {
        let gain = non_negative("gain", gain)?;
        let dt = positive("dt", dt)?;
        if delay_steps > MAX_DELAY_STEPS {
            return Err(ProprioError::OutOfRange { what: "delay_steps", value: delay_steps as f64, low: 0.0, high: MAX_DELAY_STEPS as f64 });
        }
        let x0 = finite("x0", x0)?;
        Ok(Self { gain, dt, history: core::iter::repeat_n(x0, delay_steps + 1).collect() })
    }

    /// The delay in steps, `m`.
    #[must_use]
    pub fn delay_steps(&self) -> usize {
        self.history.len().saturating_sub(1)
    }

    /// The current value of `x`.
    #[must_use]
    pub fn x(&self) -> f64 {
        self.history.back().copied().unwrap_or(0.0)
    }

    /// One step; returns the new `x`.
    pub fn step(&mut self) -> f64 {
        let delayed = self.history.front().copied().unwrap_or(0.0);
        let next = self.x() - self.dt * self.gain * delayed;
        self.history.pop_front();
        self.history.push_back(next);
        next
    }
}

/// The largest stable gain of `ẋ = −K x(t − τ)`: `π/(2τ)`, 1/s. `None` for a non-positive delay.
#[must_use]
pub fn critical_gain(delay_s: f64) -> Option<f64> {
    if delay_s > 0.0 && delay_s.is_finite() { Some(PI / (2.0 * delay_s)) } else { None }
}

/// The period that loop rings with at its critical gain: `4τ`, seconds. `None` for a
/// non-positive delay.
#[must_use]
pub fn ringing_period(delay_s: f64) -> Option<f64> {
    if delay_s > 0.0 && delay_s.is_finite() { Some(4.0 * delay_s) } else { None }
}

/// The largest stable gain of the DISCRETE loop `x_{n+1} = x_n − hK x_{n−m}`:
/// `(2/h) sin(π/(2(2m + 1)))`, 1/s (Levin and May, 1976). With no delay (`m = 0`) it is `2/h`, the
/// familiar Euler limit. `None` for a non-positive `dt`.
#[must_use]
pub fn critical_gain_discrete(delay_steps: usize, dt: f64) -> Option<f64> {
    if !(dt > 0.0) || !dt.is_finite() {
        return None;
    }
    Some(2.0 / dt * (PI / (2.0 * (2.0 * delay_steps as f64 + 1.0))).sin())
}

/// The discrete loop's ringing period at that gain: `(4m + 2) h`, seconds. `None` for a
/// non-positive `dt`.
#[must_use]
pub fn ringing_period_discrete(delay_steps: usize, dt: f64) -> Option<f64> {
    if dt > 0.0 && dt.is_finite() { Some((4.0 * delay_steps as f64 + 2.0) * dt) } else { None }
}

// ---------------------------------------------------------------------------------------------
// Fusimotor drive
// ---------------------------------------------------------------------------------------------

/// Fusimotor drive as two multipliers on a [`Spindle`]: the dynamic gain `γ_d` on its velocity
/// term and the static gain `γ_s` on its length term. `(1, 1)` is the undriven receptor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Fusimotor {
    /// `γ_d`: multiplies the response to stretch velocity.
    pub dynamic_gain: f64,
    /// `γ_s`: multiplies the response to steady length.
    pub static_gain: f64,
}

impl Fusimotor {
    /// The undriven receptor: both gains one.
    pub const NEUTRAL: Self = Self { dynamic_gain: 1.0, static_gain: 1.0 };

    /// Build.
    ///
    /// # Errors
    ///
    /// [`ProprioError::OutOfRange`] for a negative or non-finite gain.
    pub fn new(dynamic_gain: f64, static_gain: f64) -> Result<Self, ProprioError> {
        Ok(Self { dynamic_gain: non_negative("dynamic_gain", dynamic_gain)?, static_gain: non_negative("static_gain", static_gain)? })
    }
}

impl Spindle {
    /// [`Spindle::rate`] under fusimotor drive:
    /// `baseline + γ_s · length_gain · stretch + γ_d · velocity_gain · sign(v) |v|^exponent`,
    /// floored at zero.
    ///
    /// # Errors
    ///
    /// As [`Spindle::rate`], and [`ProprioError::OutOfRange`] for a negative gain in `drive`.
    pub fn rate_under(&self, drive: Fusimotor, stretch: f64, velocity: f64) -> Result<f64, ProprioError> {
        let drive = Fusimotor::new(drive.dynamic_gain, drive.static_gain)?;
        let (stretch, velocity) = (finite("stretch", stretch)?, finite("velocity", velocity)?);
        let dynamic = if velocity == 0.0 { 0.0 } else { self.velocity_gain * velocity.signum() * velocity.abs().powf(self.exponent) };
        Ok((self.baseline + drive.static_gain * self.length_gain * stretch + drive.dynamic_gain * dynamic).max(0.0))
    }
}

/// The Hill saturation that turns a gamma firing rate into an activation in `[0, 1)`:
/// `γ^p / (γ^p + γ_half^p)`. `None` for a negative rate or a non-positive `half` or `power`.
#[must_use]
pub fn hill_activation(gamma_hz: f64, half_hz: f64, power: f64) -> Option<f64> {
    if !(gamma_hz >= 0.0) || !gamma_hz.is_finite() || !(half_hz > 0.0) || !half_hz.is_finite() || !(power > 0.0) || !power.is_finite() {
        return None;
    }
    // Written in the ratio so that a large rate cannot overflow the numerator and denominator
    // into `inf / inf`.
    let x = (gamma_hz / half_hz).powf(power);
    Some(if x.is_finite() { x / (1.0 + x) } else { 1.0 })
}

/// A fusimotor activation driven by gamma SPIKES rather than by a rate (Vannucci, Falotico and
/// Laschi, 2017): each spike moves the activation a fraction `response` of the way to one, and
/// between spikes it decays with time constant `tau`. It cannot leave `[0, 1)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Activation {
    /// The activation, in `[0, 1)`.
    pub f: f64,
    /// The fraction of the remaining headroom one spike takes, in `(0, 1)`.
    pub response: f64,
    /// Decay time constant, seconds.
    pub tau: f64,
}

impl Activation {
    /// An activation at rest.
    ///
    /// # Errors
    ///
    /// [`ProprioError::OutOfRange`] for a `response` outside `(0, 1)` or a non-positive `tau`.
    pub fn new(response: f64, tau: f64) -> Result<Self, ProprioError> {
        if !(response > 0.0) || !(response < 1.0) {
            return Err(ProprioError::OutOfRange { what: "response", value: response, low: 0.0, high: 1.0 });
        }
        Ok(Self { f: 0.0, response, tau: positive("tau", tau)? })
    }

    /// Let `dt` seconds pass with no spike. Returns the activation.
    ///
    /// # Errors
    ///
    /// [`ProprioError::OutOfRange`] for a negative `dt`.
    pub fn decay(&mut self, dt: f64) -> Result<f64, ProprioError> {
        let dt = non_negative("dt", dt)?;
        self.f *= (-dt / self.tau).exp();
        Ok(self.f)
    }

    /// Receive one gamma spike. Returns the activation.
    pub fn spike(&mut self) -> f64 {
        self.f += self.response * (1.0 - self.f);
        self.f
    }

    /// The activation just after each spike of a regular train at `rate_hz`, once settled:
    /// `r / (1 − (1 − r) e^{−1/(ντ)})`. `None` for a non-positive rate.
    #[must_use]
    pub fn settled_peak(&self, rate_hz: f64) -> Option<f64> {
        if !(rate_hz > 0.0) || !rate_hz.is_finite() {
            return None;
        }
        let d = (-1.0 / (rate_hz * self.tau)).exp();
        Some(self.response / (1.0 - (1.0 - self.response) * d))
    }
}

/// Partial occlusion, the rule by which two intrafusal contributions share one afferent: the
/// larger drives it and the smaller adds a fraction `s` of itself.
#[must_use]
pub fn occlusion(a: f64, b: f64, s: f64) -> f64 {
    a.max(b) + s * a.min(b)
}

// ---------------------------------------------------------------------------------------------
// The intrafusal fibre
// ---------------------------------------------------------------------------------------------

/// One intrafusal fibre in the first-order form of Vannucci, Falotico and Laschi (2017) of the
/// model of Mileusnic, Brown, Lan and Loeb (2006).
///
/// A **sensory region**, a spring `K_SR` of rest length `L0_SR` on which the afferent ending
/// sits, is in series with a **polar region** of length `L_PR = L − L0_SR − T / K_SR` that carries
/// the same tension through a spring, a velocity-power damper and an active force:
///
/// ```text
/// T = β C (L_PR − R) sign(L̇_PR) |L̇_PR|^a + K_PR (L_PR − L0_PR) + Γ
/// β = β₀ + β₁ f_dynamic + β₂ f_static          Γ = Γ₁ f_dynamic + Γ₂ f_static
/// ```
///
/// Lengths are in units of the muscle's rest fascicle length and tension in the model's force
/// units. Every constant is a field: see the module documentation for why none is supplied.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IntrafusalFibre {
    /// Sensory-region stiffness `K_SR`.
    pub k_sr: f64,
    /// Polar-region stiffness `K_PR`.
    pub k_pr: f64,
    /// The damper's coefficient of asymmetry `C` while the polar region lengthens.
    pub c_lengthening: f64,
    /// `C` while it shortens.
    pub c_shortening: f64,
    /// Passive damping `β₀`.
    pub beta0: f64,
    /// Damping added per unit of dynamic activation, `β₁` — what makes `γ_d` a velocity gain.
    pub beta_dynamic: f64,
    /// Damping added per unit of static activation, `β₂`.
    pub beta_static: f64,
    /// Active force per unit of dynamic activation, `Γ₁`.
    pub force_dynamic: f64,
    /// Active force per unit of static activation, `Γ₂`.
    pub force_static: f64,
    /// The damper's velocity exponent `a`, in `(0, 1]`.
    pub a: f64,
    /// The polar length `R` below which the damper produces no force.
    pub r: f64,
    /// Sensory-region rest length `L0_SR`.
    pub l0_sr: f64,
    /// Polar-region rest length `L0_PR`.
    pub l0_pr: f64,
    /// The fibre's tension, the model's one state variable.
    pub tension: f64,
}

impl IntrafusalFibre {
    fn validated(&self) -> Result<(), ProprioError> {
        positive("k_sr", self.k_sr)?;
        positive("k_pr", self.k_pr)?;
        positive("c_lengthening", self.c_lengthening)?;
        positive("c_shortening", self.c_shortening)?;
        positive("beta0", self.beta0)?;
        // The increments may be negative (a drive that LOWERS the damping); what must stay
        // positive is the coefficient at both ends of each activation's range.
        finite("beta_dynamic", self.beta_dynamic)?;
        finite("beta_static", self.beta_static)?;
        positive("beta at full drive", self.beta0 + self.beta_dynamic.min(0.0) + self.beta_static.min(0.0))?;
        non_negative("force_dynamic", self.force_dynamic)?;
        non_negative("force_static", self.force_static)?;
        if !(self.a > 0.0) || !(self.a <= 1.0) {
            return Err(ProprioError::OutOfRange { what: "a", value: self.a, low: f64::MIN_POSITIVE, high: 1.0 });
        }
        finite("r", self.r)?;
        finite("l0_sr", self.l0_sr)?;
        finite("l0_pr", self.l0_pr)?;
        finite("tension", self.tension)?;
        Ok(())
    }

    fn activations(f_dynamic: f64, f_static: f64) -> Result<(f64, f64), ProprioError> {
        for (what, f) in [("f_dynamic", f_dynamic), ("f_static", f_static)] {
            if !(f >= 0.0) || !(f <= 1.0) {
                return Err(ProprioError::OutOfRange { what, value: f, low: 0.0, high: 1.0 });
            }
        }
        Ok((f_dynamic, f_static))
    }

    /// The damping coefficient under drive, `β₀ + β₁ f_d + β₂ f_s`.
    #[must_use]
    pub fn beta(&self, f_dynamic: f64, f_static: f64) -> f64 {
        self.beta0 + self.beta_dynamic * f_dynamic + self.beta_static * f_static
    }

    /// The active force under drive, `Γ₁ f_d + Γ₂ f_s`.
    #[must_use]
    pub fn active_force(&self, f_dynamic: f64, f_static: f64) -> f64 {
        self.force_dynamic * f_dynamic + self.force_static * f_static
    }

    /// The polar region's length at fascicle length `length` and the present tension.
    #[must_use]
    pub fn polar_length(&self, length: f64) -> f64 {
        length - self.l0_sr - self.tension / self.k_sr
    }

    /// The tension the fibre holds at a fixed `length`:
    /// `[K_PR (L − L0_SR − L0_PR) + Γ] / (1 + K_PR / K_SR)`.
    ///
    /// # Errors
    ///
    /// [`ProprioError`] for an invalid fibre, a non-finite length or an activation outside
    /// `[0, 1]`.
    pub fn equilibrium_tension(&self, length: f64, f_dynamic: f64, f_static: f64) -> Result<f64, ProprioError> {
        self.validated()?;
        let (fd, fs) = Self::activations(f_dynamic, f_static)?;
        let length = finite("length", length)?;
        Ok((self.k_pr * (length - self.l0_sr - self.l0_pr) + self.active_force(fd, fs)) / (1.0 + self.k_pr / self.k_sr))
    }

    /// The polar region's constitutive law: the tension it carries at the present length when it
    /// is changing length at `polar_velocity`.
    ///
    /// # Errors
    ///
    /// As [`IntrafusalFibre::equilibrium_tension`].
    pub fn polar_tension(&self, length: f64, polar_velocity: f64, f_dynamic: f64, f_static: f64) -> Result<f64, ProprioError> {
        self.validated()?;
        let (fd, fs) = Self::activations(f_dynamic, f_static)?;
        let (length, v) = (finite("length", length)?, finite("polar_velocity", polar_velocity)?);
        let lpr = self.polar_length(length);
        let c = if v >= 0.0 { self.c_lengthening } else { self.c_shortening };
        let damper = if v == 0.0 { 0.0 } else { self.beta(fd, fs) * c * (lpr - self.r) * v.signum() * v.abs().powf(self.a) };
        Ok(damper + self.k_pr * (lpr - self.l0_pr) + self.active_force(fd, fs))
    }

    /// `dT/dt` at fascicle length `length` changing at `velocity`:
    /// `K_SR [L̇ − sign(δ) |δ / (β C (L_PR − R))|^{1/a}]`, where
    /// `δ = T − K_PR (L_PR − L0_PR) − Γ` is the force the damper is carrying.
    ///
    /// # Errors
    ///
    /// As [`IntrafusalFibre::equilibrium_tension`], and [`ProprioError::OutOfRange`] when the polar
    /// region is no longer than `R` — the damper's force has gone to zero there and the equation
    /// has no velocity to give.
    pub fn tension_rate(&self, length: f64, velocity: f64, f_dynamic: f64, f_static: f64) -> Result<f64, ProprioError> {
        self.validated()?;
        let (fd, fs) = Self::activations(f_dynamic, f_static)?;
        let (length, velocity) = (finite("length", length)?, finite("velocity", velocity)?);
        self.rate_at(self.tension, length, velocity, fd, fs)
    }

    fn rate_at(&self, tension: f64, length: f64, velocity: f64, fd: f64, fs: f64) -> Result<f64, ProprioError> {
        let lpr = length - self.l0_sr - tension / self.k_sr;
        let reach = lpr - self.r;
        if !(reach > 0.0) {
            return Err(ProprioError::OutOfRange { what: "polar length above R", value: reach, low: f64::MIN_POSITIVE, high: f64::INFINITY });
        }
        let carried = tension - self.k_pr * (lpr - self.l0_pr) - self.active_force(fd, fs);
        // The damper carries a positive force exactly when the polar region is lengthening.
        let c = if carried >= 0.0 { self.c_lengthening } else { self.c_shortening };
        let x = carried / (self.beta(fd, fs) * c * reach);
        let polar_velocity = if x == 0.0 { 0.0 } else { x.signum() * x.abs().powf(1.0 / self.a) };
        Ok(self.k_sr * (velocity - polar_velocity))
    }

    /// Advance the tension by `dt` with the length and drive held over the step, by backward
    /// Euler: the damper's law is a steep power of the force (`1/a` is above three for the
    /// published exponent), which an explicit step overshoots. Returns the new tension.
    ///
    /// # Errors
    ///
    /// As [`IntrafusalFibre::tension_rate`], and [`ProprioError::OutOfRange`] for a non-positive
    /// `dt`.
    pub fn step(&mut self, dt: f64, length: f64, velocity: f64, f_dynamic: f64, f_static: f64) -> Result<f64, ProprioError> {
        self.validated()?;
        let dt = positive("dt", dt)?;
        let (fd, fs) = Self::activations(f_dynamic, f_static)?;
        let (length, velocity) = (finite("length", length)?, finite("velocity", velocity)?);
        let t0 = self.tension;
        // Solve g(T) = T − t0 − dt · rate(T) = 0. The tension at which the polar region shrinks to
        // `R` bounds it above (the rate falls without limit there); below, the rate is bounded, so
        // g is eventually negative.
        let ceiling = self.k_sr * (length - self.l0_sr - self.r);
        let g = |t: f64| -> Result<f64, ProprioError> { Ok(t - t0 - dt * self.rate_at(t, length, velocity, fd, fs)?) };
        let mut hi = ceiling - ceiling.abs().max(1.0) * 1e-12;
        let mut lo = t0.min(hi);
        let mut span = (ceiling - lo).abs().max(1.0);
        let mut tries = 0;
        while g(lo)? > 0.0 {
            lo -= span;
            span *= 2.0;
            tries += 1;
            if tries > 200 {
                return Err(ProprioError::NonFinite { what: "implicit step bracket" });
            }
        }
        if g(hi)? < 0.0 {
            return Err(ProprioError::OutOfRange { what: "polar length above R", value: hi, low: f64::MIN_POSITIVE, high: f64::INFINITY });
        }
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            if mid <= lo || mid >= hi {
                break;
            }
            if g(mid)? > 0.0 { hi = mid } else { lo = mid }
        }
        self.tension = 0.5 * (lo + hi);
        Ok(self.tension)
    }

    /// The fibre's contribution to the PRIMARY afferent, impulses per second: the stretch of the
    /// sensory region past its threshold length `ln_sr`, times the gain `g`,
    /// `g [T / K_SR − (LN_SR − L0_SR)]`, floored at zero.
    #[must_use]
    pub fn primary_contribution(&self, g: f64, ln_sr: f64) -> f64 {
        (g * (self.tension / self.k_sr - (ln_sr - self.l0_sr))).max(0.0)
    }
}

// ---------------------------------------------------------------------------------------------
// The stretch reflex with its three gains
// ---------------------------------------------------------------------------------------------

/// A first-order limb held by a delayed stretch reflex:
/// `b ẋ = F − k x − g (r(t − τ) − r_rest)`, where `r` is a [`Spindle`] under [`Fusimotor`] drive
/// reading the limb's displacement `x` and velocity, and `g` is the synaptic gain — force per
/// impulse per second of afferent firing above rest. Stepped as
/// `x_{n+1} = x_n + (h/b) [F − k x_n − g (r_{n−m} − r_rest)]`.
#[derive(Debug, Clone, PartialEq)]
pub struct StretchReflex {
    /// The receptor.
    pub spindle: Spindle,
    /// Its fusimotor gains.
    pub drive: Fusimotor,
    /// Synaptic gain `g`, force per (impulse per second).
    pub synaptic_gain: f64,
    /// Passive stiffness `k`, force per unit displacement.
    pub stiffness: f64,
    /// Viscosity `b`, force per unit velocity.
    pub viscosity: f64,
    /// Time step `h`, seconds.
    pub dt: f64,
    /// The last `m + 1` afferent rates, oldest first; its length fixes the delay.
    pub afferent: VecDeque<f64>,
    /// Displacement now.
    pub x: f64,
    /// Displacement one step ago, for the velocity the spindle sees.
    pub x_before: f64,
}

impl StretchReflex {
    /// A limb that has been at rest at `x = 0` for at least the loop delay.
    ///
    /// # Errors
    ///
    /// [`ProprioError::OutOfRange`] for a negative gain or stiffness, a non-positive viscosity or
    /// `dt`, or a delay past [`MAX_DELAY_STEPS`].
    pub fn new(spindle: Spindle, drive: Fusimotor, synaptic_gain: f64, stiffness: f64, viscosity: f64, dt: f64, delay_steps: usize) -> Result<Self, ProprioError> {
        let drive = Fusimotor::new(drive.dynamic_gain, drive.static_gain)?;
        let synaptic_gain = non_negative("synaptic_gain", synaptic_gain)?;
        let stiffness = non_negative("stiffness", stiffness)?;
        let viscosity = positive("viscosity", viscosity)?;
        let dt = positive("dt", dt)?;
        if delay_steps > MAX_DELAY_STEPS {
            return Err(ProprioError::OutOfRange { what: "delay_steps", value: delay_steps as f64, low: 0.0, high: MAX_DELAY_STEPS as f64 });
        }
        let rest = spindle.rate_under(drive, 0.0, 0.0)?;
        Ok(Self { spindle, drive, synaptic_gain, stiffness, viscosity, dt, afferent: core::iter::repeat_n(rest, delay_steps + 1).collect(), x: 0.0, x_before: 0.0 })
    }

    /// The stiffness the reflex adds to the limb, `g γ_s k_L`.
    #[must_use]
    pub fn reflex_stiffness(&self) -> f64 {
        self.synaptic_gain * self.drive.static_gain * self.spindle.length_gain
    }

    /// Where a constant `load` leaves the limb: `F / (k + g γ_s k_L)`. `None` if nothing resists
    /// the load.
    #[must_use]
    pub fn settled_displacement(&self, load: f64) -> Option<f64> {
        let k = self.stiffness + self.reflex_stiffness();
        if k > 0.0 && load.is_finite() { Some(load / k) } else { None }
    }

    /// The synaptic gain above which the loop rings, for a limb with no passive spring and a
    /// spindle read in its linear range: `b K_c / (γ_s k_L)`, with `K_c` the discrete loop's
    /// boundary [`critical_gain_discrete`]. `None` when the spindle has no length response for the
    /// gain to act through.
    #[must_use]
    pub fn critical_synaptic_gain(&self) -> Option<f64> {
        let through = self.drive.static_gain * self.spindle.length_gain;
        if !(through > 0.0) {
            return None;
        }
        let m = self.afferent.len().saturating_sub(1);
        critical_gain_discrete(m, self.dt).map(|kc| self.viscosity * kc / through)
    }

    /// One step under `load`; returns the new displacement.
    ///
    /// # Errors
    ///
    /// [`ProprioError::NonFinite`] for a non-finite load or a displacement that has overflowed.
    pub fn step(&mut self, load: f64) -> Result<f64, ProprioError> {
        let load = finite("load", load)?;
        let rest = self.spindle.rate_under(self.drive, 0.0, 0.0)?;
        let delayed = self.afferent.front().copied().unwrap_or(rest);
        let next = self.x + self.dt / self.viscosity * (load - self.stiffness * self.x - self.synaptic_gain * (delayed - rest));
        let next = finite("displacement", next)?;
        self.x_before = self.x;
        self.x = next;
        let velocity = (self.x - self.x_before) / self.dt;
        let rate = self.spindle.rate_under(self.drive, self.x, velocity)?;
        self.afferent.pop_front();
        self.afferent.push_back(rate);
        Ok(next)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_spindle_answers_velocity_as_a_power_and_falls_silent_in_fast_shortening() {
        let ia = Spindle::hamstring_form(3.0, 5.0).unwrap();
        assert_eq!((ia.exponent, ia.baseline), (0.6, 82.0));
        assert_eq!(ia.rate(0.0, 0.0).unwrap(), 82.0);
        assert_eq!(ia.rate(4.0, 0.0).unwrap(), 82.0 + 12.0);
        // The dynamic response: 5·v^0.6, so doubling the velocity multiplies it by 2^0.6 = 1.516.
        let dynamic = |v: f64| ia.rate(0.0, v).unwrap() - 82.0;
        assert!((dynamic(10.0) - 5.0 * 10f64.powf(0.6)).abs() < 1e-12);
        assert!((dynamic(20.0) / dynamic(10.0) - 2f64.powf(0.6)).abs() < 1e-12);
        assert!((2f64.powf(0.6) - 1.5157).abs() < 1e-4);
        // A hundredfold in velocity is under sixteenfold in rate: that is the point of p < 1.
        assert!((dynamic(100.0) / dynamic(1.0) - 100f64.powf(0.6)).abs() < 1e-9 && 100f64.powf(0.6) < 16.0);
        // Odd in velocity while the afferent is firing…
        assert!((dynamic(-10.0) + dynamic(10.0)).abs() < 1e-12);
        // …and silent once shortening is fast enough: 82 = 5·v^0.6 at v = (82/5)^(1/0.6) = 105.9.
        let silent_at = (82.0f64 / 5.0).powf(1.0 / 0.6);
        assert!(ia.rate(0.0, -0.99 * silent_at).unwrap() > 0.0);
        assert_eq!(ia.rate(0.0, -1.01 * silent_at).unwrap(), 0.0);
        assert_eq!(ia.rate(0.0, -1e6).unwrap(), 0.0);
        // A linear spindle (exponent 1) for contrast: doubling doubles.
        let linear = Spindle::new(10.0, 0.0, 2.0, 1.0).unwrap();
        assert_eq!(linear.rate(0.0, 6.0).unwrap() - 10.0, 2.0 * (linear.rate(0.0, 3.0).unwrap() - 10.0));
    }

    #[test]
    fn the_tendon_organ_is_a_line_above_its_threshold() {
        let ib = TendonOrgan::new(4.0, 0.5).unwrap();
        assert_eq!(ib.rate(0.0).unwrap(), 0.0);
        assert_eq!(ib.rate(0.5).unwrap(), 0.0);
        assert_eq!(ib.rate(3.0).unwrap(), 10.0);
        assert_eq!(ib.rate(-2.0).unwrap(), 0.0, "a tendon cannot push");
    }

    #[test]
    fn the_tendon_organ_overshoots_a_step_of_force_and_relaxes_at_two_rates() {
        let (k, bg, br, cg, cr) = (3.0, 0.8, 40.0, 0.3, 2.5);
        let closed = |force: f64, t: f64| k * force * (1.0 + bg * (-br * t).exp() + cg * (-cr * t).exp());
        // One long step, and the same interval in a thousand short ones: the update is exact.
        let mut one = TendonDynamics::new(k, bg, br, cg, cr).unwrap();
        assert!((one.step(0.2, 5.0).unwrap() - closed(5.0, 0.2)).abs() < 1e-12);
        let mut many = TendonDynamics::new(k, bg, br, cg, cr).unwrap();
        let mut last = 0.0;
        for _ in 0..1000 {
            last = many.step(2e-4, 5.0).unwrap();
        }
        assert!((last - closed(5.0, 0.2)).abs() < 1e-10, "{last} vs {}", closed(5.0, 0.2));
        // The first instant overshoots by K F (B + C); held long enough it settles on K F.
        let mut fresh = TendonDynamics::new(k, bg, br, cg, cr).unwrap();
        assert!((fresh.step(1e-9, 5.0).unwrap() - 15.0 * 2.1).abs() < 1e-5);
        for _ in 0..200 {
            fresh.step(0.1, 5.0).unwrap();
        }
        assert!((fresh.step(0.1, 5.0).unwrap() - 15.0).abs() < 1e-12);
        // Linear: the response to a second step on top of the first is the sum of the two.
        let mut both = TendonDynamics::new(k, bg, br, cg, cr).unwrap();
        both.step(0.05, 2.0).unwrap();
        let got = both.step(0.03, 5.0).unwrap();
        assert!((got - (closed(2.0, 0.08) + closed(3.0, 0.03))).abs() < 1e-12, "{got}");
        // Unloading undershoots, and a rate cannot go below zero: it is floored, not negative.
        assert_eq!(both.step(1e-6, 0.0).unwrap(), 0.0);
        // With no overshoot terms it is the static line.
        let mut plain = TendonDynamics::new(4.0, 0.0, 1.0, 0.0, 1.0).unwrap();
        assert_eq!(plain.step(0.01, 2.5).unwrap(), 10.0);
        assert!(matches!(TendonDynamics::new(0.0, 0.1, 1.0, 0.1, 1.0), Err(ProprioError::OutOfRange { what: "k", .. })));
        assert!(matches!(TendonDynamics::new(1.0, -0.1, 1.0, 0.1, 1.0), Err(ProprioError::OutOfRange { what: "b_gain", .. })));
        assert!(matches!(TendonDynamics::new(1.0, 0.1, 0.0, 0.1, 1.0), Err(ProprioError::OutOfRange { what: "b_rate", .. })));
        assert!(matches!(TendonDynamics::new(1.0, 0.1, 1.0, f64::NAN, 1.0), Err(ProprioError::OutOfRange { what: "c_gain", .. })));
        assert!(matches!(TendonDynamics::new(1.0, 0.1, 1.0, 0.1, -1.0), Err(ProprioError::OutOfRange { what: "c_rate", .. })));
        assert!(matches!(plain.step(0.0, 1.0), Err(ProprioError::OutOfRange { what: "dt", .. })));
        assert!(matches!(plain.step(0.1, f64::NAN), Err(ProprioError::NonFinite { what: "force" })));
    }

    #[test]
    fn the_encoder_emits_the_integral_of_the_rate_and_nothing_else() {
        let mut enc = SpikeEncoder::default();
        let dt = 1e-3;
        let mut times = Vec::new();
        for k in 0..2000u32 {
            if enc.step(dt, 37.0).unwrap() > 0 {
                times.push(f64::from(k + 1) * dt);
            }
        }
        // 37 Hz for 2 s is 74 spikes, to within the rounding of 2000 additions of 0.037.
        assert_eq!(enc.spikes, 74);
        assert_eq!(times.len(), 74);
        for pair in times.windows(2) {
            assert!((pair[1] - pair[0] - 1.0 / 37.0).abs() <= dt, "interval {}", pair[1] - pair[0]);
        }
        // A varying rate: ∫₀¹ 200 t dt = 100.
        let mut ramp = SpikeEncoder::default();
        for k in 0..10_000u32 {
            ramp.step(1e-4, 200.0 * (f64::from(k) + 0.5) * 1e-4).unwrap();
        }
        assert_eq!(ramp.spikes, 100 - u64::from(ramp.phase > 0.5), "{} spikes with {} left over", ramp.spikes, ramp.phase);
        // A burst in one step, and silence.
        let mut burst = SpikeEncoder::default();
        assert_eq!(burst.step(0.1, 55.0).unwrap(), 5);
        assert!((burst.phase - 0.5).abs() < 1e-12);
        assert_eq!(burst.step(10.0, 0.0).unwrap(), 0);
        assert_eq!(burst.spikes, 5);
    }

    fn envelope(loop_: &mut DelayedReflex, periods_to_skip: usize, period_steps: usize) -> (f64, f64) {
        let mut peak = |steps: usize| (0..steps).map(|_| loop_.step().abs()).fold(0.0f64, f64::max);
        peak(periods_to_skip * period_steps);
        let early = peak(period_steps);
        peak(30 * period_steps);
        (early, peak(period_steps))
    }

    #[test]
    fn the_loop_rings_at_the_gain_its_delay_allows_and_not_below_it() {
        let (m, h) = (20usize, 1e-3);
        let edge = critical_gain_discrete(m, h).unwrap();
        assert!((edge - 2000.0 * (PI / 82.0).sin()).abs() < 1e-9);
        let period_steps = 4 * m + 2;
        assert_eq!(ringing_period_discrete(m, h), Some(0.082));
        let mut under = DelayedReflex::new(0.98 * edge, h, m, 1.0).unwrap();
        let (early, late) = envelope(&mut under, 10, period_steps);
        assert!(late < 0.5 * early, "2% under the edge the ringing went {early} → {late}");
        let mut over = DelayedReflex::new(1.02 * edge, h, m, 1.0).unwrap();
        let (early, late) = envelope(&mut over, 10, period_steps);
        assert!(late > 2.0 * early, "2% over the edge the ringing went {early} → {late}");
        // AT the edge it neither grows nor dies, and its period is (4m + 2) h: count the upward
        // zero crossings over a hundred periods.
        let mut at = DelayedReflex::new(edge, h, m, 1.0).unwrap();
        for _ in 0..50 * period_steps {
            at.step();
        }
        let (mut crossings, mut last, mut peak) = (0u32, at.x(), 0.0f64);
        for _ in 0..100 * period_steps {
            let x = at.step();
            crossings += u32::from(last < 0.0 && x >= 0.0);
            peak = peak.max(x.abs());
            last = x;
        }
        assert!((99..=101).contains(&crossings), "{crossings} cycles in a hundred predicted periods");
        assert!(peak > 0.1 && peak < 10.0, "at the edge the amplitude wandered to {peak}");
    }

    #[test]
    fn the_discrete_edge_converges_on_pi_over_two_tau() {
        let tau = 0.04;
        assert!((critical_gain(tau).unwrap() - 39.269_908_169_872_416).abs() < 1e-12);
        // 40 ms round the loop rings at 6.25 Hz — where ankle clonus is.
        assert_eq!(ringing_period(tau), Some(0.16));
        assert!((1.0 / ringing_period(tau).unwrap() - 6.25).abs() < 1e-12);
        let mut last_gap = f64::INFINITY;
        for m in [4usize, 40, 400, 4000] {
            let h = tau / m as f64;
            let discrete = critical_gain_discrete(m, h).unwrap();
            // To first order the discrete edge is π/(2τ + h): the step adds half of itself to the
            // delay. What is left is the sine's cubic term, relative size (π/(4m+2))²/6.
            let first_order = PI / (2.0 * tau + h);
            let x = PI / (4.0 * m as f64 + 2.0);
            assert!((discrete / first_order - 1.0 + x * x / 6.0).abs() < x.powi(4), "m = {m}");
            let gap = (discrete - critical_gain(tau).unwrap()).abs();
            assert!(gap < last_gap);
            last_gap = gap;
            assert!((ringing_period_discrete(m, h).unwrap() - (4.0 * tau + 2.0 * h)).abs() < 1e-15);
        }
        assert!(last_gap < 0.01, "at 4000 steps per delay the edges still differ by {last_gap}");
        // No delay at all: the Euler limit 2/h, with a two-step flip.
        assert_eq!(critical_gain_discrete(0, 0.5), Some(4.0));
        assert_eq!(ringing_period_discrete(0, 0.5), Some(1.0));
        for none in [critical_gain(0.0), critical_gain(f64::NAN), ringing_period(-1.0), critical_gain_discrete(3, 0.0), ringing_period_discrete(3, f64::INFINITY)] {
            assert_eq!(none, None);
        }
    }

    #[test]
    fn the_loop_delays_by_exactly_the_steps_it_was_given() {
        // Gain 1, h = 0.5, m = 2, held at 8: the first three steps each see the held value.
        let mut r = DelayedReflex::new(1.0, 0.5, 2, 8.0).unwrap();
        assert_eq!(r.delay_steps(), 2);
        assert_eq!((r.step(), r.step(), r.step()), (4.0, 0.0, -4.0));
        // The fourth sees the first computed value, 4: −4 − 0.5·4 = −6.
        assert_eq!(r.step(), -6.0);
        assert_eq!(r.x(), -6.0);
        // No delay and zero gain: nothing moves.
        let mut still = DelayedReflex::new(0.0, 0.1, 0, 3.0).unwrap();
        assert_eq!((still.delay_steps(), still.step()), (0, 3.0));
    }

    #[test]
    fn bad_arguments_are_refused() {
        assert!(matches!(Spindle::new(-1.0, 1.0, 1.0, 0.6), Err(ProprioError::OutOfRange { what: "baseline", .. })));
        assert!(matches!(Spindle::new(1.0, -1.0, 1.0, 0.6), Err(ProprioError::OutOfRange { what: "length_gain", .. })));
        assert!(matches!(Spindle::new(1.0, 1.0, f64::NAN, 0.6), Err(ProprioError::OutOfRange { what: "velocity_gain", .. })));
        assert!(matches!(Spindle::new(1.0, 1.0, 1.0, 0.0), Err(ProprioError::OutOfRange { what: "exponent", .. })));
        assert!(matches!(Spindle::new(1.0, 1.0, 1.0, 1.2), Err(ProprioError::OutOfRange { what: "exponent", .. })));
        let ia = Spindle::hamstring_form(1.0, 1.0).unwrap();
        assert!(matches!(ia.rate(f64::NAN, 0.0), Err(ProprioError::NonFinite { what: "stretch" })));
        assert!(matches!(ia.rate(0.0, f64::INFINITY), Err(ProprioError::NonFinite { what: "velocity" })));
        assert!(matches!(TendonOrgan::new(0.0, 0.0), Err(ProprioError::OutOfRange { what: "gain", .. })));
        assert!(matches!(TendonOrgan::new(1.0, -0.1), Err(ProprioError::OutOfRange { what: "threshold", .. })));
        assert!(matches!(TendonOrgan::new(1.0, 0.0).unwrap().rate(f64::NAN), Err(ProprioError::NonFinite { what: "force" })));
        let mut enc = SpikeEncoder::default();
        assert!(matches!(enc.step(0.0, 10.0), Err(ProprioError::OutOfRange { what: "dt", .. })));
        assert!(matches!(enc.step(1e-3, -1.0), Err(ProprioError::OutOfRange { what: "rate", .. })));
        assert!(matches!(enc.step(1e-3, f64::NAN), Err(ProprioError::OutOfRange { what: "rate", .. })));
        assert_eq!(enc, SpikeEncoder::default(), "a refused step changed nothing");
        assert!(matches!(DelayedReflex::new(-1.0, 1e-3, 5, 0.0), Err(ProprioError::OutOfRange { what: "gain", .. })));
        assert!(matches!(DelayedReflex::new(1.0, 0.0, 5, 0.0), Err(ProprioError::OutOfRange { what: "dt", .. })));
        assert!(matches!(DelayedReflex::new(1.0, 1e-3, MAX_DELAY_STEPS + 1, 0.0), Err(ProprioError::OutOfRange { what: "delay_steps", .. })));
        assert!(matches!(DelayedReflex::new(1.0, 1e-3, 5, f64::NAN), Err(ProprioError::NonFinite { what: "x0" })));
    }

    // ---- fusimotor gains ---------------------------------------------------------------------

    #[test]
    fn each_fusimotor_gain_multiplies_its_own_term_and_no_other() {
        let ia = Spindle::hamstring_form(3.0, 5.0).unwrap();
        assert_eq!(ia.rate_under(Fusimotor::NEUTRAL, 4.0, 10.0).unwrap(), ia.rate(4.0, 10.0).unwrap());
        let drive = Fusimotor::new(2.5, 0.5).unwrap();
        // Length term 3·4 = 12 → 6 under γ_s = ½; velocity term 5·10^0.6 → 2.5 times it.
        let dynamic = 5.0 * 10f64.powf(0.6);
        assert!((ia.rate_under(drive, 4.0, 0.0).unwrap() - (82.0 + 6.0)).abs() < 1e-12);
        assert!((ia.rate_under(drive, 0.0, 10.0).unwrap() - (82.0 + 2.5 * dynamic)).abs() < 1e-12);
        assert!((ia.rate_under(drive, 4.0, 10.0).unwrap() - (82.0 + 6.0 + 2.5 * dynamic)).abs() < 1e-12);
        // Shortening: the dynamic term is odd, and a high dynamic gain silences the afferent sooner.
        assert!((ia.rate_under(drive, 0.0, -10.0).unwrap() - (82.0 - 2.5 * dynamic)).abs() < 1e-12);
        assert_eq!(ia.rate_under(Fusimotor::new(10.0, 1.0).unwrap(), 0.0, -10.0).unwrap(), 0.0);
        assert!(ia.rate_under(Fusimotor::NEUTRAL, 0.0, -10.0).unwrap() > 0.0);
        assert!(Fusimotor::new(-1.0, 1.0).is_err() && Fusimotor::new(1.0, f64::NAN).is_err());
        assert!(ia.rate_under(Fusimotor { dynamic_gain: 1.0, static_gain: -2.0 }, 1.0, 0.0).is_err());
        assert!(ia.rate_under(Fusimotor::NEUTRAL, f64::NAN, 0.0).is_err());
        assert!(ia.rate_under(Fusimotor::NEUTRAL, 0.0, f64::INFINITY).is_err());
    }

    #[test]
    fn the_static_gain_sets_the_spikes_a_held_stretch_costs() {
        // Hold a 4-unit stretch for 2.503 s through the exact encoder: ⌊(82 + γ_s·3·4)·2.503⌋
        // spikes — 205.2, 235.3 and 265.3 before the floor, clear of a whole number.
        let ia = Spindle::hamstring_form(3.0, 5.0).unwrap();
        let count = |gamma_s: f64| {
            let drive = Fusimotor::new(1.0, gamma_s).unwrap();
            let mut encoder = SpikeEncoder::default();
            for _ in 0..2503 {
                encoder.step(1e-3, ia.rate_under(drive, 4.0, 0.0).unwrap()).unwrap();
            }
            encoder.spikes
        };
        assert_eq!(count(0.0), 205);
        assert_eq!(count(1.0), 235);
        assert_eq!(count(2.0), 265);
        // So the stretch itself costs γ_s · k_L · x · T spikes: 30 at unit gain, 60 at two.
        assert_eq!(count(2.0) - count(0.0), 2 * (count(1.0) - count(0.0)));
    }

    #[test]
    fn fusimotor_activation_saturates_by_the_hill_law_and_by_spikes() {
        assert_eq!(hill_activation(60.0, 60.0, 2.0), Some(0.5));
        assert_eq!(hill_activation(0.0, 60.0, 2.0), Some(0.0));
        assert!((hill_activation(120.0, 60.0, 2.0).unwrap() - 0.8).abs() < 1e-15);
        assert!((hill_activation(120.0, 60.0, 1.0).unwrap() - 2.0 / 3.0).abs() < 1e-15);
        assert_eq!(hill_activation(1e300, 1e-300, 2.0), Some(1.0));
        assert!(hill_activation(-1.0, 60.0, 2.0).is_none() && hill_activation(1.0, 0.0, 2.0).is_none());
        assert!(hill_activation(1.0, 60.0, 0.0).is_none() && hill_activation(f64::NAN, 60.0, 2.0).is_none());

        // The spike-driven form. One spike from rest is `r`; a second straight after takes `r` of
        // what is left; between spikes it decays as e^{−t/τ}.
        let mut f = Activation::new(0.25, 0.1).unwrap();
        assert_eq!(f.spike(), 0.25);
        assert_eq!(f.spike(), 0.25 + 0.25 * 0.75);
        let before = f.f;
        assert!((f.decay(0.05).unwrap() - before * (-0.5f64).exp()).abs() < 1e-15);
        // A regular 40 Hz train settles with its peaks at r / (1 − (1 − r) e^{−1/(ντ)}).
        let mut f = Activation::new(0.25, 0.1).unwrap();
        let mut peak = 0.0;
        for _ in 0..400 {
            f.decay(1.0 / 40.0).unwrap();
            peak = f.spike();
        }
        let want = 0.25 / (1.0 - 0.75 * (-0.25f64).exp());
        assert!((peak - want).abs() < 1e-12 && (f.settled_peak(40.0).unwrap() - want).abs() < 1e-15);
        assert!((want - 0.601_6).abs() < 1e-3, "the settled 40 Hz peak is {want}");
        // It rises with the rate and cannot reach one.
        let (slow, fast) = (f.settled_peak(10.0).unwrap(), f.settled_peak(1e4).unwrap());
        assert!(slow < want && want < fast && fast < 1.0);
        assert!(f.settled_peak(0.0).is_none() && f.decay(-1.0).is_err());
        assert!(Activation::new(0.0, 0.1).is_err() && Activation::new(1.0, 0.1).is_err() && Activation::new(0.5, 0.0).is_err());

        assert_eq!(occlusion(50.0, 20.0, 0.156), 50.0 + 0.156 * 20.0);
        assert_eq!(occlusion(20.0, 50.0, 0.156), 50.0 + 0.156 * 20.0);
    }

    // ---- the intrafusal fibre ----------------------------------------------------------------

    /// ILLUSTRATIVE constants of the order used for a bag-type fibre; NOT the published fit, which
    /// this review could not read (see the module documentation).
    fn fibre() -> IntrafusalFibre {
        IntrafusalFibre {
            k_sr: 10.0,
            k_pr: 0.15,
            c_lengthening: 1.0,
            c_shortening: 0.4,
            beta0: 0.06,
            beta_dynamic: 0.26,
            beta_static: -0.02,
            force_dynamic: 0.03,
            force_static: 0.05,
            a: 0.3,
            r: 0.46,
            l0_sr: 0.04,
            l0_pr: 0.76,
            tension: 0.03,
        }
    }

    #[test]
    fn a_held_fibre_carries_two_springs_in_series_plus_the_active_force() {
        let mut fibre = fibre();
        for &(length, fd, fs) in &[(1.0, 0.0, 0.0), (1.05, 0.7, 0.0), (0.95, 0.0, 1.0), (1.1, 0.4, 0.6)] {
            let t_star = fibre.equilibrium_tension(length, fd, fs).unwrap();
            // Referee: the constitutive law itself. At rest the polar region carries
            // K_PR (L_PR − L0_PR) + Γ, and that must equal the tension that sets L_PR.
            fibre.tension = t_star;
            assert!((fibre.polar_tension(length, 0.0, fd, fs).unwrap() - t_star).abs() < 1e-15);
            // Zero up to the rounding of `T*`: an ulp of force through the damper's 1/a power.
            assert!(fibre.tension_rate(length, 0.0, fd, fs).unwrap().abs() < 1e-40);
            // Either side of it the tension heads back.
            fibre.tension = t_star + 0.01;
            assert!(fibre.tension_rate(length, 0.0, fd, fs).unwrap() < 0.0);
            fibre.tension = t_star - 0.01;
            assert!(fibre.tension_rate(length, 0.0, fd, fs).unwrap() > 0.0);
        }
        // The numbers: at L = 1 undriven, 0.15·0.2 / 1.015.
        assert!((fibre.equilibrium_tension(1.0, 0.0, 0.0).unwrap() - 0.03 / 1.015).abs() < 1e-15);
        // The length sensitivity is the series stiffness and does NOT depend on the drive …
        let slope = |fd: f64, fs: f64| (fibre.equilibrium_tension(1.1, fd, fs).unwrap() - fibre.equilibrium_tension(1.0, fd, fs).unwrap()) / 0.1;
        let series = 10.0 * 0.15 / 10.15;
        assert!((slope(0.0, 0.0) - series).abs() < 1e-13 && (slope(1.0, 1.0) - series).abs() < 1e-13);
        // … which enters as a bias: Γ K_SR / (K_SR + K_PR).
        let bias = fibre.equilibrium_tension(1.0, 0.0, 1.0).unwrap() - fibre.equilibrium_tension(1.0, 0.0, 0.0).unwrap();
        assert!((bias - 0.05 * 10.0 / 10.15).abs() < 1e-15);
        let bias = fibre.equilibrium_tension(1.0, 1.0, 0.0).unwrap() - fibre.equilibrium_tension(1.0, 0.0, 0.0).unwrap();
        assert!((bias - 0.03 * 10.0 / 10.15).abs() < 1e-15);
        // In afferent terms, with gain 20 000 and threshold length 0.0423:
        fibre.tension = fibre.equilibrium_tension(1.0, 0.0, 0.0).unwrap();
        let want = 20_000.0 * (fibre.tension / 10.0 - (0.0423 - 0.04));
        assert!((fibre.primary_contribution(20_000.0, 0.0423) - want).abs() < 1e-9 && want > 10.0);
        fibre.tension = 0.0;
        assert_eq!(fibre.primary_contribution(20_000.0, 0.0423), 0.0);
    }

    #[test]
    fn both_fusimotor_drives_move_the_damping_and_the_active_force() {
        // β = β₀ + β₁ f_d + β₂ f_s and Γ = Γ₁ f_d + Γ₂ f_s, read off one at a time. The
        // self-consistency of the tension rate cannot see these — the same β appears on both
        // sides of it — so they are checked against their arithmetic here.
        let fibre = fibre();
        assert_eq!(fibre.beta(0.0, 0.0), 0.06);
        assert!((fibre.beta(1.0, 0.0) - 0.32).abs() < 1e-15 && (fibre.beta(0.0, 1.0) - 0.04).abs() < 1e-15);
        assert!((fibre.beta(0.5, 0.5) - (0.06 + 0.13 - 0.01)).abs() < 1e-15);
        assert_eq!(fibre.active_force(0.0, 0.0), 0.0);
        assert!((fibre.active_force(1.0, 0.0) - 0.03).abs() < 1e-15 && (fibre.active_force(0.0, 1.0) - 0.05).abs() < 1e-15);
        // The damping is what the drives buy in the rate: at the same state, more damping means a
        // polar region that yields more slowly, so the sensory region takes up more of the stretch.
        // Each drive is isolated from the active force it also exerts, because the two pull the
        // tension opposite ways and their sum is not monotone in either.
        let mut probe = fibre;
        probe.tension = 0.08;
        probe.force_dynamic = 0.0;
        probe.force_static = 0.0;
        let rate = |fd: f64, fs: f64| probe.tension_rate(1.05, 0.3, fd, fs).unwrap();
        assert!(rate(1.0, 0.0) > rate(0.0, 0.0), "β₁ is positive, so dynamic drive must raise the rate");
        assert!(rate(0.0, 1.0) < rate(0.0, 0.0), "this fibre's β₂ is negative, so static drive must lower it");
        // And the active force alone, with the damping held fixed, pulls the tension up.
        let mut still = fibre;
        still.tension = 0.08;
        still.beta_dynamic = 0.0;
        still.beta_static = 0.0;
        let forced = |fd: f64, fs: f64| still.tension_rate(1.05, 0.3, fd, fs).unwrap();
        assert!(forced(1.0, 0.0) > forced(0.0, 0.0) && forced(0.0, 1.0) > forced(1.0, 0.0), "Γ₂ > Γ₁ > 0");
    }

    #[test]
    fn the_tension_rate_satisfies_the_constitutive_law() {
        // Whatever rate the fibre reports, the polar region's velocity it implies —
        // L̇_PR = L̇ − Ṫ / K_SR — must, put back through the damper, carry exactly the tension.
        // This is the test that tells the exponent 1/a from the a printed in the source's eq. 8.
        let mut fibre = fibre();
        let mut checked = (0, 0);
        for &(tension, length, velocity, fd, fs) in &[
            (0.05, 1.0, 0.0, 0.0, 0.0),
            (0.02, 1.0, 0.0, 0.0, 0.0),
            (0.08, 1.05, 0.3, 1.0, 0.0),
            (0.01, 0.98, -0.3, 0.0, 1.0),
            (0.12, 1.08, 1.0, 0.5, 0.5),
        ] {
            fibre.tension = tension;
            let rate = fibre.tension_rate(length, velocity, fd, fs).unwrap();
            let polar_velocity = velocity - rate / fibre.k_sr;
            if polar_velocity > 0.0 { checked.0 += 1 } else { checked.1 += 1 }
            let carried = fibre.polar_tension(length, polar_velocity, fd, fs).unwrap();
            assert!((carried - tension).abs() < 1e-13, "T = {tension}: the damper carries {carried}");
        }
        assert!(checked.0 >= 2 && checked.1 >= 2, "both the lengthening and shortening branches: {checked:?}");
    }

    /// Ramp-and-hold of the illustrative fibre; returns the tension at the end of the ramp and
    /// after the hold.
    fn ramp_and_hold(fd: f64, dt: f64) -> (f64, f64) {
        let mut fibre = fibre();
        fibre.tension = fibre.equilibrium_tension(0.95, fd, 0.0).unwrap();
        let (velocity, ramp_s, hold_s) = (0.5, 0.2, 1.0);
        let steps = (ramp_s / dt).round() as usize;
        for n in 0..steps {
            fibre.step(dt, 0.95 + velocity * dt * (n + 1) as f64, velocity, fd, 0.0).unwrap();
        }
        let peak = fibre.tension;
        for _ in 0..(hold_s / dt).round() as usize {
            fibre.step(dt, 0.95 + velocity * ramp_s, 0.0, fd, 0.0).unwrap();
        }
        (peak, fibre.tension)
    }

    /// The illustrative fibre at half dynamic drive, `t_end` seconds into a 0.5 L0/s ramp from
    /// rest at 0.95 L0, by the implicit step.
    fn ramp_to(t_end: f64, dt: f64) -> f64 {
        let mut fibre = fibre();
        fibre.tension = fibre.equilibrium_tension(0.95, 0.5, 0.0).unwrap();
        for n in 0..(t_end / dt).round() as usize {
            fibre.step(dt, 0.95 + 0.5 * dt * (n + 1) as f64, 0.5, 0.5, 0.0).unwrap();
        }
        fibre.tension
    }

    #[test]
    fn the_implicit_step_converges_at_first_order_on_a_fine_explicit_integration() {
        // 20 ms into the ramp, while the tension is still catching the stretch up (by 200 ms it
        // is in a quasi-steady state every step size agrees on, which would test nothing).
        // Referee: forward Euler at a step a thousand times finer, which shares no code with the
        // implicit solve beyond the rate itself.
        let mut fine = fibre();
        fine.tension = fine.equilibrium_tension(0.95, 0.5, 0.0).unwrap();
        let h = 1e-6;
        for n in 0..20_000 {
            let rate = fine.tension_rate(0.95 + 0.5 * h * n as f64, 0.5, 0.5, 0.0).unwrap();
            fine.tension += h * rate;
        }
        let error = |dt: f64| (ramp_to(0.02, dt) - fine.tension).abs();
        let (coarse, half) = (error(2e-3), error(1e-3));
        assert!(coarse < 0.02 * fine.tension, "a 2 ms step is {coarse} from {}", fine.tension);
        assert!(half > 1e-5, "the comparison has to be made where the step size still matters: {half}");
        let ratio = coarse / half;
        assert!(ratio > 1.8 && ratio < 2.2, "halving the step divided the error by {ratio}");
    }

    #[test]
    fn dynamic_gamma_drive_raises_the_response_to_stretch_velocity_and_not_the_held_one() {
        let run = |fd: f64| {
            let (peak, held) = ramp_and_hold(fd, 1e-3);
            let rest = fibre().equilibrium_tension(1.05, fd, 0.0).unwrap();
            (peak - rest, held - rest)
        };
        let (off, half, full) = (run(0.0), run(0.5), run(1.0));
        // MEASURED on the illustrative fibre: the overshoot at the end of a 0.5 L0/s ramp, above
        // the tension the final length holds at rest.
        assert!(off.0 > 0.0 && half.0 > 1.5 * off.0 && full.0 > half.0, "dynamic overshoots {} {} {}", off.0, half.0, full.0);
        // After a second's hold the overshoot has relaxed most of the way, from above, at every
        // drive: the held response is the equilibrium, which the damping does not enter.
        for (overshoot, left) in [off, half, full] {
            assert!(left >= 0.0 && left < 0.5 * overshoot, "a second later {left} of {overshoot} is left");
        }
    }

    #[test]
    fn a_step_far_too_long_for_an_explicit_method_lands_between_the_start_and_the_equilibrium() {
        // From well above and well below the held tension, one step of 0.1 s. The damper's law is
        // a 3.3rd power, so forward Euler from the high side throws the tension far past the
        // equilibrium (the rate there is of order −10⁴/s); the implicit step cannot overshoot.
        for start in [0.5, -0.2] {
            let mut fibre = fibre();
            let rest = fibre.equilibrium_tension(1.0, 0.0, 0.0).unwrap();
            fibre.tension = start;
            let explicit = start + 0.1 * fibre.tension_rate(1.0, 0.0, 0.0, 0.0).unwrap();
            assert!((explicit - rest) * (start - rest) < 0.0, "the explicit step should overshoot from {start}: {explicit}");
            let after = fibre.step(0.1, 1.0, 0.0, 0.0, 0.0).unwrap();
            assert!((after - rest) * (start - rest) > 0.0 && (after - rest).abs() < (start - rest).abs(), "{start} → {after} around {rest}");
            // And it is the backward-Euler point: T' − T = dt · rate(T').
            let residual = after - start - 0.1 * fibre.tension_rate(1.0, 0.0, 0.0, 0.0).unwrap();
            assert!(residual.abs() < 1e-9, "residual {residual}");
        }
    }

    #[test]
    fn the_fibre_refuses_what_it_cannot_integrate() {
        let mut bad = fibre();
        bad.a = 0.0;
        assert!(bad.equilibrium_tension(1.0, 0.0, 0.0).is_err());
        let mut bad = fibre();
        bad.beta_static = -0.07;
        assert!(bad.tension_rate(1.0, 0.0, 0.0, 0.0).is_err(), "β would go negative at full static drive");
        let mut fibre = fibre();
        assert!(fibre.tension_rate(1.0, 0.0, 1.5, 0.0).is_err() && fibre.tension_rate(1.0, 0.0, 0.0, -0.1).is_err());
        assert!(fibre.step(0.0, 1.0, 0.0, 0.0, 0.0).is_err() && fibre.step(1e-3, f64::NAN, 0.0, 0.0, 0.0).is_err());
        // A fascicle so short that the polar region is inside R: the damper has no force to give.
        assert!(matches!(fibre.tension_rate(0.45, 0.0, 0.0, 0.0), Err(ProprioError::OutOfRange { what: "polar length above R", .. })));
        assert!(fibre.step(1e-3, 0.45, 0.0, 0.0, 0.0).is_err());
        assert_eq!(fibre.tension, 0.03, "a refused step must not move the state");
    }

    // ---- the stretch reflex ------------------------------------------------------------------

    fn reflex(drive: Fusimotor, synaptic_gain: f64, stiffness: f64, m: usize) -> StretchReflex {
        // No velocity response, so the loop is the linear one the closed forms are for.
        let spindle = Spindle::new(82.0, 3.0, 0.0, 0.6).unwrap();
        StretchReflex::new(spindle, drive, synaptic_gain, stiffness, 2.0, 1e-3, m).unwrap()
    }

    #[test]
    fn the_reflex_is_a_spring_of_stiffness_synaptic_gain_times_static_gain_times_length_gain() {
        for &(gamma_s, g, k, m) in &[(1.0, 0.5, 0.0, 10usize), (2.0, 0.5, 0.0, 10), (1.0, 0.5, 4.0, 0), (0.5, 1.0, 1.0, 25)] {
            let mut limb = reflex(Fusimotor::new(1.0, gamma_s).unwrap(), g, k, m);
            assert_eq!(limb.reflex_stiffness(), g * gamma_s * 3.0);
            let want = 6.0 / (k + g * gamma_s * 3.0);
            assert_eq!(limb.settled_displacement(6.0), Some(want));
            for _ in 0..60_000 {
                limb.step(6.0).unwrap();
            }
            assert!((limb.x - want).abs() < 1e-9 * want, "γ_s = {gamma_s}, g = {g}, k = {k}: settled at {} for {want}", limb.x);
        }
        // Doubling the static gain halves the give under the same load: stiff against compliant.
        let give = |gamma_s: f64| reflex(Fusimotor::new(1.0, gamma_s).unwrap(), 0.5, 0.0, 10).settled_displacement(6.0).unwrap();
        assert_eq!(give(2.0), 0.5 * give(1.0));
        assert!(reflex(Fusimotor::new(1.0, 0.0).unwrap(), 0.5, 0.0, 10).settled_displacement(6.0).is_none());
    }

    /// Tap the limb, then the early and late peak displacement, as [`envelope`] does.
    fn tapped(limb: &mut StretchReflex, period_steps: usize) -> (f64, f64) {
        for _ in 0..5 {
            limb.step(1.0).unwrap();
        }
        let mut peak = |steps: usize| (0..steps).map(|_| limb.step(0.0).unwrap().abs()).fold(0.0f64, f64::max);
        peak(10 * period_steps);
        let early = peak(period_steps);
        peak(30 * period_steps);
        (early, peak(period_steps))
    }

    #[test]
    fn clonus_starts_where_the_product_of_the_gains_crosses_the_delay_s_ceiling() {
        // A 32 ms loop at a 1 ms step. The boundary is on g·γ_s·k_L / b, so it can be crossed by
        // the synaptic gain or by the fusimotor gain alone.
        let m = 32usize;
        let period_steps = 4 * m + 2;
        let edge = critical_gain_discrete(m, 1e-3).unwrap();
        let at_unit_drive = reflex(Fusimotor::NEUTRAL, 1.0, 0.0, m).critical_synaptic_gain().unwrap();
        assert!((at_unit_drive - 2.0 * edge / 3.0).abs() < 1e-12);
        assert!((reflex(Fusimotor::new(1.0, 4.0).unwrap(), 1.0, 0.0, m).critical_synaptic_gain().unwrap() - at_unit_drive / 4.0).abs() < 1e-12);
        assert!(reflex(Fusimotor::new(1.0, 0.0).unwrap(), 1.0, 0.0, m).critical_synaptic_gain().is_none());

        let (early, late) = tapped(&mut reflex(Fusimotor::NEUTRAL, 0.98 * at_unit_drive, 0.0, m), period_steps);
        assert!(late < 0.5 * early, "2% under: {early} → {late}");
        let (early, late) = tapped(&mut reflex(Fusimotor::NEUTRAL, 1.02 * at_unit_drive, 0.0, m), period_steps);
        assert!(late > 2.0 * early, "2% over by synaptic gain: {early} → {late}");
        // The same synaptic gain that was safe, with the static fusimotor gain turned up 4%.
        let (early, late) = tapped(&mut reflex(Fusimotor::new(1.0, 1.04).unwrap(), 0.98 * at_unit_drive, 0.0, m), period_steps);
        assert!(late > 2.0 * early, "over by fusimotor gain alone: {early} → {late}");
        // And the dynamic gain, with no velocity response to act through, changes nothing.
        let (early_d, late_d) = tapped(&mut reflex(Fusimotor::new(9.0, 1.0).unwrap(), 0.98 * at_unit_drive, 0.0, m), period_steps);
        let (early_n, late_n) = tapped(&mut reflex(Fusimotor::NEUTRAL, 0.98 * at_unit_drive, 0.0, m), period_steps);
        assert_eq!((early_d, late_d), (early_n, late_n));
    }

    #[test]
    fn one_step_of_the_velocity_loop_is_the_arithmetic_it_claims() {
        // Length gain zero, velocity gain 5 at exponent 0.6, γ_d = 2, no delay, b = 2, h = 1 ms.
        let spindle = Spindle::new(82.0, 0.0, 5.0, 0.6).unwrap();
        let mut limb = StretchReflex::new(spindle, Fusimotor::new(2.0, 1.0).unwrap(), 0.25, 0.0, 2.0, 1e-3, 0).unwrap();
        // Step 1 under load 4: nothing has moved yet, so x = h F / b = 0.002, at velocity 2.
        assert_eq!(limb.step(4.0).unwrap(), 1e-3 * 4.0 / 2.0);
        // Step 2 sees the afferent that velocity raised: 2·5·2^0.6 above rest.
        let above_rest = 2.0 * 5.0 * 2f64.powf(0.6);
        let want = 0.002 + 1e-3 / 2.0 * (4.0 - 0.25 * above_rest);
        assert!((limb.step(4.0).unwrap() - want).abs() < 1e-15);
        assert!(limb.step(f64::NAN).is_err());
        // A loop wound up until the displacement overflows says so rather than returning an
        // infinity for the next step to turn into a NaN.
        let mut runaway = StretchReflex::new(spindle, Fusimotor::NEUTRAL, 0.0, 0.0, f64::MIN_POSITIVE, 1e-3, 0).unwrap();
        let mut refused = None;
        for _ in 0..10_000 {
            if let Err(e) = runaway.step(f64::MAX) {
                refused = Some(e);
                break;
            }
        }
        assert_eq!(refused, Some(ProprioError::NonFinite { what: "displacement" }), "the loop never overflowed, so the guard was never reached");
        assert!(StretchReflex::new(spindle, Fusimotor::NEUTRAL, -1.0, 0.0, 2.0, 1e-3, 0).is_err());
        assert!(StretchReflex::new(spindle, Fusimotor::NEUTRAL, 1.0, -1.0, 2.0, 1e-3, 0).is_err());
        assert!(StretchReflex::new(spindle, Fusimotor::NEUTRAL, 1.0, 0.0, 0.0, 1e-3, 0).is_err());
        assert!(StretchReflex::new(spindle, Fusimotor::NEUTRAL, 1.0, 0.0, 2.0, 0.0, 0).is_err());
        assert!(StretchReflex::new(spindle, Fusimotor::NEUTRAL, 1.0, 0.0, 2.0, 1e-3, MAX_DELAY_STEPS + 1).is_err());
    }

    #[test]
    fn past_the_ceiling_the_silent_afferent_turns_divergence_into_a_sustained_oscillation() {
        // 5% over the boundary the LINEAR loop grows without limit (tested above on
        // `DelayedReflex`). This one cannot: once the limb has shortened past 82/(γ_s k_L) = 27.3
        // the afferent is silent and the reflex has nothing more to withdraw, so the growth stops
        // and an oscillation of fixed size remains — clonus rather than an overflow.
        let m = 32usize;
        let critical = reflex(Fusimotor::NEUTRAL, 1.0, 0.0, m).critical_synaptic_gain().unwrap();
        let mut limb = reflex(Fusimotor::NEUTRAL, 1.05 * critical, 0.0, m);
        for _ in 0..5 {
            limb.step(1.0).unwrap();
        }
        let mut block = |steps: usize| {
            let (mut low, mut high, mut cycles, mut last) = (f64::INFINITY, f64::NEG_INFINITY, 0u32, limb.x);
            for _ in 0..steps {
                let x = limb.step(0.0).unwrap();
                (low, high) = (low.min(x), high.max(x));
                cycles += u32::from(last < 0.0 && x >= 0.0);
                last = x;
            }
            (low, high, cycles)
        };
        block(20_000);
        let (low_a, high_a, _) = block(13_000);
        let (low_b, high_b, cycles) = block(13_000);
        // Sustained and of fixed size (MEASURED: it swings between −33.6 and +31.3) …
        assert!((low_a - low_b).abs() < 1e-3 * low_a.abs() && (high_a - high_b).abs() < 1e-3 * high_a);
        // … the shortening swing having gone past the point of silence, which is what bounds it …
        assert!(low_b < -82.0 / 3.0 && low_b > -2.0 * 82.0 / 3.0 && high_b > 0.5 * 82.0 / 3.0, "[{low_b}, {high_b}]");
        // … at the frequency the delay sets: 13 s is a hundred periods of (4m + 2) h = 130 ms.
        assert!((98..=102).contains(&cycles), "{cycles} cycles in a hundred predicted periods");
    }
}
