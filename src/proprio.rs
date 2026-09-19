//! Proprioception: what a muscle tells the spinal cord about its own length, speed and force, and
//! what happens when that report comes back late — the power-law spindle, the tendon organ, an
//! exact rate-to-spike encoder, and a reflex loop whose delay sets the gain at which it rings.
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
//! The report travels at a finite speed. A stretch reflex is therefore a feedback loop with a
//! dead time — tens of milliseconds for a human ankle — and a loop with a dead time has a gain it
//! cannot exceed without oscillating. That ceiling, and the frequency it rings at, depend on the
//! delay alone ([`DelayedReflex`]).
//!
//! # Why it is in a neuromorphic crate
//!
//! A robot limb with spiking proprioceptors is the standard neuromorphic control demonstration,
//! and its two recurring problems are here in closed form: turning a rate model into spikes
//! without changing the rate ([`SpikeEncoder`]), and knowing how much reflex gain a given
//! conduction-plus-computation delay can carry before the limb shakes.
//!
//! # The closed forms this module is checked against
//!
//! - **The spindle's power law.** Doubling the lengthening velocity multiplies the dynamic
//!   response by `2^p` — `1.516` at `p = 0.6`, not 2 — and the response is odd in velocity until
//!   the rate reaches zero, where the afferent falls silent, as spindles do in rapid shortening.
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
//! - Fusimotor (gamma) drive, which retunes a spindle during movement; intrafusal mechanics; the
//!   initial burst and history dependence of real afferents.
//! - The tendon organ's dynamic response (Houk and Simon, 1967); [`TendonOrgan`] is its static
//!   force–rate line only.
//! - A reflex loop through the power-law term, a muscle, or a limb with inertia. The closed form
//!   is for the linear first-order loop, and that is the loop simulated.

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
}
