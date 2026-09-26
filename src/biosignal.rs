//! Biosignals for a spiking front end: a synthetic electrocardiogram, a level-crossing encoder, a
//! spiking R-peak detector and rhythm monitor, and the electromyogram's amplitude read straight
//! off an event rate.
//!
//! # What the mechanism is
//!
//! **The ECG.** `McSharry`, Clifford, Tarassenko and Smith (*A dynamical model for generating
//! synthetic electrocardiogram signals*, IEEE Transactions on Biomedical Engineering
//! 50(3):289–294, 2003) generate a realistic ECG from a point circling the unit circle once per
//! heartbeat, while a third coordinate is pushed up and down as the point passes five angles —
//! the P, Q, R, S and T waves:
//!
//! ```text
//! θ̇ = 2π / RR        ż = −Σᵢ aᵢ Δθᵢ exp(−Δθᵢ² / 2bᵢ²) − (z − z₀(t))        Δθᵢ = θ − θᵢ
//! ```
//!
//! Each term of the sum is the `θ`-derivative of a Gaussian bump `aᵢ bᵢ² exp(−Δθᵢ²/2bᵢ²)`, so `z`
//! traces out five bumps per revolution, relaxing to a baseline `z₀` that breathes at the
//! respiratory frequency. The constants here are those of the authors' reference implementation
//! (ECGSYN, `PhysioNet`), read from its source: the wave angles, amplitudes and widths and
//! their rescaling with heart rate from `ecgsyn.m`, and the baseline wander from
//! `derivsecgsyn.m`, the right-hand side that `ecgsyn.m` hands to `ode45`.
//!
//! **Where the code and the paper differ.** This paragraph used to name `ecgsyn.m` alone, and did
//! not say that two of its angles are not the paper's. Table I of the paper gives
//! `θ_P = −π/3` and `θ_T = π/2`, that is −60° and 90°; `ecgsyn.m` has −70° and 100°, and so does
//! [`ECGSYN_ANGLES_DEG`]. Q, R, S and every `aᵢ` and `bᵢ` agree. The two baseline amplitudes
//! are in different units. The paper's is `A = 0.15 mV` in its equation (2),
//! `z₀(t) = A sin(2π f₂ t)`, in millivolts; ECGSYN's `0.005` is in the model's units, before the
//! finished record is rescaled to `[−0.4, 1.2]` mV ([`to_millivolts`]). Put through that rescale
//! at 60 beats per minute it comes to roughly 0.13 mV: 0.126 mV over a settled 16-second record,
//! 0.137 mV over one beat with the wander off, both measured in
//! `the_baseline_amplitude_is_in_the_models_units_and_the_papers_order_after_the_rescale`. That
//! is the paper's order, and `0.005` against `0.15 mV` is not a like-for-like disagreement.
//!
//! **The front end.** A neuromorphic sensor does not sample: it emits an UP or DOWN event each
//! time the signal has moved by `δ` since the last event ([`LevelCrossing`]) — silent on a flat
//! baseline, busy on a QRS complex. [`RPeakDetector`] is one leaky integrator of UP events with
//! a threshold and a refractory period: only the R wave's upstroke delivers events fast enough
//! to fire it. [`RhythmMonitor`] turns the detections into RR intervals and flags the ones that
//! leave a running mean by more than a tolerance.
//!
//! **The EMG.** A surface electromyogram is well described as band-limited Gaussian noise whose
//! standard deviation follows the muscle's activation (Hogan and Mann, *Myoelectric signal
//! processing: optimal estimation applied to electromyography — part I*, IEEE Transactions on
//! Biomedical Engineering 27(7):382–395, 1980). The conventional estimate of activation is
//! rectify-and-smooth. A level-crossing encoder makes that unnecessary: its event rate is the
//! signal's total variation per second divided by `δ`, and for Gaussian noise that is
//! proportional to the standard deviation — so the spike COUNT is the envelope.
//!
//! # Why it is in a neuromorphic crate
//!
//! Always-on biosignal monitoring is the application where event-driven sensing pays: the
//! signal is sparse in time and the power budget is a coin cell's. It is what Bauer, Muir and
//! Indiveri (*Real-time ultra-low power ECG anomaly detection using an event-driven neuromorphic
//! processor*, IEEE Transactions on Biomedical Circuits and Systems 13(6):1575–1582, 2019) and
//! Donati, Payvand, Risi, Krause and Indiveri (*Discrimination of EMG signals using a
//! neuromorphic implementation of a spiking neural network*, same journal 13(5):795–803, 2019)
//! built, both on delta-modulated inputs; this module is the signal side of those systems, with
//! each stage's behaviour in closed form.
//!
//! # The closed forms this module is checked against
//!
//! - **The ECG's drive is a derivative.** `−Σ aᵢ Δθᵢ exp(−Δθᵢ²/2bᵢ²)` against the numerical
//!   `θ`-derivative of the Gaussian sum, and continuous across `θ = ±π`.
//! - **The ECG solves its equation.** `z` is a linear filter of a known drive, so
//!   `z(t) = z(0) e^{−t} + ∫₀ᵗ e^{−(t−s)} [drive(θ(s)) + z₀(s)] ds`; the stepped signal is checked
//!   against Simpson quadrature of that integral, which shares nothing with the stepper but the
//!   drive.
//! - **The R wave.** R peaks fall where the phase crosses zero — at `(k + ½) RR` from a start at
//!   `θ = −π` — and the R wave stands `a_R b_R² / ω` above its surroundings, less what one QRS
//!   width of relaxation removes; the five waves come in order with the signs P+, Q−, R+, S−, T+.
//! - **Level crossing.** After every sample the reconstruction is within `δ` of the signal and
//!   equals `x₀ + δ (ups − downs)`; the event count times `δ` lies within `2δ` per sample below
//!   the signal's total variation and never above it.
//! - **The detector's rate threshold.** Under a regular event train at rate `r` the integrator
//!   peaks at `w / (1 − e^{−1/(rτ)})`, so it fires exactly when `r > −1 / (τ ln(1 − w/θ))`.
//! - **The rhythm monitor's mean** after `k` equal intervals `r` from `m₀`:
//!   `r + (m₀ − r)(1 − η)^k`.
//! - **EMG.** For Gaussian samples of standard deviation `aσ` the mean absolute value is
//!   `aσ √(2/π)`. The difference of two independent such samples is Gaussian of standard
//!   deviation `√2 aσ`, so the mean absolute increment is `√2 aσ √(2/π) = 2aσ/√π`, and the
//!   level-crossing event rate is `2aσ / (δ√π)` per sample — linear in the activation `a`. This
//!   item used to attribute the increment to a bare "(Rice)" that named no work. It needs no
//!   citation, since the arithmetic above is the whole of it. The work that name suggests is
//!   S. O. Rice, *Mathematical Analysis of Random Noise*, Bell System Technical Journal
//!   23(3):282–332 (1944), doi:10.1002/j.1538-7305.1944.tb00874.x, and 24(1):46–156 (1945),
//!   doi:10.1002/j.1538-7305.1945.tb00453.x; this review did not confirm that it was the one
//!   meant, and nothing here rests on it.
//!
//! # What this module has NOT reproduced
//!
//! - ECGSYN's RR-interval generator (a bimodal power spectrum for the Mayer and respiratory
//!   rhythms). The RR interval is an input here, beat by beat.
//! - One deliberate difference: `derivsecgsyn.m` (the ODE right-hand side distributed with
//!   `ecgsyn.m`, which reaches it only through `ode45`) takes `rem(θ − θᵢ, 2π)`, which does not
//!   bring the difference into `(−π, π]`, so the T wave's tail is cut off where the phase wraps.
//!   This module wraps it. The effect is of order `10⁻³` of the R wave's drive. This item used to
//!   say that `ecgsyn.m` takes the remainder; `ecgsyn.m` has no `rem` or `mod` call, and the line
//!   is `dti = rem(ta - ti, 2*pi);` in `derivsecgsyn.m`. The C version does the same with `fmod`.
//! - Real recordings, noise and artefact models, lead geometry, and any clinical claim: the
//!   detector is checked on the synthetic signal, and its figures are measured on it.
//! - The EMG's spectrum. The samples here are white; the closed forms need only Gaussianity of
//!   the samples and of their increments.

use crate::rng::Rng;
use core::f64::consts::{PI, TAU};
use core::fmt;

/// ECGSYN's wave angles for P, Q, R, S, T at 60 beats per minute, degrees: those of `ecgsyn.m`.
/// Table I of the paper gives −60° for P and 90° for T; see the module documentation.
pub const ECGSYN_ANGLES_DEG: [f64; 5] = [-70.0, -15.0, 0.0, 15.0, 100.0];
/// ECGSYN's wave amplitudes `aᵢ`.
pub const ECGSYN_A: [f64; 5] = [1.2, -5.0, 30.0, -7.5, 0.75];
/// ECGSYN's wave widths `bᵢ` at 60 beats per minute, radians.
pub const ECGSYN_B: [f64; 5] = [0.25, 0.1, 0.1, 0.1, 0.4];
/// ECGSYN's baseline wander, from `derivsecgsyn.m`: amplitude in the model's units, and
/// respiratory frequency in hertz. The amplitude is taken before the rescale to millivolts, so it
/// is not to be compared with the paper's `A = 0.15 mV` as it stands; see the module documentation.
pub const ECGSYN_BASELINE: (f64, f64) = (0.005, 0.25);

/// What went wrong, named rather than guessed around.
#[derive(Debug, Clone, PartialEq)]
pub enum BiosignalError {
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
}

impl fmt::Display for BiosignalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfRange { what, value } => write!(f, "{what} = {value} is out of range"),
            Self::NonFinite { what } => write!(f, "{what} is not finite"),
        }
    }
}

impl std::error::Error for BiosignalError {}

fn positive(what: &'static str, value: f64) -> Result<f64, BiosignalError> {
    if value > 0.0 && value.is_finite() { Ok(value) } else { Err(BiosignalError::OutOfRange { what, value }) }
}

fn finite(what: &'static str, value: f64) -> Result<f64, BiosignalError> {
    if value.is_finite() { Ok(value) } else { Err(BiosignalError::NonFinite { what }) }
}

/// An angle brought into `(−π, π]`.
#[must_use]
pub fn wrapped(angle: f64) -> f64 {
    let a = angle.rem_euclid(TAU);
    if a > PI { a - TAU } else { a }
}

// ---------------------------------------------------------------------------------------------
// The ECG
// ---------------------------------------------------------------------------------------------

/// The ECGSYN dynamical model, on its limit cycle.
#[derive(Debug, Clone, PartialEq)]
pub struct Ecg {
    /// Wave angles `θᵢ`, radians.
    pub angles: [f64; 5],
    /// Wave amplitudes `aᵢ`.
    pub a: [f64; 5],
    /// Wave widths `bᵢ`, radians.
    pub b: [f64; 5],
    /// Baseline wander amplitude.
    pub baseline: f64,
    /// Baseline wander frequency, hertz.
    pub respiration_hz: f64,
    /// Phase `θ`, in `(−π, π]`. The R wave is at zero.
    pub theta: f64,
    /// The signal `z`, in the model's units (ECGSYN rescales a finished record to millivolts).
    pub z: f64,
    /// Seconds elapsed.
    pub t: f64,
}

impl Ecg {
    /// ECGSYN's default morphology adjusted, as `ecgsyn.m` adjusts it, for a mean heart rate:
    /// with `h = √(bpm/60)` the widths are scaled by `h` and the angles of P, Q, S, T by
    /// `√h, h, h, √h`. Starts between beats, at `θ = −π`, `z = 0`.
    ///
    /// # Errors
    ///
    /// [`BiosignalError::OutOfRange`] for a heart rate that is not positive and finite.
    pub fn ecgsyn(mean_bpm: f64) -> Result<Self, BiosignalError> {
        let h = (positive("mean_bpm", mean_bpm)? / 60.0).sqrt();
        let scale = [h.sqrt(), h, 1.0, h, h.sqrt()];
        let mut angles = [0.0; 5];
        let mut b = [0.0; 5];
        for i in 0..5 {
            angles[i] = scale[i] * ECGSYN_ANGLES_DEG[i].to_radians();
            b[i] = h * ECGSYN_B[i];
        }
        Ok(Self { angles, a: ECGSYN_A, b, baseline: ECGSYN_BASELINE.0, respiration_hz: ECGSYN_BASELINE.1, theta: -PI, z: 0.0, t: 0.0 })
    }

    /// The Gaussian sum whose `θ`-derivative drives `z`: `Σ aᵢ bᵢ² exp(−Δθᵢ² / 2bᵢ²)`.
    #[must_use]
    pub fn bumps(&self, theta: f64) -> f64 {
        (0..5).map(|i| self.a[i] * self.b[i] * self.b[i] * (-0.5 * (wrapped(theta - self.angles[i]) / self.b[i]).powi(2)).exp()).sum()
    }

    /// The drive `−Σ aᵢ Δθᵢ exp(−Δθᵢ² / 2bᵢ²)`.
    #[must_use]
    pub fn drive(&self, theta: f64) -> f64 {
        -(0..5)
            .map(|i| {
                let d = wrapped(theta - self.angles[i]);
                self.a[i] * d * (-0.5 * (d / self.b[i]).powi(2)).exp()
            })
            .sum::<f64>()
    }

    /// The baseline `z₀` at time `t`.
    #[must_use]
    pub fn baseline_at(&self, t: f64) -> f64 {
        self.baseline * (TAU * self.respiration_hz * t).sin()
    }

    /// Advance by `dt` with the current beat lasting `rr` seconds, by fourth-order Runge–Kutta on
    /// `z` with the phase advancing uniformly. Returns `z` and, if the phase crossed zero during
    /// the step — an R peak — how far into the step, in seconds, it did.
    ///
    /// # Errors
    ///
    /// [`BiosignalError::OutOfRange`] for a non-positive `dt` or `rr`, or a `dt` so long that the
    /// phase would go more than half way round in one step.
    pub fn step(&mut self, dt: f64, rr: f64) -> Result<(f64, Option<f64>), BiosignalError> {
        let (dt, rr) = (positive("dt", dt)?, positive("rr", rr)?);
        let omega = TAU / rr;
        if !(omega * dt < PI) {
            return Err(BiosignalError::OutOfRange { what: "dt", value: dt });
        }
        let rate = |s: f64, z: f64| self.drive(self.theta + omega * s) - (z - self.baseline_at(self.t + s));
        let k1 = rate(0.0, self.z);
        let k2 = rate(0.5 * dt, self.z + 0.5 * dt * k1);
        let k3 = rate(0.5 * dt, self.z + 0.5 * dt * k2);
        let k4 = rate(dt, self.z + dt * k3);
        self.z += dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
        let advanced = self.theta + omega * dt;
        let r_peak = if self.theta < 0.0 && advanced >= 0.0 { Some(-self.theta / omega) } else { None };
        self.theta = wrapped(advanced);
        self.t += dt;
        Ok((self.z, r_peak))
    }
}

/// A record rescaled the way ECGSYN rescales it: linearly, so that its minimum is −0.4 mV and its
/// maximum 1.2 mV. `None` for a record that is empty, constant or not finite.
#[must_use]
pub fn to_millivolts(z: &[f64]) -> Option<Vec<f64>> {
    let (low, high) = z.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), &v| (l.min(v), h.max(v)));
    if !(high > low) || !low.is_finite() || !high.is_finite() {
        return None;
    }
    Some(z.iter().map(|v| (v - low) * 1.6 / (high - low) - 0.4).collect())
}

// ---------------------------------------------------------------------------------------------
// The spiking front end
// ---------------------------------------------------------------------------------------------

/// The events one sample produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Crossings {
    /// UP events: the signal rose through this many levels.
    pub up: u32,
    /// DOWN events.
    pub down: u32,
}

/// A level-crossing (send-on-delta) encoder for one channel: an event each time the signal has
/// moved `delta` from where the last event left the reference.
#[derive(Debug, Clone, PartialEq)]
pub struct LevelCrossing {
    /// The level spacing `δ`.
    pub delta: f64,
    /// The encoder's reconstruction of the signal; `None` until the first sample sets it.
    pub reference: Option<f64>,
    /// UP events so far.
    pub ups: u64,
    /// DOWN events so far.
    pub downs: u64,
}

impl LevelCrossing {
    /// Build.
    ///
    /// # Errors
    ///
    /// [`BiosignalError::OutOfRange`] for a `delta` that is not positive and finite.
    pub fn new(delta: f64) -> Result<Self, BiosignalError> {
        Ok(Self { delta: positive("delta", delta)?, reference: None, ups: 0, downs: 0 })
    }

    /// Feed one sample. The first sets the reference and emits nothing.
    ///
    /// # Errors
    ///
    /// [`BiosignalError::NonFinite`] for a non-finite sample.
    pub fn sample(&mut self, x: f64) -> Result<Crossings, BiosignalError> {
        let x = finite("x", x)?;
        let Some(reference) = self.reference else {
            self.reference = Some(x);
            return Ok(Crossings::default());
        };
        // Whole levels between the reference and the sample, toward the sample.
        let levels = ((x - reference) / self.delta).trunc();
        let moved = levels.abs().min(f64::from(u32::MAX)) as u32;
        self.reference = Some(reference + levels * self.delta);
        if levels > 0.0 {
            self.ups += u64::from(moved);
            Ok(Crossings { up: moved, down: 0 })
        } else {
            self.downs += u64::from(moved);
            Ok(Crossings { up: 0, down: moved })
        }
    }
}

/// A leaky integrator of UP events that fires on the R wave's upstroke and nowhere else.
#[derive(Debug, Clone, PartialEq)]
pub struct RPeakDetector {
    /// What one event adds.
    pub weight: f64,
    /// The integrator's time constant, seconds.
    pub tau: f64,
    /// The level at which it fires.
    pub threshold: f64,
    /// Seconds after a detection during which it cannot fire again.
    pub refractory: f64,
    /// The integrator.
    pub v: f64,
    /// Seconds of refractory period left.
    pub blocked: f64,
}

impl RPeakDetector {
    /// Build.
    ///
    /// # Errors
    ///
    /// [`BiosignalError::OutOfRange`] unless `0 < weight < threshold`, `tau > 0` and
    /// `refractory ≥ 0`.
    pub fn new(weight: f64, tau: f64, threshold: f64, refractory: f64) -> Result<Self, BiosignalError> {
        let (weight, tau, threshold) = (positive("weight", weight)?, positive("tau", tau)?, positive("threshold", threshold)?);
        if !(weight < threshold) {
            return Err(BiosignalError::OutOfRange { what: "weight", value: weight });
        }
        if !(refractory >= 0.0) || !refractory.is_finite() {
            return Err(BiosignalError::OutOfRange { what: "refractory", value: refractory });
        }
        Ok(Self { weight, tau, threshold, refractory, v: 0.0, blocked: 0.0 })
    }

    /// The event rate, per second, above which a regular train fires the detector:
    /// `−1 / (τ ln(1 − w/θ))`.
    #[must_use]
    pub fn rate_threshold(&self) -> f64 {
        -1.0 / (self.tau * (1.0 - self.weight / self.threshold).ln())
    }

    /// Let `dt` seconds pass, then receive `events` UP events. Returns whether it fired.
    ///
    /// # Errors
    ///
    /// [`BiosignalError::OutOfRange`] for a negative `dt`.
    pub fn step(&mut self, dt: f64, events: u32) -> Result<bool, BiosignalError> {
        if !(dt >= 0.0) || !dt.is_finite() {
            return Err(BiosignalError::OutOfRange { what: "dt", value: dt });
        }
        self.v = self.v * (-dt / self.tau).exp() + self.weight * f64::from(events);
        self.blocked = (self.blocked - dt).max(0.0);
        if self.v >= self.threshold && self.blocked == 0.0 {
            self.v = 0.0;
            self.blocked = self.refractory;
            return Ok(true);
        }
        Ok(false)
    }
}

/// What the rhythm monitor made of one beat.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Beat {
    /// The interval since the previous beat, seconds.
    pub rr: f64,
    /// The running mean it was compared with (before this beat updated it).
    pub expected: f64,
    /// Whether it left the mean by more than the tolerance.
    pub anomalous: bool,
}

/// A running mean of the RR interval and a flag for beats that leave it.
#[derive(Debug, Clone, PartialEq)]
pub struct RhythmMonitor {
    /// How fast the mean follows, in `(0, 1]`: `mean ← mean + rate · (rr − mean)`.
    pub rate: f64,
    /// The fraction of the mean a beat may differ by.
    pub tolerance: f64,
    /// The running mean, seconds; `None` until two beats have been seen.
    pub mean: Option<f64>,
    /// When the last beat was.
    pub last_beat: Option<f64>,
}

impl RhythmMonitor {
    /// Build.
    ///
    /// # Errors
    ///
    /// [`BiosignalError::OutOfRange`] for a `rate` outside `(0, 1]` or a non-positive `tolerance`.
    pub fn new(rate: f64, tolerance: f64) -> Result<Self, BiosignalError> {
        if !(rate > 0.0) || !(rate <= 1.0) {
            return Err(BiosignalError::OutOfRange { what: "rate", value: rate });
        }
        Ok(Self { rate, tolerance: positive("tolerance", tolerance)?, mean: None, last_beat: None })
    }

    /// A beat was detected at `time`. `None` for the first beat, which has no interval. An
    /// anomalous beat does NOT update the mean: one ectopic beat should not move the rhythm it is
    /// judged against.
    ///
    /// # Errors
    ///
    /// [`BiosignalError::OutOfRange`] for a beat that is not after the last one.
    pub fn beat(&mut self, time: f64) -> Result<Option<Beat>, BiosignalError> {
        let time = finite("time", time)?;
        let Some(last) = self.last_beat else {
            self.last_beat = Some(time);
            return Ok(None);
        };
        if !(time > last) {
            return Err(BiosignalError::OutOfRange { what: "time", value: time });
        }
        self.last_beat = Some(time);
        let rr = time - last;
        let Some(mean) = self.mean else {
            self.mean = Some(rr);
            return Ok(Some(Beat { rr, expected: rr, anomalous: false }));
        };
        let anomalous = (rr - mean).abs() > self.tolerance * mean;
        if !anomalous {
            self.mean = Some(mean + self.rate * (rr - mean));
        }
        Ok(Some(Beat { rr, expected: mean, anomalous }))
    }
}

// ---------------------------------------------------------------------------------------------
// The EMG
// ---------------------------------------------------------------------------------------------

/// Amplitude-modulated Gaussian noise: one sample is `activation · sigma · N(0, 1)`.
#[derive(Debug, Clone)]
pub struct Emg {
    /// The standard deviation at full activation.
    pub sigma: f64,
    rng: Rng,
}

impl Emg {
    /// Build.
    ///
    /// # Errors
    ///
    /// [`BiosignalError::OutOfRange`] for a `sigma` that is not positive and finite.
    pub fn new(sigma: f64, seed: u64) -> Result<Self, BiosignalError> {
        Ok(Self { sigma: positive("sigma", sigma)?, rng: Rng::new(seed) })
    }

    /// One sample at the given activation.
    ///
    /// # Errors
    ///
    /// [`BiosignalError::OutOfRange`] for a negative or non-finite activation.
    pub fn sample(&mut self, activation: f64) -> Result<f64, BiosignalError> {
        if !(activation >= 0.0) || !activation.is_finite() {
            return Err(BiosignalError::OutOfRange { what: "activation", value: activation });
        }
        let u = (1.0 - self.rng.next_f64()).max(f64::MIN_POSITIVE);
        let normal = (-2.0 * u.ln()).sqrt() * (TAU * self.rng.next_f64()).cos();
        Ok(activation * self.sigma * normal)
    }
}

/// The mean absolute value of Gaussian samples of standard deviation `std`: `std √(2/π)`.
#[must_use]
pub fn gaussian_mean_absolute(std: f64) -> f64 {
    std * (2.0 / PI).sqrt()
}

/// The level-crossing events per sample expected from white Gaussian samples of standard
/// deviation `std`: their mean absolute increment `2 std / √π`, over `delta`. An upper estimate —
/// the encoder ignores up to `2δ` of movement per sample — that becomes exact as `δ/std → 0`.
#[must_use]
pub fn gaussian_event_rate(std: f64, delta: f64) -> f64 {
    2.0 * std / (PI.sqrt() * delta)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_constants_are_ecgsyns_and_scale_with_heart_rate_as_it_scales_them() {
        let ecg = Ecg::ecgsyn(60.0).unwrap();
        for i in 0..5 {
            assert!((ecg.angles[i] - ECGSYN_ANGLES_DEG[i].to_radians()).abs() < 1e-15);
            assert_eq!(ecg.b[i], ECGSYN_B[i]);
        }
        assert_eq!((ecg.a, ecg.baseline, ecg.respiration_hz), ([1.2, -5.0, 30.0, -7.5, 0.75], 0.005, 0.25));
        assert_eq!((ecg.theta, ecg.z, ecg.t), (-PI, 0.0, 0.0));
        // At 240 bpm h = 2: widths double, Q and S angles double, P and T angles grow by √2, R stays.
        let fast = Ecg::ecgsyn(240.0).unwrap();
        let want = [-70.0 * 2f64.sqrt(), -30.0, 0.0, 30.0, 100.0 * 2f64.sqrt()];
        for i in 0..5 {
            assert!((fast.angles[i].to_degrees() - want[i]).abs() < 1e-12, "angle {i}");
            assert!((fast.b[i] - 2.0 * ECGSYN_B[i]).abs() < 1e-15);
        }
        assert!(Ecg::ecgsyn(0.0).is_err() && Ecg::ecgsyn(f64::NAN).is_err());
        assert_eq!(wrapped(PI), PI);
        assert!((wrapped(-PI) - PI).abs() < 1e-15 && (wrapped(3.0 * PI + 0.25) - (-PI + 0.25)).abs() < 1e-12);
    }

    #[test]
    fn the_baseline_amplitude_is_in_the_models_units_and_the_papers_order_after_the_rescale() {
        // The paper's A = 0.15 mV is AFTER ECGSYN's rescale of the record to [−0.4, 1.2] mV;
        // ECGSYN's 0.005 is before it. Put 0.005 through the millivolts-per-unit that
        // `to_millivolts` applies to a record at 60 bpm, one beat a second, 2048 steps a beat.
        let dt = 1.0 / 2048.0;
        let rescaled = |wander: bool, record: usize| {
            let mut ecg = Ecg::ecgsyn(60.0).unwrap();
            if !wander {
                ecg.baseline = 0.0;
            }
            // Sixteen unit time constants of relaxation leave e^{−16} of the start.
            for _ in 0..16 * 2048 {
                ecg.step(dt, 1.0).unwrap();
            }
            let z: Vec<f64> = (0..record).map(|_| ecg.step(dt, 1.0).unwrap().0).collect();
            let mv = to_millivolts(&z).unwrap();
            let span = |v: &[f64]| v.iter().fold(f64::NEG_INFINITY, |m, &x| m.max(x)) - v.iter().fold(f64::INFINITY, |m, &x| m.min(x));
            ECGSYN_BASELINE.0 * span(&mv) / span(&z)
        };
        // Sixteen seconds is four breaths, so the record carries the wander at its full swing.
        let settled = rescaled(true, 16 * 2048);
        let one_beat = rescaled(false, 2048);
        assert!((settled - 0.126).abs() < 5e-4, "0.005 over a settled 16 s record is {settled} mV");
        assert!((one_beat - 0.137).abs() < 5e-4, "0.005 over one beat without wander is {one_beat} mV");
        // The paper's order — within 20% of 0.15 mV — and nowhere near 0.005 read as millivolts.
        assert!(settled > 0.8 * 0.15 && one_beat < 0.15 && settled > 20.0 * ECGSYN_BASELINE.0);
    }

    #[test]
    fn the_drive_is_the_derivative_of_five_gaussian_bumps_all_the_way_round() {
        let ecg = Ecg::ecgsyn(72.0).unwrap();
        let h = 1e-6;
        let mut largest = 0.0f64;
        for k in -31..=31 {
            let theta = 0.1 * f64::from(k);
            let fd = (ecg.bumps(theta + h) - ecg.bumps(theta - h)) / (2.0 * h);
            assert!((ecg.drive(theta) - fd).abs() < 1e-8, "θ = {theta}: {} against {fd}", ecg.drive(theta));
            largest = largest.max(fd.abs());
        }
        // The R wave's drive peaks at a_R b_R e^{−½}, one width from zero.
        let b = ecg.b[2];
        assert!(largest > 1.0 && (ecg.drive(-b) - (30.0 * b * (-0.5f64).exp() + ecg.drive(-b) - ecg.a[2] * b * (-0.5f64).exp())).abs() < 1e-12);
        // Continuous where the phase wraps — which a difference left unwrapped is not.
        assert!((ecg.drive(PI - 1e-9) - ecg.drive(-PI + 1e-9)).abs() < 1e-8);
        let unwrapped = |theta: f64| -(0..5).map(|i| ecg.a[i] * (theta - ecg.angles[i]) * (-0.5 * ((theta - ecg.angles[i]) / ecg.b[i]).powi(2)).exp()).sum::<f64>();
        let cut = (unwrapped(PI - 1e-9) - unwrapped(-PI + 1e-9)).abs();
        assert!(cut > 1e-4 && cut < 1e-2 * largest, "the unwrapped form jumps by {cut} at the wrap");
    }

    #[test]
    fn the_stepped_signal_is_the_linear_filter_of_its_drive() {
        let mut ecg = Ecg::ecgsyn(60.0).unwrap();
        let (dt, rr) = (1e-3, 0.8);
        let mut z = 0.0;
        for _ in 0..1_500 {
            z = ecg.step(dt, rr).unwrap().0;
        }
        // Referee: z(T) = ∫₀ᵀ e^{−(T−s)} [drive(−π + ωs) + z₀(s)] ds by Simpson at 10 µs.
        let reference = Ecg::ecgsyn(60.0).unwrap();
        let (end, omega, panels) = (1.5f64, TAU / rr, 150_000usize);
        let f = |s: f64| (-(end - s)).exp() * (reference.drive(-PI + omega * s) + reference.baseline_at(s));
        let step = end / panels as f64;
        let mut integral = f(0.0) + f(end);
        for k in 1..panels {
            integral += f(step * k as f64) * if k % 2 == 1 { 4.0 } else { 2.0 };
        }
        integral *= step / 3.0;
        assert!((z - integral).abs() < 1e-9, "stepped {z} against the integral {integral}");
        assert!(z.abs() > 1e-3, "the comparison was made where the signal is all but zero");
        assert!((ecg.t - 1.5).abs() < 1e-12);
    }

    /// One settled beat at 1 kHz: the samples and the index of the R peak's crossing.
    fn one_beat() -> (Vec<f64>, usize) {
        let mut ecg = Ecg::ecgsyn(60.0).unwrap();
        ecg.baseline = 0.0;
        for _ in 0..3_000 {
            ecg.step(1e-3, 1.0).unwrap();
        }
        let mut z = Vec::new();
        let mut r_at = None;
        for k in 0..1_000 {
            let (v, r) = ecg.step(1e-3, 1.0).unwrap();
            z.push(v);
            if r.is_some() {
                r_at = Some(k);
            }
        }
        (z, r_at.unwrap())
    }

    #[test]
    fn the_r_peak_is_reported_at_the_instant_the_phase_crosses_zero() {
        // Starting at θ = −π, the phase reaches zero half a beat later and every beat after, so
        // the k-th R peak is at (k + ½) RR — to the resolution of the offset the step returns,
        // not to the tick. The tick here (7 ms) is deliberately no divisor of the beat.
        let mut ecg = Ecg::ecgsyn(60.0).unwrap();
        let (dt, rr) = (7e-3, 0.83);
        let (mut peaks, mut offsets) = (Vec::new(), Vec::new());
        for _ in 0..600 {
            let t_before = ecg.t;
            if let (_, Some(offset)) = ecg.step(dt, rr).unwrap() {
                peaks.push(t_before + offset);
                offsets.push(offset);
            }
        }
        assert_eq!(peaks.len(), 5);
        for (k, &peak) in peaks.iter().enumerate() {
            let want = (k as f64 + 0.5) * rr;
            assert!((peak - want).abs() < 1e-12, "peak {k} at {peak} for {want}");
        }
        // The offset is a real position inside the step — for most peaks well inside it, which is
        // the whole point of returning one rather than the tick the crossing was noticed on.
        assert!(offsets.iter().all(|&o| o > 0.0 && o <= dt), "{offsets:?}");
        assert!(offsets.iter().filter(|&&o| o > 0.05 * dt && o < 0.95 * dt).count() >= 3, "{offsets:?}");
        // A step that would take the phase more than half way round is refused rather than
        // skipping a beat in silence.
        // Half a beat is the boundary and it is EXCLUSIVE: at exactly half, the phase would land
        // on ±π and there would be no telling which way it had gone round.
        let mut ecg = Ecg::ecgsyn(60.0).unwrap();
        assert!(ecg.step(0.399, 0.8).is_ok());
        assert!(Ecg::ecgsyn(60.0).unwrap().step(0.4, 0.8).is_err(), "exactly half a beat must be refused");
        assert!(ecg.step(0.0, 0.8).is_err() && ecg.step(1e-3, 0.0).is_err() && ecg.step(f64::NAN, 0.8).is_err());
    }

    #[test]
    fn a_beat_has_its_five_waves_in_order_and_an_r_wave_of_the_predicted_height() {
        let (z, r_at) = one_beat();
        // The phase starts a beat at −π and crosses zero half way through it.
        assert!((499..=500).contains(&r_at), "R at sample {r_at}");
        // Local extrema that stand clear of the noise floor of the sampling.
        let extrema: Vec<(usize, f64)> = (1..z.len() - 1).filter(|&k| (z[k] - z[k - 1]) * (z[k + 1] - z[k]) < 0.0).map(|k| (k, z[k])).collect();
        let signs: Vec<bool> = extrema.iter().map(|&(k, _)| z[k] > z[k - 1]).collect();
        assert_eq!(signs[..5], [true, false, true, false, true], "P+ Q− R+ S− T+: {extrema:?}");
        // What follows the T wave is not a sixth wave but the filter's undershoot: `z` is the
        // bumps minus their own running average, so it dips below zero before the next P.
        assert!(extrema.len() == 6 && !signs[5] && extrema[5].1 < 0.0 && extrema[5].1 > extrema[1].1, "{extrema:?}");
        let (p, q, r, s, t) = (extrema[0], extrema[1], extrema[2], extrema[3], extrema[4]);
        // Each wave sits near its angle: θ = −π + 2π k/1000, so −70° is sample 306, ±15° are 458
        // and 542, 100° is 778. The relaxation drags each extremum a little late.
        for ((k, _), nominal) in [(p, 306.0), (q, 458.0), (r, 500.0), (s, 542.0), (t, 778.0)] {
            assert!((k as f64 - nominal).abs() < 12.0, "an extremum at sample {k}, expected near {nominal}");
        }
        // The R wave stands a_R b_R²/ω above the mean of its neighbours' levels, less the Q and S
        // bumps' own depth and what a QRS width (3 b/ω ≈ 48 ms) of unit-rate relaxation removes.
        let omega = TAU;
        let height = r.1 - 0.5 * (q.1 + s.1);
        let bump = |a: f64, b: f64| a * b * b / omega;
        let want = bump(30.0, 0.1) + 0.5 * (bump(5.0, 0.1) + bump(7.5, 0.1));
        assert!(height < want && height > want * (1.0 - 0.06), "R stands {height}, predicted just under {want}");
        // The bumps' heights a b² are 0.075, 0.3 and 0.12 for P, R and T: R is 2.5 T and T is
        // 1.6 P, before the filter's sag shifts each by a few thousandths.
        assert!(p.1 > 0.0 && t.1 > 1.2 * p.1 && r.1 > 2.0 * t.1 && r.1 < 4.0 * t.1, "P {}, T {}, R {}", p.1, t.1, r.1);
        // Rescaled as ECGSYN rescales it.
        let mv = to_millivolts(&z).unwrap();
        let (low, high) = mv.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), &v| (l.min(v), h.max(v)));
        assert!((low + 0.4).abs() < 1e-12 && (high - 1.2).abs() < 1e-12 && (mv[r.0] - 1.2).abs() < 1e-12);
        assert!(to_millivolts(&[]).is_none() && to_millivolts(&[1.0, 1.0]).is_none() && to_millivolts(&[1.0, f64::NAN]).is_none());
    }

    #[test]
    fn level_crossing_tracks_within_one_level_and_counts_the_total_variation() {
        let mut encoder = LevelCrossing::new(0.01).unwrap();
        let signal: Vec<f64> = (0..4_000).map(|k| 0.7 * (TAU * f64::from(k) / 400.0).sin() + 0.2 * (TAU * f64::from(k) / 37.0).cos()).collect();
        let (mut variation, mut events) = (0.0, 0u64);
        for (k, &x) in signal.iter().enumerate() {
            let c = encoder.sample(x).unwrap();
            assert!(c.up == 0 || c.down == 0);
            events += u64::from(c.up + c.down);
            if k > 0 {
                variation += (x - signal[k - 1]).abs();
            }
            let reference = encoder.reference.unwrap();
            assert!((x - reference).abs() < 0.01, "sample {k}: the reconstruction is {} from the signal", (x - reference).abs());
            let rebuilt = signal[0] + 0.01 * (encoder.ups as f64 - encoder.downs as f64);
            assert!((reference - rebuilt).abs() < 1e-9);
        }
        assert_eq!(events, encoder.ups + encoder.downs);
        let covered = events as f64 * 0.01;
        assert!(covered <= variation && covered > variation - 2.0 * 0.01 * 4_000.0, "{covered} of a total variation of {variation}");
        assert!(covered > 0.5 * variation, "the bound above is slack here; the encoder still saw most of it: {covered} of {variation}");
        // A jump of 5.5 levels is five events, and the half level is kept for later.
        let mut encoder = LevelCrossing::new(0.1).unwrap();
        assert_eq!(encoder.sample(1.0).unwrap(), Crossings::default());
        assert_eq!(encoder.sample(1.55).unwrap(), Crossings { up: 5, down: 0 });
        assert_eq!(encoder.sample(1.61).unwrap(), Crossings { up: 1, down: 0 });
        assert_eq!(encoder.sample(1.35).unwrap(), Crossings { up: 0, down: 2 });
        assert!(encoder.sample(f64::NAN).is_err() && LevelCrossing::new(0.0).is_err());
    }

    #[test]
    fn the_detector_fires_above_a_rate_known_in_closed_form() {
        let detector = RPeakDetector::new(1.0, 0.01, 8.0, 0.0).unwrap();
        let edge = detector.rate_threshold();
        assert!((edge - -1.0 / (0.01 * (7.0f64 / 8.0).ln())).abs() < 1e-9 && (edge - 748.9).abs() < 0.1);
        let fires = |rate: f64| {
            let mut d = detector.clone();
            (0..5_000).any(|_| d.step(1.0 / rate, 1).unwrap())
        };
        assert!(!fires(0.99 * edge) && fires(1.01 * edge));
        // Firing empties the integrator, and the refractory period holds it off. With a
        // refractory period SHORTER than the recharge (5 ticks against the 15 it takes to reach 8
        // at 1 kHz), the interval is set by the recharge — a detector that kept its charge would
        // come back over threshold the instant the period ended, and fire every 5.
        let mut d = RPeakDetector::new(1.0, 0.01, 8.0, 5e-3).unwrap();
        let (mut fired, mut left_behind) = (Vec::new(), Vec::new());
        for k in 0..100 {
            if d.step(1e-3, 1).unwrap() {
                fired.push(k);
                left_behind.push(d.v);
            }
        }
        // DERIVED, not typed: from empty, `n` events at 1 kHz leave (1 − e^{−n/10})/(1 − e^{−1/10}).
        let recharge = (1..100).find(|&n| (1.0 - (-f64::from(n) / 10.0).exp()) / (1.0 - (-0.1f64).exp()) >= 8.0).unwrap() as usize;
        assert!(recharge > 5, "{recharge} ticks to recharge is not longer than the 5-tick refractory period");
        let gaps: Vec<usize> = fired.windows(2).map(|p| p[1] - p[0]).collect();
        assert_eq!(fired[0], recharge - 1, "from rest it takes {recharge} ticks to reach the threshold");
        assert!(gaps.iter().all(|&g| g == recharge), "the intervals were {gaps:?}, not the {recharge}-tick recharge");
        assert!(left_behind.iter().all(|&v| v == 0.0), "the integrator was not empty after a detection: {left_behind:?}");
        // With a refractory period LONGER than the recharge, that period sets the interval.
        let mut d = RPeakDetector::new(1.0, 0.01, 8.0, 0.2).unwrap();
        let mut fired = Vec::new();
        for k in 0..600 {
            if d.step(1e-3, 1).unwrap() {
                fired.push(k);
            }
        }
        assert!(fired.len() == 3 && fired[1] - fired[0] == 200 && fired[2] - fired[1] == 200, "{fired:?}");
        assert!(RPeakDetector::new(8.0, 0.01, 8.0, 0.2).is_err() && RPeakDetector::new(1.0, 0.0, 8.0, 0.2).is_err());
        assert!(RPeakDetector::new(1.0, 0.01, 8.0, -0.1).is_err() && d.step(-1.0, 0).is_err());
    }

    /// Run an ECG with the given beat lengths through the front end; returns the true R times
    /// and the detections.
    fn monitor(beats: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let mut ecg = Ecg::ecgsyn(60.0).unwrap();
        let mut encoder = LevelCrossing::new(0.002).unwrap();
        // In this model every wave's height is a b²/ω — proportional to the beat's length — so a
        // premature beat is also a smaller one (its R upstroke is 12 levels, not 24), and the
        // threshold has to be set for it: 5, which the P and T waves (under 1) never approach.
        let mut detector = RPeakDetector::new(1.0, 0.01, 5.0, 0.2).unwrap();
        let (mut truth, mut found) = (Vec::new(), Vec::new());
        let dt = 1e-3;
        let mut beat = 0usize;
        while beat < beats.len() {
            let before = ecg.theta;
            let (z, r) = ecg.step(dt, beats[beat]).unwrap();
            if let Some(offset) = r {
                truth.push(ecg.t - dt + offset);
            }
            if ecg.theta < before {
                beat += 1;
            }
            if detector.step(dt, encoder.sample(z).unwrap().up).unwrap() {
                found.push(ecg.t);
            }
        }
        (truth, found)
    }

    #[test]
    fn the_front_end_finds_every_r_peak_and_nothing_else_and_flags_the_ectopic_beat() {
        // A steady 0.9 s rhythm with one premature beat (0.5 s) and its compensatory pause (1.3 s).
        let mut beats = vec![0.9; 24];
        beats[12] = 0.5;
        beats[13] = 1.3;
        let (truth, found) = monitor(&beats);
        assert_eq!(truth.len(), 24);
        assert_eq!(found.len(), truth.len(), "detections {found:?} for R peaks {truth:?}");
        // Each detection is on the R wave's upstroke, which begins about three widths before the
        // peak: 3 b_R/ω = 0.048 of the beat. The lead is that fraction of the beat or less, so
        // beats of equal length are timed alike and the INTERVALS come out right.
        for (k, (f, t)) in found.iter().zip(&truth).enumerate() {
            assert!(t - f > 0.0 && t - f < 0.048 * beats[k], "beat {k}: detected {f} for a peak at {t}");
        }
        let mut rhythm = RhythmMonitor::new(0.2, 0.2).unwrap();
        let mut flagged = Vec::new();
        for (k, &time) in found.iter().enumerate() {
            if let Some(beat) = rhythm.beat(time).unwrap() {
                let true_rr = truth[k] - truth[k - 1];
                // Between beats of one length the leads cancel and the interval is right to a
                // sample either way at each end. Between beats of different lengths they do not
                // — MEASURED: the small premature beat is caught only at the top of its
                // upstroke, 12 ms later in its wave than its neighbours — and the bound is the
                // lead's own: 0.048 of the longer beat.
                let slack = if beats[k] == beats[k - 1] { 2e-3 } else { 0.048 * beats[k].max(beats[k - 1]) };
                assert!((beat.rr - true_rr).abs() < slack, "beat {k}: RR {} against {true_rr}", beat.rr);
                if beat.anomalous {
                    flagged.push(k);
                }
            }
        }
        // R peaks sit mid-beat, so the short beat shortens intervals 12 and 13's neighbours:
        // R₁₂ − R₁₁ = 0.45 + 0.25 = 0.7 s, R₁₃ − R₁₂ = 0.25 + 0.65 = 0.9 s, R₁₄ − R₁₃ = 0.65 + 0.45 = 1.1 s.
        assert_eq!(flagged, vec![12, 14], "0.7 s and 1.1 s against a 0.9 s rhythm at 20% tolerance");
    }

    #[test]
    fn the_rhythm_monitors_mean_is_the_geometric_approach_and_ignores_what_it_flags() {
        let mut rhythm = RhythmMonitor::new(0.25, 0.3).unwrap();
        assert_eq!(rhythm.beat(0.0).unwrap(), None);
        assert_eq!(rhythm.beat(1.0).unwrap(), Some(Beat { rr: 1.0, expected: 1.0, anomalous: false }));
        let mut t = 1.0;
        for k in 1..=10 {
            t += 0.8;
            let beat = rhythm.beat(t).unwrap().unwrap();
            let want = 0.8 + 0.2 * 0.75f64.powi(k - 1);
            assert!(!beat.anomalous && (beat.expected - want).abs() < 1e-12, "beat {k}: mean {} against {want}", beat.expected);
        }
        let settled = rhythm.mean.unwrap();
        let beat = rhythm.beat(t + 0.4).unwrap().unwrap();
        assert!(beat.anomalous && rhythm.mean == Some(settled), "a flagged beat moved the mean");
        assert!(rhythm.beat(t).is_err() && rhythm.beat(f64::NAN).is_err());
        assert!(RhythmMonitor::new(0.0, 0.2).is_err() && RhythmMonitor::new(1.5, 0.2).is_err() && RhythmMonitor::new(0.5, 0.0).is_err());
    }

    #[test]
    fn the_emg_envelope_is_the_event_rate() {
        // σ = 0.4, so that a model which dropped it would be reporting four times the amplitude.
        let (n, sigma) = (200_000usize, 0.4);
        let measure = |activation: f64| {
            let mut emg = Emg::new(sigma, 42).unwrap();
            let mut encoder = LevelCrossing::new(0.05 * sigma).unwrap();
            let (mut absolute, mut variation, mut last) = (0.0, 0.0, None::<f64>);
            for _ in 0..n {
                let x = emg.sample(activation).unwrap();
                absolute += x.abs();
                if let Some(l) = last {
                    variation += (x - l).abs();
                }
                last = Some(x);
                encoder.sample(x).unwrap();
            }
            (absolute / n as f64, variation / n as f64, (encoder.ups + encoder.downs) as f64 / n as f64)
        };
        let std = 0.5 * sigma;
        let (mav, variation, events) = measure(0.5);
        // The sample mean of |x| has standard error 0.6 std/√n = 0.13% of the mean; 1% is seven of them.
        assert!((mav / gaussian_mean_absolute(std) - 1.0).abs() < 0.01, "mean absolute value {mav} for a standard deviation of {std}");
        assert!((gaussian_mean_absolute(1.0) - 0.797_884_560_803).abs() < 1e-12 && gaussian_mean_absolute(std) < 0.16);
        assert!((variation / (2.0 * std / PI.sqrt()) - 1.0).abs() < 0.01, "mean absolute increment {variation}");
        // Events × δ is the total variation, less at most 2δ a sample.
        let delta = 0.05 * sigma;
        assert!(events * delta <= variation && events * delta > variation - 2.0 * delta, "{events} events a sample");
        assert!(events <= gaussian_event_rate(std, delta) * 1.01 && events > 0.85 * gaussian_event_rate(std, delta));
        // Linear in the activation: MEASURED, doubling it multiplies the event rate by 2.0.
        let (_, _, doubled) = measure(1.0);
        assert!((doubled / events - 2.0).abs() < 0.1, "doubling the activation multiplied the events by {}", doubled / events);
        let (_, _, silent) = measure(0.0);
        assert_eq!(silent, 0.0);
        let mut emg = Emg::new(1.0, 1).unwrap();
        assert!(emg.sample(-0.1).is_err() && emg.sample(f64::NAN).is_err() && Emg::new(0.0, 1).is_err());
    }
}
