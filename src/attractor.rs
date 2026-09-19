//! The ring attractor: a bump of activity that holds a direction after the cue has gone, with a
//! width and a height you can write down — checked against both.
//!
//! # What the mechanism is
//!
//! `N` rate neurons sit on a ring, neuron `i` preferring the angle `θ_i = 2πi/N`. Each excites
//! its neighbours and inhibits the far side through the kernel `J(θ − θ′) = J₀ + J₁ cos(θ − θ′)`,
//! and its rate follows `τ dm/dt = −m + [I + (1/N) Σ_j J(θ_i − θ_j) m_j]₊` (Ben-Yishai, Bar-Or and
//! Sompolinsky, *Theory of orientation tuning in visual cortex*, PNAS 92(9):3844–3848, 1995;
//! Hansel and Sompolinsky, *Modeling feature selectivity in local cortical circuits*, in Koch and
//! Segev (eds), *Methods in Neuronal Modeling*, 2nd ed., MIT Press, 1998, ch. 13). When the tuned
//! part of the kernel is strong enough, `J₁ > 2`, a UNIFORM input is answered by a NON-uniform
//! state: a bump `m(θ) = A [cos(θ − ψ) − cos θ_c]₊`. Where the bump sits, `ψ`, is not decided by
//! the input at all — every position is an equally good fixed point — so a brief cue places it
//! and it stays: a memory of a direction, held by the dynamics. This is the standard model of the
//! head-direction system (Zhang, *Representation of spatial orientation by the intrinsic dynamics
//! of the head-direction cell ensemble: a theory*, Journal of Neuroscience 16(6):2112–2126, 1996)
//! and of the fly's compass (Kim, Rouault, Druckmann and Jayaraman, *Ring attractor dynamics in
//! the Drosophila central brain*, Science 356(6340):849–853, 2017).
//!
//! # Why it is in a neuromorphic crate
//!
//! It is the smallest circuit that keeps a continuous variable without a register: a robot's
//! heading held as a pattern of activity, on a substrate whose only state is activity. It is
//! also a circuit whose cost has a shortcut worth knowing — the cosine kernel is rank three, so
//! the `N²` recurrent sum is three population averages ([`Ring::recurrent`] against
//! [`Ring::recurrent_pairwise`]), which is what a chip would wire.
//!
//! # The closed forms this module is checked against
//!
//! With `r₀` the mean rate and `r₁` the magnitude of the first Fourier component, substituting the
//! bump into the fixed-point equation gives, in the continuum:
//!
//! - **Width from `J₁` alone**: `J₁ f₁(θ_c) = 1`, `f₁(θ) = (θ − ½ sin 2θ)/(2π)`
//!   ([`bump_half_width`]). `f₁` rises from `0` to `½`, so a bump exists iff `J₁ > 2`.
//! - **Height from the rest**: `A = −I₀ / (cos θ_c + J₀ f₀(θ_c))`, `f₀(θ) = (sin θ − θ cos θ)/π`
//!   ([`bump_amplitude`]); the peak rate is `A (1 − cos θ_c)`. It is positive and finite only when
//!   `J₀ < −cos θ_c / f₀(θ_c)` — otherwise the bump's height runs away, and the function says so.
//! - **Below `J₁ = 2`** the uniform state `m = I₀/(1 − J₀)` is what the ring settles on
//!   ([`uniform_rate`]).
//! - **The low-rank identity**: the three-average recurrent input equals the pairwise sum to
//!   rounding.
//! - **A tuned input** `I₀ + I₁ cos(θ − θ₀)` pins the bump at `θ₀` ([`tuned_bump`]). Its half-width
//!   is the root of `I₀/I₁ = −(cos θ_c + J₀ f₀(θ_c))/(1 − J₁ f₁(θ_c))` and its amplitude is
//!   `A = I₁/(1 − J₁ f₁(θ_c))` — the input's tuning multiplied by a gain the recurrence sets, which
//!   is the amplification of weakly tuned input the model was proposed for. When the root would
//!   pass `π` nothing is rectified and the state is the linear `I₀/(1 − J₀) + I₁/(1 − J₁/2) cos`.
//! - **The memory**: after the cue is removed the bump's population-vector angle stays where the
//!   cue put it, and a bump cued anywhere has the same width and height — the attractor is a ring.
//!
//! A ring of `N` neurons is a Riemann sum of the continuum; the simulated width and height are
//! held to the closed forms within the square of the grid spacing, `(2π/N)²`.
//!
//! # What this module has NOT reproduced
//!
//! - **Path integration.** Turning the bump at a commanded angular velocity needs an asymmetric
//!   kernel, and the threshold-linear ring has no closed form for the speed it then moves at: the
//!   moving bump drags an exponential tail, its support is not the region of positive input, and
//!   the first-draft guess `ω = tan(φ)/τ` for a kernel shifted by `φ` holds only for a bump that
//!   keeps its resting shape, which it does not. It is left out rather than shipped with a
//!   tolerance chosen to pass.
//! - Spiking neurons and the noise-driven diffusion of the bump.
//! - The tuned-input solution for `J₁ ≥ 2`, where the bump exists without the tuning and the
//!   width equation can have more than one root; [`tuned_bump`] covers `J₁ < 2` and says so.

use core::f64::consts::{PI, TAU};
use core::fmt;

/// The most steps one `run` call will take; a request past it is refused.
pub const MAX_STEPS: u64 = 100_000_000;

/// What went wrong, named rather than guessed around.
#[derive(Debug, Clone, PartialEq)]
pub enum AttractorError {
    /// Too few neurons to carry a first Fourier component.
    TooFew {
        /// Neurons asked for.
        n: usize,
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

impl fmt::Display for AttractorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooFew { n } => write!(f, "a ring of {n} neurons cannot carry a bump (needs three)"),
            Self::Dimension { what, got, want } => write!(f, "{what} has length {got}, expected {want}"),
            Self::OutOfRange { what, value, low, high } => {
                write!(f, "{what} = {value} is outside [{low}, {high}]")
            }
            Self::NonFinite { what, index } => write!(f, "{what} is not finite at {index}"),
        }
    }
}

impl std::error::Error for AttractorError {}

fn positive(what: &'static str, v: f64) -> Result<f64, AttractorError> {
    if v.is_finite() && v > 0.0 {
        Ok(v)
    } else {
        Err(AttractorError::OutOfRange { what, value: v, low: f64::MIN_POSITIVE, high: f64::INFINITY })
    }
}

fn finite_all(what: &'static str, v: &[f64]) -> Result<(), AttractorError> {
    if let Some(i) = v.iter().position(|x| !x.is_finite()) {
        return Err(AttractorError::NonFinite { what, index: i });
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// The closed forms
// ---------------------------------------------------------------------------------------------

/// `f₁(θ) = (θ − ½ sin 2θ)/(2π)`: the first Fourier component of a unit rectified-cosine bump of
/// half-width `θ`, per unit amplitude.
#[must_use]
pub fn f1(theta_c: f64) -> f64 {
    (theta_c - 0.5 * (2.0 * theta_c).sin()) / TAU
}

/// `f₀(θ) = (sin θ − θ cos θ)/π`: the mean of the same bump, per unit amplitude.
#[must_use]
pub fn f0(theta_c: f64) -> f64 {
    (theta_c.sin() - theta_c * theta_c.cos()) / PI
}

/// The bump's half-width `θ_c`, radians: the root of `J₁ f₁(θ_c) = 1` on `(0, π)`, by bisection.
/// `None` unless `J₁ > 2` (and finite) — below that no bump exists.
#[must_use]
pub fn bump_half_width(j1: f64) -> Option<f64> {
    if !(j1 > 2.0) || !j1.is_finite() {
        return None;
    }
    let (mut lo, mut hi) = (0.0f64, PI);
    // f₁ is increasing (its derivative is (1 − cos 2θ)/2π ≥ 0), so bisection is safe; 60
    // halvings of π reach the last bit.
    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        if j1 * f1(mid) < 1.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Some(0.5 * (lo + hi))
}

/// The bump's amplitude `A = −I₀/(cos θ_c + J₀ f₀(θ_c))` under a uniform input `i0`; the rate
/// profile is `A [cos(θ − ψ) − cos θ_c]₊`. `None` when no bump exists (`J₁ ≤ 2`), when the input
/// is not positive, or when the amplitude is not positive and finite — `J₀ ≥ −cos θ_c/f₀(θ_c)`,
/// where the uniform inhibition is too weak to stop the bump growing without bound.
#[must_use]
pub fn bump_amplitude(i0: f64, j0: f64, j1: f64) -> Option<f64> {
    let theta_c = bump_half_width(j1)?;
    if !(i0 > 0.0) || !i0.is_finite() || !j0.is_finite() {
        return None;
    }
    let a = -i0 / (theta_c.cos() + j0 * f0(theta_c));
    if a > 0.0 && a.is_finite() { Some(a) } else { None }
}

/// The uniform fixed point `m = I₀/(1 − J₀)` under a uniform input, which the ring settles on
/// when `J₁ < 2`. `None` for a non-positive input or `J₀ ≥ 1` (where it runs away).
#[must_use]
pub fn uniform_rate(i0: f64, j0: f64) -> Option<f64> {
    if !(i0 > 0.0) || !i0.is_finite() || !(j0 < 1.0) || !j0.is_finite() {
        return None;
    }
    Some(i0 / (1.0 - j0))
}

/// What a tuned input `I₀ + I₁ cos(θ − θ₀)` settles the ring into, for `J₁ < 2`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Tuned {
    /// Part of the ring is silent: the profile is `amplitude · [cos(θ − θ₀) − cos(half_width)]₊`.
    Rectified {
        /// Half-width `θ_c` of the active region, radians.
        half_width: f64,
        /// Amplitude `A = I₁/(1 − J₁ f₁(θ_c))`.
        amplitude: f64,
    },
    /// Every neuron is active and nothing is rectified: `mean + amplitude · cos(θ − θ₀)`.
    Linear {
        /// `I₀/(1 − J₀)`.
        mean: f64,
        /// `I₁/(1 − J₁/2)`.
        amplitude: f64,
    },
}

/// The state a tuned input `i0 + i1 cos(θ − θ₀)` settles the ring into. `None` unless `i1 > 0`,
/// `j0 < 1`, `j1 < 2` and the arguments are finite — and `None` when the input never reaches
/// threshold anywhere (`i0 + i1 ≤ 0`), where the ring stays silent.
#[must_use]
pub fn tuned_bump(i0: f64, i1: f64, j0: f64, j1: f64) -> Option<Tuned> {
    let finite = i0.is_finite() && i1.is_finite() && j0.is_finite() && j1.is_finite();
    if !finite || !(i1 > 0.0) || !(j0 < 1.0) || !(j1 < 2.0) || i0 + i1 <= 0.0 {
        return None;
    }
    let (mean, linear_amplitude) = (i0 / (1.0 - j0), i1 / (1.0 - 0.5 * j1));
    if mean >= linear_amplitude {
        return Some(Tuned::Linear { mean, amplitude: linear_amplitude });
    }
    // g(θ_c) = −(cos θ_c + J₀ f₀)/(1 − J₁ f₁) − I₀/I₁ is −1 − I₀/I₁ < 0 at θ_c = 0 and positive at
    // π exactly when the linear state fails, which is the case here. Bisect the first crossing.
    let g = |t: f64| -(t.cos() + j0 * f0(t)) / (1.0 - j1 * f1(t)) - i0 / i1;
    let steps = 4000;
    let mut lo = 0.0;
    let mut hi = PI;
    for k in 1..=steps {
        let t = PI * f64::from(k) / f64::from(steps);
        if g(t) >= 0.0 {
            hi = t;
            break;
        }
        lo = t;
    }
    for _ in 0..80 {
        let mid = 0.5 * (lo + hi);
        if g(mid) < 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let half_width = 0.5 * (lo + hi);
    Some(Tuned::Rectified { half_width, amplitude: i1 / (1.0 - j1 * f1(half_width)) })
}

// ---------------------------------------------------------------------------------------------
// The ring
// ---------------------------------------------------------------------------------------------

/// A ring of threshold-linear rate neurons with the kernel `J₀ + J₁ cos(θ − θ′)`.
#[derive(Debug, Clone, PartialEq)]
pub struct Ring {
    /// Uniform coupling `J₀` (negative: global inhibition).
    pub j0: f64,
    /// Tuned coupling `J₁`; a bump forms above `2`.
    pub j1: f64,
    /// Rate time constant `τ`, seconds.
    pub tau: f64,
    /// Rates, one per neuron; neuron `i` prefers `2πi/N`.
    pub m: Vec<f64>,
}

impl Ring {
    /// A silent ring of `n` neurons.
    ///
    /// # Errors
    ///
    /// [`AttractorError::TooFew`] for fewer than three neurons, [`AttractorError::NonFinite`] for
    /// a non-finite coupling, [`AttractorError::OutOfRange`] for a non-positive `tau`.
    pub fn new(n: usize, j0: f64, j1: f64, tau: f64) -> Result<Self, AttractorError> {
        if n < 3 {
            return Err(AttractorError::TooFew { n });
        }
        if !j0.is_finite() {
            return Err(AttractorError::NonFinite { what: "j0", index: 0 });
        }
        if !j1.is_finite() {
            return Err(AttractorError::NonFinite { what: "j1", index: 0 });
        }
        let tau = positive("tau", tau)?;
        Ok(Self { j0, j1, tau, m: vec![0.0; n] })
    }

    /// The preferred angle of neuron `i`, radians.
    #[must_use]
    pub fn angle(&self, i: usize) -> f64 {
        TAU * i as f64 / self.m.len() as f64
    }

    /// The three population averages the kernel needs: `(r₀, c₁, s₁)` — the mean rate and the
    /// cosine and sine components of the first Fourier mode.
    #[must_use]
    pub fn moments(&self) -> (f64, f64, f64) {
        let n = self.m.len() as f64;
        let (mut r0, mut c1, mut s1) = (0.0, 0.0, 0.0);
        for (i, &mi) in self.m.iter().enumerate() {
            let (s, c) = self.angle(i).sin_cos();
            r0 += mi;
            c1 += mi * c;
            s1 += mi * s;
        }
        (r0 / n, c1 / n, s1 / n)
    }

    /// The recurrent input to every neuron from the three averages — `N` operations.
    #[must_use]
    pub fn recurrent(&self) -> Vec<f64> {
        let (r0, c1, s1) = self.moments();
        (0..self.m.len())
            .map(|i| {
                let (s, c) = self.angle(i).sin_cos();
                self.j0 * r0 + self.j1 * (c1 * c + s1 * s)
            })
            .collect()
    }

    /// The same input by the literal pairwise sum — `N²` operations. The reference
    /// [`Ring::recurrent`] is checked against; nothing else calls it.
    #[must_use]
    pub fn recurrent_pairwise(&self) -> Vec<f64> {
        let n = self.m.len();
        (0..n)
            .map(|i| {
                let sum: f64 = (0..n).map(|j| (self.j0 + self.j1 * (self.angle(i) - self.angle(j)).cos()) * self.m[j]).sum();
                sum / n as f64
            })
            .collect()
    }

    /// The bump's position and strength: the angle of the population vector in `(−π, π]` and the
    /// magnitude `r₁` of the first Fourier component. The angle is `None` for a state with no
    /// first component — a silent or uniform ring points nowhere.
    #[must_use]
    pub fn population_vector(&self) -> (Option<f64>, f64) {
        let (_, c1, s1) = self.moments();
        let r1 = c1.hypot(s1);
        let scale = self.m.iter().fold(0.0f64, |a, m| a.max(m.abs()));
        // A uniform state's first component is rounding, of the order of N ulps of the rates.
        if r1 <= self.m.len() as f64 * f64::EPSILON * scale {
            (None, r1)
        } else {
            (Some(s1.atan2(c1)), r1)
        }
    }

    /// The half-width of the active region, radians: half the fraction of neurons with a positive
    /// rate, times `2π`. Good to half a grid spacing.
    #[must_use]
    pub fn active_half_width(&self) -> f64 {
        let active = self.m.iter().filter(|m| **m > 0.0).count();
        PI * active as f64 / self.m.len() as f64
    }

    /// One Euler step of `dt` seconds under the input `input` (one value per neuron). Rates are
    /// non-negative, and a rate that has decayed below `f64::MIN_POSITIVE` is set to zero.
    ///
    /// # Errors
    ///
    /// [`AttractorError::Dimension`] for an input of the wrong length,
    /// [`AttractorError::NonFinite`] for a bad entry, [`AttractorError::OutOfRange`] for a `dt`
    /// that is not in `(0, τ]` — past `τ` the Euler step overshoots its own target.
    pub fn step(&mut self, dt: f64, input: &[f64]) -> Result<(), AttractorError> {
        if input.len() != self.m.len() {
            return Err(AttractorError::Dimension { what: "input", got: input.len(), want: self.m.len() });
        }
        finite_all("input", input)?;
        if !(dt > 0.0) || !(dt <= self.tau) {
            return Err(AttractorError::OutOfRange { what: "dt", value: dt, low: f64::MIN_POSITIVE, high: self.tau });
        }
        let rec = self.recurrent();
        let a = dt / self.tau;
        for i in 0..self.m.len() {
            let drive = (input[i] + rec[i]).max(0.0);
            let next = self.m[i] + a * (drive - self.m[i]);
            // A rate below the smallest normal number IS zero. Without this the silent side of the
            // ring is never silent: Euler decay stalls on a denormal (4 × 5e-324 × 0.9 rounds back
            // to 4 × 5e-324), which the first run of this module's bump test found by reporting
            // every neuron active.
            self.m[i] = if next < f64::MIN_POSITIVE { 0.0 } else { next };
        }
        Ok(())
    }

    /// Take `steps` steps of `dt` under a constant input.
    ///
    /// # Errors
    ///
    /// As [`Ring::step`], plus [`AttractorError::OutOfRange`] for more than [`MAX_STEPS`].
    pub fn run(&mut self, dt: f64, input: &[f64], steps: u64) -> Result<(), AttractorError> {
        if steps > MAX_STEPS {
            return Err(AttractorError::OutOfRange { what: "steps", value: steps as f64, low: 0.0, high: MAX_STEPS as f64 });
        }
        for _ in 0..steps {
            self.step(dt, input)?;
        }
        Ok(())
    }

    /// A uniform input `i0` with a cosine cue of relative depth `depth` centred on `at`:
    /// `i0 (1 + depth cos(θ − at))`.
    #[must_use]
    pub fn cue(&self, i0: f64, depth: f64, at: f64) -> Vec<f64> {
        (0..self.m.len()).map(|i| i0 * (1.0 + depth * (self.angle(i) - at).cos())).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wrap_pi(x: f64) -> f64 {
        let r = x.rem_euclid(TAU);
        if r > PI { r - TAU } else { r }
    }

    #[test]
    fn the_width_equation_has_the_root_the_function_returns() {
        for j1 in [2.5, 4.0, 6.0, 12.0] {
            let theta_c = bump_half_width(j1).unwrap();
            assert!((j1 * f1(theta_c) - 1.0).abs() < 1e-14, "J1 = {j1}: residual {}", j1 * f1(theta_c) - 1.0);
            assert!(theta_c > 0.0 && theta_c < PI);
        }
        // A stronger tuned coupling makes a NARROWER bump.
        assert!(bump_half_width(12.0).unwrap() < bump_half_width(4.0).unwrap());
        // The two integrals, against the quadrature they abbreviate: the mean and the first
        // Fourier component of [cos θ − cos θ_c]₊, by the midpoint rule on 200 000 points.
        let theta_c = 1.1f64;
        let n = 200_000;
        let (mut mean, mut first) = (0.0, 0.0);
        for k in 0..n {
            let t = -PI + TAU * (f64::from(k) + 0.5) / f64::from(n);
            let v = (t.cos() - theta_c.cos()).max(0.0);
            mean += v;
            first += v * t.cos();
        }
        assert!((mean / f64::from(n) - f0(theta_c)).abs() < 1e-9, "{} vs {}", mean / f64::from(n), f0(theta_c));
        assert!((first / f64::from(n) - f1(theta_c)).abs() < 1e-9, "{} vs {}", first / f64::from(n), f1(theta_c));
        // The ends: f₁(π) = ½ is why the threshold is J₁ = 2.
        assert!((f1(PI) - 0.5).abs() < 1e-15 && f1(0.0) == 0.0 && f0(0.0) == 0.0);
        assert!((f0(PI) - 1.0).abs() < 1e-15, "a full cosine plus one has mean one");
        for none in [bump_half_width(2.0), bump_half_width(1.0), bump_half_width(f64::NAN), bump_half_width(f64::INFINITY)] {
            assert_eq!(none, None);
        }
    }

    #[test]
    fn the_low_rank_input_is_the_pairwise_sum() {
        let mut ring = Ring::new(64, -3.0, 5.0, 10e-3).unwrap();
        for i in 0..64 {
            ring.m[i] = ((i * 37 % 64) as f64 / 64.0 - 0.3).max(0.0);
        }
        let (fast, slow) = (ring.recurrent(), ring.recurrent_pairwise());
        let worst = fast.iter().zip(&slow).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        // 64 terms of size ≤ |J₀| + |J₁| = 8, each good to an ulp.
        assert!(worst < 64.0 * 8.0 * f64::EPSILON, "low-rank and pairwise differ by {worst}");
        assert!(slow.iter().any(|v| v.abs() > 0.1), "the comparison is between two zeros");
    }

    #[test]
    fn the_bump_has_the_width_and_height_the_theory_gives() {
        let (n, j0, j1, i0) = (512usize, -4.0, 6.0, 1.0);
        let mut ring = Ring::new(n, j0, j1, 10e-3).unwrap();
        let at = ring.angle(100);
        let cue = ring.cue(i0, 0.1, at);
        ring.run(1e-3, &cue, 2_000).unwrap();
        let flat = vec![i0; n];
        ring.run(1e-3, &flat, 40_000).unwrap();
        let theta_c = bump_half_width(j1).unwrap();
        let a = bump_amplitude(i0, j0, j1).unwrap();
        assert!((a - -1.0 / (theta_c.cos() - 4.0 * f0(theta_c))).abs() < 1e-15);
        let grid = TAU / n as f64;
        // The active half-width counts neurons, so it is good to half a grid spacing.
        assert!((ring.active_half_width() - theta_c).abs() <= 0.5 * grid + 1e-12, "half-width {} vs {theta_c}", ring.active_half_width());
        // The peak and the two moments are Riemann sums of a function with a kink: second order
        // in the grid spacing, relative.
        let tol = grid * grid;
        let peak = ring.m.iter().fold(0.0f64, |p, m| p.max(*m));
        let want_peak = a * (1.0 - theta_c.cos());
        assert!((peak / want_peak - 1.0).abs() < tol, "peak {peak}, theory {want_peak}");
        let (r0, _, _) = ring.moments();
        assert!((r0 / (a * f0(theta_c)) - 1.0).abs() < tol, "mean rate {r0}, theory {}", a * f0(theta_c));
        let (angle, r1) = ring.population_vector();
        assert!((r1 / (a * f1(theta_c)) - 1.0).abs() < tol, "r1 {r1}, theory {}", a * f1(theta_c));
        // And the whole profile, neuron by neuron.
        let psi = angle.unwrap();
        for i in 0..n {
            let want = a * ((ring.angle(i) - psi).cos() - theta_c.cos()).max(0.0);
            // The amplitude and the offset −A cos θ_c are each second order in the grid.
            assert!((ring.m[i] - want).abs() < 2.0 * tol * want_peak, "neuron {i}: {} vs {want}", ring.m[i]);
        }
        // The cue was removed 400 time constants ago and the bump is still where it was put.
        assert!(wrap_pi(psi - at).abs() < 1e-9, "the bump drifted {} rad from its cue", wrap_pi(psi - at));
    }

    #[test]
    fn every_position_on_the_ring_holds_the_same_bump() {
        let (n, i0) = (256usize, 1.0);
        let settle = |at_index: usize| {
            let mut ring = Ring::new(n, -4.0, 6.0, 10e-3).unwrap();
            let cue = ring.cue(i0, 0.1, ring.angle(at_index));
            ring.run(1e-3, &cue, 2_000).unwrap();
            ring.run(1e-3, &vec![i0; n], 20_000).unwrap();
            ring
        };
        let (a, b) = (settle(10), settle(137));
        let (pa, pb) = (a.population_vector(), b.population_vector());
        assert!(wrap_pi(pa.0.unwrap() - a.angle(10)).abs() < 1e-9);
        assert!(wrap_pi(pb.0.unwrap() - b.angle(137)).abs() < 1e-9);
        // The same bump, rotated by 127 neurons.
        for i in 0..n {
            assert!((a.m[i] - b.m[(i + 127) % n]).abs() < 1e-9, "neuron {i}");
        }
        assert!((pa.1 - pb.1).abs() < 1e-12);
    }

    #[test]
    fn below_the_threshold_coupling_the_ring_goes_uniform() {
        let (n, j0, i0) = (128usize, -1.5, 2.0);
        let mut ring = Ring::new(n, j0, 1.5, 10e-3).unwrap();
        let cue = ring.cue(i0, 0.3, 1.0);
        ring.run(1e-3, &cue, 2_000).unwrap();
        assert!(ring.population_vector().0.is_some(), "under the cue the state is tuned");
        ring.run(1e-3, &vec![i0; n], 40_000).unwrap();
        let want = uniform_rate(i0, j0).unwrap();
        assert_eq!(want, 0.8);
        for (i, &m) in ring.m.iter().enumerate() {
            assert!((m - want).abs() < 1e-12, "neuron {i} at {m}, uniform state {want}");
        }
        // The first Fourier mode relaxes at (1 − J₁/2)/τ = 25 1/s: 40 s leaves nothing of it.
        assert!(ring.population_vector().1 < 1e-12);
        assert_eq!(bump_amplitude(i0, j0, 1.5), None, "no bump below J₁ = 2");
        for none in [uniform_rate(0.0, -1.0), uniform_rate(1.0, 1.0), uniform_rate(1.0, f64::NAN), uniform_rate(f64::INFINITY, 0.0)] {
            assert_eq!(none, None);
        }
    }

    #[test]
    fn a_tuned_input_is_amplified_by_the_gain_the_recurrence_sets() {
        let (n, j0, j1, i0, i1) = (512usize, -2.0, 1.5, -0.2, 1.0);
        let Some(Tuned::Rectified { half_width, amplitude }) = tuned_bump(i0, i1, j0, j1) else { panic!("this input is rectified") };
        // The two fixed-point conditions, substituted back.
        assert!((amplitude * (1.0 - j1 * f1(half_width)) - i1).abs() < 1e-13);
        assert!((i0 + amplitude * (half_width.cos() + j0 * f0(half_width))).abs() < 1e-12);
        assert!(amplitude > i1, "the recurrence amplifies the tuning: A = {amplitude} from I₁ = {i1}");
        let mut ring = Ring::new(n, j0, j1, 10e-3).unwrap();
        let at = ring.angle(200);
        let input: Vec<f64> = (0..n).map(|i| i0 + i1 * (ring.angle(i) - at).cos()).collect();
        ring.run(1e-3, &input, 20_000).unwrap();
        let grid = TAU / n as f64;
        assert!((ring.active_half_width() - half_width).abs() <= 0.5 * grid + 1e-12, "{} vs {half_width}", ring.active_half_width());
        let peak = ring.m.iter().fold(0.0f64, |p, m| p.max(*m));
        let want = amplitude * (1.0 - half_width.cos());
        assert!((peak / want - 1.0).abs() < grid * grid, "peak {peak}, theory {want}");
        assert!(wrap_pi(ring.population_vector().0.unwrap() - at).abs() < 1e-9, "the bump sits on the input's peak");
        // A strong uniform part leaves nothing rectified, and the state is linear and exact.
        let Some(Tuned::Linear { mean, amplitude }) = tuned_bump(3.0, 0.2, -1.0, 1.0) else { panic!("this input is not rectified") };
        assert_eq!((mean, amplitude), (1.5, 0.4));
        let mut flat = Ring::new(64, -1.0, 1.0, 10e-3).unwrap();
        let input: Vec<f64> = (0..64).map(|i| 3.0 + 0.2 * flat.angle(i).cos()).collect();
        flat.run(1e-3, &input, 20_000).unwrap();
        for i in 0..64 {
            assert!((flat.m[i] - (1.5 + 0.4 * flat.angle(i).cos())).abs() < 1e-12, "neuron {i}");
        }
        // The boundary between the two: with J₀ = −1, J₁ = 1 the mean is I₀/2 and the amplitude 2I₁,
        // so I₀ = 4I₁ touches zero at the trough and is still linear; a little less is rectified,
        // with all but a sliver of the ring active.
        assert_eq!(tuned_bump(4.0, 1.0, -1.0, 1.0), Some(Tuned::Linear { mean: 2.0, amplitude: 2.0 }));
        let Some(Tuned::Rectified { half_width, .. }) = tuned_bump(3.9, 1.0, -1.0, 1.0) else { panic!("just under the boundary") };
        assert!(half_width > 0.8 * PI && half_width < PI, "{half_width}");
        for none in [tuned_bump(1.0, 0.0, -1.0, 1.0), tuned_bump(1.0, 1.0, 1.0, 1.0), tuned_bump(1.0, 1.0, -1.0, 2.0), tuned_bump(-2.0, 1.0, -1.0, 1.0), tuned_bump(f64::NAN, 1.0, -1.0, 1.0)] {
            assert_eq!(none, None);
        }
    }

    #[test]
    fn a_bump_whose_inhibition_is_too_weak_has_no_height() {
        let theta_c = bump_half_width(6.0).unwrap();
        let edge = -theta_c.cos() / f0(theta_c);
        assert!(bump_amplitude(1.0, edge - 0.01, 6.0).is_some());
        assert_eq!(bump_amplitude(1.0, edge + 0.01, 6.0), None);
        assert_eq!(bump_amplitude(0.0, -4.0, 6.0), None);
        assert_eq!(bump_amplitude(-1.0, -4.0, 6.0), None);
        assert_eq!(bump_amplitude(1.0, f64::NAN, 6.0), None);
        // The height is linear in the input and nothing else about the bump changes.
        let (one, three) = (bump_amplitude(1.0, -4.0, 6.0).unwrap(), bump_amplitude(3.0, -4.0, 6.0).unwrap());
        assert!((three / one - 3.0).abs() < 1e-15);
    }

    #[test]
    fn bad_arguments_are_refused() {
        assert_eq!(Ring::new(2, -1.0, 3.0, 1e-2), Err(AttractorError::TooFew { n: 2 }));
        assert!(matches!(Ring::new(8, f64::NAN, 3.0, 1e-2), Err(AttractorError::NonFinite { what: "j0", .. })));
        assert!(matches!(Ring::new(8, -1.0, f64::INFINITY, 1e-2), Err(AttractorError::NonFinite { what: "j1", .. })));
        assert!(matches!(Ring::new(8, -1.0, 3.0, 0.0), Err(AttractorError::OutOfRange { what: "tau", .. })));
        let mut ring = Ring::new(8, -1.0, 3.0, 1e-2).unwrap();
        assert!(matches!(ring.step(1e-3, &[0.0; 7]), Err(AttractorError::Dimension { what: "input", got: 7, want: 8 })));
        let mut bad = [0.0; 8];
        bad[3] = f64::NAN;
        assert!(matches!(ring.step(1e-3, &bad), Err(AttractorError::NonFinite { what: "input", index: 3 })));
        assert!(matches!(ring.step(0.0, &[0.0; 8]), Err(AttractorError::OutOfRange { what: "dt", .. })));
        assert!(matches!(ring.step(1.1e-2, &[0.0; 8]), Err(AttractorError::OutOfRange { what: "dt", .. })));
        assert!(ring.step(1e-2, &[0.0; 8]).is_ok(), "dt = τ is the last legal step");
        assert!(matches!(ring.run(1e-3, &[0.0; 8], MAX_STEPS + 1), Err(AttractorError::OutOfRange { what: "steps", .. })));
        // A silent ring points nowhere; so does a uniform one.
        assert_eq!(ring.population_vector(), (None, 0.0));
        ring.m = vec![0.7; 8];
        assert_eq!(ring.population_vector().0, None);
        assert_eq!(ring.active_half_width(), PI);
        // A decaying rate reaches zero and stays there, instead of stalling on a denormal.
        let mut fading = Ring::new(8, -1.0, 3.0, 1e-2).unwrap();
        fading.m = vec![1e-300; 8];
        fading.run(1e-3, &[-1.0; 8], 1000).unwrap();
        assert_eq!(fading.m, vec![0.0; 8]);
        assert_eq!(fading.active_half_width(), 0.0);
        // The rectifier is on the DRIVE, so an inhibited neuron decays at its own rate and no
        // faster: from 1.0 under a strongly negative input, one step of τ/4 leaves exactly 0.75.
        // (Without the rectifier the step lands below zero and the flush above hides it — every
        // fixed point is the same, and this module's mutation sweep could not tell them apart.)
        let mut inhibited = Ring::new(8, -1.0, 3.0, 1e-2).unwrap();
        inhibited.m = vec![1.0; 8];
        inhibited.step(2.5e-3, &[-5.0; 8]).unwrap();
        assert_eq!(inhibited.m, vec![0.75; 8]);
        // One step from silence under input 1 with dt = τ/4 lands on a quarter of the drive.
        let mut fresh = Ring::new(8, -1.0, 3.0, 1e-2).unwrap();
        fresh.step(2.5e-3, &[1.0; 8]).unwrap();
        assert_eq!(fresh.m, vec![0.25; 8]);
        assert_eq!(fresh.angle(2), PI / 2.0);
        let cue = fresh.cue(2.0, 0.5, 0.0);
        assert_eq!(cue[0], 3.0);
        assert!((cue[4] - 1.0).abs() < 1e-15);
    }
}
