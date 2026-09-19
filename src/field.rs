//! The Amari neural field: a sheet of neurons with local excitation and broader inhibition, which
//! either forgets a stimulus or holds it as a bump of a width you can compute — and the width
//! below which a bump dies is computable too. Both are checked against the simulated field.
//!
//! # What the mechanism is
//!
//! Amari (*Dynamics of pattern formation in lateral-inhibition type neural fields*, Biological
//! Cybernetics 27(2):77–87, 1977) studied `τ ∂u/∂t = −u + ∫ w(x − y) H(u(y)) dy + h + s(x, t)`: a
//! continuum of units with activation `u`, a Heaviside output, a resting level `h < 0`, an input
//! `s`, and a "Mexican hat" kernel `w` — excitatory nearby, inhibitory further out. With a step
//! output the whole theory reduces to one function, the kernel's integral `W(a) = ∫₀ᵃ w`:
//!
//! - a stationary bump of width `a` exists iff `W(a) + h = 0` — the field at the bump's edge is
//!   exactly zero;
//! - it is stable iff `w(a) < 0` — widening the bump brings in net inhibition.
//!
//! `W` rises to a maximum `W_m` where the kernel changes sign and then falls toward `W_∞`. So with
//! `−h` between `W_∞` and `W_m` there are two roots: a narrow UNSTABLE bump `a₁` and a wide STABLE
//! one `a₂`. The field is bistable. A stimulus that leaves more than `a₁` active ignites a bump
//! that settles at `a₂` and stays after the stimulus is gone; one that leaves less dies. This is
//! the working-memory and decision element of dynamic field theory (Schöner, Spencer and the DFT
//! Research Group, *Dynamic Thinking: A Primer on Dynamic Field Theory*, Oxford University Press,
//! 2016), which has been run on neuromorphic hardware as a robot's attention and memory layer.
//!
//! # Why it is in a neuromorphic crate
//!
//! It is the continuum companion of [`crate::attractor`]'s ring: that module's bump has a width
//! set by a Fourier condition on a cosine kernel with threshold-linear units; this one's is set by
//! an integral condition on any kernel with step units, and it adds what the ring does not have —
//! a DETECTION THRESHOLD, the unstable bump, which is what lets a field ignore weak input and
//! commit to strong input. With a step output the recurrent input is a sum over ACTIVE units only,
//! which [`Field::step`] counts.
//!
//! # The closed forms this module is checked against
//!
//! - `W` is the integral of `w` (quadrature), in closed form through the error function; `w`
//!   changes sign at `x₀ = √(2 ln(A_e/A_i)/(1/σ_e² − 1/σ_i²))`, where `W` peaks.
//! - The two widths are roots of `W(a) + h = 0` with `w(a₁) > 0 > w(a₂)`; the three regimes —
//!   no bump, bistable, unbounded — are told apart by `W_m + h` and `W_∞ + h`.
//! - **The edge.** The stationary field is `u(x) = W(x) − W(x − a) + h`, which crosses zero at the
//!   bump's edge with slope `w(0) − w(a)`.
//! - **The unstable width is the separatrix.** A patch of activity 20% narrower than `a₁` dies; one
//!   20% wider grows and settles, and the width it settles at is `a₂` to within two grid steps.
//! - **Memory.** After a cue is removed the bump persists at `a₂`; in the no-bump regime the same
//!   cue leaves nothing.
//!
//! # What this module has NOT reproduced
//!
//! - Sigmoidal outputs, for which the widths have no closed form; two-dimensional fields; coupled
//!   fields; travelling bumps under asymmetric kernels.
//! - The unbounded regime's spreading front. [`regime`] names it and [`Field`] will simulate it,
//!   but nothing here checks the front's speed.
//! - Any DFT architecture — this is one field, the element those are built from.

use core::f64::consts::{FRAC_PI_2, SQRT_2};
use core::fmt;

use crate::surrogate::erf;

/// The most steps one `run` call will take; a request past it is refused.
pub const MAX_STEPS: u64 = 100_000_000;
/// The most grid points a field may have; the recurrent sum is quadratic in it.
pub const MAX_POINTS: usize = 1 << 16;

/// What went wrong, named rather than guessed around.
#[derive(Debug, Clone, PartialEq)]
pub enum FieldError {
    /// Too few grid points to hold a bump.
    TooFew {
        /// Points asked for.
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

impl fmt::Display for FieldError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooFew { n } => write!(f, "a field of {n} points cannot hold a bump (needs eight)"),
            Self::Dimension { what, got, want } => write!(f, "{what} has length {got}, expected {want}"),
            Self::OutOfRange { what, value, low, high } => {
                write!(f, "{what} = {value} is outside [{low}, {high}]")
            }
            Self::NonFinite { what, index } => write!(f, "{what} is not finite at {index}"),
        }
    }
}

impl std::error::Error for FieldError {}

fn positive(what: &'static str, v: f64) -> Result<f64, FieldError> {
    if v.is_finite() && v > 0.0 {
        Ok(v)
    } else {
        Err(FieldError::OutOfRange { what, value: v, low: f64::MIN_POSITIVE, high: f64::INFINITY })
    }
}

// ---------------------------------------------------------------------------------------------
// The kernel and what follows from it
// ---------------------------------------------------------------------------------------------

/// A Mexican-hat kernel `w(x) = A_e e^{−x²/2σ_e²} − A_i e^{−x²/2σ_i²}`: a narrow tall excitation
/// minus a wide low inhibition.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MexicanHat {
    /// Excitatory amplitude `A_e`, per unit length.
    pub a_e: f64,
    /// Excitatory width `σ_e`, length.
    pub s_e: f64,
    /// Inhibitory amplitude `A_i`, per unit length; below `A_e`.
    pub a_i: f64,
    /// Inhibitory width `σ_i`, length; above `σ_e`.
    pub s_i: f64,
}

/// Which of Amari's three cases a kernel and resting level are in.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Regime {
    /// `W_m + h < 0`: no bump can sustain itself; every stimulus is forgotten.
    NoBump,
    /// `W_∞ + h < 0 < W_m + h`: a narrow unstable bump and a wide stable one.
    Bistable {
        /// The unstable width `a₁`: activity narrower than this dies.
        ignition: f64,
        /// The stable width `a₂` a bump settles at.
        settled: f64,
    },
    /// `W_∞ + h ≥ 0`: once ignited, activity spreads without limit.
    Unbounded {
        /// The unstable width `a₁`.
        ignition: f64,
    },
}

impl MexicanHat {
    /// Build.
    ///
    /// # Errors
    ///
    /// [`FieldError::OutOfRange`] unless `A_e > A_i > 0` and `σ_i > σ_e > 0` — the conditions for
    /// the kernel to be excitatory at the centre and inhibitory in the surround.
    pub fn new(a_e: f64, s_e: f64, a_i: f64, s_i: f64) -> Result<Self, FieldError> {
        let (a_i, s_e) = (positive("a_i", a_i)?, positive("s_e", s_e)?);
        if !(a_e > a_i) || !a_e.is_finite() {
            return Err(FieldError::OutOfRange { what: "a_e", value: a_e, low: a_i, high: f64::INFINITY });
        }
        if !(s_i > s_e) || !s_i.is_finite() {
            return Err(FieldError::OutOfRange { what: "s_i", value: s_i, low: s_e, high: f64::INFINITY });
        }
        Ok(Self { a_e, s_e, a_i, s_i })
    }

    /// The kernel at separation `x`.
    #[must_use]
    pub fn w(&self, x: f64) -> f64 {
        self.a_e * (-x * x / (2.0 * self.s_e * self.s_e)).exp() - self.a_i * (-x * x / (2.0 * self.s_i * self.s_i)).exp()
    }

    /// Its integral `W(a) = ∫₀ᵃ w`: `√(π/2) [A_e σ_e erf(a/σ_e√2) − A_i σ_i erf(a/σ_i√2)]`.
    #[must_use]
    pub fn integral(&self, a: f64) -> f64 {
        FRAC_PI_2.sqrt() * (self.a_e * self.s_e * erf(a / (self.s_e * SQRT_2)) - self.a_i * self.s_i * erf(a / (self.s_i * SQRT_2)))
    }

    /// Where the kernel changes sign, and `W` peaks: `x₀ = √(2 ln(A_e/A_i)/(1/σ_e² − 1/σ_i²))`.
    #[must_use]
    pub fn zero_crossing(&self) -> f64 {
        (2.0 * (self.a_e / self.a_i).ln() / (1.0 / (self.s_e * self.s_e) - 1.0 / (self.s_i * self.s_i))).sqrt()
    }

    /// `W_∞ = √(π/2) (A_e σ_e − A_i σ_i)`: the net weight of half the kernel.
    #[must_use]
    pub fn integral_at_infinity(&self) -> f64 {
        FRAC_PI_2.sqrt() * (self.a_e * self.s_e - self.a_i * self.s_i)
    }
}

fn bisect(f: impl Fn(f64) -> f64, mut lo: f64, mut hi: f64) -> f64 {
    let rising = f(hi) > f(lo);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if (f(mid) > 0.0) == rising {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Amari's classification of a kernel and a resting level.
///
/// # Errors
///
/// [`FieldError::OutOfRange`] unless `h` is negative and finite: at `h ≥ 0` the resting field is
/// already active everywhere and there is no bump to speak of.
pub fn regime(kernel: &MexicanHat, h: f64) -> Result<Regime, FieldError> {
    if !(h < 0.0) || !h.is_finite() {
        return Err(FieldError::OutOfRange { what: "h", value: h, low: f64::NEG_INFINITY, high: 0.0 });
    }
    let x0 = kernel.zero_crossing();
    if kernel.integral(x0) + h < 0.0 {
        return Ok(Regime::NoBump);
    }
    let ignition = bisect(|a| kernel.integral(a) + h, 0.0, x0);
    if kernel.integral_at_infinity() + h >= 0.0 {
        return Ok(Regime::Unbounded { ignition });
    }
    // W has all but reached W_∞ forty inhibitory widths out.
    let settled = bisect(|a| kernel.integral(a) + h, x0, x0 + 40.0 * kernel.s_i);
    Ok(Regime::Bistable { ignition, settled })
}

// ---------------------------------------------------------------------------------------------
// The field
// ---------------------------------------------------------------------------------------------

/// A one-dimensional field on a periodic grid, with a step output.
#[derive(Debug, Clone, PartialEq)]
pub struct Field {
    /// The kernel.
    pub kernel: MexicanHat,
    /// Resting level `h`, negative.
    pub h: f64,
    /// Time constant `τ`, seconds.
    pub tau: f64,
    /// Length of the (periodic) domain.
    pub length: f64,
    /// Activation at each grid point; point `i` is at `i · length / n`.
    pub u: Vec<f64>,
}

impl Field {
    /// A field at rest, `u = h` everywhere.
    ///
    /// # Errors
    ///
    /// [`FieldError::TooFew`] for fewer than eight points, [`FieldError::OutOfRange`] for more than
    /// [`MAX_POINTS`], a non-negative `h`, a non-positive `tau`, or a domain shorter than twenty
    /// inhibitory widths — on a shorter ring a bump feels its own far side and the infinite-line
    /// theory stops applying.
    pub fn new(kernel: MexicanHat, h: f64, tau: f64, length: f64, n: usize) -> Result<Self, FieldError> {
        if n < 8 {
            return Err(FieldError::TooFew { n });
        }
        if n > MAX_POINTS {
            return Err(FieldError::OutOfRange { what: "n", value: n as f64, low: 8.0, high: MAX_POINTS as f64 });
        }
        if !(h < 0.0) || !h.is_finite() {
            return Err(FieldError::OutOfRange { what: "h", value: h, low: f64::NEG_INFINITY, high: 0.0 });
        }
        let tau = positive("tau", tau)?;
        if !(length >= 20.0 * kernel.s_i) || !length.is_finite() {
            return Err(FieldError::OutOfRange { what: "length", value: length, low: 20.0 * kernel.s_i, high: f64::INFINITY });
        }
        Ok(Self { kernel, h, tau, length, u: vec![h; n] })
    }

    /// Grid spacing.
    #[must_use]
    pub fn dx(&self) -> f64 {
        self.length / self.u.len() as f64
    }

    /// The total width of the active region, `dx` times the number of points with `u > 0`.
    #[must_use]
    pub fn active_width(&self) -> f64 {
        self.dx() * self.u.iter().filter(|u| **u > 0.0).count() as f64
    }

    /// Make a patch of `width` centred on `centre` just active (`u = −h/100`) and put the rest of
    /// the field at rest: the initial condition the separatrix is tested from.
    ///
    /// # Errors
    ///
    /// [`FieldError::OutOfRange`] for a width that is not in `(0, length/2]`,
    /// [`FieldError::NonFinite`] for a non-finite centre.
    pub fn seed(&mut self, centre: f64, width: f64) -> Result<(), FieldError> {
        if !centre.is_finite() {
            return Err(FieldError::NonFinite { what: "centre", index: 0 });
        }
        if !(width > 0.0) || !(width <= 0.5 * self.length) {
            return Err(FieldError::OutOfRange { what: "width", value: width, low: f64::MIN_POSITIVE, high: 0.5 * self.length });
        }
        let (dx, length, h) = (self.dx(), self.length, self.h);
        for (i, u) in self.u.iter_mut().enumerate() {
            let d = (i as f64 * dx - centre).rem_euclid(length);
            let d = d.min(length - d);
            *u = if d < 0.5 * width { -h / 100.0 } else { h };
        }
        Ok(())
    }

    /// One Euler step of `dt` under the input `input` (one value per grid point); returns the
    /// number of ACTIVE points, which is what the recurrent sum cost — `n × active` kernel
    /// evaluations, not `n²`.
    ///
    /// # Errors
    ///
    /// [`FieldError::Dimension`] for an input of the wrong length, [`FieldError::NonFinite`] for a
    /// bad entry, [`FieldError::OutOfRange`] for a `dt` outside `(0, τ]`.
    pub fn step(&mut self, dt: f64, input: &[f64]) -> Result<usize, FieldError> {
        let n = self.u.len();
        if input.len() != n {
            return Err(FieldError::Dimension { what: "input", got: input.len(), want: n });
        }
        if let Some(i) = input.iter().position(|x| !x.is_finite()) {
            return Err(FieldError::NonFinite { what: "input", index: i });
        }
        if !(dt > 0.0) || !(dt <= self.tau) {
            return Err(FieldError::OutOfRange { what: "dt", value: dt, low: f64::MIN_POSITIVE, high: self.tau });
        }
        let dx = self.dx();
        let active: Vec<usize> = (0..n).filter(|&j| self.u[j] > 0.0).collect();
        let a = dt / self.tau;
        // The kernel depends on the separation only, and on a ring that is at most n/2 hops.
        let by_hops: Vec<f64> = (0..=n / 2).map(|hops| self.kernel.w(dx * hops as f64)).collect();
        let next: Vec<f64> = (0..n)
            .map(|i| {
                let lateral: f64 = active
                    .iter()
                    .map(|&j| {
                        let hops = i.abs_diff(j);
                        by_hops[hops.min(n - hops)]
                    })
                    .sum();
                self.u[i] + a * (-self.u[i] + lateral * dx + self.h + input[i])
            })
            .collect();
        self.u = next;
        Ok(active.len())
    }

    /// Take `steps` steps of `dt` under a constant input.
    ///
    /// # Errors
    ///
    /// As [`Field::step`], plus [`FieldError::OutOfRange`] for more than [`MAX_STEPS`].
    pub fn run(&mut self, dt: f64, input: &[f64], steps: u64) -> Result<(), FieldError> {
        if steps > MAX_STEPS {
            return Err(FieldError::OutOfRange { what: "steps", value: steps as f64, low: 0.0, high: MAX_STEPS as f64 });
        }
        for _ in 0..steps {
            self.step(dt, input)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hat() -> MexicanHat {
        MexicanHat::new(2.0, 1.0, 1.0, 3.0).unwrap()
    }

    #[test]
    fn the_integral_is_the_integral_and_peaks_where_the_kernel_changes_sign() {
        let k = hat();
        for a in [0.3, 1.0, 2.5, 6.0, 15.0] {
            let n = 100_000;
            let quad: f64 = (0..n).map(|i| k.w(a * (f64::from(i) + 0.5) / f64::from(n))).sum::<f64>() * a / f64::from(n);
            assert!((quad - k.integral(a)).abs() < 1e-9, "W({a}) = {}, quadrature {quad}", k.integral(a));
        }
        assert_eq!(k.integral(0.0), 0.0);
        assert!((k.integral(-2.0) + k.integral(2.0)).abs() < 1e-15, "W is odd");
        // x₀² = 2 ln 2 / (1 − 1/9).
        let x0 = k.zero_crossing();
        assert!((x0 - (2.0 * 2.0f64.ln() / (8.0 / 9.0)).sqrt()).abs() < 1e-15);
        assert!(k.w(x0).abs() < 1e-15 && k.w(0.9 * x0) > 0.0 && k.w(1.1 * x0) < 0.0);
        assert!(k.integral(x0) > k.integral(0.9 * x0) && k.integral(x0) > k.integral(1.1 * x0));
        assert_eq!(k.w(0.0), 1.0);
        // W_∞ = √(π/2)(2·1 − 1·3): this kernel is net inhibitory.
        assert!((k.integral_at_infinity() + FRAC_PI_2.sqrt()).abs() < 1e-15);
        assert!((k.integral(200.0) - k.integral_at_infinity()).abs() < 1e-12);
    }

    #[test]
    fn the_three_regimes_are_told_apart_by_w_max_and_w_infinity() {
        let k = hat();
        let w_max = k.integral(k.zero_crossing());
        assert!(w_max > 0.5 && w_max < 1.0, "W_m = {w_max}");
        assert_eq!(regime(&k, -1.01 * w_max).unwrap(), Regime::NoBump);
        let Regime::Bistable { ignition, settled } = regime(&k, -0.5).unwrap() else { panic!("h = −0.5 is bistable") };
        assert!((k.integral(ignition) - 0.5).abs() < 1e-13 && (k.integral(settled) - 0.5).abs() < 1e-13);
        assert!(ignition < k.zero_crossing() && k.zero_crossing() < settled);
        assert!(k.w(ignition) > 0.0 && k.w(settled) < 0.0, "the narrow root is unstable, the wide one stable");
        // Closer to the top of W the two roots close on each other.
        let Regime::Bistable { ignition: i2, settled: s2 } = regime(&k, -0.95 * w_max).unwrap() else { panic!() };
        assert!(i2 > ignition && s2 < settled);
        // A kernel whose excitation outweighs its inhibition has W_∞ > 0: nothing stops the spread.
        let greedy = MexicanHat::new(2.0, 2.0, 1.0, 3.0).unwrap();
        assert!(greedy.integral_at_infinity() > 0.0);
        assert!(matches!(regime(&greedy, -0.5).unwrap(), Regime::Unbounded { ignition } if (greedy.integral(ignition) - 0.5).abs() < 1e-13));
        assert!(matches!(regime(&k, 0.0), Err(FieldError::OutOfRange { what: "h", .. })));
        assert!(matches!(regime(&k, f64::NAN), Err(FieldError::OutOfRange { what: "h", .. })));
    }

    #[test]
    fn the_unstable_width_is_the_line_between_forgetting_and_holding() {
        let k = hat();
        let h = -0.5;
        let Regime::Bistable { ignition, settled } = regime(&k, h).unwrap() else { panic!() };
        // a₁ is 0.55 here, so the grid has to be fine: dx = 0.025 puts 0.8 a₁ and 1.2 a₁ nine
        // points apart. (At the first draft's dx = 0.1 they were two points apart, and the guard
        // below is what said so.)
        let n = 2400;
        let quiet = vec![0.0; n];
        let mut narrow = Field::new(k, h, 10e-3, 60.0, n).unwrap();
        let dx = narrow.dx();
        assert!(ignition > 16.0 * dx, "the grid cannot tell 0.8 a₁ from 1.2 a₁: a₁ = {ignition}, dx = {dx}");
        narrow.seed(30.0, 0.8 * ignition).unwrap();
        narrow.run(5e-3, &quiet, 600).unwrap();
        assert_eq!(narrow.active_width(), 0.0, "a patch narrower than a₁ = {ignition} survived");
        assert!(narrow.u.iter().all(|u| (u - h).abs() < 1e-6), "and the field is back at rest");
        let mut wide = Field::new(k, h, 10e-3, 60.0, n).unwrap();
        wide.seed(30.0, 1.2 * ignition).unwrap();
        wide.run(5e-3, &quiet, 600).unwrap();
        assert!((wide.active_width() - settled).abs() <= 2.0 * dx, "settled at {}, Amari says {settled} (dx = {dx})", wide.active_width());
        // The settled bump's edge sits at zero, where the stationary field u(x) = W(x) − W(x − a) + h
        // has slope w(0) − w(a). So the last point inside and the first outside are within one
        // grid step of a zero of that slope — two, allowing the edge itself to be a step off.
        // (The first draft bounded this by |w(a)|·dx, the wrong derivative, and failed at 0.037.)
        let slope = k.w(0.0) - k.w(settled);
        let inside = wide.u.iter().copied().filter(|u| *u > 0.0).fold(f64::INFINITY, f64::min);
        let outside = wide.u.iter().copied().filter(|u| *u <= 0.0).fold(f64::NEG_INFINITY, f64::max);
        assert!(inside < 2.0 * slope * dx && -outside < 2.0 * slope * dx, "{inside} {outside} vs {}", 2.0 * slope * dx);
        assert!(inside + -outside > 0.5 * slope * dx, "the field is flat at the edge: {inside} {outside}");
        // The domain is a RING: a patch seeded across the seam at x = 0 is the same patch, and
        // settles at the same width. (Every other bump here sits mid-domain, where a kernel that
        // forgot to wrap its distances is indistinguishable — that mutation survived.)
        let mut seam = Field::new(k, h, 10e-3, 60.0, n).unwrap();
        seam.seed(0.0, 1.2 * ignition).unwrap();
        seam.run(5e-3, &quiet, 600).unwrap();
        assert!((seam.active_width() - settled).abs() <= 2.0 * dx, "across the seam it settled at {}", seam.active_width());
        assert!(seam.u[0] > 0.0 && seam.u[n - 1] > 0.0 && seam.u[n / 2] < 0.0);
        // Seeded from the wide side it comes DOWN to the same width.
        let mut fat = Field::new(k, h, 10e-3, 60.0, n).unwrap();
        fat.seed(30.0, 2.0 * settled).unwrap();
        fat.run(5e-3, &quiet, 600).unwrap();
        assert!((fat.active_width() - settled).abs() <= 2.0 * dx);
    }

    #[test]
    fn a_cue_is_remembered_in_the_bistable_regime_and_forgotten_outside_it() {
        let k = hat();
        let n = 800;
        let cue: Vec<f64> = (0..n).map(|i| { let x = 0.1 * f64::from(i) - 30.0; 1.5 * (-x * x / 8.0).exp() }).collect();
        let quiet = vec![0.0; n as usize];
        let mut memory = Field::new(k, -0.5, 10e-3, 80.0, n as usize).unwrap();
        let active = memory.step(1e-3, &cue).unwrap();
        assert_eq!(active, 0, "a field at rest has no active point, and its recurrent sum costs nothing");
        memory.run(1e-3, &cue, 500).unwrap();
        assert!(memory.active_width() > 0.0);
        memory.run(1e-3, &quiet, 3000).unwrap();
        let Regime::Bistable { settled, .. } = regime(&k, -0.5).unwrap() else { panic!() };
        assert!((memory.active_width() - settled).abs() <= 2.0 * memory.dx());
        // The bump is where the cue was: its centre of activity is x = 30.
        let (mut sum, mut count) = (0.0, 0.0);
        for (i, u) in memory.u.iter().enumerate() {
            if *u > 0.0 {
                sum += 0.1 * i as f64;
                count += 1.0;
            }
        }
        assert!((sum / count - 30.0).abs() <= memory.dx());
        // With the resting level below −W_m the same cue ignites the field and leaves nothing.
        let mut forgetful = Field::new(k, -0.9, 10e-3, 80.0, n as usize).unwrap();
        forgetful.run(1e-3, &cue, 500).unwrap();
        assert!(forgetful.active_width() > 0.0, "the cue is strong enough to activate the field while it lasts");
        forgetful.run(1e-3, &quiet, 3000).unwrap();
        assert_eq!(forgetful.active_width(), 0.0);
    }

    #[test]
    fn bad_arguments_are_refused() {
        assert!(matches!(MexicanHat::new(1.0, 1.0, 1.0, 3.0), Err(FieldError::OutOfRange { what: "a_e", .. })));
        assert!(matches!(MexicanHat::new(2.0, 1.0, 0.0, 3.0), Err(FieldError::OutOfRange { what: "a_i", .. })));
        assert!(matches!(MexicanHat::new(2.0, 0.0, 1.0, 3.0), Err(FieldError::OutOfRange { what: "s_e", .. })));
        assert!(matches!(MexicanHat::new(2.0, 3.0, 1.0, 3.0), Err(FieldError::OutOfRange { what: "s_i", .. })));
        assert!(matches!(MexicanHat::new(f64::INFINITY, 1.0, 1.0, 3.0), Err(FieldError::OutOfRange { what: "a_e", .. })));
        let k = hat();
        assert!(matches!(Field::new(k, -0.5, 1e-2, 80.0, 7), Err(FieldError::TooFew { n: 7 })));
        assert!(matches!(Field::new(k, -0.5, 1e-2, 80.0, MAX_POINTS + 1), Err(FieldError::OutOfRange { what: "n", .. })));
        assert!(matches!(Field::new(k, 0.0, 1e-2, 80.0, 100), Err(FieldError::OutOfRange { what: "h", .. })));
        assert!(matches!(Field::new(k, -0.5, 0.0, 80.0, 100), Err(FieldError::OutOfRange { what: "tau", .. })));
        assert!(matches!(Field::new(k, -0.5, 1e-2, 59.0, 100), Err(FieldError::OutOfRange { what: "length", .. })));
        let mut f = Field::new(k, -0.5, 1e-2, 60.0, 100).unwrap();
        assert_eq!(f.dx(), 0.6);
        assert!(f.u.iter().all(|u| *u == -0.5));
        assert!(matches!(f.step(1e-3, &[0.0; 99]), Err(FieldError::Dimension { what: "input", got: 99, want: 100 })));
        let mut bad = vec![0.0; 100];
        bad[7] = f64::NAN;
        assert!(matches!(f.step(1e-3, &bad), Err(FieldError::NonFinite { what: "input", index: 7 })));
        assert!(matches!(f.step(0.0, &[0.0; 100]), Err(FieldError::OutOfRange { what: "dt", .. })));
        assert!(matches!(f.step(2e-2, &[0.0; 100]), Err(FieldError::OutOfRange { what: "dt", .. })));
        assert!(matches!(f.run(1e-3, &[0.0; 100], MAX_STEPS + 1), Err(FieldError::OutOfRange { what: "steps", .. })));
        assert!(matches!(f.seed(f64::NAN, 1.0), Err(FieldError::NonFinite { what: "centre", .. })));
        assert!(matches!(f.seed(30.0, 0.0), Err(FieldError::OutOfRange { what: "width", .. })));
        assert!(matches!(f.seed(30.0, 31.0), Err(FieldError::OutOfRange { what: "width", .. })));
        // A seed across the periodic seam is one patch, not two: 3 units wide about x = 0 on a grid
        // of 0.6 is the points at 0, ±0.6, ±1.2.
        f.seed(0.0, 3.0).unwrap();
        assert_eq!(f.active_width(), 5.0 * 0.6);
        assert!(f.u[0] > 0.0 && f.u[2] > 0.0 && f.u[98] > 0.0 && f.u[3] < 0.0 && f.u[97] < 0.0);
        assert_eq!(f.u[0], 0.005);
        // One step at dt = τ puts a resting field exactly on rest plus input.
        let mut fresh = Field::new(k, -0.5, 1e-2, 60.0, 100).unwrap();
        fresh.step(1e-2, &[0.25; 100]).unwrap();
        assert!(fresh.u.iter().all(|u| *u == -0.25));
    }
}
