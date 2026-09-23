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
//! - **Two dimensions** ([`Field2`]). A circular bump of radius `R` is stationary iff the field at
//!   its rim is zero, `W₂(R) + h = 0`, with `W₂(R)` the kernel integrated over the disc from a
//!   point ON its rim. For a Gaussian that integral is `πσ² [1 − e^{−R²/σ²} I₀(R²/σ²)]` — half the
//!   Gaussian's mass as `R → ∞`, where the rim is a straight edge — checked against a
//!   two-dimensional quadrature, with the scaled Bessel function [`bessel_i0e`] checked against
//!   its integral representation. The simulated sheet forgets a disc smaller than the unstable
//!   radius and holds a larger one at the stable radius, to a grid step.
//!
//! # What this module has NOT reproduced
//!
//! - Sigmoidal outputs, for which the widths have no closed form; coupled fields; travelling
//!   bumps under asymmetric kernels; non-circular two-dimensional solutions (stripes, rings,
//!   multi-bump states), which the same sheet can support.
//! - The unbounded regime's spreading front. [`regime`] names it and [`Field`] will simulate it,
//!   but nothing here checks the front's speed.
//! - Any DFT architecture — this is one field, the element those are built from.

use core::f64::consts::{FRAC_PI_2, PI, SQRT_2, TAU};
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

// ---------------------------------------------------------------------------------------------
// Two dimensions
// ---------------------------------------------------------------------------------------------

/// The exponentially scaled modified Bessel function `e^{−x} I₀(x)`, for `x ≥ 0` (the argument's
/// sign is ignored; `I₀` is even). The power series below 20 and the asymptotic series above it.
#[must_use]
pub fn bessel_i0e(x: f64) -> f64 {
    let x = x.abs();
    if x <= 20.0 {
        let q = 0.25 * x * x;
        let (mut term, mut sum) = (1.0f64, 1.0f64);
        for k in 1..400u32 {
            term *= q / (f64::from(k) * f64::from(k));
            sum += term;
            if term < 1e-17 * sum {
                break;
            }
        }
        sum * (-x).exp()
    } else {
        let (mut term, mut sum) = (1.0f64, 1.0f64);
        for k in 1..40u32 {
            let odd = 2.0 * f64::from(k) - 1.0;
            let next = term * odd * odd / (f64::from(k) * 8.0 * x);
            // An asymptotic series: stop at its smallest term.
            if next >= term {
                break;
            }
            term = next;
            sum += term;
        }
        sum / (TAU * x).sqrt()
    }
}

impl MexicanHat {
    /// The kernel integrated over a disc of radius `r`, seen from a point ON the disc's rim, with
    /// the amplitudes read per unit AREA: `Σ ±A πσ² [1 − e^{−r²/σ²} I₀(r²/σ²)]`.
    #[must_use]
    pub fn rim_integral(&self, r: f64) -> f64 {
        let part = |a: f64, s: f64| a * PI * s * s * (1.0 - bessel_i0e(r * r / (s * s)));
        part(self.a_e, self.s_e) - part(self.a_i, self.s_i)
    }

    /// Its limit for a large disc, `π (A_e σ_e² − A_i σ_i²)`: half the kernel's total weight.
    #[must_use]
    pub fn rim_integral_at_infinity(&self) -> f64 {
        PI * (self.a_e * self.s_e * self.s_e - self.a_i * self.s_i * self.s_i)
    }
}

/// Amari's classification for a circular bump on a sheet: [`Regime`] with RADII in place of
/// widths. The peak of `W₂` has no closed form and is found by a scan of 4000 points out to
/// twenty inhibitory widths.
///
/// # Errors
///
/// [`FieldError::OutOfRange`] unless `h` is negative and finite.
pub fn regime2(kernel: &MexicanHat, h: f64) -> Result<Regime, FieldError> {
    if !(h < 0.0) || !h.is_finite() {
        return Err(FieldError::OutOfRange { what: "h", value: h, low: f64::NEG_INFINITY, high: 0.0 });
    }
    let far = 20.0 * kernel.s_i;
    let peak = (1..=4000).map(|k| far * f64::from(k) / 4000.0).fold((0.0, f64::NEG_INFINITY), |best, r| {
        let w = kernel.rim_integral(r);
        if w > best.1 { (r, w) } else { best }
    });
    if peak.1 + h < 0.0 {
        return Ok(Regime::NoBump);
    }
    let ignition = bisect(|r| kernel.rim_integral(r) + h, 0.0, peak.0);
    if kernel.rim_integral_at_infinity() + h >= 0.0 {
        return Ok(Regime::Unbounded { ignition });
    }
    let settled = bisect(|r| kernel.rim_integral(r) + h, peak.0, 2.0 * far);
    Ok(Regime::Bistable { ignition, settled })
}

/// A two-dimensional field on a periodic `n × n` grid, with a step output.
#[derive(Debug, Clone, PartialEq)]
pub struct Field2 {
    /// The kernel, its amplitudes read per unit area.
    pub kernel: MexicanHat,
    /// Resting level `h`, negative.
    pub h: f64,
    /// Time constant `τ`, seconds.
    pub tau: f64,
    /// Side of the (periodic) square domain.
    pub length: f64,
    /// Grid points per side.
    pub n: usize,
    /// Activation, row-major `n × n`.
    pub u: Vec<f64>,
}

impl Field2 {
    /// A sheet at rest.
    ///
    /// # Errors
    ///
    /// [`FieldError::TooFew`] for fewer than eight points a side, [`FieldError::OutOfRange`] for
    /// more than 512, a non-negative `h`, a non-positive `tau`, or a side shorter than twelve
    /// inhibitory widths.
    pub fn new(kernel: MexicanHat, h: f64, tau: f64, length: f64, n: usize) -> Result<Self, FieldError> {
        if n < 8 {
            return Err(FieldError::TooFew { n });
        }
        if n > 512 {
            return Err(FieldError::OutOfRange { what: "n", value: n as f64, low: 8.0, high: 512.0 });
        }
        if !(h < 0.0) || !h.is_finite() {
            return Err(FieldError::OutOfRange { what: "h", value: h, low: f64::NEG_INFINITY, high: 0.0 });
        }
        let tau = positive("tau", tau)?;
        if !(length >= 12.0 * kernel.s_i) || !length.is_finite() {
            return Err(FieldError::OutOfRange { what: "length", value: length, low: 12.0 * kernel.s_i, high: f64::INFINITY });
        }
        Ok(Self { kernel, h, tau, length, n, u: vec![h; n * n] })
    }

    /// Grid spacing.
    #[must_use]
    pub fn dx(&self) -> f64 {
        self.length / self.n as f64
    }

    /// The radius of the disc whose area equals the active area: `√(dx² · active/π)`.
    #[must_use]
    pub fn active_radius(&self) -> f64 {
        let area = self.dx() * self.dx() * self.u.iter().filter(|u| **u > 0.0).count() as f64;
        (area / PI).sqrt()
    }

    /// Make a disc of `radius` about `centre` just active and put the rest of the sheet at rest.
    ///
    /// # Errors
    ///
    /// [`FieldError::OutOfRange`] for a radius not in `(0, length/4]`, [`FieldError::NonFinite`] for
    /// a non-finite centre.
    pub fn seed(&mut self, centre: [f64; 2], radius: f64) -> Result<(), FieldError> {
        if !centre[0].is_finite() || !centre[1].is_finite() {
            return Err(FieldError::NonFinite { what: "centre", index: 0 });
        }
        if !(radius > 0.0) || !(radius <= 0.25 * self.length) {
            return Err(FieldError::OutOfRange { what: "radius", value: radius, low: f64::MIN_POSITIVE, high: 0.25 * self.length });
        }
        let (dx, length, h, n) = (self.dx(), self.length, self.h, self.n);
        let wrapped = |a: f64| {
            let d = a.rem_euclid(length);
            d.min(length - d)
        };
        for iy in 0..n {
            for ix in 0..n {
                let (ddx, ddy) = (wrapped(ix as f64 * dx - centre[0]), wrapped(iy as f64 * dx - centre[1]));
                self.u[iy * n + ix] = if ddx.hypot(ddy) < radius { -h / 100.0 } else { h };
            }
        }
        Ok(())
    }

    /// One Euler step of `dt` with no input; returns the number of active points.
    ///
    /// # Errors
    ///
    /// [`FieldError::OutOfRange`] for a `dt` outside `(0, τ]`.
    pub fn step(&mut self, dt: f64) -> Result<usize, FieldError> {
        if !(dt > 0.0) || !(dt <= self.tau) {
            return Err(FieldError::OutOfRange { what: "dt", value: dt, low: f64::MIN_POSITIVE, high: self.tau });
        }
        let (n, dx) = (self.n, self.dx());
        let half = n / 2 + 1;
        let table: Vec<f64> = (0..half * half).map(|q| self.kernel.w(dx * ((q / half) as f64).hypot((q % half) as f64))).collect();
        let active: Vec<(usize, usize)> = (0..n * n).filter(|&q| self.u[q] > 0.0).map(|q| (q / n, q % n)).collect();
        let a = dt / self.tau;
        let hops = |p: usize, q: usize| {
            let d = p.abs_diff(q);
            d.min(n - d)
        };
        let next: Vec<f64> = (0..n * n)
            .map(|q| {
                let (iy, ix) = (q / n, q % n);
                let lateral: f64 = active.iter().map(|&(jy, jx)| table[hops(iy, jy) * half + hops(ix, jx)]).sum();
                self.u[q] + a * (-self.u[q] + lateral * dx * dx + self.h)
            })
            .collect();
        self.u = next;
        Ok(active.len())
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
    fn the_scaled_bessel_function_is_its_integral_representation() {
        // e^{−x} I₀(x) = (1/π) ∫₀^π e^{x (cos t − 1)} dt, by the midpoint rule on 200 000 points.
        for x in [0.0, 0.5, 3.0, 12.0, 19.9, 20.1, 45.0, 300.0] {
            let n = 200_000;
            let quad: f64 = (0..n).map(|k| (x * ((PI * (f64::from(k) + 0.5) / f64::from(n)).cos() - 1.0)).exp()).sum::<f64>() / f64::from(n);
            assert!((bessel_i0e(x) / quad - 1.0).abs() < 1e-10, "x = {x}: {} vs {quad}", bessel_i0e(x));
        }
        assert_eq!(bessel_i0e(0.0), 1.0);
        assert_eq!(bessel_i0e(-3.0), bessel_i0e(3.0));
        // The two branches meet at x = 20.
        assert!((bessel_i0e(20.0) / bessel_i0e(20.0 + 1e-9) - 1.0).abs() < 1e-9);
        // Far out it is 1/√(2πx).
        assert!((bessel_i0e(1e6) * (TAU * 1e6f64).sqrt() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn the_rim_integral_is_the_kernel_summed_over_the_disc_from_its_edge() {
        let k = MexicanHat::new(2.0, 1.0, 1.0, 2.0).unwrap();
        for r in [0.4, 1.0, 2.5] {
            // Polar quadrature about the disc's centre, the observer at (r, 0).
            let (nr, nt) = (1500, 1500);
            let mut quad = 0.0;
            for i in 0..nr {
                let rho = r * (f64::from(i) + 0.5) / f64::from(nr);
                for j in 0..nt {
                    let t = TAU * (f64::from(j) + 0.5) / f64::from(nt);
                    quad += k.w((rho * t.cos() - r).hypot(rho * t.sin())) * rho;
                }
            }
            quad *= (r / f64::from(nr)) * (TAU / f64::from(nt));
            assert!((quad - k.rim_integral(r)).abs() < 1e-5, "W₂({r}) = {}, quadrature {quad}", k.rim_integral(r));
        }
        assert_eq!(k.rim_integral(0.0), 0.0);
        assert!((k.rim_integral_at_infinity() - PI * (2.0 - 4.0)).abs() < 1e-15);
        // The rim of a huge disc is a straight edge, approached as 1/R: e^{−x} I₀(x) → 1/√(2πx)
        // gives W₂(R) ≈ W₂(∞) − √(π/2)(A_e σ_e³ − A_i σ_i³)/R, which is +7.52/R for this kernel,
        // and what is left after that term falls as 1/R³. (The first draft asserted "within 0.2%
        // of W₂(∞) at R = 400"; it is 0.3%, and the approach has a law, so the law is tested.)
        let correction = |r: f64| -FRAC_PI_2.sqrt() * (2.0 - 8.0) / r;
        for r in [200.0, 400.0] {
            let left = k.rim_integral(r) - k.rim_integral_at_infinity() - correction(r);
            assert!(left.abs() < 1.0 / (r * r * r) * 50.0, "R = {r}: {left} left after the 1/R term");
        }
    }

    #[test]
    fn a_sheet_forgets_a_small_disc_and_holds_a_large_one_at_the_stable_radius() {
        let k = MexicanHat::new(2.0, 1.0, 0.6, 2.0).unwrap();
        let h = -1.274;
        let Regime::Bistable { ignition, settled } = regime2(&k, h).unwrap() else { panic!("h = {h} is bistable") };
        // Found independently, by a scan in another language, while choosing this fixture.
        assert!((ignition - 0.664).abs() < 2e-3 && (settled - 1.75).abs() < 2e-3, "{ignition} {settled}");
        assert!((k.rim_integral(ignition) + h).abs() < 1e-12 && (k.rim_integral(settled) + h).abs() < 1e-12);
        assert!(ignition < settled);
        let n = 144;
        let mut small = Field2::new(k, h, 10e-3, 24.0, n).unwrap();
        let dx = small.dx();
        assert!(0.6 * ignition > 2.0 * dx, "the grid cannot draw a disc of 0.6 R₁: R₁ = {ignition}, dx = {dx}");
        small.seed([12.0, 12.0], 0.6 * ignition).unwrap();
        for _ in 0..100 {
            small.step(5e-3).unwrap();
        }
        assert_eq!(small.active_radius(), 0.0, "a disc smaller than R₁ = {ignition} survived");
        let mut large = Field2::new(k, h, 10e-3, 24.0, n).unwrap();
        large.seed([12.0, 12.0], 1.5 * ignition).unwrap();
        let mut active = 0;
        for _ in 0..100 {
            active = large.step(5e-3).unwrap();
        }
        assert!(active > 0);
        assert!((large.active_radius() - settled).abs() <= dx, "settled at radius {}, Amari says {settled} (dx = {dx})", large.active_radius());
        // A disc across the periodic corner is the same disc.
        let mut corner = Field2::new(k, h, 10e-3, 24.0, n).unwrap();
        corner.seed([0.0, 0.0], 1.5 * ignition).unwrap();
        for _ in 0..100 {
            corner.step(5e-3).unwrap();
        }
        assert!((corner.active_radius() - settled).abs() <= dx);
        assert!(corner.u[0] > 0.0 && corner.u[n * n - 1] > 0.0 && corner.u[(n / 2) * n + n / 2] < 0.0);
        // The regimes, as in one dimension.
        assert_eq!(regime2(&k, -50.0).unwrap(), Regime::NoBump);
        let greedy = MexicanHat::new(2.0, 1.5, 0.6, 2.0).unwrap();
        assert!(matches!(regime2(&greedy, -0.1).unwrap(), Regime::Unbounded { .. }));
        assert!(matches!(regime2(&k, 0.0), Err(FieldError::OutOfRange { what: "h", .. })));
    }

    #[test]
    fn a_sheet_refuses_bad_arguments() {
        let k = MexicanHat::new(2.0, 1.0, 1.0, 2.0).unwrap();
        assert!(matches!(Field2::new(k, -0.5, 1e-2, 24.0, 7), Err(FieldError::TooFew { n: 7 })));
        assert!(matches!(Field2::new(k, -0.5, 1e-2, 24.0, 513), Err(FieldError::OutOfRange { what: "n", .. })));
        assert!(matches!(Field2::new(k, 0.5, 1e-2, 24.0, 16), Err(FieldError::OutOfRange { what: "h", .. })));
        assert!(matches!(Field2::new(k, -0.5, 0.0, 24.0, 16), Err(FieldError::OutOfRange { what: "tau", .. })));
        assert!(matches!(Field2::new(k, -0.5, 1e-2, 23.0, 16), Err(FieldError::OutOfRange { what: "length", .. })));
        let mut f = Field2::new(k, -0.5, 1e-2, 24.0, 16).unwrap();
        assert_eq!((f.dx(), f.u.len()), (1.5, 256));
        assert!(matches!(f.step(0.0), Err(FieldError::OutOfRange { what: "dt", .. })));
        assert!(matches!(f.step(2e-2), Err(FieldError::OutOfRange { what: "dt", .. })));
        assert!(matches!(f.seed([f64::NAN, 0.0], 1.0), Err(FieldError::NonFinite { what: "centre", .. })));
        assert!(matches!(f.seed([0.0, 0.0], 0.0), Err(FieldError::OutOfRange { what: "radius", .. })));
        assert!(matches!(f.seed([0.0, 0.0], 6.5), Err(FieldError::OutOfRange { what: "radius", .. })));
        // A disc of radius 2 about a grid point on a grid of 1.5 is that point and its four
        // neighbours (the diagonals are 2.12 away): area 5 · 2.25, radius √(11.25/π).
        f.seed([6.0, 6.0], 2.0).unwrap();
        assert!((f.active_radius() - (11.25f64 / PI).sqrt()).abs() < 1e-12);
        assert_eq!(f.step(1e-2).unwrap(), 5);
        // A resting sheet stays at rest.
        let mut rest = Field2::new(k, -0.5, 1e-2, 24.0, 16).unwrap();
        assert_eq!(rest.step(5e-3).unwrap(), 0);
        assert!(rest.u.iter().all(|u| *u == -0.5));
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

    /// Pins the wording of the out-of-range refusal: the LOWEST admissible value is printed first.
    /// Every test here matches that variant with `..` and reads its `what` only, so nothing in the
    /// suite ever formatted one — a refusal that printed its interval end for end, `[tau, 0]` for a
    /// step that must be in `(0, tau]`, said the opposite of what it meant and still passed.
    #[test]
    fn the_out_of_range_refusal_prints_its_lowest_bound_first() {
        let refusal = FieldError::OutOfRange { what: "dt", value: 0.0, low: 1.0, high: 2.0 };
        assert_eq!(refusal.to_string(), "dt = 0 is outside [1, 2]");
        assert_eq!(FieldError::TooFew { n: 3 }.to_string(), "a field of 3 points cannot hold a bump (needs eight)");
        let mismatch = FieldError::Dimension { what: "input", got: 9, want: 10 };
        assert_eq!(mismatch.to_string(), "input has length 9, expected 10");
        assert_eq!(FieldError::NonFinite { what: "centre", index: 4 }.to_string(), "centre is not finite at 4");
    }

    /// Pins that an INFINITY is refused by every range guard in this module, not only a `NaN` or a
    /// value on the wrong side of the bound. Those are the two shapes the suite probes with, and an
    /// infinity passes neither test: `inf > 0.0`, `inf > s_e` and `inf >= 20 s_i` are all true, so
    /// five separate finiteness checks here could be deleted without a test noticing.
    #[test]
    fn an_infinity_is_not_a_value_any_of_these_guards_admits() {
        let k = hat();
        // `positive`, which screens `tau` and the inhibitory amplitude.
        assert!(matches!(Field::new(k, -0.5, f64::INFINITY, 60.0, 100), Err(FieldError::OutOfRange { what: "tau", .. })));
        assert!(matches!(Field2::new(k, -0.5, f64::INFINITY, 60.0, 16), Err(FieldError::OutOfRange { what: "tau", .. })));
        assert!(matches!(MexicanHat::new(2.0, 1.0, f64::INFINITY, 3.0), Err(FieldError::OutOfRange { what: "a_i", .. })));
        // The kernel's inhibitory width, which is only ever compared against the excitatory one.
        assert!(matches!(MexicanHat::new(2.0, 1.0, 1.0, f64::INFINITY), Err(FieldError::OutOfRange { what: "s_i", .. })));
        // The resting level, in one dimension and two.
        assert!(matches!(regime(&k, f64::NEG_INFINITY), Err(FieldError::OutOfRange { what: "h", .. })));
        assert!(matches!(regime2(&k, f64::NEG_INFINITY), Err(FieldError::OutOfRange { what: "h", .. })));
        // The domain, which is only ever compared against twenty inhibitory widths.
        assert!(matches!(Field::new(k, -0.5, 1e-2, f64::INFINITY, 100), Err(FieldError::OutOfRange { what: "length", .. })));
        assert!(matches!(Field2::new(k, -0.5, 1e-2, f64::INFINITY, 16), Err(FieldError::OutOfRange { what: "length", .. })));
        // An input entry, at both signs: the check is finiteness, not `is_nan`.
        let mut f = Field::new(k, -0.5, 1e-2, 60.0, 100).unwrap();
        let mut blown = vec![0.0; 100];
        blown[3] = f64::INFINITY;
        assert!(matches!(f.step(1e-3, &blown), Err(FieldError::NonFinite { what: "input", index: 3 })));
        blown[3] = f64::NEG_INFINITY;
        assert!(matches!(f.step(1e-3, &blown), Err(FieldError::NonFinite { what: "input", index: 3 })));
        // And the SECOND coordinate of a sheet's seed, which no test had ever made bad.
        let mut sheet = Field2::new(k, -0.5, 1e-2, 60.0, 16).unwrap();
        assert!(matches!(sheet.seed([0.0, f64::INFINITY], 1.0), Err(FieldError::NonFinite { what: "centre", .. })));
        assert!(matches!(sheet.seed([0.0, f64::NAN], 1.0), Err(FieldError::NonFinite { what: "centre", .. })));
        assert!(matches!(sheet.seed([f64::NEG_INFINITY, 0.0], 1.0), Err(FieldError::NonFinite { what: "centre", .. })));
    }

    /// Pins that `regime` looks for `W`'s maximum at the kernel's ZERO CROSSING rather than at the
    /// excitatory width. Measured for this kernel: `W(s_e) = 0.729463`, `W(x0) = 0.762226`. At
    /// `h = -0.75` the field is bistable, but a classifier that took `s_e` for the peak sees
    /// `0.729463 - 0.75 < 0` and answers `NoBump`. The suite's resting levels are `-0.5`, which is
    /// below both, and `-0.95 W_m` and `-1.01 W_m`, both derived FROM `W(x0)` and so always on the
    /// same side of it as of `W(s_e)` — no fixture fell in the gap between the two candidates.
    #[test]
    fn the_classifier_finds_w_s_maximum_at_the_zero_crossing_not_at_the_excitatory_width() {
        let k = hat();
        let h = -0.75;
        assert!(
            k.integral(k.s_e) < -h && -h < k.integral(k.zero_crossing()),
            "the fixture is the gap between the two candidate peaks: W(s_e) = {}, W(x0) = {}",
            k.integral(k.s_e),
            k.integral(k.zero_crossing())
        );
        let Regime::Bistable { ignition, settled } = regime(&k, h).unwrap() else {
            panic!("-h lies below W's true maximum, so this kernel is bistable at h = {h}")
        };
        assert!((k.integral(ignition) + h).abs() < 1e-13 && (k.integral(settled) + h).abs() < 1e-13);
        assert!(ignition < k.zero_crossing() && k.zero_crossing() < settled);
        assert!(k.w(ignition) > 0.0 && k.w(settled) < 0.0, "the narrow root is unstable, the wide one stable");
    }

    /// Pins that the STABLE width is bisected from the peak of `W` upward, not from zero. Started
    /// at zero the bracket holds both roots, and `bisect` reads its direction from the two ends:
    /// on a kernel whose half-integral `W_inf` is POSITIVE those ends are `h` and `W_inf + h`, so
    /// `f(hi) > f(lo)` and it treats the function as rising. It then walks its LOWER end up to the
    /// far edge and returns `x0 + 40 s_i` — measured here as `123.159` against a true stable width
    /// of `5.324`. Every bistable fixture in the suite is net-inhibitory (`W_inf < 0`), where a
    /// bracket started at zero happens to converge on the right root anyway.
    #[test]
    fn the_stable_width_is_bracketed_above_the_peak_even_when_the_half_kernel_is_net_excitatory() {
        let broad = MexicanHat::new(2.0, 2.0, 1.0, 3.0).unwrap();
        let h = -1.5;
        assert!(broad.integral_at_infinity() > 0.0, "the half-kernel is net excitatory: W_inf = {}", broad.integral_at_infinity());
        assert!(broad.integral_at_infinity() + h < 0.0, "and still bounded at this resting level");
        let Regime::Bistable { ignition, settled } = regime(&broad, h).unwrap() else {
            panic!("W_inf + h < 0 < W_m + h is the bistable case")
        };
        assert!(
            (broad.integral(settled) + h).abs() < 1e-13,
            "the stable width came back as {settled}, where W + h = {}",
            broad.integral(settled) + h
        );
        assert!((broad.integral(ignition) + h).abs() < 1e-13);
        assert!(broad.w(ignition) > 0.0 && broad.w(settled) < 0.0, "the narrow root is unstable, the wide one stable");
        assert!(settled < broad.zero_crossing() + 40.0 * broad.s_i);
    }

    /// Pins that the sheet's scan reaches TWENTY inhibitory widths. Measured for this kernel:
    /// `W2` is still `+0.2048` at `r = 4 s_i`, so at `h = -0.15` the stable radius is `9.549` —
    /// past `2 far` if `far` were only two inhibitory widths, which leaves the bisection with a
    /// bracket that is positive at both ends and returns that edge, `8.0`, as the radius. The
    /// suite's one sheet fixture settles at `1.75`, comfortably inside either span, and its
    /// `W2` has already gone negative by `2 s_i`, so no scan length could matter there.
    #[test]
    fn the_sheets_scan_reaches_far_enough_to_bracket_a_wide_stable_radius() {
        let shallow = MexicanHat::new(2.0, 1.0, 0.51, 2.0).unwrap();
        let h = -0.15;
        assert!(shallow.rim_integral(4.0 * shallow.s_i) > -h, "the stable radius is past four inhibitory widths");
        let Regime::Bistable { ignition, settled } = regime2(&shallow, h).unwrap() else {
            panic!("W2_inf + h < 0 < W2_m + h is the bistable case")
        };
        assert!(
            (shallow.rim_integral(settled) + h).abs() < 1e-12,
            "the stable radius came back as {settled}, where W2 + h = {}",
            shallow.rim_integral(settled) + h
        );
        assert!((shallow.rim_integral(ignition) + h).abs() < 1e-12);
        assert!(settled > 4.0 * shallow.s_i && ignition < settled);
        assert!(shallow.rim_integral_at_infinity() + h < 0.0);
    }

    /// Pins that the sheet's STABLE radius is bisected from the peak of `W2` upward. Started at
    /// zero the bracket holds both roots and its lower end never moves, because `bisect` raises it
    /// only on a positive midpoint — so the midpoints are the dyadic points `40 s_i / 2^k`, here
    /// `2.5`, `1.25`, `0.625`. Measured at `h = -1.80`: the two roots are `1.0354` and `1.2261`,
    /// which fall in the gap between `0.625` and `1.25`, so every midpoint is negative, the upper
    /// end collapses and the stable radius comes back as `2.5e-59`. The suite's `h = -1.274` puts
    /// the roots at `0.664` and `1.75`, straddling `1.25`, where the same broken bracket lands on
    /// the right root — a single-root-per-gap fixture cannot tell a correct bracket from a lucky one.
    #[test]
    fn the_sheets_stable_radius_is_bracketed_above_the_peak_of_its_rim_integral() {
        let squeezed = MexicanHat::new(2.0, 1.0, 0.6, 2.0).unwrap();
        let h = -1.80;
        let Regime::Bistable { ignition, settled } = regime2(&squeezed, h).unwrap() else {
            panic!("-h lies below the maximum of W2, so this sheet is bistable at h = {h}")
        };
        let hi = 2.0 * 20.0 * squeezed.s_i;
        assert!(
            ignition > hi / 128.0 && settled < hi / 64.0,
            "the fixture is one dyadic gap of the bracket [0, {hi}]: {ignition} {settled}"
        );
        assert!(
            (squeezed.rim_integral(settled) + h).abs() < 1e-12,
            "the stable radius came back as {settled}, where W2 + h = {}",
            squeezed.rim_integral(settled) + h
        );
        assert!((squeezed.rim_integral(ignition) + h).abs() < 1e-12);
        assert!(ignition < settled);
    }

    /// Pins that `seed` puts everything OUTSIDE the patch back to the resting level. Every other
    /// seed in the suite is applied to a field that is already at rest, where "write `h` outside
    /// the patch" and "leave the outside as it was" put down the same numbers; a seed that only
    /// ever adds activity would carry the previous trial's bump into the next one.
    #[test]
    fn seeding_returns_the_field_outside_the_patch_to_rest() {
        let k = hat();
        let mut f = Field::new(k, -0.5, 1e-2, 60.0, 100).unwrap();
        f.u.fill(7.0);
        f.seed(30.0, 3.0).unwrap();
        // dx = 0.6, so |x - 30| < 1.5 is the five points 48..=52 and nothing else.
        for (i, u) in f.u.iter().enumerate() {
            let want = if (48..=52).contains(&i) { 0.005 } else { -0.5 };
            assert_eq!(*u, want, "point {i}");
        }
        assert_eq!(f.active_width(), 5.0 * 0.6);
    }

    /// Pins that a patch is seeded about the centre it was GIVEN and not about its reflection
    /// through the origin. Every other seeded patch in the suite is centred at `0` or at half the
    /// domain — the two points on a ring that a reflection leaves exactly where they are.
    #[test]
    fn a_patch_is_seeded_about_the_centre_given_not_about_its_reflection() {
        let k = hat();
        let mut f = Field::new(k, -0.5, 1e-2, 60.0, 120).unwrap();
        assert_eq!(f.dx(), 0.5);
        f.seed(10.0, 3.0).unwrap();
        // |x - 10| < 1.5 is the five points 18..=22; reflected, it would be 98..=102.
        for (i, u) in f.u.iter().enumerate() {
            let want = if (18..=22).contains(&i) { 0.005 } else { -0.5 };
            assert_eq!(*u, want, "point {i}");
        }
    }

    /// Pins that a disc is seeded at `[x, y]` — column from the FIRST coordinate, row from the
    /// second — and stored row-major. The suite's discs are centred at `[12, 12]`, `[0, 0]` and
    /// `[6, 6]`, all on the diagonal, and a diagonal disc is unchanged both by transposing the
    /// sheet and by swapping the centre's two coordinates, so neither edit was visible anywhere.
    #[test]
    fn a_disc_is_seeded_at_its_column_and_row_in_that_order() {
        let k = MexicanHat::new(2.0, 1.0, 1.0, 2.0).unwrap();
        let mut sheet = Field2::new(k, -0.5, 1e-2, 24.0, 16).unwrap();
        // dx = 1.5, so [6, 12] is the grid point at column 4 of row 8; a radius of 2 takes it and
        // its four edge neighbours, the diagonals being 2.12 away.
        sheet.seed([6.0, 12.0], 2.0).unwrap();
        let patch = [7 * 16 + 4, 8 * 16 + 3, 8 * 16 + 4, 8 * 16 + 5, 9 * 16 + 4];
        for (q, u) in sheet.u.iter().enumerate() {
            let want = if patch.contains(&q) { 0.005 } else { -0.5 };
            assert_eq!(*u, want, "point {q}");
        }
    }

    /// Pins that the sheet's recurrent sum reads each active point's ROW before its column. The
    /// kernel table is indexed by row separation then column separation, so a pair read the other
    /// way round is looked up at a different distance. The suite only ever steps a sheet whose
    /// active set is a disc about a point on the diagonal, and such a set is its own transpose —
    /// every entry it reads is the one it should have read. The two values below are the whole
    /// Euler step written out in the same operations and the same order, so they are exact.
    #[test]
    fn the_sheets_recurrent_sum_reads_each_active_points_row_before_its_column() {
        let k = MexicanHat::new(2.0, 1.0, 1.0, 2.0).unwrap();
        let (dt, tau, dx, h) = (1e-2, 1e-2, 1.5, -0.5);
        let mut sheet = Field2::new(k, h, tau, 24.0, 16).unwrap();
        assert_eq!(sheet.dx(), dx);
        sheet.u[1] = 1.0; // exactly one active point: column 1 of row 0.
        assert_eq!(sheet.step(dt).unwrap(), 1);
        let euler = dt / tau;
        let settle = |lateral: f64| h + euler * (-h + lateral * dx * dx + h);
        // Two columns along, same row: a separation of (0, 2).
        assert_eq!(sheet.u[3], settle(k.w(dx * (0.0f64).hypot(2.0))));
        // Three rows down, column 0: a separation of (3, 1), which is a different distance.
        assert_eq!(sheet.u[3 * 16], settle(k.w(dx * (3.0f64).hypot(1.0))));
        assert_ne!(sheet.u[3], sheet.u[3 * 16], "the two separations must differ for this to pin anything");
    }
}
