//! Intrinsic plasticity: a sigmoid neuron that tunes its own gain and bias until its firing rate is
//! as nearly exponentially distributed as two parameters allow.
//!
//! # The rule
//!
//! Triesch, *A gradient rule for the plasticity of a neuron's intrinsic excitability*, in Duch et
//! al. (eds.), ICANN 2005, Lecture Notes in Computer Science 3696:65–70, 2005
//! (`doi:10.1007/11550822_11`). A neuron turns its total synaptic current `x` into a firing rate `y`,
//! a fraction of its maximum, through a sigmoid whose gain `a` and bias `b` are its intrinsic
//! excitability, and after every presented input it moves both:
//!
//! ```text
//! y  = 1/(1 + exp(−(a x + b)))                              (5)
//! Δa = η (1/a + x − (2 + 1/µ) x y + (1/µ) x y²)             (12)
//! Δb = η (1 − (2 + 1/µ) y + (1/µ) y²)                       (13)
//! ```
//!
//! [`Sigmoid::rate`] is eq. 5, [`Triesch::update`] is eqs. 12 and 13, and [`Triesch::learn`]
//! applies them, `a := a + Δa` and `b := b + Δb` (p. 67). The rule reads only `x`, `y` and the
//! neuron's own gain `a`: it is local. P. 67 names only `x` and `y`; eq. 12's `1/a` is the neuron's own
//! parameter.
//!
//! # What it descends
//!
//! The rule is derived (pp. 66–67) as stochastic gradient descent on the Kullback–Leibler divergence
//! of the firing-rate density `f_y` from an exponential density of mean `µ`:
//!
//! ```text
//! D = ∫ f_y(y) log( f_y(y) / ((1/µ) exp(−y/µ)) ) dy        (2)
//!   = −H(y) + E(y)/µ + log µ                               (4)
//! ```
//!
//! Minimising `D` maximises the entropy of the rate while keeping its mean low. The exponential is
//! the most entropic density of a non-negative variable with a given mean (p. 65), which is why it
//! is the target. With `f_y = f_x/(dy/dx)` (eq. 1) and `log(dy/dx) = log a + log y + log(1 − y)`
//! (eq. 9), the gradient is
//!
//! ```text
//! ∂D/∂a = −1/a + E(−x + (2 + 1/µ) x y − (1/µ) x y²)        (8)
//! ∂D/∂b = −1 + E((2 + 1/µ) y − (1/µ) y²)                   (11)
//! ```
//!
//! — [`Triesch::gradient`] — and eqs. 12 and 13 are `−η` times the part of it one sample carries.
//! [`Triesch::objective`] computes `D` by eq. 4 with the expectations taken by quadrature over the
//! input ([`Input::expect`]); [`Triesch::mean_update`] is the rule averaged over the input, the
//! mean-field rule; and [`Triesch::fixed_point`] is where that stands still, found by Newton's
//! method on eqs. 8 and 11.
//!
//! # What is exact, and what the tests check
//!
//! - **The rule is the stochastic gradient of `D`.** The averaged rule divided by `−η` equals a
//!   central difference of `D` for the three inputs of the paper's Fig. 1, to the difference's own
//!   truncation; the gradient equals `SciPy`'s adaptive quadrature of eqs. 8 and 11; and `D` by
//!   eq. 4 equals `SciPy`'s integral of eq. 2 taken in `y` itself.
//! - **One identity ties the two lines**: `Δa = η/a + x Δb` exactly, and [`Triesch::update`]
//!   computes eq. 12 through it. It agrees with eq. 12 expanded as printed to rounding, and it stays
//!   finite where the expanded terms `(2 + 1/µ) x y` and `x y²/µ` would each overflow and leave
//!   `∞ − ∞` (at `x = 2 × 10³⁰⁷` the printed form gives NaN). Dropping the `1/µ` terms leaves
//!   exactly Bell and Sejnowski's rule for a single logistic unit, `Δa = η(1/a + x(1 − 2y))`,
//!   `Δb = η(1 − 2y)`, entropy maximisation alone; p. 67 calls the full rule "very similar" to it,
//!   with "additional terms" that keep the mean rate low.
//! - **The fixed points** for `N(0, 1)`, `U[0, 1]` and exponential inputs of mean 0.1, at
//!   `µ = 0.1`, are `(a, b) = (1.2383, −2.7024)`, `(4.2363, −4.8666)` and `(12.2595, −3.7365)`, the
//!   values `SciPy`'s root finder gives. Each is a minimum of `D`. The mean rate there is 0.1028,
//!   0.0989 and 0.1172, not `µ`: two parameters trade the mean against the entropy, and the rule
//!   does not hold the mean at its target.
//! - **A run of the rule settles `O(η)` from the fixed point, not on it.** Constant-step stochastic
//!   gradient descent fluctuates about the fixed point with covariance `ηΣ₁ + O(η²)`, where
//!   `JΣ₁ + Σ₁Jᵀ + C = 0`, `J` is the Jacobian of the averaged rule over `η` (minus the Hessian of
//!   `D`) and `C` the covariance of one input's update over `η`; and because the averaged rule
//!   `F = −∇D` is curved, the fluctuation moves the mean, to `(a*, b*) + ηδ + O(η²)` with
//!   `δ = −½ J⁻¹ (∂²F : Σ₁)`. For `N(0, 1)`, `U[0, 1]` and the exponential, `δ` is
//!   `(+0.474, −0.255)`, `(+0.249, −0.286)` and `(+1.089, −0.269)`, and the stationary standard
//!   deviation of `a` at `η = 0.001` is 0.0231, 0.0171 and 0.0334; the tests compute both by
//!   quadrature and match `SciPy`'s. Runs agree: sixteen seeds per `η` in the tests land within
//!   0.7 standard errors of `ηδ` at `η = 0.01` and 0.02. Out of the crate, seeds 1 to 480, each
//!   started at the fixed point and averaged over 2 × 10⁶ inputs after 2 × 10⁴, give
//!   `0.481η ± 0.008η`, `0.479η ± 0.003η` and `0.481η ± 0.002η` in `a` at `η = 0.004`, 0.01 and
//!   0.02, 0.9, 1.6 and 4.1 standard errors above `δ`, the last consistent with the `O(η²)`
//!   remainder the closed form leaves out; at the paper's `η = 0.001`, seeds 1 to 14,400 in three
//!   sets of 4,800 give `0.488η`, `0.470η` and `0.480η`, each `± 0.010η`. At that `η` the offset is
//!   `4.7 × 10⁻⁴` in `a`, `3.8 × 10⁻⁴` of `a*`.
//! - **An affine change of input moves the fixed point exactly.** If `x = s x′ + c`, the fixed point
//!   for `x′` is `(s a*, b* + c a*)`, so the tenfold narrowing of Fig. 2 sends `(a*, b*)` to
//!   `(10 a*, b*)`.
//! - **`D` never reaches zero.** The sigmoid's rate lives in `(0, 1)` and the target on `[0, ∞)`, so
//!   `D ≥ −log(1 − e^{−1/µ})`, minus the log of the target's mass on `[0, 1]`: `4.54 × 10⁻⁵` at
//!   `µ = 0.1`, where it is close to the mass above one, `e^{−10} = 4.54 × 10⁻⁵`; at `µ = 1` the
//!   floor is 0.459 and the mass above one 0.368.
//! - **`D` is convex in `(a, b)` exactly when `µ ≥ ½`.** At the paper's `µ = 0.1` it is not, so
//!   [`Triesch::fixed_point`] descends on `D` itself rather than trusting every Newton step. Its
//!   Hessian can be indefinite there but, whatever the input, never negative definite; at
//!   `µ = 0.01` it can be.
//! - **The optimal transfer function** `−µ log(1 − F_x(x))`, [`Triesch::optimal_transfer`], makes
//!   the rate exactly exponential with mean `µ`. It is the curve the paper's text describes and
//!   Figs. 1c and 1d draw; Fig. 1b's dotted curve differs from it (below).
//!
//! # References run
//!
//! No code accompanies the paper, and this review did not locate an implementation by its author.
//! The tests run two independent ones. reservoirpy 0.4.2's `IPReservoir`, whose sigmoid rule cites
//! this paper, was run unmodified (one unit, its internal state set to the input) on the seeded
//! input streams this crate draws, reproduced bit for bit in Python: after 1, 10, …, 10⁵ inputs its
//! `(a, b)` agrees with [`Triesch::learn`]'s within one unit in the last place, 2.1 × 10⁻¹⁶
//! relative, for all three inputs, and after every one of the 10⁵ inputs within three units,
//! 4.3 × 10⁻¹⁶ (fed the crate's own draws, one input at a time); its
//! `delta_a = 1/a + delta_b x` is the identity [`Triesch::update`] computes eq. 12 by. `SciPy`
//! 1.13.1's adaptive quadrature and root finder, over each input's whole support, give the fixed
//! points, gradients, objectives and Fig. 1a's nullclines and trajectories that the tests compare
//! against.
//!
//! # The paper against its own figures
//!
//! Fig. 1 (p. 68) prints `µ = 0.1` and `η = 0.001` and names its inputs "gaussian, uniform, and
//! exponential" without their parameters. `N(0, 1)`, `U[0, 1]` and an exponential of mean 0.1 are
//! read from the figure's dashed density curves ([`Input::FIG1_GAUSSIAN`] and its siblings); the
//! figure's coordinates quoted here are readings of the PDF's vector paths (`pdftocairo -svg`),
//! mapped through each panel's tick marks, and where a reading moves with that calibration the range
//! is quoted, from a least-squares fit to the grid lines to the panel's frame. The averaged rule for
//! `N(0, 1)` draws Fig. 1a: its nullclines run within 0.023 of every drawn vertex, 27 on the
//! `b`-nullcline and 18 on the `a`-nullcline; its two trajectories bulge within 0.0020 of the drawn
//! ones; and its fixed point is 0.0021 in `a` and 0.0037 in `b` from the circle at
//! `(1.2363, −2.7062)` (0.0020 and 0.0036 through the frame). The misses are the figure's: the
//! caption calls the nullclines "approximate locations (found numerically)". The drawn
//! `b`-nullcline takes values on a grid of 0.02 in `b` (it repeats at `a = 0.95` and 1.0, 1.30 and
//! 1.35, 1.40 and 1.45, 1.55 and 1.60, 1.65 and 1.70), a step as large as its largest miss, 0.021 at
//! `a = 1.55`; the `a`-nullcline's values lie on a grid of 0.004 in `a`, and its largest miss, 0.022
//! at `b = −0.5`, is more than five of those steps. Fig. 1c's dotted curve is `−0.1 log(1 − x)`:
//! its 89 vertices sit on steps of 0.01 in `x` up to 0.99, each within 6.7 × 10⁻⁴ of its step, and
//! at those steps within 3.7 × 10⁻⁴ of the formula; taken where the calibration puts them, where
//! the curve is steep, they are within 0.0012 to 0.0014 of it up to `x = 0.97`. Fig. 1d's is the
//! identity, within 2.2 × 10⁻⁴ to 4.2 × 10⁻⁴ at each of its 96 vertices, from a fit to the grid
//! lines to the two outermost grid lines alone: each is `−µ log(1 − F_x)` for its input.
//!
//! ⚠ **Fig. 1b's dotted "optimal transfer fct." is not `−µ log(1 − Φ(x))`.** The formula lies above
//! the drawn curve wherever either is visibly off zero, by at least 0.0026 from `x = −1` up: 0.0693
//! against 0.0619 at `x = 0` and 1.036 against 0.927 at `x = 4`. It passes 1, the maximum rate, at
//! `x = 3.914`, where the drawn curve is at 0.909. Panels c and d follow the formula, and the paper
//! prints none for any panel. Up to `x = 2.5` every one of the drawn curve's 396 vertices lies within
//! `4.1 × 10⁻⁴` to `6.3 × 10⁻⁴` of the formula shifted right by 0.1, as the calibration moves from
//! a fit to the tick marks or grid lines, through the frame, to the two outermost grid lines alone,
//! the largest at `x = 2.017` in each, and a least-squares shift over `x ∈ [−2, 2.5]` is 0.0985 to
//! 0.0998. From `x = 3` the drawn curve falls below the shifted formula as well, by at least 0.0013,
//! so this review has no formula for the whole curve.
//!
//! ⚠ **Fig. 1d matches an unconverged run, about 70 stationary standard deviations from the fixed
//! point.** Its sigmoid fits `(9.95, −3.49)` by least squares on the rate (`a` from 9.860 to 9.954
//! over fits to the rate or its logit and over calibrations); the fixed point is
//! `(12.2595, −3.7365)`, and the rule's stationary spread in `a` there has a standard deviation of
//! 0.0334 at `η = 0.001` by the closed form above (single runs of 2 × 10⁶ inputs from it, seeds 1
//! to 100, measure 0.019 to 0.044), so every reading is 69 to 72 standard deviations below the
//! fixed point, and stationary noise cannot put the curve where it is. Runs of 10⁵ inputs from
//! `(1, 0)` land on the drawn curve (six seeds at `a = 9.92` to 10.00, and reservoirpy's run at
//! 9.94), and 5 × 10⁵ bring `a` within 0.08 of the fixed point: the slowest mode relaxes over
//! `1/(ηλ) ≈ 1.2 × 10⁵` inputs. The paper prints no start and no run length; `(1, 0)` is an
//! assumption, and that the panel shows a finite run is an inference.
//!
//! ⚠ **Fig. 1b's learned curve fits `a` between 1.2143 and 1.2197, 0.0186 to 0.0240 below the
//! fixed point.** Least squares on the rate gives 1.2187 to 1.2196 over calibrations; least squares
//! on the logit, which weights the tails where a hundredth of a point on the page is a quarter of
//! the rate, gives 1.2143 to 1.2197 as fewer of the tail's vertices are kept. The rule's stationary
//! standard deviation in `a` at `η = 0.001` is 0.0231, so every reading is within 1.04 standard
//! deviations of the fixed point, and the rate fits within 0.85: consistent with one draw of a
//! stochastic run, though the paper does not say what panel b plots.
//!
//! ⚠ **The constant's sign.** Eq. 4 prints `+ log µ`; the paragraph after it speaks of "the constant
//! `− log µ`". That matches the term inside eq. 3's second integrand, `−y/µ − log µ`, rather than
//! eq. 4's constant, so it is ambiguous wording rather than a certain slip. Eq. 2 integrated
//! directly gives `+ log µ`, as eq. 4 has it; the two readings differ by `2 log 10 = 4.6` at
//! `µ = 0.1`, and the constant does not reach the rule. One smaller slip: p. 67 asks for `g`
//! "differentiable with respect to `y`", where eq. 1 needs `dy/dx`. The logarithm's base is not
//! printed; eq. 8's `1/a` needs it natural.
//!
//! ⚠ **Fig. 2's deprivation experiment prints its protocol and none of its parameters.** The input's
//! standard deviation falls tenfold at input 10,000, every 20th rate is plotted, and the run is
//! 5 × 10⁴ inputs long (p. 69); the input distribution, `µ`, `η` and the start are not printed.
//! Under `N(0, 1)` narrowing to `N(0, 0.1²)`, Fig. 1's `µ` and `η`, and a start at the fixed point,
//! the rule reproduces the figure's shape, a collapse of variability and then a slow regrowth, but
//! not its pace. The figure's rate spread in the eight windows of 5,000 inputs after the switch,
//! read on the frame, is 0.028, 0.041, 0.058, 0.058, 0.071, 0.075, 0.073 and 0.085. Its path has
//! 2,508 vertices where every 20th of 5 × 10⁴ inputs makes 2,500, so the window a vertex falls in
//! depends on the reading of the time axis, and the first window ranges over 0.027 to 0.029 and the
//! last over 0.084 to 0.085. Over 400 seeds the first window averages 0.0215 with a standard
//! deviation of 0.0012 and a largest value of 0.0248, the figure 4.5 to 6.0 standard deviations
//! above the mean; the last averages 0.066 (0.005, at most 0.082); and the figure lies above every
//! seed in six of the eight windows (seven in one reading). The gain reaches 7.745 to 7.795 by input
//! 50,000, far short of the 12.38 the new fixed point asks. With `η = 0.002` and nothing else
//! changed, the figure falls inside the simulated spread in every window, with 33% to 93% of the
//! seeds at or above it, and the gain reaches 9.65 to 9.76: the mismatch is consistent with Fig. 2
//! using twice Fig. 1's learning rate.
//!
//! # Refusals
//!
//! A gain `a ≤ 0` is refused by name ([`IntrinsicError::Gain`]): eq. 12 divides by it, eq. 9 takes
//! its logarithm, and the derivation needs the sigmoid strictly increasing. A step of the rule that
//! would carry `a` there is refused as well ([`IntrinsicError::Overshoot`]) rather than clamped; the
//! paper prints no safeguard. Every other parameter that is not finite, a scale that is not
//! positive and an empty interval are refused, naming what was sent. A result that `f64` cannot
//! hold, from parameters each finite and in range — an update, `D` or a derivative of it, or the
//! bias after a step, come out infinite or NaN — is refused by name as well
//! ([`IntrinsicError::Unrepresentable`]) rather than returned: a gain below
//! `1/f64::MAX ≈ 5.56 × 10⁻³⁰⁹` makes `1/a` infinite, a learning rate of `10³⁰⁸` makes `Δb` so,
//! and a gain of `10³⁰⁸` makes `D` so.
//!
//! # Units
//!
//! None are printed. `x` and `y` are dimensionless, `y` a fraction of the maximum rate (Fig. 2's
//! axis), and time is the number of presented inputs.

use core::fmt;

use crate::meanfield::erfcx;
use crate::rng::Rng;

/// Why a question about the neuron, its input or its rule could not be answered.
#[derive(Debug, Clone, PartialEq)]
pub enum IntrinsicError {
    /// A gain `a` that is not finite and positive. Eq. 12 divides by it, eq. 9 takes its
    /// logarithm, and the derivation needs the transfer function strictly increasing, which is
    /// `a > 0`.
    Gain {
        /// The gain.
        a: f64,
    },
    /// A parameter that must be finite and positive was not.
    NotPositive {
        /// Which.
        what: &'static str,
        /// Its value.
        value: f64,
    },
    /// A value that must be finite was not.
    NonFinite {
        /// Which.
        what: &'static str,
        /// Its value.
        value: f64,
    },
    /// A uniform input whose interval is empty or reversed.
    EmptyInterval {
        /// The lower end.
        lo: f64,
        /// The upper end.
        hi: f64,
    },
    /// A quadrature asked for with no panels, or with more than [`Input::MAX_PANELS`].
    Panels {
        /// The number asked for.
        panels: usize,
    },
    /// One step of the rule would carry the gain out of `(0, ∞)`, where the rule is undefined.
    Overshoot {
        /// The gain before the step.
        a: f64,
        /// Where the step would put it.
        next: f64,
    },
    /// A result that `f64` cannot hold: an update, a derivative of `D`, `D` itself or the bias after
    /// a step came out infinite or NaN, because the rule's parameters, the neuron or the input are
    /// too extreme for the arithmetic, although each is finite and in range.
    Unrepresentable {
        /// Which result.
        what: &'static str,
        /// What it came out as.
        value: f64,
    },
    /// Newton's method did not reach a stationary point of the objective.
    NoFixedPoint {
        /// Newton steps taken.
        iterations: usize,
        /// The length of the gradient of `D` where it stopped.
        residual: f64,
    },
}

impl fmt::Display for IntrinsicError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Gain { a } => {
                write!(f, "gain a = {a} must be finite and positive: eq. 12 divides by it and eq. 9 takes its logarithm")
            }
            Self::NotPositive { what, value } => write!(f, "{what} = {value} must be finite and positive"),
            Self::NonFinite { what, value } => write!(f, "{what} = {value} is not finite"),
            Self::EmptyInterval { lo, hi } => write!(f, "the uniform input's interval [{lo}, {hi}] is empty; it needs lo < hi"),
            Self::Panels { panels } => {
                write!(f, "a quadrature needs from 1 to {} panels, not {panels}", Input::MAX_PANELS)
            }
            Self::Overshoot { a, next } => write!(
                f,
                "one step of eq. 12 takes the gain from {a} to {next}, where the rule is undefined; the learning rate is too large for this input"
            ),
            Self::Unrepresentable { what, value } => write!(
                f,
                "{what} comes out as {value}, which f64 cannot hold: the rule's parameters, the neuron or the input are too extreme for the arithmetic"
            ),
            Self::NoFixedPoint { iterations, residual } => write!(
                f,
                "Newton's method found no stationary point: after {iterations} steps the gradient of D is still {residual}"
            ),
        }
    }
}

impl std::error::Error for IntrinsicError {}

fn finite(what: &'static str, value: f64) -> Result<f64, IntrinsicError> {
    if value.is_finite() { Ok(value) } else { Err(IntrinsicError::NonFinite { what, value }) }
}

fn positive(what: &'static str, value: f64) -> Result<f64, IntrinsicError> {
    if value.is_finite() && value > 0.0 { Ok(value) } else { Err(IntrinsicError::NotPositive { what, value }) }
}

fn gain(a: f64) -> Result<f64, IntrinsicError> {
    if a.is_finite() && a > 0.0 { Ok(a) } else { Err(IntrinsicError::Gain { a }) }
}

/// A computed result, refused by name where it is not finite.
fn held(what: &'static str, value: f64) -> Result<f64, IntrinsicError> {
    if value.is_finite() { Ok(value) } else { Err(IntrinsicError::Unrepresentable { what, value }) }
}

/// `ln(1 + eᵗ)`, without overflow for large `t` or loss of digits for very negative `t`.
fn softplus(t: f64) -> f64 {
    t.max(0.0) + (-t.abs()).exp().ln_1p()
}

/// The neuron of eq. 5: `y = 1/(1 + exp(−(ax + b)))`, a firing rate as a fraction of the maximum.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sigmoid {
    /// The gain `a`, per unit of input. Positive: the rule is derived for a strictly increasing
    /// transfer function and divides by `a`.
    pub a: f64,
    /// The bias `b`, dimensionless.
    pub b: f64,
}

impl Sigmoid {
    /// A neuron with gain `a` and bias `b`.
    ///
    /// # Errors
    ///
    /// [`IntrinsicError::Gain`] for an `a` that is not finite and positive;
    /// [`IntrinsicError::NonFinite`] for a `b` that is not finite.
    pub fn new(a: f64, b: f64) -> Result<Self, IntrinsicError> {
        Ok(Self { a: gain(a)?, b: finite("b", b)? })
    }

    /// The same checks as [`Sigmoid::new`], for a value whose public fields were set directly.
    ///
    /// # Errors
    ///
    /// As [`Sigmoid::new`].
    pub fn check(&self) -> Result<(), IntrinsicError> {
        gain(self.a)?;
        finite("b", self.b)?;
        Ok(())
    }

    /// Eq. 5, the firing rate for total synaptic current `x`, in `[0, 1]`.
    ///
    /// It is exactly `1` once `e^{−(ax + b)}` is no more than half a unit in the last place of 1,
    /// from `ax + b = 53 ln 2 ≈ 36.74` up, and exactly `0` once `e^{−(ax + b)}` overflows, below
    /// `ax + b ≈ −709.78`; [`Sigmoid::slope`] is then exactly `0` as well. Between the two it is
    /// strictly inside `(0, 1)`.
    #[must_use]
    pub fn rate(&self, x: f64) -> f64 {
        1.0 / (1.0 + (-(self.a * x + self.b)).exp())
    }

    /// `dy/dx = a y (1 − y)`: the slope eq. 1 divides the input density by, and whose logarithm
    /// eq. 9 splits into `log a + log y + log(1 − y)`.
    #[must_use]
    pub fn slope(&self, x: f64) -> f64 {
        let y = self.rate(x);
        self.a * y * (1.0 - y)
    }
}

/// A distribution of the total synaptic current `x`: the three kinds the paper's Fig. 1 uses.
///
/// The paper names them, "gaussian, uniform, and exponential" (p. 68), and prints none of their
/// parameters. [`Input::FIG1_GAUSSIAN`], [`Input::FIG1_UNIFORM`] and [`Input::FIG1_EXPONENTIAL`]
/// are read from the figure's own dashed density curves.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Input {
    /// Normal with mean `mean` and standard deviation `sd`.
    Gaussian {
        /// The mean.
        mean: f64,
        /// The standard deviation, positive.
        sd: f64,
    },
    /// Uniform on `[lo, hi]`.
    Uniform {
        /// The lower end.
        lo: f64,
        /// The upper end, above `lo`.
        hi: f64,
    },
    /// Exponential on `[0, ∞)` with mean `mean`.
    Exponential {
        /// The mean, positive.
        mean: f64,
    },
}

impl Input {
    /// Composite Simpson panels [`Input::expect`] uses.
    pub const PANELS: usize = 4096;

    /// The most panels [`Input::expect_with`] accepts: `2²⁰`, two million evaluations, and well
    /// short of the count at which the node count `2 · panels` would wrap.
    pub const MAX_PANELS: usize = 1 << 20;

    /// Fig. 1a,b: `N(0, 1)`, from the dashed curve's `e^{−x²/2}` shape.
    pub const FIG1_GAUSSIAN: Self = Self::Gaussian { mean: 0.0, sd: 1.0 };

    /// Fig. 1c: `U[0, 1]`, from the dashed box's edges.
    pub const FIG1_UNIFORM: Self = Self::Uniform { lo: 0.0, hi: 1.0 };

    /// Fig. 1d: exponential with mean 0.1, from the dashed curve's `e^{−x/0.1}` shape.
    pub const FIG1_EXPONENTIAL: Self = Self::Exponential { mean: 0.1 };

    /// Every parameter finite, the scales positive and the interval non-empty.
    ///
    /// # Errors
    ///
    /// [`IntrinsicError::NonFinite`] for a mean or an end that is not finite;
    /// [`IntrinsicError::NotPositive`] for a standard deviation or exponential mean that is not
    /// finite and positive; [`IntrinsicError::EmptyInterval`] for `lo ≥ hi`.
    pub fn check(&self) -> Result<(), IntrinsicError> {
        match *self {
            Self::Gaussian { mean, sd } => {
                finite("mean", mean)?;
                positive("sd", sd)?;
            }
            Self::Uniform { lo, hi } => {
                finite("lo", lo)?;
                finite("hi", hi)?;
                if lo >= hi {
                    return Err(IntrinsicError::EmptyInterval { lo, hi });
                }
            }
            Self::Exponential { mean } => {
                positive("mean", mean)?;
            }
        }
        Ok(())
    }

    /// The probability density `f_x(x)`, zero outside the support.
    #[must_use]
    pub fn density(&self, x: f64) -> f64 {
        match *self {
            Self::Gaussian { mean, sd } => {
                let z = (x - mean) / sd;
                (-0.5 * z * z).exp() / (sd * core::f64::consts::TAU.sqrt())
            }
            Self::Uniform { lo, hi } => {
                if lo <= x && x <= hi {
                    1.0 / (hi - lo)
                } else {
                    0.0
                }
            }
            Self::Exponential { mean } => {
                if x >= 0.0 {
                    (-x / mean).exp() / mean
                } else {
                    0.0
                }
            }
        }
    }

    /// `ln(1 − F_x(x))`, the logarithm of the probability that the input exceeds `x`.
    ///
    /// Accurate in the far upper tail, where `1 − F` itself underflows: the Gaussian's is
    /// `ln(½ erfcx(z/√2)) − z²/2` with `z = (x − mean)/sd`, through [`crate::meanfield::erfcx`], so
    /// `z = 40` gives `−804.6…` rather than `ln 0`. `−∞` at and above a uniform input's upper end.
    #[must_use]
    pub fn ln_survival(&self, x: f64) -> f64 {
        match *self {
            Self::Gaussian { mean, sd } => {
                let z = (x - mean) / sd;
                let t = z / core::f64::consts::SQRT_2;
                if z >= 0.0 {
                    (0.5 * erfcx(t)).ln() - 0.5 * z * z
                } else {
                    (-0.5 * erfcx(-t) * (-0.5 * z * z).exp()).ln_1p()
                }
            }
            Self::Uniform { lo, hi } => {
                if x <= lo {
                    0.0
                } else if x >= hi {
                    f64::NEG_INFINITY
                } else {
                    ((hi - x) / (hi - lo)).ln()
                }
            }
            Self::Exponential { mean } => {
                if x <= 0.0 {
                    0.0
                } else {
                    -x / mean
                }
            }
        }
    }

    /// The differential entropy `H(x) = −E[ln f_x(x)]`, in nats, in closed form:
    /// `½ ln(2πe σ²)`, `ln(hi − lo)` and `1 + ln(mean)`.
    #[must_use]
    pub fn entropy(&self) -> f64 {
        match *self {
            Self::Gaussian { sd, .. } => 0.5 * (1.0 + core::f64::consts::TAU.ln()) + sd.ln(),
            Self::Uniform { lo, hi } => (hi - lo).ln(),
            Self::Exponential { mean } => 1.0 + mean.ln(),
        }
    }

    /// The interval [`Input::expect_with`] integrates over: the whole support where it is finite,
    /// ten standard deviations either side of a Gaussian's mean, and forty means of an exponential.
    #[must_use]
    pub fn domain(&self) -> (f64, f64) {
        match *self {
            Self::Gaussian { mean, sd } => (mean - 10.0 * sd, mean + 10.0 * sd),
            Self::Uniform { lo, hi } => (lo, hi),
            Self::Exponential { mean } => (0.0, 40.0 * mean),
        }
    }

    /// `E[f(x)]` by composite Simpson's rule with `panels` panels over [`Input::domain`].
    ///
    /// # Errors
    ///
    /// [`IntrinsicError::Panels`] for zero panels or more than [`Input::MAX_PANELS`]; whatever
    /// [`Input::check`] refuses.
    pub fn expect_with(&self, panels: usize, f: impl Fn(f64) -> f64) -> Result<f64, IntrinsicError> {
        self.check()?;
        if panels == 0 || panels > Self::MAX_PANELS {
            return Err(IntrinsicError::Panels { panels });
        }
        let (lo, hi) = self.domain();
        let n = 2 * panels;
        let h = (hi - lo) / n as f64;
        let g = |x: f64| f(x) * self.density(x);
        let mut sum = g(lo) + g(hi);
        for i in 1..n {
            let w = if i.is_multiple_of(2) { 2.0 } else { 4.0 };
            sum += w * g(lo + h * i as f64);
        }
        Ok(sum * h / 3.0)
    }

    /// `E[f(x)]` with [`Input::PANELS`] panels.
    ///
    /// # Errors
    ///
    /// Whatever [`Input::check`] refuses.
    pub fn expect(&self, f: impl Fn(f64) -> f64) -> Result<f64, IntrinsicError> {
        self.expect_with(Self::PANELS, f)
    }

    /// One draw, from the crate's seeded generator: Box–Muller's cosine branch for the Gaussian,
    /// `lo + (hi − lo)u` for the uniform and `−mean · ln(1 − u)` for the exponential.
    ///
    /// # Errors
    ///
    /// Whatever [`Input::check`] refuses.
    pub fn sample(&self, rng: &mut Rng) -> Result<f64, IntrinsicError> {
        self.check()?;
        Ok(match *self {
            Self::Gaussian { mean, sd } => {
                let u = 1.0 - rng.next_f64();
                let v = rng.next_f64();
                mean + sd * (-2.0 * u.ln()).sqrt() * (core::f64::consts::TAU * v).cos()
            }
            Self::Uniform { lo, hi } => lo + (hi - lo) * rng.next_f64(),
            Self::Exponential { mean } => -mean * (1.0 - rng.next_f64()).ln(),
        })
    }
}

/// A stationary point of the averaged rule, and how Newton's method reached it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FixedPoint {
    /// The neuron at which both expected updates vanish.
    pub sigmoid: Sigmoid,
    /// Newton steps taken from the start.
    pub iterations: usize,
}

/// Triesch's rule, eqs. 12 and 13, with its target mean rate `µ` and learning rate `η`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Triesch {
    /// The desired mean firing rate `µ`, as a fraction of the maximum.
    pub mu: f64,
    /// The learning rate `η`.
    pub eta: f64,
}

impl Triesch {
    /// The Fig. 1 caption's "Parameters were `µ = 0.1`, `η = 0.001`" (p. 68).
    pub const FIG1: Self = Self { mu: 0.1, eta: 0.001 };

    /// Newton steps [`Triesch::fixed_point`] takes before it refuses.
    pub const NEWTON_STEPS: usize = 100;

    /// The gradient length at which [`Triesch::fixed_point`] stops.
    pub const NEWTON_TOLERANCE: f64 = 1e-12;

    /// Halvings of one Newton step before [`Triesch::fixed_point`] gives up on lowering `D` with it.
    pub const HALVINGS: usize = 60;

    /// A rule with target mean `mu` and learning rate `eta`.
    ///
    /// # Errors
    ///
    /// [`IntrinsicError::NotPositive`] for either that is not finite and positive.
    pub fn new(mu: f64, eta: f64) -> Result<Self, IntrinsicError> {
        Ok(Self { mu: positive("mu", mu)?, eta: positive("eta", eta)? })
    }

    /// The same checks as [`Triesch::new`], for a value whose public fields were set directly.
    ///
    /// # Errors
    ///
    /// As [`Triesch::new`].
    pub fn check(&self) -> Result<(), IntrinsicError> {
        positive("mu", self.mu)?;
        positive("eta", self.eta)?;
        Ok(())
    }

    /// `2 + 1/µ`, the factor eqs. 8, 11, 12 and 13 share.
    fn k(&self) -> f64 {
        2.0 + 1.0 / self.mu
    }

    /// Eq. 13, `Δb = η(1 − (2 + 1/µ) y + y²/µ)`, then eq. 12 through the identity that ties the two
    /// lines, `Δa = η/a + x Δb`. That is eq. 12 to rounding, and it stays finite where eq. 12's
    /// expanded terms `(2 + 1/µ) x y` and `x y²/µ` would each overflow and leave `∞ − ∞`.
    fn raw_update(&self, s: Sigmoid, x: f64) -> (f64, f64) {
        let y = s.rate(x);
        let db = self.eta * (1.0 - self.k() * y + y * y / self.mu);
        (self.eta / s.a + x * db, db)
    }

    fn advance(s: Sigmoid, (da, db): (f64, f64)) -> Result<Sigmoid, IntrinsicError> {
        let next = Sigmoid { a: s.a + da, b: s.b + db };
        if !(next.a.is_finite() && next.a > 0.0) {
            return Err(IntrinsicError::Overshoot { a: s.a, next: next.a });
        }
        held("the bias after the step", next.b)?;
        Ok(next)
    }

    /// `(Δa, Δb)` for one presented input `x`, eqs. 12 and 13: `Δb` as printed and `Δa` through
    /// `Δa = η/a + x Δb`, which equals eq. 12 as printed to rounding.
    ///
    /// # Errors
    ///
    /// [`IntrinsicError::Gain`] for a gain that is not finite and positive;
    /// [`IntrinsicError::NonFinite`] for a bias or input that is not finite;
    /// [`IntrinsicError::Unrepresentable`] for a `Δa` or `Δb` that comes out infinite or NaN;
    /// whatever [`Triesch::check`] refuses.
    pub fn update(&self, s: Sigmoid, x: f64) -> Result<(f64, f64), IntrinsicError> {
        self.check()?;
        s.check()?;
        finite("x", x)?;
        let (da, db) = self.raw_update(s, x);
        let db = held("Δb", db)?;
        Ok((held("Δa", da)?, db))
    }

    /// The neuron after one presented input: `a := a + Δa`, `b := b + Δb`.
    ///
    /// # Errors
    ///
    /// [`IntrinsicError::Overshoot`] where the step would leave the gain not finite and positive;
    /// [`IntrinsicError::Unrepresentable`] where it would leave the bias not finite; whatever
    /// [`Triesch::update`] refuses.
    pub fn learn(&self, s: Sigmoid, x: f64) -> Result<Sigmoid, IntrinsicError> {
        Self::advance(s, self.update(s, x)?)
    }

    /// `(E[Δa], E[Δb])` over the input: eqs. 12 and 13 averaged by [`Input::expect`].
    ///
    /// # Errors
    ///
    /// [`IntrinsicError::Unrepresentable`] for an average that comes out infinite or NaN; whatever
    /// [`Triesch::check`], [`Sigmoid::check`] and [`Input::check`] refuse.
    pub fn mean_update(&self, s: Sigmoid, input: &Input) -> Result<(f64, f64), IntrinsicError> {
        self.check()?;
        s.check()?;
        let da = input.expect(|x| self.raw_update(s, x).0)?;
        let db = input.expect(|x| self.raw_update(s, x).1)?;
        let db = held("E[Δb]", db)?;
        Ok((held("E[Δa]", da)?, db))
    }

    /// One step of the averaged rule: the neuron moved by [`Triesch::mean_update`].
    ///
    /// # Errors
    ///
    /// [`IntrinsicError::Overshoot`] where the step would leave the gain not finite and positive;
    /// [`IntrinsicError::Unrepresentable`] where it would leave the bias not finite; whatever
    /// [`Triesch::mean_update`] refuses.
    pub fn mean_step(&self, s: Sigmoid, input: &Input) -> Result<Sigmoid, IntrinsicError> {
        Self::advance(s, self.mean_update(s, input)?)
    }

    /// The objective `D`, eq. 4: `−H(y) + E(y)/µ + ln µ`, the Kullback–Leibler divergence of the
    /// output density from the exponential of mean `µ`.
    ///
    /// `H(y) = H(x) + E[ln(dy/dx)]` by eq. 1 and `ln(dy/dx) = ln a + ln y + ln(1 − y)` by eq. 9, so
    /// `D = −H(x) − ln a + E[softplus(−u) + softplus(u) + y/µ] + ln µ` with `u = ax + b`.
    ///
    /// # Errors
    ///
    /// [`IntrinsicError::Unrepresentable`] for a `D` that comes out infinite or NaN, as it does
    /// where `ax` overflows; whatever [`Triesch::check`], [`Sigmoid::check`] and [`Input::check`]
    /// refuse.
    pub fn objective(&self, s: Sigmoid, input: &Input) -> Result<f64, IntrinsicError> {
        self.check()?;
        s.check()?;
        let e = input.expect(|x| {
            let u = s.a * x + s.b;
            softplus(-u) + softplus(u) + s.rate(x) / self.mu
        })?;
        held("D", e - input.entropy() - s.a.ln() + self.mu.ln())
    }

    /// `(∂D/∂a, ∂D/∂b)`, eqs. 8 and 11 as printed.
    ///
    /// # Errors
    ///
    /// [`IntrinsicError::Unrepresentable`] for a derivative that comes out infinite or NaN, as
    /// `1/a` does for a gain below `1/f64::MAX ≈ 5.56 × 10⁻³⁰⁹`; whatever [`Triesch::check`],
    /// [`Sigmoid::check`] and [`Input::check`] refuse.
    pub fn gradient(&self, s: Sigmoid, input: &Input) -> Result<(f64, f64), IntrinsicError> {
        self.check()?;
        s.check()?;
        let ga = -1.0 / s.a
            + input.expect(|x| {
                let y = s.rate(x);
                -x + self.k() * x * y - x * y * y / self.mu
            })?;
        let gb = -1.0
            + input.expect(|x| {
                let y = s.rate(x);
                self.k() * y - y * y / self.mu
            })?;
        let gb = held("∂D/∂b", gb)?;
        Ok((held("∂D/∂a", ga)?, gb))
    }

    /// The Hessian of `D`: `[[1/a² + E[x²c], E[xc]], [E[xc], E[c]]]` with
    /// `c = y(1 − y)(2 + 1/µ − 2y/µ)`, the derivatives of eqs. 8 and 11.
    ///
    /// # Errors
    ///
    /// [`IntrinsicError::Unrepresentable`] for an entry that comes out infinite or NaN, as `1/a²`
    /// does for a gain below about `7.5 × 10⁻¹⁵⁵`; whatever [`Triesch::check`], [`Sigmoid::check`] and
    /// [`Input::check`] refuse.
    pub fn hessian(&self, s: Sigmoid, input: &Input) -> Result<[[f64; 2]; 2], IntrinsicError> {
        self.check()?;
        s.check()?;
        let c = |x: f64| {
            let y = s.rate(x);
            y * (1.0 - y) * (self.k() - 2.0 * y / self.mu)
        };
        let aa = 1.0 / (s.a * s.a) + input.expect(|x| x * x * c(x))?;
        let ab = input.expect(|x| x * c(x))?;
        let bb = input.expect(c)?;
        let ab = held("∂²D/∂a∂b", ab)?;
        Ok([[held("∂²D/∂a²", aa)?, ab], [ab, held("∂²D/∂b²", bb)?]])
    }

    /// The neuron at which the averaged rule stands still, by Newton's method on eqs. 8 and 11 from
    /// `start`, globalised on `D` itself.
    ///
    /// Each step goes along the Newton direction `−H⁻¹∇D` where the Hessian is positive definite
    /// and along `−∇D` where it is not, halved until `D` falls; a trial that would carry the gain
    /// out of `(0, ∞)` is not taken. Near the minimum the fall a step would buy drops below the
    /// rounding of `D` itself, and no halving can show it; there a full Newton step is taken
    /// instead if it keeps the gain positive and shortens the gradient.
    ///
    /// # Errors
    ///
    /// [`IntrinsicError::NoFixedPoint`] if no step can be accepted, or [`Triesch::NEWTON_STEPS`]
    /// steps do not bring the gradient under [`Triesch::NEWTON_TOLERANCE`]; whatever
    /// [`Triesch::gradient`], [`Triesch::hessian`] and [`Triesch::objective`] refuse, among them a
    /// `D` that `f64` cannot hold at the start, as at a gain of `10³⁰⁸`.
    pub fn fixed_point(&self, start: Sigmoid, input: &Input) -> Result<FixedPoint, IntrinsicError> {
        let mut s = start;
        let mut g = self.gradient(s, input)?;
        let mut d = self.objective(s, input)?;
        for iterations in 0..Self::NEWTON_STEPS {
            let norm = g.0.hypot(g.1);
            if norm <= Self::NEWTON_TOLERANCE {
                return Ok(FixedPoint { sigmoid: s, iterations });
            }
            let [[haa, hab], [_, hbb]] = self.hessian(s, input)?;
            let det = haa * hbb - hab * hab;
            let newton = haa > 0.0 && det > 0.0;
            let step = if newton { ((hab * g.1 - hbb * g.0) / det, (hab * g.0 - haa * g.1) / det) } else { (-g.0, -g.1) };
            let mut accepted = None;
            let mut t = 1.0;
            for _ in 0..Self::HALVINGS {
                let trial = Sigmoid { a: s.a + t * step.0, b: s.b + t * step.1 };
                if trial.check().is_ok() && self.objective(trial, input)? < d {
                    accepted = Some(trial);
                    break;
                }
                t *= 0.5;
            }
            if accepted.is_none() && newton {
                let trial = Sigmoid { a: s.a + step.0, b: s.b + step.1 };
                if trial.check().is_ok() {
                    let gt = self.gradient(trial, input)?;
                    if gt.0.hypot(gt.1) < norm {
                        accepted = Some(trial);
                    }
                }
            }
            let Some(next) = accepted else {
                return Err(IntrinsicError::NoFixedPoint { iterations, residual: norm });
            };
            s = next;
            d = self.objective(s, input)?;
            g = self.gradient(s, input)?;
        }
        Err(IntrinsicError::NoFixedPoint { iterations: Self::NEWTON_STEPS, residual: g.0.hypot(g.1) })
    }

    /// The transfer function that would make the output EXACTLY exponential with mean `µ`:
    /// `−µ ln(1 − F_x(x))`, through [`Input::ln_survival`].
    ///
    /// # Errors
    ///
    /// [`IntrinsicError::NonFinite`] for an `x` that is not finite; whatever [`Triesch::check`] and
    /// [`Input::check`] refuse.
    pub fn optimal_transfer(&self, input: &Input, x: f64) -> Result<f64, IntrinsicError> {
        self.check()?;
        input.check()?;
        finite("x", x)?;
        Ok(-self.mu * input.ln_survival(x))
    }
}

#[cfg(test)]
mod tests {
    use super::{FixedPoint, Input, IntrinsicError, Sigmoid, Triesch};
    use crate::rng::Rng;

    /// The reference: `SciPy` 1.13.1, `scipy.integrate.quad` over each input's whole support and
    /// `scipy.optimize.root(method="hybr")` on eqs. 8 and 11 at `µ = 0.1`, tolerance `10⁻¹⁴`. Each
    /// row is `(input, a*, b*, E[y] at the fixed point, D there by eq. 2 integrated in y)`.
    const SCIPY: [(Input, f64, f64, f64, f64); 3] = [
        (Input::FIG1_GAUSSIAN, 1.238334674487386, -2.7024451622097443, 0.10277452896224934, 0.0330027331675882),
        (Input::FIG1_UNIFORM, 4.236307470878487, -4.866636643428572, 0.09894847924975304, 0.2121029183692919),
        (Input::FIG1_EXPONENTIAL, 12.259508378489462, -3.7365050759862433, 0.11715267979561032, 0.5101412483214258),
    ];

    fn neuron(a: f64, b: f64) -> Sigmoid {
        Sigmoid::new(a, b).unwrap()
    }

    fn fixed(r: &Triesch, input: &Input) -> FixedPoint {
        r.fixed_point(neuron(1.0, 0.0), input).unwrap()
    }

    /// `ln σ(u)` written the long way round, so that it shares no text with the module's softplus.
    fn ln_logistic(u: f64) -> f64 {
        if u > 0.0 { -((-u).exp().ln_1p()) } else { u - u.exp().ln_1p() }
    }

    /// The rule's stationary statistics to first order in `η`, the closed form for constant-step
    /// stochastic gradient descent, at the fixed point `at`: `(Σ₁, δ)`, with `Σ₁` as
    /// `[s_aa, s_ab, s_bb]`, such that a long run's covariance is `ηΣ₁ + O(η²)` and its mean
    /// `(a*, b*) + ηδ + O(η²)`.
    ///
    /// With `F = −∇D` the averaged rule over `η`, `J = −H` its Jacobian ([`Triesch::hessian`]) and
    /// `C` the covariance of one input's update over `η`, `Σ₁` solves `JΣ₁ + Σ₁Jᵀ + C = 0`; and
    /// stationarity, `E[F(θ)] = 0`, expanded to second order gives `δ = −½ J⁻¹ (∂²F : Σ₁)`. The
    /// second derivatives come from `∂y/∂u = y(1 − y)`: with
    /// `q(y) = [(2/µ) y(1 − y) + (2y/µ − 2 − 1/µ)(1 − 2y)] y(1 − y)`, `∂²F_b` is
    /// `[[E x²q, E xq], [E xq, E q]]` and `∂²F_a` is `[[2/a³ + E x³q, E x²q], [E x²q, E xq]]`.
    fn first_order(r: &Triesch, input: &Input, at: Sigmoid) -> ([f64; 3], [f64; 2]) {
        let (mu, a) = (r.mu, at.a);
        let e = |f: &dyn Fn(f64, f64) -> f64| input.expect(|x| f(x, at.rate(x))).unwrap();
        let h = |y: f64| 1.0 - (2.0 + 1.0 / mu) * y + y * y / mu;
        let ua = |x: f64, y: f64| 1.0 / a + x * h(y);
        let [[haa, hab], [_, hbb]] = r.hessian(at, input).unwrap();
        let (jaa, jab, jbb) = (-haa, -hab, -hbb);
        let (fa, fb) = (e(&|x, y| ua(x, y)), e(&|_, y| h(y)));
        let caa = e(&|x, y| ua(x, y) * ua(x, y)) - fa * fa;
        let cab = e(&|x, y| ua(x, y) * h(y)) - fa * fb;
        let cbb = e(&|_, y| h(y) * h(y)) - fb * fb;
        // JΣ + ΣJᵀ = −C: `jaa s_aa + jab s_ab = −caa/2`, `jab s_ab + jbb s_bb = −cbb/2`, and
        // `jab s_aa + (jaa + jbb) s_ab + jab s_bb = −cab`, solved for `s_ab` first.
        let s_ab = (-cab + jab * caa / (2.0 * jaa) + jab * cbb / (2.0 * jbb)) / (jaa + jbb - jab * jab / jaa - jab * jab / jbb);
        let s_aa = (-caa / 2.0 - jab * s_ab) / jaa;
        let s_bb = (-cbb / 2.0 - jab * s_ab) / jbb;
        let q = |y: f64| ((2.0 / mu) * y * (1.0 - y) + (2.0 * y / mu - 2.0 - 1.0 / mu) * (1.0 - 2.0 * y)) * y * (1.0 - y);
        let m = |p: i32| e(&|x, y| x.powi(p) * q(y));
        let (q0, q1, q2, q3) = (m(0), m(1), m(2), m(3));
        let va = (2.0 / (a * a * a) + q3) * s_aa + 2.0 * q2 * s_ab + q1 * s_bb;
        let vb = q2 * s_aa + 2.0 * q1 * s_ab + q0 * s_bb;
        let det = jaa * jbb - jab * jab;
        ([s_aa, s_ab, s_bb], [-0.5 * (jbb * va - jab * vb) / det, -0.5 * (jaa * vb - jab * va) / det])
    }

    /// Eq. 5 at points where it is exact in binary, its slope against a central difference
    /// (measured within 6.4 × 10⁻¹¹), and where it saturates, without a NaN.
    ///
    /// `1 + e^{−u}` rounds to 1 once `e^{−u}` is at most half a unit in the last place of 1,
    /// `2⁻⁵³`, which is `u ≥ 53 ln 2 = 36.7368`; and `e^{−u}` overflows once `−u` exceeds
    /// `ln(f64::MAX) = 709.7827`. So `u = 36.5` is short of 1 and `36.75` is 1, and `u = −709.75`
    /// is a subnormal above 0 and `−709.8125` is 0 — binary fractions either side of each edge.
    #[test]
    fn the_neuron_is_eq_5() {
        let unit = neuron(1.0, 0.0);
        assert!(unit.rate(36.5) < 1.0 && unit.slope(36.5) > 0.0);
        assert_eq!((unit.rate(36.75), unit.slope(36.75)), (1.0, 0.0));
        assert!(unit.rate(-709.75) > 0.0 && unit.slope(-709.75) > 0.0);
        assert_eq!((unit.rate(-709.8125), unit.slope(-709.8125)), (0.0, 0.0));
        assert!((36.5..36.75).contains(&(53.0 * core::f64::consts::LN_2)) && (709.75..709.8125).contains(&f64::MAX.ln()));
        let n = neuron(2.0, -1.0);
        assert_eq!(n.rate(0.5), 0.5, "ax + b = 0");
        assert_eq!(n.slope(0.5), 0.5, "the steepest slope is a/4");
        for x in [-3.0, -0.7, 0.2, 1.9] {
            let want = 1.0 / (1.0 + (-(2.0 * x - 1.0_f64)).exp());
            assert_eq!(n.rate(x), want);
            let h = 1e-6;
            let fd = (n.rate(x + h) - n.rate(x - h)) / (2.0 * h);
            assert!((n.slope(x) - fd).abs() < 3e-10, "x = {x}: {} against {fd}", n.slope(x));
            assert!(n.slope(x) < 0.5);
        }
        assert_eq!(neuron(1.0, 0.0).rate(-1000.0), 0.0);
        assert_eq!(neuron(1.0, 0.0).rate(1000.0), 1.0);
        assert_eq!(neuron(1.0, 0.0).slope(1000.0), 0.0);
        assert_eq!(neuron(3.0, 0.0).rate(0.0), 0.5, "the bias alone sets the rate at x = 0");
        assert!((neuron(1.0, 1.0).rate(0.0) - 0.731_058_578_630_004_9).abs() < 1e-16, "σ(1)");
    }

    /// Eqs. 12 and 13 at a point where every term is a binary fraction, and eq. 12 expanded as
    /// printed against the identity [`Triesch::update`] computes it by, `Δa = η/a + x·Δb`.
    ///
    /// With `µ = 1/8` the shared factor `2 + 1/µ` is 10; with `a = 2`, `b = −1`, `x = 1/2` the neuron
    /// sits at `ax + b = 0`, so `y = 1/2` exactly, and `Δa = ½(½ + ½ − 10·¼ + 8·⅛) = −¼`,
    /// `Δb = ½(1 − 5 + 2) = −1`. Over 123 inputs at three neurons the printed form and the identity
    /// agree within 3.9 × 10⁻¹⁶ of eq. 12's largest term (measured), and `Δb` is eq. 13 to the bit.
    /// At `x = 2 × 10³⁰⁷` the printed form's `(2 + 1/µ) x y` and `x y²/µ` both overflow and it gives
    /// NaN; the identity gives `Δa = 0.001 − 2 × 10³⁰⁴`, which [`Triesch::learn`] refuses as the
    /// overshoot it is.
    #[test]
    fn eqs_12_and_13_are_as_printed() {
        let r = Triesch::new(0.125, 0.5).unwrap();
        let n = neuron(2.0, -1.0);
        assert_eq!(r.update(n, 0.5).unwrap(), (-0.25, -1.0));
        assert_eq!(r.learn(n, 0.5).unwrap(), neuron(1.75, -2.0));
        // The paper's own parameters at y = 1/2: Δa = η(1/a) and Δb = η(1 − 12/2 + 10/4) = −2.5η.
        let (da, db) = Triesch::FIG1.update(neuron(1.0, 0.0), 0.0).unwrap();
        assert!((da - 0.001).abs() < 1e-18 && (db + 0.0025).abs() < 1e-18, "{da} {db}");
        let p = Triesch::FIG1;
        let twelve = 2.0 + 1.0 / p.mu;
        let printed = |a: f64, y: f64, x: f64| p.eta * (1.0 / a + x - twelve * x * y + x * y * y / p.mu);
        for (a, b) in [(0.7, -2.0), (1.3, 0.4), (9.0, -3.5)] {
            let n = neuron(a, b);
            for k in -20..=20 {
                let x = f64::from(k) / 5.0;
                let y = n.rate(x);
                let (da, db) = p.update(n, x).unwrap();
                assert_eq!(db, p.eta * (1.0 - twelve * y + y * y / p.mu), "a = {a}, x = {x}");
                let largest = p.eta * (1.0 / a).max(x.abs()).max(twelve * x.abs() * y).max(x.abs() * y * y / p.mu);
                assert!((da - printed(a, y, x)).abs() <= 1.5e-15 * largest, "a = {a}, b = {b}, x = {x}: {da} against {}", printed(a, y, x));
            }
        }
        assert!(printed(1.0, 1.0, 2e307).is_nan());
        let (da, db) = p.update(neuron(1.0, 0.0), 2e307).unwrap();
        assert_eq!(db, -0.001);
        assert_eq!(da, 0.001 + 2e307 * -0.001);
        let err = p.learn(neuron(1.0, 0.0), 2e307).unwrap_err();
        assert_eq!(err, IntrinsicError::Overshoot { a: 1.0, next: 1.0 + da });
        assert!(err.to_string().starts_with("one step of eq. 12 takes the gain from 1 to -2000"), "{err}");
    }

    /// Without the `1/µ` terms the rule is exactly Bell and Sejnowski's (1995) rule for a single
    /// logistic unit maximising its entropy, `Δa = η(1/a + x(1 − 2y))`, `Δb = η(1 − 2y)`, the
    /// paper's reference 10 — p. 67 calls the full rule "very similar" to it — to the last bit at
    /// `µ = 10³⁰⁰`, where `1/µ` is lost against every other term.
    #[test]
    fn without_the_mean_terms_it_is_bell_and_sejnowskis_rule() {
        let r = Triesch::new(1e300, 1.0).unwrap();
        for (a, b) in [(0.5, -1.0), (2.0, 0.3)] {
            let n = neuron(a, b);
            for x in [-2.5, -0.3, 0.0, 0.8, 3.1] {
                let y = n.rate(x);
                assert_eq!(r.update(n, x).unwrap(), (1.0 / a + x * (1.0 - 2.0 * y), 1.0 - 2.0 * y), "a = {a}, x = {x}");
            }
        }
    }

    /// reservoirpy 0.4.2's `IPReservoir` runs the same trajectory on the same inputs.
    ///
    /// `IPReservoir`'s sigmoid rule cites this paper and computes `delta_b = 1 − (2 + 1/µ)y + y²/µ`
    /// and `delta_a = 1/a + delta_b · x`, then steps by `η` times each — eqs. 13 and 12 written
    /// through the identity [`Triesch::update`] uses too, with `η` applied after the sum rather than
    /// before it, and its sigmoid as `exp(u)/(exp(u) + 1)` for negative `u`: it shares the paper and
    /// the identity with this module, and not its rounding. It was run unmodified with one unit,
    /// `W = 0`, `Win = 1`, zero bias and leak rate 1, so that its internal state is exactly the
    /// input, at `µ = 0.1` and learning rate 0.001, from its own start `(1, 0)`. The inputs are this
    /// crate's seed-2005 streams of Fig. 1's three inputs, reproduced in Python from the generator's
    /// integer arithmetic and [`Input::sample`]'s transforms; the first three draws of each are
    /// below. Measured: the draws identical, and every checkpoint's `(a, b)` within 2.1 × 10⁻¹⁶
    /// relative, one unit in the last place; over all 10⁵ steps, run again on the crate's own draws,
    /// within 4.3 × 10⁻¹⁶, three units. (Eq. 12 computed in its expanded, printed form instead
    /// drifts to 4.2 × 10⁻¹⁶ at the checkpoints.)
    #[test]
    fn reservoirpys_ip_reservoir_runs_the_same_trajectory() {
        type Checkpoints = [(usize, f64, f64); 6];
        let rows: [(Input, [f64; 3], Checkpoints); 3] = [
            (
                Input::FIG1_GAUSSIAN,
                [-0.8717696942322773, -0.7474919873020599, 0.02264885237896574],
                [
                    (1, 1.002455032163198, -0.0016690556838859455),
                    (10, 1.0071491747387757, -0.020404169006321506),
                    (100, 1.0504316228778003, -0.20457652911300278),
                    (1000, 1.1369051521099434, -1.4564055901932131),
                    (10_000, 1.2350453849634733, -2.6877742100258333),
                    (100_000, 1.2561957898447926, -2.6846039298635613),
                ],
            ),
            (
                Input::FIG1_UNIFORM,
                [0.47563814811799554, 0.3891776819079418, 0.7174173217354051],
                [
                    (1, 0.9997646700813151, -0.0025972052989713095),
                    (10, 0.9973015968046645, -0.025583735546638536),
                    (100, 0.9729192368434407, -0.2552095511672447),
                    (1000, 1.0481084668741072, -1.8616068013949183),
                    (10_000, 2.947475321417038, -3.9138597211186026),
                    (100_000, 4.2164842529411075, -4.855950923978239),
                ],
            ),
            (
                Input::FIG1_EXPONENTIAL,
                [0.06455732759591608, 0.04929491671891881, 0.12637841047907603],
                [
                    (1, 1.0008366916210791, -0.0025296644858499983),
                    (10, 1.0077301898426678, -0.025319073608067635),
                    (100, 1.073042249133139, -0.24791429118641536),
                    (1000, 1.6029528896306078, -1.6471872543235746),
                    (10_000, 4.255886049622659, -2.7663092839124377),
                    (100_000, 9.943335946008208, -3.4685264614876776),
                ],
            ),
        ];
        let r = Triesch::FIG1;
        let close = |got: f64, want: f64| (got - want).abs() <= 1e-15 * want.abs();
        for (input, first, checkpoints) in rows {
            let mut rng = Rng::new(2005);
            let mut n = neuron(1.0, 0.0);
            let mut next = checkpoints.iter();
            let mut due = next.next();
            for k in 1..=100_000 {
                let x = input.sample(&mut rng).unwrap();
                if k <= 3 {
                    assert!(close(x, first[k - 1]), "{input:?} draw {k}: {x} against {}", first[k - 1]);
                }
                n = r.learn(n, x).unwrap();
                if let Some(&(at, a, b)) = due
                    && at == k
                {
                    assert!(close(n.a, a) && close(n.b, b), "{input:?} after {k}: {n:?} against reservoirpy's ({a}, {b})");
                    due = next.next();
                }
            }
            assert!(due.is_none(), "{input:?}: a checkpoint was never reached");
        }
    }

    /// The averaged rule is `−η∇D`: eqs. 8 and 11 against a central difference of eq. 4, and eqs. 12
    /// and 13 averaged against eqs. 8 and 11 — for Fig. 1's three inputs and a shifted Gaussian.
    ///
    /// Central differences with `h = 10⁻⁴` agree with eqs. 8 and 11 to 4.8 × 10⁻⁹ at worst (the
    /// `h²` truncation; `h = 10⁻³` gives 4.7 × 10⁻⁷). The averaged rule over `η` agrees with the
    /// gradient to 5.1 × 10⁻¹⁵ where the density integrates to one to rounding, and to
    /// 3.2 × 10⁻¹² for the exponential, whose density Simpson integrates to `1 + 3.15 × 10⁻¹²`:
    /// eq. 12 carries `1/a` and `1` inside the expectation, eqs. 8 and 11 outside it.
    #[test]
    fn the_averaged_rule_is_minus_eta_times_the_gradient_of_d() {
        let r = Triesch::FIG1;
        let cases = [
            (Input::FIG1_GAUSSIAN, 1.0, -2.0, 2e-14),
            (Input::FIG1_UNIFORM, 3.0, -4.0, 2e-14),
            (Input::FIG1_EXPONENTIAL, 8.0, -3.0, 1e-11),
            (Input::Gaussian { mean: 0.5, sd: 0.7 }, 2.0, 0.5, 2e-14),
        ];
        for (input, a, b, averaged) in cases {
            let n = neuron(a, b);
            let g = r.gradient(n, &input).unwrap();
            let d = |a: f64, b: f64| r.objective(neuron(a, b), &input).unwrap();
            let h = 1e-4;
            let fd = ((d(a + h, b) - d(a - h, b)) / (2.0 * h), (d(a, b + h) - d(a, b - h)) / (2.0 * h));
            assert!((fd.0 - g.0).abs() < 2e-8 && (fd.1 - g.1).abs() < 2e-8, "{input:?}: {fd:?} against {g:?}");
            let m = r.mean_update(n, &input).unwrap();
            assert!((m.0 / r.eta + g.0).abs() < averaged && (m.1 / r.eta + g.1).abs() < averaged, "{input:?}: {m:?} against {g:?}");
            let stepped = r.mean_step(n, &input).unwrap();
            assert_eq!(stepped, neuron(a + m.0, b + m.1));
        }
    }

    /// Eqs. 8 and 11 against `SciPy`'s adaptive quadrature of the same expressions over each
    /// input's whole support. Measured: 6.7 × 10⁻¹⁵ for the Gaussian and the uniform, 9.4 × 10⁻¹³
    /// for the exponential, whose edge at zero limits Simpson to fourth order.
    #[test]
    fn the_gradient_is_scipys() {
        let r = Triesch::FIG1;
        let rows = [
            (Input::FIG1_GAUSSIAN, 1.0, -2.0, -0.12536382782410782, 0.46850487322608836, 2e-14),
            (Input::FIG1_UNIFORM, 3.0, -4.0, -0.12712621385131978, 0.033258544426620684, 2e-14),
            (Input::FIG1_EXPONENTIAL, 8.0, -3.0, -0.036962728163208836, 0.18158364330811216, 3e-12),
        ];
        for (input, a, b, ga, gb, tol) in rows {
            let g = r.gradient(neuron(a, b), &input).unwrap();
            assert!((g.0 - ga).abs() < tol && (g.1 - gb).abs() < tol, "{input:?}: {g:?} against ({ga}, {gb})");
        }
    }

    /// The Hessian is the derivative of the gradient: each entry against a central difference of
    /// eqs. 8 and 11 with `h = 2 × 10⁻⁵`, measured within 3.9 × 10⁻¹⁰, including at a neuron where
    /// it is indefinite. (At `h = 10⁻⁴` the difference's own truncation is 1.0 × 10⁻⁸.)
    #[test]
    fn the_hessian_is_the_derivative_of_the_gradient() {
        let r = Triesch::FIG1;
        let cases = [
            (Input::FIG1_GAUSSIAN, 1.0, -2.0),
            (Input::FIG1_UNIFORM, 3.0, -4.0),
            (Input::FIG1_EXPONENTIAL, 8.0, -3.0),
            (Input::Gaussian { mean: 0.5, sd: 0.7 }, 2.0, 0.5),
        ];
        for (input, a, b) in cases {
            let hs = r.hessian(neuron(a, b), &input).unwrap();
            let g = |a: f64, b: f64| r.gradient(neuron(a, b), &input).unwrap();
            let h = 2e-5;
            let (pa, ma, pb, mb) = (g(a + h, b), g(a - h, b), g(a, b + h), g(a, b - h));
            let fd = [[(pa.0 - ma.0) / (2.0 * h), (pb.0 - mb.0) / (2.0 * h)], [(pa.1 - ma.1) / (2.0 * h), (pb.1 - mb.1) / (2.0 * h)]];
            for i in 0..2 {
                for j in 0..2 {
                    assert!((hs[i][j] - fd[i][j]).abs() < 2e-9, "{input:?} [{i}][{j}]: {hs:?} against {fd:?}");
                }
            }
        }
        let odd = r.hessian(neuron(2.0, 0.5), &Input::Gaussian { mean: 0.5, sd: 0.7 }).unwrap();
        assert!(odd[1][1] < 0.0 && odd[0][0] > 0.0, "indefinite: {odd:?}");
    }

    /// Eq. 4, as implemented, is eq. 2: against eq. 2 integrated here with `f_y` built from eq. 1,
    /// and against `SciPy`'s integral of eq. 2 in `y` itself.
    ///
    /// The second comparison is what fixes the constant's sign: eq. 4 prints `+ log µ`, the prose
    /// after it speaks of "the constant `− log µ`" (eq. 3's integrand term), and read as eq. 4's
    /// constant the two differ by `2 log 10 = 4.6` at `µ = 0.1`. Measured agreement
    /// with `SciPy`: within 2.3 × 10⁻¹⁴ for the Gaussian, 1.0 × 10⁻¹⁴ for the uniform and
    /// 2.3 × 10⁻¹¹ for the exponential; eq. 2 by eq. 1 here within 2.4 × 10⁻¹⁴, 1.0 × 10⁻¹⁴ and
    /// 2.9 × 10⁻¹² of eq. 4.
    #[test]
    fn eq_4_is_eq_2_integrated_directly() {
        let r = Triesch::FIG1;
        let rows = [
            (Input::FIG1_GAUSSIAN, 1.0, -2.0, 0.19857550323541978, 1e-13),
            (Input::FIG1_UNIFORM, 3.0, -4.0, 0.296118848628201, 5e-14),
            (Input::FIG1_EXPONENTIAL, 8.0, -3.0, 0.6372588078898075, 1e-10),
            (SCIPY[0].0, SCIPY[0].1, SCIPY[0].2, SCIPY[0].4, 1e-13),
            (SCIPY[1].0, SCIPY[1].1, SCIPY[1].2, SCIPY[1].4, 5e-14),
            (SCIPY[2].0, SCIPY[2].1, SCIPY[2].2, SCIPY[2].4, 1e-10),
        ];
        for (input, a, b, scipy, tol) in rows {
            let n = neuron(a, b);
            let d = r.objective(n, &input).unwrap();
            assert!((d - scipy).abs() < tol, "{input:?} ({a}, {b}): {d} against SciPy's {scipy}");
            let eq2 = input
                .expect(|x| {
                    let u = a * x + b;
                    let ln_fy = input.density(x).ln() - (a.ln() + ln_logistic(u) + ln_logistic(-u));
                    let ln_fexp = -n.rate(x) / r.mu - r.mu.ln();
                    ln_fy - ln_fexp
                })
                .unwrap();
            assert!((eq2 - d).abs() < tol, "{input:?} ({a}, {b}): eq. 2 by eq. 1 gives {eq2}, eq. 4 {d}");
            let prose = d - 2.0 * r.mu.ln();
            assert!((prose - scipy - 2.0 * 10.0_f64.ln()).abs() < 1e-10, "the prose's `− log µ` would be 4.6 away");
        }
    }

    /// `D` cannot reach zero: the output lives in `(0, 1)` and the target on `[0, ∞)`, so `D` is at
    /// least `−log Q([0, 1]) = −log(1 − e^{−1/µ})`, minus the log of the target's mass on the rate's
    /// range (Jensen: `D = −E log(q/f_y) ≥ −log ∫₀¹ q`). At the fixed points, for three values of
    /// `µ`, and on a grid of neurons. Only at small `µ` is the floor close to the target's mass above
    /// one, `e^{−1/µ}`: `4.5401 × 10⁻⁵` against `4.5400 × 10⁻⁵` at `µ = 0.1`, 0.4587 against 0.3679
    /// at `µ = 1`.
    #[test]
    fn d_is_at_least_minus_the_log_of_the_targets_mass_on_the_rate_range() {
        for mu in [0.1, 0.5, 1.0] {
            let r = Triesch::new(mu, 0.001).unwrap();
            let floor = -(1.0 - (-1.0 / mu).exp()).ln();
            for input in [Input::FIG1_GAUSSIAN, Input::FIG1_UNIFORM, Input::FIG1_EXPONENTIAL] {
                let at = fixed(&r, &input).sigmoid;
                let d = r.objective(at, &input).unwrap();
                assert!(d > floor, "µ = {mu}, {input:?}: D = {d} at the fixed point, floor {floor}");
                for (a, b) in [(0.5, -3.0), (2.0, 0.0), (6.0, -4.0)] {
                    let dd = r.objective(neuron(a, b), &input).unwrap();
                    assert!(dd > d && dd > floor, "µ = {mu}, {input:?}, ({a}, {b}): {dd} against {d}");
                }
            }
        }
        assert!((-(1.0 - (-10.0_f64).exp()).ln() - 4.540_096e-5).abs() < 1e-11, "the floor at µ = 0.1");
        assert!((-(1.0 - (-1.0_f64).exp()).ln() - 0.458_675).abs() < 1e-6 && ((-1.0_f64).exp() - 0.367_879).abs() < 1e-6, "µ = 1");
    }

    /// Newton's method finds the fixed points `SciPy` finds, and the ones the extraction of the paper
    /// reports; from `(1, 0)` it takes 10, 8 and 12 steps.
    ///
    /// Measured against `SciPy`: `a*` within 2.2 × 10⁻¹⁶, 2.8 × 10⁻¹⁴ and 1.1 × 10⁻¹⁰, `b*` within
    /// 7.1 × 10⁻¹⁵, 2.2 × 10⁻¹⁴ and 1.2 × 10⁻¹¹, and `E[y]` within 4.5 × 10⁻¹³; the exponential's
    /// is the quadrature's `10⁻¹²` divided by the Hessian's smaller eigenvalue, 0.0086. The
    /// extraction gives `(1.2383, −2.7024)`, `(4.2363, −4.8666)` and `(12.2595, −3.7365)` for the
    /// same computation. Fig. 1a draws its circle at `(1.2363, −2.7062)` (the centre of its vector
    /// path, through the tick marks at `a = 1` and 2): the Gaussian's fixed point is 0.0021 from it
    /// in `a` and 0.0037 in `b`.
    #[test]
    fn newton_finds_the_fixed_points_scipy_finds() {
        let r = Triesch::FIG1;
        let tol = [(1e-15, 3e-14), (1e-13, 1e-13), (5e-10, 5e-11)];
        let rounded = [(1.2383, -2.7024), (4.2363, -4.8666), (12.2595, -3.7365)];
        let steps = [10, 8, 12];
        for (k, (input, a, b, mean, _)) in SCIPY.iter().enumerate() {
            let fp = fixed(&r, input);
            let s = fp.sigmoid;
            assert_eq!(fp.iterations, steps[k], "{input:?}");
            assert!((s.a - a).abs() < tol[k].0 && (s.b - b).abs() < tol[k].1, "{input:?}: {s:?} against ({a}, {b})");
            let round4 = |v: f64| (v * 1e4).round() / 1e4;
            assert_eq!((round4(s.a), round4(s.b)), rounded[k], "{input:?}");
            let ey = input.expect(|x| s.rate(x)).unwrap();
            assert!((ey - mean).abs() < 2e-12, "{input:?}: E[y] = {ey} against {mean}");
            let (ga, gb) = r.gradient(s, input).unwrap();
            assert!(ga.hypot(gb) <= Triesch::NEWTON_TOLERANCE);
        }
        let g = fixed(&r, &Input::FIG1_GAUSSIAN).sigmoid;
        assert!((g.a - 1.2363).abs() < 0.004 && (g.b + 2.7062).abs() < 0.008, "Fig. 1a's circle: {g:?}");
        let e = fixed(&r, &Input::FIG1_EXPONENTIAL).sigmoid;
        assert_eq!(((e.a * 100.0).round() / 100.0, (e.b * 100.0).round() / 100.0), (12.26, -3.74));
    }

    /// From starts across and beyond Fig. 1a's window — saturated, nearly flat, far out in gain —
    /// Newton's method reaches the same fixed point, and from the fixed point itself it takes no
    /// step. Measured against `SciPy`: within 1.35 × 10⁻¹⁰ in `a` and 1.5 × 10⁻¹¹ in `b` at worst,
    /// the exponential's, in at most 47 steps.
    #[test]
    fn newton_converges_from_anywhere_in_the_window() {
        let r = Triesch::FIG1;
        for (input, a, b, _, _) in SCIPY {
            for (a0, b0) in [(0.2, 3.0), (5.0, -8.0), (30.0, 1.0), (0.05, -0.5), (12.0, -3.7), (1.6, 0.5)] {
                let fp = r.fixed_point(neuron(a0, b0), &input).unwrap();
                assert!((fp.sigmoid.a - a).abs() < 5e-10 && (fp.sigmoid.b - b).abs() < 5e-11, "{input:?} from ({a0}, {b0}): {fp:?}");
                assert!(fp.iterations < 60, "{input:?} from ({a0}, {b0}): {fp:?}");
            }
            let at = fixed(&r, &input);
            assert_eq!(r.fixed_point(at.sigmoid, &input).unwrap(), FixedPoint { sigmoid: at.sigmoid, iterations: 0 });
        }
    }

    /// The long descents still arrive. Deep in saturation, at `(1, 80)`, every input drives the
    /// neuron to one and each steepest-descent step lowers `b` by about one: Newton's method takes
    /// 93, 88 and 85 of its hundred steps for the three inputs (from `(1, 90)` the Gaussian's run
    /// out). From a gain of `10⁶` or `10⁸` the sigmoid is a step between Simpson's nodes, and the
    /// first Newton step is accepted only after 17 to 27 halvings; the descents take 14, 69 and 15
    /// steps, and 16, 28 and 24. Measured against `SciPy`'s fixed point: within 1.2 × 10⁻¹¹ in `a`
    /// and 9.3 × 10⁻¹² in `b` for the Gaussian and the uniform, 1.1 × 10⁻¹⁰ and 1.2 × 10⁻¹¹ for
    /// the exponential.
    #[test]
    fn the_long_descents_still_arrive() {
        let r = Triesch::FIG1;
        for ((input, a, b, _, _), tol) in SCIPY.into_iter().zip([5e-11, 5e-11, 5e-10]) {
            let fp = r.fixed_point(neuron(1.0, 80.0), &input).unwrap();
            assert!(fp.iterations > 80 && fp.iterations < Triesch::NEWTON_STEPS, "{input:?}: {fp:?}");
            assert!((fp.sigmoid.a - a).abs() < tol && (fp.sigmoid.b - b).abs() < 5e-11, "{input:?}: {fp:?}");
            for a0 in [1e6, 1e8] {
                let fp = r.fixed_point(neuron(a0, 0.0), &input).unwrap();
                assert!((fp.sigmoid.a - a).abs() < tol && (fp.sigmoid.b - b).abs() < 5e-11, "{input:?} from a = {a0}: {fp:?}");
            }
        }
        let err = r.fixed_point(neuron(1.0, 90.0), &Input::FIG1_GAUSSIAN).unwrap_err();
        assert!(matches!(err, IntrinsicError::NoFixedPoint { iterations: 100, .. }), "{err:?}");
    }

    /// Where no step can lower `D` Newton's method refuses at once, where `D` itself overflows it
    /// refuses naming `D`, and where the descent is too long it refuses after
    /// [`Triesch::NEWTON_STEPS`].
    ///
    /// At `a = 10³⁰⁸` the edges of the Gaussian's domain overflow `ax`, and `D` comes out infinite:
    /// refused by name before any step. From `a = 10¹⁸` the neuron is a step function between
    /// Simpson's nodes, the gradient is `(E|x|, 0) = (√(2/π), 0)` to Simpson's error on the step
    /// (measured 1.65 × 10⁻⁶), the Hessian's `a`-entry is `1/a²` alone, and the Newton step is
    /// `−0.798a²`: even 2⁻⁵⁹ of it, the last of [`Triesch::HALVINGS`] halvings, carries the gain below
    /// zero, and so does the full step, which is refused as well, so the method reports that it
    /// cannot descend rather than a gain the caller never sent. From `5 × 10¹⁷` 2⁻⁵⁹ of the step is
    /// the first to keep the gain positive, `5 × 10¹⁷ − 0.798 · 2.5 × 10³⁵ · 2⁻⁵⁹ ≈ 1.5 × 10¹⁷`, and
    /// the method converges in 24 steps; from `10¹⁷`, where 2⁻⁵⁷ fits, in 26. Deep in saturation at
    /// `b = 10⁶` the neuron fires at exactly one for every input, the Hessian is singular, and each
    /// steepest-descent step lowers `b` by about one: a hundred steps are not enough.
    #[test]
    fn newton_refuses_where_it_cannot_descend() {
        let r = Triesch::FIG1;
        let root = (2.0 / core::f64::consts::PI).sqrt();
        let err = r.fixed_point(neuron(1e308, 0.0), &Input::FIG1_GAUSSIAN).unwrap_err();
        assert_eq!(err, IntrinsicError::Unrepresentable { what: "D", value: f64::INFINITY });
        let err = r.fixed_point(neuron(1e18, 0.0), &Input::FIG1_GAUSSIAN).unwrap_err();
        let IntrinsicError::NoFixedPoint { iterations: 0, residual } = err else { panic!("{err:?}") };
        assert!((residual - root).abs() < 5e-6, "{residual}");
        assert!(err.to_string().starts_with("Newton's method found no stationary point: after 0 steps the gradient of D is still 0.79"), "{err}");
        let (g, h) = (r.gradient(neuron(5e17, 0.0), &Input::FIG1_GAUSSIAN).unwrap(), r.hessian(neuron(5e17, 0.0), &Input::FIG1_GAUSSIAN).unwrap());
        let step = -g.0 / h[0][0];
        assert!((step / 2.5e35 + 0.798).abs() < 1e-3 && 5e17 + step * 0.5_f64.powi(58) < 0.0 && 5e17 + step * 0.5_f64.powi(59) > 0.0, "{step}");
        let fp = r.fixed_point(neuron(5e17, 0.0), &Input::FIG1_GAUSSIAN).unwrap();
        assert_eq!(fp.iterations, 24);
        assert!((fp.sigmoid.a - SCIPY[0].1).abs() < 1e-14 && (fp.sigmoid.b - SCIPY[0].2).abs() < 5e-14, "{fp:?}");
        assert_eq!(r.fixed_point(neuron(1e17, 0.0), &Input::FIG1_GAUSSIAN).unwrap().iterations, 26);
        let err = r.fixed_point(neuron(1.0, 1e6), &Input::FIG1_GAUSSIAN).unwrap_err();
        let IntrinsicError::NoFixedPoint { iterations, residual } = err else { panic!("{err:?}") };
        assert_eq!(iterations, Triesch::NEWTON_STEPS);
        assert!(residual > 0.5, "{residual}");
        assert!(err.to_string().starts_with("Newton's method found no stationary point: after 100 steps"), "{err}");
    }

    /// The fixed points are minima of `D` — the Hessian positive definite — so the averaged rule,
    /// whose linearisation there is `I − ηH`, contracts onto them. The smaller eigenvalues are
    /// 0.499, 0.0601 and 0.0086; the exponential's slowest mode relaxes over `1/(ηλ) ≈ 1.2 × 10⁵`
    /// inputs.
    #[test]
    fn the_fixed_points_are_minima_and_attract_the_averaged_rule() {
        let r = Triesch::FIG1;
        let mut slowest = Vec::new();
        for (input, ..) in SCIPY {
            let at = fixed(&r, &input).sigmoid;
            let [[haa, hab], [_, hbb]] = r.hessian(at, &input).unwrap();
            let (tr, det) = (haa + hbb, haa * hbb - hab * hab);
            let low = 0.5 * (tr - (tr * tr - 4.0 * det).sqrt());
            assert!(haa > 0.0 && det > 0.0 && low > 0.0, "{input:?}");
            assert!(1.0 - r.eta * low < 1.0 && 1.0 - r.eta * (tr - low) > -1.0);
            slowest.push(low);
            // One averaged step from beside the fixed point moves towards it.
            let off = neuron(at.a + 0.01, at.b - 0.01);
            let next = r.mean_step(off, &input).unwrap();
            assert!((next.a - at.a).hypot(next.b - at.b) < (off.a - at.a).hypot(off.b - at.b), "{input:?}");
        }
        assert!((slowest[0] - 0.499).abs() < 5e-4 && (slowest[1] - 0.0601).abs() < 5e-5, "{slowest:?}");
        assert!((slowest[2] - 0.0086).abs() < 5e-5, "{slowest:?}");
        assert!((1.0 / (r.eta * slowest[2]) - 1.16e5).abs() < 1e3, "{slowest:?}");
    }

    /// If `x = s·x′ + c`, the neuron `(sa, b + ca)` sees the same `ax + b`, so the fixed point for the
    /// transformed input is the transformed fixed point — for Fig. 2's tenfold narrowing, exactly
    /// `(10a*, b*)`. Measured: at the transformed point the gradient is already under
    /// [`Triesch::NEWTON_TOLERANCE`], so Newton's method takes no step; from `(1, 0)` it lands within
    /// 4.9 × 10⁻¹⁴ of it in `a`, relatively, and 3.1 × 10⁻¹³ in `b`.
    #[test]
    fn the_fixed_point_moves_with_the_input() {
        let r = Triesch::FIG1;
        let g = fixed(&r, &Input::FIG1_GAUSSIAN).sigmoid;
        let u = fixed(&r, &Input::FIG1_UNIFORM).sigmoid;
        let e = fixed(&r, &Input::FIG1_EXPONENTIAL).sigmoid;
        let cases = [
            (Input::Gaussian { mean: 0.0, sd: 0.1 }, neuron(10.0 * g.a, g.b)),
            (Input::Gaussian { mean: 3.0, sd: 0.5 }, neuron(2.0 * g.a, g.b - 6.0 * g.a)),
            (Input::Uniform { lo: -1.0, hi: 1.0 }, neuron(0.5 * u.a, u.b + 0.5 * u.a)),
            (Input::Exponential { mean: 1.0 }, neuron(0.1 * e.a, e.b)),
        ];
        for (input, want) in cases {
            assert_eq!(r.fixed_point(want, &input).unwrap(), FixedPoint { sigmoid: want, iterations: 0 }, "{input:?}");
            let from_far = r.fixed_point(neuron(1.0, 0.0), &input).unwrap().sigmoid;
            assert!((from_far.a - want.a).abs() < 2e-13 * want.a && (from_far.b - want.b).abs() < 1e-12, "{input:?}: {from_far:?}");
        }
        let deprived = r.fixed_point(neuron(10.0, -2.7), &Input::Gaussian { mean: 0.0, sd: 0.1 }).unwrap().sigmoid;
        assert!((deprived.a - 12.383).abs() < 1e-3 && (deprived.b + 2.7024).abs() < 1e-4, "{deprived:?}");
    }

    /// `D` is convex in `(a, b)` exactly when `µ ≥ ½`: at `µ = ½` the curvature weight
    /// `c = y(1 − y)(2 + 1/µ − 2y/µ) = 4y(1 − y)²` is never negative and every Hessian on a grid is
    /// positive definite; just below, at `µ = 0.49`, a saturated neuron already has `∂²D/∂b² < 0`;
    /// and at the paper's `µ = 0.1` the Hessian at `(1, 0)` under the uniform input is indefinite.
    #[test]
    fn d_is_convex_exactly_when_mu_is_at_least_a_half() {
        let half = Triesch::new(0.5, 0.001).unwrap();
        for input in [Input::FIG1_GAUSSIAN, Input::FIG1_UNIFORM, Input::FIG1_EXPONENTIAL] {
            for (a, b) in [(0.3, -4.0), (1.0, 0.0), (1.0, 8.0), (5.0, 3.0), (20.0, -1.0), (2.0, 30.0)] {
                let [[haa, hab], [_, hbb]] = half.hessian(neuron(a, b), &input).unwrap();
                assert!(haa > 0.0 && haa * hbb - hab * hab > 0.0, "µ = ½, {input:?}, ({a}, {b})");
            }
        }
        let below = Triesch::new(0.49, 0.001).unwrap();
        assert!(below.hessian(neuron(1.0, 8.0), &Input::FIG1_GAUSSIAN).unwrap()[1][1] < 0.0);
        assert!(half.hessian(neuron(1.0, 8.0), &Input::FIG1_GAUSSIAN).unwrap()[1][1] > 0.0);
        let [[haa, hab], [_, hbb]] = Triesch::FIG1.hessian(neuron(1.0, 0.0), &Input::FIG1_UNIFORM).unwrap();
        assert!(haa * hbb - hab * hab < 0.0, "µ = 0.1 at (1, 0): indefinite");
    }

    /// Where the Hessian is negative definite the Newton direction climbs, and the method takes
    /// `−∇D` there instead. That cannot happen at the paper's `µ = 0.1`, for any input: along
    /// `v = (a, b − u*)` the curvature is `vᵀHv = 1 + E[(u − u*)² c(u)]` with `u = ax + b`, and with
    /// `u* = 2` the weight `(u − u*)² max(−c(u), 0)` never exceeds 0.558 (measured on a grid of step
    /// 10⁻⁴ over `u ∈ [−5, 80]`; the smallest such bound over `u*` is 0.548, at `u* = 1.99`), so
    /// `vᵀHv > 0.44` whatever the input. At `µ = 0.01` the same weight reaches 12.2, and each input
    /// has starts with a negative-definite Hessian: from `(0.5, 1.5)`, `(1.25, 0.75)` and `(4, 0.5)`
    /// the method reaches the minimum it reaches from `(1, 0)`, within 1.5 × 10⁻¹² (measured).
    #[test]
    fn where_the_hessian_is_negative_definite_the_method_descends_along_the_gradient() {
        let weight = |mu: f64, u: f64| {
            let y = 1.0 / (1.0 + (-u).exp());
            let c = y * (1.0 - y) * (2.0 + (1.0 - 2.0 * y) / mu);
            (u - 2.0) * (u - 2.0) * (-c).max(0.0)
        };
        let sup = |mu: f64| (0..=850_000).map(|i| weight(mu, -5.0 + 1e-4 * f64::from(i))).fold(0.0, f64::max);
        let (tenth, hundredth) = (sup(0.1), sup(0.01));
        assert!((tenth - 0.558).abs() < 5e-4 && hundredth > 12.0, "{tenth} {hundredth}");
        let r = Triesch::new(0.01, 0.001).unwrap();
        for (input, a0, b0) in [(Input::FIG1_GAUSSIAN, 0.5, 1.5), (Input::FIG1_UNIFORM, 1.25, 0.75), (Input::FIG1_EXPONENTIAL, 4.0, 0.5)] {
            let [[haa, hab], [_, hbb]] = r.hessian(neuron(a0, b0), &input).unwrap();
            assert!(haa < 0.0 && haa * hbb - hab * hab > 0.0, "{input:?} ({a0}, {b0}): not negative definite");
            let got = r.fixed_point(neuron(a0, b0), &input).unwrap().sigmoid;
            let want = fixed(&r, &input).sigmoid;
            assert!((got.a - want.a).abs() < 5e-12 && (got.b - want.b).abs() < 5e-12, "{input:?}: {got:?} against {want:?}");
            let [[haa, hab], [_, hbb]] = r.hessian(want, &input).unwrap();
            assert!(haa > 0.0 && haa * hbb - hab * hab > 0.0, "{input:?}: the fixed point is a minimum");
        }
    }

    /// Fig. 1a, read against the averaged rule for `N(0, 1)`: where its dotted nullclines run, and how
    /// far its two drawn trajectories bulge.
    ///
    /// Every vertex of the figure's dotted path, read from the PDF (`pdftocairo -svg`, p. 68) and
    /// mapped through the panel's tick marks at `a = 1` and 2 and `b = −4` and 0.5, whose labels
    /// the page's glyphs confirm: the `b`-nullcline has 27 vertices, each within 3 × 10⁻⁴ of
    /// `a = 0.70, 0.75, …, 2.00`, and the `a`-nullcline 17, each within 1.0 × 10⁻³ of
    /// `b = −4.00, −3.75, …, 0.00`, and an eighteenth where it leaves the window, `(2.0003, 0.2333)`.
    /// For each, the averaged rule's nullcline crosses the segment of half-length 0.025 through the
    /// vertex, in `b` for the first and in `a` for the second; `SciPy`'s `brentq` on the rule
    /// averaged by `quad` puts the largest misses at 0.0213 (`a = 1.55`) and 0.0222 (`b = −0.5`),
    /// or 0.0212 and 0.0223 with the panel's frame as the calibration, so at half-length 0.02 those
    /// two segments are not crossed. At six points, three on each nullcline, the rule's nullcline,
    /// found here by bisection, is `brentq`'s within 8.9 × 10⁻¹⁶ (measured). The two trajectories:
    /// trajectory 1 starts at `(1.6, 0.5)` and bulges to `a = 1.6538`; trajectory 2 enters at
    /// `(0.7, −3.3915)`, the window's left edge, and bulges to 1.2745: within 0.0020 of the averaged
    /// rule iterated 600 and 3,000 times, which is `SciPy`'s iteration within 6.7 × 10⁻¹⁶
    /// (measured). The paper prints neither start.
    #[test]
    fn the_averaged_rule_draws_fig_1a() {
        let r = Triesch::FIG1;
        let input = Input::FIG1_GAUSSIAN;
        let drawn_b = [
            -2.4591, -2.4797, -2.4992, -2.5199, -2.5394, -2.5795, -2.5795, -2.6197, -2.6392, -2.6599, -2.6793, -2.7195, -2.7390, -2.7390,
            -2.7997, -2.7997, -2.8191, -2.8788, -2.8788, -2.9190, -2.9190, -2.9396, -2.9799, -2.9993, -3.0394, -3.0589, -3.0796,
        ];
        let drawn_a = [
            (-4.0, 1.6839),
            (-3.75, 1.5921),
            (-3.5, 1.5080),
            (-3.25, 1.4079),
            (-3.0, 1.3320),
            (-2.75, 1.2640),
            (-2.5, 1.1880),
            (-2.25, 1.1320),
            (-2.0, 1.0879),
            (-1.75, 1.0641),
            (-1.5, 1.0441),
            (-1.25, 1.0599),
            (-1.0, 1.0960),
            (-0.75, 1.1842),
            (-0.5, 1.3320),
            (-0.25, 1.4959),
            (0.0, 1.7399),
            (0.2333, 2.0003),
        ];
        let db = |a: f64, b: f64| r.mean_update(neuron(a, b), &input).unwrap().1;
        let da = |a: f64, b: f64| r.mean_update(neuron(a, b), &input).unwrap().0;
        // E[Δb] falls through zero as b rises across its nullcline, and E[Δa] as a rises across its.
        let crosses_b = |a: f64, b: f64, half: f64| db(a, b - half) > 0.0 && db(a, b + half) < 0.0;
        let crosses_a = |a: f64, b: f64, half: f64| da(a - half, b) > 0.0 && da(a + half, b) < 0.0;
        for (i, &b) in drawn_b.iter().enumerate() {
            let a = 0.7 + 0.05 * i as f64;
            assert!(crosses_b(a, b, 0.025), "b-nullcline at a = {a}: not within 0.025 of the drawn {b}");
        }
        for &(b, a) in &drawn_a {
            assert!(crosses_a(a, b, 0.025), "a-nullcline at b = {b}: not within 0.025 of the drawn {a}");
        }
        assert!(!crosses_b(1.55, drawn_b[17], 0.02) && !crosses_a(drawn_a[14].1, -0.5, 0.02), "the two largest misses");
        // Bisection, sixty halvings of a bracket that is checked first.
        let root = |f: &dyn Fn(f64) -> f64, mut lo: f64, mut hi: f64| {
            assert!(f(lo).signum() != f(hi).signum(), "no bracket");
            for _ in 0..60 {
                let mid = 0.5 * (lo + hi);
                if f(mid).signum() == f(lo).signum() {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            0.5 * (lo + hi)
        };
        for (a, scipy) in [(0.7, -2.459823142109769), (1.55, -2.85753670272606), (2.0, -3.077942894444489)] {
            let b = root(&|b| db(a, b), -6.0, 1.0);
            assert!((b - scipy).abs() < 4e-15, "b-nullcline at a = {a}: {b} against {scipy}");
        }
        for (b, scipy) in [(-4.0, 1.6808231567727612), (-1.5, 1.0450801662704188), (0.233, 2.005277523271776)] {
            let a = root(&|a| da(a, b), 0.8, 3.0);
            assert!((a - scipy).abs() < 4e-15, "a-nullcline at b = {b}: {a} against {scipy}");
        }
        for ((a0, b0), steps, drawn, scipy) in [((1.6, 0.5), 600, 1.6538, 1.6557749305517255), ((0.7, -3.3915), 3000, 1.2745, 1.2743843152550378)] {
            let mut n = neuron(a0, b0);
            let mut top = n.a;
            for _ in 0..steps {
                n = r.mean_step(n, &input).unwrap();
                top = top.max(n.a);
            }
            assert!((top - scipy).abs() < 4e-15 && (top - drawn).abs() < 0.005, "from ({a0}, {b0}): {top}");
        }
    }

    /// A seeded run of the stochastic rule from `(1, 0)` settles on the averaged rule's fixed point
    /// within this test's sampling error, for all three inputs — an error too coarse to see the
    /// `O(η)` offset of the next test.
    ///
    /// Eight independent seeds per input; each run's time average after a burn-in is one sample, so
    /// the standard error is honest however slowly a run decorrelates — the exponential's slowest
    /// mode relaxes over about 10⁵ inputs, and blocks shorter than that inside one run gave standard
    /// errors four times too small. Measured: offsets up to 1.9 standard errors with these eight
    /// seeds, and 0.35 to 2.1 with sixteen; standard errors under 5.2 × 10⁻⁴ of the value. The
    /// Gaussian's standard error in `a` is 6.4 × 10⁻⁴, above the stationary offset `ηδ = 4.7 × 10⁻⁴`
    /// of the closed form (below), so this test cannot see it; it checks that the runs settle, not
    /// where to the last `η`.
    #[test]
    fn a_seeded_run_settles_on_the_fixed_point() {
        let r = Triesch::FIG1;
        for ((input, a, b, _, _), (burn, span)) in SCIPY.into_iter().zip([(100_000, 400_000), (300_000, 1_000_000), (1_000_000, 2_000_000)]) {
            let (mut sa, mut sb) = (Vec::new(), Vec::new());
            for seed in 0..8 {
                let mut rng = Rng::new(500 + seed);
                let mut n = neuron(1.0, 0.0);
                for _ in 0..burn {
                    n = r.learn(n, input.sample(&mut rng).unwrap()).unwrap();
                }
                let (mut ta, mut tb) = (0.0, 0.0);
                for _ in 0..span {
                    n = r.learn(n, input.sample(&mut rng).unwrap()).unwrap();
                    ta += n.a;
                    tb += n.b;
                }
                sa.push(ta / f64::from(span));
                sb.push(tb / f64::from(span));
            }
            for (v, want, what) in [(&sa, a, "a"), (&sb, b, "b")] {
                let m = v.iter().sum::<f64>() / v.len() as f64;
                let se = (v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / ((v.len() - 1) * v.len()) as f64).sqrt();
                assert!((m - want).abs() < 4.0 * se, "{input:?} {what}: {m} against {want} ± {se}");
                assert!(se < 2e-3 * want.abs(), "{input:?} {what}: the runs did not settle, se = {se}");
            }
        }
    }

    /// The stationary mean of a run sits `O(η)` from the fixed point, where the first-order closed
    /// form of [`first_order`] puts it: for `N(0, 1)` at `(a*, b*) + ηδ`, `δ = (+0.474, −0.255)`.
    ///
    /// The closed form against `SciPy`'s adaptive quadrature of the same expressions (and a finite
    /// difference of the averaged rule for `∂²F`, which agrees to 10⁻⁷), for Fig. 1's three
    /// inputs: `δ = (0.4742, −0.2551)`, `(0.2492, −0.2862)` and `(1.0886, −0.2692)`, and `Σ₁`'s
    /// `a`-entry 0.5345, 0.2939 and 1.1127, so that `a`'s stationary standard deviation at
    /// `η = 0.001` is 0.0231, 0.0171 and 0.0334. Measured agreement within 7.7 × 10⁻¹⁵ for the
    /// Gaussian and the uniform and 2.3 × 10⁻¹¹ for the exponential, whose edge limits Simpson.
    ///
    /// Then the runs: sixteen seeds per learning rate, each from the fixed point with 2 × 10⁴
    /// inputs of burn-in (the slowest mode relaxes over `1/(ηλ) = 200` inputs at `η = 0.01`) and
    /// 6.25 × 10⁵ averaged. Measured: `a` sits `+0.475η ± 0.031η` above `a*` at `η = 0.01` and
    /// `+0.477η ± 0.015η` at 0.02, 15 and 31 standard errors from zero and 0.02 and 0.2 from `δ`;
    /// `b` sits `−0.229η ± 0.040η` and `−0.244η ± 0.020η`, 5.7 and 12 standard errors below `b*`
    /// and 0.6 and 0.5 from `δ`; and the ratio of the offsets in `a` is 2.01 ± 0.15. Longer runs
    /// out of the crate, seeds 1 to 480 from the fixed point with 2 × 10⁴ inputs of burn-in and
    /// 2 × 10⁶ averaged, give `0.481η ± 0.008η`, `0.479η ± 0.003η` and `0.481η ± 0.002η` at
    /// `η = 0.004`, 0.01 and 0.02: 0.9, 1.6 and 4.1 standard errors above `δ = 0.4742`, the last
    /// consistent with the `O(η²)` remainder the closed form leaves out.
    #[test]
    fn the_stationary_mean_sits_order_eta_from_the_fixed_point() {
        let references = [
            (0.4741850490278908, -0.2550765305375389, [0.5344841680878853, 0.16062790437613236, 0.49819907104556477], 3e-14),
            (0.2491888101179917, -0.28622454868758146, [0.2938579925173856, 0.12246292343699683, 0.4612318037633809], 3e-14),
            (1.0885709380525364, -0.26923346814013893, [1.1126527283327168, 0.10646672310551433, 0.6546358918572149], 1e-10),
        ];
        for ((input, ..), (da, db, s, tol)) in SCIPY.into_iter().zip(references) {
            let (sigma, delta) = first_order(&Triesch::FIG1, &input, fixed(&Triesch::FIG1, &input).sigmoid);
            let worst = (delta[0] - da).abs().max((delta[1] - db).abs()).max((0..3).map(|i| (sigma[i] - s[i]).abs()).fold(0.0, f64::max));
            assert!(worst < tol, "{input:?}: δ = {delta:?}, Σ₁ = {sigma:?} against SciPy's ({da}, {db}), {s:?}: {worst}");
        }
        let input = Input::FIG1_GAUSSIAN;
        let at = fixed(&Triesch::FIG1, &input).sigmoid;
        let (_, delta) = first_order(&Triesch::FIG1, &input, at);
        let offsets = |eta: f64| {
            let r = Triesch::new(0.1, eta).unwrap();
            let (mut oa, mut ob) = (Vec::new(), Vec::new());
            for seed in 1..=16 {
                let mut rng = Rng::new(seed);
                let mut n = at;
                for _ in 0..20_000 {
                    n = r.learn(n, input.sample(&mut rng).unwrap()).unwrap();
                }
                let (mut ta, mut tb, span) = (0.0, 0.0, 625_000);
                for _ in 0..span {
                    n = r.learn(n, input.sample(&mut rng).unwrap()).unwrap();
                    ta += n.a;
                    tb += n.b;
                }
                oa.push(ta / f64::from(span) - at.a);
                ob.push(tb / f64::from(span) - at.b);
            }
            let mean_se = |v: &[f64]| {
                let m = v.iter().sum::<f64>() / v.len() as f64;
                (m, (v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / ((v.len() - 1) * v.len()) as f64).sqrt())
            };
            (mean_se(&oa), mean_se(&ob))
        };
        let ((a1, sa1), (b1, sb1)) = offsets(0.01);
        let ((a2, sa2), (b2, sb2)) = offsets(0.02);
        assert!(a1 > 8.0 * sa1 && a2 > 8.0 * sa2, "a above a*: {a1} ± {sa1}, {a2} ± {sa2}");
        assert!(b1 < -4.0 * sb1 && b2 < -4.0 * sb2, "b below b*: {b1} ± {sb1}, {b2} ± {sb2}");
        let ratio = a2 / a1;
        let se = ratio * ((sa1 / a1).powi(2) + (sa2 / a2).powi(2)).sqrt();
        assert!((ratio - 2.0).abs() < 4.0 * se, "linear in η: the offset grows {ratio} ± {se} times when η doubles");
        for (eta, (a, sa), (b, sb)) in [(0.01, (a1, sa1), (b1, sb1)), (0.02, (a2, sa2), (b2, sb2))] {
            assert!((a - eta * delta[0]).abs() < 4.0 * sa && (b - eta * delta[1]).abs() < 4.0 * sb, "η = {eta}: ({a}, {b}) against η δ = {delta:?}");
        }
    }

    /// ⚠ Fig. 1d matches an unconverged run, not the fixed point: from `(1, 0)`, 10⁵ inputs land on
    /// the drawn curve and 5 × 10⁵ come close to the fixed point, and the drawn curve is about 70
    /// stationary standard deviations from it.
    ///
    /// The drawn sigmoid fits `(9.95, −3.49)`, `a` from 9.860 to 9.954 over fits and calibrations;
    /// the fixed point is `(12.2595, −3.7365)`. Six seeds land at `a = 9.92` to `10.00`,
    /// `b = −3.49` to `−3.44` after 10⁵ inputs — reservoirpy's own run at `(9.943, −3.469)` — and at
    /// `a = 12.18` to `12.24` after 5 × 10⁵. The stationary standard deviation of `a` is
    /// `√(ηΣ₁) = 0.0334` by [`first_order`]; from the fixed point, seed 7's 2 × 10⁶ inputs measure
    /// 0.0328 about their own mean, 1.7% below it (seeds 1 to 100 give 0.019 to 0.044 over runs of
    /// that length, the slowest mode relaxing over 1.2 × 10⁵ inputs). So every reading of the drawn
    /// curve is 69 to 72 standard deviations below the fixed point. The start is an assumption: the
    /// paper prints none, and that the panel shows a finite run is an inference.
    #[test]
    fn fig_1d_matches_an_unconverged_run() {
        let r = Triesch::FIG1;
        let input = Input::FIG1_EXPONENTIAL;
        for seed in 1..=6 {
            let mut rng = Rng::new(seed);
            let mut n = neuron(1.0, 0.0);
            for _ in 0..100_000 {
                n = r.learn(n, input.sample(&mut rng).unwrap()).unwrap();
            }
            assert!((n.a - 9.95).abs() < 0.1 && (n.b + 3.49).abs() < 0.12, "seed {seed} at 10⁵: {n:?}");
            for _ in 0..400_000 {
                n = r.learn(n, input.sample(&mut rng).unwrap()).unwrap();
            }
            assert!(n.a > 12.1 && n.a < 12.3, "seed {seed} at 5 × 10⁵: {n:?}");
        }
        let at = fixed(&r, &input).sigmoid;
        let mut rng = Rng::new(7);
        let mut n = at;
        let (mut s1, mut s2, count) = (0.0, 0.0, 2_000_000);
        for _ in 0..count {
            n = r.learn(n, input.sample(&mut rng).unwrap()).unwrap();
            s1 += n.a;
            s2 += n.a * n.a;
        }
        let mean = s1 / f64::from(count);
        let sd = (s2 / f64::from(count) - mean * mean).sqrt();
        let spread = (r.eta * first_order(&r, &input, at).0[0]).sqrt();
        assert!((spread - 0.0334).abs() < 5e-5 && (sd - spread).abs() < 0.25 * spread, "the stationary spread of a: {sd} against {spread}");
        for fit in [9.860, 9.954] {
            let away = (at.a - fit) / spread;
            assert!(away > 69.0 && away < 73.0, "a fit of {fit}: {away} standard deviations");
        }
    }

    /// Fig. 1b's learned sigmoid fits `a` between 1.2143 and 1.2197, 0.0186 to 0.0240 below the
    /// fixed point: least squares on the rate over the path's 551 vertices gives 1.2187 to 1.2196
    /// as the axis calibration moves from the frame to the tick marks, and least squares on the
    /// logit, which weights the tails, 1.2143 to 1.2197 as the vertices kept run from those with
    /// rates between 0.001 and 0.999 to those between 0.05 and 0.95 (at a rate of 5 × 10⁻⁴ the two
    /// calibrations differ by a hundredth of a point, a quarter of the rate). The rule's stationary
    /// standard deviation in `a` at `η = 0.001` is `√(ηΣ₁) = 0.0231` by [`first_order`], and 10⁶
    /// inputs here measure 0.0227, 2.0% below it; every reading lies within 1.04 of those standard
    /// deviations of the fixed point, and the rate fits within 0.85 — consistent with one draw of a
    /// stochastic run.
    #[test]
    fn fig_1b_learned_curve_is_inside_the_stationary_spread() {
        let r = Triesch::FIG1;
        let input = Input::FIG1_GAUSSIAN;
        let at = fixed(&r, &input).sigmoid;
        let mut rng = Rng::new(68);
        let mut n = at;
        for _ in 0..100_000 {
            n = r.learn(n, input.sample(&mut rng).unwrap()).unwrap();
        }
        let (mut s1, mut s2, count) = (0.0, 0.0, 1_000_000);
        for _ in 0..count {
            n = r.learn(n, input.sample(&mut rng).unwrap()).unwrap();
            s1 += n.a;
            s2 += n.a * n.a;
        }
        let mean = s1 / f64::from(count);
        let sd = (s2 / f64::from(count) - mean * mean).sqrt();
        let spread = (r.eta * first_order(&r, &input, at).0[0]).sqrt();
        assert!((spread - 0.0231).abs() < 5e-5 && (sd - spread).abs() < 0.05 * spread, "the stationary spread of a: {sd} against {spread}");
        for (fit, within) in [(1.2187, 0.9), (1.2196, 0.9), (1.2143, 1.1), (1.2197, 1.1)] {
            let gap = at.a - fit;
            assert!(gap > 0.0 && gap < within * spread, "a fit of {fit}: {gap} below the fixed point against a spread of {spread}");
        }
    }

    /// ⚠ Fig. 2 under assumptions the paper does not print — `N(0, 1)` narrowed to `N(0, 0.1²)` at
    /// input 10,000, `µ = 0.1`, starting at the fixed point — reproduces its shape, and its pace with
    /// twice Fig. 1's learning rate, not with Fig. 1's own.
    ///
    /// The figure's rate spreads, read from its vector path (`pdftocairo -svg`, p. 69; every 20th
    /// rate, windows of 5,000 inputs after the switch) with the time axis calibrated on the frame:
    /// 0.0278, 0.0410, 0.0580, 0.0578, 0.0712, 0.0748, 0.0726 and 0.0846, and 0.1028 before it; a
    /// least-squares fit to the tick marks gives the same but 0.0720 and 0.0850 in the last two
    /// windows. The path has 2,508 vertices, not 2,500, so the input a vertex stands for depends on
    /// the reading: sorted in time and spread evenly over the axis, or numbered 20 inputs apart from
    /// the first input, from the twentieth or back from the last, they put the first window between
    /// 0.0270 and 0.0288 and the last between 0.0843 and 0.0850. The tests take the lowest of each,
    /// 0.027 and 0.0843, because they assert that the figure lies above the seeds there.
    ///
    /// Twenty-four seeds at `η = 0.001`: the first window averages 0.0215 with a standard deviation
    /// of 0.0012 and a largest value of 0.0240, so even 0.027 is 4.45 standard deviations above the
    /// mean and above every seed (400 seeds: 0.0215, 0.0012, at most 0.0248); the last window is at
    /// most 0.080. At `η = 0.002` the figure lies within 1.5 standard deviations of the seeds' mean in
    /// every window (measured −1.44 to +0.16). Seed 2 at `η = 0.001`: 0.122 before the switch, the
    /// first twenty rates after it averaging 0.066 (the logistic of `b* = −2.70` is 0.063), and a
    /// gain of 7.78 at input 50,000, far short of the new fixed point's 12.38.
    #[test]
    fn fig_2_deprivation_under_stated_assumptions() {
        let figure = [0.027, 0.0410, 0.0580, 0.0578, 0.0712, 0.0748, 0.0726, 0.0843];
        let start = fixed(&Triesch::FIG1, &Input::FIG1_GAUSSIAN).sigmoid;
        // One run: every rate, and the neuron at the end.
        let run = |eta: f64, seed: u64| {
            let r = Triesch::new(0.1, eta).unwrap();
            let mut n = start;
            let mut rng = Rng::new(seed);
            let mut rates = Vec::with_capacity(50_000);
            for t in 0..50_000 {
                let input = if t < 10_000 { Input::FIG1_GAUSSIAN } else { Input::Gaussian { mean: 0.0, sd: 0.1 } };
                let x = input.sample(&mut rng).unwrap();
                rates.push(n.rate(x));
                n = r.learn(n, x).unwrap();
            }
            (rates, n)
        };
        // The mean and standard deviation of every 20th rate in [lo, hi).
        let spread = |rates: &[f64], lo: usize, hi: usize| {
            let v: Vec<f64> = (lo..hi).step_by(20).map(|i| rates[i]).collect();
            let m = v.iter().sum::<f64>() / v.len() as f64;
            (m, (v.iter().map(|y| (y - m) * (y - m)).sum::<f64>() / (v.len() - 1) as f64).sqrt())
        };
        let windows = |eta: f64| -> Vec<Vec<f64>> {
            let mut w = vec![Vec::new(); 8];
            for seed in 1..=24 {
                let (rates, _) = run(eta, seed);
                for (k, column) in w.iter_mut().enumerate() {
                    column.push(spread(&rates, 10_000 + 5_000 * k, 15_000 + 5_000 * k).1);
                }
            }
            w
        };
        let mean_sd = |v: &[f64]| {
            let m = v.iter().sum::<f64>() / v.len() as f64;
            (m, (v.iter().map(|y| (y - m) * (y - m)).sum::<f64>() / (v.len() - 1) as f64).sqrt())
        };
        let slow = windows(0.001);
        let (m, sd) = mean_sd(&slow[0]);
        assert!(slow[0].iter().all(|&s| s < figure[0]) && figure[0] - m > 4.0 * sd, "the first window: {m} ± {sd} against {}", figure[0]);
        assert!(slow[7].iter().all(|&s| s < figure[7]), "the last window: {:?}", slow[7]);
        let fast = windows(0.002);
        for (k, column) in fast.iter().enumerate() {
            let (m, sd) = mean_sd(column);
            assert!((figure[k] - m).abs() < 3.0 * sd, "η = 0.002, window {k}: {m} ± {sd} against {}", figure[k]);
        }
        let (rates, n) = run(0.001, 2);
        let (m0, before) = spread(&rates, 0, 10_000);
        let (m2, _) = spread(&rates, 45_000, 50_000);
        assert!((before - 0.11).abs() < 0.03 && (m0 - 0.1).abs() < 0.015 && (m2 - 0.095).abs() < 0.015, "{before} {m0} {m2}");
        let first: f64 = rates[10_000..10_020].iter().sum::<f64>() / 20.0;
        assert!((first - 0.063).abs() < 0.008, "{first}");
        assert!((n.a - 7.77).abs() < 0.1 && n.a < 12.38 - 4.0, "{n:?}");
    }

    /// The optimal transfer function `−µ log(1 − F_x(x))` makes the output exactly exponential with
    /// mean `µ`, and Figs. 1c and 1d draw it.
    ///
    /// Pushed through it, 200,000 draws of each input average `µ` within four standard errors and
    /// exceed `µ` with probability `e⁻¹`. Fig. 1c's dotted curve, with the axes fitted by least
    /// squares to the tick marks, reads 0.1608 at `x = 0.8` and 0.3940 at 0.98 (interpolated) and
    /// ends at 0.4602 on its last vertex, the step at 0.99, which that calibration puts at 0.9895;
    /// `−0.1 log(1 − x)` gives 0.16094, 0.39120 and 0.46052. Fig. 1d's is the identity within
    /// 2.2 × 10⁻⁴ at every one of its 96 vertices with the axes fitted to the grid lines,
    /// 2.8 × 10⁻⁴ to the tick marks, 4.0 × 10⁻⁴ through the frame and 4.2 × 10⁻⁴ through the two
    /// outermost grid lines alone, which is `−0.1 log(e^{−x/0.1})`.
    #[test]
    fn the_optimal_transfer_makes_the_output_exponential() {
        let r = Triesch::FIG1;
        for input in [Input::FIG1_GAUSSIAN, Input::FIG1_UNIFORM, Input::FIG1_EXPONENTIAL, Input::Gaussian { mean: -1.0, sd: 3.0 }] {
            let mut rng = Rng::new(7);
            let count = 200_000;
            let (mut sum, mut above) = (0.0, 0);
            for _ in 0..count {
                let y = r.optimal_transfer(&input, input.sample(&mut rng).unwrap()).unwrap();
                sum += y;
                if y > r.mu {
                    above += 1;
                }
            }
            let mean = sum / f64::from(count);
            assert!((mean - r.mu).abs() < 4.0 * r.mu / f64::from(count).sqrt(), "{input:?}: {mean}");
            let p = f64::from(above) / f64::from(count);
            let e = (-1.0_f64).exp();
            assert!((p - e).abs() < 4.0 * (e * (1.0 - e) / f64::from(count)).sqrt(), "{input:?}: {p}");
        }
        let u = Input::FIG1_UNIFORM;
        for (x, drawn, formula) in [(0.8, 0.1608, 0.160_943_791_243_410_07), (0.98, 0.3940, 0.391_202_300_542_814_5), (0.99, 0.4602, 0.460_517_018_598_809_1)] {
            let y = r.optimal_transfer(&u, x).unwrap();
            assert!((y - formula).abs() < 1e-15 && (y - drawn).abs() < 0.004, "x = {x}: {y}");
        }
        for x in [0.0, 0.15, 0.3, 0.45, 0.9] {
            let y = r.optimal_transfer(&Input::FIG1_EXPONENTIAL, x).unwrap();
            assert!((y - x).abs() < 1e-16, "the identity at {x}: {y}");
        }
        assert_eq!(r.optimal_transfer(&u, -0.5).unwrap(), 0.0, "below the support nothing is lost");
        assert_eq!(r.optimal_transfer(&u, 1.0).unwrap(), f64::INFINITY);
        assert_eq!(r.optimal_transfer(&u, 1.5).unwrap(), f64::INFINITY, "above it, not a NaN");
        assert_eq!(r.optimal_transfer(&Input::FIG1_EXPONENTIAL, -2.0).unwrap(), 0.0);
    }

    /// ⚠ Fig. 1b's dotted curve is NOT `−µ log(1 − Φ(x))`: the formula, checked against `SciPy`'s
    /// `log_ndtr` (measured within 3 × 10⁻¹⁶ relative), lies above every tabulated point below, and
    /// above every vertex of the figure's vector path by at least 0.0026 from `x = −1` up; it passes
    /// 1, the maximum rate, between `x = 3.8` and `x = 3.95` (at 3.914), where the drawn curve is at
    /// 0.909.
    ///
    /// The tabulated points are the path (`pdftocairo -svg`, p. 68) interpolated at each `x`, with
    /// the axes fitted by least squares to the panel's tick marks. Up to `x = 2.5` they lie within
    /// 2.9 × 10⁻⁴ of the formula shifted right by 0.1, `−0.1 log(1 − Φ(x − 0.1))`; over all 396
    /// vertices up to 2.5 the path lies within 4.2 × 10⁻⁴ of it through the tick marks, 4.1 × 10⁻⁴
    /// through a fit to the grid lines, 5.5 × 10⁻⁴ through the frame and 6.3 × 10⁻⁴ through the
    /// two outermost grid lines alone, the largest at `x = 2.017` in each, and a least-squares
    /// shift over `x ∈ [−2, 2.5]` is 0.0998, 0.0997, 0.0989 and 0.0985 in the four readings. From
    /// `x = 3` the drawn curve falls below the shifted formula too, by 0.0019 at 3, 0.067 at 4 and
    /// at least 0.0013 at every vertex between, in every reading. Recorded as a measurement, not an
    /// explanation: the paper prints no formula for the curve.
    #[test]
    fn fig_1b_dotted_curve_is_not_the_formula() {
        let r = Triesch::FIG1;
        let g = Input::FIG1_GAUSSIAN;
        let shifted = Input::Gaussian { mean: 0.1, sd: 1.0 };
        let rows = [
            (-1.0, 0.0144, 0.017275377902344985),
            (0.0, 0.0618, 0.06931471805599453),
            (1.0, 0.1693, 0.18410216450092634),
            (2.0, 0.3550, 0.3783184333682032),
            (2.5, 0.4801, f64::NAN),
            (3.0, 0.6265, 0.660772622151035),
            (3.8, 0.8793, 0.9534022141532573),
            (3.95, 0.9172, 1.0150012413755696),
            (4.0, 0.9272, 1.0360101486527291),
        ];
        for (x, drawn, scipy) in rows {
            let y = r.optimal_transfer(&g, x).unwrap();
            if !scipy.is_nan() {
                assert!((y - scipy).abs() < 1e-15 * scipy, "x = {x}: {y} against SciPy's {scipy}");
            }
            assert!(y - drawn > 0.0025, "x = {x}: the figure's {drawn} is not the formula's {y}");
            let moved = r.optimal_transfer(&shifted, x).unwrap() - drawn;
            if x <= 2.5 {
                assert!(moved.abs() < 5e-4, "x = {x}: shifted by 0.1 the formula is {moved} from the figure");
            } else {
                assert!(moved > 1.5e-3, "x = {x}: the figure falls below even the shifted formula, {moved}");
            }
        }
        assert!((r.optimal_transfer(&g, 0.0).unwrap() - 0.1 * core::f64::consts::LN_2).abs() < 1e-17);
        assert!(r.optimal_transfer(&g, 3.8).unwrap() < 1.0 && r.optimal_transfer(&g, 3.95).unwrap() > 1.0);
    }

    /// `log(1 − F)` where `1 − F` itself underflows: the Gaussian's through `erfcx`, against
    /// `SciPy`'s `log_ndtr(−z)` (measured within 8 × 10⁻¹⁶ relative); the uniform's and the
    /// exponential's in closed form, `0` below the support and `−∞` — not a NaN — above it.
    #[test]
    fn the_survival_function_is_accurate_in_both_tails() {
        let g = Input::Gaussian { mean: 1.0, sd: 2.0 };
        let rows = [
            (-3.0, -0.001350809964748193),
            (-1.0, -0.17275377902344985),
            (0.0, -core::f64::consts::LN_2),
            (0.5, -1.175911761593619),
            (1.0, -1.8410216450092634),
            (2.0, -3.7831843336820317),
            (10.0, -53.23128515051248),
            (40.0, -804.6084420137538),
        ];
        for (z, scipy) in rows {
            let got = g.ln_survival(1.0 + 2.0 * z);
            assert!((got - scipy).abs() <= 3e-15 * scipy.abs(), "z = {z}: {got} against {scipy}");
        }
        assert_eq!(g.ln_survival(1.0 + 2.0 * -40.0), 0.0, "SciPy: -0.0");
        let u = Input::Uniform { lo: -1.0, hi: 3.0 };
        assert_eq!(u.ln_survival(-1.0), 0.0);
        assert_eq!(u.ln_survival(-7.0), 0.0);
        assert_eq!(u.ln_survival(1.0), 0.5_f64.ln());
        assert_eq!(u.ln_survival(3.0), f64::NEG_INFINITY);
        assert_eq!(u.ln_survival(3.5), f64::NEG_INFINITY);
        let e = Input::Exponential { mean: 2.0 };
        assert_eq!(e.ln_survival(-1.0), 0.0);
        assert_eq!(e.ln_survival(0.0), 0.0);
        assert_eq!(e.ln_survival(3.0), -1.5);
    }

    /// Simpson's rule: exact for cubics with one panel and not for quartics, and fourth order where
    /// the support has an edge — errors falling by 16.2, 16.1 and 16.0 as the panels double from 16
    /// to 128 for the uniform input. For the Gaussian, [`Input::PANELS`] is within 2.2 × 10⁻¹⁴ of a
    /// run with eight times the panels, and 64 panels are 1.4 × 10⁻³ away.
    #[test]
    fn simpson_is_fourth_order_where_the_support_has_an_edge() {
        let u = Input::FIG1_UNIFORM;
        assert_eq!(u.expect_with(1, |x| x * x * x).unwrap(), 0.25);
        assert_eq!(u.expect_with(2, |x| x * x * x).unwrap(), 0.25);
        assert_eq!(u.expect_with(1, |x| x * x * x * x).unwrap(), 1.25 / 6.0);
        let s = neuron(8.0, -3.0);
        let f = |x: f64| {
            let y = s.rate(x);
            x * y * y
        };
        let fine = u.expect_with(1 << 15, f).unwrap();
        let err: Vec<f64> = [16, 32, 64, 128].iter().map(|&p| (u.expect_with(p, f).unwrap() - fine).abs()).collect();
        for pair in err.windows(2) {
            let ratio = pair[0] / pair[1];
            assert!((15.8..16.4).contains(&ratio), "{err:?}: ratio {ratio}");
        }
        let g = Input::Gaussian { mean: 0.3, sd: 2.0 };
        let fine = g.expect_with(1 << 15, f).unwrap();
        assert!((g.expect(f).unwrap() - fine).abs() < 1e-13, "the Gaussian at the default panels");
        assert!((g.expect_with(64, f).unwrap() - fine).abs() > 1e-3, "and not at 64");
        assert_eq!(u.expect_with(Input::MAX_PANELS, |_| 1.0).unwrap().round(), 1.0, "the cap itself is accepted");
    }

    /// Every density integrates to one — the exponential's to `1 + 3.15 × 10⁻¹²` at the default
    /// panels, whatever its mean, the size of its edge's fourth-order error — and has its closed-form
    /// mean and entropy (to 1.7 × 10⁻¹¹ at worst, the exponential's).
    #[test]
    fn the_densities_have_their_closed_forms() {
        // Measured: E[1] within 4.3 × 10⁻¹⁵ and E[x] within 4.4 × 10⁻¹⁵ of the mean for the
        // Gaussian and the uniform; the exponential's mean 9.5 × 10⁻¹² low, relatively.
        let inputs = [
            (Input::Gaussian { mean: 0.3, sd: 2.0 }, 0.3, 1e-14, 2e-14),
            (Input::Uniform { lo: -1.0, hi: 3.0 }, 1.0, 1e-14, 2e-14),
            (Input::Exponential { mean: 0.1 }, 0.1, 4e-12, 3e-11),
            (Input::Exponential { mean: 2.0 }, 2.0, 4e-12, 3e-11),
        ];
        for (input, mean, one, relative) in inputs {
            assert!((input.expect(|_| 1.0).unwrap() - 1.0).abs() < one, "{input:?}");
            assert!((input.expect(|x| x).unwrap() - mean).abs() < relative * mean, "{input:?}");
            let h = -input.expect(|x| input.density(x).ln()).unwrap();
            assert!((h - input.entropy()).abs() < 3e-11, "{input:?}: {h} against {}", input.entropy());
        }
        for mean in [0.1, 2.0, 50.0] {
            let one = Input::Exponential { mean }.expect(|_| 1.0).unwrap();
            assert!((one - 1.0 - 3.15e-12).abs() < 0.05e-12, "mean {mean}: {}", one - 1.0);
        }
        let g = Input::Gaussian { mean: 0.3, sd: 2.0 };
        assert!((g.density(0.3) - 1.0 / (2.0 * core::f64::consts::TAU.sqrt())).abs() < 1e-17);
        assert!((g.entropy() - 0.5 * (core::f64::consts::TAU * core::f64::consts::E * 4.0).ln()).abs() < 1e-15);
        let u = Input::Uniform { lo: -1.0, hi: 3.0 };
        assert_eq!((u.density(-1.0), u.density(3.0), u.density(3.5), u.density(-1.5)), (0.25, 0.25, 0.0, 0.0));
        let e = Input::Exponential { mean: 2.0 };
        assert_eq!((e.density(0.0), e.density(-0.5)), (0.5, 0.0));
        assert_eq!(e.entropy(), 1.0 + 2.0_f64.ln());
        assert_eq!(Input::FIG1_GAUSSIAN.domain(), (-10.0, 10.0));
        assert_eq!(Input::FIG1_EXPONENTIAL.domain(), (0.0, 4.0));
        assert_eq!(u.domain(), (-1.0, 3.0));
    }

    /// Draws have their input's mean and variance, within five standard errors of 200,000 draws,
    /// and the Gaussian's are symmetric about its mean.
    #[test]
    fn the_draws_have_their_inputs_moments() {
        let inputs = [
            (Input::Gaussian { mean: 0.3, sd: 2.0 }, 0.3, 4.0),
            (Input::Uniform { lo: -1.0, hi: 3.0 }, 1.0, 16.0 / 12.0),
            (Input::Exponential { mean: 0.1 }, 0.1, 0.01),
        ];
        for (input, mean, var) in inputs {
            let mut rng = Rng::new(11);
            let count = 200_000;
            let draws: Vec<f64> = (0..count).map(|_| input.sample(&mut rng).unwrap()).collect();
            let m = draws.iter().sum::<f64>() / f64::from(count);
            let v = draws.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / f64::from(count - 1);
            assert!((m - mean).abs() < 5.0 * (var / f64::from(count)).sqrt(), "{input:?}: mean {m}");
            assert!((v - var).abs() < 0.02 * var, "{input:?}: variance {v}");
            if let Input::Gaussian { mean, .. } = input {
                let below = draws.iter().filter(|&&x| x < mean).count() as f64 / f64::from(count);
                assert!((below - 0.5).abs() < 5.0 * (0.25 / f64::from(count)).sqrt(), "{below}");
            }
        }
        let mut rng = Rng::new(3);
        assert!((0..10_000).all(|_| Input::FIG1_EXPONENTIAL.sample(&mut rng).unwrap() >= 0.0));
        assert!((0..10_000).all(|_| {
            let x = Input::FIG1_UNIFORM.sample(&mut rng).unwrap();
            (0.0..1.0).contains(&x)
        }));
    }

    /// A step that would carry the gain to zero, below it, or past `f64::MAX` is refused and names
    /// both ends; so is an averaged step. A step whose bias `f64` cannot hold is refused naming the
    /// bias.
    ///
    /// With `µ = η = 1/8` and a neuron saturated at `y = 1`, `h(1) = 1 − 10 + 8 = −1`, so
    /// `Δa = ⅛(1/a − x)`: `x = 64` takes `a = 1` to `−6.875` and `x = 9` to exactly `0`. With
    /// `η = 5 × 10³⁰⁷` the neuron `(1.7 × 10³⁰⁸, −1.5 × 10³⁰⁸)` at `x = 1` fires at exactly one,
    /// so `Δb = −η` and `Δa = η/a − η`: both finite, the gain lands at `1.2 × 10³⁰⁸`, and the bias at
    /// `−2 × 10³⁰⁸`, past `f64::MAX`. With `η = 10³⁰⁸` the neuron `(1.7 × 10³⁰⁸, −1.79 × 10³⁰⁸)` at
    /// `x = 1` is silent, so `Δb = η` and `Δa = η/a + η`, finite, and the gain lands past `f64::MAX`.
    #[test]
    fn a_step_that_loses_the_gain_is_refused() {
        let r = Triesch::new(0.125, 0.125).unwrap();
        let saturated = neuron(1.0, 64.0);
        assert_eq!(saturated.rate(9.0), 1.0);
        let err = r.learn(neuron(1.0, 0.0), 64.0).unwrap_err();
        assert_eq!(err, IntrinsicError::Overshoot { a: 1.0, next: -6.875 });
        assert_eq!(
            err.to_string(),
            "one step of eq. 12 takes the gain from 1 to -6.875, where the rule is undefined; the learning rate is too large for this input"
        );
        assert_eq!(r.learn(saturated, 9.0).unwrap_err(), IntrinsicError::Overshoot { a: 1.0, next: 0.0 });
        assert_eq!(r.learn(saturated, 8.0).unwrap(), neuron(0.125, 63.875), "one input short of zero");
        let big = Triesch::new(0.125, 5e307).unwrap();
        let far = neuron(1.7e308, -1.5e308);
        let (da, db) = big.update(far, 1.0).unwrap();
        assert!(db == -5e307 && da.is_finite() && far.a + da > 1e308, "{da} {db}");
        let err = big.learn(far, 1.0).unwrap_err();
        assert_eq!(err, IntrinsicError::Unrepresentable { what: "the bias after the step", value: f64::NEG_INFINITY });
        let err = Triesch::new(0.125, 1e308).unwrap().learn(neuron(1.7e308, -1.79e308), 1.0).unwrap_err();
        assert_eq!(err, IntrinsicError::Overshoot { a: 1.7e308, next: f64::INFINITY }, "the gain past f64::MAX");
        let fast = Triesch::new(0.1, 64.0).unwrap();
        let err = fast.mean_step(neuron(4.0, 20.0), &Input::FIG1_UNIFORM).unwrap_err();
        assert!(matches!(err, IntrinsicError::Overshoot { a, next } if a == 4.0 && next < 0.0), "{err:?}");
    }

    /// Every refusal, rendered.
    #[test]
    fn every_refusal_names_what_it_refused() {
        let r = Triesch::FIG1;
        let n = neuron(1.0, 0.0);
        let g = Input::FIG1_GAUSSIAN;
        let cases: Vec<(Result<(), IntrinsicError>, &str)> = vec![
            (Sigmoid::new(0.0, 0.0).map(|_| ()), "gain a = 0 must be finite and positive: eq. 12 divides by it and eq. 9 takes its logarithm"),
            (Sigmoid::new(-1.5, 0.0).map(|_| ()), "gain a = -1.5 must be finite and positive: eq. 12 divides by it and eq. 9 takes its logarithm"),
            (Sigmoid::new(f64::INFINITY, 0.0).map(|_| ()), "gain a = inf must be finite and positive: eq. 12 divides by it and eq. 9 takes its logarithm"),
            (Sigmoid::new(1.0, f64::NAN).map(|_| ()), "b = NaN is not finite"),
            (Sigmoid { a: f64::NAN, b: 0.0 }.check(), "gain a = NaN must be finite and positive: eq. 12 divides by it and eq. 9 takes its logarithm"),
            (Sigmoid { a: 1.0, b: f64::NEG_INFINITY }.check(), "b = -inf is not finite"),
            (Triesch::new(0.0, 0.001).map(|_| ()), "mu = 0 must be finite and positive"),
            (Triesch::new(0.1, -1.0).map(|_| ()), "eta = -1 must be finite and positive"),
            (Triesch { mu: f64::NAN, eta: 0.001 }.check(), "mu = NaN must be finite and positive"),
            (Triesch { mu: 0.1, eta: f64::INFINITY }.check(), "eta = inf must be finite and positive"),
            (Input::Gaussian { mean: f64::NAN, sd: 1.0 }.check(), "mean = NaN is not finite"),
            (Input::Gaussian { mean: 0.0, sd: 0.0 }.check(), "sd = 0 must be finite and positive"),
            (Input::Uniform { lo: f64::NEG_INFINITY, hi: 1.0 }.check(), "lo = -inf is not finite"),
            (Input::Uniform { lo: 0.0, hi: f64::INFINITY }.check(), "hi = inf is not finite"),
            (Input::Uniform { lo: 1.0, hi: 1.0 }.check(), "the uniform input's interval [1, 1] is empty; it needs lo < hi"),
            (Input::Uniform { lo: 2.0, hi: 1.0 }.check(), "the uniform input's interval [2, 1] is empty; it needs lo < hi"),
            (Input::Uniform { lo: f64::NAN, hi: 1.0 }.check(), "lo = NaN is not finite"),
            (Input::Exponential { mean: -0.1 }.check(), "mean = -0.1 must be finite and positive"),
            (g.expect_with(0, |x| x).map(|_| ()), "a quadrature needs from 1 to 1048576 panels, not 0"),
            (g.expect_with(Input::MAX_PANELS + 1, |x| x).map(|_| ()), "a quadrature needs from 1 to 1048576 panels, not 1048577"),
            (r.update(n, f64::NAN).map(|_| ()), "x = NaN is not finite"),
            (r.update(Sigmoid { a: -1.0, b: 0.0 }, 0.0).map(|_| ()), "gain a = -1 must be finite and positive: eq. 12 divides by it and eq. 9 takes its logarithm"),
            (Triesch { mu: 0.1, eta: 0.0 }.update(n, 0.0).map(|_| ()), "eta = 0 must be finite and positive"),
            (r.learn(n, f64::INFINITY).map(|_| ()), "x = inf is not finite"),
            (r.optimal_transfer(&g, f64::NAN).map(|_| ()), "x = NaN is not finite"),
            (r.optimal_transfer(&Input::Exponential { mean: 0.0 }, 1.0).map(|_| ()), "mean = 0 must be finite and positive"),
            (Triesch { mu: -1.0, eta: 0.1 }.optimal_transfer(&g, 1.0).map(|_| ()), "mu = -1 must be finite and positive"),
        ];
        for (got, want) in cases {
            assert_eq!(got.unwrap_err().to_string(), want);
        }
        // Results that f64 cannot hold, each from parameters that are finite and in range. The
        // b-components are checked first: where `Δb` is not finite, `Δa = η/a + x Δb` is not either.
        let tiny = Sigmoid { a: 5e-324, b: 0.0 };
        let huge_eta = Triesch { mu: 0.1, eta: 1e308 };
        let tiny_mu = Triesch { mu: 5e-324, eta: 0.001 };
        let narrow = Input::Uniform { lo: 0.0, hi: 1e-10 };
        let unheld: Vec<(Result<(), IntrinsicError>, &str)> = vec![
            (huge_eta.update(n, 0.0).map(|_| ()), "Δb comes out as -inf"),
            (r.update(tiny, 0.0).map(|_| ()), "Δa comes out as inf"),
            (Triesch { mu: 0.125, eta: 5e307 }.learn(Sigmoid { a: 1.7e308, b: -1.5e308 }, 1.0).map(|_| ()), "the bias after the step comes out as -inf"),
            (huge_eta.mean_update(n, &g).map(|_| ()), "E[Δb] comes out as NaN"),
            (r.mean_update(tiny, &g).map(|_| ()), "E[Δa] comes out as inf"),
            (r.objective(Sigmoid { a: 1e308, b: 0.0 }, &g).map(|_| ()), "D comes out as inf"),
            (tiny_mu.gradient(n, &g).map(|_| ()), "∂D/∂b comes out as NaN"),
            (r.gradient(tiny, &g).map(|_| ()), "∂D/∂a comes out as -inf"),
            (tiny_mu.hessian(n, &g).map(|_| ()), "∂²D/∂a∂b comes out as NaN"),
            (r.hessian(Sigmoid { a: 1e-155, b: 0.0 }, &g).map(|_| ()), "∂²D/∂a² comes out as inf"),
            (Triesch { mu: 1e-300, eta: 0.001 }.hessian(Sigmoid { a: 1000.0, b: -2.0 }, &narrow).map(|_| ()), "∂²D/∂b² comes out as inf"),
        ];
        let tail = "which f64 cannot hold: the rule's parameters, the neuron or the input are too extreme for the arithmetic";
        for (got, head) in unheld {
            assert_eq!(got.unwrap_err().to_string(), format!("{head}, {tail}"));
        }
        // The edge of `1/a`: finite down to `1/f64::MAX = 5.56 × 10⁻³⁰⁹`, infinite below it.
        assert!((1.0 / f64::MAX - 5.5627e-309).abs() < 1e-313);
        assert!(r.gradient(Sigmoid { a: 5.57e-309, b: 0.0 }, &g).is_ok(), "a gain just above the edge");
        assert!(r.gradient(Sigmoid { a: 5.55e-309, b: 0.0 }, &g).is_err(), "a gain just below it");
        // Every computation over an input checks the rule, the neuron and the input, in that order.
        let bad_rule = Triesch { mu: 0.0, eta: 0.001 };
        let bad_neuron = Sigmoid { a: 0.0, b: 0.0 };
        let bad_input = Input::Gaussian { mean: 0.0, sd: -1.0 };
        type Probe = fn(&Triesch, Sigmoid, &Input) -> Result<(), IntrinsicError>;
        let probes: [(&str, Probe); 6] = [
            ("mean_update", |r, s, i| r.mean_update(s, i).map(|_| ())),
            ("mean_step", |r, s, i| r.mean_step(s, i).map(|_| ())),
            ("objective", |r, s, i| r.objective(s, i).map(|_| ())),
            ("gradient", |r, s, i| r.gradient(s, i).map(|_| ())),
            ("hessian", |r, s, i| r.hessian(s, i).map(|_| ())),
            ("fixed_point", |r, s, i| r.fixed_point(s, i).map(|_| ())),
        ];
        for (name, probe) in probes {
            assert_eq!(probe(&bad_rule, n, &g).unwrap_err().to_string(), "mu = 0 must be finite and positive", "{name}");
            assert_eq!(
                probe(&r, bad_neuron, &g).unwrap_err().to_string(),
                "gain a = 0 must be finite and positive: eq. 12 divides by it and eq. 9 takes its logarithm",
                "{name}"
            );
            assert_eq!(probe(&r, n, &bad_input).unwrap_err().to_string(), "sd = -1 must be finite and positive", "{name}");
        }
        let mut rng = Rng::new(1);
        assert_eq!(bad_input.sample(&mut rng).unwrap_err().to_string(), "sd = -1 must be finite and positive");
        assert_eq!(bad_input.expect(|x| x).unwrap_err(), IntrinsicError::NotPositive { what: "sd", value: -1.0 });
    }

    /// The paper's parameters, pinned: `µ = 0.1`, `η = 0.001` from the Fig. 1 caption, and the three
    /// inputs read from the figure's density curves. The numerical method's own constants are not
    /// pinned here: each is held by what it does — [`Triesch::HALVINGS`] by the start from
    /// `5 × 10¹⁷` that only its last halving can serve and the start from `10¹⁸` that one more
    /// would, and the others by the fixed points, step counts and refusals of the tests above.
    #[test]
    fn the_papers_parameters_are_pinned() {
        assert_eq!((Triesch::FIG1.mu, Triesch::FIG1.eta), (0.1, 0.001));
        assert_eq!(Triesch::FIG1.check(), Ok(()));
        assert_eq!(Input::FIG1_GAUSSIAN, Input::Gaussian { mean: 0.0, sd: 1.0 });
        assert_eq!(Input::FIG1_UNIFORM, Input::Uniform { lo: 0.0, hi: 1.0 });
        assert_eq!(Input::FIG1_EXPONENTIAL, Input::Exponential { mean: 0.1 });
        assert_eq!(Triesch::new(0.1, 0.001), Ok(Triesch::FIG1));
        assert_eq!(Sigmoid::new(1.5, -2.0), Ok(Sigmoid { a: 1.5, b: -2.0 }));
    }
}
