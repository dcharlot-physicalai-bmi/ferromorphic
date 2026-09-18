//! Surrogate gradients: how a spiking network is trained by backpropagation at all.
//!
//! # The problem, stated exactly
//!
//! A spike is a Heaviside step. The neuron emits `S = Theta(U - theta)`, one when the membrane
//! reaches threshold and zero otherwise. That function has derivative **zero everywhere it is
//! defined** and is undefined at the threshold itself, where the derivative is a Dirac delta. So
//! `dS/dU` is either 0 or infinite, never anything a gradient step can use, and backpropagation
//! through a spiking network is impossible as stated. This is not a numerical difficulty that a
//! smaller learning rate fixes; the true gradient of the loss with respect to a recurrent weight is
//! *identically zero* almost everywhere, and
//! `the_true_gradient_of_a_spiking_network_is_zero_which_is_why_the_surrogate_exists` in this module
//! measures exactly that with finite differences.
//!
//! # The field's answer
//!
//! Keep the Heaviside on the **forward** pass — the network really does emit binary spikes, which is
//! the entire reason event-driven hardware saves energy — and substitute a smooth function on the
//! **backward** pass. The backward pass then computes the gradient of a network that was never run.
//! That sounds like a cheat and it is one; the justification is empirical and it is strong, and the
//! honest framing is Neftci, Mostafa and Zenke's: surrogate gradients are a *biased* gradient
//! estimator whose bias is controlled by the shape and, much more importantly, the **scale** of the
//! surrogate.
//!
//! Primary sources, all open:
//!
//! - Neftci, Mostafa & Zenke, "Surrogate Gradient Learning in Spiking Neural Networks",
//!   IEEE Signal Process. Mag. 36(6):51-63, 2019 — the review, and the source of the
//!   discrete-time network this module trains.
//! - Zenke & Ganguli, "`SuperSpike`: Supervised Learning in Multilayer Spiking Neural Networks",
//!   Neural Comput. 30:1514-1541, 2018 — the fast-sigmoid surrogate.
//! - Wu, Deng, Li, Zhu & Shi, "Spatio-Temporal Backpropagation for Training High-Performance
//!   Spiking Neural Networks", Front. Neurosci. 12:331, 2018 — the rectangular surrogate (STBP).
//! - Bellec, Salaj, Subramoney, Legenstein & Maass, "Long short-term memory and learning-to-learn
//!   in networks of spiking neurons", `NeurIPS` 2018 — the triangular pseudo-derivative.
//! - Shrestha & Orchard, "SLAYER: Spike Layer Error Reassignment in Time", `NeurIPS` 2018 — the
//!   exponential surrogate.
//! - Bengio, Leonard & Courville, "Estimating or Propagating Gradients Through Stochastic Neurons
//!   for Conditional Computation", arXiv:1308.3432, 2013 — the straight-through estimator, which
//!   predates the spiking literature and is where the trick comes from.
//! - Zenke & Vogels, "The Remarkable Robustness of Surrogate Gradient Learning for Instilling
//!   Complex Function in Spiking Neural Networks", Neural Comput. 33:899-925, 2021 — the finding
//!   that the *shape* barely matters and the *scale* does.
//!
//! # Every surrogate here is a mollifier of the same delta
//!
//! The derivative being replaced is `delta(x)`. A sensible replacement integrates to one over the
//! real line, is non-negative, and peaks where the delta sits. Those three properties are the
//! module's organising idea and they are checked numerically for every family:
//!
//! | family | `mass()` | normalised by construction? |
//! |---|---|---|
//! | [`ArcTan`] | 1 | yes |
//! | [`SigmoidDeriv`] | 1 | yes |
//! | [`Rectangular`] | 1 | yes |
//! | [`Exponential`] | 1 | yes |
//! | [`Gaussian`] | 1 | yes |
//! | [`FastSigmoid`] | `2 / beta` | **no** — peak-normalised instead |
//! | [`Triangular`] | `peak * half_width` | **no** — peak-normalised instead |
//! | [`StraightThrough`] | `2 * half_width` | **no** — it is a window, not a density |
//!
//! The three that are not normalised are not wrong, they are *scaled*, and the field gets away with
//! it because a constant factor on every gradient is absorbed by the learning rate. It stops being
//! absorbable the moment two layers use different scales, or the moment a scale is reported as if it
//! were a shape. [`Scaled::unit_mass`] renormalises any of them, and the gain it applies is exactly
//! the number that was hiding in the learning rate.
//!
//! # The width is the hyperparameter
//!
//! As the sharpness parameter grows, every family here narrows toward the delta:
//! [`Surrogate::fwhm`] shrinks as `1 / factor` under [`Surrogate::sharpened`], and for the
//! normalised families the mass stays at one while the height grows. That is the delta sequence.
//! **It is also the failure mode**: in the sharp limit the surrogate is zero at every membrane
//! potential that is not already at threshold, so the gradient vanishes and training stops. The test
//! `an_over_sharp_surrogate_stops_learning` takes one network, one seed and one task and changes
//! only the width: at a full width at half maximum of `0.637` thresholds the loss falls to `4.2e-5`
//! in 300 steps, and at `0.00212` it sits at `0.69315` — `ln(2)`, the class prior, which is a
//! network that has learned nothing.
//!
//! # What this module can and cannot verify
//!
//! [`LifLayer`] carries a hand-rolled reverse-mode gradient for one recurrent LIF layer with a
//! leaky readout, trained through time. [`SpikeFn::Smooth`] replaces the Heaviside on the forward
//! pass with the surrogate's own antiderivative, which makes the network genuinely differentiable —
//! and in that mode the backward pass is the *exact* gradient, so it can be checked against central
//! finite differences. That check passes to a relative `1e-6`, and it is what makes the plumbing
//! trustworthy: the time recurrence, the two-stage synapse-then-membrane filter, the soft reset, the
//! recurrent weights and the readout.
//!
//! **Beside that figure, the caveat**: the finite-difference check validates the *reverse-mode
//! machinery*, not the surrogate approximation. The approximation cannot be validated that way,
//! because the quantity it approximates — the gradient of the Heaviside network — is zero. There is
//! no ground truth to compare against. Anyone who tells you their surrogate gradient was "verified
//! against autograd" has verified the same thing this module verifies, no more.
//!
//! # Units
//!
//! The network's time constants enter in **seconds** at [`LifLayerSpec`] and are converted there to
//! the dimensionless decay factors `alpha = exp(-dt / tau_syn)`, `beta = exp(-dt / tau_mem)`,
//! `kappa = exp(-dt / tau_out)`. Inside [`LifLayer`] everything is dimensionless with threshold
//! `theta = 1`, exactly as Neftci et al. print it, so that the update equations can be compared
//! against the paper line by line. The surrogates themselves take a dimensionless argument: `x` is
//! `U - theta` in units of the threshold, which is why a width of `1.0` means "one threshold wide".

use crate::rng::Rng;

/// Why a surrogate or a training step could not produce an answer.
///
/// Every variant names the offending quantity and its value. The module returns these rather than
/// producing a plausible number, because a spiking network that silently trains on a `NaN` gradient
/// reports a falling loss for several epochs before every weight becomes non-finite at once.
#[derive(Debug, Clone, PartialEq)]
pub enum SurrogateError {
    /// A scalar parameter was not finite.
    NonFiniteParam {
        /// Which parameter, by its field name.
        what: &'static str,
        /// The offending value, so the caller can see whether it was an infinity or a `NaN`.
        value: f64,
    },
    /// A scalar parameter had to be strictly positive and was not.
    ///
    /// A zero membrane time constant is not "very fast", it is a division by zero in
    /// `exp(-dt / tau)`; a zero sharpness is not "very smooth", it is a surrogate of zero mass.
    NotPositive {
        /// Which parameter, by its field name.
        what: &'static str,
        /// The offending value.
        value: f64,
    },
    /// An element of the input array was not finite.
    NonFiniteInput {
        /// Flat index into the `t_steps * n_in` input array.
        index: usize,
        /// The offending value.
        value: f64,
    },
    /// An array's length did not match the layer it was handed to.
    ShapeMismatch {
        /// Which array, by its role.
        what: &'static str,
        /// The length that arrived.
        got: usize,
        /// The length or divisor that was required.
        want: usize,
    },
    /// The forward pass produced a non-finite state and stopped there.
    ///
    /// Reported with the step and neuron so that a diverging run can be traced to its first bad
    /// value rather than to the `NaN` loss it becomes twenty steps later.
    Diverged {
        /// Time step at which the state went non-finite.
        step: usize,
        /// Recurrent-unit index whose state went non-finite.
        neuron: usize,
        /// The offending value.
        value: f64,
    },
    /// A classification target named a class the readout does not have.
    TargetOutOfRange {
        /// The target class that was requested.
        target: usize,
        /// The number of readout units available.
        n_out: usize,
    },
    /// A batch with no patterns in it has no mean loss, and zero is the wrong answer.
    EmptyBatch,
}

impl core::fmt::Display for SurrogateError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NonFiniteParam { what, value } => write!(f, "{what} is not finite ({value})"),
            Self::NotPositive { what, value } => write!(f, "{what} must be > 0, got {value}"),
            Self::NonFiniteInput { index, value } => {
                write!(f, "input element {index} is not finite ({value})")
            }
            Self::ShapeMismatch { what, got, want } => {
                write!(f, "{what} has length {got}, which does not fit {want}")
            }
            Self::Diverged { step, neuron, value } => {
                write!(f, "state of unit {neuron} went non-finite ({value}) at step {step}")
            }
            Self::TargetOutOfRange { target, n_out } => {
                write!(f, "target class {target} is past the {n_out} readout units")
            }
            Self::EmptyBatch => write!(f, "an empty batch has no mean loss"),
        }
    }
}

/// So that `?` works in a caller whose error type is `Box<dyn Error>`, which is what every example
/// and doctest in this crate uses.
impl std::error::Error for SurrogateError {}

/// The Heaviside step, which is what a neuron actually computes.
///
/// `1.0` at and above zero, `0.0` below. The value **at** zero is a convention and this crate's
/// convention is "reaching threshold fires", matching [`crate::neuron::Lif`], which spikes on
/// `v >= v_th`. Its derivative is zero everywhere it exists, which is the whole reason this module
/// exists.
#[must_use]
pub fn heaviside(x: f64) -> f64 {
    if x >= 0.0 { 1.0 } else { 0.0 }
}

/// A forward Heaviside and a backward mollifier, kept deliberately apart.
///
/// Implementors supply the derivative they *pretend* the step function has, plus three closed forms
/// about that derivative — its integral, its height and its width — which exist so the
/// implementation can be checked against algebra rather than against itself. All four of
/// [`Surrogate::backward`], [`Surrogate::antiderivative`], [`Surrogate::mass`] and
/// [`Surrogate::fwhm`] are verified numerically in this module's tests for every family.
///
/// The argument `x` is `U - theta` in **units of the threshold**, so `x = 0` is a neuron exactly at
/// threshold and `x = -1` is a neuron one full threshold below it.
///
/// Object-safe on purpose: [`catalogue`] hands back `Box<dyn Surrogate>` so that a test or a lesson
/// can iterate over every published family and assert the same property of each. That is why the
/// trait requires `Debug` and not `Clone`.
pub trait Surrogate: core::fmt::Debug {
    /// The family's name, for a table or an error message.
    ///
    /// Stable across versions: it is used as a key in printed comparisons, not only for display.
    fn name(&self) -> &'static str;

    /// The Heaviside step. **No implementor overrides this**, and that is the point of the module:
    /// every surrogate runs the identical forward pass and differs only on the way back.
    fn forward(&self, x: f64) -> f64 {
        heaviside(x)
    }

    /// The surrogate derivative at `x`, in units of one over the threshold.
    ///
    /// Non-negative and maximal at `x = 0` for every family here.
    ///
    /// `x` must be finite. This method is called once per unit per time step per training step, so
    /// it does not branch on finiteness; the contract is enforced upstream instead, by
    /// [`LifLayer::forward`] rejecting a non-finite input and by every constructor in this module
    /// rejecting a non-finite parameter. A non-finite `x` returns a non-finite result.
    fn backward(&self, x: f64) -> f64;

    /// The antiderivative `Phi` with `Phi'(x) == backward(x)` and `Phi(-inf) == 0`.
    ///
    /// `Phi(+inf)` is [`Surrogate::mass`], which is `1` only for the normalised families.
    ///
    /// This is the smooth spike function whose gradient the surrogate is pretending to be, and it
    /// is public because it is what makes the backward pass checkable: [`SpikeFn::Smooth`] runs the
    /// network with `Phi` in place of the step, and in that network the module's reverse-mode
    /// gradient is exact and can be compared against finite differences.
    fn antiderivative(&self, x: f64) -> f64;

    /// The analytic integral of [`Surrogate::backward`] over the whole real line.
    ///
    /// `1.0` exactly for a family normalised by construction. Otherwise it is the factor by which
    /// this family over- or under-weights every gradient it produces — a constant the learning rate
    /// silently absorbs until two layers disagree about it.
    fn mass(&self) -> f64;

    /// The analytic value of `backward(0)`, the height of the mollifier.
    ///
    /// A separate expression from the implementation of `backward`, on purpose: the test that
    /// compares them is comparing algebra against code.
    fn peak(&self) -> f64;

    /// Full width at half maximum, in thresholds: the width of the region where `backward` is at
    /// least half its peak.
    ///
    /// This is the module's single comparable measure of "how sharp" a surrogate is, because the
    /// families parameterise sharpness differently and their parameters are not interchangeable.
    /// `fwhm` is the number to report when comparing two surrogates, not `beta` or `alpha`.
    fn fwhm(&self) -> f64;

    /// The same family with its width divided by `factor`.
    ///
    /// A pure horizontal rescale, `backward(x) -> backward(factor * x)`, followed by whatever
    /// renormalisation the family's own parameterisation performs: the normalised families keep
    /// `mass() == 1` and grow taller, while [`FastSigmoid`], [`Triangular`] and
    /// [`StraightThrough`] keep their peak and lose mass as `1 / factor`.
    ///
    /// `None` for a `factor` that is not finite or not strictly positive. There is no sensible
    /// surrogate of negative width, and returning the unsharpened one would hide the caller's bug.
    fn sharpened(&self, factor: f64) -> Option<Box<dyn Surrogate>>;
}

/// `true` when `factor` may be used as a width divisor.
fn ok_factor(factor: f64) -> bool {
    factor.is_finite() && factor > 0.0
}

/// The Gauss error function, to about 3e-14 relative.
///
/// Evaluated by the all-positive confluent series
/// `erf(x) = (2/sqrt(pi)) * x * exp(-x^2) * sum_n (2 x^2)^n / (2n+1)!!`
/// (Abramowitz & Stegun 7.1.6), which has **no cancellation** — every term is positive — unlike the
/// alternating Taylor series, which loses most of its digits by `x = 3`. Saturated to `+/-1` beyond
/// `|x| = 6`, where `erfc(6) = 2.2e-17` is already below the spacing of `f64` near one.
///
/// Here because [`Gaussian::antiderivative`] is the normal CDF and there is no elementary closed
/// form for it. Checked against published values of `erf(0.5)`, `erf(1)`, `erf(2)` and `erf(3)` to
/// `1e-12`.
#[must_use]
pub fn erf(x: f64) -> f64 {
    let a = x.abs();
    if !a.is_finite() {
        return if x.is_nan() { x } else { x.signum() };
    }
    if a > 6.0 {
        return if x < 0.0 { -1.0 } else { 1.0 };
    }
    let z = 2.0 * a * a;
    let mut term = 1.0_f64;
    let mut sum = 1.0_f64;
    for n in 1..500u32 {
        term *= z / (2.0 * f64::from(n) + 1.0);
        sum += term;
        if term < 1e-18 * sum {
            break;
        }
    }
    let r = 2.0 / core::f64::consts::PI.sqrt() * a * (-a * a).exp() * sum;
    if x < 0.0 { -r } else { r }
}

/// The standard normal cumulative distribution, `0.5 * (1 + erf(x / sqrt(2)))`.
///
/// Accurate to roughly `1e-14` absolute. **In the far left tail it is accurate absolutely and not
/// relatively**: below about `x = -8` it returns exactly zero, because `1 + erf` cancels there. That
/// is fine for its use here — it is a spike probability, and a spike probability of `1e-16` and one
/// of `0` produce the same network — and it would not be fine for a rare-event calculation.
#[must_use]
pub fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / core::f64::consts::SQRT_2))
}

/// Fast sigmoid, the `SuperSpike` surrogate: `1 / (1 + beta |x|)^2`.
///
/// Zenke & Ganguli, Neural Comput. 30:1514-1541, 2018. It is the derivative of the *fast sigmoid*
/// `x / (1 + |x|)`, chosen there because it costs no exponential and because its tails are heavy —
/// they fall as `x^-2`, so a neuron far below threshold still receives a gradient. That is a real
/// property and not a rounding error: at ten thresholds below firing, this surrogate is about
/// `1e-4` of its peak while [`Gaussian`] is about `1e-87`.
///
/// **Peak-normalised, not mass-normalised**: `mass() == 2 / beta`, so at the common `beta = 100`
/// every gradient is scaled by `0.02` relative to a proper mollifier. Wrap it in
/// [`Scaled::unit_mass`] to see that factor explicitly instead of letting the learning rate carry
/// it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FastSigmoid {
    /// Sharpness, one over a threshold. Larger is narrower: `fwhm == 2 (sqrt(2) - 1) / beta`.
    /// Must be finite and strictly positive.
    pub beta: f64,
}

impl Default for FastSigmoid {
    /// `beta = 10`, a readable teaching default and **not** a published value. Zenke's `spytorch`
    /// reference implementation accompanying Neftci et al. 2019 uses `100`; Zenke & Vogels
    /// (Neural Comput. 33:899-925, 2021) report that the surrogate's *shape* matters little while
    /// its *scale* matters, which is why this crate reports [`Surrogate::fwhm`] rather than `beta`.
    fn default() -> Self {
        Self { beta: 10.0 }
    }
}

impl FastSigmoid {
    /// # Errors
    ///
    /// [`SurrogateError::NonFiniteParam`] or [`SurrogateError::NotPositive`] for a `beta` that is
    /// not a finite positive number.
    pub fn new(beta: f64) -> Result<Self, SurrogateError> {
        check_positive("beta", beta)?;
        Ok(Self { beta })
    }
}

impl Surrogate for FastSigmoid {
    fn name(&self) -> &'static str {
        "fast-sigmoid"
    }
    fn backward(&self, x: f64) -> f64 {
        let d = 1.0 + self.beta * x.abs();
        1.0 / (d * d)
    }
    fn antiderivative(&self, x: f64) -> f64 {
        let t = 1.0 / (1.0 + self.beta * x.abs());
        if x < 0.0 { t / self.beta } else { (2.0 - t) / self.beta }
    }
    fn mass(&self) -> f64 {
        2.0 / self.beta
    }
    fn peak(&self) -> f64 {
        1.0
    }
    fn fwhm(&self) -> f64 {
        2.0 * (core::f64::consts::SQRT_2 - 1.0) / self.beta
    }
    fn sharpened(&self, factor: f64) -> Option<Box<dyn Surrogate>> {
        ok_factor(factor).then(|| Box::new(Self { beta: self.beta * factor }) as Box<dyn Surrogate>)
    }
}

/// Arctangent surrogate: `alpha / (2 (1 + (pi alpha x / 2)^2))`.
///
/// The derivative of `1/2 + (1/pi) arctan(pi alpha x / 2)`, a Cauchy density in disguise. This is
/// the default in `SpikingJelly` (`surrogate.ATan`), which is where most published `PyTorch`
/// spiking results come from; `alpha = 2` is transcribed from that library's default and a reader
/// comparing against a current release should verify it rather than take this line for it.
///
/// **Normalised by construction**: `mass() == 1` exactly, for every `alpha`. Its tails are as heavy
/// as [`FastSigmoid`]'s — both fall as `x^-2` — which is the practical reason these two dominate
/// the literature.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArcTan {
    /// Sharpness, one over a threshold. `fwhm == 4 / (pi alpha)`. Finite and strictly positive.
    pub alpha: f64,
}

impl Default for ArcTan {
    /// `alpha = 2`, `SpikingJelly`'s documented default for `ATan`.
    fn default() -> Self {
        Self { alpha: 2.0 }
    }
}

impl ArcTan {
    /// # Errors
    ///
    /// As [`FastSigmoid::new`], for `alpha`.
    pub fn new(alpha: f64) -> Result<Self, SurrogateError> {
        check_positive("alpha", alpha)?;
        Ok(Self { alpha })
    }
}

impl Surrogate for ArcTan {
    fn name(&self) -> &'static str {
        "arctan"
    }
    fn backward(&self, x: f64) -> f64 {
        let b = core::f64::consts::FRAC_PI_2 * self.alpha * x;
        self.alpha / (2.0 * (1.0 + b * b))
    }
    fn antiderivative(&self, x: f64) -> f64 {
        0.5 + (core::f64::consts::FRAC_PI_2 * self.alpha * x).atan() / core::f64::consts::PI
    }
    fn mass(&self) -> f64 {
        1.0
    }
    fn peak(&self) -> f64 {
        self.alpha / 2.0
    }
    fn fwhm(&self) -> f64 {
        4.0 / (core::f64::consts::PI * self.alpha)
    }
    fn sharpened(&self, factor: f64) -> Option<Box<dyn Surrogate>> {
        ok_factor(factor)
            .then(|| Box::new(Self { alpha: self.alpha * factor }) as Box<dyn Surrogate>)
    }
}

/// The logistic derivative: `beta s(beta x) (1 - s(beta x))`, with `s(z) = 1 / (1 + exp(-z))`.
///
/// The oldest choice and the one a reader coming from ordinary neural networks expects: it is what
/// you get by replacing the step with a sigmoid and differentiating honestly. `SpikingJelly` ships
/// it as `surrogate.Sigmoid` with `alpha = 4`, transcribed here as the default.
///
/// **Normalised by construction**, since its antiderivative `s(beta x)` runs from 0 to 1. Its tails
/// are exponential, so a unit far from threshold gets essentially nothing — the "dead neuron"
/// problem is sharper with this family than with [`ArcTan`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SigmoidDeriv {
    /// Sharpness, one over a threshold. `fwhm == 4 ln(1 + sqrt(2)) / beta`, about `3.5255 / beta`.
    /// Finite and strictly positive.
    pub beta: f64,
}

impl Default for SigmoidDeriv {
    /// `beta = 4`, `SpikingJelly`'s documented default for its `Sigmoid` surrogate.
    fn default() -> Self {
        Self { beta: 4.0 }
    }
}

impl SigmoidDeriv {
    /// # Errors
    ///
    /// As [`FastSigmoid::new`], for `beta`.
    pub fn new(beta: f64) -> Result<Self, SurrogateError> {
        check_positive("beta", beta)?;
        Ok(Self { beta })
    }

    /// The logistic function itself, `1 / (1 + exp(-z))`, written so that the exponential is always
    /// taken of a non-positive argument and cannot overflow for large `|z|`.
    #[must_use]
    pub fn logistic(z: f64) -> f64 {
        if z >= 0.0 {
            1.0 / (1.0 + (-z).exp())
        } else {
            let e = z.exp();
            e / (1.0 + e)
        }
    }
}

impl Surrogate for SigmoidDeriv {
    fn name(&self) -> &'static str {
        "sigmoid-derivative"
    }
    fn backward(&self, x: f64) -> f64 {
        let s = Self::logistic(self.beta * x);
        self.beta * s * (1.0 - s)
    }
    fn antiderivative(&self, x: f64) -> f64 {
        Self::logistic(self.beta * x)
    }
    fn mass(&self) -> f64 {
        1.0
    }
    fn peak(&self) -> f64 {
        self.beta / 4.0
    }
    fn fwhm(&self) -> f64 {
        4.0 * (1.0 + core::f64::consts::SQRT_2).ln() / self.beta
    }
    fn sharpened(&self, factor: f64) -> Option<Box<dyn Surrogate>> {
        ok_factor(factor).then(|| Box::new(Self { beta: self.beta * factor }) as Box<dyn Surrogate>)
    }
}

/// Triangular pseudo-derivative: `peak * max(0, 1 - |x| / half_width)`.
///
/// Bellec et al., `NeurIPS` 2018, use exactly this as the pseudo-derivative of their LSNN units, with
/// a damping factor of `0.3` and a width of one threshold — the defaults here. Its compact support
/// is the point: a unit more than `half_width` from threshold contributes **exactly zero** gradient,
/// so the backward pass is as sparse as the forward pass, which is what makes it attractive for
/// on-chip learning.
///
/// **Peak-normalised, not mass-normalised**: `mass() == peak * half_width`, `0.3` at the defaults.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Triangular {
    /// Half-width of the support, in thresholds. `backward` is zero for `|x| >= half_width`, and
    /// `fwhm == half_width`. Finite and strictly positive.
    pub half_width: f64,
    /// Height at `x = 0`, dimensionless. Bellec et al.'s damping factor `gamma`. Finite and
    /// strictly positive.
    pub peak: f64,
}

impl Default for Triangular {
    /// `half_width = 1` threshold and `peak = 0.3`, the LSNN pseudo-derivative of Bellec et al.,
    /// `NeurIPS` 2018, in this module's threshold units.
    fn default() -> Self {
        Self { half_width: 1.0, peak: 0.3 }
    }
}

impl Triangular {
    /// # Errors
    ///
    /// As [`FastSigmoid::new`], for `half_width` and `peak`.
    pub fn new(half_width: f64, peak: f64) -> Result<Self, SurrogateError> {
        check_positive("half_width", half_width)?;
        check_positive("peak", peak)?;
        Ok(Self { half_width, peak })
    }

    /// The unit-mass triangle of the given half-width: `peak = 1 / half_width`.
    ///
    /// # Errors
    ///
    /// As [`Triangular::new`].
    pub fn unit_mass(half_width: f64) -> Result<Self, SurrogateError> {
        check_positive("half_width", half_width)?;
        Self::new(half_width, 1.0 / half_width)
    }
}

impl Surrogate for Triangular {
    fn name(&self) -> &'static str {
        "triangular"
    }
    fn backward(&self, x: f64) -> f64 {
        let t = 1.0 - x.abs() / self.half_width;
        if t > 0.0 { self.peak * t } else { 0.0 }
    }
    fn antiderivative(&self, x: f64) -> f64 {
        let h = self.half_width;
        let half = 0.5 * self.peak * h;
        if x <= -h {
            0.0
        } else if x >= h {
            2.0 * half
        } else if x < 0.0 {
            self.peak * (x + x * x / (2.0 * h) + 0.5 * h)
        } else {
            half + self.peak * (x - x * x / (2.0 * h))
        }
    }
    fn mass(&self) -> f64 {
        self.peak * self.half_width
    }
    fn peak(&self) -> f64 {
        self.peak
    }
    fn fwhm(&self) -> f64 {
        self.half_width
    }
    fn sharpened(&self, factor: f64) -> Option<Box<dyn Surrogate>> {
        ok_factor(factor).then(|| {
            Box::new(Self { half_width: self.half_width / factor, peak: self.peak })
                as Box<dyn Surrogate>
        })
    }
}

/// Rectangular (boxcar) surrogate: `1 / width` inside `|x| < width / 2`, zero outside.
///
/// The original STBP form, Wu et al., Front. Neurosci. 12:331, 2018, whose `h1(u)` is
/// `(1 / a1) * 1{|u - V_th| < a1 / 2}` — written there already divided by its width, so it is
/// **normalised by construction**: `mass() == 1` for every `width`. Of the eight families here it is
/// the cheapest to evaluate, a comparison and a constant.
///
/// The discontinuity is real and has a cost: the loss surface it induces is piecewise linear in the
/// membrane potential, so a gradient step can carry a unit straight across the window and receive no
/// gradient at all on the next step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rectangular {
    /// Full width of the window, in thresholds; `fwhm == width` and the height is `1 / width`.
    /// Finite and strictly positive.
    pub width: f64,
}

impl Default for Rectangular {
    /// `width = 1` threshold. Wu et al. treat `a1` as a tuned hyperparameter rather than publishing
    /// one value, so this is a round choice in their units and not a transcription.
    fn default() -> Self {
        Self { width: 1.0 }
    }
}

impl Rectangular {
    /// # Errors
    ///
    /// As [`FastSigmoid::new`], for `width`.
    pub fn new(width: f64) -> Result<Self, SurrogateError> {
        check_positive("width", width)?;
        Ok(Self { width })
    }
}

impl Surrogate for Rectangular {
    fn name(&self) -> &'static str {
        "rectangular"
    }
    fn backward(&self, x: f64) -> f64 {
        if x.abs() < 0.5 * self.width { 1.0 / self.width } else { 0.0 }
    }
    fn antiderivative(&self, x: f64) -> f64 {
        ((x + 0.5 * self.width) / self.width).clamp(0.0, 1.0)
    }
    fn mass(&self) -> f64 {
        1.0
    }
    fn peak(&self) -> f64 {
        1.0 / self.width
    }
    fn fwhm(&self) -> f64 {
        self.width
    }
    fn sharpened(&self, factor: f64) -> Option<Box<dyn Surrogate>> {
        ok_factor(factor)
            .then(|| Box::new(Self { width: self.width / factor }) as Box<dyn Surrogate>)
    }
}

/// The straight-through estimator: derivative identically `1` inside `|x| <= half_width`, zero
/// outside.
///
/// Bengio, Leonard & Courville, arXiv:1308.3432, 2013 — it predates the spiking literature by five
/// years and arrived there from quantised networks, where "pass the gradient through the
/// non-differentiable op unchanged, but only where the input was in range" is the standard trick.
/// The clipping window is Hubara et al.'s addition (binarised neural networks, `NeurIPS` 2016);
/// without it, training diverges because units far outside the window keep receiving full gradient.
///
/// **Not a density**: `mass() == 2 * half_width`, which is `2` at the default. It is the only family
/// here that is not even trying to be a mollifier, and it is included because it is what a great
/// deal of quantisation-aware training actually uses.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StraightThrough {
    /// Half-width of the pass-through window, in thresholds. `fwhm == 2 * half_width`. Finite and
    /// strictly positive.
    pub half_width: f64,
}

impl Default for StraightThrough {
    /// `half_width = 1`, the `|x| <= 1` clip of Hubara et al., `NeurIPS` 2016, in threshold units.
    fn default() -> Self {
        Self { half_width: 1.0 }
    }
}

impl StraightThrough {
    /// # Errors
    ///
    /// As [`FastSigmoid::new`], for `half_width`.
    pub fn new(half_width: f64) -> Result<Self, SurrogateError> {
        check_positive("half_width", half_width)?;
        Ok(Self { half_width })
    }
}

impl Surrogate for StraightThrough {
    fn name(&self) -> &'static str {
        "straight-through"
    }
    fn backward(&self, x: f64) -> f64 {
        if x.abs() <= self.half_width { 1.0 } else { 0.0 }
    }
    fn antiderivative(&self, x: f64) -> f64 {
        x.clamp(-self.half_width, self.half_width) + self.half_width
    }
    fn mass(&self) -> f64 {
        2.0 * self.half_width
    }
    fn peak(&self) -> f64 {
        1.0
    }
    fn fwhm(&self) -> f64 {
        2.0 * self.half_width
    }
    fn sharpened(&self, factor: f64) -> Option<Box<dyn Surrogate>> {
        ok_factor(factor).then(|| {
            Box::new(Self { half_width: self.half_width / factor }) as Box<dyn Surrogate>
        })
    }
}

/// Exponential (Laplace) surrogate: `(alpha / 2) exp(-alpha |x|)`.
///
/// The shape SLAYER uses (Shrestha & Orchard, `NeurIPS` 2018), whose spike-response derivative is
/// `(1 / alpha_s) exp(-beta_s |v - theta|)` — a two-parameter form that is this one times a scale.
/// [`Exponential::slayer`] builds that exact combination as a [`Scaled`] so the scale is visible
/// rather than folded in.
///
/// **Normalised by construction** in the one-parameter form: `mass() == 1`. Its kink at `x = 0` is
/// the sharpest peak of any smooth family here, which makes it the most aggressive at concentrating
/// credit on units that are exactly at threshold.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Exponential {
    /// Decay rate, one over a threshold. `fwhm == 2 ln(2) / alpha`. Finite and strictly positive.
    pub alpha: f64,
}

impl Default for Exponential {
    /// `alpha = 2`, giving a full width at half maximum of `ln(2)` thresholds. A round choice, not a
    /// transcription: SLAYER tunes its two parameters per task.
    fn default() -> Self {
        Self { alpha: 2.0 }
    }
}

impl Exponential {
    /// # Errors
    ///
    /// As [`FastSigmoid::new`], for `alpha`.
    pub fn new(alpha: f64) -> Result<Self, SurrogateError> {
        check_positive("alpha", alpha)?;
        Ok(Self { alpha })
    }

    /// SLAYER's two-parameter form `(1 / alpha_s) exp(-beta_s |x|)`, as a scaled Laplace.
    ///
    /// Its mass is `2 / (alpha_s beta_s)`, which is the number SLAYER's `alpha_s` is really
    /// setting. Returned as a [`Scaled`] so that number is on the outside of the object where
    /// [`Surrogate::mass`] reports it.
    ///
    /// # Errors
    ///
    /// As [`FastSigmoid::new`], for `alpha_s` and `beta_s`.
    pub fn slayer(alpha_s: f64, beta_s: f64) -> Result<Scaled, SurrogateError> {
        check_positive("alpha_s", alpha_s)?;
        let inner = Self::new(beta_s)?;
        Scaled::new(Box::new(inner), 2.0 / (alpha_s * beta_s))
    }
}

impl Surrogate for Exponential {
    fn name(&self) -> &'static str {
        "exponential"
    }
    fn backward(&self, x: f64) -> f64 {
        0.5 * self.alpha * (-self.alpha * x.abs()).exp()
    }
    fn antiderivative(&self, x: f64) -> f64 {
        if x < 0.0 {
            0.5 * (self.alpha * x).exp()
        } else {
            1.0 - 0.5 * (-self.alpha * x).exp()
        }
    }
    fn mass(&self) -> f64 {
        1.0
    }
    fn peak(&self) -> f64 {
        0.5 * self.alpha
    }
    fn fwhm(&self) -> f64 {
        2.0 * core::f64::consts::LN_2 / self.alpha
    }
    fn sharpened(&self, factor: f64) -> Option<Box<dyn Surrogate>> {
        ok_factor(factor)
            .then(|| Box::new(Self { alpha: self.alpha * factor }) as Box<dyn Surrogate>)
    }
}

/// Gaussian surrogate: `exp(-x^2 / (2 sigma^2)) / (sigma sqrt(2 pi))`.
///
/// The canonical mollifier, and the one with a probabilistic reading: it is exactly the escape-noise
/// interpretation, in which a neuron with a jittered threshold fires with probability
/// `normal_cdf(x / sigma)` and the "surrogate" is the honest derivative of that probability. Under
/// that reading the surrogate gradient stops being a heuristic and becomes the gradient of an
/// expectation — the argument Neftci et al. 2019 give for why the trick works at all.
///
/// Yin, Corradi & Bohte (Nature Machine Intelligence 3:905-913, 2021) use a multi-Gaussian variant
/// with a negative side lobe; this implementation ships the plain Gaussian, and this review did not
/// locate a single agreed parameterisation of the multi-Gaussian form to transcribe.
///
/// **Normalised by construction**: `mass() == 1`. Its tails are the lightest here by far, which is
/// the dead-neuron problem at its worst — at five sigma the gradient is `1e-6` of peak, at ten sigma
/// it underflows to zero.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Gaussian {
    /// Standard deviation, in thresholds. `fwhm == 2 sigma sqrt(2 ln 2)`, about `2.3548 sigma`.
    /// Finite and strictly positive.
    pub sigma: f64,
}

impl Default for Gaussian {
    /// `sigma = 0.5` thresholds, a round choice giving a full width at half maximum of about
    /// `1.18` thresholds — comparable to [`Rectangular::default`]. Not a transcription.
    fn default() -> Self {
        Self { sigma: 0.5 }
    }
}

impl Gaussian {
    /// # Errors
    ///
    /// As [`FastSigmoid::new`], for `sigma`.
    pub fn new(sigma: f64) -> Result<Self, SurrogateError> {
        check_positive("sigma", sigma)?;
        Ok(Self { sigma })
    }
}

impl Surrogate for Gaussian {
    fn name(&self) -> &'static str {
        "gaussian"
    }
    fn backward(&self, x: f64) -> f64 {
        let z = x / self.sigma;
        (-0.5 * z * z).exp() / (self.sigma * (2.0 * core::f64::consts::PI).sqrt())
    }
    fn antiderivative(&self, x: f64) -> f64 {
        normal_cdf(x / self.sigma)
    }
    fn mass(&self) -> f64 {
        1.0
    }
    fn peak(&self) -> f64 {
        1.0 / (self.sigma * (2.0 * core::f64::consts::PI).sqrt())
    }
    fn fwhm(&self) -> f64 {
        2.0 * self.sigma * (2.0 * core::f64::consts::LN_2).sqrt()
    }
    fn sharpened(&self, factor: f64) -> Option<Box<dyn Surrogate>> {
        ok_factor(factor)
            .then(|| Box::new(Self { sigma: self.sigma / factor }) as Box<dyn Surrogate>)
    }
}

/// Any surrogate multiplied by a constant gain.
///
/// The gain is the quantity the literature hides in the learning rate. [`Scaled::unit_mass`] makes
/// it explicit by choosing `gain = 1 / inner.mass()`, turning a peak-normalised family into a
/// genuine mollifier without changing its shape — after which two families can be compared at equal
/// mass and only their widths and tails differ.
#[derive(Debug)]
pub struct Scaled {
    /// The surrogate being scaled. Its shape, width and support are untouched.
    pub inner: Box<dyn Surrogate>,
    /// Multiplier applied to `backward` and `antiderivative`, dimensionless. Finite and non-zero;
    /// a gain of zero is a surrogate that trains nothing and is refused at construction.
    pub gain: f64,
}

impl Scaled {
    /// Scale `inner` by `gain`.
    ///
    /// # Errors
    ///
    /// [`SurrogateError::NonFiniteParam`] for a non-finite gain, [`SurrogateError::NotPositive`]
    /// for a gain that is not strictly positive — a negative gain would send every gradient step
    /// uphill, which is a bug rather than a configuration.
    pub fn new(inner: Box<dyn Surrogate>, gain: f64) -> Result<Self, SurrogateError> {
        check_positive("gain", gain)?;
        Ok(Self { inner, gain })
    }

    /// Renormalise `inner` to unit mass: `gain = 1 / inner.mass()`.
    ///
    /// # Errors
    ///
    /// [`SurrogateError::NonFiniteParam`] or [`SurrogateError::NotPositive`] if the inner mass is
    /// not a finite positive number, which would leave nothing to normalise by.
    pub fn unit_mass(inner: Box<dyn Surrogate>) -> Result<Self, SurrogateError> {
        let m = inner.mass();
        check_positive("inner mass", m)?;
        Self::new(inner, 1.0 / m)
    }
}

impl Surrogate for Scaled {
    fn name(&self) -> &'static str {
        "scaled"
    }
    fn backward(&self, x: f64) -> f64 {
        self.gain * self.inner.backward(x)
    }
    fn antiderivative(&self, x: f64) -> f64 {
        self.gain * self.inner.antiderivative(x)
    }
    fn mass(&self) -> f64 {
        self.gain * self.inner.mass()
    }
    fn peak(&self) -> f64 {
        self.gain * self.inner.peak()
    }
    fn fwhm(&self) -> f64 {
        // A vertical scale moves the peak and the half-maximum by the same factor, so the width
        // where `backward >= peak / 2` is unchanged. This is why `fwhm` is the comparable measure
        // of sharpness and `peak` is not.
        self.inner.fwhm()
    }
    fn sharpened(&self, factor: f64) -> Option<Box<dyn Surrogate>> {
        // Sharpening AT CONSTANT MASS, which is the mollifier semantics: whatever mass the inner
        // family loses under a horizontal rescale is put back into the gain. A `Scaled::unit_mass`
        // therefore stays unit-mass however far it is sharpened, which is the property
        // `a_normalised_surrogate_is_a_delta_sequence` asserts.
        let m0 = self.inner.mass();
        let inner = self.inner.sharpened(factor)?;
        let m1 = inner.mass();
        if !(m0.is_finite() && m1.is_finite() && m1 > 0.0) {
            return None;
        }
        Some(Box::new(Self { inner, gain: self.gain * m0 / m1 }))
    }
}

/// Reject a parameter that is not a finite positive number.
fn check_positive(what: &'static str, value: f64) -> Result<(), SurrogateError> {
    if !value.is_finite() {
        return Err(SurrogateError::NonFiniteParam { what, value });
    }
    if !(value > 0.0) {
        return Err(SurrogateError::NotPositive { what, value });
    }
    Ok(())
}

/// One of every published family, at its documented default.
///
/// Built for the module's own property tests — every assertion about mass, peak, width and
/// sharpening runs over this list — and for a lesson that wants to print the same table the module
/// doc contains, computed rather than typed.
#[must_use]
pub fn catalogue() -> Vec<Box<dyn Surrogate>> {
    vec![
        Box::new(FastSigmoid::default()),
        Box::new(ArcTan::default()),
        Box::new(SigmoidDeriv::default()),
        Box::new(Triangular::default()),
        Box::new(Rectangular::default()),
        Box::new(StraightThrough::default()),
        Box::new(Exponential::default()),
        Box::new(Gaussian::default()),
    ]
}

/// Panel count used by [`integrated_mass`] when a caller has no reason to choose one.
///
/// 400,000 panels puts the quadrature's own error below `1e-6` relative for the smooth families and
/// below `1e-4` for the two with a jump, which is a decade inside the `1e-3` tolerance the tests
/// use. Costs about a millisecond per surrogate.
pub const DEFAULT_PANELS: usize = 400_000;

/// Numerically integrate a surrogate's `backward` over the **whole** real line.
///
/// Midpoint rule under the substitution `x = tan(t)`, which maps `(-pi/2, pi/2)` onto the real line
/// and carries the Jacobian `1 + x^2`. The substitution is not a nicety: [`FastSigmoid`] and
/// [`ArcTan`] have tails that fall only as `x^-2`, so a truncated integral over `[-R, R]` misses a
/// fraction of order `1 / (beta R)` of the mass — at `beta = 10` and `R = 4` that is a 2.4% error,
/// which would read as a failed normalisation rather than as a failed integrator.
///
/// `panels` is rounded up to the next even number of at least two, so that the kink at `x = 0` that
/// [`FastSigmoid`], [`Triangular`] and [`Exponential`] all have lands on a panel boundary rather
/// than inside a panel, where it would cost an order of accuracy.
///
/// Accuracy: second order in the smooth case; first order, bounded by one panel's area, for the
/// step discontinuities of [`Rectangular`] and [`StraightThrough`].
#[must_use]
pub fn integrated_mass(s: &dyn Surrogate, panels: usize) -> f64 {
    let n = (panels.max(2) + 1) & !1;
    let h = core::f64::consts::PI / n as f64;
    let mut sum = 0.0;
    for k in 0..n {
        let t = -core::f64::consts::FRAC_PI_2 + (k as f64 + 0.5) * h;
        let x = t.tan();
        sum += s.backward(x) * (1.0 + x * x);
    }
    sum * h
}

/// Find the full width at half maximum by bisection, for checking [`Surrogate::fwhm`].
///
/// Assumes `backward` is non-increasing for `x > 0`, which holds for every family in this module.
/// `None` when `backward(0)` is not strictly positive, or when the half-maximum point is not found
/// below `x = 2^80`, either of which means the argument is not a mollifier and has no width.
#[must_use]
pub fn fwhm_numeric(s: &dyn Surrogate) -> Option<f64> {
    let peak = s.backward(0.0);
    if !(peak > 0.0) || !peak.is_finite() {
        return None;
    }
    let half = 0.5 * peak;
    let mut hi = 1e-3;
    let mut guard = 0u32;
    while s.backward(hi) >= half {
        hi *= 2.0;
        guard += 1;
        if guard > 80 {
            return None;
        }
    }
    let mut lo = 0.0;
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if s.backward(mid) >= half {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Some(lo + hi)
}

/// Which function stands in for the spike on the **forward** pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpikeFn {
    /// The real thing: [`heaviside`], a binary spike. This is what a network deployed on
    /// event-driven hardware runs, and the gradient computed through it is the surrogate gradient —
    /// biased, useful, and the only one available.
    Heaviside,
    /// The surrogate's own [`Surrogate::antiderivative`], giving a continuous "spike" in
    /// `[0, mass]`.
    ///
    /// Not a deployable network — it emits real numbers, so it buys none of the energy that makes
    /// spiking interesting. It exists so that the reverse-mode gradient in [`LifLayer::backward`]
    /// is the **exact** gradient of something, and can therefore be checked against finite
    /// differences. Every claim this module makes about its gradient machinery being correct is a
    /// claim made in this mode.
    Smooth,
}

/// Parameters of a recurrent LIF layer, in SI where the quantity is physical.
///
/// Converted once, at [`LifLayerSpec::build`], into the dimensionless decay factors the update
/// equations use. The conversion happens here and nowhere else, which is what keeps
/// [`LifLayer`]'s equations comparable line by line against Neftci et al. 2019 eq. (16)-(17).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LifLayerSpec {
    /// Number of input channels per time step. Must be at least one.
    pub n_in: usize,
    /// Number of recurrent spiking units. Must be at least one.
    pub n_rec: usize,
    /// Number of readout units, which is the number of classes. Must be at least one.
    pub n_out: usize,
    /// Simulation time step, **seconds**. Finite and strictly positive.
    pub dt: f64,
    /// Membrane time constant, **seconds**. Becomes `beta = exp(-dt / tau_mem)`.
    pub tau_mem: f64,
    /// Synaptic time constant, **seconds**. Becomes `alpha = exp(-dt / tau_syn)`. The second filter
    /// is what makes an input spike a smooth current rather than an instantaneous kick, and it is
    /// also what gives the gradient a path more than one step long.
    pub tau_syn: f64,
    /// Readout time constant, **seconds**. Becomes `kappa = exp(-dt / tau_out)`.
    pub tau_out: f64,
    /// Firing threshold, **dimensionless**, in the paper's frame. `1.0` unless you are reproducing
    /// something that uses another value; the surrogates' widths are quoted in units of it.
    pub theta: f64,
    /// Whether the recurrent weight block `V` is used.
    ///
    /// When `false` the block is still present in [`LifLayer::p`] and is held at zero: the forward
    /// pass ignores it and the gradient there is exactly zero, so a finite-difference probe and the
    /// analytic gradient agree on zero rather than disagreeing about an unused weight.
    pub recurrent: bool,
    /// Initial weight scale, dimensionless. Weights are drawn uniformly from
    /// `[-w_scale, w_scale] / sqrt(fan_in)`.
    ///
    /// The default `0.35` is `7 * (1 - beta)` at `tau_mem = 20 ms` and `dt = 1 ms`, the heuristic
    /// in Zenke's `spytorch` tutorial accompanying Neftci et al. 2019. That tutorial draws from a
    /// normal rather than a uniform; this crate uses a uniform because [`crate::rng::Rng`] provides
    /// one exactly and a Gaussian would need a transform whose tails are a second thing to verify.
    pub w_scale: f64,
    /// Initial bias on every unit's input current, dimensionless.
    ///
    /// **This is the dead-neuron control and it is the most consequential number in the spec.** A
    /// population that never reaches threshold has `backward(x) ~ 0` at every unit, so the gradient
    /// is zero, so it never starts learning and the loss curve is flat forever. A small positive
    /// bias puts the population near threshold at initialisation. The steady-state potential under
    /// a constant bias `b` with no other input is `b / ((1 - alpha) (1 - beta))`, so at the default
    /// time constants a bias of `0.004` sits at roughly half a threshold.
    pub b_init: f64,
    /// Seed for the weight draw. Same seed, same weights, every platform.
    pub seed: u64,
}

impl Default for LifLayerSpec {
    /// One input channel, sixteen recurrent units, two classes; `dt = 1 ms`, `tau_mem = 20 ms`,
    /// `tau_syn = 5 ms`, `tau_out = 20 ms`, threshold 1. These are the `spytorch` tutorial's time
    /// constants, which are themselves the textbook cortical values used throughout
    /// [`crate::neuron`].
    fn default() -> Self {
        Self {
            n_in: 1,
            n_rec: 16,
            n_out: 2,
            dt: 1e-3,
            tau_mem: 20e-3,
            tau_syn: 5e-3,
            tau_out: 20e-3,
            theta: 1.0,
            recurrent: true,
            w_scale: 0.35,
            b_init: 4e-3,
            seed: 1,
        }
    }
}

impl LifLayerSpec {
    /// Convert to decay factors, draw the weights and return the layer.
    ///
    /// # Errors
    ///
    /// [`SurrogateError::NotPositive`] or [`SurrogateError::NonFiniteParam`] for a time constant,
    /// time step, threshold or scale that is not a finite positive number; also
    /// [`SurrogateError::NotPositive`] with `what = "n_in"`, `"n_rec"` or `"n_out"` for a zero
    /// population, which has no state to integrate.
    pub fn build(&self) -> Result<LifLayer, SurrogateError> {
        for (what, n) in [("n_in", self.n_in), ("n_rec", self.n_rec), ("n_out", self.n_out)] {
            if n == 0 {
                return Err(SurrogateError::NotPositive { what, value: 0.0 });
            }
        }
        check_positive("dt", self.dt)?;
        check_positive("tau_mem", self.tau_mem)?;
        check_positive("tau_syn", self.tau_syn)?;
        check_positive("tau_out", self.tau_out)?;
        check_positive("theta", self.theta)?;
        if !self.w_scale.is_finite() {
            return Err(SurrogateError::NonFiniteParam { what: "w_scale", value: self.w_scale });
        }
        if !self.b_init.is_finite() {
            return Err(SurrogateError::NonFiniteParam { what: "b_init", value: self.b_init });
        }

        let mut layer = LifLayer {
            n_in: self.n_in,
            n_rec: self.n_rec,
            n_out: self.n_out,
            alpha: (-self.dt / self.tau_syn).exp(),
            beta: (-self.dt / self.tau_mem).exp(),
            kappa: (-self.dt / self.tau_out).exp(),
            theta: self.theta,
            recurrent: self.recurrent,
            p: vec![0.0; self.n_rec * self.n_in + self.n_rec * self.n_rec
                + self.n_out * self.n_rec + self.n_rec],
        };

        let mut rng = Rng::new(self.seed);
        let draw = |rng: &mut Rng, fan_in: usize| {
            (2.0 * rng.next_f64() - 1.0) * self.w_scale / (fan_in as f64).sqrt()
        };
        for j in 0..self.n_rec {
            for i in 0..self.n_in {
                let idx = layer.idx_w(j, i);
                layer.p[idx] = draw(&mut rng, self.n_in);
            }
        }
        if self.recurrent {
            for j in 0..self.n_rec {
                for k in 0..self.n_rec {
                    let idx = layer.idx_v(j, k);
                    layer.p[idx] = draw(&mut rng, self.n_rec);
                }
            }
        }
        for c in 0..self.n_out {
            for j in 0..self.n_rec {
                let idx = layer.idx_r(c, j);
                layer.p[idx] = draw(&mut rng, self.n_rec);
            }
        }
        for j in 0..self.n_rec {
            let idx = layer.idx_b(j);
            layer.p[idx] = self.b_init;
        }
        Ok(layer)
    }
}

/// One recurrent current-based LIF layer with a leaky readout, trainable through time.
///
/// The update, dimensionless, threshold `theta`, exactly Neftci et al. 2019 eq. (16)-(17) with a
/// bias added:
///
/// ```text
/// I[t] = alpha I[t-1] + W x[t] + V S[t-1] + b
/// U[t] = beta  U[t-1] + I[t] - theta S[t-1]        <- SOFT reset, by subtraction
/// S[t] = Theta(U[t] - theta)
/// Y[t] = kappa Y[t-1] + R S[t]                      <- non-spiking readout
/// ```
///
/// The reset is by **subtraction**, not by clamping to a reset potential. That is the choice the
/// surrogate-gradient literature makes and it is not cosmetic: a hard reset multiplies the state by
/// `(1 - S[t-1])`, which puts the non-differentiable spike variable into a *product* with the
/// membrane, and every implementation that does it detaches that factor from the graph and stops
/// reporting that it did. Subtraction keeps the whole recurrence differentiable given the
/// surrogate, which is why this layer's gradient can be checked exactly and a hard-reset layer's
/// cannot. It also means the membrane is not clamped during a refractory period; this layer has no
/// refractory period at all, unlike [`crate::neuron::Lif`].
///
/// The loss is softmax cross-entropy on the **time-averaged** readout, `logits[c] = mean_t Y[c][t]`,
/// which is the rate-coded readout the literature calls "mean over time". A max-over-time readout
/// is also common and gives a sparser gradient; this implementation ships the mean because its
/// gradient has a closed form that is one line and can be checked.
#[derive(Debug, Clone, PartialEq)]
pub struct LifLayer {
    /// Input channels per time step.
    pub n_in: usize,
    /// Recurrent spiking units.
    pub n_rec: usize,
    /// Readout units, one per class.
    pub n_out: usize,
    /// Synaptic decay per step, `exp(-dt / tau_syn)`, in `(0, 1)`.
    pub alpha: f64,
    /// Membrane decay per step, `exp(-dt / tau_mem)`, in `(0, 1)`.
    pub beta: f64,
    /// Readout decay per step, `exp(-dt / tau_out)`, in `(0, 1)`.
    pub kappa: f64,
    /// Firing threshold, dimensionless. The surrogate's argument is `U - theta`.
    pub theta: f64,
    /// Whether the `V` block participates. See [`LifLayerSpec::recurrent`].
    pub recurrent: bool,
    /// Flat parameter vector, in this order: `W` (`n_rec` by `n_in`, row-major), `V` (`n_rec` by
    /// `n_rec`), `R` (`n_out` by `n_rec`), `b` (`n_rec`). Use [`LifLayer::idx_w`],
    /// [`LifLayer::idx_v`], [`LifLayer::idx_r`] and [`LifLayer::idx_b`] rather than arithmetic.
    ///
    /// One flat vector rather than four arrays so that a gradient, a finite-difference probe and an
    /// optimiser's moment estimates are all the same shape and can be compared element by element —
    /// which is what makes `the_bptt_gradient_matches_central_finite_differences` a loop over one
    /// index instead of a traversal of four structures.
    pub p: Vec<f64>,
}

impl LifLayer {
    /// Total number of parameters, and the required length of a gradient.
    #[must_use]
    pub fn n_params(&self) -> usize {
        self.p.len()
    }

    /// Index of input weight `W[j][i]`, unit `j` from channel `i`.
    #[must_use]
    pub fn idx_w(&self, j: usize, i: usize) -> usize {
        j * self.n_in + i
    }

    /// Index of recurrent weight `V[j][k]`, into unit `j` from unit `k`.
    #[must_use]
    pub fn idx_v(&self, j: usize, k: usize) -> usize {
        self.n_rec * self.n_in + j * self.n_rec + k
    }

    /// Index of readout weight `R[c][j]`, into class `c` from unit `j`.
    #[must_use]
    pub fn idx_r(&self, c: usize, j: usize) -> usize {
        self.n_rec * self.n_in + self.n_rec * self.n_rec + c * self.n_rec + j
    }

    /// Index of bias `b[j]` on unit `j`'s input current.
    #[must_use]
    pub fn idx_b(&self, j: usize) -> usize {
        self.n_rec * self.n_in + self.n_rec * self.n_rec + self.n_out * self.n_rec + j
    }

    /// Run the network forward and keep every state it visited.
    ///
    /// `x` is `t_steps * n_in` flat, row-major in time. The whole trace is retained because
    /// backpropagation through time needs `U[t]` at every step to evaluate the surrogate on the way
    /// back — that storage, `O(t_steps * n_rec)`, is the memory cost of BPTT and the reason
    /// on-chip learning rules look for forward-mode alternatives instead.
    ///
    /// # Errors
    ///
    /// [`SurrogateError::ShapeMismatch`] if `x` is empty or its length is not a multiple of
    /// `n_in`; [`SurrogateError::NonFiniteInput`] naming the first non-finite element;
    /// [`SurrogateError::Diverged`] naming the first step and unit whose state left the finite
    /// numbers, which is what a too-large learning rate looks like from inside.
    pub fn forward(
        &self,
        sur: &dyn Surrogate,
        x: &[f64],
        spike_fn: SpikeFn,
    ) -> Result<Trace, SurrogateError> {
        if x.is_empty() || !x.len().is_multiple_of(self.n_in) {
            return Err(SurrogateError::ShapeMismatch {
                what: "input",
                got: x.len(),
                want: self.n_in,
            });
        }
        for (k, &v) in x.iter().enumerate() {
            if !v.is_finite() {
                return Err(SurrogateError::NonFiniteInput { index: k, value: v });
            }
        }
        let t_steps = x.len() / self.n_in;
        let (nr, no) = (self.n_rec, self.n_out);
        let mut tr = Trace {
            t_steps,
            n_rec: nr,
            n_out: no,
            i_syn: vec![0.0; t_steps * nr],
            u: vec![0.0; t_steps * nr],
            s: vec![0.0; t_steps * nr],
            y: vec![0.0; t_steps * no],
            logits: vec![0.0; no],
        };

        for t in 0..t_steps {
            for j in 0..nr {
                let mut drive = self.p[self.idx_b(j)];
                for i in 0..self.n_in {
                    drive += self.p[self.idx_w(j, i)] * x[t * self.n_in + i];
                }
                if self.recurrent && t > 0 {
                    for k in 0..nr {
                        drive += self.p[self.idx_v(j, k)] * tr.s[(t - 1) * nr + k];
                    }
                }
                let i_prev = if t > 0 { tr.i_syn[(t - 1) * nr + j] } else { 0.0 };
                let u_prev = if t > 0 { tr.u[(t - 1) * nr + j] } else { 0.0 };
                let s_prev = if t > 0 { tr.s[(t - 1) * nr + j] } else { 0.0 };
                let i_now = self.alpha * i_prev + drive;
                let u_now = self.beta * u_prev + i_now - self.theta * s_prev;
                if !u_now.is_finite() || !i_now.is_finite() {
                    let value = if i_now.is_finite() { u_now } else { i_now };
                    return Err(SurrogateError::Diverged { step: t, neuron: j, value });
                }
                tr.i_syn[t * nr + j] = i_now;
                tr.u[t * nr + j] = u_now;
                tr.s[t * nr + j] = match spike_fn {
                    SpikeFn::Heaviside => sur.forward(u_now - self.theta),
                    SpikeFn::Smooth => sur.antiderivative(u_now - self.theta),
                };
            }
            for c in 0..no {
                let y_prev = if t > 0 { tr.y[(t - 1) * no + c] } else { 0.0 };
                let mut acc = self.kappa * y_prev;
                for j in 0..nr {
                    acc += self.p[self.idx_r(c, j)] * tr.s[t * nr + j];
                }
                tr.y[t * no + c] = acc;
            }
        }
        for c in 0..no {
            let mut acc = 0.0;
            for t in 0..t_steps {
                acc += tr.y[t * no + c];
            }
            tr.logits[c] = acc / t_steps as f64;
        }
        Ok(tr)
    }

    /// Reverse-mode gradient of the loss with respect to [`LifLayer::p`], given `d_logits`.
    ///
    /// The whole of backpropagation through time for this layer, and it is short enough to read:
    /// four accumulators carried backwards in time (`gy` through the readout's leak, `gs` into the
    /// spike, `gu` through the membrane's leak, `gi` through the synapse's leak) and three
    /// outer-product accumulations. **The surrogate enters at exactly one line**, the multiplication
    /// by `sur.backward(U[t] - theta)`; everything else is ordinary calculus.
    ///
    /// Exact for [`SpikeFn::Smooth`]. For [`SpikeFn::Heaviside`] it is the surrogate gradient,
    /// which is not the gradient of anything that was run — see the module doc.
    ///
    /// # Errors
    ///
    /// [`SurrogateError::ShapeMismatch`] if `tr` was not produced by this layer or `d_logits` is
    /// not `n_out` long.
    pub fn backward(
        &self,
        sur: &dyn Surrogate,
        x: &[f64],
        tr: &Trace,
        d_logits: &[f64],
    ) -> Result<Vec<f64>, SurrogateError> {
        let (nr, no, t_steps) = (self.n_rec, self.n_out, tr.t_steps);
        if tr.n_rec != nr || tr.n_out != no || x.len() != t_steps * self.n_in {
            return Err(SurrogateError::ShapeMismatch {
                what: "trace",
                got: tr.n_rec,
                want: nr,
            });
        }
        if d_logits.len() != no {
            return Err(SurrogateError::ShapeMismatch {
                what: "d_logits",
                got: d_logits.len(),
                want: no,
            });
        }
        let mut g = vec![0.0; self.p.len()];
        let mut gy = vec![0.0; no];
        let mut gu_next = vec![0.0; nr];
        let mut gi_next = vec![0.0; nr];
        let mut gu = vec![0.0; nr];
        let mut gi = vec![0.0; nr];
        let inv_t = 1.0 / t_steps as f64;

        for t in (0..t_steps).rev() {
            // The readout leak, backwards: L sees Y[t] directly through the time average and
            // indirectly through Y[t+1] = kappa Y[t] + ...
            for c in 0..no {
                gy[c] = d_logits[c] * inv_t + self.kappa * gy[c];
            }
            for j in 0..nr {
                // Everything downstream of spike S[t]: the readout at t, the soft reset at t+1, and
                // the recurrent synapses at t+1.
                let mut gs = 0.0;
                for c in 0..no {
                    gs += gy[c] * self.p[self.idx_r(c, j)];
                }
                gs -= self.theta * gu_next[j];
                if self.recurrent {
                    for m in 0..nr {
                        gs += gi_next[m] * self.p[self.idx_v(m, j)];
                    }
                }
                // THE ONE SURROGATE LINE.
                let dsdu = sur.backward(tr.u[t * nr + j] - self.theta);
                gu[j] = gs * dsdu + self.beta * gu_next[j];
                gi[j] = gu[j] + self.alpha * gi_next[j];
            }
            for c in 0..no {
                for j in 0..nr {
                    let idx = self.idx_r(c, j);
                    g[idx] += gy[c] * tr.s[t * nr + j];
                }
            }
            for j in 0..nr {
                let gij = gi[j];
                for i in 0..self.n_in {
                    let idx = self.idx_w(j, i);
                    g[idx] += gij * x[t * self.n_in + i];
                }
                if self.recurrent && t > 0 {
                    for k in 0..nr {
                        let idx = self.idx_v(j, k);
                        g[idx] += gij * tr.s[(t - 1) * nr + k];
                    }
                }
                let idx = self.idx_b(j);
                g[idx] += gij;
            }
            gu_next.copy_from_slice(&gu);
            gi_next.copy_from_slice(&gi);
        }
        Ok(g)
    }

    /// Forward, loss and gradient for one pattern.
    ///
    /// # Errors
    ///
    /// As [`LifLayer::forward`], [`cross_entropy`] and [`LifLayer::backward`].
    pub fn loss_and_grad(
        &self,
        sur: &dyn Surrogate,
        x: &[f64],
        target: usize,
        spike_fn: SpikeFn,
    ) -> Result<(f64, Vec<f64>), SurrogateError> {
        let tr = self.forward(sur, x, spike_fn)?;
        let (loss, d_logits) = cross_entropy(&tr.logits, target)?;
        let g = self.backward(sur, x, &tr, &d_logits)?;
        Ok((loss, g))
    }

    /// Mean loss and mean gradient over a batch of `(input, target)` pairs.
    ///
    /// The mean, not the sum, so that a learning rate transfers between batch sizes.
    ///
    /// # Errors
    ///
    /// [`SurrogateError::EmptyBatch`] for an empty batch — a mean over nothing is undefined and
    /// zero is the wrong answer, because zero is also what a perfectly trained batch returns.
    /// Otherwise as [`LifLayer::loss_and_grad`].
    pub fn batch_loss_and_grad(
        &self,
        sur: &dyn Surrogate,
        batch: &[(Vec<f64>, usize)],
        spike_fn: SpikeFn,
    ) -> Result<(f64, Vec<f64>), SurrogateError> {
        if batch.is_empty() {
            return Err(SurrogateError::EmptyBatch);
        }
        let mut loss = 0.0;
        let mut g = vec![0.0; self.p.len()];
        for (x, target) in batch {
            let (l, gi) = self.loss_and_grad(sur, x, *target, spike_fn)?;
            loss += l;
            for (a, b) in g.iter_mut().zip(gi.iter()) {
                *a += *b;
            }
        }
        let inv = 1.0 / batch.len() as f64;
        loss *= inv;
        for a in &mut g {
            *a *= inv;
        }
        Ok((loss, g))
    }

    /// The class with the largest time-averaged readout, under the real Heaviside forward pass.
    ///
    /// Ties go to the lower class index, deterministically.
    ///
    /// # Errors
    ///
    /// As [`LifLayer::forward`].
    pub fn predict(&self, sur: &dyn Surrogate, x: &[f64]) -> Result<usize, SurrogateError> {
        let tr = self.forward(sur, x, SpikeFn::Heaviside)?;
        let mut best = 0;
        for c in 1..self.n_out {
            if tr.logits[c] > tr.logits[best] {
                best = c;
            }
        }
        Ok(best)
    }
}

/// Everything the forward pass visited, kept because the backward pass needs it.
///
/// All state arrays are flat and row-major in time: element `t * n_rec + j` (or `t * n_out + c`).
#[derive(Debug, Clone, PartialEq)]
pub struct Trace {
    /// Number of time steps, derived from the input length and `n_in`.
    pub t_steps: usize,
    /// Recurrent units, copied from the layer so the trace can be validated against it.
    pub n_rec: usize,
    /// Readout units, copied from the layer.
    pub n_out: usize,
    /// Synaptic current `I[t][j]`, dimensionless.
    pub i_syn: Vec<f64>,
    /// Membrane potential `U[t][j]`, dimensionless, in units of the threshold.
    pub u: Vec<f64>,
    /// Spike output `S[t][j]`: exactly `0.0` or `1.0` under [`SpikeFn::Heaviside`], and a real
    /// number in `[0, mass]` under [`SpikeFn::Smooth`].
    pub s: Vec<f64>,
    /// Readout trace `Y[t][c]`, dimensionless.
    pub y: Vec<f64>,
    /// Time-averaged readout, one per class. The argument to the softmax.
    pub logits: Vec<f64>,
}

impl Trace {
    /// Total spikes emitted over the whole trace.
    ///
    /// Meaningful only under [`SpikeFn::Heaviside`], where `s` is binary; under
    /// [`SpikeFn::Smooth`] it is a sum of real numbers and is not a spike count.
    #[must_use]
    pub fn spike_count(&self) -> f64 {
        self.s.iter().sum()
    }

    /// Mean firing rate in **hertz**, given the time step in seconds.
    ///
    /// `None` for a non-positive or non-finite `dt`, or an empty trace. This is the one place the
    /// layer's dimensionless interior is converted back to SI, and it is a conversion rather than a
    /// measurement: it assumes the spikes in the trace came from [`SpikeFn::Heaviside`].
    #[must_use]
    pub fn mean_rate(&self, dt: f64) -> Option<f64> {
        if !(dt > 0.0) || !dt.is_finite() || self.t_steps == 0 || self.n_rec == 0 {
            return None;
        }
        Some(self.spike_count() / (self.t_steps as f64 * self.n_rec as f64 * dt))
    }
}

/// Softmax cross-entropy and its gradient with respect to the logits.
///
/// Returns `(loss, d_loss/d_logits)` with `d_loss/d_logits[c] = softmax[c] - 1{c == target}`, the
/// closed form that makes the whole backward pass start from one subtraction.
///
/// Computed through a log-sum-exp shifted by the maximum logit, so that a logit of 800 gives a
/// finite loss rather than `inf - inf`.
///
/// # Errors
///
/// [`SurrogateError::TargetOutOfRange`] if `target` names a class the readout does not have;
/// [`SurrogateError::ShapeMismatch`] for empty logits; [`SurrogateError::NonFiniteInput`] naming
/// the first non-finite logit, because a `NaN` logit produces a `NaN` loss that then looks like a
/// diverged learning rate rather than like the upstream bug it is.
pub fn cross_entropy(logits: &[f64], target: usize) -> Result<(f64, Vec<f64>), SurrogateError> {
    if logits.is_empty() {
        return Err(SurrogateError::ShapeMismatch { what: "logits", got: 0, want: 1 });
    }
    if target >= logits.len() {
        return Err(SurrogateError::TargetOutOfRange { target, n_out: logits.len() });
    }
    for (k, &v) in logits.iter().enumerate() {
        if !v.is_finite() {
            return Err(SurrogateError::NonFiniteInput { index: k, value: v });
        }
    }
    let mut m = logits[0];
    for &v in logits {
        if v > m {
            m = v;
        }
    }
    let mut z = 0.0;
    for &v in logits {
        z += (v - m).exp();
    }
    let log_z = m + z.ln();
    let loss = log_z - logits[target];
    let mut d = Vec::with_capacity(logits.len());
    for (c, &v) in logits.iter().enumerate() {
        let p = (v - log_z).exp();
        d.push(if c == target { p - 1.0 } else { p });
    }
    Ok((loss, d))
}

/// Adam, hand-rolled and deterministic.
///
/// Kingma & Ba, "Adam: A Method for Stochastic Optimization", ICLR 2015. Here rather than plain
/// stochastic gradient descent for one specific reason: a surrogate gradient's **magnitude** is
/// arbitrary — it carries the surrogate's [`Surrogate::mass`] as a multiplicative factor, which
/// differs by a factor of fifty between [`FastSigmoid`] at `beta = 100` and [`Rectangular`] — and
/// Adam's per-parameter normalisation divides that factor out. With plain gradient descent, changing
/// the surrogate would require re-tuning the learning rate, and the comparison between surrogates
/// would be a comparison between learning rates.
///
/// No randomness, no clock: the same gradients in the same order give the same parameters, on every
/// platform.
#[derive(Debug, Clone, PartialEq)]
pub struct Adam {
    /// Step size. `1e-2` is a reasonable starting point for the small networks in this module.
    pub lr: f64,
    /// First-moment decay. Kingma & Ba's default is `0.9`.
    pub beta1: f64,
    /// Second-moment decay. Kingma & Ba's default is `0.999`.
    pub beta2: f64,
    /// Denominator floor, preventing a division by zero for a parameter whose gradient has been
    /// exactly zero — which happens constantly here, because a unit that never reaches the
    /// surrogate's support contributes nothing. Kingma & Ba's default is `1e-8`.
    pub eps: f64,
    m: Vec<f64>,
    v: Vec<f64>,
    t: u64,
}

impl Adam {
    /// An optimiser for `n_params` parameters at step size `lr`, with the paper's defaults.
    ///
    /// # Errors
    ///
    /// [`SurrogateError::NotPositive`] or [`SurrogateError::NonFiniteParam`] for a non-positive or
    /// non-finite `lr`, or for `n_params == 0`.
    pub fn new(n_params: usize, lr: f64) -> Result<Self, SurrogateError> {
        check_positive("lr", lr)?;
        if n_params == 0 {
            return Err(SurrogateError::NotPositive { what: "n_params", value: 0.0 });
        }
        Ok(Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            m: vec![0.0; n_params],
            v: vec![0.0; n_params],
            t: 0,
        })
    }

    /// Steps taken so far, which is the `t` in the paper's bias correction.
    #[must_use]
    pub fn steps(&self) -> u64 {
        self.t
    }

    /// Apply one update in place.
    ///
    /// # Errors
    ///
    /// [`SurrogateError::ShapeMismatch`] if `p` or `g` is not the length this optimiser was built
    /// for; [`SurrogateError::NonFiniteInput`] naming the first non-finite gradient element, which
    /// is refused rather than absorbed — a single `NaN` in `g` would otherwise poison `m` and `v`
    /// permanently and every subsequent parameter through them.
    pub fn step(&mut self, p: &mut [f64], g: &[f64]) -> Result<(), SurrogateError> {
        if p.len() != self.m.len() {
            return Err(SurrogateError::ShapeMismatch {
                what: "parameters",
                got: p.len(),
                want: self.m.len(),
            });
        }
        if g.len() != self.m.len() {
            return Err(SurrogateError::ShapeMismatch {
                what: "gradient",
                got: g.len(),
                want: self.m.len(),
            });
        }
        for (k, &v) in g.iter().enumerate() {
            if !v.is_finite() {
                return Err(SurrogateError::NonFiniteInput { index: k, value: v });
            }
        }
        self.t += 1;
        let bc1 = 1.0 - self.beta1.powi(i32::try_from(self.t).unwrap_or(i32::MAX));
        let bc2 = 1.0 - self.beta2.powi(i32::try_from(self.t).unwrap_or(i32::MAX));
        for k in 0..p.len() {
            self.m[k] = self.beta1 * self.m[k] + (1.0 - self.beta1) * g[k];
            self.v[k] = self.beta2 * self.v[k] + (1.0 - self.beta2) * g[k] * g[k];
            let mh = self.m[k] / bc1;
            let vh = self.v[k] / bc2;
            p[k] -= self.lr * mh / (vh.sqrt() + self.eps);
        }
        Ok(())
    }
}

/// The delayed-XOR task: a benchmark that a memoryless network cannot do.
///
/// One input channel. A cue may be present in an early window and may be present in a later one;
/// the label is the exclusive-or of the two presences, and by the time the second window arrives the
/// first is long over. The network must therefore (a) **remember** whether the first cue happened,
/// across a silent gap, and (b) combine the two answers **non-linearly** — a linear readout of a
/// leaky integrator can do neither.
///
/// The four patterns and their labels:
///
/// | first window | second window | class |
/// |---|---|---|
/// | absent | absent | 0 |
/// | absent | present | 1 |
/// | present | absent | 1 |
/// | present | present | 0 |
///
/// At initialisation a network gets two of the four right by construction, because the two classes
/// are balanced, and its cross-entropy sits near `ln(2) = 0.693`. That is the "impossible at
/// initialisation" baseline the module's learning test asserts against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DelayedXor {
    /// Total time steps in a pattern. Must exceed `2 * cue + gap`, leaving a silent tail during
    /// which the answer has to be held.
    pub t_steps: usize,
    /// Length of each cue window, in steps.
    pub cue: usize,
    /// Silent steps between the end of the first window and the start of the second.
    pub gap: usize,
}

impl Default for DelayedXor {
    /// 40 steps, 5-step cues, a 10-step gap: the second cue ends at step 20 and the remaining 20
    /// steps are silent. At `dt = 1 ms` and `tau_mem = 20 ms` that silent tail is one full membrane
    /// constant, so the answer cannot simply be sitting in the membrane at readout time.
    fn default() -> Self {
        Self { t_steps: 40, cue: 5, gap: 10 }
    }
}

impl DelayedXor {
    /// # Errors
    ///
    /// [`SurrogateError::ShapeMismatch`] if the windows do not fit inside `t_steps`, or if `cue` is
    /// zero — a zero-length cue makes all four patterns identical and the task unlearnable in a way
    /// that would read as a training failure.
    pub fn new(t_steps: usize, cue: usize, gap: usize) -> Result<Self, SurrogateError> {
        if cue == 0 || 2 * cue + gap > t_steps {
            return Err(SurrogateError::ShapeMismatch {
                what: "delayed-xor windows",
                got: 2 * cue + gap,
                want: t_steps,
            });
        }
        Ok(Self { t_steps, cue, gap })
    }

    /// Input channels the task produces: always 1.
    #[must_use]
    pub fn n_in(&self) -> usize {
        1
    }

    /// Classes the task produces: always 2.
    #[must_use]
    pub fn n_out(&self) -> usize {
        2
    }

    /// All four patterns, each a `t_steps`-long input and its class, in a fixed order.
    ///
    /// The cue amplitude is `1.0` and the silent value is `0.0`. Deterministic and allocation-only:
    /// there is no noise in this task, so a failure to learn it is a failure of the learning rule
    /// rather than a sampling accident.
    #[must_use]
    pub fn patterns(&self) -> Vec<(Vec<f64>, usize)> {
        let mut out = Vec::with_capacity(4);
        for a in [false, true] {
            for b in [false, true] {
                let mut x = vec![0.0; self.t_steps];
                if a {
                    for t in 0..self.cue {
                        x[t] = 1.0;
                    }
                }
                if b {
                    let start = self.cue + self.gap;
                    for t in start..start + self.cue {
                        x[t] = 1.0;
                    }
                }
                out.push((x, usize::from(a != b)));
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Adam, ArcTan, DelayedXor, Exponential, FastSigmoid, Gaussian, LifLayer, LifLayerSpec,
        Rectangular, Scaled, SigmoidDeriv, SpikeFn, StraightThrough, Surrogate, SurrogateError,
        Triangular, catalogue, cross_entropy, erf, fwhm_numeric, heaviside, integrated_mass,
        normal_cdf, DEFAULT_PANELS,
    };

    /// The forward pass is the same step function for every family. If this ever fails, the module's
    /// premise has been quietly abandoned somewhere.
    #[test]
    fn every_surrogate_runs_the_identical_heaviside_forward() {
        for s in catalogue() {
            for k in -40..=40 {
                let x = f64::from(k) * 0.1;
                let got = s.forward(x);
                let want = if x >= 0.0 { 1.0 } else { 0.0 };
                assert!(
                    (got - want).abs() < 1e-15,
                    "{}: forward({x}) = {got}, want {want}",
                    s.name()
                );
                assert!(got == 0.0 || got == 1.0, "{}: forward is not binary", s.name());
            }
        }
        assert!((heaviside(0.0) - 1.0).abs() < 1e-15, "reaching threshold must fire");
        assert!(heaviside(-1e-300) == 0.0);
    }

    /// (a) EVERY surrogate integrates to the mass it claims, over the whole real line.
    ///
    /// The quadrature is the tan-substitution midpoint rule, which is what makes this check
    /// possible at all for the two families whose tails fall as `x^-2`: truncating at `|x| = 4`
    /// loses 2.4% of the fast sigmoid's mass, which is 24 times the tolerance here.
    #[test]
    fn every_surrogate_integrates_to_its_analytic_mass() {
        for s in catalogue() {
            let got = integrated_mass(s.as_ref(), DEFAULT_PANELS);
            let want = s.mass();
            let rel = (got - want).abs() / want;
            assert!(rel < 1e-3, "{}: quadrature {got} vs analytic mass {want}", s.name());
        }
    }

    /// (a) continued: the five families that claim to be normalised really do integrate to one, and
    /// the three that do not are off by exactly the factor their `mass` reports.
    #[test]
    fn the_normalised_families_integrate_to_one_and_the_others_do_not() {
        let normalised = ["arctan", "sigmoid-derivative", "rectangular", "exponential", "gaussian"];
        for s in catalogue() {
            let m = integrated_mass(s.as_ref(), DEFAULT_PANELS);
            if normalised.contains(&s.name()) {
                assert!((m - 1.0).abs() < 1e-3, "{} claims unit mass but integrates to {m}", s.name());
            } else {
                assert!(
                    (m - 1.0).abs() > 1e-2,
                    "{} is documented as un-normalised but integrates to {m}",
                    s.name()
                );
            }
            // ... and renormalising any of them lands on one.
            let scaled = Scaled::unit_mass(s).expect("positive finite mass");
            let m2 = integrated_mass(&scaled, DEFAULT_PANELS);
            assert!((m2 - 1.0).abs() < 1e-3, "renormalised surrogate integrates to {m2}");
        }
    }

    /// The antiderivative is an independent statement of the same integral, in closed form. Two
    /// checks: its total rise equals `mass`, and its slope equals `backward` pointwise.
    #[test]
    fn the_antiderivative_is_the_integral_of_the_backward_pass() {
        for s in catalogue() {
            let big = 1e9;
            let rise = s.antiderivative(big) - s.antiderivative(-big);
            let rel = (rise - s.mass()).abs() / s.mass();
            assert!(rel < 1e-6, "{}: Phi rises by {rise}, mass is {}", s.name(), s.mass());

            // Slope, by central differences, away from the kinks and jumps every family has at
            // x = 0 and at the edges of its support. The offsets below are deliberately irrational
            // multiples of the widths so they cannot land on an edge.
            let h = 1e-6;
            for k in 1..=12 {
                // Offsets chosen to miss every kink and support edge in the catalogue, which sit
                // at 0, +/-0.5 and +/-1: the first draft used a stride that landed exactly on -1,
                // the straight-through estimator's edge, and the central difference there read the
                // average of 1 and 0 rather than either.
                let x = 0.41_f64 * f64::from(k) - 2.07;
                let fd = (s.antiderivative(x + h) - s.antiderivative(x - h)) / (2.0 * h);
                let an = s.backward(x);
                assert!(
                    (fd - an).abs() < 1e-4 * (1.0 + an.abs()),
                    "{}: d/dx Phi({x}) = {fd} but backward = {an}",
                    s.name()
                );
            }
        }
    }

    /// (c) Non-negative everywhere, and the maximum is at threshold. A surrogate that peaked
    /// somewhere else would be crediting the wrong neurons.
    #[test]
    fn every_surrogate_is_non_negative_and_peaks_at_zero() {
        for s in catalogue() {
            let p0 = s.backward(0.0);
            assert!(p0 > 0.0, "{}: backward(0) = {p0}", s.name());
            for k in -2000..=2000 {
                let x = f64::from(k) * 0.005;
                let v = s.backward(x);
                assert!(v >= 0.0, "{}: backward({x}) = {v} is negative", s.name());
                assert!(v <= p0 + 1e-15, "{}: backward({x}) = {v} exceeds the peak {p0}", s.name());
            }
        }
    }

    /// The closed-form peak against the implementation. Different expressions, same number.
    #[test]
    fn the_closed_form_peak_matches_the_implementation() {
        for s in catalogue() {
            let a = s.peak();
            let b = s.backward(0.0);
            assert!((a - b).abs() < 1e-14 * (1.0 + a.abs()), "{}: peak {a} vs backward(0) {b}", s.name());
        }
    }

    /// (b) part one: the closed-form width against a bisection search for the half-maximum point.
    #[test]
    fn the_closed_form_fwhm_matches_a_bisection_search() {
        for s in catalogue() {
            let want = s.fwhm();
            let got = fwhm_numeric(s.as_ref()).expect("every family here has a width");
            assert!(
                (got - want).abs() < 1e-9 * (1.0 + want),
                "{}: bisection {got} vs closed form {want}",
                s.name()
            );
        }
    }

    /// (b) part two: THE WIDTH SHRINKS AS 1/factor. This is what "approaches the Heaviside's
    /// derivative" means, made into a number.
    #[test]
    fn sharpening_shrinks_the_width_in_exact_proportion() {
        for s in catalogue() {
            let w0 = s.fwhm();
            for factor in [2.0, 4.0, 16.0, 256.0, 4096.0] {
                let sharp = s.sharpened(factor).expect("a positive finite factor");
                let want = w0 / factor;
                let got = sharp.fwhm();
                assert!(
                    (got - want).abs() < 1e-12 * (1.0 + want),
                    "{} sharpened by {factor}: width {got}, want {want}",
                    s.name()
                );
                // And the closed form still agrees with a search at the new width.
                let num = fwhm_numeric(sharp.as_ref()).expect("still a mollifier");
                assert!((num - got).abs() < 1e-9 * (1.0 + got), "{}: {num} vs {got}", s.name());
            }
        }
    }

    /// (b) part three: a normalised surrogate sharpened without limit IS a delta sequence — unit
    /// mass forever, zero at any fixed non-zero point, and all of its mass inside a shrinking
    /// window. That is the precise sense in which every surrogate here approximates `delta`.
    #[test]
    fn a_normalised_surrogate_is_a_delta_sequence() {
        for s in catalogue() {
            let name = s.name();
            let base = Scaled::unit_mass(s).expect("positive finite mass");
            let mut prev_at_x0 = f64::INFINITY;
            let x0 = 0.25; // a fixed point away from threshold
            for factor in [1.0, 10.0, 100.0, 1000.0] {
                let d = base.sharpened(factor).expect("positive factor");
                let m = integrated_mass(d.as_ref(), DEFAULT_PANELS);
                // The tolerance is the QUADRATURE's own error bound, not a fudge. A step
                // discontinuity costs at most one panel's area per edge, `h * peak`, and a
                // thousandfold-sharpened boxcar has a peak of 1000 — so at 400,000 panels the
                // integrator cannot do better than about 2.4e-2 there, while the smooth families
                // stay at 1e-6. Asserting a flat 2e-3 would have been asserting that the integrator
                // is better than it is.
                let tol = 2e-3_f64.max(3.0 * core::f64::consts::PI / DEFAULT_PANELS as f64 * d.peak());
                assert!((m - 1.0).abs() < tol, "{name} at factor {factor}: mass {m}, tol {tol}");
                let v = d.backward(x0);
                // Non-INCREASING rather than strictly falling: the light-tailed families underflow
                // to exactly 0.0 well before the last factor here (the logistic derivative is
                // already 0 at factor 100), and demanding a strict fall would assert that 0 < 0.
                assert!(v <= prev_at_x0, "{name} at factor {factor}: backward({x0}) rose");
                if factor > 1.0 {
                    assert!(v < base.backward(x0), "{name} at factor {factor}: no narrowing at all");
                }
                prev_at_x0 = v;
                assert!(d.peak() > base.peak() * 0.99 * factor, "{name}: peak did not grow");
            }
            // By a thousandfold sharpening, what is left away from threshold is at most a
            // hundredth of what was there — RELATIVE, because the two heavy-tailed families fall
            // only as `x^-2` and a fixed absolute bound would pass for the Gaussian, which
            // underflows to zero here, while failing for the arctangent at 1.6e-3.
            let d = base.sharpened(1000.0).expect("positive factor");
            assert!(
                d.backward(x0) < 0.01 * base.backward(x0),
                "{name}: {} at x0, from {}",
                d.backward(x0),
                base.backward(x0)
            );
        }
    }

    /// `sharpened` refuses what it cannot interpret rather than returning the unsharpened surrogate,
    /// which would look exactly like a successful call.
    #[test]
    fn sharpening_refuses_a_non_positive_or_non_finite_factor() {
        for s in catalogue() {
            for bad in [0.0, -1.0, -0.5, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                assert!(s.sharpened(bad).is_none(), "{} accepted factor {bad}", s.name());
            }
        }
    }

    /// Constructors refuse a parameter that cannot describe a surrogate, naming it.
    #[test]
    fn a_non_finite_or_non_positive_parameter_is_refused_by_name() {
        // `assert_eq!` cannot be used on the NaN case: NaN != NaN, so an equality assertion there
        // fails on a correct answer. The variant and its field name are what is being checked.
        assert!(matches!(
            FastSigmoid::new(f64::NAN),
            Err(SurrogateError::NonFiniteParam { what: "beta", value }) if value.is_nan()
        ));
        assert_eq!(
            ArcTan::new(0.0),
            Err(SurrogateError::NotPositive { what: "alpha", value: 0.0 })
        );
        assert_eq!(
            Gaussian::new(-1.0),
            Err(SurrogateError::NotPositive { what: "sigma", value: -1.0 })
        );
        assert_eq!(
            Rectangular::new(0.0),
            Err(SurrogateError::NotPositive { what: "width", value: 0.0 })
        );
        assert_eq!(
            Triangular::new(1.0, 0.0),
            Err(SurrogateError::NotPositive { what: "peak", value: 0.0 })
        );
        assert_eq!(
            StraightThrough::new(f64::INFINITY),
            Err(SurrogateError::NonFiniteParam { what: "half_width", value: f64::INFINITY })
        );
        assert_eq!(
            SigmoidDeriv::new(-2.0),
            Err(SurrogateError::NotPositive { what: "beta", value: -2.0 })
        );
        assert_eq!(
            Exponential::new(0.0),
            Err(SurrogateError::NotPositive { what: "alpha", value: 0.0 })
        );
    }

    /// SLAYER's two-parameter form really is a scaled Laplace, and the scale is the number its
    /// `alpha_s` was setting.
    #[test]
    fn the_slayer_form_is_a_scaled_laplace_of_the_documented_mass() {
        let (a_s, b_s) = (5.0, 3.0);
        let s = Exponential::slayer(a_s, b_s).expect("positive parameters");
        // (1 / alpha_s) exp(-beta_s |x|), checked pointwise against the published expression.
        for k in -20..=20 {
            let x = f64::from(k) * 0.1;
            let want = (1.0 / a_s) * (-b_s * x.abs()).exp();
            assert!((s.backward(x) - want).abs() < 1e-14, "at {x}");
        }
        let want_mass = 2.0 / (a_s * b_s);
        assert!((s.mass() - want_mass).abs() < 1e-14);
        let num = integrated_mass(&s, DEFAULT_PANELS);
        assert!((num - want_mass).abs() < 1e-3 * want_mass, "quadrature {num} vs {want_mass}");
    }

    /// The error function against published values. Four anchors across the range where the series
    /// works hardest.
    #[test]
    fn erf_matches_published_values() {
        let cases = [
            (0.0, 0.0),
            (0.5, 0.520_499_877_813_046_5),
            (1.0, 0.842_700_792_949_714_9),
            (2.0, 0.995_322_265_018_952_7),
            (3.0, 0.999_977_909_503_001_4),
        ];
        for (x, want) in cases {
            let got = erf(x);
            assert!((got - want).abs() < 1e-12, "erf({x}) = {got}, want {want}");
            assert!((erf(-x) + want).abs() < 1e-12, "erf is not odd at {x}");
        }
        assert!((erf(7.0) - 1.0).abs() < 1e-15);
        assert!(erf(f64::NAN).is_nan());
    }

    /// The normal CDF against published values, and its derivative against the density.
    #[test]
    fn the_normal_cdf_matches_published_values() {
        let cases = [
            (0.0, 0.5),
            (1.0, 0.841_344_746_068_542_9),
            (-1.0, 0.158_655_253_931_457_1),
            (2.0, 0.977_249_868_051_820_8),
        ];
        for (x, want) in cases {
            let got = normal_cdf(x);
            assert!((got - want).abs() < 1e-12, "Phi({x}) = {got}, want {want}");
        }
        // Its slope is the standard normal density, which is `Gaussian { sigma: 1 }`.
        let g = Gaussian::new(1.0).expect("positive sigma");
        let h = 1e-6;
        for k in -30..=30 {
            let x = f64::from(k) * 0.1;
            let fd = (normal_cdf(x + h) - normal_cdf(x - h)) / (2.0 * h);
            assert!((fd - g.backward(x)).abs() < 1e-8, "at {x}: {fd} vs {}", g.backward(x));
        }
    }

    /// Cross-entropy on equal logits is `ln(n)` exactly, and its gradient sums to zero because the
    /// softmax's probabilities sum to one. Both are closed forms, not regression values.
    #[test]
    fn cross_entropy_matches_its_closed_form_on_uniform_logits() {
        for n in 2..=8usize {
            let logits = vec![0.37; n];
            let (loss, d) = cross_entropy(&logits, 0).expect("valid target");
            let want = (n as f64).ln();
            assert!((loss - want).abs() < 1e-14, "n = {n}: loss {loss}, want ln(n) = {want}");
            let sum: f64 = d.iter().sum();
            assert!(sum.abs() < 1e-14, "gradient sums to {sum}, not zero");
            let inv = 1.0 / n as f64;
            assert!((d[0] - (inv - 1.0)).abs() < 1e-14);
            assert!((d[1] - inv).abs() < 1e-14);
        }
        // Shifted logits give the identical loss: log-sum-exp is shift-invariant, which is the
        // property the max-subtraction is there to exploit and could silently break.
        let a = cross_entropy(&[1.0, 2.0, 3.0], 2).expect("valid").0;
        let b = cross_entropy(&[801.0, 802.0, 803.0], 2).expect("valid").0;
        assert!((a - b).abs() < 1e-12, "{a} vs {b}");
        assert!(b.is_finite(), "log-sum-exp overflowed");
    }

    /// Targets and shapes that have no answer are refused, naming what was wrong.
    #[test]
    fn a_target_or_a_shape_that_has_no_answer_is_refused() {
        assert_eq!(
            cross_entropy(&[0.0, 1.0], 5),
            Err(SurrogateError::TargetOutOfRange { target: 5, n_out: 2 })
        );
        assert!(matches!(
            cross_entropy(&[0.0, f64::NAN], 0),
            Err(SurrogateError::NonFiniteInput { index: 1, .. })
        ));
        let layer = LifLayerSpec::default().build().expect("valid spec");
        let sur = ArcTan::default();
        assert!(matches!(
            layer.forward(&sur, &[], SpikeFn::Heaviside),
            Err(SurrogateError::ShapeMismatch { .. })
        ));
        assert!(matches!(
            layer.forward(&sur, &[0.0, f64::INFINITY, 0.0], SpikeFn::Heaviside),
            Err(SurrogateError::NonFiniteInput { index: 1, .. })
        ));
        assert!(matches!(
            layer.batch_loss_and_grad(&sur, &[], SpikeFn::Heaviside),
            Err(SurrogateError::EmptyBatch)
        ));
        let bad = LifLayerSpec { tau_mem: 0.0, ..LifLayerSpec::default() };
        assert_eq!(
            bad.build().err(),
            Some(SurrogateError::NotPositive { what: "tau_mem", value: 0.0 })
        );
        let bad = LifLayerSpec { n_rec: 0, ..LifLayerSpec::default() };
        assert_eq!(
            bad.build().err(),
            Some(SurrogateError::NotPositive { what: "n_rec", value: 0.0 })
        );
    }

    /// The layer's linear machinery against its closed form. With every weight zero and a threshold
    /// no membrane can reach, the two-stage filter is a pure double geometric and has an exact
    /// solution, so this pins `alpha` and `beta` to the right recurrences — a transposition of the
    /// two would leave every other test in this module green.
    ///
    /// `I[t] = b (1 - alpha^(t+1)) / (1 - alpha)` and
    /// `U[t] = b/(1-alpha) * [ (1 - beta^(t+1))/(1 - beta) - alpha (beta^(t+1) - alpha^(t+1))/(beta - alpha) ]`.
    #[test]
    fn the_two_stage_filter_matches_its_closed_form() {
        let spec = LifLayerSpec {
            n_in: 1,
            n_rec: 1,
            n_out: 1,
            theta: 1e9,
            b_init: 0.0,
            w_scale: 0.0,
            ..LifLayerSpec::default()
        };
        let mut layer = spec.build().expect("valid spec");
        let b = 0.017;
        let idx = layer.idx_b(0);
        layer.p[idx] = b;
        let (alpha, beta) = (layer.alpha, layer.beta);
        assert!((alpha - (-1e-3 / 5e-3_f64).exp()).abs() < 1e-15, "alpha is not exp(-dt/tau_syn)");
        assert!((beta - (-1e-3 / 20e-3_f64).exp()).abs() < 1e-15, "beta is not exp(-dt/tau_mem)");

        // 400 steps is 20 membrane constants, so `beta^400 = 2e-9` and the steady-state check
        // below is comparing against the limit rather than against a partly-charged membrane.
        let t_steps = 400;
        let x = vec![0.0; t_steps];
        let sur = ArcTan::default();
        let tr = layer.forward(&sur, &x, SpikeFn::Heaviside).expect("valid input");
        assert!(tr.spike_count() == 0.0, "nothing may spike below a threshold of 1e9");

        for t in 0..t_steps {
            let n = i32::try_from(t).expect("small") + 1;
            let want_i = b * (1.0 - alpha.powi(n)) / (1.0 - alpha);
            let want_u = b / (1.0 - alpha)
                * ((1.0 - beta.powi(n)) / (1.0 - beta)
                    - alpha * (beta.powi(n) - alpha.powi(n)) / (beta - alpha));
            assert!(
                (tr.i_syn[t] - want_i).abs() < 1e-13 * (1.0 + want_i.abs()),
                "step {t}: I = {}, closed form {want_i}",
                tr.i_syn[t]
            );
            assert!(
                (tr.u[t] - want_u).abs() < 1e-12 * (1.0 + want_u.abs()),
                "step {t}: U = {}, closed form {want_u}",
                tr.u[t]
            );
        }
        // The steady state quoted in `LifLayerSpec::b_init`'s doc, which is the number that decides
        // whether a population is alive at initialisation.
        let ss = b / ((1.0 - alpha) * (1.0 - beta));
        assert!(
            (tr.u[t_steps - 1] - ss).abs() / ss < 1e-6,
            "U reached {} rather than b/((1-alpha)(1-beta)) = {ss}",
            tr.u[t_steps - 1]
        );
    }

    /// A deterministic, reproducible input for the gradient checks: nothing random, nothing
    /// symmetric, and non-zero on every channel at some step.
    fn probe_input(t_steps: usize, n_in: usize) -> Vec<f64> {
        let mut x = vec![0.0; t_steps * n_in];
        for t in 0..t_steps {
            for i in 0..n_in {
                let k = (t * n_in + i) as f64;
                x[t * n_in + i] = 0.5 + 0.5 * (0.7 * k).sin();
            }
        }
        x
    }

    /// (d) THE CHECK THAT MAKES THIS MODULE TRUSTWORTHY.
    ///
    /// In [`SpikeFn::Smooth`] the network is genuinely differentiable and the module's reverse-mode
    /// pass is its exact gradient, so central finite differences have something to agree with. They
    /// agree to a relative `1e-6` on every one of the layer's parameters — input weights, recurrent
    /// weights, readout weights and biases — over a six-step unroll.
    ///
    /// Run on the three families whose antiderivative is infinitely differentiable. The others are
    /// checked at a looser tolerance in the next test, for the reason stated there.
    #[test]
    fn the_bptt_gradient_matches_central_finite_differences() {
        let spec = LifLayerSpec {
            n_in: 2,
            n_rec: 4,
            n_out: 3,
            recurrent: true,
            w_scale: 0.6,
            b_init: 0.05,
            seed: 20260917,
            ..LifLayerSpec::default()
        };
        let layer = spec.build().expect("valid spec");
        let x = probe_input(6, 2);
        let surs: Vec<Box<dyn Surrogate>> = vec![
            Box::new(ArcTan::default()),
            Box::new(SigmoidDeriv::default()),
            Box::new(Gaussian::default()),
        ];
        for sur in &surs {
            for target in 0..3 {
                let (_, g) = layer
                    .loss_and_grad(sur.as_ref(), &x, target, SpikeFn::Smooth)
                    .expect("valid");
                let h = 1e-6;
                let mut worst = 0.0_f64;
                for k in 0..layer.n_params() {
                    let mut up = layer.clone();
                    up.p[k] += h;
                    let mut dn = layer.clone();
                    dn.p[k] -= h;
                    let lu = up
                        .loss_and_grad(sur.as_ref(), &x, target, SpikeFn::Smooth)
                        .expect("valid")
                        .0;
                    let ld = dn
                        .loss_and_grad(sur.as_ref(), &x, target, SpikeFn::Smooth)
                        .expect("valid")
                        .0;
                    let fd = (lu - ld) / (2.0 * h);
                    let err = (fd - g[k]).abs() / (1.0 + g[k].abs());
                    worst = worst.max(err);
                    assert!(
                        err < 1e-6,
                        "{} param {k} target {target}: analytic {} vs finite difference {fd}",
                        sur.name(),
                        g[k]
                    );
                }
                assert!(worst > 0.0, "the finite differences were all exactly zero");
            }
        }
    }

    /// (d) continued: the same check on the five remaining families, at `1e-4`.
    ///
    /// Looser on purpose and the reason is in the maths, not in the code. [`FastSigmoid`],
    /// [`Exponential`] and [`Triangular`] have a kink at `x = 0`, and [`Rectangular`] and
    /// [`StraightThrough`] have a jump at the edge of their support, so their antiderivatives are
    /// only once differentiable there. A central difference across such a point has an error of
    /// order `h` rather than `h^2`. Nothing here is random, so this tolerance is a measured
    /// property of a fixed configuration and not a hedge against flakiness.
    #[test]
    fn the_gradient_check_also_holds_for_the_piecewise_families() {
        let spec = LifLayerSpec {
            n_in: 2,
            n_rec: 3,
            n_out: 2,
            recurrent: true,
            w_scale: 0.6,
            b_init: 0.05,
            seed: 7,
            ..LifLayerSpec::default()
        };
        let layer = spec.build().expect("valid spec");
        let x = probe_input(5, 2);
        let surs: Vec<Box<dyn Surrogate>> = vec![
            Box::new(FastSigmoid::default()),
            Box::new(Exponential::default()),
            Box::new(Triangular::default()),
            Box::new(Rectangular::default()),
            Box::new(StraightThrough::default()),
        ];
        for sur in &surs {
            let (_, g) = layer.loss_and_grad(sur.as_ref(), &x, 1, SpikeFn::Smooth).expect("valid");
            let h = 1e-6;
            for k in 0..layer.n_params() {
                let mut up = layer.clone();
                up.p[k] += h;
                let mut dn = layer.clone();
                dn.p[k] -= h;
                let lu = up.loss_and_grad(sur.as_ref(), &x, 1, SpikeFn::Smooth).expect("valid").0;
                let ld = dn.loss_and_grad(sur.as_ref(), &x, 1, SpikeFn::Smooth).expect("valid").0;
                let fd = (lu - ld) / (2.0 * h);
                let err = (fd - g[k]).abs() / (1.0 + g[k].abs());
                assert!(
                    err < 1e-4,
                    "{} param {k}: analytic {} vs finite difference {fd}",
                    sur.name(),
                    g[k]
                );
            }
        }
    }

    /// THE MODULE'S THESIS, MEASURED.
    ///
    /// With the real Heaviside on the forward pass, perturbing an input weight, a recurrent weight
    /// or a bias changes the loss by **exactly zero** — every finite difference is `0.0`, not small,
    /// because no spike moved. Meanwhile the surrogate gradient at those same parameters is
    /// non-zero. That gap is not an approximation error; it is the entire reason this module exists.
    ///
    /// The readout weights are the control: they are an ordinary linear layer downstream of the
    /// spikes, their true gradient is real, and the finite differences find it.
    #[test]
    fn the_true_gradient_of_a_spiking_network_is_zero_which_is_why_the_surrogate_exists() {
        let spec = LifLayerSpec {
            n_in: 2,
            n_rec: 5,
            n_out: 2,
            recurrent: true,
            w_scale: 0.8,
            b_init: 0.06,
            seed: 424242,
            ..LifLayerSpec::default()
        };
        let layer = spec.build().expect("valid spec");
        let x = probe_input(12, 2);
        let sur = ArcTan::default();
        let tr = layer.forward(&sur, &x, SpikeFn::Heaviside).expect("valid");
        assert!(tr.spike_count() > 0.0, "a silent network would make this test vacuous");
        assert!(
            tr.spike_count() < (tr.t_steps * tr.n_rec) as f64,
            "a saturated network would also make it vacuous"
        );

        let (_, g) = layer.loss_and_grad(&sur, &x, 1, SpikeFn::Heaviside).expect("valid");
        let h = 1e-9;
        let fd_of = |k: usize| {
            let mut up = layer.clone();
            up.p[k] += h;
            let mut dn = layer.clone();
            dn.p[k] -= h;
            let lu = up.loss_and_grad(&sur, &x, 1, SpikeFn::Heaviside).expect("valid").0;
            let ld = dn.loss_and_grad(&sur, &x, 1, SpikeFn::Heaviside).expect("valid").0;
            (lu - ld) / (2.0 * h)
        };

        let mut surrogate_norm = 0.0_f64;
        for j in 0..layer.n_rec {
            for i in 0..layer.n_in {
                let k = layer.idx_w(j, i);
                assert!(fd_of(k) == 0.0, "W[{j}][{i}]: the true gradient was not exactly zero");
                surrogate_norm += g[k] * g[k];
            }
            let k = layer.idx_b(j);
            assert!(fd_of(k) == 0.0, "b[{j}]: the true gradient was not exactly zero");
            surrogate_norm += g[k] * g[k];
            for m in 0..layer.n_rec {
                let k = layer.idx_v(j, m);
                assert!(fd_of(k) == 0.0, "V[{j}][{m}]: the true gradient was not exactly zero");
                surrogate_norm += g[k] * g[k];
            }
        }
        assert!(
            surrogate_norm.sqrt() > 1e-6,
            "the surrogate gradient was also zero, so nothing was demonstrated"
        );

        // The control: the readout's gradient is real and the finite differences see it.
        let mut readout_seen = 0u32;
        for c in 0..layer.n_out {
            for j in 0..layer.n_rec {
                let k = layer.idx_r(c, j);
                let fd = fd_of(k);
                if fd != 0.0 {
                    readout_seen += 1;
                    assert!(
                        (fd - g[k]).abs() < 1e-3 * (1.0 + g[k].abs()),
                        "R[{c}][{j}]: analytic {} vs finite difference {fd}",
                        g[k]
                    );
                }
            }
        }
        assert!(readout_seen > 0, "even the readout's true gradient was zero");
    }

    /// Train a layer on `batch` with Adam, returning the loss history.
    fn train(
        layer: &mut LifLayer,
        sur: &dyn Surrogate,
        batch: &[(Vec<f64>, usize)],
        lr: f64,
        steps: usize,
    ) -> Vec<f64> {
        let mut opt = Adam::new(layer.n_params(), lr).expect("positive lr");
        let mut hist = Vec::with_capacity(steps);
        for _ in 0..steps {
            let (loss, g) =
                layer.batch_loss_and_grad(sur, batch, SpikeFn::Heaviside).expect("non-empty");
            hist.push(loss);
            opt.step(&mut layer.p, &g).expect("finite gradient");
        }
        hist
    }

    /// (e) IT ACTUALLY LEARNS. Delayed XOR, which needs both memory across a silent gap and a
    /// non-linear combination of two cues, from a deterministic seed.
    ///
    /// At initialisation the network is at chance: measured cross-entropy `0.6982`, just above
    /// `ln(2) = 0.6931`, and two of the four patterns right — which for a balanced two-class task is
    /// what chance looks like. After 300 full-batch Adam steps at `lr = 0.01` with the arctangent
    /// surrogate the loss is `4.2e-5` and all four are right. Nothing here is random beyond the
    /// seed, so both numbers are properties of a fixed computation and the assertions are set an
    /// order of magnitude away from them.
    ///
    /// **Beside that figure**: 64 recurrent units for a four-pattern task is a very large network,
    /// and this test shows that the gradient works, not that the architecture is efficient. It also
    /// does not generalise to anything — there is no held-out set, because the task has only four
    /// patterns in total.
    #[test]
    fn a_recurrent_lif_layer_learns_delayed_xor() {
        let task = DelayedXor::default();
        let batch = task.patterns();
        let spec = LifLayerSpec {
            n_in: task.n_in(),
            n_rec: 64,
            n_out: task.n_out(),
            recurrent: true,
            w_scale: 0.4,
            b_init: 3e-3,
            seed: 8,
            ..LifLayerSpec::default()
        };
        let mut layer = spec.build().expect("valid spec");
        let sur = ArcTan::default();

        let (loss0, _) =
            layer.batch_loss_and_grad(&sur, &batch, SpikeFn::Heaviside).expect("non-empty");
        let correct0 = batch
            .iter()
            .filter(|(x, y)| layer.predict(&sur, x).expect("valid") == *y)
            .count();
        assert!(loss0 > 0.68, "initial loss {loss0} was already below chance");
        assert!(correct0 <= 2, "{correct0} of 4 correct before any training");

        let hist = train(&mut layer, &sur, &batch, 1e-2, 300);
        let final_loss = *hist.last().expect("300 steps");
        let correct = batch
            .iter()
            .filter(|(x, y)| layer.predict(&sur, x).expect("valid") == *y)
            .count();
        assert!(final_loss < 0.01, "loss fell only to {final_loss} (from {loss0})");
        assert!(hist[0] > hist[hist.len() / 2], "the loss did not fall monotonically in the large");
        assert_eq!(correct, 4, "only {correct} of 4 patterns classified correctly");

        // The trained network still emits real spikes; it did not solve the task by going silent
        // and letting the biases decide.
        let tr = layer.forward(&sur, &batch[3].0, SpikeFn::Heaviside).expect("valid");
        assert!(tr.spike_count() > 0.0, "the trained network is silent");
        assert!(tr.mean_rate(1e-3).expect("positive dt") > 0.0);
    }

    /// (b) part four, and the practical consequence of the whole module: THE WIDTH IS A
    /// HYPERPARAMETER AND THE SHARP LIMIT IS A DEAD END.
    ///
    /// Same seed, same task, same optimiser, same number of steps. The only change is the
    /// surrogate's width: sharpened by 300, the arctangent's full width at half maximum falls from
    /// `0.637` thresholds to `0.00212`, almost no unit is ever inside it, the gradient is starved,
    /// and the loss sits at `0.69315` — `ln(2)` to five figures, which is a network that has learned
    /// the class prior and nothing else. The wide surrogate reaches `4.2e-5` on the same run.
    #[test]
    fn an_over_sharp_surrogate_stops_learning() {
        let task = DelayedXor::default();
        let batch = task.patterns();
        let spec = LifLayerSpec {
            n_in: task.n_in(),
            n_rec: 64,
            n_out: task.n_out(),
            recurrent: true,
            w_scale: 0.4,
            b_init: 3e-3,
            seed: 8,
            ..LifLayerSpec::default()
        };
        let base = ArcTan::default();
        let sharp = base.sharpened(300.0).expect("positive factor");
        assert!(sharp.fwhm() < 0.003, "the sharpened width was {}", sharp.fwhm());

        let mut a = spec.build().expect("valid spec");
        let mut b = spec.build().expect("valid spec");
        let ha = train(&mut a, &base, &batch, 1e-2, 300);
        let hb = train(&mut b, sharp.as_ref(), &batch, 1e-2, 300);
        let (fa, fb) = (*ha.last().expect("steps"), *hb.last().expect("steps"));
        assert!(fa < 0.1, "the wide surrogate should have learned, loss {fa}");
        assert!(fb > 0.6, "the over-sharp surrogate learned anyway, loss {fb}");
        assert!(fb > 10.0 * fa, "wide {fa} vs sharp {fb}: the gap is not the point being made");
        // It is starved, not diverged: it still spikes, and its loss is the class prior.
        let tr = b.forward(sharp.as_ref(), &batch[0].0, SpikeFn::Heaviside).expect("valid");
        assert!(tr.spike_count() > 0.0, "the sharp run went silent, which is a different failure");
        assert!(
            (fb - core::f64::consts::LN_2).abs() < 0.05,
            "the starved loss {fb} is not the class prior ln(2)"
        );
    }

    /// Training is reproducible bit for bit. Two layers from the same seed, trained on the same
    /// batch, end at the same parameters — no clock, no OS entropy, no iteration-order dependence.
    #[test]
    fn training_is_deterministic_for_a_fixed_seed() {
        let task = DelayedXor::new(24, 4, 6).expect("windows fit");
        let batch = task.patterns();
        let spec = LifLayerSpec {
            n_in: task.n_in(),
            n_rec: 8,
            n_out: task.n_out(),
            seed: 99,
            ..LifLayerSpec::default()
        };
        let sur = FastSigmoid::default();
        let mut a = spec.build().expect("valid spec");
        let mut b = spec.build().expect("valid spec");
        assert_eq!(a.p, b.p, "two builds from one seed differed at initialisation");
        let ha = train(&mut a, &sur, &batch, 1e-2, 40);
        let hb = train(&mut b, &sur, &batch, 1e-2, 40);
        assert_eq!(ha, hb, "two identical training runs produced different losses");
        assert_eq!(a.p, b.p, "two identical training runs produced different weights");
    }

    /// A layer built without recurrence really ignores its `V` block: changing those entries by any
    /// amount changes nothing, and their gradient is exactly zero rather than a real gradient for an
    /// unused weight.
    #[test]
    fn a_non_recurrent_layer_ignores_and_zeroes_its_recurrent_block() {
        let spec = LifLayerSpec { n_in: 2, n_rec: 3, recurrent: false, ..LifLayerSpec::default() };
        let mut layer = spec.build().expect("valid spec");
        for j in 0..layer.n_rec {
            for k in 0..layer.n_rec {
                let idx = layer.idx_v(j, k);
                assert!(layer.p[idx] == 0.0, "V was initialised non-zero without recurrence");
            }
        }
        let x = probe_input(8, 2);
        let sur = ArcTan::default();
        let (l0, g) = layer.loss_and_grad(&sur, &x, 0, SpikeFn::Heaviside).expect("valid");
        for j in 0..layer.n_rec {
            for k in 0..layer.n_rec {
                let idx = layer.idx_v(j, k);
                assert!(g[idx] == 0.0, "V[{j}][{k}] had a gradient in a non-recurrent layer");
            }
        }
        let idx = layer.idx_v(0, 0);
        layer.p[idx] = 12.5;
        let l1 = layer.loss_and_grad(&sur, &x, 0, SpikeFn::Heaviside).expect("valid").0;
        assert!((l0 - l1).abs() < 1e-15, "an unused weight changed the loss: {l0} vs {l1}");
    }

    /// Adam refuses a non-finite gradient rather than absorbing it into moments it can never
    /// recover from.
    #[test]
    fn adam_refuses_a_non_finite_gradient_and_a_wrong_shape() {
        let mut opt = Adam::new(3, 1e-2).expect("positive lr");
        let mut p = vec![0.0; 3];
        assert!(matches!(
            opt.step(&mut p, &[0.0, f64::NAN, 0.0]),
            Err(SurrogateError::NonFiniteInput { index: 1, .. })
        ));
        assert!(matches!(
            opt.step(&mut p, &[0.0, 0.0]),
            Err(SurrogateError::ShapeMismatch { what: "gradient", got: 2, want: 3 })
        ));
        assert_eq!(opt.steps(), 0, "a refused step still advanced the counter");
        opt.step(&mut p, &[1.0, -1.0, 0.0]).expect("finite gradient");
        assert_eq!(opt.steps(), 1);
        // Adam's first step is +/- lr for any non-zero gradient, whatever its magnitude: that is the
        // per-parameter normalisation this module relies on to compare surrogates of different mass.
        assert!((p[0] + 1e-2).abs() < 1e-9, "first step was {}", p[0]);
        assert!((p[1] - 1e-2).abs() < 1e-9, "first step was {}", p[1]);
        assert!(p[2] == 0.0, "a zero gradient moved a parameter");
        assert!(Adam::new(0, 1e-2).is_err());
        assert!(Adam::new(3, 0.0).is_err());
    }
}
