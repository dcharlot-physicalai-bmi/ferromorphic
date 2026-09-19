//! Noise as a resource: a threshold unit cannot see a signal that never reaches its threshold —
//! until noise is added, and then there is a BEST amount of noise, which has a closed form. A
//! population of identical noisy units does better still, and how much better is a sum you can
//! evaluate exactly. Each is checked against its formula and against sampling.
//!
//! # What the mechanism is
//!
//! **Stochastic resonance** (Benzi, Sutera and Vulpiani, *The mechanism of stochastic resonance*,
//! Journal of Physics A 14(11):L453–L457, 1981; for neurons, Longtin, *Stochastic resonance in neuron
//! models*, Journal of Statistical Physics 70(1–2):309–327, 1993). A unit fires when its input
//! plus noise exceeds a threshold `θ`. If both values of a signal lie below `θ`, a noiseless unit
//! is silent for both and carries nothing. A little noise lets the HIGHER value cross more often
//! than the lower; a lot of noise drowns the difference. In between there is an optimum.
//!
//! **A population's count.** Give `N` identical units the same signal and independent noise, and
//! read the COUNT that fired (Stocks, *Suprathreshold stochastic resonance in multilevel
//! threshold systems*, Physical Review Letters 84(11):2310–2313, 2000). Without noise they all do
//! the same thing; with noise their disagreements carry information that no one of them does, and
//! the count of many units recovers a signal that is invisible to each.
//!
//! **Dither.** Uniform noise exactly as wide as a quantiser's step makes its AVERAGE output linear
//! in its input — the principle behind stochastic rounding and one-bit converters.
//!
//! # Why it is in a neuromorphic crate
//!
//! Analog neurons, memristive synapses and p-bits are noisy whether one wants it or not. These
//! are the results that say when that is a cost and when it is the mechanism: how much device
//! noise a threshold readout wants, and why a population of sloppy identical comparators can
//! out-resolve one precise one.
//!
//! # The closed forms this module is checked against
//!
//! - A unit with Gaussian noise `σ` fires with probability `Φ((s − θ)/σ)`, sampled within four
//!   standard errors.
//! - **The best noise for two sub-threshold values** `s₋ < s₊ < θ`: with `a = θ − s₊` and
//!   `b = θ − s₋`, the hit-rate difference `Φ(−a/σ) − Φ(−b/σ)` is greatest at
//!   `σ*² = (b² − a²)/(2 ln(b/a))`. Checked as a stationary point, as a maximum against its
//!   neighbours, and against a literal. If `s₊ ≥ θ` there is no optimum: noise only hurts.
//! - **Information in a population's count**, by exact summation over two binomials: equal to the
//!   binary-channel formula at `N = 1`; zero without noise for a sub-threshold signal; never more
//!   than one bit; non-decreasing in `N` at fixed noise; and approaching one bit as `N` grows —
//!   4096 units recover 0.999 of a bit that one unit sees a tenth of. The best noise level falls
//!   as `N` rises.
//! - **Dither**: with uniform noise of full width `Δ` the firing probability is
//!   `clamp((s − θ)/Δ + ½, 0, 1)` — exactly linear across the step.
//! - **Suprathreshold resonance** ([`levels_information`]): for a signal of many equiprobable
//!   levels straddling the threshold, ONE unit carries most with no noise at all, and its
//!   information only falls as noise is added; SIXTY-THREE units carry exactly one bit with no
//!   noise — they all agree — and MORE than that at a non-zero noise, under the ceiling
//!   `log₂ min(M, N + 1)`. With two levels the function is [`population_information`].
//!
//! # What this module has NOT reproduced
//!
//! - Dynamical stochastic resonance in a bistable well or in a spiking neuron driven by a
//!   periodic signal, where the measure is a spectral signal-to-noise ratio; this module is the
//!   static threshold case, where the optimum is algebra.
//! - Stocks's curves for a CONTINUOUS Gaussian signal, or his large-`N` asymptote. The signal in
//!   [`levels_information`] is a finite set of equiprobable levels, which is what can be summed
//!   exactly. (With TWO levels the effect cannot appear — a straddling binary signal is already
//!   one clean bit at zero noise for any `N` — and the first draft of this module's doc claimed
//!   otherwise until its test was written.)

use core::fmt;

use crate::surrogate::normal_cdf;

/// The largest population [`population_information`] will sum over.
pub const MAX_UNITS: usize = 1 << 20;

/// What went wrong, named rather than guessed around.
#[derive(Debug, Clone, PartialEq)]
pub enum ResonanceError {
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

impl fmt::Display for ResonanceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfRange { what, value, low, high } => {
                write!(f, "{what} = {value} is outside [{low}, {high}]")
            }
            Self::NonFinite { what } => write!(f, "{what} is not finite"),
        }
    }
}

impl std::error::Error for ResonanceError {}

fn finite(what: &'static str, v: f64) -> Result<f64, ResonanceError> {
    if v.is_finite() { Ok(v) } else { Err(ResonanceError::NonFinite { what }) }
}

fn non_negative(what: &'static str, v: f64) -> Result<f64, ResonanceError> {
    if v.is_finite() && v >= 0.0 {
        Ok(v)
    } else {
        Err(ResonanceError::OutOfRange { what, value: v, low: 0.0, high: f64::INFINITY })
    }
}

/// The probability that a unit with threshold `threshold` fires on `signal` under Gaussian noise of
/// standard deviation `sigma`: `Φ((s − θ)/σ)`. With no noise it is a step, firing AT the threshold.
///
/// # Errors
///
/// [`ResonanceError::NonFinite`] for a non-finite signal or threshold,
/// [`ResonanceError::OutOfRange`] for a negative `sigma`.
pub fn fire_probability(signal: f64, threshold: f64, sigma: f64) -> Result<f64, ResonanceError> {
    let margin = finite("signal", signal)? - finite("threshold", threshold)?;
    let sigma = non_negative("sigma", sigma)?;
    if sigma == 0.0 {
        return Ok(if margin >= 0.0 { 1.0 } else { 0.0 });
    }
    Ok(normal_cdf(margin / sigma))
}

/// How much more often the unit fires on `high` than on `low`: the hit rate minus the false-alarm
/// rate of a detector asked which of the two it was shown.
///
/// # Errors
///
/// As [`fire_probability`].
pub fn discriminability(low: f64, high: f64, threshold: f64, sigma: f64) -> Result<f64, ResonanceError> {
    Ok(fire_probability(high, threshold, sigma)? - fire_probability(low, threshold, sigma)?)
}

/// The noise level that maximises [`discriminability`] for two SUB-threshold values:
/// `σ* = √((b² − a²)/(2 ln(b/a)))`, `a = θ − high`, `b = θ − low`. `None` unless
/// `low < high < threshold` — when the higher value already reaches the threshold the best noise is
/// none at all.
#[must_use]
pub fn optimal_noise(low: f64, high: f64, threshold: f64) -> Option<f64> {
    if !(low < high && high < threshold) || !low.is_finite() || !threshold.is_finite() {
        return None;
    }
    let (a, b) = (threshold - high, threshold - low);
    Some(((b * b - a * a) / (2.0 * (b / a).ln())).sqrt())
}

/// The firing probability under UNIFORM noise of full width `width` (dither):
/// `clamp((s − θ)/Δ + ½, 0, 1)`, linear in the signal across one step.
///
/// # Errors
///
/// [`ResonanceError::NonFinite`] for a non-finite signal or threshold,
/// [`ResonanceError::OutOfRange`] for a non-positive width.
pub fn dithered_probability(signal: f64, threshold: f64, width: f64) -> Result<f64, ResonanceError> {
    let margin = finite("signal", signal)? - finite("threshold", threshold)?;
    if !(width > 0.0) || !width.is_finite() {
        return Err(ResonanceError::OutOfRange { what: "width", value: width, low: f64::MIN_POSITIVE, high: f64::INFINITY });
    }
    Ok((margin / width + 0.5).clamp(0.0, 1.0))
}

fn h2(p: f64) -> f64 {
    let term = |x: f64| if x > 0.0 { -x * x.log2() } else { 0.0 };
    term(p) + term(1.0 - p)
}

/// The information, bits, that one firing/silent outcome carries about an equiprobable binary
/// signal, given the firing probability under each value: `H₂(p̄) − ½[H₂(p₋) + H₂(p₊)]`.
/// `None` for a probability outside `[0, 1]`.
#[must_use]
pub fn binary_channel_information(p_low: f64, p_high: f64) -> Option<f64> {
    if !(0.0..=1.0).contains(&p_low) || !(0.0..=1.0).contains(&p_high) {
        return None;
    }
    Some((h2(0.5 * (p_low + p_high)) - 0.5 * (h2(p_low) + h2(p_high))).max(0.0))
}

/// The binomial probabilities `P(K = k)`, `k = 0..=n`, built in log space so that a population of
/// thousands does not underflow.
fn binomial(n: usize, p: f64) -> Vec<f64> {
    if p <= 0.0 || p >= 1.0 {
        let mut pmf = vec![0.0; n + 1];
        pmf[if p <= 0.0 { 0 } else { n }] = 1.0;
        return pmf;
    }
    let ratio = (p / (1.0 - p)).ln();
    let mut log = n as f64 * (1.0 - p).ln();
    let mut pmf = Vec::with_capacity(n + 1);
    for k in 0..=n {
        pmf.push(log.exp());
        log += ((n - k) as f64 / (k + 1) as f64).ln() + ratio;
    }
    pmf
}

/// The information, bits, that the COUNT of `n` identical units with independent Gaussian noise
/// carries about an equiprobable binary signal `{low, high}` — by exact summation over the two
/// binomial count distributions.
///
/// # Errors
///
/// As [`fire_probability`], plus [`ResonanceError::OutOfRange`] for `n` of zero or past
/// [`MAX_UNITS`].
pub fn population_information(low: f64, high: f64, threshold: f64, sigma: f64, n: usize) -> Result<f64, ResonanceError> {
    if n == 0 || n > MAX_UNITS {
        return Err(ResonanceError::OutOfRange { what: "n", value: n as f64, low: 1.0, high: MAX_UNITS as f64 });
    }
    let lo = binomial(n, fire_probability(low, threshold, sigma)?);
    let hi = binomial(n, fire_probability(high, threshold, sigma)?);
    let mut bits = 0.0;
    for (&a, &b) in lo.iter().zip(&hi) {
        let mix = 0.5 * (a + b);
        for cond in [a, b] {
            if cond > 0.0 {
                bits += 0.5 * cond * (cond / mix).log2();
            }
        }
    }
    Ok(bits.clamp(0.0, 1.0))
}

/// The information, bits, that the count of `n` identical noisy units carries about a signal drawn
/// uniformly from `levels` — the many-valued form of [`population_information`], and the setting
/// in which noise helps a signal that is NOT sub-threshold. Bounded by `log₂ min(M, n + 1)`.
///
/// # Errors
///
/// As [`population_information`], plus [`ResonanceError::OutOfRange`] for fewer than two levels or
/// more than [`MAX_UNITS`] of them, and [`ResonanceError::NonFinite`] for a non-finite level.
pub fn levels_information(levels: &[f64], threshold: f64, sigma: f64, n: usize) -> Result<f64, ResonanceError> {
    if levels.len() < 2 || levels.len() > MAX_UNITS {
        return Err(ResonanceError::OutOfRange { what: "levels", value: levels.len() as f64, low: 2.0, high: MAX_UNITS as f64 });
    }
    if n == 0 || n > MAX_UNITS || n.saturating_mul(levels.len()) > 64 * MAX_UNITS {
        return Err(ResonanceError::OutOfRange { what: "n", value: n as f64, low: 1.0, high: MAX_UNITS as f64 });
    }
    let mut conditionals = Vec::with_capacity(levels.len());
    for &level in levels {
        conditionals.push(binomial(n, fire_probability(level, threshold, sigma)?));
    }
    let weight = 1.0 / levels.len() as f64;
    let mut bits = 0.0;
    for k in 0..=n {
        let mix: f64 = conditionals.iter().map(|c| weight * c[k]).sum();
        for c in &conditionals {
            if c[k] > 0.0 {
                bits += weight * c[k] * (c[k] / mix).log2();
            }
        }
    }
    let ceiling = (levels.len().min(n + 1) as f64).log2();
    Ok(bits.clamp(0.0, ceiling))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fusion::gaussian;
    use crate::rng::Rng;

    #[test]
    fn a_noisy_threshold_fires_with_the_gaussian_tail_probability() {
        assert_eq!(fire_probability(0.3, 1.0, 0.0).unwrap(), 0.0);
        assert_eq!(fire_probability(1.0, 1.0, 0.0).unwrap(), 1.0);
        assert_eq!(fire_probability(1.0, 1.0, 0.7).unwrap(), 0.5, "AT the threshold it is a coin, whatever the noise");
        // One standard deviation below: Φ(−1) = 0.158655.
        let p = fire_probability(0.5, 1.0, 0.5).unwrap();
        assert!((p - 0.158_655_253_931_457).abs() < 1e-12);
        assert!((fire_probability(1.5, 1.0, 0.5).unwrap() + p - 1.0).abs() < 1e-14, "symmetric about the threshold");
        let mut rng = Rng::new(31);
        let trials = 40_000;
        let fired = (0..trials).filter(|_| 0.5 + 0.5 * gaussian(&mut rng) > 1.0).count();
        let se = (p * (1.0 - p) / f64::from(trials)).sqrt();
        assert!((fired as f64 / f64::from(trials) - p).abs() < 4.0 * se, "{fired} of {trials} fired, Φ(−1) says {p}");
    }

    #[test]
    fn a_sub_threshold_signal_has_a_best_noise_and_this_is_it() {
        let (low, high, theta) = (0.2, 0.6, 1.0);
        let best = optimal_noise(low, high, theta).unwrap();
        // a = 0.4, b = 0.8: σ*² = (0.64 − 0.16)/(2 ln 2).
        assert!((best * best - 0.48 / (2.0 * 2f64.ln())).abs() < 1e-15);
        assert!((best - 0.5884).abs() < 1e-4);
        let d = |s: f64| discriminability(low, high, theta, s).unwrap();
        // A maximum: better than 1% either side, and flat to second order there.
        assert!(d(best) > d(0.99 * best) && d(best) > d(1.01 * best));
        let slope = (d(best * (1.0 + 1e-5)) - d(best * (1.0 - 1e-5))) / (2e-5 * best);
        assert!(slope.abs() < 1e-8, "dD/dσ at σ* is {slope}");
        // Nothing without noise, nothing with too much, and a fifth of the trials in between.
        assert_eq!(d(0.0), 0.0);
        assert!(d(1e-2) < 1e-12 && d(1e3) < 1e-3);
        assert!((d(best) - (normal_cdf(-0.4 / best) - normal_cdf(-0.8 / best))).abs() < 1e-15 && d(best) > 0.15);
        // Moving BOTH values and the threshold together changes nothing; scaling all three scales σ*.
        assert!((optimal_noise(5.2, 5.6, 6.0).unwrap() - best).abs() < 1e-12);
        assert!((optimal_noise(2.0, 6.0, 10.0).unwrap() - 10.0 * best).abs() < 1e-12);
        // Once the higher value reaches the threshold, noise only hurts: there is no optimum.
        assert_eq!(optimal_noise(0.2, 1.0, 1.0), None);
        assert_eq!(optimal_noise(0.2, 1.4, 1.0), None);
        assert_eq!(optimal_noise(0.6, 0.2, 1.0), None);
        assert_eq!(optimal_noise(f64::NEG_INFINITY, 0.2, 1.0), None);
        let straddle = |s: f64| discriminability(0.6, 1.4, 1.0, s).unwrap();
        assert_eq!(straddle(0.0), 1.0);
        assert!(straddle(0.1) < 1.0 && straddle(0.5) < straddle(0.1));
    }

    #[test]
    fn dither_makes_a_step_linear_on_average() {
        for (s, want) in [(0.5, 0.0), (0.75, 0.0), (0.875, 0.25), (1.0, 0.5), (1.125, 0.75), (1.25, 1.0), (2.0, 1.0)] {
            assert_eq!(dithered_probability(s, 1.0, 0.5).unwrap(), want, "signal {s}");
        }
        // Sampled: uniform noise on [−¼, ¼) added to 1.1 crosses 1.0 seven times in ten.
        let mut rng = Rng::new(5);
        let trials = 40_000;
        let fired = (0..trials).filter(|_| 1.1 + 0.5 * (rng.next_f64() - 0.5) >= 1.0).count();
        let p = dithered_probability(1.1, 1.0, 0.5).unwrap();
        assert!((p - 0.7).abs() < 1e-12);
        assert!((fired as f64 / f64::from(trials) - p).abs() < 4.0 * (p * (1.0 - p) / f64::from(trials)).sqrt());
    }

    #[test]
    fn one_unit_is_a_binary_channel() {
        assert_eq!(binary_channel_information(0.0, 1.0), Some(1.0));
        assert_eq!(binary_channel_information(0.3, 0.3), Some(0.0));
        assert_eq!(binary_channel_information(1.0, 0.0), Some(1.0));
        // p = (0, ½): H₂(¼) − ½ = 0.3113.
        let z = binary_channel_information(0.0, 0.5).unwrap();
        assert!((z - (-(0.25f64 * 0.25f64.log2() + 0.75 * 0.75f64.log2()) - 0.5)).abs() < 1e-15 && (z - 0.3113).abs() < 1e-4);
        assert_eq!(binary_channel_information(-0.1, 0.5), None);
        assert_eq!(binary_channel_information(0.1, 1.5), None);
        for sigma in [0.1, 0.4, 1.0, 3.0] {
            let (pl, ph) = (fire_probability(0.2, 1.0, sigma).unwrap(), fire_probability(0.6, 1.0, sigma).unwrap());
            let one = population_information(0.2, 0.6, 1.0, sigma, 1).unwrap();
            assert!((one - binary_channel_information(pl, ph).unwrap()).abs() < 1e-14, "σ = {sigma}");
        }
    }

    #[test]
    fn a_population_of_noisy_units_out_resolves_a_quiet_one() {
        let info = |sigma: f64, n: usize| population_information(0.6, 1.4, 1.0, sigma, n).unwrap();
        // The signal straddles the threshold. ONE unit is best with no noise: a clean bit.
        assert_eq!(info(0.0, 1), 1.0);
        assert!(info(0.3, 1) < 1.0 && info(1.0, 1) < info(0.3, 1));
        // Sub-threshold, nothing gets through without noise, however many units there are.
        assert_eq!(population_information(0.2, 0.6, 1.0, 0.0, 64).unwrap(), 0.0);
        // More units never carry less, at any noise…
        for sigma in [0.2, 0.6, 1.5, 4.0] {
            let sub = |n: usize| population_information(0.2, 0.6, 1.0, sigma, n).unwrap();
            assert!(sub(1) <= sub(8) + 1e-12 && sub(8) <= sub(64) + 1e-12 && sub(64) <= 1.0, "σ = {sigma}");
        }
        // …and enough of them recover the whole bit from a signal no single unit can see.
        assert!(population_information(0.2, 0.6, 1.0, 0.6, 4096).unwrap() > 0.999);
        assert!(population_information(0.2, 0.6, 1.0, 0.6, 1).unwrap() < 0.1);
        // Stocks's result needs MORE than two signal values to show in the information — with two,
        // a straddling signal is already a clean bit at σ = 0 for any N. What two values do show
        // is the sub-threshold resonance sharpening with N: the best σ on a grid moves, and the
        // information at it grows.
        let grid: Vec<f64> = (1..=60).map(|k| 0.05 * f64::from(k)).collect();
        let peak = |n: usize| {
            grid.iter().map(|&s| (population_information(0.2, 0.6, 1.0, s, n).unwrap(), s)).fold((0.0, 0.0), |a, b| if b.0 > a.0 { b } else { a })
        };
        let (one, many) = (peak(1), peak(64));
        assert!(one.1 > 0.05 && one.1 < 3.0, "the single unit's best noise {} is at the edge of the grid", one.1);
        assert!(many.0 > 5.0 * one.0, "64 units at their best carry {} bits against {}", many.0, one.0);
        assert!(many.1 < one.1, "and want LESS noise to do it: {} against {}", many.1, one.1);
    }

    #[test]
    fn many_units_want_noise_even_when_the_signal_crosses_the_threshold() {
        // Thirty-two equiprobable levels, symmetric about the threshold and none on it.
        let levels: Vec<f64> = (0..32).map(|k| (f64::from(k) - 15.5) / 16.0).collect();
        let info = |sigma: f64, n: usize| levels_information(&levels, 0.0, sigma, n).unwrap();
        // ONE unit: a clean sign bit with no noise, and less with any.
        assert_eq!(info(0.0, 1), 1.0);
        let mut last = 1.0;
        for sigma in [0.1, 0.3, 0.6, 1.0, 2.0] {
            let now = info(sigma, 1);
            assert!(now < last, "one unit gained from noise: {last} → {now} at σ = {sigma}");
            last = now;
        }
        // SIXTY-THREE units with no noise are sixty-three copies of that one bit…
        assert_eq!(info(0.0, 63), 1.0);
        // …and with noise they disagree in proportion to the signal, and carry more.
        let grid: Vec<f64> = (1..=40).map(|k| 0.05 * f64::from(k)).collect();
        let (best, at) = grid.iter().map(|&s| (info(s, 63), s)).fold((0.0, 0.0), |a, b| if b.0 > a.0 { b } else { a });
        assert!(best > 2.0, "63 units at their best noise carry {best} bits");
        assert!(at > 0.1 && at < 1.9, "the best noise {at} is at the edge of the grid");
        assert!(info(2.0 * at, 63) < best && info(0.25 * at, 63) < best);
        assert!(best <= 5.0, "and never more than log₂ 32");
        // The ceiling is log₂ min(M, N + 1): three units have four counts, so at most two bits.
        assert!(info(at, 3) <= 2.0);
        // With two levels it is the binary function.
        for sigma in [0.0, 0.3, 1.2] {
            let two = levels_information(&[0.2, 0.6], 1.0, sigma, 8).unwrap();
            assert!((two - population_information(0.2, 0.6, 1.0, sigma, 8).unwrap()).abs() < 1e-14);
        }
        assert!(matches!(levels_information(&[0.1], 0.0, 0.5, 4), Err(ResonanceError::OutOfRange { what: "levels", .. })));
        assert!(matches!(levels_information(&[0.1, f64::NAN], 0.0, 0.5, 4), Err(ResonanceError::NonFinite { what: "signal" })));
        assert!(matches!(levels_information(&[0.1, 0.2], 0.0, 0.5, 0), Err(ResonanceError::OutOfRange { what: "n", .. })));
    }

    #[test]
    fn bad_arguments_are_refused() {
        assert!(matches!(fire_probability(f64::NAN, 1.0, 0.5), Err(ResonanceError::NonFinite { what: "signal" })));
        assert!(matches!(fire_probability(0.5, f64::INFINITY, 0.5), Err(ResonanceError::NonFinite { what: "threshold" })));
        assert!(matches!(fire_probability(0.5, 1.0, -0.1), Err(ResonanceError::OutOfRange { what: "sigma", .. })));
        assert!(matches!(discriminability(0.1, 0.2, 1.0, f64::NAN), Err(ResonanceError::OutOfRange { what: "sigma", .. })));
        assert!(matches!(dithered_probability(0.5, 1.0, 0.0), Err(ResonanceError::OutOfRange { what: "width", .. })));
        assert!(matches!(dithered_probability(f64::NAN, 1.0, 0.5), Err(ResonanceError::NonFinite { what: "signal" })));
        assert!(matches!(population_information(0.2, 0.6, 1.0, 0.5, 0), Err(ResonanceError::OutOfRange { what: "n", .. })));
        assert!(matches!(population_information(0.2, 0.6, 1.0, 0.5, MAX_UNITS + 1), Err(ResonanceError::OutOfRange { what: "n", .. })));
        assert!(matches!(population_information(0.2, 0.6, 1.0, -1.0, 4), Err(ResonanceError::OutOfRange { what: "sigma", .. })));
        // The count distribution is a distribution, for a population large enough to underflow a
        // naive (1 − p)^n: it sums to one and has the binomial mean.
        let pmf = binomial(5000, 0.3);
        assert!((pmf.iter().sum::<f64>() - 1.0).abs() < 1e-10);
        let mean: f64 = pmf.iter().enumerate().map(|(k, p)| k as f64 * p).sum();
        assert!((mean - 1500.0).abs() < 1e-6);
        assert_eq!(binomial(3, 0.0), vec![1.0, 0.0, 0.0, 0.0]);
        assert_eq!(binomial(3, 1.0), vec![0.0, 0.0, 0.0, 1.0]);
        assert_eq!(binomial(2, 0.5), vec![0.25, 0.5, 0.25]);
    }
}
