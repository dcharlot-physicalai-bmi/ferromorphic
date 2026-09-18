//! The rest of the neural codes, and the decoders that invert them.
//!
//! A neuron emits identical pulses. Everything it can say must therefore be said with *which* cell
//! fires, *when* it fires, and *how many times* — those three axes are the entire design space of a
//! spiking representation, and a "neural code" is a choice of where on them to put the number.
//! [`crate::encode`] holds three of those choices: rate (how many), latency (when, once) and delta
//! (only on change). This module holds the others, because a library that implements one code and
//! calls it *the* encoding has quietly decided the energy budget, the latency and the noise
//! robustness of every experiment run on it.
//!
//! # The map
//!
//! | code | where the information is | spikes per value | invariant to | destroyed by |
//! |---|---|---|---|---|
//! | [`GaussianPopulation`] | which cells are active, and how much | `n` × window | the loss of any one cell | noise correlated across the population |
//! | [`CosinePopulation`] | the vector sum of preferred directions | `n` × window | the tuning baseline `b0`, **exactly** | preferred directions that do not tile evenly |
//! | [`RankOrderCode`] | the ORDER of the first spikes | one per cell, once | **any positive rescaling of the input** | jitter larger than the gap between arrivals |
//! | [`PhaseEncoder`] | spike time within an oscillation cycle | one per cycle | the oscillation's amplitude | a shift in the oscillation's phase |
//! | [`BurstCode`] | number of spikes in a burst | `k ≤ max_spikes` | spike times inside the burst | one lost or one inserted spike |
//! | [`Bsa`], [`Hsa`] | the whole train, read through a filter | about one per sample | slow baseline drift | signal energy above the filter's cutoff |
//! | [`TemporalContrast`] | threshold crossings of the signal | one per crossing | any constant offset | slew faster than one threshold per sample |
//!
//! # Population coding: spend neurons to buy precision
//!
//! One neuron with a monotone tuning curve can report a value, badly: its firing rate is noisy, its
//! range is whatever its saturation allows, and if it dies the value is gone. A *population* gives
//! every cell a preferred value and a bell-shaped tuning curve overlapping its neighbours', so a
//! stimulus lights a hill of activity whose **position** is the answer. Precision then comes from
//! the number of cells rather than from any cell's reliability, and it improves as `1/sqrt(n)`
//! because that is how independent noise averages — which is a bound, not a hope, and
//! the test suite fits the exponent rather than asserting the direction.
//!
//! This is how motor cortex encodes reach direction (Georgopoulos, Schwartz & Kettner, *Science*
//! 233:1416-1419, 1986), how a cricket encodes wind direction with four cells, and how essentially
//! every neuromorphic sensor front end turns an analogue channel into spikes. Two decoders are
//! given because they are not the same estimator: the **population vector** is a weighted sum of
//! preferred directions, cheap enough for a single fused multiply-add per cell and exact for
//! cosine tuning on a uniform tiling; **maximum likelihood** is a search that knows the noise model
//! and reaches the Cramér-Rao bound, and costs a grid sweep to do it (Dayan & Abbott, *Theoretical
//! Neuroscience*, MIT Press, 2001, ch. 3; Seung & Sompolinsky, *PNAS* 90, 1993).
//!
//! # Rank order: the only code here that does not care how bright the light is
//!
//! Thorpe and Gautrais observed that a retina hit by a flash fires its most strongly driven
//! ganglion cells first, and that the **order of arrival** already carries most of what the image
//! is — before any cell has fired twice (Thorpe & Gautrais, *Rank order coding*, in *Computational
//! Neuroscience: Trends in Research*, Plenum, 1998; Gautrais & Thorpe, *Biosystems* 48, 1998).
//!
//! The consequence is the property no other code in this crate has. Multiply every input by the
//! same positive constant — turn the lights down, halve the contrast, change the gain of the
//! sensor — and every spike moves later, but **the order does not change at all**. The decoded
//! symbol is bit-identical. A rate code, a latency code, a burst code and a temporal-contrast code
//! all change their output under that transformation; rank order does not, and
//! `the_order_is_bit_identical_under_any_positive_rescaling` is the test.
//!
//! What is paid for it: `n` spikes carry at most `log2(n!)` bits and no magnitude at all. Eight
//! cells give 15.3 bits of order and zero bits of amplitude. If you need to know *how bright*,
//! this is the wrong code, and [`RankOrderCode::decode_amplitudes`] recovers amplitudes only by
//! **assuming** the geometric law the readout was built on — which the tests demonstrate by
//! showing it is exact for a geometric input and badly wrong for a linear one.
//!
//! # Phase: a clock you did not have to send
//!
//! If a population shares a background oscillation — hippocampal theta, a gamma cycle, a
//! sub-threshold membrane oscillation — then a spike's *position within the cycle* is a number,
//! and the reference is free because every receiver already has it. O'Keefe & Recce (*Hippocampus*
//! 3:317-330, 1993) found place cells advancing their theta phase as an animal crosses a field;
//! Hopfield (*Nature* 376:33-36, 1995) proposed phase against a sub-threshold oscillation as a
//! general analogue representation; Montemurro et al. (*Current Biology* 18, 2008) measured phase
//! carrying information beyond the spike count in V1.
//!
//! The invariance is precise and is tested: the code does not depend on the oscillation's
//! **amplitude** at all (the amplitude never enters [`PhaseEncoder::encode_tick`]), and it depends
//! on the oscillation's **phase** exactly — shift the reference by `δ` radians and every decoded
//! value shifts by `δ / 2π`, which the test asserts as an equality rather than as a tendency. That
//! is the whole risk of phase coding in one sentence: it buys a free reference and pays for it by
//! being wrong by exactly the reference's error.
//!
//! # Burst: a small integer, sent robustly and read back exactly
//!
//! A burst is several spikes close together followed by a gap. Treating the **count** as the
//! message gives a short, quantised, easily detected symbol: bursts cross unreliable synapses far
//! better than single spikes, and the receiver only has to count (Izhikevich, Desai, Walcott &
//! Hoppensteadt, *Trends in Neurosciences* 26:161-167, 2003; Kepecs & Lisman, *Network:
//! Computation in Neural Systems* 14, 2003). The honest limit is in the same sentence: the code is
//! `log2(max_spikes + 1)` bits, a single lost spike is a full quantum of error, and a value of zero
//! is *silence*, which is indistinguishable from a cell that was never asked.
//!
//! # `BSA` and `HSA`: encoding by deconvolution
//!
//! The previous codes all pick a feature of the signal. These two do something different and
//! stranger: they choose the spike train whose **convolution with a fixed filter** best
//! reconstructs the signal. Decoding is then one convolution — a finite impulse response filter
//! driven by spikes — so the decoder is the cheapest object in this module, and the encoder does
//! the work.
//!
//! `HSA`, the Hough Spiker Algorithm, is the greedy version: at each sample, if the filter fits
//! entirely underneath the remaining signal, fire and subtract it. `BSA`, Ben's Spiker Algorithm
//! (Schrauwen & Van Campenhout, *`BSA`, a fast and accurate spike train encoding scheme*, `IJCNN`
//! 2003), replaces the pointwise test with an error comparison and a threshold, which lets it fire
//! where `HSA` will not and reconstructs better on most signals.
//!
//! `HSA`'s greediness buys an exact invariant, and this module tests it. The strict algorithm only
//! subtracts a filter that fits entirely underneath the residual, so after every firing the
//! residual is non-negative at every sample the filter touched — and therefore **the
//! reconstruction never exceeds the signal at any sample**. `HSA` is a systematic under-estimate,
//! by construction. `BSA` has no such guarantee and does overshoot, by 0.014 of full scale on this
//! module's reference signal.
//!
//! The precondition of that invariant is worth being exact about, because the first draft of this
//! module got it wrong. It is a condition on the **signal** — non-negative everywhere — and on the
//! threshold being zero. It is **not** a condition on the filter: a windowed-sinc kernel with ten
//! negative taps satisfies the invariant just as exactly, which a sweep over 20 filters and dozens
//! of signals confirms at zero violations. Break the signal's non-negativity instead and it fails
//! at once: a sinusoid offset to −0.2 is over-reconstructed by 0.70.
//!
//! What they fail on, stated plainly: both reconstruct the signal as a non-negative sum of shifted
//! copies of one filter, so they cannot represent anything the filter cannot build. Energy above
//! the filter's cutoff is lost, a signal that must go below zero is unrepresentable (the
//! convention is to offset into `[0, 1]` first), and a transient steeper than the filter's leading
//! edge is smeared. The test suite measures the error on a slow sinusoid *and* on a fast one and
//! prints both, so the failure is a number on the page rather than a caveat.
//! `BSA` is the encoder the biosignal spiking literature reaches for — electroencephalogram and
//! electromyogram pipelines in particular. This implementation did not locate a vendor
//! specification stating which encoder any specific always-on biosignal part uses internally, so
//! no such claim is made here.
//!
//! # Temporal contrast, and what [`crate::encode::DeltaEncoder`] already is
//!
//! Threshold-based encoding sends an event when the signal moves past a threshold from a
//! reference, and nothing otherwise. Petro, Kasabov & Whittington (*IEEE Transactions on Neural
//! Networks and Learning Systems*, 2020) name three variants, and all three are here:
//! **step-forward** (the reference moves by one threshold on each event), **moving window** (the
//! reference is the mean of the last `w` samples) and **threshold-based representation** (fire on
//! the derivative, against a threshold derived from the signal's own statistics — which is
//! *non-causal*, since it needs the whole signal before it can encode the first sample, and that
//! is said in [`TemporalContrast::threshold_by_statistics`] rather than left to be discovered).
//!
//! [`crate::encode::DeltaEncoder`] is **already** step-forward encoding, with one addition: it may
//! emit several events for one sample so a large jump is reported in full.
//! [`TemporalContrast`] in [`ContrastMode::StepForward`] emits at most one event per sample, which
//! is the published algorithm, and
//! `step_forward_is_the_existing_delta_encoder_capped_at_one_event` asserts the two produce
//! event-for-event identical output when the delta encoder's cap is set to one. They are the same
//! mechanism and this crate now says so in a test rather than in a comment.
//!
//! # A worked example
//!
//! ```
//! use ferromorphic::coding::{CosinePopulation, RankOrderCode, rates_from_counts};
//! use ferromorphic::rng::Rng;
//!
//! // A motor-cortex population: 12 cells with cosine tuning, counted for half a second.
//! let pop = CosinePopulation::new(12, 20.0, 20.0, 0.0)?;
//! let want = 2.1_f64; // radians
//!
//! // Noise-free, the population vector is EXACT — for every n >= 3, not just for large n.
//! let clean = pop.decode_population_vector(&pop.rates(want)?)?;
//! assert!((clean - want).abs() < 1e-12);
//!
//! // With Poisson counts it is close, and the error falls as 1/sqrt(n).
//! let mut rng = Rng::new(7);
//! let counts = pop.sample_counts(&mut rng, want, 0.5)?;
//! let noisy = pop.decode_population_vector(&rates_from_counts(&counts, 0.5)?)?;
//! assert!((noisy - want).abs() < 0.5);
//!
//! // Rank order: turn the lights down by nine orders of magnitude and the symbol is identical.
//! let code = RankOrderCode::new(4, 0.9)?;
//! let bright = code.encode(&[0.9, 0.2, 0.7, 0.4])?;
//! let dim = code.encode(&[0.9e-9, 0.2e-9, 0.7e-9, 0.4e-9])?;
//! assert_eq!(bright, dim);
//! assert_eq!(bright.order, vec![0, 2, 3, 1]);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # Units and refusals
//!
//! Rates are hertz, windows and time constants are seconds, angles are radians, and tick indices
//! are integers to be multiplied by `dt` once at the point of use ([`crate::spike`] explains why).
//! Every encoder rejects a non-finite input by naming it rather than letting it reach a membrane
//! potential, and every decoder returns an error where the evidence does not determine an answer —
//! a population whose cells all fire equally has **no** preferred direction, and
//! [`CosinePopulation::decode_population_vector`] says so instead of returning the zero vector's
//! `atan2`, which is due east.

use crate::rng::Rng;
use crate::spike::{Event, Polarity, Spike, Train};
use core::f64::consts::{PI, TAU};

/// Why an encode or a decode could not be performed.
///
/// Every variant names the offending quantity. The alternative — a `None` with no reason, or a
/// clamped value — makes a mis-parameterised encoder look like a quiet one.
#[derive(Debug, Clone, PartialEq)]
pub enum CodeError {
    /// An input was not a finite number.
    NotFinite {
        /// Which quantity: `"stimulus"`, `"rate"`, `"signal sample"`, and so on.
        what: &'static str,
        /// Index within the offending slice, or `0` for a scalar.
        index: usize,
    },
    /// A quantity that must be strictly positive was zero or negative.
    NotPositive {
        /// Which quantity, e.g. `"sigma"` or `"window_s"`.
        what: &'static str,
        /// The value supplied.
        value: f64,
    },
    /// A collection that must be non-empty was empty.
    Empty {
        /// Which collection, e.g. `"filter"` or `"population"`.
        what: &'static str,
    },
    /// A slice of observations did not match the population it was decoded against.
    LengthMismatch {
        /// Length the callee required.
        expected: usize,
        /// Length it received.
        got: usize,
    },
    /// An interval was empty or inverted: `lo` must be strictly below `hi`.
    EmptyRange {
        /// Lower bound as supplied.
        lo: f64,
        /// Upper bound as supplied.
        hi: f64,
    },
    /// A value fell outside the closed unit interval the code is defined on.
    OutOfUnitRange {
        /// Which quantity, e.g. `"stimulus"`.
        what: &'static str,
        /// The value supplied.
        value: f64,
    },
    /// The observations carry no answer, so none is returned.
    ///
    /// This is the refusal that matters most in this module. A silent population has no decoded
    /// value, a uniformly active circular population has no decoded direction, and an empty spike
    /// train has no decoded rank. Returning `0.0` for any of them is a number that will be plotted.
    NoEvidence {
        /// What was asked for and could not be determined.
        what: &'static str,
    },
}

impl core::fmt::Display for CodeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotFinite { what, index } => {
                write!(f, "{what} at index {index} was not a finite number")
            }
            Self::NotPositive { what, value } => {
                write!(f, "{what} must be strictly positive, was {value}")
            }
            Self::Empty { what } => write!(f, "{what} was empty"),
            Self::LengthMismatch { expected, got } => {
                write!(f, "expected {expected} values, got {got}")
            }
            Self::EmptyRange { lo, hi } => {
                write!(f, "range [{lo}, {hi}] is empty or inverted")
            }
            Self::OutOfUnitRange { what, value } => {
                write!(f, "{what} must lie in [0, 1], was {value}")
            }
            Self::NoEvidence { what } => {
                write!(f, "the observations do not determine {what}")
            }
        }
    }
}

/// So `?` works in a caller whose error type is `Box<dyn Error>`, as every example here uses.
impl std::error::Error for CodeError {}

fn check_finite(what: &'static str, xs: &[f64]) -> Result<(), CodeError> {
    for (i, &x) in xs.iter().enumerate() {
        if !x.is_finite() {
            return Err(CodeError::NotFinite { what, index: i });
        }
    }
    Ok(())
}

fn check_unit(what: &'static str, x: f64) -> Result<(), CodeError> {
    if !x.is_finite() {
        return Err(CodeError::NotFinite { what, index: 0 });
    }
    if !(0.0..=1.0).contains(&x) {
        return Err(CodeError::OutOfUnitRange { what, value: x });
    }
    Ok(())
}

/// Wrap an angle into `[0, 2π)`, radians.
///
/// Used so that two decoders that disagree by a full turn compare equal. `f64::rem_euclid` is the
/// operation, named here because writing `%` instead gives `-0.1` for `-0.1` and a decoded
/// direction that is negative half the time.
#[must_use]
pub fn wrap_angle(a: f64) -> f64 {
    if !a.is_finite() {
        return a;
    }
    a.rem_euclid(TAU)
}

/// The signed difference `a - b` wrapped into `(-π, π]`, radians.
///
/// This is the only correct way to measure an angular error. The unwrapped difference between
/// 0.01 rad and 6.27 rad is 6.26 rad; the answer is −0.013 rad, and a root-mean-square error
/// computed the first way is dominated entirely by trials that were nearly perfect.
#[must_use]
pub fn angular_difference(a: f64, b: f64) -> f64 {
    let d = wrap_angle(a - b);
    if d > PI { d - TAU } else { d }
}

/// A Poisson draw with mean `lambda`, deterministic for a given [`Rng`] state.
///
/// `None` when `lambda` is negative or not finite; a mean of exactly zero returns `Some(0)`,
/// because a cell driven at zero hertz emits zero spikes and that is a fact rather than a refusal.
///
/// # Algorithm, and why it is not the textbook one line
///
/// Knuth's product method multiplies uniforms until the product falls below `exp(-lambda)`. At
/// `lambda > 745` that exponential underflows to exactly zero in `f64` and the loop never
/// terminates — a hang, not a wrong answer, and one that only appears when somebody raises a
/// firing rate or lengthens a window. The fix used here is exact rather than approximate: a
/// Poisson variable of mean `lambda` is the sum of independent Poisson variables whose means sum
/// to `lambda`, so the mean is split into chunks of at most 500 and the draws are added.
#[must_use]
pub fn poisson_count(rng: &mut Rng, lambda: f64) -> Option<u64> {
    if !lambda.is_finite() || lambda < 0.0 {
        return None;
    }
    let mut total = 0u64;
    let mut left = lambda;
    while left > 0.0 {
        let chunk = left.min(500.0);
        left -= chunk;
        let limit = (-chunk).exp();
        let mut k = 0u64;
        let mut p = 1.0f64;
        loop {
            p *= rng.next_f64();
            if p <= limit {
                break;
            }
            k += 1;
        }
        total += k;
    }
    Some(total)
}

/// The normalised cardinal sine, `sin(πx) / (πx)`, with `sinc(0) = 1`.
///
/// Public because it is the only special function this module needs and a caller designing its own
/// reconstruction filter needs the same one. The normalised convention (the first zero at `x = 1`)
/// is the signal-processing one, not the mathematician's `sin(x)/x`.
#[must_use]
pub fn sinc(x: f64) -> f64 {
    if x == 0.0 {
        return 1.0;
    }
    let px = PI * x;
    px.sin() / px
}

/// A Hamming-windowed sinc low-pass filter of `taps` coefficients, normalised to sum to 1.
///
/// `cutoff` is in cycles per sample and must lie in `(0, 0.5)`; 0.5 is the Nyquist frequency of the
/// sample stream. The unit sum means the filter has a gain of exactly 1 at DC, which is the
/// property [`Bsa`] and [`Hsa`] need: a spike on every sample then reconstructs the constant 1, so
/// a constant input of `c` is encoded at a spike density of `c`, and that identity is what the
/// round-trip test for a constant signal checks.
///
/// # Errors
///
/// [`CodeError::Empty`] for `taps == 0`, [`CodeError::NotPositive`] for a `cutoff` outside
/// `(0, 0.5)` or not finite, and [`CodeError::NoEvidence`] in the pathological case of a tap sum
/// that is not positive, which no cutoff in range produces but which would otherwise divide by
/// zero.
pub fn fir_lowpass(taps: usize, cutoff: f64) -> Result<Vec<f64>, CodeError> {
    if taps == 0 {
        return Err(CodeError::Empty { what: "filter taps" });
    }
    if !cutoff.is_finite() || cutoff <= 0.0 || cutoff >= 0.5 {
        return Err(CodeError::NotPositive { what: "cutoff (cycles/sample, in (0, 0.5))", value: cutoff });
    }
    if taps == 1 {
        return Ok(vec![1.0]);
    }
    let m = taps - 1;
    let centre = m as f64 / 2.0;
    let mut h = Vec::with_capacity(taps);
    for k in 0..taps {
        let t = k as f64 - centre;
        // Hamming: 0.54 - 0.46 cos(2πk/M). Kept as the two printed constants rather than as one
        // "alpha", so it is comparable line by line with any signal-processing text.
        let w = 0.54 - 0.46 * (TAU * k as f64 / m as f64).cos();
        h.push(sinc(2.0 * cutoff * t) * w);
    }
    let sum: f64 = h.iter().sum();
    if !(sum > 0.0) {
        return Err(CodeError::NoEvidence { what: "a normalisable filter" });
    }
    for v in &mut h {
        *v /= sum;
    }
    Ok(h)
}

/// Reconstruct a signal by driving `filter` with `spikes`: `s[t] = Σ_k spikes[t-k] · filter[k]`.
///
/// This is the decoder that both [`Bsa`] and [`Hsa`] use, and it is the whole decoder — the
/// asymmetry of deconvolution coding is that the encoder searches and the reader only filters,
/// which is why these schemes appear in front of hardware that has a finite impulse response block
/// and no spare cycles.
///
/// The output has the same length as `spikes`; the filter's tail past the end of the record is
/// dropped, exactly as a real-time reader would have to drop it.
#[must_use]
pub fn spike_convolve(spikes: &[bool], filter: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; spikes.len()];
    for (t, &fired) in spikes.iter().enumerate() {
        if !fired {
            continue;
        }
        for (k, &h) in filter.iter().enumerate() {
            if t + k < out.len() {
                out[t + k] += h;
            }
        }
    }
    out
}

/// Convert observed spike counts to rates in hertz over a window of `window_s` seconds.
///
/// The one line every population decoder needs before it can weight anything, kept as a free
/// function because it has no state and belongs to no one population.
///
/// # Errors
///
/// [`CodeError::NotPositive`] for a non-positive or non-finite `window_s`.
pub fn rates_from_counts(counts: &[u64], window_s: f64) -> Result<Vec<f64>, CodeError> {
    if !(window_s > 0.0) || !window_s.is_finite() {
        return Err(CodeError::NotPositive { what: "window_s", value: window_s });
    }
    Ok(counts.iter().map(|&k| k as f64 / window_s).collect())
}

// ---------------------------------------------------------------------------------------------
// POPULATION CODING
// ---------------------------------------------------------------------------------------------

/// Maximise the Poisson log-likelihood of `counts` over a grid on `[lo, hi]`, then refine.
///
/// `rate_at(i, x)` must give cell `i`'s expected rate in hertz at stimulus `x`. The likelihood
/// dropped of its `ln(k!)` term, which does not depend on `x` and so cannot move the maximum:
/// `L(x) = Σ_i [ k_i · ln(λ_i(x) · T) − λ_i(x) · T ]`.
///
/// The refinement is a parabola through the best grid point and its two neighbours, which removes
/// the grid quantisation from the estimate; without it the estimator's variance is inflated by
/// `h² / 12` and a Cramér-Rao comparison measures the grid rather than the code.
fn ml_on_grid<F>(
    counts: &[u64],
    window_s: f64,
    lo: f64,
    hi: f64,
    grid: usize,
    mut rate_at: F,
) -> Result<f64, CodeError>
where
    F: FnMut(usize, f64) -> f64,
{
    if !(window_s > 0.0) || !window_s.is_finite() {
        return Err(CodeError::NotPositive { what: "window_s", value: window_s });
    }
    if !(hi > lo) || !lo.is_finite() || !hi.is_finite() {
        return Err(CodeError::EmptyRange { lo, hi });
    }
    if grid < 3 {
        return Err(CodeError::NotPositive { what: "grid points (need at least 3)", value: grid as f64 });
    }
    let h = (hi - lo) / (grid - 1) as f64;
    let mut best = 0usize;
    let mut best_l = f64::NEG_INFINITY;
    let mut logl = vec![f64::NEG_INFINITY; grid];
    for g in 0..grid {
        let x = lo + h * g as f64;
        let mut acc = 0.0f64;
        for (i, &k) in counts.iter().enumerate() {
            let mu = rate_at(i, x) * window_s;
            if mu > 0.0 {
                acc += k as f64 * mu.ln() - mu;
            } else if k > 0 {
                // An observed spike from a cell the model says is silent has probability zero.
                acc = f64::NEG_INFINITY;
                break;
            }
        }
        logl[g] = acc;
        if acc > best_l {
            best_l = acc;
            best = g;
        }
    }
    if !best_l.is_finite() {
        return Err(CodeError::NoEvidence { what: "a stimulus consistent with these counts" });
    }
    let x0 = lo + h * best as f64;
    if best == 0 || best + 1 == grid {
        return Ok(x0);
    }
    let (y1, y2, y3) = (logl[best - 1], logl[best], logl[best + 1]);
    if !y1.is_finite() || !y3.is_finite() {
        return Ok(x0);
    }
    let denom = y1 - 2.0 * y2 + y3;
    if denom == 0.0 || !denom.is_finite() {
        return Ok(x0);
    }
    let shift = (h * (y1 - y3) / (2.0 * denom)).clamp(-h, h);
    Ok((x0 + shift).clamp(lo, hi))
}

/// A population of cells with Gaussian tuning curves evenly tiling `[lo, hi]`.
///
/// Cell `i` prefers `lo + i · (hi − lo) / (n − 1)` and fires at
/// `r_base + (r_max − r_base) · exp(−(x − φ_i)² / (2 σ²))` hertz. This is the standard
/// bell-curve population of the textbooks (Dayan & Abbott, *Theoretical Neuroscience*, 2001) and
/// the shape a neuromorphic sensor front end uses to turn one analogue channel into `n` spike
/// lines.
///
/// # Choosing `sigma`: the one parameter that matters
///
/// Narrow curves give a sharp peak and a population that is mostly silent, so precision between
/// two adjacent preferred values collapses — there is nothing active to interpolate with. Wide
/// curves keep everything active and the peak position becomes hard to localise. The usable range
/// is roughly one to three times the spacing `(hi − lo) / (n − 1)`. Measured, for 24 cells with
/// `sigma` at two spacings and the range normalised to 1: the noise-free centre-of-mass error is
/// below `2.4e-4` over the middle half of the range and `5.7e-2` at the very ends. Those two
/// numbers are four hundred times apart and both belong beside any population-coded figure.
///
/// # The edges are not the interior, and this type does not pretend otherwise
///
/// A tiling that stops has no cells beyond its last preferred value, so the centre of mass of a
/// stimulus near `hi` is pulled inward — a real bias, present in every finite population, and
/// larger than the interior error by orders of magnitude. [`GaussianPopulation::interior`] gives
/// the sub-range where the bias is below a caller-supplied tolerance, so the limitation is a
/// computable number rather than a warning.
#[derive(Debug, Clone, PartialEq)]
pub struct GaussianPopulation {
    /// Number of cells. At least 1; at least 3 before any decoder is informative.
    pub n: usize,
    /// Lower end of the encoded range, in the stimulus's own units.
    pub lo: f64,
    /// Upper end of the encoded range, strictly above `lo`.
    pub hi: f64,
    /// Tuning-curve standard deviation, in stimulus units. See the type doc for the usable range.
    pub sigma: f64,
    /// Peak firing rate at a cell's preferred value, hertz.
    pub r_max: f64,
    /// Baseline firing rate far from the preferred value, hertz. Non-negative; a strictly positive
    /// baseline is what makes the Poisson likelihood finite everywhere, and what the
    /// centre-of-mass decoder must subtract before it weights anything.
    pub r_base: f64,
}

impl GaussianPopulation {
    /// Build and validate.
    ///
    /// # Errors
    ///
    /// [`CodeError::Empty`] for `n == 0`; [`CodeError::EmptyRange`] unless `lo < hi` and both are
    /// finite; [`CodeError::NotPositive`] for a non-positive `sigma` or a negative `r_base`;
    /// [`CodeError::NotFinite`] for any non-finite parameter.
    pub fn new(
        n: usize,
        lo: f64,
        hi: f64,
        sigma: f64,
        r_max: f64,
        r_base: f64,
    ) -> Result<Self, CodeError> {
        if n == 0 {
            return Err(CodeError::Empty { what: "population" });
        }
        check_finite("population parameter", &[lo, hi, sigma, r_max, r_base])?;
        if !(hi > lo) {
            return Err(CodeError::EmptyRange { lo, hi });
        }
        if !(sigma > 0.0) {
            return Err(CodeError::NotPositive { what: "sigma", value: sigma });
        }
        if r_base < 0.0 {
            return Err(CodeError::NotPositive { what: "r_base (may be zero, not negative)", value: r_base });
        }
        if !(r_max > r_base) {
            return Err(CodeError::NotPositive { what: "r_max - r_base", value: r_max - r_base });
        }
        Ok(Self { n, lo, hi, sigma, r_max, r_base })
    }

    /// Spacing between adjacent preferred values, stimulus units. `hi − lo` for a single cell.
    #[must_use]
    pub fn spacing(&self) -> f64 {
        if self.n <= 1 { self.hi - self.lo } else { (self.hi - self.lo) / (self.n - 1) as f64 }
    }

    /// Preferred value of cell `i`. Cell 0 sits at `lo` and cell `n − 1` at `hi`.
    ///
    /// # Panics
    ///
    /// If `i >= n`. A preferred value for a cell that is not in the population is not a number to
    /// be invented; the caller has an index bug and should see it here.
    #[must_use]
    pub fn preferred(&self, i: usize) -> f64 {
        assert!(i < self.n, "cell {i} is not in a population of {}", self.n);
        if self.n == 1 { 0.5 * (self.lo + self.hi) } else { self.lo + self.spacing() * i as f64 }
    }

    /// Expected rate of cell `i` at stimulus `x`, hertz. Noise-free: this is the tuning curve.
    ///
    /// # Panics
    ///
    /// If `i >= n`, for the reason given on [`GaussianPopulation::preferred`].
    #[must_use]
    pub fn rate_of(&self, i: usize, x: f64) -> f64 {
        let d = x - self.preferred(i);
        self.r_base + (self.r_max - self.r_base) * (-(d * d) / (2.0 * self.sigma * self.sigma)).exp()
    }

    /// The whole noise-free tuning profile at `x`, hertz, one entry per cell.
    ///
    /// # Errors
    ///
    /// [`CodeError::NotFinite`] if `x` is not finite.
    pub fn rates(&self, x: f64) -> Result<Vec<f64>, CodeError> {
        if !x.is_finite() {
            return Err(CodeError::NotFinite { what: "stimulus", index: 0 });
        }
        Ok((0..self.n).map(|i| self.rate_of(i, x)).collect())
    }

    /// Draw Poisson spike counts for a `window_s`-second observation of stimulus `x`.
    ///
    /// Deterministic for a given [`Rng`] state, which is what makes the Cramér-Rao test in this
    /// module reproducible rather than flaky.
    ///
    /// # Errors
    ///
    /// [`CodeError::NotFinite`] for a non-finite `x`; [`CodeError::NotPositive`] for a non-positive
    /// `window_s`.
    pub fn sample_counts(&self, rng: &mut Rng, x: f64, window_s: f64) -> Result<Vec<u64>, CodeError> {
        if !(window_s > 0.0) || !window_s.is_finite() {
            return Err(CodeError::NotPositive { what: "window_s", value: window_s });
        }
        let rates = self.rates(x)?;
        let mut out = Vec::with_capacity(self.n);
        for r in rates {
            out.push(poisson_count(rng, r * window_s).ok_or(CodeError::NotFinite {
                what: "expected count",
                index: out.len(),
            })?);
        }
        Ok(out)
    }

    /// Encode `x` as a spike [`Train`] over `ticks` ticks of `dt` seconds, cell `i` on source `i`.
    ///
    /// Bernoulli per tick with probability `1 − exp(−rate · dt)`, the same exact form
    /// [`crate::encode::RateEncoder`] uses and for the same reason: `rate · dt` is a probability
    /// greater than one as soon as the tick is coarse relative to the peak rate.
    ///
    /// # Errors
    ///
    /// [`CodeError::NotFinite`] for a non-finite `x`; [`CodeError::NotPositive`] for a non-positive
    /// `dt`.
    pub fn encode_train(
        &self,
        rng: &mut Rng,
        x: f64,
        ticks: u64,
        dt: f64,
    ) -> Result<Train, CodeError> {
        if !(dt > 0.0) || !dt.is_finite() {
            return Err(CodeError::NotPositive { what: "dt", value: dt });
        }
        let rates = self.rates(x)?;
        let p: Vec<f64> = rates.iter().map(|r| 1.0 - (-r * dt).exp()).collect();
        let mut spikes = Vec::new();
        for t in 0..ticks {
            for i in 0..self.n {
                if rng.next_f64() < p[i] {
                    spikes.push(Spike { t, source: i as u32 });
                }
            }
        }
        Ok(Train::from_spikes(spikes))
    }

    /// Centre-of-mass decode: the baseline-subtracted, rate-weighted mean of the preferred values.
    ///
    /// `x̂ = Σ_i (r_i − r_base)⁺ φ_i / Σ_i (r_i − r_base)⁺`, with the positive part taken because a
    /// noisy count below baseline is evidence of nothing and a negative weight would drag the
    /// estimate away from the peak rather than toward it.
    ///
    /// This is the linear analogue of the population-vector decoder of Georgopoulos, Schwartz &
    /// Kettner (*Science* 233:1416-1419, 1986) — one multiply-add per cell, no search, no noise
    /// model. It is also demonstrably worse than [`GaussianPopulation::decode_max_likelihood`],
    /// and this module measures by how much rather than asserting it.
    ///
    /// # Errors
    ///
    /// [`CodeError::LengthMismatch`] if `rates` is not `n` long, [`CodeError::NotFinite`] for a
    /// non-finite rate, and [`CodeError::NoEvidence`] when no cell exceeds baseline — a silent
    /// population has no decoded value, and reporting the midpoint of the range would be a
    /// plausible number produced by no stimulus at all.
    pub fn decode_center_of_mass(&self, rates: &[f64]) -> Result<f64, CodeError> {
        if rates.len() != self.n {
            return Err(CodeError::LengthMismatch { expected: self.n, got: rates.len() });
        }
        check_finite("rate", rates)?;
        let mut num = 0.0;
        let mut den = 0.0;
        for (i, &r) in rates.iter().enumerate() {
            let w = (r - self.r_base).max(0.0);
            num += w * self.preferred(i);
            den += w;
        }
        if !(den > 0.0) {
            return Err(CodeError::NoEvidence { what: "a stimulus (no cell exceeded baseline)" });
        }
        Ok(num / den)
    }

    /// Maximum-likelihood decode of Poisson `counts` observed over `window_s` seconds.
    ///
    /// Searches `grid` equally spaced points across `[lo, hi]` and refines with a parabola. Unlike
    /// [`GaussianPopulation::decode_center_of_mass`] this estimator knows the noise model, and it
    /// attains the Cramér-Rao bound `1 / I(x)` in the large-count limit — which is the check this
    /// module runs, against [`GaussianPopulation::fisher_information`] computed in closed form.
    ///
    /// # Errors
    ///
    /// [`CodeError::LengthMismatch`] for the wrong number of counts; [`CodeError::NotPositive`] for
    /// a non-positive `window_s` or fewer than three grid points; [`CodeError::NoEvidence`] if no
    /// grid point has finite likelihood, which happens only when a cell with a zero modelled rate
    /// was observed to spike.
    pub fn decode_max_likelihood(
        &self,
        counts: &[u64],
        window_s: f64,
        grid: usize,
    ) -> Result<f64, CodeError> {
        if counts.len() != self.n {
            return Err(CodeError::LengthMismatch { expected: self.n, got: counts.len() });
        }
        ml_on_grid(counts, window_s, self.lo, self.hi, grid, |i, x| self.rate_of(i, x))
    }

    /// Fisher information about `x` carried by one `window_s`-second Poisson observation, in
    /// inverse squared stimulus units.
    ///
    /// `I(x) = T · Σ_i λ_i′(x)² / λ_i(x)` for independent Poisson cells, with
    /// `λ_i′(x) = (r_max − r_base) · g_i(x) · (φ_i − x) / σ²`. The Cramér-Rao bound says no
    /// unbiased decoder can have variance below `1 / I(x)`, so this is the number a decoder is
    /// graded against — and it is the reason `r_base > 0` is worth having, since `λ_i` in the
    /// denominator makes the information infinite where a cell's modelled rate reaches zero.
    ///
    /// # Errors
    ///
    /// [`CodeError::NotFinite`] for a non-finite `x`; [`CodeError::NotPositive`] for a non-positive
    /// `window_s`; [`CodeError::NoEvidence`] when the information is zero, which is what a
    /// population that is entirely flat at `x` actually carries.
    pub fn fisher_information(&self, x: f64, window_s: f64) -> Result<f64, CodeError> {
        if !x.is_finite() {
            return Err(CodeError::NotFinite { what: "stimulus", index: 0 });
        }
        if !(window_s > 0.0) || !window_s.is_finite() {
            return Err(CodeError::NotPositive { what: "window_s", value: window_s });
        }
        let s2 = self.sigma * self.sigma;
        let amp = self.r_max - self.r_base;
        let mut info = 0.0;
        for i in 0..self.n {
            let d = x - self.preferred(i);
            let g = (-(d * d) / (2.0 * s2)).exp();
            let lam = self.r_base + amp * g;
            if lam <= 0.0 {
                continue;
            }
            let dlam = amp * g * (-d) / s2;
            info += dlam * dlam / lam;
        }
        let info = info * window_s;
        if !(info > 0.0) {
            return Err(CodeError::NoEvidence { what: "any information about the stimulus" });
        }
        Ok(info)
    }

    /// The sub-range over which the noise-free centre-of-mass decode is accurate to `tol`.
    ///
    /// Computed, not asserted: the interior is found by scanning inward from each end until the
    /// noise-free round-trip error falls below `tol` and stays there. Returns `None` when no such
    /// sub-range exists, which is the honest answer for a population of one or two cells, or for a
    /// `sigma` so wide that the whole range is edge.
    ///
    /// The number this returns is the one to quote beside a population-coded result. Quoting the
    /// full `[lo, hi]` overstates the usable range of every finite tiling ever built.
    #[must_use]
    pub fn interior(&self, tol: f64) -> Option<(f64, f64)> {
        if self.n < 3 || !(tol > 0.0) {
            return None;
        }
        let steps = 400usize;
        let h = (self.hi - self.lo) / steps as f64;
        let ok = |x: f64| {
            self.rates(x)
                .ok()
                .and_then(|r| self.decode_center_of_mass(&r).ok())
                .is_some_and(|xh| (xh - x).abs() <= tol)
        };
        let mut a = None;
        for s in 0..=steps {
            if ok(self.lo + h * s as f64) {
                a = Some(self.lo + h * s as f64);
                break;
            }
        }
        let mut b = None;
        for s in (0..=steps).rev() {
            if ok(self.lo + h * s as f64) {
                b = Some(self.lo + h * s as f64);
                break;
            }
        }
        match (a, b) {
            (Some(a), Some(b)) if b > a => Some((a, b)),
            _ => None,
        }
    }
}

/// A circular population with cosine tuning: the motor-cortex code, and its population vector.
///
/// Cell `i` prefers direction `offset + 2π i / n` and fires at `b0 + b1 · cos(θ − φ_i)` hertz.
/// Georgopoulos, Schwartz & Kettner (*Science* 233:1416-1419, 1986) fitted exactly this form to
/// motor cortical cells during reaching, then decoded the reach direction as the vector sum of
/// preferred directions weighted by firing rate — the population vector.
///
/// # Why this one is exact, and where the exactness comes from
///
/// For preferred directions that tile the circle evenly with `n ≥ 3`, the population vector of the
/// noise-free rates is
///
/// ```text
/// Σ_i [b0 + b1 cos(θ − φ_i)] · (cos φ_i, sin φ_i) = (b1 n / 2) · (cos θ, sin θ)
/// ```
///
/// because `Σ_i (cos φ_i, sin φ_i) = 0` and `Σ_i cos(θ − 2φ_i) = 0` for any `n ≥ 3`. Two
/// consequences are worth stating separately. The decoded angle is `θ` **exactly**, to
/// floating-point noise, at every `n ≥ 3` — this is not an approximation that improves with cells.
/// And the baseline `b0` cancels **completely**: a population with a 100 Hz baseline and a 1 Hz
/// modulation decodes the same angle as one with no baseline at all, though it needs far longer to
/// do it against Poisson noise. The tests assert both.
///
/// # What breaks it
///
/// Uneven preferred directions (`Σ_i u_i ≠ 0`), and rectification. Real cells cannot fire at a
/// negative rate, so a population with `b1 > b0` is half-wave rectified in the animal, and the
/// cancellation above no longer holds — the decoder acquires a small angle-dependent ripple.
/// This type therefore requires `b1 ≤ b0` so its rates are non-negative without rectification, and
/// the test `rectifying_a_cosine_population_costs_a_measurable_angular_ripple` measures what
/// rectification would have cost instead of leaving it as a caveat.
#[derive(Debug, Clone, PartialEq)]
pub struct CosinePopulation {
    /// Number of cells; at least 3 for the population vector to be defined.
    pub n: usize,
    /// Baseline rate, hertz. Must be at least `b1`, so rates stay non-negative.
    pub b0: f64,
    /// Modulation depth, hertz: the rate swings `±b1` about `b0` around the circle.
    pub b1: f64,
    /// Rotation of the whole preferred-direction tiling, radians. Zero puts cell 0 at angle 0.
    pub offset: f64,
}

impl CosinePopulation {
    /// Build and validate.
    ///
    /// # Errors
    ///
    /// [`CodeError::Empty`] for fewer than three cells (a population vector needs a spanning set,
    /// and two opposed cells span a line); [`CodeError::NotFinite`] for a non-finite parameter;
    /// [`CodeError::NotPositive`] for `b1 <= 0` or for `b1 > b0`, which would make some cell's
    /// modelled rate negative.
    pub fn new(n: usize, b0: f64, b1: f64, offset: f64) -> Result<Self, CodeError> {
        if n < 3 {
            return Err(CodeError::Empty { what: "population (a population vector needs n >= 3)" });
        }
        check_finite("population parameter", &[b0, b1, offset])?;
        if !(b1 > 0.0) {
            return Err(CodeError::NotPositive { what: "b1", value: b1 });
        }
        if b1 > b0 {
            return Err(CodeError::NotPositive { what: "b0 - b1 (rates would go negative)", value: b0 - b1 });
        }
        Ok(Self { n, b0, b1, offset })
    }

    /// Preferred direction of cell `i`, radians in `[0, 2π)`.
    ///
    /// # Panics
    ///
    /// If `i >= n`.
    #[must_use]
    pub fn preferred(&self, i: usize) -> f64 {
        assert!(i < self.n, "cell {i} is not in a population of {}", self.n);
        wrap_angle(self.offset + TAU * i as f64 / self.n as f64)
    }

    /// Expected rate of cell `i` at direction `theta`, hertz.
    ///
    /// # Panics
    ///
    /// If `i >= n`.
    #[must_use]
    pub fn rate_of(&self, i: usize, theta: f64) -> f64 {
        self.b0 + self.b1 * (theta - self.preferred(i)).cos()
    }

    /// The whole noise-free tuning profile at `theta`, hertz.
    ///
    /// # Errors
    ///
    /// [`CodeError::NotFinite`] if `theta` is not finite.
    pub fn rates(&self, theta: f64) -> Result<Vec<f64>, CodeError> {
        if !theta.is_finite() {
            return Err(CodeError::NotFinite { what: "direction", index: 0 });
        }
        Ok((0..self.n).map(|i| self.rate_of(i, theta)).collect())
    }

    /// Draw Poisson counts for a `window_s`-second observation of direction `theta`.
    ///
    /// # Errors
    ///
    /// [`CodeError::NotFinite`] for a non-finite `theta`; [`CodeError::NotPositive`] for a
    /// non-positive `window_s`.
    pub fn sample_counts(
        &self,
        rng: &mut Rng,
        theta: f64,
        window_s: f64,
    ) -> Result<Vec<u64>, CodeError> {
        if !(window_s > 0.0) || !window_s.is_finite() {
            return Err(CodeError::NotPositive { what: "window_s", value: window_s });
        }
        let rates = self.rates(theta)?;
        let mut out = Vec::with_capacity(self.n);
        for r in rates {
            let lam = (r * window_s).max(0.0);
            out.push(poisson_count(rng, lam).ok_or(CodeError::NotFinite {
                what: "expected count",
                index: out.len(),
            })?);
        }
        Ok(out)
    }

    /// The raw population vector `Σ_i r_i · (cos φ_i, sin φ_i)`, in hertz.
    ///
    /// Returned rather than folded straight into an angle because its **length** is the readout's
    /// confidence: a long vector is a population that agrees, a short one is a population that is
    /// either silent or uniformly driven, and the difference is invisible once `atan2` has been
    /// taken.
    ///
    /// # Errors
    ///
    /// [`CodeError::LengthMismatch`] if `rates` is not `n` long; [`CodeError::NotFinite`] for a
    /// non-finite rate.
    pub fn population_vector(&self, rates: &[f64]) -> Result<(f64, f64), CodeError> {
        if rates.len() != self.n {
            return Err(CodeError::LengthMismatch { expected: self.n, got: rates.len() });
        }
        check_finite("rate", rates)?;
        let mut x = 0.0;
        let mut y = 0.0;
        for (i, &r) in rates.iter().enumerate() {
            let p = self.preferred(i);
            x += r * p.cos();
            y += r * p.sin();
        }
        Ok((x, y))
    }

    /// Decode the direction as the population vector's angle, radians in `[0, 2π)`.
    ///
    /// # Errors
    ///
    /// [`CodeError::LengthMismatch`] or [`CodeError::NotFinite`] as
    /// [`CosinePopulation::population_vector`], and [`CodeError::NoEvidence`] when the vector's
    /// length is below `1e-12 · n · max|r|` — a population firing equally in every direction
    /// carries **no** direction, and `atan2(0, 0)` is `0.0`, which is due east and would be plotted
    /// as a result.
    pub fn decode_population_vector(&self, rates: &[f64]) -> Result<f64, CodeError> {
        let (x, y) = self.population_vector(rates)?;
        let scale = rates.iter().fold(0.0f64, |a, b| a.max(b.abs())) * self.n as f64;
        let len = (x * x + y * y).sqrt();
        if len <= 1e-12 * scale.max(1.0) {
            return Err(CodeError::NoEvidence { what: "a direction (the population vector vanished)" });
        }
        Ok(wrap_angle(y.atan2(x)))
    }

    /// Maximum-likelihood decode of Poisson `counts` over `window_s` seconds, radians in `[0, 2π)`.
    ///
    /// The grid spans a full turn. The wrap is handled by searching `[0, 2π]` inclusive and
    /// wrapping the answer, which costs one duplicated grid point and avoids the usual bug of a
    /// circular estimator that cannot return an angle near zero.
    ///
    /// # Errors
    ///
    /// [`CodeError::LengthMismatch`] for the wrong count length; [`CodeError::NotPositive`] for a
    /// non-positive `window_s` or fewer than three grid points; [`CodeError::NoEvidence`] if no
    /// grid point has finite likelihood.
    pub fn decode_max_likelihood(
        &self,
        counts: &[u64],
        window_s: f64,
        grid: usize,
    ) -> Result<f64, CodeError> {
        if counts.len() != self.n {
            return Err(CodeError::LengthMismatch { expected: self.n, got: counts.len() });
        }
        let theta = ml_on_grid(counts, window_s, 0.0, TAU, grid, |i, x| self.rate_of(i, x).max(0.0))?;
        Ok(wrap_angle(theta))
    }
}

// ---------------------------------------------------------------------------------------------
// RANK-ORDER CODING
// ---------------------------------------------------------------------------------------------

/// The result of a rank-order encode: who fired, and in what order.
///
/// The magnitudes are gone. That is not an omission — it is the code.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RankOrder {
    /// Cell indices in firing order, most strongly driven first. Only cells that fired appear.
    pub order: Vec<u32>,
    /// `rank[i]` is `Some(k)` when cell `i` fired `k`-th (0-based) and `None` when it never fired.
    ///
    /// Always `n` long, so a consumer can index it by cell without a bounds question. The `None`
    /// is load-bearing: a cell with zero drive has no rank, and giving it rank `n` would place a
    /// silent cell in the code as though it had spoken last.
    pub rank: Vec<Option<u32>>,
}

impl RankOrder {
    /// How many cells fired.
    #[must_use]
    pub fn fired(&self) -> usize {
        self.order.len()
    }

    /// Whether nothing fired at all, in which case there is no code to read.
    #[must_use]
    pub fn is_silent(&self) -> bool {
        self.order.is_empty()
    }
}

/// Rank-order coding: the information is in the ORDER of the first spikes (Thorpe & Gautrais,
/// *Rank order coding*, in *Computational Neuroscience: Trends in Research*, Plenum, 1998).
///
/// Each cell fires at most once. The most strongly driven fires first, the next-most second, and
/// the message is the permutation. A downstream cell reads it with a *desensitising* weight: the
/// `k`-th arrival is multiplied by `modulation^k`, so early spikes dominate and the same synaptic
/// weights produce a different answer for a different arrival order — which is how one set of
/// weights becomes a pattern detector over sequences.
///
/// # Contrast invariance, precisely
///
/// Multiply every input by any constant `c > 0` and the encoded order is **bit-identical**. No
/// other code in this crate has that: a rate code's counts all scale, a latency code's times all
/// move, a burst code's counts all change, and a temporal-contrast code's event stream changes
/// completely. This is why rank order is the classical answer to how a retina can recognise a
/// scene in 100 ms across four orders of magnitude of illumination.
///
/// The floating-point caveat, stated rather than hidden: the invariance is exact in real
/// arithmetic, and holds in `f64` as long as the rescaling does not collapse two distinct inputs
/// onto the same float or flush one to zero. Powers of two rescale exactly; a rescaling that
/// pushes inputs into the subnormal range can create a tie, and ties here break by cell index.
///
/// # What it costs
///
/// `n` cells firing once carry at most `log2(n!)` bits ([`RankOrderCode::capacity_bits`]) and zero
/// bits of magnitude. [`RankOrderCode::decode_amplitudes`] returns amplitudes only by *assuming*
/// the geometric law `modulation^k`; it is exact when the input followed that law and arbitrarily
/// wrong otherwise, which the tests show both ways round.
#[derive(Debug, Clone, PartialEq)]
pub struct RankOrderCode {
    /// Number of cells the code is defined over. Inputs must be exactly this long.
    pub n: usize,
    /// Desensitisation factor per rank, in `(0, 1]`. Thorpe's readout weights the `k`-th arrival
    /// by `modulation^k`; 1.0 turns the readout into a plain sum and discards the order, which is
    /// allowed and is the degenerate case worth being able to write down.
    pub modulation: f64,
}

impl RankOrderCode {
    /// Build and validate.
    ///
    /// # Errors
    ///
    /// [`CodeError::Empty`] for `n == 0`; [`CodeError::NotPositive`] for a `modulation` outside
    /// `(0, 1]` or not finite.
    pub fn new(n: usize, modulation: f64) -> Result<Self, CodeError> {
        if n == 0 {
            return Err(CodeError::Empty { what: "population" });
        }
        if !modulation.is_finite() || modulation <= 0.0 || modulation > 1.0 {
            return Err(CodeError::NotPositive { what: "modulation (must be in (0, 1])", value: modulation });
        }
        Ok(Self { n, modulation })
    }

    /// The code's capacity in bits: `log2(n!)`, summed term by term rather than via a factorial
    /// that overflows at `n = 21`.
    #[must_use]
    pub fn capacity_bits(&self) -> f64 {
        (1..=self.n).map(|k| (k as f64).log2()).sum()
    }

    /// Encode drives into an order. A cell with drive `<= 0` does not fire and has no rank.
    ///
    /// Ties break by ascending cell index. That is a decision, not an accident: the alternative,
    /// leaving the order of equal drives to the sort's internal state, makes the encoder
    /// non-deterministic across implementations while every test still passes.
    ///
    /// # Errors
    ///
    /// [`CodeError::LengthMismatch`] if `x` is not `n` long; [`CodeError::NotFinite`] for a
    /// non-finite drive.
    pub fn encode(&self, x: &[f64]) -> Result<RankOrder, CodeError> {
        if x.len() != self.n {
            return Err(CodeError::LengthMismatch { expected: self.n, got: x.len() });
        }
        check_finite("drive", x)?;
        let mut idx: Vec<u32> = (0..self.n as u32).filter(|&i| x[i as usize] > 0.0).collect();
        // `total_cmp` rather than `partial_cmp().unwrap()`: a total order over every f64, so the
        // sort is deterministic even if a caller later relaxes the finiteness check.
        idx.sort_unstable_by(|&a, &b| {
            x[b as usize].total_cmp(&x[a as usize]).then(a.cmp(&b))
        });
        let mut rank = vec![None; self.n];
        for (k, &i) in idx.iter().enumerate() {
            rank[i as usize] = Some(k as u32);
        }
        Ok(RankOrder { order: idx, rank })
    }

    /// Lay the order out as spikes: rank `k` fires at `first_tick + k · gap_ticks`.
    ///
    /// `gap_ticks` of zero puts every spike on the same tick, which erases the code — allowed so
    /// the degenerate case can be written down and tested, and documented here so it is not
    /// reached by accident.
    #[must_use]
    pub fn to_train(ro: &RankOrder, first_tick: u64, gap_ticks: u64) -> Train {
        let spikes = ro
            .order
            .iter()
            .enumerate()
            .map(|(k, &i)| Spike { t: first_tick + gap_ticks * k as u64, source: i })
            .collect();
        Train::from_spikes(spikes)
    }

    /// Recover the order from a real spike train, using each source's FIRST spike only.
    ///
    /// This is the decoder a downstream layer actually implements, and it is why rank order
    /// tolerates a network that spikes more than once: everything after the first spike of a cell
    /// is ignored. Sources at or past `n` are ignored rather than treated as an error, so a train
    /// from a larger network can be read through a smaller code.
    ///
    /// Ties on the same tick break by ascending source index, matching
    /// [`RankOrderCode::encode`].
    #[must_use]
    pub fn from_train(&self, train: &Train) -> RankOrder {
        let mut first: Vec<Option<u64>> = vec![None; self.n];
        for s in train.spikes() {
            let i = s.source as usize;
            if i < self.n && first[i].is_none() {
                first[i] = Some(s.t);
            }
        }
        let mut idx: Vec<u32> = (0..self.n as u32).filter(|&i| first[i as usize].is_some()).collect();
        idx.sort_unstable_by(|&a, &b| {
            first[a as usize].cmp(&first[b as usize]).then(a.cmp(&b))
        });
        let mut rank = vec![None; self.n];
        for (k, &i) in idx.iter().enumerate() {
            rank[i as usize] = Some(k as u32);
        }
        RankOrder { order: idx, rank }
    }

    /// Reconstruct amplitudes under the assumption that the input obeyed `modulation^k`.
    ///
    /// Cell of rank `k` gets `modulation^k`; a silent cell gets `0.0`. Exact — to floating-point
    /// noise — when the input really was geometric with this ratio, up to the overall scale that
    /// the code by construction cannot know. Otherwise it recovers the order and nothing else, and
    /// `rank_order_recovers_order_exactly_and_magnitude_not_at_all` is the test that says so with
    /// numbers.
    #[must_use]
    pub fn decode_amplitudes(&self, ro: &RankOrder) -> Vec<f64> {
        ro.rank
            .iter()
            .map(|r| r.map_or(0.0, |k| self.modulation.powi(k as i32)))
            .collect()
    }

    /// Thorpe's rank-order readout: `Σ_i w_i · modulation^rank(i)` over the cells that fired.
    ///
    /// One number per pattern, computed with one multiply-add per arriving spike and no state
    /// beyond a running rank counter. A cell with these weights responds maximally to exactly one
    /// order of arrival, which is the property that makes rank order a *recognition* code rather
    /// than only a compression.
    ///
    /// # Errors
    ///
    /// [`CodeError::LengthMismatch`] if `weights` is not `n` long; [`CodeError::NotFinite`] for a
    /// non-finite weight.
    pub fn readout(&self, ro: &RankOrder, weights: &[f64]) -> Result<f64, CodeError> {
        if weights.len() != self.n {
            return Err(CodeError::LengthMismatch { expected: self.n, got: weights.len() });
        }
        check_finite("weight", weights)?;
        let mut acc = 0.0;
        for (k, &i) in ro.order.iter().enumerate() {
            acc += weights[i as usize] * self.modulation.powi(k as i32);
        }
        Ok(acc)
    }
}

// ---------------------------------------------------------------------------------------------
// PHASE CODING
// ---------------------------------------------------------------------------------------------

/// A background oscillation that a phase code measures against.
///
/// `value(t) = amplitude · cos(2π f t + phase0)`. The amplitude is carried so a caller can plot or
/// inject the reference waveform — it is deliberately **not** used by [`PhaseEncoder`], which is
/// the property that makes phase coding amplitude-invariant.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Oscillator {
    /// Frequency, hertz. Hippocampal theta is 4-10 Hz; a gamma reference is 30-80 Hz.
    pub f_hz: f64,
    /// Peak amplitude in whatever unit the reference is carried in (volts, usually). It affects
    /// [`Oscillator::value_at`] and nothing else.
    pub amplitude: f64,
    /// Phase at `t = 0`, radians. This is the reference every receiver must agree on, and the
    /// quantity a phase code is exactly as wrong as.
    pub phase0: f64,
}

impl Oscillator {
    /// Build and validate.
    ///
    /// # Errors
    ///
    /// [`CodeError::NotFinite`] for a non-finite parameter; [`CodeError::NotPositive`] for a
    /// frequency that is not strictly positive.
    pub fn new(f_hz: f64, amplitude: f64, phase0: f64) -> Result<Self, CodeError> {
        check_finite("oscillator parameter", &[f_hz, amplitude, phase0])?;
        if !(f_hz > 0.0) {
            return Err(CodeError::NotPositive { what: "f_hz", value: f_hz });
        }
        Ok(Self { f_hz, amplitude, phase0 })
    }

    /// Period, seconds.
    #[must_use]
    pub fn period_s(&self) -> f64 {
        1.0 / self.f_hz
    }

    /// Phase at time `t_s` seconds, wrapped into `[0, 2π)`.
    #[must_use]
    pub fn phase_at(&self, t_s: f64) -> f64 {
        wrap_angle(TAU * self.f_hz * t_s + self.phase0)
    }

    /// The reference waveform at `t_s` seconds. The only method the amplitude appears in.
    #[must_use]
    pub fn value_at(&self, t_s: f64) -> f64 {
        self.amplitude * (TAU * self.f_hz * t_s + self.phase0).cos()
    }
}

/// Phase coding: a value is where in the oscillation cycle the spike lands.
///
/// `x` in `[0, 1]` maps to phase `2π x` measured from the oscillation's own zero, so `x = 0` fires
/// at the reference's peak and `x = 0.5` half a cycle later. One spike per cycle carries the
/// value, and the receiver needs no timestamp of its own — it already has the oscillation
/// (O'Keefe & Recce, *Hippocampus* 3:317-330, 1993; Hopfield, *Nature* 376:33-36, 1995).
///
/// # The two invariances, and they are not the same
///
/// **Amplitude**: [`PhaseEncoder::encode_tick`] never reads [`Oscillator::amplitude`]. Multiply the
/// reference by a thousand and every spike tick is unchanged, bit for bit.
///
/// **Phase**: shift [`Oscillator::phase0`] by `δ` and every decoded value shifts by exactly
/// `δ / 2π`, modulo 1. A phase code inherits its reference's error at unity gain, with no
/// averaging and no attenuation, which is the whole argument for and against the scheme in one
/// equation. The test asserts this as an equality to `1e-12`, not as a tendency.
///
/// # Resolution
///
/// A cycle holds `1 / (f · dt)` ticks, so the code is quantised at
/// [`PhaseEncoder::quantisation`] — `0.5 / ticks_per_cycle` of full scale. At 8 Hz theta and a
/// 0.1 ms tick that is 1250 ticks per cycle and a resolution of 4 parts in 10,000; at 40 Hz gamma
/// and a 1 ms tick it is 25 ticks and 2%. The number is worth computing before the code is chosen.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhaseEncoder {
    /// The reference oscillation.
    pub osc: Oscillator,
    /// Simulation tick length, seconds. Must give at least two ticks per cycle.
    pub dt: f64,
}

impl PhaseEncoder {
    /// Build and validate.
    ///
    /// # Errors
    ///
    /// [`CodeError::NotPositive`] for a non-positive `dt`, or for a `dt` so coarse that fewer than
    /// two ticks fit in a cycle — below that the phase is not merely imprecise, it is aliased, and
    /// an encoder that accepted it would report a phase that moves backwards as the frequency
    /// rises.
    pub fn new(osc: Oscillator, dt: f64) -> Result<Self, CodeError> {
        if !dt.is_finite() || dt <= 0.0 {
            return Err(CodeError::NotPositive { what: "dt", value: dt });
        }
        let tpc = 1.0 / (osc.f_hz * dt);
        if !(tpc >= 2.0) {
            return Err(CodeError::NotPositive { what: "ticks per cycle (need at least 2)", value: tpc });
        }
        Ok(Self { osc, dt })
    }

    /// Ticks in one cycle of the reference, as a real number — it need not be an integer.
    #[must_use]
    pub fn ticks_per_cycle(&self) -> f64 {
        1.0 / (self.osc.f_hz * self.dt)
    }

    /// Worst-case round-trip error from tick quantisation, in units of full scale.
    #[must_use]
    pub fn quantisation(&self) -> f64 {
        0.5 / self.ticks_per_cycle()
    }

    /// The tick at which `x` fires within cycle number `cycle`.
    ///
    /// # Errors
    ///
    /// [`CodeError::OutOfUnitRange`] or [`CodeError::NotFinite`] for an `x` outside `[0, 1]`;
    /// [`CodeError::NotFinite`] with `what: "spike tick"` if the requested cycle is so far out that
    /// the tick index is not representable, which is a real limit at `u64` and is refused rather
    /// than wrapped.
    pub fn encode_tick(&self, x: f64, cycle: u64) -> Result<u64, CodeError> {
        check_unit("stimulus", x)?;
        let tpc = self.ticks_per_cycle();
        let frac = (x - self.osc.phase0 / TAU).rem_euclid(1.0);
        let t = ((cycle as f64 + frac) * tpc).round();
        if !t.is_finite() || t < 0.0 || t >= 9.0e18 {
            return Err(CodeError::NotFinite { what: "spike tick", index: 0 });
        }
        Ok(t as u64)
    }

    /// Recover the value a spike at tick `t` encodes, in `[0, 1)`.
    ///
    /// Which cycle the spike fell in is irrelevant and is discarded, which is the point: a phase
    /// code is read without knowing how long the receiver has been listening.
    #[must_use]
    pub fn decode_tick(&self, t: u64) -> f64 {
        let tpc = self.ticks_per_cycle();
        let frac = (t as f64 / tpc).rem_euclid(1.0);
        (frac + self.osc.phase0 / TAU).rem_euclid(1.0)
    }

    /// Encode a vector: value `i` fires once on source `i` within `cycle`.
    ///
    /// # Errors
    ///
    /// As [`PhaseEncoder::encode_tick`], for the first offending value.
    pub fn encode(&self, x: &[f64], cycle: u64) -> Result<Train, CodeError> {
        let mut spikes = Vec::with_capacity(x.len());
        for (i, &v) in x.iter().enumerate() {
            if !v.is_finite() {
                return Err(CodeError::NotFinite { what: "stimulus", index: i });
            }
            spikes.push(Spike { t: self.encode_tick(v, cycle)?, source: i as u32 });
        }
        Ok(Train::from_spikes(spikes))
    }

    /// Decode a train back to `n` values, using each source's first spike.
    ///
    /// `None` for a source that never fired: a phase code has nothing to say about a silent line,
    /// and the value 0.0 would be a specific, wrong, plottable phase.
    #[must_use]
    pub fn decode_train(&self, train: &Train, n: usize) -> Vec<Option<f64>> {
        let mut out = vec![None; n];
        for s in train.spikes() {
            let i = s.source as usize;
            if i < n && out[i].is_none() {
                out[i] = Some(self.decode_tick(s.t));
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------------------------
// BURST CODING
// ---------------------------------------------------------------------------------------------

/// One detected burst: where it started, where it ended, and how many spikes it held.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Burst {
    /// Tick of the burst's first spike.
    pub start: u64,
    /// Tick of the burst's last spike; equal to `start` for a single-spike burst.
    pub end: u64,
    /// Spikes in the burst. At least 1 — a burst of zero spikes is silence and is never detected.
    pub spikes: u32,
}

/// Burst coding: the message is the number of spikes in the burst.
///
/// Bursts cross unreliable synapses far better than isolated spikes, and a receiver that only has
/// to *count* is cheap and robust to the exact timing inside the burst (Izhikevich, Desai, Walcott
/// & Hoppensteadt, *Trends in Neurosciences* 26:161-167, 2003; Kepecs & Lisman, *Network:
/// Computation in Neural Systems* 14, 2003).
///
/// `x` in `[0, 1]` becomes `round(x · max_spikes)` spikes, `intra_ticks` apart, and the value is
/// read back by counting. The code carries `log2(max_spikes + 1)` bits
/// ([`BurstCode::capacity_bits`]) and is quantised at `0.5 / max_spikes` of full scale.
///
/// # The zero problem, which is the reason there are two decoders
///
/// A value of zero is *silence*, and silence is indistinguishable from a cell that was never
/// asked. [`BurstCode::decode_detected`] segments a train by gaps and therefore **cannot see a
/// zero**: it returns one value per burst it found, and encoding `[0.5, 0.0, 0.7]` gives it two.
/// [`BurstCode::decode_scheduled`] knows when each burst was due and counts spikes in that window,
/// so it recovers the zero exactly. Which one is available is a property of the link, not of the
/// code, and both are here so the difference is a choice rather than a surprise.
///
/// # What it is fragile to
///
/// One lost spike is a full quantum of error — `1 / max_spikes` of full scale — with no
/// redundancy to absorb it. Burst coding buys robustness *of transmission* and spends it on
/// precision; it is not a robust code for the value it carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BurstCode {
    /// Spikes representing full scale. The code has `max_spikes + 1` distinguishable values.
    pub max_spikes: u32,
    /// Ticks between successive spikes inside one burst. At least 1.
    pub intra_ticks: u64,
    /// Ticks of silence that separate two bursts. Must exceed `intra_ticks`, or the segmenter
    /// cannot tell a gap between bursts from a gap inside one.
    pub gap_ticks: u64,
}

impl BurstCode {
    /// Build and validate.
    ///
    /// # Errors
    ///
    /// [`CodeError::NotPositive`] for `max_spikes == 0`, `intra_ticks == 0`, or
    /// `gap_ticks <= intra_ticks`.
    pub fn new(max_spikes: u32, intra_ticks: u64, gap_ticks: u64) -> Result<Self, CodeError> {
        if max_spikes == 0 {
            return Err(CodeError::NotPositive { what: "max_spikes", value: 0.0 });
        }
        if intra_ticks == 0 {
            return Err(CodeError::NotPositive { what: "intra_ticks", value: 0.0 });
        }
        if gap_ticks <= intra_ticks {
            return Err(CodeError::NotPositive {
                what: "gap_ticks - intra_ticks (a gap must be longer than an intra-burst interval)",
                value: gap_ticks as f64 - intra_ticks as f64,
            });
        }
        Ok(Self { max_spikes, intra_ticks, gap_ticks })
    }

    /// Bits carried by one burst: `log2(max_spikes + 1)`.
    #[must_use]
    pub fn capacity_bits(&self) -> f64 {
        (f64::from(self.max_spikes) + 1.0).log2()
    }

    /// Worst-case round-trip error from spike quantisation, in units of full scale.
    #[must_use]
    pub fn quantisation(&self) -> f64 {
        0.5 / f64::from(self.max_spikes)
    }

    /// Ticks from one burst's start to the next's, chosen so the separation holds for any count.
    ///
    /// `(max_spikes − 1) · intra_ticks + gap_ticks`: the stride a full-scale burst needs, applied
    /// uniformly so the schedule does not depend on the values being sent — which is what lets
    /// [`BurstCode::decode_scheduled`] exist.
    #[must_use]
    pub fn stride_ticks(&self) -> u64 {
        u64::from(self.max_spikes - 1) * self.intra_ticks + self.gap_ticks
    }

    /// Spikes representing `x`.
    ///
    /// # Errors
    ///
    /// [`CodeError::OutOfUnitRange`] or [`CodeError::NotFinite`] for an `x` outside `[0, 1]`.
    pub fn spikes_for(&self, x: f64) -> Result<u32, CodeError> {
        check_unit("stimulus", x)?;
        Ok((x * f64::from(self.max_spikes)).round() as u32)
    }

    /// The value a burst of `k` spikes carries. Counts above `max_spikes` saturate at 1.0.
    #[must_use]
    pub fn value_of(&self, k: u32) -> f64 {
        (f64::from(k) / f64::from(self.max_spikes)).min(1.0)
    }

    /// Encode a sequence of values as successive bursts on one source, one stride apart.
    ///
    /// # Errors
    ///
    /// As [`BurstCode::spikes_for`], for the first offending value.
    pub fn encode_sequence(
        &self,
        x: &[f64],
        source: u32,
        start_tick: u64,
    ) -> Result<Train, CodeError> {
        let mut spikes = Vec::new();
        for (j, &v) in x.iter().enumerate() {
            let k = self.spikes_for(v)?;
            let base = start_tick + self.stride_ticks() * j as u64;
            for s in 0..u64::from(k) {
                spikes.push(Spike { t: base + self.intra_ticks * s, source });
            }
        }
        Ok(Train::from_spikes(spikes))
    }

    /// Encode a vector as one burst per source, all starting at the same tick.
    ///
    /// # Errors
    ///
    /// As [`BurstCode::spikes_for`], for the first offending value.
    pub fn encode_parallel(&self, x: &[f64], start_tick: u64) -> Result<Train, CodeError> {
        let mut spikes = Vec::new();
        for (i, &v) in x.iter().enumerate() {
            let k = self.spikes_for(v)?;
            for s in 0..u64::from(k) {
                spikes.push(Spike { t: start_tick + self.intra_ticks * s, source: i as u32 });
            }
        }
        Ok(Train::from_spikes(spikes))
    }

    /// Segment one source's spikes into bursts, breaking wherever the gap reaches `gap_ticks`.
    ///
    /// This is the detector a receiver with no schedule has to use, and it can only report bursts
    /// that happened.
    #[must_use]
    pub fn detect(&self, train: &Train, source: u32) -> Vec<Burst> {
        let mut out: Vec<Burst> = Vec::new();
        for s in train.spikes().iter().filter(|s| s.source == source) {
            match out.last_mut() {
                Some(b) if s.t - b.end < self.gap_ticks => {
                    b.end = s.t;
                    b.spikes += 1;
                }
                _ => out.push(Burst { start: s.t, end: s.t, spikes: 1 }),
            }
        }
        out
    }

    /// Decode by segmentation: one value per burst found. **Cannot recover a zero.**
    #[must_use]
    pub fn decode_detected(&self, train: &Train, source: u32) -> Vec<f64> {
        self.detect(train, source).into_iter().map(|b| self.value_of(b.spikes)).collect()
    }

    /// Decode by schedule: count the spikes in each of `count` windows starting at `start_tick`.
    ///
    /// Recovers zeros exactly, because a window with no spikes in it is a value of zero rather than
    /// a burst that did not happen. Requires the receiver to know the schedule, which on a
    /// synchronous neuromorphic fabric it does.
    #[must_use]
    pub fn decode_scheduled(
        &self,
        train: &Train,
        source: u32,
        start_tick: u64,
        count: usize,
    ) -> Vec<f64> {
        let span = u64::from(self.max_spikes - 1) * self.intra_ticks;
        let mut out = vec![0u32; count];
        for s in train.spikes().iter().filter(|s| s.source == source) {
            if s.t < start_tick {
                continue;
            }
            let rel = s.t - start_tick;
            let slot = (rel / self.stride_ticks()) as usize;
            if slot < count && rel - (slot as u64) * self.stride_ticks() <= span {
                out[slot] += 1;
            }
        }
        out.into_iter().map(|k| self.value_of(k)).collect()
    }
}

// ---------------------------------------------------------------------------------------------
// DECONVOLUTION CODING: BSA AND HSA
// ---------------------------------------------------------------------------------------------

/// Root-mean-square difference between two equal-length records.
///
/// `None` for a length mismatch, an empty record or a non-finite sample — the three cases where a
/// reconstruction error is not a number and reporting zero would flatter the encoder.
#[must_use]
pub fn rms_error(a: &[f64], b: &[f64]) -> Option<f64> {
    if a.len() != b.len() || a.is_empty() {
        return None;
    }
    let mut acc = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        if !x.is_finite() || !y.is_finite() {
            return None;
        }
        let d = x - y;
        acc += d * d;
    }
    Some((acc / a.len() as f64).sqrt())
}

/// A Hann (raised-cosine) reconstruction kernel of `taps` strictly positive coefficients, summing
/// to 1.
///
/// `h[k] ∝ 0.5 · (1 − cos(2π (k+1) / (taps+1)))`, so no tap is zero or negative and the
/// reconstruction is a non-negative superposition of identical bumps: a spike can only add. Use
/// this where that monotonicity is wanted, and [`fir_lowpass`] where a sharper frequency response
/// matters more. On this module's reference sinusoid a 24-tap Hann kernel gives [`Bsa`] a
/// root-mean-square error of 0.007 against 0.009 for a 32-tap windowed sinc, so the two are close
/// and the choice is about shape rather than accuracy.
///
/// It does **not** affect [`Hsa`]'s under-reconstruction guarantee, which depends on the signal
/// rather than on the kernel — see [`Hsa`].
///
/// # Errors
///
/// [`CodeError::Empty`] for `taps == 0`.
pub fn fir_hann(taps: usize) -> Result<Vec<f64>, CodeError> {
    if taps == 0 {
        return Err(CodeError::Empty { what: "filter taps" });
    }
    let denom = (taps + 1) as f64;
    let mut h: Vec<f64> = (0..taps)
        .map(|k| 0.5 * (1.0 - (TAU * (k + 1) as f64 / denom).cos()))
        .collect();
    let sum: f64 = h.iter().sum();
    for v in &mut h {
        *v /= sum;
    }
    Ok(h)
}

/// Ben's Spiker Algorithm: choose the spike train whose filtered version best matches the signal
/// (Schrauwen & Van Campenhout, *`BSA`, a fast and accurate spike train encoding scheme*, `IJCNN`
/// 2003).
///
/// At each sample the encoder asks whether subtracting one filter copy from the remaining signal
/// reduces the absolute error. `e1` is the error if it fires, `e2` the error if it does not, and it
/// fires when `e1 ≤ e2 − threshold`.
///
/// **The threshold is not a detail and it is not optional.** With a filter normalised to unit sum,
/// a constant residual makes `e1 − e2` equal to `−Σh = −1` regardless of the residual's level, so
/// at `threshold = 0` the encoder fires almost unconditionally and over-reconstructs: measured on
/// this module's reference sinusoid, a root-mean-square error of 0.242 at `threshold = 0` against
/// 0.007 at 0.955, a factor of 34. Schrauwen & Van Campenhout's published value of about 0.955 is
/// therefore "just below the filter's DC gain" rather than an arbitrary constant, and a caller who
/// renormalises the filter must move the threshold with it. The optimum is signal-dependent and is
/// not a constant this crate will pretend to know.
///
/// Decoding is [`spike_convolve`] and nothing else.
///
/// # What it cannot represent
///
/// The reconstruction is a sum of shifted copies of one non-negative-weighted filter, so `BSA`
/// inherits the filter's limits exactly: energy above the cutoff is lost, the slew rate is capped
/// by the filter's leading edge, and a signal that must go below zero is unreachable — the
/// convention is to offset and scale into `[0, 1]` first, and the crate does not do that silently
/// on the caller's behalf.
#[derive(Debug, Clone, PartialEq)]
pub struct Bsa {
    filter: Vec<f64>,
    /// Firing threshold on the error improvement, in the signal's own units summed over the
    /// filter's length. Larger fires less often and under-reconstructs.
    pub threshold: f64,
}

impl Bsa {
    /// Build and validate.
    ///
    /// # Errors
    ///
    /// [`CodeError::Empty`] for an empty filter; [`CodeError::NotFinite`] for a non-finite tap or
    /// threshold.
    pub fn new(filter: Vec<f64>, threshold: f64) -> Result<Self, CodeError> {
        if filter.is_empty() {
            return Err(CodeError::Empty { what: "filter" });
        }
        check_finite("filter tap", &filter)?;
        check_finite("threshold", &[threshold])?;
        Ok(Self { filter, threshold })
    }

    /// The reconstruction filter.
    #[must_use]
    pub fn filter(&self) -> &[f64] {
        &self.filter
    }

    /// Encode a signal into one spike decision per sample.
    ///
    /// # Errors
    ///
    /// [`CodeError::NotFinite`] for a non-finite sample.
    pub fn encode(&self, signal: &[f64]) -> Result<Vec<bool>, CodeError> {
        check_finite("signal sample", signal)?;
        let n = signal.len();
        let m = self.filter.len();
        let mut s = signal.to_vec();
        let mut out = vec![false; n];
        for t in 0..n {
            let mut e1 = 0.0;
            let mut e2 = 0.0;
            for k in 0..m {
                if t + k >= n {
                    break;
                }
                e1 += (s[t + k] - self.filter[k]).abs();
                e2 += s[t + k].abs();
            }
            if e1 <= e2 - self.threshold {
                out[t] = true;
                for k in 0..m {
                    if t + k >= n {
                        break;
                    }
                    s[t + k] -= self.filter[k];
                }
            }
        }
        Ok(out)
    }

    /// Encode into a [`Train`] on one source, for feeding a network directly.
    ///
    /// # Errors
    ///
    /// As [`Bsa::encode`].
    pub fn encode_train(&self, signal: &[f64], source: u32) -> Result<Train, CodeError> {
        let bits = self.encode(signal)?;
        let spikes = bits
            .iter()
            .enumerate()
            .filter(|&(_, &b)| b)
            .map(|(t, _)| Spike { t: t as u64, source })
            .collect();
        Ok(Train::from_spikes(spikes))
    }

    /// Reconstruct the signal from the spikes. One convolution; see [`spike_convolve`].
    #[must_use]
    pub fn decode(&self, spikes: &[bool]) -> Vec<f64> {
        spike_convolve(spikes, &self.filter)
    }
}

/// The Hough Spiker Algorithm: fire wherever the filter fits underneath the remaining signal.
///
/// Originally due to Hough, de Garis, Korkin, Gers & Nawa (1999); this implementation follows the
/// description and the thresholded generalisation given by Schrauwen & Van Campenhout (`IJCNN`
/// 2003), and did not obtain the 1999 original.
///
/// At each sample, the *deficit* `Σ_k max(0, h[k] − s[t+k])` measures how far the filter pokes above
/// the residual signal. `HSA` fires when the deficit is at most `threshold`, and subtracts.
/// `threshold = 0` is the original strict algorithm: fire only where the filter fits entirely
/// underneath.
///
/// # The invariant, which is exact, and its real precondition
///
/// With `threshold == 0`, firing at `t` requires `residual[t+k] ≥ h[k]` for every `k`, so after the
/// subtraction the residual is non-negative at every sample the filter touched. Nothing else
/// lowers it. The reconstruction is the signal minus the residual, hence
///
/// ```text
/// reconstruction[t] ≤ signal[t]   for every t where signal[t] ≥ 0
/// ```
///
/// `HSA` is a systematic under-estimate by construction: nothing it reconstructs was never there.
///
/// The precondition is on the **signal**, not on the filter. This module's first draft claimed the
/// filter had to be non-negative; a sweep over 20 filters — including windowed-sinc kernels with
/// ten negative taps — and dozens of signals found **zero** violations, because a negative tap
/// makes the firing test easier without ever driving the residual below zero. What does break it
/// is a signal that goes negative: a sinusoid offset to −0.2 is over-reconstructed by 0.70, since
/// an uncovered sample reconstructs as 0 and 0 is above it. [`Hsa::never_over_reconstructs`]
/// reports the part of the precondition this type can see.
#[derive(Debug, Clone, PartialEq)]
pub struct Hsa {
    filter: Vec<f64>,
    /// Tolerated deficit before firing, in the signal's units summed over the filter. Zero is the
    /// original algorithm; a positive value is Schrauwen & Van Campenhout's relaxation, which fires
    /// more often and reconstructs higher at the cost of the under-estimate guarantee.
    pub threshold: f64,
}

impl Hsa {
    /// Build and validate.
    ///
    /// # Errors
    ///
    /// [`CodeError::Empty`] for an empty filter; [`CodeError::NotFinite`] for a non-finite tap or
    /// threshold; [`CodeError::NotPositive`] for a negative threshold, which would mean requiring
    /// the filter to fit with room to spare and has no published meaning.
    pub fn new(filter: Vec<f64>, threshold: f64) -> Result<Self, CodeError> {
        if filter.is_empty() {
            return Err(CodeError::Empty { what: "filter" });
        }
        check_finite("filter tap", &filter)?;
        check_finite("threshold", &[threshold])?;
        if threshold < 0.0 {
            return Err(CodeError::NotPositive { what: "threshold (may be zero, not negative)", value: threshold });
        }
        Ok(Self { filter, threshold })
    }

    /// The reconstruction filter.
    #[must_use]
    pub fn filter(&self) -> &[f64] {
        &self.filter
    }

    /// Whether this instance carries the never-over-reconstruct guarantee, **given a signal that
    /// is non-negative everywhere**: true exactly when the threshold is zero.
    ///
    /// The other half of the precondition is a property of the signal, which this type has not
    /// seen, so this method reports half an answer and says which half. A positive threshold is
    /// Schrauwen & Van Campenhout's relaxation and gives the guarantee up in exchange for firing
    /// where the filter nearly fits.
    #[must_use]
    pub fn never_over_reconstructs(&self) -> bool {
        self.threshold == 0.0
    }

    /// Encode a signal into one spike decision per sample.
    ///
    /// # Errors
    ///
    /// [`CodeError::NotFinite`] for a non-finite sample.
    pub fn encode(&self, signal: &[f64]) -> Result<Vec<bool>, CodeError> {
        check_finite("signal sample", signal)?;
        let n = signal.len();
        let m = self.filter.len();
        let mut s = signal.to_vec();
        let mut out = vec![false; n];
        for t in 0..n {
            let mut deficit = 0.0;
            for k in 0..m {
                if t + k >= n {
                    break;
                }
                deficit += (self.filter[k] - s[t + k]).max(0.0);
            }
            if deficit <= self.threshold {
                out[t] = true;
                for k in 0..m {
                    if t + k >= n {
                        break;
                    }
                    s[t + k] -= self.filter[k];
                }
            }
        }
        Ok(out)
    }

    /// Encode into a [`Train`] on one source.
    ///
    /// # Errors
    ///
    /// As [`Hsa::encode`].
    pub fn encode_train(&self, signal: &[f64], source: u32) -> Result<Train, CodeError> {
        let bits = self.encode(signal)?;
        let spikes = bits
            .iter()
            .enumerate()
            .filter(|&(_, &b)| b)
            .map(|(t, _)| Spike { t: t as u64, source })
            .collect();
        Ok(Train::from_spikes(spikes))
    }

    /// Reconstruct the signal from the spikes.
    #[must_use]
    pub fn decode(&self, spikes: &[bool]) -> Vec<f64> {
        spike_convolve(spikes, &self.filter)
    }
}

// ---------------------------------------------------------------------------------------------
// TEMPORAL CONTRAST
// ---------------------------------------------------------------------------------------------

/// Which temporal-contrast rule to apply. The three variants named by Petro, Kasabov &
/// Whittington (*IEEE Transactions on Neural Networks and Learning Systems*, 2020).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContrastMode {
    /// Step-forward (SF): a baseline that advances by exactly one threshold per event.
    ///
    /// This is what [`crate::encode::DeltaEncoder`] implements, and with that encoder's
    /// `max_events_per_sample` set to 1 the two produce identical events. Its reconstruction error
    /// is bounded by the threshold whenever the signal moves by no more than one threshold per
    /// sample — the only bounded-error member of this family.
    StepForward,
    /// Moving window (MW): the baseline is the mean of the last `window` samples.
    ///
    /// Tracks a drifting baseline without emitting events for the drift itself, at the cost of a
    /// reconstruction whose error is **not** bounded by the threshold; the decoder must rebuild
    /// the same moving average from its own output, so its errors feed back.
    MovingWindow {
        /// Samples averaged for the baseline. At least 1.
        window: usize,
    },
    /// Threshold-based representation (TBR): fire on the first difference of the signal.
    ///
    /// The cheapest of the three and the only one with no baseline at all, which is also its
    /// defect: the reconstruction integrates a quantised derivative, so the residual of every
    /// sample accumulates and the reconstruction **drifts without bound**. Measured, not assumed,
    /// in `threshold_based_reconstruction_drifts_and_step_forward_does_not`.
    ThresholdBased,
}

/// Threshold-based (temporal contrast) encoding of a scalar signal, and its decoder.
///
/// Events carry the sample index in [`Event::t`] and the caller's channel in [`Event::address`],
/// so a multi-channel front end calls this once per channel and merges.
///
/// See the module header for the relationship to [`crate::encode::DeltaEncoder`]: step-forward mode
/// **is** that encoder, capped at one event per sample, and the equivalence is asserted in a test
/// rather than claimed here.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TemporalContrast {
    /// Change required to emit one event, in the signal's own units. Strictly positive.
    pub threshold: f64,
    /// Which of the three published rules to apply.
    pub mode: ContrastMode,
}

impl TemporalContrast {
    /// Build and validate.
    ///
    /// # Errors
    ///
    /// [`CodeError::NotPositive`] for a non-positive or non-finite threshold, or for a
    /// [`ContrastMode::MovingWindow`] of length zero.
    pub fn new(threshold: f64, mode: ContrastMode) -> Result<Self, CodeError> {
        if !threshold.is_finite() || threshold <= 0.0 {
            return Err(CodeError::NotPositive { what: "threshold", value: threshold });
        }
        if let ContrastMode::MovingWindow { window } = mode
            && window == 0
        {
            return Err(CodeError::NotPositive { what: "window", value: 0.0 });
        }
        Ok(Self { threshold, mode })
    }

    /// The threshold Petro, Kasabov & Whittington derive from the signal itself:
    /// `mean|Δs| + factor · std|Δs|`.
    ///
    /// **This rule is non-causal.** It needs the entire record before it can encode the first
    /// sample, so it cannot be used by a sensor and can only be used offline, on a dataset. That is
    /// stated here because the rule is widely quoted without it, and a pipeline that tunes its
    /// threshold on the signal it is about to encode has already seen the future.
    ///
    /// # Errors
    ///
    /// [`CodeError::Empty`] for a signal shorter than two samples; [`CodeError::NotFinite`] for a
    /// non-finite sample; [`CodeError::NoEvidence`] for a constant signal, whose differences are
    /// all zero and which therefore suggests a threshold of zero — a threshold that would fire on
    /// every sample of any other signal.
    pub fn threshold_by_statistics(signal: &[f64], factor: f64) -> Result<f64, CodeError> {
        if signal.len() < 2 {
            return Err(CodeError::Empty { what: "signal (need at least two samples)" });
        }
        check_finite("signal sample", signal)?;
        check_finite("factor", &[factor])?;
        let d: Vec<f64> = signal.windows(2).map(|w| (w[1] - w[0]).abs()).collect();
        let mean = d.iter().sum::<f64>() / d.len() as f64;
        let var = d.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / d.len() as f64;
        let th = mean + factor * var.sqrt();
        if !(th > 0.0) {
            return Err(CodeError::NoEvidence { what: "a usable threshold (the signal does not change)" });
        }
        Ok(th)
    }

    /// Encode a signal into events on `address`, one event index per sample index.
    ///
    /// Sample 0 never emits: step-forward and moving-window both need a baseline first, and
    /// threshold-based needs a previous sample to difference against. This matches
    /// [`crate::encode::DeltaEncoder`], whose first sample sets the reference and emits nothing.
    ///
    /// # Errors
    ///
    /// [`CodeError::NotFinite`] for a non-finite sample.
    pub fn encode(&self, signal: &[f64], address: u32) -> Result<Vec<Event>, CodeError> {
        check_finite("signal sample", signal)?;
        let mut out = Vec::new();
        if signal.is_empty() {
            return Ok(out);
        }
        match self.mode {
            ContrastMode::StepForward => {
                let mut base = signal[0];
                for t in 1..signal.len() {
                    // `>=`, not `>`, so this is bit-identical to `DeltaEncoder`'s test. The
                    // difference shows up only on a signal that lands exactly on a threshold, which
                    // is exactly what a synthetic test signal does.
                    if signal[t] - base >= self.threshold {
                        base += self.threshold;
                        out.push(Event { t: t as u64, address, polarity: Polarity::On });
                    } else if signal[t] - base <= -self.threshold {
                        base -= self.threshold;
                        out.push(Event { t: t as u64, address, polarity: Polarity::Off });
                    }
                }
            }
            ContrastMode::MovingWindow { window } => {
                for t in 1..signal.len() {
                    let lo = t.saturating_sub(window);
                    let base = signal[lo..t].iter().sum::<f64>() / (t - lo) as f64;
                    if signal[t] - base >= self.threshold {
                        out.push(Event { t: t as u64, address, polarity: Polarity::On });
                    } else if signal[t] - base <= -self.threshold {
                        out.push(Event { t: t as u64, address, polarity: Polarity::Off });
                    }
                }
            }
            ContrastMode::ThresholdBased => {
                for t in 1..signal.len() {
                    let d = signal[t] - signal[t - 1];
                    if d >= self.threshold {
                        out.push(Event { t: t as u64, address, polarity: Polarity::On });
                    } else if d <= -self.threshold {
                        out.push(Event { t: t as u64, address, polarity: Polarity::Off });
                    }
                }
            }
        }
        Ok(out)
    }

    /// Reconstruct `samples` samples from `events` on `address`, starting from `start`.
    ///
    /// Step-forward and threshold-based share one accumulator: the reconstruction moves by one
    /// threshold per event and holds otherwise. Moving-window rebuilds its own baseline from its
    /// own output, which is the only way a receiver without the original signal can invert it.
    ///
    /// `start` should be the first sample of the original record. It is the one piece of side
    /// information every threshold code needs and none of them transmits — an absolute level. A
    /// caller who does not have it gets the right shape at the wrong offset.
    #[must_use]
    pub fn decode(&self, events: &[Event], address: u32, samples: usize, start: f64) -> Vec<f64> {
        let mut out = vec![start; samples];
        if samples == 0 {
            return out;
        }
        // One signed step per sample index, so several events on one sample sum rather than fight.
        let mut step = vec![0.0f64; samples];
        for e in events.iter().filter(|e| e.address == address) {
            let t = e.t as usize;
            if t < samples {
                step[t] += e.polarity.sign() * self.threshold;
            }
        }
        match self.mode {
            ContrastMode::StepForward | ContrastMode::ThresholdBased => {
                for t in 1..samples {
                    out[t] = out[t - 1] + step[t];
                }
            }
            ContrastMode::MovingWindow { window } => {
                for t in 1..samples {
                    let lo = t.saturating_sub(window);
                    let base = out[lo..t].iter().sum::<f64>() / (t - lo) as f64;
                    out[t] = base + step[t];
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Bsa, BurstCode, CodeError, ContrastMode, CosinePopulation, GaussianPopulation, Hsa,
        Oscillator, PhaseEncoder, RankOrderCode, TemporalContrast, angular_difference, fir_hann,
        fir_lowpass, poisson_count, rates_from_counts, rms_error, sinc, spike_convolve, wrap_angle,
    };
    use crate::encode::DeltaEncoder;
    use crate::rng::Rng;
    use crate::spike::Polarity;
    use core::f64::consts::{PI, TAU};

    /// `off + amp·sin(2π f t)` at `n` integer sample times. `f` is in cycles per SAMPLE, which is
    /// the unit every filter cutoff in this module is also in — mixing the two is the classic way
    /// to conclude a reconstruction works when it was never asked to do anything.
    fn sine(n: usize, f: f64, amp: f64, off: f64) -> Vec<f64> {
        (0..n).map(|t| off + amp * (TAU * f * t as f64).sin()).collect()
    }

    // ---------------------------------------------------------------------------------------
    // SHARED NUMERICS
    // ---------------------------------------------------------------------------------------

    /// `sinc` against its closed form at the points where the closed form is exact.
    #[test]
    fn sinc_is_one_at_zero_and_vanishes_at_every_other_integer() {
        assert!((sinc(0.0) - 1.0).abs() < 1e-15);
        for k in 1..=40 {
            assert!(sinc(f64::from(k)).abs() < 1e-15, "sinc({k}) = {}", sinc(f64::from(k)));
            assert!(sinc(f64::from(-k)).abs() < 1e-15);
        }
        // sinc(1/2) = sin(π/2)/(π/2) = 2/π, exactly.
        assert!((sinc(0.5) - 2.0 / PI).abs() < 1e-15, "sinc(0.5) = {}", sinc(0.5));
        // Even, exactly, because sin is odd and so is the denominator.
        for &x in &[0.3, 1.7, 4.25] {
            assert!((sinc(x) - sinc(-x)).abs() < 1e-16);
        }
    }

    /// Both kernels must have a DC gain of exactly one, because that identity is what makes a
    /// spike density equal to a signal level.
    #[test]
    fn both_reconstruction_kernels_have_unit_dc_gain() {
        for taps in [1usize, 2, 7, 16, 24, 33, 64] {
            let h = fir_hann(taps).expect("taps > 0");
            let s: f64 = h.iter().sum();
            assert!((s - 1.0).abs() < 1e-14, "hann {taps} sums to {s}");
            assert!(h.iter().all(|&v| v > 0.0), "hann {taps} has a non-positive tap");
            if taps >= 4 {
                let g = fir_lowpass(taps, 0.1).expect("valid cutoff");
                let s: f64 = g.iter().sum();
                assert!((s - 1.0).abs() < 1e-14, "sinc {taps} sums to {s}");
            }
        }
        // A windowed sinc really does have negative taps; the HSA invariant test depends on it.
        let g = fir_lowpass(24, 0.1).expect("valid cutoff");
        assert!(g.iter().filter(|&&v| v < 0.0).count() >= 4, "no negative side lobes to test with");
    }

    #[test]
    fn filter_design_refuses_a_cutoff_it_cannot_honour() {
        assert!(matches!(fir_lowpass(0, 0.1), Err(CodeError::Empty { .. })));
        assert!(matches!(fir_lowpass(16, 0.0), Err(CodeError::NotPositive { .. })));
        assert!(matches!(fir_lowpass(16, 0.5), Err(CodeError::NotPositive { .. })));
        assert!(matches!(fir_lowpass(16, f64::NAN), Err(CodeError::NotPositive { .. })));
        assert!(matches!(fir_hann(0), Err(CodeError::Empty { .. })));
    }

    /// The Poisson sampler against the three closed forms a Poisson variable has: mean `λ`,
    /// variance `λ`, and `P(k = 0) = exp(−λ)`. Tolerances are set from the sampling error of the
    /// estimator being formed, not chosen to pass.
    #[test]
    fn poisson_draws_match_mean_variance_and_the_zero_probability() {
        let n = 200_000usize;
        for &lam in &[0.5f64, 5.0] {
            let mut rng = Rng::new(3);
            let (mut s, mut q, mut zeros) = (0.0f64, 0.0f64, 0u64);
            for _ in 0..n {
                let k = poisson_count(&mut rng, lam).expect("finite non-negative mean");
                s += k as f64;
                q += (k * k) as f64;
                zeros += u64::from(k == 0);
            }
            let mean = s / n as f64;
            let var = q / n as f64 - mean * mean;
            // The sample mean of n Poisson draws has standard deviation sqrt(lam/n); six of those.
            let tol_m = 6.0 * (lam / n as f64).sqrt();
            assert!((mean - lam).abs() < tol_m, "lam {lam}: mean {mean}, tolerance {tol_m}");
            // Var of the sample variance for a Poisson is (lam + 2 lam^2)/n; six of those.
            let tol_v = 6.0 * ((lam + 2.0 * lam * lam) / n as f64).sqrt();
            assert!((var - lam).abs() < tol_v, "lam {lam}: var {var}, tolerance {tol_v}");
            let p0 = (-lam).exp();
            let got0 = zeros as f64 / n as f64;
            let tol_0 = 6.0 * (p0 * (1.0 - p0) / n as f64).sqrt();
            assert!((got0 - p0).abs() < tol_0, "lam {lam}: P(0) {got0} vs exp(-lam) {p0}");
        }
        assert_eq!(poisson_count(&mut Rng::new(1), 0.0), Some(0), "zero rate is zero spikes");
        assert!(poisson_count(&mut Rng::new(1), -1.0).is_none());
        assert!(poisson_count(&mut Rng::new(1), f64::NAN).is_none());
    }

    /// The defect the chunked sampler exists to avoid: at `lambda > 745`, `exp(-lambda)` is exactly
    /// zero in `f64` and Knuth's loop never terminates. This test would HANG, not fail, on the
    /// textbook one-liner — which is why it is here with a mean well past the underflow point.
    #[test]
    fn a_mean_that_underflows_the_textbook_algorithm_still_terminates_and_is_unbiased() {
        assert_eq!((-800.0f64).exp(), 0.0, "the test is not exercising the defect");
        let lam = 800.0;
        let n = 20_000usize;
        let mut rng = Rng::new(5);
        let mut s = 0.0;
        for _ in 0..n {
            s += poisson_count(&mut rng, lam).expect("finite mean") as f64;
        }
        let mean = s / n as f64;
        let tol = 6.0 * (lam / n as f64).sqrt();
        assert!((mean - lam).abs() < tol, "mean {mean} vs {lam}, tolerance {tol}");
    }

    #[test]
    fn angular_difference_takes_the_short_way_round() {
        assert!((angular_difference(0.01, TAU - 0.003) - 0.013).abs() < 1e-12);
        assert!((angular_difference(TAU - 0.003, 0.01) + 0.013).abs() < 1e-12);
        assert!(angular_difference(PI, 0.0) <= PI && angular_difference(PI, 0.0) > PI - 1e-12);
        assert!((wrap_angle(-0.1) - (TAU - 0.1)).abs() < 1e-12, "wrap must not return a negative");
    }

    // ---------------------------------------------------------------------------------------
    // POPULATION CODING
    // ---------------------------------------------------------------------------------------

    /// The closed form: for evenly tiled cosine tuning with `n >= 3` the population vector points
    /// exactly at the stimulus, at every `n`. Not approximately, and not better with more cells.
    #[test]
    fn the_population_vector_recovers_the_angle_exactly_at_every_n_from_three_up() {
        for &n in &[3usize, 4, 5, 8, 12, 37, 100] {
            let p = CosinePopulation::new(n, 50.0, 30.0, 0.31).expect("valid population");
            let mut worst: f64 = 0.0;
            for s in 0..720 {
                let th = TAU * f64::from(s) / 720.0;
                let r = p.rates(th).expect("finite angle");
                let est = p.decode_population_vector(&r).expect("a driven population");
                worst = worst.max(angular_difference(est, th).abs());
            }
            assert!(worst < 1e-12, "n = {n}: worst angular error {worst} rad");
        }
    }

    /// The baseline cancels exactly. A population with a hundred-fold baseline decodes the same
    /// angle, which is the algebra `Σ_i (cos φ_i, sin φ_i) = 0` made into an assertion.
    #[test]
    fn the_population_vector_is_independent_of_the_tuning_baseline() {
        let lean = CosinePopulation::new(9, 5.0, 5.0, 0.0).expect("valid");
        let fat = CosinePopulation::new(9, 500.0, 5.0, 0.0).expect("valid");
        for s in 0..360 {
            let th = TAU * f64::from(s) / 360.0;
            let a = lean.decode_population_vector(&lean.rates(th).expect("finite")).expect("driven");
            let b = fat.decode_population_vector(&fat.rates(th).expect("finite")).expect("driven");
            assert!(angular_difference(a, b).abs() < 1e-12, "{a} vs {b} at {th}");
        }
    }

    /// Refuse rather than guess. A population firing equally in every direction has no direction,
    /// and `atan2(0, 0)` is 0.0 — due east, and plottable.
    #[test]
    fn a_uniformly_active_population_has_no_direction_and_refuses_to_invent_one() {
        let p = CosinePopulation::new(8, 20.0, 10.0, 0.0).expect("valid");
        let flat = vec![37.0; 8];
        assert!(matches!(p.decode_population_vector(&flat), Err(CodeError::NoEvidence { .. })));
        let silent = vec![0.0; 8];
        assert!(matches!(p.decode_population_vector(&silent), Err(CodeError::NoEvidence { .. })));
        assert!(matches!(
            p.decode_population_vector(&[1.0, 2.0]),
            Err(CodeError::LengthMismatch { expected: 8, got: 2 })
        ));
        let mut bad = vec![1.0; 8];
        bad[3] = f64::NAN;
        assert!(matches!(p.decode_population_vector(&bad), Err(CodeError::NotFinite { .. })));
    }

    /// The exactness is a property of the un-rectified cosine. Real cells cannot fire negatively,
    /// and this is what that costs — measured, so the caveat is a number.
    #[test]
    fn rectifying_a_cosine_population_costs_a_measurable_angular_ripple() {
        let mut worst_by_n = Vec::new();
        for &n in &[8usize, 16] {
            let p = CosinePopulation::new(n, 10.0, 10.0, 0.0).expect("valid");
            let mut worst: f64 = 0.0;
            for s in 0..360 {
                let th = TAU * f64::from(s) / 360.0;
                // b1 > b0 would be refused by the constructor, so the rectified profile is built
                // here directly: this is the population the animal has, not the one the algebra
                // likes.
                let r: Vec<f64> = (0..n)
                    .map(|i| (10.0f64 + 30.0 * (th - p.preferred(i)).cos()).max(0.0))
                    .collect();
                let est = p.decode_population_vector(&r).expect("driven");
                worst = worst.max(angular_difference(est, th).abs());
            }
            worst_by_n.push(worst);
        }
        // Measured: 0.0160 rad at n = 8 and 0.0062 rad at n = 16 — about 0.9 and 0.36 degrees.
        // Small, real, and absent from the un-rectified case, where the error is 1e-15.
        assert!(worst_by_n[0] > 5e-3, "rectification ripple vanished: {worst_by_n:?}");
        assert!(worst_by_n[0] < 5e-2, "ripple larger than expected: {worst_by_n:?}");
        assert!(worst_by_n[1] < worst_by_n[0], "more cells did not reduce the ripple: {worst_by_n:?}");
    }

    /// Requirement (c): fit the exponent, do not assert the direction. Independent Poisson noise
    /// averages as `1/sqrt(n)`, so the log-log slope of angular error against population size must
    /// be −0.5.
    #[test]
    fn the_population_vector_error_falls_as_the_inverse_square_root_of_population_size() {
        let mut rng = Rng::new(11);
        let window = 0.5;
        let mut pts = Vec::new();
        for &n in &[8usize, 16, 32, 64, 128, 256] {
            let p = CosinePopulation::new(n, 20.0, 20.0, 0.0).expect("valid");
            let trials = 400usize;
            let mut acc = 0.0;
            for t in 0..trials {
                // Golden-ratio stepping covers the circle without ever repeating an angle, so the
                // error is averaged over phase rather than measured at one lucky direction.
                let th = wrap_angle(TAU * (t as f64 * 0.618_033_988_749_894_9));
                let c = p.sample_counts(&mut rng, th, window).expect("valid");
                let r = rates_from_counts(&c, window).expect("positive window");
                let est = p.decode_population_vector(&r).expect("driven");
                let d = angular_difference(est, th);
                acc += d * d;
            }
            let rms = (acc / trials as f64).sqrt();
            assert!(rms > 0.0, "n = {n} produced a zero error, which means no noise was applied");
            pts.push(((n as f64).ln(), rms.ln()));
        }
        let mx = pts.iter().map(|p| p.0).sum::<f64>() / pts.len() as f64;
        let my = pts.iter().map(|p| p.1).sum::<f64>() / pts.len() as f64;
        let num: f64 = pts.iter().map(|p| (p.0 - mx) * (p.1 - my)).sum();
        let den: f64 = pts.iter().map(|p| (p.0 - mx) * (p.0 - mx)).sum();
        let slope = num / den;
        // Measured −0.492 with this seed. A slope of −0.25 or −1.0 would both be "improves with n".
        assert!((slope + 0.5).abs() < 0.06, "fitted exponent {slope}, expected -0.5");
    }

    /// Requirement (a) for the Gaussian population: a sweep, not one value, with the edge bias
    /// pinned as a real effect rather than tolerated.
    #[test]
    fn the_gaussian_population_round_trips_across_its_interior_and_is_biased_at_its_edges() {
        let n = 24;
        let p = GaussianPopulation::new(n, 0.0, 1.0, 2.0 / (n - 1) as f64, 100.0, 2.0)
            .expect("valid population");
        let (mut worst_interior, mut worst_edge): (f64, f64) = (0.0, 0.0);
        for s in 0..=200 {
            let x = f64::from(s) / 200.0;
            let r = p.rates(x).expect("finite");
            let est = p.decode_center_of_mass(&r).expect("driven");
            let e = (est - x).abs();
            if (0.25..=0.75).contains(&x) {
                worst_interior = worst_interior.max(e);
            } else {
                worst_edge = worst_edge.max(e);
            }
        }
        // Measured: 2.40e-4 over the middle half, 5.66e-2 at the ends. Both are asserted, because
        // a change that quietly widened the tuning curves would improve one and ruin the other.
        assert!(worst_interior < 3e-4, "interior error {worst_interior}");
        assert!(worst_edge > 1e-2, "the edge bias disappeared ({worst_edge}); it is real and large");
    }

    /// The reported interior must be the range where the decode actually holds, on both sides.
    #[test]
    fn the_reported_interior_is_exactly_where_the_decode_holds() {
        let n = 24;
        let p = GaussianPopulation::new(n, 0.0, 1.0, 2.0 / (n - 1) as f64, 100.0, 2.0)
            .expect("valid");
        let (a, b) = p.interior(1e-3).expect("a 24-cell population has an interior");
        assert!(a > 0.0 && b < 1.0 && b > a, "interior ({a}, {b}) is not a sub-range");
        for s in 0..=100 {
            let x = a + (b - a) * f64::from(s) / 100.0;
            let est = p.decode_center_of_mass(&p.rates(x).expect("finite")).expect("driven");
            assert!((est - x).abs() <= 1.1e-3, "inside the interior at {x}: error {}", est - x);
        }
        let outside = p
            .decode_center_of_mass(&p.rates(0.0).expect("finite"))
            .expect("driven");
        assert!(outside.abs() > 1e-3, "the range end was not biased, so `interior` is not measuring");
        // A tolerance nothing meets has no interior, and that is the answer, not an empty range.
        assert!(p.interior(1e-18).is_none());
        assert!(GaussianPopulation::new(2, 0.0, 1.0, 0.5, 100.0, 1.0).expect("valid").interior(1e-3).is_none());
    }

    /// The published grading of a decoder: maximum likelihood attains the Cramér-Rao bound and the
    /// population vector does not. Both variances are measured against
    /// `GaussianPopulation::fisher_information`, computed in closed form.
    #[test]
    fn maximum_likelihood_reaches_the_cramer_rao_bound_and_centre_of_mass_does_not() {
        let n = 24;
        let p = GaussianPopulation::new(n, 0.0, 1.0, 2.0 / (n - 1) as f64, 100.0, 2.0)
            .expect("valid");
        let window = 0.2;
        let x0 = 0.5;
        let crb = 1.0 / p.fisher_information(x0, window).expect("informative");
        let trials = 4_000usize;
        let mut rng = Rng::new(7);
        let (mut ml_s, mut ml_q, mut com_s, mut com_q) = (0.0, 0.0, 0.0, 0.0);
        for _ in 0..trials {
            let c = p.sample_counts(&mut rng, x0, window).expect("valid");
            let ml = p.decode_max_likelihood(&c, window, 401).expect("consistent counts");
            ml_s += ml;
            ml_q += ml * ml;
            let r = rates_from_counts(&c, window).expect("positive window");
            let com = p.decode_center_of_mass(&r).expect("driven");
            com_s += com;
            com_q += com * com;
        }
        let ml_m = ml_s / trials as f64;
        let ml_v = ml_q / trials as f64 - ml_m * ml_m;
        let com_m = com_s / trials as f64;
        let com_v = com_q / trials as f64 - com_m * com_m;
        // Measured: ML variance is 1.008 x CRB, centre of mass 1.332 x. The bound is a LOWER bound,
        // so a ratio below 1 by more than sampling error would mean the estimator is biased, not
        // that it is good; both sides are checked.
        assert!(ml_v / crb > 0.85, "ML variance {ml_v} is BELOW the Cramer-Rao bound {crb}");
        assert!(ml_v / crb < 1.25, "ML variance {ml_v} is {} x the bound", ml_v / crb);
        assert!(com_v / crb > 1.15, "centre of mass tied the bound at {} x", com_v / crb);
        assert!((ml_m - x0).abs() < 3.0 * (ml_v / trials as f64).sqrt() + 1e-3, "ML bias {}", ml_m - x0);
        assert!(com_v > ml_v, "centre of mass {com_v} was not worse than ML {ml_v}");
    }

    /// A population that says nothing decodes to nothing.
    #[test]
    fn a_silent_gaussian_population_has_no_decoded_value() {
        let p = GaussianPopulation::new(10, 0.0, 1.0, 0.2, 50.0, 3.0).expect("valid");
        let at_baseline = vec![3.0; 10];
        assert!(matches!(p.decode_center_of_mass(&at_baseline), Err(CodeError::NoEvidence { .. })));
        assert!(matches!(p.decode_center_of_mass(&[0.0; 10]), Err(CodeError::NoEvidence { .. })));
        assert!(matches!(p.rates(f64::INFINITY), Err(CodeError::NotFinite { .. })));
        assert!(matches!(
            GaussianPopulation::new(0, 0.0, 1.0, 0.2, 50.0, 1.0),
            Err(CodeError::Empty { .. })
        ));
        assert!(matches!(
            GaussianPopulation::new(4, 1.0, 1.0, 0.2, 50.0, 1.0),
            Err(CodeError::EmptyRange { .. })
        ));
        assert!(matches!(
            GaussianPopulation::new(4, 0.0, 1.0, -0.2, 50.0, 1.0),
            Err(CodeError::NotPositive { .. })
        ));
    }

    /// The population code has to survive contact with the crate's own spike representation, not
    /// just with a vector of rates.
    #[test]
    fn a_population_round_trips_through_a_real_spike_train() {
        let n = 24;
        let p = GaussianPopulation::new(n, 0.0, 1.0, 2.0 / (n - 1) as f64, 200.0, 2.0)
            .expect("valid");
        let dt = 1e-3;
        let ticks = 400u64; // 0.4 s
        let mut rng = Rng::new(31);
        let mut worst: f64 = 0.0;
        for s in 0..=20 {
            let x = 0.3 + 0.4 * f64::from(s) / 20.0;
            let train = p.encode_train(&mut rng, x, ticks, dt).expect("valid");
            let mut counts = vec![0u64; n];
            for sp in train.spikes() {
                counts[sp.source as usize] += 1;
            }
            let r = rates_from_counts(&counts, ticks as f64 * dt).expect("positive window");
            let est = p.decode_center_of_mass(&r).expect("driven");
            worst = worst.max((est - x).abs());
        }
        // 0.4 s at a 200 Hz peak is about 80 spikes in the best-driven cell; the Cramer-Rao
        // standard deviation there is near 0.008, so a worst case of 0.05 over 21 stimuli is a
        // loose but honest bound on a stochastic quantity. It is NOT a tight one, and a tighter
        // one would be a flaky test rather than a better one.
        assert!(worst < 0.05, "worst round-trip error through a spike train: {worst}");
    }

    // ---------------------------------------------------------------------------------------
    // RANK-ORDER CODING
    // ---------------------------------------------------------------------------------------

    /// Requirement (b), and the property no other code in this crate has: scale every input by any
    /// positive constant and the decoded order is bit-identical.
    #[test]
    fn the_order_is_bit_identical_under_any_positive_rescaling() {
        let code = RankOrderCode::new(8, 0.9).expect("valid");
        let mut rng = Rng::new(21);
        for _ in 0..4_000 {
            let x: Vec<f64> = (0..8).map(|_| rng.next_f64() + 1e-3).collect();
            let base = code.encode(&x).expect("finite drives");
            for &c in &[1e-9f64, 1e-3, 0.25, 3.7, 64.0, 1e6, 1e9] {
                let y: Vec<f64> = x.iter().map(|v| v * c).collect();
                let got = code.encode(&y).expect("finite drives");
                assert_eq!(got, base, "scaling by {c} changed the order");
            }
        }
        // And the contrast that a rate code cannot survive: the same comparison for `RateEncoder`
        // would change every count by the factor, which is the point of the whole scheme.
        let dim = code.encode(&[1e-12, 2e-12, 3e-12, 4e-12, 5e-12, 6e-12, 7e-12, 8e-12]).expect("finite");
        let bright = code.encode(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).expect("finite");
        assert_eq!(dim, bright, "twelve orders of magnitude of contrast changed the code");
        assert_eq!(bright.order, vec![7, 6, 5, 4, 3, 2, 1, 0]);
    }

    /// The honest half: the order is exact, the magnitude is an assumption. Both are measured.
    #[test]
    fn rank_order_recovers_order_exactly_and_magnitude_only_under_its_own_law() {
        let m = 0.75;
        let code = RankOrderCode::new(6, m).expect("valid");
        // A geometric input IS the law the readout assumes, so the amplitudes come back exactly,
        // up to the overall scale the code cannot carry.
        let x: Vec<f64> = (0..6).map(|k| 4.2 * m.powi(k)).collect();
        let ro = code.encode(&x).expect("finite");
        let back = code.decode_amplitudes(&ro);
        for k in 0..6 {
            assert!((back[k] - x[k] / x[0]).abs() < 1e-12, "k {k}: {} vs {}", back[k], x[k] / x[0]);
        }
        // A linear input has the same ORDER and a completely different shape. The decoded value of
        // the last cell is 0.178 where the truth is 0.167 of the first — close here only because
        // six cells is short; the point is that the reconstruction is the assumption, not the data.
        let y: Vec<f64> = (0..6).map(|k| 6.0 - k as f64).collect();
        let ro_y = code.encode(&y).expect("finite");
        assert_eq!(ro_y.order, ro.order, "the order must be identical; only the magnitudes differ");
        let back_y = code.decode_amplitudes(&ro_y);
        let truth: Vec<f64> = y.iter().map(|v| v / y[0]).collect();
        let err = rms_error(&back_y, &truth).expect("same length");
        assert!(err > 0.05, "a linear input decoded too well ({err}); the test is not exercising the gap");
        // ...while the order it implies is still perfect.
        for k in 0..6 {
            assert_eq!(ro_y.rank[k], Some(k as u32));
        }
    }

    /// `log2(n!)` against a directly computed factorial, which `f64` holds exactly to `n = 18`.
    #[test]
    fn the_capacity_of_a_rank_code_is_log_two_of_n_factorial() {
        for n in 1..=18usize {
            let code = RankOrderCode::new(n, 0.9).expect("valid");
            let mut fact = 1.0f64;
            for k in 1..=n {
                fact *= k as f64;
            }
            assert!(
                (code.capacity_bits() - fact.log2()).abs() < 1e-9,
                "n {n}: {} vs log2({fact})",
                code.capacity_bits()
            );
        }
        // Eight cells, each firing once: 15.3 bits and not one bit of amplitude.
        assert!((RankOrderCode::new(8, 0.5).expect("valid").capacity_bits() - 15.299_208).abs() < 1e-5);
    }

    #[test]
    fn a_cell_with_no_drive_has_no_rank_and_a_dark_retina_has_no_code() {
        let code = RankOrderCode::new(5, 0.8).expect("valid");
        let ro = code.encode(&[0.5, 0.0, 0.9, -1.0, 0.1]).expect("finite");
        assert_eq!(ro.order, vec![2, 0, 4]);
        assert_eq!(ro.rank, vec![Some(1), None, Some(0), None, Some(2)]);
        assert_eq!(ro.fired(), 3);
        let dark = code.encode(&[0.0; 5]).expect("finite");
        assert!(dark.is_silent());
        assert!(code.decode_amplitudes(&dark).iter().all(|&v| v == 0.0));
        assert!(matches!(code.encode(&[1.0, 2.0]), Err(CodeError::LengthMismatch { .. })));
        assert!(matches!(code.encode(&[1.0, f64::NAN, 3.0, 4.0, 5.0]), Err(CodeError::NotFinite { .. })));
        assert!(matches!(RankOrderCode::new(4, 1.5), Err(CodeError::NotPositive { .. })));
        assert!(matches!(RankOrderCode::new(0, 0.5), Err(CodeError::Empty { .. })));
    }

    /// The order must survive being turned into spikes and read back out of a `Train`, including
    /// the tie-break, which is where an implementation that leaned on the sort's internals breaks.
    #[test]
    fn the_order_round_trips_through_a_real_spike_train() {
        let code = RankOrderCode::new(12, 0.85).expect("valid");
        let mut rng = Rng::new(404);
        for _ in 0..500 {
            let x: Vec<f64> = (0..12).map(|_| rng.next_f64()).collect();
            let ro = code.encode(&x).expect("finite");
            let train = RankOrderCode::to_train(&ro, 7, 3);
            let back = code.from_train(&train);
            assert_eq!(back, ro, "the order did not survive the train");
        }
        // Ties on one tick break by source index, in the encoder and in the reader alike.
        let flat = code.encode(&[1.0; 12]).expect("finite");
        assert_eq!(flat.order, (0..12u32).collect::<Vec<_>>());
        let squashed = RankOrderCode::to_train(&flat, 0, 0);
        assert_eq!(code.from_train(&squashed).order, flat.order);
    }

    /// Thorpe's readout is maximised by the order it was built for. That is the rearrangement
    /// inequality, checked against every one of the 120 permutations of five cells rather than
    /// against a couple of hand-picked ones.
    #[test]
    fn the_rank_order_readout_is_maximal_for_the_order_it_encodes() {
        let m = 0.6;
        let code = RankOrderCode::new(5, m).expect("valid");
        // Weights tuned for the order 0,1,2,3,4.
        let w: Vec<f64> = (0..5).map(|k| m.powi(k)).collect();
        let target: Vec<f64> = (0..5).map(|k| 5.0 - k as f64).collect();
        let best = code.readout(&code.encode(&target).expect("finite"), &w).expect("valid");
        let mut perm = [0usize, 1, 2, 3, 4];
        let mut checked = 0;
        // Heap's algorithm, iterative, so no dependency and no recursion.
        let mut c = [0usize; 5];
        let mut i = 0;
        loop {
            let mut drive = [0.0f64; 5];
            for (slot, &cell) in perm.iter().enumerate() {
                drive[cell] = 5.0 - slot as f64;
            }
            let got = code.readout(&code.encode(&drive).expect("finite"), &w).expect("valid");
            assert!(got <= best + 1e-12, "permutation {perm:?} scored {got} above the target {best}");
            checked += 1;
            while i < 5 && c[i] >= i {
                c[i] = 0;
                i += 1;
            }
            if i >= 5 {
                break;
            }
            if i % 2 == 0 {
                perm.swap(0, i);
            } else {
                perm.swap(c[i], i);
            }
            c[i] += 1;
            i = 0;
        }
        assert_eq!(checked, 120, "Heap's algorithm did not enumerate every permutation");
    }

    // ---------------------------------------------------------------------------------------
    // PHASE CODING
    // ---------------------------------------------------------------------------------------

    /// Requirement (a) for phase: a sweep, and the bound is the tick quantisation, which the code
    /// meets exactly rather than approximately.
    #[test]
    fn phase_coding_round_trips_to_exactly_its_tick_quantisation() {
        let osc = Oscillator::new(8.0, 1.0, 0.0).expect("valid");
        let pe = PhaseEncoder::new(osc, 1e-4).expect("valid");
        assert!((pe.ticks_per_cycle() - 1250.0).abs() < 1e-9);
        let q = pe.quantisation();
        assert!((q - 4e-4).abs() < 1e-12, "quantisation {q}");
        let mut worst: f64 = 0.0;
        for s in 0..=2_000 {
            let x = f64::from(s) / 2_000.0;
            let t = pe.encode_tick(x, 3).expect("x in range");
            let back = pe.decode_tick(t);
            // Phase is circular, so the error is a circular distance. Measuring it linearly makes
            // x = 0.9999 decoding to 0.0001 look like an error of 0.9998 instead of 0.0002.
            let d = (back - x).abs();
            worst = worst.max(d.min(1.0 - d));
        }
        assert!(worst <= q + 1e-12, "worst circular round-trip error {worst} vs quantisation {q}");
        // A non-zero reference phase must not change the bound, only the tick.
        let shifted = PhaseEncoder::new(Oscillator::new(8.0, 1.0, 2.1).expect("valid"), 1e-4)
            .expect("valid");
        for s in 0..=500 {
            let x = f64::from(s) / 500.0;
            let back = shifted.decode_tick(shifted.encode_tick(x, 5).expect("in range"));
            let d = (back - x).abs();
            assert!(d.min(1.0 - d) <= q + 1e-12, "x {x} -> {back}");
        }
    }

    /// Requirement (e), first half: the amplitude is not in the code at all. Not nearly — the ticks
    /// are bit-identical, because [`PhaseEncoder::encode_tick`] never reads it.
    #[test]
    fn phase_coding_is_exactly_invariant_to_the_oscillation_amplitude() {
        let quiet = PhaseEncoder::new(Oscillator::new(40.0, 1e-6, 0.7).expect("valid"), 2.5e-5)
            .expect("valid");
        let loud = PhaseEncoder::new(Oscillator::new(40.0, 1e6, 0.7).expect("valid"), 2.5e-5)
            .expect("valid");
        for s in 0..=1_000 {
            let x = f64::from(s) / 1_000.0;
            for cycle in [0u64, 1, 17] {
                assert_eq!(
                    quiet.encode_tick(x, cycle).expect("in range"),
                    loud.encode_tick(x, cycle).expect("in range"),
                    "amplitude changed the tick for x {x}"
                );
            }
            assert!((quiet.decode_tick(s as u64) - loud.decode_tick(s as u64)).abs() < 1e-15);
        }
        // And the amplitude really is present in the reference waveform, so the invariance is a
        // property of the CODE rather than of an amplitude that never mattered.
        assert!((loud.osc.value_at(0.001) / quiet.osc.value_at(0.001) - 1e12).abs() < 1e3);
    }

    /// Requirement (e), second half: it is exactly as wrong as its reference. Shift the
    /// oscillation's phase by δ and every decoded value shifts by δ / 2π, as an equality.
    #[test]
    fn a_shift_in_the_reference_phase_shifts_every_decoded_value_by_exactly_that_shift() {
        let base = PhaseEncoder::new(Oscillator::new(8.0, 1.0, 0.0).expect("valid"), 1e-4)
            .expect("valid");
        for &delta in &[0.001f64, 0.3, 1.234, PI, 5.9] {
            let moved = PhaseEncoder::new(Oscillator::new(8.0, 1e6, delta).expect("valid"), 1e-4)
                .expect("valid");
            let want = (delta / TAU).rem_euclid(1.0);
            let mut worst: f64 = 0.0;
            for t in 0..5_000u64 {
                let got = (moved.decode_tick(t) - base.decode_tick(t)).rem_euclid(1.0);
                let d = (got - want).abs();
                worst = worst.max(d.min(1.0 - d));
            }
            assert!(worst < 1e-12, "delta {delta}: worst deviation from delta/2pi is {worst}");
        }
    }

    #[test]
    fn a_phase_encoder_refuses_a_tick_too_coarse_to_resolve_a_cycle() {
        let osc = Oscillator::new(100.0, 1.0, 0.0).expect("valid");
        // 100 Hz with a 10 ms tick is one tick per cycle: aliased, not merely coarse.
        assert!(matches!(PhaseEncoder::new(osc, 1e-2), Err(CodeError::NotPositive { .. })));
        assert!(matches!(PhaseEncoder::new(osc, 0.0), Err(CodeError::NotPositive { .. })));
        assert!(PhaseEncoder::new(osc, 5e-3).is_ok(), "exactly two ticks per cycle is the boundary");
        assert!(matches!(Oscillator::new(0.0, 1.0, 0.0), Err(CodeError::NotPositive { .. })));
        assert!(matches!(Oscillator::new(f64::NAN, 1.0, 0.0), Err(CodeError::NotFinite { .. })));
        let pe = PhaseEncoder::new(osc, 1e-3).expect("valid");
        assert!(matches!(pe.encode_tick(1.5, 0), Err(CodeError::OutOfUnitRange { .. })));
        assert!(matches!(pe.encode_tick(f64::NAN, 0), Err(CodeError::NotFinite { .. })));
        assert!(matches!(pe.encode_tick(0.5, u64::MAX), Err(CodeError::NotFinite { .. })));
    }

    #[test]
    fn a_phase_code_round_trips_through_a_train_and_a_silent_line_stays_silent() {
        let pe = PhaseEncoder::new(Oscillator::new(8.0, 1.0, 0.9).expect("valid"), 1e-4)
            .expect("valid");
        let x = [0.0, 0.12, 0.5, 0.77, 0.99];
        let train = pe.encode(&x, 2).expect("all in range");
        assert_eq!(train.len(), 5, "one spike per value");
        let back = pe.decode_train(&train, 7);
        for (i, &want) in x.iter().enumerate() {
            let got = back[i].expect("this line fired");
            let d = (got - want).abs();
            assert!(d.min(1.0 - d) <= pe.quantisation() + 1e-12, "line {i}: {got} vs {want}");
        }
        assert!(back[5].is_none() && back[6].is_none(), "a line that never fired must decode to None");
    }

    // ---------------------------------------------------------------------------------------
    // BURST CODING
    // ---------------------------------------------------------------------------------------

    /// Requirement (a) for bursts: a full sweep, and the bound is the spike quantisation, which is
    /// the only error the code has.
    #[test]
    fn burst_coding_round_trips_to_its_spike_quantisation_across_the_range() {
        let bc = BurstCode::new(20, 2, 10).expect("valid");
        let q = bc.quantisation();
        assert!((q - 0.025).abs() < 1e-15);
        let xs: Vec<f64> = (0..=1_000).map(|s| f64::from(s) / 1_000.0).collect();
        let train = bc.encode_sequence(&xs, 0, 5).expect("all in range");
        let back = bc.decode_scheduled(&train, 0, 5, xs.len());
        assert_eq!(back.len(), xs.len());
        let mut worst: f64 = 0.0;
        for (i, &x) in xs.iter().enumerate() {
            worst = worst.max((back[i] - x).abs());
        }
        assert!(worst <= q + 1e-12, "worst burst round-trip error {worst} vs quantisation {q}");
        assert!((bc.capacity_bits() - 21.0f64.log2()).abs() < 1e-12);
    }

    /// The zero problem, which is why there are two decoders. A segmenting receiver cannot see a
    /// value of zero, because zero is silence.
    #[test]
    fn a_value_of_zero_is_silence_and_only_the_scheduled_decoder_recovers_it() {
        let bc = BurstCode::new(20, 2, 10).expect("valid");
        let xs = [0.5, 0.0, 0.7];
        let train = bc.encode_sequence(&xs, 0, 0).expect("in range");
        let detected = bc.decode_detected(&train, 0);
        assert_eq!(detected.len(), 2, "the segmenter saw {} bursts", detected.len());
        assert!((detected[0] - 0.5).abs() < 1e-12 && (detected[1] - 0.7).abs() < 1e-12);
        let scheduled = bc.decode_scheduled(&train, 0, 0, 3);
        assert_eq!(scheduled.len(), 3);
        assert!((scheduled[0] - 0.5).abs() < 1e-12);
        assert!(scheduled[1] == 0.0, "the scheduled decoder lost the zero");
        assert!((scheduled[2] - 0.7).abs() < 1e-12);
    }

    /// One lost spike is one whole quantum. Stated in the doc; asserted here, because a code whose
    /// fragility is only in prose is a code whose fragility will be forgotten.
    #[test]
    fn losing_one_spike_costs_exactly_one_quantum() {
        let bc = BurstCode::new(16, 3, 20).expect("valid");
        let train = bc.encode_sequence(&[0.5], 0, 0).expect("in range");
        assert_eq!(train.len(), 8);
        let mut kept: Vec<_> = train.spikes().to_vec();
        kept.pop();
        let damaged = crate::spike::Train::from_spikes(kept);
        let got = bc.decode_scheduled(&damaged, 0, 0, 1)[0];
        assert!((got - (0.5 - 2.0 * bc.quantisation())).abs() < 1e-12, "decoded {got}");
    }

    #[test]
    fn burst_detection_splits_on_the_gap_and_the_constructor_refuses_an_ambiguous_one() {
        let bc = BurstCode::new(8, 2, 9).expect("valid");
        let train = bc.encode_parallel(&[1.0, 0.25, 0.0], 100).expect("in range");
        assert_eq!(bc.detect(&train, 0), vec![super::Burst { start: 100, end: 114, spikes: 8 }]);
        assert_eq!(bc.detect(&train, 1), vec![super::Burst { start: 100, end: 102, spikes: 2 }]);
        assert!(bc.detect(&train, 2).is_empty(), "a zero emits nothing at all");
        assert!(matches!(BurstCode::new(0, 2, 9), Err(CodeError::NotPositive { .. })));
        assert!(matches!(BurstCode::new(8, 0, 9), Err(CodeError::NotPositive { .. })));
        // A gap no longer than an intra-burst interval cannot segment anything.
        assert!(matches!(BurstCode::new(8, 5, 5), Err(CodeError::NotPositive { .. })));
        assert!(matches!(bc.spikes_for(1.5), Err(CodeError::OutOfUnitRange { .. })));
        assert!(matches!(bc.spikes_for(f64::NAN), Err(CodeError::NotFinite { .. })));
    }

    // ---------------------------------------------------------------------------------------
    // BSA AND HSA
    // ---------------------------------------------------------------------------------------

    /// A single spike must reconstruct the filter exactly. That is the definition of the decoder,
    /// and it is the one part of deconvolution coding with no error term at all.
    #[test]
    fn one_spike_reconstructs_exactly_the_filter() {
        let h = fir_hann(12).expect("valid");
        let mut bits = vec![false; 40];
        bits[5] = true;
        let rec = spike_convolve(&bits, &h);
        for k in 0..12 {
            assert!((rec[5 + k] - h[k]).abs() < 1e-18, "tap {k}");
        }
        assert!(rec[..5].iter().all(|&v| v == 0.0), "the decoder is not causal");
        assert!(rec[17..].iter().all(|&v| v == 0.0));
        // Two spikes superpose linearly; the decoder is a filter and nothing more.
        let mut two = vec![false; 40];
        two[5] = true;
        two[9] = true;
        let rec2 = spike_convolve(&two, &h);
        for t in 0..40 {
            let want = if (5..17).contains(&t) { h[t - 5] } else { 0.0 }
                + if (9..21).contains(&t) { h[t - 9] } else { 0.0 };
            assert!((rec2[t] - want).abs() < 1e-18, "t {t}");
        }
    }

    /// Requirement (d): a stated bound on a stated filter, swept, AND an honest statement of what
    /// the scheme fails on — measured on the same axis, so the failure is a number.
    #[test]
    fn bsa_reconstructs_below_its_kernel_cutoff_and_fails_abruptly_above_it() {
        let h = fir_hann(24).expect("valid");
        let bsa = Bsa::new(h, 0.955).expect("valid");
        let n = 1_000usize;
        let warm = 30usize;
        // Below the 24-tap Hann kernel's cutoff, which sits near 0.04 cycles/sample.
        for &f in &[0.001f64, 0.002, 0.005, 0.01, 0.02, 0.03] {
            let s = sine(n, f, 0.4, 0.5);
            let bits = bsa.encode(&s).expect("finite");
            let rec = bsa.decode(&bits);
            let e = rms_error(&s[warm..n - warm], &rec[warm..n - warm]).expect("same length");
            assert!(e < 0.025, "f {f} cycles/sample: rms error {e}, bound 0.025");
        }
        // Above it the scheme does not degrade, it collapses: the kernel cannot build the signal.
        // 0.47 is worse than emitting the constant 0.5, whose error would be 0.283.
        for &f in &[0.08f64, 0.12, 0.2] {
            let s = sine(n, f, 0.4, 0.5);
            let bits = bsa.encode(&s).expect("finite");
            let rec = bsa.decode(&bits);
            let e = rms_error(&s[warm..n - warm], &rec[warm..n - warm]).expect("same length");
            assert!(e > 0.4, "f {f}: rms error {e} — the failure did not appear");
        }
    }

    /// The threshold is not a detail. At zero it over-fires by a factor of 34 in error, and the
    /// reason is algebraic: for a unit-sum filter the error improvement is a constant `-1`.
    #[test]
    fn the_bsa_threshold_is_load_bearing_and_zero_is_the_wrong_value() {
        let h = fir_hann(24).expect("valid");
        let s = sine(1_000, 0.01, 0.4, 0.5);
        let mut errs = Vec::new();
        for &th in &[0.0f64, 0.5, 0.9, 0.955] {
            let b = Bsa::new(h.clone(), th).expect("valid");
            let bits = b.encode(&s).expect("finite");
            let rec = b.decode(&bits);
            errs.push(rms_error(&s[30..970], &rec[30..970]).expect("same length"));
        }
        assert!(errs[0] > 0.2, "threshold 0 error {} — expected gross over-firing", errs[0]);
        assert!(errs[3] < 0.01, "threshold 0.955 error {}", errs[3]);
        assert!(errs[0] / errs[3] > 20.0, "the threshold barely mattered: {errs:?}");
        assert!(errs[1] > errs[2] && errs[2] > errs[3], "error is not monotone in threshold: {errs:?}");
    }

    /// The identity a unit-gain kernel buys: a constant signal is encoded at a spike density equal
    /// to its own level, because every spike contributes exactly `Σh = 1` spread over the kernel.
    #[test]
    fn a_constant_signal_is_encoded_at_a_spike_density_equal_to_its_level() {
        let h = fir_hann(24).expect("valid");
        let bsa = Bsa::new(h, 0.955).expect("valid");
        let n = 1_000usize;
        for &c in &[0.1f64, 0.2, 0.3, 0.4, 0.5] {
            let s = vec![c; n];
            let bits = bsa.encode(&s).expect("finite");
            let d = bits[40..n - 40].iter().filter(|&&b| b).count() as f64 / (n - 80) as f64;
            assert!((d - c).abs() < 0.015, "level {c}: spike density {d}");
        }
        // And where it stops holding, stated: near full scale the threshold keeps it from firing
        // often enough, so the density under-reports by about 0.04 at a level of 0.7.
        let s = vec![0.7f64; n];
        let bits = bsa.encode(&s).expect("finite");
        let d = bits[40..n - 40].iter().filter(|&&b| b).count() as f64 / (n - 80) as f64;
        assert!(d < 0.7 - 0.01, "level 0.7: density {d} did not under-report as measured");
    }

    /// The exact invariant, over every filter this module can build — including windowed sincs with
    /// ten negative taps, which the first draft of this module wrongly believed would break it.
    #[test]
    fn strict_hsa_never_reconstructs_above_a_non_negative_signal() {
        let mut filters: Vec<(String, Vec<f64>)> = Vec::new();
        for &taps in &[8usize, 16, 24, 32] {
            filters.push((format!("hann{taps}"), fir_hann(taps).expect("valid")));
            for &cut in &[0.05f64, 0.15, 0.3, 0.45] {
                filters.push((format!("sinc{taps}/{cut}"), fir_lowpass(taps, cut).expect("valid")));
            }
        }
        let with_negative_taps =
            filters.iter().filter(|(_, h)| h.iter().any(|&v| v < 0.0)).count();
        assert!(with_negative_taps >= 8, "only {with_negative_taps} filters have negative taps");
        let mut worst = f64::NEG_INFINITY;
        for (name, h) in &filters {
            let hsa = Hsa::new(h.clone(), 0.0).expect("valid");
            assert!(hsa.never_over_reconstructs());
            let mut signals: Vec<Vec<f64>> = Vec::new();
            for &f in &[0.001f64, 0.01, 0.05, 0.2, 0.45] {
                for &(a, o) in &[(0.4f64, 0.5f64), (0.5, 0.5), (0.05, 0.05), (0.49, 0.51)] {
                    signals.push(sine(400, f, a, o));
                }
            }
            let mut step = vec![0.02f64; 300];
            for v in step.iter_mut().skip(150) {
                *v = 0.95;
            }
            signals.push(step);
            signals.push((0..300).map(|t| f64::from(t) / 300.0).collect());
            for s in &signals {
                let bits = hsa.encode(s).expect("finite");
                let rec = hsa.decode(&bits);
                for t in 0..s.len() {
                    worst = worst.max(rec[t] - s[t]);
                    assert!(rec[t] <= s[t] + 1e-15, "{name} at t {t}: {} > {}", rec[t], s[t]);
                }
            }
        }
        assert!(worst <= 1e-15, "worst over-reconstruction {worst}");
    }

    /// The precondition is on the SIGNAL. Break its non-negativity and the guarantee goes at once.
    #[test]
    fn the_hsa_guarantee_is_about_the_signal_and_not_about_the_filter() {
        let hsa = Hsa::new(fir_hann(16).expect("valid"), 0.0).expect("valid");
        let s = sine(300, 0.01, 0.5, -0.2);
        assert!(s.iter().any(|&v| v < 0.0), "the test signal never goes negative");
        let bits = hsa.encode(&s).expect("finite");
        let rec = hsa.decode(&bits);
        let over = (0..300).map(|t| rec[t] - s[t]).fold(f64::NEG_INFINITY, f64::max);
        // Measured 0.70: an uncovered sample reconstructs as 0.0, and 0.0 is above -0.7.
        assert!(over > 0.5, "over-reconstruction of a negative-going signal was only {over}");
        // The `never_over_reconstructs` flag reports the half of the precondition it can see, and
        // says so — it is true here while the guarantee does not hold.
        assert!(hsa.never_over_reconstructs());
        // The other half it CAN see: a positive threshold gives the guarantee up outright.
        assert!(!Hsa::new(fir_hann(16).expect("valid"), 0.1).expect("valid").never_over_reconstructs());
    }

    /// `HSA` must also actually reconstruct something, or the under-estimate guarantee is satisfied
    /// by emitting no spikes at all.
    #[test]
    fn hsa_reconstructs_a_slow_signal_within_a_stated_bound() {
        for (taps, bound) in [(16usize, 0.045f64), (24, 0.04), (32, 0.025)] {
            let hsa = Hsa::new(fir_hann(taps).expect("valid"), 0.0).expect("valid");
            let s = sine(600, 0.01, 0.4, 0.5);
            let bits = hsa.encode(&s).expect("finite");
            let rec = hsa.decode(&bits);
            let e = rms_error(&s[taps..600 - taps], &rec[taps..600 - taps]).expect("same length");
            assert!(e < bound, "hann{taps}: rms error {e}, bound {bound}");
            let density = bits.iter().filter(|&&b| b).count() as f64 / 600.0;
            assert!(density > 0.2, "hann{taps} fired at density {density}; the guarantee is vacuous");
        }
    }

    #[test]
    fn the_deconvolution_encoders_refuse_an_empty_filter_and_a_non_finite_sample() {
        assert!(matches!(Bsa::new(vec![], 0.9), Err(CodeError::Empty { .. })));
        assert!(matches!(Hsa::new(vec![], 0.0), Err(CodeError::Empty { .. })));
        assert!(matches!(Bsa::new(vec![f64::NAN], 0.9), Err(CodeError::NotFinite { .. })));
        assert!(matches!(Hsa::new(vec![0.5], -0.1), Err(CodeError::NotPositive { .. })));
        let b = Bsa::new(fir_hann(4).expect("valid"), 0.9).expect("valid");
        assert!(matches!(b.encode(&[0.1, f64::INFINITY]), Err(CodeError::NotFinite { .. })));
        assert_eq!(b.filter().len(), 4);
        assert!(rms_error(&[1.0], &[1.0, 2.0]).is_none());
        assert!(rms_error(&[], &[]).is_none());
        assert!(rms_error(&[f64::NAN], &[0.0]).is_none());
    }

    #[test]
    fn a_deconvolution_encoder_produces_a_train_the_rest_of_the_crate_can_consume() {
        let bsa = Bsa::new(fir_hann(16).expect("valid"), 0.955).expect("valid");
        let s = sine(200, 0.01, 0.4, 0.5);
        let bits = bsa.encode(&s).expect("finite");
        let train = bsa.encode_train(&s, 9).expect("finite");
        assert_eq!(train.len(), bits.iter().filter(|&&b| b).count());
        assert!(train.spikes().iter().all(|sp| sp.source == 9));
        let hsa = Hsa::new(fir_hann(16).expect("valid"), 0.0).expect("valid");
        let ht = hsa.encode_train(&s, 0).expect("finite");
        assert_eq!(ht.len(), hsa.encode(&s).expect("finite").iter().filter(|&&b| b).count());
    }

    // ---------------------------------------------------------------------------------------
    // TEMPORAL CONTRAST
    // ---------------------------------------------------------------------------------------

    /// The relationship the module header claims, asserted event for event rather than described.
    #[test]
    fn step_forward_is_the_existing_delta_encoder_capped_at_one_event() {
        for &th in &[0.01f64, 0.05, 0.2] {
            for &f in &[0.002f64, 0.01, 0.05] {
                let s = sine(500, f, 0.4, 0.5);
                let tc = TemporalContrast::new(th, ContrastMode::StepForward).expect("valid");
                let mine = tc.encode(&s, 0).expect("finite");
                let mut de = DeltaEncoder::new(1, th, 1);
                let mut theirs = Vec::new();
                for (t, &v) in s.iter().enumerate() {
                    theirs.extend(de.sample(t as u64, &[v]));
                }
                assert_eq!(mine, theirs, "th {th}, f {f}: the two encoders disagree");
            }
        }
        // And the difference from the UNCAPPED delta encoder is exactly the catch-up on a jump.
        let mut capped = DeltaEncoder::new(1, 0.1, 1);
        let mut free = DeltaEncoder::new(1, 0.1, 100);
        capped.sample(0, &[0.0]);
        free.sample(0, &[0.0]);
        assert_eq!(capped.sample(1, &[0.55]).len(), 1);
        assert_eq!(free.sample(1, &[0.55]).len(), 5);
    }

    /// Requirement (a) for temporal contrast, with the precondition made explicit: the bound is the
    /// threshold, and it holds exactly while the signal moves by no more than one threshold per
    /// sample. A sweep, plus the case where the precondition fails.
    #[test]
    fn step_forward_reconstruction_is_bounded_by_the_threshold_while_the_signal_is_slow_enough() {
        for &amp in &[0.1f64, 0.4, 1.0] {
            for &f in &[0.001f64, 0.004, 0.01] {
                let s = sine(800, f, amp, 0.5);
                let max_step = s.windows(2).map(|w| (w[1] - w[0]).abs()).fold(0.0f64, f64::max);
                let th = max_step * 1.2;
                let tc = TemporalContrast::new(th, ContrastMode::StepForward).expect("valid");
                let ev = tc.encode(&s, 0).expect("finite");
                let rec = tc.decode(&ev, 0, s.len(), s[0]);
                let worst = (0..s.len()).map(|t| (rec[t] - s[t]).abs()).fold(0.0f64, f64::max);
                assert!(worst <= th + 1e-12, "amp {amp} f {f}: worst {worst} vs threshold {th}");
            }
        }
        // Precondition broken: a threshold below the per-sample change, and the bound goes with it.
        let s = sine(600, 0.01, 0.4, 0.5);
        let max_step = s.windows(2).map(|w| (w[1] - w[0]).abs()).fold(0.0f64, f64::max);
        let th = max_step * 0.4;
        let tc = TemporalContrast::new(th, ContrastMode::StepForward).expect("valid");
        let ev = tc.encode(&s, 0).expect("finite");
        let rec = tc.decode(&ev, 0, s.len(), s[0]);
        let worst = (0..s.len()).map(|t| (rec[t] - s[t]).abs()).fold(0.0f64, f64::max);
        assert!(worst > 10.0 * th, "the bound survived a precondition it should not have: {worst}");
    }

    /// The three rules differ in a way that is not a matter of taste, and a ramp separates them in
    /// closed form. Step-forward tracks it and pays a countable number of events; threshold-based
    /// is blind to it and drifts without bound; moving-window is blind to it and emits nothing.
    #[test]
    fn a_slow_ramp_separates_the_three_temporal_contrast_rules_in_closed_form() {
        let th = 0.01f64;
        let slope = 0.0005f64;
        let n = 2_000usize;
        let s: Vec<f64> = (0..n).map(|t| slope * t as f64).collect();
        let total = slope * (n - 1) as f64; // 0.9995

        let sf = TemporalContrast::new(th, ContrastMode::StepForward).expect("valid");
        let ev = sf.encode(&s, 0).expect("finite");
        // Closed form: the baseline moves one threshold per event, so an event is spent for every
        // threshold of rise and for nothing else.
        let want = (total / th).floor() as usize;
        assert!(
            ev.len() == want || ev.len() == want + 1,
            "step-forward emitted {} events for a rise of {total} at threshold {th}; expected {want}",
            ev.len()
        );
        let rec = sf.decode(&ev, 0, n, s[0]);
        let worst = (0..n).map(|t| (rec[t] - s[t]).abs()).fold(0.0f64, f64::max);
        assert!(worst <= th + 1e-12, "step-forward error {worst} exceeded the threshold");

        // Threshold-based differences the signal: 0.0005 per sample never reaches 0.01, so it
        // fires not once, and its reconstruction stays flat while the signal walks away.
        let tbr = TemporalContrast::new(th, ContrastMode::ThresholdBased).expect("valid");
        let ev = tbr.encode(&s, 0).expect("finite");
        assert!(ev.is_empty(), "threshold-based emitted {} events on a sub-threshold ramp", ev.len());
        let rec = tbr.decode(&ev, 0, n, s[0]);
        let worst = (0..n).map(|t| (rec[t] - s[t]).abs()).fold(0.0f64, f64::max);
        assert!((worst - total).abs() < 1e-9, "drift {worst} should be the whole rise {total}");

        // Moving window: its baseline lags by slope * (w+1)/2 = 0.00425, below the threshold, so it
        // is silent too — and silence here means the drift is discarded on purpose rather than
        // lost by accident. That is the trade it exists to make.
        let mw = TemporalContrast::new(th, ContrastMode::MovingWindow { window: 16 }).expect("valid");
        let ev = mw.encode(&s, 0).expect("finite");
        assert!(ev.len() < 5, "moving window emitted {} events on a drift it should ignore", ev.len());
    }

    /// Moving window is the one rule here whose reconstruction error is NOT bounded by the
    /// threshold, because the decoder's baseline is built from the decoder's own output. Measured.
    #[test]
    fn the_moving_window_reconstruction_error_is_not_bounded_by_the_threshold() {
        let th = 0.02f64;
        let s = sine(600, 0.01, 0.4, 0.5);
        let sf = TemporalContrast::new(th, ContrastMode::StepForward).expect("valid");
        let mw = TemporalContrast::new(th, ContrastMode::MovingWindow { window: 8 }).expect("valid");
        let e_sf = {
            let ev = sf.encode(&s, 0).expect("finite");
            let rec = sf.decode(&ev, 0, s.len(), s[0]);
            rms_error(&s, &rec).expect("same length")
        };
        let e_mw = {
            let ev = mw.encode(&s, 0).expect("finite");
            let rec = mw.decode(&ev, 0, s.len(), s[0]);
            rms_error(&s, &rec).expect("same length")
        };
        assert!(e_mw > 3.0 * e_sf, "moving window {e_mw} vs step forward {e_sf}");
    }

    #[test]
    fn the_statistical_threshold_is_refused_where_it_has_nothing_to_measure() {
        let s = sine(600, 0.01, 0.4, 0.5);
        let th = TemporalContrast::threshold_by_statistics(&s, 1.0).expect("a varying signal");
        // mean|Δs| + 1 std for this sinusoid; asserted as a range rather than a magic constant.
        assert!(th > 0.02 && th < 0.03, "statistical threshold {th}");
        assert!(matches!(
            TemporalContrast::threshold_by_statistics(&[1.0, 1.0, 1.0], 1.0),
            Err(CodeError::NoEvidence { .. })
        ));
        assert!(matches!(
            TemporalContrast::threshold_by_statistics(&[1.0], 1.0),
            Err(CodeError::Empty { .. })
        ));
        assert!(matches!(
            TemporalContrast::threshold_by_statistics(&[1.0, f64::NAN], 1.0),
            Err(CodeError::NotFinite { .. })
        ));
        assert!(matches!(
            TemporalContrast::new(0.0, ContrastMode::StepForward),
            Err(CodeError::NotPositive { .. })
        ));
        assert!(matches!(
            TemporalContrast::new(0.1, ContrastMode::MovingWindow { window: 0 }),
            Err(CodeError::NotPositive { .. })
        ));
        assert!(TemporalContrast::new(0.1, ContrastMode::StepForward)
            .expect("valid")
            .encode(&[], 0)
            .expect("an empty record encodes to nothing")
            .is_empty());
    }

    #[test]
    fn temporal_contrast_ignores_events_addressed_to_another_channel() {
        let tc = TemporalContrast::new(0.05, ContrastMode::StepForward).expect("valid");
        let s = sine(200, 0.01, 0.4, 0.5);
        let mine = tc.encode(&s, 3).expect("finite");
        assert!(mine.iter().all(|e| e.address == 3));
        let mut mixed = mine.clone();
        mixed.extend(tc.encode(&s, 4).expect("finite"));
        let a = tc.decode(&mine, 3, s.len(), s[0]);
        let b = tc.decode(&mixed, 3, s.len(), s[0]);
        assert_eq!(a, b, "a foreign channel leaked into the reconstruction");
        assert!(mine.iter().any(|e| e.polarity == Polarity::Off), "a sinusoid must fall as well as rise");
    }

    // ---------------------------------------------------------------------------------------
    // DETERMINISM
    // ---------------------------------------------------------------------------------------

    /// Same seed, same spikes. Every stochastic path in this module at once, because a determinism
    /// test that covers one of them is a test that will be true of one of them.
    #[test]
    fn every_stochastic_encoder_here_is_reproducible_from_its_seed() {
        let gp = GaussianPopulation::new(12, 0.0, 1.0, 0.2, 80.0, 2.0).expect("valid");
        let cp = CosinePopulation::new(9, 20.0, 15.0, 0.4).expect("valid");
        let run = |seed: u64| {
            let mut rng = Rng::new(seed);
            let a = gp.sample_counts(&mut rng, 0.41, 0.3).expect("valid");
            let b = gp.encode_train(&mut rng, 0.41, 100, 1e-3).expect("valid");
            let c = cp.sample_counts(&mut rng, 2.2, 0.3).expect("valid");
            let d = poisson_count(&mut rng, 12.5).expect("finite");
            (a, b, c, d)
        };
        assert_eq!(run(1234), run(1234), "the same seed produced different spikes");
        assert_ne!(run(1234).0, run(9999).0, "two seeds produced the same draw");
    }
}
