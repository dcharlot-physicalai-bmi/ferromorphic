//! How different are two spike trains? The Victor–Purpura edit distance, the van Rossum distance,
//! vector strength and the Fano factor — the four numbers a spiking experiment is most often
//! summarised by, each checked against the cases where its value is known exactly.
//!
//! # What the mechanisms are
//!
//! - **Victor–Purpura** (Victor and Purpura, *Nature and precision of temporal coding in visual
//!   cortex: a metric-space analysis*, Journal of Neurophysiology 76(2):1310–1326, 1996). The
//!   cheapest way to turn one train into the other, where deleting or inserting a spike costs 1
//!   and moving one by `Δt` costs `q |Δt|`. The cost `q` (per second) sets the question being
//!   asked: at `q = 0` only spike COUNTS matter; as `q → ∞` only exact coincidences do; `2/q` is
//!   the time shift at which moving a spike stops being cheaper than deleting and re-inserting it.
//! - **van Rossum** (van Rossum, *A novel spike distance*, Neural Computation 13(4):751–763,
//!   2001). Filter each train with a causal exponential of time constant `τ` and take the `L²`
//!   distance between the traces. The integral has a closed form in the spike times, which is what
//!   [`van_rossum_squared`] evaluates; [`van_rossum_by_quadrature`] does the integral the slow way
//!   and is the referee.
//! - **Vector strength** (Goldberg and Brown, *Response of binaural neurons of dog superior
//!   olivary complex to dichotic tonal stimuli*, Journal of Neurophysiology 32(4):613–636, 1969).
//!   Put every spike on the unit circle at its phase in a stimulus period and take the length of
//!   the mean vector: 1 for perfect phase locking, near 0 for none.
//! - **The Fano factor**: the variance of the spike count in a window over its mean. One for a
//!   Poisson process, less for anything more regular.
//!
//! # Why it is in a neuromorphic crate
//!
//! "The spiking network reproduces the reference" is a claim about spike trains, and these are the
//! units it is made in. A deployment that moves a network between simulators or onto hardware
//! (see [`crate::nir`]) needs a distance with a time scale, not a spike-count comparison, to say
//! how far the result moved — and a learning rule that targets spike times (see
//! [`crate::plasticity`]) is descending one of these.
//!
//! # The closed forms this module is checked against
//!
//! - **Victor–Purpura**: two single spikes `Δ` apart are `min(qΔ, 2)` apart; at `q = 0` the
//!   distance is `|n − m|`; a train and its copy shifted by `δ` (less than half the smallest
//!   interval) are `n · min(qδ, 2)` apart; and it is a metric — zero on identical trains,
//!   symmetric, and obeying the triangle inequality on random triples — between the bounds
//!   `|n − m|` and `n + m`, non-decreasing in `q`.
//! - **van Rossum**: two single spikes `Δ` apart have `D² = 1 − e^{−Δ/τ}`; a train against
//!   silence has `D² = ½ Σ_ij e^{−|t_i − t_j|/τ}` — `n/2` when the spikes are far apart compared
//!   with `τ`; and the closed form equals the quadrature of the filtered traces.
//! - **Vector strength**: exactly 1 for a phase-locked train, exactly 0 for spikes at opposite
//!   phases, and `e^{−2π²σ²/T²}` in expectation for Gaussian jitter `σ` — the characteristic
//!   function of the jitter — held within four standard errors.
//! - **Fano factor**: for a perfectly periodic train counted in windows of `m + f` periods the
//!   count is `m` or `m + 1`, so the factor is exactly `f(1 − f)/(m + f)`; for a Poisson train it
//!   is 1 within sampling error.
//!
//! # What this module has NOT reproduced
//!
//! - The ISI-distance and SPIKE-distance of Kreuz and colleagues, which are parameter-free and
//!   need their own piecewise-linear profiles.
//! - Multi-neuron (labelled-line) extensions of either metric.
//! - The Rayleigh test beyond its first-order p-value `e^{−n r²}`, which is loose for small `n`.

use core::f64::consts::{PI, TAU};
use core::fmt;

/// The most spike pairs one call will compare; past it the request is refused rather than started.
pub const MAX_PAIRS: usize = 400_000_000;

/// What went wrong, named rather than guessed around.
#[derive(Debug, Clone, PartialEq)]
pub enum DistanceError {
    /// A train with nothing in it where a statistic needs at least one spike or window.
    Empty {
        /// What was empty.
        what: &'static str,
    },
    /// A spike train that is not in time order.
    Unsorted {
        /// Which train.
        what: &'static str,
        /// Index of the first spike earlier than its predecessor.
        index: usize,
    },
    /// A `NaN` or infinity.
    NonFinite {
        /// Which quantity.
        what: &'static str,
        /// Position in the offending array, `0` for a scalar.
        index: usize,
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
    /// More spike pairs than [`MAX_PAIRS`].
    TooLarge {
        /// Pairs asked for, saturating.
        pairs: usize,
    },
}

impl fmt::Display for DistanceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty { what } => write!(f, "{what} is empty"),
            Self::Unsorted { what, index } => write!(f, "{what} is out of time order at {index}"),
            Self::NonFinite { what, index } => write!(f, "{what} is not finite at {index}"),
            Self::OutOfRange { what, value, low, high } => {
                write!(f, "{what} = {value} is outside [{low}, {high}]")
            }
            Self::TooLarge { pairs } => write!(f, "{pairs} spike pairs is more than MAX_PAIRS"),
        }
    }
}

impl std::error::Error for DistanceError {}

fn train(what: &'static str, t: &[f64]) -> Result<(), DistanceError> {
    if let Some(i) = t.iter().position(|x| !x.is_finite()) {
        return Err(DistanceError::NonFinite { what, index: i });
    }
    if let Some(i) = t.windows(2).position(|p| p[1] < p[0]) {
        return Err(DistanceError::Unsorted { what, index: i + 1 });
    }
    Ok(())
}

fn positive(what: &'static str, v: f64) -> Result<f64, DistanceError> {
    if v.is_finite() && v > 0.0 {
        Ok(v)
    } else {
        Err(DistanceError::OutOfRange { what, value: v, low: f64::MIN_POSITIVE, high: f64::INFINITY })
    }
}

fn pairs(a: &[f64], b: &[f64]) -> Result<(), DistanceError> {
    let n = (a.len() + b.len()).saturating_mul(a.len() + b.len());
    if n > MAX_PAIRS { Err(DistanceError::TooLarge { pairs: n }) } else { Ok(()) }
}

// ---------------------------------------------------------------------------------------------
// Victor–Purpura
// ---------------------------------------------------------------------------------------------

/// The Victor–Purpura distance between two trains at cost `q` per second of shift.
///
/// # Errors
///
/// [`DistanceError::NonFinite`] or [`DistanceError::Unsorted`] for a bad train,
/// [`DistanceError::OutOfRange`] for a negative or non-finite `q`, [`DistanceError::TooLarge`] past
/// [`MAX_PAIRS`].
pub fn victor_purpura(a: &[f64], b: &[f64], q: f64) -> Result<f64, DistanceError> {
    train("a", a)?;
    train("b", b)?;
    if !(q >= 0.0) || !q.is_finite() {
        return Err(DistanceError::OutOfRange { what: "q", value: q, low: 0.0, high: f64::INFINITY });
    }
    pairs(a, b)?;
    // The usual edit-distance table, one row at a time: row[j] is the distance between the first
    // i spikes of `a` and the first j of `b`.
    let mut row: Vec<f64> = (0..=b.len()).map(|j| j as f64).collect();
    for (i, &ta) in a.iter().enumerate() {
        let mut diagonal = row[0];
        row[0] = (i + 1) as f64;
        for (j, &tb) in b.iter().enumerate() {
            let shift = diagonal + q * (ta - tb).abs();
            let best = shift.min(row[j] + 1.0).min(row[j + 1] + 1.0);
            diagonal = row[j + 1];
            row[j + 1] = best;
        }
    }
    Ok(row[b.len()])
}

// ---------------------------------------------------------------------------------------------
// van Rossum
// ---------------------------------------------------------------------------------------------

fn kernel_sum(x: &[f64], y: &[f64], tau: f64) -> f64 {
    x.iter().map(|s| y.iter().map(|t| (-(s - t).abs() / tau).exp()).sum::<f64>()).sum()
}

/// The squared van Rossum distance `D² = (1/τ) ∫ (f − g)² dt`, with `f` and `g` the trains
/// filtered by `e^{−t/τ}` from each spike: in closed form,
/// `½ [Σ_ij e^{−|a_i − a_j|/τ} + Σ_kl e^{−|b_k − b_l|/τ} − 2 Σ_ik e^{−|a_i − b_k|/τ}]`.
///
/// # Errors
///
/// As [`victor_purpura`], with [`DistanceError::OutOfRange`] for a non-positive `tau`.
pub fn van_rossum_squared(a: &[f64], b: &[f64], tau: f64) -> Result<f64, DistanceError> {
    train("a", a)?;
    train("b", b)?;
    let tau = positive("tau", tau)?;
    pairs(a, b)?;
    // Rounding can leave a hair below zero for identical trains; the quantity is a squared norm.
    Ok((0.5 * (kernel_sum(a, a, tau) + kernel_sum(b, b, tau) - 2.0 * kernel_sum(a, b, tau))).max(0.0))
}

/// The van Rossum distance, the square root of [`van_rossum_squared`].
///
/// # Errors
///
/// As [`van_rossum_squared`].
pub fn van_rossum(a: &[f64], b: &[f64], tau: f64) -> Result<f64, DistanceError> {
    Ok(van_rossum_squared(a, b, tau)?.sqrt())
}

/// The same squared distance by filtering both trains on a grid of step `dt` from the first spike
/// to `tail` time constants past the last, and integrating by the midpoint rule. Slow, and the
/// referee for the closed form; nothing else needs it.
///
/// # Errors
///
/// As [`van_rossum_squared`], plus [`DistanceError::OutOfRange`] for a non-positive `dt` or `tail`
/// and [`DistanceError::TooLarge`] for a grid past [`MAX_PAIRS`] points.
pub fn van_rossum_by_quadrature(a: &[f64], b: &[f64], tau: f64, dt: f64, tail: f64) -> Result<f64, DistanceError> {
    train("a", a)?;
    train("b", b)?;
    let (tau, dt, tail) = (positive("tau", tau)?, positive("dt", dt)?, positive("tail", tail)?);
    let first = a.first().into_iter().chain(b.first()).copied().fold(f64::INFINITY, f64::min);
    let last = a.last().into_iter().chain(b.last()).copied().fold(f64::NEG_INFINITY, f64::max);
    if !first.is_finite() {
        return Ok(0.0);
    }
    let steps = ((last - first + tail * tau) / dt).ceil();
    if steps > MAX_PAIRS as f64 {
        return Err(DistanceError::TooLarge { pairs: usize::MAX });
    }
    let trace = |spikes: &[f64], t: f64| spikes.iter().take_while(|s| **s <= t).map(|s| (-(t - s) / tau).exp()).sum::<f64>();
    let mut acc = 0.0;
    for k in 0..steps as usize {
        let t = first + (k as f64 + 0.5) * dt;
        let d = trace(a, t) - trace(b, t);
        acc += d * d;
    }
    Ok(acc * dt / tau)
}

// ---------------------------------------------------------------------------------------------
// Vector strength
// ---------------------------------------------------------------------------------------------

/// Vector strength and mean phase of a train against a stimulus of period `period`:
/// `r = |Σ_k e^{iφ_k}|/n`, `φ_k = 2π t_k / period`. The phase is in `(−π, π]` and is meaningless
/// when `r` is near zero.
///
/// # Errors
///
/// [`DistanceError::Empty`] for no spikes, [`DistanceError::NonFinite`] for a bad spike time,
/// [`DistanceError::OutOfRange`] for a non-positive period. The train need not be sorted.
pub fn vector_strength(spikes: &[f64], period: f64) -> Result<(f64, f64), DistanceError> {
    if spikes.is_empty() {
        return Err(DistanceError::Empty { what: "spikes" });
    }
    if let Some(i) = spikes.iter().position(|x| !x.is_finite()) {
        return Err(DistanceError::NonFinite { what: "spikes", index: i });
    }
    let period = positive("period", period)?;
    let (mut c, mut s) = (0.0, 0.0);
    for t in spikes {
        // Reduce to one period BEFORE multiplying by 2π, so that a late spike keeps its phase.
        let phase = TAU * (t / period).rem_euclid(1.0);
        c += phase.cos();
        s += phase.sin();
    }
    let n = spikes.len() as f64;
    Ok(((c / n).hypot(s / n), s.atan2(c)))
}

/// The vector strength expected of a perfectly locked train whose spikes carry independent
/// Gaussian jitter of standard deviation `sigma`: `e^{−2π²σ²/T²}`. `None` for a negative jitter
/// or a non-positive period.
#[must_use]
pub fn jittered_vector_strength(sigma: f64, period: f64) -> Option<f64> {
    if !(sigma >= 0.0) || !(period > 0.0) || !sigma.is_finite() || !period.is_finite() {
        return None;
    }
    Some((-2.0 * PI * PI * sigma * sigma / (period * period)).exp())
}

/// The Rayleigh statistic `Z = n r²` and its first-order p-value `e^{−Z}` for the hypothesis that
/// the phases are uniform. `None` for `r` outside `[0, 1]` or no spikes.
#[must_use]
pub fn rayleigh(n: usize, r: f64) -> Option<(f64, f64)> {
    if n == 0 || !(0.0..=1.0).contains(&r) {
        return None;
    }
    let z = n as f64 * r * r;
    Some((z, (-z).exp()))
}

// ---------------------------------------------------------------------------------------------
// The Fano factor
// ---------------------------------------------------------------------------------------------

/// The Fano factor of the spike counts in the consecutive windows of length `window` that fit in
/// `[start, end)`: population variance over mean.
///
/// # Errors
///
/// [`DistanceError::NonFinite`] or [`DistanceError::Unsorted`] for a bad train,
/// [`DistanceError::OutOfRange`] for a non-positive window or an interval that holds fewer than
/// two windows, [`DistanceError::Empty`] when no window holds a spike — a variance over a mean of
/// zero is not a number.
pub fn fano_factor(spikes: &[f64], start: f64, end: f64, window: f64) -> Result<f64, DistanceError> {
    train("spikes", spikes)?;
    let window = positive("window", window)?;
    if !start.is_finite() || !end.is_finite() {
        return Err(DistanceError::NonFinite { what: "interval", index: 0 });
    }
    let windows = ((end - start) / window).floor();
    if !(windows >= 2.0) || windows > MAX_PAIRS as f64 {
        return Err(DistanceError::OutOfRange { what: "windows", value: windows, low: 2.0, high: MAX_PAIRS as f64 });
    }
    let mut counts = vec![0u64; windows as usize];
    for &t in spikes {
        let k = ((t - start) / window).floor();
        if k >= 0.0 && k < windows {
            counts[k as usize] += 1;
        }
    }
    let n = counts.len() as f64;
    let mean = counts.iter().sum::<u64>() as f64 / n;
    if mean == 0.0 {
        return Err(DistanceError::Empty { what: "spikes inside the windows" });
    }
    let var = counts.iter().map(|&c| (c as f64 - mean) * (c as f64 - mean)).sum::<f64>() / n;
    Ok(var / mean)
}

/// The Fano factor of a perfectly periodic train counted in windows of `window`: with
/// `window/period = m + f`, the count is `m + 1` a fraction `f` of the time and `m` otherwise, so
/// the factor is `f (1 − f)/(m + f)`. `None` for a non-positive argument.
#[must_use]
pub fn periodic_fano(period: f64, window: f64) -> Option<f64> {
    if !(period > 0.0) || !(window > 0.0) || !period.is_finite() || !window.is_finite() {
        return None;
    }
    let ratio = window / period;
    let f = ratio - ratio.floor();
    Some(f * (1.0 - f) / ratio)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fusion::gaussian;
    use crate::rng::Rng;

    fn random_train(n: usize, span: f64, rng: &mut Rng) -> Vec<f64> {
        let mut t: Vec<f64> = (0..n).map(|_| span * rng.next_f64()).collect();
        t.sort_by(f64::total_cmp);
        t
    }

    #[test]
    fn two_spikes_are_as_far_apart_as_the_cheaper_of_moving_and_replacing() {
        // q = 8 /s: moving costs 8 per second, so the two options meet at Δ = 2/q = 0.25 s.
        for (delta, want) in [(0.0, 0.0), (0.125, 1.0), (0.25, 2.0), (0.5, 2.0), (4.0, 2.0)] {
            assert_eq!(victor_purpura(&[1.0], &[1.0 + delta], 8.0).unwrap(), want, "Δ = {delta}");
        }
        // q = 0 counts spikes and nothing else.
        assert_eq!(victor_purpura(&[0.1, 0.2, 0.9], &[5.0], 0.0).unwrap(), 2.0);
        assert_eq!(victor_purpura(&[], &[0.3, 0.4], 3.0).unwrap(), 2.0);
        assert_eq!(victor_purpura(&[], &[], 3.0).unwrap(), 0.0);
        // A train and its copy shifted by δ = 1/16 s, intervals at least 0.5 s: n·min(qδ, 2).
        let a = [0.5, 1.0, 1.75, 3.0, 3.5];
        let b: Vec<f64> = a.iter().map(|t| t + 0.0625).collect();
        assert_eq!(victor_purpura(&a, &b, 4.0).unwrap(), 5.0 * 0.25);
        assert_eq!(victor_purpura(&a, &b, 64.0).unwrap(), 10.0, "past q = 2/δ = 32 it is cheaper to replace every spike");
        // An extra spike in the middle is one insertion; the rest still pair with their copies,
        // not with their neighbours.
        let mut extra = b.clone();
        extra.insert(2, 1.4);
        assert_eq!(victor_purpura(&a, &extra, 4.0).unwrap(), 1.0 + 5.0 * 0.25);
    }

    #[test]
    fn victor_purpura_is_a_metric_between_its_two_bounds() {
        let mut rng = Rng::new(14);
        for _ in 0..60 {
            let (a, b, c) = (random_train(7, 2.0, &mut rng), random_train(5, 2.0, &mut rng), random_train(9, 2.0, &mut rng));
            let q = 6.0 * rng.next_f64();
            let d = |x: &[f64], y: &[f64]| victor_purpura(x, y, q).unwrap();
            assert_eq!(d(&a, &a), 0.0);
            assert_eq!(d(&a, &b), d(&b, &a));
            assert!(d(&a, &c) <= d(&a, &b) + d(&b, &c) + 1e-12, "the triangle inequality failed at q = {q}");
            assert!(d(&a, &b) >= 2.0 - 1e-12 && d(&a, &b) <= 12.0 + 1e-12, "outside [|n − m|, n + m]: {}", d(&a, &b));
            assert!(victor_purpura(&a, &b, q + 1.0).unwrap() >= d(&a, &b) - 1e-12, "the distance fell as q rose");
        }
        // The bounds are reached: q = 0 gives |n − m|, and a huge q with no coincidences gives n + m.
        let (a, b) = (random_train(7, 2.0, &mut rng), random_train(5, 2.0, &mut rng));
        assert_eq!(victor_purpura(&a, &b, 0.0).unwrap(), 2.0);
        assert_eq!(victor_purpura(&a, &b, 1e9).unwrap(), 12.0);
    }

    #[test]
    fn van_rossum_has_the_closed_form_of_its_own_integral() {
        let tau = 0.02;
        // Two single spikes: D² = 1 − e^{−Δ/τ}; one τ apart that is 1 − 1/e.
        assert!((van_rossum_squared(&[0.1], &[0.12], tau).unwrap() - (1.0 - (-1.0f64).exp())).abs() < 1e-15);
        assert_eq!(van_rossum_squared(&[0.1], &[0.1], tau).unwrap(), 0.0);
        // Against silence a lone spike is ½; n spikes far apart compared with τ are n/2.
        assert_eq!(van_rossum_squared(&[0.3], &[], tau).unwrap(), 0.5);
        assert!((van_rossum_squared(&[0.0, 1.0, 2.0, 3.0], &[], tau).unwrap() - 2.0).abs() < 1e-15);
        // Two spikes CLOSE together against silence interfere: ½ (2 + 2 e^{−Δ/τ}).
        assert!((van_rossum_squared(&[0.0, 0.01], &[], tau).unwrap() - (1.0 + (-0.5f64).exp())).abs() < 1e-15);
        // And the closed form is the integral it abbreviates, on trains with nothing special
        // about them: midpoint rule at dt = τ/2000, tail of 25 τ (e^{−50} of the energy left out).
        let mut rng = Rng::new(3);
        let (a, b) = (random_train(12, 0.4, &mut rng), random_train(9, 0.4, &mut rng));
        let closed = van_rossum_squared(&a, &b, tau).unwrap();
        let slow = van_rossum_by_quadrature(&a, &b, tau, tau / 2000.0, 25.0).unwrap();
        // The traces jump at spikes, so the midpoint rule is first order there: 21 jumps of
        // size ≤ a few, each misplaced by at most dt/2.
        assert!((closed - slow).abs() < 1e-3 * closed, "closed form {closed}, quadrature {slow}");
        assert!(closed > 1.0, "the two trains are all but identical and the comparison is empty: {closed}");
        assert_eq!(van_rossum(&a, &b, tau).unwrap(), closed.sqrt());
        assert_eq!(van_rossum_by_quadrature(&[], &[], tau, 1e-4, 10.0).unwrap(), 0.0);
        // Symmetric, and zero only on identical trains.
        assert_eq!(van_rossum_squared(&b, &a, tau).unwrap(), closed);
        assert!(van_rossum_squared(&a, &a, tau).unwrap() < 1e-12);
    }

    #[test]
    fn vector_strength_is_one_when_locked_and_the_jitters_characteristic_function_when_not() {
        let period = 4e-3;
        let locked: Vec<f64> = (0..200).map(|k| 1e-3 + period * f64::from(k)).collect();
        let (r, phase) = vector_strength(&locked, period).unwrap();
        assert!((r - 1.0).abs() < 1e-12);
        assert!((phase - PI / 2.0).abs() < 1e-9, "a quarter of the way through the cycle is a phase of π/2: {phase}");
        // Alternating between opposite phases cancels exactly; a third spike at a quarter turn
        // leaves a third.
        assert!(vector_strength(&[0.0, 2e-3, 8e-3, 14e-3], period).unwrap().0 < 1e-12);
        assert!((vector_strength(&[0.0, 2e-3, 1e-3], period).unwrap().0 - 1.0 / 3.0).abs() < 1e-12);
        // Gaussian jitter of σ = 0.6 ms on a 4 ms cycle: e^{−2π²·0.0225} = 0.6414.
        let want = jittered_vector_strength(0.6e-3, period).unwrap();
        assert!((want - (-2.0 * PI * PI * 0.0225f64).exp()).abs() < 1e-15 && (want - 0.6414).abs() < 1e-4);
        let mut rng = Rng::new(8);
        let n = 20_000;
        let jittered: Vec<f64> = (0..n).map(|k| 1e-3 + period * f64::from(k) + 0.6e-3 * gaussian(&mut rng)).collect();
        let (r, phase) = vector_strength(&jittered, period).unwrap();
        // The mean vector's length has a standard error below 1/√n.
        assert!((r - want).abs() < 4.0 / f64::from(n).sqrt(), "r = {r}, the characteristic function says {want}");
        assert!((phase - PI / 2.0).abs() < 0.05);
        assert_eq!(jittered_vector_strength(0.0, period), Some(1.0));
        assert_eq!(jittered_vector_strength(-1.0, period), None);
        assert_eq!(jittered_vector_strength(1e-3, 0.0), None);
        // Rayleigh: Z = n r², p ≈ e^{−Z}.
        assert_eq!(rayleigh(64, 0.25), Some((4.0, (-4.0f64).exp())), "binary fractions, so Z is exactly 4");
        assert_eq!(rayleigh(0, 0.2), None);
        assert_eq!(rayleigh(10, 1.5), None);
    }

    #[test]
    fn the_fano_factor_of_a_clock_is_f_times_one_minus_f_over_the_mean() {
        // Period 1, window 2.25: over any four windows the counts are 2, 2, 2, 3 in some order.
        let clock: Vec<f64> = (0..900).map(|k| 0.1 + f64::from(k)).collect();
        let got = fano_factor(&clock, 0.0, 900.0, 2.25).unwrap();
        assert_eq!(periodic_fano(1.0, 2.25), Some(0.25 * 0.75 / 2.25));
        assert!((got - 0.25 * 0.75 / 2.25).abs() < 1e-12, "{got}");
        // A window of a whole number of periods always holds the same count: no variance at all.
        assert_eq!(fano_factor(&clock, 0.0, 900.0, 3.0).unwrap(), 0.0);
        assert_eq!(periodic_fano(1.0, 3.0), Some(0.0));
        // Poisson: exponential intervals at 50 /s, 4000 windows of 0.2 s (mean count 10). The Fano
        // factor of N Poisson counts has standard error √(2/N) = 0.022.
        let mut rng = Rng::new(10);
        let mut t = 0.0;
        let mut poisson = Vec::new();
        while t < 800.0 {
            t -= (1.0 - rng.next_f64()).ln() / 50.0;
            poisson.push(t);
        }
        let ff = fano_factor(&poisson, 0.0, 800.0, 0.2).unwrap();
        assert!((ff - 1.0).abs() < 4.0 * (2.0f64 / 4000.0).sqrt(), "a Poisson train has a Fano factor of {ff}");
        for none in [periodic_fano(0.0, 1.0), periodic_fano(1.0, 0.0), periodic_fano(f64::NAN, 1.0)] {
            assert_eq!(none, None);
        }
    }

    #[test]
    fn bad_arguments_are_refused() {
        assert!(matches!(victor_purpura(&[0.2, 0.1], &[], 1.0), Err(DistanceError::Unsorted { what: "a", index: 1 })));
        assert!(matches!(victor_purpura(&[], &[0.1, f64::NAN], 1.0), Err(DistanceError::NonFinite { what: "b", index: 1 })));
        assert!(matches!(victor_purpura(&[], &[], -1.0), Err(DistanceError::OutOfRange { what: "q", .. })));
        assert!(matches!(victor_purpura(&[], &[], f64::INFINITY), Err(DistanceError::OutOfRange { what: "q", .. })));
        let long = vec![0.0; 15_000];
        assert!(matches!(victor_purpura(&long, &long, 1.0), Err(DistanceError::TooLarge { .. })));
        assert!(matches!(van_rossum_squared(&long, &long, 1.0), Err(DistanceError::TooLarge { .. })));
        assert!(matches!(van_rossum_squared(&[], &[], 0.0), Err(DistanceError::OutOfRange { what: "tau", .. })));
        assert!(matches!(van_rossum(&[1.0, 0.5], &[], 1.0), Err(DistanceError::Unsorted { .. })));
        assert!(matches!(van_rossum_by_quadrature(&[0.0], &[], 1.0, 0.0, 5.0), Err(DistanceError::OutOfRange { what: "dt", .. })));
        assert!(matches!(van_rossum_by_quadrature(&[0.0], &[], 1.0, 1e-3, 0.0), Err(DistanceError::OutOfRange { what: "tail", .. })));
        assert!(matches!(van_rossum_by_quadrature(&[0.0], &[1e9], 1.0, 1e-3, 5.0), Err(DistanceError::TooLarge { .. })));
        assert!(matches!(vector_strength(&[], 1.0), Err(DistanceError::Empty { .. })));
        assert!(matches!(vector_strength(&[f64::INFINITY], 1.0), Err(DistanceError::NonFinite { .. })));
        assert!(matches!(vector_strength(&[0.1], 0.0), Err(DistanceError::OutOfRange { what: "period", .. })));
        // A spike 10¹² cycles late keeps its phase, because the time is reduced to one period BEFORE
        // it is multiplied by 2π: 2π·10¹² has an ulp of a milliradian, and at a mere million cycles
        // — the first version of this line — the loss was 7e-10 and skipping the reduction survived.
        let (r, phase) = vector_strength(&[0.25, 1_000_000_000_000.25], 1.0).unwrap();
        assert!((r - 1.0).abs() < 1e-12 && (phase - PI / 2.0).abs() < 1e-12, "r = {r}, phase = {phase}");
        assert!(matches!(fano_factor(&[0.1], 0.0, 1.0, 0.0), Err(DistanceError::OutOfRange { what: "window", .. })));
        assert!(matches!(fano_factor(&[0.1], 0.0, 1.0, 0.6), Err(DistanceError::OutOfRange { what: "windows", .. })));
        assert!(matches!(fano_factor(&[0.1], f64::NAN, 1.0, 0.1), Err(DistanceError::NonFinite { what: "interval", .. })));
        assert!(matches!(fano_factor(&[5.0], 0.0, 1.0, 0.1), Err(DistanceError::Empty { .. })));
        assert!(matches!(fano_factor(&[0.2, 0.1], 0.0, 1.0, 0.1), Err(DistanceError::Unsorted { .. })));
        // Spikes outside the interval are not counted: one spike in the first of two windows.
        assert_eq!(fano_factor(&[-1.0, 0.1, 7.0], 0.0, 1.0, 0.5).unwrap(), 0.25 / 0.5);
    }
}
