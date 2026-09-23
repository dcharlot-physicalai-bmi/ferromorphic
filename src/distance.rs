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
//! - **The ISI-distance and the SPIKE-distance** (Kreuz, Haas, Morelli, Abarbanel and Politi,
//!   *Measuring spike train synchrony*, Journal of Neuroscience Methods 165(1):151–161, 2007;
//!   Kreuz, Chicharro, Houghton, Andrzejak and Mormann, *Monitoring spike train synchrony*, Journal
//!   of Neurophysiology 109(5):1457–1472, 2013). Both are PARAMETER-FREE: there is no `q` or `τ` to
//!   choose. At every instant the first compares the lengths of the two interspike intervals that
//!   contain it, `|x₁ − x₂|/max(x₁, x₂)`; the second compares spike TIMES, weighting each train's
//!   distance to the other's nearest spikes by how close the instant is to them. Each is the time
//!   average of its profile, which is piecewise constant for the first and piecewise linear for
//!   the second, so both integrals are exact.
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
//! - **ISI-distance**: two clocks of periods `p₁` and `p₂` are `|p₁ − p₂|/max(p₁, p₂)` apart
//!   whatever their phases. **SPIKE-distance**: two clocks of the same period `p` offset by
//!   `δ ≤ p/2` are exactly `δ/p` apart. Both are zero on identical trains, symmetric, and at
//!   most one. For a POPULATION the distance is the mean over all pairs ([`multi_train`]): three
//!   clocks of periods 2, 3 and 4 are `(1/3 + 1/2 + 1/4)/3` apart.
//!
//! # What this module has NOT reproduced
//!
//! - An edge convention for the Kreuz distances. Published implementations differ in what they
//!   do before a train's first spike and after its last; [`isi_distance`] and [`spike_distance`]
//!   instead REFUSE a window that either train does not bracket with a spike on each side.
//! - The adaptive and real-time variants of the Kreuz distances. ([`multi_train`] is their
//!   multi-train form, the mean over all pairs.)
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
// The Kreuz distances
// ---------------------------------------------------------------------------------------------

/// Index `k` with `t[k] <= x < t[k + 1]`, for a train that brackets `x`.
fn interval_of(t: &[f64], x: f64) -> usize {
    t.partition_point(|s| *s <= x).saturating_sub(1).min(t.len().saturating_sub(2))
}

fn nearest_gap(spike: f64, other: &[f64]) -> f64 {
    let k = other.partition_point(|s| *s < spike);
    let after = other.get(k).map_or(f64::INFINITY, |s| s - spike);
    let before = if k > 0 { spike - other[k - 1] } else { f64::INFINITY };
    after.min(before)
}

fn bracketed(what: &'static str, t: &[f64], start: f64, end: f64) -> Result<(), DistanceError> {
    train(what, t)?;
    let ok = t.first().is_some_and(|s| *s <= start) && t.last().is_some_and(|s| *s >= end) && t.windows(2).all(|p| p[1] > p[0]);
    if ok { Ok(()) } else { Err(DistanceError::Empty { what: "a spike on each side of the window (or the train repeats a spike)" }) }
}

fn window(start: f64, end: f64) -> Result<(), DistanceError> {
    if !start.is_finite() || !end.is_finite() {
        return Err(DistanceError::NonFinite { what: "window", index: 0 });
    }
    if !(end > start) {
        return Err(DistanceError::OutOfRange { what: "end", value: end, low: start, high: f64::INFINITY });
    }
    Ok(())
}

/// The breakpoints of both profiles inside `[start, end]`: the window's ends and every spike of
/// either train strictly between them, in order.
fn breakpoints(a: &[f64], b: &[f64], start: f64, end: f64) -> Vec<f64> {
    let mut cuts: Vec<f64> = a.iter().chain(b).copied().filter(|t| *t > start && *t < end).collect();
    cuts.push(start);
    cuts.push(end);
    cuts.sort_by(f64::total_cmp);
    cuts.dedup();
    cuts
}

/// The ISI-distance over `[start, end]`: the time average of `|x₁ − x₂|/max(x₁, x₂)`, with `x_n`
/// the length of the interspike interval of train `n` that contains the instant.
///
/// # Errors
///
/// [`DistanceError::NonFinite`] or [`DistanceError::Unsorted`] for a bad train,
/// [`DistanceError::OutOfRange`] for an empty window, and [`DistanceError::Empty`] unless BOTH
/// trains have a spike at or before `start` and at or after `end`, with no repeated spike.
pub fn isi_distance(a: &[f64], b: &[f64], start: f64, end: f64) -> Result<f64, DistanceError> {
    window(start, end)?;
    bracketed("a", a, start, end)?;
    bracketed("b", b, start, end)?;
    let cuts = breakpoints(a, b, start, end);
    let mut acc = 0.0;
    for pair in cuts.windows(2) {
        let mid = 0.5 * (pair[0] + pair[1]);
        let (i, j) = (interval_of(a, mid), interval_of(b, mid));
        let (x1, x2) = (a[i + 1] - a[i], b[j + 1] - b[j]);
        acc += (x1 - x2).abs() / x1.max(x2) * (pair[1] - pair[0]);
    }
    Ok(acc / (end - start))
}

/// The SPIKE-distance over `[start, end]`: the time average of
/// `S(t) = (S₁ x₂ + S₂ x₁)/(2 ⟨x⟩²)`, where for train `n` with previous spike `P` and following
/// spike `F`, `S_n = (Δ_P (t_F − t) + Δ_F (t − t_P))/x_n`, `Δ` being a spike's distance to the
/// nearest spike of the OTHER train, `x_n = t_F − t_P`, and `⟨x⟩` the mean of the two intervals.
///
/// # Errors
///
/// As [`isi_distance`].
pub fn spike_distance(a: &[f64], b: &[f64], start: f64, end: f64) -> Result<f64, DistanceError> {
    window(start, end)?;
    bracketed("a", a, start, end)?;
    bracketed("b", b, start, end)?;
    let cuts = breakpoints(a, b, start, end);
    let mut acc = 0.0;
    for pair in cuts.windows(2) {
        let mid = 0.5 * (pair[0] + pair[1]);
        let (i, j) = (interval_of(a, mid), interval_of(b, mid));
        let (x1, x2) = (a[i + 1] - a[i], b[j + 1] - b[j]);
        let gaps = [nearest_gap(a[i], b), nearest_gap(a[i + 1], b), nearest_gap(b[j], a), nearest_gap(b[j + 1], a)];
        let profile = |t: f64| {
            let s1 = (gaps[0] * (a[i + 1] - t) + gaps[1] * (t - a[i])) / x1;
            let s2 = (gaps[2] * (b[j + 1] - t) + gaps[3] * (t - b[j])) / x2;
            let mean = 0.5 * (x1 + x2);
            (s1 * x2 + s2 * x1) / (2.0 * mean * mean)
        };
        // Linear between breakpoints, so the trapezoid is the integral.
        acc += 0.5 * (profile(pair[0]) + profile(pair[1])) * (pair[1] - pair[0]);
    }
    Ok(acc / (end - start))
}

/// A pairwise distance over a window, as [`isi_distance`] and [`spike_distance`] are.
pub type PairDistance = fn(&[f64], &[f64], f64, f64) -> Result<f64, DistanceError>;

/// The multi-train form of a Kreuz distance: the mean of `distance` over all PAIRS of trains — how
/// far from synchronous a population is, as one number in `[0, 1]`.
///
/// # Errors
///
/// [`DistanceError::Empty`] for fewer than two trains, and whatever `distance` returns for any
/// pair.
pub fn multi_train(trains: &[&[f64]], start: f64, end: f64, distance: PairDistance) -> Result<f64, DistanceError> {
    if trains.len() < 2 {
        return Err(DistanceError::Empty { what: "trains (needs two)" });
    }
    let mut acc = 0.0;
    let mut pairs = 0.0;
    for i in 0..trains.len() {
        for j in (i + 1)..trains.len() {
            acc += distance(trains[i], trains[j], start, end)?;
            pairs += 1.0;
        }
    }
    Ok(acc / pairs)
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

    fn clock(period: f64, offset: f64, until: f64) -> Vec<f64> {
        let mut t = Vec::new();
        let mut k = 0.0;
        while offset + period * k <= until {
            t.push(offset + period * k);
            k += 1.0;
        }
        t
    }

    #[test]
    fn the_kreuz_distances_of_two_clocks_are_their_closed_forms() {
        // Periods 2 and 3, any phases: every instant sits in an interval of 2 and one of 3.
        let (a, b) = (clock(2.0, -1.5, 40.0), clock(3.0, -2.25, 40.0));
        assert!((isi_distance(&a, &b, 0.0, 30.0).unwrap() - 1.0 / 3.0).abs() < 1e-15);
        assert!((isi_distance(&a, &b, 3.7, 29.1).unwrap() - 1.0 / 3.0).abs() < 1e-15, "and on any window");
        assert_eq!(isi_distance(&a, &a, 0.0, 30.0).unwrap(), 0.0);
        assert_eq!(isi_distance(&a, &b, 0.0, 30.0).unwrap(), isi_distance(&b, &a, 0.0, 30.0).unwrap());
        // The same period, offset by δ = 0.25 of a period of 2: SPIKE-distance δ/p = 0.125 exactly.
        let (c, d) = (clock(2.0, -4.0, 40.0), clock(2.0, -3.5, 40.0));
        assert!((spike_distance(&c, &d, 0.0, 30.0).unwrap() - 0.25).abs() < 1e-15, "δ/p = 0.5/2");
        let e = clock(2.0, -3.75, 40.0);
        assert!((spike_distance(&c, &e, 1.3, 27.9).unwrap() - 0.125).abs() < 1e-15);
        assert_eq!(spike_distance(&c, &c, 0.0, 30.0).unwrap(), 0.0);
        assert_eq!(spike_distance(&c, &e, 0.0, 30.0).unwrap(), spike_distance(&e, &c, 0.0, 30.0).unwrap());
        // The ISI-distance cannot see a shift at all — that is what the SPIKE-distance is for.
        assert_eq!(isi_distance(&c, &d, 0.0, 30.0).unwrap(), 0.0);
        // By hand, one window between two spikes of each: a = {0, 4}, b = {0, 2, 4}.
        // ISI profile is |4 − 2|/4 = ½ throughout.
        assert_eq!(isi_distance(&[0.0, 4.0], &[0.0, 2.0, 4.0], 0.0, 4.0).unwrap(), 0.5);
        // SPIKE: train a's spikes coincide with b's (Δ = 0) so S₁ = 0; b's middle spike is 2 from
        // a's nearest, so on (0, 2) S₂ = 2·t/2 = t and S = (0·2 + t·4)/(2·3²) = 2t/9, mean 2/9;
        // (2, 4) mirrors it.
        assert!((spike_distance(&[0.0, 4.0], &[0.0, 2.0, 4.0], 0.0, 4.0).unwrap() - 2.0 / 9.0).abs() < 1e-15);
        // Both stay within [0, 1] on random trains.
        let mut rng = Rng::new(44);
        for _ in 0..30 {
            let mut x = random_train(20, 10.0, &mut rng);
            let mut y = random_train(14, 10.0, &mut rng);
            for t in [&mut x, &mut y] {
                t.insert(0, -1.0);
                t.push(11.0);
            }
            for d in [isi_distance(&x, &y, 0.0, 10.0).unwrap(), spike_distance(&x, &y, 0.0, 10.0).unwrap()] {
                assert!((0.0..=1.0).contains(&d) && d > 0.01, "{d}");
            }
        }
    }

    #[test]
    fn a_population_is_as_far_from_synchrony_as_the_mean_of_its_pairs() {
        let (a, b, c) = (clock(2.0, -1.5, 60.0), clock(3.0, -2.25, 60.0), clock(4.0, -0.5, 60.0));
        let got = multi_train(&[&a, &b, &c], 0.0, 48.0, isi_distance).unwrap();
        // |2−3|/3, |2−4|/4 and |3−4|/4.
        assert!((got - (1.0 / 3.0 + 0.5 + 0.25) / 3.0).abs() < 1e-15, "{got}");
        // Four copies of one clock offset by 0, ⅛, ¼ and ⅜ of a period of 2: the six pairwise
        // offsets are ⅛ (three times), ¼ (twice) and ⅜ (once) of the period.
        let shifted: Vec<Vec<f64>> = (0..4).map(|k| clock(2.0, -4.0 + 0.25 * f64::from(k), 60.0)).collect();
        let refs: Vec<&[f64]> = shifted.iter().map(Vec::as_slice).collect();
        let got = multi_train(&refs, 0.0, 48.0, spike_distance).unwrap();
        assert!((got - (3.0 * 0.125 + 2.0 * 0.25 + 0.375) / 6.0).abs() < 1e-14, "{got}");
        assert_eq!(multi_train(&[&a, &a, &a], 0.0, 48.0, spike_distance).unwrap(), 0.0);
        // Two trains: the pairwise distance itself.
        assert_eq!(multi_train(&[&a, &b], 0.0, 48.0, isi_distance).unwrap(), isi_distance(&a, &b, 0.0, 48.0).unwrap());
        assert!(matches!(multi_train(&[&a], 0.0, 48.0, isi_distance), Err(DistanceError::Empty { .. })));
        assert!(matches!(multi_train(&[&a, &[1.0, 2.0]], 0.0, 48.0, isi_distance), Err(DistanceError::Empty { .. })), "one train does not bracket the window");
    }

    /// Across three modules: two LIF cells under different currents, their spike times taken from
    /// the membrane equation rather than from tick boundaries, are apart by the ISI-distance of
    /// their two closed-form intervals — at a 1 ms tick that neither interval is a multiple of.
    #[test]
    fn two_cells_are_as_far_apart_as_their_closed_form_intervals() {
        use crate::neuron::{Lif, Neuron};
        let cell = Lif::default();
        let (i1, i2) = (4e-9, 2.5e-9);
        let (t1, t2) = (cell.isi(i1).unwrap(), cell.isi(i2).unwrap());
        let dt = 1e-3;
        let trains = |exact: bool| {
            let mut out = Vec::new();
            for i in [i1, i2] {
                let (mut c, mut times, mut offsets) = (cell, Vec::new(), Vec::new());
                for k in 0..4000u32 {
                    if exact {
                        offsets.clear();
                        c.step_exact_times(dt, i, &mut offsets).unwrap();
                        times.extend(offsets.iter().map(|o| f64::from(k) * dt + o));
                    } else if c.step(dt, i) {
                        times.push(f64::from(k + 1) * dt);
                    }
                }
                out.push(times);
            }
            out
        };
        let want = (t1 - t2).abs() / t1.max(t2);
        let exact = trains(true);
        let got = isi_distance(&exact[0], &exact[1], 0.5, 3.5).unwrap();
        assert!((got - want).abs() < 1e-10, "ISI-distance {got}, the two intervals say {want}");
        // Spike times read off tick boundaries give the distance between the ROUNDED intervals —
        // 12 and 21 ms for 11.4 and 20.3 — which is a different number.
        let ticked = trains(false);
        let rounded = isi_distance(&ticked[0], &ticked[1], 0.5, 3.5).unwrap();
        assert!((rounded - want).abs() > 0.005, "tick-boundary spike times gave {rounded} against {want}");
        assert!((rounded - 9.0 / 21.0).abs() < 1e-9, "{rounded}");
    }

    /// The two profiles written straight from their definitions — linear scans, no bisection, no
    /// breakpoints — and integrated on a fine grid: the referee for the exact integration above.
    #[test]
    fn the_kreuz_distances_equal_the_quadrature_of_their_definitions() {
        let around = |t: &[f64], x: f64| {
            let p = t.iter().copied().filter(|s| *s <= x).fold(f64::NEG_INFINITY, f64::max);
            let f = t.iter().copied().filter(|s| *s > x).fold(f64::INFINITY, f64::min);
            (p, f)
        };
        let gap = |spike: f64, other: &[f64]| other.iter().map(|s| (s - spike).abs()).fold(f64::INFINITY, f64::min);
        let mut rng = Rng::new(52);
        for _ in 0..4 {
            let mut a = random_train(9, 6.0, &mut rng);
            let mut b = random_train(6, 6.0, &mut rng);
            for t in [&mut a, &mut b] {
                t.insert(0, -0.7);
                t.push(6.4);
            }
            b[0] = -0.2;
            let n = 400_000;
            let (mut isi, mut spk) = (0.0, 0.0);
            for k in 0..n {
                let t = 6.0 * (f64::from(k) + 0.5) / f64::from(n);
                let ((p1, f1), (p2, f2)) = (around(&a, t), around(&b, t));
                let (x1, x2) = (f1 - p1, f2 - p2);
                isi += (x1 - x2).abs() / x1.max(x2);
                let s1 = (gap(p1, &b) * (f1 - t) + gap(f1, &b) * (t - p1)) / x1;
                let s2 = (gap(p2, &a) * (f2 - t) + gap(f2, &a) * (t - p2)) / x2;
                spk += (s1 * x2 + s2 * x1) / (2.0 * (0.5 * (x1 + x2)).powi(2));
            }
            let (isi, spk) = (isi / f64::from(n), spk / f64::from(n));
            // The ISI profile jumps at 15 spikes, each misplaced by at most half a grid step.
            assert!((isi_distance(&a, &b, 0.0, 6.0).unwrap() - isi).abs() < 15.0 * 6.0 / f64::from(n));
            assert!((spike_distance(&a, &b, 0.0, 6.0).unwrap() - spk).abs() < 15.0 * 6.0 / f64::from(n), "{} vs {spk}", spike_distance(&a, &b, 0.0, 6.0).unwrap());
        }
    }

    #[test]
    fn the_kreuz_distances_refuse_a_window_they_would_have_to_invent_an_edge_for() {
        let inside = [1.0, 2.0, 3.0];
        let around = [-1.0, 2.0, 5.0];
        for f in [isi_distance, spike_distance] {
            assert!(matches!(f(&inside, &around, 0.0, 4.0), Err(DistanceError::Empty { .. })), "train a starts inside the window");
            assert!(matches!(f(&around, &inside, 0.0, 4.0), Err(DistanceError::Empty { .. })));
            assert!(matches!(f(&[], &around, 0.0, 4.0), Err(DistanceError::Empty { .. })));
            assert!(matches!(f(&[-1.0, 2.0, 2.0, 5.0], &around, 0.0, 4.0), Err(DistanceError::Empty { .. })), "a repeated spike is an interval of zero");
            assert!(matches!(f(&around, &around, 4.0, 4.0), Err(DistanceError::OutOfRange { what: "end", .. })));
            assert!(matches!(f(&around, &around, f64::NAN, 4.0), Err(DistanceError::NonFinite { what: "window", .. })));
            assert!(matches!(f(&[5.0, -1.0], &around, 0.0, 4.0), Err(DistanceError::Unsorted { .. })));
            // Spikes exactly ON the window's ends bracket it.
            assert!(f(&[0.0, 4.0], &[0.0, 1.0, 4.0], 0.0, 4.0).is_ok());
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

    /// The published pair ceiling is four hundred million. Pinned as a compile-time assertion
    /// because the one fixture that reaches it asks for nine hundred million pairs, which is past
    /// a ceiling a decade lower just as surely — the constant could be cut tenfold and every
    /// refusal in this module would still be raised.
    #[test]
    fn the_published_pair_ceiling_is_four_hundred_million() {
        const { assert!(MAX_PAIRS == 400_000_000) };
    }

    /// The RENDERED text of every refusal this module raises, and the pair COUNT the too-large one
    /// carries. Pinned because every other refusal test here destructures the variant and matches
    /// on `what` alone: a message that says "fewer" where the guard means "more", one that prints
    /// its range backwards, and a count reported as zero are all fluent, and none of them could
    /// fail a `matches!`.
    #[test]
    fn a_refusal_renders_what_was_wrong_and_carries_the_count_it_refused() {
        assert_eq!(victor_purpura(&[], &[], -1.0).unwrap_err().to_string(), "q = -1 is outside [0, inf]");
        assert_eq!(victor_purpura(&[0.2, 0.1], &[], 1.0).unwrap_err().to_string(), "a is out of time order at 1");
        assert_eq!(victor_purpura(&[], &[0.1, f64::NAN], 1.0).unwrap_err().to_string(), "b is not finite at 1");
        assert_eq!(vector_strength(&[], 1.0).unwrap_err().to_string(), "spikes is empty");
        // 15 000 spikes against 15 000 is 30 000², and the refusal names that number.
        let long = vec![0.0; 15_000];
        assert_eq!(victor_purpura(&long, &long, 1.0), Err(DistanceError::TooLarge { pairs: 900_000_000 }));
        assert_eq!(
            van_rossum_squared(&long, &long, 1.0).unwrap_err().to_string(),
            "900000000 spike pairs is more than MAX_PAIRS"
        );
        // …and the window ceiling prints MAX_PAIRS itself.
        assert_eq!(
            fano_factor(&[0.1], 0.0, 1.0, 0.6).unwrap_err().to_string(),
            "windows = 1 is outside [2, 400000000]"
        );
    }

    /// A time constant, a quadrature step, a tail, a period and a window must all be FINITE as
    /// well as positive, and the refusal names the smallest positive double as its floor. Pinned
    /// because every existing fixture for these guards passes `0.0`, which `v > 0.0` rejects on
    /// its own — and infinity is not absurd here but plausible: with `τ = ∞` every kernel term is
    /// `e⁰ = 1` and the van Rossum distance becomes `½ (n − m)²`, a perfectly ordinary number.
    #[test]
    fn a_positive_parameter_must_be_finite_and_the_refusal_names_its_floor() {
        let Err(DistanceError::OutOfRange { what, value, low, high }) = van_rossum_squared(&[0.0], &[1.0], f64::INFINITY) else {
            panic!("an infinite time constant was accepted");
        };
        assert_eq!((what, value), ("tau", f64::INFINITY));
        assert_eq!((low, high), (f64::MIN_POSITIVE, f64::INFINITY));
        assert!(matches!(
            van_rossum_by_quadrature(&[0.0], &[1.0], 1.0, f64::INFINITY, 5.0),
            Err(DistanceError::OutOfRange { what: "dt", .. })
        ));
        assert!(matches!(
            van_rossum_by_quadrature(&[0.0], &[1.0], 1.0, 1e-3, f64::INFINITY),
            Err(DistanceError::OutOfRange { what: "tail", .. })
        ));
        assert!(matches!(vector_strength(&[0.1], f64::INFINITY), Err(DistanceError::OutOfRange { what: "period", .. })));
        assert!(matches!(fano_factor(&[0.1], 0.0, 1.0, f64::INFINITY), Err(DistanceError::OutOfRange { what: "window", .. })));
        // The parameter-free formulas refuse an infinite period or window too, rather than
        // answering `e^{-0} = 1` and `∞ − ∞`.
        assert_eq!(jittered_vector_strength(1e-3, f64::INFINITY), None);
        assert_eq!(periodic_fano(1.0, f64::INFINITY), None);
        assert_eq!(periodic_fano(f64::INFINITY, 1.0), None);
    }

    /// The squared van Rossum distance of two trains a tenth of a femtosecond apart is zero, not a
    /// negative number. Pinned because it is the rounding floor of the closed form that is being
    /// held down, and that floor has to be REACHED: 38 spikes give three sums of 1444 terms each
    /// of order one, so the difference carries about `38² · ε ≈ 3e-13` of rounding, while the true
    /// value, `n (δ/τ)²` with `δ = 1e-16 s` and `τ = 0.1 s`, is about `4e-30`. Measured, the
    /// unclamped closed form returns −6.8e-13 here. The existing near-identity assertion is
    /// `< 1e-12`, which a negative number satisfies.
    #[test]
    fn a_squared_norm_of_two_all_but_identical_trains_is_zero_and_not_negative() {
        let a: Vec<f64> = (0..38).map(|k| 0.001 * f64::from(k)).collect();
        let b: Vec<f64> = a.iter().map(|t| t + 1e-16).collect();
        assert_eq!(van_rossum_squared(&a, &b, 0.1).unwrap(), 0.0);
        assert_eq!(van_rossum(&a, &b, 0.1).unwrap(), 0.0, "and its square root is a number");
    }

    /// The interval index is always one a train can be READ at: `i + 1` indexes the train, so the
    /// LAST spike is never an interval start. Pinned because the clamp is unreachable through the
    /// public functions — they refuse a window the train does not bracket, so the instant is
    /// always strictly inside — and the two callers index `t[i + 1]` immediately, which is a panic
    /// rather than a wrong answer when the clamp is loosened by one.
    #[test]
    fn an_interval_index_never_names_the_last_spike_as_a_start() {
        let three = [0.0, 1.0, 2.0];
        for x in [-1.0, 0.0, 0.5, 1.0, 1.9, 2.0, 5.0] {
            let i = interval_of(&three, x);
            assert!(i + 1 < three.len(), "x = {x} was answered with interval {i} of a train of three");
        }
        assert_eq!(interval_of(&three, 0.5), 0);
        assert_eq!(interval_of(&three, 1.0), 1, "an instant AT a spike starts that spike's interval");
        assert_eq!(interval_of(&three, 2.0), 1, "and the last spike is clamped back to the last interval");
    }

    /// A spike with no later neighbour in the other train is INFINITELY far from one, not on top of
    /// one. Pinned because both trains in every existing fixture end on the same spike — the
    /// referee quadrature pads both with `-0.7` and `6.4`, the clocks run to the same time — so
    /// the "no spike after this one" arm of `nearest_gap` was never taken with a finite answer
    /// riding on it.
    ///
    /// By hand, on `a = {0, 10}` against `b = {0, 2, 4}` over `[0, 4]`: `a`'s closing spike at 10
    /// is past `b`'s last, so its gap is the 6 back to `b = 4`. Both segments then have the same
    /// profile `S(t) = (0.6t · 2 + t · 10)/(2 · 6²) = 11.2 t/72`, whose average over `[0, 4]` is
    /// `2 · 11.2/72 = 14/45`. With that spike read as coincident the first arm is zero throughout
    /// and the answer would be `20/72`.
    #[test]
    fn a_spike_with_no_later_neighbour_is_not_coincident_with_one() {
        let got = spike_distance(&[0.0, 10.0], &[0.0, 2.0, 4.0], 0.0, 4.0).unwrap();
        assert!((got - 14.0 / 45.0).abs() < 1e-15, "{got}");
        assert_eq!(spike_distance(&[0.0, 2.0, 4.0], &[0.0, 10.0], 0.0, 4.0).unwrap(), got, "and it is symmetric");
    }

    /// The Fano factor's windows are laid out from `start`, and a non-finite `end` is reported as a
    /// non-finite INTERVAL rather than as an impossible window count. Pinned because every window
    /// fixture in this module starts at zero, where `t − start` and `t` are the same number, and
    /// because `(∞ − 0)/w` floors to infinity, which the count guard refuses on its own with a
    /// different name.
    #[test]
    fn the_windows_are_laid_out_from_the_start_of_the_interval() {
        // Four spikes in [10, 11) at 0.5 s windows: one in the first, three in the second, so the
        // mean is 2, the population variance 1, and the factor a half.
        assert_eq!(fano_factor(&[10.1, 10.6, 10.7, 10.8], 10.0, 11.0, 0.5).unwrap(), 0.5);
        // Counted from zero instead, all four would fall past the second window and none would be
        // counted at all, which is the one state this function cannot answer in.
        //
        // An interval with a non-finite end is a non-finite INTERVAL, not an impossible count.
        assert!(matches!(fano_factor(&[0.1], 0.0, f64::INFINITY, 0.1), Err(DistanceError::NonFinite { what: "interval", .. })));
        assert!(matches!(fano_factor(&[0.1], 0.0, f64::NAN, 0.1), Err(DistanceError::NonFinite { what: "interval", .. })));
    }
}
