//! Synchrony and order in spike trains: which spikes coincide, which train leads, and how the
//! classical histograms of an experiment are formed — each checked against a closed form, and the
//! two event-based measures against the reference implementation their authors maintain.
//!
//! # What is here
//!
//! - **SPIKE-synchronization** (Kreuz, Mulansky and Bozanic, *SPIKY: a graphical user interface for
//!   monitoring spike train synchrony*, Journal of Neurophysiology 113:3432–3445, 2015). A spike is
//!   COINCIDENT when its nearest spike in the other train is closer than a window set by the trains
//!   themselves: half the shortest of the four interspike intervals around the two spikes. No
//!   parameter to choose. The measure is the fraction of all spikes that are coincident — 1 for
//!   trains that fire together spike for spike, 0 for trains that never do — and for a population
//!   the fraction over every pair: [`spike_sync`], [`spike_sync_multi`].
//! - **SPIKE-order and the synfire indicator** (Kreuz, Satuvuori, Pofahl, Mulansky, *Leaders and
//!   followers: quantifying consistency in spatio-temporal propagation patterns*, New Journal of
//!   Physics 19:043028, 2017). Within each coincident pair, the earlier spike LEADS (+1) and the later
//!   follows (−1); the synfire indicator `F` is the net leading of lower-numbered trains over
//!   higher-numbered ones, per spike: +1 for a volley that always runs through the trains in the
//!   order given, −1 for one that always runs backwards — [`spike_order`], [`synfire_indicator`].
//! - **The PSTH** — spikes per bin per trial per second — [`psth`].
//! - **The cross-correlogram** and what it would be for independent trains:
//!   [`cross_correlogram`] against [`correlogram_expectation`], the triangle
//!   `n_a n_b (T − |s|)/T²` integrated over each bin.
//! - **The Schreiber correlation** (Schreiber, Fellous, Whitmer, Tiesinga and Sejnowski, *A new
//!   correlation-based measure of spike timing reliability*, Neurocomputing 52–54:925–931, 2003):
//!   the cosine between two trains filtered by a Gaussian of width `σ`, which in the spike times is
//!   the closed form `Σ e^{−(aᵢ − bⱼ)²/4σ²}` over the square root of the two self-terms —
//!   [`schreiber`], and its mean over pairs of trials, [`schreiber_reliability`].
//! - **The spike time tiling coefficient** (Cutts and Eglen, *Detecting pairwise correlations in
//!   spike trains: an objective comparison of methods and application to the study of retinal
//!   waves*, Journal of Neuroscience 34:14288–14303, 2014): `T_A` is the fraction of the recording
//!   within `dt` of a spike of A, `P_A` the fraction of A's spikes within `dt` of a spike of B, and
//!   `STTC = ½[(P_A − T_B)/(1 − P_A T_B) + (P_B − T_A)/(1 − P_B T_A)]` — 1 for trains that tile each
//!   other, 0 in expectation for independent trains whatever their rates, negative for trains that
//!   avoid each other: [`sttc`], [`tiled_fraction`]. Checked against the authors' own C
//!   (`spike_time_tiling_coefficient.c`, github.com/CCutts, commit `5f18868`) on 3,000 random pairs,
//!   to 2.8 × 10⁻¹³ wherever that code's answer is defined. The paper itself was not read here; the
//!   definition is the authors' code's.
//!
//! # Against the reference, and three things the reference does
//!
//! The SPIKE measures are maintained by their authors in `PySpike` (Mulansky and Kreuz, *`PySpike` — A
//! Python library for analyzing spike train synchrony*, `SoftwareX` 5:183–189, 2016). The tests hold
//! `PySpike` 0.9.0's own output for random trains: SPIKE-synchronization and the synfire indicator,
//! bivariate and for populations, agree to the bit — both are ratios of integer counts. The window
//! edges follow `PySpike`'s convention: a spike with no neighbour on one side takes the length of the
//! whole window as that interval. And [`crate::distance`]'s ISI- and SPIKE-distances were checked
//! against the same library, on 400 random pairs of trains while this module was written and on the
//! twenty its tests now hold: they agree to `2.2 × 10⁻¹⁶`. `tools/pyspike_reference.py` regenerates
//! both tables.
//!
//! Reading `PySpike`'s source to do that turned up three behaviours this module does not share, each
//! reproduced by the tests from `PySpike`'s own output:
//!
//! ⚠ **Its `psth` changes the bin width it was given.** It takes `int((t_end − t_start)/bin_size)`
//! bins and spreads them over the window, and `int` truncates: a 0.3 s window binned at 0.1 s has
//! `0.3/0.1 = 2.9999999999999996`, so it returns TWO bins of 150 ms; a 0.7 s window returns six of
//! 116.7 ms. [`psth`] takes the number of bins from the caller and never recomputes the width.
//!
//! ⚠ **Its `max_tau` does not bound the coincidence window**, although its documentation says it is
//! the "maximum coincidence window size". It replaces only the missing interval at a window edge;
//! between two interior spikes the window is still half their neighbouring intervals. Three pairs of
//! spikes 20 ms apart with `max_tau = 1 ms` come out one-third synchronous: the middle pair, whose
//! window is 200 ms. [`spike_sync_within`] bounds every window.
//!
//! ⚠ **Its population synfire indicator counts every pair of silent trains as one perfectly ordered
//! spike.** The bivariate routine returns `(1, 1)` for two empty trains — "spike sync = 1 by
//! definition" — and the population sum adds it in. Two trains that always fire in reverse order
//! (`F = −1`) plus two silent trains give `F = −0.28`, where the ratio the paper defines is `−8/24`.
//! [`synfire_indicator`] adds nothing for a pair with no spikes, and refuses a population with none
//! at all rather than calling it synchronous.
//!
//! # Where the STTC's reference implementations disagree with each other
//!
//! ⚠ **A train whose tiles cover the whole recording makes half the formula `0/0`.** The authors'
//! C computes `T` by subtracting overlaps from `2·N·dt`, lands a hair either side of 1, and so
//! returns NaN or a number decided by rounding: of the 205 saturated pairs in the 3,000 checked,
//! 12 came back NaN and 193 as numbers from 0.19 to 1. `Elephant` sets that half-index to 1 by
//! fiat. [`sttc`] decides coverage by the geometry (the first tile reaches the start, the last the
//! end, no gap wider than `2·dt`), also treats a gap too narrow for the sum to see as coverage, and
//! refuses both with [`SyncError::Saturated`].
//!
//! ⚠ **The authors' C misses one edge of a lone spike.** A single spike within `dt` of BOTH ends of
//! the window gets only the start correction (an `if … else if`): a spike at 4 in `[0, 8]` with
//! `dt` = 5 gets `T` = 1.125. [`tiled_fraction`] clips the tile at both ends and gives 1.
//!
//! ⚠ **`Elephant` widens the window with the spike time.** Its `P` uses
//! `numpy.isclose(a, b, atol=dt)`, whose default `rtol = 1e-5` adds `10⁻⁵·|b|` to `dt`: at
//! `dt` = 5 ms a spike 11 ms from its partner counts as tiled 600 s into a recording (read in
//! `elephant/spike_train_correlation.py`, commit `32f1b56`). [`sttc`] uses the authors' test,
//! `|aᵢ − bⱼ| ≤ dt`, and nothing else.
//!
//! # Trains
//!
//! Every train is strictly increasing, finite, and inside the window it is measured over. A repeated
//! spike time is refused rather than merged — `PySpike` 0.9.0 silently drops duplicates, drops every
//! spike more than 10⁻⁶ outside the window and KEEPS one less than 10⁻⁶ outside it, whatever the time
//! unit (`reconcile_spike_trains`, `Eps = 1e-6`; for a population the window runs from the earliest
//! start to the latest end) — because a duplicate is a recording artefact the caller should see.
//! Earlier releases said it dropped the spikes within 10⁻⁶; it keeps them.

use core::fmt;

/// Why a synchrony measure could not be computed.
#[derive(Debug, Clone, PartialEq)]
pub enum SyncError {
    /// A window that is not a finite interval of positive length.
    Window {
        /// Its start.
        start: f64,
        /// Its end.
        end: f64,
    },
    /// A spike time that is not finite.
    NonFinite {
        /// Which train.
        train: usize,
        /// Which spike.
        index: usize,
        /// Its value.
        value: f64,
    },
    /// A spike at or before the one before it.
    NotIncreasing {
        /// Which train.
        train: usize,
        /// The later spike's index.
        index: usize,
    },
    /// A spike outside the window.
    Outside {
        /// Which train.
        train: usize,
        /// Which spike.
        index: usize,
        /// Its time.
        value: f64,
    },
    /// No spike in any train, so the ratio has nothing to count.
    NoSpikes,
    /// A population measure given fewer than two trains.
    TooFewTrains {
        /// How many it was given.
        got: usize,
    },
    /// A width or bound that must be finite and positive.
    NotPositive {
        /// Which parameter.
        what: &'static str,
        /// Its value.
        value: f64,
    },
    /// A histogram with no bins.
    NoBins,
    /// A measure of a pair given a train with no spikes.
    EmptyTrain {
        /// Which train.
        train: usize,
    },
    /// A train within `dt` of a spike everywhere in the window, so its tiling covers the whole
    /// recording and leaves the other train nothing to be compared against.
    Saturated {
        /// Which train.
        train: usize,
        /// The tiling half-width.
        dt: f64,
    },
}

impl fmt::Display for SyncError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Window { start, end } => write!(f, "[{start}, {end}] is not a window of positive length"),
            Self::NonFinite { train, index, value } => write!(f, "train {train}, spike {index} is {value}"),
            Self::NotIncreasing { train, index } => write!(f, "train {train}, spike {index} is not after the spike before it"),
            Self::Outside { train, index, value } => write!(f, "train {train}, spike {index} at {value} is outside the window"),
            Self::NoSpikes => write!(f, "no train has a spike, so there is nothing to count"),
            Self::TooFewTrains { got } => write!(f, "a population measure needs two trains or more, not {got}"),
            Self::NotPositive { what, value } => write!(f, "{what} = {value} must be finite and positive"),
            Self::NoBins => write!(f, "a histogram needs at least one bin"),
            Self::EmptyTrain { train } => write!(f, "train {train} has no spikes, so there is nothing to tile"),
            Self::Saturated { train, dt } => {
                write!(f, "train {train} is within dt = {dt} of a spike everywhere in the window, so its tiling covers the whole recording")
            }
        }
    }
}

impl std::error::Error for SyncError {}

fn positive(what: &'static str, value: f64) -> Result<f64, SyncError> {
    if value.is_finite() && value > 0.0 { Ok(value) } else { Err(SyncError::NotPositive { what, value }) }
}

fn window(start: f64, end: f64) -> Result<f64, SyncError> {
    if start.is_finite() && end.is_finite() && end > start { Ok(end - start) } else { Err(SyncError::Window { start, end }) }
}

/// Finite and strictly increasing.
fn train(k: usize, t: &[f64]) -> Result<(), SyncError> {
    for (i, &x) in t.iter().enumerate() {
        if !x.is_finite() {
            return Err(SyncError::NonFinite { train: k, index: i, value: x });
        }
        if i > 0 && !(x > t[i - 1]) {
            return Err(SyncError::NotIncreasing { train: k, index: i });
        }
    }
    Ok(())
}

/// Finite, strictly increasing and inside `[start, end]`.
fn inside(k: usize, t: &[f64], start: f64, end: f64) -> Result<(), SyncError> {
    train(k, t)?;
    for (i, &x) in t.iter().enumerate() {
        if x < start || x > end {
            return Err(SyncError::Outside { train: k, index: i, value: x });
        }
    }
    Ok(())
}

/// Half the shortest of the four interspike intervals around `a[i]` and `b[j]`, a missing neighbour
/// counting as the window's length.
fn tau(a: &[f64], i: usize, b: &[f64], j: usize, length: f64) -> f64 {
    let around = |t: &[f64], k: usize| {
        let before = if k > 0 { t[k] - t[k - 1] } else { length };
        let after = if k + 1 < t.len() { t[k + 1] - t[k] } else { length };
        before.min(after)
    };
    around(a, i).min(around(b, j)) / 2.0
}

/// The spike of `b` that `a[i]` coincides with, if any.
///
/// Only the two spikes of `b` either side of `a[i]` can qualify, and at most one does: a window is
/// at most half the interval between them, so a spike cannot be inside the windows of both.
fn partner(a: &[f64], i: usize, b: &[f64], length: f64, cap: f64) -> Option<usize> {
    let k = b.partition_point(|&t| t < a[i]);
    let before = k.checked_sub(1);
    let after = (k < b.len()).then_some(k);
    [before, after].into_iter().flatten().find(|&j| (a[i] - b[j]).abs() < tau(a, i, b, j, length).min(cap))
}

/// Coincidences of `a` with `b` and of `b` with `a`, as counts.
fn coincidences(a: &[f64], b: &[f64], length: f64, cap: f64) -> (usize, usize) {
    let ca = (0..a.len()).filter(|&i| partner(a, i, b, length, cap).is_some()).count();
    let cb = (0..b.len()).filter(|&j| partner(b, j, a, length, cap).is_some()).count();
    (ca, cb)
}

/// SPIKE-synchronization of two trains over `[start, end]`: the fraction of their spikes that are
/// coincident.
///
/// # Errors
///
/// [`SyncError::Window`], [`SyncError::NonFinite`], [`SyncError::NotIncreasing`] or
/// [`SyncError::Outside`] for a bad window or train; [`SyncError::NoSpikes`] when both are empty.
pub fn spike_sync(a: &[f64], b: &[f64], start: f64, end: f64) -> Result<f64, SyncError> {
    spike_sync_multi(&[a, b], start, end)
}

/// [`spike_sync`] with every coincidence window also bounded by `max_tau` — what `PySpike` documents its
/// `max_tau` as doing.
///
/// # Errors
///
/// As [`spike_sync`], and [`SyncError::NotPositive`] for a `max_tau` that is not.
pub fn spike_sync_within(a: &[f64], b: &[f64], start: f64, end: f64, max_tau: f64) -> Result<f64, SyncError> {
    positive("max_tau", max_tau)?;
    population(&[a, b], start, end, max_tau)
}

/// SPIKE-synchronization of a population: coincidences summed over every pair of trains, over the
/// spikes summed over every pair — equal to the mean over all spikes of the fraction of the OTHER
/// trains each one coincides with.
///
/// # Errors
///
/// As [`spike_sync`], and [`SyncError::TooFewTrains`] for fewer than two trains.
pub fn spike_sync_multi(trains: &[&[f64]], start: f64, end: f64) -> Result<f64, SyncError> {
    population(trains, start, end, f64::INFINITY)
}

fn population(trains: &[&[f64]], start: f64, end: f64, cap: f64) -> Result<f64, SyncError> {
    let length = window(start, end)?;
    if trains.len() < 2 {
        return Err(SyncError::TooFewTrains { got: trains.len() });
    }
    for (k, t) in trains.iter().enumerate() {
        inside(k, t, start, end)?;
    }
    let (mut hits, mut spikes) = (0usize, 0usize);
    for (m, a) in trains.iter().enumerate() {
        for b in &trains[m + 1..] {
            let (ca, cb) = coincidences(a, b, length, cap);
            hits += ca + cb;
            spikes += a.len() + b.len();
        }
    }
    if spikes == 0 {
        return Err(SyncError::NoSpikes);
    }
    Ok(hits as f64 / spikes as f64)
}

/// SPIKE-order of two trains: for every spike, +1 if it leads the spike it coincides with, −1 if it
/// follows it, and 0 if it coincides with none or with a spike at exactly its own time.
///
/// # Errors
///
/// As [`spike_sync`], except that two empty trains give two empty vectors.
pub fn spike_order(a: &[f64], b: &[f64], start: f64, end: f64) -> Result<(Vec<i8>, Vec<i8>), SyncError> {
    let length = window(start, end)?;
    inside(0, a, start, end)?;
    inside(1, b, start, end)?;
    Ok((order(a, b, length), order(b, a, length)))
}

/// Each spike of `x`: +1 leading its partner in `y`, −1 following it, 0 for none or a tie.
fn order(x: &[f64], y: &[f64], length: f64) -> Vec<i8> {
    (0..x.len())
        .map(|i| match partner(x, i, y, length, f64::INFINITY) {
            Some(j) if x[i] < y[j] => 1,
            Some(j) if x[i] > y[j] => -1,
            _ => 0,
        })
        .collect()
}

/// The synfire indicator `F` of a population, in the order the trains are given: over every pair
/// `m < n`, the spikes of `m` that lead minus those that follow, doubled — each coincident pair has
/// two spikes — and divided by all the spikes of every pair.
///
/// +1 when every spike of every train is coincident with, and later than, its partner in every
/// earlier train; −1 for the reverse; near 0 for coincidences with no consistent order, or few
/// coincidences at all.
///
/// # Errors
///
/// As [`spike_sync_multi`]. A pair of silent trains contributes nothing — see the module notes on
/// `PySpike` — and a population with no spike at all is [`SyncError::NoSpikes`].
pub fn synfire_indicator(trains: &[&[f64]], start: f64, end: f64) -> Result<f64, SyncError> {
    let length = window(start, end)?;
    if trains.len() < 2 {
        return Err(SyncError::TooFewTrains { got: trains.len() });
    }
    for (k, t) in trains.iter().enumerate() {
        inside(k, t, start, end)?;
    }
    let (mut net, mut spikes) = (0i64, 0usize);
    for (m, a) in trains.iter().enumerate() {
        for b in &trains[m + 1..] {
            net += 2 * order(a, b, length).iter().map(|&d| i64::from(d)).sum::<i64>();
            spikes += a.len() + b.len();
        }
    }
    if spikes == 0 {
        return Err(SyncError::NoSpikes);
    }
    Ok(net as f64 / spikes as f64)
}

/// A peri-stimulus time histogram.
#[derive(Debug, Clone, PartialEq)]
pub struct Psth {
    /// Spikes per second per trial in each bin.
    pub rate: Vec<f64>,
    /// Spikes that fell in no bin.
    pub outside: usize,
}

/// The PSTH of `trials` over `n_bins` bins of width `bin` from `start`.
///
/// Bin `k` is `[start + k·bin, start + (k + 1)·bin)`, with its edges computed exactly that way and a
/// spike placed by comparing it with them. That matters more than it looks: `floor((t − start)/bin)`
/// puts a spike at 0.3 s in bin 2 of a 0.1 s histogram, because `0.3/0.1` rounds to
/// `2.9999999999999996`, while the edge `3 × 0.1` is `0.30000000000000004`, above the spike. Here
/// the two agree by construction: the spike IS below that edge, so it IS in bin 2.
///
/// # Errors
///
/// [`SyncError::NotPositive`] for a `bin` that is not; [`SyncError::NoBins`] for zero bins;
/// [`SyncError::TooFewTrains`] for no trials; [`SyncError::NonFinite`] or
/// [`SyncError::NotIncreasing`] for a bad train; [`SyncError::Window`] for a non-finite `start`.
pub fn psth(trials: &[&[f64]], start: f64, bin: f64, n_bins: usize) -> Result<Psth, SyncError> {
    positive("bin", bin)?;
    if n_bins == 0 {
        return Err(SyncError::NoBins);
    }
    if trials.is_empty() {
        return Err(SyncError::TooFewTrains { got: 0 });
    }
    if !start.is_finite() {
        return Err(SyncError::Window { start, end: start });
    }
    let edge = |k: usize| start + k as f64 * bin;
    let mut counts = vec![0usize; n_bins];
    let mut outside = 0;
    for (k, t) in trials.iter().enumerate() {
        train(k, t)?;
        for &x in *t {
            if x < start || x >= edge(n_bins) {
                outside += 1;
                continue;
            }
            // Division's guess, corrected against the edges. The corrections stay inside the bins, so
            // no edit of the range check above can turn them into an unbounded walk.
            let mut b = (((x - start) / bin) as usize).min(n_bins - 1);
            while b > 0 && edge(b) > x {
                b -= 1;
            }
            while b + 1 < n_bins && edge(b + 1) <= x {
                b += 1;
            }
            counts[b] += 1;
        }
    }
    let per = trials.len() as f64 * bin;
    Ok(Psth { rate: counts.iter().map(|&c| c as f64 / per).collect(), outside })
}

/// The cross-correlogram of `b` against `a`: how many pairs `(aᵢ, bⱼ)` have `bⱼ − aᵢ` in each of
/// `n_bins` bins of width `bin` starting at lag `lo`, bin `k` being `[lo + k·bin, lo + (k + 1)·bin)`.
/// A peak at a positive lag means `b` tends to fire after `a`.
///
/// # Errors
///
/// As [`psth`], with [`SyncError::Window`] for a non-finite `lo`.
pub fn cross_correlogram(a: &[f64], b: &[f64], lo: f64, bin: f64, n_bins: usize) -> Result<Vec<u64>, SyncError> {
    positive("bin", bin)?;
    if n_bins == 0 {
        return Err(SyncError::NoBins);
    }
    if !lo.is_finite() {
        return Err(SyncError::Window { start: lo, end: lo });
    }
    train(0, a)?;
    train(1, b)?;
    let edge = |k: usize| lo + k as f64 * bin;
    let hi = edge(n_bins);
    let mut counts = vec![0u64; n_bins];
    for &x in a {
        let first = b.partition_point(|&y| y - x < lo);
        for &y in &b[first..] {
            let lag = y - x;
            if lag >= hi {
                break;
            }
            let mut k = (((lag - lo) / bin) as usize).min(n_bins - 1);
            while k > 0 && edge(k) > lag {
                k -= 1;
            }
            while k + 1 < n_bins && edge(k + 1) <= lag {
                k += 1;
            }
            counts[k] += 1;
        }
    }
    Ok(counts)
}

/// The expected number of pairs with lag in `[lo, hi]` for two INDEPENDENT trains of `n_a` and `n_b`
/// spikes, each spike placed uniformly on a window of length `duration`: `n_a n_b/T²` times the
/// integral of the triangle `T − |s|` over `[lo, hi] ∩ [−T, T]`.
///
/// The triangle is the fraction of the window where a lag of `s` fits: two uniform times differ by
/// `s` with density `(T − |s|)/T²`. A flat expectation — the one a correlogram is usually compared
/// with — is its tip, right only for lags much shorter than the window; the tests show the error
/// of the flat line at long lags.
///
/// # Errors
///
/// [`SyncError::NotPositive`] for a `duration` that is not; [`SyncError::Window`] unless
/// `lo ≤ hi`, both finite.
pub fn correlogram_expectation(n_a: usize, n_b: usize, duration: f64, lo: f64, hi: f64) -> Result<f64, SyncError> {
    let t = positive("duration", duration)?;
    if !(lo.is_finite() && hi.is_finite() && lo <= hi) {
        return Err(SyncError::Window { start: lo, end: hi });
    }
    // An antiderivative of T − |s| on [−T, T], clamped outside it.
    let area = |s: f64| {
        let s = s.clamp(-t, t);
        t * s - s * s.abs() / 2.0
    };
    Ok(n_a as f64 * n_b as f64 * (area(hi) - area(lo)) / (t * t))
}

/// The Schreiber correlation of two trains under a Gaussian filter of width `sigma`:
/// `⟨s_a, s_b⟩/(‖s_a‖ ‖s_b‖)` for the filtered traces, which in closed form is
/// `Σᵢⱼ e^{−(aᵢ − bⱼ)²/4σ²}` over `√(Σ e^{−(aᵢ − aₖ)²/4σ²} · Σ e^{−(bⱼ − bₗ)²/4σ²})` — the overlap of two
/// Gaussians of width `σ` being a Gaussian of width `σ√2`, whose normalising constant cancels.
///
/// # Errors
///
/// [`SyncError::NotPositive`] for a `sigma` that is not; [`SyncError::NoSpikes`] when either train is
/// empty, whose filtered trace has no length to divide by; [`SyncError::NonFinite`] or
/// [`SyncError::NotIncreasing`] for a bad train.
pub fn schreiber(a: &[f64], b: &[f64], sigma: f64) -> Result<f64, SyncError> {
    positive("sigma", sigma)?;
    train(0, a)?;
    train(1, b)?;
    if a.is_empty() || b.is_empty() {
        return Err(SyncError::NoSpikes);
    }
    let overlap = |x: &[f64], y: &[f64]| -> f64 {
        x.iter().map(|&p| y.iter().map(|&q| (-(p - q) * (p - q) / (4.0 * sigma * sigma)).exp()).sum::<f64>()).sum()
    };
    Ok(overlap(a, b) / (overlap(a, a) * overlap(b, b)).sqrt())
}

/// Schreiber et al.'s reliability `R_corr`: the mean of [`schreiber`] over every pair of trials.
///
/// # Errors
///
/// [`SyncError::TooFewTrains`] for fewer than two trials, and whatever [`schreiber`] returns for a
/// pair.
pub fn schreiber_reliability(trials: &[&[f64]], sigma: f64) -> Result<f64, SyncError> {
    if trials.len() < 2 {
        return Err(SyncError::TooFewTrains { got: trials.len() });
    }
    let mut sum = 0.0;
    let mut pairs = 0usize;
    for (m, a) in trials.iter().enumerate() {
        for b in &trials[m + 1..] {
            sum += schreiber(a, b, sigma)?;
            pairs += 1;
        }
    }
    Ok(sum / pairs as f64)
}

/// The length of `[start, end]` lying within `dt` of a spike of `t`: the union of the tiles
/// `[tᵢ − dt, tᵢ + dt]`, clipped to the window. `t` is increasing and inside the window.
fn tiled(t: &[f64], start: f64, end: f64, dt: f64) -> f64 {
    let mut covered = 0.0;
    let mut reach = start;
    // Never negative: the spikes are increasing and inside the window, so each tile's end reaches
    // at least as far as the last one's (`reach`) and at least as far as `start`.
    for &x in t {
        let lo = (x - dt).max(reach);
        let hi = (x + dt).min(end);
        covered += hi - lo;
        reach = reach.max(hi);
    }
    covered
}

/// Whether the tiles of `t` cover all of `[start, end]`, decided by the geometry — the first tile
/// reaches the start, the last the end, and no two neighbours are more than `2·dt` apart — rather
/// than by whether a sum of lengths rounds to the window's.
fn covers(t: &[f64], start: f64, end: f64, dt: f64) -> bool {
    match (t.first(), t.last()) {
        (Some(&first), Some(&last)) => first - dt <= start && last + dt >= end && t.windows(2).all(|w| w[1] - w[0] <= 2.0 * dt),
        _ => false,
    }
}

/// How many spikes of `a` have a spike of `b` within `dt`, by the authors' own test,
/// `|aᵢ − bⱼ| ≤ dt`. Both increasing; `b` is walked once.
fn near(a: &[f64], b: &[f64], dt: f64) -> usize {
    let (mut j, mut n) = (0usize, 0usize);
    for &x in a {
        while j < b.len() && x - b[j] > dt {
            j += 1;
        }
        if j < b.len() && (x - b[j]).abs() <= dt {
            n += 1;
        }
    }
    n
}

/// Cutts and Eglen's `T`: the fraction of `[start, end]` within `dt` of a spike of `t`, the tiles
/// merged where they overlap and clipped at the window's edges.
///
/// # Errors
///
/// [`SyncError::Window`], [`SyncError::NotPositive`] for `dt`, and a train that is not finite,
/// increasing and inside the window.
pub fn tiled_fraction(t: &[f64], start: f64, end: f64, dt: f64) -> Result<f64, SyncError> {
    let length = window(start, end)?;
    let dt = positive("dt", dt)?;
    inside(0, t, start, end)?;
    Ok(if covers(t, start, end, dt) { 1.0 } else { tiled(t, start, end, dt) / length })
}

/// The spike time tiling coefficient of Cutts and Eglen (2014):
///
/// ```text
/// STTC = ½ [ (P_A − T_B)/(1 − P_A T_B) + (P_B − T_A)/(1 − P_B T_A) ]
/// ```
///
/// where `T_A` is [`tiled_fraction`] of `a` and `P_A` the fraction of `a`'s spikes within `dt` of a
/// spike of `b`. 1 for trains that tile each other completely, 0 in expectation for independent
/// ones whatever their rates, negative for trains that avoid each other.
///
/// # Errors
///
/// As [`tiled_fraction`] for either train; [`SyncError::EmptyTrain`] for a train with no spikes;
/// [`SyncError::Saturated`] for a train whose tiling covers the whole window, where a half of the
/// formula is `0/0`.
pub fn sttc(a: &[f64], b: &[f64], start: f64, end: f64, dt: f64) -> Result<f64, SyncError> {
    let length = window(start, end)?;
    let dt = positive("dt", dt)?;
    inside(0, a, start, end)?;
    inside(1, b, start, end)?;
    for (k, t) in [a, b].into_iter().enumerate() {
        if t.is_empty() {
            return Err(SyncError::EmptyTrain { train: k });
        }
    }
    for (k, t) in [a, b].into_iter().enumerate() {
        if covers(t, start, end, dt) {
            return Err(SyncError::Saturated { train: k, dt });
        }
    }
    let t_a = tiled(a, start, end, dt) / length;
    let t_b = tiled(b, start, end, dt) / length;
    // A gap narrower than the rounding of the sum is a tiling of 1 to the arithmetic, and a
    // half of the formula is 0/0 there just the same.
    for (k, t) in [t_a, t_b].into_iter().enumerate() {
        if t >= 1.0 {
            return Err(SyncError::Saturated { train: k, dt });
        }
    }
    let p_a = near(a, b, dt) as f64 / a.len() as f64;
    let p_b = near(b, a, dt) as f64 / b.len() as f64;
    Ok(0.5 * (p_a - t_b) / (1.0 - p_a * t_b) + 0.5 * (p_b - t_a) / (1.0 - p_b * t_a))
}

#[cfg(test)]
mod tests {
    use super::{
        Psth, SyncError, correlogram_expectation, cross_correlogram, psth, schreiber, schreiber_reliability, spike_order,
        spike_sync, spike_sync_multi, spike_sync_within, sttc, synfire_indicator, tiled_fraction,
    };
    use crate::rng::Rng;

    // Two populations of six trains on [0, 1] and what PySpike 0.9.0 computes for them. `VOLLEY` is
    // one volley of twelve spikes run through the trains 3 ms apart, jittered by 4 ms, a fifth of the
    // spikes dropped and two random ones added to each train; `INDEPENDENT` is six independent
    // uniform trains. Pairs are in the order (0, 1), (0, 2), … (4, 5).
    const VOLLEY: [&[f64]; 6] = [
        &[0.0678, 0.0998, 0.3818, 0.4326, 0.5619, 0.6042, 0.6164, 0.6317, 0.6557, 0.7053, 0.7145, 0.8764, 0.8907],
        &[0.1026, 0.3797, 0.4348, 0.4852, 0.5578, 0.6215, 0.6406, 0.7309, 0.8304, 0.8686, 0.8911],
        &[0.0267, 0.0629, 0.1144, 0.1861, 0.3875, 0.4401, 0.5713, 0.6239, 0.6292, 0.6421, 0.7346, 0.8873, 0.9051],
        &[0.0768, 0.1082, 0.1855, 0.386, 0.3882, 0.4423, 0.4505, 0.5694, 0.6267, 0.711, 0.7383, 0.877],
        &[0.0794, 0.1123, 0.1944, 0.3801, 0.3968, 0.4486, 0.5736, 0.5837, 0.6376, 0.6452, 0.7402, 0.9054],
        &[0.077, 0.1189, 0.1918, 0.1967, 0.4032, 0.4439, 0.6314, 0.6543, 0.7239, 0.7441, 0.8913, 0.9095, 0.9781],
    ];
    const VOLLEY_SYNC: [f64; 15] = [0.5, 0.5384615384615384, 0.48, 0.32, 0.38461538461538464, 0.6666666666666666, 0.43478260869565216, 0.43478260869565216, 0.4166666666666667, 0.64, 0.64, 0.6153846153846154, 0.5833333333333334, 0.4, 0.56];
    const VOLLEY_ORDER: [f64; 15] = [0.16666666666666666, 0.07692307692307693, 0.16, 0.16, 0.07692307692307693, 0.5, 0.43478260869565216, 0.2608695652173913, 0.08333333333333333, 0.0, 0.32, 0.6153846153846154, 0.4166666666666667, 0.4, 0.24];
    const VOLLEY_SYNC_ALL: f64 = 0.5081081081081081;
    const VOLLEY_F: f64 = 0.2594594594594595;
    const INDEPENDENT: [&[f64]; 6] = [
        &[0.8008, 0.8577, 0.9085],
        &[0.4063],
        &[0.003, 0.2998, 0.3399, 0.4577, 0.6768, 0.7647, 0.8313, 0.9369, 0.9437, 0.9699],
        &[0.023, 0.0775, 0.5162, 0.6296, 0.682, 0.8366, 0.9725],
        &[0.6866],
        &[0.1023, 0.1764, 0.4095, 0.4321, 0.456, 0.4995, 0.6987, 0.7314, 0.8724, 0.9098, 0.9544],
    ];
    const INDEPENDENT_SYNC: [f64; 15] = [0.0, 0.0, 0.2, 0.0, 0.2857142857142857, 0.18181818181818182, 0.0, 1.0, 0.16666666666666666, 0.47058823529411764, 0.18181818181818182, 0.09523809523809523, 0.25, 0.3333333333333333, 0.16666666666666666];
    const INDEPENDENT_ORDER: [f64; 15] = [0.0, 0.0, -0.2, 0.0, 0.2857142857142857, 0.18181818181818182, 0.0, 1.0, 0.16666666666666666, 0.47058823529411764, 0.18181818181818182, -0.09523809523809523, 0.25, -0.1111111111111111, 0.16666666666666666];
    const INDEPENDENT_SYNC_ALL: f64 = 0.20606060606060606;
    const INDEPENDENT_F: f64 = 0.10909090909090909;
    const DIRECTIONALITY_0: &[i8] = &[0, 1, -1, 1, -1, 0, 1, 0, 0, 0, 0, 0, 1];
    const DIRECTIONALITY_1: &[i8] = &[-1, 1, -1, 0, 1, -1, 0, 0, 0, 0, -1];

    const PAIRS: [(usize, usize); 15] =
        [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5), (3, 4), (3, 5), (4, 5)];

    /// SPIKE-synchronization and the synfire indicator are `PySpike`'s, for every pair and for both
    /// populations — exactly, since both are ratios of integer counts — and so is every spike's
    /// order.
    #[test]
    fn every_measure_is_pyspikes_to_the_bit() {
        for (trains, sync, order, sync_all, f) in [
            (&VOLLEY, &VOLLEY_SYNC, &VOLLEY_ORDER, VOLLEY_SYNC_ALL, VOLLEY_F),
            (&INDEPENDENT, &INDEPENDENT_SYNC, &INDEPENDENT_ORDER, INDEPENDENT_SYNC_ALL, INDEPENDENT_F),
        ] {
            for (k, &(i, j)) in PAIRS.iter().enumerate() {
                assert_eq!(spike_sync(trains[i], trains[j], 0.0, 1.0), Ok(sync[k]), "sync ({i}, {j})");
                assert_eq!(synfire_indicator(&[trains[i], trains[j]], 0.0, 1.0), Ok(order[k]), "order ({i}, {j})");
            }
            assert_eq!(spike_sync_multi(trains, 0.0, 1.0), Ok(sync_all));
            assert_eq!(synfire_indicator(trains, 0.0, 1.0), Ok(f));
        }
        let (d0, d1) = spike_order(VOLLEY[0], VOLLEY[1], 0.0, 1.0).unwrap();
        assert_eq!((d0.as_slice(), d1.as_slice()), (DIRECTIONALITY_0, DIRECTIONALITY_1));
        const { assert!(VOLLEY_F > 0.2 && INDEPENDENT_SYNC_ALL < VOLLEY_SYNC_ALL, "the volley is ordered and synchronous") };
    }

    /// The window-edge convention, and what it implies: a spike with no neighbour on one side takes
    /// the whole window as that interval, so two lone spikes 280 ms apart in a one-second window are
    /// coincident — half the window is 500 ms — and `PySpike` says the same (its pair (1, 4) above).
    #[test]
    fn two_lone_spikes_are_coincident_within_half_the_window() {
        assert_eq!(spike_sync(&[0.4063], &[0.6866], 0.0, 1.0), Ok(1.0));
        assert_eq!(INDEPENDENT_SYNC[7], 1.0);
        // In binary fractions so the boundary is exact: 0.75 − 0.25 IS half the window, which is not
        // less than it. (0.7 − 0.2 would not do: it is 0.49999999999999994.)
        assert_eq!(spike_sync(&[0.25], &[0.75], 0.0, 1.0), Ok(0.0), "exactly half the window is not less than it");
        assert_eq!(spike_sync(&[0.25], &[0.75], 0.0, 1.0 + 1e-9), Ok(1.0));
    }

    /// Closed cases: identical trains are fully synchronous, trains that never come close are not at
    /// all, a tie is coincident and leads nobody, and the population value is the mean over spikes of
    /// the fraction of other trains each one coincides with.
    #[test]
    fn synchronization_in_the_cases_with_known_answers() {
        let a = [0.1, 0.3, 0.35, 0.8];
        assert_eq!(spike_sync(&a, &a, 0.0, 1.0), Ok(1.0));
        assert_eq!(spike_order(&a, &a, 0.0, 1.0), Ok((vec![0; 4], vec![0; 4])), "ties lead nobody");
        // Every time below is a binary fraction, so each comparison with a window is exact — decimal
        // times put `0.3 − 0.2 = 0.09999999999999998` inside a window of `0.09999999999999999`.
        //
        // Spikes halfway between the other train's: exactly at the edge of a window of half an
        // interval, which is not inside it.
        assert_eq!(spike_sync(&[0.125, 0.375, 0.625], &[0.25, 0.5, 0.75], 0.0, 1.0), Ok(0.0));
        // A spike of `b` 1/64 after one of `a`, whose next spike is 1/16 later: the window is 1/32.
        let a2 = [0.25, 0.3125];
        assert_eq!(spike_sync(&a2, &[0.265625], 0.0, 1.0), Ok(2.0 / 3.0));
        assert_eq!(spike_order(&a2, &[0.265625], 0.0, 1.0), Ok((vec![1, 0], vec![-1])));
        assert_eq!(spike_sync(&a2, &[0.28125], 0.0, 1.0), Ok(0.0), "1/32 after, and 1/32 before the next: on both edges");
        // Population: spike by spike, the fraction of the other two trains it coincides with. `x₀`
        // pairs with `y₀` and `x₁` with `z₀`; nothing else coincides.
        let (x, y, z): (&[f64], &[f64], &[f64]) = (&[0.25, 0.625], &[0.2578125, 0.5, 0.875], &[0.6328125, 0.75]);
        let per_spike = [0.5, 0.5, 0.5, 0.0, 0.0, 0.5, 0.0];
        let mean = per_spike.iter().sum::<f64>() / 7.0;
        assert!((spike_sync_multi(&[x, y, z], 0.0, 1.0).unwrap() - mean).abs() < 1e-15);
        assert_eq!(spike_sync(x, y, 0.0, 1.0), Ok(0.4));
        assert_eq!(spike_sync(x, z, 0.0, 1.0), Ok(0.5));
        assert_eq!(spike_sync(y, z, 0.0, 1.0), Ok(0.0));
    }

    /// A volley that runs through the trains in the order given has `F = 1`; run backwards, `−1`;
    /// with the trains listed in a scrambled order, in between.
    #[test]
    fn a_perfect_volley_has_synfire_indicator_one() {
        let trains: Vec<Vec<f64>> = (0..5).map(|k| (0..8).map(|v| 0.1 * f64::from(v) + 0.05 + 0.004 * f64::from(k)).collect()).collect();
        let forward: Vec<&[f64]> = trains.iter().map(Vec::as_slice).collect();
        let backward: Vec<&[f64]> = forward.iter().rev().copied().collect();
        assert_eq!(synfire_indicator(&forward, 0.0, 1.0), Ok(1.0));
        assert_eq!(synfire_indicator(&backward, 0.0, 1.0), Ok(-1.0));
        assert_eq!(spike_sync_multi(&forward, 0.0, 1.0), Ok(1.0));
        let scrambled = [forward[2], forward[0], forward[4], forward[1], forward[3]];
        // Pairs in order: (2,0) −, (2,4) +, (2,1) −, (2,3) +, (0,4) +, (0,1) +, (0,3) +, (4,1) −, (4,3) −, (1,3) +.
        assert_eq!(synfire_indicator(&scrambled, 0.0, 1.0), Ok(0.2));
    }

    /// What `PySpike` 0.9.0 does and this module does not, each with `PySpike`'s own numbers.
    ///
    /// Its `psth([train on (0, 0.3)], 0.1)` returned edges `[0, 0.15, 0.3]` and counts `[2, 1]`: two
    /// bins of 150 ms for a request of 100 ms. Its `spike_sync(A, B, max_tau = 0.001)` for three pairs
    /// of spikes 20 ms apart returned `1/3`. Its `spike_train_order([B, A, C, D])` with `B` always
    /// following `A` and `C`, `D` silent returned `−0.28` — the phantom `(1, 1)` of the silent pair in
    /// `(−8 + 1)/(24 + 1)`.
    #[test]
    fn three_things_pyspike_does_that_this_module_does_not() {
        let p = psth(&[&[0.05, 0.12, 0.25]], 0.0, 0.1, 3).unwrap();
        assert_eq!(p, Psth { rate: vec![10.0, 10.0, 10.0], outside: 0 }, "three bins of 100 ms, as asked");
        let (a, b) = ([0.1, 0.5, 0.9], [0.12, 0.52, 0.92]);
        assert_eq!(spike_sync_within(&a, &b, 0.0, 1.0, 0.001), Ok(0.0), "every window bounded by 1 ms");
        assert_eq!(spike_sync_within(&a, &b, 0.0, 1.0, 0.05), Ok(1.0));
        assert_eq!(spike_sync(&a, &b, 0.0, 1.0), Ok(1.0));
        let lead: &[f64] = &[0.1, 0.3, 0.5, 0.7];
        let follow: &[f64] = &[0.15, 0.35, 0.55, 0.75];
        assert_eq!(synfire_indicator(&[follow, lead], 0.0, 1.0), Ok(-1.0));
        assert_eq!(synfire_indicator(&[follow, lead, &[], &[]], 0.0, 1.0), Ok(-8.0 / 24.0));
        assert_eq!(synfire_indicator(&[&[], &[]], 0.0, 1.0), Err(SyncError::NoSpikes));
        assert_eq!(spike_sync_multi(&[&[], &[], &[]], 0.0, 1.0), Err(SyncError::NoSpikes));
    }

    /// PSTH bins are the edges `start + k·bin`, and a spike is placed by those edges: at 0.3 s in a
    /// 0.1 s histogram it is in bin 2, because the edge `3 × 0.1` is `0.30000000000000004`.
    #[test]
    fn a_psth_places_each_spike_by_its_computed_edges() {
        let (x, bin) = (0.3_f64, 0.1_f64);
        assert!(3.0 * bin > x && x / bin < 3.0, "the trap: division says bin 2.99…, the edge says 0.3 is below bin 3");
        let p = psth(&[&[0.3]], 0.0, 0.1, 5).unwrap();
        assert_eq!(p.rate, [0.0, 0.0, 10.0, 0.0, 0.0]);
        // A spike exactly on a computed edge opens that bin; the window's end is outside it.
        let edge = 2.0 * 0.25;
        let p = psth(&[&[0.0, edge, 1.0], &[-0.1, 0.99, 1.2]], 0.0, 0.25, 4).unwrap();
        assert_eq!(p.rate, [2.0, 0.0, 2.0, 2.0], "two trials: one spike each is 1/(2 × 0.25) = 2 Hz");
        assert_eq!(p.outside, 3, "1.0, 1.2 and −0.1 are in no bin");
        // Far along, the first guess from division is corrected in both directions.
        let p = psth(&[&[0.7, 0.8, 0.9]], 0.0, 0.1, 10).unwrap();
        assert_eq!(p.rate.iter().filter(|&&r| r > 0.0).count(), 3);
        let bins: Vec<usize> = p.rate.iter().enumerate().filter(|(_, r)| **r > 0.0).map(|(k, _)| k).collect();
        let by_edges: Vec<usize> =
            [0.7, 0.8, 0.9].iter().map(|&x: &f64| (0..10).rev().find(|&k| f64::from(k) * 0.1 <= x).unwrap() as usize).collect();
        assert_eq!(bins, by_edges);
        // Where division and the edges disagree, in both directions. `1.46` IS the edge
        // `1.3 + 16 × 0.01`, but `(1.46 − 1.3)/0.01` is 15.99…; `10.999999999999998` is below the edge
        // `33 × (1/3)`, but divides to 33.
        let at = |x: f64, start: f64, bin: f64| psth(&[&[x]], start, bin, 40).unwrap().rate.iter().position(|&r| r > 0.0);
        let (x, start, bin) = (1.46_f64, 1.3_f64, 0.01_f64);
        assert_eq!(start + 16.0 * bin, x);
        assert!((x - start) / bin < 16.0);
        assert_eq!(at(1.46, 1.3, 0.01), Some(16), "corrected up to the edge it sits on");
        let (x, third) = (10.999999999999998_f64, 1.0_f64 / 3.0);
        assert!(x / third >= 33.0 && 33.0 * third > x);
        assert_eq!(at(10.999999999999998, 0.0, 1.0 / 3.0), Some(32), "corrected down below the edge above it");
    }

    /// The cross-correlogram counts lags `b − a`, and for independent uniform trains its mean is the
    /// triangle, not the flat line.
    ///
    /// Four thousand pairs of independent trains of twenty spikes on one second, lags binned at 0.1 s
    /// over `[−1, 1)`: every bin's mean is within four standard errors of
    /// [`correlogram_expectation`]. The flat expectation `n_a n_b w/T = 40` is right only near zero
    /// lag; in the bin `[0.8, 0.9)` the triangle gives 6 and the flat line is too high by a factor
    /// of 6.7.
    #[test]
    fn the_correlogram_of_independent_trains_is_the_triangle() {
        assert_eq!(cross_correlogram(&[0.5], &[0.52, 0.49, 0.8], -0.1, 0.05, 4), Err(SyncError::NotIncreasing { train: 1, index: 1 }));
        assert_eq!(cross_correlogram(&[0.5], &[0.49, 0.52, 0.8], -0.1, 0.05, 4), Ok(vec![0, 1, 1, 0]));
        assert_eq!(cross_correlogram(&[0.2, 0.5], &[0.52], -0.4, 0.2, 4), Ok(vec![0, 0, 1, 1]));
        assert_eq!(cross_correlogram(&[0.0], &[0.25], -0.25, 0.25, 2), Ok(vec![0, 0]), "a lag on the last edge is past it");
        // The same edge corrections as the PSTH, as lags.
        let mut c = cross_correlogram(&[0.0], &[1.46], 1.3, 0.01, 20).unwrap();
        assert_eq!(c.iter().position(|&n| n > 0), Some(16));
        c = cross_correlogram(&[0.0], &[10.999999999999998], 0.0, 1.0 / 3.0, 40).unwrap();
        assert_eq!(c.iter().position(|&n| n > 0), Some(32));
        let (n, trials) = (20usize, 4000);
        let mut rng = Rng::new(7);
        let mut sum = [0.0; 20];
        let mut sq = [0.0; 20];
        for _ in 0..trials {
            let mut draw = || {
                let mut t: Vec<f64> = (0..n).map(|_| rng.next_f64()).collect();
                t.sort_by(f64::total_cmp);
                t
            };
            let (a, b) = (draw(), draw());
            let c = cross_correlogram(&a, &b, -1.0, 0.1, 20).unwrap();
            for k in 0..20 {
                sum[k] += c[k] as f64;
                sq[k] += (c[k] * c[k]) as f64;
            }
        }
        for k in 0..20 {
            let lo = -1.0 + k as f64 * 0.1;
            let want = correlogram_expectation(n, n, 1.0, lo, lo + 0.1).unwrap();
            let mean = sum[k] / trials as f64;
            let se = ((sq[k] / trials as f64 - mean * mean) / trials as f64).sqrt();
            assert!((mean - want).abs() < 4.0 * se, "bin {k}: {mean} against {want} ± {se}");
        }
        let far = correlogram_expectation(n, n, 1.0, 0.8, 0.9).unwrap();
        assert!((far - 6.0).abs() < 1e-12, "{far}");
        let whole = correlogram_expectation(n, n, 1.0, -5.0, 5.0).unwrap();
        assert!((whole - 400.0).abs() < 1e-12, "every pair has some lag: {whole}");
        assert!((correlogram_expectation(n, n, 1.0, -0.05, 0.05).unwrap() - 39.0).abs() < 1e-12);
        // On a window of 2 s: still every pair somewhere, and the central 0.1 s holds `400 × (0.2 −
        // 0.0025)/4`.
        assert!((correlogram_expectation(n, n, 2.0, -3.0, 3.0).unwrap() - 400.0).abs() < 1e-12);
        assert!((correlogram_expectation(n, n, 2.0, -0.05, 0.05).unwrap() - 19.75).abs() < 1e-12);
    }

    /// The Schreiber correlation's closed form, against the filtered traces integrated numerically,
    /// and its known values.
    ///
    /// Two single spikes `Δ` apart correlate as `e^{−Δ²/4σ²}` exactly, identical trains as 1, and for
    /// two random trains the closed form equals the trapezoid of the Gaussian-filtered traces on a
    /// 50 µs grid to `10⁻⁹`.
    #[test]
    fn the_schreiber_correlation_is_the_cosine_of_the_filtered_traces() {
        let sigma = 0.01;
        let d: f64 = 0.013;
        assert!((schreiber(&[0.4], &[0.4 + d], sigma).unwrap() - (-d * d / (4.0 * sigma * sigma)).exp()).abs() < 1e-15);
        assert!((schreiber(VOLLEY[0], VOLLEY[0], sigma).unwrap() - 1.0).abs() < 1e-15);
        let trace = |t: &[f64], x: f64| t.iter().map(|&s| (-(x - s) * (x - s) / (2.0 * sigma * sigma)).exp()).sum::<f64>();
        let (a, b) = (VOLLEY[0], VOLLEY[3]);
        let (mut ab, mut aa, mut bb) = (0.0, 0.0, 0.0);
        let h = 5e-5;
        for k in 0..=24_000 {
            let x = -0.1 + k as f64 * h;
            let w = if k == 0 || k == 24_000 { 0.5 } else { 1.0 };
            let (p, q) = (trace(a, x), trace(b, x));
            ab += w * p * q;
            aa += w * p * p;
            bb += w * q * q;
        }
        let numeric = ab / (aa * bb).sqrt();
        let closed = schreiber(a, b, sigma).unwrap();
        assert!((numeric - closed).abs() < 1e-9, "{numeric} against {closed}");
        assert!(closed > 0.3 && closed < 0.9);
        // Four trials, so that the six pairs are not also the number of trials.
        let trials = [a, b, VOLLEY[5], VOLLEY[2]];
        let r = schreiber_reliability(&trials, sigma).unwrap();
        let mut by_hand = 0.0;
        for i in 0..4 {
            for j in i + 1..4 {
                by_hand += schreiber(trials[i], trials[j], sigma).unwrap();
            }
        }
        assert!((r - by_hand / 6.0).abs() < 1e-15);
    }

    /// Every refusal, rendered.
    #[test]
    fn every_refusal_names_what_it_refused() {
        let cases: Vec<(Result<f64, SyncError>, &str)> = vec![
            (spike_sync(&[0.1], &[0.2], 1.0, 1.0), "[1, 1] is not a window of positive length"),
            (spike_sync(&[0.1], &[0.2], 0.0, f64::INFINITY), "[0, inf] is not a window of positive length"),
            (spike_sync(&[0.1, f64::NAN], &[0.2], 0.0, 1.0), "train 0, spike 1 is NaN"),
            (spike_sync(&[0.1], &[0.2, 0.2], 0.0, 1.0), "train 1, spike 1 is not after the spike before it"),
            (spike_sync(&[0.1], &[0.2, 1.5], 0.0, 1.0), "train 1, spike 1 at 1.5 is outside the window"),
            (spike_sync(&[-0.1], &[0.2], 0.0, 1.0), "train 0, spike 0 at -0.1 is outside the window"),
            (spike_sync(&[], &[], 0.0, 1.0), "no train has a spike, so there is nothing to count"),
            (spike_sync_multi(&[&[0.1]], 0.0, 1.0), "a population measure needs two trains or more, not 1"),
            (synfire_indicator(&[&[0.1]], 0.0, 1.0), "a population measure needs two trains or more, not 1"),
            (synfire_indicator(&[&[0.1], &[0.3], &[0.2, 0.1]], 0.0, 1.0), "train 2, spike 1 is not after the spike before it"),
            (synfire_indicator(&[&[0.1], &[0.3]], 2.0, 1.0), "[2, 1] is not a window of positive length"),
            (spike_sync_within(&[0.1], &[0.2], 0.0, 1.0, 0.0), "max_tau = 0 must be finite and positive"),
            (schreiber(&[0.1], &[], 0.01), "no train has a spike, so there is nothing to count"),
            (schreiber(&[], &[0.1], 0.01), "no train has a spike, so there is nothing to count"),
            (schreiber(&[0.1], &[0.2], f64::NAN), "sigma = NaN must be finite and positive"),
            (schreiber(&[0.1], &[0.2], f64::INFINITY), "sigma = inf must be finite and positive"),
            (schreiber(&[0.2, 0.1], &[0.2], 0.01), "train 0, spike 1 is not after the spike before it"),
            (schreiber(&[0.1], &[f64::INFINITY], 0.01), "train 1, spike 0 is inf"),
            (schreiber_reliability(&[&[0.1]], 0.01), "a population measure needs two trains or more, not 1"),
            (correlogram_expectation(1, 1, 0.0, 0.0, 0.1), "duration = 0 must be finite and positive"),
            (correlogram_expectation(1, 1, 1.0, 0.2, 0.1), "[0.2, 0.1] is not a window of positive length"),
        ];
        for (got, want) in cases {
            assert_eq!(got.unwrap_err().to_string(), want);
        }
        assert_eq!(spike_order(&[0.1], &[0.2, 0.1], 0.0, 1.0).unwrap_err().to_string(), "train 1, spike 1 is not after the spike before it");
        assert_eq!(spike_order(&[0.1], &[0.2], 0.5, 1.0).unwrap_err().to_string(), "train 0, spike 0 at 0.1 is outside the window");
        assert_eq!(spike_order(&[], &[], 1.0, 0.0).unwrap_err().to_string(), "[1, 0] is not a window of positive length");
        assert_eq!(psth(&[&[0.1]], 0.0, 0.0, 3).unwrap_err().to_string(), "bin = 0 must be finite and positive");
        assert_eq!(psth(&[&[0.1]], 0.0, 0.1, 0).unwrap_err().to_string(), "a histogram needs at least one bin");
        assert_eq!(psth(&[], 0.0, 0.1, 3).unwrap_err().to_string(), "a population measure needs two trains or more, not 0");
        assert_eq!(psth(&[&[0.1]], f64::NAN, 0.1, 3).unwrap_err().to_string(), "[NaN, NaN] is not a window of positive length");
        assert_eq!(psth(&[&[0.2, 0.1]], 0.0, 0.1, 3).unwrap_err().to_string(), "train 0, spike 1 is not after the spike before it");
        assert_eq!(cross_correlogram(&[0.1], &[0.2], 0.0, -1.0, 3).unwrap_err().to_string(), "bin = -1 must be finite and positive");
        assert_eq!(cross_correlogram(&[0.1], &[0.2], 0.0, 0.1, 0).unwrap_err().to_string(), "a histogram needs at least one bin");
        assert_eq!(cross_correlogram(&[0.1], &[0.2], f64::NAN, 0.1, 3).unwrap_err().to_string(), "[NaN, NaN] is not a window of positive length");
        assert_eq!(cross_correlogram(&[f64::NAN], &[0.2], 0.0, 0.1, 3).unwrap_err().to_string(), "train 0, spike 0 is NaN");
    }

    /// The pair in `Elephant`'s documentation of its STTC (train 1 = 1.3, 7.56, 15.87, 28.23, 30.9,
    /// 34.2, 38.2, 43.2 ms; train 2 = 1.02, 2.71, 18.82, 28.46, 28.79, 43.6 ms; a 50 ms window;
    /// `dt` = 5 ms), whose value the authors' own C (`spike_time_tiling_coefficient.c`, commit
    /// `5f18868`) also gives to the last digit: 0.4958601655933762. `T_A` = 0.9168 and `T_B` =
    /// 0.7536 are the same C's `run_T` over the window.
    #[test]
    fn the_sttc_of_the_documented_pair_is_the_authors_number() {
        let a = [1.3, 7.56, 15.87, 28.23, 30.9, 34.2, 38.2, 43.2];
        let b = [1.02, 2.71, 18.82, 28.46, 28.79, 43.6];
        let x = sttc(&a, &b, 0.0, 50.0, 5.0).unwrap();
        assert!((x - 0.495_860_165_593_376_2).abs() < 1e-15, "{x}");
        assert!((tiled_fraction(&a, 0.0, 50.0, 5.0).unwrap() - 0.9168).abs() < 1e-15);
        assert!((tiled_fraction(&b, 0.0, 50.0, 5.0).unwrap() - 0.7536).abs() < 1e-15);
        // Symmetric in its two trains.
        assert!((sttc(&b, &a, 0.0, 50.0, 5.0).unwrap() - x).abs() < 1e-15);
    }

    /// Three random pairs from a sweep of 3,000 run through the authors' C (`run_sttc`, commit
    /// `5f18868`, compiled with a two-line `R.h` stub). Over the whole sweep the two agree to
    /// 2.8e-13 wherever the C's answer is defined — `T` is summed in a different order here — and
    /// the 205 pairs they do not share are the saturated ones, which
    /// `a_train_that_tiles_the_whole_window_is_refused_where_the_authors_c_returns_noise` covers.
    #[test]
    fn the_sttc_is_the_authors_c_on_random_pairs() {
        // dt, window start and end, the two trains, and the C's STTC.
        type Case<'a> = (f64, f64, f64, &'a [f64], &'a [f64], f64);
        let cases: [Case; 3] = [
            (
                0.001,
                -3.0,
                7.0,
                &[-2.92619, -2.170546, -1.377623, 0.80246, 1.725056, 3.716438, 3.988784, 4.045382, 4.34728, 6.723101],
                &[-2.924797, -2.247494, -2.170866, -1.37689, 0.802904, 1.726634, 3.716156, 3.989263, 4.044752, 4.345437, 5.52675, 6.722988],
                0.640_393_143_979_298_9,
            ),
            (
                0.1,
                -3.0,
                7.0,
                &[-1.654725, -1.488358, -1.25249, -0.470439, 0.058956, 2.099415, 3.096051, 5.168832, 6.375943, 6.901969],
                &[-2.997313, -2.845982, 0.906318, 2.667314, 2.979884, 3.563851, 3.863408, 4.43257, 4.629185, 6.651991],
                -0.190_751_55,
            ),
            (
                0.5,
                -3.0,
                47.0,
                &[2.015211, 3.511205, 11.205389, 13.678555, 16.061985, 19.084735, 28.537784, 34.865488, 35.027835, 44.490748],
                &[1.271584, 4.415861, 4.950109, 11.934593, 13.420811, 16.283233, 18.304414, 28.86067, 34.274559, 34.841615, 44.565667],
                0.374_409_014_725_058_66,
            ),
        ];
        for (dt, start, end, a, b, c) in cases {
            let x = sttc(a, b, start, end, dt).unwrap();
            assert!((x - c).abs() < 1e-13, "dt {dt}: {x} against the C's {c}");
        }
    }

    /// Closed forms. Identical trains tile each other exactly: `P = 1` on both sides and each half is
    /// `(1 − T)/(1 − T)`, so the STTC is 1 to the bit. Trains that never come within `dt` of each
    /// other have `P = 0`, and the STTC is `−(T_A + T_B)/2` exactly — negative, as an index of
    /// avoidance should be. And the within-`dt` test is the authors' `|a − b| ≤ dt`, inclusive: at a
    /// gap of exactly `dt` (0.25, binary) a spike counts, one ulp further it does not.
    #[test]
    fn the_sttc_meets_its_closed_forms() {
        let a = [0.5, 1.75, 3.0, 6.125];
        assert_eq!(sttc(&a, &a, 0.0, 8.0, 0.25).unwrap(), 1.0);
        let apart = [1.0, 2.0, 4.0];
        let other = [1.5, 3.0, 5.0];
        let (t_a, t_b) = (tiled_fraction(&apart, 0.0, 8.0, 0.25).unwrap(), tiled_fraction(&other, 0.0, 8.0, 0.25).unwrap());
        assert_eq!((t_a, t_b), (0.1875, 0.1875), "three whole half-second tiles in eight seconds");
        assert_eq!(sttc(&apart, &other, 0.0, 8.0, 0.25).unwrap(), -(t_a + t_b) / 2.0);
        // P at the boundary: one spike each, 0.25 apart, is half-tiled on each side.
        let on = sttc(&[1.0], &[1.25], 0.0, 8.0, 0.25).unwrap();
        let t = 0.0625;
        assert_eq!(on, (1.0 - t) / (1.0 - t), "exactly dt apart counts: {on}");
        let off = sttc(&[1.0], &[1.25f64.next_up()], 0.0, 8.0, 0.25).unwrap();
        assert!(off < 0.0, "one ulp beyond dt does not count: {off}");
        // Tiles merge where they overlap and are clipped at the edges: spikes at 0.125 and 0.375
        // with dt = 0.25 cover [0, 0.625] once, not 0.5 + 0.5 − 0.125.
        assert_eq!(tiled_fraction(&[0.125, 0.375], 0.0, 8.0, 0.25).unwrap(), 0.625 / 8.0);
        assert_eq!(tiled_fraction(&[7.875], 0.0, 8.0, 0.25).unwrap(), 0.375 / 8.0);
    }

    /// Cutts and Eglen's point against the correlation index: for independent trains the STTC is 0
    /// in expectation WHATEVER THE RATES. Two hundred pairs of independent Poisson trains, 5 Hz
    /// against 50 Hz over 20 s with `dt` = 5 ms, seeded: the mean is within three standard errors
    /// of zero. Measured: mean 0.0035, standard error 0.0024 (1.46 standard errors).
    #[test]
    fn independent_trains_have_an_sttc_of_zero_whatever_their_rates() {
        let mut rng = Rng::new(20_140_919);
        let mut poisson = |rate: f64| {
            let (mut t, mut out) = (0.0f64, Vec::new());
            for _ in 0..100_000 {
                t += -(1.0 - rng.next_f64()).ln() / rate;
                if t > 20.0 {
                    break;
                }
                out.push(t);
            }
            out
        };
        let xs: Vec<f64> = (0..200).map(|_| sttc(&poisson(5.0), &poisson(50.0), 0.0, 20.0, 5e-3).unwrap()).collect();
        let n = xs.len() as f64;
        let mean = xs.iter().sum::<f64>() / n;
        let se = (xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0) / n).sqrt();
        assert!(mean.abs() < 3.0 * se && se < 0.01, "mean {mean}, standard error {se}");
    }

    /// ⚠ The authors' C decides a saturated train by rounding. A train whose tiles cover the whole
    /// window makes a half of the formula `0/0` in exact arithmetic; the C computes `T` by
    /// subtracting overlaps from `2·N·dt`, lands a hair either side of 1, and returns NaN (12 of the
    /// sweep's 205 such pairs) or a number anywhere from 0.19 to 1 (the other 193). Its lone-spike
    /// branch also misses one edge: a single spike at 4 in `[0, 8]` with `dt` = 5 gets `T` = 1.125
    /// and an STTC of 1. This module decides coverage by the geometry — the first tile reaches the
    /// start, the last the end, no gap wider than `2·dt` — gives that spike `T` = 1, and refuses;
    /// it also refuses a tiling whose gap is too narrow for the sum of lengths to see.
    #[test]
    fn a_train_that_tiles_the_whole_window_is_refused_where_the_authors_c_returns_noise() {
        assert_eq!(tiled_fraction(&[4.0], 0.0, 8.0, 5.0).unwrap(), 1.0);
        assert_eq!(sttc(&[4.0], &[4.0], 0.0, 8.0, 5.0), Err(SyncError::Saturated { train: 0, dt: 5.0 }));
        // Spikes exactly 2·dt apart from dt to end − dt tile everything; widen one gap by an ulp
        // and they do not.
        let full = [0.25, 0.75, 1.25, 1.75];
        assert_eq!(tiled_fraction(&full, 0.0, 2.0, 0.25).unwrap(), 1.0);
        assert_eq!(sttc(&[1.0], &full, 0.0, 2.0, 0.25), Err(SyncError::Saturated { train: 1, dt: 0.25 }));
        let e = 1.0 / 1_048_576.0;
        let gap = [0.25, 0.75, 1.25 + e, 1.75];
        assert!(tiled_fraction(&gap, 0.0, 2.0, 0.25).unwrap() < 1.0);
        assert!(sttc(&[1.0], &gap, 0.0, 2.0, 0.25).is_ok());
        // Each edge on its own: a first tile short of the start, a last short of the end.
        assert!(tiled_fraction(&[0.25 + e, 0.75, 1.25, 1.75], 0.0, 2.0, 0.25).unwrap() < 1.0);
        assert!(tiled_fraction(&[0.25, 0.75, 1.25, 1.75 - e], 0.0, 2.0, 0.25).unwrap() < 1.0);
        // And a gap of one ulp at the start is real to the geometry but invisible to the sum, which
        // comes to the whole window: refused too, rather than handed to a 0/0.
        // The other way round: spikes at 0.5 and 0.7 tile `[0.3, 0.9]` at `dt` = 0.3 by the geometry,
        // while the sum of their tile lengths comes to 0.9999999999999998 of the window. Coverage is
        // reported as exactly 1, and refused, rather than left to that rounding.
        assert_eq!(tiled_fraction(&[0.5, 0.7], 0.3, 0.9, 0.3).unwrap(), 1.0);
        assert_eq!(sttc(&[0.6], &[0.5, 0.7], 0.3, 0.9, 0.3), Err(SyncError::Saturated { train: 1, dt: 0.3 }));
        // A window that does not start at zero divides by its length, not its end: half a second of
        // tile in a two-second window.
        assert_eq!(tiled_fraction(&[1.0], 0.5, 2.5, 0.25).unwrap(), 0.25);
        // And a train with no spikes tiles nothing.
        assert_eq!(tiled_fraction(&[], 0.0, 4.0, 0.1).unwrap(), 0.0);
        let ulp = [0.25f64.next_up(), 0.75, 1.25, 1.75];
        assert_eq!(tiled_fraction(&ulp, 0.0, 2.0, 0.25).unwrap(), 1.0);
        assert_eq!(sttc(&[1.0], &ulp, 0.0, 2.0, 0.25), Err(SyncError::Saturated { train: 1, dt: 0.25 }));
        assert_eq!(
            sttc(&[4.0], &[4.0], 0.0, 8.0, 5.0).unwrap_err().to_string(),
            "train 0 is within dt = 5 of a spike everywhere in the window, so its tiling covers the whole recording"
        );
    }

    /// Every refusal, by its message.
    #[test]
    fn the_sttc_refuses_what_it_cannot_measure() {
        let a = [1.0, 2.0];
        assert_eq!(sttc(&[], &a, 0.0, 4.0, 0.1), Err(SyncError::EmptyTrain { train: 0 }));
        assert_eq!(sttc(&a, &[], 0.0, 4.0, 0.1), Err(SyncError::EmptyTrain { train: 1 }));
        assert_eq!(sttc(&a, &[], 0.0, 4.0, 0.1).unwrap_err().to_string(), "train 1 has no spikes, so there is nothing to tile");
        assert_eq!(sttc(&a, &a, 0.0, 4.0, 0.0), Err(SyncError::NotPositive { what: "dt", value: 0.0 }));
        assert!(sttc(&a, &a, 0.0, 4.0, f64::NAN).is_err());
        assert_eq!(sttc(&a, &a, 4.0, 0.0, 0.1), Err(SyncError::Window { start: 4.0, end: 0.0 }));
        assert_eq!(sttc(&a, &[5.0], 0.0, 4.0, 0.1), Err(SyncError::Outside { train: 1, index: 0, value: 5.0 }));
        assert_eq!(sttc(&[2.0, 1.0], &a, 0.0, 4.0, 0.1), Err(SyncError::NotIncreasing { train: 0, index: 1 }));
        assert_eq!(tiled_fraction(&a, 0.0, 4.0, -1.0), Err(SyncError::NotPositive { what: "dt", value: -1.0 }));
        assert_eq!(tiled_fraction(&[5.0], 0.0, 4.0, 0.1), Err(SyncError::Outside { train: 0, index: 0, value: 5.0 }));
        assert_eq!(tiled_fraction(&a, 1.0, 1.0, 0.1), Err(SyncError::Window { start: 1.0, end: 1.0 }));
    }
}
