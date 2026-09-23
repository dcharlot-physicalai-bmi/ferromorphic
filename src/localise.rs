//! Sound localisation by coincidence: the Jeffress delay-line array, which turns a time difference
//! between two ears into a PLACE — checked against the geometry, the quantisation bound, the
//! aliasing frequency and the coincidence probability under spike jitter.
//!
//! # What the mechanism is
//!
//! A sound from azimuth `θ` reaches the nearer ear first. For a distant source and ears a distance
//! `d` apart the interaural time difference is the path difference over the speed of sound,
//! `ITD = (d/c) sin θ`; for a rigid spherical head of radius `r` at high frequency it is
//! Woodworth's `(r/c)(θ + sin θ)` (Woodworth, *Experimental Psychology*, Holt, 1938). Jeffress (*A
//! place theory of sound localization*, Journal of Comparative and Physiological Psychology
//! 41(1):35–39, 1948) proposed the circuit that reads it: axons from the two ears run in opposite
//! directions along an array of coincidence detectors, so detector `k` sees the two inputs with a
//! built-in delay difference `Δ_k`, and fires most when `Δ_k` cancels the ITD. The time code
//! becomes a place code. Carr and Konishi (*A circuit for detection of interaural time differences
//! in the brain stem of the barn owl*, Journal of Neuroscience 10(10):3227–3246, 1990) found that
//! circuit in the owl's nucleus laminaris.
//!
//! # Why it is in a neuromorphic crate
//!
//! It is the canonical computation done WITH spike timing rather than despite it — microsecond
//! differences resolved by neurons whose time constants are a hundred times longer, because the
//! computation is a coincidence and not an integration — and it is what a pair of silicon cochleae
//! ([`crate::cochlea`]) feeds. Its cost is a count: every spike visits every detector.
//!
//! # The closed forms this module is checked against
//!
//! - **Geometry.** `ITD = (d/c) sin θ` and its inverse; Woodworth's `(r/c)(θ + sin θ)` and its
//!   inverse by bisection; the speed of sound `331.3 √(1 + ϑ/273.15)` m/s.
//! - **The array reads the ITD to half a detector spacing.** For phase-locked trains the estimate
//!   is within `δ/2` of the truth at every ITD in range.
//! - **Counts are exact.** `N` cycles give `N` coincidences at the matched detector, none at a
//!   detector more than the window away, and `N − |m|` at a detector an integer `m` periods away.
//! - **Aliasing.** A tone of period `T` is unambiguous on an array spanning `±ITD_max` for every
//!   source iff `T > 2·ITD_max`, i.e. below `1/(2·ITD_max)`; above it a second peak appears one
//!   period away, and [`Estimate::peaks`] counts it. On a recording of `N` cycles the alias is
//!   exactly ONE coincidence shorter than the true peak — `N − 1` against `N` — so a finite
//!   recording can tell them apart and a long one barely can. (The first draft of this module
//!   flagged ambiguity only on an exact tie, which for that reason never happens.)
//! - **Resolution in space.** One detector spacing `δ` is `c δ/(d cos θ)` of azimuth: finest
//!   straight ahead, unbounded toward the side.
//! - **Jitter.** With independent Gaussian jitter `σ` on every spike, a same-cycle pair coincides
//!   with probability `Φ((w − μ)/(σ√2)) − Φ((−w − μ)/(σ√2))`, `μ` the detector's mismatch; the
//!   sampled count is held to `N` times that within four binomial standard errors.
//! - **Cost.** Synaptic events = detectors × (left spikes + right spikes).
//!
//! # What this module has NOT reproduced
//!
//! - Cochlear filtering and the statistics of phase locking; the spike trains here are made by
//!   [`phase_locked`], one spike per cycle, not by an auditory nerve model.
//! - Interaural LEVEL differences, which carry localisation above a few kilohertz.
//! - The mammalian alternative. In gerbils and guinea pigs the best delays sit outside the
//!   physiological range and the ITD appears to be read from the SLOPE of two broad channels
//!   rather than from a peak in an array (`McAlpine` and Grothe, *Sound localization and delay
//!   lines — do mammals fit the model?*, Trends in Neurosciences 26(7):347–350, 2003). This
//!   module is the place code, and does not claim it is the one a mammal uses.

use core::f64::consts::FRAC_PI_2;
use core::fmt;

use crate::fusion::gaussian;
use crate::rng::Rng;
use crate::surrogate::normal_cdf;

/// What went wrong, named rather than guessed around.
#[derive(Debug, Clone, PartialEq)]
pub enum LocaliseError {
    /// A count of zero where at least one is needed.
    Empty {
        /// What was empty.
        what: &'static str,
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
    /// A spike train that is not in time order.
    Unsorted {
        /// Which train.
        what: &'static str,
        /// Index of the first spike earlier than its predecessor.
        index: usize,
    },
}

impl fmt::Display for LocaliseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty { what } => write!(f, "{what} is empty"),
            Self::OutOfRange { what, value, low, high } => {
                write!(f, "{what} = {value} is outside [{low}, {high}]")
            }
            Self::NonFinite { what, index } => write!(f, "{what} is not finite at {index}"),
            Self::Unsorted { what, index } => write!(f, "{what} is out of time order at {index}"),
        }
    }
}

impl std::error::Error for LocaliseError {}

fn positive(what: &'static str, v: f64) -> Result<f64, LocaliseError> {
    if v.is_finite() && v > 0.0 {
        Ok(v)
    } else {
        // NOT `f64::INFINITY`: the guard above requires `is_finite()`, so an infinity is refused
        // — and a refusal that names an inadmissible value as its upper bound is telling the
        // caller to send something this function will reject again.
        Err(LocaliseError::OutOfRange { what, value: v, low: f64::MIN_POSITIVE, high: f64::MAX })
    }
}

fn azimuth_in_range(azimuth: f64) -> Result<f64, LocaliseError> {
    if azimuth.is_finite() && azimuth.abs() <= FRAC_PI_2 {
        Ok(azimuth)
    } else {
        Err(LocaliseError::OutOfRange { what: "azimuth", value: azimuth, low: -FRAC_PI_2, high: FRAC_PI_2 })
    }
}

fn sorted(what: &'static str, t: &[f64]) -> Result<(), LocaliseError> {
    if let Some(i) = t.iter().position(|x| !x.is_finite()) {
        return Err(LocaliseError::NonFinite { what, index: i });
    }
    if let Some(i) = t.windows(2).position(|p| p[1] < p[0]) {
        return Err(LocaliseError::Unsorted { what, index: i + 1 });
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------------------------

/// The speed of sound in dry air at `celsius`, m/s: `331.3 √(1 + ϑ/273.15)`. `None` at or below
/// absolute zero or for a non-finite temperature.
#[must_use]
pub fn speed_of_sound(celsius: f64) -> Option<f64> {
    if !celsius.is_finite() || celsius <= -273.15 {
        return None;
    }
    Some(331.3 * (1.0 + celsius / 273.15).sqrt())
}

/// The far-field ITD `(d/c) sin θ`, seconds, for ears `d` metres apart; positive when the source
/// is toward the RIGHT ear (positive azimuth), meaning the right ear hears it first.
///
/// # Errors
///
/// [`LocaliseError::OutOfRange`] for a non-positive `d` or `c`, or an azimuth outside `±π/2`.
pub fn itd_far_field(d: f64, c: f64, azimuth: f64) -> Result<f64, LocaliseError> {
    let (d, c) = (positive("d", d)?, positive("c", c)?);
    Ok(d / c * azimuth_in_range(azimuth)?.sin())
}

/// The azimuth a far-field ITD implies, `asin(c·ITD/d)`, radians. `None` for an ITD larger than
/// the head allows, or a non-positive `d` or `c`.
#[must_use]
pub fn azimuth_far_field(d: f64, c: f64, itd: f64) -> Option<f64> {
    if !(d > 0.0) || !(c > 0.0) || !d.is_finite() || !c.is_finite() || !itd.is_finite() {
        return None;
    }
    let s = c * itd / d;
    if s.abs() > 1.0 { None } else { Some(s.asin()) }
}

/// Woodworth's ITD for a rigid sphere of radius `r`, `(r/c)(θ + sin θ)`, seconds.
///
/// # Errors
///
/// As [`itd_far_field`].
pub fn itd_woodworth(r: f64, c: f64, azimuth: f64) -> Result<f64, LocaliseError> {
    let (r, c) = (positive("r", r)?, positive("c", c)?);
    let a = azimuth_in_range(azimuth)?;
    Ok(r / c * (a + a.sin()))
}

/// The azimuth a Woodworth ITD implies, by bisection on the monotone `θ + sin θ`. `None` past
/// `(r/c)(π/2 + 1)`, or for a non-positive `r` or `c`.
#[must_use]
pub fn azimuth_woodworth(r: f64, c: f64, itd: f64) -> Option<f64> {
    if !(r > 0.0) || !(c > 0.0) || !r.is_finite() || !c.is_finite() || !itd.is_finite() {
        return None;
    }
    let target = c * itd.abs() / r;
    if target > FRAC_PI_2 + 1.0 {
        return None;
    }
    let (mut lo, mut hi) = (0.0f64, FRAC_PI_2);
    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        if mid + mid.sin() < target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Some((0.5 * (lo + hi)).copysign(itd))
}

/// The highest tone frequency, hertz, at which an array spanning `±max_itd` shows ONE
/// full-height peak wherever the source is: `1/(2·max_itd)`. `None` for a non-positive span.
#[must_use]
pub fn unambiguous_below(max_itd: f64) -> Option<f64> {
    if max_itd > 0.0 && max_itd.is_finite() { Some(0.5 / max_itd) } else { None }
}

/// The azimuth one detector spacing `delta` covers at `azimuth`, radians: `c δ/(d cos θ)`.
/// `None` at `±π/2`, where the ITD stops changing with angle, or for a non-positive argument.
#[must_use]
pub fn azimuth_resolution(d: f64, c: f64, delta: f64, azimuth: f64) -> Option<f64> {
    let ok = d > 0.0 && c > 0.0 && delta > 0.0 && d.is_finite() && c.is_finite() && delta.is_finite();
    if !ok || !azimuth.is_finite() || azimuth.abs() >= FRAC_PI_2 {
        return None;
    }
    Some(c * delta / (d * azimuth.cos()))
}

/// The probability that a same-cycle spike pair coincides within `window` at a detector whose
/// delay misses the ITD by `mismatch`, when every spike carries independent Gaussian jitter of
/// standard deviation `sigma`: the difference of two jittered times is `N(mismatch, 2σ²)`.
/// `None` for a non-positive `window` or `sigma`.
#[must_use]
pub fn coincidence_probability(mismatch: f64, window: f64, sigma: f64) -> Option<f64> {
    if !(window > 0.0) || !(sigma > 0.0) || !mismatch.is_finite() || !window.is_finite() || !sigma.is_finite() {
        return None;
    }
    let s = sigma * core::f64::consts::SQRT_2;
    Some(normal_cdf((window - mismatch) / s) - normal_cdf((-window - mismatch) / s))
}

// ---------------------------------------------------------------------------------------------
// Spike trains
// ---------------------------------------------------------------------------------------------

/// A phase-locked train: `cycles` spikes, one per `period`, the first at `start`, each displaced
/// by Gaussian jitter of standard deviation `sigma` (zero for none) and then put in time order.
///
/// # Errors
///
/// [`LocaliseError::Empty`] for zero cycles, [`LocaliseError::OutOfRange`] for a non-positive
/// period or a negative jitter, [`LocaliseError::NonFinite`] for a non-finite start.
pub fn phase_locked(start: f64, period: f64, cycles: usize, sigma: f64, rng: &mut Rng) -> Result<Vec<f64>, LocaliseError> {
    if cycles == 0 {
        return Err(LocaliseError::Empty { what: "cycles" });
    }
    if !start.is_finite() {
        return Err(LocaliseError::NonFinite { what: "start", index: 0 });
    }
    let period = positive("period", period)?;
    if !(sigma >= 0.0) || !sigma.is_finite() {
        return Err(LocaliseError::OutOfRange { what: "sigma", value: sigma, low: 0.0, high: f64::INFINITY });
    }
    let mut t: Vec<f64> = (0..cycles)
        .map(|n| start + period * n as f64 + if sigma > 0.0 { sigma * gaussian(rng) } else { 0.0 })
        .collect();
    t.sort_by(f64::total_cmp);
    Ok(t)
}

// ---------------------------------------------------------------------------------------------
// The array
// ---------------------------------------------------------------------------------------------

/// An array of coincidence detectors. Detector `k` delays the RIGHT input by `delays[k]` relative
/// to the left, so it is matched to a source whose ITD (right ear leading) is `delays[k]`.
#[derive(Debug, Clone, PartialEq)]
pub struct Jeffress {
    /// Built-in delay differences, seconds, increasing.
    pub delays: Vec<f64>,
    /// Coincidence half-window, seconds: two spikes coincide when they arrive within it.
    pub window: f64,
}

/// What the array concluded.
#[derive(Debug, Clone, PartialEq)]
pub struct Estimate {
    /// The ITD read from the array: the mean delay of the first run of neighbouring detectors
    /// tied for the most coincidences, seconds.
    pub itd: f64,
    /// That largest count.
    pub count: u64,
    /// Separate peaks: runs of neighbouring detectors whose count is at least HALF the largest
    /// (the usual full-width-at-half-maximum convention). One for an unambiguous reading; more
    /// when the tone's period fits inside the array and an alias stands beside the true peak.
    pub peaks: usize,
    /// Synaptic events spent: every spike visited every detector.
    pub syn_events: u64,
}

impl Jeffress {
    /// `detectors` delays evenly spread over `[−max_itd, +max_itd]`, ends included.
    ///
    /// # Errors
    ///
    /// [`LocaliseError::Empty`] for fewer than two detectors, [`LocaliseError::OutOfRange`] for a
    /// non-positive span or window.
    pub fn uniform(max_itd: f64, detectors: usize, window: f64) -> Result<Self, LocaliseError> {
        if detectors < 2 {
            return Err(LocaliseError::Empty { what: "detectors (needs two)" });
        }
        let max_itd = positive("max_itd", max_itd)?;
        let window = positive("window", window)?;
        let step = 2.0 * max_itd / (detectors - 1) as f64;
        Ok(Self { delays: (0..detectors).map(|k| -max_itd + step * k as f64).collect(), window })
    }

    /// The spacing between neighbouring detectors, seconds.
    #[must_use]
    pub fn spacing(&self) -> f64 {
        if self.delays.len() < 2 { 0.0 } else { (self.delays[self.delays.len() - 1] - self.delays[0]) / (self.delays.len() - 1) as f64 }
    }

    /// Coincidences at every detector: pairs `(l, r)` with `|t_l − (t_r + Δ_k)| ≤ window`.
    ///
    /// # Errors
    ///
    /// [`LocaliseError::NonFinite`] or [`LocaliseError::Unsorted`] for a bad train.
    pub fn coincidences(&self, left: &[f64], right: &[f64]) -> Result<Vec<u64>, LocaliseError> {
        // `window` is a `pub` field, so `Jeffress::uniform`'s check can be bypassed by a struct
        // literal or a later assignment — and a negative window is meaningless. It is currently
        // survivable only because `if hi < lo { hi = lo }` pulls the high pointer forward; without
        // that line `hi - lo` wraps and the detector reports 18,446,744,073,709,551,615
        // coincidences. Invariant 10 of this crate's own guidance: an input the code cannot
        // understand is an error naming what was sent, not a silently plausible answer.
        positive("window", self.window)?;
        sorted("left", left)?;
        sorted("right", right)?;
        Ok(self
            .delays
            .iter()
            .map(|&delta| {
                // Both trains are sorted, so the window of right spikes that can pair with each
                // left spike only ever moves forward.
                let (mut lo, mut hi) = (0usize, 0usize);
                let mut count = 0u64;
                for &tl in left {
                    while lo < right.len() && right[lo] + delta < tl - self.window {
                        lo += 1;
                    }
                    if hi < lo {
                        hi = lo;
                    }
                    while hi < right.len() && right[hi] + delta <= tl + self.window {
                        hi += 1;
                    }
                    count += (hi - lo) as u64;
                }
                count
            })
            .collect())
    }

    /// Read the ITD off the array.
    ///
    /// # Errors
    ///
    /// As [`Jeffress::coincidences`], plus [`LocaliseError::Empty`] if no detector saw a single
    /// coincidence — there is then no peak to read, and zero would be a claim.
    pub fn estimate(&self, left: &[f64], right: &[f64]) -> Result<Estimate, LocaliseError> {
        let counts = self.coincidences(left, right)?;
        let best = counts.iter().copied().max().unwrap_or(0);
        if best == 0 {
            return Err(LocaliseError::Empty { what: "coincidences" });
        }
        // The run of detectors tied for the most, starting at the first of them.
        let first = counts.iter().position(|&c| c == best).unwrap_or(0);
        let run = counts[first..].iter().take_while(|&&c| c == best).count();
        let itd = self.delays[first..first + run].iter().sum::<f64>() / run as f64;
        let tall: Vec<bool> = counts.iter().map(|&c| 2 * c >= best).collect();
        let peaks = (0..tall.len()).filter(|&k| tall[k] && (k == 0 || !tall[k - 1])).count();
        let syn_events = (self.delays.len() as u64).saturating_mul((left.len() + right.len()) as u64);
        Ok(Estimate { itd, count: best, peaks, syn_events })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const D: f64 = 0.18;
    const C: f64 = 343.2;

    #[test]
    fn the_geometry_is_the_path_difference() {
        assert!((speed_of_sound(20.0).unwrap() - 343.2).abs() < 0.05);
        assert_eq!(speed_of_sound(0.0), Some(331.3));
        assert_eq!(speed_of_sound(-273.15), None);
        assert_eq!(speed_of_sound(f64::NAN), None);
        // Straight ahead there is no difference; at the side it is d/c = 524 µs for these ears.
        assert_eq!(itd_far_field(D, C, 0.0).unwrap(), 0.0);
        assert!((itd_far_field(D, C, FRAC_PI_2).unwrap() - D / C).abs() < 1e-18);
        assert!((itd_far_field(D, C, 0.5).unwrap() - D / C * 0.5f64.sin()).abs() < 1e-18);
        assert!(itd_far_field(D, C, -0.5).unwrap() < 0.0, "a source to the left leads at the left ear");
        for az in [-1.2, -0.3, 0.0, 0.7, 1.5] {
            let back = azimuth_far_field(D, C, itd_far_field(D, C, az).unwrap()).unwrap();
            assert!((back - az).abs() < 1e-12, "{az} → {back}");
            let r = 0.0875;
            let w = itd_woodworth(r, C, az).unwrap();
            assert!((w - r / C * (az + az.sin())).abs() < 1e-18);
            assert!((azimuth_woodworth(r, C, w).unwrap() - az).abs() < 1e-12);
        }
        // The sphere's detour makes its ITD larger than the straight path's for the same head
        // width, by (θ + sin θ)/(2 sin θ): at the side, (π/2 + 1)/2 = 1.285.
        let ratio = itd_woodworth(0.09, C, FRAC_PI_2).unwrap() / itd_far_field(0.18, C, FRAC_PI_2).unwrap();
        assert!((ratio - (FRAC_PI_2 + 1.0) / 2.0).abs() < 1e-12);
        assert_eq!(azimuth_far_field(D, C, 1.01 * D / C), None);
        assert_eq!(azimuth_far_field(0.0, C, 1e-4), None);
        assert_eq!(azimuth_woodworth(0.09, C, 1.01 * 0.09 / C * (FRAC_PI_2 + 1.0)), None);
        assert_eq!(azimuth_woodworth(0.09, 0.0, 1e-4), None);
    }

    #[test]
    fn the_array_reads_the_itd_to_half_a_detector_spacing() {
        let max_itd = D / C;
        let array = Jeffress::uniform(max_itd, 41, 20e-6).unwrap();
        let delta = array.spacing();
        assert!((delta - 2.0 * max_itd / 40.0).abs() < 1e-18);
        assert_eq!(array.delays[20], 0.0);
        let mut rng = Rng::new(1);
        let period = 1.0 / 500.0; // 500 Hz: below the aliasing limit of 954 Hz.
        assert!(500.0 < unambiguous_below(max_itd).unwrap());
        for k in 0..=24 {
            let az = -1.4 + 2.8 * f64::from(k) / 24.0;
            let itd = itd_far_field(D, C, az).unwrap();
            let right = phase_locked(0.01, period, 100, 0.0, &mut rng).unwrap();
            let left = phase_locked(0.01 + itd, period, 100, 0.0, &mut rng).unwrap();
            let est = array.estimate(&left, &right).unwrap();
            assert_eq!(est.count, 100, "every cycle coincides at the matched detector");
            assert_eq!(est.peaks, 1);
            // Half a spacing, plus the rounding of times near 60 ms.
            assert!((est.itd - itd).abs() <= 0.5 * delta + 1e-15, "azimuth {az}: read {} for {itd}", est.itd);
            assert_eq!(est.syn_events, 41 * 200);
        }
    }

    #[test]
    fn counts_are_exact_at_the_match_off_it_and_one_period_away() {
        // Detectors at exactly −T, 0, +T/4 and +T relative to a source with zero ITD.
        let period = 1e-3;
        let array = Jeffress { delays: vec![-period, 0.0, 0.25 * period, period], window: 10e-6 };
        let mut rng = Rng::new(2);
        let train = phase_locked(0.0, period, 50, 0.0, &mut rng).unwrap();
        let counts = array.coincidences(&train, &train).unwrap();
        // A delay of a whole period pairs cycle n with cycle n ± 1: one pair fewer.
        assert_eq!(counts, vec![49, 50, 0, 49]);
        // The window is closed: a pair exactly `window` apart counts, one just past it does not.
        let edge = Jeffress { delays: vec![0.0], window: 0.25 };
        assert_eq!(edge.coincidences(&[1.0], &[1.25]).unwrap(), vec![1]);
        assert_eq!(edge.coincidences(&[1.0], &[0.75]).unwrap(), vec![1]);
        assert_eq!(edge.coincidences(&[1.0], &[1.2500001]).unwrap(), vec![0]);
        assert_eq!(edge.coincidences(&[1.0], &[0.7499999]).unwrap(), vec![0]);
        // Several spikes inside one window are several pairs.
        assert_eq!(edge.coincidences(&[1.0, 1.1], &[0.9, 1.0, 1.2]).unwrap(), vec![6]);
        // The delay is applied to the RIGHT train, with its sign.
        let signed = Jeffress { delays: vec![-0.5, 0.5], window: 0.01 };
        assert_eq!(signed.coincidences(&[2.0], &[1.5]).unwrap(), vec![0, 1]);
    }

    #[test]
    fn above_the_aliasing_frequency_a_second_peak_appears_and_is_reported() {
        let max_itd = D / C; // 524 µs → unambiguous below 954 Hz.
        let limit = unambiguous_below(max_itd).unwrap();
        assert!((limit - C / (2.0 * D)).abs() < 1e-9);
        assert_eq!(unambiguous_below(0.0), None);
        // Detector spacing of 1/40 ms, so that a 2 kHz period (0.5 ms) is exactly 20 detectors.
        let array = Jeffress::uniform(0.5e-3, 41, 5e-6).unwrap();
        let mut rng = Rng::new(3);
        let run = |freq: f64, itd: f64, rng: &mut Rng| {
            let right = phase_locked(0.0, 1.0 / freq, 200, 0.0, rng).unwrap();
            let left = phase_locked(itd, 1.0 / freq, 200, 0.0, rng).unwrap();
            array.coincidences(&left, &right).unwrap()
        };
        // 2 kHz, source at +0.25 ms: detectors at +0.25 ms (index 30) and −0.25 ms (index 10) are
        // one period apart. The alias pairs cycle n with n + 1, so it is one short of full height…
        let counts = run(2000.0, 0.25e-3, &mut rng);
        assert_eq!((counts[30], counts[10]), (200, 199));
        // …which a finite recording can see, by one cycle in two hundred. The estimate reads the
        // taller peak and reports that there are two.
        let right = phase_locked(0.0, 0.5e-3, 200, 0.0, &mut rng).unwrap();
        let left = phase_locked(0.25e-3, 0.5e-3, 200, 0.0, &mut rng).unwrap();
        let est = array.estimate(&left, &right).unwrap();
        assert_eq!((est.count, est.peaks), (200, 2));
        assert!((est.itd - 0.25e-3).abs() < 1e-12);
        // At 800 Hz (period 1.25 ms, longer than the whole array) there is one peak and nothing
        // else anywhere.
        let low = run(800.0, 0.25e-3, &mut rng);
        assert_eq!(low[30], 200);
        assert_eq!(low.iter().filter(|c| **c > 0).count(), 1);
        let right = phase_locked(0.0, 1.25e-3, 200, 0.0, &mut rng).unwrap();
        let left = phase_locked(0.25e-3, 1.25e-3, 200, 0.0, &mut rng).unwrap();
        assert_eq!(array.estimate(&left, &right).unwrap().peaks, 1);
        // A source that falls between every detector by more than the window is not read as zero.
        let sparse = Jeffress { delays: vec![-0.25e-3, 0.0, 0.25e-3], window: 5e-6 };
        let between = phase_locked(0.125e-3, 1.25e-3, 200, 0.0, &mut rng).unwrap();
        assert!(matches!(sparse.estimate(&between, &right), Err(LocaliseError::Empty { what: "coincidences" })));
        // A wide window makes a plateau of tied detectors; the reading is its middle.
        let wide = Jeffress::uniform(0.5e-3, 41, 30e-6).unwrap();
        let est = wide.estimate(&left, &right).unwrap();
        assert_eq!((est.count, est.peaks), (200, 1));
        assert!((est.itd - 0.25e-3).abs() < 1e-12, "plateau of 3 detectors centred on the truth: {}", est.itd);
    }

    #[test]
    fn resolution_is_finest_straight_ahead() {
        let delta = 10e-6;
        let ahead = azimuth_resolution(D, C, delta, 0.0).unwrap();
        assert!((ahead - C * delta / D).abs() < 1e-18);
        // 10 µs is 1.1° straight ahead and twice that at 60°.
        assert!((ahead.to_degrees() - 1.0925).abs() < 1e-3, "{}", ahead.to_degrees());
        let side = azimuth_resolution(D, C, delta, core::f64::consts::FRAC_PI_3).unwrap();
        assert!((side / ahead - 2.0).abs() < 1e-12);
        // And it is the derivative of the geometry: a small step in ITD moves the azimuth by it.
        let az = 0.6;
        let itd = itd_far_field(D, C, az).unwrap();
        let moved = azimuth_far_field(D, C, itd + 1e-9).unwrap() - az;
        assert!((moved / azimuth_resolution(D, C, 1e-9, az).unwrap() - 1.0).abs() < 1e-5);
        for none in [azimuth_resolution(D, C, delta, FRAC_PI_2), azimuth_resolution(0.0, C, delta, 0.0), azimuth_resolution(D, C, 0.0, 0.0), azimuth_resolution(D, C, delta, f64::NAN)] {
            assert_eq!(none, None);
        }
    }

    #[test]
    fn jittered_spikes_coincide_as_often_as_the_gaussian_says() {
        let (sigma, window, period, cycles) = (40e-6, 50e-6, 5e-3, 20_000usize);
        let array = Jeffress { delays: vec![0.0, 60e-6, 200e-6], window };
        let mut rng = Rng::new(9);
        let right = phase_locked(0.0, period, cycles, sigma, &mut rng).unwrap();
        let left = phase_locked(0.0, period, cycles, sigma, &mut rng).unwrap();
        let counts = array.coincidences(&left, &right).unwrap();
        for (k, &delay) in array.delays.iter().enumerate() {
            // The detector adds `delay` to the right train, so the pair's difference has mean −delay.
            let p = coincidence_probability(-delay, window, sigma).unwrap();
            let want = p * cycles as f64;
            let se = (cycles as f64 * p * (1.0 - p)).sqrt();
            assert!((counts[k] as f64 - want).abs() < 4.0 * se, "detector {k}: {} coincidences, {want} ± {se} expected", counts[k]);
        }
        // The closed form on its own terms: symmetric in the mismatch, and the matched detector
        // with window w = 50 µs against σ√2 = 56.6 µs catches erf(0.625) = 62.3% of pairs.
        let matched = coincidence_probability(0.0, window, sigma).unwrap();
        assert!((matched - crate::surrogate::erf(50.0 / (40.0 * 2.0))).abs() < 1e-12);
        assert!((matched - 0.6232).abs() < 1e-4, "{matched}");
        assert_eq!(coincidence_probability(30e-6, window, sigma), coincidence_probability(-30e-6, window, sigma));
        assert!(coincidence_probability(60e-6, window, sigma).unwrap() < matched);
        for none in [coincidence_probability(0.0, 0.0, sigma), coincidence_probability(0.0, window, 0.0), coincidence_probability(f64::NAN, window, sigma)] {
            assert_eq!(none, None);
        }
    }

    #[test]
    fn bad_arguments_are_refused() {
        assert!(matches!(itd_far_field(0.0, C, 0.1), Err(LocaliseError::OutOfRange { what: "d", .. })));
        assert!(matches!(itd_far_field(D, -1.0, 0.1), Err(LocaliseError::OutOfRange { what: "c", .. })));
        assert!(matches!(itd_far_field(D, C, 1.6), Err(LocaliseError::OutOfRange { what: "azimuth", .. })));
        assert!(matches!(itd_woodworth(0.0, C, 0.1), Err(LocaliseError::OutOfRange { what: "r", .. })));
        assert!(matches!(itd_woodworth(0.09, C, f64::NAN), Err(LocaliseError::OutOfRange { what: "azimuth", .. })));
        assert!(matches!(Jeffress::uniform(1e-3, 1, 1e-5), Err(LocaliseError::Empty { .. })));
        assert!(matches!(Jeffress::uniform(0.0, 5, 1e-5), Err(LocaliseError::OutOfRange { what: "max_itd", .. })));
        assert!(matches!(Jeffress::uniform(1e-3, 5, 0.0), Err(LocaliseError::OutOfRange { what: "window", .. })));
        let array = Jeffress::uniform(1e-3, 5, 1e-5).unwrap();
        assert_eq!(array.delays, vec![-1e-3, -0.5e-3, 0.0, 0.5e-3, 1e-3]);
        assert!(matches!(array.coincidences(&[0.2, 0.1], &[0.1]), Err(LocaliseError::Unsorted { what: "left", index: 1 })));
        assert!(matches!(array.coincidences(&[0.1], &[0.1, f64::NAN]), Err(LocaliseError::NonFinite { what: "right", index: 1 })));
        assert!(matches!(array.estimate(&[0.1], &[0.9]), Err(LocaliseError::Empty { what: "coincidences" })));
        assert!(matches!(array.estimate(&[], &[0.9]), Err(LocaliseError::Empty { what: "coincidences" })));
        let mut rng = Rng::new(4);
        assert!(matches!(phase_locked(0.0, 1e-3, 0, 0.0, &mut rng), Err(LocaliseError::Empty { .. })));
        assert!(matches!(phase_locked(f64::NAN, 1e-3, 5, 0.0, &mut rng), Err(LocaliseError::NonFinite { .. })));
        assert!(matches!(phase_locked(0.0, 0.0, 5, 0.0, &mut rng), Err(LocaliseError::OutOfRange { what: "period", .. })));
        assert!(matches!(phase_locked(0.0, 1e-3, 5, -1e-6, &mut rng), Err(LocaliseError::OutOfRange { what: "sigma", .. })));
        assert_eq!(phase_locked(1.0, 0.5, 3, 0.0, &mut rng).unwrap(), vec![1.0, 1.5, 2.0]);
        // Jitter larger than the period still comes back in time order.
        let wild = phase_locked(0.0, 1e-3, 200, 5e-3, &mut rng).unwrap();
        assert!(wild.windows(2).all(|p| p[0] <= p[1]));
        assert_eq!(Jeffress { delays: vec![0.0], window: 1.0 }.spacing(), 0.0);
    }

    /// Pins the kelvin zero of the temperature ratio, `331.3 √(1 + ϑ/273.15)`, in BOTH places it
    /// appears: the denominator and the absolute-zero cutoff. 273.15 is not a rounding — the
    /// Celsius scale is DEFINED by `T/K = ϑ/°C + 273.15`, so the ratio is `T/T₀` with `T₀` the
    /// kelvin temperature of the ice point, exactly. The suite could not see a transcription of
    /// 273.0: its only value check was `speed_of_sound(20.0)` to ±0.05 m/s, and the two readings
    /// differ there by a measured 6.4 mm/s, eight times inside that tolerance; its only cutoff
    /// check was at −273.15 °C itself, which both readings refuse.
    #[test]
    fn the_temperature_ratio_carries_the_kelvin_zero_to_the_hundredth_of_a_degree() {
        // Same operations in the same order as the module, so this is exact rather than close.
        for t in [20.0, 0.0, -40.0, -273.0, -273.1] {
            assert_eq!(speed_of_sound(t), Some(331.3 * (1.0 + t / 273.15).sqrt()), "{t} °C");
        }
        // The two transcriptions ARE distinguishable at room temperature: 343.21462 m/s against
        // 343.22105 m/s, a measured 6.4 mm/s, and 14.4 mm/s at −40 °C.
        assert!((speed_of_sound(20.0).unwrap() - 331.3 * (1.0 + 20.0 / 273.0_f64).sqrt()).abs() > 6e-3);
        assert!((speed_of_sound(-40.0).unwrap() - 331.3 * (1.0 + -40.0 / 273.0_f64).sqrt()).abs() > 1.4e-2);
        // Air at −273.1 °C is 0.05 K above absolute zero and still carries sound, slowly: the
        // cutoff is the ice point less 273.15, not less 273.
        assert!(speed_of_sound(-273.1).unwrap() > 0.0);
        assert_eq!(speed_of_sound(-273.15), None);
        assert_eq!(speed_of_sound(f64::NEG_INFINITY), None);
    }

    /// Pins that an INFINITE length, speed, period, span or window is refused by the shared
    /// positive guard, not only a zero or negative one. The suite could not see it: every refusal
    /// it checked was at 0.0 or −1.0, and `v > 0.0` on its own admits `f64::INFINITY` — which
    /// leaves an infinite ITD, an array whose every delay is `NaN` (`-∞ + ∞·k`), or a train of
    /// infinite spike times for something further downstream to trip over.
    #[test]
    fn an_infinite_length_speed_period_span_or_window_is_refused_like_a_negative_one() {
        let mut rng = Rng::new(11);
        assert!(matches!(itd_far_field(f64::INFINITY, C, 0.1), Err(LocaliseError::OutOfRange { what: "d", .. })));
        assert!(matches!(itd_far_field(D, f64::INFINITY, 0.1), Err(LocaliseError::OutOfRange { what: "c", .. })));
        assert!(matches!(itd_woodworth(f64::INFINITY, C, 0.1), Err(LocaliseError::OutOfRange { what: "r", .. })));
        assert!(matches!(itd_woodworth(0.09, f64::INFINITY, 0.1), Err(LocaliseError::OutOfRange { what: "c", .. })));
        assert!(matches!(Jeffress::uniform(f64::INFINITY, 5, 1e-5), Err(LocaliseError::OutOfRange { what: "max_itd", .. })));
        assert!(matches!(Jeffress::uniform(1e-3, 5, f64::INFINITY), Err(LocaliseError::OutOfRange { what: "window", .. })));
        assert!(matches!(phase_locked(0.0, f64::INFINITY, 5, 0.0, &mut rng), Err(LocaliseError::OutOfRange { what: "period", .. })));
    }

    /// Pins the LOW BOUND a positive-guard refusal names: the smallest admissible value, which the
    /// module then accepts, and not the zero it has just refused. The suite could not see it —
    /// every refusal it matched ended in `..`, which reads the variant and the parameter name and
    /// never the range they carry, so a message offering the caller the one value that call
    /// rejected passed every assertion in the module.
    #[test]
    fn a_refused_non_positive_value_names_a_low_bound_the_guard_itself_accepts() {
        let Err(LocaliseError::OutOfRange { what, value, low, high }) = itd_far_field(0.0, C, 0.0) else {
            panic!("a zero head width is refused");
        };
        assert_eq!((what, value, low, high), ("d", 0.0, f64::MIN_POSITIVE, f64::MAX));
        // BOTH named ends are head widths the module accepts, and the value it refused is not.
        // The high end used to be `f64::INFINITY`, which this very guard rejects — a refusal that
        // offers the caller a value it will refuse again.
        assert!(itd_far_field(low, C, 0.0).is_ok());
        assert!(itd_far_field(high, C, 0.0).is_ok());
        assert!(itd_far_field(value, C, 0.0).is_err());
        assert!(itd_far_field(f64::INFINITY, C, 0.0).is_err(), "the old high end was inadmissible");
    }

    /// Pins what a refusal SAYS, which is the only part of an error a caller reads: the offending
    /// value, the admissible range low end first, and the name of what was empty. The suite could
    /// not see any of it — it matched the error variant and its `what` field and never called
    /// `Display`, so a range printed backwards or a message that drops the value entirely was
    /// invisible to every test in the module.
    #[test]
    fn a_refusal_prints_the_offending_value_and_its_range_low_end_first() {
        let too_far = itd_far_field(D, C, 1.6).unwrap_err();
        assert_eq!(too_far.to_string(), "azimuth = 1.6 is outside [-1.5707963267948966, 1.5707963267948966]");
        let LocaliseError::OutOfRange { low, high, .. } = too_far else {
            panic!("an azimuth past the side is out of range");
        };
        // The bounds it names are themselves admissible azimuths, and in that order.
        assert!(low < high);
        assert!(itd_far_field(D, C, low).is_ok() && itd_far_field(D, C, high).is_ok());
        let array = Jeffress::uniform(1e-3, 5, 1e-5).unwrap();
        let nothing = array.estimate(&[0.1], &[0.9]).unwrap_err();
        assert_eq!(nothing.to_string(), "coincidences is empty");
    }

    /// Pins that a non-finite spike time is reported AS non-finite even when it also breaks time
    /// order, and at its own index. The suite could not see it: its only non-finite train was
    /// `[0.1, NaN]`, and every ordered comparison with `NaN` is false, so the order check could
    /// not reach it whichever check ran first. An infinity does compare, and would have come back
    /// "out of time order" — sending the caller to look at the sorting of a train whose defect is
    /// that one of its times is not a time.
    #[test]
    fn a_non_finite_spike_is_named_non_finite_even_when_it_also_breaks_time_order() {
        let array = Jeffress::uniform(1e-3, 5, 1e-5).unwrap();
        assert_eq!(
            array.coincidences(&[0.0, f64::NEG_INFINITY], &[0.0]),
            Err(LocaliseError::NonFinite { what: "left", index: 1 })
        );
        assert_eq!(
            array.coincidences(&[0.0], &[f64::INFINITY, 0.5]),
            Err(LocaliseError::NonFinite { what: "right", index: 0 })
        );
    }

    /// Pins that two spikes in the SAME instant are in time order. Two sensors can register one
    /// wavefront in the same microsecond, a train quantised to a tick can repeat one, and
    /// `phase_locked` itself returns ties once the jitter is large — so a train that repeats a
    /// time is data, and only one that goes BACKWARDS is a defect. The suite could not see it:
    /// every train it built was strictly increasing, which a `<=` order check accepts while
    /// refusing every tie.
    #[test]
    fn two_spikes_that_arrive_in_the_same_instant_are_in_time_order() {
        let detector = Jeffress { delays: vec![0.0], window: 0.25 };
        assert_eq!(detector.coincidences(&[1.0, 1.0], &[1.0]).unwrap(), vec![2]);
        assert_eq!(detector.coincidences(&[1.0], &[1.0, 1.0]).unwrap(), vec![2]);
        // A train that does go backwards is still refused, at the index of the early spike.
        assert_eq!(
            detector.coincidences(&[1.0, 1.0, 0.9], &[1.0]),
            Err(LocaliseError::Unsorted { what: "left", index: 2 })
        );
    }

    /// Pins the three weakenings of the far-field inverse's guard that return a NUMBER instead of
    /// nothing: a zero head width with a zero ITD (`c·0/0` is `NaN`, and `NaN > 1.0` is false, so
    /// the magnitude check passes it straight through to `asin`), a zero speed of sound (which
    /// reads every ITD in the world as 0 = straight ahead), and a `NaN` ITD. The suite could not
    /// see them: its zero-head case used a NON-zero ITD, which divides to an infinity the
    /// magnitude check happens to refuse for it, and it never passed a non-finite ITD at all.
    #[test]
    fn the_far_field_inverse_refuses_a_zero_head_a_zero_speed_and_a_non_finite_itd() {
        assert_eq!(azimuth_far_field(0.0, C, 0.0), None);
        assert_eq!(azimuth_far_field(0.0, C, 1e-4), None);
        assert_eq!(azimuth_far_field(D, 0.0, 1e-4), None);
        assert_eq!(azimuth_far_field(D, 0.0, 0.0), None);
        assert_eq!(azimuth_far_field(D, C, f64::NAN), None);
        assert_eq!(azimuth_far_field(D, C, f64::INFINITY), None);
    }

    /// Pins that an ITD of exactly `d/c` — a source at exactly 90°, the largest the geometry can
    /// produce — HAS an azimuth, `±π/2`, rather than being refused as past the head. The suite
    /// could not see it: it checked the refusal at 1.01·d/c and round trips only out to 1.5 rad,
    /// and `asin` is flat at the end point, so a boundary that throws the end point away moves
    /// nothing any other assertion in the module reads.
    #[test]
    fn an_itd_exactly_at_the_head_s_limit_is_the_source_at_ninety_degrees() {
        // 0.25 m and 2 m/s make d/c exactly 0.125 s, so `c·ITD/d` is exactly 1.0 and the
        // assertion is about the boundary rather than about a rounding of it.
        let itd = itd_far_field(0.25, 2.0, FRAC_PI_2).unwrap();
        assert_eq!(itd, 0.125);
        assert_eq!(azimuth_far_field(0.25, 2.0, itd), Some(FRAC_PI_2));
        assert_eq!(azimuth_far_field(0.25, 2.0, -itd), Some(-FRAC_PI_2));
        // One ulp further out there is no azimuth: that ITD is not a sound the head can make.
        assert_eq!(azimuth_far_field(0.25, 2.0, f64::from_bits(itd.to_bits() + 1)), None);
    }

    /// Pins that a Woodworth ITD of exactly `(r/c)(π/2 + 1)` — the sphere's largest — has an
    /// azimuth rather than being refused. The suite could not see it: it checked the refusal at
    /// 1.01× the limit, and its round trips stop at 1.5 rad, so the end point of the bisection's
    /// bracket was never asked for.
    #[test]
    fn a_woodworth_itd_exactly_at_the_sphere_s_limit_has_an_azimuth() {
        // r = c = 1 makes the ITD and the bisection's target the same number, so the comparison
        // under test sees the limit exactly.
        let limit = FRAC_PI_2 + 1.0;
        let theta = azimuth_woodworth(1.0, 1.0, limit).expect("the sphere's largest ITD is 90°");
        // The `f64` spacing at the ITD (≈2.57) is twice the spacing at the angle (≈1.57), so two
        // representable angles share each representable ITD and 2 ulps is as close as any inverse
        // of this can come; the bisection's own bracket, `π/2 · 2⁻⁶⁰`, is far below that.
        assert!(FRAC_PI_2 - theta <= 2.0 * f64::EPSILON, "{theta}");
        assert_eq!(azimuth_woodworth(1.0, 1.0, -limit), Some(-theta));
        assert_eq!(azimuth_woodworth(1.0, 1.0, f64::from_bits(limit.to_bits() + 1)), None);
    }

    /// Pins that an INFINITE array span has no aliasing limit rather than a limit of 0 Hz. The
    /// suite could not see it: it checked a zero span and finite ones, and `0.5/∞` is a clean
    /// `0.0` that reads as a frequency — one saying every tone whatsoever aliases, which is the
    /// opposite of what an array of unbounded reach does.
    #[test]
    fn an_infinite_array_span_has_no_aliasing_limit_rather_than_a_limit_of_zero_hertz() {
        assert_eq!(unambiguous_below(f64::INFINITY), None);
        assert_eq!(unambiguous_below(f64::NAN), None);
        assert_eq!(unambiguous_below(-1e-3), None);
        assert_eq!(unambiguous_below(0.5e-3), Some(1000.0));
    }

    /// Pins that a non-finite jitter is REFUSED rather than read as no jitter. The suite could not
    /// see it: it checked a negative sigma only, and with `sigma < 0.0` as the whole guard a `NaN`
    /// passes it, then fails `sigma > 0.0` as well — so the caller who asked for jitter gets a
    /// clean deterministic train back, which is indistinguishable from a correct one until the
    /// coincidence counts it feeds are compared against the Gaussian.
    #[test]
    fn a_non_finite_jitter_is_refused_rather_than_read_as_no_jitter() {
        let mut rng = Rng::new(12);
        assert!(matches!(phase_locked(0.0, 1e-3, 5, f64::NAN, &mut rng), Err(LocaliseError::OutOfRange { what: "sigma", .. })));
        assert!(matches!(phase_locked(0.0, 1e-3, 5, f64::INFINITY, &mut rng), Err(LocaliseError::OutOfRange { what: "sigma", .. })));
    }

    /// Pins that the zero-jitter path draws NOTHING from the rng. The suite could not see it:
    /// `0.0 * gaussian(rng)` is `0.0`, so the train that comes back is identical either way and
    /// only the generator's position moves — which silently changes every jittered train made
    /// later from the same generator, and this module's determinism is the claim that a seed fixes
    /// a spike train.
    #[test]
    fn the_zero_jitter_path_draws_nothing_from_the_rng() {
        let mut mixed = Rng::new(13);
        assert_eq!(phase_locked(0.0, 0.25, 4, 0.0, &mut mixed).unwrap(), vec![0.0, 0.25, 0.5, 0.75]);
        let after = phase_locked(0.0, 1e-3, 5, 1e-5, &mut mixed).unwrap();
        let mut fresh = Rng::new(13);
        assert_eq!(after, phase_locked(0.0, 1e-3, 5, 1e-5, &mut fresh).unwrap());
    }

    /// Pins that a detector whose half-window is NEGATIVE is REFUSED, naming the field.
    ///
    /// `Jeffress::uniform` validates `window`, but the field is `pub`, so a struct literal or a
    /// later assignment bypasses that — and `coincidences` used to accept it and answer `0`. The
    /// two-pointer walk survived only because `if hi < lo { hi = lo }` pulls the high pointer
    /// forward; without that line `hi - lo` wraps and the detector reports
    /// 18,446,744,073,709,551,615 coincidences. A plausible zero and an absurd count are both
    /// wrong answers to a meaningless question, and invariant 10 of this crate's own guidance says
    /// an input the code cannot understand is an error naming what was sent.
    ///
    /// The suite could not see this: with a non-negative window the pull-forward is unreachable —
    /// every right spike the low pointer skipped satisfies the high pointer's condition too, since
    /// `t − w ≤ t + w` — so the one input that reaches it is the one no test used.
    #[test]
    fn a_negative_half_window_is_refused_rather_than_answered() {
        let backwards = Jeffress { delays: vec![0.0], window: -1.0 };
        assert!(matches!(
            backwards.coincidences(&[0.0, 10.0], &[0.0]),
            Err(LocaliseError::OutOfRange { what: "window", .. })
        ));
        // Zero is refused for the same reason `uniform` refuses it, and a positive one works.
        let zero = Jeffress { delays: vec![0.0], window: 0.0 };
        assert!(zero.coincidences(&[0.0], &[0.0]).is_err());
        let ok = Jeffress { delays: vec![0.0], window: 0.01 };
        assert_eq!(ok.coincidences(&[0.0, 1.0, 2.0], &[0.0, 1.0, 2.0]).unwrap(), vec![3]);
    }

    /// Pins that the reading is the mean of the FIRST RUN of tied detectors, not of every tied
    /// detector in the array. The suite could not see it: every tie it produced was contiguous —
    /// a wide window's plateau — where a run and a filter count the same detectors. Two equally
    /// tall peaks apart from each other separate them, and the filtered count then averages a
    /// slice that starts at the first peak and runs across the valley between them.
    #[test]
    fn the_reading_is_the_mean_of_the_first_tied_run_not_of_every_tied_detector() {
        // Two left spikes a second apart against one right spike: the detectors at 0 and at 1 s
        // each catch one, the detector between them catches nothing.
        let array = Jeffress { delays: vec![0.0, 0.5, 1.0], window: 0.01 };
        assert_eq!(array.coincidences(&[0.0, 1.0], &[0.0]).unwrap(), vec![1, 0, 1]);
        let est = array.estimate(&[0.0, 1.0], &[0.0]).unwrap();
        assert_eq!((est.itd, est.count, est.peaks), (0.0, 1, 2));
    }

    /// Pins the full-width-at-HALF-maximum convention [`Estimate::peaks`] states: a detector at a
    /// quarter of the largest count is not a peak, one at exactly half is. The suite could not see
    /// it: every count it made away from a peak was 0 or within one of the maximum, and both of
    /// those fall the same side of a half and a quarter threshold alike — so the module would have
    /// reported an alias at a quarter height, which is a second source that is not there.
    #[test]
    fn a_separate_peak_reaches_half_the_largest_count_and_a_quarter_of_it_does_not() {
        let array = Jeffress { delays: vec![0.0, 1.0, 2.0], window: 0.01 };
        let four = [0.0, 0.001, 0.002, 0.003];
        let quarter: Vec<f64> = four.iter().copied().chain([2.0]).collect();
        assert_eq!(array.coincidences(&quarter, &[0.0]).unwrap(), vec![4, 0, 1]);
        assert_eq!(array.estimate(&quarter, &[0.0]).unwrap().peaks, 1);
        let half: Vec<f64> = four.iter().copied().chain([2.0, 2.001]).collect();
        assert_eq!(array.coincidences(&half, &[0.0]).unwrap(), vec![4, 0, 2]);
        assert_eq!(array.estimate(&half, &[0.0]).unwrap().peaks, 2);
    }
}
