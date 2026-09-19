//! Delays as a resource: a neuron whose synapses arrive at different times detects a PATTERN IN
//! TIME, a learning rule that moves the delays makes it detect the pattern it is shown, and the
//! number of patterns a set of delays can stand for is a count — each checked exactly.
//!
//! # What the mechanism is
//!
//! An axon takes time. If synapse `i` delays its spike by `d_i`, a presynaptic volley with spike
//! times `t_i` arrives at `t_i + d_i`, and the arrivals coincide exactly when `t_i = T − d_i`: the
//! delays are a template, and the neuron is a matched filter for one spatiotemporal pattern and
//! blind to the same spikes in another order. Izhikevich (*Polychronization: computation with
//! spikes*, Neural Computation 18(2):245–282, 2006) built a theory of memory on this — groups of
//! neurons that fire in reproducible time-locked, not synchronous, patterns — and observed that
//! the number of such groups can exceed the number of neurons.
//!
//! Delays can be learned. The simplest rule shifts each delay against its arrival's deviation
//! from the volley's mean arrival, `Δd_i = −η (a_i − ā)` (in the family studied by Hüning, Glünder
//! and Palm, *Synaptic delay learning in pulse-coupled neurons*, Neural Computation
//! 10(3):555–565, 1998, and Eurich, Pawelzik, Ernst, Cowan and Milton, *Dynamics of self-organized
//! delay adaptation*, Physical Review Letters 82(7):1594–1597, 1999). Its modern descendant learns
//! delays by gradient descent (Hammouamri, Khalfaoui-Hassani and Masquelier, *Learning delays in
//! spiking neural networks using dilated convolutions with learnable spacings*, ICLR 2024).
//!
//! # Why it is in a neuromorphic crate
//!
//! Programmable synaptic delay is a hardware feature — a ring buffer per synapse or per axon — and
//! what it costs is memory that grows with the longest delay, which [`buffer_bits`] counts. What
//! it buys is temporal selectivity with no extra neurons.
//!
//! # The closed forms this module is checked against
//!
//! - **The matched pattern** `t_i = T − d_i` arrives all at once; the SAME spikes in reversed
//!   order arrive spread over exactly `2 (d_max − d_min)`.
//! - **Delay learning is a geometric contraction.** Every arrival's deviation from the mean
//!   arrival is multiplied by exactly `1 − η` per presentation; the mean arrival and the sum of
//!   the delays are conserved; the delays converge on `d_i(0) − (a_i(0) − ā(0))`.
//! - **Capacity is a count.** With `n` synapses and whole delays in `0..=D`, the patterns
//!   distinguishable up to a common shift number `(D+1)ⁿ − Dⁿ` — the delay vectors whose smallest
//!   entry is zero — checked against brute-force enumeration.
//! - **Cost.** A ring buffer of one bit per tick costs `Σ_i d_i` bits per-synapse, or `d_max` bits
//!   per axon when synapses share their axon's line.
//!
//! # What this module has NOT reproduced
//!
//! - Izhikevich's network simulation or his counts of polychronous groups, which are empirical.
//! - Gradient-based delay learning (the dilated-convolution method) and its benchmark results.
//! - Delays that are clipped to a hardware range DURING learning: the contraction above is the
//!   unclipped rule, and [`DelayNeuron::learn`] reports when a bound was hit, because then it is
//!   no longer that rule.

use core::fmt;

/// What went wrong, named rather than guessed around.
#[derive(Debug, Clone, PartialEq)]
pub enum DelayError {
    /// A count of zero where at least one is needed.
    Empty {
        /// What was empty.
        what: &'static str,
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
        /// Position in the offending array.
        index: usize,
    },
}

impl fmt::Display for DelayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty { what } => write!(f, "{what} is empty"),
            Self::Dimension { what, got, want } => write!(f, "{what} has length {got}, expected {want}"),
            Self::OutOfRange { what, value, low, high } => {
                write!(f, "{what} = {value} is outside [{low}, {high}]")
            }
            Self::NonFinite { what, index } => write!(f, "{what} is not finite at {index}"),
        }
    }
}

impl std::error::Error for DelayError {}

fn finite_all(what: &'static str, v: &[f64]) -> Result<(), DelayError> {
    if let Some(i) = v.iter().position(|x| !x.is_finite()) {
        return Err(DelayError::NonFinite { what, index: i });
    }
    Ok(())
}

/// A neuron with one delayed synapse per input.
#[derive(Debug, Clone, PartialEq)]
pub struct DelayNeuron {
    /// Synaptic delays, seconds, each in `[0, max_delay]`.
    pub delays: Vec<f64>,
    /// The longest delay the hardware can hold, seconds.
    pub max_delay: f64,
}

/// What one learning presentation did.
#[derive(Debug, Clone, PartialEq)]
pub struct Presentation {
    /// The spread of arrival times (latest minus earliest) BEFORE the update, seconds.
    pub spread: f64,
    /// The mean arrival time before the update, seconds.
    pub mean_arrival: f64,
    /// Delays that hit `0` or `max_delay` and were clipped: when non-zero this step was not the
    /// contraction the module doc describes.
    pub clipped: usize,
}

impl DelayNeuron {
    /// Build.
    ///
    /// # Errors
    ///
    /// [`DelayError::Empty`] for no synapses, [`DelayError::OutOfRange`] for a non-positive
    /// `max_delay` or a delay outside `[0, max_delay]`, [`DelayError::NonFinite`] for a bad delay.
    pub fn new(delays: Vec<f64>, max_delay: f64) -> Result<Self, DelayError> {
        if delays.is_empty() {
            return Err(DelayError::Empty { what: "synapses" });
        }
        if !(max_delay > 0.0) || !max_delay.is_finite() {
            return Err(DelayError::OutOfRange { what: "max_delay", value: max_delay, low: f64::MIN_POSITIVE, high: f64::INFINITY });
        }
        finite_all("delays", &delays)?;
        if let Some(&d) = delays.iter().find(|d| **d < 0.0 || **d > max_delay) {
            return Err(DelayError::OutOfRange { what: "delay", value: d, low: 0.0, high: max_delay });
        }
        Ok(Self { delays, max_delay })
    }

    /// Arrival times `t_i + d_i` of a volley with one spike per synapse.
    ///
    /// # Errors
    ///
    /// [`DelayError::Dimension`] for a volley of the wrong size, [`DelayError::NonFinite`] for a bad
    /// spike time.
    pub fn arrivals(&self, spikes: &[f64]) -> Result<Vec<f64>, DelayError> {
        if spikes.len() != self.delays.len() {
            return Err(DelayError::Dimension { what: "spikes", got: spikes.len(), want: self.delays.len() });
        }
        finite_all("spikes", spikes)?;
        Ok(spikes.iter().zip(&self.delays).map(|(t, d)| t + d).collect())
    }

    /// The spread of a volley's arrivals, latest minus earliest, seconds: zero for the matched
    /// pattern.
    ///
    /// # Errors
    ///
    /// As [`DelayNeuron::arrivals`].
    pub fn spread(&self, spikes: &[f64]) -> Result<f64, DelayError> {
        let a = self.arrivals(spikes)?;
        let (lo, hi) = a.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &x| (lo.min(x), hi.max(x)));
        Ok(hi - lo)
    }

    /// The largest number of arrivals inside any window of `window` seconds (closed at both ends):
    /// what a coincidence detector with that integration time would see.
    ///
    /// # Errors
    ///
    /// As [`DelayNeuron::arrivals`], plus [`DelayError::OutOfRange`] for a negative window.
    pub fn coincident(&self, spikes: &[f64], window: f64) -> Result<usize, DelayError> {
        if !(window >= 0.0) {
            return Err(DelayError::OutOfRange { what: "window", value: window, low: 0.0, high: f64::INFINITY });
        }
        let mut a = self.arrivals(spikes)?;
        a.sort_by(f64::total_cmp);
        let mut best = 0;
        let mut lo = 0;
        for hi in 0..a.len() {
            while a[hi] - a[lo] > window {
                lo += 1;
            }
            best = best.max(hi - lo + 1);
        }
        Ok(best)
    }

    /// The volley this neuron is matched to, arriving all together at `at`: `t_i = at − d_i`.
    #[must_use]
    pub fn matched_pattern(&self, at: f64) -> Vec<f64> {
        self.delays.iter().map(|d| at - d).collect()
    }

    /// One presentation of the delay-shift rule, `d_i ← d_i − η (a_i − ā)`, clipped to
    /// `[0, max_delay]`.
    ///
    /// # Errors
    ///
    /// As [`DelayNeuron::arrivals`], plus [`DelayError::OutOfRange`] for an `eta` outside `(0, 1]`.
    pub fn learn(&mut self, spikes: &[f64], eta: f64) -> Result<Presentation, DelayError> {
        if !(eta > 0.0) || !(eta <= 1.0) {
            return Err(DelayError::OutOfRange { what: "eta", value: eta, low: f64::MIN_POSITIVE, high: 1.0 });
        }
        let a = self.arrivals(spikes)?;
        let mean_arrival = a.iter().sum::<f64>() / a.len() as f64;
        let (lo, hi) = a.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &x| (lo.min(x), hi.max(x)));
        let mut clipped = 0;
        for (d, ai) in self.delays.iter_mut().zip(&a) {
            let moved = *d - eta * (ai - mean_arrival);
            let held = moved.clamp(0.0, self.max_delay);
            clipped += usize::from(held != moved);
            *d = held;
        }
        Ok(Presentation { spread: hi - lo, mean_arrival, clipped })
    }
}

/// Patterns of `n` spikes distinguishable up to a common time shift by whole delays in `0..=d_max`:
/// `(D+1)ⁿ − Dⁿ`, the delay vectors whose smallest entry is zero. `None` on overflow of `u128`
/// or for `n = 0`.
#[must_use]
pub fn distinct_patterns(n: u32, d_max: u64) -> Option<u128> {
    if n == 0 {
        return None;
    }
    let all = u128::from(d_max).checked_add(1)?.checked_pow(n)?;
    let shifted = u128::from(d_max).checked_pow(n)?;
    Some(all - shifted)
}

/// Bits of ring buffer for a set of whole-tick delays: one bit per tick of delay. Per synapse that
/// is `Σ d_i`; when the synapses tap ONE axonal line it is `max d_i`. Returns
/// `(per_synapse, shared_line)`, saturating.
#[must_use]
pub fn buffer_bits(delays_ticks: &[u64]) -> (u64, u64) {
    let sum = delays_ticks.iter().fold(0u64, |a, d| a.saturating_add(*d));
    (sum, delays_ticks.iter().copied().max().unwrap_or(0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_matched_pattern_arrives_at_once_and_its_reverse_does_not() {
        let cell = DelayNeuron::new(vec![1e-3, 4e-3, 9e-3, 2.5e-3], 10e-3).unwrap();
        let pattern = cell.matched_pattern(20e-3);
        for (t, want) in pattern.iter().zip([19e-3, 16e-3, 11e-3, 17.5e-3]) {
            assert!((t - want).abs() < 1e-17);
        }
        assert!(cell.spread(&pattern).unwrap() < 1e-17);
        assert_eq!(cell.coincident(&pattern, 1e-6).unwrap(), 4);
        // The same four spikes, the same intervals, played backwards in time about t = 15 ms:
        // t_i' = 30 ms − t_i = 10 ms + d_i, so arrivals are 10 ms + 2 d_i, spread 2·(9 − 1) ms.
        let reversed: Vec<f64> = pattern.iter().map(|t| 30e-3 - t).collect();
        assert!((cell.spread(&reversed).unwrap() - 16e-3).abs() < 1e-17);
        assert_eq!(cell.coincident(&reversed, 1e-6).unwrap(), 1);
        // A window of 3 ms catches the two reversed arrivals 3 ms apart (d = 1 and 2.5 ms) and no
        // third: arrivals are 12, 18, 28, 15 ms.
        assert_eq!(cell.coincident(&reversed, 3e-3).unwrap(), 2);
        assert_eq!(cell.coincident(&reversed, 2.9e-3).unwrap(), 1);
        assert_eq!(cell.coincident(&reversed, 6e-3).unwrap(), 3);
        assert_eq!(cell.arrivals(&[0.0; 4]).unwrap(), cell.delays);
        // The window is CLOSED: arrivals exactly a window apart coincide. In binary fractions, so
        // that "exactly" is exact — 15 ms − 12 ms above is 2.9999999999999996 ms, which is why an
        // open window survived this module's first mutation sweep.
        let exact = DelayNeuron::new(vec![0.0, 0.25, 0.5], 1.0).unwrap();
        assert_eq!(exact.coincident(&[0.0; 3], 0.25).unwrap(), 2);
        assert_eq!(exact.coincident(&[0.0; 3], 0.5).unwrap(), 3);
        assert_eq!(exact.coincident(&[0.0; 3], 0.0).unwrap(), 1);
    }

    #[test]
    fn delay_learning_contracts_every_deviation_by_one_minus_eta() {
        let start = vec![2e-3, 7e-3, 4e-3, 5e-3, 3e-3];
        let mut cell = DelayNeuron::new(start.clone(), 20e-3).unwrap();
        let volley = [5e-3, 1e-3, 6e-3, 2e-3, 4.5e-3];
        let a0 = cell.arrivals(&volley).unwrap();
        let mean0 = a0.iter().sum::<f64>() / 5.0;
        let eta = 0.25;
        for k in 0..40 {
            let a = cell.arrivals(&volley).unwrap();
            for i in 0..5 {
                let want = (a0[i] - mean0) * 0.75f64.powi(k);
                assert!((a[i] - mean0 - want).abs() < 1e-17, "presentation {k}, synapse {i}");
            }
            let seen = cell.learn(&volley, eta).unwrap();
            assert_eq!(seen.clipped, 0);
            assert!((seen.mean_arrival - mean0).abs() < 1e-17, "the mean arrival moved");
            let (lo, hi) = a.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), &x| (l.min(x), h.max(x)));
            assert_eq!(seen.spread, hi - lo);
            assert!((cell.delays.iter().sum::<f64>() - start.iter().sum::<f64>()).abs() < 1e-17, "the total delay moved");
        }
        // It converges on d_i(0) − (a_i(0) − ā(0)), and the volley it was shown is now its pattern.
        for i in 0..5 {
            assert!((cell.delays[i] - (start[i] - (a0[i] - mean0))).abs() < 1e-7);
        }
        assert!(cell.spread(&volley).unwrap() < 1e-7);
        assert_eq!(cell.coincident(&volley, 1e-6).unwrap(), 5);
        // η = 1 does it in one presentation.
        let mut fast = DelayNeuron::new(start, 20e-3).unwrap();
        fast.learn(&volley, 1.0).unwrap();
        assert!(fast.spread(&volley).unwrap() < 1e-17);
    }

    #[test]
    fn a_delay_that_hits_its_bound_is_reported() {
        // Arrivals 0.5 and 19.5 ms, mean 10: the rule asks for delays of exactly 10 and 0 ms — ON the
        // walls, which is allowed and is not a clip.
        let mut cell = DelayNeuron::new(vec![0.5e-3, 9.5e-3], 10e-3).unwrap();
        assert_eq!(cell.learn(&[0.0, 10e-3], 1.0).unwrap().clipped, 0);
        assert!(cell.spread(&[0.0, 10e-3]).unwrap() < 1e-17);
        // Two milliseconds more between the spikes and it asks for 11 and −1 ms: both are clipped,
        // the step says so, and the volley is left 2 ms apart — the hardware cannot match it.
        let mut cell = DelayNeuron::new(vec![0.5e-3, 9.5e-3], 10e-3).unwrap();
        let seen = cell.learn(&[0.0, 12e-3], 1.0).unwrap();
        assert_eq!(seen.clipped, 2);
        assert_eq!(cell.delays, vec![10e-3, 0.0]);
        assert!((cell.spread(&[0.0, 12e-3]).unwrap() - 2e-3).abs() < 1e-17);
    }

    #[test]
    fn capacity_is_the_count_of_delay_vectors_whose_smallest_entry_is_zero() {
        for n in 1..=4u32 {
            for d_max in 0..=5u64 {
                let mut count = 0u128;
                let total = (d_max + 1).pow(n);
                for code in 0..total {
                    let (mut c, mut least) = (code, u64::MAX);
                    for _ in 0..n {
                        least = least.min(c % (d_max + 1));
                        c /= d_max + 1;
                    }
                    count += u128::from(least == 0);
                }
                assert_eq!(distinct_patterns(n, d_max), Some(count), "n = {n}, D = {d_max}");
            }
        }
        // Eight synapses with 31 ticks of delay: 32⁸ − 31⁸ ≈ 2.5e11 patterns from one neuron.
        assert_eq!(distinct_patterns(8, 31), Some(1_099_511_627_776 - 852_891_037_441));
        assert_eq!(distinct_patterns(0, 5), None);
        assert_eq!(distinct_patterns(40, u64::MAX), None);
        assert_eq!(buffer_bits(&[3, 0, 31, 8]), (42, 31));
        assert_eq!(buffer_bits(&[]), (0, 0));
        assert_eq!(buffer_bits(&[u64::MAX, 5]), (u64::MAX, u64::MAX));
    }

    #[test]
    fn bad_arguments_are_refused() {
        assert!(matches!(DelayNeuron::new(vec![], 1e-2), Err(DelayError::Empty { .. })));
        assert!(matches!(DelayNeuron::new(vec![1e-3], 0.0), Err(DelayError::OutOfRange { what: "max_delay", .. })));
        assert!(matches!(DelayNeuron::new(vec![-1e-3], 1e-2), Err(DelayError::OutOfRange { what: "delay", .. })));
        assert!(matches!(DelayNeuron::new(vec![2e-2], 1e-2), Err(DelayError::OutOfRange { what: "delay", .. })));
        assert!(matches!(DelayNeuron::new(vec![f64::NAN], 1e-2), Err(DelayError::NonFinite { what: "delays", index: 0 })));
        let mut cell = DelayNeuron::new(vec![1e-3, 2e-3], 1e-2).unwrap();
        assert!(matches!(cell.arrivals(&[0.0]), Err(DelayError::Dimension { what: "spikes", got: 1, want: 2 })));
        assert!(matches!(cell.spread(&[0.0, f64::INFINITY]), Err(DelayError::NonFinite { what: "spikes", index: 1 })));
        assert!(matches!(cell.coincident(&[0.0, 0.0], -1.0), Err(DelayError::OutOfRange { what: "window", .. })));
        assert!(matches!(cell.coincident(&[0.0, 0.0], f64::NAN), Err(DelayError::OutOfRange { what: "window", .. })));
        assert!(matches!(cell.learn(&[0.0, 0.0], 0.0), Err(DelayError::OutOfRange { what: "eta", .. })));
        assert!(matches!(cell.learn(&[0.0, 0.0], 1.5), Err(DelayError::OutOfRange { what: "eta", .. })));
        assert!(matches!(cell.learn(&[0.0], 0.5), Err(DelayError::Dimension { .. })));
        assert_eq!(cell.delays, vec![1e-3, 2e-3], "a refused presentation moved nothing");
    }
}
