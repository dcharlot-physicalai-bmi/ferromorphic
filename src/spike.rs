//! Spikes and spike trains: the one representation everything else in this crate agrees on.
//!
//! A spike carries almost nothing. It has a source, a time, and — on a sensor — a sign. All the
//! information is in *when* it arrived and *which* line it arrived on, which is what makes the
//! representation cheap to move and awkward to reason about. This module fixes the conventions so
//! the rest of the crate does not have to re-decide them.
//!
//! # Time is an integer tick, not a float
//!
//! [`Spike::t`] is a `u64` tick index, not a `f64` of seconds. Two reasons, and the second is the
//! one that matters.
//!
//! First, spike times are compared and sorted constantly, and floating-point time makes equality
//! ill-defined exactly where the model needs it to be sharp: two spikes arriving "at the same
//! time" is a physically meaningful statement on a synchronous fabric, and `0.003 == 0.003` is not
//! reliably true after a few additions.
//!
//! Second, and this is the real reason: **an accumulated float time drifts**. A simulation that
//! advances `t += dt` a million times has added a million rounding errors into the quantity every
//! spike time is measured against. At `dt = 1e-6` and `f64`, that drift is small; at `f32` on a
//! microcontroller, which is where this crate expects to run, it is not. An integer tick multiplied
//! by `dt` at the point of use has exactly one rounding, and it is the same one every time.
//!
//! # Address-event representation
//!
//! The address-event representation is Misha Mahowald's (M. Mahowald, *VLSI analogs of neuronal
//! visual processing: a synthesis of form and function*, doctoral dissertation, California
//! Institute of Technology (1992), §3.3 "The Address-Event Representation", p. 84,
//! doi:10.7907/4bdw-fg34; the same text was also issued as Caltech technical report CS-TR-92-15,
//! doi:10.7907/Z9CZ35CD). In her scheme a spike is its address and nothing more: "Whenever a neuron
//! signals an event, the multiplexing circuitry broadcasts that neuron's address on the inter-chip
//! data bus", and "I have chosen to transmit only the neuron address, which corresponds to a
//! digital amplitude event". Time is not a field. It is the moment the address appears on the bus,
//! and "the detailed timing of the events is preserved" (Figure 3.1). This review did not locate a
//! polarity field in §3.3. The explicit `(time, address, polarity)` triple is the form event
//! cameras record: the time written down, and a sign for a sensor that reports a change in either
//! direction. [`Event`] is that triple, and [`Spike`] is the same thing without a polarity for the
//! many places inside a network where there is only one kind. [`crate::aer`] reads and writes the
//! file formats.
//!
//! # How irregular a train is, locally
//!
//! [`Train::cv`] is the global coefficient of variation, which reads a slow change of rate as
//! irregularity. [`local_variation`] (Shinomoto, Shima and Tanji 2003), its refractoriness-corrected
//! [`revised_local_variation`] (Shinomoto et al. 2009) and Holt et al.'s [`cv2`] compare each
//! interval only with its neighbour. All three return the same doubles as `Elephant`'s `lv`, `lvr`
//! and `cv2` on that library's own examples, and `Lv` meets its closed form for gamma intervals.
//! ⚠ `Elephant`'s `lvr` example passes `R=0.005` without units, and the function then assumes
//! milliseconds: the example is `R` = 0.005 ms, not the 5 ms the paper recommends. And its `lv` and
//! `cv2` docstrings print the prefactor as `1/N` where the code, and Shinomoto's Eq. 2.2, divide by
//! `N − 1`.
//!
//! This section used to call the `(time, address, polarity)` triple "Mahowald's address-event
//! representation, 1992, and every event camera since", with no title or identifier. The author and
//! the year were right. The work went unnamed, and the sentence credited Mahowald with the whole
//! triple, polarity included.

/// One spike: a neuron fired at a tick.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Spike {
    /// Tick index. Multiply by the simulation's `dt` to get seconds, once, at the point of use.
    ///
    /// Ordered first in the struct so that the derived `Ord` sorts by time before address, which
    /// is the order every consumer in this crate wants.
    pub t: u64,
    /// Index of the neuron that fired.
    pub source: u32,
}

/// A sensor event: a spike that also carries a sign.
///
/// The address-event representation as event cameras emit it. The polarity distinguishes a
/// brightness increase from a decrease, and losing it — by taking the absolute value, or by feeding
/// both signs into one channel — throws away half the signal while leaving the event rate
/// unchanged, which is a failure that looks like nothing at all in a spike raster.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Event {
    /// Tick index.
    pub t: u64,
    /// Address: which sensor element fired. For a 2-D array this is the flattened `y * w + x`.
    pub address: u32,
    /// Sign of the change that produced this event.
    pub polarity: Polarity,
}

/// Which direction a sensor's signal moved.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Polarity {
    /// The signal fell past the threshold.
    Off,
    /// The signal rose past the threshold.
    On,
}

impl Polarity {
    /// `+1.0` for [`Polarity::On`], `-1.0` for [`Polarity::Off`].
    ///
    /// Provided so that a caller converting events into a current does not invent its own
    /// convention; a sign flip here is invisible in an event count and inverts every downstream
    /// receptive field.
    #[must_use]
    pub fn sign(self) -> f64 {
        match self {
            Self::On => 1.0,
            Self::Off => -1.0,
        }
    }
}

/// A recorded spike train, kept sorted by `(t, source)`.
///
/// The sort order is an invariant rather than a convenience: [`Train::rate`] and
/// [`Train::intervals`] both assume it, and a consumer that merges two trains and forgets to
/// re-sort gets an interval sequence containing negative numbers, which no plot would show as
/// wrong.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Train {
    spikes: Vec<Spike>,
}

impl Train {
    /// An empty train.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Build from spikes in any order; they are sorted here so the invariant holds on exit.
    #[must_use]
    pub fn from_spikes(mut spikes: Vec<Spike>) -> Self {
        spikes.sort_unstable();
        Self { spikes }
    }

    /// Append a spike.
    ///
    /// # Panics
    ///
    /// If `s` is earlier than the last spike already recorded. A train is built forward in time by
    /// a simulation that advances monotonically, so an out-of-order push is a bug in the caller,
    /// and silently sorting it away would hide the bug while producing a plausible train.
    pub fn push(&mut self, s: Spike) {
        if let Some(last) = self.spikes.last() {
            assert!(
                s.t >= last.t,
                "spike at tick {} pushed after tick {}; a train is built forward in time",
                s.t,
                last.t
            );
        }
        self.spikes.push(s);
    }

    /// The spikes, sorted by `(t, source)`.
    #[must_use]
    pub fn spikes(&self) -> &[Spike] {
        &self.spikes
    }

    /// How many spikes are recorded.
    #[must_use]
    pub fn len(&self) -> usize {
        self.spikes.len()
    }

    /// Whether the train is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.spikes.is_empty()
    }

    /// Spikes from one source, in order.
    #[must_use]
    pub fn of(&self, source: u32) -> Vec<Spike> {
        self.spikes.iter().copied().filter(|s| s.source == source).collect()
    }

    /// Mean firing rate of one source in hertz, over `ticks` ticks of `dt` seconds.
    ///
    /// `None` when no time elapsed. Counting spikes over zero seconds is a division by zero, and
    /// the answer a caller wants for it is "you did not run anything", not an infinity.
    #[must_use]
    pub fn rate(&self, source: u32, ticks: u64, dt: f64) -> Option<f64> {
        if ticks == 0 || dt <= 0.0 {
            return None;
        }
        let n = self.spikes.iter().filter(|s| s.source == source).count();
        Some(n as f64 / (ticks as f64 * dt))
    }

    /// Inter-spike intervals of one source, in seconds.
    ///
    /// Empty when the source fired fewer than twice: one spike has no interval, and returning a
    /// zero or the time since the start would both be numbers where there is no measurement.
    #[must_use]
    pub fn intervals(&self, source: u32, dt: f64) -> Vec<f64> {
        let t: Vec<u64> = self.spikes.iter().filter(|s| s.source == source).map(|s| s.t).collect();
        t.windows(2).map(|w| (w[1] - w[0]) as f64 * dt).collect()
    }

    /// Coefficient of variation of the intervals of one source.
    ///
    /// `None` when there are fewer than two intervals to compare, or when the mean is zero. A
    /// regular pacemaker gives ~0; a Poisson process gives ~1; anything above 1 is bursty. It is
    /// the standard one-number summary of whether a train carries timing structure or only a rate.
    #[must_use]
    pub fn cv(&self, source: u32, dt: f64) -> Option<f64> {
        let iv = self.intervals(source, dt);
        if iv.len() < 2 {
            return None;
        }
        let n = iv.len() as f64;
        let mean = iv.iter().sum::<f64>() / n;
        if mean <= 0.0 {
            return None;
        }
        // Sample variance, n-1: the intervals ARE a sample, and using n here would report a
        // spuriously low CV on the short trains this is most often called on.
        let var = iv.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (n - 1.0);
        Some(var.sqrt() / mean)
    }
}

/// Consecutive intervals `I_i`, at least two, every one finite and positive — or `None`.
fn adjacent(intervals: &[f64]) -> Option<f64> {
    (intervals.len() >= 2 && intervals.iter().all(|&x| x.is_finite() && x > 0.0)).then(|| (intervals.len() - 1) as f64)
}

/// Shinomoto, Shima and Tanji's local variation of a sequence of intervals (*Differences in spiking
/// patterns among cortical neurons*, Neural Computation 15:2823–2842, 2003, Eq. 2.2):
///
/// ```text
/// Lv = 3/(n − 1) · Σ_{i<n} (I_i − I_{i+1})² / (I_i + I_{i+1})²
/// ```
///
/// Each term compares only NEIGHBOURING intervals, so a slow change of rate that the global
/// coefficient of variation ([`Train::cv`]) reads as irregularity leaves it alone. 0 for a regular
/// train; for intervals drawn from a gamma distribution of order `z` its expectation is
/// `3/(2z + 1)` (their Eq. B.7) — 1 for a Poisson process, 0.6 for `z = 2`. `None` for fewer than two
/// intervals or any that is not finite and positive. Written in the order `Elephant`'s `lv` sums,
/// so it returns the same double.
#[must_use]
pub fn local_variation(intervals: &[f64]) -> Option<f64> {
    let m = adjacent(intervals)?;
    let sum: f64 = intervals.windows(2).map(|w| ((w[1] - w[0]) / (w[0] + w[1])).powi(2)).sum();
    Some(3.0 * (sum / m))
}

/// Shinomoto et al.'s revised local variation (*Relating neuronal firing patterns to functional
/// differentiation of cerebral cortex*, `PLoS` Computational Biology 5:e1000433, 2009, Eq. 3), with a
/// refractoriness constant `r` in the intervals' unit:
///
/// ```text
/// LvR = 3/(n − 1) · Σ_{i<n} (1 − 4 I_i I_{i+1}/(I_i + I_{i+1})²) · (1 + 4r/(I_i + I_{i+1}))
/// ```
///
/// the first-order expansion in `r` of [`local_variation`] with `r` taken off each interval. Equal
/// to `Lv` at `r = 0`; they found `r` = 5 ms discriminated cortical areas best. `None` as for
/// [`local_variation`], and for an `r` that is negative or not finite.
#[must_use]
pub fn revised_local_variation(intervals: &[f64], r: f64) -> Option<f64> {
    let m = adjacent(intervals)?;
    if !(r.is_finite() && r >= 0.0) {
        return None;
    }
    let sum: f64 = intervals
        .windows(2)
        .map(|w| {
            let t = w[0] + w[1];
            (1.0 - 4.0 * w[0] * w[1] / (t * t)) * (1.0 + 4.0 * r / t)
        })
        .sum();
    Some(3.0 / m * sum)
}

/// Holt, Softky, Koch and Douglas's `CV2` (Journal of Neurophysiology 75:1806–1814, 1996), as
/// `Elephant` computes it: `2/(n − 1) · Σ_{i<n} |I_{i+1} − I_i|/(I_{i+1} + I_i)`. The same
/// neighbour-by-neighbour comparison as [`local_variation`], with an absolute value where `Lv`
/// squares. The 1996 paper was not read here; the definition is `Elephant`'s code. `None` as for
/// [`local_variation`].
#[must_use]
pub fn cv2(intervals: &[f64]) -> Option<f64> {
    let m = adjacent(intervals)?;
    let sum: f64 = intervals.windows(2).map(|w| ((w[1] - w[0]) / (w[0] + w[1])).abs()).sum();
    Some(2.0 * (sum / m))
}

#[cfg(test)]
mod tests {
    use super::{Event, Polarity, Spike, Train, cv2, local_variation, revised_local_variation};
    use crate::rng::Rng;

    #[test]
    fn spikes_sort_by_time_then_source() {
        let mut v = [
            Spike { t: 5, source: 1 },
            Spike { t: 1, source: 9 },
            Spike { t: 5, source: 0 },
        ];
        v.sort_unstable();
        assert_eq!(v[0], Spike { t: 1, source: 9 });
        assert_eq!(v[1], Spike { t: 5, source: 0 });
        assert_eq!(v[2], Spike { t: 5, source: 1 });
    }

    #[test]
    #[should_panic(expected = "built forward in time")]
    fn pushing_backwards_in_time_panics_rather_than_silently_sorting() {
        let mut tr = Train::new();
        tr.push(Spike { t: 10, source: 0 });
        tr.push(Spike { t: 9, source: 0 });
    }

    #[test]
    fn rate_counts_only_the_source_asked_for() {
        let tr = Train::from_spikes(vec![
            Spike { t: 0, source: 0 },
            Spike { t: 1, source: 1 },
            Spike { t: 2, source: 0 },
            Spike { t: 3, source: 0 },
        ]);
        // 3 spikes from source 0 over 4 ticks of 1 ms = 0.004 s -> 750 Hz.
        let r = tr.rate(0, 4, 1e-3).unwrap();
        assert!((r - 750.0).abs() < 1e-9, "{r}");
        let r1 = tr.rate(1, 4, 1e-3).unwrap();
        assert!((r1 - 250.0).abs() < 1e-9, "{r1}");
    }

    #[test]
    fn a_run_of_no_length_has_no_rate() {
        let tr = Train::from_spikes(vec![Spike { t: 0, source: 0 }]);
        assert!(tr.rate(0, 0, 1e-3).is_none());
        assert!(tr.rate(0, 10, 0.0).is_none());
    }

    #[test]
    fn one_spike_has_no_interval() {
        let tr = Train::from_spikes(vec![Spike { t: 4, source: 0 }]);
        assert!(tr.intervals(0, 1e-3).is_empty());
        assert!(tr.cv(0, 1e-3).is_none());
    }

    /// A pacemaker has zero variability. If this ever returns something else, the interval
    /// arithmetic is wrong in a way that would make every CV in the crate meaningless.
    #[test]
    fn a_perfectly_regular_train_has_zero_cv() {
        let sp: Vec<Spike> = (0..20).map(|k| Spike { t: k * 10, source: 0 }).collect();
        let tr = Train::from_spikes(sp);
        let cv = tr.cv(0, 1e-3).unwrap();
        assert!(cv.abs() < 1e-12, "cv {cv}");
    }

    /// A regular train has zero variance, so EVERY way of normalising it gives zero: the existing
    /// pacemaker test cannot tell a coefficient of variation from a standard deviation, from a
    /// variance, or from a population estimator. This one uses intervals of one, two and three
    /// ticks, whose CV is exactly one half, and works the arithmetic out by hand.
    #[test]
    fn the_coefficient_of_variation_is_the_sample_one_normalised_by_the_mean() {
        let dt = 1e-3;
        let tr = Train::from_spikes(vec![
            Spike { t: 0, source: 0 },
            Spike { t: 1, source: 0 },
            Spike { t: 3, source: 0 },
            Spike { t: 6, source: 0 },
            // A second source, interleaved, so that anything reading the whole train sees it.
            Spike { t: 0, source: 1 },
            Spike { t: 2, source: 1 },
            Spike { t: 4, source: 1 },
        ]);
        // Intervals in SECONDS, from this source only, in order.
        assert_eq!(tr.intervals(0, dt), vec![1e-3, 2e-3, 3e-3]);
        assert_eq!(tr.intervals(1, dt), vec![2e-3, 2e-3]);
        // mean 2 ms; deviations -1, 0, +1 ms; sample variance (n-1 = 2) is 1e-6 s^2, so the
        // standard deviation is 1 ms and the CV is exactly one half.
        let cv = tr.cv(0, dt).expect("three intervals is enough");
        assert!((cv - 0.5).abs() < 1e-15, "cv = {cv}");
        // The population estimator would give sqrt(2/3)/2 = 0.408, the unnormalised deviation
        // 1e-3, and the variance itself 5e-4. None of those is 0.5.
        assert!((cv - (2.0f64 / 3.0).sqrt() / 2.0).abs() > 0.09, "this is the n, not n-1, answer");
        assert!(cv > 1e-2 && cv < 1e2, "cv = {cv} is on the scale of a raw variance or deviation");
        // The second source is regular, so ITS cv is zero — which is the case the old test had.
        assert_eq!(tr.cv(1, dt), Some(0.0));
    }

    #[test]
    fn a_source_is_read_apart_from_the_others_and_the_train_reports_its_own_size() {
        let tr = Train::from_spikes(vec![
            Spike { t: 4, source: 7 },
            Spike { t: 0, source: 3 },
            Spike { t: 2, source: 7 },
        ]);
        // `from_spikes` sorts, which `rate` and `intervals` both rely on.
        assert_eq!(tr.spikes().iter().map(|s| (s.t, s.source)).collect::<Vec<_>>(), vec![(0, 3), (2, 7), (4, 7)]);
        assert_eq!(tr.of(7), vec![Spike { t: 2, source: 7 }, Spike { t: 4, source: 7 }]);
        assert_eq!(tr.of(3), vec![Spike { t: 0, source: 3 }]);
        assert_eq!(tr.of(9), vec![]);
        assert_eq!(tr.len(), 3);
        assert!(!tr.is_empty());
        // `len` is checked against the slice it is meant to count, not against zero, so that a
        // `len` which always returned zero would not pass by agreeing with `is_empty`.
        let fresh = Train::new();
        assert!(fresh.is_empty() && fresh.len() == fresh.spikes().len());
        // A train built by pushing has a capacity that grows in jumps, so `len` reporting the
        // capacity would be right for some sizes and wrong for this one.
        let mut built = Train::new();
        for t in 0..3 {
            built.push(Spike { t, source: 0 });
        }
        assert_eq!(built.len(), 3);
        assert_eq!(built.spikes().len(), built.len());
    }

    #[test]
    fn two_neurons_may_fire_on_the_same_tick_and_a_source_that_never_waits_has_no_variation() {
        // Simultaneity is not an error: a push at the tick just recorded is ordinary.
        let mut tr = Train::new();
        tr.push(Spike { t: 5, source: 0 });
        tr.push(Spike { t: 5, source: 1 });
        tr.push(Spike { t: 5, source: 0 });
        tr.push(Spike { t: 5, source: 0 });
        assert_eq!(tr.len(), 4);
        assert_eq!(tr.of(0).len(), 3, "three spikes from one source is what gives TWO intervals");
        // Three spikes at one tick give two intervals of zero, so the mean is zero and there is no
        // coefficient of variation to report — a division this must refuse rather than return an
        // infinity or a NaN from.
        assert_eq!(tr.intervals(0, 1e-3), vec![0.0, 0.0]);
        assert_eq!(tr.cv(0, 1e-3), None);
    }

    #[test]
    fn polarity_signs_are_the_obvious_way_round() {
        assert!((Polarity::On.sign() - 1.0).abs() < 1e-15);
        assert!((Polarity::Off.sign() + 1.0).abs() < 1e-15);
    }

    #[test]
    fn events_sort_by_time_first() {
        let a = Event { t: 1, address: 99, polarity: Polarity::On };
        let b = Event { t: 2, address: 0, polarity: Polarity::Off };
        assert!(a < b);
    }

    /// `Elephant`'s own examples (`elephant/statistics.py`, commit `32f1b56`): `lv`, `lvr` with
    /// `R=0.005` and `cv2` of the intervals `[0.3, 4.5, 6.7, 9.3]` print 0.8306154336734695,
    /// 0.833907445980624 and 0.8226190476190478, and these are the same doubles.
    #[test]
    fn the_local_measures_are_elephants_on_its_own_examples() {
        let x = [0.3, 4.5, 6.7, 9.3];
        assert_eq!(local_variation(&x), Some(0.830_615_433_673_469_5));
        assert_eq!(revised_local_variation(&x, 0.005), Some(0.833_907_445_980_624));
        assert_eq!(cv2(&x), Some(0.822_619_047_619_047_8));
    }

    /// Shinomoto 2003, Eq. B.7: for gamma intervals of order `z` the expected `Lv` is `3/(2z + 1)`.
    /// Two hundred thousand seeded intervals for each of `z` = 1, 2, 4, their standard error taken
    /// from a hundred batches: measured 0.99612, 0.60137 and 0.33333 against 1, 0.6 and 0.33333,
    /// within 1.96, 0.69 and 0.01 standard errors. Neighbouring terms share an interval, which is
    /// why the error comes from batches and not from the terms.
    #[test]
    fn local_variation_meets_its_gamma_closed_form() {
        let mut rng = Rng::new(2003);
        for z in [1usize, 2, 4] {
            let iv: Vec<f64> = (0..200_000).map(|_| (0..z).map(|_| -(1.0 - rng.next_f64()).ln()).sum::<f64>()).collect();
            let lv = local_variation(&iv).unwrap();
            let batches: Vec<f64> = iv.chunks(2_000).map(|c| local_variation(c).unwrap()).collect();
            let k = batches.len() as f64;
            let mean = batches.iter().sum::<f64>() / k;
            let se = (batches.iter().map(|b| (b - mean).powi(2)).sum::<f64>() / (k - 1.0) / k).sqrt();
            let want = 3.0 / (2.0 * z as f64 + 1.0);
            assert!((lv - want).abs() < 3.0 * se && se < 0.003, "z = {z}: Lv {lv} against {want}, standard error {se}");
        }
    }

    /// Closed forms. A regular train has `Lv = LvR = CV2 = 0` exactly. `LvR` at `r = 0` is `Lv`
    /// (to rounding: the terms are written differently). `Lv` and `CV2` are ratios of neighbouring
    /// intervals, so doubling every interval changes neither by a bit — a rate change they ignore —
    /// while `LvR` with a fixed `r` rises, because `r` becomes a smaller share of each interval. And
    /// intervals alternating short and long — Shinomoto 2009's Fig. 1C — are maximally irregular
    /// locally: with ratio 3, each term of `Lv` is `3·(2/4)² = 3/4` and of `CV2` is `2·(2/4) = 1`.
    #[test]
    fn the_local_measures_meet_their_closed_forms() {
        let regular = [0.25; 5];
        assert_eq!(local_variation(&regular), Some(0.0));
        assert_eq!(revised_local_variation(&regular, 0.5), Some(0.0));
        assert_eq!(cv2(&regular), Some(0.0));
        let x = [0.3, 4.5, 6.7, 9.3, 0.8, 2.25];
        let lv = local_variation(&x).unwrap();
        assert!((revised_local_variation(&x, 0.0).unwrap() - lv).abs() < 1e-15 * lv);
        let doubled: Vec<f64> = x.iter().map(|v| v * 2.0).collect();
        assert_eq!(local_variation(&doubled), Some(lv));
        assert_eq!(cv2(&doubled), cv2(&x));
        assert!(revised_local_variation(&x, 0.25).unwrap() > revised_local_variation(&doubled, 0.25).unwrap());
        let alternating = [1.0, 3.0, 1.0, 3.0, 1.0];
        assert_eq!(local_variation(&alternating), Some(0.75));
        assert_eq!(cv2(&alternating), Some(1.0));
        // LvR's correction by hand for one pair: (1 − 4·1·3/16)(1 + 4·0.5/4) · 3 = 0.25 · 1.5 · 3.
        assert_eq!(revised_local_variation(&[1.0, 3.0], 0.5), Some(1.125));
    }

    /// Fewer than two intervals, an interval that is not finite and positive, or a refractoriness
    /// that is negative or not finite: no number.
    #[test]
    fn the_local_measures_refuse_what_they_cannot_measure() {
        for bad in [&[][..], &[1.0], &[1.0, 0.0], &[1.0, -2.0], &[1.0, f64::NAN], &[f64::INFINITY, 1.0]] {
            assert_eq!(local_variation(bad), None, "{bad:?}");
            assert_eq!(revised_local_variation(bad, 0.0), None, "{bad:?}");
            assert_eq!(cv2(bad), None, "{bad:?}");
        }
        assert_eq!(revised_local_variation(&[1.0, 2.0], -0.001), None);
        assert_eq!(revised_local_variation(&[1.0, 2.0], f64::NAN), None);
        assert_eq!(revised_local_variation(&[1.0, 2.0], f64::INFINITY), None);
        assert!(revised_local_variation(&[1.0, 2.0], 0.0).is_some());
    }
}
