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
//! The `(time, address, polarity)` triple is the format neuromorphic sensors and chips actually
//! speak — Mahowald's address-event representation, 1992, and every event camera since. [`Event`]
//! is that triple, and [`Spike`] is the same thing without a polarity for the many places inside a
//! network where there is only one kind.

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

#[cfg(test)]
mod tests {
    use super::{Event, Polarity, Spike, Train};

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
}
