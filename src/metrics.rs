//! Benchmark metrics: what `NeuroBench` measures, computed here from a model and a run.
//!
//! # The lesson: why a field invents a complexity metric
//!
//! Suppose you want to compare two spiking networks. The obvious comparison is joules, and the
//! obvious problem is that almost nobody has the chip. Neuromorphic parts are research silicon:
//! `Loihi` is loaned, `TrueNorth` was never sold, `SpiNNaker` is a machine in Manchester. If the
//! only admissible comparison required hardware, the algorithm half of the field could not report
//! anything at all.
//!
//! So the field defines **complexity metrics**: quantities computable from the model's parameters
//! and from one run's activations, with no hardware, no power meter and no vendor. They are
//! deliberately *proxies*. What they buy is that two groups on two laptops can report the same
//! number for the same model and have it mean the same thing. What they cost is that a proxy is
//! not the quantity anybody cares about — a synaptic operation count is not a joule, and the gap
//! between them is where this crate's [`crate::ledger`] lives.
//!
//! `NeuroBench` is the field's attempt to standardise those proxies, and it is a big one: Yik et
//! al., *`NeuroBench`: A Framework for Benchmarking Neuromorphic Computing Algorithms and Systems*,
//! Nature Communications 16:1545 (2025), with roughly a hundred co-authors across academia and
//! industry. Its **algorithm track** defines the metrics in this module. They are all computable
//! from a model description and a recorded run, which is why they can live in a zero-dependency
//! library rather than in a lab.
//!
//! # ⚠ The distinction that gets misreported
//!
//! **`NeuroBench`'s algorithm track has no energy metric.** Not a weak one, not an estimated one —
//! none. Its complexity metrics are footprint, connection sparsity, activation sparsity and
//! synaptic operations, and every one of those is a count, not a joule. An algorithm-track result
//! says nothing about power.
//!
//! **`NeuroBench`'s system track is a different track**, with different entries, different rules
//! and different submissions, and it *does* mandate measured execution time and measured energy on
//! the hardware being claimed. That is where the `SynSense` Xylo Audio 2 figure quoted in
//! [`crate::ledger`] comes from.
//!
//! Conflating the two produces a sentence like "`NeuroBench` shows this model uses 4.2 µJ", which
//! is a system-track claim attached to an algorithm-track submission. Nothing in this module can
//! produce a joule. If you want one, price the run with [`crate::ledger::Ledger::bill`] and read
//! the refusal it gives you.
//!
//! # The metrics
//!
//! **Activation sparsity** — the fraction of activations that are exactly zero, aggregated over
//! every layer and every timestep. For a spiking network, where an activation is a spike or the
//! absence of one, it is exactly the fraction of neuron-timesteps on which nothing fired. It is the
//! quantity the whole event-driven argument rests on: hardware that skips zeros only wins if there
//! are zeros to skip.
//!
//! **Synaptic operations, split into effective `MACs` and effective `ACs`.** A synaptic operation
//! is a weight meeting an activation. `NeuroBench` counts one only when **both** are non-zero —
//! zero times anything is zero, and hardware that computes it anyway is doing work the metric
//! declines to credit. The split is the part that matters:
//!
//! - an **`AC`** (accumulate) is what a *binary* activation costs: a spike is a 1, so the multiply
//!   is a no-op and the hardware adds the weight;
//! - a **`MAC`** (multiply-accumulate) is what a *real-valued* activation costs, including a
//!   graded spike of the sort `Loihi 2` sends.
//!
//! A metric that reported a single "synaptic operation" count would make a binary network and a
//! graded one look identical, and the arithmetic cost ratio between an `AC` and a `MAC` is roughly
//! where half the field's energy claims live. [`SynOps`] keeps them apart and refuses to add them
//! into one number without the caller saying so.
//!
//! **Footprint** — bytes of parameters plus bytes of state. The metric a model that does not fit
//! on the chip fails, and the reason quantisation is not a detail.
//!
//! **Connection sparsity** — the fraction of the model's connections whose weight is exactly zero.
//! Distinct from activation sparsity in the way a pruned network is distinct from a quiet one, and
//! multiplicative with it: both zeros remove the same operation.
//!
//! # What this module does NOT implement
//!
//! The metrics, not the benchmark. `NeuroBench` is also a set of datasets (keyword spotting,
//! event-camera object detection, a primate reaching task, chaotic-system prediction), reference
//! models, and a `PyTorch` harness that installs hooks to capture activations. None of that is
//! here, and a number this module computes is not a `NeuroBench` submission. It is the same
//! arithmetic on your own data.
//!
//! One definition is a genuine approximation and is labelled where it appears:
//! [`mean_average_precision`] computes the all-point (non-interpolated) average precision used by
//! ranking evaluations, per class, and not the `IoU`-thresholded `COCO` `mAP@[.5:.95]` that an
//! object-detection task needs. See its doc.
//!
//! # The bridge to the rest of this crate
//!
//! The same counters answer both questions. [`summarise`] takes a [`Net`], a finished
//! [`Ledger`] and the [`Train`] a run produced, and returns the algorithm-track numbers **beside**
//! the crossover verdict from [`crate::crossover`] — because a model with 98% activation sparsity
//! and a spikes-per-synapse ratio above every published threshold is a model whose benchmark
//! numbers look excellent and whose energy argument has already lost. Reporting one without the
//! other is the habit this crate exists to break.
//!
//! ```
//! use ferromorphic::metrics::{ActivationCensus, SynOps, activation_sparsity_of_run};
//!
//! // Four neurons over five ticks is twenty neuron-timesteps; three spikes fired.
//! let mut census = ActivationCensus::default();
//! census.observe(&[0.0, 1.0, 0.0, 0.0])?; // tick 0
//! census.observe(&[0.0, 0.0, 0.0, 0.0])?; // tick 1
//! census.observe(&[1.0, 0.0, 1.0, 0.0])?; // tick 2
//! census.observe(&[0.0, 0.0, 0.0, 0.0])?; // tick 3
//! census.observe(&[0.0, 0.0, 0.0, 0.0])?; // tick 4
//! assert_eq!(census.total, 20);
//! assert_eq!(census.zeros, 17);
//! assert!((census.sparsity().unwrap() - 0.85).abs() < 1e-15);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use core::fmt;

use crate::crossover::{THRESHOLDS, Verdict};
use crate::ledger::Ledger;
use crate::net::Net;
use crate::spike::Train;

/// Why a metric has no answer.
///
/// Every variant names the offending quantity rather than saying "invalid input", because a metric
/// is usually computed in a loop over a test set and the useful information is *which sample*.
#[derive(Debug, Clone, PartialEq)]
pub enum MetricError {
    /// A value was `NaN` or infinite. Rejected at the boundary: a single non-finite activation
    /// propagates into a mean and turns an entire benchmark result into `NaN` with no indication
    /// of where it entered.
    NonFinite {
        /// Which array the bad value was in.
        what: &'static str,
        /// Its index in that array.
        index: usize,
    },
    /// Two arrays that must be the same length were not.
    LengthMismatch {
        /// Length of the first array.
        a: usize,
        /// Length of the second array.
        b: usize,
    },
    /// A quantity was asked of nothing at all. The mean of an empty set is not zero.
    Empty {
        /// Which array was empty.
        what: &'static str,
    },
    /// A weight matrix's length was not `fan_in * neurons`, so its shape is not what was declared.
    BadShape {
        /// Length actually supplied.
        got: usize,
        /// Length implied by the declared shape.
        want: usize,
    },
    /// An activation declared binary was neither exactly `0.0` nor exactly `1.0`.
    ///
    /// Refused rather than rounded. Counting a `0.7` activation as an `AC` charges a
    /// multiply-accumulate at accumulate prices, which is an understatement in the direction that
    /// flatters the result.
    NotBinary {
        /// Index of the offending activation.
        index: usize,
        /// The value found there.
        value: f64,
    },
    /// A class label was at or past the declared class count.
    ClassOutOfRange {
        /// Index of the sample carrying the bad label.
        index: usize,
        /// The label itself.
        label: usize,
        /// The class count it exceeded.
        classes: usize,
    },
    /// `k` was zero or larger than the class count, so "top `k`" names no set of classes.
    BadTopK {
        /// The `k` requested.
        k: usize,
        /// The class count available.
        classes: usize,
    },
    /// A bit-width was zero or wider than 64 bits.
    ///
    /// Zero is refused because a parameter that occupies no bits is not a parameter; 64 is the
    /// ceiling only because nothing in this crate stores a wider scalar and a wider one should be
    /// stated as several.
    BadBitWidth {
        /// The width requested, in bits.
        bits: u32,
    },
    /// The targets were all the same value, so the total sum of squares is zero and R-squared is
    /// `0/0`.
    ///
    /// Returned rather than `0.0` or `1.0`: a constant target has no variance for a model to
    /// explain, and both of the tempting answers are claims about model quality that the data
    /// cannot support.
    ZeroVariance,
    /// An average-precision query had no positive examples in a class, so precision and recall are
    /// both undefined for it.
    NoPositives {
        /// The class with no positives, or `0` for a single-class query.
        class: usize,
    },
    /// A spike named a neuron index at or past the network's neuron count.
    SourceOutOfRange {
        /// The offending source index.
        source: u32,
        /// The neuron count it exceeded.
        n: usize,
    },
    /// More spikes were recorded than there are neuron-timesteps to hold them.
    ///
    /// In this crate a neuron fires at most once per tick, so a train with more spikes than
    /// `neurons * ticks` describes a run that did not happen — usually two trains concatenated, or
    /// a tick count taken from the wrong variable.
    ///
    /// This is the *gross* form of that error. The subtle forms — a train under capacity that
    /// still names one neuron-timestep twice, or one outside the run — are
    /// [`MetricError::ImpossibleSpike`].
    TooManySpikes {
        /// Spikes in the train.
        spikes: u64,
        /// Neuron-timesteps available, `neurons * ticks`.
        capacity: u64,
    },
    /// One spike could not have come from the run it is being measured against.
    ///
    /// Either it names a tick at or past the end of the run, or it names a neuron-timestep a
    /// second time — and in this crate a neuron fires at most once per tick. Both shapes usually
    /// mean two trains were concatenated, or a tick count was taken from the wrong variable.
    ///
    /// Refused rather than counted, because [`activation_sparsity_of_run`] credits one active
    /// neuron-timestep per spike: a train that repeats a spike reports a run as denser than it
    /// was, by an amount no plot would show as wrong. [`crate::spike::Train::push`] and
    /// [`crate::spike::Train::from_spikes`] are both public and neither enforces one spike per
    /// neuron per tick, so it is enforced at the metric.
    ImpossibleSpike {
        /// The tick the offending spike names.
        t: u64,
        /// The neuron the offending spike names.
        source: u32,
        /// What is wrong with it, as a phrase completing "the spike … ". One of
        /// `"is on a tick at or past the end of the run"` or
        /// `"repeats a neuron-timestep, and a neuron fires at most once per tick"`.
        why: &'static str,
    },
    /// More connections were counted than the declared architecture holds.
    ///
    /// Raised when a zero-weight count exceeds the total connection count, or when a [`Net`]
    /// connects more distinct ordered pairs than the declared dense count allows. Kept apart from
    /// [`MetricError::TooManySpikes`] because these quantities are weights, not spikes, and an
    /// error message that says "spikes in neuron-timesteps" sends a reader to the wrong half of
    /// their pipeline.
    TooManyConnections {
        /// Connections counted.
        counted: u64,
        /// Connections the declared architecture holds.
        declared: u64,
    },
    /// A count exceeded `u64`. Reported rather than wrapped, because a wrapped synaptic-operation
    /// count is a small number where an enormous one belongs.
    Overflow {
        /// Which product overflowed.
        what: &'static str,
    },
}

impl fmt::Display for MetricError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFinite { what, index } => {
                write!(f, "{what}[{index}] is not a finite number")
            }
            Self::LengthMismatch { a, b } => write!(f, "arrays of length {a} and {b} do not match"),
            Self::Empty { what } => write!(f, "{what} is empty; there is nothing to average"),
            Self::BadShape { got, want } => {
                write!(f, "weight matrix has {got} entries where the declared shape needs {want}")
            }
            Self::NotBinary { index, value } => {
                write!(f, "activation[{index}] is {value}, which is neither 0 nor 1")
            }
            Self::ClassOutOfRange { index, label, classes } => {
                write!(f, "sample {index} has label {label} for a {classes}-class problem")
            }
            Self::BadTopK { k, classes } => {
                write!(f, "top-{k} is not defined over {classes} classes")
            }
            Self::BadBitWidth { bits } => write!(f, "{bits} bits per value is out of range 1..=64"),
            Self::ZeroVariance => {
                f.write_str("the targets are constant; R-squared has no denominator")
            }
            Self::NoPositives { class } => {
                write!(f, "class {class} has no positive examples; average precision is undefined")
            }
            Self::SourceOutOfRange { source, n } => {
                write!(f, "a spike came from neuron {source} in a network of {n}")
            }
            Self::TooManySpikes { spikes, capacity } => {
                write!(f, "{spikes} spikes in {capacity} neuron-timesteps")
            }
            Self::ImpossibleSpike { t, source, why } => {
                write!(f, "the spike at tick {t} from neuron {source} {why}")
            }
            Self::TooManyConnections { counted, declared } => {
                write!(f, "{counted} connections against a declared {declared}")
            }
            Self::Overflow { what } => write!(f, "{what} overflowed a u64 count"),
        }
    }
}

/// So that `?` works in a caller whose error type is `Box<dyn Error>`, as every example and
/// doctest in this crate uses. See the same note on [`crate::net::NetError`].
impl std::error::Error for MetricError {}

/// `a * b`, or an overflow error naming the product.
fn mul(a: u64, b: u64, what: &'static str) -> Result<u64, MetricError> {
    a.checked_mul(b).ok_or(MetricError::Overflow { what })
}

/// Reject any non-finite entry, naming the array and the index.
fn all_finite(xs: &[f64], what: &'static str) -> Result<(), MetricError> {
    for (i, x) in xs.iter().enumerate() {
        if !x.is_finite() {
            return Err(MetricError::NonFinite { what, index: i });
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Activation sparsity
// ---------------------------------------------------------------------------------------------

/// Running counts of zero and total activations, from which activation sparsity is a ratio.
///
/// # How `NeuroBench` aggregates, and why it is not the mean of the per-step ratios
///
/// The metric is defined over the whole run: **total zeros divided by total activations**, pooled
/// across layers and timesteps. That is what this struct accumulates.
///
/// It is not the same as averaging each timestep's sparsity, and the two differ whenever the
/// timesteps have different numbers of activations — a network whose layers are 1000 and 10 wide
/// weights the wide layer 100 times more heavily under pooling and equally under averaging. For a
/// fixed-width spiking layer they coincide exactly, which is why the difference is easy to miss and
/// worth stating. `the_pooled_ratio_is_not_the_mean_of_the_per_step_ratios` in this module's tests
/// exhibits a case where they differ.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ActivationCensus {
    /// Activations observed that were exactly zero. `-0.0` counts as zero, because `-0.0 == 0.0`
    /// in IEEE 754 and hardware that skips zeros skips it.
    pub zeros: u64,
    /// Activations observed in total, across every layer and every timestep.
    pub total: u64,
}

impl ActivationCensus {
    /// Record one layer's activations on one timestep.
    ///
    /// # Errors
    ///
    /// [`MetricError::NonFinite`] if any activation is `NaN` or infinite, or
    /// [`MetricError::Overflow`] if the running total exceeds `u64`.
    pub fn observe(&mut self, activations: &[f64]) -> Result<(), MetricError> {
        all_finite(activations, "activation")?;
        let zeros = activations.iter().filter(|x| **x == 0.0).count() as u64;
        self.zeros = self
            .zeros
            .checked_add(zeros)
            .ok_or(MetricError::Overflow { what: "zero activations" })?;
        self.total = self
            .total
            .checked_add(activations.len() as u64)
            .ok_or(MetricError::Overflow { what: "total activations" })?;
        Ok(())
    }

    /// Record a timestep by counts alone, for a caller that never materialised the vector.
    ///
    /// `fired` is the number of non-zero activations among `n` of them.
    ///
    /// # Errors
    ///
    /// [`MetricError::TooManySpikes`] if `fired > n`, which describes a timestep that could not
    /// have happened, and [`MetricError::Overflow`] on a running total past `u64`.
    pub fn observe_counts(&mut self, fired: u64, n: u64) -> Result<(), MetricError> {
        if fired > n {
            return Err(MetricError::TooManySpikes { spikes: fired, capacity: n });
        }
        self.zeros = self
            .zeros
            .checked_add(n - fired)
            .ok_or(MetricError::Overflow { what: "zero activations" })?;
        self.total =
            self.total.checked_add(n).ok_or(MetricError::Overflow { what: "total activations" })?;
        Ok(())
    }

    /// Fraction of activations that were zero, in `0.0..=1.0`.
    ///
    /// `None` when nothing was observed. Not `1.0`: a network that was never run is not a sparse
    /// network, and the distinction is exactly the one an empty test set silently erases.
    #[must_use]
    pub fn sparsity(&self) -> Option<f64> {
        if self.total == 0 {
            return None;
        }
        Some(self.zeros as f64 / self.total as f64)
    }

    /// Fraction of activations that were non-zero — the firing density, `1 - sparsity`.
    ///
    /// `None` on an empty census, for the same reason as [`ActivationCensus::sparsity`].
    ///
    /// # Why this is a division and not `1.0 - sparsity()`
    ///
    /// Because the two are different numbers in the middle of the range, and this one is the
    /// correctly rounded quotient. At the *ends* they agree exactly — `sparsity()` is exactly
    /// `0.0` or exactly `1.0` there and subtracting from one is exact — so the ends cannot tell
    /// them apart. One zero in three can: `(3 - 1) as f64 / 3.0` is the double nearest 2/3,
    /// `0.666_666_666_666_666_63`, while `1.0 - (1.0 / 3.0)` is one ulp above it,
    /// `0.666_666_666_666_666_74`. `sparsity() + density() == 1.0` holds under **both** forms at
    /// every ratio, so it is not a check on this choice;
    /// `the_density_is_the_quotient_and_not_one_minus_the_sparsity` in this module's tests is.
    #[must_use]
    pub fn density(&self) -> Option<f64> {
        if self.total == 0 {
            return None;
        }
        Some((self.total - self.zeros) as f64 / self.total as f64)
    }

    /// Fold another census into this one — two layers, two runs, two workers.
    ///
    /// # Errors
    ///
    /// [`MetricError::Overflow`] if either running total exceeds `u64`.
    pub fn merge(&mut self, other: Self) -> Result<(), MetricError> {
        self.zeros = self
            .zeros
            .checked_add(other.zeros)
            .ok_or(MetricError::Overflow { what: "zero activations" })?;
        self.total = self
            .total
            .checked_add(other.total)
            .ok_or(MetricError::Overflow { what: "total activations" })?;
        Ok(())
    }
}

/// Refuse a train that could not have come from one run of `neurons` neurons over `ticks` ticks.
///
/// Three ways it could not: a spike from a neuron the run does not have, a spike on a tick at or
/// past the end of the run, and the same neuron-timestep named twice. The counts of
/// [`activation_sparsity_of_run`] rest on each spike being one distinct active neuron-timestep
/// inside the window, and none of the three is caught by comparing the spike count against the
/// capacity: three spikes on `(t = 0, source = 0)` over two neurons and five ticks are well under
/// a capacity of ten and describe a run that did not happen.
fn check_one_run(train: &Train, neurons: u64, ticks: u64) -> Result<(), MetricError> {
    let spikes = train.spikes();
    for s in spikes {
        if u64::from(s.source) >= neurons {
            return Err(MetricError::SourceOutOfRange {
                source: s.source,
                n: usize::try_from(neurons).unwrap_or(usize::MAX),
            });
        }
        if s.t >= ticks {
            return Err(MetricError::ImpossibleSpike {
                t: s.t,
                source: s.source,
                why: "is on a tick at or past the end of the run",
            });
        }
    }
    // A `Train` is documented as sorted by `(t, source)`, so the usual case finds a repeat among
    // neighbours and allocates nothing. `Train::push` checks only that ticks do not go backwards,
    // so a hand-assembled train may be unordered *within* a tick; that case is sorted here rather
    // than trusted, because an unnoticed repeat is exactly the defect this function exists for.
    let repeat = if spikes.windows(2).all(|w| w[0] <= w[1]) {
        spikes.windows(2).find(|w| w[0] == w[1]).map(|w| w[0])
    } else {
        let mut sorted = spikes.to_vec();
        sorted.sort_unstable();
        sorted.windows(2).find(|w| w[0] == w[1]).map(|w| w[0])
    };
    if let Some(s) = repeat {
        return Err(MetricError::ImpossibleSpike {
            t: s.t,
            source: s.source,
            why: "repeats a neuron-timestep, and a neuron fires at most once per tick",
        });
    }
    Ok(())
}

/// Activation sparsity of a recorded spiking run: the fraction of neuron-timesteps on which
/// nothing fired.
///
/// A spiking activation is binary, so the general definition collapses to counting: there are
/// `neurons * ticks` activations and `train.len()` of them are one.
///
/// That collapse needs the train to be one run's worth of distinct neuron-timesteps, so the train
/// is checked against `neurons` and `ticks` before it is counted rather than assumed to be
/// well-formed — see [`MetricError::ImpossibleSpike`].
///
/// # Errors
///
/// [`MetricError::Empty`] when `neurons` or `ticks` is zero — no neuron-timesteps means no ratio;
/// [`MetricError::TooManySpikes`] when the train holds more spikes than there are neuron-timesteps,
/// which in this crate cannot happen from one run and usually means two trains were concatenated;
/// [`MetricError::SourceOutOfRange`] for a spike from a neuron the run does not have;
/// [`MetricError::ImpossibleSpike`] for a spike outside the run's ticks or a neuron-timestep named
/// twice; [`MetricError::Overflow`] if `neurons * ticks` exceeds `u64`.
pub fn activation_sparsity_of_run(
    train: &Train,
    neurons: u64,
    ticks: u64,
) -> Result<f64, MetricError> {
    if neurons == 0 || ticks == 0 {
        return Err(MetricError::Empty { what: "neuron-timesteps" });
    }
    let capacity = mul(neurons, ticks, "neuron-timesteps")?;
    let spikes = train.len() as u64;
    if spikes > capacity {
        return Err(MetricError::TooManySpikes { spikes, capacity });
    }
    check_one_run(train, neurons, ticks)?;
    Ok((capacity - spikes) as f64 / capacity as f64)
}

// ---------------------------------------------------------------------------------------------
// Synaptic operations
// ---------------------------------------------------------------------------------------------

/// Whether a layer's incoming activations are binary spikes or real numbers.
///
/// This is the flag that decides whether a synaptic operation is counted as an `AC` or a `MAC`,
/// and it is a declaration by the caller rather than something inferred from the data: a layer
/// whose activations happened to be all zeros and ones on one test sample is not thereby a spiking
/// layer. [`SynOpMeter::layer`] does check the declaration against the data and refuses a
/// `Spiking` layer carrying a `0.7`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationKind {
    /// Binary: every activation is exactly `0.0` or exactly `1.0`. The multiply by one is a no-op,
    /// so each operation is an **accumulate**.
    Spiking,
    /// Real-valued, including the graded spikes `Loihi 2` can send. Each operation is a
    /// **multiply-accumulate**.
    RealValued,
}

/// Synaptic operation counts, `NeuroBench`'s definition, kept split.
///
/// `dense` counts every connection on every timestep whether or not it did anything. The effective
/// counts include an operation only when the input activation **and** the weight are both non-zero
/// — the criterion is a conjunction, which is why activation sparsity and connection sparsity
/// compound rather than add.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SynOps {
    /// Operations a fully dense evaluation would perform: `fan_in * neurons` per layer per
    /// timestep, sparsity ignored. The denominator the effective counts are read against.
    pub dense: u64,
    /// Effective multiply-accumulates: operations whose input activation was real-valued.
    pub effective_macs: u64,
    /// Effective accumulates: operations whose input activation was a binary spike, so no
    /// multiplier was needed.
    pub effective_acs: u64,
}

impl SynOps {
    /// Effective operations of both kinds added together, or `None` when the sum does not fit in
    /// a `u64`.
    ///
    /// Provided as a named method rather than a field so that adding a `MAC` to an `AC` is
    /// something a caller writes deliberately. The two are not interchangeable in energy: an
    /// integer accumulate is cheaper than a multiply-accumulate at every bit-width anyone has
    /// published, and by how much depends on the datapath, which is why this crate declines to
    /// supply a ratio.
    ///
    /// # Why `Option` rather than a `u64`
    ///
    /// The fields are public and [`SynOps::add`] checks each of the three counts *separately*, so
    /// a pair of counts that each fit can still have a sum that does not. A bare `+` panics on
    /// that pair in a debug build and wraps in a release one, and a wrapped synaptic-operation
    /// count is precisely what [`MetricError::Overflow`]'s own doc says must not happen: a small
    /// number where an enormous one belongs. `None` says "does not fit", which is the only true
    /// answer; [`Footprint::bytes`] returns `None` on its own overflow for the same reason.
    #[must_use]
    pub fn effective_total(&self) -> Option<u64> {
        self.effective_macs.checked_add(self.effective_acs)
    }

    /// How many times fewer effective operations than dense ones, as `dense / effective`.
    ///
    /// `None` when the effective count is zero, and also when it does not fit in a `u64`. A
    /// network that did nothing has not achieved infinite efficiency; it has produced no output,
    /// and an infinity in a results table reads as the former.
    #[must_use]
    pub fn reduction(&self) -> Option<f64> {
        let e = self.effective_total()?;
        if e == 0 {
            return None;
        }
        Some(self.dense as f64 / e as f64)
    }

    /// Fraction of effective operations that were accumulates rather than multiply-accumulates.
    ///
    /// `None` when there were no effective operations at all, and when their sum does not fit in a
    /// `u64`. `1.0` for a purely binary spiking network, which is the regime the hardware argument
    /// assumes and which a hybrid model quietly leaves.
    #[must_use]
    pub fn ac_fraction(&self) -> Option<f64> {
        let e = self.effective_total()?;
        if e == 0 {
            return None;
        }
        Some(self.effective_acs as f64 / e as f64)
    }

    /// Add another layer's or timestep's counts into this one.
    ///
    /// # Errors
    ///
    /// [`MetricError::Overflow`] if any of the three counts exceeds `u64`.
    pub fn add(&mut self, other: Self) -> Result<(), MetricError> {
        self.dense =
            self.dense.checked_add(other.dense).ok_or(MetricError::Overflow { what: "dense ops" })?;
        self.effective_macs = self
            .effective_macs
            .checked_add(other.effective_macs)
            .ok_or(MetricError::Overflow { what: "effective MACs" })?;
        self.effective_acs = self
            .effective_acs
            .checked_add(other.effective_acs)
            .ok_or(MetricError::Overflow { what: "effective ACs" })?;
        Ok(())
    }
}

/// Accumulates [`SynOps`] across the layers and timesteps of a run.
///
/// Fed layer by layer with the **actual** activations and the **actual** weights, because the
/// effective count is a property of where the two sets of zeros coincide and cannot be recovered
/// from the two sparsities separately. A network that is 90% activation-sparse and 90%
/// connection-sparse has an effective count anywhere between 1% and 10% of dense depending on
/// whether the zeros line up, and reporting the product as if the two were independent is an
/// assumption about the model, not a measurement of it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SynOpMeter {
    /// Counts accumulated so far.
    pub ops: SynOps,
    /// Layer-timesteps recorded, so an empty meter is distinguishable from one that saw only
    /// silent layers.
    pub layers_seen: u64,
}

impl SynOpMeter {
    /// Record one dense layer on one timestep.
    ///
    /// `weights` is row-major by output neuron: the weight from input `i` to neuron `j` is
    /// `weights[j * activations.len() + i]`. That layout is the one a fully-connected layer has in
    /// memory in every framework this review checked, and stating it here is cheaper than
    /// discovering it from a transposed result that is still plausible — a transposed weight
    /// matrix on a square layer produces the *same* effective count, so the tests below use a
    /// non-square layer deliberately.
    ///
    /// # Errors
    ///
    /// [`MetricError::Empty`] if `activations` is empty or `neurons` is zero;
    /// [`MetricError::BadShape`] if `weights.len()` is not `activations.len() * neurons`;
    /// [`MetricError::NonFinite`] for a `NaN` or infinite activation or weight;
    /// [`MetricError::NotBinary`] if `kind` is [`ActivationKind::Spiking`] and an activation is
    /// neither `0.0` nor `1.0`; [`MetricError::Overflow`] on a count past `u64`.
    pub fn layer(
        &mut self,
        activations: &[f64],
        weights: &[f64],
        neurons: usize,
        kind: ActivationKind,
    ) -> Result<(), MetricError> {
        let ops = layer_synops(activations, weights, neurons, kind)?;
        self.ops.add(ops)?;
        self.layers_seen += 1;
        Ok(())
    }
}

/// Effective and dense synaptic operations for one dense layer on one timestep.
///
/// The counting rule is `NeuroBench`'s: an operation is effective when the input activation is
/// non-zero **and** the weight is non-zero. Everything else is dense-only.
///
/// # Errors
///
/// As [`SynOpMeter::layer`], which is this function with an accumulator around it.
pub fn layer_synops(
    activations: &[f64],
    weights: &[f64],
    neurons: usize,
    kind: ActivationKind,
) -> Result<SynOps, MetricError> {
    if activations.is_empty() {
        return Err(MetricError::Empty { what: "activations" });
    }
    if neurons == 0 {
        return Err(MetricError::Empty { what: "neurons" });
    }
    all_finite(activations, "activation")?;
    all_finite(weights, "weight")?;
    let fan_in = activations.len();
    let want = fan_in
        .checked_mul(neurons)
        .ok_or(MetricError::Overflow { what: "fan_in * neurons" })?;
    if weights.len() != want {
        return Err(MetricError::BadShape { got: weights.len(), want });
    }
    if kind == ActivationKind::Spiking {
        for (i, a) in activations.iter().enumerate() {
            if *a != 0.0 && *a != 1.0 {
                return Err(MetricError::NotBinary { index: i, value: *a });
            }
        }
    }

    let dense = mul(fan_in as u64, neurons as u64, "dense synaptic operations")?;
    let mut effective = 0u64;
    for i in 0..fan_in {
        if activations[i] == 0.0 {
            continue;
        }
        for j in 0..neurons {
            if weights[j * fan_in + i] != 0.0 {
                effective += 1;
            }
        }
    }
    Ok(match kind {
        ActivationKind::Spiking => SynOps { dense, effective_macs: 0, effective_acs: effective },
        ActivationKind::RealValued => SynOps { dense, effective_macs: effective, effective_acs: 0 },
    })
}

/// What a recorded spike train did to a [`Net`], counted the way `NeuroBench` counts and the way
/// [`Ledger`] counts, side by side.
///
/// The two do not agree by construction, and the gap is the interesting part — see
/// [`SpikeAccounting::in_flight`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SpikeAccounting {
    /// Synaptic operations in `NeuroBench`'s sense. All `ACs`, because a spike is binary.
    pub ops: SynOps,
    /// Deliveries the spikes in the train set in motion, delivered or not: one per outgoing
    /// synapse of every firing neuron.
    pub posted: u64,
    /// Deliveries that land on a tick inside the run. **This is what [`Ledger::syn_ops`] counts**,
    /// and the two are required to be equal in `the_ledger_and_the_metric_agree_on_a_real_run`.
    pub delivered: u64,
    /// Deliveries posted by a spike late enough that their arrival tick is past the end of the run.
    ///
    /// A run that stops does not deliver what is in flight, and the last `max_delay` ticks of any
    /// finite run therefore under-count. It is small, it is real, and this review did not locate it
    /// stated in any benchmark's methodology — which matters most for exactly the workloads where
    /// delays carry the computation.
    pub in_flight: u64,
    /// Deliveries that arrived across a synapse whose weight is exactly zero.
    ///
    /// Charged by [`Ledger`] — the hardware still fetched and still added — and **not** counted as
    /// an effective operation by `NeuroBench`, which is the whole of the difference between the two
    /// once `in_flight` is accounted for.
    pub zero_weight: u64,
}

/// One more of something, or an overflow error naming the counter.
fn bump(x: u64, what: &'static str) -> Result<u64, MetricError> {
    x.checked_add(1).ok_or(MetricError::Overflow { what })
}

/// Count what a recorded train did to a network, over a run of `ticks` ticks.
///
/// A spike emitted on tick `t` across a synapse of delay `d` arrives on tick `t + 1 + d`, which is
/// inside a run of `ticks` ticks only when `t + 1 + d < ticks`. That offset by one is
/// [`crate::sim::Sim::step`]'s posting rule, not a convention chosen here, and getting it wrong
/// moves the count by one tick's worth of deliveries in a way that no plot would show as wrong.
///
/// # Arithmetic at the end of `u64`
///
/// [`crate::spike::Spike::t`] is a public `u64` and [`crate::net::Net::delay`] a public `u32`, so
/// `t + 1 + d` can leave the range — `Train::from_spikes(vec![Spike { t: u64::MAX, source: 0 }])`
/// is an ordinary call and made the sum panic in a debug build and wrap in a release one. The
/// arrival tick is computed with checked adds, and an arrival that does not fit in a `u64` is past
/// the end of every finite run, so it counts as in flight.
///
/// # Errors
///
/// [`MetricError::SourceOutOfRange`] if the train names a neuron the network does not have, and
/// [`MetricError::Overflow`] if `n_syn * ticks` exceeds `u64` or a delivery counter does.
pub fn account_for_train(
    net: &Net,
    train: &Train,
    ticks: u64,
) -> Result<SpikeAccounting, MetricError> {
    let dense = mul(net.n_syn as u64, ticks, "dense synaptic operations")?;
    let mut acc = SpikeAccounting { ops: SynOps { dense, ..SynOps::default() }, ..Default::default() };
    for s in train.spikes() {
        if s.source as usize >= net.n {
            return Err(MetricError::SourceOutOfRange { source: s.source, n: net.n });
        }
        for (_post, w, d) in net.out_of(s.source as usize) {
            acc.posted = bump(acc.posted, "posted deliveries")?;
            let arrival = s.t.checked_add(1).and_then(|x| x.checked_add(u64::from(d)));
            match arrival {
                Some(at) if at < ticks => {
                    acc.delivered = bump(acc.delivered, "delivered spikes")?;
                    if w == 0.0 {
                        acc.zero_weight = bump(acc.zero_weight, "zero-weight deliveries")?;
                    } else {
                        acc.ops.effective_acs = bump(acc.ops.effective_acs, "effective ACs")?;
                    }
                }
                _ => acc.in_flight = bump(acc.in_flight, "in-flight deliveries")?,
            }
        }
    }
    Ok(acc)
}

// ---------------------------------------------------------------------------------------------
// Footprint and connection sparsity
// ---------------------------------------------------------------------------------------------

/// A model's memory, in bits, split into parameters and state.
///
/// `NeuroBench` defines footprint as the bytes needed to represent the model: its parameters —
/// weights, biases, learned thresholds — plus its buffers, which for a spiking network is the
/// per-neuron state a membrane potential and its adaptation variables occupy.
///
/// # Why this is stored in bits
///
/// Because the interesting models are quantised below a byte. A ternary weight is 1.58 bits of
/// information and is usually packed at 2; a binary one is 1. A footprint computed from a
/// framework's tensor sizes is byte-granular per tensor and cannot express any of that, so the
/// bit-width here is **declared by the caller**, and the doc on [`Footprint::new`] says what that
/// declaration means. The conversion to bytes rounds up once, on the total, and
/// [`Footprint::parameter_bytes`] and [`Footprint::state_bytes`] round up separately — so their sum
/// can exceed [`Footprint::bytes`] by one byte. That is arithmetic, not a defect, and it is stated
/// here because a table whose parts do not sum to its total invites a correction that would be
/// wrong.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Footprint {
    /// Bits occupied by learned parameters.
    pub parameter_bits: u64,
    /// Bits occupied by per-neuron state — membrane potentials, adaptation variables, refractory
    /// counters. Counted because a chip has to hold it and a footprint that omits it understates
    /// the model in proportion to how stateful it is.
    pub state_bits: u64,
}

impl Footprint {
    /// Build from counts and widths.
    ///
    /// `parameters` is a count of learned scalars at `parameter_bits` each; `state_values` is a
    /// count of per-neuron state scalars at `state_bits` each. The widths are the **stored** widths
    /// as the model would be deployed, not the widths a training framework happens to hold them in
    /// — a model trained in `f32` and deployed at 4 bits has a 4-bit footprint, and the whole
    /// reason to report the metric is to make that difference visible.
    ///
    /// # Errors
    ///
    /// [`MetricError::BadBitWidth`] if either width is zero or above 64, and
    /// [`MetricError::Overflow`] if either product exceeds `u64`.
    pub fn new(
        parameters: u64,
        parameter_bits: u32,
        state_values: u64,
        state_bits: u32,
    ) -> Result<Self, MetricError> {
        for bits in [parameter_bits, state_bits] {
            if bits == 0 || bits > 64 {
                return Err(MetricError::BadBitWidth { bits });
            }
        }
        Ok(Self {
            parameter_bits: mul(parameters, u64::from(parameter_bits), "parameter bits")?,
            state_bits: mul(state_values, u64::from(state_bits), "state bits")?,
        })
    }

    /// Total bytes, rounding the combined bit count up to the next whole byte.
    ///
    /// `None` on an overflow of the bit sum, which needs a model of more than two exabits.
    #[must_use]
    pub fn bytes(&self) -> Option<u64> {
        let bits = self.parameter_bits.checked_add(self.state_bits)?;
        Some(bits.div_ceil(8))
    }

    /// Bytes of parameters alone, rounded up.
    #[must_use]
    pub fn parameter_bytes(&self) -> u64 {
        self.parameter_bits.div_ceil(8)
    }

    /// Bytes of per-neuron state alone, rounded up.
    #[must_use]
    pub fn state_bytes(&self) -> u64 {
        self.state_bits.div_ceil(8)
    }
}

/// Connection sparsity: the fraction of a model's connections whose weight is exactly zero.
///
/// # Errors
///
/// [`MetricError::Empty`] when `total` is zero — a model with no connections has no ratio — and
/// [`MetricError::TooManyConnections`] when `zero > total`, which names both numbers in the unit
/// they were counted in.
pub fn connection_sparsity(zero: u64, total: u64) -> Result<f64, MetricError> {
    if total == 0 {
        return Err(MetricError::Empty { what: "connections" });
    }
    if zero > total {
        return Err(MetricError::TooManyConnections { counted: zero, declared: total });
    }
    Ok(zero as f64 / total as f64)
}

/// Connection sparsity of a [`Net`] against a stated dense connection count.
///
/// # Why the dense count has to be supplied
///
/// A [`Net`] is already sparse: it stores the synapses that exist and nothing about the pairs that
/// do not. So it cannot distinguish a connection that was **pruned to zero** — a parameter of the
/// model, counted in the denominator — from one that **never existed in the architecture** and is
/// not a parameter at all. `NeuroBench` computes connection sparsity against the model's declared
/// parameter tensors, where every absent connection is an explicit zero, so the denominator has to
/// come from the architecture.
///
/// For a fully-connected recurrent network of `n` neurons that is `n * n`. Supply what your
/// architecture declares; this function will not guess it, and a guess of `net.n_syn` would report
/// every network as 0% sparse, which is the flattering direction for a footprint claim and the
/// unflattering one for an efficiency claim.
///
/// # Parallel synapses are one connection, not several
///
/// A [`Net`] may hold several synapses between the same ordered pair — two edges `0 -> 1` at
/// delays 1 and 2 is how a delay line is built, and [`crate::net::NetBuilder::connect`] accepts
/// it. The denominator here counts *connections of the architecture*, `n * n` for a
/// fully-connected recurrent network, so the numerator has to as well: this function counts
/// distinct `(pre, post)` pairs, not [`Net::n_syn`]. Counting stored synapses instead subtracts
/// one connection too many from the absent ones for every extra delay line — a two-neuron network
/// with one pair wired twice reported `0.5` where the architecture has three of its four
/// connections absent, `0.75`.
///
/// A pair counts as zero only when **every** synapse on it is exactly zero. A pair with one live
/// delay line and one pruned to zero still carries signal, and is not an absent connection.
///
/// # Errors
///
/// [`MetricError::Empty`] if `dense_connections` is zero, [`MetricError::TooManyConnections`] if
/// the network connects more distinct pairs than the declared dense count allows.
pub fn connection_sparsity_of(net: &Net, dense_connections: u64) -> Result<f64, MetricError> {
    let (stored, stored_zero) = connection_pairs(net);
    if stored > dense_connections {
        return Err(MetricError::TooManyConnections {
            counted: stored,
            declared: dense_connections,
        });
    }
    connection_sparsity(dense_connections - stored + stored_zero, dense_connections)
}

/// Distinct ordered pairs a [`Net`] connects, and how many of those carry no non-zero weight.
///
/// Returned as `(pairs, zero_pairs)`. See [`connection_sparsity_of`] for why the unit is the pair
/// and not the stored synapse.
fn connection_pairs(net: &Net) -> (u64, u64) {
    let mut pairs = 0u64;
    let mut zero_pairs = 0u64;
    let mut row: Vec<(u32, bool)> = Vec::new();
    for pre in 0..net.n {
        row.clear();
        row.extend(net.out_of(pre).map(|(post, w, _)| (post, w == 0.0)));
        // `NetBuilder::build` leaves each CSR row in ascending `post` order, but `Net`'s fields
        // are public and a hand-assembled one need not be, so this sorts rather than trusting it.
        row.sort_unstable_by_key(|&(post, _)| post);
        let mut i = 0;
        while i < row.len() {
            let post = row[i].0;
            let mut all_zero = true;
            while i < row.len() && row[i].0 == post {
                all_zero &= row[i].1;
                i += 1;
            }
            pairs += 1;
            if all_zero {
                zero_pairs += 1;
            }
        }
    }
    (pairs, zero_pairs)
}

// ---------------------------------------------------------------------------------------------
// Correctness metrics
// ---------------------------------------------------------------------------------------------

/// Fraction of samples whose predicted class equals the true class, in `0.0..=1.0`.
///
/// # Errors
///
/// [`MetricError::LengthMismatch`] if the two arrays differ in length, and [`MetricError::Empty`]
/// on an empty test set — the accuracy of nothing is not 1.0, and not 0.0 either.
pub fn accuracy(predicted: &[usize], truth: &[usize]) -> Result<f64, MetricError> {
    if predicted.len() != truth.len() {
        return Err(MetricError::LengthMismatch { a: predicted.len(), b: truth.len() });
    }
    if truth.is_empty() {
        return Err(MetricError::Empty { what: "test set" });
    }
    let hits = predicted.iter().zip(truth).filter(|(p, t)| p == t).count();
    Ok(hits as f64 / truth.len() as f64)
}

/// Fraction of samples whose true class is among the `k` highest-scoring, in `0.0..=1.0`.
///
/// `scores` is row-major: sample `s`'s score for class `c` is `scores[s * classes + c]`.
///
/// # How ties are broken, which changes the number
///
/// A sample counts as correct when **fewer than `k` classes score strictly higher** than the true
/// class. That resolves ties in favour of the true class, so a model that scores every class
/// identically is credited with top-1 correctness on every sample. The alternative — breaking ties
/// by class index, which is what a `topk` on a tensor does — would credit it only when the true
/// class has the lowest index. Neither is wrong; they are different numbers, and an untrained model
/// with constant outputs is exactly where they diverge, which is exactly where a smoke test looks.
///
/// # Errors
///
/// [`MetricError::Empty`] on an empty test set or zero classes; [`MetricError::LengthMismatch`] if
/// `scores.len()` is not `truth.len() * classes`; [`MetricError::BadTopK`] if `k` is zero or above
/// `classes`; [`MetricError::ClassOutOfRange`] for a label past the class count;
/// [`MetricError::NonFinite`] for a `NaN` or infinite score.
pub fn top_k_accuracy(
    scores: &[f64],
    classes: usize,
    truth: &[usize],
    k: usize,
) -> Result<f64, MetricError> {
    if truth.is_empty() {
        return Err(MetricError::Empty { what: "test set" });
    }
    if classes == 0 {
        return Err(MetricError::Empty { what: "classes" });
    }
    if k == 0 || k > classes {
        return Err(MetricError::BadTopK { k, classes });
    }
    let want = truth
        .len()
        .checked_mul(classes)
        .ok_or(MetricError::Overflow { what: "samples * classes" })?;
    if scores.len() != want {
        return Err(MetricError::LengthMismatch { a: scores.len(), b: want });
    }
    all_finite(scores, "score")?;

    let mut hits = 0usize;
    for (s, &label) in truth.iter().enumerate() {
        if label >= classes {
            return Err(MetricError::ClassOutOfRange { index: s, label, classes });
        }
        let row = &scores[s * classes..(s + 1) * classes];
        let mine = row[label];
        let above = row.iter().filter(|x| **x > mine).count();
        if above < k {
            hits += 1;
        }
    }
    Ok(hits as f64 / truth.len() as f64)
}

/// Mean squared error between predictions and targets, in the square of whatever unit they carry.
///
/// # Errors
///
/// [`MetricError::LengthMismatch`], [`MetricError::Empty`] on no samples, and
/// [`MetricError::NonFinite`] for a `NaN` or infinite entry in either array.
pub fn mean_squared_error(predicted: &[f64], truth: &[f64]) -> Result<f64, MetricError> {
    if predicted.len() != truth.len() {
        return Err(MetricError::LengthMismatch { a: predicted.len(), b: truth.len() });
    }
    if truth.is_empty() {
        return Err(MetricError::Empty { what: "test set" });
    }
    all_finite(predicted, "prediction")?;
    all_finite(truth, "target")?;
    let sum: f64 = predicted.iter().zip(truth).map(|(p, t)| (p - t) * (p - t)).sum();
    Ok(sum / truth.len() as f64)
}

/// The coefficient of determination, `1 - SS_res / SS_tot`.
///
/// # What the two ends mean, and why they are exact here
///
/// `1.0` exactly when the prediction equals the target everywhere: `SS_res` is then exactly zero
/// and `1 - 0/SS_tot` is exactly `1.0`. `0.0` exactly when the prediction is the target's mean
/// everywhere: `SS_res` is then computed from the same terms in the same order as `SS_tot`, so the
/// two are bit-identical and the difference is exactly zero. Those two values are the definition's
/// anchors — one says "explains everything", the other "explains nothing, and is no better than a
/// constant" — and the tests pin both exactly rather than to a tolerance.
///
/// Values **below zero** are legal and common: a model worse than predicting the mean has a
/// negative R-squared, and clamping it to zero would hide the only case where the metric is telling
/// you something urgent. Above `1.0` is impossible.
///
/// # Errors
///
/// [`MetricError::LengthMismatch`], [`MetricError::Empty`], [`MetricError::NonFinite`] as
/// [`mean_squared_error`], and [`MetricError::ZeroVariance`] when every target is identical, in
/// which case the denominator is zero and the metric has no value at all.
pub fn r_squared(predicted: &[f64], truth: &[f64]) -> Result<f64, MetricError> {
    if predicted.len() != truth.len() {
        return Err(MetricError::LengthMismatch { a: predicted.len(), b: truth.len() });
    }
    if truth.is_empty() {
        return Err(MetricError::Empty { what: "test set" });
    }
    all_finite(predicted, "prediction")?;
    all_finite(truth, "target")?;
    let mean = truth.iter().sum::<f64>() / truth.len() as f64;
    // Both sums walk `truth` in the same order with the same shape of term, so predicting the mean
    // gives ss_res == ss_tot bit for bit and an R-squared of exactly zero.
    let ss_tot: f64 = truth.iter().map(|t| (t - mean) * (t - mean)).sum();
    let ss_res: f64 = predicted.iter().zip(truth).map(|(p, t)| (p - t) * (p - t)).sum();
    if ss_tot == 0.0 {
        return Err(MetricError::ZeroVariance);
    }
    Ok(1.0 - ss_res / ss_tot)
}

/// Average precision of one ranking: the mean of the precisions at the ranks where a positive was
/// retrieved.
///
/// `AP = (1 / P) * sum over positive ranks k of (positives in top k) / k`, with items sorted by
/// descending score and ties broken by ascending index. This is the **all-point, non-interpolated**
/// average precision — the area under the precision-recall curve as a ranking evaluation computes
/// it. `Pascal VOC`'s interpolated variant, which replaces each precision by the maximum precision
/// at any equal-or-greater recall, gives a slightly higher number on the same ranking; this
/// implementation does not compute that one.
///
/// # Errors
///
/// [`MetricError::LengthMismatch`], [`MetricError::Empty`], [`MetricError::NonFinite`] for a `NaN`
/// score, and [`MetricError::NoPositives`] when no item is positive — precision at every rank is
/// then `0/k` and recall is `0/0`, so there is nothing to average.
pub fn average_precision(scores: &[f64], labels: &[bool]) -> Result<f64, MetricError> {
    if scores.len() != labels.len() {
        return Err(MetricError::LengthMismatch { a: scores.len(), b: labels.len() });
    }
    if scores.is_empty() {
        return Err(MetricError::Empty { what: "ranking" });
    }
    all_finite(scores, "score")?;
    let positives = labels.iter().filter(|b| **b).count();
    if positives == 0 {
        return Err(MetricError::NoPositives { class: 0 });
    }
    let mut order: Vec<usize> = (0..scores.len()).collect();
    // A STABLE sort, so equal scores keep ascending index order — the tie-break stated in the doc.
    // `total_cmp` rather than `partial_cmp` because it is a total order and needs no unwrap; it
    // orders `-0.0` below `0.0`, which is the only place the two disagree on finite input.
    order.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));
    let mut hits = 0u64;
    let mut sum = 0.0;
    for (rank, &idx) in order.iter().enumerate() {
        if labels[idx] {
            hits += 1;
            sum += hits as f64 / (rank as f64 + 1.0);
        }
    }
    Ok(sum / positives as f64)
}

/// Mean average precision over classes: [`average_precision`] per class column, averaged over the
/// classes that have at least one positive.
///
/// `scores` and `labels` are row-major: sample `s`'s entry for class `c` is at `s * classes + c`.
///
/// # ⚠ This is not object-detection `mAP`
///
/// `NeuroBench`'s event-camera object-detection task reports `COCO`-style `mAP@[.5:.95]`, which
/// first matches predicted boxes to ground-truth boxes at ten intersection-over-union thresholds
/// and averages over those thresholds as well as over classes. This function does none of that: it
/// takes a score and a label per class per sample and has no notion of a box. Use it for
/// multi-label classification and retrieval, and say which one you computed, because the two share
/// a name and do not share a value.
///
/// Classes with no positive examples are **excluded** from the mean rather than scored zero.
/// Scoring them zero is the other common convention and it drags the mean down in proportion to how
/// many classes your test split happens to miss, which makes the metric depend on the split.
///
/// # Errors
///
/// [`MetricError::Empty`] on no samples or zero classes; [`MetricError::LengthMismatch`] if either
/// array is not `samples * classes` long; [`MetricError::NonFinite`] for a `NaN` score; and
/// [`MetricError::NoPositives`] with `class: 0` when **no** class has a positive, which means the
/// label matrix is empty of information.
pub fn mean_average_precision(
    scores: &[f64],
    labels: &[bool],
    classes: usize,
) -> Result<f64, MetricError> {
    if classes == 0 {
        return Err(MetricError::Empty { what: "classes" });
    }
    if scores.len() != labels.len() {
        return Err(MetricError::LengthMismatch { a: scores.len(), b: labels.len() });
    }
    if scores.is_empty() {
        return Err(MetricError::Empty { what: "test set" });
    }
    if !scores.len().is_multiple_of(classes) {
        return Err(MetricError::LengthMismatch { a: scores.len(), b: classes });
    }
    all_finite(scores, "score")?;
    let samples = scores.len() / classes;

    let mut sum = 0.0;
    let mut scored = 0usize;
    let mut col_scores = vec![0.0f64; samples];
    let mut col_labels = vec![false; samples];
    for c in 0..classes {
        for s in 0..samples {
            col_scores[s] = scores[s * classes + c];
            col_labels[s] = labels[s * classes + c];
        }
        match average_precision(&col_scores, &col_labels) {
            Ok(ap) => {
                sum += ap;
                scored += 1;
            }
            // A class with no positives is excluded, as the doc says. Any other error is a defect
            // in the inputs and propagates.
            Err(MetricError::NoPositives { .. }) => {}
            Err(e) => return Err(e),
        }
    }
    if scored == 0 {
        return Err(MetricError::NoPositives { class: 0 });
    }
    Ok(sum / scored as f64)
}

// ---------------------------------------------------------------------------------------------
// The bridge: benchmark numbers and the energy verdict from the same counters
// ---------------------------------------------------------------------------------------------

/// One run, reported as `NeuroBench`'s algorithm-track numbers **and** as this crate's energy
/// verdict, from the same counters.
///
/// The pairing is the point. Activation sparsity and the effective-operation reduction are the
/// numbers a benchmark submission carries; [`RunSummary::spikes_per_synapse`] and
/// [`RunSummary::verdicts`] are what the published crossover analyses say about whether the
/// resulting network can beat a dense one at all. A submission can look excellent on the first pair
/// and be refuted by every threshold in the second, and it is the same run.
#[derive(Debug, Clone, PartialEq)]
pub struct RunSummary {
    /// Ticks the run covered.
    pub ticks: u64,
    /// Neurons in the network.
    pub neurons: u64,
    /// Synapses in the network — stored once each, as [`Net::n_syn`].
    pub synapses: u64,
    /// Spikes recorded in the train.
    pub spikes: u64,
    /// Fraction of neuron-timesteps on which nothing fired, in `0.0..=1.0`.
    pub activation_sparsity: f64,
    /// Synaptic operations: dense against effective, with the `AC`/`MAC` split. All `ACs` for a
    /// binary spiking run.
    pub ops: SynOps,
    /// The delivery bookkeeping behind [`RunSummary::ops`], including what was still in flight when
    /// the run stopped.
    pub accounting: SpikeAccounting,
    /// [`Ledger::syn_ops`] from the same run, for comparison against
    /// [`SpikeAccounting::delivered`]. The two are computed by entirely separate paths — one by the
    /// simulator as it ran, one by re-reading the train against the network afterwards — so their
    /// agreement is a real check rather than a restatement.
    pub ledger_syn_ops: u64,
    /// Whether those two independent counts agree exactly.
    pub agrees_with_ledger: bool,
    /// Fraction of membrane updates that did no work, from [`Ledger::idle_fraction`]. `None` when
    /// nothing was updated.
    pub idle_fraction: Option<f64>,
    /// Spikes per synapse per inference — the quantity every published crossover threshold is
    /// stated in. `None` when there were no synapses or no inferences.
    pub spikes_per_synapse: Option<f64>,
    /// This workload against every threshold in [`crate::crossover`], as `(name, verdict)`. Empty
    /// when [`RunSummary::spikes_per_synapse`] has no answer.
    pub verdicts: Vec<(&'static str, Verdict)>,
    /// Model memory in bytes, as supplied to [`summarise`]. `None` only on a footprint whose bit
    /// count overflowed.
    pub footprint_bytes: Option<u64>,
}

impl RunSummary {
    /// The identity that ties this module to [`crate::crossover`]: effective operations over dense
    /// operations, per inference.
    ///
    /// With one inference per tick this equals [`RunSummary::spikes_per_synapse`] exactly whenever
    /// no synapse carries a zero weight, because both reduce to *deliveries over synapse-ticks*.
    /// The two communities compute the same ratio and give it different names, and having both
    /// printed from one run is how a reader notices.
    ///
    /// `None` when the dense count is zero, or when the effective count does not fit in a `u64`.
    #[must_use]
    pub fn effective_density(&self) -> Option<f64> {
        if self.ops.dense == 0 {
            return None;
        }
        Some(self.ops.effective_total()? as f64 / self.ops.dense as f64)
    }
}

impl fmt::Display for RunSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "{} neurons, {} synapses, {} ticks, {} spikes",
            self.neurons, self.synapses, self.ticks, self.spikes
        )?;
        writeln!(f, "  activation sparsity   {:.6}", self.activation_sparsity)?;
        match self.ops.effective_total() {
            Some(e) => writeln!(
                f,
                "  synaptic ops          dense {}, effective {} ({} AC + {} MAC)",
                self.ops.dense, e, self.ops.effective_acs, self.ops.effective_macs
            )?,
            None => writeln!(
                f,
                "  synaptic ops          dense {}, effective OVERFLOWED u64 ({} AC + {} MAC)",
                self.ops.dense, self.ops.effective_acs, self.ops.effective_macs
            )?,
        }
        writeln!(
            f,
            "  deliveries            posted {}, delivered {}, in flight at the end {}",
            self.accounting.posted, self.accounting.delivered, self.accounting.in_flight
        )?;
        // The cross-check is printed, not merely stored. A disagreement means the ledger and the
        // train are from different runs, and every line below this one — spikes per synapse and
        // all three published verdicts — is computed from the ledger. Reporting those while the
        // check silently reads `false` is the failure this pairing exists to prevent.
        if self.agrees_with_ledger {
            writeln!(
                f,
                "  ledger cross-check    {} delivered == {} billed",
                self.accounting.delivered, self.ledger_syn_ops
            )?;
        } else {
            writeln!(
                f,
                "  ledger cross-check    DISAGREES — {} delivered vs {} billed; the verdicts below are from the ledger, not from this train",
                self.accounting.delivered, self.ledger_syn_ops
            )?;
        }
        match self.spikes_per_synapse {
            Some(x) => writeln!(f, "  spikes per synapse    {x:.6} per inference")?,
            None => writeln!(f, "  spikes per synapse    no answer (no synapses or no inferences)")?,
        }
        for (name, v) in &self.verdicts {
            writeln!(f, "    vs {name}: {v:?}")?;
        }
        match self.footprint_bytes {
            Some(b) => writeln!(f, "  footprint             {b} bytes")?,
            None => writeln!(f, "  footprint             overflowed")?,
        }
        f.write_str("  energy                NOT MEASURED — the algorithm track has no energy metric")
    }
}

/// Compute every algorithm-track number this crate can get from one run, beside the crossover
/// verdict.
///
/// `inferences` is how many inferences the run represents — one per presented sample, which is
/// usually *not* one per tick, since a spiking classifier integrates over a window of ticks. Pass
/// `ticks` if the workload is a free-running simulation and every tick is its own answer.
///
/// # Errors
///
/// [`MetricError::Empty`] if `ticks` is zero or the network has no neurons;
/// [`MetricError::SourceOutOfRange`] if the train names a neuron outside the network;
/// [`MetricError::TooManySpikes`] if the train holds more spikes than neuron-timesteps;
/// [`MetricError::ImpossibleSpike`] if a spike falls outside the run's ticks or repeats a
/// neuron-timestep; [`MetricError::Overflow`] on a count past `u64`.
pub fn summarise(
    net: &Net,
    ledger: &Ledger,
    train: &Train,
    ticks: u64,
    inferences: u64,
    footprint: Footprint,
) -> Result<RunSummary, MetricError> {
    if net.n == 0 {
        return Err(MetricError::Empty { what: "neurons" });
    }
    let neurons = net.n as u64;
    let activation_sparsity = activation_sparsity_of_run(train, neurons, ticks)?;
    let accounting = account_for_train(net, train, ticks)?;
    let spikes_per_synapse = ledger.spikes_per_synapse(net.n_syn as u64, inferences);
    let verdicts = spikes_per_synapse
        .map(|sps| THRESHOLDS.iter().map(|(n, c)| (*n, c.verdict(sps))).collect())
        .unwrap_or_default();
    Ok(RunSummary {
        ticks,
        neurons,
        synapses: net.n_syn as u64,
        spikes: train.len() as u64,
        activation_sparsity,
        ops: accounting.ops,
        accounting,
        ledger_syn_ops: ledger.syn_ops,
        agrees_with_ledger: accounting.delivered == ledger.syn_ops,
        idle_fraction: ledger.idle_fraction(),
        spikes_per_synapse,
        verdicts,
        footprint_bytes: footprint.bytes(),
    })
}

#[cfg(test)]
mod tests {
    use super::{
        ActivationCensus, ActivationKind, Footprint, MetricError, SynOpMeter, SynOps,
        account_for_train, accuracy, activation_sparsity_of_run, average_precision,
        connection_sparsity, connection_sparsity_of, layer_synops, mean_average_precision,
        mean_squared_error, r_squared, summarise, top_k_accuracy,
    };
    use crate::crossover::Verdict;
    use crate::ledger::Ledger;
    use crate::net::NetBuilder;
    use crate::neuron::Lif;
    use crate::sim::{Mode, Sim};
    use crate::spike::{Spike, Train};

    // ---- (a) activation sparsity of a hand-constructed pattern ---------------------------------

    /// Six zeros in eight activations is three quarters, and `0.75` is exact in binary so the
    /// comparison can be `==` rather than a tolerance.
    #[test]
    fn activation_sparsity_of_a_hand_counted_pattern_is_exact() {
        let mut c = ActivationCensus::default();
        c.observe(&[0.0, 1.0, 0.0, 0.0]).unwrap();
        c.observe(&[2.5, 0.0, 0.0, 0.0]).unwrap();
        assert_eq!(c.total, 8);
        assert_eq!(c.zeros, 6);
        assert_eq!(c.sparsity().unwrap(), 0.75);
        assert_eq!(c.density().unwrap(), 0.25);
    }

    /// `density()` is the correctly rounded quotient `(total - zeros) / total`, and that is a
    /// DIFFERENT double from `1.0 - sparsity()` wherever the ratio is not exact in binary.
    ///
    /// The assertion this replaced was `sparsity() + density() == 1.0` on six zeros in eight, and
    /// it cannot fail: the sum is exactly one under both forms at every ratio tried — 6/8, 1/3,
    /// 2/3, 1/7, 3/10 — including the ratios where the two forms return different doubles. One
    /// zero in three does discriminate, so that is the case pinned here.
    #[test]
    fn the_density_is_the_quotient_and_not_one_minus_the_sparsity() {
        let mut c = ActivationCensus::default();
        c.observe(&[0.0, 1.0, 1.0]).unwrap();
        assert_eq!(c.zeros, 1);
        assert_eq!(c.total, 3);

        // The quotient, correctly rounded: the nearest double to 2/3.
        assert_eq!(c.density().unwrap(), 2.0 / 3.0);
        // One minus the sparsity is one ulp above it. If `density()` were implemented as
        // `1.0 - sparsity()` this assertion would fail, and the sum-is-one assertion would not.
        let subtracted = 1.0 - c.sparsity().unwrap();
        assert_ne!(subtracted, c.density().unwrap(), "the two forms coincided; pick another ratio");
        assert_eq!(subtracted.to_bits(), c.density().unwrap().to_bits() + 1);

        // And the ends, where both forms are exact and therefore cannot tell them apart — stated
        // so that the doc's claim about where they diverge is pinned from both sides.
        for (zeros, total) in [(0u64, 8u64), (8, 8)] {
            let c = ActivationCensus { zeros, total };
            assert_eq!(c.density().unwrap(), 1.0 - c.sparsity().unwrap());
        }
    }

    /// `observe_counts` is `observe` for a caller that never materialised the vector, and it must
    /// credit the FIRED activations as non-zero. Recording them as zeros instead — one `n - fired`
    /// written as `fired` — inverts the module's headline metric: a wide layer reporting 3 of 100
    /// fired would come out 0.03 sparse where the run was 0.97.
    #[test]
    fn observe_counts_credits_the_fired_activations_as_non_zero() {
        let mut c = ActivationCensus::default();
        c.observe_counts(3, 100).unwrap();
        assert_eq!(c.zeros, 97);
        assert_eq!(c.total, 100);
        assert_eq!(c.sparsity().unwrap(), 0.97);
        assert_eq!(c.density().unwrap(), 0.03);

        // The two ends, where the arms are as far apart as they get.
        let mut all_fired = ActivationCensus::default();
        all_fired.observe_counts(8, 8).unwrap();
        assert_eq!(all_fired.sparsity().unwrap(), 0.0);
        let mut none_fired = ActivationCensus::default();
        none_fired.observe_counts(0, 8).unwrap();
        assert_eq!(none_fired.sparsity().unwrap(), 1.0);
    }

    /// The counted path and the vector path are the same metric, so they must agree bit for bit on
    /// the same timestep.
    #[test]
    fn observe_counts_agrees_with_observe_on_the_same_timestep() {
        let activations = [0.0, 1.0, 0.0, 2.5, 0.0, -1.0];
        let mut by_vector = ActivationCensus::default();
        by_vector.observe(&activations).unwrap();
        let fired = activations.iter().filter(|x| **x != 0.0).count() as u64;
        let mut by_counts = ActivationCensus::default();
        by_counts.observe_counts(fired, activations.len() as u64).unwrap();
        assert_eq!(by_vector, by_counts);
        assert_eq!(by_vector.sparsity(), by_counts.sparsity());
    }

    /// More activations fired than existed is a timestep that did not happen, and it is refused
    /// without half-counting the census.
    #[test]
    fn observe_counts_refuses_more_fired_than_present() {
        let mut c = ActivationCensus::default();
        assert_eq!(
            c.observe_counts(9, 8).unwrap_err(),
            MetricError::TooManySpikes { spikes: 9, capacity: 8 }
        );
        assert_eq!(c, ActivationCensus::default(), "a rejected observation must change nothing");
        // And the boundary is inclusive: every activation firing is legal.
        c.observe_counts(8, 8).unwrap();
        assert_eq!(c.total, 8);
        assert_eq!(c.zeros, 0);
    }

    /// `merge` folds — two workers, two layers, two runs — and the fold is an addition, not a
    /// replacement and not a no-op. A merge that did nothing at all would leave the receiver
    /// reporting only its own share of the run.
    #[test]
    fn merge_adds_both_counts_rather_than_replacing_or_ignoring_them() {
        let mut a = ActivationCensus::default();
        a.observe(&[0.0, 0.0, 1.0]).unwrap(); // 2 of 3 zero
        let mut b = ActivationCensus::default();
        b.observe(&[0.0, 1.0, 1.0, 1.0, 1.0]).unwrap(); // 1 of 5 zero
        let before = a;
        a.merge(b).unwrap();
        assert_ne!(a, before, "the merge did nothing");
        assert_ne!(a, b, "the merge replaced rather than folded");
        assert_eq!(a.zeros, 3);
        assert_eq!(a.total, 8);
        assert_eq!(a.sparsity().unwrap(), 0.375);

        // Pooling, not averaging: the mean of the two per-census sparsities is 2/3 + 1/5 over 2,
        // which is 0.4333..., and it is NOT what a merged census reports.
        let mean_of_ratios = (2.0 / 3.0 + 1.0 / 5.0) / 2.0;
        assert!((a.sparsity().unwrap() - mean_of_ratios).abs() > 0.05);

        // Merging an empty census is the identity, and merging into an empty one is a copy.
        let kept = a;
        a.merge(ActivationCensus::default()).unwrap();
        assert_eq!(a, kept);
        let mut fresh = ActivationCensus::default();
        fresh.merge(kept).unwrap();
        assert_eq!(fresh, kept);
    }

    /// Both of `merge`'s counters overflow loudly rather than wrapping, and each is checked on its
    /// own — a census that wrapped its total would report a sparsity above one or below zero.
    #[test]
    fn merge_reports_overflow_rather_than_wrapping() {
        let mut zeros_full = ActivationCensus { zeros: u64::MAX, total: u64::MAX };
        assert_eq!(
            zeros_full.merge(ActivationCensus { zeros: 1, total: 0 }).unwrap_err(),
            MetricError::Overflow { what: "zero activations" }
        );
        let mut total_full = ActivationCensus { zeros: 0, total: u64::MAX };
        assert_eq!(
            total_full.merge(ActivationCensus { zeros: 0, total: 1 }).unwrap_err(),
            MetricError::Overflow { what: "total activations" }
        );
        let mut c = ActivationCensus { zeros: 0, total: u64::MAX };
        assert_eq!(
            c.observe_counts(0, 1).unwrap_err(),
            MetricError::Overflow { what: "total activations" }
        );
    }

    /// Negative zero is zero. IEEE 754 says `-0.0 == 0.0`, hardware that skips zeros skips it, and
    /// a census that treated it as activity would report a network as dense because of a sign bit.
    #[test]
    fn negative_zero_counts_as_a_zero_activation() {
        let mut c = ActivationCensus::default();
        c.observe(&[-0.0, 0.0, 1.0, -1.0]).unwrap();
        assert_eq!(c.zeros, 2);
    }

    /// The pooled ratio and the mean of the per-step ratios are different numbers, and this is the
    /// case that separates them. Step one is 1 of 2 zero (0.5); step two is 4 of 4 zero (1.0). The
    /// mean of the ratios is 0.75; the pooled ratio is 5/6 = 0.8333. `NeuroBench` pools.
    #[test]
    fn the_pooled_ratio_is_not_the_mean_of_the_per_step_ratios() {
        let mut c = ActivationCensus::default();
        c.observe(&[0.0, 1.0]).unwrap();
        c.observe(&[0.0, 0.0, 0.0, 0.0]).unwrap();
        let pooled = c.sparsity().unwrap();
        assert!((pooled - 5.0 / 6.0).abs() < 1e-15, "{pooled}");
        let mean_of_ratios = (0.5 + 1.0) / 2.0;
        assert!((pooled - mean_of_ratios).abs() > 0.08, "the two aggregations coincided");
    }

    // ---- (c) the degenerate ends ---------------------------------------------------------------

    /// A fully dense pattern is sparsity exactly 0 and a silent one exactly 1. These are where an
    /// off-by-one lives: a `<` written as `<=` in a counting loop moves one of them by `1/total`
    /// and nothing else changes.
    #[test]
    fn a_dense_pattern_is_zero_sparse_and_a_silent_one_is_one_sparse() {
        let mut dense = ActivationCensus::default();
        dense.observe(&[1.0, 2.0, -3.0, 4.0]).unwrap();
        assert_eq!(dense.sparsity().unwrap(), 0.0);

        let mut silent = ActivationCensus::default();
        silent.observe(&[0.0; 4]).unwrap();
        assert_eq!(silent.sparsity().unwrap(), 1.0);
    }

    /// The same two ends through the spike-train path, where the counting is over
    /// `neurons * ticks`.
    #[test]
    fn a_silent_train_is_exactly_one_sparse_and_a_saturated_one_exactly_zero() {
        let silent = Train::new();
        assert_eq!(activation_sparsity_of_run(&silent, 4, 5).unwrap(), 1.0);

        let mut every = Vec::new();
        for t in 0..5u64 {
            for s in 0..4u32 {
                every.push(Spike { t, source: s });
            }
        }
        let saturated = Train::from_spikes(every);
        assert_eq!(activation_sparsity_of_run(&saturated, 4, 5).unwrap(), 0.0);
    }

    /// A run of no ticks has no sparsity, rather than a perfect one.
    #[test]
    fn a_run_that_never_happened_has_no_sparsity() {
        let t = Train::new();
        assert!(matches!(
            activation_sparsity_of_run(&t, 4, 0),
            Err(MetricError::Empty { what: "neuron-timesteps" })
        ));
        assert!(ActivationCensus::default().sparsity().is_none());
    }

    /// More spikes than neuron-timesteps is a run that did not happen, and it is refused rather
    /// than producing a negative sparsity.
    #[test]
    fn a_train_that_could_not_have_happened_is_refused() {
        let t = Train::from_spikes(vec![
            Spike { t: 0, source: 0 },
            Spike { t: 0, source: 1 },
            Spike { t: 1, source: 0 },
        ]);
        let e = activation_sparsity_of_run(&t, 1, 2).unwrap_err();
        assert_eq!(e, MetricError::TooManySpikes { spikes: 3, capacity: 2 });
    }

    /// The SUBTLE form of the same error, which the capacity guard above cannot see. Three spikes
    /// on `(t = 0, source = 0)` is three spikes in a capacity of ten, so it passes the count
    /// check — and it credits one neuron-timestep three times, returning 0.7 where the run that
    /// could have produced those spikes was 0.9 sparse. Two recordings of one run merged with
    /// `Train::from_spikes` is exactly this shape, and neither `from_spikes` nor `Train::push`
    /// enforces one spike per neuron per tick.
    #[test]
    fn a_neuron_timestep_named_twice_is_refused_rather_than_counted_twice() {
        let repeated = Train::from_spikes(vec![
            Spike { t: 0, source: 0 },
            Spike { t: 0, source: 0 },
            Spike { t: 0, source: 0 },
        ]);
        assert!(repeated.len() as u64 <= 2 * 5, "the fixture must be under capacity to be the case");
        assert_eq!(
            activation_sparsity_of_run(&repeated, 2, 5).unwrap_err(),
            MetricError::ImpossibleSpike {
                t: 0,
                source: 0,
                why: "repeats a neuron-timestep, and a neuron fires at most once per tick",
            }
        );

        // The run those three spikes could have come from — one of them — is 9 of 10 silent.
        let once = Train::from_spikes(vec![Spike { t: 0, source: 0 }]);
        assert_eq!(activation_sparsity_of_run(&once, 2, 5).unwrap(), 0.9);

        // A repeat that is NOT adjacent in the stored order, because the train was assembled by
        // `push`, which checks only that ticks do not go backwards. Sources 2, 1, 2 within tick 0.
        let mut unordered = Train::new();
        for source in [2u32, 1, 2] {
            unordered.push(Spike { t: 0, source });
        }
        assert_eq!(
            activation_sparsity_of_run(&unordered, 4, 5).unwrap_err(),
            MetricError::ImpossibleSpike {
                t: 0,
                source: 2,
                why: "repeats a neuron-timestep, and a neuron fires at most once per tick",
            }
        );
        // Distinct sources in a non-ascending order are a legal train and are counted.
        let mut distinct = Train::new();
        for source in [2u32, 1, 3] {
            distinct.push(Spike { t: 0, source });
        }
        assert_eq!(activation_sparsity_of_run(&distinct, 4, 5).unwrap(), 17.0 / 20.0);
    }

    /// A spike on a tick the run does not contain is refused too. It is the same defect wearing a
    /// different hat — a tick count taken from the wrong variable, or the second half of a train
    /// measured against the first half's length — and counting it removes a neuron-timestep from
    /// the silent pile that was never in the window.
    #[test]
    fn a_spike_outside_the_runs_ticks_is_refused() {
        let late = Train::from_spikes(vec![Spike { t: 0, source: 0 }, Spike { t: 100, source: 1 }]);
        assert_eq!(
            activation_sparsity_of_run(&late, 2, 5).unwrap_err(),
            MetricError::ImpossibleSpike {
                t: 100,
                source: 1,
                why: "is on a tick at or past the end of the run",
            }
        );
        // The last legal tick is `ticks - 1`, and it is legal.
        let last = Train::from_spikes(vec![Spike { t: 4, source: 1 }]);
        assert_eq!(activation_sparsity_of_run(&last, 2, 5).unwrap(), 0.9);
        // A neuron the run does not have is named as such.
        let absent = Train::from_spikes(vec![Spike { t: 0, source: 7 }]);
        assert_eq!(
            activation_sparsity_of_run(&absent, 2, 5).unwrap_err(),
            MetricError::SourceOutOfRange { source: 7, n: 2 }
        );
    }

    // ---- (b) effective synaptic operations by hand ----------------------------------------------

    /// A 4-input, 3-neuron layer, counted by hand.
    ///
    /// Activations `[1, 0, 1, 1]`; weights row-major by neuron:
    /// `n0 = [0.5, 0.5, 0.0, 0.5]`, `n1 = [0.0, 1.0, 2.0, 0.0]`, `n2 = [1, 1, 1, 1]`.
    /// Dense is 4*3 = 12. For input 0 the non-zero weights are n0 and n2 (2); for input 2 they are
    /// n1 and n2 (2); for input 3 they are n0 and n2 (2). Input 1 is silent and contributes
    /// nothing. Effective = 6, all accumulates.
    ///
    /// The layer is deliberately NOT square: a transposed weight matrix gives the same count on a
    /// square layer, so a square fixture would pass with the indexing wrong.
    #[test]
    fn effective_synaptic_operations_match_a_hand_count() {
        let a = [1.0, 0.0, 1.0, 1.0];
        let w = [
            0.5, 0.5, 0.0, 0.5, // neuron 0
            0.0, 1.0, 2.0, 0.0, // neuron 1
            1.0, 1.0, 1.0, 1.0, // neuron 2
        ];
        let ops = layer_synops(&a, &w, 3, ActivationKind::Spiking).unwrap();
        assert_eq!(ops.dense, 12);
        assert_eq!(ops.effective_acs, 6);
        assert_eq!(ops.effective_macs, 0);
        assert_eq!(ops.effective_total(), Some(6));
        assert_eq!(ops.reduction().unwrap(), 2.0);
        assert_eq!(ops.ac_fraction().unwrap(), 1.0);
    }

    /// The same layer with real-valued inputs books the identical operations as `MACs`. The counts
    /// move between the two fields and the total does not, which is the split doing its job.
    #[test]
    fn the_same_operations_are_macs_when_the_input_is_real_valued() {
        let a = [0.3, 0.0, -2.0, 7.5];
        let w = [
            0.5, 0.5, 0.0, 0.5, //
            0.0, 1.0, 2.0, 0.0, //
            1.0, 1.0, 1.0, 1.0, //
        ];
        let ops = layer_synops(&a, &w, 3, ActivationKind::RealValued).unwrap();
        assert_eq!(ops.effective_macs, 6);
        assert_eq!(ops.effective_acs, 0);
        assert_eq!(ops.ac_fraction().unwrap(), 0.0);
    }

    /// A layer declared spiking whose activation is 0.7 is refused, naming the index and the value.
    /// Counting it as an accumulate would charge a multiply-accumulate at accumulate prices.
    #[test]
    fn a_graded_activation_in_a_spiking_layer_is_refused() {
        let a = [1.0, 0.7];
        let w = [1.0, 1.0];
        let e = layer_synops(&a, &w, 1, ActivationKind::Spiking).unwrap_err();
        assert_eq!(e, MetricError::NotBinary { index: 1, value: 0.7 });
        // And the same data is fine once it says what it is.
        assert!(layer_synops(&a, &w, 1, ActivationKind::RealValued).is_ok());
    }

    /// Both ends of the synaptic-operation count: everything firing through a full matrix gives
    /// effective == dense, and nothing firing gives effective == 0 with no reduction figure.
    #[test]
    fn a_fully_active_layer_matches_dense_and_a_silent_one_refuses_a_ratio() {
        let full = layer_synops(&[1.0; 3], &[1.0; 12], 4, ActivationKind::Spiking).unwrap();
        assert_eq!(full.dense, 12);
        assert_eq!(full.effective_acs, 12);
        assert_eq!(full.reduction().unwrap(), 1.0);

        let silent = layer_synops(&[0.0; 3], &[1.0; 12], 4, ActivationKind::Spiking).unwrap();
        assert_eq!(silent.effective_total(), Some(0));
        assert!(silent.reduction().is_none(), "a silent network is not infinitely efficient");
        assert!(silent.ac_fraction().is_none());
    }

    /// A zero weight removes the operation even when the activation fired. The two sparsities
    /// compound through a conjunction, which is why they cannot be multiplied from the margins.
    #[test]
    fn a_zero_weight_removes_an_operation_a_firing_activation_would_have_caused() {
        let a = [1.0, 1.0];
        let all = layer_synops(&a, &[1.0, 1.0, 1.0, 1.0], 2, ActivationKind::Spiking).unwrap();
        assert_eq!(all.effective_acs, 4);
        let pruned = layer_synops(&a, &[1.0, 0.0, 0.0, 1.0], 2, ActivationKind::Spiking).unwrap();
        assert_eq!(pruned.effective_acs, 2);
        assert_eq!(pruned.dense, 4, "dense does not notice pruning; that is what it is for");
    }

    /// A shape that does not match is an error naming both lengths, not a truncated count.
    #[test]
    fn a_misshapen_weight_matrix_is_refused() {
        let e = layer_synops(&[1.0, 1.0], &[1.0; 5], 2, ActivationKind::Spiking).unwrap_err();
        assert_eq!(e, MetricError::BadShape { got: 5, want: 4 });
    }

    /// The meter is the accumulator around the same function, and it must add rather than replace.
    #[test]
    fn the_meter_accumulates_across_layers_and_timesteps() {
        let mut m = SynOpMeter::default();
        for _ in 0..3 {
            m.layer(&[1.0, 0.0], &[1.0, 1.0, 1.0, 1.0], 2, ActivationKind::Spiking).unwrap();
        }
        assert_eq!(m.layers_seen, 3);
        assert_eq!(m.ops.dense, 12);
        assert_eq!(m.ops.effective_acs, 6);
    }

    /// A non-finite activation is refused at the boundary with its index, rather than becoming a
    /// `NaN` somewhere downstream of the sum.
    #[test]
    fn a_non_finite_input_is_named_rather_than_propagated() {
        let e = layer_synops(&[1.0, f64::NAN], &[1.0; 2], 1, ActivationKind::Spiking).unwrap_err();
        assert_eq!(e, MetricError::NonFinite { what: "activation", index: 1 });
        let mut c = ActivationCensus::default();
        assert!(c.observe(&[0.0, f64::INFINITY]).is_err());
        assert_eq!(c.total, 0, "a rejected observation must not have been half-counted");
    }

    /// A non-finite WEIGHT is refused as well, and named as a weight. Without the guard a `NaN`
    /// weight passes the `!= 0.0` test — `NaN` is not equal to anything, including zero — and
    /// silently inflates the effective count by counting a weight that is not a number as a live
    /// synapse. Both `# Errors` blocks promise this error; only the activation half was pinned.
    #[test]
    fn a_non_finite_weight_is_named_rather_than_counted_as_live() {
        let e = layer_synops(&[1.0, 1.0], &[1.0, f64::NAN], 1, ActivationKind::Spiking).unwrap_err();
        assert_eq!(e, MetricError::NonFinite { what: "weight", index: 1 });
        let e = layer_synops(&[1.0], &[f64::INFINITY], 1, ActivationKind::RealValued).unwrap_err();
        assert_eq!(e, MetricError::NonFinite { what: "weight", index: 0 });

        // What the guard prevents: with the same shape and a finite zero in that slot, the
        // effective count is 1 — so an unguarded `NaN` would have reported 2.
        let ok = layer_synops(&[1.0, 1.0], &[1.0, 0.0], 1, ActivationKind::Spiking).unwrap();
        assert_eq!(ok.effective_acs, 1);

        // And through the meter, which documents the same error.
        let mut m = SynOpMeter::default();
        assert!(m.layer(&[1.0], &[f64::NAN], 1, ActivationKind::Spiking).is_err());
        assert_eq!(m.layers_seen, 0, "a rejected layer must not have been counted");
    }

    // ---- (d) footprint arithmetic ----------------------------------------------------------------

    /// 1,200,000 weights at 4 bits is 4,800,000 bits is 600,000 bytes; 1,000 neurons holding one
    /// 32-bit membrane potential each is 32,000 bits is 4,000 bytes; the total is 604,000 bytes.
    /// Every step of that is exact integer arithmetic and the test does it by hand.
    #[test]
    fn footprint_is_exact_integer_arithmetic_at_a_stated_bit_width() {
        let f = Footprint::new(1_200_000, 4, 1_000, 32).unwrap();
        assert_eq!(f.parameter_bits, 4_800_000);
        assert_eq!(f.state_bits, 32_000);
        assert_eq!(f.parameter_bytes(), 600_000);
        assert_eq!(f.state_bytes(), 4_000);
        assert_eq!(f.bytes().unwrap(), 604_000);
    }

    /// Sub-byte widths round UP, once, on the total: 7 parameters at 3 bits is 21 bits, which is 3
    /// bytes, not 2. A byte-granular footprint cannot express this case at all, which is the reason
    /// this type stores bits.
    #[test]
    fn a_sub_byte_width_rounds_up_on_the_total() {
        let f = Footprint::new(7, 3, 0, 8).unwrap();
        assert_eq!(f.parameter_bits, 21);
        assert_eq!(f.bytes().unwrap(), 3);

        // And the documented consequence: separately rounded parts can exceed the rounded total.
        let g = Footprint::new(1, 4, 1, 4).unwrap();
        assert_eq!(g.bytes().unwrap(), 1);
        assert_eq!(g.parameter_bytes() + g.state_bytes(), 2);
    }

    /// A ternary model packed at 2 bits against the same model at `f32`: exactly 16x, which is the
    /// kind of claim a footprint metric exists to make checkable.
    #[test]
    fn quantisation_shows_up_in_the_footprint_as_an_exact_ratio() {
        let wide = Footprint::new(1_000_000, 32, 0, 8).unwrap();
        let packed = Footprint::new(1_000_000, 2, 0, 8).unwrap();
        assert_eq!(wide.bytes().unwrap(), 4_000_000);
        assert_eq!(packed.bytes().unwrap(), 250_000);
        assert_eq!(wide.bytes().unwrap() / packed.bytes().unwrap(), 16);
    }

    #[test]
    fn a_zero_or_oversized_bit_width_is_refused() {
        assert_eq!(Footprint::new(10, 0, 0, 8).unwrap_err(), MetricError::BadBitWidth { bits: 0 });
        assert_eq!(Footprint::new(10, 65, 0, 8).unwrap_err(), MetricError::BadBitWidth { bits: 65 });
    }

    // ---- connection sparsity ---------------------------------------------------------------------

    #[test]
    fn connection_sparsity_is_zeros_over_declared_connections() {
        assert_eq!(connection_sparsity(750, 1_000).unwrap(), 0.75);
        assert_eq!(connection_sparsity(0, 1_000).unwrap(), 0.0);
        assert_eq!(connection_sparsity(1_000, 1_000).unwrap(), 1.0);
        assert!(connection_sparsity(1, 0).is_err());
        assert!(connection_sparsity(2, 1).is_err());
    }

    /// The error a connection-sparsity path raises must name CONNECTIONS. It used to borrow
    /// `TooManySpikes`, so `connection_sparsity(16, 2)` told a user "16 spikes in 2
    /// neuron-timesteps" about a weight count, which sends them to the wrong half of a pipeline
    /// that has both.
    #[test]
    fn a_connection_error_names_connections_and_not_spikes() {
        let e = connection_sparsity(16, 2).unwrap_err();
        assert_eq!(e, MetricError::TooManyConnections { counted: 16, declared: 2 });
        let text = e.to_string();
        assert!(text.contains("connections"), "{text}");
        assert!(!text.contains("spike"), "{text}");
        assert!(!text.contains("neuron-timestep"), "{text}");

        let mut b = NetBuilder::new(4);
        b.connect(0, 1, 5e-3, 1).unwrap();
        b.connect(1, 2, 5e-3, 1).unwrap();
        b.connect(2, 3, 5e-3, 1).unwrap();
        let net = b.build();
        let e = connection_sparsity_of(&net, 2).unwrap_err();
        assert_eq!(e, MetricError::TooManyConnections { counted: 3, declared: 2 });
        assert!(!e.to_string().contains("spike"), "{e}");

        // The spike path keeps its own nouns.
        let spikes = MetricError::TooManySpikes { spikes: 3, capacity: 2 }.to_string();
        assert!(spikes.contains("spikes") && spikes.contains("neuron-timesteps"), "{spikes}");
    }

    /// Two synapses between one ordered pair — an ordinary delay line — are ONE connection of the
    /// architecture, not two. Counting `Net::n_syn` instead subtracts one connection too many from
    /// the absent pile: this two-neuron network reported 0.5 where three of its four connections
    /// are absent, 0.75.
    #[test]
    fn parallel_synapses_between_one_pair_are_one_connection() {
        let mut b = NetBuilder::new(2);
        b.connect(0, 1, 5e-3, 1).unwrap();
        b.connect(0, 1, 5e-3, 2).unwrap(); // the same pair, a second delay line
        let net = b.build();
        assert_eq!(net.n_syn, 2, "the builder accepts the parallel edge");
        assert_eq!(connection_sparsity_of(&net, 4).unwrap(), 0.75);

        // A pair counts as ZERO only when every synapse on it is zero. One live line and one
        // pruned still carries signal.
        let mut mixed = NetBuilder::new(2);
        mixed.connect(0, 1, 5e-3, 1).unwrap();
        mixed.connect(0, 1, 0.0, 2).unwrap();
        assert_eq!(connection_sparsity_of(&mixed.build(), 4).unwrap(), 0.75);

        let mut both_zero = NetBuilder::new(2);
        both_zero.connect(0, 1, 0.0, 1).unwrap();
        both_zero.connect(0, 1, 0.0, 2).unwrap();
        assert_eq!(connection_sparsity_of(&both_zero.build(), 4).unwrap(), 1.0);

        // And the declared-count guard counts pairs too: two pairs are more than one declared
        // connection, three parallel edges on one pair are not.
        let mut two_pairs = NetBuilder::new(2);
        two_pairs.connect(0, 1, 5e-3, 1).unwrap();
        two_pairs.connect(1, 0, 5e-3, 1).unwrap();
        assert!(connection_sparsity_of(&two_pairs.build(), 1).is_err());
        let mut one_pair = NetBuilder::new(2);
        for d in 1..4 {
            one_pair.connect(0, 1, 5e-3, d).unwrap();
        }
        assert_eq!(connection_sparsity_of(&one_pair.build(), 1).unwrap(), 0.0);
    }

    /// A sparse `Net` does not store the absent connections, so the dense count is supplied. Four
    /// neurons fully connected is 16 declared connections; three stored, one of them zero-weighted,
    /// leaves 14 zeros — 13 never stored plus the stored zero.
    #[test]
    fn a_nets_connection_sparsity_is_measured_against_a_declared_dense_count() {
        let mut b = NetBuilder::new(4);
        b.connect(0, 1, 5e-3, 1).unwrap();
        b.connect(1, 2, 0.0, 1).unwrap();
        b.connect(2, 3, 5e-3, 1).unwrap();
        let net = b.build();
        assert_eq!(net.n_syn, 3);
        let s = connection_sparsity_of(&net, 16).unwrap();
        assert_eq!(s, 14.0 / 16.0);
        // Declaring fewer dense connections than the network stores is refused rather than
        // producing a negative sparsity.
        assert!(connection_sparsity_of(&net, 2).is_err());
    }

    // ---- correctness metrics ----------------------------------------------------------------------

    #[test]
    fn accuracy_counts_exact_matches() {
        assert_eq!(accuracy(&[0, 1, 2, 3], &[0, 1, 2, 3]).unwrap(), 1.0);
        assert_eq!(accuracy(&[0, 1, 2, 3], &[3, 2, 1, 0]).unwrap(), 0.0);
        assert_eq!(accuracy(&[0, 1, 9, 9], &[0, 1, 2, 3]).unwrap(), 0.5);
        assert!(accuracy(&[], &[]).is_err(), "the accuracy of nothing is not 1.0");
        assert!(accuracy(&[0], &[0, 1]).is_err());
    }

    /// Top-1 must reduce to plain accuracy on the arg-max, and top-`k` must be monotone in `k` and
    /// reach exactly 1.0 when `k` is the class count — the two ends that pin the definition.
    #[test]
    fn top_k_accuracy_is_monotone_and_saturates_at_the_class_count() {
        // Three samples, four classes. Row-major.
        let scores = [
            0.1, 0.9, 0.3, 0.2, // arg-max 1
            0.5, 0.2, 0.1, 0.0, // arg-max 0
            0.0, 0.1, 0.2, 0.8, // arg-max 3
        ];
        let truth = [1usize, 2, 3];
        assert_eq!(top_k_accuracy(&scores, 4, &truth, 1).unwrap(), 2.0 / 3.0);
        // Sample 1's true class (2) is third by score, so it enters at k = 3.
        assert_eq!(top_k_accuracy(&scores, 4, &truth, 2).unwrap(), 2.0 / 3.0);
        assert_eq!(top_k_accuracy(&scores, 4, &truth, 3).unwrap(), 1.0);
        assert_eq!(top_k_accuracy(&scores, 4, &truth, 4).unwrap(), 1.0);
        assert!(top_k_accuracy(&scores, 4, &truth, 5).is_err());
        assert!(top_k_accuracy(&scores, 4, &truth, 0).is_err());
        assert!(top_k_accuracy(&scores, 4, &[1, 2, 9], 1).is_err(), "label 9 of 4 classes");
    }

    /// The documented tie-break, exhibited. An untrained model scoring every class the same is
    /// credited as top-1 correct here, and would be credited on one class in four under an
    /// index-order tie-break. The number is a choice; this pins which choice.
    #[test]
    fn ties_resolve_in_favour_of_the_true_class() {
        let scores = [0.25, 0.25, 0.25, 0.25];
        assert_eq!(top_k_accuracy(&scores, 4, &[3], 1).unwrap(), 1.0);
    }

    #[test]
    fn mean_squared_error_is_the_mean_of_the_squares() {
        // Errors 1, -1, 2, 0 -> squares 1, 1, 4, 0 -> mean 6/4 = 1.5, exact in binary.
        let p = [1.0, 1.0, 4.0, 5.0];
        let t = [0.0, 2.0, 2.0, 5.0];
        assert_eq!(mean_squared_error(&p, &t).unwrap(), 1.5);
        assert_eq!(mean_squared_error(&t, &t).unwrap(), 0.0);
        assert!(mean_squared_error(&[f64::NAN], &[0.0]).is_err());
    }

    // ---- (e) R-squared at both ends ---------------------------------------------------------------

    /// R-squared is EXACTLY 1 for a perfect fit and EXACTLY 0 for predicting the mean. Those two
    /// values pin both ends of the definition: the first says the residual sum is zero, the second
    /// says the residual sum equals the total sum. A tolerance here would let a swapped numerator
    /// and denominator through on symmetric data.
    #[test]
    fn r_squared_is_exactly_one_for_a_perfect_fit_and_exactly_zero_for_the_mean() {
        let truth = [1.0, 2.0, 3.0, 4.0]; // mean 2.5, exact in binary
        assert_eq!(r_squared(&truth, &truth).unwrap(), 1.0);

        let mean_pred = [2.5, 2.5, 2.5, 2.5];
        assert_eq!(r_squared(&mean_pred, &truth).unwrap(), 0.0);
    }

    /// A model worse than the mean has a NEGATIVE R-squared, and it is reported rather than
    /// clamped. Reflecting each target through the mean, `pred = 2*mean - y`, makes every residual
    /// exactly twice the mean-predictor's, so `SS_res` is 4x `SS_tot` and R-squared is exactly -3.
    #[test]
    fn a_model_worse_than_the_mean_reports_a_negative_r_squared() {
        let truth = [1.0, 2.0, 3.0, 4.0];
        let mean = 2.5;
        let pred: Vec<f64> = truth.iter().map(|y| 2.0 * mean - y).collect();
        assert_eq!(r_squared(&pred, &truth).unwrap(), -3.0);
    }

    /// Constant targets have no variance to explain, so the metric has no value — not 0.0, which
    /// would read as "no better than the mean" when the prediction is in fact perfect, and not 1.0,
    /// which would read as a triumph over a problem with no content.
    #[test]
    fn constant_targets_have_no_r_squared() {
        let t = [7.0, 7.0, 7.0];
        assert_eq!(r_squared(&t, &t).unwrap_err(), MetricError::ZeroVariance);
    }

    // ---- average precision -------------------------------------------------------------------------

    /// The ranking `[+, -, +, +, -]` has precisions 1/1, 2/3, 3/4 at its three positive ranks, so
    /// `AP = (1 + 2/3 + 3/4) / 3 = 29/36`. Computed by hand on paper and compared as a fraction.
    #[test]
    fn average_precision_matches_a_hand_computed_ranking() {
        let scores = [0.9, 0.8, 0.7, 0.6, 0.5];
        let labels = [true, false, true, true, false];
        let ap = average_precision(&scores, &labels).unwrap();
        assert!((ap - 29.0 / 36.0).abs() < 1e-15, "{ap}");
    }

    /// A perfect ranking is exactly 1.0 — every positive rank has precision `k/k` — and a single
    /// positive at the bottom of `n` items is exactly `1/n`. The two ends of the definition.
    #[test]
    fn a_perfect_ranking_is_one_and_a_worst_case_is_one_over_n() {
        let scores = [0.9, 0.8, 0.7, 0.6];
        assert_eq!(average_precision(&scores, &[true, true, false, false]).unwrap(), 1.0);
        assert_eq!(average_precision(&scores, &[false, false, false, true]).unwrap(), 0.25);
    }

    /// The documented tie-break, exhibited: equal scores keep ASCENDING INDEX order, and that
    /// choice moves the number. On two tied items labelled `[false, true]` the documented order
    /// puts the positive second, precision 1/2, AP 0.5; the reverse order puts it first, precision
    /// 1/1, AP 1.0. Every other fixture in this module uses strictly distinct scores, so none of
    /// them can see this.
    #[test]
    fn average_precision_breaks_ties_by_ascending_index() {
        assert_eq!(average_precision(&[1.0, 1.0], &[false, true]).unwrap(), 0.5);
        assert_eq!(average_precision(&[1.0, 1.0], &[true, false]).unwrap(), 1.0);

        // A tie wide enough that an unstable sort actually permutes it, with the answer pinned to
        // a closed form. Forty items: the even indices score 1.0 and are all negative, the odd
        // indices score 0.5 and the first five of THOSE are positive. Under ascending-index ties
        // the positives land at ranks 21..=25, so
        // `AP = (1/21 + 2/22 + 3/23 + 4/24 + 5/25) / 5`.
        let n = 40usize;
        let scores: Vec<f64> = (0..n).map(|i| if i % 2 == 0 { 1.0 } else { 0.5 }).collect();
        let labels: Vec<bool> = (0..n).map(|i| i % 2 == 1 && i <= 9).collect();
        let want = (1.0 / 21.0 + 2.0 / 22.0 + 3.0 / 23.0 + 4.0 / 24.0 + 5.0 / 25.0) / 5.0;
        assert_eq!(average_precision(&scores, &labels).unwrap(), want);

        // `-0.0` sorts below `0.0` under `total_cmp`, which is the one place it differs from
        // `partial_cmp` on finite input, and the doc says so.
        assert_eq!(average_precision(&[0.0, -0.0], &[false, true]).unwrap(), 0.5);
        assert_eq!(average_precision(&[-0.0, 0.0], &[false, true]).unwrap(), 1.0);
    }

    /// With no positives there is no precision to average, and the answer is a refusal rather than
    /// a zero that would read as "the model failed" on a class the test set simply does not carry.
    #[test]
    fn a_ranking_with_no_positives_has_no_average_precision() {
        assert_eq!(
            average_precision(&[0.9, 0.1], &[false, false]).unwrap_err(),
            MetricError::NoPositives { class: 0 }
        );
    }

    /// Classes with no positives are excluded from the mean rather than scored zero. Here class 0
    /// is perfect (AP 1.0), class 1 has no positives, so the mean is 1.0 over one class — not 0.5
    /// over two.
    #[test]
    fn mean_average_precision_excludes_classes_with_no_positives() {
        // Two samples, two classes, row-major.
        let scores = [0.9, 0.4, 0.1, 0.3];
        let labels = [true, false, false, false];
        let m = mean_average_precision(&scores, &labels, 2).unwrap();
        assert_eq!(m, 1.0);
        // Nothing positive anywhere is a refusal, not a zero.
        assert!(mean_average_precision(&scores, &[false; 4], 2).is_err());
    }

    // ---- (f) the ledger bridge ----------------------------------------------------------------------

    /// **The cross-check.** `Ledger::syn_ops` is counted by the simulator as deliveries happen;
    /// `account_for_train` recounts them afterwards from the train and the network's delay
    /// structure. Two entirely separate paths to one integer, and they must agree exactly.
    ///
    /// It does **not** pin the `t + 1 + d < ticks` arrival rule, although an earlier version of
    /// this comment claimed it did. Measured on this fixture: 138 posted, 137 delivered, 1 in
    /// flight, and that one crosses the delay-7 recurrent edge from a spike at tick 2996, so it
    /// arrives at 3004 against a 3000-tick run — four ticks clear of the boundary, where a `<`
    /// written as `<=` or a dropped `+ 1` moves nothing. Both of those mutants survive this test
    /// and die in `a_delivery_scheduled_past_the_end_of_the_run_is_in_flight_not_delivered`, which
    /// is where the rule is actually pinned.
    #[test]
    fn the_ledger_and_the_metric_agree_on_a_real_run() {
        let n = 6usize;
        let ticks = 3_000u64;
        let mut b = NetBuilder::new(n);
        for i in 0..n - 1 {
            // 20 mV per spike clears the default cell's 15 mV rest-to-threshold gap in one
            // arrival, so the chain actually propagates. A weight of 8 mV does not: two arrivals
            // an inter-spike interval apart sum to 11.6 mV after the first has decayed, the chain
            // stays silent past neuron 0, and the test would be counting one neuron's deliveries.
            b.connect(i as u32, (i + 1) as u32, 20e-3, 3).unwrap();
        }
        // A recurrent edge back to the start, so the network keeps itself busy.
        b.connect((n - 1) as u32, 0, 4e-3, 7).unwrap();
        let net = b.build();

        let mut ext = vec![0.0; n];
        ext[0] = 3e-9;
        let mut sim = Sim::new(net.clone(), vec![Lif::default(); n], 1e-4, Mode::Clocked).unwrap();
        let train = sim.run(ticks, &ext);
        assert!(train.len() > 20, "only {} spikes; the fixture did not exercise anything", train.len());

        let acc = account_for_train(&net, &train, ticks).unwrap();
        assert_eq!(
            acc.delivered, sim.ledger.syn_ops,
            "the recount disagrees with the simulator's own counter"
        );
        // Every weight here is non-zero, so every delivery is an effective accumulate.
        assert_eq!(acc.ops.effective_acs, sim.ledger.syn_ops);
        assert_eq!(acc.ops.effective_macs, 0);
        assert_eq!(acc.zero_weight, 0);

        // `posted == delivered + in_flight` is an invariant of this loop's own shape — it does one
        // `posted += 1` and then exactly one of the other two — so it can only ever detect a
        // structural edit. The independent statement is that `posted` is the sum of the firing
        // neurons' out-degrees, computed here from the net rather than from the accounting.
        let by_degree: u64 =
            train.spikes().iter().map(|s| net.out_degree(s.source as usize) as u64).sum();
        assert_eq!(acc.posted, by_degree, "posted is not the sum of the out-degrees that fired");
        assert_eq!(acc.posted, acc.delivered + acc.in_flight);

        // 6 neurons: 5 chain synapses plus 1 recurrent edge, over 3,000 ticks, is 18,000 dense
        // synapse-ticks. Stated as the literal rather than re-typing `n_syn * ticks`, which is the
        // expression the code under test evaluates.
        assert_eq!(net.n_syn, 6);
        assert_eq!(acc.ops.dense, 18_000);
    }

    /// The summary carries both stories from the same run, and the identity between them holds:
    /// with one inference per tick, effective-over-dense IS spikes-per-synapse-per-inference.
    #[test]
    fn the_summary_reports_the_benchmark_number_and_the_energy_verdict_together() {
        let n = 5usize;
        let ticks = 2_000u64;
        let mut b = NetBuilder::new(n);
        for i in 0..n - 1 {
            b.connect(i as u32, (i + 1) as u32, 20e-3, 2).unwrap();
        }
        let net = b.build();
        let mut ext = vec![0.0; n];
        ext[0] = 3e-9;
        let mut sim =
            Sim::new(net.clone(), vec![Lif::default(); n], 1e-4, Mode::EventDriven).unwrap();
        let train = sim.run(ticks, &ext);

        let fp = Footprint::new(net.n_syn as u64, 8, n as u64, 32).unwrap();
        let s = summarise(&net, &sim.ledger, &train, ticks, ticks, fp).unwrap();

        assert!(s.agrees_with_ledger, "{} vs {}", s.accounting.delivered, s.ledger_syn_ops);
        assert_eq!(s.neurons, 5);
        assert_eq!(s.synapses, 4);
        // Counted through a different accessor than the one `summarise` uses, so this compares two
        // paths rather than re-typing `train.len()`.
        let per_neuron: u64 = (0..5u32).map(|i| train.of(i).len() as u64).sum();
        assert_eq!(s.spikes, per_neuron);
        assert!(per_neuron > 0, "the fixture produced no spikes at all");
        // Activation sparsity, recomputed by hand from the counts.
        let want = (5.0 * 2000.0 - train.len() as f64) / (5.0 * 2000.0);
        assert!((s.activation_sparsity - want).abs() < 1e-15);
        assert!(s.activation_sparsity > 0.9, "a chain of LIF cells should be quiet");

        // THE IDENTITY. Deliveries over synapse-ticks, computed two ways. They differ only by the
        // deliveries still in flight at the end of the run, which the ledger never saw either — so
        // with the ledger's own count on both sides they are equal to the last bit.
        let density = s.effective_density().unwrap();
        let sps = s.spikes_per_synapse.unwrap();
        assert_eq!(density, sps, "effective/dense and spikes/synapse/inference disagree");

        // Sparse enough to be plausible under every published threshold — and the summary says so
        // for all three rather than picking one.
        assert_eq!(s.verdicts.len(), 3);
        assert!(s.verdicts.iter().all(|(_, v)| *v == Verdict::Plausible), "{:?}", s.verdicts);

        // 4 synapses at 8 bits is 4 bytes; 5 neurons at 32 bits is 20 bytes; 24 in total.
        assert_eq!(s.footprint_bytes.unwrap(), 24);

        // And the honest part: nothing here is a joule.
        let text = format!("{s}");
        assert!(text.contains("NOT MEASURED"), "{text}");
        // The cross-check is on the printed surface, not only in the struct.
        assert!(text.contains("ledger cross-check"), "{text}");
        assert!(!text.contains("DISAGREES"), "{text}");
    }

    /// **The cross-check has to be able to read false, and the report has to say so.** `summarise`
    /// adjudicates three published crossover thresholds from `Ledger::syn_ops`; if that ledger is
    /// not the one this train came from, all three verdicts are about a different run. Hardcoding
    /// `agrees_with_ledger: true` passed every test in this module, and no line of the printed
    /// report mentioned the check at all.
    #[test]
    fn the_summary_says_when_its_two_counts_disagree() {
        let mut b = NetBuilder::new(2);
        b.connect(0, 1, 20e-3, 1).unwrap();
        let net = b.build();
        let train = Train::from_spikes(vec![Spike { t: 0, source: 0 }, Spike { t: 4, source: 0 }]);
        let fp = Footprint::new(1, 8, 2, 32).unwrap();

        // The honest ledger: both spikes cross the one synapse and arrive inside a 10-tick run.
        let honest = Ledger { syn_ops: 2, ..Ledger::default() };
        let good = summarise(&net, &honest, &train, 10, 10, fp).unwrap();
        assert_eq!(good.accounting.delivered, 2);
        assert!(good.agrees_with_ledger);
        assert!(format!("{good}").contains("2 delivered == 2 billed"), "{good}");

        // A ledger from some other run, a hundred times the recount. `summarise` still returns
        // `Ok` — it is a report, not a gate — but the flag reads false and the report says so
        // ABOVE the verdicts, which are computed from that ledger and not from this train.
        let foreign = Ledger { syn_ops: 200, ..Ledger::default() };
        let bad = summarise(&net, &foreign, &train, 10, 10, fp).unwrap();
        assert_eq!(bad.accounting.delivered, 2);
        assert!(!bad.agrees_with_ledger, "the cross-check cannot read false");
        let text = format!("{bad}");
        assert!(text.contains("DISAGREES"), "{text}");
        assert!(text.contains("2 delivered vs 200 billed"), "{text}");
        let disagreement = text.find("DISAGREES").unwrap();
        let first_verdict = text.find("    vs ").unwrap();
        assert!(disagreement < first_verdict, "the warning must precede the verdicts:\n{text}");
        // And the verdicts really are adjudicated, which is why the warning has to be there.
        assert_eq!(bad.verdicts.len(), 3);
    }

    /// **The in-flight path, through `summarise`.** A delivery posted late enough that its arrival
    /// tick is past the end of the run is not delivered, is not an effective operation, and is not
    /// billed by the ledger either. No summary fixture in this module exercised it — both measure
    /// `in_flight == 0` — so the whole systematic under-count that
    /// `SpikeAccounting::in_flight` documents was pinned by a single delivery in a single test.
    #[test]
    fn the_summary_accounts_for_deliveries_still_in_flight() {
        let mut b = NetBuilder::new(2);
        b.connect(0, 1, 20e-3, 5).unwrap();
        let net = b.build();
        // Tick 0 arrives at 6, inside a 10-tick run. Tick 9 arrives at 15, which is not.
        let train = Train::from_spikes(vec![Spike { t: 0, source: 0 }, Spike { t: 9, source: 0 }]);
        let ledger = Ledger { syn_ops: 1, ..Ledger::default() };
        let fp = Footprint::new(1, 8, 2, 32).unwrap();
        let s = summarise(&net, &ledger, &train, 10, 10, fp).unwrap();

        assert_eq!(s.accounting.posted, 2);
        assert_eq!(s.accounting.delivered, 1);
        assert_eq!(s.accounting.in_flight, 1, "the fixture must leave a delivery in flight");
        assert_eq!(s.ops.effective_acs, 1, "an in-flight delivery is not an effective operation");
        // The simulator never saw it either, so the two counts still agree — which is the point:
        // the under-count is shared, not a disagreement, and that is why it needs saying out loud.
        assert!(s.agrees_with_ledger);
        assert!(format!("{s}").contains("in flight at the end 1"), "{s}");

        // Run the same train two ticks longer and the delivery lands: 16 ticks puts the arrival at
        // 15 inside the run, and the in-flight count falls to zero.
        let longer = summarise(&net, &Ledger { syn_ops: 2, ..Ledger::default() }, &train, 16, 16, fp)
            .unwrap();
        assert_eq!(longer.accounting.delivered, 2);
        assert_eq!(longer.accounting.in_flight, 0);
    }

    /// A zero-weight synapse is charged by the ledger — the hardware fetched it and added it — and
    /// is NOT an effective operation under `NeuroBench`. That is the entire difference between the
    /// two counts once flight time is accounted for, and this is where it is exhibited.
    #[test]
    fn a_zero_weight_synapse_is_billed_but_is_not_an_effective_operation() {
        let mut b = NetBuilder::new(3);
        b.connect(0, 1, 20e-3, 1).unwrap();
        b.connect(0, 2, 0.0, 1).unwrap(); // a pruned synapse that still costs a fetch
        let net = b.build();
        let mut ext = vec![0.0; 3];
        ext[0] = 3e-9;
        let mut sim = Sim::new(net.clone(), vec![Lif::default(); 3], 1e-4, Mode::Clocked).unwrap();
        let train = sim.run(1_000, &ext);

        let acc = account_for_train(&net, &train, 1_000).unwrap();
        assert_eq!(acc.delivered, sim.ledger.syn_ops);
        assert!(acc.zero_weight > 0, "the fixture never delivered across the pruned synapse");
        assert_eq!(acc.ops.effective_acs + acc.zero_weight, acc.delivered);
        assert!(
            acc.ops.effective_acs < sim.ledger.syn_ops,
            "the benchmark count must be BELOW the billed count here"
        );
    }

    /// A spike from a neuron the network does not have is refused rather than silently contributing
    /// nothing, which is what `Net::out_of` would do on its own.
    #[test]
    fn a_train_naming_an_absent_neuron_is_refused() {
        let net = NetBuilder::new(2).build();
        let t = Train::from_spikes(vec![Spike { t: 0, source: 5 }]);
        assert_eq!(
            account_for_train(&net, &t, 10).unwrap_err(),
            MetricError::SourceOutOfRange { source: 5, n: 2 }
        );
    }

    /// Deliveries scheduled past the end of the run are counted as in flight rather than as
    /// delivered. A one-synapse network with a delay of 5 and a spike on the last tick posts one
    /// delivery and delivers none.
    #[test]
    fn a_delivery_scheduled_past_the_end_of_the_run_is_in_flight_not_delivered() {
        let mut b = NetBuilder::new(2);
        b.connect(0, 1, 1e-3, 5).unwrap();
        let net = b.build();
        let t = Train::from_spikes(vec![Spike { t: 9, source: 0 }]);
        let acc = account_for_train(&net, &t, 10).unwrap();
        assert_eq!(acc.posted, 1);
        assert_eq!(acc.delivered, 0);
        assert_eq!(acc.in_flight, 1);

        // The same spike in a longer run arrives on tick 15, so a 16-tick run delivers it and a
        // 15-tick one does not. That boundary is the `t + 1 + d < ticks` rule.
        let short = account_for_train(&net, &t, 15).unwrap();
        assert_eq!(short.delivered, 0);
        let long = account_for_train(&net, &t, 16).unwrap();
        assert_eq!(long.delivered, 1);
    }

    /// `SynOps::add` must overflow loudly rather than wrap, on EACH of its three counts. A wrapped
    /// operation count is a small number where an enormous one belongs, which is the direction
    /// that flatters a result — and `dense` is only the denominator, while the two effective
    /// counts are what a submission reports.
    #[test]
    fn an_overflowing_count_is_an_error_rather_than_a_wrap() {
        let mut a = SynOps { dense: u64::MAX, ..SynOps::default() };
        let e = a.add(SynOps { dense: 1, ..SynOps::default() }).unwrap_err();
        assert_eq!(e, MetricError::Overflow { what: "dense ops" });
        assert_eq!(a.dense, u64::MAX, "a refused add must not have wrapped the field");

        let mut m = SynOps { effective_macs: u64::MAX, ..SynOps::default() };
        assert_eq!(
            m.add(SynOps { effective_macs: 1, ..SynOps::default() }).unwrap_err(),
            MetricError::Overflow { what: "effective MACs" }
        );
        assert_eq!(m.effective_macs, u64::MAX);

        let mut c = SynOps { effective_acs: u64::MAX, ..SynOps::default() };
        assert_eq!(
            c.add(SynOps { effective_acs: 1, ..SynOps::default() }).unwrap_err(),
            MetricError::Overflow { what: "effective ACs" }
        );
        assert_eq!(c.effective_acs, u64::MAX);
    }

    /// `effective_total` adds two public `u64` fields, and `add` checks them SEPARATELY, so a pair
    /// that each fit can have a sum that does not. A bare `+` panicked here in a debug build and
    /// wrapped in a release one, handing `reduction()` and `ac_fraction()` a tiny denominator.
    #[test]
    fn an_effective_total_that_does_not_fit_is_none_rather_than_a_panic_or_a_wrap() {
        let over = SynOps { dense: 10, effective_macs: u64::MAX, effective_acs: 1 };
        assert_eq!(over.effective_total(), None);
        assert_eq!(over.reduction(), None);
        assert_eq!(over.ac_fraction(), None);

        // Both counts can be assembled this way through the CHECKED path: each `add` succeeds.
        let mut built = SynOps::default();
        built.add(SynOps { effective_macs: u64::MAX, ..SynOps::default() }).unwrap();
        built.add(SynOps { effective_acs: 1, ..SynOps::default() }).unwrap();
        assert_eq!(built.effective_total(), None);

        // One less and it fits exactly, which is where the boundary is.
        let fits = SynOps { dense: 10, effective_macs: u64::MAX - 1, effective_acs: 1 };
        assert_eq!(fits.effective_total(), Some(u64::MAX));
    }

    /// `account_for_train` computes an arrival tick as `t + 1 + d`, and `Spike::t` is a public
    /// `u64`. At the top of the range that sum left it: this exact train panicked in a debug build
    /// and wrapped in a release one, where the wrap made a delivery that can never arrive look
    /// like one that arrived on tick 0.
    #[test]
    fn a_spike_at_the_end_of_time_is_in_flight_rather_than_a_panic() {
        let mut b = NetBuilder::new(2);
        b.connect(0, 1, 1e-3, 5).unwrap();
        let net = b.build();

        let t = Train::from_spikes(vec![Spike { t: u64::MAX, source: 0 }]);
        let acc = account_for_train(&net, &t, 10).unwrap();
        assert_eq!(acc.posted, 1);
        assert_eq!(acc.delivered, 0, "a wrapped arrival tick would have landed inside the run");
        assert_eq!(acc.in_flight, 1);
        assert_eq!(acc.ops.effective_acs, 0);

        // Every offset that overflows behaves the same way, including the one that wraps to
        // exactly zero: `t = u64::MAX`, `d = 0` gives `t + 1 == 0` in wrapping arithmetic.
        let mut b0 = NetBuilder::new(2);
        b0.connect(0, 1, 1e-3, 0).unwrap();
        let net0 = b0.build();
        let acc0 = account_for_train(&net0, &t, 10).unwrap();
        assert_eq!(acc0.delivered, 0);
        assert_eq!(acc0.in_flight, 1);
    }
}
