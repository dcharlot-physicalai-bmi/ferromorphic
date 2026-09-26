//! Making a trained spiking network fit the part you can actually buy.
//!
//! # What this module is
//!
//! A network that trains well on a workstation rarely fits the chip. [`crate::hardware`] says why:
//! a core holds a fixed number of synapses, a weight is 8 bits wide or 4 or 1, and a fan-in past
//! the part's limit is not a slow network but an unmappable one. So the trained thing has to be
//! made smaller, and there are exactly four levers anybody has found:
//!
//! 1. **Pruning** — set weights to zero and stop storing them. Han, Pool, Tran and Dally,
//!    *Learning both Weights and Connections for Efficient Neural Networks*, NIPS 2015
//!    (arXiv:1506.02626v3), is the paper that made magnitude pruning standard, and retraining is
//!    its own required third step: "The final step retrains the network to learn the final
//!    weights for the remaining sparse connections. This step is critical. If the pruned
//!    network is used without retraining, accuracy is significantly impacted." (Sec. 3). Frankle
//!    and Carbin, *The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks*,
//!    ICLR 2019 (arXiv:1803.03635v5, the camera-ready), added a different step: reset every
//!    surviving weight to its ORIGINAL initialisation and train the subnetwork in isolation. The
//!    subnetworks that step finds, the *winning tickets*, "reach test accuracy comparable to the
//!    original network in a similar number of iterations" (abstract). That paper is why people
//!    rewind to the initial weights, not why they retrain. ⛔ This item used to say Frankle and
//!    Carbin are "why people now prune and retrain rather than prune and ship". Their own
//!    footnote 1 quotes Han et al. 2015 on retraining, and Sec. 1 names the reset as the step
//!    that is theirs: "Unique to our work, each unpruned connection's value is then reset to its
//!    initialization from original network before it was trained."
//! 2. **Quantisation** — store each surviving weight in fewer bits. [`crate::hardware::Quantiser`]
//!    already does the arithmetic and reports what it cost; this module reuses it rather than
//!    reimplementing the round trip.
//! 3. **Fewer timesteps** — run the network for `T` ticks instead of `4T`. For a rate-coded
//!    spiking network this is the **largest lever there is**, because every count in
//!    [`crate::ledger::Ledger`] is proportional to `T` while the accuracy cost is only
//!    `O(1/sqrt(T))`. That asymmetry is the entire argument, and it is checked here against the
//!    closed form rather than asserted.
//! 4. **Distillation** — train the small network to imitate a large one's soft outputs instead of
//!    the hard labels. Hinton, Vinyals and Dean, *Distilling the Knowledge in a Neural Network*,
//!    Deep Learning and Representation Learning Workshop, NIPS 2014.
//!
//! # The lesson: why `1/sqrt(T)` decides everything
//!
//! A rate code says "this unit's activation is `p`" by firing with probability `p` on each of `T`
//! ticks. The decoder counts spikes and divides. That count is `Binomial(T, p)`, so the estimate
//! `p̂ = k/T` has **standard deviation exactly `sqrt(p(1-p)/T)`** — a closed form, not an
//! approximation, for every `T` and every `p`. [`rate_rms`] is that expression and
//! [`rate_error_vs_ticks`] measures it.
//!
//! Now put it beside the cost. A tick costs one pass over every active synapse, so work is
//! **linear** in `T` and error is `T^(-1/2)`. Halving `T` halves the joules and multiplies the
//! error by `sqrt(2) ≈ 1.414`. Going the other way is brutal: **to halve the error you need four
//! times the ticks and four times the energy.** A network run for 250 ticks that could have run
//! for 16 is paying 15.6x the bill for a 4x tighter rate estimate that the classifier downstream
//! very likely cannot use.
//!
//! That is why the first question to ask of any published spiking-network efficiency figure is
//! *how many timesteps*, and why [`ticks_for_rms`] exists: it inverts the closed form and returns
//! the **smallest** tick budget that reaches a stated error, so the number is chosen rather than
//! inherited from a tutorial.
//!
//! # The honesty this module is built around
//!
//! **Pruning a weight to zero does not, by itself, save a single joule.** [`crate::sim`] queues a
//! delivery for every synapse a spike traverses and charges
//! [`crate::ledger::Ledger::syn_ops`] for it, whatever the weight. Zeroing changes the *footprint*
//! under a sparse storage format and changes nothing else until the synapse is removed from the
//! graph — which is [`compact_net`]. Both halves are tested, including the equality that shows
//! zeroing alone costs exactly the same as not pruning at all.
//!
//! **A sparse format can be bigger than the dense one it replaces.** Storing a surviving weight
//! sparsely costs its value *plus an index*. At 4-bit weights and 16-bit indices, 75% sparsity
//! still occupies 25% more memory than storing everything densely. [`Storage`] makes the three
//! layouts computable so the comparison is arithmetic instead of folklore, and the case above is
//! a test.
//!
//! **This module does not predict accuracy.** [`Point`] computes bytes and synaptic operations
//! exactly and takes the error from a measurement you supply. The one error term it will compute
//! is the rate-coding sampling error, because that one has a closed form; it is not the whole
//! error, and [`rate_rms`]'s doc says so.
//!
//! # Units
//!
//! Weights are in the volts-per-spike of [`crate::net::Net::w`]; footprints are in bits and bytes;
//! tick budgets are dimensionless counts, converted to seconds by multiplying by the simulator's
//! `dt`. The distillation temperature is dimensionless and is kept verbatim as Hinton's papers
//! print it, because its only meaning is as a divisor of logits.
//!
//! # What this review did not locate
//!
//! A published, measured accuracy-versus-sparsity curve for a spiking network on neuromorphic
//! silicon with the energy measured on the same run. The ANN literature has many; the spiking
//! literature reports sparsity and accuracy from simulation and energy from a model. That is the
//! gap [`Point`] is shaped for — it will hold a measured error the moment somebody measures one.

use core::fmt;

use crate::convert::ErrorCurve;
use crate::hardware::{HardwareError, Part, Quantised, Quantiser, Rounding};
use crate::ledger::Ledger;
use crate::metrics::{Footprint, MetricError};
use crate::net::{Net, NetBuilder, NetError};
use crate::rng::Rng;

// -------------------------------------------------------------------------------------------
// Errors
// -------------------------------------------------------------------------------------------

/// Why a compression step could not be carried out.
///
/// Every variant names the quantity that was wrong and the value it held, because a compression
/// pipeline is a chain of transforms and "invalid argument" three stages in is not a diagnosis.
#[derive(Debug, Clone, PartialEq)]
pub enum CompressError {
    /// A quantity was asked of nothing at all.
    Empty {
        /// Which array or set was empty, e.g. `"weights"`.
        what: &'static str,
    },
    /// An array element was `NaN` or infinite, rejected at the boundary.
    ///
    /// A non-finite weight poisons a quantiser's scale, then every code derived from it, then a
    /// round-trip error that reports as `NaN` long after the cause has gone out of scope.
    NonFinite {
        /// Which array, e.g. `"weights"` or `"teacher logits"`.
        what: &'static str,
        /// Position in that array.
        index: usize,
    },
    /// A scalar parameter was outside the range its meaning allows, or was not finite.
    BadValue {
        /// Which parameter, with its admissible range, e.g. `"sparsity, which must be in 0..=1"`.
        what: &'static str,
        /// The value supplied.
        value: f64,
    },
    /// Two arrays that must be the same length were not.
    LengthMismatch {
        /// Length of the first.
        a: usize,
        /// Length of the second.
        b: usize,
    },
    /// A weight matrix's length was not `n_out * n_in`, so its declared shape is not its shape.
    BadShape {
        /// Length supplied.
        got: usize,
        /// Length the declared shape implies.
        want: usize,
    },
    /// A class label named a class at or past the number of classes present.
    LabelOutOfRange {
        /// The label supplied.
        label: usize,
        /// How many classes the logit vector has.
        classes: usize,
    },
    /// A vector offered as a probability distribution had a negative entry or did not sum to one.
    ///
    /// The tolerance on the sum is `1e-9`, which is far above the `~1e-16 * n` a softmax
    /// accumulates and far below any real error.
    NotADistribution {
        /// Which distribution, e.g. `"teacher"`.
        what: &'static str,
        /// What its entries summed to, or the offending negative entry's value.
        sum: f64,
    },
    /// A divergence was asked for where the second distribution assigns zero probability to an
    /// outcome the first assigns positive probability, so the answer is `+inf`.
    ///
    /// Returned rather than an infinity, because an infinite loss silently poisons an optimiser
    /// and the cause — usually a temperature so low the softmax underflowed — is recoverable.
    ZeroSupport {
        /// Which distribution held the zero, e.g. `"student"`.
        what: &'static str,
        /// The outcome index it was zero at.
        index: usize,
    },
    /// [`Rounding::Stochastic`] was asked for without a generator to draw from.
    ///
    /// Refused rather than defaulted to a fresh seed: a stochastic quantisation whose seed the
    /// caller did not choose is not reproducible, and this crate's determinism promise is that a
    /// seed fixes the output on every platform.
    NeedsRng,
    /// A count exceeded `u64`, reported rather than wrapped.
    Overflow {
        /// Which product or sum overflowed, e.g. `"parameter bits"`.
        what: &'static str,
    },
    /// No candidate in the set fitted the budget.
    NoFeasiblePoint {
        /// How many candidates were considered.
        candidates: usize,
        /// The byte budget they had to fit in.
        max_bytes: u64,
        /// The synaptic-operation budget they had to fit in.
        max_syn_ops: u64,
    },
    /// A quantiser refused; see [`HardwareError`].
    Hardware(HardwareError),
    /// A metric refused; see [`MetricError`].
    Metric(MetricError),
    /// A rebuilt network refused; see [`NetError`].
    Net(NetError),
}

impl fmt::Display for CompressError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty { what } => write!(f, "no {what} to work with"),
            Self::NonFinite { what, index } => {
                write!(f, "{what}[{index}] is not a finite number")
            }
            Self::BadValue { what, value } => write!(f, "{what}: got {value}"),
            Self::LengthMismatch { a, b } => write!(f, "lengths {a} and {b} differ"),
            Self::BadShape { got, want } => {
                write!(f, "weight matrix has {got} entries; the declared shape needs {want}")
            }
            Self::LabelOutOfRange { label, classes } => {
                write!(f, "label {label} names a class past the {classes} present")
            }
            Self::NotADistribution { what, sum } => {
                write!(f, "{what} is not a probability distribution (sum or entry {sum})")
            }
            Self::ZeroSupport { what, index } => write!(
                f,
                "{what} assigns zero probability at {index} where the other assigns positive \
                 probability, so the divergence is infinite"
            ),
            Self::NeedsRng => f.write_str(
                "stochastic rounding needs a seeded generator; a seed this crate chose would not \
                 be reproducible by you",
            ),
            Self::Overflow { what } => write!(f, "{what} exceeded u64"),
            Self::NoFeasiblePoint { candidates, max_bytes, max_syn_ops } => write!(
                f,
                "none of the {candidates} candidates fits {max_bytes} bytes and {max_syn_ops} \
                 synaptic operations"
            ),
            Self::Hardware(e) => write!(f, "{e}"),
            Self::Metric(e) => write!(f, "{e}"),
            Self::Net(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for CompressError {}

impl From<HardwareError> for CompressError {
    fn from(e: HardwareError) -> Self {
        Self::Hardware(e)
    }
}

impl From<MetricError> for CompressError {
    fn from(e: MetricError) -> Self {
        Self::Metric(e)
    }
}

impl From<NetError> for CompressError {
    fn from(e: NetError) -> Self {
        Self::Net(e)
    }
}

/// Reject a non-finite element by index, naming the array.
fn all_finite(v: &[f64], what: &'static str) -> Result<(), CompressError> {
    if v.is_empty() {
        return Err(CompressError::Empty { what });
    }
    for (i, &x) in v.iter().enumerate() {
        if !x.is_finite() {
            return Err(CompressError::NonFinite { what, index: i });
        }
    }
    Ok(())
}

/// Reject a fraction outside `0..=1`, naming it.
fn unit_fraction(x: f64, what: &'static str) -> Result<(), CompressError> {
    if !x.is_finite() || !(0.0..=1.0).contains(&x) {
        return Err(CompressError::BadValue { what, value: x });
    }
    Ok(())
}

// -------------------------------------------------------------------------------------------
// Magnitude pruning
// -------------------------------------------------------------------------------------------

/// A pruned weight vector and **what was actually removed**, which is not always what was asked
/// for.
///
/// Pruning here means **setting a weight to exactly zero**, not deleting it. The distinction is
/// load-bearing: a [`Net`] is CSR and deleting an entry renumbers every synapse after it, while
/// [`crate::metrics::connection_sparsity_of`] counts a connection as absent exactly when every
/// synapse on it is exactly zero. Use [`compact_net`] to turn the zeros into a smaller graph once
/// you are finished moving them around.
#[derive(Debug, Clone, PartialEq)]
pub struct Pruned {
    /// The weights after pruning, same length and order as the input. Removed entries are `+0.0`.
    pub w: Vec<f64>,
    /// `true` where the weight survived selection, parallel to [`Pruned::w`].
    pub kept: Vec<bool>,
    /// How many entries were selected for removal.
    ///
    /// Equals the requested count when the request was an exact multiple of `1/n`. It counts
    /// *selections*, so an already-zero weight that was selected is counted here even though
    /// nothing changed — see [`Pruned::newly_zeroed`].
    pub n_removed: usize,
    /// How many of the removed entries were non-zero before, and so actually lost information.
    pub newly_zeroed: usize,
    /// Largest magnitude removed, or `None` when nothing was.
    pub cut: Option<f64>,
    /// Smallest magnitude kept, or `None` when everything was removed.
    pub smallest_kept: Option<f64>,
    /// How many entries of the whole vector share the cut magnitude, when the cut is ambiguous.
    ///
    /// Zero when the cut separates cleanly. A non-zero value says the boundary fell inside a group
    /// of equal magnitudes and was broken by array index — deterministic, and arbitrary. Reported
    /// because a reader comparing two pruning runs of the same network needs to know when the
    /// difference between them is a tie-break rather than a decision.
    pub tied_at_cut: usize,
}

impl Pruned {
    /// Fraction of the **output** that is exactly zero: the sparsity a storage format can exploit.
    ///
    /// This is the number to report. It differs from [`Pruned::removed_fraction`] whenever the
    /// input already held zeros, and in that direction: a vector that was already half zero and is
    /// pruned by a further 25% of its entries comes out 50% sparse if those 25% were the zeros it
    /// already had.
    #[must_use]
    pub fn sparsity(&self) -> f64 {
        let zeros = self.w.iter().filter(|x| **x == 0.0).count();
        zeros as f64 / self.w.len() as f64
    }

    /// Fraction of entries this call selected for removal.
    #[must_use]
    pub fn removed_fraction(&self) -> f64 {
        self.n_removed as f64 / self.w.len() as f64
    }

    /// How many entries survived.
    ///
    /// Every field of this record is public, so a caller who sets `n_removed` past `w.len()` gets
    /// a `usize` underflow here. The record's own constructors cannot produce that state, and the
    /// consistency test over both of them is what guarantees it.
    #[must_use]
    pub fn n_kept(&self) -> usize {
        self.w.len().saturating_sub(self.n_removed)
    }

    /// Whether every removed magnitude is strictly below every kept one.
    ///
    /// `false` means the cut fell inside a tie and [`Pruned::tied_at_cut`] says how wide it was.
    /// `true` for a vector of distinct magnitudes, and for the two degenerate cases where nothing
    /// or everything was removed.
    #[must_use]
    pub fn separates_cleanly(&self) -> bool {
        match (self.cut, self.smallest_kept) {
            (Some(c), Some(s)) => c < s,
            _ => true,
        }
    }
}

/// Zero the `sparsity` fraction of `w` with the smallest magnitudes.
///
/// The count removed is `round(sparsity * w.len())`, so a request that is not a multiple of
/// `1/n` is rounded to the nearest achievable one and [`Pruned::removed_fraction`] reports what
/// happened. Ties in magnitude are broken by ascending array index, which is deterministic and
/// arbitrary; [`Pruned::tied_at_cut`] says when that mattered.
///
/// # Global versus per-layer
///
/// This function prunes whatever vector it is handed, so pruning layer by layer and pruning
/// globally are the same call on different inputs — concatenate the layers for global. ⚠ Doing so
/// prunes against a single magnitude scale, and a layer whose weights are ten times smaller than
/// its neighbours' will be removed **entirely** before the neighbours lose anything. That is a
/// real result in the literature and it is usually not what the caller wanted.
///
/// # Errors
///
/// [`CompressError::Empty`] on an empty slice, [`CompressError::NonFinite`] naming the first
/// non-finite weight, [`CompressError::BadValue`] for a `sparsity` outside `0..=1`.
pub fn prune_magnitude(w: &[f64], sparsity: f64) -> Result<Pruned, CompressError> {
    all_finite(w, "weights")?;
    unit_fraction(sparsity, "sparsity, which must be in 0..=1")?;
    let n = w.len();
    let requested = (sparsity * n as f64).round();
    let n_removed = if requested >= n as f64 { n } else { requested as usize };

    // Ascending by magnitude, ties by index. `total_cmp` is a total order on finite f64 and
    // `abs()` maps -0.0 to +0.0, so the comparison never sees a sign.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_unstable_by(|&a, &b| w[a].abs().total_cmp(&w[b].abs()).then(a.cmp(&b)));

    let mut out = w.to_vec();
    let mut kept = vec![true; n];
    let mut newly_zeroed = 0usize;
    for &idx in &order[..n_removed] {
        if out[idx] != 0.0 {
            newly_zeroed += 1;
        }
        out[idx] = 0.0;
        kept[idx] = false;
    }

    let cut = if n_removed == 0 { None } else { Some(w[order[n_removed - 1]].abs()) };
    let smallest_kept = if n_removed == n { None } else { Some(w[order[n_removed]].abs()) };
    let tied_at_cut = match (cut, smallest_kept) {
        (Some(c), Some(s)) if c == s => w.iter().filter(|x| x.abs() == c).count(),
        _ => 0,
    };

    Ok(Pruned { w: out, kept, n_removed, newly_zeroed, cut, smallest_kept, tied_at_cut })
}

/// Zero every weight whose magnitude is **strictly below** `threshold`.
///
/// The form Han et al. (2015) describe: a threshold chosen from the weight distribution rather
/// than a target sparsity. The sparsity it achieves is whatever the distribution gives, which is
/// why [`Pruned::sparsity`] is the reported quantity and the threshold is not.
///
/// # Errors
///
/// [`CompressError::Empty`], [`CompressError::NonFinite`], or [`CompressError::BadValue`] for a
/// negative or non-finite `threshold`.
pub fn prune_below(w: &[f64], threshold: f64) -> Result<Pruned, CompressError> {
    all_finite(w, "weights")?;
    if !threshold.is_finite() || threshold < 0.0 {
        return Err(CompressError::BadValue {
            what: "threshold, which must be finite and non-negative",
            value: threshold,
        });
    }
    let n = w.len();
    let mut out = w.to_vec();
    let mut kept = vec![true; n];
    let mut n_removed = 0usize;
    let mut newly_zeroed = 0usize;
    let mut cut: Option<f64> = None;
    let mut smallest_kept: Option<f64> = None;
    for i in 0..n {
        let m = w[i].abs();
        if m < threshold {
            if out[i] != 0.0 {
                newly_zeroed += 1;
            }
            out[i] = 0.0;
            kept[i] = false;
            n_removed += 1;
            cut = Some(cut.map_or(m, |c: f64| c.max(m)));
        } else {
            smallest_kept = Some(smallest_kept.map_or(m, |s: f64| s.min(m)));
        }
    }
    // A strict `<` cut cannot tie: every removed magnitude is below `threshold` and every kept one
    // is at or above it, so the two sets are separated by construction.
    Ok(Pruned { w: out, kept, n_removed, newly_zeroed, cut, smallest_kept, tied_at_cut: 0 })
}

// -------------------------------------------------------------------------------------------
// Structured pruning
// -------------------------------------------------------------------------------------------

/// Which whole thing structured pruning removes.
///
/// The weight matrix is `n_out * n_in`, **row-major in the output index**: the weight from input
/// `i` to output `o` is at `o * n_in + i`. This is the layout [`crate::convert::DenseRelu`] uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unit {
    /// A whole output neuron — one row. The neuron stops existing: its state, its threshold and
    /// its refractory counter all go, which is why this is the form that actually shrinks a core's
    /// neuron allocation on [`crate::hardware::LOIHI`] or [`crate::hardware::AKD1000`].
    OutputNeuron,
    /// A whole input channel — one column. Every output stops listening to that input, so the
    /// upstream neuron's fan-out drops and, if nothing else listens, the upstream neuron can go
    /// too.
    InputChannel,
}

/// How a unit's importance is scored before it is removed.
///
/// All three are the obvious norms of the unit's weight vector. Li, Kadav, Durdanovic, Samet and
/// Graf, *Pruning Filters for Efficient `ConvNets`*, ICLR 2017, use [`Saliency::L1`]; the L2 form is
/// at least as common. ⚠ This module makes no claim that any of them is the right criterion. They
/// are cheap, they are what the filter-pruning literature uses, and a norm is not an importance:
/// a low-norm unit feeding a high-gain downstream stage can matter more than a high-norm one that
/// nothing reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Saliency {
    /// Sum of absolute values.
    L1,
    /// Euclidean norm — square root of the sum of squares.
    L2,
    /// Largest absolute value.
    Max,
}

impl Saliency {
    /// Score one unit's weights.
    ///
    /// Zero for an empty slice, which is the only defensible value: a unit with no weights
    /// contributes nothing and should be the first removed.
    #[must_use]
    pub fn of(self, v: &[f64]) -> f64 {
        match self {
            Self::L1 => v.iter().map(|x| x.abs()).sum(),
            Self::L2 => v.iter().map(|x| x * x).sum::<f64>().sqrt(),
            Self::Max => v.iter().fold(0.0f64, |m, x| m.max(x.abs())),
        }
    }
}

/// A structurally pruned weight matrix: whole units zeroed, the rest untouched **bit for bit**.
#[derive(Debug, Clone, PartialEq)]
pub struct Structured {
    /// The weights after pruning, in the same `n_out * n_in` row-major layout as the input.
    pub w: Vec<f64>,
    /// Output count of the matrix.
    pub n_out: usize,
    /// Input count of the matrix.
    pub n_in: usize,
    /// Which whole thing was removed.
    pub unit: Unit,
    /// Which score decided.
    pub saliency: Saliency,
    /// Indices of the removed units, ascending.
    pub removed: Vec<usize>,
    /// The saliency of every unit, in unit order, before anything was removed.
    pub saliencies: Vec<f64>,
}

impl Structured {
    /// How many units the matrix has along the pruned axis.
    #[must_use]
    pub fn units(&self) -> usize {
        match self.unit {
            Unit::OutputNeuron => self.n_out,
            Unit::InputChannel => self.n_in,
        }
    }

    /// Fraction of units removed. The figure that decides whether a core's allocation shrinks.
    #[must_use]
    pub fn unit_sparsity(&self) -> f64 {
        self.removed.len() as f64 / self.units() as f64
    }

    /// Fraction of individual weights that are exactly zero after pruning.
    ///
    /// Equals [`Structured::unit_sparsity`] only when the surviving units held no zeros of their
    /// own. Reported separately because the two answer different questions: unit sparsity says
    /// what the chip's allocator sees, weight sparsity says what the storage format sees.
    #[must_use]
    pub fn weight_sparsity(&self) -> f64 {
        let zeros = self.w.iter().filter(|x| **x == 0.0).count();
        zeros as f64 / self.w.len() as f64
    }

    /// Whether unit `u` was removed.
    #[must_use]
    pub fn is_removed(&self, u: usize) -> bool {
        self.removed.binary_search(&u).is_ok()
    }

    /// Row `o` of the pruned matrix — the weights of output neuron `o`.
    ///
    /// `None` past the end rather than a panic, for the reason [`Net::out_of`] gives.
    #[must_use]
    pub fn row(&self, o: usize) -> Option<&[f64]> {
        if o >= self.n_out {
            return None;
        }
        Some(&self.w[o * self.n_in..(o + 1) * self.n_in])
    }
}

/// Remove the `fraction` of whole units with the lowest saliency.
///
/// The count removed is `round(fraction * units)`, and ties are broken by ascending unit index.
///
/// # Why this is worth more than the same sparsity unstructured
///
/// Unstructured pruning leaves a matrix with holes in it, and a hole is only a saving if the
/// runtime can skip it — which costs an index per surviving weight ([`Storage::Sparse`]) or a
/// mask bit per position ([`Storage::Bitmask`]). Removing a whole unit removes a whole row, a
/// whole neuron's state, a whole entry in a routing table, and needs no index at all: the matrix
/// is simply smaller. The price is that it is a much coarser instrument, and at equal weight
/// sparsity it costs more accuracy in every study this review located.
///
/// # Errors
///
/// [`CompressError::Empty`] for a zero dimension, [`CompressError::BadShape`] if `w.len()` is not
/// `n_out * n_in`, [`CompressError::NonFinite`], [`CompressError::BadValue`] for a `fraction`
/// outside `0..=1`.
pub fn prune_structured(
    w: &[f64],
    n_out: usize,
    n_in: usize,
    unit: Unit,
    saliency: Saliency,
    fraction: f64,
) -> Result<Structured, CompressError> {
    if n_out == 0 || n_in == 0 {
        return Err(CompressError::Empty { what: "weight matrix" });
    }
    let want = n_out.checked_mul(n_in).ok_or(CompressError::Overflow { what: "matrix size" })?;
    if w.len() != want {
        return Err(CompressError::BadShape { got: w.len(), want });
    }
    all_finite(w, "weights")?;
    unit_fraction(fraction, "fraction, which must be in 0..=1")?;

    let units = match unit {
        Unit::OutputNeuron => n_out,
        Unit::InputChannel => n_in,
    };
    // One scratch buffer, refilled per unit: a column is not contiguous and has to be gathered.
    let mut buf: Vec<f64> = Vec::with_capacity(n_in.max(n_out));
    let mut saliencies = Vec::with_capacity(units);
    for u in 0..units {
        buf.clear();
        match unit {
            Unit::OutputNeuron => buf.extend_from_slice(&w[u * n_in..(u + 1) * n_in]),
            Unit::InputChannel => buf.extend((0..n_out).map(|o| w[o * n_in + u])),
        }
        saliencies.push(saliency.of(&buf));
    }

    let requested = (fraction * units as f64).round();
    let n_removed = if requested >= units as f64 { units } else { requested as usize };
    let mut order: Vec<usize> = (0..units).collect();
    order.sort_unstable_by(|&a, &b| saliencies[a].total_cmp(&saliencies[b]).then(a.cmp(&b)));

    let mut out = w.to_vec();
    let mut removed: Vec<usize> = order[..n_removed].to_vec();
    removed.sort_unstable();
    for &u in &removed {
        match unit {
            Unit::OutputNeuron => {
                for x in &mut out[u * n_in..(u + 1) * n_in] {
                    *x = 0.0;
                }
            }
            Unit::InputChannel => {
                for o in 0..n_out {
                    out[o * n_in + u] = 0.0;
                }
            }
        }
    }

    Ok(Structured { w: out, n_out, n_in, unit, saliency, removed, saliencies })
}

// -------------------------------------------------------------------------------------------
// Pruning a Net, and making the zeros pay
// -------------------------------------------------------------------------------------------

/// Prune a [`Net`]'s synaptic weights by magnitude, leaving the graph's shape untouched.
///
/// Returns the pruned network and the [`Pruned`] record of what happened to the weight array. The
/// CSR structure, the delays and [`Net::n_syn`] are unchanged, so the result is a drop-in for the
/// original everywhere — including in [`crate::sim`], which will keep delivering the zeroed
/// synapses and keep charging for them. [`compact_net`] is the step that makes the saving real.
///
/// # Errors
///
/// As [`prune_magnitude`].
pub fn prune_net(net: &Net, sparsity: f64) -> Result<(Net, Pruned), CompressError> {
    let p = prune_magnitude(&net.w, sparsity)?;
    let mut out = net.clone();
    out.w.clone_from(&p.w);
    Ok((out, p))
}

/// Rebuild a [`Net`] with every exactly-zero synapse removed.
///
/// This is where pruning starts saving joules. After it, [`crate::sim`] has fewer deliveries to
/// queue, so [`crate::ledger::Ledger::syn_ops`] and [`crate::ledger::Ledger::syn_fetches`] fall in
/// proportion to the synapses that went.
///
/// [`Net::max_delay`] may shrink if the longest delay was on a pruned synapse, which shortens the
/// simulator's delivery ring. Neuron count is unchanged: a neuron with no surviving synapses is
/// still a neuron, still updated, still counted — removing it is [`prune_structured`]'s job and
/// needs to know what reads it.
///
/// # Errors
///
/// [`CompressError::Net`] if the network's own arrays are inconsistent — a postsynaptic index past
/// [`Net::n`], or a non-finite surviving weight. Both are reachable because [`Net`]'s fields are
/// public and a hand-assembled network need not have gone through [`NetBuilder`].
pub fn compact_net(net: &Net) -> Result<Net, CompressError> {
    let mut b = NetBuilder::new(net.n);
    for pre in 0..net.n {
        for (post, w, d) in net.out_of(pre) {
            if w != 0.0 {
                b.connect(
                    u32::try_from(pre).map_err(|_| CompressError::Overflow {
                        what: "presynaptic index in u32",
                    })?,
                    post,
                    w,
                    d,
                )?;
            }
        }
    }
    Ok(b.build())
}

// -------------------------------------------------------------------------------------------
// Quantisation
// -------------------------------------------------------------------------------------------

/// Quantise `w` to `bits`, scaled to the vector's own largest magnitude.
///
/// A thin, checked front end to [`Quantiser::from_weights`] — the round trip, the error
/// statistics and the half-least-significant-bit bound all live in [`crate::hardware`] and are not
/// reimplemented here. What this adds is the `rng` contract: [`Rounding::Stochastic`] without a
/// generator is [`CompressError::NeedsRng`] rather than a silently seeded stream.
///
/// # Order matters, and the safe order is prune first
///
/// Pruning does not change the largest magnitude — it removes the smallest — so the quantiser's
/// step is identical before and after, and a pruned zero maps to code 0 exactly. Quantising first
/// and pruning after works too but prunes against a magnitude ladder with only `2^(bits-1)` rungs,
/// where ties are the rule rather than the exception and [`Pruned::tied_at_cut`] will say so.
///
/// # Errors
///
/// [`CompressError::NeedsRng`], plus anything [`Quantiser::from_weights`] and
/// [`Quantiser::quantise_nearest`] reject, as [`CompressError::Hardware`].
pub fn quantise(
    w: &[f64],
    bits: u32,
    rounding: Rounding,
    rng: Option<&mut Rng>,
) -> Result<Quantised, CompressError> {
    let q = Quantiser::from_weights(bits, w)?;
    match (rounding, rng) {
        (Rounding::Nearest, _) => Ok(q.quantise_nearest(w)?),
        (Rounding::Stochastic, Some(r)) => Ok(q.quantise_stochastic(w, r)?),
        (Rounding::Stochastic, None) => Err(CompressError::NeedsRng),
    }
}

/// Quantise `w` to the weight width `part` states.
///
/// # Errors
///
/// [`CompressError::Hardware`] carrying [`HardwareError::UnstatedSpec`] for a part whose weight
/// width this crate's review did not locate — which is the correct outcome and not a defect — plus
/// everything [`quantise`] rejects.
pub fn quantise_for_part(
    part: &Part,
    w: &[f64],
    rounding: Rounding,
    rng: Option<&mut Rng>,
) -> Result<Quantised, CompressError> {
    let bits = Quantiser::for_part(part, w)?.bits;
    quantise(w, bits, rounding, rng)
}

// -------------------------------------------------------------------------------------------
// Footprint: how the survivors are actually stored
// -------------------------------------------------------------------------------------------

/// How a pruned weight matrix is laid out in memory, and therefore what pruning saved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Storage {
    /// Every parameter stored, zero or not. **Pruning saves nothing here.**
    ///
    /// Named rather than omitted because it is what most deployments actually do: a dense kernel
    /// with zeros in it runs at exactly the speed and exactly the footprint of a dense kernel
    /// without them, and a sparsity figure reported against this layout is a statement about the
    /// training run, not about the deployed model.
    Dense,
    /// Only the survivors, each stored with an explicit index — the CSR column index, or an
    /// equivalent.
    ///
    /// Costs `bits + index_bits` per surviving parameter and nothing for the absent ones, so it
    /// wins only past a break-even sparsity of `index_bits / (bits + index_bits)`. At 4-bit
    /// weights and 16-bit indices that break-even is 80%, and everything below it is a loss.
    Sparse {
        /// Width of one stored index, in bits. `ceil(log2(fan_in))` for a per-row column index.
        index_bits: u32,
    },
    /// One bit per position saying whether a parameter is there, plus the survivors' values.
    ///
    /// Costs `total + kept * bits`. Break-even against dense is `1/bits` sparsity, so it is a win
    /// almost immediately and never worse than dense by more than one bit per parameter. It is
    /// the layout a fixed crossbar with a validity mask would imply; this review did not locate a
    /// storage-format description for any commercial part, so that is one of three computable
    /// possibilities and not a claim about silicon.
    Bitmask,
}

impl Storage {
    /// Bits to hold `kept` surviving parameters out of `total` positions, at `bits` each.
    ///
    /// `None` when `kept > total`, when a width is zero or past 64, or on `u64` overflow.
    #[must_use]
    pub fn parameter_bits(self, total: u64, kept: u64, bits: u32) -> Option<u64> {
        if kept > total || bits == 0 || bits > 64 {
            return None;
        }
        match self {
            Self::Dense => total.checked_mul(u64::from(bits)),
            Self::Sparse { index_bits } => {
                if index_bits == 0 || index_bits > 64 {
                    return None;
                }
                kept.checked_mul(u64::from(bits) + u64::from(index_bits))
            }
            Self::Bitmask => total.checked_add(kept.checked_mul(u64::from(bits))?),
        }
    }

    /// The sparsity past which this layout beats [`Storage::Dense`] at the same bit width.
    ///
    /// `None` for [`Storage::Dense`], which cannot beat itself. The closed forms are
    /// `index_bits / (bits + index_bits)` for [`Storage::Sparse`] and `1 / bits` for
    /// [`Storage::Bitmask`], and both are checked against the byte counts in this module's tests.
    #[must_use]
    pub fn break_even_sparsity(self, bits: u32) -> Option<f64> {
        if bits == 0 || bits > 64 {
            return None;
        }
        match self {
            Self::Dense => None,
            Self::Sparse { index_bits } => {
                if index_bits == 0 || index_bits > 64 {
                    return None;
                }
                Some(f64::from(index_bits) / f64::from(bits + index_bits))
            }
            Self::Bitmask => Some(1.0 / f64::from(bits)),
        }
    }
}

/// The shape of the model being compressed, in the units a footprint is computed from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelShape {
    /// Learned scalars before pruning — synaptic weights, and any biases stored with them.
    pub parameters: u64,
    /// Per-neuron state scalars: membrane potentials, adaptation variables, refractory counters.
    ///
    /// Counted because a chip has to hold it, and because pruning weights does not reduce it. On
    /// a heavily pruned network the state can become the larger term, which is the argument for
    /// [`prune_structured`] over [`prune_magnitude`].
    pub state_values: u64,
    /// Stored width of one state scalar, in bits, 1 to 64.
    pub state_bits: u32,
}

impl ModelShape {
    /// Reject a shape that cannot be stored.
    ///
    /// # Errors
    ///
    /// [`CompressError::Empty`] for zero parameters, [`CompressError::BadValue`] for a
    /// `state_bits` of zero or above 64.
    pub fn validate(&self) -> Result<(), CompressError> {
        if self.parameters == 0 {
            return Err(CompressError::Empty { what: "parameters" });
        }
        if self.state_bits == 0 || self.state_bits > 64 {
            return Err(CompressError::BadValue {
                what: "state_bits, which must be in 1..=64",
                value: f64::from(self.state_bits),
            });
        }
        Ok(())
    }
}

/// One complete compression setting: how much to prune, how narrow to store, how long to run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Plan {
    /// Fraction of parameters pruned away, `0.0..=1.0`.
    pub sparsity: f64,
    /// Stored weight width in bits, 2 to 31 — the range [`Quantiser`] accepts.
    pub weight_bits: u32,
    /// Ticks per inference, at least 1. The rate-coding window.
    pub ticks: u64,
    /// How the survivors are laid out, which decides whether `sparsity` buys anything.
    pub storage: Storage,
}

impl Plan {
    /// Build and check a plan.
    ///
    /// # Errors
    ///
    /// [`CompressError::BadValue`] for a `sparsity` outside `0..=1`, a `weight_bits` outside
    /// `2..=31`, a `ticks` of zero, or a [`Storage::Sparse`] index width outside `1..=64`.
    pub fn new(
        sparsity: f64,
        weight_bits: u32,
        ticks: u64,
        storage: Storage,
    ) -> Result<Self, CompressError> {
        let p = Self { sparsity, weight_bits, ticks, storage };
        p.validate()?;
        Ok(p)
    }

    /// Reject a plan that cannot be carried out.
    ///
    /// # Errors
    ///
    /// As [`Plan::new`].
    pub fn validate(&self) -> Result<(), CompressError> {
        unit_fraction(self.sparsity, "sparsity, which must be in 0..=1")?;
        if !(2..=31).contains(&self.weight_bits) {
            return Err(CompressError::BadValue {
                what: "weight_bits, which must be in 2..=31",
                value: f64::from(self.weight_bits),
            });
        }
        if self.ticks == 0 {
            return Err(CompressError::BadValue {
                what: "ticks, which must be at least 1",
                value: 0.0,
            });
        }
        if let Storage::Sparse { index_bits } = self.storage
            && (index_bits == 0 || index_bits > 64)
        {
            return Err(CompressError::BadValue {
                what: "index_bits, which must be in 1..=64",
                value: f64::from(index_bits),
            });
        }
        Ok(())
    }

    /// How many of `total` parameters survive: `total - round(sparsity * total)`.
    #[must_use]
    pub fn kept(&self, total: u64) -> u64 {
        let removed = (self.sparsity * total as f64).round();
        let removed = if removed >= total as f64 { total } else { removed as u64 };
        total - removed
    }

    /// The stored footprint of a model of this shape under this plan.
    ///
    /// Builds [`Footprint`] from its public fields rather than through [`Footprint::new`], because
    /// that constructor takes one count and one width and [`Storage::Sparse`] and
    /// [`Storage::Bitmask`] mix widths — an index is not a weight and a mask bit is neither. For
    /// [`Storage::Dense`] the two routes agree exactly, and a test asserts it so this path cannot
    /// drift from the crate's own accounting.
    ///
    /// # Errors
    ///
    /// Whatever [`Plan::validate`] and [`ModelShape::validate`] reject, plus
    /// [`CompressError::Overflow`] on a bit count past `u64`.
    pub fn footprint(&self, shape: &ModelShape) -> Result<Footprint, CompressError> {
        self.validate()?;
        shape.validate()?;
        let kept = self.kept(shape.parameters);
        let parameter_bits = self
            .storage
            .parameter_bits(shape.parameters, kept, self.weight_bits)
            .ok_or(CompressError::Overflow { what: "parameter bits" })?;
        let state_bits = shape
            .state_values
            .checked_mul(u64::from(shape.state_bits))
            .ok_or(CompressError::Overflow { what: "state bits" })?;
        Ok(Footprint { parameter_bits, state_bits })
    }

    /// Synaptic operations per inference: the surviving fraction of `syn_ops_per_dense_tick`,
    /// times [`Plan::ticks`].
    ///
    /// ⚠ **This assumes a runtime that skips pruned synapses.** [`crate::sim`] does not — see
    /// [`compact_net`] — so on a network that was zeroed but not compacted the true count is
    /// `syn_ops_per_dense_tick * ticks` regardless of sparsity. The assumption is stated here
    /// rather than buried, because it is exactly the assumption that turns a sparsity figure into
    /// an energy claim.
    ///
    /// `None` on `u64` overflow.
    #[must_use]
    pub fn work(&self, syn_ops_per_dense_tick: u64) -> Option<u64> {
        self.kept(syn_ops_per_dense_tick).checked_mul(self.ticks)
    }
}

// -------------------------------------------------------------------------------------------
// The timestep lever
// -------------------------------------------------------------------------------------------

/// Root-mean-square error of a rate estimate from `ticks` Bernoulli trials at probability `p`.
///
/// **`sqrt(p(1-p)/T)`, exactly.** The spike count is `Binomial(T, p)` and `p̂ = k/T` has variance
/// `p(1-p)/T`; this is its square root, which is both the standard deviation and — since `p̂` is
/// unbiased — the RMS error. Dimensionless, in the same units as `p`.
///
/// `None` for a `p` outside `0..=1`, a non-finite `p`, or `ticks == 0`.
///
/// ⚠ This is the **sampling** term alone. It is the error a perfect rate decoder still makes
/// because it counted a finite number of spikes. It says nothing about what pruning or
/// quantisation did to `p` itself, and adding those is the caller's job.
#[must_use]
pub fn rate_rms(p: f64, ticks: u64) -> Option<f64> {
    if !p.is_finite() || !(0.0..=1.0).contains(&p) || ticks == 0 {
        return None;
    }
    Some((p * (1.0 - p) / ticks as f64).sqrt())
}

/// The **smallest** tick budget whose rate estimate reaches `target` RMS error: `ceil(p(1-p)/t²)`.
///
/// The inverse of [`rate_rms`], and the reason the timestep lever is a decision rather than a
/// default. At `p = 0.5` — the worst case, where the variance is largest — 5% error needs 100
/// ticks, 2.5% needs 400, and 1.25% needs 1600. **Four times the ticks and four times the joules
/// for each halving.**
///
/// Returns `Some(1)` when `p` is exactly 0 or 1, where the estimate is exact at any budget.
/// `None` for an invalid `p`, a non-positive or non-finite `target`, or a requirement so tight the
/// answer does not fit `u64`.
///
/// # Why this does not simply return `ceil(p(1-p)/target²)`
///
/// ⛔ Because that expression and [`rate_rms`] **disagree at the last place**, and the first draft
/// of this function shipped the disagreement. At `p = 0.1` and `target = 0.02` the true answer is
/// exactly 225, but `0.1 * 0.9` is `0.09000000000000001` in binary, the quotient evaluates to
/// `225.00000000000003`, and the ceiling returns 226 — while `rate_rms(0.1, 225)` rounds to
/// `0.019999999999999997` and already meets the target. A caller who trusted the returned budget
/// would have run one tick per inference more than it needed, forever, for a rounding artefact.
///
/// So the analytic inversion is used as a *starting point* and is then reconciled against
/// [`rate_rms`] itself, stepping until the returned `T` is the smallest one whose own reported
/// error meets the target. Both loops terminate because [`rate_rms`] is non-increasing in `T`
/// (not strictly decreasing: past about `2.5e17` consecutive `u64` budgets cast to the same
/// `f64` and report the identical error) and each step is an integer, and in practice each moves
/// by at most one. The contract is therefore exact and self-consistent:
/// `rate_rms(p, ticks_for_rms(p, e)) <= e`, and `rate_rms(p, t - 1) > e`.
///
/// ⛔ The second loop is not dead code, though every early test input left it so. At
/// `p = 0.49, target = 7e-6` the naive ceiling is `5_100_000_000` and `rate_rms` there is
/// `7.000000000000001e-6` — **above** the target — so the honest answer is one tick more. That is
/// the mirror of the defect above, and the one direction that ships a wrong budget rather than a
/// wasted tick. It appears only at targets below about `3e-5`, which the test now reaches.
#[must_use]
pub fn ticks_for_rms(p: f64, target: f64) -> Option<u64> {
    if !p.is_finite() || !(0.0..=1.0).contains(&p) || !target.is_finite() || target <= 0.0 {
        return None;
    }
    let var = p * (1.0 - p);
    if var == 0.0 {
        return Some(1);
    }
    let guess = (var / (target * target)).ceil();
    if !guess.is_finite() || guess >= 9.0e18 {
        return None;
    }
    let mut t = (guess as u64).max(1);
    let meets = |k: u64| rate_rms(p, k).is_some_and(|e| e <= target);
    while t > 1 && meets(t - 1) {
        t -= 1;
    }
    while !meets(t) {
        t = t.checked_add(1)?;
    }
    Some(t)
}

/// How much the rate-sampling error grows when the budget moves from `from` ticks to `to`:
/// `sqrt(from / to)`.
///
/// Independent of `p`, because `p` cancels: the whole `p`-dependence of [`rate_rms`] is a factor
/// that both budgets share. `None` if either count is zero.
#[must_use]
pub fn error_multiplier(from: u64, to: u64) -> Option<f64> {
    if from == 0 || to == 0 {
        return None;
    }
    Some((from as f64 / to as f64).sqrt())
}

/// How much the work changes when the budget moves from `from` ticks to `to`: `to / from`.
///
/// **Linear**, against [`error_multiplier`]'s square root. The gap between these two functions is
/// the whole case for the timestep lever. `None` if `from` is zero.
#[must_use]
pub fn work_multiplier(from: u64, to: u64) -> Option<f64> {
    if from == 0 {
        return None;
    }
    Some(to as f64 / from as f64)
}

/// Rescale a [`Ledger`]'s counts from a `from`-tick run to a `to`-tick one.
///
/// ⚠ **A model, not a measurement.** It multiplies every per-tick count by `to/from` and leaves
/// [`Ledger::reads`] alone, because a readout happens once per inference rather than once per
/// tick. That is correct when firing *rates* are unchanged by the budget — true for rate coding at
/// a fixed input, and false for any latency or rank-order code, where shortening the window
/// removes late spikes preferentially rather than proportionally. Re-run the simulation if you
/// need the real number; this is for sweeping a Pareto front without re-running it hundreds of
/// times.
///
/// Integer arithmetic throughout, truncating toward zero, so the result is exactly reproducible.
/// `None` if `from` is zero or a product overflows `u64`.
#[must_use]
pub fn ledger_at_ticks(base: &Ledger, from: u64, to: u64) -> Option<Ledger> {
    if from == 0 {
        return None;
    }
    let scale = |x: u64| -> Option<u64> { Some(x.checked_mul(to)? / from) };
    Some(Ledger {
        syn_ops: scale(base.syn_ops)?,
        syn_fetches: scale(base.syn_fetches)?,
        neuron_updates_idle: scale(base.neuron_updates_idle)?,
        neuron_updates_driven: scale(base.neuron_updates_driven)?,
        spikes_out: scale(base.spikes_out)?,
        reads: base.reads,
    })
}

/// A measured rate-decoding error as a function of the tick budget.
#[derive(Debug, Clone, PartialEq)]
pub struct RateSweep {
    /// The probability that was encoded and decoded.
    pub p: f64,
    /// Tick budgets, in the order they were measured.
    pub ticks: Vec<u64>,
    /// Root-mean-square decoding error at each budget, same length as [`RateSweep::ticks`].
    pub rms_error: Vec<f64>,
    /// Mean absolute decoding error at each budget, same length as [`RateSweep::ticks`].
    pub mean_abs_error: Vec<f64>,
    /// Independent estimates averaged at each budget.
    pub trials: usize,
}

impl RateSweep {
    /// The exponent `q` in `rms_error ~ T^q`, fitted by least squares on the logs.
    ///
    /// The analytic answer is **exactly −1/2** and does not depend on `p`. A fit that comes back
    /// near −1 is measuring something else — most often a deterministic quantisation residual
    /// rather than a sampling error, which is the `p = -1` case
    /// [`crate::convert::ErrorCurve::fit_exponent`] documents for analog input with reset by
    /// subtraction.
    ///
    /// `None` when fewer than two budgets produced a strictly positive finite error.
    #[must_use]
    pub fn fit_exponent(&self) -> Option<f64> {
        fit_power_law(&self.ticks, &self.rms_error)
    }

    /// The closed-form RMS error at each measured budget, for comparison with the measurement.
    ///
    /// `None` if [`RateSweep::p`] is not a probability or a budget was zero, neither of which a
    /// sweep built by [`rate_error_vs_ticks`] can contain.
    #[must_use]
    pub fn predicted(&self) -> Option<Vec<f64>> {
        self.ticks.iter().map(|&t| rate_rms(self.p, t)).collect()
    }
}

/// Least squares for the exponent `q` in `error ~ T^q`.
///
/// Delegates to [`crate::convert::ErrorCurve::fit_exponent`] rather than repeating the
/// regression: that function drops non-positive and non-finite points, refuses a fit through
/// fewer than two, and is already checked there. Its field is named for conversion's own error
/// measure; the regression does not care which measure it is handed, and this module hands it an
/// RMS.
///
/// `None` on mismatched lengths, or whenever the underlying fit refuses.
#[must_use]
pub fn fit_power_law(ticks: &[u64], error: &[f64]) -> Option<f64> {
    if ticks.len() != error.len() {
        return None;
    }
    ErrorCurve { ticks: ticks.to_vec(), mean_abs_error: error.to_vec() }.fit_exponent()
}

/// Measure the rate-decoding error over a sweep of tick budgets.
///
/// Encodes `p` as an independent Bernoulli draw per tick, decodes by counting and dividing, and
/// repeats `trials` times per budget. Deterministic under `rng`'s seed.
///
/// This is the experiment the module doc's `1/sqrt(T)` claim rests on, and it is run against
/// [`rate_rms`]'s closed form rather than against a previous run of itself.
///
/// # Cost
///
/// `trials · sum(ticks)` random draws. Nothing bounds it: `ticks = [1_000_000_000]` at 4000 trials
/// is 4e12 draws, and this function will attempt them. Stated rather than refused, because the
/// budgets a caller wants to measure are the caller's; the sweep in this module's tests runs
/// 8 to 1024 ticks at 4000 trials.
///
/// # Errors
///
/// [`CompressError::BadValue`] for a `p` outside `0..=1`, a zero entry in `ticks`, or zero
/// `trials`; [`CompressError::Empty`] for an empty `ticks`.
pub fn rate_error_vs_ticks(
    p: f64,
    ticks: &[u64],
    trials: usize,
    rng: &mut Rng,
) -> Result<RateSweep, CompressError> {
    unit_fraction(p, "p, which must be a probability in 0..=1")?;
    if ticks.is_empty() {
        return Err(CompressError::Empty { what: "tick budgets" });
    }
    if trials == 0 {
        return Err(CompressError::BadValue { what: "trials, which must be at least 1", value: 0.0 });
    }
    let mut rms_error = Vec::with_capacity(ticks.len());
    let mut mean_abs_error = Vec::with_capacity(ticks.len());
    for &t in ticks {
        if t == 0 {
            return Err(CompressError::BadValue {
                what: "a tick budget, which must be at least 1",
                value: 0.0,
            });
        }
        let mut sum_sq = 0.0;
        let mut sum_abs = 0.0;
        for _ in 0..trials {
            let mut count = 0u64;
            for _ in 0..t {
                if rng.next_f64() < p {
                    count += 1;
                }
            }
            let e = count as f64 / t as f64 - p;
            sum_sq += e * e;
            sum_abs += e.abs();
        }
        rms_error.push((sum_sq / trials as f64).sqrt());
        mean_abs_error.push(sum_abs / trials as f64);
    }
    Ok(RateSweep { p, ticks: ticks.to_vec(), rms_error, mean_abs_error, trials })
}

// -------------------------------------------------------------------------------------------
// Knowledge distillation
// -------------------------------------------------------------------------------------------

/// Softmax of `logits` divided by `temperature`.
///
/// Computed with the maximum subtracted first, which changes nothing — softmax is invariant under
/// adding a constant to every logit — and keeps `exp` away from overflow. A raised temperature
/// flattens the distribution; the limit as it grows is uniform, and as it falls to zero it is a
/// one-hot at the largest logit.
///
/// # Errors
///
/// [`CompressError::Empty`] for no logits, [`CompressError::NonFinite`] naming the first bad one,
/// [`CompressError::BadValue`] for a `temperature` that is not finite and strictly positive.
///
/// [`CompressError::NotADistribution`] naming `"softmax output"` when the exponentials do not sum
/// to a positive finite number. ⛔ That is **not** dead code, though subtracting the maximum
/// makes the largest term `exp(0) = 1` and so puts the sum in `1..=n` for every representable
/// input. It is reached through the temperature: a logit divided by a small enough one overflows
/// to an infinity, the shift is then `inf - inf`, which is `NaN`, and so is the sum. Refusing is
/// the honest answer rather than a case to handle, because once two logits have both overflowed
/// to `+inf` the ratio that decided the answer is gone — at a temperature of `1e-308`,
/// `[1e10, 1e10]` and `[1e10, 1e10 - 1]` are a uniform distribution and a one-hot and are the same
/// pair of infinities.
pub fn softmax_t(logits: &[f64], temperature: f64) -> Result<Vec<f64>, CompressError> {
    all_finite(logits, "logits")?;
    if !temperature.is_finite() || temperature <= 0.0 {
        return Err(CompressError::BadValue {
            what: "temperature, which must be finite and strictly positive",
            value: temperature,
        });
    }
    let scaled: Vec<f64> = logits.iter().map(|z| z / temperature).collect();
    let m = scaled.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let mut e: Vec<f64> = scaled.iter().map(|z| (z - m).exp()).collect();
    let sum: f64 = e.iter().sum();
    if !(sum > 0.0) || !sum.is_finite() {
        return Err(CompressError::NotADistribution { what: "softmax output", sum });
    }
    for x in &mut e {
        *x /= sum;
    }
    Ok(e)
}

/// Reject a vector that is not a probability distribution.
fn check_distribution(p: &[f64], what: &'static str) -> Result<(), CompressError> {
    all_finite(p, what)?;
    let mut sum = 0.0;
    for &x in p {
        if x < 0.0 {
            return Err(CompressError::NotADistribution { what, sum: x });
        }
        sum += x;
    }
    if (sum - 1.0).abs() > 1e-9 {
        return Err(CompressError::NotADistribution { what, sum });
    }
    Ok(())
}

/// Kullback-Leibler divergence `D(p || q) = Σ p ln(p/q)`, in nats.
///
/// Zero exactly when `p` and `q` are the same values, non-negative always (Gibbs' inequality), and
/// asymmetric: `D(p||q)` and `D(q||p)` are different numbers and the distillation loss uses the
/// first with the teacher as `p`.
///
/// `0 ln 0` is taken as 0, which is its limit.
///
/// # Errors
///
/// [`CompressError::NotADistribution`] if either argument is not one,
/// [`CompressError::LengthMismatch`], or [`CompressError::ZeroSupport`] where `q` is zero and `p`
/// is not — the case whose true value is `+inf`.
pub fn kl_divergence(p: &[f64], q: &[f64]) -> Result<f64, CompressError> {
    if p.len() != q.len() {
        return Err(CompressError::LengthMismatch { a: p.len(), b: q.len() });
    }
    check_distribution(p, "p")?;
    check_distribution(q, "q")?;
    let mut acc = 0.0;
    for i in 0..p.len() {
        if p[i] == 0.0 {
            continue;
        }
        if q[i] == 0.0 {
            return Err(CompressError::ZeroSupport { what: "q", index: i });
        }
        acc += p[i] * (p[i] / q[i]).ln();
    }
    Ok(acc)
}

/// Cross-entropy of a predicted distribution against a hard label: `-ln(q[label])`, in nats.
///
/// # Errors
///
/// [`CompressError::NotADistribution`], [`CompressError::LabelOutOfRange`], or
/// [`CompressError::ZeroSupport`] when the predicted probability of the true class is exactly
/// zero and the loss is infinite.
pub fn cross_entropy(q: &[f64], label: usize) -> Result<f64, CompressError> {
    check_distribution(q, "q")?;
    if label >= q.len() {
        return Err(CompressError::LabelOutOfRange { label, classes: q.len() });
    }
    if q[label] == 0.0 {
        return Err(CompressError::ZeroSupport { what: "q", index: label });
    }
    Ok(-q[label].ln())
}

/// The two terms of a distillation loss, kept apart.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DistillLoss {
    /// `T² · D(teacher_T || student_T)` in nats — the soft-target term, already carrying Hinton's
    /// `T²` factor.
    pub soft: f64,
    /// `-ln(student_1[label])` in nats — the ordinary hard-label cross-entropy at temperature 1.
    pub hard: f64,
    /// `alpha * soft + (1 - alpha) * hard`.
    pub total: f64,
}

/// The distillation objective of Hinton, Vinyals and Dean (NIPS 2014 Deep Learning Workshop).
///
/// # The mechanism, and why it works at all
///
/// A trained teacher's output is not a label, it is a distribution, and the small probabilities
/// in it carry the teacher's similarity structure: that this 2 looks somewhat like a 3 and not at
/// all like a 7. Hinton calls it *dark knowledge*. A one-hot label throws all of it away. The
/// student is trained on the teacher's distribution instead — but at temperature 1 those small
/// probabilities are numerically negligible, so both networks' logits are divided by a
/// temperature `T > 1` first, which raises the small probabilities into a range where a gradient
/// can see them.
///
/// # The `T²` factor is not cosmetic
///
/// The gradient of the soft cross-entropy with respect to a student logit is `(q_i - p_i) / T`,
/// and at high temperature `q_i - p_i` is itself `O(1/T)` — so the soft gradient vanishes as
/// `1/T²` while the hard-label gradient does not move. Blending the two at a fixed `alpha` would
/// silently turn the soft term off as the temperature rose. Multiplying the soft loss by `T²`
/// restores it, which is what the paper prescribes and what [`Distillation::soft_gradient`]
/// returns. Both the exact gradient and the `T²` cancellation are checked in this module's tests.
///
/// # Divergence rather than cross-entropy
///
/// The paper writes the soft term as a cross-entropy `-Σ p ln q`. This uses the
/// Kullback-Leibler divergence `Σ p ln(p/q)`, which differs by the teacher's own entropy — a
/// constant with respect to the student, so **every gradient is identical**. The reported number
/// is better behaved: it is zero exactly when the student matches the teacher, where the
/// cross-entropy would report the teacher's entropy and leave a reader wondering what the floor
/// was.
///
/// # This is the loss, not a trainer
///
/// There is no optimiser here. [`Distillation::soft_gradient`] returns `∂soft/∂z_student`, which
/// is what a caller feeds to whatever it is using — [`crate::surrogate`] for a spiking student.
/// The hard term's gradient is the standard `q - onehot` and is not duplicated.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Distillation {
    /// Softmax temperature, dimensionless, strictly positive. Kept verbatim as the paper's `T`
    /// because its only meaning is as a divisor of logits; the paper's experiments use 1 to 20.
    pub temperature: f64,
    /// Weight on the soft term, `0.0..=1.0`; the hard term gets `1 - alpha`.
    ///
    /// The paper (arXiv:1503.02531v1, Sec. 2) reports that "the best results were generally
    /// obtained by using a condiderably \[sic\] lower weight on the second objective function",
    /// the second objective being the cross-entropy with the correct labels: a **low** weight on
    /// the hard term, i.e. an `alpha` near 1. ⛔ This doc used to put "considerably better
    /// results" in quotation marks as the paper's words; this review did not locate that phrase,
    /// nor the word "considerably", in its full text (v1 is the only version on arXiv). This
    /// crate does not endorse a value: the right one depends on how good the teacher is, and
    /// nothing here measures that.
    pub alpha: f64,
}

impl Distillation {
    /// Build and check an objective.
    ///
    /// # Errors
    ///
    /// [`CompressError::BadValue`] for a non-positive or non-finite `temperature`, or an `alpha`
    /// outside `0..=1`.
    pub fn new(temperature: f64, alpha: f64) -> Result<Self, CompressError> {
        if !temperature.is_finite() || temperature <= 0.0 {
            return Err(CompressError::BadValue {
                what: "temperature, which must be finite and strictly positive",
                value: temperature,
            });
        }
        unit_fraction(alpha, "alpha, which must be in 0..=1")?;
        Ok(Self { temperature, alpha })
    }

    /// Both terms of the loss and their blend, in nats.
    ///
    /// # Errors
    ///
    /// [`CompressError::LengthMismatch`] if the two logit vectors differ in length, plus whatever
    /// [`softmax_t`], [`kl_divergence`] and [`cross_entropy`] reject.
    pub fn loss(
        &self,
        teacher_logits: &[f64],
        student_logits: &[f64],
        label: usize,
    ) -> Result<DistillLoss, CompressError> {
        if teacher_logits.len() != student_logits.len() {
            return Err(CompressError::LengthMismatch {
                a: teacher_logits.len(),
                b: student_logits.len(),
            });
        }
        let p = softmax_t(teacher_logits, self.temperature)?;
        let q = softmax_t(student_logits, self.temperature)?;
        let soft = self.temperature * self.temperature * kl_divergence(&p, &q)?;
        let hard = cross_entropy(&softmax_t(student_logits, 1.0)?, label)?;
        let total = self.alpha * soft + (1.0 - self.alpha) * hard;
        Ok(DistillLoss { soft, hard, total })
    }

    /// `∂(T² · D(p_T || q_T)) / ∂z_student = T · (q_i - p_i)`.
    ///
    /// The closed form, not a finite difference — and a test compares it against one, which is the
    /// check that catches a missing `1/T`, a missing `T²` or a swapped sign.
    ///
    /// # Errors
    ///
    /// [`CompressError::LengthMismatch`], plus whatever [`softmax_t`] rejects.
    pub fn soft_gradient(
        &self,
        teacher_logits: &[f64],
        student_logits: &[f64],
    ) -> Result<Vec<f64>, CompressError> {
        if teacher_logits.len() != student_logits.len() {
            return Err(CompressError::LengthMismatch {
                a: teacher_logits.len(),
                b: student_logits.len(),
            });
        }
        let p = softmax_t(teacher_logits, self.temperature)?;
        let q = softmax_t(student_logits, self.temperature)?;
        Ok((0..p.len()).map(|i| self.temperature * (q[i] - p[i])).collect())
    }
}

// -------------------------------------------------------------------------------------------
// The Pareto front
// -------------------------------------------------------------------------------------------

/// One operating point: a plan, what it costs, and what it was measured to cost in accuracy.
///
/// The two costs are computed exactly from the plan. **The error is supplied**, because this
/// module does not predict accuracy and will not pretend to. [`rate_rms`] will give you the
/// rate-sampling part of it if that is the only lever you moved.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point {
    /// The setting this point describes.
    pub plan: Plan,
    /// Stored bytes, from [`Plan::footprint`].
    pub bytes: u64,
    /// Synaptic operations per inference, from [`Plan::work`].
    pub syn_ops: u64,
    /// Measured error — lower is better, and the unit is whatever you measured in. Must be finite
    /// and non-negative.
    pub error: f64,
}

impl Point {
    /// Compute a point's costs from its plan and take its error from a measurement.
    ///
    /// # Errors
    ///
    /// Whatever [`Plan::footprint`] rejects, [`CompressError::Overflow`] if the footprint or the
    /// work exceeds `u64`, and [`CompressError::BadValue`] for a non-finite or negative `error`.
    pub fn new(
        plan: Plan,
        shape: &ModelShape,
        syn_ops_per_dense_tick: u64,
        error: f64,
    ) -> Result<Self, CompressError> {
        if !error.is_finite() || error < 0.0 {
            return Err(CompressError::BadValue {
                what: "error, which must be finite and non-negative",
                value: error,
            });
        }
        let bytes = plan
            .footprint(shape)?
            .bytes()
            .ok_or(CompressError::Overflow { what: "footprint bytes" })?;
        let syn_ops = plan
            .work(syn_ops_per_dense_tick)
            .ok_or(CompressError::Overflow { what: "synaptic operations" })?;
        Ok(Self { plan, bytes, syn_ops, error })
    }

    /// Whether this point is at least as good as `other` on all three objectives and strictly
    /// better on at least one.
    ///
    /// All three are costs, so smaller wins on every axis. Two identical points do not dominate
    /// each other — the strict part fails — so duplicates both survive onto the front, which is
    /// correct: neither is a reason to discard the other.
    ///
    /// Requires finite errors on both sides; [`pareto_front`] enforces that before calling this.
    /// With a `NaN` error every comparison is false and the point would neither dominate nor be
    /// dominated, which is why the refusal is at the boundary rather than here.
    #[must_use]
    pub fn dominates(&self, other: &Self) -> bool {
        let no_worse = self.bytes <= other.bytes
            && self.syn_ops <= other.syn_ops
            && self.error <= other.error;
        let better = self.bytes < other.bytes
            || self.syn_ops < other.syn_ops
            || self.error < other.error;
        no_worse && better
    }
}

/// Indices of the non-dominated points, ascending.
///
/// The answer to "which combination of pruning, quantisation and timestep reduction is best" is
/// **a set, not a point**, and this returns the set. Anything not in it is beaten outright by
/// something in it on every axis at once, and can be discarded without knowing how the caller
/// weighs bytes against operations against accuracy. Choosing among the survivors needs a weight,
/// and the weight is the caller's; [`best_under_budget`] is one such rule.
///
/// O(n²), which is the right algorithm for the tens-to-hundreds of candidates a compression sweep
/// produces and the wrong one for millions.
///
/// # Errors
///
/// [`CompressError::Empty`] for no candidates, [`CompressError::NonFinite`] naming the first point
/// with a non-finite error — see [`Point::dominates`] for why that cannot be handled downstream.
pub fn pareto_front(points: &[Point]) -> Result<Vec<usize>, CompressError> {
    if points.is_empty() {
        return Err(CompressError::Empty { what: "candidate points" });
    }
    for (i, p) in points.iter().enumerate() {
        if !p.error.is_finite() {
            return Err(CompressError::NonFinite { what: "candidate errors", index: i });
        }
    }
    let mut front = Vec::new();
    for i in 0..points.len() {
        if !points.iter().enumerate().any(|(j, q)| j != i && q.dominates(&points[i])) {
            front.push(i);
        }
    }
    Ok(front)
}

/// The lowest-error point that fits both budgets.
///
/// Ties are broken by fewest bytes, then fewest synaptic operations, then lowest index — so the
/// answer is deterministic and does not depend on the order the candidates were generated in.
///
/// # Errors
///
/// [`CompressError::Empty`], [`CompressError::NonFinite`] as [`pareto_front`], and
/// [`CompressError::NoFeasiblePoint`] naming both budgets when nothing fits — rather than
/// returning the least infeasible point, which is the failure mode that ships a model that does
/// not board.
pub fn best_under_budget(
    points: &[Point],
    max_bytes: u64,
    max_syn_ops: u64,
) -> Result<usize, CompressError> {
    if points.is_empty() {
        return Err(CompressError::Empty { what: "candidate points" });
    }
    for (i, p) in points.iter().enumerate() {
        if !p.error.is_finite() {
            return Err(CompressError::NonFinite { what: "candidate errors", index: i });
        }
    }
    let mut best: Option<usize> = None;
    for (i, p) in points.iter().enumerate() {
        if p.bytes > max_bytes || p.syn_ops > max_syn_ops {
            continue;
        }
        let better = match best {
            None => true,
            Some(b) => {
                let c = points[b];
                (p.error, p.bytes, p.syn_ops) < (c.error, c.bytes, c.syn_ops)
            }
        };
        if better {
            best = Some(i);
        }
    }
    best.ok_or(CompressError::NoFeasiblePoint {
        candidates: points.len(),
        max_bytes,
        max_syn_ops,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        CompressError, Distillation, ModelShape, Plan, Point, Pruned, Saliency, Storage, Unit,
        best_under_budget, compact_net, cross_entropy, fit_power_law, kl_divergence,
        ledger_at_ticks, pareto_front, prune_below, prune_magnitude, prune_net, prune_structured,
        quantise, rate_error_vs_ticks, rate_rms, softmax_t, ticks_for_rms, work_multiplier,
    };
    use super::{error_multiplier, quantise_for_part};
    use crate::hardware::{DARWIN, HardwareError, LOIHI, Quantiser, Rounding};
    use crate::metrics::MetricError;
    use crate::ledger::Ledger;
    use crate::metrics::Footprint;
    use crate::net::NetBuilder;
    use crate::neuron::Lif;
    use crate::rng::Rng;
    use crate::sim::{Mode, Sim};

    // ---- (a) exact sparsity, and exactly the smallest weights ---------------------------------

    /// Requirement (a), with no tolerance anywhere in it: the count is exact and the SET is exact.
    ///
    /// The second assertion is the one that matters. A pruner that removes the right NUMBER of
    /// weights but the wrong ones passes a count check and destroys the network, so this compares
    /// every removed magnitude against every kept one.
    #[test]
    fn pruning_to_a_requested_sparsity_removes_exactly_the_smallest_weights() {
        // Distinct magnitudes, deliberately unsorted and straddling zero, so that "smallest" means
        // smallest in magnitude and not smallest in value — a pruner that sorted signed values
        // would remove -9.0 and -7.0 first and pass a count check.
        let w = [3.0, -9.0, 0.5, 7.0, -1.5, 2.25, -0.25, 6.0, -4.0, 8.5];
        for (k, frac) in [(0usize, 0.0), (2, 0.2), (5, 0.5), (7, 0.7), (10, 1.0)] {
            let p = prune_magnitude(&w, frac).expect("valid input");
            assert_eq!(p.n_removed, k, "requested {frac} of 10");
            assert_eq!(p.w.iter().filter(|x| **x == 0.0).count(), k);
            assert!((p.removed_fraction() - frac).abs() < 1e-15);
            assert!((p.sparsity() - frac).abs() < 1e-15, "no zeros in the input, so they agree");

            // Every removed magnitude is strictly below every kept one. Exact comparison.
            let removed: Vec<f64> =
                (0..w.len()).filter(|&i| !p.kept[i]).map(|i| w[i].abs()).collect();
            let kept: Vec<f64> = (0..w.len()).filter(|&i| p.kept[i]).map(|i| w[i].abs()).collect();
            for r in &removed {
                for s in &kept {
                    assert!(r < s, "removed |{r}| but kept |{s}|");
                }
            }
            assert!(p.separates_cleanly());
            assert_eq!(p.tied_at_cut, 0, "the magnitudes are distinct");
            assert_eq!(p.newly_zeroed, k, "nothing in the input was already zero");
            // Surviving weights are untouched, bit for bit.
            for i in 0..w.len() {
                if p.kept[i] {
                    assert_eq!(p.w[i].to_bits(), w[i].to_bits());
                }
            }
        }
    }

    /// Requirement (b). `0.0` is the identity BIT FOR BIT, not merely numerically: a pruner that
    /// rewrote `-0.0` to `0.0` on the way through would pass an `==` comparison and would have
    /// changed the data.
    #[test]
    fn pruning_nothing_is_the_identity_and_pruning_everything_leaves_nothing() {
        let w = [-0.0f64, 3.0, -9.0, 0.5, 7.0];
        let none = prune_magnitude(&w, 0.0).expect("valid");
        assert_eq!(none.n_removed, 0);
        assert_eq!(none.newly_zeroed, 0);
        assert!(none.cut.is_none());
        for i in 0..w.len() {
            assert_eq!(none.w[i].to_bits(), w[i].to_bits(), "entry {i} was rewritten");
            assert!(none.kept[i]);
        }

        let all = prune_magnitude(&w, 1.0).expect("valid");
        assert_eq!(all.n_removed, w.len());
        assert_eq!(all.newly_zeroed, 4, "the -0.0 was already zero and lost nothing");
        assert!(all.smallest_kept.is_none());
        assert!((all.sparsity() - 1.0).abs() < 1e-15);
        assert_eq!(all.n_kept(), 0);
        for v in &all.w {
            assert_eq!(v.to_bits(), 0u64, "every entry is positive zero");
        }
    }

    /// Achieved, not requested. The two diverge exactly when the input already held zeros, and
    /// this pins the divergence rather than trusting the doc.
    #[test]
    fn achieved_sparsity_is_reported_not_requested() {
        let w = [0.0, 0.0, 1.0, 2.0];
        let p = prune_magnitude(&w, 0.25).expect("valid");
        assert_eq!(p.n_removed, 1, "one of four selected");
        assert_eq!(p.newly_zeroed, 0, "and it was already zero, so nothing was lost");
        assert!((p.removed_fraction() - 0.25).abs() < 1e-15);
        assert!((p.sparsity() - 0.5).abs() < 1e-15, "the output is half zeros, not a quarter");
    }

    /// A sparsity that is not a multiple of `1/n` rounds to the nearest achievable count, and
    /// `round` is not `floor`.
    ///
    /// ⛔ Added after a mutation sweep: replacing `.round()` with `.floor()` in BOTH
    /// `prune_magnitude` and `Plan::kept` left every other test in this module green, because
    /// every sparsity they used multiplied out to a whole number. A pruner silently one weight
    /// short of its request is the kind of defect that never surfaces until two runs of the same
    /// sweep disagree.
    #[test]
    fn a_sparsity_that_is_not_achievable_rounds_to_the_nearest_count() {
        let three = [1.0, 2.0, 3.0];
        // 0.5 * 3 = 1.5 -> 2 under round-half-away-from-zero, 1 under floor.
        assert_eq!(prune_magnitude(&three, 0.5).expect("valid").n_removed, 2);
        // 0.4 * 3 = 1.2 -> 1 either way; 0.6 * 3 = 1.8 -> 2 under round, 1 under floor.
        assert_eq!(prune_magnitude(&three, 0.4).expect("valid").n_removed, 1);
        assert_eq!(prune_magnitude(&three, 0.6).expect("valid").n_removed, 2);
        let seven: Vec<f64> = (1..=7).map(f64::from).collect();
        assert_eq!(prune_magnitude(&seven, 0.5).expect("valid").n_removed, 4, "3.5 rounds up");

        // `Plan::kept` is a separate implementation of the same rounding and needs its own pin.
        let plan = |s: f64| Plan::new(s, 8, 1, Storage::Dense).expect("valid");
        assert_eq!(plan(0.5).kept(3), 1, "2 of 3 removed");
        assert_eq!(plan(0.6).kept(3), 1);
        assert_eq!(plan(0.4).kept(3), 2);
        assert_eq!(plan(0.5).kept(7), 3, "4 of 7 removed");
        assert_eq!(plan(1.0).kept(7), 0);
        assert_eq!(plan(0.0).kept(7), 7);

        // And `prune_structured` rounds its unit count the same way, by its own third copy of the
        // arithmetic — which a mutation sweep found the first draft of this test did not reach.
        let w = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let units = |f: f64| {
            prune_structured(&w, 3, 2, Unit::OutputNeuron, Saliency::L2, f)
                .expect("valid")
                .removed
                .len()
        };
        assert_eq!(units(0.5), 2, "1.5 rows rounds to 2, not 1");
        assert_eq!(units(0.4), 1);
        assert_eq!(units(0.6), 2);
        assert_eq!(units(1.0), 3);
        assert_eq!(units(0.0), 0);
    }

    /// The index tie-break is load-bearing, and `sort_unstable_by` will not supply it.
    ///
    /// ⛔ Two drafts of this test failed to catch the mutation that deletes `.then(a.cmp(&b))`.
    /// The first used a four-element tie, where the sort's small-input path is insertion sort and
    /// happens to be stable; the second used thirty tied weights in one contiguous block, where
    /// the pattern-defeating sort recognises the run and leaves it alone. A tie only gets
    /// scrambled when the sort has to *move* its members, so this interleaves the tied group with
    /// larger weights and sweeps six lengths — the point being that the guarantee must come from
    /// the comparator and not from luck about the sort's internals.
    #[test]
    fn the_tie_break_is_by_ascending_index_however_the_tie_is_arranged() {
        for n in [7usize, 16, 31, 64, 129, 256] {
            for stride in [2usize, 3, 5] {
                // Members of the tie sit at every `stride`-th index, interleaved with weights the
                // sort must partition past.
                let w: Vec<f64> =
                    (0..n).map(|i| if i % stride == 0 { 1.0 } else { 5.0 }).collect();
                let tied: Vec<usize> = (0..n).filter(|i| i % stride == 0).collect();
                let take = tied.len() / 2;
                if take == 0 {
                    continue;
                }
                let p = prune_magnitude(&w, take as f64 / n as f64).expect("valid");
                assert_eq!(p.n_removed, take, "n = {n}, stride = {stride}");
                let removed: Vec<usize> = (0..n).filter(|&i| !p.kept[i]).collect();
                assert_eq!(
                    removed,
                    tied[..take].to_vec(),
                    "n = {n}, stride = {stride}: the tie must break to the LOWEST indices"
                );
                assert_eq!(p.tied_at_cut, tied.len(), "the whole tie is reported");
                assert!(!p.separates_cleanly());
            }
        }
        // The same guarantee for structured pruning, whose comparator carries its OWN tie-break —
        // and which a contiguous block of identical rows did not exercise either, for the same
        // reason. Interleaved, swept, both axes.
        for n_units in [16usize, 33, 96] {
            for stride in [2usize, 3] {
                for unit in [Unit::OutputNeuron, Unit::InputChannel] {
                    let (n_out, n_in) = match unit {
                        Unit::OutputNeuron => (n_units, 3),
                        Unit::InputChannel => (3, n_units),
                    };
                    let mut w = vec![0.0f64; n_out * n_in];
                    for o in 0..n_out {
                        for i in 0..n_in {
                            let u = match unit {
                                Unit::OutputNeuron => o,
                                Unit::InputChannel => i,
                            };
                            w[o * n_in + i] = if u % stride == 0 { 1.0 } else { 5.0 };
                        }
                    }
                    let tied: Vec<usize> = (0..n_units).filter(|u| u % stride == 0).collect();
                    let take = tied.len() / 2;
                    let s = prune_structured(
                        &w,
                        n_out,
                        n_in,
                        unit,
                        Saliency::L2,
                        take as f64 / n_units as f64,
                    )
                    .expect("valid shape");
                    assert_eq!(
                        s.removed,
                        tied[..take].to_vec(),
                        "{unit:?}, {n_units} units, stride {stride}: tie must break to low indices"
                    );
                }
            }
        }
    }

    /// A tie at the cut is arbitrary, and saying so is the difference between a reproducible
    /// result and a mysterious one.
    #[test]
    fn a_tie_at_the_cut_is_reported_rather_than_hidden() {
        let w = [1.0, -1.0, 1.0, 5.0];
        let p = prune_magnitude(&w, 0.25).expect("valid");
        assert_eq!(p.n_removed, 1);
        assert_eq!(p.cut, Some(1.0));
        assert_eq!(p.smallest_kept, Some(1.0));
        assert!(!p.separates_cleanly());
        assert_eq!(p.tied_at_cut, 3, "three weights share magnitude 1.0");
        assert!(!p.kept[0], "the tie was broken by ascending index");

        // And a clean cut reports zero, so the field is not a constant.
        let q = prune_magnitude(&[1.0, 2.0, 3.0, 4.0], 0.5).expect("valid");
        assert_eq!(q.tied_at_cut, 0);
        assert!(q.separates_cleanly());

        // ⛔ `cut` is the LARGEST removed magnitude, and with one weight removed the largest and
        // the smallest coincide, so "smallest removed" passed above. Two removed from
        // [1, 2, 2, 5]: the cut is 2, the smallest kept is 2, and the tie is real — under the
        // wrong definition the cut is 1, the record says "clean", and the tie count is 0.
        let r = prune_magnitude(&[1.0, 2.0, 2.0, 5.0], 0.5).expect("valid");
        assert_eq!(r.n_removed, 2);
        assert_eq!(r.cut, Some(2.0));
        assert_eq!(r.smallest_kept, Some(2.0));
        assert!(!r.separates_cleanly());
        assert_eq!(r.tied_at_cut, 2);
    }

    /// Threshold pruning does not get to choose its sparsity, and the record says what it got.
    #[test]
    fn threshold_pruning_reports_the_sparsity_it_achieved() {
        let w = [0.1, 0.9, 0.2, 1.4, -0.05, -2.0];
        let p = prune_below(&w, 0.25).expect("valid");
        assert_eq!(p.n_removed, 3, "0.1, 0.2 and -0.05 are below 0.25");
        assert!((p.sparsity() - 0.5).abs() < 1e-15);
        assert_eq!(p.cut, Some(0.2), "largest magnitude removed");
        assert_eq!(p.smallest_kept, Some(0.9));
        assert!(p.separates_cleanly());
        // A threshold at exactly a weight's magnitude keeps it: the cut is strictly below.
        let q = prune_below(&w, 0.2).expect("valid");
        assert_eq!(q.n_removed, 2, "0.2 is not strictly below 0.2");
    }

    // ---- (d) structured pruning removes whole units -------------------------------------------

    /// Requirement (d), asserted STRUCTURALLY: every element of a removed row is zero and every
    /// element of a kept row is bit-identical. A count of zeroed weights would pass for a pruner
    /// that scattered its removals across rows, which is the defect being excluded.
    #[test]
    fn structured_pruning_removes_whole_rows_and_leaves_the_rest_bit_identical() {
        // 4 outputs x 3 inputs, rows with L2 norms 1, 2, 3, 4 times a common shape.
        let base = [0.6, 0.8, 0.0];
        let mut w = Vec::new();
        for scale in [3.0, 1.0, 4.0, 2.0] {
            w.extend(base.iter().map(|x| x * scale));
        }
        let s = prune_structured(&w, 4, 3, Unit::OutputNeuron, Saliency::L2, 0.5)
            .expect("valid shape");
        assert_eq!(s.removed, vec![1, 3], "the two smallest-norm rows, by index");
        assert!((s.unit_sparsity() - 0.5).abs() < 1e-15);

        for o in 0..4 {
            let row = s.row(o).expect("in range");
            if s.is_removed(o) {
                for (i, v) in row.iter().enumerate() {
                    assert_eq!(v.to_bits(), 0u64, "row {o} element {i} survived a removal");
                }
            } else {
                for i in 0..3 {
                    assert_eq!(
                        row[i].to_bits(),
                        w[o * 3 + i].to_bits(),
                        "row {o} element {i} was disturbed"
                    );
                }
            }
        }
        // Weight sparsity exceeds unit sparsity here because `base` holds a zero of its own: 2 of
        // 3 columns are non-zero in the surviving rows. 6 removed + 2 pre-existing of 12.
        assert!((s.weight_sparsity() - 8.0 / 12.0).abs() < 1e-15);
        assert!(s.row(4).is_none(), "past the end is None, not a panic");
    }

    /// The column form, asserted the same way: a whole input channel goes, across every row.
    #[test]
    fn structured_pruning_by_column_removes_whole_input_channels() {
        // 3 outputs x 4 inputs; column magnitudes 1, 9, 2, 8 in every row.
        let col = [1.0, 9.0, 2.0, 8.0];
        let mut w = Vec::new();
        for r in 0..3 {
            w.extend(col.iter().map(|x| x * f64::from(r + 1)));
        }
        let s = prune_structured(&w, 3, 4, Unit::InputChannel, Saliency::L1, 0.5)
            .expect("valid shape");
        assert_eq!(s.removed, vec![0, 2]);
        for o in 0..3 {
            for i in 0..4 {
                let got = s.w[o * 4 + i];
                if s.is_removed(i) {
                    assert_eq!(got.to_bits(), 0u64, "({o},{i}) should be gone");
                } else {
                    assert_eq!(got.to_bits(), w[o * 4 + i].to_bits(), "({o},{i}) was disturbed");
                }
            }
        }
        assert!((s.weight_sparsity() - 0.5).abs() < 1e-15);
        // The reported fields are not echoes: `units()` is the INPUT count on this axis, and the
        // saliencies are the L1 column sums, col * (1 + 2 + 3).
        assert_eq!(s.units(), 4);
        assert!((s.unit_sparsity() - 0.5).abs() < 1e-15);
        assert_eq!(s.saliencies, vec![6.0, 54.0, 12.0, 48.0]);
        assert_eq!((s.unit, s.saliency), (Unit::InputChannel, Saliency::L1));
    }

    /// ⛔ THE SORT IS LOAD-BEARING. `is_removed` binary-searches `removed`, which must therefore be
    /// ascending; both fixtures above happened to remove units already in index order, so the
    /// sort could be deleted with every test green — and then a caller filtering by `is_removed`
    /// got `false` for a unit that had been removed. Norms 9, 5, 1 at fraction 2/3 remove units
    /// in saliency order [2, 1], which is descending in index.
    #[test]
    fn the_removed_list_is_ascending_however_the_saliencies_order_it() {
        let w = [9.0, 0.0, 5.0, 0.0, 1.0, 0.0]; // 3 rows x 2 inputs, L2 norms 9, 5, 1
        let s = prune_structured(&w, 3, 2, Unit::OutputNeuron, Saliency::L2, 2.0 / 3.0)
            .expect("valid");
        assert_eq!(s.removed, vec![1, 2]);
        for u in 0..3 {
            assert_eq!(s.is_removed(u), s.removed.contains(&u), "unit {u}");
        }
        assert!(!s.is_removed(0));
        assert!(s.is_removed(2), "the lowest-saliency unit, which an unsorted list hid");
        assert_eq!(s.saliencies, vec![9.0, 5.0, 1.0]);
        assert_eq!((s.unit, s.saliency), (Unit::OutputNeuron, Saliency::L2));
        assert_eq!(s.units(), 3);
        assert!((s.unit_sparsity() - 2.0 / 3.0).abs() < 1e-15);
    }

    /// The saliency scores themselves, against hand arithmetic with no tolerance. 3-4-5.
    #[test]
    fn the_saliencies_match_hand_arithmetic() {
        let v = [3.0, -4.0];
        assert_eq!(Saliency::L2.of(&v), 5.0);
        assert_eq!(Saliency::L1.of(&v), 7.0);
        assert_eq!(Saliency::Max.of(&v), 4.0);
        assert_eq!(Saliency::L2.of(&[]), 0.0, "a unit with no weights scores zero");
    }

    // ---- quantisation ------------------------------------------------------------------------

    /// Pruning survives quantisation exactly, which is the property that makes the order
    /// prune-then-quantise safe. Zero divided by any step rounds to code 0 and back to +0.0.
    #[test]
    fn a_pruned_zero_survives_quantisation_exactly() {
        let w = [0.02, -0.9, 0.31, 0.004, -0.55, 0.12, -0.08, 1.0];
        let p = prune_magnitude(&w, 0.5).expect("valid");
        let q = quantise(&p.w, 4, Rounding::Nearest, None).expect("finite, non-empty");
        let back = q.values();
        for i in 0..w.len() {
            if p.kept[i] {
                assert_ne!(q.codes[i], 0, "a kept weight was quantised away at index {i}");
            } else {
                assert_eq!(q.codes[i], 0, "a pruned weight took a non-zero code at {i}");
                assert_eq!(back[i], 0.0);
            }
        }
        // The step is set by the largest magnitude, which pruning does not touch.
        let dense = quantise(&w, 4, Rounding::Nearest, None).expect("valid");
        assert_eq!(q.step, dense.step, "pruning the smallest weights cannot move the scale");
        assert!(q.within_half_lsb(1e-12));

        // ⛔ THE WIDTH THE CALLER ASKED FOR. `q.bits` was read by nothing: `bits + 1`, `31` and `8`
        // in place of `bits` all passed, because the scale comes from the same maximum magnitude
        // at every width and every assertion above is width-free. The step is not: a symmetric
        // quantiser's step is `max / (2^(bits-1) - 1)`, which halves with every added bit.
        assert_eq!(q.bits, 4);
        for bits in [2u32, 3, 5, 8, 16] {
            let qb = quantise(&w, bits, Rounding::Nearest, None).expect("valid");
            assert_eq!(qb.bits, bits);
            let want_step = 1.0 / ((1u64 << (bits - 1)) - 1) as f64;
            assert!(
                (qb.step - want_step).abs() < 1e-15,
                "bits {bits}: step {} against max/(2^(bits-1)-1) = {want_step}",
                qb.step
            );
        }
    }

    /// ⛔ `quantise_for_part` had no test at all — the one public function in the module with
    /// none. It takes its width from the part: Loihi states "any weight precision between one
    /// and nine bits, signed or unsigned" (Davies et al., *Loihi: A Neuromorphic Manycore
    /// Processor with On-Chip Learning*, IEEE Micro 38(1):82-99 (2018), feature list, "Variable
    /// synaptic formats"), so the widest is 9; Table 1 (learning-rule product terms, encoding 9)
    /// lists the synaptic weight as 9b signed. ⛔ This doc used to cite "Table 1: 1 to 9 bits";
    /// the one-to-nine range is in the feature bullet, and Table 1 supports only the 9-bit field.
    /// A part whose width this review did not locate is a refusal that names the part and the
    /// field.
    #[test]
    fn quantise_for_part_takes_the_width_from_the_part_or_refuses_by_name() {
        let w = [0.02, -0.9, 0.31, 1.0];
        let q = quantise_for_part(&LOIHI, &w, Rounding::Nearest, None).expect("Loihi states a width");
        assert_eq!(q.bits, 9);
        let same = quantise(&w, 9, Rounding::Nearest, None).expect("valid");
        assert_eq!(q.codes, same.codes);
        assert_eq!(q.step, same.step);
        assert!(matches!(
            quantise_for_part(&DARWIN, &w, Rounding::Nearest, None),
            Err(CompressError::Hardware(HardwareError::UnstatedSpec { field: "weight_bits", .. }))
        ));
    }

    /// Stochastic rounding without a seed is a refusal, not a silent seed — and with one, it
    /// really is stochastic rounding.
    ///
    /// ⛔ The second half was added after a mutation sweep: making the `Stochastic` arm call
    /// `quantise_nearest` survived, because nothing checked that the dispatch reached the mode it
    /// was asked for. The seeded run must both *report* the mode and *differ* from the
    /// deterministic one.
    #[test]
    fn stochastic_rounding_needs_a_generator_and_then_actually_rounds_stochastically() {
        let w = [0.5, -0.25];
        assert_eq!(quantise(&w, 4, Rounding::Stochastic, None).unwrap_err(), CompressError::NeedsRng);
        let mut r = Rng::new(7);
        assert!(quantise(&w, 4, Rounding::Stochastic, Some(&mut r)).is_ok());

        // 64 weights placed a third of a step above a code, where the two modes disagree with
        // probability 1/3 each: the chance they agree everywhere is (2/3)^64, about 1e-12.
        let q = Quantiser::symmetric(6, 1.0).expect("valid");
        let w: Vec<f64> = (0..64).map(|k| (f64::from(k % 20) + 1.0 / 3.0) * q.step).collect();
        let near = quantise(&w, 6, Rounding::Nearest, None).expect("valid");
        let mut r = Rng::new(20260918);
        let stoch = quantise(&w, 6, Rounding::Stochastic, Some(&mut r)).expect("valid");
        assert_eq!(near.mode, Rounding::Nearest);
        assert_eq!(stoch.mode, Rounding::Stochastic, "the mode asked for is the mode reported");
        assert_ne!(stoch.codes, near.codes, "stochastic rounding produced the deterministic codes");
        // And it is deterministic under its seed, which is the whole reason the rng is required.
        let mut r2 = Rng::new(20260918);
        let again = quantise(&w, 6, Rounding::Stochastic, Some(&mut r2)).expect("valid");
        assert_eq!(stoch.codes, again.codes);
    }

    // ---- (e) footprint arithmetic --------------------------------------------------------------

    /// Requirement (e), against a hand calculation done in the doc comment below and nowhere else.
    ///
    /// 1000 parameters, 75% pruned to 250 survivors, 4-bit weights, 16-bit indices, plus 64
    /// neurons of state at 32 bits.
    ///
    /// - state: 64 * 32 = 2048 bits = 256 bytes.
    /// - Dense:   1000 * 4                = 4000 bits; +2048 = 6048 bits -> 756 bytes.
    /// - Sparse:  250 * (4 + 16)          = 5000 bits; +2048 = 7048 bits -> 881 bytes.
    /// - Bitmask: 1000 + 250 * 4 = 2000 bits; +2048 = 4048 bits -> 506 bytes.
    #[test]
    fn the_planned_footprint_matches_the_hand_calculation_at_the_width_the_quantiser_delivers() {
        // ⛔ This test was named "after quantisation" and never quantised anything. `Plan::weight_bits`
        // is now tied to the width `quantise` actually delivers, so the plan's arithmetic is
        // about the model that would ship rather than about an integer nobody checked.
        let q = quantise(&[0.5, -0.25, 1.0, 0.125], 4, Rounding::Nearest, None).expect("valid");
        let shape = ModelShape { parameters: 1000, state_values: 64, state_bits: 32 };
        let cases = [
            (Storage::Dense, 4000u64, 756u64),
            (Storage::Sparse { index_bits: 16 }, 5000, 881),
            (Storage::Bitmask, 2000, 506),
        ];
        for (storage, param_bits, bytes) in cases {
            let plan = Plan::new(0.75, 4, 16, storage).expect("valid plan");
            assert_eq!(plan.weight_bits, q.bits, "the plan's width is the quantiser's");
            assert_eq!(plan.kept(1000), 250);
            let f = plan.footprint(&shape).expect("valid");
            assert_eq!(f.parameter_bits, param_bits, "{storage:?}");
            assert_eq!(f.state_bits, 2048);
            assert_eq!(f.bytes(), Some(bytes), "{storage:?}");
        }
    }

    /// The dense path must agree with the crate's own `Footprint::new` exactly, so that
    /// constructing the struct directly cannot drift from `metrics`'s accounting.
    #[test]
    fn the_dense_footprint_agrees_with_the_metrics_constructor() {
        let shape = ModelShape { parameters: 4096, state_values: 512, state_bits: 16 };
        for bits in [2u32, 4, 8, 16, 31] {
            let plan = Plan::new(0.0, bits, 1, Storage::Dense).expect("valid");
            let mine = plan.footprint(&shape).expect("valid");
            let theirs = Footprint::new(4096, bits, 512, 16).expect("valid");
            assert_eq!(mine, theirs, "at {bits} bits");
        }
    }

    /// The uncomfortable arithmetic, stated as a test: a sparse layout at 50% sparsity is BIGGER
    /// than storing everything dense, and the break-even is a closed form.
    #[test]
    fn sparse_storage_of_a_lightly_pruned_model_is_larger_than_dense() {
        let shape = ModelShape { parameters: 1000, state_values: 1, state_bits: 1 };
        let dense = Plan::new(0.5, 4, 1, Storage::Dense)
            .expect("valid")
            .footprint(&shape)
            .expect("valid");
        let sparse = Plan::new(0.5, 4, 1, Storage::Sparse { index_bits: 16 })
            .expect("valid")
            .footprint(&shape)
            .expect("valid");
        assert!(
            sparse.parameter_bits > dense.parameter_bits,
            "sparse {} vs dense {}",
            sparse.parameter_bits,
            dense.parameter_bits
        );

        // The closed form says break-even is index_bits / (bits + index_bits) = 16/20 = 0.8.
        let be = Storage::Sparse { index_bits: 16 }.break_even_sparsity(4).expect("valid");
        assert!((be - 0.8).abs() < 1e-15);
        // And the byte counts agree with it: just under is a loss, just over is a win.
        let bits_at = |s: f64| {
            Plan::new(s, 4, 1, Storage::Sparse { index_bits: 16 })
                .expect("valid")
                .footprint(&shape)
                .expect("valid")
                .parameter_bits
        };
        assert!(bits_at(be - 0.01) > dense.parameter_bits);
        assert!(bits_at(be + 0.01) < dense.parameter_bits);
        // Bitmask breaks even at 1/bits = 0.25 and is a win well before sparse is.
        assert!((Storage::Bitmask.break_even_sparsity(4).expect("valid") - 0.25).abs() < 1e-15);
        assert!(Storage::Dense.break_even_sparsity(4).is_none());
    }

    // ---- (c) the 1/sqrt(T) law -----------------------------------------------------------------

    /// The fitter itself, on planted data with no noise in it at all. If this fails, every
    /// exponent this module reports is meaningless — so it is checked before the physics is.
    #[test]
    fn the_power_law_fitter_recovers_a_planted_exponent_exactly() {
        let ticks: Vec<u64> = (0..11).map(|k| 1u64 << k).collect();
        for planted in [-0.5f64, -1.0, -0.25, 0.5] {
            let err: Vec<f64> = ticks.iter().map(|&t| 3.0 * (t as f64).powf(planted)).collect();
            let got = fit_power_law(&ticks, &err).expect("eleven positive points");
            assert!((got - planted).abs() < 1e-12, "planted {planted}, fitted {got}");
        }
        assert!(fit_power_law(&[1, 2], &[1.0]).is_none(), "mismatched lengths refuse");
        assert!(fit_power_law(&[4], &[1.0]).is_none(), "a fit through one point is not a fit");
    }

    /// ⭐ REQUIREMENT (c). The rate-coding error falls as `T^(-1/2)`, measured over a sweep and
    /// fitted, against the analytic −0.5 that justifies the entire timestep lever.
    ///
    /// Three probabilities, so the exponent is shown to be independent of `p` — a fit at one `p`
    /// could be an accident of that `p`. 4000 trials per point gives an RMS estimate with about
    /// 1.1% relative noise, which over eight budgets spanning 8 to 1024 puts the standard error of
    /// the slope near 0.0024; the 0.03 window below is therefore about twelve of those, tight
    /// enough that a `T^(-1)` law misses it by about seventeen windows and a `T^(-1/4)` law by
    /// about eight.
    #[test]
    fn the_rate_coding_error_falls_as_one_over_root_t() {
        let ticks: Vec<u64> = (3..11).map(|k| 1u64 << k).collect();
        for (seed, p) in [(1u64, 0.5f64), (2, 0.2), (3, 0.05)] {
            let mut rng = Rng::new(seed);
            let sweep = rate_error_vs_ticks(p, &ticks, 4_000, &mut rng).expect("valid sweep");
            let q = sweep.fit_exponent().expect("eight positive points");
            assert!(
                (q + 0.5).abs() < 0.03,
                "p = {p}: fitted exponent {q}, analytic -0.5, errors {:?}",
                sweep.rms_error
            );
            // The mean absolute error obeys the same law, by the same argument.
            let qa = fit_power_law(&sweep.ticks, &sweep.mean_abs_error).expect("positive");
            assert!((qa + 0.5).abs() < 0.05, "p = {p}: mean-abs exponent {qa}");
            // ⛔ And `fit_exponent` fits the RMS, not the mean absolute error. Both obey the same
            // law, so swapping the two arrays survived a mutation sweep with every exponent
            // assertion above still green; only an identity check sees it.
            assert_eq!(
                q,
                fit_power_law(&sweep.ticks, &sweep.rms_error).expect("positive"),
                "fit_exponent must fit rms_error"
            );
            assert_ne!(q, qa, "the two measures are different numbers on the same run");
        }
    }

    /// The exponent is only half the law. This pins the CONSTANT against `sqrt(p(1-p)/T)`, which
    /// a fitted slope cannot see: an implementation that drew from `Binomial(T, 2p)` would keep
    /// the exponent and miss the constant by 40%.
    #[test]
    fn the_measured_rate_error_matches_the_binomial_closed_form() {
        let ticks = [16u64, 64, 256, 1024];
        for (seed, p) in [(11u64, 0.5f64), (12, 0.3), (13, 0.1)] {
            let mut rng = Rng::new(seed);
            let sweep = rate_error_vs_ticks(p, &ticks, 4_000, &mut rng).expect("valid");
            let want = sweep.predicted().expect("p and ticks are valid");
            for k in 0..ticks.len() {
                let rel = (sweep.rms_error[k] - want[k]).abs() / want[k];
                assert!(
                    rel < 0.08,
                    "p = {p}, T = {}: measured {} vs closed form {}",
                    ticks[k],
                    sweep.rms_error[k],
                    want[k]
                );
            }
        }
        // The degenerate ends are exact, not approximate: a certain outcome has no sampling error.
        assert_eq!(rate_rms(0.0, 10), Some(0.0));
        assert_eq!(rate_rms(1.0, 10), Some(0.0));
        assert_eq!(rate_rms(0.5, 100), Some(0.05), "sqrt(0.25/100) is exactly 0.05");
        assert!(rate_rms(0.5, 0).is_none());
        assert!(rate_rms(1.5, 10).is_none());
    }

    /// `ticks_for_rms` must return the SMALLEST budget that reaches the target. Both halves are
    /// asserted — that it reaches it, and that one tick fewer does not — which is what an
    /// off-by-one in the ceiling breaks.
    #[test]
    fn ticks_for_a_target_error_is_the_smallest_budget_that_reaches_it() {
        assert_eq!(ticks_for_rms(0.5, 0.05), Some(100), "0.25 / 0.0025");
        assert_eq!(ticks_for_rms(0.5, 0.025), Some(400), "four times the ticks to halve the error");
        // The case that caught the naive ceiling: the quotient evaluates to 225.00000000000003 and
        // `ceil` says 226, but 225 already meets the target. See the function's doc.
        assert_eq!(ticks_for_rms(0.1, 0.02), Some(225), "the last-place disagreement is resolved");
        // 199 probabilities x 5 targets, exhaustive in both directions and with NO tolerance.
        for k in 1..200u32 {
            let p = f64::from(k) / 200.0;
            for target in [0.1f64, 0.05, 0.02, 0.013, 0.007] {
                let t = ticks_for_rms(p, target).expect("valid");
                let got = rate_rms(p, t).expect("t >= 1");
                assert!(got <= target, "p {p} target {target}: got {got} at T={t}");
                if t > 1 {
                    let worse = rate_rms(p, t - 1).expect("t-1 >= 1");
                    assert!(worse > target, "T={t} was not minimal for p {p} target {target}");
                }
            }
        }
        // ⛔ THE OTHER DIRECTION: the naive ceiling UNDERSHOOTS. At these two inputs the ceiling's
        // own `rate_rms` is above the target by one part in 1e15, so the honest budget is one tick
        // more, and the increment loop — dead under every input above — is what supplies it.
        for &(p, target, naive) in &[(0.49f64, 7e-6f64, 5_100_000_000u64), (0.075, 1e-6, 69_375_000_000)] {
            let t = ticks_for_rms(p, target).expect("valid");
            assert_eq!(t, naive + 1, "p {p} target {target}");
            assert!(rate_rms(p, naive).expect("valid") > target, "the naive ceiling met the target after all");
            assert!(rate_rms(p, t).expect("valid") <= target);
        }
        assert_eq!(ticks_for_rms(0.0, 1e-9), Some(1), "a certain outcome needs one tick");
        assert!(ticks_for_rms(0.5, 0.0).is_none());
        assert!(ticks_for_rms(0.5, f64::NAN).is_none());
    }

    /// The trade, in one assertion: work is linear and error is a square root, and the two
    /// helpers agree with `rate_rms` rather than being an independent story.
    #[test]
    fn halving_the_budget_halves_the_work_and_multiplies_the_error_by_root_two() {
        assert_eq!(work_multiplier(100, 50), Some(0.5));
        assert_eq!(work_multiplier(100, 400), Some(4.0));
        let e = error_multiplier(100, 50).expect("non-zero");
        assert!((e - core::f64::consts::SQRT_2).abs() < 1e-15);
        // To HALVE the error you need four times the ticks — the asymmetry the doc claims.
        assert!((error_multiplier(100, 400).expect("non-zero") - 0.5).abs() < 1e-15);
        for p in [0.5f64, 0.2, 0.05] {
            let ratio = rate_rms(p, 50).expect("valid") / rate_rms(p, 100).expect("valid");
            assert!((ratio - e).abs() < 1e-12, "the two routes to the same number disagree");
        }
        assert!(error_multiplier(0, 10).is_none());
        assert!(work_multiplier(0, 10).is_none());
    }

    /// Scaling a ledger is integer-exact, and readout does NOT scale — a distinction that is the
    /// whole reason the function exists rather than a single multiply at the call site.
    #[test]
    fn scaling_a_ledger_to_fewer_ticks_leaves_the_readout_alone() {
        let base = Ledger {
            syn_ops: 1000,
            syn_fetches: 1000,
            neuron_updates_idle: 640,
            neuron_updates_driven: 360,
            spikes_out: 77,
            reads: 10,
        };
        let half = ledger_at_ticks(&base, 100, 50).expect("valid");
        assert_eq!(half.syn_ops, 500);
        assert_eq!(half.syn_fetches, 500);
        assert_eq!(half.neuron_updates_idle, 320);
        assert_eq!(half.neuron_updates_driven, 180);
        assert_eq!(half.spikes_out, 38, "77 * 50 / 100 truncates");
        assert_eq!(half.reads, 10, "one readout per inference, not per tick");
        // The idle fraction is a ratio and survives the scaling.
        assert!(
            (half.idle_fraction().expect("non-zero") - base.idle_fraction().expect("non-zero"))
                .abs()
                < 1e-12
        );
        assert!(ledger_at_ticks(&base, 0, 10).is_none());
        let huge = Ledger { syn_ops: u64::MAX, ..Ledger::default() };
        assert!(ledger_at_ticks(&huge, 1, 2).is_none(), "overflow refuses rather than wrapping");
    }

    // ---- pruning a Net, and what it does and does not buy ---------------------------------------

    /// The honest half of pruning, measured: zeroing a weight changes the ledger by NOTHING, and
    /// compacting the graph is what changes it. Both directions are asserted, so neither can be
    /// quietly dropped.
    #[test]
    fn zeroing_a_synapse_saves_nothing_until_the_graph_is_compacted() {
        // Neuron 0 drives 1, 2 and 3. The synapse to 3 is a hundredth the weight of the others.
        let mut b = NetBuilder::new(4);
        b.connect(0, 1, 20e-3, 1).expect("in range");
        b.connect(0, 2, 20e-3, 1).expect("in range");
        b.connect(0, 3, 0.2e-3, 1).expect("in range");
        let net = b.build();

        let (pruned, record) = prune_net(&net, 1.0 / 3.0).expect("valid");
        assert_eq!(record.n_removed, 1);
        assert_eq!(pruned.n_syn, 3, "the graph's shape is untouched");
        assert_eq!(pruned.w[2], 0.0, "CSR order is (pre, post), so 0 -> 3 is last");

        let compact = compact_net(&pruned).expect("well-formed net");
        assert_eq!(compact.n_syn, 2, "the zero is gone");
        assert_eq!(compact.n, 4, "the neuron is not");
        let survivors: Vec<(u32, f64, u32)> = compact.out_of(0).collect();
        assert_eq!(survivors, vec![(1, 20e-3, 1), (2, 20e-3, 1)]);

        let run = |n: crate::net::Net| {
            let mut ext = vec![0.0; 4];
            ext[0] = 3e-9;
            let mut s = Sim::new(n, vec![Lif::default(); 4], 1e-4, Mode::Clocked)
                .expect("four neurons, four models");
            s.run(2_000, &ext);
            s.ledger
        };
        let before = run(net);
        let zeroed = run(pruned);
        let after = run(compact);
        assert!(before.syn_ops > 0, "the network did something");
        assert_eq!(
            zeroed.syn_ops, before.syn_ops,
            "a zero weight is still delivered and still charged"
        );
        assert!(
            after.syn_ops < zeroed.syn_ops,
            "compacting removed deliveries: {} vs {}",
            after.syn_ops,
            zeroed.syn_ops
        );
        // Two of three synapses survive and the source neuron's rate is unchanged by a zero
        // weight, so the count falls by exactly a third.
        assert_eq!(after.syn_ops * 3, zeroed.syn_ops * 2);
    }

    // ---- distillation ---------------------------------------------------------------------------

    /// Softmax's two defining properties, and the limit that makes temperature mean anything.
    #[test]
    fn softmax_is_shift_invariant_and_sums_to_one() {
        let z = [2.0, -1.0, 0.5, 3.25];
        for t in [0.5f64, 1.0, 4.0, 100.0] {
            let a = softmax_t(&z, t).expect("valid");
            let sum: f64 = a.iter().sum();
            assert!((sum - 1.0).abs() < 1e-15, "T = {t} summed to {sum}");
            let shifted: Vec<f64> = z.iter().map(|x| x + 1000.0).collect();
            let b = softmax_t(&shifted, t).expect("valid");
            for k in 0..z.len() {
                assert!((a[k] - b[k]).abs() < 1e-12, "shift changed the distribution at T = {t}");
            }
        }
        // High temperature tends to uniform; low temperature tends to one-hot at the argmax.
        let hot = softmax_t(&z, 1e6).expect("valid");
        for v in &hot {
            assert!((v - 0.25).abs() < 1e-5, "T = 1e6 should be near-uniform, got {v}");
        }
        let cold = softmax_t(&z, 0.01).expect("valid");
        assert!(cold[3] > 1.0 - 1e-12, "T = 0.01 should be one-hot at the largest logit");
        assert!(softmax_t(&z, 0.0).is_err());
        assert!(softmax_t(&[], 1.0).is_err());
    }

    /// The divergence's own properties, since the loss is built on it.
    #[test]
    fn the_divergence_is_zero_on_itself_and_never_negative() {
        let p = softmax_t(&[1.0, 2.0, -0.5], 2.0).expect("valid");
        assert_eq!(kl_divergence(&p, &p).expect("valid"), 0.0, "exactly zero, not nearly");
        let mut rng = Rng::new(99);
        for _ in 0..200 {
            let a: Vec<f64> = (0..5).map(|_| rng.next_f64() * 6.0 - 3.0).collect();
            let b: Vec<f64> = (0..5).map(|_| rng.next_f64() * 6.0 - 3.0).collect();
            let pa = softmax_t(&a, 1.5).expect("valid");
            let pb = softmax_t(&b, 1.5).expect("valid");
            assert!(kl_divergence(&pa, &pb).expect("valid") >= 0.0, "Gibbs' inequality");
        }
        assert!(kl_divergence(&[0.5, 0.5], &[1.0, 0.0]).is_err(), "zero support refuses");
        assert!(kl_divergence(&[0.5, 0.4], &[0.5, 0.5]).is_err(), "not a distribution");
        assert!(cross_entropy(&[0.25, 0.75], 2).is_err(), "label past the end");
    }

    /// ⭐ The gradient against a central finite difference of the loss it is supposed to be the
    /// gradient OF. This is the check that catches a missing `1/T`, a missing `T²`, a sign flip,
    /// or a teacher and student swapped — none of which any of the other tests here would see.
    #[test]
    fn the_soft_target_gradient_matches_a_finite_difference_of_the_loss() {
        let teacher = [2.0, 0.5, -1.0, 0.25, 1.75];
        let student = [0.3, -0.2, 0.9, 1.1, -0.6];
        for t in [1.0f64, 2.0, 4.0, 10.0] {
            let d = Distillation::new(t, 1.0).expect("valid");
            let want = d.soft_gradient(&teacher, &student).expect("same length");
            let h = 1e-5;
            for i in 0..student.len() {
                let mut up = student;
                let mut down = student;
                up[i] += h;
                down[i] -= h;
                let lu = d.loss(&teacher, &up, 0).expect("valid").soft;
                let ld = d.loss(&teacher, &down, 0).expect("valid").soft;
                let fd = (lu - ld) / (2.0 * h);
                assert!(
                    (fd - want[i]).abs() < 1e-6 * want[i].abs().max(1e-3),
                    "T = {t}, logit {i}: closed form {} vs finite difference {fd}",
                    want[i]
                );
            }
            // The gradient of a soft-target term sums to zero: moving every logit together moves
            // no probability. That is a structural property and it is free to check.
            let s: f64 = want.iter().sum();
            assert!(s.abs() < 1e-12, "T = {t}: gradient summed to {s}");
        }
    }

    /// Hinton §2.1: at high temperature, and with both logit sets zero-meaned, distillation is
    /// equivalent to minimising the squared difference of the logits — so the gradient approaches
    /// `(z_i - v_i) / N`. This is the published limit, and it is the justification for the whole
    /// soft-target construction.
    #[test]
    fn at_high_temperature_distillation_reduces_to_matching_logits() {
        let zero_mean = |v: &[f64]| {
            let m = v.iter().sum::<f64>() / v.len() as f64;
            v.iter().map(|x| x - m).collect::<Vec<f64>>()
        };
        let teacher = zero_mean(&[2.0, 0.5, -1.0, 0.25, 1.75]);
        let student = zero_mean(&[0.3, -0.2, 0.9, 1.1, -0.6]);
        let n = teacher.len() as f64;
        let d = Distillation::new(200.0, 1.0).expect("valid");
        let got = d.soft_gradient(&teacher, &student).expect("same length");
        for i in 0..teacher.len() {
            let want = (student[i] - teacher[i]) / n;
            assert!(
                (got[i] - want).abs() < 0.01 * want.abs(),
                "logit {i}: got {}, high-temperature limit {want}",
                got[i]
            );
        }
        // And the approximation is an approximation: at T = 1 it is nowhere near.
        let cool = Distillation::new(1.0, 1.0).expect("valid");
        let near = cool.soft_gradient(&teacher, &student).expect("same length");
        let far = (0..teacher.len())
            .map(|i| (near[i] - (student[i] - teacher[i]) / n).abs())
            .fold(0.0f64, f64::max);
        assert!(far > 0.05, "the limit should NOT hold at T = 1, but the gap was only {far}");
    }

    /// Why the `T²` factor exists, measured: the raw divergence falls as `1/T²`, and the scaled
    /// loss does not. Without the factor, raising the temperature would silently switch the soft
    /// term off.
    #[test]
    fn the_temperature_squared_factor_keeps_the_soft_term_from_vanishing() {
        let teacher = [2.0, 0.5, -1.0, 0.25, 1.75];
        let student = [0.3, -0.2, 0.9, 1.1, -0.6];
        let soft_at = |t: f64| Distillation::new(t, 1.0).expect("valid")
            .loss(&teacher, &student, 0)
            .expect("valid")
            .soft;
        let raw_at = |t: f64| {
            let p = softmax_t(&teacher, t).expect("valid");
            let q = softmax_t(&student, t).expect("valid");
            kl_divergence(&p, &q).expect("valid")
        };
        let ratio_raw = raw_at(100.0) / raw_at(50.0);
        assert!((ratio_raw - 0.25).abs() < 0.02, "raw divergence should quarter, got {ratio_raw}");
        let ratio_scaled = soft_at(100.0) / soft_at(50.0);
        assert!((ratio_scaled - 1.0).abs() < 0.02, "scaled loss should hold, got {ratio_scaled}");
    }

    /// A student that is its own teacher pays exactly zero soft loss, and the blend weights are
    /// the blend weights.
    #[test]
    fn a_student_that_matches_its_teacher_pays_exactly_zero_soft_loss() {
        let z = [1.0, -0.5, 2.0];
        let d = Distillation::new(3.0, 0.7).expect("valid");
        let l = d.loss(&z, &z, 2).expect("valid");
        assert_eq!(l.soft, 0.0, "identical logits, exactly zero divergence");
        assert!(l.hard > 0.0, "but the hard term still charges for the label");
        assert!((l.total - 0.3 * l.hard).abs() < 1e-15);
        // ⛔ The hard term is at temperature 1, NOT at the objective's temperature. Computing it at
        // T = 3 survived a mutation sweep, because nothing pinned its value — only its sign and
        // its weight. This pins it against the independent computation, and against the wrong one.
        let want_hard = cross_entropy(&softmax_t(&z, 1.0).expect("valid"), 2).expect("valid");
        assert!((l.hard - want_hard).abs() < 1e-15, "hard term is at T = 1");
        let at_t = cross_entropy(&softmax_t(&z, 3.0).expect("valid"), 2).expect("valid");
        assert!((l.hard - at_t).abs() > 0.1, "and T = 3 is a visibly different number");

        let all_soft = Distillation::new(3.0, 1.0).expect("valid");
        let ls = all_soft.loss(&z, &[0.0, 0.0, 0.0], 2).expect("valid");
        assert_eq!(ls.total, ls.soft);
        let all_hard = Distillation::new(3.0, 0.0).expect("valid");
        let lh = all_hard.loss(&z, &[0.0, 0.0, 0.0], 2).expect("valid");
        assert_eq!(lh.total, lh.hard);
        assert!(Distillation::new(0.0, 0.5).is_err());
        assert!(Distillation::new(1.0, 1.5).is_err());
        assert!(d.loss(&z, &[1.0, 2.0], 0).is_err(), "length mismatch");
    }

    // ---- the Pareto front -----------------------------------------------------------------------

    fn point(bytes: u64, syn_ops: u64, error: f64) -> Point {
        Point {
            plan: Plan::new(0.0, 8, 1, Storage::Dense).expect("valid"),
            bytes,
            syn_ops,
            error,
        }
    }

    /// The front is characterised COMPLETELY, in both directions: everything returned is
    /// undominated and everything omitted is dominated by something. A stub returning every index
    /// fails the second half; a stub returning one index fails the first.
    ///
    /// ⛔ The reference predicate is spelled out here rather than calling [`Point::dominates`].
    /// Reusing that method would make this a check that `pareto_front` agrees with `dominates`,
    /// which it does by construction — any mutation of the rule would move both sides together
    /// and the test would stay green while reporting the wrong front.
    #[test]
    fn the_pareto_front_is_exactly_the_set_nothing_dominates() {
        // Spelled out, independently of the implementation: no worse on all three, better on one.
        let beats = |a: &Point, b: &Point| {
            a.bytes <= b.bytes
                && a.syn_ops <= b.syn_ops
                && a.error <= b.error
                && (a.bytes < b.bytes || a.syn_ops < b.syn_ops || a.error < b.error)
        };
        let mut rng = Rng::new(4242);
        for _ in 0..40 {
            let pts: Vec<Point> = (0..25)
                .map(|_| {
                    point(
                        u64::from(rng.below(6)) * 100,
                        u64::from(rng.below(6)) * 10,
                        f64::from(rng.below(6)) * 0.1,
                    )
                })
                .collect();
            let front = pareto_front(&pts).expect("non-empty, finite");
            assert!(!front.is_empty(), "some point is always undominated");
            assert!(front.windows(2).all(|w| w[0] < w[1]), "ascending");
            for i in 0..pts.len() {
                let dominated = pts.iter().enumerate().any(|(j, q)| j != i && beats(q, &pts[i]));
                assert_eq!(
                    front.contains(&i),
                    !dominated,
                    "point {i} = {:?} disagrees with the definition",
                    pts[i]
                );
                // And the method has to agree with the spelled-out rule on every pair.
                for (j, q) in pts.iter().enumerate() {
                    assert_eq!(q.dominates(&pts[i]), beats(q, &pts[i]), "pair ({j}, {i})");
                }
            }
        }
    }

    /// The hand cases the random sweep cannot pin: a strictly dominated point must go, ties must
    /// stay, and equal points do not eat each other.
    #[test]
    fn domination_is_strict_on_at_least_one_axis() {
        let a = point(100, 10, 0.1);
        let b = point(200, 20, 0.2); // strictly worse everywhere
        let c = point(100, 10, 0.1); // an exact duplicate of a
        let d = point(50, 40, 0.1); // cheaper in bytes, dearer in ops: incomparable
        assert!(a.dominates(&b));
        assert!(!b.dominates(&a));
        assert!(!a.dominates(&c), "equal on every axis is not domination");
        assert!(!a.dominates(&d));
        assert!(!d.dominates(&a));

        // ⛔ The case that separates the real rule from "strictly better on EVERY axis", which
        // survived a mutation sweep against every other assertion here: `e` ties `a` on bytes and
        // operations and is strictly better on error, so it dominates and `a` must leave the
        // front. Under the wrong rule neither dominates and both survive.
        let e = point(100, 10, 0.05);
        assert!(e.dominates(&a), "a tie on two axes and a win on the third IS domination");
        assert!(!a.dominates(&e));
        assert_eq!(pareto_front(&[a, e]).expect("valid"), vec![1]);
        // The same shape on each of the other two axes, so no single axis carries the rule.
        assert!(point(90, 10, 0.1).dominates(&a));
        assert!(point(100, 9, 0.1).dominates(&a));

        let front = pareto_front(&[a, b, c, d]).expect("valid");
        assert_eq!(front, vec![0, 2, 3], "only the strictly dominated point is removed");
        assert!(pareto_front(&[]).is_err());
        assert!(pareto_front(&[point(1, 1, f64::NAN)]).is_err(), "a NaN error refuses");
    }

    /// The budget rule: the lowest error that fits, deterministic ties, and a refusal naming both
    /// budgets when nothing fits.
    #[test]
    fn the_best_point_under_a_budget_is_the_lowest_error_that_fits() {
        let pts = [
            point(1000, 100, 0.01), // best error, too big
            point(400, 60, 0.05),   // fits
            point(400, 60, 0.05),   // fits, identical -> lowest index wins
            point(300, 90, 0.04),   // fits, better error
            point(200, 50, 0.20),   // fits, cheapest, worst error
        ];
        let k = best_under_budget(&pts, 500, 100).expect("something fits");
        assert_eq!(k, 3, "0.04 beats 0.05 and 0.20, and the 0.01 point does not fit");
        // Tighten the byte budget until only the cheapest survives.
        assert_eq!(best_under_budget(&pts, 250, 100).expect("one fits"), 4);
        // Tighten the operation budget instead: byte-feasible points can still be excluded.
        assert_eq!(best_under_budget(&pts, 500, 60).expect("one fits"), 1, "index ties go low");
        // ⛔ The documented tie-break, exercised. The pair above is equal on error, bytes AND ops,
        // so it pins the index rule and nothing else; `p.error < c.error` alone was green. Equal
        // error: fewer bytes wins. Equal error and bytes: fewer operations wins. Both are put in
        // the order that makes the answer the higher index, so generation order cannot be it.
        let tie = [point(900, 10, 0.05), point(100, 90, 0.05)];
        assert_eq!(best_under_budget(&tie, 1000, 100).expect("fits"), 1, "fewer bytes at equal error");
        let tie2 = [point(100, 90, 0.05), point(100, 10, 0.05)];
        assert_eq!(best_under_budget(&tie2, 1000, 100).expect("fits"), 1, "fewer ops at equal error and bytes");
        let err = best_under_budget(&pts, 10, 10).unwrap_err();
        assert_eq!(
            err,
            CompressError::NoFeasiblePoint { candidates: 5, max_bytes: 10, max_syn_ops: 10 }
        );
    }

    /// The end-to-end shape: build a sweep of plans, cost them exactly, price the rate-sampling
    /// error from the closed form, and read the front off. The assertions are the ones the
    /// construction guarantees — more ticks never cost more error, more sparsity never costs more
    /// bytes — so a wiring mistake between `Plan` and `Point` shows up here.
    #[test]
    fn a_plan_sweep_costs_out_and_produces_a_front() {
        let shape = ModelShape { parameters: 20_000, state_values: 256, state_bits: 16 };
        let mut pts = Vec::new();
        for &sparsity in &[0.0f64, 0.5, 0.9] {
            for &bits in &[8u32, 4, 2] {
                for &ticks in &[16u64, 64, 256] {
                    let plan =
                        Plan::new(sparsity, bits, ticks, Storage::Bitmask).expect("valid plan");
                    let err = rate_rms(0.5, ticks).expect("valid");
                    pts.push(Point::new(plan, &shape, 20_000, err).expect("valid"));
                }
            }
        }
        assert_eq!(pts.len(), 27);
        for p in &pts {
            let expect_ops = p.plan.kept(20_000) * p.plan.ticks;
            assert_eq!(p.syn_ops, expect_ops);
            assert_eq!(p.error, rate_rms(0.5, p.plan.ticks).expect("valid"));
        }
        let front = pareto_front(&pts).expect("valid");
        // The cheapest-and-most-accurate corner cannot be dominated: 90% sparse, 2-bit, 256 ticks
        // is the smallest and slowest-but-most-accurate combination present.
        let best_error = pts.iter().map(|p| p.error).fold(f64::INFINITY, f64::min);
        assert!(
            front.iter().any(|&i| pts[i].error == best_error),
            "the most accurate setting is always on the front"
        );
        let smallest = pts.iter().map(|p| p.bytes).min().expect("non-empty");
        assert!(front.iter().any(|&i| pts[i].bytes == smallest));
        // And the budget rule picks something that actually fits.
        let k = best_under_budget(&pts, 10_000, 1_000_000).expect("something fits");
        assert!(pts[k].bytes <= 10_000 && pts[k].syn_ops <= 1_000_000);
    }

    /// Every boundary refuses rather than guessing, and names what was wrong.
    #[test]
    fn every_boundary_refuses_rather_than_guessing() {
        assert_eq!(
            prune_magnitude(&[], 0.5).unwrap_err(),
            CompressError::Empty { what: "weights" }
        );
        assert_eq!(
            prune_magnitude(&[1.0, f64::NAN], 0.5).unwrap_err(),
            CompressError::NonFinite { what: "weights", index: 1 }
        );
        assert!(matches!(
            prune_magnitude(&[1.0], 1.5).unwrap_err(),
            CompressError::BadValue { .. }
        ));
        assert!(matches!(
            prune_below(&[1.0], -1.0).unwrap_err(),
            CompressError::BadValue { .. }
        ));
        assert_eq!(
            prune_structured(&[1.0, 2.0, 3.0], 2, 2, Unit::OutputNeuron, Saliency::L2, 0.5)
                .unwrap_err(),
            CompressError::BadShape { got: 3, want: 4 }
        );
        assert!(matches!(
            prune_structured(&[1.0], 0, 1, Unit::OutputNeuron, Saliency::L2, 0.5).unwrap_err(),
            CompressError::Empty { .. }
        ));
        assert!(matches!(Plan::new(0.5, 1, 1, Storage::Dense).unwrap_err(), CompressError::BadValue { .. }));
        assert!(matches!(Plan::new(0.5, 8, 0, Storage::Dense).unwrap_err(), CompressError::BadValue { .. }));
        assert!(matches!(
            Plan::new(0.5, 8, 1, Storage::Sparse { index_bits: 0 }).unwrap_err(),
            CompressError::BadValue { .. }
        ));
        let bad_shape = ModelShape { parameters: 0, state_values: 1, state_bits: 8 };
        assert!(bad_shape.validate().is_err());
        let wide = ModelShape { parameters: 1, state_values: 1, state_bits: 65 };
        assert!(wide.validate().is_err());
        assert!(Storage::Dense.parameter_bits(10, 20, 8).is_none(), "kept past total");
        assert!(Storage::Dense.parameter_bits(u64::MAX, 1, 64).is_none(), "overflow");
        assert!(rate_error_vs_ticks(0.5, &[], 10, &mut Rng::new(1)).is_err());
        assert!(rate_error_vs_ticks(0.5, &[4, 0], 10, &mut Rng::new(1)).is_err());
        assert!(rate_error_vs_ticks(0.5, &[4], 0, &mut Rng::new(1)).is_err());
        assert!(rate_error_vs_ticks(-0.1, &[4], 1, &mut Rng::new(1)).is_err());
        // Guards that were deletable with every test green.
        let shape = ModelShape { parameters: 1000, state_values: 64, state_bits: 32 };
        let plan = Plan::new(0.75, 4, 16, Storage::Dense).expect("valid plan");
        assert!(matches!(Point::new(plan, &shape, 100, -1.0), Err(CompressError::BadValue { .. })));
        assert!(matches!(Point::new(plan, &shape, 100, f64::NAN), Err(CompressError::BadValue { .. })));
        assert!(Storage::Dense.parameter_bits(10, 5, 65).is_none(), "a 65-bit width");
        assert!(Storage::Sparse { index_bits: 0 }.parameter_bits(10, 5, 8).is_none());
        assert!(Storage::Sparse { index_bits: 0 }.break_even_sparsity(4).is_none());
        assert!(Storage::Bitmask.break_even_sparsity(0).is_none());
        assert_eq!(Storage::Dense.break_even_sparsity(4), None, "dense cannot beat itself");
        let mut record = prune_magnitude(&[1.0, 2.0], 0.5).expect("valid");
        record.n_removed = 9;
        assert_eq!(record.n_kept(), 0, "a corrupted public record saturates rather than underflows");
        // ⛔ BOTH counts, not just the source. `error_multiplier`'s doc says `None` "if either
        // count is zero"; only a source of zero was tested, and a DESTINATION of zero divides by
        // zero and reports an infinite growth in error as an ordinary number.
        assert!(error_multiplier(10, 0).is_none(), "a destination budget of zero");
        assert!(error_multiplier(0, 0).is_none());
        assert!(work_multiplier(10, 0).is_some(), "but no work at all is a legal workload");
        // ⛔ `best_under_budget`'s `# Errors` names `NonFinite` "as `pareto_front`", and only
        // `pareto_front` was ever handed one. A NaN error compares false against every incumbent,
        // so the unguarded rule returns the NaN point whenever it is the first candidate that fits
        // — the single worst answer available, chosen and reported as the best.
        assert_eq!(
            best_under_budget(&[point(1, 1, f64::NAN)], 10, 10).unwrap_err(),
            CompressError::NonFinite { what: "candidate errors", index: 0 }
        );
        assert_eq!(
            best_under_budget(&[point(1, 1, 0.5), point(1, 1, f64::INFINITY)], 10, 10).unwrap_err(),
            CompressError::NonFinite { what: "candidate errors", index: 1 }
        );
        // And the Display impl names the numbers, for every variant this module defines and for
        // the two wrapped errors it can build here. This comment used to say "for every variant"
        // above two of fourteen.
        let every = [
            (CompressError::Empty { what: "weights" }, "weights"),
            (CompressError::NonFinite { what: "teacher logits", index: 7 }, "7"),
            (CompressError::BadValue { what: "sparsity, which must be in 0..=1", value: 1.5 }, "1.5"),
            (CompressError::LengthMismatch { a: 3, b: 5 }, "5"),
            (CompressError::BadShape { got: 3, want: 4 }, "4"),
            (CompressError::LabelOutOfRange { label: 9, classes: 4 }, "9"),
            (CompressError::NotADistribution { what: "teacher", sum: 1.25 }, "1.25"),
            (CompressError::ZeroSupport { what: "student", index: 2 }, "2"),
            (CompressError::NeedsRng, "reproducible"),
            (CompressError::Overflow { what: "parameter bits" }, "parameter bits"),
            (CompressError::NoFeasiblePoint { candidates: 5, max_bytes: 10, max_syn_ops: 11 }, "11"),
            (CompressError::Hardware(HardwareError::UnstatedSpec { part: "Darwin", field: "weight_bits" }), "weight_bits"),
            (CompressError::Metric(MetricError::Overflow { what: "dense ops" }), "dense ops"),
        ];
        for (e, needle) in every {
            let text = e.to_string();
            assert!(text.contains(needle), "{e:?} prints {text:?} without {needle:?}");
        }
    }

    /// ⛔ REACHABLE ZEROS. `softmax_t(&[400, 0, -400], 0.5)` underflows to two exact zeros, so the
    /// `p[i] == 0` guard in `kl_divergence` (without it: `0 * ln(0/q) = NaN`) and the zero-support
    /// refusal in `cross_entropy` (without it: `+inf`, which the error's own doc says must never
    /// reach an optimiser) are both on a path a low-temperature distillation takes. Deleting either
    /// was green. So was accepting a "distribution" with a negative entry that sums to one.
    #[test]
    fn the_zero_probability_guards_are_on_a_reachable_path() {
        let q = softmax_t(&[400.0, 0.0, -400.0], 0.5).expect("valid");
        assert_eq!(q, vec![1.0, 0.0, 0.0], "the softmax must underflow to exact zeros here");
        assert_eq!(cross_entropy(&q, 1), Err(CompressError::ZeroSupport { what: "q", index: 1 }));
        assert_eq!(cross_entropy(&q, 0), Ok(0.0));
        let uniform = [1.0 / 3.0; 3];
        let d = kl_divergence(&q, &uniform).expect("p's zeros contribute nothing");
        assert!((d - 3f64.ln()).abs() < 1e-12, "KL(delta || uniform) = ln 3, got {d}");
        assert_eq!(
            kl_divergence(&uniform, &q),
            Err(CompressError::ZeroSupport { what: "q", index: 1 }),
            "the other direction is a refusal, not an infinity"
        );
        assert!(matches!(
            kl_divergence(&[-0.5, 1.5], &[0.5, 0.5]),
            Err(CompressError::NotADistribution { what: "p", .. })
        ));
    }

    /// Determinism: the same seed gives the same sweep, on any platform.
    #[test]
    fn the_same_seed_gives_the_same_sweep() {
        let ticks = [8u64, 32, 128];
        let a = rate_error_vs_ticks(0.35, &ticks, 200, &mut Rng::new(5150)).expect("valid");
        let b = rate_error_vs_ticks(0.35, &ticks, 200, &mut Rng::new(5150)).expect("valid");
        assert_eq!(a, b, "same seed, same numbers");
        assert_eq!((a.trials, a.p, a.ticks.as_slice()), (200, 0.35, &ticks[..]), "the sweep records what it ran");
        let c = rate_error_vs_ticks(0.35, &ticks, 200, &mut Rng::new(5151)).expect("valid");
        assert_ne!(a.rms_error, c.rms_error, "a different seed is a different realisation");
    }

    /// `Pruned` is a record, and its derived fields have to agree with its own arrays. A record
    /// whose summary disagrees with its data is worse than no summary.
    #[test]
    fn the_pruning_record_agrees_with_its_own_arrays() {
        let w = [0.4, -0.1, 2.0, 0.0, -3.0, 0.05, 1.2, -0.7];
        for frac in [0.0f64, 0.125, 0.375, 0.625, 1.0] {
            let p: Pruned = prune_magnitude(&w, frac).expect("valid");
            assert_eq!(p.kept.iter().filter(|k| **k).count(), p.n_kept());
            assert_eq!(p.kept.len(), p.w.len());
            assert_eq!(p.n_removed + p.n_kept(), w.len());
            for i in 0..w.len() {
                if !p.kept[i] {
                    assert_eq!(p.w[i], 0.0);
                }
            }
            let newly: usize =
                (0..w.len()).filter(|&i| !p.kept[i] && w[i] != 0.0).count();
            assert_eq!(p.newly_zeroed, newly);
        }
        // ⛔ And the same loop over `prune_below`, whose record could contradict itself with
        // nothing noticing: `kept[i] = false` deleted gave n_removed 3, n_kept 3 and six `true`s.
        for threshold in [0.0f64, 0.05, 0.25, 1.0, 5.0] {
            let p: Pruned = prune_below(&w, threshold).expect("valid");
            assert_eq!(p.kept.iter().filter(|k| **k).count(), p.n_kept());
            assert_eq!(p.n_removed + p.n_kept(), w.len());
            for i in 0..w.len() {
                assert_eq!(p.kept[i], w[i].abs() >= threshold, "threshold {threshold} index {i}");
                if !p.kept[i] {
                    assert_eq!(p.w[i], 0.0);
                }
            }
            let newly: usize =
                (0..w.len()).filter(|&i| !p.kept[i] && w[i] != 0.0).count();
            assert_eq!(p.newly_zeroed, newly, "threshold {threshold}");
        }
    }

    // ---- non-finite inputs, at every boundary that promises to reject them ----------------------

    /// ⛔ NON-FINITE MEANS INFINITE TOO. [`CompressError::NonFinite`]'s doc says an element that is
    /// "`NaN` or infinite" is rejected at the boundary, and `all_finite` is the single place that
    /// promise is kept — but every fixture in this module reached it with `f64::NAN`, so narrowing
    /// the test to `x.is_nan()` passed the whole suite. An infinite weight then reaches
    /// [`Quantiser::from_weights`], whose scale is the largest magnitude present: the step becomes
    /// infinite, every finite weight rounds to code 0, and the round trip reports a perfect one.
    ///
    /// Every caller of that helper is swept, because each one names its own array and an index,
    /// and a guard that fires for the wrong array is not the guard the message claims.
    #[test]
    fn an_infinite_weight_is_refused_by_name_and_index_exactly_as_a_nan_is() {
        for bad in [f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(
                prune_magnitude(&[1.0, bad, 2.0], 0.5).unwrap_err(),
                CompressError::NonFinite { what: "weights", index: 1 },
                "magnitude pruning accepted {bad}"
            );
            assert_eq!(
                prune_below(&[bad, 1.0], 0.5).unwrap_err(),
                CompressError::NonFinite { what: "weights", index: 0 },
                "threshold pruning accepted {bad}"
            );
            assert_eq!(
                prune_structured(&[1.0, 2.0, bad, 4.0], 2, 2, Unit::OutputNeuron, Saliency::L2, 0.5)
                    .unwrap_err(),
                CompressError::NonFinite { what: "weights", index: 2 },
                "structured pruning accepted {bad}"
            );
            assert_eq!(
                softmax_t(&[0.0, 1.0, bad], 1.0).unwrap_err(),
                CompressError::NonFinite { what: "logits", index: 2 },
                "the softmax accepted {bad}"
            );
            // The distribution checks share the helper and each names its own argument.
            assert_eq!(
                kl_divergence(&[0.5, bad], &[0.5, 0.5]).unwrap_err(),
                CompressError::NonFinite { what: "p", index: 1 }
            );
            assert_eq!(
                kl_divergence(&[0.5, 0.5], &[bad, 0.5]).unwrap_err(),
                CompressError::NonFinite { what: "q", index: 0 }
            );
            assert_eq!(
                cross_entropy(&[bad, 1.0], 0).unwrap_err(),
                CompressError::NonFinite { what: "q", index: 0 }
            );
        }
        // And the place an infinite weight would have landed, which is why the guard is at the
        // boundary rather than downstream: the quantiser's scale is taken from the largest
        // magnitude in the vector it is handed.
        assert!(matches!(
            quantise(&[1.0, f64::INFINITY], 4, Rounding::Nearest, None).unwrap_err(),
            CompressError::Hardware(HardwareError::NonFiniteWeight { index: 1, .. })
        ));
    }

    /// ⛔ [`prune_below`]'s `# Errors` promises a refusal for a "negative or non-finite"
    /// threshold, and the only one ever passed was `-1.0` — which `threshold < 0.0` catches by
    /// itself. The two non-finite thresholds are the ones that do damage quietly: a `NaN` makes
    /// every `m < threshold` false, so the call reports a record for a network it did not prune,
    /// and an infinite one makes every comparison true, so it zeroes the whole vector and reports
    /// that as an achieved sparsity of 1.
    #[test]
    fn a_non_finite_threshold_is_refused_rather_than_keeping_or_erasing_everything() {
        let w = [0.1, 0.9, 0.2, 1.4, -0.05, -2.0];
        assert_eq!(
            prune_below(&w, f64::INFINITY).unwrap_err(),
            CompressError::BadValue {
                what: "threshold, which must be finite and non-negative",
                value: f64::INFINITY,
            }
        );
        // A `NaN` cannot be compared with `assert_eq!` — it is not equal to itself — so the
        // variant and the name are matched and the value is tested for what it is.
        match prune_below(&w, f64::NAN).unwrap_err() {
            CompressError::BadValue { what, value } => {
                assert_eq!(what, "threshold, which must be finite and non-negative");
                assert!(value.is_nan(), "the refusal must report the value it was given");
            }
            other => panic!("a NaN threshold was accepted: {other:?}"),
        }
        // The guard is the non-finite test and not a widened sign test: `-0.0` is not negative and
        // is still a legal threshold, and a finite one still prunes.
        assert!(prune_below(&w, -0.0).is_ok(), "negative zero is not a negative threshold");
        assert_eq!(prune_below(&w, 0.25).expect("finite and non-negative").n_removed, 3);
    }

    /// ⛔ [`Pruned::tied_at_cut`] is `0` by construction in [`prune_below`] — a strict `<` cut puts
    /// every removed magnitude below `threshold` and every kept one at or above it, so the two
    /// sets cannot share a magnitude. No test in this module read that field off a
    /// threshold-pruned record, so writing `n_removed` into it was green; and the field is
    /// precisely what tells a reader that the difference between two runs of the same network was
    /// an arbitrary index tie-break rather than a decision. Reporting every removal as a tie says
    /// the whole result was arbitrary.
    ///
    /// The fixture is deliberately full of duplicate magnitudes, so the same vector cut to a
    /// count by [`prune_magnitude`] DOES report a tie. The contrast is the point.
    #[test]
    fn threshold_pruning_reports_no_tie_because_a_strict_cut_cannot_have_one() {
        let w = [1.0, -1.0, 1.0, 2.0, 0.5, -0.5];
        for threshold in [0.25f64, 0.75, 1.5, 2.0, 3.0] {
            let p = prune_below(&w, threshold).expect("finite and non-negative");
            assert_eq!(p.tied_at_cut, 0, "threshold {threshold} reported a tie at the cut");
            assert!(p.separates_cleanly(), "threshold {threshold}");
            if let (Some(c), Some(s)) = (p.cut, p.smallest_kept) {
                assert!(c < threshold, "threshold {threshold}: the cut {c} is not below it");
                assert!(s >= threshold, "threshold {threshold}: the kept {s} is below it");
            }
        }
        // Those thresholds are not all vacuous: two of them remove a non-zero count, which is the
        // only case in which `n_removed` and `0` are different numbers.
        assert_eq!(prune_below(&w, 0.75).expect("valid").n_removed, 2);
        assert_eq!(prune_below(&w, 1.5).expect("valid").n_removed, 5);
        // The same vector cut by count instead: now the cut really does land inside a tie, three
        // weights share magnitude 1.0, and the record says so.
        let m = prune_magnitude(&w, 0.5).expect("valid");
        assert_eq!(m.n_removed, 3);
        assert_eq!(m.cut, Some(1.0));
        assert_eq!(m.smallest_kept, Some(1.0));
        assert_eq!(m.tied_at_cut, 3, "the magnitude cut lands inside a tie of three");
        assert!(!m.separates_cleanly());
    }

    /// ⛔ [`CompressError::BadShape`]'s doc says the refusal is for a matrix whose "declared shape
    /// is not its shape", and a matrix can miss in either direction — but only the short one was
    /// tested, and `w.len() < want` still catches a short matrix. One element too many is then
    /// accepted: every saliency is scored from the first `n_out * n_in` entries, the tail is
    /// copied into [`Structured::w`] untouched, and [`Structured::weight_sparsity`] divides its
    /// zero count by a length that is not the matrix.
    #[test]
    fn a_weight_matrix_longer_than_its_declared_shape_is_refused_like_a_short_one() {
        let short = [1.0f64, 2.0, 3.0];
        let exact = [1.0f64, 2.0, 3.0, 4.0];
        let long = [1.0f64, 2.0, 3.0, 4.0, 5.0];
        let prune = |m: &[f64]| prune_structured(m, 2, 2, Unit::OutputNeuron, Saliency::L2, 0.5);
        assert_eq!(prune(&short).unwrap_err(), CompressError::BadShape { got: 3, want: 4 });
        assert_eq!(prune(&long).unwrap_err(), CompressError::BadShape { got: 5, want: 4 });
        let s = prune(&exact).expect("two by two is four entries");
        assert_eq!(s.w.len(), 4, "the record's matrix is exactly the declared shape");
        assert_eq!(s.removed, vec![0], "row [1, 2] carries the smaller L2 norm");
        assert!((s.weight_sparsity() - 0.5).abs() < 1e-15, "two of four entries are zero");
    }

    /// ⭐⛔ THE SCALE IS THE WEIGHTS' OWN. [`quantise`]'s doc says the quantiser is "scaled to the
    /// vector's own largest magnitude", and that claim is the entire argument for
    /// prune-then-quantise being safe: pruning removes the smallest weights, so the maximum does
    /// not move and neither does the step. Every weight fixture in this module has a largest
    /// magnitude of exactly 1.0, so a fixed `Quantiser::symmetric(bits, 1.0)` produced a
    /// bit-identical step and identical codes on all of them — a normaliser that is 1 in every
    /// fixture makes the division by it invisible. These fixtures are scaled away from 1 in both
    /// directions.
    ///
    /// The step is `max |w| / (2^(bits-1) - 1)`. It is recomputed here by the same two operations
    /// in the same order, so the comparison is an equality rather than a tolerance.
    #[test]
    fn the_quantisers_scale_is_the_weights_own_largest_magnitude_and_not_a_fixed_unit() {
        // The largest magnitude of `base` is exactly 1.0, so the largest magnitude of
        // `base * scale` is exactly `scale` — one multiplication by one, which is exact.
        let base = [0.5f64, -1.0, 0.25, 0.125, -0.75];
        for scale in [0.25f64, 3.5, 1e-3, 64.0] {
            let w: Vec<f64> = base.iter().map(|x| x * scale).collect();
            for bits in [2u32, 4, 8, 16] {
                let q = quantise(&w, bits, Rounding::Nearest, None).expect("finite, non-empty");
                let max_code = (1i32 << (bits - 1)) - 1;
                assert_eq!(
                    q.step,
                    scale / f64::from(max_code),
                    "scale {scale} at {bits} bits: the step is not max |w| / (2^(bits-1) - 1)"
                );
                // Two consequences of a scale taken FROM the data, one on each side of it: the
                // largest magnitude reaches full scale, and nothing clips. A fixed unit scale
                // either leaves the top of the range unused or clamps weights into it.
                assert_eq!(q.codes[1], -max_code, "the -1.0 * scale entry is not full scale");
                assert_eq!(q.clipped, 0, "scale {scale} at {bits} bits clipped");
                assert!(q.within_half_lsb(1e-12), "scale {scale} at {bits} bits");
            }
        }
        // The order property, at a scale that is not 1: pruning takes the smallest weights, so it
        // cannot move the largest magnitude and the step before and after is the same number.
        let w: Vec<f64> = base.iter().map(|x| x * 3.5).collect();
        let p = prune_magnitude(&w, 0.6).expect("valid");
        assert_eq!(p.n_removed, 3);
        let before = quantise(&w, 6, Rounding::Nearest, None).expect("valid");
        let after = quantise(&p.w, 6, Rounding::Nearest, None).expect("valid");
        assert_eq!(after.step, before.step, "pruning moved the quantiser's scale");
        assert_eq!(after.step, 3.5 / 31.0, "max |w| = 3.5 over the 6-bit maximum code 31");
    }

    /// ⛔ THREE WIDTH RANGES, ALL TESTED AT THE TOP ONLY. [`Storage::parameter_bits`]'s doc says
    /// `None` "when a width is zero or past 64", [`ModelShape::validate`]'s says `state_bits` runs
    /// 1 to 64, and [`Plan::weight_bits`]'s says "2 to 31 — the range [`Quantiser`] accepts". The
    /// suite tested 65, 65 and 1, so deleting the two `== 0` tests and widening `2..=31` to
    /// `2..=32` were all green. A zero-bit weight charges nothing at all for every parameter in
    /// the model, and a 32-bit plan validates although no quantiser in this crate will execute it.
    ///
    /// The last block is what makes the `weight_bits` range a fact rather than a number: the plan
    /// and [`Quantiser::symmetric`] are asserted to accept and refuse the same widths, so the two
    /// cannot drift apart without this failing.
    #[test]
    fn the_stored_widths_are_refused_at_the_bottom_of_their_ranges_as_well_as_the_top() {
        for layout in [Storage::Dense, Storage::Sparse { index_bits: 16 }, Storage::Bitmask] {
            assert!(layout.parameter_bits(10, 5, 0).is_none(), "{layout:?} at zero bits");
            assert!(layout.parameter_bits(10, 5, 65).is_none(), "{layout:?} at 65 bits");
            assert!(layout.parameter_bits(10, 5, 1).is_some(), "{layout:?} at one bit");
            assert!(layout.parameter_bits(10, 5, 64).is_some(), "{layout:?} at 64 bits");
            assert!(layout.break_even_sparsity(0).is_none(), "{layout:?} at zero bits");
        }
        assert!(Storage::Sparse { index_bits: 65 }.parameter_bits(10, 5, 8).is_none());
        assert!(Storage::Sparse { index_bits: 65 }.break_even_sparsity(8).is_none());

        let shape = |b: u32| ModelShape { parameters: 1, state_values: 1, state_bits: b };
        assert!(matches!(
            shape(0).validate().unwrap_err(),
            CompressError::BadValue { what: "state_bits, which must be in 1..=64", .. }
        ));
        assert!(shape(1).validate().is_ok(), "one bit of state is a legal width");
        assert!(shape(64).validate().is_ok());
        assert!(shape(65).validate().is_err());

        for bits in [2u32, 31] {
            assert!(Plan::new(0.5, bits, 1, Storage::Dense).is_ok(), "the plan refused {bits}");
            assert!(Quantiser::symmetric(bits, 1.0).is_ok(), "the quantiser refused {bits}");
        }
        for bits in [0u32, 1, 32, 33, 64] {
            assert!(Plan::new(0.5, bits, 1, Storage::Dense).is_err(), "the plan accepted {bits}");
            assert!(
                Quantiser::symmetric(bits, 1.0).is_err(),
                "the quantiser accepted {bits}"
            );
        }
        // The consequence for the width the plan used to admit: it cannot be carried out.
        assert!(matches!(
            quantise(&[0.5, -1.0], 32, Rounding::Nearest, None).unwrap_err(),
            CompressError::Hardware(HardwareError::BadBits { bits: 32 })
        ));
    }

    /// ⭐⛔ A MEAN, NOT A SUM. [`RateSweep::mean_abs_error`]'s doc says "mean absolute decoding
    /// error at each budget", and the only thing any test read off that array was the exponent of
    /// a least-squares fit through its logarithm. A constant factor is an intercept in that
    /// regression and not a slope, so multiplying every entry by `trials` — four thousand of them
    /// — left the fitted exponent bit-identical and the suite green.
    ///
    /// The pin is a budget of one tick at `p = 0.5`, where the arithmetic is exact and does not
    /// depend on the draws at all: one Bernoulli trial gives a count of 0 or 1, so the error
    /// `count/1 - 0.5` is `+0.5` or `-0.5` on every trial whatever the seed. The mean absolute
    /// error is therefore exactly 0.5 and so is the RMS, and with a power-of-two trial count both
    /// accumulations are exact in binary — 64 halves sum to 32 and 64 quarters to 16. A sum over
    /// those 64 trials would report 32 instead of 0.5.
    #[test]
    fn the_mean_absolute_error_is_a_mean_over_trials_and_not_a_sum() {
        for seed in [1u64, 20260920, 7] {
            let s = rate_error_vs_ticks(0.5, &[1], 64, &mut Rng::new(seed)).expect("valid");
            assert_eq!(s.mean_abs_error, vec![0.5], "seed {seed}");
            assert_eq!(s.rms_error, vec![0.5], "seed {seed}");
            assert_eq!(s.trials, 64);
        }
        // Where the errors are not all equal, the mean absolute error is strictly below the RMS —
        // Cauchy-Schwarz over the same `trials` numbers, with equality only when every `|e|` is
        // identical. A sum over four thousand trials sits three orders of magnitude above it.
        let ticks = [8u64, 64, 512];
        let sweep = rate_error_vs_ticks(0.3, &ticks, 4_000, &mut Rng::new(31)).expect("valid");
        assert_eq!(sweep.mean_abs_error.len(), ticks.len());
        for k in 0..ticks.len() {
            assert!(
                sweep.mean_abs_error[k] > 0.0 && sweep.mean_abs_error[k] < sweep.rms_error[k],
                "T = {}: mean |e| {} against RMS {}",
                ticks[k],
                sweep.mean_abs_error[k],
                sweep.rms_error[k]
            );
        }
    }

    /// ⛔ THE SOFTMAX'S SUM GUARD IS NOT DEAD CODE. With the maximum subtracted first the largest
    /// term is `exp(0) = 1`, so the sum is at least 1 and at most the number of classes — which
    /// reads as though the `NotADistribution { what: "softmax output" }` branch can never be
    /// taken, and deleting it was green against every fixture here. It is reached through the
    /// temperature: a logit divided by a small enough temperature overflows to an infinity, the
    /// shift is then `inf - inf`, which is `NaN`, and the sum is `NaN`.
    ///
    /// Refusing is the honest answer rather than a defect to repair. Once two logits have both
    /// overflowed to `+inf` the ratio that decided the answer is gone: at this temperature
    /// `[1e10, 1e10]` and `[1e10, 1e10 - 1]` — a uniform distribution and a one-hot — are the same
    /// pair of infinities, so there is nothing left to return.
    #[test]
    fn a_temperature_small_enough_to_overflow_the_scaled_logits_is_refused() {
        let tiny = 1e-308f64;
        assert!(tiny.is_finite() && tiny > 0.0, "the temperature itself is a legal one");
        for logits in [[1e10f64, 0.0], [1e10, -1e10], [-1e10, -2e10]] {
            match softmax_t(&logits, tiny).unwrap_err() {
                CompressError::NotADistribution { what, sum } => {
                    assert_eq!(what, "softmax output");
                    assert!(sum.is_nan(), "{logits:?} summed to {sum}");
                }
                other => panic!("{logits:?} at a temperature of {tiny} gave {other:?}"),
            }
        }
        // The same logits at a temperature that does not overflow are an ordinary one-hot, so what
        // is being refused is the overflow and not the size of the logits.
        assert_eq!(
            softmax_t(&[1e10, 0.0], 1.0).expect("nothing overflows at T = 1"),
            vec![1.0, 0.0]
        );
    }

    /// ⛔ A `NaN` PASSES BOTH OF `check_distribution`'s OWN TESTS. `x < 0.0` is false for a `NaN`,
    /// and once the running sum has become a `NaN` so is `(sum - 1.0).abs() > 1e-9`, so the only
    /// thing that stops a `NaN` entry reaching the divergence is the `all_finite` call at the top
    /// — which no test exercised, because every non-distribution fixture in this module was a
    /// vector that summed to the wrong number or held a negative entry. Without it
    /// [`kl_divergence`] returns `Ok(NaN)`: a loss that is not a number, reported as a success,
    /// with the cause several stages upstream.
    #[test]
    fn a_distribution_with_a_nan_entry_is_refused_rather_than_summing_to_nan() {
        // The two tests a `NaN` walks past, computed off the fixture so the hole is named rather
        // than implied: it is not negative, and the sum it poisons is not far from one either.
        let poisoned = [0.5f64, f64::NAN];
        assert!(!(poisoned[1] < 0.0), "a NaN entry is not a negative entry");
        let sum: f64 = poisoned.iter().sum();
        assert!(!((sum - 1.0).abs() > 1e-9), "and a NaN sum is not a sum far from one");

        assert_eq!(
            kl_divergence(&[0.5, f64::NAN], &[0.5, 0.5]).unwrap_err(),
            CompressError::NonFinite { what: "p", index: 1 }
        );
        assert_eq!(
            kl_divergence(&[0.5, 0.5], &[f64::NAN, 0.5]).unwrap_err(),
            CompressError::NonFinite { what: "q", index: 0 }
        );
        assert_eq!(
            cross_entropy(&[0.25, 0.25, f64::NAN, 0.5], 0).unwrap_err(),
            CompressError::NonFinite { what: "q", index: 2 }
        );
        // The same call is what refuses an empty distribution, and it names the emptiness rather
        // than reporting a sum of zero against a tolerance.
        assert_eq!(kl_divergence(&[], &[]).unwrap_err(), CompressError::Empty { what: "p" });
    }

    /// ⛔ [`kl_divergence`]'s `# Errors` names [`CompressError::LengthMismatch`], but the only
    /// length-mismatched call in this module went through [`Distillation::loss`], which has a
    /// check of its own and never reaches this one. With the check gone the loop runs to
    /// `p.len()`: a shorter `p` is silently scored against a prefix of `q` — `[1.0]` against the
    /// first half of `[0.5, 0.5]` comes back as an entirely plausible 0.693 nats — and a longer
    /// one indexes `q` past its end.
    #[test]
    fn a_divergence_between_different_lengths_is_refused_by_kl_divergence_itself() {
        assert_eq!(
            kl_divergence(&[1.0], &[0.5, 0.5]).unwrap_err(),
            CompressError::LengthMismatch { a: 1, b: 2 }
        );
        assert_eq!(
            kl_divergence(&[0.5, 0.5], &[1.0]).unwrap_err(),
            CompressError::LengthMismatch { a: 2, b: 1 }
        );
        // Both arguments above are proper distributions on their own lengths, so nothing
        // downstream would have complained; and equal lengths still go through.
        assert_eq!(kl_divergence(&[0.5, 0.5], &[0.5, 0.5]).expect("same length"), 0.0);
    }

    /// ⭐⛔ THE HARD TERM IS THE STUDENT'S. [`DistillLoss::hard`]'s doc says
    /// `-ln(student_1[label])` — the student's own cross-entropy at temperature 1, which is the
    /// only term attaching the student to the true label at all. Every fixture in this module that
    /// inspected `hard` either passed the same vector as teacher and student, or set `alpha` to 0
    /// or 1 so that `total` was one term whatever the other held, so charging the TEACHER's
    /// cross-entropy instead was green. A label term that does not depend on the student has no
    /// gradient toward the label, and distillation would be soft targets and nothing else.
    ///
    /// Both terms are recomputed from the public functions, by the same operations in the same
    /// order, so the comparisons are equalities and not tolerances.
    #[test]
    fn the_hard_term_charges_the_students_cross_entropy_and_not_the_teachers() {
        let teacher = [4.0f64, 0.0, -1.0];
        let student = [0.0f64, 0.0, 0.0];
        let d = Distillation::new(3.0, 0.4).expect("valid");
        let l = d.loss(&teacher, &student, 2).expect("same length");

        let want_hard = cross_entropy(&softmax_t(&student, 1.0).expect("valid"), 2).expect("valid");
        assert_eq!(l.hard, want_hard, "the hard term is the student's own cross-entropy");
        assert!((want_hard - 3f64.ln()).abs() < 1e-15, "a flat student over three classes is ln 3");

        // The teacher's cross-entropy, at the same label and the same temperature, is a visibly
        // different number — which is what makes the assertion above able to fail.
        let teachers = cross_entropy(&softmax_t(&teacher, 1.0).expect("valid"), 2).expect("valid");
        assert!(teachers > want_hard + 3.0, "teacher {teachers} against student {want_hard}");

        // `alpha` is strictly inside the unit interval, so `total` carries both terms and a wrong
        // `hard` cannot hide behind a zero weight.
        let want_soft = 3.0
            * 3.0
            * kl_divergence(
                &softmax_t(&teacher, 3.0).expect("valid"),
                &softmax_t(&student, 3.0).expect("valid"),
            )
            .expect("valid");
        assert_eq!(l.soft, want_soft);
        assert_eq!(l.total, 0.4 * want_soft + 0.6 * want_hard);
        assert!(l.soft > 0.0 && l.hard > 0.0, "both terms are live in this fixture");
    }

    /// ⭐⛔ THE WORK IS THE DENSE TICK COST, NOT THE PARAMETER COUNT. [`Point::syn_ops`]'s doc says
    /// it comes from [`Plan::work`] applied to `syn_ops_per_dense_tick`. The one sweep fixture in
    /// this module set `shape.parameters` and `syn_ops_per_dense_tick` to the same 20,000, so the
    /// two arguments were indistinguishable and costing the work against the parameter count was
    /// green. They are different quantities — parameters are what a model STORES and the dense
    /// tick cost is what it SPENDS — and a network with shared weights, a recurrent loop or an
    /// input it only reads once has no reason to make them equal.
    ///
    /// The two are separated here and then asserted in opposite directions, so neither argument
    /// can be substituted for the other.
    #[test]
    fn a_points_work_is_the_dense_tick_cost_and_not_the_parameter_count() {
        let shape = ModelShape { parameters: 20_000, state_values: 256, state_bits: 16 };
        let dense_tick = 7_000u64;
        assert_ne!(dense_tick, shape.parameters, "the fixture has to separate the two");
        let rows = [(0.0f64, 7_000u64, 20_000u64), (0.5, 3_500, 10_000), (0.9, 700, 2_000)];
        for &(sparsity, survivors, kept_params) in &rows {
            for &ticks in &[1u64, 16, 256] {
                let plan = Plan::new(sparsity, 4, ticks, Storage::Bitmask).expect("valid plan");
                let p = Point::new(plan, &shape, dense_tick, 0.01).expect("valid");
                // Spelled out from the fixture rather than from `Plan::work`, so this is not a
                // check that the point agrees with the method it is built from.
                assert_eq!(p.syn_ops, survivors * ticks, "sparsity {sparsity}, {ticks} ticks");
                // And the bytes go the other way: the footprint is charged against the parameter
                // count, at one mask bit per position plus four bits per survivor.
                let f = plan.footprint(&shape).expect("valid");
                assert_eq!(f.parameter_bits, 20_000 + kept_params * 4, "sparsity {sparsity}");
                assert_eq!(p.bytes, f.bytes().expect("fits"));
            }
        }
    }

    /// ⛔ [`fit_power_law`]'s doc says `None` "on mismatched lengths", and the only mismatch tested
    /// was two budgets against one error — where [`crate::convert::ErrorCurve::fit_exponent`] zips
    /// the two arrays, is left with a single point, and refuses for a reason of its own. The guard
    /// was therefore deletable. The zip is exactly what makes a longer mismatch invisible: it
    /// truncates silently to the shorter array, so a caller who hands in budgets from one sweep
    /// and errors from another gets a confident exponent fitted through whatever prefix the two
    /// happen to share, reported as a fit over the sweep.
    #[test]
    fn a_fit_through_mismatched_arrays_is_refused_even_when_the_zip_would_succeed() {
        let ticks: Vec<u64> = (0..8).map(|k| 1u64 << k).collect();
        let error: Vec<f64> = ticks.iter().map(|&t| 3.0 * (t as f64).powf(-0.5)).collect();
        assert!(
            (fit_power_law(&ticks, &error).expect("eight positive points") + 0.5).abs() < 1e-12,
            "the matched fit is the control"
        );
        assert!(fit_power_law(&ticks, &error[..4]).is_none(), "eight budgets, four errors");
        assert!(fit_power_law(&ticks[..4], &error).is_none(), "four budgets, eight errors");
        // The regression underneath really would have answered, which is the whole hole: it sees
        // four usable points and fits them.
        let zipped = crate::convert::ErrorCurve {
            ticks: ticks.clone(),
            mean_abs_error: error[..4].to_vec(),
        }
        .fit_exponent();
        assert!(zipped.is_some(), "the underlying regression does not refuse this on its own");
    }
}
