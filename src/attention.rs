//! Spiking attention and spiking transformers — and an honest count of what is actually spiking.
//!
//! # What the mechanism is
//!
//! A standard transformer's attention layer computes `softmax(Q Kᵀ / √d) V`. Every term in it is
//! real-valued: the projections multiply real activations by real weights, the score matrix is real,
//! the exponential inside the softmax is real, and the weighted sum of values is real. On a
//! multiplier-array accelerator that is fine, and it is why transformers run on GPUs.
//!
//! Spikformer (Zhou, Zhu, He, Wang, Yan, Tian and Yuan, *Spikformer: When Spiking Neural Network
//! Meets Transformer*, ICLR 2023, `arXiv`:2209.15425) removes the softmax and makes `Q`, `K` and `V`
//! **spike tensors** — matrices whose every entry is exactly 0 or exactly 1, produced by putting each
//! projection's output through a leaky integrate-and-fire neuron. The attention becomes
//!
//! ```text
//! SSA(Q, K, V) = SN( s · (Q Kᵀ) V )
//! ```
//!
//! with `s` a fixed scalar and `SN` another layer of spiking neurons. There is no softmax, no
//! exponential and no division. The paper's argument is that when an operand is binary, the multiply
//! disappears: `1 · w` is `w` and `0 · w` is nothing, so a multiply-accumulate (`MAC`) collapses into
//! an **accumulate** (`AC`), and an accumulate is the cheaper primitive on every datapath anyone has
//! published a number for.
//!
//! # What it buys, and what it costs
//!
//! It buys the collapse above, at the sites where it really happens, and it buys sparsity: an
//! operand that is zero does no work at all, so a network firing at 5% density does 5% of the dense
//! projection work. Both are real.
//!
//! It costs accuracy per timestep — a binary `Q` carries one bit where a real `Q` carried thirty-two
//! — which is bought back by running the whole model for `T` timesteps and letting the membranes
//! integrate, so every operation count below is *per timestep* and the model's true bill is `T`
//! times it. And it costs the thing this module exists to measure: **the parts that did not become
//! accumulates**. Spikformer puts a `BatchNorm` after every linear layer, and the honest reckoning
//! of it cuts both ways: at inference the **gain folds into the preceding weight matrix for free**,
//! so it is not a multiply anyone pays, while the **shift becomes a bias that does not fold away**
//! and costs one add per output element per timestep — charged by [`Linear`] and switched on by
//! [`Spec::bias`]. The scalar `s` is a real multiply per
//! element. The membrane update inside every spiking neuron is a multiply, two adds and a comparison
//! per neuron per timestep. Spikformer's residual shortcut *adds two spike tensors*, which produces
//! values in `{0, 1, 2}` — no longer binary — so **every projection downstream of the first shortcut
//! stops being an accumulate**, which is the criticism Yao et al. make in *Spike-driven Transformer*
//! (`NeurIPS` 2023, `arXiv`:2307.01694) when they redesign the block to keep the stream binary.
//!
//! The headline efficiency figures in this literature are **operation counts multiplied by a
//! published joules-per-operation** — usually Horowitz's 45 nm 32-bit numbers from *Computing's
//! Energy Problem (and what we can do about it)*, ISSCC 2014: 0.9 pJ for an `AC` and 4.6 pJ for a
//! `MAC`. This module deliberately does **not** multiply anything by those constants. It reports
//! counts, because the counts are the part that can be checked; see [`crate::ledger`] for why this
//! crate refuses to convert a synaptic operation count into joules, and [`crate::crossover`] for the
//! published thresholds that decide whether the conversion could ever have favoured the spiking side.
//!
//! ⚠ **On the headline multiples.** Claims of the form "tens of times more efficient than the
//! equivalent transformer" in this line of work are *theoretical operation counts multiplied by a
//! published per-operation energy*, not measurements, and the `AC`/`MAC` split they rest on is
//! exactly what [`Audit`] recomputes. This review did not locate a metered comparison of a spiking
//! transformer against its dense equivalent on the same silicon. What it did locate is narrower and
//! checkable: the counts those multiples are built from have no slot for the adds, multiplies,
//! shifts, comparisons and gates that [`Audit::split`] counts separately — so the multiple is
//! computed over a subset of the arithmetic, whatever it is worth.
//!
//! # The audit is the point of this module
//!
//! [`Audit`] is what everything here exists to produce. Every arithmetic site in a forward pass
//! declares its operand **domains** ([`Domain`]), and the kind of operation follows from them by one
//! rule ([`OpKind::product`]): *if either operand is binary there is no multiplier, otherwise there
//! is*. The audit then reports the split — accumulates, multiply-accumulates, bare adds, bare
//! multiplies, shifts, comparisons and logic gates — with a dense count and an effective count for
//! each.
//!
//! Two fractions come out of it and they are not the same number:
//!
//! - [`Audit::synaptic_ac_fraction`] counts accumulates against accumulates plus multiply-accumulates
//!   — *the synaptic operations only*. This is the figure the literature reports, and for a
//!   well-formed spiking transformer it is 1.0.
//! - [`Audit::ac_fraction`] counts accumulates against **every operation the pass performed**.
//!
//! On the two-token toy model in this module's tests, hand-counted on paper, the first is `1.0` and
//! the second is `0.125`. The gap is not an error in either number. It is the arithmetic that the
//! first number does not have a slot for — and neither does `NeuroBench`'s [`crate::metrics::SynOps`],
//! which is why [`Audit::synops`] exists next to [`Audit::split`] and why their totals differ.
//!
//! ⚠ **Neither fraction is a property of an architecture on its own.** Both are ratios of
//! *effective* counts, so both move with how densely the model fires, and on a randomly
//! initialised model that is set by [`Spec::gain`] — a constant this crate invented and says so.
//! On the eight-token, sixteen-channel, two-block shape these tests use, with `IAND` shortcuts over
//! four timesteps, [`Audit::ac_fraction`] runs `0.3962` at a gain of 1, `0.4304` at 4, `0.4579` at
//! the default 6 and `0.4915` at 12, while [`Audit::synaptic_ac_fraction`] is `1.0` at every one of
//! them. Any single honest fraction quoted without its gain is a number nobody can reproduce, which
//! is why the test sweeps it instead of pinning one.
//!
//! # Units
//!
//! There is no ampere in a transformer. A learned weight matrix is dimensionless, so the activations
//! here are dimensionless and the neuron constants are the paper's own: `tau = 2.0`, `v_th = 1.0`,
//! `v_reset = 0.0`, kept verbatim where a reader can compare them to Spikformer's section 3.1 rather
//! than rescaled into volts they do not have. The one physical quantity is the duration of a
//! timestep, which is not a property of this model at all — it is a property of the deployment, and
//! it enters only when a caller hands an operation count to a ledger. That is stated here rather than
//! hidden in a constructor, because a dimensionless neuron in a crate whose other neurons take
//! seconds and amperes is a trap unless it is announced.
//!
//! For the same reason [`LifLayer`] deliberately does **not** implement [`crate::neuron::Neuron`]:
//! that trait's `step` takes `dt` in seconds and a current in amperes, and it carries
//! `EXACT_OVER_GAPS`, a claim about jumping a model across quiet ticks. Neither has a meaning for a
//! layer that is evaluated exactly once per timestep on activations with no unit. Implementing the
//! trait would require inventing a membrane resistance, and a wrong `EXACT_OVER_GAPS` is the most
//! dangerous constant in this crate.
//!
//! # What this module does not claim
//!
//! It is not a trained model and has no weights from any paper. It reproduces the *architecture* and
//! counts its arithmetic exactly; it does not reproduce any published accuracy, and this review did
//! not locate an open checkpoint it could have loaded. The positional encodings in [`Position`] are
//! partly this crate's construction and say so per variant.

use crate::metrics::SynOps;
use crate::rng::Rng;
use core::fmt;

// ---------------------------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------------------------

/// What went wrong, named specifically enough to fix.
///
/// `PartialEq` but not `Eq`: two variants carry the offending `f64` so that a caller can print it,
/// and a `NaN` payload is exactly the case where equality has no answer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AttnError {
    /// A tensor or weight matrix had the wrong number of elements.
    BadShape {
        /// Which array.
        what: &'static str,
        /// Elements supplied.
        got: usize,
        /// Elements required.
        want: usize,
    },
    /// A dimension was zero. A zero-token sequence and a zero-channel model are both refused
    /// rather than silently producing an empty audit that reads as "nothing cost anything".
    Empty {
        /// Which dimension.
        what: &'static str,
    },
    /// A `NaN` or an infinity reached a boundary. Rejected here because a non-finite activation
    /// propagates into every downstream spike time and then into the audit as a silently wrong
    /// effective count — a `NaN` weight is neither zero nor non-zero to the counting rule.
    NonFinite {
        /// Which array.
        what: &'static str,
        /// Index of the first offending element.
        index: usize,
    },
    /// A tensor declared [`Domain::Binary`] carried a value that is not exactly `0.0` or `1.0`.
    NotBinary {
        /// Which array.
        what: &'static str,
        /// Index of the first offending element.
        index: usize,
        /// The value found there.
        value: f64,
    },
    /// A tensor declared [`Domain::Integer`] carried a negative value, a non-integer, or one above
    /// its stated bound.
    OutOfDomain {
        /// Which array.
        what: &'static str,
        /// Index of the first offending element.
        index: usize,
        /// The value found there.
        value: f64,
        /// The inclusive upper bound the domain declared.
        bound: u32,
    },
    /// `d_model` was not divisible by the head count, so the heads would not tile the channels.
    HeadsDoNotDivide {
        /// Channels in the model.
        channels: usize,
        /// Heads requested.
        heads: usize,
    },
    /// A parameter was outside the range its model is defined on.
    BadParameter {
        /// Which parameter.
        what: &'static str,
        /// The value supplied.
        value: f64,
    },
    /// Folding the attention scale into the neuron's threshold was requested where it would not be
    /// bit-exact. See [`Ssa::new`] for the two conditions.
    UnfoldableScale {
        /// The scale that cannot be folded.
        scale: f64,
        /// Which condition failed.
        why: &'static str,
    },
    /// An operation requiring binary operands was given a wider one.
    NotSpikeDriven {
        /// Which operation.
        what: &'static str,
    },
    /// A count exceeded `u64`. Reported rather than wrapped, because a wrapped operation count is a
    /// small number where an enormous one belongs.
    Overflow {
        /// Which count.
        what: &'static str,
    },
    /// An internal invariant of the audit failed — an effective count exceeded its dense count.
    /// This is a bug in this module, surfaced rather than reported as a plausible figure.
    Inconsistent {
        /// Which site.
        what: &'static str,
    },
}

impl fmt::Display for AttnError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadShape { what, got, want } => {
                write!(f, "{what}: got {got} elements, need {want}")
            }
            Self::Empty { what } => write!(f, "{what} is zero"),
            Self::NonFinite { what, index } => {
                write!(f, "{what}[{index}] is not finite")
            }
            Self::NotBinary { what, index, value } => {
                write!(f, "{what}[{index}] = {value} is not a spike (0 or 1)")
            }
            Self::OutOfDomain { what, index, value, bound } => {
                write!(f, "{what}[{index}] = {value} is not an integer in 0..={bound}")
            }
            Self::HeadsDoNotDivide { channels, heads } => {
                write!(f, "{heads} heads do not divide {channels} channels")
            }
            Self::BadParameter { what, value } => write!(f, "{what} = {value} is out of range"),
            Self::UnfoldableScale { scale, why } => {
                write!(f, "scale {scale} cannot be folded into the threshold: {why}")
            }
            Self::NotSpikeDriven { what } => {
                write!(f, "{what} requires binary operands and was given wider ones")
            }
            Self::Overflow { what } => write!(f, "{what} exceeded u64"),
            Self::Inconsistent { what } => {
                write!(f, "audit invariant broken at {what}: effective exceeded dense")
            }
        }
    }
}

impl std::error::Error for AttnError {}

// ---------------------------------------------------------------------------------------------
// Domains and operation kinds
// ---------------------------------------------------------------------------------------------

/// What kind of values a tensor is **declared** to hold.
///
/// This is a static property of the datapath, not a measurement of one sample. A tensor whose values
/// all happened to be 0 and 1 on one input is not thereby binary, for the same reason
/// [`crate::metrics::ActivationKind`] is a declaration rather than an inference: hardware is built
/// for the bound, and a multiplier that was installed because the bound is 3 is still installed on
/// the day the data comes out binary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Domain {
    /// Every element is exactly `0.0` or exactly `1.0`. This is the domain that removes multipliers.
    Binary,
    /// Every element is a non-negative integer no greater than the bound, which is at least 2.
    ///
    /// Produced by [`Residual::SpikeAdd`]: adding two spike tensors gives `{0, 1, 2}`. A bound of 1
    /// is promoted to [`Domain::Binary`] at construction, so this variant never carries one.
    Integer(u32),
    /// Real-valued and finite. Needs a multiplier.
    Real,
}

impl Domain {
    /// Whether a product with this operand can skip the multiplier.
    #[must_use]
    pub fn is_binary(self) -> bool {
        matches!(self, Self::Binary)
    }

    /// The domain of an elementwise sum of two tensors: bounds add, and anything real stays real.
    ///
    /// Saturating on the bound, because a bound that wrapped would turn a wide operand back into a
    /// narrow one and remove multipliers from the audit that the hardware would still need.
    #[must_use]
    pub fn sum(self, other: Self) -> Self {
        match (self, other) {
            (Self::Real, _) | (_, Self::Real) => Self::Real,
            (Self::Binary, Self::Binary) => Self::Integer(2),
            (Self::Binary, Self::Integer(b)) | (Self::Integer(b), Self::Binary) => {
                Self::Integer(b.saturating_add(1))
            }
            (Self::Integer(a), Self::Integer(b)) => Self::Integer(a.saturating_add(b)),
        }
    }

    /// The inclusive upper bound this domain promises, or `None` for [`Domain::Real`].
    #[must_use]
    pub fn bound(self) -> Option<u32> {
        match self {
            Self::Binary => Some(1),
            Self::Integer(b) => Some(b),
            Self::Real => None,
        }
    }
}

impl fmt::Display for Domain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Binary => f.write_str("binary"),
            Self::Integer(b) => write!(f, "integer 0..={b}"),
            Self::Real => f.write_str("real"),
        }
    }
}

/// The arithmetic primitive one operation needs.
///
/// Ordered by nothing in particular; this crate declines to supply an energy ratio between them,
/// for the reason [`crate::metrics::SynOps::effective_total`] gives — the ratio depends on the
/// datapath and quoting one from a paper about a different datapath is how the field arrived at
/// figures that disagree by three orders of magnitude.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpKind {
    /// **Accumulate**: a product where at least one operand is binary, so the multiplier is not
    /// needed and the work is one addition into a running sum.
    Ac,
    /// **Multiply-accumulate**: a product where both operands are multi-valued. The primitive a
    /// spiking architecture claims to have removed.
    Mac,
    /// A bare addition with no product: a bias, a residual shortcut, a membrane update term.
    Add,
    /// A bare multiplication by a real scalar with no accumulation: an attention scale, a
    /// normalisation gain, a membrane leak factor that is not a power of two.
    Mul,
    /// A multiplication by an exact power of two — a shift on a fixed-point datapath, and exact in
    /// binary floating point. Cheaper than [`OpKind::Mul`] and not free.
    Shift,
    /// A threshold comparison: the spike test inside a neuron.
    Compare,
    /// A one-bit logical operation on binary operands — an `AND`, an `OR`, a negation.
    Logic,
}

impl OpKind {
    /// Every kind, in the order a report lists them.
    pub const ALL: [Self; 7] =
        [Self::Ac, Self::Mac, Self::Add, Self::Mul, Self::Shift, Self::Compare, Self::Logic];

    /// The kind of a product of operands drawn from two domains.
    ///
    /// **This one rule is the whole audit.** If either operand is binary the multiplier is
    /// unnecessary and the operation is an accumulate; otherwise it is a multiply-accumulate. A
    /// [`Domain::Integer`] operand gets no credit here even though a 2-valued operand could be done
    /// in two adds, because how much credit depends on the datapath and this crate does not supply
    /// that number.
    #[must_use]
    pub fn product(a: Domain, b: Domain) -> Self {
        if a.is_binary() || b.is_binary() { Self::Ac } else { Self::Mac }
    }

    /// Whether this kind occupies a hardware multiplier.
    #[must_use]
    pub fn needs_multiplier(self) -> bool {
        matches!(self, Self::Mac | Self::Mul)
    }

    /// Whether this kind is one of the two `NeuroBench` counts as a synaptic operation.
    #[must_use]
    pub fn is_synaptic(self) -> bool {
        matches!(self, Self::Ac | Self::Mac)
    }

    /// A short label for a report.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Ac => "AC",
            Self::Mac => "MAC",
            Self::Add => "add",
            Self::Mul => "mul",
            Self::Shift => "shift",
            Self::Compare => "compare",
            Self::Logic => "logic",
        }
    }
}

impl fmt::Display for OpKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// Whether `x` is a positive normal power of two, so a multiplication by it is a shift and is exact
/// in binary floating point.
///
/// Subnormal powers of two return `false`: they are powers of two, but a datapath that reaches them
/// has already lost precision and this predicate is used to license an exactness claim.
#[must_use]
pub fn is_exact_power_of_two(x: f64) -> bool {
    if !(x > 0.0) || !x.is_finite() {
        return false;
    }
    let bits = x.to_bits();
    let exponent = (bits >> 52) & 0x7ff;
    let mantissa = bits & ((1u64 << 52) - 1);
    // `mantissa == 0` alone already rejects every subnormal — a subnormal power of two is
    // `0.mantissa x 2^-1022` and so has exactly one mantissa bit set — and the `x > 0.0` guard
    // above has already taken the only zero-mantissa subnormal pattern, `+0.0`. The exponent test
    // is kept as the explicit statement of the documented rule rather than as the deciding clause.
    exponent != 0 && mantissa == 0
}

// ---------------------------------------------------------------------------------------------
// The audit
// ---------------------------------------------------------------------------------------------

/// One accounted arithmetic site, summed over every timestep the audit saw.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Site {
    /// Where in the model, dotted: `block0.attn.q_proj`. Sites with the same path and kind merge.
    pub path: String,
    /// What primitive this site needs.
    pub kind: OpKind,
    /// Operations a dense evaluation performs here — every operand pair, zero or not. This is the
    /// denominator that [`Site::effective`] is read against, and it is a property of the model
    /// rather than of the data, so it does not change when the input goes silent.
    pub dense: u64,
    /// Operations actually performed: those whose operands were all non-zero, so the accumulator
    /// changed.
    ///
    /// For a site with one real operand (a weight matrix) this is exact. For a site where **both**
    /// operands are activations — the `Q Kᵀ` product — it is a **lower bound** on the datapath's
    /// work, because a real implementation must at minimum enumerate one operand's set bits even
    /// where the other is zero. The upper bound is [`Site::dense`]; the truth is between them and
    /// depends on the dataflow.
    pub effective: u64,
    /// Why this site has this kind, in terms of its operands. Not a restatement of the kind: it
    /// names which operand was binary, or says that neither was.
    pub because: &'static str,
}

/// The operation split of a forward pass — this module's most valuable export.
///
/// Every field is a count of **effective** operations, summed over the timesteps the audit saw.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Split {
    /// Products with a binary operand: additions, no multiplier.
    pub accumulates: u64,
    /// Products with no binary operand: the multiplier is needed.
    pub multiply_accumulates: u64,
    /// Bare additions: biases, residual shortcuts, membrane updates.
    pub adds: u64,
    /// Bare multiplications by a real scalar.
    pub multiplies: u64,
    /// Multiplications by an exact power of two.
    pub shifts: u64,
    /// Threshold comparisons inside spiking neurons.
    pub comparisons: u64,
    /// One-bit logical operations.
    pub logic: u64,
}

impl Split {
    /// Every counted operation added together, or `None` if the sum does not fit in a `u64`.
    ///
    /// Adding an accumulate to a comparison is something a caller does deliberately, which is why
    /// this is a method rather than a field.
    #[must_use]
    pub fn total(&self) -> Option<u64> {
        [
            self.accumulates,
            self.multiply_accumulates,
            self.adds,
            self.multiplies,
            self.shifts,
            self.comparisons,
            self.logic,
        ]
        .iter()
        .try_fold(0u64, |a, b| a.checked_add(*b))
    }
}

/// Every arithmetic site a forward pass touched, with its dense and effective counts.
///
/// Built by passing `&mut Audit` through a forward pass. Sites merge by path and kind, so running
/// `T` timesteps through the same model accumulates into the same rows and
/// [`Audit::timesteps`] says how many.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Audit {
    sites: Vec<Site>,
    timesteps: u64,
}

impl Audit {
    /// An audit that has seen nothing.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record `effective` of `dense` operations of one kind at one site.
    ///
    /// Public so a caller can account for a layer this module does not implement and have it appear
    /// in the same report.
    ///
    /// # Errors
    ///
    /// [`AttnError::Inconsistent`] if `effective > dense`, which is a counting bug rather than a
    /// property of any workload; [`AttnError::Overflow`] if a running count exceeds `u64`.
    pub fn charge(
        &mut self,
        path: &str,
        kind: OpKind,
        because: &'static str,
        dense: u64,
        effective: u64,
    ) -> Result<(), AttnError> {
        if effective > dense {
            return Err(AttnError::Inconsistent { what: because });
        }
        if let Some(s) = self.sites.iter_mut().find(|s| s.path == path && s.kind == kind) {
            s.dense = s.dense.checked_add(dense).ok_or(AttnError::Overflow { what: "dense" })?;
            s.effective = s
                .effective
                .checked_add(effective)
                .ok_or(AttnError::Overflow { what: "effective" })?;
            return Ok(());
        }
        self.sites.push(Site { path: path.to_string(), kind, dense, effective, because });
        Ok(())
    }

    /// Note that one more timestep has been evaluated.
    pub fn tick(&mut self) {
        self.timesteps = self.timesteps.saturating_add(1);
    }

    /// Timesteps recorded. Counts here are totals over all of them, not per-timestep figures.
    #[must_use]
    pub fn timesteps(&self) -> u64 {
        self.timesteps
    }

    /// Every site, in the order they were first charged.
    #[must_use]
    pub fn sites(&self) -> &[Site] {
        &self.sites
    }

    /// The site at `path` with kind `kind`, if the pass touched one.
    #[must_use]
    pub fn site(&self, path: &str, kind: OpKind) -> Option<&Site> {
        self.sites.iter().find(|s| s.path == path && s.kind == kind)
    }

    /// Effective operations of one kind, summed over every site carrying it.
    ///
    /// **Saturating.** A total past `u64::MAX` is reported as `u64::MAX` rather than wrapped or
    /// panicked on. [`Audit::charge`] already refuses a single site that overflows, but two sites
    /// that each fit can still sum past the end, and a plain `sum()` here answered that by
    /// panicking in a debug build and **wrapping in a release one** — reporting an enormous count
    /// as a small one, which is the failure [`crate::metrics::MetricError`]'s overflow variant
    /// exists to prevent. [`Split::total`] still refuses rather than saturates, so a caller who
    /// needs the distinction has it.
    #[must_use]
    pub fn effective_of(&self, kind: OpKind) -> u64 {
        self.sites
            .iter()
            .filter(|s| s.kind == kind)
            .fold(0u64, |a, s| a.saturating_add(s.effective))
    }

    /// Dense operations of one kind, summed over every site carrying it. Saturating, for the
    /// reason [`Audit::effective_of`] gives.
    #[must_use]
    pub fn dense_of(&self, kind: OpKind) -> u64 {
        self.sites.iter().filter(|s| s.kind == kind).fold(0u64, |a, s| a.saturating_add(s.dense))
    }

    /// The split: what this pass actually did, by primitive.
    #[must_use]
    pub fn split(&self) -> Split {
        Split {
            accumulates: self.effective_of(OpKind::Ac),
            multiply_accumulates: self.effective_of(OpKind::Mac),
            adds: self.effective_of(OpKind::Add),
            multiplies: self.effective_of(OpKind::Mul),
            shifts: self.effective_of(OpKind::Shift),
            comparisons: self.effective_of(OpKind::Compare),
            logic: self.effective_of(OpKind::Logic),
        }
    }

    /// Every effective operation the pass performed, or `None` on overflow.
    #[must_use]
    pub fn effective_total(&self) -> Option<u64> {
        self.split().total()
    }

    /// **The honest fraction**: accumulates as a share of *every* operation the pass performed.
    ///
    /// `None` when the pass performed no operations at all, which is a different statement from a
    /// fraction of zero — **or** when the total does not fit in a `u64`, which [`Split::total`]
    /// refuses rather than saturating. A caller who needs to tell those apart asks
    /// [`Audit::effective_total`], and a printed report names which one it hit.
    #[must_use]
    pub fn ac_fraction(&self) -> Option<f64> {
        let total = self.effective_total()?;
        if total == 0 {
            return None;
        }
        Some(self.effective_of(OpKind::Ac) as f64 / total as f64)
    }

    /// **The reported fraction**: accumulates as a share of synaptic operations only — accumulates
    /// plus multiply-accumulates, ignoring every add, multiply, shift, comparison and gate.
    ///
    /// This is the quantity a "fully spike-driven" claim is about, and for a well-formed spiking
    /// transformer it is `1.0` while [`Audit::ac_fraction`] is not. Reported side by side rather
    /// than instead, because a reader needs to be able to reproduce the published figure to argue
    /// with it.
    ///
    /// `None` when the pass performed no synaptic operations, so that an all-silent run does not
    /// report perfect spike-drivenness, or when accumulates plus multiply-accumulates does not fit
    /// in a `u64`.
    ///
    /// ⚠ **It is a ratio of effective counts, so it is conditional on the data.** A model that
    /// contains live multiply-accumulate layers reports `Some(1.0)` on any input that happens to
    /// leave every one of them silent — a one-block `SpikeAdd` model with a zeroed `MLP` does
    /// exactly that with `dense_of(OpKind::Mac) == 8`, and this module's tests pin that case. The
    /// architectural claim "fully spike-driven" is `dense_of(OpKind::Mac) == 0`; this is not it,
    /// and a report that quotes one without the other is quoting the flattering half.
    #[must_use]
    pub fn synaptic_ac_fraction(&self) -> Option<f64> {
        let ac = self.effective_of(OpKind::Ac);
        let mac = self.effective_of(OpKind::Mac);
        let syn = ac.checked_add(mac)?;
        if syn == 0 {
            return None;
        }
        Some(ac as f64 / syn as f64)
    }

    /// Effective operations that occupy a hardware multiplier: multiply-accumulates plus bare
    /// multiplies. Shifts are excluded and counted on their own.
    ///
    /// Which kinds those are is [`OpKind::needs_multiplier`] and nothing else — the predicate is
    /// the single source of truth for "occupies a multiplier", rather than a list repeated here
    /// that could drift away from it.
    ///
    /// Saturating rather than wrapping. A workload reaching `u64::MAX` multiplies is not one this
    /// crate can simulate, but a wrapped count would report an enormous number as a small one —
    /// the failure mode [`crate::metrics::MetricError`]'s overflow variant exists to prevent.
    #[must_use]
    pub fn multiplier_ops(&self) -> u64 {
        OpKind::ALL
            .iter()
            .filter(|k| k.needs_multiplier())
            .fold(0u64, |a, k| a.saturating_add(self.effective_of(*k)))
    }

    /// This pass expressed in `NeuroBench`'s [`crate::metrics::SynOps`].
    ///
    /// **Lossy, deliberately, and that is the finding.** `SynOps` has exactly three slots — dense,
    /// effective `MACs`, effective `ACs` — so the adds, multiplies, shifts, comparisons and gates
    /// that [`Audit::split`] counts have nowhere to go. Comparing `synops().effective_total()` with
    /// [`Audit::effective_total`] measures how much of a spiking transformer's arithmetic the
    /// field's own benchmark metric cannot see.
    ///
    /// The `dense` denominator is **every synaptic site of the model** — dense accumulates plus
    /// dense multiply-accumulates, whether or not the data made them fire. Which kinds count as
    /// synaptic is [`OpKind::is_synaptic`] and nothing else, so the predicate and this sum cannot
    /// drift apart. Saturating, as [`Audit::dense_of`].
    #[must_use]
    pub fn synops(&self) -> SynOps {
        let dense = OpKind::ALL
            .iter()
            .filter(|k| k.is_synaptic())
            .fold(0u64, |a, k| a.saturating_add(self.dense_of(*k)));
        SynOps {
            dense,
            effective_macs: self.effective_of(OpKind::Mac),
            effective_acs: self.effective_of(OpKind::Ac),
        }
    }
}

impl fmt::Display for Audit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "operation audit over {} timestep(s)", self.timesteps)?;
        for kind in OpKind::ALL {
            let dense = self.dense_of(kind);
            let eff = self.effective_of(kind);
            if dense == 0 {
                continue;
            }
            writeln!(f, "  {:<8} dense {dense:>12}  effective {eff:>12}", kind.label())?;
        }
        // `None` from either fraction means one of two different things, and a report that called
        // an unrepresentable total "nothing ran" would be the same class of error as a wrapped
        // count: a large number printed as a small statement.
        let synaptic = self.effective_of(OpKind::Ac).checked_add(self.effective_of(OpKind::Mac));
        match (self.synaptic_ac_fraction(), synaptic) {
            (Some(x), _) => writeln!(f, "  AC share of synaptic operations (as reported): {x:.4}")?,
            (None, None) => {
                writeln!(f, "  AC share of synaptic operations: the total does not fit in u64")?;
            }
            (None, Some(_)) => {
                writeln!(f, "  AC share of synaptic operations: no synaptic operations")?;
            }
        }
        match (self.ac_fraction(), self.effective_total()) {
            (Some(x), _) => write!(f, "  AC share of ALL operations: {x:.4}"),
            (None, None) => write!(f, "  AC share of ALL operations: the total does not fit in u64"),
            (None, Some(_)) => write!(f, "  AC share of ALL operations: nothing ran"),
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Tensors
// ---------------------------------------------------------------------------------------------

/// A `tokens × channels` activation matrix with a declared [`Domain`], row-major by token.
///
/// One type for spikes and for real values, because the audit's rule is about the *declared*
/// domain and separating the two into distinct types would put the declaration in the type system
/// where a residual shortcut cannot widen it at runtime — which is exactly what a residual shortcut
/// does.
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    tokens: usize,
    channels: usize,
    values: Vec<f64>,
    domain: Domain,
}

impl Tensor {
    /// Build from values with an explicit domain, validating every element against it.
    ///
    /// A [`Domain::Integer`] bound of 0 or 1 is promoted to [`Domain::Binary`], so no `Integer`
    /// tensor ever carries a bound a binary datapath would have handled.
    ///
    /// # Errors
    ///
    /// [`AttnError::Empty`] for a zero dimension; [`AttnError::BadShape`] if `values.len()` is not
    /// `tokens * channels`; [`AttnError::NonFinite`], [`AttnError::NotBinary`] or
    /// [`AttnError::OutOfDomain`] for an element the domain does not admit.
    pub fn new(
        tokens: usize,
        channels: usize,
        values: Vec<f64>,
        domain: Domain,
    ) -> Result<Self, AttnError> {
        if tokens == 0 {
            return Err(AttnError::Empty { what: "tokens" });
        }
        if channels == 0 {
            return Err(AttnError::Empty { what: "channels" });
        }
        let want = tokens.checked_mul(channels).ok_or(AttnError::Overflow { what: "tensor" })?;
        if values.len() != want {
            return Err(AttnError::BadShape { what: "tensor", got: values.len(), want });
        }
        for (i, v) in values.iter().enumerate() {
            if !v.is_finite() {
                return Err(AttnError::NonFinite { what: "tensor", index: i });
            }
        }
        let domain = match domain {
            Domain::Integer(b) if b <= 1 => Domain::Binary,
            d => d,
        };
        match domain {
            Domain::Binary => {
                for (i, v) in values.iter().enumerate() {
                    if *v != 0.0 && *v != 1.0 {
                        return Err(AttnError::NotBinary { what: "tensor", index: i, value: *v });
                    }
                }
            }
            Domain::Integer(bound) => {
                for (i, v) in values.iter().enumerate() {
                    if *v < 0.0 || *v > f64::from(bound) || v.fract() != 0.0 {
                        return Err(AttnError::OutOfDomain {
                            what: "tensor",
                            index: i,
                            value: *v,
                            bound,
                        });
                    }
                }
            }
            Domain::Real => {}
        }
        Ok(Self { tokens, channels, values, domain })
    }

    /// A real-valued tensor. Every element must be finite.
    ///
    /// # Errors
    ///
    /// As [`Tensor::new`] with [`Domain::Real`].
    pub fn real(tokens: usize, channels: usize, values: Vec<f64>) -> Result<Self, AttnError> {
        Self::new(tokens, channels, values, Domain::Real)
    }

    /// A spike tensor from a bit pattern, row-major by token.
    ///
    /// # Errors
    ///
    /// As [`Tensor::new`] with [`Domain::Binary`].
    pub fn spikes(tokens: usize, channels: usize, bits: &[bool]) -> Result<Self, AttnError> {
        let values = bits.iter().map(|b| f64::from(u8::from(*b))).collect();
        Self::new(tokens, channels, values, Domain::Binary)
    }

    /// A silent spike tensor.
    ///
    /// # Errors
    ///
    /// [`AttnError::Empty`] for a zero dimension; [`AttnError::Overflow`] if `tokens * channels`
    /// does not fit in a `usize`. The shape is checked **before** the element count is formed, so
    /// this refuses where an unchecked `tokens * channels` would have panicked in the allocation —
    /// [`Tensor::new`] has always refused it and these constructors now agree.
    pub fn silent(tokens: usize, channels: usize) -> Result<Self, AttnError> {
        if tokens == 0 {
            return Err(AttnError::Empty { what: "tokens" });
        }
        if channels == 0 {
            return Err(AttnError::Empty { what: "channels" });
        }
        let n = tokens.checked_mul(channels).ok_or(AttnError::Overflow { what: "tensor" })?;
        Self::new(tokens, channels, vec![0.0; n], Domain::Binary)
    }

    /// Positional spike table from the sinusoidal encoding of Vaswani et al., *Attention Is All You
    /// Need*, `NeurIPS` 2017, thresholded at zero: channel `2i` is `sin(pos / 10000^(2i/d))`,
    /// channel `2i+1` is the matching cosine, and a spike is emitted where the value is `>= 0`.
    ///
    /// **The thresholding is this crate's construction, not a published encoding.** This review did
    /// not locate a standard spiking positional code for a 1-D sequence; Spikformer replaces
    /// absolute positions with a convolution over neighbours instead, which is
    /// [`Position::Conditional`]. What this variant has going for it is that it is deterministic,
    /// parameter-free and checkable against the closed form — row 0 is all ones, because
    /// `sin(0) = 0 >= 0` and `cos(0) = 1`.
    ///
    /// # Errors
    ///
    /// [`AttnError::Empty`] for a zero dimension; [`AttnError::Overflow`] if `tokens * channels`
    /// does not fit in a `usize`, refused before the capacity is reserved rather than panicking
    /// in the multiplication.
    pub fn sinusoidal_spikes(tokens: usize, channels: usize) -> Result<Self, AttnError> {
        if tokens == 0 {
            return Err(AttnError::Empty { what: "tokens" });
        }
        if channels == 0 {
            return Err(AttnError::Empty { what: "channels" });
        }
        let n = tokens.checked_mul(channels).ok_or(AttnError::Overflow { what: "tensor" })?;
        let d = channels as f64;
        let mut bits = Vec::with_capacity(n);
        for pos in 0..tokens {
            for c in 0..channels {
                let pair = (c / 2) as f64;
                let arg = pos as f64 / 10000f64.powf(2.0 * pair / d);
                let value = if c % 2 == 0 { arg.sin() } else { arg.cos() };
                bits.push(value >= 0.0);
            }
        }
        Self::spikes(tokens, channels, &bits)
    }

    /// Tokens (sequence positions).
    #[must_use]
    pub fn tokens(&self) -> usize {
        self.tokens
    }

    /// Channels per token.
    #[must_use]
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// The declared domain, which is what decides every operation kind downstream.
    #[must_use]
    pub fn domain(&self) -> Domain {
        self.domain
    }

    /// Every value, row-major by token.
    #[must_use]
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// One token's channels, or `None` if `t` is past the end.
    #[must_use]
    pub fn row(&self, t: usize) -> Option<&[f64]> {
        if t >= self.tokens {
            return None;
        }
        self.values.get(t * self.channels..(t + 1) * self.channels)
    }

    /// The value at `(token, channel)`, or `None` if either index is past the end.
    #[must_use]
    pub fn at(&self, token: usize, channel: usize) -> Option<f64> {
        if token >= self.tokens || channel >= self.channels {
            return None;
        }
        self.values.get(token * self.channels + channel).copied()
    }

    /// Non-zero elements — the count that drives every effective operation count downstream.
    #[must_use]
    pub fn nonzero(&self) -> u64 {
        self.values.iter().filter(|v| **v != 0.0).count() as u64
    }

    /// Fraction of elements that are non-zero, or `None` for an empty tensor.
    #[must_use]
    pub fn density(&self) -> Option<f64> {
        let n = self.values.len();
        if n == 0 {
            return None;
        }
        Some(self.nonzero() as f64 / n as f64)
    }

    fn slice_channels(&self, lo: usize, width: usize) -> Result<Self, AttnError> {
        let mut v = Vec::with_capacity(self.tokens * width);
        for t in 0..self.tokens {
            for c in 0..width {
                v.push(self.values[t * self.channels + lo + c]);
            }
        }
        Self::new(self.tokens, width, v, self.domain)
    }
}

/// Per-channel mean firing rate over a run: the standard rate readout of a spiking classifier.
///
/// Averaged over timesteps and tokens, so the result has one entry per channel and each is in
/// `[0, 1]` for a binary output.
///
/// # Errors
///
/// [`AttnError::Empty`] if `outputs` is empty; [`AttnError::BadShape`] if the outputs disagree on
/// their shape.
pub fn rate_readout(outputs: &[Tensor]) -> Result<Vec<f64>, AttnError> {
    let first = outputs.first().ok_or(AttnError::Empty { what: "outputs" })?;
    let (tokens, channels) = (first.tokens, first.channels);
    let mut acc = vec![0.0f64; channels];
    for o in outputs {
        if o.tokens != tokens || o.channels != channels {
            return Err(AttnError::BadShape {
                what: "readout",
                got: o.tokens * o.channels,
                want: tokens * channels,
            });
        }
        for t in 0..tokens {
            for c in 0..channels {
                acc[c] += o.values[t * channels + c];
            }
        }
    }
    let n = (outputs.len() * tokens) as f64;
    for a in &mut acc {
        *a /= n;
    }
    Ok(acc)
}

// ---------------------------------------------------------------------------------------------
// Counting helpers
// ---------------------------------------------------------------------------------------------

fn mul2(a: usize, b: usize, what: &'static str) -> Result<u64, AttnError> {
    (a as u64).checked_mul(b as u64).ok_or(AttnError::Overflow { what })
}

fn mul3(a: usize, b: usize, c: usize, what: &'static str) -> Result<u64, AttnError> {
    mul2(a, b, what)?.checked_mul(c as u64).ok_or(AttnError::Overflow { what })
}

const WHY_AC_WEIGHT: &str =
    "the activation operand is binary, so the weight is gated rather than multiplied";
const WHY_MAC_WEIGHT: &str =
    "neither operand is binary: the activation is multi-valued and needs a multiplier";
const WHY_BIAS: &str = "a bias is added to every output whether or not anything fired";
const WHY_MEMBRANE: &str = "membrane update terms: the leak difference and the sum back into v";
const WHY_LEAK: &str = "the membrane leak factor 1/tau multiplies the whole difference";
const WHY_THRESHOLD: &str = "one threshold comparison per neuron per timestep";
const WHY_SCALE: &str = "the attention scale multiplies every element of the attended output";
const WHY_RESIDUAL_ADD: &str = "a shortcut adds two tensors elementwise, widening the domain";
const WHY_RESIDUAL_LOGIC: &str = "a shortcut combines two binary tensors with one gate per element";

// ---------------------------------------------------------------------------------------------
// Linear projection
// ---------------------------------------------------------------------------------------------

/// A dense projection with real weights, row-major by output: the weight from input `i` to output
/// `j` is `w[j * n_in + i]`.
///
/// That layout is stated because a transposed weight matrix on a square layer produces the same
/// effective operation count, so a transposition survives every count-based test — the same trap
/// [`crate::metrics::SynOpMeter::layer`] documents.
#[derive(Debug, Clone, PartialEq)]
pub struct Linear {
    n_in: usize,
    n_out: usize,
    w: Vec<f64>,
    bias: Option<Vec<f64>>,
}

impl Linear {
    /// Build from weights and an optional bias.
    ///
    /// # Errors
    ///
    /// [`AttnError::Empty`] for a zero dimension; [`AttnError::BadShape`] if `w.len()` is not
    /// `n_in * n_out` or the bias is not `n_out` long; [`AttnError::NonFinite`] for a `NaN` or
    /// infinite weight, which would make the "weight is non-zero" test meaningless.
    pub fn new(
        n_in: usize,
        n_out: usize,
        w: Vec<f64>,
        bias: Option<Vec<f64>>,
    ) -> Result<Self, AttnError> {
        if n_in == 0 {
            return Err(AttnError::Empty { what: "n_in" });
        }
        if n_out == 0 {
            return Err(AttnError::Empty { what: "n_out" });
        }
        let want = n_in.checked_mul(n_out).ok_or(AttnError::Overflow { what: "weights" })?;
        if w.len() != want {
            return Err(AttnError::BadShape { what: "weights", got: w.len(), want });
        }
        for (i, v) in w.iter().enumerate() {
            if !v.is_finite() {
                return Err(AttnError::NonFinite { what: "weights", index: i });
            }
        }
        if let Some(b) = &bias {
            if b.len() != n_out {
                return Err(AttnError::BadShape { what: "bias", got: b.len(), want: n_out });
            }
            for (i, v) in b.iter().enumerate() {
                if !v.is_finite() {
                    return Err(AttnError::NonFinite { what: "bias", index: i });
                }
            }
        }
        Ok(Self { n_in, n_out, w, bias })
    }

    /// Input channels.
    #[must_use]
    pub fn n_in(&self) -> usize {
        self.n_in
    }

    /// Output channels.
    #[must_use]
    pub fn n_out(&self) -> usize {
        self.n_out
    }

    /// Non-zero weights — the connection density the effective count is conditioned on.
    #[must_use]
    pub fn nonzero_weights(&self) -> u64 {
        self.w.iter().filter(|v| **v != 0.0).count() as u64
    }

    /// Project `x`, charging the audit.
    ///
    /// The operation kind follows from `x`'s domain: binary in means accumulates, anything wider
    /// means multiply-accumulates. The result is always [`Domain::Real`] — a projection of spikes
    /// by real weights is real, which is why a spiking neuron always follows one.
    ///
    /// # Errors
    ///
    /// [`AttnError::BadShape`] if `x` has the wrong channel count; [`AttnError::Overflow`] or
    /// [`AttnError::Inconsistent`] from the audit; [`AttnError::NonFinite`] if an output overflowed
    /// to infinity.
    pub fn forward(
        &self,
        x: &Tensor,
        path: &str,
        audit: &mut Audit,
    ) -> Result<Tensor, AttnError> {
        if x.channels != self.n_in {
            return Err(AttnError::BadShape {
                what: "projection input",
                got: x.channels,
                want: self.n_in,
            });
        }
        let tokens = x.tokens;
        let mut out = vec![0.0f64; tokens * self.n_out];
        let kind = OpKind::product(x.domain, Domain::Real);
        let because = if kind == OpKind::Ac { WHY_AC_WEIGHT } else { WHY_MAC_WEIGHT };
        let dense = mul3(tokens, self.n_in, self.n_out, "projection")?;
        let mut effective = 0u64;
        for t in 0..tokens {
            for i in 0..self.n_in {
                let a = x.values[t * self.n_in + i];
                if a == 0.0 {
                    continue;
                }
                for j in 0..self.n_out {
                    let w = self.w[j * self.n_in + i];
                    if w != 0.0 {
                        effective += 1;
                        out[t * self.n_out + j] += a * w;
                    }
                }
            }
        }
        audit.charge(path, kind, because, dense, effective)?;
        if let Some(b) = &self.bias {
            for t in 0..tokens {
                for j in 0..self.n_out {
                    out[t * self.n_out + j] += b[j];
                }
            }
            let n = mul2(tokens, self.n_out, "bias")?;
            audit.charge(&format!("{path}.bias"), OpKind::Add, WHY_BIAS, n, n)?;
        }
        Tensor::real(tokens, self.n_out, out)
    }
}

// ---------------------------------------------------------------------------------------------
// The spiking neuron layer
// ---------------------------------------------------------------------------------------------

/// `base^exp` by repeated squaring, over the full `u32` exponent range that `f64::powi`'s `i32`
/// cannot carry. `base` is a membrane decay in `[0, 1)` at every call site, so the running product
/// underflows toward zero and never overflows.
fn pow_u32(base: f64, exp: u32) -> f64 {
    let mut acc = 1.0f64;
    let mut b = base;
    let mut e = exp;
    while e > 0 {
        if e & 1 == 1 {
            acc *= b;
        }
        e >>= 1;
        if e > 0 {
            b *= b;
        }
    }
    acc
}

/// A layer of leaky integrate-and-fire neurons, one per `(token, channel)`, in Spikformer's
/// discrete dimensionless form.
///
/// The update is the paper's, transcribed rather than rescaled:
///
/// ```text
/// H[t] = V[t-1] + (1/tau) * ( X[t] - (V[t-1] - v_reset) )
/// S[t] = 1 if H[t] >= v_th else 0
/// V[t] = v_reset if S[t] else H[t]
/// ```
///
/// with the paper's `tau = 2.0`, `v_th = 1.0`, `v_reset = 0.0`. There is no `dt`: one call is one
/// transformer timestep, and how long a timestep lasts is a property of the deployment rather than
/// of the model. See this module's header for why this type deliberately does not implement
/// [`crate::neuron::Neuron`].
///
/// # The closed form it is checked against
///
/// With no input and no spike the membrane relaxes geometrically toward `v_reset`:
///
/// ```text
/// V[t] = v_reset + (V[0] - v_reset) * (1 - 1/tau)^t
/// ```
///
/// exactly — the recurrence `U <- U * (1 - 1/tau)` on `U = V - v_reset` has no approximation in it.
/// That expression is [`LifLayer::relaxed`].
#[derive(Debug, Clone, PartialEq)]
pub struct LifLayer {
    tokens: usize,
    channels: usize,
    tau: f64,
    v_th: f64,
    v_reset: f64,
    v: Vec<f64>,
}

impl LifLayer {
    /// Build a layer of `tokens * channels` neurons at rest.
    ///
    /// # Errors
    ///
    /// [`AttnError::Empty`] for a zero dimension; [`AttnError::BadParameter`] if `tau < 1`
    /// (the leak `1/tau` would exceed 1 and the membrane would oscillate rather than decay), if
    /// `v_th <= v_reset` (the neuron would fire on every timestep including a silent one), or if
    /// any parameter is not finite.
    pub fn new(
        tokens: usize,
        channels: usize,
        tau: f64,
        v_th: f64,
        v_reset: f64,
    ) -> Result<Self, AttnError> {
        if tokens == 0 {
            return Err(AttnError::Empty { what: "tokens" });
        }
        if channels == 0 {
            return Err(AttnError::Empty { what: "channels" });
        }
        if !tau.is_finite() || tau < 1.0 {
            return Err(AttnError::BadParameter { what: "tau", value: tau });
        }
        if !v_th.is_finite() || !v_reset.is_finite() || v_th <= v_reset {
            return Err(AttnError::BadParameter { what: "v_th", value: v_th });
        }
        let n = tokens.checked_mul(channels).ok_or(AttnError::Overflow { what: "neurons" })?;
        Ok(Self { tokens, channels, tau, v_th, v_reset, v: vec![v_reset; n] })
    }

    /// The membrane decay per timestep, `1 - 1/tau`, in `[0, 1)`.
    #[must_use]
    pub fn decay(&self) -> f64 {
        1.0 - 1.0 / self.tau
    }

    /// The firing threshold.
    #[must_use]
    pub fn v_th(&self) -> f64 {
        self.v_th
    }

    /// The closed-form membrane after `steps` silent timesteps starting from `v0`, assuming no
    /// spike occurred: `v_reset + (v0 - v_reset) * decay^steps`.
    ///
    /// Used to check [`LifLayer::forward`] against something other than itself, so it is exact
    /// over the **whole** `u32` range. The power is taken by repeated squaring on the `u32`
    /// itself: `self.decay().powi(steps as i32)` turned every `steps >= 2^31` into a negative
    /// exponent and returned `inf` — for `tau = 2`, `v_reset = 0`, `v0 = 1` it answered `inf` at
    /// `steps = 2_147_483_648` and `0.0` one step earlier.
    #[must_use]
    pub fn relaxed(&self, v0: f64, steps: u32) -> f64 {
        self.v_reset + (v0 - self.v_reset) * pow_u32(self.decay(), steps)
    }

    /// Current membrane potentials, row-major by token.
    #[must_use]
    pub fn membranes(&self) -> &[f64] {
        &self.v
    }

    /// The timestep at which a neuron starting at rest first fires under a **constant** input `x`,
    /// in closed form, or `None` when it never does.
    ///
    /// Under constant input the membrane has no spike to interrupt it, so the recurrence
    /// `V <- V·d + (x + v_reset)/tau` with `d = 1 - 1/tau` solves exactly:
    ///
    /// ```text
    /// V[t] = (x + v_reset) - x · d^t
    /// ```
    ///
    /// and the first spike is the smallest `t >= 1` with `V[t] >= v_th`, which is
    /// `ceil( ln((x + v_reset - v_th) / x) / ln(d) )`.
    ///
    /// `None` when `x <= v_th - v_reset`: the membrane converges to `x + v_reset` from below and
    /// never reaches the threshold, which is a different statement from "fires eventually, slowly"
    /// and is kept different here — a rate-coded readout cannot recover the difference later. The
    /// tangent case `x + v_reset == v_th` is `None` for `tau > 1`, where the membrane approaches
    /// the threshold without touching it, and `Some(1)` for `tau == 1`, where it lands on it
    /// exactly in one step.
    ///
    /// ⚠ `None` carries **three** distinct meanings and only `x` itself tells them apart:
    ///
    /// 1. `x` is not finite, so there is no trajectory to solve;
    /// 2. `x <= v_th - v_reset`, the sub-threshold asymptote above;
    /// 3. the neuron **does** fire, later than `u32::MAX` timesteps. That is reachable: at
    ///    `tau = 1e10`, `v_th = 0.5`, `v_reset = 0` and `x = 1.0` the closed form gives step
    ///    `6.93e9`. Reporting it would need a wider integer, and letting the cast saturate would
    ///    claim a first spike at `u32::MAX` — a step no run reaches, which is worse than `None`.
    ///
    /// ⚠ The prediction is exact for the recurrence as written. Where a membrane lands within a
    /// floating-point ulp of the threshold the iteration's accumulated rounding could in principle
    /// disagree by one timestep; the sweep in this module's tests pins 48 combinations of `tau`,
    /// `v_reset` and drive and this review did not find such a case, which is weaker than a proof.
    ///
    /// **This is the latency that buys back the accuracy a binary activation gave up**, and it is
    /// why a spiking transformer is run for `T` timesteps: a neuron below this threshold
    /// contributes nothing at all, and one just above it contributes only after several timesteps
    /// have been paid for in full.
    #[must_use]
    pub fn first_spike_step(&self, x: f64) -> Option<u32> {
        if !x.is_finite() {
            return None;
        }
        let d = self.decay();
        if d == 0.0 {
            // tau == 1: the membrane forgets everything each step and lands EXACTLY on its
            // asymptote in one step, so the tangent case `x + v_reset == v_th` fires here and
            // does not below. Ordering this before the guard is deliberate; a sweep over
            // `tau = 1` with a negative `v_reset` caught the other order returning None for a
            // neuron that fires on the first timestep.
            return if x + self.v_reset >= self.v_th { Some(1) } else { None };
        }
        if x <= self.v_th - self.v_reset {
            return None;
        }
        let r = (x + self.v_reset - self.v_th) / x;
        let t = (r.ln() / d.ln()).ceil();
        if !t.is_finite() || t > f64::from(u32::MAX) {
            return None;
        }
        Some((t as u32).max(1))
    }

    /// Return every membrane to `v_reset`, forgetting the sequence seen so far.
    pub fn reset_state(&mut self) {
        self.v.fill(self.v_reset);
    }

    /// One timestep: integrate `x`, emit spikes, charge the audit.
    ///
    /// # What is charged, and why it is not zero
    ///
    /// Per neuron per timestep: one [`OpKind::Compare`] for the threshold test, one
    /// [`OpKind::Shift`] or [`OpKind::Mul`] for the `1/tau` factor depending on whether that factor
    /// is an exact power of two (it is for the paper's `tau = 2`), and the additions of the update —
    /// two when `v_reset` is zero, three otherwise, because `V - v_reset` disappears when
    /// `v_reset` is zero. The reset is a multiplexer and is charged nothing.
    ///
    /// None of these are gated on activity: a silent neuron pays all of them. That is the term a
    /// synaptic-operation count has no slot for, and on the toy model in this module's tests it is
    /// seven eighths of the bill.
    ///
    /// # Errors
    ///
    /// [`AttnError::BadShape`] if `x` is not this layer's shape; [`AttnError::NonFinite`] if a
    /// membrane left the finite range; [`AttnError::Overflow`] from the audit.
    pub fn forward(
        &mut self,
        x: &Tensor,
        path: &str,
        audit: &mut Audit,
    ) -> Result<Tensor, AttnError> {
        if x.tokens != self.tokens || x.channels != self.channels {
            return Err(AttnError::BadShape {
                what: "neuron input",
                got: x.tokens * x.channels,
                want: self.tokens * self.channels,
            });
        }
        let leak = 1.0 / self.tau;
        let mut bits = vec![false; self.v.len()];
        for idx in 0..self.v.len() {
            let v = self.v[idx];
            let h = v + leak * (x.values[idx] - (v - self.v_reset));
            if !h.is_finite() {
                return Err(AttnError::NonFinite { what: "membrane", index: idx });
            }
            let fired = h >= self.v_th;
            self.v[idx] = if fired { self.v_reset } else { h };
            bits[idx] = fired;
        }
        let n = self.v.len() as u64;
        let adds = if self.v_reset == 0.0 { 2 * n } else { 3 * n };
        audit.charge(&format!("{path}.membrane"), OpKind::Add, WHY_MEMBRANE, adds, adds)?;
        let leak_kind = if is_exact_power_of_two(leak) { OpKind::Shift } else { OpKind::Mul };
        audit.charge(&format!("{path}.leak"), leak_kind, WHY_LEAK, n, n)?;
        audit.charge(&format!("{path}.threshold"), OpKind::Compare, WHY_THRESHOLD, n, n)?;
        Tensor::spikes(self.tokens, self.channels, &bits)
    }
}

// ---------------------------------------------------------------------------------------------
// The attention product
// ---------------------------------------------------------------------------------------------

/// Which way to associate the triple product `Q Kᵀ V`.
///
/// The two give **identical results** when the operands are binary — the intermediate sums are
/// exact integers in `f64` while they stay below `2^53` — and **different operation counts**. That
/// makes the choice a pure cost decision, and which one is cheaper depends on whether the sequence
/// is longer than a head is wide.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Order {
    /// `(Q Kᵀ) V`, the form Spikformer writes. Costs `n_q · n_k · d + n_q · n_k · d_v`, so it is
    /// quadratic in sequence length and materialises the `n_q × n_k` score matrix.
    ScoresFirst,
    /// `Q (Kᵀ V)`. Costs `d · n_k · d_v + n_q · d · d_v`, so it is linear in sequence length and
    /// materialises a `d × d_v` matrix instead. Cheaper whenever the sequence is longer than the
    /// head is wide, which is the usual case.
    ValuesFirst,
}

/// The spiking attention product, without softmax and without scaling.
///
/// `q` is `n_q × d`, `k` is `n_k × d`, `v` is `n_k × d_v`; the result is `n_q × d_v` and is
/// [`Domain::Real`] even when every input is binary, because the entries are sums of up to
/// `n_k · d` ones.
///
/// # What removing the softmax means
///
/// In a standard transformer the softmax turns the scores into a convex combination, so the output
/// is an average of value rows. Here it is a **weighted sum with integer weights** and nothing
/// normalises it, so the output's magnitude grows with the sequence length and with the firing
/// density. That is what the scalar `s` and the following neuron's threshold are for, and it is why
/// a spiking transformer's scale is not the `1/√d` of Vaswani et al. — the quantity being tamed is
/// different.
///
/// # Errors
///
/// [`AttnError::BadShape`] if `q` and `k` disagree on `d` or `k` and `v` disagree on `n_k`;
/// [`AttnError::Overflow`] or [`AttnError::Inconsistent`] from the audit; [`AttnError::NonFinite`]
/// if a sum left the finite range.
pub fn attend(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    order: Order,
    path: &str,
    audit: &mut Audit,
) -> Result<Tensor, AttnError> {
    if q.channels != k.channels {
        return Err(AttnError::BadShape { what: "key width", got: k.channels, want: q.channels });
    }
    if k.tokens != v.tokens {
        return Err(AttnError::BadShape { what: "value rows", got: v.tokens, want: k.tokens });
    }
    let (nq, nk, d, dv) = (q.tokens, k.tokens, q.channels, v.channels);
    let mut out = vec![0.0f64; nq * dv];
    match order {
        Order::ScoresFirst => {
            // S = Q Kᵀ. Both operands are activations, so the kind is decided by whichever is
            // binary; when both are, this is the purest accumulate in the architecture.
            let kind = OpKind::product(q.domain, k.domain);
            let because = if kind == OpKind::Ac { WHY_AC_WEIGHT } else { WHY_MAC_WEIGHT };
            let mut scores = vec![0.0f64; nq * nk];
            let mut eff = 0u64;
            for a in 0..nq {
                for b in 0..nk {
                    let mut acc = 0.0;
                    for c in 0..d {
                        let x = q.values[a * d + c];
                        let y = k.values[b * d + c];
                        if x != 0.0 && y != 0.0 {
                            eff += 1;
                            acc += x * y;
                        }
                    }
                    scores[a * nk + b] = acc;
                }
            }
            audit.charge(
                &format!("{path}.scores"),
                kind,
                because,
                mul3(nq, nk, d, "scores")?,
                eff,
            )?;
            // O = S V. S is an integer matrix, so the binary side here is V.
            let kind2 = OpKind::product(Domain::Real, v.domain);
            let because2 = if kind2 == OpKind::Ac { WHY_AC_WEIGHT } else { WHY_MAC_WEIGHT };
            let mut eff2 = 0u64;
            for a in 0..nq {
                for b in 0..nk {
                    let s = scores[a * nk + b];
                    if s == 0.0 {
                        continue;
                    }
                    for e in 0..dv {
                        let y = v.values[b * dv + e];
                        if y != 0.0 {
                            eff2 += 1;
                            out[a * dv + e] += s * y;
                        }
                    }
                }
            }
            audit.charge(
                &format!("{path}.attend"),
                kind2,
                because2,
                mul3(nq, nk, dv, "attend")?,
                eff2,
            )?;
        }
        Order::ValuesFirst => {
            // G = Kᵀ V, a d × dv matrix.
            let kind = OpKind::product(k.domain, v.domain);
            let because = if kind == OpKind::Ac { WHY_AC_WEIGHT } else { WHY_MAC_WEIGHT };
            let mut g = vec![0.0f64; d * dv];
            let mut eff = 0u64;
            for b in 0..nk {
                for c in 0..d {
                    let x = k.values[b * d + c];
                    if x == 0.0 {
                        continue;
                    }
                    for e in 0..dv {
                        let y = v.values[b * dv + e];
                        if y != 0.0 {
                            eff += 1;
                            g[c * dv + e] += x * y;
                        }
                    }
                }
            }
            audit.charge(
                &format!("{path}.gram"),
                kind,
                because,
                mul3(d, nk, dv, "gram")?,
                eff,
            )?;
            let kind2 = OpKind::product(q.domain, Domain::Real);
            let because2 = if kind2 == OpKind::Ac { WHY_AC_WEIGHT } else { WHY_MAC_WEIGHT };
            let mut eff2 = 0u64;
            for a in 0..nq {
                for c in 0..d {
                    let x = q.values[a * d + c];
                    if x == 0.0 {
                        continue;
                    }
                    for e in 0..dv {
                        let y = g[c * dv + e];
                        if y != 0.0 {
                            eff2 += 1;
                            out[a * dv + e] += x * y;
                        }
                    }
                }
            }
            audit.charge(
                &format!("{path}.attend"),
                kind2,
                because2,
                mul3(nq, d, dv, "attend")?,
                eff2,
            )?;
        }
    }
    Tensor::real(nq, dv, out)
}

// ---------------------------------------------------------------------------------------------
// Spiking self-attention
// ---------------------------------------------------------------------------------------------

/// Spikformer's spiking self-attention block: four projections, five neuron layers, and the
/// softmax-free product.
///
/// ```text
/// Q = SN_q(X W_q)   K = SN_k(X W_k)   V = SN_v(X W_v)      (all binary)
/// A = SN_a( s · attend(Q, K, V) )                          (binary)
/// Y = SN_o( A W_o )                                        (binary)
/// ```
///
/// Heads tile the channels: head `h` owns channels `h·d_head .. (h+1)·d_head` of `Q`, `K` and `V`
/// and writes its output into the same slice, which is how a multi-head attention is a single
/// projection plus a reshape rather than `h` separate projections.
///
/// # Folding the scale into the threshold
///
/// `SN_a(s · Z)` compares `s · Z` against `v_th`. With `v_reset = 0` the membrane recurrence is
/// linear in its input and the reset is scale-invariant, so running the neuron on `Z` against a
/// threshold of `v_th / s` gives **the identical spike train** — and when `s` is an exact power of
/// two, gives it bit for bit, because scaling by a power of two commutes exactly with rounding.
/// That removes one real multiply per element per timestep from the bill for free.
///
/// [`Ssa::new`] refuses the fold rather than performing it approximately, and there are **three**
/// conditions, not two:
///
/// 1. `s` is an exact positive normal power of two ([`is_exact_power_of_two`]); a non-power-of-two
///    `s` would make the two paths differ in the last place.
/// 2. `v_reset == 0`; a non-zero one breaks the linearity the argument rests on.
/// 3. Both paths stay in the range where the rewrite is exact. Every operand of the attention
///    product is binary, so an attended element is at most `tokens · d_head` — the number of
///    (key, channel) pairs that can contribute a one — and the unfolded path must be able to form
///    `s · tokens · d_head` finitely, while the folded path must be able to form `v_th / s` as a
///    normal number. Without this third condition `s = 2^1022` satisfies the first two and the
///    two paths disagree **completely**: on a 2×2 all-ones input the unfolded path overflows to
///    infinity and returns [`AttnError::NonFinite`] while the folded path returns a full row of
///    spikes.
///
/// This crate did not locate the fold described in the spiking-transformer literature; it is
/// offered here as an exact rewrite with the conditions stated, not as a reproduction of anyone's
/// result.
#[derive(Debug, Clone, PartialEq)]
pub struct Ssa {
    tokens: usize,
    d_model: usize,
    heads: usize,
    d_head: usize,
    wq: Linear,
    wk: Linear,
    wv: Linear,
    wo: Linear,
    nq: LifLayer,
    nk: LifLayer,
    nv: LifLayer,
    na: LifLayer,
    no: LifLayer,
    scale: f64,
    fold_scale: bool,
    order: Order,
}

impl Ssa {
    /// Assemble from four `d_model × d_model` projections.
    ///
    /// `scale` is Spikformer's `s`; the paper uses `0.125`, which is a power of two and therefore
    /// foldable. A `scale` of exactly `1.0` is not an operation and is charged nothing.
    ///
    /// # Errors
    ///
    /// [`AttnError::BadShape`] if any projection is not `d_model × d_model`;
    /// [`AttnError::HeadsDoNotDivide`] if `heads` does not divide `d_model`;
    /// [`AttnError::BadParameter`] for a non-finite or non-positive `scale`, or from
    /// [`LifLayer::new`]; [`AttnError::UnfoldableScale`] if `fold_scale` is set where the fold
    /// would not be exact.
    pub fn new(
        tokens: usize,
        heads: usize,
        wq: Linear,
        wk: Linear,
        wv: Linear,
        wo: Linear,
        tau: f64,
        v_th: f64,
        v_reset: f64,
        scale: f64,
        fold_scale: bool,
        order: Order,
    ) -> Result<Self, AttnError> {
        let d_model = wq.n_out;
        for p in [&wq, &wk, &wv, &wo] {
            if p.n_in != d_model || p.n_out != d_model {
                return Err(AttnError::BadShape {
                    what: "attention projection",
                    got: p.n_in * p.n_out,
                    want: d_model * d_model,
                });
            }
        }
        if heads == 0 {
            return Err(AttnError::Empty { what: "heads" });
        }
        if !d_model.is_multiple_of(heads) {
            return Err(AttnError::HeadsDoNotDivide { channels: d_model, heads });
        }
        if !scale.is_finite() || scale <= 0.0 {
            return Err(AttnError::BadParameter { what: "scale", value: scale });
        }
        if fold_scale {
            if !is_exact_power_of_two(scale) {
                return Err(AttnError::UnfoldableScale {
                    scale,
                    why: "not an exact power of two, so the folded path would differ in the last place",
                });
            }
            if v_reset != 0.0 {
                return Err(AttnError::UnfoldableScale {
                    scale,
                    why: "v_reset is not zero, so the membrane recurrence is not scale-invariant",
                });
            }
            // The third condition. q, k and v are all neuron outputs and therefore binary, so an
            // attended element is a count of (key, channel) pairs and cannot exceed
            // `tokens * d_head`. If the unfolded path cannot even form `s` times that, the two
            // paths do not agree on this block's own inputs and the fold is not a rewrite.
            let peak = (tokens as f64) * ((d_model / heads) as f64);
            if !(peak * scale).is_finite() {
                return Err(AttnError::UnfoldableScale {
                    scale,
                    why: "s times the largest attainable attended element overflows, so the \
                          unfolded path would return NonFinite where the folded one returns spikes",
                });
            }
            if !(v_th / scale).is_normal() {
                return Err(AttnError::UnfoldableScale {
                    scale,
                    why: "v_th / s is not a normal number, so the folded threshold is not exact",
                });
            }
        }
        let attn_th = if fold_scale { v_th / scale } else { v_th };
        Ok(Self {
            tokens,
            d_model,
            heads,
            d_head: d_model / heads,
            wq,
            wk,
            wv,
            wo,
            nq: LifLayer::new(tokens, d_model, tau, v_th, v_reset)?,
            nk: LifLayer::new(tokens, d_model, tau, v_th, v_reset)?,
            nv: LifLayer::new(tokens, d_model, tau, v_th, v_reset)?,
            na: LifLayer::new(tokens, d_model, tau, attn_th, v_reset)?,
            no: LifLayer::new(tokens, d_model, tau, v_th, v_reset)?,
            scale,
            fold_scale,
            order,
        })
    }

    /// Channels the block carries.
    #[must_use]
    pub fn d_model(&self) -> usize {
        self.d_model
    }

    /// Attention heads.
    #[must_use]
    pub fn heads(&self) -> usize {
        self.heads
    }

    /// Sequence positions this block was built for.
    #[must_use]
    pub fn tokens(&self) -> usize {
        self.tokens
    }

    /// Whether the scale is folded into the attention neuron's threshold.
    #[must_use]
    pub fn folds_scale(&self) -> bool {
        self.fold_scale
    }

    /// Forget every membrane.
    pub fn reset_state(&mut self) {
        for n in [&mut self.nq, &mut self.nk, &mut self.nv, &mut self.na, &mut self.no] {
            n.reset_state();
        }
    }

    /// One timestep of spiking self-attention.
    ///
    /// # Errors
    ///
    /// [`AttnError::BadShape`] if `x` is not `tokens × d_model`; anything [`Linear::forward`],
    /// [`LifLayer::forward`] or [`attend`] returns.
    pub fn forward(
        &mut self,
        x: &Tensor,
        path: &str,
        audit: &mut Audit,
    ) -> Result<Tensor, AttnError> {
        if x.tokens != self.tokens || x.channels != self.d_model {
            return Err(AttnError::BadShape {
                what: "attention input",
                got: x.tokens * x.channels,
                want: self.tokens * self.d_model,
            });
        }
        let q_pre = self.wq.forward(x, &format!("{path}.q_proj"), audit)?;
        let k_pre = self.wk.forward(x, &format!("{path}.k_proj"), audit)?;
        let v_pre = self.wv.forward(x, &format!("{path}.v_proj"), audit)?;
        let q = self.nq.forward(&q_pre, &format!("{path}.q_neuron"), audit)?;
        let k = self.nk.forward(&k_pre, &format!("{path}.k_neuron"), audit)?;
        let v = self.nv.forward(&v_pre, &format!("{path}.v_neuron"), audit)?;

        let mut merged = vec![0.0f64; self.tokens * self.d_model];
        for h in 0..self.heads {
            let lo = h * self.d_head;
            let qh = q.slice_channels(lo, self.d_head)?;
            let kh = k.slice_channels(lo, self.d_head)?;
            let vh = v.slice_channels(lo, self.d_head)?;
            // Every head charges the same two site paths, so the report has one row per
            // operation rather than one per head.
            let oh = attend(&qh, &kh, &vh, self.order, &format!("{path}.ssa"), audit)?;
            for t in 0..self.tokens {
                for c in 0..self.d_head {
                    merged[t * self.d_model + lo + c] = oh.values[t * self.d_head + c];
                }
            }
        }

        if !self.fold_scale && self.scale != 1.0 {
            let mut effective = 0u64;
            for m in &mut merged {
                if *m != 0.0 {
                    effective += 1;
                    *m *= self.scale;
                }
            }
            let kind = if is_exact_power_of_two(self.scale) { OpKind::Shift } else { OpKind::Mul };
            let dense = mul2(self.tokens, self.d_model, "scale")?;
            audit.charge(&format!("{path}.scale"), kind, WHY_SCALE, dense, effective)?;
        }
        let scaled = Tensor::real(self.tokens, self.d_model, merged)?;
        let a = self.na.forward(&scaled, &format!("{path}.attn_neuron"), audit)?;
        let o_pre = self.wo.forward(&a, &format!("{path}.out_proj"), audit)?;
        self.no.forward(&o_pre, &format!("{path}.out_neuron"), audit)
    }
}

// ---------------------------------------------------------------------------------------------
// The spiking MLP
// ---------------------------------------------------------------------------------------------

/// Spikformer's two-layer spiking `MLP`: `SN(W2 · SN(W1 · X))`.
///
/// The hidden width is conventionally four times `d_model`, following the original transformer;
/// nothing here enforces that and the tests use a hidden width of `d_model` so the operation counts
/// stay hand-computable.
#[derive(Debug, Clone, PartialEq)]
pub struct SpikingMlp {
    fc1: Linear,
    n1: LifLayer,
    fc2: Linear,
    n2: LifLayer,
}

impl SpikingMlp {
    /// Assemble from two projections whose widths meet.
    ///
    /// # Errors
    ///
    /// [`AttnError::BadShape`] if `fc1.n_out()` is not `fc2.n_in()`; [`AttnError::BadParameter`]
    /// from [`LifLayer::new`].
    pub fn new(
        tokens: usize,
        fc1: Linear,
        fc2: Linear,
        tau: f64,
        v_th: f64,
        v_reset: f64,
    ) -> Result<Self, AttnError> {
        if fc1.n_out != fc2.n_in {
            return Err(AttnError::BadShape {
                what: "mlp hidden width",
                got: fc2.n_in,
                want: fc1.n_out,
            });
        }
        let (h, o) = (fc1.n_out, fc2.n_out);
        Ok(Self {
            n1: LifLayer::new(tokens, h, tau, v_th, v_reset)?,
            n2: LifLayer::new(tokens, o, tau, v_th, v_reset)?,
            fc1,
            fc2,
        })
    }

    /// Output channels.
    #[must_use]
    pub fn n_out(&self) -> usize {
        self.fc2.n_out
    }

    /// Forget both membranes.
    pub fn reset_state(&mut self) {
        self.n1.reset_state();
        self.n2.reset_state();
    }

    /// One timestep.
    ///
    /// # Errors
    ///
    /// Anything [`Linear::forward`] or [`LifLayer::forward`] returns.
    pub fn forward(
        &mut self,
        x: &Tensor,
        path: &str,
        audit: &mut Audit,
    ) -> Result<Tensor, AttnError> {
        let h_pre = self.fc1.forward(x, &format!("{path}.fc1"), audit)?;
        let h = self.n1.forward(&h_pre, &format!("{path}.neuron1"), audit)?;
        let o_pre = self.fc2.forward(&h, &format!("{path}.fc2"), audit)?;
        self.n2.forward(&o_pre, &format!("{path}.neuron2"), audit)
    }
}

// ---------------------------------------------------------------------------------------------
// Residual shortcuts
// ---------------------------------------------------------------------------------------------

/// How a block's shortcut combines the identity path with the branch output.
///
/// **This choice decides whether the rest of the model is spike-driven at all**, which is why it is
/// an enum the caller must pick rather than a `+` buried in a forward pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Residual {
    /// No shortcut: the block's output is its branch's output. Keeps the stream binary and is not
    /// what any of the cited architectures do — included so a test can isolate the branch.
    None,
    /// Elementwise addition, as Spikformer writes it. Two spike tensors sum to `{0, 1, 2}`, so the
    /// stream leaves [`Domain::Binary`] at the first shortcut and **every projection after it is a
    /// multiply-accumulate**. The bound grows by one at each further shortcut.
    ///
    /// This is exactly the objection Yao et al. raise in *Spike-driven Transformer* (`NeurIPS`
    /// 2023): the architecture is described as spike-driven while its residual stream is not.
    SpikeAdd,
    /// `shortcut AND NOT branch`, in the family of spike-element-wise shortcuts from Fang, Yu,
    /// Chen, Huang, Masquelier and Tian, *Deep Residual Learning in Spiking Neural Networks*,
    /// `NeurIPS` 2021. The output stays binary, so the whole model stays multiplier-free.
    ///
    /// ⚠ The paper names three gates — `ADD`, `AND` and `IAND` — and this review took `IAND` to
    /// mean `shortcut ∧ ¬branch` from the paper's description rather than from its code. If that
    /// reading is wrong, the gate implemented here is still well defined and still binary; only the
    /// attribution would be.
    SewIand,
}

impl Residual {
    /// Combine a shortcut with a branch output.
    ///
    /// # Errors
    ///
    /// [`AttnError::BadShape`] if the two disagree; [`AttnError::NotSpikeDriven`] if
    /// [`Residual::SewIand`] is given an operand that is not binary; audit errors.
    pub fn combine(
        self,
        shortcut: &Tensor,
        branch: &Tensor,
        path: &str,
        audit: &mut Audit,
    ) -> Result<Tensor, AttnError> {
        if self == Self::None {
            return Ok(branch.clone());
        }
        if shortcut.tokens != branch.tokens || shortcut.channels != branch.channels {
            return Err(AttnError::BadShape {
                what: "shortcut",
                got: branch.tokens * branch.channels,
                want: shortcut.tokens * shortcut.channels,
            });
        }
        let n = mul2(shortcut.tokens, shortcut.channels, "shortcut")?;
        match self {
            Self::None => unreachable!(),
            Self::SpikeAdd => {
                let values: Vec<f64> =
                    shortcut.values.iter().zip(&branch.values).map(|(a, b)| a + b).collect();
                audit.charge(path, OpKind::Add, WHY_RESIDUAL_ADD, n, n)?;
                Tensor::new(
                    shortcut.tokens,
                    shortcut.channels,
                    values,
                    shortcut.domain.sum(branch.domain),
                )
            }
            Self::SewIand => {
                if !shortcut.domain.is_binary() || !branch.domain.is_binary() {
                    return Err(AttnError::NotSpikeDriven { what: "IAND shortcut" });
                }
                let values: Vec<f64> = shortcut
                    .values
                    .iter()
                    .zip(&branch.values)
                    .map(|(a, b)| if *a != 0.0 && *b == 0.0 { 1.0 } else { 0.0 })
                    .collect();
                audit.charge(path, OpKind::Logic, WHY_RESIDUAL_LOGIC, n, n)?;
                Tensor::new(shortcut.tokens, shortcut.channels, values, Domain::Binary)
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// A transformer block
// ---------------------------------------------------------------------------------------------

/// One spiking transformer block: attention with a shortcut, then the `MLP` with a shortcut.
#[derive(Debug, Clone, PartialEq)]
pub struct Block {
    attn: Ssa,
    mlp: SpikingMlp,
    residual: Residual,
}

impl Block {
    /// Assemble. The `MLP` must return the attention's channel count so the shortcut lines up.
    ///
    /// # Errors
    ///
    /// [`AttnError::BadShape`] if the `MLP`'s output width is not the attention's `d_model`.
    pub fn new(attn: Ssa, mlp: SpikingMlp, residual: Residual) -> Result<Self, AttnError> {
        if mlp.n_out() != attn.d_model {
            return Err(AttnError::BadShape {
                what: "block width",
                got: mlp.n_out(),
                want: attn.d_model,
            });
        }
        Ok(Self { attn, mlp, residual })
    }

    /// Channels the block carries.
    #[must_use]
    pub fn d_model(&self) -> usize {
        self.attn.d_model
    }

    /// Forget every membrane in the block.
    pub fn reset_state(&mut self) {
        self.attn.reset_state();
        self.mlp.reset_state();
    }

    /// One timestep.
    ///
    /// # Errors
    ///
    /// Anything the attention, the `MLP` or the shortcut returns.
    pub fn forward(
        &mut self,
        x: &Tensor,
        path: &str,
        audit: &mut Audit,
    ) -> Result<Tensor, AttnError> {
        let a = self.attn.forward(x, &format!("{path}.attn"), audit)?;
        let h = self.residual.combine(x, &a, &format!("{path}.shortcut1"), audit)?;
        let m = self.mlp.forward(&h, &format!("{path}.mlp"), audit)?;
        self.residual.combine(&h, &m, &format!("{path}.shortcut2"), audit)
    }
}

// ---------------------------------------------------------------------------------------------
// Positional information
// ---------------------------------------------------------------------------------------------

/// A depthwise 1-D convolution over the token axis with zero padding, one kernel per channel.
///
/// Depthwise means no mixing across channels: channel `c` is convolved with `kernel[c·k .. (c+1)·k]`
/// and nothing else. That is what makes it cheap enough to sit in a positional encoder — the cost is
/// `tokens · channels · k` rather than `tokens · channels² · k`.
#[derive(Debug, Clone, PartialEq)]
pub struct DepthwiseConv1d {
    channels: usize,
    k: usize,
    kernel: Vec<f64>,
}

impl DepthwiseConv1d {
    /// Build from a `channels × k` kernel, row-major by channel.
    ///
    /// # Errors
    ///
    /// [`AttnError::Empty`] for a zero dimension; [`AttnError::BadParameter`] if `k` is even (an
    /// even kernel has no centre, so "same" padding would shift the sequence by half a position);
    /// [`AttnError::BadShape`] on the wrong length; [`AttnError::NonFinite`] for a bad tap.
    pub fn new(channels: usize, k: usize, kernel: Vec<f64>) -> Result<Self, AttnError> {
        if channels == 0 {
            return Err(AttnError::Empty { what: "channels" });
        }
        if k == 0 {
            return Err(AttnError::Empty { what: "kernel width" });
        }
        if k.is_multiple_of(2) {
            return Err(AttnError::BadParameter { what: "kernel width", value: k as f64 });
        }
        let want = channels.checked_mul(k).ok_or(AttnError::Overflow { what: "kernel" })?;
        if kernel.len() != want {
            return Err(AttnError::BadShape { what: "kernel", got: kernel.len(), want });
        }
        for (i, v) in kernel.iter().enumerate() {
            if !v.is_finite() {
                return Err(AttnError::NonFinite { what: "kernel", index: i });
            }
        }
        Ok(Self { channels, k, kernel })
    }

    /// A kernel that is a single unit tap at the centre for every channel: the identity.
    ///
    /// # Errors
    ///
    /// As [`DepthwiseConv1d::new`].
    pub fn centre_tap(channels: usize, k: usize) -> Result<Self, AttnError> {
        if k == 0 || k.is_multiple_of(2) {
            return Err(AttnError::BadParameter { what: "kernel width", value: k as f64 });
        }
        let mut kernel = vec![0.0; channels.max(1) * k];
        for c in 0..channels {
            kernel[c * k + k / 2] = 1.0;
        }
        Self::new(channels, k, kernel)
    }

    /// Convolve, charging the audit.
    ///
    /// # Errors
    ///
    /// [`AttnError::BadShape`] if `x` has the wrong channel count; audit errors;
    /// [`AttnError::NonFinite`] if a sum left the finite range.
    pub fn forward(
        &self,
        x: &Tensor,
        path: &str,
        audit: &mut Audit,
    ) -> Result<Tensor, AttnError> {
        if x.channels != self.channels {
            return Err(AttnError::BadShape {
                what: "conv input",
                got: x.channels,
                want: self.channels,
            });
        }
        let tokens = x.tokens;
        let pad = (self.k / 2) as isize;
        let mut out = vec![0.0f64; tokens * self.channels];
        let kind = OpKind::product(x.domain, Domain::Real);
        let because = if kind == OpKind::Ac { WHY_AC_WEIGHT } else { WHY_MAC_WEIGHT };
        let mut effective = 0u64;
        for t in 0..tokens {
            for c in 0..self.channels {
                let mut acc = 0.0;
                for j in 0..self.k {
                    let src = t as isize + j as isize - pad;
                    if src < 0 || src >= tokens as isize {
                        continue;
                    }
                    let a = x.values[src as usize * self.channels + c];
                    let w = self.kernel[c * self.k + j];
                    if a != 0.0 && w != 0.0 {
                        effective += 1;
                        acc += a * w;
                    }
                }
                out[t * self.channels + c] = acc;
            }
        }
        // Dense counts the full kernel at every position, including the taps that fall off the
        // ends: a padded convolution on real hardware still issues those multiply-adds against
        // zeros unless the loop is specialised at the boundary.
        let dense = mul3(tokens, self.channels, self.k, "conv")?;
        audit.charge(path, kind, because, dense, effective)?;
        Tensor::real(tokens, self.channels, out)
    }
}

/// How positional information enters a spiking sequence.
///
/// A transformer without one is permutation-equivariant: shuffling the tokens shuffles the outputs
/// and changes nothing else. The three mechanisms here differ in what they cost and in whether they
/// leave the stream binary.
#[derive(Debug, Clone, PartialEq)]
pub enum Position {
    /// None. The model is permutation-equivariant, which is occasionally what you want and is
    /// usually a bug.
    None,
    /// Add a fixed spike table to the input, elementwise. Faithful to how Vaswani et al. and
    /// Spikformer both introduce position — by addition — and therefore **widens the stream to
    /// `{0, 1, 2}` before the first projection**, turning every projection in the model into a
    /// multiply-accumulate. Build the table with [`Tensor::sinusoidal_spikes`].
    AddTable(Tensor),
    /// `OR` the same table in instead of adding it. Keeps the stream binary at the cost of losing
    /// the distinction between "the token fired" and "the position fired".
    ///
    /// **This crate's variant, not a published one.** It is here because the additive form's cost
    /// is invisible until you audit it, and a reader deserves to see the same model with that one
    /// decision changed.
    OrTable(Tensor),
    /// Spikformer's relative position embedding: a depthwise convolution over neighbouring
    /// positions, through a spiking neuron, **added** to the input — so this too widens the stream.
    /// It follows the conditional positional encoding of Chu et al., *Conditional Positional
    /// Encodings for Vision Transformers* (`arXiv`:2102.10882), reduced from 2-D to a sequence.
    Conditional(DepthwiseConv1d, LifLayer),
}

impl Position {
    /// Apply to one timestep's input.
    ///
    /// # Errors
    ///
    /// [`AttnError::BadShape`] if the table or the convolution does not match `x`; audit errors.
    pub fn apply(
        &mut self,
        x: &Tensor,
        path: &str,
        audit: &mut Audit,
    ) -> Result<Tensor, AttnError> {
        match self {
            Self::None => Ok(x.clone()),
            Self::AddTable(table) => Residual::SpikeAdd.combine(x, table, path, audit),
            Self::OrTable(table) => {
                if x.tokens != table.tokens || x.channels != table.channels {
                    return Err(AttnError::BadShape {
                        what: "position table",
                        got: table.tokens * table.channels,
                        want: x.tokens * x.channels,
                    });
                }
                if !x.domain.is_binary() || !table.domain.is_binary() {
                    return Err(AttnError::NotSpikeDriven { what: "OR position table" });
                }
                let values: Vec<f64> = x
                    .values
                    .iter()
                    .zip(&table.values)
                    .map(|(a, b)| if *a != 0.0 || *b != 0.0 { 1.0 } else { 0.0 })
                    .collect();
                let n = mul2(x.tokens, x.channels, "position")?;
                audit.charge(path, OpKind::Logic, WHY_RESIDUAL_LOGIC, n, n)?;
                Tensor::new(x.tokens, x.channels, values, Domain::Binary)
            }
            Self::Conditional(conv, neuron) => {
                let pre = conv.forward(x, &format!("{path}.conv"), audit)?;
                let spikes = neuron.forward(&pre, &format!("{path}.neuron"), audit)?;
                Residual::SpikeAdd.combine(x, &spikes, path, audit)
            }
        }
    }

    /// Forget the conditional encoder's membranes, if it has any.
    pub fn reset_state(&mut self) {
        if let Self::Conditional(_, n) = self {
            n.reset_state();
        }
    }
}

// ---------------------------------------------------------------------------------------------
// The model
// ---------------------------------------------------------------------------------------------

/// A stack of spiking transformer blocks with a positional encoder in front.
///
/// Built for a **fixed** token count, because every spiking neuron carries a membrane per sequence
/// position and a model that silently resized them would carry one sequence's state into the next.
/// A mismatched input is refused with both shapes named.
#[derive(Debug, Clone, PartialEq)]
pub struct Model {
    tokens: usize,
    d_model: usize,
    position: Position,
    blocks: Vec<Block>,
}

impl Model {
    /// Assemble.
    ///
    /// # Errors
    ///
    /// [`AttnError::Empty`] if `blocks` is empty; [`AttnError::BadShape`] if a block's width or
    /// token count disagrees with the first block's.
    pub fn new(tokens: usize, position: Position, blocks: Vec<Block>) -> Result<Self, AttnError> {
        let first = blocks.first().ok_or(AttnError::Empty { what: "blocks" })?;
        let d_model = first.d_model();
        for b in &blocks {
            if b.d_model() != d_model {
                return Err(AttnError::BadShape {
                    what: "block width",
                    got: b.d_model(),
                    want: d_model,
                });
            }
            if b.attn.tokens != tokens {
                return Err(AttnError::BadShape {
                    what: "block tokens",
                    got: b.attn.tokens,
                    want: tokens,
                });
            }
        }
        Ok(Self { tokens, d_model, position, blocks })
    }

    /// Sequence positions.
    #[must_use]
    pub fn tokens(&self) -> usize {
        self.tokens
    }

    /// Channels.
    #[must_use]
    pub fn d_model(&self) -> usize {
        self.d_model
    }

    /// Blocks in the stack.
    #[must_use]
    pub fn depth(&self) -> usize {
        self.blocks.len()
    }

    /// Forget every membrane in the model, so the next sequence starts from rest.
    pub fn reset_state(&mut self) {
        self.position.reset_state();
        for b in &mut self.blocks {
            b.reset_state();
        }
    }

    /// One timestep, charging `audit` and advancing its timestep counter.
    ///
    /// # Errors
    ///
    /// [`AttnError::BadShape`] if `x` is not `tokens × d_model`; anything a block returns.
    pub fn forward(&mut self, x: &Tensor, audit: &mut Audit) -> Result<Tensor, AttnError> {
        if x.tokens != self.tokens || x.channels != self.d_model {
            return Err(AttnError::BadShape {
                what: "model input",
                got: x.tokens * x.channels,
                want: self.tokens * self.d_model,
            });
        }
        let mut h = self.position.apply(x, "position", audit)?;
        for (i, b) in self.blocks.iter_mut().enumerate() {
            h = b.forward(&h, &format!("block{i}"), audit)?;
        }
        audit.tick();
        Ok(h)
    }

    /// A whole sequence of timesteps from the model's current state, with the audit that covers it.
    ///
    /// State is **not** reset first: a caller who wants an independent run calls
    /// [`Model::reset_state`]. Making that explicit is deliberate — a model that reset itself here
    /// could not be driven one timestep at a time, which is the mode a deployment runs in.
    ///
    /// # Errors
    ///
    /// [`AttnError::Empty`] if `inputs` is empty; anything [`Model::forward`] returns.
    pub fn run(&mut self, inputs: &[Tensor]) -> Result<(Vec<Tensor>, Audit), AttnError> {
        if inputs.is_empty() {
            return Err(AttnError::Empty { what: "inputs" });
        }
        let mut audit = Audit::new();
        let mut out = Vec::with_capacity(inputs.len());
        for x in inputs {
            out.push(self.forward(x, &mut audit)?);
        }
        Ok((out, audit))
    }
}

/// A model's shape and hyperparameters, with a deterministic random initialiser.
///
/// The defaults are Spikformer's where the paper states one: `tau = 2.0`, `v_th = 1.0`,
/// `v_reset = 0.0`, `scale = 0.125`, `SpikeAdd` shortcuts. The initialisation gain is **not** from
/// any paper — it is chosen so that an untrained model fires at all, and a model that never fires
/// makes every downstream test vacuous.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Spec {
    /// Sequence positions. Fixed for the life of the model.
    pub tokens: usize,
    /// Channels carried between blocks.
    pub d_model: usize,
    /// Attention heads; must divide `d_model`.
    pub heads: usize,
    /// Hidden width of each block's `MLP`. Conventionally `4 * d_model`.
    pub mlp_hidden: usize,
    /// Blocks in the stack.
    pub depth: usize,
    /// Membrane time constant in timesteps, dimensionless, `>= 1`.
    pub tau: f64,
    /// Firing threshold, dimensionless.
    pub v_th: f64,
    /// Post-spike membrane, dimensionless. Zero is required to fold the attention scale.
    pub v_reset: f64,
    /// Spikformer's `s`, applied to the attended output before the attention neuron.
    pub scale: f64,
    /// Whether to fold `scale` into the attention neuron's threshold. See [`Ssa::new`].
    pub fold_scale: bool,
    /// How to associate the triple product.
    pub order: Order,
    /// How shortcuts combine, which decides whether the model stays multiplier-free.
    pub residual: Residual,
    /// Uniform weight init is `±gain / sqrt(fan_in)`. Not a published initialisation.
    ///
    /// ⚠ **It moves the headline.** [`Audit::ac_fraction`] is a function of firing density while
    /// the per-neuron overhead is fixed, so the fraction rises with the gain; quoting one without
    /// naming the gain it was measured at is quoting a number nobody can reproduce. This module's
    /// tests sweep it rather than pin one value.
    pub gain: f64,
    /// Whether every projection carries the bias a folded `BatchNorm` leaves behind.
    ///
    /// Spikformer puts a `BatchNorm` after every linear layer, and folding it at inference cuts
    /// both ways: the BN **gain folds into the preceding weight matrix for free**, so it is not a
    /// multiply anyone pays, but the BN **shift becomes a bias that does not fold away** — one
    /// [`OpKind::Add`] per output element per timestep, paid whether or not anything fired. That
    /// is `tokens * (5 * d_model + mlp_hidden)` adds per block per timestep: four attention
    /// projections and the output projection at `d_model` outputs each, plus the `MLP`'s
    /// `mlp_hidden` and `d_model`.
    ///
    /// At initialisation the folded shift is exactly zero (`running_mean = 0`, `gamma = 1`,
    /// `beta = 0`), so switching this on changes **no spike** — only the bill. Left settable so a
    /// reader can have the bias-free count back and see the difference it makes.
    pub bias: bool,
}

impl Spec {
    /// Spikformer's stated hyperparameters at a given shape, with four-times `MLP` expansion.
    #[must_use]
    pub fn spikformer_like(tokens: usize, d_model: usize, heads: usize, depth: usize) -> Self {
        Self {
            tokens,
            d_model,
            heads,
            mlp_hidden: d_model * 4,
            depth,
            tau: 2.0,
            v_th: 1.0,
            v_reset: 0.0,
            scale: 0.125,
            fold_scale: false,
            order: Order::ScoresFirst,
            residual: Residual::SpikeAdd,
            gain: 6.0,
            bias: true,
        }
    }

    /// The folded-`BatchNorm` shift for one projection: zero at initialisation, or absent.
    fn bias_for(&self, n_out: usize) -> Option<Vec<f64>> {
        if self.bias { Some(vec![0.0; n_out]) } else { None }
    }

    /// Build a model with weights drawn from `rng`.
    ///
    /// Deterministic: the same seed gives the same weights and therefore the same spikes on every
    /// platform, which is what makes any result here checkable against any other.
    ///
    /// # Errors
    ///
    /// Anything [`Linear::new`], [`Ssa::new`], [`SpikingMlp::new`], [`Block::new`] or
    /// [`Model::new`] returns.
    pub fn build(&self, rng: &mut Rng, position: Position) -> Result<Model, AttnError> {
        let mut blocks = Vec::with_capacity(self.depth);
        for _ in 0..self.depth {
            let d = self.d_model;
            let wq = Linear::new(d, d, draw(self.gain, d, d, rng), self.bias_for(d))?;
            let wk = Linear::new(d, d, draw(self.gain, d, d, rng), self.bias_for(d))?;
            let wv = Linear::new(d, d, draw(self.gain, d, d, rng), self.bias_for(d))?;
            let wo = Linear::new(d, d, draw(self.gain, d, d, rng), self.bias_for(d))?;
            let attn = Ssa::new(
                self.tokens,
                self.heads,
                wq,
                wk,
                wv,
                wo,
                self.tau,
                self.v_th,
                self.v_reset,
                self.scale,
                self.fold_scale,
                self.order,
            )?;
            let fc1 = Linear::new(
                d,
                self.mlp_hidden,
                draw(self.gain, d, self.mlp_hidden, rng),
                self.bias_for(self.mlp_hidden),
            )?;
            let fc2 = Linear::new(
                self.mlp_hidden,
                d,
                draw(self.gain, self.mlp_hidden, d, rng),
                self.bias_for(d),
            )?;
            let mlp = SpikingMlp::new(self.tokens, fc1, fc2, self.tau, self.v_th, self.v_reset)?;
            blocks.push(Block::new(attn, mlp, self.residual)?);
        }
        Model::new(self.tokens, position, blocks)
    }
}

/// Uniform weights in `±gain / sqrt(fan_in)`, drawn deterministically from `rng`.
fn draw(gain: f64, n_in: usize, n_out: usize, rng: &mut Rng) -> Vec<f64> {
    let a = gain / (n_in as f64).sqrt();
    (0..n_in * n_out).map(|_| (rng.next_f64() * 2.0 - 1.0) * a).collect()
}

#[cfg(test)]
mod tests {
    use super::{
        Audit, AttnError, Block, DepthwiseConv1d, Domain, LifLayer, Linear, Model, OpKind, Order,
        Position, Residual, Spec, SpikingMlp, Ssa, Tensor, attend, is_exact_power_of_two,
        rate_readout,
    };
    use crate::rng::Rng;

    fn eye(n: usize) -> Vec<f64> {
        let mut w = vec![0.0; n * n];
        for i in 0..n {
            w[i * n + i] = 1.0;
        }
        w
    }

    fn random_spikes(tokens: usize, channels: usize, p: f64, rng: &mut Rng) -> Tensor {
        let bits: Vec<bool> = (0..tokens * channels).map(|_| rng.next_f64() < p).collect();
        Tensor::spikes(tokens, channels, &bits).expect("valid spikes")
    }

    /// The same toy with Spikformer's additive shortcut, so the residual stream leaves
    /// [`Domain::Binary`] at the first shortcut and the `MLP`'s first layer is a live
    /// multiply-accumulate — the one shape the audit's `MAC` paths can be read on by hand.
    fn spikeadd_toy() -> Model {
        let d = 2;
        let lin = || Linear::new(d, d, eye(d), None).expect("square identity");
        let attn = Ssa::new(
            2,
            1,
            lin(),
            lin(),
            lin(),
            lin(),
            2.0,
            0.5,
            0.0,
            1.0,
            false,
            Order::ScoresFirst,
        )
        .expect("toy attention");
        let mlp = SpikingMlp::new(2, lin(), lin(), 2.0, 0.5, 0.0).expect("toy mlp");
        let block = Block::new(attn, mlp, Residual::SpikeAdd).expect("toy block");
        Model::new(2, Position::None, vec![block]).expect("toy model")
    }

    /// The reason text a site of this path and kind must carry, written out as literals so that a
    /// swap or a rewrite in the source fails here rather than passing a length check.
    fn expected_reason(path: &str, kind: OpKind) -> &'static str {
        if path.ends_with(".bias") {
            return "a bias is added to every output whether or not anything fired";
        }
        if path.ends_with(".membrane") {
            return "membrane update terms: the leak difference and the sum back into v";
        }
        if path.ends_with(".leak") {
            return "the membrane leak factor 1/tau multiplies the whole difference";
        }
        if path.ends_with(".threshold") {
            return "one threshold comparison per neuron per timestep";
        }
        if path.ends_with(".scale") {
            return "the attention scale multiplies every element of the attended output";
        }
        match kind {
            OpKind::Ac => {
                "the activation operand is binary, so the weight is gated rather than multiplied"
            }
            OpKind::Mac => {
                "neither operand is binary: the activation is multi-valued and needs a multiplier"
            }
            OpKind::Add => "a shortcut adds two tensors elementwise, widening the domain",
            OpKind::Logic => "a shortcut combines two binary tensors with one gate per element",
            OpKind::Mul | OpKind::Shift | OpKind::Compare => {
                panic!("{path} carries a bare {kind} with no reason of its own")
            }
        }
    }

    /// The hand-counted toy: 2 tokens, 2 channels, 1 head, 1 block, identity weights everywhere,
    /// no bias, no positional encoder, no shortcut, scale exactly 1 (so it is not an operation),
    /// `tau = 2` (so the leak is a shift) and `v_th = 0.5` (so a single unit input fires).
    fn toy() -> Model {
        let d = 2;
        let lin = || Linear::new(d, d, eye(d), None).expect("square identity");
        let attn = Ssa::new(
            2,
            1,
            lin(),
            lin(),
            lin(),
            lin(),
            2.0,
            0.5,
            0.0,
            1.0,
            false,
            Order::ScoresFirst,
        )
        .expect("toy attention");
        let mlp = SpikingMlp::new(2, lin(), lin(), 2.0, 0.5, 0.0).expect("toy mlp");
        let block = Block::new(attn, mlp, Residual::None).expect("toy block");
        Model::new(2, Position::None, vec![block]).expect("toy model")
    }

    // ---------------------------------------------------------------------------------------
    // (a) The audit against a model whose operation split was computed on paper.
    // ---------------------------------------------------------------------------------------

    /// ⭐ THE CENTRAL TEST. Every number below was counted by hand before the code ran.
    ///
    /// Eight product sites, each `2 tokens × 2 in × 2 out = 8` dense: `q/k/v/out` projections,
    /// `Q Kᵀ`, `S V`, and the `MLP`'s two layers. `8 × 8 = 64` dense accumulates.
    ///
    /// With identity weights and the input `[[1,0],[0,1]]`, exactly one input channel is set per
    /// token and exactly one weight is non-zero in its column, so each site does `2` effective
    /// operations: `8 × 2 = 16`.
    ///
    /// Seven neuron layers (`q`, `k`, `v`, attention, out, `MLP` 1, `MLP` 2) of `2 × 2 = 4`
    /// neurons: 28 neurons. Each pays 2 adds (`v_reset` is zero, so `V - v_reset` vanishes), 1
    /// shift (`1/tau = 0.5`) and 1 comparison: 56 adds, 28 shifts, 28 comparisons.
    ///
    /// Nothing else: no bias, no shortcut, no positional encoder, and a scale of exactly 1 is not
    /// an operation.
    #[test]
    fn the_operation_split_matches_a_hand_count_on_a_two_token_model() {
        let mut m = toy();
        let x = Tensor::spikes(2, 2, &[true, false, false, true]).unwrap();
        let mut audit = Audit::new();
        let y = m.forward(&x, &mut audit).unwrap();

        // The model is not silent — without this the split below could be a table of zeros.
        assert_eq!(y.values(), &[1.0, 0.0, 0.0, 1.0], "the toy must pass the identity through");
        assert_eq!(audit.timesteps(), 1);

        let s = audit.split();
        assert_eq!(s.accumulates, 16, "accumulates");
        assert_eq!(s.multiply_accumulates, 0, "nothing here has two real operands");
        assert_eq!(s.adds, 56, "2 per neuron per timestep, 28 neurons");
        assert_eq!(s.multiplies, 0, "the leak is a power of two and the scale is one");
        assert_eq!(s.shifts, 28, "1 leak per neuron per timestep");
        assert_eq!(s.comparisons, 28, "1 threshold per neuron per timestep");
        assert_eq!(s.logic, 0, "no shortcut");
        assert_eq!(s.total(), Some(128));
        assert_eq!(audit.dense_of(OpKind::Ac), 64, "8 sites of 8 dense operations");

        // ⭐ THE FINDING, as two numbers that do not agree.
        assert_eq!(audit.synaptic_ac_fraction(), Some(1.0), "as the literature reports it");
        assert_eq!(audit.ac_fraction(), Some(0.125), "counting everything the pass performed");
    }

    /// The same hand count, one site at a time, so that a change to any single site's rule fails
    /// here rather than being absorbed into a total.
    #[test]
    fn every_site_in_the_toy_carries_the_count_it_was_hand_given() {
        let mut m = toy();
        let x = Tensor::spikes(2, 2, &[true, false, false, true]).unwrap();
        let mut audit = Audit::new();
        m.forward(&x, &mut audit).unwrap();
        for path in [
            "block0.attn.q_proj",
            "block0.attn.k_proj",
            "block0.attn.v_proj",
            "block0.attn.ssa.scores",
            "block0.attn.ssa.attend",
            "block0.attn.out_proj",
            "block0.mlp.fc1",
            "block0.mlp.fc2",
        ] {
            let site = audit.site(path, OpKind::Ac).unwrap_or_else(|| panic!("{path} missing"));
            assert_eq!(site.dense, 8, "{path} dense");
            assert_eq!(site.effective, 2, "{path} effective");
        }
        for path in [
            "block0.attn.q_neuron",
            "block0.attn.k_neuron",
            "block0.attn.v_neuron",
            "block0.attn.attn_neuron",
            "block0.attn.out_neuron",
            "block0.mlp.neuron1",
            "block0.mlp.neuron2",
        ] {
            assert_eq!(audit.site(&format!("{path}.membrane"), OpKind::Add).unwrap().effective, 8);
            assert_eq!(audit.site(&format!("{path}.leak"), OpKind::Shift).unwrap().effective, 4);
            assert_eq!(
                audit.site(&format!("{path}.threshold"), OpKind::Compare).unwrap().effective,
                4
            );
        }
        // A scale of exactly one is charged nothing, and this is what says so.
        assert!(audit.site("block0.attn.scale", OpKind::Shift).is_none());
        assert!(audit.site("block0.attn.scale", OpKind::Mul).is_none());
    }

    // ---------------------------------------------------------------------------------------
    // (b) Exact attention behaviour.
    // ---------------------------------------------------------------------------------------

    /// A query matching exactly one key returns that key's value row, exactly, with no tolerance.
    /// Checked under both association orders, because they are different code paths.
    #[test]
    fn a_one_hot_key_attends_to_exactly_the_matching_value() {
        // One query, firing only on channel 2.
        let q = Tensor::spikes(1, 4, &[false, false, true, false]).unwrap();
        // Three keys; only key 1 fires on channel 2.
        let k = Tensor::spikes(
            3,
            4,
            &[true, false, false, false, false, false, true, false, false, true, false, false],
        )
        .unwrap();
        // Three distinct value rows.
        let v =
            Tensor::spikes(3, 3, &[true, false, true, false, true, true, true, true, false])
                .unwrap();
        for order in [Order::ScoresFirst, Order::ValuesFirst] {
            let mut audit = Audit::new();
            let o = attend(&q, &k, &v, order, "t", &mut audit).unwrap();
            assert_eq!(o.tokens(), 1);
            assert_eq!(o.channels(), 3);
            assert_eq!(o.values(), v.row(1).unwrap(), "{order:?} did not select key 1's value");
            // And it is NOT one of the other rows, so the assertion above is discriminating.
            assert_ne!(o.values(), v.row(0).unwrap());
            assert_ne!(o.values(), v.row(2).unwrap());
        }
    }

    /// Two keys matching means the two value rows are summed — there is no softmax to normalise
    /// them, which is the property the scale and the following threshold exist to tame.
    #[test]
    fn two_matching_keys_sum_rather_than_average_their_values() {
        let q = Tensor::spikes(1, 2, &[true, false]).unwrap();
        let k = Tensor::spikes(2, 2, &[true, false, true, true]).unwrap();
        let v = Tensor::spikes(2, 2, &[true, false, true, true]).unwrap();
        let mut audit = Audit::new();
        let o = attend(&q, &k, &v, Order::ScoresFirst, "t", &mut audit).unwrap();
        // Both keys score 1, so the output is v[0] + v[1] = [2, 1] — not an average of [1, 0.5].
        assert_eq!(o.values(), &[2.0, 1.0]);
    }

    /// (f) The two association orders are the same function and a different bill.
    #[test]
    fn the_two_association_orders_agree_exactly_and_cost_differently() {
        let mut rng = Rng::new(9);
        // Short sequence, wide head: materialising the scores is the cheaper order.
        let q = random_spikes(3, 4, 0.5, &mut rng);
        let k = random_spikes(3, 4, 0.5, &mut rng);
        let v = random_spikes(3, 5, 0.5, &mut rng);
        let mut a1 = Audit::new();
        let mut a2 = Audit::new();
        let o1 = attend(&q, &k, &v, Order::ScoresFirst, "t", &mut a1).unwrap();
        let o2 = attend(&q, &k, &v, Order::ValuesFirst, "t", &mut a2).unwrap();
        assert_eq!(o1.values(), o2.values(), "the orders disagreed");
        assert!(o1.nonzero() > 0, "both orders produced nothing; the comparison is vacuous");
        assert_eq!(a1.dense_of(OpKind::Ac), 3 * 3 * 4 + 3 * 3 * 5, "scores-first dense");
        assert_eq!(a2.dense_of(OpKind::Ac), 4 * 3 * 5 + 3 * 4 * 5, "values-first dense");
        assert!(a1.dense_of(OpKind::Ac) < a2.dense_of(OpKind::Ac));

        // Long sequence, narrow head: the order flips.
        let q = random_spikes(16, 2, 0.5, &mut rng);
        let k = random_spikes(16, 2, 0.5, &mut rng);
        let v = random_spikes(16, 2, 0.5, &mut rng);
        let mut b1 = Audit::new();
        let mut b2 = Audit::new();
        let p1 = attend(&q, &k, &v, Order::ScoresFirst, "t", &mut b1).unwrap();
        let p2 = attend(&q, &k, &v, Order::ValuesFirst, "t", &mut b2).unwrap();
        assert_eq!(p1.values(), p2.values());
        assert_eq!(b1.dense_of(OpKind::Ac), 16 * 16 * 2 + 16 * 16 * 2);
        assert_eq!(b2.dense_of(OpKind::Ac), 2 * 16 * 2 + 16 * 2 * 2);
        assert!(b2.dense_of(OpKind::Ac) * 4 < b1.dense_of(OpKind::Ac), "the flip is eightfold");
    }

    // ---------------------------------------------------------------------------------------
    // (c) The residual path.
    // ---------------------------------------------------------------------------------------

    fn zeroed_block(residual: Residual) -> Block {
        let d = 2;
        let zero = || Linear::new(d, d, vec![0.0; d * d], None).unwrap();
        let attn = Ssa::new(
            2,
            1,
            zero(),
            zero(),
            zero(),
            zero(),
            2.0,
            0.5,
            0.0,
            1.0,
            false,
            Order::ScoresFirst,
        )
        .unwrap();
        let mlp = SpikingMlp::new(2, zero(), zero(), 2.0, 0.5, 0.0).unwrap();
        Block::new(attn, mlp, residual).unwrap()
    }

    /// (c) With every branch weight zero, the shortcut is the identity on the values — and under
    /// `SpikeAdd` the *declared domain widens anyway*, which is the whole cost of that shortcut.
    #[test]
    fn a_zeroed_branch_leaves_the_residual_stream_unchanged() {
        let x = Tensor::spikes(2, 2, &[true, false, true, true]).unwrap();

        let mut add = zeroed_block(Residual::SpikeAdd);
        let mut audit = Audit::new();
        let y = add.forward(&x, "b", &mut audit).unwrap();
        assert_eq!(y.values(), x.values(), "SpikeAdd shortcut was not the identity");
        assert_eq!(
            y.domain(),
            Domain::Integer(3),
            "two shortcuts widened the declared domain even though no value changed"
        );

        let mut iand = zeroed_block(Residual::SewIand);
        let mut audit = Audit::new();
        let y = iand.forward(&x, "b", &mut audit).unwrap();
        assert_eq!(y.values(), x.values(), "IAND shortcut was not the identity");
        assert_eq!(y.domain(), Domain::Binary, "IAND must keep the stream binary");

        // And without a shortcut the block returns the branch, which here is silence — so the two
        // assertions above are about the shortcut and not about the zeroed weights.
        let mut none = zeroed_block(Residual::None);
        let mut audit = Audit::new();
        let y = none.forward(&x, "b", &mut audit).unwrap();
        assert_eq!(y.values(), &[0.0; 4]);
    }

    /// ⭐ The criticism Yao et al. make, as a test: one additive shortcut and every projection
    /// after it needs a multiplier.
    #[test]
    fn the_spike_add_shortcut_turns_every_downstream_projection_into_a_multiply_accumulate() {
        let spec = Spec {
            residual: Residual::SpikeAdd,
            ..Spec::spikformer_like(4, 4, 2, 2)
        };
        let mut m = spec.build(&mut Rng::new(3), Position::None).unwrap();
        let x = random_spikes(4, 4, 0.5, &mut Rng::new(77));
        let mut audit = Audit::new();
        m.forward(&x, &mut audit).unwrap();

        // Before the first shortcut the input is still binary.
        assert!(audit.site("block0.attn.q_proj", OpKind::Ac).is_some());
        assert!(audit.site("block0.attn.q_proj", OpKind::Mac).is_none());
        // After it, everything needs a multiplier.
        assert!(audit.site("block0.mlp.fc1", OpKind::Mac).is_some(), "the MLP escaped the widening");
        assert!(audit.site("block0.mlp.fc1", OpKind::Ac).is_none());
        assert!(audit.site("block1.attn.q_proj", OpKind::Mac).is_some());
        assert!(audit.multiplier_ops() > 0);
        let f = audit.synaptic_ac_fraction().unwrap();
        assert!(f < 1.0, "a SpikeAdd model reported itself fully spike-driven at {f}");
    }

    /// And the gated shortcut keeps the promise: not one multiplier anywhere in the model.
    #[test]
    fn an_iand_shortcut_keeps_the_whole_model_multiplier_free() {
        let spec = Spec { residual: Residual::SewIand, ..Spec::spikformer_like(4, 4, 2, 3) };
        let mut m = spec.build(&mut Rng::new(3), Position::None).unwrap();
        let x = random_spikes(4, 4, 0.5, &mut Rng::new(77));
        let mut audit = Audit::new();
        let y = m.forward(&x, &mut audit).unwrap();
        assert_eq!(y.domain(), Domain::Binary);
        assert_eq!(audit.dense_of(OpKind::Mac), 0, "a multiply-accumulate site appeared");
        assert_eq!(audit.effective_of(OpKind::Mul), 0, "a bare multiply appeared");
        assert_eq!(audit.synaptic_ac_fraction(), Some(1.0));
        // Which is exactly the claim that the honest fraction contradicts.
        assert!(audit.ac_fraction().unwrap() < 0.5, "{}", audit.ac_fraction().unwrap());
    }

    // ---------------------------------------------------------------------------------------
    // (d) Shapes compose.
    // ---------------------------------------------------------------------------------------

    /// Three blocks, five timesteps, and the declared domain grows by exactly two per block —
    /// one per shortcut — which is a closed form for how far the residual stream is from binary.
    #[test]
    fn shapes_compose_across_a_three_block_model() {
        let spec = Spec { mlp_hidden: 8, ..Spec::spikformer_like(4, 8, 2, 3) };
        let mut m = spec.build(&mut Rng::new(11), Position::None).unwrap();
        assert_eq!(m.depth(), 3);
        assert_eq!(m.d_model(), 8);
        let mut rng = Rng::new(12);
        let inputs: Vec<Tensor> = (0..5).map(|_| random_spikes(4, 8, 0.4, &mut rng)).collect();
        let (outs, audit) = m.run(&inputs).unwrap();
        assert_eq!(outs.len(), 5);
        assert_eq!(audit.timesteps(), 5);
        for o in &outs {
            assert_eq!(o.tokens(), 4);
            assert_eq!(o.channels(), 8);
            assert_eq!(o.domain(), Domain::Integer(7), "1 + 2 shortcuts per block × 3 blocks");
        }
        let rates = rate_readout(&outs).unwrap();
        assert_eq!(rates.len(), 8);
        assert!(rates.iter().any(|r| *r > 0.0), "the whole run was silent");
    }

    /// A model built for two tokens refuses three, naming both shapes, rather than resizing its
    /// membranes and carrying one sequence's state into the next.
    /// A model built for two tokens refuses three at **its own** boundary, not several frames
    /// deeper.
    ///
    /// The `what` field is the whole assertion. With [`Model::forward`]'s guard removed the input
    /// reaches [`Ssa::forward`], whose own guard reports
    /// `BadShape { what: "attention input", got: 6, want: 4 }` — the identical pair of numbers —
    /// so a `matches!` that elided `what` passed either way. A caller told "attention input" goes
    /// looking inside a block for a shape the model was never built to take.
    #[test]
    fn a_model_built_for_two_tokens_refuses_three() {
        let mut m = toy();
        let wide = Tensor::spikes(3, 2, &[true; 6]).unwrap();
        let err = m.forward(&wide, &mut Audit::new()).unwrap_err();
        assert!(
            matches!(err, AttnError::BadShape { what: "model input", got: 6, want: 4 }),
            "{err}"
        );
        // And the channel count, which the token count alone would not have caught: the same six
        // elements arranged the other way round are refused for the other reason.
        let narrow = Tensor::spikes(2, 3, &[true; 6]).unwrap();
        let err = m.forward(&narrow, &mut Audit::new()).unwrap_err();
        assert!(
            matches!(err, AttnError::BadShape { what: "model input", got: 6, want: 4 }),
            "{err}"
        );
    }

    // ---------------------------------------------------------------------------------------
    // (e) Silence.
    // ---------------------------------------------------------------------------------------

    /// ⭐ Zero spikes in, zero out, zero accumulates — **and 112 operations anyway**.
    ///
    /// The neuron updates are not gated on activity, so a model that does nothing still pays its
    /// membrane arithmetic in full. A synaptic operation count reports this workload as free.
    #[test]
    fn a_silent_input_costs_zero_accumulates_and_not_zero_operations() {
        let mut m = toy();
        let x = Tensor::silent(2, 2).unwrap();
        let mut audit = Audit::new();
        let y = m.forward(&x, &mut audit).unwrap();
        assert_eq!(y.values(), &[0.0; 4], "a silent input produced a spike");
        assert_eq!(y.nonzero(), 0);

        let s = audit.split();
        assert_eq!(s.accumulates, 0, "a silent input performed an accumulate");
        assert_eq!(s.multiply_accumulates, 0);
        assert_eq!(s.adds, 56, "the membranes updated anyway");
        assert_eq!(s.shifts, 28);
        assert_eq!(s.comparisons, 28);
        assert_eq!(s.total(), Some(112));
        assert_eq!(audit.ac_fraction(), Some(0.0));
        // Not Some(1.0): a run with no synaptic operations is not perfectly spike-driven, it is
        // a run that did not answer the question.
        assert_eq!(audit.synaptic_ac_fraction(), None);
        // The dense counts are unchanged, because dense is a property of the model, not the data.
        assert_eq!(audit.dense_of(OpKind::Ac), 64);
    }

    // ---------------------------------------------------------------------------------------
    // The scale fold.
    // ---------------------------------------------------------------------------------------

    fn folding_spec(fold: bool) -> Spec {
        Spec {
            mlp_hidden: 8,
            scale: 0.125,
            fold_scale: fold,
            residual: Residual::SewIand,
            ..Spec::spikformer_like(4, 8, 2, 1)
        }
    }

    /// Folding `s` into the threshold changes no spike and removes every multiply it was paying
    /// for. Compared bit for bit across eight timesteps, because the claim is exactness.
    #[test]
    fn folding_the_scale_into_the_threshold_changes_no_spike_and_removes_every_multiply() {
        let mut plain = folding_spec(false).build(&mut Rng::new(5), Position::None).unwrap();
        let mut folded = folding_spec(true).build(&mut Rng::new(5), Position::None).unwrap();
        assert!(!plain.blocks[0].attn.folds_scale());
        assert!(folded.blocks[0].attn.folds_scale());

        let mut rng = Rng::new(6);
        let inputs: Vec<Tensor> = (0..8).map(|_| random_spikes(4, 8, 0.5, &mut rng)).collect();
        let (a_out, a_audit) = plain.run(&inputs).unwrap();
        let (b_out, b_audit) = folded.run(&inputs).unwrap();

        let spikes: u64 = a_out.iter().map(Tensor::nonzero).sum();
        assert!(spikes > 0, "both models were silent; the comparison proves nothing");
        for (a, b) in a_out.iter().zip(&b_out) {
            assert_eq!(a.values(), b.values(), "the fold changed a spike");
        }

        // The unfolded model pays one shift per attended element per timestep: 4 tokens × 8
        // channels × 8 timesteps = 256 dense, and it is the ONLY such site.
        let site = a_audit.site("block0.attn.scale", OpKind::Shift).expect("the scale is charged");
        assert_eq!(site.dense, 256);
        assert!(site.effective > 0, "the scale site never fired; its removal would cost nothing");
        assert!(b_audit.site("block0.attn.scale", OpKind::Shift).is_none(), "the fold left a site");
        assert_eq!(
            a_audit.effective_of(OpKind::Shift) - b_audit.effective_of(OpKind::Shift),
            site.effective,
            "the fold removed something other than the scale"
        );
        // Every other kind is untouched.
        assert_eq!(a_audit.split().accumulates, b_audit.split().accumulates);
        assert_eq!(a_audit.split().comparisons, b_audit.split().comparisons);
    }

    /// The fold is refused where it would not be bit-exact, rather than performed approximately.
    #[test]
    fn the_fold_is_refused_when_it_would_not_be_exact() {
        let d = 2;
        let lin = || Linear::new(d, d, eye(d), None).unwrap();
        let build = |scale: f64, v_reset: f64| {
            Ssa::new(
                2,
                1,
                lin(),
                lin(),
                lin(),
                lin(),
                2.0,
                1.0,
                v_reset,
                scale,
                true,
                Order::ScoresFirst,
            )
        };
        let e = build(0.1, 0.0).unwrap_err();
        assert!(matches!(e, AttnError::UnfoldableScale { .. }), "{e}");
        assert!(format!("{e}").contains("power of two"));
        let e = build(0.125, -0.2).unwrap_err();
        assert!(format!("{e}").contains("v_reset"), "{e}");
        // And the exactly-representable case is accepted.
        assert!(build(0.125, 0.0).is_ok());
        // A scale that is not a positive finite number is refused with or without the fold.
        assert!(matches!(
            Ssa::new(2, 1, lin(), lin(), lin(), lin(), 2.0, 1.0, 0.0, 0.0, false, Order::ScoresFirst),
            Err(AttnError::BadParameter { .. })
        ));
    }

    // ---------------------------------------------------------------------------------------
    // The neuron, against its closed form.
    // ---------------------------------------------------------------------------------------

    /// The membrane relaxes geometrically toward `v_reset` with ratio `1 - 1/tau`, and the
    /// simulation must follow the closed form rather than a previous run of itself.
    ///
    /// `tau = 3` deliberately: `1/3` is not a power of two, so this also pins the audit's choice of
    /// [`OpKind::Mul`] over [`OpKind::Shift`].
    #[test]
    fn the_membrane_decays_geometrically_toward_its_reset() {
        let mut n = LifLayer::new(1, 1, 3.0, 10.0, 0.0).unwrap();
        let drive = Tensor::real(1, 1, vec![0.9]).unwrap();
        let mut audit = Audit::new();
        let fired = n.forward(&drive, "n", &mut audit).unwrap();
        assert_eq!(fired.values(), &[0.0], "the threshold is 10; nothing should have fired");
        let v0 = n.membranes()[0];
        assert!((v0 - 0.3).abs() < 1e-15, "one third of 0.9 is 0.3, got {v0}");

        let silent = Tensor::real(1, 1, vec![0.0]).unwrap();
        for step in 1..=6u32 {
            n.forward(&silent, "n", &mut audit).unwrap();
            let want = n.relaxed(v0, step);
            let got = n.membranes()[0];
            assert!((got - want).abs() < 1e-14, "step {step}: {got} vs closed form {want}");
        }
        // The closed form is not a restatement of the loop: check it against an independent power.
        let want = 0.3 * (2.0f64 / 3.0).powi(6);
        assert!((n.membranes()[0] - want).abs() < 1e-14, "{}", n.membranes()[0]);
        // And the decay actually moved — otherwise the comparison above is between two constants.
        assert!(n.membranes()[0] < 0.1 * v0, "the membrane barely decayed");
        // A leak of 1/3 is not a shift.
        assert!(audit.site("n.leak", OpKind::Mul).is_some());
        assert!(audit.site("n.leak", OpKind::Shift).is_none());
    }

    /// A neuron with three adds per update when `v_reset` is non-zero, two when it is zero. The
    /// difference is the `V - v_reset` term, and a model that charged three either way would
    /// over-bill every Spikformer-style network by one add per neuron per timestep.
    #[test]
    fn the_membrane_add_count_follows_the_reset_potential() {
        let x = Tensor::real(1, 4, vec![0.0; 4]).unwrap();
        let mut zero = LifLayer::new(1, 4, 2.0, 1.0, 0.0).unwrap();
        let mut offset = LifLayer::new(1, 4, 2.0, 1.0, -0.5).unwrap();
        let mut a = Audit::new();
        let mut b = Audit::new();
        zero.forward(&x, "n", &mut a).unwrap();
        offset.forward(&x, "n", &mut b).unwrap();
        assert_eq!(a.effective_of(OpKind::Add), 8);
        assert_eq!(b.effective_of(OpKind::Add), 12);
    }

    /// A neuron whose threshold is not above its reset would fire on every timestep including a
    /// silent one, which would make every sparsity figure in the crate meaningless.
    #[test]
    fn a_neuron_with_an_unreachable_threshold_is_refused() {
        assert!(LifLayer::new(1, 1, 2.0, 0.0, 0.0).is_err(), "v_th == v_reset was accepted");
        assert!(LifLayer::new(1, 1, 0.5, 1.0, 0.0).is_err(), "tau < 1 was accepted");
        assert!(LifLayer::new(1, 1, f64::NAN, 1.0, 0.0).is_err());
        assert!(LifLayer::new(1, 1, 2.0, f64::INFINITY, 0.0).is_err());
    }

    // ---------------------------------------------------------------------------------------
    // Domains, power of two, and the metric bridge.
    // ---------------------------------------------------------------------------------------

    #[test]
    fn a_power_of_two_is_recognised_exactly() {
        for x in [0.125, 0.25, 0.5, 1.0, 2.0, 8.0, 1024.0, 2f64.powi(-60)] {
            assert!(is_exact_power_of_two(x), "{x} is a power of two");
        }
        for x in [0.1, 0.3, 3.0, 1.5, 0.0, -2.0, f64::NAN, f64::INFINITY, 2f64.powi(-1074)] {
            assert!(!is_exact_power_of_two(x), "{x} is not a positive normal power of two");
        }
    }

    /// An integer domain a binary datapath would have handled is a binary domain, and the
    /// promotion happens at construction so no site can charge a multiplier for it.
    #[test]
    fn an_integer_domain_bounded_by_one_is_binary() {
        let t = Tensor::new(1, 2, vec![1.0, 0.0], Domain::Integer(1)).unwrap();
        assert_eq!(t.domain(), Domain::Binary);
        let t = Tensor::new(1, 2, vec![0.0, 0.0], Domain::Integer(0)).unwrap();
        assert_eq!(t.domain(), Domain::Binary);
        assert_eq!(OpKind::product(t.domain(), Domain::Real), OpKind::Ac);
        // But a bound of two is not, even when the sample happens to be binary.
        let t = Tensor::new(1, 2, vec![1.0, 0.0], Domain::Integer(2)).unwrap();
        assert_eq!(t.domain(), Domain::Integer(2));
        assert_eq!(OpKind::product(t.domain(), Domain::Real), OpKind::Mac);
    }

    #[test]
    fn summing_domains_widens_the_bound_and_never_narrows_it() {
        assert_eq!(Domain::Binary.sum(Domain::Binary), Domain::Integer(2));
        assert_eq!(Domain::Integer(2).sum(Domain::Binary), Domain::Integer(3));
        assert_eq!(Domain::Integer(3).sum(Domain::Integer(4)), Domain::Integer(7));
        assert_eq!(Domain::Real.sum(Domain::Binary), Domain::Real);
        // Saturating, because a wrapped bound would turn a wide operand back into a narrow one.
        // Both arms saturate: two already-wide integer streams meet at a second `SpikeAdd`
        // shortcut, which is precisely the case the saturation comment warns about.
        assert_eq!(Domain::Integer(u32::MAX).sum(Domain::Binary), Domain::Integer(u32::MAX));
        assert_eq!(Domain::Integer(u32::MAX).sum(Domain::Integer(2)), Domain::Integer(u32::MAX));
        assert_eq!(Domain::Integer(u32::MAX - 1).sum(Domain::Integer(3)), Domain::Integer(u32::MAX));
        assert_eq!(Domain::Binary.bound(), Some(1));
        assert_eq!(Domain::Integer(5).bound(), Some(5), "the bound is the one it carries");
        assert_eq!(Domain::Integer(u32::MAX).bound(), Some(u32::MAX));
        assert_eq!(Domain::Real.bound(), None);
    }

    /// ⭐ How much of a spiking transformer's arithmetic `NeuroBench`'s own metric cannot see.
    #[test]
    fn the_neurobench_metric_cannot_see_most_of_the_bill() {
        let mut m = toy();
        let x = Tensor::spikes(2, 2, &[true, false, false, true]).unwrap();
        let mut audit = Audit::new();
        m.forward(&x, &mut audit).unwrap();
        let s = audit.synops();
        assert_eq!(s.dense, 64);
        assert_eq!(s.effective_acs, 16);
        assert_eq!(s.effective_macs, 0);
        assert_eq!(s.effective_total(), Some(16));
        assert_eq!(s.ac_fraction(), Some(1.0), "SynOps agrees with the published claim");
        // And the audit says the pass performed 128 operations, so 112 of them — seven eighths —
        // have no slot in the metric that decides the benchmark.
        assert_eq!(audit.effective_total(), Some(128));
        assert_eq!(audit.effective_total().unwrap() - s.effective_total().unwrap(), 112);
    }

    /// ⭐ The headline, on a model built from Spikformer's own stated hyperparameters.
    #[test]
    fn the_honest_fraction_is_lower_than_the_one_the_papers_report() {
        let spec = Spec { residual: Residual::SewIand, ..Spec::spikformer_like(8, 16, 4, 2) };
        let mut m = spec.build(&mut Rng::new(21), Position::None).unwrap();
        let mut rng = Rng::new(22);
        let inputs: Vec<Tensor> = (0..4).map(|_| random_spikes(8, 16, 0.3, &mut rng)).collect();
        let (outs, audit) = m.run(&inputs).unwrap();
        let spikes: u64 = outs.iter().map(Tensor::nonzero).sum();
        assert!(spikes > 0, "the model was silent, so the fractions below mean nothing");

        let reported = audit.synaptic_ac_fraction().unwrap();
        let honest = audit.ac_fraction().unwrap();
        assert_eq!(reported, 1.0, "this model really is fully spike-driven at its synapses");
        // …and that is an architectural fact about THIS model, not an accident of this input. The
        // line above passes just as well on a model whose multiply-accumulate layers merely stayed
        // silent, which is a different claim; `a_reported_fraction_of_one_is_not_a_multiplier_free_architecture`
        // is that case.
        assert_eq!(audit.dense_of(OpKind::Mac), 0, "a dense MAC site exists in this model");
        assert!(honest < reported, "the two fractions agreed, which cannot happen here");
        assert!(honest > 0.0);
        // The report a user pastes must carry both numbers, not the flattering one alone.
        let text = audit.to_string();
        assert!(text.contains("AC share of ALL operations"), "{text}");
        assert!(text.contains("AC share of synaptic operations"), "{text}");
    }

    // ---------------------------------------------------------------------------------------
    // Positional information.
    // ---------------------------------------------------------------------------------------

    /// The sinusoidal table follows the sign of the closed form. The expected pattern below is the
    /// sign of `sin` and `cos` at integer radians, read off their zeros at π, 2π and 3π — not
    /// recomputed from the same expression the code uses.
    #[test]
    fn sinusoidal_position_spikes_follow_the_sign_of_the_closed_form() {
        let t = Tensor::sinusoidal_spikes(10, 2).unwrap();
        assert_eq!(t.row(0).unwrap(), &[1.0, 1.0], "sin(0)=0 and cos(0)=1 are both >= 0");
        let sin_sign = [true, true, true, true, false, false, false, true, true, true];
        let cos_sign = [true, true, false, false, false, true, true, true, false, false];
        for pos in 0..10 {
            assert_eq!(
                t.at(pos, 0).unwrap() == 1.0,
                sin_sign[pos],
                "sin channel at position {pos}"
            );
            assert_eq!(
                t.at(pos, 1).unwrap() == 1.0,
                cos_sign[pos],
                "cos channel at position {pos}"
            );
        }
        // Different positions must be distinguishable, or the encoding encodes nothing.
        assert_ne!(t.row(0).unwrap(), t.row(4).unwrap());
    }

    /// A single centre tap is the identity, which is the only convolution whose output can be
    /// checked without reimplementing the convolution.
    #[test]
    fn a_centre_tap_convolution_is_the_identity() {
        let conv = DepthwiseConv1d::centre_tap(3, 3).unwrap();
        let mut rng = Rng::new(31);
        let x = random_spikes(5, 3, 0.5, &mut rng);
        let mut audit = Audit::new();
        let y = conv.forward(&x, "c", &mut audit).unwrap();
        assert_eq!(y.values(), x.values(), "the centre tap was not the identity");
        assert!(x.nonzero() > 0, "an all-zero input makes the identity check vacuous");
        // Dense charges the padded taps; effective charges only the one that hits.
        assert_eq!(audit.dense_of(OpKind::Ac), 5 * 3 * 3);
        assert_eq!(audit.effective_of(OpKind::Ac), x.nonzero());
        assert!(DepthwiseConv1d::new(3, 2, vec![0.0; 6]).is_err(), "an even kernel has no centre");
    }

    /// The two table encoders differ in exactly the way the docs claim: one widens the stream and
    /// costs an add, the other keeps it binary and costs a gate.
    #[test]
    fn the_additive_position_table_widens_the_stream_and_the_gated_one_does_not() {
        let x = Tensor::spikes(2, 2, &[true, false, false, false]).unwrap();
        let table = Tensor::spikes(2, 2, &[true, true, false, true]).unwrap();

        let mut add = Position::AddTable(table.clone());
        let mut a = Audit::new();
        let y = add.apply(&x, "position", &mut a).unwrap();
        assert_eq!(y.values(), &[2.0, 1.0, 0.0, 1.0]);
        assert_eq!(y.domain(), Domain::Integer(2));
        assert_eq!(a.effective_of(OpKind::Add), 4);

        let mut or = Position::OrTable(table);
        let mut b = Audit::new();
        let y = or.apply(&x, "position", &mut b).unwrap();
        assert_eq!(y.values(), &[1.0, 1.0, 0.0, 1.0]);
        assert_eq!(y.domain(), Domain::Binary);
        assert_eq!(b.effective_of(OpKind::Logic), 4);
        assert_eq!(b.effective_of(OpKind::Add), 0);
    }

    /// The conditional encoder runs, mixes neighbours, and is charged as an accumulate because its
    /// input is binary.
    #[test]
    fn the_conditional_position_encoder_mixes_neighbours() {
        let conv = DepthwiseConv1d::new(2, 3, vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0]).unwrap();
        let neuron = LifLayer::new(3, 2, 2.0, 0.4, 0.0).unwrap();
        let mut p = Position::Conditional(conv, neuron);
        // Only position 1 fires; the kernel reads the PREVIOUS position, so position 2 is affected.
        let x = Tensor::spikes(3, 2, &[false, false, true, true, false, false]).unwrap();
        let mut audit = Audit::new();
        let y = p.apply(&x, "position", &mut audit).unwrap();
        assert_eq!(y.values(), &[0.0, 0.0, 1.0, 1.0, 1.0, 1.0], "the neighbour was not reached");
        assert_eq!(y.domain(), Domain::Integer(2), "Spikformer's RPE adds, so the stream widens");
        assert!(audit.site("position.conv", OpKind::Ac).is_some());
        p.reset_state();
    }

    // ---------------------------------------------------------------------------------------
    // Refusals and determinism.
    // ---------------------------------------------------------------------------------------

    #[test]
    fn a_binary_tensor_refuses_a_value_that_is_not_a_spike() {
        let e = Tensor::new(1, 2, vec![0.5, 1.0], Domain::Binary).unwrap_err();
        assert!(matches!(e, AttnError::NotBinary { index: 0, .. }), "{e}");
        let e = Tensor::new(1, 2, vec![1.0, 3.0], Domain::Integer(2)).unwrap_err();
        assert!(matches!(e, AttnError::OutOfDomain { index: 1, bound: 2, .. }), "{e}");
        let e = Tensor::new(1, 2, vec![1.0, 0.5], Domain::Integer(2)).unwrap_err();
        assert!(matches!(e, AttnError::OutOfDomain { .. }), "a non-integer passed: {e}");
        let e = Tensor::new(1, 2, vec![1.0, -1.0], Domain::Integer(2)).unwrap_err();
        assert!(matches!(e, AttnError::OutOfDomain { .. }), "a negative passed: {e}");
    }

    #[test]
    fn non_finite_inputs_are_refused_at_the_boundary() {
        assert!(matches!(
            Tensor::real(1, 2, vec![1.0, f64::NAN]),
            Err(AttnError::NonFinite { index: 1, .. })
        ));
        assert!(matches!(
            Linear::new(1, 2, vec![1.0, f64::INFINITY], None),
            Err(AttnError::NonFinite { .. })
        ));
        assert!(matches!(
            Linear::new(1, 1, vec![1.0], Some(vec![f64::NAN])),
            Err(AttnError::NonFinite { what: "bias", .. })
        ));
        assert!(matches!(Tensor::real(0, 2, vec![]), Err(AttnError::Empty { .. })));
        assert!(matches!(Linear::new(2, 2, vec![1.0], None), Err(AttnError::BadShape { .. })));
    }

    /// An effective count above its dense count is a bug in this module, and the audit refuses it
    /// rather than reporting a sparsity above one.
    #[test]
    fn the_audit_refuses_an_effective_count_above_its_dense_count() {
        let mut a = Audit::new();
        assert!(a.charge("x", OpKind::Ac, "test", 4, 5).is_err());
        assert!(a.charge("x", OpKind::Ac, "test", 4, 4).is_ok());
        assert_eq!(a.sites().len(), 1, "the refused charge was still recorded");
    }

    /// Every site a real run produces must satisfy the invariant, checked over a model with a
    /// positional encoder, shortcuts, several heads and several timesteps.
    #[test]
    fn effective_never_exceeds_dense_anywhere_in_a_real_run() {
        let spec = Spec { mlp_hidden: 16, ..Spec::spikformer_like(6, 8, 2, 2) };
        let table = Tensor::sinusoidal_spikes(6, 8).unwrap();
        let mut m = spec.build(&mut Rng::new(41), Position::OrTable(table)).unwrap();
        let mut rng = Rng::new(42);
        let inputs: Vec<Tensor> = (0..6).map(|_| random_spikes(6, 8, 0.35, &mut rng)).collect();
        let (outs, audit) = m.run(&inputs).unwrap();
        assert!(!audit.sites().is_empty());
        for s in audit.sites() {
            assert!(s.effective <= s.dense, "{} {} {}/{}", s.path, s.kind, s.effective, s.dense);
            // Not `len() > 20`. `because` is a content field — it names WHICH operand decided the
            // kind — and a length check passes for any string of any meaning, including the
            // opposite one.
            assert_eq!(
                s.because,
                expected_reason(&s.path, s.kind),
                "{} ({}) carries the wrong reason",
                s.path,
                s.kind
            );
        }
        let spikes: u64 = outs.iter().map(Tensor::nonzero).sum();
        assert!(spikes > 0, "a silent run would satisfy the invariant trivially");
    }

    /// Same seed, same weights, same spikes, same audit — the property every other test rests on.
    #[test]
    fn the_same_seed_gives_the_same_spikes_and_the_same_audit() {
        let spec = Spec { residual: Residual::SewIand, ..Spec::spikformer_like(4, 8, 2, 2) };
        let mut rng = Rng::new(99);
        let inputs: Vec<Tensor> = (0..4).map(|_| random_spikes(4, 8, 0.4, &mut rng)).collect();

        let mut a = spec.build(&mut Rng::new(7), Position::None).unwrap();
        let mut b = spec.build(&mut Rng::new(7), Position::None).unwrap();
        let (ao, aa) = a.run(&inputs).unwrap();
        let (bo, ba) = b.run(&inputs).unwrap();
        for (x, y) in ao.iter().zip(&bo) {
            assert_eq!(x.values(), y.values());
        }
        assert_eq!(aa.split(), ba.split());
        assert_eq!(aa.sites(), ba.sites());

        // A different seed must actually differ, or the equality above proves nothing.
        let mut c = spec.build(&mut Rng::new(8), Position::None).unwrap();
        let (co, _) = c.run(&inputs).unwrap();
        assert!(
            ao.iter().zip(&co).any(|(x, y)| x.values() != y.values()),
            "two seeds produced identical models"
        );

        // And resetting state reproduces the run from the top.
        a.reset_state();
        let (ao2, _) = a.run(&inputs).unwrap();
        for (x, y) in ao.iter().zip(&ao2) {
            assert_eq!(x.values(), y.values(), "reset_state did not return the model to rest");
        }
    }

    /// ⭐ The first-spike latency of a constant-driven neuron, against the closed form.
    ///
    /// The simulation must fire on exactly the predicted timestep — not within a tolerance of it,
    /// because a spike time is an integer. Swept over several drives and time constants so that no
    /// single accidental coincidence can carry the test.
    #[test]
    fn the_first_spike_lands_on_the_timestep_the_closed_form_predicts() {
        let mut checked = 0;
        for tau in [1.0, 2.0, 4.0, 7.0] {
            for v_reset in [0.0, -0.25] {
                for x in [0.55, 0.6, 0.75, 1.0, 1.5, 3.0] {
                    let mut n = LifLayer::new(1, 1, tau, 0.5, v_reset).unwrap();
                    let want = n.first_spike_step(x);
                    let drive = Tensor::real(1, 1, vec![x]).unwrap();
                    let mut audit = Audit::new();
                    let mut got = None;
                    for step in 1..=400u32 {
                        if n.forward(&drive, "n", &mut audit).unwrap().values()[0] == 1.0 {
                            got = Some(step);
                            break;
                        }
                    }
                    assert_eq!(got, want, "tau {tau}, v_reset {v_reset}, x {x}");
                    checked += 1;
                }
            }
        }
        assert_eq!(checked, 48);
        // The sweep must contain a neuron that never fires AND several distinct latencies,
        // otherwise it is 48 copies of one case.
        let n = LifLayer::new(1, 1, 2.0, 0.5, 0.0).unwrap();
        assert_eq!(n.first_spike_step(0.4), None, "sub-threshold must be None, not a big number");
        assert_eq!(n.first_spike_step(0.5), None, "converging exactly onto the threshold is None");
        assert_eq!(n.first_spike_step(1.0), Some(1));
        assert_eq!(n.first_spike_step(0.6), Some(3));
        assert_eq!(n.first_spike_step(0.55), Some(4));
        assert_eq!(LifLayer::new(1, 1, 1.0, 0.5, 0.0).unwrap().first_spike_step(0.6), Some(1));
    }

    /// Membranes carry across timesteps at the level of a whole model, not just one layer.
    ///
    /// The weights are chosen so the query neuron's closed-form latency is **3**: the model is
    /// silent on the first two timesteps and fires on the third. Without integration across
    /// timesteps it would be silent forever, and `reset_state` would be decorative.
    #[test]
    fn a_model_integrates_a_constant_input_across_timesteps() {
        let d = 2;
        let scaled = |g: f64| {
            let mut w = eye(d);
            for v in &mut w {
                *v *= g;
            }
            Linear::new(d, d, w, None).unwrap()
        };
        let attn = Ssa::new(
            2,
            1,
            scaled(0.6),
            scaled(0.6),
            scaled(0.6),
            scaled(1.0),
            2.0,
            0.5,
            0.0,
            1.0,
            false,
            Order::ScoresFirst,
        )
        .unwrap();
        let mlp = SpikingMlp::new(2, scaled(1.0), scaled(1.0), 2.0, 0.5, 0.0).unwrap();
        let block = Block::new(attn, mlp, Residual::None).unwrap();
        let mut m = Model::new(2, Position::None, vec![block]).unwrap();

        // The closed form for the first projection's neuron, stated independently of the model.
        let probe = LifLayer::new(1, 1, 2.0, 0.5, 0.0).unwrap();
        assert_eq!(probe.first_spike_step(0.6), Some(3));

        let x = Tensor::spikes(2, 2, &[true, false, false, true]).unwrap();
        let (outs, _) = m.run(&[x.clone(), x.clone(), x.clone(), x]).unwrap();
        assert_eq!(outs[0].nonzero(), 0, "timestep 1 should be silent");
        assert_eq!(outs[1].nonzero(), 0, "timestep 2 should be silent");
        assert_eq!(outs[2].values(), &[1.0, 0.0, 0.0, 1.0], "timestep 3 should carry the identity");
        assert_ne!(outs[0].values(), outs[2].values(), "the membranes are not integrating");

        // And from rest it repeats, so the latency is a property of the model and not of history.
        m.reset_state();
        let x = Tensor::spikes(2, 2, &[true, false, false, true]).unwrap();
        let (again, _) = m.run(&[x.clone(), x.clone(), x]).unwrap();
        assert_eq!(again[2].values(), outs[2].values());
    }

    /// A bias is charged as an add on every output whether or not anything fired, which is the
    /// term a synaptic operation count omits at every layer that has one.
    #[test]
    fn a_bias_is_charged_on_every_output_including_a_silent_one() {
        let lin = Linear::new(2, 3, vec![0.0; 6], Some(vec![0.1, 0.2, 0.3])).unwrap();
        let x = Tensor::silent(4, 2).unwrap();
        let mut audit = Audit::new();
        let y = lin.forward(&x, "l", &mut audit).unwrap();
        assert_eq!(y.values(), &[0.1, 0.2, 0.3].repeat(4));
        assert_eq!(audit.site("l.bias", OpKind::Add).unwrap().effective, 12);
        assert_eq!(audit.effective_of(OpKind::Ac), 0);
        assert_eq!(lin.nonzero_weights(), 0);
    }

    /// Density and non-zero counts are what every effective count is built on, so they are checked
    /// directly rather than only through a model.
    #[test]
    fn tensor_density_counts_non_zero_elements() {
        let t = Tensor::spikes(2, 2, &[true, false, true, true]).unwrap();
        assert_eq!(t.nonzero(), 3);
        assert_eq!(t.density(), Some(0.75));
        assert_eq!(t.at(1, 1), Some(1.0));
        assert_eq!(t.at(2, 0), None);
        assert_eq!(t.row(2), None);
        assert!(matches!(rate_readout(&[]), Err(AttnError::Empty { .. })));
    }

    /// An `IAND` shortcut given a widened operand refuses rather than silently thresholding it.
    #[test]
    fn a_gated_shortcut_refuses_a_stream_that_is_no_longer_binary() {
        let wide = Tensor::new(1, 2, vec![2.0, 0.0], Domain::Integer(2)).unwrap();
        let bin = Tensor::spikes(1, 2, &[true, false]).unwrap();
        let e = Residual::SewIand.combine(&wide, &bin, "s", &mut Audit::new()).unwrap_err();
        assert!(matches!(e, AttnError::NotSpikeDriven { .. }), "{e}");
        let e = Position::OrTable(bin.clone())
            .apply(&wide, "p", &mut Audit::new())
            .unwrap_err();
        assert!(matches!(e, AttnError::NotSpikeDriven { .. }), "{e}");
    }

    /// Heads must tile the channels, and a head count that does not divide is refused rather than
    /// dropping the remainder — which would silently discard channels.
    #[test]
    fn heads_must_divide_the_channel_count() {
        let lin = || Linear::new(6, 6, eye(6), None).unwrap();
        let e = Ssa::new(2, 4, lin(), lin(), lin(), lin(), 2.0, 1.0, 0.0, 1.0, false, Order::ScoresFirst)
            .unwrap_err();
        assert!(matches!(e, AttnError::HeadsDoNotDivide { channels: 6, heads: 4 }), "{e}");
        assert!(
            Ssa::new(2, 3, lin(), lin(), lin(), lin(), 2.0, 1.0, 0.0, 1.0, false, Order::ScoresFirst)
                .is_ok()
        );
    }

    /// ⭐ Heads see disjoint channel slices, so a query cannot match a key across a head boundary.
    ///
    /// Token 0 fires on channels 0 and 2; token 1 on channels 0 and 3. With one head their
    /// similarity is 1 (they share channel 0 out of all four). With two heads, channel 0 lives in
    /// head 0 and channels 2 and 3 in head 1, so head 1 sees no match at all and the outputs
    /// differ. Both expected outputs below were computed by hand through the whole block.
    #[test]
    fn heads_see_disjoint_channel_slices() {
        let d = 4;
        let lin = || Linear::new(d, d, eye(d), None).unwrap();
        let mk = |heads: usize, order: Order| {
            Ssa::new(2, heads, lin(), lin(), lin(), lin(), 2.0, 0.5, 0.0, 1.0, false, order)
                .unwrap()
        };
        let x = Tensor::spikes(
            2,
            4,
            &[true, false, true, false, true, false, false, true],
        )
        .unwrap();

        let mut one = mk(1, Order::ScoresFirst);
        let mut two = mk(2, Order::ScoresFirst);
        let mut a = Audit::new();
        let mut b = Audit::new();
        let y1 = one.forward(&x, "s", &mut a).unwrap();
        let y2 = two.forward(&x, "s", &mut b).unwrap();
        // One head: scores [[2,1],[1,2]], attended [[3,0,2,1],[3,0,1,2]], halved by the leak and
        // thresholded at 0.5 gives [1,0,1,1] for both tokens.
        assert_eq!(y1.values(), &[1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0]);
        // Two heads: head 0 sees [[1,1],[1,1]], head 1 sees [[1,0],[0,1]], attended
        // [[2,0,1,0],[2,0,0,1]] and the second token loses channel 2 entirely.
        assert_eq!(y2.values(), &[1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0]);
        assert_ne!(y1.values(), y2.values(), "the head loop is not slicing anything");

        // Under ScoresFirst the head count does not change the bill: the same comparisons are made
        // in smaller groups. Under ValuesFirst it halves, because the gram matrix is per head.
        assert_eq!(
            a.site("s.ssa.scores", OpKind::Ac).unwrap().dense,
            b.site("s.ssa.scores", OpKind::Ac).unwrap().dense
        );
        let mut c = Audit::new();
        let mut e = Audit::new();
        mk(1, Order::ValuesFirst).forward(&x, "s", &mut c).unwrap();
        mk(2, Order::ValuesFirst).forward(&x, "s", &mut e).unwrap();
        assert_eq!(c.site("s.ssa.gram", OpKind::Ac).unwrap().dense, 4 * 2 * 4);
        assert_eq!(e.site("s.ssa.gram", OpKind::Ac).unwrap().dense, 2 * (2 * 2 * 2));
    }

    /// The weight layout is row-major **by output**, and a non-square asymmetric matrix is the only
    /// shape that can tell that from its transpose. Every count-based test in this module passes
    /// under a transposed read; this one does not.
    #[test]
    fn a_non_square_projection_uses_the_stated_weight_layout() {
        // out0 = 1·x0 + 2·x1 + 3·x2, out1 = 4·x0 + 5·x1 + 6·x2.
        let lin = Linear::new(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], None).unwrap();
        let x = Tensor::spikes(1, 3, &[true, false, true]).unwrap();
        let mut audit = Audit::new();
        let y = lin.forward(&x, "l", &mut audit).unwrap();
        assert_eq!(y.values(), &[4.0, 10.0], "the weights were read transposed");
        // The transposed reading would give [6, 8] and the same operation count, which is why the
        // count is not enough.
        assert_eq!(audit.site("l", OpKind::Ac).unwrap().dense, 6);
        assert_eq!(audit.site("l", OpKind::Ac).unwrap().effective, 4);
    }

    /// A spike resets the membrane **to** `v_reset`, not **by** `v_th`. The two agree whenever
    /// `v_reset` is zero and the membrane lands exactly on the threshold, which is every case the
    /// rest of this module exercises — so it is pinned here with neither condition holding.
    #[test]
    fn a_spike_resets_the_membrane_to_v_reset_and_not_by_the_threshold() {
        let mut n = LifLayer::new(1, 1, 2.0, 0.5, -0.25).unwrap();
        assert_eq!(n.membranes(), &[-0.25], "a layer must start at its reset potential");
        let drive = Tensor::real(1, 1, vec![5.0]).unwrap();
        let mut audit = Audit::new();
        assert_eq!(n.forward(&drive, "n", &mut audit).unwrap().values(), &[1.0]);
        // h was -0.25 + 0.5·5 = 2.25. A hard reset gives -0.25; a subtractive one would give 1.75.
        assert!((n.membranes()[0] - (-0.25)).abs() < 1e-15, "{}", n.membranes()[0]);
        n.reset_state();
        assert_eq!(n.membranes(), &[-0.25]);
    }

    /// The rate readout averages over timesteps **and** tokens, checked against a hand sum.
    #[test]
    fn the_rate_readout_averages_over_timesteps_and_tokens() {
        let a = Tensor::spikes(2, 2, &[true, false, true, true]).unwrap();
        let b = Tensor::spikes(2, 2, &[false, false, true, true]).unwrap();
        // Channel 0 fires 3 times out of 4 token-timesteps; channel 1 twice.
        assert_eq!(rate_readout(&[a.clone(), b]).unwrap(), vec![0.75, 0.5]);
        // A shape mismatch is refused rather than averaged over ragged rows.
        let wide = Tensor::silent(3, 2).unwrap();
        assert!(matches!(rate_readout(&[a, wide]), Err(AttnError::BadShape { .. })));
    }

    /// The model actually runs its positional encoder — without this, `Position` could be built,
    /// validated and then dropped on the floor by `Model::forward`.
    #[test]
    fn the_model_applies_its_positional_encoder() {
        let spec = Spec { residual: Residual::SewIand, ..Spec::spikformer_like(4, 8, 2, 1) };
        let table = Tensor::sinusoidal_spikes(4, 8).unwrap();
        let x = random_spikes(4, 8, 0.3, &mut Rng::new(51));

        let mut plain = spec.build(&mut Rng::new(52), Position::None).unwrap();
        let mut placed = spec.build(&mut Rng::new(52), Position::OrTable(table)).unwrap();
        let mut a = Audit::new();
        let mut b = Audit::new();
        let y1 = plain.forward(&x, &mut a).unwrap();
        let y2 = placed.forward(&x, &mut b).unwrap();
        assert!(a.site("position", OpKind::Logic).is_none());
        assert_eq!(b.site("position", OpKind::Logic).unwrap().effective, 32);
        assert_ne!(y1.values(), y2.values(), "the positional encoder changed nothing");
    }

    // ---------------------------------------------------------------------------------------
    // The parts an earlier test suite could not see fail.
    // ---------------------------------------------------------------------------------------

    /// ⭐ The wavelength schedule of Vaswani et al., which is the only thing that citation is for.
    ///
    /// `sinusoidal_spikes(_, 2)` cannot see it: at two channels `pair = c / 2` is zero for both, so
    /// the factor collapses to `10000^0 = 1` and every sign is `sin(pos)` or `cos(pos)`. The
    /// positions below are read off the **paper's** closed form — pair `i` of `d` channels has
    /// wavelength `2π · 10000^(2i/d)` — and written here as literals.
    ///
    /// At `d = 4`, pair 0 is `sin(pos)`, first negative at `pos = 4` (`π = 3.1416`), its cosine at
    /// `pos = 2` (`π/2 = 1.5708`). Pair 1 is `10000^(2/4) = 100`, so `sin(pos/100)` first goes
    /// negative at `⌈100π⌉ = 315` and `cos(pos/100)` at `⌈100 · π/2⌉ = 158`.
    ///
    /// At `d = 8` the four pairs are `1`, `10000^(2/8) = 10`, `10000^(4/8) = 100` and
    /// `10000^(6/8) = 1000`, so their sine channels first go negative at 4, 32, 315 and 3142.
    #[test]
    fn the_sinusoidal_table_follows_vaswanis_wavelength_schedule() {
        fn first_silent(t: &Tensor, c: usize) -> Option<usize> {
            (0..t.tokens()).find(|p| t.at(*p, c) == Some(0.0))
        }

        let four = Tensor::sinusoidal_spikes(400, 4).unwrap();
        assert_eq!(first_silent(&four, 0), Some(4), "sin(pos) crosses zero at pi");
        assert_eq!(first_silent(&four, 1), Some(2), "cos(pos) crosses zero at pi/2");
        assert_eq!(first_silent(&four, 2), Some(315), "sin(pos/100): 100 pi = 314.159");
        assert_eq!(first_silent(&four, 3), Some(158), "cos(pos/100): 100 pi/2 = 157.08");

        let eight = Tensor::sinusoidal_spikes(4000, 8).unwrap();
        assert_eq!(first_silent(&eight, 0), Some(4), "10000^0 = 1");
        assert_eq!(first_silent(&eight, 2), Some(32), "10000^(2/8) = 10, so ceil(10 pi)");
        assert_eq!(first_silent(&eight, 4), Some(315), "10000^(4/8) = 100");
        assert_eq!(first_silent(&eight, 6), Some(3142), "10000^(6/8) = 1000");

        // Row 0 is still all ones, and the schedule is what lets two far-apart positions differ.
        assert_eq!(eight.row(0).unwrap(), &[1.0; 8]);
        assert_ne!(eight.row(0).unwrap(), eight.row(1000).unwrap());
    }

    /// ⭐ `Site::because` explains every row of a report, and a length check is not a content
    /// check: swapping the accumulate and multiply-accumulate explanations left every other
    /// assertion in this module passing while every site reported the opposite of the truth.
    #[test]
    fn every_site_says_which_operand_decided_its_kind() {
        let x = Tensor::spikes(2, 2, &[true, false, false, true]).unwrap();

        let mut m = toy();
        let mut a = Audit::new();
        m.forward(&x, &mut a).unwrap();
        let ac = a.site("block0.attn.q_proj", OpKind::Ac).unwrap();
        assert_eq!(
            ac.because,
            "the activation operand is binary, so the weight is gated rather than multiplied"
        );
        assert_eq!(
            a.site("block0.attn.q_neuron.membrane", OpKind::Add).unwrap().because,
            "membrane update terms: the leak difference and the sum back into v"
        );
        assert_eq!(
            a.site("block0.attn.q_neuron.leak", OpKind::Shift).unwrap().because,
            "the membrane leak factor 1/tau multiplies the whole difference"
        );
        assert_eq!(
            a.site("block0.attn.q_neuron.threshold", OpKind::Compare).unwrap().because,
            "one threshold comparison per neuron per timestep"
        );

        let mut widened = spikeadd_toy();
        let mut b = Audit::new();
        widened.forward(&x, &mut b).unwrap();
        let mac = b.site("block0.mlp.fc1", OpKind::Mac).unwrap();
        assert_eq!(
            mac.because,
            "neither operand is binary: the activation is multi-valued and needs a multiplier"
        );
        assert_eq!(
            b.site("block0.shortcut1", OpKind::Add).unwrap().because,
            "a shortcut adds two tensors elementwise, widening the domain"
        );
        // The two explanations are opposites, so they cannot be interchangeable.
        assert_ne!(ac.because, mac.because);
        assert!(ac.because.contains("is binary"), "{}", ac.because);
        assert!(mac.because.contains("neither operand is binary"), "{}", mac.because);
    }

    /// ⭐ The rule that is "the whole audit", exercised at **both** attention products with a
    /// non-binary operand — which no test did, so either site could name the wrong operand and
    /// report a multiplier-free datapath for a product that needs a multiplier.
    ///
    /// `attend` is public and takes [`Tensor::real`], so this is reachable from the public API.
    #[test]
    fn each_attention_product_follows_the_operand_that_decides_it() {
        let qb = Tensor::spikes(2, 2, &[true, false, false, true]).unwrap();
        let kb = Tensor::spikes(2, 2, &[true, false, false, true]).unwrap();
        let vb = Tensor::spikes(2, 2, &[true, true, false, true]).unwrap();
        let qr = Tensor::real(2, 2, vec![1.5, 0.0, 0.0, 2.5]).unwrap();
        let kr = Tensor::real(2, 2, vec![1.5, 0.0, 0.0, 2.5]).unwrap();
        let vr = Tensor::real(2, 2, vec![0.5, 1.5, 0.0, 2.0]).unwrap();

        let kind = |q: &Tensor, k: &Tensor, v: &Tensor, o: Order, path: &str| {
            let mut a = Audit::new();
            attend(q, k, v, o, "t", &mut a).unwrap();
            let s = a.sites().iter().find(|s| s.path == path).expect("the site is charged");
            assert_eq!(s.because, expected_reason(path, s.kind), "{path}");
            s.kind
        };

        // S = Q Kᵀ is an integer matrix, so `S · V` is decided by V and by nothing else.
        assert_eq!(
            kind(&qb, &kb, &vr, Order::ScoresFirst, "t.attend"),
            OpKind::Mac,
            "a real V at S·V still needs a multiplier"
        );
        assert_eq!(
            kind(&qr, &kb, &vb, Order::ScoresFirst, "t.attend"),
            OpKind::Ac,
            "a real Q does not reach S·V"
        );
        assert_eq!(
            kind(&qr, &kb, &vb, Order::ScoresFirst, "t.scores"),
            OpKind::Ac,
            "one binary operand is enough to remove the multiplier from Q·Kᵀ"
        );
        assert_eq!(kind(&qr, &kr, &vb, Order::ScoresFirst, "t.scores"), OpKind::Mac);

        // `Q (Kᵀ V)` is decided by Q; the gram by K and V.
        assert_eq!(
            kind(&qr, &kb, &vb, Order::ValuesFirst, "t.attend"),
            OpKind::Mac,
            "a real Q at Q·(KᵀV) needs a multiplier"
        );
        assert_eq!(
            kind(&qb, &kr, &vb, Order::ValuesFirst, "t.attend"),
            OpKind::Ac,
            "a real K does not reach Q·(KᵀV)"
        );
        assert_eq!(
            kind(&qb, &kr, &vb, Order::ValuesFirst, "t.gram"),
            OpKind::Ac,
            "a binary V is enough at Kᵀ V"
        );
        assert_eq!(kind(&qb, &kr, &vr, Order::ValuesFirst, "t.gram"), OpKind::Mac);
    }

    /// The conditional positional encoder sits in front of a **caller-supplied** tensor, which may
    /// already have been widened. The convolution's kind has to follow that input rather than the
    /// binary case it happens to be exercised on everywhere else.
    #[test]
    fn a_depthwise_convolution_follows_its_input_domain() {
        let conv = DepthwiseConv1d::centre_tap(2, 3).unwrap();
        let bin = Tensor::spikes(2, 2, &[true, false, true, true]).unwrap();
        let wide = Tensor::new(2, 2, vec![2.0, 0.0, 1.0, 2.0], Domain::Integer(2)).unwrap();

        let mut a = Audit::new();
        conv.forward(&bin, "c", &mut a).unwrap();
        assert_eq!(a.site("c", OpKind::Ac).unwrap().dense, 2 * 2 * 3);
        assert!(a.site("c", OpKind::Mac).is_none(), "a binary input needs no multiplier");

        let mut b = Audit::new();
        let y = conv.forward(&wide, "c", &mut b).unwrap();
        assert_eq!(y.values(), wide.values(), "the centre tap is still the identity");
        assert_eq!(
            b.site("c", OpKind::Mac).unwrap().dense,
            2 * 2 * 3,
            "a widened input needs a multiplier and the conv reported an accumulate"
        );
        assert!(b.site("c", OpKind::Ac).is_none());

        // And through the encoder, which is where a caller actually meets it.
        let mut p = Position::Conditional(
            DepthwiseConv1d::centre_tap(2, 3).unwrap(),
            LifLayer::new(2, 2, 2.0, 0.4, 0.0).unwrap(),
        );
        let mut c = Audit::new();
        p.apply(&wide, "position", &mut c).unwrap();
        assert!(c.site("position.conv", OpKind::Mac).is_some(), "the encoder hid the multiplier");
    }

    /// ⭐ `Audit::synops().dense` is the denominator of every `NeuroBench` sparsity figure this
    /// module can produce, and it was only ever read on a model with no multiply-accumulates at
    /// all — the one shape where dropping its `MAC` term is invisible.
    #[test]
    fn the_synops_denominator_counts_dense_multiply_accumulates_too() {
        let mut a = Audit::new();
        a.charge("p", OpKind::Ac, "test reason, long enough", 10, 4).unwrap();
        a.charge("q", OpKind::Mac, "test reason, long enough", 7, 3).unwrap();
        a.charge("r", OpKind::Mul, "test reason, long enough", 100, 100).unwrap();
        let s = a.synops();
        assert_eq!(s.dense, 17, "10 dense accumulates plus 7 dense multiply-accumulates");
        assert_eq!(s.effective_acs, 4);
        assert_eq!(s.effective_macs, 3);
        assert_eq!(s.effective_total(), Some(7), "the 100 multiplies have no slot in SynOps");

        // And on a model, hand-counted. The SpikeAdd toy widens its stream at the first shortcut,
        // so `mlp.fc1` is 2 tokens × 2 in × 2 out = 8 dense MACs, while the other seven product
        // sites (q/k/v/out projections, Q Kᵀ, S V, fc2) are 8 dense ACs each: 56 + 8 = 64.
        let mut m = spikeadd_toy();
        let x = Tensor::spikes(2, 2, &[true, false, false, true]).unwrap();
        let mut b = Audit::new();
        m.forward(&x, &mut b).unwrap();
        assert_eq!(b.dense_of(OpKind::Ac), 56, "seven accumulate sites of 8");
        assert_eq!(b.dense_of(OpKind::Mac), 8, "the MLP's first layer after the widening");
        assert_eq!(b.synops().dense, 64, "the metric's denominator dropped the MAC layer");
    }

    /// ⭐ `synaptic_ac_fraction` is a ratio of **effective** counts, so `Some(1.0)` is a statement
    /// about one input, not about an architecture. This model contains a live multiply-accumulate
    /// layer, fires on two outputs, and reports itself perfectly spike-driven with zero multiplier
    /// operations. The architectural claim is `dense_of(Mac) == 0`, and it is false here.
    #[test]
    fn a_reported_fraction_of_one_is_not_a_multiplier_free_architecture() {
        let d = 2;
        let lin = || Linear::new(d, d, eye(d), None).unwrap();
        let zero = || Linear::new(d, d, vec![0.0; d * d], None).unwrap();
        let attn = Ssa::new(
            2,
            1,
            lin(),
            lin(),
            lin(),
            lin(),
            2.0,
            0.5,
            0.0,
            1.0,
            false,
            Order::ScoresFirst,
        )
        .unwrap();
        let mlp = SpikingMlp::new(2, zero(), zero(), 2.0, 0.5, 0.0).unwrap();
        let block = Block::new(attn, mlp, Residual::SpikeAdd).unwrap();
        let mut m = Model::new(2, Position::None, vec![block]).unwrap();

        let x = Tensor::spikes(2, 2, &[true, false, false, true]).unwrap();
        let mut a = Audit::new();
        let y = m.forward(&x, &mut a).unwrap();
        assert_eq!(y.nonzero(), 2, "a silent run would make every fraction below meaningless");

        assert_eq!(a.dense_of(OpKind::Mac), 8, "the MLP's first layer IS a multiply-accumulate");
        assert_eq!(a.effective_of(OpKind::Mac), 0, "its weights are zero, so none of them fired");
        assert_eq!(a.synaptic_ac_fraction(), Some(1.0), "and the reported fraction says perfect");
        assert_eq!(a.multiplier_ops(), 0, "so does the multiplier count");
        // The `IAND` model in `an_iand_shortcut_keeps_the_whole_model_multiplier_free` is the
        // architectural claim; this one is what the same reported number looks like without it.
    }

    /// ⭐ Two sites that each fit in a `u64` can still sum past the end of one. `Audit::charge`
    /// refuses the per-site overflow, but the aggregates answered it by panicking in a debug build
    /// and **wrapping in a release one**: `dense_of` returned 18446744073709551614 for two sites of
    /// `u64::MAX`, and `Display` reaches it, so printing a report was enough to trigger it.
    #[test]
    fn aggregate_counts_saturate_instead_of_wrapping() {
        let mut a = Audit::new();
        a.charge("one", OpKind::Ac, "test reason, long enough", u64::MAX, u64::MAX).unwrap();
        a.charge("two", OpKind::Ac, "test reason, long enough", u64::MAX, u64::MAX).unwrap();
        assert_eq!(a.dense_of(OpKind::Ac), u64::MAX, "the dense aggregate wrapped");
        assert_eq!(a.effective_of(OpKind::Ac), u64::MAX, "the effective aggregate wrapped");
        assert_eq!(a.synops().dense, u64::MAX, "the NeuroBench denominator wrapped");
        assert_eq!(a.split().accumulates, u64::MAX);

        // `Split::total` still REFUSES rather than saturating, so the two remain distinguishable.
        a.charge("three", OpKind::Compare, "test reason, long enough", 1, 1).unwrap();
        assert_eq!(a.effective_total(), None, "the grand total must refuse, not saturate");
        assert_eq!(a.ac_fraction(), None, "a fraction of an unrepresentable total is not a number");

        // And a report prints, rather than panicking or calling an enormous count "nothing ran".
        let text = a.to_string();
        assert!(text.contains("18446744073709551615"), "{text}");
        assert!(text.contains("AC share of ALL operations: the total does not fit in u64"), "{text}");

        // The synaptic total refuses from its own side.
        let mut b = Audit::new();
        b.charge("ac", OpKind::Ac, "test reason, long enough", u64::MAX, u64::MAX).unwrap();
        b.charge("mac", OpKind::Mac, "test reason, long enough", 1, 1).unwrap();
        assert_eq!(b.synaptic_ac_fraction(), None);
        assert!(
            b.to_string().contains("AC share of synaptic operations: the total does not fit in u64"),
            "{b}"
        );

        // multiplier_ops has always documented saturation; it still saturates.
        let mut c = Audit::new();
        c.charge("m1", OpKind::Mac, "test reason, long enough", u64::MAX, u64::MAX).unwrap();
        c.charge("m2", OpKind::Mul, "test reason, long enough", u64::MAX, u64::MAX).unwrap();
        assert_eq!(c.multiplier_ops(), u64::MAX);
    }

    /// ⭐ The closed form the whole neuron section is checked against, over the **whole** `u32`
    /// range and with a reset potential that is not zero.
    ///
    /// `powi(steps as i32)` turned every `steps >= 2^31` into a negative exponent: at `tau = 2`,
    /// `v_reset = 0`, `v0 = 1` it answered `inf` at `2_147_483_648` and `0.0` one step earlier.
    /// And with `v_reset = 0`, the only value the rest of the module uses, the `(v0 - v_reset)`
    /// offset vanishes identically, so the offset was never checked at all.
    #[test]
    fn the_closed_form_membrane_is_exact_over_the_whole_step_range() {
        let n = LifLayer::new(1, 1, 2.0, 1.0, 0.0).unwrap();
        assert_eq!(n.relaxed(1.0, 0), 1.0, "no steps is no decay");
        assert_eq!(n.relaxed(1.0, 1), 0.5);
        assert_eq!(n.relaxed(1.0, 10), 1.0 / 1024.0, "2^-10, as a literal");
        assert_eq!(n.relaxed(1.0, 2_147_483_647), 0.0);
        assert_eq!(n.relaxed(1.0, 2_147_483_648), 0.0, "a negative exponent returned inf here");
        assert_eq!(n.relaxed(1.0, u32::MAX), 0.0);

        // The reset offset, with a reset potential the rest of the module never uses. tau = 2 gives
        // a decay of 1/2, so -0.25 + (0.75 + 0.25) * 0.25 = 0.0 exactly, while dropping the offset
        // would give -0.25 + 0.75 * 0.25 = -0.0625.
        let o = LifLayer::new(1, 1, 2.0, 1.0, -0.25).unwrap();
        assert_eq!(o.relaxed(0.75, 2), 0.0, "the (v0 - v_reset) offset was dropped");
        assert_eq!(o.relaxed(0.75, 0), 0.75);
        assert_eq!(o.relaxed(1.75, 1), 0.75, "-0.25 + 2.0 * 0.5");
        assert_eq!(o.relaxed(-0.25, 5), -0.25, "starting at rest stays at rest");
        assert_eq!(o.relaxed(0.75, u32::MAX), -0.25, "it relaxes TO v_reset, not to zero");

        // And the closed form still tracks the simulation it exists to check, offset live.
        let mut sim = LifLayer::new(1, 1, 4.0, 10.0, -0.25).unwrap();
        let mut audit = Audit::new();
        sim.forward(&Tensor::real(1, 1, vec![2.0]).unwrap(), "n", &mut audit).unwrap();
        let v0 = sim.membranes()[0];
        assert!((v0 - 0.25).abs() < 1e-15, "-0.25 + 0.25 * 2.0 = 0.25, got {v0}");
        let silent = Tensor::real(1, 1, vec![0.0]).unwrap();
        for step in 1..=5u32 {
            sim.forward(&silent, "n", &mut audit).unwrap();
            let (got, want) = (sim.membranes()[0], sim.relaxed(v0, step));
            assert!((got - want).abs() < 1e-14, "step {step}: {got} vs closed form {want}");
        }
        assert!(sim.membranes()[0] < v0, "the membrane did not move; the comparison is vacuous");
        assert!(sim.membranes()[0] > -0.25, "it decayed past the reset it converges to");
    }

    /// ⭐ The scale fold's two stated conditions did not imply the exactness the doc claimed.
    /// `s = 2^1022` is an exact normal power of two and `v_reset` is zero — both hold — and the two
    /// paths disagree completely: on a 2×2 all-ones input the unfolded path overflows to infinity
    /// and refuses, while the folded path returns a full row of spikes.
    #[test]
    fn the_fold_is_refused_where_the_two_paths_would_not_agree() {
        let d = 2;
        let lin = || Linear::new(d, d, eye(d), None).unwrap();
        let mk = |scale: f64, v_th: f64, fold: bool| {
            Ssa::new(
                2,
                1,
                lin(),
                lin(),
                lin(),
                lin(),
                2.0,
                v_th,
                0.0,
                scale,
                fold,
                Order::ScoresFirst,
            )
        };
        let huge = 2f64.powi(1022);
        assert!(is_exact_power_of_two(huge), "condition 1 holds, so condition 3 is what refuses");
        let e = mk(huge, 0.5, true).unwrap_err();
        assert!(matches!(e, AttnError::UnfoldableScale { .. }), "{e}");
        assert!(format!("{e}").contains("overflows"), "{e}");

        // The unfolded path at that scale really does refuse — that IS the disagreement.
        let ones = Tensor::spikes(2, 2, &[true; 4]).unwrap();
        let mut plain = mk(huge, 0.5, false).unwrap();
        assert!(
            matches!(
                plain.forward(&ones, "s", &mut Audit::new()),
                Err(AttnError::NonFinite { .. })
            ),
            "the unfolded path no longer overflows, so this test has stopped testing anything"
        );

        // A folded threshold that is not a normal number is refused from the other end.
        let e = mk(2f64.powi(-1022), 2f64.powi(40), true).unwrap_err();
        assert!(format!("{e}").contains("v_th / s"), "{e}");

        // Just inside the bound the fold is still allowed AND still bit-exact: every operand is
        // binary, so an attended element is at most tokens · d_head = 4, and 4 · 2^1020 = 2^1022.
        let big = 2f64.powi(1020);
        let mut unfolded = mk(big, 0.5, false).unwrap();
        let mut folded = mk(big, 0.5, true).unwrap();
        let a = unfolded.forward(&ones, "s", &mut Audit::new()).unwrap();
        let b = folded.forward(&ones, "s", &mut Audit::new()).unwrap();
        assert_eq!(a.values(), b.values(), "the two paths disagreed inside the accepted range");
        assert!(a.nonzero() > 0, "both were silent, so the comparison proves nothing");
        // And the paper's own scale is nowhere near the edge.
        assert!(mk(0.125, 1.0, true).is_ok(), "Spikformer's s must still fold");
    }

    /// ⭐ Spikformer puts a `BatchNorm` after every linear layer. At inference the gain folds into
    /// the preceding weights for free; the **shift** becomes a bias that does not fold away, and
    /// `Spec::build` passed `None` for all six projections per block, charging none of them.
    ///
    /// The count is hand-derived: `tokens · (4 · d_model + d_model + mlp_hidden)` per block per
    /// timestep — four attention projections and the output projection at `d_model` outputs each,
    /// the `MLP`'s two layers at `mlp_hidden` and `d_model`. At 8 tokens, `d_model = 16`,
    /// `mlp_hidden = 64`, 2 blocks and 4 timesteps: `8 · 144 · 2 · 4 = 9216`.
    #[test]
    fn the_folded_batchnorm_shift_is_charged_and_changes_no_spike() {
        let spec = Spec { residual: Residual::SewIand, ..Spec::spikformer_like(8, 16, 4, 2) };
        assert!(spec.bias, "Spikformer normalises after every projection");
        let bare = Spec { bias: false, ..spec };

        let mut rng = Rng::new(22);
        let inputs: Vec<Tensor> = (0..4).map(|_| random_spikes(8, 16, 0.3, &mut rng)).collect();
        let mut with_bias = spec.build(&mut Rng::new(21), Position::None).unwrap();
        let mut without = bare.build(&mut Rng::new(21), Position::None).unwrap();
        let (o1, a1) = with_bias.run(&inputs).unwrap();
        let (o2, a2) = without.run(&inputs).unwrap();

        let bias_adds: u64 =
            a1.sites().iter().filter(|s| s.path.ends_with(".bias")).map(|s| s.effective).sum();
        assert_eq!(bias_adds, 9216, "the folded BatchNorm shift, hand-counted");
        assert_eq!(
            a2.sites().iter().filter(|s| s.path.ends_with(".bias")).count(),
            0,
            "the bias-free spec charged one anyway"
        );
        assert_eq!(a1.effective_of(OpKind::Add) - a2.effective_of(OpKind::Add), 9216);

        // The shift is exactly zero at initialisation, so it moves no spike — only the bill.
        let spikes: u64 = o1.iter().map(Tensor::nonzero).sum();
        assert!(spikes > 0, "a silent run would make the comparison vacuous");
        for (a, b) in o1.iter().zip(&o2) {
            assert_eq!(a.values(), b.values(), "the folded shift moved a spike");
        }

        // And it is not a rounding-level change to the headline.
        let (honest, bare_honest) = (a1.ac_fraction().unwrap(), a2.ac_fraction().unwrap());
        assert!((bare_honest - 0.5063).abs() < 5e-5, "without the shift: {bare_honest}");
        assert!((honest - 0.4579).abs() < 5e-5, "with the shift: {honest}");
        assert!(honest < bare_honest, "the omitted adds were flattering the fraction");
    }

    /// ⭐ The honest fraction is not a property of the architecture: it moves nearly ten points
    /// with a constant this crate invented and says so. `Spec::gain` sets the firing density while
    /// the per-neuron overhead is fixed, so the fraction rises with it — which is why it is swept
    /// here rather than pinned once and quoted as *the* number.
    #[test]
    fn the_honest_fraction_moves_with_the_initialisation_gain() {
        let mut rng = Rng::new(22);
        let inputs: Vec<Tensor> = (0..4).map(|_| random_spikes(8, 16, 0.3, &mut rng)).collect();
        // Measured on this exact shape and these exact seeds, with the folded shift charged.
        // 2.449 is the Kaiming-uniform bound sqrt(6), which the default 6.0 resembles and is not.
        let sweep = [(1.0, 0.3962), (2.449, 0.4045), (4.0, 0.4304), (6.0, 0.4579), (12.0, 0.4915)];
        let mut seen = Vec::new();
        for (gain, want) in sweep {
            let spec =
                Spec { gain, residual: Residual::SewIand, ..Spec::spikformer_like(8, 16, 4, 2) };
            let mut m = spec.build(&mut Rng::new(21), Position::None).unwrap();
            let (outs, audit) = m.run(&inputs).unwrap();
            let spikes: u64 = outs.iter().map(Tensor::nonzero).sum();
            assert!(spikes > 0, "gain {gain} was silent, so its fraction means nothing");
            let got = audit.ac_fraction().unwrap();
            assert!((got - want).abs() < 5e-5, "gain {gain}: {got}, expected {want}");
            // The REPORTED fraction is 1.0 at every gain, so it says nothing about any of this.
            assert_eq!(audit.synaptic_ac_fraction(), Some(1.0), "gain {gain}");
            seen.push(got);
        }
        assert!(seen[4] - seen[0] > 0.09, "the sweep collapsed: {seen:?}");
        assert!(seen.windows(2).all(|w| w[0] < w[1]), "density did not rise with gain: {seen:?}");
    }

    /// `Tensor::new` has always refused a shape whose element count does not fit in a `usize`. Two
    /// of its siblings formed that product first and panicked in the multiplication instead.
    #[test]
    fn a_tensor_shape_that_cannot_fit_is_refused_rather_than_panicking() {
        assert!(matches!(
            Tensor::new(usize::MAX, 2, vec![], Domain::Binary),
            Err(AttnError::Overflow { what: "tensor" })
        ));
        assert!(matches!(
            Tensor::silent(usize::MAX, 2),
            Err(AttnError::Overflow { what: "tensor" })
        ));
        assert!(matches!(
            Tensor::sinusoidal_spikes(usize::MAX, 2),
            Err(AttnError::Overflow { what: "tensor" })
        ));
        // A zero dimension is still refused first, and as Empty rather than Overflow.
        assert!(matches!(Tensor::silent(0, 2), Err(AttnError::Empty { what: "tokens" })));
        assert!(matches!(Tensor::silent(2, 0), Err(AttnError::Empty { what: "channels" })));
        assert!(matches!(Tensor::sinusoidal_spikes(0, 2), Err(AttnError::Empty { what: "tokens" })));
        assert!(matches!(
            Tensor::sinusoidal_spikes(2, 0),
            Err(AttnError::Empty { what: "channels" })
        ));
        // And the ordinary shapes still work.
        assert_eq!(Tensor::silent(3, 4).unwrap().values().len(), 12);
    }

    /// Three documented construction-time refusals that nothing could make fire. Each could be
    /// deleted and the module stayed green, which turns a named refusal into a shape error several
    /// frames deeper — or into nothing at all.
    #[test]
    fn the_construction_time_refusals_can_all_be_made_to_fire() {
        let sq = || Linear::new(2, 2, eye(2), None).unwrap();
        let mk_attn = || {
            Ssa::new(
                2,
                1,
                sq(),
                sq(),
                sq(),
                sq(),
                2.0,
                1.0,
                0.0,
                1.0,
                false,
                Order::ScoresFirst,
            )
            .unwrap()
        };

        // Ssa: every projection must be d_model × d_model.
        let tall = Linear::new(3, 2, vec![0.0; 6], None).unwrap();
        let e = Ssa::new(
            2,
            1,
            sq(),
            tall,
            sq(),
            sq(),
            2.0,
            1.0,
            0.0,
            1.0,
            false,
            Order::ScoresFirst,
        )
        .unwrap_err();
        assert!(
            matches!(e, AttnError::BadShape { what: "attention projection", got: 6, want: 4 }),
            "{e}"
        );

        // The accessors a caller reads these shapes back through.
        let good = mk_attn();
        assert_eq!(good.d_model(), 2);
        assert_eq!(good.heads(), 1);
        assert_eq!(good.tokens(), 2);
        assert!(!good.folds_scale(), "this block was built without the fold");

        // Block: the MLP must come back at the attention's width, or the shortcut cannot line up.
        let wide_mlp = SpikingMlp::new(
            2,
            Linear::new(2, 2, eye(2), None).unwrap(),
            Linear::new(2, 3, vec![0.0; 6], None).unwrap(),
            2.0,
            1.0,
            0.0,
        )
        .unwrap();
        assert_eq!(wide_mlp.n_out(), 3, "the MLP really is three channels wide");
        let e = Block::new(mk_attn(), wide_mlp, Residual::None).unwrap_err();
        assert!(matches!(e, AttnError::BadShape { what: "block width", got: 3, want: 2 }), "{e}");

        // Model: a block built for two tokens cannot join a three-token model. Membranes are per
        // position, so this is the check that stops one sequence's state entering another.
        let mlp = SpikingMlp::new(2, sq(), sq(), 2.0, 1.0, 0.0).unwrap();
        let block = Block::new(mk_attn(), mlp, Residual::None).unwrap();
        assert_eq!(block.d_model(), 2);
        let e = Model::new(3, Position::None, vec![block.clone()]).unwrap_err();
        assert!(matches!(e, AttnError::BadShape { what: "block tokens", got: 2, want: 3 }), "{e}");
        let ok = Model::new(2, Position::None, vec![block.clone()]).expect("the matching shape");
        assert_eq!(ok.tokens(), 2);
        assert_eq!(ok.depth(), 1);

        // And blocks of different widths do not stack.
        let sq4 = || Linear::new(4, 4, eye(4), None).unwrap();
        let attn4 = Ssa::new(
            2,
            1,
            sq4(),
            sq4(),
            sq4(),
            sq4(),
            2.0,
            1.0,
            0.0,
            1.0,
            false,
            Order::ScoresFirst,
        )
        .unwrap();
        let mlp4 = SpikingMlp::new(2, sq4(), sq4(), 2.0, 1.0, 0.0).unwrap();
        let wide = Block::new(attn4, mlp4, Residual::None).unwrap();
        let e = Model::new(2, Position::None, vec![block, wide]).unwrap_err();
        assert!(matches!(e, AttnError::BadShape { what: "block width", got: 4, want: 2 }), "{e}");
        assert!(matches!(Model::new(2, Position::None, vec![]), Err(AttnError::Empty { .. })));
    }

    /// `OpKind::needs_multiplier` and `OpKind::is_synaptic` are the definitions [`Audit`]'s own
    /// aggregates are built from, and neither was asserted anywhere. `OpKind::ALL` is what a report
    /// iterates, so a duplicate entry in it silently drops one whole row and doubles another.
    #[test]
    fn the_operation_kind_predicates_say_what_the_audit_means_by_them() {
        assert!(OpKind::Mac.needs_multiplier(), "a multiply-accumulate occupies the multiplier");
        assert!(OpKind::Mul.needs_multiplier(), "so does a bare multiply");
        for k in [OpKind::Ac, OpKind::Add, OpKind::Shift, OpKind::Compare, OpKind::Logic] {
            assert!(!k.needs_multiplier(), "{k} does not occupy a multiplier");
        }
        assert!(OpKind::Ac.is_synaptic());
        assert!(OpKind::Mac.is_synaptic());
        for k in [OpKind::Add, OpKind::Mul, OpKind::Shift, OpKind::Compare, OpKind::Logic] {
            assert!(!k.is_synaptic(), "{k} is not one of NeuroBench's two synaptic counts");
        }

        // ALL lists each kind exactly once.
        assert_eq!(OpKind::ALL.len(), 7);
        for k in [
            OpKind::Ac,
            OpKind::Mac,
            OpKind::Add,
            OpKind::Mul,
            OpKind::Shift,
            OpKind::Compare,
            OpKind::Logic,
        ] {
            assert_eq!(OpKind::ALL.iter().filter(|x| **x == k).count(), 1, "{k} appears once");
        }
        let mut labels: Vec<&str> = OpKind::ALL.iter().map(|k| k.label()).collect();
        labels.sort_unstable();
        labels.dedup();
        assert_eq!(labels.len(), 7, "two kinds print the same label");

        // The domain predicate the product rule is written in terms of, and how a domain prints.
        assert!(Domain::Binary.is_binary());
        assert!(!Domain::Integer(2).is_binary(), "a 0..=2 stream still needs a multiplier");
        assert!(!Domain::Real.is_binary());
        assert_eq!(Domain::Binary.to_string(), "binary");
        assert_eq!(Domain::Integer(3).to_string(), "integer 0..=3");
        assert_eq!(Domain::Real.to_string(), "real");
        assert_eq!(OpKind::Mac.to_string(), "MAC");

        // And the product rule these sit next to.
        assert_eq!(OpKind::product(Domain::Binary, Domain::Real), OpKind::Ac);
        assert_eq!(OpKind::product(Domain::Real, Domain::Binary), OpKind::Ac);
        assert_eq!(OpKind::product(Domain::Binary, Domain::Binary), OpKind::Ac);
        assert_eq!(OpKind::product(Domain::Integer(2), Domain::Real), OpKind::Mac);
        assert_eq!(OpKind::product(Domain::Real, Domain::Real), OpKind::Mac);
    }

    /// A printed report lists every kind the pass touched, in `OpKind::ALL`'s order — including the
    /// gated shortcut's logic row, which is the row a duplicate entry in `ALL` would drop.
    #[test]
    fn a_printed_report_lists_every_kind_the_pass_touched() {
        let spec = Spec { residual: Residual::SewIand, ..Spec::spikformer_like(4, 4, 2, 1) };
        let mut m = spec.build(&mut Rng::new(3), Position::None).unwrap();
        let x = random_spikes(4, 4, 0.5, &mut Rng::new(77));
        let mut audit = Audit::new();
        m.forward(&x, &mut audit).unwrap();
        let text = audit.to_string();
        let labels: Vec<&str> = text
            .lines()
            .filter(|l| l.contains("dense") && l.contains("effective"))
            .map(|l| l.split_whitespace().next().expect("a row has a label"))
            .collect();
        assert_eq!(
            labels,
            ["AC", "add", "shift", "compare", "logic"],
            "the report's rows are wrong:\n{text}"
        );
        assert!(text.starts_with("operation audit over 1 timestep(s)"), "{text}");
        assert!(text.contains("AC share of synaptic operations (as reported): 1.0000"), "{text}");
    }

    /// `multiplier_ops` is the arithmetic that occupies the multiplier array: multiply-accumulates
    /// **and** bare multiplies. Dropping either term was invisible, because the only assertions
    /// anywhere on it were `> 0` and `== 0`.
    #[test]
    fn multiplier_ops_counts_bare_multiplies_as_well_as_multiply_accumulates() {
        let mut a = Audit::new();
        a.charge("mac", OpKind::Mac, "test reason, long enough", 9, 3).unwrap();
        a.charge("mul", OpKind::Mul, "test reason, long enough", 9, 5).unwrap();
        a.charge("shift", OpKind::Shift, "test reason, long enough", 9, 7).unwrap();
        a.charge("ac", OpKind::Ac, "test reason, long enough", 9, 9).unwrap();
        assert_eq!(a.multiplier_ops(), 8, "3 multiply-accumulates plus 5 bare multiplies");
        assert_eq!(a.effective_of(OpKind::Shift), 7, "a shift is counted on its own");

        // On a real layer: a leak that is not a power of two is a bare multiply per neuron per
        // timestep, and it is the only multiplier work an otherwise binary model does.
        let mut n = LifLayer::new(2, 3, 3.0, 1.0, 0.0).unwrap();
        let mut b = Audit::new();
        n.forward(&Tensor::silent(2, 3).unwrap(), "n", &mut b).unwrap();
        assert_eq!(b.effective_of(OpKind::Mac), 0, "nothing here is a multiply-accumulate");
        assert_eq!(b.multiplier_ops(), 6, "1/3 is not a shift: 6 neurons, 6 multiplies");
    }

    /// `Linear::nonzero_weights` is the connection density every effective count is conditioned on,
    /// and the only assertion on it was `== 0` on an all-zero matrix — which `return 0` satisfies.
    #[test]
    fn nonzero_weights_counts_the_connections_that_exist() {
        assert_eq!(Linear::new(3, 3, eye(3), None).unwrap().nonzero_weights(), 3, "an identity");
        assert_eq!(
            Linear::new(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], None).unwrap().nonzero_weights(),
            6,
            "a full matrix"
        );
        let sparse = Linear::new(2, 2, vec![0.0, -1.0, 0.0, 0.0], None).unwrap();
        assert_eq!(sparse.nonzero_weights(), 1, "a negative weight is still a connection");
        assert_eq!(Linear::new(2, 2, vec![0.0; 4], None).unwrap().nonzero_weights(), 0);

        // And it is the density the effective count follows: one live weight, one accumulate.
        let mut a = Audit::new();
        sparse.forward(&Tensor::spikes(1, 2, &[false, true]).unwrap(), "l", &mut a).unwrap();
        assert_eq!(a.site("l", OpKind::Ac).unwrap().effective, 1);
        assert_eq!(a.site("l", OpKind::Ac).unwrap().dense, 4);
    }

    /// The attention scale is charged per **non-zero** element, like every other site:
    /// [`Site::effective`] is "operands that were not zero", and charging the scale on zeros would
    /// contradict the rule the rest of the audit follows. The toy attends to two of its four.
    #[test]
    fn the_attention_scale_is_charged_only_where_the_element_was_not_zero() {
        let d = 2;
        let lin = || Linear::new(d, d, eye(d), None).unwrap();
        let mut ssa = Ssa::new(
            2,
            1,
            lin(),
            lin(),
            lin(),
            lin(),
            2.0,
            0.25,
            0.0,
            0.5,
            false,
            Order::ScoresFirst,
        )
        .unwrap();
        let x = Tensor::spikes(2, 2, &[true, false, false, true]).unwrap();
        let mut a = Audit::new();
        let y = ssa.forward(&x, "s", &mut a).unwrap();
        assert_eq!(y.values(), &[1.0, 0.0, 0.0, 1.0], "the identity did not come through");
        let site = a.site("s.scale", OpKind::Shift).expect("a scale of 0.5 is a shift");
        assert_eq!(site.dense, 4, "2 tokens x 2 channels, whatever the data did");
        assert_eq!(site.effective, 2, "the attended matrix has two zeros and they cost nothing");
        assert!(site.effective < site.dense, "the scale was charged on zeros");
    }

    /// Sites merge by path **and** kind. Nothing in the module charges one path with two kinds, so
    /// the second half of that key was never exercised — and a future site that did would have had
    /// its comparisons folded into an accumulate row.
    #[test]
    fn one_path_with_two_kinds_stays_two_rows() {
        let mut a = Audit::new();
        a.charge("layer", OpKind::Ac, "test reason, long enough", 4, 2).unwrap();
        a.charge("layer", OpKind::Compare, "test reason, long enough", 3, 3).unwrap();
        a.charge("layer", OpKind::Ac, "test reason, long enough", 4, 1).unwrap();
        assert_eq!(a.sites().len(), 2, "the two kinds merged into one row");
        let ac = a.site("layer", OpKind::Ac).expect("the accumulate row");
        assert_eq!((ac.dense, ac.effective), (8, 3), "the two accumulate charges did not merge");
        let cmp = a.site("layer", OpKind::Compare).expect("the comparison row");
        assert_eq!((cmp.dense, cmp.effective), (3, 3), "the comparison absorbed the accumulates");
        assert_eq!(a.effective_of(OpKind::Ac), 3);
        assert_eq!(a.effective_of(OpKind::Compare), 3);
    }

    /// `first_spike_step` returns `None` for three different reasons and the doc now names all
    /// three. The third is reachable: a long enough time constant pushes a neuron that **does**
    /// fire past `u32::MAX` steps, and letting the cast saturate would claim a first spike at
    /// 4294967295 — a step no run reaches.
    #[test]
    fn a_first_spike_beyond_u32_is_none_and_not_a_saturated_step() {
        let slow = LifLayer::new(1, 1, 1e10, 0.5, 0.0).unwrap();
        // Not the sub-threshold case: the drive is well above v_th - v_reset.
        assert!(1.0 > slow.v_th(), "this neuron is above threshold and does eventually fire");
        assert!(slow.decay() < 1.0, "it does leak, so the closed form applies");
        // ln(0.5) / ln(1 - 1e-10) is 6.93e9, past the end of a u32.
        assert_eq!(slow.first_spike_step(1.0), None, "a saturated cast would answer u32::MAX");
        // The same drive at an ordinary time constant fires on step 1.
        assert_eq!(LifLayer::new(1, 1, 2.0, 0.5, 0.0).unwrap().first_spike_step(1.0), Some(1));
        // And the non-finite case, which is the first of the three.
        assert_eq!(slow.first_spike_step(f64::NAN), None);
        assert_eq!(slow.first_spike_step(f64::INFINITY), None);
    }

    /// ⭐ [`Split::multiplies`] is the one row of the split no test ever read as non-zero.
    ///
    /// Every model in this module runs `tau = 2` and a scale that is either exactly 1 or a power of
    /// two, so neither site that can emit an [`OpKind::Mul`] — a leak that is not a power of two,
    /// an unfolded non-power-of-two attention scale — is ever live, and every assertion on
    /// `multiplies` anywhere in the suite is `== 0`. A row of zeros agrees with a field wired to
    /// zero, so the whole bare-multiply column could be dropped out of the split, and with it out
    /// of [`Audit::effective_total`] and the honest fraction, without a test moving.
    ///
    /// `tau = 3` makes the leak `1/3`, which is not an exact power of two, so the leak site is
    /// charged as a bare multiply: 2 tokens x 3 channels = 6 neurons, one each.
    #[test]
    fn a_bare_multiply_appears_in_the_split_and_in_the_grand_total() {
        let mut n = LifLayer::new(2, 3, 3.0, 1.0, 0.0).unwrap();
        let mut audit = Audit::new();
        n.forward(&Tensor::silent(2, 3).unwrap(), "n", &mut audit).unwrap();

        let s = audit.split();
        assert_eq!(s.multiplies, 6, "the split's bare-multiply row is not the leak count");
        assert_eq!(
            s.multiplies,
            audit.effective_of(OpKind::Mul),
            "the split's multiply row disagrees with the aggregate it is built from"
        );
        assert_eq!(s.shifts, 0, "1/3 is not a shift, so nothing may land in that row");
        // By hand: `v_reset` is zero, so the membrane update is two adds per neuron and not three,
        // and every neuron pays one threshold comparison.
        assert_eq!((s.adds, s.comparisons), (12, 6));
        assert_eq!(s.total(), Some(24), "the grand total must carry the bare multiplies");
        assert_eq!(audit.effective_total(), Some(24));

        // And the honest fraction is taken against that total, so a dropped row moves it: 24
        // accumulates against 24 + 24 is one half, while against 18 + 24 it is 0.5714...
        audit.charge("syn", OpKind::Ac, "test reason, long enough", 24, 24).unwrap();
        assert_eq!(audit.ac_fraction(), Some(0.5), "the fraction is taken against a short total");
    }

    /// ⭐ `None` from [`Audit::ac_fraction`] on a pass that performed nothing, which is a different
    /// statement from a fraction of zero and from an unrepresentable total.
    ///
    /// Nothing in the suite called either fraction on an audit whose effective total is zero: the
    /// silent-model test still pays 112 membrane operations, so its total is 112 and its fraction
    /// is `Some(0.0)`. With the zero guard removed the division is `0.0 / 0.0`, and the only reason
    /// that is not `Some(0.0)` is that it is `Some(NaN)` — a value that compares unequal to itself,
    /// so an assertion written as a tolerance around zero would have passed either way.
    #[test]
    fn a_pass_that_performed_nothing_has_no_fraction_rather_than_a_nan() {
        let empty = Audit::new();
        assert_eq!(empty.effective_total(), Some(0), "an empty audit's total is zero, not `None`");
        assert_eq!(empty.ac_fraction(), None, "zero over zero is not a fraction");
        assert_eq!(empty.synaptic_ac_fraction(), None);

        // And an audit that charged dense work and no effective work: the model is there, the data
        // made none of it fire, and the total is still zero.
        let mut a = Audit::new();
        a.charge("dead", OpKind::Ac, "test reason, long enough", 64, 0).unwrap();
        assert_eq!(a.effective_total(), Some(0));
        assert_eq!(a.ac_fraction(), None, "a pass that did nothing is not 0% accumulates");
        assert_eq!(a.dense_of(OpKind::Ac), 64, "the dense side is unchanged: the model is there");

        // The report names which of the two `None`s it hit, and "nothing ran" is not "NaN".
        let text = a.to_string();
        assert!(text.contains("AC share of ALL operations: nothing ran"), "{text}");
    }

    /// ⭐ [`Audit::synaptic_ac_fraction`] refuses a synaptic total that does not fit in a `u64`.
    ///
    /// The existing overflow fixture could not see the difference between refusing and wrapping:
    /// it charged `u64::MAX` accumulates and **one** multiply-accumulate, and
    /// `u64::MAX.wrapping_add(1)` is exactly `0` — the single value the `syn == 0` guard on the
    /// next line also turns into `None`. The wrap landed on the one result indistinguishable from
    /// the refusal. Two multiply-accumulates wrap to `1` instead, and a wrapped total of one
    /// reports 18446744073709551615 accumulates as a share of a single synaptic operation.
    #[test]
    fn a_synaptic_total_that_wraps_past_one_is_still_refused() {
        let mut a = Audit::new();
        a.charge("ac", OpKind::Ac, "test reason, long enough", u64::MAX, u64::MAX).unwrap();
        a.charge("mac", OpKind::Mac, "test reason, long enough", 2, 2).unwrap();
        assert_eq!(a.effective_of(OpKind::Ac), u64::MAX);
        assert_eq!(a.effective_of(OpKind::Mac), 2, "the fixture must not wrap back to zero");
        assert_eq!(
            a.synaptic_ac_fraction(),
            None,
            "a wrapped synaptic total was reported as a fraction"
        );
        let text = a.to_string();
        assert!(
            text.contains("AC share of synaptic operations: the total does not fit in u64"),
            "{text}"
        );
    }

    /// The general constructor's own shape refusals. [`Tensor::silent`] and
    /// [`Tensor::sinusoidal_spikes`] each carry their own zero-dimension checks and both are
    /// tested; [`Tensor::new`]'s were reached only through `Tensor::real(0, 2, vec![])`, which is
    /// the **token** clause. Nothing anywhere handed it a zero-channel shape or a values vector of
    /// the wrong length, and either would have produced a tensor that reads as "nothing cost
    /// anything" instead of an error.
    #[test]
    fn the_general_tensor_constructor_refuses_a_zero_channel_shape_and_a_short_values_vector() {
        assert!(matches!(
            Tensor::new(2, 0, vec![], Domain::Binary),
            Err(AttnError::Empty { what: "channels" })
        ));
        assert!(matches!(
            Tensor::new(0, 2, vec![], Domain::Binary),
            Err(AttnError::Empty { what: "tokens" })
        ));
        assert!(matches!(
            Tensor::new(2, 2, vec![1.0, 0.0, 1.0], Domain::Binary),
            Err(AttnError::BadShape { what: "tensor", got: 3, want: 4 })
        ));
        assert!(matches!(
            Tensor::new(2, 2, vec![0.0; 5], Domain::Real),
            Err(AttnError::BadShape { what: "tensor", got: 5, want: 4 })
        ));
        // The exact length is accepted, so this is a length check and not a refusal of everything.
        assert_eq!(Tensor::new(2, 2, vec![0.0; 4], Domain::Real).unwrap().values().len(), 4);
    }

    /// [`Tensor::at`] answers `None` when **either** index is past the end, and only the token
    /// index was ever asked. A tensor is stored row-major, so dropping the channel clause does not
    /// read past the allocation — it reads the **next token's** row and returns a perfectly
    /// plausible spike. On the 2x2 tensor below `at(0, 2)` is the element `at(1, 0)` answers.
    #[test]
    fn a_channel_index_past_the_end_is_none_and_not_the_next_tokens_row() {
        let t = Tensor::spikes(2, 2, &[false, false, true, true]).unwrap();
        assert_eq!(t.at(1, 0), Some(1.0), "the value the missing clause would have leaked");
        assert_eq!(t.at(0, 0), Some(0.0), "and the one token 0 really carries");
        assert_eq!(t.at(0, 2), None, "a channel past the end read into the next token's row");
        assert_eq!(t.at(0, 9), None);
        assert_eq!(t.at(2, 0), None, "the token clause still holds");
        assert_eq!(t.at(1, 2), None, "the last row's overrun falls off the end of the values");
        // The row accessor's bound on the same tensor, which the element accessor shares.
        assert_eq!(t.row(0), Some(&[0.0, 0.0][..]));
        assert_eq!(t.row(1), Some(&[1.0, 1.0][..]));
        assert_eq!(t.row(2), None);
    }

    /// [`Linear::new`] documents `BadShape` "if `w.len()` is not `n_in * n_out` **or the bias is
    /// not `n_out` long**", and only the weight half and the non-finite bias could be made to fire.
    /// A short bias is not caught downstream either: `forward` indexes `b[j]` for every `j` below
    /// `n_out` and would panic several frames from the caller who supplied it.
    #[test]
    fn a_bias_of_the_wrong_length_is_refused_at_construction() {
        assert!(matches!(
            Linear::new(2, 2, eye(2), Some(vec![0.0])),
            Err(AttnError::BadShape { what: "bias", got: 1, want: 2 })
        ));
        assert!(matches!(
            Linear::new(2, 3, vec![0.0; 6], Some(vec![0.0; 4])),
            Err(AttnError::BadShape { what: "bias", got: 4, want: 3 })
        ));
        // The length is `n_out` and not `n_in`: a 2 -> 3 projection takes a bias of three.
        assert!(Linear::new(2, 3, vec![0.0; 6], Some(vec![0.0; 3])).is_ok());
        assert!(matches!(
            Linear::new(2, 3, vec![0.0; 6], Some(vec![0.0; 2])),
            Err(AttnError::BadShape { what: "bias", got: 2, want: 3 })
        ));
    }

    /// The first of the three meanings of `None` in [`LifLayer::first_spike_step`]: `x` is not
    /// finite, so there is no trajectory to solve.
    ///
    /// Every existing non-finite assertion runs at `tau > 1`, where that guard decides nothing —
    /// the ratio comes out `NaN`, `NaN.ln() / d.ln()` is `NaN`, and the `!t.is_finite()` test
    /// further down answers `None` on its own. The one place the guard is the only cover is
    /// `tau == 1`, where `decay()` is exactly zero and the branch above the sub-threshold test
    /// answers `x + v_reset >= v_th` directly: for an infinite drive that comparison is **true**,
    /// so without the guard an infinity is reported as a neuron that fires on timestep 1.
    #[test]
    fn a_non_finite_drive_has_no_first_spike_even_when_the_membrane_forgets_everything() {
        let forgetful = LifLayer::new(1, 1, 1.0, 1.0, 0.0).unwrap();
        assert_eq!(forgetful.decay(), 0.0, "`tau = 1` is the branch the guard is the only cover for");
        // Not vacuous: at `tau = 1` a finite drive that lands exactly on the threshold does fire on
        // the first timestep, which is the answer an infinite drive would otherwise get.
        assert_eq!(forgetful.first_spike_step(1.0), Some(1));
        assert_eq!(forgetful.first_spike_step(f64::INFINITY), None, "an infinite drive fired");
        assert_eq!(forgetful.first_spike_step(f64::NEG_INFINITY), None);
        assert_eq!(forgetful.first_spike_step(f64::NAN), None);
        // And with a reset potential below zero, where the `tau = 1` branch is ordered ahead of
        // the sub-threshold guard that would otherwise have swallowed the case.
        let shifted = LifLayer::new(1, 1, 1.0, 0.5, -1.0).unwrap();
        assert_eq!(shifted.first_spike_step(1.5), Some(1), "1.5 - 1.0 lands exactly on 0.5");
        assert_eq!(shifted.first_spike_step(f64::INFINITY), None);
    }

    /// [`LifLayer::forward`] documents `NonFinite` "if a membrane left the finite range", and
    /// nothing in the suite could reach it: every drive here is of order 1, and the update
    /// `v + (1/tau)(x - (v - v_reset))` is a contraction toward `x + v_reset` for `tau >= 1`, so
    /// from rest a single step cannot leave the range whatever finite `x` is.
    ///
    /// It takes two steps and both ends of the `f64` range. At `tau = 1` the membrane lands on its
    /// drive exactly, so one timestep at `-f64::MAX` parks it there, and the next at `+f64::MAX`
    /// forms `x - (v - v_reset)`, which is `f64::MAX + f64::MAX` and overflows to infinity. Carried
    /// forward instead of refused, that membrane compares `>= v_th` and the layer reports a spike:
    /// an infinity laundered into an ordinary bit.
    #[test]
    fn a_membrane_driven_out_of_the_finite_range_is_refused_rather_than_spiking() {
        let mut n = LifLayer::new(1, 1, 1.0, 1.0, 0.0).unwrap();
        let mut audit = Audit::new();
        let down =
            n.forward(&Tensor::real(1, 1, vec![-f64::MAX]).unwrap(), "n", &mut audit).unwrap();
        assert_eq!(down.values(), &[0.0], "a membrane at -f64::MAX is far below the threshold");
        assert_eq!(n.membranes(), &[-f64::MAX], "the membrane must really be parked there");

        let e =
            n.forward(&Tensor::real(1, 1, vec![f64::MAX]).unwrap(), "n", &mut audit).unwrap_err();
        assert!(matches!(e, AttnError::NonFinite { what: "membrane", index: 0 }), "{e}");
        // The arithmetic the refusal is about, stated without the layer.
        assert!(!(f64::MAX + f64::MAX).is_finite(), "the intermediate really does overflow");
    }

    /// ⭐ The whole **effective** side of [`Order::ValuesFirst`] was unwatched. The order is
    /// exercised for its output values and its dense counts only, and the one invariant that would
    /// have caught a miscount — effective never above dense — runs on a `ScoresFirst` model.
    /// Neither miscount can trip it in any case: counting every operand pair of a live row still
    /// gives at most `nnz x d_v`, which is below `d x n_k x d_v`.
    ///
    /// Both of this order's products skip a pair when the **second** operand is zero, and both
    /// counts below are computed from the fixture by hand rather than read back from the code.
    #[test]
    fn the_values_first_order_counts_only_the_pairs_where_both_operands_fired() {
        let q = Tensor::spikes(2, 2, &[true, false, false, true]).unwrap();
        let k = Tensor::spikes(2, 2, &[true, true, false, true]).unwrap();
        let v = Tensor::spikes(2, 2, &[true, false, false, true]).unwrap();
        let mut a = Audit::new();
        let o = attend(&q, &k, &v, Order::ValuesFirst, "t", &mut a).unwrap();

        // G = Kᵀ V by hand. Row 0 of K is (1,1) and row 0 of V is (1,0), giving two live pairs at
        // (c, e) = (0,0) and (1,0); row 1 of K is (0,1) and row 1 of V is (0,1), giving one at
        // (1,1). Three — not the six that charging every value channel against every live key
        // entry would give.
        let gram = a.site("t.gram", OpKind::Ac).expect("both operands are binary");
        assert_eq!(gram.dense, 2 * 2 * 2, "d x n_k x d_v");
        assert_eq!(gram.effective, 3, "the gram counted an operation against a silent value");
        assert!(gram.effective < gram.dense, "a fixture with no zeros in V could not tell");

        // G is then ((1,0),(1,1)). Q row 0 is (1,0), so only c = 0 contributes and only e = 0 of
        // G's row 0 is live: one pair. Q row 1 is (0,1), so c = 1, and both entries of G's row 1
        // are live: two pairs. Three — not the four that charging every e would give.
        let att = a.site("t.attend", OpKind::Ac).expect("the query operand is binary");
        assert_eq!(att.dense, 2 * 2 * 2, "n_q x d x d_v");
        assert_eq!(att.effective, 3, "the attend counted a pair against a zero of G");
        assert!(att.effective < att.dense, "a G with no zeros could not tell");

        // The two orders are the same product, so the values agree and the counts above are counts
        // of the same arithmetic rather than of a different one.
        let mut b = Audit::new();
        let o2 = attend(&q, &k, &v, Order::ScoresFirst, "t", &mut b).unwrap();
        assert_eq!(o.values(), o2.values(), "the two association orders disagreed");
        assert_eq!(o.values(), &[1.0, 0.0, 1.0, 1.0], "hand-computed Q Kᵀ V");
        assert!(o.nonzero() > 0, "a silent product would make every count above vacuous");
    }

    /// `Ssa::new`'s `Empty { what: "heads" }` refusal. `heads_must_divide_the_channel_count` runs
    /// 4-of-6 and 3-of-6 and never 0, and with that guard gone zero does not even panic in the
    /// division below it: `usize::is_multiple_of(0)` is `self == 0`, so a six-channel model answers
    /// `HeadsDoNotDivide { channels: 6, heads: 0 }` and the named refusal is simply never reached.
    #[test]
    fn a_head_count_of_zero_is_empty_rather_than_indivisible() {
        let lin = || Linear::new(6, 6, eye(6), None).unwrap();
        let e = Ssa::new(
            2,
            0,
            lin(),
            lin(),
            lin(),
            lin(),
            2.0,
            1.0,
            0.0,
            1.0,
            false,
            Order::ScoresFirst,
        )
        .unwrap_err();
        assert!(matches!(e, AttnError::Empty { what: "heads" }), "{e}");
    }

    /// [`Position::reset_state`] on the one variant that has state to forget. The existing encoder
    /// test ends with a bare `p.reset_state()` and asserts nothing after it — a call with no
    /// observation, so the body could be emptied and the suite stayed green. The membranes are read
    /// back here, and the reset potential is deliberately **not** zero so that "parked at rest" and
    /// "zeroed" are two different tables.
    #[test]
    fn the_conditional_encoders_reset_returns_its_membranes_to_their_reset_potential() {
        let conv = DepthwiseConv1d::centre_tap(2, 3).unwrap();
        // A threshold of 5 is out of reach in one step, so the membranes hold their charge instead
        // of being reset by a spike of their own.
        let neuron = LifLayer::new(3, 2, 2.0, 5.0, -0.25).unwrap();
        let mut p = Position::Conditional(conv, neuron);
        let x = Tensor::spikes(3, 2, &[true, false, false, true, true, true]).unwrap();
        let mut audit = Audit::new();
        let y = p.apply(&x, "position", &mut audit).unwrap();
        assert_eq!(y.values(), x.values(), "nothing fired, so the shortcut added zero");

        // v = v_reset, so `v - v_reset` is exactly zero and h = -0.25 + 0.5 * pre, with pre the
        // identity convolution of the input.
        let charged = match &p {
            Position::Conditional(_, n) => n.membranes().to_vec(),
            _ => unreachable!("the variant under test"),
        };
        assert_eq!(charged, vec![0.25, -0.25, -0.25, 0.25, 0.25, 0.25], "hand-computed membranes");
        assert!(
            charged.iter().any(|v| *v != -0.25),
            "a reset with nothing to forget would prove nothing"
        );

        p.reset_state();
        let rested = match &p {
            Position::Conditional(_, n) => n.membranes().to_vec(),
            _ => unreachable!("the variant under test"),
        };
        assert_eq!(rested, vec![-0.25; 6], "the conditional encoder kept its membranes");
    }

    /// ⭐ [`Model::run`] does **not** reset first, which is the whole difference between a model
    /// that can be driven one timestep at a time and one that cannot. Every existing test either
    /// runs once or calls `reset_state` before running again, so a reset at the top of `run` was
    /// invisible.
    ///
    /// The model below has a closed-form first-spike latency of 3, so the two drivers agree only if
    /// the state carries across calls: three separate one-timestep runs must reproduce the three
    /// outputs of one three-timestep run, spike for spike.
    #[test]
    fn a_run_continues_from_the_state_the_previous_one_left() {
        let d = 2;
        let build = || {
            let scaled = |g: f64| {
                let mut w = eye(d);
                for v in &mut w {
                    *v *= g;
                }
                Linear::new(d, d, w, None).unwrap()
            };
            let attn = Ssa::new(
                2,
                1,
                scaled(0.6),
                scaled(0.6),
                scaled(0.6),
                scaled(1.0),
                2.0,
                0.5,
                0.0,
                1.0,
                false,
                Order::ScoresFirst,
            )
            .unwrap();
            let mlp = SpikingMlp::new(2, scaled(1.0), scaled(1.0), 2.0, 0.5, 0.0).unwrap();
            let block = Block::new(attn, mlp, Residual::None).unwrap();
            Model::new(2, Position::None, vec![block]).unwrap()
        };
        let x = Tensor::spikes(2, 2, &[true, false, false, true]).unwrap();

        let mut batched = build();
        let (all, batch_audit) = batched.run(&[x.clone(), x.clone(), x.clone()]).unwrap();
        assert_eq!(batch_audit.timesteps(), 3);
        assert_eq!(all[2].values(), &[1.0, 0.0, 0.0, 1.0], "the third timestep must fire");
        assert_ne!(all[0].values(), all[2].values(), "without integration this proves nothing");

        let mut stepped = build();
        let mut seen: Vec<Tensor> = Vec::new();
        for _ in 0..3 {
            let (out, audit) = stepped.run(std::slice::from_ref(&x)).unwrap();
            assert_eq!(audit.timesteps(), 1, "each call audits only the timesteps it ran");
            seen.push(out.into_iter().next().expect("one input, one output"));
        }
        for t in 0..3 {
            assert_eq!(
                seen[t].values(),
                all[t].values(),
                "timestep {t} differs: run() reset the membranes it was handed"
            );
        }
    }

    /// [`Model::run`]'s `Empty { what: "inputs" }`. The only empty-sequence assertion in the module
    /// is on [`rate_readout`], which carries a refusal of its own; an empty run would otherwise
    /// come back `Ok` with no outputs and a fresh audit — a report of a pass that never happened,
    /// reading as "nothing cost anything".
    #[test]
    fn an_empty_input_sequence_is_refused_rather_than_audited_as_a_free_pass() {
        let mut m = toy();
        let e = m.run(&[]).unwrap_err();
        assert!(matches!(e, AttnError::Empty { what: "inputs" }), "{e}");
        // One input is enough, so the refusal is about emptiness and not about the call.
        let (outs, audit) = m.run(&[Tensor::silent(2, 2).unwrap()]).unwrap();
        assert_eq!(outs.len(), 1);
        assert_eq!(audit.timesteps(), 1);
    }
}
