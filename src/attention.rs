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
//! accumulates**. Normalisation is a real multiply per element. The scalar `s` is a real multiply per
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

    /// Effective operations of one kind.
    #[must_use]
    pub fn effective_of(&self, kind: OpKind) -> u64 {
        self.sites.iter().filter(|s| s.kind == kind).map(|s| s.effective).sum()
    }

    /// Dense operations of one kind.
    #[must_use]
    pub fn dense_of(&self, kind: OpKind) -> u64 {
        self.sites.iter().filter(|s| s.kind == kind).map(|s| s.dense).sum()
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
    /// fraction of zero.
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
    /// report perfect spike-drivenness.
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
    /// Saturating rather than wrapping. A workload reaching `u64::MAX` multiplies is not one this
    /// crate can simulate, but a wrapped count would report an enormous number as a small one —
    /// the failure mode [`crate::metrics::MetricError`]'s overflow variant exists to prevent.
    #[must_use]
    pub fn multiplier_ops(&self) -> u64 {
        self.effective_of(OpKind::Mac).saturating_add(self.effective_of(OpKind::Mul))
    }

    /// This pass expressed in `NeuroBench`'s [`crate::metrics::SynOps`].
    ///
    /// **Lossy, deliberately, and that is the finding.** `SynOps` has exactly three slots — dense,
    /// effective `MACs`, effective `ACs` — so the adds, multiplies, shifts, comparisons and gates
    /// that [`Audit::split`] counts have nowhere to go. Comparing `synops().effective_total()` with
    /// [`Audit::effective_total`] measures how much of a spiking transformer's arithmetic the
    /// field's own benchmark metric cannot see.
    #[must_use]
    pub fn synops(&self) -> SynOps {
        SynOps {
            dense: self.dense_of(OpKind::Ac) + self.dense_of(OpKind::Mac),
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
        match self.synaptic_ac_fraction() {
            Some(x) => writeln!(f, "  AC share of synaptic operations (as reported): {x:.4}")?,
            None => writeln!(f, "  AC share of synaptic operations: no synaptic operations")?,
        }
        match self.ac_fraction() {
            Some(x) => write!(f, "  AC share of ALL operations: {x:.4}"),
            None => write!(f, "  AC share of ALL operations: nothing ran"),
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
    /// [`AttnError::Empty`] for a zero dimension.
    pub fn silent(tokens: usize, channels: usize) -> Result<Self, AttnError> {
        Self::new(tokens, channels, vec![0.0; tokens.max(1) * channels.max(1)], Domain::Binary)
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
    /// [`AttnError::Empty`] for a zero dimension.
    pub fn sinusoidal_spikes(tokens: usize, channels: usize) -> Result<Self, AttnError> {
        if tokens == 0 {
            return Err(AttnError::Empty { what: "tokens" });
        }
        if channels == 0 {
            return Err(AttnError::Empty { what: "channels" });
        }
        let d = channels as f64;
        let mut bits = Vec::with_capacity(tokens * channels);
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
    /// spike occurred. Used to check [`LifLayer::forward`] against something other than itself.
    #[must_use]
    pub fn relaxed(&self, v0: f64, steps: u32) -> f64 {
        self.v_reset + (v0 - self.v_reset) * self.decay().powi(steps as i32)
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
/// [`Ssa::new`] refuses the fold when either condition fails rather than performing it approximately:
/// a non-power-of-two `s` would make the two paths differ in the last place, and a non-zero
/// `v_reset` would break the linearity the argument rests on. This crate did not locate the fold
/// described in the spiking-transformer literature; it is offered here as an exact rewrite with the
/// conditions stated, not as a reproduction of anyone's result.
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
    pub gain: f64,
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
        }
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
            let wq = Linear::new(d, d, draw(self.gain, d, d, rng), None)?;
            let wk = Linear::new(d, d, draw(self.gain, d, d, rng), None)?;
            let wv = Linear::new(d, d, draw(self.gain, d, d, rng), None)?;
            let wo = Linear::new(d, d, draw(self.gain, d, d, rng), None)?;
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
            let fc1 =
                Linear::new(d, self.mlp_hidden, draw(self.gain, d, self.mlp_hidden, rng), None)?;
            let fc2 =
                Linear::new(self.mlp_hidden, d, draw(self.gain, self.mlp_hidden, d, rng), None)?;
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
    #[test]
    fn a_model_built_for_two_tokens_refuses_three() {
        let mut m = toy();
        let wide = Tensor::spikes(3, 2, &[true; 6]).unwrap();
        let err = m.forward(&wide, &mut Audit::new()).unwrap_err();
        assert!(matches!(err, AttnError::BadShape { got: 6, want: 4, .. }), "{err}");
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
        assert_eq!(Domain::Integer(u32::MAX).sum(Domain::Binary), Domain::Integer(u32::MAX));
        assert_eq!(Domain::Binary.bound(), Some(1));
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
            assert!(s.because.len() > 20, "{} has no reason", s.path);
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
}
