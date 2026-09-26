//! Integer and multi-bit spikes: an activation becomes an integer spike count, and the count
//! becomes a spike train in one of three codings — the scheme that lets a spiking network run at the
//! scale of a language model.
//!
//! # What the mechanism is
//!
//! A binary spike carries one bit per neuron per time step. A network that has to represent a
//! real-valued activation with binary spikes spends time steps doing it — a rate code needs as many
//! steps as the precision it wants — and at language-model scale that cost is the whole problem. The
//! answer a line of work from the Institute of Automation, Chinese Academy of Sciences, converges on
//! is to let the neuron emit an **integer** while training, and to recover spikes only at inference,
//! by expanding the integer over "virtual" time steps.
//!
//! `SpikingBrain` (Pan, Feng, Zhuang, Ding, Xu, … Xu and Li, *`SpikingBrain`: Spiking Brain-inspired
//! Large Models*, `arXiv`:2509.05276v4, 8 May 2026, first submitted 5 September 2025, §3.3) states the
//! version implemented here. Every equation number in this module is v4's, which v3 (1 December 2025)
//! shares; v1 and v2 number the same equations differently — the adaptive threshold is v1's Eq. 22
//! and v2's Eq. 23 — so a citation without the version points at the wrong equation.
//!
//! - an **adaptive threshold** set by the input itself, `V_th(x) = mean(|x|) / k` (its Eq. 19), where
//!   the hyperparameter `k` sets the firing rate;
//! - an **integer spike count** `s_INT = round(x / V_th(x))` (its Eq. 21);
//! - the **output** `y = V_th(x) · Σ_t W s_t` (its Eq. 22) — one scale, applied once, after an
//!   accumulation made of additions;
//! - three ways to spend the count as spikes. **Binary**, `{0, 1}`, one spike per unit of count, which
//!   the paper says "only supports positive integers". **Ternary**, `{−1, 0, 1}`, one signed spike per
//!   unit. **Bitwise**, one time step per bit, in the third of the paper's three bitwise forms: "Two's
//!   complement encoding incorporates sign information into the highest bit". Over `b` bits this
//!   module reads a train as `s_INT = −2^(b−1)·s_b + Σ_{t<b} 2^(t−1)·s_t`, least significant bit first.
//!
//! The integer-training line it builds on is Luo, Yao, Chou, Xu and Li, *Integer-Valued Training and
//! Spike-Driven Inference Spiking Neural Network for High-performance and Energy-efficient Object
//! Detection* (ECCV 2024, `arXiv`:2407.20708), whose I-LIF neuron keeps integers in training and
//! extends virtual time steps at inference; and Yao, Qiu, Hu, Hu, Chou, Tian, Liao, Leng, Xu and Li,
//! *Scaling Spike-driven Transformer with Efficient Spike Firing Approximation Training* (IEEE TPAMI,
//! 2025, `arXiv`:2411.16061). This module transcribes only the `SpikingBrain` equations, which are
//! the ones this review read.
//!
//! # ⚠ Two citations this module used to get wrong
//!
//! - **The version.** The citation above used to read "`arXiv`:2509.05276, September 2025" — the date
//!   of v1, in whose numbering no equation number given here names the same equation: v1's threshold,
//!   count and output are its Eqs. 22, 25 and 26, and this review located no formula in its bitwise
//!   text.
//! - **The bitwise formula.** This module used to give the bitwise coding as `s_INT = Σ_t 2^(t−1) s_t`,
//!   "its Eq. 23". Eq. 23 is the paper's formula for its SECOND bitwise form, which "uses ±1 to
//!   represent each bit", and the text introduces it as such: "Taking Bidirectional bitwise encoding
//!   as an example, its encoding formula ... can be expressed as". Every weight in it is positive, so
//!   it is not the code implemented here. This review did not locate a formula for the two's
//!   complement form in §3.3.2. The one instance of it this review found is Figure 4(c), which draws
//!   −5 as `1 0 1 1`, most significant bit first; this module emits the same code least significant
//!   first, `1, 1, 0, 1`, the order Eq. 23's weights `2^(t−1)` imply. What the code shares with Eq. 23
//!   is a positional sum; the negative top-bit weight is this module's own, the standard two's
//!   complement one. Docs only: the top bit has always weighed `−2^(b−1)` here, and a test now reads
//!   one train both ways.
//!
//! # What this module checks, and against what
//!
//! Every claim is an identity, so every check is an equality rather than a tolerance:
//!
//! - **Expansion loses nothing.** Each coding's train reconstructs its count exactly — exhaustively,
//!   for every count a bit width can hold.
//! - **Addition-only is exact.** With integer weights — the paper's INT8 setting — the accumulate-only
//!   product over the expanded trains, [`spike_driven`], equals the dense integer product,
//!   [`dense`], to the last bit. The accumulation in [`spike_driven`] is written with additions,
//!   subtractions and shifts only, so the claim is about the code as written and not about an
//!   equivalent formula.
//! - **Quantisation is bounded.** `|x − V_th · s_INT| ≤ V_th / 2` for every element — the rounding
//!   bound, [`Encoded::max_error`].
//!
//! # ⚠ What "halves the time steps" means
//!
//! The paper says ternary coding "halves the number of time steps" relative to binary, with the
//! example that "a count of 256 requires 256 consecutive time steps in binary coding, 128 in ternary
//! coding, but only 8 steps in 8-bit bitwise encoding". **For one count that is not so**: a ternary
//! train for 256 is 256 spikes of `+1`, and [`Coding::steps`] says 256; nor does 256 fit an 8-bit
//! code, signed or unsigned, so the bitwise leg of the example fails too. What IS halved is the worst
//! case over a range of the same size. Two hundred and fifty-six levels unsigned are `0..=255`, at
//! most 255 binary steps; two hundred and fifty-six levels signed are `−128..=127`, at most 128
//! ternary steps. The saving is a statement about ranges, and the test that pins it says so.
//!
//! # ⚠ The energy claim, and the term it leaves out
//!
//! The paper's §5.5 reports a 97.7% energy reduction against FP16 multiply-accumulates and 85.2%
//! against INT8, from per-operation energies "based on published hardware energy consumption data at
//! 45nm technology" — 1.5 pJ an FP16 MAC, 0.23 pJ an INT8 MAC, 0.03 pJ an INT8 addition — and 1.13
//! spikes fired on average per channel. [`SPIKINGBRAIN_45NM`] carries those numbers. Three things
//! about the claim are this crate's business:
//!
//! 1. **It is arithmetic only.** The model is spikes times the energy of an addition, its Eq. 24,
//!    `E = Average Spikes × E_INT8Add`; memory access and data movement are not in it, and nothing was
//!    measured, so it is graded [`Evidence::Derived`].
//! 2. **Once a weight read is priced, the answer depends on the dataflow.** A spike adds one column of
//!    `W` to the accumulator. If that column is fetched once per SPIKE, 1.13 spikes per activation
//!    reads 13% more weights than a dense pass, and past a fetch energy the paper's own figures
//!    determine — [`ArithmeticModel::break_even_fetch_pj`], 1.508 pJ against INT8 — the dense layer is
//!    cheaper. If the column is HELD while all of one input's spikes are applied, it is read once per
//!    active input, and with 18.4% of channels silent that is fewer reads than a dense pass, so the
//!    spiking layer wins at any fetch energy. Same arithmetic, opposite verdicts. That is the
//!    crossover argument this crate is built on (see [`crate::crossover`]), arriving from the other
//!    direction.
//!
//!    The paper's own memory claim is the held case. Its §5.5 says the weight fetches of the silent
//!    channels "are skipped (including data transfer from off-chip DRAM to on-chip SRAM and from SRAM
//!    to compute units), thereby proportionally reducing memory access overhead", and v4's Appendix
//!    C.2 adds "The analysis assumes channel-level skipping." Reads that fall in proportion to the
//!    silent fraction need an active channel's column read once, [`Dataflow::HoldPerInput`]; fetched
//!    per spike, 1.13 spikes a channel read more than a dense pass, not fewer. Skipping a silent
//!    channel does not by itself pick between the two, since a silent channel fetches nothing under
//!    either. This review did not locate a statement of whether an active channel carrying several
//!    spikes has its column re-read at each one, and neither dataflow is priced: the §5.5 figures are
//!    additions alone.
//!
//!    **Correction.** This item used to say the verdict depends on the dataflow "and the paper does
//!    not say which". The paper is not silent: its "proportionally" implies the held dataflow, under
//!    which the spiking layer wins at any fetch energy, though it never names one. The fetch-per-spike
//!    break-even stands as what the claim costs if an active channel's column is re-read per spike.
//!    Docs only; a test now counts both reads on one layer.
//! 3. **Its two ratios imply two different spike energies.** The text gives "about 0.034 pJ" per
//!    MAC-equivalent, and 0.23 / 0.034 is the paper's 6.76×. But 1.5 / 0.034 is 44.1×, and the
//!    paper's 43.48× against FP16 needs 0.0345 pJ — 1.15 spikes rather than 1.13. Small, and recorded
//!    rather than repeated.
//!
//! # Where it sits
//!
//! [`crate::coding`] and [`crate::encode`] spend binary spikes on real values over many steps;
//! [`crate::attention`] counts what in a spiking transformer is actually a spike. This module is the
//! scale-up path those two do not take: fewer steps, integer-valued spikes, and an accumulation that
//! is exact.

use core::fmt;

use crate::ledger::Evidence;

/// The longest binary or ternary train [`Coding::expand`] will build: `2^20` steps.
///
/// A binary or ternary train is as long as its count, so a count of `2^40` would ask for a terabyte
/// of spikes. The refusal names the length rather than attempting the allocation.
pub const MAX_STEPS: usize = 1 << 20;

/// The largest `|x / V_th|` [`encode`] will round: `2^53`, past which consecutive integers are no
/// longer representable in an `f64` and "round to the nearest integer" stops meaning anything.
pub const MAX_RATIO: f64 = 9_007_199_254_740_992.0;

/// Why a spike coding could not be carried out.
#[derive(Debug, Clone, PartialEq)]
pub enum IntSpikeError {
    /// No activations were given, so there is no mean to set a threshold from.
    Empty,
    /// An activation that is not a finite number.
    NonFinite {
        /// Its position.
        index: usize,
        /// Its value.
        value: f64,
    },
    /// The rate hyperparameter `k` was not a finite positive number.
    BadRate {
        /// The value given.
        k: f64,
    },
    /// `|x / V_th|` exceeded [`MAX_RATIO`], where rounding to an integer stops being exact.
    TooLarge {
        /// The position of the first such activation.
        index: usize,
        /// The ratio it produced.
        ratio: f64,
    },
    /// Binary coding was asked to encode a negative count, which it cannot represent.
    Unsigned {
        /// The count.
        count: i64,
    },
    /// A count outside the range a bitwise coding of this width can hold.
    OutOfRange {
        /// The count.
        count: i64,
        /// The bit width.
        bits: u32,
    },
    /// A bitwise width outside `2..=32`.
    BadBits {
        /// The width given.
        bits: u32,
    },
    /// A binary or ternary train longer than [`MAX_STEPS`].
    TooLong {
        /// The number of steps it would need.
        steps: u64,
    },
    /// A train element outside the coding's alphabet — `{0, 1}` for binary and bitwise, `{−1, 0, 1}`
    /// for ternary.
    BadSpike {
        /// Its position in the train.
        index: usize,
        /// Its value.
        value: i8,
    },
    /// An array whose length does not match what the operation requires.
    Dimension {
        /// Which array.
        what: &'static str,
        /// The length given.
        got: usize,
        /// The length required.
        want: usize,
    },
    /// An integer accumulation left the range of an `i64`.
    Overflow,
}

impl fmt::Display for IntSpikeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => f.write_str("no activations: there is no mean to set a threshold from"),
            Self::NonFinite { index, value } => write!(f, "activation {index} is {value}, which is not finite"),
            Self::BadRate { k } => write!(f, "rate k = {k} must be a finite positive number"),
            Self::TooLarge { index, ratio } => write!(
                f,
                "activation {index} is {ratio} thresholds from zero, past 2^53, where rounding to an integer is no longer exact"
            ),
            Self::Unsigned { count } => write!(f, "binary coding cannot represent the negative count {count}"),
            Self::OutOfRange { count, bits } => write!(f, "{count} does not fit a {bits}-bit two's complement code"),
            Self::BadBits { bits } => write!(f, "{bits} bits: a bitwise coding needs 2 to 32"),
            Self::TooLong { steps } => write!(f, "a train of {steps} steps exceeds the {MAX_STEPS}-step limit"),
            Self::BadSpike { index, value } => write!(f, "spike {index} is {value}, outside this coding's alphabet"),
            Self::Dimension { what, got, want } => write!(f, "{what} has length {got}, expected {want}"),
            Self::Overflow => f.write_str("the integer accumulation left the range of an i64"),
        }
    }
}

impl std::error::Error for IntSpikeError {}

/// How an integer spike count is spent as spikes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Coding {
    /// `{0, 1}`: one spike per unit of count. Non-negative counts only.
    Binary,
    /// `{−1, 0, 1}`: one signed spike per unit of count.
    Ternary,
    /// One time step per bit of a two's complement code of `bits` bits, least significant first;
    /// the top bit carries weight `−2^(bits−1)`. `SpikingBrain`'s third bitwise form, for which this
    /// review did not locate a formula in the paper; its Eq. 23 is the bidirectional form.
    Bitwise {
        /// The code width, `2..=32`.
        bits: u32,
    },
}

impl Coding {
    fn check(self) -> Result<(), IntSpikeError> {
        if let Self::Bitwise { bits } = self
            && !(2..=32).contains(&bits)
        {
            return Err(IntSpikeError::BadBits { bits });
        }
        Ok(())
    }

    /// The counts this coding can represent, inclusive.
    ///
    /// Binary `0..=MAX_STEPS`; ternary `−MAX_STEPS..=MAX_STEPS`; bitwise `−2^(b−1)..=2^(b−1) − 1`.
    /// The binary and ternary limits are this crate's allocation cap, not the coding's; the bitwise
    /// limit is the coding's own.
    ///
    /// # Errors
    ///
    /// [`IntSpikeError::BadBits`] for a bitwise width outside `2..=32`.
    pub fn range(self) -> Result<(i64, i64), IntSpikeError> {
        self.check()?;
        let cap = MAX_STEPS as i64;
        Ok(match self {
            Self::Binary => (0, cap),
            Self::Ternary => (-cap, cap),
            Self::Bitwise { bits } => (-(1i64 << (bits - 1)), (1i64 << (bits - 1)) - 1),
        })
    }

    fn admit(self, count: i64) -> Result<(), IntSpikeError> {
        let (lo, hi) = self.range()?;
        match self {
            Self::Binary if count < 0 => Err(IntSpikeError::Unsigned { count }),
            Self::Bitwise { bits } if count < lo || count > hi => Err(IntSpikeError::OutOfRange { count, bits }),
            Self::Binary | Self::Ternary if count.unsigned_abs() > hi.unsigned_abs() => {
                Err(IntSpikeError::TooLong { steps: count.unsigned_abs() })
            }
            _ => Ok(()),
        }
    }

    /// The number of time steps the train for `count` occupies.
    ///
    /// Binary `count`; ternary `|count|`; bitwise always `bits`, whatever the count.
    ///
    /// # Errors
    ///
    /// As [`Coding::expand`].
    pub fn steps(self, count: i64) -> Result<usize, IntSpikeError> {
        self.admit(count)?;
        Ok(match self {
            Self::Binary | Self::Ternary => count.unsigned_abs() as usize,
            Self::Bitwise { bits } => bits as usize,
        })
    }

    /// The number of NON-ZERO spikes the train for `count` carries — what an event-driven accumulator
    /// actually pays for.
    ///
    /// Binary `count`; ternary `|count|`; bitwise the number of ones in the two's complement code.
    ///
    /// # Errors
    ///
    /// As [`Coding::expand`].
    pub fn spikes(self, count: i64) -> Result<usize, IntSpikeError> {
        self.admit(count)?;
        Ok(match self {
            Self::Binary | Self::Ternary => count.unsigned_abs() as usize,
            Self::Bitwise { bits } => (twos_complement(count, bits)).count_ones() as usize,
        })
    }

    /// The weight step `t` of a train carries: `1` for binary and ternary, `2^t` for bitwise, and
    /// `−2^(bits−1)` for the bitwise top bit.
    ///
    /// # Errors
    ///
    /// [`IntSpikeError::BadBits`] for a bitwise width outside `2..=32`.
    pub fn step_weight(self, t: usize) -> Result<i64, IntSpikeError> {
        self.check()?;
        Ok(match self {
            Self::Binary | Self::Ternary => 1,
            Self::Bitwise { bits } if t + 1 == bits as usize => -(1i64 << t),
            Self::Bitwise { .. } => 1i64 << t,
        })
    }

    /// The spike train for `count`.
    ///
    /// Binary: `count` ones. Ternary: `|count|` copies of `sign(count)`. Bitwise: the `bits` bits of
    /// the two's complement code, least significant first.
    ///
    /// # Errors
    ///
    /// [`IntSpikeError::Unsigned`] for a negative count in binary; [`IntSpikeError::OutOfRange`] for
    /// a count a bitwise width cannot hold; [`IntSpikeError::TooLong`] for a binary or ternary train
    /// longer than [`MAX_STEPS`]; [`IntSpikeError::BadBits`] for a bitwise width outside `2..=32`.
    pub fn expand(self, count: i64) -> Result<Vec<i8>, IntSpikeError> {
        let n = self.steps(count)?;
        Ok(match self {
            Self::Binary => vec![1; n],
            Self::Ternary => vec![if count < 0 { -1 } else { 1 }; n],
            Self::Bitwise { bits } => {
                let code = twos_complement(count, bits);
                (0..n).map(|t| ((code >> t) & 1) as i8).collect()
            }
        })
    }

    /// The count a train represents: `Σ_t w_t s_t`, with `w_t` from [`Coding::step_weight`].
    ///
    /// # Errors
    ///
    /// [`IntSpikeError::BadSpike`] for an element outside the coding's alphabet;
    /// [`IntSpikeError::Dimension`] for a bitwise train whose length is not `bits`;
    /// [`IntSpikeError::BadBits`] for a bitwise width outside `2..=32`.
    pub fn reconstruct(self, train: &[i8]) -> Result<i64, IntSpikeError> {
        self.check()?;
        if let Self::Bitwise { bits } = self
            && train.len() != bits as usize
        {
            return Err(IntSpikeError::Dimension { what: "bitwise train", got: train.len(), want: bits as usize });
        }
        let lowest: i8 = if self == Self::Ternary { -1 } else { 0 };
        let mut total = 0i64;
        for (t, &s) in train.iter().enumerate() {
            if s < lowest || s > 1 {
                return Err(IntSpikeError::BadSpike { index: t, value: s });
            }
            total += i64::from(s) * self.step_weight(t)?;
        }
        Ok(total)
    }
}

/// The low `bits` bits of `count` in two's complement, as an unsigned integer.
fn twos_complement(count: i64, bits: u32) -> u64 {
    (count as u64) & ((1u64 << bits) - 1)
}

/// The adaptive threshold `V_th(x) = mean(|x|) / k` — `SpikingBrain`'s Eq. 19 (v4 numbering).
///
/// # Errors
///
/// [`IntSpikeError::Empty`] for no activations; [`IntSpikeError::NonFinite`] naming the first
/// activation that is not finite; [`IntSpikeError::BadRate`] for a `k` that is not finite and
/// positive.
pub fn adaptive_threshold(x: &[f64], k: f64) -> Result<f64, IntSpikeError> {
    if x.is_empty() {
        return Err(IntSpikeError::Empty);
    }
    if let Some((index, &value)) = x.iter().enumerate().find(|(_, v)| !v.is_finite()) {
        return Err(IntSpikeError::NonFinite { index, value });
    }
    if !(k > 0.0) || !k.is_finite() {
        return Err(IntSpikeError::BadRate { k });
    }
    let mean = x.iter().map(|v| v.abs()).sum::<f64>() / x.len() as f64;
    Ok(mean / k)
}

/// Activations as integer spike counts under one adaptive threshold.
#[derive(Debug, Clone, PartialEq)]
pub struct Encoded {
    /// The threshold `V_th`, in the activations' own units. Zero exactly when every activation is
    /// zero, in which case every count is zero too.
    pub threshold: f64,
    /// `round(x / V_th)`, one per activation.
    pub counts: Vec<i64>,
}

/// Encode activations as integer spike counts: `s_INT = round(x / V_th(x))`, `SpikingBrain`'s Eq. 21 (v4).
///
/// **The tie rule is stated because the paper does not state one.** It writes `round(·)`. This uses
/// round-half-to-even, which is IEEE 754's default rounding and the rule `torch.round` applies; at an
/// exact tie it differs from round-half-away-from-zero by one count. An input that is all zeros has a
/// threshold of zero, and its counts are zero rather than `0 / 0`.
///
/// # Errors
///
/// As [`adaptive_threshold`], plus [`IntSpikeError::TooLarge`] where `|x / V_th|` exceeds
/// [`MAX_RATIO`].
pub fn encode(x: &[f64], k: f64) -> Result<Encoded, IntSpikeError> {
    let threshold = adaptive_threshold(x, k)?;
    if threshold == 0.0 {
        return Ok(Encoded { threshold, counts: vec![0; x.len()] });
    }
    let mut counts = Vec::with_capacity(x.len());
    for (index, &v) in x.iter().enumerate() {
        let ratio = v / threshold;
        if !(ratio.abs() <= MAX_RATIO) {
            return Err(IntSpikeError::TooLarge { index, ratio });
        }
        counts.push(ratio.round_ties_even() as i64);
    }
    Ok(Encoded { threshold, counts })
}

impl Encoded {
    /// `V_th · s_INT`, one per activation: what the counts stand for.
    #[must_use]
    pub fn decode(&self) -> Vec<f64> {
        self.counts.iter().map(|&c| self.threshold * c as f64).collect()
    }

    /// The largest `|x_i − V_th · s_i|`. Rounding to the nearest integer bounds it by `V_th / 2`.
    ///
    /// # Errors
    ///
    /// [`IntSpikeError::Dimension`] if `x` is not the length these counts were encoded from.
    pub fn max_error(&self, x: &[f64]) -> Result<f64, IntSpikeError> {
        if x.len() != self.counts.len() {
            return Err(IntSpikeError::Dimension { what: "activations", got: x.len(), want: self.counts.len() });
        }
        Ok(x.iter().zip(self.decode()).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max))
    }

    /// Every count expanded under `coding`, one train per activation.
    ///
    /// # Errors
    ///
    /// As [`Coding::expand`], for the first count that fails.
    pub fn trains(&self, coding: Coding) -> Result<Vec<Vec<i8>>, IntSpikeError> {
        self.counts.iter().map(|&c| coding.expand(c)).collect()
    }

    /// The time steps a layer needs to emit all its counts under `coding`: the LONGEST train, since
    /// every activation's train runs in parallel.
    ///
    /// # Errors
    ///
    /// As [`Coding::steps`].
    pub fn steps(&self, coding: Coding) -> Result<usize, IntSpikeError> {
        self.counts.iter().try_fold(0, |m, &c| Ok(m.max(coding.steps(c)?)))
    }

    /// The non-zero spikes the whole layer emits under `coding`.
    ///
    /// # Errors
    ///
    /// As [`Coding::spikes`].
    pub fn spikes(&self, coding: Coding) -> Result<u64, IntSpikeError> {
        self.counts.iter().try_fold(0u64, |s, &c| Ok(s + coding.spikes(c)? as u64))
    }
}

/// What an accumulate-only product did.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Ops {
    /// Integer additions and subtractions into an accumulator: one per output row per non-zero spike.
    pub additions: u64,
    /// Left shifts of a weight by a bit position: one per output row per non-zero bitwise spike above
    /// the lowest bit. Zero for binary and ternary. Free in most hardware and counted anyway.
    pub shifts: u64,
    /// Column reads if a column of `W` is fetched once per non-zero SPIKE.
    pub reads_per_spike: u64,
    /// Column reads if a column is fetched once per ACTIVE INPUT and held while its spikes apply.
    pub reads_held: u64,
}

fn check_matrix(w: &[i8], rows: usize, cols: usize, inputs: usize) -> Result<(), IntSpikeError> {
    let want = rows.checked_mul(cols).ok_or(IntSpikeError::Overflow)?;
    if w.len() != want {
        return Err(IntSpikeError::Dimension { what: "weights", got: w.len(), want });
    }
    if inputs != cols {
        return Err(IntSpikeError::Dimension { what: "inputs", got: inputs, want: cols });
    }
    Ok(())
}

/// The dense integer product `Σ_i W[r][i] · counts[i]` — the reference [`spike_driven`] must equal.
///
/// `w` is row-major, `rows × cols`, in INT8 as the paper's energy model assumes.
///
/// # Errors
///
/// [`IntSpikeError::Dimension`] for a weight array that is not `rows × cols` or a count array that is
/// not `cols`; [`IntSpikeError::Overflow`] if an accumulator leaves the range of an `i64`.
pub fn dense(w: &[i8], rows: usize, cols: usize, counts: &[i64]) -> Result<Vec<i64>, IntSpikeError> {
    check_matrix(w, rows, cols, counts.len())?;
    let mut y = vec![0i64; rows];
    for r in 0..rows {
        for i in 0..cols {
            let term = i64::from(w[r * cols + i]).checked_mul(counts[i]).ok_or(IntSpikeError::Overflow)?;
            y[r] = y[r].checked_add(term).ok_or(IntSpikeError::Overflow)?;
        }
    }
    Ok(y)
}

/// The accumulate-only product `Σ_i Σ_t W[r][i] · w_t · s_{i,t}` — `SpikingBrain`'s Eq. 22 (v4) before
/// the threshold scale and, for bitwise, the two's complement form of its bitwise coding. This line
/// used to say "Eq. 23 for bitwise"; Eq. 23 is the paper's bidirectional form, whose weights are all
/// positive (see the module documentation).
///
/// Written with additions, subtractions and shifts only: a spike of `+1` adds the weight, a spike of
/// `−1` subtracts it, and a bitwise spike at position `t` shifts it left by `t` first (and subtracts
/// at the top bit, whose weight is negative). No integer multiplication appears in the loop, which is
/// the claim the paper's energy model rests on, and [`dense`] is the reference it must equal exactly.
///
/// # Errors
///
/// [`IntSpikeError::Dimension`] for a weight array that is not `rows × cols` or a train array that is
/// not `cols`; [`IntSpikeError::BadSpike`] for a spike outside the coding's alphabet;
/// [`IntSpikeError::Dimension`] for a bitwise train whose length is not `bits`;
/// [`IntSpikeError::BadBits`] for a bitwise width outside `2..=32`; [`IntSpikeError::Overflow`] if an
/// accumulator leaves the range of an `i64`.
pub fn spike_driven(
    w: &[i8],
    rows: usize,
    cols: usize,
    trains: &[Vec<i8>],
    coding: Coding,
) -> Result<(Vec<i64>, Ops), IntSpikeError> {
    coding.check()?;
    check_matrix(w, rows, cols, trains.len())?;
    let rows64 = rows as u64;
    let mut y = vec![0i64; rows];
    let mut ops = Ops::default();
    for (i, train) in trains.iter().enumerate() {
        coding.reconstruct(train)?;
        let mut active = false;
        for (t, &s) in train.iter().enumerate() {
            if s == 0 {
                continue;
            }
            active = true;
            ops.reads_per_spike += 1;
            let negative_weight = coding.step_weight(t)? < 0;
            let subtract = (s < 0) != negative_weight;
            let shift = if matches!(coding, Coding::Bitwise { .. }) { t as u32 } else { 0 };
            for r in 0..rows {
                let v = i64::from(w[r * cols + i]).checked_shl(shift).ok_or(IntSpikeError::Overflow)?;
                y[r] = if subtract { y[r].checked_sub(v) } else { y[r].checked_add(v) }.ok_or(IntSpikeError::Overflow)?;
            }
            ops.additions += rows64;
            if shift > 0 {
                ops.shifts += rows64;
            }
        }
        if active {
            ops.reads_held += 1;
        }
    }
    Ok((y, ops))
}

/// Apply the threshold once, at the end: `y = V_th · Σ`, `SpikingBrain`'s Eq. 22 (v4).
#[must_use]
pub fn scale(accumulated: &[i64], threshold: f64) -> Vec<f64> {
    accumulated.iter().map(|&a| threshold * a as f64).collect()
}

/// How a spike-driven layer reads its weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dataflow {
    /// A column of `W` is fetched for every non-zero spike: reads scale with SPIKES.
    FetchPerSpike,
    /// A column is fetched once per active input and held while all that input's spikes are
    /// applied: reads scale with ACTIVE INPUTS. The dataflow `SpikingBrain`'s §5.5 memory claim
    /// implies, since only here do reads fall "proportionally" with the silent channels.
    HoldPerInput,
}

/// Per-operation energies of an arithmetic-only model, in picojoules.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArithmeticModel {
    /// One FP16 multiply-accumulate.
    pub fp16_mac_pj: f64,
    /// One INT8 multiply-accumulate.
    pub int8_mac_pj: f64,
    /// One INT8 addition — the whole cost of a spike in this model.
    pub int8_add_pj: f64,
    /// Where the figures come from, and what the model leaves out.
    pub source: &'static str,
    /// How good [`ArithmeticModel::source`] is.
    pub evidence: Evidence,
}

/// `SpikingBrain`'s energy model, §5.5 and Eq. 24 of v4, as the paper states it.
///
/// Graded [`Evidence::Derived`]: the figures are "published hardware energy consumption data at
/// 45nm technology", combined analytically, and nothing was measured on the system the model
/// describes. Memory access and data movement are outside the model entirely.
pub const SPIKINGBRAIN_45NM: ArithmeticModel = ArithmeticModel {
    fp16_mac_pj: 1.5,
    int8_mac_pj: 0.23,
    int8_add_pj: 0.03,
    source: "Pan et al., SpikingBrain: Spiking Brain-inspired Large Models, arXiv:2509.05276v4 (8 May 2026; \
             first submitted 5 Sep 2025), section 5.5, Eq. 24: 'based on published hardware \
             energy consumption data at 45nm technology'. Arithmetic only: memory access and data \
             movement are not in the model, and nothing was measured.",
    evidence: Evidence::Derived,
};

/// Average spikes fired per channel in `SpikingBrain`'s model, §5.5: "only 1.13".
pub const SPIKINGBRAIN_SPIKES_PER_CHANNEL: f64 = 1.13;

/// Fraction of channels that fire no spike at all in `SpikingBrain`'s model, §5.5: "18.4%".
pub const SPIKINGBRAIN_SILENT_FRACTION: f64 = 0.184;

/// The per-MAC-equivalent spike energy `SpikingBrain`'s text gives: "about 0.034 pJ".
pub const SPIKINGBRAIN_SPIKE_PJ: f64 = 0.034;

/// `SpikingBrain`'s stated energy-efficiency improvement against FP16 MACs: 43.48×.
pub const SPIKINGBRAIN_RATIO_VS_FP16: f64 = 43.48;

/// `SpikingBrain`'s stated energy-efficiency improvement against INT8 MACs: 6.76×.
pub const SPIKINGBRAIN_RATIO_VS_INT8: f64 = 6.76;

impl ArithmeticModel {
    /// The spike side of the model: `spikes_per_activation × E_add`, in pJ per MAC-equivalent.
    #[must_use]
    pub fn spike_pj(&self, spikes_per_activation: f64) -> f64 {
        spikes_per_activation * self.int8_add_pj
    }

    /// The fractional saving against a dense MAC of `baseline_mac_pj`: `1 − spike_pj / baseline`.
    #[must_use]
    pub fn reduction(&self, baseline_mac_pj: f64, spikes_per_activation: f64) -> f64 {
        1.0 - self.spike_pj(spikes_per_activation) / baseline_mac_pj
    }

    /// Energy per (activation, output) pair once a weight read of `fetch_pj` is priced, as
    /// `(dense, spiking)`.
    ///
    /// Dense reads one weight per MAC: `baseline_mac_pj + fetch_pj`. Spiking pays `spike_pj` in
    /// additions plus one read per spike ([`Dataflow::FetchPerSpike`]) or one per active input
    /// ([`Dataflow::HoldPerInput`], `active_fraction` of them). The weight is assumed to be the same
    /// width on both sides, which makes the INT8 comparison the fair one — an FP16 baseline reads
    /// twice the bytes per weight and this does not charge it for that.
    #[must_use]
    pub fn with_fetch(
        &self,
        baseline_mac_pj: f64,
        spikes_per_activation: f64,
        active_fraction: f64,
        fetch_pj: f64,
        dataflow: Dataflow,
    ) -> (f64, f64) {
        let reads = match dataflow {
            Dataflow::FetchPerSpike => spikes_per_activation,
            Dataflow::HoldPerInput => active_fraction,
        };
        (baseline_mac_pj + fetch_pj, self.spike_pj(spikes_per_activation) + reads * fetch_pj)
    }

    /// The weight-read energy at which the dense layer and the spiking layer cost the same, in pJ.
    ///
    /// Setting the two sides of [`ArithmeticModel::with_fetch`] equal gives
    /// `fetch = (baseline − spike_pj) / (reads − 1)`. It exists only when the spiking layer reads MORE
    /// weights than the dense one (`reads > 1`) while doing cheaper arithmetic: then it wins below this
    /// fetch energy and loses above it. `None` when it reads no more than the dense layer — it then
    /// wins at every fetch energy if its arithmetic is cheaper, and loses at every one if not.
    #[must_use]
    pub fn break_even_fetch_pj(
        &self,
        baseline_mac_pj: f64,
        spikes_per_activation: f64,
        active_fraction: f64,
        dataflow: Dataflow,
    ) -> Option<f64> {
        let reads = match dataflow {
            Dataflow::FetchPerSpike => spikes_per_activation,
            Dataflow::HoldPerInput => active_fraction,
        };
        let spare = baseline_mac_pj - self.spike_pj(spikes_per_activation);
        if reads > 1.0 && spare > 0.0 { Some(spare / (reads - 1.0)) } else { None }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ArithmeticModel, Coding, Dataflow, Encoded, IntSpikeError, MAX_RATIO, MAX_STEPS, Ops,
        SPIKINGBRAIN_45NM, SPIKINGBRAIN_RATIO_VS_FP16, SPIKINGBRAIN_RATIO_VS_INT8,
        SPIKINGBRAIN_SILENT_FRACTION, SPIKINGBRAIN_SPIKE_PJ, SPIKINGBRAIN_SPIKES_PER_CHANNEL,
        adaptive_threshold, dense, encode, scale, spike_driven,
    };
    use crate::ledger::Evidence;
    use crate::rng::Rng;

    /// Every count a coding can hold comes back from its train exactly — exhaustively.
    ///
    /// Bitwise at every width from 2 to 12 bits, over its whole two's complement range; binary over
    /// `0..=300`; ternary over `−300..=300`. An expansion that dropped a bit, put the sign on the wrong
    /// end, or weighted the top bit positively would fail somewhere in this sweep, and a spot check at
    /// a handful of counts might not.
    #[test]
    fn every_representable_count_survives_expansion_exactly() {
        for bits in 2..=12u32 {
            let c = Coding::Bitwise { bits };
            let (lo, hi) = c.range().unwrap();
            assert_eq!((lo, hi), (-(1i64 << (bits - 1)), (1i64 << (bits - 1)) - 1));
            for n in lo..=hi {
                let train = c.expand(n).unwrap();
                assert_eq!(train.len(), bits as usize);
                assert_eq!(c.reconstruct(&train).unwrap(), n, "{n} at {bits} bits");
            }
        }
        for n in 0..=300 {
            assert_eq!(Coding::Binary.reconstruct(&Coding::Binary.expand(n).unwrap()).unwrap(), n);
        }
        for n in -300..=300 {
            assert_eq!(Coding::Ternary.reconstruct(&Coding::Ternary.expand(n).unwrap()).unwrap(), n);
        }
    }

    /// The bitwise train of a negative count is its two's complement code, top bit weighted
    /// `−2^(b−1)`.
    ///
    /// Pinned on explicit trains because a round-trip test alone would pass an expansion and a
    /// reconstruction that agreed on a WRONG convention — sign-magnitude, say, or most significant
    /// bit first. −1 at 8 bits is all ones; −128 is only the top bit; 5 is `1, 0, 1` then zeros.
    #[test]
    fn a_bitwise_train_is_the_twos_complement_code_least_significant_first() {
        let b8 = Coding::Bitwise { bits: 8 };
        assert_eq!(b8.expand(-1).unwrap(), vec![1; 8]);
        assert_eq!(b8.expand(-128).unwrap(), vec![0, 0, 0, 0, 0, 0, 0, 1]);
        assert_eq!(b8.expand(5).unwrap(), vec![1, 0, 1, 0, 0, 0, 0, 0]);
        assert_eq!(b8.expand(127).unwrap(), vec![1, 1, 1, 1, 1, 1, 1, 0]);
        assert_eq!((b8.step_weight(0).unwrap(), b8.step_weight(6).unwrap(), b8.step_weight(7).unwrap()), (1, 64, -128));
        assert_eq!((b8.spikes(-1).unwrap(), b8.spikes(-128).unwrap(), b8.spikes(5).unwrap()), (8, 1, 2));
    }

    /// Ternary halves the WORST-CASE steps over a range of the same size, and not the steps of one
    /// count.
    ///
    /// The paper's example reads "a count of 256 requires 256 consecutive time steps in binary
    /// coding, 128 in ternary coding, but only 8 steps in 8-bit bitwise encoding". Two of its three
    /// legs fail for that count: a ternary train for 256 is 256 spikes of `+1`, and 256 does not fit
    /// an 8-bit code at all. What the claim means, and what is true, is that 256 levels unsigned
    /// (`0..=255`) need up to 255 binary steps while 256 levels signed (`−128..=127`) need up to 128
    /// ternary steps, and an 8-bit bitwise code needs 8 for any of them.
    #[test]
    fn ternary_halves_the_worst_case_over_a_range_and_not_the_steps_of_one_count() {
        assert_eq!(Coding::Binary.steps(256).unwrap(), 256);
        assert_eq!(Coding::Ternary.steps(256).unwrap(), 256, "one count: no saving at all");
        // And the example's bitwise leg does not hold either: 256 does not fit an 8-bit code at all —
        // signed tops out at 127, unsigned at 255 — nor a 9-bit signed one (−256..=255). It needs ten.
        assert_eq!(Coding::Bitwise { bits: 8 }.steps(256), Err(IntSpikeError::OutOfRange { count: 256, bits: 8 }));
        assert_eq!(Coding::Bitwise { bits: 9 }.steps(256), Err(IntSpikeError::OutOfRange { count: 256, bits: 9 }));
        assert_eq!(Coding::Bitwise { bits: 10 }.steps(256).unwrap(), 10);
        let worst_binary = (0..=255).map(|n| Coding::Binary.steps(n).unwrap()).max().unwrap();
        let worst_ternary = (-128..=127).map(|n| Coding::Ternary.steps(n).unwrap()).max().unwrap();
        let bitwise = Coding::Bitwise { bits: 8 };
        let worst_bitwise = (-128..=127).map(|n| bitwise.steps(n).unwrap()).max().unwrap();
        assert_eq!((worst_binary, worst_ternary, worst_bitwise), (255, 128, 8));
    }

    /// Each coding refuses what it cannot represent, and says so in the right order.
    ///
    /// Binary has no negative spikes; a bitwise width holds `−2^(b−1)..=2^(b−1) − 1` and not one past
    /// either end; a width must be `2..=32`; a train must stay inside its alphabet; and a binary or
    /// ternary train is capped at `MAX_STEPS` because its length IS its count. Every message is
    /// rendered, because in this crate a message whose two numbers are swapped has survived every
    /// test that only matched the variant.
    #[test]
    fn each_coding_refuses_what_it_cannot_represent() {
        assert_eq!(Coding::Binary.expand(-1), Err(IntSpikeError::Unsigned { count: -1 }));
        let b8 = Coding::Bitwise { bits: 8 };
        assert_eq!(b8.expand(128), Err(IntSpikeError::OutOfRange { count: 128, bits: 8 }));
        assert_eq!(b8.expand(-129), Err(IntSpikeError::OutOfRange { count: -129, bits: 8 }));
        assert!(b8.expand(127).is_ok() && b8.expand(-128).is_ok());
        assert_eq!(Coding::Bitwise { bits: 1 }.expand(0), Err(IntSpikeError::BadBits { bits: 1 }));
        assert_eq!(Coding::Bitwise { bits: 33 }.range(), Err(IntSpikeError::BadBits { bits: 33 }));
        assert!(Coding::Bitwise { bits: 2 }.range().is_ok() && Coding::Bitwise { bits: 32 }.range().is_ok());
        let cap = MAX_STEPS as i64;
        assert!(Coding::Ternary.steps(-cap).is_ok());
        assert_eq!(Coding::Ternary.steps(-cap - 1), Err(IntSpikeError::TooLong { steps: (cap + 1) as u64 }));
        assert_eq!(Coding::Binary.steps(cap + 1), Err(IntSpikeError::TooLong { steps: (cap + 1) as u64 }));
        assert_eq!(Coding::Binary.reconstruct(&[1, -1]), Err(IntSpikeError::BadSpike { index: 1, value: -1 }));
        assert_eq!(Coding::Ternary.reconstruct(&[0, 2]), Err(IntSpikeError::BadSpike { index: 1, value: 2 }));
        assert_eq!(b8.reconstruct(&[1; 7]), Err(IntSpikeError::Dimension { what: "bitwise train", got: 7, want: 8 }));
        assert_eq!(IntSpikeError::OutOfRange { count: 128, bits: 8 }.to_string(), "128 does not fit a 8-bit two's complement code");
        assert_eq!(
            IntSpikeError::Dimension { what: "weights", got: 5, want: 6 }.to_string(),
            "weights has length 5, expected 6"
        );
        assert_eq!(IntSpikeError::Unsigned { count: -3 }.to_string(), "binary coding cannot represent the negative count -3");
        assert_eq!(IntSpikeError::BadBits { bits: 1 }.to_string(), "1 bits: a bitwise coding needs 2 to 32");
    }

    /// The threshold is `mean(|x|) / k`, and rounding keeps every element within half a threshold.
    ///
    /// The values are chosen so the mean is exact in binary: `|x|` sums to 12 over 4 elements, mean 3,
    /// and `k = 2` gives `V_th = 1.5` exactly. The half-threshold bound is the rounding bound, and it is
    /// checked on random activations as well because a scale applied at the wrong end would pass the
    /// exact case and fail it.
    #[test]
    fn the_threshold_is_the_mean_magnitude_over_k_and_rounding_is_bounded_by_half_of_it() {
        let x = [3.0, -4.5, 1.5, 3.0];
        assert_eq!(adaptive_threshold(&x, 2.0).unwrap(), 1.5);
        let e = encode(&x, 2.0).unwrap();
        assert_eq!(e.counts, vec![2, -3, 1, 2]);
        assert_eq!(e.decode(), vec![3.0, -4.5, 1.5, 3.0], "every element is a whole number of thresholds");
        let mut rng = Rng::new(7);
        let noisy: Vec<f64> = (0..1000).map(|_| 6.0 * rng.next_f64() - 3.0).collect();
        for k in [0.5, 1.0, 4.0, 16.0] {
            let e = encode(&noisy, k).unwrap();
            assert!(e.max_error(&noisy).unwrap() <= e.threshold / 2.0, "k = {k}");
        }
    }

    /// Exact ties round to EVEN — a choice this module makes because the paper does not.
    ///
    /// With `x = [0.5, 1.5, 2.5, −2.5]` the mean magnitude is 1.75; at `k = 1.75` the threshold is
    /// exactly 1, so every ratio is an exact tie. Round-half-to-even gives `0, 2, 2, −2`; rounding half
    /// away from zero would give `1, 2, 3, −3`, three of four counts different.
    #[test]
    fn exact_ties_round_to_even() {
        let e = encode(&[0.5, 1.5, 2.5, -2.5], 1.75).unwrap();
        assert_eq!(e.threshold, 1.0);
        assert_eq!(e.counts, vec![0, 2, 2, -2]);
    }

    /// An all-zero input has a zero threshold and zero counts, rather than `0 / 0`.
    ///
    /// And the refusals: nothing to average, a non-finite activation named at its own position, a
    /// rate that is zero, negative, infinite or `NaN`, and a ratio past `2^53`, where rounding to an
    /// integer is no longer exact.
    #[test]
    fn an_all_zero_input_encodes_to_zero_and_the_bad_inputs_are_refused() {
        let e = encode(&[0.0, -0.0, 0.0], 3.0).unwrap();
        assert_eq!((e.threshold, e.counts.clone()), (0.0, vec![0, 0, 0]));
        assert_eq!(e.decode(), vec![0.0, 0.0, 0.0]);
        assert_eq!(encode(&[], 1.0), Err(IntSpikeError::Empty));
        let bad = encode(&[1.0, 2.0, f64::INFINITY], 1.0).unwrap_err();
        assert_eq!(bad, IntSpikeError::NonFinite { index: 2, value: f64::INFINITY });
        for k in [0.0, -1.0, f64::INFINITY] {
            assert_eq!(adaptive_threshold(&[1.0], k), Err(IntSpikeError::BadRate { k }));
        }
        assert!(matches!(adaptive_threshold(&[1.0], f64::NAN), Err(IntSpikeError::BadRate { .. })));
        // One huge activation among many tiny ones, at a large k: the ratio passes 2^53.
        let mut x = vec![1e-300; 1000];
        x[3] = 1.0;
        let err = encode(&x, 1e14).unwrap_err();
        assert!(matches!(err, IntSpikeError::TooLarge { index: 3, ratio } if ratio > MAX_RATIO));
        assert_eq!(e.max_error(&[0.0]), Err(IntSpikeError::Dimension { what: "activations", got: 1, want: 3 }));
    }

    /// The accumulate-only product equals the dense integer product exactly, in every coding.
    ///
    /// This is `SpikingBrain`'s Eq. 22 made checkable: `Σ_t W s_t` computed with additions and shifts
    /// is `W · s_INT`, not approximately but exactly, because every quantity is an integer. Random INT8
    /// weights, random counts, three codings. The operation counts are the other half of the claim:
    /// one addition per output row per non-zero spike, and a shift for each bitwise spike above bit 0.
    #[test]
    fn the_accumulate_only_product_is_the_dense_integer_product_exactly() {
        let mut rng = Rng::new(20_250_905);
        let (rows, cols) = (6, 10);
        let w: Vec<i8> = (0..rows * cols).map(|_| (rng.next_u32() % 256) as u8 as i8).collect();
        for coding in [Coding::Binary, Coding::Ternary, Coding::Bitwise { bits: 8 }] {
            let counts: Vec<i64> = (0..cols)
                .map(|_| match coding {
                    Coding::Binary => i64::from(rng.next_u32() % 40),
                    _ => i64::from(rng.next_u32() % 200) - 100,
                })
                .collect();
            let trains: Vec<Vec<i8>> = counts.iter().map(|&c| coding.expand(c).unwrap()).collect();
            let (y, ops) = spike_driven(&w, rows, cols, &trains, coding).unwrap();
            assert_eq!(y, dense(&w, rows, cols, &counts).unwrap(), "{coding:?}");
            let spikes: u64 = counts.iter().map(|&c| coding.spikes(c).unwrap() as u64).sum();
            assert_eq!(ops.additions, rows as u64 * spikes, "{coding:?}");
            assert_eq!(ops.reads_per_spike, spikes);
            assert_eq!(ops.reads_held, counts.iter().filter(|&&c| c != 0).count() as u64);
            let shifted: u64 = trains.iter().flat_map(|t| t.iter().enumerate().skip(1)).filter(|(_, s)| **s != 0).count() as u64;
            let want_shifts = if matches!(coding, Coding::Bitwise { .. }) { rows as u64 * shifted } else { 0 };
            assert_eq!(ops.shifts, want_shifts, "{coding:?}");
        }
    }

    /// Encode, expand, accumulate, scale: the whole pipeline equals `V_th · (W · s_INT)`.
    ///
    /// The threshold is applied ONCE, after the integer accumulation — which is what makes the
    /// accumulation exact — and the scaled output equals the scaled dense product bit for bit because
    /// both sides are the same integer times the same threshold.
    #[test]
    fn the_pipeline_is_the_threshold_times_the_integer_product() {
        let mut rng = Rng::new(3);
        let (rows, cols) = (4, 16);
        let x: Vec<f64> = (0..cols).map(|_| 2.0 * rng.next_f64() - 1.0).collect();
        let w: Vec<i8> = (0..rows * cols).map(|_| (rng.next_u32() % 256) as u8 as i8).collect();
        let e = encode(&x, 1.0).unwrap();
        let coding = Coding::Bitwise { bits: 8 };
        let (acc, _) = spike_driven(&w, rows, cols, &e.trains(coding).unwrap(), coding).unwrap();
        assert_eq!(scale(&acc, e.threshold), scale(&dense(&w, rows, cols, &e.counts).unwrap(), e.threshold));
        assert_eq!(e.steps(coding).unwrap(), 8);
        assert_eq!(e.steps(Coding::Ternary).unwrap() as i64, e.counts.iter().map(|c| c.abs()).max().unwrap());
        assert_eq!(e.spikes(Coding::Ternary).unwrap() as i64, e.counts.iter().map(|c| c.abs()).sum::<i64>());
    }

    /// The products refuse arrays of the wrong shape and a spike outside the alphabet.
    #[test]
    fn the_products_refuse_the_wrong_shapes() {
        let w = vec![1i8; 6];
        assert_eq!(dense(&w, 2, 4, &[1, 2, 3, 4]), Err(IntSpikeError::Dimension { what: "weights", got: 6, want: 8 }));
        assert_eq!(dense(&w, 2, 3, &[1, 2]), Err(IntSpikeError::Dimension { what: "inputs", got: 2, want: 3 }));
        let trains = vec![vec![1i8], vec![2i8], vec![0i8]];
        assert_eq!(
            spike_driven(&w, 2, 3, &trains, Coding::Binary),
            Err(IntSpikeError::BadSpike { index: 0, value: 2 })
        );
        assert_eq!(
            spike_driven(&w, 2, 3, &[vec![1], vec![1]], Coding::Binary),
            Err(IntSpikeError::Dimension { what: "inputs", got: 2, want: 3 })
        );
        assert_eq!(dense(&[i8::MAX], 1, 1, &[i64::MAX]), Err(IntSpikeError::Overflow));
        let (_, ops) = spike_driven(&w, 2, 3, &[vec![], vec![], vec![]], Coding::Binary).unwrap();
        assert_eq!(ops, Ops::default(), "no spikes, no work");
    }

    /// The paper's energy model reproduces its own percentages — and its two ratios disagree.
    ///
    /// With the "about 0.034 pJ" per MAC-equivalent the text states, the reductions are 97.7% against
    /// FP16 and 85.2% against INT8, as printed, and the INT8 ratio is the printed 6.76×. The FP16 ratio
    /// is not: 1.5 / 0.034 is 44.1×, and the printed 43.48× needs 0.0345 pJ, which is 1.15 spikes per
    /// channel rather than the stated 1.13. Recorded as measured from the paper's own numbers.
    #[test]
    fn the_papers_energy_model_reproduces_its_percentages_and_its_ratios_disagree() {
        let m = SPIKINGBRAIN_45NM;
        assert_eq!(m.evidence, Evidence::Derived);
        assert!(m.source.contains("Arithmetic only"));
        // Pinned to the version whose section and equation it names: v1 numbers the model Eq. 27.
        assert!(m.source.contains("arXiv:2509.05276v4") && m.source.contains("Eq. 24"), "{}", m.source);
        let pct = |v: f64| (v * 1000.0).round() / 10.0;
        let spikes = SPIKINGBRAIN_SPIKE_PJ / m.int8_add_pj;
        assert_eq!(pct(m.reduction(m.fp16_mac_pj, spikes)), 97.7);
        assert_eq!(pct(m.reduction(m.int8_mac_pj, spikes)), 85.2);
        let two = |v: f64| (v * 100.0).round() / 100.0;
        assert_eq!(two(m.int8_mac_pj / SPIKINGBRAIN_SPIKE_PJ), SPIKINGBRAIN_RATIO_VS_INT8);
        assert_eq!(two(m.fp16_mac_pj / SPIKINGBRAIN_SPIKE_PJ), 44.12, "not the printed 43.48");
        let implied = m.fp16_mac_pj / SPIKINGBRAIN_RATIO_VS_FP16;
        assert_eq!((implied * 10_000.0).round() / 10_000.0, 0.0345);
        assert_eq!((implied / m.int8_add_pj * 100.0).round() / 100.0, 1.15, "spikes per channel it implies");
        // And the stated statistic, 1.13 spikes at 0.03 pJ, is 0.0339 — which the text rounds to 0.034.
        assert_eq!((m.spike_pj(SPIKINGBRAIN_SPIKES_PER_CHANNEL) * 10_000.0).round() / 10_000.0, 0.0339);
    }

    /// Price the weight reads and the verdict turns on the dataflow, which the paper implies through
    /// "proportionally" and never names.
    ///
    /// At 1.13 spikes per channel, fetching a column per SPIKE reads 13% more weights than a dense
    /// pass: against INT8 the dense layer becomes cheaper once a read costs more than
    /// `(0.23 − 1.13 · 0.03) / 0.13 = 1.508` pJ, and at exactly that price the two cost the same.
    /// Holding a column while one input's spikes apply reads only the 81.6% of inputs that are
    /// active, so the spiking layer wins at every fetch energy. Same arithmetic, opposite verdicts.
    #[test]
    fn once_a_weight_read_is_priced_the_verdict_turns_on_the_dataflow() {
        let m = SPIKINGBRAIN_45NM;
        let (s, active) = (SPIKINGBRAIN_SPIKES_PER_CHANNEL, 1.0 - SPIKINGBRAIN_SILENT_FRACTION);
        let f = m.break_even_fetch_pj(m.int8_mac_pj, s, active, Dataflow::FetchPerSpike).unwrap();
        assert!((f - 1.508_461_538_461_538_5).abs() < 1e-12, "measured {f}");
        let (d, sp) = m.with_fetch(m.int8_mac_pj, s, active, f, Dataflow::FetchPerSpike);
        assert!((d - sp).abs() < 1e-12, "equal at the break-even: {d} {sp}");
        let (d, sp) = m.with_fetch(m.int8_mac_pj, s, active, 2.0 * f, Dataflow::FetchPerSpike);
        assert!(d < sp, "above it the dense layer is cheaper");
        let (d, sp) = m.with_fetch(m.int8_mac_pj, s, active, 0.5 * f, Dataflow::FetchPerSpike);
        assert!(sp < d, "below it the spiking layer is cheaper");
        assert_eq!(m.break_even_fetch_pj(m.int8_mac_pj, s, active, Dataflow::HoldPerInput), None);
        for fetch in [0.0, 1.0, 10.0, 1000.0] {
            let (d, sp) = m.with_fetch(m.int8_mac_pj, s, active, fetch, Dataflow::HoldPerInput);
            assert!(sp < d, "held columns win at every fetch energy: {fetch}");
        }
        // And with no spare arithmetic there is nothing to trade against the extra reads.
        let flat = ArithmeticModel { int8_add_pj: 1.0, ..m };
        assert_eq!(flat.break_even_fetch_pj(0.23, 1.13, 0.816, Dataflow::FetchPerSpike), None);
    }

    /// The paper's memory claim, that skipping silent channels cuts memory access "proportionally",
    /// holds for held columns and fails for columns fetched per spike.
    ///
    /// A ternary layer of eight channels with counts `0, 1, −2, 1, 0, 3, 1, 1`: a quarter of them
    /// silent and 9/8 = 1.125 spikes a channel, the eighths fraction nearest the paper's 1.13. A dense
    /// pass reads all eight columns. Held, the layer reads six, `8 × (1 − 1/4)`, so the saving is
    /// exactly the silent quarter. Fetched per spike it reads nine, one MORE than dense, though the
    /// same two channels are skipped. Skipping silent channels does not pick the dataflow;
    /// "proportionally" does. The same holds at the paper's own statistics once the arithmetic is
    /// zeroed and a read costs 1: held, `1 − 0.184` of a dense read; per spike, 1.13.
    #[test]
    fn the_papers_proportional_memory_saving_is_the_held_dataflow() {
        let counts = [0i64, 1, -2, 1, 0, 3, 1, 1];
        let trains: Vec<Vec<i8>> = counts.iter().map(|&c| Coding::Ternary.expand(c).unwrap()).collect();
        let w = vec![1i8; 3 * counts.len()];
        let (_, ops) = spike_driven(&w, 3, counts.len(), &trains, Coding::Ternary).unwrap();
        let dense_reads = counts.len() as u64;
        let silent = counts.iter().filter(|&&c| c == 0).count() as u64;
        assert_eq!((dense_reads, silent), (8, 2));
        assert_eq!(ops.reads_held, dense_reads - silent, "held: the saving is the silent quarter");
        assert_eq!(ops.reads_per_spike, 9, "per spike: one read more than the dense pass");
        let reads_only = ArithmeticModel { int8_add_pj: 0.0, ..SPIKINGBRAIN_45NM };
        let (s, active) = (SPIKINGBRAIN_SPIKES_PER_CHANNEL, 1.0 - SPIKINGBRAIN_SILENT_FRACTION);
        assert_eq!(reads_only.with_fetch(0.0, s, active, 1.0, Dataflow::HoldPerInput), (1.0, active));
        assert_eq!(reads_only.with_fetch(0.0, s, active, 1.0, Dataflow::FetchPerSpike), (1.0, 1.13));
    }

    /// The paper's Eq. 23 is its BIDIRECTIONAL bitwise form, and this module implements its two's
    /// complement form: the same train reads differently under the two.
    ///
    /// Eq. 23 weights step `t`, counted from 1, by `2^(t−1)`, every weight positive, and its spikes
    /// are `±1` ("uses ±1 to represent each bit"). The 4-bit two's complement code of −5, least
    /// significant bit first, is `1, 1, 0, 1`: Figure 4(c)'s `1 0 1 1`, which the figure draws most
    /// significant bit first, reversed. Read with Eq. 23's weights that train is `1 + 2 + 8 = 11`; read
    /// with [`Coding::step_weight`] it is `1 + 2 − 8 = −5`. Eq. 23 holds −5 as the figure's
    /// bidirectional train `−1 0 −1` instead.
    #[test]
    fn the_papers_eq_23_is_the_bidirectional_form_not_the_twos_complement_code_implemented_here() {
        let b4 = Coding::Bitwise { bits: 4 };
        let train = b4.expand(-5).unwrap();
        assert_eq!(train, vec![1, 1, 0, 1]);
        let figure_4c: Vec<i8> = vec![1, 0, 1, 1];
        assert_eq!(train.iter().rev().copied().collect::<Vec<i8>>(), figure_4c, "the figure's order, reversed");
        let eq23 = |s: &[i8]| s.iter().enumerate().map(|(t, &b)| i64::from(b) << t).sum::<i64>();
        assert_eq!(eq23(&train), 11, "Eq. 23's all-positive weights misread the two's complement code");
        assert_eq!(b4.reconstruct(&train).unwrap(), -5);
        assert_eq!(eq23(&[-1, 0, -1]), -5, "Eq. 23 holds -5 as signed bits");
    }

    /// Encoded values are exactly what was stored: the struct round-trips through its own fields.
    #[test]
    fn an_encoding_carries_its_threshold_and_counts() {
        let e = Encoded { threshold: 0.5, counts: vec![-2, 0, 3] };
        assert_eq!(e.decode(), vec![-1.0, 0.0, 1.5]);
        // At 4 bits, −2 is `1110` — three ones; 0 has none; 3 is `0011` — two.
        assert_eq!(e.spikes(Coding::Bitwise { bits: 4 }).unwrap(), 5);
        assert_eq!(e.trains(Coding::Binary), Err(IntSpikeError::Unsigned { count: -2 }));
    }

    /// This crate's two caps, and the paper's transcribed statistics, pinned by VALUE.
    ///
    /// Every other test reaches `MAX_STEPS` and `MAX_RATIO` as `MAX_STEPS + 1` or `> MAX_RATIO`, which
    /// moves with the constant, so halving either changed no test; the `const` assertions make a
    /// changed value fail to compile. The ratio cap is pinned from inside as well: a ratio between
    /// `2^52` and `2^53` is still an exact integer in an `f64` and must be accepted. The silent fraction
    /// only ever entered as `1 − f`, where 0.184 and 0.0184 both leave held columns winning, and the
    /// binary and ternary ranges were never read at all — `admit` refuses a negative binary count on
    /// its own sign test, whatever `range` says.
    #[test]
    fn the_caps_and_the_transcribed_statistics_are_pinned_by_value() {
        const { assert!(MAX_STEPS == 1 << 20) };
        const { assert!(MAX_RATIO == 9_007_199_254_740_992.0) };
        let e = encode(&[1.0, 0.0], 3e15).unwrap();
        assert!(e.counts[0] > 1 << 52 && e.counts[0] < 1 << 53, "{}", e.counts[0]);
        assert_eq!((SPIKINGBRAIN_SPIKES_PER_CHANNEL, SPIKINGBRAIN_SILENT_FRACTION), (1.13, 0.184));
        let cap = MAX_STEPS as i64;
        assert_eq!(Coding::Binary.range().unwrap(), (0, cap));
        assert_eq!(Coding::Ternary.range().unwrap(), (-cap, cap));
    }

    /// The reconstruction error, the scale and two messages, each read against a stated value.
    ///
    /// Each was previously checked only against something computed the same way: the error against
    /// a bound the smallest deviation also satisfies, and `scale` against `scale`, so dropping the
    /// threshold changed both sides equally. And two refusals were matched by variant and never
    /// rendered — the class this crate has found in almost every module, repeated here in new code.
    #[test]
    fn the_error_the_scale_and_the_messages_are_read_by_value() {
        let e = Encoded { threshold: 1.0, counts: vec![0, 1] };
        assert_eq!(e.max_error(&[0.25, 1.0]).unwrap(), 0.25, "the largest deviation, not the smallest");
        assert_eq!(scale(&[2, -3, 0], 0.5), vec![1.0, -1.5, 0.0]);
        assert_eq!(
            IntSpikeError::NonFinite { index: 2, value: f64::INFINITY }.to_string(),
            "activation 2 is inf, which is not finite"
        );
        assert_eq!(
            IntSpikeError::BadSpike { index: 1, value: 2 }.to_string(),
            "spike 1 is 2, outside this coding's alphabet"
        );
    }

}
