//! What actually fits on the chips: the structural limits of neuromorphic parts, checkable before
//! anyone buys a board.
//!
//! # The lesson
//!
//! A simulator will run any network you give it. A chip will not. Every neuromorphic part ever
//! fabricated is a fixed arrangement of memory: a core holds a fixed number of neurons, a fixed
//! amount of synapse storage, and — the limit that catches people — a fixed number of inputs per
//! neuron. `TrueNorth`'s crossbar is 256 rows by 256 columns, so a `TrueNorth` neuron has at most
//! 256 presynaptic sources. Not 256 on average. Not 256 unless you pay for more. Two hundred and
//! fifty-six, as a wall, on a part with a million neurons.
//!
//! That distinction — a **wall** versus a **budget** — is the thing to carry away from this module.
//!
//! * A **wall** is structural. `TrueNorth`'s 256-input crossbar column, `DYNAP-SE`'s 64
//!   content-addressable-memory entries per neuron: a network that needs more inputs to one neuron
//!   does not map, and buying a second chip does not help, because fan-in is a property of one
//!   neuron and not of the system. The fix is to split the neuron into a tree of partial summers,
//!   which costs neurons and adds a tick of latency per level.
//! * A **budget** is a rate. `SpiNNaker` has no fan-in cap at all, because a `SpiNNaker` synapse is
//!   a row in `SDRAM` that a software `ARM` core fetches and processes. Ask for more fan-in and it
//!   does not refuse — it misses its 1 ms real-time deadline and silently changes the answer. That
//!   is a worse failure than refusal, and it is why [`SPINNAKER`]'s fan-in field here is empty
//!   rather than large.
//!
//! What it buys: knowing this before you commit. What it costs: the tables below are thin, because
//! most of these numbers are not published.
//!
//! # ⛔ How to read the grades, and why so many fields are empty
//!
//! Every figure in this module is a [`Spec`], which carries the value, **a provenance string**, and
//! an [`Evidence`] grade. A `Spec` whose value is `None` is not a gap in the implementation. It is
//! a finding: *this review did not locate that figure.* A part with three stated fields and six
//! empty ones is a correct record. A part with nine plausible-looking numbers is a liability.
//!
//! [`Evidence`] was designed in [`crate::ledger`] for energy figures, so its grades are mapped onto
//! structural specifications explicitly here rather than by implication:
//!
//! | Grade | What it means for a structural spec in this module |
//! |---|---|
//! | [`Evidence::Unstated`] | No figure located. The `Spec`'s value is `None` and its string says what was looked for. |
//! | [`Evidence::Projected`] | A design target, a roadmap figure, a part this review does not know to be fabricated — **or a figure this review inferred rather than located**: an analogy from a sibling part, a year read off undated material, a schema choice this review made. |
//! | [`Evidence::Derived`] | **This review computed it** from other stated figures. The arithmetic is shown in the string, and `the_derived_grade_is_an_exact_census` requires every `Derived` string to show an operator and an operand. |
//! | [`Evidence::Simulated`] | From a simulation of the design rather than from the design's documentation. |
//! | [`Evidence::Measured`] | Stated for a fabricated part in a primary document — a peer-reviewed paper, a vendor document for shipping silicon, or the part's own reference software's documentation where the figure is a software convention **and the string says which**. Where it is not a paper, **the provenance string says "vendor" or names the software**. |
//! | [`Evidence::Metered`] | Not used in this module. `Metered` is about an instrument reading, and nothing here is an instrument reading. |
//!
//! ⛔ **`Derived` and `Projected` are not interchangeable, and the order matters.** [`Evidence`]
//! sorts `Projected < Derived`, so grading an inference `Derived` makes "this review guessed a year
//! off undated vendor material" sort as *stronger* evidence than a maker's published roadmap
//! figure, and [`Part::weakest_evidence`] inherits that inversion. `Derived` means arithmetic over
//! figures stated elsewhere in the same record or in the cited document. Nothing else.
//!
//! ## The mistake this crate already made once
//!
//! [`crate::ledger::LOIHI_2018`] shipped a pre-silicon `SPICE` figure graded `Measured` in two
//! releases of this crate, because its author took the grade from the citing literature instead of
//! from the cited table. The correction is left visible in that module. This module was written
//! after it, and the rule that came out of it is the reason every `Spec` string names a document
//! rather than a claim: **a grade is a statement about a piece of paper, not about a number.**
//!
//! For the same reason this module adds **no** [`crate::ledger::Prices`] entries. `ODIN`'s
//! 12.7 pJ/SOP (in the title of Frenkel et al., 2019) is a genuine silicon measurement and would be
//! a valuable addition — and transcribing it from a literature audit rather than from the paper is
//! exactly the habit that produced the defect above. It is named here as a pointer and priced
//! nowhere.
//!
//! # What this module computes
//!
//! * [`fits`] — can this network map onto this part? Never a bare boolean: it returns a [`Fit`]
//!   naming every constraint that **binds**, by how much, in natural units, plus the **headroom** on
//!   every constraint that held and the **unchecked** list of limits the part does not state.
//!   A part that states nothing gets `verdict: None`, which is not a pass.
//! * [`CoreCount`] — how many cores a network needs. This is a **bin-packing lower bound**, not an
//!   exact answer, except in the one case where it is provably tight and says so.
//! * [`Quantiser`] — weights to a part's bit-width, with the round-trip error reported, in both
//!   round-to-nearest and stochastic rounding. Round-to-nearest errs by at most half a least
//!   significant bit and is **biased**; stochastic rounding is **unbiased** and errs by more.
//!   Three closed forms are checked here, and the third is the one to read carefully. For a weight
//!   sitting a fraction `f` of a step above a level: round-to-nearest's signed error is
//!   `-f` steps (or `1-f`) *every time*, so its bias and its RMS error are both exactly
//!   `min(f, 1-f)`; stochastic rounding's expected error is exactly zero and its RMS error is
//!   exactly `sqrt(f*(1-f))`. Since `sqrt(f*(1-f)) >= min(f, 1-f)` for every `f`, that is the
//!   sense — **in expectation** — in which stochastic rounding errs by more. It is not a guarantee
//!   about any one vector: on a short vector a lucky draw inverts it, and
//!   [`Quantised::rms_error`] says so rather than promising otherwise.
//!
//! # Units
//!
//! Counts are counts. Delays are in the part's own **ticks**, never seconds: a tick is a
//! configurable algorithmic timestep on `Loihi` and `SpiNNaker`, a fixed 1 ms frame on `TrueNorth`,
//! and on an analogue part such as `BrainScaleS`-2 there is no tick at all — which is why that
//! part's delay field is empty rather than converted. Process nodes are **strings**, not numbers,
//! because "Intel 4" is a product name and not a dimension and subtracting it from "12 nm" would be
//! arithmetic on marketing.
//!
//! # Name collisions with [`crate::ledger`]
//!
//! [`crate::ledger`] exports `TRUENORTH_2014` and `LOIHI_2018` as [`crate::ledger::Prices`]. The
//! parts here are named [`TRUENORTH`] and [`LOIHI`] without a year suffix so that a caller can
//! import both modules' constants into one scope. They describe the same silicon and answer
//! different questions: the ledger's is about joules and refuses, this one is about shape.

use crate::ledger::Evidence;
use crate::net::Net;
use crate::rng::Rng;
use core::fmt;

/// One published figure about a part, with where it came from and how good that source is.
///
/// The `Option` is the point of the type. A `u32` field defaulting to zero would let "nobody has
/// published this" and "the answer is none" become the same value in a sum; a `Spec` forces the
/// caller to confront the difference, and [`fits`] refuses to grade a constraint it cannot read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Spec<T> {
    /// The figure, or `None` when this review did not locate one.
    pub value: Option<T>,
    /// WHERE the figure came from: a paper with author, venue and year, a named vendor document, or
    /// — when `value` is `None` — a statement of what was looked for and not found.
    ///
    /// Never empty. `every_stated_spec_names_a_document` asserts it.
    pub source: &'static str,
    /// How good [`Spec::source`] is, under the mapping tabulated in the module doc.
    ///
    /// Invariant, asserted by `an_empty_spec_is_graded_unstated`: `value.is_none()` implies
    /// `evidence == Evidence::Unstated`, and `value.is_some()` implies it is stronger than that.
    pub evidence: Evidence,
}

impl<T> Spec<T> {
    /// A figure this review located, with its source and grade.
    #[must_use]
    pub const fn known(value: T, source: &'static str, evidence: Evidence) -> Self {
        Self { value: Some(value), source, evidence }
    }

    /// A figure this review did **not** locate. `source` says what was looked for.
    ///
    /// Graded [`Evidence::Unstated`] by construction, so the grade cannot drift away from the
    /// absence it describes.
    #[must_use]
    pub const fn unlocated(source: &'static str) -> Self {
        Self { value: None, source, evidence: Evidence::Unstated }
    }

    /// Whether a figure was located.
    #[must_use]
    pub const fn is_known(&self) -> bool {
        self.value.is_some()
    }
}

/// A part's synaptic delay range, in the part's own **ticks**.
///
/// Inclusive at both ends. `min_ticks` is frequently 1 rather than 0, and that matters: a network
/// built with same-tick delivery (`delay == 0`, which [`crate::net`] permits) cannot map onto a
/// part whose pipeline forces at least one tick of latency, and [`fits`] reports that as
/// [`Bind::DelayTooShort`] rather than rounding it away.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DelayRange {
    /// Shortest deliverable delay, ticks. Often 1 — see the type doc.
    pub min_ticks: u32,
    /// Longest deliverable delay, ticks. The width of a hardware field, so almost always
    /// `2^k - 1`.
    pub max_ticks: u32,
}

impl DelayRange {
    /// Whether `ticks` is deliverable, inclusive at both ends.
    #[must_use]
    pub fn contains(&self, ticks: u32) -> bool {
        (self.min_ticks..=self.max_ticks).contains(&ticks)
    }
}

/// The structural limits of one neuromorphic part.
///
/// Nine of the fields are [`Spec`]s carrying a provenance and a grade; `name`, `vendor` and
/// `citation` are metadata, and `citation` is the primary source the record as a whole rests on.
/// Read `citation` before quoting any field: several of these parts have no peer-reviewed
/// architecture paper at all and the record says so there.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Part {
    /// The part as its maker names it, e.g. `"Loihi 2"`. Unique across [`PARTS`].
    pub name: &'static str,
    /// Who makes it. Kept separate from `name` so a table can be grouped by maker without parsing.
    pub vendor: &'static str,
    /// The primary source for this record: author, venue and year, or a named vendor document —
    /// **and, where there is no peer-reviewed architecture paper, a statement to that effect.**
    pub citation: &'static str,
    /// Year of `citation`, which is not always the year of first silicon; the string says which
    /// when they differ.
    pub year: Spec<u32>,
    /// Neurons one core can hold, at the part's own definition of a core.
    ///
    /// A **ceiling**, not a typical configuration: on most parts a core filled to its neuron limit
    /// has proportionally less synapse storage per neuron.
    pub neurons_per_core: Spec<u32>,
    /// Neuromorphic cores on one chip. Excludes embedded host processors that hold no neurons.
    pub cores_per_chip: Spec<u32>,
    /// Synapses one core can store.
    ///
    /// On parts whose weight precision is configurable this is precision-dependent, and the string
    /// says at which precision the figure holds.
    pub synapses_per_core: Spec<u64>,
    /// Presynaptic sources one neuron can have — **the wall**, where there is one.
    ///
    /// Empty on every software-scheduled part in this table, because on those the limit is a
    /// real-time budget rather than a structure. Empty is the correct record there; a large number
    /// would be a wrong one.
    pub max_fan_in: Spec<u32>,
    /// Weight resolution in bits, including sign where the part signs its weights.
    ///
    /// Read the provenance string before using it: on `TrueNorth` the crossbar stores **one bit**
    /// and the magnitude comes from a per-neuron indirection, and on `DYNAP-SE` the two bits select
    /// among analogue bias currents shared by a whole core. Neither is a 1-bit or 2-bit weight in
    /// the sense a quantiser means, and the strings say so.
    pub weight_bits: Spec<u32>,
    /// Deliverable synaptic delays, in the part's ticks. See [`DelayRange`].
    pub delay_ticks: Spec<DelayRange>,
    /// Whether weights can be updated by the part itself, without a host training loop.
    ///
    /// `Some(false)` is a positive claim from a source, not an absence of information; absence is
    /// `None`. And `Some(true)` covers a wide range — read the string, because `Akida`'s on-chip
    /// learning touches only the final layer and `SpiNNaker`'s spends the same processor cycles the
    /// neurons are using.
    pub on_chip_learning: Spec<bool>,
    /// Process node **as its maker names it**, e.g. `"Intel 4"`, `"GlobalFoundries 22 nm FD-SOI"`.
    ///
    /// A string rather than a number on purpose: modern node names are product names, not
    /// dimensions, and two parts on differently-named nodes are not orderable by that name alone.
    pub process: Spec<&'static str>,
    /// Neurons on one chip where the maker publishes a **chip total**, or where this review reads
    /// an exact figure out of a round one the maker published.
    ///
    /// Separate from the product of `neurons_per_core` and `cores_per_chip` because for several
    /// parts the chip total is the published quantity and the per-core split is not. Use
    /// [`Part::neurons_per_chip`], which prefers a published total and falls back to the product.
    ///
    /// ⛔ Two of these entries are **not** a maker's number transcribed. [`LOIHI_2`]'s 1,048,576 is
    /// this review reading Intel's "up to 1 million" as the power of two 128 cores of 8,192 gives,
    /// and it is graded [`Evidence::Derived`] with the rounding shown in its string;
    /// [`AKD1000`]'s 1.2 million is a model-dependent vendor capacity claim graded
    /// [`Evidence::Projected`]. Neither is a register count, and because the `Loihi` 2 figure is a
    /// rounding chosen to be consistent with the per-core split, it is **excluded from**
    /// `a_published_chip_total_agrees_with_the_product_of_the_per_core_figures`'s independent
    /// cross-checks — a cross-check against a figure derived from the thing it checks is not one.
    pub neurons_per_chip_stated: Spec<u64>,
    /// What a reader has to know about this part that does not fit in a field.
    ///
    /// Never empty. Caveats live here and beside the figures, never in a footnote.
    pub note: &'static str,
}

impl Part {
    /// Neurons on one chip: the published chip total if there is one, otherwise
    /// `neurons_per_core * cores_per_chip`.
    ///
    /// `None` when neither is available. The published total is preferred because the product is
    /// two ceilings multiplied, and two ceilings multiplied is a number no configuration reaches.
    #[must_use]
    pub fn neurons_per_chip(&self) -> Option<u64> {
        if let Some(total) = self.neurons_per_chip_stated.value {
            return Some(total);
        }
        let per_core = u64::from(self.neurons_per_core.value?);
        let cores = u64::from(self.cores_per_chip.value?);
        per_core.checked_mul(cores)
    }

    /// Synapses on one chip: `synapses_per_core * cores_per_chip`.
    ///
    /// `None` when either is unstated. Carries the same two-ceilings caveat as
    /// [`Part::neurons_per_chip`].
    #[must_use]
    pub fn synapses_per_chip(&self) -> Option<u64> {
        let per_core = self.synapses_per_core.value?;
        let cores = u64::from(self.cores_per_chip.value?);
        per_core.checked_mul(cores)
    }

    /// The weakest grade anywhere in this record, which is the grade the record as a whole carries.
    ///
    /// [`Evidence::Unstated`] for every part in [`PARTS`], because every one of them has at least
    /// one field this review could not source. That is the honest summary and it is why this
    /// method exists rather than a `confidence` field somebody would have to maintain.
    #[must_use]
    pub fn weakest_evidence(&self) -> Evidence {
        let grades = [
            self.year.evidence,
            self.neurons_per_core.evidence,
            self.cores_per_chip.evidence,
            self.synapses_per_core.evidence,
            self.max_fan_in.evidence,
            self.weight_bits.evidence,
            self.delay_ticks.evidence,
            self.on_chip_learning.evidence,
            self.process.evidence,
            self.neurons_per_chip_stated.evidence,
        ];
        let mut worst = Evidence::Metered;
        for g in grades {
            if g < worst {
                worst = g;
            }
        }
        worst
    }

    /// How many of this record's ten graded fields carry a figure.
    ///
    /// The completeness of a record, as a number, so that a table can be sorted by how much is
    /// actually known about each part rather than by how confident it sounds.
    #[must_use]
    pub fn stated_fields(&self) -> usize {
        usize::from(self.year.is_known())
            + usize::from(self.neurons_per_core.is_known())
            + usize::from(self.cores_per_chip.is_known())
            + usize::from(self.synapses_per_core.is_known())
            + usize::from(self.max_fan_in.is_known())
            + usize::from(self.weight_bits.is_known())
            + usize::from(self.delay_ticks.is_known())
            + usize::from(self.on_chip_learning.is_known())
            + usize::from(self.process.is_known())
            + usize::from(self.neurons_per_chip_stated.is_known())
    }
}

/// The string used wherever a part publishes per-core figures and no separate chip total, so that
/// the empty chip-total field says the same thing once rather than sixteen times.
const CHIP_TOTAL_IS_A_PRODUCT: &str =
    "Not separately published; the published quantities for this part are per-core. \
     `Part::neurons_per_chip` multiplies them, with the caveat that a product of two ceilings is a \
     figure no real configuration reaches.";

/// Intel Loihi, the first widely-programmed research part with a learning engine on the die.
///
/// Davies et al., *Loihi: A Neuromorphic Manycore Processor with On-Chip Learning*, IEEE Micro
/// 38(1):82–99, 2018. Read [`crate::ledger::LOIHI_2018`] alongside this: the per-synaptic-operation
/// energy everyone quotes for this part is **pre-silicon**, and this crate graded it wrong for two
/// releases.
///
/// The fan-in field is empty, and that is the interesting thing about the record. Loihi's synapses
/// live in a block of memory shared across a core's neurons, so fan-in and neuron count **trade
/// against each other** rather than each having a cap. A network of 1,024 neurons with 64 inputs
/// each and a network of 128 neurons with 512 inputs each are both a core's worth of work, and no
/// single number describes the limit.
pub const LOIHI: Part = Part {
    name: "Loihi",
    vendor: "Intel",
    citation: "Davies et al., Loihi: A Neuromorphic Manycore Processor with On-Chip Learning, \
               IEEE Micro 38(1):82-99, 2018.",
    year: Spec::known(
        2018,
        "IEEE Micro publication year. The 14 nm part was fabricated before publication; this field \
         records the paper's year, which is the document the rest of the record rests on.",
        Evidence::Measured,
    ),
    neurons_per_core: Spec::known(
        1024,
        "Davies et al., IEEE Micro 38(1), 2018: up to 1,024 neurons per neuromorphic core. A \
         ceiling — a core holding 1,024 neurons has correspondingly less synaptic memory each.",
        Evidence::Measured,
    ),
    cores_per_chip: Spec::known(
        128,
        "Davies et al., 2018: 128 neuromorphic cores. The chip also carries 3 embedded x86 \
         Lakemont cores, which are not counted here because they hold no neurons.",
        Evidence::Measured,
    ),
    synapses_per_core: Spec::known(
        1_048_576,
        "DERIVED by this review: 128 KB of synaptic SRAM per core is 1,048,576 one-bit synapses, \
         the ceiling at the smallest weight precision. Cross-checks against the paper's ~130 M \
         synapses per chip: 130e6/128 = 1.016e6, agreeing to 3%. A 9-bit configuration holds \
         proportionally fewer and this review did not locate a per-precision table.",
        Evidence::Derived,
    ),
    max_fan_in: Spec::unlocated(
        "This review did not locate a published per-neuron fan-in cap for Loihi, and suspects \
         there is not one to locate: synapses live in memory shared across a core's neurons, so \
         fan-in trades against neuron count instead of having its own limit. Treat the core's \
         synapse budget as the binding constraint.",
    ),
    weight_bits: Spec::known(
        9,
        "Davies et al., 2018: weight precision configurable from 1 to 9 bits including sign. 9 is \
         the ceiling, and a network compiled at 9 bits fits about nine times fewer synapses per \
         core than one at 1 bit.",
        Evidence::Measured,
    ),
    delay_ticks: Spec::unlocated(
        "This review did not locate an axonal delay range in the IEEE Micro paper. Loihi's SDK is \
         widely reported to cap synaptic delay at 62 algorithmic timesteps; this review could not \
         trace that to a primary document and therefore does not record it as a figure.",
    ),
    on_chip_learning: Spec::known(
        true,
        "Davies et al., 2018: a programmable microcoded learning engine evaluating sums of \
         products over pre- and post-synaptic traces, covering pairwise STDP and three-factor \
         rules. The paper is titled for this feature.",
        Evidence::Measured,
    ),
    process: Spec::known(
        "Intel 14 nm",
        "Davies et al., 2018; 60 mm2 die. A node NAME. See the module doc on why this field is a \
         string.",
        Evidence::Measured,
    ),
    neurons_per_chip_stated: Spec::unlocated(CHIP_TOTAL_IS_A_PRODUCT),
    note: "The part the field benchmarks against, and the part whose headline energy figure is a \
           pre-silicon simulation. See crate::ledger::LOIHI_2018 for that correction.",
};

/// Intel Loihi 2 — eight times the neurons per core, graded spikes, and no architecture paper.
///
/// Every figure in this record rests on Intel's 2021 technology brief. This review did not locate a
/// peer-reviewed architecture paper for Loihi 2 comparable to Davies et al. 2018, which is why
/// nothing here is graded above `Measured`-from-a-vendor-document and why the weight width — the
/// one field a quantiser needs — is empty.
///
/// The architectural change worth knowing is the **graded spike**: a Loihi 2 spike carries a 32-bit
/// payload rather than being a bare event. That is a message width and not a weight width, and
/// reading it as 32-bit weights is the obvious error here.
pub const LOIHI_2: Part = Part {
    name: "Loihi 2",
    vendor: "Intel",
    citation: "Intel, Taking Neuromorphic Computing to the Next Level with Loihi 2 (technology \
               brief), 2021. VENDOR DOCUMENT. This review did not locate a peer-reviewed \
               architecture paper for this part.",
    year: Spec::known(
        2021,
        "Announcement and first silicon, per Intel's 2021 technology brief. Vendor document.",
        Evidence::Measured,
    ),
    neurons_per_core: Spec::known(
        8192,
        "DERIVED by this review: Intel's vendor brief states up to 1 M neurons per chip across 128 \
         cores, giving 1,048,576/128 = 8,192 per core, which matches the brief's own '8x Loihi 1' \
         claim against that part's 1,024. A vendor figure divided by a vendor figure.",
        Evidence::Derived,
    ),
    cores_per_chip: Spec::known(
        128,
        "Intel technology brief, 2021: 128 neuromorphic cores, unchanged from Loihi. Vendor \
         document.",
        Evidence::Measured,
    ),
    synapses_per_core: Spec::known(
        937_500,
        "DERIVED by this review from Intel's stated 120 M synapses per chip: 120e6/128 = 937,500. \
         A vendor figure divided by a vendor figure, not a per-core datasheet number.",
        Evidence::Derived,
    ),
    max_fan_in: Spec::unlocated(
        "This review did not locate a per-neuron fan-in cap for Loihi 2. The shared-synapse-memory \
         argument recorded for Loihi applies here too.",
    ),
    weight_bits: Spec::unlocated(
        "This review did not locate a stated weight bit-width for Loihi 2 in any document it read. \
         Loihi 2's graded spike carries a 32-bit payload; that is a MESSAGE width, not a weight \
         width, and this field is left empty rather than filled with it.",
    ),
    delay_ticks: Spec::unlocated(
        "This review did not locate a delay range for Loihi 2.",
    ),
    on_chip_learning: Spec::known(
        true,
        "Intel technology brief, 2021: programmable three-factor learning rules on a microcoded \
         learning engine, widened from Loihi 1's. Vendor document.",
        Evidence::Measured,
    ),
    process: Spec::known(
        "Intel 4 (pre-production)",
        "Intel technology brief, 2021. Intel's own node NAME, not a 4 nm dimension, and not \
         comparable to a foundry's '4 nm' by any physical measurement. Vendor document.",
        Evidence::Measured,
    ),
    neurons_per_chip_stated: Spec::known(
        1_048_576,
        "DERIVED by this review, and READ IT BEFORE QUOTING IT: Intel's vendor brief states 'up to \
         1 million' neurons per chip, which is a round number and not this one. 1,048,576 is this \
         review's reading of it as the power of two the per-core split gives, 128 x 8,192 = \
         1,048,576. Intel does not print 1,048,576 anywhere this review read. Because the figure \
         is chosen to be consistent with neurons_per_core, checking one against the other is an \
         identity and not a cross-check, and \
         a_published_chip_total_agrees_with_the_product_of_the_per_core_figures excludes this \
         record from its independent count for that reason.",
        Evidence::Derived,
    ),
    note: "The weight width is the field a quantiser needs and it is the field this review could \
           not source. Quantiser::for_part refuses for this part, which is the correct outcome.",
};

/// IBM `TrueNorth` — a million neurons, 4,096 fixed 256x256 crossbars, and no learning.
///
/// Merolla et al., *A million spiking-neuron integrated circuit with a scalable communication
/// network and interface*, Science 345(6197):668–673, 2014.
///
/// This is the clearest **wall** in the table. A neuron's inputs are one column of its core's
/// 256-row crossbar, so 256 is structural: a network needing 300 inputs to one neuron does not fit
/// on a `TrueNorth` of any size. The fix is to split the neuron into a summing tree, which costs
/// neurons out of the same budget and adds a tick per level.
///
/// The weight field is the other thing to read carefully. The crossbar stores **one bit** per
/// synapse — connected or not. Magnitude comes from an indirection: each axon carries a 2-bit type,
/// and each neuron holds four programmable signed weight values, one per axon type. So a neuron
/// sees at most **four distinct weight magnitudes across all 256 of its inputs**. Recording that as
/// "1 bit" is the structural truth and recording it as a multi-bit weight would be the marketing
/// one; the field says 1 and this paragraph says why neither number alone is honest.
pub const TRUENORTH: Part = Part {
    name: "TrueNorth",
    vendor: "IBM",
    citation: "Merolla et al., A million spiking-neuron integrated circuit with a scalable \
               communication network and interface, Science 345(6197):668-673, 2014.",
    year: Spec::known(2014, "Science publication year for the fabricated 28 nm part.", Evidence::Measured),
    neurons_per_core: Spec::known(
        256,
        "Merolla et al., Science 345(6197), 2014: 256 neurons per neurosynaptic core.",
        Evidence::Measured,
    ),
    cores_per_chip: Spec::known(
        4096,
        "Merolla et al., 2014: 4,096 neurosynaptic cores, giving 1,048,576 neurons per chip.",
        Evidence::Measured,
    ),
    synapses_per_core: Spec::known(
        65536,
        "Merolla et al., Science 345(6197), 2014: a full 256x256 binary crossbar per core, 256 \
         input axons by 256 neurons. The paper's abstract reports 256 MILLION configurable \
         synapses per chip, in BINARY millions: 256 x 2^20 = 268,435,456, which is exactly 4,096 \
         cores x 65,536. The decimal rendering 268 M is this review's arithmetic and is NOT a \
         figure the paper prints; the paper's own words are '256 million'.",
        Evidence::Measured,
    ),
    max_fan_in: Spec::known(
        256,
        "STRUCTURAL: one column of the core's 256-row crossbar. Merolla et al., 2014. This is a \
         wall, not a budget — more chips do not relieve it.",
        Evidence::Measured,
    ),
    weight_bits: Spec::known(
        1,
        "The crossbar stores ONE BIT per synapse: connected or not. Magnitude comes from a 2-bit \
         axon type selecting one of four programmable signed weight values held per neuron, so a \
         neuron sees at most four distinct magnitudes across its 256 inputs. This review did not \
         locate the width of those four values. See the part's doc comment.",
        Evidence::Measured,
    ),
    delay_ticks: Spec::unlocated(
        "This review did not locate the axonal delay range in Merolla et al. 2014. The chip runs on \
         a 1 kHz global tick and does implement per-axon delay; the width of that field is not \
         recorded here because this review could not trace it to a document it read.",
    ),
    on_chip_learning: Spec::known(
        false,
        "No. TrueNorth is an inference part: weights are trained offline in a constrained-network \
         toolchain and loaded. Merolla et al., 2014. A positive claim from the paper, not an \
         absence of information.",
        Evidence::Measured,
    ),
    process: Spec::known(
        "Samsung 28 nm LPP",
        "Merolla et al., 2014; 5.4 billion transistors, 4.3 cm2 die.",
        Evidence::Measured,
    ),
    neurons_per_chip_stated: Spec::known(
        1_048_576,
        "Merolla et al., 2014: one million neurons, exactly 4,096 x 256 = 1,048,576. The title's \
         'million' is this number rounded, not a separate claim.",
        Evidence::Measured,
    ),
    note: "The 26 pJ per synaptic event in crate::ledger::TRUENORTH_2014 is, as of this writing, \
           the only per-operation energy in this crate taken on fabricated silicon.",
};

/// IBM `NorthPole` — **not a spiking part**, and in this table to say so.
///
/// Modha et al., *Neural inference at the frontier of energy, space, and time*, Science
/// 382(6668):329–335, 2023.
///
/// ⚠ `NorthPole` has no membrane potential, no spikes and no synaptic delay. It is a digital
/// inference accelerator whose design principle is that the entire model stays in 224 MB of on-chip
/// memory with no external model memory at all. It is filed under "neuromorphic" constantly,
/// because it is IBM's successor to [`TRUENORTH`] and by the same group, and a reader comparing
/// neurons per core down this table would otherwise divide by a quantity that does not exist here.
///
/// Its empty fields are marked "not applicable" in their provenance strings rather than "not
/// located", and those are different findings.
pub const NORTHPOLE: Part = Part {
    name: "NorthPole",
    vendor: "IBM",
    citation: "Modha et al., Neural inference at the frontier of energy, space, and time, Science \
               382(6668):329-335, 2023.",
    year: Spec::known(2023, "Science publication year for the fabricated 12 nm part.", Evidence::Measured),
    neurons_per_core: Spec::unlocated(
        "NOT APPLICABLE rather than not located: NorthPole has no membrane state and no spikes. \
         The comparable quantity is its 224 MB of on-chip memory, which holds weights and \
         activations for a whole network.",
    ),
    cores_per_chip: Spec::known(
        256,
        "Modha et al., Science 382(6668), 2023: 256 cores. A 'core' here is a dense compute tile, \
         not a neurosynaptic core.",
        Evidence::Measured,
    ),
    synapses_per_core: Spec::unlocated(
        "NOT APPLICABLE: weights are dense tensors in on-chip memory, not addressed synapses.",
    ),
    max_fan_in: Spec::unlocated("NOT APPLICABLE: there is no per-neuron input structure."),
    weight_bits: Spec::known(
        8,
        "Modha et al., 2023: 8-, 4- and 2-bit precisions are supported, with the headline network \
         results reported at the lower two. Recorded as 8, the ceiling.",
        Evidence::Measured,
    ),
    delay_ticks: Spec::unlocated("NOT APPLICABLE: there is no spike and therefore no delay."),
    on_chip_learning: Spec::known(
        false,
        "Inference only; training is off-chip. Modha et al., 2023.",
        Evidence::Measured,
    ),
    process: Spec::known(
        "12 nm",
        "Modha et al., 2023. A large die by neuromorphic standards, which is forced by the design \
         principle that the whole model stays on chip.",
        Evidence::Measured,
    ),
    neurons_per_chip_stated: Spec::unlocated("NOT APPLICABLE: see neurons_per_core."),
    note: "In this table as a correction, not as a spiking part. fits() will report almost every \
           constraint as unchecked for it, which is the honest answer to 'does my SNN map onto \
           NorthPole' — the question does not typecheck.",
};

/// `BrainChip` Akida AKD1000 — the first commercially purchasable event-based accelerator.
///
/// Every figure here comes from `BrainChip` vendor material. This review did not locate a
/// peer-reviewed architecture paper for the fabricated part.
///
/// The headline "1.2 million neurons and 10 billion synapses" is a **model-dependent capacity
/// claim** for an event-driven convolutional engine, not a register count. Dividing it by the 80
/// processing units to manufacture a per-core figure is exactly what this module refuses to do, so
/// `neurons_per_core` here is empty and the headline lives in `neurons_per_chip_stated` where it
/// can be read with its caveat.
pub const AKD1000: Part = Part {
    name: "Akida AKD1000",
    vendor: "BrainChip",
    citation: "BrainChip AKD1000 product brief and Akida development-kit documentation. VENDOR \
               SOURCES. This review did not locate a peer-reviewed architecture paper for the \
               fabricated part.",
    year: Spec::known(
        2021,
        "Silicon reported back in 2021; development kits shipped from 2022. Vendor announcements.",
        Evidence::Measured,
    ),
    neurons_per_core: Spec::unlocated(
        "This review did not locate a per-NPU neuron limit. BrainChip's device total divided by 80 \
         NPUs would give 15,000 each, and that division would manufacture a specification from a \
         model-dependent capacity claim.",
    ),
    cores_per_chip: Spec::known(
        80,
        "BrainChip product brief: 80 Neural Processing Units. Vendor document.",
        Evidence::Measured,
    ),
    synapses_per_core: Spec::unlocated(
        "This review did not locate a per-NPU synapse limit; the vendor's 10 billion is a device \
         capacity claim and is not divided here.",
    ),
    max_fan_in: Spec::unlocated(
        "This review did not locate a fan-in cap. Akida's fan-in is a configured convolution kernel \
         rather than a crossbar column, so a single cap may not be the right shape for this part.",
    ),
    weight_bits: Spec::known(
        4,
        "BrainChip product brief: 1-, 2- and 4-bit weights and activations, with the MetaTF \
         quantisation-aware training flow targeting the same three widths. Recorded as 4, the \
         ceiling. Vendor document.",
        Evidence::Measured,
    ),
    delay_ticks: Spec::unlocated(
        "This review did not locate an axonal delay mechanism. Akida's event model is event-driven \
         convolution over spatial events rather than a delay-line network, so a delay range may not \
         be a meaningful field for this part.",
    ),
    on_chip_learning: Spec::known(
        true,
        "Yes, but ONLY on the final fully-connected layer, as a one-shot/few-shot rule over \
         binarised activations. It is not backpropagation and it does not update the convolutional \
         stack. Vendor documentation. A bare `true` would overstate it, which is why this string is \
         long.",
        Evidence::Measured,
    ),
    process: Spec::known(
        "28 nm",
        "Vendor material describes a 28 nm process. This review did not confirm the foundry and \
         does not name one.",
        Evidence::Measured,
    ),
    neurons_per_chip_stated: Spec::known(
        1_200_000,
        "BrainChip's headline: up to 1.2 million neurons and 10 billion synapses per device. A \
         MODEL-DEPENDENT capacity claim from a vendor brief, not a register count, and it is \
         recorded here rather than divided into a per-core figure.",
        Evidence::Projected,
    ),
    note: "The first part in this table an individual could buy on a PCIe card. Its on-chip \
           learning is real and is confined to one layer.",
};

/// `BrainChip` Akida AKD1500 — the same Akida 1.0 fabric in `GlobalFoundries` 22 nm `FD-SOI`.
///
/// A deliberately thin record: three stated fields and seven empty ones. `BrainChip` describes
/// AKD1500 as carrying the same Akida 1.0 intellectual property as [`AKD1000`] in a different
/// process and package, without the embedded host processor or the external `DRAM` interface —
/// which **implies** the same 80 processing units without **stating** them, so this record does not
/// state them either.
pub const AKD1500: Part = Part {
    name: "Akida AKD1500",
    vendor: "BrainChip",
    citation: "BrainChip announcement of AKD1500 reference silicon, 2023. VENDOR ANNOUNCEMENT. \
               This review did not locate a datasheet for this part.",
    year: Spec::known(
        2023,
        "Vendor announcement of reference silicon; samples reported from 2024.",
        Evidence::Measured,
    ),
    neurons_per_core: Spec::unlocated("This review did not locate any per-core figure for AKD1500."),
    cores_per_chip: Spec::unlocated(
        "This review did not locate a stated NPU count for AKD1500. BrainChip describes it as the \
         same Akida 1.0 IP as AKD1000, which implies 80 and does not state it.",
    ),
    synapses_per_core: Spec::unlocated("This review did not locate any per-core figure for AKD1500."),
    max_fan_in: Spec::unlocated("This review did not locate a fan-in cap for AKD1500."),
    weight_bits: Spec::known(
        4,
        "INFERRED by this review, not located and not computed, from BrainChip's statement that \
         AKD1500 carries the same Akida 1.0 IP as AKD1000, whose brief gives 1, 2 and 4 bits. Not \
         separately stated for this part. Graded Projected because an analogy is not arithmetic. \
         The inference is made here and refused for on_chip_learning below on purpose: a datapath \
         width is a property of the IP itself, whereas whether a learning feature is usable also \
         depends on the parts of AKD1000 this device drops, including its embedded host processor.",
        Evidence::Projected,
    ),
    delay_ticks: Spec::unlocated("This review did not locate a delay mechanism for AKD1500."),
    on_chip_learning: Spec::unlocated(
        "This review did not locate a statement for AKD1500 specifically, and does not carry \
         AKD1000's across on the strength of shared IP.",
    ),
    process: Spec::known(
        "GlobalFoundries 22 nm FD-SOI",
        "BrainChip's 2023 announcement: AKD1500 reference silicon in GF 22FDX. Vendor \
         announcement.",
        Evidence::Measured,
    ),
    neurons_per_chip_stated: Spec::unlocated("This review did not locate a device total for AKD1500."),
    note: "Three stated fields and seven empty. This is what a correct record looks like for a part \
           whose public material is an announcement rather than a datasheet.",
};

/// `SpiNNaker` — a million `ARM` cores, and **no structural limits at all**.
///
/// Furber, Galluppi, Temple & Plana, *The `SpiNNaker` Project*, Proceedings of the IEEE
/// 102(5):652–665, 2014; the chip is described in Painkras et al., IEEE Journal of Solid-State
/// Circuits 48(8):1943–1953, 2013.
///
/// This is the **budget** end of the table and the reason the wall/budget distinction is worth
/// teaching. A `SpiNNaker` synapse is a row in die-stacked `SDRAM` that a software `ARM` core
/// fetches by `DMA` and processes in a loop. There is no crossbar, so there is no fan-in cap. Ask
/// for 4,000 inputs to one neuron and the part does not refuse — it **misses its 1 ms real-time
/// deadline**, and a missed deadline changes the answer without saying so. That is a strictly worse
/// failure mode than [`TRUENORTH`]'s refusal, and it is why `max_fan_in` here is empty rather than
/// large: an empty field makes [`fits`] report the constraint as *unchecked*, which is true.
///
/// ⛔ **`neurons_per_core` and `synapses_per_core` are empty here for exactly the same reason, and
/// they were not always.** Through v0.4.0 this record carried 1,000 and 1,000,000 — the design
/// target of ~1,000 neurons at ~1,000 inputs each in biological real time — in fields that
/// [`Part::neurons_per_chip`] multiplies by the core count and [`fits`] then grades a network
/// against. A 17,500-neuron network came back `FITS ... 17500 of 18000 (97.2%)`: a **deadline**
/// reported as a **wall** with headroom, on a part that would have accepted the network and missed
/// every timestep. The figures are not deleted — they are in the provenance strings below, where
/// they can be read and not multiplied. This is the same argument that emptied `max_fan_in`,
/// applied to the two fields that argument originally missed.
pub const SPINNAKER: Part = Part {
    name: "SpiNNaker",
    vendor: "University of Manchester",
    citation: "Furber, Galluppi, Temple & Plana, The SpiNNaker Project, Proc. IEEE 102(5):652-665, \
               2014; chip in Painkras et al., IEEE JSSC 48(8):1943-1953, 2013.",
    year: Spec::known(
        2014,
        "Proc. IEEE publication year. First silicon was 2011 and the million-core machine was \
         completed in 2018; this field records the paper this record rests on.",
        Evidence::Measured,
    ),
    neurons_per_core: Spec::unlocated(
        "There is NO structural per-core neuron cap, and this empty field is the finding rather \
         than a gap. The figure everyone quotes - the design target of ~1,000 neurons per core at \
         ~1,000 inputs each in biological real time with a 1 ms timestep, Furber et al., Proc. \
         IEEE 102(5), 2014 - is a THROUGHPUT BUDGET and not a capacity: the part accepts more \
         neurons and misses its deadline instead of refusing. This field is the wrong SHAPE for \
         that number, because Part::neurons_per_chip would multiply it by the core count and \
         fits() would then report headroom against a rate.",
    ),
    cores_per_chip: Spec::known(
        18,
        "Painkras et al., IEEE JSSC 48(8), 2013: 18 ARM968 cores per chip. One is reserved as a \
         monitor processor and one is a spare for yield, so 16 are typically deployable. This field \
         records the 18 on the die and names the 16 here rather than silently picking one.",
        Evidence::Measured,
    ),
    synapses_per_core: Spec::unlocated(
        "There is NO structural per-core synapse capacity in the sense this field means. The 10^6 \
         this review previously recorded here is 1,000 neurons x 1,000 inputs - the same 1 ms \
         throughput budget seen from the synapse side, a rate rather than a store. The hard \
         resources are the chip's 128 MB of die-stacked SDRAM holding synapse rows and each core's \
         96 KB of tightly-coupled memory, and this review did not locate a synapses-per-core \
         figure derived from those rather than from the deadline. The field is the wrong SHAPE for \
         a budget: core_count() would pack against it and fits() would report headroom.",
    ),
    max_fan_in: Spec::unlocated(
        "There is NO structural fan-in cap, and this empty field is the finding rather than a gap. \
         A synapse row is fetched from SDRAM by DMA and processed in software, so fan-in costs TIME \
         and is never refused. The limit is that the row must be fetched and processed inside the \
         1 ms timestep.",
    ),
    weight_bits: Spec::known(
        16,
        "Located in the documentation of sPyNNaker, the project's own reference software stack: \
         the standard synapse format uses 16-bit fixed-point weights. A SOFTWARE convention, not a \
         hardware limit - the cores are 32-bit ARMs and a different synapse format would be a \
         different number. Graded Measured because it was READ OFF a document rather than computed \
         here; the document is a software one and this string says so.",
        Evidence::Measured,
    ),
    delay_ticks: Spec::known(
        DelayRange { min_ticks: 1, max_ticks: 16 },
        "sPyNNaker delivers 1 to 16 timesteps natively from the synapse row's delay field; longer \
         delays are built from 'delay extension' populations that relay a spike through extra \
         neurons, at the cost of those neurons. Located in the documentation of sPyNNaker, the \
         project's own reference software stack, rather than in the JSSC paper - a software \
         document, READ OFF and not computed here, which is why it is Measured and not Derived. \
         The minimum of 1 is real: there is no same-tick delivery.",
        Evidence::Measured,
    ),
    on_chip_learning: Spec::known(
        true,
        "STDP and other rules run in software on the same ARM cores. 'On-chip' here means 'in the \
         chip's own processors', which is a different claim from Loihi's dedicated learning engine \
         and spends the same real-time budget the neurons are spending.",
        Evidence::Measured,
    ),
    process: Spec::known(
        "UMC 130 nm",
        "Painkras et al., IEEE JSSC 48(8), 2013. The node is old on purpose: the project's binding \
         constraint was cost per core at a million-core scale.",
        Evidence::Measured,
    ),
    neurons_per_chip_stated: Spec::unlocated(
        "This review did not locate a per-chip neuron figure that is a CAPACITY. The figure that \
         exists - 18 cores x ~1,000 neurons - is the 1 ms throughput budget multiplied out, and \
         recording it here would put a deadline in a capacity field. See neurons_per_core.",
    ),
    note: "The only part in this table whose limits are deadlines rather than structures, and the \
           only one with NO capacity field filled in: the three that were filled held throughput \
           budgets, so fits() reports every capacity constraint as unchecked and checks only the \
           delay range, which is the one genuine structure. Its failure mode is a missed deadline, \
           which is quieter and worse than a refusal.",
};

/// `SpiNNaker2` — 152 `Cortex-M4F` cores per chip in `GlobalFoundries` 22 nm `FDX`.
///
/// Höppner, Yan, Mayr et al., *`SpiNNaker2`: A Large-Scale Neuromorphic System for Event-Based and
/// Asynchronous Machine Learning*, arXiv:2103.08392 (2021); commercialised by `SpiNNcloud`.
///
/// The record is thin for the same reason [`SPINNAKER`]'s is: this is a software platform with
/// hardware accelerators, so neuron counts, synapse counts, weight widths and delays are properties
/// of the model being run rather than of the silicon. Adaptive body biasing and per-core
/// voltage/frequency scaling make even the throughput budget a function of the operating point, so
/// there is not a single number to record.
pub const SPINNAKER2: Part = Part {
    name: "SpiNNaker2",
    vendor: "TU Dresden / SpiNNcloud",
    citation: "Hoeppner, Yan, Mayr et al., SpiNNaker2: A Large-Scale Neuromorphic System for \
               Event-Based and Asynchronous Machine Learning, arXiv:2103.08392, 2021.",
    year: Spec::known(
        2021,
        "arXiv preprint year for the architecture; commercial systems from SpiNNcloud followed.",
        Evidence::Measured,
    ),
    neurons_per_core: Spec::unlocated(
        "This review did not locate a per-core neuron limit, and as with SpiNNaker 1 the limit is a \
         software throughput budget rather than a structure. Per-core DVFS and adaptive body \
         biasing make that budget a function of the operating point.",
    ),
    cores_per_chip: Spec::known(
        152,
        "arXiv:2103.08392: 152 ARM Cortex-M4F processing elements per chip, each with MAC and \
         exponential accelerators and its own SRAM.",
        Evidence::Measured,
    ),
    synapses_per_core: Spec::unlocated(
        "This review did not locate a per-core synapse limit; see neurons_per_core.",
    ),
    max_fan_in: Spec::unlocated(
        "There is NO structural fan-in cap, for the same reason as SpiNNaker 1: a synapse row is \
         processed in software, so fan-in costs time and is never refused. This review did not \
         locate a cap because the architecture does not have one to state.",
    ),
    weight_bits: Spec::unlocated(
        "This review did not locate a fixed weight width. SpiNNaker2 is a software platform with \
         hardware MAC accelerators, so the width is a property of the model rather than the part.",
    ),
    delay_ticks: Spec::unlocated(
        "This review did not locate a delay range for SpiNNaker2's software stack.",
    ),
    on_chip_learning: Spec::known(
        true,
        "INFERRED by this review, not located and not computed: software plasticity on the \
         Cortex-M4F cores as in SpiNNaker 1, with hardware MAC and exponential units making the \
         rule cheaper. arXiv:2103.08392 describes the accelerators; this review did not locate a \
         statement of on-chip plasticity for this part or a plasticity benchmark. An analogy from \
         a sibling architecture is an inference, so this is graded Projected and NOT Derived: \
         there is no arithmetic here, and Derived would sort this guess above a maker's roadmap \
         figure.",
        Evidence::Projected,
    ),
    process: Spec::known(
        "GlobalFoundries 22 nm FDX",
        "Hoeppner, Yan, Mayr et al., arXiv:2103.08392, 2021, which also describes adaptive body \
         biasing and per-core voltage and frequency scaling on this node.",
        Evidence::Measured,
    ),
    neurons_per_chip_stated: Spec::unlocated(
        "This review did not locate a per-chip neuron figure it could trace to a document, as \
         distinct from system-level totals quoted for SpiNNcloud machines.",
    ),
    note: "Quantiser::for_part refuses for this part: a platform whose weight width is a property \
           of the model has no bit-width to quantise to, and that is the right answer.",
};

/// `SynSense` Xylo Audio 2 (`SYNS61201`) — 1,000 `LIF` neurons for always-on audio.
///
/// The small end of the table, and one of the very few parts in this crate with a **measured**
/// energy figure attached to a stated protocol: the `NeuroBench` system track reports 0.028 mJ per
/// inference for Xylo Audio 2 against 0.934 mJ for an Arduino Nano 33 `BLE` on the same task, with
/// idle, active and dynamic power reported separately. That figure is cited in
/// [`crate::ledger`]'s module doc and is **not** turned into a [`crate::ledger::Prices`] entry
/// here, for the reason given in this module's doc.
///
/// Note what the fan-in field does **not** say. 64,000 synapses over 1,000 neurons is 64 — and 64
/// is a **mean**, not a maximum. Putting a mean in a maximum's field is the specific error this
/// module exists to prevent, so the field is empty.
pub const XYLO_AUDIO_2: Part = Part {
    name: "Xylo Audio 2",
    vendor: "SynSense",
    citation: "SynSense Xylo product documentation and the Rockpool toolchain. VENDOR SOURCES. \
               Measured per-inference energy for this part appears in the NeuroBench system track \
               (Yik et al.).",
    year: Spec::known(
        2023,
        "INFERRED by this review, not located and not computed: the year is read off the vendor \
         documentation and the NeuroBench evaluation rather than a dated launch announcement. \
         Treat it as approximate. Graded Projected, because an undated guess is an inference and \
         not arithmetic over stated figures.",
        Evidence::Projected,
    ),
    neurons_per_core: Spec::known(
        1000,
        "SynSense Xylo documentation: up to 1,000 LIF neurons in the digital SNN core. Vendor \
         document.",
        Evidence::Measured,
    ),
    cores_per_chip: Spec::known(
        1,
        "SynSense documentation describes a single SNN core; recording 1 here keeps the per-core \
         and per-chip columns consistent. Vendor document.",
        Evidence::Measured,
    ),
    synapses_per_core: Spec::known(
        64000,
        "SynSense Xylo documentation: up to 64,000 synaptic connections. Vendor document.",
        Evidence::Measured,
    ),
    max_fan_in: Spec::unlocated(
        "This review did not locate a per-neuron fan-in cap. 64,000/1,000 = 64 is a MEAN, not a \
         maximum, and a mean in a maximum's field would let fits() pass a network it should refuse.",
    ),
    weight_bits: Spec::known(
        8,
        "SynSense Xylo documentation: 8-bit signed synaptic weights in the digital core. Vendor \
         document.",
        Evidence::Measured,
    ),
    delay_ticks: Spec::unlocated("This review did not locate a synaptic delay range for Xylo."),
    on_chip_learning: Spec::known(
        false,
        "Inference only. Training is off-chip in Rockpool and the network is deployed as a fixed \
         configuration. Vendor documentation.",
        Evidence::Measured,
    ),
    process: Spec::unlocated("This review did not locate a process node for Xylo."),
    neurons_per_chip_stated: Spec::unlocated(CHIP_TOTAL_IS_A_PRODUCT),
    note: "The part behind one of the few published spiking energy figures with a stated \
           measurement protocol. See crate::ledger's module doc.",
};

/// `SynSense` Speck — a `DVS` pixel array and nine event-driven convolutional cores in one package.
///
/// The record is thin on purpose. What this review could source is the shape of the pipeline: an
/// integrated 128x128 dynamic-vision pixel array feeding nine event-driven convolutional spiking
/// cores, so **a network deeper than nine convolutional layers does not map** without
/// time-multiplexing. That is a real and checkable structural limit, and it is a different shape
/// from every other limit in this table — it constrains **depth**, not width.
pub const SPECK: Part = Part {
    name: "Speck",
    vendor: "SynSense",
    citation: "SynSense Speck product materials. VENDOR SOURCES. This review did not locate a \
               peer-reviewed paper for the packaged part; the DYNAP-CNN convolutional core it \
               builds on comes out of the Institute of Neuroinformatics, Zurich.",
    year: Spec::known(
        2022,
        "INFERRED by this review from undated vendor material, not located and not computed; this \
         review did not locate a dated launch document, so treat the year as approximate. Graded \
         Projected, because a year guessed off undated material is an inference and not arithmetic \
         over stated figures.",
        Evidence::Projected,
    ),
    neurons_per_core: Spec::unlocated(
        "This review did not locate a per-core neuron limit. The vendor quotes a device total in \
         the hundreds of thousands that this review could not trace to a document it read, so no \
         figure is recorded in either the per-core or the per-chip field.",
    ),
    cores_per_chip: Spec::known(
        9,
        "SynSense Speck material: nine event-driven convolutional SNN cores, mapping one per \
         network layer. Vendor document. A network deeper than nine layers does not fit without \
         time-multiplexing.",
        Evidence::Measured,
    ),
    synapses_per_core: Spec::unlocated("This review did not locate a per-core synapse limit."),
    max_fan_in: Spec::unlocated(
        "A convolutional core's fan-in is its kernel size, a property of the configured network \
         rather than a fixed cap. This review did not locate the kernel-size limits.",
    ),
    weight_bits: Spec::unlocated(
        "This review did not locate a weight width it could trace to a vendor document it read.",
    ),
    delay_ticks: Spec::unlocated("This review did not locate a delay mechanism for Speck."),
    on_chip_learning: Spec::known(
        false,
        "Inference only; networks are trained off-chip and deployed. Vendor documentation.",
        Evidence::Measured,
    ),
    process: Spec::unlocated("This review did not locate a process node for Speck."),
    neurons_per_chip_stated: Spec::unlocated(
        "This review did not locate a device total it could trace to a document it read; see \
         neurons_per_core.",
    ),
    note: "The sensor is ON the die. Speck is the clearest example in this table of the thing that \
           makes event-based hardware interesting — no frame ever exists, so no frame is ever moved.",
};

/// ODIN — 256 neurons, 64k synapses, 0.086 mm2, and **the spec is in the title**.
///
/// Frenkel, Lefebvre, Legat & Bol, *A 0.086-mm² 12.7-pJ/SOP 64k-Synapse 256-Neuron Online-Learning
/// Digital Spiking Neuromorphic Processor in 28-nm CMOS*, IEEE Transactions on Biomedical Circuits
/// and Systems 13(1):145–158, 2019.
///
/// That title is why this is the best-documented record in the table: the neuron count, the synapse
/// count, the die area, the process and the per-operation energy are all in it, and the energy is a
/// **silicon measurement** rather than a pre-silicon simulation. This module does not turn that
/// 12.7 pJ/SOP into a [`crate::ledger::Prices`] entry — see the module doc for why not.
///
/// ODIN is in this table as the small end of the scale: four orders of magnitude smaller than
/// [`LOIHI`], with per-synapse plasticity on every one of its 65,536 synapses.
pub const ODIN: Part = Part {
    name: "ODIN",
    vendor: "Universite catholique de Louvain",
    citation: "Frenkel, Lefebvre, Legat & Bol, A 0.086-mm2 12.7-pJ/SOP 64k-Synapse 256-Neuron \
               Online-Learning Digital Spiking Neuromorphic Processor in 28-nm CMOS, IEEE \
               Transactions on Biomedical Circuits and Systems 13(1):145-158, 2019.",
    year: Spec::known(2019, "IEEE TBioCAS publication year for the fabricated part.", Evidence::Measured),
    neurons_per_core: Spec::known(
        256,
        "In the paper's title: 256 neurons. Each can be configured as a LIF cell or to reproduce \
         Izhikevich behaviours.",
        Evidence::Measured,
    ),
    cores_per_chip: Spec::known(
        1,
        "One core, 0.086 mm2. ODIN is a single-core research part.",
        Evidence::Measured,
    ),
    synapses_per_core: Spec::known(
        65536,
        "'64k-Synapse' in the paper's title: a 256x256 crossbar, the same shape as one TrueNorth \
         core.",
        Evidence::Measured,
    ),
    max_fan_in: Spec::known(
        256,
        "STRUCTURAL: one column of the 256x256 crossbar. Frenkel et al., IEEE TBioCAS 13(1), 2019.",
        Evidence::Measured,
    ),
    weight_bits: Spec::known(
        4,
        "Frenkel et al., 2019: 4-bit synaptic weights, with an additional bit per synapse carrying \
         the SDSP mapping state. This review records the 4 and names the fifth bit here rather \
         than folding it in, because it is a plasticity bit and not magnitude.",
        Evidence::Measured,
    ),
    delay_ticks: Spec::unlocated("This review did not locate a synaptic delay mechanism in ODIN."),
    on_chip_learning: Spec::known(
        true,
        "Spike-driven synaptic plasticity (Brader, Senn & Fusi, Neural Computation 19(11), 2007) \
         in hardware, on every synapse. 'Online-Learning' is in the paper's title.",
        Evidence::Measured,
    ),
    process: Spec::known(
        "28 nm FDSOI",
        "Frenkel et al., 2019. '28-nm CMOS' is in the title; FDSOI is stated in the paper.",
        Evidence::Measured,
    ),
    neurons_per_chip_stated: Spec::unlocated(CHIP_TOTAL_IS_A_PRODUCT),
    note: "Reports 12.7 pJ per synaptic operation at 0.55 V, MEASURED on fabricated silicon. This \
           module deliberately does not transcribe that into a Prices entry; see the module doc.",
};

/// `DYNAP-SE` — analogue neurons, and the tightest fan-in wall in this table: **64**.
///
/// Moradi, Qiao, Stefanini & Indiveri, *A Scalable Multicore Architecture With Heterogeneous Memory
/// Structures for Dynamic Neuromorphic Asynchronous Processors (`DYNAPs`)*, IEEE Transactions on
/// Biomedical Circuits and Systems 12(1):106–122, 2018.
///
/// Each neuron has **64** content-addressable-memory entries, each holding the address of one
/// presynaptic source. Sixty-four. A network whose neurons need 200 inputs does not map without
/// splitting neurons across the fabric, and no amount of silicon changes that.
///
/// The weight field needs its paragraph. A `DYNAP-SE` synapse **does not store a number.** A `CAM`
/// entry selects one of four synapse types per neuron, and the amplitude of each type is set by an
/// **analogue bias current shared across the whole core**. So the digital resolution is 2 bits of
/// type selection, the magnitude is a core-level parameter rather than a per-synapse one, and the
/// achieved weight varies neuron to neuron with device mismatch. Recording "2 bits" is the
/// structural truth; treating the part as having 2-bit digital weights would be wrong in both
/// directions at once.
pub const DYNAP_SE: Part = Part {
    name: "DYNAP-SE",
    vendor: "Institute of Neuroinformatics, Zurich",
    citation: "Moradi, Qiao, Stefanini & Indiveri, A Scalable Multicore Architecture With \
               Heterogeneous Memory Structures for Dynamic Neuromorphic Asynchronous Processors \
               (DYNAPs), IEEE Transactions on Biomedical Circuits and Systems 12(1):106-122, 2018.",
    year: Spec::known(2018, "IEEE TBioCAS publication year for the fabricated part.", Evidence::Measured),
    neurons_per_core: Spec::known(
        256,
        "Moradi et al., IEEE TBioCAS 12(1), 2018: 256 analogue adaptive-exponential neuron \
         circuits per core.",
        Evidence::Measured,
    ),
    cores_per_chip: Spec::known(
        4,
        "Moradi et al., 2018: four cores per chip, 1,024 neurons in total.",
        Evidence::Measured,
    ),
    synapses_per_core: Spec::known(
        16384,
        "DERIVED by this review: 256 x 64 = 16,384, that is, 256 neurons per core at 64 CAM \
         entries each. The paper states the 64 per neuron and the 256 per core; the product is \
         this review's arithmetic.",
        Evidence::Derived,
    ),
    max_fan_in: Spec::known(
        64,
        "STRUCTURAL and small: 64 content-addressable-memory entries per neuron, each holding one \
         presynaptic address. Moradi et al., 2018. The tightest fan-in in this table.",
        Evidence::Measured,
    ),
    weight_bits: Spec::known(
        2,
        "2 bits of SYNAPSE TYPE selection, not 2 bits of magnitude. The amplitude of each of the \
         four types is an analogue bias current shared across the whole core, so the weight is a \
         core-level parameter modulated by per-device mismatch. See the part's doc comment before \
         quantising anything to this field.",
        Evidence::Measured,
    ),
    delay_ticks: Spec::unlocated(
        "There is no digital delay line, and this field is the wrong SHAPE for this part rather \
         than merely empty: synaptic time constants are set by analogue bias currents and produce \
         a continuous temporal response, not an integer tick delay.",
    ),
    on_chip_learning: Spec::known(
        false,
        "DYNAP-SE1 has no on-chip plasticity. A sibling part, DYNAP-SEL, carries plastic synapses \
         with on-chip learning on a smaller neuron count; this record is for DYNAP-SE and does not \
         borrow it.",
        Evidence::Measured,
    ),
    process: Spec::known(
        "180 nm CMOS",
        "Moradi et al., 2018. Mixed-signal: subthreshold analogue neuron and synapse circuits with \
         asynchronous digital routing.",
        Evidence::Measured,
    ),
    neurons_per_chip_stated: Spec::known(
        1024,
        "Moradi et al., 2018 states 1,024 neurons per chip directly, which also equals 4 x 256.",
        Evidence::Measured,
    ),
    note: "Subthreshold analogue circuits mean two nominally identical neurons are not identical. \
           Device mismatch is a first-class property of this part, not a defect, and no field in \
           this record captures it.",
};

/// Darwin — Zhejiang University's 2016 co-processor, 2,048 time-multiplexed neurons.
///
/// Shen, Ma, Deng et al., *Darwin: a neuromorphic hardware co-processor based on spiking neural
/// networks*, Science China Information Sciences 59, 2016.
///
/// The neuron count is a **virtual** one: the part time-multiplexes a small number of physical
/// neuron circuits to present 2,048 addressable neurons. This review did not confirm how many
/// physical circuits. The distinction matters for energy — the joules are paid per physical update
/// — and not for mapping, which is what this module checks, so the virtual count is what the field
/// records and this paragraph is where the caveat lives.
pub const DARWIN: Part = Part {
    name: "Darwin",
    vendor: "Zhejiang University",
    citation: "Shen, Ma, Deng et al., Darwin: a neuromorphic hardware co-processor based on spiking \
               neural networks, Science China Information Sciences 59, 2016.",
    year: Spec::known(2016, "Publication year for the fabricated 180 nm part.", Evidence::Measured),
    neurons_per_core: Spec::known(
        2048,
        "Shen et al., 2016: up to 2,048 neurons, presented by time-multiplexing a smaller number of \
         physical neuron circuits. This review did not confirm the physical count.",
        Evidence::Measured,
    ),
    cores_per_chip: Spec::known(1, "A single-core co-processor.", Evidence::Measured),
    synapses_per_core: Spec::known(
        4_194_304,
        "DERIVED by this review: the paper reports over 4 million synapses, and 2,048^2 = 4,194,304 \
         is all-to-all among its neurons, which is this review's reading of where that figure comes \
         from rather than a number the paper prints.",
        Evidence::Derived,
    ),
    max_fan_in: Spec::unlocated(
        "This review did not locate a fan-in cap. If the 4-million figure is indeed all-to-all \
         storage then the cap is 2,048, and that inference is not strong enough to record as a \
         specification.",
    ),
    weight_bits: Spec::unlocated("This review did not locate a weight width for Darwin."),
    delay_ticks: Spec::unlocated("This review did not locate a delay range for Darwin."),
    on_chip_learning: Spec::unlocated(
        "This review did not locate a statement about on-chip plasticity in Darwin, in either \
         direction.",
    ),
    process: Spec::known(
        "180 nm",
        "Shen et al., 2016. This review did not confirm the foundry and does not name one.",
        Evidence::Measured,
    ),
    neurons_per_chip_stated: Spec::unlocated(CHIP_TOTAL_IS_A_PRODUCT),
    note: "A reminder that this field is not only American and European. Darwin and its successor \
           are the most-cited Chinese academic neuromorphic line and are routinely missing from \
           English-language comparison tables.",
};

/// Darwin3 — a 2024 large-scale part with on-chip learning, and a record with almost nothing in it.
///
/// Ma, Zhang, Shen et al., *Darwin3: A large-scale neuromorphic chip with a novel instruction set
/// and on-chip learning*, National Science Review, 2024.
///
/// This review located the part, its year, its on-chip learning and a **chip-level** neuron figure
/// of up to 2.35 million. It did not locate a core count, so it does not divide that figure into a
/// per-core one, and every per-core field here is therefore empty. That is the whole record, and a
/// record with one interesting number and eight empty fields is a truthful one.
pub const DARWIN3: Part = Part {
    name: "Darwin3",
    vendor: "Zhejiang University",
    citation: "Ma, Zhang, Shen et al., Darwin3: A large-scale neuromorphic chip with a novel \
               instruction set and on-chip learning, National Science Review, 2024.",
    year: Spec::known(
        2024,
        "Ma et al., National Science Review, 2024 - publication year. This review did not locate a \
         separate first-silicon date for the part.",
        Evidence::Measured,
    ),
    neurons_per_core: Spec::unlocated(
        "This review did not locate a core count for Darwin3 and therefore does not divide the \
         published chip total into a per-core figure.",
    ),
    cores_per_chip: Spec::unlocated("This review did not locate a core count for Darwin3."),
    synapses_per_core: Spec::unlocated("This review did not locate a per-core synapse figure."),
    max_fan_in: Spec::unlocated("This review did not locate a fan-in cap for Darwin3."),
    weight_bits: Spec::unlocated("This review did not locate a weight width for Darwin3."),
    delay_ticks: Spec::unlocated("This review did not locate a delay range for Darwin3."),
    on_chip_learning: Spec::known(
        true,
        "On-chip learning is in the paper's title. This review did not locate which rules the \
         instruction set implements, so the field says yes and claims nothing about scope.",
        Evidence::Measured,
    ),
    process: Spec::unlocated(
        "This review did not locate a process node for Darwin3 in a document it could confirm.",
    ),
    neurons_per_chip_stated: Spec::known(
        2_350_000,
        "Ma et al., National Science Review, 2024: up to 2.35 million neurons per chip. A CHIP \
         total; this review found no core count to divide it by.",
        Evidence::Measured,
    ),
    note: "One stated capacity, no structure. fits() will report nearly everything as unchecked for \
           this part, which is the correct report.",
};

/// Innatera Spiking Neural Processor T1 — a record that exists to say **nothing is public**.
///
/// What this review could establish: Innatera announced the T1 in 2024 as a mixed-signal spiking
/// processor for always-on sensor front ends, pairing an analogue spiking core with a `RISC-V`
/// processor and a conventional accelerator, and quoting energy and latency **ratios** against
/// conventional parts.
///
/// What it could not establish: neuron count, synapse count, fan-in cap, weight width, delay range,
/// on-chip learning, process node. Every structural field below is empty.
///
/// That is the finding. A comparison table that filled these in from a press release would be a
/// liability, and a comparison table that omitted the part would imply it does not exist. This
/// record does neither.
pub const INNATERA_T1: Part = Part {
    name: "Spiking Neural Processor T1",
    vendor: "Innatera",
    citation: "Innatera product announcement for the Spiking Neural Processor T1, 2024. A PRODUCT \
               ANNOUNCEMENT quoting energy and latency ratios, not a datasheet quoting structural \
               limits. This review did not locate a datasheet or a peer-reviewed paper.",
    year: Spec::known(2024, "Product announcement year.", Evidence::Measured),
    neurons_per_core: Spec::unlocated(
        "This review did not locate a neuron count for the T1 in any document it read.",
    ),
    cores_per_chip: Spec::unlocated("This review did not locate a core count for the T1."),
    synapses_per_core: Spec::unlocated("This review did not locate a synapse count for the T1."),
    max_fan_in: Spec::unlocated("This review did not locate a fan-in cap for the T1."),
    weight_bits: Spec::unlocated(
        "This review did not locate a weight width for the T1. The spiking core is described as \
         analogue, in which case the field may not have an integer answer at all.",
    ),
    delay_ticks: Spec::unlocated("This review did not locate a delay range for the T1."),
    on_chip_learning: Spec::unlocated(
        "This review did not locate a statement about on-chip plasticity in the T1, in either \
         direction.",
    ),
    process: Spec::unlocated("This review did not locate a process node for the T1."),
    neurons_per_chip_stated: Spec::unlocated("This review did not locate a device total for the T1."),
    note: "Nine empty structural fields and a year. fits() returns verdict: None for this part — \
           not a pass, and not a failure. Nothing was checkable.",
};

/// `BrainScaleS`-2 — analogue neurons running about **a thousand times faster than biology**.
///
/// Pehle, Billaudelle, Cramer et al., *The `BrainScaleS`-2 Accelerated Neuromorphic System With
/// Hybrid Plasticity*, Frontiers in Neuroscience 16:795876, 2022.
///
/// The part that breaks this table's assumptions, which is why it is in it. `BrainScaleS`-2 is
/// analogue and **accelerated**: its membrane dynamics run roughly 1,000x faster than the biological
/// system they model, so there is no tick, and the delay field is empty not because a figure is
/// missing but because the field is the wrong shape. A student who has understood why that field is
/// empty has understood the difference between a clocked digital part and an analogue one.
///
/// Its learning story is the other distinctive one: not a fixed rule in hardware but embedded 32-bit
/// `SIMD` plasticity processors that execute **arbitrary** learning rules over the analogue
/// observables. That generality is the paper's title.
pub const BRAINSCALES_2: Part = Part {
    name: "BrainScaleS-2",
    vendor: "Heidelberg University",
    citation: "Pehle, Billaudelle, Cramer et al., The BrainScaleS-2 Accelerated Neuromorphic System \
               With Hybrid Plasticity, Frontiers in Neuroscience 16:795876, 2022.",
    year: Spec::known(2022, "Frontiers in Neuroscience publication year for the fabricated part.", Evidence::Measured),
    neurons_per_core: Spec::known(
        512,
        "Pehle et al., Front. Neurosci. 16:795876, 2022: 512 adaptive-exponential (AdEx) neuron \
         circuits on the single-chip system. This table records the whole chip as one core because \
         this review did not confirm the internal partitioning.",
        Evidence::Measured,
    ),
    cores_per_chip: Spec::known(
        1,
        "One, a SCHEMA CHOICE by this review rather than a figure located or computed: this table \
         maps the whole chip to one core because this review did not confirm the die's internal \
         partitioning. See neurons_per_core. Graded Projected, because this review's own \
         convention is an inference about how to read the part and not arithmetic over stated \
         figures - and because a cores_per_chip of 1 makes the chip-total cross-check an identity, \
         which is why that test excludes this record from its independent count.",
        Evidence::Projected,
    ),
    synapses_per_core: Spec::known(
        131072,
        "Pehle et al., 2022: a 512 x 256 synapse array, 131,072 synapses.",
        Evidence::Measured,
    ),
    max_fan_in: Spec::known(
        256,
        "DERIVED by this review: 256 synapse rows per neuron column in the 512 x 256 array. Stated \
         as a row count in the paper; read as a fan-in cap here.",
        Evidence::Derived,
    ),
    weight_bits: Spec::known(
        6,
        "Pehle et al., 2022: 6-bit synaptic weights, with the sign carried by the row's \
         excitatory/inhibitory configuration rather than by a weight bit.",
        Evidence::Measured,
    ),
    delay_ticks: Spec::unlocated(
        "There is no tick. BrainScaleS-2 is analogue and runs at roughly 1,000x biological speed, \
         so its time base is continuous and accelerated. This field is the wrong SHAPE for this \
         part; it is left empty rather than converted, because converting it would invent a \
         timestep the part does not have.",
    ),
    on_chip_learning: Spec::known(
        true,
        "Hybrid plasticity: embedded 32-bit SIMD plasticity processors execute arbitrary learning \
         rules over the analogue observables. The generality is the paper's title.",
        Evidence::Measured,
    ),
    process: Spec::known(
        "65 nm CMOS",
        "Pehle et al., 2022. Mixed-signal: analogue neuron and synapse circuits with a digital \
         plasticity processor.",
        Evidence::Measured,
    ),
    neurons_per_chip_stated: Spec::known(
        512,
        "The single-chip system carries 512 neurons; larger BrainScaleS-2 systems are built by \
         composing chips. Pehle et al., 2022.",
        Evidence::Measured,
    ),
    note: "Accelerated time is the feature: an experiment that would take an hour in biology takes \
           about four seconds here, which is what makes plasticity studies tractable. Nothing in \
           this table's schema expresses that.",
};

/// Every part this review could source, oldest first.
///
/// Sixteen records. Not one of them is complete: [`Part::weakest_evidence`] returns
/// [`Evidence::Unstated`] for every single entry, because every part in the open literature has at
/// least one structural field nobody published. Sort by [`Part::stated_fields`] to see which parts
/// are actually documented — **[`TRUENORTH`], [`DYNAP_SE`] and [`BRAINSCALES_2`] at the top with
/// nine of ten**, [`ODIN`] next with eight, [`INNATERA_T1`] at the bottom with one. (Through
/// v0.4.0 this sentence named `ODIN` and `TrueNorth` as the top pair, which was simply wrong:
/// `ODIN` was never in the leading group and two other records tie `TrueNorth`. The census in
/// `the_completeness_ranking_is_the_one_the_table_doc_claims` is now the thing that has to agree
/// with this paragraph.)
///
/// Two entries are in the table as corrections rather than as spiking parts: [`NORTHPOLE`], which
/// has no spikes, and [`BRAINSCALES_2`], which has no tick.
pub const PARTS: [Part; 16] = [
    TRUENORTH,
    SPINNAKER,
    DARWIN,
    LOIHI,
    DYNAP_SE,
    ODIN,
    AKD1000,
    SPINNAKER2,
    LOIHI_2,
    SPECK,
    BRAINSCALES_2,
    XYLO_AUDIO_2,
    NORTHPOLE,
    AKD1500,
    DARWIN3,
    INNATERA_T1,
];

/// The part with this exact [`Part::name`], or `None`.
///
/// Case-sensitive and exact: a fuzzy lookup that matched `"loihi"` to `"Loihi 2"` would be a
/// silent way to quote the wrong chip's limits.
#[must_use]
pub fn part_named(name: &str) -> Option<&'static Part> {
    PARTS.iter().find(|p| p.name == name)
}

/// A constraint the network **violates**, with the numbers that violate it.
///
/// Never a bare "does not fit". Each variant carries what the network needs and what the part
/// allows, in the constraint's own units, plus how many other places the same constraint is
/// violated so that a caller knows whether it is looking at one bad neuron or at a systematic
/// mismatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bind {
    /// One neuron has more presynaptic sources than the part allows. **The wall.**
    FanIn {
        /// Index of the worst offending neuron. The highest in-degree, ties broken by lowest index.
        neuron: u32,
        /// That neuron's in-degree.
        fan_in: u64,
        /// The part's cap, from [`Part::max_fan_in`].
        cap: u64,
        /// How many neurons in total exceed the cap, this one included.
        offenders: usize,
    },
    /// A synapse's delay exceeds the part's longest deliverable delay.
    DelayTooLong {
        /// Index into [`crate::net::Net::delay`] of the worst offender.
        synapse: usize,
        /// Presynaptic neuron of that synapse.
        pre: u32,
        /// Postsynaptic neuron of that synapse.
        post: u32,
        /// That synapse's delay, ticks.
        delay: u32,
        /// The part's longest deliverable delay, ticks.
        cap: u32,
        /// How many synapses in total exceed it.
        offenders: usize,
    },
    /// A synapse's delay is shorter than the part's shortest deliverable delay.
    ///
    /// Usually `delay == 0` against a floor of 1: [`crate::net`] permits same-tick delivery and
    /// most hardware does not. Reported rather than rounded up, because rounding it up silently
    /// adds a tick of latency to every affected path.
    DelayTooShort {
        /// Index into [`crate::net::Net::delay`] of the worst offender.
        synapse: usize,
        /// Presynaptic neuron of that synapse.
        pre: u32,
        /// Postsynaptic neuron of that synapse.
        post: u32,
        /// That synapse's delay, ticks.
        delay: u32,
        /// The part's shortest deliverable delay, ticks.
        floor: u32,
        /// How many synapses in total fall below it.
        offenders: usize,
    },
    /// More neurons than one chip holds.
    Neurons {
        /// Neurons in the network.
        needed: u64,
        /// Neurons on one chip, from [`Part::neurons_per_chip`].
        cap: u64,
    },
    /// More synapses than one chip stores.
    Synapses {
        /// Synapses in the network, which is [`crate::net::Net::n_syn`] and **not** half of it.
        needed: u64,
        /// Synapses on one chip, from [`Part::synapses_per_chip`].
        cap: u64,
    },
    /// More cores than one chip has, by the lower bound in [`CoreCount`].
    ///
    /// Note which bound this is: a **lower** bound exceeding the chip's core count proves the
    /// network does not fit. The converse does not hold, and [`Fit`] says so.
    Cores {
        /// Cores the network needs at minimum.
        needed: u64,
        /// Cores on one chip.
        cap: u64,
    },
}

impl Bind {
    /// The constraint's name, as it appears in [`Fit::unchecked`] and [`Headroom::constraint`].
    #[must_use]
    pub fn constraint(&self) -> &'static str {
        match self {
            Self::FanIn { .. } => "maximum fan-in per neuron",
            Self::DelayTooLong { .. } => "longest synaptic delay",
            Self::DelayTooShort { .. } => "shortest synaptic delay",
            Self::Neurons { .. } => "neurons per chip",
            Self::Synapses { .. } => "synapses per chip",
            Self::Cores { .. } => "cores per chip",
        }
    }

    /// **By how much**, in the constraint's own units: the excess over a cap, or the shortfall
    /// below a floor.
    ///
    /// Always at least 1 for any `Bind` [`fits`] produces, because [`fits`] only produces one for a
    /// real violation and a violation by zero is not a violation. Saturating rather than
    /// wrapping, because `Bind`'s fields are `pub` and a caller may hand this method a value that
    /// is not a violation at all; that case answers 0 instead of panicking or wrapping to 1.8e19.
    /// Pinned variant by variant, with distinct numbers, in
    /// `every_bind_variant_reports_its_own_overflow_reach_and_precedence`.
    #[must_use]
    pub fn overflow(&self) -> u64 {
        match self {
            Self::FanIn { fan_in, cap, .. } => fan_in.saturating_sub(*cap),
            Self::DelayTooLong { delay, cap, .. } => u64::from(delay.saturating_sub(*cap)),
            Self::DelayTooShort { delay, floor, .. } => u64::from(floor.saturating_sub(*delay)),
            Self::Neurons { needed, cap } | Self::Synapses { needed, cap } | Self::Cores { needed, cap } => {
                needed.saturating_sub(*cap)
            }
        }
    }

    /// Whether buying more chips could relieve this constraint.
    ///
    /// **The distinction the module is written around.** `false` for [`Bind::FanIn`] and for both
    /// delay variants: fan-in is a property of one neuron's wiring and delay is a property of one
    /// synapse's field width, and neither is affected by how much silicon you own. `true` for the
    /// three capacity constraints, which are exactly the ones a bigger machine fixes.
    #[must_use]
    pub fn relieved_by_more_chips(&self) -> bool {
        match self {
            Self::FanIn { .. } | Self::DelayTooLong { .. } | Self::DelayTooShort { .. } => false,
            Self::Neurons { .. } | Self::Synapses { .. } | Self::Cores { .. } => true,
        }
    }

    /// Reporting precedence: 0 is the most severe.
    ///
    /// Fixed by KIND rather than by magnitude, and deliberately: a fan-in wall exceeded by one
    /// synapse is a harder problem than a neuron count exceeded by a factor of ten, because the
    /// second is solved by buying a second chip and the first is solved by redesigning the network.
    /// [`Bind::relieved_by_more_chips`] is the same fact as a boolean.
    #[must_use]
    pub fn precedence(&self) -> u8 {
        match self {
            Self::FanIn { .. } => 0,
            Self::DelayTooShort { .. } => 1,
            Self::DelayTooLong { .. } => 2,
            Self::Neurons { .. } => 3,
            Self::Synapses { .. } => 4,
            Self::Cores { .. } => 5,
        }
    }
}

impl fmt::Display for Bind {
    // ⛔ Every "over by" here is [`Bind::overflow`] and NOT a raw subtraction. `Bind` and its
    // fields are `pub`, so a caller can construct `Bind::FanIn { fan_in: 1, cap: 5, .. }` — a
    // nonsensical value, but a legal one — and `fan_in - cap` on it panicked in debug and printed
    // 18446744073709551612 in release. `overflow()` saturates, which is what the doc promised all
    // along, so the formatter now agrees with the method instead of duplicating it wrongly.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let over = self.overflow();
        match self {
            Self::FanIn { neuron, fan_in, cap, offenders } => write!(
                f,
                "fan-in: neuron {neuron} has {fan_in} presynaptic sources, cap is {cap} \
                 (over by {over}); {offenders} neuron(s) exceed it"
            ),
            Self::DelayTooLong { synapse, pre, post, delay, cap, offenders } => write!(
                f,
                "delay: synapse {synapse} ({pre} -> {post}) is {delay} ticks, longest deliverable \
                 is {cap}; {offenders} synapse(s) exceed it"
            ),
            Self::DelayTooShort { synapse, pre, post, delay, floor, offenders } => write!(
                f,
                "delay: synapse {synapse} ({pre} -> {post}) is {delay} ticks, shortest deliverable \
                 is {floor}; {offenders} synapse(s) fall below it"
            ),
            Self::Neurons { needed, cap } => {
                write!(f, "neurons: {needed} needed, {cap} per chip (over by {over})")
            }
            Self::Synapses { needed, cap } => {
                write!(f, "synapses: {needed} needed, {cap} per chip (over by {over})")
            }
            Self::Cores { needed, cap } => {
                write!(f, "cores: at least {needed} needed, {cap} per chip (over by {over})")
            }
        }
    }
}

/// A constraint the network satisfies, and by how much.
///
/// Reported alongside the violations because "it fits" with 99.6% of the synapse memory used is a
/// different engineering position from "it fits" with 4% used, and a boolean loses that.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Headroom {
    /// The constraint's name, matching [`Bind::constraint`].
    pub constraint: &'static str,
    /// What the network uses, in the constraint's units.
    pub used: u64,
    /// What the part allows.
    pub cap: u64,
}

impl Headroom {
    /// `used / cap`, in `[0, 1]` for a satisfied constraint.
    ///
    /// `None` when `cap == 0`, where the ratio has no value — and a part with a cap of zero is a
    /// part that holds nothing, which is a different report from "fully utilised".
    #[must_use]
    pub fn utilisation(&self) -> Option<f64> {
        if self.cap == 0 {
            return None;
        }
        Some(self.used as f64 / self.cap as f64)
    }

    /// What is left, in the constraint's units.
    #[must_use]
    pub fn spare(&self) -> u64 {
        self.cap.saturating_sub(self.used)
    }
}

/// How many cores a network needs — **a bin-packing lower bound**, except where it says otherwise.
///
/// # Why a bound and not an answer
///
/// Assigning neurons to cores subject to a neuron cap and a synapse cap is bin packing, which is
/// NP-hard, so an exact minimum is not something this function computes in general. What it
/// computes instead:
///
/// * `by_neurons` and `by_synapses` are two independent **lower** bounds, each a ceiling division.
/// * `lower_bound` is the larger of them. No assignment uses fewer cores than this.
/// * `greedy` runs first-fit-decreasing by in-degree, which produces a valid assignment and is
///   therefore an **upper** bound. The true minimum lies in `lower_bound ..= greedy`.
/// * `exact` is `true` in the one case where the bound is provably attained, described below.
///
/// # When the bound is exact
///
/// If `max_in_degree * neurons_per_core <= synapses_per_core`, then any `neurons_per_core` neurons
/// packed together always fit their synapses too, so filling cores in any order achieves
/// `by_neurons` — which is already a lower bound, so it is the minimum. That is the common case for
/// a sparse network on a crossbar part, and it is the case `the_core_count_for_a_uniform_network_is_exactly_the_analytic_answer`
/// pins against the closed form `ceil(n / neurons_per_core)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CoreCount {
    /// `ceil(neurons / neurons_per_core)`. A lower bound.
    pub by_neurons: usize,
    /// `ceil(synapses / synapses_per_core)`, or `None` when the part states no synapse capacity.
    /// A lower bound where it exists.
    pub by_synapses: Option<usize>,
    /// The larger of the two bounds above: the number below which no assignment can go.
    pub lower_bound: usize,
    /// Cores used by first-fit-decreasing on in-degree — a valid assignment, hence an **upper**
    /// bound.
    ///
    /// `None` when the part states no synapse capacity (nothing to pack against), or when one
    /// neuron's fan-in alone exceeds a whole core's synapse capacity, in which case **no**
    /// assignment exists and the network must be restructured rather than repacked.
    pub greedy: Option<usize>,
    /// Whether `lower_bound` is provably the exact minimum. See the type doc for the condition.
    pub exact: bool,
}

impl CoreCount {
    /// Chips needed at minimum, given a part's cores per chip.
    ///
    /// Ceiling division of [`CoreCount::lower_bound`], so it inherits the same "lower bound, not
    /// an answer" caveat — and one more: a network split across chips pays inter-chip routing that
    /// this crate does not model at all.
    #[must_use]
    pub fn chips(&self, cores_per_chip: u32) -> Option<u64> {
        if cores_per_chip == 0 {
            return None;
        }
        Some((self.lower_bound as u64).div_ceil(u64::from(cores_per_chip)))
    }
}

/// The result of asking whether a network maps onto a part.
///
/// Read [`Fit::unchecked`] before [`Fit::verdict`]. A `Some(true)` verdict means "no constraint
/// this part **states** was violated", which on a part like [`INNATERA_T1`] or [`DARWIN3`] is a
/// very weak claim — and on those parts the verdict is `None` instead, because nothing was
/// checkable at all.
///
/// Everything here is for **one chip**. [`CoreCount::chips`] gives the multi-chip lower bound, and
/// [`Bind::relieved_by_more_chips`] says which of the reported violations a bigger machine would
/// actually fix.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fit {
    /// The part this was checked against, by [`Part::name`].
    pub part: &'static str,
    /// `Some(false)` if any constraint binds, `Some(true)` if at least one was checked and none
    /// bound, and **`None` if nothing was checkable**.
    ///
    /// `None` is not a pass. It is the answer for a part whose limits are not public.
    pub verdict: Option<bool>,
    /// Constraints violated, most severe first by [`Bind::precedence`]. Empty when none are.
    pub binds: Vec<Bind>,
    /// Constraints checked and satisfied, with their margins, in the order they were checked.
    pub headroom: Vec<Headroom>,
    /// Constraints this part does not state, by the same names [`Bind::constraint`] uses.
    ///
    /// The list a caller has to read before believing a `Some(true)`.
    ///
    /// Every entry really is one of the six [`Bind::constraint`] strings, and
    /// `every_unchecked_name_is_a_bind_constraint_name` asserts it over the whole table so that a
    /// caller can correlate the two lists by string. One `Spec` covers two of those names: a part
    /// with no [`Part::delay_ticks`] contributes **both** `"shortest synaptic delay"` and
    /// `"longest synaptic delay"`, because neither end of the range is known. Through v0.4.0 this
    /// list instead carried a seventh string, `"synaptic delay range"`, which
    /// [`Bind::constraint`] never returns — so a caller correlating by name silently dropped the
    /// delay entry.
    pub unchecked: Vec<&'static str>,
    /// Core allocation, where the part states enough to compute one.
    pub cores: Option<CoreCount>,
    /// Chips needed at minimum, from [`CoreCount::chips`].
    pub chips_lower_bound: Option<u64>,
}

impl Fit {
    /// The most severe violated constraint, or `None` when nothing binds.
    ///
    /// "Most severe" is by [`Bind::precedence`], which ranks the constraints more silicon cannot
    /// fix above the ones it can — not by magnitude. See that method's doc.
    #[must_use]
    pub fn binding(&self) -> Option<&Bind> {
        self.binds.first()
    }

    /// Whether every violated constraint would be relieved by a larger machine.
    ///
    /// `true` with a non-empty [`Fit::binds`] means "buy more chips"; `false` means "change the
    /// network". `false` for an empty `binds` too, because there is nothing to relieve.
    #[must_use]
    pub fn scales_out(&self) -> bool {
        !self.binds.is_empty() && self.binds.iter().all(Bind::relieved_by_more_chips)
    }
}

impl fmt::Display for Fit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let verdict = match self.verdict {
            Some(true) => "FITS (on the constraints this part states)",
            Some(false) => "DOES NOT FIT",
            None => "NO VERDICT: this part states no constraint that could be checked",
        };
        writeln!(f, "{} — {verdict}", self.part)?;
        for b in &self.binds {
            writeln!(f, "  BINDS  {b}")?;
        }
        for h in &self.headroom {
            // A cap of zero has no utilisation, and printing -1.0% for it — as this did through
            // v0.4.0 — is a number a reader will believe. `Headroom::utilisation` returns None
            // there and this says so.
            match h.utilisation() {
                Some(u) => writeln!(
                    f,
                    "  ok     {}: {} of {} ({:.1}%)",
                    h.constraint,
                    h.used,
                    h.cap,
                    u * 100.0
                )?,
                None => writeln!(
                    f,
                    "  ok     {}: {} of {} (no utilisation: this part's cap is zero)",
                    h.constraint, h.used, h.cap
                )?,
            }
        }
        for u in &self.unchecked {
            writeln!(f, "  ?      {u}: not stated for this part")?;
        }
        if let Some(c) = self.cores {
            let how = if c.exact { "exactly" } else { "at least" };
            write!(f, "  cores  {how} {}", c.lower_bound)?;
            if let Some(g) = c.greedy {
                write!(f, ", first-fit-decreasing uses {g}")?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

/// Can this network be mapped onto one of these parts? Reports **which constraint binds and by how
/// much**, never a bare boolean.
///
/// Checks **six** constraints, reported in the precedence order [`Bind::precedence`] gives:
/// per-neuron fan-in, the shortest and the longest deliverable synaptic delay, neurons per chip,
/// synapses per chip, and cores per chip against [`CoreCount::lower_bound`]. The two delay ends
/// come from one [`Spec`] and are counted separately because they are separate constraints with
/// separate [`Bind`]s. Every constraint the part does not state lands in [`Fit::unchecked`]
/// instead of being assumed satisfied.
///
/// ```
/// use ferromorphic::hardware::{fits, TRUENORTH};
/// use ferromorphic::net::NetBuilder;
///
/// // One neuron with 300 inputs, on a part whose crossbar column is 256 rows.
/// let mut b = NetBuilder::new(400);
/// for pre in 0..300u32 {
///     b.connect(pre, 399, 1e-3, 1)?;
/// }
/// let net = b.build();
///
/// let fit = fits(&net, &TRUENORTH);
/// assert_eq!(fit.verdict, Some(false));
/// // And it names the neuron and the numbers, not just "no".
/// assert_eq!(fit.binding().unwrap().overflow(), 300 - 256);
/// // More chips would not help: fan-in is a property of one neuron.
/// assert!(!fit.scales_out());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[must_use]
pub fn fits(net: &Net, part: &Part) -> Fit {
    let mut binds: Vec<Bind> = Vec::new();
    let mut headroom: Vec<Headroom> = Vec::new();
    let mut unchecked: Vec<&'static str> = Vec::new();

    let deg = net.in_degrees();
    let max_deg = deg.iter().copied().max().unwrap_or(0);

    // --- fan-in, the wall ---
    if let Some(cap) = part.max_fan_in.value {
        let cap_u = u64::from(cap);
        let mut worst: Option<(u32, u64)> = None;
        let mut offenders = 0usize;
        for (i, &d) in deg.iter().enumerate() {
            let d = d as u64;
            if d > cap_u {
                offenders += 1;
                if worst.is_none_or(|(_, wd)| d > wd) {
                    worst = Some((i as u32, d));
                }
            }
        }
        match worst {
            Some((neuron, fan_in)) => binds.push(Bind::FanIn { neuron, fan_in, cap: cap_u, offenders }),
            None => headroom.push(Headroom {
                constraint: "maximum fan-in per neuron",
                used: max_deg as u64,
                cap: cap_u,
            }),
        }
    } else {
        unchecked.push("maximum fan-in per neuron");
    }

    // --- delay range ---
    if let Some(range) = part.delay_ticks.value {
        let mut long: Option<(usize, u32, u32, u32)> = None;
        let mut short: Option<(usize, u32, u32, u32)> = None;
        let mut n_long = 0usize;
        let mut n_short = 0usize;
        let mut max_seen = 0u32;
        for pre in 0..net.n {
            let (a, b) = (net.offset[pre], net.offset[pre + 1]);
            for k in a..b {
                let d = net.delay[k];
                max_seen = max_seen.max(d);
                if d > range.max_ticks {
                    n_long += 1;
                    if long.is_none_or(|(_, _, _, wd)| d > wd) {
                        long = Some((k, pre as u32, net.post[k], d));
                    }
                } else if d < range.min_ticks {
                    n_short += 1;
                    if short.is_none_or(|(_, _, _, wd)| d < wd) {
                        short = Some((k, pre as u32, net.post[k], d));
                    }
                }
            }
        }
        // Reported in the order the scan above finds them — ceiling first, as the loop body tests
        // it — and NOT pre-arranged into precedence order. `binds.sort_by_key(Bind::precedence)`
        // below is what puts them in order, and that is deliberate: while every push here happened
        // to be in precedence order already, the sort was a no-op that no test could distinguish
        // from a deleted one, and "most severe first" was a promise kept by accident of layout.
        // Now the promise is kept by the sort, and
        // `the_binds_are_reported_most_severe_first` fails if the sort goes away.
        if let Some((synapse, pre, post, delay)) = long {
            binds.push(Bind::DelayTooLong {
                synapse,
                pre,
                post,
                delay,
                cap: range.max_ticks,
                offenders: n_long,
            });
        }
        if let Some((synapse, pre, post, delay)) = short {
            binds.push(Bind::DelayTooShort {
                synapse,
                pre,
                post,
                delay,
                floor: range.min_ticks,
                offenders: n_short,
            });
        }
        if short.is_none() && long.is_none() {
            headroom.push(Headroom {
                constraint: "longest synaptic delay",
                used: u64::from(max_seen),
                cap: u64::from(range.max_ticks),
            });
        }
    } else {
        // Two names, not one: the Spec is a range and covers both ends, and every string in
        // `unchecked` has to be one `Bind::constraint()` returns or a caller correlating the two
        // lists loses the entry silently.
        unchecked.push("shortest synaptic delay");
        unchecked.push("longest synaptic delay");
    }

    // --- neurons per chip ---
    match part.neurons_per_chip() {
        Some(cap) => {
            let needed = net.n as u64;
            if needed > cap {
                binds.push(Bind::Neurons { needed, cap });
            } else {
                headroom.push(Headroom { constraint: "neurons per chip", used: needed, cap });
            }
        }
        None => unchecked.push("neurons per chip"),
    }

    // --- synapses per chip ---
    match part.synapses_per_chip() {
        Some(cap) => {
            let needed = net.n_syn as u64;
            if needed > cap {
                binds.push(Bind::Synapses { needed, cap });
            } else {
                headroom.push(Headroom { constraint: "synapses per chip", used: needed, cap });
            }
        }
        None => unchecked.push("synapses per chip"),
    }

    // --- cores ---
    let cores = core_count(net, part);
    let mut chips_lower_bound = None;
    match (cores, part.cores_per_chip.value) {
        (Some(c), Some(per_chip)) => {
            chips_lower_bound = c.chips(per_chip);
            let needed = c.lower_bound as u64;
            let cap = u64::from(per_chip);
            if needed > cap {
                binds.push(Bind::Cores { needed, cap });
            } else {
                headroom.push(Headroom { constraint: "cores per chip", used: needed, cap });
            }
        }
        _ => unchecked.push("cores per chip"),
    }

    // Load-bearing, not cosmetic: `Fit::binding()` returns `binds.first()`, so this line is what
    // makes "the most severe violated constraint" true. The delay pair above is pushed in scan
    // order rather than precedence order precisely so that this sort has an observable job.
    binds.sort_by_key(Bind::precedence);
    let verdict = if !binds.is_empty() {
        Some(false)
    } else if headroom.is_empty() {
        None
    } else {
        Some(true)
    };

    Fit { part: part.name, verdict, binds, headroom, unchecked, cores, chips_lower_bound }
}

/// Cores needed for `net` on `part`: a lower bound, plus a greedy upper bound. See [`CoreCount`].
///
/// `None` when the part does not state `neurons_per_core`, without which there is no bound to
/// compute. A part that states neurons but not synapses per core still gets a `CoreCount`, with
/// `by_synapses` and `greedy` empty.
///
/// # Zero capacities are refused, not substituted
///
/// `None` also when a stated capacity is **zero** — `neurons_per_core == Some(0)` or
/// `synapses_per_core == Some(0)`. A core that holds nothing admits no assignment of any network
/// with anything in it, and there is no bound to report. Through v0.4.0 a zero synapse capacity
/// instead produced `by_synapses = usize::MAX`, which propagated into `lower_bound` and out of
/// [`CoreCount::chips`] as a nonsense chip count with no indication that anything had gone wrong.
/// It was the one place in this module where an unreadable input got a substituted value rather
/// than a refusal. No part in [`PARTS`] states a zero capacity; `Part` is `pub` with `pub` fields,
/// so a caller can.
#[must_use]
pub fn core_count(net: &Net, part: &Part) -> Option<CoreCount> {
    let npc = part.neurons_per_core.value?;
    if npc == 0 {
        return None;
    }
    let npc_u = npc as usize;
    let by_neurons = net.n.div_ceil(npc_u);
    let spc = part.synapses_per_core.value;
    if spc == Some(0) {
        return None;
    }
    let by_synapses = spc.map(|s| (net.n_syn as u64).div_ceil(s) as usize);
    let lower_bound = by_synapses.map_or(by_neurons, |s| by_neurons.max(s));

    let deg = net.in_degrees();
    let max_deg = deg.iter().copied().max().unwrap_or(0) as u64;
    // The provably-tight case: any `npc` neurons together always fit their synapses, so filling
    // cores in any order attains the neuron bound, which is already a lower bound.
    let exact = spc.is_some_and(|s| max_deg.saturating_mul(u64::from(npc)) <= s);

    let greedy = spc.and_then(|s| first_fit_decreasing(&deg, npc_u, s));

    Some(CoreCount { by_neurons, by_synapses, lower_bound, greedy, exact })
}

/// First-fit-decreasing by in-degree: a valid assignment, hence an upper bound on the core count.
///
/// `None` when one neuron's fan-in alone exceeds a core's synapse capacity, in which case no
/// assignment exists at all — the network has to be restructured, not repacked, and returning a
/// large number instead would hide that.
///
/// Cost is `O(neurons * cores)`. That is fine for the sizes this crate is asked about and is stated
/// here rather than discovered by someone packing a million neurons.
fn first_fit_decreasing(deg: &[usize], neurons_per_core: usize, synapses_per_core: u64) -> Option<usize> {
    let mut order: Vec<usize> = (0..deg.len()).collect();
    // Descending in-degree, ties by ascending index: deterministic, which the crate requires.
    order.sort_unstable_by(|&a, &b| deg[b].cmp(&deg[a]).then(a.cmp(&b)));
    let mut cores: Vec<(usize, u64)> = Vec::new();
    for &i in &order {
        let d = deg[i] as u64;
        if d > synapses_per_core {
            return None;
        }
        let mut placed = false;
        for c in &mut cores {
            if c.0 < neurons_per_core && c.1 + d <= synapses_per_core {
                c.0 += 1;
                c.1 += d;
                placed = true;
                break;
            }
        }
        if !placed {
            cores.push((1, d));
        }
    }
    Some(cores.len())
}

/// Why a hardware question could not be answered.
///
/// Each variant names the offending value, which is the difference between an error a user can act
/// on and one they have to bisect for.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HardwareError {
    /// A weight was not a finite number.
    ///
    /// Rejected at the boundary for the reason [`crate::net::NetError::NonFiniteWeight`] gives: a
    /// NaN does not fail loudly, it propagates into a scale, then into every code, then into a
    /// round-trip error that reports as NaN long after the cause is out of scope.
    NonFiniteWeight {
        /// Position in the slice.
        index: usize,
        /// The offending value, reproduced so the caller does not have to go and look.
        value: f64,
    },
    /// The quantiser's scale was not finite.
    NonFiniteScale {
        /// The offending value.
        value: f64,
    },
    /// There is no step to quantise with: either every weight is zero, or the full-scale magnitude
    /// divided by the largest code **underflowed to zero**.
    ///
    /// Refused rather than answered with a step of zero, in both cases and for the same reason. A
    /// vector of zeros round-trips exactly at any step, so a "successful" quantisation there would
    /// report a perfect result for a network that has no weights — almost always a bug upstream
    /// and never a useful answer. The underflow case is worse: the weights are real, the step is
    /// zero, and the round-trip statistics report a near-perfect result for a vector every element
    /// of which was destroyed. See [`Quantiser::symmetric`] for the worked case.
    NoScale,
    /// An empty weight slice. There is nothing to quantise and no scale to derive.
    NoWeights,
    /// A bit-width a symmetric signed quantiser cannot use.
    ///
    /// Below 2: a 1-bit signed code has exactly one non-negative level and cannot carry a
    /// magnitude. [`TRUENORTH`]'s 1-bit crossbar is real and is **not** a weight quantiser in this
    /// sense — see that part's doc. Above 31: the code would not fit an `i32`.
    BadBits {
        /// The width asked for.
        bits: u32,
    },
    /// The part does not state the figure this operation needs.
    ///
    /// The honest outcome for [`LOIHI_2`] and [`SPINNAKER2`], whose weight widths this review could
    /// not source. Not a defect in either the part or the code.
    UnstatedSpec {
        /// [`Part::name`].
        part: &'static str,
        /// The field, by its Rust name, e.g. `"weight_bits"`.
        field: &'static str,
    },
}

impl fmt::Display for HardwareError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFiniteWeight { index, value } => {
                write!(f, "weight {index} is {value}, which is not finite")
            }
            Self::NonFiniteScale { value } => write!(f, "scale {value} is not finite"),
            Self::NoScale => f.write_str(
                "no usable step: either every weight is zero, or the full scale divided by the \
                 largest code underflowed to zero. A step of zero reports a near-perfect round \
                 trip for weights it has destroyed",
            ),
            Self::NoWeights => f.write_str("no weights: nothing to quantise and no scale to derive"),
            Self::BadBits { bits } => write!(
                f,
                "{bits} bits: a symmetric signed quantiser needs 2 to 31 (1 signed bit carries no \
                 magnitude; above 31 the code leaves i32)"
            ),
            Self::UnstatedSpec { part, field } => write!(
                f,
                "{part} states no {field}: this review did not locate one, so there is no value to \
                 compute with"
            ),
        }
    }
}

impl std::error::Error for HardwareError {}

/// How a real weight is mapped onto the nearest integer code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rounding {
    /// Round to the nearest level, ties away from zero.
    ///
    /// Error is at most **half a least significant bit**, which is the tightest possible
    /// deterministic bound — and it is **biased**: a weight that always sits 0.3 of a step above a
    /// level always rounds down, by 0.3 of a step, every time it is quantised. Over a layer whose
    /// weights share a systematic offset, those errors add instead of cancelling.
    Nearest,
    /// Round up with probability equal to the fractional part, down otherwise.
    ///
    /// **Unbiased**: the expected dequantised value equals the original exactly, which is why it is
    /// the standard choice for quantised training and for accumulating small updates into a
    /// low-precision weight. What it costs is variance. For a weight a fraction `f` of a step above
    /// a level, its RMS error is `step * sqrt(f * (1 - f))` against round-to-nearest's
    /// `step * min(f, 1 - f)`, and `sqrt(f * (1 - f)) >= min(f, 1 - f)` for every `f` — so it errs
    /// by more **in expectation**, with equality only where `f` is 0 or 1/2. Both closed forms are
    /// checked in this module, and neither is a promise about one draw: on a short vector the
    /// ordering inverts often enough to measure (about 13% of 8-weight vectors at 4 bits).
    ///
    /// Uses [`crate::rng::Rng`], so it is deterministic under a seed: same seed, same codes, every
    /// platform.
    Stochastic,
}

/// A symmetric signed uniform quantiser onto a part's weight bit-width.
///
/// Codes run from `-(2^(bits-1) - 1)` to `+(2^(bits-1) - 1)`. The most negative two's-complement
/// code is **left unused**, which is the standard symmetric choice: it keeps the mapping odd
/// (`quantise(-w) == -quantise(w)`) at the cost of one level out of `2^bits`. A part whose hardware
/// uses that level represents slightly more range than this quantiser will produce, and this
/// module does not claim otherwise.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quantiser {
    /// Bit-width including sign, 2 to 31.
    pub bits: u32,
    /// Volts per code. The least significant bit, in the weights' own units.
    pub step: f64,
    /// Largest code, `2^(bits-1) - 1`. The smallest is its negation.
    pub max_code: i32,
}

impl Quantiser {
    /// A quantiser whose full-scale magnitude is `max_abs`.
    ///
    /// # Errors
    ///
    /// [`HardwareError::BadBits`] outside 2..=31, [`HardwareError::NonFiniteScale`] for a
    /// non-finite `max_abs`, and [`HardwareError::NoScale`] when `max_abs <= 0` **or when the
    /// quotient `max_abs / max_code` is not itself a positive normal-or-subnormal number**.
    ///
    /// ⛔ The second half of that last clause is not hypothetical. `max_abs` positive does not make
    /// the step positive: `symmetric(31, 1e-320)` divides a subnormal by 1,073,741,823 and the
    /// quotient **underflows to exactly zero**, which is the state [`HardwareError::NoScale`]'s
    /// doc promises is impossible. A quantiser with `step == 0` then divides every weight by zero,
    /// so a zero weight becomes `0.0/0.0 = NaN` and every other weight becomes an infinity that
    /// clamps to a code limit; `values()` comes back all zeros while `max_abs_error` reports
    /// 1e-320 and `mean_error` reports 0 — every weight destroyed, reported as a near-perfect
    /// round trip. It is checked here, after the division, because that is the only place the
    /// underflow exists.
    pub fn symmetric(bits: u32, max_abs: f64) -> Result<Self, HardwareError> {
        if !(2..=31).contains(&bits) {
            return Err(HardwareError::BadBits { bits });
        }
        if !max_abs.is_finite() {
            return Err(HardwareError::NonFiniteScale { value: max_abs });
        }
        if !(max_abs > 0.0) {
            return Err(HardwareError::NoScale);
        }
        let max_code = (1i32 << (bits - 1)) - 1;
        let step = max_abs / f64::from(max_code);
        if !(step > 0.0) || !step.is_finite() {
            return Err(HardwareError::NoScale);
        }
        Ok(Self { bits, step, max_code })
    }

    /// A quantiser scaled to the largest magnitude present in `w`.
    ///
    /// The usual choice, and the one under which the half-least-significant-bit bound holds for
    /// every element: nothing clips, because nothing is outside the range the scale was taken
    /// from.
    ///
    /// # Errors
    ///
    /// [`HardwareError::NoWeights`] for an empty slice, [`HardwareError::NonFiniteWeight`] naming
    /// the first non-finite element, plus the errors of [`Quantiser::symmetric`].
    pub fn from_weights(bits: u32, w: &[f64]) -> Result<Self, HardwareError> {
        let max_abs = max_abs_of(w)?;
        Self::symmetric(bits, max_abs)
    }

    /// A quantiser at `part`'s weight bit-width, scaled to `w`.
    ///
    /// # Errors
    ///
    /// [`HardwareError::UnstatedSpec`] when the part states no weight width — the correct outcome
    /// for [`LOIHI_2`], [`SPINNAKER2`], [`SPECK`], [`DARWIN`], [`DARWIN3`] and [`INNATERA_T1`] —
    /// plus the errors of [`Quantiser::from_weights`].
    pub fn for_part(part: &Part, w: &[f64]) -> Result<Self, HardwareError> {
        let bits = part
            .weight_bits
            .value
            .ok_or(HardwareError::UnstatedSpec { part: part.name, field: "weight_bits" })?;
        Self::from_weights(bits, w)
    }

    /// Half a least significant bit: the tightest bound round-to-nearest can achieve, and the one
    /// `round_to_nearest_never_errs_by_more_than_half_an_lsb` checks over a sweep.
    #[must_use]
    pub fn half_lsb(&self) -> f64 {
        self.step / 2.0
    }

    /// Round `w` to the nearest code, ties away from zero.
    ///
    /// # Errors
    ///
    /// [`HardwareError::NoWeights`] for an empty slice, [`HardwareError::NonFiniteWeight`] naming
    /// the first non-finite element.
    pub fn quantise_nearest(&self, w: &[f64]) -> Result<Quantised, HardwareError> {
        self.quantise_with(w, Rounding::Nearest, None)
    }

    /// Round `w` stochastically, using `rng`.
    ///
    /// Unbiased in expectation; see [`Rounding::Stochastic`] for what that buys and costs.
    ///
    /// # Errors
    ///
    /// As [`Quantiser::quantise_nearest`].
    pub fn quantise_stochastic(&self, w: &[f64], rng: &mut Rng) -> Result<Quantised, HardwareError> {
        self.quantise_with(w, Rounding::Stochastic, Some(rng))
    }

    fn quantise_with(
        &self,
        w: &[f64],
        mode: Rounding,
        mut rng: Option<&mut Rng>,
    ) -> Result<Quantised, HardwareError> {
        if w.is_empty() {
            return Err(HardwareError::NoWeights);
        }
        for (i, &v) in w.iter().enumerate() {
            if !v.is_finite() {
                return Err(HardwareError::NonFiniteWeight { index: i, value: v });
            }
        }
        let lo = f64::from(-self.max_code);
        let hi = f64::from(self.max_code);
        let mut codes = Vec::with_capacity(w.len());
        let mut clipped = 0usize;
        let mut max_abs_error = 0.0f64;
        let mut sum_error = 0.0f64;
        let mut sum_sq = 0.0f64;
        for &v in w {
            let x = v / self.step;
            let raw = match mode {
                Rounding::Nearest => x.round(),
                Rounding::Stochastic => {
                    let floor = x.floor();
                    let frac = x - floor;
                    // P(round up) == frac, so E[code] == floor + frac == x exactly.
                    let u = rng.as_deref_mut().map_or(0.0, Rng::next_f64);
                    if u < frac { floor + 1.0 } else { floor }
                }
            };
            let clamped = raw.clamp(lo, hi);
            if clamped != raw {
                clipped += 1;
            }
            let code = clamped as i32;
            codes.push(code);
            let err = f64::from(code) * self.step - v;
            max_abs_error = max_abs_error.max(err.abs());
            sum_error += err;
            sum_sq += err * err;
        }
        let n = w.len() as f64;
        Ok(Quantised {
            codes,
            step: self.step,
            bits: self.bits,
            mode,
            clipped,
            max_abs_error,
            mean_error: sum_error / n,
            rms_error: (sum_sq / n).sqrt(),
        })
    }
}

/// The largest magnitude in `w`, rejecting non-finite entries by index.
fn max_abs_of(w: &[f64]) -> Result<f64, HardwareError> {
    if w.is_empty() {
        return Err(HardwareError::NoWeights);
    }
    let mut m = 0.0f64;
    for (i, &v) in w.iter().enumerate() {
        if !v.is_finite() {
            return Err(HardwareError::NonFiniteWeight { index: i, value: v });
        }
        m = m.max(v.abs());
    }
    Ok(m)
}

/// A quantised weight vector and **what it cost**.
///
/// The error fields are the point. A quantisation that reports only codes lets a caller deploy a
/// network without ever seeing how far the weights moved; these four numbers are computed in the
/// same pass and are free.
#[derive(Debug, Clone, PartialEq)]
pub struct Quantised {
    /// One integer code per input weight, in `-(max_code)..=max_code`.
    pub codes: Vec<i32>,
    /// Volts per code, copied from the quantiser so this record is self-contained.
    pub step: f64,
    /// Bit-width including sign.
    pub bits: u32,
    /// Which rounding produced these codes.
    pub mode: Rounding,
    /// How many weights hit a code limit and were clamped.
    ///
    /// **Non-zero breaks the half-least-significant-bit bound**, because a clipped weight's error
    /// is unbounded by the step. Zero for any quantiser built by [`Quantiser::from_weights`], which
    /// is why that constructor is the default advice.
    pub clipped: usize,
    /// Largest `|dequantised - original|` over the vector, in the weights' units.
    pub max_abs_error: f64,
    /// Mean **signed** error. The bias.
    ///
    /// Signed on purpose: a mean of zero is the property stochastic rounding has and
    /// round-to-nearest does not, and taking the absolute value first would hide exactly that.
    pub mean_error: f64,
    /// Root-mean-square error. The magnitude, as opposed to the bias.
    ///
    /// Larger for [`Rounding::Stochastic`] than for [`Rounding::Nearest`] on the same data **in
    /// expectation, and only in expectation**: the closed forms are `step * sqrt(f * (1 - f))`
    /// against `step * min(f, 1 - f)` for a weight a fraction `f` of a step above a level, and the
    /// first dominates the second for every `f`. A *realisation* of stochastic rounding can be
    /// better, and on short vectors often is — a sweep of 500 eight-weight vectors at 4 bits
    /// inverted the ordering 65 times. The trade is asserted at both scales:
    /// `the_stochastic_rms_error_matches_its_closed_form` pins the expectation and
    /// `stochastic_rounding_costs_variance_for_its_lack_of_bias` pins a vector long enough for the
    /// expectation to bite.
    pub rms_error: f64,
}

impl Quantised {
    /// The codes back in the weights' units: `code * step`, exactly.
    #[must_use]
    pub fn values(&self) -> Vec<f64> {
        self.codes.iter().map(|&c| f64::from(c) * self.step).collect()
    }

    /// Half a least significant bit, the bound [`Rounding::Nearest`] respects when nothing clipped.
    #[must_use]
    pub fn half_lsb(&self) -> f64 {
        self.step / 2.0
    }

    /// Whether the round-trip respected the half-least-significant-bit bound.
    ///
    /// `tol` is a relative slack for floating-point comparison; `1e-12` is ample. Returns `false`
    /// whenever anything clipped, because the bound genuinely does not hold there.
    ///
    /// **Both halves are load-bearing and both are pinned.** `clipped == 0` alone would certify a
    /// vector whose error is unbounded, and it is refused by the edge case in
    /// `a_clipped_weight_breaks_the_half_lsb_bound_and_says_so`; the numeric bound alone would
    /// certify a [`Rounding::Stochastic`] vector that missed by 0.7 of a step, and it is refused by
    /// `an_unclipped_vector_past_the_bound_is_not_certified` — which also uses an error inside
    /// `1.5 * half_lsb`, so widening the bound does not rescue it.
    #[must_use]
    pub fn within_half_lsb(&self, tol: f64) -> bool {
        self.clipped == 0 && self.max_abs_error <= self.half_lsb() * (1.0 + tol)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        AKD1000, AKD1500, BRAINSCALES_2, Bind, CoreCount, DARWIN, DARWIN3, DYNAP_SE, DelayRange,
        Evidence, Fit, HardwareError, Headroom, INNATERA_T1, LOIHI, LOIHI_2, NORTHPOLE, ODIN,
        PARTS, Part, Quantiser, Rounding, SPECK, SPINNAKER, SPINNAKER2, Spec, TRUENORTH,
        XYLO_AUDIO_2, core_count, fits, part_named,
    };
    use crate::net::{Net, NetBuilder};
    use crate::rng::Rng;

    /// The ten graded fields of a record, flattened so a test can sweep them without naming each
    /// one ten times.
    fn graded(p: &Part) -> Vec<(&'static str, bool, &'static str, Evidence)> {
        vec![
            ("year", p.year.is_known(), p.year.source, p.year.evidence),
            (
                "neurons_per_core",
                p.neurons_per_core.is_known(),
                p.neurons_per_core.source,
                p.neurons_per_core.evidence,
            ),
            (
                "cores_per_chip",
                p.cores_per_chip.is_known(),
                p.cores_per_chip.source,
                p.cores_per_chip.evidence,
            ),
            (
                "synapses_per_core",
                p.synapses_per_core.is_known(),
                p.synapses_per_core.source,
                p.synapses_per_core.evidence,
            ),
            ("max_fan_in", p.max_fan_in.is_known(), p.max_fan_in.source, p.max_fan_in.evidence),
            ("weight_bits", p.weight_bits.is_known(), p.weight_bits.source, p.weight_bits.evidence),
            ("delay_ticks", p.delay_ticks.is_known(), p.delay_ticks.source, p.delay_ticks.evidence),
            (
                "on_chip_learning",
                p.on_chip_learning.is_known(),
                p.on_chip_learning.source,
                p.on_chip_learning.evidence,
            ),
            ("process", p.process.is_known(), p.process.source, p.process.evidence),
            (
                "neurons_per_chip_stated",
                p.neurons_per_chip_stated.is_known(),
                p.neurons_per_chip_stated.source,
                p.neurons_per_chip_stated.evidence,
            ),
        ]
    }

    /// A network of `n` neurons where every neuron has exactly `fan_in` presynaptic sources, taken
    /// from its predecessors modulo `n`. Uniform by construction, which is what makes the core
    /// count analytically exact.
    fn uniform_net(n: u32, fan_in: u32, delay: u32) -> Net {
        let mut b = NetBuilder::new(n as usize);
        for post in 0..n {
            for k in 1..=fan_in {
                let pre = (post + n - (k % n)) % n;
                b.connect(pre, post, 1e-3, delay).expect("indices in range");
            }
        }
        b.build()
    }

    // ---------------------------------------------------------------------------------------
    // (a) Round-to-nearest errs by at most half a least significant bit.
    // ---------------------------------------------------------------------------------------

    /// The closed form: `code = round(w/step)` gives `|code - w/step| <= 1/2`, so the round-trip
    /// error is at most `step/2`. Swept over every usable bit-width and several random weight
    /// vectors, because a bound that holds at 8 bits and fails at 3 is a bound nobody checked.
    #[test]
    fn round_to_nearest_never_errs_by_more_than_half_an_lsb() {
        for bits in 2..=12u32 {
            for seed in 0..4u64 {
                let mut rng = Rng::new(seed * 97 + u64::from(bits));
                let amp = 0.001 + rng.next_f64() * 10.0;
                let w: Vec<f64> = (0..500).map(|_| (rng.next_f64() * 2.0 - 1.0) * amp).collect();
                let q = Quantiser::from_weights(bits, &w).expect("finite, non-zero weights");
                let out = q.quantise_nearest(&w).expect("finite weights");
                assert_eq!(out.clipped, 0, "scaling from the data must not clip");
                assert!(
                    out.max_abs_error <= q.half_lsb() * (1.0 + 1e-9),
                    "{bits} bits seed {seed}: max error {} exceeds half an LSB {}",
                    out.max_abs_error,
                    q.half_lsb()
                );
                assert!(out.within_half_lsb(1e-9));
                // And the reported error really is the round trip, not a separate accumulator.
                let back = out.values();
                let direct =
                    w.iter().zip(&back).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
                assert!((direct - out.max_abs_error).abs() < 1e-15, "{direct} vs {}", out.max_abs_error);
            }
        }
    }

    /// The bound is TIGHT, not slack: a weight exactly on a half-step attains it exactly. With
    /// `step == 1` by construction, the error is exactly `0.5` and is compared with `==`.
    #[test]
    fn the_half_lsb_bound_is_attained_exactly_at_a_tie() {
        // max_abs 7.0 at 4 bits gives max_code 7 and step exactly 1.0.
        let q = Quantiser::symmetric(4, 7.0).expect("4 bits, positive scale");
        assert_eq!(q.max_code, 7);
        assert_eq!(q.step, 1.0);
        let w = [0.5, 1.5, 2.5, -0.5, -1.5];
        let out = q.quantise_nearest(&w).expect("finite");
        assert_eq!(out.codes, vec![1, 2, 3, -1, -2], "ties must round away from zero");
        assert_eq!(out.max_abs_error, 0.5);
        assert_eq!(out.max_abs_error, q.half_lsb());
    }

    /// Clipping breaks the bound, which is exactly why `clipped` is reported and why
    /// `within_half_lsb` returns false whenever it is non-zero.
    #[test]
    fn a_clipped_weight_breaks_the_half_lsb_bound_and_says_so() {
        let q = Quantiser::symmetric(4, 1.0).expect("valid"); // step 1/7, codes -7..=7
        let w = [0.1, 5.0, -0.2];
        let out = q.quantise_nearest(&w).expect("finite");
        assert_eq!(out.clipped, 1);
        assert!(out.max_abs_error > q.half_lsb() * 10.0, "{}", out.max_abs_error);
        assert!(!out.within_half_lsb(1e-12));

        // ⛔ The discriminating case, and the reason `clipped` GATES the bound rather than sitting
        // beside it. A weight exactly half a step past the top code clips, and its round-trip
        // error is exactly half an LSB — so the numeric bound HOLDS on a vector where it certifies
        // nothing at all, because the next weight further out would break it without limit.
        let edge = Quantiser::symmetric(4, 7.0).expect("step exactly 1.0");
        let out2 = edge.quantise_nearest(&[7.5]).expect("finite");
        assert_eq!(out2.clipped, 1, "7.5 rounds to 8 and is clamped to 7");
        assert_eq!(out2.codes, vec![7]);
        assert_eq!(out2.max_abs_error, 0.5);
        assert_eq!(out2.max_abs_error, edge.half_lsb(), "the bound holds numerically");
        assert!(
            !out2.within_half_lsb(1e-12),
            "a clipped vector must not certify the half-LSB bound even when it numerically meets it"
        );
    }

    /// ⛔ The other half of [`Quantised::within_half_lsb`], and the half nothing asserted. A
    /// vector where **nothing clipped** and the error is past half an LSB must not certify.
    ///
    /// Without this, `self.clipped == 0 && self.max_abs_error <= self.half_lsb() * (1.0 + tol)`
    /// could be cut down to `self.clipped == 0` — the entire numeric bound deleted from the
    /// module's flagship certification predicate — and all 38 tests stayed green. The mirror
    /// mutation (deleting `clipped == 0`) was caught by the edge case in
    /// `a_clipped_weight_breaks_the_half_lsb_bound_and_says_so`; this is the complement.
    ///
    /// The error is chosen at 0.7 of a step deliberately: past half an LSB, and still inside
    /// **1.5** half-LSBs, so widening the bound by half does not rescue it either.
    #[test]
    fn an_unclipped_vector_past_the_bound_is_not_certified() {
        let q = Quantiser::symmetric(4, 7.0).expect("step exactly 1.0");
        assert_eq!(q.step, 1.0);
        let w = vec![2.3f64; 64];
        let st = q.quantise_stochastic(&w, &mut Rng::new(4)).expect("finite");

        assert_eq!(st.clipped, 0, "2.3 is far inside -7..=7, so nothing clips");
        assert!(st.codes.contains(&3), "at least one weight rounded up, which is the 0.7 error");
        assert!((st.max_abs_error - 0.7).abs() < 1e-12, "{}", st.max_abs_error);
        assert!(st.max_abs_error > st.half_lsb(), "the bound IS broken: 0.7 > 0.5");
        assert!(
            !st.within_half_lsb(1e-12),
            "an unclipped vector whose error exceeds half an LSB must not certify the bound"
        );
        assert!(
            st.max_abs_error < st.half_lsb() * 1.5,
            "and it must not certify under a bound loosened by half, either: {} vs {}",
            st.max_abs_error,
            st.half_lsb() * 1.5
        );
        // The predicate is a conjunction, so the round-to-nearest vector it does certify has to be
        // right here beside it — otherwise this test would pass against a method returning false.
        let near = q.quantise_nearest(&w).expect("finite");
        assert_eq!(near.clipped, 0);
        assert!(near.max_abs_error <= near.half_lsb());
        assert!(near.within_half_lsb(1e-12), "round-to-nearest respects it and must still certify");
    }

    // ---------------------------------------------------------------------------------------
    // (b) Stochastic rounding is unbiased; round-to-nearest is not.
    // ---------------------------------------------------------------------------------------

    /// `step == 1` and every weight at `2.3`, so the fractional part is 0.3 and the closed forms
    /// are exact:
    ///
    /// * round-to-nearest gives code 2 EVERY time, so its mean error is exactly `2 - 2.3 = -0.3`;
    /// * stochastic rounding gives code 3 with probability 0.3, so `E[code] = 2.3` and the mean
    ///   error is zero, with a standard error of `sqrt(0.3*0.7/N)`.
    ///
    /// Both halves are asserted. A test that checked only the second would pass for a
    /// round-to-nearest that happened to be centred.
    #[test]
    fn stochastic_rounding_is_unbiased_and_round_to_nearest_is_not() {
        const N: usize = 20_000;
        let q = Quantiser::symmetric(4, 7.0).expect("valid"); // step exactly 1.0
        let w = vec![2.3f64; N];

        let near = q.quantise_nearest(&w).expect("finite");
        assert!(near.codes.iter().all(|&c| c == 2));
        assert!(
            (near.mean_error - (2.0 - 2.3)).abs() < 1e-12,
            "round-to-nearest bias {} is not the closed form -0.3",
            near.mean_error
        );

        // sd of the mean is sqrt(p(1-p)/N) = sqrt(0.21/20000) = 0.00324. Six of those is 0.019.
        let sd_mean = (0.3f64 * 0.7 / N as f64).sqrt();
        let mut worst = 0.0f64;
        for seed in 0..64u64 {
            let mut rng = Rng::new(seed);
            let st = q.quantise_stochastic(&w, &mut rng).expect("finite");
            worst = worst.max(st.mean_error.abs());
            assert!(
                st.mean_error.abs() < 6.0 * sd_mean,
                "seed {seed}: stochastic bias {} exceeds six standard errors {}",
                st.mean_error,
                6.0 * sd_mean
            );
        }
        assert!(
            worst < near.mean_error.abs() / 10.0,
            "the worst stochastic bias {worst} is not an order below round-to-nearest's {}",
            near.mean_error.abs()
        );
    }

    /// The sharper form of the same claim: the COUNT rounded up is binomial with `p = 0.3`, so it
    /// must land within a few standard deviations of `0.3 * N`. This checks the distribution, not
    /// just its first moment.
    #[test]
    fn the_stochastic_round_up_count_is_binomial_at_the_fractional_part() {
        const N: usize = 40_000;
        let q = Quantiser::symmetric(4, 7.0).expect("valid");
        let w = vec![2.3f64; N];
        let sd = (N as f64 * 0.3 * 0.7).sqrt(); // 91.7
        for seed in 0..8u64 {
            let mut rng = Rng::new(1000 + seed);
            let st = q.quantise_stochastic(&w, &mut rng).expect("finite");
            let up = st.codes.iter().filter(|&&c| c == 3).count();
            let down = st.codes.iter().filter(|&&c| c == 2).count();
            assert_eq!(up + down, N, "stochastic rounding produced a code outside {{2,3}}");
            let want = 0.3 * N as f64;
            assert!(
                (up as f64 - want).abs() < 5.0 * sd,
                "seed {seed}: {up} rounded up, expected {want} +/- {sd}"
            );
        }
    }

    /// Unbiased is not free. Stochastic rounding trades bias for variance, and on the same data its
    /// RMS error is larger than round-to-nearest's. Stating the cost beside the benefit.
    #[test]
    fn stochastic_rounding_costs_variance_for_its_lack_of_bias() {
        let mut rng = Rng::new(4);
        let w: Vec<f64> = (0..5_000).map(|_| rng.next_f64() * 2.0 - 1.0).collect();
        let q = Quantiser::from_weights(6, &w).expect("valid");
        let near = q.quantise_nearest(&w).expect("finite");
        let mut r2 = Rng::new(11);
        let st = q.quantise_stochastic(&w, &mut r2).expect("finite");
        assert!(
            st.rms_error > near.rms_error,
            "stochastic RMS {} did not exceed nearest's {}",
            st.rms_error,
            near.rms_error
        );
        // And the max error can exceed half an LSB, which round-to-nearest's never does.
        assert!(near.within_half_lsb(1e-9));
        assert!(st.max_abs_error > q.half_lsb(), "{}", st.max_abs_error);
        // Which means this vector must NOT certify the bound — nothing clipped, and the error is
        // past it. `clipped == 0` alone would have said it did.
        assert_eq!(st.clipped, 0, "scaling from the data does not clip");
        assert!(!st.within_half_lsb(1e-12), "max error {} vs {}", st.max_abs_error, st.half_lsb());
    }

    /// ⛔ The RMS trade, against its **closed form** rather than one empirical seed pair.
    ///
    /// For a weight sitting a fraction `f` of a step above a level, the signed error of
    /// round-to-nearest is the same every time, so its RMS error is exactly `step * min(f, 1-f)`.
    /// Stochastic rounding takes `(1-f)` steps up with probability `f` and `f` steps down
    /// otherwise, so `E[e^2] = f(1-f)^2 + (1-f)f^2 = f(1-f)` in units of `step^2` and its RMS
    /// error is exactly `step * sqrt(f(1-f))`.
    ///
    /// `sqrt(f(1-f)) >= min(f, 1-f)` for every `f`, which is the whole of the claim that
    /// stochastic rounding errs by more — **in expectation**. The module used to state that as a
    /// property of the data and test it with a single pair of seeds.
    #[test]
    fn the_stochastic_rms_error_matches_its_closed_form() {
        const N: usize = 40_000;
        let q = Quantiser::symmetric(4, 7.0).expect("step exactly 1.0");
        assert_eq!(q.step, 1.0);

        for f in [0.3f64, 0.1, 0.25] {
            let w = vec![2.0 + f; N];
            let near = q.quantise_nearest(&w).expect("finite");
            let want_near = f.min(1.0 - f);
            assert!(
                (near.rms_error - want_near).abs() < 1e-12,
                "f={f}: nearest RMS {} is not the closed form {want_near}",
                near.rms_error
            );

            // E[e^2] = f(1-f) exactly. Var[e^2] = f(1-f)^4 + (1-f)f^4 - (f(1-f))^2, so the mean of
            // N of them has that standard error; six of those is the band.
            let want_sq = f * (1.0 - f);
            let var_sq =
                f * (1.0 - f).powi(4) + (1.0 - f) * f.powi(4) - want_sq * want_sq;
            let se = (var_sq / N as f64).sqrt();
            for seed in 0..6u64 {
                let st = q.quantise_stochastic(&w, &mut Rng::new(seed * 31 + 5)).expect("finite");
                let got_sq = st.rms_error * st.rms_error;
                assert!(
                    (got_sq - want_sq).abs() < 6.0 * se,
                    "f={f} seed {seed}: mean square error {got_sq} is not f(1-f)={want_sq} \
                     within six standard errors {}",
                    6.0 * se
                );
                assert!(
                    st.rms_error > near.rms_error,
                    "f={f} seed {seed}: sqrt(f(1-f))={} should exceed min(f,1-f)={want_near}",
                    st.rms_error
                );
            }
        }

        // ⛔ The boundary of the claim, and the reason it is stated as an expectation. At f = 1/2
        // the two closed forms COINCIDE — `sqrt(1/4) == min(1/2, 1/2)` — and stochastic rounding's
        // error is +-step/2 on every draw, so its RMS error is EXACTLY round-to-nearest's, on
        // every seed, with no sampling in it at all. "Larger" is false here and the doc says so.
        let half = vec![2.5f64; 512];
        let near_half = q.quantise_nearest(&half).expect("finite");
        assert_eq!(near_half.rms_error, 0.5);
        for seed in 0..8u64 {
            let st = q.quantise_stochastic(&half, &mut Rng::new(seed)).expect("finite");
            assert_eq!(st.rms_error, near_half.rms_error, "seed {seed}: f=1/2 is the equality case");
            // The bias, which is what actually separates them here.
            assert_eq!(near_half.mean_error, 0.5, "ties away from zero, every time");
            assert!(st.mean_error.abs() < 0.1, "seed {seed}: {}", st.mean_error);
        }
    }

    /// ⛔ And the inequality is about the **expectation**, not about any one vector: on short
    /// vectors a lucky draw inverts it often. This sweep finds the inversions rather than avoiding
    /// them, so the doc cannot go back to stating the ordering as a property of the data.
    #[test]
    fn on_short_vectors_the_rms_ordering_inverts_often_enough_to_measure() {
        let mut inverted = 0usize;
        let trials = 500;
        for seed in 0..trials as u64 {
            let mut rng = Rng::new(seed);
            let w: Vec<f64> = (0..8).map(|_| rng.next_f64() * 2.0 - 1.0).collect();
            let Ok(q) = Quantiser::from_weights(4, &w) else { continue };
            let near = q.quantise_nearest(&w).expect("finite");
            let st = q.quantise_stochastic(&w, &mut Rng::new(seed + 9_000)).expect("finite");
            if st.rms_error <= near.rms_error {
                inverted += 1;
            }
        }
        assert!(
            inverted > trials / 50,
            "only {inverted} of {trials} short vectors inverted the RMS ordering; if that ever \
             goes to zero, the 'in expectation' wording is the thing to re-examine, not this test"
        );
        assert!(
            inverted < trials / 2,
            "{inverted} of {trials} inverted: the expectation should still dominate"
        );
    }

    #[test]
    fn stochastic_quantisation_is_deterministic_under_a_seed() {
        let w: Vec<f64> = (0..200).map(|i| f64::from(i) * 0.011 - 1.0).collect();
        let q = Quantiser::from_weights(5, &w).expect("valid");
        let a = q.quantise_stochastic(&w, &mut Rng::new(7)).expect("finite");
        let b = q.quantise_stochastic(&w, &mut Rng::new(7)).expect("finite");
        assert_eq!(a.codes, b.codes);
        let c = q.quantise_stochastic(&w, &mut Rng::new(8)).expect("finite");
        assert_ne!(a.codes, c.codes, "two seeds gave identical codes, which is not randomness");
    }

    #[test]
    fn dequantisation_is_exactly_code_times_step() {
        let w = [0.9, -0.4, 0.0, 0.25];
        let q = Quantiser::from_weights(8, &w).expect("valid");
        let out = q.quantise_nearest(&w).expect("finite");
        for (i, v) in out.values().iter().enumerate() {
            assert_eq!(*v, f64::from(out.codes[i]) * out.step);
        }
        assert_eq!(out.mode, Rounding::Nearest);
    }

    // ---------------------------------------------------------------------------------------
    // Refusals: no guessing, no NaN, no invented specs.
    // ---------------------------------------------------------------------------------------

    /// A NaN weight is refused at the boundary, naming its index — not folded into a scale where
    /// it would poison every code and report the damage as a NaN round-trip error.
    #[test]
    fn a_non_finite_weight_is_refused_with_its_index() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let w = [0.1, 0.2, bad, 0.4];
            match Quantiser::from_weights(8, &w) {
                Err(HardwareError::NonFiniteWeight { index, value }) => {
                    assert_eq!(index, 2);
                    assert!(!value.is_finite());
                }
                other => panic!("expected a refusal naming index 2, got {other:?}"),
            }
            // And the quantiser itself refuses too, so a caller who built the scale elsewhere
            // cannot slip one past.
            let q = Quantiser::symmetric(8, 1.0).expect("valid");
            match q.quantise_nearest(&w) {
                Err(HardwareError::NonFiniteWeight { index, .. }) => assert_eq!(index, 2),
                other => panic!("expected a refusal naming index 2, got {other:?}"),
            }
        }
    }

    #[test]
    fn a_one_bit_signed_quantiser_is_refused_with_its_reason() {
        assert_eq!(Quantiser::symmetric(1, 1.0), Err(HardwareError::BadBits { bits: 1 }));
        assert_eq!(Quantiser::symmetric(0, 1.0), Err(HardwareError::BadBits { bits: 0 }));
        assert_eq!(Quantiser::symmetric(32, 1.0), Err(HardwareError::BadBits { bits: 32 }));
        // And the reason is in the message rather than only in the variant.
        let msg = HardwareError::BadBits { bits: 1 }.to_string();
        assert!(msg.contains("magnitude"), "{msg}");
    }

    #[test]
    fn an_all_zero_weight_vector_has_no_scale_rather_than_a_perfect_score() {
        assert_eq!(Quantiser::from_weights(8, &[0.0; 16]), Err(HardwareError::NoScale));
        assert_eq!(Quantiser::symmetric(8, 0.0), Err(HardwareError::NoScale));
        assert_eq!(Quantiser::from_weights(8, &[]), Err(HardwareError::NoWeights));
    }

    #[test]
    fn a_non_finite_scale_is_refused() {
        assert!(matches!(
            Quantiser::symmetric(8, f64::INFINITY),
            Err(HardwareError::NonFiniteScale { .. })
        ));
        assert!(matches!(
            Quantiser::symmetric(8, f64::NAN),
            Err(HardwareError::NonFiniteScale { .. })
        ));
    }

    /// The part states no weight width, so there is nothing to quantise to. An error naming the
    /// part and the field, not a default of 8.
    #[test]
    fn quantising_for_a_part_with_no_stated_weight_width_is_an_error_naming_it() {
        let w = [0.1, -0.2, 0.3];
        for p in [&LOIHI_2, &SPINNAKER2, &SPECK, &DARWIN, &DARWIN3, &INNATERA_T1] {
            assert_eq!(
                Quantiser::for_part(p, &w),
                Err(HardwareError::UnstatedSpec { part: p.name, field: "weight_bits" }),
                "{} should refuse: this review did not source its weight width",
                p.name
            );
        }
        // And where the width IS stated, it is used.
        let q = Quantiser::for_part(&AKD1000, &w).expect("Akida states 4 bits");
        assert_eq!(q.bits, 4);
        assert_eq!(q.max_code, 7);
        let odin = Quantiser::for_part(&ODIN, &w).expect("ODIN states 4 bits");
        assert_eq!(odin.bits, 4);
    }

    /// `TrueNorth`'s crossbar really is one bit, and a one-bit SIGNED quantiser is refused — which
    /// is the correct outcome, because that bit is connectivity and the magnitude comes from a
    /// per-neuron indirection. See the part's doc.
    #[test]
    fn truenorths_single_crossbar_bit_is_not_a_weight_quantiser() {
        assert_eq!(TRUENORTH.weight_bits.value, Some(1));
        assert_eq!(
            Quantiser::for_part(&TRUENORTH, &[0.1, -0.2]),
            Err(HardwareError::BadBits { bits: 1 })
        );
    }

    // ---------------------------------------------------------------------------------------
    // (c) A network exceeding a fan-in cap is refused with the offending neuron and the numbers.
    // ---------------------------------------------------------------------------------------

    /// ⛔ The offender reported is the WORST one, not the first one seen, and this fixture is built
    /// so the two differ. Neuron 7 has 100 inputs and neuron 9 has 300, so a scan that kept the
    /// first neuron past the cap would answer 7 and a scan that keeps the highest in-degree
    /// answers 9. The fixture used to be the other way round — 300 at index 7, 100 at index 9 —
    /// under which "first" and "worst" are the same neuron, `worst.is_none_or(|(_, wd)| d > wd)`
    /// could be cut down to `worst.is_none()`, and every test stayed green.
    #[test]
    fn a_fan_in_cap_is_refused_with_the_worst_offending_neuron_and_the_numbers() {
        let mut b = NetBuilder::new(400);
        for pre in 0..100u32 {
            b.connect(pre, 7, 1e-3, 1).expect("in range");
        }
        for pre in 0..300u32 {
            b.connect(pre, 9, 1e-3, 1).expect("in range");
        }
        let net = b.build();

        let fit = fits(&net, &TRUENORTH);
        assert_eq!(fit.verdict, Some(false));
        match fit.binding().expect("something binds") {
            Bind::FanIn { neuron, fan_in, cap, offenders } => {
                assert_eq!(*neuron, 9, "the WORST offender, not the first one scanned");
                assert_eq!(*fan_in, 300);
                assert_eq!(*cap, 256);
                assert_eq!(*offenders, 1, "only neuron 9 exceeds TrueNorth's 256");
            }
            other => panic!("expected a fan-in bind, got {other:?}"),
        }
        assert_eq!(fit.binding().expect("bind").overflow(), 44);
        assert!(!fit.scales_out(), "fan-in is not relieved by more chips");
        assert!(fit.to_string().contains("neuron 9"), "{fit}");

        // The same network against the tightest cap in the table: both neurons now offend, and the
        // one reported is still the worse of the two and still not the first.
        let tight = fits(&net, &DYNAP_SE);
        match tight.binding().expect("binds") {
            Bind::FanIn { neuron, fan_in, cap, offenders } => {
                assert_eq!((*neuron, *fan_in, *cap, *offenders), (9, 300, 64, 2));
            }
            other => panic!("expected a fan-in bind, got {other:?}"),
        }
    }

    /// The other half of `Bind::FanIn::neuron`'s doc: "ties broken by lowest index". Two neurons
    /// with the same in-degree over the cap, and the lower index is the one reported — which is
    /// what makes the report deterministic rather than an artefact of the scan.
    #[test]
    fn equal_worst_fan_ins_are_reported_at_the_lowest_index() {
        let mut b = NetBuilder::new(400);
        for post in [11u32, 3, 300] {
            for pre in 0..300u32 {
                b.connect(pre, post, 1e-3, 1).expect("in range");
            }
        }
        let net = b.build();
        match fits(&net, &TRUENORTH).binding().expect("binds") {
            Bind::FanIn { neuron, fan_in, offenders, .. } => {
                assert_eq!(*neuron, 3, "three neurons tie at 300; the lowest index is reported");
                assert_eq!(*fan_in, 300);
                assert_eq!(*offenders, 3);
            }
            other => panic!("expected a fan-in bind, got {other:?}"),
        }
    }

    /// A fan-in wall is not relieved by scale, and this pins it: shrink the network to almost
    /// nothing and the bind survives, because it is a property of one neuron.
    #[test]
    fn a_fan_in_wall_survives_an_otherwise_trivial_network() {
        let mut b = NetBuilder::new(300);
        for pre in 0..257u32 {
            b.connect(pre, 299, 1e-3, 1).expect("in range");
        }
        let net = b.build();
        let fit = fits(&net, &TRUENORTH);
        assert_eq!(fit.verdict, Some(false));
        assert_eq!(fit.binding().expect("binds").overflow(), 1, "over by exactly one synapse");
        assert!(!fit.binding().expect("binds").relieved_by_more_chips());
        // Everything else has enormous headroom, and the report says both things at once.
        assert!(fit.headroom.iter().any(|h| h.constraint == "neurons per chip"));
    }

    #[test]
    fn a_network_inside_every_stated_cap_fits_and_reports_its_headroom() {
        let net = uniform_net(100, 5, 1);
        let fit = fits(&net, &XYLO_AUDIO_2);
        assert_eq!(fit.verdict, Some(true));
        assert!(fit.binds.is_empty());
        // Xylo states neurons, synapses and cores; it states no fan-in and no delay range.
        assert_eq!(fit.headroom.len(), 3, "{fit}");
        // Three unchecked, not two: an unstated delay range leaves BOTH ends of it unknown, and
        // each end is named with the string `Bind::constraint` uses for it.
        assert_eq!(fit.unchecked.len(), 3, "{fit}");
        assert!(fit.unchecked.contains(&"maximum fan-in per neuron"));
        assert!(fit.unchecked.contains(&"shortest synaptic delay"));
        assert!(fit.unchecked.contains(&"longest synaptic delay"));
        assert!(
            !fit.unchecked.contains(&"synaptic delay range"),
            "a name no Bind::constraint() returns is uncorrelatable: {fit}"
        );
        let neurons = fit
            .headroom
            .iter()
            .find(|h| h.constraint == "neurons per chip")
            .expect("checked");
        assert_eq!(neurons.used, 100);
        assert_eq!(neurons.cap, 1000);
        assert!((neurons.utilisation().expect("non-zero cap") - 0.1).abs() < 1e-12);
        assert_eq!(neurons.spare(), 900);
    }

    /// A delay of zero is same-tick delivery, which `SpiNNaker` cannot do. Reported rather than
    /// rounded up, because rounding it up adds a tick of latency to every affected path.
    #[test]
    fn a_same_tick_network_is_refused_by_a_part_with_a_delay_floor() {
        let net = uniform_net(10, 2, 0);
        let fit = fits(&net, &SPINNAKER);
        assert_eq!(fit.verdict, Some(false));
        match fit.binding().expect("binds") {
            Bind::DelayTooShort { delay, floor, offenders, .. } => {
                assert_eq!(*delay, 0);
                assert_eq!(*floor, 1);
                assert_eq!(*offenders, 20, "every synapse is same-tick");
            }
            other => panic!("expected a delay floor bind, got {other:?}"),
        }
        assert!(!fit.scales_out());
        // Move every delay inside the range and the same network fits.
        let ok = uniform_net(10, 2, 3);
        assert_eq!(fits(&ok, &SPINNAKER).verdict, Some(true));
        // Past the ceiling and it binds at the other end.
        let long = uniform_net(10, 2, 40);
        assert!(matches!(
            fits(&long, &SPINNAKER).binding().expect("binds"),
            Bind::DelayTooLong { delay: 40, cap: 16, .. }
        ));
    }

    /// A capacity overflow IS relieved by more chips, and the report distinguishes it from a wall.
    #[test]
    fn a_capacity_overflow_scales_out_and_says_so() {
        let net = uniform_net(2000, 2, 1);
        let fit = fits(&net, &XYLO_AUDIO_2); // 1,000 neurons on one core, one core
        assert_eq!(fit.verdict, Some(false));
        assert!(fit.scales_out(), "a neuron-count overflow is exactly what more chips fix");
        assert!(fit.binds.iter().any(|b| matches!(b, Bind::Neurons { needed: 2000, cap: 1000 })));
        assert!(fit.binding().expect("binds").relieved_by_more_chips());
        assert_eq!(fit.chips_lower_bound, Some(2));
    }

    /// ⛔ The honesty case. A part whose limits are not public gets NO VERDICT — not a pass.
    #[test]
    fn a_part_with_no_public_limits_has_no_verdict_rather_than_a_pass() {
        let net = uniform_net(50, 3, 1);
        let fit = fits(&net, &INNATERA_T1);
        assert_eq!(fit.verdict, None, "Innatera's record states nothing checkable");
        assert!(fit.binds.is_empty());
        assert!(fit.headroom.is_empty());
        assert_eq!(fit.unchecked.len(), 6, "{fit}");
        // All six constraint names, because Innatera states none of them.
        for name in [
            "maximum fan-in per neuron",
            "shortest synaptic delay",
            "longest synaptic delay",
            "neurons per chip",
            "synapses per chip",
            "cores per chip",
        ] {
            assert!(fit.unchecked.contains(&name), "{name} missing from {fit}");
        }
        assert!(fit.cores.is_none());
        assert!(fit.to_string().contains("NO VERDICT"), "{fit}");
    }

    /// ⛔ The weaker honesty case, and the reason [`Fit::unchecked`] has to be read first. Darwin3
    /// publishes ONE figure — a chip-level neuron total — so a 50-neuron network gets a
    /// `Some(true)` verdict off a single satisfied constraint with five unchecked beside it. That
    /// verdict is true and nearly worthless, and the report says both things.
    #[test]
    fn a_verdict_from_one_checkable_constraint_carries_its_five_unchecked_ones() {
        let net = uniform_net(50, 3, 1);
        let fit = fits(&net, &DARWIN3);
        assert_eq!(fit.verdict, Some(true));
        assert_eq!(fit.headroom.len(), 1, "{fit}");
        assert_eq!(fit.headroom[0].constraint, "neurons per chip");
        assert_eq!(fit.headroom[0].cap, 2_350_000);
        assert_eq!(fit.unchecked.len(), 5, "{fit}");
        assert!(fit.cores.is_none(), "no core count without neurons per core");
        assert!(fit.chips_lower_bound.is_none());
    }

    /// `NorthPole` has no neurons, so the question does not typecheck and the report says so
    /// rather than dividing by a quantity that does not exist.
    #[test]
    fn northpole_answers_a_spiking_question_with_unchecked_constraints() {
        let net = uniform_net(50, 3, 1);
        let fit = fits(&net, &NORTHPOLE);
        assert_eq!(fit.verdict, None);
        assert!(fit.unchecked.contains(&"neurons per chip"));
        assert!(NORTHPOLE.neurons_per_core.source.contains("NOT APPLICABLE"));
        assert!(NORTHPOLE.delay_ticks.source.contains("NOT APPLICABLE"));
    }

    // ---------------------------------------------------------------------------------------
    // (d) The core count for a trivially-packable network is exactly the analytic answer.
    // ---------------------------------------------------------------------------------------

    /// With every in-degree equal to 4 and a `TrueNorth` core holding 256 neurons and 65,536
    /// synapses, `4 * 256 = 1024 <= 65536`, so the synapse cap never binds and the minimum is
    /// exactly `ceil(3000 / 256) = 12`. The closed form is computed in the test rather than
    /// hard-coded, and the greedy packing must attain it.
    #[test]
    fn the_core_count_for_a_uniform_network_is_exactly_the_analytic_answer() {
        let net = uniform_net(3000, 4, 1);
        let c = core_count(&net, &TRUENORTH).expect("TrueNorth states neurons per core");
        let closed_form = 3000usize.div_ceil(256);
        assert_eq!(closed_form, 12);
        assert!(c.exact, "4 * 256 <= 65536, so the neuron bound is provably tight");
        assert_eq!(c.lower_bound, closed_form);
        assert_eq!(c.by_neurons, closed_form);
        assert_eq!(c.by_synapses, Some(1));
        assert_eq!(c.greedy, Some(closed_form), "first-fit-decreasing must attain a tight bound");
        assert_eq!(c.chips(4096), Some(1));
    }

    /// Where the synapse cap binds instead, the bound is not claimed to be exact and the greedy
    /// packing is allowed to be worse. Both statements are asserted so neither can drift.
    #[test]
    fn a_synapse_bound_network_reports_a_bound_rather_than_an_answer() {
        // ODIN: 256 neurons and 65,536 synapses per core. 256 neurons at 512 inputs each would
        // need 131,072 synapses, so the synapse cap binds first.
        let net = uniform_net(600, 512, 1);
        let c = core_count(&net, &ODIN).expect("ODIN states both");
        assert!(!c.exact, "512 * 256 > 65536, so the neuron bound is not provably tight");
        assert_eq!(c.by_neurons, 600usize.div_ceil(256));
        assert_eq!(c.by_synapses, Some((600u64 * 512).div_ceil(65536) as usize));
        assert_eq!(c.lower_bound, c.by_neurons.max(c.by_synapses.expect("stated")));
        let greedy = c.greedy.expect("a packing exists");
        assert!(greedy >= c.lower_bound, "greedy {greedy} beat the lower bound {}", c.lower_bound);
    }

    /// The invariant that makes the pair meaningful: a valid assignment can never use fewer cores
    /// than a lower bound. Swept over random irregular networks, which is where a sloppy bound
    /// would break.
    #[test]
    fn the_greedy_packing_never_beats_the_lower_bound() {
        for seed in 0..12u64 {
            let mut rng = Rng::new(seed);
            let n = 200 + rng.below(600) as usize;
            let mut b = NetBuilder::new(n);
            for post in 0..n as u32 {
                let k = rng.below(40);
                for _ in 0..k {
                    let pre = rng.below(n as u32);
                    b.connect(pre, post, 1e-3, 1).expect("in range");
                }
            }
            let net = b.build();
            for part in [&TRUENORTH, &ODIN, &DYNAP_SE, &XYLO_AUDIO_2] {
                let Some(c) = core_count(&net, part) else { continue };
                let Some(g) = c.greedy else { continue };
                assert!(
                    g >= c.lower_bound,
                    "{} seed {seed}: greedy {g} below the lower bound {}",
                    part.name,
                    c.lower_bound
                );
                if c.exact {
                    assert_eq!(g, c.lower_bound, "{} seed {seed}: an exact bound was not attained", part.name);
                }
            }
        }
    }

    /// A neuron whose fan-in alone exceeds a whole core's synapse capacity cannot be packed at all,
    /// and that is reported as "no assignment" rather than as a large number.
    #[test]
    fn a_neuron_too_big_for_any_core_has_no_packing() {
        let mut b = NetBuilder::new(20_000);
        for pre in 0..70_000u32 {
            b.connect(pre % 19_999, 19_999, 1e-3, 1).expect("in range");
        }
        let net = b.build();
        let c = core_count(&net, &ODIN).expect("ODIN states both");
        assert!(c.greedy.is_none(), "70,000 inputs cannot fit a 65,536-synapse core");
        assert!(c.lower_bound >= 2);
    }

    // ---------------------------------------------------------------------------------------
    // (e) Every record's provenance is present and every stated figure is traceable.
    // ---------------------------------------------------------------------------------------

    /// The gate on the whole table: a figure without a document behind it does not ship.
    #[test]
    fn every_spec_names_a_document_and_grades_it_consistently() {
        for p in PARTS {
            assert!(!p.name.is_empty());
            assert!(!p.vendor.is_empty(), "{} names no maker", p.name);
            assert!(p.citation.len() > 40, "{} cites nothing substantial", p.name);
            assert!(p.note.len() > 30, "{} carries no note", p.name);
            for (field, known, source, evidence) in graded(&p) {
                assert!(
                    source.len() >= 20,
                    "{}.{field} has a provenance of {} characters",
                    p.name,
                    source.len()
                );
                assert_eq!(
                    known,
                    evidence != Evidence::Unstated,
                    "{}.{field}: a stated figure must be graded above Unstated and an empty one \
                     must be graded Unstated",
                    p.name
                );
                assert_ne!(
                    evidence,
                    Evidence::Metered,
                    "{}.{field} claims Metered; nothing in this module is an instrument reading",
                    p.name
                );
                if !known {
                    assert!(
                        source.contains("did not locate")
                            || source.contains("NOT APPLICABLE")
                            || source.contains("NO structural")
                            || source.contains("wrong SHAPE")
                            || source.contains("Not separately"),
                        "{}.{field} is empty but does not say what was looked for: {source}",
                        p.name
                    );
                }
            }
        }
    }

    /// Every `(part, field)` graded [`Evidence::Derived`], as an exact list, plus the property the
    /// grade means: **arithmetic, shown**.
    ///
    /// ⛔ The predecessor of this test looked for the substring `"derive"` in the provenance and
    /// nothing else. That enforces a WORD, not a property: "Graded Derived for that reason."
    /// satisfies it with no arithmetic anywhere, and nine specs passed it while containing an
    /// inference, a software convention, an undated guess or a schema choice. The census below
    /// cannot be satisfied by wording — re-grading a field either shows up as an unexpected entry
    /// or as a missing one, and either way somebody has to come here and say which document moved.
    #[test]
    fn the_derived_grade_is_an_exact_census() {
        // The complete list of this review's own arithmetic. Every entry's string shows the sum.
        let expected: &[(&str, &str)] = &[
            ("Loihi", "synapses_per_core"),
            ("Loihi 2", "neurons_per_core"),
            ("Loihi 2", "synapses_per_core"),
            ("Loihi 2", "neurons_per_chip_stated"),
            ("DYNAP-SE", "synapses_per_core"),
            ("Darwin", "synapses_per_core"),
            ("BrainScaleS-2", "max_fan_in"),
        ];
        let mut found: Vec<(&str, &str)> = Vec::new();
        for p in PARTS {
            for (field, _, source, evidence) in graded(&p) {
                if evidence != Evidence::Derived {
                    continue;
                }
                found.push((p.name, field));
                let lower = source.to_lowercase();
                assert!(
                    lower.contains("derive"),
                    "{}.{field} is graded Derived and its provenance does not say so: {source}",
                    p.name
                );
                // And the arithmetic is SHOWN: an operator and a digit on either side of it.
                assert!(
                    shows_arithmetic(source),
                    "{}.{field} is graded Derived and shows no arithmetic. Derived means this \
                     review COMPUTED the figure; an inference, a convention or a guess is \
                     Projected. Provenance: {source}",
                    p.name
                );
            }
        }
        found.sort_unstable();
        let mut want = expected.to_vec();
        want.sort_unstable();
        assert_eq!(found, want, "the set of Derived figures changed");
    }

    /// Whether a provenance string actually displays a calculation: some `digit operator digit`,
    /// ignoring spaces and the thousands separators the strings use.
    fn shows_arithmetic(source: &str) -> bool {
        let b: Vec<char> = source.chars().filter(|c| !c.is_whitespace()).collect();
        for (i, c) in b.iter().enumerate() {
            if !matches!(c, 'x' | '*' | '/' | '^') {
                continue;
            }
            let before = i > 0 && b[i - 1].is_ascii_digit();
            let after = i + 1 < b.len() && b[i + 1].is_ascii_digit();
            if before && after {
                return true;
            }
        }
        false
    }

    /// The mirror image: every `(part, field)` graded [`Evidence::Projected`], as an exact list.
    ///
    /// This grade is where an **inference** belongs — an analogy from a sibling part, a year read
    /// off undated material, a vendor capacity claim, a schema choice this review made. It matters
    /// which side of the line these sit on, because [`Evidence`] sorts `Projected < Derived`: a
    /// guess graded `Derived` sorts as stronger evidence than a maker's own roadmap figure, and
    /// [`Part::weakest_evidence`] inherits the inversion. Five of these were graded `Derived`
    /// through v0.4.0.
    #[test]
    fn the_projected_grade_is_an_exact_census_of_this_reviews_inferences() {
        let expected: &[(&str, &str)] = &[
            ("Akida AKD1000", "neurons_per_chip_stated"),
            ("Akida AKD1500", "weight_bits"),
            ("SpiNNaker2", "on_chip_learning"),
            ("Xylo Audio 2", "year"),
            ("Speck", "year"),
            ("BrainScaleS-2", "cores_per_chip"),
        ];
        let mut found: Vec<(&str, &str)> = Vec::new();
        for p in PARTS {
            for (field, _, source, evidence) in graded(&p) {
                if evidence != Evidence::Projected {
                    continue;
                }
                found.push((p.name, field));
                let lower = source.to_lowercase();
                assert!(
                    lower.contains("inferred")
                        || lower.contains("schema choice")
                        || lower.contains("capacity claim"),
                    "{}.{field} is graded Projected and its provenance does not say what kind of \
                     unlocated figure it is: {source}",
                    p.name
                );
            }
        }
        found.sort_unstable();
        let mut want = expected.to_vec();
        want.sort_unstable();
        assert_eq!(found, want, "the set of Projected figures changed");
        // And the ordering fact the census exists to protect.
        assert!(
            Evidence::Projected < Evidence::Derived,
            "if this ever stops holding, the argument for splitting these two grades changes"
        );
    }

    /// The arithmetic the provenance strings claim, actually done — **reading the record's own
    /// fields**, never a literal retyped from the string.
    ///
    /// ⛔ Three of these checks used to compare against `Some(256 * 64)`, `Some(1000 * 1000)` and
    /// `Some(2048 * 2048)`: literals typed into the test, which is exactly the drift the check
    /// exists to catch. Under those, `DARWIN.neurons_per_core` could be changed from 2,048 to
    /// 1,024 and the whole suite stayed green, leaving a record whose synapse count is documented
    /// as "2,048 squared" beside a neuron count of 1,024. Every multiplicand below now comes out of
    /// the `Part` itself.
    #[test]
    fn the_derived_figures_reproduce_the_arithmetic_their_provenance_claims() {
        // Loihi: 1,048,576 one-bit synapses per core x 128 cores should land within a few percent
        // of the paper's ~130 M per chip, which is the cross-check the provenance string cites.
        let loihi_chip = LOIHI.synapses_per_chip().expect("both stated");
        assert_eq!(loihi_chip, 1_048_576 * 128);
        let rel = (loihi_chip as f64 - 130e6).abs() / 130e6;
        assert!(rel < 0.05, "Loihi's derivation is {:.1}% from the published figure", rel * 100.0);

        // Loihi 2: 120 M synapses / 128 cores, and 1,048,576 neurons / 128 cores, both exact.
        assert_eq!(LOIHI_2.synapses_per_chip(), Some(120_000_000));
        assert_eq!(LOIHI_2.neurons_per_core.value.map(u64::from), Some(1_048_576 / 128));

        // DYNAP-SE: neurons per core x CAM entries per neuron, both read off the record.
        let dyn_n = u64::from(DYNAP_SE.neurons_per_core.value.expect("stated"));
        let dyn_f = u64::from(DYNAP_SE.max_fan_in.value.expect("stated"));
        assert_eq!(DYNAP_SE.synapses_per_core.value, Some(dyn_n * dyn_f));
        // Darwin: its own neuron count squared, read off the record.
        let dar_n = u64::from(DARWIN.neurons_per_core.value.expect("stated"));
        assert_eq!(DARWIN.synapses_per_core.value, Some(dar_n * dar_n));
        // SpiNNaker states NEITHER: both fields held a 1 ms throughput budget and are now empty,
        // so there is no arithmetic left to reproduce and nothing for fits() to cap against.
        assert!(SPINNAKER.neurons_per_core.value.is_none());
        assert!(SPINNAKER.synapses_per_core.value.is_none());
        // BrainScaleS-2: a 512 x 256 array.
        assert_eq!(
            BRAINSCALES_2.synapses_per_core.value,
            Some(u64::from(BRAINSCALES_2.neurons_per_core.value.expect("stated"))
                * u64::from(BRAINSCALES_2.max_fan_in.value.expect("stated")))
        );
    }

    /// Where a maker publishes a chip total AND per-core figures, the two must agree. A
    /// transcription error in either one shows up here.
    ///
    /// ⛔ **Two of the four triples cannot fail, and the count says so.** `Loihi 2`'s
    /// `neurons_per_core` was *defined* as 1,048,576 / 128, so `1,048,576 == 8,192 * 128` is an
    /// identity; `BrainScaleS-2`'s `cores_per_chip = 1` is this review's own schema mapping, so
    /// `512 == 512 * 1` is an identity too. The predecessor of this test asserted `checked >= 4`
    /// as though all four were independent evidence. The product is still asserted for all four —
    /// a transcription error in any of them is worth catching — but the number that counts is the
    /// **independent** one, and a triple is independent only when none of its three figures was
    /// computed or inferred from the others. `Evidence::Measured` on all three is that test.
    #[test]
    fn a_published_chip_total_agrees_with_the_product_of_the_per_core_figures() {
        let mut checked: Vec<&str> = Vec::new();
        let mut independent: Vec<&str> = Vec::new();
        for p in PARTS {
            let (Some(total), Some(per_core), Some(cores)) =
                (p.neurons_per_chip_stated.value, p.neurons_per_core.value, p.cores_per_chip.value)
            else {
                continue;
            };
            checked.push(p.name);
            assert_eq!(
                total,
                u64::from(per_core) * u64::from(cores),
                "{}: published total {total} disagrees with {per_core} x {cores}",
                p.name
            );
            let sourced = [
                p.neurons_per_chip_stated.evidence,
                p.neurons_per_core.evidence,
                p.cores_per_chip.evidence,
            ]
            .iter()
            .all(|e| *e == Evidence::Measured);
            if sourced {
                independent.push(p.name);
            }
        }
        checked.sort_unstable();
        independent.sort_unstable();
        assert_eq!(checked, ["BrainScaleS-2", "DYNAP-SE", "Loihi 2", "TrueNorth"]);
        assert_eq!(
            independent,
            ["DYNAP-SE", "TrueNorth"],
            "an independent cross-check needs three separately sourced figures; a record whose \
             per-core figure was divided out of its chip total is checking itself"
        );
        // And the identities really are identities — stated here so nobody re-counts them as
        // evidence. Each of these is the record's own arithmetic played back.
        assert_eq!(LOIHI_2.neurons_per_core.evidence, Evidence::Derived);
        assert_eq!(BRAINSCALES_2.cores_per_chip.value, Some(1));
        assert_ne!(BRAINSCALES_2.cores_per_chip.evidence, Evidence::Measured);
    }

    /// The crossbar parts: synapses per core is neurons per core times the fan-in cap, because the
    /// core IS the crossbar. Checked only where all three figures are separately sourced —
    /// `DYNAP-SE`'s synapse count is itself derived from the other two, so including it would be
    /// asserting an identity against itself.
    #[test]
    fn a_crossbar_cores_synapse_count_is_its_two_sides_multiplied() {
        for p in [&TRUENORTH, &ODIN] {
            let n = u64::from(p.neurons_per_core.value.expect("stated"));
            let f = u64::from(p.max_fan_in.value.expect("stated"));
            assert_eq!(
                p.synapses_per_core.value,
                Some(n * f),
                "{}: {n} neurons x {f} inputs is not its stated synapse count",
                p.name
            );
            assert_eq!(p.synapses_per_core.evidence, Evidence::Measured);
        }
    }

    /// ⛔ What Merolla et al. 2014 actually prints, against what this record used to claim it
    /// printed.
    ///
    /// The provenance string for `TRUENORTH.synapses_per_core` said 4,096 cores "gives 268 M
    /// synapses per chip, which is the figure the paper reports". The paper's abstract reports
    /// **256 million** configurable synapses. The arithmetic was right and the attribution was
    /// wrong: 268,435,456 is the decimal rendering of the paper's binary 256 million,
    /// `256 * 2^20`, and it is this review's rendering rather than a number the paper prints. That
    /// is the same failure the module doc records for `crate::ledger::LOIHI_2018` — a statement
    /// about a piece of paper that the paper does not support — so it gets the same treatment: the
    /// identity is asserted here and the string now names the paper's own words.
    #[test]
    fn truenorths_chip_synapse_count_is_the_papers_256_million_in_binary() {
        let chip = TRUENORTH.synapses_per_chip().expect("both stated");
        assert_eq!(chip, 268_435_456);
        // The paper's own quantity: 256 million, where a million is 2^20.
        assert_eq!(chip, 256 * 1024 * 1024);
        // ...which is also the crossbar read off the die: 4,096 cores of a 256 x 256 array.
        assert_eq!(chip, 4096 * 256 * 256);
        // And the record says which of those the paper prints, and which is this review's.
        let src = TRUENORTH.synapses_per_core.source;
        assert!(src.contains("256 MILLION"), "{src}");
        assert!(
            src.contains("NOT a figure the paper prints"),
            "the record must not attribute this review's decimal rendering to the paper: {src}"
        );
    }

    /// The invariant that ties [`Part::weakest_evidence`] to [`Part::stated_fields`]: a record
    /// grades `Unstated` **exactly when** it has an empty field.
    ///
    /// ⛔ This replaces a gate that asserted `stated_fields() < 10` for every part — which
    /// required the table to stay incomplete forever and would have failed the day somebody
    /// sourced a missing figure. That is the wrong direction for a gate: it punishes the work it
    /// should reward. What is actually worth protecting is that the two summaries cannot
    /// contradict each other, which holds at ten stated fields as well as at one.
    #[test]
    fn a_records_weakest_grade_is_unstated_exactly_when_a_field_is_empty() {
        for p in PARTS {
            assert_eq!(
                p.weakest_evidence() == Evidence::Unstated,
                p.stated_fields() < 10,
                "{}: weakest_evidence() is {:?} beside {} of 10 stated fields",
                p.name,
                p.weakest_evidence(),
                p.stated_fields()
            );
        }
    }

    /// The completeness census: how much is actually known about each part, as a number, per part.
    ///
    /// This is also where the [`PARTS`] doc's ranking claim gets checked. That doc named "`ODIN`
    /// and `TrueNorth` at the top", which was wrong in both directions — `ODIN` has eight of ten
    /// and is not in the leading group, and `DYNAP-SE` and `BrainScaleS-2` tie `TrueNorth` at
    /// nine. The old guard asserted only `TRUENORTH >= 9` and `ODIN >= 8`, which passes no matter
    /// what sits above them.
    ///
    /// When a figure is genuinely sourced and a count here goes up, this table is the place to
    /// record it, beside the document it came from. That is a legitimate change and the test says
    /// so; what it refuses is a count moving without anyone noticing.
    #[test]
    fn the_completeness_ranking_is_the_one_the_table_doc_claims() {
        let expected: &[(&str, usize)] = &[
            ("TrueNorth", 9),
            ("DYNAP-SE", 9),
            ("BrainScaleS-2", 9),
            ("ODIN", 8),
            ("Loihi", 7),
            ("Loihi 2", 7),
            ("Akida AKD1000", 6),
            ("Xylo Audio 2", 6),
            ("SpiNNaker", 6),
            ("Darwin", 5),
            ("NorthPole", 5),
            ("SpiNNaker2", 4),
            ("Speck", 3),
            ("Akida AKD1500", 3),
            ("Darwin3", 3),
            ("Spiking Neural Processor T1", 1),
        ];
        for (name, want) in expected {
            let p = part_named(name).expect("named in the census");
            assert_eq!(p.stated_fields(), *want, "{name}'s completeness moved");
        }
        assert_eq!(expected.len(), PARTS.len(), "a part is missing from the census");

        // The ranking claim in the PARTS doc, computed rather than asserted part by part.
        let best = PARTS.iter().map(Part::stated_fields).max().expect("non-empty");
        let mut top: Vec<&str> =
            PARTS.iter().filter(|p| p.stated_fields() == best).map(|p| p.name).collect();
        top.sort_unstable();
        assert_eq!(best, 9);
        assert_eq!(
            top,
            ["BrainScaleS-2", "DYNAP-SE", "TrueNorth"],
            "the PARTS doc names the best-documented records; keep the two in step"
        );
        assert!(!top.contains(&"ODIN"), "ODIN is not in the leading group and the doc said it was");
        assert_eq!(INNATERA_T1.stated_fields(), 1, "Innatera's record is a year and nothing else");
        // Not one part in the open literature is fully documented today. Stated as an observation
        // about the table as it stands, not as a rule the table must keep obeying.
        assert!(best < 10, "a record reached ten stated fields: update this census and the doc");
    }

    /// A record resting on vendor material says so in capitals, where a reader quoting a figure
    /// cannot miss it.
    #[test]
    fn a_vendor_sourced_record_flags_itself() {
        for p in [&LOIHI_2, &AKD1000, &AKD1500, &XYLO_AUDIO_2, &SPECK, &INNATERA_T1] {
            assert!(
                p.citation.contains("VENDOR") || p.citation.contains("ANNOUNCEMENT"),
                "{} rests on vendor material and does not flag it: {}",
                p.name,
                p.citation
            );
        }
        // And the peer-reviewed ones cite a venue with a volume locator and a four-digit year, and
        // do NOT carry a vendor flag.
        //
        // ⛔ This used to be `citation.contains("20") && citation.contains(':')`, which "Vendor
        // blurb 2020: hi" satisfies. A locator here is a digit-colon-digit — `38(1):82-99`,
        // `16:795876` — which prose with a colon after it does not produce.
        for p in [&LOIHI, &TRUENORTH, &ODIN, &DYNAP_SE, &BRAINSCALES_2, &NORTHPOLE] {
            assert!(
                has_volume_locator(p.citation),
                "{} cites no volume locator (digits, colon, digits): {}",
                p.name,
                p.citation
            );
            assert!(
                four_digit_year(p.citation).is_some_and(|y| (1990..=2030).contains(&y)),
                "{} cites no plausible year: {}",
                p.name,
                p.citation
            );
            assert!(
                !p.citation.contains("VENDOR") && !p.citation.contains("ANNOUNCEMENT"),
                "{} is in the peer-reviewed list and flags itself as vendor material: {}",
                p.name,
                p.citation
            );
        }
        // The check has teeth: the string the predecessor accepted is rejected.
        assert!(!has_volume_locator("Vendor blurb 2020: hi"));
        assert!(has_volume_locator("IEEE Micro 38(1):82-99, 2018."));
        assert!(has_volume_locator("Frontiers in Neuroscience 16:795876, 2022."));
    }

    /// A volume locator: a colon with a digit immediately after it and either a digit or a closing
    /// parenthesis immediately before — `38(1):82-99`, `16:795876`, `345(6197):668-673`. Journal
    /// citations carry one; prose with a colon in it does not.
    fn has_volume_locator(s: &str) -> bool {
        let b = s.as_bytes();
        (1..b.len().saturating_sub(1)).any(|i| {
            b[i] == b':'
                && b[i + 1].is_ascii_digit()
                && (b[i - 1].is_ascii_digit() || b[i - 1] == b')')
        })
    }

    /// The last four-digit run in a citation, which is where the year sits in all of them.
    fn four_digit_year(s: &str) -> Option<u32> {
        let b = s.as_bytes();
        let mut found = None;
        for i in 0..b.len().saturating_sub(3) {
            if b[i..i + 4].iter().all(u8::is_ascii_digit)
                && (i == 0 || !b[i - 1].is_ascii_digit())
                && (i + 4 == b.len() || !b[i + 4].is_ascii_digit())
            {
                found = s[i..i + 4].parse().ok();
            }
        }
        found
    }

    #[test]
    fn the_table_has_no_duplicate_names_and_is_reachable_by_name() {
        for (i, a) in PARTS.iter().enumerate() {
            for b in PARTS.iter().skip(i + 1) {
                assert_ne!(a.name, b.name, "two records share the name {}", a.name);
            }
            assert_eq!(part_named(a.name).map(|p| p.name), Some(a.name));
        }
        assert!(part_named("loihi").is_none(), "lookup must be exact, not fuzzy");
        assert!(part_named("Loihi 3").is_none());
    }

    /// The correction this module was written after. The ledger's Loihi entry is graded
    /// `Simulated`, and nothing in this module quietly promotes it.
    #[test]
    fn this_module_does_not_reprice_anything_the_ledger_refused() {
        assert_eq!(crate::ledger::LOIHI_2018.evidence, Evidence::Simulated);
        assert!(ODIN.note.contains("does not transcribe"), "{}", ODIN.note);
        assert!(LOIHI.note.contains("pre-silicon"), "{}", LOIHI.note);
    }

    #[test]
    fn neurons_per_chip_prefers_a_published_total_over_a_product() {
        // Akida publishes a device total and no per-core figure.
        assert_eq!(AKD1000.neurons_per_chip(), Some(1_200_000));
        assert!(AKD1000.neurons_per_core.value.is_none());
        // Loihi publishes per-core figures and no total, so the product is used.
        assert_eq!(LOIHI.neurons_per_chip(), Some(1024 * 128));
        // Innatera publishes neither.
        assert_eq!(INNATERA_T1.neurons_per_chip(), None);
        assert_eq!(INNATERA_T1.synapses_per_chip(), None);
    }

    #[test]
    fn an_empty_network_is_a_question_with_an_answer() {
        let net = NetBuilder::new(0).build();
        let fit = fits(&net, &TRUENORTH);
        assert_eq!(fit.verdict, Some(true), "zero neurons fit anything that holds any");
        let c = core_count(&net, &TRUENORTH).expect("stated");
        assert_eq!(c.lower_bound, 0);
        assert_eq!(c.greedy, Some(0));
    }

    // ---------------------------------------------------------------------------------------
    // (f) The wall-versus-budget machinery, pinned variant by variant.
    //
    // ⛔ Everything in this block was unasserted. `Bind::overflow` was checked for `FanIn` and
    // nothing else, so returning 0 for the other five variants was green; `relieved_by_more_chips`
    // was pinned for `FanIn` and `DelayTooShort` only, so moving `DelayTooLong` to "buy more
    // chips" was green; and `Fit::scales_out`'s `.all` could be `.any` because no test ever built
    // a network that violates a wall AND a capacity at once. That distinction is the thing the
    // module doc says it is written around.
    // ---------------------------------------------------------------------------------------

    /// A part with every field empty, for tests that need a shape [`PARTS`] does not contain.
    /// `Part` is `pub` with `pub` fields, so this is a thing a caller can build too.
    fn bare_part(name: &'static str) -> Part {
        const WHY: &str = "A fixture for this module's tests; this review did not locate anything, \
                           because there is nothing to locate.";
        Part {
            name,
            vendor: "test fixture",
            citation: "A fixture, not a part. Not in PARTS and not a claim about any silicon.",
            year: Spec::unlocated(WHY),
            neurons_per_core: Spec::unlocated(WHY),
            cores_per_chip: Spec::unlocated(WHY),
            synapses_per_core: Spec::unlocated(WHY),
            max_fan_in: Spec::unlocated(WHY),
            weight_bits: Spec::unlocated(WHY),
            delay_ticks: Spec::unlocated(WHY),
            on_chip_learning: Spec::unlocated(WHY),
            process: Spec::unlocated(WHY),
            neurons_per_chip_stated: Spec::unlocated(WHY),
            note: WHY,
        }
    }

    /// A fixture part that states **every** limit, so one network can violate all six at once.
    fn tiny_part_stating_everything() -> Part {
        const WHY: &str = "A fixture figure, chosen small so a small network violates it.";
        let mut p = bare_part("Tiny");
        p.neurons_per_core = Spec::known(4, WHY, Evidence::Measured);
        p.cores_per_chip = Spec::known(2, WHY, Evidence::Measured);
        p.synapses_per_core = Spec::known(8, WHY, Evidence::Measured);
        p.max_fan_in = Spec::known(3, WHY, Evidence::Measured);
        p.delay_ticks =
            Spec::known(DelayRange { min_ticks: 2, max_ticks: 5 }, WHY, Evidence::Measured);
        p
    }

    /// Each variant's `overflow`, `relieved_by_more_chips`, `precedence` and `constraint`, with
    /// six different numbers so that a version answering 0 — or answering the other subtraction —
    /// cannot pass.
    #[test]
    fn every_bind_variant_reports_its_own_overflow_reach_and_precedence() {
        let all = [
            Bind::FanIn { neuron: 3, fan_in: 300, cap: 256, offenders: 2 },
            Bind::DelayTooShort { synapse: 6, pre: 1, post: 2, delay: 0, floor: 3, offenders: 20 },
            Bind::DelayTooLong { synapse: 5, pre: 1, post: 2, delay: 40, cap: 16, offenders: 3 },
            Bind::Neurons { needed: 2000, cap: 1000 },
            Bind::Synapses { needed: 70_000, cap: 65_536 },
            Bind::Cores { needed: 12, cap: 4 },
        ];
        // (overflow, relieved by more chips, precedence, constraint name, a phrase the Display
        // must carry so that the numbers reach a reader and not only a matcher)
        let want = [
            (44u64, false, 0u8, "maximum fan-in per neuron", "neuron 3 has 300"),
            (3, false, 1, "shortest synaptic delay", "shortest deliverable is 3"),
            (24, false, 2, "longest synaptic delay", "longest deliverable is 16"),
            (1000, true, 3, "neurons per chip", "2000 needed, 1000 per chip"),
            (4464, true, 4, "synapses per chip", "70000 needed, 65536 per chip"),
            (8, true, 5, "cores per chip", "at least 12 needed, 4 per chip"),
        ];
        for (b, (over, relieved, prec, name, shown)) in all.iter().zip(want) {
            assert_eq!(b.overflow(), over, "{b:?} overflow");
            assert_eq!(b.relieved_by_more_chips(), relieved, "{b:?} relieved_by_more_chips");
            assert_eq!(b.precedence(), prec, "{b:?} precedence");
            assert_eq!(b.constraint(), name, "{b:?} constraint");
            assert!(b.to_string().contains(shown), "{b}");
        }
        // The two facts the doc says are the same fact: a wall is exactly a constraint more
        // silicon cannot relieve, and it sorts above every capacity.
        for b in &all {
            assert_eq!(
                b.relieved_by_more_chips(),
                b.precedence() >= 3,
                "{b:?}: precedence and relieved_by_more_chips disagree about wall versus budget"
            );
        }
        // Distinct precedences, so "most severe first" is a total order and not a tie.
        let mut p: Vec<u8> = all.iter().map(Bind::precedence).collect();
        p.sort_unstable();
        p.dedup();
        assert_eq!(p.len(), all.len());
    }

    /// `Bind`'s fields are `pub`, so `fan_in < cap` is constructible — and the formatter used raw
    /// subtraction on it while [`Bind::overflow`] saturated. In debug that panicked; in release it
    /// printed 18446744073709551612. Both are crashes in a published crate, one of them silent.
    #[test]
    fn a_bind_displays_a_saturating_overflow_rather_than_panicking() {
        let cases = [
            Bind::FanIn { neuron: 0, fan_in: 1, cap: 5, offenders: 1 },
            Bind::Neurons { needed: 1, cap: 5 },
            Bind::Synapses { needed: 1, cap: 5 },
            Bind::Cores { needed: 1, cap: 5 },
        ];
        for b in cases {
            let s = b.to_string();
            assert_eq!(b.overflow(), 0, "{b:?}");
            assert!(s.contains("over by 0"), "{s}");
            assert!(!s.contains("18446744073709551612"), "{s}");
        }
        // And on a real violation the formatter and the method still agree.
        let real = Bind::Cores { needed: 12, cap: 4 };
        assert!(real.to_string().contains("over by 8"), "{real}");
    }

    /// ⛔ A network that violates a **wall** and a **capacity** at once must not report "buy more
    /// chips". No test built one, so `Fit::scales_out`'s `.all` could be `.any` — which is exactly
    /// backwards, and the backwards answer is the expensive one to act on.
    #[test]
    fn a_mixed_bind_network_does_not_report_buy_more_chips() {
        // 2,000 neurons at 100 inputs each, against DYNAP-SE: 64 inputs (a wall), 1,024 neurons,
        // 65,536 synapses and 4 cores per chip (three capacities).
        let net = uniform_net(2000, 100, 1);
        let fit = fits(&net, &DYNAP_SE);
        assert_eq!(fit.verdict, Some(false));
        let kinds: Vec<u8> = fit.binds.iter().map(Bind::precedence).collect();
        assert_eq!(kinds, vec![0, 3, 4, 5], "a wall and all three capacities should bind: {fit}");
        assert!(
            fit.binds.iter().any(|b| !b.relieved_by_more_chips()),
            "the fan-in wall is in there"
        );
        assert!(
            fit.binds.iter().any(Bind::relieved_by_more_chips),
            "and so are the capacities, which is what makes this the discriminating case"
        );
        assert!(
            !fit.scales_out(),
            "a network with a fan-in wall does not become mappable on a bigger machine: {fit}"
        );
        // Drop the fan-in below the wall and the very same overflows DO scale out.
        let ok = uniform_net(2000, 60, 1);
        let scaled = fits(&ok, &DYNAP_SE);
        assert_eq!(scaled.verdict, Some(false));
        assert!(scaled.binds.iter().all(Bind::relieved_by_more_chips));
        assert!(scaled.scales_out(), "{scaled}");
    }

    /// "Most severe first" is load-bearing: [`Fit::binding`] returns `binds.first()`. Every test
    /// that called `binding()` had exactly one bind, so the order was asserted by nothing. This
    /// builds a network that violates **all six** constraints of one part at once.
    #[test]
    fn the_binds_are_reported_most_severe_first() {
        let part = tiny_part_stating_everything();
        let mut b = NetBuilder::new(20);
        // Neuron 5 takes six inputs against a cap of 3, one of them same-tick and one far too
        // long, so fan-in and both delay ends bind together.
        for pre in 0..6u32 {
            let delay = match pre {
                0 => 0,  // below the floor of 2
                1 => 9,  // past the ceiling of 5
                _ => 3,  // inside the range
            };
            b.connect(pre, 5, 1e-3, delay).expect("in range");
        }
        for pre in 6..30u32 {
            b.connect(pre % 20, (pre % 19) + 1, 1e-3, 3).expect("in range");
        }
        let net = b.build();
        let fit = fits(&net, &part);

        assert_eq!(fit.verdict, Some(false));
        let order: Vec<u8> = fit.binds.iter().map(Bind::precedence).collect();
        assert_eq!(order, vec![0, 1, 2, 3, 4, 5], "all six bind, in precedence order: {fit}");
        assert!(
            order.windows(2).all(|w| w[0] <= w[1]),
            "Fit::binds must be sorted by precedence, because binding() takes the first"
        );
        assert!(matches!(fit.binding(), Some(Bind::FanIn { .. })), "{fit}");
        assert_eq!(
            fit.binding().map(Bind::precedence),
            fit.binds.iter().map(Bind::precedence).min(),
            "binding() must be the most severe bind, not merely the first one found"
        );
        assert!(fit.headroom.is_empty(), "{fit}");
        assert!(fit.unchecked.is_empty(), "this fixture states every limit: {fit}");
        assert!(!fit.scales_out());
    }

    /// Every string in [`Fit::unchecked`] is one [`Bind::constraint`] returns, over the whole
    /// table. Through v0.4.0 an unstated delay range pushed `"synaptic delay range"`, which is not
    /// one of the six, so a caller correlating the two lists by name dropped it silently.
    #[test]
    fn every_unchecked_name_is_a_bind_constraint_name() {
        let names = [
            "maximum fan-in per neuron",
            "shortest synaptic delay",
            "longest synaptic delay",
            "neurons per chip",
            "synapses per chip",
            "cores per chip",
        ];
        let net = uniform_net(40, 3, 1);
        let mut seen = std::collections::BTreeSet::new();
        for p in PARTS {
            let fit = fits(&net, &p);
            for u in &fit.unchecked {
                assert!(names.contains(u), "{}: unchecked name {u} is not a Bind::constraint", p.name);
                seen.insert(*u);
            }
            for h in &fit.headroom {
                assert!(
                    names.contains(&h.constraint),
                    "{}: headroom name {} is not a Bind::constraint",
                    p.name,
                    h.constraint
                );
            }
            // A constraint is checked or unchecked, never both and never neither.
            for n in names {
                let checked = fit.headroom.iter().any(|h| h.constraint == n)
                    || fit.binds.iter().any(|b| b.constraint() == n);
                let unchecked = fit.unchecked.contains(&n);
                // "shortest synaptic delay" gets no Headroom entry of its own — a floor is not a
                // utilisation — so it is the one name that can be neither, and only when the part
                // states a range.
                if n == "shortest synaptic delay" && p.delay_ticks.is_known() {
                    continue;
                }
                assert!(checked != unchecked, "{}: {n} is {checked}/{unchecked}", p.name);
            }
        }
        // Every one of the six really does turn up somewhere in this table.
        assert_eq!(seen.len(), names.len(), "unchecked names seen: {seen:?}");
    }

    /// ⛔ The regression this record was corrected for. `SpiNNaker`'s per-core neuron and synapse
    /// figures were throughput budgets, and [`fits`] multiplied them into capacities: a
    /// 17,500-neuron network came back `FITS ... neurons per chip: 17500 of 18000 (97.2%)`, a
    /// deadline reported as a wall with headroom, on a network past the ~16,000 the record's own
    /// provenance says is deployable. Both fields are empty now, so every capacity is reported as
    /// unchecked and the one genuine structure — the delay range — is the only thing graded.
    #[test]
    fn spinnaker_reports_no_capacity_because_its_capacities_were_deadlines() {
        let mut b = NetBuilder::new(17_500);
        for i in 0..17_500u32 {
            b.connect(i, i, 1e-3, 1).expect("in range");
        }
        let net = b.build();
        let fit = fits(&net, &SPINNAKER);

        for n in ["neurons per chip", "synapses per chip", "cores per chip"] {
            assert!(fit.unchecked.contains(&n), "{n} must be unchecked for SpiNNaker: {fit}");
            assert!(
                !fit.headroom.iter().any(|h| h.constraint == n),
                "{n} must not be reported with headroom: {fit}"
            );
        }
        assert!(fit.cores.is_none(), "no core count off a throughput budget: {fit}");
        assert!(fit.chips_lower_bound.is_none(), "{fit}");
        assert!(SPINNAKER.neurons_per_chip().is_none(), "a budget must not become a chip capacity");
        assert!(SPINNAKER.synapses_per_chip().is_none());
        assert!(core_count(&net, &SPINNAKER).is_none());
        assert!(!fit.to_string().contains("of 18000"), "{fit}");

        // What IS still checked: the delay range, which is a real structure — and it still
        // refuses a same-tick network.
        assert_eq!(fit.headroom.len(), 1, "{fit}");
        assert_eq!(fit.headroom[0].constraint, "longest synaptic delay");
        assert_eq!(fit.verdict, Some(true), "the delay range is satisfied, and that is all it says");
        assert_eq!(fit.unchecked.len(), 4, "{fit}");
    }

    // ---------------------------------------------------------------------------------------
    // (g) Refusals rather than substituted values.
    // ---------------------------------------------------------------------------------------

    /// ⛔ A positive full scale does not make a positive step. `symmetric(31, 1e-320)` divides a
    /// subnormal by 1,073,741,823 and the quotient underflows to exactly zero — the state
    /// [`HardwareError::NoScale`]'s doc says is impossible.
    ///
    /// What the old code did with it, measured: `quantise_nearest(&[0.0, 1e-320, -1e-320])`
    /// returned codes `[0, 1073741823, -1073741823]` and `values()` of `[0.0, 0.0, -0.0]` — every
    /// weight destroyed — while reporting `max_abs_error = 1e-320` and `mean_error = 0`, which
    /// reads as a near-perfect round trip. It is also the module's only NaN path: `0.0 / 0.0`.
    #[test]
    fn a_step_that_underflows_to_zero_is_refused_rather_than_destroying_every_weight() {
        assert_eq!(Quantiser::symmetric(31, 1e-320), Err(HardwareError::NoScale));
        assert_eq!(Quantiser::from_weights(31, &[0.0, 1e-320, -1e-320]), Err(HardwareError::NoScale));

        // The boundary is real and is where the arithmetic puts it, not where a magic number puts
        // it: for each width, the smallest scale that still yields a non-zero step is accepted and
        // anything below it is refused.
        for bits in 2..=31u32 {
            let max_code = f64::from((1i32 << (bits - 1)) - 1);
            // Smallest positive f64 is 5e-324; a scale of max_code times it still divides cleanly.
            let just_enough = 5e-324 * max_code;
            let q = Quantiser::symmetric(bits, just_enough)
                .unwrap_or_else(|e| panic!("{bits} bits, scale {just_enough:e}: {e}"));
            assert!(q.step > 0.0, "{bits} bits: step {:e}", q.step);
            assert!(q.half_lsb() >= 0.0);
            // And every accepted quantiser round-trips its own full scale to a finite number.
            let out = q.quantise_nearest(&[just_enough]).expect("finite");
            assert!(out.values()[0].is_finite());
            assert_eq!(out.clipped, 0, "{bits} bits");
        }
        // One rung below the boundary at the widest width, the quotient is zero and is refused.
        assert_eq!(Quantiser::symmetric(31, 5e-324), Err(HardwareError::NoScale));

        // No path out of this constructor yields a step that is not a usable positive number.
        for bits in [2u32, 8, 16, 31] {
            for scale in [1e-300, 1e-200, 1e-8, 1.0, 1e8, 1e300] {
                let q = Quantiser::symmetric(bits, scale).expect("a usable scale");
                assert!(q.step > 0.0 && q.step.is_finite(), "{bits} bits at {scale:e}");
            }
        }
    }

    /// A stated capacity of **zero** is refused, not substituted. `synapses_per_core = Some(0)`
    /// used to make `by_synapses` — and therefore `lower_bound`, and therefore
    /// [`CoreCount::chips`] — `usize::MAX`, silently. No part in [`PARTS`] states one; `Part` is
    /// `pub` with `pub` fields, so a caller can.
    #[test]
    fn a_zero_capacity_is_refused_rather_than_answered_with_usize_max() {
        const WHY: &str = "A fixture figure: a core that holds nothing.";
        let net = uniform_net(50, 3, 1);

        let mut zero_syn = bare_part("Zero synapses");
        zero_syn.neurons_per_core = Spec::known(16, WHY, Evidence::Measured);
        zero_syn.cores_per_chip = Spec::known(4, WHY, Evidence::Measured);
        zero_syn.synapses_per_core = Spec::known(0, WHY, Evidence::Measured);
        assert_eq!(core_count(&net, &zero_syn), None, "a zero capacity admits no bound");
        let fit = fits(&net, &zero_syn);
        assert!(fit.cores.is_none(), "{fit}");
        assert!(fit.chips_lower_bound.is_none(), "{fit}");
        assert!(fit.unchecked.contains(&"cores per chip"), "{fit}");

        let mut zero_neu = bare_part("Zero neurons");
        zero_neu.neurons_per_core = Spec::known(0, WHY, Evidence::Measured);
        zero_neu.cores_per_chip = Spec::known(4, WHY, Evidence::Measured);
        assert_eq!(core_count(&net, &zero_neu), None);

        // And the same part with a real capacity does produce a bound, so the refusal above is the
        // zero and not the fixture.
        zero_syn.synapses_per_core = Spec::known(64, WHY, Evidence::Measured);
        let c = core_count(&net, &zero_syn).expect("a real capacity");
        assert_eq!(c.by_neurons, 50usize.div_ceil(16));
        assert_eq!(c.by_synapses, Some(150u64.div_ceil(64) as usize));
        assert!(c.lower_bound < usize::MAX);
    }

    /// The guards on the two ratios: a cap of zero has no utilisation and no chips-per-core, and
    /// both say so instead of dividing.
    #[test]
    fn a_zero_cap_has_no_ratio_and_the_report_says_so() {
        let h = Headroom { constraint: "neurons per chip", used: 0, cap: 0 };
        assert_eq!(h.utilisation(), None, "0/0 is not 'fully utilised' and not 0%");
        assert_eq!(h.spare(), 0);
        // A non-zero cap does divide, so the guard above is the zero and not a broken method.
        let h2 = Headroom { constraint: "neurons per chip", used: 3, cap: 4 };
        assert_eq!(h2.utilisation(), Some(0.75));
        assert_eq!(h2.spare(), 1);
        // Used past the cap cannot underflow `spare` either.
        let h3 = Headroom { constraint: "neurons per chip", used: 9, cap: 4 };
        assert_eq!(h3.spare(), 0);
        assert_eq!(h3.utilisation(), Some(2.25));

        // And the report prints the absence rather than a percentage. It used to print -1.0%.
        let fit = Fit {
            part: "fixture",
            verdict: Some(true),
            binds: Vec::new(),
            headroom: vec![h],
            unchecked: Vec::new(),
            cores: None,
            chips_lower_bound: None,
        };
        let s = fit.to_string();
        assert!(s.contains("no utilisation"), "{s}");
        assert!(!s.contains("-1.0%"), "{s}");

        let c = CoreCount {
            by_neurons: 3,
            by_synapses: None,
            lower_bound: 3,
            greedy: None,
            exact: false,
        };
        assert_eq!(c.chips(0), None, "a chip with no cores holds no network");
        assert_eq!(c.chips(2), Some(2));
    }

    /// [`Part::neurons_per_chip`] prefers the published total **where the two disagree**, which is
    /// the only case where the preference is observable. Every record in [`PARTS`] that has both
    /// already agrees, so inverting the preference left all 38 tests green.
    #[test]
    fn a_published_chip_total_wins_over_the_product_when_they_disagree() {
        const WHY: &str = "A fixture figure, chosen so the total and the product differ.";
        let mut p = bare_part("Disagreeing");
        p.neurons_per_core = Spec::known(50, WHY, Evidence::Measured);
        p.cores_per_chip = Spec::known(4, WHY, Evidence::Measured);
        p.neurons_per_chip_stated = Spec::known(1000, WHY, Evidence::Measured);
        assert_eq!(
            p.neurons_per_chip(),
            Some(1000),
            "the published total wins; the product of two ceilings is a figure no configuration \
             reaches"
        );
        assert_ne!(p.neurons_per_chip(), Some(200), "the product must not win");

        // Remove the total and the product is the fallback, so the preference is a preference and
        // not a hard-coded field.
        p.neurons_per_chip_stated = Spec::unlocated("This fixture did not locate a device total.");
        assert_eq!(p.neurons_per_chip(), Some(200));

        // And a capacity overflow is judged against the stated total, not the product.
        let net = uniform_net(600, 2, 1);
        let mut with_total = p;
        with_total.neurons_per_chip_stated = Spec::known(1000, WHY, Evidence::Measured);
        assert!(!fits(&net, &with_total).binds.iter().any(|b| matches!(b, Bind::Neurons { .. })));
        assert!(fits(&net, &p).binds.iter().any(|b| matches!(b, Bind::Neurons { .. })));
    }

    #[test]
    fn a_delay_range_is_inclusive_at_both_ends() {
        let r = SPINNAKER.delay_ticks.value.expect("SpiNNaker states one");
        assert!(r.contains(r.min_ticks));
        assert!(r.contains(r.max_ticks));
        assert!(!r.contains(r.min_ticks - 1));
        assert!(!r.contains(r.max_ticks + 1));
    }

    // ---------------------------------------------------------------------------------------
    // (h) What the first mutation audit of this module could not see.
    //
    // ⛔ Thirteen edits survived the suite as it stood. Every one of them is a place where a
    // number, a boundary or a field was written down and then never read back: the two census
    // methods could not distinguish ten graded fields from nine, no network ever sat exactly ON a
    // cap, no test read a `Headroom::used` on either of the two constraints that compute one, and
    // four transcribed constants reached no assertion at all.
    // ---------------------------------------------------------------------------------------

    /// What a fixture's provenance says once a test has emptied one of its fields.
    const FIXTURE_GAP: &str = "A fixture gap: this test emptied the field so that the record's \
                               two census methods have something to notice.";

    /// A fixture record with all ten graded fields stated, so a sweep can knock exactly one of
    /// them down at a time.
    ///
    /// Every part in [`PARTS`] is missing at least one field — that is the honest state of the
    /// literature and it is also why no real record can isolate a single field's contribution to
    /// [`Part::weakest_evidence`].
    fn fully_stated_part() -> Part {
        const WHY: &str = "A fixture figure, not a claim about any silicon: this record exists so \
                           that a test can degrade one graded field at a time.";
        Part {
            name: "Fully stated",
            vendor: "test fixture",
            citation: "A fixture, not a part. Not in PARTS and not a claim about any silicon.",
            year: Spec::known(2026, WHY, Evidence::Measured),
            neurons_per_core: Spec::known(16, WHY, Evidence::Measured),
            cores_per_chip: Spec::known(4, WHY, Evidence::Measured),
            synapses_per_core: Spec::known(256, WHY, Evidence::Measured),
            max_fan_in: Spec::known(8, WHY, Evidence::Measured),
            weight_bits: Spec::known(8, WHY, Evidence::Measured),
            delay_ticks: Spec::known(
                DelayRange { min_ticks: 1, max_ticks: 7 },
                WHY,
                Evidence::Measured,
            ),
            on_chip_learning: Spec::known(false, WHY, Evidence::Measured),
            process: Spec::known("fixture 1 nm", WHY, Evidence::Measured),
            neurons_per_chip_stated: Spec::known(64, WHY, Evidence::Measured),
            note: WHY,
        }
    }

    /// Change one named field's [`Evidence`] grade in place, leaving its value alone.
    type Regrade = fn(&mut Part, Evidence);

    /// Empty one named field in place, so that both census methods have to notice the gap.
    type Unlocate = fn(&mut Part);

    /// The ten fields [`Part::weakest_evidence`] and [`Part::stated_fields`] both claim to read,
    /// each with a way to regrade it and a way to empty it.
    ///
    /// Named, so a failure in the sweep below says which field stopped being read rather than
    /// only that one did.
    const GRADED_FIELD_SETTERS: [(&str, Regrade, Unlocate); 10] = [
        ("year", |p, e| p.year.evidence = e, |p| p.year = Spec::unlocated(FIXTURE_GAP)),
        (
            "neurons_per_core",
            |p, e| p.neurons_per_core.evidence = e,
            |p| p.neurons_per_core = Spec::unlocated(FIXTURE_GAP),
        ),
        (
            "cores_per_chip",
            |p, e| p.cores_per_chip.evidence = e,
            |p| p.cores_per_chip = Spec::unlocated(FIXTURE_GAP),
        ),
        (
            "synapses_per_core",
            |p, e| p.synapses_per_core.evidence = e,
            |p| p.synapses_per_core = Spec::unlocated(FIXTURE_GAP),
        ),
        (
            "max_fan_in",
            |p, e| p.max_fan_in.evidence = e,
            |p| p.max_fan_in = Spec::unlocated(FIXTURE_GAP),
        ),
        (
            "weight_bits",
            |p, e| p.weight_bits.evidence = e,
            |p| p.weight_bits = Spec::unlocated(FIXTURE_GAP),
        ),
        (
            "delay_ticks",
            |p, e| p.delay_ticks.evidence = e,
            |p| p.delay_ticks = Spec::unlocated(FIXTURE_GAP),
        ),
        (
            "on_chip_learning",
            |p, e| p.on_chip_learning.evidence = e,
            |p| p.on_chip_learning = Spec::unlocated(FIXTURE_GAP),
        ),
        ("process", |p, e| p.process.evidence = e, |p| p.process = Spec::unlocated(FIXTURE_GAP)),
        (
            "neurons_per_chip_stated",
            |p, e| p.neurons_per_chip_stated.evidence = e,
            |p| p.neurons_per_chip_stated = Spec::unlocated(FIXTURE_GAP),
        ),
    ];

    /// [`Part::weakest_evidence`] reads all ten graded fields, and the value it starts the
    /// minimum from is an identity rather than a floor.
    ///
    /// ⛔ The suite could not see either fact, for the same reason twice: every record in
    /// [`PARTS`] is missing at least one field, so `weakest_evidence()` is `Unstated` for all
    /// sixteen of them. That makes the method indistinguishable from one that returns `Unstated`
    /// unconditionally, and `a_records_weakest_grade_is_unstated_exactly_when_a_field_is_empty`
    /// compares it against `stated_fields() < 10`, a condition that is likewise true for all
    /// sixteen. The same census also hid a narrower hole: the only three records at nine of ten
    /// are missing `delay_ticks`, and every record missing `process` is missing several other
    /// fields too, so deleting `process` from the list of grades read moved no minimum anywhere in
    /// the table.
    ///
    /// The fixture is the smallest thing that separates them — ten stated fields, so the minimum
    /// has somewhere to move from.
    #[test]
    fn the_weakest_grade_reads_every_one_of_the_ten_fields_and_starts_above_all_of_them() {
        let base = fully_stated_part();
        assert_eq!(base.stated_fields(), 10, "the fixture states all ten graded fields");
        assert_eq!(
            base.weakest_evidence(),
            Evidence::Measured,
            "ten Measured fields make a Measured record, not an Unstated one"
        );

        // The seed is the identity of a minimum, not a ceiling clamped onto the answer: a record
        // graded above Measured in every field comes back at that grade. These are fixture grades
        // and not a claim about any silicon — the module doc says Metered is used by nothing in
        // this table, and this record is not in it.
        let mut strongest = base;
        for (_, regrade, _) in GRADED_FIELD_SETTERS {
            regrade(&mut strongest, Evidence::Metered);
        }
        assert_eq!(
            strongest.weakest_evidence(),
            Evidence::Metered,
            "the starting value must not cap the answer it is only there to be replaced by"
        );

        // One field at a time. Each of the ten has to be able to drag the whole record down,
        // which is the whole content of "the weakest grade ANYWHERE in this record".
        for (name, regrade, unlocate) in GRADED_FIELD_SETTERS {
            let mut degraded = base;
            regrade(&mut degraded, Evidence::Projected);
            assert_eq!(
                degraded.weakest_evidence(),
                Evidence::Projected,
                "{name}: a Projected grade in this field never reached weakest_evidence()"
            );
            assert_eq!(degraded.stated_fields(), 10, "{name}: regrading does not empty a field");

            // And the two censuses read the SAME ten fields: emptying one has to move both, or a
            // record can carry a gap that only one of them reports.
            let mut emptied = base;
            unlocate(&mut emptied);
            assert_eq!(
                emptied.weakest_evidence(),
                Evidence::Unstated,
                "{name}: an empty field did not reach weakest_evidence()"
            );
            assert_eq!(
                emptied.stated_fields(),
                9,
                "{name}: an empty field did not reach stated_fields()"
            );
        }
        assert_eq!(GRADED_FIELD_SETTERS.len(), 10, "ten graded fields, as both methods claim");
    }

    /// The two weight widths in this table that nothing quantises, pinned against the documents
    /// their own provenance names.
    ///
    /// ⛔ [`Quantiser::for_part`] is exercised on `AKD1000`, `ODIN`, `TrueNorth` and the six parts
    /// that refuse for want of a width, so `LOIHI.weight_bits` and `SPINNAKER.weight_bits` were
    /// transcribed constants that no assertion in the module reached. Each is one substitution
    /// away from a plausible wrong number, and the wrong number is the same one a careless reader
    /// would reach for: Loihi's 9 bits including sign reads as a byte, and `SpiNNaker`'s 16-bit
    /// `sPyNNaker` synapse format reads as the 32-bit word of the `ARM` core that processes it —
    /// which is the misreading that record's own string is written to head off.
    ///
    /// The width is carried through to the quantiser it exists for, because one bit is a factor of
    /// two in the code range and that is where a caller would feel it.
    #[test]
    fn the_weight_widths_no_test_quantised_are_the_ones_their_documents_state() {
        assert_eq!(
            LOIHI.weight_bits.value,
            Some(9),
            "Davies et al. 2018 configures weight precision from 1 to 9 bits INCLUDING SIGN; 8 \
             would be a byte nobody published"
        );
        assert!(
            LOIHI.weight_bits.source.contains("from 1 to 9 bits including sign"),
            "{}",
            LOIHI.weight_bits.source
        );
        let loihi = Quantiser::for_part(&LOIHI, &[1.0, -0.5]).expect("Loihi states a width");
        assert_eq!(loihi.bits, 9);
        assert_eq!(loihi.max_code, 255, "9 bits including sign leaves 2^8 - 1 codes each side");

        assert_eq!(
            SPINNAKER.weight_bits.value,
            Some(16),
            "the sPyNNaker synapse format is 16-bit fixed point; 32 is the word width of the core \
             that processes it, which is the error this record's own string names"
        );
        assert!(
            SPINNAKER.weight_bits.source.contains("16-bit fixed-point weights"),
            "{}",
            SPINNAKER.weight_bits.source
        );
        assert!(
            SPINNAKER.weight_bits.source.contains("the cores are 32-bit ARMs"),
            "the string has to keep naming the misreading it exists to prevent: {}",
            SPINNAKER.weight_bits.source
        );
        let spin = Quantiser::for_part(&SPINNAKER, &[1.0, -0.5]).expect("SpiNNaker states a width");
        assert_eq!(spin.bits, 16);
        assert_eq!(spin.max_code, 32_767);
        // And recording the ARM word there would not merely be wrong, it would make the part
        // unquantisable: 32 bits is outside this quantiser's 2..=31, signed codes and all.
        assert_eq!(Quantiser::symmetric(32, 1.0), Err(HardwareError::BadBits { bits: 32 }));
    }

    /// `Xylo Audio 2`'s 64,000 synapses are the numerator of the mean its own fan-in field
    /// refuses to record, so the two have to be checked against each other.
    ///
    /// ⛔ Nothing read `XYLO_AUDIO_2.synapses_per_core.value`. The only network [`fits`] puts on
    /// this part is 100 neurons carrying 500 synapses, two orders of magnitude inside the cap, so
    /// a cap ten times too small was still a pass reported with headroom. The record's distinctive
    /// claim — "64,000/1,000 = 64 is a MEAN, not a maximum" — is arithmetic over two fields of
    /// this same record, and it is true for exactly one value of the numerator.
    #[test]
    fn xylos_synapse_count_is_the_numerator_of_its_own_mean_fan_in_caveat() {
        let syn = XYLO_AUDIO_2.synapses_per_core.value.expect("Xylo states a synapse count");
        let neu = XYLO_AUDIO_2.neurons_per_core.value.expect("Xylo states a neuron count");
        assert_eq!(syn, 64_000, "SynSense documents up to 64,000 synaptic connections");
        assert_eq!(neu, 1000);
        // Recomputed from the fields rather than read back off the string. Both are exact
        // integers, so this is equality and not a tolerance.
        assert_eq!(syn / u64::from(neu), 64, "the mean fan-in this record refuses to record");
        assert!(
            XYLO_AUDIO_2.max_fan_in.source.contains("64,000/1,000 = 64 is a MEAN"),
            "{}",
            XYLO_AUDIO_2.max_fan_in.source
        );
        assert!(
            XYLO_AUDIO_2.max_fan_in.value.is_none(),
            "a mean in a maximum's field would let fits() pass a network it should refuse"
        );

        // One core, so the chip total is that same figure and fits() grades a network against it.
        assert_eq!(XYLO_AUDIO_2.cores_per_chip.value, Some(1));
        assert_eq!(XYLO_AUDIO_2.synapses_per_chip(), Some(64_000));

        // And the cap is where the arithmetic puts it: 1,000 neurons at the mean fan-in fill the
        // store exactly, and the report says 100.0% rather than passing with room to spare.
        let full = uniform_net(1000, 64, 1);
        assert_eq!(full.n_syn, 64_000);
        let fit = fits(&full, &XYLO_AUDIO_2);
        assert_eq!(fit.verdict, Some(true), "{fit}");
        let h = fit
            .headroom
            .iter()
            .find(|h| h.constraint == "synapses per chip")
            .expect("Xylo states a synapse capacity, so it is checked");
        assert_eq!((h.used, h.cap), (64_000, 64_000));
        assert_eq!(h.spare(), 0, "full, and full is not over");
    }

    /// `Speck`'s nine cores are a **depth** limit, the only one in this table, and [`fits`] cannot
    /// see it.
    ///
    /// ⛔ `SPECK.cores_per_chip` reaches no assertion through the fitting path: the part states no
    /// neurons per core, so [`core_count`] refuses and `"cores per chip"` lands in
    /// [`Fit::unchecked`] instead of being compared against anything. The one checkable structure
    /// this record carries therefore has to be pinned on the record itself — against the prose
    /// beside it, which spells the count as an English word precisely so that a slip in the
    /// numeral is visible against something.
    #[test]
    fn specks_core_count_is_the_layer_depth_its_own_prose_spells_out() {
        let cores = SPECK.cores_per_chip.value.expect("Speck states a core count");
        assert_eq!(cores, 9, "nine event-driven convolutional cores, mapping one per layer");

        // The numeral and the word are two statements of one fact, written in two notations so
        // that they can be checked against each other.
        let spelled = [
            "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
            "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
        ];
        let word = spelled.get(cores as usize).copied().unwrap_or("a count past this test's table");
        assert!(
            SPECK
                .cores_per_chip
                .source
                .contains(&format!("{word} event-driven convolutional SNN cores")),
            "the numeral and the prose state different counts: {}",
            SPECK.cores_per_chip.source
        );
        assert!(
            SPECK.cores_per_chip.source.contains("deeper than nine layers does not fit"),
            "the depth claim is the reason this field is here: {}",
            SPECK.cores_per_chip.source
        );

        // And this is why the number has to be pinned here rather than through a network: a depth
        // constraint is not a count of anything fits() is handed.
        let net = uniform_net(40, 3, 1);
        let fit = fits(&net, &SPECK);
        assert_eq!(fit.verdict, None, "{fit}");
        assert!(fit.unchecked.contains(&"cores per chip"), "{fit}");
        assert!(fit.cores.is_none(), "{fit}");
    }

    /// A neuron sitting exactly ON `TrueNorth`'s 256-row crossbar column maps, and the headroom
    /// entry says 256 of 256 rather than 0 of 256.
    ///
    /// ⛔ Two holes closed by one network. No fixture in this module ever gave a neuron an
    /// in-degree **equal** to a cap — the numbers are 257 against 256, 300 against 256 and against
    /// 64, and 6 and 2 against the tiny fixture's 3 — so `d > cap` and `d >= cap` agreed on every
    /// network the suite builds, and [`Part::max_fan_in`]'s "sources one neuron CAN have" was
    /// never tested at the one place the word "can" does any work. And no test had ever read a
    /// [`Headroom::used`] on the fan-in constraint, so the field could report a constant zero and
    /// stay green. Exactly at the wall is the network where both are visible at once, and it is
    /// also where `utilisation()` is exactly 1.0 and `spare()` is exactly 0.
    #[test]
    fn a_neuron_exactly_at_the_crossbar_column_maps_and_its_headroom_says_so() {
        let cap = TRUENORTH.max_fan_in.value.expect("TrueNorth states the wall");
        assert_eq!(cap, 256, "one column of a 256-row crossbar");

        let mut b = NetBuilder::new(300);
        for pre in 0..cap {
            b.connect(pre, 299, 1e-3, 1).expect("in range");
        }
        let at_the_wall = b.build();
        assert_eq!(at_the_wall.in_degrees()[299], cap as usize);

        let fit = fits(&at_the_wall, &TRUENORTH);
        assert_eq!(fit.verdict, Some(true), "256 sources fit a 256-row column: {fit}");
        assert!(fit.binds.is_empty(), "{fit}");
        let h = fit
            .headroom
            .iter()
            .find(|h| h.constraint == "maximum fan-in per neuron")
            .expect("a satisfied fan-in constraint is reported with its headroom");
        assert_eq!(h.used, 256, "the deepest in-degree in the network, not zero");
        assert_eq!(h.cap, 256);
        assert_eq!(h.spare(), 0);
        assert_eq!(h.utilisation(), Some(1.0), "256/256 is exactly one, not approximately one");
        assert!(fit.to_string().contains("256 of 256"), "{fit}");

        // One more source and the same shape of network is refused, so the boundary is where the
        // crossbar puts it and not one row to either side.
        let mut past = NetBuilder::new(300);
        for pre in 0..=cap {
            past.connect(pre, 299, 1e-3, 1).expect("in range");
        }
        let over = past.build();
        match fits(&over, &TRUENORTH).binding().expect("binds") {
            Bind::FanIn { neuron, fan_in, cap: wall, offenders } => {
                assert_eq!((*neuron, *fan_in, *wall, *offenders), (299, 257, 256, 1));
            }
            other => panic!("expected a fan-in bind, got {other:?}"),
        }
    }

    /// A synapse whose delay is exactly `SpiNNaker`'s sixteenth tick is deliverable, and the
    /// headroom entry reports the longest delay in the network rather than zero.
    ///
    /// ⛔ The same two holes as the fan-in pair, at the other end of [`fits`].
    /// `a_delay_range_is_inclusive_at_both_ends` proves [`DelayRange::contains`] inclusive, but
    /// [`fits`] re-implements the comparison against `range.max_ticks` and no network in this
    /// module ever carried a delay equal to a ceiling: the fixtures use 0, 1, 3, 9 and 40 against
    /// 16, and 0, 3 and 9 against the tiny fixture's 5. And [`Headroom::used`] on the delay
    /// constraint was read by nothing, so the running maximum could be a constant.
    ///
    /// The longest delay here is on the FIRST synapse scanned and the shortest is on the last, so
    /// a running value that kept the latest delay rather than the largest reads 2 and not 16.
    #[test]
    fn a_delay_exactly_at_the_ceiling_is_deliverable_and_its_headroom_reports_it() {
        let range = SPINNAKER.delay_ticks.value.expect("SpiNNaker states a range");
        assert_eq!((range.min_ticks, range.max_ticks), (1, 16));

        let mut b = NetBuilder::new(8);
        for (pre, d) in [(0u32, range.max_ticks), (1, 7), (2, range.min_ticks), (3, 2)] {
            b.connect(pre, pre + 1, 1e-3, d).expect("in range");
        }
        let net = b.build();

        let fit = fits(&net, &SPINNAKER);
        assert_eq!(fit.verdict, Some(true), "16 is inside a range whose ceiling is 16: {fit}");
        assert!(fit.binds.is_empty(), "{fit}");
        assert_eq!(fit.headroom.len(), 1, "the delay range is SpiNNaker's one structure: {fit}");
        let h = &fit.headroom[0];
        assert_eq!(h.constraint, "longest synaptic delay");
        assert_eq!(h.used, 16, "the longest delay in the network, not the last one and not zero");
        assert_eq!(h.cap, 16);
        assert_eq!(h.spare(), 0);
        assert_eq!(h.utilisation(), Some(1.0), "16/16 is exactly one");
        assert!(fit.to_string().contains("16 of 16"), "{fit}");

        // One tick further and the same network is refused, so the ceiling is a ceiling and the
        // comparison is not a tick out in either direction.
        let mut b2 = NetBuilder::new(8);
        b2.connect(0, 1, 1e-3, range.max_ticks + 1).expect("in range");
        let over = b2.build();
        match fits(&over, &SPINNAKER).binding().expect("binds") {
            Bind::DelayTooLong { delay, cap, offenders, .. } => {
                assert_eq!((*delay, *cap, *offenders), (17, 16, 1));
            }
            other => panic!("expected a delay ceiling bind, got {other:?}"),
        }
    }

    /// [`Fit::scales_out`] is `false` when nothing binds, because there is nothing to relieve.
    ///
    /// ⛔ `.all()` is vacuously true on an empty list, so the emptiness guard is the whole of this
    /// promise — and every test that called `scales_out()` had first built a network that violates
    /// something, which is exactly the case the guard does not cover. Asked of a network that
    /// fits, the unguarded form answers "buy more chips", which is advice to spend money on a
    /// machine that is already big enough, given about a network that already maps.
    #[test]
    fn a_network_that_violates_nothing_does_not_advise_buying_more_chips() {
        let net = uniform_net(100, 5, 1);

        let passes = fits(&net, &XYLO_AUDIO_2);
        assert_eq!(passes.verdict, Some(true));
        assert!(passes.binds.is_empty());
        assert!(passes.binding().is_none());
        assert!(!passes.scales_out(), "nothing binds, so nothing is relieved: {passes}");

        // And the honest no-verdict case, where nothing was checkable in the first place.
        let unknowable = fits(&net, &INNATERA_T1);
        assert_eq!(unknowable.verdict, None);
        assert!(unknowable.binds.is_empty());
        assert!(
            !unknowable.scales_out(),
            "a part with no public limits does not get a purchasing recommendation: {unknowable}"
        );

        // Asked of a Fit built by hand, since Fit's fields are pub and a caller can make one: an
        // empty binds list is false and one relievable bind in the same record is true, so what
        // the guard reads is the emptiness and not some property of the method.
        let empty = Fit {
            part: "fixture",
            verdict: Some(true),
            binds: Vec::new(),
            headroom: Vec::new(),
            unchecked: Vec::new(),
            cores: None,
            chips_lower_bound: None,
        };
        assert!(!empty.scales_out());
        let one = Fit { binds: vec![Bind::Neurons { needed: 2, cap: 1 }], ..empty.clone() };
        assert!(one.scales_out(), "one relievable bind IS what more chips fix");
    }

    /// First-fit-**decreasing**, and the direction is load-bearing.
    ///
    /// ⛔ `the_greedy_packing_never_beats_the_lower_bound` asserts `greedy >= lower_bound` always
    /// and `greedy == lower_bound` wherever `exact` holds — and `exact` is precisely the condition
    /// under which any `neurons_per_core` neurons fit one core together, so under it EVERY order
    /// attains the bound. Both assertions therefore hold for an ascending sort, and twelve seeds
    /// of random irregular networks never happened to build one where the two orders differ.
    ///
    /// This one differs by construction. Six in-degrees — 4, 4, 3, 3, 2, 2 — into cores of six
    /// synapses: descending packs 4+2, 4+2 and 3+3 into three cores, which is the bound, while
    /// ascending fills its first core with 2+2 and its second with 3+3 and then has nowhere to put
    /// either 4, spending four cores on eighteen synapses that need three.
    #[test]
    fn first_fit_decreasing_packs_the_largest_in_degrees_first() {
        const WHY: &str = "A fixture figure, chosen so that the two packing orders differ.";
        let mut part = bare_part("Packing order");
        part.neurons_per_core = Spec::known(6, WHY, Evidence::Measured);
        part.synapses_per_core = Spec::known(6, WHY, Evidence::Measured);
        part.cores_per_chip = Spec::known(8, WHY, Evidence::Measured);

        let degrees = [4usize, 4, 3, 3, 2, 2];
        let mut b = NetBuilder::new(degrees.len());
        for (post, &d) in degrees.iter().enumerate() {
            for pre in 0..d {
                b.connect(pre as u32, post as u32, 1e-3, 1).expect("in range");
            }
        }
        let net = b.build();
        assert_eq!(net.in_degrees(), degrees.to_vec(), "the fixture's shape is the whole point");
        assert_eq!(net.n_syn, 18);

        let c = core_count(&net, &part).expect("the fixture states both capacities");
        assert!(!c.exact, "4 * 6 > 6, so the neuron bound is not provably tight here");
        assert_eq!(c.by_neurons, 1, "six neurons fit one core's neuron cap");
        assert_eq!(c.by_synapses, Some(3), "eighteen synapses at six per core");
        assert_eq!(c.lower_bound, 3);
        assert_eq!(
            c.greedy,
            Some(3),
            "first-fit-DECREASING attains the bound here; ascending needs a fourth core"
        );
        // The neuron cap is not what separates the two orders: no core in either packing holds
        // more than three of the six neurons, so this is the synapse cap and the order alone.
        assert!(c.greedy.expect("a packing exists") <= degrees.len());
    }

    /// Quantising an empty slice is refused, in both rounding modes.
    ///
    /// ⛔ [`HardwareError::NoWeights`] was raised only through `max_abs_of`, on the
    /// [`Quantiser::from_weights`] path, so every test that saw it had already gone looking for a
    /// scale in the weights. A quantiser built by [`Quantiser::symmetric`] carries a scale of its
    /// own and can be handed `&[]` directly — and without the guard the accumulation loop never
    /// runs, `sum_error / 0.0` and `(sum_sq / 0.0).sqrt()` are both `NaN`, and the call succeeds
    /// with a bias and an RMS error of `NaN` beside a `max_abs_error` of 0 and no codes. This
    /// implementation measures exactly that when the guard is removed: `mean_error` and
    /// `rms_error` come back `NaN` and `clipped` comes back 0, which reads as a flawless round
    /// trip of nothing.
    #[test]
    fn quantising_an_empty_slice_is_refused_rather_than_answered_with_a_nan_round_trip() {
        let q = Quantiser::symmetric(8, 1.0).expect("a usable scale");
        assert_eq!(q.quantise_nearest(&[]), Err(HardwareError::NoWeights));
        let mut rng = Rng::new(7);
        assert_eq!(q.quantise_stochastic(&[], &mut rng), Err(HardwareError::NoWeights));

        // The same refusal on the scale-finding path, so the two guards are reached by different
        // callers and neither is a duplicate of the other.
        assert_eq!(Quantiser::from_weights(8, &[]), Err(HardwareError::NoWeights));

        // One weight is enough to succeed, so what is refused above is the emptiness and not the
        // call — and the statistics it reports are finite numbers rather than NaN.
        let one = q.quantise_nearest(&[0.5]).expect("one weight is quantisable");
        assert_eq!(one.codes.len(), 1);
        assert!(one.mean_error.is_finite(), "mean_error {}", one.mean_error);
        assert!(one.rms_error.is_finite(), "rms_error {}", one.rms_error);
        assert_eq!(one.clipped, 0);
    }
}
