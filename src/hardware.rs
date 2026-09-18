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
//! | [`Evidence::Projected`] | A design target, a roadmap figure, or a part this review does not know to be fabricated. |
//! | [`Evidence::Derived`] | **This review computed it** from other stated figures. The arithmetic is shown in the string. |
//! | [`Evidence::Simulated`] | From a simulation of the design rather than from the design's documentation. |
//! | [`Evidence::Measured`] | Stated for a fabricated part in a primary document — a peer-reviewed paper, or a vendor document for shipping silicon. Where it is a vendor document rather than a paper, **the provenance string says "vendor"**. |
//! | [`Evidence::Metered`] | Not used in this module. `Metered` is about an instrument reading, and nothing here is an instrument reading. |
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
//!   Both facts are tested against closed forms in this module.
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
    /// Neurons on one chip where the maker publishes a **chip total** directly.
    ///
    /// Separate from the product of `neurons_per_core` and `cores_per_chip` because for several
    /// parts the chip total is the published quantity and the per-core split is not. Use
    /// [`Part::neurons_per_chip`], which prefers a published total and falls back to the product.
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
        "DERIVED by this review: Intel's vendor brief states up to 1 million neurons per chip, and \
         this review records the nearest power of two, 1,048,576, which is exactly what 128 cores \
         of 8,192 gives.",
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
        "A full 256x256 binary crossbar per core: 256 input axons by 256 neurons. 4,096 cores \
         gives 268 M synapses per chip, which is the figure the paper reports.",
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
        "DERIVED by this review from BrainChip's statement that AKD1500 carries the same Akida 1.0 \
         IP as AKD1000, whose brief gives 1, 2 and 4 bits. Not separately stated for this part.",
        Evidence::Derived,
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
    neurons_per_core: Spec::known(
        1000,
        "The design target: ~1,000 neurons per core at ~1,000 inputs each in biological real time \
         with a 1 ms timestep. A THROUGHPUT BUDGET, not a structural cap — the part will accept \
         more and miss its deadline. Graded Derived for that reason.",
        Evidence::Derived,
    ),
    cores_per_chip: Spec::known(
        18,
        "Painkras et al., IEEE JSSC 48(8), 2013: 18 ARM968 cores per chip. One is reserved as a \
         monitor processor and one is a spare for yield, so 16 are typically deployable. This field \
         records the 18 on the die and names the 16 here rather than silently picking one.",
        Evidence::Measured,
    ),
    synapses_per_core: Spec::known(
        1_000_000,
        "DERIVED by this review: 1,000 neurons x 1,000 inputs, the rate the design was sized for. \
         The hard resources are the chip's 128 MB of die-stacked SDRAM holding synapse rows and the \
         core's 96 KB of tightly-coupled memory; the 10^6 is a budget, not a capacity.",
        Evidence::Derived,
    ),
    max_fan_in: Spec::unlocated(
        "There is NO structural fan-in cap, and this empty field is the finding rather than a gap. \
         A synapse row is fetched from SDRAM by DMA and processed in software, so fan-in costs TIME \
         and is never refused. The limit is that the row must be fetched and processed inside the \
         1 ms timestep.",
    ),
    weight_bits: Spec::known(
        16,
        "sPyNNaker's standard synapse format uses 16-bit fixed-point weights. A SOFTWARE \
         convention, not a hardware limit: the cores are 32-bit ARMs and a different synapse format \
         would be a different number. Graded Derived accordingly.",
        Evidence::Derived,
    ),
    delay_ticks: Spec::known(
        DelayRange { min_ticks: 1, max_ticks: 16 },
        "sPyNNaker delivers 1 to 16 timesteps natively from the synapse row's delay field; longer \
         delays are built from 'delay extension' populations that relay a spike through extra \
         neurons, at the cost of those neurons. Reported from the software's documented limit \
         rather than from the JSSC paper, hence Derived. The minimum of 1 is real: there is no \
         same-tick delivery.",
        Evidence::Derived,
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
    neurons_per_chip_stated: Spec::unlocated(CHIP_TOTAL_IS_A_PRODUCT),
    note: "The only part in this table whose limits are deadlines rather than structures. Its \
           failure mode is a missed deadline, which is quieter and worse than a refusal.",
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
        "DERIVED by this review: software plasticity on the Cortex-M4F cores as in SpiNNaker 1, \
         with hardware MAC and exponential units making the rule cheaper. arXiv:2103.08392 \
         describes the accelerators; this review did not locate a plasticity benchmark.",
        Evidence::Derived,
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
        "DERIVED by this review: the year is inferred from the vendor documentation and the \
         NeuroBench evaluation rather than read off a launch announcement. Treat it as \
         approximate.",
        Evidence::Derived,
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
        "DERIVED by this review from undated vendor material; this review did not locate a dated \
         launch document, so treat the year as approximate.",
        Evidence::Derived,
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
        "DERIVED by this review: 256 neurons x 64 CAM entries each. The paper states the 64 per \
         neuron; the product is this review's arithmetic.",
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
        "One, DERIVED from this review's mapping of the whole chip to one core: see \
         neurons_per_core. Not a claim about the die's internal structure.",
        Evidence::Derived,
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
/// are actually documented — [`ODIN`] and [`TRUENORTH`] at the top, [`INNATERA_T1`] at the bottom
/// with one.
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
    /// Always at least 1 for a real violation, because a violation by zero is not a violation.
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
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FanIn { neuron, fan_in, cap, offenders } => write!(
                f,
                "fan-in: neuron {neuron} has {fan_in} presynaptic sources, cap is {cap} \
                 (over by {}); {offenders} neuron(s) exceed it",
                fan_in - cap
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
                write!(f, "neurons: {needed} needed, {cap} per chip (over by {})", needed - cap)
            }
            Self::Synapses { needed, cap } => {
                write!(f, "synapses: {needed} needed, {cap} per chip (over by {})", needed - cap)
            }
            Self::Cores { needed, cap } => write!(
                f,
                "cores: at least {needed} needed, {cap} per chip (over by {})",
                needed - cap
            ),
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
            let pct = h.utilisation().map_or(-1.0, |u| u * 100.0);
            writeln!(f, "  ok     {}: {} of {} ({pct:.1}%)", h.constraint, h.used, h.cap)?;
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
/// Checks, in reporting precedence: per-neuron fan-in, synaptic delay range, neurons per chip,
/// synapses per chip, and cores per chip against [`CoreCount::lower_bound`]. Every constraint the
/// part does not state lands in [`Fit::unchecked`] instead of being assumed satisfied.
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
        if short.is_none() && long.is_none() {
            headroom.push(Headroom {
                constraint: "longest synaptic delay",
                used: u64::from(max_seen),
                cap: u64::from(range.max_ticks),
            });
        }
    } else {
        unchecked.push("synaptic delay range");
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
#[must_use]
pub fn core_count(net: &Net, part: &Part) -> Option<CoreCount> {
    let npc = part.neurons_per_core.value?;
    if npc == 0 {
        return None;
    }
    let npc_u = npc as usize;
    let by_neurons = net.n.div_ceil(npc_u);
    let spc = part.synapses_per_core.value;
    let by_synapses = spc.map(|s| {
        if s == 0 { usize::MAX } else { (net.n_syn as u64).div_ceil(s) as usize }
    });
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
    /// Every weight is zero, so there is no range to quantise over.
    ///
    /// Refused rather than answered with a step of zero. A vector of zeros round-trips exactly at
    /// any step, so a "successful" quantisation here would report a perfect result for a network
    /// that has no weights — which is almost always a bug upstream and never a useful answer.
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
                "every weight is zero: there is no range to quantise over, and a step of zero \
                 would report a perfect round trip for a network with no weights",
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
    /// low-precision weight. What it costs is variance — its RMS error is larger than
    /// round-to-nearest's, and any single quantisation can be worse. Both facts are tested against
    /// closed forms in this module.
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
    /// non-finite `max_abs`, and [`HardwareError::NoScale`] when `max_abs <= 0`.
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
        Ok(Self { bits, step: max_abs / f64::from(max_code), max_code })
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
    /// Larger for [`Rounding::Stochastic`] than for [`Rounding::Nearest`] on the same data. That is
    /// the trade and it is asserted in `stochastic_rounding_costs_variance_for_its_lack_of_bias`.
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
    #[must_use]
    pub fn within_half_lsb(&self, tol: f64) -> bool {
        self.clipped == 0 && self.max_abs_error <= self.half_lsb() * (1.0 + tol)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        AKD1000, AKD1500, BRAINSCALES_2, Bind, DARWIN, DARWIN3, DYNAP_SE, Evidence, HardwareError,
        INNATERA_T1, LOIHI, LOIHI_2, NORTHPOLE, ODIN, PARTS, Part, Quantiser, Rounding, SPECK,
        SPINNAKER, SPINNAKER2, TRUENORTH, XYLO_AUDIO_2, core_count, fits, part_named,
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

    #[test]
    fn a_fan_in_cap_is_refused_with_the_offending_neuron_and_the_numbers() {
        let mut b = NetBuilder::new(400);
        for pre in 0..300u32 {
            b.connect(pre, 7, 1e-3, 1).expect("in range");
        }
        for pre in 0..100u32 {
            b.connect(pre, 9, 1e-3, 1).expect("in range");
        }
        let net = b.build();

        let fit = fits(&net, &TRUENORTH);
        assert_eq!(fit.verdict, Some(false));
        match fit.binding().expect("something binds") {
            Bind::FanIn { neuron, fan_in, cap, offenders } => {
                assert_eq!(*neuron, 7);
                assert_eq!(*fan_in, 300);
                assert_eq!(*cap, 256);
                assert_eq!(*offenders, 1, "only neuron 7 exceeds TrueNorth's 256");
            }
            other => panic!("expected a fan-in bind, got {other:?}"),
        }
        assert_eq!(fit.binding().expect("bind").overflow(), 44);
        assert!(!fit.scales_out(), "fan-in is not relieved by more chips");
        assert!(fit.to_string().contains("neuron 7"), "{fit}");

        // The same network against the tightest cap in the table: both neurons now offend.
        let tight = fits(&net, &DYNAP_SE);
        match tight.binding().expect("binds") {
            Bind::FanIn { neuron, fan_in, cap, offenders } => {
                assert_eq!((*neuron, *fan_in, *cap, *offenders), (7, 300, 64, 2));
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
        assert_eq!(fit.unchecked.len(), 2, "{fit}");
        assert!(fit.unchecked.contains(&"maximum fan-in per neuron"));
        assert!(fit.unchecked.contains(&"synaptic delay range"));
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
        assert_eq!(fit.unchecked.len(), 5, "{fit}");
        assert!(fit.cores.is_none());
        assert!(fit.to_string().contains("NO VERDICT"), "{fit}");
    }

    /// ⛔ The weaker honesty case, and the reason [`Fit::unchecked`] has to be read first. Darwin3
    /// publishes ONE figure — a chip-level neuron total — so a 50-neuron network gets a
    /// `Some(true)` verdict off a single satisfied constraint with four unchecked beside it. That
    /// verdict is true and nearly worthless, and the report says both things.
    #[test]
    fn a_verdict_from_one_checkable_constraint_carries_its_four_unchecked_ones() {
        let net = uniform_net(50, 3, 1);
        let fit = fits(&net, &DARWIN3);
        assert_eq!(fit.verdict, Some(true));
        assert_eq!(fit.headroom.len(), 1, "{fit}");
        assert_eq!(fit.headroom[0].constraint, "neurons per chip");
        assert_eq!(fit.headroom[0].cap, 2_350_000);
        assert_eq!(fit.unchecked.len(), 4, "{fit}");
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

    /// A derivation has to say it is one. Otherwise this review's arithmetic reads as somebody's
    /// published figure — which is the exact shape of the error `crate::ledger::LOIHI_2018`
    /// records.
    #[test]
    fn every_derived_figure_says_it_was_derived() {
        for p in PARTS {
            for (field, _, source, evidence) in graded(&p) {
                if evidence == Evidence::Derived {
                    let lower = source.to_lowercase();
                    assert!(
                        lower.contains("derive"),
                        "{}.{field} is graded Derived and its provenance does not say so: {source}",
                        p.name
                    );
                }
            }
        }
    }

    /// The arithmetic the provenance strings claim, actually done.
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

        // DYNAP-SE: 256 neurons x 64 CAM entries.
        assert_eq!(DYNAP_SE.synapses_per_core.value, Some(256 * 64));
        // SpiNNaker: 1,000 neurons x 1,000 inputs.
        assert_eq!(SPINNAKER.synapses_per_core.value, Some(1000 * 1000));
        // Darwin: 2,048 squared.
        assert_eq!(DARWIN.synapses_per_core.value, Some(2048 * 2048));
        // BrainScaleS-2: a 512 x 256 array.
        assert_eq!(
            BRAINSCALES_2.synapses_per_core.value,
            Some(u64::from(BRAINSCALES_2.neurons_per_core.value.expect("stated"))
                * u64::from(BRAINSCALES_2.max_fan_in.value.expect("stated")))
        );
    }

    /// Where a maker publishes a chip total AND per-core figures, the two must agree. A
    /// transcription error in either one shows up here.
    #[test]
    fn a_published_chip_total_agrees_with_the_product_of_the_per_core_figures() {
        let mut checked = 0;
        for p in PARTS {
            let (Some(total), Some(per_core), Some(cores)) =
                (p.neurons_per_chip_stated.value, p.neurons_per_core.value, p.cores_per_chip.value)
            else {
                continue;
            };
            checked += 1;
            assert_eq!(
                total,
                u64::from(per_core) * u64::from(cores),
                "{}: published total {total} disagrees with {per_core} x {cores}",
                p.name
            );
        }
        assert!(checked >= 4, "only {checked} records could be cross-checked");
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

    /// Not one part in the open literature is fully documented, and the table says so rather than
    /// implying completeness by filling the gaps.
    #[test]
    fn no_part_in_this_table_is_completely_documented() {
        for p in PARTS {
            assert_eq!(
                p.weakest_evidence(),
                Evidence::Unstated,
                "{} claims a complete record; check whether a field was guessed",
                p.name
            );
            assert!(p.stated_fields() < 10, "{} states all ten fields", p.name);
        }
        assert_eq!(INNATERA_T1.stated_fields(), 1, "Innatera's record is a year and nothing else");
        assert!(TRUENORTH.stated_fields() >= 9);
        assert!(ODIN.stated_fields() >= 8);
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
        // And the peer-reviewed ones cite a venue with a volume and a year.
        for p in [&LOIHI, &TRUENORTH, &ODIN, &DYNAP_SE, &BRAINSCALES_2, &NORTHPOLE] {
            assert!(
                p.citation.contains("20") && p.citation.contains(':'),
                "{} does not cite a volume and year: {}",
                p.name,
                p.citation
            );
        }
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

    #[test]
    fn a_delay_range_is_inclusive_at_both_ends() {
        let r = SPINNAKER.delay_ticks.value.expect("SpiNNaker states one");
        assert!(r.contains(r.min_ticks));
        assert!(r.contains(r.max_ticks));
        assert!(!r.contains(r.min_ticks - 1));
        assert!(!r.contains(r.max_ticks + 1));
    }
}
