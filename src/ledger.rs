//! The joules ledger: what a spiking workload costs, and when that question has no honest answer.
//!
//! # The claim this module exists to refuse
//!
//! Almost every energy figure published for a spiking network is a **synaptic operation count**
//! multiplied by a per-operation energy taken from a chip's datasheet. It is an appealing method:
//! the count is exact, the constant is published, and the resulting number is enormous compared to
//! a GPU. It is also the same mistake this Institute made in a different crate, where omitting the
//! attention score matrix from an arithmetic-intensity figure made it wrong by 847x, and the
//! omission was invisible because the number it produced looked reasonable.
//!
//! A synaptic operation is not free-standing. Before a spike can update a neuron, **the synapse's
//! weight has to be read out of memory**. On a chip where the synapse lives in on-core SRAM that
//! read is cheap; on one where it lives in DRAM it can dominate everything else; and in a workload
//! whose connectivity does not fit on-core, the traffic is the bill. Counting synaptic operations
//! and pricing them from a per-operation figure prices the arithmetic and silently sets the memory
//! traffic to zero — which is precisely the term that separates a real deployment from a benchmark.
//!
//! So [`Prices`] has a `e_syn_fetch` field, every device table in this module leaves it
//! **`None`**, and [`Ledger::joules`] therefore returns `None` for every one of them. That is not
//! an unfinished implementation. It is the finding: *this review did not locate a published
//! per-synapse memory-fetch energy for any commercially available neuromorphic processor.* The
//! nearest it found is on a research part: `TrueNorth`'s supplementary material measures about
//! 47 fJ per bit moved from on-core SRAM to the core's controller, which is per bit rather than per
//! synapse and, as the next paragraph says, already inside `TrueNorth`'s published 26 pJ. If you
//! have one — measured, for a stated device, at a stated boundary — supply it on a per-operation
//! table, with prices for the other terms your workload uses, and the ledger will price your
//! workload.
//!
//! Not every published per-event energy is a per-operation figure, and the other kind fails the
//! other way. `TrueNorth`'s 26 pJ is total chip energy divided by synaptic events at one operating
//! point, so memory reads, routing, leakage and neuron updates are already inside it
//! ([`TRUENORTH_2014`]). Adding a fetch price to it counts the fetch twice, and applying it to a
//! workload with a different firing rate, synaptic density or spike distance assumes that workload
//! sits at the same point. [`Scope`] records which kind a table holds, and [`Ledger::bill`] never
//! charges separately for a term that a table's figure already contains.
//!
//! # What this crate does NOT claim about the lane
//!
//! Measured joules for spiking workloads are not unheard of, and saying so would be false. The
//! **`NeuroBench`** system track mandates them and publishes them. Yik et al., *The neurobench
//! framework for benchmarking neuromorphic computing algorithms and systems*, Nature
//! Communications 16:1545 (2025), doi:10.1038/s41467-025-56739-4, Table 6, gives `SynSense` Xylo
//! Audio 2 at 0.028 mJ of dynamic energy per inference against an Arduino Nano 33 BLE at 0.934 mJ,
//! with idle, active and dynamic power reported separately and the analog front end priced on its
//! own. The two sides are not metered at the same boundary, and the table's note says so: "CPU
//! power is measured over the full Arduino system ... while Xylo power measures power consumed by
//! the Xylo Audio 2 ASIC only." **`Rockpool`**'s `XyloSamna(device, config,
//! power_frequency=5.0).evolve(input, record_power=True)` returns per-rail watts from the board —
//! `io_power`, `snn_core_power`, `afe_core_power` and `afe_ldo_power` in its record dictionary —
//! and **`BrainChip`'s** tooling divides on-SoC power-meter samples by frames. Those are real, and
//! they are more than this crate does: it measures nothing, and has no hardware.
//!
//! Through 0.22.0 the `Rockpool` call above was written `XyloSamna(record_power=True)`, which is
//! not a valid call. In `rockpool/devices/xylo/syns61201/xylo_samna.py` on the `develop` branch
//! (last changed at `869761016e`), `record_power` is a keyword of `evolve()`; the constructor
//! requires a `device`, takes `power_frequency` (default 5 Hz) for the sampling rate, and would
//! pass a stray `record_power` into `**kwargs`, where nothing reads it.
//!
//! What this review did not locate is narrower and survives: **a published per-synapse
//! memory-FETCH energy for any commercially available neuromorphic processor**, and any study that
//! instrumented a DRAM rail while running a spiking network and reconciled the reading against the
//! models. Models of the memory term exist, and the shares quoted from them need their
//! qualifiers. `EnforceSNN` — Putra, Hanif and Shafique, *`EnforceSNN`: Enabling resilient and
//! energy-efficient spiking neural network inference considering approximate DRAMs for embedded
//! systems*, Frontiers in Neuroscience 16:937782 (2022), doi:10.3389/fnins.2022.937782 — puts
//! memory access at "50–75% of the total system energy across different SNN hardware platforms"
//! (PEASE, SNNAP and `TrueNorth`, its Fig. 1B), a breakdown its caption says is "adapted from
//! studies in Krithivasan et al. (2019)": Krithivasan, Sen, Venkataramani and Raghunathan,
//! *Dynamic Spike Bundling for Energy-Efficient Spiking Neural Networks*, ISLPED 2019,
//! doi:10.1109/ISLPED.2019.8824897, whose text this review did not read. `DRAMPower` is
//! `EnforceSNN`'s tool for its own DRAM access energy, not the source of that range. SATA — Yin,
//! Moitra, Bhattacharjee, Kim and Panda, *SATA: Sparsity-Aware Training Accelerator for Spiking
//! Neural Networks*, IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems
//! 42(6):1926–1938 (2023), doi:10.1109/TCAD.2022.3213211 — models BPTT *training* of VGG5 on
//! CIFAR10 at T = 8 in 65 nm with its `SATA-Sim` tool and memories simulated in `CACTI`. Its 78%
//! is "the DRAM access energy of filters (78% of the total memory energy ...)": a share of MEMORY
//! energy, not of the bill. Its own ratios, 1.35x total, 3.28x compute and 1.28x memory for the
//! SNN over the ANN, put memory at about 91% of the SNN's training energy by this crate's
//! arithmetic, and anywhere from 90% to 93% across the rounding of those three ratios. SATA gives
//! no memory share of the bill for SNN inference; `EnforceSNN` quotes one rather than producing it
//! from its own model; and this review did not locate either checked against a meter.
//!
//! Through 0.22.0 the paragraph above said "`SATA_Sim` and `EnforceSNN` put memory at 50–78% of
//! the bill from `CACTI` and `DRAMPower`", with no citation for either. The 50 is `EnforceSNN`'s,
//! its top is 75, and it is quoted from Krithivasan et al.; the 78 is SATA's within-memory share
//! for training, and does not extend the range.
//!
//! ⚠ **How badly the unchecked models disagree is measurable, and it is worse than it sounds.**
//! `SpikingJelly` ships its own cross-validation of energy models: its
//! `benchmark/energy_model_validation.py` writes
//! `docs/source/_static/tutorials/op_counter/energy_model_cross_validation.csv`, read here at
//! commit `4c98a4a3f3` (2026-09-24; the file last changed at `a688b40`, 2026-08-30). It holds 28
//! cases and five columns in pJ — `Simple`, `Lemaire`, `NeuroMC`, and `SpikeSim` in its dense and
//! its event mode, so four models, one of them twice. On identical workloads the largest column
//! over the smallest has a **median of 556, spanning 218x (VGG-11-56) to 753x
//! (`SEW-ResNet-50-32`)** — one network, VGG-16-32, priced at 553.4 µJ by `NeuroMC` and 1.13 µJ by
//! `SpikeSim`'s event mode. Four models, one workload, a factor of up to 753. That is the state of
//! the art this ledger declines to add another entry to. Through 0.22.0 this paragraph gave no file
//! or commit and counted the five columns as five published methods.
//!
//! # The flattering number is still available, and it is labelled
//!
//! [`Ledger::joules_synops_only`] computes exactly the figure the literature reports, ignoring the
//! fetch term — or, against an all-in table, that table's whole-chip figure at its operating point.
//! It is there because a user needs to reproduce published numbers to argue with them, and because
//! hiding it would not make anybody stop using it. Its doc says what it omits, and [`Bill`] names
//! every term that went unpriced so a caller cannot report the total without also having been
//! handed the list of what is missing from it.

use core::fmt;

/// What kind of evidence a price table stands on.
///
/// Typed rather than left to the prose, so that a comparison across grades is something the
/// compiler's user can catch rather than something a careful reader has to notice. Mixing a
/// `Metered` number with a `Projected` one and reporting the ratio is the standard way a
/// neuromorphic speedup claim comes to be meaningless.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Evidence {
    /// No published or measured figure at all. The weakest grade, and the honest one.
    Unstated,
    /// A roadmap number or design target for hardware that does not exist yet.
    Projected,
    /// Computed analytically from stated parameters. Nothing was instrumented.
    Derived,
    /// Circuit simulation of a design — SPICE or equivalent. No silicon was measured.
    Simulated,
    /// Measured on physical hardware, without a fully stated measurement protocol.
    Measured,
    /// Metered on physical silicon with the protocol stated: instrument, baseline, and a
    /// reproduced control. The strongest grade, and the rarest.
    Metered,
}

impl fmt::Display for Evidence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Unstated => "unstated",
            Self::Projected => "projected",
            Self::Derived => "derived",
            Self::Simulated => "simulated",
            Self::Measured => "measured",
            Self::Metered => "metered",
        };
        f.write_str(s)
    }
}

/// The weaker of two grades.
///
/// A figure built from two sources is only as good as its worst input, and this is the rule that
/// says so in one place rather than in every call site that combines them.
#[must_use]
pub fn weaker(a: Evidence, b: Evidence) -> Evidence {
    if a <= b { a } else { b }
}

/// What a table's [`Prices::e_syn_op`] covers.
///
/// Two kinds of figure are published under one name, "energy per synaptic operation", and they
/// fail in opposite directions when taken for each other. A per-operation figure prices the
/// operation alone and leaves the fetch, the neuron update, the routing and the readout to be
/// priced apart; reading it as a whole bill understates. An all-in figure is total chip energy
/// divided by synaptic events at one operating point; pricing a fetch or a hop on top of it counts
/// that term twice, and applying it to a different workload assumes the workload sits at the same
/// point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scope {
    /// `e_syn_op` is the synaptic operation alone. Every other term is priced by its own field.
    PerTerm,
    /// `e_syn_op` is total chip energy over synaptic events at one stated operating point.
    ///
    /// The synapse fetch, the neuron update and the spike emission are inside it, so this module
    /// never charges them separately: a price supplied for one of them is not used, and the term is
    /// listed as unpriced whenever it has work, because the aggregate cannot be split to reprice a
    /// workload whose mix differs from the operating point's. Readout off the chip is outside it
    /// and is charged from [`Prices::e_read`] as usual.
    AllIn,
}

/// Per-operation energies for one device model, in joules. `None` means **nobody has published
/// this number**, which is a different statement from zero.
///
/// The distinction is the entire point of the type. A `f64` field defaulting to `0.0` would let an
/// unpriced term vanish into a sum and make the total look complete; an `Option` forces the caller
/// to confront it, and [`Ledger::joules`] refuses rather than guessing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Prices {
    /// One synaptic operation: a spike arriving at a synapse and accumulating into the
    /// post-synaptic neuron's state. This is the term the literature calls a SOP and the only one
    /// most vendors publish.
    ///
    /// What the number covers is [`Prices::scope`]: the operation alone, or — for a figure computed
    /// as total chip energy over synaptic events — everything the chip did at one operating point.
    pub e_syn_op: f64,
    /// **Reading one synapse's weight out of memory so that the operation above can happen.**
    ///
    /// `None` in every table in this module, because this review did not locate a published figure
    /// for it on any commercial part. Under [`Scope::AllIn`] the fetch is inside `e_syn_op`, and a
    /// value here is not charged. It is separated from `e_syn_op` rather than folded into it
    /// because the two scale with different things: the operation count is a property of the
    /// network's activity, and the fetch cost is a property of where the weights live, which
    /// changes with the model's size and not with its firing rate. A chip whose synapses fit
    /// on-core and the same chip running a model that spills are the same `e_syn_op` and a
    /// different bill.
    pub e_syn_fetch: Option<f64>,
    /// One membrane-potential update of one neuron on a clock tick.
    ///
    /// The cost event-driven designs exist to avoid, and therefore the term that decides whether
    /// they succeeded. A clock-driven simulation pays it for every neuron on every tick whether or
    /// not anything happened; an event-driven one pays it only for neurons that received a spike.
    /// [`Ledger`] counts them separately so the difference is a number rather than an argument.
    pub e_neuron_update: f64,
    /// One spike emitted and routed to its targets.
    pub e_spike_out: Option<f64>,
    /// One neuron's state read out to the chip edge — the host's view of the answer.
    ///
    /// Frequently the largest single term in a small workload and almost always omitted from a
    /// published figure, for the same reason it was omitted in the sibling crate until five
    /// hand-written collection loops were found each reporting their readback as exactly zero.
    pub e_read: Option<f64>,
    /// WHAT these numbers describe and where they came from.
    ///
    /// Not documentation. A joules figure is a claim about a machine, and a `Prices` without a
    /// subject can be applied to any machine at all.
    pub source: &'static str,
    /// What kind of evidence [`Prices::source`] describes.
    pub evidence: Evidence,
    /// What [`Prices::e_syn_op`] covers: the operation alone, or the whole chip at one operating
    /// point. See [`Scope`].
    pub scope: Scope,
}

impl Prices {
    /// Prices for a device nobody has characterised: every term `None`.
    ///
    /// The default, and deliberately so. A crate whose default price table was some real chip's
    /// numbers would have every user's laptop reporting that chip's energy, which is exactly the
    /// failure the sibling crate spent a release fixing.
    pub const UNSTATED: Self = Self {
        e_syn_op: f64::NAN,
        e_syn_fetch: None,
        e_neuron_update: f64::NAN,
        e_spike_out: None,
        e_read: None,
        source: "no device. Nothing here has been priced; ask a device model for its numbers.",
        evidence: Evidence::Unstated,
        scope: Scope::PerTerm,
    };

    /// Whether every term needed for a complete bill has a number.
    ///
    /// False for every table currently in this module, because none of them prices a synapse
    /// fetch. That is the finding, not a defect in the type. Also false for any [`Scope::AllIn`]
    /// table, whose fetch, neuron update and spike emission cannot be priced apart from its
    /// aggregate. Defined as an empty [`Prices::unpriced`] so that the two cannot disagree.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.unpriced().is_empty()
    }

    /// The terms this table cannot price, by name, in the order a bill would list them.
    ///
    /// Under [`Scope::AllIn`] that always includes the synapse fetch, the neuron update and the
    /// spike emission, whatever their fields hold: they are inside `e_syn_op`, not beside it.
    #[must_use]
    pub fn unpriced(&self) -> Vec<&'static str> {
        let mut v = Vec::new();
        if !self.e_syn_op.is_finite() {
            v.push("synaptic operation");
        }
        if !priced(self.separate(self.e_syn_fetch)) {
            v.push("synapse memory fetch");
        }
        if !priced(self.separate(finite(self.e_neuron_update))) {
            v.push("neuron update");
        }
        if !priced(self.separate(self.e_spike_out)) {
            v.push("spike emission");
        }
        if !priced(self.e_read) {
            v.push("state readout");
        }
        v
    }

    /// The price this table gives a term that an all-in `e_syn_op` already contains: the synapse
    /// fetch, the neuron update or the spike emission.
    ///
    /// `None` under [`Scope::AllIn`] whatever the field holds, so that no path through this module
    /// can add such a term on top of a figure that includes it.
    fn separate(&self, price: Option<f64>) -> Option<f64> {
        match self.scope {
            Scope::PerTerm => price,
            Scope::AllIn => None,
        }
    }
}

/// `TrueNorth`'s published energy per synaptic event — **an all-in figure, which this crate read as
/// a per-operation one through 0.22.0.**
///
/// Merolla et al., *A million spiking-neuron integrated circuit with a scalable communication
/// network and interface*, Science 345(6197):668–673, 2014. The fabricated 28 nm part, measured on
/// silicon, and graded accordingly.
///
/// ⛔ **What the 26 pJ is.** Total chip power divided by the synaptic-event rate, at one operating
/// point: "At the operating point where neurons fire on average at 20Hz and have 128 active
/// synapses, the total measured power was 72mW (at 0.775V operating voltage), corresponding to 26pJ
/// per synaptic event (considering total energy)." The caption of Fig. 4C calls it "the total
/// energy (passive plus active) per synaptic event", which "decreases with higher synaptic density
/// because leakage power and baseline core power are amortized over additional synapses". The
/// benchmark networks behind it (supplementary section S7) were built "to push `TrueNorth`'s power
/// consumption as high as possible for a given spike rate and synaptic density", with spikes sent
/// "21.3 cores away on average in both x and y dimensions". The arithmetic agrees: 72 mW over the
/// 4,096 cores of 256 neurons, firing at 20 Hz on average into 128 synapses each, is 26.8 pJ per
/// event, 3.2% above the 26 pJ printed — a gap this review does not resolve, since the paper's own
/// inputs ("one million neurons", "on average at 20Hz") are rounded. On-core SRAM reads, spike
/// routing, leakage and neuron updates are therefore inside the 26 pJ; only readout off the chip is
/// plausibly outside it.
///
/// **What this crate used to say.** Through 0.22.0 this table read the 26 pJ as the price of the
/// synaptic event alone, said this review "did not locate a separately published memory-fetch
/// energy, spike-routing energy or host-readout energy for the part", and left those terms as the
/// missing pieces of a bill. Both halves were wrong. A fetch or a hop priced on top of this figure
/// is counted twice. And the supplementary material does publish the pieces, as active energies
/// measured apart (section S5 and Fig. S4): about 47 fJ per bit from on-core SRAM to the core's
/// controller, 72 fJ per bit between adjacent cores, 2.3 pJ to send a spike one hop, and 2 pJ per
/// bit from the internal to the external periphery.
///
/// **What changed.** The table is [`Scope::AllIn`]. A bill against it charges `syn_ops` at 26 pJ
/// and never charges a fetch, a neuron update or a spike emission on top, even when a caller
/// supplies a price for one. Those terms still stop the total, because the aggregate holds at its
/// own operating point and cannot be split to reprice a workload with a different rate, density or
/// spike distance. The per-bit and per-hop figures above are not turned into prices: a per-synapse
/// fetch or a per-spike emission built from them needs a bits-per-synapse figure or a hop count
/// this crate would have to assume, and it would still be inside the 26 pJ. `e_read` stays `None`
/// for a like reason: the 2 pJ is per bit, and a per-readout price needs a readout width this crate
/// would have to assume.
pub const TRUENORTH_2014: Prices = Prices {
    e_syn_op: 26e-12,
    e_syn_fetch: None,
    e_neuron_update: f64::NAN,
    e_spike_out: None,
    e_read: None,
    source: "TrueNorth 28 nm, 26 pJ per synaptic event — Merolla et al., Science 345(6197), 2014. \
             Silicon, measured. ALL-IN: total chip energy (passive plus active) over synaptic \
             events at one operating point — 20 Hz, 128 active synapses, spikes travelling 21.3 \
             cores on average, 0.775 V — so on-core SRAM reads, spike routing, leakage and neuron \
             updates are inside it and only off-chip readout is plausibly outside. Applies to that \
             part at that point and to nothing else.",
    evidence: Evidence::Measured,
    scope: Scope::AllIn,
};

/// Loihi's published per-synaptic-operation energy — **and this crate graded it wrong first time.**
///
/// Davies et al., *Loihi: A Neuromorphic Manycore Processor with On-Chip Learning*, IEEE Micro
/// 38(1):82–99, 2018, reports ~23.6 pJ per synaptic operation for the 14 nm part. This crate
/// shipped that figure at [`Evidence::Measured`] in 0.1.0 and 0.2.0.
///
/// ⛔ **It is not measured. The table it comes from is captioned "pre-silicon" and the figure is
/// sourced from pre-silicon SDF and SPICE simulations.** No Loihi was on a meter. The grade here is
/// now [`Evidence::Simulated`], which is what the primary source supports.
///
/// The error is worth leaving visible rather than quietly correcting, because it is exactly the
/// failure this module exists to prevent, committed inside the module that exists to prevent it.
/// 23.6 pJ/SynOp is cited across 2024–2026 as a measured number by people who did not open Table 2,
/// and this crate joined them on its first release. An `Evidence` enum does not help if the value
/// assigned to it is taken from the citing literature instead of the cited table.
///
/// ⚠ **The direction of the error, corrected.** Through 0.22.0 this paragraph said that
/// per-synaptic-operation energies measured on silicon "sit well ABOVE this simulated figure", with
/// `TrueNorth` at 26 pJ and "ODIN and Darwin3 reported in the same band or higher relative to their
/// nodes", so that a pre-silicon benchmark flattered the field. The primaries say otherwise.
///
/// - ODIN ([`crate::hardware::ODIN`]): Frenkel, Lefebvre, Legat and Bol, *A 0.086-mm² 12.7-pJ/SOP
///   64k-Synapse 256-Neuron Online-Learning Digital Spiking Neuromorphic Processor in 28-nm CMOS*,
///   IEEE Transactions on Biomedical Circuits and Systems 13(1):145–158 (2019),
///   doi:10.1109/TBCAS.2018.2880425. Measured in 28 nm FDSOI, it "consumes a minimum energy per
///   synaptic operation (SOP) of 12.7pJ at 0.55V": the global energy per SOP at its maximum SOP
///   rate, leakage and idle power included. That is about 46% below 23.6 pJ and about half
///   `TrueNorth`'s all-in 26 pJ at the same node, and the incremental energy with leakage and idle
///   power subtracted, 8.43 pJ, is lower still. Only its leakage-dominated figure at biological
///   time, 54 pJ, is above.
/// - Darwin3 ([`crate::hardware::DARWIN3`]): Ma, Jin, Sun et al., *Darwin3: a large-scale
///   neuromorphic chip with a novel ISA and on-chip learning*, National Science Review
///   11(5):nwae102 (2024), doi:10.1093/nsr/nwae102. In 22 nm FDSOI at 0.8 V and 333 MHz, "The
///   measured average SOP power consumption is 5.47 pj/SOP" — about 4.3 times lower, and an
///   incremental term, with baseline and per-neuron power kept apart in its Eq. 7.
/// - `TrueNorth`'s 26 pJ is all-in ([`TRUENORTH_2014`]), so it is not a per-operation figure to
///   set beside this one at all.
///
/// Measured silicon straddles 23.6 pJ; it does not sit above it. What survives is narrower: the
/// Loihi row is a minimum — "Energy per synaptic spike op (min)" — and ODIN's authors, comparing
/// against it, call it "a lower bound as it includes only the contribution of the synaptic
/// operation". A benchmark against it is a benchmark against a simulated lower bound on one term,
/// not against a chip.
///
/// Neither ODIN nor Darwin3 is given a `Prices` constant here. That would be a new row in
/// [`CATALOGUE`] with a grading argument of its own; this note exists to get the direction of the
/// comparison right.
///
/// **What else Table 2 publishes.** Through 0.22.0 the source line below ended "The fetch, routing
/// and readout terms were not published at all." Table 2 prices a within-tile spike at 1.7 pJ, a
/// tile hop at 3.0 pJ east-west and 4.0 pJ north-south, and a neuron update at 81 pJ active and
/// 52 pJ inactive, all pre-silicon at 0.75 V. It has no row for a synaptic-memory fetch or a host
/// readout. The two it does publish are not priced here. [`Prices`] has one neuron-update field,
/// and this review did not locate a definition of "active" and "inactive" in the paper to map onto
/// this ledger's driven and idle updates. A per-spike emission is 1.7 pJ plus 3.0 or 4.0 pJ per
/// hop, so it needs a hop count, which is placement ([`crate::mapping`]) rather than a table
/// constant. A per-hop energy belongs with the fabric price tables there, such as
/// [`crate::mapping::LOIHI_FABRIC`], which [`crate::mapping::SpikeHops::bill`] multiplies by a
/// counted hop total.
pub const LOIHI_2018: Prices = Prices {
    e_syn_op: 23.6e-12,
    e_syn_fetch: None,
    e_neuron_update: f64::NAN,
    e_spike_out: None,
    e_read: None,
    source: "Loihi 14 nm, ~23.6 pJ per synaptic operation — Davies et al., IEEE Micro 38(1), 2018, \
             Table 2, CAPTIONED \"pre-silicon\" and sourced from pre-silicon SDF and SPICE \
             simulations; the row is a minimum, \"(min)\". Not a measurement of any fabricated \
             part, despite being cited as one throughout 2024-2026. The same table gives a \
             within-tile spike at 1.7 pJ, a tile hop at 3.0 pJ (E-W) or 4.0 pJ (N-S) and a neuron \
             update at 81 pJ active or 52 pJ inactive, all at 0.75 V; it has no synaptic-memory \
             fetch or host-readout row.",
    evidence: Evidence::Simulated,
    scope: Scope::PerTerm,
};

/// Every device table in this crate, for a caller that wants to sweep them.
///
/// Three entries; one is unstated, one is a 2014 all-in measurement and one is a 2018 SIMULATION.
/// Exactly one number in this crate was taken on fabricated silicon.
///
/// That shortness is the state of the field as this review found it. Per-operation energies for the
/// current commercial parts are quoted in marketing units — TOPS/W, "1000x more efficient" — that
/// do not reduce to a per-operation joule anybody can tabulate, and the one figure the field does
/// quote universally turns out to be pre-silicon. See [`LOIHI_2018`].
pub const CATALOGUE: [(&str, Prices); 3] = [
    ("unstated", Prices::UNSTATED),
    ("truenorth-2014", TRUENORTH_2014),
    ("loihi-2018", LOIHI_2018),
];

/// Exact counts of everything a spiking workload did.
///
/// Counts are integers and are exact; only their conversion to joules is uncertain, and that
/// uncertainty lives entirely in [`Prices`]. Keeping the two apart is what lets the same run be
/// priced against several devices without being re-run.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Ledger {
    /// Spikes delivered across synapses — the SOP count.
    pub syn_ops: u64,
    /// Synapse weights read from memory. Equal to `syn_ops` on a design that fetches per delivery,
    /// and smaller on one that caches or batches — which is why it is counted rather than assumed.
    pub syn_fetches: u64,
    /// Membrane updates performed on a clock tick for a neuron that received nothing.
    ///
    /// Counted apart from [`Ledger::neuron_updates_driven`] because the whole argument for
    /// event-driven hardware is that this number should be zero, and an argument whose key
    /// quantity is not measured is not an argument.
    pub neuron_updates_idle: u64,
    /// Membrane updates performed for a neuron that received at least one spike this tick.
    pub neuron_updates_driven: u64,
    /// Spikes emitted and routed.
    pub spikes_out: u64,
    /// Neuron states read out to the host.
    pub reads: u64,
}

/// A priced ledger, with the terms that could not be priced named.
///
/// Returned instead of a bare number so that a caller cannot obtain a total without also being
/// handed the list of what is missing from it. Reporting `total` while ignoring `unpriced` is still
/// possible; doing it accidentally is not.
#[derive(Debug, Clone, PartialEq)]
pub struct Bill {
    /// Joules for the terms that had prices, or `None` if any charged term did not.
    pub total: Option<f64>,
    /// Joules attributable to synaptic operations — or, against a [`Scope::AllIn`] table, `syn_ops`
    /// times that table's whole-chip figure at its operating point.
    pub synaptic: Option<f64>,
    /// Joules attributable to synapse memory fetches — the term a per-operation figure omits.
    /// `None` whenever fetches were counted against a [`Scope::AllIn`] table, which contains them.
    pub fetch: Option<f64>,
    /// Joules attributable to membrane updates, idle and driven together.
    pub neurons: Option<f64>,
    /// Joules attributable to spike emission and routing.
    pub routing: Option<f64>,
    /// Joules attributable to host readout.
    pub readout: Option<f64>,
    /// Names of the terms that had work to price but no price for it. Against a [`Scope::AllIn`]
    /// table these include the terms inside its aggregate, which it cannot price separately.
    pub unpriced: Vec<&'static str>,
    /// The weakest grade among the prices actually used.
    pub evidence: Evidence,
}

impl fmt::Display for Bill {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.total {
            Some(j) => write!(f, "{j:.4e} J ({})", self.evidence)?,
            None => write!(f, "REFUSED — no total ({})", self.evidence)?,
        }
        if !self.unpriced.is_empty() {
            write!(f, "; unpriced: {}", self.unpriced.join(", "))?;
        }
        Ok(())
    }
}

impl Ledger {
    /// Total membrane updates, idle and driven.
    #[must_use]
    pub fn neuron_updates(&self) -> u64 {
        self.neuron_updates_idle + self.neuron_updates_driven
    }

    /// The fraction of membrane updates that did no work.
    ///
    /// `None` when nothing was updated at all. This is the quantity that decides whether an
    /// event-driven claim is true for a given workload: a network whose neurons are mostly idle
    /// and whose simulator updates them anyway is paying a clock-driven bill and calling it
    /// event-driven.
    #[must_use]
    pub fn idle_fraction(&self) -> Option<f64> {
        let n = self.neuron_updates();
        if n == 0 {
            return None;
        }
        Some(self.neuron_updates_idle as f64 / n as f64)
    }

    /// Price this workload against a device model.
    ///
    /// `Bill::total` is `None` if any term with a non-zero count has no price. A term with a zero
    /// count is not charged and its missing price is not held against the total — a workload that
    /// never read anything back does not need a readout price.
    ///
    /// Against a [`Scope::AllIn`] table the synapse fetch, the neuron update and the spike emission
    /// have no separate price, whatever the table's fields hold: they are inside its `e_syn_op`, and
    /// charging them again would count them twice.
    ///
    /// # Panics
    ///
    /// Never. Every arithmetic path here is a multiply of a count by a finite price, guarded by the
    /// `is_finite` checks that decide whether the term is priceable at all.
    #[must_use]
    pub fn bill(&self, p: &Prices) -> Bill {
        let mut unpriced = Vec::new();
        let mut grade = Evidence::Metered;
        let mut any = false;

        // A term contributes only if it has work to price. `charge` returns None when there is work
        // and no price, which is what makes the total refuse.
        let mut charge = |count: u64, price: Option<f64>, name: &'static str| -> Option<f64> {
            if count == 0 {
                return Some(0.0);
            }
            match price {
                Some(e) if e.is_finite() => {
                    any = true;
                    grade = weaker(grade, p.evidence);
                    Some(count as f64 * e)
                }
                _ => {
                    unpriced.push(name);
                    None
                }
            }
        };

        let synaptic = charge(self.syn_ops, finite(p.e_syn_op), "synaptic operation");
        let fetch = charge(self.syn_fetches, p.separate(p.e_syn_fetch), "synapse memory fetch");
        let neurons =
            charge(self.neuron_updates(), p.separate(finite(p.e_neuron_update)), "neuron update");
        let routing = charge(self.spikes_out, p.separate(p.e_spike_out), "spike emission");
        let readout = charge(self.reads, p.e_read, "state readout");

        let total = match (synaptic, fetch, neurons, routing, readout) {
            (Some(a), Some(b), Some(c), Some(d), Some(e)) => Some(a + b + c + d + e),
            _ => None,
        };
        if !any {
            grade = Evidence::Unstated;
        }
        Bill { total, synaptic, fetch, neurons, routing, readout, unpriced, evidence: grade }
    }

    /// Joules, or `None` when any charged term is unpriced.
    ///
    /// The short form of [`Ledger::bill`], for a caller that wants the refusal and not the
    /// breakdown.
    #[must_use]
    pub fn joules(&self, p: &Prices) -> Option<f64> {
        self.bill(p).total
    }

    /// **The figure the literature reports**: synaptic operations only, priced and summed.
    ///
    /// Against a [`Scope::PerTerm`] table that is the arithmetic, with memory traffic, neuron
    /// updates, routing and readout omitted. Against a [`Scope::AllIn`] table such as
    /// [`TRUENORTH_2014`] it is something else: `syn_ops` times a whole-chip figure, which already
    /// includes on-chip memory reads, routing, leakage and neuron updates at that table's operating
    /// point and omits only readout off the chip. Through 0.22.0 this doc said the first of those
    /// about every table.
    ///
    /// This is here so that published numbers can be reproduced and argued with, not because it is
    /// a good way to price a workload. On a per-operation table it is systematically optimistic by
    /// construction — it charges for the arithmetic and sets the data movement that feeds the
    /// arithmetic to zero — and on a model whose synapses do not fit on-core, the omitted term can
    /// be the larger one. On an all-in table its error is the operating point: the figure holds
    /// only for a workload at the rate, density and spike distance it was measured at.
    ///
    /// Use [`Ledger::bill`] for a number you intend to defend.
    #[must_use]
    pub fn joules_synops_only(&self, p: &Prices) -> Option<f64> {
        if !p.e_syn_op.is_finite() {
            return None;
        }
        Some(self.syn_ops as f64 * p.e_syn_op)
    }

    /// How much larger a complete bill is than the synapse-only figure, as a ratio.
    ///
    /// `None` when either side has no answer. Against a per-operation table this is the quantity
    /// that says how much a published SOP-counted figure understated a given workload, and it is
    /// the number this crate exists to make computable. Against a [`Scope::AllIn`] table the only
    /// term a bill can add is readout, so the ratio measures what the aggregate left off the chip;
    /// the fetch and routing already inside it are never added a second time.
    #[must_use]
    pub fn understatement(&self, p: &Prices) -> Option<f64> {
        let full = self.joules(p)?;
        let partial = self.joules_synops_only(p)?;
        if partial <= 0.0 {
            return None;
        }
        Some(full / partial)
    }
}

/// `Some(x)` for a finite `x`, `None` for `NaN` — the bridge between a price stored as `f64::NAN`
/// to mean "unstated" and the `Option` that the charging path and [`Prices::unpriced`] work in.
fn finite(x: f64) -> Option<f64> {
    if x.is_finite() { Some(x) } else { None }
}

/// Whether an optional price is a usable number.
///
/// `Some(NaN)` is a price-shaped hole: [`Ledger::bill`] declines to charge it and lists the term as
/// unpriced, so [`Prices::is_complete`] and [`Prices::unpriced`] have to agree with that. They used
/// to ask only whether the `Option` was inhabited, which meant a table carrying a `NaN` called
/// itself complete while a bill built from it refused — the two answering different questions under
/// the same name.
fn priced(x: Option<f64>) -> bool {
    matches!(x, Some(e) if e.is_finite())
}

#[cfg(test)]
mod tests {
    use super::{CATALOGUE, Evidence, LOIHI_2018, Ledger, Prices, Scope, TRUENORTH_2014, weaker};

    /// A table with every term priced, so that dropping any ONE of them is visible. Nothing in
    /// this crate is complete, which is the finding — so the only way to test completeness is to
    /// build a table that would be.
    const COMPLETE: Prices = Prices {
        e_syn_op: 2e-12,
        e_syn_fetch: Some(3e-12),
        e_neuron_update: 5e-12,
        e_spike_out: Some(7e-12),
        e_read: Some(11e-12),
        source: "a table invented by this test so that completeness has something to be true of.",
        evidence: Evidence::Derived,
        scope: Scope::PerTerm,
    };

    /// Thirteen of this module's mutations survived its first recorded audit, and they were all in
    /// the same three places: the evidence ladder's ORDER, what `is_complete` and `unpriced` say
    /// about each individual term, and whether a bill's grade comes from the price it used. This
    /// test covers the ladder.
    #[test]
    fn the_evidence_ladder_runs_from_unstated_to_metered_and_weaker_picks_the_lower() {
        // The ordering IS the semantics: `weaker` is a comparison, so a ladder in the wrong order
        // silently promotes a claim. Written out in full rather than derived from the enum.
        let ladder = [
            Evidence::Unstated,
            Evidence::Projected,
            Evidence::Derived,
            Evidence::Simulated,
            Evidence::Measured,
            Evidence::Metered,
        ];
        for pair in ladder.windows(2) {
            assert!(pair[0] < pair[1], "{:?} should be weaker than {:?}", pair[0], pair[1]);
        }
        assert_eq!(*ladder.iter().min().unwrap(), Evidence::Unstated);
        assert_eq!(*ladder.iter().max().unwrap(), Evidence::Metered);
        // `weaker` returns the lower of the two, whichever way round it is handed them.
        for (i, a) in ladder.iter().enumerate() {
            for (j, b) in ladder.iter().enumerate() {
                let want = ladder[i.min(j)];
                assert_eq!(weaker(*a, *b), want, "weaker({a:?}, {b:?})");
            }
        }
        assert_eq!(weaker(Evidence::Metered, Evidence::Projected), Evidence::Projected);
        assert_eq!(weaker(Evidence::Unstated, Evidence::Metered), Evidence::Unstated);
    }

    #[test]
    fn completeness_and_the_unpriced_list_answer_for_each_term_separately() {
        assert!(COMPLETE.is_complete());
        assert_eq!(COMPLETE.unpriced(), Vec::<&str>::new());
        // Drop each term in turn: every one of the five is load-bearing, and the name it reports
        // is the name of the term that went.
        let cases: [(Prices, &str); 5] = [
            (Prices { e_syn_op: f64::NAN, ..COMPLETE }, "synaptic operation"),
            (Prices { e_syn_fetch: None, ..COMPLETE }, "synapse memory fetch"),
            (Prices { e_neuron_update: f64::NAN, ..COMPLETE }, "neuron update"),
            (Prices { e_spike_out: None, ..COMPLETE }, "spike emission"),
            (Prices { e_read: None, ..COMPLETE }, "state readout"),
        ];
        for (p, name) in cases {
            assert!(!p.is_complete(), "a table without a {name} price called itself complete");
            assert_eq!(p.unpriced(), vec![name], "the unpriced list should name exactly the {name}");
        }
        // The list is in bill order, and it names only what is missing.
        let bare = Prices { e_syn_fetch: None, e_read: None, ..COMPLETE };
        assert_eq!(bare.unpriced(), vec!["synapse memory fetch", "state readout"]);
        assert_eq!(Prices::UNSTATED.unpriced().len(), 5, "the unstated table prices nothing at all");
    }

    #[test]
    fn every_table_carries_the_grade_its_source_earns_and_a_bill_is_graded_by_what_it_used() {
        // The exact grade of every table in the catalogue, not merely that one of them is
        // simulated. A table that quietly moved up the ladder would otherwise pass.
        assert_eq!(Prices::UNSTATED.evidence, Evidence::Unstated);
        assert_eq!(TRUENORTH_2014.evidence, Evidence::Measured, "TrueNorth is measured silicon, not metered");
        assert_eq!(LOIHI_2018.evidence, Evidence::Simulated, "Loihi's table is captioned pre-silicon");
        assert_eq!(CATALOGUE.len(), 3);
        // And what each table's per-event figure covers. TrueNorth's is "considering total
        // energy"; Loihi's row is the synaptic operation alone; the default claims neither.
        assert_eq!(TRUENORTH_2014.scope, Scope::AllIn, "TrueNorth's 26 pJ is total chip energy");
        assert_eq!(LOIHI_2018.scope, Scope::PerTerm, "Loihi's 23.6 pJ is one synaptic op");
        assert_eq!(Prices::UNSTATED.scope, Scope::PerTerm, "the default claims no aggregate");
        assert_eq!(CATALOGUE[0].1.evidence, Evidence::Unstated, "the default must price nothing");
        assert!(CATALOGUE[0].1.unpriced().len() == 5);
        // A bill's grade is the grade of the prices it actually used — not the Metered the
        // accumulator starts from.
        let led = Ledger { syn_ops: 10, ..Ledger::default() };
        assert_eq!(led.bill(&TRUENORTH_2014).evidence, Evidence::Measured);
        assert_eq!(led.bill(&LOIHI_2018).evidence, Evidence::Simulated);
        assert_eq!(led.bill(&COMPLETE).evidence, Evidence::Derived);
        // And a bill that priced nothing claims nothing.
        let nothing = Ledger::default();
        assert_eq!(nothing.bill(&TRUENORTH_2014).evidence, Evidence::Unstated);
    }

    #[test]
    fn a_price_that_is_not_a_number_is_not_a_price_and_every_term_reaches_the_total() {
        // `Some(NaN)` is a price-shaped hole. Charging it would put a NaN in the total, which
        // compares false against every threshold it is ever checked against.
        let poisoned = Prices { e_syn_fetch: Some(f64::NAN), ..COMPLETE };
        let led = Ledger { syn_ops: 10, syn_fetches: 10, ..Ledger::default() };
        let bill = led.bill(&poisoned);
        assert!(bill.total.is_none(), "a NaN price produced a total of {:?}", bill.total);
        assert!(bill.unpriced.contains(&"synapse memory fetch"));
        assert!(!poisoned.is_complete());
        // Every one of the five terms is in the total: build a workload that uses all of them and
        // check the total is the sum of the parts, each of which is a count times its price.
        let led = Ledger {
            syn_ops: 2,
            syn_fetches: 3,
            neuron_updates_idle: 5,
            neuron_updates_driven: 7,
            spikes_out: 11,
            reads: 13,
        };
        let bill = led.bill(&COMPLETE);
        assert_eq!(bill.synaptic, Some(2.0 * 2e-12));
        assert_eq!(bill.fetch, Some(3.0 * 3e-12));
        assert_eq!(bill.neurons, Some(12.0 * 5e-12), "idle and driven updates are both charged");
        assert_eq!(bill.routing, Some(11.0 * 7e-12));
        assert_eq!(bill.readout, Some(13.0 * 11e-12));
        let parts = 2.0 * 2e-12 + 3.0 * 3e-12 + 12.0 * 5e-12 + 11.0 * 7e-12 + 13.0 * 11e-12;
        assert!((bill.total.unwrap() - parts).abs() < 1e-30, "{:?} against {parts}", bill.total);
    }

    #[test]
    fn an_understatement_with_nothing_underneath_it_has_no_ratio() {
        // A workload that performed no synaptic operations has a synapse-only figure of exactly
        // zero, and the ratio of anything to zero is not a number this should report.
        let led = Ledger { syn_ops: 0, reads: 100, ..Ledger::default() };
        assert_eq!(led.joules_synops_only(&COMPLETE), Some(0.0));
        assert!(led.joules(&COMPLETE).is_some());
        assert_eq!(led.understatement(&COMPLETE), None, "a ratio over zero is not a ratio");
        // With work underneath it, the ratio is the full bill over the synapse-only one and is
        // greater than one whenever anything else was charged.
        let led = Ledger { syn_ops: 10, reads: 10, ..Ledger::default() };
        let r = led.understatement(&COMPLETE).unwrap();
        let want = (10.0 * 2e-12 + 10.0 * 11e-12) / (10.0 * 2e-12);
        assert!((r - want).abs() < 1e-12, "{r} against {want}");
        assert!(r > 1.0);
    }

    /// The finding, asserted. If someone later fills in a fetch price without a source, this test
    /// is where the argument has to happen.
    #[test]
    fn no_device_table_in_this_crate_prices_a_synapse_fetch() {
        for (name, p) in CATALOGUE {
            assert!(
                p.e_syn_fetch.is_none(),
                "{name} gained a synapse-fetch price; it needs a cited, graded source"
            );
            assert!(!p.is_complete(), "{name} claims to be complete");
        }
    }

    /// The refusal is the behaviour, not an error path.
    #[test]
    fn a_workload_with_fetches_and_no_fetch_price_has_no_total() {
        let led = Ledger { syn_ops: 1_000, syn_fetches: 1_000, ..Ledger::default() };
        assert!(led.joules(&TRUENORTH_2014).is_none());
        let bill = led.bill(&TRUENORTH_2014);
        assert!(bill.total.is_none());
        assert!(bill.unpriced.contains(&"synapse memory fetch"), "{:?}", bill.unpriced);
        // The term that IS priced still reports, so the refusal is informative rather than blank.
        assert!((bill.synaptic.unwrap() - 1_000.0 * 26e-12).abs() < 1e-18);
    }

    /// A term with no work is not charged and its missing price is not held against the total.
    /// Without this, every workload would refuse for want of a readout price it never used.
    #[test]
    fn an_unused_term_does_not_block_the_total() {
        let p = Prices {
            e_syn_op: 1e-12,
            e_syn_fetch: Some(2e-12),
            e_neuron_update: 3e-13,
            e_spike_out: None, // unpriced, but nothing was routed
            e_read: None,      // unpriced, but nothing was read
            source: "test",
            evidence: Evidence::Derived,
            scope: Scope::PerTerm,
        };
        let led = Ledger {
            syn_ops: 10,
            syn_fetches: 10,
            neuron_updates_driven: 4,
            spikes_out: 0,
            reads: 0,
            ..Ledger::default()
        };
        let want = 10.0 * 1e-12 + 10.0 * 2e-12 + 4.0 * 3e-13;
        assert!((led.joules(&p).unwrap() - want).abs() < 1e-24);
        assert!(led.bill(&p).unpriced.is_empty());
    }

    /// The point of the crate, as arithmetic: with a fetch as expensive as the operation, the
    /// honest bill is more than twice the published one.
    #[test]
    fn the_understatement_ratio_is_computable_and_greater_than_one() {
        let p = Prices {
            e_syn_op: 26e-12,
            e_syn_fetch: Some(26e-12),
            e_neuron_update: 1e-12,
            e_spike_out: Some(1e-12),
            e_read: Some(1e-11),
            source: "a hypothetical complete table, for the test only",
            evidence: Evidence::Derived,
            scope: Scope::PerTerm,
        };
        let led = Ledger {
            syn_ops: 1_000_000,
            syn_fetches: 1_000_000,
            neuron_updates_idle: 500_000,
            neuron_updates_driven: 100_000,
            spikes_out: 20_000,
            reads: 1_000,
        };
        let r = led.understatement(&p).expect("both sides priced");
        assert!(r > 2.0, "understatement was only {r}x");
        let synops = led.joules_synops_only(&p).unwrap();
        let full = led.joules(&p).unwrap();
        assert!(full > synops, "the complete bill was not larger");
    }

    /// The flattering number must remain available even when the honest one refuses — that is what
    /// makes reproducing a published figure possible.
    #[test]
    fn the_synops_only_figure_is_available_when_the_total_refuses() {
        let led = Ledger { syn_ops: 42, syn_fetches: 42, ..Ledger::default() };
        assert!(led.joules(&LOIHI_2018).is_none());
        let flattering = led.joules_synops_only(&LOIHI_2018).expect("SOPs are priced");
        assert!((flattering - 42.0 * 23.6e-12).abs() < 1e-21);
    }

    #[test]
    fn an_unstated_device_prices_nothing_and_says_so() {
        let led = Ledger { syn_ops: 5, ..Ledger::default() };
        assert!(led.joules(&Prices::UNSTATED).is_none());
        assert!(led.joules_synops_only(&Prices::UNSTATED).is_none());
        assert_eq!(led.bill(&Prices::UNSTATED).evidence, Evidence::Unstated);
    }

    /// Idle fraction is how an event-driven claim gets checked.
    #[test]
    fn the_idle_fraction_reports_what_a_clock_driven_run_wasted() {
        let led = Ledger {
            syn_ops: 0,
            syn_fetches: 0,
            neuron_updates_idle: 900,
            neuron_updates_driven: 100,
            spikes_out: 0,
            reads: 0,
        };
        assert!((led.idle_fraction().unwrap() - 0.9).abs() < 1e-15);
        assert!(Ledger::default().idle_fraction().is_none(), "nothing ran; there is no fraction");
    }

    /// ⛔ THE REGRESSION TEST FOR THIS CRATE'S OWN PUBLISHED DEFECT.
    ///
    /// 0.1.0 and 0.2.0 shipped `LOIHI_2018` at `Evidence::Measured`. The table it comes from is
    /// captioned "pre-silicon" and the figure is from SDF and SPICE simulation. The whole field
    /// cites it as measured; this crate joined them, inside the module written to stop exactly
    /// that. Pinned here so a future edit has to argue with a test rather than with a comment.
    #[test]
    fn the_loihi_figure_is_simulated_because_its_table_says_pre_silicon() {
        assert_eq!(
            LOIHI_2018.evidence,
            Evidence::Simulated,
            "Davies et al. 2018 Table 2 is captioned pre-silicon; no Loihi was on a meter"
        );
        assert!(LOIHI_2018.source.contains("pre-silicon"), "the caption must stay in the source line");
        // Through 0.22.0 the source line said "The fetch, routing and readout terms were not
        // published at all". Table 2 publishes routing and neuron-update energies; what it lacks is
        // a fetch and a readout row. The correction is pinned so the old sentence cannot return.
        assert!(!LOIHI_2018.source.contains("not published at all"), "{}", LOIHI_2018.source);
        for row in ["1.7 pJ", "3.0 pJ", "4.0 pJ", "81 pJ", "52 pJ"] {
            assert!(LOIHI_2018.source.contains(row), "Table 2's {row} row left the source line");
        }
    }

    /// ⛔ THE REGRESSION TEST FOR THE SECOND PUBLISHED DEFECT IN THIS MODULE'S TABLES.
    ///
    /// Through 0.22.0 `TRUENORTH_2014` read the 26 pJ as the synaptic event alone, with the fetch
    /// and the routing as missing terms. Merolla et al. compute it from total chip power: "the
    /// total measured power was 72mW ... corresponding to 26pJ per synaptic event (considering
    /// total energy)", at 20 Hz and 128 active synapses on all 4,096 cores of 256 neurons. The
    /// arithmetic is checked here, from those printed figures, rather than asserted.
    #[test]
    fn truenorths_figure_is_total_chip_power_over_synaptic_events() {
        let neurons = 4_096.0_f64 * 256.0;
        let events_per_second = neurons * 20.0 * 128.0;
        let all_in = 72e-3 / events_per_second;
        // 26.82 pJ against the 26 printed. A per-operation energy would sit BELOW total power over
        // the event rate by the leakage, baseline and routing it excludes; this one is that ratio,
        // to within the rounding of the paper's own inputs ("one million neurons", "on average at
        // 20Hz"). The residual is 3.16%, computed before the tolerance of 4% was chosen.
        assert!((all_in - 26.822e-12).abs() < 1e-15, "{all_in}");
        let residual = (all_in - TRUENORTH_2014.e_syn_op) / TRUENORTH_2014.e_syn_op;
        assert!((residual - 0.0316).abs() < 1e-4, "residual {residual}");
        assert!(residual.abs() < 0.04, "26 pJ is not total power over events: {residual}");
        assert_eq!(TRUENORTH_2014.scope, Scope::AllIn);
        assert!(TRUENORTH_2014.source.contains("passive plus active"), "{}", TRUENORTH_2014.source);
        assert!(TRUENORTH_2014.source.contains("0.775 V"), "the operating point left the source");
    }

    /// The double count, refused. An all-in figure already contains the fetch, the neuron update
    /// and the spike emission, so a price supplied for any of them is not charged on top, and the
    /// term still stops the total because the aggregate cannot be split to reprice it. Readout is
    /// outside the aggregate and is charged as usual.
    #[test]
    fn an_all_in_table_never_charges_what_its_figure_already_contains() {
        let all_in = Prices { scope: Scope::AllIn, ..COMPLETE };
        let inside = vec!["synapse memory fetch", "neuron update", "spike emission"];
        assert_eq!(all_in.unpriced(), inside, "every priced field, and still these three");
        assert!(!all_in.is_complete());
        assert!(COMPLETE.is_complete(), "the same fields on a per-term table are complete");

        // Every term at once: the contained three are refused, the other two are charged.
        let led = Ledger {
            syn_ops: 2,
            syn_fetches: 3,
            neuron_updates_idle: 5,
            neuron_updates_driven: 7,
            spikes_out: 11,
            reads: 13,
        };
        let bill = led.bill(&all_in);
        assert_eq!(bill.total, None);
        assert_eq!((bill.fetch, bill.neurons, bill.routing), (None, None, None));
        assert_eq!(bill.synaptic, Some(2.0 * 2e-12));
        assert_eq!(bill.readout, Some(13.0 * 11e-12), "readout is outside the aggregate");
        assert_eq!(bill.unpriced, inside);

        // Each contained term alone refuses the total, and names itself.
        let alone: [(Ledger, &str); 4] = [
            (Ledger { syn_ops: 4, syn_fetches: 4, ..Ledger::default() }, "synapse memory fetch"),
            (Ledger { syn_ops: 4, neuron_updates_idle: 4, ..Ledger::default() }, "neuron update"),
            (Ledger { syn_ops: 4, neuron_updates_driven: 4, ..Ledger::default() }, "neuron update"),
            (Ledger { syn_ops: 4, spikes_out: 4, ..Ledger::default() }, "spike emission"),
        ];
        for (led, name) in alone {
            let bill = led.bill(&all_in);
            assert_eq!(bill.total, None, "{name} was charged on top of an all-in figure");
            assert_eq!(bill.unpriced, vec![name]);
        }

        // With only the aggregate and the readout in play, the bill totals, and the understatement
        // is exactly what the aggregate left off the chip: (8·2 + 8·11) / (8·2) = 6.5.
        let led = Ledger { syn_ops: 8, reads: 8, ..Ledger::default() };
        assert_eq!(led.joules(&all_in), Some(8.0 * 2e-12 + 8.0 * 11e-12));
        let r = led.understatement(&all_in).expect("both sides priced");
        assert!((r - 6.5).abs() < 1e-12, "{r}");

        // The case the correction exists for: a caller who reads TrueNorth's 26 pJ as one
        // operation and supplies a fetch price. It is not added; the synaptic line stays at 26 pJ.
        let doubled = Prices { e_syn_fetch: Some(1e-12), ..TRUENORTH_2014 };
        let led = Ledger { syn_ops: 1_000, syn_fetches: 1_000, ..Ledger::default() };
        let bill = led.bill(&doubled);
        assert_eq!(bill.total, None);
        assert_eq!(bill.fetch, None, "a fetch was charged on top of an all-in figure");
        assert_eq!(bill.unpriced, vec!["synapse memory fetch"]);
        assert!((bill.synaptic.unwrap() - 1_000.0 * 26e-12).abs() < 1e-21);
    }

    /// The module doc says SATA's own ratios put memory at about 91% of SNN training energy, and at
    /// 90% to 93% across their rounding. Yin et al. give the SNN over the ANN as 1.35x in total,
    /// 3.28x in computation and 1.28x in memory movement; with the ANN's compute `c` and memory `m`,
    /// `3.28c + 1.28m = 1.35(c + m)` fixes `m / c`, and the SNN's memory share follows. Checked
    /// here because the 78% this module used to quote as the bill's share is not that number.
    #[test]
    fn satas_printed_ratios_put_memory_near_ninety_one_percent_of_snn_training_energy() {
        let share = |total: f64, compute: f64, memory: f64| {
            let m_over_c = (compute - total) / (total - memory);
            memory * m_over_c / (compute + memory * m_over_c)
        };
        let printed = share(1.35, 3.28, 1.28);
        assert!((printed - 0.915).abs() < 1e-3, "{printed}");
        // Every corner of the half-a-hundredth rounding box on the three ratios: 8 corners.
        let (mut lo, mut hi) = (1.0_f64, 0.0_f64);
        for corner in 0..8_u32 {
            let pick = |bit: u32, x: f64| if corner & (1 << bit) == 0 { x - 0.005 } else { x + 0.005 };
            let s = share(pick(0, 1.35), pick(1, 3.28), pick(2, 1.28));
            lo = lo.min(s);
            hi = hi.max(s);
        }
        assert!(lo > 0.90 && hi < 0.93, "{lo} to {hi}");
        assert!(lo > 0.78, "even the lowest corner is far above the 78% once quoted as the bill's");
    }

    /// Exactly one number in this crate was taken on fabricated silicon. If that count rises, the
    /// new figure needs a primary source somebody in this building actually opened.
    #[test]
    fn exactly_one_price_in_this_crate_came_from_fabricated_silicon() {
        let measured = CATALOGUE
            .iter()
            .filter(|(_, p)| p.evidence >= Evidence::Measured)
            .map(|(n, _)| *n)
            .collect::<Vec<_>>();
        assert_eq!(measured, vec!["truenorth-2014"], "silicon-grade tables: {measured:?}");
    }

    #[test]
    fn a_grade_is_no_better_than_its_worst_input() {
        assert_eq!(weaker(Evidence::Metered, Evidence::Projected), Evidence::Projected);
        assert_eq!(weaker(Evidence::Unstated, Evidence::Metered), Evidence::Unstated);
        assert_eq!(weaker(Evidence::Measured, Evidence::Measured), Evidence::Measured);
    }

    /// A bill's Display must carry the refusal, not swallow it — this is the string a user pastes
    /// into a report.
    #[test]
    fn a_refused_bill_prints_as_refused() {
        let led = Ledger { syn_ops: 1, syn_fetches: 1, ..Ledger::default() };
        let s = led.bill(&TRUENORTH_2014).to_string();
        assert!(s.contains("REFUSED"), "{s}");
        assert!(s.contains("synapse memory fetch"), "{s}");
    }

    /// Every table names its subject. A `Prices` that could be applied to any machine is how a
    /// laptop came to report another company's unfabricated accelerator's energy in the sibling
    /// crate, and this is the check that stops it recurring here.
    #[test]
    fn every_table_names_what_it_describes() {
        for (name, p) in CATALOGUE {
            assert!(p.source.len() > 24, "{name} has no real source line");
            if p.evidence != Evidence::Unstated {
                assert!(
                    p.source.contains("—") || p.source.contains("-"),
                    "{name} does not cite anything"
                );
            }
        }
    }
}
