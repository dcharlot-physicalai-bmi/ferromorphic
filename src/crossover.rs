//! The published crossover thresholds, as a check a workload can run against itself.
//!
//! # The finding this module operationalises
//!
//! There is a mature, in-field literature arguing that spiking energy claims are inflated by the way
//! they are counted, and it does not merely say so — it gives a **number**. The quantity that
//! decides whether a spiking network can beat the equivalent dense one is *spikes per synapse per
//! inference*: each spike that crosses a synapse costs an accumulate and, far more importantly, a
//! weight fetch, and once a network fires often enough that it re-reads its weights more than a
//! dense pass would, the dense pass wins.
//!
//! Every threshold below is under 2, and several are under 1.
//!
//! **This is not an outside critique.** Davidson and Furber's paper is the strongest of them, and
//! Steve Furber designed `SpiNNaker`.
//!
//! # Why it lives in a library rather than a paper
//!
//! This review located the argument in at least six papers and **did not locate it implemented as a
//! check in any spiking-network library**. The thresholds are quoted in related work sections and
//! then not applied, which is a strange place for a number that decides whether a whole approach
//! helps. [`Ledger::spikes_per_synapse`] computes the left-hand side exactly — it is a ratio of two
//! integer counts the simulator already keeps — so the comparison costs nothing and can run on every
//! workload.
//!
//! ```
//! use ferromorphic::crossover::{DAMPFHOFFER_2023, DAVIDSON_FURBER_2021, Verdict};
//! use ferromorphic::ledger::Ledger;
//!
//! // 10,000 synapses, 100 inferences, 1.2M deliveries -> 1.2 spikes per synapse per inference.
//! let led = Ledger { syn_ops: 1_200_000, ..Ledger::default() };
//! let sps = led.spikes_per_synapse(10_000, 100).unwrap();
//! assert!((sps - 1.2).abs() < 1e-12);
//!
//! // Under Davidson & Furber's ~1.72 it is still plausible; under Dampfhoffer's 1.38 it is
//! // marginal; under the pessimistic end of that same paper it is already lost.
//! assert_eq!(DAVIDSON_FURBER_2021.verdict(sps), Verdict::Plausible);
//! assert_eq!(DAMPFHOFFER_2023.verdict(sps), Verdict::Marginal);
//! ```
//!
//! # What a verdict is not
//!
//! It is not a measurement of your workload on your hardware. Every threshold here was derived for a
//! stated technology, a stated dense baseline and a stated dataflow, and transplanting it to a
//! different part is exactly the borrowing this crate's [`crate::ledger`] refuses to do with joules.
//! A [`Verdict`] says *which side of somebody else's published line this workload falls on*, which is
//! a much weaker and much more checkable statement. Read [`Crossover::source`] before quoting one.

use crate::ledger::{Evidence, Ledger};

/// A published threshold in spikes per synapse per inference, above which the cited work found the
/// spiking implementation no longer beats its dense equivalent.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Crossover {
    /// The threshold itself: spikes per synapse per inference.
    pub max_spikes_per_synapse: f64,
    /// The lower end of the cited range, where the work gives one. Equal to
    /// [`Crossover::max_spikes_per_synapse`] where it gives a single figure.
    ///
    /// Carried because a range is honest and a midpoint is not: a workload between the two ends is
    /// *marginal*, and collapsing the range to one number would report it as a pass or a fail.
    pub min_spikes_per_synapse: f64,
    /// Who found it, in what, under what assumptions.
    pub source: &'static str,
    /// What kind of evidence [`Crossover::source`] is.
    pub evidence: Evidence,
}

/// Which side of a published line a workload falls on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// Below the whole cited range. The spiking implementation is in the regime where the cited
    /// work found it could win.
    Plausible,
    /// Inside the cited range, where the answer depended on assumptions the citation states.
    Marginal,
    /// Above the whole cited range. The cited work found the dense equivalent wins here.
    Refuted,
}

impl Crossover {
    /// Which side of this threshold `spikes_per_synapse` falls on.
    #[must_use]
    pub fn verdict(&self, spikes_per_synapse: f64) -> Verdict {
        if spikes_per_synapse < self.min_spikes_per_synapse {
            Verdict::Plausible
        } else if spikes_per_synapse <= self.max_spikes_per_synapse {
            Verdict::Marginal
        } else {
            Verdict::Refuted
        }
    }
}

/// Davidson and Furber, *Comparison of Artificial and Spiking Neural Networks on Digital Hardware*,
/// Frontiers in Neuroscience 15:651141 (2021).
///
/// > Assuming identical underlying silicon technology we show that most rate-coded spiking network
/// > implementations will not be more energy or resource efficient than the original ANN.
///
/// The derived threshold is about **1.72 spikes per synapse per inference**. The weight this
/// citation carries is not only its arithmetic: **Steve Furber designed `SpiNNaker`**, so this is the
/// field auditing itself rather than an outsider objecting to it.
pub const DAVIDSON_FURBER_2021: Crossover = Crossover {
    max_spikes_per_synapse: 1.72,
    min_spikes_per_synapse: 1.72,
    source: "Davidson & Furber, Front. Neurosci. 15:651141 (2021) — derived at identical silicon \
             technology against the original ANN. Applies under those assumptions and not to a \
             different technology node or dataflow.",
    evidence: Evidence::Derived,
};

/// Dampfhoffer, Mesquida, Valentian and Anghel, *Are SNNs Really More Energy-Efficient Than ANNs?*,
/// IEEE Transactions on Emerging Topics in Computational Intelligence 7(3):731–741 (2023),
/// doi:10.1109/tetci.2022.3214509.
///
/// > Many studies do not consider memory accesses, which account for an important fraction of the
/// > energy consumption, use naïve ANN hardware implementations, or lack generality.
///
/// That sentence is the reason [`crate::ledger::Prices`] carries `e_syn_fetch` at all. The paper
/// finds spiking networks competitive only in the band **0.15 to 1.38 spikes per synapse per
/// inference**, depending on the architecture assumed.
///
/// **Corrected quotation.** This entry used to quote the paper as "many previous studies did not
/// consider memory accesses, which account for an important fraction of the energy consumption."
/// That is not the paper's sentence. The words above are its abstract's, as printed in the
/// accepted version on HAL (cea-03852141), and this review did not locate "many previous studies
/// did not consider memory accesses" anywhere in that full text. The meaning was right and the
/// quotation marks were not; no constant or test depended on the wording.
pub const DAMPFHOFFER_2023: Crossover = Crossover {
    max_spikes_per_synapse: 1.38,
    min_spikes_per_synapse: 0.15,
    source: "Dampfhoffer, Mesquida, Valentian & Anghel, IEEE TETCI 7(3):731-741 (2023) — a band, \
             not a point, because the answer depends on the dataflow assumed. The paper's own \
             stated reason for revisiting the question is that earlier work omitted memory accesses.",
    evidence: Evidence::Derived,
};

/// Yan, Bai and Wong (National University of Singapore), *Reconsidering the energy efficiency of
/// spiking neural networks*, arXiv:2409.08290v1 (29 Aug 2024).
///
/// > However, most SNN works only consider the counting of additions to evaluate energy
/// > consumption, neglecting other overheads such as memory accesses and data movement operations.
///
/// Their spiking VGG16 (CIFAR-10, T = 6, **94.19% sparsity**) reaches **0.85x** and **0.78x** the
/// energy of the best-case ANN on classical GPU-like and spatial-dataflow architectures
/// respectively — at best a 22% saving. The best case is the paper's own framing: "A fair
/// comparison should involve an ideally optimized ANN", one assumed to have "optimal sparsity and
/// weight reuse to minimize the DRAM energy". The threshold here is that sparsity expressed as the
/// complementary spike density, and it is the tightest of the three.
///
/// Worth reading next to any headline in the hundreds: this is what the same comparison looks like
/// when the data movement is counted.
///
/// **Corrected: this entry mixed two versions of one paper.** It used to credit "Yan, Bai, Tang and
/// Wong", cite the unversioned arXiv:2409.08290, and say the figures were reached "under a fair
/// mapping against an equivalent quantised network". The figures it pins (0.78x, 0.85x, 94.19%,
/// T = 6) are v1's. The v1 PDF names three authors, and prints those figures in its Table 5
/// ("VGG16 92.76 94.19 0.85/0.78 6") and in its conclusion, against "the best-case ANN". Kaiwen
/// Tang joins the author list from v2 (3 Jul 2025), and so does the equivalent-QNN baseline:
/// "functionally equivalent QNNs with ⌈log2(T+1)⌉ bits" in v2 to v4, "capacity-matched QNNs" in v5
/// and v6. This review did not locate 94.19 or 0.78 in the v2 to v6 PDFs. v1's ANN is quantised
/// during training (its Eq. 2, `Q_T(x) = ⌊x·T⌋/T`) and costed with 8-bit operations (its Table 1),
/// so the word "quantised" was not foreign to it; "equivalent" was.
///
/// The quotation above replaces one that was verbatim in no version. "op-count evaluations neglect
/// critical overheads like comprehensive data movements and memory accesses" put this review's own
/// words in front of the v5 and v6 abstract's "neglecting critical overheads like comprehensive
/// data movements and memory accesses", and set the result beside figures only v1 prints.
///
/// The published version is Yan, Bai, Tang and Wong, *Reconsidering the Energy Efficiency of
/// Spiking Neural Networks Inference from Analytical Perspectives*, IEEE Transactions on
/// Computer-Aided Design of Integrated Circuits and Systems (2026), doi:10.1109/tcad.2026.3718799.
/// Its headline is a different figure: in the v6 abstract, a spike rate below 5.7% at T = 5 to
/// outperform equivalent QNNs. The band below is still v1's figure read as spike density, so the
/// constants, the `yan-2024` key and the verdicts are unchanged; a band re-derived from the
/// published version would be a different threshold, not a correction to this one.
pub const YAN_2024: Crossover = Crossover {
    max_spikes_per_synapse: 0.35,
    min_spikes_per_synapse: 0.06,
    source: "Yan, Bai & Wong (NUS), arXiv:2409.08290v1 (2024) — spiking VGG16 on CIFAR-10 at \
             94.19% sparsity with T=6 reaches 0.85x (classical GPU-like) and 0.78x (spatial \
             dataflow) the energy of the best-case ANN. Those figures are printed in v1 only; \
             the four-author IEEE TCAD 2026 version does not carry them. The band here is that \
             sparsity read as spike density across the reported timesteps, so \
             it is this review's reading of their figure rather than a number they print.",
    evidence: Evidence::Derived,
};

/// Every threshold in this module, weakest constraint first.
///
/// Three, and all three are under 2. That is the shape of the finding: the regime where a spiking
/// network can win on energy is narrow, and it is narrow in a quantity a simulator already counts.
pub const THRESHOLDS: [(&str, Crossover); 3] = [
    ("davidson-furber-2021", DAVIDSON_FURBER_2021),
    ("dampfhoffer-2023", DAMPFHOFFER_2023),
    ("yan-2024", YAN_2024),
];

impl Ledger {
    /// Spikes per synapse per inference — the quantity every published crossover threshold is
    /// stated in.
    ///
    /// `syn_ops` counts deliveries, so this is deliveries divided by the number of synapses that
    /// could have carried one, divided again by how many inferences the run represents.
    ///
    /// `None` when either divisor is zero: a network with no synapses and a run with no inferences
    /// both have no answer, and returning zero would read as "extremely sparse, certain to win".
    #[must_use]
    pub fn spikes_per_synapse(&self, n_syn: u64, inferences: u64) -> Option<f64> {
        if n_syn == 0 || inferences == 0 {
            return None;
        }
        Some(self.syn_ops as f64 / (n_syn as f64 * inferences as f64))
    }

    /// This workload against every published threshold, as `(name, verdict)`.
    ///
    /// `None` when [`Ledger::spikes_per_synapse`] has no answer.
    ///
    /// Reported against ALL of them rather than against a chosen one, because picking the threshold
    /// that a workload passes is the same move as picking the price table that flatters a device.
    #[must_use]
    pub fn crossover_verdicts(
        &self,
        n_syn: u64,
        inferences: u64,
    ) -> Option<Vec<(&'static str, Verdict)>> {
        let sps = self.spikes_per_synapse(n_syn, inferences)?;
        Some(THRESHOLDS.iter().map(|(name, c)| (*name, c.verdict(sps))).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::{DAMPFHOFFER_2023, DAVIDSON_FURBER_2021, THRESHOLDS, Verdict, YAN_2024};
    use crate::ledger::Ledger;

    #[test]
    fn spikes_per_synapse_is_deliveries_over_synapses_over_inferences() {
        let led = Ledger { syn_ops: 1_200_000, ..Ledger::default() };
        let sps = led.spikes_per_synapse(10_000, 100).unwrap();
        assert!((sps - 1.2).abs() < 1e-12, "{sps}");
    }

    /// Zero is not "extremely sparse"; it is "you did not ask a question".
    #[test]
    fn a_missing_divisor_has_no_answer_rather_than_a_flattering_one() {
        let led = Ledger { syn_ops: 5_000, ..Ledger::default() };
        assert!(led.spikes_per_synapse(0, 10).is_none());
        assert!(led.spikes_per_synapse(10, 0).is_none());
        assert!(led.crossover_verdicts(0, 10).is_none());
    }

    /// The three-way verdict, at the boundaries of a band.
    #[test]
    fn a_band_gives_three_verdicts_rather_than_two() {
        // Dampfhoffer is 0.15 to 1.38.
        assert_eq!(DAMPFHOFFER_2023.verdict(0.10), Verdict::Plausible);
        assert_eq!(DAMPFHOFFER_2023.verdict(0.15), Verdict::Marginal, "the lower edge is inside");
        assert_eq!(DAMPFHOFFER_2023.verdict(0.80), Verdict::Marginal);
        assert_eq!(DAMPFHOFFER_2023.verdict(1.38), Verdict::Marginal, "the upper edge is inside");
        assert_eq!(DAMPFHOFFER_2023.verdict(1.50), Verdict::Refuted);
    }

    /// A point threshold has no marginal region except exactly on the line.
    #[test]
    fn a_point_threshold_is_a_degenerate_band() {
        assert_eq!(DAVIDSON_FURBER_2021.verdict(1.0), Verdict::Plausible);
        assert_eq!(DAVIDSON_FURBER_2021.verdict(1.72), Verdict::Marginal);
        assert_eq!(DAVIDSON_FURBER_2021.verdict(2.0), Verdict::Refuted);
    }

    /// The headline shape of the finding, asserted so a future edit cannot quietly soften it.
    #[test]
    fn every_published_threshold_is_below_two_spikes_per_synapse() {
        for (name, c) in THRESHOLDS {
            assert!(
                c.max_spikes_per_synapse < 2.0,
                "{name} claims a threshold of {} — that needs a citation and an argument",
                c.max_spikes_per_synapse
            );
            assert!(c.min_spikes_per_synapse <= c.max_spikes_per_synapse, "{name} has an inverted band");
            assert!(c.source.len() > 40, "{name} does not cite anything");
        }
    }

    /// The Yan figures (0.85x, 0.78x, 94.19%, T=6) are arXiv:2409.08290v1's, 0.78 and 94.19
    /// appear in no later version, and v1 sets them against its best-case ANN, not an equivalent
    /// QNN. The source string used to cite the unversioned paper with the later versions' author
    /// list and baseline, so a reader following it found a paper that does not print the numbers.
    /// This pins the version and the baseline the numbers belong to.
    #[test]
    fn the_yan_source_names_the_version_and_the_baseline_its_figures_come_from() {
        let s = YAN_2024.source;
        assert!(s.contains("arXiv:2409.08290v1"), "{s}");
        assert!(s.contains("best-case ANN"), "{s}");
        assert!(!s.contains("equivalent quantised"), "{s}");
        for figure in ["0.85x", "0.78x", "94.19%", "T=6"] {
            assert!(s.contains(figure), "{figure} is missing from {s}");
        }
    }

    /// A busy network is refuted by all three, and the report says so for each rather than picking
    /// the one it passes.
    /// The table is the module's whole claim, and two things about it were unguarded: that every
    /// entry is the constant it names, and that not one of them claims to have been MEASURED.
    /// That second grade is the point of grading at all — every threshold here is an analysis of
    /// somebody's silicon, not a reading from it, and a module that quietly upgraded one to
    /// `Measured` would be making the exact claim this crate refuses to make.
    #[test]
    fn every_published_threshold_is_derived_evidence_and_appears_once() {
        use crate::ledger::Evidence;
        assert_eq!(THRESHOLDS.len(), 3);
        for (name, c) in THRESHOLDS {
            assert_eq!(c.evidence, Evidence::Derived, "{name} claims to be measured");
            assert!(!c.source.is_empty(), "{name} cites nothing");
            assert!(c.min_spikes_per_synapse <= c.max_spikes_per_synapse, "{name} has its band inverted");
        }
        // Three distinct names, each paired with the constant it names.
        let names: Vec<&str> = THRESHOLDS.iter().map(|(n, _)| *n).collect();
        let mut sorted = names.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), 3, "the table repeats a name: {names:?}");
        assert_eq!(THRESHOLDS[0], ("davidson-furber-2021", DAVIDSON_FURBER_2021));
        assert_eq!(THRESHOLDS[1], ("dampfhoffer-2023", DAMPFHOFFER_2023));
        assert_eq!(THRESHOLDS[2], ("yan-2024", YAN_2024));
        // And the three are genuinely different thresholds, so a table that listed one of them
        // twice would be reporting a workload against two fewer opinions than it says it does.
        assert!(DAVIDSON_FURBER_2021 != DAMPFHOFFER_2023 && DAMPFHOFFER_2023 != YAN_2024 && DAVIDSON_FURBER_2021 != YAN_2024);
    }

    #[test]
    fn a_dense_firing_workload_is_refuted_by_every_threshold() {
        let led = Ledger { syn_ops: 50_000_000, ..Ledger::default() };
        let v = led.crossover_verdicts(10_000, 100).unwrap(); // 50 spikes per synapse
        assert_eq!(v.len(), 3);
        assert!(v.iter().all(|(_, x)| *x == Verdict::Refuted), "{v:?}");
    }

    /// And a genuinely sparse one passes all three — the regime the hardware argument needs.
    #[test]
    fn a_very_sparse_workload_is_plausible_under_every_threshold() {
        let led = Ledger { syn_ops: 50_000, ..Ledger::default() };
        let sps = led.spikes_per_synapse(10_000, 100).unwrap(); // 0.05
        assert!(sps < YAN_2024.min_spikes_per_synapse);
        let v = led.crossover_verdicts(10_000, 100).unwrap();
        assert!(v.iter().all(|(_, x)| *x == Verdict::Plausible), "{v:?}");
    }
}
