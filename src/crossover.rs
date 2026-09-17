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
/// IEEE Transactions on Emerging Topics in Computational Intelligence 7(3):731–741 (2023).
///
/// > many previous studies did not consider memory accesses, which account for an important
/// > fraction of the energy consumption.
///
/// That sentence is the reason [`crate::ledger::Prices`] carries `e_syn_fetch` at all. The paper
/// finds spiking networks competitive only in the band **0.15 to 1.38 spikes per synapse per
/// inference**, depending on the architecture assumed.
pub const DAMPFHOFFER_2023: Crossover = Crossover {
    max_spikes_per_synapse: 1.38,
    min_spikes_per_synapse: 0.15,
    source: "Dampfhoffer, Mesquida, Valentian & Anghel, IEEE TETCI 7(3):731-741 (2023) — a band, \
             not a point, because the answer depends on the dataflow assumed. The paper's own \
             stated reason for revisiting the question is that earlier work omitted memory accesses.",
    evidence: Evidence::Derived,
};

/// Yan, Bai, Tang and Wong (National University of Singapore), arXiv:2409.08290.
///
/// > op-count evaluations neglect critical overheads like comprehensive data movements and memory
/// > accesses
///
/// Under a fair mapping against an equivalent quantised network their spiking model reaches
/// **0.78x** the energy on a spatial-dataflow architecture and 0.85x on a GPU-like one — a 22%
/// saving, achieved at **94.19% sparsity** with four timesteps over the nominal. The threshold here
/// is that sparsity expressed as the complementary spike density, and it is the tightest of the
/// three.
///
/// Worth reading next to any headline in the hundreds: this is what the same comparison looks like
/// when the data movement is counted.
pub const YAN_2024: Crossover = Crossover {
    max_spikes_per_synapse: 0.35,
    min_spikes_per_synapse: 0.06,
    source: "Yan, Bai, Tang & Wong (NUS), arXiv:2409.08290 — 0.78x energy on spatial dataflow and \
             0.85x GPU-like, at 94.19% sparsity with T=6, against an equivalent quantised network. \
             The band here is that sparsity read as spike density across the reported timesteps, so \
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

    /// A busy network is refuted by all three, and the report says so for each rather than picking
    /// the one it passes.
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
