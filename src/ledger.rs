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
//! and pricing them from a datasheet prices the arithmetic and silently sets the memory traffic to
//! zero — which is precisely the term that separates a real deployment from a benchmark.
//!
//! So [`Prices`] has a `e_syn_fetch` field, every device table in this module leaves it
//! **`None`**, and [`Ledger::joules`] therefore returns `None` for every one of them. That is not
//! an unfinished implementation. It is the finding: *this review did not locate a published
//! per-synapse memory-fetch energy for any commercially available neuromorphic processor.* If you
//! have one — measured, for a stated device, at a stated boundary — supply it and the ledger will
//! price your workload.
//!
//! # The flattering number is still available, and it is labelled
//!
//! [`Ledger::joules_synops_only`] computes exactly the figure the literature reports, ignoring the
//! fetch term. It is there because a user needs to reproduce published numbers to argue with them,
//! and because hiding it would not make anybody stop using it. Its doc says what it omits, and
//! [`Bill`] names every term that went unpriced so a caller cannot report the total without also
//! having been handed the list of what is missing from it.

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
    pub e_syn_op: f64,
    /// **Reading one synapse's weight out of memory so that the operation above can happen.**
    ///
    /// `None` in every table in this module, because this review did not locate a published figure
    /// for it on any commercial part. It is separated from `e_syn_op` rather than folded into it
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
    };

    /// Whether every term needed for a complete bill has a number.
    ///
    /// False for every table currently in this module, because none of them prices a synapse
    /// fetch. That is the finding, not a defect in the type.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.e_syn_op.is_finite()
            && self.e_neuron_update.is_finite()
            && self.e_syn_fetch.is_some()
            && self.e_spike_out.is_some()
            && self.e_read.is_some()
    }

    /// The terms this table cannot price, by name, in the order a bill would list them.
    #[must_use]
    pub fn unpriced(&self) -> Vec<&'static str> {
        let mut v = Vec::new();
        if !self.e_syn_op.is_finite() {
            v.push("synaptic operation");
        }
        if self.e_syn_fetch.is_none() {
            v.push("synapse memory fetch");
        }
        if !self.e_neuron_update.is_finite() {
            v.push("neuron update");
        }
        if self.e_spike_out.is_none() {
            v.push("spike emission");
        }
        if self.e_read.is_none() {
            v.push("state readout");
        }
        v
    }
}

/// `TrueNorth`'s published per-synaptic-event energy.
///
/// Merolla et al., *A million spiking-neuron integrated circuit with a scalable communication
/// network and interface*, Science 345(6197):668–673, 2014. The paper reports 26 pJ per synaptic
/// event for the fabricated 28 nm part, which is a measurement of silicon and is graded
/// accordingly.
///
/// **What is not in it.** The paper's figure prices the synaptic event. This review did not locate
/// a separately published memory-fetch energy, spike-routing energy or host-readout energy for the
/// part, so those three stay `None` and [`Ledger::joules`] will refuse. The neuron update is
/// likewise not separable from the published aggregate; it is left unstated rather than
/// back-derived, because a number computed from a total by assuming what the other terms were is
/// not a measurement of anything.
pub const TRUENORTH_2014: Prices = Prices {
    e_syn_op: 26e-12,
    e_syn_fetch: None,
    e_neuron_update: f64::NAN,
    e_spike_out: None,
    e_read: None,
    source: "TrueNorth 28 nm, 26 pJ per synaptic event — Merolla et al., Science 345(6197), 2014. \
             Silicon, measured. Applies to that part and to nothing else; the fetch, routing and \
             readout terms were not separately published and are not guessed here.",
    evidence: Evidence::Measured,
};

/// Loihi's published per-synaptic-operation energy.
///
/// Davies et al., *Loihi: A Neuromorphic Manycore Processor with On-Chip Learning*, IEEE Micro
/// 38(1):82–99, 2018, which reports on the order of 23.6 pJ per synaptic operation for the 14 nm
/// part at its nominal operating point.
///
/// **Read the grade.** This is a vendor-published figure for a research part distributed through a
/// research community rather than sold, and the measurement protocol behind it is not stated to the
/// standard [`Evidence::Metered`] requires. It is `Measured`, not `Metered`, and the gap between
/// those two words is where most of this field's energy claims live.
pub const LOIHI_2018: Prices = Prices {
    e_syn_op: 23.6e-12,
    e_syn_fetch: None,
    e_neuron_update: f64::NAN,
    e_spike_out: None,
    e_read: None,
    source: "Loihi 14 nm, ~23.6 pJ per synaptic operation — Davies et al., IEEE Micro 38(1), 2018. \
             Vendor-published for a research part; protocol not stated to a metered standard. The \
             fetch, routing and readout terms were not separately published.",
    evidence: Evidence::Measured,
};

/// Every device table in this crate, for a caller that wants to sweep them.
///
/// Three entries, and two of them are historical parts from 2014 and 2018. That shortness is
/// itself the state of the field as this review found it: per-operation energies for the current
/// commercial parts are quoted in marketing units — TOPS/W, "1000x more efficient" — that do not
/// reduce to a per-operation joule figure anybody can put in a table.
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
    /// Joules attributable to synaptic operations.
    pub synaptic: Option<f64>,
    /// Joules attributable to synapse memory fetches — the term the literature omits.
    pub fetch: Option<f64>,
    /// Joules attributable to membrane updates, idle and driven together.
    pub neurons: Option<f64>,
    /// Joules attributable to spike emission and routing.
    pub routing: Option<f64>,
    /// Joules attributable to host readout.
    pub readout: Option<f64>,
    /// Names of the terms that had work to price but no price for it.
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
        let fetch = charge(self.syn_fetches, p.e_syn_fetch, "synapse memory fetch");
        let neurons = charge(self.neuron_updates(), finite(p.e_neuron_update), "neuron update");
        let routing = charge(self.spikes_out, p.e_spike_out, "spike emission");
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

    /// **The figure the literature reports**: synaptic operations only, priced and summed, with
    /// memory traffic, routing and readout omitted.
    ///
    /// This is here so that published numbers can be reproduced and argued with, not because it is
    /// a good way to price a workload. It is systematically optimistic by construction — it charges
    /// for the arithmetic and sets the data movement that feeds the arithmetic to zero — and on a
    /// model whose synapses do not fit on-core, the omitted term can be the larger one.
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
    /// `None` when either side has no answer. This is the quantity that says how much a published
    /// SOP-counted figure understated a given workload, and it is the number this crate exists to
    /// make computable.
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
/// to mean "unstated" and the `Option` the charging path works in.
fn finite(x: f64) -> Option<f64> {
    if x.is_finite() { Some(x) } else { None }
}

#[cfg(test)]
mod tests {
    use super::{CATALOGUE, Evidence, LOIHI_2018, Ledger, Prices, TRUENORTH_2014, weaker};

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
