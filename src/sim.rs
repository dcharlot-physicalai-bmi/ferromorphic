//! The simulator, in two modes that must agree.
//!
//! # Why two modes
//!
//! The case for neuromorphic hardware is that a spiking network does almost nothing almost all of
//! the time, so a machine that only pays for the "something" wins. That is an empirical claim about
//! a workload, and it is usually asserted rather than measured.
//!
//! [`Mode::Clocked`] updates every neuron on every tick, the way a dense simulator does.
//! [`Mode::EventDriven`] updates a neuron only on the ticks when a spike reaches it, jumping it
//! across the quiet interval in one step. Both write into the same [`Ledger`], so the difference
//! between them is a count — [`Ledger::idle_fraction`] — rather than a claim. On a sparse network
//! that fraction is most of the work, and on a busy one it is not, and the point of having both
//! modes is that you find out which without being told.
//!
//! # The two modes are required to agree, and that is tested
//!
//! Jumping a neuron across quiet ticks is only legal for a model whose state composes across an
//! interval, which is what [`Neuron::EXACT_OVER_GAPS`] declares. [`Sim::new`] **refuses** to build
//! an event-driven simulation of a model that lacks it, rather than producing spike times that
//! depend on which ticks happened to be quiet. For models that have it,
//! `the_two_modes_produce_the_same_spike_train` runs the same network both ways and compares the
//! trains spike for spike.
//!
//! One honest caveat on that agreement: membrane potentials match to floating-point noise, not bit
//! for bit. `exp(-dt/τ)` applied `k` times and `exp(-k·dt/τ)` applied once are equal as real numbers
//! and differ in the last place as `f64`. Spike *times* are identical because a potential would have
//! to sit within ~1e-16 V of threshold for that difference to change a comparison; if you build a
//! network that does sit there, the modes can disagree by one tick and the test above is where you
//! will see it.
//!
//! # What the ledger counts, and what a real chip would count instead
//!
//! Here, one delivery is one synaptic operation and one weight fetch, because this implementation
//! reads a weight at the moment it delivers. A chip that fetches a presynaptic neuron's whole
//! outgoing row on the spike would record the same totals; one that caches rows across ticks would
//! record fewer fetches than operations, and one that re-reads a row per target would record more.
//! The two counters are kept separate so that a device model can say which it is.

use crate::ledger::Ledger;
use crate::net::Net;
use crate::neuron::Neuron;
use crate::spike::{Spike, Train};

/// How the simulator spends its time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Update every neuron on every tick. Simple, obviously correct, and the reference the other
    /// mode is checked against.
    Clocked,
    /// Update a neuron only on ticks when a spike reaches it, jumping it across the quiet interval.
    ///
    /// Legal only for models where [`Neuron::EXACT_OVER_GAPS`] holds.
    EventDriven,
}

/// Why a simulation could not be built.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SimError {
    /// [`Mode::EventDriven`] was asked for with a neuron model that cannot be jumped across a gap.
    NotExactOverGaps,
    /// The neuron count did not match the network's.
    WrongNeuronCount {
        /// How many neurons were supplied.
        got: usize,
        /// How many the network expects.
        want: usize,
    },
}

impl core::fmt::Display for SimError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotExactOverGaps => f.write_str(
                "this neuron model is not exact over gaps, so event-driven simulation of it would \
                 change its spike times; use Mode::Clocked",
            ),
            Self::WrongNeuronCount { got, want } => {
                write!(f, "{got} neurons supplied for a network of {want}")
            }
        }
    }
}

/// See the note on [`crate::net::NetError`]: a library error has to be able to cross a
/// `Box<dyn Error>` boundary or its callers reach for `.unwrap()`.
impl std::error::Error for SimError {}

/// A spiking network being stepped forward in time.
#[derive(Debug, Clone)]
pub struct Sim<N: Neuron> {
    /// The connectivity.
    pub net: Net,
    /// Per-neuron state, in one contiguous allocation.
    pub neurons: Vec<N>,
    /// Tick length in seconds.
    pub dt: f64,
    /// Which way this simulation spends its time.
    pub mode: Mode,
    /// Exact counts of everything that happened. Price it with [`crate::ledger::Prices`].
    pub ledger: Ledger,
    /// Ticks elapsed. The next call to [`Sim::step`] processes this tick.
    pub t: u64,
    /// Pending synaptic deliveries, indexed `(tick % ring.len())`.
    ///
    /// A ring rather than a sorted queue: delivery ticks are bounded above by
    /// `t + net.max_delay`, so the whole future fits in `max_delay + 1` buckets and insertion is
    /// O(1) with no comparisons. A priority queue would be the obvious structure and would spend
    /// its time keeping an order nothing needs.
    ring: Vec<Vec<(u32, f64)>>,
    /// Volts arriving at each neuron on the tick being processed.
    inject: Vec<f64>,
    /// Neurons with something arriving this tick, so event-driven mode does not scan.
    touched: Vec<u32>,
    /// For each neuron, the tick its state is current as of.
    as_of: Vec<u64>,
}

impl<N: Neuron> Sim<N> {
    /// Build a simulation.
    ///
    /// # Errors
    ///
    /// [`SimError::WrongNeuronCount`] if `neurons.len()` does not match `net.n`, or
    /// [`SimError::NotExactOverGaps`] if [`Mode::EventDriven`] was asked for with a model that
    /// cannot legally be jumped across quiet ticks.
    pub fn new(net: Net, neurons: Vec<N>, dt: f64, mode: Mode) -> Result<Self, SimError> {
        if neurons.len() != net.n {
            return Err(SimError::WrongNeuronCount { got: neurons.len(), want: net.n });
        }
        if mode == Mode::EventDriven && !N::EXACT_OVER_GAPS {
            return Err(SimError::NotExactOverGaps);
        }
        let n = net.n;
        let ring_len = net.max_delay as usize + 1;
        Ok(Self {
            net,
            neurons,
            dt,
            mode,
            ledger: Ledger::default(),
            t: 0,
            ring: vec![Vec::new(); ring_len],
            inject: vec![0.0; n],
            touched: Vec::new(),
            as_of: vec![0; n],
        })
    }

    /// Advance one tick. `external` is a per-neuron input current in amperes, or empty for none.
    ///
    /// Returns the neurons that fired on this tick.
    ///
    /// # Panics
    ///
    /// If `external` is neither empty nor `net.n` long — a half-length current vector would
    /// silently drive only the first half of the network.
    pub fn step(&mut self, external: &[f64]) -> Vec<u32> {
        assert!(
            external.is_empty() || external.len() == self.net.n,
            "external current has {} entries for {} neurons",
            external.len(),
            self.net.n
        );

        // ---- 1. collect what arrives this tick -------------------------------------------------
        let slot = (self.t as usize) % self.ring.len();
        self.touched.clear();
        let deliveries = core::mem::take(&mut self.ring[slot]);
        for &(post, w) in &deliveries {
            let p = post as usize;
            if self.inject[p] == 0.0 {
                self.touched.push(post);
            }
            self.inject[p] += w;
            // One delivery is one operation AND one weight read. See the module doc for the device
            // models where those two counts come apart.
            self.ledger.syn_ops += 1;
            self.ledger.syn_fetches += 1;
        }
        self.ring[slot] = deliveries;
        self.ring[slot].clear();

        // A neuron with an external current is driven even with no synaptic input.
        if !external.is_empty() {
            for i in 0..self.net.n {
                if external[i] != 0.0 && self.inject[i] == 0.0 && !self.touched.contains(&(i as u32))
                {
                    self.touched.push(i as u32);
                }
            }
        }

        // ---- 2. update ------------------------------------------------------------------------
        let mut fired = Vec::new();
        match self.mode {
            Mode::Clocked => {
                for i in 0..self.net.n {
                    let driven = self.inject[i] != 0.0
                        || (!external.is_empty() && external[i] != 0.0);
                    if driven {
                        self.ledger.neuron_updates_driven += 1;
                    } else {
                        self.ledger.neuron_updates_idle += 1;
                    }
                    let ext = if external.is_empty() { 0.0 } else { external[i] };
                    self.neurons[i].bump(self.inject[i]);
                    if self.neurons[i].step(self.dt, ext) {
                        fired.push(i as u32);
                    }
                    self.as_of[i] = self.t + 1;
                }
            }
            Mode::EventDriven => {
                let mut touched = core::mem::take(&mut self.touched);
                // Sorted so that `fired` comes out in neuron order, matching the clocked mode's
                // output exactly. Without this the two modes would produce the same SET of spikes
                // in a different ORDER, and the equivalence test would compare unequal vectors of
                // equal content — a failure that reads as a physics bug and is a bookkeeping one.
                touched.sort_unstable();
                for &idx in &touched {
                    let i = idx as usize;
                    self.catch_up(i);
                    self.ledger.neuron_updates_driven += 1;
                    let ext = if external.is_empty() { 0.0 } else { external[i] };
                    self.neurons[i].bump(self.inject[i]);
                    if self.neurons[i].step(self.dt, ext) {
                        fired.push(idx);
                    }
                    self.as_of[i] = self.t + 1;
                }
                self.touched = touched;
            }
        }

        // ---- 3. post the spikes forward -------------------------------------------------------
        for &src in &fired {
            self.ledger.spikes_out += 1;
            for (post, w, d) in self.net.out_of(src as usize) {
                let when = (self.t + 1 + u64::from(d)) as usize % self.ring.len();
                self.ring[when].push((post, w));
            }
        }

        for &idx in &self.touched {
            self.inject[idx as usize] = 0.0;
        }
        if self.mode == Mode::Clocked {
            self.inject.iter_mut().for_each(|x| *x = 0.0);
        }
        self.t += 1;
        fired
    }

    /// Bring neuron `i` forward from the tick it is current as of to the current tick.
    ///
    /// Splits the jump at the end of any refractory period, **rounded up to a tick boundary**, so
    /// that the neuron resumes integrating on exactly the tick the clocked mode would have resumed
    /// it. The rounding is what makes the two modes agree: the clocked loop decrements the
    /// refractory counter once per tick, so a neuron is refractory for `ceil(t_ref / dt)` ticks
    /// whatever `t_ref` is, and jumping to the exact real-valued end of the period would let the
    /// event-driven run start up to one tick early.
    fn catch_up(&mut self, i: usize) {
        let gap = self.t.saturating_sub(self.as_of[i]);
        if gap == 0 {
            return;
        }
        let gap_s = gap as f64 * self.dt;
        let r = self.neurons[i].refractory_left();
        if r > 0.0 {
            let r_ticks = (r / self.dt).ceil().min(gap as f64);
            let r_s = r_ticks * self.dt;
            self.neurons[i].step(r_s, 0.0);
            let rest = gap_s - r_s;
            if rest > 0.0 {
                self.neurons[i].step(rest, 0.0);
            }
        } else {
            self.neurons[i].step(gap_s, 0.0);
        }
        // A quiet interval cannot produce a spike in a model whose resting potential is below
        // threshold, which is why the jump is legal at all. The return value of the steps above is
        // therefore discarded deliberately rather than by oversight, and this is the note saying so.
        self.as_of[i] = self.t;
    }

    /// Run for `ticks` ticks with a constant external current, recording every spike.
    ///
    /// `external` is empty or `net.n` long, as in [`Sim::step`].
    ///
    /// # Panics
    ///
    /// As [`Sim::step`].
    pub fn run(&mut self, ticks: u64, external: &[f64]) -> Train {
        let mut train = Train::new();
        for _ in 0..ticks {
            let t = self.t;
            for src in self.step(external) {
                train.push(Spike { t, source: src });
            }
        }
        train
    }

    /// Read every neuron's membrane potential, charging the ledger for the readout.
    ///
    /// In event-driven mode this first brings every neuron up to the current tick, because a
    /// neuron that has not been touched for a thousand ticks holds a stale potential and reporting
    /// it would be reporting the past.
    ///
    /// Charged rather than free because on real hardware it is not free, and in the sibling crate
    /// five hand-written collection loops each reported their readback as exactly zero — which on
    /// one endpoint was 98.9% of the answer's energy.
    pub fn read_potentials(&mut self) -> Vec<f64> {
        if self.mode == Mode::EventDriven {
            for i in 0..self.net.n {
                self.catch_up(i);
            }
        }
        self.ledger.reads += self.net.n as u64;
        self.neurons.iter().map(Neuron::potential).collect()
    }

    /// Reset every neuron and clear the pending deliveries, keeping the ledger.
    ///
    /// The ledger survives because it is a record of what the hardware did, and a run that resets
    /// its network halfway through still paid for the first half.
    pub fn reset_state(&mut self) {
        for n in &mut self.neurons {
            n.reset();
        }
        for r in &mut self.ring {
            r.clear();
        }
        self.inject.iter_mut().for_each(|x| *x = 0.0);
        self.as_of.iter_mut().for_each(|x| *x = self.t);
        self.touched.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::{Mode, Sim, SimError};
    use crate::net::NetBuilder;
    use crate::neuron::{Izhikevich, Lif, Neuron};

    fn chain(n: usize, w: f64, delay: u32) -> crate::net::Net {
        let mut b = NetBuilder::new(n);
        for i in 0..n - 1 {
            b.connect(i as u32, i as u32 + 1, w, delay).unwrap();
        }
        b.build()
    }

    /// The claim the two-mode design rests on.
    #[test]
    fn the_two_modes_produce_the_same_spike_train() {
        let net = chain(6, 20e-3, 2);
        let proto = Lif { t_ref: 2e-3, ..Lif::default() };
        let dt = 1e-4; // t_ref is exactly 20 ticks, so the refractory boundary is not the variable
        let mut ext = vec![0.0; 6];
        ext[0] = 3e-9;

        let mut a = Sim::new(net.clone(), vec![proto; 6], dt, Mode::Clocked).unwrap();
        let mut b = Sim::new(net, vec![proto; 6], dt, Mode::EventDriven).unwrap();
        let ta = a.run(4_000, &ext);
        let tb = b.run(4_000, &ext);

        assert!(ta.len() > 20, "only {} spikes; the test would prove nothing", ta.len());
        assert_eq!(ta.spikes(), tb.spikes(), "the two modes disagreed");
    }

    /// And the point of having both: the event-driven run does strictly less work for the same
    /// answer, and the ledger says how much less.
    #[test]
    fn the_event_driven_run_does_less_work_for_the_same_answer() {
        let net = chain(6, 20e-3, 2);
        let proto = Lif::default();
        let dt = 1e-4;
        let mut ext = vec![0.0; 6];
        ext[0] = 3e-9;

        let mut a = Sim::new(net.clone(), vec![proto; 6], dt, Mode::Clocked).unwrap();
        let mut b = Sim::new(net, vec![proto; 6], dt, Mode::EventDriven).unwrap();
        a.run(4_000, &ext);
        b.run(4_000, &ext);

        assert_eq!(a.ledger.syn_ops, b.ledger.syn_ops, "the same synapses did the same work");
        assert!(
            b.ledger.neuron_updates() < a.ledger.neuron_updates() / 4,
            "event-driven did {} updates against clocked {}",
            b.ledger.neuron_updates(),
            a.ledger.neuron_updates()
        );
        // The clocked run's idle fraction IS the case for the hardware, stated as a number.
        let idle = a.ledger.idle_fraction().unwrap();
        assert!(idle > 0.8, "clocked idle fraction was only {idle}");
        assert_eq!(b.ledger.neuron_updates_idle, 0, "event-driven updated an idle neuron");
    }

    /// The refusal, which is the safety property. Izhikevich is not exact over gaps, so an
    /// event-driven simulation of it is not built at all rather than built and quietly wrong.
    #[test]
    fn an_inexact_model_cannot_be_run_event_driven() {
        let net = chain(3, 20e-3, 1);
        let cells = vec![Izhikevich::regular_spiking(); 3];
        const { assert!(!Izhikevich::EXACT_OVER_GAPS) };
        assert_eq!(
            Sim::new(net.clone(), cells.clone(), 1e-4, Mode::EventDriven).err(),
            Some(SimError::NotExactOverGaps)
        );
        assert!(Sim::new(net, cells, 1e-4, Mode::Clocked).is_ok(), "clocked must still work");
    }

    #[test]
    fn a_neuron_count_mismatch_is_refused() {
        let net = chain(4, 1e-3, 0);
        let err = Sim::new(net, vec![Lif::default(); 3], 1e-4, Mode::Clocked).unwrap_err();
        assert_eq!(err, SimError::WrongNeuronCount { got: 3, want: 4 });
    }

    /// Delay has to actually delay. A synapse with `d` ticks must land `d + 1` ticks after the
    /// spike — one tick for the spike to be emitted, `d` for the axon.
    #[test]
    fn a_delayed_synapse_delivers_when_it_says_it_will() {
        let mut b = NetBuilder::new(2);
        // A weight large enough that one arriving spike fires the target immediately.
        b.connect(0, 1, 30e-3, 5).unwrap();
        let net = b.build();
        let dt = 1e-4;
        let proto = Lif { t_ref: 0.0, ..Lif::default() };
        let mut sim = Sim::new(net, vec![proto; 2], dt, Mode::Clocked).unwrap();

        let mut ext = vec![0.0; 2];
        ext[0] = 5e-9;
        let train = sim.run(2_000, &ext);
        let src = train.of(0);
        let dst = train.of(1);
        assert!(!src.is_empty() && !dst.is_empty(), "{} / {}", src.len(), dst.len());
        assert_eq!(dst[0].t, src[0].t + 1 + 5, "first arrival was not at +6 ticks");
    }

    /// A readout is charged, and in event-driven mode it reports the present rather than the last
    /// tick a neuron happened to be touched.
    #[test]
    fn reading_potentials_costs_the_ledger_and_reports_the_present() {
        let net = chain(4, 1e-3, 0);
        let proto = Lif { v: -40e-3, ..Lif::default() };
        let dt = 1e-4;
        let mut sim = Sim::new(net, vec![proto; 4], dt, Mode::EventDriven).unwrap();
        sim.run(2_000, &[]);
        assert_eq!(sim.ledger.reads, 0, "nothing was read yet");
        let v = sim.read_potentials();
        assert_eq!(sim.ledger.reads, 4);
        // 2,000 ticks of 0.1 ms is 200 ms, TEN membrane constants, so the residual is
        // 25 mV * exp(-10) = 1.1e-6 V and a 1e-5 V tolerance is comfortable. The first draft ran
        // five time constants and demanded 1e-4 V, which the physics cannot deliver: the residual
        // there is 1.7e-4 V. The test was wrong about the exponential, not the code.
        for (i, &x) in v.iter().enumerate() {
            assert!((x - (-65e-3)).abs() < 1e-5, "neuron {i} held a stale {x} V");
        }
    }

    /// An empty external slice means no current, and must not be read as a zero-length network.
    #[test]
    fn an_empty_external_current_is_allowed() {
        let net = chain(3, 1e-3, 0);
        let mut sim = Sim::new(net, vec![Lif::default(); 3], 1e-4, Mode::Clocked).unwrap();
        let train = sim.run(100, &[]);
        assert!(train.is_empty(), "a quiet network fired");
        assert_eq!(sim.ledger.neuron_updates_idle, 300);
    }

    #[test]
    #[should_panic(expected = "external current has")]
    fn a_wrong_length_external_current_panics_rather_than_driving_half_the_network() {
        let net = chain(4, 1e-3, 0);
        let mut sim = Sim::new(net, vec![Lif::default(); 4], 1e-4, Mode::Clocked).unwrap();
        sim.step(&[1e-9, 1e-9]);
    }

    /// Determinism across runs, which every claim in this crate depends on.
    #[test]
    fn the_same_network_run_twice_gives_the_same_spikes() {
        let net = chain(8, 18e-3, 3);
        let proto = Lif::default();
        let mut ext = vec![0.0; 8];
        ext[0] = 4e-9;
        let run = || {
            let mut s = Sim::new(net.clone(), vec![proto; 8], 1e-4, Mode::Clocked).unwrap();
            s.run(3_000, &ext)
        };
        assert_eq!(run().spikes(), run().spikes());
    }
}
