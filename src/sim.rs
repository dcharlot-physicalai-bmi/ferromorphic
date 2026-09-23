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
//! # Exact spike timing
//!
//! A tick-based update decides once per tick whether a neuron has reached threshold, so every
//! inter-spike interval is rounded UP to a whole number of ticks and every rate is biased low — by
//! 17% for a cell whose true interval is 3.3 ticks. [`Sim::with_exact_timing`] switches both modes
//! to [`Neuron::step_counted`], which for a model with [`Neuron::EXACT_TIMING`] solves for the
//! crossing inside the tick: a driven neuron then fires `⌊(T + t_ref)/isi⌋` times in `T` seconds
//! whatever the tick, more than one spike can fall in a tick, and each is posted to the network.
//!
//! What this does NOT change is WHEN a spike arrives. Deliveries stay on the tick grid — a spike
//! emitted anywhere inside tick `t` reaches its targets at tick `t + 1 + delay` — so exact timing
//! here repairs the spike COUNT (the rate) and leaves the delivery jitter of up to one tick where
//! it was. The two modes still agree spike for spike, and
//! `the_two_modes_agree_with_exact_timing` is the test; the event-driven catch-up walks a
//! refractory period tick by tick, because the refractory clock's rounding depends on how the
//! interval was cut, and jumps only the quiet remainder.
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
    /// [`Sim::with_exact_timing`] was asked of a model that cannot resolve a spike inside a tick.
    NoExactTiming,
    /// The tick is not a positive, finite number of seconds.
    BadTick,
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
            Self::NoExactTiming => f.write_str(
                "this neuron model cannot resolve a spike inside a tick, so a simulation of it \
                 cannot keep exact spike counts",
            ),
            Self::BadTick => f.write_str("the tick must be a positive, finite number of seconds"),
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
    /// Whether neurons are advanced by [`Neuron::step_counted`]. See [`Sim::with_exact_timing`].
    exact_timing: bool,
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
            exact_timing: false,
        })
    }

    /// Resolve spikes INSIDE the tick: a neuron may fire more than once in a tick, no interval is
    /// rounded up to the grid, and spike counts stop depending on `dt`. See the module
    /// documentation for what this does and does not change.
    ///
    /// # Errors
    ///
    /// [`SimError::NoExactTiming`] for a model without [`Neuron::EXACT_TIMING`];
    /// [`SimError::BadTick`] for a tick the exact solvers would refuse on every step.
    pub fn with_exact_timing(mut self) -> Result<Self, SimError> {
        if !N::EXACT_TIMING {
            return Err(SimError::NoExactTiming);
        }
        if !(self.dt > 0.0) || !self.dt.is_finite() {
            return Err(SimError::BadTick);
        }
        self.exact_timing = true;
        Ok(self)
    }

    /// Whether this simulation resolves spikes inside the tick.
    #[must_use]
    pub fn exact_timing(&self) -> bool {
        self.exact_timing
    }

    /// Advance neuron `i` one tick and append it to `fired` once per spike.
    fn advance(&mut self, i: usize, ext: f64, fired: &mut Vec<u32>) {
        let count = if self.exact_timing {
            self.neurons[i]
                .step_counted(self.dt, ext)
                .unwrap_or_else(|| panic!("neuron {i} refused a step under external current {ext}"))
        } else {
            u32::from(self.neurons[i].step(self.dt, ext))
        };
        for _ in 0..count {
            fired.push(i as u32);
        }
    }

    /// Advance one tick. `external` is a per-neuron input current in amperes, or empty for none.
    ///
    /// Returns the neurons that fired on this tick — with exact timing, once for EACH spike, so a
    /// neuron that fired twice in the tick appears twice.
    ///
    /// # Panics
    ///
    /// If `external` is neither empty nor `net.n` long — a half-length current vector would
    /// silently drive only the first half of the network. With exact timing, also if a neuron
    /// refuses the step, which a non-finite external current makes it do.
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
                    self.advance(i, ext, &mut fired);
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
                    self.advance(i, ext, &mut fired);
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
        if self.exact_timing {
            // The exact solvers keep the refractory clock in seconds and serve it as
            // `min(remaining, step)`, so how much is left after k ticks depends — in the last
            // place — on whether it was subtracted k times or once. A remainder of 4e-19 s is
            // enough to make a neuron ignore a delivery the clocked run accepted. So the refractory
            // part of the gap is walked tick by tick, the same subtractions in the same order as
            // the clocked mode, and only the quiet remainder is jumped.
            let mut left = gap;
            while left > 0 && self.neurons[i].refractory_left() > 0.0 {
                let _ = self.neurons[i].step_counted(self.dt, 0.0);
                left -= 1;
            }
            if left > 0 {
                let _ = self.neurons[i].step_counted(left as f64 * self.dt, 0.0);
            }
        } else if r > 0.0 {
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

    // ---- exact spike timing ------------------------------------------------------------------

    /// Spikes of neuron `source` in a train.
    fn count_of(train: &crate::spike::Train, source: u32) -> usize {
        train.spikes().iter().filter(|s| s.source == source).count()
    }

    #[test]
    fn with_exact_timing_a_driven_neuron_fires_at_its_closed_form_rate_at_any_tick() {
        // One LIF under 3 nA for one second. Its interval is τ ln 2 + t_ref = 15.863 ms, the first
        // spike comes one t_ref early, so it fires ⌊(1 s + t_ref)/isi⌋ = 63 times — 63.17 before
        // the floor, clear of a whole number.
        let cell = Lif::default();
        let isi = cell.isi(3e-9).unwrap();
        let exact = ((1.0 + cell.t_ref) / isi).floor() as usize;
        assert_eq!(exact, 63);
        assert!(((1.0 + cell.t_ref) / isi).fract() > 0.1 && ((1.0 + cell.t_ref) / isi).fract() < 0.9);
        for (ticks, dt) in [(10_000u64, 1e-4), (1_000, 1e-3), (400, 2.5e-3), (200, 5e-3)] {
            for mode in [Mode::Clocked, Mode::EventDriven] {
                let net = NetBuilder::new(1).build();
                let mut sim = Sim::new(net, vec![cell], dt, mode).unwrap().with_exact_timing().unwrap();
                assert!(sim.exact_timing());
                assert_eq!(sim.run(ticks, &[3e-9]).len(), exact, "dt = {dt}, {mode:?}");
            }
            // Without it the interval is rounded up to the grid: ⌈charge/dt⌉ + ⌈t_ref/dt⌉ ticks,
            // the first spike after ⌈charge/dt⌉. Derived here, not typed.
            let charge = ((isi - cell.t_ref) / dt).ceil() as u64;
            let period = charge + (cell.t_ref / dt).ceil() as u64;
            let on_the_grid = ((ticks - charge) / period + 1) as usize;
            let mut sim = Sim::new(NetBuilder::new(1).build(), vec![cell], dt, Mode::Clocked).unwrap();
            assert!(!sim.exact_timing());
            assert_eq!(sim.run(ticks, &[3e-9]).len(), on_the_grid, "dt = {dt} on the grid");
            assert!(on_the_grid <= exact);
            if dt >= 1e-3 {
                assert!(on_the_grid < exact, "dt = {dt}: the grid should cost spikes");
            }
        }
    }

    #[test]
    fn every_spike_inside_a_tick_is_posted() {
        // Neuron 0 is driven hard enough to fire more than once a tick; neuron 1 is a counter —
        // no leak to speak of, a threshold it never reaches — so its potential is the number of
        // deliveries it received.
        let fast = Lif { t_ref: 1e-4, ..Lif::default() };
        let counter = Lif { tau_m: 1e9, v_th: 1.0, ..Lif::default() };
        let mut b = NetBuilder::new(2);
        b.connect(0, 1, 1e-3, 0).unwrap();
        let (dt, ticks) = (1e-3, 150u64);
        let isi = fast.isi(60e-9).unwrap();
        assert!(isi < dt, "the fixture needs more than one spike a tick: isi = {isi}");
        let want = (ticks as f64 * dt + fast.t_ref) / isi;
        assert!(want.fract() > 0.1 && want.fract() < 0.9, "{want} is too near a whole number to test a count");
        let want = want.floor() as usize;
        for mode in [Mode::Clocked, Mode::EventDriven] {
            let mut sim = Sim::new(b.clone().build(), vec![fast, counter], dt, mode).unwrap().with_exact_timing().unwrap();
            let train = sim.run(ticks, &[60e-9, 0.0]);
            assert_eq!(count_of(&train, 0), want, "{mode:?}");
            assert!(want > ticks as usize, "{want} spikes in {ticks} ticks");
            assert_eq!(sim.ledger.spikes_out, want as u64);
            // One more tick delivers the last of them.
            sim.step(&[60e-9, 0.0]);
            let delivered = (sim.read_potentials()[1] - counter.v_rest) / 1e-3;
            assert!((delivered - want as f64).abs() < 1e-6, "{mode:?}: {delivered} deliveries for {want} spikes");
            assert_eq!(count_of(&train, 1), 0);
        }
    }

    #[test]
    fn the_two_modes_agree_with_exact_timing() {
        // A chain in which each spike fires the next neuron at the very start of a tick, and the
        // driver's interval (4.01 ticks) keeps landing deliveries on the tick a 4-tick refractory
        // period ends — where 0.0004 − 4 × 0.0001 is 2.7e-20, not zero, and a catch-up that
        // jumped the period in one subtraction would disagree with the clocked run about whether
        // the neuron was still refractory.
        let dt = 1e-4;
        let mut r = 4e-4f64;
        for _ in 0..4 {
            r -= r.min(dt);
        }
        assert!(r > 0.0, "the fixture depends on this rounding: {r}");
        let net = chain(6, 20e-3, 2);
        let mut cells = vec![Lif { t_ref: 4e-4, ..Lif::default() }; 6];
        cells[0].t_ref = 2e-4;
        let mut ext = vec![0.0; 6];
        ext[0] = 150e-9;
        let mut a = Sim::new(net.clone(), cells.clone(), dt, Mode::Clocked).unwrap().with_exact_timing().unwrap();
        let mut b = Sim::new(net, cells, dt, Mode::EventDriven).unwrap().with_exact_timing().unwrap();
        let (ta, tb) = (a.run(4_000, &ext), b.run(4_000, &ext));
        assert!(count_of(&ta, 5) > 100, "only {} spikes reached the end of the chain", count_of(&ta, 5));
        assert!(count_of(&ta, 1) < count_of(&ta, 0), "no delivery was ever refused as refractory");
        assert_eq!(ta.spikes(), tb.spikes(), "the two modes disagreed");
        for (va, vb) in a.read_potentials().iter().zip(b.read_potentials()) {
            assert!((va - vb).abs() < 1e-12, "{va} against {vb}");
        }
        assert_eq!(a.ledger.syn_ops, b.ledger.syn_ops);
        assert!(b.ledger.neuron_updates() < a.ledger.neuron_updates());
    }

    #[test]
    fn with_exact_timing_a_long_quiet_gap_is_jumped_by_exactly_its_own_length() {
        // The two earlier tests keep every neuron busy, so a catch-up never has much quiet to
        // jump. Here neuron 0 fires about every forty ticks and the cell it drives is given just
        // over half the voltage it needs, so whether it fires on the SECOND bump is decided by
        // how much of the first has leaked away in between — which is exactly what the quiet part
        // of a gap integrates, and what a catch-up that skipped it would keep.
        let dt = 1e-4;
        let proto = Lif { t_ref: 5e-4, ..Lif::default() };
        let drive = 9e-9;
        // The driver's interval in ticks, MEASURED rather than typed, and the fixture's arithmetic
        // derived from it.
        let gap = {
            let mut probe = proto;
            let mut ticks = 0u64;
            while probe.step_exact(dt, drive).unwrap() == 0 {
                ticks += 1;
                assert!(ticks < 10_000, "the driver never fired at {drive} A");
            }
            ticks
        };
        assert!((30..80).contains(&gap), "the driver fires every {gap} ticks");
        let (span, decay) = (proto.v_th - proto.v_rest, (-(gap as f64) * dt / proto.tau_m).exp());
        let bump = 0.53 * span;
        assert!(bump < span, "one bump alone must not fire the cell");
        assert!(2.0 * bump > span, "two bumps with NO leak between them must fire it");
        assert!(bump * (1.0 + decay) < span, "two bumps with the leak must NOT: {decay}");
        let mut b = NetBuilder::new(2);
        b.connect(0, 1, bump, 0).unwrap();
        let ext = vec![drive, 0.0];
        let mut clocked = Sim::new(b.clone().build(), vec![proto; 2], dt, Mode::Clocked).unwrap().with_exact_timing().unwrap();
        let mut driven = Sim::new(b.build(), vec![proto; 2], dt, Mode::EventDriven).unwrap().with_exact_timing().unwrap();
        let (ta, tb) = (clocked.run(4_000, &ext), driven.run(4_000, &ext));
        let (driver, follower) = (count_of(&ta, 0), count_of(&ta, 1));
        assert!(follower > 5, "the driven cell fired {follower} times");
        assert!(follower * 2 < driver, "it fired on {follower} of {driver} bumps, so the leak decided little");
        assert_eq!(ta.spikes(), tb.spikes(), "the two modes disagreed");
        // And the potentials agree after the run's quiet tail: a catch-up one tick too long shows
        // up here as a factor exp(-dt/tau) that no spike time need have noticed.
        let (va, vb) = (clocked.read_potentials(), driven.read_potentials());
        for (a, b) in va.iter().zip(&vb) {
            assert!((a - b).abs() < 1e-15, "{a} against {b}");
        }
        // That comparison is only worth anything on a neuron that is NOT sitting at its resting
        // potential — where any amount of extra decay changes nothing at all, and where the
        // follower above lands, because it resets to rest when it fires. So: one bump and then
        // silence, against the closed form the membrane must be on when the run ends.
        let mut b = NetBuilder::new(2);
        b.connect(0, 1, bump, 0).unwrap();
        let ticks = 200u64;
        let mut quiet = Sim::new(b.build(), vec![proto; 2], dt, Mode::EventDriven).unwrap().with_exact_timing().unwrap();
        let mut when = None;
        for t in 0..ticks {
            // The driver is fed until it fires once, and then the network is left alone.
            let now = if when.is_none() { vec![drive, 0.0] } else { vec![0.0, 0.0] };
            if quiet.step(&now).contains(&0) {
                when = Some(t);
            }
        }
        let when = when.expect("the driver never fired");
        // Delivered at the tick after it fired, and decaying from the end of that tick onward.
        let elapsed = (ticks - when - 1) as f64 * dt;
        let want = proto.v_rest + bump * (-elapsed / proto.tau_m).exp();
        assert!(want - proto.v_rest > 1e-4, "the bump had decayed to rest, so this compares nothing");
        let got = quiet.read_potentials()[1];
        assert!((got - want).abs() < 1e-15, "the caught-up potential is {got} for a closed form of {want}");
    }

    #[test]
    fn exact_timing_is_refused_where_it_cannot_be_kept() {
        const { assert!(!Izhikevich::EXACT_TIMING && Lif::EXACT_TIMING && crate::neuron::AdaptiveLif::EXACT_TIMING) };
        let cells = vec![Izhikevich::regular_spiking(); 3];
        let sim = Sim::new(chain(3, 20e-3, 1), cells, 1e-4, Mode::Clocked).unwrap();
        assert_eq!(sim.with_exact_timing().err(), Some(SimError::NoExactTiming));
        for dt in [0.0, -1e-4, f64::NAN, f64::INFINITY] {
            let sim = Sim::new(chain(3, 20e-3, 1), vec![Lif::default(); 3], dt, Mode::Clocked).unwrap();
            assert_eq!(sim.with_exact_timing().err(), Some(SimError::BadTick), "dt = {dt}");
        }
        assert!(SimError::NoExactTiming.to_string().contains("inside a tick"));
        assert!(SimError::BadTick.to_string().contains("positive"));
        // The default hook is `step`, counted: an Izhikevich cell reports one spike where `step`
        // reports true, and the LIF's hook is its exact solver.
        let (mut a, mut b) = (Izhikevich::regular_spiking(), Izhikevich::regular_spiking());
        let mut spikes = 0;
        for _ in 0..2_000 {
            let counted = a.step_counted(1e-3, 10.0).unwrap();
            assert_eq!(counted, u32::from(b.step(1e-3, 10.0)));
            spikes += counted;
        }
        assert!(spikes > 3, "the comparison never saw a spike");
        let (mut a, mut b) = (Lif::default(), Lif::default());
        assert_eq!(a.step_counted(0.1, 3e-9), b.step_exact(0.1, 3e-9));
        assert_eq!(a.step_counted(0.1, 3e-9), Some(6));
        assert_eq!(a.step_counted(0.1, f64::NAN), None);
    }

    #[test]
    #[should_panic(expected = "refused a step")]
    fn with_exact_timing_a_non_finite_current_is_a_panic_not_a_silent_tick() {
        let mut sim = Sim::new(NetBuilder::new(1).build(), vec![Lif::default()], 1e-3, Mode::Clocked).unwrap().with_exact_timing().unwrap();
        sim.step(&[f64::NAN]);
    }

    /// Pins the catch-up against the ticks that actually passed, twice in a row. Every other
    /// event-driven test keeps its neurons busy, so a neuron's `as_of` is always the tick it was
    /// last updated on and a gap is never more than a delivery apart; here one neuron is never
    /// touched at all, so the whole elapsed run is one jump. That exposes three things the suite
    /// could not see: the tick every neuron is DATED from when the simulation is built, whether a
    /// gap in ticks becomes seconds by multiplying or by dividing, and which tick a catch-up
    /// leaves the neuron dated as of. The neuron starts below threshold and only decays, so the
    /// jump cannot spike, and both readings are the same operations in the same order as the
    /// membrane's own, so they are equalities.
    #[test]
    fn an_untouched_neuron_is_caught_up_over_exactly_the_ticks_that_have_passed() {
        let dt = 1e-3;
        let proto = Lif { v: -55e-3, ..Lif::default() };
        let mut sim = Sim::new(NetBuilder::new(1).build(), vec![proto], dt, Mode::EventDriven).unwrap();
        let decay = (-(5.0 * dt) / proto.tau_m).exp();
        sim.run(5, &[]);
        let after_five = proto.v_rest + (proto.v - proto.v_rest) * decay;
        assert_eq!(sim.read_potentials()[0], after_five);
        // A second read five ticks later jumps five more, not four: the catch-up dates the neuron
        // as of the tick it reached, which is the tick the simulation is ON.
        sim.run(5, &[]);
        let after_ten = proto.v_rest + (after_five - proto.v_rest) * decay;
        assert_eq!(sim.read_potentials()[0], after_ten);
        assert!(after_ten < after_five && after_ten > proto.v_rest, "{after_ten} against {after_five}");
    }

    /// Pins where a refractory catch-up stops at both ends. The refractory period is five and a
    /// HALF ticks here, so rounding it up and rounding it down land on different ticks — at the
    /// two-millisecond default and a 0.1 ms tick it is exactly twenty ticks and they do not, which
    /// is why `the_two_modes_produce_the_same_spike_train` says so in a comment and cannot see
    /// this. Four things are pinned: that the refractory part is jumped separately from the quiet
    /// part, that its length is rounded UP, that the quiet part is the REMAINDER rather than the
    /// whole gap, and that a gap shorter than the period stops at the current tick instead of
    /// running the countdown out. The last of those is invisible in the potential — it is the
    /// countdown itself — so the countdown is read.
    #[test]
    fn a_refractory_catch_up_rounds_up_to_a_tick_and_stops_at_the_current_tick() {
        let dt = 1e-4;
        let proto = Lif { t_ref: 5.5e-4, v_reset: -55e-3, ..Lif::default() };
        assert!((proto.t_ref / dt).fract() > 0.4, "the fixture needs a fractional refractory period");
        let settle = |quiet_ticks: u64| -> (f64, f64) {
            let mut sim =
                Sim::new(NetBuilder::new(1).build(), vec![proto], dt, Mode::EventDriven).unwrap();
            let mut spent = 0u64;
            while !sim.step(&[3e-9]).contains(&0) {
                spent += 1;
                assert!(spent < 10_000, "the cell never fired");
            }
            // The drive is withdrawn, so nothing touches the cell again and the gap is exactly
            // `quiet_ticks` long when the readout catches it up.
            for _ in 0..quiet_ticks {
                sim.step(&[0.0]);
            }
            let v = sim.read_potentials()[0];
            let left = sim.neurons[0].refractory;
            (v, left)
        };
        // A gap LONGER than the period: six of its fifty ticks are refractory — 5.5 rounded up —
        // and the other forty-four are integrated, once.
        let remainder = 50.0 * dt - 6.0 * dt;
        let (v, left) = settle(50);
        assert_eq!(v, proto.v_rest + (proto.v_reset - proto.v_rest) * (-remainder / proto.tau_m).exp());
        assert_eq!(left, proto.t_ref - 6.0 * dt);
        assert!(v < proto.v_th && v > proto.v_rest, "the jump must decay without firing: {v}");
        // A gap SHORTER than the period: the jump stops at the current tick, and what is left of
        // the countdown is still to run.
        let (v, left) = settle(3);
        assert_eq!(v, proto.v_reset);
        assert_eq!(left, proto.t_ref - 3.0 * dt);
    }

    /// Pins the two counters nothing else reads: a weight FETCH for every delivery, and a neuron
    /// held up only by an external current counted as DRIVEN work rather than idle. The suite's
    /// only statement about the split is `idle_fraction() > 0.8` on a clocked run, which a mutant
    /// that bills external drive as idle satisfies even more comfortably; and `syn_fetches` was
    /// read by no test in this module at all, while it is the counter the crate's whole position
    /// on energy rests on. All three numbers are exact integer counts.
    #[test]
    fn every_delivery_is_billed_as_a_fetch_and_an_external_drive_is_billed_as_driven_work() {
        let dt = 1e-4;
        let ticks = 5_000u64;
        let proto = Lif::default();
        let mut bld = NetBuilder::new(2);
        bld.connect(0, 1, 5e-3, 0).unwrap();
        let mut sim = Sim::new(bld.build(), vec![proto; 2], dt, Mode::Clocked).unwrap();
        let train = sim.run(ticks, &[3e-9, 0.0]);
        // A spike of neuron 0 at tick `t` is delivered at `t + 1`, so the ones at the very last
        // tick of the run are still in the ring when it ends.
        let landed = train.of(0).iter().filter(|s| s.t + 1 < ticks).count() as u64;
        assert!(landed > 20, "only {landed} deliveries; the count would prove little");
        assert_eq!(sim.ledger.syn_ops, landed);
        assert_eq!(sim.ledger.syn_fetches, landed, "a delivery was billed as an operation but not as a fetch");
        // Neuron 0 is held up by a current and nothing else, on every tick; neuron 1 is driven
        // only on the ticks a spike reaches it.
        assert_eq!(sim.ledger.neuron_updates_driven, ticks + landed);
        assert_eq!(sim.ledger.neuron_updates_idle, ticks - landed);
    }

    /// Pins that the external-current scan does not queue a neuron the delivery loop has already
    /// queued. The guard is only reachable when the volts arriving at a neuron SUM to zero — the
    /// running total is what the loop tests — and every fixture in this module has one delivery
    /// per target per tick, so the guard was never exercised. In event-driven mode the queue IS
    /// the update list, so a duplicate entry steps that neuron twice inside one tick: the two
    /// modes then disagree, and the ledger over-counts. Every neuron here carries a current, so
    /// every neuron is queued on every tick — once.
    #[test]
    fn the_external_current_scan_never_queues_a_neuron_that_is_already_queued() {
        let dt = 1e-4;
        let ticks = 6_000u64;
        let proto = Lif::default();
        let mut bld = NetBuilder::new(3);
        bld.connect(0, 2, 12e-3, 0).unwrap();
        bld.connect(1, 2, -12e-3, 0).unwrap();
        let mut clocked = Sim::new(bld.clone().build(), vec![proto; 3], dt, Mode::Clocked).unwrap();
        let mut driven = Sim::new(bld.build(), vec![proto; 3], dt, Mode::EventDriven).unwrap();
        let ext = [3e-9, 3e-9, 2.5e-9];
        let ta = clocked.run(ticks, &ext);
        let tb = driven.run(ticks, &ext);
        // Neurons 0 and 1 are identical cells under an identical current, so they fire together
        // and their two weights cancel exactly at neuron 2.
        assert_eq!(driven.ledger.syn_ops % 2, 0, "the two presynaptic cells did not fire in step");
        assert!(driven.ledger.syn_ops > 60, "only {} deliveries", driven.ledger.syn_ops);
        assert_eq!(ta.spikes(), tb.spikes(), "the two modes disagreed");
        assert_eq!(driven.ledger.neuron_updates_driven, 3 * ticks);
    }

    /// Pins the tick a spike is STAMPED with, against a bare neuron stepped by hand. Every timing
    /// test in this module compares one spike time with another — a delivery against its source, a
    /// clocked train against an event-driven one — and all of those hold when every stamp in the
    /// run is shifted by the same tick. This one fixes the origin.
    #[test]
    fn a_spike_is_stamped_with_the_tick_it_fired_on() {
        let dt = 1e-3;
        let proto = Lif::default();
        let drive = 3e-9;
        let mut probe = proto;
        let mut first = 0u64;
        while !probe.step(dt, drive) {
            first += 1;
            assert!(first < 10_000, "the probe never fired");
        }
        let mut sim = Sim::new(NetBuilder::new(1).build(), vec![proto], dt, Mode::Clocked).unwrap();
        let train = sim.run(200, &[drive]);
        assert!(train.len() > 5, "only {} spikes", train.len());
        assert_eq!(train.spikes()[0].t, first);
        assert!(train.spikes().last().unwrap().t < 200, "a spike was stamped past the end of the run");
    }

    /// Pins what a reset does and what it does NOT do. `reset_state` was called by no test in this
    /// module, so both halves of its contract were unread: that every neuron goes back to rest,
    /// and that the ledger — a record of what the hardware already did — survives.
    #[test]
    fn a_reset_returns_every_neuron_to_rest_and_keeps_the_ledger() {
        let dt = 1e-4;
        let proto = Lif { v_reset: -60e-3, ..Lif::default() };
        let mut bld = NetBuilder::new(3);
        bld.connect(0, 1, 20e-3, 2).unwrap();
        bld.connect(1, 2, 20e-3, 2).unwrap();
        let mut sim = Sim::new(bld.build(), vec![proto; 3], dt, Mode::Clocked).unwrap();
        sim.run(1_000, &[3e-9, 0.0, 0.0]);
        let paid = sim.ledger;
        assert!(paid.syn_ops > 0 && paid.spikes_out > 0 && paid.neuron_updates() > 0, "{paid:?}");
        assert!(
            sim.neurons.iter().any(|c| c.v != c.v_rest || c.refractory != 0.0),
            "every neuron was already at rest, so a reset that did nothing would look the same"
        );
        sim.reset_state();
        assert_eq!(sim.ledger, paid, "the run had already paid for the first half");
        for (i, cell) in sim.neurons.iter().enumerate() {
            assert_eq!(cell.v, cell.v_rest, "neuron {i} kept its potential");
            assert_eq!(cell.refractory, 0.0, "neuron {i} kept its refractory countdown");
        }
        // And the deliveries still in flight went with it: a quiet run after the reset is silent.
        let after = sim.run(500, &[]);
        assert!(after.is_empty(), "{} spikes after a reset into silence", after.len());
        assert_eq!(sim.ledger.syn_ops, paid.syn_ops, "a delivery outlived the reset");
    }
}
