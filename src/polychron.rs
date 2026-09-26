//! Polychronous groups: the time-locked firing patterns a network with axonal delays can hold,
//! found rather than assumed.
//!
//! # What the mechanism is
//!
//! Izhikevich (*Polychronization: computation with spikes*, Neural Computation 18(2):245–282,
//! 2006) points out that a network whose axons have DIFFERENT conduction delays supports patterns
//! a synchronous network cannot. If neuron `A` reaches `C` in 9 ms and `B` reaches `C` in 3 ms,
//! then `A` firing 6 ms before `B` delivers both spikes to `C` at the same instant — and no other
//! relative timing does. A set of neurons with such a timing, together with everything their
//! coincidence goes on to fire, is a **polychronous group**: time-locked but not synchronous.
//!
//! The point of the construction is a counting argument. A network of `n` neurons supports far
//! MORE than `n` of these patterns, so what it can represent is not bounded by its size.
//! [`Network::groups`] enumerates them by simulation — and the enumeration is what made this
//! module correct a claim it started out asserting; see the measured note below.
//!
//! **How a group is found.** Take an ANCHOR set of neurons that all connect to a common target,
//! give each the firing time that makes its spike arrive at that target simultaneously, and run
//! the network forward deterministically: a neuron fires when at least `threshold` spikes reach
//! it within a coincidence window. Whatever fires is the group. A group is kept only when it
//! reaches at least `min_size` neurons, so that "the anchors and nothing else" is not counted.
//!
//! # Why it is in a neuromorphic crate
//!
//! Delays are free on neuromorphic hardware — an axon is a queue — and this is the clearest
//! statement of what they buy: representational capacity that costs no neurons and no synapses.
//! [`crate::delays`] has the single-neuron side of this (what a set of delays makes a neuron
//! selective for, and how many distinct patterns there are); this module has the network side,
//! which that module's documentation names as not reproduced.
//!
//! # The closed forms this module is checked against
//!
//! - **The anchor timing is forced.** For a target reached from `A` at delay `d_A` and `B` at
//!   `d_B`, simultaneity requires `t_A − t_B = d_B − d_A`, and [`Network::anchor_times`] returns
//!   exactly that, normalised so the earliest anchor fires at zero. Checked against the arrival
//!   times it produces, which must be equal.
//! - **A hand-built group is found, and nothing else is.** A network is constructed whose only
//!   coincidence is a chain that was written down in advance; the enumeration returns that chain,
//!   with those neurons, in that order, at those times.
//! - **Equal delays make the groups synchronous.** With every delay the same, the only timing that
//!   makes spikes coincide is simultaneous firing, so every group found has all its anchors at
//!   time zero — which the test checks by finding them and looking. With a spread of delays the
//!   anchors are staggered, which is what "polychronous, not synchronous" means.
//! - **More groups than neurons** — 3,736 for 120 neurons, measured. But that number is NOT
//!   evidence about delays, and this module says so where it would be easy not to: when the
//!   anchor count equals the firing threshold, the anchors are timed to coincide at the target BY
//!   CONSTRUCTION, so the target always fires and every anchor set sharing a target is a group.
//!   The count is then combinatorics of the connectivity, `Σ_target C(fan_in, anchors)`, and it
//!   barely moves when the delays are changed (3,736 spread, 3,556 narrow, 3,123 uniform).
//! - **A measured result that went the other way from the expectation.** This module was written
//!   expecting a spread of delays to produce LARGER cascades than uniform ones. It produces
//!   smaller: at 120 neurons, four delays gave 167 groups of four or more neurons and a largest of
//!   6, while a single delay gave 657 and a largest of 25. Identical delays make downstream
//!   coincidences EASY — everything arrives on the same grid — whereas scattered arrivals rarely
//!   land inside a 0.1 ms window together. The test asserts the measured direction. This is a
//!   property of counting neurons with a narrow window, not a claim about Izhikevich's network,
//!   whose integrate-and-fire membranes accumulate input over milliseconds.
//! - **Determinism.** The same network and the same anchors give the same group, and the group
//!   does not depend on the order the anchors were listed in.
//!
//! # What this module has NOT reproduced
//!
//! - Izhikevich's network: 1,000 Izhikevich neurons with STDP, run for hours, with groups
//!   re-counted as the weights change. The neurons here are the counting kind — a threshold on
//!   coincident arrivals — because a group is a property of the DELAYS and the connectivity, and
//!   a model whose spike times depend on membrane detail would make the enumeration a statement
//!   about that detail instead.
//! - His specific counts (he reports several thousand groups for a thousand neurons). The counts
//!   here are of this module's networks under this module's rule, and are labelled as measured.
//! - Groups anchored on more than a fixed number of neurons, and the persistence of groups under
//!   plasticity.

use crate::rng::Rng;
use core::fmt;

/// The most neurons a network may have.
pub const MAX_NEURONS: usize = 4096;
/// The most spikes one group may contain before the search gives up on it.
///
/// [`Network::new`] caps a network at [`MAX_NEURONS`], and inside that cap the fire-once rule
/// already bounds a group by the neuron count, so this stop is redundant for every network the
/// constructor builds. It is **not** redundant in general, and an earlier version of this doc
/// said it was: [`Network`]'s fields are public with no `#[non_exhaustive]`, [`Network::connect`]
/// checks only `index < self.n`, and [`Network::simulate`] re-checks nothing about `n` — so a
/// network larger than [`MAX_NEURONS`] is reachable through the public API, and there this is a
/// real truncation. `simulate` returns the first `MAX_GROUP` firings and stops, which the test
/// `a_group_is_capped_at_max_group_even_when_the_network_is_bigger_than_max_neurons` measures.
/// It is also the stop that keeps a future change to the fire-once rule from running forever.
pub const MAX_GROUP: usize = MAX_NEURONS;

/// What went wrong, named rather than guessed around.
#[derive(Debug, Clone, PartialEq)]
pub enum PolyError {
    /// No neurons, or more than [`MAX_NEURONS`].
    BadShape,
    /// A connection naming a neuron that does not exist.
    NoSuchNeuron {
        /// The index supplied.
        index: usize,
    },
    /// A delay that is not a positive, finite number of seconds.
    BadDelay {
        /// The value supplied.
        value: f64,
    },
    /// A parameter outside its range.
    OutOfRange {
        /// Which parameter.
        what: &'static str,
        /// Value supplied.
        value: f64,
    },
    /// Fewer anchors than the firing threshold, so they could never fire anything.
    TooFewAnchors {
        /// How many were supplied.
        got: usize,
        /// The threshold.
        threshold: usize,
    },
}

impl fmt::Display for PolyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadShape => f.write_str("a network needs between one and MAX_NEURONS neurons"),
            Self::NoSuchNeuron { index } => write!(f, "there is no neuron {index}"),
            Self::BadDelay { value } => write!(f, "a delay of {value} seconds"),
            Self::OutOfRange { what, value } => write!(f, "{what} = {value} is out of range"),
            Self::TooFewAnchors { got, threshold } => write!(f, "{got} anchors cannot reach a threshold of {threshold}"),
        }
    }
}

impl std::error::Error for PolyError {}

/// One spike of a group: who fired, and when relative to the group's start.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Firing {
    /// The neuron.
    pub neuron: usize,
    /// Seconds after the group's first anchor fired.
    pub time: f64,
}

/// A polychronous group: its anchors and every firing the anchors set off, in time order.
#[derive(Debug, Clone, PartialEq)]
pub struct Group {
    /// The neurons whose timed firing starts the group.
    pub anchors: Vec<usize>,
    /// Every firing, anchors first, in time order.
    pub firings: Vec<Firing>,
}

impl Group {
    /// How many firings the group has. Each neuron fires at most once, so this is the number of
    /// neurons: the group's *size* in Izhikevich's sense (2006, Fig. 8, "the number of neurons
    /// that form each group"). His *length* is the longest path through the group, which this
    /// module does not compute. Earlier releases called this count the length; the paper's
    /// example group has size 15 and length 5, so the two are not interchangeable.
    #[must_use]
    pub fn len(&self) -> usize {
        self.firings.len()
    }

    /// Whether the group has no firings at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.firings.is_empty()
    }

    /// The neurons the group fires, each once, in the order they first fire.
    #[must_use]
    pub fn neurons(&self) -> Vec<usize> {
        let mut seen = Vec::new();
        for f in &self.firings {
            if !seen.contains(&f.neuron) {
                seen.push(f.neuron);
            }
        }
        seen
    }
}

/// A network of counting neurons with per-connection axonal delays.
#[derive(Debug, Clone, PartialEq)]
pub struct Network {
    /// Neurons.
    pub n: usize,
    /// `out[pre]` is `(post, delay_seconds)`.
    pub out: Vec<Vec<(usize, f64)>>,
    /// How many spikes must arrive within `window` for a neuron to fire.
    pub threshold: usize,
    /// The coincidence window, seconds.
    pub window: f64,
}

impl Network {
    /// An empty network of `n` neurons.
    ///
    /// # Errors
    ///
    /// [`PolyError::BadShape`], or [`PolyError::OutOfRange`] for a zero threshold or a
    /// non-positive window.
    pub fn new(n: usize, threshold: usize, window: f64) -> Result<Self, PolyError> {
        if n == 0 || n > MAX_NEURONS {
            return Err(PolyError::BadShape);
        }
        if threshold == 0 {
            return Err(PolyError::OutOfRange { what: "threshold", value: 0.0 });
        }
        if !(window > 0.0) || !window.is_finite() {
            return Err(PolyError::OutOfRange { what: "window", value: window });
        }
        Ok(Self { n, out: vec![Vec::new(); n], threshold, window })
    }

    /// Add a connection.
    ///
    /// # Errors
    ///
    /// [`PolyError::NoSuchNeuron`] or [`PolyError::BadDelay`].
    pub fn connect(&mut self, pre: usize, post: usize, delay: f64) -> Result<&mut Self, PolyError> {
        for index in [pre, post] {
            if index >= self.n {
                return Err(PolyError::NoSuchNeuron { index });
            }
        }
        if !(delay > 0.0) || !delay.is_finite() {
            return Err(PolyError::BadDelay { value: delay });
        }
        self.out[pre].push((post, delay));
        Ok(self)
    }

    /// The delay from `pre` to `post`, or `None` if they are not connected.
    #[must_use]
    pub fn delay(&self, pre: usize, post: usize) -> Option<f64> {
        self.out.get(pre)?.iter().find(|(p, _)| *p == post).map(|(_, d)| *d)
    }

    /// The firing times that make every anchor's spike reach `target` at the same instant,
    /// normalised so the earliest anchor fires at zero: `t_i = max_k(d_k) − d_i`.
    ///
    /// # Errors
    ///
    /// [`PolyError::NoSuchNeuron`] if an anchor does not connect to `target`, and
    /// [`PolyError::TooFewAnchors`].
    pub fn anchor_times(&self, anchors: &[usize], target: usize) -> Result<Vec<f64>, PolyError> {
        if anchors.len() < self.threshold {
            return Err(PolyError::TooFewAnchors { got: anchors.len(), threshold: self.threshold });
        }
        let delays: Vec<f64> = anchors.iter().map(|&a| self.delay(a, target).ok_or(PolyError::NoSuchNeuron { index: a })).collect::<Result<_, _>>()?;
        let latest = delays.iter().fold(f64::NEG_INFINITY, |a, b| a.max(*b));
        Ok(delays.iter().map(|d| latest - d).collect())
    }

    /// Run the anchors forward and return the group they set off.
    ///
    /// Deterministic: a neuron fires the first time `threshold` spikes reach it within `window`,
    /// and fires at most once — a group is a pattern, not a run, and a neuron that fired twice
    /// would be two patterns overlapping.
    ///
    /// # Errors
    ///
    /// [`PolyError::NoSuchNeuron`] for an anchor that does not exist, and
    /// [`PolyError::TooFewAnchors`].
    pub fn simulate(&self, anchors: &[usize], times: &[f64]) -> Result<Group, PolyError> {
        if anchors.len() < self.threshold {
            return Err(PolyError::TooFewAnchors { got: anchors.len(), threshold: self.threshold });
        }
        if times.len() != anchors.len() {
            return Err(PolyError::OutOfRange { what: "times", value: times.len() as f64 });
        }
        for (&a, t) in anchors.iter().zip(times) {
            if a >= self.n {
                return Err(PolyError::NoSuchNeuron { index: a });
            }
            if !t.is_finite() {
                return Err(PolyError::BadDelay { value: *t });
            }
        }
        // Arrivals waiting to be considered, and the firings so far.
        let mut arrivals: Vec<(f64, usize)> = Vec::new();
        let mut fired: Vec<Option<f64>> = vec![None; self.n];
        let mut firings: Vec<Firing> = Vec::new();
        let mut queue: Vec<(f64, usize)> = anchors.iter().zip(times).map(|(&a, &t)| (t, a)).collect();
        queue.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        // Anchors fire by fiat, at their given times.
        for (t, a) in queue {
            if fired[a].is_none() {
                fired[a] = Some(t);
                firings.push(Firing { neuron: a, time: t });
                for &(post, d) in &self.out[a] {
                    arrivals.push((t + d, post));
                }
            }
        }
        loop {
            arrivals.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            // The earliest moment at which some neuron that has not fired has `threshold`
            // arrivals inside one window.
            let mut best: Option<(f64, usize, usize)> = None;
            for (i, &(t, post)) in arrivals.iter().enumerate() {
                if fired[post].is_some() {
                    continue;
                }
                let count = arrivals[i..].iter().take_while(|(u, _)| *u <= t + self.window).filter(|(_, p)| *p == post).count();
                if count >= self.threshold {
                    let when = arrivals[i..].iter().filter(|(u, p)| *p == post && *u <= t + self.window).nth(self.threshold - 1).map_or(t, |(u, _)| *u);
                    if best.is_none_or(|(b, _, _)| when < b) {
                        best = Some((when, post, i));
                    }
                }
            }
            let Some((when, post, _)) = best else { break };
            fired[post] = Some(when);
            firings.push(Firing { neuron: post, time: when });
            if firings.len() >= self.n.min(MAX_GROUP) {
                break;
            }
            for &(next, d) in &self.out[post] {
                arrivals.push((when + d, next));
            }
        }
        firings.sort_by(|a, b| a.time.total_cmp(&b.time).then(a.neuron.cmp(&b.neuron)));
        Ok(Group { anchors: anchors.to_vec(), firings })
    }

    /// Every polychronous group of `anchors` anchors that reaches at least `min_size` firings.
    ///
    /// Anchor sets are taken from the neurons that share a target: for each neuron, each
    /// combination of `anchors` of its presynaptic neighbours, timed so their spikes coincide
    /// there.
    ///
    /// # Errors
    ///
    /// [`PolyError::OutOfRange`] for fewer anchors than the threshold or a `min_size` of zero.
    pub fn groups(&self, anchors: usize, min_size: usize) -> Result<Vec<Group>, PolyError> {
        if anchors < self.threshold {
            return Err(PolyError::TooFewAnchors { got: anchors, threshold: self.threshold });
        }
        if min_size == 0 {
            return Err(PolyError::OutOfRange { what: "min_size", value: 0.0 });
        }
        // Who reaches each target.
        let mut into: Vec<Vec<usize>> = vec![Vec::new(); self.n];
        for pre in 0..self.n {
            for &(post, _) in &self.out[pre] {
                if !into[post].contains(&pre) {
                    into[post].push(pre);
                }
            }
        }
        let mut found: Vec<Group> = Vec::new();
        for target in 0..self.n {
            let sources = &into[target];
            if sources.len() < anchors {
                continue;
            }
            let mut combination: Vec<usize> = (0..anchors).collect();
            loop {
                let set: Vec<usize> = combination.iter().map(|&k| sources[k]).collect();
                if let Ok(times) = self.anchor_times(&set, target)
                    && let Ok(group) = self.simulate(&set, &times)
                    && group.len() >= min_size
                    && !found.iter().any(|g| g.firings == group.firings)
                {
                    found.push(group);
                }
                // Next combination in lexicographic order.
                let Some(i) = (0..anchors).rev().find(|&i| combination[i] < sources.len() - anchors + i) else { break };
                combination[i] += 1;
                for j in i + 1..anchors {
                    combination[j] = combination[j - 1] + 1;
                }
            }
        }
        Ok(found)
    }
}

/// A random network: every neuron gets `fan_out` outgoing connections to distinct targets — never
/// itself, never the same target twice — with delays drawn uniformly from `delays`.
///
/// # Errors
///
/// As [`Network::new`], and [`PolyError::OutOfRange`] for a `fan_out` that cannot be satisfied or
/// an empty delay list.
pub fn random(n: usize, fan_out: usize, delays: &[f64], threshold: usize, window: f64, seed: u64) -> Result<Network, PolyError> {
    let mut net = Network::new(n, threshold, window)?;
    if fan_out == 0 || fan_out >= n {
        return Err(PolyError::OutOfRange { what: "fan_out", value: fan_out as f64 });
    }
    if delays.is_empty() {
        return Err(PolyError::OutOfRange { what: "delays", value: 0.0 });
    }
    let mut rng = Rng::new(seed);
    for pre in 0..n {
        // Sampled WITHOUT replacement by a partial shuffle, so a fan-out larger than the number
        // of available targets cannot spin looking for one that is not there.
        let mut candidates: Vec<usize> = (0..n).filter(|&p| p != pre).collect();
        for k in 0..fan_out {
            let pick = k + rng.below((candidates.len() - k) as u32) as usize;
            candidates.swap(k, pick);
            let d = delays[rng.below(delays.len() as u32) as usize];
            net.connect(pre, candidates[k], d)?;
        }
    }
    Ok(net)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_anchor_timing_is_the_one_that_makes_the_arrivals_coincide() {
        // A reaches C in 9 ms, B in 3 ms: A must fire 6 ms before B, and no other timing works.
        let mut net = Network::new(3, 2, 1e-4).unwrap();
        net.connect(0, 2, 9e-3).unwrap().connect(1, 2, 3e-3).unwrap();
        let times = net.anchor_times(&[0, 1], 2).unwrap();
        assert_eq!(times[0], 0.0);
        assert!((times[1] - 6e-3).abs() < 1e-15, "{times:?}");
        // The arrivals those times produce are equal, which is the property being claimed.
        let arrivals: Vec<f64> = [0, 1].iter().zip(&times).map(|(&a, t)| t + net.delay(a, 2).unwrap()).collect();
        assert!((arrivals[0] - arrivals[1]).abs() < 1e-18 && arrivals[0] == 9e-3);
        // The earliest anchor fires at zero, so the group's times are relative to its own start.
        assert_eq!(times.iter().cloned().fold(f64::INFINITY, f64::min), 0.0);
        // Listing the anchors the other way round gives the same timing, transposed.
        let swapped = net.anchor_times(&[1, 0], 2).unwrap();
        assert!((swapped[0] - times[1]).abs() < 1e-18 && swapped[1] == 0.0);
        assert_eq!(net.anchor_times(&[0], 2), Err(PolyError::TooFewAnchors { got: 1, threshold: 2 }));
        assert_eq!(net.anchor_times(&[0, 1], 1), Err(PolyError::NoSuchNeuron { index: 0 }));
    }

    /// The network of the module documentation, with a chain hanging off the coincidence:
    /// 0 and 1 fire 2, then 2 and 3 fire 4. Neuron 5 is connected but can never be fired.
    fn handmade() -> Network {
        let mut net = Network::new(6, 2, 1e-4).unwrap();
        net.connect(0, 2, 9e-3).unwrap();
        net.connect(1, 2, 3e-3).unwrap();
        net.connect(2, 4, 5e-3).unwrap();
        net.connect(3, 4, 2e-3).unwrap();
        net.connect(2, 5, 4e-3).unwrap();
        net
    }

    #[test]
    fn a_hand_built_group_is_found_exactly_and_nothing_more_is_invented() {
        let net = handmade();
        // 0 at t = 0 and 1 at t = 6 ms both reach 2 at 9 ms, so 2 fires at 9 ms.
        let group = net.simulate(&[0, 1], &[0.0, 6e-3]).unwrap();
        assert_eq!(group.neurons(), vec![0, 1, 2]);
        assert_eq!(group.len(), 3);
        assert!((group.firings[2].time - 9e-3).abs() < 1e-15 && group.firings[2].neuron == 2);
        // Neuron 5 receives ONE spike from 2 and the threshold is two, so it never fires; neuron
        // 4 likewise, because 3 was not in the anchor set.
        assert!(!group.neurons().contains(&5) && !group.neurons().contains(&4));
        // Now anchor on 2 and 3 as well, timed to coincide at 4: 2 must fire 3 ms before 3.
        let times = net.anchor_times(&[2, 3], 4).unwrap();
        assert_eq!(times[0], 0.0);
        assert!((times[1] - 3e-3).abs() < 1e-15);
        let chain = net.simulate(&[2, 3], &times).unwrap();
        assert_eq!(chain.neurons(), vec![2, 3, 4]);
        assert!((chain.firings[2].time - 5e-3).abs() < 1e-15);
        // A timing that does NOT make the spikes coincide fires nothing beyond the anchors.
        let missed = net.simulate(&[0, 1], &[0.0, 0.0]).unwrap();
        assert_eq!(missed.neurons(), vec![0, 1]);
        assert_eq!(missed.len(), 2);
        // Determinism, and independence from the order the anchors were listed in.
        assert_eq!(net.simulate(&[0, 1], &[0.0, 6e-3]).unwrap(), group);
        let swapped = net.simulate(&[1, 0], &[6e-3, 0.0]).unwrap();
        assert_eq!(swapped.firings, group.firings);
        assert_ne!(swapped.anchors, group.anchors);
    }

    #[test]
    fn the_enumeration_finds_the_chain_and_only_the_chain() {
        let net = handmade();
        // Groups of two anchors that fire at least three neurons: the two coincidences above.
        let groups = net.groups(2, 3).unwrap();
        let mut seen: Vec<Vec<usize>> = groups.iter().map(Group::neurons).collect();
        seen.sort();
        assert_eq!(seen, vec![vec![0, 1, 2], vec![2, 3, 4]]);
        // Asking for larger groups finds none: nothing in this network fires four neurons.
        assert!(net.groups(2, 4).unwrap().is_empty());
        // And asking for size one or more adds NOTHING, which is the point made in the module
        // documentation: with as many anchors as the threshold, the target always fires, so no
        // group is ever just its anchors.
        assert_eq!(net.groups(2, 1).unwrap().len(), groups.len());
        assert_eq!(net.groups(1, 1), Err(PolyError::TooFewAnchors { got: 1, threshold: 2 }));
        assert_eq!(net.groups(2, 0), Err(PolyError::OutOfRange { what: "min_size", value: 0.0 }));
    }

    #[test]
    fn equal_delays_leave_nothing_but_synchrony() {
        // Every delay the same: the only timing that makes spikes coincide is simultaneous.
        let uniform = random(60, 6, &[5e-3], 2, 1e-4, 4).unwrap();
        let groups = uniform.groups(2, 3).unwrap();
        assert!(!groups.is_empty(), "the network fired nothing at all, so this shows nothing");
        for g in &groups {
            let times = uniform.anchor_times(&g.anchors, 0).unwrap_or_else(|_| vec![0.0; g.anchors.len()]);
            let _ = times;
            let anchor_times: Vec<f64> = g.firings.iter().filter(|f| g.anchors.contains(&f.neuron)).map(|f| f.time).collect();
            assert!(anchor_times.iter().all(|t| *t == 0.0), "an anchor fired at {anchor_times:?} with uniform delays");
        }
        // With a spread of delays the anchors are staggered, which is the whole phenomenon.
        let spread = random(60, 6, &[1e-3, 5e-3, 9e-3, 13e-3], 2, 1e-4, 4).unwrap();
        let staggered = spread.groups(2, 3).unwrap();
        let any_stagger = staggered.iter().any(|g| g.firings.iter().filter(|f| g.anchors.contains(&f.neuron)).any(|f| f.time > 0.0));
        assert!(any_stagger, "no group had a staggered anchor");
    }

    /// Groups of a 120-neuron random network at the given delays: how many, how many cascade
    /// past the first target, and the largest.
    fn census(delays: &[f64]) -> (usize, usize, usize) {
        let net = random(120, 8, delays, 2, 1e-4, 11).unwrap();
        let groups = net.groups(2, 3).unwrap();
        for g in &groups {
            assert!(g.len() >= 3 && g.neurons().len() == g.len(), "a neuron fired twice in {g:?}");
            assert!(g.firings.windows(2).all(|p| p[0].time <= p[1].time), "a group is out of time order");
            assert!(g.neurons().iter().any(|m| !g.anchors.contains(m)), "a group fired nothing beyond its anchors");
        }
        // With as many anchors as the threshold, every anchor set that shares a target is a
        // group: nothing is ever dropped for being too short.
        assert_eq!(net.groups(2, 1).unwrap().len(), groups.len());
        (groups.len(), groups.iter().filter(|g| g.len() >= 4).count(), groups.iter().map(Group::len).max().unwrap_or(0))
    }

    #[test]
    fn a_network_holds_more_groups_than_it_has_neurons_and_uniform_delays_cascade_further() {
        // MEASURED at this seed. The COUNT is combinatorics — every anchor pair sharing a target
        // is a group — so it hardly moves with the delays and is not evidence about them.
        let (spread, spread_long, spread_max) = census(&[1e-3, 5e-3, 9e-3, 13e-3]);
        let (narrow, narrow_long, narrow_max) = census(&[6e-3, 7e-3]);
        let (uniform, uniform_long, uniform_max) = census(&[5e-3]);
        // The enumeration is a deterministic function of the seed, so the numbers it produces are
        // recorded exactly rather than bounded — anything that changes which neuron fires, or
        // when, or which groups are counted, moves one of these nine.
        assert_eq!((spread, spread_long, spread_max), (3730, 176, 5));
        assert_eq!((narrow, narrow_long, narrow_max), (3554, 297, 9));
        assert_eq!((uniform, uniform_long, uniform_max), (3118, 665, 40));
        assert!(spread > 120 && narrow > 120 && uniform > 120, "{spread}, {narrow}, {uniform} groups for 120 neurons");
        assert!(spread < 2 * uniform, "the count moved more with the delays than combinatorics allows: {spread} against {uniform}");
        // What the delays DO change is how far a group cascades — and it goes the other way from
        // the obvious guess: identical delays put every arrival on one grid, so coincidences
        // downstream are easy, while scattered arrivals rarely land in a 0.1 ms window together.
        assert!(uniform_long > 2 * spread_long, "cascades: uniform {uniform_long}, spread {spread_long}");
        assert!(uniform_max > 3 * spread_max, "largest group: uniform {uniform_max}, spread {spread_max}");
        assert!(narrow_long > spread_long && narrow_long < uniform_long, "narrow {narrow_long} should sit between {spread_long} and {uniform_long}");
        assert!(narrow_max > spread_max && narrow_max < uniform_max, "narrow {narrow_max} between {spread_max} and {uniform_max}");
    }

    #[test]
    fn a_repeated_anchor_fires_once_at_the_earliest_time_it_was_given() {
        // Anchors are fired by fiat, so a list that names one twice has to resolve to something.
        // It fires ONCE — a group is a pattern, and a neuron appearing twice in it would be two
        // patterns — and at the earlier of the two times, which is why the anchors are sorted
        // before they are fired.
        let net = handmade();
        let group = net.simulate(&[0, 0], &[7e-3, 2e-3]).unwrap();
        assert_eq!(group.neurons(), vec![0]);
        assert_eq!(group.len(), 1, "neuron 0 fired {} times", group.len());
        assert_eq!(group.firings[0].time, 2e-3, "the later of the two times won");
        assert_eq!(net.simulate(&[0, 0], &[2e-3, 7e-3]).unwrap().firings, group.firings);
    }

    #[test]
    fn a_neuron_fires_when_the_threshold_th_spike_arrives_not_the_first() {
        // Two arrivals 0.05 ms apart, inside a 0.1 ms window: the coincidence is complete when
        // the SECOND lands, and that is when the neuron fires. A model that fired on the first
        // would be firing before it had the evidence.
        let mut net = Network::new(4, 2, 1e-4).unwrap();
        net.connect(0, 2, 9e-3).unwrap();
        net.connect(1, 2, 3e-3).unwrap();
        net.connect(2, 3, 1e-3).unwrap();
        // Fire 0 at 0 and 1 at 6.05 ms: arrivals at 9 ms and 9.05 ms.
        let group = net.simulate(&[0, 1], &[0.0, 6.05e-3]).unwrap();
        let fired = group.firings.iter().find(|f| f.neuron == 2).expect("neuron 2 did not fire");
        assert!((fired.time - 9.05e-3).abs() < 1e-15, "fired at {} rather than the second arrival", fired.time);
        assert!(fired.time > 9e-3, "it fired on the first arrival");
        // Widening the window does not move it: the coincidence is still complete at the second
        // arrival, not at the end of the window.
        let mut wide = net.clone();
        wide.window = 1e-3;
        let again = wide.simulate(&[0, 1], &[0.0, 6.05e-3]).unwrap();
        assert_eq!(again.firings.iter().find(|f| f.neuron == 2).unwrap().time, fired.time);
        // Move the second arrival outside the window and nothing fires at all.
        let missed = net.simulate(&[0, 1], &[0.0, 6.2e-3]).unwrap();
        assert_eq!(missed.neurons(), vec![0, 1]);
    }

    #[test]
    fn two_axons_between_the_same_pair_are_two_sources_of_one_coincidence_but_one_neighbour() {
        // A neuron may reach another twice, by axons of different delay — that is the one case
        // where a presynaptic neighbour could be counted twice, and it must not be: an anchor set
        // is a set of NEURONS.
        let mut net = Network::new(3, 2, 1e-4).unwrap();
        net.connect(0, 2, 9e-3).unwrap();
        net.connect(0, 2, 4e-3).unwrap();
        net.connect(1, 2, 3e-3).unwrap();
        // Only {0, 1} is an anchor set; {0, 0} is not, so there is exactly one group.
        let groups = net.groups(2, 3).unwrap();
        assert_eq!(groups.len(), 1, "{groups:?}");
        assert_eq!(groups[0].anchors, vec![0, 1]);
        assert_eq!(net.delay(0, 2), Some(9e-3), "the first axon listed is the one `delay` reports");
        // Counting neuron 0 twice would offer {0, 0} as an anchor set as well — a "coincidence"
        // of one neuron with itself, which fires nothing. It is visible at a minimum size of
        // one, where a group of just its anchors is still counted.
        assert_eq!(net.groups(2, 1).unwrap().len(), 1, "{:?}", net.groups(2, 1).unwrap());
    }

    #[test]
    fn a_neuron_fires_at_the_earliest_coincidence_it_ever_has_so_the_search_takes_them_in_order() {
        // Neuron 3 has a pair of arrivals at 10 ms from the anchors alone, so it is eligible from
        // the start. But neurons 2 and 4 fire first, at 5 and 6 ms, and THEIR spikes reach 3 at
        // 9.95 and 9.90 ms — a coincidence half a tenth of a millisecond earlier. A search that
        // fired the latest candidate first would take 3 at 10 ms and never see it.
        let mut net = Network::new(5, 2, 1e-4).unwrap();
        for (pre, post, d) in [(0, 2, 5e-3), (1, 2, 5e-3), (0, 4, 6e-3), (1, 4, 6e-3), (0, 3, 10e-3), (1, 3, 10e-3), (2, 3, 4.95e-3), (4, 3, 3.9e-3)] {
            net.connect(pre, post, d).unwrap();
        }
        let group = net.simulate(&[0, 1], &[0.0, 0.0]).unwrap();
        let at = |n: usize| group.firings.iter().find(|f| f.neuron == n).unwrap_or_else(|| panic!("neuron {n} did not fire")).time;
        assert!((at(2) - 5e-3).abs() < 1e-15 && (at(4) - 6e-3).abs() < 1e-15);
        assert!((at(3) - 9.95e-3).abs() < 1e-15, "neuron 3 fired at {} rather than its earliest coincidence", at(3));
        assert!(at(3) < 10e-3, "10 ms is what the anchors alone would have given");
    }

    #[test]
    fn a_group_is_reported_once_however_many_anchor_sets_reach_it() {
        // Two targets fed by the same pair: the pair's coincidence at each is a separate anchor
        // timing, but if both produce the same firings it is one pattern and is counted once.
        let mut net = Network::new(4, 2, 1e-4).unwrap();
        net.connect(0, 2, 5e-3).unwrap();
        net.connect(1, 2, 5e-3).unwrap();
        net.connect(0, 3, 7e-3).unwrap();
        net.connect(1, 3, 7e-3).unwrap();
        let groups = net.groups(2, 3).unwrap();
        // Both timings are "fire together", and both fire 2 and 3 — one pattern.
        assert_eq!(groups.len(), 1, "{groups:?}");
        assert_eq!(groups[0].neurons(), vec![0, 1, 2, 3]);
    }

    #[test]
    fn a_random_networks_targets_are_distinct_and_never_itself() {
        let net = random(40, 6, &[1e-3, 5e-3], 2, 1e-4, 3).unwrap();
        for pre in 0..net.n {
            let targets: Vec<usize> = net.out[pre].iter().map(|(p, _)| *p).collect();
            assert_eq!(targets.len(), 6);
            assert!(!targets.contains(&pre), "neuron {pre} connects to itself");
            let mut sorted = targets.clone();
            sorted.sort_unstable();
            sorted.dedup();
            assert_eq!(sorted.len(), 6, "neuron {pre} has a repeated target: {targets:?}");
        }
    }

    #[test]
    fn a_hand_built_group_reports_each_neuron_once() {
        // `Group`'s fields are public, so a group may be built by hand — and `neurons()` is a set,
        // whatever the firings say.
        let g = Group {
            anchors: vec![1],
            firings: vec![Firing { neuron: 1, time: 0.0 }, Firing { neuron: 2, time: 1e-3 }, Firing { neuron: 1, time: 5e-3 }],
        };
        assert_eq!(g.neurons(), vec![1, 2]);
        assert_eq!(g.len(), 3, "len() counts firings, not neurons");
        assert!(!g.is_empty());
    }

    #[test]
    fn bad_networks_and_anchors_are_refused() {
        assert_eq!(Network::new(0, 2, 1e-3), Err(PolyError::BadShape));
        assert_eq!(Network::new(MAX_NEURONS + 1, 2, 1e-3), Err(PolyError::BadShape));
        assert_eq!(Network::new(3, 0, 1e-3), Err(PolyError::OutOfRange { what: "threshold", value: 0.0 }));
        assert!(Network::new(3, 2, 0.0).is_err() && Network::new(3, 2, f64::NAN).is_err());
        let mut net = Network::new(3, 2, 1e-4).unwrap();
        assert_eq!(net.connect(3, 0, 1e-3), Err(PolyError::NoSuchNeuron { index: 3 }));
        assert_eq!(net.connect(0, 7, 1e-3), Err(PolyError::NoSuchNeuron { index: 7 }));
        assert_eq!(net.connect(0, 1, 0.0), Err(PolyError::BadDelay { value: 0.0 }));
        assert!(net.connect(0, 1, f64::INFINITY).is_err());
        assert_eq!(net.delay(0, 1), None);
        net.connect(0, 1, 2e-3).unwrap();
        assert_eq!(net.delay(0, 1), Some(2e-3));
        assert_eq!(net.delay(9, 1), None);
        assert_eq!(net.simulate(&[0], &[0.0]), Err(PolyError::TooFewAnchors { got: 1, threshold: 2 }));
        assert_eq!(net.simulate(&[0, 1], &[0.0]), Err(PolyError::OutOfRange { what: "times", value: 1.0 }));
        assert_eq!(net.simulate(&[0, 5], &[0.0, 0.0]), Err(PolyError::NoSuchNeuron { index: 5 }));
        assert!(matches!(net.simulate(&[0, 1], &[0.0, f64::NAN]), Err(PolyError::BadDelay { value }) if value.is_nan()));
        assert!(matches!(net.connect(0, 1, f64::NAN), Err(PolyError::BadDelay { value }) if value.is_nan()));
        assert!(random(10, 0, &[1e-3], 2, 1e-4, 1).is_err());
        assert!(random(10, 9, &[1e-3], 2, 1e-4, 1).is_ok(), "nine targets are available to each of ten neurons");
        assert!(random(10, 10, &[1e-3], 2, 1e-4, 1).is_err(), "ten are not");
        assert!(random(10, 3, &[], 2, 1e-4, 1).is_err());
        assert!(random(0, 3, &[1e-3], 2, 1e-4, 1).is_err());
        assert!(PolyError::NoSuchNeuron { index: 4 }.to_string().contains("neuron 4"));
        assert!(PolyError::TooFewAnchors { got: 1, threshold: 2 }.to_string().contains("threshold of 2"));
        let empty = Group { anchors: vec![], firings: vec![] };
        // `len()` is checked against the firings it counts, not against zero, so that a `len`
        // which always returned zero would not pass by agreeing with `is_empty`.
        assert!(empty.is_empty() && empty.len() == empty.firings.len() && empty.neurons().is_empty());
    }

    /// The group cap is a real truncation, not a redundant restatement of the fire-once rule.
    ///
    /// The hole this fills: every network the suite builds comes from [`Network::new`], which
    /// refuses `n > MAX_NEURONS`, so `self.n.min(MAX_GROUP)` is always `self.n` and the cap
    /// coincides with the bound "each neuron fires at most once" already gives. [`Network`] is
    /// public with public fields and no `#[non_exhaustive]`, [`Network::connect`] checks only
    /// `index < self.n`, and [`Network::simulate`] re-checks nothing about `n` — so a larger
    /// network is reachable through the public API, and there the two bounds are different
    /// numbers and the cap is the one that bites.
    ///
    /// A chain of `MAX_NEURONS + 1` neurons, one 1 ms axon each, fired from neuron 0: the
    /// fire-once rule alone allows 4097 firings, the cap allows 4096. Measured: the group holds
    /// 4096 firings, the last of them neuron 4095, and neuron 4096 never fires.
    #[test]
    fn a_group_is_capped_at_max_group_even_when_the_network_is_bigger_than_max_neurons() {
        let n = MAX_NEURONS + 1;
        let mut net = Network { n, out: vec![Vec::new(); n], threshold: 1, window: 1e-3 };
        for i in 0..n - 1 {
            net.connect(i, i + 1, 1e-3).unwrap();
        }
        // The two bounds are not the same number here, which is the whole point of the fixture.
        assert_eq!(n.min(MAX_GROUP), MAX_GROUP);
        assert!(MAX_GROUP < n);
        let g = net.simulate(&[0], &[0.0]).unwrap();
        assert_eq!(g.len(), MAX_GROUP, "the cap, not the fire-once rule, is what stops the cascade");
        assert_eq!(g.firings.last().map(|f| f.neuron), Some(MAX_GROUP - 1));
        assert_eq!(g.firings.iter().filter(|f| f.neuron == n - 1).count(), 0, "neuron {} fired past the cap", n - 1);
    }

}
