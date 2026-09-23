//! Graph algorithms done by spikes: a shortest path is the time a wavefront takes to arrive, and
//! a boundary-value problem is the fraction of random walkers that leave by each door — both
//! checked against the exact answer, with the spikes counted.
//!
//! # What the mechanism is
//!
//! **The wavefront.** Give every vertex one neuron and every edge one synapse whose *delay* is the
//! edge's length. Make the neurons fire on their first input and never again. Inject one spike at
//! the source: a vertex's first spike arrives at exactly its shortest-path distance, because the
//! first arrival over all paths is the minimum over all paths. The neuron that latches WHO woke it
//! holds its parent in the shortest-path tree (Aimone, Ho, Parekh, Phillips, Pinar, Severa and
//! Wang, *Provable advantages for graph algorithms in spiking neural networks*, SPAA 2021,
//! pp. 35–47; Hamilton, Mintz and Schuman, *Spike-based primitives for graph algorithms*,
//! arXiv:1903.10574, 2019).
//!
//! **The walkers.** A spike that hops to a random neighbour is a random walker, and a population
//! of them is a Monte Carlo solver for the diffusion equation: the solution of Laplace's equation
//! at a point is the expected boundary value where a walker released there first exits (Smith,
//! Hill, Reeder, Franke, Lehoucq, Parekh, Severa and Aimone, *Neuromorphic scaling advantages for
//! energy-efficient random walk computations*, Nature Electronics 5:102–112, 2022).
//!
//! # Why it is in a neuromorphic crate
//!
//! These are the workloads where the argument for the hardware is a theorem and not a benchmark.
//! The wavefront finishes in a TIME equal to the longest shortest path and spends exactly one
//! spike per reachable vertex and one synaptic event per edge leaving a reachable vertex — it
//! never touches the rest of the graph, and it never compares or sorts anything. Those three
//! numbers are returned by [`Graph::wavefront`] and asserted below against the graph itself.
//!
//! # The closed forms this module is checked against
//!
//! - **First-spike time is the shortest-path distance**, exactly, against a Bellman–Ford referee
//!   ([`Graph::bellman_ford`]) that shares no code with the wavefront, and against distances worked
//!   by hand on a small graph; an unreachable vertex never fires.
//! - **The cost is the graph's**: spikes = reachable vertices; synaptic events = edges leaving
//!   them; ticks = the largest finite distance.
//! - **The latched parents form a shortest-path tree**: walking them back from any vertex gives a
//!   path whose length is that vertex's distance.
//! - **Gambler's ruin.** On a path of `n + 1` vertices with both ends absorbing, a walker started
//!   at `k` exits at the far end with probability `k/n` and takes `k (n − k)` steps on average;
//!   the sampled values are held to those within four standard errors, the standard error being
//!   arithmetic (`√(p(1−p)/walkers)` for the probability; for the time, the exact variance
//!   `k(n−k)[(k² + (n−k)² − 2)]/3` of the exit time).
//! - **One spike per step**: the walkers' spike count is the total number of steps taken.
//!
//! # What this module has NOT reproduced
//!
//! - The constant-factor and scaling claims of either paper on hardware; the counts here are
//!   events, not joules. Price them with [`crate::ledger`].
//! - Delays that are not whole ticks. A chip's delay lines are integers and so are these.
//! - Walks with drift, or in more than the graph's own geometry; the closed form checked is the
//!   one-dimensional one because that is the one that exists.

use core::cmp::Reverse;
use core::fmt;
use std::collections::BinaryHeap;

use crate::rng::Rng;

/// What went wrong, named rather than guessed around.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraphError {
    /// A count of zero where at least one is needed.
    Empty {
        /// What was empty.
        what: &'static str,
    },
    /// A vertex index past the graph.
    Index {
        /// Which index.
        what: &'static str,
        /// The value.
        index: usize,
        /// The count it had to be below.
        count: usize,
    },
    /// A synapse with no delay: a spike cannot arrive when it leaves.
    ZeroDelay {
        /// Source vertex.
        from: usize,
        /// Target vertex.
        to: usize,
    },
    /// A path length past `u64`.
    Overflow,
    /// More vertices than a `u32` can name.
    TooLarge {
        /// Vertices asked for.
        vertices: usize,
    },
    /// A walker that had nowhere to go: a non-absorbing vertex with no outgoing edge.
    DeadEnd {
        /// The vertex.
        vertex: usize,
    },
    /// A walker still walking at the step limit.
    StepLimit {
        /// The limit.
        limit: u64,
    },
}

impl fmt::Display for GraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty { what } => write!(f, "{what} is empty"),
            Self::Index { what, index, count } => write!(f, "{what} {index} is past the {count} available"),
            Self::ZeroDelay { from, to } => write!(f, "the synapse {from} → {to} has zero delay"),
            Self::Overflow => write!(f, "a path length overflowed u64"),
            Self::TooLarge { vertices } => write!(f, "{vertices} vertices is more than a u32 can name"),
            Self::DeadEnd { vertex } => write!(f, "vertex {vertex} is not absorbing and has no way out"),
            Self::StepLimit { limit } => write!(f, "a walker was still walking after {limit} steps"),
        }
    }
}

impl std::error::Error for GraphError {}

/// A directed graph whose edges carry whole-tick delays.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Graph {
    /// `out[u]` lists `(v, delay)` for every synapse `u → v`, in insertion order.
    pub out: Vec<Vec<(u32, u64)>>,
}

/// What a wavefront found, and what it cost.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Wavefront {
    /// First-spike tick of every vertex — its shortest-path distance — `None` if it never fired.
    pub first_spike: Vec<Option<u64>>,
    /// The presynaptic vertex whose spike arrived first; `None` for the source and for silent
    /// vertices. Ties go to the LOWER presynaptic index.
    pub parent: Vec<Option<u32>>,
    /// Spikes emitted: one per vertex that fired.
    pub spikes: u64,
    /// Synaptic events delivered: one per edge leaving a vertex that fired.
    pub syn_events: u64,
    /// The tick of the last first-spike: how long the chip ran.
    pub ticks: u64,
}

impl Graph {
    /// An edgeless graph on `n` vertices.
    ///
    /// # Errors
    ///
    /// [`GraphError::Empty`] for `n = 0`, [`GraphError::TooLarge`] past `u32::MAX` vertices.
    pub fn new(n: usize) -> Result<Self, GraphError> {
        if n == 0 {
            return Err(GraphError::Empty { what: "vertices" });
        }
        if u32::try_from(n).is_err() {
            return Err(GraphError::TooLarge { vertices: n });
        }
        Ok(Self { out: vec![Vec::new(); n] })
    }

    /// Vertices.
    #[must_use]
    pub fn vertices(&self) -> usize {
        self.out.len()
    }

    /// Add the synapse `from → to` with `delay` ticks.
    ///
    /// # Errors
    ///
    /// [`GraphError::Index`] for an endpoint past the graph, [`GraphError::ZeroDelay`] for a delay
    /// of zero.
    pub fn connect(&mut self, from: usize, to: usize, delay: u64) -> Result<(), GraphError> {
        let n = self.out.len();
        if from >= n {
            return Err(GraphError::Index { what: "from", index: from, count: n });
        }
        if to >= n {
            return Err(GraphError::Index { what: "to", index: to, count: n });
        }
        if delay == 0 {
            return Err(GraphError::ZeroDelay { from, to });
        }
        // `to < n ≤ u32::MAX` by construction.
        self.out[from].push((to as u32, delay));
        Ok(())
    }

    /// Add a synapse in each direction with the same delay.
    ///
    /// # Errors
    ///
    /// As [`Graph::connect`].
    pub fn connect_both(&mut self, a: usize, b: usize, delay: u64) -> Result<(), GraphError> {
        self.connect(a, b, delay)?;
        self.connect(b, a, delay)
    }

    /// A path `0 — 1 — … — n` of `n + 1` vertices with unit delays both ways.
    ///
    /// # Errors
    ///
    /// [`GraphError::Empty`] for `n = 0`; as [`Graph::new`].
    pub fn path(n: usize) -> Result<Self, GraphError> {
        if n == 0 {
            return Err(GraphError::Empty { what: "path edges" });
        }
        let mut g = Self::new(n + 1)?;
        for i in 0..n {
            g.connect_both(i, i + 1, 1)?;
        }
        Ok(g)
    }

    /// Inject one spike at `source` and let it spread: every neuron fires on its first input and
    /// is silent afterwards.
    ///
    /// # Errors
    ///
    /// [`GraphError::Index`] for a source past the graph, [`GraphError::Overflow`] if an arrival
    /// time passes `u64`.
    pub fn wavefront(&self, source: usize) -> Result<Wavefront, GraphError> {
        let n = self.out.len();
        if source >= n {
            return Err(GraphError::Index { what: "source", index: source, count: n });
        }
        let mut first_spike = vec![None; n];
        let mut parent = vec![None; n];
        let (mut spikes, mut syn_events, mut ticks) = (0u64, 0u64, 0u64);
        // Pending arrivals, earliest first; among equal ticks the lower presynaptic index first,
        // which is the tie rule the doc states. `u32::MAX` marks the injected spike.
        let mut pending = BinaryHeap::new();
        pending.push(Reverse((0u64, u32::MAX, source as u32)));
        while let Some(Reverse((t, pre, v))) = pending.pop() {
            let v = v as usize;
            if first_spike[v].is_some() {
                continue;
            }
            first_spike[v] = Some(t);
            parent[v] = if pre == u32::MAX { None } else { Some(pre) };
            spikes += 1;
            ticks = t;
            for &(w, delay) in &self.out[v] {
                syn_events += 1;
                let arrive = t.checked_add(delay).ok_or(GraphError::Overflow)?;
                if first_spike[w as usize].is_none() {
                    pending.push(Reverse((arrive, v as u32, w)));
                }
            }
        }
        Ok(Wavefront { first_spike, parent, spikes, syn_events, ticks })
    }

    /// Shortest-path distances by Bellman–Ford: relax every edge until nothing changes. The
    /// referee for [`Graph::wavefront`] — slower, and sharing nothing with it.
    ///
    /// # Errors
    ///
    /// [`GraphError::Index`] for a source past the graph, [`GraphError::Overflow`] if a path length
    /// passes `u64`.
    pub fn bellman_ford(&self, source: usize) -> Result<Vec<Option<u64>>, GraphError> {
        let n = self.out.len();
        if source >= n {
            return Err(GraphError::Index { what: "source", index: source, count: n });
        }
        let mut dist: Vec<Option<u64>> = vec![None; n];
        dist[source] = Some(0);
        for _ in 0..n {
            let mut changed = false;
            for u in 0..n {
                let Some(du) = dist[u] else { continue };
                for &(v, delay) in &self.out[u] {
                    let through = du.checked_add(delay).ok_or(GraphError::Overflow)?;
                    if dist[v as usize].is_none_or(|dv| through < dv) {
                        dist[v as usize] = Some(through);
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }
        Ok(dist)
    }
}

impl Wavefront {
    /// The path from the source to `v`, source first, read off the latched parents. `None` if `v`
    /// never fired or is past the graph — and `None` if the parents do not lead back to a source
    /// within as many hops as there are vertices: the fields are public, and a cycle of parents
    /// would otherwise be walked for ever. (This module's own mutation sweep did exactly that, and
    /// the test process was killed for the memory it took.)
    #[must_use]
    pub fn path_to(&self, v: usize) -> Option<Vec<usize>> {
        self.first_spike.get(v).copied().flatten()?;
        let mut path = vec![v];
        let mut at = v;
        while let Some(p) = self.parent[at] {
            at = p as usize;
            path.push(at);
            if at >= self.parent.len() || path.len() > self.parent.len() {
                return None;
            }
        }
        path.reverse();
        Some(path)
    }
}

// ---------------------------------------------------------------------------------------------
// Random walkers
// ---------------------------------------------------------------------------------------------

/// What a population of walkers did.
#[derive(Debug, Clone, PartialEq)]
pub struct ExitStats {
    /// Walkers absorbed at each vertex (zero for a vertex that is not absorbing).
    pub exits: Vec<u64>,
    /// Walkers released.
    pub walkers: u64,
    /// Steps taken by all walkers together — one spike each.
    pub spikes: u64,
}

impl ExitStats {
    /// The fraction of walkers absorbed at `v`; `None` past the graph or with no walkers.
    #[must_use]
    pub fn exit_probability(&self, v: usize) -> Option<f64> {
        if self.walkers == 0 {
            return None;
        }
        self.exits.get(v).map(|&e| e as f64 / self.walkers as f64)
    }

    /// Mean steps per walker; `None` with no walkers.
    #[must_use]
    pub fn mean_steps(&self) -> Option<f64> {
        if self.walkers == 0 { None } else { Some(self.spikes as f64 / self.walkers as f64) }
    }

    /// The Monte Carlo solution of Laplace's equation at the release point for boundary values
    /// `boundary[v]`: the mean boundary value over where the walkers exited. `None` for a length
    /// mismatch or no walkers.
    #[must_use]
    pub fn harmonic_value(&self, boundary: &[f64]) -> Option<f64> {
        if boundary.len() != self.exits.len() || self.walkers == 0 {
            return None;
        }
        Some(self.exits.iter().zip(boundary).map(|(&e, b)| e as f64 * b).sum::<f64>() / self.walkers as f64)
    }
}

/// Release `walkers` walkers at `start`; each hops to a uniformly chosen out-neighbour until it
/// lands on an absorbing vertex. Delays are ignored — a walker's step is one tick.
///
/// # Errors
///
/// [`GraphError::Index`] for a start past the graph or an `absorbing` of the wrong length,
/// [`GraphError::Empty`] for zero walkers, [`GraphError::DeadEnd`] if a walker reaches a
/// non-absorbing vertex with no way out, [`GraphError::StepLimit`] if one is still walking after
/// `max_steps`.
pub fn release_walkers(
    graph: &Graph,
    start: usize,
    absorbing: &[bool],
    walkers: u64,
    max_steps: u64,
    rng: &mut Rng,
) -> Result<ExitStats, GraphError> {
    let n = graph.vertices();
    if start >= n {
        return Err(GraphError::Index { what: "start", index: start, count: n });
    }
    if absorbing.len() != n {
        return Err(GraphError::Index { what: "absorbing (length)", index: absorbing.len(), count: n });
    }
    if walkers == 0 {
        return Err(GraphError::Empty { what: "walkers" });
    }
    let mut exits = vec![0u64; n];
    let mut spikes = 0u64;
    for _ in 0..walkers {
        let mut at = start;
        let mut steps = 0u64;
        while !absorbing[at] {
            if steps == max_steps {
                return Err(GraphError::StepLimit { limit: max_steps });
            }
            let ways = &graph.out[at];
            if ways.is_empty() {
                return Err(GraphError::DeadEnd { vertex: at });
            }
            // `ways.len() ≤ u32::MAX` is not guaranteed by construction for a multigraph, so the
            // draw is refused rather than truncated.
            let count = u32::try_from(ways.len()).map_err(|_| GraphError::TooLarge { vertices: ways.len() })?;
            at = ways[rng.below(count) as usize].0 as usize;
            steps += 1;
        }
        exits[at] += 1;
        spikes += steps;
    }
    Ok(ExitStats { exits, walkers, spikes })
}

/// Gambler's ruin on a path of `n` edges: the probability that a walker started at `k` reaches
/// vertex `n` before vertex `0`, which is `k/n`. `None` for `n = 0` or `k > n`.
#[must_use]
pub fn ruin_probability(k: usize, n: usize) -> Option<f64> {
    if n == 0 || k > n { None } else { Some(k as f64 / n as f64) }
}

/// The mean exit time `k (n − k)` of that walk, steps. `None` for `n = 0` or `k > n`.
#[must_use]
pub fn ruin_mean_steps(k: usize, n: usize) -> Option<f64> {
    if n == 0 || k > n { None } else { Some((k * (n - k)) as f64) }
}

/// The variance `k (n − k) (k² + (n − k)² − 2) / 3` of that exit time, steps². `None` for `n = 0`
/// or `k > n`.
#[must_use]
pub fn ruin_steps_variance(k: usize, n: usize) -> Option<f64> {
    if n == 0 || k > n {
        return None;
    }
    let (a, b) = (k as f64, (n - k) as f64);
    Some(a * b * (a * a + b * b - 2.0) / 3.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A seven-vertex graph whose distances from 0 are worked by hand in the test below.
    fn worked() -> Graph {
        let mut g = Graph::new(7).unwrap();
        for &(u, v, d) in &[(0, 1, 7), (0, 2, 2), (2, 1, 3), (1, 3, 1), (2, 3, 8), (3, 4, 2), (4, 0, 1), (2, 5, 10), (4, 5, 1)] {
            g.connect(u, v, d).unwrap();
        }
        g
    }

    #[test]
    fn the_first_spike_arrives_at_the_shortest_path_distance() {
        let g = worked();
        let wave = g.wavefront(0).unwrap();
        // 0→2 (2), 0→2→1 (5, beating the direct 7), →3 (6), →4 (8), →5 (9, beating 2→5 at 12).
        // Vertex 6 has no way in.
        assert_eq!(wave.first_spike, vec![Some(0), Some(5), Some(2), Some(6), Some(8), Some(9), None]);
        assert_eq!(wave.parent, vec![None, Some(2), Some(0), Some(1), Some(3), Some(4), None]);
        assert_eq!(wave.path_to(5), Some(vec![0, 2, 1, 3, 4, 5]));
        assert_eq!(wave.path_to(0), Some(vec![0]));
        assert_eq!(wave.path_to(6), None);
        assert_eq!(wave.path_to(7), None);
        // A record whose parents chase each other is refused, not walked for ever.
        let mut looped = wave.clone();
        looped.parent[2] = Some(1);
        assert_eq!(looped.path_to(5), None);
        looped.parent[2] = Some(9);
        assert_eq!(looped.path_to(5), None, "a parent past the graph ends the walk without a source");
        // The cost is the graph's: six vertices fired; every edge leaves one of them; the chip ran
        // for as long as the farthest vertex is far.
        assert_eq!((wave.spikes, wave.syn_events, wave.ticks), (6, 9, 9));
        assert_eq!(g.bellman_ford(0).unwrap(), wave.first_spike);
        // From vertex 5, which has no way out, nothing else fires and nothing is delivered.
        let stuck = g.wavefront(5).unwrap();
        assert_eq!((stuck.spikes, stuck.syn_events, stuck.ticks), (1, 0, 0));
        assert_eq!(stuck.first_spike.iter().filter(|t| t.is_some()).count(), 1);
    }

    #[test]
    fn the_wavefront_agrees_with_bellman_ford_on_random_graphs() {
        let mut rng = Rng::new(41);
        for trial in 0..20 {
            let n = 30;
            let mut g = Graph::new(n).unwrap();
            for u in 0..n {
                for v in 0..n {
                    if u != v && rng.next_f64() < 0.08 {
                        g.connect(u, v, 1 + u64::from(rng.below(20))).unwrap();
                    }
                }
            }
            let source = rng.below(n as u32) as usize;
            let wave = g.wavefront(source).unwrap();
            let want = g.bellman_ford(source).unwrap();
            assert_eq!(wave.first_spike, want, "trial {trial}");
            let reached: Vec<usize> = (0..n).filter(|&v| want[v].is_some()).collect();
            assert_eq!(wave.spikes, reached.len() as u64);
            assert_eq!(wave.syn_events, reached.iter().map(|&v| g.out[v].len() as u64).sum::<u64>());
            assert_eq!(wave.ticks, want.iter().flatten().copied().max().unwrap());
            // Every latched path is a real path of exactly the claimed length.
            for &v in &reached {
                let path = wave.path_to(v).unwrap();
                assert_eq!(path[0], source);
                let mut length = 0;
                for hop in path.windows(2) {
                    let best = g.out[hop[0]].iter().filter(|(w, _)| *w as usize == hop[1]).map(|(_, d)| *d).min();
                    length += best.expect("the latched parent is not a neighbour");
                }
                assert_eq!(Some(length), want[v]);
            }
            if trial == 0 {
                assert!(reached.len() > 5 && reached.len() <= n, "the graph is too sparse to test anything: {}", reached.len());
            }
        }
    }

    #[test]
    fn a_tie_latches_the_lower_presynaptic_index() {
        // Two routes of equal length into vertex 3: through 1 and through 2.
        let mut g = Graph::new(4).unwrap();
        for &(u, v, d) in &[(0, 2, 1), (0, 1, 1), (2, 3, 4), (1, 3, 4)] {
            g.connect(u, v, d).unwrap();
        }
        let wave = g.wavefront(0).unwrap();
        assert_eq!(wave.first_spike[3], Some(5));
        assert_eq!(wave.parent[3], Some(1));
        // On a path with unit delays the first-spike tick is the hop count.
        let line = Graph::path(9).unwrap();
        let hops = line.wavefront(3).unwrap();
        let want: Vec<Option<u64>> = (0..10u64).map(|v| Some(v.abs_diff(3))).collect();
        assert_eq!(hops.first_spike, want);
        assert_eq!((hops.spikes, hops.syn_events, hops.ticks), (10, 18, 6));
    }

    #[test]
    fn walkers_leave_by_each_door_as_often_as_gamblers_ruin_says() {
        let (n, k) = (10usize, 3usize);
        let g = Graph::path(n).unwrap();
        let mut absorbing = vec![false; n + 1];
        absorbing[0] = true;
        absorbing[n] = true;
        let walkers = 40_000u64;
        let stats = release_walkers(&g, k, &absorbing, walkers, 100_000, &mut Rng::new(12)).unwrap();
        assert_eq!(stats.exits.iter().sum::<u64>(), walkers);
        assert!(stats.exits[1..n].iter().all(|&e| e == 0));
        let p = ruin_probability(k, n).unwrap();
        assert_eq!(p, 0.3);
        let se_p = (p * (1.0 - p) / walkers as f64).sqrt();
        let got = stats.exit_probability(n).unwrap();
        assert!((got - p).abs() < 4.0 * se_p, "{got} exited at the far end, ruin says {p} ± {se_p}");
        // The mean exit time is k(n − k) = 21 steps, and its standard error comes from the exact
        // variance 21·(9 + 49 − 2)/3 = 392.
        assert_eq!(ruin_mean_steps(k, n), Some(21.0));
        assert_eq!(ruin_steps_variance(k, n), Some(392.0));
        let se_t = (392.0 / walkers as f64).sqrt();
        let mean = stats.mean_steps().unwrap();
        assert!((mean - 21.0).abs() < 4.0 * se_t, "mean exit time {mean}, ruin says 21 ± {se_t}");
        // Laplace on a line is linear: with u(0) = 2 and u(n) = 12 the solution at k is 2 + 10·k/n.
        let mut boundary = vec![0.0; n + 1];
        boundary[0] = 2.0;
        boundary[n] = 12.0;
        let u = stats.harmonic_value(&boundary).unwrap();
        assert!((u - 5.0).abs() < 4.0 * 10.0 * se_p, "harmonic value {u}, the line says 5");
        // A walker released ON a door takes no steps and spends no spikes.
        let at_door = release_walkers(&g, 0, &absorbing, 100, 10, &mut Rng::new(1)).unwrap();
        assert_eq!((at_door.exits[0], at_door.spikes), (100, 0));
    }

    #[test]
    fn the_exit_time_variance_is_the_one_the_standard_error_used() {
        // The variance formula referees the test above, so it is itself checked: against the
        // sample variance of exit times, walker by walker, on a smaller walk where it is 8·…
        let (n, k) = (6usize, 2usize);
        assert_eq!(ruin_steps_variance(k, n), Some(2.0 * 4.0 * (4.0 + 16.0 - 2.0) / 3.0));
        let g = Graph::path(n).unwrap();
        let mut absorbing = vec![false; n + 1];
        absorbing[0] = true;
        absorbing[n] = true;
        let mut rng = Rng::new(77);
        let trials = 40_000;
        let (mut sum, mut sum_sq) = (0.0, 0.0);
        for _ in 0..trials {
            let one = release_walkers(&g, k, &absorbing, 1, 100_000, &mut rng).unwrap();
            let t = one.spikes as f64;
            sum += t;
            sum_sq += t * t;
        }
        let mean = sum / f64::from(trials);
        let var = sum_sq / f64::from(trials) - mean * mean;
        // A sample variance of a distribution with excess kurtosis below ~10 is good to about
        // var·√(12/trials) = 1.7%; 5% is three of those.
        assert!((var / 48.0 - 1.0).abs() < 0.05, "sample variance {var}, formula 48");
        assert!((mean - 8.0).abs() < 4.0 * (48.0f64 / f64::from(trials)).sqrt());
        // The end cases of the closed forms.
        assert_eq!(ruin_probability(0, 5), Some(0.0));
        assert_eq!(ruin_probability(5, 5), Some(1.0));
        assert_eq!(ruin_mean_steps(5, 5), Some(0.0));
        assert_eq!(ruin_steps_variance(1, 2), Some(0.0));
        for bad in [ruin_probability(6, 5), ruin_probability(0, 0), ruin_mean_steps(6, 5), ruin_steps_variance(6, 5)] {
            assert_eq!(bad, None);
        }
    }

    #[test]
    fn bad_arguments_are_refused() {
        assert_eq!(Graph::new(0), Err(GraphError::Empty { what: "vertices" }));
        assert_eq!(Graph::path(0), Err(GraphError::Empty { what: "path edges" }));
        let mut g = Graph::new(3).unwrap();
        assert_eq!(g.connect(3, 0, 1), Err(GraphError::Index { what: "from", index: 3, count: 3 }));
        assert_eq!(g.connect(0, 3, 1), Err(GraphError::Index { what: "to", index: 3, count: 3 }));
        assert_eq!(g.connect(0, 1, 0), Err(GraphError::ZeroDelay { from: 0, to: 1 }));
        assert!(g.out.iter().all(Vec::is_empty), "a refused synapse was stored anyway");
        assert_eq!(g.wavefront(3).unwrap_err(), GraphError::Index { what: "source", index: 3, count: 3 });
        assert_eq!(g.bellman_ford(3).unwrap_err(), GraphError::Index { what: "source", index: 3, count: 3 });
        // A path length past u64 is refused by both, not wrapped.
        g.connect(0, 1, u64::MAX).unwrap();
        g.connect(1, 2, 1).unwrap();
        assert_eq!(g.wavefront(0).unwrap_err(), GraphError::Overflow);
        assert_eq!(g.bellman_ford(0).unwrap_err(), GraphError::Overflow);
        let line = Graph::path(4).unwrap();
        let doors = [true, false, false, false, true];
        let mut rng = Rng::new(2);
        assert_eq!(release_walkers(&line, 5, &doors, 1, 10, &mut rng).unwrap_err(), GraphError::Index { what: "start", index: 5, count: 5 });
        assert!(matches!(release_walkers(&line, 2, &doors[..4], 1, 10, &mut rng), Err(GraphError::Index { what: "absorbing (length)", .. })));
        assert_eq!(release_walkers(&line, 2, &doors, 0, 10, &mut rng).unwrap_err(), GraphError::Empty { what: "walkers" });
        // No doors: the walker is still walking at the limit, and says so.
        assert_eq!(release_walkers(&line, 2, &[false; 5], 1, 50, &mut rng).unwrap_err(), GraphError::StepLimit { limit: 50 });
        let mut trap = Graph::new(2).unwrap();
        trap.connect(0, 1, 1).unwrap();
        assert_eq!(release_walkers(&trap, 0, &[false, false], 1, 50, &mut rng).unwrap_err(), GraphError::DeadEnd { vertex: 1 });
        let none = ExitStats { exits: vec![0; 2], walkers: 0, spikes: 0 };
        assert_eq!((none.exit_probability(0), none.mean_steps(), none.harmonic_value(&[0.0, 1.0])), (None, None, None));
        let some = ExitStats { exits: vec![1, 3], walkers: 4, spikes: 10 };
        assert_eq!(some.exit_probability(1), Some(0.75));
        assert_eq!(some.exit_probability(2), None);
        assert_eq!(some.harmonic_value(&[1.0]), None);
        assert_eq!(some.mean_steps(), Some(2.5));
    }

    /// The RENDERED text of all seven refusals: which way round a synapse runs, and which of the
    /// two numbers in an index refusal is the offending one. Pinned because every existing refusal
    /// test compares the STRUCT — `Err(GraphError::Index { what, index, count })` — so the `Display`
    /// body was never run at all, and a message that reads "from 3 is past the 5 available" for an
    /// index of 5 in a graph of 3 is fluent in both directions.
    #[test]
    fn a_refusal_renders_its_endpoints_and_its_two_numbers_the_right_way_round() {
        let mut g = Graph::new(3).unwrap();
        assert_eq!(g.connect(0, 1, 0).unwrap_err().to_string(), "the synapse 0 → 1 has zero delay");
        assert_eq!(g.connect(2, 0, 0).unwrap_err().to_string(), "the synapse 2 → 0 has zero delay");
        // Five is the index and three is the count, and an index refusal that named them the other
        // way round would read the same for the (3, 3) case the existing test uses.
        assert_eq!(g.connect(5, 0, 1).unwrap_err().to_string(), "from 5 is past the 3 available");
        assert_eq!(g.connect(0, 5, 1).unwrap_err().to_string(), "to 5 is past the 3 available");
        assert_eq!(g.wavefront(9).unwrap_err().to_string(), "source 9 is past the 3 available");
        assert_eq!(Graph::new(0).unwrap_err().to_string(), "vertices is empty");
        assert_eq!(Graph::new(1usize << 33).unwrap_err().to_string(), "8589934592 vertices is more than a u32 can name");
        g.connect(0, 1, u64::MAX).unwrap();
        g.connect(1, 2, 1).unwrap();
        assert_eq!(g.wavefront(0).unwrap_err().to_string(), "a path length overflowed u64");
        let mut trap = Graph::new(2).unwrap();
        trap.connect(0, 1, 1).unwrap();
        let mut rng = Rng::new(2);
        assert_eq!(
            release_walkers(&trap, 0, &[false, false], 1, 50, &mut rng).unwrap_err().to_string(),
            "vertex 1 is not absorbing and has no way out"
        );
        let line = Graph::path(4).unwrap();
        assert_eq!(
            release_walkers(&line, 2, &[false; 5], 1, 50, &mut rng).unwrap_err().to_string(),
            "a walker was still walking after 50 steps"
        );
    }

    /// A two-way synapse whose FORWARD half is refused is refused as a whole, and the refusal is
    /// the forward half's. Pinned because the two halves fail under exactly the same conditions —
    /// a bad endpoint or a zero delay is bad in both directions — so swallowing the first refusal
    /// and returning the second one leaves a graph that is still empty and an error that is still
    /// an error, differing only in which endpoint it names.
    #[test]
    fn a_two_way_synapse_is_refused_by_its_forward_half() {
        let mut g = Graph::new(3).unwrap();
        assert_eq!(g.connect_both(0, 1, 0), Err(GraphError::ZeroDelay { from: 0, to: 1 }));
        assert_eq!(g.connect_both(5, 0, 1), Err(GraphError::Index { what: "from", index: 5, count: 3 }));
        assert!(g.out.iter().all(Vec::is_empty), "a refused two-way synapse stored an edge anyway");
        // And when both halves are legal, both are stored.
        g.connect_both(0, 1, 4).unwrap();
        assert_eq!(g.out[0], [(1, 4)]);
        assert_eq!(g.out[1], [(0, 4)]);
    }

    /// A shortest path through EVERY vertex is a path, not a cycle. Pinned because the hop budget
    /// that stops a doctored parent array from being walked for ever is exactly the vertex count,
    /// and the longest path this module's other fixtures ask for is six vertices of seven — a
    /// budget one hop short refuses nothing they build.
    #[test]
    fn a_path_through_every_vertex_is_returned_and_not_refused_as_a_cycle() {
        let line = Graph::path(9).unwrap();
        let wave = line.wavefront(0).unwrap();
        assert_eq!(wave.parent.len(), 10);
        assert_eq!(wave.path_to(9), Some((0..=9).collect::<Vec<usize>>()), "ten vertices is a legal path of ten");
        // One vertex fewer is still returned, so the budget is not simply generous.
        assert_eq!(wave.path_to(8), Some((0..=8).collect::<Vec<usize>>()));
    }

    /// The step limit is the number of steps a walker MAY take: a walk of exactly that many steps
    /// is answered, and a walk of one more is refused. Pinned because the existing limit fixture
    /// releases a walker into a graph with no absorbing vertex at all, which is refused whether
    /// the limit is counted before or after the step — the walk never ends either way.
    #[test]
    fn a_walker_is_refused_on_the_step_that_would_pass_the_limit_and_not_after_it() {
        let mut chain = Graph::new(4).unwrap();
        for v in 0..3 {
            chain.connect(v, v + 1, 1).unwrap();
        }
        // Every vertex before the door has exactly one way out, so the walk is forced and takes
        // three steps whatever the draws are.
        let doors = [false, false, false, true];
        let mut rng = Rng::new(5);
        let done = release_walkers(&chain, 0, &doors, 1, 3, &mut rng).unwrap();
        assert_eq!((done.exits[3], done.spikes), (1, 3));
        assert_eq!(
            release_walkers(&chain, 0, &doors, 1, 2, &mut rng).unwrap_err(),
            GraphError::StepLimit { limit: 2 },
            "a limit of two must refuse the third step, not take it"
        );
    }
}
