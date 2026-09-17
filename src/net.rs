//! Sparse directed connectivity in CSR, with a per-synapse delay.
//!
//! # Directed, and each synapse stored once
//!
//! This is the point where a reader coming from the sibling thermodynamic crate has to change
//! gear. `ferrotherm`'s `Graph` is an *undirected* Ising coupling structure and stores every edge
//! **twice**, once from each end, so `nbr.len()` is `2 * n_edges`. A synapse is not like that: it
//! runs from a presynaptic neuron to a postsynaptic one and not back, so it is stored **once**,
//! from the presynaptic side, and [`Net::n_syn`] is `post.len()` rather than half of it.
//!
//! The difference is stated this loudly because the two crates share a house style and a reader who
//! transfers the `/ 2` would halve every synapse count — and therefore halve every energy figure,
//! in the direction that flatters the result.
//!
//! # Delay is a first-class field, not an afterthought
//!
//! Axonal delay is where a spiking network keeps information that a rate code cannot hold: a
//! coincidence detector is a neuron whose inputs arrive together *because their delays line up*.
//! Storing delay per synapse rather than per neuron is what makes that expressible, and it is also
//! what a neuromorphic chip's routing table actually holds.
//!
//! Delay is in **ticks**, not seconds, for the reason given in [`crate::spike`]: it is added to a
//! tick index, and a delay in seconds would be converted at every delivery.

/// A sparse directed network of synapses.
#[derive(Debug, Clone, PartialEq)]
pub struct Net {
    /// Neuron count.
    pub n: usize,
    /// CSR row starts, length `n + 1`: the outgoing synapses of neuron `i` are the entries
    /// `offset[i]..offset[i+1]` of [`Net::post`], [`Net::w`] and [`Net::delay`].
    pub offset: Vec<usize>,
    /// Postsynaptic neuron of each synapse, grouped by presynaptic neuron.
    pub post: Vec<u32>,
    /// Weight of each synapse, parallel to [`Net::post`]. Volts of instantaneous membrane
    /// displacement per arriving spike; negative for inhibition.
    pub w: Vec<f64>,
    /// Delay of each synapse in ticks, parallel to [`Net::post`]. Zero means same-tick delivery.
    pub delay: Vec<u32>,
    /// Synapse count, which is `post.len()` and **not** `post.len() / 2`.
    pub n_syn: usize,
    /// The largest delay in the network, cached because the simulator sizes its delivery ring from
    /// it and recomputing it per step would be a scan of every synapse per tick.
    pub max_delay: u32,
}

impl Net {
    /// The outgoing synapses of `pre`, as `(post, weight, delay)`.
    ///
    /// Empty for an out-of-range index rather than a panic: a caller sweeping neuron indices past
    /// the end is asking a question with the answer "no synapses", and panicking there turns a
    /// loop bound off by one into a crash rather than a no-op.
    pub fn out_of(&self, pre: usize) -> impl Iterator<Item = (u32, f64, u32)> + '_ {
        let (a, b) = if pre + 1 < self.offset.len() {
            (self.offset[pre], self.offset[pre + 1])
        } else {
            (0, 0)
        };
        (a..b).map(move |k| (self.post[k], self.w[k], self.delay[k]))
    }

    /// Out-degree of `pre`.
    #[must_use]
    pub fn out_degree(&self, pre: usize) -> usize {
        if pre + 1 < self.offset.len() { self.offset[pre + 1] - self.offset[pre] } else { 0 }
    }

    /// In-degree of every neuron, computed by a single pass over the synapse list.
    ///
    /// Fan-in is the quantity a neuromorphic chip has a hard limit on — a core holds a fixed number
    /// of synapses per neuron — so a network that will not fit is a network whose maximum in-degree
    /// exceeds the part's, and this is how that gets checked before anything is mapped.
    #[must_use]
    pub fn in_degrees(&self) -> Vec<usize> {
        let mut d = vec![0usize; self.n];
        for &p in &self.post {
            d[p as usize] += 1;
        }
        d
    }

    /// Mean number of synapses per neuron — the density figure that decides whether the weights fit
    /// in a core's local memory.
    #[must_use]
    pub fn fan_out_mean(&self) -> f64 {
        if self.n == 0 { 0.0 } else { self.n_syn as f64 / self.n as f64 }
    }
}

/// Builds a [`Net`], checking indices as they arrive rather than after the fact.
#[derive(Debug, Clone, Default)]
pub struct NetBuilder {
    n: usize,
    edges: Vec<(u32, u32, f64, u32)>,
}

/// Why a network could not be built.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NetError {
    /// A synapse named a neuron index at or past the neuron count.
    OutOfRange {
        /// The offending index.
        index: u32,
        /// The neuron count it exceeded.
        n: usize,
    },
    /// A weight was not a finite number.
    ///
    /// Rejected at the boundary because a NaN weight does not fail loudly: it silently poisons one
    /// membrane potential, then every spike time that neuron produces, then every downstream
    /// neuron, and the run completes and reports a spike count of zero.
    NonFiniteWeight {
        /// Presynaptic neuron of the offending synapse.
        pre: u32,
        /// Postsynaptic neuron of the offending synapse.
        post: u32,
    },
}

impl core::fmt::Display for NetError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::OutOfRange { index, n } => {
                write!(f, "neuron index {index} is past the network's {n} neurons")
            }
            Self::NonFiniteWeight { pre, post } => {
                write!(f, "synapse {pre} -> {post} has a non-finite weight")
            }
        }
    }
}

/// So that `?` works in a caller whose error type is `Box<dyn Error>`, which is what every example
/// and doctest in this crate uses. A library error that cannot cross that boundary forces every
/// caller to write a conversion, and the ones that do not write it reach for `.unwrap()`.
impl std::error::Error for NetError {}

impl NetBuilder {
    /// A network of `n` neurons and no synapses.
    #[must_use]
    pub fn new(n: usize) -> Self {
        Self { n, edges: Vec::new() }
    }

    /// Add one synapse `pre -> post` with `weight` volts per spike and `delay` ticks.
    ///
    /// # Errors
    ///
    /// [`NetError::OutOfRange`] if either index is past the neuron count, or
    /// [`NetError::NonFiniteWeight`] if the weight is not finite.
    pub fn connect(&mut self, pre: u32, post: u32, weight: f64, delay: u32) -> Result<&mut Self, NetError> {
        for idx in [pre, post] {
            if idx as usize >= self.n {
                return Err(NetError::OutOfRange { index: idx, n: self.n });
            }
        }
        // `!weight.is_finite()` rather than a comparison, because the comparison forms that clippy
        // suggests here accept NaN. See the lint note in Cargo.toml.
        if !weight.is_finite() {
            return Err(NetError::NonFiniteWeight { pre, post });
        }
        self.edges.push((pre, post, weight, delay));
        Ok(self)
    }

    /// Connect every neuron in `pre` to every neuron in `post`, uniform weight and delay.
    ///
    /// # Errors
    ///
    /// As [`NetBuilder::connect`].
    pub fn connect_all(
        &mut self,
        pre: &[u32],
        post: &[u32],
        weight: f64,
        delay: u32,
    ) -> Result<&mut Self, NetError> {
        for &a in pre {
            for &b in post {
                self.connect(a, b, weight, delay)?;
            }
        }
        Ok(self)
    }

    /// Finish, sorting synapses into CSR order.
    ///
    /// Sorted by `(pre, post)` so that the layout is deterministic for a given set of calls
    /// regardless of the order they were made in — which is what lets two runs of a network built
    /// by different code paths be compared spike for spike.
    #[must_use]
    pub fn build(mut self) -> Net {
        self.edges.sort_unstable_by_key(|&(a, b, _, _)| (a, b));
        let n_syn = self.edges.len();
        let mut offset = vec![0usize; self.n + 1];
        for &(a, _, _, _) in &self.edges {
            offset[a as usize + 1] += 1;
        }
        for i in 0..self.n {
            offset[i + 1] += offset[i];
        }
        let mut post = Vec::with_capacity(n_syn);
        let mut w = Vec::with_capacity(n_syn);
        let mut delay = Vec::with_capacity(n_syn);
        let mut max_delay = 0u32;
        for &(_, b, ww, d) in &self.edges {
            post.push(b);
            w.push(ww);
            delay.push(d);
            max_delay = max_delay.max(d);
        }
        Net { n: self.n, offset, post, w, delay, n_syn, max_delay }
    }
}

#[cfg(test)]
mod tests {
    use super::{NetBuilder, NetError};

    #[test]
    fn csr_groups_synapses_under_their_presynaptic_neuron() {
        let mut b = NetBuilder::new(3);
        b.connect(2, 0, 1.0, 0).unwrap();
        b.connect(0, 1, 2.0, 3).unwrap();
        b.connect(0, 2, 3.0, 1).unwrap();
        let net = b.build();

        assert_eq!(net.n_syn, 3, "three synapses, stored once each");
        assert_eq!(net.out_degree(0), 2);
        assert_eq!(net.out_degree(1), 0);
        assert_eq!(net.out_degree(2), 1);
        let from0: Vec<_> = net.out_of(0).collect();
        assert_eq!(from0, vec![(1, 2.0, 3), (2, 3.0, 1)], "sorted by post");
        assert_eq!(net.max_delay, 3);
    }

    /// The property the module doc is loudest about: a synapse is stored once, not twice.
    #[test]
    fn a_synapse_is_stored_once_unlike_an_ising_edge() {
        let mut b = NetBuilder::new(2);
        b.connect(0, 1, 1.0, 0).unwrap();
        let net = b.build();
        assert_eq!(net.n_syn, 1);
        assert_eq!(net.post.len(), 1, "post.len() IS the synapse count");
        assert_eq!(net.out_degree(1), 0, "the reverse synapse was not invented");
    }

    #[test]
    fn an_out_of_range_index_is_an_error_naming_both_numbers() {
        let mut b = NetBuilder::new(2);
        assert_eq!(b.connect(0, 5, 1.0, 0).err(), Some(NetError::OutOfRange { index: 5, n: 2 }));
        assert_eq!(b.connect(7, 0, 1.0, 0).err(), Some(NetError::OutOfRange { index: 7, n: 2 }));
    }

    /// A NaN weight completes the run and reports zero spikes. It is rejected at the boundary.
    #[test]
    fn a_non_finite_weight_is_refused_at_the_boundary() {
        let mut b = NetBuilder::new(2);
        assert_eq!(
            b.connect(0, 1, f64::NAN, 0).err(),
            Some(NetError::NonFiniteWeight { pre: 0, post: 1 })
        );
        assert_eq!(
            b.connect(0, 1, f64::INFINITY, 0).err(),
            Some(NetError::NonFiniteWeight { pre: 0, post: 1 })
        );
    }

    /// Build order must not change the layout, or two runs of the same network cannot be compared.
    #[test]
    fn the_layout_is_independent_of_the_order_the_synapses_were_added() {
        let mut a = NetBuilder::new(4);
        a.connect(3, 1, 0.5, 2).unwrap();
        a.connect(0, 2, 1.5, 0).unwrap();
        a.connect(0, 1, 2.5, 1).unwrap();
        let mut b = NetBuilder::new(4);
        b.connect(0, 1, 2.5, 1).unwrap();
        b.connect(3, 1, 0.5, 2).unwrap();
        b.connect(0, 2, 1.5, 0).unwrap();
        assert_eq!(a.build(), b.build());
    }

    #[test]
    fn in_degrees_are_counted_over_the_whole_synapse_list() {
        let mut b = NetBuilder::new(3);
        b.connect_all(&[0, 1], &[2], 1.0, 0).unwrap();
        b.connect(2, 0, 1.0, 0).unwrap();
        let net = b.build();
        assert_eq!(net.in_degrees(), vec![1, 0, 2]);
        assert!((net.fan_out_mean() - 1.0).abs() < 1e-15);
    }

    #[test]
    fn an_index_past_the_end_has_no_synapses_rather_than_a_panic() {
        let net = NetBuilder::new(2).build();
        assert_eq!(net.out_degree(99), 0);
        assert_eq!(net.out_of(99).count(), 0);
    }
}
