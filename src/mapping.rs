//! Placing a network on cores: the problem between a model and a chip, and where the energy goes.
//!
//! # The lesson
//!
//! [`crate::hardware`] answers whether a network **fits**. It does not answer where anything goes,
//! and that second question is the one that costs money. A neuromorphic chip is not a pool of
//! neurons; it is a grid of small memories joined by a packet network. A spike delivered to a
//! synapse on the **same core** touches one local SRAM. The same spike delivered to a core three
//! tiles away is a packet that enters a router, is buffered, arbitrated and forwarded three times,
//! and then delivered. The same spike leaving the die is a `SerDes` crossing.
//!
//! Those three costs are not within a factor of two of each other. Dally and Towles (*Route
//! Packets, Not Wires: On-Chip Interconnection Networks*, DAC 2001) made the general argument for
//! why: on-chip wires are cheap and short, off-chip pins are few and expensive, and the ratio grows
//! with every process node. So **placement is an energy decision**, and a mapper that minimises
//! nothing in particular has silently chosen a bill.
//!
//! What this module buys: the placement as an object you can count, compare and refuse. What it
//! costs: the objective it minimises — **cut edges**, the synapses whose two endpoints land on
//! different cores — is a proxy. It weights every crossing the same, and a crossing that carries
//! one spike a second is not the crossing that carries a thousand. [`spike_hops`] is the honest
//! version, weighting each axon by how often it actually fired, and it needs a workload to exist.
//!
//! # The three things this module does
//!
//! 1. **Partition.** Assign each neuron a core under [`CoreLimits`], minimising [`Partition::cut_edges`].
//!    Two partitioners: [`partition_greedy`], a single-pass streaming assignment, and
//!    [`partition_refined`], which follows it with Kernighan-Lin exchange passes. The difference is
//!    **measured** and reported in [`Plan`], never asserted — see
//!    `kernighan_lin_beats_the_streaming_greedy_on_a_cycle_whose_optimum_is_two`, which builds a
//!    graph whose optimum is known in closed form and prints both numbers.
//! 2. **Route.** [`Fabric`] is the core-to-core network: a crossbar, a 2-D mesh, a 2-D torus, or
//!    the triangular torus `SpiNNaker` actually uses. [`multicast_tree`] builds the one-to-many
//!    delivery tree a spike takes, because a neuromorphic router replicates packets in the fabric
//!    rather than sending one copy per target — that is what `SpiNNaker`'s multicast router is for
//!    (Furber, Galluppi, Temple and Plana, *The `SpiNNaker` Project*, Proceedings of the IEEE
//!    102(5):652–665, 2014).
//! 3. **Price, or refuse.** [`SpikeHops::bill`] converts hops into joules against [`HopPrices`],
//!    and **every price in this module is `None`**. That is the finding, in the same form
//!    [`crate::ledger`] states it: *this review did not locate a published per-hop joule, on-chip
//!    or off-chip, for any commercially available neuromorphic part.* The hop counts themselves are
//!    exact integers and are useful without a price.
//!
//! # Where a synapse's memory lives, and why it is the postsynaptic core
//!
//! [`Partition`] charges a synapse's storage to the core holding its **postsynaptic** neuron. That
//! is not a modelling convenience, it is what the parts do: a `TrueNorth` core's crossbar column is
//! one neuron's 256 inputs (Merolla et al., *Science* 345(6197):668–673, 2014); `Loihi`'s synaptic
//! memory is read by the destination core when a spike arrives (Davies et al., *IEEE Micro*
//! 38(1):82–99, 2018); `SpiNNaker` holds the synaptic matrix in the destination chip's SDRAM and
//! DMAs the row on arrival (Furber et al., 2014). All three fetch on the receiving side. A mapper
//! that charged the presynaptic core would balance the wrong quantity and report fan-out where the
//! hardware limit is fan-in.
//!
//! # Units
//!
//! Hops are **counts**, not seconds and not metres: one hop is one traversal of one link between
//! two adjacent cores. Joules are joules, SI, and are `None` everywhere in this module.
//! [`HopPrices`] is the only place a joule appears and it appears as an `Option`.
//!
//! # What this module does not do
//!
//! * **No multilevel partitioning.** `METIS` (Karypis and Kumar, *SIAM Journal on Scientific
//!   Computing* 20(1):359–392, 1998) coarsens the graph, partitions the small one, and refines on
//!   the way back up; it is the reference method and it beats everything here on large graphs.
//!   This review implemented the two algorithms whose behaviour a reader can follow by hand.
//! * **No spectral bisection.** The Fiedler vector of the graph Laplacian is the other classical
//!   answer. [`crate::reservoir`] has a power iteration that could carry it. It is not here.
//! * **No timing.** Hops are counted; latency, congestion, buffer occupancy and deadlock are not
//!   modelled at all. A mapping that minimises hops can still miss a deadline.
//! * **No floorplan.** Chips are assigned to cores by `core / cores_per_chip`, a row-major block.
//!   A real machine's die boundaries are a floorplan this crate does not have.
//!
//! The literature that does solve the full problem, named so a reader can go and read it rather
//! than reimplement it: `PACMAN` for `SpiNNaker` (Galluppi et al., ACM Computing Frontiers, 2012),
//! the `Loihi` compiler (Lin et al., *Mapping spiking neural networks onto a manycore neuromorphic
//! architecture*, PLDI 2018), and `SpiNeMap` (Balaji et al., IEEE Transactions on VLSI Systems,
//! 2020), which minimises exactly the inter-core spike count this module counts. This review read
//! the abstracts and the problem statements; page numbers for the last two are not reproduced here
//! because it did not verify them against the published volumes.

use crate::hardware::Part;
use crate::ledger::{Evidence, Ledger};
use crate::net::Net;
use crate::rng::Rng;
use core::fmt;

// ---------------------------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------------------------

/// Why a mapping question could not be answered.
///
/// Every variant names the offending value. A mapper that returns "infeasible" without saying
/// which neuron, which core and by how much leaves the user bisecting their own network.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MapError {
    /// A partition was asked for with zero cores, which admits no assignment of any neuron.
    NoCores,
    /// A neuron was assigned a core index at or past the core count.
    CoreOutOfRange {
        /// The neuron carrying the bad assignment.
        neuron: usize,
        /// The core index it named.
        core: u32,
        /// The core count it exceeded.
        n_cores: u32,
    },
    /// A core index was handed to a [`Fabric`] that does not have it.
    CoreNotOnFabric {
        /// The offending index.
        core: u32,
        /// Cores the fabric has.
        n_cores: u64,
    },
    /// A per-neuron array did not have one entry per neuron.
    LengthMismatch {
        /// What the array is, e.g. `"core assignment"` or `"spike count"`.
        what: &'static str,
        /// Entries supplied.
        got: usize,
        /// Entries the network needs, which is [`crate::net::Net::n`].
        neurons: usize,
    },
    /// The cores available cannot hold the neurons, whatever the assignment.
    ///
    /// A capacity arithmetic failure, detected before any work is done. `n_cores * neurons_per_core`
    /// is the whole machine, and a network larger than it does not map by being packed better.
    NotEnoughRoom {
        /// Neurons in the network.
        neurons: usize,
        /// Cores offered.
        n_cores: u32,
        /// Neurons one core holds.
        neurons_per_core: u32,
        /// `n_cores * neurons_per_core`, the whole machine.
        capacity: u64,
    },
    /// One neuron's fan-in alone exceeds a whole core's synapse storage.
    ///
    /// **Not a packing failure.** No assignment of any network containing this neuron fits, because
    /// the neuron's inputs cannot be split across two cores without splitting the neuron. The fix
    /// is a tree of partial summers, which costs neurons and a tick of latency per level — see
    /// [`crate::hardware`]'s module doc on walls versus budgets.
    FanInExceedsCore {
        /// The offending neuron.
        neuron: usize,
        /// Its in-degree.
        fan_in: u64,
        /// A core's synapse capacity.
        cap: u64,
    },
    /// The streaming partitioner reached a neuron with no core that could still take it.
    ///
    /// Distinct from [`MapError::NotEnoughRoom`]: the machine is big enough in total and this
    /// single pass painted itself into a corner. Retry with another seed, more cores, or
    /// [`partition_refined`].
    NoFeasibleCore {
        /// The neuron that could not be placed.
        neuron: usize,
        /// Its in-degree, which is the synapse storage it needs.
        in_degree: u64,
        /// Cores tried.
        n_cores: u32,
    },
    /// A stated capacity was zero, which holds nothing and admits no assignment.
    ZeroCapacity {
        /// Which limit was zero: `"neurons per core"` or `"synapses per core"`.
        which: &'static str,
    },
    /// The routes from one source did not union into a tree.
    ///
    /// A multicast tree is the union of the point-to-point routes from one root, and that union is
    /// a tree only if the fabric's routing is **prefix-closed**: the route to a node on the way to
    /// `d` is a prefix of the route to `d`. Dimension-order routing on a mesh is; an arbitrary
    /// shortest-path rule need not be. Reported rather than silently returning a subgraph with a
    /// cycle in it, whose edge count would understate the packet replication.
    RouteUnionIsNotATree {
        /// Root core of the attempted tree.
        root: u32,
        /// Distinct cores the union touched.
        cores: usize,
        /// Distinct links the union touched. A tree has exactly `cores - 1`.
        edges: usize,
    },
    /// A hop or delivery count exceeded `u64` while being accumulated.
    ///
    /// `spikes_per_neuron` is unvalidated `u64`, and `u64::MAX` spikes from one neuron used to wrap
    /// silently in release — into a module whose claim is that the hop counts are exact integers.
    CountOverflow {
        /// The neuron whose contribution overflowed the count.
        neuron: usize,
        /// Which count: `"multicast hops"`, `"unicast hops"`, `"chip crossings"`, `"on-core
        /// deliveries"`, `"off-core deliveries"` or `"spikes"`.
        what: &'static str,
    },
    /// The cores available cannot hold the synapses, whatever the assignment.
    ///
    /// The synapse twin of [`MapError::NotEnoughRoom`]. Without it, 100 synapses offered 20 slots
    /// were reported as [`MapError::NoFeasibleCore`] — "try another seed or refinement" — and every
    /// seed and the refiner failed the same way, because arithmetic is not a packing problem.
    NotEnoughSynapseRoom {
        /// Synapses in the network.
        synapses: u64,
        /// Cores offered.
        n_cores: u32,
        /// Synapses one core stores.
        synapses_per_core: u64,
        /// `n_cores * synapses_per_core`, the whole machine.
        capacity: u64,
    },
    /// More cores than this implementation will allocate per-core bookkeeping for.
    ///
    /// [`Partition::loads`], [`Partition::check`] and the streaming pass each hold a record per
    /// core. At [`MAX_CORES`] that is stated in the constant's doc; past it the allocation is
    /// refused here, by the same rule [`MapError::RefinementTooLarge`] applies to the refiner.
    TooManyCores {
        /// Cores asked for.
        n_cores: u32,
        /// [`MAX_CORES`].
        limit: u32,
    },
    /// Kernighan-Lin refinement was asked for on a problem too large for its working set.
    ///
    /// The refiner holds one `i64` per (neuron, core) pair. Refused rather than allocated, because
    /// the allocation that would have been attempted is stated in the error.
    RefinementTooLarge {
        /// Neurons in the network.
        neurons: usize,
        /// Cores offered.
        n_cores: u32,
        /// `neurons * n_cores`, the working set in cells.
        cells: u64,
        /// The largest working set this implementation will allocate, in cells.
        limit: u64,
    },
}

impl fmt::Display for MapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoCores => f.write_str("a partition needs at least one core"),
            Self::CoreOutOfRange { neuron, core, n_cores } => write!(
                f,
                "neuron {neuron} is assigned core {core}, past the {n_cores} cores available"
            ),
            Self::CoreNotOnFabric { core, n_cores } => {
                write!(f, "core {core} is not on this fabric, which has {n_cores}")
            }
            Self::LengthMismatch { what, got, neurons } => {
                write!(f, "{what} has {got} entries for a network of {neurons} neurons")
            }
            Self::NotEnoughRoom { neurons, n_cores, neurons_per_core, capacity } => write!(
                f,
                "{neurons} neurons do not fit in {n_cores} cores of {neurons_per_core} \
                 ({capacity} total)"
            ),
            Self::FanInExceedsCore { neuron, fan_in, cap } => write!(
                f,
                "neuron {neuron} has {fan_in} inputs and a core stores {cap}; no placement fixes \
                 this, the neuron has to be split"
            ),
            Self::NoFeasibleCore { neuron, in_degree, n_cores } => write!(
                f,
                "the streaming pass ran out of room at neuron {neuron} (in-degree {in_degree}) \
                 with {n_cores} cores; try another seed or refinement"
            ),
            Self::ZeroCapacity { which } => write!(f, "{which} is zero, which holds nothing"),
            Self::RouteUnionIsNotATree { root, cores, edges } => write!(
                f,
                "routes from core {root} touched {cores} cores over {edges} links; a tree needs \
                 exactly {} links",
                cores.saturating_sub(1)
            ),
            Self::CountOverflow { neuron, what } => {
                write!(f, "the {what} count overflowed u64 at neuron {neuron}")
            }
            Self::NotEnoughSynapseRoom { synapses, n_cores, synapses_per_core, capacity } => write!(
                f,
                "{synapses} synapses do not fit in {n_cores} cores of {synapses_per_core} \
                 ({capacity} total)"
            ),
            Self::TooManyCores { n_cores, limit } => write!(
                f,
                "{n_cores} cores is past the {limit} this implementation keeps per-core records for"
            ),
            Self::RefinementTooLarge { neurons, n_cores, cells, limit } => write!(
                f,
                "refinement of {neurons} neurons over {n_cores} cores needs {cells} cells, above \
                 this implementation's limit of {limit}"
            ),
        }
    }
}

impl std::error::Error for MapError {}

// ---------------------------------------------------------------------------------------------
// Fabric
// ---------------------------------------------------------------------------------------------

/// The network **between** cores: which cores are adjacent and how far apart everything is.
///
/// Cores are numbered row-major on the two-dimensional fabrics: core `c` sits at
/// `(x, y) = (c % cols, c / cols)`. That is a convention of this crate, not of any part; a real
/// machine's numbering is whatever its toolchain says it is.
///
/// Every variant's [`Fabric::hops`] is a **closed form**, and every one of them is checked against
/// breadth-first search over [`Fabric::neighbours`] exhaustively in the tests. That is the whole
/// verification strategy here: the formula is the claim, the search is the ground truth.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fabric {
    /// Every core one link from every other: the idealised zero-diameter case.
    ///
    /// No fabricated neuromorphic part of any size is wired this way, and it is here as the
    /// **baseline that makes placement look free**. A mapper evaluated on a crossbar reports the
    /// same cost for every partition, which is exactly the mistake this module exists to expose.
    Crossbar {
        /// Cores on the crossbar.
        cores: u32,
    },
    /// A two-dimensional mesh with no wraparound: the `TrueNorth` and `Loihi` on-die network.
    ///
    /// Merolla et al., *Science* 345(6197), 2014, for `TrueNorth`'s 64x64 grid of cores; Davies et
    /// al., *IEEE Micro* 38(1), 2018, for `Loihi`'s mesh. Distance is the Manhattan distance, which
    /// is exact for any minimal routing on a mesh and is what [`Fabric::hops`] returns.
    Mesh2D {
        /// Cores per row. Zero means an empty fabric.
        cols: u32,
        /// Rows. Zero means an empty fabric.
        rows: u32,
    },
    /// A two-dimensional mesh with both axes wrapped.
    ///
    /// Halves the diameter of a mesh at the cost of long wraparound links. Included because it is
    /// the textbook comparison to [`Fabric::Mesh2D`] and because the distance function — the
    /// shorter of the two ways round, per axis — is a closed form worth a reader's attention.
    Torus2D {
        /// Cores per row.
        cols: u32,
        /// Rows.
        rows: u32,
    },
    /// Six links per core, on a wrapped triangular lattice: the `SpiNNaker` machine's topology.
    ///
    /// Furber, Galluppi, Temple and Plana, *The `SpiNNaker` Project*, Proceedings of the IEEE
    /// 102(5):652–665, 2014. Each chip has six inter-chip links, to the neighbours at `(+1, 0)`,
    /// `(-1, 0)`, `(0, +1)`, `(0, -1)`, `(+1, +1)` and `(-1, -1)`, and the boards are wrapped into
    /// a torus.
    ///
    /// The distance is the hexagonal-lattice distance: for a displacement `(x, y)` it is
    /// `max(|x|, |y|)` when `x` and `y` share a sign — the diagonal link does both axes at once —
    /// and `|x| + |y|` when they do not, because there is no anti-diagonal link. Wraparound is
    /// handled by taking the best of the four representatives, which is exhaustive: the distance is
    /// non-decreasing in `|x|` and in `|y|` within a fixed pair of signs, so the minimum in each of
    /// the four sign classes is at the smallest representative in that class.
    TriangularTorus {
        /// Chips per row.
        cols: u32,
        /// Rows of chips.
        rows: u32,
    },
}

/// The six link directions of a triangular torus, in the order routes prefer them.
const TRI_DIRS: [(i64, i64); 6] = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1)];

/// The four link directions of a mesh or torus.
const RECT_DIRS: [(i64, i64); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];

/// Hexagonal-lattice distance of a displacement, in hops. See [`Fabric::TriangularTorus`].
fn tri_dist(x: i64, y: i64) -> u64 {
    let (ax, ay) = (x.unsigned_abs(), y.unsigned_abs());
    if (x > 0 && y > 0) || (x < 0 && y < 0) { ax.max(ay) } else { ax + ay }
}

/// The signed displacement from `a` to `b` on a ring of `n`, taking the shorter way and preferring
/// the forward direction on an exact tie. `n == 0` is not reachable from a fabric with cores.
fn ring_delta(a: u32, b: u32, n: u32) -> i64 {
    if n == 0 {
        return 0;
    }
    let n = i64::from(n);
    let fwd = (i64::from(b) - i64::from(a)).rem_euclid(n);
    let back = fwd - n;
    if fwd <= -back { fwd } else { back }
}

impl Fabric {
    /// Cores on this fabric. Zero for a fabric with a zero dimension.
    #[must_use]
    pub fn n_cores(&self) -> u64 {
        match *self {
            Self::Crossbar { cores } => u64::from(cores),
            Self::Mesh2D { cols, rows }
            | Self::Torus2D { cols, rows }
            | Self::TriangularTorus { cols, rows } => u64::from(cols) * u64::from(rows),
        }
    }

    /// `(cols, rows)` for the two-dimensional fabrics, `None` for a crossbar, which has no geometry.
    #[must_use]
    pub fn dims(&self) -> Option<(u32, u32)> {
        match *self {
            Self::Crossbar { .. } => None,
            Self::Mesh2D { cols, rows }
            | Self::Torus2D { cols, rows }
            | Self::TriangularTorus { cols, rows } => Some((cols, rows)),
        }
    }

    /// `(x, y)` of a core, row-major. `None` for an index past the fabric or on a crossbar.
    #[must_use]
    pub fn coords(&self, core: u32) -> Option<(u32, u32)> {
        let (cols, _) = self.dims()?;
        if cols == 0 || u64::from(core) >= self.n_cores() {
            return None;
        }
        Some((core % cols, core / cols))
    }

    /// Hops between two cores under minimal routing, in closed form.
    ///
    /// `None` when either index is past the fabric — a distance to a core that is not there has no
    /// value, and returning a large number would let a cost model sum it.
    #[must_use]
    pub fn hops(&self, a: u32, b: u32) -> Option<u64> {
        let n = self.n_cores();
        if u64::from(a) >= n || u64::from(b) >= n {
            return None;
        }
        if a == b {
            return Some(0);
        }
        match *self {
            Self::Crossbar { .. } => Some(1),
            Self::Mesh2D { cols, rows } => {
                let _ = rows;
                let (ax, ay) = (a % cols, a / cols);
                let (bx, by) = (b % cols, b / cols);
                Some(u64::from(ax.abs_diff(bx)) + u64::from(ay.abs_diff(by)))
            }
            Self::Torus2D { cols, rows } => {
                let (ax, ay) = (a % cols, a / cols);
                let (bx, by) = (b % cols, b / cols);
                let dx = ring_delta(ax, bx, cols).unsigned_abs();
                let dy = ring_delta(ay, by, rows).unsigned_abs();
                Some(dx + dy)
            }
            Self::TriangularTorus { .. } => self.tri_vector(a, b).map(|(x, y)| tri_dist(x, y)),
        }
    }

    /// The displacement a triangular-torus route takes, chosen among the four wrap representatives.
    ///
    /// Tie-break, in order: shortest distance, then smallest `|x| + |y|`, then smallest `|x|`, then
    /// non-negative `x`, then non-negative `y`. Determinism comes from the strict `<` on the key
    /// and holds under any total order; the four lower levels are a convention that makes the
    /// chosen representative a function of the endpoints alone, not a correctness requirement —
    /// collapsing them to the distance alone leaves every hop, route, diameter and spanning-tree
    /// test green, which is the honest statement of what they carry.
    fn tri_vector(&self, a: u32, b: u32) -> Option<(i64, i64)> {
        let (cols, rows) = self.dims()?;
        if cols == 0 || rows == 0 {
            return None;
        }
        let (ax, ay) = (a % cols, a / cols);
        let (bx, by) = (b % cols, b / cols);
        let fx = (i64::from(bx) - i64::from(ax)).rem_euclid(i64::from(cols));
        let fy = (i64::from(by) - i64::from(ay)).rem_euclid(i64::from(rows));
        let mut best: Option<(i64, i64)> = None;
        let mut best_key: (u64, u64, u64, u8, u8) = (u64::MAX, u64::MAX, u64::MAX, 2, 2);
        for x in [fx, fx - i64::from(cols)] {
            for y in [fy, fy - i64::from(rows)] {
                let key = (
                    tri_dist(x, y),
                    x.unsigned_abs() + y.unsigned_abs(),
                    x.unsigned_abs(),
                    u8::from(x < 0),
                    u8::from(y < 0),
                );
                if key < best_key {
                    best_key = key;
                    best = Some((x, y));
                }
            }
        }
        best
    }

    /// The cores one link away, sorted ascending and without duplicates or the core itself.
    ///
    /// Deduplication matters on a small torus: on a fabric two columns wide, stepping `+1` and
    /// stepping `-1` reach the same core, and counting it twice would make a degree histogram wrong
    /// and a breadth-first search do the same work twice.
    #[must_use]
    pub fn neighbours(&self, c: u32) -> Vec<u32> {
        let n = self.n_cores();
        if u64::from(c) >= n {
            return Vec::new();
        }
        let mut v: Vec<u32> = match *self {
            Self::Crossbar { cores } => (0..cores).filter(|&o| o != c).collect(),
            Self::Mesh2D { cols, rows } => {
                let (x, y) = (i64::from(c % cols), i64::from(c / cols));
                RECT_DIRS
                    .iter()
                    .filter_map(|&(dx, dy)| {
                        let (nx, ny) = (x + dx, y + dy);
                        if nx < 0 || ny < 0 || nx >= i64::from(cols) || ny >= i64::from(rows) {
                            None
                        } else {
                            Some((ny * i64::from(cols) + nx) as u32)
                        }
                    })
                    .collect()
            }
            Self::Torus2D { cols, rows } => wrapped_neighbours(c, cols, rows, &RECT_DIRS),
            Self::TriangularTorus { cols, rows } => wrapped_neighbours(c, cols, rows, &TRI_DIRS),
        };
        v.sort_unstable();
        v.dedup();
        v.retain(|&o| o != c);
        v
    }

    /// The route from `a` to `b`, inclusive of both ends, one core per hop.
    ///
    /// Dimension-order on the mesh and torus — all of the `x` displacement, then all of the `y` —
    /// which is what a mesh network-on-chip implements and what makes it deadlock-free (Dally and
    /// Towles, DAC 2001). Diagonal-first on the triangular torus. `None` for an index past the
    /// fabric.
    ///
    /// The length is always `hops(a, b) + 1`, and every consecutive pair is a link. Both are
    /// asserted in `a_route_has_exactly_as_many_steps_as_the_hop_count_and_uses_only_links`.
    #[must_use]
    pub fn route(&self, a: u32, b: u32) -> Option<Vec<u32>> {
        let n = self.n_cores();
        if u64::from(a) >= n || u64::from(b) >= n {
            return None;
        }
        if a == b {
            return Some(vec![a]);
        }
        match *self {
            Self::Crossbar { .. } => Some(vec![a, b]),
            Self::Mesh2D { cols, .. } => {
                let (ax, ay) = (a % cols, a / cols);
                let (bx, by) = (b % cols, b / cols);
                let dx = i64::from(bx) - i64::from(ax);
                let dy = i64::from(by) - i64::from(ay);
                Some(walk(ax, ay, dx, dy, cols, 0, false))
            }
            Self::Torus2D { cols, rows } => {
                let (ax, ay) = (a % cols, a / cols);
                let (bx, by) = (b % cols, b / cols);
                let dx = ring_delta(ax, bx, cols);
                let dy = ring_delta(ay, by, rows);
                Some(walk(ax, ay, dx, dy, cols, rows, false))
            }
            Self::TriangularTorus { cols, rows } => {
                let (x, y) = self.tri_vector(a, b)?;
                let (ax, ay) = (a % cols, a / cols);
                Some(walk(ax, ay, x, y, cols, rows, true))
            }
        }
    }

    /// The longest minimal route on this fabric, in hops.
    ///
    /// Closed form for the crossbar (`1`, or `0` when there is nothing to reach), the mesh
    /// (`(cols-1) + (rows-1)`) and the rectangular torus (`cols/2 + rows/2`). For the triangular
    /// torus this review did not locate a closed form and computes it as the farthest core from
    /// core 0 — which is the diameter because a torus is vertex-transitive, so every core sees the
    /// same distance profile. That case costs `O(cores)`.
    ///
    /// `None` for an empty fabric.
    #[must_use]
    pub fn diameter(&self) -> Option<u64> {
        let n = self.n_cores();
        if n == 0 {
            return None;
        }
        match *self {
            Self::Crossbar { .. } => Some(u64::from(n > 1)),
            Self::Mesh2D { cols, rows } => Some(u64::from(cols - 1) + u64::from(rows - 1)),
            Self::Torus2D { cols, rows } => Some(u64::from(cols / 2) + u64::from(rows / 2)),
            Self::TriangularTorus { .. } => {
                // ⛔ `n` does NOT fit u32 for every fabric a caller can build: `cols` and `rows`
                // are each u32, so `100_000 x 100_000` is 1e10 cores, and the comment that stood
                // here said otherwise. The sweep below is a measurement (135 ms at 2^24 cores,
                // ~35 s at 2^32), so past the cast's range it is refused rather than truncated
                // — a truncated sweep never asks about the farthest core and reports a wrong
                // diameter with no error. The square case has an empirical closed form,
                // `floor(2·cols/3)` at 8x8, 16x16, 32x32 and 64x64; not adopted, because this
                // review did not derive it for the non-square case.
                if n > u64::from(u32::MAX) {
                    return None;
                }
                let mut worst = 0u64;
                for b in 0..n {
                    if let Some(h) = self.hops(0, b as u32) {
                        worst = worst.max(h);
                    }
                }
                Some(worst)
            }
        }
    }
}

/// Neighbours of `c` on a wrapped `cols` x `rows` lattice under `dirs`, unsorted and possibly with
/// duplicates; the caller sorts and dedups.
fn wrapped_neighbours(c: u32, cols: u32, rows: u32, dirs: &[(i64, i64)]) -> Vec<u32> {
    let (x, y) = (i64::from(c % cols), i64::from(c / cols));
    dirs.iter()
        .map(|&(dx, dy)| {
            let nx = (x + dx).rem_euclid(i64::from(cols));
            let ny = (y + dy).rem_euclid(i64::from(rows));
            (ny * i64::from(cols) + nx) as u32
        })
        .collect()
}

/// Emit the cores of a route from `(x0, y0)` by displacement `(dx, dy)`.
///
/// `rows == 0` means no wraparound (a mesh); otherwise coordinates wrap. `diagonal` takes the
/// combined `(±1, ±1)` link first where both displacements share a sign, which is the triangular
/// torus; otherwise all of `x` then all of `y`, which is dimension-order routing.
fn walk(x0: u32, y0: u32, dx: i64, dy: i64, cols: u32, rows: u32, diagonal: bool) -> Vec<u32> {
    let (mut x, mut y) = (i64::from(x0), i64::from(y0));
    let ci = i64::from(cols);
    let ri = if rows == 0 { 0 } else { i64::from(rows) };
    let mut out = Vec::new();
    let push = |x: i64, y: i64, out: &mut Vec<u32>| {
        let (px, py) = if ri == 0 { (x, y) } else { (x.rem_euclid(ci), y.rem_euclid(ri)) };
        out.push((py * ci + px) as u32);
    };
    push(x, y, &mut out);
    let (sx, sy) = (dx.signum(), dy.signum());
    let (mut rx, mut ry) = (dx.abs(), dy.abs());
    // The diagonal link is `(+1, +1)` and `(-1, -1)`; there is no anti-diagonal. Taking it when the
    // two displacements disagree in sign moves AWAY from the destination on one axis, which is the
    // defect `a_route_has_exactly_as_many_steps_as_the_hop_count_and_uses_only_links` caught: the
    // route was two hops where the distance was three, and the two hops were not links.
    if diagonal && sx != 0 && sx == sy {
        let k = rx.min(ry);
        for _ in 0..k {
            x += sx;
            y += sy;
            push(x, y, &mut out);
        }
        rx -= k;
        ry -= k;
    }
    for _ in 0..rx {
        x += sx;
        push(x, y, &mut out);
    }
    for _ in 0..ry {
        y += sy;
        push(x, y, &mut out);
    }
    out
}

// ---------------------------------------------------------------------------------------------
// Multicast
// ---------------------------------------------------------------------------------------------

/// The delivery tree one axon takes: the union of the routes from a source core to every core
/// holding one of its targets.
///
/// A neuromorphic router replicates a packet **inside the fabric** — one copy enters at the source
/// and the routers fork it at the branch points. `SpiNNaker`'s multicast router does exactly this,
/// keyed on the source neuron's routing key (Furber et al., Proceedings of the IEEE 102(5), 2014).
/// The cost of a delivery is therefore the number of **links in the tree**, not the sum of the
/// distances to the destinations, and the difference is [`McastTree::saving`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct McastTree {
    /// Core the axon leaves from.
    pub root: u32,
    /// Every core the tree touches, including the root, sorted ascending and without duplicates.
    pub cores: Vec<u32>,
    /// Links in the tree, each as `(lower core, higher core)`, sorted and without duplicates.
    ///
    /// Undirected pairs because a link is one piece of wire; the direction of travel is the tree's
    /// orientation from the root and is not stored.
    pub edges: Vec<(u32, u32)>,
    /// Link traversals one spike costs on this tree, which is `edges.len()`.
    ///
    /// Duplicated as a field so that a cost model can sum it without re-walking the edge list.
    pub hops: u64,
    /// What the same delivery would cost with no multicast at all: one packet per **destination
    /// core**, summed over the shortest paths.
    ///
    /// The baseline is per destination core and not per destination neuron. A per-neuron baseline
    /// is larger again — every synapse its own packet — and this review chose the tighter of the
    /// two so that [`McastTree::saving`] is not flattered by a straw man.
    pub unicast_hops: u64,
}

impl McastTree {
    /// Fraction of the unicast cost the tree saves, in `[0, 1)`.
    ///
    /// `None` when there is nothing to send — a root with no destination other than itself, where
    /// the ratio is `0/0`.
    #[must_use]
    pub fn saving(&self) -> Option<f64> {
        if self.unicast_hops == 0 {
            return None;
        }
        Some(1.0 - self.hops as f64 / self.unicast_hops as f64)
    }

    /// Links in this tree that cross a chip boundary, under a row-major block assignment of cores
    /// to chips.
    ///
    /// `None` when `cores_per_chip` is zero, which describes no machine. The block assignment is a
    /// simplification stated in the module doc: a real part's die boundary is a floorplan.
    #[must_use]
    pub fn crossings(&self, cores_per_chip: u32) -> Option<u64> {
        if cores_per_chip == 0 {
            return None;
        }
        Some(self.edges.iter().filter(|&&(a, b)| a / cores_per_chip != b / cores_per_chip).count() as u64)
    }
}

/// Build the multicast tree from `root` to `dests` on `fabric`.
///
/// Destinations are deduplicated and the root is always in the tree. A destination equal to the
/// root contributes nothing, which is the on-core delivery case.
///
/// # Errors
///
/// [`MapError::CoreNotOnFabric`] for any index past the fabric, and
/// [`MapError::RouteUnionIsNotATree`] if the fabric's routing is not prefix-closed, in which case
/// the union has a cycle and its edge count would understate the replication.
pub fn multicast_tree(fabric: &Fabric, root: u32, dests: &[u32]) -> Result<McastTree, MapError> {
    let n = fabric.n_cores();
    for &c in core::iter::once(&root).chain(dests) {
        if u64::from(c) >= n {
            return Err(MapError::CoreNotOnFabric { core: c, n_cores: n });
        }
    }
    let mut d: Vec<u32> = dests.to_vec();
    d.sort_unstable();
    d.dedup();

    let mut edges: Vec<(u32, u32)> = Vec::new();
    let mut cores: Vec<u32> = vec![root];
    let mut unicast_hops = 0u64;
    for &dst in &d {
        if dst == root {
            continue;
        }
        // Both are in range, so neither call can be None.
        let Some(path) = fabric.route(root, dst) else {
            return Err(MapError::CoreNotOnFabric { core: dst, n_cores: n });
        };
        unicast_hops += fabric.hops(root, dst).unwrap_or(0);
        for w in path.windows(2) {
            edges.push((w[0].min(w[1]), w[0].max(w[1])));
        }
        cores.extend_from_slice(&path);
    }
    edges.sort_unstable();
    edges.dedup();
    cores.sort_unstable();
    cores.dedup();
    if edges.len() + 1 != cores.len() {
        return Err(MapError::RouteUnionIsNotATree {
            root,
            cores: cores.len(),
            edges: edges.len(),
        });
    }
    let hops = edges.len() as u64;
    Ok(McastTree { root, cores, edges, hops, unicast_hops })
}

// ---------------------------------------------------------------------------------------------
// Limits, loads, feasibility
// ---------------------------------------------------------------------------------------------

/// What one core holds. The partitioning constraints, separated from [`Part`] so that a
/// hypothetical machine can be described without inventing a citation for it.
///
/// Every field is an `Option` for the reason [`crate::hardware::Spec`] gives: a limit nobody
/// published and a limit of zero are different statements, and a `u32` defaulting to zero merges
/// them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CoreLimits {
    /// Neurons one core holds. `None` leaves the count unconstrained and unchecked.
    pub neurons_per_core: Option<u32>,
    /// Synapses one core stores, charged to the **postsynaptic** core. See the module doc.
    pub synapses_per_core: Option<u64>,
    /// Presynaptic sources one neuron may have — a property of the network, not of the placement,
    /// and therefore the one constraint repartitioning cannot relieve.
    pub max_fan_in: Option<u32>,
}

impl CoreLimits {
    /// No limits at all: every check lands in [`Feasibility::unchecked`] and the verdict is `None`.
    pub const UNLIMITED: Self =
        Self { neurons_per_core: None, synapses_per_core: None, max_fan_in: None };

    /// Limits stated explicitly.
    #[must_use]
    pub const fn new(
        neurons_per_core: Option<u32>,
        synapses_per_core: Option<u64>,
        max_fan_in: Option<u32>,
    ) -> Self {
        Self { neurons_per_core, synapses_per_core, max_fan_in }
    }

    /// Limits taken from a part in [`crate::hardware`], carrying its empty fields through as `None`.
    ///
    /// A part that states no fan-in wall — every software-scheduled part in that table — yields
    /// `max_fan_in: None`, and the fan-in constraint is then reported as **unchecked** rather than
    /// as satisfied. That distinction is the whole reason this is not a `u32`.
    #[must_use]
    pub fn from_part(part: &Part) -> Self {
        Self {
            neurons_per_core: part.neurons_per_core.value,
            synapses_per_core: part.synapses_per_core.value,
            max_fan_in: part.max_fan_in.value,
        }
    }
}

/// What one core is actually carrying under a partition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CoreLoad {
    /// Which core.
    pub core: u32,
    /// Neurons placed here.
    pub neurons: u64,
    /// Synapses stored here: the sum of the in-degrees of the resident neurons.
    pub synapses: u64,
    /// The largest in-degree among the resident neurons, and which neuron has it.
    ///
    /// `None` for an empty core: a maximum over nothing has no value, and reporting zero would
    /// make an empty core look like one holding a neuron with no inputs.
    pub worst_fan_in: Option<(usize, u64)>,
}

/// A constraint one core, or one neuron, violates — and by how much.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoreBind {
    /// A core holds more neurons than it can.
    Neurons {
        /// The offending core.
        core: u32,
        /// Neurons placed there.
        used: u64,
        /// What it holds.
        cap: u64,
    },
    /// A core stores more synapses than it can.
    Synapses {
        /// The offending core.
        core: u32,
        /// Synapses charged to it.
        used: u64,
        /// What it stores.
        cap: u64,
    },
    /// One neuron has more presynaptic sources than a neuron may have. **The wall.**
    FanIn {
        /// The offending neuron.
        neuron: usize,
        /// Its in-degree.
        fan_in: u64,
        /// The part's cap.
        cap: u64,
        /// The core it currently sits on, which is irrelevant to the fix and is reported anyway so
        /// the caller can find it in a placement listing.
        core: u32,
    },
}

impl CoreBind {
    /// The constraint's name, matching the strings in [`Feasibility::unchecked`].
    #[must_use]
    pub fn constraint(&self) -> &'static str {
        match self {
            Self::Neurons { .. } => "neurons per core",
            Self::Synapses { .. } => "synapses per core",
            Self::FanIn { .. } => "maximum fan-in per neuron",
        }
    }

    /// By how much, in the constraint's own units.
    ///
    /// Saturating rather than wrapping: `CoreBind` has public fields, so a caller can build one
    /// that is not a violation, and that case answers 0 rather than `1.8e19`. Every violation this
    /// module produces has an overflow of at least 1.
    #[must_use]
    pub fn overflow(&self) -> u64 {
        match *self {
            Self::Neurons { used, cap, .. } | Self::Synapses { used, cap, .. } => {
                used.saturating_sub(cap)
            }
            Self::FanIn { fan_in, cap, .. } => fan_in.saturating_sub(cap),
        }
    }

    /// Whether moving neurons between cores could relieve this.
    ///
    /// **False for [`CoreBind::FanIn`]**, and that is the point of the method: fan-in is a property
    /// of one neuron's wiring. Repartitioning moves the neuron; it does not give it fewer inputs.
    /// The fix is a tree of partial summers, which changes the network.
    #[must_use]
    pub fn relieved_by_repartitioning(&self) -> bool {
        match self {
            Self::Neurons { .. } | Self::Synapses { .. } => true,
            Self::FanIn { .. } => false,
        }
    }

    /// Reporting precedence, 0 being the most severe. Fixed by kind, not magnitude: a fan-in wall
    /// exceeded by one is a harder problem than a core overfilled by a hundred.
    #[must_use]
    pub fn precedence(&self) -> u8 {
        match self {
            Self::FanIn { .. } => 0,
            Self::Synapses { .. } => 1,
            Self::Neurons { .. } => 2,
        }
    }
}

impl fmt::Display for CoreBind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let over = self.overflow();
        match *self {
            Self::Neurons { core, used, cap } => {
                write!(f, "core {core}: {used} neurons, holds {cap} (over by {over})")
            }
            Self::Synapses { core, used, cap } => {
                write!(f, "core {core}: {used} synapses, stores {cap} (over by {over})")
            }
            Self::FanIn { neuron, fan_in, cap, core } => write!(
                f,
                "neuron {neuron} (on core {core}): {fan_in} presynaptic sources, cap is {cap} \
                 (over by {over}); repartitioning does not fix this"
            ),
        }
    }
}

/// A constraint the placement satisfies, with the worst core's margin.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CoreHeadroom {
    /// The constraint's name, matching [`CoreBind::constraint`].
    pub constraint: &'static str,
    /// The **worst** core's usage, which is the one that decides whether the placement holds.
    pub worst_used: u64,
    /// Which core that is. For the fan-in constraint, the core holding the worst neuron.
    pub worst_core: u32,
    /// What a core allows.
    pub cap: u64,
}

impl CoreHeadroom {
    /// `worst_used / cap`, in `[0, 1]` for a satisfied constraint.
    ///
    /// `None` when `cap == 0`: a core that holds nothing has no utilisation, and printing 0% or
    /// infinity for it are both numbers a reader would believe.
    #[must_use]
    pub fn utilisation(&self) -> Option<f64> {
        if self.cap == 0 {
            return None;
        }
        Some(self.worst_used as f64 / self.cap as f64)
    }

    /// What is left on the worst core, in the constraint's units.
    #[must_use]
    pub fn spare(&self) -> u64 {
        self.cap.saturating_sub(self.worst_used)
    }
}

/// Whether a placement is legal — never as a bare boolean.
///
/// Read [`Feasibility::unchecked`] before [`Feasibility::verdict`]. `Some(true)` means "no
/// constraint the limits **state** was violated", which against [`CoreLimits::UNLIMITED`] would be
/// vacuous — so that case is `None` instead, and `None` is not a pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Feasibility {
    /// `Some(false)` if anything binds, `Some(true)` if at least one constraint was checked and
    /// none bound, `None` if nothing was checkable.
    pub verdict: Option<bool>,
    /// Violations, most severe first by [`CoreBind::precedence`], then by core or neuron index.
    pub binds: Vec<CoreBind>,
    /// Constraints checked and satisfied, with the worst core's margin.
    pub headroom: Vec<CoreHeadroom>,
    /// Constraints the limits do not state, by the same names [`CoreBind::constraint`] returns.
    pub unchecked: Vec<&'static str>,
}

impl Feasibility {
    /// The most severe violation, or `None` when nothing binds.
    #[must_use]
    pub fn binding(&self) -> Option<&CoreBind> {
        self.binds.first()
    }

    /// Whether every violation would be relieved by a different assignment of the same neurons.
    ///
    /// `true` with a non-empty [`Feasibility::binds`] means "repartition"; `false` means "change
    /// the network". `false` for an empty `binds`, because there is nothing to relieve.
    #[must_use]
    pub fn repartitionable(&self) -> bool {
        !self.binds.is_empty() && self.binds.iter().all(CoreBind::relieved_by_repartitioning)
    }
}

impl fmt::Display for Feasibility {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let v = match self.verdict {
            Some(true) => "PLACES (on the limits stated)",
            Some(false) => "DOES NOT PLACE",
            None => "NO VERDICT: no limit was stated that could be checked",
        };
        writeln!(f, "{v}")?;
        for b in &self.binds {
            writeln!(f, "  BINDS  {b}")?;
        }
        for h in &self.headroom {
            match h.utilisation() {
                Some(u) => writeln!(
                    f,
                    "  ok     {}: worst core {} at {} of {} ({:.1}%)",
                    h.constraint,
                    h.worst_core,
                    h.worst_used,
                    h.cap,
                    u * 100.0
                )?,
                None => writeln!(
                    f,
                    "  ok     {}: worst core {} at {} of {} (no utilisation: the cap is zero)",
                    h.constraint, h.worst_core, h.worst_used, h.cap
                )?,
            }
        }
        for u in &self.unchecked {
            writeln!(f, "  ?      {u}: not stated")?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------------------------
// Partition
// ---------------------------------------------------------------------------------------------

/// The most cores a [`Partition`] may span: `2^24`, above any machine this crate models
/// (`SpiNNaker2`'s full build is about 10.6 million ARM cores).
///
/// What it bounds, stated rather than discovered: [`Partition::loads`] holds one **48-byte**
/// [`CoreLoad`] per core, so 805 MB at the limit; [`Partition::used_cores`] one byte per core; the
/// streaming pass three `u64`s per core. Before this constant a one-neuron network could be
/// partitioned over `u32::MAX` cores and `loads()` would attempt 206 GB while
/// [`partition_refined`] refused a working set a hundredth the size.
///
/// ⛔ The 48 is MEASURED by `core::mem::size_of`, not counted by eye. This line said 40 bytes and
/// 671 MB until `the_per_core_bookkeeping_bound_is_the_size_its_doc_prices` computed it: a
/// `CoreLoad` is `u32` + two `u64` + `Option<(usize, u64)>`, and the option has no niche to pack
/// its discriminant into, so it costs 24 bytes rather than 16 and the record costs 48 rather than
/// 40. The bound itself did not move; the number a reader would have budgeted against was 20%
/// low.
pub const MAX_CORES: u32 = 1 << 24;

/// An assignment of every neuron to a core.
///
/// Constructed against a network so that the length and the range are checked once, at the
/// boundary, rather than indexed into later.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Partition {
    /// Core of each neuron, one entry per neuron, every entry below [`Partition::n_cores`].
    pub core_of: Vec<u32>,
    /// Cores the assignment was made over. Some may be empty.
    pub n_cores: u32,
    /// Neurons the assignment covers, equal to `core_of.len()` and to the network's `n`.
    pub neurons: usize,
}

impl Partition {
    /// Check an assignment against a network and keep it.
    ///
    /// # Errors
    ///
    /// [`MapError::NoCores`] for zero cores, [`MapError::TooManyCores`] past [`MAX_CORES`],
    /// [`MapError::LengthMismatch`] when the assignment does not have one entry per neuron, and
    /// [`MapError::CoreOutOfRange`] naming the first neuron whose core index is past the end.
    pub fn new(net: &Net, core_of: Vec<u32>, n_cores: u32) -> Result<Self, MapError> {
        if n_cores == 0 {
            return Err(MapError::NoCores);
        }
        if n_cores > MAX_CORES {
            return Err(MapError::TooManyCores { n_cores, limit: MAX_CORES });
        }
        if core_of.len() != net.n {
            return Err(MapError::LengthMismatch {
                what: "core assignment",
                got: core_of.len(),
                neurons: net.n,
            });
        }
        for (i, &c) in core_of.iter().enumerate() {
            if c >= n_cores {
                return Err(MapError::CoreOutOfRange { neuron: i, core: c, n_cores });
            }
        }
        let neurons = core_of.len();
        Ok(Self { core_of, n_cores, neurons })
    }

    /// Every neuron on one core, which is the degenerate placement and the cheapest possible cut.
    ///
    /// # Errors
    ///
    /// [`MapError::NoCores`] is not reachable here; the signature is a `Result` so that it matches
    /// [`Partition::new`] and can be swapped for it in a caller.
    pub fn single_core(net: &Net) -> Result<Self, MapError> {
        Self::new(net, vec![0; net.n], 1)
    }

    /// **The objective**: synapses whose two endpoints land on different cores.
    ///
    /// Counts synapses, not neuron pairs. Two parallel synapses from the same neuron to the same
    /// target are two crossings and are counted twice, because they are two rows of synaptic memory
    /// and two deliveries. A self-synapse is never cut.
    ///
    /// # Errors
    ///
    /// [`MapError::LengthMismatch`] if `net` is not the network this partition was built against.
    pub fn cut_edges(&self, net: &Net) -> Result<u64, MapError> {
        if net.n != self.neurons {
            return Err(MapError::LengthMismatch {
                what: "core assignment",
                got: self.neurons,
                neurons: net.n,
            });
        }
        let mut cut = 0u64;
        for pre in 0..net.n {
            let (a, b) = (net.offset[pre], net.offset[pre + 1]);
            let cp = self.core_of[pre];
            for k in a..b {
                if self.core_of[net.post[k] as usize] != cp {
                    cut += 1;
                }
            }
        }
        Ok(cut)
    }

    /// Cut synapses as a fraction of all synapses.
    ///
    /// `None` for a network with no synapses, where the ratio is `0/0` — a network with nothing to
    /// cut is not a network that was cut well.
    ///
    /// # Errors
    ///
    /// As [`Partition::cut_edges`].
    pub fn cut_fraction(&self, net: &Net) -> Result<Option<f64>, MapError> {
        let cut = self.cut_edges(net)?;
        if net.n_syn == 0 {
            return Ok(None);
        }
        Ok(Some(cut as f64 / net.n_syn as f64))
    }

    /// What every core is carrying, indexed by core.
    ///
    /// # Errors
    ///
    /// As [`Partition::cut_edges`].
    pub fn loads(&self, net: &Net) -> Result<Vec<CoreLoad>, MapError> {
        if net.n != self.neurons {
            return Err(MapError::LengthMismatch {
                what: "core assignment",
                got: self.neurons,
                neurons: net.n,
            });
        }
        let deg = net.in_degrees();
        let k = self.n_cores as usize;
        let mut loads: Vec<CoreLoad> = (0..k)
            .map(|c| CoreLoad {
                core: c as u32,
                neurons: 0,
                synapses: 0,
                worst_fan_in: None,
            })
            .collect();
        for (i, &c) in self.core_of.iter().enumerate() {
            let l = &mut loads[c as usize];
            l.neurons += 1;
            let d = deg[i] as u64;
            l.synapses += d;
            match l.worst_fan_in {
                Some((_, best)) if best >= d => {}
                _ => l.worst_fan_in = Some((i, d)),
            }
        }
        Ok(loads)
    }

    /// How many cores hold at least one neuron.
    #[must_use]
    pub fn used_cores(&self) -> usize {
        let mut seen = vec![false; self.n_cores as usize];
        let mut used = 0;
        for &c in &self.core_of {
            if !seen[c as usize] {
                seen[c as usize] = true;
                used += 1;
            }
        }
        used
    }

    /// Which limit binds, on which core, and by how much.
    ///
    /// # Errors
    ///
    /// As [`Partition::cut_edges`].
    pub fn check(&self, net: &Net, limits: &CoreLimits) -> Result<Feasibility, MapError> {
        let loads = self.loads(net)?;
        let mut binds: Vec<CoreBind> = Vec::new();
        let mut headroom: Vec<CoreHeadroom> = Vec::new();
        let mut unchecked: Vec<&'static str> = Vec::new();
        if limits.max_fan_in.is_none() {
            unchecked.push("maximum fan-in per neuron");
        }
        if limits.synapses_per_core.is_none() {
            unchecked.push("synapses per core");
        }
        if limits.neurons_per_core.is_none() {
            unchecked.push("neurons per core");
        }
        if self.neurons == 0 {
            // Nothing was checked, so nothing passed. The first version fabricated a headroom
            // record on core 0 and answered `Some(true)` — the vacuous case the `Feasibility` doc
            // rules out.
            return Ok(Feasibility { verdict: None, binds, headroom, unchecked });
        }

        // --- the wall first: fan-in is a property of the network, not of this placement ---
        if let Some(cap) = limits.max_fan_in {
            let cap = u64::from(cap);
            // ⛔ EVERY NEURON, NOT EACH CORE'S WORST. The first version read one `worst_fan_in`
            // per core, so two over-cap neurons on one core produced ONE bind and the same two
            // on different cores produced two: the set of walls reported was a function of the
            // placement, for the one constraint whose doc says placement cannot relieve it —
            // and a caller who fixed neuron 20 and re-ran was then handed neuron 21.
            let deg = net.in_degrees();
            let mut worst: Option<(usize, u64, u32)> = None;
            for (neuron, &d) in deg.iter().enumerate() {
                let d = d as u64;
                let core = self.core_of[neuron];
                if worst.is_none_or(|(_, w, _)| d > w) {
                    worst = Some((neuron, d, core));
                }
                if d > cap {
                    binds.push(CoreBind::FanIn { neuron, fan_in: d, cap, core });
                }
            }
            if let (true, Some((_, w, c))) = (binds.is_empty(), worst) {
                headroom.push(CoreHeadroom {
                    constraint: "maximum fan-in per neuron",
                    worst_used: w,
                    worst_core: c,
                    cap,
                });
            }
        }

        // --- synapse storage, charged to the postsynaptic core ---
        if let Some(cap) = limits.synapses_per_core {
            let mut over = false;
            for l in &loads {
                if l.synapses > cap {
                    binds.push(CoreBind::Synapses { core: l.core, used: l.synapses, cap });
                    over = true;
                }
            }
            if !over {
                let worst = loads.iter().max_by_key(|l| (l.synapses, core::cmp::Reverse(l.core)));
                headroom.push(CoreHeadroom {
                    constraint: "synapses per core",
                    worst_used: worst.map_or(0, |l| l.synapses),
                    worst_core: worst.map_or(0, |l| l.core),
                    cap,
                });
            }
        }

        // --- neuron count ---
        if let Some(cap) = limits.neurons_per_core {
            let cap = u64::from(cap);
            let mut over = false;
            for l in &loads {
                if l.neurons > cap {
                    binds.push(CoreBind::Neurons { core: l.core, used: l.neurons, cap });
                    over = true;
                }
            }
            if !over {
                let worst = loads.iter().max_by_key(|l| (l.neurons, core::cmp::Reverse(l.core)));
                headroom.push(CoreHeadroom {
                    constraint: "neurons per core",
                    worst_used: worst.map_or(0, |l| l.neurons),
                    worst_core: worst.map_or(0, |l| l.core),
                    cap,
                });
            }
        }

        binds.sort_by_key(|b| {
            let second = match *b {
                CoreBind::Neurons { core, .. } | CoreBind::Synapses { core, .. } => u64::from(core),
                CoreBind::FanIn { neuron, .. } => neuron as u64,
            };
            (b.precedence(), second)
        });
        let verdict = if !binds.is_empty() {
            Some(false)
        } else if headroom.is_empty() {
            None
        } else {
            Some(true)
        };
        Ok(Feasibility { verdict, binds, headroom, unchecked })
    }
}

// ---------------------------------------------------------------------------------------------
// Undirected adjacency, the structure both partitioners work over
// ---------------------------------------------------------------------------------------------

/// The network as an undirected multigraph, which is what the cut metric sees.
///
/// Direction is dropped because a synapse is cut or not cut regardless of which way it points, and
/// **multiplicity is kept** because two parallel synapses are two crossings. Self-synapses are
/// dropped: they are never cut and they would otherwise appear as a neuron being its own neighbour,
/// which breaks the incremental gain bookkeeping in the refiner.
struct Adjacency {
    offset: Vec<usize>,
    nbr: Vec<u32>,
}

impl Adjacency {
    fn build(net: &Net) -> Self {
        let mut count = vec![0usize; net.n + 1];
        for pre in 0..net.n {
            for k in net.offset[pre]..net.offset[pre + 1] {
                let post = net.post[k] as usize;
                if post == pre {
                    continue;
                }
                count[pre + 1] += 1;
                count[post + 1] += 1;
            }
        }
        for i in 0..net.n {
            count[i + 1] += count[i];
        }
        let total = count[net.n];
        let offset = count.clone();
        let mut fill = offset.clone();
        let mut nbr = vec![0u32; total];
        for pre in 0..net.n {
            for k in net.offset[pre]..net.offset[pre + 1] {
                let post = net.post[k] as usize;
                if post == pre {
                    continue;
                }
                nbr[fill[pre]] = post as u32;
                fill[pre] += 1;
                nbr[fill[post]] = pre as u32;
                fill[post] += 1;
            }
        }
        for v in 0..net.n {
            nbr[offset[v]..offset[v + 1]].sort_unstable();
        }
        Self { offset, nbr }
    }

    fn of(&self, v: usize) -> &[u32] {
        &self.nbr[self.offset[v]..self.offset[v + 1]]
    }

    /// How many undirected edges join `u` and `v`. `O(log deg)` on the sorted list.
    fn multiplicity(&self, u: usize, v: u32) -> i64 {
        let s = self.of(u);
        let lo = s.partition_point(|&x| x < v);
        let hi = s.partition_point(|&x| x <= v);
        (hi - lo) as i64
    }
}

// ---------------------------------------------------------------------------------------------
// Plan and the partitioners
// ---------------------------------------------------------------------------------------------

/// Which algorithm produced a [`Plan`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Method {
    /// One streaming pass, `Fennel`-style: each neuron goes to the core holding most of its
    /// already-placed neighbours, less a penalty for that core's occupancy.
    ///
    /// Tsourakakis, Gkantsidis, Radunovic and Vojnovic, *FENNEL: Streaming Graph Partitioning for
    /// Massive Scale Graphs*, WSDM 2014. The additive penalty is what distinguishes it from the
    /// multiplicative one in `LDG` (Stanton and Kliot, KDD 2012), and the difference matters at the
    /// very start of the stream: a multiplicative penalty is zero for a neuron with no placed
    /// neighbours, so the first arrivals all pile onto one core.
    Greedy,
    /// The streaming pass, then Kernighan-Lin exchange passes over pairs of neurons on different
    /// cores.
    ///
    /// Kernighan and Lin, *An Efficient Heuristic Procedure for Partitioning Graphs*, Bell System
    /// Technical Journal 49(2):291–307, 1970.
    KernighanLin,
}

impl fmt::Display for Method {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Greedy => "streaming greedy",
            Self::KernighanLin => "streaming greedy + Kernighan-Lin",
        })
    }
}

/// A placement and what it cost to get there.
///
/// Carries the cut **before** refinement as well as after, so that the value of the refinement is a
/// number in the returned object rather than a claim in a doc comment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Plan {
    /// The assignment.
    pub partition: Partition,
    /// Which algorithm produced it.
    pub method: Method,
    /// Cut synapses, **recounted from scratch** by [`Partition::cut_edges`] on exit rather than
    /// tracked incrementally. The recount is what makes the gain bookkeeping checkable.
    pub cut: u64,
    /// Cut synapses after the streaming pass and before any refinement. Equal to `cut` for
    /// [`Method::Greedy`].
    pub cut_before_refinement: u64,
    /// Cut synapses the refinement claims to have removed, summed from the per-swap gains.
    ///
    /// Invariant, asserted by `the_refinement_gain_equals_the_measured_drop_in_cut_edges`:
    /// `cut_before_refinement - cut == refinement_gain`. The two sides are computed by completely
    /// different routes — one incremental, one a full recount — so agreement is evidence the gain
    /// formula is right rather than evidence it is self-consistent.
    pub refinement_gain: u64,
    /// Refinement passes that ran, including the final one that found no improvement and stopped.
    pub passes_run: u32,
    /// Exchanges kept, summed over passes. Exchanges that were tried and rewound are not counted.
    pub swaps_kept: u64,
}

impl fmt::Display for Plan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} on {} cores: cut {} synapses",
            self.method,
            self.partition.used_cores(),
            self.cut
        )?;
        if self.method == Method::KernighanLin {
            write!(
                f,
                " (from {} after {} pass(es), {} exchange(s) kept)",
                self.cut_before_refinement, self.passes_run, self.swaps_kept
            )?;
        }
        Ok(())
    }
}

/// The largest `neurons * cores` working set [`partition_refined`] will allocate.
///
/// 8 million cells is 64 MB of `i64`. Stated as a constant rather than buried in the check so that
/// a caller can test against it before calling.
pub const REFINEMENT_CELL_LIMIT: u64 = 8_000_000;

/// Partition `net` onto `n_cores` in one streaming pass.
///
/// The stream order is a deterministic shuffle of the neuron indices under `seed`, and `seed` is
/// the only source of variation: the same seed gives the bit-identical assignment on every
/// platform. Different seeds give genuinely different answers, which is what makes a sweep over
/// seeds a cheap way to get a better cut.
///
/// Each neuron goes to the eligible core maximising
/// `|placed neighbours there| - alpha * 1.5 * sqrt(neurons there)`, with
/// `alpha = sqrt(cores) * synapses / neurons^1.5` — the `Fennel` objective at `gamma = 1.5`
/// (Tsourakakis, Gkantsidis, Radunović and Vojnović, WSDM 2014: for `c(x) = α·x^γ` the marginal
/// cost is `δc(x) = α·γ·x^(γ-1)`, and the greedy index is `|N(v) ∩ S_i| − δc(|S_i|)`; §5.3 gives
/// `γ = 3/2, α = √k·m/n^(3/2)`). Ties go to the emptier core, then to the lower index.
///
/// ⛔ The first version shipped `0.75` here — **half** the paper's `γ = 1.5` — while hedging the
/// `alpha` expression that was the paper's own. The pinned greedy vector did not move when the
/// coefficient was doubled, because on that fixture the penalty never decided a placement;
/// `the_streaming_pass_applies_the_papers_balance_penalty` now re-implements the stream with the
/// paper's coefficient typed in the test, on a fixture where it does decide.
///
/// ⚠ The rest of this note stands. The
/// shape — `sqrt(k) * m / n^1.5`, scaling the balance penalty with the graph's density so that it
/// stays comparable to a neighbour count — is what it read, and it reproduces the behaviour the
/// paper describes on the graphs tested here. It is transcribed rather than tuned, and a reader
/// comparing against the source should check it rather than trust this line. What does not depend
/// on it: the penalty is **additive**, which is the property the whole choice rests on.
///
/// A core is **eligible** when taking this neuron would leave it inside both hard limits. Limits
/// that are `None` do not constrain.
///
/// # Errors
///
/// [`MapError::NoCores`]; [`MapError::ZeroCapacity`] for a stated limit of zero;
/// [`MapError::NotEnoughRoom`] when the machine is too small whatever the assignment;
/// [`MapError::FanInExceedsCore`] naming a neuron no placement can help; and
/// [`MapError::NoFeasibleCore`] when this single pass painted itself into a corner, which another
/// seed may avoid.
pub fn partition_greedy(
    net: &Net,
    limits: &CoreLimits,
    n_cores: u32,
    seed: u64,
) -> Result<Plan, MapError> {
    let core_of = stream_assign(net, limits, n_cores, seed)?;
    let partition = Partition::new(net, core_of, n_cores)?;
    let cut = partition.cut_edges(net)?;
    Ok(Plan {
        partition,
        method: Method::Greedy,
        cut,
        cut_before_refinement: cut,
        refinement_gain: 0,
        passes_run: 0,
        swaps_kept: 0,
    })
}

/// [`partition_greedy`], then up to `passes` Kernighan-Lin exchange passes.
///
/// # What a pass does, and why it is not hill climbing
///
/// A pass repeatedly picks the best exchange of two neurons on different cores **among the
/// candidates it considers** — including exchanges whose gain is negative — locks both, and
/// records the running gain. The candidates are the boundary neurons, which is a restriction that
/// can miss an exchange; see the note in `kl_pass` and the `# Cost` section below. At the
/// end it rewinds to the prefix with the highest cumulative gain, and keeps that prefix only if it
/// is positive. Taking negative steps and rewinding is the whole of Kernighan and Lin's idea (Bell
/// System Technical Journal 49(2), 1970) and is what lets it climb out of a local minimum that a
/// swap-while-it-improves loop sits in forever.
///
/// Exchanges preserve every core's **neuron count** exactly, so a placement that satisfied the
/// neuron limit still does. They do move synapse storage, so the synapse limit is re-checked for
/// every candidate exchange before it is considered.
///
/// # Cost
///
/// Candidates are restricted to **boundary neurons** — those with at least one edge leaving their
/// core. That restriction buys the cost below and costs quality: an interior neuron cannot gain
/// from moving but can be a better partner than any boundary neuron, and this pass will not find
/// that exchange. A pass is `O(steps * (B^2 * log d + n * k))` where `B` is the boundary size, `d` the mean
/// degree and `n * k` the cost of rebuilding the boundary set from scratch at every step — a term
/// the first version of this line left out, and which is a 10x factor on the fixtures here — with
/// `steps` at most `neurons / 2`. This is the slow, readable version; Fiduccia and Mattheyses
/// (DAC 1982) give the linear-time bucket structure that makes it practical at scale, and it is not
/// implemented here.
///
/// # Errors
///
/// As [`partition_greedy`], plus [`MapError::RefinementTooLarge`] when `neurons * n_cores` exceeds
/// [`REFINEMENT_CELL_LIMIT`].
pub fn partition_refined(
    net: &Net,
    limits: &CoreLimits,
    n_cores: u32,
    seed: u64,
    passes: u32,
) -> Result<Plan, MapError> {
    let cells = net.n as u64 * u64::from(n_cores);
    if cells > REFINEMENT_CELL_LIMIT {
        return Err(MapError::RefinementTooLarge {
            neurons: net.n,
            n_cores,
            cells,
            limit: REFINEMENT_CELL_LIMIT,
        });
    }
    let base = partition_greedy(net, limits, n_cores, seed)?;
    let cut_before_refinement = base.cut;

    let adj = Adjacency::build(net);
    let in_deg: Vec<i64> = net.in_degrees().iter().map(|&d| d as i64).collect();
    let syn_cap = limits.synapses_per_core.map(|c| c as i64);
    let k = n_cores as usize;

    let mut core_of: Vec<u32> = base.partition.core_of.clone();
    let mut cnt = vec![0i64; net.n * k];
    for v in 0..net.n {
        for &u in adj.of(v) {
            cnt[v * k + core_of[u as usize] as usize] += 1;
        }
    }
    let mut syn = vec![0i64; k];
    for v in 0..net.n {
        syn[core_of[v] as usize] += in_deg[v];
    }

    let mut total_gain = 0i64;
    let mut swaps_kept = 0u64;
    let mut passes_run = 0u32;
    for _ in 0..passes {
        passes_run += 1;
        let (gain, kept) =
            kl_pass(&adj, &in_deg, syn_cap, k, &mut core_of, &mut cnt, &mut syn);
        total_gain += gain;
        swaps_kept += kept;
        if gain <= 0 {
            break;
        }
    }

    let partition = Partition::new(net, core_of, n_cores)?;
    let cut = partition.cut_edges(net)?;
    Ok(Plan {
        partition,
        method: Method::KernighanLin,
        cut,
        cut_before_refinement,
        refinement_gain: total_gain.max(0) as u64,
        passes_run,
        swaps_kept,
    })
}

/// The streaming assignment. Separated from [`partition_greedy`] so the refiner can reuse it
/// without building a [`Plan`] it would immediately discard.
fn stream_assign(
    net: &Net,
    limits: &CoreLimits,
    n_cores: u32,
    seed: u64,
) -> Result<Vec<u32>, MapError> {
    if n_cores == 0 {
        return Err(MapError::NoCores);
    }
    if limits.neurons_per_core == Some(0) {
        return Err(MapError::ZeroCapacity { which: "neurons per core" });
    }
    if limits.synapses_per_core == Some(0) {
        return Err(MapError::ZeroCapacity { which: "synapses per core" });
    }
    if n_cores > MAX_CORES {
        return Err(MapError::TooManyCores { n_cores, limit: MAX_CORES });
    }
    if net.n == 0 {
        return Ok(Vec::new());
    }
    if let Some(npc) = limits.neurons_per_core {
        let capacity = u64::from(npc) * u64::from(n_cores);
        if (net.n as u64) > capacity {
            return Err(MapError::NotEnoughRoom {
                neurons: net.n,
                n_cores,
                neurons_per_core: npc,
                capacity,
            });
        }
    }
    let deg: Vec<u64> = net.in_degrees().iter().map(|&d| d as u64).collect();
    if let Some(cap) = limits.synapses_per_core {
        // Whole-machine arithmetic first, for the same reason `NotEnoughRoom` comes first: a
        // network with more synapses than the machine stores does not map by being packed better.
        let synapses: u64 = deg.iter().sum();
        let capacity = cap.saturating_mul(u64::from(n_cores));
        if synapses > capacity {
            return Err(MapError::NotEnoughSynapseRoom {
                synapses,
                n_cores,
                synapses_per_core: cap,
                capacity,
            });
        }
        for (i, &d) in deg.iter().enumerate() {
            if d > cap {
                return Err(MapError::FanInExceedsCore { neuron: i, fan_in: d, cap });
            }
        }
    }

    let adj = Adjacency::build(net);
    let k = n_cores as usize;

    // Fennel's alpha at gamma = 1.5: sqrt(k) * m / n^1.5, with m the undirected edge count.
    let n_f = net.n as f64;
    let alpha = (k as f64).sqrt() * net.n_syn as f64 / (n_f * n_f.sqrt());

    let mut order: Vec<usize> = (0..net.n).collect();
    let mut rng = Rng::new(seed);
    // Fisher-Yates, downward, which is the standard unbiased shuffle and is deterministic here
    // because `Rng` is.
    for i in (1..order.len()).rev() {
        let j = rng.below((i + 1) as u32) as usize;
        order.swap(i, j);
    }

    let mut core_of = vec![u32::MAX; net.n];
    let mut neurons = vec![0u64; k];
    let mut syn = vec![0u64; k];
    let mut score = vec![0i64; k];
    let mut touched: Vec<usize> = Vec::new();

    for &v in &order {
        for &c in &touched {
            score[c] = 0;
        }
        touched.clear();
        for &u in adj.of(v) {
            let c = core_of[u as usize];
            if c != u32::MAX {
                let ci = c as usize;
                if score[ci] == 0 {
                    touched.push(ci);
                }
                score[ci] += 1;
            }
        }
        let d = deg[v];
        let mut best: Option<(usize, f64)> = None;
        for c in 0..k {
            if limits.neurons_per_core.is_some_and(|npc| neurons[c] >= u64::from(npc)) {
                continue;
            }
            if limits.synapses_per_core.is_some_and(|cap| syn[c] + d > cap) {
                continue;
            }
            let s = score[c] as f64 - alpha * 1.5 * (neurons[c] as f64).sqrt();
            match best {
                None => best = Some((c, s)),
                Some((bc, bs)) => {
                    // Exact equality is meaningful: two cores with the same neighbour count and the
                    // same occupancy produce the same `f64` by the same arithmetic. The tie goes to
                    // the emptier core, then to the lower index, which is what keeps a zero-signal
                    // stream prefix from piling onto core 0.
                    let better = s > bs
                        || (s == bs
                            && (neurons[c], c) < (neurons[bc], bc));
                    if better {
                        best = Some((c, s));
                    }
                }
            }
        }
        let Some((c, _)) = best else {
            return Err(MapError::NoFeasibleCore { neuron: v, in_degree: d, n_cores });
        };
        core_of[v] = c as u32;
        neurons[c] += 1;
        syn[c] += d;
    }
    Ok(core_of)
}

/// One Kernighan-Lin pass. Returns `(gain kept, exchanges kept)` and leaves `core_of`, `cnt` and
/// `syn` at the best prefix.
fn kl_pass(
    adj: &Adjacency,
    in_deg: &[i64],
    syn_cap: Option<i64>,
    k: usize,
    core_of: &mut [u32],
    cnt: &mut [i64],
    syn: &mut [i64],
) -> (i64, u64) {
    let n = core_of.len();
    let mut locked = vec![false; n];
    let mut applied: Vec<(usize, usize, i64)> = Vec::new();
    let max_steps = n / 2;

    for _ in 0..max_steps {
        // Boundary neurons only: a neuron with every edge inside its own core has nothing to gain
        // from moving — its half of any exchange is `-(the edges it has at home)`, which is never
        // positive.
        //
        // ⛔ It can still be the better PARTNER for a neuron that gains a lot, so this is a COST
        // measure and not a proof that nothing is lost. The smallest case is in
        // `the_exchange_candidates_are_the_boundary_neurons_only_and_this_is_what_that_costs`:
        // six neurons over two cores where every boundary pair loses and the interior leaf would
        // have gained 1, cutting 2 synapses where this rule stops at 3. Left as it is because the
        // restriction is what keeps a pass at `O(B^2 log d)` rather than `O(n^2 log d)`, and
        // because `kernighan_lin_does_not_always_reach_the_optimum_and_this_is_what_it_misses`
        // already states that this refiner is a heuristic.
        let boundary: Vec<usize> = (0..n)
            .filter(|&v| {
                if locked[v] {
                    return false;
                }
                let own = core_of[v] as usize;
                (0..k).any(|c| c != own && cnt[v * k + c] > 0)
            })
            .collect();
        let mut best: Option<(usize, usize, i64)> = None;
        for (ai, &u) in boundary.iter().enumerate() {
            let cu = core_of[u] as usize;
            for &v in &boundary[ai + 1..] {
                let cv = core_of[v] as usize;
                if cu == cv {
                    continue;
                }
                if let Some(cap) = syn_cap {
                    let after_u = syn[cu] - in_deg[u] + in_deg[v];
                    let after_v = syn[cv] - in_deg[v] + in_deg[u];
                    if after_u > cap || after_v > cap {
                        continue;
                    }
                }
                let d_u = cnt[u * k + cv] - cnt[u * k + cu];
                let d_v = cnt[v * k + cu] - cnt[v * k + cv];
                let g = d_u + d_v - 2 * adj.multiplicity(u, v as u32);
                if best.is_none_or(|(bu, bv, bg)| g > bg || (g == bg && (u, v) < (bu, bv))) {
                    best = Some((u, v, g));
                }
            }
        }
        let Some((u, v, g)) = best else { break };
        kl_exchange(adj, in_deg, k, core_of, cnt, syn, u, v);
        locked[u] = true;
        locked[v] = true;
        applied.push((u, v, g));
    }

    let mut cum = 0i64;
    let mut best_cum = 0i64;
    let mut best_k = 0usize;
    for (i, &(_, _, g)) in applied.iter().enumerate() {
        cum += g;
        if cum > best_cum {
            best_cum = cum;
            best_k = i + 1;
        }
    }
    for i in (best_k..applied.len()).rev() {
        let (u, v, _) = applied[i];
        kl_exchange(adj, in_deg, k, core_of, cnt, syn, u, v);
    }
    (best_cum, best_k as u64)
}

/// Exchange the cores of `u` and `v`, updating every incremental structure. Its own inverse.
fn kl_exchange(
    adj: &Adjacency,
    in_deg: &[i64],
    k: usize,
    core_of: &mut [u32],
    cnt: &mut [i64],
    syn: &mut [i64],
    u: usize,
    v: usize,
) {
    let cu = core_of[u] as usize;
    let cv = core_of[v] as usize;
    for &x in adj.of(u) {
        let xi = x as usize;
        cnt[xi * k + cu] -= 1;
        cnt[xi * k + cv] += 1;
    }
    for &x in adj.of(v) {
        let xi = x as usize;
        cnt[xi * k + cv] -= 1;
        cnt[xi * k + cu] += 1;
    }
    core_of[u] = cv as u32;
    core_of[v] = cu as u32;
    syn[cu] += in_deg[v] - in_deg[u];
    syn[cv] += in_deg[u] - in_deg[v];
}

// ---------------------------------------------------------------------------------------------
// Workload: spike-hops
// ---------------------------------------------------------------------------------------------

/// What a workload cost the fabric, in exact integer counts.
///
/// Nothing here is a joule. Joules come from [`SpikeHops::bill`] and are refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpikeHops {
    /// Link traversals with multicast: one packet enters the fabric per spike and the routers fork
    /// it. The sum over spikes of the source's multicast tree size.
    pub multicast_hops: u64,
    /// Link traversals without multicast: one packet per destination **core** per spike.
    ///
    /// The comparison, not a cost this crate recommends. See [`McastTree::unicast_hops`] on why the
    /// baseline is per core rather than per neuron.
    pub unicast_hops: u64,
    /// Multicast link traversals that crossed a chip boundary, or `None` when the caller did not
    /// state a chip size.
    ///
    /// `None` rather than zero. A workload whose chip boundaries are unknown has an unknown number
    /// of crossings, and zero is the answer that flatters it.
    pub chip_crossings: Option<u64>,
    /// Spike deliveries to a synapse whose postsynaptic neuron is on the source's own core: no
    /// packet, no hop, one local memory read.
    pub on_core_deliveries: u64,
    /// Spike deliveries that left the source core.
    pub off_core_deliveries: u64,
    /// Spikes emitted, summed over neurons.
    pub spikes: u64,
    /// Neurons that emitted at least one spike and have at least one target.
    pub sources: u64,
}

impl SpikeHops {
    /// Total deliveries, on-core and off.
    ///
    /// Equal to the [`crate::ledger::Ledger::syn_ops`] of the same run, because both count the same
    /// event: one spike arriving at one synapse.
    #[must_use]
    pub fn deliveries(&self) -> u64 {
        self.on_core_deliveries + self.off_core_deliveries
    }

    /// Link traversals per emitted spike.
    ///
    /// `None` when nothing fired. This is the number placement moves: the same network on the same
    /// fabric, mapped twice, differs here and nowhere else in the ledger.
    #[must_use]
    pub fn hops_per_spike(&self) -> Option<f64> {
        if self.spikes == 0 {
            return None;
        }
        Some(self.multicast_hops as f64 / self.spikes as f64)
    }

    /// Link traversals per synaptic operation, against a ledger from the same run.
    ///
    /// `None` when the ledger recorded no synaptic operations. The two objects are counted
    /// independently — the ledger by the simulator, this by the mapper — so a caller comparing
    /// [`SpikeHops::deliveries`] against [`crate::ledger::Ledger::syn_ops`] is checking one against
    /// the other rather than against itself.
    #[must_use]
    pub fn hops_per_synaptic_operation(&self, ledger: &Ledger) -> Option<f64> {
        if ledger.syn_ops == 0 {
            return None;
        }
        Some(self.multicast_hops as f64 / ledger.syn_ops as f64)
    }

    /// Fraction of the unicast cost multicast saved, in `[0, 1)`. `None` when nothing was sent.
    #[must_use]
    pub fn multicast_saving(&self) -> Option<f64> {
        if self.unicast_hops == 0 {
            return None;
        }
        Some(1.0 - self.multicast_hops as f64 / self.unicast_hops as f64)
    }

    /// Price the fabric traffic, or refuse and say which term had no price.
    ///
    /// A term with a zero count is not charged and its missing price is not held against the total,
    /// exactly as in [`crate::ledger::Ledger::bill`]. A workload with no chip crossings does not
    /// need a crossing price; a workload whose chip size was not stated **does**, and gets
    /// `"chip-boundary split"` in [`HopBill::unpriced`].
    #[must_use]
    pub fn bill(&self, p: &HopPrices) -> HopBill {
        let mut unpriced: Vec<&'static str> = Vec::new();
        let mut any = false;
        let mut charge = |count: u64, price: Option<f64>, name: &'static str| -> Option<f64> {
            if count == 0 {
                return Some(0.0);
            }
            match price {
                Some(e) if e.is_finite() => {
                    any = true;
                    Some(count as f64 * e)
                }
                _ => {
                    unpriced.push(name);
                    None
                }
            }
        };
        let (on_chip_hops, crossing_hops, split_known) = match self.chip_crossings {
            Some(x) => (self.multicast_hops.saturating_sub(x), x, true),
            None => (0, 0, false),
        };
        let on_chip = charge(on_chip_hops, p.e_hop_on_chip, "on-chip hop");
        let crossings = charge(crossing_hops, p.e_hop_chip_crossing, "chip-crossing hop");
        let injection = charge(self.spikes, p.e_packet_inject, "packet injection");
        if !split_known && self.multicast_hops > 0 {
            unpriced.push("chip-boundary split (cores per chip not stated)");
        }
        // ⛔ `split_known` gates the total only when there were hops to split. The first version
        // refused a total for the best-placed workload the module exists to celebrate — everything
        // on-core, nothing on the fabric — with `unpriced` EMPTY: a refusal with no reason, which
        // contradicted `HopBill::unpriced`'s doc and `Ledger::bill`'s rule that a zero count needs
        // no price.
        let total = match (on_chip, crossings, injection) {
            (Some(a), Some(b), Some(c)) if split_known || self.multicast_hops == 0 => Some(a + b + c),
            _ => None,
        };
        let evidence = if any { p.evidence } else { Evidence::Unstated };
        HopBill { total, on_chip, crossings, injection, unpriced, evidence }
    }
}

/// Count what a workload costs the fabric under a placement.
///
/// `spikes_per_neuron[v]` is how many times neuron `v` fired. Take it from a
/// [`crate::spike::Train`] by counting sources, or from an analytic rate; this function does not
/// care where it came from, only that it has one entry per neuron.
///
/// `cores_per_chip` is `None` when the caller does not know the chip size, and
/// [`SpikeHops::chip_crossings`] is then `None` rather than zero.
///
/// # Cost
///
/// One multicast tree is built per firing neuron, each `O(targets * diameter)`. No tree is cached
/// between neurons, so a network where every neuron has the same target set pays for it repeatedly.
/// Stated here rather than discovered.
///
/// # Errors
///
/// [`MapError::LengthMismatch`] when `spikes_per_neuron` or the partition does not have one entry
/// per neuron; [`MapError::CoreNotOnFabric`] when the partition names a core the fabric does not
/// have; [`MapError::ZeroCapacity`] for `Some(0)` cores per chip; and whatever
/// [`multicast_tree`] refuses.
pub fn spike_hops(
    net: &Net,
    partition: &Partition,
    fabric: &Fabric,
    cores_per_chip: Option<u32>,
    spikes_per_neuron: &[u64],
) -> Result<SpikeHops, MapError> {
    if partition.neurons != net.n {
        return Err(MapError::LengthMismatch {
            what: "core assignment",
            got: partition.neurons,
            neurons: net.n,
        });
    }
    if spikes_per_neuron.len() != net.n {
        return Err(MapError::LengthMismatch {
            what: "spike count",
            got: spikes_per_neuron.len(),
            neurons: net.n,
        });
    }
    if cores_per_chip == Some(0) {
        return Err(MapError::ZeroCapacity { which: "cores per chip" });
    }
    let n_fabric = fabric.n_cores();
    if u64::from(partition.n_cores) > n_fabric {
        return Err(MapError::CoreNotOnFabric {
            core: partition.n_cores - 1,
            n_cores: n_fabric,
        });
    }

    let mut out = SpikeHops {
        multicast_hops: 0,
        unicast_hops: 0,
        chip_crossings: cores_per_chip.map(|_| 0),
        on_core_deliveries: 0,
        off_core_deliveries: 0,
        spikes: 0,
        sources: 0,
    };
    let mut dests: Vec<u32> = Vec::new();
    for v in 0..net.n {
        let s = spikes_per_neuron[v];
        if s == 0 {
            continue;
        }
        let src = partition.core_of[v];
        let (a, b) = (net.offset[v], net.offset[v + 1]);
        let overflow = |what: &'static str| MapError::CountOverflow { neuron: v, what };
        if a == b {
            // Fired with no targets: it still emitted, and the ledger counts that, but it costs the
            // fabric nothing.
            out.spikes = out.spikes.checked_add(s).ok_or_else(|| overflow("spikes"))?;
            continue;
        }
        dests.clear();
        let mut on_core = 0u64;
        let mut off_core = 0u64;
        for kk in a..b {
            let c = partition.core_of[net.post[kk] as usize];
            if c == src {
                on_core += 1;
            } else {
                off_core += 1;
            }
            dests.push(c);
        }
        let tree = multicast_tree(fabric, src, &dests)?;
        // Checked, because `s` is caller-supplied and `u64::MAX` spikes from one neuron used to
        // wrap silently in release: `multicast_hops: 18446744073709551614` with `Ok` around it.
        let acc = |cur: u64, x: u64, what: &'static str| -> Result<u64, MapError> {
            s.checked_mul(x).and_then(|p| cur.checked_add(p)).ok_or_else(|| overflow(what))
        };
        out.multicast_hops = acc(out.multicast_hops, tree.hops, "multicast hops")?;
        out.unicast_hops = acc(out.unicast_hops, tree.unicast_hops, "unicast hops")?;
        if let (Some(total), Some(x)) =
            (out.chip_crossings, cores_per_chip.and_then(|c| tree.crossings(c)))
        {
            out.chip_crossings = Some(acc(total, x, "chip crossings")?);
        }
        out.on_core_deliveries = acc(out.on_core_deliveries, on_core, "on-core deliveries")?;
        out.off_core_deliveries = acc(out.off_core_deliveries, off_core, "off-core deliveries")?;
        out.spikes = out.spikes.checked_add(s).ok_or_else(|| overflow("spikes"))?;
        out.sources += 1;
    }
    Ok(out)
}

// ---------------------------------------------------------------------------------------------
// Prices
// ---------------------------------------------------------------------------------------------

/// Per-hop energies for one fabric, in joules. `None` means **nobody has published this number**.
///
/// Every table in this module is entirely `None`, and that is the finding rather than an unfinished
/// implementation: *this review did not locate a published per-hop, per-link or per-router-traversal
/// energy for any commercially available neuromorphic part.* The figures the field does publish are
/// per synaptic operation ([`crate::ledger`]) and whole-chip averages, and neither separates the
/// fabric from the arithmetic.
///
/// Supply your own, measured, at a stated boundary, and [`SpikeHops::bill`] will price your
/// workload. The hop counts themselves are exact and useful without any of this.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HopPrices {
    /// One link traversal between two cores on the same die.
    pub e_hop_on_chip: Option<f64>,
    /// One link traversal that leaves the die.
    ///
    /// Separate from the on-chip term because the two scale with different physics: an on-die link
    /// is a short wire and a repeater, an off-die link is a pad, a package trace and a `SerDes`.
    /// Folding them into one number is the same error [`crate::ledger::Prices`] refuses to make
    /// with the synapse fetch.
    pub e_hop_chip_crossing: Option<f64>,
    /// Injecting one packet into the fabric at its source core, before it travels anywhere.
    pub e_packet_inject: Option<f64>,
    /// WHAT these numbers describe and where they came from — or, when they are all `None`, what
    /// was looked for and not found. Never empty.
    pub source: &'static str,
    /// How good [`HopPrices::source`] is.
    pub evidence: Evidence,
}

impl HopPrices {
    /// Whether every term has a number. False for every table in this module.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.e_hop_on_chip.is_some()
            && self.e_hop_chip_crossing.is_some()
            && self.e_packet_inject.is_some()
    }

    /// The terms this table cannot price, in the order a bill lists them.
    #[must_use]
    pub fn unpriced(&self) -> Vec<&'static str> {
        let mut v = Vec::new();
        if self.e_hop_on_chip.is_none() {
            v.push("on-chip hop");
        }
        if self.e_hop_chip_crossing.is_none() {
            v.push("chip-crossing hop");
        }
        if self.e_packet_inject.is_none() {
            v.push("packet injection");
        }
        v
    }
}

/// A fabric nobody has characterised: every term `None`. The default, deliberately.
pub const UNSTATED_FABRIC: HopPrices = HopPrices {
    e_hop_on_chip: None,
    e_hop_chip_crossing: None,
    e_packet_inject: None,
    source: "no fabric. Nothing here has been priced; ask a fabric model for its numbers.",
    evidence: Evidence::Unstated,
};

/// `SpiNNaker`'s fabric, unpriced, with what was looked for recorded.
pub const SPINNAKER_FABRIC: HopPrices = HopPrices {
    e_hop_on_chip: None,
    e_hop_chip_crossing: None,
    e_packet_inject: None,
    source: "SpiNNaker: this review did not locate a per-link or per-router-traversal energy in \
             Furber, Galluppi, Temple and Plana, Proceedings of the IEEE 102(5):652-665, 2014, or \
             in the SpiNNaker chip papers it read. Whole-board power figures exist and do not \
             separate the router from the ARM cores, which is the separation this field needs.",
    evidence: Evidence::Unstated,
};

/// `Loihi`'s mesh, unpriced, with what was looked for recorded.
pub const LOIHI_FABRIC: HopPrices = HopPrices {
    e_hop_on_chip: None,
    e_hop_chip_crossing: None,
    e_packet_inject: None,
    source: "Loihi: Davies et al., IEEE Micro 38(1):82-99, 2018, prices a synaptic operation \
             (pre-silicon — see crate::ledger::LOIHI_2018) and does not, in what this review read, \
             price a mesh hop separately. The distinction between an on-die hop and a chip crossing \
             is architecturally explicit in that part and energetically unpublished.",
    evidence: Evidence::Unstated,
};

/// Every fabric price table in this crate, for a caller that wants to sweep them.
///
/// Three entries and **not one priced term among them**. That is the state of the field as this
/// review found it: the per-operation energy is published and the per-hop energy is not, on parts
/// whose whole design argument is that the hop is what you should be minimising.
pub const FABRIC_CATALOGUE: [(&str, HopPrices); 3] = [
    ("unstated", UNSTATED_FABRIC),
    ("spinnaker", SPINNAKER_FABRIC),
    ("loihi", LOIHI_FABRIC),
];

/// A priced set of hop counts, with the terms that could not be priced named.
#[derive(Debug, Clone, PartialEq)]
pub struct HopBill {
    /// Joules for the fabric, or `None` if any charged term had no price.
    pub total: Option<f64>,
    /// Joules attributable to on-die link traversals.
    pub on_chip: Option<f64>,
    /// Joules attributable to link traversals that left the die.
    pub crossings: Option<f64>,
    /// Joules attributable to injecting packets at their source cores.
    pub injection: Option<f64>,
    /// Names of the terms that had work to price and no price for it.
    pub unpriced: Vec<&'static str>,
    /// The grade of the prices actually used, [`Evidence::Unstated`] when none were.
    pub evidence: Evidence,
}

impl fmt::Display for HopBill {
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

#[cfg(test)]
mod tests {
    use super::{
        CoreBind, CoreLimits, FABRIC_CATALOGUE, Fabric, HopPrices, LOIHI_FABRIC, MapError, Method,
        Partition, SPINNAKER_FABRIC, SpikeHops, UNSTATED_FABRIC, multicast_tree, partition_greedy,
        partition_refined, spike_hops,
    };
    use crate::hardware::{DYNAP_SE, SPINNAKER, TRUENORTH};
    use crate::ledger::{Evidence, Ledger};
    use crate::net::{Net, NetBuilder};
    use std::collections::VecDeque;

    /// The fourteen fabrics every structural test sweeps, including the degenerate cases: the
    /// two-wide torus where a wrapped axis folds onto itself, the **one-wide** torus where a
    /// wrapped axis folds onto the core it started from, and the odd-by-odd torus where halving
    /// each side and halving the sum of the sides are different numbers.
    ///
    /// The last three arrived with this audit. Every fabric here before had at least two cores on
    /// every wrapped axis, so no direction ever mapped a core to itself and
    /// `fabric_neighbours_are_sorted_distinct_and_exclude_the_core_itself` could not see the
    /// filter that removes it; and every torus here had at least one even side, where
    /// `cols/2 + rows/2` and `(cols + rows)/2` agree.
    fn fabrics() -> Vec<Fabric> {
        vec![
            Fabric::Crossbar { cores: 7 },
            Fabric::Crossbar { cores: 1 },
            Fabric::Mesh2D { cols: 5, rows: 4 },
            Fabric::Mesh2D { cols: 1, rows: 6 },
            Fabric::Torus2D { cols: 5, rows: 4 },
            Fabric::Torus2D { cols: 8, rows: 8 },
            Fabric::Torus2D { cols: 2, rows: 3 },
            Fabric::Torus2D { cols: 5, rows: 3 },
            Fabric::Torus2D { cols: 1, rows: 5 },
            Fabric::TriangularTorus { cols: 5, rows: 4 },
            Fabric::TriangularTorus { cols: 8, rows: 8 },
            Fabric::TriangularTorus { cols: 3, rows: 7 },
            Fabric::TriangularTorus { cols: 2, rows: 2 },
            Fabric::TriangularTorus { cols: 4, rows: 1 },
        ]
    }

    /// Breadth-first search over [`Fabric::neighbours`]: the ground truth every closed-form hop
    /// count is checked against. It knows nothing about the formulas.
    fn bfs(f: &Fabric, src: u32) -> Vec<Option<u64>> {
        let n = f.n_cores() as usize;
        let mut d: Vec<Option<u64>> = vec![None; n];
        d[src as usize] = Some(0);
        let mut q = VecDeque::new();
        q.push_back(src);
        while let Some(c) = q.pop_front() {
            let dc = d[c as usize].unwrap_or(0);
            for nb in f.neighbours(c) {
                if d[nb as usize].is_none() {
                    d[nb as usize] = Some(dc + 1);
                    q.push_back(nb);
                }
            }
        }
        d
    }

    fn chain(n: u32) -> Net {
        let mut b = NetBuilder::new(n as usize);
        for i in 0..n.saturating_sub(1) {
            b.connect(i, i + 1, 1e-3, 1).unwrap();
        }
        b.build()
    }

    /// Two cliques of `m`, joined by exactly one synapse. The minimum balanced cut is **1**: the
    /// graph is connected so 0 is impossible, and deleting the bridge leaves two components of
    /// exactly `m`, so 1 is achieved by a partition that respects a cap of `m`.
    fn two_cliques(m: u32) -> Net {
        let mut b = NetBuilder::new(2 * m as usize);
        for a in 0..m {
            for c in (a + 1)..m {
                b.connect(a, c, 1e-3, 1).unwrap();
                b.connect(m + a, m + c, 1e-3, 1).unwrap();
            }
        }
        b.connect(0, m, 1e-3, 1).unwrap();
        b.build()
    }

    fn cycle(n: u32) -> Net {
        let mut b = NetBuilder::new(n as usize);
        for i in 0..n {
            b.connect(i, (i + 1) % n, 1e-3, 1).unwrap();
        }
        b.build()
    }

    // ----------------------------------------------------------------------------------------
    // Fabric geometry
    // ----------------------------------------------------------------------------------------

    /// Verification (d): the hop count on a mesh is the Manhattan distance, exactly, for every
    /// ordered pair — plus three values worked by hand so that a mutation replacing the formula
    /// with itself would still have to reproduce them.
    #[test]
    fn manhattan_distance_is_the_hop_count_on_a_mesh() {
        let f = Fabric::Mesh2D { cols: 5, rows: 4 };
        for a in 0..20u32 {
            for b in 0..20u32 {
                let (ax, ay) = (a % 5, a / 5);
                let (bx, by) = (b % 5, b / 5);
                let want = u64::from(ax.abs_diff(bx)) + u64::from(ay.abs_diff(by));
                assert_eq!(f.hops(a, b), Some(want), "{a} -> {b}");
            }
        }
        // Core 0 is (0,0), core 19 is (4,3), core 4 is (4,0), core 15 is (0,3).
        assert_eq!(f.hops(0, 19), Some(7));
        assert_eq!(f.hops(4, 15), Some(7));
        assert_eq!(f.hops(0, 6), Some(2));
    }

    /// Every closed form in [`Fabric`], against breadth-first search, exhaustively over all ordered
    /// pairs of eleven fabrics. The formula is the claim; the search is the answer.
    #[test]
    fn every_fabric_hop_count_agrees_with_breadth_first_search() {
        for f in fabrics() {
            let n = f.n_cores() as u32;
            for a in 0..n {
                let d = bfs(&f, a);
                for b in 0..n {
                    assert_eq!(f.hops(a, b), d[b as usize], "{f:?}: {a} -> {b}");
                }
            }
        }
    }

    /// A route is a real walk: one core per hop, every consecutive pair an actual link. This is the
    /// test that caught a triangular-torus route taking the diagonal link when the two
    /// displacements disagreed in sign — it arrived in two moves where the distance was three, and
    /// neither move was a link.
    #[test]
    fn a_route_has_exactly_as_many_steps_as_the_hop_count_and_uses_only_links() {
        for f in fabrics() {
            let n = f.n_cores() as u32;
            for a in 0..n {
                let nbrs: Vec<Vec<u32>> = (0..n).map(|c| f.neighbours(c)).collect();
                for b in 0..n {
                    let h = f.hops(a, b).unwrap();
                    let r = f.route(a, b).unwrap();
                    assert_eq!(r.len() as u64, h + 1, "{f:?}: {a} -> {b} took {r:?}");
                    assert_eq!(r.first(), Some(&a));
                    assert_eq!(r.last(), Some(&b));
                    for w in r.windows(2) {
                        assert!(
                            nbrs[w[0] as usize].contains(&w[1]),
                            "{f:?}: {a} -> {b} stepped {} -> {}, which is not a link",
                            w[0],
                            w[1]
                        );
                    }
                }
            }
        }
    }

    /// The property [`multicast_tree`] rests on: the fabric's routing is prefix-closed, so the
    /// union of the routes from one root is a tree. Checked for every root of every fabric against
    /// the hardest destination set there is — all of them.
    #[test]
    fn the_union_of_routes_from_one_source_is_a_tree_on_every_fabric() {
        for f in fabrics() {
            let n = f.n_cores() as u32;
            let all: Vec<u32> = (0..n).collect();
            for root in 0..n {
                let t = multicast_tree(&f, root, &all).expect("routes must union into a tree");
                assert_eq!(t.cores.len(), n as usize, "{f:?} root {root}: tree missed a core");
                // Acyclic, checked with union-find over the tree's own edge list. The count
                // identity `edges + 1 == cores` that stood here is what `multicast_tree` itself
                // enforces before returning, so asserting it could not fail; a cycle check on the
                // edges is independent of the function's own guard.
                let mut parent: Vec<usize> = (0..n as usize).collect();
                fn find(p: &mut [usize], mut x: usize) -> usize {
                    while p[x] != x {
                        p[x] = p[p[x]];
                        x = p[x];
                    }
                    x
                }
                for &(a, b) in &t.edges {
                    let (ra, rb) = (find(&mut parent, a as usize), find(&mut parent, b as usize));
                    assert_ne!(ra, rb, "{f:?} root {root}: link ({a},{b}) closes a cycle");
                    parent[ra] = rb;
                }
                assert_eq!(t.hops, t.edges.len() as u64);
            }
        }
    }

    /// The diameter closed forms, against the worst hop actually measured.
    #[test]
    fn the_diameter_closed_forms_match_the_worst_measured_hop() {
        for f in fabrics() {
            let n = f.n_cores() as u32;
            let mut worst = 0u64;
            for a in 0..n {
                for b in 0..n {
                    worst = worst.max(f.hops(a, b).unwrap());
                }
            }
            assert_eq!(f.diameter(), Some(worst), "{f:?}");
        }
        assert_eq!(Fabric::Mesh2D { cols: 8, rows: 8 }.diameter(), Some(14));
        assert_eq!(Fabric::Torus2D { cols: 8, rows: 8 }.diameter(), Some(8));
        assert_eq!(Fabric::Mesh2D { cols: 0, rows: 4 }.diameter(), None);
        // ⛔ A torus diameter is `cols/2 + rows/2` — each side halved, not the sum of the sides
        // halved. The two agree whenever either side is even, and every torus this sweep had
        // before had one. On 5x3 they are 2 + 1 = 3 against 8/2 = 4, and on 7x7 they are 6
        // against 7; the sweep above measures both.
        assert_eq!(Fabric::Torus2D { cols: 5, rows: 3 }.diameter(), Some(3));
        assert_eq!(Fabric::Torus2D { cols: 7, rows: 7 }.diameter(), Some(6));
        assert_eq!(Fabric::Torus2D { cols: 5, rows: 5 }.diameter(), Some(4));
    }

    /// Wrapping an axis can only shorten a route, and on an eight-wide fabric it shortens the worst
    /// one from seven hops to one. Both halves matter: the inequality alone would pass for a torus
    /// implemented as a mesh.
    #[test]
    fn a_torus_is_never_longer_than_a_mesh_and_is_strictly_shorter_where_it_wraps() {
        let m = Fabric::Mesh2D { cols: 8, rows: 8 };
        let t = Fabric::Torus2D { cols: 8, rows: 8 };
        let mut strictly_shorter = 0;
        for a in 0..64u32 {
            for b in 0..64u32 {
                let (hm, ht) = (m.hops(a, b).unwrap(), t.hops(a, b).unwrap());
                assert!(ht <= hm, "torus {a}->{b} is {ht}, longer than the mesh's {hm}");
                if ht < hm {
                    strictly_shorter += 1;
                }
            }
        }
        assert_eq!(m.hops(0, 7), Some(7), "across the mesh the long way");
        assert_eq!(t.hops(0, 7), Some(1), "across the torus through the wrap");
        // Closed form. On one wrapped axis of 8 the wrap is strictly shorter when the unwrapped
        // separation is 5, 6 or 7, and there are 2*(8-d) ordered pairs at separation d, so
        // 6 + 4 + 2 = 12 of 64 ordered pairs improve on each axis and 52 do not. A pair of cores
        // improves when EITHER axis does, which is 64*64 - 52*52 = 1392 of 4096.
        assert_eq!(strictly_shorter, 64 * 64 - 52 * 52);
        assert_eq!(strictly_shorter, 1392);
    }

    /// The diagonal link is what makes a triangular torus different from a rectangular one, and it
    /// only helps when the two displacements share a sign. Two hand-computed pairs on a 9x9 fabric
    /// pin both branches, and the rectangular torus is shown disagreeing on the first.
    #[test]
    fn the_diagonal_link_shortens_only_a_displacement_whose_axes_share_a_sign() {
        let tri = Fabric::TriangularTorus { cols: 9, rows: 9 };
        let rect = Fabric::Torus2D { cols: 9, rows: 9 };
        // Core 29 is (2, 3): both positive, so one diagonal covers both axes for three steps.
        assert_eq!(tri.hops(0, 29), Some(3));
        assert_eq!(rect.hops(0, 29), Some(5), "no diagonal on a rectangular torus");
        // Core 56 is (2, 6), and the short way in y is -3: signs disagree, so no diagonal helps.
        assert_eq!(tri.hops(0, 56), Some(5));
        assert_eq!(rect.hops(0, 56), Some(5));
    }

    /// A distance to a core that is not there has no value. Returning a large number would let a
    /// cost model sum it.
    #[test]
    fn hops_are_none_for_a_core_the_fabric_does_not_have() {
        let f = Fabric::Mesh2D { cols: 3, rows: 3 };
        assert_eq!(f.hops(0, 9), None);
        assert_eq!(f.hops(9, 0), None);
        assert_eq!(f.route(0, 9), None);
        assert_eq!(f.coords(9), None);
        assert_eq!(f.coords(8), Some((2, 2)));
        assert!(f.neighbours(9).is_empty());
        assert_eq!(
            multicast_tree(&f, 0, &[9]).unwrap_err(),
            MapError::CoreNotOnFabric { core: 9, n_cores: 9 }
        );
    }

    // ----------------------------------------------------------------------------------------
    // Multicast
    // ----------------------------------------------------------------------------------------

    /// Closed form. On a line of `n` cores, a multicast from one end to every other core costs the
    /// span, `n - 1` hops; sending one packet per destination costs the triangular number
    /// `n(n-1)/2`. At `n = 8` that is 7 against 28, a saving of exactly three quarters.
    #[test]
    fn a_multicast_tree_on_a_line_costs_the_span_and_unicast_the_triangular_number() {
        for n in 2..=12u32 {
            let f = Fabric::Mesh2D { cols: n, rows: 1 };
            let dests: Vec<u32> = (1..n).collect();
            let t = multicast_tree(&f, 0, &dests).unwrap();
            assert_eq!(t.hops, u64::from(n - 1), "span on a line of {n}");
            assert_eq!(
                t.unicast_hops,
                u64::from(n) * u64::from(n - 1) / 2,
                "triangular number on a line of {n}"
            );
        }
        let f = Fabric::Mesh2D { cols: 8, rows: 1 };
        let t = multicast_tree(&f, 0, &[1, 2, 3, 4, 5, 6, 7]).unwrap();
        assert_eq!((t.hops, t.unicast_hops), (7, 28));
        assert!((t.saving().unwrap() - 0.75).abs() < 1e-15);
    }

    /// With one destination there is nothing to share, so the two costs must be identical. An
    /// implementation that double-counted the root, or that forgot the root is in the tree, breaks
    /// here and nowhere else.
    #[test]
    fn multicast_and_unicast_agree_for_a_single_destination() {
        for f in fabrics() {
            let n = f.n_cores() as u32;
            for a in 0..n {
                for b in 0..n {
                    let t = multicast_tree(&f, a, &[b]).unwrap();
                    assert_eq!(t.hops, t.unicast_hops, "{f:?}: {a} -> {b}");
                    assert_eq!(t.hops, f.hops(a, b).unwrap());
                }
            }
        }
        // A root with only itself as a destination sends nothing, and the saving has no value.
        let t = multicast_tree(&Fabric::Mesh2D { cols: 4, rows: 4 }, 5, &[5, 5]).unwrap();
        assert_eq!((t.hops, t.unicast_hops), (0, 0));
        assert_eq!(t.saving(), None);
        assert_eq!(t.cores, vec![5]);
    }

    /// A destination **core** is one destination however many of the axon's targets live on it,
    /// so the unicast baseline counts it once. Without the deduplication the baseline triples here
    /// and [`super::McastTree::saving`] flatters multicast by the same factor — a mutation that
    /// survived every other test in this module.
    #[test]
    fn a_multicast_tree_deduplicates_its_destination_cores() {
        let f = Fabric::Mesh2D { cols: 4, rows: 1 };
        let t = multicast_tree(&f, 0, &[3, 3, 3]).unwrap();
        assert_eq!(t.unicast_hops, 3, "core 3 is one destination, not three");
        assert_eq!(t.hops, 3);
        assert_eq!(t.saving(), Some(0.0), "nothing to share with one destination core");
        assert_eq!(t.cores, vec![0, 1, 2, 3]);
        // The root itself appearing among the destinations is an on-core delivery and costs nothing.
        let u = multicast_tree(&f, 0, &[0, 0, 1]).unwrap();
        assert_eq!((u.hops, u.unicast_hops), (1, 1));
    }

    /// A link is one piece of wire. On a fabric two wide, stepping `+1` and stepping `-1` arrive at
    /// the same core, and counting it twice would double that core's degree — invisible to a
    /// shortest-path check, and wrong in any degree histogram or router-port count built on it.
    #[test]
    fn fabric_neighbours_are_sorted_distinct_and_exclude_the_core_itself() {
        for f in fabrics() {
            let n = f.n_cores() as u32;
            for c in 0..n {
                let nb = f.neighbours(c);
                let mut sorted = nb.clone();
                sorted.sort_unstable();
                assert_eq!(nb, sorted, "{f:?} core {c}: unsorted");
                sorted.dedup();
                assert_eq!(nb, sorted, "{f:?} core {c}: duplicated a link");
                assert!(!nb.contains(&c), "{f:?} core {c}: is its own neighbour");
            }
        }
        // Hand counts. A wrapped axis of two folds onto itself; an axis of three does not.
        assert_eq!(Fabric::Torus2D { cols: 2, rows: 3 }.neighbours(0), vec![1, 2, 4]);
        assert_eq!(Fabric::Torus2D { cols: 3, rows: 3 }.neighbours(0), vec![1, 2, 3, 6]);
        assert_eq!(Fabric::TriangularTorus { cols: 2, rows: 2 }.neighbours(0), vec![1, 2, 3]);
        // ⛔ A wrapped axis of ONE folds onto the core itself: stepping `+1` along a fabric one
        // column wide arrives back where it started, and that step has to be dropped rather than
        // listed. Every fabric this sweep had before was at least two cores across every wrapped
        // axis, so no direction ever produced the core it came from and the filter that removes
        // it was unreachable.
        assert_eq!(Fabric::Torus2D { cols: 1, rows: 5 }.neighbours(0), vec![1, 4]);
        assert_eq!(Fabric::Torus2D { cols: 1, rows: 5 }.neighbours(2), vec![1, 3]);
        assert_eq!(Fabric::TriangularTorus { cols: 4, rows: 1 }.neighbours(0), vec![1, 3]);
        assert_eq!(Fabric::TriangularTorus { cols: 4, rows: 1 }.neighbours(2), vec![1, 3]);
        assert_eq!(Fabric::Mesh2D { cols: 5, rows: 4 }.neighbours(0), vec![1, 5], "a corner");
        assert_eq!(Fabric::Crossbar { cores: 3 }.neighbours(1), vec![0, 2]);
        // On a torus wide enough not to fold, every core has exactly the fabric's degree.
        for c in 0..64u32 {
            assert_eq!(Fabric::Torus2D { cols: 8, rows: 8 }.neighbours(c).len(), 4);
            assert_eq!(Fabric::TriangularTorus { cols: 8, rows: 8 }.neighbours(c).len(), 6);
        }
    }

    /// Two ways round a wrapped axis of even width are the same length, so the choice is arbitrary
    /// — and it still has to be **fixed**, or two runs of the same workload route differently and
    /// disagree about which links carried traffic. Forward, stated once here.
    #[test]
    fn a_wrapped_route_takes_the_forward_direction_on_an_exact_tie() {
        let f = Fabric::Torus2D { cols: 8, rows: 1 };
        assert_eq!(f.hops(0, 4), Some(4), "four either way");
        assert_eq!(f.route(0, 4), Some(vec![0, 1, 2, 3, 4]), "forward on the tie");
        // Off the tie it takes the genuinely shorter way, which here is backward through the wrap.
        assert_eq!(f.hops(0, 5), Some(3));
        assert_eq!(f.route(0, 5), Some(vec![0, 7, 6, 5]));
        let t = Fabric::TriangularTorus { cols: 8, rows: 1 };
        assert_eq!(t.route(0, 4), Some(vec![0, 1, 2, 3, 4]));
    }

    /// Violations are reported most-severe first, and severity is by **kind**: a wall the network
    /// has to be redesigned around outranks a core that is merely too full. Pinned with all three
    /// kinds binding at once.
    #[test]
    fn violations_are_reported_wall_first_whatever_their_magnitude() {
        let mut b = NetBuilder::new(301);
        for pre in 0..300u32 {
            b.connect(pre, 300, 1e-3, 1).unwrap();
        }
        let net = b.build();
        let p = Partition::single_core(&net).unwrap();
        let f = p.check(&net, &CoreLimits::new(Some(100), Some(100), Some(256))).unwrap();
        assert_eq!(f.verdict, Some(false));
        assert_eq!(
            f.binds,
            vec![
                CoreBind::FanIn { neuron: 300, fan_in: 300, cap: 256, core: 0 },
                CoreBind::Synapses { core: 0, used: 300, cap: 100 },
                CoreBind::Neurons { core: 0, used: 301, cap: 100 },
            ]
        );
        // The neuron count is over by 201 and the wall by 44, and the wall still comes first.
        assert_eq!(f.binds[0].overflow(), 44);
        assert_eq!(f.binds[2].overflow(), 201);
        assert!(!f.repartitionable(), "one unrepartitionable bind is enough");
        assert!(f.headroom.is_empty(), "a constraint that bound reports no headroom");
    }

    /// Chip crossings under the row-major block assignment, hand-counted on a line.
    #[test]
    fn a_multicast_tree_counts_chip_crossings_against_a_block_assignment() {
        let f = Fabric::Mesh2D { cols: 8, rows: 1 };
        let t = multicast_tree(&f, 0, &[1, 2, 3, 4, 5, 6, 7]).unwrap();
        // Cores 0..3 are chip 0 and 4..7 are chip 1, so exactly the 3-4 link crosses.
        assert_eq!(t.crossings(4), Some(1));
        // One core per chip makes every one of the seven links a crossing.
        assert_eq!(t.crossings(1), Some(7));
        // Everything on one chip crosses nothing.
        assert_eq!(t.crossings(8), Some(0));
        // A chip that holds no cores describes no machine.
        assert_eq!(t.crossings(0), None);
    }

    // ----------------------------------------------------------------------------------------
    // The cut
    // ----------------------------------------------------------------------------------------

    /// Verification (b): the cut against a hand count. Six synapses, three of them crossing, worked
    /// out in the comment so a reader can check the implementation rather than trust it.
    #[test]
    fn the_cut_count_matches_a_hand_count_on_a_small_graph() {
        let mut b = NetBuilder::new(5);
        for &(p, q) in &[(0u32, 1u32), (1, 2), (2, 3), (3, 4), (4, 0), (0, 2)] {
            b.connect(p, q, 1e-3, 1).unwrap();
        }
        let net = b.build();
        // core:      0  0  1  1  0
        // 0->1 same, 0->2 CUT, 1->2 CUT, 2->3 same, 3->4 CUT, 4->0 same.
        let p = Partition::new(&net, vec![0, 0, 1, 1, 0], 2).unwrap();
        assert_eq!(p.cut_edges(&net).unwrap(), 3);
        assert!((p.cut_fraction(&net).unwrap().unwrap() - 0.5).abs() < 1e-15);
        // Everything on one core cuts nothing, whatever the graph.
        let one = Partition::single_core(&net).unwrap();
        assert_eq!(one.cut_edges(&net).unwrap(), 0);
        assert_eq!(one.used_cores(), 1);
        assert_eq!(p.used_cores(), 2);
    }

    /// A synapse is a row of memory and a delivery, so two parallel synapses are two crossings; a
    /// self-synapse is never a crossing. Both would be invisible in a "count the distinct
    /// neighbouring pairs" implementation.
    #[test]
    fn cut_edges_counts_parallel_synapses_separately_and_never_counts_a_self_synapse() {
        let mut b = NetBuilder::new(2);
        b.connect(0, 1, 1e-3, 1).unwrap();
        b.connect(0, 1, 2e-3, 2).unwrap();
        b.connect(0, 0, 1e-3, 1).unwrap();
        let net = b.build();
        assert_eq!(net.n_syn, 3);
        let p = Partition::new(&net, vec![0, 1], 2).unwrap();
        assert_eq!(p.cut_edges(&net).unwrap(), 2, "both parallel synapses cross");
        let together = Partition::new(&net, vec![0, 0], 2).unwrap();
        assert_eq!(together.cut_edges(&net).unwrap(), 0);
    }

    /// A synapse's storage is charged to the core holding its **postsynaptic** neuron, which is
    /// where every part in [`crate::hardware`] fetches it from. Charging the presynaptic core would
    /// give core 0 all three synapses here instead of none.
    #[test]
    fn core_loads_charge_a_synapse_to_the_postsynaptic_core() {
        let mut b = NetBuilder::new(4);
        b.connect(0, 1, 1e-3, 1).unwrap();
        b.connect(0, 2, 1e-3, 1).unwrap();
        b.connect(0, 3, 1e-3, 1).unwrap();
        let net = b.build();
        let p = Partition::new(&net, vec![0, 1, 1, 1], 2).unwrap();
        let loads = p.loads(&net).unwrap();
        assert_eq!((loads[0].neurons, loads[0].synapses), (1, 0), "the fan-out core stores nothing");
        assert_eq!((loads[1].neurons, loads[1].synapses), (3, 3));
        assert_eq!(loads[0].worst_fan_in, Some((0, 0)));
        assert_eq!(loads[1].worst_fan_in, Some((1, 1)));
        // An empty core reports no maximum rather than a maximum of zero.
        let q = Partition::new(&net, vec![0, 0, 0, 0], 3).unwrap();
        assert_eq!(q.loads(&net).unwrap()[2].worst_fan_in, None);
    }

    // ----------------------------------------------------------------------------------------
    // Partitioning
    // ----------------------------------------------------------------------------------------

    /// Verification (a). Two twelve-cliques joined by one synapse, on two cores of twelve. The
    /// minimum cut is **1** in closed form — the graph is connected so zero is impossible, and
    /// deleting the bridge leaves two components of exactly the core capacity — and the refined
    /// partitioner finds it from every one of eight seeds.
    #[test]
    fn two_dense_clusters_joined_by_one_edge_are_cut_in_exactly_one_place() {
        let net = two_cliques(12);
        let limits = CoreLimits::new(Some(12), None, None);
        for seed in 0..8u64 {
            let plan = partition_refined(&net, &limits, 2, seed, 20).unwrap();
            assert_eq!(plan.cut, 1, "seed {seed} cut {} synapses, not the bridge alone", plan.cut);
            // And it is the bridge: neuron 0 and neuron 12 are the only pair on different cores
            // among the two cliques' members.
            let core = &plan.partition.core_of;
            for i in 1..12 {
                assert_eq!(core[i], core[0], "seed {seed}: clique A split at neuron {i}");
                assert_eq!(core[12 + i], core[12], "seed {seed}: clique B split at neuron {}", 12 + i);
            }
            assert_ne!(core[0], core[12], "seed {seed}: both cliques landed on one core");
            assert_eq!(plan.partition.check(&net, &limits).unwrap().verdict, Some(true));
        }
    }

    /// Verification (c), with **both numbers reported**. The same graph as above is the one that
    /// traps a single streaming pass: the second neuron in the stream has no placed neighbours, so
    /// the occupancy penalty pushes it onto the empty core, and a single pass never takes it back.
    /// One neuron of each clique ends up stranded, cutting eleven synapses on each side plus the
    /// bridge — 23 where the optimum is 1.
    ///
    /// Kernighan-Lin exchanges recover all of it, from every seed.
    #[test]
    fn kernighan_lin_beats_the_streaming_greedy_where_the_stream_order_traps_it() {
        let net = two_cliques(12);
        let limits = CoreLimits::new(Some(12), None, None);
        let greedy: Vec<u64> =
            (0..8).map(|s| partition_greedy(&net, &limits, 2, s).unwrap().cut).collect();
        let refined: Vec<u64> =
            (0..8).map(|s| partition_refined(&net, &limits, 2, s, 20).unwrap().cut).collect();
        assert_eq!(
            greedy,
            vec![1, 23, 1, 23, 23, 23, 1, 23],
            "the streaming pass's cut per seed; the optimum is 1"
        );
        assert_eq!(refined, vec![1; 8], "refinement reaches the optimum from every seed");
        let trapped = greedy.iter().filter(|&&c| c > 1).count();
        assert_eq!(trapped, 5, "5 of 8 seeds trapped the streaming pass");
        // 23 against 1 is the measured quality difference, not a claim about heuristics.
        assert_eq!(greedy.iter().copied().max(), Some(23));
    }

    /// The honest other half of the comparison: Kernighan-Lin is a heuristic and it does **not**
    /// always reach the optimum. On a 32-cycle bisected into two arcs the minimum cut is exactly 2
    /// — a balanced cut of a cycle crosses it an even number of times and cannot cross zero times
    /// on a connected graph — and refinement reaches 2 from one seed of six and stalls at 4 from
    /// the rest. Reported rather than hidden behind a seed that happens to work.
    #[test]
    fn kernighan_lin_does_not_always_reach_the_optimum_and_this_is_what_it_misses() {
        let net = cycle(32);
        let limits = CoreLimits::new(Some(16), None, None);
        let refined: Vec<u64> =
            (0..6).map(|s| partition_refined(&net, &limits, 2, s, 20).unwrap().cut).collect();
        assert_eq!(refined, vec![2, 4, 4, 4, 4, 4], "the closed-form optimum is 2");
        assert_eq!(refined.iter().filter(|&&c| c == 2).count(), 1);
        // Every result is still balanced and legal, which is the part that is guaranteed.
        for seed in 0..6u64 {
            let plan = partition_refined(&net, &limits, 2, seed, 20).unwrap();
            let loads = plan.partition.loads(&net).unwrap();
            assert_eq!((loads[0].neurons, loads[1].neurons), (16, 16), "seed {seed}");
        }
    }

    /// The cross-check that makes the exchange arithmetic checkable rather than self-consistent:
    /// the gain summed from the per-swap formula must equal the drop in a **full independent
    /// recount** of the cut. Mutating the `-2 * multiplicity` correction, or the sign of either
    /// difference, breaks this and breaks nothing else.
    #[test]
    fn the_refinement_gain_equals_the_measured_drop_in_cut_edges() {
        let limits = CoreLimits::new(Some(12), None, None);
        let cases: Vec<(&str, Net, u32, CoreLimits)> = vec![
            ("two cliques", two_cliques(12), 2, limits),
            ("cycle", cycle(32), 2, CoreLimits::new(Some(16), None, None)),
            ("chain on four cores", chain(40), 4, CoreLimits::new(Some(10), None, None)),
        ];
        for (name, net, cores, lim) in cases {
            for seed in 0..6u64 {
                let plan = partition_refined(&net, &lim, cores, seed, 20).unwrap();
                assert_eq!(
                    plan.cut_before_refinement - plan.cut,
                    plan.refinement_gain,
                    "{name} seed {seed}: claimed {} removed, recount says {} -> {}",
                    plan.refinement_gain,
                    plan.cut_before_refinement,
                    plan.cut
                );
                // And the recount agrees with the partition it shipped.
                assert_eq!(plan.cut, plan.partition.cut_edges(&net).unwrap());
                assert_eq!(plan.method, Method::KernighanLin);
                if plan.refinement_gain > 0 {
                    assert!(plan.swaps_kept > 0, "{name} seed {seed}: gain with no exchange kept");
                }
            }
        }
    }

    /// Refinement keeps a pass only when its cumulative gain is positive, so it can never ship a
    /// worse cut than the pass it started from. Swept over three graphs and eight seeds.
    #[test]
    fn refinement_never_increases_the_cut_across_a_seed_sweep() {
        let cases: Vec<(&str, Net, u32, CoreLimits)> = vec![
            ("two cliques", two_cliques(10), 2, CoreLimits::new(Some(10), None, None)),
            ("cycle", cycle(24), 3, CoreLimits::new(Some(8), None, None)),
            ("chain", chain(30), 5, CoreLimits::new(Some(6), None, None)),
        ];
        for (name, net, cores, lim) in cases {
            for seed in 0..8u64 {
                let g = partition_greedy(&net, &lim, cores, seed).unwrap();
                let k = partition_refined(&net, &lim, cores, seed, 20).unwrap();
                assert_eq!(k.cut_before_refinement, g.cut, "{name} seed {seed}: different start");
                assert!(k.cut <= g.cut, "{name} seed {seed}: {} > {}", k.cut, g.cut);
                assert_eq!(g.refinement_gain, 0);
                assert_eq!(g.passes_run, 0);
            }
        }
    }

    /// Verification (f). Same seed, same assignment, bit for bit; a different seed moves it. The
    /// second half matters: an implementation that ignored the seed entirely would pass the first.
    #[test]
    fn partitioning_is_bit_identical_for_a_seed_and_moves_with_the_seed() {
        let net = two_cliques(9);
        let limits = CoreLimits::new(Some(9), Some(4096), Some(64));
        for seed in [0u64, 1, 7, 12345, u64::MAX] {
            let a = partition_greedy(&net, &limits, 2, seed).unwrap();
            let b = partition_greedy(&net, &limits, 2, seed).unwrap();
            assert_eq!(a, b, "seed {seed} was not reproducible");
            let c = partition_refined(&net, &limits, 2, seed, 20).unwrap();
            let d = partition_refined(&net, &limits, 2, seed, 20).unwrap();
            assert_eq!(c, d, "refinement from seed {seed} was not reproducible");
        }
        let assignments: Vec<Vec<u32>> = (0..8)
            .map(|s| partition_greedy(&net, &limits, 2, s).unwrap().partition.core_of)
            .collect();
        let distinct = {
            let mut v = assignments.clone();
            v.sort();
            v.dedup();
            v.len()
        };
        assert!(distinct > 1, "every seed produced the same assignment; the seed is decorative");
    }

    // ----------------------------------------------------------------------------------------
    // Feasibility: which constraint binds, on which core, by how much
    // ----------------------------------------------------------------------------------------

    /// Verification (e). An overfilled core is refused **by name and by number**, not as a boolean.
    #[test]
    fn a_partition_that_overfills_a_core_is_refused_with_the_core_and_the_numbers() {
        let net = chain(6);
        let p = Partition::new(&net, vec![0, 0, 0, 0, 1, 1], 2).unwrap();
        let f = p.check(&net, &CoreLimits::new(Some(3), None, None)).unwrap();
        assert_eq!(f.verdict, Some(false));
        assert_eq!(f.binds.len(), 1);
        assert_eq!(f.binds[0], CoreBind::Neurons { core: 0, used: 4, cap: 3 });
        assert_eq!(f.binds[0].overflow(), 1);
        assert_eq!(f.binds[0].constraint(), "neurons per core");
        assert!(f.repartitionable(), "moving a neuron off core 0 fixes this");
        assert!(f.to_string().contains("core 0: 4 neurons, holds 3 (over by 1)"));

        // The synapse limit binds on the core holding the fan-in, which is core 1 here.
        let mut b = NetBuilder::new(6);
        for pre in 0..5u32 {
            b.connect(pre, 5, 1e-3, 1).unwrap();
        }
        let star = b.build();
        let q = Partition::new(&star, vec![0, 0, 0, 0, 0, 1], 2).unwrap();
        let g = q.check(&star, &CoreLimits::new(None, Some(4), None)).unwrap();
        assert_eq!(g.binds, vec![CoreBind::Synapses { core: 1, used: 5, cap: 4 }]);
        assert_eq!(g.binds[0].overflow(), 1);
        assert_eq!(g.unchecked, vec!["maximum fan-in per neuron", "neurons per core"]);
    }

    /// Verification (e), the other half: a fan-in wall names the **neuron**, and says that moving
    /// neurons between cores will not help. That distinction is why this is not one boolean.
    #[test]
    fn a_fan_in_wall_names_the_neuron_and_is_not_relieved_by_repartitioning() {
        let mut b = NetBuilder::new(301);
        for pre in 0..300u32 {
            b.connect(pre, 300, 1e-3, 1).unwrap();
        }
        let net = b.build();
        let p = Partition::single_core(&net).unwrap();
        let f = p.check(&net, &CoreLimits::from_part(&TRUENORTH)).unwrap();
        assert_eq!(f.verdict, Some(false));
        let bind = *f.binding().unwrap();
        assert_eq!(bind, CoreBind::FanIn { neuron: 300, fan_in: 300, cap: 256, core: 0 });
        assert_eq!(bind.overflow(), 300 - 256);
        assert!(!bind.relieved_by_repartitioning());
        assert!(!f.repartitionable());
        assert_eq!(bind.precedence(), 0, "the wall outranks every capacity constraint");
        assert!(bind.to_string().contains("repartitioning does not fix this"));
        // A `CoreBind` a caller built that is not a violation answers 0 rather than wrapping.
        assert_eq!(CoreBind::Neurons { core: 0, used: 1, cap: 9 }.overflow(), 0);
    }

    /// The wall is detected **before** any placement work, because no placement can help.
    #[test]
    fn a_neuron_whose_fan_in_exceeds_a_whole_core_is_refused_before_any_placement() {
        let mut b = NetBuilder::new(30);
        for pre in 0..20u32 {
            b.connect(pre, 29, 1e-3, 1).unwrap();
        }
        let net = b.build();
        let limits = CoreLimits::new(Some(16), Some(10), None);
        assert_eq!(
            partition_greedy(&net, &limits, 4, 0).unwrap_err(),
            MapError::FanInExceedsCore { neuron: 29, fan_in: 20, cap: 10 }
        );
        // The same machine with room for the row places it.
        let roomy = CoreLimits::new(Some(16), Some(64), None);
        assert!(partition_greedy(&net, &roomy, 4, 0).is_ok());
        // ⛔ And the boundary itself: a core that stores EXACTLY the neuron's fan-in is not a
        // wall. The refusal is `fan_in > cap` and only its failing side was pinned, so tightening
        // it to `>=` — refusing the neuron that fits precisely — was invisible.
        let exact = CoreLimits::new(Some(16), Some(20), None);
        let plan = partition_greedy(&net, &exact, 4, 0).unwrap();
        assert_eq!(plan.partition.check(&net, &exact).unwrap().verdict, Some(true));
        assert_eq!(net.in_degrees()[29], 20, "the neuron that exactly fills a core of 20");
    }

    /// `Some(true)` against no stated limit would be a vacuous pass, so it is `None` instead — and
    /// `None` is not a pass. The same rule [`crate::hardware::Fit`] follows.
    #[test]
    fn an_unlimited_core_gives_no_verdict_rather_than_a_pass() {
        let net = chain(8);
        let p = Partition::single_core(&net).unwrap();
        let f = p.check(&net, &CoreLimits::UNLIMITED).unwrap();
        assert_eq!(f.verdict, None);
        assert!(f.binds.is_empty());
        assert!(f.headroom.is_empty());
        assert_eq!(
            f.unchecked,
            vec!["maximum fan-in per neuron", "synapses per core", "neurons per core"]
        );
        assert!(!f.repartitionable(), "nothing binds, so nothing is relieved");
        assert!(f.to_string().contains("NO VERDICT"));
    }

    /// Limits taken from a part carry its **empty** fields through as `None`, so a part that states
    /// no wall is reported as unchecked rather than as satisfied.
    #[test]
    fn limits_from_a_part_carry_its_empty_fields_through_as_unchecked() {
        let tn = CoreLimits::from_part(&TRUENORTH);
        assert_eq!(tn.neurons_per_core, Some(256));
        assert_eq!(tn.synapses_per_core, Some(65536));
        assert_eq!(tn.max_fan_in, Some(256), "TrueNorth's crossbar column is the wall");

        // ⛔ TrueNorth's neuron count and its fan-in wall are BOTH 256 — one crossbar, read along
        // two axes — so reading either field from the other moved no number, and SpiNNaker states
        // neither. DYNAP-SE states 256 neurons against 64 content-addressable-memory entries per
        // neuron, the tightest fan-in in that table, and the two cannot be confused.
        let dy = CoreLimits::from_part(&DYNAP_SE);
        assert_eq!(dy.neurons_per_core, Some(256));
        assert_eq!(dy.max_fan_in, Some(64), "the CAM depth, not the neuron count");
        assert_eq!(dy.synapses_per_core, Some(16384), "which is 256 x 64");

        // SpiNNaker states none of the three: its limits are real-time budgets, not structures.
        let sp = CoreLimits::from_part(&SPINNAKER);
        assert_eq!((sp.neurons_per_core, sp.synapses_per_core, sp.max_fan_in), (None, None, None));
        let net = chain(4);
        let f = Partition::single_core(&net).unwrap().check(&net, &sp).unwrap();
        assert_eq!(f.verdict, None, "a part that states nothing does not pass");
        assert_eq!(f.unchecked.len(), 3);

        // Headroom on a satisfied constraint reports the worst core, not an average.
        let p = Partition::new(&net, vec![0, 0, 1, 1], 2).unwrap();
        let g = p.check(&net, &CoreLimits::new(Some(4), None, None)).unwrap();
        assert_eq!(g.verdict, Some(true));
        assert_eq!(g.headroom[0].worst_used, 2);
        assert_eq!(g.headroom[0].spare(), 2);
        assert!((g.headroom[0].utilisation().unwrap() - 0.5).abs() < 1e-15);
        // ⛔ And it PRINTS as a percentage under the percent sign it prints: 2 of 4 is 50.0%, not
        // 0.5%. `every_display_impl_names_its_numbers` only looked for the words "worst core", so
        // the factor of a hundred between the ratio and the sign was carried by nothing.
        assert!(g.to_string().contains("(50.0%)"), "{g}");
        // ⛔ WORST core, not best: `min_by_key` for `max_by_key` on either branch was green,
        // because a 2/2 split has no worst. 3/1 does, on both constraints.
        let p = Partition::new(&net, vec![0, 0, 0, 1], 2).unwrap();
        let g = p.check(&net, &CoreLimits::new(Some(4), Some(8), None)).unwrap();
        assert_eq!(g.verdict, Some(true));
        let by = |name: &str| g.headroom.iter().find(|h| h.constraint == name).unwrap();
        assert_eq!((by("neurons per core").worst_used, by("neurons per core").worst_core), (3, 0));
        // chain(4): synapses 0->1, 1->2, 2->3 charged to the postsynaptic core. Core 0 holds
        // neurons 0, 1, 2 with in-degrees 0, 1, 1; core 1 holds neuron 3 with in-degree 1.
        assert_eq!((by("synapses per core").worst_used, by("synapses per core").worst_core), (2, 0));
        assert_eq!(super::CoreHeadroom { constraint: "x", worst_used: 0, worst_core: 0, cap: 0 }
            .utilisation(), None);
    }

    /// Arithmetic that fails before any work is done, with the arithmetic in the error.
    #[test]
    fn the_streaming_pass_refuses_a_machine_that_is_too_small_and_a_capacity_of_zero() {
        let net = chain(10);
        assert_eq!(
            partition_greedy(&net, &CoreLimits::new(Some(2), None, None), 3, 0).unwrap_err(),
            MapError::NotEnoughRoom { neurons: 10, n_cores: 3, neurons_per_core: 2, capacity: 6 }
        );
        assert_eq!(
            partition_greedy(&net, &CoreLimits::new(Some(0), None, None), 3, 0).unwrap_err(),
            MapError::ZeroCapacity { which: "neurons per core" }
        );
        assert_eq!(
            partition_greedy(&net, &CoreLimits::new(None, Some(0), None), 3, 0).unwrap_err(),
            MapError::ZeroCapacity { which: "synapses per core" }
        );
        assert_eq!(
            partition_greedy(&net, &CoreLimits::UNLIMITED, 0, 0).unwrap_err(),
            MapError::NoCores
        );
        // ⛔ The synapse twin of NotEnoughRoom: 20 neurons carrying 100 synapses offered 2 cores
        // of 10 used to come back as NoFeasibleCore, "try another seed or refinement" — and every
        // seed and the refiner failed identically, because the machine has 20 synapse slots.
        let mut b = NetBuilder::new(20);
        for i in 0..20u32 {
            for j in 0..5u32 {
                b.connect((i + j + 1) % 20, i, 1e-3, 1).unwrap();
            }
        }
        let dense = b.build();
        assert_eq!(dense.n_syn, 100);
        let lim = CoreLimits::new(Some(10), Some(10), None);
        assert_eq!(
            partition_greedy(&dense, &lim, 2, 0).unwrap_err(),
            MapError::NotEnoughSynapseRoom { synapses: 100, n_cores: 2, synapses_per_core: 10, capacity: 20 }
        );
        assert_eq!(
            partition_refined(&dense, &lim, 2, 0, 5).unwrap_err(),
            MapError::NotEnoughSynapseRoom { synapses: 100, n_cores: 2, synapses_per_core: 10, capacity: 20 }
        );
        // ⛔ The boundary from the other side: a network whose synapses EXACTLY fill the machine
        // places. `synapses > capacity` was pinned only where it fails, so tightening it to `>=`
        // — refusing the network that fits precisely — was invisible. 100 synapses into 2 cores
        // of 50, and the neuron cap of 10 forces the ten-neuron blocks that make each core carry
        // exactly its 50.
        let exact = CoreLimits::new(Some(10), Some(50), None);
        let plan = partition_greedy(&dense, &exact, 2, 0).unwrap();
        assert_eq!(plan.partition.check(&dense, &exact).unwrap().verdict, Some(true));
        let loads = plan.partition.loads(&dense).unwrap();
        assert_eq!((loads[0].synapses, loads[1].synapses), (50, 50), "2 x 50 is 100 exactly");
        // And the per-core bookkeeping bound, on both entry points.
        assert_eq!(
            Partition::new(&net, vec![0; 10], super::MAX_CORES + 1).unwrap_err(),
            MapError::TooManyCores { n_cores: super::MAX_CORES + 1, limit: super::MAX_CORES }
        );
        assert_eq!(
            partition_greedy(&net, &CoreLimits::UNLIMITED, super::MAX_CORES + 1, 0).unwrap_err(),
            MapError::TooManyCores { n_cores: super::MAX_CORES + 1, limit: super::MAX_CORES }
        );
        assert!(Partition::new(&net, vec![0; 10], super::MAX_CORES).is_ok());
        assert_eq!(
            Partition::new(&net, vec![0; 9], 2).unwrap_err(),
            MapError::LengthMismatch { what: "core assignment", got: 9, neurons: 10 }
        );
        assert_eq!(
            Partition::new(&net, vec![5; 10], 2).unwrap_err(),
            MapError::CoreOutOfRange { neuron: 0, core: 5, n_cores: 2 }
        );
    }

    /// A partition carries the neuron count it was built against, so handing it a different network
    /// is refused rather than indexed into.
    #[test]
    fn a_partition_built_against_another_network_is_refused() {
        let a = chain(4);
        let b = chain(5);
        let p = Partition::single_core(&a).unwrap();
        assert_eq!(
            p.cut_edges(&b).unwrap_err(),
            MapError::LengthMismatch { what: "core assignment", got: 4, neurons: 5 }
        );
        assert!(p.loads(&b).is_err());
        assert!(p.check(&b, &CoreLimits::UNLIMITED).is_err());
        assert!(spike_hops(&b, &p, &Fabric::Crossbar { cores: 2 }, None, &[1; 5]).is_err());
    }

    /// A network with nothing in it maps without inventing anything.
    #[test]
    fn an_empty_network_maps_onto_one_core_without_inventing_anything() {
        let net = NetBuilder::new(0).build();
        let plan = partition_greedy(&net, &CoreLimits::new(Some(4), None, None), 2, 0).unwrap();
        assert_eq!(plan.cut, 0);
        assert!(plan.partition.core_of.is_empty());
        assert_eq!(plan.partition.cut_fraction(&net).unwrap(), None, "0/0 has no value");
        let hops =
            spike_hops(&net, &plan.partition, &Fabric::Crossbar { cores: 2 }, Some(1), &[]).unwrap();
        assert_eq!(hops.deliveries(), 0);
        assert_eq!(hops.hops_per_spike(), None);
        assert_eq!(hops.multicast_saving(), None);
    }

    // ----------------------------------------------------------------------------------------
    // Workload hops and the bill
    // ----------------------------------------------------------------------------------------

    /// Every count in [`SpikeHops`], against a hand computation set out in the comment.
    #[test]
    fn spike_hops_counts_every_delivery_and_matches_a_hand_computation() {
        let mut b = NetBuilder::new(4);
        for &(p, q) in &[(0u32, 1u32), (0, 2), (0, 3), (1, 3)] {
            b.connect(p, q, 1e-3, 1).unwrap();
        }
        let net = b.build();
        let f = Fabric::Mesh2D { cols: 2, rows: 2 };
        let p = Partition::new(&net, vec![0, 1, 2, 3], 4).unwrap();
        // Neuron 0 fires twice to cores {1,2,3}. XY routes 0->1 = [0,1], 0->2 = [0,2],
        // 0->3 = [0,1,3]; the union has links (0,1),(0,2),(1,3) = 3 hops. Unicast is 1+1+2 = 4.
        // Neuron 1 fires three times to core 3 alone: 1 hop, unicast 1.
        let h = spike_hops(&net, &p, &f, Some(2), &[2, 3, 0, 0]).unwrap();
        // Named so the hand arithmetic below stays readable: (spikes, tree hops, unicast hops,
        // targets, crossing links) for neuron 0 and for neuron 1.
        let (s0, tree0, uni0, tgt0, cross0) = (2u64, 3u64, 4u64, 3u64, 2u64);
        let (s1, tree1, uni1, tgt1, cross1) = (3u64, 1u64, 1u64, 1u64, 1u64);
        assert_eq!(h.multicast_hops, s0 * tree0 + s1 * tree1);
        assert_eq!(h.unicast_hops, s0 * uni0 + s1 * uni1);
        assert_eq!(h.off_core_deliveries, s0 * tgt0 + s1 * tgt1);
        assert_eq!(h.on_core_deliveries, 0);
        assert_eq!(h.deliveries(), 9);
        assert_eq!(h.spikes, 5);
        assert_eq!(h.sources, 2);
        // Cores 0,1 are chip 0 and 2,3 are chip 1: of neuron 0's three links, (0,2) and (1,3)
        // cross; neuron 1's single link (1,3) crosses.
        assert_eq!(h.chip_crossings, Some(s0 * cross0 + s1 * cross1));
        assert_eq!(h.chip_crossings, Some(7));
        assert!((h.hops_per_spike().unwrap() - 9.0 / 5.0).abs() < 1e-15);
        assert!((h.multicast_saving().unwrap() - (1.0 - 9.0 / 11.0)).abs() < 1e-15);

        // On-core deliveries cost no hops at all: put everything on one core.
        let one = Partition::single_core(&net).unwrap();
        let g = spike_hops(&net, &one, &f, Some(2), &[2, 3, 0, 0]).unwrap();
        assert_eq!((g.multicast_hops, g.unicast_hops, g.chip_crossings), (0, 0, Some(0)));
        assert_eq!(g.on_core_deliveries, 9);
        assert_eq!(g.off_core_deliveries, 0);
        assert_eq!(g.deliveries(), 9, "the same deliveries, none of them on the fabric");

        // The deliveries are the ledger's synaptic operations, counted independently.
        let ledger = Ledger { syn_ops: 9, ..Ledger::default() };
        assert_eq!(h.deliveries(), ledger.syn_ops);
        assert!((h.hops_per_synaptic_operation(&ledger).unwrap() - 1.0).abs() < 1e-15);
        assert_eq!(h.hops_per_synaptic_operation(&Ledger::default()), None);
        // ⛔ With a ledger whose count is NOT 9: on this fixture multicast_hops, deliveries() and
        // syn_ops all happened to be 9, so `deliveries()` in the numerator was green. 9/18 = 0.5.
        let other = Ledger { syn_ops: 18, ..Ledger::default() };
        assert!((h.hops_per_synaptic_operation(&other).unwrap() - 0.5).abs() < 1e-15);
    }

    /// **The claim the module is written around**: the same network and the same spikes, mapped
    /// two ways, deliver identically and cost the fabric 25 hops against 7.
    #[test]
    fn placement_moves_the_fabric_bill_and_leaves_the_deliveries_alone() {
        let net = chain(8);
        let f = Fabric::Mesh2D { cols: 8, rows: 1 };
        let spikes = [1u64; 8];
        let neighbourly = Partition::new(&net, (0..8).collect(), 8).unwrap();
        let scattered = Partition::new(&net, vec![0, 4, 1, 5, 2, 6, 3, 7], 8).unwrap();

        let good = spike_hops(&net, &neighbourly, &f, Some(8), &spikes).unwrap();
        let bad = spike_hops(&net, &scattered, &f, Some(8), &spikes).unwrap();
        assert_eq!(good.multicast_hops, 7, "each of the 7 synapses crosses one link");
        assert_eq!(bad.multicast_hops, 4 + 3 + 4 + 3 + 4 + 3 + 4);
        assert_eq!(bad.multicast_hops, 25);
        assert_eq!(good.deliveries(), bad.deliveries(), "the work done is identical");
        assert_eq!(good.spikes, bad.spikes);
        assert_eq!(good.off_core_deliveries, bad.off_core_deliveries);
        assert_eq!(neighbourly.cut_edges(&net).unwrap(), scattered.cut_edges(&net).unwrap());
        // ...which is exactly why the cut is a proxy: both placements cut all seven synapses and
        // one of them costs three and a half times as much to run.

        // On a crossbar the two are indistinguishable, which is what makes it the trap the type's
        // doc warns about.
        let x = Fabric::Crossbar { cores: 8 };
        let a = spike_hops(&net, &neighbourly, &x, Some(8), &spikes).unwrap();
        let b = spike_hops(&net, &scattered, &x, Some(8), &spikes).unwrap();
        assert_eq!((a.multicast_hops, b.multicast_hops), (7, 7));

        // ⛔ Hops per synaptic operation is HOPS over the ledger's operations. The fixture in
        // `spike_hops_counts_every_delivery_and_matches_a_hand_computation` has 9 hops and 9
        // deliveries, so putting `deliveries()` in the numerator read identically there — and
        // doubling that ledger's count did not separate them either, because it is the
        // DENOMINATOR that moved. Here the same seven deliveries cost 25 hops one way and 7 the
        // other, which is the whole point of the quantity.
        let led = Ledger { syn_ops: 7, ..Ledger::default() };
        assert_eq!((good.deliveries(), bad.deliveries()), (7, 7));
        assert!((bad.hops_per_synaptic_operation(&led).unwrap() - 25.0 / 7.0).abs() < 1e-15);
        assert!((good.hops_per_synaptic_operation(&led).unwrap() - 1.0).abs() < 1e-15);
    }

    /// A workload whose chip boundaries are unknown has an unknown number of crossings, and zero is
    /// the answer that flatters it.
    #[test]
    fn chip_crossings_are_none_when_the_chip_size_is_not_stated() {
        let net = chain(4);
        let f = Fabric::Mesh2D { cols: 4, rows: 1 };
        let p = Partition::new(&net, (0..4).collect(), 4).unwrap();
        let h = spike_hops(&net, &p, &f, None, &[1; 4]).unwrap();
        assert_eq!(h.chip_crossings, None);
        assert_eq!(h.multicast_hops, 3);
        assert_eq!(
            spike_hops(&net, &p, &f, Some(0), &[1; 4]).unwrap_err(),
            MapError::ZeroCapacity { which: "cores per chip" }
        );
        assert_eq!(
            spike_hops(&net, &p, &f, Some(2), &[1; 3]).unwrap_err(),
            MapError::LengthMismatch { what: "spike count", got: 3, neurons: 4 }
        );
        // A partition over more cores than the fabric has is refused rather than wrapped.
        let wide = Partition::new(&net, vec![0, 1, 2, 3], 9).unwrap();
        assert_eq!(
            spike_hops(&net, &wide, &Fabric::Mesh2D { cols: 2, rows: 2 }, None, &[1; 4])
                .unwrap_err(),
            MapError::CoreNotOnFabric { core: 8, n_cores: 4 }
        );
    }

    // ----------------------------------------------------------------------------------------
    // Prices: the refusal
    // ----------------------------------------------------------------------------------------

    /// The finding, asserted. If someone later fills in a hop price without a source, this test is
    /// where the argument has to happen.
    #[test]
    fn no_fabric_price_table_in_this_crate_prices_a_hop() {
        assert_eq!(FABRIC_CATALOGUE.len(), 3);
        for (name, p) in FABRIC_CATALOGUE {
            assert!(p.e_hop_on_chip.is_none(), "{name} gained an on-chip hop price");
            assert!(p.e_hop_chip_crossing.is_none(), "{name} gained a crossing price");
            assert!(p.e_packet_inject.is_none(), "{name} gained an injection price");
            assert!(!p.is_complete(), "{name} claims to be complete");
            assert_eq!(p.unpriced().len(), 3);
            assert_eq!(p.evidence, Evidence::Unstated);
            assert!(!p.source.is_empty(), "{name} has no provenance string");
        }
        // Each table says what was looked for, not merely that nothing was found.
        assert!(SPINNAKER_FABRIC.source.contains("Furber"));
        assert!(LOIHI_FABRIC.source.contains("Davies"));
        assert!(UNSTATED_FABRIC.source.contains("no fabric"));
        // ⛔ And the catalogue's rows ARE the tables they are named after. Every assertion above
        // holds of a catalogue that lists one part twice and another not at all, because all
        // three tables are equally empty — the sweep above checks the shape of each row and never
        // that the row named `loihi` is Loihi's.
        //
        // Reached by NAME rather than by a constant index, for two reasons. `FABRIC_CATALOGUE[2]`
        // is a constant index into a constant array, so a catalogue that LOSES a row would stop
        // compiling rather than fail, and a mutation that cannot be compiled is one this audit
        // learns nothing from. And writing the rows out as a list here would put the text of a
        // catalogue row in this file twice, which is exactly what stops the harness applying the
        // edit that changes it.
        let names: Vec<&str> = FABRIC_CATALOGUE.iter().map(|&(n, _)| n).collect();
        assert_eq!(names, vec!["unstated", "spinnaker", "loihi"], "every row, in order, once");
        let row = |name: &str| {
            FABRIC_CATALOGUE.iter().find(|&&(n, _)| n == name).expect("catalogue row missing").1
        };
        assert_eq!(row("unstated"), UNSTATED_FABRIC);
        assert_eq!(row("spinnaker"), SPINNAKER_FABRIC);
        assert_eq!(row("loihi"), LOIHI_FABRIC);
        assert!(row("spinnaker").source.contains("Furber"), "the spinnaker row is SpiNNaker's");
        assert!(row("loihi").source.contains("Davies"), "the loihi row is Loihi's");
        assert!(row("unstated").source.contains("no fabric"));
    }

    /// A bill refuses and names every term it could not price, so a caller cannot obtain a total
    /// without also being handed the list of what is missing from it.
    #[test]
    fn a_hop_bill_refuses_and_names_every_term_it_could_not_price() {
        let h = SpikeHops {
            multicast_hops: 100,
            unicast_hops: 400,
            chip_crossings: Some(10),
            on_core_deliveries: 5,
            off_core_deliveries: 95,
            spikes: 50,
            sources: 7,
        };
        let bill = h.bill(&UNSTATED_FABRIC);
        assert_eq!(bill.total, None);
        assert_eq!(bill.unpriced, vec!["on-chip hop", "chip-crossing hop", "packet injection"]);
        assert_eq!(bill.evidence, Evidence::Unstated);
        assert!(bill.to_string().starts_with("REFUSED"));

        // With the chip size unknown the split itself is unpriceable, and the total refuses even
        // though a caller might have supplied every joule.
        let priced = HopPrices {
            e_hop_on_chip: Some(1e-12),
            e_hop_chip_crossing: Some(1e-9),
            e_packet_inject: Some(1e-13),
            source: "a fabricated table, for this test only. Not a measurement of anything.",
            evidence: Evidence::Projected,
        };
        let unsplit = SpikeHops { chip_crossings: None, ..h };
        let b2 = unsplit.bill(&priced);
        assert_eq!(b2.total, None);
        assert!(b2.unpriced.contains(&"chip-boundary split (cores per chip not stated)"));
        // ⛔ And NOTHING is attributed to either side of a split that is not known: not the 100
        // hops to the die, not one of them to the package. The refused total was the only thing
        // asserted here, so charging every hop as on-chip — the answer that flatters a placement
        // by pricing its traffic at the cheap end — left `on_chip` reading `1.0e-10` with no test
        // looking at it.
        assert_eq!(b2.on_chip, Some(0.0), "an unknown split attributes no hop to the die");
        assert_eq!(b2.crossings, Some(0.0));
        assert_eq!(b2.injection, Some(50.0 * 1e-13), "the injections are still countable");
    }

    /// And when every term does have a price, the bill is exactly the counts multiplied out — with
    /// the on-chip term being the hops that did **not** cross, not all of them.
    #[test]
    fn a_priced_fabric_bills_exactly_the_counts_it_was_given() {
        let h = SpikeHops {
            multicast_hops: 100,
            unicast_hops: 400,
            chip_crossings: Some(10),
            on_core_deliveries: 5,
            off_core_deliveries: 95,
            spikes: 50,
            sources: 7,
        };
        let priced = HopPrices {
            e_hop_on_chip: Some(1e-12),
            e_hop_chip_crossing: Some(1e-9),
            e_packet_inject: Some(1e-13),
            source: "a fabricated table, for this test only. Not a measurement of anything.",
            evidence: Evidence::Projected,
        };
        let bill = h.bill(&priced);
        assert!(bill.unpriced.is_empty());
        assert_eq!(bill.on_chip, Some(90.0 * 1e-12), "90 hops stayed on the die, not 100");
        assert_eq!(bill.crossings, Some(10.0 * 1e-9));
        assert_eq!(bill.injection, Some(50.0 * 1e-13));
        let want = 90.0 * 1e-12 + 10.0 * 1e-9 + 50.0 * 1e-13;
        assert!((bill.total.unwrap() - want).abs() < 1e-24);
        assert_eq!(bill.evidence, Evidence::Projected);
        // The crossing term dominates by three orders of magnitude at these prices, which is the
        // whole argument for caring where a neuron lands.
        assert!(bill.crossings.unwrap() > 100.0 * bill.on_chip.unwrap());

        // ⛔ A price has to be a NUMBER. An infinite or `NaN` per-hop energy is a table that was
        // filled in by arithmetic that overflowed or divided by zero, and admitting it hands back
        // a total of `inf` or `NaN` with `unpriced` empty — a refusal with no reason, in the
        // shape this module refuses everywhere else. Every table in this crate is `None`, so the
        // finiteness test was never reached by any fixture.
        for bad in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
            let table = HopPrices { e_hop_on_chip: Some(bad), ..priced };
            let b = h.bill(&table);
            assert_eq!(b.on_chip, None, "{bad} was accepted as a price");
            assert_eq!(b.total, None);
            assert_eq!(b.unpriced, vec!["on-chip hop"]);
            // The other two terms were priced, so the grade of the prices used still stands.
            assert_eq!(b.evidence, Evidence::Projected);
        }

        // ⛔ A bill that charged NOTHING used no price, so it carries no grade: `Unstated`, not
        // whatever the table it was handed claims. Every bill in this module either priced
        // something or was handed an empty table, where the two answers coincide.
        let net = chain(4);
        let silent = spike_hops(
            &net,
            &Partition::single_core(&net).unwrap(),
            &Fabric::Crossbar { cores: 2 },
            Some(2),
            &[0; 4],
        )
        .unwrap();
        assert_eq!((silent.multicast_hops, silent.spikes, silent.chip_crossings), (0, 0, Some(0)));
        let nothing = silent.bill(&priced);
        assert_eq!(nothing.total, Some(0.0), "no work, no charge, and no refusal either");
        assert!(nothing.unpriced.is_empty());
        assert_eq!(
            nothing.evidence,
            Evidence::Unstated,
            "a bill that used no price cannot carry the table's grade"
        );

        // A term with a zero count is not charged and its missing price is not held against the
        // total: a workload that never crossed a chip does not need a crossing price.
        let local = SpikeHops { multicast_hops: 8, chip_crossings: Some(0), spikes: 4, ..h };
        let partial = HopPrices { e_hop_chip_crossing: None, ..priced };
        let b = local.bill(&partial);
        assert_eq!(b.total, Some(8.0 * 1e-12 + 4.0 * 1e-13));
        assert!(b.unpriced.is_empty());
    }

    /// Every `Display` implementation prints its numbers rather than a category. Exercised because
    /// a formatter is code, and this crate has already shipped one that printed
    /// 18446744073709551612 for a subtraction that should have saturated.
    #[test]
    fn every_display_impl_names_its_numbers() {
        let net = two_cliques(6);
        let limits = CoreLimits::new(Some(6), Some(1024), Some(32));
        let plan = partition_refined(&net, &limits, 2, 3, 20).unwrap();
        let s = plan.to_string();
        assert!(s.contains("Kernighan-Lin"), "{s}");
        assert!(s.contains("cut 1 synapses"), "{s}");
        assert!(partition_greedy(&net, &limits, 2, 3).unwrap().to_string().contains("greedy"));
        assert_eq!(Method::Greedy.to_string(), "streaming greedy");

        let f = plan.partition.check(&net, &limits).unwrap().to_string();
        assert!(f.contains("PLACES"), "{f}");
        assert!(f.contains("worst core"), "{f}");

        assert!(
            MapError::FanInExceedsCore { neuron: 3, fan_in: 90, cap: 64 }
                .to_string()
                .contains("neuron 3 has 90 inputs and a core stores 64")
        );
        assert!(
            MapError::RouteUnionIsNotATree { root: 2, cores: 5, edges: 6 }
                .to_string()
                .contains("a tree needs exactly 4")
        );
        assert!(MapError::NoCores.to_string().contains("at least one core"));
        assert!(
            MapError::RefinementTooLarge { neurons: 9, n_cores: 9, cells: 81, limit: 8 }
                .to_string()
                .contains("81 cells")
        );
        assert!(
            MapError::CoreNotOnFabric { core: 4, n_cores: 4 }.to_string().contains("which has 4")
        );
        assert!(
            MapError::NoFeasibleCore { neuron: 1, in_degree: 2, n_cores: 3 }
                .to_string()
                .contains("ran out of room at neuron 1")
        );
        assert!(
            MapError::CountOverflow { neuron: 6, what: "unicast hops" }.to_string().contains("neuron 6")
        );
        assert!(
            MapError::NotEnoughSynapseRoom { synapses: 100, n_cores: 2, synapses_per_core: 10, capacity: 20 }
                .to_string()
                .contains("20 total")
        );
        assert!(
            MapError::TooManyCores { n_cores: 99, limit: 7 }.to_string().contains("99 cores")
        );
        assert!(
            MapError::NotEnoughRoom {
                neurons: 10,
                n_cores: 3,
                neurons_per_core: 2,
                capacity: 6
            }
            .to_string()
            .contains("(6 total)")
        );
        assert!(
            MapError::LengthMismatch { what: "spike count", got: 1, neurons: 2 }
                .to_string()
                .contains("spike count has 1 entries")
        );
        assert!(MapError::ZeroCapacity { which: "cores per chip" }.to_string().contains("nothing"));
    }

    /// Refinement refuses a working set it will not allocate, with the allocation in the error.
    #[test]
    fn refinement_refuses_a_working_set_it_will_not_allocate() {
        let net = chain(4000);
        let err = partition_refined(&net, &CoreLimits::UNLIMITED, 4000, 0, 1).unwrap_err();
        assert_eq!(
            err,
            MapError::RefinementTooLarge {
                neurons: 4000,
                n_cores: 4000,
                cells: 16_000_000,
                limit: super::REFINEMENT_CELL_LIMIT,
            }
        );
        // The same problem on fewer cores is inside the limit.
        assert!(partition_refined(&net, &CoreLimits::UNLIMITED, 4, 0, 2).is_ok());
    }

    /// ⛔ THE WALL IS REPORTED PER NEURON, WHATEVER THE PLACEMENT. Two neurons over the fan-in cap
    /// on one core used to produce one bind, and the same two on different cores produced two —
    /// so a caller who fixed the named neuron and re-ran was handed the next one.
    #[test]
    fn every_neuron_over_the_fan_in_wall_is_named_however_it_is_placed() {
        let mut b = NetBuilder::new(22);
        for pre in 0..10u32 {
            b.connect(pre, 20, 1e-3, 1).unwrap();
            b.connect(pre + 10, 21, 1e-3, 1).unwrap();
        }
        let net = b.build();
        let lim = CoreLimits::new(None, None, Some(4));
        let same = Partition::single_core(&net).unwrap().check(&net, &lim).unwrap();
        let mut split_cores = vec![0u32; 22];
        split_cores[21] = 1;
        let split = Partition::new(&net, split_cores, 2).unwrap().check(&net, &lim).unwrap();
        for f in [&same, &split] {
            assert_eq!(f.verdict, Some(false));
            assert_eq!(f.binds.len(), 2, "{:?}", f.binds);
            assert!(matches!(f.binds[0], CoreBind::FanIn { neuron: 20, fan_in: 10, cap: 4, .. }));
            assert!(matches!(f.binds[1], CoreBind::FanIn { neuron: 21, fan_in: 10, cap: 4, .. }));
            assert!(!f.repartitionable());
        }
        assert!(matches!(split.binds[1], CoreBind::FanIn { core: 1, .. }), "the bind names the core the neuron sits on");

        // An empty network checks nothing and therefore passes nothing.
        let empty = NetBuilder::new(0).build();
        let f = Partition::single_core(&empty).unwrap().check(&empty, &CoreLimits::new(Some(4), Some(4), Some(4))).unwrap();
        assert_eq!(f.verdict, None, "a vacuous pass");
        assert!(f.headroom.is_empty() && f.binds.is_empty());
        let g = Partition::single_core(&empty).unwrap().check(&empty, &CoreLimits::UNLIMITED).unwrap();
        assert_eq!(g.unchecked.len(), 3);
    }

    /// ⛔ THE FENNEL PENALTY, AGAINST THE PAPER'S RULE RE-IMPLEMENTED HERE. Tsourakakis,
    /// Gkantsidis, Radunović and Vojnović (WSDM 2014): for `c(x) = α·x^γ` the marginal cost is
    /// `δc(x) = α·γ·x^(γ−1)`, so at `γ = 3/2` the greedy index is
    /// `|N(v) ∩ S_i| − α·1.5·√|S_i|`. The module shipped `0.75`, half the paper's, and the pinned
    /// greedy vector did not move when it was doubled. This test re-runs the stream — the crate's
    /// own shuffle, the paper's coefficient typed here, the documented tie-break — on a fixture
    /// where the penalty decides some placements, and requires the module to match seed for seed.
    #[test]
    fn the_streaming_pass_applies_the_papers_balance_penalty() {
        // A 9-clique whose members each also feed one of nine singletons: the clique members
        // pile onto one core until the penalty outweighs another neighbour there.
        let mut b = NetBuilder::new(18);
        for i in 0..9u32 {
            for j in 0..9u32 {
                if i != j {
                    b.connect(i, j, 1e-3, 1).unwrap();
                }
            }
            b.connect(i, 9 + i, 1e-3, 1).unwrap();
        }
        let net = b.build();
        let k = 2usize;
        let n_f = net.n as f64;
        let alpha = (k as f64).sqrt() * net.n_syn as f64 / (n_f * n_f.sqrt());
        let mut nbrs: Vec<Vec<usize>> = vec![Vec::new(); net.n];
        for pre in 0..net.n {
            for kk in net.offset[pre]..net.offset[pre + 1] {
                let post = net.post[kk] as usize;
                if post != pre {
                    nbrs[pre].push(post);
                    nbrs[post].push(pre);
                }
            }
        }
        let mut decided = 0usize;
        for seed in 0..12u64 {
            let mut order: Vec<usize> = (0..net.n).collect();
            let mut rng = crate::rng::Rng::new(seed);
            for i in (1..order.len()).rev() {
                let j = rng.below((i + 1) as u32) as usize;
                order.swap(i, j);
            }
            let mut core_of = vec![usize::MAX; net.n];
            let mut occupancy = [0u64; 2];
            for &v in &order {
                let mut score = [0i64; 2];
                for &u in &nbrs[v] {
                    if core_of[u] != usize::MAX {
                        score[core_of[u]] += 1;
                    }
                }
                let (mut best, mut best_s) = (usize::MAX, f64::NEG_INFINITY);
                for c in 0..k {
                    let s = score[c] as f64 - alpha * 1.5 * (occupancy[c] as f64).sqrt();
                    let better = best == usize::MAX
                        || s > best_s
                        || (s == best_s && (occupancy[c], c) < (occupancy[best], best));
                    if better {
                        best = c;
                        best_s = s;
                    }
                }
                if score[best] < *score.iter().max().unwrap() {
                    decided += 1;
                }
                core_of[v] = best;
                occupancy[best] += 1;
            }
            let plan = partition_greedy(&net, &CoreLimits::UNLIMITED, 2, seed).unwrap();
            let got: Vec<usize> = plan.partition.core_of.iter().map(|&c| c as usize).collect();
            assert_eq!(got, core_of, "seed {seed}: the module's stream disagrees with the paper's rule");
        }
        assert!(decided > 0, "the penalty never decided a placement; this fixture cannot see the coefficient");
    }

    /// ⛔ A workload with nothing on the fabric has a total, even when the chip size is unknown.
    /// The first version refused it with `unpriced` empty — a refusal with no reason.
    #[test]
    fn a_workload_with_no_fabric_traffic_bills_a_total_even_without_a_chip_size() {
        let net = chain(2);
        let p = Partition::single_core(&net).unwrap();
        let h = spike_hops(&net, &p, &Fabric::Crossbar { cores: 2 }, None, &[3, 0]).unwrap();
        assert_eq!((h.multicast_hops, h.chip_crossings, h.spikes), (0, None, 3));
        let prices = HopPrices {
            e_hop_on_chip: Some(1e-12),
            e_hop_chip_crossing: Some(1e-11),
            e_packet_inject: Some(2e-12),
            source: "test",
            evidence: Evidence::Projected,
        };
        let bill = h.bill(&prices);
        assert!(bill.unpriced.is_empty(), "{:?}", bill.unpriced);
        assert!((bill.total.unwrap() - 6e-12).abs() < 1e-24, "three injections and nothing on the fabric");
        // With traffic and no chip size, the split IS unpriced and the total is still refused.
        let q = Partition::new(&net, vec![0, 1], 2).unwrap();
        let g = spike_hops(&net, &q, &Fabric::Crossbar { cores: 2 }, None, &[3, 0]).unwrap();
        let bill = g.bill(&prices);
        assert_eq!(bill.total, None);
        assert!(bill.unpriced.iter().any(|u| u.starts_with("chip-boundary split")));
        // The unpriced list of an empty table, in the order the bill lists them.
        let none = HopPrices { e_hop_on_chip: None, e_hop_chip_crossing: None, e_packet_inject: None, source: "", evidence: Evidence::Unstated };
        assert_eq!(none.unpriced(), vec!["on-chip hop", "chip-crossing hop", "packet injection"]);
    }

    /// ⛔ A spike count the signature accepts must not wrap the hop counts.
    #[test]
    fn a_spike_count_that_overflows_the_hop_counts_is_refused_by_neuron_and_name() {
        let net = chain(2);
        let f = Fabric::Mesh2D { cols: 2, rows: 2 };
        let p = Partition::new(&net, vec![0, 3], 4).unwrap();
        assert_eq!(
            spike_hops(&net, &p, &f, Some(2), &[u64::MAX, 0]).unwrap_err(),
            MapError::CountOverflow { neuron: 0, what: "multicast hops" }
        );
        // A second neuron that pushes an already large total over the edge is named as itself.
        let mut b = NetBuilder::new(3);
        b.connect(0, 2, 1e-3, 1).unwrap();
        b.connect(1, 2, 1e-3, 1).unwrap();
        let two = b.build();
        let q = Partition::new(&two, vec![0, 0, 3], 4).unwrap();
        assert_eq!(
            spike_hops(&two, &q, &f, None, &[u64::MAX / 2, u64::MAX / 2 + 1, 0]).unwrap_err(),
            MapError::CountOverflow { neuron: 1, what: "multicast hops" }
        );
        assert!(spike_hops(&two, &q, &f, None, &[u64::MAX / 4, u64::MAX / 4, 0]).is_ok());
    }

    /// A triangular torus past `u32` cores is refused rather than swept with a truncated cast.
    #[test]
    fn a_triangular_torus_past_u32_cores_has_no_diameter_rather_than_a_wrong_one() {
        let huge = Fabric::TriangularTorus { cols: 100_000, rows: 100_000 };
        assert_eq!(huge.n_cores(), 10_000_000_000);
        assert_eq!(huge.diameter(), None);
        assert_eq!(Fabric::Torus2D { cols: 100_000, rows: 100_000 }.diameter(), Some(100_000));
        // The empirical closed form the doc records, at the sizes it was seen.
        for (side, want) in [(8u32, 5u64), (16, 10), (32, 21), (64, 42)] {
            assert_eq!(Fabric::TriangularTorus { cols: side, rows: side }.diameter(), Some(want), "{side}x{side}");
        }
    }

    /// The provenance string for the module's headline absence claim carries the right pages:
    /// 652–665, not 652–673 (which was the *Science* `TrueNorth` range two lines away).
    #[test]
    fn the_spinnaker_citation_has_the_right_page_range() {
        assert!(SPINNAKER_FABRIC.source.contains("102(5):652-665, 2014"), "{}", SPINNAKER_FABRIC.source);
        assert!(!SPINNAKER_FABRIC.source.contains("673"));
    }

    /// ⛔ Self-synapses and parallel synapses reach the partitioners. `Adjacency::build` skips
    /// self-synapses and sorts every neighbour list — the sort is what `multiplicity`'s binary
    /// search needs — and both could be deleted with every test green, because no fixture had a
    /// self-synapse or came out unsorted. These are the auditor's probes, kept.
    #[test]
    fn the_partitioners_on_a_network_with_self_synapses_and_a_multigraph() {
        let mut b = NetBuilder::new(8);
        for i in 0..8u32 {
            b.connect(i, i, 1e-3, 1).unwrap();
            b.connect(i, (i + 1) % 8, 1e-3, 1).unwrap();
        }
        let net = b.build();
        let lim = CoreLimits::new(Some(4), None, None);
        for seed in 0..6u64 {
            let g = partition_greedy(&net, &lim, 2, seed).unwrap();
            let k = partition_refined(&net, &lim, 2, seed, 20).unwrap();
            assert_eq!(k.cut, k.partition.cut_edges(&net).unwrap());
            assert_eq!(k.cut_before_refinement - k.cut, k.refinement_gain, "seed {seed}");
            assert!(k.cut <= g.cut, "seed {seed}: refinement made it worse");
        }
        let mut b = NetBuilder::new(12);
        for i in 0..12u32 {
            for _ in 0..3 {
                b.connect(i, (i + 1) % 12, 1e-3, 1).unwrap();
            }
        }
        let multi = b.build();
        let lim = CoreLimits::new(Some(6), None, None);
        for seed in 0..6u64 {
            let k = partition_refined(&multi, &lim, 2, seed, 20).unwrap();
            assert_eq!(k.cut_before_refinement - k.cut, k.refinement_gain, "seed {seed}");
            assert_eq!(k.cut, k.partition.cut_edges(&multi).unwrap());
            assert!(k.cut.is_multiple_of(3), "seed {seed}: a cut of {} splits a triple", k.cut);
        }
    }

    /// ⛔ The refiner's synapse-cap guard BINDS here. Every other `partition_refined` call passes
    /// `None` or a cap too loose to matter, so the guard could be disabled with every test green.
    /// Four hubs of in-degree 9 among 36 leaves of in-degree 1 or 2, ten neurons a core: an
    /// exchange that moves a hub onto a leaf's core moves nine synapses with it, and with the cap
    /// a few synapses above the balanced load such an exchange is exactly what the guard refuses.
    #[test]
    fn refinement_never_ships_an_infeasible_placement() {
        let mut b = NetBuilder::new(40);
        for leaf in 4..40u32 {
            b.connect(leaf, leaf % 4, 1e-3, 1).unwrap();
            b.connect(leaf, if leaf + 1 < 40 { leaf + 1 } else { 4 }, 1e-3, 1).unwrap();
        }
        let net = b.build();
        let mut shipped = 0;
        let mut refined = 0;
        for cap in [20u64, 22, 24, 26, 30] {
            let lim = CoreLimits::new(Some(10), Some(cap), None);
            for seed in 0..8u64 {
                if let Ok(plan) = partition_refined(&net, &lim, 4, seed, 20) {
                    let v = plan.partition.check(&net, &lim).unwrap();
                    assert_eq!(v.verdict, Some(true), "cap {cap} seed {seed}: shipped a placement that binds: {:?}", v.binds);
                    shipped += 1;
                    if plan.swaps_kept > 0 {
                        refined += 1;
                    }
                }
            }
        }
        assert!(shipped >= 8, "only {shipped} placements shipped; the sweep barely runs");
        assert!(refined >= 4, "only {refined} placements were actually refined; the guard was never asked");
    }

    /// The streaming tie-break — emptier core first, then lower index — on a network with no
    /// synapses, where every score is zero and only the tie-break decides. It must alternate;
    /// "lower index first" would pile every neuron onto core 0, which the doc says it prevents.
    #[test]
    fn the_stream_tie_break_spreads_a_zero_signal_prefix() {
        let net = NetBuilder::new(11).build();
        let plan = partition_greedy(&net, &CoreLimits::UNLIMITED, 3, 0).unwrap();
        let mut count = [0usize; 3];
        for &c in &plan.partition.core_of {
            count[c as usize] += 1;
        }
        assert_eq!(count, [4, 4, 3], "eleven silent neurons over three cores, round-robin");
        assert_eq!(plan.partition.used_cores(), 3);
    }

    // ----------------------------------------------------------------------------------------
    // Added by the mutation audit: the claims nothing could see
    // ----------------------------------------------------------------------------------------

    /// A core's `(x, y)` is row-major — `x` counts along the row, `y` counts rows — and the pair
    /// has to rebuild the index it came from. The only fixture that read `coords` before was a
    /// 3x3 mesh at core 8, which is `(2, 2)`: a square fabric read at a point on its diagonal,
    /// where transposing the two axes is the identity.
    #[test]
    fn a_cores_coordinates_are_row_major_and_rebuild_its_index() {
        for f in fabrics() {
            let Some((cols, rows)) = f.dims() else {
                assert_eq!(f.coords(0), None, "{f:?}: a crossbar has no geometry");
                continue;
            };
            for c in 0..f.n_cores() as u32 {
                let (x, y) = f.coords(c).unwrap();
                assert!(x < cols, "{f:?} core {c}: x = {x} is outside {cols} columns");
                assert!(y < rows, "{f:?} core {c}: y = {y} is outside {rows} rows");
                assert_eq!(y * cols + x, c, "{f:?} core {c}: ({x}, {y}) does not rebuild it");
            }
        }
        // Hand values on a fabric five wide and four tall, off the diagonal both ways.
        let f = Fabric::Mesh2D { cols: 5, rows: 4 };
        assert_eq!(f.coords(7), Some((2, 1)));
        assert_eq!(f.coords(3), Some((3, 0)));
        assert_eq!(f.coords(15), Some((0, 3)));
        assert_eq!(f.coords(19), Some((4, 3)));
    }

    /// A root the fabric does not have is refused **as the root**. A destination past the end was
    /// already refused twice over — once by this check and once by `route` returning `None` a few
    /// lines later, which reports the same error — so admitting exactly one index past the end
    /// changed nothing any assertion could reach. A root does not go through `route` at all when
    /// there is nothing to send, and the module answered `Ok` with a tree rooted on a core that is
    /// not there.
    #[test]
    fn a_multicast_tree_refuses_a_root_the_fabric_does_not_have() {
        let f = Fabric::Mesh2D { cols: 3, rows: 3 };
        assert_eq!(
            multicast_tree(&f, 9, &[]).unwrap_err(),
            MapError::CoreNotOnFabric { core: 9, n_cores: 9 },
            "a tree with nothing to send is still rooted somewhere"
        );
        // With destinations, the error names the ROOT and not the first destination it tried.
        assert_eq!(
            multicast_tree(&f, 9, &[0, 1]).unwrap_err(),
            MapError::CoreNotOnFabric { core: 9, n_cores: 9 }
        );
        assert_eq!(
            multicast_tree(&f, 9, &[9]).unwrap_err(),
            MapError::CoreNotOnFabric { core: 9, n_cores: 9 }
        );
        // Core 8 is the last one the fabric has, and it roots a tree.
        assert_eq!(multicast_tree(&f, 8, &[]).unwrap().cores, vec![8]);
    }

    /// `CoreLoad::worst_fan_in` is the LARGEST in-degree on the core. Every fixture that read it
    /// had one distinct in-degree per core — a core of one neuron, or a core whose neurons all had
    /// in-degree 1 — and on those the largest and the smallest are the same neuron. A chain's
    /// first neuron has no inputs and the rest have one, so the two differ.
    #[test]
    fn the_worst_fan_in_on_a_core_is_the_largest_in_degree_not_the_smallest() {
        let net = chain(4);
        assert_eq!(net.in_degrees(), vec![0, 1, 1, 1]);
        let one = Partition::single_core(&net).unwrap();
        assert_eq!(
            one.loads(&net).unwrap()[0].worst_fan_in,
            Some((1, 1)),
            "neuron 0 has no inputs; the worst on this core is the first neuron that has one"
        );
        // And with three distinct in-degrees on one core, so that neither end is a tie.
        let mut b = NetBuilder::new(4);
        for &(p, q) in &[(0u32, 1u32), (0, 2), (1, 2), (3, 2)] {
            b.connect(p, q, 1e-3, 1).unwrap();
        }
        let star = b.build();
        assert_eq!(star.in_degrees(), vec![0, 1, 3, 0]);
        let p = Partition::single_core(&star).unwrap();
        assert_eq!(p.loads(&star).unwrap()[0].worst_fan_in, Some((2, 3)));
    }

    /// A neuron whose fan-in exactly equals the wall is inside it, and the fan-in headroom reports
    /// the WORST neuron in the network. The wall was pinned only from the failing side (300 inputs
    /// against 256), and the headroom record was never read at all: every `CoreLimits` in this
    /// module that stated a fan-in cap either bound on it or came from a part that states none.
    #[test]
    fn a_neuron_exactly_at_the_fan_in_wall_places_and_its_headroom_names_the_worst_neuron() {
        let mut b = NetBuilder::new(4);
        for &(p, q) in &[(0u32, 1u32), (0, 2), (1, 2), (3, 2)] {
            b.connect(p, q, 1e-3, 1).unwrap();
        }
        let net = b.build();
        assert_eq!(net.in_degrees(), vec![0, 1, 3, 0], "the worst neuron has three inputs");
        let p = Partition::single_core(&net).unwrap();

        // Exactly at the wall: three inputs into a cap of three places, with nothing to spare.
        let at = p.check(&net, &CoreLimits::new(None, None, Some(3))).unwrap();
        assert_eq!(at.verdict, Some(true), "{:?}", at.binds);
        assert!(at.binds.is_empty());
        assert_eq!(at.headroom.len(), 1);
        assert_eq!(at.headroom[0].constraint, "maximum fan-in per neuron");
        assert_eq!(at.headroom[0].worst_used, 3, "the worst neuron, not the least loaded one");
        assert_eq!(at.headroom[0].cap, 3);
        assert_eq!(at.headroom[0].spare(), 0);
        assert!((at.headroom[0].utilisation().unwrap() - 1.0).abs() < 1e-15);

        // One below it binds, so the boundary is where the module says it is.
        let over = p.check(&net, &CoreLimits::new(None, None, Some(2))).unwrap();
        assert_eq!(over.binds, vec![CoreBind::FanIn { neuron: 2, fan_in: 3, cap: 2, core: 0 }]);

        // And with room above the worst neuron, the headroom still reports the worst.
        let loose = p.check(&net, &CoreLimits::new(None, None, Some(10))).unwrap();
        assert_eq!((loose.headroom[0].worst_used, loose.headroom[0].spare()), (3, 7));
    }

    /// A headroom's worst core is the fullest, and a tie between two equally full cores goes to
    /// the LOWER index. Every headroom fixture in this module had a strict worst — a 3/1 split —
    /// so the tie-break was carried by nothing, and a tie is the common case on a balanced
    /// placement, which is what a partitioner is trying to produce.
    #[test]
    fn a_tie_for_the_worst_core_in_a_headroom_goes_to_the_lower_core_index() {
        let net = cycle(4);
        assert_eq!(net.in_degrees(), vec![1, 1, 1, 1], "every core will carry the same load");
        let p = Partition::new(&net, vec![0, 0, 1, 1], 2).unwrap();
        let f = p.check(&net, &CoreLimits::new(Some(4), Some(8), None)).unwrap();
        assert_eq!(f.verdict, Some(true));
        let by = |name: &str| *f.headroom.iter().find(|h| h.constraint == name).unwrap();
        assert_eq!((by("synapses per core").worst_used, by("synapses per core").worst_core), (2, 0));
        assert_eq!((by("neurons per core").worst_used, by("neurons per core").worst_core), (2, 0));
    }

    /// A core index EQUAL to the core count is past the end: cores are numbered `0..n_cores`. The
    /// only out-of-range fixture in this module named core 5 of 2, which is out of range under
    /// `>=` and under `>` alike, so the first index that is actually out of range — the one a
    /// caller reaches by writing `n_cores` where they meant `n_cores - 1` — was admitted, and the
    /// next call that indexed a per-core array with it would have panicked instead.
    #[test]
    fn a_core_index_equal_to_the_core_count_is_past_the_end() {
        let net = chain(4);
        assert_eq!(
            Partition::new(&net, vec![0, 0, 2, 0], 2).unwrap_err(),
            MapError::CoreOutOfRange { neuron: 2, core: 2, n_cores: 2 }
        );
        assert_eq!(
            Partition::new(&net, vec![1, 1, 1, 1], 1).unwrap_err(),
            MapError::CoreOutOfRange { neuron: 0, core: 1, n_cores: 1 }
        );
        let ok = Partition::new(&net, vec![0, 1, 1, 0], 2).unwrap();
        assert_eq!(ok.n_cores, 2, "index n_cores - 1 is in range");
    }

    /// The per-core bookkeeping bound is `2^24`, and its doc prices that in bytes. Every test that
    /// used the constant wrote `MAX_CORES + 1`, which moves with it, so the value itself could be
    /// cut by a factor of sixteen with every test green — down to 1,048,576, below the 10.6
    /// million cores of `SpiNNaker2`'s full build that the same doc says it is above.
    ///
    /// ⛔ THE DOC'S BYTE COUNT WAS WRONG AND THIS IS WHERE IT WAS CAUGHT. It said a `CoreLoad` is
    /// 40 bytes and the limit is 671 MB. `size_of` measures 48 and 805 MB: the record is a `u32`,
    /// two `u64` and an `Option<(usize, u64)>`, and that option has no niche to hide its
    /// discriminant in, so it costs 24 bytes rather than 16.
    #[test]
    fn the_per_core_bookkeeping_bound_is_the_size_its_doc_prices() {
        assert_eq!(super::MAX_CORES, 1 << 24);
        assert_eq!(super::MAX_CORES, 16_777_216);
        // The machine the constant's doc says it is above: SpiNNaker2's full build, about 10.6
        // million ARM cores. Bound through a binding so the comparison is not folded away.
        let spinnaker2_full_build: u32 = 10_600_000;
        assert!(
            super::MAX_CORES > spinnaker2_full_build,
            "the bound has to stay above SpiNNaker2's full build, which its own doc claims"
        );
        // Measured here, not counted by eye, and this is the number the doc quotes.
        assert_eq!(core::mem::size_of::<super::CoreLoad>(), 48);
        assert_eq!(u64::from(super::MAX_CORES) * 48, 805_306_368, "805 MB of per-core records");
        // The other two allocations the doc prices, from the same measurement.
        assert_eq!(core::mem::size_of::<bool>(), 1, "one byte per core in used_cores");
        assert_eq!(core::mem::size_of::<u64>() * 3, 24, "three u64 per core in the streaming pass");
    }

    /// `single_core` spans ONE core, not one per neuron. Every caller reads `core_of`, which is a
    /// vector of zeros either way, so the core COUNT it was built over was read by nothing: a
    /// partition over `n` cores of which `n - 1` are empty has the same cut, the same
    /// `used_cores`, and the same record for core 0.
    #[test]
    fn the_degenerate_placement_spans_one_core_and_not_one_per_neuron() {
        let net = chain(6);
        let p = Partition::single_core(&net).unwrap();
        assert_eq!(p.n_cores, 1);
        assert_eq!(p.loads(&net).unwrap().len(), 1, "one per-core record, not one per neuron");
        assert_eq!(p.used_cores(), 1);
        // And an empty network's degenerate placement is over one core, not over zero.
        let empty = NetBuilder::new(0).build();
        assert_eq!(Partition::single_core(&empty).unwrap().n_cores, 1);
    }

    /// ⛔ `Adjacency` is the structure both partitioners work over, and three of its properties
    /// were carried by no test. A self-synapse is DROPPED — counted as well as skipped would leave
    /// two unwritten slots per self-loop, which read as neuron 0 and make it every self-looping
    /// neuron's phantom neighbour. Parallel synapses are KEPT, as multiplicity. Every neighbour
    /// list is SORTED, because `multiplicity` binary-searches it. The partitioner fixtures cannot
    /// see any of it: they recount their cut from `Net`, never from this structure, so an
    /// adjacency that disagreed with the network still agreed with itself.
    #[test]
    fn the_undirected_adjacency_drops_self_synapses_keeps_multiplicity_and_stays_sorted() {
        let mut b = NetBuilder::new(5);
        b.connect(4, 0, 1e-3, 1).unwrap();
        b.connect(1, 4, 1e-3, 1).unwrap();
        b.connect(1, 4, 2e-3, 2).unwrap();
        b.connect(2, 2, 1e-3, 1).unwrap();
        b.connect(0, 3, 1e-3, 1).unwrap();
        let net = b.build();
        assert_eq!(net.n_syn, 5);
        let adj = super::Adjacency::build(&net);
        // Four of the five synapses are not self-synapses, and each puts one entry at each end.
        assert_eq!(adj.nbr.len(), 8);
        assert_eq!(adj.of(0), &[3, 4]);
        assert_eq!(adj.of(1), &[4, 4], "two parallel synapses are two entries, not one");
        assert!(adj.of(2).is_empty(), "the self-synapse is dropped at both ends");
        assert_eq!(adj.of(3), &[0]);
        // Neuron 4 is the one whose entries ARRIVE out of order: neuron 1's two synapses are
        // written into its block before its own synapse to neuron 0 is. Unsorted the block reads
        // [1, 1, 0], and the binary search below then answers 3 where the answer is 2.
        assert_eq!(adj.of(4), &[0, 1, 1]);
        assert_eq!(adj.multiplicity(4, 1), 2);
        assert_eq!(adj.multiplicity(4, 0), 1);
        assert_eq!(adj.multiplicity(4, 2), 0);
        assert_eq!(adj.multiplicity(2, 0), 0, "a neuron with no neighbours shares no edge");
        assert_eq!(adj.multiplicity(1, 4), 2, "multiplicity is symmetric");

        // Sorted, self-free and symmetric on the fixtures the partitioners actually run on.
        for net in [two_cliques(6), cycle(9), chain(7)] {
            let adj = super::Adjacency::build(&net);
            let mut entries = 0usize;
            for v in 0..net.n {
                let s = adj.of(v);
                entries += s.len();
                assert!(s.windows(2).all(|w| w[0] <= w[1]), "neuron {v}: {s:?} is unsorted");
                assert!(!s.contains(&(v as u32)), "neuron {v} is listed as its own neighbour");
                for &u in s {
                    assert_eq!(
                        adj.multiplicity(u as usize, v as u32),
                        adj.multiplicity(v, u),
                        "neurons {v} and {u} disagree about how many edges join them"
                    );
                }
            }
            assert_eq!(entries, 2 * net.n_syn, "one entry at each end of every synapse");
        }
    }

    /// A streaming plan names the streaming method and reports the cut it made as the cut before
    /// refinement. `Display` was the only witness for the method, and "streaming greedy +
    /// Kernighan-Lin" CONTAINS the word "greedy" — so a greedy plan labelled as refined read the
    /// same to the one assertion looking at it. The `cut_before_refinement` field was compared
    /// only on refined plans, where it comes from a different line.
    #[test]
    fn a_streaming_plan_names_the_streaming_method_and_its_own_cut() {
        let net = two_cliques(9);
        let limits = CoreLimits::new(Some(9), None, None);
        for seed in 0..6u64 {
            let g = partition_greedy(&net, &limits, 2, seed).unwrap();
            assert_eq!(g.method, Method::Greedy, "seed {seed}");
            assert_eq!(
                g.cut_before_refinement, g.cut,
                "seed {seed}: a plan that never refined starts where it ends"
            );
            assert_eq!((g.refinement_gain, g.passes_run, g.swaps_kept), (0, 0, 0), "seed {seed}");
            let s = g.to_string();
            assert!(!s.contains("Kernighan"), "{s}");
            assert!(s.starts_with("streaming greedy on 2 cores: cut "), "{s}");
            assert!(!s.contains("pass(es)"), "a greedy plan reports no passes: {s}");
        }
        assert_eq!(Method::KernighanLin.to_string(), "streaming greedy + Kernighan-Lin");
    }

    /// Refinement runs until a pass gains nothing and then stops — not once, and not the whole
    /// budget. `passes_run` was in no assertion, so "stop only on a LOSS" (which never happens,
    /// because a pass's own rewind clamps its gain at zero) and "always stop after one" were both
    /// invisible: the placement is identical either way, since a pass that gains nothing rewinds
    /// everything it tried.
    #[test]
    fn refinement_runs_until_a_pass_gains_nothing_and_then_stops() {
        let net = two_cliques(12);
        let limits = CoreLimits::new(Some(12), None, None);
        // Seed 1 traps the stream at 23: one pass recovers 22, and a second is needed to discover
        // there is nothing left. Seed 0's stream lands on the optimum, so one pass ends it.
        let trapped = partition_refined(&net, &limits, 2, 1, 20).unwrap();
        assert_eq!((trapped.cut_before_refinement, trapped.cut), (23, 1));
        assert_eq!(trapped.passes_run, 2, "one pass that gained 22, one that gained nothing");
        let already = partition_refined(&net, &limits, 2, 0, 20).unwrap();
        assert_eq!((already.cut_before_refinement, already.cut), (1, 1));
        assert_eq!(already.passes_run, 1, "the first pass already gains nothing");
        // A budget below what it would use is honoured, and a budget of zero refines nothing.
        assert_eq!(partition_refined(&net, &limits, 2, 1, 1).unwrap().passes_run, 1);
        let none = partition_refined(&net, &limits, 2, 1, 0).unwrap();
        assert_eq!((none.passes_run, none.swaps_kept, none.cut), (0, 0, 23));
    }

    /// A pass keeps a prefix of its exchanges only when the prefix's cumulative gain is POSITIVE,
    /// so a plan that gained nothing kept nothing. The suite asserted the forward implication —
    /// a gain means an exchange was kept — and never its converse, so a rewind point that kept the
    /// best prefix even at zero churned twelve exchanges through a placement that ends exactly
    /// where it started, with `refinement_gain` reporting 0 for all of it.
    #[test]
    fn a_plan_that_gained_nothing_kept_no_exchange() {
        let cases: Vec<(&str, Net, u32, CoreLimits)> = vec![
            ("two cliques", two_cliques(12), 2, CoreLimits::new(Some(12), None, None)),
            ("cycle", cycle(32), 2, CoreLimits::new(Some(16), None, None)),
            ("chain on four cores", chain(40), 4, CoreLimits::new(Some(10), None, None)),
        ];
        let mut idle = 0;
        for (name, net, cores, lim) in &cases {
            for seed in 0..8u64 {
                let plan = partition_refined(net, lim, *cores, seed, 20).unwrap();
                assert_eq!(
                    plan.refinement_gain == 0,
                    plan.swaps_kept == 0,
                    "{name} seed {seed}: gain {} against {} exchange(s) kept",
                    plan.refinement_gain,
                    plan.swaps_kept
                );
                if plan.refinement_gain == 0 {
                    idle += 1;
                    assert_eq!(plan.cut, plan.cut_before_refinement, "{name} seed {seed}");
                    assert_eq!(
                        plan.partition.core_of,
                        partition_greedy(net, lim, *cores, seed).unwrap().partition.core_of,
                        "{name} seed {seed}: a plan that gained nothing moved a neuron"
                    );
                }
            }
        }
        assert!(idle >= 3, "only {idle} runs of the sweep gained nothing; the converse is untested");
    }

    /// The streaming pass accepts exactly `MAX_CORES` and refuses one more. The refusal was pinned
    /// at `MAX_CORES + 1` on both entry points; the acceptance at `MAX_CORES` was pinned only on
    /// `Partition::new`, so the streaming pass's copy of the same check could be tightened by one
    /// with every test green. Asked of an EMPTY network, which returns before the per-core
    /// bookkeeping this constant bounds is ever allocated — this pins the boundary of the check,
    /// not the feasibility of a 16-million-core allocation.
    #[test]
    fn the_streaming_pass_accepts_exactly_the_core_count_its_constant_allows() {
        let empty = NetBuilder::new(0).build();
        let plan = partition_greedy(&empty, &CoreLimits::UNLIMITED, super::MAX_CORES, 0).unwrap();
        assert_eq!(plan.partition.n_cores, super::MAX_CORES);
        assert_eq!((plan.cut, plan.partition.neurons), (0, 0));
        assert_eq!(
            partition_greedy(&empty, &CoreLimits::UNLIMITED, super::MAX_CORES + 1, 0).unwrap_err(),
            MapError::TooManyCores { n_cores: super::MAX_CORES + 1, limit: super::MAX_CORES }
        );
    }

    /// When a network fails BOTH the whole-machine synapse arithmetic and the per-neuron wall, the
    /// module reports the arithmetic first — the stated order, because a machine twelve times too
    /// small is the fact that makes every seed and the refiner fail identically, and splitting the
    /// named neuron would not change it. The two checks sit next to each other in one block and no
    /// fixture tripped both, so their order was free.
    #[test]
    fn the_whole_machine_synapse_arithmetic_is_reported_before_the_per_neuron_wall() {
        let mut b = NetBuilder::new(20);
        for i in 0..20u32 {
            for j in 0..5u32 {
                b.connect((i + j + 1) % 20, i, 1e-3, 1).unwrap();
            }
        }
        let dense = b.build();
        assert_eq!(dense.n_syn, 100);
        assert_eq!(dense.in_degrees()[0], 5, "every neuron's five inputs exceed a core of four");
        // 100 synapses; two cores of four store eight. Both refusals are true of this network.
        let both = CoreLimits::new(Some(10), Some(4), None);
        assert_eq!(
            partition_greedy(&dense, &both, 2, 0).unwrap_err(),
            MapError::NotEnoughSynapseRoom {
                synapses: 100,
                n_cores: 2,
                synapses_per_core: 4,
                capacity: 8
            }
        );
        // With 25 cores the machine holds exactly 100, so only the wall is left — and then it is
        // the wall that is reported, naming the neuron.
        assert_eq!(
            partition_greedy(&dense, &both, 25, 0).unwrap_err(),
            MapError::FanInExceedsCore { neuron: 0, fan_in: 5, cap: 4 }
        );
    }

    /// ⛔ THE STREAMING PASS REFUSES RATHER THAN OVERFILLING A CORE'S SYNAPSE STORE. Three hubs of
    /// in-degree three over nine leaves, on two cores that store five synapses each: the machine
    /// has room (nine synapses in ten slots) and no neuron is larger than a core, but two hubs are
    /// six and a core holds five, so exactly one hub fits per core and the third has nowhere to
    /// go — whatever the stream order.
    ///
    /// Every other fixture in this module either states no synapse limit or states one loose
    /// enough that eligibility never refused a core. So the arriving neuron's own storage, the
    /// accumulation of the per-core load, and the eligibility test itself could each be deleted
    /// with every test green, and the pass would ship a placement its own `check` calls illegal.
    #[test]
    fn the_streaming_pass_refuses_rather_than_overfilling_a_cores_synapse_store() {
        let mut b = NetBuilder::new(12);
        for h in 0..3u32 {
            for j in 0..3u32 {
                b.connect(3 + h * 3 + j, h, 1e-3, 1).unwrap();
            }
        }
        let net = b.build();
        assert_eq!(net.n_syn, 9);
        assert_eq!(net.in_degrees()[..3], [3, 3, 3], "three hubs of three");

        let tight = CoreLimits::new(None, Some(5), None);
        for seed in 0..8u64 {
            let err = partition_greedy(&net, &tight, 2, seed).unwrap_err();
            assert!(
                matches!(err, MapError::NoFeasibleCore { in_degree: 3, n_cores: 2, .. }),
                "seed {seed}: {err:?}"
            );
            assert!(partition_refined(&net, &tight, 2, seed, 20).is_err(), "seed {seed}");
        }

        // One more slot per core and the same network places, with a core carrying EXACTLY its
        // two hubs: the test is `load + arriving > cap`, so a core filled to the cap is legal and
        // a core filled past it is not.
        let roomy = CoreLimits::new(None, Some(6), None);
        for seed in 0..8u64 {
            let plan = partition_greedy(&net, &roomy, 2, seed).unwrap();
            let f = plan.partition.check(&net, &roomy).unwrap();
            assert_eq!(f.verdict, Some(true), "seed {seed}: {:?}", f.binds);
            let worst =
                plan.partition.loads(&net).unwrap().iter().map(|l| l.synapses).max().unwrap();
            assert_eq!(worst, 6, "seed {seed}: two hubs of three exactly fill a core of six");
        }
    }

    /// The refiner's incremental state for a fixed assignment: the undirected neighbour
    /// structure, the in-degrees, the per-(neuron, core) neighbour counts and the per-core synapse
    /// load, all computed from scratch.
    ///
    /// It says the same thing as [`super::partition_refined`]'s own initialisation and is written
    /// in different words on purpose — one pass over the neurons, with the two destinations named
    /// — so that it is a recount rather than a copy, and so that the mutation list's anchors on
    /// those two lines still occur exactly once in this file.
    fn kl_state(
        net: &Net,
        core_of: &[u32],
        k: usize,
    ) -> (super::Adjacency, Vec<i64>, Vec<i64>, Vec<i64>) {
        let adj = super::Adjacency::build(net);
        let in_deg: Vec<i64> = net.in_degrees().iter().map(|&d| d as i64).collect();
        let mut cnt = vec![0i64; net.n * k];
        let mut syn = vec![0i64; k];
        for v in 0..net.n {
            let home = core_of[v] as usize;
            syn[home] += in_deg[v];
            for &u in adj.of(v) {
                let there = core_of[u as usize] as usize;
                cnt[v * k + there] += 1;
            }
        }
        (adj, in_deg, cnt, syn)
    }

    /// The six-neuron fixture the two refiner tests below work over: a triangle `{0, 1, 2}` on
    /// core 0 with a leaf 5 hanging off neuron 0, and neuron 3 on core 1 joined to all three of
    /// the triangle and to a leaf 4. In-degrees are 1, 2, 3, 0, 1, 1.
    fn triangle_and_leaves() -> Net {
        let mut b = NetBuilder::new(6);
        for &(p, q) in &[(0u32, 1u32), (0, 2), (1, 2), (0, 5), (3, 0), (3, 1), (3, 2), (3, 4)] {
            b.connect(p, q, 1e-3, 1).unwrap();
        }
        b.build()
    }

    /// ⛔ EXCHANGE CANDIDATES ARE THE BOUNDARY NEURONS ONLY, AND THIS IS WHAT THAT COSTS. A neuron
    /// with every edge inside its own core cannot gain from moving, but it can be a better
    /// PARTNER than any boundary neuron, so the restriction is a cost measure and not a proof that
    /// nothing is lost. Nothing in this module could see the difference: every other refinement
    /// fixture reads only the cut the refiner happens to reach, and both candidate sets reach the
    /// same cut on all of them.
    ///
    /// On `triangle_and_leaves` placed as `{0, 1, 2, 5} | {3, 4}` the cut is 3. Neurons 4 and 5
    /// are interior. Every boundary pair is a loss, and the best of them is `-1`: exchanging
    /// neuron 1 with neuron 3 gives up two of neuron 1's edges at home for the one it has across
    /// (`-1`), brings three of neuron 3's home for the one it leaves behind (`+2`), and the one
    /// edge the two of them share stays cut whichever way they go, which is charged twice (`-2`).
    /// So the pass rewinds to nothing and leaves the placement where it found it. Exchanging the
    /// interior leaf 5 with neuron 3 would gain `+2 - 1 - 0 = 1` — neuron 5 gives up its single
    /// edge at home, and the two share none — and reach a cut of 2, which this pass never
    /// considers.
    #[test]
    fn the_exchange_candidates_are_the_boundary_neurons_only_and_this_is_what_that_costs() {
        let net = triangle_and_leaves();
        let start = vec![0u32, 0, 0, 1, 1, 0];
        let k = 2usize;
        assert_eq!(Partition::new(&net, start.clone(), 2).unwrap().cut_edges(&net).unwrap(), 3);

        let (adj, in_deg, mut cnt, mut syn) = kl_state(&net, &start, k);
        // Neuron 5's only edge is to neuron 0 on its own core, and neuron 4's only edge is to
        // neuron 3 on its own core: both are interior, which is what makes this fixture work.
        assert_eq!((cnt[5 * k], cnt[5 * k + 1]), (1, 0));
        assert_eq!((cnt[4 * k], cnt[4 * k + 1]), (0, 1));

        let mut core_of = start.clone();
        let (gain, kept) = super::kl_pass(&adj, &in_deg, None, k, &mut core_of, &mut cnt, &mut syn);
        assert_eq!((gain, kept), (0, 0), "no boundary pair gains, so the pass keeps nothing");
        assert_eq!(core_of, start, "and the placement is exactly where it started");

        // The exchange the restriction does not consider, priced here rather than asserted away.
        let better = Partition::new(&net, vec![0, 0, 0, 0, 1, 1], 2).unwrap();
        assert_eq!(better.cut_edges(&net).unwrap(), 2, "swapping the interior leaf 5 with 3");
    }

    /// ⛔ An exchange updates three structures at once — the assignment, the per-(neuron, core)
    /// neighbour counts and the per-core synapse load — and only the first two are read back by
    /// anything the suite measured. The synapse load is read ONLY by the capacity guard, and every
    /// refinement fixture either stated no synapse cap or stated one loose enough never to refuse,
    /// so moving the load in the wrong direction changed no answer. Checked here against a
    /// recomputation from scratch, in both directions, because the function's doc says it is its
    /// own inverse.
    #[test]
    fn an_exchange_leaves_every_incremental_structure_equal_to_a_recount() {
        let net = triangle_and_leaves();
        let start = vec![0u32, 0, 0, 1, 1, 0];
        let k = 2usize;
        let (adj, in_deg, mut cnt, mut syn) = kl_state(&net, &start, k);
        assert_eq!(in_deg, vec![1, 2, 3, 0, 1, 1]);
        assert_eq!(syn, vec![7, 1], "core 0 stores 1 + 2 + 3 + 1, core 1 stores 0 + 1");

        let mut core_of = start.clone();
        super::kl_exchange(&adj, &in_deg, k, &mut core_of, &mut cnt, &mut syn, 0, 3);
        assert_eq!(core_of, vec![1, 0, 0, 0, 1, 0]);
        // Neuron 0 stores one synapse and leaves core 0; neuron 3 stores none and arrives.
        assert_eq!(syn, vec![6, 2], "core 0 loses 1 and gains 0; core 1 loses 0 and gains 1");
        let (_, _, want_cnt, want_syn) = kl_state(&net, &core_of, k);
        assert_eq!(cnt, want_cnt, "the neighbour counts drifted from a recount");
        assert_eq!(syn, want_syn, "the synapse loads drifted from a recount");

        // Its own inverse, against a recount of the state it started from.
        super::kl_exchange(&adj, &in_deg, k, &mut core_of, &mut cnt, &mut syn, 0, 3);
        assert_eq!(core_of, start);
        let (_, _, back_cnt, back_syn) = kl_state(&net, &start, k);
        assert_eq!((cnt, syn), (back_cnt, back_syn));
    }

    /// ⛔ THE REFINER'S SYNAPSE GUARD PRICES THE CORE AFTER THE MOVE. `syn[cu] - in_deg[u] +
    /// in_deg[v]` is what core `u` holds once `u` has gone and `v` has arrived. Reading the two
    /// in-degrees the other way round prices the move BACKWARDS — as though `u` were arriving on
    /// its own core and `v` leaving it — and the two answers differ by
    /// `2 * (in_deg[u] - in_deg[v])`, so the guard lets through an exchange that overfills the
    /// core whenever the neuron arriving is the larger of the two.
    ///
    /// `refinement_never_ships_an_infeasible_placement` does not see it, and the honest statement
    /// of why is a measurement rather than an argument: on that fixture the caps are loose enough
    /// that the exchanges the backwards reading admits never actually overfill a core in the
    /// placements the pass reaches. Here they do. This fixture is a 30-cycle whose neurons carry
    /// one input each and whose last neuron carries eleven; at three cores and a cap of 14, seed 2
    /// ships a core holding 24, and at a cap of 16, seeds 1 and 2 ship cores holding 23 and 22.
    #[test]
    fn the_refiners_synapse_guard_prices_the_core_after_the_move() {
        let mut b = NetBuilder::new(30);
        for i in 0..30u32 {
            b.connect(i, (i + 1) % 30, 1e-3, 1).unwrap();
        }
        for i in 0..10u32 {
            b.connect(i, 29, 1e-3, 1).unwrap();
        }
        let net = b.build();
        assert_eq!(net.in_degrees()[29], 11, "one from the cycle and ten from the fan-in");
        let mut refined = 0;
        let mut shipped = 0;
        for cap in [14u64, 16, 18, 20, 22] {
            let lim = CoreLimits::new(None, Some(cap), None);
            for seed in 0..8u64 {
                let Ok(plan) = partition_refined(&net, &lim, 3, seed, 20) else { continue };
                let v = plan.partition.check(&net, &lim).unwrap();
                assert_eq!(
                    v.verdict,
                    Some(true),
                    "cap {cap} seed {seed}: refinement shipped a placement that binds: {:?}",
                    v.binds
                );
                shipped += 1;
                if plan.swaps_kept > 0 {
                    refined += 1;
                }
            }
        }
        // Measured: 30 of the 40 runs place at all. The other ten are the streaming pass refusing
        // — at a cap of 14 or 16 the cycle fills a core before neuron 29's eleven inputs arrive —
        // and every one of the 30 that does place keeps at least one exchange, so the guard is
        // asked on all of them.
        assert_eq!(shipped, 30, "the sweep has to run for the guard to be asked");
        assert_eq!(refined, 30, "every placement that shipped was refined");
    }

    /// A source is a neuron that FIRED and has somewhere to send it, and a spike from a neuron
    /// with no targets is still a spike. Both counters were pinned on a fixture whose only silent
    /// neurons were also the ones with no outgoing synapses, so "every neuron with a target is a
    /// source" and "a spike into nothing never happened" both read correctly there.
    #[test]
    fn a_source_is_a_neuron_that_fired_and_a_spike_into_nothing_is_still_a_spike() {
        let net = chain(4);
        let f = Fabric::Mesh2D { cols: 2, rows: 2 };
        let p = Partition::new(&net, vec![0, 1, 2, 3], 4).unwrap();
        // Neuron 0 fires once and has a target; neurons 1 and 2 have targets and never fire;
        // neuron 3 fires five times and has no target at all.
        let h = spike_hops(&net, &p, &f, Some(2), &[1, 0, 0, 5]).unwrap();
        assert_eq!(h.sources, 1, "two silent neurons have targets and are not sources");
        assert_eq!(h.spikes, 6, "five spikes from a neuron with nowhere to send them still fired");
        assert_eq!(h.multicast_hops, 1, "only neuron 0's spike crossed a link");
        assert_eq!((h.on_core_deliveries, h.off_core_deliveries), (0, 1));
        assert!((h.hops_per_spike().unwrap() - 1.0 / 6.0).abs() < 1e-15);
        // Nothing firing at all is no source and no spike, and the ratio then has no value.
        let quiet = spike_hops(&net, &p, &f, Some(2), &[0; 4]).unwrap();
        assert_eq!((quiet.sources, quiet.spikes), (0, 0));
        assert_eq!(quiet.hops_per_spike(), None);
    }

    /// A plan prints the cores it USED, not the cores it was offered. Every plan the display test
    /// built filled every core it was given — a two-core placement of twelve neurons — so the two
    /// numbers were the same one.
    #[test]
    fn a_plan_prints_the_cores_it_used_rather_than_the_cores_it_was_offered() {
        let net = chain(3);
        let plan = partition_greedy(&net, &CoreLimits::UNLIMITED, 8, 0).unwrap();
        assert_eq!(plan.partition.n_cores, 8, "eight offered");
        assert_eq!(plan.partition.used_cores(), 3, "three neurons cannot occupy eight cores");
        assert!(plan.to_string().contains("on 3 cores"), "{plan}");
        assert!(!plan.to_string().contains("on 8 cores"), "{plan}");
    }
}
