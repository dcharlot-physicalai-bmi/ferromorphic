//! Optimisation by stochastic spiking: quadratic unconstrained binary problems, the Ising-style
//! formulations that turn max-cut and graph colouring into them, and an annealed Glauber sampler
//! whose fixed-temperature statistics are checked against the exact Boltzmann distribution and
//! whose answers are checked against brute force.
//!
//! # What the mechanism is
//!
//! A network of binary stochastic neurons with symmetric couplings has a stationary distribution
//! `p(x) ∝ exp(−β E(x))` over its states, and a quadratic energy
//!
//! ```text
//! E(x) = Σ_i q_ii x_i + Σ_{i<j} q_ij x_i x_j,      x ∈ {0, 1}^n
//! ```
//!
//! is a QUBO — the form every NP-hard combinatorial problem has been written in (Lucas, *Ising
//! formulations of many NP problems*, Frontiers in Physics 2:5, 2014). Let the network run at a
//! temperature and it samples the low-energy states more often; raise `β` slowly and it settles
//! into one of them. That is simulated annealing done by neurons: each unit's flip probability is
//! the logistic of `β` times its local field, which is exactly the membrane-to-firing map of
//! [`crate::bayes`]'s neural sampler, and this module builds on that module's [`Boltzmann`] so
//! the two agree on what a field is.
//!
//! # Why it is in a neuromorphic crate
//!
//! Constraint satisfaction was one of the first workloads run on `SpiNNaker` as a stochastic
//! spiking network (Fonseca Guerra and Furber, *Using stochastic spiking neural networks on
//! `SpiNNaker` to solve constraint satisfaction problems*, Frontiers in Neuroscience 11:714, 2017)
//! and one of the results Intel reports for Loihi (Davies et al., Proceedings of the IEEE
//! 109(5):911–934, 2021, on constraint satisfaction and, in the Lava optimisation library, QUBO);
//! graph partitioning as QUBO was run on both `TrueNorth` and Loihi (Mniszewski, *Graph partitioning
//! as quadratic unconstrained binary optimization (QUBO) on spiking neuromorphic hardware*,
//! ICONS 2019). What the substrate buys is stated in this module's counts: a unit update is one
//! membrane evaluation — `n` synaptic operations at most, its degree in a sparse problem — and a
//! **flip is a spike**; [`Solution::flips`] and [`Solution::evaluations`] are the two integers a
//! bill needs. Nothing here is a measurement of any chip.
//!
//! # The closed forms this module is checked against
//!
//! - The QUBO energy against hand values, and the max-cut mapping against an independent edge
//!   count over every state of a small graph.
//! - Brute-force optima: the 6-cycle cuts 6, `K_5` cuts 6, `K_{3,3}` cuts 9.
//! - At fixed `β` the Glauber sampler's state histogram matches [`Boltzmann::exact`] in total
//!   variation — the dynamics is a correct sampler before it is an optimiser.
//! - The annealer reaches the brute-force optimum on random 12-variable problems at a stated rate.
//! - Graph colouring: the 6-cycle is 2-colourable (energy 0 found), `K_4` is not 3-colourable (the
//!   brute-force minimum is 1, a fact about the chromatic number), the Petersen graph is
//!   3-colourable and the annealer finds a proper colouring, verified by an independent decoder.
//!
//! # What this module has NOT reproduced
//!
//! - Loihi's or `SpiNNaker`'s solvers, their noise sources, or their reported times-to-solution.
//! - Any problem larger than 64 variables: the state is a bitmask, as in [`crate::bayes`]. That is
//!   a teaching-and-verification limit, stated; a production solver keeps a vector.
//! - Parallel tempering, tabu, or any schedule beyond geometric annealing with restarts.

use core::fmt;

use crate::bayes::{BayesError, Boltzmann, logistic};
use crate::rng::Rng;

/// What went wrong, named rather than guessed around.
#[derive(Debug, Clone, PartialEq)]
pub enum OptimiseError {
    /// A count of zero where at least one is needed.
    Empty {
        /// What was empty.
        what: &'static str,
    },
    /// A matrix of the wrong size.
    Dimension {
        /// Which object.
        what: &'static str,
        /// Length supplied.
        got: usize,
        /// Length required.
        want: usize,
    },
    /// A `NaN` or infinity.
    NonFinite {
        /// Which quantity.
        what: &'static str,
        /// Position in the offending array, `0` for a scalar.
        index: usize,
    },
    /// A parameter outside its range.
    OutOfRange {
        /// Which parameter.
        what: &'static str,
        /// Value supplied.
        value: f64,
        /// Lowest admissible.
        low: f64,
        /// Highest admissible.
        high: f64,
    },
    /// A vertex or variable index past the problem.
    Index {
        /// Which index.
        what: &'static str,
        /// The value.
        index: usize,
        /// The count it had to be below.
        count: usize,
    },
    /// The underlying Boltzmann model refused; carries its reason.
    Bayes(BayesError),
}

impl fmt::Display for OptimiseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty { what } => write!(f, "{what} is empty"),
            Self::Dimension { what, got, want } => write!(f, "{what} has {got} entries, needs {want}"),
            Self::NonFinite { what, index } => write!(f, "{what} is not finite at {index}"),
            Self::OutOfRange { what, value, low, high } => {
                write!(f, "{what} = {value} is outside [{low}, {high}]")
            }
            Self::Index { what, index, count } => write!(f, "{what} {index} is past the {count} available"),
            Self::Bayes(e) => write!(f, "model refused: {e}"),
        }
    }
}

impl std::error::Error for OptimiseError {}

impl From<BayesError> for OptimiseError {
    fn from(e: BayesError) -> Self {
        Self::Bayes(e)
    }
}

/// The largest problem [`Qubo::brute_force`] will enumerate: `2^24` states.
pub const MAX_BRUTE_FORCE: usize = 24;

// ---------------------------------------------------------------------------------------------
// QUBO
// ---------------------------------------------------------------------------------------------

/// A quadratic unconstrained binary problem: minimise `Σ_i q_ii x_i + Σ_{i<j} q_ij x_i x_j + c`.
#[derive(Debug, Clone, PartialEq)]
pub struct Qubo {
    /// Variables.
    pub n: usize,
    /// Row-major `n × n`; only `q[i][j]` with `i ≤ j` is read.
    pub q: Vec<f64>,
    /// A constant added to every energy, so a problem with a known target can put it at zero.
    pub offset: f64,
}

impl Qubo {
    /// An all-zero problem of `n` variables.
    ///
    /// # Errors
    ///
    /// [`OptimiseError::Empty`] for `n = 0`, [`OptimiseError::OutOfRange`] for `n > 64`.
    pub fn zeros(n: usize) -> Result<Self, OptimiseError> {
        if n == 0 {
            return Err(OptimiseError::Empty { what: "variables" });
        }
        if n > 64 {
            return Err(OptimiseError::OutOfRange { what: "variables", value: n as f64, low: 1.0, high: 64.0 });
        }
        Ok(Self { n, q: vec![0.0; n * n], offset: 0.0 })
    }

    /// Build from an explicit upper-triangular matrix.
    ///
    /// # Errors
    ///
    /// As [`Qubo::zeros`], plus [`OptimiseError::Dimension`] and [`OptimiseError::NonFinite`].
    pub fn new(n: usize, q: Vec<f64>, offset: f64) -> Result<Self, OptimiseError> {
        let mut me = Self::zeros(n)?;
        if q.len() != n * n {
            return Err(OptimiseError::Dimension { what: "q", got: q.len(), want: n * n });
        }
        if let Some(i) = q.iter().position(|x| !x.is_finite()) {
            return Err(OptimiseError::NonFinite { what: "q", index: i });
        }
        if !offset.is_finite() {
            return Err(OptimiseError::NonFinite { what: "offset", index: 0 });
        }
        me.q = q;
        me.offset = offset;
        Ok(me)
    }

    /// Add `v` to the coefficient of `x_i` (`i == j`) or of `x_i x_j` (`i < j`; a pair given the
    /// other way round is folded into the upper triangle).
    ///
    /// # Errors
    ///
    /// [`OptimiseError::Index`] past `n`, [`OptimiseError::NonFinite`].
    pub fn add(&mut self, i: usize, j: usize, v: f64) -> Result<(), OptimiseError> {
        if i >= self.n {
            return Err(OptimiseError::Index { what: "variable", index: i, count: self.n });
        }
        if j >= self.n {
            return Err(OptimiseError::Index { what: "variable", index: j, count: self.n });
        }
        if !v.is_finite() {
            return Err(OptimiseError::NonFinite { what: "coefficient", index: 0 });
        }
        let (a, b) = if i <= j { (i, j) } else { (j, i) };
        self.q[a * self.n + b] += v;
        Ok(())
    }

    /// The energy of a state, bit `k` for `x_k`. `None` for a bit set at or above `n`.
    #[must_use]
    pub fn energy(&self, x: u64) -> Option<f64> {
        if self.n < 64 && x >> self.n != 0 {
            return None;
        }
        let mut e = self.offset;
        for i in 0..self.n {
            if x >> i & 1 == 0 {
                continue;
            }
            e += self.q[i * self.n + i];
            for j in (i + 1)..self.n {
                if x >> j & 1 == 1 {
                    e += self.q[i * self.n + j];
                }
            }
        }
        Some(e)
    }

    /// The change in energy from flipping bit `k` of `x`: `E(x ⊕ k) − E(x)`, in `O(n)`.
    /// `None` for an out-of-range `k` or `x`.
    #[must_use]
    pub fn delta(&self, x: u64, k: usize) -> Option<f64> {
        if k >= self.n || (self.n < 64 && x >> self.n != 0) {
            return None;
        }
        // The linear-in-x_k part of E: q_kk + Σ_{j≠k, x_j = 1} q_{min,max}.
        let mut field = self.q[k * self.n + k];
        for j in 0..self.n {
            if j != k && x >> j & 1 == 1 {
                let (a, b) = if k < j { (k, j) } else { (j, k) };
                field += self.q[a * self.n + b];
            }
        }
        Some(if x >> k & 1 == 1 { -field } else { field })
    }

    /// The Boltzmann model at inverse temperature `beta` whose log-probability is `−β E(x)`
    /// (without the offset): `b_k = −β q_kk`, `W_kj = −β q_kj`.
    ///
    /// # Errors
    ///
    /// [`OptimiseError::OutOfRange`] for a non-positive `beta`, [`OptimiseError::Bayes`] if the
    /// model is refused.
    pub fn boltzmann(&self, beta: f64) -> Result<Boltzmann, OptimiseError> {
        if !(beta > 0.0) || !beta.is_finite() {
            return Err(OptimiseError::OutOfRange { what: "beta", value: beta, low: f64::MIN_POSITIVE, high: f64::INFINITY });
        }
        let n = self.n;
        let bias: Vec<f64> = (0..n).map(|k| -beta * self.q[k * n + k]).collect();
        let mut w = vec![0.0; n * n];
        for i in 0..n {
            for j in (i + 1)..n {
                let v = -beta * self.q[i * n + j];
                w[i * n + j] = v;
                w[j * n + i] = v;
            }
        }
        Ok(Boltzmann::new(&bias, &w)?)
    }

    /// The exact minimum by enumeration: `(state, energy)`, the lowest state index on a tie.
    /// `None` past [`MAX_BRUTE_FORCE`] variables.
    #[must_use]
    pub fn brute_force(&self) -> Option<(u64, f64)> {
        if self.n > MAX_BRUTE_FORCE {
            return None;
        }
        let mut best = (0u64, f64::INFINITY);
        for x in 0..(1u64 << self.n) {
            let e = self.energy(x)?;
            if e < best.1 {
                best = (x, e);
            }
        }
        Some(best)
    }

    /// Max-cut of an undirected graph on `vertices` as a QUBO: minimising the energy maximises the
    /// number of edges whose endpoints take different values, and `−energy` **is** the cut.
    ///
    /// Each edge contributes `−(x_u + x_v − 2 x_u x_v)`: `−1` to `q_uu` and `q_vv`, `+2` to `q_uv`.
    ///
    /// # Errors
    ///
    /// As [`Qubo::zeros`], plus [`OptimiseError::Index`] for an edge past `vertices`.
    pub fn max_cut(vertices: usize, edges: &[(usize, usize)]) -> Result<Self, OptimiseError> {
        let mut q = Self::zeros(vertices)?;
        for &(u, v) in edges {
            if u == v {
                continue;
            }
            q.add(u, u, -1.0)?;
            q.add(v, v, -1.0)?;
            q.add(u, v, 2.0)?;
        }
        Ok(q)
    }
}

// ---------------------------------------------------------------------------------------------
// Graph colouring
// ---------------------------------------------------------------------------------------------

/// `k`-colouring of a graph as a QUBO over one-hot variables `x_{v,c}`:
/// `E = A Σ_v (1 − Σ_c x_vc)² + B Σ_{(u,v)} Σ_c x_uc x_vc`, which is zero exactly for a proper
/// colouring and positive otherwise.
#[derive(Debug, Clone, PartialEq)]
pub struct GraphColouring {
    /// Vertices.
    pub vertices: usize,
    /// Colours.
    pub colours: usize,
    /// Edges.
    pub edges: Vec<(usize, usize)>,
    /// The one-hot penalty `A`.
    pub a: f64,
    /// The conflict penalty `B`.
    pub b: f64,
}

impl GraphColouring {
    /// Build. `vertices · colours` must not exceed 64.
    ///
    /// # Errors
    ///
    /// [`OptimiseError::Empty`], [`OptimiseError::OutOfRange`] for too many variables or a
    /// non-positive penalty, [`OptimiseError::Index`] for an edge past the vertices.
    pub fn new(vertices: usize, colours: usize, edges: &[(usize, usize)], a: f64, b: f64) -> Result<Self, OptimiseError> {
        if vertices == 0 {
            return Err(OptimiseError::Empty { what: "vertices" });
        }
        if colours == 0 {
            return Err(OptimiseError::Empty { what: "colours" });
        }
        let vars = vertices * colours;
        if vars > 64 {
            return Err(OptimiseError::OutOfRange { what: "vertices × colours", value: vars as f64, low: 1.0, high: 64.0 });
        }
        for (what, v) in [("a", a), ("b", b)] {
            if !(v > 0.0) || !v.is_finite() {
                return Err(OptimiseError::OutOfRange { what, value: v, low: f64::MIN_POSITIVE, high: f64::INFINITY });
            }
        }
        for &(u, v) in edges {
            if u >= vertices {
                return Err(OptimiseError::Index { what: "vertex", index: u, count: vertices });
            }
            if v >= vertices {
                return Err(OptimiseError::Index { what: "vertex", index: v, count: vertices });
            }
        }
        Ok(Self { vertices, colours, edges: edges.to_vec(), a, b })
    }

    /// Variable index of `(vertex, colour)`.
    #[must_use]
    pub fn var(&self, vertex: usize, colour: usize) -> usize {
        vertex * self.colours + colour
    }

    /// The QUBO. `(1 − Σ_c x_c)² = 1 − Σ_c x_c + 2 Σ_{c<c'} x_c x_c'` since `x² = x`, so each
    /// vertex adds `A` to the offset, `−A` to every `q_{vc,vc}` and `2A` to every `q_{vc,vc'}`;
    /// each edge adds `B` to `q_{uc,vc}` for every colour.
    ///
    /// # Errors
    ///
    /// As [`Qubo::zeros`].
    pub fn qubo(&self) -> Result<Qubo, OptimiseError> {
        let mut q = Qubo::zeros(self.vertices * self.colours)?;
        q.offset = self.a * self.vertices as f64;
        for v in 0..self.vertices {
            for c in 0..self.colours {
                q.add(self.var(v, c), self.var(v, c), -self.a)?;
                for c2 in (c + 1)..self.colours {
                    q.add(self.var(v, c), self.var(v, c2), 2.0 * self.a)?;
                }
            }
        }
        for &(u, v) in &self.edges {
            if u == v {
                continue;
            }
            for c in 0..self.colours {
                q.add(self.var(u, c), self.var(v, c), self.b)?;
            }
        }
        Ok(q)
    }

    /// Read a colouring off a state, **independently of the energy**: `Some(colour per vertex)`
    /// only if every vertex has exactly one colour and no edge joins two of the same colour.
    #[must_use]
    pub fn decode(&self, x: u64) -> Option<Vec<usize>> {
        let mut colouring = Vec::with_capacity(self.vertices);
        for v in 0..self.vertices {
            let mut chosen = None;
            for c in 0..self.colours {
                if x >> self.var(v, c) & 1 == 1 {
                    if chosen.is_some() {
                        return None;
                    }
                    chosen = Some(c);
                }
            }
            colouring.push(chosen?);
        }
        for &(u, v) in &self.edges {
            if u != v && colouring[u] == colouring[v] {
                return None;
            }
        }
        Some(colouring)
    }
}

// ---------------------------------------------------------------------------------------------
// The sampler and the annealer
// ---------------------------------------------------------------------------------------------

/// What a run produced.
#[derive(Debug, Clone, PartialEq)]
pub struct Solution {
    /// The best state seen.
    pub state: u64,
    /// Its energy, offset included.
    pub energy: f64,
    /// The sweep on which the best state was first seen.
    pub found_at_sweep: usize,
    /// Unit updates performed: one membrane evaluation each, `n` synaptic operations at most.
    pub evaluations: u64,
    /// Updates that changed the unit's state — the spikes.
    pub flips: u64,
}

/// A Glauber sampler over a [`Qubo`] at a temperature, and the annealer built from it.
#[derive(Debug, Clone, PartialEq)]
pub struct Annealer {
    /// The problem.
    pub qubo: Qubo,
    /// Current state.
    pub state: u64,
}

impl Annealer {
    /// Start from the all-zero state.
    #[must_use]
    pub fn new(qubo: Qubo) -> Self {
        Self { qubo, state: 0 }
    }

    /// One sweep at inverse temperature `beta`: every unit, in a random order, resamples itself
    /// from its conditional `p(x_k = 1 | rest) = logistic(−β · Δ_k)` where `Δ_k` is the energy of
    /// setting the bit against clearing it. Returns `(evaluations, flips)`.
    ///
    /// # Errors
    ///
    /// [`OptimiseError::OutOfRange`] for a non-positive `beta`.
    pub fn sweep(&mut self, beta: f64, rng: &mut Rng) -> Result<(u64, u64), OptimiseError> {
        if !(beta > 0.0) || !beta.is_finite() {
            return Err(OptimiseError::OutOfRange { what: "beta", value: beta, low: f64::MIN_POSITIVE, high: f64::INFINITY });
        }
        let n = self.qubo.n;
        let mut order: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = rng.below((i + 1) as u32) as usize;
            order.swap(i, j);
        }
        let mut flips = 0u64;
        for &k in &order {
            // Energy with the bit set minus with it clear, independent of its current value.
            let set = self.state | (1 << k);
            let clear = self.state & !(1 << k);
            let delta_on = self.qubo.delta(clear, k).unwrap_or(0.0);
            let p_on = logistic(-beta * delta_on);
            let next = if rng.next_f64() < p_on { set } else { clear };
            if next != self.state {
                flips += 1;
            }
            self.state = next;
        }
        Ok((n as u64, flips))
    }

    /// Anneal from `beta_start` to `beta_end` geometrically over `sweeps` sweeps, returning the
    /// best state seen.
    ///
    /// # Errors
    ///
    /// [`OptimiseError::OutOfRange`] for a non-positive or decreasing schedule,
    /// [`OptimiseError::Empty`] for zero sweeps.
    pub fn anneal(&mut self, beta_start: f64, beta_end: f64, sweeps: usize, rng: &mut Rng) -> Result<Solution, OptimiseError> {
        if sweeps == 0 {
            return Err(OptimiseError::Empty { what: "sweeps" });
        }
        if !(beta_start > 0.0) || !beta_start.is_finite() {
            return Err(OptimiseError::OutOfRange { what: "beta_start", value: beta_start, low: f64::MIN_POSITIVE, high: f64::INFINITY });
        }
        if !(beta_end >= beta_start) || !beta_end.is_finite() {
            return Err(OptimiseError::OutOfRange { what: "beta_end", value: beta_end, low: beta_start, high: f64::INFINITY });
        }
        let ratio = if sweeps > 1 { (beta_end / beta_start).powf(1.0 / (sweeps as f64 - 1.0)) } else { 1.0 };
        let mut beta = beta_start;
        let mut best = (self.state, self.qubo.energy(self.state).unwrap_or(f64::INFINITY));
        let mut found_at = 0;
        let (mut evaluations, mut flips) = (0u64, 0u64);
        for s in 0..sweeps {
            let (e, f) = self.sweep(beta, rng)?;
            evaluations += e;
            flips += f;
            let energy = self.qubo.energy(self.state).unwrap_or(f64::INFINITY);
            if energy < best.1 {
                best = (self.state, energy);
                found_at = s;
            }
            beta *= ratio;
        }
        Ok(Solution { state: best.0, energy: best.1, found_at_sweep: found_at, evaluations, flips })
    }

    /// Anneal `restarts` times from random states and keep the best, the way a solver is run in
    /// practice. Returns the best solution and how many restarts reached its energy.
    ///
    /// # Errors
    ///
    /// As [`Annealer::anneal`], plus [`OptimiseError::Empty`] for zero restarts.
    pub fn anneal_restarts(
        &mut self,
        beta_start: f64,
        beta_end: f64,
        sweeps: usize,
        restarts: usize,
        rng: &mut Rng,
    ) -> Result<(Solution, usize), OptimiseError> {
        if restarts == 0 {
            return Err(OptimiseError::Empty { what: "restarts" });
        }
        let mut best: Option<Solution> = None;
        let mut hits = 0;
        for _ in 0..restarts {
            self.state = if self.qubo.n == 64 { u64::from(rng.next_u32()) << 32 | u64::from(rng.next_u32()) } else { (u64::from(rng.next_u32()) << 32 | u64::from(rng.next_u32())) & ((1u64 << self.qubo.n) - 1) };
            let mut s = self.anneal(beta_start, beta_end, sweeps, rng)?;
            match &mut best {
                None => {
                    hits = 1;
                    best = Some(s);
                }
                Some(b) => {
                    if s.energy < b.energy - 1e-12 {
                        s.evaluations += b.evaluations;
                        s.flips += b.flips;
                        *b = s;
                        hits = 1;
                    } else {
                        b.evaluations += s.evaluations;
                        b.flips += s.flips;
                        if (s.energy - b.energy).abs() <= 1e-12 {
                            hits += 1;
                        }
                    }
                }
            }
        }
        Ok((best.unwrap_or(Solution { state: 0, energy: f64::INFINITY, found_at_sweep: 0, evaluations: 0, flips: 0 }), hits))
    }
}

#[cfg(test)]
mod tests {
    use super::{Annealer, GraphColouring, OptimiseError, Qubo};
    use crate::bayes::Histogram;
    use crate::rng::Rng;

    fn cycle(n: usize) -> Vec<(usize, usize)> {
        (0..n).map(|i| (i, (i + 1) % n)).collect()
    }

    fn complete(n: usize) -> Vec<(usize, usize)> {
        let mut e = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                e.push((i, j));
            }
        }
        e
    }

    /// The Petersen graph: vertices 0–4 an outer 5-cycle, 5–9 an inner pentagram, spokes between.
    fn petersen() -> Vec<(usize, usize)> {
        let mut e = Vec::new();
        for i in 0..5 {
            e.push((i, (i + 1) % 5));
            e.push((5 + i, 5 + (i + 2) % 5));
            e.push((i, 5 + i));
        }
        e
    }

    /// The energy against hand values, the flip delta against a recomputation, and the max-cut
    /// mapping against an independent edge count over every state of the 6-cycle.
    #[test]
    fn the_energy_the_delta_and_the_cut_are_what_they_say() {
        // E = 2 x0 − 1 x1 + 3 x0 x1, offset 0.5.
        let q = Qubo::new(2, vec![2.0, 3.0, 0.0, -1.0], 0.5).unwrap();
        assert_eq!(q.energy(0b00), Some(0.5));
        assert_eq!(q.energy(0b01), Some(2.5));
        assert_eq!(q.energy(0b10), Some(-0.5));
        assert_eq!(q.energy(0b11), Some(4.5));
        assert_eq!(q.energy(0b100), None);
        for x in 0..4u64 {
            for k in 0..2 {
                let want = q.energy(x ^ (1 << k)).unwrap() - q.energy(x).unwrap();
                assert!((q.delta(x, k).unwrap() - want).abs() < 1e-15, "x {x} k {k}");
            }
        }
        assert_eq!(q.delta(0, 2), None);
        let edges = cycle(6);
        let cut = Qubo::max_cut(6, &edges).unwrap();
        for x in 0..64u64 {
            let crossing = edges.iter().filter(|(u, v)| (x >> u & 1) != (x >> v & 1)).count();
            assert_eq!(-cut.energy(x).unwrap(), crossing as f64, "state {x:#b}");
        }
        assert_eq!(cut.brute_force().unwrap().1, -6.0, "a 6-cycle is bipartite: every edge cuts");
        assert_eq!(Qubo::max_cut(5, &complete(5)).unwrap().brute_force().unwrap().1, -6.0, "K5 cuts floor(25/4) = 6");
        let k33: Vec<(usize, usize)> = (0..3).flat_map(|i| (3..6).map(move |j| (i, j))).collect();
        assert_eq!(Qubo::max_cut(6, &k33).unwrap().brute_force().unwrap().1, -9.0, "K3,3 cuts all nine");
        assert_eq!(Qubo::zeros(30).unwrap().brute_force(), None, "past the enumeration bound");
    }

    /// Before it is an optimiser the sweep is a sampler: at fixed β its state histogram matches
    /// the exact Boltzmann distribution of `−β E` in total variation, on a coupled 6-variable
    /// problem, to 0.02 over 200,000 sweeps. This is the check that the flip probability is the
    /// right conditional and not merely a downhill rule.
    #[test]
    fn at_a_fixed_temperature_the_sweep_samples_the_boltzmann_distribution() {
        let mut rng = Rng::new(3);
        let mut q = Qubo::zeros(6).unwrap();
        for i in 0..6 {
            q.add(i, i, 0.4 * (i as f64 - 2.5)).unwrap();
            q.add(i, (i + 1) % 6, if i % 2 == 0 { -0.8 } else { 0.6 }).unwrap();
        }
        let beta = 1.3;
        let model = q.boltzmann(beta).unwrap();
        let exact = model.exact().unwrap();
        // The Boltzmann model's log-probability is −β E; the sampler's flip rule uses the same E.
        for x in 0..64u64 {
            assert!((model.energy(x).unwrap() + beta * q.energy(x).unwrap()).abs() < 1e-12, "state {x}");
        }
        let mut ann = Annealer::new(q);
        let mut hist = Histogram::new(6).unwrap();
        for _ in 0..2_000 {
            ann.sweep(beta, &mut rng).unwrap();
        }
        for _ in 0..200_000 {
            ann.sweep(beta, &mut rng).unwrap();
            hist.observe(ann.state);
        }
        let tv = hist.total_variation(&exact).unwrap();
        assert!(tv < 0.02, "total variation {tv} from the exact distribution");
        // A wrong rule is visible: sampling at half the temperature is a different distribution.
        let cold = ann.qubo.boltzmann(2.0 * beta).unwrap().exact().unwrap();
        assert!(hist.total_variation(&cold).unwrap() > 0.05);
        assert!(matches!(ann.sweep(0.0, &mut rng), Err(OptimiseError::OutOfRange { what: "beta", .. })));
    }

    /// The annealer reaches the brute-force optimum on random 12-variable problems: twenty
    /// problems, four restarts each, at least nineteen solved, and the counts are the integers
    /// the doc says.
    #[test]
    fn the_annealer_reaches_the_brute_force_optimum_on_random_problems() {
        let mut rng = Rng::new(4);
        let n = 12;
        let sweeps = 200;
        let mut solved = 0;
        for trial in 0..20 {
            let mut q = Qubo::zeros(n).unwrap();
            for i in 0..n {
                for j in i..n {
                    q.add(i, j, 2.0 * rng.next_f64() - 1.0).unwrap();
                }
            }
            let (_, optimum) = q.brute_force().unwrap();
            let mut ann = Annealer::new(q);
            let (sol, hits) = ann.anneal_restarts(0.1, 10.0, sweeps, 4, &mut rng).unwrap();
            assert_eq!(sol.evaluations, 4 * sweeps as u64 * n as u64, "one evaluation per unit per sweep");
            assert!(sol.flips <= sol.evaluations);
            assert!(sol.flips > 0);
            assert!(sol.energy >= optimum - 1e-12, "trial {trial}: below the optimum, which cannot be");
            if (sol.energy - optimum).abs() < 1e-9 {
                solved += 1;
                assert!(hits >= 1);
            }
        }
        assert!(solved >= 19, "solved {solved} of 20");
    }

    /// Graph colouring: the 6-cycle is 2-colourable and the annealer finds a proper colouring
    /// (energy 0, decoded independently); K4 is NOT 3-colourable, and the brute-force minimum of
    /// its 12-variable QUBO is exactly 1 — one conflict or one uncoloured vertex, whichever the
    /// penalties make cheaper; the Petersen graph is 3-colourable and a 30-variable anneal finds
    /// it.
    #[test]
    fn colouring_finds_proper_colourings_and_refuses_the_impossible_by_energy() {
        let mut rng = Rng::new(5);
        let c6 = GraphColouring::new(6, 2, &cycle(6), 1.0, 1.0).unwrap();
        let q = c6.qubo().unwrap();
        assert_eq!(q.brute_force().unwrap().1, 0.0);
        let mut ann = Annealer::new(q);
        let (sol, _) = ann.anneal_restarts(0.2, 8.0, 100, 3, &mut rng).unwrap();
        assert_eq!(sol.energy, 0.0);
        let colouring = c6.decode(sol.state).expect("a proper 2-colouring");
        assert_eq!(colouring.len(), 6);
        for (u, v) in cycle(6) {
            assert_ne!(colouring[u], colouring[v]);
        }

        let k4 = GraphColouring::new(4, 3, &complete(4), 1.0, 1.0).unwrap();
        let (state, min) = k4.qubo().unwrap().brute_force().unwrap();
        assert_eq!(min, 1.0, "K4 with three colours costs exactly one violation");
        assert_eq!(k4.decode(state), None, "the minimum-energy state is not a proper colouring");
        // The decoder is independent of the energy: it refuses a two-colour vertex and an edge
        // conflict on states whose energy it never reads.
        let two_colours = 0b011; // vertex 0 has colours 0 and 1
        assert_eq!(k4.decode(two_colours), None);
        let conflict = 0b001_001_001_001; // every vertex colour 0
        assert_eq!(k4.decode(conflict), None);

        let pet = GraphColouring::new(10, 3, &petersen(), 1.0, 1.0).unwrap();
        let mut ann = Annealer::new(pet.qubo().unwrap());
        let (sol, hits) = ann.anneal_restarts(0.2, 8.0, 300, 8, &mut rng).unwrap();
        assert_eq!(sol.energy, 0.0, "the Petersen graph is 3-colourable and the annealer did not find it: best {}", sol.energy);
        let colouring = pet.decode(sol.state).expect("a proper 3-colouring");
        for (u, v) in petersen() {
            assert_ne!(colouring[u], colouring[v]);
        }
        println!("optimise: Petersen 3-colouring found in {hits} of 8 restarts, {} flips over {} evaluations", sol.flips, sol.evaluations);
        assert!(hits >= 1);
    }

    /// Every refusal names the problem.
    #[test]
    fn the_refusals_name_the_problem() {
        assert!(matches!(Qubo::zeros(0), Err(OptimiseError::Empty { what: "variables" })));
        assert!(matches!(Qubo::zeros(65), Err(OptimiseError::OutOfRange { what: "variables", .. })));
        assert!(matches!(Qubo::new(2, vec![0.0; 3], 0.0), Err(OptimiseError::Dimension { .. })));
        assert!(matches!(Qubo::new(1, vec![f64::NAN], 0.0), Err(OptimiseError::NonFinite { what: "q", .. })));
        assert!(matches!(Qubo::new(1, vec![0.0], f64::INFINITY), Err(OptimiseError::NonFinite { what: "offset", .. })));
        let mut q = Qubo::zeros(2).unwrap();
        assert!(matches!(q.add(2, 0, 1.0), Err(OptimiseError::Index { index: 2, count: 2, .. })));
        assert!(matches!(q.add(0, 0, f64::NAN), Err(OptimiseError::NonFinite { .. })));
        q.add(1, 0, 1.0).unwrap();
        assert_eq!(q.q[1], 1.0, "a pair given the other way round lands in the upper triangle (index 0·n + 1)");
        assert!(matches!(q.boltzmann(0.0), Err(OptimiseError::OutOfRange { what: "beta", .. })));
        assert!(matches!(Qubo::max_cut(2, &[(0, 5)]), Err(OptimiseError::Index { .. })));
        assert!(matches!(GraphColouring::new(0, 2, &[], 1.0, 1.0), Err(OptimiseError::Empty { .. })));
        assert!(matches!(GraphColouring::new(30, 3, &[], 1.0, 1.0), Err(OptimiseError::OutOfRange { .. })));
        assert!(matches!(GraphColouring::new(3, 2, &[], 0.0, 1.0), Err(OptimiseError::OutOfRange { what: "a", .. })));
        assert!(matches!(GraphColouring::new(3, 2, &[(0, 7)], 1.0, 1.0), Err(OptimiseError::Index { .. })));
        let mut ann = Annealer::new(Qubo::zeros(3).unwrap());
        let mut rng = Rng::new(1);
        assert!(matches!(ann.anneal(1.0, 2.0, 0, &mut rng), Err(OptimiseError::Empty { what: "sweeps" })));
        assert!(matches!(ann.anneal(2.0, 1.0, 5, &mut rng), Err(OptimiseError::OutOfRange { what: "beta_end", .. })));
        assert!(matches!(ann.anneal_restarts(1.0, 2.0, 5, 0, &mut rng), Err(OptimiseError::Empty { what: "restarts" })));
        let e: OptimiseError = crate::bayes::BayesError::Empty { what: "bias" }.into();
        assert!(matches!(e, OptimiseError::Bayes(_)));
        for e in [
            OptimiseError::Empty { what: "x" },
            OptimiseError::Dimension { what: "y", got: 1, want: 2 },
            OptimiseError::NonFinite { what: "z", index: 0 },
            OptimiseError::OutOfRange { what: "w", value: 9.0, low: 0.0, high: 1.0 },
            OptimiseError::Index { what: "v", index: 3, count: 2 },
        ] {
            assert!(!e.to_string().is_empty());
        }
    }
}
