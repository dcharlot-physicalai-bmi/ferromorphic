//! Associative memory as an energy landscape: the classical Hopfield network, Krotov and Hopfield's
//! dense associative memory, and the modern continuous Hopfield network — each with its energy
//! function, its update as a descent on that energy, and its capacity checked against the number
//! the theory gives rather than asserted from the paper.
//!
//! # What the mechanism is
//!
//! Store patterns in a symmetric weight matrix; then, from a corrupted cue, let every neuron
//! repeatedly align itself with the field the others produce. Hopfield's 1982 observation
//! (*Neural networks and physical systems with emergent collective computational abilities*, PNAS
//! 79(8):2554–2558) was that with symmetric weights this dynamics never increases the energy
//! `E = −½ xᵀ W x`, so it must stop — and it stops at a stored pattern if there are few enough of
//! them. "Few enough" is `0.138 N` (Amit, Gutfreund and Sompolinsky, *Storing infinite numbers of
//! patterns in a spin-glass model of neural networks*, Physical Review Letters 55(14):1530–1533,
//! 1985): past it the crosstalk between patterns, whose mean square is exactly
//! `(P − 1)(N − 1)/N²` per neuron, overwhelms the signal.
//!
//! Two later ideas removed that limit. Krotov and Hopfield (*Dense associative memory for pattern
//! recognition*, `NeurIPS` 29, 2016) replaced the quadratic energy with `−Σ_μ F(ξ^μ · x)` for a
//! steeper `F` — a polynomial of degree `n` — and the capacity grows as `N^{n−1}`. Ramsauer et al.
//! (*Hopfield networks is all you need*, ICLR 2021; Demircigil et al., *On a model of associative
//! memory with huge storage capacity*, Journal of Statistical Physics 168:288–299, 2017 for the
//! exponential capacity) took `F` to the exponential and the state to continuous values, and the
//! update rule became `ξ ← Xᵀ softmax(β X ξ)` — the attention mechanism of a transformer, with
//! the stored patterns as keys and values and the query as the state.
//!
//! # Why it is in a neuromorphic crate
//!
//! An associative memory is the workload a crossbar was made for: the recall dynamics is a
//! matrix-vector product followed by a threshold, per neuron per update, and the "learning" is
//! writing an outer product into the conductances once. Every neuromorphic substrate this crate
//! models has demonstrated one, and the energy that decides whether it retrieves is the same
//! Lyapunov function this module descends. What this module charges for is exact:
//! [`Classical::update_cost`] is `N` multiply-accumulates per neuron update, and the dense and
//! modern forms pay `P · N` per update because they visit every stored pattern — the capacity
//! was bought with work, and the ledger says how much.
//!
//! # The closed forms this module is checked against
//!
//! - The energy is non-increasing under asynchronous updates, on every step of every recall.
//! - The crosstalk on a stored pattern's local field has mean square `(P − 1)(N − 1)/N²`, measured
//!   over sixteen draws to 8%.
//! - Below `0.138 N` every stored pattern is a fixed point and is recovered from a corrupted cue;
//!   well above it, most are not.
//! - The dense memory with `n = 3` holds `P = N` patterns as fixed points, which the classical
//!   network cannot, and its energy is non-increasing under its own update.
//! - Degree two is the classical network only without the rectification: `−Σ_μ (ξ^μ · x)²`
//!   equals `2N·E − PN` on every state tried. At `N = 100` with five patterns the rectified memory
//!   puts a stored pattern and its anti-pattern more than `N²/2` apart in energy, and it leaves
//!   the anti-pattern, which the classical network keeps as a fixed point.
//! - The modern network's update decreases its energy, retrieves in one step, and holds
//!   `P ≫ N` random patterns as fixed points at large `β` — and its update IS softmax attention.
//!
//! # What this module has NOT reproduced
//!
//! - Any hardware demonstration. The crossbar mapping is described; nothing here talks to one.
//! - The full phase diagram of the classical network (the 0.138 line is a numerical result of a
//!   mean-field theory; this module measures which side of it a population is on, not the line).
//! - Training a modern Hopfield layer inside a network. The update is here; the gradient is not.

use core::fmt;

use crate::rng::Rng;

/// What went wrong, named rather than guessed around.
#[derive(Debug, Clone, PartialEq)]
pub enum HopfieldError {
    /// A count of zero where at least one is needed.
    Empty {
        /// What was empty.
        what: &'static str,
    },
    /// A pattern or state of the wrong length.
    Dimension {
        /// Which object.
        what: &'static str,
        /// Length supplied.
        got: usize,
        /// Length required.
        want: usize,
    },
    /// A bipolar pattern held something other than `±1`.
    NotBipolar {
        /// Position of the first offending element.
        index: usize,
        /// The value found there.
        value: f64,
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
}

impl fmt::Display for HopfieldError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty { what } => write!(f, "{what} is empty"),
            Self::Dimension { what, got, want } => write!(f, "{what} has {got} entries, needs {want}"),
            Self::NotBipolar { index, value } => write!(f, "element {index} is {value}, not ±1"),
            Self::NonFinite { what, index } => write!(f, "{what} is not finite at {index}"),
            Self::OutOfRange { what, value, low, high } => {
                write!(f, "{what} = {value} is outside [{low}, {high}]")
            }
        }
    }
}

impl std::error::Error for HopfieldError {}

fn bipolar(v: &[f64], n: usize, what: &'static str) -> Result<(), HopfieldError> {
    if v.len() != n {
        return Err(HopfieldError::Dimension { what, got: v.len(), want: n });
    }
    for (i, &x) in v.iter().enumerate() {
        if x != 1.0 && x != -1.0 {
            return Err(HopfieldError::NotBipolar { index: i, value: x });
        }
    }
    Ok(())
}

fn finite(v: &[f64], n: usize, what: &'static str) -> Result<(), HopfieldError> {
    if v.len() != n {
        return Err(HopfieldError::Dimension { what, got: v.len(), want: n });
    }
    if let Some(i) = v.iter().position(|x| !x.is_finite()) {
        return Err(HopfieldError::NonFinite { what, index: i });
    }
    Ok(())
}

/// A random `±1` pattern of length `n`.
#[must_use]
pub fn random_pattern(n: usize, rng: &mut Rng) -> Vec<f64> {
    (0..n).map(|_| if rng.next_u32() & 1 == 1 { 1.0 } else { -1.0 }).collect()
}

/// `pattern` with a fraction `p` of its entries flipped, chosen by the generator. The number
/// flipped is `round(p · n)`, exactly, so a test knows what it corrupted.
///
/// # Errors
///
/// [`HopfieldError::OutOfRange`] for `p` outside `[0, 1]`.
pub fn corrupt(pattern: &[f64], p: f64, rng: &mut Rng) -> Result<Vec<f64>, HopfieldError> {
    if !(0.0..=1.0).contains(&p) {
        return Err(HopfieldError::OutOfRange { what: "p", value: p, low: 0.0, high: 1.0 });
    }
    let n = pattern.len();
    let k = (p * n as f64).round() as usize;
    let mut idx: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = rng.below((i + 1) as u32) as usize;
        idx.swap(i, j);
    }
    let mut out = pattern.to_vec();
    for &i in idx.iter().take(k) {
        out[i] = -out[i];
    }
    Ok(out)
}

/// Fraction of positions at which two patterns agree, in `[0, 1]`. `None` for lengths that
/// differ or an empty pattern.
#[must_use]
pub fn overlap(a: &[f64], b: &[f64]) -> Option<f64> {
    if a.len() != b.len() || a.is_empty() {
        return None;
    }
    Some(a.iter().zip(b).filter(|(x, y)| x == y).count() as f64 / a.len() as f64)
}

/// What a recall produced.
#[derive(Debug, Clone, PartialEq)]
pub struct Recall {
    /// The final state.
    pub state: Vec<f64>,
    /// Asynchronous sweeps run (a sweep visits every neuron once).
    pub sweeps: usize,
    /// Whether a sweep changed nothing — a fixed point.
    pub converged: bool,
    /// Energy before and after.
    pub energy: (f64, f64),
    /// The largest energy increase seen on any single neuron update: zero for a correct
    /// implementation of a symmetric network. For [`Classical`] this is a measurement — the update
    /// rule never consults the energy, and an asymmetric `w` makes it positive. For [`Dense`] it is
    /// zero BY CONSTRUCTION, because a flip is only taken when it lowers the energy; what is
    /// checkable there is that the incrementally maintained final energy equals
    /// [`Dense::energy`] of the final state.
    pub worst_increase: f64,
}

// ---------------------------------------------------------------------------------------------
// Classical
// ---------------------------------------------------------------------------------------------

/// The classical Hopfield network over `{−1, +1}^N` with Hebbian weights.
#[derive(Debug, Clone, PartialEq)]
pub struct Classical {
    /// Neurons.
    pub n: usize,
    /// Symmetric weights, row-major `n × n`, zero on the diagonal.
    pub w: Vec<f64>,
    /// Patterns stored so far.
    pub stored: usize,
}

impl Classical {
    /// An empty network of `n` neurons.
    ///
    /// # Errors
    ///
    /// [`HopfieldError::Empty`] for `n = 0`.
    pub fn new(n: usize) -> Result<Self, HopfieldError> {
        if n == 0 {
            return Err(HopfieldError::Empty { what: "neurons" });
        }
        Ok(Self { n, w: vec![0.0; n * n], stored: 0 })
    }

    /// Hebbian storage: `W_ij += ξ_i ξ_j / N` for `i ≠ j`.
    ///
    /// # Errors
    ///
    /// [`HopfieldError::Dimension`], [`HopfieldError::NotBipolar`].
    pub fn store(&mut self, pattern: &[f64]) -> Result<(), HopfieldError> {
        bipolar(pattern, self.n, "pattern")?;
        let n = self.n;
        let scale = 1.0 / n as f64;
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    self.w[i * n + j] += pattern[i] * pattern[j] * scale;
                }
            }
        }
        self.stored += 1;
        Ok(())
    }

    /// The local field `h_i = Σ_j W_ij x_j`.
    ///
    /// # Errors
    ///
    /// [`HopfieldError::Dimension`], [`HopfieldError::NotBipolar`].
    pub fn field(&self, x: &[f64]) -> Result<Vec<f64>, HopfieldError> {
        bipolar(x, self.n, "state")?;
        let n = self.n;
        Ok((0..n).map(|i| (0..n).map(|j| self.w[i * n + j] * x[j]).sum()).collect())
    }

    /// `E(x) = −½ xᵀ W x`.
    ///
    /// # Errors
    ///
    /// As [`Classical::field`].
    pub fn energy(&self, x: &[f64]) -> Result<f64, HopfieldError> {
        let h = self.field(x)?;
        Ok(-0.5 * h.iter().zip(x).map(|(a, b)| a * b).sum::<f64>())
    }

    /// Whether `x` is a fixed point: `sign(h_i) = x_i` for every `i` (a zero field counts as
    /// agreeing).
    ///
    /// # Errors
    ///
    /// As [`Classical::field`].
    pub fn is_fixed_point(&self, x: &[f64]) -> Result<bool, HopfieldError> {
        let h = self.field(x)?;
        Ok(h.iter().zip(x).all(|(hi, xi)| *hi == 0.0 || hi.signum() == *xi))
    }

    /// Asynchronous recall from `cue`: sweep the neurons in a fixed order, each taking the sign of
    /// its field, until a sweep changes nothing or `max_sweeps` is reached.
    ///
    /// # Errors
    ///
    /// As [`Classical::field`].
    pub fn recall(&self, cue: &[f64], max_sweeps: usize) -> Result<Recall, HopfieldError> {
        bipolar(cue, self.n, "cue")?;
        let n = self.n;
        let mut x = cue.to_vec();
        let start = self.energy(&x)?;
        let mut last = start;
        let mut worst_increase = 0.0f64;
        let mut sweeps = 0;
        let mut converged = false;
        while sweeps < max_sweeps {
            sweeps += 1;
            let mut changed = false;
            for i in 0..n {
                let h: f64 = (0..n).map(|j| self.w[i * n + j] * x[j]).sum();
                let next = if h > 0.0 { 1.0 } else if h < 0.0 { -1.0 } else { x[i] };
                if next != x[i] {
                    x[i] = next;
                    changed = true;
                    let e = self.energy(&x)?;
                    worst_increase = worst_increase.max(e - last);
                    last = e;
                }
            }
            if !changed {
                converged = true;
                break;
            }
        }
        Ok(Recall { state: x, sweeps, converged, energy: (start, last), worst_increase })
    }

    /// Multiply-accumulates one neuron update costs: `N`, one per synapse of the neuron.
    #[must_use]
    pub fn update_cost(&self) -> u64 {
        self.n as u64
    }

    /// The Amit-Gutfreund-Sompolinsky load `P/N` at which retrieval breaks down: `0.138`.
    pub const CRITICAL_LOAD: f64 = 0.138;
}

// ---------------------------------------------------------------------------------------------
// Dense associative memory
// ---------------------------------------------------------------------------------------------

/// Krotov and Hopfield's dense associative memory (Krotov and Hopfield, *Dense associative memory
/// for pattern recognition*, `NeurIPS` 29:1172–1180 (2016), arXiv:1606.01164): energy
/// `−Σ_μ F(ξ^μ · x)` with the rectified polynomial `F(z) = z^n` for `z > 0`, else `0`, which is
/// the paper's Eq. 3.
///
/// The update of neuron `i` compares the energy with `x_i = +1` against `x_i = −1` and takes the
/// lower, which is the paper's rule (their Eq. 4) and is what makes the energy non-increasing.
/// This doc used to cite Eq. 3 for the rule. In arXiv v2 and in the proceedings alike, Eq. 3 is
/// the rectified `F` above, and the update is Eq. 4, a sign whose argument, in the paper's words,
/// "is the difference of two energies".
#[derive(Debug, Clone, PartialEq)]
pub struct Dense {
    /// Neurons.
    pub n: usize,
    /// Polynomial degree `n ≥ 2`.
    ///
    /// With the rectified `F` used here, degree two is NOT the classical network. This doc used
    /// to say "Two recovers the classical network up to a constant". The paper states that
    /// reduction for the unrectified polynomial only (Sec. 2, after Eq. 3: "In the case of the
    /// polynomial function with n = 2 the network reduces to the standard model of associative
    /// memory"), and there it holds: `F(z) = z²` gives `−Σ_μ (ξ^μ · x)² = 2N·E − PN`, where `E` is
    /// the energy of a [`Classical`] network with its `1/N` zero-diagonal weights, the same energy
    /// up to a positive scale and a constant. Rectified, a pattern whose overlap with the state is
    /// negative contributes nothing, so a stored `ξ` and its anti-pattern `−ξ`, which the classical
    /// network cannot tell apart, have very different energies here. A test checks the identity,
    /// the asymmetry, and that the anti-pattern is a fixed point of one network and not the other.
    pub degree: u32,
    /// Stored patterns, each of length `n`.
    pub patterns: Vec<Vec<f64>>,
}

impl Dense {
    /// An empty memory of `n` neurons with energy exponent `degree`.
    ///
    /// # Errors
    ///
    /// [`HopfieldError::Empty`] for `n = 0`, [`HopfieldError::OutOfRange`] for `degree < 2`.
    pub fn new(n: usize, degree: u32) -> Result<Self, HopfieldError> {
        if n == 0 {
            return Err(HopfieldError::Empty { what: "neurons" });
        }
        if degree < 2 {
            return Err(HopfieldError::OutOfRange { what: "degree", value: f64::from(degree), low: 2.0, high: f64::INFINITY });
        }
        Ok(Self { n, degree, patterns: Vec::new() })
    }

    /// Store a pattern.
    ///
    /// # Errors
    ///
    /// [`HopfieldError::Dimension`], [`HopfieldError::NotBipolar`].
    pub fn store(&mut self, pattern: &[f64]) -> Result<(), HopfieldError> {
        bipolar(pattern, self.n, "pattern")?;
        self.patterns.push(pattern.to_vec());
        Ok(())
    }

    fn f(&self, z: f64) -> f64 {
        if z > 0.0 { z.powi(self.degree as i32) } else { 0.0 }
    }

    /// `E(x) = −Σ_μ F(ξ^μ · x)`.
    ///
    /// # Errors
    ///
    /// [`HopfieldError::Dimension`], [`HopfieldError::NotBipolar`].
    pub fn energy(&self, x: &[f64]) -> Result<f64, HopfieldError> {
        bipolar(x, self.n, "state")?;
        Ok(-self
            .patterns
            .iter()
            .map(|p| self.f(p.iter().zip(x).map(|(a, b)| a * b).sum()))
            .sum::<f64>())
    }

    /// Asynchronous recall: each neuron in turn takes the sign that lowers the energy.
    ///
    /// # Errors
    ///
    /// As [`Dense::energy`].
    pub fn recall(&self, cue: &[f64], max_sweeps: usize) -> Result<Recall, HopfieldError> {
        bipolar(cue, self.n, "cue")?;
        let mut x = cue.to_vec();
        let start = self.energy(&x)?;
        let mut last = start;
        let mut worst_increase = 0.0f64;
        let mut sweeps = 0;
        let mut converged = false;
        // Dot products with every pattern, maintained incrementally: flipping x_i changes each
        // by 2·ξ^μ_i·x_i(new).
        let mut dots: Vec<f64> =
            self.patterns.iter().map(|p| p.iter().zip(&x).map(|(a, b)| a * b).sum()).collect();
        while sweeps < max_sweeps {
            sweeps += 1;
            let mut changed = false;
            for i in 0..self.n {
                // Energy with x_i flipped, from the maintained dots.
                let e_now: f64 = -dots.iter().map(|&d| self.f(d)).sum::<f64>();
                let e_flip: f64 = -self
                    .patterns
                    .iter()
                    .zip(&dots)
                    .map(|(p, &d)| self.f(d - 2.0 * p[i] * x[i]))
                    .sum::<f64>();
                if e_flip < e_now {
                    for (p, d) in self.patterns.iter().zip(dots.iter_mut()) {
                        *d -= 2.0 * p[i] * x[i];
                    }
                    x[i] = -x[i];
                    changed = true;
                    worst_increase = worst_increase.max(e_flip - last);
                    last = e_flip;
                }
            }
            if !changed {
                converged = true;
                break;
            }
        }
        Ok(Recall { state: x, sweeps, converged, energy: (start, last), worst_increase })
    }

    /// Whether `x` is a fixed point of the update.
    ///
    /// # Errors
    ///
    /// As [`Dense::energy`].
    pub fn is_fixed_point(&self, x: &[f64]) -> Result<bool, HopfieldError> {
        let r = self.recall(x, 1)?;
        Ok(r.converged)
    }

    /// Multiply-accumulates one neuron update costs: `2 · P · N` — every pattern's dot product,
    /// evaluated for both signs — against the classical network's `N`. The capacity was bought
    /// with this.
    #[must_use]
    pub fn update_cost(&self) -> u64 {
        2 * self.patterns.len() as u64 * self.n as u64
    }
}

// ---------------------------------------------------------------------------------------------
// Modern continuous Hopfield
// ---------------------------------------------------------------------------------------------

/// The modern continuous Hopfield network of Ramsauer et al.: real-valued state, stored patterns
/// as rows of `X`, energy `−β⁻¹ lse(β, Xξ) + ½ ξᵀξ + β⁻¹ ln P + ½ M²`, and the update
/// `ξ ← Xᵀ softmax(β X ξ)`.
#[derive(Debug, Clone, PartialEq)]
pub struct Modern {
    /// Pattern dimension.
    pub n: usize,
    /// Inverse temperature `β`.
    pub beta: f64,
    /// Stored patterns, each of length `n`.
    pub patterns: Vec<Vec<f64>>,
}

impl Modern {
    /// An empty memory over `n`-dimensional patterns at inverse temperature `beta`.
    ///
    /// # Errors
    ///
    /// [`HopfieldError::Empty`] for `n = 0`, [`HopfieldError::OutOfRange`] for a non-positive or
    /// non-finite `beta`.
    pub fn new(n: usize, beta: f64) -> Result<Self, HopfieldError> {
        if n == 0 {
            return Err(HopfieldError::Empty { what: "dimension" });
        }
        if !(beta > 0.0) || !beta.is_finite() {
            return Err(HopfieldError::OutOfRange { what: "beta", value: beta, low: f64::MIN_POSITIVE, high: f64::INFINITY });
        }
        Ok(Self { n, beta, patterns: Vec::new() })
    }

    /// Store a real pattern.
    ///
    /// # Errors
    ///
    /// [`HopfieldError::Dimension`], [`HopfieldError::NonFinite`].
    pub fn store(&mut self, pattern: &[f64]) -> Result<(), HopfieldError> {
        finite(pattern, self.n, "pattern")?;
        self.patterns.push(pattern.to_vec());
        Ok(())
    }

    /// The largest stored norm `M`, or zero with nothing stored.
    #[must_use]
    pub fn max_norm(&self) -> f64 {
        self.patterns.iter().map(|p| p.iter().map(|x| x * x).sum::<f64>().sqrt()).fold(0.0, f64::max)
    }

    /// `softmax(β X ξ)`: the attention weights over the stored patterns.
    ///
    /// # Errors
    ///
    /// [`HopfieldError::Empty`] with nothing stored, [`HopfieldError::Dimension`],
    /// [`HopfieldError::NonFinite`].
    pub fn attention(&self, xi: &[f64]) -> Result<Vec<f64>, HopfieldError> {
        if self.patterns.is_empty() {
            return Err(HopfieldError::Empty { what: "patterns" });
        }
        finite(xi, self.n, "state")?;
        let logits: Vec<f64> =
            self.patterns.iter().map(|p| self.beta * p.iter().zip(xi).map(|(a, b)| a * b).sum::<f64>()).collect();
        let top = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = logits.iter().map(|l| (l - top).exp()).collect();
        let z: f64 = exps.iter().sum();
        Ok(exps.iter().map(|e| e / z).collect())
    }

    /// One update: `ξ ← Xᵀ softmax(β X ξ)`.
    ///
    /// # Errors
    ///
    /// As [`Modern::attention`].
    pub fn update(&self, xi: &[f64]) -> Result<Vec<f64>, HopfieldError> {
        let a = self.attention(xi)?;
        let mut out = vec![0.0; self.n];
        for (w, p) in a.iter().zip(&self.patterns) {
            for (o, &x) in out.iter_mut().zip(p) {
                *o += w * x;
            }
        }
        Ok(out)
    }

    /// The energy `−β⁻¹ lse(β, Xξ) + ½ ξᵀξ + β⁻¹ ln P + ½ M²`, non-negative by the paper's bound.
    ///
    /// # Errors
    ///
    /// As [`Modern::attention`].
    pub fn energy(&self, xi: &[f64]) -> Result<f64, HopfieldError> {
        if self.patterns.is_empty() {
            return Err(HopfieldError::Empty { what: "patterns" });
        }
        finite(xi, self.n, "state")?;
        let logits: Vec<f64> =
            self.patterns.iter().map(|p| self.beta * p.iter().zip(xi).map(|(a, b)| a * b).sum::<f64>()).collect();
        let top = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let lse = top + logits.iter().map(|l| (l - top).exp()).sum::<f64>().ln();
        let half_sq = 0.5 * xi.iter().map(|x| x * x).sum::<f64>();
        let m = self.max_norm();
        Ok(-lse / self.beta + half_sq + (self.patterns.len() as f64).ln() / self.beta + 0.5 * m * m)
    }

    /// The separation of stored pattern `mu`: `Δ_μ = ξ^μ·ξ^μ − max_{ν≠μ} ξ^μ·ξ^ν`, the quantity the
    /// paper's retrieval theorems are stated in. `None` for an index past the store or a store of
    /// one pattern.
    #[must_use]
    pub fn separation(&self, mu: usize) -> Option<f64> {
        if mu >= self.patterns.len() || self.patterns.len() < 2 {
            return None;
        }
        let p = &self.patterns[mu];
        let own: f64 = p.iter().map(|x| x * x).sum();
        let other = self
            .patterns
            .iter()
            .enumerate()
            .filter(|(k, _)| *k != mu)
            .map(|(_, q)| p.iter().zip(q).map(|(a, b)| a * b).sum::<f64>())
            .fold(f64::NEG_INFINITY, f64::max);
        Some(own - other)
    }

    /// Multiply-accumulates one update costs: `2 · P · N`, the attention and the readout.
    #[must_use]
    pub fn update_cost(&self) -> u64 {
        2 * self.patterns.len() as u64 * self.n as u64
    }
}

#[cfg(test)]
mod tests {
    use super::{Classical, Dense, HopfieldError, Modern, corrupt, overlap, random_pattern};
    use crate::rng::Rng;

    // ---- classical ----

    /// The energy never increases on any single asynchronous update — Hopfield's theorem, checked
    /// on every step of every recall in the test rather than once.
    #[test]
    fn asynchronous_updates_never_raise_the_energy() {
        let mut rng = Rng::new(1);
        let n = 120;
        let mut net = Classical::new(n).unwrap();
        let patterns: Vec<Vec<f64>> = (0..8).map(|_| random_pattern(n, &mut rng)).collect();
        for p in &patterns {
            net.store(p).unwrap();
        }
        assert_eq!(net.stored, 8);
        for p in &patterns {
            let cue = corrupt(p, 0.25, &mut rng).unwrap();
            let r = net.recall(&cue, 50).unwrap();
            assert!(r.worst_increase <= 0.0, "energy rose by {}", r.worst_increase);
            assert!(r.energy.1 <= r.energy.0);
            assert!(r.converged);
        }
        // And the energy of a stored pattern is what the Hebbian weights say: −½ Σ_{i≠j} W_ij ξ_i ξ_j
        // = −½ (N − 1)/N · (1 + crosstalk) per neuron, near −(N−1)/2 for one pattern alone.
        let mut one = Classical::new(n).unwrap();
        one.store(&patterns[0]).unwrap();
        let e = one.energy(&patterns[0]).unwrap();
        assert!((e + 0.5 * (n as f64 - 1.0)).abs() < 1e-12, "energy {e}");
    }

    /// The crosstalk on a stored pattern's local field has mean square exactly
    /// `(P − 1)(N − 1)/N²`. Derivation: with `O_μν = Σ_j ξ^μ_j ξ^ν_j` the full overlap,
    /// `h_i ξ^ν_i = (N−1)/N + (1/N) Σ_{μ≠ν} ξ^μ_i ξ^ν_i O_μν − (P−1)/N`; the mean of the sum is
    /// `(P−1)/N` (the `j = i` term of each overlap), its mean square is
    /// `(P−1)/N + (P−1)(P−2)/N²` (the diagonal, plus the `j = k = i` term of each cross pair), and
    /// subtracting the offset leaves `(P−1)/N − (P−1)/N²`. The first draft wrote
    /// `(P−1)/N + ((P−1)/N)²` — it had the offset adding when the self terms make it cancel — and
    /// P = 41 caught it at 9.7% low.
    ///
    /// The closed form is an expectation over pattern draws, and one draw's crosstalk is set by
    /// its `C(P, 2)` realised overlaps — with P = 5 that is ten numbers and the measurement lands
    /// anywhere within a factor of two (0.0054 against 0.01 on the first run). So the measurement
    /// averages sixteen independent draws, which puts its own scatter near 3%, and the tolerance
    /// is 8%.
    #[test]
    fn the_crosstalk_variance_is_p_minus_one_over_n() {
        let n = 400;
        for p in [11usize, 21, 41] {
            let mut acc = 0.0;
            let mut count = 0usize;
            for seed in 0..16u64 {
                let mut rng = Rng::new(100 + seed);
                let mut net = Classical::new(n).unwrap();
                let patterns: Vec<Vec<f64>> = (0..p).map(|_| random_pattern(n, &mut rng)).collect();
                for q in &patterns {
                    net.store(q).unwrap();
                }
                for q in &patterns {
                    let h = net.field(q).unwrap();
                    for (hi, qi) in h.iter().zip(q) {
                        // h_i ξ_i = (N−1)/N + crosstalk_i.
                        let crosstalk = hi * qi - (n as f64 - 1.0) / n as f64;
                        acc += crosstalk * crosstalk;
                        count += 1;
                    }
                }
            }
            let measured = acc / count as f64;
            let want = (p as f64 - 1.0) * (n as f64 - 1.0) / (n as f64 * n as f64);
            assert!((measured / want - 1.0).abs() < 0.08, "P {p}: crosstalk mean square {measured} vs {want}");
        }
    }

    /// Below the critical load every stored pattern is a fixed point and is recovered from a
    /// 15%-corrupted cue; well above it, most stored patterns are no longer fixed points. The
    /// 0.138 line is measured from both sides, not quoted.
    #[test]
    fn retrieval_works_below_the_critical_load_and_fails_above_it() {
        let mut rng = Rng::new(3);
        let n = 300;
        let below = (0.05 * n as f64) as usize; // 15 patterns
        let above = (0.35 * n as f64) as usize; // 105 patterns
        let mut fixed_below = 0;
        let mut recovered = 0;
        let mut net = Classical::new(n).unwrap();
        let pats: Vec<Vec<f64>> = (0..below).map(|_| random_pattern(n, &mut rng)).collect();
        for p in &pats {
            net.store(p).unwrap();
        }
        for p in &pats {
            if net.is_fixed_point(p).unwrap() {
                fixed_below += 1;
            }
            let cue = corrupt(p, 0.15, &mut rng).unwrap();
            assert!((overlap(&cue, p).unwrap() - 0.85).abs() < 1e-12);
            let r = net.recall(&cue, 50).unwrap();
            if overlap(&r.state, p).unwrap() == 1.0 {
                recovered += 1;
            }
        }
        assert_eq!(fixed_below, below, "at load 0.05 every stored pattern is a fixed point");
        assert_eq!(recovered, below, "at load 0.05 every corrupted cue is recovered exactly");

        let mut net = Classical::new(n).unwrap();
        let pats: Vec<Vec<f64>> = (0..above).map(|_| random_pattern(n, &mut rng)).collect();
        for p in &pats {
            net.store(p).unwrap();
        }
        let fixed_above = pats.iter().filter(|p| net.is_fixed_point(p).unwrap()).count();
        assert!(
            (fixed_above as f64) < 0.5 * above as f64,
            "at load 0.35, {fixed_above} of {above} stored patterns are still fixed points"
        );
        // The two loads straddle the constant the module quotes; a literal, so a change to the
        // quoted value is a change here.
        let (lo, hi) = (below as f64 / n as f64, above as f64 / n as f64);
        assert!(lo < 0.138 && 0.138 < hi && Classical::CRITICAL_LOAD == 0.138);
    }

    // ---- dense ----

    /// The dense memory with a cubic energy holds `P = N` patterns as fixed points and recovers
    /// them from corruption — a load of 1.0, seven times the classical limit — and its energy is
    /// non-increasing under its own update. The classical network at the same load holds almost
    /// none, which is the comparison that makes the claim a claim.
    #[test]
    fn a_cubic_energy_stores_as_many_patterns_as_neurons() {
        let mut rng = Rng::new(4);
        let n = 100;
        let patterns: Vec<Vec<f64>> = (0..n).map(|_| random_pattern(n, &mut rng)).collect();
        let mut dense = Dense::new(n, 3).unwrap();
        let mut classical = Classical::new(n).unwrap();
        for p in &patterns {
            dense.store(p).unwrap();
            classical.store(p).unwrap();
        }
        let dense_fixed = patterns.iter().filter(|p| dense.is_fixed_point(p).unwrap()).count();
        let classical_fixed = patterns.iter().filter(|p| classical.is_fixed_point(p).unwrap()).count();
        assert_eq!(dense_fixed, n, "the cubic memory lost {} of {n} patterns", n - dense_fixed);
        assert!(classical_fixed < n / 4, "the classical network kept {classical_fixed} of {n} at load 1");
        let mut recovered = 0;
        for p in patterns.iter().take(20) {
            let cue = corrupt(p, 0.1, &mut rng).unwrap();
            let r = dense.recall(&cue, 30).unwrap();
            assert!(r.worst_increase <= 0.0, "energy rose by {}", r.worst_increase);
            if overlap(&r.state, p).unwrap() == 1.0 {
                recovered += 1;
            }
        }
        assert!(recovered >= 18, "recovered {recovered} of 20 from 10% corruption");
        assert_eq!(dense.update_cost(), 2 * 100 * 100);
        assert_eq!(classical.update_cost(), 100);
        // The rectified energy is never positive — F is zero on negative overlaps — so the
        // ANTI-pattern of a stored pattern has energy at most zero. Unrectified z³ gives it about
        // +N³, and survived the first mutation sweep because every other assertion held.
        let anti: Vec<f64> = patterns[0].iter().map(|x| -x).collect();
        let e_anti = dense.energy(&anti).unwrap();
        assert!(e_anti <= 0.0, "the rectified energy of the anti-pattern is {e_anti}");
        assert!(dense.energy(&patterns[0]).unwrap() <= -(100.0f64.powi(3)), "the stored pattern's own term is −N³");
        // Degree 2 with the rectified F is NOT the classical energy. This comment used to say it
        // was, "up to the diagonal", and the five assertions under it were read as the proof. What
        // they show is narrower: on a sparse store the stored patterns are fixed points of both.
        let mut quad = Dense::new(n, 2).unwrap();
        let mut cls = Classical::new(n).unwrap();
        for p in patterns.iter().take(5) {
            quad.store(p).unwrap();
            cls.store(p).unwrap();
        }
        for p in patterns.iter().take(5) {
            assert!(quad.is_fixed_point(p).unwrap() && cls.is_fixed_point(p).unwrap());
        }
        // The reduction Krotov and Hopfield state (Sec. 2, after Eq. 3) is for the UNRECTIFIED
        // z²: Σ_μ (ξ^μ·x)² = −2N·E_classical + PN on every state, stored or not.
        let mut states: Vec<Vec<f64>> = patterns.iter().take(5).cloned().collect();
        states.extend((0..8).map(|_| random_pattern(n, &mut rng)));
        for x in &states {
            let sq: f64 = patterns.iter().take(5).map(|p| p.iter().zip(x).map(|(a, b)| a * b).sum::<f64>().powi(2)).sum();
            let e = cls.energy(x).unwrap();
            let want = -2.0 * n as f64 * e + 5.0 * n as f64;
            assert!((sq - want).abs() < 1e-6, "Σ(ξ·x)² = {sq}, −2N·E + PN = {want}");
        }
        // Rectified, the anti-pattern is not the pattern's twin. Classical: E(−ξ) = E(ξ), exactly,
        // and −ξ is a fixed point because ξ is. Dense degree 2: ξ's own term is −N² and −ξ's is 0.
        let anti0: Vec<f64> = patterns[0].iter().map(|x| -x).collect();
        assert_eq!(cls.energy(&anti0).unwrap(), cls.energy(&patterns[0]).unwrap());
        assert!(cls.is_fixed_point(&anti0).unwrap());
        let (e_own, e_anti) = (quad.energy(&patterns[0]).unwrap(), quad.energy(&anti0).unwrap());
        assert!(e_own <= -(n as f64).powi(2), "the stored pattern's own term is −N², energy {e_own}");
        assert!(e_anti > -0.5 * (n as f64).powi(2), "the anti-pattern's energy is {e_anti}");
        assert!(!quad.is_fixed_point(&anti0).unwrap(), "the rectified degree-2 memory keeps −ξ");
    }

    // ---- modern ----

    /// The modern network: its update is softmax attention over the stored patterns; the update
    /// decreases the energy; a corrupted cue is retrieved in ONE step to 1e-6 when the patterns
    /// are separated; and two thousand random patterns in 64 dimensions — thirty times more than
    /// neurons — are all fixed points at `β = 1`, because the separation is about 36 and
    /// `e^{36}` is the margin.
    #[test]
    fn the_modern_network_retrieves_in_one_step_and_holds_exponentially_many_patterns() {
        let mut rng = Rng::new(5);
        let n = 64;
        let p = 2000;
        let mut net = Modern::new(n, 1.0).unwrap();
        let patterns: Vec<Vec<f64>> = (0..p).map(|_| random_pattern(n, &mut rng)).collect();
        for q in &patterns {
            net.store(q).unwrap();
        }
        assert_eq!(net.max_norm(), 8.0);
        let mut min_sep = f64::INFINITY;
        for mu in 0..p {
            min_sep = min_sep.min(net.separation(mu).unwrap());
        }
        assert!(min_sep > 20.0, "the smallest separation is {min_sep}; the fixed-point claim needs a margin");
        let mut worst = 0.0f64;
        for q in patterns.iter().take(200) {
            let next = net.update(q).unwrap();
            let err = next.iter().zip(q).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
            worst = worst.max(err);
        }
        assert!(worst < 1e-6, "a stored pattern moved by {worst} under the update");

        // One-step retrieval from a cue with 8 of 64 signs flipped: the cue's dot with its own
        // pattern is 48 and with a stranger N(0, 8), whose largest over two thousand is about 28,
        // so the nearest wrong pattern carries weight e^{28−48} ≈ 2e-9 and the update lands on the
        // pattern to 1e-6. (At 12 flips the margin is 12 and e^{−12} = 6e-6: measured, 21 of 100
        // reached 1e-6. The tolerance is the arithmetic, not a knob.)
        let mut retrieved = 0;
        for q in patterns.iter().take(100) {
            let cue = corrupt(q, 8.0 / 64.0, &mut rng).unwrap();
            let e0 = net.energy(&cue).unwrap();
            let next = net.update(&cue).unwrap();
            let e1 = net.energy(&next).unwrap();
            assert!(e1 <= e0 + 1e-12, "the update raised the energy from {e0} to {e1}");
            assert!(e0 >= -1e-12, "the energy is bounded below by zero: {e0}");
            let err = next.iter().zip(q).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
            if err < 1e-6 {
                retrieved += 1;
            }
        }
        assert!(retrieved >= 95, "retrieved {retrieved} of 100 in one step");

        // The update IS attention: the weights sum to one and the readout is their combination.
        let a = net.attention(&patterns[7]).unwrap();
        assert!((a.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(a[7] > 0.999, "the stored pattern attends to itself: {}", a[7]);
        assert_eq!(net.update_cost(), 2 * 2000 * 64);

        // At a low β the attention is spread and the fixed point is a blur, not the pattern:
        // the temperature is what buys the capacity.
        let warm = Modern { beta: 0.01, ..net.clone() };
        let blur = warm.update(&patterns[0]).unwrap();
        let err = blur.iter().zip(&patterns[0]).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        assert!(err > 0.5, "at β = 0.01 the update still returned the pattern: error {err}");
    }

    /// Every refusal names the problem.
    #[test]
    fn the_refusals_name_the_problem() {
        assert!(matches!(Classical::new(0), Err(HopfieldError::Empty { .. })));
        let mut c = Classical::new(3).unwrap();
        assert!(matches!(c.store(&[1.0, 1.0]), Err(HopfieldError::Dimension { what: "pattern", got: 2, want: 3 })));
        assert!(matches!(c.store(&[1.0, 0.5, 1.0]), Err(HopfieldError::NotBipolar { index: 1, .. })));
        assert!(matches!(Dense::new(3, 1), Err(HopfieldError::OutOfRange { what: "degree", .. })));
        assert!(matches!(Modern::new(3, 0.0), Err(HopfieldError::OutOfRange { what: "beta", .. })));
        let m = Modern::new(3, 1.0).unwrap();
        assert!(matches!(m.update(&[1.0, 1.0, 1.0]), Err(HopfieldError::Empty { what: "patterns" })));
        let mut m = m;
        assert!(matches!(m.store(&[1.0, f64::NAN, 1.0]), Err(HopfieldError::NonFinite { what: "pattern", index: 1 })));
        m.store(&[1.0, 1.0, 1.0]).unwrap();
        assert_eq!(m.separation(0), None, "one pattern has no separation");
        assert!(matches!(corrupt(&[1.0], 1.5, &mut Rng::new(1)), Err(HopfieldError::OutOfRange { .. })));
        assert_eq!(overlap(&[1.0], &[1.0, 1.0]), None);
        assert_eq!(overlap(&[], &[]), None);
        assert_eq!(overlap(&[1.0, -1.0, 1.0], &[1.0, 1.0, 1.0]), Some(2.0 / 3.0));
        // A zero field leaves a neuron as it is: an empty network recalls its cue unchanged and
        // converges at once. Flipping ties to +1 survived the first mutation sweep.
        let empty = Classical::new(4).unwrap();
        let r = empty.recall(&[-1.0, 1.0, -1.0, 1.0], 5).unwrap();
        assert_eq!(r.state, vec![-1.0, 1.0, -1.0, 1.0]);
        assert!(r.converged && r.sweeps == 1);
        // `round`, not `floor`: 15% of 310 is 46.5, which rounds to 47 flips.
        let pat = vec![1.0; 310];
        let c = corrupt(&pat, 0.15, &mut Rng::new(2)).unwrap();
        assert_eq!(c.iter().filter(|x| **x < 0.0).count(), 47);
        for e in [
            HopfieldError::Empty { what: "x" },
            HopfieldError::Dimension { what: "y", got: 1, want: 2 },
            HopfieldError::NotBipolar { index: 3, value: 0.5 },
            HopfieldError::NonFinite { what: "z", index: 0 },
            HopfieldError::OutOfRange { what: "w", value: 9.0, low: 0.0, high: 1.0 },
        ] {
            assert!(!e.to_string().is_empty());
        }
    }


    /// The second mutation sweep blinded the energy monitor and nothing noticed, because on a
    /// symmetric network it never has anything to report. It does on an asymmetric one.
    #[test]
    fn the_energy_monitor_fires_on_an_asymmetric_network() {
        let mut net = Classical::new(2).unwrap();
        net.w = vec![0.0, 2.0, -1.0, 0.0];
        // From (1, 1): neuron 0 sees +2 and stays, neuron 1 sees −1 and flips, and
        // E = −½ Σ h_i x_i goes from −½ to +½.
        let r = net.recall(&[1.0, 1.0], 6).unwrap();
        assert_eq!(r.worst_increase, 1.0);
        assert!(!r.converged && r.sweeps == 6, "an asymmetric pair chases itself forever");
    }

    /// The rectified energy is FLAT wherever every overlap is negative, and a flip is taken only
    /// when it strictly lowers the energy — so the anti-pattern of a lone stored pattern is a
    /// fixed point, reached in one sweep. `≤` in that comparison survived the second sweep.
    #[test]
    fn the_dense_plateau_is_a_fixed_point_and_the_running_energy_is_the_energy() {
        let p = vec![1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0];
        let mut dense = Dense::new(8, 3).unwrap();
        dense.store(&p).unwrap();
        let anti: Vec<f64> = p.iter().map(|x| -x).collect();
        let r = dense.recall(&anti, 5).unwrap();
        assert_eq!(r.state, anti);
        assert!(r.converged && r.sweeps == 1);
        assert_eq!(r.energy, (0.0, 0.0));
        // Two wrong bits: recalled, and the energy carried through the flips is the energy of
        // the state it ended on — 8³ below zero.
        let mut cue = p.clone();
        cue[0] = -cue[0];
        cue[5] = -cue[5];
        let r = dense.recall(&cue, 5).unwrap();
        assert_eq!(r.state, p);
        assert_eq!(r.energy.1, dense.energy(&r.state).unwrap());
        assert_eq!(r.energy, (-64.0, -512.0));
    }

    /// **Recorded as an equivalent mutation: `worst_increase.max(0.0 * (e_flip - last))` in
    /// [`Dense::recall`].** This is the fixture that decides whether it is, and the two steps the
    /// recorded argument leaves out.
    ///
    /// The shape of the hole: every assertion on a dense `worst_increase` in this module is
    /// `<= 0.0` or `== 0.0`, and `+0.0 == -0.0` under every f64 comparison Rust has — so no
    /// comparison in the suite can separate the monitor from one whose argument has been
    /// multiplied by zero. Two things do differ. First, `0.0 * (a strictly negative double)` is
    /// `-0.0`, so the mutant evaluates `0.0f64.max(-0.0)`, whose result Rust documents as
    /// non-deterministic for inputs that compare equal; the SIGN of the reported zero is
    /// therefore an observable that no `assert_eq!` can reach. Second, the recorded argument is
    /// about the wrong subtraction: the accept test is `e_flip < e_now` but the code records
    /// `e_flip - last`, and those agree only because `last` is bit-identical to `e_now` at every
    /// accepted flip — the incremental `*d -= 2.0 * p[i] * x[i]` is character for character the
    /// expression `e_flip` was evaluated on, and [`Dense::energy`] folds the same per-pattern
    /// terms in the same order. Both are pinned here.
    #[test]
    fn the_dense_energy_monitor_differences_the_energy_it_accepted_and_reports_a_positive_zero() {
        let p = vec![1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0];
        let mut dense = Dense::new(8, 3).unwrap();
        dense.store(&p).unwrap();
        let mut cue = p.clone();
        cue[0] = -cue[0];
        cue[5] = -cue[5];
        let r = dense.recall(&cue, 5).unwrap();
        // The monitor only runs on an accepted flip, so the fixture has to take some.
        assert_eq!(r.state, p);
        assert_eq!(r.energy, (-64.0, -512.0), "measured: two wrong bits out of eight, degree 3");
        // `last` is the energy of the state the recall ended on, bit for bit. That identity is
        // what makes `e_flip - last` the same number as `e_flip - e_now`.
        assert_eq!(r.energy.1, dense.energy(&r.state).unwrap());
        // The monitor's own value: not merely `<= 0.0`, which `-0.0` satisfies too, but the
        // `+0.0` the accumulator was initialised to and never moved off.
        let worst = r.worst_increase;
        assert!(worst.is_sign_positive(), "the reported zero is negative: measured {worst:?}");
        assert_eq!(format!("{worst:?}"), "0.0");
    }

    // ---- the fourth sweep's repairs ----

    /// The field and the recall's inlined copy of it both read ROW `i` of the weight matrix,
    /// `h_i = Σ_j W_ij x_j`, not column `i`.
    ///
    /// Why the suite could not see it: [`Classical::store`] adds the Hebbian outer product
    /// `ξ_i ξ_j / N`, which is symmetric, and skips the diagonal, so every matrix built through
    /// the constructor and `store` satisfies `W = Wᵀ` exactly and a transpose changes no number
    /// anywhere. It is observable all the same, because `w` is a public field: a caller can write
    /// an asymmetric matrix, and `the_energy_monitor_fires_on_an_asymmetric_network` already
    /// does — but its matrix has `W_01 = 2, W_10 = −1`, where the transposed recall happens to
    /// chase the same non-converging cycle and report the same worst increase. This one is
    /// one-directional: neuron 1 is driven by neuron 0 and does not drive it back.
    #[test]
    fn the_field_and_the_recall_read_the_row_of_the_weight_matrix_not_the_column() {
        let mut net = Classical::new(2).unwrap();
        net.w = vec![0.0, 0.0, 4.0, 0.0]; // W_01 = 0, W_10 = 4.
        // W x at x = (1, −1) is (0·1 + 0·(−1), 4·1 + 0·(−1)) = (0, 4);
        // Wᵀ x would be (0·1 + 4·(−1), 0·1 + 0·(−1)) = (−4, 0).
        assert_eq!(net.field(&[1.0, -1.0]).unwrap(), vec![0.0, 4.0]);
        // The recall does not call `field`; it recomputes the same sum. Neuron 0 sees nothing and
        // holds at +1, neuron 1 sees +4 and flips up, and the second sweep confirms (1, 1).
        // Under the transpose neuron 0 sees −4 and flips DOWN, landing on (−1, −1) instead.
        let r = net.recall(&[1.0, -1.0], 5).unwrap();
        assert_eq!(r.state, vec![1.0, 1.0]);
        assert!(r.converged && r.sweeps == 2);
        // E = −½ Σ h_i x_i: −½·(0·1 + 4·(−1)) = 2 at the cue, −½·(0·1 + 4·1) = −2 at the end.
        assert_eq!(r.energy, (2.0, -2.0));
    }

    /// A neuron whose local field is exactly zero is already at rest: a zero field agrees with
    /// either sign, which is what [`Classical::is_fixed_point`] says it does.
    ///
    /// Why the suite could not see it: `f64::signum` returns `+1.0` for `+0.0`, so dropping the
    /// zero clause changes the verdict only for a neuron sitting at `−1` with no field — and the
    /// only states given to `is_fixed_point` elsewhere are stored patterns of a Hebbian network,
    /// whose fields are never zero. The empty network is the fixture that separates them, and the
    /// one empty-network check in the suite goes through `recall`, which carries its own,
    /// separate tie clause.
    #[test]
    fn a_neuron_with_no_field_is_already_at_rest() {
        let empty = Classical::new(4).unwrap();
        let x = [1.0, -1.0, 1.0, -1.0];
        assert_eq!(empty.field(&x).unwrap(), vec![0.0; 4]);
        assert!(empty.is_fixed_point(&x).unwrap(), "no field, nothing to disagree with");
        assert!(empty.is_fixed_point(&[-1.0; 4]).unwrap());
        assert!(empty.is_fixed_point(&[1.0; 4]).unwrap());
    }

    /// The recall reports the energy of the state it ENDED on, not the one it started from.
    ///
    /// Why the suite could not see it: every other reading of `energy.1` on a classical recall is
    /// an inequality — `energy.1 <= energy.0` — which an unchanged copy of the starting energy
    /// satisfies with equality, and the only exact pair asserted anywhere is the dense plateau's
    /// `(0.0, 0.0)`, a recall that takes no flip at all. A recall that does take one pins it.
    #[test]
    fn the_classical_recall_reports_the_energy_of_the_state_it_ended_on() {
        let mut net = Classical::new(4).unwrap();
        net.store(&[1.0; 4]).unwrap(); // W_ij = 1/4 off the diagonal, exactly.
        // From (1, 1, 1, −1): the first three neurons see ¼ and hold, the last sees ¾ and flips
        // up. E = −½ Σ h_i x_i goes from −½·(3·¼ − ¾) = 0 to −½·(4·¾) = −3/2.
        let r = net.recall(&[1.0, 1.0, 1.0, -1.0], 5).unwrap();
        assert_eq!(r.state, vec![1.0; 4]);
        assert!(r.converged && r.sweeps == 2);
        assert_eq!(r.energy, (0.0, -1.5));
        assert_eq!(r.energy.1, net.energy(&r.state).unwrap(), "the reported end is the end");
    }

    /// The number of corrupted entries is `round(p·N)`: the NEAREST integer, neither the floor
    /// nor the ceiling.
    ///
    /// Why the suite could not see it: the single case it asserts is 15% of 310, and `0.15 * 310`
    /// is exactly 46.5 in binary — a tie, which `f64::round` takes away from zero to 47 and
    /// `ceil` also takes to 47. Only a fraction strictly below one half separates them:
    /// `0.125 * 10` is exactly 1.25, where `round` flips one entry and `ceil` flips two.
    #[test]
    fn the_number_of_corrupted_entries_is_the_nearest_integer() {
        let flipped = |n: usize, p: f64| {
            let pattern = vec![1.0; n];
            corrupt(&pattern, p, &mut Rng::new(7)).unwrap().iter().filter(|x| **x < 0.0).count()
        };
        assert_eq!(flipped(10, 0.125), 1, "1.25 rounds to 1; ceil gives 2");
        assert_eq!(flipped(310, 0.15), 47, "46.5 is a tie and rounds away from zero; floor gives 46");
        assert_eq!(flipped(10, 0.25), 3, "2.5 is a tie and rounds away from zero");
        assert_eq!(flipped(10, 0.0), 0, "p = 0 corrupts nothing");
        assert_eq!(flipped(10, 1.0), 10, "p = 1 corrupts everything");
    }

    /// The corruption can land on any index, because the shuffle really swaps. A swap of an
    /// element with itself leaves the identity permutation, which puts every flip on the lowest
    /// `k` indices of every pattern for every seed.
    ///
    /// Why the suite could not see it: every other use of `corrupt` reads the NUMBER of changed
    /// entries, or the overlap that follows from it, and both are invariant under which positions
    /// were chosen. This reads the positions. With `p·N = 1` exactly one entry is flipped and its
    /// index is the first element of a uniform permutation, so it is uniform over the `N`
    /// positions; the chance that some index is never chosen in `d` draws is at most
    /// `N·(1 − 1/N)^d`, and at `N = 16`, `d = 400` that bound is `16·(15/16)^400 = 1.0e-10`.
    #[test]
    fn the_corruption_can_land_on_any_index_not_only_the_lowest() {
        let n = 16usize;
        let pattern = vec![1.0; n];
        let mut seen = vec![false; n];
        for seed in 0..400u64 {
            let c = corrupt(&pattern, 1.0 / 16.0, &mut Rng::new(seed)).unwrap();
            let hit: Vec<usize> = (0..n).filter(|&i| c[i] < 0.0).collect();
            assert_eq!(hit.len(), 1, "1/16 of 16 neurons is one flip");
            seen[hit[0]] = true;
        }
        assert!(seen.iter().all(|s| *s), "an index was never corrupted in 400 draws: {seen:?}");
    }

    /// Every constructor refuses the degenerate argument it names: a dense memory of no neurons,
    /// a modern memory of no dimensions, and an infinite or `NaN` inverse temperature.
    ///
    /// Why the suite could not see it: `the_refusals_name_the_problem` exercises
    /// `Classical::new(0)`, `Dense::new(_, 1)` and `Modern::new(_, 0.0)` and stops there, so the
    /// zero-size guards of the other two constructors and the finiteness half of the beta guard
    /// were never reached. `!(beta > 0.0)` already rejects `NaN`, but `INFINITY` passes it and is
    /// refused only by the `is_finite` clause — which nothing called.
    #[test]
    fn the_constructors_refuse_a_zero_size_or_a_non_finite_temperature() {
        assert!(matches!(Dense::new(0, 3), Err(HopfieldError::Empty { what: "neurons" })));
        assert!(matches!(Modern::new(0, 1.0), Err(HopfieldError::Empty { what: "dimension" })));
        assert!(matches!(Modern::new(4, f64::INFINITY), Err(HopfieldError::OutOfRange { what: "beta", .. })));
        assert!(matches!(Modern::new(4, f64::NAN), Err(HopfieldError::OutOfRange { what: "beta", .. })));
        assert!(matches!(Modern::new(4, -1.0), Err(HopfieldError::OutOfRange { what: "beta", .. })));
        // One neuron and degree two are the smallest legal arguments, and they are accepted.
        assert_eq!(Dense::new(1, 2).unwrap().n, 1);
        assert_eq!(Modern::new(1, f64::MIN_POSITIVE).unwrap().n, 1);
    }

    /// A modern network keeps the inverse temperature it was given, and the softmax divides by
    /// that one.
    ///
    /// Why the suite could not see it: the only network the modern tests construct is built AT
    /// `beta = 1.0`, where substituting 1.0 is not a substitution, and the cold network they
    /// compare it against is made by a struct update — `Modern { beta: 0.01, ..net }` — which
    /// never goes through the constructor.
    #[test]
    fn a_modern_network_keeps_the_inverse_temperature_it_was_given() {
        let mut net = Modern::new(2, 4.0).unwrap();
        assert_eq!(net.beta, 4.0);
        net.store(&[1.0, 0.0]).unwrap();
        net.store(&[-1.0, 0.0]).unwrap();
        // At ξ = (1, 0) the logits are β·(±1) = ±4, so after the shift by the larger the
        // exponentials are 1 and e^{−8}. At β = 1 they would be 1 and e^{−2}.
        let z = 1.0 + (-8.0f64).exp();
        assert_eq!(net.attention(&[1.0, 0.0]).unwrap(), vec![1.0 / z, (-8.0f64).exp() / z]);
    }

    /// `max_norm` is the LARGEST stored norm, and an empty store has norm zero — it is the `M` of
    /// the energy's `½M²` term, so both ends matter.
    ///
    /// Why the suite could not see it: the one store it is read on holds two thousand bipolar
    /// patterns of the same length, where every norm is 8 and the largest, the first and the last
    /// are the same number; and it is never called before anything is stored, so the fold's
    /// identity element was never the answer.
    #[test]
    fn the_largest_stored_norm_is_the_largest_and_an_empty_store_has_norm_zero() {
        let mut net = Modern::new(2, 1.0).unwrap();
        assert_eq!(net.max_norm(), 0.0, "nothing stored, so no norm, and ½M² contributes nothing");
        net.store(&[3.0, 4.0]).unwrap(); // ‖·‖ = 5.
        net.store(&[1.0, 0.0]).unwrap(); // ‖·‖ = 1, and it is the LAST.
        assert_eq!(net.max_norm(), 5.0);
        net.store(&[0.0, 12.0]).unwrap(); // now the last is also the largest.
        assert_eq!(net.max_norm(), 12.0);
    }

    /// The softmax is shifted by the LARGEST logit and divided by the partition function.
    ///
    /// Why the suite could not see it: a softmax is invariant to the shift in exact arithmetic,
    /// and the modern test's logits span about ±64, so shifting by the smallest still
    /// exponentiates finite numbers and returns the same weights to the last place. Its store is
    /// also so sharply peaked that `Z` is 1 to well within the 1e-12 the normalisation is checked
    /// at, so multiplying by `Z` where the code divides moves nothing it asserts. Logits two
    /// thousand apart overflow the wrong shift to infinity, and a two-way tie makes `Z` exactly
    /// 2 and the weights exactly a half.
    #[test]
    fn the_softmax_is_shifted_by_the_largest_logit_and_divided_by_the_partition_function() {
        let mut tie = Modern::new(2, 1.0).unwrap();
        tie.store(&[1.0, 0.0]).unwrap();
        tie.store(&[-1.0, 0.0]).unwrap();
        // Both logits are zero at the origin: exps (1, 1), Z = 2, weights (½, ½). Multiplying by
        // Z instead gives (2, 2), which is not even a distribution.
        assert_eq!(tie.attention(&[0.0, 0.0]).unwrap(), vec![0.5, 0.5]);
        let mut far = Modern::new(1, 1.0).unwrap();
        far.store(&[1000.0]).unwrap();
        far.store(&[-1000.0]).unwrap();
        // Logits (1000, −1000) at ξ = (1). Shifted by the largest: exp(0) = 1 and exp(−2000),
        // which underflows to exactly 0. Shifted by the smallest it asks for exp(+2000), which is
        // infinite, and infinity over infinity is NaN.
        let a = far.attention(&[1.0]).unwrap();
        assert_eq!(a, vec![1.0, 0.0]);
        assert!(a.iter().all(|w| w.is_finite()), "the shift exists to keep these finite");
    }

    /// The modern energy is `−β⁻¹ lse(β, Xξ) + ½ ξᵀξ + β⁻¹ ln P + ½M²`, term by term.
    ///
    /// Why the suite could not see it: every assertion on this energy is an INEQUALITY — the
    /// update does not raise it, and it is bounded below by zero — and on the store those tests
    /// use the term `½M²` is 32 while the rest are single digits, so dropping the log-sum-exp's
    /// shift, doubling the quadratic term, multiplying by `β` where the formula divides, or
    /// dropping `ln P` altogether leaves both inequalities comfortably true. This fixture is
    /// chosen so that every term is exact in binary: two stored patterns at ±800 in one
    /// dimension, `β = ½` and the cue at 1, so the logits are ±400, the shifted exponentials are
    /// exactly 1 and exactly 0 (`exp(−800)` is far below the smallest subnormal, which is about
    /// `exp(−745)`), `lse` is exactly 400, `‖ξ‖²` is 1 and `M` is exactly 800. So the whole
    /// expression is compared with `==`, in the order the code sums it.
    #[test]
    fn the_modern_energy_is_its_closed_form_term_by_term() {
        let mut net = Modern::new(1, 0.5).unwrap();
        net.store(&[800.0]).unwrap();
        net.store(&[-800.0]).unwrap();
        let p = net.patterns.len() as f64;
        let want = -400.0_f64 / 0.5 + 0.5 + p.ln() / 0.5 + 0.5 * 800.0 * 800.0;
        assert_eq!(net.energy(&[1.0]).unwrap(), want);
        // β is not a scale on the answer: at β = 1 the logits are ±800 and both β-divided terms
        // double their contribution the other way.
        let cold = Modern { beta: 1.0, ..net.clone() };
        let want_cold = -800.0_f64 / 1.0 + 0.5 + p.ln() / 1.0 + 0.5 * 800.0 * 800.0;
        assert_eq!(cold.energy(&[1.0]).unwrap(), want_cold);
    }

    /// The separation of a stored pattern is its own overlap LESS the largest overlap with any
    /// other pattern — the nearest neighbour, and subtracted.
    ///
    /// Why the suite could not see it: it is read exactly once, as `min_sep > 20.0` over two
    /// thousand random bipolar patterns in 64 dimensions. There `own = 64`, the nearest other
    /// overlap is about +28 and the furthest about −28, so taking the furthest (64 − (−28) = 92)
    /// or adding the nearest (64 + 28 = 92) both give a LARGER number, and a one-sided bound
    /// cannot tell a larger number from the right one. Three patterns with hand-chosen overlaps
    /// separate all three formulas at once.
    #[test]
    fn the_separation_subtracts_the_nearest_other_pattern() {
        let mut net = Modern::new(4, 1.0).unwrap();
        net.store(&[1.0, 1.0, 1.0, 1.0]).unwrap();
        net.store(&[1.0, 1.0, 1.0, -1.0]).unwrap();
        net.store(&[-1.0, -1.0, -1.0, -1.0]).unwrap();
        // Own overlaps are 4. The pairwise overlaps are 0·1 = +2, 0·2 = −4, 1·2 = −2.
        // Pattern 0: 4 − max(2, −4) = 2. The furthest would give 4 − (−4) = 8, and adding the
        // nearest 4 + 2 = 6.
        assert_eq!(net.separation(0), Some(2.0));
        // Pattern 1: 4 − max(2, −2) = 2. Furthest 4 − (−2) = 6, adding 4 + 2 = 6.
        assert_eq!(net.separation(1), Some(2.0));
        // Pattern 2: 4 − max(−4, −2) = 6. Furthest 4 − (−4) = 8, adding 4 + (−2) = 2.
        assert_eq!(net.separation(2), Some(6.0));
        assert_eq!(net.separation(3), None, "there is no pattern 3");
    }

    /// The dense recall runs at most `max_sweeps` sweeps, and a state that only settles on the
    /// SECOND sweep is not a fixed point.
    ///
    /// Why the suite could not see it: both dense recalls in the suite are given budgets (5 and
    /// 30) they never come near, so an off-by-one in the budget changes nothing they read; and
    /// the only states given to `is_fixed_point` are stored patterns and an anti-pattern, every
    /// one of which is unchanged by its first sweep, so widening that call's own budget from one
    /// sweep to two changes no verdict either. A cue one bit away from a stored pattern does
    /// both: it moves on the first sweep, and a budget of one runs out before the sweep that
    /// would prove it had stopped moving.
    #[test]
    fn the_dense_recall_honours_its_sweep_budget_and_a_settling_state_is_not_a_fixed_point() {
        let p = vec![1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0];
        let mut dense = Dense::new(8, 3).unwrap();
        dense.store(&p).unwrap();
        let mut cue = p.clone();
        cue[0] = -cue[0];
        // A budget of zero runs no sweep at all: the cue comes back untouched, unconverged, with
        // its own energy twice. One bit wrong makes ξ·x = 6, and 6³ = 216.
        let none = dense.recall(&cue, 0).unwrap();
        assert_eq!(none.sweeps, 0);
        assert_eq!(none.state, cue);
        assert!(!none.converged);
        assert_eq!(none.energy, (-216.0, -216.0));
        // A budget of one runs exactly one sweep: it repairs the bit and stops, without the
        // second sweep that would show nothing more changes.
        let one = dense.recall(&cue, 1).unwrap();
        assert_eq!(one.sweeps, 1);
        assert_eq!(one.state, p);
        assert!(!one.converged);
        // So the cue is NOT a fixed point, though the pattern it settles to is.
        assert!(!dense.is_fixed_point(&cue).unwrap());
        assert!(dense.is_fixed_point(&p).unwrap());
    }

    /// A real-valued pattern of the wrong length is refused, at storage and at every read.
    ///
    /// Why the suite could not see it: the only bad argument it ever hands the finiteness check
    /// is a `NaN` of the RIGHT length, which is caught after the length check has already passed.
    /// Nothing gave the modern network a short pattern — and a short one would be stored, then
    /// zipped silently against every state, because `zip` stops at the shorter side: the dot
    /// products would be taken over a prefix and the store would hold a pattern of a dimension it
    /// does not have.
    #[test]
    fn a_real_valued_pattern_of_the_wrong_length_is_refused() {
        let mut net = Modern::new(3, 1.0).unwrap();
        assert!(matches!(
            net.store(&[1.0, 2.0]),
            Err(HopfieldError::Dimension { what: "pattern", got: 2, want: 3 })
        ));
        assert!(matches!(
            net.store(&[1.0, 2.0, 3.0, 4.0]),
            Err(HopfieldError::Dimension { what: "pattern", got: 4, want: 3 })
        ));
        assert!(net.patterns.is_empty(), "a refused pattern is not stored");
        net.store(&[1.0, 2.0, 3.0]).unwrap();
        assert!(matches!(
            net.attention(&[1.0, 2.0]),
            Err(HopfieldError::Dimension { what: "state", got: 2, want: 3 })
        ));
        assert!(matches!(
            net.energy(&[1.0, 2.0]),
            Err(HopfieldError::Dimension { what: "state", got: 2, want: 3 })
        ));
    }
}
