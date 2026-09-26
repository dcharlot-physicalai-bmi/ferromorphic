//! The cerebellum as a machine: an adaptive filter that learns by decorrelating its error from its
//! inputs, and Albus's CMAC, the table-lookup controller that theory became — each checked
//! against what it must converge to and how fast.
//!
//! # What the mechanism is
//!
//! Marr (*A theory of cerebellar cortex*, Journal of Physiology 202(2):437–470, 1969,
//! doi:10.1113/jphysiol.1969.sp008820) and Albus (*A theory of cerebellar function*, Mathematical
//! Biosciences 10(1–2):25–61, 1971, doi:10.1016/0025-5564(71)90051-4) read the cerebellar cortex
//! as a learning device with three parts. A small number of mossy-fibre inputs is **recoded** by an
//! enormous number of granule cells into many parallel-fibre signals. One Purkinje cell sums about
//! two hundred thousand of them through adjustable synapses. And a single climbing fibre carries a
//! **teaching** signal that changes those synapses. The two theories disagree on the sign of that
//! change. Marr predicted that parallel-fibre synapses active together with the climbing fibre are
//! **facilitated**; Albus proposed instead that they are **weakened**. The adaptive filter below
//! takes Albus's sign and reads the climbing fibre as an error (Fujita 1982; Dean et al. 2010).
//!
//! ⚠ CORRECTED, against Marr's own abstract. This paragraph previously said that Marr and Albus
//! both read the climbing fibre as an **error** signal that depresses the synapses active with it.
//! For Marr the sign is wrong, and for learned actions so is the word. When the cerebellum learns
//! actions he assumes that "each olivary cell responds to a cerebral instruction for an elemental
//! movement", an instruction rather than an error, and his prediction 5(a) is that "the synapses
//! from parallel fibres to Purkinje cells are facilitated by the conjunction of presynaptic and
//! climbing fibre (or post-synaptic) activity". Only for maintenance reflexes (his 2′) does the
//! olivary drive act as negative feedback — each olivary cell is stimulated by receptors "all of
//! whose activities are usually reduced by the results of stimulating the corresponding Purkinje
//! cell" — and even there the change he predicts is facilitation. The depression sign is Albus's,
//! and the Fujita adaptive-filter line kept it. This review did not locate a readable copy of Albus
//! (1971); his sign rests on Table 1, row "Sign of cf-trained pf-PC change" (Marr LTP;
//! Albus/perceptron LTD; Fujita/adaptive filter LTD), of Gilbert, *Gating by Memory: a Theory of
//! Learning in the Cerebellum*, The Cerebellum 21(6):926–943, 2022, doi:10.1007/s12311-021-01325-9.
//! The rule this module implements, `Δw_i = −β e p_i`, and every test of it are unchanged: only the
//! history it was credited to moved.
//!
//! Two machines follow from the three-part reading, and both are here.
//!
//! - **The adaptive filter** ([`AdaptiveFilter`]; Fujita, *Adaptive filter model of the cerebellum*,
//!   Biological Cybernetics 45(3):195–206, 1982; Dean, Porrill, Ekerot and Jörntell, *The cerebellar
//!   microcircuit as an adaptive filter: experimental and computational evidence*, Nature Reviews
//!   Neuroscience 11(1):30–43, 2010, doi:10.1038/nrn2756). The output is `z = Σ w_i p_i(t)`, the
//!   rule is `Δw_i = −β e p_i` — the covariance of the climbing-fibre error with each parallel
//!   fibre — and learning stops exactly when the error is uncorrelated with every input: the
//!   decorrelation principle. The granule layer is stood in for by a bank of leaky integrators
//!   ([`GranuleBank`]), which gives the filter a spread of time courses to weigh.
//! - **The CMAC** ([`Cmac`]; Albus, *A new approach to manipulator control: the cerebellar model
//!   articulation controller*, Journal of Dynamic Systems, Measurement, and Control
//!   97(3):220–227, 1975). The recoding is made explicit: `C` overlapping tilings of the input
//!   space, one active tile in each, the output their summed weights. Nearby inputs share tiles,
//!   so what is learned at one point generalises to its neighbours and to nothing else.
//!
//! # Why it is in a neuromorphic crate
//!
//! Both are learning rules a chip can run: the weight change needs only the presynaptic signal and
//! one broadcast error, and a CMAC lookup touches exactly `C` weights however large its table —
//! the cost of a control step is a constant, which [`Cmac::active`] makes a count.
//!
//! # The closed forms this module is checked against
//!
//! - **What it converges to.** Batch learning is gradient descent on the mean squared error, whose
//!   minimum is the Wiener solution `w* = R⁻¹ r`, `R = ⟨p pᵀ⟩`, `r = ⟨p d⟩` ([`wiener`], by
//!   Cholesky). For a target `a sin + b cos` and those two inputs, `w* = (a, b)` exactly.
//! - **How fast.** Along an eigenvector of `R` with eigenvalue `λ` the weight error is multiplied
//!   by exactly `1 − βλ` per epoch. For quadrature inputs over whole periods `R = ½ I`, so every
//!   epoch leaves `1 − β/2` of the error.
//! - **When it stops.** At the Wiener solution `⟨e p_i⟩ = 0` for every input — and not before.
//! - **When it diverges.** Stable iff `β < 2/λ_max`; [`safe_beta`] returns the Gershgorin bound
//!   `1/max_i Σ_j |R_ij|`, which is always inside it.
//! - **The granule bank** steps each leaky integrator by its exact exponential, so its response to
//!   a step of input is `1 − e^{−t/τ}` whatever the time step.
//! - **CMAC generalisation is a triangle.** In one dimension two inputs `δ` quantisation steps
//!   apart share exactly `max(0, C − |δ|)` tiles; after one full correction at `x₀` the output at
//!   `x` is the target times that over `C`. In more dimensions, with Albus's diagonal offsets, the
//!   count lies between `C − Σ|δ_j|` and `C − max|δ_j|`, checked exhaustively on a grid.
//! - **CMAC learning at a point** multiplies the error by exactly `1 − β`; training points at
//!   least `C` steps apart do not interfere and are all fitted in one pass.
//!
//! # What this module has NOT reproduced
//!
//! - Spiking Purkinje cells, simple and complex spikes, or the ~1 Hz climbing-fibre rate that
//!   makes the biological error signal so sparse. The error here is a number per sample.
//! - The real granule-layer code, which is not known; a bank of leaky integrators is one
//!   conventional stand-in among several (delay lines, spectral timing, random projections).
//! - Marr's capacity estimates for pattern separation, which depend on that code.
//! - Hashing. Albus's CMAC hashes its tiles into a smaller table and accepts collisions; this one
//!   allocates every tile and reports how many that is.

use core::fmt;

use crate::reservoir::{ReservoirError, cholesky};

/// What went wrong, named rather than guessed around.
#[derive(Debug, Clone, PartialEq)]
pub enum CerebellumError {
    /// A count of zero where at least one is needed.
    Empty {
        /// What was empty.
        what: &'static str,
    },
    /// Two lengths that had to agree.
    Dimension {
        /// Which array.
        what: &'static str,
        /// Length supplied.
        got: usize,
        /// Length required.
        want: usize,
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
    /// A `NaN` or infinity.
    NonFinite {
        /// Which quantity.
        what: &'static str,
        /// Position in the offending array, `0` for a scalar.
        index: usize,
    },
    /// A table too large to allocate.
    TooLarge {
        /// Weights asked for, saturated at `usize::MAX`.
        weights: usize,
    },
    /// The Wiener solve failed: the inputs are linearly dependent over the samples.
    Solve(ReservoirError),
}

impl fmt::Display for CerebellumError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty { what } => write!(f, "{what} is empty"),
            Self::Dimension { what, got, want } => write!(f, "{what} has length {got}, expected {want}"),
            Self::OutOfRange { what, value, low, high } => {
                write!(f, "{what} = {value} is outside [{low}, {high}]")
            }
            Self::NonFinite { what, index } => write!(f, "{what} is not finite at {index}"),
            Self::TooLarge { weights } => write!(f, "a table of {weights} weights is more than MAX_WEIGHTS"),
            Self::Solve(e) => write!(f, "Wiener solve: {e}"),
        }
    }
}

impl std::error::Error for CerebellumError {}

impl From<ReservoirError> for CerebellumError {
    fn from(e: ReservoirError) -> Self {
        Self::Solve(e)
    }
}

/// The largest CMAC table this module will allocate, weights.
pub const MAX_WEIGHTS: usize = 1 << 26;

fn finite_all(what: &'static str, v: &[f64]) -> Result<(), CerebellumError> {
    if let Some(i) = v.iter().position(|x| !x.is_finite()) {
        return Err(CerebellumError::NonFinite { what, index: i });
    }
    Ok(())
}

fn dims(what: &'static str, got: usize, want: usize) -> Result<(), CerebellumError> {
    if got == want { Ok(()) } else { Err(CerebellumError::Dimension { what, got, want }) }
}

fn positive(what: &'static str, v: f64) -> Result<f64, CerebellumError> {
    if v.is_finite() && v > 0.0 {
        Ok(v)
    } else {
        Err(CerebellumError::OutOfRange { what, value: v, low: f64::MIN_POSITIVE, high: f64::INFINITY })
    }
}

fn check_samples(samples: &[Vec<f64>], targets: &[f64], n: usize) -> Result<(), CerebellumError> {
    if samples.is_empty() {
        return Err(CerebellumError::Empty { what: "samples" });
    }
    dims("targets", targets.len(), samples.len())?;
    for s in samples {
        dims("sample", s.len(), n)?;
        finite_all("sample", s)?;
    }
    finite_all("targets", targets)
}

// ---------------------------------------------------------------------------------------------
// The granule layer
// ---------------------------------------------------------------------------------------------

/// A bank of leaky integrators of one mossy-fibre signal, `τ_i ds_i/dt = u − s_i`: the spread of
/// time courses the adaptive filter weighs.
#[derive(Debug, Clone, PartialEq)]
pub struct GranuleBank {
    /// Time constants, seconds.
    pub taus: Vec<f64>,
    /// States, one per time constant.
    pub state: Vec<f64>,
}

impl GranuleBank {
    /// A bank at rest.
    ///
    /// # Errors
    ///
    /// [`CerebellumError::Empty`] for no time constants, [`CerebellumError::OutOfRange`] for a
    /// non-positive one.
    pub fn new(taus: Vec<f64>) -> Result<Self, CerebellumError> {
        if taus.is_empty() {
            return Err(CerebellumError::Empty { what: "time constants" });
        }
        for &t in &taus {
            positive("tau", t)?;
        }
        let n = taus.len();
        Ok(Self { taus, state: vec![0.0; n] })
    }

    /// Advance by `dt` under a constant input `u` — exact for a constant input, whatever `dt`.
    ///
    /// # Errors
    ///
    /// [`CerebellumError::OutOfRange`] for a non-positive `dt`, [`CerebellumError::NonFinite`] for
    /// a non-finite input.
    pub fn step(&mut self, dt: f64, u: f64) -> Result<&[f64], CerebellumError> {
        let dt = positive("dt", dt)?;
        if !u.is_finite() {
            return Err(CerebellumError::NonFinite { what: "input", index: 0 });
        }
        for (s, tau) in self.state.iter_mut().zip(&self.taus) {
            *s = u + (*s - u) * (-dt / tau).exp();
        }
        Ok(&self.state)
    }
}

// ---------------------------------------------------------------------------------------------
// The adaptive filter
// ---------------------------------------------------------------------------------------------

/// One Purkinje cell as an adaptive linear combiner of its parallel fibres.
#[derive(Debug, Clone, PartialEq)]
pub struct AdaptiveFilter {
    /// Parallel-fibre weights.
    pub w: Vec<f64>,
    /// Learning rate `β`.
    pub beta: f64,
}

impl AdaptiveFilter {
    /// A filter of `n` inputs with zero weights.
    ///
    /// # Errors
    ///
    /// [`CerebellumError::Empty`] for no inputs, [`CerebellumError::OutOfRange`] for a
    /// non-positive `beta`.
    pub fn new(n: usize, beta: f64) -> Result<Self, CerebellumError> {
        if n == 0 {
            return Err(CerebellumError::Empty { what: "inputs" });
        }
        Ok(Self { w: vec![0.0; n], beta: positive("beta", beta)? })
    }

    /// The output `Σ w_i p_i`.
    ///
    /// # Errors
    ///
    /// [`CerebellumError::Dimension`] or [`CerebellumError::NonFinite`] for a bad input.
    pub fn output(&self, p: &[f64]) -> Result<f64, CerebellumError> {
        dims("input", p.len(), self.w.len())?;
        finite_all("input", p)?;
        Ok(self.w.iter().zip(p).map(|(w, x)| w * x).sum())
    }

    /// One online update from a climbing-fibre error: `w_i ← w_i − β e p_i`. An input that was
    /// silent when the error came is not changed. A fibre active with a positive error is
    /// depressed, which is Albus's sign and the adaptive filter's; Marr (1969) predicted
    /// facilitation instead (see the module documentation).
    ///
    /// # Errors
    ///
    /// As [`AdaptiveFilter::output`], plus [`CerebellumError::NonFinite`] for a non-finite error.
    pub fn learn(&mut self, p: &[f64], error: f64) -> Result<(), CerebellumError> {
        dims("input", p.len(), self.w.len())?;
        finite_all("input", p)?;
        if !error.is_finite() {
            return Err(CerebellumError::NonFinite { what: "error", index: 0 });
        }
        for (w, x) in self.w.iter_mut().zip(p) {
            *w -= self.beta * error * x;
        }
        Ok(())
    }

    /// The correlation of the error `z − d` with each input over a sample set, `⟨e p_i⟩` — the
    /// quantity the learning rule drives to zero.
    ///
    /// # Errors
    ///
    /// [`CerebellumError::Empty`] for no samples, [`CerebellumError::Dimension`] or
    /// [`CerebellumError::NonFinite`] for a bad sample or target.
    pub fn error_correlation(&self, samples: &[Vec<f64>], targets: &[f64]) -> Result<Vec<f64>, CerebellumError> {
        check_samples(samples, targets, self.w.len())?;
        let mut c = vec![0.0; self.w.len()];
        for (p, d) in samples.iter().zip(targets) {
            let e = self.output(p)? - d;
            for (ci, x) in c.iter_mut().zip(p) {
                *ci += e * x;
            }
        }
        let n = samples.len() as f64;
        Ok(c.into_iter().map(|v| v / n).collect())
    }

    /// One batch epoch: `w ← w − β ⟨e p⟩`. Returns the mean squared error BEFORE the update.
    ///
    /// # Errors
    ///
    /// As [`AdaptiveFilter::error_correlation`].
    pub fn epoch(&mut self, samples: &[Vec<f64>], targets: &[f64]) -> Result<f64, CerebellumError> {
        let c = self.error_correlation(samples, targets)?;
        let mut mse = 0.0;
        for (p, d) in samples.iter().zip(targets) {
            let e = self.output(p)? - d;
            mse += e * e;
        }
        for (w, ci) in self.w.iter_mut().zip(&c) {
            *w -= self.beta * ci;
        }
        Ok(mse / samples.len() as f64)
    }
}

/// The input correlation matrix `R = ⟨p pᵀ⟩` over a sample set, row-major `n × n`.
///
/// # Errors
///
/// [`CerebellumError::Empty`] for no samples or empty samples, [`CerebellumError::Dimension`] for
/// ragged samples, [`CerebellumError::NonFinite`] for a bad entry.
pub fn correlation(samples: &[Vec<f64>]) -> Result<Vec<f64>, CerebellumError> {
    let n = samples.first().map_or(0, Vec::len);
    if n == 0 {
        return Err(CerebellumError::Empty { what: "samples" });
    }
    let mut r = vec![0.0; n * n];
    for p in samples {
        dims("sample", p.len(), n)?;
        finite_all("sample", p)?;
        for a in 0..n {
            for b in 0..n {
                r[a * n + b] += p[a] * p[b];
            }
        }
    }
    let count = samples.len() as f64;
    Ok(r.into_iter().map(|v| v / count).collect())
}

/// The Wiener solution `w* = R⁻¹ r`: the weights batch learning converges to, computed without
/// learning.
///
/// # Errors
///
/// As [`correlation`], plus [`CerebellumError::Dimension`] for a wrong number of targets and
/// [`CerebellumError::Solve`] when the inputs are linearly dependent over the samples.
pub fn wiener(samples: &[Vec<f64>], targets: &[f64]) -> Result<Vec<f64>, CerebellumError> {
    let r = correlation(samples)?;
    let n = samples[0].len();
    check_samples(samples, targets, n)?;
    let mut rhs = vec![0.0; n];
    for (p, d) in samples.iter().zip(targets) {
        for (acc, x) in rhs.iter_mut().zip(p) {
            *acc += x * d;
        }
    }
    let count = samples.len() as f64;
    for v in &mut rhs {
        *v /= count;
    }
    Ok(cholesky(&r, n, 1e-12)?.solve(&rhs)?)
}

/// A learning rate at which batch learning on these samples cannot diverge: `1/max_i Σ_j |R_ij|`.
/// The largest eigenvalue of `R` is at most that row sum (Gershgorin), and learning is stable
/// below `2/λ_max`, so this is inside the stable range by at least a factor of two.
///
/// # Errors
///
/// As [`correlation`], plus [`CerebellumError::OutOfRange`] if every input is identically zero.
pub fn safe_beta(samples: &[Vec<f64>]) -> Result<f64, CerebellumError> {
    let r = correlation(samples)?;
    let n = samples[0].len();
    let worst = (0..n).map(|a| r[a * n..(a + 1) * n].iter().map(|v| v.abs()).sum::<f64>()).fold(0.0, f64::max);
    if worst > 0.0 {
        Ok(1.0 / worst)
    } else {
        Err(CerebellumError::OutOfRange { what: "input power", value: worst, low: f64::MIN_POSITIVE, high: f64::INFINITY })
    }
}

// ---------------------------------------------------------------------------------------------
// The CMAC
// ---------------------------------------------------------------------------------------------

/// Albus's cerebellar model articulation controller: `c` tilings of a box, tiling `k` displaced by
/// `k` quantisation steps along the diagonal, every tile a weight.
#[derive(Debug, Clone, PartialEq)]
pub struct Cmac {
    /// Lower corner of the input box.
    pub low: Vec<f64>,
    /// Quantisation step, the same in every dimension.
    pub res: f64,
    /// Quantisation cells per dimension.
    pub cells: Vec<usize>,
    /// Tilings — the number of weights one lookup touches, and the width of the generalisation.
    pub c: usize,
    /// Learning rate `β` in `(0, 1]`; `1` corrects the whole error at once.
    pub beta: f64,
    /// Every tile's weight: tiling by tiling, row-major within a tiling.
    pub w: Vec<f64>,
}

impl Cmac {
    /// A CMAC over the box `[low, high)` with all weights zero.
    ///
    /// # Errors
    ///
    /// [`CerebellumError::Empty`] for no dimensions or no tilings, [`CerebellumError::Dimension`]
    /// for corners of different lengths, [`CerebellumError::OutOfRange`] for a non-positive `res`,
    /// an empty box or a `beta` outside `(0, 1]`, [`CerebellumError::TooLarge`] past
    /// [`MAX_WEIGHTS`].
    pub fn new(low: Vec<f64>, high: &[f64], res: f64, c: usize, beta: f64) -> Result<Self, CerebellumError> {
        if low.is_empty() {
            return Err(CerebellumError::Empty { what: "dimensions" });
        }
        if c == 0 {
            return Err(CerebellumError::Empty { what: "tilings" });
        }
        dims("high", high.len(), low.len())?;
        finite_all("low", &low)?;
        finite_all("high", high)?;
        let res = positive("res", res)?;
        if !(beta > 0.0) || !(beta <= 1.0) {
            return Err(CerebellumError::OutOfRange { what: "beta", value: beta, low: f64::MIN_POSITIVE, high: 1.0 });
        }
        let mut cells = Vec::with_capacity(low.len());
        for (lo, hi) in low.iter().zip(high) {
            if !(hi > lo) {
                return Err(CerebellumError::OutOfRange { what: "high", value: *hi, low: *lo, high: f64::INFINITY });
            }
            let n = ((hi - lo) / res).ceil();
            if n > MAX_WEIGHTS as f64 {
                return Err(CerebellumError::TooLarge { weights: usize::MAX });
            }
            cells.push(n as usize);
        }
        let mut me = Self { low, res, cells, c, beta, w: Vec::new() };
        let total = me.memory();
        if total > MAX_WEIGHTS {
            return Err(CerebellumError::TooLarge { weights: total });
        }
        me.w = vec![0.0; total];
        Ok(me)
    }

    /// Tiles along dimension `j` in any one tiling: the last cell, displaced by up to `c − 1`,
    /// still has to land in one.
    fn tiles_along(&self, j: usize) -> usize {
        (self.cells[j] - 1).div_ceil(self.c) + 1
    }

    fn tiles_per_tiling(&self) -> usize {
        (0..self.cells.len()).fold(1usize, |acc, j| acc.saturating_mul(self.tiles_along(j)))
    }

    /// Weights in the whole table: tilings × tiles per tiling, saturating.
    #[must_use]
    pub fn memory(&self) -> usize {
        self.c.saturating_mul(self.tiles_per_tiling())
    }

    /// The quantisation cell of an input, one index per dimension.
    ///
    /// # Errors
    ///
    /// [`CerebellumError::Dimension`] for a wrong length, [`CerebellumError::NonFinite`] for a bad
    /// coordinate, [`CerebellumError::OutOfRange`] for a point outside the box — a CMAC has no
    /// tile there, and the nearest one would be a guess.
    pub fn quantise(&self, x: &[f64]) -> Result<Vec<usize>, CerebellumError> {
        dims("input", x.len(), self.low.len())?;
        finite_all("input", x)?;
        let mut u = Vec::with_capacity(x.len());
        for j in 0..x.len() {
            let cell = ((x[j] - self.low[j]) / self.res).floor();
            if !(cell >= 0.0) || cell >= self.cells[j] as f64 {
                return Err(CerebellumError::OutOfRange {
                    what: "input",
                    value: x[j],
                    low: self.low[j],
                    high: self.low[j] + self.res * self.cells[j] as f64,
                });
            }
            u.push(cell as usize);
        }
        Ok(u)
    }

    fn active_of(&self, u: &[usize]) -> Vec<usize> {
        let per = self.tiles_per_tiling();
        (0..self.c)
            .map(|k| {
                let mut index = 0usize;
                for j in 0..u.len() {
                    index = index * self.tiles_along(j) + (u[j] + k) / self.c;
                }
                k * per + index
            })
            .collect()
    }

    /// The `c` weights an input addresses, one per tiling.
    ///
    /// # Errors
    ///
    /// As [`Cmac::quantise`].
    pub fn active(&self, x: &[f64]) -> Result<Vec<usize>, CerebellumError> {
        Ok(self.active_of(&self.quantise(x)?))
    }

    /// How many tiles two inputs share: what is learned at one reaches the other in that
    /// proportion of `c`.
    ///
    /// # Errors
    ///
    /// As [`Cmac::quantise`].
    pub fn shared(&self, x: &[f64], y: &[f64]) -> Result<usize, CerebellumError> {
        let (a, b) = (self.active(x)?, self.active(y)?);
        Ok(a.iter().zip(&b).filter(|(p, q)| p == q).count())
    }

    /// The output: the sum of the addressed weights.
    ///
    /// # Errors
    ///
    /// As [`Cmac::quantise`].
    pub fn output(&self, x: &[f64]) -> Result<f64, CerebellumError> {
        Ok(self.active(x)?.iter().map(|&i| self.w[i]).sum())
    }

    /// One correction toward `target`: each addressed weight moves by `β (target − output)/c`.
    /// Returns the error `output − target` BEFORE the correction.
    ///
    /// # Errors
    ///
    /// As [`Cmac::quantise`], plus [`CerebellumError::NonFinite`] for a non-finite target.
    pub fn train(&mut self, x: &[f64], target: f64) -> Result<f64, CerebellumError> {
        if !target.is_finite() {
            return Err(CerebellumError::NonFinite { what: "target", index: 0 });
        }
        let cells = self.active(x)?;
        let error = cells.iter().map(|&i| self.w[i]).sum::<f64>() - target;
        let step = self.beta * error / self.c as f64;
        for &i in &cells {
            self.w[i] -= step;
        }
        Ok(error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::TAU;

    /// Quadrature inputs over two whole periods, and the target `a sin + b cos`.
    fn quadrature(a: f64, b: f64) -> (Vec<Vec<f64>>, Vec<f64>) {
        let n = 200;
        let samples: Vec<Vec<f64>> = (0..n).map(|k| { let t = 2.0 * TAU * k as f64 / n as f64; vec![t.sin(), t.cos()] }).collect();
        let targets = samples.iter().map(|p| a * p[0] + b * p[1]).collect();
        (samples, targets)
    }

    #[test]
    fn batch_learning_converges_on_the_wiener_solution_at_one_minus_beta_lambda() {
        let (samples, targets) = quadrature(1.5, -0.7);
        // Quadrature over whole periods: R = ½ I, exactly up to rounding.
        let r = correlation(&samples).unwrap();
        for (k, want) in [0.5, 0.0, 0.0, 0.5].iter().enumerate() {
            assert!((r[k] - want).abs() < 1e-15, "R[{k}] = {}", r[k]);
        }
        let best = wiener(&samples, &targets).unwrap();
        assert!((best[0] - 1.5).abs() < 1e-13 && (best[1] + 0.7).abs() < 1e-13, "{best:?}");
        // β = 0.4 and λ = ½: every epoch leaves 1 − 0.2 = 0.8 of the weight error, on both axes.
        let mut cell = AdaptiveFilter::new(2, 0.4).unwrap();
        for k in 0..30 {
            for (w, b) in cell.w.iter().zip(&best) {
                assert!((w - b * (1.0 - 0.8f64.powi(k))).abs() < 1e-13, "epoch {k}: {w} vs {}", b * (1.0 - 0.8f64.powi(k)));
            }
            cell.epoch(&samples, &targets).unwrap();
        }
        // The mean squared error is ½|w − w*|² here, so it falls by 0.64 an epoch.
        let mut fresh = AdaptiveFilter::new(2, 0.4).unwrap();
        let first = fresh.epoch(&samples, &targets).unwrap();
        let second = fresh.epoch(&samples, &targets).unwrap();
        assert!((first - 0.5 * (1.5f64 * 1.5 + 0.7 * 0.7)).abs() < 1e-13);
        assert!((second / first - 0.64).abs() < 1e-12);
    }

    #[test]
    fn learning_stops_exactly_when_the_error_is_decorrelated_from_every_input() {
        // Correlated inputs of unequal power, and a target that is NOT in their span: the error
        // cannot go to zero, and its correlation with the inputs still does.
        let n = 300;
        let samples: Vec<Vec<f64>> = (0..n)
            .map(|k| { let t = TAU * k as f64 / n as f64; vec![t.sin() + 0.3, 2.0 * (t + 0.6).sin(), (3.0 * t).cos() * 0.5] })
            .collect();
        let targets: Vec<f64> = (0..n).map(|k| { let t = TAU * k as f64 / n as f64; (2.0 * t).sin() + 0.8 * t.sin() - 0.2 }).collect();
        let best = wiener(&samples, &targets).unwrap();
        let beta = safe_beta(&samples).unwrap();
        let mut cell = AdaptiveFilter::new(3, beta).unwrap();
        let before = cell.error_correlation(&samples, &targets).unwrap();
        assert!(before.iter().any(|c| c.abs() > 0.1), "an untrained filter's error is correlated with its inputs: {before:?}");
        let mut mse = f64::INFINITY;
        for _ in 0..20_000 {
            let now = cell.epoch(&samples, &targets).unwrap();
            assert!(now <= mse * (1.0 + 1e-12), "the error rose from {mse} to {now} at the safe rate");
            mse = now;
        }
        for (w, b) in cell.w.iter().zip(&best) {
            assert!((w - b).abs() < 1e-9, "{w} vs the Wiener weight {b}");
        }
        let after = cell.error_correlation(&samples, &targets).unwrap();
        assert!(after.iter().all(|c| c.abs() < 1e-10), "{after:?}");
        assert!(mse > 0.1, "the target is outside the span, so an error of {mse} remains — decorrelated, not zero");
        // Past 2/λ_max it diverges. λ_max ≤ 1/safe_beta, so 2.5/λ_max is reached by 2.5/safe… only
        // if the bound is tight; use the exact λ of the quadrature problem instead.
        let (qs, qt) = quadrature(1.0, 1.0);
        let mut wild = AdaptiveFilter::new(2, 5.0).unwrap(); // β λ = 2.5 → factor −1.5
        let e0 = wild.epoch(&qs, &qt).unwrap();
        let e1 = wild.epoch(&qs, &qt).unwrap();
        assert!((e1 / e0 - 2.25).abs() < 1e-9, "at βλ = 2.5 the error grows by 1.5² an epoch: {}", e1 / e0);
        assert!((safe_beta(&qs).unwrap() - 2.0).abs() < 1e-12, "R = ½ I → 1/(row sum) = 2, half of 2/λ = 4");
        // The OFF-diagonal entries are what make it a bound. Two identical unit inputs have
        // R = [[1, 1], [1, 1]], λ_max = 2: the row sum gives β = ½; the diagonal alone would give
        // β = 1 = 2/λ_max, which is the edge itself — and on quadrature inputs the two agree, which
        // is why ignoring the off-diagonals survived this module's first mutation sweep.
        let twins = vec![vec![1.0, 1.0], vec![1.0, 1.0]];
        assert_eq!(correlation(&twins).unwrap(), vec![1.0; 4]);
        assert_eq!(safe_beta(&twins).unwrap(), 0.5);
    }

    #[test]
    fn online_learning_changes_only_the_fibres_that_were_active() {
        let mut cell = AdaptiveFilter::new(3, 0.1).unwrap();
        cell.w = vec![1.0, 1.0, 1.0];
        cell.learn(&[2.0, 0.0, -1.0], 0.5).unwrap();
        // Albus's sign, not Marr's (see the module doc). Active with a positive error: depressed.
        // Silent: untouched. Negative input: potentiated.
        assert_eq!(cell.w, vec![1.0 - 0.1, 1.0, 1.0 + 0.05]);
        assert_eq!(cell.output(&[1.0, 1.0, 1.0]).unwrap(), 0.9 + 1.0 + 1.05);
    }

    #[test]
    fn the_granule_bank_steps_by_the_exact_exponential() {
        let mut bank = GranuleBank::new(vec![0.01, 0.1, 1.0]).unwrap();
        bank.step(0.05, 2.0).unwrap();
        for (s, tau) in bank.state.iter().zip(&bank.taus) {
            assert!((s - 2.0 * (1.0 - (-0.05 / tau).exp())).abs() < 1e-15);
        }
        // Fifty small steps land where one large one does.
        let mut fine = GranuleBank::new(vec![0.01, 0.1, 1.0]).unwrap();
        for _ in 0..50 {
            fine.step(0.001, 2.0).unwrap();
        }
        for (a, b) in fine.state.iter().zip(&bank.state) {
            assert!((a - b).abs() < 1e-13);
        }
        // And the input removed, each decays at its own rate.
        let held = bank.state.clone();
        bank.step(0.1, 0.0).unwrap();
        for ((s, h), tau) in bank.state.iter().zip(&held).zip(&bank.taus) {
            assert!((s - h * (-0.1 / tau).exp()).abs() < 1e-15);
        }
    }

    #[test]
    fn in_one_dimension_generalisation_is_a_triangle() {
        let c = 8;
        let mut cmac = Cmac::new(vec![0.0], &[100.0], 1.0, c, 1.0).unwrap();
        assert_eq!(cmac.cells, vec![100]);
        // (99 + 7)/8 + 1 = 14 tiles a tiling, 8 tilings.
        assert_eq!(cmac.memory(), 8 * 14);
        assert_eq!(cmac.w.len(), 112);
        for u in 0..100usize {
            let a = cmac.active(&[u as f64 + 0.5]).unwrap();
            assert_eq!(a.len(), c);
            let mut sorted = a.clone();
            sorted.dedup();
            assert_eq!(sorted.len(), c, "a lookup addresses {c} DIFFERENT weights");
            for v in 0..100usize {
                let shared = cmac.shared(&[u as f64 + 0.5], &[v as f64 + 0.5]).unwrap();
                assert_eq!(shared, c.saturating_sub(u.abs_diff(v)), "cells {u} and {v}");
            }
        }
        // One full correction at cell 40 and the output everywhere is the target times the
        // shared fraction: a triangle of half-width C.
        assert_eq!(cmac.train(&[40.5], 4.0).unwrap(), -4.0);
        for v in 0..100usize {
            let want = 4.0 * c.saturating_sub(v.abs_diff(40)) as f64 / c as f64;
            assert!((cmac.output(&[v as f64 + 0.5]).unwrap() - want).abs() < 1e-15, "cell {v}");
        }
    }

    #[test]
    fn in_two_dimensions_the_shared_count_lies_between_its_two_bounds() {
        let c = 5;
        let cmac = Cmac::new(vec![0.0, 0.0], &[14.0, 14.0], 1.0, c, 1.0).unwrap();
        let at = |u: usize, v: usize| [u as f64 + 0.5, v as f64 + 0.5];
        let (mut lower_hit, mut upper_hit, mut strictly_between) = (false, false, false);
        for u0 in 0..14usize {
            for v0 in 0..14usize {
                for u1 in 0..14usize {
                    for v1 in 0..14usize {
                        let (du, dv) = (u0.abs_diff(u1), v0.abs_diff(v1));
                        let shared = cmac.shared(&at(u0, v0), &at(u1, v1)).unwrap();
                        let (lo, hi) = (c.saturating_sub(du + dv), c.saturating_sub(du.max(dv)));
                        assert!(shared >= lo && shared <= hi, "({u0},{v0})–({u1},{v1}): {shared} outside [{lo}, {hi}]");
                        if du > 0 && dv > 0 && lo < hi {
                            lower_hit |= shared == lo;
                            upper_hit |= shared == hi;
                            strictly_between |= shared > lo && shared < hi;
                        }
                    }
                }
            }
        }
        assert!(lower_hit && upper_hit, "both bounds are attained somewhere: {lower_hit} {upper_hit}");
        assert!(strictly_between || c < 4, "and the count is not always at a bound");
        // Along the diagonal the tilings move WITH the displacement and the upper bound is met.
        assert_eq!(cmac.shared(&at(3, 3), &at(5, 5)).unwrap(), 3);
    }

    #[test]
    fn learning_at_a_point_is_geometric_and_distant_points_do_not_interfere() {
        let mut cmac = Cmac::new(vec![-1.0], &[1.0], 0.01, 10, 0.25).unwrap();
        for k in 0..12 {
            let e = cmac.train(&[0.305], 2.0).unwrap();
            assert!((e + 2.0 * 0.75f64.powi(k)).abs() < 1e-14, "update {k}: error {e}");
        }
        // Training points C = 10 steps apart share no tile: one pass at β = 1 fits them all.
        let mut table = Cmac::new(vec![0.0], &[2.0], 0.01, 10, 1.0).unwrap();
        let points: Vec<f64> = (0..20).map(|k| 0.005 + 0.1 * f64::from(k)).collect();
        for &x in &points {
            table.train(&[x], (7.0 * x).sin()).unwrap();
        }
        for &x in &points {
            assert!((table.output(&[x]).unwrap() - (7.0 * x).sin()).abs() < 1e-15, "x = {x}");
        }
        // Nine steps apart they share one tile in ten, and the second correction disturbs the
        // first by a tenth of itself.
        let mut close = Cmac::new(vec![0.0], &[2.0], 0.01, 10, 1.0).unwrap();
        close.train(&[0.005], 1.0).unwrap();
        close.train(&[0.095], 3.0).unwrap();
        // The second point started at 0.1 (one shared tile), was corrected by 2.9, and a tenth of
        // that correction landed on the shared tile.
        assert!((close.output(&[0.095]).unwrap() - 3.0).abs() < 1e-15);
        assert!((close.output(&[0.005]).unwrap() - (1.0 + 0.29)).abs() < 1e-15);
    }

    #[test]
    fn bad_arguments_are_refused() {
        assert!(matches!(GranuleBank::new(vec![]), Err(CerebellumError::Empty { .. })));
        assert!(matches!(GranuleBank::new(vec![0.1, 0.0]), Err(CerebellumError::OutOfRange { what: "tau", .. })));
        let mut bank = GranuleBank::new(vec![0.1]).unwrap();
        assert!(matches!(bank.step(0.0, 1.0), Err(CerebellumError::OutOfRange { what: "dt", .. })));
        assert!(matches!(bank.step(0.1, f64::NAN), Err(CerebellumError::NonFinite { what: "input", .. })));
        assert!(matches!(AdaptiveFilter::new(0, 0.1), Err(CerebellumError::Empty { .. })));
        assert!(matches!(AdaptiveFilter::new(2, 0.0), Err(CerebellumError::OutOfRange { what: "beta", .. })));
        let mut cell = AdaptiveFilter::new(2, 0.1).unwrap();
        assert!(matches!(cell.output(&[1.0]), Err(CerebellumError::Dimension { what: "input", .. })));
        assert!(matches!(cell.learn(&[1.0, f64::NAN], 0.1), Err(CerebellumError::NonFinite { what: "input", .. })));
        assert!(matches!(cell.learn(&[1.0, 1.0], f64::INFINITY), Err(CerebellumError::NonFinite { what: "error", .. })));
        assert_eq!(cell.w, vec![0.0, 0.0], "a refused update wrote nothing");
        assert!(matches!(cell.epoch(&[], &[]), Err(CerebellumError::Empty { what: "samples" })));
        assert!(matches!(cell.epoch(&[vec![1.0, 2.0]], &[]), Err(CerebellumError::Dimension { what: "targets", .. })));
        assert!(matches!(cell.epoch(&[vec![1.0]], &[0.0]), Err(CerebellumError::Dimension { what: "sample", .. })));
        assert!(matches!(correlation(&[]), Err(CerebellumError::Empty { .. })));
        assert!(matches!(correlation(&[vec![1.0, 2.0], vec![1.0]]), Err(CerebellumError::Dimension { .. })));
        // Two identical inputs are linearly dependent: there is no unique Wiener solution.
        let twin: Vec<Vec<f64>> = (0..10).map(|k| vec![f64::from(k), f64::from(k)]).collect();
        assert!(matches!(wiener(&twin, &[0.0; 10]), Err(CerebellumError::Solve(_))));
        assert!(matches!(safe_beta(&[vec![0.0, 0.0]]), Err(CerebellumError::OutOfRange { what: "input power", .. })));
        assert!(matches!(Cmac::new(vec![], &[], 0.1, 4, 1.0), Err(CerebellumError::Empty { what: "dimensions" })));
        assert!(matches!(Cmac::new(vec![0.0], &[1.0], 0.1, 0, 1.0), Err(CerebellumError::Empty { what: "tilings" })));
        assert!(matches!(Cmac::new(vec![0.0], &[1.0, 2.0], 0.1, 4, 1.0), Err(CerebellumError::Dimension { what: "high", .. })));
        assert!(matches!(Cmac::new(vec![0.0], &[1.0], 0.0, 4, 1.0), Err(CerebellumError::OutOfRange { what: "res", .. })));
        assert!(matches!(Cmac::new(vec![0.0], &[0.0], 0.1, 4, 1.0), Err(CerebellumError::OutOfRange { what: "high", .. })));
        assert!(matches!(Cmac::new(vec![0.0], &[1.0], 0.1, 4, 1.5), Err(CerebellumError::OutOfRange { what: "beta", .. })));
        assert!(matches!(Cmac::new(vec![0.0], &[1.0], 0.1, 4, 0.0), Err(CerebellumError::OutOfRange { what: "beta", .. })));
        assert!(matches!(Cmac::new(vec![0.0; 4], &[1.0; 4], 1e-3, 2, 1.0), Err(CerebellumError::TooLarge { .. })));
        assert!(matches!(Cmac::new(vec![0.0], &[1e300], 1e-3, 2, 1.0), Err(CerebellumError::TooLarge { .. })));
        let mut cmac = Cmac::new(vec![0.0, -1.0], &[1.0, 1.0], 0.1, 4, 1.0).unwrap();
        assert_eq!(cmac.cells, vec![10, 20]);
        assert_eq!(cmac.memory(), 4 * 4 * 6, "(9+3)/4+1 = 4 and (19+3)/4+1 = 6 tiles, four tilings");
        assert_eq!(cmac.quantise(&[0.95, -1.0]).unwrap(), vec![9, 0]);
        assert!(matches!(cmac.output(&[1.0, 0.0]), Err(CerebellumError::OutOfRange { what: "input", .. })));
        assert!(matches!(cmac.output(&[-0.01, 0.0]), Err(CerebellumError::OutOfRange { what: "input", .. })));
        assert!(matches!(cmac.output(&[0.5]), Err(CerebellumError::Dimension { what: "input", .. })));
        assert!(matches!(cmac.output(&[0.5, f64::NAN]), Err(CerebellumError::NonFinite { what: "input", .. })));
        assert!(matches!(cmac.train(&[0.5, 0.0], f64::NAN), Err(CerebellumError::NonFinite { what: "target", .. })));
        assert!(cmac.w.iter().all(|w| *w == 0.0), "a refused correction wrote nothing");
        // The far corner's last tiling still lands inside the table.
        let corner = cmac.active(&[0.999, 0.999]).unwrap();
        assert!(corner.iter().all(|&i| i < cmac.w.len()));
        assert_eq!(corner[3], 3 * 24 + 3 * 6 + 5);
    }

    /// A non-finite entry is refused at ITS OWN index, whether it is a `NaN` or an infinity.
    /// The scan is written `!x.is_finite()`; an `is_nan()` in its place still rejects everything
    /// the suite hands it, because every existing non-finite fixture in this module is a `NaN`,
    /// and every one of them sits in slot 0 or slot 1 of a two-element array, where an index
    /// hard-coded to zero is right half the time and unread the other half.
    #[test]
    fn a_non_finite_entry_is_refused_at_its_own_index_whether_nan_or_infinite() {
        let mut cell = AdaptiveFilter::new(4, 0.1).unwrap();
        for (slot, bad) in [(0usize, f64::NAN), (1, f64::INFINITY), (3, f64::NEG_INFINITY)] {
            let mut p = vec![1.0; 4];
            p[slot] = bad;
            match cell.output(&p) {
                Err(CerebellumError::NonFinite { what, index }) => assert_eq!((what, index), ("input", slot)),
                other => panic!("an input of {bad} at slot {slot} was accepted: {other:?}"),
            }
            match cell.learn(&p, 1.0) {
                Err(CerebellumError::NonFinite { what, index }) => assert_eq!((what, index), ("input", slot)),
                other => panic!("a learning input of {bad} at slot {slot} was accepted: {other:?}"),
            }
        }
        // The same scan under the names `check_samples` and the CMAC give it.
        match cell.epoch(&[vec![1.0, 1.0, f64::INFINITY, 1.0]], &[0.0]) {
            Err(CerebellumError::NonFinite { what, index }) => assert_eq!((what, index), ("sample", 2)),
            other => panic!("a sample carrying an infinity was accepted: {other:?}"),
        }
        let cmac = Cmac::new(vec![0.0, 0.0], &[1.0, 1.0], 0.1, 4, 1.0).unwrap();
        match cmac.quantise(&[0.5, f64::INFINITY]) {
            Err(CerebellumError::NonFinite { what, index }) => assert_eq!((what, index), ("input", 1)),
            other => panic!("a lookup at an infinite coordinate was accepted: {other:?}"),
        }
    }

    /// An infinite parameter is not a positive one. `positive` reads `is_finite() && v > 0.0`,
    /// and the suite's only probes of it are zero and a negative number, both of which the
    /// comparison alone still rejects — so `+∞` seconds of time constant, of time step or of
    /// learning rate walked through a guard that exists to stop exactly that.
    #[test]
    fn an_infinite_parameter_is_not_a_positive_one() {
        assert!(matches!(GranuleBank::new(vec![0.1, f64::INFINITY]), Err(CerebellumError::OutOfRange { what: "tau", .. })));
        let mut bank = GranuleBank::new(vec![0.1]).unwrap();
        assert!(matches!(bank.step(f64::INFINITY, 1.0), Err(CerebellumError::OutOfRange { what: "dt", .. })));
        assert!(matches!(AdaptiveFilter::new(2, f64::INFINITY), Err(CerebellumError::OutOfRange { what: "beta", .. })));
        assert!(matches!(Cmac::new(vec![0.0], &[1.0], f64::INFINITY, 4, 1.0), Err(CerebellumError::OutOfRange { what: "res", .. })));
    }

    /// A non-finite target is refused, by name and at its own index. `check_samples` ends on
    /// the targets, and that call's value is the function's return value — so replacing it with
    /// `Ok(())` removes the check and leaves the signature and every caller untouched. A `NaN`
    /// target does not panic: it makes the mean squared error `NaN` and poisons every weight,
    /// which every existing assertion about convergence reads as a run that simply did not
    /// converge on the fixtures it is given, none of which carry one.
    #[test]
    fn a_non_finite_target_is_refused_before_it_can_poison_a_weight() {
        let mut cell = AdaptiveFilter::new(2, 0.1).unwrap();
        let samples = vec![vec![1.0, 2.0], vec![0.5, -1.0]];
        for (slot, bad) in [(0usize, f64::NAN), (1, f64::INFINITY)] {
            let mut targets = vec![1.0, 2.0];
            targets[slot] = bad;
            match cell.error_correlation(&samples, &targets) {
                Err(CerebellumError::NonFinite { what, index }) => assert_eq!((what, index), ("targets", slot)),
                other => panic!("a target of {bad} at slot {slot} was accepted: {other:?}"),
            }
            assert!(matches!(cell.epoch(&samples, &targets), Err(CerebellumError::NonFinite { what: "targets", .. })));
            assert!(matches!(wiener(&samples, &targets), Err(CerebellumError::NonFinite { what: "targets", .. })));
        }
        assert_eq!(cell.w, vec![0.0, 0.0], "a refused epoch wrote nothing");
        // The same sample set with finite targets does solve, so the refusal is the target's.
        assert!(wiener(&samples, &[1.0, 2.0]).is_ok());
    }

    /// The bank holds one state per time constant and the step hands every one of them back.
    /// The exponential test zips the states with the time constants, so a bank holding a single
    /// state checks a single integrator and passes; and it discards the step's return value
    /// entirely, reading `bank.state` afterwards, so a step handing back an empty slice — the
    /// only thing a caller who is not inside the module can see — was read by nothing.
    #[test]
    fn the_bank_holds_one_state_per_time_constant_and_hands_every_one_back() {
        let mut bank = GranuleBank::new(vec![0.01, 0.1, 1.0]).unwrap();
        assert_eq!(bank.state.len(), bank.taus.len());
        assert_eq!(bank.state, vec![0.0; 3]);
        let returned = bank.step(0.05, 2.0).unwrap().to_vec();
        assert_eq!(returned.len(), 3, "the step handed back {} of 3 states", returned.len());
        assert_eq!(returned, bank.state);
        // 50 ms is five time constants for the fastest integrator and a twentieth for the
        // slowest, so the three are strictly ordered and all three have moved off rest.
        assert!(returned[0] > returned[1] && returned[1] > returned[2] && returned[2] > 0.0, "{returned:?}");
    }

    /// A non-finite sample never enters the correlation matrix. `correlation` is the entry point
    /// `wiener` and `safe_beta` both go through FIRST, before `check_samples` scans anything, so
    /// its own scan is the only thing standing between a `NaN` sample and an `R` that is `NaN`
    /// throughout — and the existing bad-argument probes of `correlation` are all ragged lengths.
    #[test]
    fn a_non_finite_sample_never_enters_the_correlation_matrix() {
        for (slot, bad) in [(0usize, f64::NAN), (1, f64::INFINITY)] {
            let mut p = vec![1.0, 2.0];
            p[slot] = bad;
            match correlation(&[vec![1.0, 1.0], p]) {
                Err(CerebellumError::NonFinite { what, index }) => assert_eq!((what, index), ("sample", slot)),
                other => panic!("a sample of {bad} at slot {slot} entered the matrix: {other:?}"),
            }
        }
        assert!(matches!(safe_beta(&[vec![1.0, f64::NAN]]), Err(CerebellumError::NonFinite { what: "sample", .. })));
        assert!(matches!(wiener(&[vec![1.0, f64::NAN]], &[1.0]), Err(CerebellumError::NonFinite { what: "sample", .. })));
    }

    /// The Wiener solve refuses a sample set that is NEARLY dependent, not only one that is
    /// exactly so. The guard `cholesky` is given is RELATIVE — a pivot is rejected at or below
    /// `1e-12` of the largest diagonal entry — and the only ill-conditioned fixture in this
    /// module is two identical inputs, whose second pivot comes out at or below zero and is
    /// therefore refused by a guard of zero as well. Two inputs differing by one part in ten
    /// million separate the two: measured, the second pivot is 4.33e-15 against a scale of
    /// 0.5000000000000047 — positive, so a guard of zero admits it, and 8.7e-15 of the scale,
    /// so the relative guard does not.
    #[test]
    fn the_wiener_solve_refuses_a_nearly_dependent_sample_set() {
        let n = 400;
        let near: Vec<Vec<f64>> = (0..n)
            .map(|k| { let t = TAU * k as f64 / n as f64; vec![t.sin(), t.sin() + 1e-7 * t.cos()] })
            .collect();
        let targets: Vec<f64> = near.iter().map(|p| 2.0 * p[0] - 0.5 * p[1]).collect();
        match wiener(&near, &targets) {
            Err(CerebellumError::Solve(ReservoirError::IllConditioned { index, .. })) => assert_eq!(index, 1),
            other => panic!("two inputs a part in ten million apart were solved: {other:?}"),
        }
        // The same size and the same targets on genuinely independent inputs still solve, so the
        // refusal is the conditioning and not the fixture.
        let apart: Vec<Vec<f64>> = (0..n)
            .map(|k| { let t = TAU * k as f64 / n as f64; vec![t.sin(), t.cos()] })
            .collect();
        assert!(wiener(&apart, &targets).is_ok());
    }

    /// A CMAC refuses a non-finite lower corner and a `NaN` learning rate. The corner scan's
    /// result is discarded by the mutation rather than its call, so the box is then built from
    /// a `NaN` bound and fails later with a DIFFERENT complaint — which is why the variant is
    /// asserted here and not merely the failure. And `!(beta > 0.0) || !(beta <= 1.0)` is the
    /// rejecting form of the range check: written as `beta <= 0.0 || beta > 1.0` it admits a
    /// `NaN`, which then multiplies every correction the table ever makes.
    #[test]
    fn a_cmac_refuses_a_non_finite_corner_or_a_nan_learning_rate() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            match Cmac::new(vec![0.0, bad], &[1.0, 1.0], 0.1, 4, 1.0) {
                Err(CerebellumError::NonFinite { what, index }) => assert_eq!((what, index), ("low", 1)),
                other => panic!("a lower corner of {bad} was accepted: {other:?}"),
            }
        }
        assert!(matches!(Cmac::new(vec![0.0], &[1.0], 0.1, 4, f64::NAN), Err(CerebellumError::OutOfRange { what: "beta", .. })));
    }

    /// The quantisation cell count COVERS the top of the box: a box one and a half steps wide
    /// has two cells, the upper one a half-cell. Every box in this module's existing tests is a
    /// whole number of steps across — 100/1, 1/0.1, 2/0.1, 2/0.01 — where rounding the count up
    /// and rounding it down give the same answer, so a count that truncated would have left the
    /// top slice of every ragged box outside the table with nothing to say so.
    #[test]
    fn the_cell_count_covers_the_top_of_a_box_that_is_not_a_whole_number_of_steps() {
        let ragged = Cmac::new(vec![0.0], &[1.5], 1.0, 2, 1.0).unwrap();
        assert_eq!(ragged.cells, vec![2]);
        assert_eq!(ragged.quantise(&[1.2]).unwrap(), vec![1], "the top half-step of the box has a cell");
        assert!(matches!(ragged.quantise(&[2.0]), Err(CerebellumError::OutOfRange { what: "input", .. })));
        // And a box that IS a whole number of steps is not rounded up past it.
        assert_eq!(Cmac::new(vec![0.0], &[3.0], 1.0, 2, 1.0).unwrap().cells, vec![3]);
    }

    /// The table ceiling is the published one, the per-axis cell count is capped against it, and
    /// the table size SATURATES. The existing size probes reach for a table so large that every
    /// route to refusing it reports the same saturated count, so which check fired was invisible
    /// and a ceiling a thousand times higher refused them just the same. The three fixtures here
    /// separate them: 67108865 cells on one axis is past the per-axis cap but its tile count is
    /// only 67108868, so the two checks name different numbers; and three axes of 4194302 cells
    /// at two tilings is 2 × 2^63 tiles, a product that OVERFLOWS a `usize` and wraps to exactly
    /// zero, which would build a table of no weights and index past the end of it on the first
    /// lookup.
    #[test]
    fn the_table_ceiling_is_the_published_one_and_the_size_saturates() {
        const { assert!(MAX_WEIGHTS == 1 << 26, "the published table ceiling is 2^26 weights") };
        assert_eq!(
            Cmac::new(vec![0.0], &[67108865.0], 1.0, 4, 1.0),
            Err(CerebellumError::TooLarge { weights: usize::MAX })
        );
        assert_eq!(
            Cmac::new(vec![0.0; 3], &[4194302.0; 3], 1.0, 2, 1.0),
            Err(CerebellumError::TooLarge { weights: usize::MAX })
        );
    }

    /// The tiles of one tiling are laid out ROW-MAJOR, as the `Cmac::w` doc says: the last
    /// dimension moves fastest. The two-dimensional tests use a square box, where reversing the
    /// dimension order is a transposition — a bijection on tile indices — so every shared count
    /// is preserved exactly; and the one explicit index the suite checks is `3 × 6 + 5 = 23`
    /// against a transposed `5 × 4 + 3 = 23`, which is the same number by coincidence. This
    /// fixture has four tiles along one axis and six along the other and reads a tile where the
    /// two orders disagree.
    #[test]
    fn the_tiles_of_one_tiling_are_laid_out_row_major() {
        let cmac = Cmac::new(vec![0.0, -1.0], &[1.0, 1.0], 0.1, 4, 1.0).unwrap();
        assert_eq!(cmac.cells, vec![10, 20]);
        assert_eq!(cmac.memory(), 96, "4 tiles along the first axis and 6 along the second, four tilings");
        assert_eq!(cmac.quantise(&[0.45, -0.95]).unwrap(), vec![4, 0]);
        // Quantisation cell (4, 0) is tile (1, 0) of tiling 0, and row-major puts it at 1×6 + 0.
        assert_eq!(cmac.active(&[0.45, -0.95]).unwrap(), vec![6, 30, 54, 78]);
        // A step of one tile along the LAST axis is a step of one in the index.
        assert_eq!(cmac.active(&[0.45, -0.55]).unwrap(), vec![7, 31, 55, 79]);
        // A step of one tile along the FIRST axis is a step of a whole row, six.
        assert_eq!(cmac.active(&[0.85, -0.95]).unwrap(), vec![12, 36, 60, 84]);
    }

    /// Every error variant renders its own fields in the order its sentence names them. Every
    /// other test in this module destructures the variant and reads `what`, so nothing renders
    /// one — and a message reporting the required length as the supplied one, and the supplied
    /// as the required, says the exact opposite of the truth to the only audience a message has.
    #[test]
    fn every_error_renders_its_values_in_the_order_its_sentence_names_them() {
        assert_eq!(CerebellumError::Empty { what: "tilings" }.to_string(), "tilings is empty");
        let dimension = CerebellumError::Dimension { what: "targets", got: 3, want: 5 };
        assert_eq!(dimension.to_string(), "targets has length 3, expected 5");
        let out_of_range = CerebellumError::OutOfRange { what: "beta", value: 1.5, low: 0.0, high: 1.0 };
        assert_eq!(out_of_range.to_string(), "beta = 1.5 is outside [0, 1]");
        assert_eq!(CerebellumError::NonFinite { what: "sample", index: 2 }.to_string(), "sample is not finite at 2");
        let too_large = CerebellumError::TooLarge { weights: 7 };
        assert_eq!(too_large.to_string(), "a table of 7 weights is more than MAX_WEIGHTS");
    }
}
