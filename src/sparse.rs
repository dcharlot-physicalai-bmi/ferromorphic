//! Sparse coding in spikes: the locally competitive algorithm, its spiking form, and the
//! optimisation problem both of them solve — checked against that problem's own optimality
//! conditions rather than against a previous run.
//!
//! # What the mechanism is
//!
//! Given a dictionary of `n` atoms `Φ` and a signal `y`, find a few coefficients `a` such that
//! `Φ a ≈ y`. That is the LASSO,
//!
//! ```text
//! minimise  ½ |y − Φ a|² + λ |a|₁
//! ```
//!
//! and Rozell, Johnson, Baraniuk and Olshausen, *Sparse coding via thresholding and local
//! competition in neural circuits*, Neural Computation 20(10):2526–2563, 2008, showed it is solved
//! by a network of neurons that each carry an internal state `u_i`, output a thresholded
//! coefficient `a_i = T_λ(u_i)`, and **inhibit each other in proportion to how alike their atoms
//! are**:
//!
//! ```text
//! τ du_i/dt = −u_i + φ_iᵀ y − Σ_{j ≠ i} (φ_iᵀ φ_j) a_j
//! ```
//!
//! The first driving term is how well the atom matches the signal; the second is the local
//! competition — a neuron whose atom is already explaining the signal suppresses its near
//! duplicates. The fixed points of this dynamics are exactly the minimisers of the LASSO when
//! `T_λ` is the soft threshold `sign(u)·max(|u| − λ, 0)`, and the objective is non-increasing
//! along every trajectory, which is the theorem the module's tests turn into assertions.
//!
//! # Why it is in a neuromorphic crate
//!
//! The LCA is the canonical non-machine-learning workload for neuromorphic hardware, and the
//! paper that introduced Loihi ran it against a CPU: Davies et al., *Loihi: a neuromorphic
//! manycore processor with on-chip learning*, IEEE Micro 38(1):82–99 (2018),
//! doi:10.1109/MM.2018.112130359. Its abstract: "Running a spiking convolutional form of the
//! Locally Competitive Algorithm, Loihi can solve LASSO optimization problems with over three
//! orders of magnitude superior energy-delay-product compared to conventional solvers running on
//! a CPU iso-process/voltage/area." Davies et al., *Advancing neuromorphic computing with Loihi:
//! a survey of results and outlook*, Proceedings of the IEEE 109(5):911–934 (2021),
//! doi:10.1109/JPROC.2021.3067593, reviews the result; its Fig. 4 sets LCA on Loihi against
//! FISTA on an i7 CPU, as time to solution and dynamic energy at about 1% of the optimal LASSO
//! objective.
//!
//! **Correction.** This paragraph used to say that the LCA "was the first algorithm demonstrated
//! on Loihi to beat a CPU on energy-delay product for the same solution quality", citing the 2021
//! survey as "reviewing Tang, Lin and Davies" (2017). The survey credits other work. Its LCA
//! section (pp. 918–919): "Previous results \[13\], \[53\] demonstrated the efficiency of
//! neuromorphic architectures, such as Loihi, for solving LASSO problems with LCA, especially the
//! convolutional form of the problem." Its \[13\] is the 2018 IEEE Micro paper above; its \[53\] is
//! Shapero, Rozell and Hasler, *Configurable hardware integrate and fire neurons for sparse
//! approximation*, Neural Networks 45:134–143 (2013), doi:10.1016/j.neunet.2013.03.012. The survey
//! cites Tang, Lin and Davies once, as \[51\], and for something else: that "it is often possible
//! to rigorously link the rate neuron dynamics with the corresponding dynamics of an equivalent
//! SNN". That paper is Tang, Lin and Davies, *Sparse coding by spiking neural networks:
//! convergence theory and computational results*, arXiv:1705.05475 (2017): convergence theory for
//! spiking LCA and a simulation of it on one core of a 2.3 GHz Intel Xeon, timed against FISTA on
//! the same CPU, where "the spiking network delivers much faster early convergence than FISTA"
//! (§4.3, Fig. 3). This review did not locate the word Loihi, or any energy or energy-delay figure,
//! in it. "First" was this crate's word: this review did not locate it in the survey's LCA
//! section, and the 2018 abstract calls the result "an unambiguous example of spike-based
//! computation". Docs only; no test or constant rests on this paragraph.
//!
//! What makes it a fit is the shape of the computation: `n` neurons, an `n × n` lateral
//! inhibition, and a solution that is mostly zeros — so most neurons are silent most of the time,
//! and a spiking implementation pays for the lateral matrix **only when a neuron fires**.
//! [`SpikingLca`] counts exactly that: each spike delivers `n − 1` lateral operations, and
//! [`lateral_ops_per_step`] is the `n(n − 1)` a rate implementation pays every step whether
//! anything changed or not. The ratio is the sparsity of the *activity*, which is the number the
//! whole argument turns on, and it is measured here rather than assumed.
//!
//! # The two thresholds
//!
//! [`Threshold::Soft`] is the proximal operator of `λ|a|` and gives the LASSO. [`Threshold::Hard`]
//! is the proximal operator of `(λ²/2)·|a|₀` and gives a non-convex problem whose fixed points are
//! local minima. Both are exact proximal maps and the test checks each against a grid search over
//! its own objective, so the correspondence is verified rather than quoted.
//!
//! # The spiking form
//!
//! [`SpikingLca`] encodes each coefficient as a spike rate: neuron `i`'s output `a_i` accumulates
//! `a_i · dt` per step and emits a spike each time the accumulator crosses one, so its rate **is**
//! `a_i` in hertz-per-unit-coefficient; the neighbours see it through a synaptic trace normalised
//! to unit gain, whose mean is that rate. This is the mean-field equivalence spiking LCA rests on
//! (Shapero, Zhu, Hasler and Rozell, *Optimal sparse approximation with integrate and fire
//! neurons*, International Journal of Neural Systems 24(5):1440001, 2014, and Tang, Lin and Davies
//! above), implemented with this crate's sigma-delta encoding rather than either paper's neuron
//! model — so what is claimed is that the rates converge to the same solution, and that is what
//! is tested, to the percent the trace noise allows.
//!
//! # What this module has NOT reproduced
//!
//! - Dictionary learning. The atoms are given; learning them (Olshausen and Field, *Emergence of
//!   simple-cell receptive field properties by learning a sparse code for natural images*, Nature
//!   381(6583):607–609 (1996), doi:10.1038/381607a0) is a gradient step outside the loop and is
//!   not here. This item used to cite "Olshausen and Field, 1996", which fits two papers of theirs
//!   from that year; the other is *Natural image statistics and efficient coding*, Network:
//!   Computation in Neural Systems 7(2):333–339 (1996), doi:10.1088/0954-898x_7_2_014, and the
//!   Nature paper is the one meant.
//! - Loihi's fixed-point arithmetic, its 8-bit weights or its published energy-delay figures.
//!   Nothing here is a measurement of any chip; the operation counts are exact integers about
//!   this implementation.
//! - A convergence rate. The tests assert convergence at a stated horizon, not a rate, and none of
//!   the rate results below is reproduced. This item used to say that "Tang, Lin and Davies bound
//!   the time to an `ε`-solution". They do not: they prove convergence, not a rate, of the spike
//!   rates to the solution of the non-negative LASSO (their CLASSO) as `t → ∞`. Their Eq. (8),
//!   `T_λ(u_i(t)) − a_i(t) → 0`, rests on "a moderate assumption that inter-spike
//!   duration `t_{i,k+1} − t_{i,k}` cannot be arbitrarily long unless neuron-i stops spiking
//!   altogether", and the limit point comes from Bolzano–Weierstrass. The `1/t` bounds in their
//!   appendix are steps inside the proof of Theorem 2, on one neuron's rate-versus-current gap,
//!   not on the distance to the optimum. Rates are published elsewhere. Shapero, Zhu, Hasler and
//!   Rozell (2014, above) report that "the firing rate of the Spiking LCA converges on the same
//!   solution as the analog LCA, with an error inversely proportional to the sampling time".
//!   Balavoine, Romberg and Rozell, *Convergence and rate analysis of neural networks for sparse
//!   approximation*, IEEE Transactions on Neural Networks and Learning Systems 23(9):1377–1389
//!   (2012), doi:10.1109/TNNLS.2012.2202400, show that under restrictions on the problem the
//!   analog LCA "converges exponentially fast with an analytically bounded convergence rate".
//!   Docs only; no test or constant implemented a rate.

use core::fmt;

use crate::ledger::Ledger;
use crate::rng::Rng;

/// What went wrong, named rather than guessed around.
#[derive(Debug, Clone, PartialEq)]
pub enum SparseError {
    /// A count of zero where at least one is needed.
    Empty {
        /// What was empty.
        what: &'static str,
    },
    /// A vector or matrix of the wrong size.
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
    /// A parameter outside its admissible range.
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

impl fmt::Display for SparseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty { what } => write!(f, "{what} is empty"),
            Self::Dimension { what, got, want } => write!(f, "{what} has {got} entries, needs {want}"),
            Self::NonFinite { what, index } => write!(f, "{what} is not finite at {index}"),
            Self::OutOfRange { what, value, low, high } => {
                write!(f, "{what} = {value} is outside [{low}, {high}]")
            }
        }
    }
}

impl std::error::Error for SparseError {}

fn finite(what: &'static str, v: &[f64]) -> Result<(), SparseError> {
    if let Some(i) = v.iter().position(|x| !x.is_finite()) {
        return Err(SparseError::NonFinite { what, index: i });
    }
    Ok(())
}

fn len(what: &'static str, got: usize, want: usize) -> Result<(), SparseError> {
    if got == want { Ok(()) } else { Err(SparseError::Dimension { what, got, want }) }
}

fn positive(what: &'static str, value: f64) -> Result<(), SparseError> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(SparseError::OutOfRange { what, value, low: f64::MIN_POSITIVE, high: f64::INFINITY })
    }
}

// ---------------------------------------------------------------------------------------------
// Dictionaries
// ---------------------------------------------------------------------------------------------

/// A dictionary of `n` atoms in `m` dimensions, row-major `m × n`: `atoms[r * n + j]` is
/// component `r` of atom `j`.
#[derive(Debug, Clone, PartialEq)]
pub struct Dictionary {
    /// Signal dimension.
    pub m: usize,
    /// Number of atoms.
    pub n: usize,
    /// The atoms, row-major `m × n`.
    pub atoms: Vec<f64>,
}

impl Dictionary {
    /// Build from an explicit matrix.
    ///
    /// # Errors
    ///
    /// [`SparseError::Empty`] for `m = 0` or `n = 0`, [`SparseError::Dimension`] if `atoms` is
    /// not `m × n`, [`SparseError::NonFinite`] for a non-finite entry.
    pub fn new(m: usize, n: usize, atoms: Vec<f64>) -> Result<Self, SparseError> {
        if m == 0 {
            return Err(SparseError::Empty { what: "signal dimension" });
        }
        if n == 0 {
            return Err(SparseError::Empty { what: "atoms" });
        }
        len("atoms", atoms.len(), m * n)?;
        finite("atoms", &atoms)?;
        Ok(Self { m, n, atoms })
    }

    /// The identity: `n` atoms that are the standard basis of `R^n`, an orthonormal dictionary
    /// for which every answer has a closed form.
    ///
    /// # Errors
    ///
    /// [`SparseError::Empty`] for `n = 0`.
    pub fn identity(n: usize) -> Result<Self, SparseError> {
        let mut atoms = vec![0.0; n * n];
        for i in 0..n {
            atoms[i * n + i] = 1.0;
        }
        Self::new(n, n, atoms)
    }

    /// `n` random unit atoms in `m` dimensions, Gaussian directions normalised.
    ///
    /// # Errors
    ///
    /// As [`Dictionary::new`].
    pub fn random_unit(m: usize, n: usize, rng: &mut Rng) -> Result<Self, SparseError> {
        if m == 0 {
            return Err(SparseError::Empty { what: "signal dimension" });
        }
        if n == 0 {
            return Err(SparseError::Empty { what: "atoms" });
        }
        let mut atoms = vec![0.0; m * n];
        for j in 0..n {
            let mut col: Vec<f64> = (0..m).map(|_| normal(rng)).collect();
            let mut norm = col.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm == 0.0 {
                col[0] = 1.0;
                norm = 1.0;
            }
            for (r, c) in col.iter().enumerate() {
                atoms[r * n + j] = c / norm;
            }
        }
        Self::new(m, n, atoms)
    }

    /// Atom `j` as a vector.
    #[must_use]
    pub fn column(&self, j: usize) -> Vec<f64> {
        (0..self.m).map(|r| self.atoms[r * self.n + j]).collect()
    }

    /// `Φᵀ y`: how well each atom matches the signal.
    ///
    /// # Errors
    ///
    /// [`SparseError::Dimension`], [`SparseError::NonFinite`].
    pub fn project(&self, y: &[f64]) -> Result<Vec<f64>, SparseError> {
        len("signal", y.len(), self.m)?;
        finite("signal", y)?;
        Ok((0..self.n).map(|j| (0..self.m).map(|r| self.atoms[r * self.n + j] * y[r]).sum()).collect())
    }

    /// `Φ a`: the signal the coefficients reconstruct.
    ///
    /// # Errors
    ///
    /// [`SparseError::Dimension`], [`SparseError::NonFinite`].
    pub fn synthesise(&self, a: &[f64]) -> Result<Vec<f64>, SparseError> {
        len("coefficients", a.len(), self.n)?;
        finite("coefficients", a)?;
        Ok((0..self.m).map(|r| (0..self.n).map(|j| self.atoms[r * self.n + j] * a[j]).sum()).collect())
    }

    /// The Gram matrix `ΦᵀΦ`, row-major `n × n`: the lateral inhibition, before its diagonal is
    /// removed.
    #[must_use]
    pub fn gram(&self) -> Vec<f64> {
        let n = self.n;
        let mut g = vec![0.0; n * n];
        for i in 0..n {
            for j in i..n {
                let dot: f64 = (0..self.m).map(|r| self.atoms[r * n + i] * self.atoms[r * n + j]).sum();
                g[i * n + j] = dot;
                g[j * n + i] = dot;
            }
        }
        g
    }

    /// The mutual coherence `max_{i ≠ j} |φ_iᵀ φ_j|`, the standard measure of how hard the
    /// dictionary makes the problem. Zero for an orthogonal dictionary; `None` for one atom.
    #[must_use]
    pub fn coherence(&self) -> Option<f64> {
        if self.n < 2 {
            return None;
        }
        let g = self.gram();
        let n = self.n;
        let mut worst = 0.0f64;
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    worst = worst.max(g[i * n + j].abs());
                }
            }
        }
        Some(worst)
    }
}

/// A source of uniform draws on `[0, 1)`.
///
/// [`normal`] reaches its generator through this trait rather than through [`Rng`] directly, for
/// one reason: the guard on a zero draw. `ln(0)` is `-inf`, and `Rng::next_f64` builds its result
/// from 53 bits, so it returns exactly zero with probability `2^-53` per draw — about one draw in
/// nine thousand million million. No test can wait for that, so the guard was reachable from
/// nothing and a sweep that deleted it saw every test pass. A one-line stub that always hands back
/// zero reaches it in one call.
trait Uniform {
    /// The next uniform draw on `[0, 1)`.
    fn next_f64(&mut self) -> f64;
}

impl Uniform for Rng {
    fn next_f64(&mut self) -> f64 {
        Rng::next_f64(self)
    }
}

/// A standard normal draw by Box-Muller on the crate's generator.
fn normal<R: Uniform>(rng: &mut R) -> f64 {
    let u1 = rng.next_f64().max(1e-300);
    let u2 = rng.next_f64();
    (-2.0 * u1.ln()).sqrt() * (core::f64::consts::TAU * u2).cos()
}

// ---------------------------------------------------------------------------------------------
// Thresholds
// ---------------------------------------------------------------------------------------------

/// The output nonlinearity `a = T_λ(u)`, and the penalty it is the proximal operator of.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Threshold {
    /// `sign(u) · max(|u| − λ, 0)`: the proximal operator of `λ|a|`, giving the LASSO.
    Soft,
    /// `u` if `|u| > λ`, else `0`: the proximal operator of `(λ²/2)·[a ≠ 0]`.
    Hard,
}

impl Threshold {
    /// `T_λ(u)`.
    #[must_use]
    pub fn apply(self, u: f64, lambda: f64) -> f64 {
        match self {
            Self::Soft => {
                if u > lambda {
                    u - lambda
                } else if u < -lambda {
                    u + lambda
                } else {
                    0.0
                }
            }
            Self::Hard => {
                if u.abs() > lambda { u } else { 0.0 }
            }
        }
    }

    /// The penalty this threshold is the proximal operator of, at coefficient `a`: `λ|a|` for
    /// soft, `λ²/2` for a non-zero `a` under hard.
    #[must_use]
    pub fn penalty(self, a: f64, lambda: f64) -> f64 {
        match self {
            Self::Soft => lambda * a.abs(),
            Self::Hard => {
                if a == 0.0 { 0.0 } else { 0.5 * lambda * lambda }
            }
        }
    }
}

/// The lateral operations a rate implementation performs per step: every neuron reads every other
/// neuron's coefficient, `n(n − 1)` multiply-accumulates, whether or not anything changed.
#[must_use]
pub fn lateral_ops_per_step(n: u64) -> u64 {
    n.saturating_mul(n.saturating_sub(1))
}

// ---------------------------------------------------------------------------------------------
// The rate LCA
// ---------------------------------------------------------------------------------------------

/// The locally competitive algorithm in rate form.
#[derive(Debug, Clone, PartialEq)]
pub struct Lca {
    /// The dictionary.
    pub dict: Dictionary,
    /// The threshold `λ`, which is the sparsity penalty.
    pub lambda: f64,
    /// Neuron time constant, seconds.
    pub tau: f64,
    /// The output nonlinearity.
    pub threshold: Threshold,
    /// Internal states `u`.
    pub u: Vec<f64>,
    /// `ΦᵀΦ − I`, row-major, the lateral inhibition with the self term removed.
    lateral: Vec<f64>,
}

/// What a run of the rate LCA produced.
#[derive(Debug, Clone, PartialEq)]
pub struct LcaRun {
    /// The coefficients at the end.
    pub coefficients: Vec<f64>,
    /// The objective at the start and the end.
    pub objective: (f64, f64),
    /// The largest single-step increase of the objective along the trajectory — zero, to
    /// rounding, for a correct implementation.
    pub worst_increase: f64,
    /// How many coefficients are non-zero at the end.
    pub nonzeros: usize,
    /// Steps run.
    pub steps: usize,
}

impl Lca {
    /// Build over `dict` with threshold `lambda` and time constant `tau`, at rest.
    ///
    /// # Errors
    ///
    /// [`SparseError::OutOfRange`] for a non-positive `lambda` or `tau`.
    pub fn new(dict: Dictionary, lambda: f64, tau: f64, threshold: Threshold) -> Result<Self, SparseError> {
        positive("lambda", lambda)?;
        positive("tau", tau)?;
        let n = dict.n;
        let mut lateral = dict.gram();
        for i in 0..n {
            lateral[i * n + i] = 0.0;
        }
        Ok(Self { dict, lambda, tau, threshold, u: vec![0.0; n], lateral })
    }

    /// Return every state to zero.
    pub fn reset(&mut self) {
        self.u.iter_mut().for_each(|x| *x = 0.0);
    }

    /// The current coefficients `T_λ(u)`.
    #[must_use]
    pub fn coefficients(&self) -> Vec<f64> {
        self.u.iter().map(|&u| self.threshold.apply(u, self.lambda)).collect()
    }

    /// One step of `dt` seconds toward the signal whose projection is `b = Φᵀ y`, holding the
    /// coefficients fixed over the step (exponential Euler on the linear part, exact for that).
    ///
    /// # Errors
    ///
    /// [`SparseError::Dimension`] for a `b` of the wrong length, [`SparseError::OutOfRange`] for a
    /// non-positive `dt`.
    pub fn step(&mut self, dt: f64, b: &[f64]) -> Result<(), SparseError> {
        len("projection", b.len(), self.dict.n)?;
        positive("dt", dt)?;
        let n = self.dict.n;
        let a = self.coefficients();
        let decay = (-dt / self.tau).exp();
        for i in 0..n {
            let mut inhibition = 0.0;
            for j in 0..n {
                let w = self.lateral[i * n + j];
                if w != 0.0 && a[j] != 0.0 {
                    inhibition += w * a[j];
                }
            }
            let target = b[i] - inhibition;
            self.u[i] = target + (self.u[i] - target) * decay;
        }
        Ok(())
    }

    /// Run `steps` steps of `dt` on signal `y`, from the current state, tracking the objective.
    ///
    /// # Errors
    ///
    /// As [`Lca::step`] and [`Dictionary::project`].
    pub fn run(&mut self, y: &[f64], dt: f64, steps: usize) -> Result<LcaRun, SparseError> {
        let b = self.dict.project(y)?;
        let start = self.objective(y, &self.coefficients())?;
        let mut last = start;
        let mut worst_increase = 0.0f64;
        for _ in 0..steps {
            self.step(dt, &b)?;
            let now = self.objective(y, &self.coefficients())?;
            worst_increase = worst_increase.max(now - last);
            last = now;
        }
        let coefficients = self.coefficients();
        let nonzeros = coefficients.iter().filter(|a| **a != 0.0).count();
        Ok(LcaRun { coefficients, objective: (start, last), worst_increase, nonzeros, steps })
    }

    /// `½ |y − Φ a|² + Σ_i penalty(a_i)`.
    ///
    /// # Errors
    ///
    /// As [`Dictionary::synthesise`], plus [`SparseError::Dimension`] for a `y` of the wrong
    /// length.
    pub fn objective(&self, y: &[f64], a: &[f64]) -> Result<f64, SparseError> {
        len("signal", y.len(), self.dict.m)?;
        let recon = self.dict.synthesise(a)?;
        let fit: f64 = y.iter().zip(&recon).map(|(p, q)| (p - q) * (p - q)).sum::<f64>() / 2.0;
        let pen: f64 = a.iter().map(|&v| self.threshold.penalty(v, self.lambda)).sum();
        Ok(fit + pen)
    }

    /// The LASSO optimality residual at `a`, for the soft threshold: with `c = Φᵀ(y − Φ a)`, a
    /// minimiser has `c_i = λ · sign(a_i)` where `a_i ≠ 0` and `|c_i| ≤ λ` where `a_i = 0`. Returns
    /// the largest violation over `i`; zero at the optimum.
    ///
    /// For the hard threshold the same quantity is reported against the local condition
    /// `c_i = 0` on the support and `|c_i| ≤ λ` off it, which every fixed point of the dynamics
    /// satisfies.
    ///
    /// # Errors
    ///
    /// As [`Lca::objective`].
    pub fn optimality_residual(&self, y: &[f64], a: &[f64]) -> Result<f64, SparseError> {
        len("signal", y.len(), self.dict.m)?;
        let recon = self.dict.synthesise(a)?;
        let resid: Vec<f64> = y.iter().zip(&recon).map(|(p, q)| p - q).collect();
        let c = self.dict.project(&resid)?;
        let mut worst = 0.0f64;
        for (i, &ai) in a.iter().enumerate() {
            let v = if ai != 0.0 {
                match self.threshold {
                    Threshold::Soft => (c[i] - self.lambda * ai.signum()).abs(),
                    Threshold::Hard => c[i].abs(),
                }
            } else {
                (c[i].abs() - self.lambda).max(0.0)
            };
            worst = worst.max(v);
        }
        Ok(worst)
    }
}

// ---------------------------------------------------------------------------------------------
// The spiking LCA
// ---------------------------------------------------------------------------------------------

/// The locally competitive algorithm with spiking neurons: each coefficient is a firing rate,
/// each neuron sees the others through a unit-gain synaptic trace, and the lateral matrix is
/// paid for only when a neuron fires.
#[derive(Debug, Clone, PartialEq)]
pub struct SpikingLca {
    /// The dictionary.
    pub dict: Dictionary,
    /// The threshold `λ`.
    pub lambda: f64,
    /// Neuron time constant, seconds.
    pub tau: f64,
    /// Synaptic trace time constant, seconds. Longer averages more spikes per estimate and
    /// responds more slowly.
    pub tau_syn: f64,
    /// Spikes per second per unit coefficient: the gain that turns a coefficient into a rate.
    ///
    /// This is the precision-versus-energy dial. A neighbour reads a coefficient off a trace whose
    /// ripple is about `1 / (a · rate_scale · τ_syn)` of itself, so a coefficient of `1` read to 5%
    /// through a 20 ms trace needs a thousand spikes a second — and every one of them delivers
    /// `n − 1` lateral operations. Halve the scale and the bill halves and the estimate doubles
    /// its noise; that trade is the whole of what a spiking LCA costs.
    pub rate_scale: f64,
    /// The output nonlinearity.
    pub threshold: Threshold,
    /// Internal states.
    pub u: Vec<f64>,
    /// Sigma-delta accumulators, in `[0, 1)` after every step.
    pub acc: Vec<f64>,
    /// Synaptic traces: each neuron's rate estimate as its neighbours see it.
    pub traces: Vec<f64>,
    /// Spikes emitted per neuron since the last reset.
    pub spikes: Vec<u64>,
    /// Exact counts of what the network did.
    pub ledger: Ledger,
    /// Ticks stepped since the last reset.
    pub ticks: u64,
    lateral: Vec<f64>,
}

impl SpikingLca {
    /// Build over `dict`.
    ///
    /// # Errors
    ///
    /// [`SparseError::OutOfRange`] for a non-positive `lambda`, `tau`, `tau_syn` or `rate_scale`.
    pub fn new(
        dict: Dictionary,
        lambda: f64,
        tau: f64,
        tau_syn: f64,
        rate_scale: f64,
        threshold: Threshold,
    ) -> Result<Self, SparseError> {
        positive("lambda", lambda)?;
        positive("tau", tau)?;
        positive("tau_syn", tau_syn)?;
        positive("rate_scale", rate_scale)?;
        let n = dict.n;
        let mut lateral = dict.gram();
        for i in 0..n {
            lateral[i * n + i] = 0.0;
        }
        Ok(Self {
            dict,
            lambda,
            tau,
            tau_syn,
            rate_scale,
            threshold,
            u: vec![0.0; n],
            acc: vec![0.0; n],
            traces: vec![0.0; n],
            spikes: vec![0; n],
            ledger: Ledger::default(),
            ticks: 0,
            lateral,
        })
    }

    /// Zero the spike counts, the ledger and the tick count, leaving the states, accumulators and
    /// traces where they are — so a measurement can start after the transient.
    pub fn clear_counts(&mut self) {
        self.spikes.iter_mut().for_each(|s| *s = 0);
        self.ledger = Ledger::default();
        self.ticks = 0;
    }

    /// Return every state, accumulator, trace and counter to zero.
    pub fn reset(&mut self) {
        for v in [&mut self.u, &mut self.acc, &mut self.traces] {
            v.iter_mut().for_each(|x| *x = 0.0);
        }
        self.spikes.iter_mut().for_each(|s| *s = 0);
        self.ledger = Ledger::default();
        self.ticks = 0;
    }

    /// One step of `dt` toward the projection `b = Φᵀ y`. Returns how many neurons fired.
    ///
    /// The traces decay, then every neuron integrates `b_i − Σ_j w_ij · trace_j` (exponential
    /// Euler), then its coefficient `T_λ(u_i)` times `rate_scale` is accumulated and a spike is
    /// emitted for each unit crossed, each spike depositing `1/(τ_syn · rate_scale)` into that
    /// neuron's trace — so the trace's mean is the coefficient — and delivering `n − 1` lateral
    /// operations to the ledger.
    ///
    /// # Errors
    ///
    /// [`SparseError::Dimension`] for a `b` of the wrong length, [`SparseError::OutOfRange`] for a
    /// non-positive `dt`.
    pub fn step(&mut self, dt: f64, b: &[f64]) -> Result<usize, SparseError> {
        len("projection", b.len(), self.dict.n)?;
        positive("dt", dt)?;
        let n = self.dict.n;
        let syn_decay = (-dt / self.tau_syn).exp();
        for t in &mut self.traces {
            *t *= syn_decay;
        }
        let decay = (-dt / self.tau).exp();
        let mut fired = 0usize;
        for i in 0..n {
            let mut inhibition = 0.0;
            for j in 0..n {
                let w = self.lateral[i * n + j];
                if w != 0.0 && self.traces[j] != 0.0 {
                    inhibition += w * self.traces[j];
                }
            }
            let target = b[i] - inhibition;
            if target == 0.0 && self.u[i] == 0.0 {
                self.ledger.neuron_updates_idle += 1;
            } else {
                self.ledger.neuron_updates_driven += 1;
            }
            self.u[i] = target + (self.u[i] - target) * decay;
            let a = self.threshold.apply(self.u[i], self.lambda);
            // A negative coefficient is a rate too: the sign rides on the trace, the spike count
            // does not. The accumulator counts magnitude; the trace deposit carries the sign.
            self.acc[i] += a.abs() * self.rate_scale * dt;
            if self.acc[i] >= 1.0 {
                let k = self.acc[i].floor();
                self.acc[i] -= k;
                let k_u = k as u64;
                self.spikes[i] += k_u;
                self.ledger.spikes_out += k_u;
                let delivered = k_u.saturating_mul(n as u64 - 1);
                self.ledger.syn_ops += delivered;
                self.ledger.syn_fetches += delivered;
                self.traces[i] += a.signum() * k / (self.tau_syn * self.rate_scale);
                fired += 1;
            }
        }
        self.ticks += 1;
        Ok(fired)
    }

    /// Run `steps` steps of `dt` on signal `y`.
    ///
    /// # Errors
    ///
    /// As [`SpikingLca::step`] and [`Dictionary::project`].
    pub fn run(&mut self, y: &[f64], dt: f64, steps: usize) -> Result<(), SparseError> {
        let b = self.dict.project(y)?;
        for _ in 0..steps {
            self.step(dt, &b)?;
        }
        Ok(())
    }

    /// The coefficients as the neighbours see them at this instant: the traces, whose mean is each
    /// coefficient and whose ripple is about `1 / (a · rate_scale · τ_syn)` of it.
    #[must_use]
    pub fn rates(&self) -> &[f64] {
        &self.traces
    }

    /// The coefficients as a spike count reads them: `spikes / (ticks · dt) / rate_scale`, unsigned
    /// (a spike carries no sign; the trace does). Exact to one spike in the window. `None` before
    /// any tick.
    #[must_use]
    pub fn counted_magnitudes(&self, dt: f64) -> Option<Vec<f64>> {
        if self.ticks == 0 || !(dt > 0.0) {
            return None;
        }
        let window = self.ticks as f64 * dt;
        Some(self.spikes.iter().map(|&s| s as f64 / window / self.rate_scale).collect())
    }

    /// Lateral operations actually performed per step so far, against the `n(n − 1)` a rate
    /// implementation pays each step. `None` before any step.
    #[must_use]
    pub fn lateral_saving(&self) -> Option<f64> {
        if self.ticks == 0 {
            return None;
        }
        let rate_cost = lateral_ops_per_step(self.dict.n as u64) as f64 * self.ticks as f64;
        Some(1.0 - self.ledger.syn_ops as f64 / rate_cost)
    }
}

#[cfg(test)]
mod tests {
    use super::{Dictionary, Lca, SparseError, SpikingLca, Threshold, Uniform, lateral_ops_per_step, normal};
    use crate::rng::Rng;

    /// Both thresholds are the proximal operators the doc names, checked by grid search over
    /// each one's own objective `½(a − u)² + penalty(a)` — the correspondence is verified, not
    /// quoted.
    #[test]
    fn each_threshold_is_the_proximal_operator_of_its_penalty() {
        let lambda = 0.7;
        for t in [Threshold::Soft, Threshold::Hard] {
            for u in [-3.0, -1.2, -0.71, -0.69, -0.2, 0.0, 0.3, 0.7, 0.75, 2.5] {
                let got = t.apply(u, lambda);
                // Grid search the scalar objective on [-4, 4] at 1e-3, plus the candidates 0 and u.
                let mut best = (f64::INFINITY, f64::NAN);
                let mut consider = |a: f64| {
                    let v = 0.5 * (a - u) * (a - u) + t.penalty(a, lambda);
                    if v < best.0 {
                        best = (v, a);
                    }
                };
                for k in 0..=8000 {
                    consider(-4.0 + k as f64 * 1e-3);
                }
                consider(0.0);
                consider(u);
                consider(got);
                assert!((best.1 - got).abs() < 2e-3, "{t:?} at u = {u}: threshold {got}, argmin {}", best.1);
            }
        }
        assert_eq!(Threshold::Soft.apply(0.7, 0.7), 0.0, "at the threshold exactly, zero");
        assert_eq!(Threshold::Hard.apply(0.7, 0.7), 0.0);
        assert_eq!(Threshold::Hard.apply(0.7000001, 0.7), 0.7000001);
    }

    /// For an orthonormal dictionary the LCA fixed point is the threshold of the projection,
    /// exactly: `a* = T_λ(Φᵀ y)`. Both thresholds, to 1e-10 after twenty time constants.
    #[test]
    fn on_an_orthonormal_dictionary_the_fixed_point_is_the_thresholded_projection() {
        let y = [0.9, -0.05, 0.3, -1.4, 0.0, 0.31, 0.29, 2.0];
        for t in [Threshold::Soft, Threshold::Hard] {
            let mut lca = Lca::new(Dictionary::identity(8).unwrap(), 0.3, 10e-3, t).unwrap();
            // Thirty time constants: the transient of the linear part is e^{-30} ≈ 1e-13.
            let run = lca.run(&y, 1e-4, 3000).unwrap();
            for (i, (&got, &yi)) in run.coefficients.iter().zip(&y).enumerate() {
                let want = t.apply(yi, 0.3);
                assert!((got - want).abs() < 1e-10, "{t:?} atom {i}: {got} vs {want}");
            }
            assert_eq!(run.nonzeros, run.coefficients.iter().filter(|a| **a != 0.0).count());
            assert!(run.worst_increase <= 1e-12, "{t:?}: the objective rose by {}", run.worst_increase);
            assert!(lca.optimality_residual(&y, &run.coefficients).unwrap() < 1e-9);
        }
        // Where the hard and soft answers differ, they differ: a coefficient of 0.9 at λ = 0.3.
        assert!((Threshold::Soft.apply(0.9, 0.3) - 0.6).abs() < 1e-15);
        assert_eq!(Threshold::Hard.apply(0.9, 0.3), 0.9);
    }

    /// On a random overcomplete dictionary the soft-threshold LCA converges to the LASSO optimum:
    /// the optimality residual is below 1e-6 and the objective never rose along the way. A sweep
    /// over λ shows the trade: more penalty, fewer non-zeros, worse fit.
    #[test]
    fn on_an_overcomplete_dictionary_the_lca_reaches_the_lasso_optimum() {
        let mut rng = Rng::new(21);
        let dict = Dictionary::random_unit(16, 32, &mut rng).unwrap();
        assert!(dict.coherence().unwrap() < 0.95, "the atoms are nearly duplicated");
        // A signal made from three atoms plus a little noise.
        let mut truth = vec![0.0; 32];
        truth[3] = 1.0;
        truth[17] = -0.7;
        truth[25] = 0.5;
        let mut y = dict.synthesise(&truth).unwrap();
        for v in &mut y {
            *v += 0.02 * (rng.next_f64() - 0.5);
        }
        let mut last_nonzeros = usize::MAX;
        let mut last_fit = 0.0f64;
        for lambda in [0.02, 0.05, 0.1, 0.2] {
            let mut lca = Lca::new(dict.clone(), lambda, 10e-3, Threshold::Soft).unwrap();
            let run = lca.run(&y, 2e-4, 6000).unwrap();
            let resid = lca.optimality_residual(&y, &run.coefficients).unwrap();
            assert!(resid < 1e-6, "λ {lambda}: optimality residual {resid}");
            assert!(run.worst_increase <= 1e-10, "λ {lambda}: objective rose by {}", run.worst_increase);
            assert!(run.objective.1 < run.objective.0);
            assert!(run.nonzeros <= last_nonzeros, "λ {lambda}: {} non-zeros after {last_nonzeros}", run.nonzeros);
            let recon = dict.synthesise(&run.coefficients).unwrap();
            let fit: f64 = y.iter().zip(&recon).map(|(p, q)| (p - q) * (p - q)).sum();
            assert!(fit >= last_fit - 1e-12, "λ {lambda}: fit improved with more penalty");
            last_nonzeros = run.nonzeros;
            last_fit = fit;
            // At the smallest λ the support includes the three true atoms.
            if lambda == 0.02 {
                for &k in &[3usize, 17, 25] {
                    assert!(run.coefficients[k] != 0.0, "true atom {k} was not selected");
                    assert_eq!(run.coefficients[k].signum(), truth[k].signum());
                }
            }
        }
        assert!(last_nonzeros < 32, "the largest λ left every atom active");
        // And a wrong coefficient vector is not optimal: the residual test can fail.
        let lca = Lca::new(dict.clone(), 0.05, 10e-3, Threshold::Soft).unwrap();
        assert!(lca.optimality_residual(&y, &truth).unwrap() > 1e-3);
        assert!(lca.optimality_residual(&y, &vec![0.0; 32]).unwrap() > 0.1);
    }

    /// The spiking LCA's rates land on the rate LCA's coefficients: exactly known on the identity
    /// dictionary, and within a few percent of the rate solution on a random one. Its ledger
    /// counts `n − 1` lateral operations per spike and nothing on silent ticks, so the lateral
    /// saving is the activity sparsity, measured.
    #[test]
    fn the_spiking_lca_converges_to_the_same_coefficients_and_pays_only_for_spikes() {
        let y = [3.0, -0.2, 1.2, -2.5, 0.0, 0.31, 0.9, 4.0];
        let lambda = 0.3;
        let (dt, tau, tau_syn, scale) = (1e-4, 10e-3, 20e-3, 1000.0);
        let mut sp = SpikingLca::new(Dictionary::identity(8).unwrap(), lambda, tau, tau_syn, scale, Threshold::Soft).unwrap();
        let steps = 20_000; // 2 s: a hundred trace constants
        sp.run(&y, dt, steps).unwrap();
        let rates = sp.rates().to_vec();
        let counted = sp.counted_magnitudes(dt).unwrap();
        for (i, (&r, &yi)) in rates.iter().zip(&y).enumerate() {
            let want = Threshold::Soft.apply(yi, lambda);
            // The trace ripples by about 1/(a·scale·τ_syn) of itself: 8% at the smallest live
            // coefficient here, under 2% at the largest. A silent neuron's trace is exactly zero.
            let tol = if want == 0.0 { 1e-12 } else { 1.5 / (scale * tau_syn) * 1.0 + 0.02 * want.abs() };
            assert!((r - want).abs() <= tol, "atom {i}: trace {r} against {want} (tol {tol})");
            // The spike count is exact to one spike in the window, plus the 10 ms transient.
            let count_tol = 1.0 / (steps as f64 * dt * scale) + 0.02 * want.abs();
            assert!((counted[i] - want.abs()).abs() <= count_tol, "atom {i}: counted {} against {want} (tol {count_tol})", counted[i]);
        }
        assert_eq!(rates[4], 0.0, "a zero coefficient never fires");
        assert_eq!(sp.spikes[4], 0);
        assert!(sp.spikes[0] > 0 && sp.spikes[3] > 0, "negative coefficients spike too: {:?}", sp.spikes);
        assert!(rates[3] < 0.0, "and their trace carries the sign");
        assert_eq!(sp.ledger.spikes_out, sp.spikes.iter().sum::<u64>());
        assert_eq!(sp.ledger.syn_ops, sp.ledger.spikes_out * 7, "n − 1 lateral operations per spike");
        assert_eq!(sp.ledger.syn_fetches, sp.ledger.syn_ops);
        assert_eq!(sp.ledger.neuron_updates(), 8 * steps as u64);
        let saving = sp.lateral_saving().unwrap();
        // About one spike a step across the whole population (Σ|a| · scale · dt ≈ 1) against 56
        // lateral operations a step for the rate form.
        assert!(saving > 0.8, "the spiking form paid {} of the rate form's lateral cost", 1.0 - saving);
        assert_eq!(lateral_ops_per_step(8), 56);

        // A random dictionary: the spiking rates against the rate LCA's coefficients.
        let mut rng = Rng::new(22);
        let dict = Dictionary::random_unit(12, 20, &mut rng).unwrap();
        let mut truth = vec![0.0; 20];
        truth[2] = 3.0;
        truth[9] = -2.0;
        truth[15] = 2.5;
        let y = dict.synthesise(&truth).unwrap();
        let mut rate = Lca::new(dict.clone(), 0.2, tau, Threshold::Soft).unwrap();
        let run = rate.run(&y, dt, 20_000).unwrap();
        assert!(rate.optimality_residual(&y, &run.coefficients).unwrap() < 1e-6);
        let mut sp = SpikingLca::new(dict, 0.2, tau, tau_syn, scale, Threshold::Soft).unwrap();
        // Two seconds to settle — the lateral inhibition has to build up through the traces
        // before a coefficient the optimum leaves at zero stops firing — then the counts are
        // cleared and two more seconds are measured.
        sp.run(&y, dt, 20_000).unwrap();
        let transient_spikes = sp.ledger.spikes_out;
        sp.clear_counts();
        assert_eq!(sp.ticks, 0);
        sp.run(&y, dt, 20_000).unwrap();
        assert!(transient_spikes > 0);
        let mut worst = 0.0f64;
        let counted = sp.counted_magnitudes(dt).unwrap();
        for (i, (&r, &a)) in sp.rates().iter().zip(&run.coefficients).enumerate() {
            // The instantaneous trace ripples by one deposit, 1/(scale·τ_syn) = 0.05, around the
            // coefficient; the two-second spike count is exact to one spike in two thousand.
            let trace_tol = 1.0 / (scale * tau_syn) + 0.05 * a.abs();
            assert!((r - a).abs() <= trace_tol, "atom {i}: spiking trace {r} against rate {a}");
            let count_tol = 0.05 * a.abs() + 0.01;
            assert!((counted[i] - a.abs()).abs() <= count_tol, "atom {i}: counted {} against rate {a}", counted[i]);
            worst = worst.max((counted[i] - a.abs()).abs());
        }
        println!("spiking lca: worst counted-vs-rate error {worst:.4} over 20 atoms");
        assert!(worst > 1e-6, "spiking and rate agree exactly, which a spike train cannot do");
        sp.reset();
        assert_eq!(sp.lateral_saving(), None);
        assert!(sp.spikes.iter().all(|&s| s == 0));
    }

    /// The dictionary's algebra: projection, synthesis, Gram and coherence against hand values.
    #[test]
    fn the_dictionary_algebra_matches_hand_arithmetic() {
        // Two atoms in R²: (1, 0) and (0.6, 0.8).
        let d = Dictionary::new(2, 2, vec![1.0, 0.6, 0.0, 0.8]).unwrap();
        assert_eq!(d.column(1), vec![0.6, 0.8]);
        assert_eq!(d.project(&[1.0, 1.0]).unwrap(), vec![1.0, 1.4]);
        assert_eq!(d.synthesise(&[2.0, 1.0]).unwrap(), vec![2.6, 0.8]);
        let g = d.gram();
        assert_eq!(g, vec![1.0, 0.6, 0.6, 1.0]);
        assert_eq!(d.coherence(), Some(0.6));
        assert_eq!(Dictionary::identity(3).unwrap().coherence(), Some(0.0));
        assert_eq!(Dictionary::new(2, 1, vec![1.0, 0.0]).unwrap().coherence(), None);
        let mut rng = Rng::new(1);
        let r = Dictionary::random_unit(5, 7, &mut rng).unwrap();
        for j in 0..7 {
            let n: f64 = r.column(j).iter().map(|x| x * x).sum();
            assert!((n - 1.0).abs() < 1e-12, "atom {j} has squared norm {n}");
        }
    }

    /// Every refusal names the problem.
    #[test]
    fn the_refusals_name_the_problem() {
        assert!(matches!(Dictionary::new(0, 1, vec![]), Err(SparseError::Empty { what: "signal dimension" })));
        assert!(matches!(Dictionary::new(1, 0, vec![]), Err(SparseError::Empty { what: "atoms" })));
        assert!(matches!(Dictionary::new(2, 2, vec![1.0; 3]), Err(SparseError::Dimension { what: "atoms", got: 3, want: 4 })));
        assert!(matches!(Dictionary::new(1, 1, vec![f64::NAN]), Err(SparseError::NonFinite { what: "atoms", index: 0 })));
        let d = Dictionary::identity(3).unwrap();
        assert!(matches!(d.project(&[1.0]), Err(SparseError::Dimension { what: "signal", .. })));
        assert!(matches!(d.synthesise(&[1.0, f64::INFINITY, 0.0]), Err(SparseError::NonFinite { .. })));
        assert!(matches!(Lca::new(d.clone(), 0.0, 1e-2, Threshold::Soft), Err(SparseError::OutOfRange { what: "lambda", .. })));
        assert!(matches!(Lca::new(d.clone(), 0.1, -1e-2, Threshold::Soft), Err(SparseError::OutOfRange { what: "tau", .. })));
        let mut lca = Lca::new(d.clone(), 0.1, 1e-2, Threshold::Soft).unwrap();
        assert!(matches!(lca.step(0.0, &[0.0; 3]), Err(SparseError::OutOfRange { what: "dt", .. })));
        assert!(matches!(lca.step(1e-3, &[0.0; 2]), Err(SparseError::Dimension { what: "projection", .. })));
        assert!(matches!(SpikingLca::new(d.clone(), 0.1, 1e-2, 0.0, 100.0, Threshold::Soft), Err(SparseError::OutOfRange { what: "tau_syn", .. })));
        assert!(matches!(SpikingLca::new(d, 0.1, 1e-2, 1e-2, 0.0, Threshold::Soft), Err(SparseError::OutOfRange { what: "rate_scale", .. })));
        assert!(lateral_ops_per_step(0) == 0 && lateral_ops_per_step(1) == 0);
        for e in [
            SparseError::Empty { what: "x" },
            SparseError::Dimension { what: "y", got: 1, want: 2 },
            SparseError::NonFinite { what: "z", index: 3 },
            SparseError::OutOfRange { what: "w", value: 9.0, low: 0.0, high: 1.0 },
        ] {
            assert!(!e.to_string().is_empty());
        }
    }


    /// The ledger's two blind spots from the second mutation sweep. A neuron whose drive has gone
    /// but whose membrane still holds charge is DRIVEN — it is decaying, and a chip pays for that
    /// update — and a tick long enough to hold many spikes is charged for every one of them.
    #[test]
    fn a_decaying_membrane_is_not_idle_and_a_burst_is_every_spike() {
        let mut lca = SpikingLca::new(Dictionary::identity(3).unwrap(), 0.5, 10e-3, 10e-3, 1000.0, Threshold::Soft).unwrap();
        lca.step(1e-3, &[0.4, 0.0, 0.0]).unwrap();
        assert_eq!((lca.ledger.neuron_updates_driven, lca.ledger.neuron_updates_idle), (1, 2));
        assert!(lca.u[0] > 0.0 && lca.ledger.spikes_out == 0, "sub-threshold: charged, silent");
        lca.step(1e-3, &[0.0, 0.0, 0.0]).unwrap();
        assert_eq!((lca.ledger.neuron_updates_driven, lca.ledger.neuron_updates_idle), (2, 4));
        // u = 5 (1 − e^{−5}) = 4.9663; T_λ(u) = 4.4663; × 1000 /s × 50 ms = 223.3 → 223 spikes,
        // from ONE neuron in ONE tick, each delivered to the one other neuron.
        let mut burst = SpikingLca::new(Dictionary::identity(2).unwrap(), 0.5, 10e-3, 10e-3, 1000.0, Threshold::Soft).unwrap();
        assert_eq!(burst.step(50e-3, &[5.0, 0.0]).unwrap(), 1, "one neuron fired");
        assert_eq!(burst.spikes, vec![223, 0]);
        assert_eq!((burst.ledger.spikes_out, burst.ledger.syn_ops, burst.ledger.syn_fetches), (223, 223, 223));
    }

    // ------------------------------------------------------------------------------------------
    // The fourth mutation sweep: seventeen edits this module's suite could not see.
    // ------------------------------------------------------------------------------------------

    /// Every guard on the way in refuses what it was written to refuse. Four of them had never
    /// been handed a bad input: [`Dictionary::project`] was only ever given finite signals (the
    /// suite checked [`Dictionary::synthesise`]'s finiteness guard and assumed the twin),
    /// [`Lca::objective`] was only ever called from [`Lca::run`], which supplies a signal of the
    /// right length by construction, and [`SpikingLca::step`] was only ever reached through
    /// [`SpikingLca::run`], which does the same for `b` and passes the same `dt` every tick — so
    /// deleting any of those four checks left the whole suite green. A signal of the wrong length
    /// scores the part that overlaps and reports a number, which is the failure mode worth
    /// refusing: it is a smaller objective, not an error.
    #[test]
    fn a_projection_an_objective_and_a_spiking_step_refuse_the_input_they_cannot_score() {
        let d = Dictionary::identity(3).unwrap();
        assert!(matches!(
            d.project(&[1.0, f64::NAN, 0.0]),
            Err(SparseError::NonFinite { what: "signal", index: 1 })
        ));
        assert!(matches!(
            d.project(&[f64::INFINITY, 0.0, 0.0]),
            Err(SparseError::NonFinite { what: "signal", index: 0 })
        ));
        assert!(matches!(
            d.project(&[0.0, 0.0, f64::NEG_INFINITY]),
            Err(SparseError::NonFinite { what: "signal", index: 2 })
        ));

        let lca = Lca::new(d.clone(), 0.1, 1e-2, Threshold::Soft).unwrap();
        // Short and long alike: a shorter signal is the dangerous one, because `zip` truncates
        // and the objective comes back finite and smaller than the true one.
        assert!(matches!(
            lca.objective(&[1.0, 0.0], &[0.0; 3]),
            Err(SparseError::Dimension { what: "signal", got: 2, want: 3 })
        ));
        assert!(matches!(
            lca.objective(&[1.0, 0.0, 0.0, 0.0], &[0.0; 3]),
            Err(SparseError::Dimension { what: "signal", got: 4, want: 3 })
        ));

        let mut sp = SpikingLca::new(d, 0.1, 1e-2, 2e-2, 100.0, Threshold::Soft).unwrap();
        assert!(matches!(
            sp.step(1e-3, &[0.0; 2]),
            Err(SparseError::Dimension { what: "projection", got: 2, want: 3 })
        ));
        assert!(matches!(
            sp.step(1e-3, &[0.0; 4]),
            Err(SparseError::Dimension { what: "projection", got: 4, want: 3 })
        ));
        for bad in [0.0, -1e-3, f64::NAN, f64::INFINITY] {
            assert!(
                matches!(sp.step(bad, &[0.0; 3]), Err(SparseError::OutOfRange { what: "dt", .. })),
                "the spiking step accepted dt = {bad}"
            );
        }
        // A refused step is not a step: nothing ticked, nothing fired, nothing was charged.
        assert_eq!(sp.ticks, 0);
        assert_eq!(sp.ledger.neuron_updates(), 0);
        assert!(sp.u.iter().all(|&x| x == 0.0));
    }

    /// The spike-count readout refuses a window that is not a positive length of time, `NaN`
    /// included. The suite only ever called [`SpikingLca::counted_magnitudes`] with the same `dt`
    /// the run used, so the guard was never handed anything to refuse — and the comparison that
    /// does the refusing is the negated one, `!(dt > 0.0)`, precisely because the readable-looking
    /// `dt <= 0.0` is false for `NaN` and would hand back a vector of `NaN` counts instead of
    /// `None`. This crate allows `clippy::neg_cmp_op_on_partial_ord` for exactly that reason.
    #[test]
    fn the_counted_magnitudes_refuse_a_window_that_is_not_a_positive_length_of_time() {
        let mut sp =
            SpikingLca::new(Dictionary::identity(2).unwrap(), 0.3, 1e-2, 2e-2, 1000.0, Threshold::Soft).unwrap();
        assert_eq!(sp.counted_magnitudes(1e-4), None, "before any tick there is no window");
        sp.run(&[1.0, 0.0], 1e-4, 200).unwrap();
        let counted = sp.counted_magnitudes(1e-4).expect("after 200 ticks there is a window");
        assert!(counted[0] > 0.0, "the driven neuron fired: {counted:?}");
        for bad in [f64::NAN, 0.0, -1e-4, f64::NEG_INFINITY] {
            assert_eq!(sp.counted_magnitudes(bad), None, "a tick length of {bad} is not a window");
        }
    }

    /// Mutual coherence is a maximum of MAGNITUDES, so a pair of atoms that point away from each
    /// other is exactly as hard as a pair that points together. Every dictionary the suite fed to
    /// [`Dictionary::coherence`] had its largest off-diagonal Gram entry POSITIVE — the hand case
    /// `(1, 0)`, `(0.6, 0.8)`, the identity, and a random one only ever compared against an upper
    /// bound — so dropping the `.abs()` and taking the largest signed inner product changed no
    /// answer the suite read. Here the largest entry is negative, and the signed form reports the
    /// dictionary as orthogonal.
    #[test]
    fn the_mutual_coherence_is_a_magnitude_so_an_anticorrelated_pair_is_as_hard_as_a_parallel_one() {
        // Two unit atoms in R² at 180° − 53°: (1, 0) and (−0.6, −0.8). Their inner product is
        // −0.6 and the coherence is 0.6.
        let d = Dictionary::new(2, 2, vec![1.0, -0.6, 0.0, -0.8]).unwrap();
        assert_eq!(d.gram(), vec![1.0, -0.6, -0.6, 1.0]);
        assert_eq!(d.coherence(), Some(0.6));
        // Three atoms where every off-diagonal entry is negative: a signed maximum reports zero.
        let t = Dictionary::new(2, 3, vec![1.0, -0.5, -0.5, 0.0, 0.75f64.sqrt(), -0.75f64.sqrt()]).unwrap();
        let c = t.coherence().unwrap();
        assert!((c - 0.5).abs() < 1e-15, "the three-atom frame's coherence is 0.5, got {c}");
        let tg = t.gram();
        for i in 0..3 {
            for j in 0..3 {
                if i != j {
                    assert!(tg[i * 3 + j] < 0.0, "off-diagonal {i},{j} is {} and not negative", tg[i * 3 + j]);
                }
            }
        }
        // And on a random dictionary it is the largest |g_ij| off the diagonal, recomputed here.
        let mut rng = Rng::new(404);
        let r = Dictionary::random_unit(6, 14, &mut rng).unwrap();
        let g = r.gram();
        let mut want = 0.0f64;
        let mut signed = 0.0f64;
        for i in 0..14 {
            for j in 0..14 {
                if i != j {
                    want = want.max(g[i * 14 + j].abs());
                    signed = signed.max(g[i * 14 + j]);
                }
            }
        }
        assert_eq!(r.coherence(), Some(want));
        assert!(want > signed, "measured: largest magnitude {want} against largest signed {signed}");
    }

    /// The Box-Muller draw is standard normal — mean 0, variance 1 — pinned on the DRAW rather
    /// than on anything downstream of it. Nothing downstream can see it: [`normal`]'s only caller
    /// is [`Dictionary::random_unit`], which divides each column by its own norm, so scaling every
    /// component of a column by a constant leaves the normalised atom identical to within a bit or
    /// two. Halving the radius' factor of two halves the variance and changes no dictionary this
    /// module can build. The tolerances are the sample size: for `n = 100_000` the standard error
    /// of the mean is `1/sqrt(n)` = 3.2e-3 and of the variance is `sqrt(2/n)` = 4.5e-3, so the
    /// bounds below are 6.3 and 22 standard errors — and a variance of 0.5 misses the second one
    /// five times over.
    #[test]
    fn the_normal_draws_have_unit_variance_before_the_normalisation_hides_it() {
        let mut rng = Rng::new(2718);
        let n = 100_000;
        let (mut s, mut s2) = (0.0f64, 0.0f64);
        let mut extreme = 0usize;
        for _ in 0..n {
            let x = normal(&mut rng);
            assert!(x.is_finite(), "a draw was {x}");
            s += x;
            s2 += x * x;
            extreme += usize::from(x.abs() > 3.0);
        }
        let mean = s / f64::from(n);
        let var = s2 / f64::from(n) - mean * mean;
        assert!(mean.abs() < 0.02, "mean {mean}, bound 0.02 = 6.3 standard errors");
        assert!((var - 1.0).abs() < 0.10, "variance {var}, bound 0.10 = 22 standard errors");
        // A standard normal puts 0.27% of its mass outside ±3; a variance of 1/2 puts 0.003% there,
        // which is about three draws in this sample rather than about 270.
        assert!(extreme > 120, "only {extreme} of {n} draws exceeded 3 sigma");
        // And the normalisation really is what hides it: the atoms come out unit-norm whatever the
        // radius is scaled by.
        let mut rng = Rng::new(2718);
        let dict = Dictionary::random_unit(9, 13, &mut rng).unwrap();
        for j in 0..13 {
            let sq: f64 = dict.column(j).iter().map(|x| x * x).sum();
            assert!((sq - 1.0).abs() < 1e-15, "atom {j} has squared norm {sq}");
        }
    }

    /// A uniform draw of exactly zero does not reach the logarithm as an infinity. `ln(0)` is
    /// `-inf`, the radius is then `inf`, and every atom of the column it lands in comes out `NaN`
    /// once the normalisation divides by an infinite norm — so [`Dictionary::random_unit`] would
    /// return `Err(NonFinite)` rather than a dictionary. No test of the public surface can reach
    /// it: [`Rng::next_f64`] builds a 53-bit fraction and returns exactly zero with probability
    /// `2^-53`, so the branch is unreachable in any run anyone will ever make, and the sweep that
    /// deleted the guard saw a green suite. The [`Uniform`] seam exists to hand it that zero.
    #[test]
    fn a_zero_uniform_draw_does_not_reach_the_logarithm_as_an_infinity() {
        /// A generator that returns the one draw the guard exists for.
        struct AlwaysZero;
        impl Uniform for AlwaysZero {
            fn next_f64(&mut self) -> f64 {
                0.0
            }
        }
        let x = normal(&mut AlwaysZero);
        assert!(x.is_finite(), "a zero uniform produced {x}");
        // The guarded draw is exactly the one the floor defines: sqrt(-2 ln 1e-300) · cos(0).
        assert_eq!(x, (-2.0 * 1e-300f64.ln()).sqrt());
        // The floor is a floor and not a replacement: any draw above it passes through untouched.
        /// A generator that hands back a fixed pair of draws, the first of them ordinary.
        struct Fixed(f64, f64, bool);
        impl Uniform for Fixed {
            fn next_f64(&mut self) -> f64 {
                self.2 = !self.2;
                if self.2 { self.0 } else { self.1 }
            }
        }
        let y = normal(&mut Fixed(0.25, 0.5, false));
        assert_eq!(y, (-2.0 * 0.25f64.ln()).sqrt() * (core::f64::consts::TAU * 0.5).cos());
    }

    /// The soft threshold returns exactly zero at its own threshold, and never manufactures a
    /// `NaN` from operands that are not `NaN`. The boundary comparison is the mechanism: `u > λ`
    /// and `u >= λ` disagree on one input, `u == λ`, and for every FINITE `λ` both branches return
    /// `+0.0` there — IEEE-754 makes `λ − λ` exactly `+0.0` — so the suite could not have seen the
    /// difference on any admissible threshold whatever it tested. Measured, comparing the two
    /// forms directly: 441,066 `(u, λ)` pairs with `λ` finite and positive, drawn over random bit
    /// patterns and over every neighbour of the boundary, 0 differences. What separates them is
    /// `λ = ∞`, where the strict form thresholds an infinite state to zero and the inclusive one
    /// computes `∞ − ∞`. `λ` is a public field on both [`Lca`] and [`SpikingLca`], so a threshold
    /// that no constructor would accept is one assignment away, and a `NaN` coefficient poisons
    /// every state it then touches without ever raising anything.
    #[test]
    fn the_soft_threshold_is_zero_at_its_threshold_and_never_manufactures_a_nan() {
        for lambda in [5e-324, f64::MIN_POSITIVE, 1e-300, 1e-12, 0.3, 0.7, 1.0, 1e12, f64::MAX] {
            assert_eq!(Threshold::Soft.apply(lambda, lambda), 0.0, "at +λ = {lambda}");
            assert_eq!(Threshold::Soft.apply(-lambda, lambda), 0.0, "at −λ = {lambda}");
            assert_eq!(Threshold::Hard.apply(lambda, lambda), 0.0);
        }
        // An infinite threshold silences everything, including an infinite state.
        assert_eq!(Threshold::Soft.apply(f64::INFINITY, f64::INFINITY), 0.0);
        assert_eq!(Threshold::Soft.apply(f64::NEG_INFINITY, f64::INFINITY), 0.0);
        assert_eq!(Threshold::Hard.apply(f64::INFINITY, f64::INFINITY), 0.0);
        for t in [Threshold::Soft, Threshold::Hard] {
            for u in [f64::INFINITY, f64::NEG_INFINITY, f64::MAX, f64::MIN, 0.0, -0.0, 1.0, -1.0, 1e-300] {
                for lambda in [f64::INFINITY, f64::MAX, 1.0, 1e-300, f64::MIN_POSITIVE] {
                    let a = t.apply(u, lambda);
                    assert!(!a.is_nan(), "{t:?} made a NaN out of u = {u}, λ = {lambda}");
                }
            }
        }
    }

    /// [`Lca::reset`] returns the states to zero, which nothing in this suite had ever asked it
    /// to do: every rate-LCA test built a fresh [`Lca`] for each run, so a `reset` that did
    /// nothing at all was invisible. The states are what the coefficients are read off, so a
    /// no-op reset silently continues the previous signal's solution into the next one's run.
    #[test]
    fn the_rate_lcas_reset_returns_every_state_to_zero() {
        let mut lca = Lca::new(Dictionary::identity(3).unwrap(), 0.1, 10e-3, Threshold::Soft).unwrap();
        lca.run(&[1.0, -2.0, 0.5], 1e-3, 100).unwrap();
        assert!(lca.u.iter().any(|&x| x != 0.0), "the run left no charge to clear: {:?}", lca.u);
        assert!(lca.coefficients().iter().any(|&a| a != 0.0));
        lca.reset();
        assert_eq!(lca.u, vec![0.0; 3]);
        assert_eq!(lca.coefficients(), vec![0.0; 3]);
        // And a reset run reproduces a fresh one exactly, which is what the reset is for.
        let mut fresh = Lca::new(Dictionary::identity(3).unwrap(), 0.1, 10e-3, Threshold::Soft).unwrap();
        let y = [0.4, 0.9, -1.1];
        let a = lca.run(&y, 1e-3, 50).unwrap();
        let b = fresh.run(&y, 1e-3, 50).unwrap();
        assert_eq!(a, b);
    }

    /// One step of the rate LCA holds the coefficients fixed across the sweep: neuron `i` is
    /// inhibited by the coefficients as they were at the START of the step, not by whatever the
    /// neurons before it in the loop have just become. Reading them again inside the loop turns a
    /// Jacobi step into a Gauss-Seidel one, which converges to the SAME fixed point — so every
    /// test in this suite, all of which check the fixed point or the objective's monotonicity,
    /// passes either way. What changes is the trajectory, and that is what is pinned here: one
    /// step from a known state against hand arithmetic, in the same operations and the same order,
    /// so the comparison is exact.
    #[test]
    fn one_step_inhibits_with_the_coefficients_the_step_began_with() {
        let (lambda, tau, dt) = (0.1, 10e-3, 1e-3);
        // Two unit atoms in R² with inner product 0.6.
        let d = Dictionary::new(2, 2, vec![1.0, 0.6, 0.0, 0.8]).unwrap();
        let mut lca = Lca::new(d, lambda, tau, Threshold::Soft).unwrap();
        lca.u = vec![1.0, 1.0];
        let b = [1.0, 1.4];
        lca.step(dt, &b).unwrap();

        let decay = (-dt / tau).exp();
        let a0 = Threshold::Soft.apply(1.0, lambda);
        let a1 = Threshold::Soft.apply(1.0, lambda);
        let target0 = b[0] - 0.6 * a1;
        let target1 = b[1] - 0.6 * a0;
        assert_eq!(lca.u[0], target0 + (1.0 - target0) * decay);
        assert_eq!(lca.u[1], target1 + (1.0 - target1) * decay);

        // The Gauss-Seidel step this is not: neuron 1 would have been inhibited by the coefficient
        // neuron 0 acquired half a step ago. Measured, the two sweeps put u_1 at 0.986_677_238_525
        // and 0.989_611_355_635 — a gap of 2.9e-3 after a single step of a tenth of a time
        // constant, which is what makes the exact assertions above load-bearing rather than
        // decorative.
        let swept = Threshold::Soft.apply(lca.u[0], lambda);
        let target1_swept = b[1] - 0.6 * swept;
        let u1_swept = target1_swept + (1.0 - target1_swept) * decay;
        assert!((lca.u[1] - u1_swept).abs() > 2e-3, "the two sweeps differ by {}", (lca.u[1] - u1_swept).abs());
    }

    /// The Gram matrix, and the lateral matrix built from it, are symmetric to the last bit —
    /// not to a tolerance. [`Dictionary::gram`] computes each inner product once and writes that
    /// one `f64` into both `g[i·n + j]` and `g[j·n + i]`, and `Lca::new`/`SpikingLca::new` then
    /// zero the diagonal, which preserves symmetry. This is the invariant that makes reading the
    /// lateral matrix transposed an equivalent edit rather than a defect, so it is asserted here
    /// rather than left as a remark: `lateral` is private, written only from `gram()`, and has no
    /// setter, so an asymmetric one is not a state the public API can produce. If a future
    /// constructor ever admits one, this test fails and that equivalence argument fails with it.
    #[test]
    fn the_gram_matrix_and_the_lateral_matrix_are_symmetric_to_the_last_bit() {
        let mut rng = Rng::new(31);
        for (m, n) in [(2usize, 2usize), (5, 7), (16, 32), (3, 11)] {
            let dict = Dictionary::random_unit(m, n, &mut rng).unwrap();
            let g = dict.gram();
            let lca = Lca::new(dict.clone(), 0.1, 1e-2, Threshold::Soft).unwrap();
            let sp = SpikingLca::new(dict, 0.1, 1e-2, 2e-2, 100.0, Threshold::Soft).unwrap();
            let mut off_diagonal_nonzero = 0usize;
            for i in 0..n {
                for j in 0..n {
                    assert_eq!(g[i * n + j].to_bits(), g[j * n + i].to_bits(), "gram {i},{j} of {m}x{n}");
                    assert_eq!(lca.lateral[i * n + j].to_bits(), lca.lateral[j * n + i].to_bits());
                    assert_eq!(sp.lateral[i * n + j].to_bits(), sp.lateral[j * n + i].to_bits());
                    off_diagonal_nonzero += usize::from(i != j && lca.lateral[i * n + j] != 0.0);
                }
            }
            assert!(off_diagonal_nonzero > 0, "{m}x{n} has no lateral coupling to be symmetric about");
            assert!((0..n).all(|i| lca.lateral[i * n + i] == 0.0), "the diagonal survived");
        }
        // The hand case, where the dictionary itself is not square and could not be symmetric.
        let d = Dictionary::new(2, 3, vec![1.0, 0.0, 0.6, 0.0, 1.0, 0.8]).unwrap();
        assert_eq!(d.gram(), vec![1.0, 0.0, 0.6, 0.0, 1.0, 0.8, 0.6, 0.8, 0.6 * 0.6 + 0.8 * 0.8]);
    }

    /// The worst single-step increase of the objective is floored at zero, so a run that only ever
    /// fell reports no increase rather than reporting how far it fell. Every assertion in this
    /// suite is of the form `worst_increase <= 1e-10`, which a large NEGATIVE number passes just as
    /// happily as zero — so seeding the running maximum at `-inf` instead of `0.0` was invisible,
    /// and `LcaRun::worst_increase` would have carried `-inf` for a run of no steps at all.
    #[test]
    fn a_run_that_only_falls_reports_no_increase_and_a_run_of_no_steps_reports_zero() {
        let mut lca = Lca::new(Dictionary::identity(4).unwrap(), 0.2, 10e-3, Threshold::Soft).unwrap();
        let y = [1.0, -0.8, 0.05, 2.0];
        let idle = lca.run(&y, 1e-3, 0).unwrap();
        assert_eq!(idle.worst_increase, 0.0, "a run of no steps has no increase to report");
        assert_eq!(idle.steps, 0);
        assert_eq!(idle.objective.0, idle.objective.1);
        let run = lca.run(&y, 1e-3, 500).unwrap();
        assert!(run.objective.1 < run.objective.0, "the objective did not fall at all");
        assert!(
            (0.0..=1e-12).contains(&run.worst_increase),
            "a monotone run reported {} rather than a number in [0, 1e-12]",
            run.worst_increase
        );
    }

    /// [`SpikingLca::clear_counts`] leaves the sigma-delta accumulators exactly where they are,
    /// and [`SpikingLca::reset`] discharges the synaptic traces. Both had blind spots. The suite
    /// calls `clear_counts` between a transient and a measurement and then compares spike counts
    /// against a rate solution at 5%, which is far too loose to see a fraction of one spike being
    /// thrown away per neuron; and it calls `reset` and then checks the spike counts and the
    /// ledger, never the traces — so a reset that left every neighbour's rate estimate charged
    /// carried the previous signal's answer into the next run.
    #[test]
    fn clearing_the_counts_keeps_the_accumulators_and_resetting_discharges_the_traces() {
        let mut sp =
            SpikingLca::new(Dictionary::identity(2).unwrap(), 0.3, 10e-3, 20e-3, 1000.0, Threshold::Soft).unwrap();
        sp.run(&[1.0, 0.0], 1e-4, 250).unwrap();
        assert!(sp.acc.iter().any(|&a| a != 0.0), "no sigma-delta residue to preserve: {:?}", sp.acc);
        assert!(sp.traces.iter().any(|&t| t != 0.0), "no trace to discharge: {:?}", sp.traces);
        assert!(sp.ledger.spikes_out > 0);

        let (acc, u, traces) = (sp.acc.clone(), sp.u.clone(), sp.traces.clone());
        sp.clear_counts();
        assert_eq!(sp.acc, acc, "clear_counts spent the fraction of a spike each neuron had banked");
        assert_eq!(sp.u, u);
        assert_eq!(sp.traces, traces);
        assert_eq!(sp.ticks, 0);
        assert_eq!(sp.ledger.spikes_out, 0);
        assert!(sp.spikes.iter().all(|&s| s == 0));

        sp.reset();
        assert_eq!(sp.traces, vec![0.0; 2], "reset left the synaptic traces charged");
        assert_eq!(sp.u, vec![0.0; 2]);
        assert_eq!(sp.acc, vec![0.0; 2]);
        // A reset network reproduces a fresh one tick for tick.
        let mut fresh =
            SpikingLca::new(Dictionary::identity(2).unwrap(), 0.3, 10e-3, 20e-3, 1000.0, Threshold::Soft).unwrap();
        sp.run(&[0.7, 0.4], 1e-4, 300).unwrap();
        fresh.run(&[0.7, 0.4], 1e-4, 300).unwrap();
        assert_eq!(sp.traces, fresh.traces);
        assert_eq!(sp.spikes, fresh.spikes);
    }

    /// The lateral saving is measured against what a RATE implementation would pay, and that is
    /// `n(n − 1)` in the number of ATOMS — the size of the lateral matrix — not `m(m − 1)` in the
    /// signal dimension. Every call to [`SpikingLca::lateral_saving`] in this suite is on the
    /// identity dictionary, where `m = n = 8` and the two are the same number, so sizing the
    /// baseline by the wrong one changed nothing. An overcomplete dictionary is the whole point of
    /// sparse coding, and there `n > m`: here three atoms in two dimensions, where the true
    /// baseline is 6 operations a step and the signal dimension would claim 2.
    #[test]
    fn the_lateral_saving_is_sized_by_the_atom_count_and_not_by_the_signal_dimension() {
        // Three unit atoms in R²: (1, 0), (0, 1) and (0.6, 0.8).
        let dict = Dictionary::new(2, 3, vec![1.0, 0.0, 0.6, 0.0, 1.0, 0.8]).unwrap();
        assert_eq!((dict.m, dict.n), (2, 3));
        let mut sp = SpikingLca::new(dict, 0.2, 10e-3, 20e-3, 500.0, Threshold::Soft).unwrap();
        let ticks = 400;
        sp.run(&[1.0, 0.9], 1e-4, ticks).unwrap();
        assert!(sp.ledger.syn_ops > 0, "nothing fired, so every baseline reports a saving of 1");
        assert_eq!(sp.ticks, ticks as u64);
        // n(n − 1) = 3 · 2 = 6 lateral operations a step, against m(m − 1) = 2 for the dimension.
        assert_eq!(lateral_ops_per_step(3), 6);
        assert_eq!(lateral_ops_per_step(2), 2);
        let want = 1.0 - sp.ledger.syn_ops as f64 / (6.0 * ticks as f64);
        assert_eq!(sp.lateral_saving(), Some(want));
        let by_dimension = 1.0 - sp.ledger.syn_ops as f64 / (2.0 * ticks as f64);
        assert!(
            (want - by_dimension).abs() > 1e-6,
            "measured: atom-count baseline {want}, signal-dimension baseline {by_dimension}"
        );
    }
}
