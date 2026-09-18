//! Reservoir computing: leave the recurrent weights alone and train one linear layer.
//!
//! # The idea
//!
//! A recurrent network is hard to train because the gradient has to be carried backwards through
//! time, and the thing it is carried through is a product of Jacobians that either vanishes or
//! explodes. Reservoir computing refuses the problem. Build a recurrent network at random, **never
//! touch its weights again**, and treat it as a fixed nonlinear filter that maps an input stream
//! into a high-dimensional trajectory. Then fit a single linear map from that trajectory to
//! whatever you wanted. The only thing trained is that last layer, and fitting a linear map is a
//! least-squares problem with a closed-form solution.
//!
//! Two papers invented it independently in the same window. Maass, Natschläger & Markram, *Real-Time
//! Computing Without Stable States: A New Framework for Neural Computation Based on Perturbations of
//! Neural Circuits*, Neural Computation 14:2531–2560, 2002, built the **liquid state machine**: a
//! column of spiking neurons with distance-dependent connectivity, whose transient response to an
//! input is read out by a memoryless classifier. Jaeger, *The "Echo State" Approach to Analysing and
//! Training Recurrent Neural Networks*, GMD Report 148, German National Research Center for
//! Information Technology, 2001, built the **echo state network**: the rate-coded twin, a sparse
//! random matrix whose spectral radius is tuned just below one.
//!
//! # Why this matters on a neuromorphic substrate
//!
//! This is the cheapest thing that works. The expensive part of a spiking network — the recurrent
//! weights — never needs a gradient, never needs to be stored in a differentiable form, and never
//! needs to be updated. It can be fixed silicon, a random resistor crossbar, or an analogue medium
//! nobody designed. The only thing that has to be programmable is the readout, which is `units ×
//! targets` numbers. On a chip where fetching a weight costs more than the multiply that uses it
//! (see [`crate::ledger`]), "the weights never change" is not a convenience — it is the entire
//! energy argument.
//!
//! What it costs: a reservoir needs **many more units than a trained recurrent network** to reach
//! the same accuracy, because it is searching a random feature space rather than learning one. And
//! its memory is finite and measurable — see [`memory_capacity`], which is bounded above by the unit
//! count and in practice comes nowhere near it.
//!
//! # The three properties a reservoir has to have
//!
//! 1. **The echo state property.** The reservoir must forget its initial condition: after a long
//!    enough input, the state must depend on the input and not on where the state started. Without
//!    it the readout is fitting a function of an arbitrary initial condition and will not generalise.
//!    [`echo_state_check`] measures it by running two copies from different starts on the same input
//!    and watching the distance between them.
//!
//!    The usual recipe is "scale the recurrent matrix to a spectral radius below 1". That recipe is
//!    **neither necessary nor sufficient** in general — Yildiz, Jaeger & Kiebel, *Re-visiting the
//!    echo state property*, Neural Networks 35:1–9, 2012, give counterexamples in both directions.
//!    A spectral radius below 1 makes the zero state locally stable under zero input, which is a
//!    weaker statement than the echo state property; the largest singular value below 1 is
//!    sufficient for every input but is far more conservative than anything anyone uses. This module
//!    implements the radius recipe because that is what the field does, says here that it is a
//!    heuristic, and gives you the measurement instead of asking you to trust it.
//!
//! 2. **The separation property** (Maass, §3). Two different input streams must drive the reservoir
//!    to different states. If they do not, no readout can tell them apart, however it is trained.
//!    [`separation`] is the raw quantity from the paper — the Euclidean distance between the two
//!    liquid states — and [`separation_ratio`] is the class-based statistic from Goodman & Ventura,
//!    *Spatiotemporal Pattern Recognition via Liquid State Machines*, IJCNN 2006, which divides the
//!    distance between class centroids by the spread within a class.
//!
//! 3. **The approximation property** (Maass, §3). The readout class must be rich enough to extract
//!    whatever the state holds. With a linear readout this is exactly a least-squares residual, and
//!    [`approximation_residual`] returns it.
//!
//! Maass's theorem is that separation plus approximation gives universal real-time computing power
//! on time-varying inputs. Both halves are needed and both are measurable, which is why they are
//! functions here rather than adjectives.
//!
//! # Units
//!
//! [`Esn`] is **dimensionless**. It is a discrete-time rate model; its state has no volts in it and
//! its "time" is the index of the input sequence. The bridge back to SI is
//! [`EsnSpec::leak_from_tau`], which turns a tick length in seconds and a membrane constant in
//! seconds into the leak rate `a = dt / tau`.
//!
//! [`Liquid`] is in SI throughout, because it is built out of [`crate::neuron::Lif`] and
//! [`crate::net::Net`]: positions in metres on a lattice, delays in seconds converted to ticks at
//! the boundary, weights in volts of membrane displacement per arriving spike.
//!
//! # What is transcribed and what is a convention
//!
//! The liquid's **connection probabilities are the paper's**: `C · exp(-D(a,b)² / λ²)` with
//! `C = 0.3` for excitatory→excitatory, `0.2` for excitatory→inhibitory, `0.4` for
//! inhibitory→excitatory and `0.1` for inhibitory→inhibitory, on a 15×3×3 column with `λ = 2` and
//! 20% inhibitory neurons. So are the transmission delays, 1.5 ms for excitatory→excitatory and
//! 0.8 ms otherwise.
//!
//! The **weights are not the paper's**, and this is stated rather than buried. Maass uses dynamic
//! synapses with gamma-distributed amplitudes in amperes; this crate's synapse is a delta synapse in
//! volts ([`crate::net::Net::w`]). There is no faithful conversion between the two, so
//! [`LiquidSpec`]'s weight defaults are round numbers chosen to put the default column in a regime
//! where it fires without saturating, and they are exposed as fields so a user can replace them.
//! Any figure produced with the defaults is a figure about this implementation, not a reproduction
//! of the 2002 paper.
//!
//! # What is verified here
//!
//! [`Ridge`] solves a system with a known solution to 1e-10, which is also the only test of a linear
//! solver anywhere in this crate. [`power_iteration`] is checked against matrices whose eigenvalues
//! are known by construction — a Householder-conjugated diagonal, a triangular matrix, and a
//! rotation-scaling whose spectrum is a pure complex pair. The echo state property is shown to hold
//! below radius 1 and fail above it **with the same matrix**, scaled two ways. And the delayed-XOR
//! test puts a linear readout on the raw input window beside the reservoir on the same data: the
//! baseline gets strictly more of the relevant input than the reservoir sees at any instant, and
//! still cannot do it, because the obstacle is nonlinearity rather than memory.

use crate::net::{Net, NetBuilder, NetError};
use crate::neuron::Lif;
use crate::rng::Rng;
use crate::sim::{Mode, Sim, SimError};

/// Why a reservoir operation could not be carried out.
///
/// Every variant names the quantity that was wrong and the number that made it wrong. A reservoir
/// that silently accepted a `NaN` would propagate it into a Gram matrix, out of a Cholesky
/// factorisation as a non-finite pivot, and into a readout that predicts `NaN` for every input —
/// with no step in that chain reporting anything.
#[derive(Debug, Clone, PartialEq)]
pub enum ReservoirError {
    /// A supplied number was not finite.
    NonFinite {
        /// Which array it was in, for example `"input"` or `"gram"`.
        what: &'static str,
        /// Index of the offending element within that array.
        index: usize,
    },
    /// A slice had the wrong length for the shape it was used at.
    ShapeMismatch {
        /// Which array, for example `"state"`.
        what: &'static str,
        /// The length supplied.
        got: usize,
        /// The length required.
        want: usize,
    },
    /// A scalar parameter fell outside the range the mechanism is defined on.
    OutOfRange {
        /// Which parameter, for example `"leak"`.
        what: &'static str,
        /// The value supplied.
        value: f64,
        /// Lowest acceptable value, inclusive.
        low: f64,
        /// Highest acceptable value, inclusive.
        high: f64,
    },
    /// An array that must be non-empty was empty.
    Empty {
        /// Which array.
        what: &'static str,
    },
    /// A Cholesky pivot fell to or below the conditioning guard, so the normal equations have no
    /// numerically trustworthy solution.
    ///
    /// The usual cause is a design matrix with linearly dependent columns and no ridge penalty —
    /// two identical reservoir units, or fewer samples than features. Raising
    /// [`Ridge::alpha`] above zero adds to the diagonal and makes the matrix positive definite,
    /// which is what the penalty is for.
    IllConditioned {
        /// Row of the factorisation where the pivot failed, 0-based.
        index: usize,
        /// The pivot value that failed, in the units of the matrix's diagonal.
        pivot: f64,
        /// The largest diagonal entry of the matrix, which the guard is relative to.
        scale: f64,
    },
    /// The recurrent matrix has no measurable spectral radius, so it cannot be scaled to a target
    /// one. A matrix of all zeros is the usual cause: too few units at too low a density.
    Degenerate {
        /// The radius measured, which is at or near zero.
        radius: f64,
    },
    /// A fit was asked for with no samples accumulated.
    NoSamples,
    /// The network underlying a liquid could not be built.
    Net(NetError),
    /// The simulation of a liquid could not be built.
    Sim(SimError),
}

impl core::fmt::Display for ReservoirError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NonFinite { what, index } => {
                write!(f, "{what}[{index}] is not a finite number")
            }
            Self::ShapeMismatch { what, got, want } => {
                write!(f, "{what} has {got} entries where {want} are required")
            }
            Self::OutOfRange { what, value, low, high } => {
                write!(f, "{what} = {value} is outside [{low}, {high}]")
            }
            Self::Empty { what } => write!(f, "{what} is empty"),
            Self::IllConditioned { index, pivot, scale } => write!(
                f,
                "Cholesky pivot {pivot} at row {index} fell to the guard against a diagonal scale \
                 of {scale}; the normal equations are rank deficient, raise the ridge penalty"
            ),
            Self::Degenerate { radius } => {
                write!(f, "the recurrent matrix has spectral radius {radius} and cannot be scaled")
            }
            Self::NoSamples => f.write_str("no samples were accumulated, so there is nothing to fit"),
            Self::Net(e) => write!(f, "liquid connectivity: {e}"),
            Self::Sim(e) => write!(f, "liquid simulation: {e}"),
        }
    }
}

/// As [`crate::net::NetError`]: an error that cannot cross a `Box<dyn Error>` boundary forces every
/// caller to write a conversion, and the ones who do not write it reach for `.unwrap()`.
impl std::error::Error for ReservoirError {}

impl From<NetError> for ReservoirError {
    fn from(e: NetError) -> Self {
        Self::Net(e)
    }
}

impl From<SimError> for ReservoirError {
    fn from(e: SimError) -> Self {
        Self::Sim(e)
    }
}

/// Reject a non-finite entry at the boundary, naming the array and the index.
fn finite(xs: &[f64], what: &'static str) -> Result<(), ReservoirError> {
    for (i, &x) in xs.iter().enumerate() {
        if !x.is_finite() {
            return Err(ReservoirError::NonFinite { what, index: i });
        }
    }
    Ok(())
}

/// Require a length, naming both numbers when it is wrong.
fn shape(got: usize, want: usize, what: &'static str) -> Result<(), ReservoirError> {
    if got == want { Ok(()) } else { Err(ReservoirError::ShapeMismatch { what, got, want }) }
}

/// A uniform draw in `[-1, 1)` from the crate's seeded generator.
fn sym(rng: &mut Rng) -> f64 {
    2.0 * rng.next_f64() - 1.0
}

/// Euclidean norm.
fn norm2(x: &[f64]) -> f64 {
    x.iter().map(|v| v * v).sum::<f64>().sqrt()
}

/// Euclidean distance, used by every separation and echo measurement here.
fn dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(p, q)| (p - q) * (p - q)).sum::<f64>().sqrt()
}

// ---------------------------------------------------------------------------------------------
// Linear algebra. Nothing else in this crate has any, so it is written here and tested against
// matrices whose answers are known before the code runs.
// ---------------------------------------------------------------------------------------------

/// What [`power_iteration`] found.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Spectrum {
    /// Estimate of the spectral radius `|λ_max|`, dimensionless, non-negative.
    ///
    /// Computed as the geometric mean of the per-step growth factors over the last quarter of the
    /// iteration rather than as the last growth factor alone. The difference matters when the
    /// dominant eigenvalue is a **complex conjugate pair**, which is the normal case for a random
    /// matrix: the iterate then rotates inside a two-dimensional invariant subspace and the
    /// single-step growth oscillates around `|λ|` instead of settling on it. A geometric mean over
    /// a window averages the rotation out; a single step does not.
    pub radius: f64,
    /// The Rayleigh quotient `xᵀAx` at the final unit iterate.
    ///
    /// This is the **signed** dominant eigenvalue when that eigenvalue is real, which is the case
    /// for every symmetric matrix. When the dominant eigenvalue is a complex pair it is not an
    /// eigenvalue of anything and should be ignored; [`Spectrum::radius`] is the quantity that
    /// remains meaningful. There is no flag for which case you are in, because deciding that
    /// reliably needs more than one vector.
    pub rayleigh: f64,
    /// Relative change in the radius estimate between the third and fourth quarters of the
    /// iteration, dimensionless. Small means the estimate has settled.
    pub residual: f64,
    /// Iterations actually run, which is `max_iters` unless the iterate collapsed to zero.
    pub iters: usize,
    /// Whether [`Spectrum::residual`] fell below the requested tolerance.
    ///
    /// `false` is not an error and not a wrong answer — power iteration converges at the rate
    /// `|λ₂ / λ₁|`, so a matrix whose two largest eigenvalues are close needs more iterations than
    /// one whose spectrum is well separated. It means "this figure has not stopped moving yet".
    pub converged: bool,
}

/// Estimate the spectral radius of a dense square matrix by power iteration.
///
/// `a` is row-major, `n × n`. The iteration is `x ← A x / ‖A x‖`, started from a seeded random unit
/// vector so that the result is reproducible and so that a caller cannot accidentally start on a
/// vector orthogonal to the dominant eigenvector (which is a measure-zero event for a random start
/// and a certainty for `x = e₁` on some structured matrices).
///
/// This is the operation an echo state network needs and it is the reason it is here: the spectral
/// radius is the one knob the field turns, and computing it needs an eigenvalue routine that a
/// zero-dependency crate has to own.
///
/// # Nilpotent matrices
///
/// If `A x` reaches exactly zero the radius is reported as `0.0` with `converged` true. That is the
/// right answer for a nilpotent matrix — `[[0,1],[0,0]]` has both eigenvalues at zero — and it is
/// reached rather than guessed: the iterate is in the kernel of a power of `A`.
///
/// # Errors
///
/// [`ReservoirError::ShapeMismatch`] if `a.len() != n * n`, [`ReservoirError::Empty`] for `n == 0`,
/// [`ReservoirError::NonFinite`] for a non-finite entry, and [`ReservoirError::OutOfRange`] if
/// `max_iters` is below 8 (the estimator needs four quarters to compare) or `tol` is not a positive
/// finite number.
pub fn power_iteration(
    a: &[f64],
    n: usize,
    max_iters: usize,
    tol: f64,
    seed: u64,
) -> Result<Spectrum, ReservoirError> {
    if n == 0 {
        return Err(ReservoirError::Empty { what: "matrix" });
    }
    shape(a.len(), n * n, "matrix")?;
    finite(a, "matrix")?;
    if max_iters < 8 {
        return Err(ReservoirError::OutOfRange {
            what: "max_iters",
            value: max_iters as f64,
            low: 8.0,
            high: f64::INFINITY,
        });
    }
    if !(tol > 0.0) || !tol.is_finite() {
        return Err(ReservoirError::OutOfRange {
            what: "tol",
            value: tol,
            low: f64::MIN_POSITIVE,
            high: f64::INFINITY,
        });
    }

    let mut rng = Rng::new(seed);
    let mut x = vec![0.0f64; n];
    // A random start, renormalised. The loop guards against the astronomically unlikely all-zero
    // draw rather than assuming it away, because "unlikely" and "impossible" differ by a panic.
    let mut nx = 0.0;
    for _ in 0..16 {
        for v in &mut x {
            *v = sym(&mut rng);
        }
        nx = norm2(&x);
        if nx > 0.0 {
            break;
        }
    }
    if nx == 0.0 {
        x[0] = 1.0;
        nx = 1.0;
    }
    for v in &mut x {
        *v /= nx;
    }

    let mut y = vec![0.0f64; n];
    let mut logs = Vec::with_capacity(max_iters);
    for _ in 0..max_iters {
        for i in 0..n {
            let row = &a[i * n..i * n + n];
            let mut s = 0.0;
            for j in 0..n {
                s += row[j] * x[j];
            }
            y[i] = s;
        }
        let g = norm2(&y);
        if g == 0.0 {
            return Ok(Spectrum {
                radius: 0.0,
                rayleigh: 0.0,
                residual: 0.0,
                iters: logs.len(),
                converged: true,
            });
        }
        logs.push(g.ln());
        for i in 0..n {
            x[i] = y[i] / g;
        }
    }

    // Rayleigh quotient at the final unit iterate: xᵀ A x. `x` is normalised, so the denominator
    // is 1 and is not divided out.
    let mut rayleigh = 0.0;
    for i in 0..n {
        let row = &a[i * n..i * n + n];
        let mut s = 0.0;
        for j in 0..n {
            s += row[j] * x[j];
        }
        rayleigh += x[i] * s;
    }

    let q = max_iters / 4;
    let mean = |slice: &[f64]| slice.iter().sum::<f64>() / slice.len() as f64;
    let late = mean(&logs[max_iters - q..]).exp();
    let early = mean(&logs[max_iters - 2 * q..max_iters - q]).exp();
    let residual = if late > 0.0 { (late - early).abs() / late } else { (late - early).abs() };

    Ok(Spectrum {
        radius: late,
        rayleigh,
        residual,
        iters: max_iters,
        converged: residual < tol,
    })
}

/// A Cholesky factorisation `A = L Lᵀ` of a symmetric positive-definite matrix.
///
/// Only the **lower triangle** of the input is read. Passing a matrix whose two triangles disagree
/// therefore factorises the lower one silently, which is documented here rather than checked
/// because the matrices this crate factorises are Gram matrices built by [`Ridge`] and symmetric by
/// construction.
#[derive(Debug, Clone, PartialEq)]
pub struct Cholesky {
    /// Order of the matrix.
    pub n: usize,
    /// The lower-triangular factor, row-major `n × n`, with exact zeros above the diagonal.
    pub l: Vec<f64>,
    /// Ratio of the largest to the smallest Cholesky pivot `l[j][j]²`, dimensionless, at least 1.
    ///
    /// This is a **lower bound** on the 2-norm condition number of the factorised matrix, not the
    /// condition number: the pivots are the leading entries of successive Schur complements, whose
    /// eigenvalues interlace those of the original matrix, so `max(d) / min(d) ≤ λ_max / λ_min`.
    /// It is reported because it costs nothing and because a readout fitted through a ratio of 1e12
    /// is a readout whose coefficients are noise, whatever its training error says. For an exactly
    /// diagonal matrix the bound is tight and this equals the condition number.
    pub pivot_ratio: f64,
}

/// Factorise `a` (row-major `n × n`, lower triangle read) as `L Lᵀ`.
///
/// `guard` is **relative**: a pivot is rejected when it falls to or below `guard × d_max`, where
/// `d_max` is the largest diagonal entry of `a`. Relative rather than absolute because a Gram
/// matrix accumulated over ten thousand samples has entries ten thousand times larger than one
/// accumulated over one, and an absolute guard would be vacuous for the first and paranoid for the
/// second.
///
/// # Errors
///
/// [`ReservoirError::Empty`] for `n == 0`, [`ReservoirError::ShapeMismatch`] for a wrong length,
/// [`ReservoirError::NonFinite`] for a non-finite entry, [`ReservoirError::OutOfRange`] if `guard`
/// is negative or not finite or if every diagonal entry is at or below zero, and
/// [`ReservoirError::IllConditioned`] naming the row where a pivot failed.
pub fn cholesky(a: &[f64], n: usize, guard: f64) -> Result<Cholesky, ReservoirError> {
    if n == 0 {
        return Err(ReservoirError::Empty { what: "matrix" });
    }
    shape(a.len(), n * n, "matrix")?;
    finite(a, "matrix")?;
    if !(guard >= 0.0) || !guard.is_finite() {
        return Err(ReservoirError::OutOfRange {
            what: "guard",
            value: guard,
            low: 0.0,
            high: f64::INFINITY,
        });
    }
    let mut scale = 0.0f64;
    for j in 0..n {
        scale = scale.max(a[j * n + j]);
    }
    if !(scale > 0.0) {
        return Err(ReservoirError::OutOfRange {
            what: "largest diagonal entry",
            value: scale,
            low: f64::MIN_POSITIVE,
            high: f64::INFINITY,
        });
    }

    let mut l = vec![0.0f64; n * n];
    let mut dmin = f64::INFINITY;
    let mut dmax = 0.0f64;
    for j in 0..n {
        let mut d = a[j * n + j];
        for k in 0..j {
            d -= l[j * n + k] * l[j * n + k];
        }
        if d <= guard * scale {
            return Err(ReservoirError::IllConditioned { index: j, pivot: d, scale });
        }
        dmin = dmin.min(d);
        dmax = dmax.max(d);
        let ljj = d.sqrt();
        l[j * n + j] = ljj;
        for i in j + 1..n {
            let mut t = a[i * n + j];
            for k in 0..j {
                t -= l[i * n + k] * l[j * n + k];
            }
            l[i * n + j] = t / ljj;
        }
    }
    Ok(Cholesky { n, l, pivot_ratio: dmax / dmin })
}

impl Cholesky {
    /// Solve `A z = b` by forward then back substitution.
    ///
    /// # Errors
    ///
    /// [`ReservoirError::ShapeMismatch`] if `b.len() != n`, or [`ReservoirError::NonFinite`] for a
    /// non-finite right-hand side.
    pub fn solve(&self, b: &[f64]) -> Result<Vec<f64>, ReservoirError> {
        shape(b.len(), self.n, "right-hand side")?;
        finite(b, "right-hand side")?;
        let n = self.n;
        let mut z = b.to_vec();
        // L y = b
        for i in 0..n {
            let mut s = z[i];
            for k in 0..i {
                s -= self.l[i * n + k] * z[k];
            }
            z[i] = s / self.l[i * n + i];
        }
        // Lᵀ x = y
        for i in (0..n).rev() {
            let mut s = z[i];
            for k in i + 1..n {
                s -= self.l[k * n + i] * z[k];
            }
            z[i] = s / self.l[i * n + i];
        }
        Ok(z)
    }
}

/// Ridge regression by accumulated normal equations.
///
/// `(XᵀX + αI) W = XᵀY`, solved by [`cholesky`]. The design matrix `X` is **never held**: samples
/// are folded into the `p × p` Gram matrix and the `p × q` cross-moment as they arrive, so training
/// a readout on a million reservoir states costs `p²` memory rather than `p` times a million. That
/// is the property that makes this usable on a device, and it is why the accumulate/solve split is
/// the shape of the API rather than a single `fit(x, y)` call — which is also provided, as
/// [`Ridge::fit`], for the case where the data already fits in memory.
///
/// The penalty `α` is the standard Tikhonov regularisation: it trades bias for variance and, more
/// importantly here, it makes the normal matrix positive definite when the reservoir has linearly
/// dependent units. A reservoir **always** has nearly dependent units — that is what a random
/// recurrent matrix produces — so `α = 0` is a test setting rather than a working one.
#[derive(Debug, Clone, PartialEq)]
pub struct Ridge {
    /// Features per sample, **excluding** the bias column.
    pub features: usize,
    /// Targets per sample, at least 1.
    pub targets: usize,
    /// Tikhonov penalty, non-negative, in the units of the Gram matrix's diagonal.
    pub alpha: f64,
    /// Whether a constant `1.0` feature is appended to every sample.
    ///
    /// The bias column is **not** penalised: `α` is added to the first [`Ridge::features`] diagonal
    /// entries and not to the last. Shrinking an intercept toward zero would bias every prediction
    /// toward zero by an amount that depends on the penalty, which is not what the penalty is for.
    pub bias: bool,
    /// Samples folded in so far.
    pub samples: usize,
    /// The Gram matrix `XᵀX`, row-major `p × p` with `p = features + bias as usize`. Symmetric by
    /// construction; both triangles are filled.
    pub gram: Vec<f64>,
    /// The cross-moment `XᵀY`, row-major `p × targets`.
    pub cross: Vec<f64>,
}

impl Ridge {
    /// The conditioning guard used by [`Ridge::solve`], relative to the largest diagonal entry of
    /// the penalised Gram matrix.
    ///
    /// `1e-12` is roughly `f64` epsilon times `4500`: a pivot that has lost all but the last twelve
    /// digits of the matrix's scale carries no information the solve could use. It is a constant
    /// rather than a parameter because a caller who wants a looser guard wants a larger `α`, which
    /// is the knob that actually fixes the problem.
    pub const GUARD: f64 = 1e-12;

    /// An empty accumulator.
    ///
    /// # Errors
    ///
    /// [`ReservoirError::Empty`] if `features` or `targets` is zero, or
    /// [`ReservoirError::OutOfRange`] if `alpha` is negative or not finite.
    pub fn new(
        features: usize,
        targets: usize,
        alpha: f64,
        bias: bool,
    ) -> Result<Self, ReservoirError> {
        if features == 0 {
            return Err(ReservoirError::Empty { what: "features" });
        }
        if targets == 0 {
            return Err(ReservoirError::Empty { what: "targets" });
        }
        if !(alpha >= 0.0) || !alpha.is_finite() {
            return Err(ReservoirError::OutOfRange {
                what: "alpha",
                value: alpha,
                low: 0.0,
                high: f64::INFINITY,
            });
        }
        let p = features + usize::from(bias);
        Ok(Self {
            features,
            targets,
            alpha,
            bias,
            samples: 0,
            gram: vec![0.0; p * p],
            cross: vec![0.0; p * targets],
        })
    }

    /// Width of the augmented design, `features + 1` when there is a bias and `features` otherwise.
    #[must_use]
    pub fn width(&self) -> usize {
        self.features + usize::from(self.bias)
    }

    /// Fold one sample into the normal equations.
    ///
    /// # Errors
    ///
    /// [`ReservoirError::ShapeMismatch`] if `x` is not [`Ridge::features`] long or `y` is not
    /// [`Ridge::targets`] long, and [`ReservoirError::NonFinite`] for a non-finite entry in either.
    pub fn accumulate(&mut self, x: &[f64], y: &[f64]) -> Result<(), ReservoirError> {
        shape(x.len(), self.features, "sample")?;
        shape(y.len(), self.targets, "target")?;
        finite(x, "sample")?;
        finite(y, "target")?;
        let p = self.width();
        let at = |i: usize| if i < self.features { x[i] } else { 1.0 };
        for i in 0..p {
            let xi = at(i);
            for j in 0..p {
                self.gram[i * p + j] += xi * at(j);
            }
            for t in 0..self.targets {
                self.cross[i * self.targets + t] += xi * y[t];
            }
        }
        self.samples += 1;
        Ok(())
    }

    /// Solve the accumulated normal equations.
    ///
    /// # Errors
    ///
    /// [`ReservoirError::NoSamples`] if nothing was accumulated, and
    /// [`ReservoirError::IllConditioned`] if the penalised Gram matrix is numerically rank
    /// deficient — which with `alpha = 0` is exactly the case of linearly dependent features or
    /// fewer samples than features, and with `alpha > 0` should not happen at all.
    pub fn solve(&self) -> Result<Readout, ReservoirError> {
        if self.samples == 0 {
            return Err(ReservoirError::NoSamples);
        }
        let p = self.width();
        let mut m = self.gram.clone();
        // The penalty goes on the FEATURE diagonal only. See the note on `Ridge::bias`.
        for i in 0..self.features {
            m[i * p + i] += self.alpha;
        }
        let chol = cholesky(&m, p, Self::GUARD)?;
        let mut w = vec![0.0f64; self.targets * p];
        let mut rhs = vec![0.0f64; p];
        for t in 0..self.targets {
            for i in 0..p {
                rhs[i] = self.cross[i * self.targets + t];
            }
            let sol = chol.solve(&rhs)?;
            for i in 0..p {
                w[t * p + i] = sol[i];
            }
        }
        Ok(Readout {
            features: self.features,
            targets: self.targets,
            bias: self.bias,
            w,
            pivot_ratio: chol.pivot_ratio,
            samples: self.samples,
        })
    }

    /// Accumulate every sample in `x` against `y` and solve, for data that already fits in memory.
    ///
    /// # Errors
    ///
    /// As [`Ridge::new`], [`Ridge::accumulate`] and [`Ridge::solve`], plus
    /// [`ReservoirError::ShapeMismatch`] if `x` and `y` have different lengths.
    pub fn fit(
        x: &[Vec<f64>],
        y: &[Vec<f64>],
        alpha: f64,
        bias: bool,
    ) -> Result<Readout, ReservoirError> {
        if x.is_empty() {
            return Err(ReservoirError::Empty { what: "design" });
        }
        shape(y.len(), x.len(), "targets")?;
        let mut r = Self::new(x[0].len(), y[0].len(), alpha, bias)?;
        for (xi, yi) in x.iter().zip(y) {
            r.accumulate(xi, yi)?;
        }
        r.solve()
    }
}

/// A fitted linear readout: the only trained parameters in a reservoir computer.
#[derive(Debug, Clone, PartialEq)]
pub struct Readout {
    /// Features per sample, excluding the bias column.
    pub features: usize,
    /// Targets per sample.
    pub targets: usize,
    /// Whether a constant `1.0` was appended during the fit; [`Readout::predict`] appends it again.
    pub bias: bool,
    /// Coefficients, row-major `targets × (features + bias as usize)`.
    pub w: Vec<f64>,
    /// The [`Cholesky::pivot_ratio`] of the fit, carried forward so a caller can see how far the
    /// solve was from rank deficiency without refitting.
    pub pivot_ratio: f64,
    /// Samples the fit was computed from.
    pub samples: usize,
}

impl Readout {
    /// Predict the targets for one feature vector.
    ///
    /// # Errors
    ///
    /// [`ReservoirError::ShapeMismatch`] for the wrong feature count, [`ReservoirError::NonFinite`]
    /// for a non-finite feature.
    pub fn predict(&self, x: &[f64]) -> Result<Vec<f64>, ReservoirError> {
        shape(x.len(), self.features, "sample")?;
        finite(x, "sample")?;
        let p = self.features + usize::from(self.bias);
        let mut out = vec![0.0f64; self.targets];
        for t in 0..self.targets {
            let row = &self.w[t * p..t * p + p];
            let mut s = 0.0;
            for i in 0..self.features {
                s += row[i] * x[i];
            }
            if self.bias {
                s += row[self.features];
            }
            out[t] = s;
        }
        Ok(out)
    }

    /// Root-mean-square error over a dataset, averaged across samples **and** targets.
    ///
    /// # Errors
    ///
    /// [`ReservoirError::Empty`] for no samples, [`ReservoirError::ShapeMismatch`] if the lengths
    /// disagree, and whatever [`Readout::predict`] returns.
    pub fn rmse(&self, x: &[Vec<f64>], y: &[Vec<f64>]) -> Result<f64, ReservoirError> {
        if x.is_empty() {
            return Err(ReservoirError::Empty { what: "design" });
        }
        shape(y.len(), x.len(), "targets")?;
        let mut acc = 0.0;
        let mut n = 0usize;
        for (xi, yi) in x.iter().zip(y) {
            shape(yi.len(), self.targets, "target")?;
            let p = self.predict(xi)?;
            for t in 0..self.targets {
                let e = p[t] - yi[t];
                acc += e * e;
                n += 1;
            }
        }
        Ok((acc / n as f64).sqrt())
    }
}

// ---------------------------------------------------------------------------------------------
// The echo state network: Jaeger's rate-coded reservoir.
// ---------------------------------------------------------------------------------------------

/// How to build an [`Esn`].
///
/// Every field is a knob the echo state literature actually turns. The defaults are the values a
/// first experiment usually starts from — 100 units, radius 0.9, 10% density — and are round
/// numbers rather than a fit to any task, stated here so that a figure produced with the defaults
/// is reproducible from the documentation alone.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EsnSpec {
    /// Reservoir units, at least 1. This is the readout's feature count and therefore the thing
    /// that costs parameters; it is also the ceiling on [`memory_capacity`].
    pub units: usize,
    /// Target spectral radius of the recurrent matrix after scaling, dimensionless, positive.
    ///
    /// Below 1 is the usual echo-state regime. See the module doc for why that is a heuristic and
    /// not a theorem.
    pub spectral_radius: f64,
    /// Fraction of recurrent entries that are non-zero, in `(0, 1]`.
    ///
    /// Sparsity is not an accuracy trick — Jaeger's own note is that it mainly buys speed and a
    /// slightly richer decoupling of unit dynamics. 0.1 is the conventional starting point.
    pub density: f64,
    /// Half-width of the uniform distribution the input weights are drawn from, dimensionless.
    ///
    /// This is the single most task-sensitive parameter: large values push `tanh` into saturation
    /// and make the reservoir a nonlinear switch, small values keep it near-linear and make it a
    /// memory. There is no default that is right for two tasks.
    pub input_scaling: f64,
    /// Half-width of the uniform distribution the per-unit biases are drawn from, dimensionless.
    /// Zero gives an unbiased reservoir whose zero state is a fixed point under zero input.
    pub bias_scaling: f64,
    /// Leak rate `a` in `(0, 1]`: the state update is `x ← (1-a)·x + a·tanh(...)`.
    ///
    /// `a = 1` is the classic non-leaky echo state network. Below 1 the reservoir low-pass filters
    /// its own state, which is how a discrete-time reservoir is matched to an input slower than its
    /// sample rate. [`EsnSpec::leak_from_tau`] computes it from SI time constants.
    pub leak: f64,
    /// Seed for the recurrent, input and bias draws. Same seed, same reservoir, every platform.
    pub seed: u64,
}

impl Default for EsnSpec {
    fn default() -> Self {
        Self {
            units: 100,
            spectral_radius: 0.9,
            density: 0.1,
            input_scaling: 1.0,
            bias_scaling: 0.0,
            leak: 1.0,
            seed: 0x05EE_D0E5,
        }
    }
}

impl EsnSpec {
    /// The leak rate corresponding to a tick of `dt` seconds and a state time constant of `tau`
    /// seconds: `a = dt / tau`.
    ///
    /// This is the one place the dimensionless reservoir touches SI, and it is a function rather
    /// than a field so that the conversion is visible at the call site. A leak above 1 is an
    /// unstable forward-Euler step of the continuous leaky equation, not a fast reservoir, so it is
    /// refused rather than clamped.
    ///
    /// # Errors
    ///
    /// [`ReservoirError::OutOfRange`] if `dt` or `tau` is not positive and finite, or if the ratio
    /// leaves `(0, 1]`.
    pub fn leak_from_tau(dt: f64, tau: f64) -> Result<f64, ReservoirError> {
        for (what, v) in [("dt", dt), ("tau", tau)] {
            if !(v > 0.0) || !v.is_finite() {
                return Err(ReservoirError::OutOfRange {
                    what,
                    value: v,
                    low: f64::MIN_POSITIVE,
                    high: f64::INFINITY,
                });
            }
        }
        let a = dt / tau;
        if a > 1.0 {
            return Err(ReservoirError::OutOfRange {
                what: "dt / tau",
                value: a,
                low: f64::MIN_POSITIVE,
                high: 1.0,
            });
        }
        Ok(a)
    }
}

/// A discrete-time leaky-integrator echo state network.
///
/// `x ← (1 - a)·x + a·tanh(W x + W_in u + b)`, with `W` scaled to a chosen spectral radius. Nothing
/// in `W`, `W_in` or `b` is ever trained; see [`Ridge`] for the part that is.
///
/// # The state is bounded, always
///
/// `tanh` maps into `(-1, 1)` and the update is a convex combination of the old state and that, so
/// **if every component of the state starts in `[-1, 1]` it stays in `[-1, 1]` forever**, for any
/// weights, any input and any number of steps. This is an algebraic invariant rather than an
/// empirical observation, it is the reason an echo state network cannot produce a `NaN` from a
/// finite input, and it is asserted in the tests against a reservoir with spectral radius 50 driven
/// by inputs of magnitude 1e6.
#[derive(Debug, Clone, PartialEq)]
pub struct Esn {
    /// Reservoir units.
    pub units: usize,
    /// Input channels.
    pub n_in: usize,
    /// Recurrent weights, row-major `units × units`. Row `i` holds the weights **into** unit `i`.
    pub w: Vec<f64>,
    /// Input weights, row-major `units × n_in`.
    pub w_in: Vec<f64>,
    /// Per-unit bias, `units` long, dimensionless.
    pub bias: Vec<f64>,
    /// Leak rate `a` in `(0, 1]`, as [`EsnSpec::leak`].
    pub leak: f64,
    /// Current state, `units` long, every component in `[-1, 1]`.
    pub x: Vec<f64>,
    /// The spectral radius measured after scaling, dimensionless.
    ///
    /// Not identical to [`EsnSpec::spectral_radius`]: the matrix was scaled by the ratio of the
    /// target to a *measured* radius, so this is the target up to the power iteration's own
    /// residual. It is stored rather than recomputed because a reader comparing two reservoirs
    /// needs the number that was actually realised, not the one that was asked for.
    pub radius: f64,
}

impl Esn {
    /// Build a reservoir from a spec and an input width.
    ///
    /// The recurrent matrix is drawn entry by entry: each of the `units²` positions is non-zero with
    /// probability [`EsnSpec::density`] and then uniform in `[-1, 1)`. It is then scaled so its
    /// spectral radius is the requested one. The diagonal is **not** excluded — a self-connection
    /// is a legitimate recurrent weight and excluding it would change the spectrum in a way nobody
    /// asked for.
    ///
    /// # Errors
    ///
    /// [`ReservoirError::Empty`] for zero units or zero input channels,
    /// [`ReservoirError::OutOfRange`] for a density outside `(0, 1]`, a leak outside `(0, 1]`, a
    /// non-positive or non-finite spectral radius, or a non-finite scaling, and
    /// [`ReservoirError::Degenerate`] if the drawn matrix has no measurable radius to scale — which
    /// happens when `units² × density` is small enough that every entry came out zero.
    pub fn new(spec: &EsnSpec, n_in: usize) -> Result<Self, ReservoirError> {
        if spec.units == 0 {
            return Err(ReservoirError::Empty { what: "units" });
        }
        if n_in == 0 {
            return Err(ReservoirError::Empty { what: "input channels" });
        }
        if !(spec.density > 0.0) || spec.density > 1.0 {
            return Err(ReservoirError::OutOfRange {
                what: "density",
                value: spec.density,
                low: f64::MIN_POSITIVE,
                high: 1.0,
            });
        }
        if !(spec.leak > 0.0) || spec.leak > 1.0 {
            return Err(ReservoirError::OutOfRange {
                what: "leak",
                value: spec.leak,
                low: f64::MIN_POSITIVE,
                high: 1.0,
            });
        }
        if !(spec.spectral_radius > 0.0) || !spec.spectral_radius.is_finite() {
            return Err(ReservoirError::OutOfRange {
                what: "spectral_radius",
                value: spec.spectral_radius,
                low: f64::MIN_POSITIVE,
                high: f64::INFINITY,
            });
        }
        for (what, v) in [("input_scaling", spec.input_scaling), ("bias_scaling", spec.bias_scaling)]
        {
            if !v.is_finite() {
                return Err(ReservoirError::NonFinite { what, index: 0 });
            }
        }

        let n = spec.units;
        let mut rng = Rng::new(spec.seed);
        let mut w = vec![0.0f64; n * n];
        for v in &mut w {
            // Two draws per entry, always, whether or not the mask keeps it. Drawing only on a hit
            // would make the stream depend on the density, so two reservoirs that differ only in
            // density would share no structure at all and could not be compared.
            let keep = rng.next_f64() < spec.density;
            let value = sym(&mut rng);
            if keep {
                *v = value;
            }
        }
        let mut w_in = vec![0.0f64; n * n_in];
        for v in &mut w_in {
            *v = spec.input_scaling * sym(&mut rng);
        }
        let mut bias = vec![0.0f64; n];
        for v in &mut bias {
            *v = spec.bias_scaling * sym(&mut rng);
        }

        let mut esn = Self {
            units: n,
            n_in,
            w,
            w_in,
            bias,
            leak: spec.leak,
            x: vec![0.0; n],
            radius: 0.0,
        };
        esn.radius = esn.rescale(spec.spectral_radius)?;
        Ok(esn)
    }

    /// Rescale the recurrent matrix to a new spectral radius, returning the radius measured after
    /// scaling.
    ///
    /// This exists so that two reservoirs can differ **only** in their spectral radius — same
    /// sparsity pattern, same signs, same relative magnitudes — which is what the echo-state-property
    /// test needs to make its contrast a controlled one rather than two unrelated draws.
    ///
    /// # Errors
    ///
    /// [`ReservoirError::OutOfRange`] for a non-positive or non-finite target, and
    /// [`ReservoirError::Degenerate`] if the current matrix has no measurable radius.
    pub fn rescale(&mut self, target: f64) -> Result<f64, ReservoirError> {
        if !(target > 0.0) || !target.is_finite() {
            return Err(ReservoirError::OutOfRange {
                what: "target radius",
                value: target,
                low: f64::MIN_POSITIVE,
                high: f64::INFINITY,
            });
        }
        let n = self.units;
        // 400 iterations at 1e-10: enough that a random sparse matrix, whose two largest eigenvalues
        // are typically well separated, settles to the last few digits, and cheap enough that
        // building a 1000-unit reservoir is still a fraction of a second.
        let s = power_iteration(&self.w, n, 400, 1e-10, 0x9E37_79B9)?;
        if !(s.radius > 0.0) {
            return Err(ReservoirError::Degenerate { radius: s.radius });
        }
        let k = target / s.radius;
        for v in &mut self.w {
            *v *= k;
        }
        let after = power_iteration(&self.w, n, 400, 1e-10, 0x9E37_79B9)?;
        Ok(after.radius)
    }

    /// Advance one step under input `u` and return the new state.
    ///
    /// # Errors
    ///
    /// [`ReservoirError::ShapeMismatch`] if `u` is not [`Esn::n_in`] long, or
    /// [`ReservoirError::NonFinite`] for a non-finite input.
    pub fn step(&mut self, u: &[f64]) -> Result<&[f64], ReservoirError> {
        shape(u.len(), self.n_in, "input")?;
        finite(u, "input")?;
        let n = self.units;
        let mut next = vec![0.0f64; n];
        for i in 0..n {
            let row = &self.w[i * n..i * n + n];
            let mut s = self.bias[i];
            for j in 0..n {
                s += row[j] * self.x[j];
            }
            let rin = &self.w_in[i * self.n_in..i * self.n_in + self.n_in];
            for j in 0..self.n_in {
                s += rin[j] * u[j];
            }
            next[i] = (1.0 - self.leak) * self.x[i] + self.leak * s.tanh();
        }
        self.x = next;
        Ok(&self.x)
    }

    /// Run over a sequence and return the state after each step, discarding the first `washout`.
    ///
    /// The washout is the price of the echo state property: the first states still remember where
    /// the run started, so training on them fits a function of an initial condition that will never
    /// recur. Discarding them is not optional and is not free — it is why a reservoir needs a
    /// warm-up sequence before every evaluation as well as before training.
    ///
    /// # Errors
    ///
    /// [`ReservoirError::Empty`] if `inputs` is empty or `washout` is at or past its length, plus
    /// whatever [`Esn::step`] returns.
    pub fn collect(
        &mut self,
        inputs: &[Vec<f64>],
        washout: usize,
    ) -> Result<Vec<Vec<f64>>, ReservoirError> {
        if inputs.is_empty() {
            return Err(ReservoirError::Empty { what: "inputs" });
        }
        if washout >= inputs.len() {
            return Err(ReservoirError::Empty { what: "inputs after washout" });
        }
        let mut out = Vec::with_capacity(inputs.len() - washout);
        for (k, u) in inputs.iter().enumerate() {
            self.step(u)?;
            if k >= washout {
                out.push(self.x.clone());
            }
        }
        Ok(out)
    }

    /// The current state.
    #[must_use]
    pub fn state(&self) -> &[f64] {
        &self.x
    }

    /// Set the state, which is how two copies are started apart for an echo-state check.
    ///
    /// # Errors
    ///
    /// [`ReservoirError::ShapeMismatch`] for the wrong length, [`ReservoirError::NonFinite`] for a
    /// non-finite component, and [`ReservoirError::OutOfRange`] for a component outside `[-1, 1]` —
    /// which is refused because the boundedness invariant in the type doc holds only for states
    /// that start inside the box.
    pub fn set_state(&mut self, x: &[f64]) -> Result<(), ReservoirError> {
        shape(x.len(), self.units, "state")?;
        finite(x, "state")?;
        for &v in x {
            if !(-1.0..=1.0).contains(&v) {
                return Err(ReservoirError::OutOfRange {
                    what: "state component",
                    value: v,
                    low: -1.0,
                    high: 1.0,
                });
            }
        }
        self.x.copy_from_slice(x);
        Ok(())
    }

    /// Return the state to all zeros.
    pub fn reset(&mut self) {
        self.x.iter_mut().for_each(|v| *v = 0.0);
    }
}

// ---------------------------------------------------------------------------------------------
// The three properties, as numbers.
// ---------------------------------------------------------------------------------------------

/// What [`echo_state_check`] observed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EchoReport {
    /// `‖x₀ − y₀‖`, the distance between the two initial states.
    pub start: f64,
    /// The distance after the last input.
    pub end: f64,
    /// The largest distance seen at **any** step.
    ///
    /// This can exceed [`EchoReport::start`] even for a reservoir that forgets perfectly, because a
    /// non-normal matrix expands some directions transiently before the asymptotic contraction
    /// takes over. A reservoir whose peak is a hundred times its start is one whose readout sees
    /// enormous excursions during the washout, which is a real effect and not a bug in the check.
    pub peak: f64,
    /// `end / start`, dimensionless. Below 1 means the two states moved together.
    pub ratio: f64,
    /// Whether `end` fell at or below the tolerance the caller asked for.
    pub forgets: bool,
    /// Steps run, which is `inputs.len()`.
    pub steps: usize,
}

/// Measure the echo state property: run two copies of a reservoir from different initial states on
/// the **same** input and watch the distance between them.
///
/// This is the definition made operational. The echo state property says the state is a function of
/// the input history alone; if it holds, two copies that differ only in where they started must
/// converge, and if it fails they will not. Nothing about the spectral radius enters here — the
/// measurement is independent of the heuristic used to aim for it, which is the point.
///
/// # Errors
///
/// [`ReservoirError::Empty`] if `inputs` is empty, [`ReservoirError::OutOfRange`] if `tol` is
/// negative or not finite, and whatever [`Esn::set_state`] and [`Esn::step`] return.
pub fn echo_state_check(
    esn: &Esn,
    inputs: &[Vec<f64>],
    x0: &[f64],
    y0: &[f64],
    tol: f64,
) -> Result<EchoReport, ReservoirError> {
    if inputs.is_empty() {
        return Err(ReservoirError::Empty { what: "inputs" });
    }
    if !(tol >= 0.0) || !tol.is_finite() {
        return Err(ReservoirError::OutOfRange {
            what: "tol",
            value: tol,
            low: 0.0,
            high: f64::INFINITY,
        });
    }
    let mut a = esn.clone();
    let mut b = esn.clone();
    a.set_state(x0)?;
    b.set_state(y0)?;
    let start = dist(a.state(), b.state());
    let mut peak = start;
    for u in inputs {
        a.step(u)?;
        b.step(u)?;
        peak = peak.max(dist(a.state(), b.state()));
    }
    let end = dist(a.state(), b.state());
    Ok(EchoReport {
        start,
        end,
        peak,
        ratio: if start > 0.0 { end / start } else { 0.0 },
        forgets: end <= tol,
        steps: inputs.len(),
    })
}

/// Maass's separation: the Euclidean distance between the reservoir states two input streams drove
/// the reservoir to.
///
/// From Maass, Natschläger & Markram, Neural Computation 14:2531–2560, 2002, §3: a liquid has the
/// separation property for a class of input streams if different streams give different states. The
/// quantity is the raw distance, and it is **not normalised by dimension**. A reservoir with four
/// times the units separates roughly twice as much on the same task simply by having more
/// coordinates to differ in — which is not an artefact but the trade the paper describes, since
/// those extra coordinates are also extra readout parameters to fit. If you want a
/// dimension-insensitive figure, use [`separation_ratio`], which divides by the within-class spread.
///
/// # Errors
///
/// [`ReservoirError::ShapeMismatch`] if the two states have different lengths,
/// [`ReservoirError::Empty`] for empty states, [`ReservoirError::NonFinite`] for a non-finite
/// component.
pub fn separation(a: &[f64], b: &[f64]) -> Result<f64, ReservoirError> {
    if a.is_empty() {
        return Err(ReservoirError::Empty { what: "state" });
    }
    shape(b.len(), a.len(), "second state")?;
    finite(a, "state")?;
    finite(b, "second state")?;
    Ok(dist(a, b))
}

/// The class-based separation statistic and its two halves.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SeparationReport {
    /// `Cd`: mean Euclidean distance between class centroids, averaged over **all ordered pairs of
    /// classes including the diagonal**, which is the averaging the source paper uses.
    pub inter_class: f64,
    /// `Cv`: mean distance from a state to its own class centroid, averaged over classes and then
    /// over the members of each class.
    pub intra_class: f64,
    /// `Cd / (Cv + 1)`. The `+ 1` is in the source and keeps the statistic finite for a class with
    /// no spread; it also means the figure is **not** scale-free, so comparing it between
    /// reservoirs whose states have different magnitudes compares two different things.
    pub ratio: f64,
    /// Number of distinct classes found in the labels.
    pub classes: usize,
    /// Number of states.
    pub samples: usize,
}

/// The separation statistic from Goodman & Ventura, *Spatiotemporal Pattern Recognition via Liquid
/// State Machines*, IJCNN 2006: `Sep = Cd / (Cv + 1)`.
///
/// `classes[i]` is the class label of `states[i]`; labels are arbitrary `usize` values and need not
/// be contiguous. This is the form used when a liquid is being judged before a readout is trained —
/// it says whether the information is there, without saying whether a linear map can get it out.
/// That second question is [`approximation_residual`].
///
/// # Errors
///
/// [`ReservoirError::Empty`] if `states` is empty or a state is empty,
/// [`ReservoirError::ShapeMismatch`] if `classes` is a different length or the states have
/// different widths, and [`ReservoirError::NonFinite`] for a non-finite component.
pub fn separation_ratio(
    states: &[Vec<f64>],
    classes: &[usize],
) -> Result<SeparationReport, ReservoirError> {
    if states.is_empty() {
        return Err(ReservoirError::Empty { what: "states" });
    }
    shape(classes.len(), states.len(), "class labels")?;
    let d = states[0].len();
    if d == 0 {
        return Err(ReservoirError::Empty { what: "state" });
    }
    for s in states {
        shape(s.len(), d, "state")?;
        finite(s, "state")?;
    }

    let mut labels: Vec<usize> = classes.to_vec();
    labels.sort_unstable();
    labels.dedup();
    let k = labels.len();

    let mut centroids = vec![vec![0.0f64; d]; k];
    let mut counts = vec![0usize; k];
    for (s, &c) in states.iter().zip(classes) {
        // `partition_point` rather than `binary_search().unwrap()`: every label in `classes` is in
        // `labels` by construction, so the two agree, and this form has no panic path at all.
        let idx = labels.partition_point(|&l| l < c);
        counts[idx] += 1;
        for j in 0..d {
            centroids[idx][j] += s[j];
        }
    }
    for i in 0..k {
        let n = counts[i] as f64;
        for j in 0..d {
            centroids[i][j] /= n;
        }
    }

    let mut cd = 0.0;
    for i in 0..k {
        for j in 0..k {
            cd += dist(&centroids[i], &centroids[j]);
        }
    }
    cd /= (k * k) as f64;

    let mut cv = 0.0;
    for i in 0..k {
        let mut acc = 0.0;
        for (s, &c) in states.iter().zip(classes) {
            if labels.partition_point(|&l| l < c) == i {
                acc += dist(s, &centroids[i]);
            }
        }
        cv += acc / counts[i] as f64;
    }
    cv /= k as f64;

    Ok(SeparationReport {
        inter_class: cd,
        intra_class: cv,
        ratio: cd / (cv + 1.0),
        classes: k,
        samples: states.len(),
    })
}

/// Maass's approximation property as a number: the residual of the best linear readout.
///
/// Fits a ridge readout from `states` to `targets` and returns its root-mean-square residual **on
/// the same data**. That is the in-sample residual and it is optimistic by construction; it answers
/// "is the information linearly present", which is what the approximation property asks, and does
/// not answer "will this generalise", which needs a held-out split the caller makes.
///
/// A residual of zero means the target is exactly a linear function of the state. That is the case
/// this is tested against: a synthetic target built as `A·state` comes back with a residual at
/// floating-point noise.
///
/// # Errors
///
/// As [`Ridge::fit`] and [`Readout::rmse`].
pub fn approximation_residual(
    states: &[Vec<f64>],
    targets: &[Vec<f64>],
    alpha: f64,
) -> Result<f64, ReservoirError> {
    let r = Ridge::fit(states, targets, alpha, true)?;
    r.rmse(states, targets)
}

/// Jaeger's short-term memory capacity, per delay and in total.
#[derive(Debug, Clone, PartialEq)]
pub struct MemoryCapacity {
    /// `MC = Σₖ r²(u(t−k), ŷₖ(t))`, dimensionless.
    pub total: f64,
    /// `r²` for each delay `k = 1 ..= max_delay`, in order. Each entry is in `[0, 1]`.
    pub per_delay: Vec<f64>,
    /// Reservoir units, which is the theoretical ceiling on [`MemoryCapacity::total`].
    pub units: usize,
    /// Samples the reconstruction was fitted on.
    pub samples: usize,
}

/// Measure short-term memory capacity: how much of an i.i.d. input the reservoir still holds.
///
/// From Jaeger, *Short Term Memory in Echo State Networks*, GMD Report 152, German National Research
/// Center for Information Technology, 2002. Drive the reservoir with an i.i.d. uniform scalar input,
/// train one linear readout per delay `k` to reconstruct `u(t − k)` from the state at `t`, and sum
/// the squared correlation coefficients.
///
/// **The bound is `MC ≤ units`**, and it is a theorem rather than an observation: every
/// reconstruction is a linear functional of the same `units`-dimensional state, so the capacities
/// cannot sum past the dimension. In practice a reservoir reaches a fraction of it — the tests here
/// see roughly a fifth — and the gap is the cost of a nonlinearity that is buying something else.
///
/// The readout is fitted on the same data the `r²` is computed from, which is the definition in the
/// source. It makes the figure optimistic for small sample counts, so `samples` should be several
/// times `units`; this implementation refuses fewer samples than delays.
///
/// # Errors
///
/// [`ReservoirError::ShapeMismatch`] if the reservoir does not have exactly one input channel,
/// [`ReservoirError::Empty`] for `max_delay == 0`, [`ReservoirError::OutOfRange`] if `samples` is
/// not greater than `max_delay`, and whatever [`Ridge`] returns.
pub fn memory_capacity(
    esn: &Esn,
    max_delay: usize,
    samples: usize,
    washout: usize,
    alpha: f64,
    seed: u64,
) -> Result<MemoryCapacity, ReservoirError> {
    shape(esn.n_in, 1, "input channels")?;
    if max_delay == 0 {
        return Err(ReservoirError::Empty { what: "delays" });
    }
    if samples <= max_delay {
        return Err(ReservoirError::OutOfRange {
            what: "samples",
            value: samples as f64,
            low: max_delay as f64 + 1.0,
            high: f64::INFINITY,
        });
    }

    let total_steps = washout + max_delay + samples;
    let mut rng = Rng::new(seed);
    let u: Vec<f64> = (0..total_steps).map(|_| sym(&mut rng)).collect();

    let mut r = esn.clone();
    r.reset();
    let mut states: Vec<Vec<f64>> = Vec::with_capacity(samples);
    let mut targets: Vec<Vec<f64>> = Vec::with_capacity(samples);
    for t in 0..total_steps {
        r.step(&u[t..=t])?;
        if t >= washout + max_delay {
            states.push(r.state().to_vec());
            targets.push((1..=max_delay).map(|k| u[t - k]).collect());
        }
    }

    let readout = Ridge::fit(&states, &targets, alpha, true)?;
    let mut per_delay = Vec::with_capacity(max_delay);
    let preds: Vec<Vec<f64>> =
        states.iter().map(|s| readout.predict(s)).collect::<Result<_, _>>()?;
    for k in 0..max_delay {
        let p: Vec<f64> = preds.iter().map(|v| v[k]).collect();
        let q: Vec<f64> = targets.iter().map(|v| v[k]).collect();
        per_delay.push(r2(&p, &q));
    }
    let total = per_delay.iter().sum();
    Ok(MemoryCapacity { total, per_delay, units: esn.units, samples: states.len() })
}

/// Squared Pearson correlation, with zero rather than a division by zero for a constant series.
///
/// A prediction that never moves explains none of the target's variance, so `0.0` is the answer
/// rather than an error: it is a real value of the statistic, reached by a real reservoir whose
/// memory of that delay has decayed to nothing.
fn r2(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let ma = a.iter().sum::<f64>() / n;
    let mb = b.iter().sum::<f64>() / n;
    let mut sab = 0.0;
    let mut saa = 0.0;
    let mut sbb = 0.0;
    for (&x, &y) in a.iter().zip(b) {
        let da = x - ma;
        let db = y - mb;
        sab += da * db;
        saa += da * da;
        sbb += db * db;
    }
    if saa <= 0.0 || sbb <= 0.0 {
        return 0.0;
    }
    let r = sab / (saa * sbb).sqrt();
    (r * r).min(1.0)
}

// ---------------------------------------------------------------------------------------------
// The liquid state machine: Maass's spiking reservoir.
// ---------------------------------------------------------------------------------------------

/// Whether a liquid neuron excites or inhibits everything it projects to.
///
/// The split is not decoration. Dale's principle — a neuron releases the same transmitter at all of
/// its terminals — is what makes the sign a property of the **presynaptic cell** rather than of the
/// synapse, and it is what the connection-probability table below is indexed by.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cell {
    /// Positive outgoing weights: every arriving spike displaces the target's membrane upward.
    Excitatory,
    /// Negative outgoing weights. 20% of a cortical column and of the default [`LiquidSpec`].
    Inhibitory,
}

/// How to build a [`Liquid`].
///
/// [`LiquidSpec::maass_column`] is the 15×3×3 column from the 2002 paper. Read the module doc for
/// exactly which of these numbers are the paper's and which are this crate's convention: the
/// probabilities and delays are transcribed, the weights are not.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LiquidSpec {
    /// Column extent along x, in lattice sites.
    pub nx: usize,
    /// Column extent along y, in lattice sites.
    pub ny: usize,
    /// Column extent along z, in lattice sites.
    pub nz: usize,
    /// Fraction of neurons that are inhibitory, in `[0, 1]`. The count is `round(frac · n)`.
    pub inhibitory_fraction: f64,
    /// `λ` in `C · exp(−D²/λ²)`, in lattice sites.
    ///
    /// Controls both the mean number of connections and the mean distance between connected
    /// neurons, which the paper points out cannot be tuned separately: raising `λ` makes the liquid
    /// both denser and less local, and those are the two things a reservoir trades against each
    /// other.
    pub lambda: f64,
    /// `C` for excitatory → excitatory. The paper's value is 0.3.
    pub c_ee: f64,
    /// `C` for excitatory → inhibitory. The paper's value is 0.2.
    pub c_ei: f64,
    /// `C` for inhibitory → excitatory. The paper's value is 0.4.
    pub c_ie: f64,
    /// `C` for inhibitory → inhibitory. The paper's value is 0.1.
    pub c_ii: f64,
    /// Weight of an excitatory → excitatory synapse, volts of membrane displacement per spike,
    /// positive. **Not the paper's number** — see the module doc.
    pub w_ee: f64,
    /// Weight of an excitatory → inhibitory synapse, volts, positive.
    pub w_ei: f64,
    /// Weight of an inhibitory → excitatory synapse, volts, negative.
    pub w_ie: f64,
    /// Weight of an inhibitory → inhibitory synapse, volts, negative.
    pub w_ii: f64,
    /// Transmission delay for excitatory → excitatory, seconds. The paper's value is 1.5 ms.
    pub delay_ee: f64,
    /// Transmission delay for every other pair, seconds. The paper's value is 0.8 ms.
    pub delay_other: f64,
    /// Tick length, seconds. Delays are converted to ticks at build time by
    /// `round(delay / dt)`, because [`crate::net::Net::delay`] is in ticks.
    pub dt: f64,
    /// Seed for the inhibitory assignment and the connection draws.
    pub seed: u64,
}

impl LiquidSpec {
    /// The column from Maass, Natschläger & Markram, Neural Computation 14:2531–2560, 2002:
    /// 15×3×3 = 135 neurons, 20% inhibitory, `λ = 2`, `C` of 0.3/0.2/0.4/0.1, delays of 1.5 ms for
    /// excitatory→excitatory and 0.8 ms otherwise, at a 0.1 ms tick.
    ///
    /// The weights are this crate's, for the reason in the module doc. They are sized by measurement
    /// rather than taken from the paper: with 4 nA driven into a tenth of the cells, the default
    /// column runs at a mean rate of **5.8 Hz** and at 6 nA at **11.0 Hz**, which is a regime where
    /// it neither falls silent nor saturates. Those two figures are from this crate's own probe on
    /// this crate's own synapse model and are not a reproduction of anything published.
    #[must_use]
    pub fn maass_column() -> Self {
        Self {
            nx: 15,
            ny: 3,
            nz: 3,
            inhibitory_fraction: 0.2,
            lambda: 2.0,
            c_ee: 0.3,
            c_ei: 0.2,
            c_ie: 0.4,
            c_ii: 0.1,
            w_ee: 1.6e-3,
            w_ei: 1.6e-3,
            w_ie: -2.4e-3,
            w_ii: -2.4e-3,
            delay_ee: 1.5e-3,
            delay_other: 0.8e-3,
            dt: 1e-4,
            seed: 0x11D,
        }
    }

    /// Neurons in the column, `nx · ny · nz`.
    #[must_use]
    pub fn neurons(&self) -> usize {
        self.nx * self.ny * self.nz
    }

    /// The `C` for a presynaptic/postsynaptic type pair.
    #[must_use]
    pub fn c_for(&self, pre: Cell, post: Cell) -> f64 {
        match (pre, post) {
            (Cell::Excitatory, Cell::Excitatory) => self.c_ee,
            (Cell::Excitatory, Cell::Inhibitory) => self.c_ei,
            (Cell::Inhibitory, Cell::Excitatory) => self.c_ie,
            (Cell::Inhibitory, Cell::Inhibitory) => self.c_ii,
        }
    }

    /// The weight for a presynaptic/postsynaptic type pair, volts per spike.
    #[must_use]
    pub fn w_for(&self, pre: Cell, post: Cell) -> f64 {
        match (pre, post) {
            (Cell::Excitatory, Cell::Excitatory) => self.w_ee,
            (Cell::Excitatory, Cell::Inhibitory) => self.w_ei,
            (Cell::Inhibitory, Cell::Excitatory) => self.w_ie,
            (Cell::Inhibitory, Cell::Inhibitory) => self.w_ii,
        }
    }
}

/// A spiking reservoir on a three-dimensional lattice.
///
/// Built once from a [`LiquidSpec`] and then never retrained. [`Liquid::net`] is an ordinary
/// [`Net`], so everything in [`crate::sim`] and [`crate::ledger`] applies to it unchanged — which is
/// the point of putting the reservoir on the crate's own network type rather than inventing a
/// parallel one.
#[derive(Debug, Clone, PartialEq)]
pub struct Liquid {
    /// The connectivity, ready for [`crate::sim::Sim`].
    pub net: Net,
    /// Type of each neuron, indexed by neuron id.
    pub kinds: Vec<Cell>,
    /// Lattice position of each neuron as `[x, y, z]` in sites, indexed by neuron id.
    ///
    /// Integer-valued but stored as `f64` because the only thing done with them is a Euclidean
    /// distance, and converting per pair would be the same arithmetic done `n²` times.
    pub positions: Vec<[f64; 3]>,
    /// The spec it was built from, kept so [`Liquid::expected_synapses`] can be asked afterwards.
    pub spec: LiquidSpec,
}

impl Liquid {
    /// Build the column.
    ///
    /// Inhibitory neurons are chosen by a partial Fisher-Yates shuffle over the neuron indices, so
    /// the assignment is a uniformly random subset of exactly the right size and is reproducible
    /// from the seed. Connections are then drawn for every **ordered** pair of distinct neurons:
    /// self-connections are excluded, and a pair can produce a synapse in each direction
    /// independently with the two different probabilities its two type orderings give.
    ///
    /// # Errors
    ///
    /// [`ReservoirError::Empty`] for a zero-sized lattice, [`ReservoirError::OutOfRange`] for an
    /// inhibitory fraction outside `[0, 1]`, a non-positive `lambda`, a non-positive `dt`, a
    /// negative delay, a `C` outside `[0, 1]`, a non-positive excitatory weight or a non-negative
    /// inhibitory one, [`ReservoirError::NonFinite`] for a non-finite parameter, and
    /// [`ReservoirError::Net`] if the underlying builder refuses a synapse.
    pub fn build(spec: &LiquidSpec) -> Result<Self, ReservoirError> {
        let n = spec.neurons();
        if n == 0 {
            return Err(ReservoirError::Empty { what: "lattice" });
        }
        if !(0.0..=1.0).contains(&spec.inhibitory_fraction) {
            return Err(ReservoirError::OutOfRange {
                what: "inhibitory_fraction",
                value: spec.inhibitory_fraction,
                low: 0.0,
                high: 1.0,
            });
        }
        for (what, v) in [("lambda", spec.lambda), ("dt", spec.dt)] {
            if !(v > 0.0) || !v.is_finite() {
                return Err(ReservoirError::OutOfRange {
                    what,
                    value: v,
                    low: f64::MIN_POSITIVE,
                    high: f64::INFINITY,
                });
            }
        }
        for (what, v) in
            [("c_ee", spec.c_ee), ("c_ei", spec.c_ei), ("c_ie", spec.c_ie), ("c_ii", spec.c_ii)]
        {
            if !(0.0..=1.0).contains(&v) {
                return Err(ReservoirError::OutOfRange { what, value: v, low: 0.0, high: 1.0 });
            }
        }
        for (what, v) in [("w_ee", spec.w_ee), ("w_ei", spec.w_ei)] {
            if !(v > 0.0) || !v.is_finite() {
                return Err(ReservoirError::OutOfRange {
                    what,
                    value: v,
                    low: f64::MIN_POSITIVE,
                    high: f64::INFINITY,
                });
            }
        }
        for (what, v) in [("w_ie", spec.w_ie), ("w_ii", spec.w_ii)] {
            if !(v < 0.0) || !v.is_finite() {
                return Err(ReservoirError::OutOfRange {
                    what,
                    value: v,
                    low: f64::NEG_INFINITY,
                    high: -f64::MIN_POSITIVE,
                });
            }
        }
        for (what, v) in [("delay_ee", spec.delay_ee), ("delay_other", spec.delay_other)] {
            if !(v >= 0.0) || !v.is_finite() {
                return Err(ReservoirError::OutOfRange {
                    what,
                    value: v,
                    low: 0.0,
                    high: f64::INFINITY,
                });
            }
        }

        let mut positions = Vec::with_capacity(n);
        for i in 0..n {
            let x = i / (spec.ny * spec.nz);
            let rem = i % (spec.ny * spec.nz);
            let y = rem / spec.nz;
            let z = rem % spec.nz;
            positions.push([x as f64, y as f64, z as f64]);
        }

        let mut rng = Rng::new(spec.seed);
        let n_inh = (spec.inhibitory_fraction * n as f64).round() as usize;
        let n_inh = n_inh.min(n);
        let mut order: Vec<u32> = (0..n as u32).collect();
        // Partial Fisher-Yates: the first `n_inh` entries end up a uniformly random subset.
        for i in 0..n_inh {
            let j = i + rng.below((n - i) as u32) as usize;
            order.swap(i, j);
        }
        let mut kinds = vec![Cell::Excitatory; n];
        for &idx in &order[..n_inh] {
            kinds[idx as usize] = Cell::Inhibitory;
        }

        let ticks = |seconds: f64| (seconds / spec.dt).round() as u32;
        let d_ee = ticks(spec.delay_ee);
        let d_other = ticks(spec.delay_other);

        let mut b = NetBuilder::new(n);
        let l2 = spec.lambda * spec.lambda;
        for a in 0..n {
            for c in 0..n {
                if a == c {
                    continue;
                }
                let p = spec.c_for(kinds[a], kinds[c]) * (-sq_dist(&positions[a], &positions[c]) / l2).exp();
                // The draw happens for EVERY ordered pair whether or not it lands, so the stream
                // position depends only on the lattice size. Drawing conditionally would make two
                // liquids that differ in one probability diverge completely rather than differ in
                // the edges that probability governs.
                let hit = rng.next_f64() < p;
                if hit {
                    let w = spec.w_for(kinds[a], kinds[c]);
                    let d = if kinds[a] == Cell::Excitatory && kinds[c] == Cell::Excitatory {
                        d_ee
                    } else {
                        d_other
                    };
                    b.connect(a as u32, c as u32, w, d)?;
                }
            }
        }

        Ok(Self { net: b.build(), kinds, positions, spec: *spec })
    }

    /// Neurons in the liquid.
    #[must_use]
    pub fn n(&self) -> usize {
        self.kinds.len()
    }

    /// Inhibitory neurons in the liquid.
    #[must_use]
    pub fn n_inhibitory(&self) -> usize {
        self.kinds.iter().filter(|&&k| k == Cell::Inhibitory).count()
    }

    /// The probability the builder used for the ordered pair `(a, b)`: `C · exp(−D² / λ²)`, or
    /// `0.0` for `a == b` since self-connections are excluded.
    ///
    /// Exposed because it is the closed form the synapse count is checked against, and because a
    /// reader who wants to know how local a liquid is should be able to ask rather than re-derive.
    ///
    /// # Panics
    ///
    /// If either index is past the neuron count.
    #[must_use]
    pub fn connection_probability(&self, a: usize, b: usize) -> f64 {
        assert!(a < self.n() && b < self.n(), "neuron index past the liquid's {} cells", self.n());
        if a == b {
            return 0.0;
        }
        let l2 = self.spec.lambda * self.spec.lambda;
        self.spec.c_for(self.kinds[a], self.kinds[b])
            * (-sq_dist(&self.positions[a], &self.positions[b]) / l2).exp()
    }

    /// The expected synapse count, `Σ p` over all ordered pairs of distinct neurons.
    ///
    /// The connections are independent Bernoulli draws, so this is the mean of the count the
    /// builder produces and [`Liquid::synapse_sd`] is its standard deviation. Together they are the
    /// closed form that `the_synapse_count_matches_the_analytic_expectation` checks the sampler
    /// against — a real check, because the expectation is a different computation from the draw.
    #[must_use]
    pub fn expected_synapses(&self) -> f64 {
        let n = self.n();
        let mut s = 0.0;
        for a in 0..n {
            for b in 0..n {
                s += self.connection_probability(a, b);
            }
        }
        s
    }

    /// Standard deviation of the synapse count, `sqrt(Σ p(1−p))` over the same ordered pairs.
    #[must_use]
    pub fn synapse_sd(&self) -> f64 {
        let n = self.n();
        let mut s = 0.0;
        for a in 0..n {
            for b in 0..n {
                let p = self.connection_probability(a, b);
                s += p * (1.0 - p);
            }
        }
        s.sqrt()
    }

    /// Choose `per_channel` distinct excitatory neurons for each of `channels` input channels.
    ///
    /// Input is injected into excitatory cells only, which is the arrangement in the paper: an
    /// afferent that drove the inhibitory population directly would suppress the liquid rather than
    /// perturb it. The sets for different channels are drawn independently and may overlap.
    ///
    /// # Errors
    ///
    /// [`ReservoirError::Empty`] for zero channels or zero sites per channel, and
    /// [`ReservoirError::OutOfRange`] if the liquid has fewer excitatory neurons than
    /// `per_channel`.
    pub fn input_sites(
        &self,
        channels: usize,
        per_channel: usize,
        seed: u64,
    ) -> Result<Vec<Vec<u32>>, ReservoirError> {
        if channels == 0 {
            return Err(ReservoirError::Empty { what: "channels" });
        }
        if per_channel == 0 {
            return Err(ReservoirError::Empty { what: "sites per channel" });
        }
        let exc: Vec<u32> = (0..self.n() as u32)
            .filter(|&i| self.kinds[i as usize] == Cell::Excitatory)
            .collect();
        if exc.len() < per_channel {
            return Err(ReservoirError::OutOfRange {
                what: "per_channel",
                value: per_channel as f64,
                low: 1.0,
                high: exc.len() as f64,
            });
        }
        let mut rng = Rng::new(seed);
        let mut out = Vec::with_capacity(channels);
        for _ in 0..channels {
            let mut pool = exc.clone();
            for i in 0..per_channel {
                let j = i + rng.below((pool.len() - i) as u32) as usize;
                pool.swap(i, j);
            }
            let mut pick = pool[..per_channel].to_vec();
            pick.sort_unstable();
            out.push(pick);
        }
        Ok(out)
    }

    /// Build a simulation of this liquid, giving excitatory and inhibitory cells different
    /// membranes.
    ///
    /// # Errors
    ///
    /// [`ReservoirError::Sim`] if [`crate::sim::Sim::new`] refuses — which for [`Lif`] can only be
    /// a neuron-count mismatch, since `Lif` is exact over gaps and both modes are therefore legal.
    pub fn to_sim(
        &self,
        excitatory: Lif,
        inhibitory: Lif,
        dt: f64,
        mode: Mode,
    ) -> Result<Sim<Lif>, ReservoirError> {
        let cells: Vec<Lif> = self
            .kinds
            .iter()
            .map(|k| if *k == Cell::Excitatory { excitatory } else { inhibitory })
            .collect();
        Ok(Sim::new(self.net.clone(), cells, dt, mode)?)
    }

    /// Run the liquid and return the filtered liquid state after every tick.
    ///
    /// `input[k]` is the external current in amperes for every neuron on tick `k`, so
    /// `input.len()` is the run length. The liquid state is the exponentially filtered spike train,
    /// which is what Maass reads out: a spike adds 1 to its neuron's trace and every trace decays
    /// by `exp(−dt / tau)` per tick.
    ///
    /// # Errors
    ///
    /// [`ReservoirError::Empty`] for an empty input, [`ReservoirError::ShapeMismatch`] if a tick's
    /// current vector is not `n` long, [`ReservoirError::OutOfRange`] for a non-positive `dt` or
    /// `tau`, [`ReservoirError::NonFinite`] for a non-finite current, and [`ReservoirError::Sim`]
    /// from the simulator.
    pub fn respond(
        &self,
        excitatory: Lif,
        inhibitory: Lif,
        dt: f64,
        tau: f64,
        input: &[Vec<f64>],
    ) -> Result<Vec<Vec<f64>>, ReservoirError> {
        if input.is_empty() {
            return Err(ReservoirError::Empty { what: "input" });
        }
        for (what, v) in [("dt", dt), ("tau", tau)] {
            if !(v > 0.0) || !v.is_finite() {
                return Err(ReservoirError::OutOfRange {
                    what,
                    value: v,
                    low: f64::MIN_POSITIVE,
                    high: f64::INFINITY,
                });
            }
        }
        let n = self.n();
        for row in input {
            shape(row.len(), n, "tick current")?;
            finite(row, "tick current")?;
        }
        let mut sim = self.to_sim(excitatory, inhibitory, dt, Mode::Clocked)?;
        let mut filter = SpikeFilter::new(n, tau, dt)?;
        let mut out = Vec::with_capacity(input.len());
        for row in input {
            let fired = sim.step(row);
            filter.step(&fired);
            out.push(filter.state().to_vec());
        }
        Ok(out)
    }
}

/// Squared Euclidean distance on the lattice, in sites².
fn sq_dist(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    dx * dx + dy * dy + dz * dz
}

/// An exponential trace per neuron: the liquid state of a spiking reservoir.
///
/// A spike train is a sequence of instants and a readout needs a vector, so something has to turn
/// one into the other. The standard choice, and Maass's, is a low-pass filter: each neuron carries
/// a trace that jumps by 1 on a spike and decays with time constant `tau`.
///
/// # The closed form this is checked against
///
/// A single spike at tick 0 leaves the trace at `exp(−k·dt/tau)` on tick `k`, so the trace summed
/// over all subsequent ticks is the geometric series `1 / (1 − exp(−dt/tau))` **exactly**. That is
/// an identity, not an approximation, and it is what the filter's test asserts — which pins the
/// decay factor, the increment and the order of the two operations in one comparison.
#[derive(Debug, Clone, PartialEq)]
pub struct SpikeFilter {
    /// Filter time constant, seconds. Longer means the state remembers spikes for longer and
    /// separates slower inputs; shorter means it tracks fast structure and forgets.
    pub tau: f64,
    /// Tick length, seconds.
    pub dt: f64,
    /// Per-neuron trace, dimensionless and non-negative.
    pub trace: Vec<f64>,
    /// `exp(−dt / tau)`, the per-tick decay, in `(0, 1)`. Cached because it is otherwise an
    /// exponential per neuron per tick.
    pub decay: f64,
}

impl SpikeFilter {
    /// A filter over `n` neurons.
    ///
    /// # Errors
    ///
    /// [`ReservoirError::Empty`] for `n == 0`, and [`ReservoirError::OutOfRange`] if `tau` or `dt`
    /// is not positive and finite.
    pub fn new(n: usize, tau: f64, dt: f64) -> Result<Self, ReservoirError> {
        if n == 0 {
            return Err(ReservoirError::Empty { what: "neurons" });
        }
        for (what, v) in [("tau", tau), ("dt", dt)] {
            if !(v > 0.0) || !v.is_finite() {
                return Err(ReservoirError::OutOfRange {
                    what,
                    value: v,
                    low: f64::MIN_POSITIVE,
                    high: f64::INFINITY,
                });
            }
        }
        Ok(Self { tau, dt, trace: vec![0.0; n], decay: (-dt / tau).exp() })
    }

    /// Decay every trace by one tick, then add 1 for each neuron in `fired`.
    ///
    /// **Decay first, then add.** The other order would make a spike's own tick worth
    /// `exp(−dt/tau)` instead of 1 and shift the whole trace by a tick — a difference that is
    /// invisible in a plot and moves the closed form above.
    ///
    /// Out-of-range ids in `fired` are ignored rather than panicking, matching
    /// [`crate::net::Net::out_of`]: a caller sweeping neuron indices past the end is asking a
    /// question whose answer is "nothing".
    pub fn step(&mut self, fired: &[u32]) {
        for v in &mut self.trace {
            *v *= self.decay;
        }
        for &i in fired {
            if let Some(v) = self.trace.get_mut(i as usize) {
                *v += 1.0;
            }
        }
    }

    /// The current trace.
    #[must_use]
    pub fn state(&self) -> &[f64] {
        &self.trace
    }

    /// Return every trace to zero.
    pub fn reset(&mut self) {
        self.trace.iter_mut().for_each(|v| *v = 0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Cell, Cholesky, Esn, EsnSpec, Liquid, LiquidSpec, Readout, ReservoirError, Ridge,
        SpikeFilter, approximation_residual, cholesky, echo_state_check, memory_capacity,
        power_iteration, separation, separation_ratio,
    };
    use crate::neuron::Lif;
    use crate::rng::Rng;
    use crate::sim::Mode;

    /// `Q D Q` with `Q = I − 2vvᵀ` a Householder reflection: orthogonal and symmetric, so the
    /// product is symmetric with **exactly** the eigenvalues in `d`, whatever `v` is. This is how a
    /// matrix with a known spectrum is manufactured without an eigenvalue routine to check against.
    fn householder_conjugate(d: &[f64], v: &[f64]) -> Vec<f64> {
        let n = d.len();
        let nv: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        let u: Vec<f64> = v.iter().map(|x| x / nv).collect();
        let mut q = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                q[i * n + j] = if i == j { 1.0 } else { 0.0 } - 2.0 * u[i] * u[j];
            }
        }
        // A = Q D Q
        let mut qd = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                qd[i * n + j] = q[i * n + j] * d[j];
            }
        }
        let mut a = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0;
                for k in 0..n {
                    s += qd[i * n + k] * q[k * n + j];
                }
                a[i * n + j] = s;
            }
        }
        a
    }

    fn esn(units: usize, radius: f64, leak: f64, seed: u64, n_in: usize) -> Esn {
        let spec = EsnSpec {
            units,
            spectral_radius: radius,
            density: 0.2,
            input_scaling: 1.0,
            bias_scaling: 0.0,
            leak,
            seed,
        };
        Esn::new(&spec, n_in).expect("a well-formed spec")
    }

    // -------------------------------------------------------------------------------------------
    // (c) Power iteration against matrices whose eigenvalues are known before the code runs.
    // -------------------------------------------------------------------------------------------

    #[test]
    fn power_iteration_recovers_a_known_symmetric_spectrum() {
        let a = householder_conjugate(&[4.0, -2.0, 1.0], &[1.0, 2.0, -0.5]);
        let s = power_iteration(&a, 3, 400, 1e-12, 5).unwrap();
        assert!((s.radius - 4.0).abs() < 1e-10, "radius {}", s.radius);
        // Symmetric, so the Rayleigh quotient IS the dominant eigenvalue, sign included.
        assert!((s.rayleigh - 4.0).abs() < 1e-10, "rayleigh {}", s.rayleigh);
        assert!(s.converged, "residual {}", s.residual);
    }

    /// The radius is a magnitude and cannot see a sign; the Rayleigh quotient can, and must.
    #[test]
    fn power_iteration_reports_a_negative_dominant_eigenvalue_through_the_rayleigh_quotient() {
        let a = householder_conjugate(&[-5.0, 2.0, 1.0], &[0.3, -1.0, 2.0]);
        let s = power_iteration(&a, 3, 400, 1e-12, 6).unwrap();
        assert!((s.radius - 5.0).abs() < 1e-10, "radius {}", s.radius);
        assert!((s.rayleigh + 5.0).abs() < 1e-10, "rayleigh {}", s.rayleigh);
    }

    /// A triangular matrix's eigenvalues are its diagonal, which is a fact about the matrix rather
    /// than about this implementation — and it is non-symmetric, so it exercises a path the
    /// Householder cases do not.
    #[test]
    fn power_iteration_recovers_the_diagonal_of_a_triangular_matrix() {
        #[rustfmt::skip]
        let a = vec![
            0.5, 1.0, -2.0,
            0.0, 2.0,  0.7,
            0.0, 0.0,  1.0,
        ];
        let s = power_iteration(&a, 3, 400, 1e-12, 7).unwrap();
        assert!((s.radius - 2.0).abs() < 1e-10, "radius {}", s.radius);
    }

    /// The case the geometric-mean estimator exists for. `[[0, −r], [r, 0]]` has eigenvalues `±ir`,
    /// so there is no real dominant eigenvector at all — but the matrix is a scaled rotation, every
    /// single-step growth factor is exactly `r`, and the radius must come out exact.
    #[test]
    fn power_iteration_handles_a_purely_complex_spectrum() {
        let r = 0.7;
        let a = vec![0.0, -r, r, 0.0];
        let s = power_iteration(&a, 2, 400, 1e-12, 8).unwrap();
        assert!((s.radius - r).abs() < 1e-12, "radius {}", s.radius);
        assert!(s.converged);
    }

    /// The case the window mean is **necessary** for, not merely convenient.
    ///
    /// Conjugating the scaled rotation by `diag(1, 3)` leaves the eigenvalues at `±0.7i` — the
    /// characteristic polynomial is still `λ² + 0.49` — but destroys normality, so the single-step
    /// growth factor oscillates between roughly `r/3` and `3r` forever and never settles on
    /// anything. A radius read off the last iterate would be wrong by a factor of three in either
    /// direction; the geometric mean over a window recovers `0.7`.
    #[test]
    fn a_non_normal_complex_pair_needs_the_window_mean_and_gets_the_exact_radius() {
        let r = 0.7;
        // S B S⁻¹ with S = diag(1, 3) and B = [[0, −r], [r, 0]].
        let a = vec![0.0, -r / 3.0, 3.0 * r, 0.0];
        let s = power_iteration(&a, 2, 400, 1e-9, 12).unwrap();
        assert!((s.radius - r).abs() < 1e-9, "radius {} against the exact {r}", s.radius);
        // And the claim about the single steps, so the test is not passing for a reason that
        // silently stopped being true: the iterate's growth really does swing by a factor of ~9.
        let mut x = vec![1.0, 0.0];
        let mut lo = f64::INFINITY;
        let mut hi = 0.0f64;
        for _ in 0..40 {
            let y = [a[0] * x[0] + a[1] * x[1], a[2] * x[0] + a[3] * x[1]];
            let g = (y[0] * y[0] + y[1] * y[1]).sqrt();
            lo = lo.min(g);
            hi = hi.max(g);
            x = vec![y[0] / g, y[1] / g];
        }
        assert!(hi / lo > 5.0, "the single-step growth only varied by {}", hi / lo);
    }

    #[test]
    fn the_spectral_radius_is_homogeneous_in_the_matrix() {
        let a = householder_conjugate(&[3.0, 1.5, -0.5], &[1.0, 1.0, 1.0]);
        let base = power_iteration(&a, 3, 400, 1e-12, 9).unwrap().radius;
        let scaled: Vec<f64> = a.iter().map(|x| -3.0 * x).collect();
        let got = power_iteration(&scaled, 3, 400, 1e-12, 9).unwrap().radius;
        assert!((got - 3.0 * base).abs() < 1e-9, "{got} vs {}", 3.0 * base);
    }

    /// A nilpotent matrix has every eigenvalue at zero, and the iteration reaches that answer by
    /// landing in the kernel rather than by guessing it.
    #[test]
    fn a_nilpotent_matrix_has_spectral_radius_zero() {
        let a = vec![0.0, 1.0, 0.0, 0.0];
        let s = power_iteration(&a, 2, 64, 1e-12, 10).unwrap();
        assert_eq!(s.radius, 0.0);
        assert!(s.converged);
        let zero = vec![0.0; 9];
        assert_eq!(power_iteration(&zero, 3, 64, 1e-12, 11).unwrap().radius, 0.0);
    }

    #[test]
    fn power_iteration_refuses_a_malformed_matrix_naming_what_was_wrong() {
        assert_eq!(
            power_iteration(&[1.0, 2.0, 3.0], 2, 64, 1e-12, 1).unwrap_err(),
            ReservoirError::ShapeMismatch { what: "matrix", got: 3, want: 4 }
        );
        assert_eq!(
            power_iteration(&[1.0, f64::NAN, 0.0, 1.0], 2, 64, 1e-12, 1).unwrap_err(),
            ReservoirError::NonFinite { what: "matrix", index: 1 }
        );
        assert!(matches!(
            power_iteration(&[1.0], 1, 2, 1e-12, 1).unwrap_err(),
            ReservoirError::OutOfRange { what: "max_iters", .. }
        ));
        assert!(matches!(
            power_iteration(&[], 0, 64, 1e-12, 1).unwrap_err(),
            ReservoirError::Empty { what: "matrix" }
        ));
    }

    // -------------------------------------------------------------------------------------------
    // (a) Ridge regression and the linear solver, against systems with known answers.
    // -------------------------------------------------------------------------------------------

    /// **The check the whole module's arithmetic rests on.** Build a design matrix, pick a
    /// coefficient matrix, manufacture the targets as their exact product, and require the solver
    /// to recover the coefficients to 1e-10. Nothing else in this crate exercises a linear solve,
    /// so if this is wrong every readout in the module is wrong in a way no other test can see.
    #[test]
    fn ridge_with_no_penalty_solves_a_consistent_system_exactly() {
        let mut rng = Rng::new(31);
        let p = 4;
        let q = 2;
        let want = [1.0, -2.0, 0.5, 3.0, -0.25, 0.75, 2.0, -1.5];
        let mut x = Vec::new();
        let mut y = Vec::new();
        for _ in 0..24 {
            let xi: Vec<f64> = (0..p).map(|_| 2.0 * rng.next_f64() - 1.0).collect();
            let yi: Vec<f64> =
                (0..q).map(|t| (0..p).map(|j| want[t * p + j] * xi[j]).sum()).collect();
            x.push(xi);
            y.push(yi);
        }
        let r = Ridge::fit(&x, &y, 0.0, false).unwrap();
        assert_eq!(r.w.len(), q * p);
        for t in 0..q {
            for j in 0..p {
                let got = r.w[t * p + j];
                assert!(
                    (got - want[t * p + j]).abs() < 1e-10,
                    "w[{t}][{j}] = {got} against {}",
                    want[t * p + j]
                );
            }
        }
        assert!(r.rmse(&x, &y).unwrap() < 1e-12, "residual {}", r.rmse(&x, &y).unwrap());
    }

    /// On an orthonormal design the ridge estimator has a closed form with no linear algebra in it
    /// at all: `w = Xᵀy / (1 + α)`. This pins the penalty's placement and its magnitude separately
    /// from the solve.
    #[test]
    fn ridge_on_an_orthonormal_design_matches_the_closed_form_shrinkage() {
        // Rows of a Householder reflection are orthonormal, so XᵀX is the identity exactly.
        let q = householder_conjugate(&[1.0, 1.0, 1.0], &[0.4, -1.0, 0.6]);
        let x: Vec<Vec<f64>> = (0..3).map(|i| q[i * 3..i * 3 + 3].to_vec()).collect();
        let yv = [0.8, -1.3, 2.5];
        let y: Vec<Vec<f64>> = yv.iter().map(|&v| vec![v]).collect();
        for &alpha in &[0.0, 0.5, 2.0, 10.0] {
            let r = Ridge::fit(&x, &y, alpha, false).unwrap();
            for j in 0..3 {
                // (Xᵀy)_j = Σ_i X[i][j] y_i
                let xty: f64 = (0..3).map(|i| x[i][j] * yv[i]).sum();
                let want = xty / (1.0 + alpha);
                assert!(
                    (r.w[j] - want).abs() < 1e-12,
                    "alpha {alpha}: w[{j}] = {} against {want}",
                    r.w[j]
                );
            }
        }
    }

    /// The whole reason a reservoir readout is a RIDGE and not a least squares: a reservoir always
    /// has nearly dependent units. With `α = 0` the normal equations are singular and the solver
    /// refuses, naming the row; with a penalty the same data fits.
    #[test]
    fn a_rank_deficient_design_is_refused_without_a_penalty_and_solved_with_one() {
        let mut rng = Rng::new(77);
        let mut x = Vec::new();
        let mut y = Vec::new();
        for _ in 0..30 {
            let a = 2.0 * rng.next_f64() - 1.0;
            let b = 2.0 * rng.next_f64() - 1.0;
            // Third column is a copy of the first: rank 2 in a 3-column design.
            x.push(vec![a, b, a]);
            y.push(vec![2.0 * a - b]);
        }
        let err = Ridge::fit(&x, &y, 0.0, false).unwrap_err();
        assert!(
            matches!(err, ReservoirError::IllConditioned { .. }),
            "expected a conditioning refusal, got {err}"
        );
        let r = Ridge::fit(&x, &y, 1e-6, false).unwrap();
        assert!(r.rmse(&x, &y).unwrap() < 1e-4, "penalised fit residual {}", r.rmse(&x, &y).unwrap());
        assert!(r.pivot_ratio > 1e6, "a rank-deficient fit reported pivot ratio {}", r.pivot_ratio);
    }

    /// For a diagonal matrix the Cholesky pivots ARE the eigenvalues, so the lower bound the type
    /// doc claims is tight and can be checked as an equality.
    #[test]
    fn the_cholesky_pivot_ratio_is_the_condition_number_of_a_diagonal_matrix() {
        let a = vec![4.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.25];
        let c: Cholesky = cholesky(&a, 3, 1e-12).unwrap();
        assert!((c.pivot_ratio - 16.0).abs() < 1e-12, "pivot ratio {}", c.pivot_ratio);
        // And the factorisation solves what it says it solves.
        let z = c.solve(&[8.0, 3.0, 1.0]).unwrap();
        for (got, want) in z.iter().zip([2.0, 3.0, 4.0]) {
            assert!((got - want).abs() < 1e-12, "{got} vs {want}");
        }
    }

    #[test]
    fn a_non_positive_definite_matrix_is_refused_rather_than_factorised() {
        // Symmetric, but with a negative eigenvalue: [[1, 2], [2, 1]] has eigenvalues 3 and −1.
        let a = vec![1.0, 2.0, 2.0, 1.0];
        assert!(matches!(
            cholesky(&a, 2, 1e-12).unwrap_err(),
            ReservoirError::IllConditioned { index: 1, .. }
        ));
    }

    /// The penalty has to do what a penalty does, monotonically.
    #[test]
    fn a_larger_penalty_shrinks_the_coefficients() {
        let mut rng = Rng::new(404);
        let x: Vec<Vec<f64>> =
            (0..40).map(|_| (0..5).map(|_| 2.0 * rng.next_f64() - 1.0).collect()).collect();
        let y: Vec<Vec<f64>> = x.iter().map(|xi| vec![xi[0] - 2.0 * xi[3]]).collect();
        let mut last = f64::INFINITY;
        for &alpha in &[0.0, 0.01, 0.1, 1.0, 10.0, 100.0] {
            let r: Readout = Ridge::fit(&x, &y, alpha, true).unwrap();
            let norm: f64 = r.w.iter().map(|v| v * v).sum::<f64>().sqrt();
            assert!(norm < last, "alpha {alpha} gave norm {norm}, not below {last}");
            last = norm;
        }
    }

    /// The intercept is **not** shrunk, which is the invariant on [`Ridge::bias`] and is otherwise
    /// invisible: with an enormous penalty the feature weights go to zero and the normal equation
    /// for the bias column collapses to `n·b = Σy`, so the intercept must land on the mean of the
    /// targets. A penalised intercept would land on `Σy / (n + α)`, which for `α = 1e8` is zero —
    /// and every prediction the readout ever made would be biased toward zero by an amount nobody
    /// chose.
    #[test]
    fn the_intercept_is_not_shrunk_by_the_penalty() {
        let mut rng = Rng::new(606);
        let x: Vec<Vec<f64>> =
            (0..64).map(|_| (0..3).map(|_| 2.0 * rng.next_f64() - 1.0).collect()).collect();
        let y: Vec<Vec<f64>> = x.iter().map(|xi| vec![7.0 + xi[0]]).collect();
        let mean: f64 = y.iter().map(|v| v[0]).sum::<f64>() / y.len() as f64;
        let r = Ridge::fit(&x, &y, 1e8, true).unwrap();
        let intercept = r.w[3];
        assert!((intercept - mean).abs() < 1e-3, "intercept {intercept} against mean {mean}");
        for j in 0..3 {
            assert!(r.w[j].abs() < 1e-3, "feature {j} survived a 1e8 penalty at {}", r.w[j]);
        }
    }

    /// The accumulate/solve split must give the same answer as the all-at-once path, because the
    /// streaming form is the one a device would use and the batch form is the one that gets tested.
    #[test]
    fn streaming_and_batch_fits_agree() {
        let mut rng = Rng::new(55);
        let x: Vec<Vec<f64>> =
            (0..50).map(|_| (0..4).map(|_| 2.0 * rng.next_f64() - 1.0).collect()).collect();
        let y: Vec<Vec<f64>> = x.iter().map(|xi| vec![xi[1] + 0.5 * xi[2]]).collect();
        let batch = Ridge::fit(&x, &y, 1e-3, true).unwrap();
        let mut acc = Ridge::new(4, 1, 1e-3, true).unwrap();
        for (xi, yi) in x.iter().zip(&y) {
            acc.accumulate(xi, yi).unwrap();
        }
        assert_eq!(batch, acc.solve().unwrap());
    }

    #[test]
    fn a_fit_with_nothing_in_it_is_refused() {
        let r = Ridge::new(3, 1, 1.0, true).unwrap();
        assert_eq!(r.solve().unwrap_err(), ReservoirError::NoSamples);
    }

    #[test]
    fn a_non_finite_sample_is_refused_at_the_boundary_naming_its_index() {
        let mut r = Ridge::new(3, 1, 1.0, true).unwrap();
        assert_eq!(
            r.accumulate(&[1.0, f64::INFINITY, 0.0], &[1.0]).unwrap_err(),
            ReservoirError::NonFinite { what: "sample", index: 1 }
        );
        assert_eq!(
            r.accumulate(&[1.0, 2.0], &[1.0]).unwrap_err(),
            ReservoirError::ShapeMismatch { what: "sample", got: 2, want: 3 }
        );
        assert_eq!(r.samples, 0, "a refused sample was still folded in");
    }

    // -------------------------------------------------------------------------------------------
    // (b) The echo state property.
    // -------------------------------------------------------------------------------------------

    /// **The same matrix, scaled two ways.** Below radius 1 both copies converge to the same state;
    /// above it they do not. Using one draw rescaled rather than two independent draws makes the
    /// spectral radius the only thing that differs, which is what turns a demonstration into a
    /// controlled comparison.
    ///
    /// The measured behaviour on this crate's generator: radii 0.5, 0.9 and 0.99 forget on 12 seeds
    /// out of 12; radii 1.5, 2, 3 and 5 forget on 0 out of 12. The assertion below is the weaker
    /// 6-of-6 against 0-of-6 so that it is not pinned to one lucky reservoir.
    #[test]
    fn the_echo_state_property_holds_below_radius_one_and_fails_above_it() {
        let run = |radius: f64, seed: u64| -> bool {
            let mut e = esn(60, 0.9, 1.0, 100 + seed, 1);
            e.rescale(radius).unwrap();
            let mut rng = Rng::new(7 + seed);
            let inputs: Vec<Vec<f64>> =
                (0..300).map(|_| vec![2.0 * rng.next_f64() - 1.0]).collect();
            let x0: Vec<f64> = (0..60).map(|_| 2.0 * rng.next_f64() - 1.0).collect();
            let y0: Vec<f64> = (0..60).map(|_| 2.0 * rng.next_f64() - 1.0).collect();
            let rep = echo_state_check(&e, &inputs, &x0, &y0, 1e-9).unwrap();
            assert!(rep.start > 0.1, "the two starts were not actually apart: {}", rep.start);
            rep.forgets
        };
        let below = (0..6u64).filter(|&s| run(0.9, s)).count();
        let above = (0..6u64).filter(|&s| run(3.0, s)).count();
        assert_eq!(below, 6, "a contracting reservoir failed to forget its initial state");
        assert_eq!(above, 0, "an expanding reservoir forgot its initial state");
    }

    /// And the rate is not arbitrary. Started 2e-9 apart, where `tanh` is linear to a part in 1e18,
    /// two copies separate exactly as the recurrent matrix's powers do, so the per-step contraction
    /// over the last hundred steps must approach the spectral radius itself.
    ///
    /// Measured: 0.599457 against 0.6, 0.799277 against 0.8, 0.949141 against 0.95 — all within
    /// 0.1%, converging to the dominant eigenvalue from below. The 1% tolerance is loose enough for
    /// the sub-dominant modes that have not quite died at step 200 and tight enough to catch a
    /// scaling that is wrong by a factor.
    #[test]
    fn the_contraction_rate_approaches_the_spectral_radius() {
        for &radius in &[0.6, 0.8, 0.95] {
            let mut e = esn(40, 0.9, 1.0, 5, 1);
            let measured = e.rescale(radius).unwrap();
            assert!((measured - radius).abs() < 1e-6, "rescale gave {measured} for {radius}");
            let mut a = e.clone();
            let mut b = e;
            let mut x0 = vec![0.0; 40];
            let mut y0 = vec![0.0; 40];
            x0[0] = 1e-9;
            y0[0] = -1e-9;
            a.set_state(&x0).unwrap();
            b.set_state(&y0).unwrap();
            let mut hist = Vec::new();
            for _ in 0..300 {
                a.step(&[0.0]).unwrap();
                b.step(&[0.0]).unwrap();
                hist.push(super::dist(a.state(), b.state()));
            }
            let n = hist.len();
            let rate = (hist[n - 1] / hist[n - 101]).powf(0.01);
            assert!(
                (rate - radius).abs() / radius < 0.01,
                "radius {radius}: contraction per step {rate}"
            );
        }
    }

    /// The invariant in the type doc, against a reservoir built to break it.
    #[test]
    fn the_state_stays_inside_the_unit_box_whatever_the_weights_and_input() {
        let spec = EsnSpec {
            units: 30,
            spectral_radius: 50.0,
            density: 1.0,
            input_scaling: 1e6,
            bias_scaling: 1e3,
            leak: 0.3,
            seed: 2,
        };
        let mut e = Esn::new(&spec, 2).unwrap();
        for k in 0..500 {
            let u = if k % 2 == 0 { [1e6, -1e6] } else { [-1e6, 1e6] };
            let x = e.step(&u).unwrap();
            for (i, &v) in x.iter().enumerate() {
                assert!(v.is_finite() && (-1.0..=1.0).contains(&v), "unit {i} left the box at {v}");
            }
        }
    }

    #[test]
    fn a_state_outside_the_box_is_refused_rather_than_clamped() {
        let mut e = esn(4, 0.9, 1.0, 3, 1);
        assert!(matches!(
            e.set_state(&[0.0, 1.5, 0.0, 0.0]).unwrap_err(),
            ReservoirError::OutOfRange { what: "state component", .. }
        ));
        assert_eq!(
            e.step(&[f64::NAN]).unwrap_err(),
            ReservoirError::NonFinite { what: "input", index: 0 }
        );
    }

    #[test]
    fn the_same_spec_builds_the_same_reservoir_on_every_run() {
        let spec = EsnSpec::default();
        assert_eq!(Esn::new(&spec, 3).unwrap(), Esn::new(&spec, 3).unwrap());
        let other = EsnSpec { seed: spec.seed + 1, ..spec };
        assert_ne!(Esn::new(&spec, 3).unwrap().w, Esn::new(&other, 3).unwrap().w);
    }

    #[test]
    fn the_leak_bridge_to_si_refuses_an_unstable_step() {
        assert!((EsnSpec::leak_from_tau(1e-3, 20e-3).unwrap() - 0.05).abs() < 1e-15);
        assert!(matches!(
            EsnSpec::leak_from_tau(40e-3, 20e-3).unwrap_err(),
            ReservoirError::OutOfRange { what: "dt / tau", .. }
        ));
        assert!(EsnSpec::leak_from_tau(0.0, 1.0).is_err());
    }

    // -------------------------------------------------------------------------------------------
    // (d) Separation, and (the approximation property) the residual of the best linear readout.
    // -------------------------------------------------------------------------------------------

    /// Maass's separation, measured on one reservoir at four sizes with the same two input streams.
    ///
    /// Measured: 1.39, 3.38, 6.38, 13.00 for 10, 40, 160 and 640 units. The growth is close to
    /// `sqrt(units)` — the per-coordinate separation is roughly constant at 0.44 to 0.53 — which is
    /// the caveat stated in [`separation`]'s doc rather than a hidden one: a larger liquid separates
    /// more partly by having more coordinates, and those coordinates are also readout parameters.
    #[test]
    fn separation_grows_with_reservoir_size() {
        let mut last = 0.0;
        for &n in &[10usize, 40, 160, 640] {
            let mut a = esn(n, 0.9, 1.0, 77, 1);
            let mut b = a.clone();
            let mut rng = Rng::new(21);
            let ina: Vec<Vec<f64>> = (0..60).map(|_| vec![2.0 * rng.next_f64() - 1.0]).collect();
            let inb: Vec<Vec<f64>> = (0..60).map(|_| vec![2.0 * rng.next_f64() - 1.0]).collect();
            for u in &ina {
                a.step(u).unwrap();
            }
            for u in &inb {
                b.step(u).unwrap();
            }
            let s = separation(a.state(), b.state()).unwrap();
            assert!(s > last, "separation {s} at {n} units did not exceed {last}");
            last = s;
        }
    }

    /// Identical input streams must give identical states, so separation is exactly zero — not
    /// nearly zero. If it were not, every separation figure in the module would carry a floor that
    /// nothing in the input put there.
    #[test]
    fn separation_is_exactly_zero_for_identical_inputs() {
        let mut a = esn(50, 0.9, 1.0, 12, 1);
        let mut b = a.clone();
        let mut rng = Rng::new(3);
        let inputs: Vec<Vec<f64>> = (0..80).map(|_| vec![2.0 * rng.next_f64() - 1.0]).collect();
        for u in &inputs {
            a.step(u).unwrap();
            b.step(u).unwrap();
        }
        assert_eq!(separation(a.state(), b.state()).unwrap(), 0.0);
        assert_eq!(
            separation(&[1.0, 2.0], &[1.0]).unwrap_err(),
            ReservoirError::ShapeMismatch { what: "second state", got: 1, want: 2 }
        );
    }

    /// The class statistic, against the two cases whose answers are known without measuring: a
    /// labelling where the classes are the same stream gives a ratio near zero, and one where they
    /// are different streams gives a ratio well above it.
    ///
    /// Measured on 20, 80 and 320 units with two jittered input classes: 1.46, 2.99, 4.67.
    #[test]
    fn the_separation_ratio_separates_real_classes_and_not_invented_ones() {
        let base = esn(80, 0.9, 1.0, 77, 1);
        let mut rng = Rng::new(5);
        let proto: Vec<Vec<f64>> = (0..50).map(|_| vec![2.0 * rng.next_f64() - 1.0]).collect();
        let other: Vec<Vec<f64>> = (0..50).map(|_| vec![2.0 * rng.next_f64() - 1.0]).collect();

        let mut states = Vec::new();
        let mut real = Vec::new();
        let mut invented = Vec::new();
        for (c, stream) in [(0usize, &proto), (1usize, &other)] {
            for k in 0..10 {
                let mut e = base.clone();
                for u in stream {
                    let jitter = 0.15 * (2.0 * rng.next_f64() - 1.0);
                    e.step(&[u[0] + jitter]).unwrap();
                }
                states.push(e.state().to_vec());
                real.push(c);
                // A labelling that cuts across the real classes carries no information at all.
                invented.push(k % 2);
            }
        }
        let good = separation_ratio(&states, &real).unwrap();
        let junk = separation_ratio(&states, &invented).unwrap();
        assert_eq!(good.classes, 2);
        assert_eq!(good.samples, 20);
        assert!(good.ratio > 2.0, "real classes separated by only {}", good.ratio);
        assert!(
            junk.ratio < good.ratio / 3.0,
            "an arbitrary labelling separated by {} against the real {}",
            junk.ratio,
            good.ratio
        );
    }

    /// The approximation property is a residual, and for a target that IS a linear function of the
    /// state the residual is zero. This is the closed-form end of the measurement.
    #[test]
    fn the_approximation_residual_vanishes_for_a_linearly_realisable_target() {
        let mut rng = Rng::new(88);
        let states: Vec<Vec<f64>> =
            (0..60).map(|_| (0..6).map(|_| 2.0 * rng.next_f64() - 1.0).collect()).collect();
        let a = [0.3, -1.2, 0.0, 2.0, 0.5, -0.4];
        let targets: Vec<Vec<f64>> = states
            .iter()
            .map(|s| vec![(0..6).map(|j| a[j] * s[j]).sum::<f64>() + 0.75])
            .collect();
        let r = approximation_residual(&states, &targets, 0.0).unwrap();
        assert!(r < 1e-12, "a linear target left a residual of {r}");

        // And a target that is NOT in the readout's reach leaves one that is not small. A product
        // of two state components is quadratic and no linear map recovers it.
        let hard: Vec<Vec<f64>> = states.iter().map(|s| vec![s[0] * s[1]]).collect();
        let r2 = approximation_residual(&states, &hard, 0.0).unwrap();
        assert!(r2 > 1e-3, "a quadratic target was fitted linearly to {r2}");
    }

    // -------------------------------------------------------------------------------------------
    // (e) The whole point: a reservoir does what a linear readout on the raw input cannot.
    // -------------------------------------------------------------------------------------------

    /// **Delayed XOR, with the honest baseline beside it.**
    ///
    /// Target is `u(t−1) XOR u(t−2)` on a ±1 stream. The baseline readout is given the raw window
    /// `[u(t), u(t−1), u(t−2)]` — *strictly more* of the relevant input than the reservoir sees at
    /// any instant, since the reservoir only ever receives `u(t)` — and a bias. It still cannot do
    /// it, because the obstacle is that XOR is not linearly separable, not that the past is missing.
    ///
    /// Measured on 500 held-out samples: reservoir RMSE 0.108 and accuracy 1.000; baseline RMSE
    /// 0.997 and accuracy 0.514, which is chance. The assertions below are loosened from those
    /// figures so that the test reports a broken reservoir rather than a reseeded one.
    #[test]
    fn a_reservoir_solves_delayed_xor_that_a_linear_readout_on_the_raw_input_cannot() {
        let (wash, n_train, n_test) = (100usize, 1_500usize, 500usize);
        let total = wash + n_train + n_test;
        let mut rng = Rng::new(1234);
        let bits: Vec<f64> =
            (0..total).map(|_| if rng.next_f64() < 0.5 { -1.0 } else { 1.0 }).collect();
        // +1 when the two past bits differ, −1 when they agree: XOR on a ±1 encoding.
        let target = |t: usize| if bits[t - 1] * bits[t - 2] < 0.0 { 1.0 } else { -1.0 };
        let ys: Vec<Vec<f64>> = (wash..total).map(|t| vec![target(t)]).collect();
        let (ytr, yte) = ys.split_at(n_train);

        let spec = EsnSpec {
            units: 80,
            spectral_radius: 0.9,
            density: 0.2,
            input_scaling: 1.0,
            bias_scaling: 0.2,
            leak: 0.7,
            seed: 909,
        };
        let mut e = Esn::new(&spec, 1).unwrap();
        let mut states = Vec::with_capacity(total - wash);
        for (t, &b) in bits.iter().enumerate() {
            e.step(&[b]).unwrap();
            if t >= wash {
                states.push(e.state().to_vec());
            }
        }
        let (xtr, xte) = states.split_at(n_train);
        let ro = Ridge::fit(xtr, ytr, 1e-6, true).unwrap();
        let rmse = ro.rmse(xte, yte).unwrap();
        let acc = xte
            .iter()
            .zip(yte)
            .filter(|(x, y)| ro.predict(x).unwrap()[0] * y[0] > 0.0)
            .count() as f64
            / n_test as f64;

        let bx: Vec<Vec<f64>> =
            (wash..total).map(|t| vec![bits[t], bits[t - 1], bits[t - 2]]).collect();
        let (bxtr, bxte) = bx.split_at(n_train);
        let bro = Ridge::fit(bxtr, ytr, 1e-6, true).unwrap();
        let brmse = bro.rmse(bxte, yte).unwrap();
        let bacc = bxte
            .iter()
            .zip(yte)
            .filter(|(x, y)| bro.predict(x).unwrap()[0] * y[0] > 0.0)
            .count() as f64
            / n_test as f64;

        assert!(acc > 0.97, "the reservoir reached only {acc} on held-out delayed XOR");
        assert!(rmse < 0.4, "reservoir RMSE {rmse}");
        assert!(bacc < 0.60, "the linear baseline reached {bacc}, which is not chance");
        assert!(brmse > 0.9, "baseline RMSE {brmse} was not the no-information value of about 1");
        assert!(brmse > 5.0 * rmse, "the gap was only {brmse} against {rmse}");
    }

    /// Jaeger's bound, `MC ≤ units`, is a theorem about linear readouts on a fixed-dimensional
    /// state and must hold for every reservoir this module can build.
    ///
    /// Measured at 50 units: 4.43 at radius 0.05, 9.97 at 0.5, 13.61 at 0.9, 14.45 at 0.99 — a
    /// quarter of the ceiling at best, which is the honest figure for a `tanh` reservoir.
    #[test]
    fn memory_capacity_stays_under_jaegers_bound_and_grows_with_the_spectral_radius() {
        let mut last = 0.0;
        for &radius in &[0.05, 0.5, 0.9] {
            let spec = EsnSpec {
                units: 50,
                spectral_radius: radius,
                density: 0.2,
                input_scaling: 0.5,
                bias_scaling: 0.0,
                leak: 1.0,
                seed: 3,
            };
            let e = Esn::new(&spec, 1).unwrap();
            let mc = memory_capacity(&e, 40, 1_500, 200, 1e-8, 42).unwrap();
            assert_eq!(mc.units, 50);
            assert_eq!(mc.per_delay.len(), 40);
            assert!(
                mc.total <= mc.units as f64 + 1e-9,
                "MC {} exceeded the {}-unit bound",
                mc.total,
                mc.units
            );
            for (k, &v) in mc.per_delay.iter().enumerate() {
                assert!((0.0..=1.0).contains(&v), "r2 at delay {} was {v}", k + 1);
            }
            assert!(mc.total > last, "radius {radius} gave MC {} against {last}", mc.total);
            last = mc.total;
        }
    }

    /// A reservoir with a single unit cannot hold more than one unit's worth, which is the bound at
    /// its tightest and the case where an off-by-one in the bookkeeping would show.
    #[test]
    fn a_one_unit_reservoir_obeys_the_bound_at_its_tightest() {
        let spec = EsnSpec {
            units: 1,
            spectral_radius: 0.9,
            density: 1.0,
            input_scaling: 0.5,
            bias_scaling: 0.0,
            leak: 1.0,
            seed: 17,
        };
        let e = Esn::new(&spec, 1).unwrap();
        let mc = memory_capacity(&e, 10, 600, 100, 1e-8, 5).unwrap();
        assert!(mc.total <= 1.0 + 1e-9, "a single unit reported MC {}", mc.total);
    }

    // -------------------------------------------------------------------------------------------
    // The spiking liquid.
    // -------------------------------------------------------------------------------------------

    /// The probability formula, against the arithmetic written out by hand. A one-dimensional
    /// lattice of excitatory cells has `p(0, k) = 0.3 · exp(−k² / 4)` and nothing else.
    #[test]
    fn the_connection_probability_is_the_papers_formula_site_by_site() {
        let spec = LiquidSpec {
            nx: 5,
            ny: 1,
            nz: 1,
            inhibitory_fraction: 0.0,
            ..LiquidSpec::maass_column()
        };
        let l = Liquid::build(&spec).unwrap();
        assert_eq!(l.n_inhibitory(), 0);
        assert_eq!(l.connection_probability(0, 0), 0.0, "a self-connection was given a probability");
        let mut last = f64::INFINITY;
        for k in 1..5 {
            let want = 0.3 * (-((k * k) as f64) / 4.0).exp();
            let got = l.connection_probability(0, k);
            assert!((got - want).abs() < 1e-15, "p(0,{k}) = {got} against {want}");
            assert!(got < last, "probability did not fall with distance at {k}");
            last = got;
        }
    }

    /// The expectation, against two neurons and one exponential.
    #[test]
    fn the_expected_synapse_count_is_the_hand_written_closed_form_on_two_neurons() {
        let spec = LiquidSpec {
            nx: 2,
            ny: 1,
            nz: 1,
            inhibitory_fraction: 0.0,
            lambda: 1.0,
            ..LiquidSpec::maass_column()
        };
        let l = Liquid::build(&spec).unwrap();
        // Two ordered pairs, both excitatory→excitatory, both at distance 1.
        let want = 2.0 * 0.3 * (-1.0f64).exp();
        assert!((l.expected_synapses() - want).abs() < 1e-15, "{} vs {want}", l.expected_synapses());
        let one = LiquidSpec { nx: 1, ny: 1, nz: 1, ..spec };
        assert_eq!(Liquid::build(&one).unwrap().expected_synapses(), 0.0);
    }

    /// The sampler against the analytic mean and standard deviation. The connections are
    /// independent Bernoulli draws, so the realised count must sit within a few standard deviations
    /// of `Σ p` — a genuine check, because the expectation is computed by a different loop from the
    /// one that draws.
    ///
    /// Measured on the default column at seed 0x11D: 619 synapses against an expectation of 638.06
    /// with a standard deviation of 23.35, which is 0.82 σ.
    #[test]
    fn the_synapse_count_matches_the_analytic_expectation() {
        for seed in 0..6u64 {
            let spec = LiquidSpec { seed: 0x11D + seed, ..LiquidSpec::maass_column() };
            let l = Liquid::build(&spec).unwrap();
            let mu = l.expected_synapses();
            let sd = l.synapse_sd();
            assert!(sd > 0.0);
            let got = l.net.n_syn as f64;
            assert!(
                (got - mu).abs() < 4.0 * sd,
                "seed {seed}: {got} synapses against {mu} ± {sd}"
            );
        }
    }

    /// Dale's principle as an exact invariant over every synapse in the column: the sign of a weight
    /// is a property of the presynaptic cell and of nothing else.
    #[test]
    fn every_synapse_takes_its_sign_from_its_presynaptic_cell() {
        let l = Liquid::build(&LiquidSpec::maass_column()).unwrap();
        assert_eq!(l.n(), 135);
        assert_eq!(l.n_inhibitory(), 27, "0.2 x 135 is exactly 27");
        let mut seen = 0usize;
        for pre in 0..l.n() {
            for (post, w, delay) in l.net.out_of(pre) {
                assert_ne!(post as usize, pre, "neuron {pre} connects to itself");
                match l.kinds[pre] {
                    Cell::Excitatory => assert!(w > 0.0, "excitatory {pre} has weight {w}"),
                    Cell::Inhibitory => assert!(w < 0.0, "inhibitory {pre} has weight {w}"),
                }
                // 1.5 ms and 0.8 ms at a 0.1 ms tick are 15 and 8 ticks.
                let want = if l.kinds[pre] == Cell::Excitatory
                    && l.kinds[post as usize] == Cell::Excitatory
                {
                    15
                } else {
                    8
                };
                assert_eq!(delay, want, "synapse {pre} -> {post} carried {delay} ticks");
                seen += 1;
            }
        }
        assert_eq!(seen, l.net.n_syn, "the CSR walk missed synapses");
    }

    #[test]
    fn the_same_spec_builds_the_same_liquid() {
        let spec = LiquidSpec::maass_column();
        assert_eq!(Liquid::build(&spec).unwrap(), Liquid::build(&spec).unwrap());
    }

    #[test]
    fn a_malformed_liquid_spec_is_refused_naming_the_field() {
        let base = LiquidSpec::maass_column();
        assert!(matches!(
            Liquid::build(&LiquidSpec { w_ie: 1e-3, ..base }).unwrap_err(),
            ReservoirError::OutOfRange { what: "w_ie", .. }
        ));
        assert!(matches!(
            Liquid::build(&LiquidSpec { lambda: 0.0, ..base }).unwrap_err(),
            ReservoirError::OutOfRange { what: "lambda", .. }
        ));
        assert!(matches!(
            Liquid::build(&LiquidSpec { c_ee: 1.5, ..base }).unwrap_err(),
            ReservoirError::OutOfRange { what: "c_ee", .. }
        ));
        assert!(matches!(
            Liquid::build(&LiquidSpec { nx: 0, ..base }).unwrap_err(),
            ReservoirError::Empty { what: "lattice" }
        ));
    }

    /// The filter's closed form: one spike, summed forever, is the geometric series
    /// `1 / (1 − exp(−dt/tau))` exactly. Pins the decay factor, the increment and the order of the
    /// two operations in one comparison.
    #[test]
    fn one_spike_integrates_to_the_geometric_series_exactly() {
        let (tau, dt) = (30e-3, 1e-4);
        let mut f = SpikeFilter::new(1, tau, dt).unwrap();
        f.step(&[0]);
        assert_eq!(f.state()[0], 1.0, "the spike's own tick was decayed");
        let mut sum = f.state()[0];
        for _ in 0..20_000 {
            f.step(&[]);
            sum += f.state()[0];
        }
        let want = 1.0 / (1.0 - (-dt / tau).exp());
        assert!((sum - want).abs() < 1e-9, "trace integral {sum} against {want}");
        // Ids past the end are ignored rather than panicking, as Net::out_of does.
        f.step(&[99]);
        f.reset();
        assert_eq!(f.state()[0], 0.0);
    }

    /// The liquid, end to end: two different input streams must drive it to different states, the
    /// same stream twice to identical ones, and a bigger column to a larger separation.
    ///
    /// Measured at 45, 135 and 225 neurons: 1.46, 2.80 and 3.86, with `sep(a, a)` exactly zero.
    #[test]
    fn a_liquid_separates_two_input_streams_and_separates_more_when_it_is_larger() {
        let dt = LiquidSpec::maass_column().dt;
        let exc = Lif::default();
        let inh = Lif { t_ref: 1e-3, ..Lif::default() };
        let mut last = 0.0;
        for &nx in &[5usize, 15, 25] {
            let spec = LiquidSpec { nx, ..LiquidSpec::maass_column() };
            let l = Liquid::build(&spec).unwrap();
            let sites = l.input_sites(2, (l.n() / 10).max(2), 99).unwrap();
            assert_eq!(sites.len(), 2);
            for set in &sites {
                for &s in set {
                    assert_eq!(l.kinds[s as usize], Cell::Excitatory, "input hit an inhibitory cell");
                }
            }
            let stream = |ch: usize| -> Vec<Vec<f64>> {
                let mut rng = Rng::new(9 + ch as u64);
                (0..600)
                    .map(|_| {
                        let mut row = vec![0.0; l.n()];
                        if rng.next_f64() < 0.4 {
                            for &s in &sites[ch] {
                                row[s as usize] = 4e-9;
                            }
                        }
                        row
                    })
                    .collect()
            };
            let a = l.respond(exc, inh, dt, 30e-3, &stream(0)).unwrap();
            let b = l.respond(exc, inh, dt, 30e-3, &stream(1)).unwrap();
            let again = l.respond(exc, inh, dt, 30e-3, &stream(0)).unwrap();
            assert_eq!(a.len(), 600);
            let sep = separation(a.last().unwrap(), b.last().unwrap()).unwrap();
            let none = separation(a.last().unwrap(), again.last().unwrap()).unwrap();
            assert_eq!(none, 0.0, "the same input twice gave different states");
            assert!(sep > 0.5, "{} neurons separated the two streams by only {sep}", l.n());
            assert!(sep > last, "separation {sep} at {} neurons did not exceed {last}", l.n());
            last = sep;
        }
    }

    /// The liquid is an ordinary network, so the crate's own simulator and ledger apply to it
    /// unchanged — which is the reason it is built on [`crate::net::Net`] rather than on something
    /// private to this module.
    #[test]
    fn a_liquid_runs_through_the_crates_own_simulator_and_ledger() {
        let spec = LiquidSpec::maass_column();
        let l = Liquid::build(&spec).unwrap();
        let sites = l.input_sites(1, 12, 99).unwrap();
        let mut ext = vec![0.0; l.n()];
        for &s in &sites[0] {
            ext[s as usize] = 6e-9;
        }
        let mut sim = l.to_sim(Lif::default(), Lif::default(), spec.dt, Mode::EventDriven).unwrap();
        let train = sim.run(1_000, &ext);
        assert!(!train.is_empty(), "the liquid was silent under 6 nA");
        assert!(sim.ledger.syn_ops > 0, "no synaptic operation was charged");
        assert_eq!(sim.ledger.neuron_updates_idle, 0, "event-driven updated an idle neuron");
    }
}
