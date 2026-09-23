//! Equilibrium propagation: the gradient of a loss, read off the difference between two
//! relaxations of the same physical network — checked against the gradient it claims to be, with
//! the order of its error measured.
//!
//! # What the mechanism is
//!
//! Take a network whose dynamics descend an energy `E(s; θ)` — symmetric couplings, leaky units —
//! and let it settle with the input clamped: the **free phase**, fixed point `s⁰`. Now pull the
//! output units toward the target with a weak spring, `F = E + β C`, `C = ½ |y − s_out|²`, and let
//! it settle again: the **nudged phase**, fixed point `s^β`. Scellier and Bengio (*Equilibrium
//! propagation: bridging the gap between energy-based models and backpropagation*, Frontiers in
//! Computational Neuroscience 11:24, 2017) prove that
//!
//! `dJ/dθ = lim_{β→0} (1/β) [∂E/∂θ(s^β) − ∂E/∂θ(s⁰)]`, with `J(θ) = C(s⁰(θ))`.
//!
//! For a coupling `W_ij` the bracket is `−(ρ_i ρ_j)^β + (ρ_i ρ_j)⁰`: the change in the product of
//! the two activities the synapse already sees. No backward pass, no derivative of the
//! activation at the synapse, no second set of weights — the same circuit runs twice and each
//! synapse subtracts. The estimate is biased at finite `β`; nudging both ways,
//! `(1/2β)[∂E/∂θ(s^{+β}) − ∂E/∂θ(s^{−β})]`, cancels the first-order bias (Laborieux, Ernoult,
//! Scellier, Bengio, Grollier and Querlioz, *Scaling equilibrium propagation to deep `ConvNets` by
//! drastically reducing its gradient estimator bias*, Frontiers in Neuroscience 15:633674, 2021).
//!
//! # Why it is in a neuromorphic crate
//!
//! It is the learning rule for a substrate that computes by settling — resistor networks, coupled
//! oscillators, analog Hopfield circuits — where a relaxation is free and a backward pass does not
//! exist. It is the sibling of [`crate::predictive`]: there the right limit turned out to be a
//! weakly clamped output, and `β → 0` here is the same limit, reached from the other side.
//!
//! # The closed forms this module is checked against
//!
//! - **The dynamics are the gradient flow of the energy**: [`Network::velocity`] equals `−∂F/∂s`
//!   by central differences of [`Network::total_energy`].
//! - **The theorem.** The estimate is compared with `dJ/dθ` obtained WITHOUT it — by perturbing
//!   each parameter, re-relaxing the free phase and differencing the loss. The one-sided estimate's
//!   error is first order in `β` (tenfold smaller `β`, tenfold smaller error) and the symmetric
//!   estimate's is second order (a hundredfold) — measured between `β = 0.1` and `0.01`. Between
//!   `0.3` and `0.03` the same ratio reads 111: the `β⁴` term, 9% of the `β²` one out there, and
//!   the first version of the test, which looked only there, failed on it.
//! - **It learns**: repeated symmetric updates drive the loss of a small regression task down by
//!   a stated factor, deterministically by seed.
//!
//! # What this module has NOT reproduced
//!
//! - The MNIST results of either paper, or the hard-sigmoid activation of the first: the units
//!   here are `tanh`, because the theorem's error orders need a smooth energy to be measurable.
//! - Any hardware relaxation. The settling here is explicit Euler on the gradient flow, and the
//!   number of steps it takes is a property of this simulation, not of a circuit.
//! - Convergence of the relaxation for large weights. With couplings this small the energy has one
//!   minimum; [`Network::relax`] reports whether it got there and the caller must look.

use core::fmt;

use crate::rng::Rng;

/// The most relaxation steps one call will take: **ten million**. A request for exactly this
/// many is admissible; one past it is refused with [`EquilibriumError::OutOfRange`]. The
/// number is written here as well as in the literal because a cap nobody states is a cap
/// nobody can notice moving.
pub const MAX_STEPS: u64 = 10_000_000;

/// What went wrong, named rather than guessed around.
#[derive(Debug, Clone, PartialEq)]
pub enum EquilibriumError {
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
    /// A relaxation that was still moving when its step budget ran out.
    NotSettled {
        /// Largest unit movement on the last step.
        moved: f64,
        /// Steps taken.
        steps: u64,
    },
}

impl fmt::Display for EquilibriumError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty { what } => write!(f, "{what} is empty"),
            Self::Dimension { what, got, want } => write!(f, "{what} has length {got}, expected {want}"),
            Self::OutOfRange { what, value, low, high } => {
                write!(f, "{what} = {value} is outside [{low}, {high}]")
            }
            Self::NonFinite { what, index } => write!(f, "{what} is not finite at {index}"),
            Self::NotSettled { moved, steps } => write!(f, "still moving by {moved} after {steps} steps"),
        }
    }
}

impl std::error::Error for EquilibriumError {}

fn finite_all(what: &'static str, v: &[f64]) -> Result<(), EquilibriumError> {
    if let Some(i) = v.iter().position(|x| !x.is_finite()) {
        return Err(EquilibriumError::NonFinite { what, index: i });
    }
    Ok(())
}

fn dims(what: &'static str, got: usize, want: usize) -> Result<(), EquilibriumError> {
    if got == want { Ok(()) } else { Err(EquilibriumError::Dimension { what, got, want }) }
}

/// An energy-based network: `n` units `s`, the last `n_out` of them outputs, `m` clamped inputs.
///
/// `E = ½ |s|² − ½ ρ(s)ᵀ W ρ(s) − ρ(s)ᵀ (U x + b)`, `ρ = tanh`, `W` symmetric with a zero diagonal.
#[derive(Debug, Clone, PartialEq)]
pub struct Network {
    /// Units.
    pub n: usize,
    /// Output units: the last `n_out` of the `n`.
    pub n_out: usize,
    /// Inputs.
    pub m: usize,
    /// Couplings, row-major `n × n`, symmetric, zero diagonal.
    pub w: Vec<f64>,
    /// Input weights, row-major `n × m`.
    pub u: Vec<f64>,
    /// Biases, length `n`.
    pub b: Vec<f64>,
}

/// The parameter gradients of one quantity, shaped like the network's parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct Gradient {
    /// With respect to each coupling, row-major `n × n`, symmetric, zero diagonal: entry `(i, j)`
    /// is the derivative with respect to the ONE parameter shared by `W_ij` and `W_ji`.
    pub w: Vec<f64>,
    /// With respect to the input weights, row-major `n × m`.
    pub u: Vec<f64>,
    /// With respect to the biases, length `n`.
    pub b: Vec<f64>,
}

impl Gradient {
    /// The relative mismatch `|self − other| / |other|` over every independent parameter (each
    /// symmetric coupling once). `None` if the shapes differ or `other` is zero.
    #[must_use]
    pub fn relative_mismatch(&self, other: &Self) -> Option<f64> {
        if self.w.len() != other.w.len() || self.u.len() != other.u.len() || self.b.len() != other.b.len() {
            return None;
        }
        let n = self.b.len();
        let (mut diff, mut norm) = (0.0, 0.0);
        for i in 0..n {
            for j in (i + 1)..n {
                let (a, c) = (self.w[i * n + j], other.w[i * n + j]);
                diff += (a - c) * (a - c);
                norm += c * c;
            }
        }
        for (a, c) in self.u.iter().zip(&other.u).chain(self.b.iter().zip(&other.b)) {
            diff += (a - c) * (a - c);
            norm += c * c;
        }
        if norm == 0.0 { None } else { Some((diff / norm).sqrt()) }
    }
}

impl Network {
    /// A network with small random parameters: couplings and input weights uniform in
    /// `±scale/√n` and `±scale/√m`, biases zero. Every unit is coupled to every other.
    ///
    /// # Errors
    ///
    /// [`EquilibriumError::Empty`] for no units, outputs or inputs,
    /// [`EquilibriumError::OutOfRange`] for more outputs than units or a `scale` outside `(0, 1]`
    /// — past that the energy can have more than one minimum and the theorem's fixed point is no
    /// longer a function of the parameters.
    pub fn random(n: usize, n_out: usize, m: usize, scale: f64, rng: &mut Rng) -> Result<Self, EquilibriumError> {
        if n == 0 {
            return Err(EquilibriumError::Empty { what: "units" });
        }
        if n_out == 0 {
            return Err(EquilibriumError::Empty { what: "outputs" });
        }
        if m == 0 {
            return Err(EquilibriumError::Empty { what: "inputs" });
        }
        if n_out > n {
            return Err(EquilibriumError::OutOfRange { what: "n_out", value: n_out as f64, low: 1.0, high: n as f64 });
        }
        if !(scale > 0.0) || !(scale <= 1.0) {
            return Err(EquilibriumError::OutOfRange { what: "scale", value: scale, low: f64::MIN_POSITIVE, high: 1.0 });
        }
        let mut w = vec![0.0; n * n];
        let bound = scale / (n as f64).sqrt();
        for i in 0..n {
            for j in (i + 1)..n {
                let v = bound * (2.0 * rng.next_f64() - 1.0);
                w[i * n + j] = v;
                w[j * n + i] = v;
            }
        }
        let bound_u = scale / (m as f64).sqrt();
        let u = (0..n * m).map(|_| bound_u * (2.0 * rng.next_f64() - 1.0)).collect();
        Ok(Self { n, n_out, m, w, u, b: vec![0.0; n] })
    }

    fn check(&self, s: &[f64], x: &[f64], y: &[f64]) -> Result<(), EquilibriumError> {
        dims("s", s.len(), self.n)?;
        dims("x", x.len(), self.m)?;
        dims("y", y.len(), self.n_out)?;
        finite_all("s", s)?;
        finite_all("x", x)?;
        finite_all("y", y)
    }

    fn drive(&self, x: &[f64]) -> Vec<f64> {
        (0..self.n).map(|i| self.b[i] + (0..self.m).map(|k| self.u[i * self.m + k] * x[k]).sum::<f64>()).collect()
    }

    /// The loss `C = ½ |y − s_out|²` of a state.
    ///
    /// # Errors
    ///
    /// [`EquilibriumError::Dimension`] or [`EquilibriumError::NonFinite`] for a bad `s` or `y`.
    pub fn cost(&self, s: &[f64], y: &[f64]) -> Result<f64, EquilibriumError> {
        dims("s", s.len(), self.n)?;
        dims("y", y.len(), self.n_out)?;
        finite_all("s", s)?;
        finite_all("y", y)?;
        let first = self.n - self.n_out;
        Ok(0.5 * y.iter().enumerate().map(|(k, t)| (t - s[first + k]) * (t - s[first + k])).sum::<f64>())
    }

    /// The total energy `F = E + β C`.
    ///
    /// # Errors
    ///
    /// [`EquilibriumError::Dimension`] or [`EquilibriumError::NonFinite`] for a bad argument.
    pub fn total_energy(&self, s: &[f64], x: &[f64], y: &[f64], beta: f64) -> Result<f64, EquilibriumError> {
        self.check(s, x, y)?;
        if !beta.is_finite() {
            return Err(EquilibriumError::NonFinite { what: "beta", index: 0 });
        }
        let rho: Vec<f64> = s.iter().map(|v| v.tanh()).collect();
        let drive = self.drive(x);
        let mut e = 0.0;
        for i in 0..self.n {
            e += 0.5 * s[i] * s[i] - rho[i] * drive[i];
            for j in (i + 1)..self.n {
                e -= self.w[i * self.n + j] * rho[i] * rho[j];
            }
        }
        Ok(e + beta * self.cost(s, y)?)
    }

    /// The gradient flow `ds/dt = −∂F/∂s`:
    /// `−s_i + ρ′(s_i) (Σ_j W_ij ρ(s_j) + (Ux + b)_i) + β (y − s)_i` on the outputs.
    ///
    /// # Errors
    ///
    /// As [`Network::total_energy`].
    pub fn velocity(&self, s: &[f64], x: &[f64], y: &[f64], beta: f64) -> Result<Vec<f64>, EquilibriumError> {
        self.check(s, x, y)?;
        if !beta.is_finite() {
            return Err(EquilibriumError::NonFinite { what: "beta", index: 0 });
        }
        let rho: Vec<f64> = s.iter().map(|v| v.tanh()).collect();
        let drive = self.drive(x);
        let first = self.n - self.n_out;
        Ok((0..self.n)
            .map(|i| {
                let field: f64 = (0..self.n).map(|j| self.w[i * self.n + j] * rho[j]).sum::<f64>() + drive[i];
                let mut v = -s[i] + (1.0 - rho[i] * rho[i]) * field;
                if i >= first {
                    v += beta * (y[i - first] - s[i]);
                }
                v
            })
            .collect())
    }

    /// Settle from `start` by explicit Euler steps of `dt` until no unit moves by more than `tol`
    /// in a step, or refuse.
    ///
    /// # Errors
    ///
    /// As [`Network::velocity`]; [`EquilibriumError::OutOfRange`] for a `dt` outside `(0, 1]`, a
    /// negative `tol`, or more than [`MAX_STEPS`]; [`EquilibriumError::NotSettled`] if the budget
    /// runs out first.
    pub fn relax(&self, start: &[f64], x: &[f64], y: &[f64], beta: f64, dt: f64, tol: f64, max_steps: u64) -> Result<Vec<f64>, EquilibriumError> {
        if !(dt > 0.0) || !(dt <= 1.0) {
            return Err(EquilibriumError::OutOfRange { what: "dt", value: dt, low: f64::MIN_POSITIVE, high: 1.0 });
        }
        if !(tol >= 0.0) {
            return Err(EquilibriumError::OutOfRange { what: "tol", value: tol, low: 0.0, high: f64::INFINITY });
        }
        if max_steps > MAX_STEPS {
            return Err(EquilibriumError::OutOfRange { what: "max_steps", value: max_steps as f64, low: 0.0, high: MAX_STEPS as f64 });
        }
        let mut s = start.to_vec();
        let mut moved = f64::INFINITY;
        for _ in 0..max_steps {
            let v = self.velocity(&s, x, y, beta)?;
            moved = 0.0;
            for (si, vi) in s.iter_mut().zip(&v) {
                *si += dt * vi;
                moved = moved.max((dt * vi).abs());
            }
            if moved <= tol {
                return Ok(s);
            }
        }
        Err(EquilibriumError::NotSettled { moved, steps: max_steps })
    }

    /// `∂E/∂θ` at a state: `−ρ_i ρ_j` for a coupling, `−ρ_i x_k` for an input weight, `−ρ_i` for a
    /// bias — what each synapse can measure locally.
    ///
    /// # Errors
    ///
    /// [`EquilibriumError::Dimension`] or [`EquilibriumError::NonFinite`] for a bad `s` or `x`.
    pub fn energy_gradient(&self, s: &[f64], x: &[f64]) -> Result<Gradient, EquilibriumError> {
        dims("s", s.len(), self.n)?;
        dims("x", x.len(), self.m)?;
        finite_all("s", s)?;
        finite_all("x", x)?;
        let rho: Vec<f64> = s.iter().map(|v| v.tanh()).collect();
        let mut w = vec![0.0; self.n * self.n];
        for i in 0..self.n {
            for j in 0..self.n {
                if i != j {
                    w[i * self.n + j] = -rho[i] * rho[j];
                }
            }
        }
        let u = (0..self.n * self.m).map(|q| -rho[q / self.m] * x[q % self.m]).collect();
        Ok(Gradient { w, u, b: rho.iter().map(|r| -r).collect() })
    }

    /// The equilibrium-propagation estimate of `dJ/dθ`. One-sided:
    /// `(1/β)[∂E/∂θ(s^β) − ∂E/∂θ(s⁰)]`; symmetric: `(1/2β)[∂E/∂θ(s^{+β}) − ∂E/∂θ(s^{−β})]`.
    /// Each nudged phase starts from the free fixed point, as the hardware would.
    ///
    /// # Errors
    ///
    /// As [`Network::relax`], plus [`EquilibriumError::OutOfRange`] for a non-positive `beta`.
    pub fn estimate(&self, x: &[f64], y: &[f64], beta: f64, symmetric: bool, dt: f64, tol: f64, max_steps: u64) -> Result<Gradient, EquilibriumError> {
        if !(beta > 0.0) || !beta.is_finite() {
            return Err(EquilibriumError::OutOfRange { what: "beta", value: beta, low: f64::MIN_POSITIVE, high: f64::INFINITY });
        }
        let free = self.relax(&vec![0.0; self.n], x, y, 0.0, dt, tol, max_steps)?;
        let plus = self.energy_gradient(&self.relax(&free, x, y, beta, dt, tol, max_steps)?, x)?;
        let (minus, span) = if symmetric {
            (self.energy_gradient(&self.relax(&free, x, y, -beta, dt, tol, max_steps)?, x)?, 2.0 * beta)
        } else {
            (self.energy_gradient(&free, x)?, beta)
        };
        let sub = |a: &[f64], c: &[f64]| a.iter().zip(c).map(|(p, q)| (p - q) / span).collect::<Vec<f64>>();
        Ok(Gradient { w: sub(&plus.w, &minus.w), u: sub(&plus.u, &minus.u), b: sub(&plus.b, &minus.b) })
    }

    /// The loss at the free fixed point, `J(θ) = C(s⁰(θ))`.
    ///
    /// # Errors
    ///
    /// As [`Network::relax`].
    pub fn free_loss(&self, x: &[f64], y: &[f64], dt: f64, tol: f64, max_steps: u64) -> Result<f64, EquilibriumError> {
        let free = self.relax(&vec![0.0; self.n], x, y, 0.0, dt, tol, max_steps)?;
        self.cost(&free, y)
    }

    /// Step every parameter against a gradient: `θ ← θ − rate · g`, keeping `W` symmetric.
    ///
    /// # Errors
    ///
    /// [`EquilibriumError::Dimension`] for a gradient of the wrong shape,
    /// [`EquilibriumError::NonFinite`] for a non-finite rate.
    pub fn descend(&mut self, g: &Gradient, rate: f64) -> Result<(), EquilibriumError> {
        dims("gradient w", g.w.len(), self.w.len())?;
        dims("gradient u", g.u.len(), self.u.len())?;
        dims("gradient b", g.b.len(), self.b.len())?;
        if !rate.is_finite() {
            return Err(EquilibriumError::NonFinite { what: "rate", index: 0 });
        }
        for (p, d) in self.w.iter_mut().zip(&g.w).chain(self.u.iter_mut().zip(&g.u)).chain(self.b.iter_mut().zip(&g.b)) {
            *p -= rate * d;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DT: f64 = 0.5;
    const TOL: f64 = 1e-15;
    const STEPS: u64 = 20_000;

    fn net() -> Network {
        let mut n = Network::random(7, 2, 3, 0.8, &mut Rng::new(19)).unwrap();
        // Biases that are not zero, so their gradient is not tested at a special point.
        for (i, b) in n.b.iter_mut().enumerate() {
            *b = 0.1 * (i as f64 - 3.0);
        }
        n
    }

    /// `dJ/dθ` without the theorem: perturb a parameter, re-relax the free phase, difference the
    /// loss. A symmetric coupling is ONE parameter, perturbed on both sides of the diagonal.
    fn referee(net: &Network, x: &[f64], y: &[f64]) -> Gradient {
        let h = 1e-5;
        let loss = |n: &Network| n.free_loss(x, y, DT, TOL, STEPS).unwrap();
        let nn = net.n;
        let mut w = vec![0.0; nn * nn];
        for i in 0..nn {
            for j in (i + 1)..nn {
                let (mut up, mut down) = (net.clone(), net.clone());
                for (a, c) in [(i, j), (j, i)] {
                    up.w[a * nn + c] += h;
                    down.w[a * nn + c] -= h;
                }
                let g = (loss(&up) - loss(&down)) / (2.0 * h);
                w[i * nn + j] = g;
                w[j * nn + i] = g;
            }
        }
        let u = (0..net.u.len())
            .map(|q| {
                let (mut up, mut down) = (net.clone(), net.clone());
                up.u[q] += h;
                down.u[q] -= h;
                (loss(&up) - loss(&down)) / (2.0 * h)
            })
            .collect();
        let b = (0..nn)
            .map(|q| {
                let (mut up, mut down) = (net.clone(), net.clone());
                up.b[q] += h;
                down.b[q] -= h;
                (loss(&up) - loss(&down)) / (2.0 * h)
            })
            .collect();
        Gradient { w, u, b }
    }

    #[test]
    fn the_dynamics_are_the_gradient_flow_of_the_total_energy() {
        let n = net();
        let s = [0.3, -0.8, 0.5, 0.1, -0.2, 0.9, -0.4];
        let (x, y) = ([0.7, -0.5, 0.2], [0.4, -0.6]);
        for beta in [0.0, 0.3, -0.3] {
            let v = n.velocity(&s, &x, &y, beta).unwrap();
            for i in 0..7 {
                let h = 1e-6;
                let (mut up, mut down) = (s, s);
                up[i] += h;
                down[i] -= h;
                let slope = (n.total_energy(&up, &x, &y, beta).unwrap() - n.total_energy(&down, &x, &y, beta).unwrap()) / (2.0 * h);
                assert!((v[i] + slope).abs() < 1e-8, "β = {beta}, unit {i}: velocity {}, −∂F/∂s {}", v[i], -slope);
            }
        }
        // The spring acts on the outputs only, and with the sign that pulls toward the target.
        let (free, nudged) = (n.velocity(&s, &x, &y, 0.0).unwrap(), n.velocity(&s, &x, &y, 0.5).unwrap());
        assert_eq!(free[..5], nudged[..5]);
        assert!((nudged[5] - free[5] - 0.5 * (0.4 - 0.9)).abs() < 1e-15);
        assert!((nudged[6] - free[6] - 0.5 * (-0.6 + 0.4)).abs() < 1e-15);
        assert_eq!(n.cost(&s, &y).unwrap(), 0.5 * (0.25 + 0.04));
    }

    #[test]
    fn the_estimate_is_the_gradient_at_first_order_and_the_symmetric_one_at_second() {
        let n = net();
        let (x, y) = ([0.7, -0.5, 0.2], [0.4, -0.6]);
        let truth = referee(&n, &x, &y);
        assert!(truth.b.iter().all(|g| g.abs() > 1e-6), "a zero gradient would match any estimate: {:?}", truth.b);
        let miss = |beta: f64, symmetric: bool| n.estimate(&x, &y, beta, symmetric, DT, TOL, STEPS).unwrap().relative_mismatch(&truth).unwrap();
        // One-sided: the bias is c·β + O(β²).
        let (coarse, fine) = (miss(1e-1, false), miss(1e-2, false));
        assert!((coarse / fine - 10.0).abs() < 1.0, "one-sided: {coarse} → {fine}, ×{}", coarse / fine);
        // Symmetric: the first-order term cancels and the bias is c·β² + d·β⁴. Between β = 0.1 and
        // 0.01 the fourth-order term is a relative β² = 1% of the second, so the ratio is 100 to
        // within a few — at β = 0.3 it is 9% and the ratio measured 111, which is the same law
        // read too far from the limit, not a different one.
        let (coarse2, fine2) = (miss(1e-1, true), miss(1e-2, true));
        assert!((coarse2 / fine2 - 100.0).abs() < 5.0, "symmetric: {coarse2} → {fine2}, ×{}", coarse2 / fine2);
        let far = miss(3e-1, true) / miss(3e-2, true);
        assert!(far > 105.0 && far < 120.0, "and further out the β⁴ term shows: ×{far}");
        // And at the same β the symmetric estimate is the better one.
        assert!(miss(1e-1, true) < 0.1 * coarse);
    }

    #[test]
    fn repeated_updates_learn_a_small_regression() {
        let mut rng = Rng::new(5);
        let mut n = Network::random(8, 1, 2, 0.5, &mut rng).unwrap();
        let data: Vec<([f64; 2], [f64; 1])> = (0..6)
            .map(|_| {
                let x = [2.0 * rng.next_f64() - 1.0, 2.0 * rng.next_f64() - 1.0];
                (x, [0.5 * x[0] - 0.3 * x[1]])
            })
            .collect();
        let total = |n: &Network| data.iter().map(|(x, y)| n.free_loss(x, y, DT, 1e-12, STEPS).unwrap()).sum::<f64>();
        let before = total(&n);
        for _ in 0..300 {
            for (x, y) in &data {
                let g = n.estimate(x, y, 0.05, true, DT, 1e-12, STEPS).unwrap();
                n.descend(&g, 0.5).unwrap();
            }
        }
        let after = total(&n);
        assert!(after < 0.01 * before, "the loss went from {before} to {after}");
        // The couplings are still symmetric with a zero diagonal: the update preserved the energy's
        // structure, which is what makes the next relaxation a descent.
        for i in 0..8 {
            assert_eq!(n.w[i * 8 + i], 0.0);
            for j in 0..8 {
                assert_eq!(n.w[i * 8 + j], n.w[j * 8 + i]);
            }
        }
    }

    #[test]
    fn a_relaxation_that_has_not_settled_says_so() {
        let n = net();
        let (x, y) = ([0.7, -0.5, 0.2], [0.4, -0.6]);
        let err = n.relax(&[0.0; 7], &x, &y, 0.0, DT, TOL, 3).unwrap_err();
        assert!(matches!(err, EquilibriumError::NotSettled { steps: 3, moved } if moved > 1e-6));
        let s = n.relax(&[0.0; 7], &x, &y, 0.0, DT, TOL, STEPS).unwrap();
        let v = n.velocity(&s, &x, &y, 0.0).unwrap();
        assert!(v.iter().all(|vi| vi.abs() <= 2.0 * TOL / DT), "a settled state still has velocity {v:?}");
        // The fixed point does not depend on where the relaxation started.
        let other = n.relax(&[1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0], &x, &y, 0.0, DT, TOL, STEPS).unwrap();
        for (a, c) in s.iter().zip(&other) {
            assert!((a - c).abs() < 1e-13);
        }
    }

    #[test]
    fn bad_arguments_are_refused() {
        let mut rng = Rng::new(1);
        assert!(matches!(Network::random(0, 1, 1, 0.5, &mut rng), Err(EquilibriumError::Empty { what: "units" })));
        assert!(matches!(Network::random(3, 0, 1, 0.5, &mut rng), Err(EquilibriumError::Empty { what: "outputs" })));
        assert!(matches!(Network::random(3, 1, 0, 0.5, &mut rng), Err(EquilibriumError::Empty { what: "inputs" })));
        assert!(matches!(Network::random(3, 4, 1, 0.5, &mut rng), Err(EquilibriumError::OutOfRange { what: "n_out", .. })));
        assert!(matches!(Network::random(3, 1, 1, 0.0, &mut rng), Err(EquilibriumError::OutOfRange { what: "scale", .. })));
        assert!(matches!(Network::random(3, 1, 1, 1.5, &mut rng), Err(EquilibriumError::OutOfRange { what: "scale", .. })));
        let n = Network::random(4, 1, 2, 1.0, &mut rng).unwrap();
        assert!(n.w.iter().all(|v| v.abs() <= 0.5) && n.u.iter().all(|v| v.abs() <= 1.0 / 2.0f64.sqrt()));
        assert!(n.w.iter().any(|v| v.abs() > 0.05), "the couplings are all but zero");
        let (s, x, y) = ([0.0; 4], [0.0; 2], [0.0; 1]);
        assert!(matches!(n.velocity(&s[..3], &x, &y, 0.0), Err(EquilibriumError::Dimension { what: "s", .. })));
        assert!(matches!(n.velocity(&s, &x[..1], &y, 0.0), Err(EquilibriumError::Dimension { what: "x", .. })));
        assert!(matches!(n.velocity(&s, &x, &[], 0.0), Err(EquilibriumError::Dimension { what: "y", .. })));
        assert!(matches!(n.velocity(&s, &x, &y, f64::NAN), Err(EquilibriumError::NonFinite { what: "beta", .. })));
        assert!(matches!(n.total_energy(&s, &x, &y, f64::INFINITY), Err(EquilibriumError::NonFinite { what: "beta", .. })));
        assert!(matches!(n.velocity(&[f64::NAN, 0.0, 0.0, 0.0], &x, &y, 0.0), Err(EquilibriumError::NonFinite { what: "s", .. })));
        assert!(matches!(n.relax(&s, &x, &y, 0.0, 0.0, 1e-12, 10), Err(EquilibriumError::OutOfRange { what: "dt", .. })));
        assert!(matches!(n.relax(&s, &x, &y, 0.0, 1.5, 1e-12, 10), Err(EquilibriumError::OutOfRange { what: "dt", .. })));
        assert!(matches!(n.relax(&s, &x, &y, 0.0, 0.5, -1.0, 10), Err(EquilibriumError::OutOfRange { what: "tol", .. })));
        assert!(matches!(n.relax(&s, &x, &y, 0.0, 0.5, 1e-12, MAX_STEPS + 1), Err(EquilibriumError::OutOfRange { what: "max_steps", .. })));
        assert!(matches!(n.estimate(&x, &y, 0.0, true, 0.5, 1e-12, 100), Err(EquilibriumError::OutOfRange { what: "beta", .. })));
        assert!(matches!(n.energy_gradient(&s, &x[..1]), Err(EquilibriumError::Dimension { what: "x", .. })));
        let g = n.energy_gradient(&[0.5, 0.0, 0.0, -0.5], &[2.0, 3.0]).unwrap();
        let t = 0.5f64.tanh();
        assert_eq!(g.b, vec![-t, 0.0, 0.0, t]);
        assert_eq!(g.u[..2], [-t * 2.0, -t * 3.0]);
        assert_eq!((g.w[3], g.w[12], g.w[0]), (t * t, t * t, 0.0));
        let mut m = n.clone();
        let short = Gradient { w: vec![0.0; 3], u: g.u.clone(), b: g.b.clone() };
        assert!(matches!(m.descend(&short, 0.1), Err(EquilibriumError::Dimension { what: "gradient w", .. })));
        assert!(matches!(m.descend(&g, f64::NAN), Err(EquilibriumError::NonFinite { what: "rate", .. })));
        m.descend(&g, 0.1).unwrap();
        assert!((m.b[0] - (n.b[0] + 0.1 * t)).abs() < 1e-15);
        assert_eq!(g.relative_mismatch(&short), None);
        let zero = Gradient { w: vec![0.0; 16], u: vec![0.0; 8], b: vec![0.0; 4] };
        assert_eq!(g.relative_mismatch(&zero), None);
        assert_eq!(zero.relative_mismatch(&g), Some(1.0));
    }

    /// [`Network::estimate`]'s nudged phase starts from the FREE FIXED POINT, as the hardware
    /// would, and restarting it from the origin is not a detail of the relaxation but a different
    /// answer. Two things are pinned here. First the identity: the estimate is exactly
    /// `(dE/dtheta(s^beta) - dE/dtheta(s^0)) / beta` with `s^beta` relaxed FROM `s^0`, recomputed
    /// here out of the same public calls in the same order, so it is an equality and not a
    /// tolerance. Second the size of the difference: this network is bistable, so the two starts
    /// do not merely land on two neighbouring floats, they land in opposite wells.
    ///
    /// MEASURED with `w_01 = w_10 = 5`, `b = (0.05, 0)`, `x = (0)`, `y = (-1)`, `beta = 0.3`,
    /// `dt = 0.1`, `tol = 1e-12`: the free phase settles at `(1.2317852546420964,
    /// 1.2276313692648493)` in 100 steps and the nudged phase started there settles at
    /// `(1.2002315072202903, 1.0384855022129875)`, giving `g.b = (0.0312, 0.2153)`. Started from
    /// the origin instead, the spring's `-0.3` on the output unit's velocity carries the pair into
    /// the NEGATIVE well, `(-1.2191621872510363, -1.2058895935460854)`, and `g.b = (5.608, 5.591)`
    /// -- a gradient of the opposite sign and eighteen times the size, not a last-bit difference.
    ///
    /// Why the suite could not see it: every relaxation fixture in this module runs on `net()`,
    /// `Network::random(7, 2, 3, 0.8)`, whose couplings are small enough that the energy has a
    /// single minimum -- the module doc says so in as many words. The start-independence test
    /// compares two starts at `beta = 0.0` only, for one `(x, y)`, to a tolerance of `1e-13`; what
    /// it records is that two starts give two DIFFERENT f64 states, which `estimate` then divides
    /// by a `beta` it will accept as small as `1e-13`. Every field of [`Network`] is public and
    /// nothing bounds `w`, so "at these couplings" was never a statement about the code.
    #[test]
    fn the_nudged_phase_starts_from_the_free_fixed_point_and_a_bistable_network_says_so() {
        const DT: f64 = 0.1;
        const TOL: f64 = 1e-12;
        const STEPS: u64 = 200_000;
        let beta = 0.3;
        let n = Network {
            n: 2,
            n_out: 1,
            m: 1,
            w: vec![0.0, 5.0, 5.0, 0.0],
            u: vec![0.0, 0.0],
            b: vec![0.05, 0.0],
        };
        let (x, y) = ([0.0], [-1.0]);
        let origin = vec![0.0; n.n];
        let free = n.relax(&origin, &x, &y, 0.0, DT, TOL, STEPS).expect("the free phase settles");
        let from_free = n.relax(&free, &x, &y, beta, DT, TOL, STEPS).expect("the nudged phase settles");
        let from_origin = n.relax(&origin, &x, &y, beta, DT, TOL, STEPS).expect("the nudged phase settles");
        // The fixture is bistable, and that is what makes this test about physics rather than
        // about the last bit of a relaxation.
        assert!(free[0] > 1.0 && free[1] > 1.0, "the free phase left the positive well: {free:?}");
        assert!(from_free[0] > 1.0, "the nudged phase left the positive well: {from_free:?}");
        assert!(from_origin[0] < -1.0, "the fixture is no longer bistable: {from_origin:?}");
        // The identity. Same calls, same order, so equality.
        let plus = n.energy_gradient(&from_free, &x).expect("a finite state");
        let minus = n.energy_gradient(&free, &x).expect("a finite state");
        let want: Vec<f64> = plus.b.iter().zip(&minus.b).map(|(p, q)| (p - q) / beta).collect();
        let got = n.estimate(&x, &y, beta, false, DT, TOL, STEPS).expect("both phases settle");
        assert_eq!(got.b, want, "the nudged phase did not start at the free fixed point");
        // And the size of it: the origin-started estimate is in the other well and the other sign.
        let wrong = n.energy_gradient(&from_origin, &x).expect("a finite state");
        let other: Vec<f64> = wrong.b.iter().zip(&minus.b).map(|(p, q)| (p - q) / beta).collect();
        assert!(got.b[1] < 1.0 && other[1] > 5.0, "measured {} against {}", got.b[1], other[1]);
    }

    /// A non-finite entry is refused WHEREVER it sits and WHATEVER kind it is, and the error names
    /// the slot it sits in. Three things are pinned: an infinity is non-finite (not only a `NaN`);
    /// the reported `index` is the offending position, not a constant zero; and the TARGET is
    /// checked, the private `Network::check` being the only place that happens for
    /// [`Network::velocity`] and [`Network::total_energy`].
    ///
    /// Why the suite could not see it: `bad_arguments_are_refused` passes exactly one non-finite
    /// argument, `[f64::NAN, 0.0, 0.0, 0.0]`, and reads it with a `matches!` pattern that names
    /// the variant and the `what` field and discards the index. A `NaN` in slot zero cannot
    /// distinguish `!x.is_finite()` from `x.is_nan()`, and cannot distinguish `index: i` from
    /// `index: 0`; and no test anywhere in the module passed a non-finite `y`, so the target's
    /// check was never executed.
    #[test]
    fn a_non_finite_entry_is_refused_by_kind_and_reported_at_the_slot_it_sits_in() {
        let n = net();
        let (s, x, y) = ([0.0_f64; 7], [0.0_f64; 3], [0.0_f64; 2]);
        for bad in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
            let mut state = s;
            state[4] = bad;
            assert_eq!(n.velocity(&state, &x, &y, 0.0).unwrap_err(), EquilibriumError::NonFinite { what: "s", index: 4 });
            assert_eq!(n.total_energy(&state, &x, &y, 0.0).unwrap_err(), EquilibriumError::NonFinite { what: "s", index: 4 });
            assert_eq!(n.cost(&state, &y).unwrap_err(), EquilibriumError::NonFinite { what: "s", index: 4 });
            assert_eq!(n.energy_gradient(&state, &x).unwrap_err(), EquilibriumError::NonFinite { what: "s", index: 4 });
            let mut input = x;
            input[2] = bad;
            assert_eq!(n.velocity(&s, &input, &y, 0.0).unwrap_err(), EquilibriumError::NonFinite { what: "x", index: 2 });
            assert_eq!(n.energy_gradient(&s, &input).unwrap_err(), EquilibriumError::NonFinite { what: "x", index: 2 });
            // The target, which reaches the energy through the spring and the dynamics through it.
            let mut target = y;
            target[1] = bad;
            assert_eq!(n.velocity(&s, &x, &target, 0.0).unwrap_err(), EquilibriumError::NonFinite { what: "y", index: 1 });
            assert_eq!(n.total_energy(&s, &x, &target, 0.0).unwrap_err(), EquilibriumError::NonFinite { what: "y", index: 1 });
            assert_eq!(n.cost(&s, &target).unwrap_err(), EquilibriumError::NonFinite { what: "y", index: 1 });
            assert_eq!(n.relax(&s, &x, &target, 0.0, DT, TOL, 10).unwrap_err(), EquilibriumError::NonFinite { what: "y", index: 1 });
        }
        // And the message says which quantity and where, in that order.
        let mut state = s;
        state[4] = f64::INFINITY;
        assert_eq!(n.velocity(&state, &x, &y, 0.0).unwrap_err().to_string(), "s is not finite at 4");
    }

    /// A length that does not match is refused whether it is SHORT or LONG, and the error carries
    /// the supplied length as `got` and the required one as `want`, in that order, in the struct
    /// and again in the message. Covers the three arrays the private `Network::check` reads, both
    /// arrays of [`Network::energy_gradient`] and all three gradients of [`Network::descend`].
    ///
    /// Why the suite could not see it: every dimension case in `bad_arguments_are_refused` is a
    /// slice TOO SHORT (`&s[..3]`, `&x[..1]`, `&[]`, `vec![0.0; 3]`), so `got == want` and
    /// `got >= want` agree on all of them; and every one is read with `matches!(.., Dimension {
    /// what, .. })`, which discards the two numbers, so neither the struct's field order nor the
    /// `Display` order was ever read. `descend`'s bias check compared `g.b.len()` against the
    /// network's, and a mutant comparing it against itself passes every existing assertion because
    /// the only gradient the suite hands `descend` with a wrong shape is wrong in `w`.
    #[test]
    fn a_dimension_mismatch_is_refused_in_both_directions_and_names_the_two_lengths_in_order() {
        let n = net();
        let (s, x, y) = ([0.0_f64; 7], [0.0_f64; 3], [0.0_f64; 2]);
        assert_eq!(n.velocity(&[0.0; 8], &x, &y, 0.0).unwrap_err(), EquilibriumError::Dimension { what: "s", got: 8, want: 7 });
        assert_eq!(n.velocity(&[0.0; 6], &x, &y, 0.0).unwrap_err(), EquilibriumError::Dimension { what: "s", got: 6, want: 7 });
        assert_eq!(n.velocity(&s, &[0.0; 4], &y, 0.0).unwrap_err(), EquilibriumError::Dimension { what: "x", got: 4, want: 3 });
        assert_eq!(n.velocity(&s, &x, &[0.0; 3], 0.0).unwrap_err(), EquilibriumError::Dimension { what: "y", got: 3, want: 2 });
        assert_eq!(n.total_energy(&[0.0; 9], &x, &y, 0.0).unwrap_err(), EquilibriumError::Dimension { what: "s", got: 9, want: 7 });
        assert_eq!(n.cost(&[0.0; 9], &y).unwrap_err(), EquilibriumError::Dimension { what: "s", got: 9, want: 7 });
        assert_eq!(n.cost(&s, &[0.0; 7]).unwrap_err(), EquilibriumError::Dimension { what: "y", got: 7, want: 2 });
        assert_eq!(n.energy_gradient(&s, &[0.0; 5]).unwrap_err(), EquilibriumError::Dimension { what: "x", got: 5, want: 3 });
        assert_eq!(n.relax(&[0.0; 8], &x, &y, 0.0, DT, TOL, 10).unwrap_err(), EquilibriumError::Dimension { what: "s", got: 8, want: 7 });
        // The message reads supplied-then-required, which is the only order that tells a caller
        // what to change.
        assert_eq!(n.velocity(&[0.0; 8], &x, &y, 0.0).unwrap_err().to_string(), "s has length 8, expected 7");
        assert_eq!(n.velocity(&[0.0; 6], &x, &y, 0.0).unwrap_err().to_string(), "s has length 6, expected 7");
        // And a gradient of the wrong shape, one array at a time, long as well as short.
        let g = n.energy_gradient(&s, &x).unwrap();
        let mut m = n.clone();
        for (len, want) in [(50_usize, 49_usize), (48, 49)] {
            let bad = Gradient { w: vec![0.0; len], u: g.u.clone(), b: g.b.clone() };
            assert_eq!(m.descend(&bad, 0.1).unwrap_err(), EquilibriumError::Dimension { what: "gradient w", got: len, want });
        }
        for (len, want) in [(22_usize, 21_usize), (20, 21)] {
            let bad = Gradient { w: g.w.clone(), u: vec![0.0; len], b: g.b.clone() };
            assert_eq!(m.descend(&bad, 0.1).unwrap_err(), EquilibriumError::Dimension { what: "gradient u", got: len, want });
        }
        for (len, want) in [(8_usize, 7_usize), (6, 7)] {
            let bad = Gradient { w: g.w.clone(), u: g.u.clone(), b: vec![0.0; len] };
            assert_eq!(m.descend(&bad, 0.1).unwrap_err(), EquilibriumError::Dimension { what: "gradient b", got: len, want });
        }
        assert_eq!(m, n, "a refused descend wrote a parameter anyway");
    }

    /// The relative mismatch over a fixture small enough to do by hand: each INDEPENDENT
    /// parameter enters exactly once — a symmetric coupling is one parameter, not two — the biases
    /// enter at all, and a bias array of the wrong length is a shape difference rather than a
    /// prefix to truncate and compare.
    ///
    /// The arithmetic, in the order the function accumulates it. The reference holds one
    /// independent coupling at `1`, one input weight at `2` and one bias at `3`, so its squared
    /// norm is `1 + 4 + 9 + 0 = 14`. Moving any ONE of those three by three gives a squared
    /// difference of `9` and an answer of `√(9/14)` — the SAME number whichever of the three
    /// moved, which is what "each independent parameter once" means, and which a doubled coupling
    /// term or a dropped bias term breaks. Every quantity is a small integer exactly represented
    /// in binary, so these are `assert_eq!` on the f64 and not tolerances.
    ///
    /// Why the suite could not see it: the only other call that reads the NUMBER is `miss()`
    /// inside the order-of-error test, which divides one mismatch by another and asks for a RATIO
    /// of about 10 or 100 — counting every coupling twice multiplies the coupling term of both
    /// `diff` and `norm` by two and leaves that ratio very nearly unchanged, and dropping the
    /// biases removes one term from both, likewise. The three calls in `bad_arguments_are_refused`
    /// read only the degenerate answers: `None` for a `w` of the wrong length, `None` for a zero
    /// reference, and `Some(1.0)` for a zero estimate, which is `1.0` however the terms are
    /// weighted.
    #[test]
    fn the_mismatch_counts_each_symmetric_coupling_once_and_the_biases_at_all() {
        // Reference: one independent coupling at 1, one input weight at 2, one bias at 3, the
        // other bias at 0. Its squared norm is 1 + 4 + 9 + 0 = 14, whichever way it is counted.
        let base = Gradient { w: vec![0.0, 1.0, 1.0, 0.0], u: vec![2.0], b: vec![3.0, 0.0] };
        assert_eq!(base.relative_mismatch(&base), Some(0.0));
        // ONE parameter off by three, three times over: the coupling, the input weight, the bias.
        // The three answers must be the SAME number, which is what "each independent parameter
        // once" means; counting the symmetric coupling twice weighs the first of them differently
        // from the other two, and dropping the biases removes the third from both sums.
        let moved_w = Gradient { w: vec![0.0, 4.0, 4.0, 0.0], u: vec![2.0], b: vec![3.0, 0.0] };
        let moved_u = Gradient { w: vec![0.0, 1.0, 1.0, 0.0], u: vec![5.0], b: vec![3.0, 0.0] };
        let moved_b = Gradient { w: vec![0.0, 1.0, 1.0, 0.0], u: vec![2.0], b: vec![6.0, 0.0] };
        let off_by_three = Some((9.0_f64 / 14.0).sqrt());
        assert_eq!(moved_w.relative_mismatch(&base), off_by_three);
        assert_eq!(moved_u.relative_mismatch(&base), off_by_three);
        assert_eq!(moved_b.relative_mismatch(&base), off_by_three);
        // All three at once: diff = 9 + 4 + 9 = 22 against the same norm of 14.
        let all_three = Gradient { w: vec![0.0, 4.0, 4.0, 0.0], u: vec![0.0], b: vec![0.0, 0.0] };
        assert_eq!(all_three.relative_mismatch(&base), Some((22.0_f64 / 14.0).sqrt()));
        // An estimate of zero misses the reference by exactly the whole of it.
        let zero = Gradient { w: vec![0.0; 4], u: vec![0.0], b: vec![0.0, 0.0] };
        assert_eq!(zero.relative_mismatch(&base), Some(1.0));
        // A bias array of the wrong length is a shape difference, not a prefix to compare.
        let short_b = Gradient { w: vec![0.0; 4], u: vec![2.0], b: vec![3.0] };
        let long_b = Gradient { w: vec![0.0; 4], u: vec![2.0], b: vec![3.0, 0.0, 0.0] };
        assert_eq!(base.relative_mismatch(&short_b), None);
        assert_eq!(base.relative_mismatch(&long_b), None);
        assert_eq!(short_b.relative_mismatch(&base), None);
    }

    /// [`Network::random`] admits a network whose every unit is an output, starts its biases at
    /// exactly zero, and refuses a `NaN` scale.
    ///
    /// `n_out == n` is the degenerate but legal case the doc describes as "more outputs than
    /// units" being the refusal; the zero biases are what make a fresh network's free fixed point
    /// the origin when the input is zero, which the velocity assertion below states as physics
    /// rather than as a field comparison.
    ///
    /// Why the suite could not see it: `bad_arguments_are_refused` tests `random(3, 4, 1, ..)`,
    /// which is `n_out > n` and stays refused under `n_out >= n`, and never builds the boundary
    /// case; it tests `scale` at `0.0` and `1.5`, both of which the arithmetic comparison
    /// `scale <= 0.0 || scale > 1.0` also rejects, while `NaN` passes it and poisons every
    /// coupling; and the only other fixture, `net()`, OVERWRITES all seven biases on the line
    /// after the constructor returns, so what the constructor put there was never read.
    #[test]
    fn a_fresh_network_may_be_all_outputs_starts_at_zero_bias_and_refuses_a_nan_scale() {
        let mut rng = Rng::new(3);
        let all_out = Network::random(4, 4, 2, 0.5, &mut rng).expect("every unit may be an output");
        assert_eq!(all_out.n_out, 4);
        assert_eq!(all_out.b, vec![0.0; 4]);
        // n − n_out = 0, so the whole state is the output and the loss reads all four units.
        assert_eq!(all_out.cost(&[1.0, 2.0, 3.0, 4.0], &[1.0, 2.0, 3.0, 4.0]).unwrap(), 0.0);
        // Zero biases and zero input mean zero drive: the origin is a fixed point exactly.
        assert_eq!(all_out.velocity(&[0.0; 4], &[0.0; 2], &[0.0; 4], 0.0).unwrap(), vec![0.0; 4]);
        assert_eq!(all_out.relax(&[0.0; 4], &[0.0; 2], &[0.0; 4], 0.0, DT, TOL, 10).unwrap(), vec![0.0; 4]);
        assert!(all_out.w.iter().any(|v| v.abs() > 0.0), "a fixture whose couplings are all zero would say nothing");
        for scale in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(
                matches!(Network::random(3, 1, 1, scale, &mut rng), Err(EquilibriumError::OutOfRange { what: "scale", .. })),
                "scale = {scale} was accepted"
            );
        }
    }

    /// The field a unit feels is the unit's ROW of the coupling matrix, `Σ_j W_ij ρ(s_j)`, as
    /// [`Network::velocity`]'s own doc writes it — not its column.
    ///
    /// Why the suite could not see it, and why the question is real: every coupling matrix the
    /// other tests use is SYMMETRIC, for which a row and a column are the same numbers, so the
    /// transposed read is invisible to all of them. It is nonetheless reachable: [`Network`]'s
    /// every field is `pub`, the struct is constructed by literal elsewhere in this very module,
    /// and nothing in the type enforces symmetry — so "the matrix is symmetric" is a statement
    /// about the fixtures, not about the code. The fixture here is deliberately asymmetric:
    /// `W_01 = 2` and `W_10 = −3`. With one unit at zero its activity is zero, so only the OTHER
    /// unit's entry survives, and each state below reads exactly one off-diagonal entry. Both
    /// values are integers times `tanh(1)` computed by the same two operations, so the comparison
    /// is `assert_eq!` on the f64.
    #[test]
    fn the_field_a_unit_feels_is_its_row_of_the_coupling_matrix_not_its_column() {
        let n = Network { n: 2, n_out: 1, m: 1, w: vec![0.0, 2.0, -3.0, 0.0], u: vec![0.0, 0.0], b: vec![0.0, 0.0] };
        let (x, y) = ([0.0], [0.0]);
        let t = 1.0_f64.tanh();
        // s = (0, 1): unit 0 sees only W_01 = 2, and unit 1 sees nothing but its own leak.
        let v = n.velocity(&[0.0, 1.0], &x, &y, 0.0).unwrap();
        assert_eq!(v[0], 2.0 * t, "unit 0 read W_10 = -3 instead of W_01 = 2");
        assert_eq!(v[1], -1.0);
        // s = (1, 0): unit 1 sees only W_10 = -3, and unit 0 sees nothing but its own leak.
        let v = n.velocity(&[1.0, 0.0], &x, &y, 0.0).unwrap();
        assert_eq!(v[0], -1.0);
        assert_eq!(v[1], -3.0 * t, "unit 1 read W_01 = 2 instead of W_10 = -3");
    }

    /// [`Network::relax`]'s three guards at their boundaries: a `NaN` tolerance is out of range
    /// rather than a tolerance nothing can meet, a budget of exactly [`MAX_STEPS`] is a request
    /// and not an overrun, and a budget of zero reports an INFINITE last movement because no step
    /// was taken and nothing is therefore known about it.
    ///
    /// The `const` assertion pins the cap's value. It is a published constant and callers size
    /// their budgets against it, so moving it by a factor of ten is an API change, not an
    /// implementation detail.
    ///
    /// Why the suite could not see it: `bad_arguments_are_refused` writes the cap case as
    /// `MAX_STEPS + 1`, which is stated RELATIVE to the constant and so cannot see the constant
    /// move, and never asks for `MAX_STEPS` itself; its `tol` case is `-1.0`, which the arithmetic
    /// comparison `tol < 0.0` rejects too, while a `NaN` passes it and then makes `moved <= tol`
    /// false forever, turning a bad argument into a budget overrun. The unsettled-relaxation test
    /// reads its error with `steps: 3` and a movement above `1e-6`, which a budget of zero never
    /// reaches.
    #[test]
    fn the_relaxation_guards_hold_at_their_boundaries_and_a_zero_budget_knows_nothing() {
        const { assert!(MAX_STEPS == 10_000_000, "the documented cap is ten million steps") };
        let n = net();
        let (s, x, y) = ([0.0_f64; 7], [0.7, -0.5, 0.2], [0.4, -0.6]);
        for tol in [f64::NAN, -1.0, -1e-300, f64::NEG_INFINITY] {
            assert!(matches!(n.relax(&s, &x, &y, 0.0, DT, tol, 10), Err(EquilibriumError::OutOfRange { what: "tol", .. })), "tol = {tol}");
        }
        for dt in [f64::NAN, 0.0, 1.5, f64::INFINITY] {
            assert!(matches!(n.relax(&s, &x, &y, 0.0, dt, TOL, 10), Err(EquilibriumError::OutOfRange { what: "dt", .. })), "dt = {dt}");
        }
        // Exactly the cap is admissible, and asking for it changes nothing about the answer.
        let at_cap = n.relax(&s, &x, &y, 0.0, DT, TOL, MAX_STEPS).expect("exactly MAX_STEPS is a request, not an overrun");
        assert_eq!(at_cap, n.relax(&s, &x, &y, 0.0, DT, TOL, STEPS).unwrap());
        assert!(matches!(n.relax(&s, &x, &y, 0.0, DT, TOL, MAX_STEPS + 1), Err(EquilibriumError::OutOfRange { what: "max_steps", .. })));
        // A budget of zero took no step, so the movement it reports is unknown, not zero: a
        // caller reading `moved <= tol` off this must not be told the state had settled.
        let err = n.relax(&s, &x, &y, 0.0, DT, TOL, 0).unwrap_err();
        assert_eq!(err, EquilibriumError::NotSettled { moved: f64::INFINITY, steps: 0 });
        let EquilibriumError::NotSettled { moved, .. } = err else { panic!("a zero budget must not settle") };
        assert!(moved > TOL, "a zero-budget relaxation reported a movement of {moved}, which reads as settled");
    }

    /// An infinite nudge is out of range for [`Network::estimate`] and an infinite rate is
    /// non-finite for [`Network::descend`], and a refused `descend` leaves every parameter alone.
    ///
    /// Why the suite could not see it: both guards are tested only with a value that the WEAKER
    /// form also rejects — `beta = 0.0`, which fails `beta > 0.0`, and `rate = NAN`, which fails
    /// `rate.is_nan()`. An infinite `beta` passes `beta > 0.0`; dropping the finiteness half of
    /// that guard does not make the call succeed, it makes it fail LATER and as a different
    /// error, from inside the relaxation, so `is_err()` alone would not have seen it either. An
    /// infinite rate passes `is_nan()` and writes an infinity into every parameter.
    #[test]
    fn an_infinite_nudge_and_an_infinite_rate_are_refused_by_the_guard_that_names_them() {
        let n = net();
        let (x, y) = ([0.7, -0.5, 0.2], [0.4, -0.6]);
        for beta in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN, 0.0, -0.5] {
            assert!(
                matches!(n.estimate(&x, &y, beta, true, DT, TOL, 100), Err(EquilibriumError::OutOfRange { what: "beta", .. })),
                "beta = {beta} reached the relaxation"
            );
        }
        let g = n.energy_gradient(&[0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7], &x).unwrap();
        let mut m = n.clone();
        for rate in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
            assert!(matches!(m.descend(&g, rate), Err(EquilibriumError::NonFinite { what: "rate", index: 0 })), "rate = {rate}");
            assert_eq!(m, n, "a refused descend at rate {rate} wrote the parameters anyway");
        }
    }

    /// The SYMMETRIC estimator's negative phase starts from the free fixed point, exactly as its
    /// positive phase does. Same bistable fixture as
    /// `the_nudged_phase_starts_from_the_free_fixed_point_and_a_bistable_network_says_so`, with
    /// the target moved to `y = +1` so that it is the NEGATIVE phase whose starting point decides
    /// which well it lands in: at the origin the `−β` spring pushes the output unit DOWN by
    /// `−β(y − 0) = −0.3`, into the negative well, while from the free fixed point the same
    /// spring reads `−0.3(1 − 1.2276) = +0.068` and the pair stays where it is.
    ///
    /// MEASURED with `w_01 = w_10 = 5`, `b = (0.05, 0)`, `x = (0)`, `y = (1)`, `beta = 0.3`,
    /// `dt = 0.1`, `tol = 1e-12`: the free phase settles at `(1.2317852546420964,
    /// 1.2276313692648493)`, the `+β` phase at `(1.2289237306335485, 1.2071511755427389)` and the
    /// `−β` phase started there at `(1.235186191547005, 1.2531287993369165)`, giving
    /// `g.b = (0.00302, 0.02223)`. Started from the origin the `−β` phase lands at
    /// `(−1.2522763650759823, −1.5081888541152881)` and `g.b = (−2.819, −2.904)`: the opposite
    /// sign and some nine hundred times the size, not a last-bit difference.
    ///
    /// Why the suite could not see it: the existing start-from-the-free-fixed-point test calls
    /// `estimate` with `symmetric: false`, and on that branch the second phase is not a relaxation
    /// at all but `energy_gradient(&free)`, so the line this pins is never executed by it. The
    /// order-of-error test does exercise the symmetric branch, but only on `net()`, whose energy
    /// has one minimum — from any start the `−β` phase reaches the same fixed point to within the
    /// tolerance, so the estimate moves by far less than the `±1.0` slack that test allows on its
    /// ratios.
    #[test]
    fn the_symmetric_estimator_starts_its_negative_phase_from_the_free_fixed_point_too() {
        const DT: f64 = 0.1;
        const TOL: f64 = 1e-12;
        const STEPS: u64 = 200_000;
        let beta = 0.3;
        let n = Network { n: 2, n_out: 1, m: 1, w: vec![0.0, 5.0, 5.0, 0.0], u: vec![0.0, 0.0], b: vec![0.05, 0.0] };
        let (x, y) = ([0.0], [1.0]);
        let origin = vec![0.0; n.n];
        let free = n.relax(&origin, &x, &y, 0.0, DT, TOL, STEPS).expect("the free phase settles");
        let plus = n.relax(&free, &x, &y, beta, DT, TOL, STEPS).expect("the positive phase settles");
        let minus = n.relax(&free, &x, &y, -beta, DT, TOL, STEPS).expect("the negative phase settles");
        let from_origin = n.relax(&origin, &x, &y, -beta, DT, TOL, STEPS).expect("the negative phase settles");
        // The fixture is bistable and the two starts are in different wells: this is about which
        // fixed point the estimator reads, not about the last bit of a relaxation.
        assert!(free[0] > 1.0 && free[1] > 1.0, "the free phase left the positive well: {free:?}");
        assert!(minus[0] > 1.0 && minus[1] > 1.0, "the negative phase left the positive well: {minus:?}");
        assert!(from_origin[0] < -1.0 && from_origin[1] < -1.0, "the fixture is no longer bistable: {from_origin:?}");
        // The identity, out of the same public calls in the same order, so equality and not a
        // tolerance.
        let span = 2.0 * beta;
        let (gp, gm) = (n.energy_gradient(&plus, &x).unwrap(), n.energy_gradient(&minus, &x).unwrap());
        let want: Vec<f64> = gp.b.iter().zip(&gm.b).map(|(at_plus, at_minus)| (at_plus - at_minus) / span).collect();
        let got = n.estimate(&x, &y, beta, true, DT, TOL, STEPS).expect("both phases settle");
        assert_eq!(got.b, want, "the negative phase did not start at the free fixed point");
        assert_eq!(got.w, gp.w.iter().zip(&gm.w).map(|(at_plus, at_minus)| (at_plus - at_minus) / span).collect::<Vec<f64>>());
        assert_eq!(got.u, gp.u.iter().zip(&gm.u).map(|(at_plus, at_minus)| (at_plus - at_minus) / span).collect::<Vec<f64>>());
        // And the size of it: the origin-started negative phase gives the other sign entirely.
        let wrong = n.energy_gradient(&from_origin, &x).unwrap();
        let other: Vec<f64> = gp.b.iter().zip(&wrong.b).map(|(at_plus, at_origin)| (at_plus - at_origin) / span).collect();
        assert!(got.b[1] > 0.0 && other[1] < -2.0, "measured {} against {}", got.b[1], other[1]);
    }

}
