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

/// The most relaxation steps one call will take; a request past it is refused.
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

}
