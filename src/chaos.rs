//! Chaotic benchmark systems — Mackey–Glass, the delay equation `NeuroBench`'s forecasting task is
//! drawn from, and Lorenz — integrated at a stated order, checked against what is known about them
//! in closed form, and `NeuroBench`'s own Mackey–Glass series reproduced from the equation.
//!
//! # Mackey–Glass
//!
//! Mackey and Glass, *Oscillation and chaos in physiological control systems*, Science
//! 197:287–289, 1977, Eq. 4b — the white-cell model, `dP/dt = β₀θⁿP_τ/(θⁿ + P_τⁿ) − γP` — written
//! in `x = P/θ`, which removes `θ` exactly:
//!
//! ```text
//! dx/dt = β x(t − τ)/(1 + x(t − τ)ⁿ) − γ x(t)
//! ```
//!
//! The paper's parameters are the ones used here, `β₀ = 0.2` and `γ = 0.1` per day and `n = 10`
//! (Fig. 2 caption). Its figures show two delays: `τ = 6` days, "a low-amplitude oscillation with a
//! period of 20 days" (Fig. 2b), and `τ = 20`, "an aperiodic pattern" (Fig. 2c). The benchmark's
//! `τ = 17` ([`MackeyGlass::NEUROBENCH`]) is not in the paper; it comes from later work on the
//! equation's attractors. What is exact about it:
//!
//! - the positive fixed point `x* = (β/γ − 1)^{1/n}` — 1 for these parameters;
//! - the slope of the delayed term there, `a = β(1 + (1 − n)x*ⁿ)/(1 + x*ⁿ)²` — −0.4;
//! - so the linearisation `λ = −γ + a e^{−λτ}`, which crosses the imaginary axis at `λ = iω` when
//!   `cos ωτ = γ/a` and `ω² = a² − γ²`: the fixed point loses stability at
//!   `τ_c = arccos(γ/a)/√(a² − γ²)` = 4.7082… ([`MackeyGlass::hopf_delay`]), below which every
//!   solution settles and above which the delay drives an oscillation.
//!
//! The solver ([`MackeyGlass::solve`]) is classical Runge–Kutta on a grid that divides `τ`, so the
//! delayed argument of every stage falls on a grid point or a grid midpoint, and the derivative
//! discontinuities a constant history propagates — at `0, τ, 2τ, …` — fall on grid points too.
//! Midpoints are filled by the cubic Hermite interpolant of the stored solution and its derivative,
//! which is fourth-order accurate, so the whole scheme is: the tests measure it.
//!
//! ⭐ **`NeuroBench`'s series, reproduced from the equation.** Its `mg_17.npy` (from the dataset its
//! `MackeyGlass` class downloads) starts at `t = τ = 17` — where its `jitcdde` solver stands after
//! `step_on_discontinuities` — from the constant history 0.7206597, sampled every `197/75` time
//! units. This solver at `h = 0.05` matches its first 401 samples — five Lyapunov times — within
//! `10⁻⁸`, and the difference then grows at the rate chaos sets, as it must.
//!
//! ⚠ **`NeuroBench`'s `MackeyGlass` class never uses its equation.** It decides between loading a
//! file and generating the series with `if os.path.exists(self.file_path) is not None:` — and
//! `os.path.exists` returns a `bool`, which is never `None`. So it always loads, `generate_data` is
//! unreachable, and the constructor's `tau`, `beta`, `gamma`, `nmg`, `constant_past` and `seed_id`
//! change nothing: asking for `tau = 30` returns whichever file `file_path` names. And its default
//! `file_path = None` reaches `os.path.exists(None)`, which raises `TypeError`. (Its repository, as
//! read on 2026-09-24, `neurobench/datasets/mackey_glass.py`.) [`MackeyGlass::solve`] integrates
//! whatever parameters it is given.
//!
//! # Lorenz
//!
//! Lorenz, *Deterministic nonperiodic flow*, Journal of the Atmospheric Sciences 20:130–141, 1963:
//! `ẋ = σ(y − x)`, `ẏ = x(ρ − z) − y`, `ż = xy − βz`, with `σ = 10`, `ρ = 28`, `β = 8/3`
//! ([`Lorenz::LORENZ_1963`]). Exact: the fixed points `(0, 0, 0)` and
//! `(±√(β(ρ − 1)), ±√(β(ρ − 1)), ρ − 1)`; the divergence `−(σ + 1 + β)`, constant, so every volume
//! shrinks by exactly `e^{−(σ+1+β)t}` and the Lyapunov exponents sum to it; and the subcritical
//! Hopf bifurcation of the two outer fixed points at `ρ_H = σ(σ + β + 3)/(σ − β − 1)` = 24.7368….
//! [`Lorenz::lyapunov`] estimates the spectrum by re-orthonormalising the tangent flow, and the
//! tests hold it to that sum and to the largest exponent's commonly tabulated 0.9056 (Sprott,
//! *Chaos and Time-Series Analysis*, Oxford University Press, 2003).

use core::fmt;

/// Why a chaotic system could not be integrated.
#[derive(Debug, Clone, PartialEq)]
pub enum ChaosError {
    /// A parameter, step or duration that must be finite and positive was not.
    NotPositive {
        /// Which.
        what: &'static str,
        /// Its value.
        value: f64,
    },
    /// A value that must be finite was not.
    NonFinite {
        /// Which.
        what: &'static str,
        /// Its value.
        value: f64,
    },
    /// A step that does not divide the delay, so the delayed stages would fall between grid points.
    Misaligned {
        /// The delay.
        tau: f64,
        /// The step.
        h: f64,
    },
    /// More steps than [`MAX_STEPS`].
    TooLong {
        /// Steps asked for.
        steps: f64,
    },
}

impl fmt::Display for ChaosError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotPositive { what, value } => write!(f, "{what} = {value} must be finite and positive"),
            Self::NonFinite { what, value } => write!(f, "{what} = {value} is not finite"),
            Self::Misaligned { tau, h } => write!(f, "the step {h} does not divide the delay {tau}"),
            Self::TooLong { steps } => write!(f, "{steps} steps is more than {MAX_STEPS}"),
        }
    }
}

impl std::error::Error for ChaosError {}

/// The most steps one call integrates.
pub const MAX_STEPS: usize = 10_000_000;

fn positive(what: &'static str, value: f64) -> Result<f64, ChaosError> {
    if value.is_finite() && value > 0.0 { Ok(value) } else { Err(ChaosError::NotPositive { what, value }) }
}

fn finite(what: &'static str, value: f64) -> Result<f64, ChaosError> {
    if value.is_finite() { Ok(value) } else { Err(ChaosError::NonFinite { what, value }) }
}

/// The cubic Hermite interpolant on `[0, h]` at fraction `s`, from the values and derivatives at
/// both ends.
fn hermite(x0: f64, d0: f64, x1: f64, d1: f64, h: f64, s: f64) -> f64 {
    let s2 = s * s;
    let s3 = s2 * s;
    (2.0 * s3 - 3.0 * s2 + 1.0) * x0 + (s3 - 2.0 * s2 + s) * h * d0 + (3.0 * s2 - 2.0 * s3) * x1 + (s3 - s2) * h * d1
}

/// The Mackey–Glass delay equation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MackeyGlass {
    /// `β`, the production rate.
    pub beta: f64,
    /// `γ`, the decay rate.
    pub gamma: f64,
    /// `n`, the Hill exponent.
    pub n: f64,
    /// `τ`, the delay.
    pub tau: f64,
}

/// A solution of [`MackeyGlass::solve`] on its grid.
#[derive(Debug, Clone, PartialEq)]
pub struct DelaySolution {
    /// The step.
    pub h: f64,
    /// `x` at `t = k·h`.
    pub x: Vec<f64>,
    /// `dx/dt` at `t = k·h`, from the right.
    pub dx: Vec<f64>,
}

impl DelaySolution {
    /// `x(t)` by cubic Hermite interpolation between grid points; `None` outside `[0, t_end]` or
    /// for a `t` that is not finite.
    #[must_use]
    pub fn at(&self, t: f64) -> Option<f64> {
        if !(t >= 0.0) {
            return None;
        }
        let u = t / self.h;
        let k = u.floor() as usize;
        if k + 1 >= self.x.len() {
            return (k + 1 == self.x.len() && u == k as f64).then(|| self.x[k]);
        }
        Some(hermite(self.x[k], self.dx[k], self.x[k + 1], self.dx[k + 1], self.h, u - k as f64))
    }
}

impl MackeyGlass {
    /// `β = 0.2`, `γ = 0.1`, `n = 10`, `τ = 17`: the chaotic benchmark, and `NeuroBench`'s defaults.
    pub const NEUROBENCH: Self = Self { beta: 0.2, gamma: 0.1, n: 10.0, tau: 17.0 };

    /// `NeuroBench`'s constant history.
    pub const NEUROBENCH_PAST: f64 = 0.7206597;

    /// `dx/dt` given `x(t)` and `x(t − τ)`.
    #[must_use]
    pub fn field(&self, x: f64, delayed: f64) -> f64 {
        self.beta * delayed / (1.0 + delayed.powf(self.n)) - self.gamma * x
    }

    /// The positive fixed point `(β/γ − 1)^{1/n}`; `None` when `β ≤ γ`, where only zero is.
    #[must_use]
    pub fn fixed_point(&self) -> Option<f64> {
        (self.beta > self.gamma).then(|| (self.beta / self.gamma - 1.0).powf(1.0 / self.n))
    }

    /// The derivative of the delayed term at the positive fixed point, `a = β(1 + (1 − n)x*ⁿ)/(1 + x*ⁿ)²`.
    #[must_use]
    pub fn delayed_slope(&self) -> Option<f64> {
        let x = self.fixed_point()?;
        let p = x.powf(self.n);
        Some(self.beta * (1.0 + (1.0 - self.n) * p) / ((1.0 + p) * (1.0 + p)))
    }

    /// The delay at which the positive fixed point loses stability, `arccos(γ/a)/√(a² − γ²)`;
    /// `None` when `a ≥ −γ`, where no delay destabilises it.
    #[must_use]
    pub fn hopf_delay(&self) -> Option<f64> {
        let a = self.delayed_slope()?;
        (a < -self.gamma).then(|| (self.gamma / a).acos() / (a * a - self.gamma * self.gamma).sqrt())
    }

    /// Integrate from the constant history `past` on `t ≤ 0` to `t_end` with step `h`, which must
    /// divide `τ`.
    ///
    /// # Errors
    ///
    /// [`ChaosError::NotPositive`] for a parameter, `h` or `t_end` that is not; [`ChaosError::NonFinite`]
    /// for a `past` or `γ` that is not finite; [`ChaosError::Misaligned`] for an `h` that does not
    /// divide `τ` to one part in `10⁹`; [`ChaosError::TooLong`] past [`MAX_STEPS`].
    pub fn solve(&self, past: f64, h: f64, t_end: f64) -> Result<DelaySolution, ChaosError> {
        positive("beta", self.beta)?;
        finite("gamma", self.gamma)?;
        positive("n", self.n)?;
        let tau = positive("tau", self.tau)?;
        let h = positive("h", h)?;
        let t_end = positive("t_end", t_end)?;
        finite("past", past)?;
        let m = (tau / h).round();
        // `m = 0` is never aligned: `|0·h − τ| = τ`.
        if !((m * h - tau).abs() <= 1e-9 * tau) {
            return Err(ChaosError::Misaligned { tau, h });
        }
        let steps = (t_end / h).ceil();
        if steps > MAX_STEPS as f64 {
            return Err(ChaosError::TooLong { steps });
        }
        let (m, steps) = (m as usize, steps as usize);
        let mut x = Vec::with_capacity(steps + 1);
        let mut dx = Vec::with_capacity(steps + 1);
        x.push(past);
        dx.push(self.field(past, past));
        // `x(t_k + s·h − τ)`: the history before zero, the stored solution after it.
        let delayed = |x: &[f64], dx: &[f64], k: usize, s: f64| -> f64 {
            if k < m { past } else { hermite(x[k - m], dx[k - m], x[k - m + 1], dx[k - m + 1], h, s) }
        };
        for k in 0..steps {
            let (d0, dh) = if k < m { (past, past) } else { (x[k - m], delayed(&x, &dx, k, 0.5)) };
            let d1 = if k + 1 < m { past } else { x[k + 1 - m] };
            let k1 = self.field(x[k], d0);
            let k2 = self.field(x[k] + 0.5 * h * k1, dh);
            let k3 = self.field(x[k] + 0.5 * h * k2, dh);
            let k4 = self.field(x[k] + h * k3, d1);
            let next = x[k] + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
            x.push(next);
            let d_next = if k + 1 < m { past } else { x[k + 1 - m] };
            dx.push(self.field(next, d_next));
        }
        Ok(DelaySolution { h, x, dx })
    }
}

/// The Lorenz system.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Lorenz {
    /// `σ`, the Prandtl number.
    pub sigma: f64,
    /// `ρ`, the reduced Rayleigh number.
    pub rho: f64,
    /// `β`, the aspect factor.
    pub beta: f64,
}

impl Lorenz {
    /// Lorenz's own parameters: `σ = 10`, `ρ = 28`, `β = 8/3`.
    pub const LORENZ_1963: Self = Self { sigma: 10.0, rho: 28.0, beta: 8.0 / 3.0 };

    /// The vector field at `p = (x, y, z)`.
    #[must_use]
    pub fn field(&self, p: [f64; 3]) -> [f64; 3] {
        let [x, y, z] = p;
        [self.sigma * (y - x), x * (self.rho - z) - y, x * y - self.beta * z]
    }

    /// The Jacobian at `p`, row-major.
    #[must_use]
    pub fn jacobian(&self, p: [f64; 3]) -> [[f64; 3]; 3] {
        let [x, y, z] = p;
        [[-self.sigma, self.sigma, 0.0], [self.rho - z, -1.0, -x], [y, x, -self.beta]]
    }

    /// The fixed points: the origin, and for `ρ > 1` the two at `(±√(β(ρ − 1)), ±√(β(ρ − 1)), ρ − 1)`.
    #[must_use]
    pub fn fixed_points(&self) -> Vec<[f64; 3]> {
        let mut out = vec![[0.0; 3]];
        if self.rho > 1.0 {
            let r = (self.beta * (self.rho - 1.0)).sqrt();
            out.push([r, r, self.rho - 1.0]);
            out.push([-r, -r, self.rho - 1.0]);
        }
        out
    }

    /// The divergence of the field, `−(σ + 1 + β)`: the same at every point.
    #[must_use]
    pub fn divergence(&self) -> f64 {
        -(self.sigma + 1.0 + self.beta)
    }

    /// `ρ_H = σ(σ + β + 3)/(σ − β − 1)`, where the outer fixed points lose stability; `None` when
    /// `σ ≤ β + 1`, where they never do.
    #[must_use]
    pub fn hopf_rho(&self) -> Option<f64> {
        let d = self.sigma - self.beta - 1.0;
        (d > 0.0).then(|| self.sigma * (self.sigma + self.beta + 3.0) / d)
    }

    /// One classical Runge–Kutta step.
    #[must_use]
    pub fn step(&self, p: [f64; 3], h: f64) -> [f64; 3] {
        let add = |a: [f64; 3], b: [f64; 3], s: f64| [a[0] + s * b[0], a[1] + s * b[1], a[2] + s * b[2]];
        let k1 = self.field(p);
        let k2 = self.field(add(p, k1, 0.5 * h));
        let k3 = self.field(add(p, k2, 0.5 * h));
        let k4 = self.field(add(p, k3, h));
        core::array::from_fn(|i| p[i] + h / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]))
    }

    /// [`Lorenz::step`], and the tangent vectors `q` carried through it: each stage's tangent takes
    /// the Jacobian at that stage's state, which makes the result the exact derivative of the step
    /// map applied to each vector.
    pub fn step_with_tangents(&self, p: [f64; 3], q: &mut [[f64; 3]; 3], h: f64) -> [f64; 3] {
        let add = |a: [f64; 3], b: [f64; 3], s: f64| -> [f64; 3] { core::array::from_fn(|i| a[i] + s * b[i]) };
        let tangent = |p: [f64; 3], v: [f64; 3]| -> [f64; 3] {
            let j = self.jacobian(p);
            core::array::from_fn(|r| j[r][0] * v[0] + j[r][1] * v[1] + j[r][2] * v[2])
        };
        let k1 = self.field(p);
        let p2 = add(p, k1, 0.5 * h);
        let k2 = self.field(p2);
        let p3 = add(p, k2, 0.5 * h);
        let k3 = self.field(p3);
        let p4 = add(p, k3, h);
        let k4 = self.field(p4);
        for v in q.iter_mut() {
            let t1 = tangent(p, *v);
            let t2 = tangent(p2, add(*v, t1, 0.5 * h));
            let t3 = tangent(p3, add(*v, t2, 0.5 * h));
            let t4 = tangent(p4, add(*v, t3, h));
            *v = core::array::from_fn(|i| v[i] + h / 6.0 * (t1[i] + 2.0 * t2[i] + 2.0 * t3[i] + t4[i]));
        }
        core::array::from_fn(|i| p[i] + h / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]))
    }

    /// The three Lyapunov exponents, largest first, from `steps` Runge–Kutta steps of the state and
    /// its tangent flow, re-orthonormalised by Gram–Schmidt every `every` steps.
    ///
    /// # Errors
    ///
    /// [`ChaosError::NotPositive`] for an `h`, `steps` or `every` that is not; [`ChaosError::NonFinite`]
    /// for a start that is not finite; [`ChaosError::TooLong`] past [`MAX_STEPS`].
    pub fn lyapunov(&self, start: [f64; 3], h: f64, steps: usize, every: usize) -> Result<[f64; 3], ChaosError> {
        let h = positive("h", h)?;
        positive("steps", steps as f64)?;
        positive("every", every as f64)?;
        if steps > MAX_STEPS {
            return Err(ChaosError::TooLong { steps: steps as f64 });
        }
        for &c in &start {
            finite("start", c)?;
        }
        let mut p = start;
        let mut q = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let mut sums = [0.0; 3];
        let mut done = 0;
        while done < steps {
            p = self.step_with_tangents(p, &mut q, h);
            done += 1;
            if done % every == 0 || done == steps {
                // Gram–Schmidt: each vector's length after removing the earlier ones is the growth
                // along the next direction of the spectrum.
                for a in 0..3 {
                    for b in 0..a {
                        let dot: f64 = (0..3).map(|i| q[a][i] * q[b][i]).sum();
                        for i in 0..3 {
                            q[a][i] -= dot * q[b][i];
                        }
                    }
                    let norm = (0..3).map(|i| q[a][i] * q[a][i]).sum::<f64>().sqrt();
                    sums[a] += norm.ln();
                    for i in 0..3 {
                        q[a][i] /= norm;
                    }
                }
            }
        }
        let t = steps as f64 * h;
        Ok(sums.map(|s| s / t))
    }
}


#[cfg(test)]
mod tests {
    use super::{ChaosError, Lorenz, MAX_STEPS, MackeyGlass};

    /// `mg_17.npy` from `NeuroBench`'s Mackey–Glass dataset, samples 0, 10, 20, … 400.
    const MG17: [f64; 41] = [1.2667777302367338, 0.43743287972408684, 1.138314784277696, 0.6822547878629676, 1.209744811288032, 0.5272135105915416, 1.1577293670002702, 0.8117818871090186, 1.0517484241185258, 0.9650070499560879, 1.128432816170021, 0.863575485955921, 0.932012528298373, 1.0109251527741834, 1.0438705956863794, 1.0849258913137023, 0.676406674256371, 0.9361643870757593, 0.9952256271598664, 1.078508204497474, 0.7519071157645635, 1.070242195022825, 0.659400824489561, 1.2498958997447551, 0.5497633205099447, 1.1366168492052453, 0.78473301011151, 1.1216505359211502, 0.923934478827487, 1.16012954421735, 0.6888205033640936, 1.0819431410539697, 0.9955759922894293, 0.9168867451267748, 1.1459135352201215, 0.5940000332204214, 1.0312817467890307, 0.7146114255382029, 1.293364714805231, 0.4527684139187731, 1.1394723752805496];

    /// The fixed point, the slope there and the Hopf delay, in closed form: 1, −0.4 and
    /// `arccos(−1/4)/√0.15` = 4.708196289360752.
    #[test]
    fn mackey_glass_closed_forms() {
        let m = MackeyGlass::NEUROBENCH;
        assert_eq!(m.fixed_point(), Some(1.0));
        assert!((m.delayed_slope().unwrap() + 0.4).abs() < 1e-15);
        assert!((m.hopf_delay().unwrap() - 4.708_196_289_360_752).abs() < 1e-12);
        assert!(m.field(1.0, 1.0).abs() < 1e-16, "the fixed point is one");
        assert_eq!(MackeyGlass { beta: 0.1, ..m }.fixed_point(), None, "β = γ: only zero");
        assert_eq!(MackeyGlass { n: 1.0, ..m }.hopf_delay(), None, "a = 0.05: no delay destabilises it");
        assert_eq!(MackeyGlass { n: 3.0, ..m }.hopf_delay(), None, "a = −0.05 > −γ: the feedback is too weak");
        let p = MackeyGlass { beta: 0.3, gamma: 0.1, n: 4.0, tau: 17.0 };
        let x = p.fixed_point().unwrap();
        assert!((x - 2f64.powf(0.25)).abs() < 1e-15 && p.field(x, x).abs() < 1e-15);
        assert!((p.delayed_slope().unwrap() - 0.3 * (1.0 - 3.0 * 2.0) / 9.0).abs() < 1e-15);
    }

    /// Either side of the Hopf delay the solver does what the linearisation says: below it every
    /// solution settles on the fixed point, above it an oscillation grows and stays.
    #[test]
    fn the_fixed_point_loses_stability_at_the_hopf_delay() {
        let late = |tau: f64| {
            let s = MackeyGlass { tau, ..MackeyGlass::NEUROBENCH }.solve(0.5, 0.05, 1500.0).unwrap();
            let tail = &s.x[s.x.len() - 4000..];
            tail.iter().fold(0.0_f64, |m, &v| m.max((v - 1.0).abs()))
        };
        assert!(late(4.4) < 1e-6, "τ = 4.4 < τ_c: {}", late(4.4));
        assert!(late(5.0) > 0.05, "τ = 5.0 > τ_c: {}", late(5.0));
    }

    /// The one number Mackey and Glass print for Eq. 4b: at `τ = 6` days the solution "has a
    /// low-amplitude oscillation with a period of 20 days" (Fig. 2b, from the paper's initial
    /// condition `P = 0.10`). Measured here as the spacing of upward crossings of the fixed point
    /// after `t = 200`: 20.0765 days at `h = 0.01` and at `h = 0.005` (the two agree to 1e-6), so
    /// the paper's "20" is this period rounded. And at `τ = 20` the crossings are irregular — the
    /// aperiodic Fig. 2c — with spacings from under 12 days to over 70.
    #[test]
    fn mackey_and_glass_print_a_twenty_day_period_at_a_six_day_delay() {
        let crossings = |tau: f64| {
            let h = 0.01;
            let s = MackeyGlass { tau, ..MackeyGlass::NEUROBENCH }.solve(0.1, h, 600.0).unwrap();
            let mut up = Vec::new();
            for k in 1..s.x.len() {
                let (a, b, t) = (s.x[k - 1], s.x[k], k as f64 * h);
                if t > 200.0 && a < 1.0 && b >= 1.0 {
                    up.push(t - h + h * (1.0 - a) / (b - a));
                }
            }
            up.windows(2).map(|w| w[1] - w[0]).collect::<Vec<f64>>()
        };
        let six = crossings(6.0);
        assert!(six.len() >= 15, "{six:?}");
        for p in &six {
            assert!((p - 20.0765).abs() < 1e-3, "τ = 6: period {p}");
        }
        let twenty = crossings(20.0);
        let (lo, hi) = twenty.iter().fold((f64::MAX, 0.0_f64), |(a, b), &p| (a.min(p), b.max(p)));
        assert!(twenty.len() >= 5 && lo < 15.0 && hi > 60.0, "τ = 20: {twenty:?}");
    }

    /// The solver is fourth order: halving the step divides the error at `t = 100` by sixteen.
    #[test]
    fn the_delay_solver_is_fourth_order() {
        let m = MackeyGlass::NEUROBENCH;
        let reference = m.solve(MackeyGlass::NEUROBENCH_PAST, 1.0 / 64.0, 100.0).unwrap().at(100.0).unwrap();
        let err: Vec<f64> = [0.5, 0.25, 0.125]
            .iter()
            .map(|&h| (m.solve(MackeyGlass::NEUROBENCH_PAST, h, 100.0).unwrap().at(100.0).unwrap() - reference).abs())
            .collect();
        for pair in err.windows(2) {
            let ratio = pair[0] / pair[1];
            assert!((13.0..19.0).contains(&ratio), "{err:?}: {ratio}");
        }
    }

    /// `NeuroBench`'s own Mackey–Glass series, from the equation: its first 401 samples within
    /// `10⁻⁸`, from its constant history, starting at `t = τ` and sampled every `197/75`.
    #[test]
    fn neurobenchs_series_is_the_equations() {
        let m = MackeyGlass::NEUROBENCH;
        let dt = 197.0 / 75.0;
        let s = m.solve(MackeyGlass::NEUROBENCH_PAST, 0.05, 17.0 + 400.0 * dt + 1.0).unwrap();
        for (j, &want) in MG17.iter().enumerate() {
            let t = 17.0 + (10 * j) as f64 * dt;
            let got = s.at(t).unwrap();
            assert!((got - want).abs() < 1e-8, "sample {}: {got} against {want}", 10 * j);
        }
        assert!((s.at(17.0).unwrap() - MG17[0]).abs() < 1e-11);
    }

    /// Between grid points the solution is its cubic Hermite interpolant; on them it is the stored
    /// value; outside the solved span there is none.
    #[test]
    fn a_solution_is_read_between_and_on_its_grid() {
        let s = MackeyGlass::NEUROBENCH.solve(0.9, 0.5, 10.0).unwrap();
        assert_eq!(s.x.len(), 21);
        assert_eq!(s.at(0.0), Some(0.9));
        assert_eq!(s.at(3.0), Some(s.x[6]));
        assert_eq!(s.at(10.0), Some(s.x[20]));
        assert_eq!(s.at(10.25), None);
        assert_eq!(s.at(-0.1), None);
        assert_eq!(s.at(f64::NAN), None);
        let mid = s.at(1.25).unwrap();
        let (x0, x1) = (s.x[2], s.x[3]);
        assert!(mid != 0.5 * (x0 + x1) && (mid - 0.5 * (x0 + x1)).abs() < 1e-3, "Hermite, not linear: {mid}");
        // Before the first delay the history is constant, so the field is `β·past/(1 + pastⁿ) − γx`,
        // and the derivative stored at zero is exactly that.
        assert_eq!(s.dx[0], MackeyGlass::NEUROBENCH.field(0.9, 0.9));
    }

    /// Every refusal.
    #[test]
    fn every_refusal_names_what_it_refused() {
        let m = MackeyGlass::NEUROBENCH;
        let msg = |r: Result<super::DelaySolution, ChaosError>| r.unwrap_err().to_string();
        assert_eq!(msg(m.solve(0.5, 0.03, 10.0)), "the step 0.03 does not divide the delay 17");
        assert_eq!(msg(m.solve(0.5, 20.0, 40.0)), "the step 20 does not divide the delay 17");
        assert_eq!(msg(m.solve(0.5, 0.0, 10.0)), "h = 0 must be finite and positive");
        assert_eq!(msg(m.solve(0.5, 0.5, -1.0)), "t_end = -1 must be finite and positive");
        assert_eq!(msg(m.solve(0.5, 0.5, f64::INFINITY)), "t_end = inf must be finite and positive");
        assert_eq!(msg(MackeyGlass { beta: f64::INFINITY, ..m }.solve(0.5, 0.5, 1.0)), "beta = inf must be finite and positive");
        assert_eq!(msg(m.solve(f64::NAN, 0.5, 10.0)), "past = NaN is not finite");
        assert_eq!(msg(m.solve(0.5, 40.0, 80.0)), "the step 40 does not divide the delay 17");
        // One step past the limit: a solver that ignored it would run ten million steps and succeed.
        assert_eq!(msg(m.solve(0.5, 0.5, 5_000_000.5)), "10000001 steps is more than 10000000");
        assert!(m.solve(0.5, 0.5, 5_000_000.0).is_ok_and(|s| s.x.len() == MAX_STEPS + 1));
        assert_eq!(msg(MackeyGlass { beta: 0.0, ..m }.solve(0.5, 0.5, 1.0)), "beta = 0 must be finite and positive");
        assert_eq!(msg(MackeyGlass { gamma: f64::INFINITY, ..m }.solve(0.5, 0.5, 1.0)), "gamma = inf is not finite");
        assert_eq!(msg(MackeyGlass { n: -1.0, ..m }.solve(0.5, 0.5, 1.0)), "n = -1 must be finite and positive");
        assert_eq!(msg(MackeyGlass { tau: 0.0, ..m }.solve(0.5, 0.5, 1.0)), "tau = 0 must be finite and positive");
        // A negative decay rate is a legitimate — growing — equation.
        assert!(MackeyGlass { gamma: -0.01, ..m }.solve(0.5, 0.5, 1.0).is_ok());
        assert_eq!(MAX_STEPS, 10_000_000);
        let l = Lorenz::LORENZ_1963;
        assert_eq!(l.lyapunov([1.0; 3], 0.0, 10, 1).unwrap_err().to_string(), "h = 0 must be finite and positive");
        assert_eq!(l.lyapunov([1.0; 3], 0.01, 0, 1).unwrap_err().to_string(), "steps = 0 must be finite and positive");
        assert_eq!(l.lyapunov([1.0; 3], 0.01, 10, 0).unwrap_err().to_string(), "every = 0 must be finite and positive");
        assert_eq!(l.lyapunov([1.0, f64::NAN, 1.0], 0.01, 10, 1).unwrap_err().to_string(), "start = NaN is not finite");
        assert_eq!(l.lyapunov([1.0; 3], 0.01, MAX_STEPS + 1, 1).unwrap_err().to_string(), "10000001 steps is more than 10000000");
    }

    /// Lorenz's fixed points are zeros of the field; the divergence is `−(σ + 1 + β)` everywhere; and
    /// the Hopf value of `ρ` is where the characteristic polynomial at the outer fixed points has a
    /// root pair on the imaginary axis — `c₂c₁ = c₀` for `λ³ + c₂λ² + c₁λ + c₀`.
    #[test]
    fn lorenz_closed_forms() {
        let l = Lorenz::LORENZ_1963;
        let fp = l.fixed_points();
        assert_eq!(fp.len(), 3);
        for p in &fp {
            assert!(l.field(*p).iter().all(|v| v.abs() < 1e-13), "{p:?}");
        }
        assert!((fp[1][0] - 72f64.sqrt()).abs() < 1e-14 && fp[1][2] == 27.0 && fp[2][0] == -fp[1][0]);
        assert_eq!(Lorenz { rho: 0.5, ..l }.fixed_points(), vec![[0.0; 3]]);
        assert_eq!(Lorenz { rho: 1.0, ..l }.fixed_points(), vec![[0.0; 3]], "the pitchfork is at ρ = 1, and there only the origin");
        for p in [[1.0, 2.0, 3.0], [-7.0, 0.5, 30.0]] {
            let j = l.jacobian(p);
            assert!((j[0][0] + j[1][1] + j[2][2] - l.divergence()).abs() < 1e-14);
        }
        assert!((l.divergence() + 41.0 / 3.0).abs() < 1e-14);
        let rho_h = l.hopf_rho().unwrap();
        assert!((rho_h - 470.0 / 19.0).abs() < 1e-12, "{rho_h}");
        let hurwitz = |rho: f64| {
            let (s, b) = (l.sigma, l.beta);
            (s + b + 1.0) * b * (s + rho) - 2.0 * s * b * (rho - 1.0)
        };
        assert!(hurwitz(rho_h).abs() < 1e-11, "the imaginary pair at ρ_H");
        assert!(hurwitz(20.0) > 0.0 && hurwitz(28.0) < 0.0, "stable below, unstable at Lorenz's 28");
        assert_eq!(Lorenz { sigma: 3.0, ..l }.hopf_rho(), None);
        assert_eq!(Lorenz { sigma: 3.0, beta: 2.0, ..l }.hopf_rho(), None, "σ = β + 1 exactly: no Hopf point");
        // The Jacobian at a point, entry by entry, against central differences of the field.
        let p = [1.3, -0.7, 22.0];
        let j = l.jacobian(p);
        for c in 0..3 {
            let (mut a, mut b) = (p, p);
            a[c] += 1e-6;
            b[c] -= 1e-6;
            let (fa, fb) = (l.field(a), l.field(b));
            for r in 0..3 {
                assert!((j[r][c] - (fa[r] - fb[r]) / 2e-6).abs() < 1e-7, "J[{r}][{c}]");
            }
        }
    }

    /// The Lyapunov spectrum: the three exponents sum to the divergence, as Liouville's theorem makes
    /// them; the middle one is zero, as for every flow; and the largest and smallest are Sprott's
    /// tabulated 0.9056 and −14.5723.
    #[test]
    fn lorenz_lyapunov_spectrum() {
        let l = Lorenz::LORENZ_1963;
        let mut p = [1.0, 1.0, 20.0];
        for _ in 0..5_000 {
            p = l.step(p, 0.01);
        }
        let [a, b, c] = l.lyapunov(p, 0.01, 200_000, 10).unwrap();
        // Measured over 2 000 time units at h = 0.01: 0.90626, 0.00093, −14.57375, summing to
        // −13.66656 — the exact −41/3 less the Runge–Kutta step's own error in the tangent volume.
        assert!((a + b + c - l.divergence()).abs() < 1e-3, "sum {} against {}", a + b + c, l.divergence());
        assert!((a - 0.9056).abs() < 0.005, "λ₁ = {a}");
        assert!(b.abs() < 0.005, "λ₂ = {b}");
        assert!((c + 14.5723).abs() < 0.01, "λ₃ = {c}");
        assert!(a > b && b > c);
        // Any stretch, however short, sums to the divergence — here five steps, shorter than one
        // renormalisation interval, so only the final one counts.
        let short = l.lyapunov(p, 0.01, 5, 10).unwrap();
        assert!((short.iter().sum::<f64>() - l.divergence()).abs() < 1e-3, "{short:?}");
    }

    /// The tangents carried through a step are the step map's derivative: twenty steps from a point on
    /// the attractor, each tangent against central differences of [`Lorenz::step`] in its direction.
    #[test]
    fn the_tangents_are_the_derivative_of_the_step() {
        let l = Lorenz::LORENZ_1963;
        let p0 = [-3.1, 2.4, 21.5];
        let mut q = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let mut p = p0;
        for _ in 0..20 {
            p = l.step_with_tangents(p, &mut q, 0.01);
        }
        let run = |mut x: [f64; 3]| {
            for _ in 0..20 {
                x = l.step(x, 0.01);
            }
            x
        };
        assert_eq!(p, run(p0), "the state advances exactly as step does");
        for c in 0..3 {
            let (mut a, mut b) = (p0, p0);
            a[c] += 1e-6;
            b[c] -= 1e-6;
            let (ra, rb) = (run(a), run(b));
            for r in 0..3 {
                let fd = (ra[r] - rb[r]) / 2e-6;
                assert!((q[c][r] - fd).abs() < 1e-6 * (1.0 + fd.abs()), "d x[{r}] / d x0[{c}]: {} against {fd}", q[c][r]);
            }
        }
    }

    /// The Lorenz step is fourth order.
    #[test]
    fn the_lorenz_step_is_fourth_order() {
        let l = Lorenz::LORENZ_1963;
        // Over half a time unit from (1, 1, 1); measured ratios 16.99 and 16.52.
        let run = |h: f64, n: usize| (0..n).fold([1.0, 1.0, 1.0], |p, _| l.step(p, h));
        let reference = run(0.5 / 65_536.0, 65_536);
        let err: Vec<f64> = [(0.005, 100), (0.0025, 200), (0.00125, 400)]
            .iter()
            .map(|&(h, n)| {
                let p = run(h, n);
                (0..3).map(|i| (p[i] - reference[i]).abs()).fold(0.0, f64::max)
            })
            .collect();
        for pair in err.windows(2) {
            let ratio = pair[0] / pair[1];
            assert!((15.0..18.0).contains(&ratio), "{err:?}: {ratio}");
        }
    }
}
