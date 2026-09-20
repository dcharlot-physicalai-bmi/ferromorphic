//! FORCE learning: taming a chaotic recurrent network by recursive least squares on its readout,
//! with the readout fed back into the network while it is being learned.
//!
//! # What the mechanism is
//!
//! A recurrent network with strong random coupling is chaotic: it generates rich activity of its
//! own, and that activity never repeats. Sussillo and Abbott (*Generating coherent patterns of
//! activity from chaotic neural networks*, Neuron 63(4):544–557, 2009) showed that such a network
//! can be made to produce a chosen output — a periodic wave, a recorded movement — by training
//! ONLY the linear readout, provided two things are done together. The readout `z = wᵀ r` is fed
//! back into the network, so the network's activity depends on what is being learned; and the
//! readout weights are updated by **recursive least squares** fast enough that, from the first
//! few steps, the output error is small. Because the error is small the fed-back signal is
//! already close to the target, so the network is driven by (nearly) the signal it is meant to
//! produce, and the chaos is suppressed for as long as learning holds the error down. When
//! learning stops the weights hold the pattern on their own. The name is First-Order Reduced and
//! Controlled Error: the error is kept small throughout, rather than reduced gradually.
//!
//! The network is the standard rate model, `τ ẋ = −x + g J r + J_fb z`, `r = tanh(x)`, with `J` a
//! random matrix of variance `1/n` and `g` the gain that decides whether it is chaotic.
//!
//! # Why it is in a neuromorphic crate
//!
//! It is the method behind trained spiking reservoirs — Nicola and Clopath (*Supervised learning
//! in spiking neural networks with FORCE training*, Nature Communications 8:2208, 2017) apply
//! the same update to filtered spike trains — and the learning rule is one a chip can run: the
//! update at each step uses the current activity, the current error and a running matrix, never a
//! stored history. It also sits beside [`crate::reservoir`], which trains the same kind of
//! readout in one batch solve; the first closed form below says the two are the same
//! computation.
//!
//! # The closed forms this module is checked against
//!
//! - **Recursive least squares IS ridge regression.** After any sequence of updates on pairs
//!   `(r_t, f_t)`, starting from `w = 0` and `P = I/α`, the weights are exactly
//!   `(Σ r rᵀ + α I)⁻¹ Σ r f` and `P` is exactly `(Σ r rᵀ + α I)⁻¹`. Checked against a Cholesky
//!   solve of the batch problem ([`crate::reservoir::cholesky`]), which shares no code with the
//!   recursion.
//! - **One update's effect on the error.** With `e₋` the error before the update and
//!   `e₊ = w₊ᵀ r − f` the error on the same input after it, `e₊ = e₋ / (1 + rᵀ P r)`: the error
//!   shrinks by a known factor at every step, which is the "controlled error" of the name.
//!   [`RlsStep`] reports both, and the test recomputes `e₊` from the new weights.
//! - **The first update.** From `w = 0`, `w = f r / (α + |r|²)` (Sherman–Morrison for one term).
//! - **Below `g = 1` the network is silent.** The quiescent state has Jacobian `(−I + gJ)/τ`, and
//!   `J`'s spectrum fills the unit disc (Girko's circular law), so for `g < 1` every mode decays
//!   at a rate of at least about `(1 − g)/τ`; above it the network sustains activity. Whether
//!   that activity is CHAOTIC is, at a few hundred units, a property of the sample: measured
//!   here, one 200-unit network at `g = 1.5` pulls two copies started `10⁻⁹` apart back together
//!   to rounding, and the same seed at `g = 3` drives them to order one.
//! - **One Euler step** of the network, against the arithmetic done by hand on three neurons.
//!
//! # What this module has NOT reproduced
//!
//! - Spiking FORCE (Nicola and Clopath). The readout update is the same; the network here is the
//!   rate model of the original paper.
//! - Learning inside the network (the paper's internal-FORCE variants), sparse `J`, and the
//!   paper's demonstrations beyond a periodic target. The figures asserted for the learned sine
//!   are measured on this implementation at its seed, and labelled as such.
//! - A map of where it works. Measured on 300 units with `α = 1`: at `g = 1.5` all three seeds
//!   tried learned the sine (free-running rms error 0.0002 to 0.0027); at `g = 2` two of three
//!   did not (rms near one), and `a_network_that_is_too_chaotic_is_not_tamed` pins one of them.

use crate::rng::Rng;
use core::fmt;

/// The largest network this module will build: `P` is `n × n`.
pub const MAX_UNITS: usize = 4096;

/// What went wrong, named rather than guessed around.
#[derive(Debug, Clone, PartialEq)]
pub enum ForceError {
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
    },
    /// A vector of the wrong length.
    Shape {
        /// Which vector.
        what: &'static str,
        /// Length supplied.
        got: usize,
        /// Length required.
        want: usize,
    },
}

impl fmt::Display for ForceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfRange { what, value, low, high } => write!(f, "{what} = {value} is outside [{low}, {high}]"),
            Self::NonFinite { what } => write!(f, "{what} is not finite"),
            Self::Shape { what, got, want } => write!(f, "{what} has {got} entries, not {want}"),
        }
    }
}

impl std::error::Error for ForceError {}

fn positive(what: &'static str, v: f64) -> Result<f64, ForceError> {
    if v.is_finite() && v > 0.0 {
        Ok(v)
    } else {
        Err(ForceError::OutOfRange { what, value: v, low: f64::MIN_POSITIVE, high: f64::INFINITY })
    }
}

fn units(n: usize) -> Result<usize, ForceError> {
    if n == 0 || n > MAX_UNITS {
        return Err(ForceError::OutOfRange { what: "n", value: n as f64, low: 1.0, high: MAX_UNITS as f64 });
    }
    Ok(n)
}

fn all_finite(what: &'static str, v: &[f64], want: usize) -> Result<(), ForceError> {
    if v.len() != want {
        return Err(ForceError::Shape { what, got: v.len(), want });
    }
    if v.iter().all(|x| x.is_finite()) { Ok(()) } else { Err(ForceError::NonFinite { what }) }
}

// ---------------------------------------------------------------------------------------------
// Recursive least squares
// ---------------------------------------------------------------------------------------------

/// What one update did.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RlsStep {
    /// `wᵀ r − f` with the weights as they were.
    pub error_before: f64,
    /// `wᵀ r − f` on the same input with the weights as they are now: `error_before / (1 + rᵀPr)`.
    pub error_after: f64,
}

/// A linear readout `z = wᵀ r` learned by recursive least squares.
#[derive(Debug, Clone, PartialEq)]
pub struct Rls {
    /// The weights.
    pub w: Vec<f64>,
    /// The running inverse `(Σ r rᵀ + α I)⁻¹`, row-major, `n × n`.
    pub p: Vec<f64>,
}

impl Rls {
    /// A readout of `n` inputs with zero weights and `P = I/α`. `α` is the ridge penalty of the
    /// regression being solved, so a small `α` learns fast and a large one slowly.
    ///
    /// # Errors
    ///
    /// [`ForceError::OutOfRange`] for `n` of zero or past [`MAX_UNITS`], or a non-positive `alpha`.
    pub fn new(n: usize, alpha: f64) -> Result<Self, ForceError> {
        let n = units(n)?;
        let alpha = positive("alpha", alpha)?;
        let mut p = vec![0.0; n * n];
        for i in 0..n {
            p[i * n + i] = 1.0 / alpha;
        }
        Ok(Self { w: vec![0.0; n], p })
    }

    /// How many inputs.
    #[must_use]
    pub fn n(&self) -> usize {
        self.w.len()
    }

    /// `wᵀ r`. Zero for an `r` of the wrong length is not offered: the lengths are the caller's
    /// to get right, and [`Rls::update`] checks them.
    #[must_use]
    pub fn output(&self, r: &[f64]) -> f64 {
        self.w.iter().zip(r).map(|(w, r)| w * r).sum()
    }

    /// One recursive-least-squares update toward `target` on input `r`:
    /// `k = P r`, `c = 1/(1 + rᵀk)`, `P ← P − c k kᵀ`, `w ← w − e c k` with `e = wᵀ r − target`
    /// taken BEFORE the update.
    ///
    /// # Errors
    ///
    /// [`ForceError::Shape`] or [`ForceError::NonFinite`] for a bad `r` or `target`.
    pub fn update(&mut self, r: &[f64], target: f64) -> Result<RlsStep, ForceError> {
        let n = self.n();
        all_finite("r", r, n)?;
        if !target.is_finite() {
            return Err(ForceError::NonFinite { what: "target" });
        }
        let k: Vec<f64> = (0..n).map(|i| self.p[i * n..(i + 1) * n].iter().zip(r).map(|(p, r)| p * r).sum()).collect();
        let c = 1.0 / (1.0 + r.iter().zip(&k).map(|(r, k)| r * k).sum::<f64>());
        let error_before = self.output(r) - target;
        for i in 0..n {
            let scaled = c * k[i];
            for j in 0..n {
                self.p[i * n + j] -= scaled * k[j];
            }
            self.w[i] -= error_before * scaled;
        }
        Ok(RlsStep { error_before, error_after: error_before * c })
    }
}

// ---------------------------------------------------------------------------------------------
// The network
// ---------------------------------------------------------------------------------------------

/// A standard normal deviate by Box–Muller, from the crate's generator.
fn normal(rng: &mut Rng) -> f64 {
    let u = (1.0 - rng.next_f64()).max(f64::MIN_POSITIVE);
    let v = rng.next_f64();
    (-2.0 * u.ln()).sqrt() * (core::f64::consts::TAU * v).cos()
}

/// A rate network with its readout fed back: `τ ẋ = −x + g J r + J_fb z`, `r = tanh(x)`,
/// `z = wᵀ r`, stepped by forward Euler.
#[derive(Debug, Clone, PartialEq)]
pub struct Network {
    /// Time constant `τ`, seconds.
    pub tau: f64,
    /// Coupling gain `g`. Above one the free network is chaotic.
    pub g: f64,
    /// The random coupling `J`, row-major `n × n`, entries of variance `1/n`.
    pub j: Vec<f64>,
    /// The feedback weights `J_fb`, uniform in `[−1, 1]`.
    pub feedback: Vec<f64>,
    /// The state `x`.
    pub x: Vec<f64>,
    /// The rates `r = tanh(x)`.
    pub r: Vec<f64>,
    /// The readout and its learner.
    pub readout: Rls,
    /// The readout's value at the last step, which is what is being fed back.
    pub z: f64,
}

impl Network {
    /// A random network of `n` units, its state drawn at half-unit scale.
    ///
    /// # Errors
    ///
    /// [`ForceError::OutOfRange`] for a bad `n`, a non-positive `tau` or `alpha`, or a negative or
    /// non-finite `g`.
    pub fn random(n: usize, g: f64, tau: f64, alpha: f64, seed: u64) -> Result<Self, ForceError> {
        let n = units(n)?;
        let tau = positive("tau", tau)?;
        if !(g >= 0.0) || !g.is_finite() {
            return Err(ForceError::OutOfRange { what: "g", value: g, low: 0.0, high: f64::INFINITY });
        }
        let readout = Rls::new(n, alpha)?;
        let mut rng = Rng::new(seed);
        let scale = 1.0 / (n as f64).sqrt();
        let j = (0..n * n).map(|_| scale * normal(&mut rng)).collect();
        let feedback = (0..n).map(|_| 2.0 * rng.next_f64() - 1.0).collect();
        let x: Vec<f64> = (0..n).map(|_| 0.5 * normal(&mut rng)).collect();
        let r = x.iter().map(|x| x.tanh()).collect();
        Ok(Self { tau, g, j, feedback, x, r, readout, z: 0.0 })
    }

    /// How many units.
    #[must_use]
    pub fn n(&self) -> usize {
        self.x.len()
    }

    /// One Euler step of length `dt`, feeding back the readout of the step before. Returns the
    /// new readout `z`.
    ///
    /// # Errors
    ///
    /// [`ForceError::OutOfRange`] for a non-positive `dt`; [`ForceError::NonFinite`] if the state
    /// has overflowed.
    pub fn step(&mut self, dt: f64) -> Result<f64, ForceError> {
        let dt = positive("dt", dt)?;
        let n = self.n();
        let a = dt / self.tau;
        for i in 0..n {
            let recurrent: f64 = self.j[i * n..(i + 1) * n].iter().zip(&self.r).map(|(j, r)| j * r).sum();
            self.x[i] += a * (-self.x[i] + self.g * recurrent + self.feedback[i] * self.z);
        }
        if !self.x.iter().all(|x| x.is_finite()) {
            return Err(ForceError::NonFinite { what: "x" });
        }
        for i in 0..n {
            self.r[i] = self.x[i].tanh();
        }
        self.z = self.readout.output(&self.r);
        Ok(self.z)
    }

    /// One step, then one FORCE update of the readout toward `target`; the fed-back `z` is
    /// replaced by the readout AFTER the update, so the next step is driven by the corrected
    /// output.
    ///
    /// # Errors
    ///
    /// As [`Network::step`] and [`Rls::update`].
    pub fn train_step(&mut self, dt: f64, target: f64) -> Result<RlsStep, ForceError> {
        self.step(dt)?;
        let step = self.readout.update(&self.r, target)?;
        self.z = self.readout.output(&self.r);
        Ok(step)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reservoir::cholesky;

    fn samples(n: usize, count: usize, seed: u64) -> Vec<(Vec<f64>, f64)> {
        let mut rng = Rng::new(seed);
        (0..count).map(|_| ((0..n).map(|_| normal(&mut rng)).collect(), normal(&mut rng))).collect()
    }

    #[test]
    fn recursive_least_squares_is_ridge_regression_at_every_step() {
        // Fewer samples than inputs, as many, and many more: the identity does not care.
        let (n, alpha) = (7usize, 0.3);
        let data = samples(n, 40, 11);
        let mut rls = Rls::new(n, alpha).unwrap();
        for (t, (r, f)) in data.iter().enumerate() {
            rls.update(r, *f).unwrap();
            if ![0usize, 3, 6, 39].contains(&t) {
                continue;
            }
            // The batch problem, solved by a Cholesky factorisation that knows nothing of the recursion.
            let mut a = vec![0.0; n * n];
            let mut b = vec![0.0; n];
            for i in 0..n {
                a[i * n + i] = alpha;
            }
            for (r, f) in &data[..=t] {
                for i in 0..n {
                    b[i] += r[i] * f;
                    for j in 0..n {
                        a[i * n + j] += r[i] * r[j];
                    }
                }
            }
            let factor = cholesky(&a, n, 0.0).unwrap();
            let w = factor.solve(&b).unwrap();
            for i in 0..n {
                assert!((rls.w[i] - w[i]).abs() < 1e-11, "after {} samples w[{i}] = {} against {}", t + 1, rls.w[i], w[i]);
            }
            // And P is that matrix's inverse: P A = I.
            for i in 0..n {
                for j in 0..n {
                    let pa: f64 = (0..n).map(|k| rls.p[i * n + k] * a[k * n + j]).sum();
                    assert!((pa - f64::from(u8::from(i == j))).abs() < 1e-11, "(P A)[{i}][{j}] = {pa}");
                }
            }
        }
    }

    #[test]
    fn each_update_divides_the_error_on_its_own_input_by_one_plus_r_p_r() {
        let n = 5usize;
        let mut rls = Rls::new(n, 2.0).unwrap();
        // The first update in closed form: w = f r / (α + |r|²).
        let (r, f) = (vec![1.0, -2.0, 0.5, 0.0, 3.0], 1.7);
        let norm2: f64 = r.iter().map(|x| x * x).sum();
        let step = rls.update(&r, f).unwrap();
        assert_eq!(step.error_before, -1.7);
        for i in 0..n {
            assert!((rls.w[i] - f * r[i] / (2.0 + norm2)).abs() < 1e-15);
        }
        assert!((step.error_after - (-1.7) * 2.0 / (2.0 + norm2)).abs() < 1e-15);
        for (r, f) in samples(n, 30, 5) {
            let k: Vec<f64> = (0..n).map(|i| (0..n).map(|j| rls.p[i * n + j] * r[j]).sum()).collect();
            let rpr: f64 = r.iter().zip(&k).map(|(r, k)| r * k).sum();
            let before = rls.output(&r) - f;
            let step = rls.update(&r, f).unwrap();
            assert_eq!(step.error_before, before);
            let after = rls.output(&r) - f;
            assert!((step.error_after - after).abs() < 1e-13, "reported {} but the new weights give {after}", step.error_after);
            assert!((after - before / (1.0 + rpr)).abs() < 1e-13);
            assert!(after.abs() < before.abs());
        }
    }

    #[test]
    fn one_euler_step_is_the_arithmetic_it_claims() {
        let mut net = Network::random(3, 1.5, 0.01, 1.0, 1).unwrap();
        net.j = vec![0.0, 1.0, -1.0, 0.5, 0.0, 2.0, -0.25, 0.75, 0.0];
        net.feedback = vec![1.0, -1.0, 0.5];
        net.x = vec![0.1, -0.2, 0.3];
        net.r = net.x.iter().map(|x| x.tanh()).collect();
        net.readout.w = vec![2.0, 0.0, -1.0];
        net.z = 0.4;
        let r = [0.1f64.tanh(), (-0.2f64).tanh(), 0.3f64.tanh()];
        let a = 0.001 / 0.01;
        let want = [
            0.1 + a * (-0.1 + 1.5 * (r[1] - r[2]) + 0.4),
            -0.2 + a * (0.2 + 1.5 * (0.5 * r[0] + 2.0 * r[2]) - 0.4),
            0.3 + a * (-0.3 + 1.5 * (-0.25 * r[0] + 0.75 * r[1]) + 0.2),
        ];
        let z = net.step(0.001).unwrap();
        for i in 0..3 {
            assert!((net.x[i] - want[i]).abs() < 1e-15, "x[{i}] = {} against {}", net.x[i], want[i]);
            assert_eq!(net.r[i], want[i].tanh());
        }
        assert!((z - (2.0 * want[0].tanh() - want[2].tanh())).abs() < 1e-15 && z == net.z);
    }

    fn activity(net: &Network) -> f64 {
        (net.r.iter().map(|r| r * r).sum::<f64>() / net.n() as f64).sqrt()
    }

    #[test]
    fn below_unit_gain_the_free_network_falls_silent_and_above_it_it_is_chaotic() {
        // g = 0.5: every mode of (−I + gJ)/τ decays at about (1 − g)/τ or faster, so after 80 τ
        // the activity has fallen by e^{−40}, to rounding.
        let mut quiet = Network::random(200, 0.5, 0.01, 1.0, 3).unwrap();
        assert!(activity(&quiet) > 0.3);
        for _ in 0..8_000 {
            quiet.step(1e-4).unwrap();
        }
        assert!(activity(&quiet) < 1e-12, "at g = 0.5 the activity is still {}", activity(&quiet));
        // Above one the activity sustains itself. Whether it is CHAOTIC is a property of the
        // sample at this size, and both outcomes are MEASURED here: at g = 3 two copies started
        // 1e-9 apart end up unrelated, while this same seed at g = 1.5 settles into a state that
        // pulls them back together to rounding.
        let apart_after = |g: f64, steps: usize| {
            let mut a = Network::random(200, g, 0.01, 1.0, 3).unwrap();
            let mut b = a.clone();
            b.x[0] += 1e-9;
            for _ in 0..steps {
                a.step(1e-3).unwrap();
                b.step(1e-3).unwrap();
            }
            (activity(&a), (a.x.iter().zip(&b.x).map(|(a, b)| (a - b) * (a - b)).sum::<f64>() / 200.0).sqrt())
        };
        let (active, apart) = apart_after(3.0, 6_000);
        assert!(active > 0.5 && apart > 1.0, "at g = 3: activity {active}, copies {apart} apart");
        let (active, apart) = apart_after(1.5, 6_000);
        assert!(active > 0.3 && apart < 1e-12, "at g = 1.5: activity {active}, copies {apart} apart");
    }

    /// Two copies of `net`, one nudged by 1e-9, run free for `steps`: how far apart they end.
    fn separation(net: &Network, steps: usize) -> f64 {
        let (mut a, mut b) = (net.clone(), net.clone());
        b.x[0] += 1e-9;
        for _ in 0..steps {
            a.step(1e-3).unwrap();
            b.step(1e-3).unwrap();
        }
        (a.x.iter().zip(&b.x).map(|(a, b)| (a - b) * (a - b)).sum::<f64>() / a.n() as f64).sqrt()
    }

    /// Train `net` on a 2 Hz sine for ten seconds; returns the largest error met in the second
    /// half of training and the rms error over five free-running periods afterwards.
    fn learn_a_sine(net: &mut Network) -> (f64, f64) {
        let (dt, period) = (1e-3, 0.5);
        let target = |step: usize| (core::f64::consts::TAU * step as f64 * dt / period).sin();
        let (steps, test_steps) = (10_000usize, 2_500usize);
        let mut during = 0.0f64;
        for s in 0..steps {
            let step = net.train_step(dt, target(s + 1)).unwrap();
            if s >= steps / 2 {
                during = during.max(step.error_before.abs());
            }
        }
        let mut sum = 0.0;
        for s in 0..test_steps {
            let e = net.step(dt).unwrap() - target(steps + s + 1);
            sum += e * e;
        }
        (during, (sum / test_steps as f64).sqrt())
    }

    #[test]
    fn force_learning_makes_a_chaotic_network_hold_a_sine_on_its_own() {
        let mut net = Network::random(300, 1.5, 0.01, 1.0, 7).unwrap();
        // This sample IS chaotic before learning (MEASURED: copies 1e-9 apart reach 0.12).
        let before = separation(&net, 6_000);
        assert!(before > 0.01, "the untrained network only separated two copies to {before}");
        let (during, rms) = learn_a_sine(&mut net);
        // Controlled error: through the second half of training it never left a fraction of a
        // percent (MEASURED 0.0016); and with learning off the weights alone carry the pattern
        // for five more periods (MEASURED rms 0.0017, against the untrained 1/√2).
        assert!(during < 0.01, "during training the error reached {during}");
        assert!(rms < 0.01, "free-running rms error {rms}");
        // And the chaos is gone: the trained network pulls nearby copies together.
        let after = separation(&net, 6_000);
        assert!(after < 1e-6, "the trained network still separates two copies to {after}");
    }

    #[test]
    fn what_is_fed_back_during_training_is_the_readout_after_the_update() {
        // This is the detail FORCE turns on: the network is driven by the CORRECTED output, so
        // the error it sees is the small one. After `train_step` the stored `z` is the readout of
        // the present weights — `error_after` above the target — and not the one from before.
        let mut net = Network::random(60, 1.5, 0.01, 1.0, 13).unwrap();
        for s in 0..400 {
            let target = (0.05 * s as f64).sin();
            let step = net.train_step(1e-3, target).unwrap();
            assert!((net.z - net.readout.output(&net.r)).abs() < 1e-18, "the fed-back z is not the current readout");
            assert!((net.z - (target + step.error_after)).abs() < 1e-12, "z = {} for target {target} and error {}", net.z, step.error_after);
            if s > 10 {
                assert!(step.error_after.abs() < step.error_before.abs(), "the update did not reduce the error it was given");
                assert!((net.z - (target + step.error_before)).abs() > 1e-9, "the uncorrected readout would be indistinguishable here");
            }
        }
    }

    #[test]
    fn a_network_that_is_too_chaotic_is_not_tamed() {
        // The limit, MEASURED at the same size, seed and ridge: at g = 2 the error is not held
        // down during training (0.17) and the free-running output is unrelated to the target.
        let mut net = Network::random(300, 2.0, 0.01, 1.0, 7).unwrap();
        let (during, rms) = learn_a_sine(&mut net);
        assert!(during > 0.05 && rms > 0.5, "at g = 2: error during training {during}, free rms {rms}");
    }

    #[test]
    fn bad_parameters_and_bad_vectors_are_refused() {
        assert!(Rls::new(0, 1.0).is_err() && Rls::new(MAX_UNITS + 1, 1.0).is_err());
        assert!(Rls::new(3, 0.0).is_err() && Rls::new(3, f64::NAN).is_err());
        let mut rls = Rls::new(3, 1.0).unwrap();
        assert_eq!(rls.update(&[1.0, 2.0], 0.0), Err(ForceError::Shape { what: "r", got: 2, want: 3 }));
        assert_eq!(rls.update(&[1.0, f64::NAN, 0.0], 0.0), Err(ForceError::NonFinite { what: "r" }));
        assert_eq!(rls.update(&[1.0, 2.0, 3.0], f64::INFINITY), Err(ForceError::NonFinite { what: "target" }));
        assert_eq!(rls, Rls::new(3, 1.0).unwrap(), "a refused update must not move the learner");
        assert!(Network::random(10, -0.1, 0.01, 1.0, 1).is_err() && Network::random(10, 1.5, 0.0, 1.0, 1).is_err());
        assert!(Network::random(0, 1.5, 0.01, 1.0, 1).is_err() && Network::random(10, 1.5, 0.01, 0.0, 1).is_err());
        let mut net = Network::random(10, 1.5, 0.01, 1.0, 1).unwrap();
        assert!(net.step(0.0).is_err() && net.step(f64::NAN).is_err());
        net.x[0] = f64::MAX;
        net.g = f64::MAX;
        assert_eq!(net.step(1.0), Err(ForceError::NonFinite { what: "x" }));
        assert!(ForceError::Shape { what: "r", got: 2, want: 3 }.to_string().contains("2 entries"));
    }
}
