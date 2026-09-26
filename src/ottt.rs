//! Online training through time: the gradient of a spiking network computed FORWARD, in constant
//! memory, and the exact conditions under which it is the gradient backpropagation would have
//! given.
//!
//! # What the mechanism is
//!
//! Backpropagation through time keeps every step of the forward pass so it can walk back through
//! it. For a spiking network of `T` steps that is `O(T)` memory, which is the reason on-chip
//! learning is hard: the chip would have to store the run before it could learn from it.
//!
//! Xiao, Meng, Zhang, He and Lin (*Online training through time for spiking neural networks*,
//! Advances in Neural Information Processing Systems 35:20717-20730 (2022), `arXiv`:2210.04195v2)
//! compute the gradient forward instead. Their layer has ONE leak, with the reset inside it —
//! Eq. (2):
//!
//! ```text
//! u_i[t+1] = λ(u_i[t] − V_th s_i[t]) + Σ_j w_ij s_j[t] + b_i          s_i[t+1] = H(u_i[t+1] − V_th)
//! ```
//!
//! Holding the spikes fixed — they "do not apply surrogate derivatives" to the spike's own slope
//! in the temporal dependency (§4.1), which footnote 3 notes is consistent with implementations
//! that "detach the neuron reset operation from the computational graph" — the derivative of `u`
//! with respect to `W` is a single exponential of the presynaptic spikes, which the forward pass
//! carries along (§4.1, after Eq. (4)):
//!
//! ```text
//! â[t+1] = λ â[t] + s[t+1]          ∇_W L[t] = g_u[t] â[t]ᵀ
//! ```
//!
//! This module applies the same derivation to the current-based layer that
//! [`crate::surrogate::LifLayer`] runs, which filters the synaptic current as well:
//!
//! ```text
//! I[t] = α I[t−1] + W x[t]        U[t] = β U[t−1] + I[t] − θ s[t−1]        s[t] = Θ(U[t] − θ)
//! ```
//!
//! There the derivative of `U[t]` with respect to `W`, spikes held fixed, is a DOUBLE filter of the
//! input, one stage per leak:
//!
//! ```text
//! q[t] = α q[t−1] + x[t]          â[t] = β â[t−1] + q[t]          ∂U[t]/∂W_ji = â_i[t]
//! ```
//!
//! With `α = 0` the current has no memory, `q[t] = x[t]`, and the second stage is the paper's
//! trace with `λ = β`;
//! `with_no_synaptic_filter_the_trace_is_the_papers_single_exponential` checks that bit for bit.
//! Two differences survive the reduction. This layer has no bias `b`. And the paper's reset,
//! `−λ V_th s[t]` once Eq. (2) is expanded, sits INSIDE the leak, where this layer's `−θ s[t−1]`
//! sits outside it. Since OTTT drops the reset path, that moves the membrane — so the spikes and
//! the slope `σ′(U − θ)` — but never the trace.
//!
//! So the gradient of a loss that is a sum over steps is `Σ_t (∂L/∂s[t]) σ′(U[t] − θ) â[t]`, and
//! every term of it is available AT step `t`. The memory is the trace: `2 · n_in` numbers for this
//! layer's two stages, `n_in` for the paper's one, and never `T × n_in`.
//!
//! What this drops is the paths that run through the spikes themselves: the reset `−θ s[t−1]` and,
//! in a recurrent layer, `V s[t−1]`. This module keeps both switchable so that the exact case and
//! the approximate one can be told apart, and measures the second rather than asserting it.
//!
//! **Correction.** This doc used to say that Xiao et al. "observe that for the standard spiking
//! layer" the two-leak recursion above holds, and credited them with the `q`/`â` double filter.
//! Their paper has neither. Its §3.1 considers "a simple current model" with a single leak,
//! "`λ < 1` is a leaky term (typically taken as `1 − 1/τ_m`)", and no filter on the synaptic
//! current; its trace is `â^l[t] = Σ_{τ≤t} λ^{t−τ} s^l[τ]`. The authors' released code agrees:
//! `OnlineLIFNode` in `modules/neuron.py` of `github.com/pkuxmq/OTTT-SNN` updates
//! `self.v = self.v.detach() * (1 - 1. / self.tau) + x` and tracks
//! `rate_tracking * (1 - 1. / self.tau) + spike`. The two-leak layer and its double filter are
//! this crate's generalisation, and correcting the attribution changed no code.
//!
//! # Why it is in a neuromorphic crate
//!
//! Constant memory in time is the property that decides whether a learning rule can run on the
//! device that is doing the inference. It sits between [`crate::eprop`], which carries a trace per
//! SYNAPSE and a broadcast error, and [`crate::decolle`], which drops time altogether and gives
//! each layer its own loss; and it is checked against [`crate::surrogate`], which does the same
//! layer's backpropagation through time the expensive way.
//!
//! # The closed forms this module is checked against
//!
//! - **The forward pass is [`crate::surrogate::LifLayer`]'s**, step for step and to the last bit,
//!   for the same weights and input. That is what makes the comparison below a comparison of
//!   GRADIENTS rather than of two different networks.
//! - **The trace is a double exponential.** One input spike at step 0 gives
//!   `â[t] = (α^{t+1} − β^{t+1})/(α − β)` for `α ≠ β` and `â[t] = (t + 1) α^t` for `α = β`, so
//!   `â[0] = 1` — that is [`Online::impulse`] at `t + 1` — checked against both closed forms and
//!   against the recursion. At `α = 0` it is `β^t`, the paper's single exponential. ⚠ CORRECTED:
//!   this line used to read `(α^{t} − β^{t})·α/(α − β)` and `t α^{t−1}·α`. Counted as
//!   [`Online::impulse`] counts `t`, that is a factor `α` too large at every step; counted from
//!   the input's own step, as the line said, it also gave `â[0] = 0` where the recursion gives
//!   one.
//! - **Without the reset path, OTTT IS the gradient.** With `reset: false` and a differentiable
//!   spike ([`Layer::smooth`]) the online gradient equals central finite differences of the loss
//!   on every weight — it is not an approximation of the gradient, it is the gradient, computed
//!   forwards. With the hard step a real neuron emits, the loss is piecewise constant and every
//!   finite difference of it is zero, which is why the check needs the smooth spike and why the
//!   result for the hard one is a surrogate gradient, exactly as in [`crate::surrogate`].
//! - **It also equals backpropagation through time** on the same layer, which is the
//!   implementation-level statement of the same thing: `Layer::bptt` walks backwards and gets the
//!   same numbers.
//! - **With the reset path, it does not**, and the gap is measured: a cosine and a relative norm
//!   against the true gradient, reported by [`Comparison`], and the direction still descends the
//!   loss.
//! - **Memory.** [`Online`] holds `n_in + n_rec` numbers whatever `T` is, and the test runs the
//!   same network at two lengths and checks the state's size did not move.
//!
//! # What this module has NOT reproduced
//!
//! - The paper's benchmarks (CIFAR, DVS-CIFAR, `ImageNet`), its `OTTT_O` variant's convergence
//!   analysis, and its equivalence argument with spike representation methods.
//! - Multi-layer credit assignment. This is one layer with a readout; a stack would need the
//!   error at each layer's output, which is exactly what [`crate::alignment`]'s feedback matrices
//!   or [`crate::decolle`]'s local readouts supply.
//! - The surrogate's own justification. `σ′` here is whatever [`crate::surrogate::Surrogate`] is
//!   handed, and for a hard threshold the result is a surrogate gradient, not a gradient — see
//!   that module.

use crate::rng::Rng;
use crate::surrogate::Surrogate;
use core::fmt;

/// The most weights a layer may hold.
pub const MAX_WEIGHTS: usize = 1 << 22;

/// What went wrong, named rather than guessed around.
#[derive(Debug, Clone, PartialEq)]
pub enum OtttError {
    /// A layer of zero width, or one past [`MAX_WEIGHTS`].
    BadShape,
    /// A parameter outside its range.
    OutOfRange {
        /// Which parameter.
        what: &'static str,
        /// Value supplied.
        value: f64,
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
    /// A `NaN` or infinity.
    NonFinite {
        /// Which quantity.
        what: &'static str,
    },
}

impl fmt::Display for OtttError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadShape => f.write_str("a layer needs at least one input, one neuron and one readout unit"),
            Self::OutOfRange { what, value } => write!(f, "{what} = {value} is out of range"),
            Self::Shape { what, got, want } => write!(f, "{what} has {got} entries, not {want}"),
            Self::NonFinite { what } => write!(f, "{what} is not finite"),
        }
    }
}

impl std::error::Error for OtttError {}

fn checked(what: &'static str, v: &[f64], want: usize) -> Result<(), OtttError> {
    if v.len() != want {
        return Err(OtttError::Shape { what, got: v.len(), want });
    }
    if v.iter().all(|x| x.is_finite()) { Ok(()) } else { Err(OtttError::NonFinite { what }) }
}

/// One spiking layer with a leaky readout — the same recursion
/// [`crate::surrogate::LifLayer`] runs, with the two paths OTTT drops made switchable.
#[derive(Debug, Clone, PartialEq)]
pub struct Layer {
    /// Inputs.
    pub n_in: usize,
    /// Neurons.
    pub n_rec: usize,
    /// Readout units.
    pub n_out: usize,
    /// Synaptic decay `α`, in `[0, 1)`.
    pub alpha: f64,
    /// Membrane decay `β`, in `[0, 1)`.
    pub beta: f64,
    /// Readout decay `κ`, in `[0, 1)`.
    pub kappa: f64,
    /// Threshold `θ`.
    pub theta: f64,
    /// Whether a spike subtracts `θ` from the membrane on the NEXT step.
    pub reset: bool,
    /// Whether the neuron emits the surrogate's ANTIDERIVATIVE — a smooth, differentiable spike
    /// whose derivative really is [`crate::surrogate::Surrogate::backward`] — instead of the hard
    /// step a real neuron emits.
    ///
    /// False is the network you would run. True is the network whose gradient can be checked
    /// against a finite difference, because the loss of a hard step is piecewise constant and its
    /// finite differences are all zero. The gradient code is identical either way: with a hard
    /// step it is a surrogate gradient, exactly as in [`crate::surrogate`].
    pub smooth: bool,
    /// Input weights, row-major `n_rec × n_in`.
    pub w: Vec<f64>,
    /// Readout weights, row-major `n_out × n_rec`.
    pub r: Vec<f64>,
}

/// Everything one forward pass produced, for the layer that walks backwards.
#[derive(Debug, Clone, PartialEq)]
pub struct Trace {
    /// Steps.
    pub t_steps: usize,
    /// Membrane potentials, `t_steps × n_rec`.
    pub u: Vec<f64>,
    /// Spikes, `t_steps × n_rec`.
    pub s: Vec<f64>,
    /// The double-exponential input trace at each step, `t_steps × n_in` — what OTTT carries.
    pub trace: Vec<f64>,
    /// Readout, `t_steps × n_out`.
    pub y: Vec<f64>,
}

/// How an online gradient compares with the true one.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Comparison {
    /// The cosine of the angle between them.
    pub cosine: f64,
    /// `|online − true| / |true|`.
    pub relative: f64,
}

impl Layer {
    /// A random layer. Weights are uniform in `±scale/√n_in`, the readout in `±1/√n_rec`.
    ///
    /// # Errors
    ///
    /// [`OtttError::BadShape`] or [`OtttError::OutOfRange`].
    pub fn random(n_in: usize, n_rec: usize, n_out: usize, alpha: f64, beta: f64, kappa: f64, theta: f64, scale: f64, seed: u64) -> Result<Self, OtttError> {
        if n_in == 0 || n_rec == 0 || n_out == 0 || n_in.saturating_mul(n_rec) > MAX_WEIGHTS || n_out.saturating_mul(n_rec) > MAX_WEIGHTS {
            return Err(OtttError::BadShape);
        }
        for (what, v) in [("alpha", alpha), ("beta", beta), ("kappa", kappa)] {
            if !(v >= 0.0) || !(v < 1.0) {
                return Err(OtttError::OutOfRange { what, value: v });
            }
        }
        for (what, v) in [("theta", theta), ("scale", scale)] {
            if !(v > 0.0) || !v.is_finite() {
                return Err(OtttError::OutOfRange { what, value: v });
            }
        }
        let mut rng = Rng::new(seed);
        let mut draw = |count: usize, s: f64| -> Vec<f64> { (0..count).map(|_| s * (2.0 * rng.next_f64() - 1.0)).collect() };
        let w = draw(n_rec * n_in, scale / (n_in as f64).sqrt());
        let r = draw(n_out * n_rec, 1.0 / (n_rec as f64).sqrt());
        Ok(Self { n_in, n_rec, n_out, alpha, beta, kappa, theta, reset: true, smooth: false, w, r })
    }

    /// Run `x` — `t_steps × n_in`, step-major — through the layer.
    ///
    /// # Errors
    ///
    /// [`OtttError::Shape`] for an `x` that is not a whole number of steps, or
    /// [`OtttError::NonFinite`] for a bad input or a state that overflowed.
    pub fn forward(&self, sur: &dyn Surrogate, x: &[f64]) -> Result<Trace, OtttError> {
        if x.is_empty() || !x.len().is_multiple_of(self.n_in) {
            return Err(OtttError::Shape { what: "x", got: x.len(), want: self.n_in });
        }
        if !x.iter().all(|v| v.is_finite()) {
            return Err(OtttError::NonFinite { what: "x" });
        }
        let (t_steps, nr, ni, no) = (x.len() / self.n_in, self.n_rec, self.n_in, self.n_out);
        let mut tr = Trace { t_steps, u: vec![0.0; t_steps * nr], s: vec![0.0; t_steps * nr], trace: vec![0.0; t_steps * ni], y: vec![0.0; t_steps * no] };
        let (mut i_syn, mut q) = (vec![0.0; nr], vec![0.0; ni]);
        let (mut u_prev, mut s_prev, mut a) = (vec![0.0; nr], vec![0.0; nr], vec![0.0; ni]);
        let mut y_prev = vec![0.0; no];
        for t in 0..t_steps {
            // The input trace, advanced with exactly the recursion the current follows.
            for j in 0..ni {
                q[j] = self.alpha * q[j] + x[t * ni + j];
                a[j] = self.beta * a[j] + q[j];
                tr.trace[t * ni + j] = a[j];
            }
            for j in 0..nr {
                let drive: f64 = self.w[j * ni..(j + 1) * ni].iter().zip(&x[t * ni..(t + 1) * ni]).map(|(w, x)| w * x).sum();
                i_syn[j] = self.alpha * i_syn[j] + drive;
                let reset = if self.reset { self.theta * s_prev[j] } else { 0.0 };
                let u = self.beta * u_prev[j] + i_syn[j] - reset;
                if !u.is_finite() {
                    return Err(OtttError::NonFinite { what: "u" });
                }
                tr.u[t * nr + j] = u;
                tr.s[t * nr + j] = if self.smooth { sur.antiderivative(u - self.theta) } else { sur.forward(u - self.theta) };
                u_prev[j] = u;
            }
            s_prev.copy_from_slice(&tr.s[t * nr..(t + 1) * nr]);
            for c in 0..no {
                // Accumulated in this order, one term at a time, because `surrogate::LifLayer`
                // accumulates it in this order and the two are required to agree to the last bit.
                let mut acc = self.kappa * y_prev[c];
                for j in 0..nr {
                    acc += self.r[c * nr + j] * tr.s[t * nr + j];
                }
                tr.y[t * no + c] = acc;
                y_prev[c] = acc;
            }
        }
        Ok(tr)
    }

    /// `½ Σ_t |y[t] − target|²` and its derivative with respect to each `y[t]`.
    ///
    /// # Errors
    ///
    /// [`OtttError::Shape`] or [`OtttError::NonFinite`] for a bad target.
    pub fn loss(&self, tr: &Trace, target: &[f64]) -> Result<(f64, Vec<f64>), OtttError> {
        checked("target", target, self.n_out)?;
        let mut dy = vec![0.0; tr.t_steps * self.n_out];
        let mut loss = 0.0;
        for t in 0..tr.t_steps {
            for c in 0..self.n_out {
                let e = tr.y[t * self.n_out + c] - target[c];
                loss += 0.5 * e * e;
                dy[t * self.n_out + c] = e;
            }
        }
        Ok((loss, dy))
    }

    /// The ONLINE gradient of [`Layer::loss`] with respect to `w`, accumulated forward in time:
    /// at each step, `(Rᵀ e[t])_j · σ′(U[t] − θ) · â_i[t]`.
    ///
    /// Constant in memory: nothing from step `t` is kept once step `t + 1` has run, which is what
    /// [`Online`] makes explicit.
    ///
    /// # Errors
    ///
    /// [`OtttError::Shape`] or [`OtttError::NonFinite`].
    pub fn online_gradient(&self, sur: &dyn Surrogate, tr: &Trace, dy: &[f64]) -> Result<Vec<f64>, OtttError> {
        checked("dy", dy, tr.t_steps * self.n_out)?;
        let (nr, ni, no) = (self.n_rec, self.n_in, self.n_out);
        let mut g = vec![0.0; nr * ni];
        // The readout leak carries an error backwards in time by itself; OTTT's own accumulator
        // for it is a forward-running sum, which is what this is.
        let mut carried = vec![0.0; no];
        for t in (0..tr.t_steps).rev() {
            for c in 0..no {
                carried[c] = self.kappa * carried[c] + dy[t * no + c];
            }
            for j in 0..nr {
                let broadcast: f64 = (0..no).map(|c| self.r[c * nr + j] * carried[c]).sum();
                let local = broadcast * sur.backward(tr.u[t * nr + j] - self.theta);
                for i in 0..ni {
                    g[j * ni + i] += local * tr.trace[t * ni + i];
                }
            }
        }
        Ok(g)
    }

    /// Backpropagation through time for the same layer and the same loss: the reference the
    /// online gradient is compared with.
    ///
    /// # Errors
    ///
    /// As [`Layer::online_gradient`].
    pub fn bptt(&self, sur: &dyn Surrogate, x: &[f64], tr: &Trace, dy: &[f64]) -> Result<Vec<f64>, OtttError> {
        checked("dy", dy, tr.t_steps * self.n_out)?;
        checked("x", x, tr.t_steps * self.n_in)?;
        let (nr, ni, no) = (self.n_rec, self.n_in, self.n_out);
        let mut g = vec![0.0; nr * ni];
        let (mut gy, mut gu, mut gi) = (vec![0.0; no], vec![0.0; nr], vec![0.0; nr]);
        let mut gs_next = vec![0.0; nr];
        for t in (0..tr.t_steps).rev() {
            for c in 0..no {
                gy[c] = self.kappa * gy[c] + dy[t * no + c];
            }
            for j in 0..nr {
                // The spike is read by the readout now, and — if the reset is on — by the
                // membrane one step later.
                let mut gs: f64 = (0..no).map(|c| self.r[c * nr + j] * gy[c]).sum();
                gs += gs_next[j];
                let local = gs * sur.backward(tr.u[t * nr + j] - self.theta);
                // The membrane's own leak carries gradient back a step; so does the current's.
                gu[j] = local + self.beta * gu[j];
                gi[j] = gu[j] + self.alpha * gi[j];
                for i in 0..ni {
                    g[j * ni + i] += gi[j] * x[t * ni + i];
                }
            }
            for j in 0..nr {
                gs_next[j] = if self.reset { -self.theta * gu[j] } else { 0.0 };
            }
        }
        Ok(g)
    }

    /// Move the weights against a gradient.
    ///
    /// # Errors
    ///
    /// [`OtttError::Shape`] or [`OtttError::NonFinite`].
    pub fn apply(&mut self, g: &[f64], rate: f64) -> Result<(), OtttError> {
        if !rate.is_finite() {
            return Err(OtttError::NonFinite { what: "rate" });
        }
        checked("gradient", g, self.w.len())?;
        self.w.iter_mut().zip(g).for_each(|(w, g)| *w -= rate * g);
        Ok(())
    }
}

/// The state OTTT carries between steps for this module's current-based layer: two filters of the
/// input and nothing else. The single-leak layer of Xiao et al. needs only the second; the first is
/// this crate's, for the synaptic current (see the module doc).
///
/// Its size is `2 · n_in`, whatever the run's length — the point of the method.
#[derive(Debug, Clone, PartialEq)]
pub struct Online {
    /// The first filter, `q[t] = α q[t−1] + x[t]`. At `α = 0` it is the input itself.
    pub q: Vec<f64>,
    /// The second, `â[t] = β â[t−1] + q[t]`: the trace the gradient multiplies, and at `α = 0`
    /// the paper's `â[t+1] = λ â[t] + s[t+1]` with `λ = β`.
    pub a: Vec<f64>,
    /// `α`.
    pub alpha: f64,
    /// `β`.
    pub beta: f64,
}

impl Online {
    /// A trace at rest.
    ///
    /// # Errors
    ///
    /// [`OtttError::BadShape`] or [`OtttError::OutOfRange`].
    pub fn new(n_in: usize, alpha: f64, beta: f64) -> Result<Self, OtttError> {
        if n_in == 0 || n_in > MAX_WEIGHTS {
            return Err(OtttError::BadShape);
        }
        for (what, v) in [("alpha", alpha), ("beta", beta)] {
            if !(v >= 0.0) || !(v < 1.0) {
                return Err(OtttError::OutOfRange { what, value: v });
            }
        }
        Ok(Self { q: vec![0.0; n_in], a: vec![0.0; n_in], alpha, beta })
    }

    /// How many numbers this state holds, whatever the run's length.
    #[must_use]
    pub fn footprint(&self) -> usize {
        self.q.len() + self.a.len()
    }

    /// Advance by one step of input; returns the trace.
    ///
    /// # Errors
    ///
    /// [`OtttError::Shape`] or [`OtttError::NonFinite`].
    pub fn step(&mut self, x: &[f64]) -> Result<&[f64], OtttError> {
        checked("x", x, self.q.len())?;
        for j in 0..self.q.len() {
            self.q[j] = self.alpha * self.q[j] + x[j];
            self.a[j] = self.beta * self.a[j] + self.q[j];
        }
        Ok(&self.a)
    }

    /// The trace at the `t`-th step of a single unit input, the input's own step counted as
    /// `t = 1`, in closed form: `(α^t − β^t)/(α − β)`, or `t α^{t−1}` when the two decays are equal.
    /// `None` for `t = 0`, before the input. At `α = 0` it is `β^{t−1}`, the single exponential of
    /// Xiao et al. with `λ = β`.
    ///
    /// ⚠ CORRECTED. This doc used to give `α(α^t − β^t)/(α − β)` and `t α^t`, a factor `α` too
    /// large against the recursion — [`Online::step`] gives `â = 1` at the input's own step — and
    /// described `t` as the steps elapsed after the input, one fewer than the code's `t`. The code
    /// carried the same factor and divided it back out,
    /// `alpha * (alpha.powi(k) - beta.powi(k)) / (alpha - beta) / alpha`, which is `0/0` at
    /// `α = 0`: it returned `NaN` at exactly the setting that recovers the paper's trace, and
    /// returns `β^{t−1}` there now. Elsewhere the two forms differ by rounding alone, at most 2
    /// ulp over the grid `the_impulse_closed_form_carries_no_factor_of_alpha` sweeps.
    #[must_use]
    pub fn impulse(alpha: f64, beta: f64, t: usize) -> Option<f64> {
        if t == 0 {
            return None;
        }
        let k = t as i32;
        if alpha == beta {
            Some(f64::from(k) * alpha.powi(k - 1))
        } else {
            Some((alpha.powi(k) - beta.powi(k)) / (alpha - beta))
        }
    }
}

/// The cosine and relative distance between two gradients. `None` if either is zero or the
/// lengths differ.
#[must_use]
pub fn compare(online: &[f64], truth: &[f64]) -> Option<Comparison> {
    if online.len() != truth.len() {
        return None;
    }
    let dot: f64 = online.iter().zip(truth).map(|(a, b)| a * b).sum();
    let na = online.iter().map(|a| a * a).sum::<f64>().sqrt();
    let nb = truth.iter().map(|b| b * b).sum::<f64>().sqrt();
    if !(na > 0.0) || !(nb > 0.0) {
        return None;
    }
    let gap = online.iter().zip(truth).map(|(a, b)| (a - b) * (a - b)).sum::<f64>().sqrt();
    Some(Comparison { cosine: dot / (na * nb), relative: gap / nb })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::surrogate::{FastSigmoid, LifLayerSpec, SpikeFn};

    fn layer(reset: bool) -> Layer {
        let mut l = Layer::random(5, 4, 2, 0.6, 0.8, 0.7, 1.0, 3.0, 11).unwrap();
        l.reset = reset;
        l
    }

    fn input(steps: usize, n_in: usize, seed: u64) -> Vec<f64> {
        let mut rng = Rng::new(seed);
        (0..steps * n_in).map(|_| f64::from(u8::from(rng.next_f64() < 0.35))).collect()
    }

    #[test]
    fn the_forward_pass_is_the_one_the_surrogate_module_already_runs() {
        // Both modules claim to run the same recursion. If they do not, every comparison below is
        // between two different networks, so this is checked first and to the last bit.
        let sur = FastSigmoid::default();
        let spec = LifLayerSpec {
            n_in: 5,
            n_rec: 4,
            n_out: 2,
            dt: 1e-3,
            tau_mem: -1e-3 / 0.8f64.ln(),
            tau_syn: -1e-3 / 0.6f64.ln(),
            tau_out: -1e-3 / 0.7f64.ln(),
            theta: 1.0,
            recurrent: false,
            w_scale: 3.0,
            ..LifLayerSpec::default()
        };
        let mut reference = spec.build().unwrap();
        assert!((reference.alpha - 0.6).abs() < 1e-15 && (reference.beta - 0.8).abs() < 1e-15 && (reference.kappa - 0.7).abs() < 1e-15);
        let mut mine = layer(true);
        // Same weights, taken from the reference so that nothing depends on two generators.
        for j in 0..4 {
            for i in 0..5 {
                mine.w[j * 5 + i] = reference.p[reference.idx_w(j, i)];
            }
            for c in 0..2 {
                mine.r[c * 4 + j] = reference.p[reference.idx_r(c, j)];
            }
        }
        // This layer has no bias, so the reference's must be cleared for the two to be comparable.
        let bias: Vec<usize> = (0..4).map(|j| reference.idx_b(j)).collect();
        for k in bias {
            reference.p[k] = 0.0;
        }
        let x = input(40, 5, 3);
        let theirs = reference.forward(&sur, &x, SpikeFn::Heaviside).unwrap();
        let mine_tr = mine.forward(&sur, &x).unwrap();
        let mut spikes = 0.0;
        for t in 0..40 {
            for j in 0..4 {
                assert_eq!(mine_tr.u[t * 4 + j], theirs.u[t * 4 + j], "u at {t},{j}");
                assert_eq!(mine_tr.s[t * 4 + j], theirs.s[t * 4 + j], "s at {t},{j}");
                spikes += theirs.s[t * 4 + j];
            }
            for c in 0..2 {
                assert_eq!(mine_tr.y[t * 2 + c], theirs.y[t * 2 + c], "y at {t},{c}");
            }
        }
        assert!(spikes > 20.0, "only {spikes} spikes: the comparison saw almost no activity");
    }

    #[test]
    fn the_trace_is_the_double_exponential_the_current_and_membrane_make() {
        for (alpha, beta) in [(0.6, 0.8), (0.9, 0.5), (0.7, 0.7)] {
            let mut on = Online::new(2, alpha, beta).unwrap();
            assert_eq!(on.footprint(), 4);
            assert_eq!(Online::impulse(alpha, beta, 0), None);
            let mut seen = Vec::new();
            for t in 0..12 {
                let a = on.step(&[f64::from(u8::from(t == 0)), 0.0]).unwrap();
                seen.push((a[0], a[1]));
            }
            for (t, &(a, idle)) in seen.iter().enumerate() {
                let want = Online::impulse(alpha, beta, t + 1).unwrap();
                assert!((a - want).abs() < 1e-14, "α={alpha} β={beta} at {}: {a} against {want}", t + 1);
                assert_eq!(idle, 0.0);
            }
            assert!(seen[0].0 == 1.0 && seen[2].0 > 1.0, "the trace should rise before it falls: {:?}", &seen[..4]);
        }
        // And the trace the forward pass records is that same filter of the same input.
        let l = layer(true);
        let x = input(20, 5, 9);
        let tr = l.forward(&FastSigmoid::default(), &x).unwrap();
        let mut on = Online::new(5, l.alpha, l.beta).unwrap();
        for t in 0..20 {
            let a = on.step(&x[t * 5..(t + 1) * 5]).unwrap();
            for i in 0..5 {
                assert_eq!(tr.trace[t * 5 + i], a[i], "trace at {t},{i}");
            }
        }
        assert!(tr.trace.iter().any(|a| *a > 1.5), "the trace never accumulated anything");
        assert!(Online::new(0, 0.5, 0.5).is_err() && Online::new(2, 1.0, 0.5).is_err() && Online::new(2, 0.5, -0.1).is_err());
        assert!(on.step(&[1.0]).is_err() && on.step(&[f64::NAN, 0.0, 0.0, 0.0, 0.0]).is_err());
    }

    #[test]
    fn with_no_synaptic_filter_the_trace_is_the_papers_single_exponential() {
        // Xiao et al., §4.1 after Eq. (4): `â[t+1] = λ â[t] + s[t+1]`, the recursive form of
        // `â[t] = Σ_{τ≤t} λ^{t−τ} s[τ]`. At α = 0 this module's double filter must be exactly that
        // with λ = β. λ = 3/4 and 1/2 keep every partial sum over 24 steps a binary fraction of
        // at most 48 significant bits, so the comparisons below are equalities, not tolerances.
        let steps = 24;
        let spikes = input(steps, 3, 17);
        let count = spikes.iter().filter(|s| **s == 1.0).count();
        assert!(count > 12, "only {count} input spikes: the trace would barely be exercised");
        for lambda in [0.75, 0.5] {
            let mut on = Online::new(3, 0.0, lambda).unwrap();
            let mut paper = [0.0f64; 3];
            for t in 0..steps {
                let s = &spikes[t * 3..(t + 1) * 3];
                let a = on.step(s).unwrap().to_vec();
                assert_eq!(on.q.as_slice(), s, "at α = 0 the first stage is the input itself");
                for i in 0..3 {
                    paper[i] = lambda * paper[i] + s[i];
                    let sum: f64 = (0..=t).map(|tau| lambda.powi(i32::try_from(t - tau).unwrap()) * spikes[tau * 3 + i]).sum();
                    assert_eq!(a[i], paper[i], "λ={lambda} at {t},{i}: against the recursion");
                    assert_eq!(a[i], sum, "λ={lambda} at {t},{i}: against the sum");
                }
            }
            // One spike's closed form: `β^{t−1}`, which was `NaN` here before the correction.
            for t in 1..=24usize {
                let want = lambda.powi(i32::try_from(t).unwrap() - 1);
                assert_eq!(Online::impulse(0.0, lambda, t), Some(want), "λ={lambda} at {t}");
            }
        }
        // The trace the layer's forward pass records reduces the same way.
        let l = Layer::random(3, 2, 1, 0.0, 0.75, 0.5, 1.0, 1.0, 5).unwrap();
        let tr = l.forward(&FastSigmoid::default(), &spikes).unwrap();
        let mut paper = [0.0f64; 3];
        for t in 0..steps {
            for i in 0..3 {
                paper[i] = 0.75 * paper[i] + spikes[t * 3 + i];
                assert_eq!(tr.trace[t * 3 + i], paper[i], "layer trace at {t},{i}");
            }
        }
        assert!(paper.iter().any(|a| *a > 1.0), "the trace never accumulated past one spike: {paper:?}");
    }

    #[test]
    fn the_impulse_closed_form_carries_no_factor_of_alpha() {
        // The form `impulse` had before the correction: the doc's extra `α`, divided back out.
        let before = |alpha: f64, beta: f64, k: i32| alpha * (alpha.powi(k) - beta.powi(k)) / (alpha - beta) / alpha;
        let (mut worst_ulp, mut differ, mut nan_before) = (0u64, 0usize, 0usize);
        for ai in 0..20u8 {
            for bi in 0..20u8 {
                if ai == bi {
                    continue;
                }
                let (alpha, beta) = (f64::from(ai) / 20.0, f64::from(bi) / 20.0);
                let (mut q, mut a) = (0.0f64, 0.0f64);
                for k in 1..=40i32 {
                    q = alpha * q + f64::from(u8::from(k == 1));
                    a = beta * a + q;
                    let now = Online::impulse(alpha, beta, usize::try_from(k).unwrap()).unwrap();
                    // MEASURED: the worst gap to the recursion on this grid is 2.1e-15.
                    assert!((now - a).abs() < 1e-14 * a.abs().max(1.0), "α={alpha} β={beta} k={k}: {now} against {a}");
                    let old = before(alpha, beta, k);
                    if ai == 0 {
                        assert!(old.is_nan(), "α=0 β={beta} k={k}: the old form gave {old}");
                        nan_before += 1;
                    } else {
                        let gap = now.to_bits().abs_diff(old.to_bits());
                        differ += usize::from(gap > 0);
                        worst_ulp = worst_ulp.max(gap);
                    }
                }
            }
        }
        // 19 values of β at α = 0, 40 steps each: every one of them was NaN.
        assert_eq!(nan_before, 19 * 40);
        // Elsewhere the correction is rounding, and the sweep did see rounding.
        assert!(worst_ulp <= 2 && differ > 0, "worst {worst_ulp} ulp, {differ} values differ");
        // The step the input arrives on is one, whatever the decays — not α, which the old doc gave.
        assert_eq!(Online::impulse(0.5, 0.25, 1), Some(1.0));
        assert_eq!(Online::impulse(0.5, 0.5, 1), Some(1.0));
        assert_eq!(Online::impulse(0.5, 0.25, 2), Some(0.75));
        assert_eq!(Online::impulse(0.5, 0.5, 2), Some(1.0));
    }

    /// Central finite differences of the loss on every input weight.
    fn finite_differences(l: &Layer, sur: &dyn Surrogate, x: &[f64], target: &[f64]) -> Vec<f64> {
        assert!(l.smooth, "a hard step has no finite differences to take");
        let h = 1e-6;
        (0..l.w.len())
            .map(|k| {
                let mut probe = l.clone();
                probe.w[k] += h;
                let up = probe.loss(&probe.forward(sur, x).unwrap(), target).unwrap().0;
                probe.w[k] -= 2.0 * h;
                let down = probe.loss(&probe.forward(sur, x).unwrap(), target).unwrap().0;
                (up - down) / (2.0 * h)
            })
            .collect()
    }

    #[test]
    fn without_the_reset_path_the_online_gradient_is_the_gradient() {
        // The surrogate has to be the actual derivative of what ran for a finite difference to
        // mean anything, so the spike here is the smooth sigmoid itself, not a step.
        let sur = FastSigmoid::default();
        let mut l = layer(false);
        // The smooth spike, whose derivative IS `sur.backward` — the one the gradient code uses.
        l.smooth = true;
        let x = input(30, 5, 5);
        let target = [0.4, -0.3];
        let tr = l.forward(&sur, &x).unwrap();
        let (loss, dy) = l.loss(&tr, &target).unwrap();
        assert!(loss > 0.1, "the loss is {loss}, too small to differentiate meaningfully");
        let online = l.online_gradient(&sur, &tr, &dy).unwrap();
        let fd = finite_differences(&l, &sur, &x, &target);
        let mut largest = 0.0f64;
        for k in 0..l.w.len() {
            assert!((online[k] - fd[k]).abs() < 1e-6 * fd[k].abs().max(1e-3), "w[{k}]: online {} against {}", online[k], fd[k]);
            largest = largest.max(fd[k].abs());
        }
        assert!(largest > 1e-2, "every derivative was all but zero: {largest}");
        // And backpropagation through time gets the same numbers walking the other way.
        let back = l.bptt(&sur, &x, &tr, &dy).unwrap();
        for k in 0..l.w.len() {
            assert!((online[k] - back[k]).abs() < 1e-12 * back[k].abs().max(1e-9), "w[{k}]: online {} against BPTT {}", online[k], back[k]);
        }
        let c = compare(&online, &fd).unwrap();
        assert!((c.cosine - 1.0).abs() < 1e-12 && c.relative < 1e-6, "{c:?}");
        // With the reset on, BPTT still matches the finite differences — it is the online rule
        // that drops a path, not the reference.
        l.reset = true;
        let tr = l.forward(&sur, &x).unwrap();
        let (_, dy) = l.loss(&tr, &target).unwrap();
        let back = l.bptt(&sur, &x, &tr, &dy).unwrap();
        let fd = finite_differences(&l, &sur, &x, &target);
        for k in 0..l.w.len() {
            assert!((back[k] - fd[k]).abs() < 1e-6 * fd[k].abs().max(1e-3), "reset on, w[{k}]: BPTT {} against {}", back[k], fd[k]);
        }
    }

    #[test]
    fn with_the_reset_path_it_is_an_approximation_and_the_gap_is_measured() {
        let sur = FastSigmoid::default();
        let l = layer(true);
        let x = input(30, 5, 5);
        let target = [0.4, -0.3];
        let tr = l.forward(&sur, &x).unwrap();
        let (_, dy) = l.loss(&tr, &target).unwrap();
        let online = l.online_gradient(&sur, &tr, &dy).unwrap();
        let truth = l.bptt(&sur, &x, &tr, &dy).unwrap();
        let c = compare(&online, &truth).unwrap();
        // MEASURED at this seed: the reset path costs a few percent of the gradient's length and
        // almost none of its direction — which is the empirical claim the method rests on.
        assert!(c.relative > 1e-3, "the two agreed to {}, so the reset path was never exercised", c.relative);
        assert!(c.relative < 0.25 && c.cosine > 0.95, "online against BPTT: {c:?}");
        // Whatever the size of the gap, the direction descends the loss.
        let mut probe = l.clone();
        let before = probe.loss(&probe.forward(&sur, &x).unwrap(), &target).unwrap().0;
        probe.apply(&online, 1e-3).unwrap();
        let after = probe.loss(&probe.forward(&sur, &x).unwrap(), &target).unwrap().0;
        assert!(after < before, "the online step raised the loss: {before} → {after}");
        assert_eq!(compare(&online, &[0.0; 20]), None);
        assert_eq!(compare(&online, &truth[..3]), None);
    }

    #[test]
    fn the_state_carried_between_steps_does_not_grow_with_the_run() {
        let l = layer(true);
        let sur = FastSigmoid::default();
        let mut sizes = Vec::new();
        let mut losses = Vec::new();
        for steps in [10usize, 40, 160] {
            let x = input(steps, 5, 2);
            let mut on = Online::new(5, l.alpha, l.beta).unwrap();
            for t in 0..steps {
                on.step(&x[t * 5..(t + 1) * 5]).unwrap();
            }
            sizes.push(on.footprint());
            let tr = l.forward(&sur, &x).unwrap();
            losses.push(l.loss(&tr, &[0.4, -0.3]).unwrap().0);
            // The trace the forward pass stores is for the test's convenience; what the method
            // needs at step t is one row of it, which is what `Online` holds.
            assert_eq!(tr.trace.len(), steps * 5);
        }
        assert_eq!(sizes, vec![10, 10, 10], "the online state grew with the run");
        assert!(losses[2] > losses[0], "the runs were indistinguishable, so length changed nothing");
    }

    #[test]
    fn bad_shapes_and_values_are_refused() {
        assert_eq!(Layer::random(0, 3, 2, 0.5, 0.5, 0.5, 1.0, 1.0, 1), Err(OtttError::BadShape));
        assert_eq!(Layer::random(1 << 12, 1 << 11, 2, 0.5, 0.5, 0.5, 1.0, 1.0, 1), Err(OtttError::BadShape));
        assert!(matches!(Layer::random(3, 3, 2, 1.0, 0.5, 0.5, 1.0, 1.0, 1), Err(OtttError::OutOfRange { what: "alpha", .. })));
        assert!(matches!(Layer::random(3, 3, 2, 0.5, -0.1, 0.5, 1.0, 1.0, 1), Err(OtttError::OutOfRange { what: "beta", .. })));
        assert!(matches!(Layer::random(3, 3, 2, 0.5, 0.5, f64::NAN, 1.0, 1.0, 1), Err(OtttError::OutOfRange { what: "kappa", .. })));
        assert!(matches!(Layer::random(3, 3, 2, 0.5, 0.5, 0.5, 0.0, 1.0, 1), Err(OtttError::OutOfRange { what: "theta", .. })));
        assert!(matches!(Layer::random(3, 3, 2, 0.5, 0.5, 0.5, 1.0, 0.0, 1), Err(OtttError::OutOfRange { what: "scale", .. })));
        let sur = FastSigmoid::default();
        let mut l = layer(true);
        assert_eq!(l.forward(&sur, &[]), Err(OtttError::Shape { what: "x", got: 0, want: 5 }));
        assert_eq!(l.forward(&sur, &[1.0, 0.0]), Err(OtttError::Shape { what: "x", got: 2, want: 5 }));
        assert_eq!(l.forward(&sur, &[1.0, 0.0, 0.0, 0.0, f64::NAN]), Err(OtttError::NonFinite { what: "x" }));
        let x = input(6, 5, 1);
        let tr = l.forward(&sur, &x).unwrap();
        assert_eq!(l.loss(&tr, &[0.0]), Err(OtttError::Shape { what: "target", got: 1, want: 2 }));
        assert_eq!(l.loss(&tr, &[0.0, f64::NAN]), Err(OtttError::NonFinite { what: "target" }));
        let (_, dy) = l.loss(&tr, &[0.4, -0.3]).unwrap();
        assert!(l.online_gradient(&sur, &tr, &dy[..3]).is_err() && l.bptt(&sur, &x, &tr, &dy[..3]).is_err());
        assert!(l.bptt(&sur, &x[..5], &tr, &dy).is_err());
        let before = l.clone();
        assert_eq!(l.apply(&dy, 0.1), Err(OtttError::Shape { what: "gradient", got: 12, want: 20 }));
        assert_eq!(l.apply(&[0.0; 20], f64::NAN), Err(OtttError::NonFinite { what: "rate" }));
        assert_eq!(l, before, "a refused step must not move the layer");
        // A state that overflows is named, not returned.
        let mut wild = layer(false);
        wild.w.iter_mut().for_each(|w| *w = f64::MAX);
        assert_eq!(wild.forward(&sur, &input(4, 5, 1)), Err(OtttError::NonFinite { what: "u" }));
        assert!(OtttError::BadShape.to_string().contains("at least one"));
    }
}
