//! Local learning rules: training a spiking network **forward in time**, without storing its past.
//!
//! # The problem this module exists to solve
//!
//! A recurrent network is trained by backpropagation through time. BPTT unrolls the recurrence into
//! a feedforward graph `T` steps deep, runs the loss backwards along it, and to do that it must keep
//! every state the forward pass visited — [`crate::surrogate::Trace`] is exactly that storage, and
//! its size is `t_steps * (3 * n_rec + n_out)` numbers. On a workstation that is a rounding error.
//! On a chip whose entire premise is that it does not move memory, it is the whole budget: a
//! thousand-step episode over a thousand neurons is three million words that must be held, then
//! read back in reverse, before a single weight moves.
//!
//! So the field asks a different question. **Can the credit for a weight be accumulated as the
//! network runs, in a quantity the size of the weight itself?** Every rule in this module is an
//! answer, and they disagree about how much is lost in the trade.
//!
//! # E-prop: an eligibility trace times a learning signal
//!
//! Bellec, Scherr, Subramoney, Hajek, Salaj, Legenstein & Maass, *A solution to the learning dilemma
//! for recurrent networks of spiking neurons*, Nature Communications 11:3625 (2020).
//!
//! The gradient of a loss with respect to a synapse is rewritten as a sum over time of two factors:
//!
//! ```text
//! dE/dW_ji  =  sum_t  L_j[t] * e_ji[t]
//!              \____/   \____/
//!              learning  eligibility trace: a per-synapse memory of how much
//!              signal    THIS synapse has recently been able to move THIS neuron
//! ```
//!
//! The eligibility trace runs **forward**. It is the derivative of neuron `j`'s own state with
//! respect to `W_ji`, propagated by the *diagonal* of the state-transition Jacobian — the part of
//! neuron `j`'s future that depends on neuron `j`'s present, with the paths through every other
//! neuron dropped. That truncation is the approximation, and it is the only one on the forward side.
//!
//! The learning signal is the loss's derivative with respect to the spike, restricted to the direct
//! path from that spike to the readout. It is what a broadcast error channel can carry; it is also
//! where the second approximation lives, because the true derivative includes the spike's effect on
//! every later spike.
//!
//! ## What the approximation is worth, as numbers
//!
//! "Approximates BPTT" without a number is not a claim. [`compare_to_bptt`] computes both gradients
//! of the same loss and measures the angle between them. On this crate's own
//! [`crate::surrogate::LifLayer`] at initialisation — 40 steps of the delayed-XOR input, seed 3,
//! `theta = 1`, [`crate::surrogate::ArcTan`] at its default, whose pseudo-derivative peaks at
//! exactly `1`:
//!
//! | layer | [`Jacobian`] | cosine to BPTT | relative L2 error | sign agreement |
//! |---|---|---|---|---|
//! | feedforward, 12 units | [`Jacobian::Full`] | **1.000000000000000** | **3.5e-16** | 1.000 |
//! | feedforward, 12 units | [`Jacobian::Leak`] | 0.7956 | 3.382 | 1.000 |
//! | recurrent, 16 units | [`Jacobian::Full`] | 0.9722 | 0.234 | 0.944 |
//! | recurrent, 16 units | [`Jacobian::Leak`] | 0.6824 | 1.965 | 0.944 |
//!
//! Read the first row first, because it is the check that the machinery is right rather than merely
//! plausible. **With no recurrent connections and the full per-neuron Jacobian, e-prop is not an
//! approximation of BPTT — it is BPTT**, to floating-point noise. There are then no paths through
//! other neurons to drop, and the only remaining truncation, the learning signal, turns out to cost
//! nothing: a non-recurrent unit reaches the loss through the readout and through its own reset,
//! and the reset is carried on the forward side.
//! `the_eprop_gradient_is_exactly_bptt_for_a_feedforward_layer` asserts it at `1e-9` and measures
//! `3.5e-16`.
//!
//! The same thing happens for a **single** recurrent unit, whose only recurrent path is its own
//! autapse: `a_single_self_connected_unit_is_also_exact` measures 1.9e-16. Together the two cases
//! say precisely what the approximation is — it is the paths between *different* neurons, and
//! nothing else.
//!
//! Read the second row next, because it is the one this implementation did not expect. Dropping the
//! reset term — the form in which the e-prop literature writes the eligibility trace for a leaky
//! integrate-and-fire unit — makes the gradient of the *same feedforward layer*, where every other
//! term is exact, **4.1 times too long and 37 degrees off**, for a relative L2 error of 3.38. It is
//! not a small correction, and the length is the larger half of it: a rule that pointed the right
//! way at the wrong scale would still descend, and this one overshoots.
//!
//! ## ...and the number that explains all four rows
//!
//! Both truncations are multiplied by the pseudo-derivative. The reset feeds back as `theta * psi`
//! against a membrane leak of `beta = 0.951`, and the recurrent paths feed back as `V * psi`. So the
//! quality of the approximation is not a property of e-prop at all — **it is a property of how tall
//! the surrogate is**. Rescaling [`crate::surrogate::ArcTan`] with
//! [`crate::surrogate::Scaled`] and changing nothing else, relative L2 error against BPTT:
//!
//! | surrogate peak | feedforward, `Leak` | recurrent, `Full` | recurrent, `Leak` |
//! |---|---|---|---|
//! | 1.00 | 3.382 | 0.234 | 1.965 |
//! | 0.30 | 0.834 | 0.094 | 0.457 |
//! | 0.10 | 0.173 | 0.023 | 0.087 |
//! | 0.03 | 0.021 | 0.0030 | 0.010 |
//!
//! Every column falls roughly linearly in the peak. The e-prop literature damps its
//! pseudo-derivative by a factor usually written `gamma` and commonly quoted at `0.3` — a value
//! this review did not verify against the paper — and the second row is what such a damping buys.
//! That is offered as a plausible reading of why the published rule works as well as it does, and
//! as a measurement, not as the paper's own argument. The honest summary is that **a reported
//! e-prop-to-BPTT agreement is meaningless without the surrogate's gain beside it**, and this review
//! did not locate that gain reported alongside such a comparison anywhere in the literature it read.
//! `the_eprop_error_is_set_by_the_surrogates_peak` is the sweep.
//!
//! One block of the gradient is exact in every row, and it is worth knowing which: the **readout
//! weights**. `R[c][j]`'s gradient is a kappa-filtered spike trace times the error at the readout,
//! with no truncation anywhere, so
//! `the_readout_block_of_the_eprop_gradient_is_exactly_the_bptt_one` checks it as an equality.
//!
//! And an approximate gradient is only interesting if it trains: `eprop_learns_delayed_xor` takes
//! the task the BPTT test in [`crate::surrogate`] solves, swaps the gradient call and solves it
//! anyway, at the top row of the first table — cosine 0.68, relative error 1.97.
//!
//! ## What it actually saves, and when it does not
//!
//! [`eprop_state_words`] against [`bptt_state_words`], for the same layer. E-prop carries two
//! eligibility numbers per *parameter*; BPTT carries three per *neuron per step*. For the default
//! 16-unit recurrent layer that is 660 words for e-prop against `50 * t_steps` for BPTT, so
//! **e-prop is the larger of the two below 14 steps** and wins by 3x at 40 and by 7.6x at 100. The
//! crossover moves with the fan-in: e-prop's cost is set by how many synapses a neuron has, BPTT's
//! by how long the episode is. The literature tends to report only the side of that crossover where
//! e-prop wins.
//!
//! # The other rules here
//!
//! - [`Tempotron`] (Gütig & Sompolinsky, Nature Neuroscience 9:420–428, 2006) learns a **binary**
//!   decision about a spatiotemporal pattern: fire at least once, or stay silent. Its credit
//!   assignment is a single moment — the time the membrane peaked — so it needs no trace at all.
//! - [`SpikeProp`] (Bohte, Kok & La Poutré, Neurocomputing 48:17–37, 2002) does gradient descent on
//!   the **spike time** itself, using the implicit-function derivative of a threshold crossing. It
//!   is the oldest of these rules and the most fragile: the gradient is undefined the moment the
//!   neuron stops firing, and [`SpikeProp::sensitivity`] returns `None` there rather than a number.
//! - [`ReSuMe`] (Ponulak & Kasiński, Neural Computation 22:467–510, 2010) drives the output spike
//!   train toward a **desired** spike train by pairing each side against the presynaptic history
//!   with an STDP-shaped window. With the teacher alone it *is* the STDP window of
//!   [`crate::plasticity::PairStdp`]; with the output alone it is its exact negative. Both are
//!   checked as equalities, not tolerances.
//! - [`Force`] (Sussillo & Abbott, Neuron 63:544–557, 2009; applied to spiking networks by Nicola &
//!   Clopath, Nature Communications 8:2208, 2017) trains a linear readout online by recursive least
//!   squares. Its state is an `n × n` inverse correlation matrix, which is the opposite trade to
//!   e-prop's: nothing is approximated, and the memory is quadratic in the population.
//!
//! # Units
//!
//! Every time in this module's tempotron, `ReSuMe` and `SpikeProp` interfaces is **seconds**, every
//! rate **hertz**. The e-prop functions operate on [`crate::surrogate::LifLayer`], which is
//! dimensionless by construction — its `alpha`, `beta` and `kappa` are per-step decay factors that
//! [`crate::surrogate::LifLayerSpec::build`] already converted from seconds — so the conversion
//! happens there and not again here. Weights are in whatever unit the caller's postsynaptic
//! potential is measured in; the rules are linear in it.
//!
//! # Determinism
//!
//! Every function here is a pure function of its inputs. The two tests that need a dataset build it
//! from [`crate::rng::Rng`] with a stated seed.

use crate::surrogate::{LifLayer, SpikeFn, Surrogate, SurrogateError, cross_entropy};

/// Why a local learning rule could not produce an answer.
///
/// Every variant names the quantity that was wrong and its value. These are returned instead of a
/// plausible number because a learning rule that silently trains on a malformed gradient reports a
/// falling loss until every weight is non-finite at once, and by then the run is a week old.
#[derive(Debug, Clone, PartialEq)]
pub enum LearnError {
    /// A scalar parameter was not a finite number.
    NonFiniteParam {
        /// Which parameter, by its field name.
        what: &'static str,
        /// The offending value, so an infinity can be told from a `NaN`.
        value: f64,
    },
    /// A scalar parameter had to be strictly positive and was not.
    NotPositive {
        /// Which parameter, by its field name.
        what: &'static str,
        /// The offending value.
        value: f64,
    },
    /// An element of an input array was not finite.
    NonFiniteInput {
        /// Flat index into the offending array.
        index: usize,
        /// The offending value.
        value: f64,
    },
    /// An array's length did not match what it was handed to.
    ShapeMismatch {
        /// Which array, by its role.
        what: &'static str,
        /// The length that arrived.
        got: usize,
        /// The length, or the divisor, that was required.
        want: usize,
    },
    /// A collection that must carry at least one element was empty.
    Empty {
        /// What was empty, by its role.
        what: &'static str,
    },
    /// A comparison against a reference gradient was asked for where that gradient is zero.
    ///
    /// The cosine between a vector and the zero vector is `0/0`. It is not "perfect agreement" and
    /// it is not "no agreement", so [`compare_to_bptt`] refuses rather than reporting either.
    ZeroGradient,
    /// The recursive-least-squares denominator `1 + rᵀPr` left the positive reals.
    ///
    /// For a positive-definite `P` this quantity is at least one, so reaching here means `P` has
    /// lost its definiteness to rounding — which happens when the regulariser is far too small for
    /// the conditioning of the inputs. Reported with the update index so the run can be restarted
    /// from before it.
    Singular {
        /// How many updates had already been applied when the denominator failed.
        step: u64,
        /// The offending denominator.
        value: f64,
    },
    /// The tempotron's two time constants were equal, where the kernel divides by their difference.
    ///
    /// The limit exists — it is the alpha function of [`AlphaPsp`] — but taking it silently would
    /// hand back a differently-shaped kernel than the one that was asked for.
    TimeConstantsEqual {
        /// The shared value, seconds.
        tau: f64,
    },
    /// Something inside [`crate::surrogate`] refused first.
    Surrogate(SurrogateError),
}

impl core::fmt::Display for LearnError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NonFiniteParam { what, value } => write!(f, "{what} is not finite ({value})"),
            Self::NotPositive { what, value } => write!(f, "{what} must be > 0, got {value}"),
            Self::NonFiniteInput { index, value } => {
                write!(f, "input element {index} is not finite ({value})")
            }
            Self::ShapeMismatch { what, got, want } => {
                write!(f, "{what} has length {got}, which does not fit {want}")
            }
            Self::Empty { what } => write!(f, "{what} is empty"),
            Self::ZeroGradient => {
                write!(f, "the reference gradient is zero, so an angle against it is undefined")
            }
            Self::Singular { step, value } => {
                write!(f, "the least-squares denominator was {value} at update {step}")
            }
            Self::TimeConstantsEqual { tau } => {
                write!(f, "tau and tau_s are both {tau}, where the kernel divides by their difference")
            }
            Self::Surrogate(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for LearnError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Surrogate(e) => Some(e),
            _ => None,
        }
    }
}

impl From<SurrogateError> for LearnError {
    fn from(e: SurrogateError) -> Self {
        Self::Surrogate(e)
    }
}

/// `Ok` for a finite, strictly positive parameter.
fn positive(what: &'static str, value: f64) -> Result<f64, LearnError> {
    if !value.is_finite() {
        return Err(LearnError::NonFiniteParam { what, value });
    }
    if !(value > 0.0) {
        return Err(LearnError::NotPositive { what, value });
    }
    Ok(value)
}

/// `Ok` for a finite parameter of any sign.
fn finite(what: &'static str, value: f64) -> Result<f64, LearnError> {
    if value.is_finite() { Ok(value) } else { Err(LearnError::NonFiniteParam { what, value }) }
}

/// `Ok` when every element of `v` is finite, naming the first that is not.
fn finite_slice(v: &[f64]) -> Result<(), LearnError> {
    for (index, &value) in v.iter().enumerate() {
        if !value.is_finite() {
            return Err(LearnError::NonFiniteInput { index, value });
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Eligibility traces
// ---------------------------------------------------------------------------------------------

/// One exponentially decaying eligibility trace: the state every rule in this module is built from.
///
/// A trace is a leaky running sum of a driving quantity. Its value `k` steps after a single unit
/// input, with nothing driving it since, is exactly `exp(-k * dt / tau)` — which is what makes an
/// online rule that multiplies an amplitude by a trace *equal* to the integral form of the same
/// rule, rather than an approximation of it. [`Eligibility::closed_form`] is that expression, and
/// `an_eligibility_trace_decays_exactly_as_its_time_constant_says` compares the online run against
/// it rather than against a previous online run.
///
/// The decay factor is `exp(-dt / tau)`, computed once. The tempting alternative `1 - dt / tau` is
/// the first two terms of that expansion and is wrong by 0.12% per step at `dt = tau / 20`, which
/// compounds to 7.4% over a single time constant — small enough to look like a tuning difference
/// and large enough to move every learned weight.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Eligibility {
    /// Decay time constant, **seconds**. Strictly positive and finite.
    pub tau: f64,
    /// Time step the trace is advanced by, **seconds**. Strictly positive and finite.
    pub dt: f64,
    /// Current value, in the unit of whatever drives it. Starts at zero.
    pub value: f64,
    /// `exp(-dt / tau)`, in `(0, 1)`. Stored so the exponential is evaluated once per trace rather
    /// than once per step.
    decay: f64,
}

impl Eligibility {
    /// A trace at rest.
    ///
    /// # Errors
    ///
    /// [`LearnError::NotPositive`] or [`LearnError::NonFiniteParam`] for a `tau` or `dt` that is not
    /// a finite strictly positive number. A zero `tau` is not "instant decay", it is a division by
    /// zero followed by a `NaN` the first time the trace is read.
    pub fn new(tau: f64, dt: f64) -> Result<Self, LearnError> {
        let tau = positive("tau", tau)?;
        let dt = positive("dt", dt)?;
        Ok(Self { tau, dt, value: 0.0, decay: (-dt / tau).exp() })
    }

    /// The per-step decay factor `exp(-dt / tau)`.
    #[must_use]
    pub fn decay(&self) -> f64 {
        self.decay
    }

    /// Decay by one step, then add `input`. Returns the new value.
    ///
    /// Decay-then-add, not add-then-decay: the two differ by one factor of `decay` on every input,
    /// which is a systematic scale error on the learned weights and is invisible in any plot of the
    /// trace's shape.
    pub fn step(&mut self, input: f64) -> f64 {
        self.value = self.decay * self.value + input;
        self.value
    }

    /// The value this trace would hold `steps` further on with nothing driving it.
    #[must_use]
    pub fn after(&self, steps: u32) -> f64 {
        self.value * self.decay.powi(steps.min(i32::MAX as u32) as i32)
    }

    /// The value this trace would hold `seconds` further on with nothing driving it, from the
    /// continuous-time closed form `value * exp(-seconds / tau)`.
    ///
    /// This is the expression [`Eligibility::step`] is checked against. It is a function of
    /// *seconds*, not of steps, so it can be evaluated between grid points.
    #[must_use]
    pub fn closed_form(&self, seconds: f64) -> f64 {
        self.value * (-seconds / self.tau).exp()
    }

    /// Back to zero.
    pub fn clear(&mut self) {
        self.value = 0.0;
    }
}

// ---------------------------------------------------------------------------------------------
// E-prop
// ---------------------------------------------------------------------------------------------

/// Which terms of a neuron's own state-transition Jacobian the eligibility recursion carries.
///
/// The choice is not between "correct" and "approximate": both variants drop every path through
/// other neurons, and that is the approximation. The choice is only about the terms that are
/// already local.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Jacobian {
    /// The two leaks only: `alpha` on the synaptic current, `beta` on the membrane.
    ///
    /// This is the shape of the eligibility trace as the e-prop literature writes it for a leaky
    /// integrate-and-fire unit — a decaying presynaptic trace multiplied by the pseudo-derivative,
    /// with no term for the reset. This implementation did not establish from the paper whether that
    /// omission is derived or adopted; what it can state is its size, and it is **not small**:
    /// `dropping_the_reset_term_changes_the_feedforward_gradient` measures a relative L2 error of
    /// 3.38 against BPTT on a feedforward layer where every other term is exact, with the gradient
    /// coming out 4.1 times too long. That figure is proportional to the surrogate's peak; see the
    /// module doc's second table before concluding anything from it.
    Leak,
    /// The leaks, plus the soft reset `-theta * psi` on the membrane and the self-connection
    /// `V[j][j] * psi` into the current.
    ///
    /// Every term here belongs to neuron `j` alone, so carrying them costs nothing extra: the
    /// eligibility recursion already visits them, and the two extra multiplies per slot are the
    /// whole cost. With this variant and `recurrent = false`, the e-prop gradient is the BPTT
    /// gradient exactly; with recurrence it cuts the relative error from 1.97 to 0.23 on the
    /// module doc's reference layer. This implementation did not locate this variant offered as an
    /// option in any published e-prop implementation it read.
    Full,
}

/// How the e-prop gradient is computed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EpropConfig {
    /// Which local Jacobian terms the eligibility recursion carries.
    pub jacobian: Jacobian,
    /// Which forward spike function the network ran, passed through to
    /// [`crate::surrogate::LifLayer::forward`] so that the gradient and the run agree about it.
    pub spike_fn: SpikeFn,
}

impl Default for EpropConfig {
    /// [`Jacobian::Leak`] and [`SpikeFn::Heaviside`]: the literature's eligibility trace, on a
    /// network that emits real binary spikes.
    fn default() -> Self {
        Self { jacobian: Jacobian::Leak, spike_fn: SpikeFn::Heaviside }
    }
}

/// `sum_{m=0}^{n-1} kappa^m`, evaluated so that `kappa` near one does not cancel.
///
/// The direct form `(1 - kappa^n) / (1 - kappa)` loses every significant digit as `kappa -> 1`,
/// which is the regime of a long readout time constant. Written through `exp_m1` both numerator and
/// denominator are computed as small differences directly, and the `kappa == 1` limit is `n`.
fn geometric_sum(ln_kappa: f64, denom: f64, n: usize) -> f64 {
    if denom == 0.0 {
        return n as f64;
    }
    (n as f64 * ln_kappa).exp_m1() / denom
}

/// Run the layer forward, calling `on_step` after each step, and return the time-averaged logits.
///
/// The one copy of [`crate::surrogate::LifLayer`]'s recurrence that this module owns. Both
/// [`logits_streaming`] and [`eprop_grad_from_dlogits`] go through it, so a transcription error
/// here cannot make one of them disagree with the other while both look right.
///
/// `on_step` receives `(t, u, s_now, s_prev)`: the membrane potentials after the step, the spikes
/// emitted at this step, and the spikes emitted at the previous one.
fn sweep(
    layer: &LifLayer,
    sur: &dyn Surrogate,
    x: &[f64],
    spike_fn: SpikeFn,
    mut on_step: impl FnMut(usize, &[f64], &[f64], &[f64]),
) -> Result<Vec<f64>, LearnError> {
    let (ni, nr, no) = (layer.n_in, layer.n_rec, layer.n_out);
    if x.is_empty() || ni == 0 || !x.len().is_multiple_of(ni) {
        return Err(LearnError::ShapeMismatch { what: "input", got: x.len(), want: ni });
    }
    finite_slice(x)?;
    let t_steps = x.len() / ni;

    let mut i_syn = vec![0.0; nr];
    let mut u = vec![0.0; nr];
    let mut s_prev = vec![0.0; nr];
    let mut s_now = vec![0.0; nr];
    let mut y = vec![0.0; no];
    let mut acc = vec![0.0; no];

    for t in 0..t_steps {
        for j in 0..nr {
            let mut drive = layer.p[layer.idx_b(j)];
            for i in 0..ni {
                drive += layer.p[layer.idx_w(j, i)] * x[t * ni + i];
            }
            if layer.recurrent && t > 0 {
                for k in 0..nr {
                    drive += layer.p[layer.idx_v(j, k)] * s_prev[k];
                }
            }
            let i_now = layer.alpha * i_syn[j] + drive;
            let u_now = layer.beta * u[j] + i_now - layer.theta * s_prev[j];
            if !i_now.is_finite() || !u_now.is_finite() {
                let value = if i_now.is_finite() { u_now } else { i_now };
                return Err(SurrogateError::Diverged { step: t, neuron: j, value }.into());
            }
            i_syn[j] = i_now;
            u[j] = u_now;
            s_now[j] = match spike_fn {
                SpikeFn::Heaviside => sur.forward(u_now - layer.theta),
                SpikeFn::Smooth => sur.antiderivative(u_now - layer.theta),
            };
        }
        for c in 0..no {
            let mut v = layer.kappa * y[c];
            for j in 0..nr {
                v += layer.p[layer.idx_r(c, j)] * s_now[j];
            }
            y[c] = v;
            acc[c] += v;
        }
        on_step(t, &u, &s_now, &s_prev);
        s_prev.copy_from_slice(&s_now);
    }
    // Divided, not multiplied by a reciprocal: `LifLayer::forward` divides, and the two differ in
    // the last place. `streaming_logits_are_bit_identical_to_the_stored_trace` is what holds this
    // line to the same operation.
    for v in &mut acc {
        *v /= t_steps as f64;
    }
    Ok(acc)
}

/// The time-averaged readout, computed with `O(n_rec + n_out)` state instead of `O(t_steps)`.
///
/// Bit-for-bit the `logits` field of [`crate::surrogate::LifLayer::forward`] — same arithmetic in
/// the same order — but it never allocates the trace. This is what inference costs on a device that
/// is not going to train: the whole of [`crate::surrogate::Trace`] exists for the backward pass and
/// for nothing else.
///
/// # Errors
///
/// [`LearnError::ShapeMismatch`] if `x` is empty or its length is not a multiple of `n_in`;
/// [`LearnError::NonFiniteInput`] naming the first non-finite element; [`LearnError::Surrogate`]
/// wrapping [`SurrogateError::Diverged`] at the first step and unit whose state left the finite
/// numbers.
pub fn logits_streaming(
    layer: &LifLayer,
    sur: &dyn Surrogate,
    x: &[f64],
    spike_fn: SpikeFn,
) -> Result<Vec<f64>, LearnError> {
    sweep(layer, sur, x, spike_fn, |_, _, _, _| {})
}

/// The e-prop gradient of a loss whose derivative at the readout is `d_logits`.
///
/// The whole rule, and it is short enough to read. For every recurrent unit `j` and every parameter
/// slot `m` that feeds its input current, two numbers are carried forward:
///
/// ```text
/// eps_I[t] = alpha eps_I[t-1] + C[t-1] eps_U[t-1] + xi[t]
/// eps_U[t] = B[t-1] eps_U[t-1] + eps_I[t]
/// e[t]     = psi[t] * eps_U[t]                       <- the eligibility trace
/// ```
///
/// where `xi[t]` is the presynaptic factor (`x[t][i]` for an input weight, `S[t-1][k]` for a
/// recurrent one, `1` for the bias), `psi[t] = sur.backward(U[t] - theta)` is the pseudo-derivative,
/// and `(C, B)` are the local Jacobian terms selected by [`Jacobian`]. The gradient is then
/// `sum_t L_j[t] * e[t]` with `L_j[t] = sum_c gy_c[t] * R[c][j]`, the direct path from the spike to
/// the readout.
///
/// `gy_c[t]` is the readout's own backward filter, and it has a closed form —
/// `d_logits[c] / T * sum_{m=0}^{T-1-t} kappa^m` — so it is evaluated forward rather than swept
/// backwards. That is the one place where knowing the episode length `T` in advance is used, and it
/// is why this function is honest about being *forward in time* rather than *online*: an episodic
/// loss is not available until the episode ends, whatever rule consumes it.
///
/// Memory is `2 * n_rec * slots` for the eligibility plus `O(n_rec + n_out)` for the state, with no
/// dependence on `t_steps`. [`eprop_state_words`] counts it.
///
/// # Errors
///
/// [`LearnError::ShapeMismatch`] for an `x` that is empty or not a multiple of `n_in`, or a
/// `d_logits` that is not `n_out` long; [`LearnError::NonFiniteInput`] naming the first non-finite
/// element of either; [`LearnError::Surrogate`] wrapping [`SurrogateError::Diverged`] if the forward
/// state leaves the finite numbers.
pub fn eprop_grad_from_dlogits(
    layer: &LifLayer,
    sur: &dyn Surrogate,
    x: &[f64],
    d_logits: &[f64],
    cfg: EpropConfig,
) -> Result<Vec<f64>, LearnError> {
    let (ni, nr, no) = (layer.n_in, layer.n_rec, layer.n_out);
    if d_logits.len() != no {
        return Err(LearnError::ShapeMismatch { what: "d_logits", got: d_logits.len(), want: no });
    }
    finite_slice(d_logits)?;
    if x.is_empty() || ni == 0 || !x.len().is_multiple_of(ni) {
        return Err(LearnError::ShapeMismatch { what: "input", got: x.len(), want: ni });
    }
    let t_steps = x.len() / ni;

    let v_slots = if layer.recurrent { nr } else { 0 };
    let slots = ni + v_slots + 1;
    let mut eps_i = vec![0.0; nr * slots];
    let mut eps_u = vec![0.0; nr * slots];
    let mut psi_prev = vec![0.0; nr];
    let mut g = vec![0.0; layer.p.len()];

    let inv_t = 1.0 / t_steps as f64;
    let ln_kappa = layer.kappa.ln();
    let denom = ln_kappa.exp_m1();

    sweep(layer, sur, x, cfg.spike_fn, |t, u, s_now, s_prev| {
        // The eligibility recursion uses psi and S from step t-1 and the drive at step t, so it is
        // advanced here, before psi[t] is read, even though the callback fires after the state
        // update. Nothing in it depends on U[t].
        for j in 0..nr {
            let (c_iu, b_uu) = match cfg.jacobian {
                Jacobian::Leak => (0.0, layer.beta),
                Jacobian::Full => {
                    let self_w =
                        if layer.recurrent { layer.p[layer.idx_v(j, j)] } else { 0.0 };
                    (self_w * psi_prev[j], layer.beta - layer.theta * psi_prev[j])
                }
            };
            let base = j * slots;
            for m in 0..slots {
                let xi = if m < ni {
                    x[t * ni + m]
                } else if m < ni + v_slots {
                    s_prev[m - ni]
                } else {
                    1.0
                };
                let k = base + m;
                let ei = layer.alpha * eps_i[k] + c_iu * eps_u[k] + xi;
                eps_u[k] = b_uu * eps_u[k] + ei;
                eps_i[k] = ei;
            }
        }

        // gy_c[t], the readout error filtered backwards through its own leak, in closed form.
        let g_t = geometric_sum(ln_kappa, denom, t_steps - t);
        for c in 0..no {
            let gy = d_logits[c] * inv_t * g_t;
            for j in 0..nr {
                g[layer.idx_r(c, j)] += gy * s_now[j];
            }
        }
        for j in 0..nr {
            let mut l = 0.0;
            for c in 0..no {
                l += d_logits[c] * inv_t * g_t * layer.p[layer.idx_r(c, j)];
            }
            let psi = sur.backward(u[j] - layer.theta);
            let lp = l * psi;
            let base = j * slots;
            for m in 0..slots {
                let idx = if m < ni {
                    layer.idx_w(j, m)
                } else if m < ni + v_slots {
                    layer.idx_v(j, m - ni)
                } else {
                    layer.idx_b(j)
                };
                g[idx] += lp * eps_u[base + m];
            }
            psi_prev[j] = psi;
        }
    })?;
    Ok(g)
}

/// Softmax cross-entropy loss and its e-prop gradient for one pattern.
///
/// The counterpart of [`crate::surrogate::LifLayer::loss_and_grad`], which computes the same loss
/// and the BPTT gradient. Swapping one call for the other is the whole difference between the two
/// training regimes, which is the point of matching the signature.
///
/// # Errors
///
/// As [`logits_streaming`], [`crate::surrogate::cross_entropy`] and [`eprop_grad_from_dlogits`].
pub fn eprop_grad(
    layer: &LifLayer,
    sur: &dyn Surrogate,
    x: &[f64],
    target: usize,
    cfg: EpropConfig,
) -> Result<(f64, Vec<f64>), LearnError> {
    let logits = logits_streaming(layer, sur, x, cfg.spike_fn)?;
    let (loss, d_logits) = cross_entropy(&logits, target)?;
    let g = eprop_grad_from_dlogits(layer, sur, x, &d_logits, cfg)?;
    Ok((loss, g))
}

/// Mean loss and mean e-prop gradient over a batch of `(input, target)` pairs.
///
/// The mean rather than the sum, so a learning rate transfers between batch sizes — the same
/// convention as [`crate::surrogate::LifLayer::batch_loss_and_grad`].
///
/// # Errors
///
/// [`LearnError::Empty`] for an empty batch: a mean over nothing is undefined, and zero is
/// specifically the wrong answer because zero is also what a perfectly trained batch returns.
/// Otherwise as [`eprop_grad`].
pub fn eprop_batch(
    layer: &LifLayer,
    sur: &dyn Surrogate,
    batch: &[(Vec<f64>, usize)],
    cfg: EpropConfig,
) -> Result<(f64, Vec<f64>), LearnError> {
    if batch.is_empty() {
        return Err(LearnError::Empty { what: "batch" });
    }
    let mut loss = 0.0;
    let mut g = vec![0.0; layer.p.len()];
    for (x, target) in batch {
        let (l, gi) = eprop_grad(layer, sur, x, *target, cfg)?;
        loss += l;
        for (a, b) in g.iter_mut().zip(gi.iter()) {
            *a += *b;
        }
    }
    let inv = 1.0 / batch.len() as f64;
    loss *= inv;
    for a in &mut g {
        *a *= inv;
    }
    Ok((loss, g))
}

/// Numbers the eligibility state costs, independent of how long the episode is.
///
/// Two eligibility components per `(unit, parameter slot)` pair, plus the forward state: synaptic
/// current, membrane, the spike vector, the previous spike vector and the previous
/// pseudo-derivative, then the readout accumulators.
#[must_use]
pub fn eprop_state_words(layer: &LifLayer) -> usize {
    let slots = layer.n_in + if layer.recurrent { layer.n_rec } else { 0 } + 1;
    2 * layer.n_rec * slots + 5 * layer.n_rec + 2 * layer.n_out
}

/// Numbers [`crate::surrogate::Trace`] costs for the same layer over `t_steps` steps.
///
/// `t_steps * (3 * n_rec + n_out)`: the synaptic current, the membrane and the spike at every unit
/// and every step, plus the readout. `None` on overflow, because a wrapped product would report a
/// small number for an enormous run — which is the direction that flatters BPTT.
#[must_use]
pub fn bptt_state_words(layer: &LifLayer, t_steps: usize) -> Option<usize> {
    3usize
        .checked_mul(layer.n_rec)
        .and_then(|v| v.checked_add(layer.n_out))
        .and_then(|v| v.checked_mul(t_steps))
}

/// How closely an e-prop gradient agrees with the BPTT gradient of the same loss.
///
/// Three numbers rather than one, because they fail in different ways: a rule can point in very
/// nearly the right direction and be badly scaled ([`GradAgreement::cosine`] near one,
/// [`GradAgreement::relative`] large), or have the right magnitude and the wrong direction, or —
/// the case that matters for whether it descends at all — disagree about the *sign* of individual
/// coordinates while looking fine in aggregate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GradAgreement {
    /// Cosine of the angle between the two gradients, in `[-1, 1]`. One means parallel.
    pub cosine: f64,
    /// `||g_eprop - g_bptt|| / ||g_bptt||`, dimensionless and non-negative. Zero means identical.
    pub relative: f64,
    /// Fraction of the coordinates where the two gradients have the same strict sign, counted only
    /// over the coordinates where the BPTT gradient is non-zero. In `[0, 1]`.
    ///
    /// Restricted that way on purpose: a layer built with `recurrent = false` holds a whole block of
    /// parameters at exactly zero in both gradients, and counting those as agreement would push this
    /// number toward one for a rule that got everything else wrong.
    pub sign_agreement: f64,
    /// Euclidean norm of the BPTT gradient, in the parameters' own unit.
    pub bptt_norm: f64,
    /// Euclidean norm of the e-prop gradient, same unit.
    pub eprop_norm: f64,
    /// Coordinates compared in [`GradAgreement::sign_agreement`]: those with a non-zero BPTT entry.
    pub n_compared: usize,
    /// Total parameters in the layer.
    pub n_params: usize,
}

/// Compute both gradients of the same loss and measure how far apart they are.
///
/// The BPTT side is [`crate::surrogate::LifLayer::backward`], which is the exact reverse-mode
/// gradient under [`SpikeFn::Smooth`] and the surrogate gradient under [`SpikeFn::Heaviside`];
/// either way both sides are handed the same `sur`, so this measures the e-prop truncation and
/// nothing else.
///
/// # Errors
///
/// [`LearnError::ZeroGradient`] when the BPTT gradient is exactly zero, where the cosine is `0/0`.
/// Otherwise as [`eprop_grad_from_dlogits`] and [`crate::surrogate::LifLayer::backward`].
pub fn compare_to_bptt(
    layer: &LifLayer,
    sur: &dyn Surrogate,
    x: &[f64],
    target: usize,
    cfg: EpropConfig,
) -> Result<GradAgreement, LearnError> {
    let tr = layer.forward(sur, x, cfg.spike_fn)?;
    let (_, d_logits) = cross_entropy(&tr.logits, target)?;
    let gb = layer.backward(sur, x, &tr, &d_logits)?;
    let ge = eprop_grad_from_dlogits(layer, sur, x, &d_logits, cfg)?;

    let (mut dot, mut nb2, mut ne2, mut diff2) = (0.0, 0.0, 0.0, 0.0);
    let (mut agree, mut compared) = (0usize, 0usize);
    for (&a, &b) in gb.iter().zip(ge.iter()) {
        dot += a * b;
        nb2 += a * a;
        ne2 += b * b;
        diff2 += (a - b) * (a - b);
        if a != 0.0 {
            compared += 1;
            if a * b > 0.0 {
                agree += 1;
            }
        }
    }
    let bptt_norm = nb2.sqrt();
    let eprop_norm = ne2.sqrt();
    if !(bptt_norm > 0.0) {
        return Err(LearnError::ZeroGradient);
    }
    Ok(GradAgreement {
        cosine: if eprop_norm > 0.0 { dot / (bptt_norm * eprop_norm) } else { 0.0 },
        relative: diff2.sqrt() / bptt_norm,
        sign_agreement: if compared > 0 { agree as f64 / compared as f64 } else { 0.0 },
        bptt_norm,
        eprop_norm,
        n_compared: compared,
        n_params: gb.len(),
    })
}

// ---------------------------------------------------------------------------------------------
// Tempotron
// ---------------------------------------------------------------------------------------------

/// One tempotron training example: the spike times of each afferent in **seconds**, and whether the
/// neuron is supposed to fire at all in response.
///
/// The outer index is the afferent and must match [`Tempotron::w`]'s length; the inner vector is
/// that afferent's spike times, which need not be sorted.
pub type Example = (Vec<Vec<f64>>, bool);

/// The tempotron: a neuron that learns to fire, or not, in response to a spatiotemporal pattern.
///
/// Gütig & Sompolinsky, *The tempotron: a neuron that learns spike timing-based decisions*, Nature
/// Neuroscience 9:420–428 (2006).
///
/// The membrane is a weighted sum of normalised postsynaptic potentials,
///
/// ```text
/// V(t) = sum_i  w_i  sum_{t_i < t}  K(t - t_i),
/// K(s) = V0 ( exp(-s / tau) - exp(-s / tau_s) )   for s > 0,  0 otherwise
/// ```
///
/// with `V0` fixed so that `K` peaks at exactly one. The decision is binary and is taken over the
/// whole trial: the neuron "fires" if `max_t V(t)` reaches the threshold.
///
/// # Why it needs no trace
///
/// The credit assignment is a single instant. If the decision was wrong, the rule moves every weight
/// by the postsynaptic potential that synapse was contributing **at `t_max`, the time the membrane
/// peaked** — the moment at which the neuron came closest to the decision boundary. There is no
/// trace to carry and no gradient through time, because the quantity being differentiated is
/// `max_t V(t)` and the maximum's derivative is the integrand's derivative at the argmax.
///
/// That also bounds what it can learn: one binary decision per trial, and nothing about *when* the
/// neuron fires. [`ReSuMe`] and [`SpikeProp`] are the rules that care about the time.
///
/// # It cannot start from silence
///
/// With every weight at exactly zero the membrane is flat, the peak lands on the first grid point,
/// every postsynaptic potential evaluated there is zero, and the update is therefore zero as well:
/// the rule has a fixed point at the origin and cannot leave it. So the weights must be initialised
/// away from zero, and [`Tempotron::new`] starting them at zero is a deliberate choice to make that
/// the caller's decision rather than a hidden default. `a_correct_tempotron_decision_moves_no_weight`
/// documents the trap in place.
///
/// # A discretisation this implementation does not hide
///
/// `t_max` is found by scanning a grid of step `dt`, so it is accurate to `dt` and no better, and
/// the weight update inherits that. Refining it would need the root of `V'(t) = 0`, which for a sum
/// of double exponentials has no closed form. The grid is the caller's parameter and appears in
/// every method that needs it.
#[derive(Debug, Clone, PartialEq)]
pub struct Tempotron {
    /// Synaptic weights, one per afferent, in the membrane's own unit. Any sign.
    pub w: Vec<f64>,
    /// Membrane time constant, **seconds**. Must exceed `tau_s`.
    pub tau: f64,
    /// Synaptic time constant, **seconds**. Gütig & Sompolinsky use `tau / 4`.
    pub tau_s: f64,
    /// Firing threshold, in the same unit as [`Tempotron::w`]. Strictly positive.
    pub theta: f64,
    /// Learning rate, in the weights' unit per unit of postsynaptic potential. Strictly positive.
    pub lambda: f64,
    /// `V0`, the constant that normalises `K`'s peak to one. Derived from `tau` and `tau_s`.
    v0: f64,
}

impl Tempotron {
    /// A tempotron with `n_in` afferents, all weights zero.
    ///
    /// # Errors
    ///
    /// [`LearnError::Empty`] for `n_in == 0`; [`LearnError::NotPositive`] or
    /// [`LearnError::NonFiniteParam`] for a non-positive or non-finite `tau`, `tau_s`, `theta` or
    /// `lambda`, and for `tau <= tau_s` — the kernel's normalisation divides by `tau - tau_s`, and
    /// the convention here is the physical one, a membrane slower than its synapses;
    /// [`LearnError::TimeConstantsEqual`] when the two are exactly equal, where the limit exists but
    /// is a different kernel.
    pub fn new(
        n_in: usize,
        tau: f64,
        tau_s: f64,
        theta: f64,
        lambda: f64,
    ) -> Result<Self, LearnError> {
        if n_in == 0 {
            return Err(LearnError::Empty { what: "afferents" });
        }
        let tau = positive("tau", tau)?;
        let tau_s = positive("tau_s", tau_s)?;
        let theta = positive("theta", theta)?;
        let lambda = positive("lambda", lambda)?;
        if tau == tau_s {
            return Err(LearnError::TimeConstantsEqual { tau });
        }
        if tau < tau_s {
            return Err(LearnError::NotPositive { what: "tau - tau_s", value: tau - tau_s });
        }
        let peak = (tau * tau_s / (tau - tau_s)) * (tau / tau_s).ln();
        let v0 = 1.0 / ((-peak / tau).exp() - (-peak / tau_s).exp());
        Ok(Self { w: vec![0.0; n_in], tau, tau_s, theta, lambda, v0 })
    }

    /// `tau = 15 ms`, `tau_s = tau / 4`, threshold one, with the learning rate the caller's.
    ///
    /// The two time constants are the values this review read as Gütig & Sompolinsky's simulation
    /// parameters; the ratio `tau_s = tau / 4` is the one the paper is consistently cited for, and
    /// the 15 ms is the value this implementation could not independently re-derive from the
    /// figures, so treat it as a transcription rather than as a fit. The
    /// **learning rate is not** supplied here, for the same reason [`crate::plasticity::PairStdp`]
    /// declines to supply Bi & Poo's amplitudes: the paper's `lambda` is quoted relative to a weight
    /// scale this implementation does not reproduce, so a number presented as "the paper's lambda"
    /// would be one this crate could not defend. The threshold is one because the weights are then
    /// in units of the threshold, which is the frame the paper's figures use.
    ///
    /// # Errors
    ///
    /// As [`Tempotron::new`].
    pub fn gutig_sompolinsky_2006(n_in: usize, lambda: f64) -> Result<Self, LearnError> {
        Self::new(n_in, 15e-3, 15e-3 / 4.0, 1.0, lambda)
    }

    /// The kernel's normalisation constant, so that `kernel(peak_time()) == 1`.
    #[must_use]
    pub fn v0(&self) -> f64 {
        self.v0
    }

    /// The lag, **seconds**, at which a single postsynaptic potential peaks:
    /// `tau tau_s / (tau - tau_s) * ln(tau / tau_s)`.
    ///
    /// The stationary point of `exp(-s/tau) - exp(-s/tau_s)`, in closed form. For the
    /// `tau_s = tau / 4` convention it is `tau * ln(4) / 3`, about 6.93 ms at `tau = 15 ms`.
    #[must_use]
    pub fn peak_time(&self) -> f64 {
        (self.tau * self.tau_s / (self.tau - self.tau_s)) * (self.tau / self.tau_s).ln()
    }

    /// One normalised postsynaptic potential `lag` seconds after a presynaptic spike.
    ///
    /// Exactly zero at and before the spike, exactly one at [`Tempotron::peak_time`], below one
    /// everywhere else. A non-finite `lag` gives a non-finite result; the constructors and
    /// [`Tempotron::voltage`] are where finiteness is enforced.
    #[must_use]
    pub fn kernel(&self, lag: f64) -> f64 {
        if lag > 0.0 { self.v0 * ((-lag / self.tau).exp() - (-lag / self.tau_s).exp()) } else { 0.0 }
    }

    /// The membrane potential at time `t` seconds, resting potential taken as zero.
    ///
    /// # Errors
    ///
    /// [`LearnError::ShapeMismatch`] if `pattern` does not have one entry per weight;
    /// [`LearnError::NonFiniteParam`] for a non-finite `t`; [`LearnError::NonFiniteInput`] naming
    /// the first non-finite spike time within an afferent.
    pub fn voltage(&self, pattern: &[Vec<f64>], t: f64) -> Result<f64, LearnError> {
        if pattern.len() != self.w.len() {
            return Err(LearnError::ShapeMismatch {
                what: "pattern",
                got: pattern.len(),
                want: self.w.len(),
            });
        }
        finite("t", t)?;
        let mut v = 0.0;
        for (i, times) in pattern.iter().enumerate() {
            finite_slice(times)?;
            let mut psp = 0.0;
            for &ts in times {
                psp += self.kernel(t - ts);
            }
            v += self.w[i] * psp;
        }
        Ok(v)
    }

    /// The largest membrane potential on the grid `0, dt, 2 dt, ...` up to `t_end`, and where it
    /// occurred, as `(v_max, t_max)` in the weights' unit and seconds.
    ///
    /// Ties go to the earliest time, deterministically.
    ///
    /// # Errors
    ///
    /// [`LearnError::NotPositive`] for a non-positive or non-finite `dt` or `t_end`, plus anything
    /// [`Tempotron::voltage`] returns.
    pub fn peak(
        &self,
        pattern: &[Vec<f64>],
        dt: f64,
        t_end: f64,
    ) -> Result<(f64, f64), LearnError> {
        let dt = positive("dt", dt)?;
        let t_end = positive("t_end", t_end)?;
        let steps = (t_end / dt).floor() as usize;
        let mut best = (f64::NEG_INFINITY, 0.0);
        for k in 0..=steps {
            let t = k as f64 * dt;
            let v = self.voltage(pattern, t)?;
            if v > best.0 {
                best = (v, t);
            }
        }
        Ok(best)
    }

    /// Whether the neuron fires at least once during the trial: `max_t V(t) >= theta`.
    ///
    /// # Errors
    ///
    /// As [`Tempotron::peak`].
    pub fn fires(&self, pattern: &[Vec<f64>], dt: f64, t_end: f64) -> Result<bool, LearnError> {
        Ok(self.peak(pattern, dt, t_end)?.0 >= self.theta)
    }

    /// One tempotron update. Returns whether the weights moved.
    ///
    /// **A correct decision changes nothing, exactly.** The rule is error-driven, so when the
    /// decision already matches `desired` this returns `Ok(false)` having touched nothing — not a
    /// small update, not a rounded-to-zero update.
    ///
    /// # Errors
    ///
    /// As [`Tempotron::peak`].
    pub fn train_once(
        &mut self,
        pattern: &[Vec<f64>],
        desired: bool,
        dt: f64,
        t_end: f64,
    ) -> Result<bool, LearnError> {
        let (v_max, t_max) = self.peak(pattern, dt, t_end)?;
        if (v_max >= self.theta) == desired {
            return Ok(false);
        }
        let sign = if desired { 1.0 } else { -1.0 };
        for i in 0..self.w.len() {
            let mut psp = 0.0;
            for &ts in &pattern[i] {
                psp += self.kernel(t_max - ts);
            }
            self.w[i] += sign * self.lambda * psp;
        }
        Ok(true)
    }

    /// Train over `epochs` passes of `set`, in the order given, and return the error rate after
    /// each pass.
    ///
    /// The returned rate is measured **during** the pass, before that pass's updates for the
    /// patterns still to come — which is the standard online convention and is also why the last
    /// entry can be non-zero on a run that ends perfectly classified. Use [`Tempotron::accuracy`]
    /// for the state of the neuron after training.
    ///
    /// # Errors
    ///
    /// [`LearnError::Empty`] for an empty `set`, plus anything [`Tempotron::train_once`] returns.
    pub fn train(
        &mut self,
        set: &[Example],
        epochs: usize,
        dt: f64,
        t_end: f64,
    ) -> Result<Vec<f64>, LearnError> {
        if set.is_empty() {
            return Err(LearnError::Empty { what: "training set" });
        }
        let mut hist = Vec::with_capacity(epochs);
        for _ in 0..epochs {
            let mut wrong = 0usize;
            for (pattern, desired) in set {
                if self.train_once(pattern, *desired, dt, t_end)? {
                    wrong += 1;
                }
            }
            hist.push(wrong as f64 / set.len() as f64);
        }
        Ok(hist)
    }

    /// Fraction of `set` the neuron currently decides correctly, in `[0, 1]`.
    ///
    /// # Errors
    ///
    /// [`LearnError::Empty`] for an empty `set`, plus anything [`Tempotron::fires`] returns.
    pub fn accuracy(&self, set: &[Example], dt: f64, t_end: f64) -> Result<f64, LearnError> {
        if set.is_empty() {
            return Err(LearnError::Empty { what: "evaluation set" });
        }
        let mut right = 0usize;
        for (pattern, desired) in set {
            if self.fires(pattern, dt, t_end)? == *desired {
                right += 1;
            }
        }
        Ok(right as f64 / set.len() as f64)
    }
}

// ---------------------------------------------------------------------------------------------
// ReSuMe
// ---------------------------------------------------------------------------------------------

/// Remote supervised learning: drive an output spike train toward a desired one.
///
/// Ponulak & Kasiński, *Supervised learning in spiking neural networks with `ReSuMe`: sequence
/// learning, classification, and spike shifting*, Neural Computation 22:467–510 (2010).
///
/// ```text
/// dw/dt = [ S_d(t) - S_o(t) ] * ( a + integral_0^inf W(s) S_in(t - s) ds )
/// ```
///
/// `S_d` is the **desired** output train, `S_o` the one the neuron actually produced, `S_in` the
/// presynaptic train, `W` an exponential learning window and `a` a spike-count term that does not
/// look at timing at all.
///
/// # Why this is the same object as STDP
///
/// Set `a = 0` and read the two halves separately. The desired train alone gives
/// `+W(t_d - t_pre)` per pair — a presynaptic spike followed by a postsynaptic one, potentiating,
/// with an exponential window: that **is** [`crate::plasticity::PairStdp`]'s potentiation branch.
/// The actual train alone gives exactly its negative, which is anti-Hebbian. `ReSuMe` is therefore
/// STDP with the teacher wired into the Hebbian term and the neuron's own output wired into an
/// anti-Hebbian one, and its fixed point is reached when the two cancel.
///
/// `resume_reduces_to_the_stdp_window` checks both statements as **equalities**, not tolerances: the
/// two implementations evaluate the same expression in the same order and agree to the last bit.
///
/// # The fixed point is exact
///
/// When the produced train equals the desired train, [`ReSuMe::delta_w`] returns `0.0` exactly. That
/// is a property of how it is computed, not of cancellation: the two correlation sums are formed by
/// identical operations on identical data and then subtracted, so they are bit-identical. Summing
/// signed terms into one accumulator instead would leave a residue of order the rounding of the
/// larger sum, and a rule with a non-zero fixed point drifts forever after it has converged.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReSuMe {
    /// The non-Hebbian term `a`, in the weights' unit. Positive values make the rule adjust the
    /// output spike *count* independently of timing. Any sign; zero recovers the pure window.
    pub a: f64,
    /// Window amplitude `A_d` at zero lag, in the weights' unit. Non-negative.
    pub amp: f64,
    /// Window time constant, **seconds**. Strictly positive.
    pub tau: f64,
    /// Learning rate, dimensionless. Strictly positive.
    pub eta: f64,
}

impl ReSuMe {
    /// # Errors
    ///
    /// [`LearnError::NonFiniteParam`] for a non-finite `a` or `amp`, [`LearnError::NotPositive`] for
    /// a negative `amp` or a non-positive `tau` or `eta`.
    pub fn new(a: f64, amp: f64, tau: f64, eta: f64) -> Result<Self, LearnError> {
        let a = finite("a", a)?;
        let amp = finite("amp", amp)?;
        if amp < 0.0 {
            return Err(LearnError::NotPositive { what: "amp", value: amp });
        }
        Ok(Self { a, amp, tau: positive("tau", tau)?, eta: positive("eta", eta)? })
    }

    /// The learning window `W(s) = amp * exp(-s / tau)` for `s > 0`, and `0.0` otherwise.
    ///
    /// `s` is the lag from the presynaptic spike to the spike being paired with it, in seconds.
    /// Zero lag returns zero, matching [`crate::plasticity::PairStdp::window`]'s convention, so that
    /// the two agree at the boundary as well as away from it.
    #[must_use]
    pub fn window(&self, s: f64) -> f64 {
        if s > 0.0 { self.amp * (-s / self.tau).exp() } else { 0.0 }
    }

    /// Sum of `window(post - pre)` over every pair with `pre < post`.
    fn correlation(&self, pre: &[f64], post: &[f64]) -> f64 {
        let mut s = 0.0;
        for &t_post in post {
            for &t_pre in pre {
                if t_pre < t_post {
                    s += self.window(t_post - t_pre);
                }
            }
        }
        s
    }

    /// The total weight change one trial produces on one synapse.
    ///
    /// All three arguments are spike times in **seconds**; none need be sorted. The result is in the
    /// weights' unit.
    ///
    /// # Errors
    ///
    /// [`LearnError::NonFiniteInput`] naming the first non-finite time in `pre`, then in `desired`,
    /// then in `actual`.
    pub fn delta_w(
        &self,
        pre: &[f64],
        desired: &[f64],
        actual: &[f64],
    ) -> Result<f64, LearnError> {
        finite_slice(pre)?;
        finite_slice(desired)?;
        finite_slice(actual)?;
        let count = desired.len() as f64 - actual.len() as f64;
        let hebbian = self.correlation(pre, desired) - self.correlation(pre, actual);
        Ok(self.eta * (self.a * count + hebbian))
    }
}

// ---------------------------------------------------------------------------------------------
// SpikeProp
// ---------------------------------------------------------------------------------------------

/// Bohte's alpha-function postsynaptic potential, `eps(s) = (s / tau) exp(1 - s / tau)`.
///
/// Normalised by construction: it peaks at exactly `1.0` at `s = tau`, which
/// `the_alpha_psp_peaks_at_exactly_one_at_its_time_constant` checks as an equality rather than to a
/// tolerance — `(tau/tau) * exp(1 - 1)` is `1.0 * exp(0.0)` in floating point too.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AlphaPsp {
    /// Time constant, **seconds**, and also the lag at which the potential peaks. Strictly positive.
    pub tau: f64,
}

impl AlphaPsp {
    /// # Errors
    ///
    /// [`LearnError::NotPositive`] or [`LearnError::NonFiniteParam`] for a `tau` that is not a
    /// finite strictly positive number.
    pub fn new(tau: f64) -> Result<Self, LearnError> {
        Ok(Self { tau: positive("tau", tau)? })
    }

    /// The potential `lag` seconds after a presynaptic spike; exactly zero at and before it.
    #[must_use]
    pub fn eval(&self, lag: f64) -> f64 {
        if lag > 0.0 { (lag / self.tau) * (1.0 - lag / self.tau).exp() } else { 0.0 }
    }

    /// `d eps / d lag`, in units of one per second. Exactly zero at and before the spike, and zero
    /// again at `lag == tau`, where the potential peaks.
    #[must_use]
    pub fn slope(&self, lag: f64) -> f64 {
        if lag > 0.0 {
            (1.0 - lag / self.tau).exp() * (1.0 - lag / self.tau) / self.tau
        } else {
            0.0
        }
    }
}

/// Gradient descent on a **spike time**.
///
/// Bohte, Kok & La Poutré, *Error-backpropagation in temporally encoded networks of spiking
/// neurons*, Neurocomputing 48:17–37 (2002).
///
/// One target neuron receives one spike from each afferent and fires the first time its membrane
/// `u(t) = sum_i w_i eps(t - t_i)` reaches threshold. The loss is `½ (t_out - t_target)²`, and the
/// derivative that makes the whole method work comes from differentiating the threshold condition
/// implicitly:
///
/// ```text
/// u(t_out) = theta   for all w   =>   d t_out / d w_i  =  - eps(t_out - t_i) / u'(t_out)
/// ```
///
/// That is [`SpikeProp::sensitivity`], and `the_spikeprop_sensitivity_matches_a_finite_difference`
/// checks it against a central difference of the actual threshold crossing — the formula against the
/// simulation, not the formula against itself.
///
/// # The failure mode is the whole story
///
/// The denominator is the membrane's slope at the crossing. It is positive at a rising crossing and
/// goes to zero as the membrane's peak sinks to the threshold — the instant at which the neuron
/// stops firing at all and the spike time ceases to exist as a function of the weights. Bohte et al.
/// report this as the method's central difficulty. [`SpikeProp::sensitivity`] returns `None` there,
/// and [`SpikeProp::first_spike`] returns `None` when there is no crossing, rather than either of
/// them producing the very large number that a small denominator would.
#[derive(Debug, Clone, PartialEq)]
pub struct SpikeProp {
    /// Synaptic weights, one per afferent, in the membrane's unit. Any sign.
    pub w: Vec<f64>,
    /// The postsynaptic potential shape shared by every afferent.
    pub psp: AlphaPsp,
    /// Firing threshold, in the same unit as [`SpikeProp::w`]. Strictly positive.
    pub theta: f64,
    /// Learning rate, in weight units per second of timing error. Strictly positive.
    ///
    /// **It is not a number near one.** The sensitivity is of order milliseconds per unit weight
    /// and the timing error is of order milliseconds, so their product is of order `1e-6`; a
    /// learning rate of one moves the weights by a millionth and the run looks like a failure to
    /// learn rather than like a mis-scaled step. `spikeprop_moves_the_output_spike_to_its_target`
    /// uses `1e4` and converges in fifty steps. This awkward scale is intrinsic to differentiating
    /// a time rather than an activation, and is one reason the surrogate-gradient formulation of
    /// [`crate::surrogate`] displaced it.
    pub eta: f64,
}

impl SpikeProp {
    /// # Errors
    ///
    /// [`LearnError::Empty`] for no weights; [`LearnError::NonFiniteInput`] for a non-finite weight;
    /// [`LearnError::NotPositive`] or [`LearnError::NonFiniteParam`] for a `tau`, `theta` or `eta`
    /// that is not a finite strictly positive number.
    pub fn new(w: Vec<f64>, tau: f64, theta: f64, eta: f64) -> Result<Self, LearnError> {
        if w.is_empty() {
            return Err(LearnError::Empty { what: "weights" });
        }
        finite_slice(&w)?;
        Ok(Self { w, psp: AlphaPsp::new(tau)?, theta: positive("theta", theta)?, eta: positive("eta", eta)? })
    }

    /// The membrane potential at time `t` seconds, given one presynaptic spike time per afferent.
    ///
    /// # Errors
    ///
    /// [`LearnError::ShapeMismatch`] if `pre` is not one time per weight;
    /// [`LearnError::NonFiniteInput`] naming the first non-finite time;
    /// [`LearnError::NonFiniteParam`] for a non-finite `t`.
    pub fn potential(&self, pre: &[f64], t: f64) -> Result<f64, LearnError> {
        if pre.len() != self.w.len() {
            return Err(LearnError::ShapeMismatch {
                what: "presynaptic times",
                got: pre.len(),
                want: self.w.len(),
            });
        }
        finite_slice(pre)?;
        finite("t", t)?;
        let mut u = 0.0;
        for i in 0..self.w.len() {
            u += self.w[i] * self.psp.eval(t - pre[i]);
        }
        Ok(u)
    }

    /// The first threshold crossing, to full floating-point precision, or `None` if the membrane
    /// never reaches the threshold on `[0, t_end]`.
    ///
    /// The grid of step `dt` only *brackets* the crossing; the returned time is then refined by 80
    /// bisections, which is far past the point where the interval stops shrinking, so the answer is
    /// the root of `u(t) = theta` rather than a grid point. That matters because this function is
    /// differentiated by finite differences in the test that validates
    /// [`SpikeProp::sensitivity`], and a grid-quantised root would make that derivative a staircase.
    ///
    /// # Errors
    ///
    /// [`LearnError::NotPositive`] for a non-positive or non-finite `dt` or `t_end`, plus anything
    /// [`SpikeProp::potential`] returns.
    pub fn first_spike(
        &self,
        pre: &[f64],
        dt: f64,
        t_end: f64,
    ) -> Result<Option<f64>, LearnError> {
        let dt = positive("dt", dt)?;
        let t_end = positive("t_end", t_end)?;
        let steps = (t_end / dt).floor() as usize;
        let mut lo = 0.0;
        let mut found = None;
        for k in 0..=steps {
            let t = k as f64 * dt;
            if self.potential(pre, t)? >= self.theta {
                found = Some(t);
                break;
            }
            lo = t;
        }
        let Some(hi0) = found else { return Ok(None) };
        if hi0 == 0.0 {
            return Ok(Some(0.0));
        }
        let mut hi = hi0;
        for _ in 0..80 {
            let mid = 0.5 * (lo + hi);
            if self.potential(pre, mid)? >= self.theta {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        Ok(Some(hi))
    }

    /// `d t_out / d w_i` for every afferent, in seconds per weight unit.
    ///
    /// Negative wherever the afferent contributed a positive potential: raising a weight makes the
    /// neuron fire *earlier*, which is the sign convention that surprises people reading `SpikeProp`
    /// for the first time.
    ///
    /// `None` when the membrane's slope at `t_out` is not strictly positive — the crossing is then
    /// not transversal and the spike time is not a differentiable function of the weights there.
    ///
    /// # Errors
    ///
    /// As [`SpikeProp::potential`], plus [`LearnError::NonFiniteParam`] for a non-finite `t_out`.
    pub fn sensitivity(
        &self,
        pre: &[f64],
        t_out: f64,
    ) -> Result<Option<Vec<f64>>, LearnError> {
        if pre.len() != self.w.len() {
            return Err(LearnError::ShapeMismatch {
                what: "presynaptic times",
                got: pre.len(),
                want: self.w.len(),
            });
        }
        finite_slice(pre)?;
        finite("t_out", t_out)?;
        let mut slope = 0.0;
        for i in 0..self.w.len() {
            slope += self.w[i] * self.psp.slope(t_out - pre[i]);
        }
        if !(slope > 0.0) || !slope.is_finite() {
            return Ok(None);
        }
        let mut g = Vec::with_capacity(self.w.len());
        for i in 0..self.w.len() {
            g.push(-self.psp.eval(t_out - pre[i]) / slope);
        }
        Ok(Some(g))
    }

    /// One gradient-descent step on `½ (t_out - target)²`. Returns the timing error **before** the
    /// step, in seconds, or `None` if the neuron did not fire or the crossing was not transversal.
    ///
    /// # Errors
    ///
    /// As [`SpikeProp::first_spike`] and [`SpikeProp::sensitivity`], plus
    /// [`LearnError::NonFiniteParam`] for a non-finite `target`.
    pub fn train_once(
        &mut self,
        pre: &[f64],
        target: f64,
        dt: f64,
        t_end: f64,
    ) -> Result<Option<f64>, LearnError> {
        finite("target", target)?;
        let Some(t_out) = self.first_spike(pre, dt, t_end)? else { return Ok(None) };
        let Some(g) = self.sensitivity(pre, t_out)? else { return Ok(None) };
        let err = t_out - target;
        for i in 0..self.w.len() {
            self.w[i] -= self.eta * err * g[i];
        }
        Ok(Some(err))
    }
}

// ---------------------------------------------------------------------------------------------
// FORCE
// ---------------------------------------------------------------------------------------------

/// `FORCE` learning: a linear readout trained online by recursive least squares.
///
/// Sussillo & Abbott, *Generating coherent patterns of activity from chaotic neural networks*,
/// Neuron 63:544–557 (2009), applied to networks of spiking neurons by Nicola & Clopath,
/// *Supervised learning in spiking neural networks with `FORCE` training*, Nature Communications
/// 8:2208 (2017).
///
/// `FORCE` is the opposite trade to e-prop. Nothing is approximated: after `k` updates the weights
/// are **exactly** the ridge-regression solution over all `k` samples seen so far,
///
/// ```text
/// w_k = ( ridge I + sum_j r_j r_jT )^-1  sum_j d_j r_j
/// ```
///
/// and [`Force::inverse_correlation`] holds that inverse explicitly. The price is quadratic state:
/// an `n × n` matrix for `n` basis signals, updated in full every step. `force_matches_the_closed_form_ridge_solution`
/// checks both statements against a Cholesky factorisation of the batch problem, which is a wholly
/// different computation reaching the same answer.
///
/// In Nicola & Clopath the vector `r` is the filtered spike train of a spiking network, so the
/// readout is fed exactly what a downstream population would receive. What makes it "`FORCE`" rather
/// than plain least squares is that the corrections are applied *while the network runs*, and are
/// large enough from the first step that the output never departs far from the target — the error
/// stays small, so the network is never trained on a trajectory it will not visit again.
///
/// # What this implementation does not do
///
/// It trains a readout, not the recurrent weights, and it has no forgetting factor, so every sample
/// counts equally forever. A non-stationary target needs one, and this review did not locate a
/// principled default for it in the source papers.
#[derive(Debug, Clone, PartialEq)]
pub struct Force {
    /// Readout weights, `n` long, in the target's unit per unit of `r`. Start at zero.
    pub w: Vec<f64>,
    /// The ridge parameter the inverse correlation matrix was seeded with, `P(0) = I / ridge`.
    /// Strictly positive. Larger means slower, better-conditioned learning.
    pub ridge: f64,
    /// Row-major `n × n` inverse correlation matrix.
    p: Vec<f64>,
    /// Basis dimension.
    n: usize,
    /// Updates applied so far.
    updates: u64,
}

impl Force {
    /// A readout of `n` basis signals, with `P(0) = I / ridge` and `w(0) = 0`.
    ///
    /// # Errors
    ///
    /// [`LearnError::Empty`] for `n == 0`; [`LearnError::NotPositive`] or
    /// [`LearnError::NonFiniteParam`] for a `ridge` that is not a finite strictly positive number.
    pub fn new(n: usize, ridge: f64) -> Result<Self, LearnError> {
        if n == 0 {
            return Err(LearnError::Empty { what: "basis" });
        }
        let ridge = positive("ridge", ridge)?;
        let mut p = vec![0.0; n * n];
        for i in 0..n {
            p[i * n + i] = 1.0 / ridge;
        }
        Ok(Self { w: vec![0.0; n], ridge, p, n, updates: 0 })
    }

    /// Basis dimension.
    #[must_use]
    pub fn n(&self) -> usize {
        self.n
    }

    /// Updates applied so far.
    #[must_use]
    pub fn updates(&self) -> u64 {
        self.updates
    }

    /// The inverse correlation matrix `P`, row-major `n × n`.
    ///
    /// Equal to `(ridge I + sum_j r_j r_jT)^-1` over every sample passed to [`Force::update`], which
    /// is an exact identity (Sherman–Morrison applied once per sample), not a limit.
    #[must_use]
    pub fn inverse_correlation(&self) -> &[f64] {
        &self.p
    }

    /// The readout's current output for basis vector `r`.
    ///
    /// # Errors
    ///
    /// [`LearnError::ShapeMismatch`] for a wrong length; [`LearnError::NonFiniteInput`] naming the
    /// first non-finite element.
    pub fn output(&self, r: &[f64]) -> Result<f64, LearnError> {
        if r.len() != self.n {
            return Err(LearnError::ShapeMismatch { what: "basis vector", got: r.len(), want: self.n });
        }
        finite_slice(r)?;
        let mut y = 0.0;
        for i in 0..self.n {
            y += self.w[i] * r[i];
        }
        Ok(y)
    }

    /// One recursive-least-squares update. Returns the error **before** the update, in the target's
    /// unit.
    ///
    /// The a-priori error is what `FORCE` reports as its learning curve, and it is also what makes
    /// the closed form above hold: the update uses `w` from before the step and `P` from after it,
    /// which are related by `P_k r_k = P_{k-1} r_k / (1 + r_kT P_{k-1} r_k)`.
    ///
    /// # Errors
    ///
    /// [`LearnError::ShapeMismatch`] for a wrong-length `r`; [`LearnError::NonFiniteInput`] naming
    /// the first non-finite element of `r`; [`LearnError::NonFiniteParam`] for a non-finite
    /// `target`; [`LearnError::Singular`] if `1 + rᵀPr` leaves the positive reals, which cannot
    /// happen while `P` is positive definite and therefore reports that it is not.
    pub fn update(&mut self, r: &[f64], target: f64) -> Result<f64, LearnError> {
        if r.len() != self.n {
            return Err(LearnError::ShapeMismatch { what: "basis vector", got: r.len(), want: self.n });
        }
        finite_slice(r)?;
        finite("target", target)?;
        let n = self.n;
        let mut pr = vec![0.0; n];
        for i in 0..n {
            let mut s = 0.0;
            for j in 0..n {
                s += self.p[i * n + j] * r[j];
            }
            pr[i] = s;
        }
        let mut q = 0.0;
        for i in 0..n {
            q += r[i] * pr[i];
        }
        let denom = 1.0 + q;
        if !(denom > 0.0) || !denom.is_finite() {
            return Err(LearnError::Singular { step: self.updates, value: denom });
        }
        let c = 1.0 / denom;
        let mut err = -target;
        for i in 0..n {
            err += self.w[i] * r[i];
        }
        for i in 0..n {
            let ci = c * pr[i];
            for j in 0..n {
                self.p[i * n + j] -= ci * pr[j];
            }
            self.w[i] -= ci * err;
        }
        self.updates += 1;
        Ok(err)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        AlphaPsp, Eligibility, EpropConfig, Example, Force, Jacobian, LearnError, ReSuMe, SpikeProp,
        Tempotron, bptt_state_words, compare_to_bptt, eprop_batch, eprop_grad_from_dlogits,
        eprop_state_words, logits_streaming,
    };
    use crate::plasticity::{Bounds, PairStdp, WeightRule};
    use crate::reservoir::cholesky;
    use crate::rng::Rng;
    use crate::surrogate::{
        Adam, ArcTan, DelayedXor, LifLayerSpec, Rectangular, Scaled, SpikeFn, Surrogate,
        cross_entropy,
    };

    // -----------------------------------------------------------------------------------------
    // Eligibility traces
    // -----------------------------------------------------------------------------------------

    /// (c) The trace must decay as `exp(-t / tau)`, not as `(1 - dt/tau)^k`, and the difference
    /// between those two is exactly what this catches: at `dt = tau/20` over one time constant the
    /// Euler form is 7.4% high, which is four orders of magnitude outside this tolerance.
    #[test]
    fn an_eligibility_trace_decays_exactly_as_its_time_constant_says() {
        let tau = 20e-3;
        let dt = 1e-3;
        let mut e = Eligibility::new(tau, dt).expect("positive");
        e.step(1.0);
        assert_eq!(e.value, 1.0, "a unit input into a rested trace must leave exactly one");
        let start = e.value;
        for k in 1..=200u32 {
            e.step(0.0);
            let want = start * (-(f64::from(k) * dt) / tau).exp();
            let rel = (e.value - want).abs() / want;
            assert!(rel < 1e-12, "step {k}: online {} vs closed form {want}", e.value);
        }
        // And the same statement read the other way: the half-life is tau ln 2, exactly.
        let mut h = Eligibility::new(tau, dt).expect("positive");
        h.step(1.0);
        assert!((h.closed_form(tau * std::f64::consts::LN_2) - 0.5).abs() < 1e-15);
    }

    /// Decay-then-add, not add-then-decay. A single input must leave the trace at exactly one, and
    /// `after(0)` must be the value itself.
    #[test]
    fn a_trace_composes_over_gaps() {
        let mut e = Eligibility::new(7e-3, 5e-4).expect("positive");
        e.step(1.0);
        e.step(0.0);
        e.step(0.0);
        let three_at_once = e.after(0);
        assert_eq!(three_at_once, e.value);
        // Ten more one at a time against ten in one closed-form jump.
        let jumped = e.after(10);
        for _ in 0..10 {
            e.step(0.0);
        }
        assert!((jumped - e.value).abs() / e.value < 1e-14, "{jumped} vs {}", e.value);
        assert_eq!(e.decay(), (-e.dt / e.tau).exp());
        e.clear();
        assert_eq!(e.value, 0.0);
        assert_eq!(e.after(1_000_000), 0.0);
        assert_eq!(e.closed_form(1.0), 0.0);
    }

    #[test]
    fn a_trace_refuses_a_zero_time_constant() {
        assert!(matches!(
            Eligibility::new(0.0, 1e-3),
            Err(LearnError::NotPositive { what: "tau", .. })
        ));
        assert!(matches!(
            Eligibility::new(1e-3, f64::NAN),
            Err(LearnError::NonFiniteParam { what: "dt", .. })
        ));
    }

    // -----------------------------------------------------------------------------------------
    // E-prop
    // -----------------------------------------------------------------------------------------

    fn xor_pattern() -> Vec<f64> {
        let task = DelayedXor::default();
        task.patterns().expect("the windows fit")[2].0.clone()
    }

    fn spec(recurrent: bool, n_rec: usize) -> LifLayerSpec {
        LifLayerSpec { n_rec, n_out: 2, recurrent, seed: 3, ..LifLayerSpec::default() }
    }

    /// (a), the exact half. With no recurrent connections there are no paths through other neurons
    /// to drop, and with the full per-neuron Jacobian the eligibility carries the soft reset — so
    /// e-prop is not an approximation of BPTT here, it IS BPTT.
    ///
    /// Checked twice: under `Smooth`, where the reverse-mode gradient is the exact gradient of a
    /// real function, and under `Heaviside`, where both sides use the same surrogate. The algebra
    /// does not care which, and a version that only held for one of them would be a version that
    /// had smuggled in a property of the surrogate.
    #[test]
    fn the_eprop_gradient_is_exactly_bptt_for_a_feedforward_layer() {
        let layer = spec(false, 12).build().expect("valid");
        let sur = ArcTan::default();
        let x = xor_pattern();
        for (spike_fn, n_cmp) in [(SpikeFn::Smooth, 48), (SpikeFn::Heaviside, 38)] {
            let cfg = EpropConfig { jacobian: Jacobian::Full, spike_fn };
            let a = compare_to_bptt(&layer, &sur, &x, 1, cfg).expect("non-zero gradient");
            assert!(a.relative < 1e-9, "{spike_fn:?}: relative error {}", a.relative);
            assert!(1.0 - a.cosine < 1e-12, "{spike_fn:?}: cosine {}", a.cosine);
            assert_eq!(a.sign_agreement, 1.0, "{spike_fn:?}: a coordinate had the wrong sign");
            // The norms have to match too: a cosine of one with a scale error is a real failure
            // mode for a truncated gradient, and the cosine alone cannot see it.
            let rel_norm = (a.eprop_norm - a.bptt_norm).abs() / a.bptt_norm;
            assert!(rel_norm < 1e-9, "{spike_fn:?}: norms {} vs {}", a.eprop_norm, a.bptt_norm);
            // 12 input weights, 24 readout weights and 12 biases could be non-zero, and the
            // frozen recurrent block is excluded by construction. Under `Smooth` all 48 are, since
            // every unit emits a real number; under `Heaviside` ten of the readout weights read a
            // unit that never spikes and are exactly zero. Asserted rather than bounded so that a
            // change which quietly stops comparing anything cannot pass this test.
            assert_eq!(a.n_compared, n_cmp, "{spike_fn:?}: coordinates compared");
        }
    }

    /// The second exact case, and the only test that pins the self-connection term.
    ///
    /// A **single** recurrent unit with `recurrent = true` has exactly one recurrent path — its own
    /// autapse — so the diagonal of the state-transition Jacobian is the whole Jacobian again and
    /// [`Jacobian::Full`] must reproduce BPTT exactly, just as it does with no recurrence at all.
    /// Measured 1.9e-16 under `Heaviside` and 3.2e-16 under `Smooth`.
    ///
    /// Deleting the `V[j][j] * psi` term leaves every other test in this module passing — the
    /// feedforward tests never reach it and the 16-unit brackets are too wide to see it — and moves
    /// this one from 1.9e-16 to 1.3e-1.
    #[test]
    fn a_single_self_connected_unit_is_also_exact() {
        let layer = spec(true, 1).build().expect("valid");
        let sur = ArcTan::default();
        let x = xor_pattern();
        for spike_fn in [SpikeFn::Heaviside, SpikeFn::Smooth] {
            let full = EpropConfig { jacobian: Jacobian::Full, spike_fn };
            let a = compare_to_bptt(&layer, &sur, &x, 1, full).expect("non-zero gradient");
            assert!(a.relative < 1e-12, "{spike_fn:?}: relative {}", a.relative);
            assert_eq!(a.sign_agreement, 1.0, "{spike_fn:?}");
            // The autapse is carrying real weight here, so the line above is not passing because
            // the term happened to be zero.
            let leak = EpropConfig { jacobian: Jacobian::Leak, spike_fn };
            let b = compare_to_bptt(&layer, &sur, &x, 1, leak).expect("non-zero gradient");
            assert!(b.relative > 1e-2, "{spike_fn:?}: Leak was already exact at {}", b.relative);
        }
        assert!(layer.p[layer.idx_v(0, 0)].abs() > 1e-3, "the autapse weight was negligible");
    }

    /// The guard on the test above. If the reset term were worth nothing, `Jacobian::Full` would be
    /// decorative and the exactness above would be an accident of a term that never fires. It is
    /// worth 1.6% of the gradient's length.
    #[test]
    fn dropping_the_reset_term_changes_the_feedforward_gradient() {
        let layer = spec(false, 12).build().expect("valid");
        let sur = ArcTan::default();
        let x = xor_pattern();
        let cfg = EpropConfig { jacobian: Jacobian::Leak, spike_fn: SpikeFn::Heaviside };
        let a = compare_to_bptt(&layer, &sur, &x, 1, cfg).expect("non-zero gradient");
        // 3.382 measured. The interval is wide enough not to flake on a compiler's reassociation
        // and narrow enough that a change of substance moves out of it.
        assert!(a.relative > 3.0 && a.relative < 3.8, "relative {}", a.relative);
        assert!(a.cosine > 0.75 && a.cosine < 0.85, "cosine {}", a.cosine);
        // The gradient is not merely rotated, it is 4.1 times too LONG — a rule that pointed the
        // right way with the wrong scale would still descend, and this one would overshoot.
        let stretch = a.eprop_norm / a.bptt_norm;
        assert!(stretch > 3.5 && stretch < 4.6, "eprop/bptt norm ratio {stretch}");
    }

    /// The finding the module doc leads with in its second table: neither truncation is a property
    /// of e-prop on its own. Both are multiplied by the pseudo-derivative, so scaling the surrogate
    /// down scales the disagreement down with it, roughly linearly.
    ///
    /// This is also the guard on every number in the first table. If the reported agreement did not
    /// move with the surrogate's gain, the explanation offered for those numbers would be wrong.
    #[test]
    fn the_eprop_error_is_set_by_the_surrogates_peak() {
        let x = xor_pattern();
        let layer = spec(true, 16).build().expect("valid");
        assert_eq!(ArcTan::default().peak(), 1.0, "the reference surrogate's peak moved");
        let mut prev = f64::INFINITY;
        for gain in [1.0, 0.3, 0.1, 0.03] {
            let sur = Scaled::new(Box::new(ArcTan::default()), gain).expect("positive gain");
            let cfg = EpropConfig { jacobian: Jacobian::Leak, spike_fn: SpikeFn::Heaviside };
            let a = compare_to_bptt(&layer, &sur, &x, 1, cfg).expect("non-zero gradient");
            assert!(a.relative < prev, "gain {gain}: relative {} did not fall", a.relative);
            // Roughly linear: a tenfold reduction in the peak must buy at least fivefold in the
            // error, or the explanation in the module doc is not the right one.
            if prev.is_finite() {
                assert!(a.relative < 0.6 * prev, "gain {gain}: only {} from {prev}", a.relative);
            }
            prev = a.relative;
        }
        assert!(prev < 0.02, "the smallest surrogate still disagreed by {prev}");
    }

    /// (a), the approximate half — and the number the module doc reports.
    ///
    /// In a recurrent network the dropped paths are the cross-neuron ones and they dominate. The
    /// assertions bracket the measured figures from both sides, so that a change which *improves*
    /// the agreement also fails and has to be looked at rather than silently accepted.
    #[test]
    fn the_eprop_gradient_tracks_bptt_in_a_recurrent_network() {
        let layer = spec(true, 16).build().expect("valid");
        let sur = ArcTan::default();
        let x = xor_pattern();
        // Measured: Leak 0.6824 / 1.9652, Full 0.9722 / 0.2344, both at sign agreement 0.9435.
        // Bracketed from both sides so that a change which IMPROVES the agreement also fails and
        // has to be looked at rather than silently accepted.
        for (jacobian, cos_lo, cos_hi, rel_lo, rel_hi) in [
            (Jacobian::Leak, 0.65, 0.72, 1.85, 2.10),
            (Jacobian::Full, 0.96, 0.98, 0.20, 0.27),
        ] {
            let cfg = EpropConfig { jacobian, spike_fn: SpikeFn::Heaviside };
            let a = compare_to_bptt(&layer, &sur, &x, 1, cfg).expect("non-zero gradient");
            assert!(a.cosine > cos_lo && a.cosine < cos_hi, "{jacobian:?}: cosine {}", a.cosine);
            assert!(
                a.relative > rel_lo && a.relative < rel_hi,
                "{jacobian:?}: relative {}",
                a.relative
            );
            assert!(a.sign_agreement > 0.9, "{jacobian:?}: sign agreement {}", a.sign_agreement);
            assert_eq!(a.n_params, layer.p.len());
            assert_eq!(a.n_compared, 248);
        }
    }

    /// The one block that is exact even with recurrence: the readout weights. Their gradient is a
    /// kappa-filtered spike trace times the readout error, with nothing truncated anywhere, so this
    /// is an equality and not a tolerance on the approximation.
    #[test]
    fn the_readout_block_of_the_eprop_gradient_is_exactly_the_bptt_one() {
        let layer = spec(true, 16).build().expect("valid");
        let sur = ArcTan::default();
        let x = xor_pattern();
        let cfg = EpropConfig::default();
        let tr = layer.forward(&sur, &x, cfg.spike_fn).expect("valid");
        let (_, d) = cross_entropy(&tr.logits, 1).expect("valid target");
        let gb = layer.backward(&sur, &x, &tr, &d).expect("valid");
        let ge = eprop_grad_from_dlogits(&layer, &sur, &x, &d, cfg).expect("valid");
        let mut worst = 0.0f64;
        let mut scale = 0.0f64;
        for c in 0..layer.n_out {
            for j in 0..layer.n_rec {
                let k = layer.idx_r(c, j);
                worst = worst.max((gb[k] - ge[k]).abs());
                scale = scale.max(gb[k].abs());
            }
        }
        assert!(scale > 1e-6, "the readout gradient was zero, so this test checked nothing");
        assert!(worst / scale < 1e-12, "readout block differed by {worst} against scale {scale}");
    }

    /// [`GradAgreement::sign_agreement`] means the **strict** sign, and a coordinate where e-prop
    /// returns exactly zero against a non-zero BPTT entry is a disagreement, not a tie.
    ///
    /// The construction makes such coordinates exist on purpose: zeroing unit 0's readout weights
    /// in a recurrent layer cuts its learning signal to exactly zero, so every e-prop gradient that
    /// unit owns is exactly zero — while BPTT still credits it through the recurrent synapses that
    /// carry its spikes into the other units. The field is then recomputed here from the two raw
    /// gradients and compared as an equality, which is what pins the predicate.
    #[test]
    fn a_coordinate_where_eprop_is_exactly_zero_is_not_counted_as_agreeing() {
        let mut layer = spec(true, 8).build().expect("valid");
        for c in 0..layer.n_out {
            let k = layer.idx_r(c, 0);
            layer.p[k] = 0.0;
        }
        let sur = ArcTan::default();
        let x = xor_pattern();
        let cfg = EpropConfig::default();
        let tr = layer.forward(&sur, &x, cfg.spike_fn).expect("valid");
        let (_, d) = cross_entropy(&tr.logits, 1).expect("valid target");
        let gb = layer.backward(&sur, &x, &tr, &d).expect("valid");
        let ge = eprop_grad_from_dlogits(&layer, &sur, &x, &d, cfg).expect("valid");

        // The construction did what it claims: unit 0's own parameters are exactly zero under
        // e-prop and not under BPTT.
        let b0 = layer.idx_b(0);
        assert_eq!(ge[b0], 0.0, "unit 0's bias gradient was supposed to be exactly zero");
        assert!(gb[b0].abs() > 1e-9, "BPTT gave unit 0 nothing either, so nothing is being tested");

        let mut compared = 0usize;
        let mut agree = 0usize;
        let mut ties = 0usize;
        for (&a, &b) in gb.iter().zip(ge.iter()) {
            if a != 0.0 {
                compared += 1;
                if a * b > 0.0 {
                    agree += 1;
                } else if b == 0.0 {
                    ties += 1;
                }
            }
        }
        assert!(ties > 0, "no exactly-zero e-prop coordinate survived, so the predicate is untested");
        let got = compare_to_bptt(&layer, &sur, &x, 1, cfg).expect("non-zero gradient");
        assert_eq!(got.n_compared, compared);
        assert_eq!(got.sign_agreement, agree as f64 / compared as f64);
        assert!(got.sign_agreement < 1.0, "every coordinate agreed, including the ties");
    }

    /// (d) With the learning signal identically zero, no weight moves — exactly, not approximately.
    ///
    /// This is the property that makes the factorisation a factorisation. If the eligibility were
    /// accumulated into the gradient without being multiplied by the learning signal — a plausible
    /// transcription slip, and one that still trains on some tasks — every entry here would be
    /// large.
    #[test]
    fn a_zero_learning_signal_moves_no_weight_at_all() {
        let layer = spec(true, 8).build().expect("valid");
        let sur = ArcTan::default();
        let x = xor_pattern();
        for jacobian in [Jacobian::Leak, Jacobian::Full] {
            let cfg = EpropConfig { jacobian, spike_fn: SpikeFn::Heaviside };
            let g = eprop_grad_from_dlogits(&layer, &sur, &x, &[0.0, 0.0], cfg).expect("valid");
            assert_eq!(g.len(), layer.p.len());
            for (k, &v) in g.iter().enumerate() {
                assert!(v == 0.0, "{jacobian:?}: parameter {k} moved by {v}");
            }
            // ... and the same call with a real learning signal does not, so the line above is not
            // reporting that the whole function returns zeros.
            let g2 = eprop_grad_from_dlogits(&layer, &sur, &x, &[0.4, -0.4], cfg).expect("valid");
            assert!(g2.iter().any(|v| v.abs() > 1e-9), "{jacobian:?}: nothing moved either way");
        }
    }

    /// The streaming readout and the stored-trace one are the same arithmetic in the same order, so
    /// they agree bit for bit. That is what licenses `eprop_grad` to skip the trace entirely.
    #[test]
    fn streaming_logits_are_bit_identical_to_the_stored_trace() {
        let sur = ArcTan::default();
        let x = xor_pattern();
        for recurrent in [false, true] {
            let layer = spec(recurrent, 16).build().expect("valid");
            for spike_fn in [SpikeFn::Heaviside, SpikeFn::Smooth] {
                let a = layer.forward(&sur, &x, spike_fn).expect("valid").logits;
                let b = logits_streaming(&layer, &sur, &x, spike_fn).expect("valid");
                assert_eq!(a, b, "recurrent {recurrent}, {spike_fn:?}");
            }
        }
    }

    /// The memory claim, as arithmetic rather than as an adjective — including the half of it the
    /// literature does not report, which is that e-prop's state is the LARGER of the two on a short
    /// episode.
    #[test]
    fn eligibility_state_is_independent_of_episode_length() {
        let layer = spec(true, 16).build().expect("valid");
        let e = eprop_state_words(&layer);
        assert_eq!(e, 660);
        let short = bptt_state_words(&layer, 5).expect("no overflow");
        let long = bptt_state_words(&layer, 100).expect("no overflow");
        assert_eq!(short, 250);
        assert_eq!(long, 5_000);
        assert!(short < e, "BPTT should be cheaper than e-prop on a 5-step episode");
        assert!(long > 7 * e, "e-prop should win by a wide margin at 100 steps");
        // A feedforward layer has no V block, so its eligibility is far cheaper and the crossover
        // moves to a much shorter episode. That is the fan-in dependence the module doc claims.
        let ff = spec(false, 16).build().expect("valid");
        assert_eq!(eprop_state_words(&ff), 2 * 16 * 2 + 5 * 16 + 4);
        // The crossover, found rather than asserted from a formula, so the doc's "below 14 steps"
        // is a measured statement.
        let crossover = (1..200)
            .find(|&t| bptt_state_words(&layer, t).expect("no overflow") > e)
            .expect("there is one");
        assert_eq!(crossover, 14);
        let ff_crossover = (1..200)
            .find(|&t| bptt_state_words(&ff, t).expect("no overflow") > eprop_state_words(&ff))
            .expect("there is one");
        assert_eq!(ff_crossover, 3);
        assert_eq!(bptt_state_words(&layer, usize::MAX), None);
    }

    /// (a), the part that matters in practice: an approximate gradient is only interesting if it
    /// trains. This is the same task, the same layer size and the same optimiser as
    /// `a_recurrent_lif_layer_learns_delayed_xor` in `crate::surrogate`, with the BPTT call swapped
    /// for the e-prop one and nothing else changed.
    #[test]
    fn eprop_learns_delayed_xor() {
        let task = DelayedXor::default();
        let batch = task.patterns().expect("the windows fit");
        let spec = LifLayerSpec {
            n_in: task.n_in(),
            n_rec: 48,
            n_out: task.n_out(),
            recurrent: true,
            w_scale: 0.4,
            b_init: 3e-3,
            seed: 8,
            ..LifLayerSpec::default()
        };
        let mut layer = spec.build().expect("valid spec");
        let sur = ArcTan::default();
        let cfg = EpropConfig::default();

        let (loss0, _) = eprop_batch(&layer, &sur, &batch, cfg).expect("non-empty");
        let correct0 = batch
            .iter()
            .filter(|(x, y)| layer.predict(&sur, x).expect("valid") == *y)
            .count();
        assert!(loss0 > 0.68, "initial loss {loss0} was already below chance");
        assert!(correct0 <= 2, "{correct0} of 4 correct before any training");

        let mut opt = Adam::new(layer.n_params(), 1e-2).expect("valid");
        let mut hist = Vec::new();
        for _ in 0..300 {
            let (l, g) = eprop_batch(&layer, &sur, &batch, cfg).expect("non-empty");
            hist.push(l);
            opt.step(&mut layer.p, &g).expect("finite gradient");
        }
        let (final_loss, _) = eprop_batch(&layer, &sur, &batch, cfg).expect("non-empty");
        let correct = batch
            .iter()
            .filter(|(x, y)| layer.predict(&sur, x).expect("valid") == *y)
            .count();
        assert!(hist.iter().all(|l| l.is_finite()), "the loss history left the finite numbers");
        assert!(final_loss < 0.1, "loss fell only to {final_loss} (from {loss0})");
        assert_eq!(correct, 4, "only {correct} of 4 patterns classified correctly");
        // It did not solve the task by going silent and letting the biases decide.
        let tr = layer.forward(&sur, &batch[3].0, SpikeFn::Heaviside).expect("valid");
        assert!(tr.spike_count() > 0.0, "the trained network is silent");
    }

    #[test]
    fn eprop_refuses_a_malformed_call() {
        let layer = spec(true, 4).build().expect("valid");
        let sur = ArcTan::default();
        let x = xor_pattern();
        let cfg = EpropConfig::default();
        assert!(matches!(
            eprop_grad_from_dlogits(&layer, &sur, &x, &[0.1], cfg),
            Err(LearnError::ShapeMismatch { what: "d_logits", got: 1, want: 2 })
        ));
        assert!(matches!(
            eprop_grad_from_dlogits(&layer, &sur, &[], &[0.1, 0.2], cfg),
            Err(LearnError::ShapeMismatch { what: "input", .. })
        ));
        assert!(matches!(
            eprop_grad_from_dlogits(&layer, &sur, &x, &[f64::NAN, 0.0], cfg),
            Err(LearnError::NonFiniteInput { index: 0, .. })
        ));
        let mut bad = x.clone();
        bad[3] = f64::INFINITY;
        assert!(matches!(
            eprop_grad_from_dlogits(&layer, &sur, &bad, &[0.1, 0.2], cfg),
            Err(LearnError::NonFiniteInput { index: 3, .. })
        ));
        assert!(matches!(eprop_batch(&layer, &sur, &[], cfg), Err(LearnError::Empty { .. })));
    }

    /// A zero reference gradient has no direction, so the comparison refuses rather than reporting
    /// either perfect or zero agreement. Reached by a layer whose readout weights are all zero: the
    /// logits are then equal, the softmax is uniform, and every path to the parameters is cut.
    #[test]
    fn comparing_against_a_zero_gradient_refuses() {
        // Two cuts at once. Zeroing the readout weights kills every path from a spike to the loss,
        // and parking the biases far below threshold stops the units spiking at all, so the readout
        // block's own gradient — a filtered spike trace times the error — is zero too. With a
        // compactly supported surrogate the pseudo-derivative is zero out there as well, so the
        // BPTT gradient is exactly the zero vector rather than a very small one.
        let mut layer = spec(false, 4).build().expect("valid");
        for c in 0..layer.n_out {
            for j in 0..layer.n_rec {
                let k = layer.idx_r(c, j);
                layer.p[k] = 0.0;
            }
        }
        for j in 0..layer.n_rec {
            let k = layer.idx_b(j);
            layer.p[k] = -20.0;
        }
        let sur = Rectangular::default();
        let x = xor_pattern();
        let tr = layer.forward(&sur, &x, SpikeFn::Heaviside).expect("valid");
        assert_eq!(tr.spike_count(), 0.0, "the network was supposed to be silent");
        assert_eq!(
            compare_to_bptt(&layer, &sur, &x, 0, EpropConfig::default()),
            Err(LearnError::ZeroGradient)
        );
        // ... and it is the gradient that is zero, not the comparison that always refuses.
        let alive = spec(false, 4).build().expect("valid");
        assert!(compare_to_bptt(&alive, &sur, &x, 0, EpropConfig::default()).is_ok());
    }

    // -----------------------------------------------------------------------------------------
    // Tempotron
    // -----------------------------------------------------------------------------------------

    /// The kernel's normalisation, against the closed-form peak time. Both statements are exact:
    /// zero at the spike because `exp(0) - exp(0)` is `0.0` in floating point too, and one at the
    /// peak because `v0` is defined as the reciprocal of the value there.
    #[test]
    fn the_tempotron_kernel_peaks_at_exactly_one() {
        let t = Tempotron::gutig_sompolinsky_2006(4, 1e-3).expect("valid");
        assert_eq!(t.kernel(0.0), 0.0);
        assert_eq!(t.kernel(-1e-3), 0.0);
        let peak = t.peak_time();
        // tau ln(4) / 3 for the tau_s = tau/4 convention, from the closed form in the doc.
        let want = t.tau * 4.0f64.ln() / 3.0;
        assert!((peak - want).abs() < 1e-15, "peak time {peak} vs {want}");
        assert!((t.kernel(peak) - 1.0).abs() < 1e-12, "kernel peaked at {}", t.kernel(peak));
        // ... and nowhere else does it exceed one.
        for k in 0..20_000 {
            let s = k as f64 * 5e-6;
            assert!(t.kernel(s) <= 1.0 + 1e-12, "kernel({s}) = {}", t.kernel(s));
        }
    }

    fn tempotron_set(rng: &mut Rng, n_in: usize, n_pat: usize, span: f64) -> Vec<Example> {
        let mut set = Vec::with_capacity(n_pat);
        for k in 0..n_pat {
            let mut pattern = Vec::with_capacity(n_in);
            for _ in 0..n_in {
                let mut times: Vec<f64> = (0..3).map(|_| rng.next_f64() * span).collect();
                times.sort_by(f64::total_cmp);
                pattern.push(times);
            }
            set.push((pattern, k % 2 == 0));
        }
        set
    }

    /// (b) The tempotron must learn a task it fails at initialisation, with a stated seed and an
    /// asserted accuracy, not a plot.
    #[test]
    fn the_tempotron_learns_a_task_it_fails_at_initialisation() {
        let mut rng = Rng::new(20_260_918);
        let n_in = 10;
        let set = tempotron_set(&mut rng, n_in, 12, 0.18);
        let mut t = Tempotron::gutig_sompolinsky_2006(n_in, 5e-3).expect("valid");
        for w in &mut t.w {
            *w = 0.02 + 0.06 * rng.next_f64();
        }
        let (dt, t_end) = (1e-3, 0.25);

        let before = t.accuracy(&set, dt, t_end).expect("non-empty");
        assert!(before <= 0.6, "the neuron already scored {before} before training");
        let hist = t.train(&set, 400, dt, t_end).expect("non-empty");
        let after = t.accuracy(&set, dt, t_end).expect("non-empty");
        assert_eq!(after, 1.0, "accuracy reached only {after} (from {before})");
        assert!(hist[0] > 0.0, "nothing was wrong on the first pass, so nothing was learned");
        assert_eq!(*hist.last().expect("400 epochs"), 0.0, "the last pass still made corrections");
    }

    /// A correct decision must change nothing, exactly. The tempotron is error-driven, and a rule
    /// that applied a tiny update on correct trials would drift away from a solution it had found.
    #[test]
    fn a_correct_tempotron_decision_moves_no_weight() {
        let n_in = 6;
        let mut rng = Rng::new(5);
        let set = tempotron_set(&mut rng, n_in, 2, 0.1);
        let mut t = Tempotron::gutig_sompolinsky_2006(n_in, 1e-2).expect("valid");
        t.w.fill(0.05);
        // Small positive weights, NOT zero: at exactly zero the membrane is flat, the peak lands on
        // the first grid point, every postsynaptic potential there is zero and the update is zero
        // too — the tempotron cannot bootstrap from a silent membrane, which is why
        // `the_tempotron_learns_a_task_it_fails_at_initialisation` draws its weights at random.
        // Here the membrane still never reaches threshold, so the "stay silent" pattern is already
        // decided correctly and the "fire" one is not.
        let silent = &set[1];
        assert!(!silent.1);
        let before = t.w.clone();
        assert!(!t.train_once(&silent.0, silent.1, 1e-3, 0.2).expect("valid"));
        // Bit equality, which is the strongest statement available — and its limit is stated: a
        // stray addition smaller than the spacing of `f64` near 0.05, which is 6.9e-18, is not
        // representable and no test can see it. Anything a rule would plausibly add is far above
        // that.
        assert_eq!(t.w, before, "a correct decision moved the weights");
        let ulp = f64::from_bits(before[0].to_bits() + 1) - before[0];
        assert!(ulp > 6e-18 && ulp < 8e-18, "the spacing near 0.05 is {ulp}, not the 6.9e-18 above");
        let firing = &set[0];
        assert!(firing.1);
        assert!(t.train_once(&firing.0, firing.1, 1e-3, 0.2).expect("valid"));
        assert_ne!(t.w, before, "a wrong decision left the weights alone");
        assert!(t.w.iter().all(|w| *w > 0.0), "the update did not potentiate");
    }

    #[test]
    fn the_tempotron_refuses_a_degenerate_kernel() {
        assert!(matches!(
            Tempotron::new(4, 15e-3, 15e-3, 1.0, 1e-3),
            Err(LearnError::TimeConstantsEqual { .. })
        ));
        assert!(matches!(
            Tempotron::new(4, 3e-3, 15e-3, 1.0, 1e-3),
            Err(LearnError::NotPositive { what: "tau - tau_s", .. })
        ));
        assert!(matches!(
            Tempotron::new(0, 15e-3, 3e-3, 1.0, 1e-3),
            Err(LearnError::Empty { what: "afferents" })
        ));
        let t = Tempotron::gutig_sompolinsky_2006(3, 1e-3).expect("valid");
        assert!(matches!(
            t.voltage(&[vec![0.0]], 0.0),
            Err(LearnError::ShapeMismatch { what: "pattern", got: 1, want: 3 })
        ));
        assert!(matches!(
            t.peak(&vec![vec![0.0]; 3], 0.0, 0.1),
            Err(LearnError::NotPositive { what: "dt", .. })
        ));
    }

    // -----------------------------------------------------------------------------------------
    // ReSuMe
    // -----------------------------------------------------------------------------------------

    /// (e) The reduction, as an equality. With `a = 0` and `eta = 1`, the teacher half of `ReSuMe`
    /// evaluates the same expression as `PairStdp::window` and the output half evaluates its exact
    /// negative — to the last bit, over a sweep of lags, because both are `A exp(-lag / tau)`
    /// computed in the same order.
    #[test]
    fn resume_reduces_to_the_stdp_window() {
        let a_plus = 0.0125;
        let tau_plus = 16.8e-3;
        let stdp = PairStdp::new(
            a_plus,
            0.0132,
            tau_plus,
            33.7e-3,
            WeightRule::Additive,
            Bounds::new(0.0, 1.0).expect("ordered"),
        )
        .expect("valid");
        let r = ReSuMe::new(0.0, a_plus, tau_plus, 1.0).expect("valid");

        let t_pre = 0.100;
        for k in 1..=200 {
            // The lag is read back from the two times rather than used to build them: `0.1 + 0.002`
            // minus `0.1` is not `0.002` in binary, and an equality test that ignored that would be
            // measuring its own arithmetic instead of the two rules'.
            let t_post = t_pre + f64::from(k) * 5e-4;
            let lag = t_post - t_pre;
            let teacher = r.delta_w(&[t_pre], &[t_post], &[]).expect("finite");
            let output = r.delta_w(&[t_pre], &[], &[t_post]).expect("finite");
            assert_eq!(teacher, stdp.window(lag), "teacher branch at lag {lag}");
            assert_eq!(output, -stdp.window(lag), "output branch at lag {lag}");
            assert!(teacher > 0.0, "the window collapsed to zero at lag {lag}");
        }
        // The window is exactly zero at and before zero lag, matching PairStdp's convention at
        // the boundary. Pinned because `s > 0.0` reads identically to `s >= 0.0` everywhere else
        // in this module — `correlation` only ever pairs strictly ordered spikes — so this is the
        // only place the difference is visible.
        assert_eq!(r.window(0.0), 0.0);
        assert_eq!(r.window(-1e-3), 0.0);
        assert_eq!(stdp.window(0.0), 0.0);
        // Post before pre contributes nothing on either side: ReSuMe's window is causal, where
        // PairStdp has a depression branch there. That is a real difference between the two rules
        // and it is pinned so that nobody "fixes" it into agreement.
        assert_eq!(r.delta_w(&[0.2], &[0.1], &[]).expect("finite"), 0.0);
        assert!(stdp.window(-0.1) < 0.0);
    }

    /// The fixed point, exactly. When the neuron already produces the desired train the rule must
    /// return `0.0` — not a small residue — for any presynaptic history, including one where the
    /// individual terms differ by fourteen orders of magnitude and a signed single-accumulator
    /// implementation would leave a remainder.
    #[test]
    fn resume_has_an_exactly_zero_fixed_point() {
        let r = ReSuMe::new(3e-4, 0.01, 10e-3, 0.5).expect("valid");
        let pre = [0.0, 1e-9, 0.02, 0.041, 0.0413, 0.2];
        let train = [0.0205, 0.0415, 0.25];
        assert_eq!(r.delta_w(&pre, &train, &train).expect("finite"), 0.0);
        // And it is not zero because the rule is dead.
        assert!(r.delta_w(&pre, &train, &[]).expect("finite") > 0.0);
        assert!(r.delta_w(&pre, &[], &train).expect("finite") < 0.0);
    }

    /// With the window amplitude zero, what is left is the spike-count term, exactly.
    #[test]
    fn the_non_hebbian_term_counts_spikes() {
        let r = ReSuMe::new(0.002, 0.0, 10e-3, 2.0).expect("valid");
        let pre = [0.0, 0.01, 0.02];
        let d = r.delta_w(&pre, &[0.03, 0.04, 0.05], &[0.031]).expect("finite");
        assert_eq!(d, 2.0 * (0.002 * 2.0));
    }

    #[test]
    fn resume_refuses_a_non_finite_spike_time() {
        let r = ReSuMe::new(0.0, 0.01, 10e-3, 1.0).expect("valid");
        assert!(matches!(
            r.delta_w(&[0.0, f64::NAN], &[0.01], &[]),
            Err(LearnError::NonFiniteInput { index: 1, .. })
        ));
        assert!(matches!(
            ReSuMe::new(0.0, -1.0, 10e-3, 1.0),
            Err(LearnError::NotPositive { what: "amp", .. })
        ));
    }

    // -----------------------------------------------------------------------------------------
    // SpikeProp
    // -----------------------------------------------------------------------------------------

    #[test]
    fn the_alpha_psp_peaks_at_exactly_one_at_its_time_constant() {
        let p = AlphaPsp::new(7e-3).expect("positive");
        assert_eq!(p.eval(p.tau), 1.0);
        assert_eq!(p.eval(0.0), 0.0);
        assert_eq!(p.eval(-1.0), 0.0);
        assert_eq!(p.slope(p.tau), 0.0);
        for k in 1..5_000 {
            let s = k as f64 * 2e-5;
            assert!(p.eval(s) <= 1.0, "eval({s}) = {}", p.eval(s));
        }
        // The slope is the derivative of eval, checked against a central difference away from the
        // kink at zero.
        for &s in &[1e-3, 5e-3, 7e-3, 2e-2, 5e-2] {
            let h = 1e-8;
            let fd = (p.eval(s + h) - p.eval(s - h)) / (2.0 * h);
            assert!((fd - p.slope(s)).abs() < 1e-5, "at {s}: fd {fd} vs {}", p.slope(s));
        }
    }

    /// Weights and arrival times chosen so the crossing lands at 5.3 ms, **after every afferent
    /// has arrived**. That matters: a neuron that fires before an afferent's spike has a sensitivity
    /// of exactly zero for that synapse, the finite difference is exactly zero too, and the two
    /// agreeing on zero would be a comparison that proves nothing.
    fn spikeprop_fixture() -> (SpikeProp, Vec<f64>) {
        let w = vec![0.30, 0.28, 0.26, 0.24, 0.22];
        let pre = vec![0.000, 0.001, 0.002, 0.003, 0.004];
        (SpikeProp::new(w, 7e-3, 1.0, 1e4).expect("valid"), pre)
    }

    /// The implicit-function derivative against a central finite difference of the actual threshold
    /// crossing. The formula is checked against the simulation, not against itself.
    #[test]
    fn the_spikeprop_sensitivity_matches_a_finite_difference() {
        let (net, pre) = spikeprop_fixture();
        let (dt, t_end) = (1e-5, 0.08);
        let t_out = net.first_spike(&pre, dt, t_end).expect("valid").expect("it fires");
        assert!(t_out > 0.0 && t_out < t_end);
        let last_in = *pre.last().expect("non-empty");
        assert!(t_out > last_in, "the neuron fired at {t_out}, before the input at {last_in}");
        let g = net.sensitivity(&pre, t_out).expect("valid").expect("transversal crossing");
        for i in 0..net.w.len() {
            let h = 1e-6 * net.w[i];
            let mut up = net.clone();
            up.w[i] += h;
            let mut dn = net.clone();
            dn.w[i] -= h;
            let a = up.first_spike(&pre, dt, t_end).expect("valid").expect("fires");
            let b = dn.first_spike(&pre, dt, t_end).expect("valid").expect("fires");
            let fd = (a - b) / (2.0 * h);
            let rel = (fd - g[i]).abs() / fd.abs();
            assert!(rel < 1e-5, "synapse {i}: finite difference {fd} vs analytic {}", g[i]);
            assert!(g[i] < 0.0, "synapse {i} has a non-negative sensitivity {}", g[i]);
        }
    }

    /// The rule moves the spike where it is told to, by a lot, and the residual error is small in
    /// absolute terms rather than merely smaller than it was.
    #[test]
    fn spikeprop_moves_the_output_spike_to_its_target() {
        let (mut net, pre) = spikeprop_fixture();
        let (dt, t_end) = (1e-5, 0.08);
        let start = net.first_spike(&pre, dt, t_end).expect("valid").expect("fires");
        let target = start - 2e-3;
        let mut first = None;
        let mut last = 0.0;
        for _ in 0..200 {
            let e = net.train_once(&pre, target, dt, t_end).expect("valid").expect("fires");
            if first.is_none() {
                first = Some(e.abs());
            }
            last = e.abs();
        }
        let first = first.expect("at least one step");
        assert!(first > 1e-3, "the task was already solved at the start: error {first}");
        // Measured 4.1e-11 after 200 steps, from 2.0e-3 — seven orders, so the thresholds are set
        // two orders short of what the run achieves.
        assert!(last < first / 1e4, "error fell only from {first} to {last}");
        assert!(last < 1e-8, "final timing error {last} s");
        // The spike really moved, rather than the target having been where it already was.
        let moved = net.first_spike(&pre, dt, t_end).expect("valid").expect("fires");
        assert!((moved - target).abs() < 1e-8, "the spike ended at {moved}, not {target}");
        assert!((moved - start).abs() > 1.9e-3, "the spike barely moved: {start} -> {moved}");
    }

    /// The failure mode `SpikeProp` is known for: the membrane stops reaching the threshold and the
    /// spike time is no longer a function of the weights at all. `None`, not a large number.
    #[test]
    fn spikeprop_refuses_when_the_neuron_does_not_fire() {
        let (mut net, pre) = spikeprop_fixture();
        for w in &mut net.w {
            *w *= 0.1;
        }
        assert_eq!(net.first_spike(&pre, 1e-5, 0.08).expect("valid"), None);
        // And at a crossing that is not transversal — here forced by making the total slope
        // non-positive — the sensitivity refuses rather than dividing by something tiny.
        let (net2, pre2) = spikeprop_fixture();
        let late = 0.5; // long past every alpha function's peak, where every slope is negative
        assert_eq!(net2.sensitivity(&pre2, late).expect("valid"), None);
        assert!(matches!(
            net2.potential(&[0.0], 0.0),
            Err(LearnError::ShapeMismatch { what: "presynaptic times", .. })
        ));
        assert!(matches!(SpikeProp::new(vec![], 1e-3, 1.0, 1e-3), Err(LearnError::Empty { .. })));
    }

    // -----------------------------------------------------------------------------------------
    // FORCE
    // -----------------------------------------------------------------------------------------

    /// The exact identity `FORCE` rests on, against a wholly different computation: after `k`
    /// recursive updates, `P` is the inverse of the ridge-regularised correlation matrix and `w` is
    /// the batch ridge solution. Checked through a Cholesky factorisation of the batch problem,
    /// which shares no code with the recursion.
    #[test]
    fn force_matches_the_closed_form_ridge_solution() {
        let n = 6;
        let ridge = 0.5;
        let samples = 40;
        let mut rng = Rng::new(31);
        let mut f = Force::new(n, ridge).expect("valid");
        let mut a = vec![0.0; n * n];
        for i in 0..n {
            a[i * n + i] = ridge;
        }
        let mut b = vec![0.0; n];
        for _ in 0..samples {
            let r: Vec<f64> = (0..n).map(|_| 2.0 * rng.next_f64() - 1.0).collect();
            let d = 2.0 * rng.next_f64() - 1.0;
            f.update(&r, d).expect("well conditioned");
            for i in 0..n {
                for j in 0..n {
                    a[i * n + j] += r[i] * r[j];
                }
                b[i] += d * r[i];
            }
        }
        assert_eq!(f.updates(), samples);
        let chol = cholesky(&a, n, 1e-14).expect("positive definite");

        // Every column of A^-1 against the corresponding column of P.
        let mut scale = 0.0f64;
        let mut worst = 0.0f64;
        for j in 0..n {
            let mut e = vec![0.0; n];
            e[j] = 1.0;
            let col = chol.solve(&e).expect("valid");
            for i in 0..n {
                scale = scale.max(col[i].abs());
                worst = worst.max((col[i] - f.inverse_correlation()[i * n + j]).abs());
            }
        }
        assert!(scale > 1e-3, "the inverse was numerically zero, so this compared nothing");
        assert!(worst / scale < 1e-9, "P differed from A^-1 by {worst} against scale {scale}");

        let want = chol.solve(&b).expect("valid");
        let mut w_scale = 0.0f64;
        let mut w_worst = 0.0f64;
        for i in 0..n {
            w_scale = w_scale.max(want[i].abs());
            w_worst = w_worst.max((want[i] - f.w[i]).abs());
        }
        assert!(w_scale > 1e-3, "the ridge solution was zero, so this compared nothing");
        assert!(w_worst / w_scale < 1e-9, "w differed from the batch solution by {w_worst}");
    }

    /// `FORCE` on filtered spike trains, the Nicola & Clopath setting: the basis is a bank of
    /// exponentially filtered periodic spike sources and the target is a linear combination of them,
    /// so the generating weights are known exactly and can be recovered rather than merely fitted.
    #[test]
    fn force_recovers_the_weights_that_generated_its_target() {
        let n = 12;
        let dt = 1e-3;
        let mut rng = Rng::new(77);
        let periods: Vec<f64> = (0..n).map(|_| 0.03 + 0.09 * rng.next_f64()).collect();
        let phases: Vec<f64> = (0..n).map(|k| periods[k] * rng.next_f64()).collect();
        let taus: Vec<f64> = (0..n).map(|_| 0.01 + 0.03 * rng.next_f64()).collect();
        let w_true: Vec<f64> = (0..n).map(|_| 2.0 * rng.next_f64() - 1.0).collect();

        let mut traces: Vec<Eligibility> =
            taus.iter().map(|&t| Eligibility::new(t, dt).expect("positive")).collect();
        let mut f = Force::new(n, 1e-9).expect("valid");
        let mut first_err = None;
        let mut late_err = 0.0f64;
        let mut late_rms = 0.0f64;
        let steps = 4_000;
        for s in 0..steps {
            let t = s as f64 * dt;
            let mut r = vec![0.0; n];
            for k in 0..n {
                // Periodic source k fires whenever the phase wraps within this step.
                let prev = ((t - dt - phases[k]) / periods[k]).floor();
                let now = ((t - phases[k]) / periods[k]).floor();
                let input = if t >= phases[k] && now > prev { 1.0 } else { 0.0 };
                r[k] = traces[k].step(input);
            }
            let mut d = 0.0;
            for k in 0..n {
                d += w_true[k] * r[k];
            }
            let e = f.update(&r, d).expect("well conditioned");
            if s == 200 {
                first_err = Some(e.abs());
            }
            if s >= steps - 400 {
                late_err = late_err.max(e.abs());
                late_rms += d * d;
            }
        }
        let first = first_err.expect("reached step 200");
        late_rms = (late_rms / 400.0).sqrt();
        assert!(late_rms > 0.1, "the target was nearly zero, so tracking it means nothing");
        assert!(first > 0.0, "the readout was already exact at step 200");
        // Measured 5.8e-12 and 2.6e-12; the thresholds sit two orders above them, and the ridge
        // is what sets the floor — at ridge = 1e-3 the same run leaves 5.8e-6 and 2.6e-6, which is
        // the regulariser's bias and not a failure to converge.
        assert!(late_err < 1e-9, "late error {late_err} against a target of rms {late_rms}");
        let mut worst = 0.0f64;
        for k in 0..n {
            worst = worst.max((f.w[k] - w_true[k]).abs());
        }
        assert!(worst < 1e-9, "weights differ from the generators by {worst}");
        // `output` must be the same dot product the update measured its error against, or the
        // readout a caller reads back is not the readout that was trained.
        let probe: Vec<f64> = (0..n).map(|k| 0.1 + 0.01 * k as f64).collect();
        let want: f64 = (0..n).map(|k| w_true[k] * probe[k]).sum();
        let got = f.output(&probe).expect("right shape");
        assert!((got - want).abs() < 1e-8, "output {got} vs the generators' {want}");
    }

    #[test]
    fn force_refuses_a_malformed_call() {
        let mut f = Force::new(3, 1.0).expect("valid");
        assert!(matches!(
            f.update(&[1.0, 2.0], 0.0),
            Err(LearnError::ShapeMismatch { what: "basis vector", got: 2, want: 3 })
        ));
        assert!(matches!(
            f.update(&[1.0, f64::NAN, 0.0], 0.0),
            Err(LearnError::NonFiniteInput { index: 1, .. })
        ));
        assert!(matches!(
            f.update(&[1.0, 0.0, 0.0], f64::INFINITY),
            Err(LearnError::NonFiniteParam { what: "target", .. })
        ));
        assert!(matches!(Force::new(0, 1.0), Err(LearnError::Empty { what: "basis" })));
        assert!(matches!(
            Force::new(3, 0.0),
            Err(LearnError::NotPositive { what: "ridge", .. })
        ));
        assert!(matches!(
            f.output(&[1.0]),
            Err(LearnError::ShapeMismatch { what: "basis vector", .. })
        ));
    }

    /// Errors must print what went wrong, and the wrapped surrogate error must still reach a caller
    /// walking the `source` chain.
    #[test]
    fn errors_say_what_was_wrong() {
        let e = LearnError::Singular { step: 7, value: -0.5 };
        assert!(format!("{e}").contains("update 7"));
        let mut layer = spec(true, 4).build().expect("valid");
        let sur = ArcTan::default();
        // A bias at the top of the range: one step of `alpha * I + drive` at alpha = 0.819 already
        // leaves the finite numbers, so the sweep stops at the first step rather than reporting a
        // NaN loss twenty steps later.
        for j in 0..layer.n_rec {
            let k = layer.idx_b(j);
            layer.p[k] = 1e308;
        }
        let x = xor_pattern();
        let err = logits_streaming(&layer, &sur, &x, SpikeFn::Heaviside).unwrap_err();
        assert!(matches!(err, LearnError::Surrogate(_)), "{err}");
        assert!(std::error::Error::source(&err).is_some());
        assert!(format!("{err}").contains("non-finite"));
    }
}


