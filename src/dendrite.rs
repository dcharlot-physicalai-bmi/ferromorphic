//! Dendrites: a neuron with more than one compartment, and what the second compartment buys —
//! a learning rule that needs no error signal from anywhere else, and a single cell that computes
//! what a point neuron provably cannot.
//!
//! # Two compartments, and a prediction
//!
//! Give a neuron a dendrite with its own potential `V_d`, coupled to the soma by a conductance
//! `g_D`. The soma also receives what it receives from elsewhere — a "teacher", in Urbanczik and
//! Senn's phrase (*Learning by the dendritic prediction of somatic spiking*, Neuron 81(3):521–528,
//! 2014). The somatic potential obeys
//!
//! ```text
//! C dV/dt = g_L (E_L − V) + g_D (V_d − V) + g_E (E_E − V) + g_I (E_I − V)
//! ```
//!
//! and settles, under constant conductances, at the conductance-weighted mean of the reversal
//! potentials — [`TwoCompartment::steady_state`], exact. The dendrite's *prediction* of the soma
//! is what the soma would do with the teacher off:
//! `V*_w = (g_L E_L + g_D V_d) / (g_L + g_D)`. The learning rule changes the dendritic weights by
//! the mismatch between the somatic rate and the rate the dendrite predicts,
//!
//! ```text
//! dw_i/dt = η · (φ(V) − φ(V*_w)) · φ'(V*_w) · PSP_i
//! ```
//!
//! so that when the dendrite has learned, removing the teacher changes nothing: the soma fires at
//! the rate the dendrite alone drives it to. That is supervised learning with the supervision
//! delivered as a conductance and the error computed inside the cell, which is why it is the
//! learning rule the dendritic-cortical-microcircuit line of work (Sacramento, Costa, Bengio and
//! Senn, `NeurIPS` 2018) builds on. [`DendriticLearner`] is that rule in its rate form, and the
//! test drives a dendrite to a teacher-imposed potential and then takes the teacher away.
//!
//! # Two layers in one cell
//!
//! Poirazi, Brannon and Mel (*Pyramidal neuron as two-layer neural network*, Neuron
//! 37(6):989–999, 2003) showed a pyramidal cell's dendritic branches each apply their own
//! sigmoidal nonlinearity before the soma sums them — a two-layer network in one neuron. The
//! consequence is exact: a point neuron is a linear threshold unit and **cannot** compute XOR, a
//! fact [`point_neuron_can_xor`] checks by exhausting the sign patterns rather than citing
//! Minsky and Papert; a [`BranchedNeuron`] with two branches can, with weights written out in the
//! test.
//!
//! # Why it is in a neuromorphic crate
//!
//! Multi-compartment neurons are what the second generation of neuromorphic chips added — Loihi 2
//! exposes them, and the field's argument for them is exactly the two results above: a local
//! learning rule and per-branch nonlinearity, each bought with one more state variable per cell,
//! which [`crate::ledger`] counts as a membrane update. This module states what the extra
//! compartment computes so that the count has something to be weighed against.
//!
//! # What this module has NOT reproduced
//!
//! - The spiking form of the learning rule, with Poisson somatic spikes standing in for `φ(V)`.
//!   The rate form here is its expectation; the spiking form converges to the same point with
//!   noise, and that claim is not tested here.
//! - Any cable equation. The dendrite is one compartment; a real branch is a hundred.
//! - The microcircuit that turns the dendritic error into backpropagation. Named, not built.

use core::fmt;

/// What went wrong, named rather than guessed around.
#[derive(Debug, Clone, PartialEq)]
pub enum DendriteError {
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
    /// An array of the wrong length.
    Dimension {
        /// Which object.
        what: &'static str,
        /// Length supplied.
        got: usize,
        /// Length required.
        want: usize,
    },
    /// A count of zero where at least one is needed.
    Empty {
        /// What was empty.
        what: &'static str,
    },
}

impl fmt::Display for DendriteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfRange { what, value, low, high } => {
                write!(f, "{what} = {value} is outside [{low}, {high}]")
            }
            Self::NonFinite { what, index } => write!(f, "{what} is not finite at {index}"),
            Self::Dimension { what, got, want } => write!(f, "{what} has {got} entries, needs {want}"),
            Self::Empty { what } => write!(f, "{what} is empty"),
        }
    }
}

impl std::error::Error for DendriteError {}

fn finite_scalar(what: &'static str, v: f64) -> Result<f64, DendriteError> {
    if v.is_finite() { Ok(v) } else { Err(DendriteError::NonFinite { what, index: 0 }) }
}

fn non_negative(what: &'static str, v: f64) -> Result<f64, DendriteError> {
    let v = finite_scalar(what, v)?;
    if v >= 0.0 {
        Ok(v)
    } else {
        Err(DendriteError::OutOfRange { what, value: v, low: 0.0, high: f64::INFINITY })
    }
}

fn positive(what: &'static str, v: f64) -> Result<f64, DendriteError> {
    let v = finite_scalar(what, v)?;
    if v > 0.0 {
        Ok(v)
    } else {
        Err(DendriteError::OutOfRange { what, value: v, low: f64::MIN_POSITIVE, high: f64::INFINITY })
    }
}

fn finite_slice(what: &'static str, v: &[f64], want: usize) -> Result<(), DendriteError> {
    if v.len() != want {
        return Err(DendriteError::Dimension { what, got: v.len(), want });
    }
    if let Some(i) = v.iter().position(|x| !x.is_finite()) {
        return Err(DendriteError::NonFinite { what, index: i });
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Two compartments
// ---------------------------------------------------------------------------------------------

/// A soma with a leak, a dendritic coupling, and excitatory and inhibitory teacher conductances.
/// Potentials in volts, conductances in siemens, capacitance in farads.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TwoCompartment {
    /// Somatic capacitance, farads.
    pub c: f64,
    /// Leak conductance, siemens.
    pub g_l: f64,
    /// Leak reversal, volts.
    pub e_l: f64,
    /// Dendrite-to-soma coupling conductance, siemens.
    pub g_d: f64,
    /// Excitatory reversal, volts.
    pub e_e: f64,
    /// Inhibitory reversal, volts.
    pub e_i: f64,
    /// Somatic potential, volts.
    pub v: f64,
}

impl TwoCompartment {
    /// Urbanczik and Senn's illustrative constants: `C = 1 nF`, `g_L = 0.1 µS`, `E_L = −70 mV`,
    /// `g_D = 2 µS`... no. Those are not quoted from the paper; this crate's default is a round
    /// set stated here: `C = 1 nF`, `g_L = 50 nS`, `E_L = −70 mV`, `g_D = 50 nS`, `E_E = 0`,
    /// `E_I = −75 mV`, starting at rest.
    #[must_use]
    pub fn round_defaults() -> Self {
        Self { c: 1e-9, g_l: 50e-9, e_l: -70e-3, g_d: 50e-9, e_e: 0.0, e_i: -75e-3, v: -70e-3 }
    }

    /// Check every constant.
    ///
    /// # Errors
    ///
    /// [`DendriteError::OutOfRange`] for a non-positive capacitance or leak, a negative coupling,
    /// or `E_I ≥ E_E`; [`DendriteError::NonFinite`] for a non-finite value.
    pub fn validate(&self) -> Result<(), DendriteError> {
        positive("c", self.c)?;
        positive("g_l", self.g_l)?;
        non_negative("g_d", self.g_d)?;
        finite_scalar("e_l", self.e_l)?;
        finite_scalar("e_e", self.e_e)?;
        finite_scalar("e_i", self.e_i)?;
        finite_scalar("v", self.v)?;
        if self.e_i >= self.e_e {
            return Err(DendriteError::OutOfRange { what: "e_i (must be below e_e)", value: self.e_i, low: f64::NEG_INFINITY, high: self.e_e });
        }
        Ok(())
    }

    /// The potential the soma settles at under constant `v_d`, `g_e`, `g_i`:
    /// `(g_L E_L + g_D V_d + g_E E_E + g_I E_I) / (g_L + g_D + g_E + g_I)`.
    ///
    /// # Errors
    ///
    /// [`DendriteError::OutOfRange`] for a negative conductance, [`DendriteError::NonFinite`].
    pub fn steady_state(&self, v_d: f64, g_e: f64, g_i: f64) -> Result<f64, DendriteError> {
        let v_d = finite_scalar("v_d", v_d)?;
        let g_e = non_negative("g_e", g_e)?;
        let g_i = non_negative("g_i", g_i)?;
        let total = self.g_l + self.g_d + g_e + g_i;
        Ok((self.g_l * self.e_l + self.g_d * v_d + g_e * self.e_e + g_i * self.e_i) / total)
    }

    /// The dendritic prediction `V*_w = (g_L E_L + g_D V_d) / (g_L + g_D)`: where the soma would
    /// settle with the teacher off.
    #[must_use]
    pub fn prediction(&self, v_d: f64) -> f64 {
        (self.g_l * self.e_l + self.g_d * v_d) / (self.g_l + self.g_d)
    }

    /// The effective time constant `C / (g_L + g_D + g_E + g_I)`, seconds.
    #[must_use]
    pub fn time_constant(&self, g_e: f64, g_i: f64) -> f64 {
        self.c / (self.g_l + self.g_d + g_e + g_i)
    }

    /// Advance the soma by `dt` under constant `v_d`, `g_e`, `g_i`, by exponential Euler — exact
    /// for constant inputs.
    ///
    /// # Errors
    ///
    /// As [`TwoCompartment::steady_state`], plus [`DendriteError::OutOfRange`] for a non-positive
    /// `dt`.
    pub fn step(&mut self, dt: f64, v_d: f64, g_e: f64, g_i: f64) -> Result<f64, DendriteError> {
        let dt = positive("dt", dt)?;
        let v_inf = self.steady_state(v_d, g_e, g_i)?;
        let tau = self.time_constant(g_e, g_i);
        self.v = v_inf + (self.v - v_inf) * (-dt / tau).exp();
        Ok(self.v)
    }
}

// ---------------------------------------------------------------------------------------------
// The dendritic prediction learning rule
// ---------------------------------------------------------------------------------------------

/// The somatic rate function `φ(V)`: a logistic between `rate_max` and zero, centred at `v_half`
/// with slope `1 / v_scale`, hertz.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RateFunction {
    /// Maximum rate, hertz.
    pub rate_max: f64,
    /// Potential at half the maximum rate, volts.
    pub v_half: f64,
    /// Width of the transition, volts.
    pub v_scale: f64,
}

impl RateFunction {
    /// Build.
    ///
    /// # Errors
    ///
    /// [`DendriteError::OutOfRange`] for a non-positive rate or scale, [`DendriteError::NonFinite`].
    pub fn new(rate_max: f64, v_half: f64, v_scale: f64) -> Result<Self, DendriteError> {
        positive("rate_max", rate_max)?;
        finite_scalar("v_half", v_half)?;
        positive("v_scale", v_scale)?;
        Ok(Self { rate_max, v_half, v_scale })
    }

    /// `φ(V) = rate_max / (1 + exp(−(V − v_half)/v_scale))`.
    #[must_use]
    pub fn rate(&self, v: f64) -> f64 {
        self.rate_max / (1.0 + (-(v - self.v_half) / self.v_scale).exp())
    }

    /// `φ'(V) = φ(V) (1 − φ(V)/rate_max) / v_scale`, hertz per volt.
    #[must_use]
    pub fn slope(&self, v: f64) -> f64 {
        let r = self.rate(v);
        r * (1.0 - r / self.rate_max) / self.v_scale
    }
}

/// A dendrite whose potential is the resting potential plus a weighted sum of postsynaptic
/// potentials, learning by the dendritic prediction of somatic firing.
///
/// `V_d = E_L + Σ_i w_i · psp_i` with `psp_i` the (already filtered) presynaptic activity in volts
/// of depolarisation per unit weight; the weights are dimensionless.
///
/// # The fixed point, in closed form
///
/// With a teacher of total conductance `g_T` holding a target `V_T = (g_E E_E + g_I E_I)/g_T`, the
/// soma sits at the conductance-weighted mean of the prediction and the target, so the prediction
/// error vanishes only when `V*_w = V_T` — the dendrite has to reproduce the teacher's target
/// itself, at which point the teacher's nudge changes nothing. That needs
/// `V_d = V_T + (g_L / g_D)(V_T − E_L)`, and for `g_L = g_D` that is `2 V_T − E_L`. An excitatory
/// teacher alone has `V_T = E_E`, which no rate function resolves; the test uses the balanced pair
/// the paper does.
#[derive(Debug, Clone, PartialEq)]
pub struct DendriticLearner {
    /// The soma.
    pub soma: TwoCompartment,
    /// The rate function.
    pub phi: RateFunction,
    /// Dendritic weights, one per input.
    pub w: Vec<f64>,
    /// Learning rate, per volt-second of postsynaptic-potential-weighted rate error.
    pub eta: f64,
}

impl DendriticLearner {
    /// Build with `w` initial weights.
    ///
    /// # Errors
    ///
    /// [`DendriteError::Empty`] for no inputs, [`DendriteError::OutOfRange`] for a non-positive
    /// `eta`, plus [`TwoCompartment::validate`]'s refusals and [`DendriteError::NonFinite`] for a
    /// non-finite weight.
    pub fn new(soma: TwoCompartment, phi: RateFunction, w: Vec<f64>, eta: f64) -> Result<Self, DendriteError> {
        soma.validate()?;
        if w.is_empty() {
            return Err(DendriteError::Empty { what: "inputs" });
        }
        finite_slice("weights", &w, w.len())?;
        positive("eta", eta)?;
        Ok(Self { soma, phi, w, eta })
    }

    /// The dendritic potential `E_L + Σ_i w_i psp_i`, volts.
    ///
    /// # Errors
    ///
    /// [`DendriteError::Dimension`], [`DendriteError::NonFinite`].
    pub fn dendrite(&self, psp: &[f64]) -> Result<f64, DendriteError> {
        finite_slice("psp", psp, self.w.len())?;
        Ok(self.soma.e_l + self.w.iter().zip(psp).map(|(w, p)| w * p).sum::<f64>())
    }

    /// The dendritic potential at which the prediction equals a teacher target `v_t`:
    /// `V_T + (g_L / g_D)(V_T − E_L)`. `None` for a zero coupling.
    #[must_use]
    pub fn dendrite_for_target(&self, v_t: f64) -> Option<f64> {
        if self.soma.g_d == 0.0 {
            return None;
        }
        Some(v_t + (self.soma.g_l / self.soma.g_d) * (v_t - self.soma.e_l))
    }

    /// One step of `dt`: advance the soma under the dendrite and the teacher conductances, then
    /// move the weights by the rate form of the rule,
    /// `Δw_i = η · dt · (φ(V) − φ(V*_w)) · φ'(V*_w) · psp_i`.
    ///
    /// Returns `(V, V*_w)`: the somatic potential and the dendritic prediction.
    ///
    /// # Errors
    ///
    /// As [`DendriticLearner::dendrite`] and [`TwoCompartment::step`].
    pub fn step(&mut self, dt: f64, psp: &[f64], g_e: f64, g_i: f64) -> Result<(f64, f64), DendriteError> {
        let v_d = self.dendrite(psp)?;
        let v = self.soma.step(dt, v_d, g_e, g_i)?;
        let v_star = self.soma.prediction(v_d);
        let err = self.phi.rate(v) - self.phi.rate(v_star);
        let gain = self.eta * dt * err * self.phi.slope(v_star);
        for (w, p) in self.w.iter_mut().zip(psp) {
            *w += gain * p;
        }
        Ok((v, v_star))
    }

    /// The prediction error `φ(V) − φ(V*_w)` at the current state for the given inputs, hertz.
    ///
    /// # Errors
    ///
    /// As [`DendriticLearner::dendrite`].
    pub fn prediction_error(&self, psp: &[f64]) -> Result<f64, DendriteError> {
        let v_d = self.dendrite(psp)?;
        Ok(self.phi.rate(self.soma.v) - self.phi.rate(self.soma.prediction(v_d)))
    }
}

// ---------------------------------------------------------------------------------------------
// Branches
// ---------------------------------------------------------------------------------------------

/// A neuron whose inputs are grouped into branches, each summed and passed through its own
/// sigmoid before the soma sums the branches and thresholds.
///
/// `y = [ Σ_b s(Σ_{i ∈ b} w_bi x_i + θ_b) ≥ θ ]` with `s(z) = 1 / (1 + e^{−z/scale})`.
#[derive(Debug, Clone, PartialEq)]
pub struct BranchedNeuron {
    /// Per-branch weights over the shared input vector: `branches[b][i]`.
    pub branches: Vec<Vec<f64>>,
    /// Per-branch offset.
    pub branch_bias: Vec<f64>,
    /// Branch sigmoid scale. Strictly positive; small is sharp.
    pub scale: f64,
    /// Somatic threshold on the sum of branch outputs.
    pub threshold: f64,
}

impl BranchedNeuron {
    /// Build.
    ///
    /// # Errors
    ///
    /// [`DendriteError::Empty`] with no branches or no inputs, [`DendriteError::Dimension`] if the
    /// branches disagree on the input count or the biases do not match the branch count,
    /// [`DendriteError::OutOfRange`] for a non-positive scale, [`DendriteError::NonFinite`].
    pub fn new(branches: Vec<Vec<f64>>, branch_bias: Vec<f64>, scale: f64, threshold: f64) -> Result<Self, DendriteError> {
        if branches.is_empty() {
            return Err(DendriteError::Empty { what: "branches" });
        }
        let n = branches[0].len();
        if n == 0 {
            return Err(DendriteError::Empty { what: "inputs" });
        }
        for b in &branches {
            finite_slice("branch weights", b, n)?;
        }
        finite_slice("branch biases", &branch_bias, branches.len())?;
        positive("scale", scale)?;
        finite_scalar("threshold", threshold)?;
        Ok(Self { branches, branch_bias, scale, threshold })
    }

    /// The branch outputs, each in `(0, 1)`.
    ///
    /// # Errors
    ///
    /// [`DendriteError::Dimension`], [`DendriteError::NonFinite`].
    pub fn branch_outputs(&self, x: &[f64]) -> Result<Vec<f64>, DendriteError> {
        finite_slice("input", x, self.branches[0].len())?;
        Ok(self
            .branches
            .iter()
            .zip(&self.branch_bias)
            .map(|(w, b)| {
                let z: f64 = w.iter().zip(x).map(|(wi, xi)| wi * xi).sum::<f64>() + b;
                1.0 / (1.0 + (-z / self.scale).exp())
            })
            .collect())
    }

    /// Whether the soma fires: the branch outputs sum to at least the threshold.
    ///
    /// # Errors
    ///
    /// As [`BranchedNeuron::branch_outputs`].
    pub fn fires(&self, x: &[f64]) -> Result<bool, DendriteError> {
        Ok(self.branch_outputs(x)?.iter().sum::<f64>() >= self.threshold)
    }
}

/// The two bounds the XOR proof puts on `w_1 + w_2`, in units of `−b`: the lower bound that
/// adding the two middle constraints gives, and the strict upper bound the fourth one gives.
///
/// Returned rather than kept private because the VERDICT cannot see the lower bound's factor of
/// two. The verdict asks whether the half-open interval between the bounds is non-empty, and
/// `[2B, B)` and `[B, B)` are both empty for every `B > 0` — so a proof that dropped the factor
/// still answers "no", and an encoding whose comment promises that a change to the reasoning
/// changes the answer has to hand the reasoning out for that promise to be checkable. The
/// bounds themselves carry the factor; [`point_neuron_can_xor`] carries only its consequence.
#[must_use]
pub fn xor_sum_bounds() -> (f64, f64) {
    // The four constraints: b < 0, w1 + b ≥ 0, w2 + b ≥ 0, w1 + w2 + b < 0. Adding the middle two
    // gives w1 + w2 + 2b ≥ 0, so w1 + w2 ≥ −2b — TWICE the bound either middle constraint gives
    // on its own, which is the step of the proof this pair of numbers exists to record.
    let minus_b_lower = f64::MIN_POSITIVE; // −b > 0
    let sum_lower = 2.0 * minus_b_lower; // w1 + w2 ≥ −2b
    let sum_upper = minus_b_lower; // w1 + w2 < −b
    (sum_lower, sum_upper)
}

/// Whether ANY linear threshold unit `[w·x + b ≥ 0]` over two binary inputs computes XOR.
///
/// A linear threshold unit's output on the four corners of the square is determined by the signs
/// of `b`, `w_1 + b`, `w_2 + b` and `w_1 + w_2 + b`; XOR needs the first and last negative and
/// the middle two non-negative, i.e. `b < 0`, `w_1 ≥ −b`, `w_2 ≥ −b`, `w_1 + w_2 < −b` — and the
/// last contradicts the sum of the middle two, since `−b > 0`. The function returns that verdict
/// by checking the four inequalities for consistency the way the proof does, so it is a
/// statement about every unit and not about a grid of them. The bounds it reasons over are
/// [`xor_sum_bounds`], which is where the factor of two lives.
#[must_use]
pub fn point_neuron_can_xor() -> bool {
    // `w1 + w2` would have to lie in the half-open interval the two bounds cut out, and there is
    // no such number: the lower bound is twice the upper and both are strictly positive.
    let (sum_lower, sum_upper) = xor_sum_bounds();
    sum_lower < sum_upper
}

#[cfg(test)]
mod tests {
    use super::{
        BranchedNeuron, DendriteError, DendriticLearner, RateFunction, TwoCompartment,
        point_neuron_can_xor, xor_sum_bounds,
    };

    /// The steady state is the conductance-weighted mean of the reversals, exactly, and the
    /// stepped soma reaches it with the time constant `C / Σg`.
    #[test]
    fn the_soma_settles_at_the_conductance_weighted_mean() {
        let mut s = TwoCompartment::round_defaults();
        s.validate().unwrap();
        let (v_d, g_e, g_i) = (-40e-3, 20e-9, 10e-9);
        let want = (50e-9 * -70e-3 + 50e-9 * -40e-3 + 20e-9 * 0.0 + 10e-9 * -75e-3) / (50e-9 + 50e-9 + 20e-9 + 10e-9);
        assert!((s.steady_state(v_d, g_e, g_i).unwrap() - want).abs() < 1e-15);
        let tau = s.time_constant(g_e, g_i);
        assert!((tau - 1e-9 / 130e-9).abs() < 1e-15);
        // One time constant closes 1 − 1/e of the gap, exactly, in one step or in a thousand.
        let v0 = s.v;
        s.step(tau, v_d, g_e, g_i).unwrap();
        let after_one = s.v;
        assert!((after_one - (want + (v0 - want) * (-1.0f64).exp())).abs() < 1e-15);
        let mut fine = TwoCompartment::round_defaults();
        for _ in 0..1000 {
            fine.step(tau / 1000.0, v_d, g_e, g_i).unwrap();
        }
        assert!((fine.v - after_one).abs() < 1e-14, "exponential Euler composes");
        // The prediction is the steady state with the teacher off.
        assert!((s.prediction(v_d) - s.steady_state(v_d, 0.0, 0.0).unwrap()).abs() < 1e-18);
        assert!(matches!(s.steady_state(v_d, -1e-9, 0.0), Err(DendriteError::OutOfRange { what: "g_e", .. })));
        // With E_E = 0 the excitatory term vanishes from the arithmetic and dropping it survived
        // the first mutation sweep. At E_E = +10 mV it does not.
        let shifted = TwoCompartment { e_e: 10e-3, ..TwoCompartment::round_defaults() };
        let want = (50e-9 * -70e-3 + 50e-9 * v_d + 20e-9 * 10e-3 + 10e-9 * -75e-3) / 130e-9;
        assert!((shifted.steady_state(v_d, 20e-9, 10e-9).unwrap() - want).abs() < 1e-15);
        assert!(shifted.steady_state(v_d, 20e-9, 10e-9).unwrap() > s.steady_state(v_d, 20e-9, 10e-9).unwrap());
    }

    /// ⭐ The dendrite learns to predict the teacher, and then the teacher can go. Two inputs at
    /// fixed drives; a balanced teacher — 93.3 nS of excitation and 106.7 nS of inhibition, whose
    /// reversal-weighted mean is −40 mV — holds the soma; the rule moves the weights until the
    /// dendritic prediction is that −40 mV (the closed-form fixed point), which puts the dendrite
    /// at `2·V_T − E_L = −10 mV`; with the teacher removed the soma sits there on its own. Before
    /// learning the prediction error is over 40 Hz — the control that shows learning did it.
    #[test]
    fn the_dendrite_learns_the_teachers_potential_and_keeps_it_when_the_teacher_leaves() {
        let soma = TwoCompartment::round_defaults();
        let phi = RateFunction::new(100.0, -50e-3, 5e-3).unwrap();
        let psp = [8e-3, 3e-3]; // volts of depolarisation per unit weight, held constant
        // η = 1e-2 puts the rule's loop time constant, 1 / (η·φ'²·(g_D/(g_L+g_D))·Σpsp²), near
        // 0.6 s at the fixed point; ten seconds is sixteen of those.
        let mut cell = DendriticLearner::new(soma, phi, vec![0.5, 0.5], 1e-2).unwrap();
        let dt = 1e-4;
        // Teacher: V_T = (g_E·0 + g_I·(−75 mV)) / (g_E + g_I) = −40 mV needs g_I/(g_E+g_I) = 8/15.
        let g_t = 200e-9;
        let (g_e, g_i) = (g_t * 7.0 / 15.0, g_t * 8.0 / 15.0);
        let v_t = (g_e * soma.e_e + g_i * soma.e_i) / g_t;
        assert!((v_t + 40e-3).abs() < 1e-12);
        let mut learner_before = cell.clone();
        for _ in 0..(1.0 / dt) as usize {
            learner_before.soma.step(dt, learner_before.dendrite(&psp).unwrap(), g_e, g_i).unwrap();
        }
        let error_before = learner_before.prediction_error(&psp).unwrap().abs();
        assert!(error_before > 20.0, "before learning the prediction is already close: {error_before} Hz");

        for _ in 0..(10.0 / dt) as usize {
            cell.step(dt, &psp, g_e, g_i).unwrap();
        }
        let (v_taught, v_star) = cell.step(dt, &psp, g_e, g_i).unwrap();
        let error_after = cell.prediction_error(&psp).unwrap().abs();
        assert!(error_after < 0.5, "after learning the prediction error is {error_after} Hz");
        assert!((phi.rate(v_taught) - phi.rate(v_star)).abs() < 0.5);
        // The closed-form fixed point: the prediction is the teacher's target and the dendrite is
        // at 2·V_T − E_L.
        assert!((v_star - v_t).abs() < 0.3e-3, "prediction {v_star} against the target {v_t}");
        let v_d = cell.dendrite(&psp).unwrap();
        let want_d = cell.dendrite_for_target(v_t).unwrap();
        assert!((want_d + 10e-3).abs() < 1e-12);
        assert!((v_d - want_d).abs() < 0.6e-3, "dendrite {v_d} against 2V_T − E_L = {want_d}");

        // The teacher leaves. The soma settles where the dendrite alone puts it, which is where
        // the teacher had it — to the rate the rule matched.
        let taught_rate = phi.rate(v_taught);
        let w_learned = cell.w.clone();
        for _ in 0..(1.0 / dt) as usize {
            cell.step(dt, &psp, 0.0, 0.0).unwrap();
        }
        let alone_rate = phi.rate(cell.soma.v);
        assert!((alone_rate - taught_rate).abs() < 1.0, "alone {alone_rate} Hz against taught {taught_rate} Hz");
        // And with the teacher gone the rule is at its fixed point: the weights stop moving.
        let drift: f64 = cell.w.iter().zip(&w_learned).map(|(a, b)| (a - b).abs()).sum();
        assert!(drift < 1e-3 * w_learned.iter().map(|w| w.abs()).sum::<f64>(), "weights drifted by {drift} without a teacher");
        // The weights moved in the direction the drive says: both inputs were positive and the
        // teacher was above the initial prediction, so both weights grew, the stronger input
        // further, in the ratio of the drives (the rule's update is proportional to psp_i).
        assert!(cell.w[0] > 0.5 && cell.w[1] > 0.5, "{:?}", cell.w);
        let (d0, d1) = (cell.w[0] - 0.5, cell.w[1] - 0.5);
        assert!((d0 / d1 - 8.0 / 3.0).abs() < 1e-6, "weight changes {d0}:{d1} are not in the 8:3 ratio of the drives");
        assert_eq!(DendriticLearner { soma: TwoCompartment { g_d: 0.0, ..soma }, ..cell.clone() }.dendrite_for_target(v_t), None);
    }

    /// The rate function and its slope, against hand values.
    #[test]
    fn the_rate_function_and_its_slope_are_the_logistic_and_its_derivative() {
        let phi = RateFunction::new(80.0, -50e-3, 4e-3).unwrap();
        assert_eq!(phi.rate(-50e-3), 40.0);
        assert!((phi.rate(-46e-3) - 80.0 / (1.0 + (-1.0f64).exp())).abs() < 1e-12);
        let h = 1e-7;
        let numeric = (phi.rate(-48e-3 + h) - phi.rate(-48e-3 - h)) / (2.0 * h);
        assert!((phi.slope(-48e-3) - numeric).abs() < 1e-3 * numeric, "{} vs {numeric}", phi.slope(-48e-3));
        assert!(phi.rate(-100e-3) < 1e-3 && phi.rate(0.0) > 79.99);
    }

    /// A point neuron cannot compute XOR — checked by the proof's own arithmetic — and a two-branch
    /// neuron can, with the weights written out here: branch 1 fires for x1 AND NOT x2, branch 2
    /// for x2 AND NOT x1, and the soma fires if either branch does.
    #[test]
    fn two_branches_compute_the_xor_a_point_neuron_cannot() {
        assert!(!point_neuron_can_xor());
        let cell = BranchedNeuron::new(
            vec![vec![1.0, -1.0], vec![-1.0, 1.0]],
            vec![-0.5, -0.5],
            0.05,
            0.5,
        )
        .unwrap();
        for (x, want) in [([0.0, 0.0], false), ([1.0, 0.0], true), ([0.0, 1.0], true), ([1.0, 1.0], false)] {
            assert_eq!(cell.fires(&x).unwrap(), want, "input {x:?}");
        }
        let outs = cell.branch_outputs(&[1.0, 0.0]).unwrap();
        assert!(outs[0] > 0.99 && outs[1] < 0.01, "{outs:?}");
        // A branch output is a SIGMOID, bounded in (0, 1): an exponential also computes XOR here
        // and survived the first mutation sweep, so the bound is asserted on a strongly driven
        // branch and the midpoint on a zero-driven one.
        assert!(outs[0] < 1.0, "a branch saturated past one: {}", outs[0]);
        let flat = BranchedNeuron::new(vec![vec![1.0]], vec![0.0], 0.05, 0.5).unwrap();
        assert_eq!(flat.branch_outputs(&[0.0]).unwrap(), vec![0.5]);
        assert!(flat.branch_outputs(&[100.0]).unwrap()[0] <= 1.0);
        // And the same weights summed into ONE branch — a point neuron — fail on (1, 1) or (0, 0):
        // a single sigmoid of the sum sees 0 for both corners and cannot separate them from the
        // sum 0 of the two mixed corners either.
        let point = BranchedNeuron::new(vec![vec![0.0, 0.0]], vec![-1.0], 0.05, 0.5).unwrap();
        let verdicts: Vec<bool> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]].iter().map(|x| point.fires(x).unwrap()).collect();
        assert!(verdicts.iter().all(|v| !v));
    }

    /// Every refusal names the problem.
    #[test]
    fn the_refusals_name_the_problem() {
        let mut bad = TwoCompartment::round_defaults();
        bad.c = 0.0;
        assert!(matches!(bad.validate(), Err(DendriteError::OutOfRange { what: "c", .. })));
        let mut bad = TwoCompartment::round_defaults();
        bad.e_i = 0.0;
        assert!(matches!(bad.validate(), Err(DendriteError::OutOfRange { .. })));
        let mut s = TwoCompartment::round_defaults();
        assert!(matches!(s.step(0.0, -60e-3, 0.0, 0.0), Err(DendriteError::OutOfRange { what: "dt", .. })));
        assert!(matches!(s.step(1e-4, f64::NAN, 0.0, 0.0), Err(DendriteError::NonFinite { what: "v_d", .. })));
        assert!(matches!(RateFunction::new(0.0, 0.0, 1.0), Err(DendriteError::OutOfRange { what: "rate_max", .. })));
        let phi = RateFunction::new(1.0, 0.0, 1.0).unwrap();
        assert!(matches!(DendriticLearner::new(s, phi, vec![], 1.0), Err(DendriteError::Empty { what: "inputs" })));
        assert!(matches!(DendriticLearner::new(s, phi, vec![1.0], 0.0), Err(DendriteError::OutOfRange { what: "eta", .. })));
        let cell = DendriticLearner::new(s, phi, vec![1.0, 2.0], 1.0).unwrap();
        assert!(matches!(cell.dendrite(&[1.0]), Err(DendriteError::Dimension { what: "psp", got: 1, want: 2 })));
        assert!((cell.dendrite(&[1e-3, 2e-3]).unwrap() - (-70e-3 + 5e-3)).abs() < 1e-15, "rest plus the weighted PSPs");
        assert!(matches!(BranchedNeuron::new(vec![], vec![], 1.0, 0.0), Err(DendriteError::Empty { what: "branches" })));
        assert!(matches!(BranchedNeuron::new(vec![vec![1.0], vec![1.0, 2.0]], vec![0.0, 0.0], 1.0, 0.0), Err(DendriteError::Dimension { .. })));
        assert!(matches!(BranchedNeuron::new(vec![vec![1.0]], vec![0.0], 0.0, 0.0), Err(DendriteError::OutOfRange { what: "scale", .. })));
        for e in [
            DendriteError::OutOfRange { what: "w", value: 9.0, low: 0.0, high: 1.0 },
            DendriteError::NonFinite { what: "z", index: 0 },
            DendriteError::Dimension { what: "y", got: 1, want: 2 },
            DendriteError::Empty { what: "x" },
        ] {
            assert!(!e.to_string().is_empty());
        }
    }


    /// The round defaults have `g_D = g_L`, and with them four mutations survived the second sweep:
    /// the two conductances swapped in the prediction, their ratio inverted in the dendritic
    /// target, the sign of the prediction error, and a learning step that ignored `dt`. A coupling
    /// of half the leak tells the two conductances apart.
    #[test]
    fn a_coupling_that_is_not_the_leak_shows_which_conductance_is_which() {
        let soma = TwoCompartment { g_d: 25e-9, ..TwoCompartment::round_defaults() };
        let v_d = -50e-3;
        // Four operations on potentials of at most 70 mV: four ulps of that.
        let ulps = 4.0 * f64::EPSILON * 70e-3;
        assert!((soma.prediction(v_d) - (50e-9 * -70e-3 + 25e-9 * v_d) / 75e-9).abs() < ulps);
        let phi = RateFunction::new(100.0, -55e-3, 5e-3).unwrap();
        let mut learner = DendriticLearner::new(soma, phi, vec![1e-3, -2e-3], 1e-2).unwrap();
        // V_d = V_T + (g_L/g_D)(V_T − E_L) = −60 + 2·10 = −40 mV.
        assert!((learner.dendrite_for_target(-60e-3).unwrap() - -40e-3).abs() < ulps);
        // And that dendrite does predict the target: the two closed forms agree with each other.
        assert!((soma.prediction(-40e-3) - -60e-3).abs() < ulps);
        // A soma ABOVE what its dendrite predicts is a positive error.
        learner.soma.v = -50e-3;
        assert!(learner.prediction_error(&[0.0, 0.0]).unwrap() > 0.0);
        learner.soma.v = -75e-3;
        assert!(learner.prediction_error(&[0.0, 0.0]).unwrap() < 0.0);
        // One step moves each weight by η·dt·(φ(V) − φ(V*))·φ′(V*)·psp — with the dt.
        learner.soma.v = -50e-3;
        let psp = [2.0, 5.0];
        let before = learner.w.clone();
        let dt = 1e-4;
        let (v, v_star) = learner.step(dt, &psp, 0.0, 0.0).unwrap();
        let gain = 1e-2 * dt * (phi.rate(v) - phi.rate(v_star)) * phi.slope(v_star);
        assert!(gain.abs() > 1e-9, "the step taught nothing, so it would match any rule: {gain}");
        for i in 0..2 {
            let moved = learner.w[i] - before[i];
            assert!((moved / (gain * psp[i]) - 1.0).abs() < 1e-9, "weight {i} moved {moved}, the rule says {}", gain * psp[i]);
        }
    }

    /// A non-finite entry of any array is refused at ITS OWN index. The length half of
    /// `finite_slice` and the finiteness half are independent, and every bad-array probe in this
    /// module is a WRONG LENGTH — the only non-finite fixtures are scalars, which go through a
    /// different guard — so a scan that could never find anything, under any of the five names
    /// the function is called with, was read by nothing.
    #[test]
    fn a_non_finite_entry_of_an_array_is_refused_at_its_own_index() {
        let soma = TwoCompartment::round_defaults();
        let phi = RateFunction::new(100.0, -50e-3, 5e-3).unwrap();
        let cell = DendriticLearner::new(soma, phi, vec![1.0, 2.0, 3.0], 1.0).unwrap();
        for (slot, bad) in [(0usize, f64::NAN), (1, f64::INFINITY), (2, f64::NEG_INFINITY)] {
            let mut psp = vec![1e-3; 3];
            psp[slot] = bad;
            match cell.dendrite(&psp) {
                Err(DendriteError::NonFinite { what, index }) => assert_eq!((what, index), ("psp", slot)),
                other => panic!("a psp of {bad} at slot {slot} was accepted: {other:?}"),
            }
        }
        // The same scan under each of its other four names.
        assert!(matches!(
            DendriticLearner::new(soma, phi, vec![1.0, f64::NAN], 1.0),
            Err(DendriteError::NonFinite { what: "weights", index: 1 })
        ));
        assert!(matches!(
            BranchedNeuron::new(vec![vec![1.0, f64::NAN]], vec![0.0], 1.0, 0.0),
            Err(DendriteError::NonFinite { what: "branch weights", index: 1 })
        ));
        assert!(matches!(
            BranchedNeuron::new(vec![vec![1.0], vec![1.0]], vec![0.0, f64::INFINITY], 1.0, 0.0),
            Err(DendriteError::NonFinite { what: "branch biases", index: 1 })
        ));
        let branched = BranchedNeuron::new(vec![vec![1.0, 1.0]], vec![0.0], 1.0, 0.0).unwrap();
        assert!(matches!(
            branched.branch_outputs(&[1.0, f64::NAN]),
            Err(DendriteError::NonFinite { what: "input", index: 1 })
        ));
    }

    /// The round defaults are the seven constants the constructor's doc prints, and they START AT
    /// REST — `V = E_L`. Every existing use of them measures a change from wherever the soma
    /// happened to be, or compares two runs that both begin there, so the resting potential the
    /// cell is handed was never read as a value.
    #[test]
    fn the_round_defaults_are_the_documented_constants_and_start_at_rest() {
        let d = TwoCompartment::round_defaults();
        assert_eq!(
            (d.c, d.g_l, d.e_l, d.g_d, d.e_e, d.e_i, d.v),
            (1e-9, 50e-9, -70e-3, 50e-9, 0.0, -75e-3, -70e-3)
        );
        assert_eq!(d.v, d.e_l, "the defaults start at rest");
        d.validate().unwrap();
    }

    /// Validation reads the somatic potential, and a coupling of ZERO is a legal soma. The
    /// potential is the seventh of seven checks and the only one no existing fixture perturbs;
    /// and `g_d` is the one conductance that may be zero — [`DendriticLearner::dendrite_for_target`]
    /// exists to answer `None` for exactly that cell, which it could never be handed if
    /// validation refused it.
    #[test]
    fn validation_reads_the_somatic_potential_and_admits_a_coupling_of_zero() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let soma = TwoCompartment { v: bad, ..TwoCompartment::round_defaults() };
            assert!(matches!(soma.validate(), Err(DendriteError::NonFinite { what: "v", .. })), "v = {bad}");
        }
        let uncoupled = TwoCompartment { g_d: 0.0, ..TwoCompartment::round_defaults() };
        assert_eq!(uncoupled.validate(), Ok(()), "a dendrite coupled through nothing is still a soma");
        let phi = RateFunction::new(100.0, -50e-3, 5e-3).unwrap();
        let learner = DendriticLearner::new(uncoupled, phi, vec![1.0], 1.0).unwrap();
        assert_eq!(learner.dendrite_for_target(-60e-3), None);
        let negative = TwoCompartment { g_d: -1e-9, ..TwoCompartment::round_defaults() };
        assert!(matches!(negative.validate(), Err(DendriteError::OutOfRange { what: "g_d", .. })));
    }

    /// A rate function of zero transition width is refused. At `v_scale = 0` the exponent is a
    /// division by zero and `φ` becomes a step function whose slope is `0/0` — the one shape the
    /// learning rule cannot descend. The existing probe of the constructor is a zero MAXIMUM
    /// RATE, which the same guard rejects either way.
    #[test]
    fn a_rate_function_of_zero_or_negative_transition_width_is_refused() {
        assert!(matches!(RateFunction::new(100.0, -50e-3, 0.0), Err(DendriteError::OutOfRange { what: "v_scale", .. })));
        assert!(matches!(RateFunction::new(100.0, -50e-3, -1e-3), Err(DendriteError::OutOfRange { what: "v_scale", .. })));
        assert!(matches!(RateFunction::new(100.0, -50e-3, f64::NAN), Err(DendriteError::NonFinite { what: "v_scale", .. })));
        assert!(matches!(RateFunction::new(100.0, -50e-3, f64::INFINITY), Err(DendriteError::NonFinite { what: "v_scale", .. })));
    }

    /// A learner is refused if its soma does not validate. The constructor's other three
    /// refusals — no inputs, a non-positive `eta`, a non-finite weight — are all probed, and each
    /// of them fires on its own, so the soma's check could have its result discarded and every
    /// existing assertion would still hold.
    #[test]
    fn a_learner_is_refused_if_its_soma_does_not_validate() {
        let phi = RateFunction::new(100.0, -50e-3, 5e-3).unwrap();
        let no_capacitance = TwoCompartment { c: 0.0, ..TwoCompartment::round_defaults() };
        assert!(matches!(
            DendriticLearner::new(no_capacitance, phi, vec![1.0], 1.0),
            Err(DendriteError::OutOfRange { what: "c", .. })
        ));
        let no_leak = TwoCompartment { g_l: 0.0, ..TwoCompartment::round_defaults() };
        assert!(matches!(
            DendriticLearner::new(no_leak, phi, vec![1.0], 1.0),
            Err(DendriteError::OutOfRange { what: "g_l", .. })
        ));
        let inverted = TwoCompartment { e_i: 10e-3, ..TwoCompartment::round_defaults() };
        assert!(matches!(DendriticLearner::new(inverted, phi, vec![1.0], 1.0), Err(DendriteError::OutOfRange { .. })));
        let adrift = TwoCompartment { v: f64::NAN, ..TwoCompartment::round_defaults() };
        assert!(matches!(
            DendriticLearner::new(adrift, phi, vec![1.0], 1.0),
            Err(DendriteError::NonFinite { what: "v", .. })
        ));
    }

    /// The dendrite hangs off the REST potential, not off the soma. In the round defaults
    /// `E_L` and `V` are the same number — the cell starts at rest — so every existing reading of
    /// `dendrite` is taken on a cell where the two are indistinguishable; this one holds the soma
    /// 50 mV above its rest first. It matters because the dendrite is the soma's INPUT: a
    /// dendrite measured from the soma's own potential is a positive feedback loop.
    #[test]
    fn the_dendrite_hangs_off_the_rest_potential_and_not_off_the_soma() {
        let soma = TwoCompartment::round_defaults();
        let phi = RateFunction::new(100.0, -50e-3, 5e-3).unwrap();
        let mut cell = DendriticLearner::new(soma, phi, vec![1.0, 2.0], 1.0).unwrap();
        cell.soma.v = -20e-3;
        assert_eq!(cell.dendrite(&[1e-3, 2e-3]).unwrap(), -70e-3 + (1.0 * 1e-3 + 2.0 * 2e-3));
        // With no drive at all the dendrite sits at rest, wherever the soma has got to.
        assert_eq!(cell.dendrite(&[0.0, 0.0]).unwrap(), cell.soma.e_l);
        cell.soma.v = 0.0;
        assert_eq!(cell.dendrite(&[0.0, 0.0]).unwrap(), cell.soma.e_l);
    }

    /// The step reports the soma it LEAVES, and the rule reads that same potential. The existing
    /// step test reconstructs the rule's gain from the value `step` returned, so a step that
    /// returned the potential it started from is self-consistent with the weights it then wrote
    /// and passes; this one compares the returned value with the cell's own field afterwards,
    /// and recomputes the gain from the field rather than from the return.
    #[test]
    fn the_step_reports_the_soma_it_leaves_and_the_rule_reads_that_same_potential() {
        let soma = TwoCompartment { g_d: 25e-9, ..TwoCompartment::round_defaults() };
        let phi = RateFunction::new(100.0, -55e-3, 5e-3).unwrap();
        let mut cell = DendriticLearner::new(soma, phi, vec![1e-3, -2e-3], 1e-2).unwrap();
        cell.soma.v = -20e-3;
        let psp = [2.0, 5.0];
        let (dt, started_at) = (1e-3, cell.soma.v);
        let w_before = cell.w.clone();
        let (v, v_star) = cell.step(dt, &psp, 0.0, 0.0).unwrap();
        assert_ne!(v, started_at, "the step did not move the soma, so it would match either reading");
        assert_eq!(v, cell.soma.v, "the step reported the soma it started from, not the one it left");
        // Δw_i = η · dt · (φ(V) − φ(V*)) · φ′(V*) · psp_i, with V read off the cell afterwards:
        // the same four multiplications in the same order, so the comparison is exact.
        let gain = 1e-2 * dt * (phi.rate(cell.soma.v) - phi.rate(v_star)) * phi.slope(v_star);
        assert!(gain.abs() > 1e-12, "the step taught nothing, so it would match any rule: {gain}");
        assert_eq!(cell.w[0], w_before[0] + gain * psp[0]);
        assert_eq!(cell.w[1], w_before[1] + gain * psp[1]);
    }

    /// A branched neuron needs at least one input, one bias PER BRANCH, and an input vector as
    /// wide as the branches are. Two of those three checks compare a length with itself under
    /// the mutation and so can never fire, and the third is a guard the suite never probes: its
    /// branch fixtures all have inputs, matched biases, and inputs of the right width, and a
    /// mismatch that gets through does not panic — `zip` simply stops at the shorter side and
    /// returns a shorter answer.
    #[test]
    fn a_branched_neuron_needs_an_input_a_bias_per_branch_and_the_width_it_was_built_with() {
        assert!(matches!(BranchedNeuron::new(vec![vec![]], vec![0.0], 1.0, 0.0), Err(DendriteError::Empty { what: "inputs" })));
        assert!(matches!(
            BranchedNeuron::new(vec![vec![], vec![]], vec![0.0, 0.0], 1.0, 0.0),
            Err(DendriteError::Empty { what: "inputs" })
        ));
        match BranchedNeuron::new(vec![vec![1.0], vec![1.0]], vec![0.0], 1.0, 0.0) {
            Err(DendriteError::Dimension { what, got, want }) => assert_eq!((what, got, want), ("branch biases", 1, 2)),
            other => panic!("two branches were built with one bias: {other:?}"),
        }
        match BranchedNeuron::new(vec![vec![1.0]], vec![0.0, 0.0, 0.0], 1.0, 0.0) {
            Err(DendriteError::Dimension { what, got, want }) => assert_eq!((what, got, want), ("branch biases", 3, 1)),
            other => panic!("one branch was built with three biases: {other:?}"),
        }
        let cell = BranchedNeuron::new(vec![vec![1.0, -1.0], vec![-1.0, 1.0]], vec![-0.5, -0.5], 0.05, 0.5).unwrap();
        match cell.branch_outputs(&[1.0]) {
            Err(DendriteError::Dimension { what, got, want }) => assert_eq!((what, got, want), ("input", 1, 2)),
            other => panic!("a two-input neuron read a one-element input: {other:?}"),
        }
        assert!(matches!(
            cell.fires(&[1.0, 0.0, 1.0]),
            Err(DendriteError::Dimension { what: "input", got: 3, want: 2 })
        ));
    }

    /// The soma fires on the SUM of its branches, and at the threshold itself rather than past
    /// it. The XOR fixture's branches are driven to within `e^{−10}` of 0 and 1, where the sum
    /// and the largest branch are the same number to three decimal places and the threshold of
    /// ½ is nowhere near either — so neither the summation nor the inclusiveness of the
    /// comparison was tested. Two branches at zero net drive are each EXACTLY ½ and sum to
    /// exactly 1, which puts both questions on an exact `f64`.
    #[test]
    fn the_soma_sums_its_branches_and_fires_at_the_threshold_itself() {
        let poised = BranchedNeuron::new(vec![vec![1.0], vec![1.0]], vec![0.0, 0.0], 0.05, 1.0).unwrap();
        assert_eq!(poised.branch_outputs(&[0.0]).unwrap(), vec![0.5, 0.5]);
        assert!(poised.fires(&[0.0]).unwrap(), "a sum exactly at the threshold fires");
        // A threshold no single branch reaches, and the two together do.
        let together = BranchedNeuron::new(vec![vec![1.0], vec![1.0]], vec![0.0, 0.0], 0.05, 0.9).unwrap();
        assert!(together.fires(&[0.0]).unwrap(), "two branches at a half each did not reach 0.9");
        // One place past the sum is one place too far.
        let unreachable = BranchedNeuron::new(vec![vec![1.0], vec![1.0]], vec![0.0, 0.0], 0.05, 1.0 + f64::EPSILON).unwrap();
        assert!(!unreachable.fires(&[0.0]).unwrap());
    }

    /// The XOR proof carries the factor of two that adding the two middle constraints gives.
    /// The verdict cannot: the question it asks is whether the half-open interval between the
    /// bounds holds a number, and `[2B, B)` and `[B, B)` are both empty for every `B > 0`, so
    /// dropping the factor leaves `point_neuron_can_xor` answering "no" for a reason the proof
    /// does not give. That is why [`xor_sum_bounds`] hands the bounds out — this test reads the
    /// step of the reasoning, and the one below it reads only the conclusion.
    #[test]
    fn the_xor_proof_carries_the_factor_of_two_that_adding_the_middle_constraints_gives() {
        assert!(!point_neuron_can_xor());
        let (sum_lower, sum_upper) = xor_sum_bounds();
        assert_eq!(sum_upper, f64::MIN_POSITIVE, "w1 + w2 < −b");
        assert_eq!(sum_lower, 2.0 * f64::MIN_POSITIVE, "w1 ≥ −b and w2 ≥ −b add to w1 + w2 ≥ −2b");
        assert_eq!(sum_lower / sum_upper, 2.0);
        assert!(sum_lower > sum_upper, "the lower bound is ABOVE the upper, which is the contradiction");
    }
}
