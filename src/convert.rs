//! `ANN`-to-`SNN` conversion: the other way of getting a trained spiking network.
//!
//! # The idea
//!
//! Training through spikes is hard because a spike has no derivative. The surrogate-gradient
//! literature works around that; **conversion sidesteps it entirely**. You take a network that was
//! already trained the ordinary way — dense layers, `ReLU` activations, backpropagation, whatever
//! framework you like — and you *reinterpret* each `ReLU` unit as a spiking neuron whose FIRING
//! RATE carries the number the `ReLU` used to output. No spiking gradient is ever needed, because
//! no training happens here at all.
//!
//! This is not a curiosity. It is how most deployed spiking vision models are actually produced.
//! `BrainChip`'s `Akida` ships `CNN2SNN`, which takes a quantisation-aware-trained `Keras` model and
//! maps it onto the chip's event-driven form; `IBM`'s `NorthPole` flow likewise begins from a
//! conventionally trained network rather than from a spiking one. Whether either uses the specific
//! normalisation rules below, this review did not establish: the published tool descriptions say
//! "convert a trained model" and do not print the scale. What is not in doubt is the direction of
//! travel — train in the framework, convert for the part — and a library that teaches spiking
//! computing without teaching conversion is teaching the half that ships less.
//!
//! **Primary sources.** Diehl, Neil, Binas, Cook, Liu & Pfeiffer, *Fast-classifying, high-accuracy
//! spiking deep networks through weight and threshold balancing*, IJCNN 2015 — the threshold
//! balancing and weight normalisation. Rueckauer, Lungu, Hu, Pfeiffer & Liu, *Conversion of
//! continuous-valued deep networks to efficient event-driven networks for image classification*,
//! Front. Neurosci. 11:682, 2017 — the percentile-based robust normalisation and the
//! reset-by-subtraction correction. Everything in this module is one of those two papers, an exact
//! consequence of one of them, or a caveat marked as this implementation's own.
//!
//! # The equivalence, derived
//!
//! Take a non-leaky integrate-and-fire neuron: capacitance `C` farads, threshold `V_th` volts,
//! starting at zero, driven by a constant current `I` amperes.
//!
//! ```text
//! C dV/dt = I      =>   V(t) = I t / C
//! ```
//!
//! It reaches threshold at `t = C V_th / I`, and if the reset SUBTRACTS the threshold it lands back
//! at exactly zero and repeats. So the firing rate is
//!
//! ```text
//! r = I / (C V_th)          spikes per second, exactly, for I > 0
//! ```
//!
//! — **linear in the input current, with no leak term and no saturation**. That is a `ReLU`: linear
//! above zero, and exactly flat below it, because a non-positive current never reaches threshold
//! and produces not "few spikes" but *no spikes*.
//!
//! Discretise time into ticks of `dt` seconds, allow at most one spike per tick (which is what
//! digital neuromorphic hardware does), and the rate is capped at `1/dt`. Normalise by that cap and
//! the whole thing becomes dimensionless:
//!
//! ```text
//! z = I dt / (C V_th)       the "activation": spikes per tick, 0..1
//! ```
//!
//! So converting a `ReLU` unit means choosing the current gain `g = C V_th / dt` amperes per unit
//! of activation, and driving the neuron with `I = z g` where `z` is the `ReLU`'s pre-activation.
//! Only the **ratio** `C V_th / dt` enters the answer; `C`, `V_th` and `dt` are exposed separately
//! so that a reader can match a datasheet, and
//! `only_the_ratio_c_v_th_over_dt_changes_the_spike_train` proves they only matter as that ratio.
//!
//! # Where the error comes from, and why it is exactly `1/T`
//!
//! Run that neuron for `T` ticks under constant `z`. Each tick delivers `z V_th` volts of charge.
//! After `T` ticks the membrane has received `T z V_th`, of which `N V_th` has been spent on `N`
//! spikes and the remainder stays on the membrane, strictly below one threshold. Hence
//!
//! ```text
//! N = floor(T z)        and       |N/T - z| < 1/T
//! ```
//!
//! **The conversion error is at most one spike, always.** It falls as `1/T` and it is a
//! quantisation error, not a modelling error: the spiking unit is not approximating the `ReLU`
//! badly, it is reporting the `ReLU`'s value to a resolution of one part in `T`. That single line
//! is the latency/accuracy trade-off every deployment argues about, and
//! [`error_vs_ticks`] measures it while [`ErrorCurve::fit_exponent`] recovers the `-1`: the fitted
//! exponent over eight tick counts from 16 to 2048 on this module's fixture is **-1.02**, and the
//! error over that sweep falls from 0.0227 to 0.00016 of the layer's dynamic range.
//!
//! # Reset by subtraction versus reset to zero: a two-line difference worth a factor of a hundred
//!
//! The derivation above assumed the reset SUBTRACTS the threshold. Diehl et al. (2015) reset to
//! zero, as a biological neuron does. That throws the residual charge away, and the residual is not
//! small: a neuron whose input is `z` per tick needs `ceil(1/z)` ticks to cross, so its rate
//! converges to
//!
//! ```text
//! r_zero = 1 / ceil(1/z)        rather than z
//! ```
//!
//! At `z = 0.99` that is `1/2`. **An error of 0.49, and it does not shrink with `T`** — it is a
//! bias, not a variance, and running longer cannot remove it. Rueckauer et al. (2017) identified
//! this as the single largest source of conversion loss and replaced the reset with `V <- V - V_th`,
//! which keeps the overshoot for the next interval. [`Reset`] carries both,
//! [`Reset::spikes_in`] gives the exact spike count of each in closed form, and
//! `reset_by_subtraction_beats_reset_to_zero_and_here_is_by_how_much` measures the gap on one
//! network: **0.00085 versus 0.0418 mean absolute activation error at `T = 512`, a factor of 49**.
//! Beside that figure, the caveat that gives it its meaning: quadrupling the run to `T = 2048` cuts
//! the first number and leaves the second where it was, because one is a variance and the other is
//! a bias. The factor of 49 is not a constant — it grows with `T`.
//!
//! # Normalisation: why a trained network cannot be converted as it stands
//!
//! A `ReLU` is unbounded; a spiking unit cannot exceed one spike per tick. A trained layer whose
//! activations reach 7.0 would need `z = 7`, and the neuron would simply fire flat out and report a
//! saturated 1 instead. So every activation must first be squeezed into `0..1` by dividing each
//! layer by a scale `λ_l`. Because `ReLU` is positively homogeneous — `ReLU(λz) = λ ReLU(z)` for
//! `λ > 0` — this **changes nothing about the function** as long as the weights are rescaled with
//! it:
//!
//! ```text
//! W_l <- W_l λ_{l-1} / λ_l          b_l <- b_l / λ_l
//! ```
//!
//! and the output comes back in the original units after multiplying by `λ_L`. That exactness is
//! the invariant `normalisation_preserves_the_function_in_the_rate_limit` checks, to 1e-12.
//!
//! Two ways to pick `λ_l`, both implemented in [`Norm`]:
//!
//! - **Model-based** ([`Norm::ModelBased`]): the largest activation the layer *could* produce, from
//!   the weights alone, propagated forward. Needs no data, and is a true upper bound, so nothing can
//!   ever saturate. It is also hopelessly loose — it assumes every input simultaneously takes its
//!   maximum and hits only the positive weights — so real activations end up a long way below 1 and
//!   need a long `T` to be resolved.
//! - **Data-based** ([`Norm::DataBased`]): the p-th percentile of the activations actually observed
//!   on a sample. `p = 100` is Diehl et al.'s maximum. `p = 99.9` is Rueckauer et al.'s **robust
//!   normalisation**, and the reason it exists is worth stating plainly: a single outlier activation
//!   sets `λ` for the whole layer, and every other unit is then scaled down in proportion. In
//!   `the_percentile_normaliser_is_why_robust_normalisation_exists` one sample out of 201 is 100×
//!   the rest; under maximum normalisation a typical unit emits **0 spikes in 100 ticks**, and under
//!   the 99th percentile the same unit emits **50**. The price is honest and stated: activations
//!   above the percentile now saturate, so robust normalisation trades a guaranteed bound for a
//!   usable dynamic range. On this module's own fixture the model-based bound comes out **7.0x**
//!   the largest activation ever observed in the second layer, which is the same problem wearing
//!   its other face.
//!
//! # What this module does NOT do
//!
//! - **It does not train.** It converts. Supply an already-trained [`Mlp`].
//! - **Dense layers only.** Convolutions, average and max pooling, and `BatchNorm` folding are all
//!   given as conversion rules in Rueckauer et al. (2017) §2.3–2.4; this implementation did not
//!   attempt them. The dense case carries the whole argument and none of the index arithmetic.
//! - **It does not claim conversion is a good deal.** It is a latency-for-accuracy trade, and
//!   [`SpikingMlp::ledger`] prices it: `a_converted_network_at_realistic_latency_is_refuted_by_
//!   every_published_crossover` runs the converted network past [`crate::crossover`] and gets
//!   `Refuted` from all three published thresholds. The measured figure for a 20-64-10 network at
//!   `T = 128` with analog input is **91.8 spikes per synapse per inference**, against published
//!   thresholds of 1.72, 1.38 and 0.35 — fifty times the most permissive of them. A `T`-tick
//!   conversion does `T` membrane updates per neuron per inference to replace one multiply-add. The
//!   energy case for conversion has to come from somewhere other than the operation count, and this
//!   module reports the count rather than arguing about it.
//!
//! # Units
//!
//! SI at every interface: `dt` in seconds, `c` in farads, `v_th` in volts, currents in amperes. The
//! conversion literature is written in a dimensionless frame — "threshold 1, one timestep" — and
//! that frame is kept INSIDE the arithmetic where a reader can compare it against the papers, with
//! the gain `C V_th / dt` doing the conversion at the boundary. [`SpikingRelu::activation`] and
//! [`SpikingRelu::gain`] are that boundary, in both directions.

use crate::ledger::Ledger;
use crate::neuron::Neuron;
use crate::rng::Rng;
use core::fmt;

/// Why a conversion could not be performed.
///
/// Every variant names the offending object by index. A conversion that silently accepted a `NaN`
/// weight would produce a network that emits zero spikes and reports a plausible-looking accuracy
/// of chance, which is the failure mode this enum exists to prevent.
#[derive(Debug, Clone, PartialEq)]
pub enum ConvertError {
    /// A layer had zero inputs or zero units. Both counts are given because either can be the
    /// mistake and the caller usually knows which.
    EmptyLayer {
        /// Inputs the layer declared.
        n_in: usize,
        /// Units the layer declared.
        n_out: usize,
    },
    /// The weight vector's length was not `n_in * n_out`.
    WeightCount {
        /// Length implied by the declared shape.
        expected: usize,
        /// Length supplied.
        got: usize,
    },
    /// The bias vector's length was not `n_out`.
    BiasCount {
        /// Length implied by the declared shape.
        expected: usize,
        /// Length supplied.
        got: usize,
    },
    /// A weight was not a finite number, at row `unit`, column `input` of the row-major matrix.
    NonFiniteWeight {
        /// Output unit whose row holds the offending weight.
        unit: usize,
        /// Input index within that row.
        input: usize,
    },
    /// A bias was not a finite number.
    NonFiniteBias {
        /// Output unit whose bias was rejected.
        unit: usize,
    },
    /// An input component was not a finite number.
    NonFiniteInput {
        /// Position in the input vector.
        index: usize,
    },
    /// An input vector's length did not match the layer's input count.
    InputLength {
        /// Length the layer requires.
        expected: usize,
        /// Length supplied.
        got: usize,
    },
    /// Two consecutive layers do not chain: layer `layer` takes `expected` inputs and its
    /// predecessor produces `got` outputs.
    Disconnected {
        /// Index of the layer whose input count is unsatisfied.
        layer: usize,
        /// Inputs that layer requires.
        expected: usize,
        /// Outputs the previous layer produces.
        got: usize,
    },
    /// A network with no layers has no function to convert.
    NoLayers,
    /// Data-based normalisation was asked for with no samples to observe.
    ///
    /// Refused rather than falling back to model-based normalisation, because the fallback would
    /// silently change which paper's method was used and the caller would not be told.
    NoSamples,
    /// A percentile outside `(0, 100]`, or not finite.
    BadPercentile {
        /// The value supplied.
        p: f64,
    },
    /// A layer's normalisation scale came out non-positive or non-finite, which means the sample
    /// never activated the layer at all. Dividing by it would produce infinities; refusing names
    /// the layer instead.
    DegenerateScale {
        /// Which layer, indexed from the input side, with `0` meaning the input itself.
        layer: usize,
        /// The scale that was rejected.
        lambda: f64,
    },
    /// A list of normalisation scales was the wrong length. There must be one per layer plus one
    /// for the input.
    ScaleCount {
        /// Scales required: layers plus one.
        expected: usize,
        /// Scales supplied.
        got: usize,
    },
    /// A timing, threshold or capacitance parameter was non-positive or non-finite.
    BadParameter {
        /// Which parameter, as its field name.
        name: &'static str,
        /// The value that was rejected.
        value: f64,
    },
    /// A run of zero ticks was requested. A rate over zero ticks has no value — not a value of
    /// zero — and the division that would produce it is refused here rather than returning `NaN`.
    NoTicks,
}

impl fmt::Display for ConvertError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyLayer { n_in, n_out } => {
                write!(f, "a layer of shape {n_in} -> {n_out} has no weights to convert")
            }
            Self::WeightCount { expected, got } => {
                write!(f, "expected {expected} weights, got {got}")
            }
            Self::BiasCount { expected, got } => write!(f, "expected {expected} biases, got {got}"),
            Self::NonFiniteWeight { unit, input } => {
                write!(f, "weight [unit {unit}, input {input}] is not finite")
            }
            Self::NonFiniteBias { unit } => write!(f, "bias of unit {unit} is not finite"),
            Self::NonFiniteInput { index } => write!(f, "input component {index} is not finite"),
            Self::InputLength { expected, got } => {
                write!(f, "layer takes {expected} inputs, got {got}")
            }
            Self::Disconnected { layer, expected, got } => {
                let prev = layer.saturating_sub(1);
                write!(f, "layer {layer} takes {expected} inputs but layer {prev} emits {got}")
            }
            Self::NoLayers => f.write_str("a network with no layers has no function to convert"),
            Self::NoSamples => {
                f.write_str("data-based normalisation needs samples and none were supplied")
            }
            Self::BadPercentile { p } => write!(f, "percentile {p} is outside (0, 100]"),
            Self::DegenerateScale { layer, lambda } => {
                write!(f, "layer {layer} normalises by {lambda}, which is not a usable scale")
            }
            Self::ScaleCount { expected, got } => {
                write!(f, "expected {expected} normalisation scales, got {got}")
            }
            Self::BadParameter { name, value } => {
                write!(f, "{name} = {value} must be finite and strictly positive")
            }
            Self::NoTicks => f.write_str("a rate over zero ticks has no value"),
        }
    }
}

/// So that `?` works in a caller whose error type is `Box<dyn Error>`, as every example and doctest
/// in this crate uses.
impl std::error::Error for ConvertError {}

/// What a neuron does with its membrane potential when it spikes.
///
/// The choice is two lines of code and it dominates the conversion error. See the module doc.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reset {
    /// `V <- 0`. Diehl et al., IJCNN 2015, and what a biological neuron approximately does.
    ///
    /// Discards the overshoot `V - V_th`, which is charge the input actually delivered. The rate
    /// converges to `1/ceil(1/z)` instead of `z`, an error of up to 0.5 in activation units that
    /// **does not fall with the number of ticks**.
    ToZero,
    /// `V <- V - V_th`. Rueckauer et al., Front. Neurosci. 11:682, 2017, §2.1.
    ///
    /// Keeps the residual for the next interval, so no charge is lost and the only error left is
    /// the charge still on the membrane when the run ends — strictly less than one spike, hence
    /// `1/T`.
    BySubtraction,
}

impl fmt::Display for Reset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::ToZero => "reset-to-zero",
            Self::BySubtraction => "reset-by-subtraction",
        })
    }
}

impl Reset {
    /// Exact spike count over `ticks` ticks under a constant activation `z`, starting from rest.
    ///
    /// Closed form, not a simulation: `floor(T z)` for [`Reset::BySubtraction`] and
    /// `floor(T / ceil(1/z))` for [`Reset::ToZero`], both clamped at one spike per tick.
    /// `spikes_in_matches_the_simulated_neuron_exactly` checks a simulated neuron against it.
    ///
    /// `z <= 0` returns `Some(0)` rather than `None`, and the distinction from
    /// [`crate::neuron::Lif::rate`] is deliberate. A sub-threshold leaky neuron has no firing rate,
    /// so that method refuses. Here zero **is the answer**: `ReLU` is exactly flat below zero and
    /// the converted unit reproduces the flat region exactly, which is the one part of the
    /// conversion that carries no error at all.
    ///
    /// `None` only for a non-finite `z`, which is a broken caller rather than a quiet region.
    #[must_use]
    pub fn spikes_in(self, z: f64, ticks: u64) -> Option<u64> {
        if !z.is_finite() {
            return None;
        }
        if z <= 0.0 {
            return Some(0);
        }
        let t = ticks as f64;
        Some(match self {
            Self::BySubtraction => {
                let n = t * z;
                if n >= t { ticks } else { n.floor() as u64 }
            }
            Self::ToZero => {
                let k = (1.0 / z).ceil();
                if !k.is_finite() || k < 1.0 {
                    ticks
                } else {
                    // `k` is the exact inter-spike interval in ticks, so the count is an integer
                    // division rather than a rounded product.
                    ticks / (k as u64).max(1)
                }
            }
        })
    }

    /// The rate this reset rule converges to as `ticks -> infinity`, in spikes per tick.
    ///
    /// For [`Reset::BySubtraction`] that is `min(z, 1)` — the conversion is asymptotically exact.
    /// For [`Reset::ToZero`] it is `1/ceil(1/z)`, and the gap to `z` is the bias no amount of time
    /// removes. `None` for a non-finite `z`.
    #[must_use]
    pub fn rate_limit(self, z: f64) -> Option<f64> {
        if !z.is_finite() {
            return None;
        }
        if z <= 0.0 {
            return Some(0.0);
        }
        Some(match self {
            Self::BySubtraction => z.min(1.0),
            Self::ToZero => 1.0 / (1.0 / z).ceil().max(1.0),
        })
    }
}

/// A non-leaky integrate-and-fire neuron used as a `ReLU`'s rate-coded twin.
///
/// Distinct from [`crate::neuron::IntegrateAndFire`] in exactly one respect that matters: it can
/// reset by subtraction. That is the whole of Rueckauer et al.'s correction, and it is not a
/// variant of the biological model, so it lives here with the conversion rather than in the neuron
/// zoo.
///
/// Its closed form is `r = I / (C V_th)` spikes per second for `I > 0`, exactly, with no leak term.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpikingRelu {
    /// Membrane capacitance, farads. Enters the answer only through `c * v_th / dt`.
    pub c: f64,
    /// Firing threshold, volts. A spike is emitted when the potential reaches it.
    pub v_th: f64,
    /// Current membrane potential, volts. Starts at zero, not at a resting potential: there is no
    /// rest here because there is no leak.
    pub v: f64,
    /// What happens to [`SpikingRelu::v`] on a spike. See [`Reset`].
    pub reset: Reset,
    /// Whether the potential is clamped at zero from below, seconds-independent.
    ///
    /// `false` is the linear integrator, and is what the derivation in the module doc assumes. For
    /// CONSTANT input the flag changes nothing, and
    /// `a_negative_pre_activation_produces_exactly_zero_spikes` holds either way. For time-varying
    /// input it matters: without the clamp, a long negative stretch digs a well the later positive
    /// input has to climb out of, delaying the unit by an amount proportional to the depth. With
    /// it, the unit forgets. Rueckauer et al. discuss the same effect for the input layer; this
    /// implementation exposes the choice rather than picking one, because the clamp makes the unit
    /// no longer a linear integrator and that is a real cost.
    pub floor_at_zero: bool,
}

impl SpikingRelu {
    /// A neuron at rest with the given capacitance, threshold and reset rule.
    ///
    /// No validation here: the parameters are validated once, at [`Config::validate`], where the
    /// error can name the field the caller actually set.
    #[must_use]
    pub fn new(c: f64, v_th: f64, reset: Reset) -> Self {
        Self { c, v_th, v: 0.0, reset, floor_at_zero: false }
    }

    /// Amperes per unit of activation, `c * v_th / dt`.
    ///
    /// This is the conversion boundary: an activation of 1 becomes the current that produces
    /// exactly one spike per tick.
    #[must_use]
    pub fn gain(&self, dt: f64) -> f64 {
        self.c * self.v_th / dt
    }

    /// The activation a current of `i` amperes represents, `i * dt / (c * v_th)` — the inverse of
    /// [`SpikingRelu::gain`], in spikes per tick.
    #[must_use]
    pub fn activation(&self, i: f64, dt: f64) -> f64 {
        i * dt / (self.c * self.v_th)
    }

    /// Steady-state firing rate under constant current `i`, in hertz, in closed form.
    ///
    /// `i / (c * v_th)`, exactly, for `i > 0`, ignoring the one-spike-per-tick cap that discrete
    /// time imposes — divide by `dt` yourself to find where the cap bites.
    ///
    /// `Some(0.0)` for `i <= 0`: the potential is monotonically non-increasing, so the neuron
    /// provably never fires and zero is the exact answer rather than a rounded small one. `None`
    /// only for a non-finite current.
    #[must_use]
    pub fn rate(&self, i: f64) -> Option<f64> {
        if !i.is_finite() {
            return None;
        }
        if i <= 0.0 {
            return Some(0.0);
        }
        Some(i / (self.c * self.v_th))
    }

    /// Integrate for `dt` seconds under current `i` WITHOUT testing the threshold.
    ///
    /// Used for the output layer under [`Readout::MembranePotential`], where the accumulated
    /// potential is the readout and a spike would throw part of it away.
    pub fn accumulate(&mut self, dt: f64, i: f64) {
        self.v += i * dt / self.c;
    }
}

impl Neuron for SpikingRelu {
    /// True. The model is linear in `i * dt` and motionless at `i = 0`, so a gap of quiet ticks
    /// changes nothing whether it is crossed in one step or in a thousand. The `floor_at_zero`
    /// clamp preserves this: a non-positive potential clamps to zero on the first quiet step and
    /// stays there.
    const EXACT_OVER_GAPS: bool = true;

    fn step(&mut self, dt: f64, i: f64) -> bool {
        self.v += i * dt / self.c;
        if self.floor_at_zero && self.v < 0.0 {
            self.v = 0.0;
        }
        if self.v >= self.v_th {
            match self.reset {
                Reset::ToZero => self.v = 0.0,
                Reset::BySubtraction => self.v -= self.v_th,
            }
            return true;
        }
        false
    }

    fn bump(&mut self, dv: f64) {
        self.v += dv;
    }

    fn potential(&self) -> f64 {
        self.v
    }

    fn reset(&mut self) {
        self.v = 0.0;
    }
}

/// One dense `ReLU` layer of an already-trained network: `a = max(0, W x + b)`.
///
/// This module converts; it does not train. The weights arrive from somewhere else — a `PyTorch`
/// checkpoint, a `NumPy` array, a hand-built test — and this type is the smallest representation
/// that the conversion rules need. `w` is row-major, `n_out` rows of `n_in`.
#[derive(Debug, Clone, PartialEq)]
pub struct DenseRelu {
    /// Inputs the layer consumes.
    pub n_in: usize,
    /// Units the layer produces.
    pub n_out: usize,
    /// Weights, row-major: `w[u * n_in + j]` connects input `j` to unit `u`. Every entry is finite,
    /// which [`DenseRelu::new`] enforces once so that nothing downstream has to re-check.
    pub w: Vec<f64>,
    /// One bias per unit, all finite. In the converted network this becomes a constant input
    /// current applied on every tick, which is why it is normalised by `λ_l` alone while the
    /// weights are normalised by `λ_{l-1}/λ_l`.
    pub b: Vec<f64>,
}

impl DenseRelu {
    /// Build a layer, rejecting a shape that does not match and any non-finite parameter.
    ///
    /// # Errors
    ///
    /// [`ConvertError::EmptyLayer`], [`ConvertError::WeightCount`], [`ConvertError::BiasCount`],
    /// [`ConvertError::NonFiniteWeight`] or [`ConvertError::NonFiniteBias`], naming the index.
    pub fn new(n_in: usize, n_out: usize, w: Vec<f64>, b: Vec<f64>) -> Result<Self, ConvertError> {
        if n_in == 0 || n_out == 0 {
            return Err(ConvertError::EmptyLayer { n_in, n_out });
        }
        if w.len() != n_in * n_out {
            return Err(ConvertError::WeightCount { expected: n_in * n_out, got: w.len() });
        }
        if b.len() != n_out {
            return Err(ConvertError::BiasCount { expected: n_out, got: b.len() });
        }
        for u in 0..n_out {
            for j in 0..n_in {
                if !w[u * n_in + j].is_finite() {
                    return Err(ConvertError::NonFiniteWeight { unit: u, input: j });
                }
            }
            if !b[u].is_finite() {
                return Err(ConvertError::NonFiniteBias { unit: u });
            }
        }
        Ok(Self { n_in, n_out, w, b })
    }

    /// `W x + b`, before the rectifier.
    ///
    /// Exposed because the conversion's membrane-potential readout estimates exactly this quantity
    /// and nothing else — see [`Readout::MembranePotential`].
    ///
    /// # Errors
    ///
    /// [`ConvertError::InputLength`] or [`ConvertError::NonFiniteInput`].
    pub fn pre_activation(&self, x: &[f64]) -> Result<Vec<f64>, ConvertError> {
        if x.len() != self.n_in {
            return Err(ConvertError::InputLength { expected: self.n_in, got: x.len() });
        }
        for (j, v) in x.iter().enumerate() {
            if !v.is_finite() {
                return Err(ConvertError::NonFiniteInput { index: j });
            }
        }
        let mut out = vec![0.0; self.n_out];
        for u in 0..self.n_out {
            let row = &self.w[u * self.n_in..(u + 1) * self.n_in];
            let mut acc = self.b[u];
            for j in 0..self.n_in {
                acc += row[j] * x[j];
            }
            out[u] = acc;
        }
        Ok(out)
    }

    /// `max(0, W x + b)`.
    ///
    /// # Errors
    ///
    /// As [`DenseRelu::pre_activation`].
    pub fn forward(&self, x: &[f64]) -> Result<Vec<f64>, ConvertError> {
        let mut out = self.pre_activation(x)?;
        for v in &mut out {
            *v = v.max(0.0);
        }
        Ok(out)
    }

    /// The largest activation this layer could produce if every input simultaneously reached
    /// `input_max` and only the positive weights were driven.
    ///
    /// This is Diehl et al.'s model-based scale. It is a genuine upper bound — nothing can saturate
    /// under it — and it is loose by construction, because no real input hits every positive weight
    /// at once. `model_based_normalisation_bounds_every_observed_activation` checks the bound holds
    /// and `the_model_based_bound_is_loose_and_that_is_the_cost` measures how loose.
    ///
    /// Returns `0.0` for a layer whose every unit is dead, which callers must treat as degenerate
    /// rather than dividing by.
    #[must_use]
    pub fn max_possible_activation(&self, input_max: f64) -> f64 {
        let mut best = 0.0f64;
        for u in 0..self.n_out {
            let row = &self.w[u * self.n_in..(u + 1) * self.n_in];
            let mut acc = self.b[u];
            for &wv in row {
                if wv > 0.0 {
                    acc += wv * input_max;
                }
            }
            if acc > best {
                best = acc;
            }
        }
        best
    }

    /// Apply the layer's share of a normalisation: `W <- W λ_prev / λ`, `b <- b / λ`.
    ///
    /// The caller is responsible for `λ > 0`; [`Mlp::apply_scales`] checks it once for the whole
    /// stack so the check is not repeated per layer.
    pub fn apply_scales(&mut self, lambda_prev: f64, lambda: f64) {
        let f = lambda_prev / lambda;
        for v in &mut self.w {
            *v *= f;
        }
        for v in &mut self.b {
            *v /= lambda;
        }
    }

    /// The inverse of [`DenseRelu::apply_scales`].
    ///
    /// Exact in real arithmetic and exact in floating point whenever `λ` and `λ_prev` are powers of
    /// two; otherwise it round-trips to within one unit in the last place per weight, which
    /// `scaling_and_unscaling_round_trips` states as a measured bound rather than assuming.
    pub fn undo_scales(&mut self, lambda_prev: f64, lambda: f64) {
        let f = lambda / lambda_prev;
        for v in &mut self.w {
            *v *= f;
        }
        for v in &mut self.b {
            *v *= lambda;
        }
    }
}

/// A stack of dense `ReLU` layers: the thing being converted.
///
/// Deliberately minimal. It holds no optimiser, no gradients and no training loop, because this
/// module's claim is about the map from a trained network to a spiking one and a training
/// implementation here would only make that map harder to audit.
#[derive(Debug, Clone, PartialEq)]
pub struct Mlp {
    /// Layers in forward order; `layers[0]` sees the input.
    pub layers: Vec<DenseRelu>,
}

impl Mlp {
    /// Build from layers, checking they chain.
    ///
    /// # Errors
    ///
    /// [`ConvertError::NoLayers`] for an empty stack, or [`ConvertError::Disconnected`] naming the
    /// first layer whose input count does not match its predecessor's output count.
    pub fn new(layers: Vec<DenseRelu>) -> Result<Self, ConvertError> {
        if layers.is_empty() {
            return Err(ConvertError::NoLayers);
        }
        for l in 1..layers.len() {
            if layers[l].n_in != layers[l - 1].n_out {
                return Err(ConvertError::Disconnected {
                    layer: l,
                    expected: layers[l].n_in,
                    got: layers[l - 1].n_out,
                });
            }
        }
        Ok(Self { layers })
    }

    /// Inputs the network consumes. The `0` fallback is unreachable, because [`Mlp::new`] refuses
    /// an empty stack, and is written rather than unwrapped so that no public path can panic.
    #[must_use]
    pub fn n_in(&self) -> usize {
        self.layers.first().map_or(0, |l| l.n_in)
    }

    /// Units the network's last layer produces.
    #[must_use]
    pub fn n_out(&self) -> usize {
        self.layers.last().map_or(0, |l| l.n_out)
    }

    /// Total weights — the synapse count the converted network will have, for
    /// [`crate::ledger::Ledger::spikes_per_synapse`]. Biases are not synapses and are not counted.
    #[must_use]
    pub fn n_synapses(&self) -> u64 {
        self.layers.iter().map(|l| (l.n_in * l.n_out) as u64).sum()
    }

    /// The network's output.
    ///
    /// # Errors
    ///
    /// As [`DenseRelu::pre_activation`], for the first layer that rejects its input.
    pub fn forward(&self, x: &[f64]) -> Result<Vec<f64>, ConvertError> {
        let mut cur = x.to_vec();
        for l in &self.layers {
            cur = l.forward(&cur)?;
        }
        Ok(cur)
    }

    /// Every layer's activation, in forward order. `activations(x).last()` equals `forward(x)`.
    ///
    /// # Errors
    ///
    /// As [`Mlp::forward`].
    pub fn activations(&self, x: &[f64]) -> Result<Vec<Vec<f64>>, ConvertError> {
        let mut out = Vec::with_capacity(self.layers.len());
        let mut cur = x.to_vec();
        for l in &self.layers {
            cur = l.forward(&cur)?;
            out.push(cur.clone());
        }
        Ok(out)
    }

    /// The normalisation scales `λ_0 .. λ_L`, one per layer plus one for the input, WITHOUT
    /// applying them.
    ///
    /// `λ_0` is the input scale: `input_max` for [`Norm::ModelBased`], and the same percentile of
    /// the sample inputs for [`Norm::DataBased`].
    ///
    /// # Errors
    ///
    /// [`ConvertError::NoSamples`] if data-based normalisation got none,
    /// [`ConvertError::BadPercentile`], [`ConvertError::DegenerateScale`] naming a layer that the
    /// sample never activated, or anything [`Mlp::forward`] rejects.
    pub fn scales(&self, norm: Norm, samples: &[Vec<f64>]) -> Result<Vec<f64>, ConvertError> {
        let mut lam = Vec::with_capacity(self.layers.len() + 1);
        match norm {
            Norm::ModelBased { input_max } => {
                if !(input_max.is_finite() && input_max > 0.0) {
                    return Err(ConvertError::DegenerateScale { layer: 0, lambda: input_max });
                }
                lam.push(input_max);
                for (l, layer) in self.layers.iter().enumerate() {
                    let prev = lam[l];
                    let s = layer.max_possible_activation(prev);
                    if !(s.is_finite() && s > 0.0) {
                        return Err(ConvertError::DegenerateScale { layer: l + 1, lambda: s });
                    }
                    lam.push(s);
                }
            }
            Norm::DataBased { percentile: p } => {
                if !(p.is_finite() && p > 0.0 && p <= 100.0) {
                    return Err(ConvertError::BadPercentile { p });
                }
                if samples.is_empty() {
                    return Err(ConvertError::NoSamples);
                }
                let mut pool: Vec<f64> = samples.iter().flat_map(|s| s.iter().copied()).collect();
                let s0 = percentile(&mut pool, p)
                    .ok_or(ConvertError::DegenerateScale { layer: 0, lambda: f64::NAN })?;
                if !(s0.is_finite() && s0 > 0.0) {
                    return Err(ConvertError::DegenerateScale { layer: 0, lambda: s0 });
                }
                lam.push(s0);
                // One forward pass per sample, collecting each layer's activations across the whole
                // sample set before taking the percentile. Taking a per-sample percentile and
                // averaging is a different and wrong quantity: the outlier this method exists to
                // resist lives in one sample, and per-sample statistics hide it.
                let mut pools: Vec<Vec<f64>> = vec![Vec::new(); self.layers.len()];
                for s in samples {
                    let acts = self.activations(s)?;
                    for (l, a) in acts.iter().enumerate() {
                        pools[l].extend_from_slice(a);
                    }
                }
                for (l, pool) in pools.iter_mut().enumerate() {
                    let s = percentile(pool, p)
                        .ok_or(ConvertError::DegenerateScale { layer: l + 1, lambda: f64::NAN })?;
                    if !(s.is_finite() && s > 0.0) {
                        return Err(ConvertError::DegenerateScale { layer: l + 1, lambda: s });
                    }
                    lam.push(s);
                }
            }
        }
        Ok(lam)
    }

    /// Apply a list of scales `λ_0 .. λ_L` in place.
    ///
    /// # Errors
    ///
    /// [`ConvertError::ScaleCount`] for the wrong length, or [`ConvertError::DegenerateScale`] for
    /// any entry that is not finite and strictly positive.
    pub fn apply_scales(&mut self, lambdas: &[f64]) -> Result<(), ConvertError> {
        check_scales(lambdas, self.layers.len())?;
        for l in 0..self.layers.len() {
            self.layers[l].apply_scales(lambdas[l], lambdas[l + 1]);
        }
        Ok(())
    }

    /// Undo a list of scales applied by [`Mlp::apply_scales`].
    ///
    /// # Errors
    ///
    /// As [`Mlp::apply_scales`].
    pub fn undo_scales(&mut self, lambdas: &[f64]) -> Result<(), ConvertError> {
        check_scales(lambdas, self.layers.len())?;
        for l in 0..self.layers.len() {
            self.layers[l].undo_scales(lambdas[l], lambdas[l + 1]);
        }
        Ok(())
    }

    /// Compute the scales and apply them, returning `λ_0 .. λ_L`.
    ///
    /// After this the network computes `a_L / λ_L` instead of `a_L`, and every intermediate
    /// activation lies in `0..1` — exactly if the scale was model-based, and up to the saturating
    /// tail if it was a percentile.
    ///
    /// # Errors
    ///
    /// As [`Mlp::scales`] and [`Mlp::apply_scales`].
    pub fn normalise(&mut self, norm: Norm, samples: &[Vec<f64>]) -> Result<Vec<f64>, ConvertError> {
        let lam = self.scales(norm, samples)?;
        self.apply_scales(&lam)?;
        Ok(lam)
    }
}

fn check_scales(lambdas: &[f64], n_layers: usize) -> Result<(), ConvertError> {
    if lambdas.len() != n_layers + 1 {
        return Err(ConvertError::ScaleCount { expected: n_layers + 1, got: lambdas.len() });
    }
    for (l, &v) in lambdas.iter().enumerate() {
        if !(v.is_finite() && v > 0.0) {
            return Err(ConvertError::DegenerateScale { layer: l, lambda: v });
        }
    }
    Ok(())
}

/// How the per-layer scale `λ_l` is chosen.
///
/// The two entries are the two papers. See the module doc for why the second one exists.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Norm {
    /// Diehl et al., IJCNN 2015: the maximum activation the weights could ever produce, propagated
    /// layer by layer from an input bounded by `input_max`. Needs no data and cannot saturate.
    ModelBased {
        /// Upper bound on every input component, in the input's own units. For pixel data scaled to
        /// `0..1` this is `1.0`; supplying a bound the data exceeds breaks the guarantee, which is
        /// why it is a required parameter rather than a default.
        input_max: f64,
    },
    /// Diehl et al.'s data-based scale at `percentile = 100`, and Rueckauer et al.'s **robust**
    /// normalisation below that. The sample decides the scale, so the sample has to be
    /// representative — that is the method's assumption and it is not checkable from inside here.
    DataBased {
        /// Percentile of the observed activations to normalise by, in `(0, 100]`. Rueckauer et al.
        /// report 99.9 as a good default; 100 reproduces the 2015 maximum exactly.
        percentile: f64,
    },
}

/// The `p`-th percentile of `values`, by linear interpolation between order statistics.
///
/// Sorts `values` in place — the slice is taken by `&mut` rather than copied because the caller
/// already owns a scratch pool and a copy of a layer's activations over a sample set is the largest
/// allocation in the whole conversion.
///
/// Rank is `p/100 * (n - 1)` and the result interpolates between the two neighbouring order
/// statistics, which is the convention `NumPy`'s default `percentile` uses; `p = 100` is therefore
/// exactly the maximum and `p = 0` would be exactly the minimum.
///
/// `None` for an empty slice, a non-finite `p`, a `p` outside `[0, 100]`, or any non-finite value —
/// a percentile over a pool containing a `NaN` has no defined position and is refused rather than
/// silently ordered.
#[must_use]
pub fn percentile(values: &mut [f64], p: f64) -> Option<f64> {
    if values.is_empty() || !p.is_finite() || p < 0.0 || p > 100.0 {
        return None;
    }
    if values.iter().any(|v| !v.is_finite()) {
        return None;
    }
    values.sort_unstable_by(f64::total_cmp);
    let n = values.len();
    if n == 1 {
        return Some(values[0]);
    }
    let rank = p / 100.0 * (n - 1) as f64;
    let lo = rank.floor();
    let hi = rank.ceil();
    let (li, hi_i) = (lo as usize, (hi as usize).min(n - 1));
    if li == hi_i {
        return Some(values[li]);
    }
    let frac = rank - lo;
    Some(values[li] + (values[hi_i] - values[li]) * frac)
}

/// How the first layer is driven.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputCoding {
    /// A constant current proportional to the input, injected on every tick.
    ///
    /// Rueckauer et al., Front. Neurosci. 11:682, 2017 §2.5 call this analog input and recommend
    /// it, and the reason is visible in the exponents: the error falls as `1/T` because the only
    /// error left is quantisation. The cost is that the first layer is then a dense matrix-vector
    /// product on EVERY tick — `T` times the `ANN`'s work for that layer — and
    /// [`SpikingMlp::ledger`] charges for it.
    Analog,
    /// Bernoulli spikes at probability equal to the normalised input, one draw per input per tick.
    /// A normalised input above 1 spikes on every tick, which is the same saturation the rest of
    /// the conversion has and is why the input scale `λ_0` is chosen from the inputs themselves.
    ///
    /// Diehl et al., IJCNN 2015. The input layer is then genuinely sparse and genuinely
    /// event-driven, and the price is Monte-Carlo noise: the error falls as `1/sqrt(T)` instead of
    /// `1/T`, so matching the analog coding's accuracy costs roughly the SQUARE of the ticks.
    /// `poisson_input_converges_as_one_over_sqrt_t` fits the exponent and measures **-0.64** on
    /// this module's fixture, against the `-0.5` a pure standard-error argument predicts. The gap
    /// is not explained away here: the sweep is ONE realisation of a random process so the fit is
    /// itself noisy, and at the small end of it the quantisation error that falls as `1/T` is
    /// still comparable to the sampling noise. The test's band is wide for those two reasons, and
    /// the honest reading is "roughly the square root", not "-0.64".
    Poisson {
        /// Seed for the crate's `PCG32` stream. The same seed gives the same spikes on every
        /// platform, and [`SpikingMlp::reset_state`] rewinds to it so that two runs of one network
        /// on one input agree.
        seed: u64,
    },
}

/// What is read out of the last layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Readout {
    /// Spikes counted over the run, divided by ticks and multiplied by `λ_L`.
    ///
    /// This is the `ReLU`-equivalent readout: it is non-negative, it saturates at `λ_L`, and it is
    /// what every accuracy figure in the conversion literature is measured with unless stated
    /// otherwise.
    SpikeCount,
    /// The last layer's accumulated membrane potential, with no spiking in that layer at all.
    ///
    /// Rueckauer et al. suggest this for a classifier's output layer, where only the `argmax`
    /// matters. It estimates the **pre-activation**, not the `ReLU`: the value can be negative, and
    /// for constant analog input it is exact rather than merely convergent, because the membrane
    /// integrates the input without quantising it.
    /// `membrane_readout_recovers_the_pre_activation_exactly` shows it agreeing to 1e-12 at any
    /// number of ticks.
    MembranePotential,
}

/// Everything the conversion needs besides the trained weights.
///
/// [`Config::default`] is a usable starting point and every field is documented with the effect of
/// moving it, because in this module every one of them is a deployment argument.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Config {
    /// Tick length, seconds. The maximum representable rate is `1/dt`, and the conversion error is
    /// bounded by one spike, so halving `dt` at fixed wall-clock latency halves the error.
    pub dt: f64,
    /// Firing threshold, volts. Enters only through `c * v_th / dt`.
    pub v_th: f64,
    /// Membrane capacitance, farads. Enters only through `c * v_th / dt`.
    pub c: f64,
    /// Reset rule. [`Reset::BySubtraction`] unless you are reproducing the 2015 results.
    pub reset: Reset,
    /// How the first layer is driven.
    pub input: InputCoding,
    /// What the last layer reports.
    pub readout: Readout,
    /// How the per-layer scales are chosen.
    pub norm: Norm,
    /// Passed to every neuron's [`SpikingRelu::floor_at_zero`].
    pub floor_at_zero: bool,
}

impl Default for Config {
    /// 1 ms ticks, a 1 V threshold, 1 nF, reset by subtraction, analog input, spike-count readout
    /// and 99.9th-percentile data-based normalisation.
    ///
    /// Every one of those is the choice the 2017 paper argues for, except the tick length, which is
    /// the conversion literature's usual convention rather than a result. The capacitance and
    /// threshold are round numbers: only their product over `dt` matters, and 1 nF × 1 V / 1 ms is
    /// a gain of 1 µA per unit activation, which is a current a real circuit could plausibly carry.
    fn default() -> Self {
        Self {
            dt: 1e-3,
            v_th: 1.0,
            c: 1e-9,
            reset: Reset::BySubtraction,
            input: InputCoding::Analog,
            readout: Readout::SpikeCount,
            norm: Norm::DataBased { percentile: 99.9 },
            floor_at_zero: false,
        }
    }
}

impl Config {
    /// Check the physical parameters.
    ///
    /// # Errors
    ///
    /// [`ConvertError::BadParameter`] naming `dt`, `v_th` or `c`, or
    /// [`ConvertError::BadPercentile`] for a data-based scale outside `(0, 100]`.
    pub fn validate(&self) -> Result<(), ConvertError> {
        for (name, value) in [("dt", self.dt), ("v_th", self.v_th), ("c", self.c)] {
            if !(value.is_finite() && value > 0.0) {
                return Err(ConvertError::BadParameter { name, value });
            }
        }
        if let Norm::DataBased { percentile: p } = self.norm
            && !(p.is_finite() && p > 0.0 && p <= 100.0)
        {
            return Err(ConvertError::BadPercentile { p });
        }
        Ok(())
    }

    /// Amperes per unit of activation, `c * v_th / dt`. The only combination of the three that the
    /// answer depends on.
    #[must_use]
    pub fn gain(&self) -> f64 {
        self.c * self.v_th / self.dt
    }
}

/// One converted layer: normalised weights, and one [`SpikingRelu`] per unit.
#[derive(Debug, Clone, PartialEq)]
pub struct SpikingLayer {
    /// Inputs the layer consumes.
    pub n_in: usize,
    /// Units the layer produces.
    pub n_out: usize,
    /// Normalised weights, row-major, as [`DenseRelu::w`].
    pub w: Vec<f64>,
    /// Normalised biases, applied as a constant current on every tick.
    pub b: Vec<f64>,
    /// One neuron per unit.
    pub neurons: Vec<SpikingRelu>,
    /// Spikes emitted per unit since the last [`SpikingMlp::reset_state`]. This is the numerator of
    /// the rate readout, and it is a `u64` because it is an exact count.
    pub counts: Vec<u64>,
}

/// A converted network: the spiking twin of an [`Mlp`].
///
/// Built by [`SpikingMlp::from_ann`], which normalises a COPY of the source network — the original
/// is never mutated, so the two can be compared afterwards, which is what every test in this module
/// does.
#[derive(Debug, Clone)]
pub struct SpikingMlp {
    /// Layers in forward order.
    pub layers: Vec<SpikingLayer>,
    /// The parameters this network was converted under.
    pub cfg: Config,
    /// The scales `λ_0 .. λ_L` the conversion used. `lambdas[0]` divides the input and
    /// `lambdas[L]` multiplies the readout back into the original network's units.
    pub lambdas: Vec<f64>,
    /// Exact operation counts for the run since the last [`SpikingMlp::reset_state`], for
    /// [`crate::ledger`] and [`crate::crossover`].
    ///
    /// Read [`crate::ledger::Bill::unpriced`] before [`crate::ledger::Bill::total`]: this crate
    /// prices no device completely, on purpose.
    pub ledger: Ledger,
    rng: Rng,
    ticks: u64,
}

impl SpikingMlp {
    /// Convert a trained network.
    ///
    /// `samples` are only read by [`Norm::DataBased`]; pass an empty slice for
    /// [`Norm::ModelBased`]. `ann` is not modified.
    ///
    /// # Errors
    ///
    /// As [`Config::validate`] and [`Mlp::normalise`].
    pub fn from_ann(
        ann: &Mlp,
        samples: &[Vec<f64>],
        cfg: Config,
    ) -> Result<Self, ConvertError> {
        cfg.validate()?;
        let mut normed = ann.clone();
        let lambdas = normed.normalise(cfg.norm, samples)?;
        let layers = normed
            .layers
            .iter()
            .map(|l| {
                let mut n = SpikingRelu::new(cfg.c, cfg.v_th, cfg.reset);
                n.floor_at_zero = cfg.floor_at_zero;
                SpikingLayer {
                    n_in: l.n_in,
                    n_out: l.n_out,
                    w: l.w.clone(),
                    b: l.b.clone(),
                    neurons: vec![n; l.n_out],
                    counts: vec![0; l.n_out],
                }
            })
            .collect();
        let seed = match cfg.input {
            InputCoding::Poisson { seed } => seed,
            InputCoding::Analog => 0,
        };
        Ok(Self { layers, cfg, lambdas, ledger: Ledger::default(), rng: Rng::new(seed), ticks: 0 })
    }

    /// The scale that converts a readout back into the source network's units, `λ_L`.
    ///
    /// The `1.0` fallback is unreachable — `lambdas` always holds one entry per layer plus one —
    /// and is written rather than unwrapped so that no public path can panic.
    #[must_use]
    pub fn output_scale(&self) -> f64 {
        self.lambdas.last().copied().unwrap_or(1.0)
    }

    /// Synapses in the converted network — weights, not biases.
    #[must_use]
    pub fn n_synapses(&self) -> u64 {
        self.layers.iter().map(|l| (l.n_in * l.n_out) as u64).sum()
    }

    /// Spikes emitted by every layer since the last reset.
    #[must_use]
    pub fn total_spikes(&self) -> u64 {
        self.layers.iter().map(|l| l.counts.iter().sum::<u64>()).sum()
    }

    /// Ticks simulated since the last reset.
    #[must_use]
    pub fn ticks_elapsed(&self) -> u64 {
        self.ticks
    }

    /// Clear every membrane, every spike count, the ledger and the tick counter, and rewind the
    /// input encoder's random stream to its seed.
    ///
    /// The rewind is what makes two runs of one network on one input identical, which is a property
    /// a caller comparing two configurations needs and would otherwise have to rebuild the network
    /// to get.
    pub fn reset_state(&mut self) {
        for l in &mut self.layers {
            for n in &mut l.neurons {
                n.reset();
            }
            l.counts.fill(0);
        }
        self.ledger = Ledger::default();
        self.ticks = 0;
        let seed = match self.cfg.input {
            InputCoding::Poisson { seed } => seed,
            InputCoding::Analog => 0,
        };
        self.rng = Rng::new(seed);
    }

    /// Advance the whole network by one tick under input `x`, in the SOURCE network's units.
    ///
    /// Returns the output layer's spikes. Under [`Readout::MembranePotential`] the output layer
    /// never spikes and the returned vector is all `false`.
    ///
    /// # Errors
    ///
    /// [`ConvertError::InputLength`] or [`ConvertError::NonFiniteInput`].
    pub fn tick(&mut self, x: &[f64]) -> Result<Vec<bool>, ConvertError> {
        let n_in = self.layers[0].n_in;
        if x.len() != n_in {
            return Err(ConvertError::InputLength { expected: n_in, got: x.len() });
        }
        for (j, v) in x.iter().enumerate() {
            if !v.is_finite() {
                return Err(ConvertError::NonFiniteInput { index: j });
            }
        }

        let gain = self.cfg.gain();
        let dt = self.cfg.dt;
        let scale_in = self.lambdas[0];
        let mut drive: Vec<f64> = match self.cfg.input {
            InputCoding::Analog => x.iter().map(|v| v / scale_in).collect(),
            InputCoding::Poisson { .. } => x
                .iter()
                .map(|v| {
                    // A Bernoulli draw, not a Poisson one: at most one spike per input per tick is
                    // what the hardware can carry, and the two agree to first order at the rates
                    // conversion uses. The name in the literature is "Poisson input" and is kept.
                    let p = v / scale_in;
                    if self.rng.next_f64() < p { 1.0 } else { 0.0 }
                })
                .collect(),
        };

        let n_layers = self.layers.len();
        let readout = self.cfg.readout;
        for li in 0..n_layers {
            let last = li + 1 == n_layers;
            let silent = last && matches!(readout, Readout::MembranePotential);
            let layer = &mut self.layers[li];
            let active = drive.iter().filter(|v| **v != 0.0).count() as u64;
            // Every active input reaches every unit: the layer is dense, so the delivery count is
            // the product. On a sparse layer this would be a per-synapse count instead, which is
            // why `crate::net` stores a synapse once and this does not pretend to.
            self.ledger.syn_ops += active * layer.n_out as u64;
            self.ledger.syn_fetches += active * layer.n_out as u64;
            if active == 0 {
                self.ledger.neuron_updates_idle += layer.n_out as u64;
            } else {
                self.ledger.neuron_updates_driven += layer.n_out as u64;
            }

            let mut out = vec![false; layer.n_out];
            for u in 0..layer.n_out {
                let row = &layer.w[u * layer.n_in..(u + 1) * layer.n_in];
                let mut z = layer.b[u];
                for j in 0..layer.n_in {
                    if drive[j] != 0.0 {
                        z += row[j] * drive[j];
                    }
                }
                let i = z * gain;
                if silent {
                    layer.neurons[u].accumulate(dt, i);
                } else if layer.neurons[u].step(dt, i) {
                    out[u] = true;
                    layer.counts[u] += 1;
                }
            }
            let fired = out.iter().filter(|b| **b).count() as u64;
            self.ledger.spikes_out += fired;
            if last {
                self.ticks += 1;
                return Ok(out);
            }
            drive = out.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect();
        }
        // Unreachable for a network with at least one layer, which `Mlp::new` guarantees; written
        // as an empty answer rather than an `unreachable!` so that no public path can panic.
        self.ticks += 1;
        Ok(Vec::new())
    }

    /// Reset, run for `ticks` ticks under a constant input, and decode the output layer into the
    /// SOURCE network's units.
    ///
    /// Under [`Readout::SpikeCount`] the result is `counts / ticks * λ_L`, which approximates
    /// [`Mlp::forward`]. Under [`Readout::MembranePotential`] it is `v / (v_th * ticks) * λ_L`,
    /// which approximates the last layer's PRE-activation and may be negative.
    ///
    /// The ledger is counted per run and `reads` is incremented once, because the decode is one
    /// host readout of the output layer's state.
    ///
    /// # Errors
    ///
    /// [`ConvertError::NoTicks`] for `ticks == 0`, or anything [`SpikingMlp::tick`] rejects.
    pub fn run(&mut self, x: &[f64], ticks: u64) -> Result<Vec<f64>, ConvertError> {
        if ticks == 0 {
            return Err(ConvertError::NoTicks);
        }
        self.reset_state();
        for _ in 0..ticks {
            self.tick(x)?;
        }
        self.ledger.reads += 1;
        let scale = self.output_scale();
        let t = ticks as f64;
        let last = &self.layers[self.layers.len() - 1];
        Ok(match self.cfg.readout {
            Readout::SpikeCount => last.counts.iter().map(|&c| c as f64 / t * scale).collect(),
            Readout::MembranePotential => {
                last.neurons.iter().map(|n| n.v / (self.cfg.v_th * t) * scale).collect()
            }
        })
    }
}

/// Conversion error as a function of the number of ticks: the latency/accuracy trade-off, measured.
///
/// `mean_abs_error[k]` is the mean over output units of `|â_snn - â_ann|` after `ticks[k]` ticks, in
/// NORMALISED activation units — that is, divided by `λ_L`, so the numbers are comparable across
/// networks and a value of 0.01 means "one part in a hundred of the layer's dynamic range".
#[derive(Debug, Clone, PartialEq)]
pub struct ErrorCurve {
    /// Tick counts, in the order they were measured.
    pub ticks: Vec<u64>,
    /// Mean absolute error at each tick count, in normalised activation units, same length as
    /// [`ErrorCurve::ticks`].
    pub mean_abs_error: Vec<f64>,
}

impl ErrorCurve {
    /// The exponent `p` in `error ~ T^p`, by ordinary least squares on `ln(error)` against
    /// `ln(T)`.
    ///
    /// The theory says `p = -1` for [`InputCoding::Analog`] with [`Reset::BySubtraction`], because
    /// the residual error is one quantisation step out of `T`; and `p = -1/2` for
    /// [`InputCoding::Poisson`], because the error there is the standard error of a mean over `T`
    /// draws. Both are asserted in this module's tests against a measured fit: **-1.02** for
    /// analog input and **-0.64** for Poisson, the second being a noisy estimate of `-0.5` from a
    /// single realisation rather than a contradiction of it.
    ///
    /// `None` when fewer than two points have a strictly positive, finite error — a perfect run
    /// gives `ln(0)` and is dropped rather than fitted, and a fit through one point is not a fit.
    #[must_use]
    pub fn fit_exponent(&self) -> Option<f64> {
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for (t, e) in self.ticks.iter().zip(self.mean_abs_error.iter()) {
            if *t > 0 && e.is_finite() && *e > 0.0 {
                xs.push((*t as f64).ln());
                ys.push(e.ln());
            }
        }
        if xs.len() < 2 {
            return None;
        }
        let n = xs.len() as f64;
        let mx = xs.iter().sum::<f64>() / n;
        let my = ys.iter().sum::<f64>() / n;
        let mut num = 0.0;
        let mut den = 0.0;
        for k in 0..xs.len() {
            num += (xs[k] - mx) * (ys[k] - my);
            den += (xs[k] - mx) * (xs[k] - mx);
        }
        if den == 0.0 {
            return None;
        }
        Some(num / den)
    }

    /// The error measured at exactly `ticks`, if that point is in the curve.
    #[must_use]
    pub fn at(&self, ticks: u64) -> Option<f64> {
        self.ticks.iter().position(|&t| t == ticks).map(|k| self.mean_abs_error[k])
    }
}

/// Measure a converted network against its source over a sweep of tick counts.
///
/// `ann` must be the ORIGINAL, un-normalised network — [`SpikingMlp::run`] returns its answer in
/// the source's units, so comparing against a normalised copy would compare two different scales
/// and report an error of `λ_L - 1` times the answer.
///
/// # Errors
///
/// [`ConvertError::NoTicks`] if any entry of `ticks` is zero, or anything [`SpikingMlp::run`] and
/// [`Mlp::forward`] reject.
pub fn error_vs_ticks(
    snn: &mut SpikingMlp,
    ann: &Mlp,
    x: &[f64],
    ticks: &[u64],
) -> Result<ErrorCurve, ConvertError> {
    let want = ann.forward(x)?;
    let scale = snn.output_scale();
    let mut out = ErrorCurve { ticks: Vec::new(), mean_abs_error: Vec::new() };
    for &t in ticks {
        let got = snn.run(x, t)?;
        let mut acc = 0.0;
        for k in 0..want.len() {
            acc += (got[k] - want[k]).abs();
        }
        out.ticks.push(t);
        out.mean_abs_error.push(acc / want.len() as f64 / scale);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::{
        Config, ConvertError, DenseRelu, ErrorCurve, InputCoding, Mlp, Norm, Readout, Reset,
        SpikingMlp, SpikingRelu, error_vs_ticks, percentile,
    };
    use crate::crossover::Verdict;
    use crate::neuron::Neuron;
    use crate::rng::Rng;

    /// A deterministic test network: `n_in -> n_hidden -> n_out`, weights from the crate's own
    /// seeded stream so that every figure quoted in a doc here is reproducible from the seed.
    fn net(seed: u64, n_in: usize, n_hidden: usize, n_out: usize) -> Mlp {
        let mut r = Rng::new(seed);
        let mut mk = |a: usize, b: usize, bias: f64| {
            let w: Vec<f64> = (0..a * b).map(|_| r.next_f64() * 2.0 - 1.0).collect();
            let bs: Vec<f64> = (0..b).map(|_| bias + r.next_f64() * 0.1).collect();
            DenseRelu::new(a, b, w, bs).expect("shapes match by construction")
        };
        let l1 = mk(n_in, n_hidden, 0.3);
        let l2 = mk(n_hidden, n_out, 0.3);
        Mlp::new(vec![l1, l2]).expect("layers chain by construction")
    }

    fn samples(seed: u64, n: usize, dim: usize) -> Vec<Vec<f64>> {
        let mut r = Rng::new(seed);
        (0..n).map(|_| (0..dim).map(|_| r.next_f64()).collect()).collect()
    }

    /// (a) The closed form the whole conversion rests on. An `IF` neuron under constant current
    /// fires at exactly `I / (C V_th)`, and the simulated count must land within the one spike the
    /// derivation allows.
    #[test]
    fn an_if_neuron_fires_at_exactly_i_over_c_v_th() {
        let dt = 1e-4;
        for &z in &[0.07, 0.19, 0.31, 0.5, 0.83] {
            let proto = SpikingRelu::new(2e-9, 0.75, Reset::BySubtraction);
            let i = z * proto.gain(dt);
            let want_hz = proto.rate(i).expect("positive current");
            // The closed form itself, independent of the simulation: r = I/(C V_th) = z/dt.
            assert!(
                (want_hz - z / dt).abs() / (z / dt) < 1e-12,
                "closed form {want_hz} vs z/dt {}",
                z / dt
            );
            let mut n = proto;
            let ticks = 20_000u64;
            let mut spikes = 0u64;
            for _ in 0..ticks {
                if n.step(dt, i) {
                    spikes += 1;
                }
            }
            // The integer statement first: the derivation says `N = floor(T z)` in exact
            // arithmetic. The simulation accumulates `z * v_th` volts `T` times, and those `T`
            // roundings can cost ONE spike when `T z` lands within rounding of an integer — which
            // z = 0.07 at 20,000 ticks does, `T z = 1400.0000000000002`, and the first draft of
            // this test failed on exactly that. One spike of slack, named, rather than a tolerance
            // wide enough to hide a real error.
            let want_n = (ticks as f64 * z).floor() as u64;
            assert!(
                spikes.abs_diff(want_n) <= 1,
                "z {z}: {spikes} spikes vs floor(T z) = {want_n}"
            );
            let got_hz = spikes as f64 / (ticks as f64 * dt);
            let bound = 2.0 / (ticks as f64 * dt);
            assert!(
                (got_hz - want_hz).abs() <= bound,
                "z {z}: simulated {got_hz} Hz vs closed form {want_hz} Hz, bound {bound}"
            );
            // And the relative error is below one part in a thousand at this many ticks.
            assert!((got_hz - want_hz).abs() / want_hz < 1e-3, "z {z}: relative error too large");
        }
    }

    /// The exact-integer version of the same statement: with a dyadic activation the accumulation
    /// is exact in binary floating point, so the count is `T z` with no tolerance at all.
    #[test]
    fn a_dyadic_activation_gives_exactly_t_times_z_spikes() {
        let dt = 1e-3;
        for &z in &[0.5, 0.25, 0.75, 0.125, 0.375] {
            let mut n = SpikingRelu::new(1e-9, 1.0, Reset::BySubtraction);
            let i = z * n.gain(dt);
            let ticks = 4096u64;
            let mut spikes = 0u64;
            for _ in 0..ticks {
                if n.step(dt, i) {
                    spikes += 1;
                }
            }
            let want = (ticks as f64 * z) as u64;
            assert_eq!(spikes, want, "z {z}: {spikes} spikes, exact answer {want}");
        }
    }

    /// `Reset::spikes_in` is a closed form and must agree with the neuron it describes, for both
    /// reset rules, at activations chosen so that floating-point accumulation is unambiguous.
    #[test]
    fn spikes_in_matches_the_simulated_neuron_exactly() {
        let dt = 1e-3;
        // Reciprocals far from an integer, so that no accumulation lands within rounding of the
        // threshold. 1/0.37 = 2.70, 1/0.61 = 1.64, 1/0.29 = 3.45, 1/0.17 = 5.88, 1/0.83 = 1.20.
        for &z in &[0.37, 0.61, 0.29, 0.17, 0.83] {
            for reset in [Reset::BySubtraction, Reset::ToZero] {
                let mut n = SpikingRelu::new(1e-9, 1.0, reset);
                let i = z * n.gain(dt);
                let ticks = 1000u64;
                let mut spikes = 0u64;
                for _ in 0..ticks {
                    if n.step(dt, i) {
                        spikes += 1;
                    }
                }
                let want = reset.spikes_in(z, ticks).expect("finite activation");
                let diff = spikes.abs_diff(want);
                assert!(diff <= 1, "{reset} at z {z}: simulated {spikes}, closed form {want}");
            }
        }
    }

    /// (c) part one: reset-to-zero's rate is `1/ceil(1/z)` and that is a BIAS. Checked against the
    /// closed form, and then checked that doubling the run does not reduce it.
    #[test]
    fn reset_to_zero_converges_to_one_over_ceil_one_over_z_not_to_z() {
        let dt = 1e-3;
        for &z in &[0.37f64, 0.61, 0.29, 0.83] {
            let k = (1.0 / z).ceil();
            let want = 1.0 / k;
            let mut errs = Vec::new();
            for &ticks in &[2000u64, 4000, 8000] {
                let mut n = SpikingRelu::new(1e-9, 1.0, Reset::ToZero);
                let i = z * n.gain(dt);
                let mut spikes = 0u64;
                for _ in 0..ticks {
                    if n.step(dt, i) {
                        spikes += 1;
                    }
                }
                let rate = spikes as f64 / ticks as f64;
                assert!(
                    (rate - want).abs() < 2.0 / ticks as f64,
                    "z {z}: rate {rate} vs closed form 1/{k} = {want}"
                );
                errs.push((rate - z).abs());
            }
            // The gap to the ReLU's own value is a floor: four times the ticks, the same error.
            let floor = (want - z).abs();
            // At least 3% of the dynamic range, at every z tried, and 33% at z = 0.83. Three
            // percent is not a rounding artefact: it is 30x the 1/T error that reset-by-
            // subtraction has left at 2,000 ticks, and it never gets smaller.
            assert!(floor > 0.03, "z {z} was chosen badly: the bias is only {floor}");
            for e in &errs {
                assert!(
                    (e - floor).abs() < 0.01,
                    "z {z}: error {e} moved away from the bias floor {floor}"
                );
            }
        }
    }

    /// (c) part two, on a whole network and with a number attached. Same weights, same input, same
    /// tick count; the only difference is two lines inside `SpikingRelu::step`.
    #[test]
    fn reset_by_subtraction_beats_reset_to_zero_and_here_is_by_how_much() {
        let ann = net(11, 12, 40, 16);
        let data = samples(12, 64, 12);
        let x = data[0].clone();
        let want = ann.forward(&x).expect("finite input");

        let measure = |reset: Reset, ticks: u64| -> f64 {
            let cfg = Config { reset, norm: Norm::DataBased { percentile: 100.0 }, ..Config::default() };
            let mut snn = SpikingMlp::from_ann(&ann, &data, cfg).expect("convertible");
            let scale = snn.output_scale();
            let got = snn.run(&x, ticks).expect("positive ticks");
            let mut acc = 0.0;
            for k in 0..want.len() {
                acc += (got[k] - want[k]).abs();
            }
            acc / want.len() as f64 / scale
        };

        let sub = measure(Reset::BySubtraction, 512);
        let zero = measure(Reset::ToZero, 512);
        assert!(
            sub * 20.0 < zero,
            "reset-by-subtraction {sub:.6} was not 20x better than reset-to-zero {zero:.6}"
        );
        // And the reset-to-zero error is a floor: four times the ticks leaves it essentially where
        // it was, while the subtraction error falls with T.
        let zero_long = measure(Reset::ToZero, 2048);
        let sub_long = measure(Reset::BySubtraction, 2048);
        assert!(
            zero_long > 0.5 * zero,
            "reset-to-zero {zero:.6} -> {zero_long:.6} behaved like a 1/T error, not a bias"
        );
        assert!(
            sub_long < 0.6 * sub,
            "reset-by-subtraction {sub:.6} -> {sub_long:.6} did not fall with the tick count"
        );
    }

    /// (b) The headline trade-off: analog input and reset by subtraction give an error that falls
    /// as `T^-1`. The exponent is fitted, not assumed, and the band is set around the derivation's
    /// `-1` rather than around whatever the code happens to produce.
    #[test]
    fn the_conversion_error_falls_as_one_over_t() {
        let ann = net(21, 16, 48, 24);
        let data = samples(22, 96, 16);
        let x = data[3].clone();
        let cfg = Config { norm: Norm::DataBased { percentile: 100.0 }, ..Config::default() };
        let mut snn = SpikingMlp::from_ann(&ann, &data, cfg).expect("convertible");
        let ticks: Vec<u64> = vec![16, 32, 64, 128, 256, 512, 1024, 2048];
        let curve = error_vs_ticks(&mut snn, &ann, &x, &ticks).expect("finite");
        let p = curve.fit_exponent().expect("eight points with positive error");
        assert!(
            (-1.15..=-0.85).contains(&p),
            "error ~ T^{p}, which is not the 1/T the derivation predicts; curve {:?}",
            curve.mean_abs_error
        );
        // Monotone within a factor: every doubling has to help.
        let first = curve.at(16).expect("measured");
        let last = curve.at(2048).expect("measured");
        assert!(last * 40.0 < first, "error fell only from {first} to {last} over 128x the ticks");
    }

    /// The same sweep with Poisson input, whose error is a standard error over `T` draws and so
    /// falls as `T^-1/2`. Stated as its own test because the two exponents are the whole argument
    /// for analog input coding, and because a library that only measured the good case would be
    /// advertising rather than reporting.
    #[test]
    fn poisson_input_converges_as_one_over_sqrt_t() {
        let ann = net(31, 16, 48, 24);
        let data = samples(32, 96, 16);
        let x = data[5].clone();
        let cfg = Config {
            input: InputCoding::Poisson { seed: 7 },
            norm: Norm::DataBased { percentile: 100.0 },
            ..Config::default()
        };
        let mut snn = SpikingMlp::from_ann(&ann, &data, cfg).expect("convertible");
        let ticks: Vec<u64> = vec![64, 128, 256, 512, 1024, 2048, 4096];
        let curve = error_vs_ticks(&mut snn, &ann, &x, &ticks).expect("finite");
        let p = curve.fit_exponent().expect("seven points with positive error");
        assert!(
            (-0.85..=-0.25).contains(&p),
            "Poisson error ~ T^{p}, which is neither the 1/sqrt(T) predicted nor close to it; \
             curve {:?}",
            curve.mean_abs_error
        );

        // And it is worse than analog at the same tick count, which is the deployment consequence.
        let cfg_a = Config { norm: Norm::DataBased { percentile: 100.0 }, ..Config::default() };
        let mut analog = SpikingMlp::from_ann(&ann, &data, cfg_a).expect("convertible");
        let curve_a = error_vs_ticks(&mut analog, &ann, &x, &[4096]).expect("finite");
        let e_p = curve.at(4096).expect("measured");
        let e_a = curve_a.at(4096).expect("measured");
        assert!(e_a < e_p, "analog {e_a:.6} was not better than Poisson {e_p:.6} at 4096 ticks");
    }

    /// (d) Normalisation is a change of units, not a change of function. `ReLU` is positively
    /// homogeneous, so scaling every layer and rescaling the output has to return the same numbers.
    #[test]
    fn normalisation_preserves_the_function_in_the_rate_limit() {
        let ann = net(41, 10, 32, 8);
        let data = samples(42, 50, 10);
        let want = ann.forward(&data[7]).expect("finite");

        for norm in [Norm::ModelBased { input_max: 1.0 }, Norm::DataBased { percentile: 99.0 }] {
            let mut normed = ann.clone();
            let lam = normed.normalise(norm, &data).expect("normalisable");
            let scaled_in: Vec<f64> = data[7].iter().map(|v| v / lam[0]).collect();
            let got = normed.forward(&scaled_in).expect("finite");
            let l_out = lam[lam.len() - 1];
            for k in 0..want.len() {
                let rescaled = got[k] * l_out;
                let denom = want[k].abs().max(1e-12);
                assert!(
                    (rescaled - want[k]).abs() / denom < 1e-12,
                    "unit {k}: {rescaled} vs {} under {norm:?}",
                    want[k]
                );
            }
        }
    }

    /// (d) continued: the scale-and-rescale round trip. Exact to the bit when the scales are powers
    /// of two, and to one part in `1e-14` per weight otherwise — the second figure is a measured
    /// floating-point bound and is stated rather than assumed.
    #[test]
    fn scaling_and_unscaling_round_trips() {
        let ann = net(51, 6, 10, 4);

        // Powers of two: multiplication and division are exact, so this is a bitwise identity.
        let mut dyadic = ann.clone();
        let lam2 = vec![2.0, 4.0, 8.0];
        dyadic.apply_scales(&lam2).expect("three scales for two layers");
        assert_ne!(dyadic, ann, "applying scales did not change the weights");
        dyadic.undo_scales(&lam2).expect("same scales");
        assert_eq!(dyadic, ann, "a dyadic round trip was not bit-exact");

        // A general scale: exact in real arithmetic, one ulp per operation in floating point.
        let mut general = ann.clone();
        let lam = vec![0.37, 12.9, 3.3333];
        general.apply_scales(&lam).expect("three scales");
        general.undo_scales(&lam).expect("same scales");
        for (l, (a, b)) in general.layers.iter().zip(ann.layers.iter()).enumerate() {
            for k in 0..a.w.len() {
                let d = (a.w[k] - b.w[k]).abs() / b.w[k].abs().max(1e-300);
                assert!(d < 1e-14, "layer {l} weight {k} round-tripped to a relative error of {d}");
            }
        }
    }

    /// (e) `ReLU`'s flat region is the one part of the conversion with no error at all. A
    /// non-positive drive produces not few spikes but none, and the readout is exactly `0.0`.
    #[test]
    fn a_negative_pre_activation_produces_exactly_zero_spikes() {
        // Closed form first, at both resets and over a very long run.
        for reset in [Reset::BySubtraction, Reset::ToZero] {
            assert_eq!(reset.spikes_in(-0.3, 1_000_000), Some(0));
            assert_eq!(reset.spikes_in(0.0, 1_000_000), Some(0));
            assert_eq!(reset.rate_limit(-1.0), Some(0.0));
        }
        // Then the neuron itself, with and without the zero floor.
        for floor in [false, true] {
            let mut n = SpikingRelu::new(1e-9, 1.0, Reset::BySubtraction);
            n.floor_at_zero = floor;
            let i = -0.4 * n.gain(1e-3);
            assert_eq!(n.rate(i), Some(0.0));
            let mut spikes = 0u64;
            for _ in 0..200_000 {
                if n.step(1e-3, i) {
                    spikes += 1;
                }
            }
            assert_eq!(spikes, 0, "a negative current produced {spikes} spikes (floor {floor})");
        }
        // Then a whole layer whose every unit is driven negative: the readout is exactly zero, not
        // approximately zero, and `ReLU` agrees exactly.
        let w = vec![-1.0; 4 * 3];
        let b = vec![-0.5; 3];
        let layer = DenseRelu::new(4, 3, w, b).expect("shapes match");
        let ann = Mlp::new(vec![layer]).expect("one layer");
        let x = vec![0.9, 0.8, 0.7, 0.6];
        assert_eq!(ann.forward(&x).expect("finite"), vec![0.0, 0.0, 0.0]);
        let cfg = Config { norm: Norm::ModelBased { input_max: 1.0 }, ..Config::default() };
        // The model-based scale needs a positive bound; a layer with no positive activation at all
        // is refused by name rather than divided by.
        let err = SpikingMlp::from_ann(&ann, &[], cfg).expect_err("degenerate layer");
        assert!(matches!(err, ConvertError::DegenerateScale { layer: 1, .. }), "{err}");

        // With a live unit beside the dead ones, the dead ones still read exactly zero.
        let mut w2 = vec![-1.0; 4 * 3];
        w2[0] = 1.0;
        w2[1] = 1.0;
        w2[2] = 1.0;
        w2[3] = 1.0;
        let mut b2 = vec![-0.5; 3];
        b2[0] = 0.1;
        let layer2 = DenseRelu::new(4, 3, w2, b2).expect("shapes match");
        let ann2 = Mlp::new(vec![layer2]).expect("one layer");
        let mut snn = SpikingMlp::from_ann(&ann2, &[], cfg).expect("one live unit");
        let got = snn.run(&x, 500).expect("positive ticks");
        assert!(got[0] > 0.0, "the live unit did not fire");
        assert_eq!(got[1], 0.0, "a dead unit reported {}", got[1]);
        assert_eq!(got[2], 0.0, "a dead unit reported {}", got[2]);
    }

    /// Why Rueckauer et al. replaced the maximum with a percentile, in exact spike counts. One
    /// sample in 201 is a hundred times the rest; under maximum normalisation a typical unit emits
    /// nothing at all in 100 ticks.
    #[test]
    fn the_percentile_normaliser_is_why_robust_normalisation_exists() {
        let layer = DenseRelu::new(1, 1, vec![1.0], vec![0.0]).expect("shapes match");
        let ann = Mlp::new(vec![layer]).expect("one layer");
        let mut data: Vec<Vec<f64>> = (0..200).map(|k| vec![0.1 + 0.9 * (k as f64 / 199.0)]).collect();
        data.push(vec![100.0]);

        let lam_max = ann.scales(Norm::DataBased { percentile: 100.0 }, &data).expect("scalable");
        let lam_p99 = ann.scales(Norm::DataBased { percentile: 99.0 }, &data).expect("scalable");
        assert!((lam_max[1] - 100.0).abs() < 1e-12, "maximum scale was {}", lam_max[1]);
        assert!(lam_p99[1] < 1.2, "99th-percentile scale was {}, not near the bulk", lam_p99[1]);

        // A typical activation of 0.5, under each scale, in exact spike counts over 100 ticks.
        let typical = 0.5;
        let z_max = typical / lam_max[1];
        let z_p99 = typical / lam_p99[1];
        let n_max = Reset::BySubtraction.spikes_in(z_max, 100).expect("finite");
        let n_p99 = Reset::BySubtraction.spikes_in(z_p99, 100).expect("finite");
        assert_eq!(n_max, 0, "maximum normalisation resolved a typical unit in 100 ticks");
        assert!(n_p99 >= 40, "percentile normalisation emitted only {n_p99} spikes in 100 ticks");

        // The cost, stated: the outlier now saturates. Its true activation is 100 and the unit can
        // report at most `lam_p99`, so robust normalisation clips it.
        assert!(100.0 / lam_p99[1] > 1.0, "the outlier did not saturate, so there is no trade-off");
    }

    /// `percentile(_, 100)` must be the maximum exactly, which is what makes `DataBased { 100.0 }`
    /// a faithful reproduction of the 2015 method rather than an approximation of it.
    #[test]
    fn the_hundredth_percentile_is_exactly_the_maximum() {
        let mut v = vec![3.0, -1.0, 7.5, 0.0, 2.25];
        let p100 = percentile(&mut v.clone(), 100.0).expect("non-empty");
        let p0 = percentile(&mut v.clone(), 0.0).expect("non-empty");
        let max = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let min = v.iter().copied().fold(f64::INFINITY, f64::min);
        assert_eq!(p100, max);
        assert_eq!(p0, min);
        // The median of five sorted values is the middle one, exactly.
        assert_eq!(percentile(&mut v, 50.0).expect("non-empty"), 2.25);
        // Refusals.
        assert!(percentile(&mut [], 50.0).is_none());
        assert!(percentile(&mut [1.0, f64::NAN], 50.0).is_none());
        assert!(percentile(&mut [1.0, 2.0], 101.0).is_none());
        assert!(percentile(&mut [1.0, 2.0], f64::NAN).is_none());
    }

    /// The model-based scale claims to be an upper bound. Check it against a few hundred random
    /// inputs, which is the only claim it makes that can fail.
    #[test]
    fn model_based_normalisation_bounds_every_observed_activation() {
        let ann = net(61, 8, 20, 6);
        let lam = ann.scales(Norm::ModelBased { input_max: 1.0 }, &[]).expect("scalable");
        let data = samples(62, 300, 8);
        for s in &data {
            let acts = ann.activations(s).expect("finite");
            for (l, a) in acts.iter().enumerate() {
                for (u, v) in a.iter().enumerate() {
                    assert!(
                        *v <= lam[l + 1] * (1.0 + 1e-12),
                        "layer {l} unit {u} activated {v}, above its model-based bound {}",
                        lam[l + 1]
                    );
                }
            }
        }
    }

    /// The bound is also loose, and the looseness is the reason the data-based method won in Diehl
    /// et al.'s own comparison. Measured, with the ratio reported in the failure message.
    #[test]
    fn the_model_based_bound_is_loose_and_that_is_the_cost() {
        let ann = net(71, 8, 20, 6);
        let data = samples(72, 200, 8);
        let model = ann.scales(Norm::ModelBased { input_max: 1.0 }, &[]).expect("scalable");
        let observed = ann.scales(Norm::DataBased { percentile: 100.0 }, &data).expect("scalable");
        let ratio = model[2] / observed[2];
        assert!(
            ratio > 1.5,
            "the model-based bound was only {ratio}x the largest observed activation, which would \
             make the two methods interchangeable and they are not"
        );
    }

    /// Only the combination `c * v_th / dt` can change the answer. Doubling `c` and `v_th` together
    /// with the current that follows from them must leave the spike train bit-identical — these are
    /// powers of two, so this is an exact statement, not a tolerance.
    #[test]
    fn only_the_ratio_c_v_th_over_dt_changes_the_spike_train() {
        let run = |c: f64, v_th: f64, dt: f64| -> Vec<u64> {
            let mut times = Vec::new();
            let mut n = SpikingRelu::new(c, v_th, Reset::BySubtraction);
            let i = 0.37 * n.gain(dt);
            for k in 0..500u64 {
                if n.step(dt, i) {
                    times.push(k);
                }
            }
            times
        };
        let a = run(1e-9, 1.0, 1e-3);
        let b = run(2e-9, 2.0, 1e-3);
        let c = run(0.5e-9, 1.0, 0.5e-3);
        assert_eq!(a, b, "doubling c and v_th together changed the spike train");
        assert_eq!(a, c, "halving c and dt together changed the spike train");
    }

    /// The membrane readout integrates without quantising, so for a single layer under constant
    /// analog input it recovers the pre-activation exactly — at any tick count, including one.
    #[test]
    fn membrane_readout_recovers_the_pre_activation_exactly() {
        let ann = net(81, 8, 12, 5);
        let one = Mlp::new(vec![ann.layers[0].clone()]).expect("one layer");
        let data = samples(82, 40, 8);
        let x = data[2].clone();
        let want = one.layers[0].pre_activation(&x).expect("finite");

        let cfg = Config {
            readout: Readout::MembranePotential,
            norm: Norm::DataBased { percentile: 100.0 },
            ..Config::default()
        };
        let mut snn = SpikingMlp::from_ann(&one, &data, cfg).expect("convertible");
        for &ticks in &[1u64, 7, 100, 999] {
            let got = snn.run(&x, ticks).expect("positive ticks");
            for k in 0..want.len() {
                let denom = want[k].abs().max(1e-9);
                assert!(
                    (got[k] - want[k]).abs() / denom < 1e-12,
                    "{ticks} ticks, unit {k}: membrane readout {} vs pre-activation {}",
                    got[k],
                    want[k]
                );
            }
        }
        // And it reports NEGATIVE pre-activations, which the spike-count readout cannot. That is
        // the whole reason it exists, and the whole reason it is not a `ReLU`.
        assert!(want.iter().any(|v| *v < 0.0), "the fixture has no negative unit to check");
    }

    /// Two layers, a real tick budget, and a stated accuracy. The number is measured, not
    /// predicted, and the caveat beside it is that a deeper stack compounds: every layer's
    /// quantisation feeds the next one's input.
    #[test]
    fn a_two_layer_conversion_reproduces_its_source_to_a_stated_error() {
        let ann = net(91, 20, 64, 10);
        let data = samples(92, 128, 20);
        let cfg = Config { norm: Norm::DataBased { percentile: 99.9 }, ..Config::default() };
        let mut snn = SpikingMlp::from_ann(&ann, &data, cfg).expect("convertible");
        let scale = snn.output_scale();
        let mut worst = 0.0f64;
        for x in data.iter().take(8) {
            let want = ann.forward(x).expect("finite");
            let got = snn.run(x, 4096).expect("positive ticks");
            for k in 0..want.len() {
                let e = (got[k] - want[k]).abs() / scale;
                if e > worst {
                    worst = e;
                }
            }
        }
        assert!(
            worst < 0.02,
            "worst-unit error over eight inputs was {worst} of the layer's dynamic range"
        );
    }

    /// The bill for the latency. A converted network does `T` membrane updates per neuron per
    /// inference, and with analog input its first layer is a dense product on every tick. Every
    /// published crossover threshold is below two spikes per synapse per inference; this workload
    /// measures **91.8**, fifty times the most permissive of the three, and the test exists to say
    /// so rather than to pass.
    #[test]
    fn a_converted_network_at_realistic_latency_is_refuted_by_every_published_crossover() {
        let ann = net(101, 20, 64, 10);
        let data = samples(102, 64, 20);
        let mut snn = SpikingMlp::from_ann(&ann, &data, Config::default()).expect("convertible");
        let ticks = 128u64;
        snn.run(&data[0], ticks).expect("positive ticks");
        let sps = snn
            .ledger
            .spikes_per_synapse(snn.n_synapses(), 1)
            .expect("synapses and one inference");
        assert!(sps > 50.0, "spikes per synapse was only {sps}, which the assertion did not expect");
        let verdicts = snn.ledger.crossover_verdicts(snn.n_synapses(), 1).expect("countable");
        for (name, v) in &verdicts {
            assert_eq!(*v, Verdict::Refuted, "{name} did not refute {sps} spikes per synapse");
        }
        // The ledger still refuses to price it, for the reason `crate::ledger` gives.
        let bill = snn.ledger.bill(&crate::ledger::TRUENORTH_2014);
        assert!(bill.total.is_none());
        assert!(bill.unpriced.contains(&"synapse memory fetch"));
    }

    /// Determinism: the same seed gives the same spikes, and `reset_state` rewinds the stream so a
    /// second run of the same network on the same input agrees with the first.
    #[test]
    fn a_poisson_conversion_is_reproducible_from_its_seed() {
        let ann = net(111, 10, 24, 6);
        let data = samples(112, 40, 10);
        let cfg = Config { input: InputCoding::Poisson { seed: 12345 }, ..Config::default() };
        let mut a = SpikingMlp::from_ann(&ann, &data, cfg).expect("convertible");
        let mut b = SpikingMlp::from_ann(&ann, &data, cfg).expect("convertible");
        let ra = a.run(&data[1], 300).expect("positive ticks");
        let rb = b.run(&data[1], 300).expect("positive ticks");
        assert_eq!(ra, rb, "two networks from one seed disagreed");
        let ra2 = a.run(&data[1], 300).expect("positive ticks");
        assert_eq!(ra, ra2, "a second run of one network disagreed with its first");
        assert_eq!(a.total_spikes(), b.total_spikes());
        assert_eq!(a.ticks_elapsed(), 300);
    }

    /// `EXACT_OVER_GAPS` is a promise that `crate::sim` enforces. Demonstrate it rather than assert
    /// the constant: one step of a hundred ticks with no input must leave the same bits as a
    /// hundred steps of one.
    #[test]
    fn the_converted_neuron_is_exact_over_gaps() {
        const { assert!(SpikingRelu::EXACT_OVER_GAPS) };
        let dt = 1e-3;
        for floor in [false, true] {
            let mut a = SpikingRelu::new(1e-9, 1.0, Reset::BySubtraction);
            a.floor_at_zero = floor;
            a.v = 0.4;
            let mut b = a;
            for _ in 0..100 {
                assert!(!a.step(dt, 0.0));
            }
            assert!(!b.step(dt * 100.0, 0.0));
            assert_eq!(a.potential(), b.potential(), "a quiet gap moved the membrane (floor {floor})");
        }
    }

    /// Every refusal, by name. A conversion that accepted any of these would produce a network that
    /// runs and reports nonsense.
    #[test]
    fn broken_inputs_are_refused_by_name() {
        assert_eq!(
            DenseRelu::new(0, 3, vec![], vec![0.0; 3]),
            Err(ConvertError::EmptyLayer { n_in: 0, n_out: 3 })
        );
        assert_eq!(
            DenseRelu::new(2, 3, vec![0.0; 5], vec![0.0; 3]),
            Err(ConvertError::WeightCount { expected: 6, got: 5 })
        );
        assert_eq!(
            DenseRelu::new(2, 3, vec![0.0; 6], vec![0.0; 2]),
            Err(ConvertError::BiasCount { expected: 3, got: 2 })
        );
        let mut w = vec![0.1; 6];
        w[4] = f64::NAN;
        assert_eq!(
            DenseRelu::new(2, 3, w, vec![0.0; 3]),
            Err(ConvertError::NonFiniteWeight { unit: 2, input: 0 })
        );
        assert_eq!(
            DenseRelu::new(2, 3, vec![0.1; 6], vec![0.0, f64::INFINITY, 0.0]),
            Err(ConvertError::NonFiniteBias { unit: 1 })
        );

        let l = DenseRelu::new(2, 3, vec![0.1; 6], vec![0.0; 3]).expect("valid");
        assert_eq!(
            l.forward(&[1.0]),
            Err(ConvertError::InputLength { expected: 2, got: 1 })
        );
        assert_eq!(
            l.forward(&[1.0, f64::NAN]),
            Err(ConvertError::NonFiniteInput { index: 1 })
        );

        let bad = Mlp::new(vec![
            l.clone(),
            DenseRelu::new(4, 2, vec![0.1; 8], vec![0.0; 2]).expect("valid"),
        ]);
        assert_eq!(bad, Err(ConvertError::Disconnected { layer: 1, expected: 4, got: 3 }));
        assert_eq!(Mlp::new(vec![]), Err(ConvertError::NoLayers));

        let ann = Mlp::new(vec![l]).expect("one layer");
        assert_eq!(
            ann.scales(Norm::DataBased { percentile: 99.0 }, &[]),
            Err(ConvertError::NoSamples)
        );
        assert_eq!(
            ann.scales(Norm::DataBased { percentile: 0.0 }, &[vec![1.0, 1.0]]),
            Err(ConvertError::BadPercentile { p: 0.0 })
        );
        assert_eq!(
            ann.clone().apply_scales(&[1.0, 2.0, 3.0]),
            Err(ConvertError::ScaleCount { expected: 2, got: 3 })
        );
        assert_eq!(
            ann.clone().apply_scales(&[1.0, 0.0]),
            Err(ConvertError::DegenerateScale { layer: 1, lambda: 0.0 })
        );

        for (name, cfg) in [
            ("dt", Config { dt: 0.0, ..Config::default() }),
            ("v_th", Config { v_th: -1.0, ..Config::default() }),
            ("c", Config { c: f64::NAN, ..Config::default() }),
        ] {
            let e = cfg.validate().expect_err("bad parameter");
            assert!(matches!(e, ConvertError::BadParameter { name: n, .. } if n == name), "{e}");
        }

        let good = Config { norm: Norm::ModelBased { input_max: 1.0 }, ..Config::default() };
        let ann2 = net(121, 4, 6, 3);
        let mut snn = SpikingMlp::from_ann(&ann2, &[], good).expect("convertible");
        assert_eq!(snn.run(&[0.5; 4], 0), Err(ConvertError::NoTicks));
        assert_eq!(
            snn.run(&[0.5; 3], 10),
            Err(ConvertError::InputLength { expected: 4, got: 3 })
        );
        assert_eq!(
            snn.run(&[0.5, 0.5, 0.5, f64::NAN], 10),
            Err(ConvertError::NonFiniteInput { index: 3 })
        );
    }

    /// A curve with too little to fit refuses rather than returning a slope through one point.
    #[test]
    fn an_unfittable_error_curve_has_no_exponent() {
        let empty = ErrorCurve { ticks: vec![], mean_abs_error: vec![] };
        assert!(empty.fit_exponent().is_none());
        let perfect = ErrorCurve { ticks: vec![10, 20, 40], mean_abs_error: vec![0.0, 0.0, 0.0] };
        assert!(perfect.fit_exponent().is_none(), "ln(0) was fitted rather than dropped");
        let one = ErrorCurve { ticks: vec![10, 20], mean_abs_error: vec![0.1, 0.0] };
        assert!(one.fit_exponent().is_none(), "a fit through one point was returned");
        let two = ErrorCurve { ticks: vec![10, 100], mean_abs_error: vec![0.1, 0.01] };
        let p = two.fit_exponent().expect("two usable points");
        assert!((p + 1.0).abs() < 1e-12, "a perfect 1/T curve fitted to {p}");
        assert_eq!(two.at(100), Some(0.01));
        assert_eq!(two.at(50), None);
    }
}
