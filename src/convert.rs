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
//!   the weights alone, propagated forward. Needs no data, and is an upper bound — so nothing
//!   saturates — **for an input that lies in `0..input_max`**, which is the regime Diehl et al.
//!   work in and the one the bound is derived under. It sums the POSITIVE weights times
//!   `input_max`, so a SIGNED input reaches the negative weights too and the bound does not hold:
//!   a two-weight layer `w = [1, -1]` bounds at 1.0 and its `ReLU` returns 2.0 on `x = [1, -1]`,
//!   which `the_model_based_bound_holds_only_for_non_negative_inputs` measures. Every layer past
//!   the first satisfies the precondition for free, because a `ReLU` output cannot be negative;
//!   the first layer's input is the caller's to keep in range. The bound is also hopelessly loose
//!   — it assumes every input simultaneously takes its maximum and hits only the positive weights
//!   — so real activations end up a long way below 1 and need a long `T` to be resolved.
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
//!   `T = 128` with analog input is **91.8 deliveries per synapse per inference**, against
//!   published thresholds of 1.72, 1.38 and 0.35 — fifty times the most permissive of them. The
//!   convention inside that number is stated because it is large: **93% of it is not spikes**, it
//!   is the analog first layer's dense product repeated on every tick (`20 x 64 x 128 = 163,840`
//!   of `176,190` deliveries), and the published thresholds are stated in spikes. So the same
//!   network is measured again under [`InputCoding::Poisson`], where every delivery IS a spike:
//!   **38.3 spikes per synapse, and `Refuted` by all three thresholds just the same**. The verdict
//!   is convention-independent; the number is a factor of 2.4 of choice. A `T`-tick conversion
//!   does `T` membrane updates per neuron per inference to replace one multiply-add. The energy
//!   case for conversion has to come from somewhere other than the operation count, and this
//!   module reports the count rather than arguing about it.
//! - **It does not get event-driven simulation.** [`SpikingRelu::EXACT_OVER_GAPS`] is **false**,
//!   so [`crate::sim::Sim::new`] refuses [`crate::sim::Mode::EventDriven`] for a converted
//!   network, and the refusal is correct: reset by subtraction leaves a supra-threshold residual
//!   whenever a unit is driven past one spike per tick, and such a unit fires on QUIET ticks. That
//!   is the property event-driven simulation needs and the one reset by subtraction removes — the
//!   same two lines that are worth a factor of 49 on accuracy cost the whole of the quiet-tick
//!   skip. Reset to zero would keep it, at that factor of 49. The trade is real and it is not
//!   discussed in either source paper.
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
    /// A converted layer's per-unit state array did not hold one entry per unit.
    ///
    /// Unreachable through [`SpikingMlp::from_ann`], and reachable by assigning to the public
    /// [`SpikingLayer::neurons`] or [`SpikingLayer::counts`]. Checked rather than indexed past,
    /// because the alternative is an out-of-bounds panic inside [`SpikingMlp::tick`].
    StateCount {
        /// Which layer, indexed from the input side.
        layer: usize,
        /// The field that disagreed, by name: `neurons` or `counts`.
        field: &'static str,
        /// Entries required: the layer's unit count.
        expected: usize,
        /// Entries present.
        got: usize,
    },
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
            Self::StateCount { layer, field, expected, got } => {
                write!(f, "layer {layer} has {got} entries in {field} for {expected} units")
            }
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
    /// input it matters, and the amount is a closed form rather than a caveat. Under `n` ticks at
    /// `-a` followed by `+z`, the first spike lands on tick
    ///
    /// ```text
    /// n + ceil(1/z)                 with the clamp
    /// n + ceil((1 + n a)/z)         without it
    /// ```
    ///
    /// so the clamp removes exactly `n a / z` ticks of delay — the time the unit spends climbing
    /// out of the well a long negative stretch dug. At `n = 50`, `a = 0.25`, `z = 0.25` that is
    /// **50 ticks against a first spike at 54**, and at `n = 4000`, `a = 0.5`, `z = 0.125` it is
    /// the difference between firing on tick 4008 and firing on tick 20,008.
    /// `the_zero_floor_removes_a_negative_well_and_here_is_the_closed_form` checks both forms as
    /// integer equalities in a dyadic frame.
    ///
    /// Rueckauer et al. discuss the same effect for the input layer; this implementation exposes
    /// the choice rather than picking one, because the clamp makes the unit no longer a linear
    /// integrator and that is a real cost: a clamped unit forgets charge it was owed, so it is no
    /// longer true that the spike count is `floor(T z)` for a drive that changes sign.
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
    /// **False**, because of [`Reset::BySubtraction`] — the one mechanism this model exists for.
    ///
    /// The trait's contract is that one step of `k * dt` with zero input leaves exactly the state
    /// that `k` steps of `dt` leave. Subtracting the threshold instead of clearing the membrane
    /// keeps the overshoot, and the one-spike-per-tick cap means a unit driven harder than one
    /// spike per tick accumulates it: after the reset `v` can still be at or above `v_th`. Such a
    /// neuron FIRES ON A QUIET TICK, so a gap crossed in one jump emits one spike where the same
    /// gap crossed tick by tick emits `floor(v / v_th)` of them. Measured, from `v = 3.5 v_th`
    /// across 100 quiet ticks: tick by tick, 3 spikes and `v = 0.5 v_th`; in one jump, 1 spike and
    /// `v = 2.5 v_th`. `a_supra_threshold_residual_fires_on_quiet_ticks` is that measurement.
    ///
    /// That state is not exotic — it is what saturation looks like, and every normalisation in this
    /// module except the model-based bound admits it by construction. On this module's own robust-
    /// normalisation fixture, 50 ticks of the outlier sample leave the membrane at **4972.7 V
    /// against a 1 V threshold**, which
    /// `a_saturating_conversion_reaches_the_state_that_breaks_the_gap_property` reaches through the
    /// public constructor with no field poked.
    ///
    /// The consequence is a capability this module does not have: [`crate::sim::Sim::new`] refuses
    /// [`crate::sim::Mode::EventDriven`] for a converted network, and that refusal is correct
    /// rather than cautious. Declaring `true` here would have let `crate::sim::Sim` jump the quiet
    /// intervals and silently drop the spikes the residual owes —
    /// `an_event_driven_sim_of_a_converted_neuron_is_refused` pins the refusal.
    ///
    /// The constant cannot be narrowed to the reset rule that is safe, because it is a property of
    /// the TYPE and [`SpikingRelu::reset`] is a field. [`Reset::ToZero`] alone would satisfy the
    /// contract — it discards the overshoot, so `v < v_th` always holds after a spike — and that is
    /// the same charge-discarding bias the module doc prices at a factor of 49.
    /// [`crate::neuron::Lif`] and [`crate::neuron::IntegrateAndFire`] reset to a `v_reset` strictly
    /// below threshold and therefore genuinely cannot fire on a quiet tick; this is the crate's
    /// only reset-by-subtraction neuron and the only one that can.
    const EXACT_OVER_GAPS: bool = false;

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
    /// This is Diehl et al.'s model-based scale, and it is an upper bound **only for an input in
    /// `0..input_max`**. The sum runs over `w > 0` alone, so an input component that is NEGATIVE
    /// drives the negative weights upward and can exceed the answer: `w = [1, -1]`, `b = 0`,
    /// `input_max = 1` returns 1.0, while `ReLU(W x)` at `x = [1, -1]` is 2.0. Both components
    /// satisfy `x <= input_max`, and the bound is still 2x too small —
    /// `the_model_based_bound_holds_only_for_non_negative_inputs` is that measurement. The
    /// precondition is free for every layer but the first, whose input is a `ReLU` output and
    /// therefore non-negative; the network's own input is the caller's to keep in range.
    ///
    /// Within that precondition it is a genuine bound — nothing saturates — and it is loose by
    /// construction, because no real input hits every positive weight at once.
    /// `model_based_normalisation_bounds_every_observed_activation` checks the bound holds, over
    /// three values of `input_max`, and `the_model_based_bound_is_loose_and_that_is_the_cost`
    /// measures how loose.
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
    /// layer by layer from an input bounded by `input_max`. Needs no data, and cannot saturate for
    /// any input in `0..input_max` — see [`DenseRelu::max_possible_activation`] for what a NEGATIVE
    /// input component does to that guarantee.
    ModelBased {
        /// Bound on every input component, in the input's own units: the contract is
        /// `0 <= x[j] <= input_max`, not `|x[j]| <= input_max`.
        ///
        /// For pixel data scaled to `0..1` this is `1.0`, and for raw 8-bit pixels it is `255.0`.
        /// Supplying a bound the data exceeds, or data that goes below zero, breaks the guarantee
        /// silently — the network saturates and reports a smaller number with no error — which is
        /// why it is a required parameter rather than a default. Nothing here can check it,
        /// because [`Norm::ModelBased`] by definition never sees the data.
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
    /// `poisson_input_converges_as_one_over_sqrt_t` fits the exponent over SIX SEEDS of this
    /// module's fixture rather than one, because the sweep is a realisation of a random process
    /// and a single fit measures the seed as much as the process. The six are **-0.641, -0.611,
    /// -0.503, -0.473, -0.485, -0.510**, with a median of **-0.506** against the `-0.5` a standard
    /// error argument predicts. Seed 7 is the extreme of the six, and earlier versions of this
    /// module quoted it alone as "-0.64 against a predicted -0.5" with a paragraph explaining the
    /// gap; the sweep costs less than the paragraph and leaves nothing to explain. What remains
    /// true of any one seed is that the band is wide — at the small end of the sweep the
    /// quantisation error that falls as `1/T` is still comparable to the sampling noise — which is
    /// why the per-seed assertion is loose and the assertion on the median is not.
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
    /// it is exact rather than merely convergent — at any tick count, including one — **when the
    /// layer's own input is a constant analog current**, because then the membrane integrates that
    /// input without quantising it.
    /// `membrane_readout_recovers_the_pre_activation_exactly` shows that agreeing to 1e-12 on a
    /// SINGLE-layer network, which is the case where the qualifier is satisfied.
    ///
    /// In the use it is recommended for — the output layer of a deeper stack — it is NOT exact
    /// with respect to the source network, and the reason is one layer upstream: the output
    /// layer's input is the previous layer's spike train, which carries that layer's activation to
    /// one part in `T`. The readout is exact with respect to the train it is given and inherits
    /// every quantisation error made before it. What it removes is the output layer's OWN
    /// quantisation, which is one of `L` such errors and the only one a longer run does not shrink
    /// relative to.
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

    /// Re-check everything the public fields could have broken since [`SpikingMlp::from_ann`].
    ///
    /// [`SpikingMlp::layers`], [`SpikingMlp::lambdas`] and [`SpikingMlp::cfg`] are public, so the
    /// invariants the constructor established are not invariants of the type — `layers.clear()`
    /// and `lambdas[0] = 0.0` are both one assignment away, and the first used to panic inside
    /// [`SpikingMlp::tick`] while the second produced an infinite drive and a plausible finite
    /// answer. Both are named refusals now. The cost is a handful of length comparisons per tick,
    /// against a dense matrix-vector product in the same tick.
    fn check_structure(&self) -> Result<(), ConvertError> {
        if self.layers.is_empty() {
            return Err(ConvertError::NoLayers);
        }
        for (name, value) in [("dt", self.cfg.dt), ("v_th", self.cfg.v_th), ("c", self.cfg.c)] {
            if !(value.is_finite() && value > 0.0) {
                return Err(ConvertError::BadParameter { name, value });
            }
        }
        check_scales(&self.lambdas, self.layers.len())?;
        for (li, l) in self.layers.iter().enumerate() {
            if l.n_in == 0 || l.n_out == 0 {
                return Err(ConvertError::EmptyLayer { n_in: l.n_in, n_out: l.n_out });
            }
            if li > 0 && l.n_in != self.layers[li - 1].n_out {
                return Err(ConvertError::Disconnected {
                    layer: li,
                    expected: l.n_in,
                    got: self.layers[li - 1].n_out,
                });
            }
            if l.w.len() != l.n_in * l.n_out {
                return Err(ConvertError::WeightCount {
                    expected: l.n_in * l.n_out,
                    got: l.w.len(),
                });
            }
            if l.b.len() != l.n_out {
                return Err(ConvertError::BiasCount { expected: l.n_out, got: l.b.len() });
            }
            for (field, got) in [("neurons", l.neurons.len()), ("counts", l.counts.len())] {
                if got != l.n_out {
                    return Err(ConvertError::StateCount {
                        layer: li,
                        field,
                        expected: l.n_out,
                        got,
                    });
                }
            }
        }
        Ok(())
    }

    /// Advance the whole network by one tick under input `x`, in the SOURCE network's units.
    ///
    /// Returns the output layer's spikes. Under [`Readout::MembranePotential`] the output layer
    /// never spikes and the returned vector is all `false`.
    ///
    /// # Errors
    ///
    /// [`ConvertError::InputLength`] or [`ConvertError::NonFiniteInput`] for the input, and —
    /// because [`SpikingMlp::layers`], [`SpikingMlp::lambdas`] and [`SpikingMlp::cfg`] are public
    /// and can be changed after conversion — [`ConvertError::NoLayers`],
    /// [`ConvertError::BadParameter`], [`ConvertError::ScaleCount`],
    /// [`ConvertError::DegenerateScale`], [`ConvertError::EmptyLayer`],
    /// [`ConvertError::WeightCount`], [`ConvertError::BiasCount`], [`ConvertError::StateCount`] or
    /// [`ConvertError::Disconnected`] for a network those assignments have made unrunnable.
    pub fn tick(&mut self, x: &[f64]) -> Result<Vec<bool>, ConvertError> {
        self.check_structure()?;
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
            // FETCH PER DELIVERY, stated rather than implied: this device model reads the weight
            // from memory every time it is used, so `syn_fetches == syn_ops` here by construction
            // and the ratio this crate exists to expose is 1.000 for this workload. It is not
            // 1.000 in general — `crate::ledger::Ledger::syn_fetches` is a separate counter
            // precisely because a design that caches or batches a row reads it fewer times — and
            // a converted dense layer under analog input is the case where a cache would help
            // most, since the same row is re-read on every one of `T` ticks.
            // `the_ledger_records_which_updates_were_idle_and_which_were_driven` asserts the
            // identity so that the convention cannot change without a test changing with it.
            self.ledger.syn_fetches += active * layer.n_out as u64;
            // IDLE means "nothing arrived on a synapse this tick", not "did nothing": a layer with
            // a non-zero bias integrates that bias on every tick and can spike from it alone, and
            // those updates are counted here. The convention is the one event-driven hardware
            // uses — a tick with no incoming event is a tick the hardware would have skipped — and
            // `the_bias_only_layer_is_counted_idle_and_still_spikes` measures the case where the
            // two readings differ.
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
    /// [`ConvertError::NoTicks`] for `ticks == 0`, or anything [`SpikingMlp::tick`] rejects —
    /// which includes every way the public fields can be made inconsistent after conversion.
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
    /// analog input, and for Poisson a **median of -0.506 over six seeds** — the six spread from
    /// -0.641 to -0.473, which is the width a single realisation of a random process has and the
    /// reason the sweep is over six of them.
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
        // And a dyadic arm, where the accumulation is exact in binary and the closed form is
        // therefore an EQUALITY. One spike of slack is the right allowance for the rounding it was
        // introduced for, and it is also wide enough to hide an off-by-one in the count itself —
        // `ticks / k` against `ticks / k + 1` — which is a different defect and needs a case with
        // no slack at all.
        for &z in &[0.5f64, 0.25, 0.75, 0.125, 0.375] {
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
                assert_eq!(
                    spikes,
                    reset.spikes_in(z, ticks).expect("finite activation"),
                    "{reset} at a dyadic z {z}: the closed form is exact here, not within one"
                );
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
            // Through the PUBLIC method, not an inline recomputation of it: the closed form in the
            // doc and the closed form in the code have to meet somewhere, and this is where.
            let want = Reset::ToZero.rate_limit(z).expect("finite activation");
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

    /// `Reset::rate_limit` is the asymptote the module's entire reset argument is made of, and it
    /// is a public method with a closed form. Checked against hand-computed values first — a
    /// twelve-line table nothing in the code could have produced — and then against the simulated
    /// neuron it describes.
    #[test]
    fn rate_limit_is_the_asymptote_the_neuron_actually_reaches() {
        // Hand-computed. `ceil(1/0.99) = 2`, so reset-to-zero converges to one half against an
        // activation of 0.99: the 0.49 error the 2017 paper removed.
        for &(z, sub, zero) in &[
            (0.99f64, 0.99f64, 0.5f64),
            (0.5, 0.5, 0.5),
            (0.34, 0.34, 1.0 / 3.0),
            (0.25, 0.25, 0.25),
            (0.2, 0.2, 0.2),
            (1.0, 1.0, 1.0),
        ] {
            let got_sub = Reset::BySubtraction.rate_limit(z).expect("finite activation");
            let got_zero = Reset::ToZero.rate_limit(z).expect("finite activation");
            assert!(
                (got_sub - sub).abs() < 1e-15,
                "by subtraction at z {z}: {got_sub}, hand-computed {sub}"
            );
            assert!(
                (got_zero - zero).abs() < 1e-15,
                "to zero at z {z}: {got_zero}, hand-computed {zero}"
            );
        }
        // The claim the module makes and a mutation to `Self::ToZero => z` would erase: the two
        // rules DISAGREE, and by up to half the dynamic range.
        let worst = Reset::BySubtraction.rate_limit(0.99).expect("finite")
            - Reset::ToZero.rate_limit(0.99).expect("finite");
        assert!((worst - 0.49).abs() < 1e-15, "the two rules differ by {worst}, not 0.49");
        // Above one spike per tick, both saturate at the cap, and by subtraction saturates at
        // exactly 1 rather than at `z` — which a mutation to `z` alone would also erase.
        for &z in &[1.5f64, 4.0, 1e9] {
            assert_eq!(Reset::BySubtraction.rate_limit(z), Some(1.0), "z {z} exceeded the cap");
            assert_eq!(Reset::ToZero.rate_limit(z), Some(1.0), "z {z} exceeded the cap");
        }
        // The quiet region and the broken caller.
        for reset in [Reset::BySubtraction, Reset::ToZero] {
            assert_eq!(reset.rate_limit(0.0), Some(0.0));
            assert_eq!(reset.rate_limit(-1e-9), Some(0.0));
            assert_eq!(reset.rate_limit(f64::NAN), None);
            assert_eq!(reset.rate_limit(f64::INFINITY), None);
        }
        // And the asymptote is one the neuron reaches: 200,000 ticks of simulation against the
        // closed form, for both rules, at activations whose reciprocals are far from an integer.
        let dt = 1e-3;
        for &z in &[0.99f64, 0.37, 0.61, 0.29] {
            for reset in [Reset::BySubtraction, Reset::ToZero] {
                let mut n = SpikingRelu::new(1e-9, 1.0, reset);
                let i = z * n.gain(dt);
                let ticks = 200_000u64;
                let mut spikes = 0u64;
                for _ in 0..ticks {
                    if n.step(dt, i) {
                        spikes += 1;
                    }
                }
                let rate = spikes as f64 / ticks as f64;
                let want = reset.rate_limit(z).expect("finite activation");
                assert!(
                    (rate - want).abs() < 2.0 / ticks as f64,
                    "{reset} at z {z}: simulated {rate}, rate_limit said {want}"
                );
            }
        }
    }

    /// `floor_at_zero` is not a no-op, and the caveat that said its effect was unverified has a
    /// closed form. Under piecewise-constant input — `n_neg` ticks at `-a`, then `+z` — a clamped
    /// integrator first fires on tick `n_neg + ceil(1/z)` and an unclamped one on tick
    /// `n_neg + ceil((1 + n_neg*a)/z)`. The flag's benefit is the difference, `n_neg*a/z` ticks,
    /// and that is the number this test measures.
    ///
    /// Run in a dyadic frame — `c = 2^-30` F, `v_th = 1` V, `dt = 2^-10` s, so `c*v_th/dt` is
    /// `2^-20` exactly and every accumulation is exact in binary — which is why these are integer
    /// equalities and not tolerances.
    #[test]
    fn the_zero_floor_removes_a_negative_well_and_here_is_the_closed_form() {
        let (c, v_th, dt) = (2f64.powi(-30), 1.0f64, 2f64.powi(-10));
        let first_spike = |floor: bool, n_neg: u64, a: f64, z: f64| -> Option<u64> {
            let mut n = SpikingRelu::new(c, v_th, Reset::BySubtraction);
            n.floor_at_zero = floor;
            let g = n.gain(dt);
            for k in 0..200_000u64 {
                let drive = if k < n_neg { -a } else { z };
                if n.step(dt, drive * g) {
                    return Some(k + 1);
                }
            }
            None
        };
        for &(n_neg, a, z) in &[(50u64, 0.25f64, 0.25f64), (1000, 0.0625, 0.0625), (16, 0.5, 0.125)]
        {
            let nn = n_neg as f64;
            let want_clamped = n_neg + (1.0 / z).ceil() as u64;
            let want_linear = n_neg + ((1.0 + nn * a) / z).ceil() as u64;
            assert_eq!(
                first_spike(true, n_neg, a, z),
                Some(want_clamped),
                "clamped, n_neg {n_neg} a {a} z {z}"
            );
            assert_eq!(
                first_spike(false, n_neg, a, z),
                Some(want_linear),
                "linear, n_neg {n_neg} a {a} z {z}"
            );
            // The advertised benefit, as a number: the well is `n_neg*a` thresholds deep and takes
            // `n_neg*a/z` ticks to climb out of.
            assert_eq!(
                want_linear - want_clamped,
                (nn * a / z) as u64,
                "the flag saved {} ticks, not the n_neg*a/z the doc claims",
                want_linear - want_clamped
            );
            assert!(want_linear > want_clamped, "the flag made no difference at all");
        }
        // Deep enough and the unclamped unit never fires inside the run while the clamped one
        // does: the flag is the difference between a working unit and a silent one.
        let (n_neg, a, z) = (4000u64, 0.5f64, 0.125f64);
        assert_eq!(first_spike(true, n_neg, a, z), Some(n_neg + 8));
        assert_eq!(first_spike(false, n_neg, a, z), Some(n_neg + 16008));
        // For CONSTANT non-negative input the flag changes nothing, which is the other half of the
        // doc's claim and the reason the module leaves it off by default.
        for &z in &[0.125f64, 0.375, 0.75] {
            assert_eq!(first_spike(true, 0, 0.0, z), first_spike(false, 0, 0.0, z), "z {z}");
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
        let ticks: Vec<u64> = vec![64, 128, 256, 512, 1024, 2048, 4096];

        // SIX SEEDS, not one. The exponent is fitted through a single realisation of a random
        // process, so one seed measures the seed as much as the process: the six come out at
        // -0.641, -0.611, -0.503, -0.473, -0.485 and -0.510. Seed 7 — the figure this module used
        // to quote, twice — is the most extreme of them, and the median is the prediction.
        let seeds = [7u64, 1, 2, 3, 99, 12345];
        let mut exponents = Vec::new();
        for seed in seeds {
            let cfg_s = Config {
                input: InputCoding::Poisson { seed },
                norm: Norm::DataBased { percentile: 100.0 },
                ..Config::default()
            };
            let mut s = SpikingMlp::from_ann(&ann, &data, cfg_s).expect("convertible");
            let c = error_vs_ticks(&mut s, &ann, &x, &ticks).expect("finite");
            let e = c.fit_exponent().expect("seven points with positive error");
            assert!(
                (-0.85..=-0.25).contains(&e),
                "seed {seed}: Poisson error ~ T^{e}, which is neither the 1/sqrt(T) predicted nor \
                 close to it; curve {:?}",
                c.mean_abs_error
            );
            exponents.push(e);
        }
        let mut sorted = exponents.clone();
        sorted.sort_by(f64::total_cmp);
        let median = 0.5 * (sorted[2] + sorted[3]);
        assert!(
            (-0.56..=-0.44).contains(&median),
            "the median exponent over six seeds is {median}, not the -0.5 a standard-error \
             argument predicts; the six were {exponents:?}"
        );
        // The single-seed figure was an outlier, and saying so is the point of the sweep.
        assert_eq!(
            exponents[0], sorted[0],
            "seed 7 is no longer the most extreme of the six, so the doc's reading of it is stale"
        );
        let spread = sorted[5] - sorted[0];
        assert!(
            spread > 0.1,
            "the six seeds spread only {spread}, so one of them would have been representative \
             after all"
        );

        // Seed 7 is still the arm the comparison against analog uses, so that the two numbers in
        // the doc come from one run.
        let cfg7 = Config {
            input: InputCoding::Poisson { seed: 7 },
            norm: Norm::DataBased { percentile: 100.0 },
            ..Config::default()
        };
        let mut snn = SpikingMlp::from_ann(&ann, &data, cfg7).expect("convertible");
        let curve = error_vs_ticks(&mut snn, &ann, &x, &ticks).expect("finite");

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

        // A general scale: exact in real arithmetic, one ulp per operation in floating point. The
        // bound asserted is FOUR ULP — `4 * f64::EPSILON` of relative error, 8.9e-16 — because the
        // worst error this fixture produces is 0.95 ulp and a round trip is two multiplications.
        // The first draft of this test allowed 1e-14, which is 45 ulp: a bound 47x looser than
        // anything the arithmetic can produce asserts nothing about the arithmetic.
        let mut general = ann.clone();
        let lam = vec![0.37, 12.9, 3.3333];
        general.apply_scales(&lam).expect("three scales");
        general.undo_scales(&lam).expect("same scales");
        let tol = 4.0 * f64::EPSILON;
        let mut worst = 0.0f64;
        for (l, (a, b)) in general.layers.iter().zip(ann.layers.iter()).enumerate() {
            for (k, (x, y)) in a.w.iter().zip(b.w.iter()).enumerate() {
                let d = (x - y).abs() / y.abs().max(1e-300);
                assert!(
                    d < tol,
                    "layer {l} weight {k} round-tripped to {} ulp of relative error",
                    d / f64::EPSILON
                );
                worst = worst.max(d);
            }
            // The biases too: they are divided and multiplied by a different factor from the
            // weights, so a round trip that is exact for one is not evidence about the other.
            for (k, (x, y)) in a.b.iter().zip(b.b.iter()).enumerate() {
                let d = (x - y).abs() / y.abs().max(1e-300);
                assert!(
                    d < tol,
                    "layer {l} bias {k} round-tripped to {} ulp of relative error",
                    d / f64::EPSILON
                );
                worst = worst.max(d);
            }
        }
        // And the bound is not vacuous from the other side: at a non-dyadic scale the round trip
        // really does lose a bit, so a test demanding bit-equality here would fail honestly.
        assert!(
            worst > 0.5 * f64::EPSILON,
            "the worst round-trip error was {worst}, so far below the asserted 4 ulp that the \
             assertion could not fail"
        );
        assert!(worst < tol, "worst {} ulp", worst / f64::EPSILON);
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
        // Then the neuron itself, with and without the zero floor. The two arms agree on the spike
        // count and on nothing else — the clamped unit sits at exactly zero and the linear one has
        // dug a well 80,000 thresholds deep — so the loop is two measurements, not one twice.
        let mut wells = Vec::new();
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
            wells.push(n.potential());
        }
        assert_eq!(wells[1], 0.0, "the clamped unit did not sit at exactly zero");
        assert!(
            (wells[0] + 80_000.0).abs() < 1e-6,
            "the linear unit dug a well of {} thresholds, not the 80,000 the drive delivers",
            wells[0]
        );
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
        // Three values of `input_max`, with the inputs scaled to match, because at `input_max = 1`
        // alone the parameter could be ignored entirely and every assertion would still hold.
        for &input_max in &[0.5f64, 1.0, 3.0] {
            let lam = ann.scales(Norm::ModelBased { input_max }, &[]).expect("scalable");
            assert_eq!(lam[0], input_max, "the input scale is the declared bound");
            let data = samples(62, 300, 8);
            for s in &data {
                let scaled: Vec<f64> = s.iter().map(|v| v * input_max).collect();
                let acts = ann.activations(&scaled).expect("finite");
                for (l, a) in acts.iter().enumerate() {
                    for (u, v) in a.iter().enumerate() {
                        assert!(
                            *v <= lam[l + 1] * (1.0 + 1e-12),
                            "input_max {input_max}, layer {l} unit {u} activated {v}, above its \
                             model-based bound {}",
                            lam[l + 1]
                        );
                    }
                }
            }
        }
        // The bound MOVES with `input_max`, and by a hand-computed amount. One unit, three inputs,
        // one negative weight that the bound must ignore: `b + input_max * (0.5 + 0.25)`.
        let one = DenseRelu::new(3, 1, vec![0.5, 0.25, -3.0], vec![0.75]).expect("shapes match");
        for &(input_max, want) in &[(1.0f64, 1.5f64), (2.0, 2.25), (0.5, 1.125), (4.0, 3.75)] {
            let got = one.max_possible_activation(input_max);
            assert!(
                (got - want).abs() < 1e-15,
                "input_max {input_max}: bound {got}, hand-computed {want}"
            );
        }
    }

    /// The model-based bound is an upper bound only where Diehl et al. derived it: on a
    /// NON-NEGATIVE input. This is the cost of the precondition, measured, because the module used
    /// to say the method "cannot saturate" with no qualifier at all.
    #[test]
    fn the_model_based_bound_holds_only_for_non_negative_inputs() {
        let layer = DenseRelu::new(2, 1, vec![1.0, -1.0], vec![0.0]).expect("shapes match");
        let ann = Mlp::new(vec![layer]).expect("one layer");
        let bound = ann.layers[0].max_possible_activation(1.0);
        assert_eq!(bound, 1.0, "the bound sums the positive weights alone");

        // Inside the precondition, the bound holds with room to spare.
        for x in [vec![1.0, 1.0], vec![1.0, 0.0], vec![0.25, 0.75]] {
            let a = ann.forward(&x).expect("finite")[0];
            assert!(a <= bound, "x {x:?} activated {a} against a bound of {bound}");
        }
        // Outside it — a signed component that still satisfies `x <= input_max` — the bound is
        // wrong by a factor of two.
        let signed = vec![1.0, -1.0];
        let truth = ann.forward(&signed).expect("finite")[0];
        assert_eq!(truth, 2.0, "the ReLU's own answer");
        assert!(truth > 2.0 * bound * (1.0 - 1e-12), "the bound was not breached, so there is no \
             precondition to state");

        // And the breach is silent end to end: the converted network saturates at one spike per
        // tick and reports the bound, with no error and no NaN.
        let cfg = Config { norm: Norm::ModelBased { input_max: 1.0 }, ..Config::default() };
        let mut snn = SpikingMlp::from_ann(&ann, &[], cfg).expect("convertible");
        let got = snn.run(&signed, 512).expect("positive ticks");
        assert!((got[0] - bound).abs() < 1e-12, "saturated readout was {}, not {bound}", got[0]);
        assert!(got[0] < 0.51 * truth, "the readout did not saturate, so nothing was lost");
    }

    /// `gain` and `activation` are the two directions of the conversion boundary, and the module
    /// doc advertises both. Round-tripped, pinned to a hand-computed ampere, and checked against
    /// the spike train an activation of 1 is defined to produce.
    #[test]
    fn gain_and_activation_are_inverses_at_the_conversion_boundary() {
        for &(c, v_th, dt) in &[(1e-9f64, 1.0f64, 1e-3f64), (2e-9, 0.75, 1e-4), (5e-12, 0.3, 2e-5)]
        {
            let n = SpikingRelu::new(c, v_th, Reset::BySubtraction);
            for &z in &[0.0f64, 0.07, 0.5, 1.0, 3.25] {
                let i = z * n.gain(dt);
                let back = n.activation(i, dt);
                assert!(
                    (back - z).abs() <= 1e-15 * z.max(1.0),
                    "c {c} v_th {v_th} dt {dt}: activation(gain(z)) = {back}, not {z}"
                );
            }
        }
        // The units, hand-computed: 1 nF x 1 V / 1 ms is 1 microampere per unit of activation, and
        // `activation` is the INVERSE map — a mutation that multiplied instead of dividing would
        // answer 2e-18 here rather than 2.
        let n = SpikingRelu::new(1e-9, 1.0, Reset::BySubtraction);
        let dt = 1e-3;
        assert!((n.gain(dt) - 1e-6).abs() < 1e-21, "gain was {} A, not 1 uA", n.gain(dt));
        assert!((n.activation(2e-6, dt) - 2.0).abs() < 1e-15, "{}", n.activation(2e-6, dt));
        assert!((n.activation(-1e-6, dt) + 1.0).abs() < 1e-15, "the map is signed");
        // And the definition it carries: an activation of 1 is exactly one spike per tick.
        let mut m = n;
        let i = 1.0 * m.gain(dt);
        assert_eq!(m.activation(i, dt), 1.0);
        let mut spikes = 0u64;
        for _ in 0..1000 {
            if m.step(dt, i) {
                spikes += 1;
            }
        }
        assert_eq!(spikes, 1000, "an activation of 1 did not fire on every tick");
    }

    /// The interpolating path through `percentile`. `NumPy`'s linear convention is what the doc
    /// claims, and every figure in this module lands on an INTEGER rank, so the interpolation
    /// itself was checked by nothing. Hand-computed values, each one between two order statistics.
    #[test]
    fn the_percentile_interpolates_between_order_statistics() {
        // rank = p/100 * (n - 1); the answer is v[floor(rank)] + frac * (v[ceil] - v[floor]).
        for (values, p, want) in [
            (vec![0.0, 1.0], 25.0, 0.25),
            (vec![0.0, 1.0], 75.0, 0.75),
            (vec![0.0, 4.0], 75.0, 3.0),
            (vec![1.0, 2.0, 3.0, 4.0, 5.0], 62.5, 3.5),
            (vec![10.0, 0.0, 2.0, 8.0], 50.0, 5.0),
            (vec![-4.0, 4.0], 12.5, -3.0),
            (vec![0.0, 1.0, 2.0, 3.0], 10.0, 0.3),
        ] {
            let got = percentile(&mut values.clone(), p).expect("non-empty");
            assert!(
                (got - want).abs() < 1e-15,
                "percentile({values:?}, {p}) = {got}, hand-computed {want}"
            );
            // Truncating to the lower order statistic — the obvious wrong implementation — would
            // answer the floor instead, and these cases are chosen so that it differs.
            let mut sorted = values.clone();
            sorted.sort_by(f64::total_cmp);
            let rank = p / 100.0 * (sorted.len() - 1) as f64;
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let floor_value = sorted[rank.floor() as usize];
            assert!(
                (got - floor_value).abs() > 1e-9,
                "percentile({values:?}, {p}) landed on the order statistic {floor_value}, so the \
                 interpolation is not being exercised"
            );
        }
        // A percentile below zero is refused, like one above a hundred. `-0.0` is not below zero
        // in IEEE arithmetic and is therefore the minimum, not a refusal.
        assert!(percentile(&mut [1.0, 2.0], -1.0).is_none());
        assert!(percentile(&mut [1.0, 2.0], -1e-300).is_none());
        assert_eq!(percentile(&mut [1.0, 2.0], -0.0), Some(1.0));
        assert_eq!(percentile(&mut [1.0, 2.0], 100.0), Some(2.0));
        // A single value has no pair to interpolate between and is returned as it is.
        assert_eq!(percentile(&mut [7.5], 37.0), Some(7.5));
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

    /// `Config::gain` is the same conversion boundary reached from the conversion's own parameters
    /// instead of from a neuron, and it had no test: every fixture in this module leaves `v_th` at
    /// 1.0, where `c * v_th / dt` and `c / dt` are the same expression and a dropped threshold is
    /// invisible. Moved here, against hand-computed amperes, and then moved end to end.
    #[test]
    fn the_config_gain_carries_v_th_and_every_fixture_left_it_at_one() {
        for &(c, v_th, dt, want) in &[
            (1e-9f64, 1.0f64, 1e-3f64, 1e-6f64),
            (2e-9, 0.75, 1e-3, 1.5e-6),
            (1e-9, 4.0, 1e-3, 4e-6),
            (5e-12, 0.3, 2e-5, 7.5e-8),
        ] {
            let cfg = Config { c, v_th, dt, ..Config::default() };
            let got = cfg.gain();
            assert!(
                (got - want).abs() <= 1e-15 * want,
                "gain at c {c}, v_th {v_th}, dt {dt} is {got} A, hand-computed {want} A"
            );
            // The two boundaries are one boundary: `from_ann` builds its neurons out of `cfg`, so
            // these have to agree to the bit or the network is driven by a different gain from the
            // one the config reports.
            assert_eq!(cfg.gain(), SpikingRelu::new(c, v_th, cfg.reset).gain(dt));
        }

        // End to end, the invariant that makes the threshold a free parameter: the gain scales the
        // drive by exactly the threshold the membrane is compared against, so three networks at
        // three thresholds emit the SAME spike train. It is only true if `v_th` is in the gain.
        let ann = net(181, 6, 12, 4);
        let data = samples(182, 32, 6);
        let x = data[3].clone();
        let base = Config { norm: Norm::DataBased { percentile: 100.0 }, ..Config::default() };
        let mut out = Vec::new();
        for v_th in [0.25f64, 1.0, 4.0] {
            let mut snn =
                SpikingMlp::from_ann(&ann, &data, Config { v_th, ..base }).expect("convertible");
            out.push(snn.run(&x, 256).expect("positive ticks"));
        }
        assert!(out[1].iter().any(|v| *v > 0.0), "the fixture emitted nothing to compare");
        assert_eq!(out[0], out[1], "a quarter-volt threshold changed the answer");
        assert_eq!(out[1], out[2], "a four-volt threshold changed the answer");

        // The membrane readout divides by `v_th * t` for the same reason, and that divisor is
        // equally invisible at `v_th = 1`. At three thresholds it still recovers the
        // pre-activation, which it cannot do if either `v_th` is dropped.
        let one = Mlp::new(vec![ann.layers[0].clone()]).expect("one layer");
        let want = one.layers[0].pre_activation(&x).expect("finite");
        for v_th in [0.25f64, 1.0, 4.0] {
            let cfg = Config { v_th, readout: Readout::MembranePotential, ..base };
            let mut snn = SpikingMlp::from_ann(&one, &data, cfg).expect("convertible");
            let got = snn.run(&x, 64).expect("positive ticks");
            for k in 0..want.len() {
                let denom = want[k].abs().max(1e-9);
                assert!(
                    (got[k] - want[k]).abs() / denom < 1e-12,
                    "v_th {v_th}, unit {k}: membrane readout {} vs pre-activation {}",
                    got[k],
                    want[k]
                );
            }
        }
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

        // WHAT 91.8 IS MADE OF, and why the verdict does not depend on it. The published
        // thresholds are stated in SPIKES per synapse, and under analog input most of `syn_ops` is
        // not spikes at all: the first layer is a dense product on every tick, 20 inputs x 64
        // units x 128 ticks, delivered whether or not anything fired.
        let dense = 20 * 64 * ticks;
        assert_eq!(dense, 163_840);
        assert!(
            snn.ledger.syn_ops > dense,
            "the first layer accounted for every delivery, which cannot be right"
        );
        let carried_by_spikes = snn.ledger.syn_ops - dense;
        assert_eq!(carried_by_spikes, 12_350, "the spike-carried share of syn_ops moved");
        let analog_share = dense as f64 / snn.ledger.syn_ops as f64;
        assert!(
            analog_share > 0.9,
            "only {analog_share:.3} of the deliveries were the analog first layer, so the \
             convention caveat is smaller than stated"
        );

        // The constructive half: Poisson input is genuinely event-driven, so EVERY delivery is a
        // spike and the ratio is convention-independent. It measures 2.4x smaller — and is refuted
        // by all three thresholds just the same, which is what makes the conclusion robust.
        let cfg_p = Config { input: InputCoding::Poisson { seed: 7 }, ..Config::default() };
        let mut poisson = SpikingMlp::from_ann(&ann, &data, cfg_p).expect("convertible");
        poisson.run(&data[0], ticks).expect("positive ticks");
        let sps_p = poisson
            .ledger
            .spikes_per_synapse(poisson.n_synapses(), 1)
            .expect("synapses and one inference");
        assert!(
            (sps_p - 38.27).abs() < 0.01,
            "Poisson spikes per synapse was {sps_p}, not the 38.27 documented"
        );
        assert!(sps_p < 0.5 * sps, "the two codings agreed, so the convention does not matter");
        for (name, v) in &poisson.ledger.crossover_verdicts(poisson.n_synapses(), 1).expect("countable") {
            assert_eq!(*v, Verdict::Refuted, "{name} did not refute {sps_p} spikes per synapse");
        }
    }

    /// `error_vs_ticks` reports in NORMALISED units — divided by `lambda_L` — and every error
    /// figure this module quotes takes its meaning from that divisor. Dropping it leaves the fitted
    /// exponent unchanged, because the fit is scale-invariant, and leaves every monotonicity
    /// assertion unchanged, because they are ratios. So the divisor is pinned here, exactly, on a
    /// fixture whose `lambda_L` is 4.13 and not 1.
    #[test]
    fn the_error_curve_is_normalised_by_the_output_scale() {
        let ann = net(161, 8, 16, 5);
        let data = samples(162, 48, 8);
        let cfg = Config { norm: Norm::DataBased { percentile: 100.0 }, ..Config::default() };
        let mut snn = SpikingMlp::from_ann(&ann, &data, cfg).expect("convertible");
        let scale = snn.output_scale();
        assert!(
            (scale - 1.0).abs() > 1.0,
            "lambda_L is {scale}, too close to 1 for this test to see the divisor at all"
        );

        let x = data[4].clone();
        let sweep = [32u64, 256];
        let curve = error_vs_ticks(&mut snn, &ann, &x, &sweep).expect("finite");
        let want = ann.forward(&x).expect("finite");
        for &t in &sweep {
            let got = snn.run(&x, t).expect("positive ticks");
            let mut acc = 0.0;
            for k in 0..want.len() {
                acc += (got[k] - want[k]).abs();
            }
            let raw = acc / want.len() as f64;
            let reported = curve.at(t).expect("measured");
            // Bit for bit: the same sum, the same divisions, in the same order.
            assert_eq!(
                reported,
                raw / scale,
                "at {t} ticks the curve reported {reported}, not {raw} / {scale}"
            );
            // And it is NOT the same number in the source network's own units, which is what a
            // missing divisor would have produced.
            assert!(
                (reported - raw).abs() > 1e-6,
                "at {t} ticks the normalised and un-normalised errors agree, so the divisor is \
                 invisible to this fixture"
            );
            // The doc's meaning of the number: one part in a hundred of the layer's dynamic
            // range. At 256 ticks the quantisation error is under one spike in 256.
            assert!(reported < 4.0 / t as f64, "{reported} at {t} ticks is above the 1/T bound");
        }
    }

    /// Every other fixture in this module draws its inputs from `Rng::next_f64()` in `0..1`, so
    /// `lambda_0` comes out at 0.999 and the input normalisation is a no-op no assertion could
    /// see. Here the inputs are 8-bit pixel values in `0..255`, `lambda_0` is 254.7, and the
    /// difference between dividing by it and not is the difference between a converted network and
    /// a saturated one.
    #[test]
    fn the_input_scale_is_not_a_no_op_on_data_that_is_not_already_normalised() {
        let ann = net(131, 8, 16, 4);
        let mut r = Rng::new(132);
        let data: Vec<Vec<f64>> =
            (0..64).map(|_| (0..8).map(|_| r.next_f64() * 255.0).collect()).collect();
        let cfg = Config { norm: Norm::DataBased { percentile: 100.0 }, ..Config::default() };
        let x = data[0].clone();
        let want = ann.forward(&x).expect("finite");

        let mut snn = SpikingMlp::from_ann(&ann, &data, cfg).expect("convertible");
        assert!(
            snn.lambdas[0] > 100.0,
            "lambda_0 came out at {}, so the fixture is normalised already and proves nothing",
            snn.lambdas[0]
        );
        let scale = snn.output_scale();
        let worst = |got: &[f64]| -> f64 {
            want.iter().zip(got.iter()).map(|(a, b)| (a - b).abs() / scale).fold(0.0, f64::max)
        };
        let good = worst(&snn.run(&x, 2048).expect("positive ticks"));
        assert!(good < 0.01, "analog conversion of 0..255 data was off by {good}");

        // Poisson coding is where it bites hardest: the drive is a probability, so an input scale
        // of 1 would make every input spike on every tick.
        let cfg_p = Config {
            input: InputCoding::Poisson { seed: 5 },
            norm: Norm::DataBased { percentile: 100.0 },
            ..Config::default()
        };
        let mut poisson = SpikingMlp::from_ann(&ann, &data, cfg_p).expect("convertible");
        let good_p = worst(&poisson.run(&x, 4096).expect("positive ticks"));
        assert!(good_p < 0.02, "Poisson conversion of 0..255 data was off by {good_p}");

        // And the counterfactual, through the public field rather than through a mutation: set the
        // input scale to 1 and the same network saturates.
        let mut broken = SpikingMlp::from_ann(&ann, &data, cfg).expect("convertible");
        broken.lambdas[0] = 1.0;
        let bad = worst(&broken.run(&x, 2048).expect("positive ticks"));
        assert!(
            bad > 100.0 * good,
            "an input scale of 1 cost only {bad} against {good}, so lambda_0 is not doing the work \
             the module says it does"
        );
    }

    /// The ledger's idle/driven split is the whole event-driven argument, and `bill()` only ever
    /// consumes the SUM of the two — so the sense of the split is not constrained by any figure the
    /// module quotes. Constrain it here, from both sides, on a layer that is idle on every tick and
    /// on the same layer driven on every tick.
    #[test]
    fn the_ledger_records_which_updates_were_idle_and_which_were_driven() {
        // One layer, so that "the layer was driven" and "the network was driven" are the same
        // statement and the two arms differ in nothing but the input.
        let layer = DenseRelu::new(3, 4, vec![0.6; 12], vec![-0.05; 4]).expect("shapes match");
        let ann = Mlp::new(vec![layer]).expect("one layer");
        let data: Vec<Vec<f64>> = (0..8).map(|k| vec![0.2 + 0.1 * k as f64; 3]).collect();
        let cfg = Config { norm: Norm::DataBased { percentile: 100.0 }, ..Config::default() };
        let ticks = 64u64;

        // Idle: a zero input delivers nothing, and the negative bias keeps the unit silent.
        let mut idle = SpikingMlp::from_ann(&ann, &data, cfg).expect("convertible");
        idle.run(&[0.0; 3], ticks).expect("positive ticks");
        assert_eq!(idle.ledger.syn_ops, 0, "a zero input delivered across a synapse");
        assert_eq!(idle.ledger.syn_fetches, 0);
        assert_eq!(idle.ledger.spikes_out, 0, "a negative bias fired");
        assert_eq!(idle.ledger.neuron_updates_driven, 0, "an idle tick was recorded as driven");
        assert_eq!(idle.ledger.neuron_updates_idle, ticks * 4);
        assert_eq!(idle.ledger.idle_fraction(), Some(1.0));

        // Driven: every input component is non-zero, so every unit receives on every tick.
        let mut driven = SpikingMlp::from_ann(&ann, &data, cfg).expect("convertible");
        driven.run(&[0.5; 3], ticks).expect("positive ticks");
        assert_eq!(driven.ledger.neuron_updates_idle, 0, "a driven tick was recorded as idle");
        assert_eq!(driven.ledger.neuron_updates_driven, ticks * 4);
        assert_eq!(driven.ledger.idle_fraction(), Some(0.0));
        // Dense delivery, exactly: three active inputs reaching four units, every tick.
        assert_eq!(driven.ledger.syn_ops, ticks * 3 * 4);
        // One fetch per delivery is this module's device model, and it is an identity rather than
        // an estimate. `crate::ledger` keeps the two counters apart for hardware where it is not.
        assert_eq!(driven.ledger.syn_fetches, driven.ledger.syn_ops);

        // Two layers, and the exact identity the counts have to satisfy: the second layer is
        // driven only by the first layer's spikes, so its share of `syn_ops` is a spike count
        // times a fan-out and nothing else.
        let ann2 = net(151, 6, 12, 5);
        let data2 = samples(152, 32, 6);
        let mut snn = SpikingMlp::from_ann(&ann2, &data2, cfg).expect("convertible");
        snn.run(&data2[0], ticks).expect("positive ticks");
        let hidden_spikes: u64 = snn.layers[0].counts.iter().sum();
        assert!(hidden_spikes > 0, "the hidden layer never fired, so the identity is vacuous");
        assert_eq!(
            snn.ledger.syn_ops,
            ticks * 6 * 12 + hidden_spikes * 5,
            "syn_ops is not the dense first layer plus one delivery per hidden spike"
        );
        assert_eq!(snn.ledger.syn_fetches, snn.ledger.syn_ops);
    }

    /// The convention the split carries, stated as a measurement rather than a caveat: a layer
    /// whose inputs are all silent is counted IDLE even on ticks when its own bias makes it spike.
    /// That is what event-driven hardware means by idle — no event arrived — and it is the one
    /// reading of the counter that a reader could get wrong.
    #[test]
    fn the_bias_only_layer_is_counted_idle_and_still_spikes() {
        let layer = DenseRelu::new(2, 1, vec![1.0, 1.0], vec![0.5]).expect("shapes match");
        let ann = Mlp::new(vec![layer]).expect("one layer");
        let data = vec![vec![0.75, 0.75]];
        let cfg = Config { norm: Norm::DataBased { percentile: 100.0 }, ..Config::default() };
        let mut snn = SpikingMlp::from_ann(&ann, &data, cfg).expect("convertible");
        // lambda is the one observed activation, 0.75 + 0.75 + 0.5 = 2.0, so the normalised bias
        // is exactly 0.25 per tick and the unit crosses threshold every fourth tick with no input
        // at all. Dyadic, so the count is an equality and not a tolerance.
        assert_eq!(snn.lambdas[1], 2.0, "lambda was {}", snn.lambdas[1]);
        let ticks = 100u64;
        snn.run(&[0.0, 0.0], ticks).expect("positive ticks");
        assert_eq!(snn.ledger.spikes_out, 25, "the bias alone did not fire at 0.25 per tick");
        assert_eq!(snn.ledger.syn_ops, 0, "nothing was delivered across a synapse");
        assert_eq!(snn.ledger.neuron_updates_driven, 0);
        assert_eq!(snn.ledger.neuron_updates_idle, ticks);
        assert_eq!(
            snn.ledger.idle_fraction(),
            Some(1.0),
            "a spiking layer read as anything other than fully idle"
        );
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

    /// `EXACT_OVER_GAPS` is a promise that `crate::sim` enforces, and for this model it is FALSE.
    /// Demonstrate the divergence rather than assert the constant: below threshold the two ways of
    /// crossing a quiet gap agree bit for bit, and above it they do not agree at all.
    #[test]
    fn a_supra_threshold_residual_fires_on_quiet_ticks() {
        const { assert!(!SpikingRelu::EXACT_OVER_GAPS) };
        let dt = 1e-3;
        // The region the old `true` was tested on, and where it does hold: a sub-threshold
        // membrane under zero input does not move, however the gap is crossed.
        for floor in [false, true] {
            for &v0 in &[0.4f64, 0.0, 0.9375] {
                let mut a = SpikingRelu::new(1e-9, 1.0, Reset::BySubtraction);
                a.floor_at_zero = floor;
                a.v = v0;
                let mut b = a;
                for _ in 0..100 {
                    assert!(!a.step(dt, 0.0));
                }
                assert!(!b.step(dt * 100.0, 0.0));
                assert_eq!(
                    a.potential(),
                    b.potential(),
                    "a quiet gap moved a sub-threshold membrane (floor {floor}, v0 {v0})"
                );
            }
        }
        // And the region reset-by-subtraction creates, where it fails. `v0` is dyadic and the
        // subtraction is exact, so these are equalities and not tolerances.
        for &(v0, clocked_spikes, clocked_v, jumped_v) in
            &[(3.5f64, 3u32, 0.5f64, 2.5f64), (12.0, 12, 0.0, 11.0), (2.0, 2, 0.0, 1.0)]
        {
            let mut a = SpikingRelu::new(1e-9, 1.0, Reset::BySubtraction);
            a.v = v0;
            let mut b = a;
            let mut spikes = 0u32;
            for _ in 0..100 {
                if a.step(dt, 0.0) {
                    spikes += 1;
                }
            }
            let jumped = u32::from(b.step(dt * 100.0, 0.0));
            assert_eq!(spikes, clocked_spikes, "v0 {v0}: tick-by-tick spike count");
            assert_eq!(jumped, 1, "v0 {v0}: a single jump can emit at most one spike");
            assert_eq!(a.potential(), clocked_v, "v0 {v0}: tick-by-tick membrane");
            assert_eq!(b.potential(), jumped_v, "v0 {v0}: jumped membrane");
            assert_ne!(
                a.potential(),
                b.potential(),
                "v0 {v0}: the two ways of crossing the gap agreed, so the constant could be true"
            );
        }
        // Reset-to-zero would satisfy the contract — it throws the overshoot away — which is why
        // the constant is a property of the type and not of the field.
        for &v0 in &[3.5f64, 12.0] {
            let mut a = SpikingRelu::new(1e-9, 1.0, Reset::ToZero);
            a.v = v0;
            let mut b = a;
            let mut spikes = 0u32;
            for _ in 0..100 {
                if a.step(dt, 0.0) {
                    spikes += 1;
                }
            }
            let jumped = u32::from(b.step(dt * 100.0, 0.0));
            assert_eq!((spikes, a.potential()), (1, 0.0), "reset-to-zero, tick by tick");
            assert_eq!((jumped, b.potential()), (1, 0.0), "reset-to-zero, jumped");
        }
    }

    /// The broken state is reachable through the public constructor alone, on this module's OWN
    /// fixture: no field is poked, the outlier sample is simply run.
    #[test]
    fn a_saturating_conversion_reaches_the_state_that_breaks_the_gap_property() {
        let layer = DenseRelu::new(1, 1, vec![1.0], vec![0.0]).expect("shapes match");
        let ann = Mlp::new(vec![layer]).expect("one layer");
        let mut data: Vec<Vec<f64>> =
            (0..200).map(|k| vec![0.1 + 0.9 * (k as f64 / 199.0)]).collect();
        data.push(vec![100.0]);
        let cfg = Config { norm: Norm::DataBased { percentile: 99.0 }, ..Config::default() };
        let mut snn = SpikingMlp::from_ann(&ann, &data, cfg).expect("convertible");
        snn.run(&[100.0], 50).expect("positive ticks");
        let v = snn.layers[0].neurons[0].potential();
        assert!(
            v > 2.0 * cfg.v_th,
            "the saturating sample left the membrane at {v} V, below the two thresholds that make \
             a quiet tick fire"
        );
        // The measured figure the constant's doc quotes, to one part in a thousand.
        assert!((v - 4972.7).abs() < 1.0, "membrane was {v} V, not the 4972.7 V documented");
    }

    /// What the wrong constant would have cost, priced in dropped spikes. `crate::sim::catch_up`
    /// crosses a quiet gap with ONE step and discards its return value, which is legal exactly when
    /// `EXACT_OVER_GAPS` holds. Here it does not, and the bill is 20 spikes against 5.
    #[test]
    fn a_gap_jump_drops_the_spikes_the_residual_owes() {
        let dt = 1e-3;
        let deliveries: [u64; 5] = [5, 10, 15, 20, 25];
        let ticks = 30u64;
        let dv = 4.0; // four thresholds per delivery: the saturating regime, in one bump

        let mut clocked = SpikingRelu::new(1e-9, 1.0, Reset::BySubtraction);
        let mut n_clocked = 0u32;
        for t in 0..ticks {
            if deliveries.contains(&t) {
                clocked.bump(dv);
            }
            if clocked.step(dt, 0.0) {
                n_clocked += 1;
            }
        }
        // Every volt delivered is paid out as a spike, because subtraction keeps the residual:
        // 5 deliveries * 4 thresholds = 20 spikes, exactly, with nothing left on the membrane.
        assert_eq!(n_clocked, 20, "the clocked run did not pay out the whole delivery");
        assert_eq!(clocked.potential(), 0.0);

        // The event-driven arm, done exactly as `crate::sim::catch_up` does it: jump the quiet
        // interval in one step, discard that step's spike, then deliver and step one tick.
        let mut jumped = SpikingRelu::new(1e-9, 1.0, Reset::BySubtraction);
        let mut n_jumped = 0u32;
        let mut as_of = 0u64;
        for &t in &deliveries {
            let gap = t - as_of;
            if gap > 0 {
                let _dropped = jumped.step(dt * gap as f64, 0.0);
            }
            jumped.bump(dv);
            if jumped.step(dt, 0.0) {
                n_jumped += 1;
            }
            as_of = t + 1;
        }
        assert_eq!(n_jumped, 5, "one spike per delivery is all a jumped run can emit");
        assert!(
            jumped.potential() > 10.0,
            "the jumped run left {} V of unpaid charge, which is the dropped spikes",
            jumped.potential()
        );
        assert!(
            n_clocked > 3 * n_jumped,
            "clocked {n_clocked} against jumped {n_jumped}: the modes agreed, so the gate is moot"
        );
    }

    /// The gate itself: `crate::sim::Sim::new` refuses event-driven simulation of this model, and
    /// accepts the clocked mode. This is the safety property `EXACT_OVER_GAPS` exists to carry, and
    /// it fails closed.
    #[test]
    fn an_event_driven_sim_of_a_converted_neuron_is_refused() {
        use crate::net::NetBuilder;
        use crate::sim::{Mode, Sim, SimError};

        let mut b = NetBuilder::new(2);
        b.connect(0, 1, 4.0, 1).expect("indices in range");
        let net = b.build();
        let neurons = vec![SpikingRelu::new(1e-9, 1.0, Reset::BySubtraction); 2];
        assert_eq!(
            Sim::new(net.clone(), neurons.clone(), 1e-3, Mode::EventDriven).err(),
            Some(SimError::NotExactOverGaps),
            "event-driven mode was allowed for a model that cannot be jumped across a gap"
        );
        assert!(
            Sim::new(net, neurons, 1e-3, Mode::Clocked).is_ok(),
            "the clocked mode, which is always legal, was refused"
        );
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

        // The VALUE as well as the name: an error that named the field but reported a different
        // number would be as misleading as one that named the wrong field.
        for (name, value, cfg) in [
            ("dt", 0.0, Config { dt: 0.0, ..Config::default() }),
            ("v_th", -1.0, Config { v_th: -1.0, ..Config::default() }),
            ("c", f64::NEG_INFINITY, Config { c: f64::NEG_INFINITY, ..Config::default() }),
        ] {
            assert_eq!(cfg.validate(), Err(ConvertError::BadParameter { name, value }));
        }
        // `NaN` cannot be compared for equality, so it is matched rather than asserted equal.
        let e = Config { c: f64::NAN, ..Config::default() }.validate().expect_err("NaN farads");
        assert!(matches!(e, ConvertError::BadParameter { name: "c", value } if value.is_nan()));
        // Every variant can say what it is, including the one only the public fields can reach.
        let state = ConvertError::StateCount { layer: 2, field: "neurons", expected: 8, got: 7 };
        assert_eq!(state.to_string(), "layer 2 has 7 entries in neurons for 8 units");

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

    /// The public fields are a way to break the network after it is built, and every one of them
    /// used to be a panic or a plausible wrong answer. `snn.layers.clear()` indexed out of bounds;
    /// `snn.lambdas[0] = 0.0` divided by zero and returned three finite, identical, meaningless
    /// numbers. Every one is a named refusal now, and this test names them.
    #[test]
    fn public_fields_that_break_the_network_are_refused_rather_than_indexed() {
        let ann = net(141, 6, 8, 4);
        let data = samples(142, 32, 6);
        let cfg = Config { norm: Norm::DataBased { percentile: 100.0 }, ..Config::default() };
        let x = data[0].clone();
        let build = || SpikingMlp::from_ann(&ann, &data, cfg).expect("convertible");

        // The baseline: unbroken, this network runs and reports something finite and non-trivial.
        let mut ok = build();
        let base = ok.run(&x, 64).expect("positive ticks");
        assert!(base.iter().all(|v| v.is_finite()), "the intact network did not run");
        assert!(base.iter().any(|v| *v > 0.0), "the intact network reported nothing");

        // (1) No layers at all: an index panic, now `NoLayers`.
        let mut snn = build();
        snn.layers.clear();
        assert_eq!(snn.run(&x, 10), Err(ConvertError::NoLayers));
        assert_eq!(snn.tick(&x), Err(ConvertError::NoLayers));

        // (2) A zero scale: silently infinite drive, a saturated network, and three plausible
        // finite numbers out. Now refused by name, and the layer is named too.
        let mut snn = build();
        snn.lambdas[0] = 0.0;
        assert_eq!(
            snn.run(&x, 10),
            Err(ConvertError::DegenerateScale { layer: 0, lambda: 0.0 })
        );
        let mut snn = build();
        snn.lambdas[2] = f64::NAN;
        assert!(matches!(
            snn.run(&x, 10),
            Err(ConvertError::DegenerateScale { layer: 2, lambda }) if lambda.is_nan()
        ));

        // (3) The wrong number of scales.
        let mut snn = build();
        snn.lambdas.pop();
        assert_eq!(snn.run(&x, 10), Err(ConvertError::ScaleCount { expected: 3, got: 2 }));

        // (4) A physical parameter moved after conversion: `dt = 0` makes the gain infinite.
        for (name, broken) in [
            ("dt", Config { dt: 0.0, ..cfg }),
            ("v_th", Config { v_th: -1.0, ..cfg }),
            ("c", Config { c: f64::INFINITY, ..cfg }),
        ] {
            let mut snn = build();
            snn.cfg = broken;
            let e = snn.run(&x, 10).expect_err("a broken parameter must be refused");
            assert!(matches!(e, ConvertError::BadParameter { name: n, .. } if n == name), "{e}");
        }

        // (5) The per-unit state arrays, which only this module's own error variant covers.
        let mut snn = build();
        snn.layers[0].neurons.pop();
        assert_eq!(
            snn.run(&x, 10),
            Err(ConvertError::StateCount { layer: 0, field: "neurons", expected: 8, got: 7 })
        );
        let mut snn = build();
        snn.layers[1].counts.push(0);
        assert_eq!(
            snn.run(&x, 10),
            Err(ConvertError::StateCount { layer: 1, field: "counts", expected: 4, got: 5 })
        );

        // (6) The weights and biases themselves.
        let mut snn = build();
        snn.layers[0].w.pop();
        assert_eq!(snn.run(&x, 10), Err(ConvertError::WeightCount { expected: 48, got: 47 }));
        let mut snn = build();
        snn.layers[0].b.pop();
        assert_eq!(snn.run(&x, 10), Err(ConvertError::BiasCount { expected: 8, got: 7 }));

        // (7) A shape that no longer chains, and an emptied one.
        let mut snn = build();
        snn.layers[1].n_in = 99;
        assert_eq!(
            snn.run(&x, 10),
            Err(ConvertError::Disconnected { layer: 1, expected: 99, got: 8 })
        );
        let mut snn = build();
        snn.layers[0].n_out = 0;
        assert_eq!(snn.run(&x, 10), Err(ConvertError::EmptyLayer { n_in: 6, n_out: 0 }));

        // And the checks are not a one-way door: an untouched network still runs afterwards, with
        // the same answer it gave before any of this.
        let mut ok2 = build();
        assert_eq!(ok2.run(&x, 64).expect("positive ticks"), base);
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
