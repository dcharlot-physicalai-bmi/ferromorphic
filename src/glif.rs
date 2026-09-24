//! The Allen Institute's generalized leaky integrate-and-fire models, GLIF 1 to 5, read from the
//! files the Allen Cell Types Database serves and run as the `AllenSDK` runs them.
//!
//! # What these are
//!
//! Teeter et al., *Generalized leaky integrate-and-fire models classify multiple neuron types*,
//! Nature Communications 9:709 (2018), fitted five nested models to patch-clamp recordings of mouse
//! and human cortical neurons, each adding one mechanism to the last:
//!
//! | level | name | adds |
//! |---|---|---|
//! | 1 | LIF | a leaky membrane, a fixed threshold, reset to rest |
//! | 2 | LIF-R | reset rules: `V ← aV + b`, and a threshold that jumps at a spike and decays |
//! | 3 | LIF-ASC | after-spike currents, each jumping at a spike and decaying exponentially |
//! | 4 | LIF-R-ASC | both |
//! | 5 | LIF-R-ASC-A | a threshold that also adapts to the voltage (Mihalas and Niebur, Neural Computation 21:704, 2009) |
//!
//! The database serves every fitted model as a `neuron_config.json`: on 2026-09-24 its API listed
//! 1,218 models at level 1, 439 at level 2, 1,218 at level 3, 439 at level 4 and 439 at level 5.
//! [`Glif::from_neuron_config`] reads that file as served, through [`crate::json`], so a fitted
//! parameter reaches the simulation without being retyped.
//!
//! # The reference is the `AllenSDK`'s own code
//!
//! Its `allensdk/model/glif/glif_neuron.py` and `glif_neuron_methods.py` (master, last changed
//! 2026-02-20, commit `1bdca3ad88`) define what a config MEANS: which update runs first, which
//! state each formula reads, how the spike is cut out of the trace. [`Glif::run`] follows them line
//! for line, and the tests compare it with that code run unmodified on five real fitted models, one
//! per level: every spike on the same time step, every interpolated spike time within `10⁻¹²` s,
//! the same samples cut, and each whole trace — voltage, threshold, every after-spike current —
//! summing to the reference's sum within `10⁻¹²` of its sum of magnitudes. `tools/glif_reference.py`
//! fetches the `AllenSDK` at that commit and the five configs, reruns them, and prints the table the
//! tests hold; it reproduces it byte for byte.
//!
//! ⭐ **And then every model the database serves.** On 2026-09-24, `tools/glif_sweep.py` downloaded
//! all 3,753 served configs and ran each through the `AllenSDK` under a 300 ms step at twice its
//! rheobase; `examples/glif_allen.rs` ran the same files through this module. All 3,753 were read,
//! every one at the level the database files it under, and they fired 43,950 spikes between them —
//! every spike on the same step as the `AllenSDK`'s, and every voltage trace summing to the same
//! bits. The comparison is the script's `compare` step, and it prints any model that differs.
//!
//! Reading that code closely turned up three things a user of the fitted models should know.
//!
//! ⚠ **The voltage is integrated by forward Euler, in every fitted model.** The methods file defines
//! an exact exponential integrator, `dynamics_voltage_linear_exact`, but its `METHOD_LIBRARY`
//! registers only `linear_forward_euler`, and every config names that. The fits were made with
//! Euler at `dt = 50 µs`, so Euler is what reproduces them; [`Integrator::Exact`] is here for a
//! reader who wants the membrane equation solved rather than approximated, and its tests say how far
//! apart the two are.
//!
//! ⚠ **The `AllenSDK`'s interpolated spike voltage is not the voltage at the interpolated spike time.**
//! It places the spike at `t·dt + x` — `x` where the voltage and threshold lines cross inside the
//! step — and then evaluates both lines at an offset measured from `(t − 1)·dt`, that is at `x + dt`.
//! So its `interpolated_spike_voltage` is the crossing value plus one whole step's rise `v₁ − v₀`,
//! and its `interpolated_spike_threshold` the crossing plus `θ₁ − θ₀`; two numbers that must be
//! equal at a crossing differ, by 17 to 100 µV on the five models the tests use. [`GlifSpike`]
//! reports the crossing itself, where they agree to rounding.
//!
//! ⚠ **The exact voltage-dependent threshold divides by `b_v − g/C` and by `b_v`.** Its closed form,
//! `φ = a_v/(b_v − g/C)`, is a `ZeroDivisionError` in Python when the threshold's decay rate equals
//! the membrane's, and loses digits to cancellation near it. Written with `φ₁(x) = (1 − e⁻ˣ)/x`,
//! evaluated as `−expm1(−x)/x`, the same solution is continuous through both points —
//! [`threshold_voltage_component`], whose tests check it against the ODE it solves.
//!
//! # Units and reference potential
//!
//! SI throughout, as the configs are: volts, amperes, farads, ohms, seconds. Every potential in a
//! config is relative to its `El_reference`, the cell's measured resting potential, and every
//! config sets `El = 0`; the voltage reset method `zero` sets `V = 0`, which is rest only because
//! of that. [`Glif::e_l_reference`] is kept so an absolute potential can be recovered.

use core::fmt;

use crate::json::{self, Json, JsonError};

/// Why a GLIF model could not be read or run.
#[derive(Debug, Clone, PartialEq)]
pub enum GlifError {
    /// The config is not JSON.
    Json(JsonError),
    /// A key the config must have is missing or holds the wrong kind of value.
    Missing {
        /// The key, dotted from the top level.
        key: &'static str,
    },
    /// A method name the `AllenSDK`'s `METHOD_LIBRARY` does not register.
    Unknown {
        /// The method slot, e.g. `voltage_reset_method`.
        method: &'static str,
        /// The name the config gave.
        name: String,
    },
    /// Dynamics and reset methods that no GLIF level pairs, or values that must agree and do not.
    Mismatch {
        /// What disagrees.
        what: &'static str,
    },
    /// A parameter that must be finite and positive is not.
    NotPositive {
        /// Which parameter.
        what: &'static str,
        /// Its value.
        value: f64,
    },
    /// A parameter or input that must be finite is not.
    NonFinite {
        /// Which quantity.
        what: &'static str,
        /// Its value.
        value: f64,
    },
}

impl fmt::Display for GlifError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Json(e) => write!(f, "{e}"),
            Self::Missing { key } => write!(f, "the config has no usable {key}"),
            Self::Unknown { method, name } => write!(f, "{method} {name:?} is not a method the AllenSDK registers"),
            Self::Mismatch { what } => write!(f, "{what}"),
            Self::NotPositive { what, value } => write!(f, "{what} = {value} must be finite and positive"),
            Self::NonFinite { what, value } => write!(f, "{what} = {value} is not finite"),
        }
    }
}

impl std::error::Error for GlifError {}

impl From<JsonError> for GlifError {
    fn from(e: JsonError) -> Self {
        Self::Json(e)
    }
}

fn finite(what: &'static str, value: f64) -> Result<f64, GlifError> {
    if value.is_finite() { Ok(value) } else { Err(GlifError::NonFinite { what, value }) }
}

fn positive(what: &'static str, value: f64) -> Result<f64, GlifError> {
    if value.is_finite() && value > 0.0 { Ok(value) } else { Err(GlifError::NotPositive { what, value }) }
}

/// How the membrane equation is advanced by one step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Integrator {
    /// `V ← V + (I + ΣI_j − g(V − E_L))·dt/C` — the `AllenSDK`'s `linear_forward_euler`, the only one it
    /// registers, and the one every fitted model was fitted with.
    ForwardEuler,
    /// The exact solution for current held constant over the step,
    /// `V ← β + (V − β)e^{−g·dt/C}` with `β = E_L + (I + ΣI_j)/g`.
    Exact,
}

/// One after-spike current.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AfterSpikeCurrent {
    /// Its decay time constant, seconds (`asc_tau_array`).
    pub tau: f64,
    /// The current it jumps by at a spike, amperes (`asc_amp_array` times its coefficient).
    pub amp: f64,
    /// The fraction of its present value it keeps across a spike (`r`); 1 in the fitted models.
    pub r: f64,
}

/// How the voltage is reset at a spike.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VoltageReset {
    /// `V ← 0` — the `AllenSDK`'s `zero`, which is rest because every config sets `El = 0`.
    Zero,
    /// `V ← aV + b` with `V` the voltage that crossed threshold — `v_before`.
    Scaled {
        /// `a`, dimensionless.
        a: f64,
        /// `b`, volts.
        b: f64,
    },
}

/// How the threshold moves.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Threshold {
    /// Fixed at `θ∞` — the `AllenSDK`'s `inf` dynamics and `inf` reset.
    Fixed,
    /// `θ∞` plus a spike component that jumps by `a_spike` at each spike and decays at rate
    /// `b_spike` — `spike_component` dynamics with the `three_components` reset.
    Spike {
        /// The jump, volts.
        a_spike: f64,
        /// The decay rate, 1/s.
        b_spike: f64,
    },
    /// The spike component and a voltage component `dθ_v/dt = a_v(V − E_L) − b_v θ_v`, integrated
    /// exactly over each step — `three_components_exact` with the `three_components` reset.
    SpikeAndVoltage {
        /// The spike component's jump, volts.
        a_spike: f64,
        /// The spike component's decay rate, 1/s.
        b_spike: f64,
        /// The voltage component's coupling, 1/s.
        a_voltage: f64,
        /// The voltage component's decay rate, 1/s.
        b_voltage: f64,
    },
}

/// A GLIF model: the parameters of one fitted neuron, coefficients already applied.
#[derive(Debug, Clone, PartialEq)]
pub struct Glif {
    /// The time step the model was fitted at, seconds.
    pub dt: f64,
    /// `E_L`, volts relative to [`Glif::e_l_reference`]; 0 in every fitted model.
    pub e_l: f64,
    /// The resting potential every other potential is measured from, volts absolute.
    pub e_l_reference: f64,
    /// Membrane capacitance, farads.
    pub c: f64,
    /// Membrane conductance `1/R_input`, siemens.
    pub g: f64,
    /// The instantaneous threshold `θ∞`, volts.
    pub th_inf: f64,
    /// The voltage a run starts from.
    pub init_voltage: f64,
    /// The threshold a run starts from. ⚠ In the fitted models this is the UNSCALED `th_inf`,
    /// before its coefficient, and it is used only as `θ₀` in the first step's spike interpolation.
    pub init_threshold: f64,
    /// The after-spike currents a run starts from; one per entry of [`Glif::asc`].
    pub init_asc: Vec<f64>,
    /// The after-spike currents; empty for levels 1 and 2.
    pub asc: Vec<AfterSpikeCurrent>,
    /// The voltage reset.
    pub voltage_reset: VoltageReset,
    /// The threshold dynamics and reset.
    pub threshold: Threshold,
    /// Steps cut out of the trace after each spike, during which nothing is integrated.
    pub spike_cut: usize,
    /// How the membrane equation is advanced.
    pub integrator: Integrator,
}

/// One spike of a [`GlifRun`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GlifSpike {
    /// The step on which the voltage ended above threshold.
    pub step: usize,
    /// The interpolated time, seconds: `step·dt` plus where the voltage and threshold lines cross
    /// inside the step — the `AllenSDK`'s `interpolated_spike_times`, to the bit.
    pub time: f64,
    /// The voltage line at the crossing.
    pub voltage: f64,
    /// The threshold line at the crossing; equal to [`GlifSpike::voltage`] to rounding.
    pub threshold: f64,
    /// The voltage at the start of the crossing step.
    pub v0: f64,
    /// The voltage at its end, above [`GlifSpike::th1`] — the `V` the voltage reset is applied to.
    pub v1: f64,
    /// The threshold at the start of the crossing step.
    pub th0: f64,
    /// The threshold at its end.
    pub th1: f64,
}

/// What [`Glif::run`] returns: one sample per stimulus sample, `NaN` inside each spike cut.
///
/// Sample `t` is the state AFTER the step that consumed `stim[t]`. At a spike on step `t` the
/// samples `t .. t + spike_cut` are `NaN` and sample `t + spike_cut` holds the reset state, which
/// the next step starts from — the `AllenSDK`'s layout exactly.
#[derive(Debug, Clone, PartialEq)]
pub struct GlifRun {
    /// Membrane potential, volts relative to `E_L`'s reference.
    pub voltage: Vec<f64>,
    /// Total threshold, volts.
    pub threshold: Vec<f64>,
    /// Each after-spike current's trace, amperes: `asc[j][t]`.
    pub asc: Vec<Vec<f64>>,
    /// Every spike, in order.
    pub spikes: Vec<GlifSpike>,
    /// The step of the spike whose reset left the voltage above the threshold, if one did.
    ///
    /// The `AllenSDK` STOPS there: it writes the reset state into the next five samples and leaves the
    /// rest of the trace empty, because a model whose reset is above its own threshold would spike
    /// on every step. This run does the same, and says where.
    pub bad_reset: Option<usize>,
}

/// `φ₁(x) = (1 − e⁻ˣ)/x`, with `φ₁(0) = 1`, evaluated without cancellation.
fn phi1(x: f64) -> f64 {
    if x == 0.0 { 1.0 } else { -(-x).exp_m1() / x }
}

/// The voltage component of the threshold after `t` seconds, from `th0`, with the membrane starting
/// at `v0` under constant total current `i`: the exact solution of
/// `dθ_v/dt = a_v(V − E_L) − b_v θ_v` with `V(s) = β + (v0 − β)e^{−ks}`, `k = g/C`, `β = E_L + i/g`.
///
/// ```text
/// θ_v(t) = θ₀e^{−b t} + a(β − E_L)·t·φ₁(b t) + a(v₀ − β)·t·e^{−k t}·φ₁((b − k)t)
/// ```
///
/// This is the `AllenSDK`'s `voltage_component_of_threshold_exact` rearranged: its
/// `(1 − e^{−bt})/b` and `(e^{−kt} − e^{−bt})/(b − k)` are `t·φ₁(bt)` and `t·e^{−kt}·φ₁((b − k)t)`,
/// which have removable singularities at `b = 0` and `b = k` where the originals divide by zero.
#[must_use]
pub fn threshold_voltage_component(th0: f64, v0: f64, i: f64, t: f64, a_v: f64, b_v: f64, c: f64, g: f64, e_l: f64) -> f64 {
    let k = g / c;
    let beta = (i + g * e_l) / g;
    th0 * (-b_v * t).exp() + a_v * (beta - e_l) * t * phi1(b_v * t) + a_v * (v0 - beta) * t * (-k * t).exp() * phi1((b_v - k) * t)
}

/// A key's number, or [`GlifError::Missing`].
fn num(v: &Json, key: &'static str) -> Result<f64, GlifError> {
    v.as_f64().ok_or(GlifError::Missing { key })
}

/// An object member's number, or [`GlifError::Missing`] naming the member's dotted path.
fn member(v: &Json, name: &str, key: &'static str) -> Result<f64, GlifError> {
    v.get(name).ok_or(GlifError::Missing { key }).and_then(|x| num(x, key))
}

/// A list of numbers, or [`GlifError::Missing`].
fn numbers(v: Option<&Json>, key: &'static str) -> Result<Vec<f64>, GlifError> {
    let items = v.and_then(Json::as_array).ok_or(GlifError::Missing { key })?;
    items.iter().map(|x| num(x, key)).collect()
}

/// A method slot's `(name, params)`.
fn method<'a>(cfg: &'a Json, slot: &'static str) -> Result<(&'a str, &'a Json), GlifError> {
    let m = cfg.get(slot).ok_or(GlifError::Missing { key: slot })?;
    let name = m.get("name").and_then(Json::as_str).ok_or(GlifError::Missing { key: slot })?;
    let params = m.get("params").ok_or(GlifError::Missing { key: slot })?;
    Ok((name, params))
}

fn unknown(method: &'static str, name: &str) -> GlifError {
    GlifError::Unknown { method, name: name.to_owned() }
}

impl Glif {
    /// Read a model from the text of an Allen Cell Types Database `neuron_config.json`.
    ///
    /// Coefficients are applied as the `AllenSDK` applies them — `th_inf`, `C`, `G`, `a`, `b` and each
    /// after-spike amplitude multiplied by its entry in `coeffs`, a missing entry counting as 1 —
    /// and each product is formed once, exactly as the `AllenSDK` forms it on every step, so the
    /// numbers are the same bits. Method names are checked against the `AllenSDK`'s `METHOD_LIBRARY`,
    /// and dynamics and reset methods must pair as some GLIF level pairs them: an after-spike
    /// current either decays and sums (`exp`, `sum`) or does not exist (`none`, `none`); the threshold
    /// is `inf`/`inf`, `spike_component`/`three_components` or `three_components_exact`/
    /// `three_components`, and the spike component's `a_spike` and `b_spike` must be the same in the
    /// dynamics and the reset.
    ///
    /// # Errors
    ///
    /// [`GlifError::Json`] for text that is not JSON; [`GlifError::Missing`],
    /// [`GlifError::Unknown`] or [`GlifError::Mismatch`] for a config the `AllenSDK` could not run;
    /// and whatever [`Glif::check`] refuses.
    pub fn from_neuron_config(text: &str) -> Result<Self, GlifError> {
        let cfg = json::parse(text)?;
        let coeffs = cfg.get("coeffs");
        let coeff = |name: &str, key: &'static str| -> Result<f64, GlifError> {
            match coeffs.and_then(|c| c.get(name)) {
                None => Ok(1.0),
                Some(v) => num(v, key),
            }
        };
        let r_input = member(&cfg, "R_input", "R_input")?;
        let cut = member(&cfg, "spike_cut_length", "spike_cut_length")?;
        if !(cut >= 0.0 && cut.fract() == 0.0) {
            return Err(GlifError::Mismatch { what: "spike_cut_length is not a whole number of steps" });
        }

        let (v_name, _) = method(&cfg, "voltage_dynamics_method")?;
        if v_name != "linear_forward_euler" {
            return Err(unknown("voltage_dynamics_method", v_name));
        }

        let (asc_dyn, _) = method(&cfg, "AScurrent_dynamics_method")?;
        let (asc_reset, asc_params) = method(&cfg, "AScurrent_reset_method")?;
        if !matches!(asc_dyn, "exp" | "none") {
            return Err(unknown("AScurrent_dynamics_method", asc_dyn));
        }
        if !matches!(asc_reset, "sum" | "none") {
            return Err(unknown("AScurrent_reset_method", asc_reset));
        }
        let init_asc = numbers(cfg.get("init_AScurrents"), "init_AScurrents")?;
        let asc = match (asc_dyn, asc_reset) {
            ("none", "none") => {
                // The AllenSDK's `none` dynamics zeroes the currents after the first step, but that
                // step's voltage still reads the initial ones; a non-zero start would be a current
                // the model has no mechanism for, on one step only.
                if init_asc.iter().any(|&i| i != 0.0) {
                    return Err(GlifError::Mismatch { what: "init_AScurrents is non-zero in a model without after-spike currents" });
                }
                Vec::new()
            }
            ("exp", "sum") => {
                let taus = numbers(cfg.get("asc_tau_array"), "asc_tau_array")?;
                let amps = numbers(cfg.get("asc_amp_array"), "asc_amp_array")?;
                let rs = numbers(asc_params.get("r"), "AScurrent_reset_method.params.r")?;
                let amp_coeffs = match coeffs.and_then(|c| c.get("asc_amp_array")) {
                    None => vec![1.0; amps.len()],
                    Some(v) => numbers(Some(v), "coeffs.asc_amp_array")?,
                };
                if taus.len() != amps.len() || rs.len() != amps.len() || amp_coeffs.len() != amps.len() {
                    return Err(GlifError::Mismatch { what: "the after-spike current arrays differ in length" });
                }
                (0..amps.len()).map(|j| AfterSpikeCurrent { tau: taus[j], amp: amps[j] * amp_coeffs[j], r: rs[j] }).collect()
            }
            _ => return Err(GlifError::Mismatch { what: "after-spike current dynamics and reset methods that no GLIF level pairs" }),
        };

        let (vr_name, vr_params) = method(&cfg, "voltage_reset_method")?;
        let voltage_reset = match vr_name {
            "zero" => VoltageReset::Zero,
            "v_before" => VoltageReset::Scaled {
                a: member(vr_params, "a", "voltage_reset_method.params.a")?,
                b: member(vr_params, "b", "voltage_reset_method.params.b")?,
            },
            _ => return Err(unknown("voltage_reset_method", vr_name)),
        };

        let (th_dyn, th_dyn_params) = method(&cfg, "threshold_dynamics_method")?;
        let (th_reset, th_reset_params) = method(&cfg, "threshold_reset_method")?;
        if !matches!(th_dyn, "inf" | "spike_component" | "three_components_exact") {
            return Err(unknown("threshold_dynamics_method", th_dyn));
        }
        if !matches!(th_reset, "inf" | "three_components") {
            return Err(unknown("threshold_reset_method", th_reset));
        }
        // Both adaptive dynamics take all four of `a_spike`, `b_spike`, `a_voltage` and `b_voltage`
        // as arguments, so the AllenSDK cannot run a config that omits one — `spike_component` then
        // ignores the voltage pair. The reset takes `a_spike` and `b_spike` again, and one model
        // has one spike component: the two copies must be the same number.
        let adaptive = || -> Result<[f64; 4], GlifError> {
            let a = member(th_reset_params, "a_spike", "threshold_reset_method.params.a_spike")?;
            let b = member(th_reset_params, "b_spike", "threshold_reset_method.params.b_spike")?;
            let a_dyn = member(th_dyn_params, "a_spike", "threshold_dynamics_method.params.a_spike")?;
            let b_dyn = member(th_dyn_params, "b_spike", "threshold_dynamics_method.params.b_spike")?;
            if a_dyn.to_bits() != a.to_bits() || b_dyn.to_bits() != b.to_bits() {
                return Err(GlifError::Mismatch { what: "a_spike or b_spike differs between the threshold dynamics and reset" });
            }
            let a_v = member(th_dyn_params, "a_voltage", "threshold_dynamics_method.params.a_voltage")?;
            let b_v = member(th_dyn_params, "b_voltage", "threshold_dynamics_method.params.b_voltage")?;
            Ok([a, b, a_v * coeff("a", "coeffs.a")?, b_v * coeff("b", "coeffs.b")?])
        };
        let threshold = match (th_dyn, th_reset) {
            ("inf", "inf") => Threshold::Fixed,
            ("spike_component", "three_components") => {
                let [a_spike, b_spike, _, _] = adaptive()?;
                Threshold::Spike { a_spike, b_spike }
            }
            ("three_components_exact", "three_components") => {
                let [a_spike, b_spike, a_voltage, b_voltage] = adaptive()?;
                Threshold::SpikeAndVoltage { a_spike, b_spike, a_voltage, b_voltage }
            }
            _ => return Err(GlifError::Mismatch { what: "threshold dynamics and reset methods that no GLIF level pairs" }),
        };

        let model = Self {
            dt: member(&cfg, "dt", "dt")?,
            e_l: member(&cfg, "El", "El")?,
            e_l_reference: member(&cfg, "El_reference", "El_reference")?,
            c: member(&cfg, "C", "C")? * coeff("C", "coeffs.C")?,
            g: 1.0 / r_input * coeff("G", "coeffs.G")?,
            th_inf: member(&cfg, "th_inf", "th_inf")? * coeff("th_inf", "coeffs.th_inf")?,
            init_voltage: member(&cfg, "init_voltage", "init_voltage")?,
            init_threshold: member(&cfg, "init_threshold", "init_threshold")?,
            init_asc: if asc.is_empty() { Vec::new() } else { init_asc },
            asc,
            voltage_reset,
            threshold,
            spike_cut: cut as usize,
            integrator: Integrator::ForwardEuler,
        };
        model.check()?;
        Ok(model)
    }

    /// Which of Teeter et al.'s five levels this model is, from its mechanisms; `None` for a
    /// combination none of them uses.
    #[must_use]
    pub fn level(&self) -> Option<u8> {
        let has_asc = !self.asc.is_empty();
        match (self.threshold, self.voltage_reset, has_asc) {
            (Threshold::Fixed, VoltageReset::Zero, false) => Some(1),
            (Threshold::Spike { .. }, VoltageReset::Scaled { .. }, false) => Some(2),
            (Threshold::Fixed, VoltageReset::Zero, true) => Some(3),
            (Threshold::Spike { .. }, VoltageReset::Scaled { .. }, true) => Some(4),
            (Threshold::SpikeAndVoltage { .. }, VoltageReset::Scaled { .. }, true) => Some(5),
            _ => None,
        }
    }

    /// Every parameter finite, and the ones that must be positive positive.
    ///
    /// # Errors
    ///
    /// [`GlifError::NotPositive`] for `dt`, `C`, `g` or an after-spike time constant that is not;
    /// [`GlifError::NonFinite`] for any other parameter that is not finite;
    /// [`GlifError::Mismatch`] when [`Glif::init_asc`] and [`Glif::asc`] differ in length.
    pub fn check(&self) -> Result<(), GlifError> {
        positive("dt", self.dt)?;
        positive("C", self.c)?;
        positive("g", self.g)?;
        for (what, value) in [
            ("El", self.e_l),
            ("El_reference", self.e_l_reference),
            ("th_inf", self.th_inf),
            ("init_voltage", self.init_voltage),
            ("init_threshold", self.init_threshold),
        ] {
            finite(what, value)?;
        }
        for a in &self.asc {
            positive("asc tau", a.tau)?;
            finite("asc amp", a.amp)?;
            finite("asc r", a.r)?;
        }
        if self.init_asc.len() != self.asc.len() {
            return Err(GlifError::Mismatch { what: "init_AScurrents and the after-spike currents differ in length" });
        }
        for &i in &self.init_asc {
            finite("init_AScurrents", i)?;
        }
        if let VoltageReset::Scaled { a, b } = self.voltage_reset {
            finite("reset a", a)?;
            finite("reset b", b)?;
        }
        match self.threshold {
            Threshold::Fixed => {}
            Threshold::Spike { a_spike, b_spike } => {
                finite("a_spike", a_spike)?;
                finite("b_spike", b_spike)?;
            }
            Threshold::SpikeAndVoltage { a_spike, b_spike, a_voltage, b_voltage } => {
                finite("a_spike", a_spike)?;
                finite("b_spike", b_spike)?;
                finite("a_voltage", a_voltage)?;
                finite("b_voltage", b_voltage)?;
            }
        }
        Ok(())
    }

    /// Run the model over `stim`, one current sample in amperes per step of [`Glif::dt`].
    ///
    /// Each step, in the `AllenSDK`'s order and reading the state at the START of the step: the
    /// after-spike currents decay, `I_j ← I_j e^{−dt/τ_j}`; the voltage advances under the injected
    /// current plus the after-spike currents' sum; the threshold's spike component decays
    /// `θ_s ← θ_s e^{−b_s dt}` and its voltage component advances by
    /// [`threshold_voltage_component`]. If the new voltage exceeds the new threshold the neuron has
    /// spiked: each current becomes `amp + r·I_j·e^{−cut·dt/τ_j}` (carried through the cut), the
    /// voltage is reset, the spike component decays through the cut and jumps by `a_spike`, the
    /// voltage component is held, and `spike_cut` samples are cut from the trace.
    ///
    /// # Errors
    ///
    /// Whatever [`Glif::check`] refuses; [`GlifError::NonFinite`] for a stimulus sample that is not
    /// finite.
    pub fn run(&self, stim: &[f64]) -> Result<GlifRun, GlifError> {
        self.check()?;
        for &i in stim {
            finite("stimulus", i)?;
        }
        let (dt, n, n_asc) = (self.dt, stim.len(), self.asc.len());
        let mut out = GlifRun {
            voltage: vec![f64::NAN; n],
            threshold: vec![f64::NAN; n],
            asc: vec![vec![f64::NAN; n]; n_asc],
            spikes: Vec::new(),
            bad_reset: None,
        };
        let (mut v0, mut th0) = (self.init_voltage, self.init_threshold);
        let mut asc0 = self.init_asc.clone();
        let (mut th_spike, mut th_volt) = (0.0_f64, 0.0_f64);
        let mut t = 0;
        while t < n {
            let inj = stim[t];
            let asc_sum: f64 = asc0.iter().sum();
            let asc1: Vec<f64> =
                asc0.iter().zip(&self.asc).map(|(&i, a)| i * (-(1.0 / a.tau) * dt).exp()).collect();
            let v1 = match self.integrator {
                Integrator::ForwardEuler => v0 + (inj + asc_sum - self.g * (v0 - self.e_l)) * dt / self.c,
                Integrator::Exact => {
                    let beta = self.e_l + (inj + asc_sum) / self.g;
                    beta + (v0 - beta) * (-self.g / self.c * dt).exp()
                }
            };
            let (spike1, volt1) = match self.threshold {
                Threshold::Fixed => (0.0, 0.0),
                Threshold::Spike { b_spike, .. } => (th_spike * (-b_spike * dt).exp(), 0.0),
                Threshold::SpikeAndVoltage { b_spike, a_voltage, b_voltage, .. } => (
                    th_spike * (-b_spike * dt).exp(),
                    threshold_voltage_component(th_volt, v0, inj + asc_sum, dt, a_voltage, b_voltage, self.c, self.g, self.e_l),
                ),
            };
            let th1 = match self.threshold {
                Threshold::Fixed => self.th_inf,
                Threshold::Spike { .. } => spike1 + self.th_inf,
                Threshold::SpikeAndVoltage { .. } => volt1 + spike1 + self.th_inf,
            };
            (th_spike, th_volt) = (spike1, volt1);

            if v1 > th1 {
                let x = dt * (th0 - v0) / ((v1 - v0) - (th1 - th0));
                out.spikes.push(GlifSpike {
                    step: t,
                    time: t as f64 * dt + x,
                    voltage: v0 + (v1 - v0) * x / dt,
                    threshold: th0 + (th1 - th0) * x / dt,
                    v0,
                    v1,
                    th0,
                    th1,
                });
                let cut = self.spike_cut as f64;
                asc0 = asc1
                    .iter()
                    .zip(&self.asc)
                    .map(|(&i, a)| a.amp + i * a.r * (-((1.0 / a.tau) * dt * cut)).exp())
                    .collect();
                v0 = match self.voltage_reset {
                    VoltageReset::Zero => 0.0,
                    VoltageReset::Scaled { a, b } => a * v1 + b,
                };
                th0 = match self.threshold {
                    Threshold::Fixed => self.th_inf,
                    Threshold::Spike { a_spike, b_spike } | Threshold::SpikeAndVoltage { a_spike, b_spike, .. } => {
                        th_spike = th_spike * (-b_spike * (cut * dt)).exp() + a_spike;
                        th_spike + th_volt + self.th_inf
                    }
                };
                let spiked = t;
                let resumes = t.saturating_add(self.spike_cut);
                if resumes < n {
                    record(&mut out, resumes, v0, th0, &asc0);
                }
                t = resumes.saturating_add(1);
                if v0 > th0 {
                    out.bad_reset = Some(spiked);
                    for s in t..n.min(t.saturating_add(5)) {
                        record(&mut out, s, v0, th0, &asc0);
                    }
                    break;
                }
            } else {
                record(&mut out, t, v1, th1, &asc1);
                (v0, th0, asc0) = (v1, th1, asc1);
                t += 1;
            }
        }
        Ok(out)
    }
}

/// Write one sample of every trace.
fn record(out: &mut GlifRun, t: usize, v: f64, th: f64, asc: &[f64]) {
    out.voltage[t] = v;
    out.threshold[t] = th;
    for (trace, &i) in out.asc.iter_mut().zip(asc) {
        trace[t] = i;
    }
}

#[cfg(test)]
mod tests {
    use super::{AfterSpikeCurrent, Glif, GlifError, Integrator, Threshold, VoltageReset, threshold_voltage_component};
    use crate::json::JsonError;
    use crate::json;

    /// One fitted model from the Allen Cell Types Database, as its `neuron_config.json` was served on
    /// 2026-09-24, and what the `AllenSDK`'s own `GlifNeuron.run` produced for it under [`stimulus`]:
    /// 100 ms of nothing, 700 ms at twice the model's rheobase, 200 ms of nothing, at the fitted `dt`
    /// of 50 µs. `tools/glif_reference.py` regenerates every number.
    struct Reference {
        id: u64,
        level: u8,
        config: &'static str,
        steps: &'static [usize],
        times: &'static [f64],
        nan: usize,
        sum_v: f64,
        sum_th: f64,
        sum_asc: &'static [f64],
        first_allen_v: f64,
        first_allen_th: f64,
        /// The same model run with the `AllenSDK`'s `dynamics_voltage_linear_exact` registered in
        /// place of forward Euler.
        exact_steps: &'static [usize],
        exact_sum_v: f64,
    }

    /// Twice the rheobase `θ∞/R_input`, formed exactly as the reference script forms it, for 14,000
    /// of 20,000 steps.
    fn stimulus(r: &Reference, m: &Glif) -> Vec<f64> {
        let r_input = json::parse(r.config).unwrap().get("R_input").and_then(json::Json::as_f64).unwrap();
        let amp = 2.0 * (m.th_inf / r_input);
        let mut s = vec![0.0; 20_000];
        s[2000..16_000].fill(amp);
        s
    }

    const REFERENCE: [Reference; 5] = [
        Reference {
            id: 573430216,
            level: 1,
            config: r#"{"El_reference":-0.06620608520507815,"C":9.526774395363432e-11,"asc_amp_array":[-9.722192658577939e-11,4.433753178467453e-11],"init_threshold":0.017774745019418295,"threshold_reset_method":{"params":{},"name":"inf"},"th_inf":0.017774745019418295,"spike_cut_length":55,"init_AScurrents":[0.0,0.0],"init_voltage":0.0,"threshold_dynamics_method":{"params":{},"name":"inf"},"voltage_reset_method":{"params":{},"name":"zero"},"extrapolation_method_name":"endpoints","dt":5e-05,"voltage_dynamics_method":{"params":{},"name":"linear_forward_euler"},"El":0.0,"asc_tau_array":[0.03333333333333334,0.01],"R_input":188737719.9896,"AScurrent_dynamics_method":{"params":{},"name":"none"},"AScurrent_reset_method":{"params":{},"name":"none"},"dt_multiplier":10,"th_adapt":null,"coeffs":{"a":1,"C":1,"b":1,"G":1,"th_inf":0.8447210651216714,"asc_amp_array":[1.0,1.0]},"type":"GLIF"}"#,
            steps: &[
                2248, 2552, 2856, 3160, 3464, 3768, 4072, 4376, 4680, 4984, 5288, 5592,
                5896, 6200, 6504, 6808, 7112, 7416, 7720, 8024, 8328, 8632, 8936, 9240,
                9544, 9848, 10152, 10456, 10760, 11064, 11368, 11672, 11976, 12280, 12584, 12888,
                13192, 13496, 13800, 14104, 14408, 14712, 15016, 15320, 15624, 15928,
            ],
            times: &[
                0.11244588237055511, 0.12764588237055513, 0.14284588237055512, 0.1580458823705551, 0.17324588237055513,
                0.18844588237055512, 0.20364588237055511, 0.21884588237055513, 0.23404588237055513, 0.24924588237055512,
                0.26444588237055516, 0.27964588237055515, 0.29484588237055515, 0.31004588237055514, 0.32524588237055513,
                0.3404458823705552, 0.35564588237055517, 0.37084588237055516, 0.38604588237055515, 0.40124588237055514,
                0.41644588237055513, 0.4316458823705552, 0.44684588237055517, 0.46204588237055516, 0.47724588237055515,
                0.49244588237055514, 0.5076458823705552, 0.5228458823705552, 0.5380458823705552, 0.5532458823705552,
                0.5684458823705552, 0.5836458823705551, 0.5988458823705551, 0.6140458823705551, 0.6292458823705551,
                0.6444458823705552, 0.6596458823705552, 0.6748458823705552, 0.6900458823705552, 0.7052458823705552,
                0.7204458823705552, 0.7356458823705552, 0.7508458823705552, 0.7660458823705552, 0.7812458823705551,
                0.7964458823705551,
            ],
            nan: 2530,
            sum_v: 96.00451511146917,
            sum_th: 262.306835992358,
            sum_asc: &[],
            first_allen_v: 0.015056560825503157,
            first_allen_th: 0.015014701545069146,
            exact_steps: &[
                2249, 2554, 2859, 3164, 3469, 3774, 4079, 4384, 4689, 4994, 5299, 5604,
                5909, 6214, 6519, 6824, 7129, 7434, 7739, 8044, 8349, 8654, 8959, 9264,
                9569, 9874, 10179, 10484, 10789, 11094, 11399, 11704, 12009, 12314, 12619, 12924,
                13229, 13534, 13839, 14144, 14449, 14754, 15059, 15364, 15669, 15974,
            ],
            exact_sum_v: 96.10859692726739,
        },
        Reference {
            id: 555405421,
            level: 2,
            config: r#"{"El_reference":-0.07259513854980469,"C":6.345341236623519e-11,"asc_amp_array":[-7.429656820021222e-11,-2.116516022442624e-10],"init_threshold":0.0326222152674332,"threshold_reset_method":{"params":{"a_spike":0.0006548316938816345,"b_spike":14.09445117152747},"name":"three_components"},"th_inf":0.0326222152674332,"spike_cut_length":75,"init_AScurrents":[0.0,0.0],"init_voltage":0.0,"threshold_dynamics_method":{"params":{"b_voltage":0,"a_spike":0.0006548316938816345,"b_spike":14.09445117152747,"a_voltage":0},"name":"spike_component"},"voltage_reset_method":{"params":{"a":0.09283476501183618,"b":0.018126962285270307},"name":"v_before"},"extrapolation_method_name":"endpoints","dt":5e-05,"voltage_dynamics_method":{"params":{},"name":"linear_forward_euler"},"El":0.0,"asc_tau_array":[0.03333333333333334,0.0033333333333333335],"R_input":303459869.9717748,"AScurrent_dynamics_method":{"params":{},"name":"none"},"AScurrent_reset_method":{"params":{},"name":"none"},"dt_multiplier":10,"th_adapt":null,"coeffs":{"a":1,"C":1,"b":1,"G":1,"th_inf":1.1332218871790571,"asc_amp_array":[1.0,1.0]},"type":"GLIF"}"#,
            steps: &[
                2266, 2482, 2703, 2928, 3157, 3389, 3623, 3859, 4097, 4337, 4578, 4820,
                5063, 5307, 5551, 5796, 6041, 6286, 6532, 6778, 7024, 7270, 7516, 7762,
                8008, 8255, 8502, 8749, 8996, 9243, 9490, 9737, 9984, 10231, 10478, 10725,
                10972, 11219, 11466, 11713, 11960, 12207, 12454, 12701, 12948, 13195, 13442, 13689,
                13936, 14183, 14430, 14677, 14924, 15171, 15418, 15665, 15912,
            ],
            times: &[
                0.11332961959015339, 0.12411006215088276, 0.1351577731547648, 0.14642129454039893, 0.1578542120956191,
                0.16945701870578445, 0.1811866921396019, 0.19299736184006536, 0.20488897460115435, 0.21686394116166965,
                0.22892456335463104, 0.24102681206663332, 0.25316957450121413, 0.2653541111643738, 0.27758167575457426,
                0.2898073149244307, 0.3020750317408144, 0.31434221366563914, 0.32660681786325046, 0.3389150600244105,
                0.3512242540938482, 0.36353211767120014, 0.3758387362953508, 0.3881443009765979, 0.4004489792991225,
                0.41275291993780394, 0.42510214598404655, 0.43745375300613243, 0.4498052372271151, 0.4621564922354141,
                0.47450754718262234, 0.4868584335963357, 0.49920917838206275, 0.5115598041696118, 0.5239103299734681,
                0.5362607717694253, 0.5486111429802686, 0.5609614548838524, 0.5733117169559253, 0.5856619371581407,
                0.5980121221800234, 0.6103622776422574, 0.6227124082674802, 0.635062518023782, 0.6474126102452692,
                0.6597626877333623, 0.6721127528419077, 0.6844628075486875, 0.696812853515504, 0.7091628921386638,
                0.7215129245913954, 0.7338629518594895, 0.7462129747712453, 0.7585629940226312, 0.7709130101984277,
                0.7832630237899902, 0.795613035210175,
            ],
            nan: 4275,
            sum_v: 314.65314468681896,
            sum_th: 620.3825721146105,
            sum_asc: &[],
            first_allen_v: 0.037064349810698304,
            first_allen_th: 0.036968208349322094,
            exact_steps: &[
                2266, 2482, 2703, 2928, 3157, 3389, 3623, 3860, 4098, 4338, 4579, 4821,
                5064, 5308, 5552, 5797, 6042, 6288, 6534, 6780, 7026, 7272, 7518, 7765,
                8012, 8259, 8506, 8753, 9000, 9247, 9494, 9741, 9988, 10235, 10482, 10729,
                10976, 11223, 11470, 11717, 11964, 12211, 12458, 12705, 12952, 13199, 13446, 13693,
                13940, 14187, 14434, 14681, 14928, 15175, 15422, 15669, 15916,
            ],
            exact_sum_v: 314.40853058482804,
        },
        Reference {
            id: 482525253,
            level: 3,
            config: r#"{"El_reference":-0.06740524864196776,"C":1.4889066567300852e-10,"asc_amp_array":[-6.493692083311101e-10,1.224690033604069e-09],"init_threshold":0.02238599727026381,"threshold_reset_method":{"params":{},"name":"inf"},"th_inf":0.02238599727026381,"spike_cut_length":60,"init_AScurrents":[0.0,0.0],"init_voltage":0.0,"threshold_dynamics_method":{"params":{},"name":"inf"},"voltage_reset_method":{"params":{},"name":"zero"},"extrapolation_method_name":"endpoints","dt":5e-05,"voltage_dynamics_method":{"params":{},"name":"linear_forward_euler"},"El":0.0,"asc_tau_array":[0.01,0.0033333333333333335],"R_input":126083105.3864096,"AScurrent_dynamics_method":{"params":{},"name":"exp"},"AScurrent_reset_method":{"params":{"r":[1.0,1.0]},"name":"sum"},"dt_multiplier":10,"th_adapt":null,"coeffs":{"a":1,"C":1,"b":1,"G":1,"th_inf":1.1792290700522072,"asc_amp_array":[1.0,1.0]},"type":"GLIF"}"#,
            steps: &[
                2259, 2716, 3218, 3715, 4213, 4711, 5209, 5707, 6205, 6703, 7201, 7699,
                8197, 8695, 9193, 9691, 10189, 10687, 11185, 11683, 12181, 12679, 13177, 13675,
                14173, 14671, 15169, 15667,
            ],
            times: &[
                0.1129948428127597, 0.13583726312927794, 0.16093991766338675, 0.1857885718300385, 0.21066448881491517,
                0.2355571967003195, 0.2604565836221217, 0.28535653278615075, 0.31025652857133307, 0.33515652822188263,
                0.36005652819290973, 0.3849565281905076, 0.4098565281903084, 0.4347565281902919, 0.4596565281902905,
                0.48455652819029044, 0.5094565281902904, 0.5343565281902903, 0.5592565281902904, 0.5841565281902904,
                0.6090565281902903, 0.6339565281902904, 0.6588565281902904, 0.6837565281902904, 0.7086565281902903,
                0.7335565281902904, 0.7584565281902904, 0.7833565281902903,
            ],
            nan: 1680,
            sum_v: 217.18138004158396,
            sum_th: 483.61536737550534,
            sum_asc: &[-3.5338808935091564e-06, 2.3013727589122805e-06],
            first_allen_v: 0.02646869761052359,
            first_allen_th: 0.02639821874320444,
            exact_steps: &[
                2260, 2718, 3220, 3718, 4216, 4714, 5212, 5710, 6208, 6706, 7204, 7702,
                8200, 8698, 9196, 9694, 10192, 10690, 11188, 11686, 12184, 12682, 13180, 13678,
                14176, 14674, 15172, 15670,
            ],
            exact_sum_v: 216.92696035513674,
        },
        Reference {
            id: 484633260,
            level: 4,
            config: r#"{"El_reference":-0.06578562164306644,"C":3.893405651475474e-11,"asc_amp_array":[-2.7836214531817965e-12,-6.596756809480476e-11],"init_threshold":0.03156837581616369,"threshold_reset_method":{"params":{"a_spike":0.002106683597405399,"b_spike":61.38275752539194},"name":"three_components"},"th_inf":0.03156837581616369,"spike_cut_length":73,"init_AScurrents":[0.0,0.0],"init_voltage":0.0,"threshold_dynamics_method":{"params":{"b_voltage":0,"a_spike":0.002106683597405399,"b_spike":61.38275752539194,"a_voltage":0},"name":"spike_component"},"voltage_reset_method":{"params":{"a":0.941716610377029,"b":-0.012842001254528689},"name":"v_before"},"extrapolation_method_name":"endpoints","dt":5e-05,"voltage_dynamics_method":{"params":{},"name":"linear_forward_euler"},"El":0.0,"asc_tau_array":[0.3333333333333333,0.01],"R_input":322067320.75941676,"AScurrent_dynamics_method":{"params":{},"name":"exp"},"AScurrent_reset_method":{"params":{"r":[1.0,1.0]},"name":"sum"},"dt_multiplier":10,"th_adapt":null,"coeffs":{"a":1,"C":1,"b":1,"G":1,"th_inf":0.7937143801724502,"asc_amp_array":[1.0,1.0]},"type":"GLIF"}"#,
            steps: &[
                2173, 2464, 2788, 3122, 3463, 3812, 4167, 4529, 4898, 5273, 5654, 6041,
                6434, 6833, 7237, 7646, 8060, 8479, 8902, 9329, 9760, 10194, 10631, 11071,
                11514, 11960, 12408, 12858, 13310, 13764, 14220, 14677, 15135, 15595,
            ],
            times: &[
                0.10867432566427804, 0.12322021423202882, 0.13944900126366094, 0.15614255879146113, 0.17319887306365564,
                0.19060154872026436, 0.20837401983767304, 0.22647419776129754, 0.24490466297683985, 0.26366913046952656,
                0.2827392220197176, 0.3020999710733404, 0.32174538663630686, 0.34167302141557954, 0.36188168413049754,
                0.38233967529230745, 0.4030332624920652, 0.42395806922046764, 0.44511291222580845, 0.4664655041064109,
                0.4880034400944706, 0.5097234609949256, 0.5315923894675799, 0.5535984729386727, 0.5757389431506863,
                0.5980135501764569, 0.6204227107628878, 0.6429336762910671, 0.6655356177446423, 0.6882262191167046,
                0.7110053836613116, 0.7338735639267872, 0.7567975318914678, 0.779766808792014,
            ],
            nan: 2482,
            sum_v: 181.78569515405212,
            sum_th: 460.00045557661963,
            sum_asc: &[-4.329407055784738e-07, -4.1980776832669426e-07],
            first_allen_v: 0.025156378320636566,
            first_allen_th: 0.02505627384397733,
            exact_steps: &[
                2173, 2464, 2789, 3123, 3465, 3813, 4169, 4531, 4900, 5275, 5657, 6045,
                6438, 6837, 7242, 7651, 8065, 8484, 8907, 9334, 9765, 10199, 10637, 11078,
                11521, 11967, 12415, 12866, 13318, 13772, 14228, 14685, 15144, 15604,
            ],
            exact_sum_v: 181.56615018881843,
        },
        Reference {
            id: 566367589,
            level: 5,
            config: r#"{"El_reference":-0.0707990016937256,"C":7.0249928245181e-11,"asc_amp_array":[-5.576151112965266e-11,2.7578452288240807e-10],"init_threshold":0.02155961135960361,"threshold_reset_method":{"params":{"a_spike":0.0013219642653143962,"b_spike":9.385277572139637},"name":"three_components"},"th_inf":0.02155961135960361,"spike_cut_length":33,"init_AScurrents":[0.0,0.0],"init_voltage":0.0,"threshold_dynamics_method":{"params":{"b_voltage":39.74345813991383,"a_spike":0.0013219642653143962,"b_spike":9.385277572139637,"a_voltage":3.30794960492179},"name":"three_components_exact"},"voltage_reset_method":{"params":{"a":0.25329692732471387,"b":0.0034819750195955327},"name":"v_before"},"extrapolation_method_name":"endpoints","dt":5e-05,"voltage_dynamics_method":{"params":{},"name":"linear_forward_euler"},"El":0.0,"asc_tau_array":[0.01,0.0033333333333333335],"R_input":711661985.1607032,"AScurrent_dynamics_method":{"params":{},"name":"exp"},"AScurrent_reset_method":{"params":{"r":[1.0,1.0]},"name":"sum"},"dt_multiplier":10,"th_adapt":null,"coeffs":{"a":1,"C":1,"b":1,"G":1,"th_inf":0.917465684832131,"asc_amp_array":[1.0,1.0]},"type":"GLIF"}"#,
            steps: &[
                2739, 3097, 3602, 4128, 4682, 5255, 5842, 6438, 7040, 7647, 8256, 8868,
                9481, 10095, 10709, 11324, 11939, 12554, 13169, 13784, 14399, 15015, 15630,
            ],
            times: &[
                0.13695146040907885, 0.1548818817461126, 0.18014279319415596, 0.20642999070759355, 0.2341357192140623,
                0.2627866865879888, 0.2921082352034699, 0.3219139689017254, 0.3520339221912597, 0.38235579850161766,
                0.41284355009732776, 0.44340353218702144, 0.47405770927757, 0.504753506834572, 0.5354814757517388,
                0.5662094860404991, 0.5969663233770248, 0.6277274936783588, 0.6584866901855905, 0.6892436470423433,
                0.719998851033356, 0.7507527438288045, 0.781540631757312,
            ],
            nan: 759,
            sum_v: 296.4803185134748,
            sum_th: 466.01056034906816,
            sum_asc: &[-2.5451153806392535e-07, 4.259566633315773e-07],
            first_allen_v: 0.02069474554103052,
            first_allen_th: 0.02067749810098918,
            exact_steps: &[
                2739, 3098, 3603, 4129, 4684, 5257, 5844, 6440, 7043, 7650, 8260, 8872,
                9485, 10099, 10713, 11328, 11943, 12558, 13174, 13789, 14405, 15021, 15637,
            ],
            exact_sum_v: 296.3293437156176,
        },
    ];

    /// Neumaier's compensated sum of the finite samples, so that comparing against the reference's
    /// exact `math.fsum` measures the traces and not the summation.
    fn sum_finite(xs: &[f64]) -> f64 {
        let (mut s, mut c) = (0.0_f64, 0.0_f64);
        for &x in xs.iter().filter(|x| !x.is_nan()) {
            let t = s + x;
            c += if s.abs() >= x.abs() { (s - t) + x } else { (x - t) + s };
            s = t;
        }
        s + c
    }

    fn scale(xs: &[f64]) -> f64 {
        xs.iter().filter(|x| !x.is_nan()).map(|x| x.abs()).sum()
    }

    /// Every level, read from its served config, reproduces the `AllenSDK`'s own run.
    ///
    /// Same level, the same spike on the same step every time, every interpolated spike time within
    /// `10⁻¹²` s, the same samples cut, and each whole trace — voltage, threshold, every after-spike
    /// current — summing to the reference's sum within `10⁻¹²` of the sum of magnitudes, so that no
    /// sample anywhere can be off by more than about `10⁻¹⁰` of the trace's scale without failing.
    #[test]
    fn every_level_reproduces_the_allensdk_run() {
        for r in &REFERENCE {
            let m = Glif::from_neuron_config(r.config).unwrap();
            assert_eq!(m.level(), Some(r.level), "{}", r.id);
            assert_eq!(m.integrator, Integrator::ForwardEuler);
            let run = m.run(&stimulus(r, &m)).unwrap();
            assert_eq!(run.spikes.iter().map(|s| s.step).collect::<Vec<_>>(), r.steps, "{}", r.id);
            for (s, &t) in run.spikes.iter().zip(r.times) {
                assert!((s.time - t).abs() <= 1e-12, "{}: {} against {t}", r.id, s.time);
            }
            assert_eq!(run.voltage.iter().filter(|v| v.is_nan()).count(), r.nan, "{}", r.id);
            assert_eq!(run.threshold.iter().filter(|v| v.is_nan()).count(), r.nan, "{}", r.id);
            assert!((sum_finite(&run.voltage) - r.sum_v).abs() <= 1e-12 * scale(&run.voltage), "{}", r.id);
            assert!((sum_finite(&run.threshold) - r.sum_th).abs() <= 1e-12 * scale(&run.threshold), "{}", r.id);
            assert_eq!(run.asc.len(), r.sum_asc.len(), "{}", r.id);
            for (trace, &want) in run.asc.iter().zip(r.sum_asc) {
                assert!((sum_finite(trace) - want).abs() <= 1e-12 * scale(trace), "{}: {} against {want}", r.id, sum_finite(trace));
            }
            assert_eq!(run.bad_reset, None);
        }
    }

    /// The `AllenSDK`'s interpolated spike voltage is the crossing plus one whole step's rise.
    ///
    /// At a crossing the voltage line and the threshold line are equal, and this module's spikes
    /// report them equal to rounding. The `AllenSDK` evaluates both lines one step past the crossing,
    /// so its voltage is ours plus `v₁ − v₀` and its threshold ours plus `θ₁ − θ₀` — which the
    /// reference's first spike on every level reproduces to rounding, and which put its two
    /// "crossing" values 17 to 100 µV apart.
    #[test]
    fn the_allensdk_spike_voltage_is_one_step_past_the_crossing() {
        for r in &REFERENCE {
            let m = Glif::from_neuron_config(r.config).unwrap();
            let s = m.run(&stimulus(r, &m)).unwrap().spikes[0];
            assert!((s.voltage - s.threshold).abs() <= 1e-15, "{}: {} against {}", r.id, s.voltage, s.threshold);
            assert!((r.first_allen_v - (s.voltage + (s.v1 - s.v0))).abs() <= 1e-15, "{}", r.id);
            assert!((r.first_allen_th - (s.threshold + (s.th1 - s.th0))).abs() <= 1e-15, "{}", r.id);
            let gap = r.first_allen_v - r.first_allen_th;
            assert!((1.7e-5..1.01e-4).contains(&gap), "{}: the reference's two crossing values differ by {gap}", r.id);
            assert!(s.v0 <= s.th0 && s.v1 > s.th1, "the step brackets the crossing");
        }
    }

    /// The exact integrator is the `AllenSDK`'s own unregistered `dynamics_voltage_linear_exact`, and
    /// it is what Euler approximates.
    ///
    /// With it registered, the reference fires the same number of spikes on every level, each on the
    /// step this module's [`Integrator::Exact`] puts it on. Forward Euler at 50 µs fires earlier:
    /// every first spike by about a third of a step, and inter-spike intervals 0.02 % to 0.33 %
    /// shorter across the five models.
    #[test]
    fn the_exact_integrator_is_the_allensdks_unregistered_one() {
        for r in &REFERENCE {
            let mut m = Glif::from_neuron_config(r.config).unwrap();
            m.integrator = Integrator::Exact;
            let run = m.run(&stimulus(r, &m)).unwrap();
            assert_eq!(run.spikes.iter().map(|s| s.step).collect::<Vec<_>>(), r.exact_steps, "{}", r.id);
            assert!((sum_finite(&run.voltage) - r.exact_sum_v).abs() <= 1e-12 * scale(&run.voltage), "{}", r.id);
            assert_eq!(r.exact_steps.len(), r.steps.len());
            assert!(r.exact_steps[0] >= r.steps[0] && r.exact_steps.last() > r.steps.last(), "{}", r.id);
        }
    }

    /// Below threshold the exact integrator IS the membrane's solution: every sample is
    /// `β + (V₀ − β)e^{−g t/C}`, to rounding, at every step of a long run.
    #[test]
    fn below_threshold_the_exact_integrator_is_the_closed_form() {
        let mut m = Glif::from_neuron_config(REFERENCE[0].config).unwrap();
        m.integrator = Integrator::Exact;
        // Every fitted model has `E_L = 0`, which would hide a leak toward zero instead of toward `E_L`.
        m.e_l = -0.004;
        let i = 0.5 * m.th_inf * m.g;
        let run = m.run(&vec![i; 4000]).unwrap();
        assert!(run.spikes.is_empty());
        let beta = m.e_l + i / m.g;
        for (t, &v) in run.voltage.iter().enumerate() {
            let want = beta + (m.init_voltage - beta) * (-(m.g / m.c) * m.dt * (t + 1) as f64).exp();
            assert!((v - want).abs() <= 1e-15, "step {t}: {v} against {want}");
        }
        // And forward Euler is not. Its fixed point is the same `β`; its approach is not: 3.8 µV
        // apart at worst on this model, 1.7 nV after eleven membrane time constants.
        m.integrator = Integrator::ForwardEuler;
        let euler = m.run(&vec![i; 4000]).unwrap();
        let worst = euler.voltage.iter().zip(&run.voltage).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        assert!(worst > 1e-6 && worst < 1e-4, "{worst}");
        let last = (euler.voltage[3999] - run.voltage[3999]).abs();
        assert!(last < 1e-3 * worst, "the gap closes as both settle on the same steady state: {last}");
    }

    /// The voltage component of the threshold at and near its removable singularities, against
    /// values computed to 60 digits.
    ///
    /// Parameters from the level-5 reference model; `θ₀ = 1 mV`, `V₀ = 15 mV`, one step of 50 µs.
    /// At `b_v = g/C` exactly the `AllenSDK` raises `ZeroDivisionError`; at `g/C·(1 + 10⁻¹²)` its value
    /// is wrong in the fifth significant digit (relative error 1.0 × 10⁻⁴), and at `b_v = 10⁻⁹` in
    /// the sixth (7.5 × 10⁻⁶). Here all four agree with the high-precision value to a few units in
    /// the last place.
    #[test]
    fn the_voltage_threshold_is_exact_through_its_singularities() {
        let (a, c, g, i) = (3.30794960492179, 7.0249928245181e-11, 1.4051614682976022e-09, 5.5588759869720094e-11);
        let k = g / c;
        assert_eq!(k, 20.002318911891464);
        let cases = [
            (k, 0.0010014821359922708),
            (k * (1.0 + 1e-12), 0.0010014821359922697),
            (0.0, 0.0010024829928768967),
            (1e-9, 0.0010024829928768466),
        ];
        for (b, want) in cases {
            let got = threshold_voltage_component(1e-3, 0.015, i, 5e-5, a, b, c, g, 0.0);
            assert!(((got - want) / want).abs() < 1e-15, "b = {b}: {got} against {want}");
        }
    }

    /// The voltage component solves the ODE it claims to: `dθ/dt = a(V − E_L) − bθ` along the
    /// membrane's own exponential, integrated here by fine Runge–Kutta steps, for decay rates below,
    /// at and above the membrane's.
    #[test]
    fn the_voltage_threshold_solves_its_ode() {
        let (c, g, e_l, i, v0, th0, a) = (1e-10, 5e-9, -0.002, 1.2e-10, 0.004, 0.003, 2.5);
        let k = g / c;
        let beta = e_l + i / g;
        let v = |s: f64| beta + (v0 - beta) * (-k * s).exp();
        for b in [0.0, 7.0, k, 90.0] {
            let f = |s: f64, th: f64| a * (v(s) - e_l) - b * th;
            let (mut th, h, n) = (th0, 1e-5, 2000);
            for step in 0..n {
                let s = step as f64 * h;
                let k1 = f(s, th);
                let k2 = f(s + h / 2.0, th + h / 2.0 * k1);
                let k3 = f(s + h / 2.0, th + h / 2.0 * k2);
                let k4 = f(s + h, th + h * k3);
                th += h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
            }
            let got = threshold_voltage_component(th0, v0, i, n as f64 * h, a, b, c, g, e_l);
            assert!((got - th).abs() < 1e-14, "b = {b}: {got} against {th}");
        }
    }

    /// A config is read as the `AllenSDK` reads it, and one it could not run is refused, naming why.
    #[test]
    fn a_config_the_allensdk_could_not_run_is_refused() {
        let base = REFERENCE[4].config;
        let with = |from: &str, to: &str| {
            assert_eq!(base.matches(from).count(), 1, "{from}");
            Glif::from_neuron_config(&base.replace(from, to)).unwrap_err().to_string()
        };
        assert_eq!(with("\"R_input\":", "\"R_inpt\":"), "the config has no usable R_input");
        assert_eq!(
            Glif::from_neuron_config("[1"),
            Err(GlifError::Json(JsonError { offset: 2, what: "an array element followed by neither ',' nor ']'" }))
        );
        assert_eq!(Glif::from_neuron_config("{}"), Err(GlifError::Missing { key: "R_input" }));
        assert_eq!(
            with("\"linear_forward_euler\"", "\"linear_exact\""),
            "voltage_dynamics_method \"linear_exact\" is not a method the AllenSDK registers"
        );
        assert_eq!(
            with("\"name\":\"exp\"", "\"name\":\"expo\""),
            "AScurrent_dynamics_method \"expo\" is not a method the AllenSDK registers"
        );
        assert_eq!(
            with("\"name\":\"sum\"", "\"name\":\"none\""),
            "after-spike current dynamics and reset methods that no GLIF level pairs"
        );
        assert_eq!(
            with("\"name\":\"sum\"", "\"name\":\"total\""),
            "AScurrent_reset_method \"total\" is not a method the AllenSDK registers"
        );
        assert_eq!(
            with("\"name\":\"v_before\"", "\"name\":\"v_after\""),
            "voltage_reset_method \"v_after\" is not a method the AllenSDK registers"
        );
        assert_eq!(
            with("\"name\":\"three_components_exact\"", "\"name\":\"inf\""),
            "threshold dynamics and reset methods that no GLIF level pairs"
        );
        assert_eq!(
            with("\"name\":\"three_components_exact\"", "\"name\":\"three\""),
            "threshold_dynamics_method \"three\" is not a method the AllenSDK registers"
        );
        assert_eq!(
            with("\"name\":\"three_components\"}", "\"name\":\"three\"}"),
            "threshold_reset_method \"three\" is not a method the AllenSDK registers"
        );
        assert_eq!(with("\"spike_cut_length\":33", "\"spike_cut_length\":33.5"), "spike_cut_length is not a whole number of steps");
        assert_eq!(with("\"spike_cut_length\":33", "\"spike_cut_length\":-1"), "spike_cut_length is not a whole number of steps");
        assert_eq!(with("\"C\":7.0249928245181e-11", "\"C\":0"), "C = 0 must be finite and positive");
        assert_eq!(with("\"r\":[1.0,1.0]", "\"r\":[1.0]"), "the after-spike current arrays differ in length");
        assert_eq!(
            with("\"asc_tau_array\":[0.01,0.0033333333333333335]", "\"asc_tau_array\":[0.01]"),
            "the after-spike current arrays differ in length"
        );
        assert_eq!(
            with("\"asc_amp_array\":[1.0,1.0]", "\"asc_amp_array\":[1.0]"),
            "the after-spike current arrays differ in length"
        );
        assert_eq!(with("\"r\":[1.0,1.0]", "\"r\":1.0"), "the config has no usable AScurrent_reset_method.params.r");
        assert_eq!(with("\"init_AScurrents\":[0.0,0.0]", "\"init_AScurrents\":[0.0]"), "init_AScurrents and the after-spike currents differ in length");
        assert!(with("\"El\":0.0", "\"El\":0.0,").starts_with("not JSON at byte"));
        // The spike component is one mechanism, written twice: the copies must agree.
        let b_reset = "\"threshold_reset_method\":{\"params\":{\"a_spike\":0.0013219642653143962,\"b_spike\":9.385277572139637}";
        assert_eq!(
            with(b_reset, &b_reset.replace("9.385277572139637", "9.4")),
            "a_spike or b_spike differs between the threshold dynamics and reset"
        );
        assert_eq!(
            with(b_reset, &b_reset.replace("0.0013219642653143962", "0.0013")),
            "a_spike or b_spike differs between the threshold dynamics and reset"
        );
        assert_eq!(with("\"a_voltage\":3.30794960492179", "\"a_volt\":3.30794960492179"), "the config has no usable threshold_dynamics_method.params.a_voltage");
        // A level-1 model with a non-zero starting current would carry it for exactly one step.
        let lif = REFERENCE[0].config;
        assert_eq!(lif.matches("\"init_AScurrents\":[0.0,0.0]").count(), 1);
        assert_eq!(
            Glif::from_neuron_config(&lif.replace("\"init_AScurrents\":[0.0,0.0]", "\"init_AScurrents\":[0.0,1e-12]")).unwrap_err().to_string(),
            "init_AScurrents is non-zero in a model without after-spike currents"
        );
    }

    /// Coefficients multiply what the `AllenSDK` multiplies, to the bit, and a missing one is 1.
    #[test]
    fn coefficients_are_applied_as_the_allensdk_applies_them() {
        let r = &REFERENCE[4];
        let m = Glif::from_neuron_config(r.config).unwrap();
        assert_eq!(m.th_inf, 0.02155961135960361 * 0.917465684832131);
        assert_eq!(m.g, 1.0 / 711661985.1607032);
        assert_eq!(m.c, 7.0249928245181e-11);
        assert_eq!(m.spike_cut, 33);
        assert_eq!(m.init_threshold, 0.02155961135960361, "the starting threshold is NOT scaled");
        assert_eq!(m.dt, 5e-5);
        assert_eq!(m.e_l_reference, -0.0707990016937256);
        assert_eq!(m.voltage_reset, VoltageReset::Scaled { a: 0.25329692732471387, b: 0.0034819750195955327 });
        assert_eq!(
            m.threshold,
            Threshold::SpikeAndVoltage { a_spike: 0.0013219642653143962, b_spike: 9.385277572139637, a_voltage: 3.30794960492179, b_voltage: 39.74345813991383 }
        );
        assert_eq!(m.asc[1], AfterSpikeCurrent { tau: 0.0033333333333333335, amp: 2.7578452288240807e-10, r: 1.0 });
        let scaled = r
            .config
            .replace("\"a\":1,", "\"a\":2,")
            .replace("\"b\":1,", "\"b\":0.5,")
            .replace("\"C\":1,", "\"C\":3,")
            .replace("\"G\":1,", "\"G\":0.25,")
            .replace("\"asc_amp_array\":[1.0,1.0]", "\"asc_amp_array\":[1.0,4.0]");
        let s = Glif::from_neuron_config(&scaled).unwrap();
        assert_eq!(s.c, 7.0249928245181e-11 * 3.0);
        assert_eq!(s.g, 1.0 / 711661985.1607032 * 0.25);
        assert_eq!(s.asc[1].amp, 2.7578452288240807e-10 * 4.0);
        assert_eq!(s.asc[0].amp, m.asc[0].amp);
        let Threshold::SpikeAndVoltage { a_voltage, b_voltage, .. } = s.threshold else { panic!() };
        assert_eq!((a_voltage, b_voltage), (3.30794960492179 * 2.0, 39.74345813991383 * 0.5));
        let bare = r.config.split(",\"coeffs\"").next().unwrap().to_owned() + ",\"type\":\"GLIF\"}";
        let b = Glif::from_neuron_config(&bare).unwrap();
        assert_eq!(b.th_inf, 0.02155961135960361, "no coeffs: every coefficient is 1");
        assert_eq!(b.asc[1].amp, 2.7578452288240807e-10);
    }

    /// Every parameter is checked before a run, and so is every stimulus sample.
    #[test]
    fn every_parameter_and_every_sample_is_checked() {
        let m = Glif::from_neuron_config(REFERENCE[4].config).unwrap();
        let bad = |f: &dyn Fn(&mut Glif)| {
            let mut b = m.clone();
            f(&mut b);
            b.run(&[0.0]).unwrap_err().to_string()
        };
        assert_eq!(bad(&|b| b.dt = 0.0), "dt = 0 must be finite and positive");
        assert_eq!(bad(&|b| b.c = f64::INFINITY), "C = inf must be finite and positive");
        assert_eq!(bad(&|b| b.g = -1.0), "g = -1 must be finite and positive");
        assert_eq!(bad(&|b| b.e_l = f64::NAN), "El = NaN is not finite");
        assert_eq!(bad(&|b| b.e_l_reference = f64::NAN), "El_reference = NaN is not finite");
        assert_eq!(bad(&|b| b.th_inf = f64::NAN), "th_inf = NaN is not finite");
        assert_eq!(bad(&|b| b.init_voltage = f64::NAN), "init_voltage = NaN is not finite");
        assert_eq!(bad(&|b| b.init_threshold = f64::NAN), "init_threshold = NaN is not finite");
        assert_eq!(bad(&|b| b.asc[1].tau = 0.0), "asc tau = 0 must be finite and positive");
        assert_eq!(bad(&|b| b.asc[0].amp = f64::NAN), "asc amp = NaN is not finite");
        assert_eq!(bad(&|b| b.asc[1].r = f64::NAN), "asc r = NaN is not finite");
        assert_eq!(bad(&|b| b.init_asc[1] = f64::NAN), "init_AScurrents = NaN is not finite");
        assert_eq!(bad(&|b| { b.init_asc.pop(); }), "init_AScurrents and the after-spike currents differ in length");
        assert_eq!(bad(&|b| b.voltage_reset = VoltageReset::Scaled { a: f64::NAN, b: 0.0 }), "reset a = NaN is not finite");
        assert_eq!(bad(&|b| b.voltage_reset = VoltageReset::Scaled { a: 1.0, b: f64::NAN }), "reset b = NaN is not finite");
        assert_eq!(bad(&|b| b.threshold = Threshold::Spike { a_spike: f64::NAN, b_spike: 1.0 }), "a_spike = NaN is not finite");
        assert_eq!(bad(&|b| b.threshold = Threshold::Spike { a_spike: 0.0, b_spike: f64::NAN }), "b_spike = NaN is not finite");
        let full = |a_spike, b_spike, a_voltage, b_voltage| Threshold::SpikeAndVoltage { a_spike, b_spike, a_voltage, b_voltage };
        assert_eq!(bad(&|b| b.threshold = full(f64::NAN, 1.0, 1.0, 1.0)), "a_spike = NaN is not finite");
        assert_eq!(bad(&|b| b.threshold = full(0.0, f64::NAN, 1.0, 1.0)), "b_spike = NaN is not finite");
        assert_eq!(bad(&|b| b.threshold = full(0.0, 1.0, f64::NAN, 1.0)), "a_voltage = NaN is not finite");
        assert_eq!(bad(&|b| b.threshold = full(0.0, 1.0, 1.0, f64::INFINITY)), "b_voltage = inf is not finite");
        assert_eq!(m.run(&[0.0, f64::NAN]).unwrap_err().to_string(), "stimulus = NaN is not finite");
        assert!(m.check().is_ok());
    }

    /// The level follows the mechanisms, and a combination no level uses has none.
    #[test]
    fn the_level_follows_the_mechanisms() {
        let levels: Vec<Option<u8>> =
            REFERENCE.iter().map(|r| Glif::from_neuron_config(r.config).unwrap().level()).collect();
        assert_eq!(levels, [Some(1), Some(2), Some(3), Some(4), Some(5)]);
        let mut odd = Glif::from_neuron_config(REFERENCE[4].config).unwrap();
        odd.voltage_reset = VoltageReset::Zero;
        assert_eq!(odd.level(), None, "a voltage-adapting threshold with a reset to rest");
        odd.asc.clear();
        odd.init_asc.clear();
        odd.voltage_reset = VoltageReset::Scaled { a: 1.0, b: 0.0 };
        assert_eq!(odd.level(), None, "a voltage-adapting threshold without after-spike currents");
    }

    /// The spike cut and the reset, sample by sample, on a hand-built level-4 model.
    ///
    /// A spike on step `t` leaves `t .. t + cut` empty and the reset state at `t + cut`. The reset
    /// voltage is `a·V₁ + b`; each current is `amp + r·I₁·e^{−cut·dt/τ}`; the spike component has
    /// decayed through the cut and jumped by `a_spike`. With `cut = 0` the reset lands on the spike's
    /// own sample, and a spike too close to the end leaves the tail empty with no reset written.
    #[test]
    fn the_cut_and_the_reset_land_where_the_allensdk_puts_them() {
        let m = Glif {
            dt: 1e-4,
            e_l: 0.0,
            e_l_reference: -0.07,
            c: 1e-10,
            g: 1e-8,
            th_inf: 0.01,
            init_voltage: 0.0,
            init_threshold: 0.01,
            init_asc: vec![0.0],
            asc: vec![AfterSpikeCurrent { tau: 0.02, amp: -1e-11, r: 0.5 }],
            voltage_reset: VoltageReset::Scaled { a: 0.5, b: -0.001 },
            threshold: Threshold::Spike { a_spike: 0.002, b_spike: 50.0 },
            spike_cut: 3,
            integrator: Integrator::ForwardEuler,
        };
        let stim = vec![3e-10; 400];
        let run = m.run(&stim).unwrap();
        let first = run.spikes[0];
        let t = first.step;
        assert!(run.voltage[t..t + 3].iter().all(|v| v.is_nan()) && run.asc[0][t..t + 3].iter().all(|v| v.is_nan()));
        assert_eq!(run.voltage[t + 3], 0.5 * first.v1 - 0.001);
        assert_eq!(run.threshold[t + 3], 0.002 + 0.01, "a first spike's component starts from zero");
        assert_eq!(run.asc[0][t + 3], -1e-11, "no current before the first spike");
        assert!(run.voltage[t - 1].is_finite() && run.voltage[t + 4].is_finite());
        // The second spike carries the first's current and threshold through its own cut.
        let second = run.spikes[1];
        let u = second.step;
        let i1 = run.asc[0][u - 1] * (-(1.0 / 0.02) * 1e-4_f64).exp();
        assert!((run.asc[0][u + 3] - (-1e-11 + i1 * 0.5 * (-(1.0 / 0.02) * 1e-4 * 3.0_f64).exp())).abs() < 1e-24);
        let s1 = (run.threshold[u - 1] - 0.01) * (-50.0 * 1e-4_f64).exp();
        assert!((run.threshold[u + 3] - (s1 * (-50.0 * (3.0 * 1e-4_f64)).exp() + 0.002 + 0.01)).abs() < 1e-17);
        // No cut: the reset state lands on the spike's own sample.
        let uncut = Glif { spike_cut: 0, ..m.clone() }.run(&stim).unwrap();
        let s = uncut.spikes[0];
        assert_eq!(uncut.voltage[s.step], 0.5 * s.v1 - 0.001);
        assert!(uncut.voltage.iter().all(|v| v.is_finite()));
        // A spike whose cut runs past the end: the tail is empty and nothing is written after it.
        let short = &stim[..t + 2];
        let tail = m.run(short).unwrap();
        assert_eq!(tail.spikes.len(), 1);
        assert!(tail.voltage[t].is_nan() && tail.voltage[t + 1].is_nan());
    }

    /// Both of the `AllenSDK`'s comparisons are strict: a voltage that lands exactly ON the threshold
    /// has not crossed it, and a reset exactly on the threshold is not a bad reset.
    ///
    /// Built in powers of two so the equalities are exact: one Euler step from rest under
    /// `I·dt/C = 0.25 × 0.5 / 1` lands on `θ∞ = 0.125` to the bit.
    #[test]
    fn landing_on_the_threshold_is_not_crossing_it() {
        let m = Glif {
            dt: 0.5,
            e_l: 0.0,
            e_l_reference: 0.0,
            c: 1.0,
            g: 1.0,
            th_inf: 0.125,
            init_voltage: 0.0,
            init_threshold: 0.125,
            init_asc: vec![],
            asc: vec![],
            voltage_reset: VoltageReset::Zero,
            threshold: Threshold::Fixed,
            spike_cut: 0,
            integrator: Integrator::ForwardEuler,
        };
        let run = m.run(&[0.25]).unwrap();
        assert_eq!(run.voltage, [0.125]);
        assert!(run.spikes.is_empty());
        // A reset to exactly the threshold, `V ← 0·V₁ + θ∞`, is not above it, so the run goes on.
        let on = Glif { voltage_reset: VoltageReset::Scaled { a: 0.0, b: 0.125 }, ..m.clone() };
        let run = on.run(&[1.0, 0.0, 0.0]).unwrap();
        assert_eq!(run.spikes.len(), 1);
        assert_eq!(run.bad_reset, None);
        assert_eq!(run.voltage[0], 0.125, "the reset state, on the spike's own sample");
        assert!(run.voltage[1..].iter().all(|v| v.is_finite()), "and the run went on: {:?}", run.voltage);
    }

    /// A spike on the very first step is interpolated from the configured starting threshold, which
    /// in the fitted models is the unscaled `th_inf`.
    #[test]
    fn a_first_step_spike_starts_from_the_configured_threshold() {
        let mut m = Glif::from_neuron_config(REFERENCE[0].config).unwrap();
        m.init_voltage = 0.99 * m.th_inf;
        let run = m.run(&[1e-8]).unwrap();
        assert_eq!(run.spikes[0].step, 0);
        assert_eq!(run.spikes[0].th0, 0.017774745019418295);
        assert_ne!(m.init_threshold, m.th_inf);
    }

    /// A reset above the threshold stops the run, as the `AllenSDK` stops it: five samples of the reset
    /// state after the cut, then nothing.
    #[test]
    fn a_reset_above_threshold_stops_the_run() {
        let mut m = Glif::from_neuron_config(REFERENCE[1].config).unwrap();
        m.voltage_reset = VoltageReset::Scaled { a: 1.0, b: 0.05 };
        let stim = vec![1e-9; 3000];
        let run = m.run(&stim).unwrap();
        assert_eq!(run.spikes.len(), 1);
        let t = run.spikes[0].step;
        assert_eq!(run.bad_reset, Some(t));
        let resumed = t + m.spike_cut;
        let reset = run.voltage[resumed];
        assert!(reset > run.threshold[resumed]);
        assert!(run.voltage[resumed + 1..resumed + 6].iter().all(|&v| v == reset), "five copies of the reset");
        assert!(run.voltage[resumed + 6..].iter().all(|v| v.is_nan()), "and then nothing");
        assert!(run.voltage[..t].iter().all(|v| v.is_finite()));
    }
}
