//! Synapse models: the four kernels, the two current conventions, and short-term plasticity.
//!
//! A spike is an event with no width. A synapse is what gives it a width. Everything a spiking
//! network computes between one neuron's threshold crossing and the next neuron's happens inside
//! the shape this module defines, and picking the shape is not a cosmetic choice: it sets how long
//! two inputs can be apart and still sum, which is the only coincidence detector a spiking network
//! has.
//!
//! # What a kernel is, and what each one buys
//!
//! A presynaptic spike at `t = 0` opens channels. The postsynaptic effect is `w * k(t)`, where `w`
//! is the synaptic weight and `k` is the kernel — a fixed shape, normalised here so its **peak is
//! exactly 1** for a unit weight. Four shapes cover essentially the whole literature:
//!
//! | kernel | `k(t)` | peak at | integral | what it costs |
//! |---|---|---|---|---|
//! | [`Delta`] | `δ(t)` | — | — | 0 state words |
//! | [`Exponential`] | `exp(-t/τ)` | `t = 0` | `τ` | 1 state word |
//! | [`Alpha`] | `(t/τ)·exp(1 - t/τ)` | `t = τ` | `e·τ` | 2 state words |
//! | [`BiExponential`] | normalised `exp(-t/τ_d) - exp(-t/τ_r)` | closed form | closed form | 2 state words |
//!
//! [`Delta`] is what the rest of this crate has had until now: an instantaneous voltage
//! displacement, [`crate::neuron::Neuron::bump`]. It is free, it is what most digital neuromorphic
//! cores implement, and it **cannot detect coincidence** — two spikes one tick apart sum exactly as
//! two spikes a thousand ticks apart do, because the membrane's own leak is the only thing left
//! doing the temporal integration. Every other kernel here buys a coincidence window and pays for
//! it in per-synapse state: on hardware, that is the difference between a synapse that is a weight
//! and a synapse that is a weight plus a decaying variable, and the second one has to be fetched
//! and written back every tick rather than only when a spike arrives.
//!
//! [`Alpha`] is the classical compromise — a finite rise, one time constant, one parameter.
//! [`BiExponential`] separates rise from decay, which is what you need if you want `AMPA`'s
//! half-millisecond rise and `NMDA`'s hundred-millisecond tail in the same network.
//!
//! # The distinction that is usually glossed: `CUBA` versus `COBA`
//!
//! A **current-based** synapse ([`CurrentBased`], "`CUBA`") injects `I(t) = gain · w · k(t)`
//! amperes. That current does not depend on the membrane potential, so a synapse that depolarises a
//! resting cell by 5 mV also depolarises an already-depolarised cell by 5 mV, and an excitatory
//! input can in principle drive the membrane past the sodium reversal potential and keep going.
//! `CUBA` is linear, which is why the analytically tractable network literature uses it.
//!
//! A **conductance-based** synapse ([`ConductanceBased`], "`COBA`") opens a conductance and lets
//! the driving force do the rest: `I(t) = g(t) · (E_rev - V)`. Three consequences follow, and they
//! are the reason this module exists:
//!
//! 1. **The effect shrinks as `V` approaches `E_rev`.** At `V = E_rev` the current is exactly zero,
//!    not nearly zero — [`Drive::current`] returns `0.0` bit-for-bit, and there is a test for it.
//!    A `COBA` network self-limits; a `CUBA` network does not.
//! 2. **Inhibition need not hyperpolarise.** If `E_rev` sits at the resting potential, an
//!    inhibitory synapse moves the resting membrane not at all and still reduces every excitatory
//!    response passing through it, because it has raised the total conductance and therefore
//!    lowered the input resistance. This is **shunting** (or divisive) inhibition, it is what
//!    `GABA_A` does when the chloride reversal is near rest, and it is invisible in a `CUBA` model:
//!    there, inhibition is just a negative current and always moves the potential.
//! 3. **The membrane time constant is not a constant.** Total conductance goes up while synapses
//!    are open, so the effective `τ_m = C/g_total` goes down and the cell integrates faster during
//!    activity than at rest. Reported `τ_m` values from quiet slices are upper bounds on the
//!    in-vivo value for this reason.
//!
//! # Receptors
//!
//! [`AMPA`], [`NMDA`], [`GABA_A`] and [`GABA_B`] are [`Receptor`] records: a reversal potential, a
//! rise and a decay, and a provenance string. `NMDA` additionally carries [`MgBlock`], the
//! magnesium block, which makes its conductance **voltage-dependent** — near rest the pore is
//! plugged by `Mg²⁺` and the synapse does almost nothing, and depolarisation expels the block. That
//! makes `NMDA` a coincidence detector between presynaptic transmitter and postsynaptic
//! depolarisation, which is the biophysical substrate usually named when Hebbian learning is
//! justified.
//!
//! The kinetic scheme behind the shapes is [`KineticTwoState`], from Destexhe, Mainen &
//! Sejnowski, *Neural Computation* 6:14-18, 1994, with the rate constants as tabulated in
//! Destexhe, Mainen & Sejnowski, "Kinetic models of synaptic transmission", in *Methods in
//! Neuronal Modeling* (2nd ed.), 1998. [`GabaBCascade`] is the four-variable G-protein model from
//! Destexhe & Sejnowski, *PNAS* 92:9515-9519, 1995 — `GABA_B` is genuinely a second-messenger
//! cascade with a fourth-power cooperativity, and a bi-exponential conductance is a fit to it
//! rather than the mechanism.
//!
//! # Short-term plasticity
//!
//! [`TsodyksMarkram`] is the phenomenological model of Tsodyks & Markram, *PNAS* 94:719-723, 1997
//! and Markram, Wang & Tsodyks, *PNAS* 95:5323-5328, 1998. Two state variables: `x`, the fraction
//! of vesicles available, which depletes on release and recovers with `τ_d`; and `u`, the fraction
//! released per spike, which steps up on every spike and decays with `τ_f`. Depression and
//! facilitation are the same equations with different constants, and the balance flips with
//! **rate** as well as with parameters — [`TsodyksMarkram::facilitating`] facilitates at 20 Hz and
//! depresses at 100 Hz, and there is a test that pins both.
//!
//! The result the model is famous for is in [`TsodyksMarkram::limiting_transmission_rate`]: as the
//! presynaptic rate goes to infinity the transmitted signal converges to `1/τ_d` **releases per
//! second regardless of `U` and `τ_f`**, so a depressing synapse transmits rate changes and not
//! rate. That is a closed form, and it is checked.
//!
//! # Units
//!
//! SI at every interface: seconds, volts, amperes, siemens. Two documented exceptions, both inside
//! the models where a reader compares them against the source. The magnesium block's `0.062` and
//! `3.57` are **per millivolt** and **millimolar** as Jahr & Stevens, *J. Neurosci.* 10:3178-3182,
//! 1990 print them, and [`MgBlock::open_fraction`] converts volts to millivolts at its boundary.
//! The kinetic rate constants are per second and per millimolar, because transmitter concentration
//! is millimolar everywhere in that literature and rewriting `1.1e6 M⁻¹s⁻¹` into molar-SI makes it
//! unrecognisable against the table it came from.
//!
//! # What is verified, and what is not
//!
//! Verified against closed forms: the alpha kernel's peak (exactly `1` at exactly `t = τ`), the
//! exponential's integral (exactly `τ`), the alpha's integral (exactly `e·τ`), the
//! bi-exponential's peak time and its alpha limit, the incremental forms against their own
//! analytic responses to floating-point noise, the `Tsodyks`-`Markram` steady state at six rates
//! for two parameter sets, the limiting transmission rate, the two-state kinetic scheme's
//! analytic solution, the `GABA_B` cascade's steady state, and the magnesium block's half-block
//! potential.
//!
//! **Not verified: the constants themselves.** A steady-state test checks the integrator against
//! the model, never the model against a cell. Every numeric receptor parameter here is transcribed
//! from the literature, the doc on each one says which paper and how confident the transcription
//! is, and where this implementation could not confirm a figure against the primary source it says
//! so instead of rounding it off confidently.
//!
//! ```
//! use ferromorphic::synapse::{AMPA, Drive};
//!
//! // An AMPA-like conductance-based synapse: 1 nS peak conductance, reversal at 0 mV.
//! let mut syn = AMPA.conductance_based(1e-9)?;
//! syn.inject(1.0); // one spike, unit weight
//! syn.advance(1e-3); // look 1 ms later
//!
//! // Depolarising at rest, and exactly nothing at the reversal potential.
//! assert!(syn.current(-65e-3) > 0.0);
//! assert_eq!(syn.current(0.0), 0.0);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use core::fmt;
use std::f64::consts::E;

/// Why a synapse could not be built, or why a question about one has no answer.
///
/// Every variant carries the offending value so the message can print it. A synapse parameter that
/// is silently clamped produces a network that runs, reports spikes, and is wrong in a way no
/// downstream test can see — the same failure mode [`crate::net::NetError::NonFiniteWeight`] exists
/// to prevent one layer down.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SynapseError {
    /// A parameter was `NaN` or infinite.
    NonFinite {
        /// The parameter's name, spelled as this crate spells it.
        name: &'static str,
        /// The rejected value, carried so the message can name it.
        value: f64,
    },
    /// A time constant was zero or negative. Seconds; the model needs a strictly positive one,
    /// because every kernel here divides by it.
    NonPositiveTimeConstant {
        /// The parameter's name.
        name: &'static str,
        /// The rejected value, seconds.
        value: f64,
    },
    /// A bi-exponential was given a rise slower than its decay.
    ///
    /// Not a style complaint: the peak normalisation is derived for `τ_rise <= τ_decay`, and the
    /// difference `exp(-t/τ_d) - exp(-t/τ_r)` is negative for every `t > 0` when the order is
    /// reversed, so the kernel would silently flip the sign of every synapse using it.
    RiseSlowerThanDecay {
        /// Rise time constant as supplied, seconds.
        tau_rise: f64,
        /// Decay time constant as supplied, seconds.
        tau_decay: f64,
    },
    /// A quantity that is a fraction left the closed unit interval `[0, 1]`.
    FractionOutOfRange {
        /// The parameter's name.
        name: &'static str,
        /// The rejected value, dimensionless.
        value: f64,
    },
    /// A quantity that cannot be negative was negative — a conductance, a concentration, a duration.
    Negative {
        /// The parameter's name.
        name: &'static str,
        /// The rejected value, in that parameter's own unit.
        value: f64,
    },
    /// A rate was zero or negative where the model needs an inter-spike interval `1/r`.
    ///
    /// Zero hertz is not a slow steady state. A synapse that is never used has no steady-state
    /// release, and returning the rest state instead would answer a question that was not asked.
    NonPositiveRate {
        /// The rejected rate, hertz.
        value: f64,
    },
}

impl fmt::Display for SynapseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFinite { name, value } => {
                write!(f, "{name} is {value}, which is not a finite number")
            }
            Self::NonPositiveTimeConstant { name, value } => {
                write!(f, "{name} is {value} s; a time constant must be strictly positive")
            }
            Self::RiseSlowerThanDecay { tau_rise, tau_decay } => write!(
                f,
                "rise {tau_rise} s is slower than decay {tau_decay} s; \
                 the bi-exponential's normalisation needs tau_rise <= tau_decay"
            ),
            Self::FractionOutOfRange { name, value } => {
                write!(f, "{name} is {value}, outside the unit interval [0, 1]")
            }
            Self::Negative { name, value } => {
                write!(f, "{name} is {value}, which is negative")
            }
            Self::NonPositiveRate { value } => write!(
                f,
                "{value} Hz has no inter-spike interval; a synapse that is never used \
                 has no steady state"
            ),
        }
    }
}

/// So that `?` works in a caller whose error type is `Box<dyn Error>`, which is what every example
/// and doctest in this crate uses.
impl std::error::Error for SynapseError {}

fn finite(name: &'static str, value: f64) -> Result<f64, SynapseError> {
    if value.is_finite() { Ok(value) } else { Err(SynapseError::NonFinite { name, value }) }
}

fn positive_tau(name: &'static str, value: f64) -> Result<f64, SynapseError> {
    let v = finite(name, value)?;
    if v > 0.0 { Ok(v) } else { Err(SynapseError::NonPositiveTimeConstant { name, value: v }) }
}

fn non_negative(name: &'static str, value: f64) -> Result<f64, SynapseError> {
    let v = finite(name, value)?;
    if v >= 0.0 { Ok(v) } else { Err(SynapseError::Negative { name, value: v }) }
}

fn unit_fraction(name: &'static str, value: f64) -> Result<f64, SynapseError> {
    let v = finite(name, value)?;
    if (0.0..=1.0).contains(&v) {
        Ok(v)
    } else {
        Err(SynapseError::FractionOutOfRange { name, value: v })
    }
}

/// Whether a time step is usable by an incremental update.
///
/// Zero is usable-as-a-no-op rather than an error, so that a simulator may call `advance(0.0)`
/// while landing exactly on an event boundary.
fn steppable(dt: f64) -> bool {
    dt.is_finite() && dt > 0.0
}

/// Check a time step once, outside the loop, and get an error that names what was wrong.
///
/// [`Kernel::advance`] and the plasticity models cannot return an error per call without putting a
/// `Result` in the innermost loop of every simulation, so they **ignore** a `dt` that is negative
/// or non-finite and leave their state untouched. That keeps `NaN` out of every downstream membrane
/// potential, at the cost of saying nothing. This is the call that says it — make it once, where
/// the time step is chosen.
///
/// # Errors
///
/// [`SynapseError::NonFinite`] for `NaN` or an infinity, [`SynapseError::NonPositiveTimeConstant`]
/// for zero or a negative step.
pub fn check_dt(dt: f64) -> Result<(), SynapseError> {
    let v = positive_tau("dt", dt)?;
    let _ = v;
    Ok(())
}

/// A postsynaptic response shape, in both its analytic and its incremental form.
///
/// # The contract between the two forms
///
/// The incremental form is driven as `inject` → read [`Kernel::value`] → [`Kernel::advance`]. After
/// `inject(w)` on a cleared kernel followed by `k` calls to `advance(dt)`, [`Kernel::value`] equals
/// `w * response(k * dt)`. Every kernel here satisfies that **exactly**, not approximately, because
/// every incremental update is the analytic solution of the kernel's own linear system over the
/// step rather than a discretisation of it. The one exception is [`Delta`], whose analytic form is
/// a distribution and has no pointwise value to agree with; its doc says so.
///
/// # Peak normalisation, and the one kernel it cannot apply to
///
/// `response` is normalised so its **maximum is exactly 1** for a unit weight. A weight is then
/// "the peak of the postsynaptic response", which is the quantity experiments report. [`Delta`] is
/// area-normalised instead — `δ(t)` has units of inverse seconds and no finite peak — so its
/// [`Kernel::peak_value`] and [`Kernel::integral_seconds`] both refuse rather than returning a
/// number that cannot be compared against the others'.
pub trait Kernel: Clone {
    /// Whether one `advance` of `k * dt` gives exactly the state of `k` advances of `dt`.
    ///
    /// The same property [`crate::neuron::Neuron::EXACT_OVER_GAPS`] declares for a membrane, and it
    /// is true for every kernel in this module: each incremental update is an exact exponential
    /// solution, and exponentials compose across concatenated intervals. It is a constant rather
    /// than a comment so that a future kernel integrated by forward Euler has to declare `false`
    /// and be refused by anything that jumps over quiet ticks.
    const EXACT_OVER_STEPS: bool;

    /// The kernel's value `t` seconds after a unit-weight spike, peak-normalised to 1.
    ///
    /// `Some(0.0)` for `t < 0`, which is causality and not a convention. `None` where the kernel
    /// has no pointwise value — [`Delta`] at `t = 0` — and `None` for a non-finite `t`, so that a
    /// `NaN` argument cannot become a `NaN` postsynaptic current.
    fn response(&self, t: f64) -> Option<f64>;

    /// `∫₀^∞ response(t) dt`, seconds, in closed form.
    ///
    /// The total charge a unit-weight spike delivers, divided by the peak amplitude — so it is the
    /// kernel's *effective width*, and it is the quantity that decides how much a synapse
    /// contributes to a rate code. `None` for [`Delta`], whose integral is the dimensionless 1 and
    /// is not comparable with the others' seconds.
    fn integral_seconds(&self) -> Option<f64>;

    /// Time of the maximum, seconds after the spike, in closed form. `None` when there is no
    /// interior maximum to report.
    fn peak_time(&self) -> Option<f64>;

    /// Value at the maximum for a unit weight — exactly `1.0` for every peak-normalised kernel, and
    /// `None` for [`Delta`], which is unbounded there.
    fn peak_value(&self) -> Option<f64>;

    /// Deliver a spike of weight `w`, adding its response to whatever is already decaying.
    ///
    /// Superposition: the kernels are linear, so two spikes give the sum of two responses. A
    /// non-finite `w` is ignored, for the reason given on [`check_dt`].
    fn inject(&mut self, w: f64);

    /// Advance the internal state by `dt` seconds, exactly.
    ///
    /// A `dt` that is zero, negative or non-finite leaves the state unchanged. See [`check_dt`].
    fn advance(&mut self, dt: f64);

    /// The kernel's current value: the sum of every injected weight times its own elapsed response.
    ///
    /// Dimensionless. What it multiplies — amperes for [`CurrentBased`], siemens for
    /// [`ConductanceBased`] — is the caller's choice and is where the unit enters.
    #[must_use]
    fn value(&self) -> f64;

    /// Forget every spike delivered so far and return to zero.
    fn clear(&mut self);
}

/// The instantaneous synapse: `w · δ(t)`.
///
/// What [`crate::neuron::Neuron::bump`] already does, given a name and an analytic identity so it
/// can sit in the same table as the others. The postsynaptic effect is a single displacement at the
/// moment of arrival and nothing afterwards.
///
/// **It has no coincidence window.** Two spikes one tick apart and two spikes a second apart sum
/// identically; whatever temporal structure the network computes with has to come from the
/// membrane's own leak. That is the trade this kernel makes, and it is why digital neuromorphic
/// cores overwhelmingly implement it: zero per-synapse state, one add per event.
///
/// **The trap it carries**: the stateful form's [`Kernel::value`] is the weight delivered *during
/// the current step*, so a [`CurrentBased`] synapse built on `Delta` delivers `gain · w` amperes for
/// exactly one step and therefore a charge proportional to `dt`. Halve the time step and every
/// synaptic influence in the network halves while every parameter stays the same. Deliver a delta
/// synapse as a **voltage** through [`crate::neuron::Neuron::bump`], or accept that the weight is
/// in coulombs and divide by `dt` yourself.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Delta {
    /// Weight delivered during the current step, cleared by the next [`Kernel::advance`].
    /// Dimensionless; the caller's `gain` supplies the unit.
    pub pending: f64,
}

impl Delta {
    /// A delta kernel with nothing pending.
    #[must_use]
    pub fn new() -> Self {
        Self { pending: 0.0 }
    }
}

impl Kernel for Delta {
    // Clearing is idempotent: one long step and several short ones both end with nothing pending.
    const EXACT_OVER_STEPS: bool = true;

    fn response(&self, t: f64) -> Option<f64> {
        if !t.is_finite() {
            return None;
        }
        // The Dirac delta is zero everywhere except the origin, where it is not a function at all.
        // Returning a large number there would be a lie with a plausible magnitude; returning zero
        // would lose the entire kernel. `None` is the only honest value, and it is also the one
        // that forces a caller to notice that this kernel is not like the others.
        if t == 0.0 { None } else { Some(0.0) }
    }

    fn integral_seconds(&self) -> Option<f64> {
        // `δ(t)` has units of s⁻¹, so its time integral is the dimensionless 1 and cannot be
        // compared against `Exponential`'s τ seconds. Refusing keeps the two out of one table.
        None
    }

    fn peak_time(&self) -> Option<f64> {
        None
    }

    fn peak_value(&self) -> Option<f64> {
        None
    }

    fn inject(&mut self, w: f64) {
        if w.is_finite() {
            self.pending += w;
        }
    }

    fn advance(&mut self, dt: f64) {
        if steppable(dt) {
            self.pending = 0.0;
        }
    }

    fn value(&self) -> f64 {
        self.pending
    }

    fn clear(&mut self) {
        self.pending = 0.0;
    }
}

/// Single-exponential decay: `k(t) = exp(-t/τ)`, jumping to its peak instantly.
///
/// One state word and one multiply per step, which is why it is the kernel almost every large
/// simulation actually runs. Its integral is **exactly `τ`** — the cleanest closed form in this
/// module, and the one the tests check first.
///
/// What it gives up is the rise. The response is discontinuous at `t = 0`, so a postsynaptic cell
/// sees the full amplitude of an input in the same instant the spike arrives, and no synapse
/// behaves that way. Where the rise time matters — feed-forward inhibition arriving a fraction of a
/// millisecond after the excitation it gates — use [`Alpha`] or [`BiExponential`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Exponential {
    tau: f64,
    g: f64,
}

impl Exponential {
    /// A single-exponential kernel with decay constant `tau` seconds.
    ///
    /// # Errors
    ///
    /// [`SynapseError::NonPositiveTimeConstant`] for a `tau` that is not strictly positive, and
    /// [`SynapseError::NonFinite`] for `NaN` or an infinity.
    pub fn new(tau: f64) -> Result<Self, SynapseError> {
        Ok(Self { tau: positive_tau("tau", tau)?, g: 0.0 })
    }

    /// Decay time constant, seconds. Also the kernel's integral, exactly.
    #[must_use]
    pub fn tau(&self) -> f64 {
        self.tau
    }
}

impl Kernel for Exponential {
    const EXACT_OVER_STEPS: bool = true;

    fn response(&self, t: f64) -> Option<f64> {
        if !t.is_finite() {
            return None;
        }
        if t < 0.0 {
            return Some(0.0);
        }
        Some((-t / self.tau).exp())
    }

    fn integral_seconds(&self) -> Option<f64> {
        // ∫₀^∞ exp(-t/τ) dt = τ. Exact, and returned as the stored τ rather than recomputed so the
        // equality in the test is bit-for-bit rather than to a tolerance.
        Some(self.tau)
    }

    fn peak_time(&self) -> Option<f64> {
        Some(0.0)
    }

    fn peak_value(&self) -> Option<f64> {
        Some(1.0)
    }

    fn inject(&mut self, w: f64) {
        if w.is_finite() {
            self.g += w;
        }
    }

    fn advance(&mut self, dt: f64) {
        if steppable(dt) {
            self.g *= (-dt / self.tau).exp();
        }
    }

    fn value(&self) -> f64 {
        self.g
    }

    fn clear(&mut self) {
        self.g = 0.0;
    }
}

/// The alpha function: `k(t) = (t/τ)·exp(1 - t/τ)`, peak exactly `1` at exactly `t = τ`.
///
/// # The peak, derived
///
/// `dk/dt = (1/τ)·exp(1 - t/τ)·(1 - t/τ)`, which vanishes only at `t = τ`, and
/// `k(τ) = 1·exp(0) = 1`. The `exp(1)` in the definition is exactly the factor that turns the raw
/// `(t/τ)·exp(-t/τ)`, whose peak is `1/e`, into a peak of one — so a weight is the peak response
/// with no scale factor to remember. The test does not approximate this: `t/τ` with `t = τ` is
/// `1.0` bit-for-bit, `exp(0.0)` is `1.0` bit-for-bit, so `response(tau)` returns `1.0` exactly.
///
/// Its integral is **`e·τ`**, about 2.72 times the width of an exponential with the same `τ` —
/// worth knowing before swapping one kernel for the other at fixed `τ` and wondering why the
/// network's total drive nearly tripled.
///
/// # The incremental form
///
/// Two state words, `x` and `g`, with `x' = -x/τ` and `g' = -g/τ + x/τ`. Over a step of `dt`, with
/// `d = exp(-dt/τ)`:
///
/// ```text
/// g ← (g + x·dt/τ)·d
/// x ← x·d
/// ```
///
/// That is the exact solution of the pair, not an approximation, so a coarse step and a fine step
/// give the same answer to floating-point noise. A spike of weight `w` enters as `x += w·e`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Alpha {
    tau: f64,
    x: f64,
    g: f64,
}

impl Alpha {
    /// An alpha kernel with time-to-peak `tau` seconds.
    ///
    /// # Errors
    ///
    /// [`SynapseError::NonPositiveTimeConstant`] for a `tau` that is not strictly positive, and
    /// [`SynapseError::NonFinite`] for `NaN` or an infinity.
    pub fn new(tau: f64) -> Result<Self, SynapseError> {
        Ok(Self { tau: positive_tau("tau", tau)?, x: 0.0, g: 0.0 })
    }

    /// Time constant, seconds — which for this kernel is also the time to peak, exactly.
    #[must_use]
    pub fn tau(&self) -> f64 {
        self.tau
    }
}

impl Kernel for Alpha {
    const EXACT_OVER_STEPS: bool = true;

    fn response(&self, t: f64) -> Option<f64> {
        if !t.is_finite() {
            return None;
        }
        if t < 0.0 {
            return Some(0.0);
        }
        let s = t / self.tau;
        Some(s * (1.0 - s).exp())
    }

    fn integral_seconds(&self) -> Option<f64> {
        // ∫₀^∞ (t/τ)e^{1-t/τ} dt = e·(1/τ)·τ² = e·τ.
        Some(E * self.tau)
    }

    fn peak_time(&self) -> Option<f64> {
        Some(self.tau)
    }

    fn peak_value(&self) -> Option<f64> {
        Some(1.0)
    }

    fn inject(&mut self, w: f64) {
        if w.is_finite() {
            // e·w, so that the resulting g(t) is w·(t/τ)·e^{1-t/τ} and the peak is w.
            self.x += w * E;
        }
    }

    fn advance(&mut self, dt: f64) {
        if steppable(dt) {
            let d = (-dt / self.tau).exp();
            // g first: the update for g uses x at the START of the step.
            self.g = (self.g + self.x * dt / self.tau) * d;
            self.x *= d;
        }
    }

    fn value(&self) -> f64 {
        self.g
    }

    fn clear(&mut self) {
        self.x = 0.0;
        self.g = 0.0;
    }
}

/// Separate rise and decay: the normalised difference of two exponentials.
///
/// `k(t) = N·(exp(-t/τ_d) - exp(-t/τ_r))` with `N` chosen so the peak is exactly 1. This is the
/// shape fitted to real postsynaptic conductances, because a real synapse's rise and decay are set
/// by different physical processes — transmitter binding and channel closing — and are not one
/// parameter.
///
/// # The closed forms
///
/// The peak is where `d/dt` vanishes, which gives
///
/// ```text
/// t_peak = ln(τ_d/τ_r) · τ_r·τ_d / (τ_d - τ_r)
/// N      = 1 / (exp(-t_peak/τ_d) - exp(-t_peak/τ_r))
/// ∫₀^∞   = N · (τ_d - τ_r)
/// ```
///
/// # The removable singularity at `τ_r = τ_d`, and why it is not a special case
///
/// Every expression above divides by `τ_d - τ_r`, and every one of them has a finite limit as the
/// two meet: the kernel becomes the **alpha function** exactly. Writing `1/τ_d = m - h` and
/// `1/τ_r = m + h`, the difference is `2·e^{-mt}·sinh(ht)`, whose normalised shape is
/// `(t/τ)·e^{1-t/τ}` with `τ = 1/m` — the **harmonic mean** `2·τ_r·τ_d/(τ_r + τ_d)`, not the
/// arithmetic one. This implementation switches to the alpha form when
/// `|τ_d - τ_r| <= 1e-6·(τ_d + τ_r)/2`, and uses that harmonic mean.
///
/// Both halves of that choice matter. Taking the harmonic mean makes the switch **second order**
/// in the separation — the two branches differ by `O(δ²)`, about `1e-12` at the threshold — where
/// the arithmetic mean would leave a first-order step of `1e-6`. And switching at all is not
/// defensive coding: the direct formula loses precision like `ε/δ` through the cancellation in
/// `N`, so below a separation of roughly `1e-5` the alpha limit is **more** accurate than the exact
/// expression, not less.
///
/// # The incremental form
///
/// Two state words, `x' = -x/τ_r` and `g' = -g/τ_d + x`, updated exactly with
/// `K = τ_r·τ_d/(τ_d - τ_r)`, `d_r = exp(-dt/τ_r)`, `d_d = exp(-dt/τ_d)`:
///
/// ```text
/// g ← g·d_d + x·K·(d_d - d_r)
/// x ← x·d_r
/// ```
///
/// In the degenerate branch it runs the alpha update instead, so the incremental form is
/// continuous across the switch too.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BiExponential {
    tau_rise: f64,
    tau_decay: f64,
    /// True when the two constants are close enough that the alpha limit is taken.
    degenerate: bool,
    /// Harmonic mean, used only in the degenerate branch.
    tau_eff: f64,
    /// Peak normalisation `N`; `1.0` in the degenerate branch, where the alpha form is already
    /// peak-normalised.
    norm: f64,
    /// `τ_r·τ_d/(τ_d - τ_r)`; unused in the degenerate branch.
    k: f64,
    t_peak: f64,
    x: f64,
    g: f64,
}

impl BiExponential {
    /// Relative separation below which the alpha limit is taken. See the type's doc for why this
    /// value, and why the limit is the more accurate branch below it rather than the safer one.
    const DEGENERATE_BELOW: f64 = 1e-6;

    /// A bi-exponential kernel with rise `tau_rise` and decay `tau_decay`, both seconds.
    ///
    /// Equal constants are legal and give the alpha function exactly.
    ///
    /// # Errors
    ///
    /// [`SynapseError::NonPositiveTimeConstant`] or [`SynapseError::NonFinite`] for either
    /// constant, and [`SynapseError::RiseSlowerThanDecay`] when `tau_rise > tau_decay`, which would
    /// invert the sign of the kernel rather than merely reorder it.
    pub fn new(tau_rise: f64, tau_decay: f64) -> Result<Self, SynapseError> {
        let tau_rise = positive_tau("tau_rise", tau_rise)?;
        let tau_decay = positive_tau("tau_decay", tau_decay)?;
        if tau_rise > tau_decay {
            return Err(SynapseError::RiseSlowerThanDecay { tau_rise, tau_decay });
        }
        let mean = 0.5 * (tau_rise + tau_decay);
        let tau_eff = 2.0 * tau_rise * tau_decay / (tau_rise + tau_decay);
        let degenerate = (tau_decay - tau_rise) <= Self::DEGENERATE_BELOW * mean;
        let (norm, k, t_peak) = if degenerate {
            (1.0, 0.0, tau_eff)
        } else {
            let t_peak =
                (tau_decay / tau_rise).ln() * tau_rise * tau_decay / (tau_decay - tau_rise);
            let raw = (-t_peak / tau_decay).exp() - (-t_peak / tau_rise).exp();
            let k = tau_rise * tau_decay / (tau_decay - tau_rise);
            (1.0 / raw, k, t_peak)
        };
        Ok(Self { tau_rise, tau_decay, degenerate, tau_eff, norm, k, t_peak, x: 0.0, g: 0.0 })
    }

    /// Rise time constant, seconds.
    #[must_use]
    pub fn tau_rise(&self) -> f64 {
        self.tau_rise
    }

    /// Decay time constant, seconds.
    #[must_use]
    pub fn tau_decay(&self) -> f64 {
        self.tau_decay
    }

    /// Whether this kernel is running the alpha limit rather than the two-exponential formula.
    ///
    /// Exposed because a caller comparing two kernels across the switch deserves to know which
    /// branch produced each number, rather than discovering the discontinuity as a mystery at the
    /// twelfth decimal place.
    #[must_use]
    pub fn is_degenerate(&self) -> bool {
        self.degenerate
    }

    /// The single time constant the degenerate branch uses: the harmonic mean
    /// `2·τ_r·τ_d/(τ_r + τ_d)`, seconds. Equal to either constant when they are equal.
    #[must_use]
    pub fn tau_harmonic(&self) -> f64 {
        self.tau_eff
    }

    /// Peak normalisation `N`, dimensionless — `1.0` in the degenerate branch.
    #[must_use]
    pub fn normalisation(&self) -> f64 {
        self.norm
    }
}

impl Kernel for BiExponential {
    const EXACT_OVER_STEPS: bool = true;

    fn response(&self, t: f64) -> Option<f64> {
        if !t.is_finite() {
            return None;
        }
        if t < 0.0 {
            return Some(0.0);
        }
        if self.degenerate {
            let s = t / self.tau_eff;
            return Some(s * (1.0 - s).exp());
        }
        Some(self.norm * ((-t / self.tau_decay).exp() - (-t / self.tau_rise).exp()))
    }

    fn integral_seconds(&self) -> Option<f64> {
        if self.degenerate {
            // Continuous with the general branch: N·(τ_d - τ_r) → e·τ as the constants meet.
            return Some(E * self.tau_eff);
        }
        Some(self.norm * (self.tau_decay - self.tau_rise))
    }

    fn peak_time(&self) -> Option<f64> {
        Some(self.t_peak)
    }

    fn peak_value(&self) -> Option<f64> {
        Some(1.0)
    }

    fn inject(&mut self, w: f64) {
        if w.is_finite() {
            if self.degenerate {
                self.x += w * E;
            } else {
                self.x += w * self.norm / self.k;
            }
        }
    }

    fn advance(&mut self, dt: f64) {
        if !steppable(dt) {
            return;
        }
        if self.degenerate {
            let d = (-dt / self.tau_eff).exp();
            self.g = (self.g + self.x * dt / self.tau_eff) * d;
            self.x *= d;
            return;
        }
        let dr = (-dt / self.tau_rise).exp();
        let dd = (-dt / self.tau_decay).exp();
        self.g = self.g * dd + self.x * self.k * (dd - dr);
        self.x *= dr;
    }

    fn value(&self) -> f64 {
        self.g
    }

    fn clear(&mut self) {
        self.x = 0.0;
        self.g = 0.0;
    }
}

/// A synapse that can be asked what current it is delivering into a membrane at potential `V`.
///
/// The trait exists so that the `CUBA`/`COBA` choice is a **type** rather than a flag, and so that
/// the three questions a conductance-based synapse can answer and a current-based one cannot —
/// conductance, reversal potential, driving force — are `Option` rather than a silent zero. A
/// current-based synapse does not have a small conductance. It does not have one.
pub trait Drive {
    /// Deliver a presynaptic spike of weight `w`.
    fn inject(&mut self, w: f64);

    /// Advance the underlying kernel by `dt` seconds. See [`check_dt`] for the error path.
    fn advance(&mut self, dt: f64);

    /// Current delivered **into** the cell at membrane potential `v` volts, in amperes.
    ///
    /// Sign convention: positive depolarises. A `COBA` synapse returns exactly `0.0` at its
    /// reversal potential — bit-for-bit, because the driving force `E_rev - v` is exactly zero
    /// there and zero times any finite conductance is zero.
    #[must_use]
    fn current(&self, v: f64) -> f64;

    /// Open conductance in siemens, or `None` for a current-based synapse, which has no such thing.
    fn conductance(&self) -> Option<f64>;

    /// Reversal potential in volts, or `None` for a current-based synapse.
    ///
    /// The potential at which the synapse stops having an effect. Its absence in `CUBA` is exactly
    /// why a `CUBA` network can be driven arbitrarily far by arbitrarily many excitatory inputs.
    fn reversal_potential(&self) -> Option<f64>;

    /// `E_rev - v` volts, or `None` for a current-based synapse.
    fn driving_force(&self, v: f64) -> Option<f64>;

    /// Forget every spike delivered so far.
    fn clear(&mut self);
}

/// Current-based (`CUBA`): `I = gain · w · k(t)` amperes, independent of the membrane potential.
///
/// The convention the analytically tractable network literature runs on, because it makes the
/// membrane equation linear and lets the mean-field theory close. Two things it gets wrong, both
/// on purpose:
///
/// - **Excitation never saturates.** Real synapses stop working as the membrane approaches their
///   reversal potential; a `CUBA` synapse delivers its full current at any potential, so a strongly
///   driven `CUBA` cell can be pushed past every reversal potential in the model.
/// - **Inhibition is always hyperpolarising.** It is a negative current, so it always moves the
///   potential down. Shunting inhibition — a large conductance at the resting potential that
///   suppresses excitation without moving rest — cannot be represented at all. See
///   [`ConductanceBased`] and the test `shunting_inhibition_suppresses_without_moving_rest`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CurrentBased<K: Kernel> {
    kernel: K,
    gain: f64,
}

impl<K: Kernel> CurrentBased<K> {
    /// Wrap a kernel with a peak current of `gain` amperes per unit weight.
    ///
    /// Because every kernel here is peak-normalised, `gain · w` **is** the peak current of a
    /// weight-`w` spike, with no scale factor to remember. [`Delta`] is the exception named in its
    /// own doc: there `gain · w` is a current sustained for exactly one step, so the delivered
    /// charge depends on `dt`.
    ///
    /// # Errors
    ///
    /// [`SynapseError::NonFinite`] if `gain` is `NaN` or infinite. A negative gain is **allowed**
    /// and is how an inhibitory current-based synapse is built.
    pub fn new(kernel: K, gain: f64) -> Result<Self, SynapseError> {
        Ok(Self { kernel, gain: finite("gain", gain)? })
    }

    /// Peak current per unit weight, amperes. Negative for inhibition.
    #[must_use]
    pub fn gain(&self) -> f64 {
        self.gain
    }

    /// The underlying kernel, for its analytic forms.
    #[must_use]
    pub fn kernel(&self) -> &K {
        &self.kernel
    }
}

impl<K: Kernel> Drive for CurrentBased<K> {
    fn inject(&mut self, w: f64) {
        self.kernel.inject(w);
    }

    fn advance(&mut self, dt: f64) {
        self.kernel.advance(dt);
    }

    fn current(&self, _v: f64) -> f64 {
        // `_v` is ignored, and that IS the model. The parameter is kept in the signature rather
        // than split into a second trait so that swapping CUBA for COBA at a call site is a type
        // change and nothing else.
        self.gain * self.kernel.value()
    }

    fn conductance(&self) -> Option<f64> {
        None
    }

    fn reversal_potential(&self) -> Option<f64> {
        None
    }

    fn driving_force(&self, _v: f64) -> Option<f64> {
        None
    }

    fn clear(&mut self) {
        self.kernel.clear();
    }
}

/// Conductance-based (`COBA`): `I = g(t)·B(V)·(E_rev - V)` amperes.
///
/// `g(t) = g_peak · w · k(t)` siemens is the open conductance, `E_rev` is the reversal potential,
/// and `B(V)` is an optional voltage-dependent block — [`MgBlock`] for `NMDA`, and exactly `1` for
/// everything else.
///
/// # Why this is the load-bearing choice
///
/// The current a `COBA` synapse delivers **shrinks as the membrane approaches its reversal
/// potential**, reaching exactly zero there and reversing beyond it. Three consequences, each
/// absent from [`CurrentBased`]:
///
/// 1. Excitation self-limits. No amount of excitatory input drives a cell past `E_rev`.
/// 2. Inhibition need not hyperpolarise. With `E_rev` at the resting potential, an inhibitory
///    conductance leaves a resting cell exactly where it was and still divides down every
///    excitatory response, because it has lowered the input resistance. That is **shunting**
///    inhibition, and it is the reason the reversal potential of `GABA_A` — set by the chloride
///    gradient, and developmentally regulated — decides whether inhibition subtracts or divides.
/// 3. The membrane time constant falls during activity, because `τ = C/g_total` and `g_total`
///    includes every open synapse.
///
/// The price is nonlinearity: the membrane equation is no longer linear in the inputs, the
/// mean-field theory no longer closes, and the effect of a spike now depends on what else is open.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConductanceBased<K: Kernel> {
    kernel: K,
    g_peak: f64,
    e_rev: f64,
    mg: Option<MgBlock>,
}

impl<K: Kernel> ConductanceBased<K> {
    /// Wrap a kernel with a peak conductance of `g_peak` siemens per unit weight and a reversal
    /// potential of `e_rev` volts.
    ///
    /// Excitation and inhibition are **not** distinguished by the sign of `g_peak` here — they are
    /// distinguished by `e_rev` relative to the membrane potential. A negative conductance is not
    /// a physical object, and building inhibition out of one reproduces `CUBA` behaviour under a
    /// `COBA` name.
    ///
    /// # Errors
    ///
    /// [`SynapseError::Negative`] for a negative `g_peak`, and [`SynapseError::NonFinite`] for a
    /// non-finite `g_peak` or `e_rev`.
    pub fn new(kernel: K, g_peak: f64, e_rev: f64) -> Result<Self, SynapseError> {
        Ok(Self {
            kernel,
            g_peak: non_negative("g_peak", g_peak)?,
            e_rev: finite("e_rev", e_rev)?,
            mg: None,
        })
    }

    /// Attach a voltage-dependent block, which is what makes an `NMDA` synapse an `NMDA` synapse.
    #[must_use]
    pub fn with_block(mut self, mg: MgBlock) -> Self {
        self.mg = Some(mg);
        self
    }

    /// Peak conductance per unit weight, siemens.
    #[must_use]
    pub fn g_peak(&self) -> f64 {
        self.g_peak
    }

    /// Reversal potential, volts.
    #[must_use]
    pub fn e_rev(&self) -> f64 {
        self.e_rev
    }

    /// The voltage-dependent block, if this synapse has one.
    #[must_use]
    pub fn block(&self) -> Option<MgBlock> {
        self.mg
    }

    /// The underlying kernel, for its analytic forms.
    #[must_use]
    pub fn kernel(&self) -> &K {
        &self.kernel
    }

    /// Conductance actually available at potential `v`, siemens: the open conductance times the
    /// unblocked fraction.
    ///
    /// Equal to [`Drive::conductance`] for every receptor except `NMDA`, where the magnesium block
    /// makes the two differ by more than an order of magnitude at resting potential.
    #[must_use]
    pub fn open_conductance(&self, v: f64) -> f64 {
        let g = self.g_peak * self.kernel.value();
        match self.mg {
            Some(mg) => g * mg.open_fraction(v),
            None => g,
        }
    }
}

impl<K: Kernel> Drive for ConductanceBased<K> {
    fn inject(&mut self, w: f64) {
        self.kernel.inject(w);
    }

    fn advance(&mut self, dt: f64) {
        self.kernel.advance(dt);
    }

    fn current(&self, v: f64) -> f64 {
        // FACTORED, and it has to stay factored. `g·E_rev - g·V` is algebraically the same and
        // also gives exactly zero at `V == E_rev` — that much a mutation of this line proved, so
        // the exact-zero test alone does not defend it. What the expanded form loses is everything
        // NEAR the reversal potential: it subtracts two nearly equal products and keeps only the
        // digits they differ in, while `(E_rev - V)` is a subtraction of two nearby floats, which
        // IEEE arithmetic performs exactly, and the scaling afterwards costs one rounding. An
        // inhibitory synapse in a balanced network spends most of its life within a millivolt of
        // its reversal potential, which is precisely the region the expanded form gets wrong.
        self.open_conductance(v) * (self.e_rev - v)
    }

    fn conductance(&self) -> Option<f64> {
        Some(self.g_peak * self.kernel.value())
    }

    fn reversal_potential(&self) -> Option<f64> {
        Some(self.e_rev)
    }

    fn driving_force(&self, v: f64) -> Option<f64> {
        Some(self.e_rev - v)
    }

    fn clear(&mut self) {
        self.kernel.clear();
    }
}

/// The magnesium block of the `NMDA` receptor: a voltage-dependent plug in the channel pore.
///
/// `B(V) = 1 / (1 + exp(-α·V_mV)·[Mg]/K)` with `α = 0.062` per millivolt and `K = 3.57` millimolar,
/// the sigmoid of Jahr & Stevens, *J. Neurosci.* 10:3178-3182, 1990, as it is used in Wang,
/// *J. Neurosci.* 19:9587-9603, 1999 and Brunel & Wang, *J. Comput. Neurosci.* 11:63-85, 2001.
///
/// **The constants are kept in the paper's millivolt/millimolar frame**, for the same reason
/// [`crate::neuron::Izhikevich`]'s are: rewritten into volts and molar they stop being
/// recognisable against the source. [`MgBlock::open_fraction`] converts at its boundary and takes
/// volts like everything else in this crate.
///
/// # Why it matters more than a factor
///
/// At the resting potential the pore is mostly plugged — about 4% open at −70 mV with 1 mM
/// external magnesium — and depolarisation expels the block, reaching about 78% open at 0 mV. An
/// `NMDA` synapse therefore does almost nothing unless the postsynaptic cell is *already*
/// depolarised by something else, which makes it a coincidence detector between presynaptic
/// transmitter and postsynaptic state. That is the biophysics usually pointed at when Hebbian
/// learning is given a mechanism.
///
/// It also makes the `NMDA` current **non-monotonic in voltage**: near zero at very negative
/// potentials because the channel is blocked, exactly zero at `E_rev` because the driving force
/// vanishes, and maximal somewhere in between — a region of negative slope conductance, which is
/// what lets `NMDA` support bistable persistent activity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MgBlock {
    /// External magnesium concentration, **millimolar**. 1.0 is the standard physiological value;
    /// slice work often runs lower to relieve the block deliberately.
    pub mg_mm: f64,
    /// Voltage sensitivity, **per millivolt**. `0.062` in Jahr & Stevens, 1990. Larger makes the
    /// relief of the block steeper in voltage.
    pub slope_per_mv: f64,
    /// Concentration scale, **millimolar**. `3.57` in Jahr & Stevens, 1990. Together with `mg_mm`
    /// it sets where the sigmoid sits on the voltage axis.
    pub k_mm: f64,
}

impl Default for MgBlock {
    /// The Jahr & Stevens 1990 sigmoid at 1 mM external magnesium.
    fn default() -> Self {
        Self { mg_mm: 1.0, slope_per_mv: 0.062, k_mm: 3.57 }
    }
}

impl MgBlock {
    /// The published sigmoid at a chosen external magnesium concentration, millimolar.
    ///
    /// `0.0` is legal and removes the block entirely — which is the magnesium-free condition used
    /// experimentally to isolate the `NMDA` component, and is worth being able to simulate.
    ///
    /// # Errors
    ///
    /// [`SynapseError::Negative`] for a negative concentration, [`SynapseError::NonFinite`] for
    /// `NaN` or an infinity.
    pub fn with_magnesium(mg_mm: f64) -> Result<Self, SynapseError> {
        Ok(Self { mg_mm: non_negative("mg_mm", mg_mm)?, ..Self::default() })
    }

    /// Fraction of channels not blocked, at membrane potential `v` **volts**, in `[0, 1]`.
    ///
    /// Converts to millivolts at this boundary, because the `0.062` above is per millivolt. A
    /// non-finite `v` returns `0.0` — fully blocked — rather than propagating a `NaN` into a
    /// membrane potential; [`check_dt`] explains why the loop-level calls refuse silently.
    #[must_use]
    pub fn open_fraction(&self, v: f64) -> f64 {
        if !v.is_finite() {
            return 0.0;
        }
        let v_mv = v * 1e3;
        1.0 / (1.0 + (-self.slope_per_mv * v_mv).exp() * self.mg_mm / self.k_mm)
    }

    /// The potential at which exactly half the channels are unblocked, volts, in closed form.
    ///
    /// Setting `exp(-α·V_mV)·[Mg]/K = 1` gives `V_mV = ln([Mg]/K)/α`; at 1 mM and the published
    /// constants that is −20.5 mV, comfortably above rest, which is the quantitative statement of
    /// "`NMDA` needs the cell to be depolarised first".
    ///
    /// `None` at zero magnesium, where there is no block and therefore no half-block potential —
    /// the fraction is 1 at every voltage and no potential is special.
    #[must_use]
    pub fn half_block_potential(&self) -> Option<f64> {
        if self.mg_mm <= 0.0 || self.slope_per_mv == 0.0 {
            return None;
        }
        Some((self.mg_mm / self.k_mm).ln() / self.slope_per_mv * 1e-3)
    }
}

/// A named receptor: reversal potential, kinetics, and where the numbers came from.
///
/// A plain record with public fields, like [`crate::ledger::Prices`], so that a variant of a
/// published receptor is `Receptor { e_rev: -75e-3, ..GABA_A }` and the provenance string comes
/// along with it and can be corrected.
///
/// # Read `source` before `tau_decay`
///
/// Every number in the four tables below is **transcribed from the literature, not fitted here**,
/// and synaptic time constants vary by cell type, species, temperature and recording method by
/// more than a factor of two. The `source` field names the paper each set came from and the doc on
/// each constant says how firm the transcription is. Where this implementation could not confirm a
/// figure against the primary source, it says so there rather than presenting a round number
/// confidently.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Receptor {
    /// What the receptor is called in the literature, for a report that has to name it.
    pub name: &'static str,
    /// Reversal potential, **volts**. The potential at which this receptor's current is exactly
    /// zero, set by the ion gradients of whatever the channel conducts — near 0 mV for the
    /// non-selective cation channels `AMPA` and `NMDA`, near the chloride equilibrium for `GABA_A`,
    /// near the potassium equilibrium for `GABA_B`.
    pub e_rev: f64,
    /// Rise time constant, **seconds**, for a [`BiExponential`] fit to the conductance.
    pub tau_rise: f64,
    /// Decay time constant, **seconds**, for the same fit. The span between receptors here is
    /// three orders of magnitude, and that span is the point: it is what lets one network hold both
    /// a millisecond coincidence window and a several-hundred-millisecond memory.
    pub tau_decay: f64,
    /// The voltage-dependent block, present only for `NMDA`.
    pub mg_block: Option<MgBlock>,
    /// Author, venue and year for the constants above, so a disagreement has somewhere to go.
    pub source: &'static str,
}

impl Receptor {
    /// A [`BiExponential`] kernel with this receptor's rise and decay.
    ///
    /// # Errors
    ///
    /// As [`BiExponential::new`]. It cannot fail for the four tables in this module, and it is
    /// still a `Result` because `Receptor { tau_rise: .., ..AMPA }` is the supported way to vary
    /// one, and the variant has to be checked.
    pub fn kernel(&self) -> Result<BiExponential, SynapseError> {
        BiExponential::new(self.tau_rise, self.tau_decay)
    }

    /// A conductance-based synapse with this receptor's kinetics, reversal potential and block.
    ///
    /// `g_peak` is the peak conductance in **siemens** for a unit weight. Cortical unitary values
    /// are in the high picosiemens to low nanosiemens range; there is no default here because it is
    /// the parameter that actually varies between connections, and a default would be a number
    /// somebody cites.
    ///
    /// # Errors
    ///
    /// As [`BiExponential::new`] and [`ConductanceBased::new`].
    pub fn conductance_based(
        &self,
        g_peak: f64,
    ) -> Result<ConductanceBased<BiExponential>, SynapseError> {
        let mut d = ConductanceBased::new(self.kernel()?, g_peak, self.e_rev)?;
        d.mg = self.mg_block;
        Ok(d)
    }

    /// A current-based synapse with this receptor's kinetics and **none of its voltage dependence**.
    ///
    /// The reversal potential and the magnesium block are both dropped, because a current-based
    /// synapse has nowhere to put them. For `AMPA` and `GABA_A` that is the standard `CUBA`
    /// approximation and it is defensible. For `NMDA` it removes the entire mechanism — a
    /// current-based `NMDA` synapse is an `AMPA` synapse with a long tail, and calling it `NMDA`
    /// after that is the kind of shorthand that ends up in a methods section.
    ///
    /// # Errors
    ///
    /// As [`BiExponential::new`] and [`CurrentBased::new`].
    pub fn current_based(&self, gain: f64) -> Result<CurrentBased<BiExponential>, SynapseError> {
        CurrentBased::new(self.kernel()?, gain)
    }
}

/// Fast excitatory glutamate receptor. Reversal 0 mV, rise 0.5 ms, decay 2.0 ms.
///
/// The workhorse of cortical excitation and the fastest synapse in this table. The 2 ms decay is
/// the cortical value used by Brunel & Wang, *J. Comput. Neurosci.* 11:63-85, 2001; note that the
/// **kinetic** scheme of Destexhe, Mainen & Sejnowski (1998), with `β = 190 s⁻¹`, gives 5.3 ms
/// instead — see [`KineticTwoState::ampa`], which reproduces that number. The two are not
/// reconciled here and the discrepancy is real rather than a transcription error: 2 ms is a fit to
/// fast cortical `AMPA`-receptor-mediated currents, 5.3 ms comes from a kinetic fit to a different
/// preparation. Pick one deliberately.
pub const AMPA: Receptor = Receptor {
    name: "AMPA",
    e_rev: 0.0,
    tau_rise: 0.5e-3,
    tau_decay: 2.0e-3,
    mg_block: None,
    source: "Brunel & Wang, J. Comput. Neurosci. 11:63-85, 2001 (cortical AMPA)",
};

/// Slow excitatory glutamate receptor with a magnesium block. Reversal 0 mV, rise 2 ms, decay
/// 100 ms.
///
/// Same reversal potential as `AMPA` — both are non-selective cation channels — and a decay fifty
/// times longer, which is what makes `NMDA` the substrate for persistent activity in working-memory
/// models. Its conductance is voltage-gated by [`MgBlock`], so the current it delivers is small at
/// rest whatever the presynaptic rate.
///
/// Constants from Brunel & Wang, *J. Comput. Neurosci.* 11:63-85, 2001. The decay in the wider
/// literature ranges from roughly 50 ms to over 200 ms depending on subunit composition, and the
/// 100 ms here is a central value rather than a measurement.
pub const NMDA: Receptor = Receptor {
    name: "NMDA",
    e_rev: 0.0,
    tau_rise: 2.0e-3,
    tau_decay: 100.0e-3,
    mg_block: Some(MgBlock { mg_mm: 1.0, slope_per_mv: 0.062, k_mm: 3.57 }),
    source: "Brunel & Wang, J. Comput. Neurosci. 11:63-85, 2001; block from Jahr & Stevens, \
             J. Neurosci. 10:3178-3182, 1990",
};

/// Fast inhibitory `GABA` receptor, an ionotropic chloride channel. Reversal −70 mV, rise 0.5 ms,
/// decay 5.6 ms.
///
/// The decay is `1/β` with `β = 180 s⁻¹` from the two-state kinetic scheme of Destexhe, Mainen &
/// Sejnowski (1998), which is where [`KineticTwoState::gaba_a`] gets the same number — so this row
/// and that one are consistent by construction, unlike [`AMPA`]'s.
///
/// **The reversal potential is the interesting parameter, not the decay.** It is set by the
/// chloride gradient, which the cell maintains actively, and it moves: near or above rest in
/// immature neurons, where `GABA_A` is depolarising, and typically −65 to −80 mV in mature cortex.
/// −70 mV as shipped sits essentially at a typical resting potential, which makes this receptor
/// **shunting** by default — see `shunting_inhibition_suppresses_without_moving_rest`. Move it to
/// −80 mV and the same synapse hyperpolarises instead. This implementation did not locate a single
/// canonical value, because there is not one.
pub const GABA_A: Receptor = Receptor {
    name: "GABA_A",
    e_rev: -70.0e-3,
    tau_rise: 0.5e-3,
    tau_decay: 5.6e-3,
    mg_block: None,
    source: "Destexhe, Mainen & Sejnowski, Methods in Neuronal Modeling 2nd ed., 1998 (beta = 180/s)",
};

/// Slow inhibitory `GABA` receptor, metabotropic and potassium-mediated. Reversal −95 mV, rise
/// 60 ms, decay 200 ms.
///
/// ⚠ **This row is a phenomenological fit, not a mechanism.** `GABA_B` does not gate a channel
/// directly: it activates a G protein, which opens an inward-rectifying potassium channel, with a
/// fourth-power cooperativity between the two. That cascade is [`GabaBCascade`], and it produces
/// the sigmoidal onset and the strong dependence on presynaptic burst length that a bi-exponential
/// cannot. The constants here are round values in the range the slow inhibitory postsynaptic
/// potential literature reports — this implementation did not fit them to a specific published
/// trace, and a study of `GABA_B`'s nonlinearity should use the cascade instead of this row.
///
/// The reversal potential is the one firm number: −95 mV is the potassium equilibrium potential,
/// well below rest, so `GABA_B` inhibition is genuinely hyperpolarising rather than shunting.
pub const GABA_B: Receptor = Receptor {
    name: "GABA_B",
    e_rev: -95.0e-3,
    tau_rise: 60.0e-3,
    tau_decay: 200.0e-3,
    mg_block: None,
    source: "phenomenological fit; mechanism in Destexhe & Sejnowski, PNAS 92:9515-9519, 1995",
};

/// Every receptor in this module, for a caller that wants to sweep them or print the table.
///
/// Ordered fast to slow within each sign: excitation then inhibition, `AMPA`, `NMDA`, `GABA_A`,
/// `GABA_B`. The decay constants span 0.002 s to 0.2 s, a factor of 100.
pub const RECEPTORS: [Receptor; 4] = [AMPA, NMDA, GABA_A, GABA_B];

/// The two-state kinetic scheme: `dr/dt = α·[T]·(1 - r) - β·r`.
///
/// Destexhe, Mainen & Sejnowski, *Neural Computation* 6:14-18, 1994, "An efficient method for
/// computing synaptic conductances based on a kinetic model of receptor binding", with the rate
/// constants as tabulated in the same authors' "Kinetic models of synaptic transmission", *Methods
/// in Neuronal Modeling* (2nd ed.), 1998.
///
/// # Why a kernel is not the whole story
///
/// [`Alpha`] and [`BiExponential`] are shapes fitted to a measurement. This is the mechanism the
/// shapes approximate: `r` is the fraction of receptors in the open state, `[T]` is the transmitter
/// in the cleft, and a presynaptic spike releases a **brief square pulse** of transmitter — 1 mM
/// for 1 ms is the standard idealisation. The insight of the 1994 paper is that a square pulse
/// makes the equation piecewise linear and therefore **exactly solvable**, so a biophysically
/// grounded synapse costs one exponential per step rather than a differential-equation solver.
///
/// It also gives the thing a fitted kernel cannot: **saturation**. Two spikes 0.2 ms apart arrive
/// while the receptors are still open, and the second one adds far less than the first. A linear
/// kernel superposes without limit; this does not.
///
/// # The closed form, which is what the test checks
///
/// During the pulse the equation relaxes toward `r_∞ = α[T]/(α[T] + β)` with rate `α[T] + β`; after
/// it, `r` decays as `exp(-β·t)`. So a single 1 ms pulse leaves a peak of
/// `r_∞·(1 - exp(-(α[T] + β)·t_pulse))` and a tail with time constant `1/β` — for [`AMPA`],
/// 0.618 and 5.26 ms.
///
/// # Units
///
/// `alpha` is per second per millimolar and `beta` is per second, because the 1998 table is in
/// `M⁻¹s⁻¹` and transmitter concentration is millimolar throughout that literature. `1.1e6 M⁻¹s⁻¹`
/// is `1100 mM⁻¹s⁻¹`. Durations are SI seconds like everything else.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KineticTwoState {
    /// Forward binding rate, **per second per millimolar**.
    pub alpha: f64,
    /// Unbinding (closing) rate, **per second**. Its reciprocal is the decay time constant of the
    /// open fraction once transmitter has cleared.
    pub beta: f64,
    /// Transmitter concentration during a release event, **millimolar**. 1.0 in the source's
    /// idealisation.
    pub t_max_mm: f64,
    /// Duration of the transmitter pulse, **seconds**. 1 ms in the source's idealisation.
    pub t_pulse: f64,
    /// Fraction of receptors open, dimensionless, invariant `0 <= r <= 1`.
    pub r: f64,
    /// Seconds of transmitter pulse still to run; zero when the cleft is clear.
    pub pulse_left: f64,
}

impl KineticTwoState {
    /// `AMPA`: `α = 1100 mM⁻¹s⁻¹`, `β = 190 s⁻¹`, giving a 5.26 ms decay.
    ///
    /// `1.1e6 M⁻¹s⁻¹` as the 1998 table prints it. Note this decay is longer than the 2 ms in the
    /// [`AMPA`] receptor row; the discrepancy is discussed there and is not resolved here.
    #[must_use]
    pub fn ampa() -> Self {
        Self { alpha: 1100.0, beta: 190.0, t_max_mm: 1.0, t_pulse: 1e-3, r: 0.0, pulse_left: 0.0 }
    }

    /// `GABA_A`: `α = 5000 mM⁻¹s⁻¹`, `β = 180 s⁻¹`, giving a 5.56 ms decay — the constant the
    /// [`GABA_A`] receptor row is built from, so the two agree by construction.
    #[must_use]
    pub fn gaba_a() -> Self {
        Self { alpha: 5000.0, beta: 180.0, t_max_mm: 1.0, t_pulse: 1e-3, r: 0.0, pulse_left: 0.0 }
    }

    /// A scheme with stated rate constants.
    ///
    /// # Errors
    ///
    /// [`SynapseError::Negative`] for a negative rate, concentration or duration, and
    /// [`SynapseError::NonFinite`] for any non-finite argument.
    pub fn new(
        alpha: f64,
        beta: f64,
        t_max_mm: f64,
        t_pulse: f64,
    ) -> Result<Self, SynapseError> {
        Ok(Self {
            alpha: non_negative("alpha", alpha)?,
            beta: non_negative("beta", beta)?,
            t_max_mm: non_negative("t_max_mm", t_max_mm)?,
            t_pulse: non_negative("t_pulse", t_pulse)?,
            r: 0.0,
            pulse_left: 0.0,
        })
    }

    /// Steady-state open fraction under a sustained transmitter concentration, dimensionless.
    ///
    /// `α[T]/(α[T] + β)`. `None` when both rates are zero, where the equation has no dynamics and
    /// every `r` is a steady state — there is no single value to report.
    #[must_use]
    pub fn steady_open_fraction(&self, t_mm: f64) -> Option<f64> {
        let on = self.alpha * t_mm;
        let sum = on + self.beta;
        if sum <= 0.0 { None } else { Some(on / sum) }
    }

    /// Decay time constant of the open fraction once transmitter has cleared, seconds.
    ///
    /// `1/β`. `None` for `β = 0`, where the receptors never close and there is no decay to report.
    #[must_use]
    pub fn decay_tau(&self) -> Option<f64> {
        if self.beta > 0.0 { Some(1.0 / self.beta) } else { None }
    }

    /// Release transmitter: start (or restart) a pulse of `t_pulse` seconds.
    ///
    /// Restarting rather than extending is what the source's idealisation does, and it is why the
    /// scheme saturates — a second spike during a pulse does not double the transmitter.
    pub fn release(&mut self) {
        self.pulse_left = self.t_pulse;
    }

    /// Advance by `dt` seconds, exactly, splitting the step at the end of the transmitter pulse.
    ///
    /// The split is the whole reason this is exact. Integrating a step that straddles the end of
    /// the pulse with one rate constant would smear the pulse boundary by up to a full step, which
    /// is the error that makes a "biophysical" synapse's amplitude depend on the time step.
    ///
    /// A `dt` that is zero, negative or non-finite leaves the state unchanged; see [`check_dt`].
    pub fn advance(&mut self, dt: f64) {
        if !steppable(dt) {
            return;
        }
        let in_pulse = dt.min(self.pulse_left.max(0.0));
        if in_pulse > 0.0 {
            self.relax(in_pulse, self.t_max_mm);
            self.pulse_left -= in_pulse;
        }
        let after = dt - in_pulse;
        if after > 0.0 {
            self.relax(after, 0.0);
        }
    }

    fn relax(&mut self, dt: f64, t_mm: f64) {
        let on = self.alpha * t_mm;
        let rate = on + self.beta;
        if rate <= 0.0 {
            return;
        }
        let r_inf = on / rate;
        self.r = r_inf + (self.r - r_inf) * (-rate * dt).exp();
    }

    /// Forget the open fraction and any pulse in progress.
    pub fn reset(&mut self) {
        self.r = 0.0;
        self.pulse_left = 0.0;
    }
}

/// The `GABA_B` G-protein cascade: four variables and a fourth-power cooperativity.
///
/// Destexhe & Sejnowski, *PNAS* 92:9515-9519, 1995, in the form tabulated by Destexhe, Mainen &
/// Sejnowski, *Methods in Neuronal Modeling* (2nd ed.), 1998:
///
/// ```text
/// dr/dt = K1·[T]·(1 - r) - K2·r        receptor activation
/// dG/dt = K3·r - K4·G                  G-protein concentration
/// g/g_max = G^n / (G^n + Kd)           channel opening, n = 4
/// ```
///
/// # What the cascade buys that a kernel cannot
///
/// The Hill term is the point. Because `G` has to reach a threshold before the channels open at
/// all, a **single** presynaptic spike produces almost no `GABA_B` current while a **burst**
/// produces a large one — the response is a nonlinear function of the presynaptic burst, not a sum
/// of per-spike kernels. A bi-exponential fit ([`GABA_B`]) reproduces the time course of one
/// response and gets the burst dependence wrong, which is the difference between a shape and a
/// mechanism.
///
/// # ⚠ Confidence in these constants
///
/// The rate constants shipped in [`GabaBCascade::default`] are transcribed from the secondary
/// literature in millisecond units — `K1 = 0.09 ms⁻¹mM⁻¹`, `K2 = 0.0012 ms⁻¹`, `K3 = 0.18 ms⁻¹`,
/// `K4 = 0.034 ms⁻¹`, `Kd = 100`, `n = 4` — and converted to SI here. **This implementation did not
/// verify them against the original figures in the 1995 paper.** The steady-state test in this
/// module checks the integrator against the model's own closed form, which validates the arithmetic
/// and says nothing about whether the constants describe a real synapse. Treat the shape as
/// qualitative until you have checked the table yourself.
///
/// # Integration
///
/// `r` is integrated exactly. `G` is integrated by the exact convolution of its own exponential
/// against `r`'s, which contains a **removable singularity at `K4 = K1[T] + K2`**: the difference
/// quotient `(e^{-at} - e^{-K4·t})/(K4 - a)` is `0/0` there and becomes `t·e^{-K4·t}`.
///
/// It is removed algebraically rather than branched around. Factoring out `e^{-K4·t}` rewrites it
/// as `dt·e^{-K4·dt}·expm1(z)/z` with `z = (K4 - a)·dt`, and `expm1` is accurate precisely where
/// `exp(z) - 1` cancels — so one expression covers every `z` and only `z = 0` exactly needs a
/// branch. The quantity that has to be small for the naive form to fail is the **product**
/// `(K4 - a)·dt`, not the rate difference on its own, which is the trap a threshold on the rates
/// alone walks into.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GabaBCascade {
    /// Receptor binding rate, **per second per millimolar**.
    pub k1: f64,
    /// Receptor unbinding rate, **per second**.
    pub k2: f64,
    /// G-protein production rate from activated receptor, **per second**.
    pub k3: f64,
    /// G-protein decay rate, **per second**. Its reciprocal, about 29 ms, sets the tail.
    pub k4: f64,
    /// Dissociation constant of the Hill term, in the same arbitrary units as `G^n`. Dimensionless
    /// here because the source's `G` is not given a concentration scale.
    pub kd: f64,
    /// Hill coefficient — the number of G-protein subunits that must bind. `4` in the source, and
    /// the reason the response is a nonlinear function of burst length rather than a sum.
    pub n: u32,
    /// Fraction of receptors activated, dimensionless, invariant `0 <= r <= 1`.
    pub r: f64,
    /// G-protein concentration in the source's arbitrary units, invariant `g_conc >= 0`.
    pub g_conc: f64,
    /// Transmitter pulse amplitude, **millimolar**.
    pub t_max_mm: f64,
    /// Transmitter pulse duration, **seconds**. `GABA_B` uses a longer pulse than `GABA_A` in the
    /// source — 0.3 ms to several ms depending on the fit; 1 ms here.
    pub t_pulse: f64,
    /// Seconds of transmitter pulse still to run.
    pub pulse_left: f64,
}

impl Default for GabaBCascade {
    /// The constants named in the type's doc, converted from the source's millisecond units. Read
    /// the confidence warning there before citing any of them.
    fn default() -> Self {
        Self {
            k1: 90.0,
            k2: 1.2,
            k3: 180.0,
            k4: 34.0,
            kd: 100.0,
            n: 4,
            r: 0.0,
            g_conc: 0.0,
            t_max_mm: 1.0,
            t_pulse: 1e-3,
            pulse_left: 0.0,
        }
    }
}

impl GabaBCascade {
    /// Fraction of channels open for the current G-protein concentration, in `[0, 1)`.
    ///
    /// `G^n/(G^n + Kd)`. Multiply by a peak conductance in siemens to get a conductance.
    #[must_use]
    pub fn open_fraction(&self) -> f64 {
        let gn = self.g_conc.powi(i32::try_from(self.n).unwrap_or(i32::MAX));
        if gn.is_finite() { gn / (gn + self.kd) } else { 1.0 }
    }

    /// Steady state under a **sustained** transmitter concentration `t_mm` millimolar, in closed
    /// form: `(r, G, open fraction)`.
    ///
    /// `r = K1[T]/(K1[T] + K2)`, `G = K3·r/K4`, and the Hill term of that `G`. This is what the
    /// module's cascade test checks the integrator against — and, to say it once more, it checks
    /// the arithmetic, not the constants.
    ///
    /// `None` when `K4` is not positive, where `G` has no steady state to report.
    #[must_use]
    pub fn steady_state(&self, t_mm: f64) -> Option<(f64, f64, f64)> {
        if self.k4 <= 0.0 {
            return None;
        }
        let on = self.k1 * t_mm;
        let sum = on + self.k2;
        if sum <= 0.0 {
            return None;
        }
        let r = on / sum;
        let g = self.k3 * r / self.k4;
        let gn = g.powi(i32::try_from(self.n).unwrap_or(i32::MAX));
        Some((r, g, gn / (gn + self.kd)))
    }

    /// Release transmitter: start (or restart) a pulse of `t_pulse` seconds.
    pub fn release(&mut self) {
        self.pulse_left = self.t_pulse;
    }

    /// Advance by `dt` seconds, splitting the step at the end of the transmitter pulse.
    ///
    /// A `dt` that is zero, negative or non-finite leaves the state unchanged; see [`check_dt`].
    pub fn advance(&mut self, dt: f64) {
        if !steppable(dt) {
            return;
        }
        let in_pulse = dt.min(self.pulse_left.max(0.0));
        if in_pulse > 0.0 {
            self.relax(in_pulse, self.t_max_mm);
            self.pulse_left -= in_pulse;
        }
        let after = dt - in_pulse;
        if after > 0.0 {
            self.relax(after, 0.0);
        }
    }

    fn relax(&mut self, dt: f64, t_mm: f64) {
        let on = self.k1 * t_mm;
        let a = on + self.k2;
        let r0 = self.r;
        let r_inf = if a > 0.0 { on / a } else { r0 };
        let er = if a > 0.0 { (-a * dt).exp() } else { 1.0 };
        let r1 = r_inf + (r0 - r_inf) * er;

        if self.k4 > 0.0 {
            let e4 = (-self.k4 * dt).exp();
            let base = self.g_conc * e4 + self.k3 * r_inf * (1.0 - e4) / self.k4;
            // The removable singularity, removed ALGEBRAICALLY rather than branched around.
            // `(er - e4)/(k4 - a)` is a difference of two nearly equal exponentials over a nearly
            // zero denominator. Factoring `e4` out turns it into `dt · e4 · expm1(z)/z` with
            // `z = (k4 - a)·dt`, and `expm1` is built to be accurate exactly where `exp(z) - 1`
            // is not — so one expression is correct for every `z`, and only `z == 0` needs a
            // branch at all.
            //
            // A first version switched on `|k4 - a| > 1e-9·max(k4, a)` instead, and was WRONG on
            // both sides of the switch: the relevant small quantity is the product `(k4 - a)·dt`,
            // not the rate difference, so a threshold on the rate alone handed the subtractive
            // branch cases where it had already lost eight digits. The test
            // `the_cascade_handles_its_removable_singularity` caught it as a centring error that
            // refused to shrink when the perturbation was halved — curvature shrinks like `ε²`, a
            // discontinuity does not shrink at all.
            let z = (self.k4 - a) * dt;
            let shape = if z == 0.0 { dt * e4 } else { dt * e4 * z.exp_m1() / z };
            self.g_conc = base + self.k3 * (r0 - r_inf) * shape;
        }
        self.r = r1;
    }

    /// Return to the unstimulated state.
    pub fn reset(&mut self) {
        self.r = 0.0;
        self.g_conc = 0.0;
        self.pulse_left = 0.0;
    }
}

/// The steady state of a [`TsodyksMarkram`] synapse under regular stimulation, in closed form.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SteadyState {
    /// Utilisation immediately after each spike's facilitation step, dimensionless, `0 < u <= 1`.
    pub u: f64,
    /// Available fraction immediately **before** each spike, dimensionless, `0 <= x <= 1`.
    pub x: f64,
    /// Fraction of the full weight released by each spike, `u·x`. This is the number an
    /// experimenter plots as the steady-state postsynaptic amplitude, relative to the amplitude a
    /// fully recovered synapse would give.
    pub release: f64,
    /// Released fraction per second, `release · rate`. The quantity that converges to `1/τ_d`.
    pub transmitted_per_second: f64,
}

/// The `Tsodyks`-`Markram` model of short-term synaptic plasticity.
///
/// Tsodyks & Markram, *PNAS* 94:719-723, 1997, "The neural code between neocortical pyramidal
/// neurons depends on neurotransmitter release probability"; facilitation added in Markram, Wang &
/// Tsodyks, *PNAS* 95:5323-5328, 1998; the recursion used here is the discrete form of Tsodyks,
/// Pawelzik & Markram, *Neural Computation* 10:821-835, 1998.
///
/// # The model
///
/// Two state variables, both fractions:
///
/// - `x` — the fraction of the synapse's resources available. A spike consumes `u·x` of it and it
///   recovers toward 1 with time constant `τ_d`.
/// - `u` — the fraction of what is available that each spike releases. A spike steps it up by
///   `U·(1 - u)` and it decays back toward 0 with `τ_f`.
///
/// On each spike, in this order: `u ← u + U(1 - u)`, then release `A = u·x`, then `x ← x - A`. From
/// rest (`u = 0`, `x = 1`) the first spike releases exactly `U`, which is what makes `U` "the
/// release probability of a rested synapse" rather than a fitted constant with no meaning.
///
/// # Both regimes, and the rate that decides between them
///
/// With `τ_f = 0` the synapse only depletes: every spike releases `U` of whatever is left and the
/// train **depresses**. With `τ_f` long and `U` small, `u` accumulates faster than `x` depletes and
/// the train **facilitates** — but only up to a rate. [`TsodyksMarkram::facilitating`] facilitates
/// 1.8-fold at 20 Hz and depresses below its first response at 100 Hz, because `u` saturates at 1
/// while `x` keeps falling. Short-term plasticity is not a property of a synapse; it is a property
/// of a synapse **at a rate**.
///
/// # The steady state, which is the closed form this is checked against
///
/// Under regular stimulation at rate `r`, with `Δ = 1/r`, `f = exp(-Δ/τ_f)` and `d = exp(-Δ/τ_d)`:
///
/// ```text
/// u* = U / (1 - (1 - U)·f)
/// x* = (1 - d) / (1 - (1 - u*)·d)
/// A* = u*·x*
/// ```
///
/// Both are fixed points of an affine recursion, so both are exact. [`TsodyksMarkram::steady_state`]
/// returns them and a simulation is checked against them at six rates for two parameter sets, to
/// better than one part in `1e9`.
///
/// # The result the model is famous for
///
/// `A*·r → 1/τ_d` as `r → ∞`, **independently of `U` and `τ_f`**. A depressing synapse has a
/// ceiling on what it can transmit per second that depends only on how fast it refills, so above
/// that rate the postsynaptic cell stops hearing the presynaptic rate at all and hears only its
/// *changes*. That is [`TsodyksMarkram::limiting_transmission_rate`], and it is the reason this
/// model is in a library about energy: a synapse driven past its limiting rate is spending spikes
/// that carry no information.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TsodyksMarkram {
    /// `U`, the fraction released by a spike arriving at a fully rested synapse. Dimensionless,
    /// `0 < U <= 1`. Large for depressing connections, small for facilitating ones.
    pub u_rest: f64,
    /// `τ_d`, the recovery time constant of the available resources, **seconds**. Its reciprocal is
    /// the synapse's limiting transmission rate.
    pub tau_d: f64,
    /// `τ_f`, the decay time constant of the utilisation, **seconds**. Exactly `0.0` means no
    /// facilitation at all, and is handled as a branch rather than as a very small number.
    pub tau_f: f64,
    /// Current utilisation, dimensionless, invariant `0 <= u <= 1`. Zero at rest, stepping up on
    /// each spike.
    pub u: f64,
    /// Current available fraction, dimensionless, invariant `0 <= x <= 1`. One at rest.
    pub x: f64,
}

impl TsodyksMarkram {
    /// A depressing synapse: `U = 0.5`, `τ_d = 800 ms`, no facilitation.
    ///
    /// The pyramidal-to-pyramidal regime of Tsodyks & Markram, *PNAS* 94:719-723, 1997. These are
    /// **round central values** from that paper's range, not a refit of a particular recorded pair;
    /// the individual connections there vary considerably. Its limiting transmission rate is
    /// 1.25 releases per second.
    #[must_use]
    pub fn depressing() -> Self {
        Self { u_rest: 0.5, tau_d: 0.8, tau_f: 0.0, u: 0.0, x: 1.0 }
    }

    /// A facilitating synapse: `U = 0.15`, `τ_d = 130 ms`, `τ_f = 530 ms`.
    ///
    /// ⚠ **Round values inside the range** Markram, Wang & Tsodyks, *PNAS* 95:5323-5328, 1998
    /// report for facilitating pyramidal-to-interneuron connections. This implementation did not
    /// refit them to a specific table entry in that paper, and the individual synapses fitted there
    /// span more than a factor of two in every parameter. Use them for a demonstration of the
    /// regime, not as a measurement of a connection.
    #[must_use]
    pub fn facilitating() -> Self {
        Self { u_rest: 0.15, tau_d: 0.13, tau_f: 0.53, u: 0.0, x: 1.0 }
    }

    /// A synapse with stated parameters, at rest.
    ///
    /// # Errors
    ///
    /// [`SynapseError::FractionOutOfRange`] if `u_rest` is outside `(0, 1]` — zero is refused
    /// because a synapse that releases nothing has no dynamics and its steady state is a division
    /// by zero, not a small number. [`SynapseError::NonPositiveTimeConstant`] for a `tau_d` that is
    /// not strictly positive, [`SynapseError::Negative`] for a negative `tau_f` (zero is legal and
    /// means no facilitation), and [`SynapseError::NonFinite`] for any non-finite argument.
    pub fn new(u_rest: f64, tau_d: f64, tau_f: f64) -> Result<Self, SynapseError> {
        let u_rest = unit_fraction("u_rest", u_rest)?;
        if u_rest <= 0.0 {
            return Err(SynapseError::FractionOutOfRange { name: "u_rest", value: u_rest });
        }
        Ok(Self {
            u_rest,
            tau_d: positive_tau("tau_d", tau_d)?,
            tau_f: non_negative("tau_f", tau_f)?,
            u: 0.0,
            x: 1.0,
        })
    }

    /// Let `dt` seconds pass with no presynaptic spike.
    ///
    /// `x` recovers toward 1 with `τ_d` and `u` decays toward 0 with `τ_f`, both by exact
    /// exponentials, so the answer does not depend on how the interval was subdivided. A `dt` that
    /// is zero, negative or non-finite leaves the state unchanged; see [`check_dt`].
    pub fn advance(&mut self, dt: f64) {
        if !steppable(dt) {
            return;
        }
        if self.tau_f > 0.0 {
            self.u *= (-dt / self.tau_f).exp();
        } else {
            // Exactly zero means no facilitation, handled as a branch: `exp(-dt/0.0)` is `exp(-inf)`
            // which is 0.0 and happens to be right, but `exp(-0.0/0.0)` is NaN and would poison the
            // state on a zero-length step. A branch says what is meant.
            self.u = 0.0;
        }
        self.x = 1.0 + (self.x - 1.0) * (-dt / self.tau_d).exp();
    }

    /// Deliver a presynaptic spike; returns the fraction of the full weight released, in `[0, 1]`.
    ///
    /// Multiply a synaptic weight by this to get the effective weight of **this** spike. The order
    /// inside — facilitate, release, deplete — is the order in the source, and reversing the first
    /// two makes the first spike from rest release `0` instead of `U`.
    pub fn spike(&mut self) -> f64 {
        self.u += self.u_rest * (1.0 - self.u);
        let released = self.u * self.x;
        self.x -= released;
        released
    }

    /// Wait `interval` seconds and then spike; returns the released fraction.
    ///
    /// The convenience form for driving a regular train, which is exactly what the steady-state
    /// closed form describes.
    pub fn spike_after(&mut self, interval: f64) -> f64 {
        self.advance(interval);
        self.spike()
    }

    /// The closed-form steady state under regular stimulation at `rate_hz`.
    ///
    /// # Errors
    ///
    /// [`SynapseError::NonPositiveRate`] for a rate that is zero or negative, and
    /// [`SynapseError::NonFinite`] for `NaN` or an infinity. A synapse that is never stimulated has
    /// no steady-state release, and reporting its rest state instead would answer a different
    /// question.
    pub fn steady_state(&self, rate_hz: f64) -> Result<SteadyState, SynapseError> {
        let rate = finite("rate_hz", rate_hz)?;
        if rate <= 0.0 {
            return Err(SynapseError::NonPositiveRate { value: rate });
        }
        let interval = 1.0 / rate;
        let f = if self.tau_f > 0.0 { (-interval / self.tau_f).exp() } else { 0.0 };
        let d = (-interval / self.tau_d).exp();
        let u = self.u_rest / (1.0 - (1.0 - self.u_rest) * f);
        let x = (1.0 - d) / (1.0 - (1.0 - u) * d);
        let release = u * x;
        Ok(SteadyState { u, x, release, transmitted_per_second: release * rate })
    }

    /// The ceiling on released fraction per second, `1/τ_d`, in hertz.
    ///
    /// The limit of [`SteadyState::transmitted_per_second`] as the presynaptic rate goes to
    /// infinity — and it does not depend on `U` or `τ_f`, which is the model's central result. A
    /// presynaptic train faster than this transmits no additional signal, only additional spikes.
    #[must_use]
    pub fn limiting_transmission_rate(&self) -> f64 {
        1.0 / self.tau_d
    }

    /// Return to the rested state: `u = 0`, `x = 1`.
    pub fn reset(&mut self) {
        self.u = 0.0;
        self.x = 1.0;
    }
}

#[cfg(test)]
mod tests {
    use super::{
        AMPA, Alpha, BiExponential, ConductanceBased, CurrentBased, Delta, Drive, Exponential,
        GABA_A, GABA_B, GabaBCascade, Kernel, KineticTwoState, MgBlock, NMDA, RECEPTORS, Receptor,
        SynapseError, TsodyksMarkram, check_dt,
    };
    use crate::neuron::{Lif, Neuron};
    use std::f64::consts::E;

    // ---------------------------------------------------------------- kernels, analytic

    /// (a) The alpha kernel peaks at exactly `t = τ` with value exactly 1.
    ///
    /// Both halves are exact in floating point and are asserted as such: `t/τ` with `t = τ` is
    /// `1.0` bit-for-bit, `exp(1.0 - 1.0)` is `exp(0.0)` is `1.0` bit-for-bit. A tolerance here
    /// would hide a normalisation that was merely close.
    #[test]
    fn the_alpha_kernel_peaks_at_exactly_tau_with_value_exactly_one() {
        for &tau in &[1e-4, 5e-4, 2e-3, 5e-3, 20e-3, 0.1] {
            let k = Alpha::new(tau).expect("positive tau");
            assert_eq!(k.peak_time(), Some(tau));
            assert_eq!(k.peak_value(), Some(1.0));
            assert_eq!(k.response(tau), Some(1.0), "response at tau for tau = {tau}");

            // And nothing in a fine scan beats it, which is what "peak" has to mean.
            let n = 200_000;
            for i in 0..=n {
                let t = 10.0 * tau * f64::from(i) / f64::from(n);
                let v = k.response(t).expect("finite t");
                assert!(v <= 1.0, "response({t}) = {v} exceeds the peak for tau = {tau}");
            }
        }
    }

    /// The derivative of the alpha kernel vanishes at `t = τ` and nowhere else on `(0, ∞)`.
    ///
    /// The peak location is a claim about `dk/dt = (1/τ)e^{1-t/τ}(1 - t/τ)`, so checking the
    /// derivative's sign either side of `τ` tests the derivation and not just one sample.
    #[test]
    fn the_alpha_kernels_derivative_changes_sign_only_at_tau() {
        let tau = 5e-3;
        let k = Alpha::new(tau).expect("positive tau");
        let h = 1e-9;
        let slope = |t: f64| {
            (k.response(t + h).expect("finite") - k.response(t - h).expect("finite")) / (2.0 * h)
        };
        for &frac in &[0.05, 0.2, 0.5, 0.9, 0.99] {
            assert!(slope(frac * tau) > 0.0, "rising branch at {frac} tau");
        }
        for &frac in &[1.01, 1.1, 2.0, 5.0, 20.0] {
            assert!(slope(frac * tau) < 0.0, "falling branch at {frac} tau");
        }
    }

    /// (b) The exponential kernel's time integral is exactly `τ`.
    ///
    /// Asserted twice: the closed form returns the stored `τ` bit-for-bit, and a trapezoidal
    /// quadrature of the analytic response agrees with it. The quadrature is the half that would
    /// catch a `response` and an `integral_seconds` that agreed with each other and with nothing
    /// else.
    #[test]
    fn the_exponential_kernels_time_integral_is_exactly_tau() {
        for &tau in &[1e-4, 2e-3, 5e-3, 50e-3] {
            let k = Exponential::new(tau).expect("positive tau");
            assert_eq!(k.integral_seconds(), Some(tau));

            // 40 τ of tail leaves exp(-40) ≈ 4e-18 outside, far below the quadrature error.
            let span = 40.0 * tau;
            let n = 400_000u32;
            let h = span / f64::from(n);
            let mut sum = 0.5 * (k.response(0.0).expect("finite") + k.response(span).expect("f"));
            for i in 1..n {
                sum += k.response(f64::from(i) * h).expect("finite");
            }
            let numeric = sum * h;
            let rel = (numeric - tau).abs() / tau;
            assert!(rel < 1e-9, "tau = {tau}: quadrature {numeric} vs closed form {tau}");
        }
    }

    /// The alpha kernel's integral is exactly `e·τ` — about 2.72 times the exponential's at the
    /// same `τ`, which is the factor that surprises people swapping one for the other.
    #[test]
    fn the_alpha_kernels_time_integral_is_e_times_tau() {
        for &tau in &[2e-4, 1e-3, 5e-3, 20e-3] {
            let k = Alpha::new(tau).expect("positive tau");
            assert_eq!(k.integral_seconds(), Some(E * tau));

            let span = 60.0 * tau;
            let n = 600_000u32;
            let h = span / f64::from(n);
            let mut sum = 0.5 * (k.response(0.0).expect("finite") + k.response(span).expect("f"));
            for i in 1..n {
                sum += k.response(f64::from(i) * h).expect("finite");
            }
            let numeric = sum * h;
            let want = E * tau;
            let rel = (numeric - want).abs() / want;
            assert!(rel < 1e-9, "tau = {tau}: quadrature {numeric} vs closed form {want}");
        }
    }

    /// The bi-exponential's peak is exactly 1 at its closed-form peak time, over a grid of
    /// separations, and no sample anywhere beats it.
    #[test]
    fn the_bi_exponential_peaks_at_one_at_its_closed_form_time() {
        let pairs = [
            (0.5e-3, 2.0e-3),
            (2.0e-3, 100.0e-3),
            (0.1e-3, 50.0e-3),
            (1.0e-3, 1.05e-3),
            (60.0e-3, 200.0e-3),
        ];
        for &(tr, td) in &pairs {
            let k = BiExponential::new(tr, td).expect("rise <= decay, both positive");
            let tp = k.peak_time().expect("a bi-exponential has a peak");
            assert!(tp > 0.0, "peak time {tp} for ({tr}, {td})");
            let at_peak = k.response(tp).expect("finite t");
            assert!(
                (at_peak - 1.0).abs() < 1e-12,
                "({tr}, {td}): response at the closed-form peak is {at_peak}"
            );
            let n = 100_000u32;
            for i in 0..=n {
                let t = 40.0 * td * f64::from(i) / f64::from(n);
                let v = k.response(t).expect("finite t");
                assert!(v <= 1.0 + 1e-12, "({tr}, {td}): response({t}) = {v} beats the peak");
            }
        }
    }

    /// (c) The bi-exponential reduces to the alpha kernel as `τ_rise -> τ_decay`.
    ///
    /// Three claims, because the limit has three parts. **Exactly equal** constants must give the
    /// alpha function bit-for-bit. **Nearly equal** constants must converge to it through the
    /// general two-exponential formula, quadratically, which is what the harmonic mean buys — the
    /// arithmetic mean would converge only linearly and this test's tolerance would catch it. And
    /// the two branches must agree **across the switch**, so a caller sweeping the separation sees
    /// no step.
    #[test]
    fn the_bi_exponential_reduces_to_the_alpha_kernel_when_the_taus_meet() {
        let tau = 5e-3;

        // Exactly equal: the degenerate branch, and it must BE the alpha function.
        let bi = BiExponential::new(tau, tau).expect("equal taus are legal");
        assert!(bi.is_degenerate(), "equal taus should take the alpha limit");
        assert_eq!(bi.tau_harmonic(), tau);
        let al = Alpha::new(tau).expect("positive tau");
        for i in 0..=2_000u32 {
            let t = 20.0 * tau * f64::from(i) / 2000.0;
            let a = al.response(t).expect("finite t");
            let b = bi.response(t).expect("finite t");
            assert!((a - b).abs() < 1e-15, "t = {t}: alpha {a} vs bi-exponential {b}");
        }
        assert_eq!(bi.peak_time(), al.peak_time());
        assert_eq!(bi.integral_seconds(), al.integral_seconds());

        // Nearly equal, through the GENERAL branch: relative separation 2e-4, so the alpha limit
        // is wrong by O(delta^2) ~ 4e-8 and must be right to about that.
        let near = BiExponential::new(tau * (1.0 - 1e-4), tau * (1.0 + 1e-4)).expect("valid");
        assert!(!near.is_degenerate(), "2e-4 separation should use the exact formula");
        let limit = Alpha::new(near.tau_harmonic()).expect("positive tau");
        let mut worst: f64 = 0.0;
        for i in 1..=2_000u32 {
            let t = 20.0 * tau * f64::from(i) / 2000.0;
            let a = limit.response(t).expect("finite t");
            let b = near.response(t).expect("finite t");
            worst = worst.max((a - b).abs());
        }
        assert!(worst < 1e-6, "quadratic convergence to the alpha limit failed: worst {worst}");

        // Across the switch: just above and just below the threshold must agree closely.
        let above = BiExponential::new(tau * (1.0 - 1e-6), tau * (1.0 + 1e-6)).expect("valid");
        let below = BiExponential::new(tau * (1.0 - 1e-8), tau * (1.0 + 1e-8)).expect("valid");
        assert!(!above.is_degenerate());
        assert!(below.is_degenerate());
        for i in 1..=1_000u32 {
            let t = 20.0 * tau * f64::from(i) / 1000.0;
            let a = above.response(t).expect("finite t");
            let b = below.response(t).expect("finite t");
            assert!((a - b).abs() < 1e-9, "t = {t}: across the switch, {a} vs {b}");
        }
    }

    /// The degenerate branch is not merely safe, it is the more accurate one below the threshold.
    ///
    /// Evaluated against a high-precision reference built from the `sinh` form
    /// `2·e^{-mt}·sinh(ht)`, which has no cancellation: at a separation of `1e-9` the direct
    /// difference of exponentials has lost most of its significant digits while the alpha limit is
    /// still good to twelve. This is the claim in `BiExponential`'s doc, and it is the reason the
    /// switch exists at all.
    #[test]
    fn below_the_threshold_the_alpha_limit_beats_the_direct_formula() {
        let tau = 5e-3;
        // Relative separation 2e-12. The cancellation error of the direct formula is about
        // `eps/δ` ≈ 1e-4 here, so the two branches are separated by eight orders of magnitude and
        // the comparison does not depend on where exactly the threshold was put. A first draft at
        // 2e-9 left the direct formula only 6.5e-8 wrong, which is a real answer to a different
        // question.
        let (tr, td) = (tau * (1.0 - 1e-12), tau * (1.0 + 1e-12));
        let k = BiExponential::new(tr, td).expect("valid");
        assert!(k.is_degenerate());

        // Reference: normalised 2·e^{-mt}·sinh(ht) with m and h formed from the RECIPROCALS, where
        // the subtraction is benign. `sinh` of a tiny argument is accurate; the difference of two
        // exponentials near 1 is not.
        let a = 1.0 / td;
        let b = 1.0 / tr;
        let m = 0.5 * (a + b);
        let h = 0.5 * (b - a);
        let raw = |t: f64| 2.0 * (-m * t).exp() * (h * t).sinh();
        let tp = 1.0 / m;
        let peak = raw(tp);
        for i in 1..=500u32 {
            let t = 20.0 * tau * f64::from(i) / 500.0;
            let want = raw(t) / peak;
            let got = k.response(t).expect("finite t");
            assert!((got - want).abs() < 1e-11, "t = {t}: limit {got} vs sinh reference {want}");
        }

        // And the direct formula at this separation really is ruined. Two numbers near `1/e`
        // subtracted to leave 7e-13: the absolute round-off of the subtraction is a sizeable
        // fraction of the answer, so the normalisation it would produce is wrong in the fourth
        // decimal place. That is the failure the degenerate branch exists to avoid.
        let direct = |t: f64| (-t / td).exp() - (-t / tr).exp();
        let rel = ((direct(tp) / peak) - 1.0).abs();
        assert!(rel > 1e-6, "expected the direct formula to be visibly wrong here, was {rel}");
    }

    // ---------------------------------------------------------------- kernels, incremental

    /// The incremental form reproduces the analytic response exactly, for every kernel that has
    /// one.
    ///
    /// This is the contract stated on `Kernel`: `inject(w)` then `k` calls to `advance(dt)` leaves
    /// `value()` equal to `w · response(k·dt)`. It holds to floating-point noise rather than to a
    /// discretisation tolerance because every update here is the analytic solution over the step.
    #[test]
    fn the_incremental_forms_reproduce_the_analytic_response() {
        let dt = 1e-5;
        let w = 0.7;

        let mut e = Exponential::new(3e-3).expect("valid");
        let mut a = Alpha::new(3e-3).expect("valid");
        let mut b = BiExponential::new(0.5e-3, 5e-3).expect("valid");
        let mut deg = BiExponential::new(3e-3, 3e-3).expect("valid");
        e.inject(w);
        a.inject(w);
        b.inject(w);
        deg.inject(w);

        for step in 0..3_000u32 {
            let t = f64::from(step) * dt;
            for (name, got, want) in [
                ("exponential", e.value(), w * e.response(t).expect("finite")),
                ("alpha", a.value(), w * a.response(t).expect("finite")),
                ("bi-exponential", b.value(), w * b.response(t).expect("finite")),
                ("degenerate", deg.value(), w * deg.response(t).expect("finite")),
            ] {
                // 1e-12, and the number was measured rather than chosen. The recursion is exact in
                // exact arithmetic but multiplies by `d` once per step, so `d^k` computed by
                // repeated multiplication drifts from `exp(-k·dt/τ)` by about `k · eps`: at 3,000
                // steps that is 3e-13, and a first draft at 1e-14 tripped on it at step 308. This
                // is ROUND-OFF, not discretisation — the step-size test below is what tells the
                // two apart, because discretisation error would grow as the step coarsened and
                // round-off shrinks.
                assert!(
                    (got - want).abs() < 1e-12,
                    "{name} at t = {t}: incremental {got} vs analytic {want}"
                );
            }
            e.advance(dt);
            a.advance(dt);
            b.advance(dt);
            deg.advance(dt);
        }
    }

    /// A coarse step and a fine step give the same state, which is what `EXACT_OVER_STEPS` claims.
    ///
    /// Forward Euler would fail this and would still produce a plausible-looking response, which is
    /// how a synapse comes to have an amplitude that depends on the simulation's time step.
    #[test]
    fn the_incremental_forms_are_step_size_independent() {
        // `const` blocks: these are compile-time claims about the trait constants, so they are
        // checked at compile time and a kernel added later with a forward-Euler update fails the
        // BUILD rather than a test run.
        const { assert!(Delta::EXACT_OVER_STEPS) };
        const { assert!(Exponential::EXACT_OVER_STEPS) };
        const { assert!(Alpha::EXACT_OVER_STEPS) };
        const { assert!(BiExponential::EXACT_OVER_STEPS) };

        let total = 4e-3;
        let run = |steps: u32| {
            let mut a = Alpha::new(2e-3).expect("valid");
            let mut b = BiExponential::new(0.4e-3, 6e-3).expect("valid");
            let mut x = Exponential::new(2e-3).expect("valid");
            a.inject(1.0);
            b.inject(1.0);
            x.inject(1.0);
            let dt = total / f64::from(steps);
            for _ in 0..steps {
                a.advance(dt);
                b.advance(dt);
                x.advance(dt);
            }
            (a.value(), b.value(), x.value())
        };
        let fine = run(40_000);
        let coarse = run(4);
        // 1e-10: 40,000 repeated multiplications accumulate about `k · eps` = 4e-12 of round-off
        // against the 4-step run's essentially none. Forward Euler at 40,000 steps versus 4 would
        // differ here by parts in ten, not parts in 1e12, so this tolerance still separates the
        // two claims by eight orders of magnitude.
        assert!((fine.0 - coarse.0).abs() < 1e-10, "alpha {} vs {}", fine.0, coarse.0);
        assert!((fine.1 - coarse.1).abs() < 1e-10, "bi-exp {} vs {}", fine.1, coarse.1);
        assert!((fine.2 - coarse.2).abs() < 1e-10, "exp {} vs {}", fine.2, coarse.2);
    }

    /// Kernels are linear, so two spikes give the sum of two responses. A model that clipped or
    /// saturated here would be a different model, and the `KineticTwoState` scheme — which DOES
    /// saturate — is where that behaviour belongs.
    #[test]
    fn kernels_superpose_and_scale() {
        let dt = 1e-5;
        let mut single = Alpha::new(2e-3).expect("valid");
        let mut pair = Alpha::new(2e-3).expect("valid");
        single.inject(1.0);
        pair.inject(0.25);
        pair.inject(0.75);
        for _ in 0..500 {
            single.advance(dt);
            pair.advance(dt);
            assert!((single.value() - pair.value()).abs() < 1e-15);
        }

        // Scaling: doubling the weight doubles the response at every time. Compared through the
        // STATEFUL form against the ANALYTIC one, because comparing `2·response(t)` against
        // `2·response(t)` — which a first draft of this block did — asserts nothing at all.
        let analytic = Alpha::new(2e-3).expect("valid");
        let mut doubled = Alpha::new(2e-3).expect("valid");
        doubled.inject(2.0);
        for step in 1..=500u32 {
            doubled.advance(dt);
            let t = f64::from(step) * dt;
            let want = 2.0 * analytic.response(t).expect("finite");
            assert!((doubled.value() - want).abs() < 1e-14, "t = {t}: {} vs {want}", doubled.value());
        }
    }

    /// The delta kernel refuses the three analytic quantities it does not have, and says nothing
    /// pointwise at the origin.
    #[test]
    fn the_delta_kernel_refuses_the_quantities_it_does_not_have() {
        let mut d = Delta::new();
        assert_eq!(d.response(0.0), None, "delta(0) is not a number");
        assert_eq!(d.response(1e-3), Some(0.0));
        assert_eq!(d.response(-1e-3), Some(0.0));
        assert_eq!(d.response(f64::NAN), None);
        assert_eq!(d.integral_seconds(), None, "the delta's area is dimensionless, not seconds");
        assert_eq!(d.peak_time(), None);
        assert_eq!(d.peak_value(), None);

        // The stateful contract: present for one step, gone after it.
        d.inject(0.4);
        d.inject(0.6);
        assert_eq!(d.value(), 1.0);
        d.advance(1e-4);
        assert_eq!(d.value(), 0.0);
    }

    /// Causality: nothing responds before the spike.
    #[test]
    fn every_kernel_is_silent_before_the_spike() {
        let e = Exponential::new(1e-3).expect("valid");
        let a = Alpha::new(1e-3).expect("valid");
        let b = BiExponential::new(0.3e-3, 4e-3).expect("valid");
        for &t in &[-1e-9, -1e-3, -1.0] {
            assert_eq!(e.response(t), Some(0.0));
            assert_eq!(a.response(t), Some(0.0));
            assert_eq!(b.response(t), Some(0.0));
        }
    }

    /// A non-finite time or weight cannot become a non-finite response or state.
    #[test]
    fn no_kernel_leaks_a_non_finite_value() {
        let mut k = Alpha::new(1e-3).expect("valid");
        assert_eq!(k.response(f64::NAN), None);
        assert_eq!(k.response(f64::INFINITY), None);
        k.inject(f64::NAN);
        k.inject(f64::INFINITY);
        k.advance(f64::NAN);
        k.advance(-1e-3);
        assert!(k.value().is_finite(), "state became {}", k.value());
        assert_eq!(k.value(), 0.0, "an ignored injection should have changed nothing");
    }

    // ---------------------------------------------------------------- CUBA versus COBA

    /// (e) A conductance-based synapse at its reversal potential delivers exactly zero current.
    ///
    /// `assert_eq!` and not a tolerance: `e_rev - v` with `v == e_rev` is exactly `0.0`, and the
    /// multiplication is ordered in `Drive::current` so that zero propagates. A model that computed
    /// `g·e_rev - g·v` instead would leave a rounding residue here, and that residue is a current
    /// that never turns off.
    #[test]
    fn a_coba_synapse_at_its_reversal_potential_passes_exactly_zero_current() {
        for r in RECEPTORS {
            let mut s = r.conductance_based(5e-9).expect("valid receptor");
            s.inject(1.0);
            s.advance(r.tau_decay * 0.5);
            assert!(s.conductance().expect("coba has a conductance") > 0.0, "{} is open", r.name);
            assert_eq!(
                s.current(r.e_rev),
                0.0,
                "{} delivered current at its own reversal potential",
                r.name
            );
            assert_eq!(s.driving_force(r.e_rev), Some(0.0));
            assert_eq!(s.reversal_potential(), Some(r.e_rev));

            // And NEAR the reversal potential the current stays exact, which the zero above does
            // not test: `g·E - g·V` is also exactly zero AT the reversal potential, so a mutation
            // to that expanded form survived the assertion above. Here the driving force is
            // 2^-50 V — a subtraction of two nearby floats, therefore exact — and the current must
            // be exactly `-g·d`. The expanded form keeps about one significant digit of that.
            let d = f64::powi(2.0, -50);
            let near = r.e_rev + d;
            assert_eq!(
                s.current(near),
                -(s.open_conductance(near) * d),
                "{} lost precision one part in 1e14 from its reversal potential",
                r.name
            );
        }
    }

    /// The `COBA` current shrinks monotonically as the membrane approaches the reversal potential,
    /// and reverses beyond it. This is the self-limiting property, stated as an ordering rather
    /// than a single number.
    #[test]
    fn a_coba_synapse_shrinks_as_the_membrane_approaches_reversal() {
        let mut s = AMPA.conductance_based(10e-9).expect("valid");
        s.inject(1.0);
        s.advance(0.5e-3);
        let mut last = f64::INFINITY;
        for mv in [-90.0, -70.0, -50.0, -30.0, -10.0, -1.0] {
            let i = s.current(mv * 1e-3);
            assert!(i > 0.0, "AMPA should depolarise at {mv} mV");
            assert!(i < last, "current at {mv} mV was not smaller than at the previous step");
            last = i;
        }
        assert!(s.current(10e-3) < 0.0, "past the reversal potential the current must reverse");
    }

    /// A current-based synapse ignores the membrane potential entirely, and refuses the three
    /// questions it cannot answer rather than answering them with zero.
    #[test]
    fn a_cuba_synapse_ignores_voltage_and_refuses_what_it_does_not_have() {
        let mut s = AMPA.current_based(2e-9).expect("valid");
        s.inject(1.0);
        s.advance(0.5e-3);
        let reference = s.current(-65e-3);
        assert!(reference > 0.0);
        for mv in [-120.0, -65.0, -20.0, 0.0, 40.0] {
            assert_eq!(s.current(mv * 1e-3), reference, "CUBA current moved at {mv} mV");
        }
        assert_eq!(s.conductance(), None, "a CUBA synapse has no conductance, not a small one");
        assert_eq!(s.reversal_potential(), None);
        assert_eq!(s.driving_force(-65e-3), None);
        assert_eq!(s.gain(), 2e-9);
    }

    /// A passive membrane driven by synapses, used by the shunting tests below.
    ///
    /// Returns `(peak, trough, final)` volts. Threshold is put at +1 V so the cell cannot spike:
    /// this is a test about subthreshold integration, and a spike would reset the potential and
    /// replace the quantity being measured with a different one.
    fn membrane_run(g_exc: f64, g_inh: f64, e_inh: f64) -> (f64, f64, f64) {
        let mut m = Lif { v_th: 1.0, t_ref: 0.0, ..Lif::default() };
        let mut exc = AMPA.conductance_based(g_exc).expect("valid");
        let mut inh =
            Receptor { e_rev: e_inh, ..GABA_A }.conductance_based(g_inh).expect("valid");
        exc.inject(1.0);
        inh.inject(1.0);
        let dt = 1e-6;
        let mut peak = m.potential();
        let mut trough = m.potential();
        for _ in 0..100_000 {
            let v = m.potential();
            m.step(dt, exc.current(v) + inh.current(v));
            exc.advance(dt);
            inh.advance(dt);
            peak = peak.max(m.potential());
            trough = trough.min(m.potential());
        }
        (peak, trough, m.potential())
    }

    /// (f) Shunting inhibition: a `COBA` inhibitory synapse whose reversal potential sits exactly
    /// at rest suppresses an excitatory response **without moving the resting potential at all**.
    ///
    /// Three runs. Inhibition alone must leave the membrane at rest to the last bit, because the
    /// driving force is exactly zero there and the leak has nothing to do either. Excitation alone
    /// gives a reference `EPSP`. Both together must give a substantially smaller `EPSP` — the
    /// inhibitory conductance has lowered the input resistance, which is a different mechanism from
    /// adding a negative current and is the one a `CUBA` model cannot express.
    #[test]
    fn shunting_inhibition_suppresses_without_moving_rest() {
        let rest = Lif::default().v_rest;

        // 5 µS of inhibitory conductance against a 100 nS leak. That is fifty unitary synapses'
        // worth, not one, and it is stated rather than hidden: shunting is discussed in the
        // literature as a HIGH-CONDUCTANCE state, and a single synapse at a tenth of the leak
        // shunts an EPSP by about 20%, which is real but too small to separate from a tolerance.
        let shunt = 5e-6;

        // Inhibition alone, reversal exactly at rest: nothing moves, bit-for-bit.
        let (peak, trough, last) = membrane_run(0.0, shunt, rest);
        assert_eq!(peak, rest, "a shunting synapse depolarised a resting cell");
        assert_eq!(trough, rest, "a shunting synapse hyperpolarised a resting cell");
        assert_eq!(last, rest);

        // Excitation alone.
        let (alone, _, _) = membrane_run(20e-9, 0.0, rest);
        let epsp_alone = alone - rest;
        assert!(epsp_alone > 1e-3, "the reference EPSP was only {epsp_alone} V");

        // Excitation through the shunt.
        let (shunted, shunt_trough, _) = membrane_run(20e-9, shunt, rest);
        let epsp_shunted = shunted - rest;
        assert!(epsp_shunted > 0.0, "the shunted EPSP vanished entirely");
        assert!(
            epsp_shunted < 0.5 * epsp_alone,
            "shunt reduced the EPSP only from {epsp_alone} V to {epsp_shunted} V"
        );
        assert_eq!(shunt_trough, rest, "the shunt pulled the membrane below rest");

        // The contrast: move the same synapse's reversal 10 mV below rest and it hyperpolarises,
        // which is what "inhibition" is usually taken to mean and is a DIFFERENT mechanism.
        let (_, hyper_trough, _) = membrane_run(0.0, shunt, rest - 10e-3);
        assert!(
            hyper_trough < rest - 1e-3,
            "a hyperpolarising synapse left the membrane at {hyper_trough} V"
        );
    }

    /// The shunt, checked against a closed form rather than against a ratio.
    ///
    /// A large synaptic conductance drops the membrane's effective time constant `C/g_total` far
    /// below the kinetics driving it, and the potential then tracks the **instantaneous
    /// conductance-weighted mean of the reversal potentials**:
    ///
    /// ```text
    /// V(t) -> (g_leak·V_rest + g_e·E_e + g_i·E_i) / (g_leak + g_e + g_i)
    /// ```
    ///
    /// That is the steady state of the membrane equation with the conductances frozen, and it is
    /// where the word "divisive" comes from: the excitatory term appears over a denominator that
    /// the inhibition has enlarged. With 50 µS of shunt against a 100 nS leak the effective time
    /// constant is 40 µs against kinetics of hundreds of microseconds, so the simulated potential
    /// has to sit on that curve — and the residual is the lag `τ_eff · dV/dt`, which is why the
    /// tolerance is a tenth of the swing and not a thousandth.
    #[test]
    fn a_strong_shunt_puts_the_membrane_in_the_quasi_static_regime() {
        let proto = Lif { v_th: 1.0, t_ref: 0.0, ..Lif::default() };
        let g_leak = 1.0 / proto.r_m;
        let rest = proto.v_rest;
        let shunt = 50e-6;

        let mut m = proto;
        let mut exc = AMPA.conductance_based(20e-9).expect("valid");
        let mut inh = Receptor { e_rev: rest, ..GABA_A }.conductance_based(shunt).expect("valid");
        exc.inject(1.0);
        inh.inject(1.0);

        // The first 0.5 ms is NOT in the quasi-static regime and is skipped, with the reason
        // stated rather than tuned. At `t = 0` the shunt is closed and the effective time constant
        // is the membrane's own 20 ms, while the PREDICTION reaches most of its peak within a
        // microsecond — it depends on the RATIO of the two synaptic conductances, and both rise
        // with the same 0.5 ms constant, so the ratio is near its limit before either conductance
        // is. The membrane therefore starts behind and needs several of its own (shrinking)
        // effective time constants to catch up.
        //
        // Measured: from 0.2 ms the tracking error is 33% of the swing, from 0.5 ms it is 0.98%,
        // and it climbs back through 2% by 5 ms as the shunt decays and τ_eff grows again. The
        // window is where the condition holds, not where the numbers are prettiest — the 5 ms
        // figure is stated here so the boundary is visible rather than hidden by the choice.
        let dt = 1e-6;
        let settle = 500u32;
        let mut worst: f64 = 0.0;
        let mut swing: f64 = 0.0;
        for step in 0..20_000u32 {
            let v = m.potential();
            let ge = exc.conductance().expect("a COBA synapse has one");
            let gi = inh.conductance().expect("a COBA synapse has one");
            let v_qs = (g_leak * rest + ge * AMPA.e_rev + gi * rest) / (g_leak + ge + gi);
            if step >= settle {
                worst = worst.max((v - v_qs).abs());
                swing = swing.max((v_qs - rest).abs());
            }
            m.step(dt, exc.current(v) + inh.current(v));
            exc.advance(dt);
            inh.advance(dt);
        }
        assert!(swing > 1e-5, "the quasi-static prediction barely moved: {swing} V");
        assert!(
            worst < 0.03 * swing,
            "membrane tracked its quasi-static prediction only to {worst} V \
             against a swing of {swing} V"
        );
    }

    /// The same comparison in `CUBA`, to show what is lost: a current-based inhibitory synapse
    /// always moves the resting potential, because it has no reversal potential to sit at.
    #[test]
    fn a_cuba_inhibitory_synapse_cannot_shunt() {
        let rest = Lif::default().v_rest;
        let mut m = Lif { v_th: 1.0, t_ref: 0.0, ..Lif::default() };
        let mut inh = CurrentBased::new(
            BiExponential::new(GABA_A.tau_rise, GABA_A.tau_decay).expect("valid"),
            -0.5e-9,
        )
        .expect("finite gain");
        inh.inject(1.0);
        let dt = 1e-6;
        let mut trough = rest;
        for _ in 0..100_000 {
            let v = m.potential();
            m.step(dt, inh.current(v));
            inh.advance(dt);
            trough = trough.min(m.potential());
        }
        assert!(
            trough < rest - 1e-4,
            "a CUBA inhibitory synapse left the resting potential at {trough} V; \
             it is supposed to be unable to leave it alone"
        );
    }

    // ---------------------------------------------------------------- receptors

    /// The magnesium block at its closed-form half-block potential is exactly one half.
    ///
    /// `V_mV = ln([Mg]/K)/α` is where `exp(-α·V_mV)·[Mg]/K = 1`, so `B = 1/(1 + 1)`. Checked at
    /// several concentrations, because the half-block potential moves with magnesium and a
    /// hard-coded −20.5 mV would pass at 1 mM and nowhere else.
    #[test]
    fn the_magnesium_block_is_half_open_at_its_closed_form_potential() {
        for &mg in &[0.1, 0.5, 1.0, 2.0, 5.0] {
            let b = MgBlock::with_magnesium(mg).expect("non-negative");
            let v_half = b.half_block_potential().expect("a block has a half-block potential");
            let f = b.open_fraction(v_half);
            assert!((f - 0.5).abs() < 1e-12, "[Mg] = {mg} mM: open fraction {f} at {v_half} V");
        }
        // The published 1 mM figure, for a reader comparing against the paper.
        let b = MgBlock::default();
        let v_half = b.half_block_potential().expect("has one");
        assert!(
            (v_half - (-20.526e-3)).abs() < 5e-6,
            "1 mM half-block potential is {v_half} V, expected about -20.5 mV"
        );
        // Magnesium-free: no block, and therefore no potential at which half of it is relieved.
        let free = MgBlock::with_magnesium(0.0).expect("zero is legal");
        assert_eq!(free.half_block_potential(), None);
        assert_eq!(free.open_fraction(-70e-3), 1.0);
    }

    /// The block is bounded, monotonic in voltage, and lands on the figures quoted in `MgBlock`'s
    /// doc — 4% open at −70 mV, 78% at 0 mV. A doc figure nobody checks is a doc figure that drifts.
    #[test]
    fn the_magnesium_block_is_monotonic_and_matches_its_quoted_figures() {
        let b = MgBlock::default();
        let mut last = -1.0;
        for mv in [-120.0, -100.0, -80.0, -60.0, -40.0, -20.0, 0.0, 20.0, 50.0] {
            let f = b.open_fraction(mv * 1e-3);
            assert!((0.0..=1.0).contains(&f), "open fraction {f} at {mv} mV");
            assert!(f > last, "not monotonic at {mv} mV");
            last = f;
        }
        let at_rest = b.open_fraction(-70e-3);
        assert!((at_rest - 0.044).abs() < 0.003, "at -70 mV the block leaves {at_rest} open");
        let at_zero = b.open_fraction(0.0);
        assert!((at_zero - 0.781).abs() < 0.005, "at 0 mV the block leaves {at_zero} open");
        assert_eq!(b.open_fraction(f64::NAN), 0.0, "a NaN potential must not leak");
    }

    /// The `NMDA` current is non-monotonic in voltage, with an interior maximum.
    ///
    /// Zero at very negative potentials because the pore is blocked, exactly zero at the reversal
    /// potential because the driving force vanishes, and largest in between. That interior maximum
    /// is the negative-slope region that lets `NMDA` support bistability, and it is the single
    /// clearest demonstration that a receptor is more than a time constant.
    #[test]
    fn the_nmda_current_is_non_monotonic_in_voltage() {
        let mut s = NMDA.conductance_based(5e-9).expect("valid");
        s.inject(1.0);
        s.advance(10e-3);
        let n = 2_000u32;
        let mut best = (f64::NEG_INFINITY, 0.0);
        for i in 0..=n {
            let v = -0.12 + 0.12 * f64::from(i) / f64::from(n);
            let cur = s.current(v);
            if cur > best.0 {
                best = (cur, v);
            }
        }
        assert!(best.0 > 0.0, "NMDA delivered no depolarising current anywhere");
        assert!(
            best.1 > -0.09 && best.1 < -0.005,
            "the NMDA current peaked at {} V, expected an interior maximum",
            best.1
        );
        assert!(s.current(-0.12) < 0.1 * best.0, "the block should silence NMDA at -120 mV");
        assert_eq!(s.current(0.0), 0.0, "the driving force vanishes at the reversal potential");

        // Removing the magnesium removes the mechanism. An identically driven synapse without the
        // block has the same OPEN conductance and a far larger AVAILABLE one at rest — the ratio
        // is the block's open fraction, about 4%, and asserting it here keeps `open_conductance`
        // honest about applying the block where `conductance` does not.
        let mut unblocked =
            ConductanceBased::new(NMDA.kernel().expect("valid"), 5e-9, NMDA.e_rev)
                .expect("valid");
        unblocked.inject(1.0);
        unblocked.advance(10e-3);
        assert_eq!(s.conductance(), unblocked.conductance(), "the kernels must be identical");
        let ratio = s.open_conductance(-70e-3) / unblocked.open_conductance(-70e-3);
        assert!(
            (ratio - 0.044).abs() < 0.003,
            "the block left {ratio} of the conductance available at rest"
        );
    }

    /// The receptor table's own claims: every entry is constructible, ordered fast to slow, and
    /// signed the way its reversal potential says relative to a typical resting potential.
    #[test]
    fn the_receptor_table_is_internally_consistent() {
        let rest = -65e-3;
        for r in RECEPTORS {
            assert!(r.tau_rise > 0.0 && r.tau_rise <= r.tau_decay, "{} kinetics", r.name);
            assert!(!r.source.is_empty(), "{} has no provenance", r.name);
            let k = r.kernel().expect("table entries must build");
            assert!(k.peak_time().expect("has a peak") > 0.0);
        }
        assert!(AMPA.e_rev > rest, "AMPA must depolarise from rest");
        assert!(NMDA.e_rev > rest, "NMDA must depolarise from rest");
        assert!(GABA_A.e_rev < rest, "GABA_A as shipped sits just below rest");
        const { assert!(GABA_B.e_rev < GABA_A.e_rev, "GABA_B is the more hyperpolarising") };
        const { assert!(NMDA.tau_decay > 40.0 * AMPA.tau_decay, "NMDA is the slow excitatory one") };
        assert!(NMDA.mg_block.is_some(), "NMDA carries the block");
        assert!(AMPA.mg_block.is_none() && GABA_A.mg_block.is_none());
        // GABA_A as shipped is nearly shunting at a typical rest: within 5 mV of it.
        assert!((GABA_A.e_rev - rest).abs() < 6e-3);
    }

    // ---------------------------------------------------------------- kinetic schemes

    /// The two-state scheme against its analytic solution, during the pulse and after it.
    ///
    /// During a pulse of `[T]`, `r` relaxes toward `α[T]/(α[T] + β)` with rate `α[T] + β`; once the
    /// cleft clears it decays as `exp(-β·t)` from whatever it reached. Both branches are checked
    /// against the closed form at every step, which is the check that catches a step that
    /// straddles the end of the pulse and uses one rate for the whole of it.
    #[test]
    fn the_two_state_kinetic_scheme_matches_its_analytic_solution() {
        for mut k in [KineticTwoState::ampa(), KineticTwoState::gaba_a()] {
            let on = k.alpha * k.t_max_mm;
            let rate = on + k.beta;
            let r_inf = k.steady_open_fraction(k.t_max_mm).expect("rates are positive");
            assert!((r_inf - on / rate).abs() < 1e-15);

            k.reset();
            k.release();
            // Deliberately a step size that does NOT divide the pulse duration, so the split at
            // the pulse boundary is exercised rather than landed on by luck.
            let dt = 7e-6;
            let mut t = 0.0;
            let r_at_pulse_end = r_inf * (1.0 - (-rate * k.t_pulse).exp());
            for _ in 0..3_000u32 {
                k.advance(dt);
                t += dt;
                let want = if t <= k.t_pulse {
                    r_inf * (1.0 - (-rate * t).exp())
                } else {
                    r_at_pulse_end * (-k.beta * (t - k.t_pulse)).exp()
                };
                assert!(
                    (k.r - want).abs() < 1e-12,
                    "t = {t}: simulated {} vs closed form {want}",
                    k.r
                );
                assert!((0.0..=1.0).contains(&k.r), "open fraction left [0, 1]: {}", k.r);
            }
        }
    }

    /// The figures quoted in `KineticTwoState`'s doc: `AMPA` reaches 0.618 open after a 1 ms pulse
    /// and decays with a 5.26 ms time constant.
    #[test]
    fn the_ampa_kinetic_peak_and_decay_match_the_quoted_figures() {
        let k = KineticTwoState::ampa();
        let rate = k.alpha * k.t_max_mm + k.beta;
        let r_inf = k.steady_open_fraction(k.t_max_mm).expect("positive");
        let peak = r_inf * (1.0 - (-rate * k.t_pulse).exp());
        assert!((peak - 0.618).abs() < 0.002, "AMPA peak open fraction is {peak}");
        let tau = k.decay_tau().expect("beta is positive");
        assert!((tau - 5.263e-3).abs() < 1e-5, "AMPA decay tau is {tau} s");
        assert_eq!(KineticTwoState::new(0.0, 0.0, 1.0, 1e-3).expect("legal").decay_tau(), None);
    }

    /// The kinetic scheme **saturates**, which is the property the linear kernels do not have.
    ///
    /// Two spikes 0.2 ms apart open far less than twice what one opens, because the second arrives
    /// while receptors are still bound. An [`Alpha`] kernel at the same separation superposes
    /// almost exactly, and the difference between those two numbers is the whole reason to pay for
    /// a kinetic model.
    #[test]
    fn the_kinetic_scheme_saturates_where_a_linear_kernel_superposes() {
        let dt = 1e-6;
        let gap = 200u32; // 0.2 ms

        let peak_of = |spikes: u32| {
            let mut k = KineticTwoState::ampa();
            let mut peak: f64 = 0.0;
            for step in 0..4_000u32 {
                if step < spikes * gap && step % gap == 0 {
                    k.release();
                }
                k.advance(dt);
                peak = peak.max(k.r);
            }
            peak
        };
        let one = peak_of(1);
        let two = peak_of(2);
        assert!(two > one, "a second spike must still add something");
        assert!(two < 1.5 * one, "kinetic scheme superposed linearly: {one} then {two}");
        assert!(two <= 1.0, "open fraction exceeded one: {two}");

        // The linear kernel, at the same separation, adds nearly the full second response.
        let mut a = Alpha::new(1e-3).expect("valid");
        let mut a_peak: f64 = 0.0;
        let mut a_one: f64 = 0.0;
        {
            let mut single = Alpha::new(1e-3).expect("valid");
            single.inject(1.0);
            for _ in 0..4_000u32 {
                single.advance(dt);
                a_one = a_one.max(single.value());
            }
        }
        for step in 0..4_000u32 {
            if step < 2 * gap && step % gap == 0 {
                a.inject(1.0);
            }
            a.advance(dt);
            a_peak = a_peak.max(a.value());
        }
        assert!(a_peak > 1.9 * a_one, "the linear kernel should nearly double: {a_one}, {a_peak}");
    }

    /// The `GABA_B` cascade's steady state under sustained transmitter, against its closed form.
    ///
    /// ⚠ This validates the integrator, including the removable singularity branch. It says
    /// nothing about whether the rate constants describe a real synapse — see the warning on
    /// `GabaBCascade`.
    #[test]
    fn the_gaba_b_cascade_reaches_its_closed_form_steady_state() {
        let mut c = GabaBCascade::default();
        let (r_want, g_want, open_want) = c.steady_state(c.t_max_mm).expect("k4 is positive");
        // Hold the transmitter on for 5 s by refreshing the pulse every step.
        let dt = 1e-4;
        for _ in 0..50_000u32 {
            c.release();
            c.advance(dt);
        }
        assert!((c.r - r_want).abs() < 1e-9, "r {} vs closed form {r_want}", c.r);
        assert!(
            (c.g_conc - g_want).abs() / g_want < 1e-9,
            "G {} vs closed form {g_want}",
            c.g_conc
        );
        assert!((c.open_fraction() - open_want).abs() < 1e-9);
        assert!((0.0..1.0).contains(&c.open_fraction()));
    }

    /// The Hill term is the mechanism: a burst opens disproportionately more than a single spike.
    ///
    /// With `n = 4`, doubling `G` multiplies `G^n` by sixteen, so the cascade is nearly silent for
    /// one spike and substantial for a train. A bi-exponential fit cannot reproduce this, which is
    /// the caveat written on the [`GABA_B`] receptor row.
    #[test]
    fn the_gaba_b_cascade_responds_supralinearly_to_a_burst() {
        let dt = 1e-4;
        let peak_of = |spikes: u32| {
            let mut c = GabaBCascade::default();
            let mut peak: f64 = 0.0;
            for step in 0..20_000u32 {
                if step < spikes * 100 && step % 100 == 0 {
                    c.release();
                }
                c.advance(dt);
                peak = peak.max(c.open_fraction());
            }
            peak
        };
        let one = peak_of(1);
        let ten = peak_of(10);
        assert!(one > 0.0, "a single spike opened nothing at all");
        assert!(
            ten > 10.0 * one,
            "ten spikes opened {ten} against one spike's {one}; that is not supralinear"
        );
    }

    /// The removable singularity in the cascade's `G` update: at `K4 = K1[T] + K2` the difference
    /// quotient is `dt·e^{-K4 dt}`, and the branch must be continuous across it.
    #[test]
    fn the_cascade_handles_its_removable_singularity() {
        // Tune K1 — not K2 — so that `K1·[T] + K2` lands exactly on K4 during a pulse. Solving
        // through K2 instead gives `K2 = K4 - K1·[T] = -56 s⁻¹`, a negative unbinding rate, and
        // the transmitter-free phase then GROWS instead of relaxing. The first draft did that and
        // measured a discontinuity that was really an unstable model.
        let base = GabaBCascade::default();
        let exact = GabaBCascade { k1: (base.k4 - base.k2) / base.t_max_mm, ..base };
        assert!(exact.k1 > 0.0 && exact.k2 > 0.0, "every rate must stay positive");
        assert!((exact.k1 * exact.t_max_mm + exact.k2 - exact.k4).abs() < 1e-12);

        let run = |mut c: GabaBCascade| {
            for _ in 0..200u32 {
                c.release();
                c.advance(1e-4);
            }
            (c.r, c.g_conc)
        };
        let (r0, g0) = run(exact);
        assert!(r0.is_finite() && g0.is_finite(), "the singular case produced {r0}, {g0}");
        assert!(g0 > 0.0, "the singular case produced no G at all");

        // Continuity, stated as a second-order claim rather than as a tolerance picked to pass: a
        // smooth function evaluated at `k1 ± ε` averages to its value at `k1` up to `O(ε²)`, so
        // the centring error must be small COMPARED WITH the first-order change and must fall by
        // about four when ε is halved. A jump of any size would show up as a centring error of
        // half the jump, independent of ε, and would fail both halves. Without the guard the
        // singular point is `0/0` and every comparison here is against a `NaN`.
        let centring = |frac: f64| {
            let eps = frac * exact.k1;
            let (_, lo) = run(GabaBCascade { k1: exact.k1 - eps, ..exact });
            let (_, hi) = run(GabaBCascade { k1: exact.k1 + eps, ..exact });
            (0.5 * (lo + hi) - g0, hi - lo)
        };
        let (off_coarse, slope_coarse) = centring(1e-6);
        assert!(
            off_coarse.abs() < 0.05 * slope_coarse.abs(),
            "centring error {off_coarse} against a first-order change of {slope_coarse}"
        );
        let (off_fine, _) = centring(0.5e-6);
        assert!(
            off_fine.abs() < 0.4 * off_coarse.abs(),
            "the centring error did not shrink quadratically: {off_coarse} then {off_fine},              which is what a discontinuity at the singular point would look like"
        );
    }

    // ---------------------------------------------------------------- short-term plasticity

    /// (d) The `Tsodyks`-`Markram` steady state against its closed form, at six rates, for both
    /// regimes.
    ///
    /// The recursion is affine in both variables, so its fixed point is exact and the simulation
    /// has to reach it rather than approach it: the tolerance is `1e-9` relative, which a model
    /// with the facilitation step in the wrong order would miss by tens of percent.
    #[test]
    fn the_tsodyks_markram_steady_state_matches_the_closed_form() {
        for proto in [TsodyksMarkram::depressing(), TsodyksMarkram::facilitating()] {
            for &rate in &[1.0, 5.0, 10.0, 20.0, 50.0, 100.0] {
                let want = proto.steady_state(rate).expect("positive rate");
                let interval = 1.0 / rate;
                let mut s = proto;
                let mut released = s.spike();
                for _ in 0..1_500u32 {
                    released = s.spike_after(interval);
                }
                let rel = (released - want.release).abs() / want.release;
                assert!(
                    rel < 1e-9,
                    "U = {}, tau_f = {}, {rate} Hz: simulated release {released} vs closed form {}",
                    proto.u_rest,
                    proto.tau_f,
                    want.release
                );
                assert!((s.u - want.u).abs() < 1e-9, "u {} vs {}", s.u, want.u);
                assert!((0.0..=1.0).contains(&s.x) && (0.0..=1.0).contains(&s.u));
            }
        }
    }

    /// The first spike from rest releases exactly `U`. That is what makes `U` the release
    /// probability of a rested synapse rather than a fitted constant with no interpretation, and
    /// reversing the facilitation step and the release would make it zero.
    #[test]
    fn the_first_spike_from_rest_releases_exactly_u() {
        for mut s in [TsodyksMarkram::depressing(), TsodyksMarkram::facilitating()] {
            assert_eq!(s.spike(), s.u_rest);
        }
    }

    /// Both regimes are reachable, and the boundary between them is a **rate**, not a parameter.
    ///
    /// The facilitating parameter set facilitates at 20 Hz — the steady-state response is 1.8 times
    /// the first — and depresses at 100 Hz, because `u` saturates at 1 while `x` keeps falling.
    #[test]
    fn depression_and_facilitation_are_both_reachable_and_the_rate_decides() {
        let dep = TsodyksMarkram::depressing();
        let fac = TsodyksMarkram::facilitating();

        let ratio = |s: TsodyksMarkram, rate: f64| {
            let first = s.u_rest;
            s.steady_state(rate).expect("positive rate").release / first
        };

        assert!(ratio(dep, 20.0) < 0.3, "the depressing set did not depress at 20 Hz");
        assert!(ratio(dep, 100.0) < 0.1, "the depressing set did not depress at 100 Hz");

        let f20 = ratio(fac, 20.0);
        assert!(f20 > 1.5, "the facilitating set gave a ratio of only {f20} at 20 Hz");
        let f100 = ratio(fac, 100.0);
        assert!(f100 < 1.0, "the facilitating set did not depress at 100 Hz: ratio {f100}");
        assert!(f100 < f20, "the ratio should fall with rate: {f20} then {f100}");
    }

    /// The limiting transmission rate: `A*·r → 1/τ_d` as the presynaptic rate goes to infinity,
    /// independently of `U` and `τ_f`.
    ///
    /// Two parameter sets that differ in every constant converge on the same ceiling, which is the
    /// claim. The approach is `O(1/r)`, so the tolerance tightens with rate and a model whose
    /// ceiling depended on `U` would miss it by a factor.
    #[test]
    fn the_limiting_transmission_rate_is_one_over_tau_d() {
        for proto in [TsodyksMarkram::depressing(), TsodyksMarkram::facilitating()] {
            let ceiling = proto.limiting_transmission_rate();
            assert_eq!(ceiling, 1.0 / proto.tau_d);
            let mut last = 0.0;
            for &rate in &[1e3, 1e4, 1e5] {
                let s = proto.steady_state(rate).expect("positive rate");
                let rel = (s.transmitted_per_second - ceiling).abs() / ceiling;
                assert!(
                    rel < 20.0 / rate.sqrt(),
                    "{rate} Hz: transmitted {} vs ceiling {ceiling}",
                    s.transmitted_per_second
                );
                assert!(s.transmitted_per_second > last, "convergence should be from below");
                last = s.transmitted_per_second;
            }
            let tightest = proto.steady_state(1e6).expect("positive");
            assert!(
                (tightest.transmitted_per_second - ceiling).abs() / ceiling < 1e-4,
                "at 1 MHz the transmitted rate is {} against a ceiling of {ceiling}",
                tightest.transmitted_per_second
            );
        }
    }

    /// Relaxation is exact, so the state after one long wait equals the state after many short
    /// ones. A caller that subdivides its time step must not get a different synapse.
    #[test]
    fn short_term_plasticity_relaxation_is_step_size_independent() {
        let run = |steps: u32| {
            let mut s = TsodyksMarkram::facilitating();
            s.spike();
            let dt = 0.1 / f64::from(steps);
            for _ in 0..steps {
                s.advance(dt);
            }
            (s.u, s.x)
        };
        let fine = run(100_000);
        let coarse = run(1);
        // 1e-11: 100,000 exponentials accumulate about `k · eps` = 1e-11 of round-off, and the
        // measured gap is 3.3e-13. A model that relaxed by forward Euler instead would differ
        // between these two runs in the third decimal place.
        assert!((fine.0 - coarse.0).abs() < 1e-11, "u: {} vs {}", fine.0, coarse.0);
        assert!((fine.1 - coarse.1).abs() < 1e-11, "x: {} vs {}", fine.1, coarse.1);
    }

    /// Resources recover toward 1 and utilisation decays toward 0, both with their own time
    /// constants — checked against the exponentials themselves rather than against a later run.
    #[test]
    fn the_plasticity_variables_relax_along_their_own_exponentials() {
        let mut s = TsodyksMarkram::facilitating();
        s.spike();
        let (u0, x0) = (s.u, s.x);
        let dt = 1e-4;
        for k in 1..=2_000u32 {
            s.advance(dt);
            let t = f64::from(k) * dt;
            let want_u = u0 * (-t / s.tau_f).exp();
            let want_x = 1.0 + (x0 - 1.0) * (-t / s.tau_d).exp();
            assert!((s.u - want_u).abs() < 1e-14, "u at {t}: {} vs {want_u}", s.u);
            assert!((s.x - want_x).abs() < 1e-14, "x at {t}: {} vs {want_x}", s.x);
        }
    }

    /// With `τ_f = 0` there is no facilitation at all: every spike releases `U` of what remains,
    /// which is the pure-depression limit and is handled as a branch rather than as `exp(-dt/0)`.
    #[test]
    fn zero_facilitation_is_a_branch_and_not_an_infinity() {
        let mut s = TsodyksMarkram::depressing();
        assert_eq!(s.tau_f, 0.0);
        s.advance(0.0); // must not produce NaN, and must not zero anything either
        assert_eq!(s.x, 1.0);
        for _ in 0..20 {
            let released = s.spike_after(0.05);
            assert!(released.is_finite() && released > 0.0);
            assert_eq!(s.u, s.u_rest, "u must return to U between spikes with no facilitation");
        }
    }

    // ---------------------------------------------------------------- refusals

    /// Every constructor refuses what it cannot represent, and names it.
    #[test]
    fn bad_parameters_are_refused_by_name() {
        assert!(matches!(
            Exponential::new(0.0),
            Err(SynapseError::NonPositiveTimeConstant { name: "tau", .. })
        ));
        assert!(matches!(
            Exponential::new(-1e-3),
            Err(SynapseError::NonPositiveTimeConstant { .. })
        ));
        assert!(matches!(
            Alpha::new(f64::NAN),
            Err(SynapseError::NonFinite { name: "tau", .. })
        ));
        assert!(matches!(
            BiExponential::new(5e-3, 1e-3),
            Err(SynapseError::RiseSlowerThanDecay { .. })
        ));
        assert!(matches!(
            BiExponential::new(1e-3, f64::INFINITY),
            Err(SynapseError::NonFinite { name: "tau_decay", .. })
        ));
        assert!(matches!(
            ConductanceBased::new(Delta::new(), -1e-9, 0.0),
            Err(SynapseError::Negative { name: "g_peak", .. })
        ));
        assert!(matches!(
            CurrentBased::new(Delta::new(), f64::NAN),
            Err(SynapseError::NonFinite { name: "gain", .. })
        ));
        assert!(matches!(
            TsodyksMarkram::new(0.0, 0.8, 0.0),
            Err(SynapseError::FractionOutOfRange { name: "u_rest", .. })
        ));
        assert!(matches!(
            TsodyksMarkram::new(1.5, 0.8, 0.0),
            Err(SynapseError::FractionOutOfRange { .. })
        ));
        assert!(matches!(
            TsodyksMarkram::new(0.5, 0.0, 0.0),
            Err(SynapseError::NonPositiveTimeConstant { name: "tau_d", .. })
        ));
        assert!(matches!(
            TsodyksMarkram::new(0.5, 0.8, -1.0),
            Err(SynapseError::Negative { name: "tau_f", .. })
        ));
        assert!(matches!(
            MgBlock::with_magnesium(-1.0),
            Err(SynapseError::Negative { name: "mg_mm", .. })
        ));

        // And the legal edges really are legal.
        assert!(BiExponential::new(3e-3, 3e-3).is_ok(), "equal taus are the alpha limit");
        assert!(TsodyksMarkram::new(1.0, 0.8, 0.0).is_ok(), "U = 1 releases everything");
        assert!(MgBlock::with_magnesium(0.0).is_ok(), "magnesium-free is a real condition");
        assert!(ConductanceBased::new(Delta::new(), 0.0, 0.0).is_ok(), "a silent synapse is legal");
    }

    /// A rate of zero has no steady state, and saying so is the point.
    #[test]
    fn a_synapse_that_is_never_used_has_no_steady_state() {
        let s = TsodyksMarkram::depressing();
        assert!(matches!(s.steady_state(0.0), Err(SynapseError::NonPositiveRate { .. })));
        assert!(matches!(s.steady_state(-5.0), Err(SynapseError::NonPositiveRate { .. })));
        assert!(matches!(s.steady_state(f64::NAN), Err(SynapseError::NonFinite { .. })));
    }

    /// The time step check names what was wrong, which the per-call silence cannot.
    #[test]
    fn the_time_step_check_names_what_was_wrong() {
        assert!(check_dt(1e-4).is_ok());
        assert!(matches!(check_dt(0.0), Err(SynapseError::NonPositiveTimeConstant { .. })));
        assert!(matches!(check_dt(-1e-4), Err(SynapseError::NonPositiveTimeConstant { .. })));
        assert!(matches!(check_dt(f64::NAN), Err(SynapseError::NonFinite { name: "dt", .. })));
        // Every message names the offending quantity.
        let msg = check_dt(-1e-4).unwrap_err().to_string();
        assert!(msg.contains("dt"), "message was {msg}");
    }

    /// Determinism: the same construction run twice gives bit-identical state. No clock, no
    /// entropy, no accumulation order that depends on anything but the arguments.
    #[test]
    fn two_identical_runs_agree_bit_for_bit() {
        let run = || {
            let mut s = AMPA.conductance_based(3e-9).expect("valid");
            let mut tm = TsodyksMarkram::facilitating();
            let mut out = Vec::new();
            for step in 0..5_000u32 {
                if step % 250 == 0 {
                    s.inject(tm.spike());
                }
                tm.advance(1e-5);
                s.advance(1e-5);
                out.push(s.current(-65e-3));
            }
            out
        };
        assert_eq!(run(), run());
    }
}
