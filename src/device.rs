//! Analog device non-idealities: what a weight becomes when it is a physical conductance.
//!
//! # The lesson
//!
//! Every module before this one treats a synaptic weight as a number. This one treats it as a
//! **device**, and the difference is where analog neuromorphic accuracy goes.
//!
//! The mechanism is a **resistive crossbar**. Lay word lines horizontally and bit lines vertically,
//! put a programmable resistor at every intersection, drive the word lines with voltages `v_i` and
//! hold the bit lines at virtual ground. Ohm's law makes each cell contribute a current
//! `g_ij * v_i`, and Kirchhoff's current law sums the column for free, in the wire. The result is a
//! matrix-vector product computed in one analog step, with the weights never leaving the array.
//!
//! **What it buys** is exactly the term [`crate::ledger::Prices::e_syn_fetch`] exists to charge
//! for. A digital accelerator moves every weight from memory to an arithmetic unit and back; a
//! crossbar does not move them at all, so the product costs the read energy of the array and
//! nothing else. That is the whole argument for in-memory computing and it is a good one.
//!
//! **What it costs** is that `g_ij` is a physical thing. It was programmed to the wrong value, it
//! is not the same value it was yesterday, it is not the same value as its neighbour, one cell in
//! ten is not programmable at all, and the voltage that reaches it depends on **where in the array
//! it sits**, because the wires that sum the currents also carry them and wire has resistance.
//! None of that appears in the idealised model everyone simulates. This module makes each of those
//! gaps a named, seeded, measurable object so a loss of accuracy can be **attributed** rather than
//! observed.
//!
//! # The identity case is the load-bearing test
//!
//! [`DeviceModel::ideal`] has every non-ideality disabled and reproduces the weight matrix
//! **exactly** — `assert_eq!` on `f64`, not a tolerance. That is what makes every other result in
//! this module attributable: if the ideal path had a half-least-significant-bit wobble of its own,
//! a reader could never tell a drift result from a plumbing bug. `ideal` pays for that exactness
//! with a fiction it states openly — a zero off-conductance, which is an infinite on/off ratio, and
//! no fabricated device has one. [`Window::on_off_ratio`] refuses rather than reporting infinity.
//!
//! # The catalogue, and what each one actually does
//!
//! | Mechanism | What it does to the weight | Where it is |
//! |---|---|---|
//! | Conductance window | Bounds the representable range; the on/off ratio sets how much of the read current is signal | [`Window`], [`Mapping`] |
//! | Device-to-device spread | A fixed per-cell offset that survives reprogramming | [`Variability::sigma_d2d_rel`] |
//! | Cycle-to-cycle spread | A fresh error every time the cell is written | [`Variability::sigma_c2c_rel`] |
//! | Conductance drift | A slow power-law decay after programming | [`Drift`] |
//! | Retention | Thermally activated relaxation back toward the off state | [`Retention`] |
//! | Stuck-at faults | A fraction of cells pinned to a rail, ignoring what you wrote | [`StuckAt`] |
//! | Read noise | A fresh error on every read, with a thermal floor | [`ReadNoise`] |
//! | Finite states | The cell holds one of `n` levels, not a real number | [`Levels`] |
//! | Wire resistance | The answer depends on **where** the cell is | [`Crossbar`] |
//!
//! Eight of those nine are per-cell, which is why [`DeviceModel::apply`] takes a flat slice of
//! weights and never asks for the matrix's shape. The ninth is not, and that is the point of it:
//! **IR drop is the non-ideality most often omitted precisely because it does not factor per cell.**
//! It needs a circuit solve, so it lives in [`Crossbar`] and not in `apply`, and a caller who only
//! uses `apply` should know they have modelled everything except position.
//!
//! # Differential pairs, and the one thing they fix for free
//!
//! A conductance cannot be negative, so a signed weight needs two devices read as a difference.
//! That costs twice the area and it buys something specific: under a drift law that multiplies both
//! devices by the same factor, the difference is multiplied by that same factor too, so **relative
//! weights are preserved exactly** and only the overall gain moves — a single scalar a calibration
//! step can recover. A single-ended cell read against a *nominal* reference has no such luck: the
//! same drift leaves it with an additive offset of `g_off * (f - 1) / beta`, which no scalar gain
//! recovers. Both closed forms are asserted in this module, side by side, because the difference
//! between them is the reason differential pairs are the standard choice.
//!
//! That cancellation is also **not the whole story**, and the module says so: the drift exponent of
//! a real phase-change cell depends on its state, so the two devices of a pair drift by different
//! factors and the cancellation is partial. [`Drift`] carries two exponents for that reason, and
//! `state_dependent_drift_does_not_cancel` is the test that keeps the previous paragraph honest.
//!
//! # Units are SI at every interface
//!
//! Conductances in **siemens**, resistances in **ohms**, voltages in **volts**, currents in
//! **amperes**, times in **seconds**, temperatures in **kelvin**. [`Retention::ea_ev`] is the one
//! exception and it is deliberate: activation energies are universally published in electronvolts,
//! a reader checking this code against a reliability paper needs to see `1.0` and not
//! `1.602e-19`, and the conversion happens inside [`Retention::tau_s`] through
//! [`BOLTZMANN_EV_PER_K`], which is derived from the two exact SI constants rather than
//! transcribed.
//!
//! `beta` — [`DeviceModel::beta`] — is the boundary between the two worlds: **siemens per unit
//! weight**. A network's weights are dimensionless; a crossbar's cells are conductances; `beta` is
//! the single number that converts, and every range bound in this module is `span / beta` for that
//! reason.
//!
//! # ⛔ How the grades work here, and why so many are `Projected`
//!
//! Every model in this module carries a provenance string and a [`Evidence`] grade, following
//! [`crate::hardware`]. This module adds one rule to that scheme, and it is strict:
//!
//! > **A figure transcribed from the citing literature rather than read out of the cited document
//! > is [`Evidence::Projected`], not [`Evidence::Measured`].**
//!
//! That rule exists because this crate has already broken it once. [`crate::ledger::LOIHI_2018`]
//! shipped a pre-silicon `SPICE` number graded `Measured` in two releases, because its author took
//! the grade from the papers citing the table instead of from the table. The device physics
//! literature is a worse environment for that mistake than the chip literature: a drift exponent or
//! a variability sigma is a **distribution** whose reported value moves with the material stack,
//! the compliance current, the programming scheme and the measurement temperature, and the single
//! number that propagates through the secondary literature is a representative draw from it.
//!
//! So the constants below — [`PCM_DRIFT_AMORPHOUS`], [`RRAM_VARIABILITY_PLACEHOLDER`],
//! [`RRAM_STUCK_AT_PLACEHOLDER`] — are graded `Projected` and their strings say plainly what they
//! are. They exist so the machinery is runnable out of the box, and **any result reported from them
//! is a result about this crate's placeholders and not about any fabricated array.**
//! [`THERMAL_READ_NOISE_300K`] is the exception, graded [`Evidence::Derived`], because it is the
//! Johnson-Nyquist formula evaluated at stated parameters and transcribes no device figure at all.
//!
//! # What this module does not do
//!
//! It does not model the write circuit: programming is treated as a single noisy shot at a target,
//! not as the iterative write-verify loop every real array runs, which trades energy and endurance
//! for a tighter [`Variability::sigma_c2c_rel`]. It does not model endurance, read disturb,
//! sneak-path current through unselected cells (it assumes a selector, 1T1R or equivalent), the
//! nonlinearity of the cell's own `I-V` curve, or the analog-to-digital converter that eventually
//! has to read the column — which several published analyses find dominates the energy budget of a
//! real tile. Those are gaps in this implementation, named so that nobody mistakes a clean result
//! here for a clean result on silicon.
//!
//! # Primary sources
//!
//! * Johnson, *Thermal Agitation of Electricity in Conductors*, Physical Review 32:97-109, 1928;
//!   Nyquist, *Thermal Agitation of Electric Charge in Conductors*, Physical Review 32:110-113,
//!   1928 — the read-noise floor, and the one constant in this module that is not in doubt.
//! * Ielmini, Lavizzari, Sharma and Lacaita, *Physical interpretation, modeling and impact on phase
//!   change memory (PCM) reliability of resistance drift due to chalcogenide structural
//!   relaxation*, IEDM 2007 — the power-law drift model.
//! * Ielmini and Wong, *In-memory computing with resistive switching devices*, Nature Electronics
//!   1:333-343, 2018 — the review this module leans on for the non-ideality catalogue and for the
//!   wire-resistance treatment.
//! * Sebastian, Le Gallo, Khaddam-Aljameh and Eleftheriou, *Memory devices and applications for
//!   in-memory computing*, Nature Nanotechnology 15:529-544, 2020.
//! * Yu, *Neuro-inspired computing with emerging nonvolatile memories*, Proceedings of the IEEE
//!   106(2):260-285, 2018 — the finite-state-count limit.
//! * Prezioso, Merrikh-Bayat, Hoskins, Adam, Likharev and Strukov, *Training and operation of an
//!   integrated neuromorphic network based on metal-oxide memristors*, Nature 521:61-64, 2015;
//!   Ambrogio et al., *Equivalent-accuracy accelerated neural-network training using analogue
//!   memory*, Nature 558:60-67, 2018; Yao et al., *Fully hardware-implemented memristor
//!   convolutional neural network*, Nature 577:641-646, 2020 — fabricated arrays running networks,
//!   and the accuracy they actually reach.
//! * Joshi et al., *Accurate deep neural network inference using computational phase-change
//!   memory*, Nature Communications 11:2473, 2020 — drift compensation in practice.
//! * Chen, Lin, Li et al., *Accelerator-friendly neural-network training: learning variations and
//!   defects in RRAM crossbar*, DATE 2017 — the stuck-at fault model and the rates that motivate
//!   [`RRAM_STUCK_AT_PLACEHOLDER`].
//!
//! ⚠ **The list above is a reading list, not a provenance chain.** It names where each mechanism
//! comes from; it does not assert that this review opened each document and read the figure out of
//! it. Where a number rather than a mechanism was taken, the grading rule above applies and the
//! constant says so in its own string. A citation next to a model is a pointer for a reader; a
//! citation next to a **value** is a claim, and this module makes very few of those.
//!
//! # How this module was checked, and what that found
//!
//! Every mechanism here has at least one test against a closed form, an exactly-computable limit,
//! or an independent second algorithm — the crossbar solve is checked against a hand-written
//! resistive-ladder recursion that shares no code with it, and the read noise against a
//! hand-computed Johnson-Nyquist figure written as a literal.
//!
//! Those tests were then **attacked**: thirty-four deliberate defects were injected one at a time
//! and the suite was asked to catch each. A thirty-fifth attempt turned out to be an algebraically
//! equivalent rewrite of `exp(-x)` rather than a defect, and is not counted.
//!
//! **Nine survived the pass they were first tried in**, and what they were is the useful part:
//!
//! * **Three were documented decisions with no test at all** — the order retention and drift
//!   compose in, the independence of the fault map from the programming offsets, and the fact that
//!   a single-ended read ignores whatever is stored beside the cell.
//! * **Two were invisible in the answer and visible only in the generator's state**: a read that
//!   sampled a reference it then discarded, and a fault draw whose stream consumption depended on
//!   its own outcome. Neither changes a number; both re-roll every later draw in a caller's
//!   program, which is a worse failure because it looks like nothing.
//! * **One was a pair of same-typed `f64` fields** — [`ErrorStats::mean_signed`] and
//!   [`ErrorStats::rms`] — swappable in a struct literal and never noticed, because every test
//!   compared one of them against itself.
//! * **One was a fencepost** in [`ReadNoise::distinguishable_levels`] that every ratio test was
//!   blind to by construction: dropping the `+ 1` shifts both sides of a ratio.
//! * **One was a level grid applied to only half of each differential pair**, invisible while every
//!   test weight was positive.
//! * **One was a sub-model missing from the evidence fold**, which a test that weakened several
//!   models at once could never see — hence
//!   `the_weakest_evidence_is_an_exact_census_of_every_enabled_model`.
//!
//! Each has a test now, named for what it pins, and the injected defect is named in that test's doc
//! so the pairing survives a refactor.
//!
//! The ninth defect was in the code rather than the tests, and it is left visible:
//! [`Mapping::BalancedDifferential`] shipped its first draft claiming **half the weight range** of
//! the one-at-minimum scheme, with a matching `max_weight` and a test asserting it. It has the same
//! range. The real trade is [`Mapping::common_mode_conductance`], and the wrong claim survived
//! because the test was written from the doc instead of from the circuit.
//!
//! # Quickstart
//!
//! ```
//! use ferromorphic::device::{DeviceModel, Mapping, Window, Levels, PCM_DRIFT_AMORPHOUS};
//! use ferromorphic::ledger::Evidence;
//! use ferromorphic::rng::Rng;
//!
//! // A 1 uS to 100 uS window, signed weights on a differential pair, 1 uS per unit weight.
//! let window = Window::new(1e-6, 100e-6, "illustrative, not a device", Evidence::Unstated)?;
//! let mut model = DeviceModel::ideal();
//! model.window = window;
//! model.mapping = Mapping::Differential;
//! model.beta = 1e-6;
//! model.levels = Some(Levels::new(64, "illustrative", Evidence::Unstated)?);
//! model.drift = Some(PCM_DRIFT_AMORPHOUS);
//!
//! let want = [0.5, -0.25, 0.0, 12.0];
//! let mut rng = Rng::new(7);
//! let held = model.apply(&want, &mut rng)?;
//!
//! // One day later, at 85 C.
//! let later = held.aged(86_400.0, 358.15)?;
//! assert!(later.error().max_abs > held.error().max_abs);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use crate::ledger::{Evidence, weaker};
use crate::rng::Rng;
use core::fmt;

/// Boltzmann's constant, **joules per kelvin**.
///
/// Exact by definition since the 2019 SI redefinition — it is one of the seven defining constants,
/// so this is not a measurement with an uncertainty attached.
pub const BOLTZMANN_J_PER_K: f64 = 1.380649e-23;

/// The elementary charge, **coulombs**. Exact by the same 2019 redefinition.
pub const ELEMENTARY_CHARGE_C: f64 = 1.602176634e-19;

/// Boltzmann's constant in **electronvolts per kelvin**, `k_B / e`.
///
/// Derived from the two exact constants above rather than transcribed, so it cannot disagree with
/// them by a digit. Approximately `8.617333262e-5`; used by [`Retention`], whose activation energy
/// is published in electronvolts everywhere in the reliability literature.
pub const BOLTZMANN_EV_PER_K: f64 = BOLTZMANN_J_PER_K / ELEMENTARY_CHARGE_C;

/// Odd 64-bit mixing constant (the golden-ratio conjugate scaled to 64 bits), used to spread cell
/// indices across seeds so that adjacent cells get well-separated streams.
const PHI64: u64 = 0x9E37_79B9_7F4A_7C15;

/// Salt for the per-cell device-to-device stream.
const SALT_D2D: u64 = 0xD2D0_9A15_7E3C_0001;

/// Salt for the per-cell stuck-at stream. Different from [`SALT_D2D`] **on purpose**: it keeps the
/// fault map from moving when the variability model is changed, so the two can be studied apart.
const SALT_FAULT: u64 = 0xFA17_C0DE_51AB_0001;

/// Everything that can be wrong with a device model or its inputs, named.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DeviceError {
    /// A parameter was `NaN` or infinite. `what` names the parameter.
    NonFinite {
        /// Which parameter, by the name it has in this module.
        what: &'static str,
        /// The offending value, carried so a caller can print it.
        value: f64,
    },
    /// The conductance window is not a window: `g_off` negative, or `g_on` not strictly above it.
    BadWindow {
        /// Off-state conductance offered, siemens.
        g_off: f64,
        /// On-state conductance offered, siemens.
        g_on: f64,
    },
    /// A weight is `NaN` or infinite, at this index into the caller's slice.
    NonFiniteWeight {
        /// Index into the weight slice.
        index: usize,
        /// The offending weight.
        value: f64,
    },
    /// A weight is outside what this window and mapping can represent, at this index.
    ///
    /// Refused rather than clamped: a clamped weight is a silent accuracy loss of unbounded size,
    /// and the caller who wanted saturation can clamp before calling and know that they did.
    OutOfRange {
        /// Index into the weight slice.
        index: usize,
        /// The offending weight.
        weight: f64,
        /// Smallest representable weight for this mapping and window.
        min: f64,
        /// Largest representable weight.
        max: f64,
    },
    /// A probability was outside `[0, 1]`. `what` names which one.
    BadProbability {
        /// Which probability, by the name it has in this module.
        what: &'static str,
        /// The offending value.
        value: f64,
    },
    /// A quantity that must be strictly positive was not. `what` names it.
    NotPositive {
        /// Which quantity, by the name it has in this module.
        what: &'static str,
        /// The offending value.
        value: f64,
    },
    /// A quantity that must not be negative was. `what` names it.
    Negative {
        /// Which quantity, by the name it has in this module.
        what: &'static str,
        /// The offending value.
        value: f64,
    },
    /// Fewer than two conductance levels were asked for. One level is not a quantiser.
    BadLevels {
        /// The level count offered.
        levels: u32,
    },
    /// An empty weight slice or an empty crossbar. There is nothing to program.
    Empty,
    /// A conductance matrix's length does not match `rows * cols`.
    BadShape {
        /// Rows declared.
        rows: usize,
        /// Columns declared.
        cols: usize,
        /// Length of the slice actually supplied.
        len: usize,
    },
    /// A drive vector's length does not match the crossbar's row count.
    BadDrive {
        /// Rows in the crossbar.
        rows: usize,
        /// Length of the drive vector supplied.
        len: usize,
    },
    /// The circuit solve did not reach its tolerance in the sweeps allowed.
    ///
    /// Returned rather than handing back the last iterate, because an unconverged node-voltage
    /// solution looks entirely plausible — currents of the right magnitude, monotone in the right
    /// direction — and is wrong by an amount nobody downstream can bound.
    NotConverged {
        /// Sweeps performed before giving up.
        sweeps: u32,
        /// Worst Kirchhoff residual still outstanding, **amperes**.
        residual_a: f64,
        /// Absolute residual the solve was asked to reach, **amperes**.
        target_a: f64,
    },
    /// The tridiagonal line solve hit a zero or non-finite pivot.
    ///
    /// Should not be reachable for a diagonally dominant crossbar, and is returned instead of
    /// producing infinities so that if it ever happens it is visible rather than silent.
    SingularLine,
}

impl fmt::Display for DeviceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFinite { what, value } => write!(f, "{what} is not finite: {value}"),
            Self::BadWindow { g_off, g_on } => write!(
                f,
                "not a conductance window: g_off = {g_off} S, g_on = {g_on} S; need \
                 0 <= g_off < g_on, both finite"
            ),
            Self::NonFiniteWeight { index, value } => {
                write!(f, "weight {index} is not finite: {value}")
            }
            Self::OutOfRange { index, weight, min, max } => write!(
                f,
                "weight {index} = {weight} is outside the representable range [{min}, {max}] for \
                 this window and mapping; refused rather than clamped"
            ),
            Self::BadProbability { what, value } => {
                write!(f, "{what} = {value} is not a probability in [0, 1]")
            }
            Self::NotPositive { what, value } => {
                write!(f, "{what} = {value} must be strictly positive")
            }
            Self::Negative { what, value } => write!(f, "{what} = {value} must not be negative"),
            Self::BadLevels { levels } => write!(
                f,
                "{levels} conductance levels is not a quantiser; two is the minimum"
            ),
            Self::Empty => f.write_str("nothing to program: the array is empty"),
            Self::BadShape { rows, cols, len } => write!(
                f,
                "a {rows}x{cols} crossbar needs {} conductances, got {len}",
                rows * cols
            ),
            Self::BadDrive { rows, len } => {
                write!(f, "a crossbar with {rows} rows needs {rows} drive voltages, got {len}")
            }
            Self::NotConverged { sweeps, residual_a, target_a } => write!(
                f,
                "the crossbar solve did not converge in {sweeps} sweeps: worst Kirchhoff residual \
                 {residual_a:.3e} A against a target of {target_a:.3e} A"
            ),
            Self::SingularLine => {
                f.write_str("a tridiagonal line solve hit a zero or non-finite pivot")
            }
        }
    }
}

impl std::error::Error for DeviceError {}

/// A standard normal draw, by the Box-Muller transform.
///
/// Mean 0, variance 1, deterministic under [`Rng`]'s seed on every platform. Two uniforms are
/// consumed per draw and the second (sine) output of the transform is discarded, which costs half
/// the entropy and buys a generator with no hidden state — a cached second sample would make the
/// stream position depend on how many draws had been taken before, and every reproducibility claim
/// in this module rests on it not doing that.
///
/// `1.0 - u` is taken before the logarithm because [`Rng::next_f64`] returns `[0, 1)`, so `u` can
/// be exactly zero and `ln(0)` is an infinity; `1 - u` lies in `(0, 1]` and `ln` of it is finite.
#[must_use]
pub fn normal(rng: &mut Rng) -> f64 {
    let u1 = 1.0 - rng.next_f64();
    let u2 = rng.next_f64();
    (-2.0 * u1.ln()).sqrt() * (core::f64::consts::TAU * u2).cos()
}

/// A per-cell deterministic stream, salted so that different mechanisms draw independently.
fn cell_stream(seed: u64, salt: u64, index: usize) -> Rng {
    Rng::new(seed ^ salt.wrapping_add((index as u64).wrapping_mul(PHI64)))
}

/// The conductance range a device can actually be programmed into, **siemens**.
///
/// The two numbers are the whole of a device's analog capacity. Everything else in this module
/// either moves a cell inside this window or fails to.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Window {
    /// Off-state (high-resistance) conductance, **siemens**. The floor every cell has whether or
    /// not it was programmed to anything.
    pub g_off: f64,
    /// On-state (low-resistance) conductance, **siemens**. Strictly above [`Window::g_off`].
    pub g_on: f64,
    /// Which device, at which conditions, this window describes.
    ///
    /// A conductance window without a subject can be applied to any material at all, which is how
    /// a simulation comes to report a number about nothing.
    pub source: &'static str,
    /// What kind of evidence [`Window::source`] describes. See the module doc's grading rule.
    pub evidence: Evidence,
}

impl Window {
    /// A validated window.
    ///
    /// # Errors
    ///
    /// [`DeviceError::BadWindow`] unless `0 <= g_off < g_on` with both finite. `g_off == 0` is
    /// **permitted** and is a fiction: it means an infinite on/off ratio, which no fabricated
    /// device has. It exists so [`DeviceModel::ideal`] can round-trip a weight matrix bit for bit,
    /// and [`Window::on_off_ratio`] refuses for it rather than reporting an infinity.
    pub fn new(
        g_off: f64,
        g_on: f64,
        source: &'static str,
        evidence: Evidence,
    ) -> Result<Self, DeviceError> {
        let w = Self { g_off, g_on, source, evidence };
        w.validate()?;
        Ok(w)
    }

    /// Check the invariant `0 <= g_off < g_on`, both finite.
    ///
    /// Public because the fields are public: a `Window` built by struct literal can violate the
    /// invariant, and every entry point in this module calls this before computing with one.
    ///
    /// # Errors
    ///
    /// [`DeviceError::BadWindow`] when the invariant does not hold.
    pub fn validate(&self) -> Result<(), DeviceError> {
        if !self.g_off.is_finite()
            || !self.g_on.is_finite()
            || self.g_off < 0.0
            || !(self.g_on > self.g_off)
        {
            return Err(DeviceError::BadWindow { g_off: self.g_off, g_on: self.g_on });
        }
        Ok(())
    }

    /// `g_on - g_off`, **siemens**: the programmable span, and the numerator of every weight bound
    /// in this module.
    #[must_use]
    pub fn span(&self) -> f64 {
        self.g_on - self.g_off
    }

    /// The window's midpoint, **siemens**. Where [`Mapping::BalancedDifferential`] parks a zero
    /// weight.
    #[must_use]
    pub fn mid(&self) -> f64 {
        0.5 * (self.g_off + self.g_on)
    }

    /// `g_on / g_off`, or `None` when `g_off` is zero.
    ///
    /// `None` is not "very large". A zero off-conductance is a modelling fiction (see
    /// [`Window::new`]) and reporting `inf` for it would let an infinity propagate into a figure
    /// somebody prints.
    #[must_use]
    pub fn on_off_ratio(&self) -> Option<f64> {
        if self.g_off > 0.0 { Some(self.g_on / self.g_off) } else { None }
    }

    /// The fraction of a fully-on cell's read current that carries information: `1 - g_off / g_on`.
    ///
    /// Equivalently `1 - 1/ratio`. This is the quantity the on/off ratio actually controls in a
    /// **single-ended** array: the rest of the current is the floor every cell contributes whether
    /// programmed or not, and the sense amplifier has to reject it. A ratio of 2 means half the
    /// current is dead weight; a ratio of 100 means 1%. A differential pair cancels the floor in
    /// the subtraction, which is a second reason to pay for the extra device.
    #[must_use]
    pub fn signal_fraction(&self) -> f64 {
        1.0 - self.g_off / self.g_on
    }

    /// `g` brought inside the window.
    ///
    /// Written as explicit comparisons rather than [`f64::clamp`] **deliberately**: `f64::clamp`
    /// panics when its bounds cross, and the fields here are public, so a hand-built `Window` with
    /// `g_on < g_off` would turn a modelling mistake into an abort in the middle of a sweep. This
    /// form returns `g_off` for such a window, and the entry points refuse it outright through
    /// [`Window::validate`]. A `NaN` input returns `g` unchanged, because there is no sensible
    /// clamp of a `NaN` and the weight validators reject it earlier.
    #[must_use]
    pub fn clamp(&self, g: f64) -> f64 {
        if g < self.g_off {
            self.g_off
        } else if g > self.g_on {
            self.g_on
        } else {
            g
        }
    }

    /// Whether `g` lies within `[g_off, g_on]` inclusive.
    #[must_use]
    pub fn contains(&self, g: f64) -> bool {
        g >= self.g_off && g <= self.g_on
    }
}

/// How a signed weight is laid onto physical conductances.
///
/// A conductance is non-negative, so representing a signed weight needs either a reference to
/// subtract or a second device. The three schemes below differ in area, in range, and — the thing
/// that matters most in practice — in what happens to them under drift.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mapping {
    /// One device per weight, read against a **nominal constant** equal to `g_off`.
    ///
    /// Half the area, and it represents only non-negative weights: `g = g_off + w * beta`. The
    /// reference is a number in the digital domain, not a device, so it does not drift with the
    /// cell — which means drift shows up as an **additive offset** on the weight rather than as a
    /// gain, and no scalar recalibration removes it. That asymmetry is asserted in
    /// `uniform_drift_cancels_in_a_differential_pair_and_not_in_a_single_ended_read`.
    SingleEnded,
    /// Two devices, the unused one parked at `g_off`: `w >= 0` uses `G+`, `w < 0` uses `G-`.
    ///
    /// The full span in each direction, so the widest range of the three, and the common-mode floor
    /// cancels in the subtraction. Its cost is that one device of every pair sits at the bottom of
    /// its window, which is where several materials drift fastest and where the relative read noise
    /// is worst.
    Differential,
    /// Two devices placed symmetrically about the window's midpoint:
    /// `G+ = mid + w*beta/2`, `G- = mid - w*beta/2`.
    ///
    /// **The same weight range** as [`Mapping::Differential`] — the difference still swings a full
    /// span, it just swings symmetrically — so the first draft of this module's claim that it costs
    /// half the range was wrong, and `every_mapping_round_trips_and_the_schemes_differ_in_common_mode`
    /// is what caught it.
    ///
    /// What it buys is placement: both devices sit mid-window for every weight but full scale,
    /// which is where programming is most repeatable and where a state-dependent drift exponent
    /// varies least between the pair, whereas [`Mapping::Differential`] parks one device on the
    /// bottom rail for **every** weight.
    ///
    /// What it costs is current. A zero weight here holds both devices at `mid`, so the pair draws
    /// `(g_off + g_on) * v` of common-mode current that the column must carry and the sense
    /// amplifier must reject, against `2 * g_off * v` for the one-at-minimum scheme — a factor of
    /// `1 + ratio` more, which for a 100:1 window is a hundredfold. On a tall column that
    /// common-mode current, not the weight range, is what runs out first.
    /// [`Mapping::common_mode_conductance`] is the number.
    BalancedDifferential,
}

impl Mapping {
    /// How many physical devices one weight costs: 1 for [`Mapping::SingleEnded`], 2 otherwise.
    #[must_use]
    pub fn devices_per_weight(&self) -> u8 {
        match self {
            Self::SingleEnded => 1,
            Self::Differential | Self::BalancedDifferential => 2,
        }
    }

    /// Smallest representable weight for this window and `beta` (siemens per unit weight).
    ///
    /// Zero for [`Mapping::SingleEnded`], which cannot go negative at all.
    #[must_use]
    pub fn min_weight(&self, window: &Window, beta: f64) -> f64 {
        match self {
            Self::SingleEnded => 0.0,
            _ => -self.max_weight(window, beta),
        }
    }

    /// Largest representable weight: `span / beta`, the same for all three schemes.
    ///
    /// It is the same because the quantity read back is a **difference** of conductances in two of
    /// the three cases, and a difference can swing a full span whether one device does all the
    /// moving or both split it. This module's first draft claimed
    /// [`Mapping::BalancedDifferential`] cost half the range and it does not; the schemes differ in
    /// [`Mapping::common_mode_conductance`], not here.
    #[must_use]
    pub fn max_weight(&self, window: &Window, beta: f64) -> f64 {
        window.span() / beta
    }

    /// Total conductance a **zero** weight holds, per weight, **siemens**.
    ///
    /// The current the column carries and the sense amplifier has to reject before it can see any
    /// signal at all, and the quantity that actually separates the three schemes:
    /// `g_off` single-ended, `2 * g_off` for the one-at-minimum pair, `g_off + g_on` for the
    /// balanced pair. Multiply by the read voltage and by the number of rows for the column's
    /// common-mode current.
    #[must_use]
    pub fn common_mode_conductance(&self, window: &Window) -> f64 {
        match self {
            Self::SingleEnded => window.g_off,
            Self::Differential => 2.0 * window.g_off,
            Self::BalancedDifferential => window.g_off + window.g_on,
        }
    }

    /// The conductance pair a perfect programmer would write for `w`.
    ///
    /// For [`Mapping::SingleEnded`] the second element is the **nominal reference**, not a device:
    /// it is returned so that [`Mapping::read`] has one shape for all three schemes, and
    /// [`DeviceModel::apply`] leaves it untouched by noise, drift and faults.
    ///
    /// Assumes `w` is inside the representable range; [`DeviceModel::apply`] checks that first.
    #[must_use]
    pub fn program(&self, w: f64, window: &Window, beta: f64) -> (f64, f64) {
        match self {
            Self::SingleEnded => (window.g_off + w * beta, window.g_off),
            Self::Differential => {
                if w >= 0.0 {
                    (window.g_off + w * beta, window.g_off)
                } else {
                    (window.g_off, window.g_off - w * beta)
                }
            }
            Self::BalancedDifferential => {
                let mid = window.mid();
                let h = 0.5 * w * beta;
                (mid + h, mid - h)
            }
        }
    }

    /// The weight a reader recovers from a conductance pair.
    ///
    /// [`Mapping::SingleEnded`] subtracts the window's **nominal** `g_off` and ignores the second
    /// element entirely; the other two subtract the second device. That one line is the whole of
    /// the drift asymmetry described in this enum's variant docs.
    #[must_use]
    pub fn read(&self, g_plus: f64, g_minus: f64, window: &Window, beta: f64) -> f64 {
        match self {
            Self::SingleEnded => (g_plus - window.g_off) / beta,
            Self::Differential | Self::BalancedDifferential => (g_plus - g_minus) / beta,
        }
    }
}

/// Programming spread, split into the part that is a property of the **device** and the part that
/// is a property of the **write**.
///
/// The split is the point. Device-to-device spread is frozen at fabrication: reprogramming the
/// array does not move it, so a training scheme can learn around it and a calibration can measure
/// it once. Cycle-to-cycle spread is redrawn on every write, so nothing can learn around it and it
/// sets a floor on how precisely a weight can ever be placed. A model that lumps them into one
/// sigma cannot tell you which of those two situations you are in.
///
/// All three terms are applied as `g * (1 + s_d2d*z1 + s_c2c*z2) + s_floor*z3` with `z ~ N(0,1)`,
/// so the mean before clamping is exactly `g`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Variability {
    /// Device-to-device one-sigma spread, **relative** (0.10 is 10% of the target conductance).
    ///
    /// Drawn from a stream keyed by [`DeviceModel::d2d_seed`] and the cell index, so the same seed
    /// gives the same physical array across any number of reprogrammings.
    pub sigma_d2d_rel: f64,
    /// Cycle-to-cycle one-sigma spread, **relative**. Redrawn from the caller's generator on every
    /// [`DeviceModel::apply`].
    pub sigma_c2c_rel: f64,
    /// An absolute one-sigma floor, **siemens**, added on top of the relative terms.
    ///
    /// Relative-only noise vanishes at `g = 0`, which is wrong: a cell programmed near the bottom
    /// of its window is not placed more precisely in absolute terms than one at the top, and for
    /// several materials it is placed less precisely. This term is how that is expressed.
    pub sigma_floor_s: f64,
    /// Which device, measured how, this spread describes.
    pub source: &'static str,
    /// Evidence grade for [`Variability::source`]. See the module doc's grading rule.
    pub evidence: Evidence,
}

impl Variability {
    /// A validated variability model.
    ///
    /// # Errors
    ///
    /// [`DeviceError::Negative`] for a negative sigma and [`DeviceError::NonFinite`] for a
    /// non-finite one, naming the field.
    pub fn new(
        sigma_d2d_rel: f64,
        sigma_c2c_rel: f64,
        sigma_floor_s: f64,
        source: &'static str,
        evidence: Evidence,
    ) -> Result<Self, DeviceError> {
        let v = Self { sigma_d2d_rel, sigma_c2c_rel, sigma_floor_s, source, evidence };
        v.validate()?;
        Ok(v)
    }

    /// Check that every sigma is finite and non-negative.
    ///
    /// # Errors
    ///
    /// [`DeviceError::NonFinite`] or [`DeviceError::Negative`], naming the field.
    pub fn validate(&self) -> Result<(), DeviceError> {
        for (what, v) in [
            ("sigma_d2d_rel", self.sigma_d2d_rel),
            ("sigma_c2c_rel", self.sigma_c2c_rel),
            ("sigma_floor_s", self.sigma_floor_s),
        ] {
            if !v.is_finite() {
                return Err(DeviceError::NonFinite { what, value: v });
            }
            if v < 0.0 {
                return Err(DeviceError::Negative { what, value: v });
            }
        }
        Ok(())
    }
}

/// Power-law conductance drift after programming.
///
/// `G(t) = G(t0) * (t / t0)^(-nu)` for `t >= t0`, which is the resistance-drift law of
/// Ielmini et al. (IEDM 2007) written for conductance instead of resistance. `t0` is a **reference
/// time, not a start time**: the law diverges as `t -> 0`, it is fitted to data taken after the
/// programming transient has settled, and this implementation therefore returns the programmed
/// value unchanged for `t <= t0` rather than extrapolating into a region the model does not cover.
///
/// # Why there are two exponents
///
/// The drift exponent is not a constant of the material: it grows with the amorphous fraction, so a
/// cell in the high-resistance state drifts faster than one in the low-resistance state. That is
/// what stops a differential pair from cancelling drift exactly, and a single-exponent model would
/// report a clean cancellation that no fabricated array shows.
///
/// The interpolation between the two — **linear in conductance across the window** — is a
/// convention of this implementation, not a published law. It is graded accordingly wherever it is
/// used, and it is chosen because it is monotone, reduces to the single-exponent case when the two
/// are equal, and cannot produce a negative exponent from two non-negative ones.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Drift {
    /// Drift exponent at the **off** (high-resistance) rail, dimensionless.
    pub nu_at_off: f64,
    /// Drift exponent at the **on** (low-resistance) rail, dimensionless. Typically well below
    /// [`Drift::nu_at_off`].
    pub nu_at_on: f64,
    /// Reference time, **seconds**. `G` is defined to equal its programmed value here, and the law
    /// says nothing before it.
    pub t0_s: f64,
    /// Which material, at which temperature, this exponent describes.
    pub source: &'static str,
    /// Evidence grade for [`Drift::source`]. See the module doc's grading rule.
    pub evidence: Evidence,
}

impl Drift {
    /// A drift model with one exponent everywhere in the window.
    ///
    /// The state-independent case: useful because it is the one where the differential-pair
    /// cancellation is exact, and therefore the one a closed-form test can pin.
    ///
    /// # Errors
    ///
    /// As [`Drift::new`].
    pub fn uniform(
        nu: f64,
        t0_s: f64,
        source: &'static str,
        evidence: Evidence,
    ) -> Result<Self, DeviceError> {
        Self::new(nu, nu, t0_s, source, evidence)
    }

    /// A drift model with a state-dependent exponent.
    ///
    /// # Errors
    ///
    /// [`DeviceError::NonFinite`] for a non-finite parameter, [`DeviceError::Negative`] for a
    /// negative exponent, [`DeviceError::NotPositive`] for a non-positive `t0_s`.
    pub fn new(
        nu_at_off: f64,
        nu_at_on: f64,
        t0_s: f64,
        source: &'static str,
        evidence: Evidence,
    ) -> Result<Self, DeviceError> {
        let d = Self { nu_at_off, nu_at_on, t0_s, source, evidence };
        d.validate()?;
        Ok(d)
    }

    /// Check exponents are finite and non-negative and that `t0_s` is positive.
    ///
    /// # Errors
    ///
    /// [`DeviceError::NonFinite`], [`DeviceError::Negative`] or [`DeviceError::NotPositive`],
    /// naming the field.
    pub fn validate(&self) -> Result<(), DeviceError> {
        for (what, v) in [("nu_at_off", self.nu_at_off), ("nu_at_on", self.nu_at_on)] {
            if !v.is_finite() {
                return Err(DeviceError::NonFinite { what, value: v });
            }
            if v < 0.0 {
                return Err(DeviceError::Negative { what, value: v });
            }
        }
        if !self.t0_s.is_finite() {
            return Err(DeviceError::NonFinite { what: "t0_s", value: self.t0_s });
        }
        if !(self.t0_s > 0.0) {
            return Err(DeviceError::NotPositive { what: "t0_s", value: self.t0_s });
        }
        Ok(())
    }

    /// The exponent this model uses for a cell sitting at `g`, interpolated linearly across the
    /// window and held flat outside it.
    ///
    /// Falls back to [`Drift::nu_at_off`] for a degenerate window rather than dividing by zero;
    /// the entry points refuse such a window before reaching here.
    #[must_use]
    pub fn exponent_at(&self, g: f64, window: &Window) -> f64 {
        let span = window.span();
        if !(span > 0.0) {
            return self.nu_at_off;
        }
        let f = ((g - window.g_off) / span).clamp(0.0, 1.0);
        self.nu_at_off + f * (self.nu_at_on - self.nu_at_off)
    }

    /// The multiplicative factor `(t / t0)^(-nu(g))`, or exactly `1.0` for `t <= t0`.
    #[must_use]
    pub fn factor(&self, g: f64, window: &Window, t_s: f64) -> f64 {
        if !(t_s > self.t0_s) {
            return 1.0;
        }
        (t_s / self.t0_s).powf(-self.exponent_at(g, window))
    }

    /// `g` after drifting for `t_s` seconds.
    ///
    /// **Not clamped to the window.** Drift carries a cell below its programmed off-state, and that
    /// is exactly the failure retention engineering exists to bound; clamping it would hide the
    /// thing the model is for.
    #[must_use]
    pub fn apply(&self, g: f64, window: &Window, t_s: f64) -> f64 {
        g * self.factor(g, window, t_s)
    }
}

/// Thermally activated loss of the programmed state: the Arrhenius half of ageing.
///
/// `tau(T) = tau0 * exp(Ea / (k_B T))`, and the programmed excess above the off-state decays as
/// `exp(-t / tau(T))`, so a cell relaxes toward `g_off` rather than toward zero. This is a
/// **first-order lumped model**: real retention failure in a filamentary cell is a distribution of
/// activation energies across a population and is usually reported as a time-to-fail at a
/// percentile, not as a single exponential. The exponential is what is implemented here, the
/// Arrhenius temperature dependence is the part that is on solid ground, and the difference is
/// stated rather than papered over.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Retention {
    /// Activation energy, **electronvolts**. The one non-SI unit in this module, kept because
    /// every reliability paper publishes it this way; see the module doc.
    pub ea_ev: f64,
    /// Pre-exponential time constant, **seconds**: the retention time this model would give at
    /// infinite temperature, and the only free parameter once `Ea` is fixed.
    pub tau0_s: f64,
    /// Which device and which failure criterion this describes.
    pub source: &'static str,
    /// Evidence grade for [`Retention::source`]. See the module doc's grading rule.
    pub evidence: Evidence,
}

impl Retention {
    /// A validated retention model.
    ///
    /// # Errors
    ///
    /// [`DeviceError::NonFinite`] for a non-finite parameter, [`DeviceError::Negative`] for a
    /// negative `ea_ev`, [`DeviceError::NotPositive`] for a non-positive `tau0_s`.
    pub fn new(
        ea_ev: f64,
        tau0_s: f64,
        source: &'static str,
        evidence: Evidence,
    ) -> Result<Self, DeviceError> {
        let r = Self { ea_ev, tau0_s, source, evidence };
        r.validate()?;
        Ok(r)
    }

    /// Check `ea_ev >= 0` and `tau0_s > 0`, both finite.
    ///
    /// # Errors
    ///
    /// [`DeviceError::NonFinite`], [`DeviceError::Negative`] or [`DeviceError::NotPositive`].
    pub fn validate(&self) -> Result<(), DeviceError> {
        if !self.ea_ev.is_finite() {
            return Err(DeviceError::NonFinite { what: "ea_ev", value: self.ea_ev });
        }
        if self.ea_ev < 0.0 {
            return Err(DeviceError::Negative { what: "ea_ev", value: self.ea_ev });
        }
        if !self.tau0_s.is_finite() {
            return Err(DeviceError::NonFinite { what: "tau0_s", value: self.tau0_s });
        }
        if !(self.tau0_s > 0.0) {
            return Err(DeviceError::NotPositive { what: "tau0_s", value: self.tau0_s });
        }
        Ok(())
    }

    /// Retention time constant at `temp_k`, **seconds**.
    ///
    /// # Errors
    ///
    /// [`DeviceError::NotPositive`] for a non-positive temperature and [`DeviceError::NonFinite`]
    /// for a non-finite one. There is no Arrhenius rate at or below absolute zero, and returning a
    /// large number for one would be a silent lie.
    pub fn tau_s(&self, temp_k: f64) -> Result<f64, DeviceError> {
        check_temperature(temp_k)?;
        Ok(self.tau0_s * (self.ea_ev / (BOLTZMANN_EV_PER_K * temp_k)).exp())
    }

    /// The acceleration factor between a use temperature and a stress temperature:
    /// `exp(Ea/k_B * (1/T_use - 1/T_stress))`.
    ///
    /// The number every accelerated-lifetime test is reported through. Exactly `1.0` when the two
    /// temperatures are equal or when `Ea` is zero, and it composes:
    /// `AF(a,b) * AF(b,c) == AF(a,c)`. Both are asserted.
    ///
    /// # Errors
    ///
    /// As [`Retention::tau_s`], for either temperature.
    pub fn acceleration(&self, t_use_k: f64, t_stress_k: f64) -> Result<f64, DeviceError> {
        check_temperature(t_use_k)?;
        check_temperature(t_stress_k)?;
        let k = self.ea_ev / BOLTZMANN_EV_PER_K;
        Ok((k * (1.0 / t_use_k - 1.0 / t_stress_k)).exp())
    }

    /// The fraction of the programmed excess conductance still present after `t_s` at `temp_k`.
    ///
    /// `exp(-t / tau(T))`, so exactly `1.0` at `t = 0` and `exp(-1)` at `t = tau`.
    ///
    /// # Errors
    ///
    /// As [`Retention::tau_s`], plus [`DeviceError::Negative`] for a negative time.
    pub fn remaining_fraction(&self, t_s: f64, temp_k: f64) -> Result<f64, DeviceError> {
        if !t_s.is_finite() {
            return Err(DeviceError::NonFinite { what: "t_s", value: t_s });
        }
        if t_s < 0.0 {
            return Err(DeviceError::Negative { what: "t_s", value: t_s });
        }
        Ok((-t_s / self.tau_s(temp_k)?).exp())
    }
}

/// Reject a temperature that is not a positive finite number of kelvin.
fn check_temperature(temp_k: f64) -> Result<(), DeviceError> {
    if !temp_k.is_finite() {
        return Err(DeviceError::NonFinite { what: "temp_k", value: temp_k });
    }
    if !(temp_k > 0.0) {
        return Err(DeviceError::NotPositive { what: "temp_k", value: temp_k });
    }
    Ok(())
}

/// A cell that does not respond to programming.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fault {
    /// The cell programs normally.
    Healthy,
    /// Pinned at [`Window::g_on`] whatever was written — a shorted or permanently-formed cell.
    StuckOn,
    /// Pinned at [`Window::g_off`] whatever was written — an unformed or open cell.
    StuckOff,
}

impl Fault {
    /// Whether this cell ignores what was written to it.
    #[must_use]
    pub fn is_stuck(&self) -> bool {
        !matches!(self, Self::Healthy)
    }
}

/// Manufacturing defects: a fraction of cells pinned to a rail.
///
/// Stuck-at faults are the non-ideality with the largest reported spread in the literature and the
/// largest effect on accuracy per unit of it, because a stuck cell is not a small error on a weight
/// — it is an arbitrary weight, at full scale, in a random place. They are drawn from a stream
/// keyed by [`DeviceModel::d2d_seed`], **not** from the caller's generator, because a defect is a
/// property of the fabricated array and does not move when the array is rewritten.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StuckAt {
    /// Probability that a given **device** (not weight) is stuck, in `[0, 1]`.
    ///
    /// A [`Mapping::Differential`] weight costs two devices and therefore has roughly twice the
    /// chance of being touched by a fault, which is a real cost of the scheme and is not adjusted
    /// away here.
    pub rate: f64,
    /// Given that a device is stuck, the probability it is stuck **on** rather than off, in
    /// `[0, 1]`. Rarely 0.5 in a real array; the two failure mechanisms are unrelated.
    pub p_at_on: f64,
    /// Which array, at which maturity, this rate describes.
    pub source: &'static str,
    /// Evidence grade for [`StuckAt::source`]. See the module doc's grading rule.
    pub evidence: Evidence,
}

impl StuckAt {
    /// A validated fault model.
    ///
    /// # Errors
    ///
    /// [`DeviceError::BadProbability`] for a rate or split outside `[0, 1]`, or a non-finite one.
    pub fn new(
        rate: f64,
        p_at_on: f64,
        source: &'static str,
        evidence: Evidence,
    ) -> Result<Self, DeviceError> {
        let s = Self { rate, p_at_on, source, evidence };
        s.validate()?;
        Ok(s)
    }

    /// Check both probabilities lie in `[0, 1]`.
    ///
    /// # Errors
    ///
    /// [`DeviceError::BadProbability`], naming the field.
    pub fn validate(&self) -> Result<(), DeviceError> {
        for (what, v) in [("rate", self.rate), ("p_at_on", self.p_at_on)] {
            if !v.is_finite() || !(0.0..=1.0).contains(&v) {
                return Err(DeviceError::BadProbability { what, value: v });
            }
        }
        Ok(())
    }

    /// Draw this device's fault from `rng`.
    ///
    /// Always consumes **exactly two uniforms** whether or not a fault occurs, so the stream
    /// position after a draw does not depend on the outcome. That is what makes a fault map a
    /// function of the seed and the rate alone: with a variable consumption, the second device of a
    /// pair would read from an offset that depends on whether the first one failed, so changing
    /// [`StuckAt::p_at_on`] — or any earlier draw — would silently re-roll the rest of the array.
    /// `the_fault_draw_consumes_a_fixed_number_of_uniforms` pins it on the generator's state.
    #[must_use]
    pub fn draw(&self, rng: &mut Rng) -> Fault {
        let u = rng.next_f64();
        let v = rng.next_f64();
        if u < self.rate {
            if v < self.p_at_on { Fault::StuckOn } else { Fault::StuckOff }
        } else {
            Fault::Healthy
        }
    }
}

/// Noise on a read, with a thermal floor that is not negotiable.
///
/// Two terms. The **Johnson-Nyquist** term is the thermal agitation of the carriers in the cell
/// itself: a conductance `G` at temperature `T` read over a bandwidth `B` delivers a mean-square
/// current noise of `4 k_B T G B` amperes squared, from Johnson and Nyquist (Physical Review 32,
/// 1928). It depends on no device parameter beyond `G`, it cannot be engineered away, and it is
/// the reason "how many levels does this cell have" has a physical answer.
///
/// The **relative** term stands in for everything else — random telegraph noise from single defects
/// switching in the conduction path, `1/f` noise, supply and reference noise. It is expressed as a
/// fraction of the cell's own current because that is how it is usually reported. In every
/// fabricated array this review is aware of it dominates the thermal floor by orders of magnitude,
/// and this implementation **did not locate a single agreed figure for it**, which is why
/// [`THERMAL_READ_NOISE_300K`] leaves it at zero and says so.
///
/// Bandwidth is the knob a system designer actually has: halving it halves the noise power, at the
/// cost of doubling the read time. [`ReadNoise::distinguishable_levels`] is where that trade
/// becomes a number.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReadNoise {
    /// Read temperature, **kelvin**.
    pub temp_k: f64,
    /// Noise-equivalent bandwidth of the read path, **hertz**.
    pub bandwidth_hz: f64,
    /// One-sigma relative current noise beyond the thermal floor, dimensionless.
    ///
    /// `0.01` means a one-percent read-to-read spread on a cell's current. Zero means "the thermal
    /// floor only", which is a statement about the model and not about any device.
    pub sigma_rel: f64,
    /// Which read path, at which speed, this describes.
    pub source: &'static str,
    /// Evidence grade for [`ReadNoise::source`]. See the module doc's grading rule.
    pub evidence: Evidence,
}

impl ReadNoise {
    /// A validated read-noise model.
    ///
    /// # Errors
    ///
    /// [`DeviceError::NotPositive`] for a non-positive temperature or bandwidth,
    /// [`DeviceError::Negative`] for a negative `sigma_rel`, [`DeviceError::NonFinite`] for any
    /// non-finite parameter.
    pub fn new(
        temp_k: f64,
        bandwidth_hz: f64,
        sigma_rel: f64,
        source: &'static str,
        evidence: Evidence,
    ) -> Result<Self, DeviceError> {
        let n = Self { temp_k, bandwidth_hz, sigma_rel, source, evidence };
        n.validate()?;
        Ok(n)
    }

    /// Check temperature and bandwidth are positive and `sigma_rel` is non-negative.
    ///
    /// # Errors
    ///
    /// [`DeviceError::NonFinite`], [`DeviceError::NotPositive`] or [`DeviceError::Negative`].
    pub fn validate(&self) -> Result<(), DeviceError> {
        check_temperature(self.temp_k)?;
        if !self.bandwidth_hz.is_finite() {
            return Err(DeviceError::NonFinite { what: "bandwidth_hz", value: self.bandwidth_hz });
        }
        if !(self.bandwidth_hz > 0.0) {
            return Err(DeviceError::NotPositive { what: "bandwidth_hz", value: self.bandwidth_hz });
        }
        if !self.sigma_rel.is_finite() {
            return Err(DeviceError::NonFinite { what: "sigma_rel", value: self.sigma_rel });
        }
        if self.sigma_rel < 0.0 {
            return Err(DeviceError::Negative { what: "sigma_rel", value: self.sigma_rel });
        }
        Ok(())
    }

    /// One-sigma current noise for a cell of conductance `g` read at `v_read` volts, **amperes**.
    ///
    /// `sqrt(4 k_B T g B + (sigma_rel * g * v_read)^2)`: the two terms add in power, not in
    /// amplitude. Returns `None` for a negative conductance or a non-finite argument, because
    /// neither has a noise power.
    #[must_use]
    pub fn sigma_current(&self, g: f64, v_read: f64) -> Option<f64> {
        if !g.is_finite() || !v_read.is_finite() || g < 0.0 {
            return None;
        }
        let thermal = 4.0 * BOLTZMANN_J_PER_K * self.temp_k * g * self.bandwidth_hz;
        let relative = self.sigma_rel * g * v_read;
        Some((thermal + relative * relative).sqrt())
    }

    /// The same noise expressed as an uncertainty on the inferred conductance, **siemens**.
    ///
    /// `sigma_I / v_read`. `None` for a non-positive `v_read`: a read at zero volts carries no
    /// current, so no conductance can be inferred from it and the uncertainty is not large, it is
    /// undefined.
    #[must_use]
    pub fn sigma_conductance(&self, g: f64, v_read: f64) -> Option<f64> {
        if !(v_read > 0.0) {
            return None;
        }
        self.sigma_current(g, v_read).map(|s| s / v_read)
    }

    /// One noisy realisation of a read of `g`, **siemens**.
    ///
    /// Additive Gaussian, so unbiased: the expectation over many reads is exactly `g`. Returns
    /// `None` on the same conditions as [`ReadNoise::sigma_conductance`], and **does not draw from
    /// `rng` when it returns `None`**, so a refused read leaves the stream where it was.
    #[must_use]
    pub fn sample(&self, rng: &mut Rng, g: f64, v_read: f64) -> Option<f64> {
        let sigma = self.sigma_conductance(g, v_read)?;
        Some(g + sigma * normal(rng))
    }

    /// How many conductance levels this noise leaves distinguishable across `window`.
    ///
    /// `separation` is the number of sigmas demanded between adjacent levels and is the caller's
    /// convention, not a law: `separation = 6` is the usual engineering choice for a decision that
    /// should almost never be wrong, `separation = 2` is optimistic. The sigma is evaluated at the
    /// **top** of the window, where thermal noise is largest, so the answer is the conservative
    /// one.
    ///
    /// `None` for a non-positive `v_read` or `separation`, for an invalid window, or when the noise
    /// is exactly zero — in which case the number of levels is not large, it is unbounded, and this
    /// module declines to print an infinity as an integer.
    #[must_use]
    pub fn distinguishable_levels(
        &self,
        window: &Window,
        v_read: f64,
        separation: f64,
    ) -> Option<u32> {
        window.validate().ok()?;
        if !(separation > 0.0) || !separation.is_finite() {
            return None;
        }
        let sigma = self.sigma_conductance(window.g_on, v_read)?;
        if !(sigma > 0.0) {
            return None;
        }
        let n = window.span() / (separation * sigma) + 1.0;
        if !n.is_finite() || n < 2.0 {
            return Some(1);
        }
        Some(n.floor().min(f64::from(u32::MAX)) as u32)
    }
}

/// The finite number of conductance states a cell can be written to.
///
/// A weight in a simulator is a real number. A weight in a cell is one of `n` levels, and `n` is
/// small: Yu (Proceedings of the IEEE 106(2), 2018) is one of several reviews reporting that the
/// number of states a resistive cell can be written **and read back** reliably is far below what
/// training assumes, with single-digit effective bits typical. This is the same kind of object as
/// [`crate::hardware::Quantiser`] and deliberately not the same code: that one quantises a
/// **signed weight** onto a part's digital bit-width, this one quantises a **conductance** onto a
/// physical window, and conflating them is how a model comes to quantise twice or not at all.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Levels {
    /// Number of distinguishable programmed states, at least 2, evenly spaced from
    /// [`Window::g_off`] to [`Window::g_on`] inclusive.
    ///
    /// Even spacing in **conductance** is the convention here and it is a choice: several
    /// programming schemes give levels evenly spaced in resistance instead, which is a different
    /// grid entirely, and this implementation does not offer it.
    pub n: u32,
    /// Which device and which programming scheme achieved this level count.
    pub source: &'static str,
    /// Evidence grade for [`Levels::source`]. See the module doc's grading rule.
    pub evidence: Evidence,
}

impl Levels {
    /// A validated level grid.
    ///
    /// # Errors
    ///
    /// [`DeviceError::BadLevels`] for fewer than two levels.
    pub fn new(n: u32, source: &'static str, evidence: Evidence) -> Result<Self, DeviceError> {
        if n < 2 {
            return Err(DeviceError::BadLevels { levels: n });
        }
        Ok(Self { n, source, evidence })
    }

    /// Spacing between adjacent levels, **siemens**: `span / (n - 1)`.
    #[must_use]
    pub fn step(&self, window: &Window) -> f64 {
        window.span() / f64::from(self.n.max(2) - 1)
    }

    /// Effective bits, `log2(n)`. Fractional on purpose: 20 levels is 4.32 bits, not 4, and
    /// rounding it down is how a device gets reported as worse than it is.
    #[must_use]
    pub fn bits(&self) -> f64 {
        f64::from(self.n).log2()
    }

    /// `g` snapped to the nearest level, clamped into the window.
    ///
    /// The result is always exactly `g_off + k * step` for an integer `k` in `0..n`, and the error
    /// is at most half a step for any `g` already inside the window. Both are asserted.
    #[must_use]
    pub fn snap(&self, g: f64, window: &Window) -> f64 {
        let step = self.step(window);
        if !(step > 0.0) || !g.is_finite() {
            return window.clamp(g);
        }
        let k = ((g - window.g_off) / step)
            .round()
            .clamp(0.0, f64::from(self.n.max(2) - 1));
        window.clamp(window.g_off + k * step)
    }
}

/// The resistance of the metal that carries the answer.
///
/// One number: the resistance of a single wire segment between two adjacent cross-points, the same
/// on word lines and bit lines. That is a simplification — real word and bit lines are on different
/// metal layers with different sheet resistances — and it is the simplification the literature's
/// own analyses usually make.
///
/// The reason this is a separate object from everything else in the module is in
/// [`Crossbar`]: it is the only non-ideality whose effect on a cell depends on **where the cell
/// is**, so it cannot be folded into a per-weight perturbation, and folding it in anyway is the
/// standard way an analog accuracy study comes out optimistic.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Wires {
    /// Resistance of one inter-cell wire segment, **ohms**. Zero is legal and means an ideal
    /// crossbar, which is the reference case every IR-drop figure in this module is measured
    /// against.
    pub r_segment_ohm: f64,
    /// Which array geometry and metal stack this resistance describes.
    pub source: &'static str,
    /// Evidence grade for [`Wires::source`]. See the module doc's grading rule.
    pub evidence: Evidence,
}

impl Wires {
    /// A validated wire model.
    ///
    /// # Errors
    ///
    /// [`DeviceError::Negative`] for a negative resistance, [`DeviceError::NonFinite`] for a
    /// non-finite one.
    pub fn new(
        r_segment_ohm: f64,
        source: &'static str,
        evidence: Evidence,
    ) -> Result<Self, DeviceError> {
        if !r_segment_ohm.is_finite() {
            return Err(DeviceError::NonFinite { what: "r_segment_ohm", value: r_segment_ohm });
        }
        if r_segment_ohm < 0.0 {
            return Err(DeviceError::Negative { what: "r_segment_ohm", value: r_segment_ohm });
        }
        Ok(Self { r_segment_ohm, source, evidence })
    }

    /// An ideal, zero-resistance interconnect. A fiction, and the reference case.
    #[must_use]
    pub fn ideal() -> Self {
        Self {
            r_segment_ohm: 0.0,
            source: "no wire resistance: the reference case against which IR drop is measured, not \
                     a fabricated array",
            evidence: Evidence::Unstated,
        }
    }
}

/// Amorphous phase-change drift, as the secondary literature reports it.
///
/// `nu ~ 0.1` in the high-resistance (amorphous) state and `~0.02` in the low-resistance
/// (crystalline) state, with a one-second reference time. The law is Ielmini et al., IEDM 2007.
///
/// ⚠ **Graded [`Evidence::Projected`] and it should stay that way until somebody opens the table.**
/// These exponents are quoted throughout the phase-change in-memory-computing literature and this
/// review transcribed them from that citing literature rather than from the cited figure, which is
/// precisely the habit that produced the defect recorded in [`crate::ledger::LOIHI_2018`]. The
/// exponent is also a **distribution**, not a constant: it varies cell to cell and with the
/// programmed state, and a single pair of numbers is a representative draw from it.
pub const PCM_DRIFT_AMORPHOUS: Drift = Drift {
    nu_at_off: 0.1,
    nu_at_on: 0.02,
    t0_s: 1.0,
    source: "power-law drift nu ~ 0.1 amorphous / ~0.02 crystalline, t0 = 1 s. Law from Ielmini, \
             Lavizzari, Sharma and Lacaita, IEDM 2007; the exponents transcribed from the citing \
             literature and NOT read out of the cited figure, so graded Projected per this \
             module's rule. The exponent is a distribution, not a constant.",
    evidence: Evidence::Projected,
};

/// A placeholder programming spread, so the machinery runs out of the box.
///
/// ⛔ **These numbers are not a measurement of anything.** 10% device-to-device and 5%
/// cycle-to-cycle are order-of-magnitude placeholders chosen because published spreads for
/// metal-oxide cells span at least a few percent to tens of percent depending on the material
/// stack, the compliance current, the programming scheme and the state, and this review did not
/// locate a single figure it could defend as representative. Any result reported from this constant
/// is a result about this constant.
pub const RRAM_VARIABILITY_PLACEHOLDER: Variability = Variability {
    sigma_d2d_rel: 0.10,
    sigma_c2c_rel: 0.05,
    sigma_floor_s: 0.0,
    source: "PLACEHOLDER, not a measurement: 10% device-to-device, 5% cycle-to-cycle, no absolute \
             floor. Order-of-magnitude values so the model is runnable; replace with a figure \
             measured on your array before reporting anything.",
    evidence: Evidence::Projected,
};

/// A placeholder stuck-at rate for an immature resistive array.
///
/// ⛔ **A placeholder, on the same terms as [`RRAM_VARIABILITY_PLACEHOLDER`].** Fault rates around
/// ten percent motivate the fault-tolerant training literature — Chen, Lin, Li et al., DATE 2017 is
/// the usual entry point — and the rate is a property of a particular array at a particular
/// maturity, so it varies by orders of magnitude between a research wafer and a product. The
/// half-and-half split between stuck-on and stuck-off is a convention of this crate, not a
/// measurement: the two failure mechanisms are unrelated and there is no reason for them to be
/// equally likely.
pub const RRAM_STUCK_AT_PLACEHOLDER: StuckAt = StuckAt {
    rate: 0.10,
    p_at_on: 0.5,
    source: "PLACEHOLDER, not a measurement: 10% of devices stuck, split evenly between rails. \
             Motivated by the fault-tolerance literature (Chen, Lin, Li et al., DATE 2017); the \
             split is this crate's convention. Replace with your array's measured map.",
    evidence: Evidence::Projected,
};

/// The Johnson-Nyquist floor at room temperature over a one-megahertz read.
///
/// Graded [`Evidence::Derived`] — the only constant in this module that is not `Projected` —
/// because it transcribes **no device figure at all**. It is `4 k_B T G B` with `T = 300 K` and
/// `B = 1 MHz`, computed from a constant that is exact by definition. [`ReadNoise::sigma_rel`] is
/// zero, which means this constant models the floor and nothing above it; in every fabricated array
/// the floor is not the binding term.
pub const THERMAL_READ_NOISE_300K: ReadNoise = ReadNoise {
    temp_k: 300.0,
    bandwidth_hz: 1e6,
    sigma_rel: 0.0,
    source: "Johnson-Nyquist thermal floor only: 4 k_B T G B at T = 300 K, B = 1 MHz, from Johnson \
             and Nyquist, Physical Review 32, 1928. No device figure is transcribed and no \
             non-thermal noise is modelled; a real array's read noise is larger.",
    evidence: Evidence::Derived,
};

/// A complete analog device model: a window, a mapping, and whichever non-idealities are switched
/// on.
///
/// Every non-ideality is an `Option`, and `None` means **not modelled**, which is a different
/// statement from "not present". A model with five `None` fields that reports a small error is
/// reporting a small error about the one mechanism it has.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DeviceModel {
    /// The conductance range a cell can be programmed into.
    pub window: Window,
    /// How a signed weight becomes one or two conductances.
    pub mapping: Mapping,
    /// **Siemens per unit weight**: the single number converting the network's dimensionless
    /// weights into the array's physical conductances. Must be finite and strictly positive.
    pub beta: f64,
    /// Seed for the frozen, per-cell properties: device-to-device spread and the stuck-at map.
    ///
    /// Two `DeviceModel`s with the same seed are **the same physical array**, however many times
    /// they are reprogrammed and whatever generator is passed to [`DeviceModel::apply`]. That is
    /// the distinction between a defect and a write error, made operational.
    pub d2d_seed: u64,
    /// Programming spread, or `None` for a perfect programmer.
    pub variability: Option<Variability>,
    /// Finite state count, or `None` for a cell that holds a real number.
    pub levels: Option<Levels>,
    /// Manufacturing defects, or `None` for a perfect array.
    pub stuck: Option<StuckAt>,
    /// Post-programming drift, or `None` for a cell that stays where it was put.
    pub drift: Option<Drift>,
    /// Thermally activated state loss, or `None` for perfect retention.
    pub retention: Option<Retention>,
    /// Read noise, or `None` for a noiseless read.
    pub read_noise: Option<ReadNoise>,
}

impl DeviceModel {
    /// A device with no non-idealities at all.
    ///
    /// A `[0, 1] S` window, a differential mapping and `beta = 1 S` per unit weight, so a weight of
    /// `w` is stored as a conductance of exactly `w` siemens and read back as exactly `w`, bit for
    /// bit. **The zero off-conductance is a fiction** — see [`Window::new`] — and it is what makes
    /// the round trip exact rather than merely close, which is what makes every non-ideal result in
    /// this module attributable to the mechanism that was switched on.
    #[must_use]
    pub fn ideal() -> Self {
        Self {
            window: Window {
                g_off: 0.0,
                g_on: 1.0,
                source: "no device: an exactly-invertible reference window with an infinite on/off \
                         ratio, which nothing fabricated has",
                evidence: Evidence::Unstated,
            },
            mapping: Mapping::Differential,
            beta: 1.0,
            d2d_seed: 0,
            variability: None,
            levels: None,
            stuck: None,
            drift: None,
            retention: None,
            read_noise: None,
        }
    }

    /// Check the window, `beta`, and every enabled sub-model.
    ///
    /// # Errors
    ///
    /// Whatever the offending sub-model returns; see [`Window::validate`],
    /// [`Variability::validate`], [`Drift::validate`], [`Retention::validate`],
    /// [`StuckAt::validate`] and [`ReadNoise::validate`], plus [`DeviceError::NotPositive`] for a
    /// non-positive `beta`.
    pub fn validate(&self) -> Result<(), DeviceError> {
        self.window.validate()?;
        if !self.beta.is_finite() {
            return Err(DeviceError::NonFinite { what: "beta", value: self.beta });
        }
        if !(self.beta > 0.0) {
            return Err(DeviceError::NotPositive { what: "beta", value: self.beta });
        }
        if let Some(v) = &self.variability {
            v.validate()?;
        }
        if let Some(d) = &self.drift {
            d.validate()?;
        }
        if let Some(r) = &self.retention {
            r.validate()?;
        }
        if let Some(s) = &self.stuck {
            s.validate()?;
        }
        if let Some(n) = &self.read_noise {
            n.validate()?;
        }
        Ok(())
    }

    /// Largest representable weight, `Mapping::max_weight` at this window and `beta`.
    #[must_use]
    pub fn max_weight(&self) -> f64 {
        self.mapping.max_weight(&self.window, self.beta)
    }

    /// Smallest representable weight. Zero for [`Mapping::SingleEnded`].
    #[must_use]
    pub fn min_weight(&self) -> f64 {
        self.mapping.min_weight(&self.window, self.beta)
    }

    /// Which non-idealities are switched on, by name, in the order they are applied.
    ///
    /// The list a result should be reported beside. An empty list means [`DeviceModel::ideal`]'s
    /// behaviour and should produce an exact answer.
    #[must_use]
    pub fn enabled(&self) -> Vec<&'static str> {
        let mut v = Vec::new();
        if self.levels.is_some() {
            v.push("finite conductance states");
        }
        if self.variability.is_some() {
            v.push("programming variability");
        }
        if self.stuck.is_some() {
            v.push("stuck-at faults");
        }
        if self.retention.is_some() {
            v.push("retention loss");
        }
        if self.drift.is_some() {
            v.push("conductance drift");
        }
        if self.read_noise.is_some() {
            v.push("read noise");
        }
        v
    }

    /// The weakest evidence grade among the window and every enabled sub-model.
    ///
    /// Same rule as [`crate::hardware::Part::weakest_evidence`]: a figure built from several
    /// sources is only as good as its worst input. [`DeviceModel::ideal`] returns
    /// [`Evidence::Unstated`], which is correct — it describes no device.
    #[must_use]
    pub fn weakest_evidence(&self) -> Evidence {
        let mut g = self.window.evidence;
        if let Some(v) = &self.variability {
            g = weaker(g, v.evidence);
        }
        if let Some(l) = &self.levels {
            g = weaker(g, l.evidence);
        }
        if let Some(s) = &self.stuck {
            g = weaker(g, s.evidence);
        }
        if let Some(d) = &self.drift {
            g = weaker(g, d.evidence);
        }
        if let Some(r) = &self.retention {
            g = weaker(g, r.evidence);
        }
        if let Some(n) = &self.read_noise {
            g = weaker(g, n.evidence);
        }
        g
    }

    /// Program `w` onto this device and return what the hardware would actually hold.
    ///
    /// The order is the physical one: map the weight to a conductance target, snap it to the
    /// nearest writable level, perturb it by the device's frozen offset and the write's fresh
    /// error, clamp it into the window, then let any stuck cell override all of it. Drift and
    /// retention are **not** applied here — they are ageing, not programming, and they live in
    /// [`Programmed::aged`] so the same array can be aged to several times without reprogramming.
    ///
    /// `rng` supplies **only** the cycle-to-cycle term. The device-to-device offsets and the
    /// stuck-at map come from [`DeviceModel::d2d_seed`], from two independently salted per-cell
    /// streams, so changing the variability model does not move the fault map and vice versa.
    ///
    /// The shape of the matrix is not asked for and is not needed: every mechanism here is per
    /// cell. The one that is not — wire resistance — is in [`Crossbar`], and a caller who models
    /// only what is here has modelled everything except **where** each cell sits.
    ///
    /// # Errors
    ///
    /// [`DeviceError::Empty`] for an empty slice, [`DeviceError::NonFiniteWeight`] naming the first
    /// bad element, [`DeviceError::OutOfRange`] naming the first weight this window cannot hold,
    /// plus anything [`DeviceModel::validate`] returns.
    pub fn apply(&self, w: &[f64], rng: &mut Rng) -> Result<Programmed, DeviceError> {
        self.validate()?;
        if w.is_empty() {
            return Err(DeviceError::Empty);
        }
        let lo = self.min_weight();
        let hi = self.max_weight();
        for (i, &v) in w.iter().enumerate() {
            if !v.is_finite() {
                return Err(DeviceError::NonFiniteWeight { index: i, value: v });
            }
            if v < lo || v > hi {
                return Err(DeviceError::OutOfRange { index: i, weight: v, min: lo, max: hi });
            }
        }

        let two = self.mapping.devices_per_weight() == 2;
        let n = w.len();
        let mut g_plus = Vec::with_capacity(n);
        let mut g_minus = Vec::with_capacity(n);
        let mut fault_plus = Vec::with_capacity(n);
        let mut fault_minus = Vec::with_capacity(n);
        let mut clamped = 0usize;

        for (i, &want) in w.iter().enumerate() {
            let (mut gp, mut gm) = self.mapping.program(want, &self.window, self.beta);

            if let Some(levels) = &self.levels {
                gp = levels.snap(gp, &self.window);
                if two {
                    gm = levels.snap(gm, &self.window);
                }
            }

            if let Some(var) = &self.variability {
                let mut d2d = cell_stream(self.d2d_seed, SALT_D2D, i);
                let zd_p = normal(&mut d2d);
                let zd_m = normal(&mut d2d);
                let zc_p = normal(rng);
                let zf_p = normal(rng);
                let zc_m = normal(rng);
                let zf_m = normal(rng);
                let perturb = |g: f64, zd: f64, zc: f64, zf: f64| {
                    g * (1.0 + var.sigma_d2d_rel * zd + var.sigma_c2c_rel * zc)
                        + var.sigma_floor_s * zf
                };
                let raw_p = perturb(gp, zd_p, zc_p, zf_p);
                gp = self.window.clamp(raw_p);
                if gp != raw_p {
                    clamped += 1;
                }
                if two {
                    let raw_m = perturb(gm, zd_m, zc_m, zf_m);
                    gm = self.window.clamp(raw_m);
                    if gm != raw_m {
                        clamped += 1;
                    }
                }
            }

            let (fp, fm) = match &self.stuck {
                Some(model) => {
                    let mut s = cell_stream(self.d2d_seed, SALT_FAULT, i);
                    let a = model.draw(&mut s);
                    let b = model.draw(&mut s);
                    (a, if two { b } else { Fault::Healthy })
                }
                None => (Fault::Healthy, Fault::Healthy),
            };
            gp = self.pin(gp, fp);
            gm = self.pin(gm, fm);

            g_plus.push(gp);
            g_minus.push(gm);
            fault_plus.push(fp);
            fault_minus.push(fm);
        }

        Ok(Programmed {
            model: *self,
            target: w.to_vec(),
            g_plus,
            g_minus,
            fault_plus,
            fault_minus,
            clamped,
            age_s: 0.0,
            aged_at_k: None,
        })
    }

    /// A stuck device's conductance, or `g` unchanged for a healthy one.
    fn pin(&self, g: f64, f: Fault) -> f64 {
        match f {
            Fault::Healthy => g,
            Fault::StuckOn => self.window.g_on,
            Fault::StuckOff => self.window.g_off,
        }
    }
}

/// How far the hardware's weights are from the ones that were asked for.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ErrorStats {
    /// Largest `|held - target|` over the array, in the weights' own units.
    pub max_abs: f64,
    /// Mean **signed** error: the bias.
    ///
    /// Signed on purpose. An unbiased mechanism has a mean of zero however large its spread, and
    /// taking the absolute value first would hide exactly the property that distinguishes
    /// programming noise from a systematic mapping error.
    pub mean_signed: f64,
    /// Root-mean-square error: the magnitude, as opposed to the bias.
    pub rms: f64,
    /// How many devices are stuck at a rail.
    pub stuck_devices: usize,
    /// How many devices were clamped at a window rail during programming.
    ///
    /// Non-zero means [`Variability`]'s unbiasedness no longer holds for those cells: a Gaussian
    /// truncated at a rail has a mean on the other side of the target, and the bias grows as the
    /// target approaches the rail. It is counted rather than suppressed so the caller knows which
    /// regime they are in.
    pub clamped: usize,
}

/// An array as the hardware actually holds it.
///
/// Carries the model that produced it, so ageing and reading need no extra arguments and cannot be
/// done against a different device by accident.
#[derive(Debug, Clone, PartialEq)]
pub struct Programmed {
    /// The device this array was programmed onto.
    pub model: DeviceModel,
    /// The weights that were asked for.
    pub target: Vec<f64>,
    /// Conductance of the positive device of each weight, **siemens**.
    pub g_plus: Vec<f64>,
    /// Conductance of the negative device, **siemens**, or the nominal reference for
    /// [`Mapping::SingleEnded`], in which case it is not a device and never ages or reads noisily.
    pub g_minus: Vec<f64>,
    /// Fault state of each positive device.
    pub fault_plus: Vec<Fault>,
    /// Fault state of each negative device; always [`Fault::Healthy`] for
    /// [`Mapping::SingleEnded`], which has no second device to break.
    pub fault_minus: Vec<Fault>,
    /// How many devices hit a window rail during programming. See [`ErrorStats::clamped`].
    pub clamped: usize,
    /// Seconds since programming, `0.0` at [`DeviceModel::apply`].
    pub age_s: f64,
    /// Temperature the ageing was done at, **kelvin**, or `None` for an array that has not aged.
    pub aged_at_k: Option<f64>,
}

impl Programmed {
    /// How many weights this array holds.
    #[must_use]
    pub fn len(&self) -> usize {
        self.target.len()
    }

    /// Whether the array is empty. Always false for anything [`DeviceModel::apply`] returns, which
    /// refuses an empty slice.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.target.is_empty()
    }

    /// The weights the hardware actually holds, read noiselessly.
    #[must_use]
    pub fn weights(&self) -> Vec<f64> {
        self.g_plus
            .iter()
            .zip(&self.g_minus)
            .map(|(&p, &m)| self.model.mapping.read(p, m, &self.model.window, self.model.beta))
            .collect()
    }

    /// How many devices are stuck at a rail.
    #[must_use]
    pub fn stuck_devices(&self) -> usize {
        self.fault_plus.iter().filter(|f| f.is_stuck()).count()
            + self.fault_minus.iter().filter(|f| f.is_stuck()).count()
    }

    /// The gap between what was asked for and what is held.
    #[must_use]
    pub fn error(&self) -> ErrorStats {
        let held = self.weights();
        let n = held.len().max(1) as f64;
        let mut max_abs = 0.0f64;
        let mut sum = 0.0f64;
        let mut sum_sq = 0.0f64;
        for (&got, &want) in held.iter().zip(&self.target) {
            let e = got - want;
            max_abs = max_abs.max(e.abs());
            sum += e;
            sum_sq += e * e;
        }
        ErrorStats {
            max_abs,
            mean_signed: sum / n,
            rms: (sum_sq / n).sqrt(),
            stuck_devices: self.stuck_devices(),
            clamped: self.clamped,
        }
    }

    /// This array after `t_s` seconds at `temp_k`, with retention loss and drift applied.
    ///
    /// Retention first, then drift: the state relaxes toward [`Window::g_off`], and whatever is
    /// left drifts by the power law. A **stuck device does not age** — it is already pinned to a
    /// rail by a physical defect and this model has nothing to say about how that rail moves.
    /// [`Mapping::SingleEnded`]'s reference does not age either, because it is a number in the
    /// digital domain and not a cell, which is the whole of why that scheme fares worse under
    /// drift.
    ///
    /// Ages from the **programmed** state each time rather than compounding, so
    /// `aged(t)` is a function of `t` alone and two calls with the same `t` give the same array.
    ///
    /// # Errors
    ///
    /// [`DeviceError::Negative`] or [`DeviceError::NonFinite`] for a bad time, and whatever
    /// [`Retention::tau_s`] returns for a bad temperature.
    pub fn aged(&self, t_s: f64, temp_k: f64) -> Result<Self, DeviceError> {
        if !t_s.is_finite() {
            return Err(DeviceError::NonFinite { what: "t_s", value: t_s });
        }
        if t_s < 0.0 {
            return Err(DeviceError::Negative { what: "t_s", value: t_s });
        }
        check_temperature(temp_k)?;
        let win = self.model.window;
        let keep = match &self.model.retention {
            Some(r) => r.remaining_fraction(t_s, temp_k)?,
            None => 1.0,
        };
        let age_one = |g: f64, f: Fault| -> f64 {
            if f.is_stuck() {
                return g;
            }
            let relaxed = win.g_off + (g - win.g_off) * keep;
            match &self.model.drift {
                Some(d) => d.apply(relaxed, &win, t_s),
                None => relaxed,
            }
        };
        let two = self.model.mapping.devices_per_weight() == 2;
        let g_plus = self
            .g_plus
            .iter()
            .zip(&self.fault_plus)
            .map(|(&g, &f)| age_one(g, f))
            .collect();
        let g_minus = if two {
            self.g_minus
                .iter()
                .zip(&self.fault_minus)
                .map(|(&g, &f)| age_one(g, f))
                .collect()
        } else {
            self.g_minus.clone()
        };
        Ok(Self {
            model: self.model,
            target: self.target.clone(),
            g_plus,
            g_minus,
            fault_plus: self.fault_plus.clone(),
            fault_minus: self.fault_minus.clone(),
            clamped: self.clamped,
            age_s: t_s,
            aged_at_k: Some(temp_k),
        })
    }

    /// One noisy read of the whole array at `v_read` volts.
    ///
    /// Each device's conductance gets an independent draw from [`ReadNoise`]; a
    /// [`Mapping::SingleEnded`] reference gets none, because it is not a device. With no read-noise
    /// model this is exactly [`Programmed::weights`] and `rng` is not touched.
    ///
    /// # Errors
    ///
    /// [`DeviceError::NotPositive`] or [`DeviceError::NonFinite`] for a `v_read` no conductance can
    /// be inferred from.
    pub fn read(&self, rng: &mut Rng, v_read: f64) -> Result<Vec<f64>, DeviceError> {
        if !v_read.is_finite() {
            return Err(DeviceError::NonFinite { what: "v_read", value: v_read });
        }
        if !(v_read > 0.0) {
            return Err(DeviceError::NotPositive { what: "v_read", value: v_read });
        }
        let Some(noise) = &self.model.read_noise else {
            return Ok(self.weights());
        };
        let win = self.model.window;
        let two = self.model.mapping.devices_per_weight() == 2;
        let mut out = Vec::with_capacity(self.len());
        for (&p, &m) in self.g_plus.iter().zip(&self.g_minus) {
            let np = noise.sample(rng, p, v_read).unwrap_or(p);
            let nm = if two { noise.sample(rng, m, v_read).unwrap_or(m) } else { m };
            out.push(self.model.mapping.read(np, nm, &win, self.model.beta));
        }
        Ok(out)
    }

    /// The positive (or negative) devices arranged as a [`Crossbar`], so the same array can be run
    /// through the circuit solve.
    ///
    /// `positive` picks which half of a differential pair; for [`Mapping::SingleEnded`] only
    /// `true` addresses real devices and `false` returns the nominal reference column, which is not
    /// a physical array.
    ///
    /// # Errors
    ///
    /// [`DeviceError::BadShape`] when `rows * cols` is not this array's length, plus whatever
    /// [`Crossbar::new`] returns.
    pub fn crossbar(
        &self,
        rows: usize,
        cols: usize,
        positive: bool,
        wires: Wires,
    ) -> Result<Crossbar, DeviceError> {
        let g = if positive { self.g_plus.clone() } else { self.g_minus.clone() };
        Crossbar::new(rows, cols, g, wires)
    }
}

/// A resistive crossbar with wire resistance, solved as a circuit.
///
/// Word lines run left to right and are driven from the **west** edge; bit lines run top to bottom
/// and are held at virtual ground at the **south** edge. Each inter-cell wire segment has
/// [`Wires::r_segment_ohm`], including the segment between the driver and the first cell and the
/// one between the last cell and the sense node.
///
/// # Why this is not a matrix multiply
///
/// With ideal wires the column current is exactly `sum_i g_ij * v_i` and every cell sees the full
/// drive voltage. With real wires the word line loses voltage as it carries current away from the
/// driver, and the bit line gains voltage as it carries current toward the sense node, so a cell in
/// the far corner is driven by **less** than the applied voltage and its contribution is
/// understated. The error therefore depends on the cell's position and on what every other cell in
/// its row and column is doing — it is not a per-weight perturbation and cannot be modelled as one.
/// That is why this is the non-ideality most often left out, and it is the one that scales worst
/// with array size, which is exactly the direction the field wants to push.
#[derive(Debug, Clone, PartialEq)]
pub struct Crossbar {
    rows: usize,
    cols: usize,
    g: Vec<f64>,
    wires: Wires,
}

/// The solved state of a crossbar under one drive vector.
#[derive(Debug, Clone, PartialEq)]
pub struct Solution {
    /// Current out of each bit line into its sense node, **amperes**, one per column.
    pub column_currents: Vec<f64>,
    /// Volts actually across each device, row-major `rows * cols`.
    ///
    /// The diagnostic that makes IR drop visible: with ideal wires every entry equals its row's
    /// drive voltage, and the departure from that is the whole effect.
    pub cell_voltage: Vec<f64>,
    /// Line-relaxation sweeps performed. `0` for the ideal-wire case, which is solved in closed
    /// form.
    pub sweeps: u32,
    /// Worst Kirchhoff current residual left at any node, **amperes**.
    pub residual_a: f64,
}

impl Solution {
    /// Largest `|solved - ideal| / max|ideal|` over the columns, or `None` when the lengths
    /// disagree or the ideal currents are all zero.
    ///
    /// Normalised by the largest ideal column current rather than per column, because a column
    /// whose ideal current is near zero has an unbounded relative error that says nothing about the
    /// array.
    #[must_use]
    pub fn max_relative_error(&self, ideal: &[f64]) -> Option<f64> {
        if ideal.len() != self.column_currents.len() || ideal.is_empty() {
            return None;
        }
        let scale = ideal.iter().fold(0.0f64, |a, b| a.max(b.abs()));
        if !(scale > 0.0) {
            return None;
        }
        Some(
            self.column_currents
                .iter()
                .zip(ideal)
                .fold(0.0f64, |a, (&s, &i)| a.max((s - i).abs()))
                / scale,
        )
    }
}

impl Crossbar {
    /// A crossbar of `rows * cols` conductances in row-major order.
    ///
    /// # Errors
    ///
    /// [`DeviceError::Empty`] for a zero dimension, [`DeviceError::BadShape`] for a length
    /// mismatch, [`DeviceError::NonFinite`] or [`DeviceError::Negative`] for a bad conductance,
    /// plus whatever [`Wires::new`] would return for the wire model.
    pub fn new(
        rows: usize,
        cols: usize,
        g: Vec<f64>,
        wires: Wires,
    ) -> Result<Self, DeviceError> {
        if rows == 0 || cols == 0 {
            return Err(DeviceError::Empty);
        }
        if g.len() != rows * cols {
            return Err(DeviceError::BadShape { rows, cols, len: g.len() });
        }
        for &v in &g {
            if !v.is_finite() {
                return Err(DeviceError::NonFinite { what: "conductance", value: v });
            }
            if v < 0.0 {
                return Err(DeviceError::Negative { what: "conductance", value: v });
            }
        }
        Wires::new(wires.r_segment_ohm, wires.source, wires.evidence)?;
        Ok(Self { rows, cols, g, wires })
    }

    /// Word lines.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Bit lines.
    #[must_use]
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// The conductance at `(row, col)`, **siemens**, or `None` if either index is off the array.
    #[must_use]
    pub fn conductance(&self, row: usize, col: usize) -> Option<f64> {
        if row < self.rows && col < self.cols { Some(self.g[row * self.cols + col]) } else { None }
    }

    /// The column currents an ideal crossbar would deliver: `sum_i g_ij * v_i`, **amperes**.
    ///
    /// The reference every IR-drop figure in this module is measured against, and the exact answer
    /// when [`Wires::r_segment_ohm`] is zero.
    ///
    /// # Errors
    ///
    /// [`DeviceError::BadDrive`] for a drive vector of the wrong length, [`DeviceError::NonFinite`]
    /// for a non-finite voltage.
    pub fn ideal_currents(&self, v_in: &[f64]) -> Result<Vec<f64>, DeviceError> {
        if v_in.len() != self.rows {
            return Err(DeviceError::BadDrive { rows: self.rows, len: v_in.len() });
        }
        for &v in v_in {
            if !v.is_finite() {
                return Err(DeviceError::NonFinite { what: "v_in", value: v });
            }
        }
        let mut out = vec![0.0; self.cols];
        for i in 0..self.rows {
            for j in 0..self.cols {
                out[j] += self.g[i * self.cols + j] * v_in[i];
            }
        }
        Ok(out)
    }

    /// Solve the node voltages and return the currents the array actually delivers.
    ///
    /// The method is **line relaxation**: each word line is solved exactly as a tridiagonal system
    /// given the current bit-line voltages, then each bit line exactly given the word-line
    /// voltages, and the two are alternated. Both systems are diagonally dominant, so the
    /// tridiagonal solve needs no pivoting and the outer iteration converges; solving whole lines
    /// at a time rather than node by node is what keeps the sweep count small on an array whose
    /// wire conductance is much larger than its cell conductance, which is the interesting regime.
    ///
    /// `tol` is **relative to the largest ideal column current**, so `1e-9` means nine digits of
    /// the answer. Convergence is measured as the worst Kirchhoff current residual at any node —
    /// not as the change between sweeps, which can be small while the solution is still far away.
    ///
    /// ⚠ The target is additionally **floored at the arithmetic's own noise**, about
    /// `8 * eps * v_max / r_segment_ohm` amperes. A residual below that is not a property of the
    /// solution: the residual is a difference of wire currents of size `v / r_segment_ohm` that
    /// cancel down to a cell current many orders of magnitude smaller, and no number of sweeps
    /// recovers digits that cancellation threw away. Asking for `tol = 1e-15` on a low-resistance
    /// array therefore returns at the floor rather than looping to the sweep budget, and
    /// [`Solution::residual_a`] says which bound was reached.
    ///
    /// # Errors
    ///
    /// [`DeviceError::BadDrive`] or [`DeviceError::NonFinite`] as [`Crossbar::ideal_currents`],
    /// [`DeviceError::NotPositive`] for a non-positive `tol` or a zero sweep budget,
    /// [`DeviceError::NotConverged`] when the budget runs out, and [`DeviceError::SingularLine`] if
    /// a tridiagonal pivot degenerates.
    pub fn solve(
        &self,
        v_in: &[f64],
        tol: f64,
        max_sweeps: u32,
    ) -> Result<Solution, DeviceError> {
        let ideal = self.ideal_currents(v_in)?;
        if !tol.is_finite() || !(tol > 0.0) {
            return Err(DeviceError::NotPositive { what: "tol", value: tol });
        }
        if max_sweeps == 0 {
            return Err(DeviceError::NotPositive { what: "max_sweeps", value: 0.0 });
        }
        let (n, m) = (self.rows, self.cols);

        if self.wires.r_segment_ohm == 0.0 {
            // Ideal wires: every cell sees its row's drive, every bit line is at ground, and the
            // column current is the dot product exactly. Solved here rather than approached by
            // iteration because the iteration divides by the wire resistance.
            let mut cell_voltage = vec![0.0; n * m];
            for i in 0..n {
                for j in 0..m {
                    cell_voltage[i * m + j] = v_in[i];
                }
            }
            return Ok(Solution {
                column_currents: ideal,
                cell_voltage,
                sweeps: 0,
                residual_a: 0.0,
            });
        }

        let gw = 1.0 / self.wires.r_segment_ohm;
        let scale = ideal.iter().fold(0.0f64, |a, b| a.max(b.abs()));
        // A Kirchhoff residual is a difference of wire currents of magnitude `gw * v`, and those
        // cancel down to the cell current, so the residual cannot be driven below the rounding
        // noise of that cancellation however many sweeps are spent on it. Floor the target there.
        // A stiff array — small `r_segment_ohm`, so large `gw` — has a HIGHER floor, and
        // `Solution::residual_a` is reported so a caller can see which bound they stopped on.
        let v_max = v_in.iter().fold(0.0f64, |a, b| a.max(b.abs()));
        let noise_floor = 8.0 * f64::EPSILON * gw * v_max;
        let target = (tol * scale).max(noise_floor).max(f64::MIN_POSITIVE);

        // Word-line node voltages `a` and bit-line node voltages `b`, both row-major.
        let mut a = vec![0.0f64; n * m];
        let mut b = vec![0.0f64; n * m];
        for i in 0..n {
            for j in 0..m {
                a[i * m + j] = v_in[i];
            }
        }

        let line = n.max(m);
        let mut sub = vec![0.0f64; line];
        let mut diag = vec![0.0f64; line];
        let mut sup = vec![0.0f64; line];
        let mut rhs = vec![0.0f64; line];
        let mut sol = vec![0.0f64; line];
        let mut scratch = vec![0.0f64; line];

        let mut residual = f64::INFINITY;
        for sweep in 1..=max_sweeps {
            // Word lines: driven from the west by v_in[i], open at the east end.
            for i in 0..n {
                for j in 0..m {
                    let g = self.g[i * m + j];
                    sub[j] = if j > 0 { -gw } else { 0.0 };
                    sup[j] = if j + 1 < m { -gw } else { 0.0 };
                    diag[j] = gw + if j + 1 < m { gw } else { 0.0 } + g;
                    rhs[j] = g * b[i * m + j] + if j == 0 { gw * v_in[i] } else { 0.0 };
                }
                if !thomas(&sub[..m], &diag[..m], &sup[..m], &rhs[..m], &mut sol[..m], &mut scratch[..m])
                {
                    return Err(DeviceError::SingularLine);
                }
                for j in 0..m {
                    a[i * m + j] = sol[j];
                }
            }
            // Bit lines: open at the north end, grounded through one segment at the south end.
            for j in 0..m {
                for i in 0..n {
                    let g = self.g[i * m + j];
                    sub[i] = if i > 0 { -gw } else { 0.0 };
                    sup[i] = if i + 1 < n { -gw } else { 0.0 };
                    diag[i] = gw + if i > 0 { gw } else { 0.0 } + g;
                    rhs[i] = g * a[i * m + j];
                }
                if !thomas(&sub[..n], &diag[..n], &sup[..n], &rhs[..n], &mut sol[..n], &mut scratch[..n])
                {
                    return Err(DeviceError::SingularLine);
                }
                for i in 0..n {
                    b[i * m + j] = sol[i];
                }
            }

            residual = self.residual(&a, &b, v_in, gw);
            if residual <= target {
                let mut column_currents = vec![0.0; m];
                let mut cell_voltage = vec![0.0; n * m];
                for j in 0..m {
                    column_currents[j] = gw * b[(n - 1) * m + j];
                }
                for k in 0..n * m {
                    cell_voltage[k] = a[k] - b[k];
                }
                return Ok(Solution {
                    column_currents,
                    cell_voltage,
                    sweeps: sweep,
                    residual_a: residual,
                });
            }
        }
        Err(DeviceError::NotConverged {
            sweeps: max_sweeps,
            residual_a: residual,
            target_a: target,
        })
    }

    /// Worst Kirchhoff current residual over every node, **amperes**.
    fn residual(&self, a: &[f64], b: &[f64], v_in: &[f64], gw: f64) -> f64 {
        let (n, m) = (self.rows, self.cols);
        let mut worst = 0.0f64;
        for i in 0..n {
            for j in 0..m {
                let k = i * m + j;
                let g = self.g[k];
                let west = if j == 0 { v_in[i] } else { a[k - 1] };
                let mut r = gw * (west - a[k]) - g * (a[k] - b[k]);
                if j + 1 < m {
                    r += gw * (a[k + 1] - a[k]);
                }
                worst = worst.max(r.abs());

                let south = if i + 1 < n { b[k + m] } else { 0.0 };
                let mut s = gw * (south - b[k]) + g * (a[k] - b[k]);
                if i > 0 {
                    s += gw * (b[k - m] - b[k]);
                }
                worst = worst.max(s.abs());
            }
        }
        worst
    }
}

/// Solve a tridiagonal system by the Thomas algorithm, without pivoting.
///
/// `sub[0]` and `sup[n-1]` are ignored. Returns false on a zero or non-finite pivot, which cannot
/// happen for the diagonally dominant systems this module builds but is checked rather than
/// allowed to produce infinities.
fn thomas(
    sub: &[f64],
    diag: &[f64],
    sup: &[f64],
    rhs: &[f64],
    out: &mut [f64],
    c: &mut [f64],
) -> bool {
    let n = diag.len();
    if n == 0 {
        return false;
    }
    if !diag[0].is_finite() || diag[0] == 0.0 {
        return false;
    }
    c[0] = sup[0] / diag[0];
    out[0] = rhs[0] / diag[0];
    for k in 1..n {
        let m = diag[k] - sub[k] * c[k - 1];
        if !m.is_finite() || m == 0.0 {
            return false;
        }
        c[k] = sup[k] / m;
        out[k] = (rhs[k] - sub[k] * out[k - 1]) / m;
    }
    for k in (0..n - 1).rev() {
        out[k] -= c[k] * out[k + 1];
    }
    out[..n].iter().all(|v| v.is_finite())
}

#[cfg(test)]
mod tests {
    use super::{
        BOLTZMANN_EV_PER_K, BOLTZMANN_J_PER_K, Crossbar, DeviceError, DeviceModel, Drift,
        ErrorStats, Evidence, Fault, Levels, Mapping, PCM_DRIFT_AMORPHOUS,
        RRAM_STUCK_AT_PLACEHOLDER, RRAM_VARIABILITY_PLACEHOLDER, ReadNoise, Retention, StuckAt,
        THERMAL_READ_NOISE_300K, Variability, Window, Wires, normal,
    };
    use crate::rng::Rng;

    /// A plain 1 uS to 100 uS window for tests that need a realistic one.
    fn win() -> Window {
        Window::new(1e-6, 100e-6, "test fixture, not a device", Evidence::Unstated).unwrap()
    }

    // ---------------------------------------------------------------- the identity case

    /// (a) THE test that makes every other result in this module attributable. Every non-ideality
    /// off, and the round trip is exact — `assert_eq!` on `f64`, not a tolerance. If this ever has
    /// to be loosened, nothing else in the module means anything, because a drift result and a
    /// plumbing bug would be indistinguishable.
    #[test]
    fn an_ideal_device_reproduces_the_matrix_exactly() {
        let model = DeviceModel::ideal();
        assert!(model.enabled().is_empty(), "ideal() has a non-ideality switched on");
        let want: Vec<f64> = vec![0.0, 1.0, -1.0, 0.25, -0.75, 1.0 / 3.0, -2.0 / 7.0, 0.1, -0.9];
        let mut rng = Rng::new(1);
        let held = model.apply(&want, &mut rng).unwrap();
        let got = held.weights();
        for (k, (&g, &w)) in got.iter().zip(&want).enumerate() {
            assert_eq!(g, w, "weight {k} came back as {g} instead of {w}");
        }
        let e = held.error();
        assert_eq!(e.max_abs, 0.0);
        assert_eq!(e.rms, 0.0);
        assert_eq!(e.mean_signed, 0.0);
        assert_eq!(e.stuck_devices, 0);
        assert_eq!(e.clamped, 0);
    }

    /// The honest counterpart: a window with a REAL off-conductance does not round-trip exactly,
    /// because subtracting a large offset from a slightly larger one throws away bits. The error is
    /// tiny and it is not zero, and both halves are asserted — the second is what would catch a
    /// future "optimisation" that special-cased the ideal path and quietly made this exact too.
    #[test]
    fn a_realistic_window_costs_bits_on_the_round_trip() {
        let mut model = DeviceModel::ideal();
        model.window = win();
        model.beta = 1e-6;
        let want: Vec<f64> = (0..64).map(|k| f64::from(k) * 0.37 - 11.0).collect();
        let mut rng = Rng::new(2);
        let held = model.apply(&want, &mut rng).unwrap();
        let got = held.weights();
        let mut inexact = 0;
        for (&g, &w) in got.iter().zip(&want) {
            let rel = (g - w).abs() / w.abs().max(1.0);
            assert!(rel < 1e-13, "round trip lost {rel} relative, which is not rounding");
            if g != w {
                inexact += 1;
            }
        }
        assert!(
            inexact > 0,
            "every weight round-tripped exactly through a 1 uS offset, which f64 cannot do: the \
             mapping is not being exercised"
        );
    }

    /// All three schemes round-trip exactly through the exact window, they agree about weight
    /// range, and they differ by a factor of `1 + ratio` in the common-mode current a zero weight
    /// costs. The range assertion is here because this module's first draft claimed the balanced
    /// scheme cost half the range; it does not, and the doc now says where the cost actually is.
    #[test]
    fn every_mapping_round_trips_and_the_schemes_differ_in_common_mode() {
        let w = win();
        let beta = 1e-6;
        let full = w.span() / beta;
        for m in [Mapping::SingleEnded, Mapping::Differential, Mapping::BalancedDifferential] {
            assert_eq!(m.max_weight(&w, beta), full, "{m:?} range");
        }
        assert_eq!(Mapping::SingleEnded.min_weight(&w, beta), 0.0);
        assert_eq!(Mapping::Differential.min_weight(&w, beta), -full);
        assert_eq!(Mapping::BalancedDifferential.min_weight(&w, beta), -full);
        assert_eq!(Mapping::SingleEnded.devices_per_weight(), 1);
        assert_eq!(Mapping::Differential.devices_per_weight(), 2);
        assert_eq!(Mapping::BalancedDifferential.devices_per_weight(), 2);

        // The trade, as a number: a 100:1 window makes the balanced pair's idle current 50x the
        // one-at-minimum pair's, which is (1 + ratio) / 2.
        let cm_d = Mapping::Differential.common_mode_conductance(&w);
        let cm_b = Mapping::BalancedDifferential.common_mode_conductance(&w);
        assert_eq!(cm_d, 2.0 * w.g_off);
        assert_eq!(cm_b, w.g_off + w.g_on);
        let ratio = w.on_off_ratio().unwrap();
        assert!((cm_b / cm_d - 0.5 * (1.0 + ratio)).abs() < 1e-9, "{cm_b} / {cm_d}");
        assert_eq!(Mapping::SingleEnded.common_mode_conductance(&w), w.g_off);

        for m in [Mapping::SingleEnded, Mapping::Differential, Mapping::BalancedDifferential] {
            let mut model = DeviceModel::ideal();
            model.mapping = m;
            // Binary fractions, so the balanced scheme's halving is exact too, and non-negative,
            // because the single-ended scheme has no negative half.
            let want: Vec<f64> = vec![0.0, 1.0, 0.5, 0.25, 0.125];
            let mut rng = Rng::new(3);
            let held = model.apply(&want, &mut rng).unwrap();
            for (k, (&g, &t)) in held.weights().iter().zip(&want).enumerate() {
                assert_eq!(g, t, "{m:?} weight {k}: {g} instead of {t}");
            }
            // Full scale puts exactly one device on each rail in every scheme.
            assert_eq!(held.g_plus[1], model.window.g_on);
        }
    }

    /// A weight the window cannot hold is REFUSED by index, not clamped. Clamping would be an
    /// accuracy loss of unbounded size reported as zero error.
    #[test]
    fn a_weight_the_window_cannot_hold_is_refused_by_index() {
        let mut model = DeviceModel::ideal();
        model.window = win();
        model.beta = 1e-6;
        let limit = model.max_weight();
        let err = model.apply(&[0.0, 1.0, limit * 1.0001], &mut Rng::new(4)).unwrap_err();
        match err {
            DeviceError::OutOfRange { index, max, .. } => {
                assert_eq!(index, 2);
                assert_eq!(max, limit);
            }
            other => panic!("expected OutOfRange, got {other}"),
        }
        // Exactly at the limit is fine: the bound is inclusive.
        assert!(model.apply(&[limit, -limit], &mut Rng::new(4)).is_ok());
        // Single-ended refuses a negative weight through the same path.
        model.mapping = Mapping::SingleEnded;
        assert!(matches!(
            model.apply(&[-1.0], &mut Rng::new(4)),
            Err(DeviceError::OutOfRange { index: 0, .. })
        ));
    }

    /// The on/off ratio controls the signal fraction, and refuses for the ideal fiction rather than
    /// reporting an infinity.
    #[test]
    fn the_on_off_ratio_sets_the_signal_fraction_and_refuses_when_there_is_none() {
        let w = Window::new(1e-6, 2e-6, "ratio of 2", Evidence::Unstated).unwrap();
        assert_eq!(w.on_off_ratio(), Some(2.0));
        assert!((w.signal_fraction() - 0.5).abs() < 1e-15);
        let w = Window::new(1e-6, 100e-6, "ratio of 100", Evidence::Unstated).unwrap();
        assert!((w.signal_fraction() - 0.99).abs() < 1e-15);
        assert_eq!(DeviceModel::ideal().window.on_off_ratio(), None);
        assert!(Window::new(-1.0, 1.0, "", Evidence::Unstated).is_err());
        assert!(Window::new(1.0, 1.0, "", Evidence::Unstated).is_err());
        assert!(Window::new(2.0, 1.0, "", Evidence::Unstated).is_err());
    }

    /// The audit's clamp defect, in this module's terms: `f64::clamp` PANICS when its bounds cross,
    /// and `Window`'s fields are public, so a hand-built crossed window must not abort a sweep.
    #[test]
    fn a_crossed_window_clamps_without_panicking_and_is_refused_at_the_boundary() {
        let bad = Window { g_off: 5.0, g_on: 1.0, source: "crossed", evidence: Evidence::Unstated };
        assert_eq!(bad.clamp(3.0), 5.0);
        assert_eq!(bad.clamp(0.0), 5.0);
        assert!(bad.validate().is_err());
        let mut model = DeviceModel::ideal();
        model.window = bad;
        assert!(matches!(model.apply(&[0.0], &mut Rng::new(5)), Err(DeviceError::BadWindow { .. })));
    }

    // ---------------------------------------------------------------- variability

    /// The Box-Muller generator against the Gaussian's own quantiles. Mean and variance alone would
    /// pass for a scaled uniform, which has neither of these tail fractions.
    #[test]
    fn the_normal_generator_matches_the_gaussian_quantiles() {
        let mut r = Rng::new(1234);
        let n = 200_000;
        let (mut s, mut s2, mut within1, mut within2) = (0.0f64, 0.0f64, 0u32, 0u32);
        for _ in 0..n {
            let z = normal(&mut r);
            s += z;
            s2 += z * z;
            if z.abs() <= 1.0 {
                within1 += 1;
            }
            if z.abs() <= 2.0 {
                within2 += 1;
            }
        }
        let mean = s / f64::from(n);
        let var = s2 / f64::from(n) - mean * mean;
        assert!(mean.abs() < 0.01, "mean {mean}");
        assert!((var - 1.0).abs() < 0.02, "variance {var}");
        // erf(1/sqrt(2)) and erf(sqrt(2)): 0.6826894921 and 0.9544997361.
        let p1 = f64::from(within1) / f64::from(n);
        let p2 = f64::from(within2) / f64::from(n);
        assert!((p1 - 0.6826894921).abs() < 0.005, "P(|z|<1) = {p1}");
        assert!((p2 - 0.9544997361).abs() < 0.003, "P(|z|<2) = {p2}");
    }

    /// (b) Both halves asserted: the mean converges to the programmed value AND the spread is the
    /// stated sigma. The mean alone would pass a model with no variability at all; the sigma alone
    /// would pass one that is biased by a full sigma.
    #[test]
    fn device_to_device_variability_is_unbiased_and_has_the_stated_sigma() {
        let sigma = 0.10;
        let mut model = DeviceModel::ideal();
        model.window = win();
        model.beta = 1e-6;
        model.mapping = Mapping::SingleEnded;
        model.variability = Some(
            Variability::new(sigma, 0.0, 0.0, "test", Evidence::Unstated).unwrap(),
        );
        // A target mid-window, far from both rails, so nothing clamps and the mean is untruncated.
        let target = 50.0; // 50 uS of excess over the 1 uS floor -> 51 uS, mid-window.
        let n = 40_000;
        let w = vec![target; n];
        let held = model.apply(&w, &mut Rng::new(99)).unwrap();
        assert_eq!(held.clamped, 0, "the rails truncated the distribution");

        let g_target = model.window.g_off + target * model.beta;
        let mean = held.g_plus.iter().sum::<f64>() / n as f64;
        let var = held.g_plus.iter().map(|g| (g - mean) * (g - mean)).sum::<f64>() / n as f64;
        let sd = var.sqrt();

        // Standard error of the mean is sigma/sqrt(n) = 0.1/200 = 5e-4 relative; 2e-3 is 4 of them.
        let bias = (mean - g_target).abs() / g_target;
        assert!(bias < 2e-3, "device-to-device spread is biased by {bias} relative");
        // Standard error of the sd estimate is 1/sqrt(2n) = 0.35% relative; 2% is ~5.7 of them.
        let got = sd / g_target;
        assert!((got / sigma - 1.0).abs() < 0.02, "sigma came out {got}, asked for {sigma}");
    }

    /// The honest counterpart to the test above, and the reason `clamped` is counted: a target
    /// close to a rail has its Gaussian truncated, so the mean moves AWAY from the rail. Asserting
    /// the direction as well as the size is what makes this a check rather than a note.
    #[test]
    fn clamping_at_a_rail_biases_the_mean_away_from_it() {
        let mut model = DeviceModel::ideal();
        model.window = win();
        model.beta = 1e-6;
        model.mapping = Mapping::SingleEnded;
        model.variability =
            Some(Variability::new(0.30, 0.0, 0.0, "test", Evidence::Unstated).unwrap());
        let target = model.max_weight(); // right at g_on
        let n = 20_000;
        let held = model.apply(&vec![target; n], &mut Rng::new(101)).unwrap();
        assert!(held.clamped > n / 4, "only {} of {n} clamped", held.clamped);
        let g_target = model.window.g_on;
        let mean = held.g_plus.iter().sum::<f64>() / n as f64;
        assert!(mean < g_target, "mean {mean} did not fall below the rail at {g_target}");
        // Half a Gaussian truncated at the mean sits about 0.4 sigma below it.
        let shift = (g_target - mean) / (0.30 * g_target);
        assert!((0.2..0.6).contains(&shift), "the truncation shifted the mean by {shift} sigma");
    }

    /// The distinction the whole two-stream design exists for. Device-to-device spread is frozen to
    /// the array; cycle-to-cycle spread is not. A model that drew both from the caller's generator
    /// would fail the first assertion; one that drew both from the seed would fail the second.
    #[test]
    fn device_to_device_is_frozen_to_the_array_and_cycle_to_cycle_is_not() {
        let mut model = DeviceModel::ideal();
        model.window = win();
        model.beta = 1e-6;
        model.d2d_seed = 0xBEEF;
        let w: Vec<f64> = (0..200).map(|k| f64::from(k) * 0.1).collect();

        model.variability = Some(Variability::new(0.1, 0.0, 0.0, "d2d only", Evidence::Unstated).unwrap());
        let a = model.apply(&w, &mut Rng::new(1)).unwrap();
        let b = model.apply(&w, &mut Rng::new(999_999)).unwrap();
        assert_eq!(a.g_plus, b.g_plus, "device-to-device spread moved with the caller's generator");
        assert_ne!(a.g_plus, w.iter().map(|x| x * model.beta).collect::<Vec<_>>());

        model.variability = Some(Variability::new(0.0, 0.1, 0.0, "c2c only", Evidence::Unstated).unwrap());
        let a = model.apply(&w, &mut Rng::new(1)).unwrap();
        let b = model.apply(&w, &mut Rng::new(999_999)).unwrap();
        assert_ne!(a.g_plus, b.g_plus, "cycle-to-cycle spread did not move with the generator");

        // And a different array is a different array.
        model.variability = Some(Variability::new(0.1, 0.0, 0.0, "d2d only", Evidence::Unstated).unwrap());
        let c = { let mut m = model; m.d2d_seed = 0xF00D; m.apply(&w, &mut Rng::new(1)).unwrap() };
        assert_ne!(a.g_plus, c.g_plus, "two different seeds gave the same array");
    }

    // ---------------------------------------------------------------- stuck-at faults

    /// (e) The rate, within a binomial band, plus the two exact endpoints and the on/off split.
    /// The band is 4 standard deviations wide, which at this n is 0.15% absolute: tight enough to
    /// catch a rate that is wrong by a tenth and loose enough never to flake.
    #[test]
    fn stuck_at_faults_occur_at_the_requested_rate() {
        let mut model = DeviceModel::ideal();
        model.mapping = Mapping::SingleEnded;
        let n = 200_000usize;
        let w = vec![0.5; n];

        let rate = 0.03;
        model.stuck = Some(StuckAt::new(rate, 0.25, "test", Evidence::Unstated).unwrap());
        let held = model.apply(&w, &mut Rng::new(7)).unwrap();
        let stuck = held.stuck_devices();
        let expect = rate * n as f64;
        let sd = (n as f64 * rate * (1.0 - rate)).sqrt();
        assert!(
            (stuck as f64 - expect).abs() < 4.0 * sd,
            "{stuck} stuck of {n}, expected {expect} +/- {sd}"
        );
        let on = held.fault_plus.iter().filter(|f| matches!(f, Fault::StuckOn)).count();
        let split = on as f64 / stuck as f64;
        let sd_split = (0.25 * 0.75 / stuck as f64).sqrt();
        assert!(
            (split - 0.25).abs() < 4.0 * sd_split,
            "stuck-on share {split}, asked for 0.25 +/- {sd_split}"
        );

        // The endpoints are exact, not statistical.
        model.stuck = Some(StuckAt::new(0.0, 0.5, "never", Evidence::Unstated).unwrap());
        assert_eq!(model.apply(&w[..1000], &mut Rng::new(8)).unwrap().stuck_devices(), 0);
        model.stuck = Some(StuckAt::new(1.0, 1.0, "always on", Evidence::Unstated).unwrap());
        let all = model.apply(&w[..1000], &mut Rng::new(8)).unwrap();
        assert_eq!(all.stuck_devices(), 1000);
        assert!(all.g_plus.iter().all(|&g| g == model.window.g_on));

        assert!(StuckAt::new(1.5, 0.5, "", Evidence::Unstated).is_err());
        assert!(StuckAt::new(0.5, -0.1, "", Evidence::Unstated).is_err());
    }

    /// A stuck device ignores what was written to it, at every target, and stays stuck through
    /// ageing. Without this, a fault could be "modelled" as a small perturbation that a later
    /// program call quietly repaired.
    #[test]
    fn a_stuck_device_ignores_the_target_and_stays_stuck_through_ageing() {
        let mut model = DeviceModel::ideal();
        model.window = win();
        model.beta = 1e-6;
        model.mapping = Mapping::SingleEnded;
        model.stuck = Some(StuckAt::new(1.0, 0.0, "all stuck off", Evidence::Unstated).unwrap());
        model.drift = Some(PCM_DRIFT_AMORPHOUS);
        model.retention =
            Some(Retention::new(0.5, 1e-9, "fast, for the test", Evidence::Unstated).unwrap());
        let w: Vec<f64> = (0..50).map(f64::from).collect();
        let held = model.apply(&w, &mut Rng::new(11)).unwrap();
        assert!(held.g_plus.iter().all(|&g| g == model.window.g_off));
        let old = held.aged(1e6, 400.0).unwrap();
        assert_eq!(old.g_plus, held.g_plus, "a stuck device aged");
    }

    /// The two per-cell streams are salted apart, so switching the variability model on does not
    /// move the fault map. Without separate salts, every defect study would silently be a different
    /// array for each noise setting it swept.
    #[test]
    fn the_fault_map_does_not_move_when_the_variability_model_changes() {
        let mut model = DeviceModel::ideal();
        model.window = win();
        model.beta = 1e-6;
        model.d2d_seed = 42;
        model.stuck = Some(StuckAt::new(0.2, 0.5, "test", Evidence::Unstated).unwrap());
        let w = vec![10.0; 2_000];
        let bare = model.apply(&w, &mut Rng::new(3)).unwrap();
        model.variability = Some(RRAM_VARIABILITY_PLACEHOLDER);
        let noisy = model.apply(&w, &mut Rng::new(3)).unwrap();
        assert_eq!(bare.fault_plus, noisy.fault_plus, "the fault map moved");
        assert_eq!(bare.fault_minus, noisy.fault_minus);
        assert!(bare.stuck_devices() > 0, "no faults were drawn at a 20% rate");
        // And the conductances DID move, so the test above is not comparing two identical runs.
        assert_ne!(bare.g_plus, noisy.g_plus);
    }

    /// (M26) The fixed stream consumption [`StuckAt::draw`] documents, checked on the generator
    /// itself over seeds that produce both outcomes. Nothing in the fault map's VALUES can see
    /// this — a variable consumption re-rolls the rest of the array without biasing it — so the
    /// only place it is visible is the generator's state.
    #[test]
    fn the_fault_draw_consumes_a_fixed_number_of_uniforms() {
        let s = StuckAt::new(0.5, 0.5, "half", Evidence::Unstated).unwrap();
        let mut saw_stuck = 0;
        let mut saw_healthy = 0;
        for seed in 0..64u64 {
            let mut got = Rng::new(seed);
            let f = s.draw(&mut got);
            if f.is_stuck() {
                saw_stuck += 1;
            } else {
                saw_healthy += 1;
            }
            let mut want = Rng::new(seed);
            let _ = want.next_f64();
            let _ = want.next_f64();
            assert_eq!(got, want, "a {f:?} draw did not consume exactly two uniforms");
        }
        assert!(saw_stuck > 8 && saw_healthy > 8, "{saw_stuck} stuck, {saw_healthy} healthy");
        assert!(!Fault::Healthy.is_stuck());
        assert!(Fault::StuckOn.is_stuck() && Fault::StuckOff.is_stuck());
    }

    /// (M27) BOTH devices of a pair are written to the level grid, not just the one carrying the
    /// weight's sign. The negative halves are the ones that go untested by accident: in the
    /// one-at-minimum scheme the idle device sits exactly on level zero, so a missing snap is
    /// invisible until a weight goes negative, and in the balanced scheme it is off-grid always.
    #[test]
    fn both_devices_of_a_pair_are_written_to_the_level_grid() {
        let w = win();
        let levels = Levels::new(23, "a deliberately awkward count", Evidence::Unstated).unwrap();
        let step = levels.step(&w);
        let mut model = DeviceModel::ideal();
        model.window = w;
        model.beta = 1e-6;
        model.levels = Some(levels);
        let targets: Vec<f64> = (0..40).map(|k| f64::from(k) * 2.3 - 46.0).collect();
        for m in [Mapping::Differential, Mapping::BalancedDifferential] {
            model.mapping = m;
            let held = model.apply(&targets, &mut Rng::new(1)).unwrap();
            let mut off_grid_minus = 0;
            for (k, (&gp, &gm)) in held.g_plus.iter().zip(&held.g_minus).enumerate() {
                for (label, g) in [("plus", gp), ("minus", gm)] {
                    let idx = ((g - w.g_off) / step).round();
                    assert!(
                        (g - (w.g_off + idx * step)).abs() < 1e-18,
                        "{m:?} cell {k} {label} at {g} S is not on the 23-level grid"
                    );
                }
                // The raw target for the minus device, before snapping, was NOT on the grid for
                // most cells: without that this test would pass on an unsnapped array.
                let (_, raw_m) = m.program(targets[k], &w, model.beta);
                if (raw_m - (w.g_off + ((raw_m - w.g_off) / step).round() * step)).abs() > 1e-15 {
                    off_grid_minus += 1;
                }
            }
            assert!(off_grid_minus > 10, "{m:?}: only {off_grid_minus} minus devices needed a snap");
        }
    }

    /// (M28) Every enabled sub-model is in the [`DeviceModel::weakest_evidence`] fold, by exact
    /// census: start from a model graded `Metered` throughout, downgrade exactly ONE of the seven,
    /// and the answer has to move. A test that switched several `Projected` models on at once
    /// would pass with any one of them omitted from the fold.
    #[test]
    fn the_weakest_evidence_is_an_exact_census_of_every_enabled_model() {
        let m = Evidence::Metered;
        let base = || {
            let mut d = DeviceModel::ideal();
            d.window = Window::new(1e-6, 100e-6, "test", m).unwrap();
            d.beta = 1e-6;
            d.levels = Some(Levels::new(32, "test", m).unwrap());
            d.variability = Some(Variability::new(0.01, 0.01, 0.0, "test", m).unwrap());
            d.stuck = Some(StuckAt::new(0.01, 0.5, "test", m).unwrap());
            d.drift = Some(Drift::uniform(0.01, 1.0, "test", m).unwrap());
            d.retention = Some(Retention::new(1.0, 1e-12, "test", m).unwrap());
            d.read_noise = Some(ReadNoise::new(300.0, 1e6, 0.0, "test", m).unwrap());
            d
        };
        assert_eq!(base().weakest_evidence(), Evidence::Metered);
        assert_eq!(base().enabled().len(), 6, "a sub-model is missing from enabled()");
        assert_eq!(DeviceModel::ideal().weakest_evidence(), Evidence::Unstated);
        assert!(DeviceModel::ideal().enabled().is_empty());

        /// One downgrade in the census: a name and the mutation that weakens exactly that model.
        type Downgrade = (&'static str, fn(&mut DeviceModel));
        let census: [Downgrade; 7] = [
            ("window", |d| d.window.evidence = Evidence::Unstated),
            ("levels", |d| d.levels.as_mut().unwrap().evidence = Evidence::Unstated),
            ("variability", |d| d.variability.as_mut().unwrap().evidence = Evidence::Unstated),
            ("stuck", |d| d.stuck.as_mut().unwrap().evidence = Evidence::Unstated),
            ("drift", |d| d.drift.as_mut().unwrap().evidence = Evidence::Unstated),
            ("retention", |d| d.retention.as_mut().unwrap().evidence = Evidence::Unstated),
            ("read_noise", |d| d.read_noise.as_mut().unwrap().evidence = Evidence::Unstated),
        ];
        for (name, downgrade) in census {
            let mut d = base();
            downgrade(&mut d);
            assert_eq!(
                d.weakest_evidence(),
                Evidence::Unstated,
                "{name} is not in the weakest-evidence fold"
            );
        }
    }

    // ---------------------------------------------------------------- drift and retention

    /// (c) The power law against its own log-log slope, which is a different operation from the
    /// `powf` that produced it. A model that returned a constant, or an exponential, or the wrong
    /// sign, fails this.
    #[test]
    fn drift_follows_the_power_law_and_its_log_log_slope_is_the_exponent() {
        let w = win();
        let nu = 0.08;
        let d = Drift::uniform(nu, 1.0, "test", Evidence::Unstated).unwrap();
        let g0 = 50e-6;
        // At and before the reference time the model says nothing and returns the input.
        assert_eq!(d.apply(g0, &w, 1.0), g0);
        assert_eq!(d.apply(g0, &w, 0.0), g0);
        for &(t1, t2) in &[(10.0, 100.0), (100.0, 1e4), (1e3, 1e7), (2.0, 3.0)] {
            let g1 = d.apply(g0, &w, t1);
            let g2 = d.apply(g0, &w, t2);
            let slope = (g1 / g2).ln() / (t2 / t1).ln();
            assert!((slope - nu).abs() < 1e-12, "log-log slope {slope} over [{t1}, {t2}]");
            assert!(g2 < g1 && g1 < g0, "drift did not decrease conductance");
        }
        // A zero exponent is exactly no drift, at any time.
        let none = Drift::uniform(0.0, 1.0, "test", Evidence::Unstated).unwrap();
        assert_eq!(none.apply(g0, &w, 1e12), g0);
        assert!(Drift::new(-0.1, 0.0, 1.0, "", Evidence::Unstated).is_err());
        assert!(Drift::new(0.1, 0.1, 0.0, "", Evidence::Unstated).is_err());
    }

    /// (c) The closed form that decides which mapping to build, checked on both sides. Uniform
    /// drift multiplies a differential pair's weight by exactly the drift factor; a single-ended
    /// read of the same cell picks up an additive offset of `g_off*(f-1)/beta` that no gain
    /// correction removes. Two DIFFERENT closed forms, so a stub cannot satisfy both.
    #[test]
    fn uniform_drift_cancels_in_a_differential_pair_and_not_in_a_single_ended_read() {
        let w = win();
        let nu = 0.06;
        let t = 1e5;
        let f = (t / 1.0f64).powf(-nu);
        let beta = 1e-6;
        let targets = [5.0, 20.0, 40.0];

        for mapping in [Mapping::Differential, Mapping::BalancedDifferential] {
            let mut model = DeviceModel::ideal();
            model.window = w;
            model.beta = beta;
            model.mapping = mapping;
            model.drift = Some(Drift::uniform(nu, 1.0, "test", Evidence::Unstated).unwrap());
            let held = model.apply(&targets, &mut Rng::new(1)).unwrap();
            for (&got, &want) in held.aged(t, 300.0).unwrap().weights().iter().zip(&targets) {
                let closed = f * want;
                assert!(
                    (got - closed).abs() / closed < 1e-12,
                    "{mapping:?}: drifted weight {got}, closed form {closed}"
                );
            }
        }

        let mut model = DeviceModel::ideal();
        model.window = w;
        model.beta = beta;
        model.mapping = Mapping::SingleEnded;
        model.drift = Some(Drift::uniform(nu, 1.0, "test", Evidence::Unstated).unwrap());
        let held = model.apply(&targets, &mut Rng::new(1)).unwrap();
        for (&got, &want) in held.aged(t, 300.0).unwrap().weights().iter().zip(&targets) {
            let closed = f * want + w.g_off * (f - 1.0) / beta;
            assert!(
                (got - closed).abs() / closed.abs() < 1e-10,
                "single-ended drifted weight {got}, closed form {closed}"
            );
            // And the offset is real: the single-ended read is further off than a pure gain.
            assert!((got - f * want).abs() > 0.0);
        }

        // The two halves of that asymmetry, pinned directly rather than left to `apply` happening
        // to store `g_off` beside the cell. FIRST: a single-ended read IGNORES its second argument,
        // so it cannot accidentally start cancelling drift if anything ever writes a real device
        // there. A read that used the stored value would agree with this one today and diverge
        // silently the moment the reference column became physical.
        let a = Mapping::SingleEnded.read(30e-6, w.g_off, &w, beta);
        let b = Mapping::SingleEnded.read(30e-6, 77e-6, &w, beta);
        assert_eq!(a, b, "the single-ended read used its second argument");
        assert_eq!(a, (30e-6 - w.g_off) / beta);
        assert_ne!(
            Mapping::Differential.read(30e-6, w.g_off, &w, beta),
            Mapping::Differential.read(30e-6, 77e-6, &w, beta),
            "a differential read ignored its second device"
        );
        // SECOND: the reference does not age, and is exactly the window's nominal floor.
        let aged = held.aged(t, 300.0).unwrap();
        assert!(held.g_minus.iter().all(|&g| g == w.g_off));
        assert_eq!(aged.g_minus, held.g_minus, "the nominal reference aged");
        assert_ne!(aged.g_plus, held.g_plus, "the cell did not age");
    }

    /// The limit of the paragraph above, asserted so nobody quotes it too far. With a
    /// state-dependent exponent the two devices of a pair drift by different factors, so the
    /// cancellation is partial and the weight moves by more than the uniform closed form predicts.
    #[test]
    fn state_dependent_drift_does_not_cancel() {
        let w = win();
        let t = 1e6;
        let beta = 1e-6;
        let mut model = DeviceModel::ideal();
        model.window = w;
        model.beta = beta;
        model.mapping = Mapping::BalancedDifferential;
        // A large exponent spread: 0.15 at the off rail, 0.0 at the on rail.
        model.drift = Some(Drift::new(0.15, 0.0, 1.0, "test", Evidence::Unstated).unwrap());
        let target = 30.0;
        let held = model.apply(&[target], &mut Rng::new(1)).unwrap();
        let aged = held.aged(t, 300.0).unwrap();
        let got = aged.weights()[0];

        // The uniform closed form using EITHER exponent alone.
        let f_off = (t / 1.0f64).powf(-0.15);
        let f_on: f64 = 1.0;
        assert!(
            got < f_on * target && got > f_off * target,
            "a state-dependent pair drifted to {got}, outside the two uniform bounds \
             [{}, {}]",
            f_off * target,
            f_on * target
        );
        // And with the SAME exponent at both rails, the cancellation is exact again.
        model.drift = Some(Drift::uniform(0.15, 1.0, "test", Evidence::Unstated).unwrap());
        let uniform = model.apply(&[target], &mut Rng::new(1)).unwrap().aged(t, 300.0).unwrap();
        let u = uniform.weights()[0];
        assert!((u - f_off * target).abs() / target < 1e-12, "uniform pair gave {u}");
        assert!((u - got).abs() / target > 1e-3, "the two exponent models agreed, so neither bites");
    }

    /// (M19 in this module's mutation audit) Retention and drift do NOT commute, and the order is
    /// a documented decision, so it is pinned by a closed form rather than by the direction of an
    /// inequality — `aged` making the error bigger is true under either order.
    ///
    /// The check has to be **single-ended**, and that is the interesting part: for a differential
    /// pair both operations are affine in the conductance and the common `g_off` cancels, so the
    /// two orders give the identical weight and nothing can distinguish them. A single-ended read
    /// keeps the `g_off` term, and the two orders then differ by `(1 - keep) * g_off * (f - 1)`.
    #[test]
    fn retention_and_drift_compose_in_the_documented_order() {
        let w = win();
        let beta = 1e-6;
        let nu = 0.07;
        let t = 5e4;
        let temp = 340.0;
        // Chosen so that `keep` lands near one half at this time and temperature: a keep near 0
        // or near 1 makes the two orders agree and the test vacuous, which is asserted below.
        let ea = 0.5;
        let tau0 = 2.8e-3;

        // Retention alone scales a DIFFERENTIAL weight by exactly `keep`, because the excess over
        // g_off is what relaxes and the pair's common g_off subtracts out.
        let ret = Retention::new(ea, tau0, "test", Evidence::Unstated).unwrap();
        let keep = ret.remaining_fraction(t, temp).unwrap();
        assert!((0.01..0.99).contains(&keep), "keep = {keep} is too close to a trivial case");
        let mut model = DeviceModel::ideal();
        model.window = w;
        model.beta = beta;
        model.mapping = Mapping::Differential;
        model.retention = Some(ret);
        let targets = [8.0, 33.0];
        let held = model.apply(&targets, &mut Rng::new(1)).unwrap();
        for (&got, &want) in held.aged(t, temp).unwrap().weights().iter().zip(&targets) {
            let closed = keep * want;
            assert!((got - closed).abs() / closed < 1e-12, "{got} against {closed}");
        }

        // Single-ended, retention THEN drift: `f * keep * w + g_off * (f - 1) / beta`.
        // Reversing the order gives `keep * f * w + keep * g_off * (f - 1) / beta`, which differs
        // by `(1 - keep) * g_off * (f - 1) / beta` and is what this assertion refuses.
        model.mapping = Mapping::SingleEnded;
        model.drift = Some(Drift::uniform(nu, 1.0, "test", Evidence::Unstated).unwrap());
        let f = (t / 1.0f64).powf(-nu);
        let held = model.apply(&targets, &mut Rng::new(1)).unwrap();
        let reversed_gap = (1.0 - keep) * w.g_off * (f - 1.0) / beta;
        assert!(reversed_gap.abs() > 1e-3, "the two orders are indistinguishable here");
        for (&got, &want) in held.aged(t, temp).unwrap().weights().iter().zip(&targets) {
            let closed = f * keep * want + w.g_off * (f - 1.0) / beta;
            assert!(
                (got - closed).abs() < 1e-9 * want,
                "single-ended aged to {got}, retention-then-drift says {closed},                  drift-then-retention would say {}",
                closed - reversed_gap
            );
        }
    }

    /// (M20 in this module's mutation audit) The stuck-at map and the device-to-device offsets are
    /// drawn from differently salted streams, and the reason is INDEPENDENCE, not reproducibility:
    /// sharing a stream would make a cell's defect a deterministic function of the same uniforms
    /// that set its programming offset, so the surviving healthy cells would carry a biased spread
    /// and every fault-tolerance study run on them would be wrong in a way that looks fine.
    ///
    /// Conditioning on `Fault::Healthy` must leave the spread at its stated sigma. With a shared
    /// stream the Box-Muller radius is `sqrt(-2 ln(1 - u))` for the same `u` that decides the
    /// fault, so healthy cells — the ones with large `u` — get a radius bounded away from zero and
    /// their spread inflates by about 30% at a rate of one half.
    #[test]
    fn the_fault_map_is_independent_of_the_programming_offsets() {
        let sigma = 0.10;
        let mut model = DeviceModel::ideal();
        model.window = win();
        model.beta = 1e-6;
        model.mapping = Mapping::SingleEnded;
        model.d2d_seed = 77;
        model.variability =
            Some(Variability::new(sigma, 0.0, 0.0, "test", Evidence::Unstated).unwrap());
        model.stuck = Some(StuckAt::new(0.5, 1.0, "half the array", Evidence::Unstated).unwrap());

        let target = 50.0; // mid-window, ten sigma from either rail
        let n = 40_000;
        let held = model.apply(&vec![target; n], &mut Rng::new(4)).unwrap();
        assert_eq!(held.clamped, 0, "a rail truncated the spread and confounded this test");
        let stuck = held.stuck_devices();
        assert!(
            (stuck as f64 - 0.5 * n as f64).abs() < 4.0 * (n as f64 * 0.25).sqrt(),
            "{stuck} stuck of {n} at a rate of one half"
        );

        let g_target = model.window.g_off + target * model.beta;
        let healthy: Vec<f64> = held
            .g_plus
            .iter()
            .zip(&held.fault_plus)
            .filter(|(_, f)| !f.is_stuck())
            .map(|(&g, _)| g / g_target - 1.0)
            .collect();
        assert!(healthy.len() > n / 4, "only {} cells survived", healthy.len());
        let m = healthy.len() as f64;
        let mean = healthy.iter().sum::<f64>() / m;
        let sd = (healthy.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / m).sqrt();
        assert!(
            (sd / sigma - 1.0).abs() < 0.03,
            "the surviving cells' spread is {sd}, not {sigma}: the fault draw and the offset draw              share a stream"
        );
        assert!(mean.abs() < 0.01, "the surviving cells' offsets are biased by {mean}");
    }

    /// Arrhenius, checked by its own defining property: `ln(tau)` is linear in `1/T` with slope
    /// `Ea/k_B`. Recovering the slope numerically and comparing it to the constant is independent
    /// of the `exp` that produced it.
    #[test]
    fn retention_is_arrhenius_and_the_slope_recovers_the_activation_energy() {
        for &ea in &[0.3, 0.8, 1.2] {
            let r = Retention::new(ea, 1e-12, "test", Evidence::Unstated).unwrap();
            let (t1, t2) = (350.0, 420.0);
            let slope = (r.tau_s(t1).unwrap() / r.tau_s(t2).unwrap()).ln()
                / (1.0 / t1 - 1.0 / t2);
            let want = ea / BOLTZMANN_EV_PER_K;
            assert!((slope / want - 1.0).abs() < 1e-12, "slope {slope} K, wanted {want} K");
            // One over e at one time constant, exactly.
            let tau = r.tau_s(t1).unwrap();
            let frac = r.remaining_fraction(tau, t1).unwrap();
            assert!((frac - (-1.0f64).exp()).abs() < 1e-12, "remaining at tau was {frac}");
            assert_eq!(r.remaining_fraction(0.0, t1).unwrap(), 1.0);
        }
        // Zero activation energy is a temperature-independent lifetime, exactly tau0.
        let flat = Retention::new(0.0, 7.0, "test", Evidence::Unstated).unwrap();
        assert_eq!(flat.tau_s(300.0).unwrap(), 7.0);
        assert_eq!(flat.tau_s(900.0).unwrap(), 7.0);
        assert_eq!(flat.acceleration(300.0, 900.0).unwrap(), 1.0);
    }

    /// The acceleration factor composes and is one at equal temperatures — the two properties any
    /// correct Arrhenius ratio has and a plausible-looking wrong one usually does not. Plus the
    /// worked example from the doc, as a literal.
    #[test]
    fn retention_acceleration_composes_and_is_one_at_equal_temperatures() {
        let r = Retention::new(1.0, 1e-12, "test", Evidence::Unstated).unwrap();
        assert_eq!(r.acceleration(360.0, 360.0).unwrap(), 1.0);
        let (a, b, c) = (330.0, 360.0, 400.0);
        let composed = r.acceleration(a, b).unwrap() * r.acceleration(b, c).unwrap();
        let direct = r.acceleration(a, c).unwrap();
        assert!((composed / direct - 1.0).abs() < 1e-12, "{composed} vs {direct}");
        // 1.0 eV from 85 C to 125 C: exp(11604.518 * (1/358.15 - 1/398.15)) = 25.9.
        let af = r.acceleration(358.15, 398.15).unwrap();
        assert!((af - 25.93).abs() < 0.05, "85 C to 125 C at 1 eV gave {af}, expected about 25.9");
        assert!(r.tau_s(0.0).is_err());
        assert!(r.tau_s(f64::NAN).is_err());
        assert!(r.remaining_fraction(-1.0, 300.0).is_err());
    }

    // ---------------------------------------------------------------- levels and read noise

    /// (f) Half a step, and the bound is REACHED — a `snap` that did nothing would satisfy the
    /// upper bound alone. The third assertion pins the output onto the grid, which an off-by-one in
    /// the level count would break while still respecting both error bounds.
    #[test]
    fn quantisation_error_is_bounded_by_half_a_step_and_reaches_it() {
        let w = win();
        for n in [2u32, 3, 16, 64, 256] {
            let l = Levels::new(n, "test", Evidence::Unstated).unwrap();
            let step = l.step(&w);
            assert!((l.bits() - f64::from(n).log2()).abs() < 1e-15);
            let half = 0.5 * step;
            let mut worst = 0.0f64;
            let mut r = Rng::new(u64::from(n) + 17);
            for _ in 0..5_000 {
                let g = w.g_off + r.next_f64() * w.span();
                let s = l.snap(g, &w);
                worst = worst.max((s - g).abs());
                // On the grid, exactly.
                let k = ((s - w.g_off) / step).round();
                assert!(
                    (s - (w.g_off + k * step)).abs() < 1e-18 && (0.0..f64::from(n)).contains(&k),
                    "{s} is not level {k} of {n}"
                );
            }
            assert!(worst <= half * (1.0 + 1e-12), "worst error {worst} exceeded half a step {half}");
            assert!(worst > 0.45 * step, "worst error {worst} never approached half a step {half}");
            // Both rails snap to themselves.
            assert_eq!(l.snap(w.g_off, &w), w.g_off);
            assert!((l.snap(w.g_on, &w) - w.g_on).abs() < 1e-18);
        }
        assert!(Levels::new(1, "", Evidence::Unstated).is_err());
        assert!(Levels::new(0, "", Evidence::Unstated).is_err());
    }

    /// Johnson-Nyquist against a hand-computed number, then against a sampled standard deviation.
    /// The literal is the anchor: a missing factor of 4, a wrong Boltzmann constant or a forgotten
    /// division by the read voltage all move it, and none of them move a self-consistent formula.
    #[test]
    fn thermal_read_noise_matches_johnson_nyquist_at_a_hand_computed_point() {
        let n = THERMAL_READ_NOISE_300K;
        assert_eq!(n.temp_k, 300.0);
        assert_eq!(n.bandwidth_hz, 1e6);
        assert_eq!(n.evidence, Evidence::Derived);
        let g = 1e-6;
        let v = 0.2;
        // sqrt(4 * 1.380649e-23 * 300 * 1e-6 * 1e6) = 1.287159e-10 A.
        let si = n.sigma_current(g, v).unwrap();
        assert!((si - 1.287159e-10).abs() / 1.287159e-10 < 1e-5, "sigma_I = {si}");
        // ... divided by 0.2 V.
        let sg = n.sigma_conductance(g, v).unwrap();
        assert!((sg - 6.435795e-10).abs() / 6.435795e-10 < 1e-5, "sigma_G = {sg}");
        assert!((sg * v - si).abs() < 1e-24);
        // The formula is the square root of a power, so doubling the bandwidth multiplies by
        // sqrt(2) and not by 2.
        let wide = ReadNoise::new(300.0, 2e6, 0.0, "test", Evidence::Derived).unwrap();
        let ratio = wide.sigma_current(g, v).unwrap() / si;
        assert!((ratio - 2.0f64.sqrt()).abs() < 1e-12, "bandwidth scaling {ratio}");
        assert!(
            (4.0 * BOLTZMANN_J_PER_K * 300.0 * g * 1e6).sqrt() - si < 1e-24,
            "the formula moved away from 4kTGB"
        );

        // And the sampler actually delivers that spread.
        let mut r = Rng::new(555);
        let (mut s, mut s2) = (0.0f64, 0.0f64);
        let m = 100_000;
        for _ in 0..m {
            let x = n.sample(&mut r, g, v).unwrap();
            s += x - g;
            s2 += (x - g) * (x - g);
        }
        let mean = s / f64::from(m);
        let sd = (s2 / f64::from(m)).sqrt();
        assert!(mean.abs() < 0.02 * sg, "read noise is biased by {mean} S");
        assert!((sd / sg - 1.0).abs() < 0.02, "sampled sigma {sd} against {sg}");
        assert_eq!(n.sigma_conductance(g, 0.0), None);
        assert_eq!(n.sigma_conductance(g, -1.0), None);
        assert_eq!(n.sigma_current(-1.0, v), None);
    }

    /// The level count the noise actually leaves, and the trade that sets it. Refuses rather than
    /// reporting an unbounded count for a noiseless model.
    #[test]
    fn distinguishable_levels_fall_as_the_read_gets_faster() {
        let w = win();
        let slow = ReadNoise::new(300.0, 1e3, 0.0, "test", Evidence::Derived).unwrap();
        let fast = ReadNoise::new(300.0, 1e9, 0.0, "test", Evidence::Derived).unwrap();
        let a = slow.distinguishable_levels(&w, 0.2, 6.0).unwrap();
        let b = fast.distinguishable_levels(&w, 0.2, 6.0).unwrap();
        assert!(a > b, "a 1 kHz read ({a} levels) was not better than a 1 GHz read ({b})");
        // Noise power scales with bandwidth, so amplitude scales with its square root: a factor of
        // 1e6 in bandwidth is 1e3 in sigma and therefore about 1e3 in levels.
        let ratio = f64::from(a - 1) / f64::from(b - 1);
        assert!((ratio / 1e3 - 1.0).abs() < 0.05, "level ratio {ratio}, expected about 1000");
        // A demand for more separation gives fewer levels, proportionally.
        let strict = slow.distinguishable_levels(&w, 0.2, 12.0).unwrap();
        assert!((f64::from(a - 1) / f64::from(strict - 1) - 2.0).abs() < 0.02);

        // (M31) THE FENCEPOST, pinned through a different function. `n` levels have `n - 1` gaps,
        // so the answer is `span / gap + 1` and not `span / gap` — and the ratio checks above are
        // blind to that `+ 1`, because dropping it shifts both sides. The defining property is
        // that the returned count is the LARGEST whose [`Levels`] spacing still clears the demanded
        // separation, so the next count up must not.
        let coarse = ReadNoise::new(300.0, 1.0, 0.02, "2% read noise", Evidence::Unstated).unwrap();
        let sep = 6.0;
        let sigma = coarse.sigma_conductance(w.g_on, 0.2).unwrap();
        assert!((sigma - 2e-6).abs() / 2e-6 < 1e-6, "sigma came out {sigma}, expected 2 uS");
        let n = coarse.distinguishable_levels(&w, 0.2, sep).unwrap();
        // 99 uS of span, 12 uS between levels: 8.25 gaps, so 9 posts. About 3.2 bits, on a window
        // a programmer would happily ask for 64 levels in.
        assert_eq!(n, 9, "a 2% read noise over a 99 uS span left {n} levels at six sigma");
        let fits = Levels::new(n, "test", Evidence::Unstated).unwrap();
        let too_many = Levels::new(n + 1, "test", Evidence::Unstated).unwrap();
        assert!(fits.step(&w) >= sep * sigma, "{n} levels do not fit");
        let up = n + 1;
        assert!(too_many.step(&w) < sep * sigma, "{up} levels also fit, so {n} is not maximal");
        assert!((fits.bits() - 3.17).abs() < 0.01, "{} bits", fits.bits());
        // A bandwidth so small the noise power underflows to zero has NO answer: the level count
        // is not large, it is unbounded, and this module will not print an infinity as an integer.
        let quiet = ReadNoise { sigma_rel: 0.0, bandwidth_hz: 1e-300, ..slow };
        assert_eq!(quiet.sigma_conductance(w.g_on, 0.2), Some(0.0));
        assert_eq!(quiet.distinguishable_levels(&w, 0.2, 6.0), None);
        // And the other end: noise larger than the whole window leaves ONE level, which is to say
        // the cell carries no analog information at all. Reported as 1, not as 0 and not as None.
        let deafening = ReadNoise::new(300.0, 1e30, 0.0, "test", Evidence::Derived).unwrap();
        assert_eq!(deafening.distinguishable_levels(&w, 0.2, 6.0), Some(1));
        assert_eq!(slow.distinguishable_levels(&w, 0.0, 6.0), None);
        assert_eq!(slow.distinguishable_levels(&w, 0.2, 0.0), None);
        let bad = Window { g_off: 5.0, g_on: 1.0, source: "crossed", evidence: Evidence::Unstated };
        assert_eq!(slow.distinguishable_levels(&bad, 0.2, 6.0), None);
    }

    /// A read with no noise model is the noiseless read, exactly, and does not touch the generator.
    /// A read with one is unbiased over many reads.
    #[test]
    fn a_read_is_exact_without_a_noise_model_and_unbiased_with_one() {
        let mut model = DeviceModel::ideal();
        model.window = win();
        model.beta = 1e-6;
        let w = vec![10.0, -20.0, 0.0, 40.0];
        let held = model.apply(&w, &mut Rng::new(1)).unwrap();
        let mut rng = Rng::new(2);
        let before = rng;
        assert_eq!(held.read(&mut rng, 0.2).unwrap(), held.weights());
        assert_eq!(rng, before, "a noiseless read consumed randomness");
        assert!(held.read(&mut rng, 0.0).is_err());
        assert!(held.read(&mut rng, f64::NAN).is_err());

        let mut noisy = model;
        noisy.read_noise = Some(ReadNoise::new(300.0, 1e9, 0.02, "test", Evidence::Unstated).unwrap());
        let held = noisy.apply(&w, &mut Rng::new(1)).unwrap();
        let exact = held.weights();
        let mut rng = Rng::new(3);
        let trials = 4_000;
        let mut acc = vec![0.0; w.len()];
        let mut spread = vec![0.0f64; w.len()];
        for _ in 0..trials {
            let r = held.read(&mut rng, 0.2).unwrap();
            for k in 0..w.len() {
                acc[k] += r[k] - exact[k];
                spread[k] += (r[k] - exact[k]).abs();
            }
        }
        for k in 0..w.len() {
            let mean = acc[k] / f64::from(trials);
            let typical = spread[k] / f64::from(trials);
            assert!(typical > 0.0, "weight {k} read back with no noise at all");
            assert!(
                mean.abs() < 0.1 * typical,
                "weight {k} read biased by {mean} against a typical {typical}"
            );
        }
    }

    /// (M25 in this module's mutation audit) A noisy read touches exactly the devices that exist,
    /// and the spread it produces is the closed-form combination of their sigmas: one device for
    /// [`Mapping::SingleEnded`], two variances added for a pair.
    ///
    /// The window here has an on/off ratio of 2.5 on purpose. With a wide window the nominal
    /// reference sits so far below the cell that noising it would move the answer by a fraction of
    /// a percent and hide inside any reasonable tolerance; at a ratio of 2.5 it inflates the spread
    /// by nearly thirty percent, which is where a reference column that is not a device but is
    /// modelled as one becomes visible.
    #[test]
    fn a_noisy_read_has_the_closed_form_spread_of_the_devices_it_actually_touches() {
        let narrow = Window::new(40e-6, 100e-6, "a 2.5:1 window", Evidence::Unstated).unwrap();
        let beta = 1e-6;
        let v = 0.2;
        let noise = ReadNoise::new(300.0, 1e9, 0.01, "test", Evidence::Unstated).unwrap();
        let trials = 40_000;

        let spread = |model: &DeviceModel, target: f64| -> (f64, f64) {
            let held = model.apply(&[target], &mut Rng::new(1)).unwrap();
            let exact = held.weights()[0];
            let mut rng = Rng::new(21);
            let (mut sum, mut sq) = (0.0f64, 0.0f64);
            for _ in 0..trials {
                let e = held.read(&mut rng, v).unwrap()[0] - exact;
                sum += e;
                sq += e * e;
            }
            let mean = sum / f64::from(trials);
            let sd = (sq / f64::from(trials) - mean * mean).sqrt();
            let sp = noise.sigma_conductance(held.g_plus[0], v).unwrap();
            let sm = noise.sigma_conductance(held.g_minus[0], v).unwrap();
            let closed = match model.mapping {
                Mapping::SingleEnded => sp / beta,
                _ => (sp * sp + sm * sm).sqrt() / beta,
            };
            (sd, closed)
        };

        let mut model = DeviceModel::ideal();
        model.window = narrow;
        model.beta = beta;
        model.read_noise = Some(noise);

        model.mapping = Mapping::SingleEnded;
        let (sd, closed) = spread(&model, 10.0);
        assert!(
            (sd / closed - 1.0).abs() < 0.03,
            "single-ended read spread {sd}, one device's closed form {closed}"
        );
        // If the nominal reference were noised too, the variances would add and this would be the
        // answer instead. It is a quarter larger, which is why the assertion above bites.
        let sm = noise.sigma_conductance(narrow.g_off, v).unwrap() / beta;
        assert!((closed * closed + sm * sm).sqrt() / closed > 1.2, "the two cases are too close");

        for m in [Mapping::Differential, Mapping::BalancedDifferential] {
            model.mapping = m;
            let (sd, closed) = spread(&model, 10.0);
            assert!(
                (sd / closed - 1.0).abs() < 0.03,
                "{m:?} read spread {sd}, two devices' closed form {closed}"
            );
        }

        // And the count of draws, on the generator's own state. A stray sample of the nominal
        // reference would be INVISIBLE in the answer — `Mapping::SingleEnded::read` discards its
        // second argument, so the noise on it goes nowhere — and would show up only as every later
        // draw in the caller's program shifting by one. That is the worst kind of defect to leave
        // untested, so it is tested here rather than through the answer.
        let weights = [5.0, 8.0, 11.0];
        for (m, draws) in [
            (Mapping::SingleEnded, 3),
            (Mapping::Differential, 6),
            (Mapping::BalancedDifferential, 6),
        ] {
            model.mapping = m;
            let held = model.apply(&weights, &mut Rng::new(1)).unwrap();
            let mut got = Rng::new(9);
            let _ = held.read(&mut got, v).unwrap();
            let mut want = Rng::new(9);
            for _ in 0..draws {
                let _ = normal(&mut want);
            }
            assert_eq!(got, want, "{m:?} did not draw exactly {draws} times for 3 weights");
        }
    }

    /// (M21 in this module's mutation audit) `mean_signed` is the bias and `rms` is the magnitude,
    /// and they are different numbers. Two `f64` fields of the same type in one struct literal are
    /// exactly the pair that can be swapped and never noticed, so the two regimes where they
    /// SEPARATE are pinned: unbiased noise makes the first vanish while the second does not, and a
    /// uniform systematic shift makes them equal in magnitude and opposite in what they say.
    #[test]
    fn the_error_statistics_separate_the_bias_from_the_magnitude() {
        let mut model = DeviceModel::ideal();
        model.window = win();
        model.beta = 1e-6;
        model.mapping = Mapping::SingleEnded;

        // Unbiased: cycle-to-cycle programming noise, mid-window so nothing clamps.
        model.variability =
            Some(Variability::new(0.0, 0.10, 0.0, "c2c only", Evidence::Unstated).unwrap());
        let held = model.apply(&vec![50.0; 20_000], &mut Rng::new(31)).unwrap();
        assert_eq!(held.clamped, 0);
        let e = held.error();
        assert!(e.rms > 1.0, "rms {} is too small to separate from a bias", e.rms);
        assert!(
            e.mean_signed.abs() < 0.05 * e.rms,
            "unbiased noise reported a bias of {} against an rms of {}",
            e.mean_signed,
            e.rms
        );
        assert!(e.rms <= e.max_abs, "rms {} exceeded the maximum {}", e.rms, e.max_abs);

        // Systematic and identical for every cell: drift on a uniform array. The error is the same
        // number everywhere, so rms equals |mean_signed| exactly, and the SIGN is the whole point.
        model.variability = None;
        model.drift = Some(Drift::uniform(0.05, 1.0, "test", Evidence::Unstated).unwrap());
        let held = model.apply(&vec![50.0; 100], &mut Rng::new(31)).unwrap();
        let e = held.aged(1e6, 300.0).unwrap().error();
        assert!(e.mean_signed < 0.0, "drift reported a POSITIVE bias of {}", e.mean_signed);
        assert!(
            (e.rms - e.mean_signed.abs()).abs() < 1e-9,
            "a uniform error has rms {} and |bias| {}; they must agree",
            e.rms,
            e.mean_signed.abs()
        );
        // Jensen, in every case: the magnitude is never below the bias, up to the last place —
        // the two are computed by different reductions and agree only to rounding when the errors
        // are identical, which is exactly the case constructed here.
        assert!(e.rms >= e.mean_signed.abs() * (1.0 - 1e-12));
    }

    // ---------------------------------------------------------------- the crossbar

    /// (d) part one: with ideal wires the crossbar IS the dot product, bit for bit, and every cell
    /// sees its row's full drive. This is the crossbar's identity case.
    #[test]
    fn a_crossbar_with_zero_wire_resistance_is_the_exact_dot_product() {
        let (n, m) = (4, 3);
        let g: Vec<f64> = (0..n * m).map(|k| (k as f64 + 1.0) * 1e-6).collect();
        let xb = Crossbar::new(n, m, g.clone(), Wires::ideal()).unwrap();
        let v = vec![0.2, 0.1, -0.05, 0.3];
        let ideal = xb.ideal_currents(&v).unwrap();
        let s = xb.solve(&v, 1e-12, 10).unwrap();
        assert_eq!(s.sweeps, 0);
        assert_eq!(s.residual_a, 0.0);
        for j in 0..m {
            let mut want = 0.0;
            for i in 0..n {
                want += g[i * m + j] * v[i];
            }
            assert_eq!(s.column_currents[j], want);
            assert_eq!(ideal[j], want);
        }
        for i in 0..n {
            for j in 0..m {
                assert_eq!(s.cell_voltage[i * m + j], v[i]);
            }
        }
        assert_eq!(s.max_relative_error(&ideal), Some(0.0));
        assert_eq!(xb.conductance(0, 0), Some(1e-6));
        assert_eq!(xb.conductance(n, 0), None);
    }

    /// A single cell between two wire segments is a series circuit anybody can solve on paper:
    /// `I = V / (2*r_w + 1/g)`. The solver has to reproduce it.
    #[test]
    fn one_cell_with_wire_resistance_matches_the_two_resistor_closed_form() {
        for &(g, r) in &[(1e-4, 1.0), (1e-6, 50.0), (1e-3, 0.5), (1e-5, 1e3)] {
            let xb = Crossbar::new(
                1,
                1,
                vec![g],
                Wires::new(r, "test", Evidence::Unstated).unwrap(),
            )
            .unwrap();
            let v = 0.3;
            let s = xb.solve(&[v], 1e-12, 10_000).unwrap();
            let closed = v / (2.0 * r + 1.0 / g);
            let got = s.column_currents[0];
            assert!(
                (got - closed).abs() / closed < 1e-9,
                "g = {g} S, r = {r} ohm: solver {got} A against closed form {closed} A"
            );
            // And the cell sees less than the applied voltage, by exactly the two IR drops.
            assert!((s.cell_voltage[0] - (v - 2.0 * closed * r)).abs() / v < 1e-9);
        }
    }

    /// A one-row array is a resistive ladder, and a ladder has an exact recursion that owes nothing
    /// to the node solver. Agreeing with it to nine digits is a real check on the circuit model,
    /// not on the iteration.
    #[test]
    fn a_single_row_matches_an_independent_ladder_recursion() {
        let m = 12;
        let r = 2.5;
        let g: Vec<f64> = (0..m).map(|j| 1e-5 * (1.0 + 0.3 * j as f64)).collect();
        let xb =
            Crossbar::new(1, m, g.clone(), Wires::new(r, "test", Evidence::Unstated).unwrap())
                .unwrap();
        let v = 0.25;
        let s = xb.solve(&[v], 1e-10, 100_000).unwrap();

        // Ladder: each cell is 1/g_j in series with the bit line's one segment to ground.
        let r_cell: Vec<f64> = g.iter().map(|&x| 1.0 / x + r).collect();
        let mut z = vec![0.0; m];
        z[m - 1] = r_cell[m - 1];
        for j in (0..m - 1).rev() {
            let rest = r + z[j + 1];
            z[j] = 1.0 / (1.0 / r_cell[j] + 1.0 / rest);
        }
        let mut va = vec![0.0; m];
        va[0] = v * z[0] / (r + z[0]);
        for j in 0..m - 1 {
            va[j + 1] = va[j] * z[j + 1] / (r + z[j + 1]);
        }
        for j in 0..m {
            let want = va[j] / r_cell[j];
            let got = s.column_currents[j];
            assert!(
                (got - want).abs() / want < 1e-9,
                "column {j}: solver {got} A, ladder {want} A"
            );
        }
    }

    /// (d) IR drop is monotone with distance from the driver, in both directions, and STRICTLY so —
    /// a stub returning the ideal answer everywhere would fail on the strictness alone.
    #[test]
    fn ir_drop_is_monotone_with_distance_from_the_driver() {
        let r = Wires::new(3.0, "test", Evidence::Unstated).unwrap();
        let m = 10;
        // One row: the word line loses voltage as it runs east, so cells further east see less.
        let xb = Crossbar::new(1, m, vec![5e-5; m], r).unwrap();
        let s = xb.solve(&[0.3], 1e-10, 100_000).unwrap();
        for j in 1..m {
            assert!(
                s.cell_voltage[j] < s.cell_voltage[j - 1],
                "cell {j} saw {} V, cell {} saw {}",
                s.cell_voltage[j],
                j - 1,
                s.cell_voltage[j - 1]
            );
            assert!(s.column_currents[j] < s.column_currents[j - 1]);
        }
        // One column: the bit line rises above ground as it carries current south, so cells
        // further NORTH — further from the sense node — see less.
        let n = 10;
        let xb = Crossbar::new(n, 1, vec![5e-5; n], r).unwrap();
        let s = xb.solve(&vec![0.3; n], 1e-10, 100_000).unwrap();
        for i in 1..n {
            assert!(
                s.cell_voltage[i] > s.cell_voltage[i - 1],
                "cell {i} saw {} V, cell {} saw {}",
                s.cell_voltage[i],
                i - 1,
                s.cell_voltage[i - 1]
            );
        }
        // Every cell is driven by LESS than the applied voltage. Never more.
        assert!(s.cell_voltage.iter().all(|&x| x < 0.3 && x > 0.0));
    }

    /// (d) part two, and the assertion that cannot be satisfied by a stub: the IR-drop error has to
    /// VANISH as the wire resistance goes to zero, and to vanish at FIRST ORDER. Comparing the
    /// error at three resistances a decade apart pins the exponent, so a model with the wrong
    /// topology — or one that ignored the wires and returned the ideal answer — fails here even
    /// though it would pass the `r == 0` case above.
    #[test]
    fn the_ir_drop_error_vanishes_at_first_order_in_the_wire_resistance() {
        let (n, m) = (6, 6);
        let g: Vec<f64> = (0..n * m).map(|k| 2e-5 + 1e-6 * (k % 7) as f64).collect();
        let v: Vec<f64> = (0..n).map(|i| 0.1 + 0.02 * i as f64).collect();
        let mut errs = Vec::new();
        for &r in &[1e-2, 1e-1, 1.0] {
            let xb = Crossbar::new(
                n,
                m,
                g.clone(),
                Wires::new(r, "test", Evidence::Unstated).unwrap(),
            )
            .unwrap();
            let ideal = xb.ideal_currents(&v).unwrap();
            let s = xb.solve(&v, 1e-12, 200_000).unwrap();
            errs.push(s.max_relative_error(&ideal).unwrap());
        }
        assert!(errs[0] > 0.0, "a 0.01 ohm wire produced no error at all");
        for k in 1..errs.len() {
            let ratio = errs[k] / errs[k - 1];
            assert!(
                (ratio / 10.0 - 1.0).abs() < 0.05,
                "error grew by {ratio} for a tenfold wire resistance, not by 10: errors {errs:?}"
            );
        }
        // And a realistic wire is not a rounding error: at 5 ohms a 6x6 array is already off by
        // a percent, which is the point of the module.
        let xb =
            Crossbar::new(n, m, g, Wires::new(10.0, "test", Evidence::Unstated).unwrap()).unwrap();
        let ideal = xb.ideal_currents(&v).unwrap();
        let s = xb.solve(&v, 1e-10, 200_000).unwrap();
        assert!(s.max_relative_error(&ideal).unwrap() > 1e-3);
    }

    /// The solver refuses rather than handing back an unconverged iterate, and says by how much it
    /// missed.
    #[test]
    fn the_solver_refuses_rather_than_returning_an_unconverged_answer() {
        let (n, m) = (8, 8);
        let xb = Crossbar::new(
            n,
            m,
            vec![1e-4; n * m],
            Wires::new(10.0, "test", Evidence::Unstated).unwrap(),
        )
        .unwrap();
        let v = vec![0.2; n];
        match xb.solve(&v, 1e-15, 1) {
            Err(DeviceError::NotConverged { sweeps, residual_a, target_a }) => {
                assert_eq!(sweeps, 1);
                assert!(residual_a > target_a, "{residual_a} was not above {target_a}");
            }
            other => panic!("expected NotConverged, got {other:?}"),
        }
        assert!(xb.solve(&v, 0.0, 100).is_err());
        assert!(xb.solve(&v, 1e-9, 0).is_err());
        assert!(xb.solve(&[0.2], 1e-9, 100).is_err());
        assert!(xb.solve(&vec![f64::NAN; n], 1e-9, 100).is_err());
        assert!(Crossbar::new(0, 3, vec![], Wires::ideal()).is_err());
        assert!(Crossbar::new(2, 3, vec![1e-6; 5], Wires::ideal()).is_err());
        assert!(Crossbar::new(2, 3, vec![-1e-6; 6], Wires::ideal()).is_err());
        assert!(Crossbar::new(2, 3, vec![f64::INFINITY; 6], Wires::ideal()).is_err());
        assert!(Wires::new(-1.0, "", Evidence::Unstated).is_err());
    }

    // ---------------------------------------------------------------- bookkeeping

    /// Every mechanism switched on at once, for determinism and for the grade. The point of the
    /// first assertion is that nothing in `apply` reaches for a clock or a hash order.
    #[test]
    fn the_same_seed_gives_the_same_array_and_the_weakest_grade_wins() {
        let mut model = DeviceModel::ideal();
        model.window =
            Window::new(1e-6, 100e-6, "test", Evidence::Measured).unwrap();
        model.beta = 1e-6;
        model.d2d_seed = 2024;
        model.levels = Some(Levels::new(32, "test", Evidence::Simulated).unwrap());
        model.variability = Some(RRAM_VARIABILITY_PLACEHOLDER);
        model.stuck = Some(RRAM_STUCK_AT_PLACEHOLDER);
        model.drift = Some(PCM_DRIFT_AMORPHOUS);
        model.retention = Some(Retention::new(1.2, 1e-14, "test", Evidence::Measured).unwrap());
        model.read_noise = Some(THERMAL_READ_NOISE_300K);
        assert_eq!(model.enabled().len(), 6);
        assert_eq!(model.weakest_evidence(), Evidence::Projected);

        let w: Vec<f64> = (0..300).map(|k| (k as f64 % 61.0) - 30.0).collect();
        let a = model.apply(&w, &mut Rng::new(5)).unwrap();
        let b = model.apply(&w, &mut Rng::new(5)).unwrap();
        assert_eq!(a, b, "two runs of the same seed disagreed");
        assert_eq!(a.len(), 300);
        assert!(!a.is_empty());

        // Ageing makes it worse, monotonically in time, and ages from the programmed state rather
        // than compounding.
        let e0 = a.error();
        let e1 = a.aged(3_600.0, 358.15).unwrap().error();
        let e2 = a.aged(3_600.0 * 24.0 * 365.0, 358.15).unwrap().error();
        assert!(e1.rms > e0.rms && e2.rms > e1.rms, "{e0:?} {e1:?} {e2:?}");
        assert_eq!(a.aged(3_600.0, 358.15).unwrap(), a.aged(3_600.0, 358.15).unwrap());
        assert!(a.aged(-1.0, 300.0).is_err());
        assert!(a.aged(1.0, 0.0).is_err());

        // The catalogue's honesty rule: nothing transcribed is graded above Projected.
        assert_eq!(PCM_DRIFT_AMORPHOUS.evidence, Evidence::Projected);
        assert_eq!(RRAM_VARIABILITY_PLACEHOLDER.evidence, Evidence::Projected);
        assert_eq!(RRAM_STUCK_AT_PLACEHOLDER.evidence, Evidence::Projected);
        assert_eq!(THERMAL_READ_NOISE_300K.evidence, Evidence::Derived);
    }

    /// Every boundary refusal, by name. Bad inputs must be refused rather than producing a plausible
    /// array.
    #[test]
    fn bad_inputs_are_refused_by_name() {
        let model = DeviceModel::ideal();
        assert!(matches!(model.apply(&[], &mut Rng::new(1)), Err(DeviceError::Empty)));
        assert!(matches!(
            model.apply(&[0.1, f64::NAN], &mut Rng::new(1)),
            Err(DeviceError::NonFiniteWeight { index: 1, .. })
        ));
        let mut bad = model;
        bad.beta = 0.0;
        assert!(matches!(
            bad.apply(&[0.1], &mut Rng::new(1)),
            Err(DeviceError::NotPositive { what: "beta", .. })
        ));
        bad.beta = f64::NAN;
        assert!(matches!(bad.apply(&[0.1], &mut Rng::new(1)), Err(DeviceError::NonFinite { .. })));
        assert!(Variability::new(-0.1, 0.0, 0.0, "", Evidence::Unstated).is_err());
        assert!(Variability::new(0.0, f64::NAN, 0.0, "", Evidence::Unstated).is_err());
        assert!(ReadNoise::new(0.0, 1e6, 0.0, "", Evidence::Unstated).is_err());
        assert!(ReadNoise::new(300.0, 0.0, 0.0, "", Evidence::Unstated).is_err());
        assert!(ReadNoise::new(300.0, 1e6, -1.0, "", Evidence::Unstated).is_err());
        // Displays are not empty and name the quantity.
        let e = DeviceError::OutOfRange { index: 3, weight: 9.0, min: -1.0, max: 1.0 };
        assert!(e.to_string().contains("weight 3"));
        assert!(DeviceError::Empty.to_string().contains("empty"));
        assert!(
            DeviceError::NotConverged { sweeps: 2, residual_a: 1.0, target_a: 0.1 }
                .to_string()
                .contains("converge")
        );
        assert!(DeviceError::SingularLine.to_string().contains("pivot"));
        assert!(DeviceError::BadShape { rows: 2, cols: 3, len: 5 }.to_string().contains('6'));
        assert!(DeviceError::BadDrive { rows: 2, len: 5 }.to_string().contains('2'));
        assert!(DeviceError::BadLevels { levels: 1 }.to_string().contains("two"));
        assert!(
            DeviceError::BadProbability { what: "rate", value: 2.0 }.to_string().contains("rate")
        );
        assert!(DeviceError::Negative { what: "x", value: -1.0 }.to_string().contains('x'));
    }

    /// `Programmed::crossbar` hands the same array to the circuit solver, and an ideal device with
    /// ideal wires still gets the exact dot product — the two halves of the module agreeing on the
    /// identity case.
    #[test]
    fn the_programmed_array_runs_through_the_crossbar_unchanged() {
        let mut model = DeviceModel::ideal();
        model.window = win();
        model.beta = 1e-6;
        model.mapping = Mapping::SingleEnded;
        let w: Vec<f64> = (0..12).map(|k| f64::from(k) * 2.0).collect();
        let held = model.apply(&w, &mut Rng::new(6)).unwrap();
        let xb = held.crossbar(3, 4, true, Wires::ideal()).unwrap();
        assert_eq!(xb.rows(), 3);
        assert_eq!(xb.cols(), 4);
        assert_eq!(xb.conductance(2, 3), Some(held.g_plus[11]));
        assert!(held.crossbar(5, 5, true, Wires::ideal()).is_err());
        let v = vec![0.2, 0.1, 0.05];
        let ideal = xb.ideal_currents(&v).unwrap();
        let s = xb.solve(&v, 1e-12, 10).unwrap();
        assert_eq!(s.column_currents, ideal);
        let e: ErrorStats = held.error();
        assert!(e.max_abs < 1e-13);
    }
}
