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
//! | Device-to-device spread | A fixed per-cell **relative** factor that survives reprogramming | [`Variability::sigma_d2d_rel`] |
//! | Cycle-to-cycle spread | A fresh error every time the cell is written | [`Variability::sigma_c2c_rel`] |
//! | Conductance drift | A slow power-law decay after programming | [`Drift`] |
//! | Retention | Thermally activated relaxation back toward the off state | [`Retention`] |
//! | Stuck-at faults | A fraction of cells pinned to a rail, ignoring what you wrote | [`StuckAt`] |
//! | Read noise | A fresh error on every read, with a thermal floor | [`ReadNoise`] |
//! | Finite states | The cell holds one of `n` levels, not a real number | [`Levels`] |
//! | Wire resistance | The answer depends on **where** the cell is | [`Crossbar`] |
//!
//! ⚠ **Relative**, in that first row, is the whole of the distinction and it was wrong here for two
//! releases: the frozen quantity is the factor `1 + sigma_d2d*z`, so the absolute offset
//! `g * sigma_d2d * z` moves whenever the cell is reprogrammed to a different target, and only the
//! fraction stays put. A test that reprograms the same weight vector twice cannot tell the two
//! readings apart; `device_to_device_is_frozen_to_the_array_and_cycle_to_cycle_is_not` reprograms a
//! **different** vector for that reason.
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
//! **Ten survived the pass they were first tried in** — nine gaps in the tests and one defect in
//! the code — and what they were is the useful part:
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
//! The tenth was in the code rather than the tests, and it is left visible:
//! [`Mapping::BalancedDifferential`] shipped its first draft claiming **half the weight range** of
//! the one-at-minimum scheme, with a matching `max_weight` and a test asserting it. It has the same
//! range. The real trade is [`Mapping::common_mode_conductance`], and the wrong claim survived
//! because the test was written from the doc instead of from the circuit.
//!
//! # The second pass, and what a fresh attacker found that the first missed
//!
//! Seventy-five mutations were then injected by a reader who had not written any of this. Fifty-three
//! died. Four of the survivors were algebraically equivalent rewrites rather than defects. The rest
//! were real, and they cluster in one place: **the mechanisms whose absence is invisible in the
//! answer.** Each now has a test named for what it pins.
//!
//! * [`Variability::sigma_floor_s`] could be **deleted outright** with the suite green, because
//!   every `Variability` built in the tests, and [`RRAM_VARIABILITY_PLACEHOLDER`] too, set it to
//!   zero. The sentence above — "every mechanism here has at least one test" — was false for the
//!   absolute noise floor, which is the one term that does not vanish at `g = 0`.
//! * The two devices of a differential pair could share one device-to-device draw. Under a shared
//!   draw the spread stops being a two-sided error and becomes a **per-cell multiplicative gain** a
//!   single calibration scalar removes — precisely the too-good cancellation this module exists to
//!   refuse, and it is guarded for drift and was not for programming spread.
//! * [`DeviceModel::validate`] was a five-way hole: only the window branch was covered, because
//!   every refusal test went through a **constructor**, and the fields are public so the
//!   struct-literal case is the only one `validate` is for. [`Levels`] had no `validate` at all, and
//!   a zero-level grid was accepted and silently programmed every weight to the off rail.
//! * The documented snap-then-perturb order in [`DeviceModel::apply`] was unpinned, and reversing it
//!   lands every cell exactly on a level however large the write error — which is the iterative
//!   write-verify loop the "what this module does not do" section says is **not** modelled.
//! * [`Drift::exponent_at`] had no test at all, so its clamp could be deleted and
//!   [`Drift::apply`] would return a **negative** exponent — conductance growing with time — for any
//!   cell above the window.
//! * [`ReadNoise::distinguishable_levels`] returned `Some(4294967295)` for a very quiet read and
//!   `Some(1)` — no analog information at all — for the numerically *unbounded* case.
//! * [`ReadNoise::sigma_current`] returned `Some(NaN)` where its doc promised `None`, and
//!   [`Programmed::read`] handed that back as `Ok([NaN, NaN])`.
//! * [`Crossbar::solve`]'s tolerance was measured against the **ideal** column current, so its
//!   documented "nine digits of the answer" was overstated by the exact factor the wires cost.
//! * [`Programmed::stuck_devices`] could drop the entire minus half of every pair, and
//!   [`ErrorStats::clamped`] the minus half of every clamp, because every counting test used
//!   [`Mapping::SingleEnded`].
//!
//! The lesson they share is worth more than the list: **a mechanism that only ever runs with a
//! coefficient of zero in the tests is not tested**, and neither is a branch reachable only through
//! the half of a data structure the fixtures never populate.
//!
//! # The third pass, and the three ways a mechanism goes unlooked-at
//!
//! Two hundred and eighty-one mutations were then injected against the module as the second pass
//! left it. Twenty-four survived and **not one of them was an error in the arithmetic**: every one
//! was a guard, a seed, a bookkeeping field or a refusal that the suite computed and then never
//! read. They sort into three shapes, and the shapes are the useful part.
//!
//! * **A validator's census with a row nobody breaks.** [`Variability::sigma_floor_s`] and
//!   [`Drift::nu_at_on`] could each be replaced by a duplicate of the field beside them, because
//!   every ill-formed fixture in this module breaks the *first* row of its table — and
//!   [`Drift::uniform`] sets both exponents at once, so the only bad drift model here cannot reach
//!   the second row at all. The same hole covered a negative [`Retention::ea_ev`], the stress
//!   temperature of [`Retention::acceleration`], an infinite [`Window::g_on`], and an infinite
//!   [`Wires::r_segment_ohm`] — which is the one that matters downstream, because an infinite
//!   segment resistance is the only way a caller reaches [`Crossbar::solve`] with a wire
//!   conductance of exactly zero.
//! * **A distinction between two refusals.** `check_temperature` returns
//!   [`DeviceError::NonFinite`] for a `NaN` and [`DeviceError::NotPositive`] for a zero, and every
//!   test that hands it a bad temperature asserts `is_err()` and nothing else. `!(NaN > 0.0)` is
//!   **true**, so deleting the first branch returns the wrong variant for a `NaN` with the suite
//!   green — and `!(inf > 0.0)` is **false**, so it accepts an infinite temperature outright and
//!   [`Retention::tau_s`] hands back `tau0_s` as a lifetime.
//! * **State that no answer depends on.** [`DeviceModel::d2d_seed`] could stop reaching its
//!   per-cell streams entirely — nothing here ever programmed the same weights onto two different
//!   seeds, so "the same seed is the same physical array" was vacuously true of every pair of
//!   models. The two devices of a pair could share one fault draw, which every marginal count in
//!   `a_differential_pair_costs_twice_the_exposure_to_stuck_at_faults` survives. [`Programmed::age_s`]
//!   and [`Programmed::aged_at_k`] could be written with anything at all. And a refused
//!   tridiagonal solve could fill the caller's buffers with infinities, because both pivot guards
//!   return `false` either way and only the buffers say which.
//!
//! One correction to the record came out of it. This module's own argument for why
//! [`Crossbar::residual`] needs a single finiteness check rather than one per node set was
//! **wrong in the direction that matters**: the two residuals share `g * (a - b)`, but they do not
//! share the wire term, so the bit-line half of that check is load-bearing after all. The argument
//! is rewritten where it lives and the case is pinned by a test.
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
    /// A parameter — or a quantity computed from accepted parameters — was `NaN` or infinite.
    ///
    /// The second case is the one worth naming: [`Retention::tau_s`] takes an activation energy and
    /// a temperature this module accepts and can still overflow to an infinite lifetime, and an
    /// infinity that reaches a printed figure is indistinguishable from a very long one. `what`
    /// names the parameter, or the quantity, by the name it has in this module.
    NonFinite {
        /// Which parameter or computed quantity, by the name it has in this module.
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
    ///
    /// Both conductances are brought inside `window` before they are returned. That is not a
    /// saturation — a weight the window cannot hold is **refused** by [`DeviceModel::apply`], never
    /// clamped — it is the last-place rounding of the mapping's own arithmetic:
    /// [`Mapping::BalancedDifferential`] writes `mid - w*beta/2`, and at full negative scale
    /// `0.5*(g_off + g_on) - 0.5*(g_on - g_off)` is `g_off` in exact arithmetic and can miss it by
    /// a fifth of a unit in the last place in `f64`. Unclamped, this module's own invariant —
    /// every programmed conductance lies inside the window, [`Window::contains`] — was false for
    /// exactly one weight per array, the full-scale one, and only when [`DeviceModel::levels`] and
    /// [`DeviceModel::variability`] were both off, since either of those clamps on its own.
    #[must_use]
    pub fn program(&self, w: f64, window: &Window, beta: f64) -> (f64, f64) {
        let (g_plus, g_minus) = match self {
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
        };
        (window.clamp(g_plus), window.clamp(g_minus))
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
    ///
    /// Also [`DeviceError::NonFinite`] naming `tau_s` when the **result** overflows, which it does
    /// for a large activation energy at a low temperature — `Ea = 15 eV` at `200 K` is `exp(870)`
    /// and has no `f64`. Refused for the same reason [`Window::on_off_ratio`] refuses: an infinity
    /// handed back as a lifetime reads as "very long" to every caller downstream, and it is not a
    /// number this model computed, it is a number this model ran out of room for.
    pub fn tau_s(&self, temp_k: f64) -> Result<f64, DeviceError> {
        check_temperature(temp_k)?;
        let tau = self.tau0_s * (self.ea_ev / (BOLTZMANN_EV_PER_K * temp_k)).exp();
        if !tau.is_finite() {
            return Err(DeviceError::NonFinite { what: "tau_s", value: tau });
        }
        Ok(tau)
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
    /// As [`Retention::tau_s`], for either temperature, including
    /// [`DeviceError::NonFinite`] naming `acceleration` when the exponential overflows. Underflow
    /// to exactly zero is **not** refused: a zero acceleration factor is a small number the
    /// arithmetic could not hold, which is honest in the direction it errs, whereas an infinity is
    /// not.
    pub fn acceleration(&self, t_use_k: f64, t_stress_k: f64) -> Result<f64, DeviceError> {
        check_temperature(t_use_k)?;
        check_temperature(t_stress_k)?;
        let k = self.ea_ev / BOLTZMANN_EV_PER_K;
        let af = (k * (1.0 / t_use_k - 1.0 / t_stress_k)).exp();
        if !af.is_finite() {
            return Err(DeviceError::NonFinite { what: "acceleration", value: af });
        }
        Ok(af)
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
    ///
    /// Also `None` for a **model** that does not pass [`ReadNoise::validate`]. The fields here are
    /// public, and a negative `temp_k` makes the thermal term negative and its square root a `NaN`,
    /// which this function used to hand back inside a `Some` — a plausible-looking wrong answer
    /// where a refusal was documented. A negative temperature has no noise power for exactly the
    /// reason a negative conductance does not.
    #[must_use]
    pub fn sigma_current(&self, g: f64, v_read: f64) -> Option<f64> {
        self.validate().ok()?;
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
    /// `None` for a non-positive `v_read` or `separation`, for an invalid window or an invalid
    /// noise model, or whenever the count is **not bounded by the noise**: the number of levels is
    /// then not large, it is unbounded, and this module declines to print an infinity as an
    /// integer.
    ///
    /// "Not bounded by the noise" covers three arithmetically distinct cases that are one physical
    /// case, and all three return `None`:
    ///
    /// * the noise is exactly zero;
    /// * `separation * sigma` underflows to zero, so the quotient is an infinity. Routing that into
    ///   the `Some(1)` branch — as an earlier draft did — reports **no analog information at all**
    ///   for the array with the most of it, which is the numerical opposite of the truth;
    /// * the count exceeds [`u32::MAX`]. A saturated count is indistinguishable from a real one to
    ///   the caller, and `Levels::new(u32::MAX)` would then accept it and report 32 effective bits
    ///   from a resistive cell — the same infinity printed as an integer, under another name.
    ///
    /// `Some(1)` is a different statement and is reserved for its opposite: noise so large that no
    /// two conductances in the window are separable, so the cell carries no analog information.
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
        // A finite positive `separation` times a finite positive `sigma` can still UNDERFLOW to
        // zero, and `span / 0.0` is an infinity. Splitting `!n.is_finite()` away from `n < 2.0` is
        // the whole fix: the two used to share a branch that returned `Some(1)`, so the unbounded
        // case was reported as the no-information one.
        let n = window.span() / (separation * sigma) + 1.0;
        if !n.is_finite() || n > f64::from(u32::MAX) {
            return None;
        }
        if n < 2.0 {
            return Some(1);
        }
        Some(n.floor() as u32)
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
        let l = Self { n, source, evidence };
        l.validate()?;
        Ok(l)
    }

    /// Check that this is a quantiser: at least two levels.
    ///
    /// Public for the same reason [`Window::validate`] is: the fields are public, so a `Levels`
    /// built by struct literal can hold `n = 0`, and that grid quantises **every** conductance to
    /// [`Window::g_off`] — a weight of 10 programmed as the off rail, with no error reported and
    /// `bits()` of `-inf` available to print. [`DeviceModel::validate`] calls this, which is what
    /// makes [`DeviceModel::apply`] refuse such a model rather than silently zeroing the array.
    ///
    /// # Errors
    ///
    /// [`DeviceError::BadLevels`] for fewer than two levels.
    pub fn validate(&self) -> Result<(), DeviceError> {
        if self.n < 2 {
            return Err(DeviceError::BadLevels { levels: self.n });
        }
        Ok(())
    }

    /// Spacing between adjacent levels, **siemens**: `span / (n - 1)`.
    ///
    /// The `n.max(2)` is a guard, not a rounding: [`Levels::validate`] refuses `n < 2`, and this
    /// keeps a struct-literal grid from dividing by zero if one is ever reached anyway.
    #[must_use]
    pub fn step(&self, window: &Window) -> f64 {
        window.span() / f64::from(self.n.max(2) - 1)
    }

    /// Effective bits, `log2(n)`. Fractional on purpose: 20 levels is 4.32 bits, not 4, and
    /// rounding it down is how a device gets reported as worse than it is.
    ///
    /// Exactly `log2(n)` and nothing else, so an unvalidated `n = 0` gives `-inf` and `n = 1` gives
    /// `0.0`. Both are the right answers to the question asked and neither is a grid:
    /// [`Levels::validate`] is what refuses them, and it is not called from here because a
    /// `#[must_use]` arithmetic accessor that silently substituted `n = 2` would report **one bit**
    /// for a cell that holds no states at all.
    #[must_use]
    pub fn bits(&self) -> f64 {
        f64::from(self.n).log2()
    }

    /// `g` snapped to the nearest level, clamped into the window.
    ///
    /// The result is `g_off + k * step` for an integer `k` in `0..n`, and the error is at most half
    /// a step for any `g` already inside the window. Both are asserted, and the index clamp is what
    /// makes the first one true above the window: without its ceiling the arithmetic runs off the
    /// grid and [`Window::clamp`] rescues it to [`Window::g_on`], which is not a level.
    ///
    /// ⚠ **`k` is exact; the conductance is exact to a unit in the last place.** `step` is
    /// `span / (n - 1)`, and `(n - 1) * step` need not reproduce `span` in `f64`, so the top level
    /// can sit a last place either side of [`Window::g_on`] — and [`Window::clamp`] then pulls the
    /// high side back to `g_on`, which is off the grid by that one place. Measured on the 1 uS to
    /// 100 uS window used by this module's tests: at `n = 64` and `n = 256` the top level is
    /// `g_on` exactly; at `n = 6` it is 9.999999999999999e-05, one place BELOW, and `snap` returns
    /// that, on the grid; at `n = 100` it is 1.0000000000000002e-4, one place above, and `snap`
    /// returns `g_on`, off the grid by 1.36e-20 S. The clamp is kept anyway, because the
    /// alternative is a programmed conductance outside the window — see
    /// `every_programmed_conductance_lies_inside_the_window` — and one last place of grid error is
    /// the cheaper of the two. This doc said "always exactly" for three releases and it was the
    /// high side that made it false.
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
/// maturity, so it varies by orders of magnitude between a research wafer and a product.
///
/// ⛔ **The citation covers the mechanism and the order of magnitude. It does not cover
/// [`StuckAt::p_at_on`].** The half-and-half split is a convention of this crate: the two failure
/// mechanisms are unrelated, the fault-tolerance literature that motivates this rate reports
/// **markedly asymmetric** stuck-off and stuck-on rates rather than an even split, and this review
/// did not open that table to read the two numbers out of it — so it declines to transcribe them
/// here, per this module's own grading rule. `0.5` is the neutral placeholder a reader must replace,
/// not a figure from the cited work, and a reader who takes the citation as covering the whole
/// struct takes a split that is wrong by a large factor in a named direction.
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

    /// Check the window, `beta`, and every enabled sub-model — **all seven**, by the same census
    /// [`DeviceModel::weakest_evidence`] folds.
    ///
    /// Every field of every sub-model is public, so a `DeviceModel` can carry a struct-literal
    /// sub-model that its own constructor would have refused. That, and not the constructor, is the
    /// case this function exists for: `bad_inputs_are_refused_by_name` exercises the constructors
    /// and cannot see a fold that is missing a branch.
    ///
    /// # Errors
    ///
    /// Whatever the offending sub-model returns; see [`Window::validate`],
    /// [`Variability::validate`], [`Levels::validate`], [`Drift::validate`],
    /// [`Retention::validate`], [`StuckAt::validate`] and [`ReadNoise::validate`], plus
    /// [`DeviceError::NotPositive`] or [`DeviceError::NonFinite`] for a bad `beta`.
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
        if let Some(l) = &self.levels {
            l.validate()?;
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
    /// be inferred from, plus anything [`DeviceModel::validate`] returns.
    ///
    /// The model is re-checked here and not assumed sound from [`DeviceModel::apply`], because
    /// [`Programmed::model`] is a public field: a caller who installs a struct-literal
    /// [`ReadNoise`] with a negative temperature used to get back `Ok([NaN, NaN])`, which is the
    /// plausible-looking wrong answer [`DeviceError::NotConverged`]'s doc argues against, one
    /// function along.
    pub fn read(&self, rng: &mut Rng, v_read: f64) -> Result<Vec<f64>, DeviceError> {
        if !v_read.is_finite() {
            return Err(DeviceError::NonFinite { what: "v_read", value: v_read });
        }
        if !(v_read > 0.0) {
            return Err(DeviceError::NotPositive { what: "v_read", value: v_read });
        }
        self.model.validate()?;
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

    /// The wire model this crossbar was built with, provenance string and grade included.
    #[must_use]
    pub fn wires(&self) -> &Wires {
        &self.wires
    }

    /// The evidence grade an IR-drop figure from this crossbar carries: [`Wires::evidence`].
    ///
    /// Named to match [`DeviceModel::weakest_evidence`] and reported **separately** from it on
    /// purpose. That fold covers the eight per-cell mechanisms; wire resistance is the ninth, it is
    /// not per cell, it does not live in a [`DeviceModel`], and it is the one the module doc singles
    /// out as most often omitted. A caller reporting a figure that used [`Crossbar::solve`] has to
    /// take the weaker of the two grades, and before this accessor existed the wire model's
    /// provenance was carried, round-tripped through [`Wires::new`] and never readable at all.
    ///
    /// A crossbar carries no other model, so this is that one grade and not a fold.
    #[must_use]
    pub fn weakest_evidence(&self) -> Evidence {
        self.wires.evidence
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
    /// `tol` is **relative to the largest column current this solve actually returns**, so `1e-9`
    /// means nine digits of the answer. Convergence is measured as the worst Kirchhoff current
    /// residual at any node — not as the change between sweeps, which can be small while the
    /// solution is still far away.
    ///
    /// ⚠ Normalising by the largest **ideal** current instead — which an earlier draft did — reads
    /// identically in the regime where wires are negligible and overstates the accuracy by the
    /// ratio `ideal / actual` in the regime this module exists to expose. That factor is unbounded
    /// and is largest exactly where a caller most needs the bound: an 8x8 array of `1e-2 S` cells on
    /// `100 ohm` wires delivers a twentieth of its ideal current, so `tol = 1e-9` bought `2e-8` of
    /// the answer; at `g = 1e100 S` on `1e-6 ohm` wires the old target was `1e-9 * 4e299` and the
    /// first sweep "converged" with a residual of `2e5 A`.
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
        // A Kirchhoff residual is a difference of wire currents of magnitude `gw * v`, and those
        // cancel down to the cell current, so the residual cannot be driven below the rounding
        // noise of that cancellation however many sweeps are spent on it. Floor the target there.
        // A stiff array — small `r_segment_ohm`, so large `gw` — has a HIGHER floor, and
        // `Solution::residual_a` is reported so a caller can see which bound they stopped on.
        let v_max = v_in.iter().fold(0.0f64, |a, b| a.max(b.abs()));
        let noise_floor = 8.0 * f64::EPSILON * gw * v_max;

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
        let mut target = f64::MIN_POSITIVE;
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

            // The scale the tolerance is measured against is the answer THIS sweep would return —
            // the same expression `column_currents` is built from, so the two cannot disagree —
            // and not the ideal current, which the wires are in the business of not delivering.
            let mut scale = 0.0f64;
            for j in 0..m {
                scale = scale.max((gw * b[(n - 1) * m + j]).abs());
            }
            target = (tol * scale).max(noise_floor).max(f64::MIN_POSITIVE);

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

    /// Worst Kirchhoff current residual over every node, **amperes**, or [`f64::INFINITY`] if any
    /// node's residual is not finite.
    ///
    /// The explicit non-finite check is load-bearing and is **not** what `f64::max` does:
    /// `f64::max` returns the non-`NaN` operand, so an all-`NaN` node-voltage solution folded with
    /// `worst.max(r.abs())` reports a worst residual of `0.0` and
    /// [`Crossbar::solve`] declares it converged. Returning an infinity instead sends the same
    /// state to [`DeviceError::NotConverged`], which is where it belongs.
    ///
    /// It is **one** check naming both of a node's residuals rather than two branches with no
    /// input between them, and **both names are load-bearing**. The two residuals do share the
    /// term `g * (a - b)`, so a non-finite node VOLTAGE poisons both and the word-line half alone
    /// would catch it, and an infinity in either — unlike a `NaN` — already propagates through
    /// `f64::max` on its own. That much was once written here as an argument that the bit-line
    /// half was redundant, and it is incomplete: the poison also enters through the **wire** term,
    /// and there the two sides are not symmetric. The bit-line residual carries
    /// `gw * (south - b)` and `gw * (b_north - b)`; the word-line residual carries neither. At
    /// `gw == 0.0`, against node voltages whose difference overflows, `0.0 * inf` is therefore a
    /// `NaN` on the bit-line side alone, and `f64::max` drops it.
    /// `a_non_finite_bit_line_residual_is_not_dropped_by_the_fold` is that case. A zero `gw` is a
    /// segment resistance of infinity, which [`Wires::new`] refuses — so this guard and that
    /// refusal hold each other up.
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

                let south = if i + 1 < n { b[k + m] } else { 0.0 };
                let mut s = gw * (south - b[k]) + g * (a[k] - b[k]);
                if i > 0 {
                    s += gw * (b[k - m] - b[k]);
                }

                if !r.is_finite() || !s.is_finite() {
                    return f64::INFINITY;
                }
                worst = worst.max(r.abs()).max(s.abs());
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
///
/// ⚠ The two **zero**-pivot comparisons change no return value, and that is easy to mistake for
/// them changing nothing. A zero pivot divides into `out[k]`, the infinity or `NaN` that produces
/// survives the back substitution, and the closing finiteness check returns `false` from the end
/// of the function instead of from the guard. What the comparisons buy is the second half of the
/// sentence above: the caller's `out` and `c` are left without an infinity in them.
/// `a_refused_line_solve_writes_no_infinity_into_the_caller_s_buffers` is what pins that, because
/// an assertion on the boolean cannot.
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
        RRAM_STUCK_AT_PLACEHOLDER, RRAM_VARIABILITY_PLACEHOLDER, ReadNoise, Retention, Solution,
        StuckAt, THERMAL_READ_NOISE_300K, Variability, Window, Wires, normal, thomas,
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

        // FROZEN means a fixed per-cell RELATIVE factor, not a fixed offset in siemens — the module
        // doc's catalogue said "offset" for two releases and nothing could tell the two readings
        // apart, because every reprogramming above writes the SAME weight vector, where `g` does
        // not move and `g * sigma * z` therefore does not either.
        //
        // So reprogram a DIFFERENT vector on the same array. The fraction has to survive it cell by
        // cell; the offset in siemens has to move. Single-ended and mid-window so that `g_off` is
        // not the target and no rail truncates the comparison.
        let mut m2 = model;
        m2.mapping = Mapping::SingleEnded;
        m2.variability =
            Some(Variability::new(0.05, 0.0, 0.0, "d2d only", Evidence::Unstated).unwrap());
        let t1: Vec<f64> = (0..200).map(|k| 20.0 + 0.1 * f64::from(k)).collect();
        let t2: Vec<f64> = t1.iter().map(|x| 2.0 * x).collect();
        let first = m2.apply(&t1, &mut Rng::new(1)).unwrap();
        let second = m2.apply(&t2, &mut Rng::new(7)).unwrap();
        assert_eq!(first.clamped + second.clamped, 0, "a rail truncated the comparison");
        let mut offset_moved = 0;
        for k in 0..t1.len() {
            let g1 = m2.window.g_off + t1[k] * m2.beta;
            let g2 = m2.window.g_off + t2[k] * m2.beta;
            let rel1 = first.g_plus[k] / g1 - 1.0;
            let rel2 = second.g_plus[k] / g2 - 1.0;
            assert!(
                (rel1 - rel2).abs() < 1e-12,
                "cell {k}: the frozen RELATIVE factor moved, {rel1} then {rel2}"
            );
            assert!(rel1.abs() > 1e-9, "cell {k} has no device-to-device offset to test");
            let abs1 = first.g_plus[k] - g1;
            let abs2 = second.g_plus[k] - g2;
            if (abs1 - abs2).abs() > 1e-9 * g1 {
                offset_moved += 1;
            }
        }
        assert!(
            offset_moved > t1.len() / 2,
            "only {offset_moved} of {} cells moved their offset in siemens when the target \
             doubled, so 'relative' and 'a fixed offset' are still indistinguishable here",
            t1.len()
        );
    }

    /// (A1) THE ABSOLUTE NOISE FLOOR, which could be deleted outright with the suite green because
    /// every `Variability` built anywhere in this module — [`RRAM_VARIABILITY_PLACEHOLDER`]
    /// included — passed `0.0` for it. A mechanism that only ever runs with a coefficient of zero
    /// is not tested, and the module doc's claim that every mechanism here has one was false for
    /// exactly this term.
    ///
    /// Three assertions, each of which the relative terms cannot satisfy:
    ///
    /// 1. the spread is the same number of SIEMENS at two targets a factor of four apart, which is
    ///    what "absolute" means and what a relative term is definitionally not;
    /// 2. relative-only noise vanishes at `g = 0` and the floor does not — the physical argument
    ///    in [`Variability::sigma_floor_s`]'s own doc, asserted rather than asserted-in-prose;
    /// 3. with all three on, the total is their quadrature sum, which is the struct doc's formula
    ///    `g*(1 + s_d2d*z1 + s_c2c*z2) + s_floor*z3` read as a variance. Dropping the floor moves
    ///    that closed form by 4.6% here, against a 2% band.
    #[test]
    fn the_absolute_noise_floor_is_a_term_of_its_own_and_does_not_vanish_at_zero() {
        // A zero off-conductance, so a zero target really is a zero conductance and the relative
        // terms have nothing to multiply.
        let w = Window::new(0.0, 100e-6, "test fixture", Evidence::Unstated).unwrap();
        let mut model = DeviceModel::ideal();
        model.window = w;
        model.beta = 1e-6;
        model.mapping = Mapping::SingleEnded;
        model.d2d_seed = 13;

        // 1. The same sigma in siemens at 20 uS and at 80 uS.
        let floor = 2e-6;
        model.variability =
            Some(Variability::new(0.0, 0.0, floor, "floor only", Evidence::Unstated).unwrap());
        let mut spreads = Vec::new();
        for target in [20.0f64, 80.0] {
            let held = model.apply(&vec![target; 40_000], &mut Rng::new(2)).unwrap();
            assert_eq!(held.clamped, 0, "a rail truncated the floor at target {target}");
            let n = held.g_plus.len() as f64;
            let mean = held.g_plus.iter().sum::<f64>() / n;
            let sd = (held.g_plus.iter().map(|g| (g - mean) * (g - mean)).sum::<f64>() / n).sqrt();
            assert!(
                (sd / floor - 1.0).abs() < 0.02,
                "target {target}: floor came out {sd} S, asked for {floor} S"
            );
            spreads.push(sd);
        }
        assert!(
            (spreads[0] / spreads[1] - 1.0).abs() < 1e-12,
            "the floor scaled with the target ({} S then {} S), so it is a relative term",
            spreads[0],
            spreads[1]
        );

        // 2. At g = 0 the relative terms have no effect and the floor is the whole of the noise.
        model.variability =
            Some(Variability::new(0.10, 0.10, 0.0, "relative only", Evidence::Unstated).unwrap());
        let bare = model.apply(&vec![0.0; 1_000], &mut Rng::new(2)).unwrap();
        assert!(
            bare.g_plus.iter().all(|&g| g == 0.0),
            "relative-only noise moved a cell programmed to exactly zero conductance"
        );
        model.variability =
            Some(Variability::new(0.10, 0.10, floor, "with the floor", Evidence::Unstated).unwrap());
        let floored = model.apply(&vec![0.0; 1_000], &mut Rng::new(2)).unwrap();
        assert!(
            floored.g_plus.iter().any(|&g| g > 0.0),
            "the absolute floor did not place a single cell off zero, so it is not applied"
        );

        // 3. The three terms add in QUADRATURE, which is the struct doc's formula as a variance.
        let (s_d2d, s_c2c, s_floor) = (0.05f64, 0.04f64, 1e-6f64);
        model.variability =
            Some(Variability::new(s_d2d, s_c2c, s_floor, "all three", Evidence::Unstated).unwrap());
        let target = 50.0f64;
        let held = model.apply(&vec![target; 40_000], &mut Rng::new(2)).unwrap();
        assert_eq!(held.clamped, 0);
        let n = held.g_plus.len() as f64;
        let mean = held.g_plus.iter().sum::<f64>() / n;
        let sd = (held.g_plus.iter().map(|g| (g - mean) * (g - mean)).sum::<f64>() / n).sqrt();
        let g = target * model.beta;
        let closed = ((g * s_d2d).powi(2) + (g * s_c2c).powi(2) + s_floor * s_floor).sqrt();
        assert!(
            (sd / closed - 1.0).abs() < 0.02,
            "three independent terms gave {sd} S against a quadrature sum of {closed} S"
        );
        // Without the floor the closed form is 4.6% lower, which is what the band above bites on.
        let no_floor = ((g * s_d2d).powi(2) + (g * s_c2c).powi(2)).sqrt();
        assert!(
            (closed / no_floor - 1.0) > 0.04,
            "the floor contributes too little here to separate the two closed forms"
        );
    }

    /// (A2) The two devices of a differential pair draw their device-to-device offsets
    /// INDEPENDENTLY, and the pair-at-zero carries an error because of it.
    ///
    /// Sharing one draw between the two halves passes every other test in this module, and it is
    /// not a small error: with a shared `z` the read is `(gp0 - gm0)(1 + sigma*z) / beta`, so the
    /// spread stops being a two-sided programming error and becomes a per-cell **multiplicative
    /// gain** that one calibration scalar removes. That is exactly the too-good cancellation this
    /// module exists to refuse — `state_dependent_drift_does_not_cancel` guards it for drift and
    /// nothing guarded it for programming spread.
    ///
    /// The discriminator is a pair holding a weight of exactly zero: under a shared draw the two
    /// conductances are identical and the error is **exactly 0.0**.
    #[test]
    fn the_two_devices_of_a_pair_draw_their_offsets_independently() {
        let w = win();
        let sigma = 0.10;
        let beta = 1e-6;
        let mut model = DeviceModel::ideal();
        model.window = w;
        model.beta = beta;
        model.d2d_seed = 9;
        model.variability =
            Some(Variability::new(sigma, 0.0, 0.0, "d2d only", Evidence::Unstated).unwrap());

        // Balanced: both devices park at mid-window, ten sigma from either rail, so the closed form
        // is untruncated. Two independent draws of relative sigma give a weight error of
        // mid * sigma * sqrt(2) / beta.
        model.mapping = Mapping::BalancedDifferential;
        let held = model.apply(&vec![0.0; 40_000], &mut Rng::new(1)).unwrap();
        assert_eq!(held.clamped, 0, "a rail truncated the pair and confounded the closed form");
        let mid = w.mid();
        let n = held.g_plus.len() as f64;
        let xs: Vec<f64> = held.g_plus.iter().map(|g| g / mid - 1.0).collect();
        let ys: Vec<f64> = held.g_minus.iter().map(|g| g / mid - 1.0).collect();
        let mx = xs.iter().sum::<f64>() / n;
        let my = ys.iter().sum::<f64>() / n;
        let sx = (xs.iter().map(|x| (x - mx) * (x - mx)).sum::<f64>() / n).sqrt();
        let sy = (ys.iter().map(|y| (y - my) * (y - my)).sum::<f64>() / n).sqrt();
        let cov = xs.iter().zip(&ys).map(|(x, y)| (x - mx) * (y - my)).sum::<f64>() / n;
        let rho = cov / (sx * sy);
        assert!((sx / sigma - 1.0).abs() < 0.03, "the plus device's spread is {sx}");
        assert!((sy / sigma - 1.0).abs() < 0.03, "the minus device's spread is {sy}");
        assert!(
            rho.abs() < 0.05,
            "the two devices of a pair are correlated at {rho}; a shared draw gives exactly 1.0"
        );
        let rms = held.error().rms;
        let closed = mid * sigma * 2.0f64.sqrt() / beta;
        assert!(
            (rms / closed - 1.0).abs() < 0.02,
            "a pair of zero weights carried an rms error of {rms}, two independent draws say \
             {closed}, and one shared draw says exactly 0"
        );

        // One-at-minimum: both devices sit ON the bottom rail for a zero weight, so half of each
        // Gaussian is truncated there. The closed form is the difference of two independent
        // half-normals: sigma * sqrt(2 * (1/2 - 1/(2 pi))). It is still not zero, which is the
        // point — under a shared draw it is.
        model.mapping = Mapping::Differential;
        let held = model.apply(&vec![0.0; 40_000], &mut Rng::new(1)).unwrap();
        let rms = held.error().rms;
        let truncated = sigma * (2.0 * (0.5 - 1.0 / core::f64::consts::TAU)).sqrt();
        assert!(
            (rms / truncated - 1.0).abs() < 0.02,
            "a one-at-minimum pair of zero weights gave {rms}, the truncated closed form says \
             {truncated}"
        );
        assert!(rms > 0.0, "a differential array of zero weights carried no programming error");
    }

    /// (A4) The documented order in [`DeviceModel::apply`]: snap to the level grid FIRST, then
    /// perturb. Reversing the two silently implements the iterative write-verify loop the module
    /// doc says it does not model — every cell would land exactly on a level however large the
    /// write error, so any error below half a step would disappear.
    ///
    /// `both_devices_of_a_pair_are_written_to_the_level_grid` is the only other test that touches
    /// the grid and it configures no [`Variability`], where the two orders are identical.
    #[test]
    fn programming_is_one_noisy_shot_at_a_level_and_not_a_write_verify_loop() {
        let w = win();
        let levels = Levels::new(9, "test", Evidence::Unstated).unwrap();
        let step = levels.step(&w);
        let sigma_c2c = 0.05;
        let mut model = DeviceModel::ideal();
        model.window = w;
        model.beta = 1e-6;
        model.mapping = Mapping::SingleEnded;
        model.levels = Some(levels);
        model.variability =
            Some(Variability::new(0.0, sigma_c2c, 0.0, "c2c only", Evidence::Unstated).unwrap());
        let targets: Vec<f64> = (0..4_000).map(|k| 20.0 + 0.01 * f64::from(k)).collect();
        let held = model.apply(&targets, &mut Rng::new(5)).unwrap();
        assert_eq!(held.clamped, 0, "a rail truncated the write error");

        // Not one cell is on the grid. Under the reversed order every one of them would be.
        let off_grid = held
            .g_plus
            .iter()
            .filter(|&&g| g != w.g_off + ((g - w.g_off) / step).round() * step)
            .count();
        assert_eq!(
            off_grid,
            targets.len(),
            "{off_grid} of {} cells sit off the level grid; a perturb-then-snap order puts all of \
             them on it",
            targets.len()
        );

        // And the write error survives at FULL size rather than being swallowed by a re-snap: its
        // rms against the level each cell was aiming at is the cycle-to-cycle sigma itself. Under
        // the reversed order this collapses, because a deviation below half a step snaps back
        // exactly and 98.7% of a 0.05-sigma error on a 12 uS step is below half a step.
        let mut sq = 0.0f64;
        let mut want_sq = 0.0f64;
        for (k, &g) in held.g_plus.iter().enumerate() {
            let aimed = levels.snap(w.g_off + targets[k] * model.beta, &w);
            sq += (g - aimed) * (g - aimed);
            want_sq += (sigma_c2c * aimed) * (sigma_c2c * aimed);
        }
        let ratio = (sq / want_sq).sqrt();
        assert!(
            (ratio - 1.0).abs() < 0.03,
            "the write error came out {ratio} of the sigma asked for; a write-verify loop drives \
             it toward zero"
        );
        assert!(0.5 * step > 2.0 * sigma_c2c * 50e-6, "the step is too fine to separate the orders");

        // The complement, so the snap is not merely absent: with no variability every cell is ON
        // the grid, exactly.
        model.variability = None;
        let clean = model.apply(&targets, &mut Rng::new(5)).unwrap();
        for (k, &g) in clean.g_plus.iter().enumerate() {
            assert!(
                (g - (w.g_off + ((g - w.g_off) / step).round() * step)).abs() < 1e-18,
                "cell {k} at {g} S is not on the 9-level grid"
            );
        }
    }

    /// (A14) [`ErrorStats::clamped`] counts the MINUS device of a pair, which no other test can
    /// see: `clamping_at_a_rail_biases_the_mean_away_from_it` uses [`Mapping::SingleEnded`], where
    /// the second element is a nominal reference that is never perturbed and never clamps.
    ///
    /// The construction isolates it. A window with a zero off-conductance and a full-scale NEGATIVE
    /// one-at-minimum weight puts the plus device at exactly `0 S`, where a relative spread has
    /// nothing to multiply and the clamp can never fire, and the minus device exactly on the top
    /// rail, where it clamps for every positive draw. Every clamp counted here is a minus device.
    #[test]
    fn the_clamp_count_includes_the_minus_device_of_a_pair() {
        let w = Window::new(0.0, 100e-6, "test fixture", Evidence::Unstated).unwrap();
        let mut model = DeviceModel::ideal();
        model.window = w;
        model.beta = 1e-6;
        model.mapping = Mapping::Differential;
        model.d2d_seed = 3;
        model.variability =
            Some(Variability::new(0.30, 0.0, 0.0, "d2d only", Evidence::Unstated).unwrap());
        let n = 20_000;
        let held = model.apply(&vec![-model.max_weight(); n], &mut Rng::new(1)).unwrap();

        assert!(
            held.g_plus.iter().all(|&g| g == 0.0),
            "the plus device moved off zero, so this test no longer isolates the minus device"
        );
        assert!(
            held.g_minus.contains(&w.g_on),
            "no minus device reached the top rail, so nothing was there to clamp"
        );
        // Half the Gaussian is above the rail, so half the pairs clamp: 4 binomial sigmas is 283.
        let expect = 0.5 * n as f64;
        let sd = (n as f64 * 0.25).sqrt();
        assert!(
            (held.clamped as f64 - expect).abs() < 4.0 * sd,
            "{} clamps of {n} minus devices, expected {expect} +/- {sd}",
            held.clamped
        );
        assert_eq!(held.error().clamped, held.clamped, "ErrorStats lost the count");
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

    /// (A9) [`StuckAt::rate`]'s doc claims a pair "has roughly twice the chance of being touched by
    /// a fault, which is a real cost of the scheme and is not adjusted away here". That is a
    /// number, and nothing checked it: every test that counts faults uses [`Mapping::SingleEnded`],
    /// where `fault_minus` is [`Fault::Healthy`] by construction, so
    /// [`Programmed::stuck_devices`] could drop its entire minus term with the suite green.
    ///
    /// The exact-sum assertion is the one that bites: a count that omits a half cannot equal the
    /// two halves added up.
    #[test]
    fn a_differential_pair_costs_twice_the_exposure_to_stuck_at_faults() {
        let rate = 0.10;
        let n = 50_000usize;
        let mut model = DeviceModel::ideal();
        model.window = win();
        model.beta = 1e-6;
        model.d2d_seed = 5;
        model.stuck = Some(StuckAt::new(rate, 0.5, "test", Evidence::Unstated).unwrap());
        let weights = vec![10.0; n];

        model.mapping = Mapping::SingleEnded;
        let single = model.apply(&weights, &mut Rng::new(1)).unwrap();
        model.mapping = Mapping::Differential;
        let pair = model.apply(&weights, &mut Rng::new(1)).unwrap();

        // One device per weight, one binomial: 4 sigma is 268 at this n and rate.
        let sd = (n as f64 * rate * (1.0 - rate)).sqrt();
        assert!(
            (single.stuck_devices() as f64 - rate * n as f64).abs() < 4.0 * sd,
            "{} stuck of {n} single-ended devices",
            single.stuck_devices()
        );
        assert!(
            single.fault_minus.iter().all(|f| !f.is_stuck()),
            "a single-ended array broke a second device it does not have"
        );

        // Two devices per weight, so twice the exposure — the cost the doc names.
        let plus = pair.fault_plus.iter().filter(|f| f.is_stuck()).count();
        let minus = pair.fault_minus.iter().filter(|f| f.is_stuck()).count();
        assert!(minus > 0, "not one minus device of {n} pairs was stuck at a rate of {rate}");
        assert_eq!(
            plus + minus,
            pair.stuck_devices(),
            "stuck_devices() reported {} against {plus} plus and {minus} minus: it is not \
             counting both devices of a pair",
            pair.stuck_devices()
        );
        assert!(
            (minus as f64 - rate * n as f64).abs() < 4.0 * sd,
            "{minus} of {n} minus devices stuck, expected {} +/- {sd}",
            rate * n as f64
        );
        let sd2 = (2.0 * n as f64 * rate * (1.0 - rate)).sqrt();
        assert!(
            (pair.stuck_devices() as f64 - 2.0 * rate * n as f64).abs() < 4.0 * sd2,
            "{} stuck of {} paired devices, expected {} +/- {sd2}",
            pair.stuck_devices(),
            2 * n,
            2.0 * rate * n as f64
        );
        assert_eq!(pair.error().stuck_devices, pair.stuck_devices());
    }

    /// (A3) [`DeviceModel::validate`] is an EXACT CENSUS of all seven sub-models, in the same shape
    /// as `the_weakest_evidence_is_an_exact_census_of_every_enabled_model` — and for the same
    /// reason, because the same fold-with-a-missing-branch defect was found and repaired there and
    /// the lesson was not carried across. Five of the seven branches could be deleted individually
    /// with the suite green.
    ///
    /// Every row installs a **struct literal** the sub-model's own constructor would have refused.
    /// That is the only case `validate` is for: `bad_inputs_are_refused_by_name` exercises the
    /// constructors, which cannot reach a `DeviceModel` that carries a bad sub-model, and
    /// [`Window::validate`]'s doc states the rationale — the fields are public.
    ///
    /// Each row names a DIFFERENT error variant or field, so deleting any one branch fails that row
    /// and no other.
    #[test]
    fn every_sub_model_is_validated_by_an_exact_census() {
        let base = || {
            let mut d = DeviceModel::ideal();
            d.window = Window::new(1e-6, 100e-6, "test", Evidence::Unstated).unwrap();
            d.beta = 1e-6;
            d.levels = Some(Levels::new(32, "test", Evidence::Unstated).unwrap());
            d.variability = Some(Variability::new(0.01, 0.01, 0.0, "test", Evidence::Unstated).unwrap());
            d.stuck = Some(StuckAt::new(0.01, 0.5, "test", Evidence::Unstated).unwrap());
            d.drift = Some(Drift::uniform(0.01, 1.0, "test", Evidence::Unstated).unwrap());
            d.retention = Some(Retention::new(1.0, 1e-12, "test", Evidence::Unstated).unwrap());
            d.read_noise = Some(ReadNoise::new(300.0, 1e6, 0.0, "test", Evidence::Unstated).unwrap());
            d
        };
        // The base has to be sound, or every row below passes for the wrong reason.
        assert!(base().validate().is_ok());
        assert!(base().apply(&[1.0], &mut Rng::new(1)).is_ok());
        assert_eq!(base().enabled().len(), 6);

        /// One row of the census: a name, the struct literal that breaks exactly one sub-model, and
        /// the refusal it must produce.
        type Break = (&'static str, fn(&mut DeviceModel), fn(&DeviceError) -> bool);
        let census: [Break; 8] = [
            (
                "window",
                |d| d.window = Window { g_off: 5.0, g_on: 1.0, ..d.window },
                |e| matches!(e, DeviceError::BadWindow { .. }),
            ),
            (
                "beta",
                |d| d.beta = -1.0,
                |e| matches!(e, DeviceError::NotPositive { what: "beta", .. }),
            ),
            (
                "variability",
                |d| d.variability = Some(Variability { sigma_d2d_rel: -1.0, ..d.variability.unwrap() }),
                |e| matches!(e, DeviceError::Negative { what: "sigma_d2d_rel", .. }),
            ),
            (
                "levels",
                |d| d.levels = Some(Levels { n: 0, ..d.levels.unwrap() }),
                |e| matches!(e, DeviceError::BadLevels { levels: 0 }),
            ),
            (
                "stuck",
                |d| d.stuck = Some(StuckAt { rate: 5.0, p_at_on: -2.0, ..d.stuck.unwrap() }),
                |e| matches!(e, DeviceError::BadProbability { what: "rate", .. }),
            ),
            (
                "drift",
                |d| d.drift = Some(Drift { t0_s: 0.0, ..d.drift.unwrap() }),
                |e| matches!(e, DeviceError::NotPositive { what: "t0_s", .. }),
            ),
            (
                "retention",
                |d| d.retention = Some(Retention { tau0_s: -1.0, ..d.retention.unwrap() }),
                |e| matches!(e, DeviceError::NotPositive { what: "tau0_s", .. }),
            ),
            (
                "read_noise",
                |d| d.read_noise = Some(ReadNoise { bandwidth_hz: 0.0, ..d.read_noise.unwrap() }),
                |e| matches!(e, DeviceError::NotPositive { what: "bandwidth_hz", .. }),
            ),
        ];
        for (name, break_it, expected) in census {
            let mut d = base();
            break_it(&mut d);
            let from_validate = d.validate().unwrap_err();
            assert!(
                expected(&from_validate),
                "{name} is not in DeviceModel::validate's census: it returned {from_validate}"
            );
            // And `apply` refuses rather than producing a plausible array. A zero-level grid is the
            // one that made this urgent: it programmed EVERY weight to the off rail and returned
            // `Ok`, with `bits()` of -inf available to print beside it.
            let from_apply = d.apply(&[1.0, -1.0], &mut Rng::new(1)).unwrap_err();
            assert!(
                expected(&from_apply),
                "{name} reached DeviceModel::apply and it returned {from_apply}"
            );
        }

        // The sub-model validators are public and refuse on their own terms too.
        assert!(matches!(
            Levels { n: 1, source: "one level is not a quantiser", evidence: Evidence::Unstated }
                .validate(),
            Err(DeviceError::BadLevels { levels: 1 })
        ));
        assert!(Levels::new(2, "the minimum", Evidence::Unstated).unwrap().validate().is_ok());
        // `bits()` stays exactly log2(n) — the refusal lives in `validate`, not in the arithmetic,
        // because a `bits()` that substituted n = 2 would report ONE bit for a cell with no states.
        assert_eq!(Levels { n: 1, source: "", evidence: Evidence::Unstated }.bits(), 0.0);
        assert!(Levels { n: 0, source: "", evidence: Evidence::Unstated }.bits().is_infinite());
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

    /// (A5) [`Drift::exponent_at`] had no test at all, and the [`Drift`] doc names FOUR properties
    /// the interpolation is "chosen because" it has. Two of them were asserted, inside
    /// `state_dependent_drift_does_not_cancel`, and two were prose.
    ///
    /// All four are here, each against a value written from the scheme rather than read back off
    /// the struct:
    ///
    /// * **monotone** — strictly, across the window, in the direction the two exponents set;
    /// * **reduces exactly to the uniform case** when the two are equal, `assert_eq!` on `f64`;
    /// * **cannot produce a negative exponent from two non-negative ones** — the clamp, which is
    ///   what makes this a fix and not a note. Without it [`Drift::apply`] returns a conductance
    ///   that GROWS with time for any cell above the window, which is the one outcome the doc says
    ///   the scheme cannot produce;
    /// * **strictly between the two uniform bounds** inside the window.
    #[test]
    fn the_drift_exponent_interpolates_monotonically_and_never_goes_negative() {
        let w = win();
        let (nu_off, nu_on) = (0.15f64, 0.03f64);
        let d = Drift::new(nu_off, nu_on, 1.0, "test", Evidence::Unstated).unwrap();

        // The two rails are the two exponents, and the midpoint is their mean — the scheme's own
        // arithmetic, written out here rather than recomputed by calling the same interpolation.
        assert_eq!(d.exponent_at(w.g_off, &w), nu_off);
        assert!((d.exponent_at(w.g_on, &w) - nu_on).abs() < 1e-15);
        assert!((d.exponent_at(w.mid(), &w) - 0.5 * (nu_off + nu_on)).abs() < 1e-15);
        // A quarter of the way up: 0.15 + 0.25 * (0.03 - 0.15) = 0.12.
        let quarter = w.g_off + 0.25 * w.span();
        assert!((d.exponent_at(quarter, &w) - 0.12).abs() < 1e-15, "{}", d.exponent_at(quarter, &w));

        // Monotone, strictly, and strictly between the bounds inside the window.
        let mut last = f64::INFINITY;
        for k in 0..=200 {
            let g = w.g_off + f64::from(k) / 200.0 * w.span();
            let nu = d.exponent_at(g, &w);
            assert!(nu < last, "exponent {nu} at {g} S did not fall below {last}");
            assert!((nu_on..=nu_off).contains(&nu), "exponent {nu} escaped [{nu_on}, {nu_off}]");
            last = nu;
        }

        // Equal exponents reduce to the single-exponent case EXACTLY, everywhere, including
        // outside the window where the clamp is doing the work.
        let u = Drift::uniform(0.07, 1.0, "test", Evidence::Unstated).unwrap();
        for f in [-5.0f64, -1.0, 0.0, 0.3, 1.0, 2.0, 100.0] {
            let g = w.g_off + f * w.span();
            assert_eq!(u.exponent_at(g, &w), 0.07, "uniform exponent moved at f = {f}");
        }

        // THE CLAMP. Held flat outside the window in both directions, and never negative.
        assert_eq!(d.exponent_at(w.g_off - 100.0 * w.span(), &w), d.exponent_at(w.g_off, &w));
        assert_eq!(d.exponent_at(w.g_on + 100.0 * w.span(), &w), d.exponent_at(w.g_on, &w));
        assert_eq!(d.exponent_at(-1.0, &w), nu_off);
        assert_eq!(d.exponent_at(1.0, &w), d.exponent_at(w.g_on, &w));

        // And the public consequence, which is the reason the clamp matters: unclamped, a cell
        // above the window gets 0.1 + 2*(0.02 - 0.1) = -0.06 from PCM_DRIFT_AMORPHOUS, and a
        // negative exponent means conductance that GROWS with time. Drift only ever loses.
        let pcm = PCM_DRIFT_AMORPHOUS;
        for f in [0.0f64, 0.5, 1.0, 1.25, 2.0, 10.0, 1e6] {
            let g = w.g_off + f * w.span();
            let nu = pcm.exponent_at(g, &w);
            assert!(nu >= 0.0, "a non-negative pair of exponents gave {nu} at f = {f}");
            let factor = pcm.factor(g, &w, 1e6);
            assert!(
                factor <= 1.0 && factor > 0.0,
                "drift multiplied a cell at f = {f} by {factor}, so conductance grew with time"
            );
            assert!(pcm.apply(g, &w, 1e6) <= g, "drift raised a conductance at f = {f}");
        }
        // The unclamped value, written out, so the size of what the clamp prevents is on the page.
        let unclamped = pcm.nu_at_off + 2.0 * (pcm.nu_at_on - pcm.nu_at_off);
        assert!(unclamped < 0.0 && (unclamped + 0.06).abs() < 1e-15, "{unclamped}");

        // A degenerate window falls back to the off-rail exponent rather than dividing by zero.
        let flat = Window { g_off: 1e-6, g_on: 1e-6, source: "degenerate", evidence: Evidence::Unstated };
        assert!(flat.validate().is_err());
        assert_eq!(d.exponent_at(1e-6, &flat), nu_off);
        assert_eq!(d.exponent_at(50e-6, &flat), nu_off);
    }

    /// (A16d) The module's stated discipline — [`Window::on_off_ratio`] and
    /// [`ReadNoise::distinguishable_levels`] refuse rather than reporting an infinity — applied to
    /// the two places that did not: a large activation energy at a low temperature overflows
    /// [`Retention::tau_s`], and a wide temperature ratio overflows
    /// [`Retention::acceleration`]. Both used to hand back `Ok(inf)` from **accepted** parameters.
    #[test]
    fn a_retention_model_refuses_an_infinite_lifetime_rather_than_reporting_one() {
        // 15 eV at 200 K is exp(870). The parameters are unphysical and they validate, which is
        // exactly the case: the refusal has to come from the result, not from the input.
        let r = Retention::new(15.0, 1e-12, "unphysical but accepted", Evidence::Unstated).unwrap();
        assert!(r.validate().is_ok());
        assert!(
            (15.0f64 / (BOLTZMANN_EV_PER_K * 200.0)).exp().is_infinite(),
            "this parameter set no longer overflows, so the test below asserts nothing"
        );
        assert!(matches!(
            r.tau_s(200.0),
            Err(DeviceError::NonFinite { what: "tau_s", .. })
        ));
        // And it propagates: a fraction computed from an infinite lifetime is exactly 1.0, which
        // reads as "nothing was lost" rather than as "this model ran out of room".
        assert!(matches!(
            r.remaining_fraction(1.0, 200.0),
            Err(DeviceError::NonFinite { what: "tau_s", .. })
        ));
        // The same model at a temperature it CAN hold still works, so this is a ceiling and not a
        // ban on large activation energies.
        assert!(r.tau_s(2_000.0).unwrap().is_finite());

        let one_ev = Retention::new(1.0, 1e-12, "test", Evidence::Unstated).unwrap();
        assert!(matches!(
            one_ev.acceleration(1.0, 1e30),
            Err(DeviceError::NonFinite { what: "acceleration", .. })
        ));
        // Underflow the other way is NOT refused: a zero acceleration factor is a small number the
        // arithmetic could not hold, which errs in the honest direction.
        assert_eq!(one_ev.acceleration(1e30, 1.0).unwrap(), 0.0);
        // The worked example still works, so the guard did not cost the ordinary case.
        assert!((one_ev.acceleration(358.15, 398.15).unwrap() - 25.93).abs() < 0.05);
        assert!(DeviceError::NonFinite { what: "tau_s", value: f64::INFINITY }
            .to_string()
            .contains("tau_s"));
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
        // TWO-SIDED, and it was not: written without the `.abs()` this line could not fail for any
        // change that made `si` LARGER — a factor of 8, an extra additive term, a bandwidth counted
        // twice — because the difference simply went more negative. It shares
        // `BOLTZMANN_J_PER_K` with the code under test, so it pins the SHAPE of the formula and the
        // hand-computed literal two lines above is the independent anchor for the constant.
        assert!(
            ((4.0 * BOLTZMANN_J_PER_K * 300.0 * g * 1e6).sqrt() - si).abs() < 1e-24,
            "the formula moved away from 4kTGB: {si} against {}",
            (4.0 * BOLTZMANN_J_PER_K * 300.0 * g * 1e6).sqrt()
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

        // THE SATURATION, and the three arithmetically different routes into the SAME physical
        // statement — the count is not bounded by the noise. Every one of these returned a number
        // before, and `Some(4294967295)` is the module's own "infinity printed as an integer"
        // under another name: `Levels::new` accepts it and reports 32 effective bits from a
        // resistive cell.
        //
        // The boundary is pinned from BOTH sides at a decade in bandwidth, so a `min` that
        // saturated, or a check that refused every large count, fails one of the two.
        let huge = ReadNoise::new(300.0, 1e-6, 0.0, "a 1 uHz read", Evidence::Derived).unwrap();
        let over = ReadNoise::new(300.0, 1e-7, 0.0, "a 0.1 uHz read", Evidence::Derived).unwrap();
        let counted = huge.distinguishable_levels(&w, 0.2, 6.0).unwrap();
        assert!(
            (2.0e9..3.0e9).contains(&f64::from(counted)),
            "a 1 uHz read left {counted} levels, expected about 2.56e9 and below u32::MAX"
        );
        assert!(f64::from(counted) < f64::from(u32::MAX), "{counted} is not below the ceiling");
        assert_eq!(
            over.distinguishable_levels(&w, 0.2, 6.0),
            None,
            "a count above u32::MAX came back as a saturated integer"
        );
        // `separation * sigma` below the arithmetic's reach: the quotient is an infinity, which
        // means UNBOUNDED. Routing it into `Some(1)` — no analog information at all — is the
        // numerical opposite of the truth, and is what this line refuses.
        let sigma = over.sigma_conductance(w.g_on, 0.2).unwrap();
        assert!(sigma > 0.0);
        assert_eq!(1e-320 * sigma, 0.0, "the product no longer underflows, so this asserts nothing");
        assert_eq!(over.distinguishable_levels(&w, 0.2, 1e-320), None);
        // The intermediate case, where the product is SUBNORMAL rather than zero and the quotient
        // overflows instead: also unbounded, also `None`, by the other branch.
        assert!(6e-300 * sigma > 0.0 && !(w.span() / (6e-300 * sigma)).is_finite());
        assert_eq!(over.distinguishable_levels(&w, 0.2, 6e-300), None);
        // A noise model that does not validate refuses here too, the same way an invalid window
        // does. It did not: `sigma_current` handed back `Some(NaN)` and the comparisons below it
        // all read false.
        let nonsense =
            ReadNoise { temp_k: -300.0, source: "negative kelvin", ..THERMAL_READ_NOISE_300K };
        assert!(nonsense.validate().is_err());
        assert_eq!(nonsense.distinguishable_levels(&w, 0.2, 6.0), None);
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

    /// (A7) [`ReadNoise::sigma_current`]'s doc promises `None` "for a negative conductance or a
    /// non-finite argument, because neither has a noise power". A negative TEMPERATURE has no noise
    /// power for the identical reason, and it produced `Some(NaN)` — which
    /// [`Programmed::read`] then handed back as `Ok([NaN, NaN])`.
    ///
    /// The inconsistency was inside one struct: [`ReadNoise::distinguishable_levels`] already
    /// refused the same model, through `!(sigma > 0.0)` being false for a `NaN`. One entry point
    /// treated it as a refusal case and three propagated it.
    #[test]
    fn a_read_noise_model_that_does_not_validate_refuses_instead_of_returning_a_nan() {
        let w = win();
        let cold = ReadNoise {
            temp_k: -300.0,
            source: "a negative kelvin, which no read path has",
            ..THERMAL_READ_NOISE_300K
        };
        assert!(cold.validate().is_err());
        // The NaN this refusal replaces, computed here so the failure mode is on the page.
        let thermal = 4.0 * BOLTZMANN_J_PER_K * cold.temp_k * 1e-6 * cold.bandwidth_hz;
        assert!(thermal < 0.0 && thermal.sqrt().is_nan(), "the square root of {thermal}");

        assert_eq!(cold.sigma_current(1e-6, 0.2), None);
        assert_eq!(cold.sigma_conductance(1e-6, 0.2), None);
        assert_eq!(cold.sample(&mut Rng::new(1), 1e-6, 0.2), None);
        assert_eq!(cold.distinguishable_levels(&w, 0.2, 6.0), None);
        // A refused read leaves the stream where it was, which is the documented property of
        // `sample` and is the one a caller cannot see in the answer.
        let mut rng = Rng::new(7);
        let before = rng;
        assert_eq!(cold.sample(&mut rng, 1e-6, 0.2), None);
        assert_eq!(rng, before, "a refused read consumed randomness");

        // Every field `validate` covers, so the refusal is the whole census and not one branch.
        for bad in [
            ReadNoise { temp_k: 0.0, ..THERMAL_READ_NOISE_300K },
            ReadNoise { temp_k: f64::NAN, ..THERMAL_READ_NOISE_300K },
            ReadNoise { bandwidth_hz: 0.0, ..THERMAL_READ_NOISE_300K },
            ReadNoise { bandwidth_hz: f64::INFINITY, ..THERMAL_READ_NOISE_300K },
            ReadNoise { sigma_rel: -0.01, ..THERMAL_READ_NOISE_300K },
            ReadNoise { sigma_rel: f64::NAN, ..THERMAL_READ_NOISE_300K },
        ] {
            assert!(bad.validate().is_err(), "{bad:?} validated");
            assert_eq!(bad.sigma_current(1e-6, 0.2), None, "{bad:?} produced a sigma");
        }
        // And the sound model still answers, so this is a refusal and not a blanket None.
        assert!(THERMAL_READ_NOISE_300K.sigma_current(1e-6, 0.2).unwrap() > 0.0);

        // `Programmed::model` is a public field, so a caller can install one after programming.
        // The read refuses by name instead of returning an array of NaN.
        let mut model = DeviceModel::ideal();
        model.window = w;
        model.beta = 1e-6;
        let mut held = model.apply(&[10.0, 20.0], &mut Rng::new(1)).unwrap();
        held.model.read_noise = Some(cold);
        assert!(matches!(
            held.read(&mut Rng::new(1), 0.2),
            Err(DeviceError::NotPositive { what: "temp_k", .. })
        ));
        // A sound model on the same array reads fine.
        held.model.read_noise = Some(THERMAL_READ_NOISE_300K);
        assert!(held.read(&mut Rng::new(1), 0.2).unwrap().iter().all(|x| x.is_finite()));
    }

    /// (A11) The invariant every mechanism in this module is supposed to respect and which was
    /// asserted nowhere: **a programmed conductance lies inside the window**. [`Window::contains`]
    /// is the predicate that says so and had no caller at all — its body could be replaced with
    /// `true`.
    ///
    /// Two things ride on it. [`Window::mid`] could be mutated to `0.5 * g_on` with the suite
    /// green, because [`Mapping::read`] SUBTRACTS the two devices and the midpoint cancels in every
    /// answer — under that mutation a full-scale negative balanced weight is written BELOW `g_off`
    /// and nothing notices. And [`Levels::snap`]'s final clamp could be dropped, because at 18
    /// levels on this window `g_off + 17 * step` overshoots `g_on` by one unit in the last place.
    #[test]
    fn every_programmed_conductance_lies_inside_the_window() {
        let w = win();
        // The predicate itself, in both directions and at both rails.
        assert!(w.contains(w.g_off) && w.contains(w.g_on) && w.contains(w.mid()));
        assert!(!w.contains(w.g_off - 1e-18), "contains() accepted a conductance below the floor");
        assert!(!w.contains(w.g_on + 1e-18), "contains() accepted a conductance above the ceiling");
        assert!(!w.contains(f64::NAN) && !w.contains(f64::INFINITY));

        // The midpoint, as a literal. 0.5 * (1 uS + 100 uS) = 50.5 uS.
        assert_eq!(w.mid(), 50.5e-6);
        assert_eq!(Window::new(1.0, 5.0, "binary", Evidence::Unstated).unwrap().mid(), 3.0);
        // ... and the identity the BalancedDifferential doc derives its common mode from: "a zero
        // weight here holds both devices at mid". The hard-coded `g_off + g_on` has to be 2 * mid.
        assert_eq!(Mapping::BalancedDifferential.common_mode_conductance(&w), 2.0 * w.mid());
        assert_eq!(Mapping::BalancedDifferential.program(0.0, &w, 1e-6), (w.mid(), w.mid()));
        // A full-scale weight lands exactly on the rails, not past them — in BOTH directions,
        // because the escape is in `g_minus` at `+full` and in `g_plus` at `-full`, so testing one
        // sign leaves the other unguarded. With mid() wrong by half a span these land at 0.5 uS,
        // under g_off, and no read can tell because the pair subtracts.
        let full = Mapping::BalancedDifferential.max_weight(&w, 1e-6);
        for signed in [full, -full] {
            let (gp, gm) = Mapping::BalancedDifferential.program(signed, &w, 1e-6);
            assert!(
                w.contains(gp) && w.contains(gm),
                "a weight of {signed} programmed to ({gp}, {gm}), outside [{}, {}]",
                w.g_off,
                w.g_on
            );
            let (lo, hi) = if signed > 0.0 { (gm, gp) } else { (gp, gm) };
            assert_eq!(lo, w.g_off, "the low device of a full-scale pair is not on the floor");
            assert_eq!(hi, w.g_on, "the high device of a full-scale pair is not on the ceiling");
        }
        // The same at full scale for the other two schemes, so the clamp covers all three.
        for m in [Mapping::SingleEnded, Mapping::Differential] {
            for signed in [full, m.min_weight(&w, 1e-6)] {
                let (gp, gm) = m.program(signed, &w, 1e-6);
                assert!(w.contains(gp) && w.contains(gm), "{m:?} at {signed}: ({gp}, {gm})");
            }
        }

        // The invariant itself, over all three mappings with every per-cell mechanism on at once.
        // Drift and retention are deliberately excluded: `Drift::apply` is documented as NOT
        // clamped, because carrying a cell below its off-state is the failure retention
        // engineering exists to bound.
        let mut model = DeviceModel::ideal();
        model.window = w;
        model.beta = 1e-6;
        model.d2d_seed = 4242;
        model.levels = Some(Levels::new(18, "the awkward count below", Evidence::Unstated).unwrap());
        model.variability =
            Some(Variability::new(0.20, 0.10, 5e-6, "all three terms", Evidence::Unstated).unwrap());
        model.stuck = Some(StuckAt::new(0.05, 0.5, "test", Evidence::Unstated).unwrap());
        let targets: Vec<f64> = (0..500).map(|k| (f64::from(k) - 250.0) * 0.39).collect();
        for m in [Mapping::SingleEnded, Mapping::Differential, Mapping::BalancedDifferential] {
            model.mapping = m;
            let inside: Vec<f64> =
                targets.iter().copied().filter(|t| *t >= model.min_weight()).collect();
            let held = model.apply(&inside, &mut Rng::new(88)).unwrap();
            assert!(held.clamped > 0, "{m:?}: nothing clamped, so the rails are untested here");
            for (k, (&gp, &gm)) in held.g_plus.iter().zip(&held.g_minus).enumerate() {
                assert!(w.contains(gp), "{m:?} cell {k} plus at {gp} S is outside the window");
                assert!(w.contains(gm), "{m:?} cell {k} minus at {gm} S is outside the window");
            }
        }

        // `Levels::snap`'s final clamp, and the arithmetic that makes it load-bearing.
        let l = Levels::new(18, "test", Evidence::Unstated).unwrap();
        let top = w.g_off + f64::from(l.n - 1) * l.step(&w);
        assert!(
            top > w.g_on,
            "18 levels no longer overshoot this window, so the clamp below is untested: \
             {top:e} against {:e}",
            w.g_on
        );
        assert_eq!(l.snap(w.g_on, &w), w.g_on, "the top level was written above the ceiling");
        assert!(w.contains(l.snap(w.g_on, &w)));
        assert!(w.contains(l.snap(1.0, &w)) && w.contains(l.snap(-1.0, &w)));
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
        // And a realistic wire is not a rounding error: at TEN ohms this 6x6 array is off by
        // 0.76%, which is the point of the module. Both numbers here are the numbers in the code
        // below — the comment used to say five ohms and a percent, and the code said ten ohms and
        // a tenth of a percent, so neither figure a reader took away was the one being asserted.
        //
        // Bounded on BOTH sides, because `> 1e-3` is satisfied by an error an order of magnitude
        // too large just as happily as by the right one, and the wrong topologies this whole
        // section exists to catch overstate the drop as readily as they understate it.
        let xb =
            Crossbar::new(n, m, g, Wires::new(10.0, "test", Evidence::Unstated).unwrap()).unwrap();
        let ideal = xb.ideal_currents(&v).unwrap();
        let s = xb.solve(&v, 1e-10, 200_000).unwrap();
        let err = s.max_relative_error(&ideal).unwrap();
        assert!(
            (7.0e-3..8.5e-3).contains(&err),
            "a 10 ohm 6x6 array came out {err} off the ideal, not the 0.76% this array has"
        );
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

    /// (A8) THE WIRE-DOMINATED REGIME, which this module's whole thesis is about and which no test
    /// entered: every other crossbar test sits at `2*r*g <= 2e-5`, where wires are negligible.
    ///
    /// Here an 8x8 array of `1e-2 S` cells on `100 ohm` wires delivers a **twentieth** of its ideal
    /// current. `tol` used to be measured against the IDEAL column current — the quantity this
    /// module exists to show is wrong — so the doc's "`1e-9` means nine digits of the answer" was
    /// overstated by the ratio of the two, which is unbounded and is largest exactly where a caller
    /// most needs the bound.
    #[test]
    fn the_solve_tolerance_is_relative_to_the_current_it_actually_returns() {
        let n = 8;
        let xb = Crossbar::new(
            n,
            n,
            vec![1e-2; n * n],
            Wires::new(100.0, "test", Evidence::Unstated).unwrap(),
        )
        .unwrap();
        let v = vec![0.2; n];
        let ideal = xb.ideal_currents(&v).unwrap();
        let tol = 1e-9;
        let s = xb.solve(&v, tol, 200_000).unwrap();

        // The regime, asserted: the wires cost 98% of the answer. That is correct physics for this
        // array and it is what makes the two normalisations separable at all.
        let err = s.max_relative_error(&ideal).unwrap();
        assert!((0.95..1.0).contains(&err), "the wires cost {err} of the ideal current, not ~0.99");
        let returned = s.column_currents.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
        let want = ideal.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
        assert!(returned < 0.1 * want, "returned {returned} A against an ideal {want} A");

        // THE CLAIM: nine digits of the answer, where the answer is what came back.
        assert!(
            s.residual_a <= tol * returned,
            "residual {} A against {} A, which is tol times the current actually returned",
            s.residual_a,
            tol * returned
        );
        // And the old normalisation would have stopped an order of magnitude earlier, so the
        // assertion above is not satisfied by both.
        assert!(
            tol * want > 10.0 * s.residual_a,
            "normalising by the ideal current is too close to normalising by the real one here"
        );

        // The unbounded version of the same overstatement. At `g = 1e100 S` on `1e-6 ohm` wires the
        // ideal current is 4e299 A and the array delivers 2e-89 A, so a target of `1e-9 * 4e299`
        // was met on the FIRST sweep with a Kirchhoff residual of 200 kA. Refused now.
        let stiff = Crossbar::new(
            2,
            2,
            vec![1e100; 4],
            Wires::new(1e-6, "test", Evidence::Unstated).unwrap(),
        )
        .unwrap();
        match stiff.solve(&[0.2, 0.2], tol, 10) {
            Err(DeviceError::NotConverged { residual_a, target_a, sweeps }) => {
                assert_eq!(sweeps, 10);
                assert!(residual_a > target_a, "{residual_a} A against a target of {target_a} A");
                assert!(target_a < 1e-6, "the target was {target_a} A, which is not a tolerance");
            }
            other => panic!("a solve normalised by the ideal current returned {other:?}"),
        }
    }

    /// (A14) [`Solution::sweeps`] is a reported statistic and the only assertions on it were `== 0`
    /// on the closed-form branch and `== 1` inside a [`DeviceError::NotConverged`] — never on an
    /// iterative `Solution`, where `sweep + 1` passed the suite.
    ///
    /// Pinned two-sidedly, and without hard-coding a count: the number reported is the smallest
    /// budget that converges. One fewer must fail and exactly that many must succeed, which an
    /// off-by-one in either direction breaks. The iteration is `+ - * /` only, so this is exact on
    /// every platform.
    #[test]
    fn the_reported_sweep_count_is_the_budget_the_solve_actually_needed() {
        let n = 8;
        let xb = Crossbar::new(
            n,
            n,
            vec![1e-2; n * n],
            Wires::new(100.0, "test", Evidence::Unstated).unwrap(),
        )
        .unwrap();
        let v = vec![0.2; n];
        let tol = 1e-9;
        let s = xb.solve(&v, tol, 200_000).unwrap();
        assert!(s.sweeps >= 2, "this array converged in {} sweeps, too few to pin", s.sweeps);
        assert!(
            xb.solve(&v, tol, s.sweeps).is_ok(),
            "the {} sweeps reported were not enough to converge",
            s.sweeps
        );
        assert!(
            xb.solve(&v, tol, s.sweeps - 1).is_err(),
            "it converged in fewer than the {} sweeps it reported",
            s.sweeps
        );
        // The closed-form branch reports zero and does not iterate at all.
        let ideal_wires = Crossbar::new(n, n, vec![1e-2; n * n], Wires::ideal()).unwrap();
        assert_eq!(ideal_wires.solve(&v, tol, 1).unwrap().sweeps, 0);
    }

    /// (A10) [`Solution::max_relative_error`]'s documented normalisation — by the largest IDEAL
    /// current over the array, not per column — with the near-zero column the doc argues about and
    /// which no crossbar test constructs, since every one of them drives columns of similar
    /// magnitude.
    #[test]
    fn max_relative_error_normalises_by_the_array_and_not_by_the_column() {
        let one = |c: Vec<f64>| Solution {
            column_currents: c,
            cell_voltage: Vec::new(),
            sweeps: 3,
            residual_a: 0.0,
        };
        // One column carries the array; the other is a picoamp away from nothing. The second
        // column is wrong by 100% of its own ideal current and by 4e-12 of the array's.
        let s = one(vec![0.75, 0.0]);
        assert_eq!(
            s.max_relative_error(&[1.0, 1e-12]),
            Some(0.25),
            "per-column normalisation would report 1.0 here and say nothing about the array"
        );
        // The scale is the largest ABSOLUTE ideal current, so a negative column can set it.
        assert_eq!(one(vec![0.0, 0.0]).max_relative_error(&[-4.0, 1.0]), Some(1.0));
        // A perfect solve is exactly zero, not a rounding of it.
        assert_eq!(one(vec![1.0, 1e-12]).max_relative_error(&[1.0, 1e-12]), Some(0.0));
        // Refusals: a length mismatch, an empty array, and an ideal with no current in it at all —
        // where a relative error is undefined rather than large.
        assert_eq!(s.max_relative_error(&[1.0]), None);
        assert_eq!(s.max_relative_error(&[]), None);
        assert_eq!(one(Vec::new()).max_relative_error(&[]), None);
        assert_eq!(one(vec![1e-9, 1e-9]).max_relative_error(&[0.0, 0.0]), None);
        assert_eq!(one(vec![0.0, 0.0]).max_relative_error(&[0.0, -0.0]), None);
    }

    /// (A15) [`DeviceError::SingularLine`] is the module's own "visible rather than silent" branch
    /// and nothing reached it, so the guard that produces it could be deleted with the suite green.
    ///
    /// It IS reachable from the public API, through a **subnormal** wire resistance: `Wires::new`
    /// accepts it — finite and non-negative — and its reciprocal is an infinity, so every
    /// tridiagonal diagonal is infinite. Note that [`f64::MIN_POSITIVE`] is NOT such a case; it is
    /// the smallest NORMAL double and `1 / it` is `4.49e307`, which solves fine.
    ///
    /// The back-substitution guard and the residual's `NaN` handling are checked on the helpers
    /// directly, because no crossbar this module builds can reach them and the alternative is
    /// carrying two branches nobody has ever run.
    #[test]
    fn the_line_solve_refuses_a_degenerate_pivot_rather_than_producing_infinities() {
        let subnormal = Wires::new(1e-310, "a subnormal segment", Evidence::Unstated).unwrap();
        assert!(!(1.0 / subnormal.r_segment_ohm).is_finite(), "1e-310 is no longer subnormal here");
        let xb = Crossbar::new(2, 2, vec![1e-6; 4], subnormal).unwrap();
        assert!(matches!(
            xb.solve(&[0.2, 0.2], 1e-9, 10),
            Err(DeviceError::SingularLine),
            ));
        // The smallest NORMAL resistance still solves, so this is a guard and not a size limit.
        assert!((1.0f64 / f64::MIN_POSITIVE).is_finite());
        let tiny = Wires::new(f64::MIN_POSITIVE, "the smallest normal", Evidence::Unstated).unwrap();
        assert!(Crossbar::new(2, 2, vec![1e-6; 4], tiny).unwrap().solve(&[0.2, 0.2], 1e-9, 10).is_ok());

        // The Thomas solve itself. A well-posed system first, so the false cases below are not
        // false for every input.
        let (mut out, mut c) = ([0.0f64; 3], [0.0f64; 3]);
        assert!(thomas(
            &[0.0, -1.0, -1.0],
            &[2.0, 2.0, 2.0],
            &[-1.0, -1.0, 0.0],
            &[1.0, 0.0, 1.0],
            &mut out,
            &mut c
        ));
        assert!(out.iter().all(|v| v.is_finite()));
        // A non-finite right-hand side: every pivot stays sound and the OUTPUT is not finite, which
        // is the case only the final `all(is_finite)` catches.
        assert!(!thomas(
            &[0.0, -1.0, -1.0],
            &[2.0, 2.0, 2.0],
            &[-1.0, -1.0, 0.0],
            &[f64::INFINITY, 0.0, 0.0],
            &mut out,
            &mut c
        ));
        // A zero leading pivot, a non-finite one, an interior pivot that cancels to zero, and an
        // empty system.
        assert!(!thomas(&[0.0], &[0.0], &[0.0], &[1.0], &mut out[..1], &mut c[..1]));
        assert!(!thomas(&[0.0], &[f64::NAN], &[0.0], &[1.0], &mut out[..1], &mut c[..1]));
        // An INFINITE leading diagonal is the one case the leading-pivot check catches on its own:
        // every other degenerate pivot poisons the output and is caught downstream, but
        // `sup[0] / inf` and `rhs[0] / inf` are both a perfectly finite 0.0, so without this guard
        // the solve returns `true` with a node voltage of zero and no complaint at all.
        assert!(!thomas(&[0.0], &[f64::INFINITY], &[0.0], &[1.0], &mut out[..1], &mut c[..1]));
        assert_eq!(1.0f64 / f64::INFINITY, 0.0, "the silent answer the guard above prevents");
        assert!(!thomas(
            &[0.0, -1.0],
            &[1.0, 1.0],
            &[-1.0, 0.0],
            &[1.0, 1.0],
            &mut out[..2],
            &mut c[..2]
        ));
        // An INFINITE interior pivot, which is the interior check's own version of the case above:
        // `sup/inf` and `(rhs - sub*out)/inf` are both finite zeros, the back substitution stays
        // finite, and without this guard the solve returns `true` with `out = [1, 0]` — a
        // plausible-looking node voltage for a line that has no solution.
        assert!(!thomas(
            &[0.0, -1.0],
            &[1.0, f64::INFINITY],
            &[-1.0, 0.0],
            &[1.0, 1.0],
            &mut out[..2],
            &mut c[..2]
        ));
        assert!(!thomas(&[], &[], &[], &[], &mut out[..0], &mut c[..0]));

        // And the residual, which folded with `f64::max` and therefore DROPPED a NaN: an all-NaN
        // node-voltage solution was reported as converged with a residual of exactly 0.0.
        assert!(!(0.0f64).max(f64::NAN).is_nan(), "f64::max no longer drops NaN");
        let line = Crossbar::new(1, 2, vec![1e-6, 1e-6], Wires::new(1.0, "test", Evidence::Unstated).unwrap())
            .unwrap();
        let ok = line.residual(&[0.2, 0.2], &[0.0, 0.0], &[0.2], 1.0);
        assert!(ok.is_finite() && ok > 0.0, "the sound case gave {ok}");

        // Both node sets are in the fold, checked at exactly zero conductance so the two residuals
        // decouple: with `g = 0` the word-line residual is `gw*(v_in - a)` alone and the bit-line
        // residual is `-gw*b` alone. A fold that dropped either set would report 0.0 for one of
        // these two — and `Crossbar::solve` calls this right after a bit-line pass, which leaves
        // the bit-line equations satisfied by construction, so nothing downstream can see it.
        let open = Crossbar::new(1, 1, vec![0.0], Wires::new(1.0, "test", Evidence::Unstated).unwrap())
            .unwrap();
        assert_eq!(open.residual(&[0.2], &[0.05], &[0.2], 1.0), 0.05, "the bit-line nodes are not folded in");
        assert_eq!(open.residual(&[0.1], &[0.0], &[0.2], 1.0), 0.1, "the word-line nodes are not folded in");
        assert_eq!(open.residual(&[0.2], &[0.0], &[0.2], 1.0), 0.0, "a satisfied array has no residual");
        assert_eq!(line.residual(&[f64::NAN, f64::NAN], &[0.0, 0.0], &[0.2], 1.0), f64::INFINITY);
        assert_eq!(line.residual(&[0.2, 0.2], &[f64::NAN, 0.0], &[0.2], 1.0), f64::INFINITY);
        // An OVERFLOW needs no guard and gets one anyway: `f64::max` propagates an infinity, so
        // these two would come back as `INFINITY` from the fold alone. They are here to record
        // which half of the problem the guard is actually for — the `NaN` above, not these.
        let mx = f64::MAX;
        assert_eq!(line.residual(&[-mx, mx], &[0.0, 0.0], &[0.2], 1.0), f64::INFINITY);
        let col = Crossbar::new(2, 1, vec![1e-6, 1e-6], Wires::new(1.0, "test", Evidence::Unstated).unwrap())
            .unwrap();
        assert!(col.residual(&[0.0, 0.0], &[1e-3, -1e-3], &[0.2, 0.2], 1.0).is_finite());
        assert_eq!(col.residual(&[0.0, 0.0], &[mx, -mx], &[0.2, 0.2], 1.0), f64::INFINITY);
        assert_eq!(0.0f64.max(f64::INFINITY), f64::INFINITY, "an infinity needs no guard");
    }

    /// (A16e) The one non-ideality that lives outside [`DeviceModel`]'s evidence fold carries a
    /// grade a caller can actually read. [`Wires::evidence`] was carried, round-tripped through
    /// [`Wires::new`] inside [`Crossbar::new`], and never read by anything — so the provenance of
    /// the mechanism the module doc singles out as most often omitted was the only one that was
    /// structurally unreportable.
    #[test]
    fn a_crossbar_reports_the_grade_of_the_wire_model_it_was_built_with() {
        let wires = Wires::new(2.0, "a stated metal stack", Evidence::Simulated).unwrap();
        let xb = Crossbar::new(2, 2, vec![1e-6; 4], wires).unwrap();
        assert_eq!(xb.weakest_evidence(), Evidence::Simulated);
        assert_eq!(xb.wires().r_segment_ohm, 2.0);
        assert_eq!(xb.wires().source, "a stated metal stack");
        assert_eq!(xb.wires().evidence, Evidence::Simulated);
        // The reference case describes no array and says so.
        let ideal = Crossbar::new(1, 1, vec![1e-6], Wires::ideal()).unwrap();
        assert_eq!(ideal.weakest_evidence(), Evidence::Unstated);
        assert_eq!(ideal.wires().r_segment_ohm, 0.0);

        // And it is genuinely OUTSIDE `DeviceModel::weakest_evidence`, which is why it needs its
        // own accessor: a `Measured` device model on a `Simulated` crossbar still grades `Measured`
        // through the model, and a caller reporting an IR-drop figure has to take the weaker of the
        // two themselves.
        let mut model = DeviceModel::ideal();
        model.window = Window::new(1e-6, 100e-6, "test", Evidence::Measured).unwrap();
        model.beta = 1e-6;
        assert_eq!(model.weakest_evidence(), Evidence::Measured);
        let held = model.apply(&[1.0, 2.0, 3.0, 4.0], &mut Rng::new(1)).unwrap();
        let from_array = held.crossbar(2, 2, true, Wires::new(2.0, "stack", Evidence::Simulated).unwrap()).unwrap();
        assert_eq!(from_array.weakest_evidence(), Evidence::Simulated);
        assert_eq!(
            crate::ledger::weaker(model.weakest_evidence(), from_array.weakest_evidence()),
            Evidence::Simulated
        );
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

        // Every published value, as a literal, so this crate cannot move one under a user without
        // a test saying so. These four constants are the only numbers this module exports and a
        // caller's result changes silently if any of them drifts.
        assert_eq!(PCM_DRIFT_AMORPHOUS.nu_at_off, 0.1);
        assert_eq!(PCM_DRIFT_AMORPHOUS.nu_at_on, 0.02);
        assert_eq!(PCM_DRIFT_AMORPHOUS.t0_s, 1.0);
        // The physics behind the two numbers, at compile time: the amorphous (high-resistance)
        // state drifts FASTER, which is the reason `Drift` carries two exponents at all.
        const { assert!(PCM_DRIFT_AMORPHOUS.nu_at_off > PCM_DRIFT_AMORPHOUS.nu_at_on) };
        assert_eq!(RRAM_VARIABILITY_PLACEHOLDER.sigma_d2d_rel, 0.10);
        assert_eq!(RRAM_VARIABILITY_PLACEHOLDER.sigma_c2c_rel, 0.05);
        assert_eq!(RRAM_VARIABILITY_PLACEHOLDER.sigma_floor_s, 0.0);
        assert_eq!(RRAM_STUCK_AT_PLACEHOLDER.rate, 0.10);
        // 0.5 is this crate's NEUTRAL convention and is not the cited work's split, which is
        // markedly asymmetric. The constant's own doc says so; this line is what makes a silent
        // change to it visible, in either direction.
        assert_eq!(RRAM_STUCK_AT_PLACEHOLDER.p_at_on, 0.5);
        assert!(RRAM_STUCK_AT_PLACEHOLDER.source.contains("convention"));
        assert_eq!(THERMAL_READ_NOISE_300K.temp_k, 300.0);
        assert_eq!(THERMAL_READ_NOISE_300K.bandwidth_hz, 1e6);
        assert_eq!(THERMAL_READ_NOISE_300K.sigma_rel, 0.0);
        // And all four validate, which a struct-literal constant is not otherwise checked for.
        assert!(PCM_DRIFT_AMORPHOUS.validate().is_ok());
        assert!(RRAM_VARIABILITY_PLACEHOLDER.validate().is_ok());
        assert!(RRAM_STUCK_AT_PLACEHOLDER.validate().is_ok());
        assert!(THERMAL_READ_NOISE_300K.validate().is_ok());
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

    // ---------------------------------------------------------------- the third pass
    //
    // Twenty-four more mutations survived a third attack, and not one of them was an error in the
    // arithmetic. Every one was a guard, a seed, a bookkeeping field or a refusal that the suite
    // computed and then never read — the same shape as the second pass, one layer further out.
    // Three groups are worth naming, because they are three different ways of not being looked at:
    //
    // * **A validator's census with a row nobody breaks.** `Variability::sigma_floor_s` and
    //   `Drift::nu_at_on` could each be replaced by a duplicate of the field beside them, because
    //   every bad fixture in this module breaks the FIRST row of its table. Same for
    //   `Retention::ea_ev` below zero, the stress temperature of `Retention::acceleration`, and an
    //   infinite `Window::g_on` or `Wires::r_segment_ohm`.
    // * **A distinction between two refusals.** `check_temperature` returns `NonFinite` for a
    //   `NaN` and `NotPositive` for a zero, and `!(NaN > 0.0)` is true, so deleting the first
    //   branch returned the wrong variant for a `NaN` and accepted an INFINITE temperature
    //   outright — with every test asserting only `is_err()`.
    // * **State that no answer depends on.** The array seed could stop reaching its per-cell
    //   streams, the two devices of a pair could share one fault draw, `Programmed::age_s` and
    //   `Programmed::aged_at_k` could be written with anything at all, and a refused line solve
    //   could fill the caller's buffers with infinities. None of those moves a number the suite
    //   compares.

    /// The Box-Muller radius is `sqrt(-2 ln(1 - u))` and not `sqrt(-2 ln u)`, pinned by
    /// re-deriving every draw from the same stream rather than by sampling it.
    ///
    /// The hole is that `u` and `1 - u` have the **same distribution** on `Rng::next_f64`'s grid,
    /// so every test this module has of the generator — the Gaussian quantile test, the closed-form
    /// spreads in `device_to_device_variability_is_unbiased_and_has_the_stated_sigma`, the sampled
    /// read noise, the `normal`-against-`normal` stream comparisons — is blind to the reflection by
    /// construction. What the reflection prevents is `ln(0)`: `Rng::next_f64` returns `[0, 1)`, so
    /// `u` **can** be exactly zero and the radius is then an infinity. That draw has probability
    /// `2^-53` and cannot be sampled at any test length; only a per-draw re-derivation can see
    /// which of the two expressions is in the code.
    ///
    /// The two sides are the same operations on the same uniforms in the same order, so this is
    /// `assert_eq!` and not a tolerance. It also pins the two-uniforms-per-draw consumption, since
    /// any other count desynchronises the two streams, and the cosine branch, since the sine of the
    /// same phase is a different number.
    #[test]
    fn the_box_muller_radius_is_taken_from_the_reflected_uniform_so_its_logarithm_is_never_of_zero()
    {
        // The hazard, written as arithmetic because it cannot be drawn.
        assert_eq!(0.0f64.ln(), f64::NEG_INFINITY, "ln(0) is the infinity the reflection avoids");
        assert!((1.0f64 - 0.0).ln().is_finite(), "1 - u lies in (0, 1] and its logarithm is finite");

        let mut raw = Rng::new(2718);
        let mut drawn = Rng::new(2718);
        let mut reflected = 0u32;
        for k in 0..256 {
            let u1 = raw.next_f64();
            let u2 = raw.next_f64();
            // Written as a bound phase rather than inline so that this expression is not a second
            // copy of the one in `normal`: the mutation list anchors on that text.
            let phase = core::f64::consts::TAU * u2;
            let want = (-2.0 * (1.0 - u1).ln()).sqrt() * phase.cos();
            let got = normal(&mut drawn);
            assert_eq!(got, want, "draw {k} is not sqrt(-2 ln(1-u1)) * cos(tau*u2)");
            if u1 != 1.0 - u1 {
                reflected += 1;
            }
        }
        // Guard against the vacuous version of the assertion above: a stream of 0.5s would satisfy
        // it for either expression, because 0.5 is its own reflection.
        assert!(reflected > 250, "only {reflected} of 256 draws had u1 != 1 - u1");
    }

    /// `Window::validate`'s stated invariant is `0 <= g_off < g_on` with **both finite**, and the
    /// finiteness half had no test of its own: every bad window in this module is crossed or
    /// negative, and an infinite `g_on` is neither. Weakening the two `is_finite` calls to
    /// `is_nan` leaves exactly one case open — a finite non-negative `g_off` under an infinite
    /// `g_on` — and that window has an infinite `Window::span`, so `DeviceModel::max_weight` is
    /// infinite, every weight a caller offers is inside the representable range, and
    /// `DeviceModel::apply` programs the array and reports no error at all.
    #[test]
    fn an_infinite_on_conductance_is_not_a_window() {
        for (g_off, g_on) in [(1e-6, f64::INFINITY), (0.0, f64::INFINITY)] {
            assert!(
                matches!(
                    Window::new(g_off, g_on, "an unbounded on state", Evidence::Unstated),
                    Err(DeviceError::BadWindow { .. })
                ),
                "({g_off}, {g_on}) was accepted as a window"
            );
        }
        assert!(matches!(
            Window::new(f64::NEG_INFINITY, 1e-6, "", Evidence::Unstated),
            Err(DeviceError::BadWindow { .. })
        ));
        assert!(matches!(
            Window::new(f64::NAN, 1e-6, "", Evidence::Unstated),
            Err(DeviceError::BadWindow { .. })
        ));
        assert!(matches!(
            Window::new(1e-6, f64::NAN, "", Evidence::Unstated),
            Err(DeviceError::BadWindow { .. })
        ));

        // The struct-literal route, which is the case `Window::validate` exists for, and what the
        // refusal prevents further down.
        let unbounded = Window {
            g_off: 1e-6,
            g_on: f64::INFINITY,
            source: "literal",
            evidence: Evidence::Unstated,
        };
        assert!(unbounded.validate().is_err());
        assert_eq!(unbounded.span(), f64::INFINITY, "the span every weight bound is built from");
        let mut model = DeviceModel::ideal();
        model.window = unbounded;
        model.beta = 1e-6;
        assert_eq!(model.max_weight(), f64::INFINITY, "so every weight would be representable");
        assert!(matches!(
            model.apply(&[1e300], &mut Rng::new(11)),
            Err(DeviceError::BadWindow { .. })
        ));
    }

    /// The frozen per-cell properties move when `DeviceModel::d2d_seed` does — **both** of them.
    ///
    /// `the_same_seed_gives_the_same_array_and_the_weakest_grade_wins` pins one direction and only
    /// one: nothing in this module ever programmed the same weights onto **two different seeds**,
    /// so `cell_stream` could drop its `seed` argument entirely, or the device-to-device call could
    /// pass a literal zero, and every array in the suite would come back bit for bit what it was.
    /// A seed that never reaches the stream makes "two `DeviceModel`s with the same seed are the
    /// same physical array" vacuously true of every pair of models, which is the opposite of the
    /// claim.
    ///
    /// The mapping is `Mapping::BalancedDifferential` so that both devices of every pair sit
    /// mid-window: the one-at-minimum scheme parks the unused device on `Window::g_off`, where half
    /// the perturbed draws clamp to the rail and two seeds agree by construction.
    #[test]
    fn the_frozen_per_cell_streams_move_when_the_array_seed_does() {
        let n = 512usize;
        let want = vec![20.0; n];

        // Device-to-device only, cycle-to-cycle at zero, and the SAME caller stream for both runs,
        // so the frozen offset is the only thing that can separate the two arrays.
        let mut model = DeviceModel::ideal();
        model.window = win();
        model.beta = 1e-6;
        model.mapping = Mapping::BalancedDifferential;
        model.variability =
            Some(Variability::new(0.05, 0.0, 0.0, "d2d only", Evidence::Unstated).unwrap());
        model.d2d_seed = 1;
        let a = model.apply(&want, &mut Rng::new(77)).unwrap();
        model.d2d_seed = 2;
        let b = model.apply(&want, &mut Rng::new(77)).unwrap();
        assert_eq!(a.clamped, 0, "a clamped cell can agree with another seed by hitting the rail");
        assert_eq!(b.clamped, 0);
        let agreed_plus = a.g_plus.iter().zip(&b.g_plus).filter(|(x, y)| x == y).count();
        assert_eq!(agreed_plus, 0, "{agreed_plus} of {n} plus devices ignored the array seed");
        let agreed_minus = a.g_minus.iter().zip(&b.g_minus).filter(|(x, y)| x == y).count();
        assert_eq!(agreed_minus, 0, "{agreed_minus} of {n} minus devices ignored the array seed");
        // And the same seed still gives the same array, so the two lines above are about the seed
        // and not about `apply` having become non-deterministic.
        model.d2d_seed = 1;
        assert_eq!(model.apply(&want, &mut Rng::new(77)).unwrap(), a);

        // The stuck-at map is the second stream off the same seed and it moves too. Two
        // independent fault states at a rate of 0.1 with an even split disagree with probability
        // 1 - (0.9^2 + 0.05^2 + 0.05^2) = 0.185, so 512 cells give 94.7 with a one-sigma spread of
        // sqrt(512 * 0.185 * 0.815) = 8.8, and this measures 4 sigma of that.
        let mut faulty = DeviceModel::ideal();
        faulty.window = win();
        faulty.beta = 1e-6;
        faulty.mapping = Mapping::SingleEnded;
        faulty.stuck = Some(StuckAt::new(0.1, 0.5, "test", Evidence::Unstated).unwrap());
        faulty.d2d_seed = 1;
        let c = faulty.apply(&want, &mut Rng::new(77)).unwrap();
        faulty.d2d_seed = 2;
        let d = faulty.apply(&want, &mut Rng::new(77)).unwrap();
        let differ = c.fault_plus.iter().zip(&d.fault_plus).filter(|(x, y)| x != y).count();
        let expect = 0.185 * n as f64;
        let sd = (n as f64 * 0.185 * 0.815).sqrt();
        assert!(
            (differ as f64 - expect).abs() < 4.0 * sd,
            "{differ} of {n} fault states moved with the array seed, expected {expect} +/- {sd}"
        );
    }

    /// The two devices of a differential pair draw **their own** faults, from two consecutive draws
    /// of the same per-cell stream.
    ///
    /// `a_differential_pair_costs_twice_the_exposure_to_stuck_at_faults` counts the plus devices and
    /// the minus devices separately and then adds them, and a pair that shared one draw satisfies
    /// every one of those assertions: the marginal rate on each half is still `rate`, the total is
    /// still `plus + minus`, and `2 * plus` sits about 2 sigma from `2 * rate * n`, comfortably
    /// inside the 4-sigma band. Only the **joint** distribution can tell, and nothing looked at it.
    ///
    /// A shared draw is not a small error. It makes a stuck pair a pair that is stuck at the same
    /// rail, so the difference `G+ - G-` of a fully broken cell is exactly zero instead of full
    /// scale — the fault turns into a dropped weight rather than an arbitrary one, which is the
    /// error mode this mechanism exists to say is the expensive one.
    #[test]
    fn the_two_devices_of_a_pair_draw_their_faults_independently() {
        let rate = 0.2;
        let n = 20_000usize;
        let mut model = DeviceModel::ideal();
        model.window = win();
        model.beta = 1e-6;
        model.d2d_seed = 19;
        model.mapping = Mapping::Differential;
        model.stuck = Some(StuckAt::new(rate, 0.5, "test", Evidence::Unstated).unwrap());
        let held = model.apply(&vec![10.0; n], &mut Rng::new(23)).unwrap();

        let (mut both, mut exactly_one, mut opposite_rails) = (0usize, 0usize, 0usize);
        for (&p, &m) in held.fault_plus.iter().zip(&held.fault_minus) {
            match (p.is_stuck(), m.is_stuck()) {
                (true, true) => {
                    both += 1;
                    if p != m {
                        opposite_rails += 1;
                    }
                }
                (true, false) | (false, true) => exactly_one += 1,
                (false, false) => {}
            }
        }

        // Two independent Bernoulli(0.2) draws per pair. P(exactly one) = 2*0.2*0.8 = 0.32, so
        // 6400 of 20000 with a one-sigma spread of sqrt(20000*0.32*0.68) = 66. A shared draw makes
        // this EXACTLY zero.
        let expect_one = 0.32 * n as f64;
        let sd_one = (n as f64 * 0.32 * 0.68).sqrt();
        assert!(
            (exactly_one as f64 - expect_one).abs() < 4.0 * sd_one,
            "{exactly_one} of {n} pairs had one stuck device, expected {expect_one} +/- {sd_one}"
        );
        // P(both) = 0.04, so 800 +/- 27.7. A shared draw makes it 4000.
        let expect_both = 0.04 * n as f64;
        let sd_both = (n as f64 * 0.04 * 0.96).sqrt();
        assert!(
            (both as f64 - expect_both).abs() < 4.0 * sd_both,
            "{both} of {n} pairs had both devices stuck, expected {expect_both} +/- {sd_both}"
        );
        // And a stuck pair can be stuck at OPPOSITE rails, which one shared draw makes impossible.
        // P = 2 * 0.1 * 0.1 = 0.02, so 400 +/- sqrt(20000*0.02*0.98) = 19.8.
        let expect_split = 0.02 * n as f64;
        let sd_split = (n as f64 * 0.02 * 0.98).sqrt();
        assert!(
            (opposite_rails as f64 - expect_split).abs() < 5.0 * sd_split,
            "{opposite_rails} of {n} pairs sat on opposite rails, expected {expect_split} +/- {sd_split}"
        );
    }

    /// `Variability::validate` says it checks "every sigma", and the absolute floor is a sigma. The
    /// census is a three-row table whose third row could be made a duplicate of the second with the
    /// suite green, because no test in this module ever hands `Variability` a bad
    /// `sigma_floor_s`: every fixture here sets it to exactly zero, and so does
    /// `RRAM_VARIABILITY_PLACEHOLDER`. That is the same coefficient-of-zero hole the second pass
    /// recorded for this term's **arithmetic**, one function upstream of where it was repaired.
    #[test]
    fn the_absolute_noise_floor_is_named_by_the_variability_census() {
        assert!(matches!(
            Variability::new(0.0, 0.0, -1e-9, "a negative floor", Evidence::Unstated),
            Err(DeviceError::Negative { what: "sigma_floor_s", .. })
        ));
        assert!(matches!(
            Variability::new(0.0, 0.0, f64::NAN, "", Evidence::Unstated),
            Err(DeviceError::NonFinite { what: "sigma_floor_s", .. })
        ));
        assert!(matches!(
            Variability::new(0.0, 0.0, f64::INFINITY, "", Evidence::Unstated),
            Err(DeviceError::NonFinite { what: "sigma_floor_s", .. })
        ));
        // A positive floor is accepted, so the three rows above are a guard and not a wall.
        assert!(Variability::new(0.0, 0.0, 1e-9, "", Evidence::Unstated).is_ok());

        // Through `DeviceModel::validate`, which is where a struct literal reaches it.
        let mut model = DeviceModel::ideal();
        model.variability = Some(Variability {
            sigma_d2d_rel: 0.0,
            sigma_c2c_rel: 0.0,
            sigma_floor_s: -1e-9,
            source: "literal",
            evidence: Evidence::Unstated,
        });
        assert!(matches!(
            model.validate(),
            Err(DeviceError::Negative { what: "sigma_floor_s", .. })
        ));
        assert!(matches!(
            model.apply(&[0.5], &mut Rng::new(13)),
            Err(DeviceError::Negative { what: "sigma_floor_s", .. })
        ));
    }

    /// `Drift::validate` says it checks "exponents", plural, and the on-rail one had no bad
    /// fixture: the only ill-formed drift model in this module is `Drift::uniform(-0.1, ..)`, which
    /// sets **both** exponents and is therefore caught by the off-rail row before the on-rail row
    /// is reached. The two-row table could be made a duplicate of its first row and nothing would
    /// fail.
    ///
    /// A negative `Drift::nu_at_on` is not a small error either: `Drift::exponent_at` interpolates
    /// between the two rails, so it produces a negative exponent for every cell in the upper part
    /// of the window, and a negative exponent is conductance **growing** with time.
    #[test]
    fn the_on_rail_exponent_is_named_by_the_drift_census() {
        assert!(matches!(
            Drift::new(0.1, -0.02, 1.0, "a negative on-rail exponent", Evidence::Unstated),
            Err(DeviceError::Negative { what: "nu_at_on", .. })
        ));
        assert!(matches!(
            Drift::new(0.1, f64::NAN, 1.0, "", Evidence::Unstated),
            Err(DeviceError::NonFinite { what: "nu_at_on", .. })
        ));
        assert!(matches!(
            Drift::new(0.1, f64::INFINITY, 1.0, "", Evidence::Unstated),
            Err(DeviceError::NonFinite { what: "nu_at_on", .. })
        ));
        // The off-rail row still names itself, so the two are not one row under another name.
        assert!(matches!(
            Drift::new(-0.1, 0.02, 1.0, "", Evidence::Unstated),
            Err(DeviceError::Negative { what: "nu_at_off", .. })
        ));
        // Zero on both rails is a model that does not drift, and is accepted.
        assert!(Drift::new(0.1, 0.0, 1.0, "", Evidence::Unstated).is_ok());

        // Through `DeviceModel::validate`, the struct-literal route.
        let mut model = DeviceModel::ideal();
        model.window = win();
        model.beta = 1e-6;
        model.drift = Some(Drift {
            nu_at_off: 0.1,
            nu_at_on: -0.02,
            t0_s: 1.0,
            source: "literal",
            evidence: Evidence::Unstated,
        });
        assert!(matches!(model.validate(), Err(DeviceError::Negative { what: "nu_at_on", .. })));
        assert!(matches!(
            model.apply(&[10.0], &mut Rng::new(17)),
            Err(DeviceError::Negative { what: "nu_at_on", .. })
        ));
    }

    /// `Retention::validate`'s `# Errors` names `DeviceError::Negative` for a negative `ea_ev`, and
    /// nothing in this module ever built one: every retention fixture here carries a physical
    /// activation energy around 1 eV, so the comparison could be moved to any threshold below that
    /// and the suite would not notice.
    ///
    /// A negative activation energy inverts the Arrhenius law — `tau = tau0 * exp(Ea/kT)` becomes
    /// shorter at lower temperature — so a retention study run with one would report that cooling
    /// an array makes it forget faster, in a model whose whole purpose is the opposite claim.
    #[test]
    fn a_negative_activation_energy_is_refused_and_a_zero_one_is_not() {
        for ea in [-0.5, -1e-300, -1.0] {
            assert!(
                matches!(
                    Retention::new(ea, 1e-12, "", Evidence::Unstated),
                    Err(DeviceError::Negative { what: "ea_ev", .. })
                ),
                "an activation energy of {ea} eV was accepted"
            );
        }
        assert!(matches!(
            Retention::new(f64::NAN, 1e-12, "", Evidence::Unstated),
            Err(DeviceError::NonFinite { what: "ea_ev", .. })
        ));

        // Zero is the documented boundary and is accepted, so the rows above are a guard and not a
        // wall: it is the no-activation-energy case, where the acceleration factor is exactly 1.
        let flat = Retention::new(0.0, 1e-12, "no activation energy", Evidence::Unstated).unwrap();
        assert_eq!(flat.acceleration(300.0, 400.0).unwrap(), 1.0);
        assert_eq!(flat.tau_s(300.0).unwrap(), 1e-12);

        let mut model = DeviceModel::ideal();
        model.retention = Some(Retention {
            ea_ev: -0.5,
            tau0_s: 1e-12,
            source: "literal",
            evidence: Evidence::Unstated,
        });
        assert!(matches!(model.validate(), Err(DeviceError::Negative { what: "ea_ev", .. })));
    }

    /// `Retention::acceleration`'s `# Errors` says "as `Retention::tau_s`, **for either
    /// temperature**", and the second check had no test: every call in this module passes two
    /// sound temperatures, so deleting the stress-temperature check leaves the suite green.
    ///
    /// The three cases below are the three different wrong answers it would give. A stress
    /// temperature of zero makes `1/T_stress` an infinity and the factor underflows to exactly
    /// `0.0`, which is returned as a number. A negative one flips the sign of that reciprocal and
    /// returns a large finite factor computed from a temperature below absolute zero. A `NaN` is
    /// caught by the overflow check further down and comes back naming `acceleration` rather than
    /// `temp_k`, so the refusal points at the arithmetic instead of at the argument.
    #[test]
    fn the_acceleration_factor_checks_both_of_its_temperatures() {
        let r = Retention::new(1.0, 1e-12, "test", Evidence::Unstated).unwrap();
        assert!(r.acceleration(300.0, 400.0).is_ok(), "the sound case");
        assert_eq!(r.acceleration(350.0, 350.0).unwrap(), 1.0, "equal temperatures");

        for (t_use, t_stress) in [(0.0, 400.0), (-300.0, 400.0), (300.0, 0.0), (300.0, -400.0)] {
            assert!(
                matches!(
                    r.acceleration(t_use, t_stress),
                    Err(DeviceError::NotPositive { what: "temp_k", .. })
                ),
                "acceleration({t_use}, {t_stress}) was not refused as a non-positive temperature"
            );
        }
        for (t_use, t_stress) in [
            (f64::NAN, 400.0),
            (300.0, f64::NAN),
            (f64::INFINITY, 400.0),
            (300.0, f64::INFINITY),
        ] {
            assert!(
                matches!(
                    r.acceleration(t_use, t_stress),
                    Err(DeviceError::NonFinite { what: "temp_k", .. })
                ),
                "acceleration({t_use}, {t_stress}) was not refused as a non-finite temperature"
            );
        }
    }

    /// `check_temperature` has two branches and they name different things: `DeviceError::NonFinite`
    /// for a `NaN` or an infinity, `DeviceError::NotPositive` for a zero or a negative. Every test
    /// in this module that hands it a bad temperature asserts `is_err()` and nothing else, and
    /// `!(NaN > 0.0)` is **true**, so deleting the first branch returns the second's variant for a
    /// `NaN` with the suite green. That is the whole reason `DeviceError::NonFinite` exists: `what`
    /// names the quantity, and the variant says which kind of wrong it is.
    ///
    /// The infinity is the case where it stops being a naming problem. `!(inf > 0.0)` is **false**,
    /// so an infinite temperature passes the second branch outright: `tau_s(inf)` is
    /// `tau0 * exp(Ea/(k*inf))`, which is `tau0 * exp(0)`, a perfectly finite lifetime handed back
    /// as an answer.
    #[test]
    fn a_non_finite_temperature_is_refused_as_non_finite_and_not_as_non_positive() {
        let r = Retention::new(1.0, 1e-12, "test", Evidence::Unstated).unwrap();
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(
                matches!(r.tau_s(bad), Err(DeviceError::NonFinite { what: "temp_k", .. })),
                "tau_s({bad}) was not refused as a non-finite temperature"
            );
        }
        for bad in [0.0, -1e-300, -300.0] {
            assert!(
                matches!(r.tau_s(bad), Err(DeviceError::NotPositive { what: "temp_k", .. })),
                "tau_s({bad}) was not refused as a non-positive temperature"
            );
        }
        // What the infinity would otherwise return: a finite lifetime, equal to `tau0_s`.
        assert_eq!((1.0f64 / f64::INFINITY).exp(), 1.0, "the silent answer the guard prevents");

        // The same check, from the other two entry points that share it.
        assert!(matches!(
            ReadNoise::new(f64::INFINITY, 1e6, 0.0, "", Evidence::Unstated),
            Err(DeviceError::NonFinite { what: "temp_k", .. })
        ));
        assert!(matches!(
            r.remaining_fraction(1.0, f64::INFINITY),
            Err(DeviceError::NonFinite { what: "temp_k", .. })
        ));
    }

    /// `ReadNoise::distinguishable_levels` refuses a `separation` that is not positive and finite,
    /// and the guard could be deleted outright because the only value any test passes it is a
    /// positive finite one. Deleting it is not harmless in either direction, and the two cases are
    /// opposite mistakes:
    ///
    /// * a **negative** separation makes `span / (separation * sigma)` negative, so `n + 1` falls
    ///   below 2 and the function returns `Some(1)` — "this cell carries no analog information" —
    ///   for a demand that is merely nonsense;
    /// * an **infinite** separation makes the quotient exactly zero, so `n` is `1.0` and the answer
    ///   is again `Some(1)`, this time for a demand no cell could ever meet.
    ///
    /// A zero separation is the one case the guard does not change: `span / 0.0` is an infinity and
    /// the `!n.is_finite()` branch below already returns `None` for it. It is asserted here anyway,
    /// because the guard is what makes that a decision rather than an accident.
    #[test]
    fn the_separation_demand_is_refused_unless_it_is_positive_and_finite() {
        let w = win();
        let n = THERMAL_READ_NOISE_300K;
        assert!(n.distinguishable_levels(&w, 0.2, 6.0).is_some(), "the sound case");
        for bad in [0.0, -1e-300, -6.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
            assert_eq!(
                n.distinguishable_levels(&w, 0.2, bad),
                None,
                "a separation demand of {bad} produced a level count"
            );
        }
    }

    /// `Levels::snap`'s doc: "the result is always exactly `g_off + k * step` for an integer `k` in
    /// `0..n`". The **ceiling** of the index clamp is what makes that true above the window, and
    /// every existing call feeds `snap` a conductance already inside it —
    /// `quantisation_error_is_bounded_by_half_a_step_and_reaches_it` draws uniformly on
    /// `[g_off, g_on]`, and `DeviceModel::apply` snaps a target `Mapping::program` has already
    /// clamped. So the ceiling could be dropped and `Window::clamp` behind it would rescue every
    /// reachable answer to `g_on`.
    ///
    /// `g_on` is not on the grid in general, and this fixture is a case where it is not: at
    /// `n = 6` on the 1 uS to 100 uS window, `g_off + 5 * step` measures 9.999999999999999e-05,
    /// one unit in the last place **below** `g_on`. The unclamped index therefore returns a
    /// conductance the grid does not contain, for every conductance above the window.
    #[test]
    fn the_snapped_level_index_is_held_below_the_top_level_so_the_result_stays_on_the_grid() {
        let w = win();
        let l =
            Levels::new(6, "six levels, so the top one is not g_on", Evidence::Unstated).unwrap();
        let step = l.step(&w);
        let top = w.g_off + 5.0 * step;
        // The fixture's own premise. Without it this test compares `g_on` against `g_on`.
        assert!(top < w.g_on, "the top level {top} is not below g_on {}; pick another n", w.g_on);
        assert!(w.contains(top), "the top level {top} is outside the window");

        for g in [w.g_on, 1.5 * w.g_on, 2.0 * w.g_on, 1.0, f64::MAX] {
            assert_eq!(l.snap(g, &w), top, "snap({g}) left the grid");
        }
        // The floor of the same clamp, below the window.
        for g in [w.g_off, 0.0, -1.0, f64::MIN] {
            assert_eq!(l.snap(g, &w), w.g_off, "snap({g}) left the grid");
        }
        // And the index really is the last one of `n`, not one past it.
        let k = ((l.snap(f64::MAX, &w) - w.g_off) / step).round();
        assert_eq!(k, 5.0, "the top of a 6-level grid is index {k}");

        // The other side of the same last place, which is where this module's doc was wrong: at
        // `n = 100` on the same window, `g_off + 99 * step` lands one place ABOVE `g_on`, so
        // `Window::clamp` pulls it back to `g_on` and the result is off the grid by that place.
        // These are measured here, not published anywhere: the window is a test fixture.
        let hundred = Levels::new(100, "a grid whose top level overshoots", Evidence::Unstated)
            .unwrap();
        let over = w.g_off + 99.0 * hundred.step(&w);
        assert!(over > w.g_on, "at n = 100 the top level {over} no longer overshoots g_on");
        assert_eq!(over - w.g_on, 1.3552527156068805e-20, "one unit in the last place at 1e-4 S");
        assert_eq!(hundred.snap(w.g_on, &w), w.g_on, "the clamp is what keeps this in the window");
        assert!(
            hundred.snap(w.g_on, &w) != over,
            "the overshoot case no longer differs, so the doc's caveat is stale"
        );
        // And the cost is bounded by that one place and not by a step: the error `snap` makes at
        // the top of an overshooting grid is the clamp's, which measures 1.36e-14 of a step here.
        let cost = (over - w.g_on) / hundred.step(&w);
        assert!(cost < 1e-13, "the top-level clamp costs {cost} of a step");
    }

    /// `Wires::new` refuses a non-finite segment resistance, and `Crossbar::new` runs the same
    /// check on the wire model it is handed — "plus whatever `Wires::new` would return for the wire
    /// model", as its own `# Errors` says. Neither had a test: `Wires` fields are public, and every
    /// crossbar fixture in this module goes through `Wires::new` first, so the round trip inside
    /// `Crossbar::new` could be deleted, and `Wires::new` itself only ever saw finite values.
    ///
    /// The infinity is the case with a consequence one function along. `Crossbar::solve` takes
    /// `gw = 1.0 / r_segment_ohm`, so an infinite segment resistance is the **only** way a caller
    /// reaches the solve with `gw == 0.0` — the zero-resistance case is intercepted by the
    /// closed-form branch above it, which makes `gw` infinite and never uses it. That refusal is
    /// what keeps `Crossbar::residual`'s `0.0 * inf` out of the iteration; see
    /// `a_non_finite_bit_line_residual_is_not_dropped_by_the_fold`.
    #[test]
    fn an_infinite_wire_resistance_is_refused_and_a_struct_literal_is_checked_by_the_crossbar() {
        for bad in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
            assert!(
                matches!(
                    Wires::new(bad, "not a wire", Evidence::Unstated),
                    Err(DeviceError::NonFinite { what: "r_segment_ohm", .. })
                ),
                "a segment resistance of {bad} ohm was not refused as non-finite"
            );
        }
        assert!(matches!(
            Wires::new(-1.0, "", Evidence::Unstated),
            Err(DeviceError::Negative { what: "r_segment_ohm", .. })
        ));
        // Zero is the ideal reference case and is legal, so the rows above are a guard, not a wall.
        assert!(Wires::new(0.0, "the ideal reference", Evidence::Unstated).is_ok());

        // The fields are public, so `Crossbar::new` has to run the check on a struct literal.
        let g = vec![1e-6; 4];
        let open = Wires {
            r_segment_ohm: f64::INFINITY,
            source: "literal",
            evidence: Evidence::Unstated,
        };
        assert!(matches!(
            Crossbar::new(2, 2, g.clone(), open),
            Err(DeviceError::NonFinite { what: "r_segment_ohm", .. })
        ));
        let backwards =
            Wires { r_segment_ohm: -1.0, source: "literal", evidence: Evidence::Unstated };
        assert!(matches!(
            Crossbar::new(2, 2, g.clone(), backwards),
            Err(DeviceError::Negative { what: "r_segment_ohm", .. })
        ));
        // A sound literal still builds, so neither row above refuses every struct literal.
        let sound = Wires { r_segment_ohm: 2.0, source: "literal", evidence: Evidence::Unstated };
        assert!(Crossbar::new(2, 2, g, sound).is_ok());
    }

    /// `Programmed::age_s` is documented as "`0.0` at `DeviceModel::apply`" and
    /// `Programmed::aged_at_k` as "the temperature the ageing was done at". Nothing read either
    /// one: both are carried through `Programmed::aged`, both appear in the `PartialEq` that
    /// `the_same_seed_gives_the_same_array_and_the_weakest_grade_wins` compares two arrays with —
    /// and that comparison is between two arrays built the same way, so a constant written into
    /// both sides cancels out of it.
    ///
    /// They are the only record an array carries of **why** its conductances are what they are. An
    /// array that reports an age it does not have, or that forgets the temperature it was aged at,
    /// is an array whose drift and retention figures cannot be attributed to anything.
    #[test]
    fn a_freshly_programmed_array_has_no_age_and_an_aged_one_records_its_temperature() {
        let mut model = DeviceModel::ideal();
        model.window = win();
        model.beta = 1e-6;
        model.drift = Some(Drift::uniform(0.05, 1.0, "test", Evidence::Unstated).unwrap());
        model.retention = Some(Retention::new(1.0, 1e-12, "test", Evidence::Unstated).unwrap());
        let held = model.apply(&[10.0, -10.0, 0.0], &mut Rng::new(41)).unwrap();
        assert_eq!(held.age_s, 0.0, "a freshly programmed array reports an age of {}", held.age_s);
        assert_eq!(held.aged_at_k, None, "a freshly programmed array names a temperature");

        let later = held.aged(86_400.0, 358.15).unwrap();
        assert_eq!(later.age_s, 86_400.0);
        assert_eq!(later.aged_at_k, Some(358.15));

        // `aged` is a function of its arguments and not a running total, so a second ageing of the
        // SAME array replaces both fields rather than adding to them.
        let other = held.aged(3_600.0, 300.0).unwrap();
        assert_eq!(other.age_s, 3_600.0);
        assert_eq!(other.aged_at_k, Some(300.0));

        // Zero seconds is an ageing that happened, at a temperature worth recording: the array is
        // no longer the one straight out of `apply`, even though nothing moved.
        let none = held.aged(0.0, 300.0).unwrap();
        assert_eq!(none.age_s, 0.0);
        assert_eq!(none.aged_at_k, Some(300.0));
    }

    /// `ErrorStats::max_abs` is "largest `|held - target|`", and the absolute value could be
    /// dropped. Every array this module measures it on is either error-free — where it is asserted
    /// to be exactly `0.0` — or noisy in both directions, where the largest positive error and the
    /// largest magnitude are the same number to within a sample. The one assertion that would bite,
    /// `later.error().max_abs > held.error().max_abs`, is in the module doc's **doctest**, and the
    /// mutation harness runs `--lib`.
    ///
    /// The fixture here is an array whose errors are all **negative** and cannot be anything else:
    /// drift only takes conductance away, and a single-ended read subtracts a nominal reference
    /// that does not drift with the cell, so every held weight is below its target. Without the
    /// absolute value the reported maximum is then `0.0` — a drifted array reported as exact.
    #[test]
    fn the_largest_error_is_a_magnitude_on_an_array_whose_errors_are_all_negative() {
        let mut model = DeviceModel::ideal();
        model.window = win();
        model.beta = 1e-6;
        model.mapping = Mapping::SingleEnded;
        model.drift = Some(Drift::uniform(0.05, 1.0, "test", Evidence::Unstated).unwrap());
        let want: Vec<f64> = (1..=32).map(|k| f64::from(k) * 2.0).collect();
        let held = model.apply(&want, &mut Rng::new(51)).unwrap().aged(1e6, 300.0).unwrap();

        // The premise, asserted rather than assumed: without it this is every other `max_abs` test.
        let got = held.weights();
        let mut hand = 0.0f64;
        for (&g, &t) in got.iter().zip(&held.target) {
            assert!(g - t < 0.0, "target {t} came back as {g}, which is not below it");
            hand = hand.max((g - t).abs());
        }
        assert!(hand > 0.0, "the fixture produced no error at all");

        // The same fold over the same values in the same order, so this is exact.
        let e = held.error();
        assert_eq!(e.max_abs, hand, "the largest error is not the largest magnitude");
        let bias = e.mean_signed;
        assert!(bias < 0.0, "a uniformly negative error reported a bias of {bias}");
        assert!(e.max_abs >= e.mean_signed.abs(), "the largest error is below the mean one");
        assert!(e.max_abs >= e.rms, "the largest error is below the root-mean-square one");
    }

    /// `Programmed::aged` checks its own time and temperature, and both checks had no test that
    /// could see them. The only array this module ages through a bad argument carries a
    /// `Retention` model, whose `Retention::remaining_fraction` refuses first — so both guards were
    /// covered by a sub-model that need not be present, which is exactly the case they are for.
    ///
    /// With no retention model the drift path swallows both silently: `Drift::factor` returns
    /// exactly `1.0` for any `t_s` at or below `t0_s`, and a negative time is below it, so ageing
    /// for minus half a second at absolute zero comes back `Ok` with the array unchanged.
    #[test]
    fn ageing_checks_its_own_time_and_temperature_for_an_array_with_no_retention_model() {
        let mut model = DeviceModel::ideal();
        model.window = win();
        model.beta = 1e-6;
        model.drift = Some(Drift::uniform(0.05, 1.0, "test", Evidence::Unstated).unwrap());
        assert!(model.retention.is_none(), "the premise: nothing refuses before `aged` does");
        let held = model.apply(&[10.0], &mut Rng::new(61)).unwrap();
        assert!(held.aged(1.0, 300.0).is_ok(), "the sound case");
        assert!(held.aged(0.0, 300.0).is_ok(), "zero seconds is the array as programmed");

        for t in [-0.5, -1e-300, -1e9] {
            assert!(
                matches!(held.aged(t, 300.0), Err(DeviceError::Negative { what: "t_s", .. })),
                "ageing for {t} seconds was accepted"
            );
        }
        for t in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(
                matches!(held.aged(t, 300.0), Err(DeviceError::NonFinite { what: "t_s", .. })),
                "ageing for {t} seconds was accepted"
            );
        }
        for k in [0.0, -1e-300, -300.0] {
            assert!(
                matches!(held.aged(1.0, k), Err(DeviceError::NotPositive { what: "temp_k", .. })),
                "ageing at {k} K was accepted"
            );
        }
        for k in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(
                matches!(held.aged(1.0, k), Err(DeviceError::NonFinite { what: "temp_k", .. })),
                "ageing at {k} K was accepted"
            );
        }
    }

    /// `Crossbar::ideal_currents` refuses a non-finite drive voltage on its own public surface, and
    /// the only test that offers it one goes through `Crossbar::solve`, which errs anyway — the
    /// `NaN` reaches the tridiagonal solve and its closing finiteness check returns
    /// `DeviceError::SingularLine`. So the guard could be deleted and every assertion in this
    /// module would still hold, while `ideal_currents` itself — the reference every IR-drop figure
    /// here is measured against — returned `Ok` with a vector of `NaN`s.
    #[test]
    fn a_non_finite_drive_voltage_is_refused_on_the_ideal_current_s_own_surface() {
        let xb = Crossbar::new(2, 2, vec![1e-6; 4], Wires::ideal()).unwrap();
        assert!(xb.ideal_currents(&[0.2, 0.1]).is_ok(), "the sound case");
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(
                matches!(
                    xb.ideal_currents(&[bad, 0.1]),
                    Err(DeviceError::NonFinite { what: "v_in", .. })
                ),
                "a drive of {bad} V on the first word line was accepted"
            );
            assert!(
                matches!(
                    xb.ideal_currents(&[0.1, bad]),
                    Err(DeviceError::NonFinite { what: "v_in", .. })
                ),
                "a drive of {bad} V on the second word line was accepted"
            );
        }
        // The length check is a different refusal and still names itself.
        assert!(matches!(
            xb.ideal_currents(&[0.2]),
            Err(DeviceError::BadDrive { rows: 2, len: 1 })
        ));
    }

    /// `Crossbar::solve`'s `# Errors` names `DeviceError::NotPositive` "for a non-positive `tol`
    /// **or a zero sweep budget**", and the budget half had no test: `1..=0` iterates zero times
    /// and falls straight through to `DeviceError::NotConverged`, which is still an `Err`, and the
    /// test that offers a small budget asserts only `is_err()`.
    ///
    /// The ideal-wire branch is where that stops being a naming problem. It sits **below** the
    /// budget check and returns its closed form without consulting the budget at all, so a zero
    /// budget checked anywhere later comes back `Ok` with `sweeps: 0` — a caller who asked for no
    /// work at all gets an answer.
    #[test]
    fn a_zero_sweep_budget_is_a_bad_argument_and_not_a_failure_to_converge() {
        let real = Crossbar::new(
            2,
            2,
            vec![1e-6; 4],
            Wires::new(1.0, "test", Evidence::Unstated).unwrap(),
        )
        .unwrap();
        assert!(real.solve(&[0.2, 0.2], 1e-9, 50).is_ok(), "the sound case");
        assert!(matches!(
            real.solve(&[0.2, 0.2], 1e-9, 0),
            Err(DeviceError::NotPositive { what: "max_sweeps", value: 0.0 })
        ));
        // A budget that is merely too small is the NEIGHBOURING failure and is a different variant.
        assert!(matches!(
            real.solve(&[0.2, 0.2], 1e-30, 1),
            Err(DeviceError::NotConverged { sweeps: 1, .. })
        ));

        let ideal = Crossbar::new(2, 2, vec![1e-6; 4], Wires::ideal()).unwrap();
        assert_eq!(ideal.solve(&[0.2, 0.2], 1e-9, 1).unwrap().sweeps, 0, "the closed form");
        assert!(matches!(
            ideal.solve(&[0.2, 0.2], 1e-9, 0),
            Err(DeviceError::NotPositive { what: "max_sweeps", value: 0.0 })
        ));

        // A non-positive tolerance is the other half of the same sentence and names itself.
        assert!(matches!(
            real.solve(&[0.2, 0.2], 0.0, 10),
            Err(DeviceError::NotPositive { what: "tol", .. })
        ));
    }

    /// `Crossbar::residual` checks **both** of a node's residuals for finiteness, and the check on
    /// the bit-line one could be deleted with the suite green.
    ///
    /// The reason is in that function's own doc, and the doc's argument is the thing that has the
    /// hole: the two residuals do share the term `g * (a - b)`, so a non-finite **node voltage**
    /// poisons both and the word-line check alone catches it — which is what every `NaN` case in
    /// `the_line_solve_refuses_a_degenerate_pivot_rather_than_producing_infinities` exercises. The
    /// poison can also enter through the **wire** term, and there the two sides are not symmetric:
    /// the bit-line residual carries `gw * (south - b)` and `gw * (b_north - b)`, and the word-line
    /// residual carries neither.
    ///
    /// `gw == 0.0` against node voltages whose difference overflows is the case: `0.0 * inf` is a
    /// `NaN` that appears on the bit-line side alone, and `f64::max` **drops** a `NaN`, so the fold
    /// reports the largest word-line residual and `Crossbar::solve` declares the sweep converged.
    /// A zero `gw` is a segment resistance of infinity, which is exactly what `Wires::new` exists
    /// to refuse — see
    /// `an_infinite_wire_resistance_is_refused_and_a_struct_literal_is_checked_by_the_crossbar` —
    /// so these two guards hold each other up, and neither was tested.
    #[test]
    fn a_non_finite_bit_line_residual_is_not_dropped_by_the_fold() {
        let col = Crossbar::new(
            2,
            1,
            vec![1e-6, 1e-6],
            Wires::new(1.0, "test", Evidence::Unstated).unwrap(),
        )
        .unwrap();
        let mx = f64::MAX;
        // The arithmetic this rests on, so the assertion below cannot pass for another reason.
        assert!((mx - -mx).is_infinite(), "MAX - (-MAX) no longer overflows");
        assert!((0.0f64 * f64::INFINITY).is_nan(), "0 * inf is no longer a NaN");
        assert!(!(0.0f64).max(f64::NAN).is_nan(), "f64::max no longer drops a NaN");

        // Word-line residuals: `gw * (v_in - a)` is `0 * 0` and `g * (a - b)` is finite, so both
        // are finite. Bit-line residuals: `0.0 * (MAX - (-MAX))` is `0 * inf`, a NaN on that side
        // alone.
        assert_eq!(
            col.residual(&[0.2, 0.2], &[-mx, mx], &[0.2, 0.2], 0.0),
            f64::INFINITY,
            "a NaN on the bit-line side was folded away"
        );
        // The same fixture with a finite difference has a finite, non-zero residual, so the line
        // above is about the NaN and not about the fixture.
        let sound = col.residual(&[0.2, 0.2], &[0.0, 0.0], &[0.2, 0.2], 0.0);
        assert!(sound.is_finite() && sound > 0.0, "the sound case gave {sound}");
    }

    /// The two degenerate-pivot guards in `thomas` leave the caller's buffers alone, and that is
    /// the **only** thing they change.
    ///
    /// Both return `false` either way. A zero pivot makes `out[k]` an infinity or a `NaN`, that
    /// value survives the back substitution, and the closing `all(is_finite)` returns `false` from
    /// the end of the function instead of from the guard — so every assertion this module has on
    /// these two guards, which all read the boolean, passes with both `== 0.0` comparisons deleted.
    /// What they buy is in `thomas`'s own doc — "checked rather than **allowed to produce
    /// infinities**" — and that is a claim about `out` and `c`, which nothing read.
    ///
    /// It is the same shape as the two second-pass findings this module records as the worst kind:
    /// a defect invisible in the answer and visible only in the state left behind.
    #[test]
    fn a_refused_line_solve_writes_no_infinity_into_the_caller_s_buffers() {
        const SENTINEL: f64 = -7.5;

        // A zero LEADING pivot. `sup[0] / 0.0` and `rhs[0] / 0.0` are the first two writes the
        // guard prevents, and they are -inf and +inf.
        let (mut out, mut c) = ([SENTINEL; 2], [SENTINEL; 2]);
        assert!(!thomas(&[0.0, -1.0], &[0.0, 2.0], &[-1.0, 0.0], &[1.0, 1.0], &mut out, &mut c));
        assert!(
            out.iter().chain(c.iter()).all(|v| v.is_finite()),
            "a refused leading pivot left out = {out:?} and c = {c:?}"
        );

        // An INTERIOR pivot that cancels to exactly zero: `diag[1] - sub[1] * c[0]` is `1 - 1 * 1`.
        // The first row is solved before the guard fires, so `out[0]` is 1.0 and stays 1.0 — under
        // the unguarded version the back substitution subtracts `c[0] * out[1]`, which is a NaN,
        // and overwrites it.
        let (mut out, mut c) = ([SENTINEL; 2], [SENTINEL; 2]);
        assert!(!thomas(&[0.0, 1.0], &[1.0, 1.0], &[1.0, 0.0], &[1.0, 1.0], &mut out, &mut c));
        assert_eq!(out[0], 1.0, "the row before the bad pivot was overwritten");
        assert_eq!(c[0], 1.0);
        assert!(
            out.iter().chain(c.iter()).all(|v| v.is_finite()),
            "a refused interior pivot left out = {out:?} and c = {c:?}"
        );

        // The same two shapes with a sound pivot are solved, so neither row above is refused for
        // being malformed.
        let (mut out, mut c) = ([SENTINEL; 2], [SENTINEL; 2]);
        assert!(thomas(&[0.0, -1.0], &[2.0, 2.0], &[-1.0, 0.0], &[1.0, 1.0], &mut out, &mut c));
        assert!(out.iter().all(|v| v.is_finite()), "the sound case gave {out:?}");
        assert_ne!(out[0], SENTINEL, "the sound case wrote nothing");
    }
}
