//! The population density of integrate-and-fire neurons driven by white noise: Richardson's
//! threshold integration of the Fokker–Planck equation, for the stationary state and for the
//! linear response to a modulated parameter, checked against Brunel's closed forms, and used to
//! find the oscillatory instabilities of Brunel's sparse network.
//!
//! # The two papers
//!
//! Richardson, *Firing-rate response of linear and nonlinear integrate-and-fire neurons to
//! modulated current-based and conductance-based synaptic drive*, Physical Review E 76:021919,
//! 2007, doi:10.1103/PhysRevE.76.021919 — the method. Brunel, *Dynamics of sparsely connected
//! networks of excitatory and inhibitory spiking neurons*, Journal of Computational Neuroscience
//! 8:183–208, 2000, doi:10.1023/A:1008925309027 — the closed-form stationary density and the
//! closed-form linear stability this method is checked against, and the network whose stability
//! it decides.
//!
//! # The equation
//!
//! Richardson's neuron, his eqs. (25) and (39), is
//!
//! ```text
//! τ dV/dt = E − V + ψ(V) + σ √(2τ) ξ(t),        ⟨ξ(t) ξ(t')⟩ = δ(t − t')
//! ψ = 0                                          leaky IF (LIF)
//! ψ = Δ_T exp((V − V_T)/Δ_T)                     exponential IF (EIF), eq. (48)
//! ```
//!
//! and the density `P(V, t)` of an ensemble of them obeys the continuity equation (1),
//! `∂P/∂t + ∂J/∂V = 0`, with the current operator (41), `J = (E − V + ψ)P/τ − (σ²/τ) ∂P/∂V`. A
//! neuron that reaches `V_th` is removed and put back at `V_re`: `P(V_th) = 0` and the flux leaving
//! at threshold re-enters at reset, eqs. (4)–(5). Richardson also closes the bottom of the voltage
//! axis, `J(V_lb) = 0`, eq. (6), at a `V_lb` low enough not to matter; he uses −100 mV.
//!
//! # Threshold integration
//!
//! The trick, Richardson's section II B: write the steady state as two first-order equations,
//! `−∂J₀/∂V = r₀[δ(V − V_th) − δ(V − V_re)]` (14) and `𝒥P₀ = J₀` (15), divide out the unknown
//! rate `r₀` (16), and integrate BACKWARDS from threshold, where both boundary values are known —
//! `j₀ = 1`, `p₀ = 0` — to `V_lb`. The rate is then whatever normalises the density. Appendix A
//! gives the discretisation, on a lattice `V⁽ᵏ⁾ = V_lb + kΔ` with the reset on a lattice point
//! `k_re`:
//!
//! ```text
//! j₀⁽ᵏ⁻¹⁾ = j₀⁽ᵏ⁾ − δ_{k, k_re+1}                                  (A5)
//! p₀⁽ᵏ⁻¹⁾ = p₀⁽ᵏ⁾ A⁽ᵏ⁾ + Δ τ j₀⁽ᵏ⁾ B⁽ᵏ⁾                             (A6)
//! A = e^{ΔG},  B = (e^{ΔG} − 1)/(σ² ΔG),  G = (V − E − ψ)/σ²       (A4), from (43)
//! r₀ = (Σₖ Δ p₀⁽ᵏ⁾)⁻¹,   P₀ = r₀ p₀,   J₀ = r₀ j₀                   (A7), (A8)
//! ```
//!
//! [`WhiteNoiseIf::stationary`] is exactly that recursion — on four cells the tests work it by
//! hand. Two things about it are EXACT and are checked as such: the flux is `r₀` above the reset
//! and zero below it with no rounding at all, because `j₀` only ever has `1` subtracted from it
//! once; and the density integrates to one by construction (to `1 − r τ_ref` with a refractory
//! period, eq. 91). Step (A3) is exact when `G` and `H` are constant over a cell, which is why it
//! survives the EIF's exponential growth near threshold where an Euler step would not (Richardson,
//! after (A3)), and the global error is first order in `Δ`: over 56 neurons the order the tests
//! measure lies in `[0.96, 1.03]`.
//!
//! The first-order error has a reading. Far below threshold the rate goes as
//! `exp(−(V_th − E)²/2σ²)`, and the lattice's relative error there is `½Δ(V_th − E)/σ²` to within
//! 6%: the scheme behaves as if the threshold were half a cell off. At Richardson's 10 µV step the
//! lattice is 0.06% from the Siegert rate for his mean-driven Fig. 1 case i, and 9.5% for a neuron
//! 20σ below threshold; the step to use is the one that makes `Δ(V_th − E)/σ²` small, not one read
//! off a figure. The exponential form is closer still: 30σ below threshold, where
//! `½Δ(V_th − E)/σ² = 1/4`, the lattice rate is the Siegert rate times `e^{−1/4}` to `2.2 × 10⁻⁴`.
//!
//! The linear response is the same idea, eqs. (17)–(24) and (A9)–(A15). A parameter `α` modulated
//! as `α₀ + α₁ cos ωt` perturbs the operator by an inhomogeneous term `F_α = −𝒥_α P₀` (13), and the
//! first-order solution is split into a part driven by the unknown rate modulation `r̂_α` and a part
//! driven by `F_α`, each integrated backwards from threshold; `J(V_lb) = 0` then fixes
//! `r̂_α = −α₁ ĵ_α⁽⁰⁾/ĵ_r⁽⁰⁾` (A13). [`Stationary::response`] does this for five parameters —
//! the mean `E`, the variance `σ²`, the conductance `g` (eq. 33), and the EIF's spike threshold
//! `V_T` and width `Δ_T` (eq. 53) — at any complex `λ`, not only `λ = iω`, so that the same
//! recursion can find the eigenvalues of a network; and with the absolute refractory period of
//! Richardson's section V B, whose reset term carries `e^{−λτ_ref}` (eq. 92).
//!
//! Richardson's conductance modulation, carried from the LIF's (33) unchanged into the EIF's (50),
//! scales the leak `(E − V)/τ` and not the spike current `ψ/τ`: a conductance reversing at `E`, not
//! a modulation of the membrane time constant as a whole. The tests take the zero-frequency
//! derivative that way; the other reading differs by 38% and by a factor of 33 on Fig. 2's cases.
//!
//! # Two noise conventions, and the factor between them
//!
//! Richardson's `σ` is the standard deviation of the free membrane potential; Brunel's (his
//! eq. 3, and [`crate::meanfield`]'s) is the coefficient of `√τ η(t)`, whose free standard deviation
//! is `σ_B/√2`. So `σ_B = √2 σ_R`, and [`WhiteNoiseIf::siegert`] and
//! [`WhiteNoiseIf::from_siegert`] are the only places the factor is applied. In the wrong place it
//! moves Fig. 1 case i's rate by 0.7% — it would pass a casual check — and case ii's by 82%.
//!
//! # Brunel's closed forms
//!
//! For the LIF the stationary density is known in closed form, Brunel's eq. (19):
//!
//! ```text
//! P₀(V) = (2ν₀τ/σ) exp(−(V − μ)²/σ²) ∫_{(V−μ)/σ}^{(θ−μ)/σ} Θ(u − (V_r − μ)/σ) e^{u²} du
//! ```
//!
//! [`brunel_density`] evaluates it with [`crate::meanfield::siegert_integral`], through the
//! identity `2e^{u²} = erfcx(u) + erfcx(−u)`, and with `ν₀` from [`SiegertInput::rate`] — the
//! crate's existing Siegert rate, not a second copy of it.
//!
//! His Appendix A.3 solves the linear stability of the network in closed form too: eq. (46), in
//! confluent hypergeometric functions (43)–(44). Read with only a mean coupling or only a variance
//! coupling, it is the LIF's response to a modulated mean or variance, with the refractory period;
//! `tools/density_reference.py` evaluates it with `mpmath` outside the crate. Its mean response
//! agrees, to `10⁻⁹` or better at every frequency the script tabulates (it refuses to print
//! otherwise), with the independent parabolic-cylinder form of Klett and Lindner
//! (arXiv:2503.07434, eq. 10); the variance response has eq. (46) alone as its closed form. Those
//! are how this module's LIF response is checked at every frequency.
//!
//! The Siegert rate is Brunel's eq. (21), on page 188; eq. (22) is its low-rate asymptote,
//! `ν₀τ ≃ (θ − μ)/(σ√π) exp(−(θ − μ)²/σ²)`, which `eq_22_is_the_low_rate_limit_of_eq_21` checks
//! as a limit.
//!
//! ⚠ **Eq. (19) as printed has `Θ(u − V_r)`, which cannot be what is meant.** The integration
//! variable `u` is a voltage in units of `σ` measured from `μ`; `V_r` is a voltage. Read literally
//! in millivolts, with Brunel's `V_r = 10`, the step sits ten noise units above the mean, and
//! wherever `(θ − μ)/σ ≤ 10` — at Table 1's three points and at the neuron the tests use — the
//! density vanishes. Read literally in volts, `Θ(u − 0.01)`, it puts the reset at `μ + 0.01σ`.
//! The argument has to be `(V_r − μ)/σ` — the lower limit of eq. (21), and the reset of his own
//! eq. (36) — and only that reading satisfies the paper's normalisation (12):
//! `the_printed_heaviside_argument_does_not_normalise` integrates all three, to 0, 0.76 and the
//! required 0.981.
//!
//! ⚠ **The unnumbered boundary conditions after eq. (40) are misprinted.** Page 202 gives them in
//! two lines, the first from eq. (38) at threshold and the second from eq. (39) at the reset. The
//! second line's continuity condition is a repeat of the first line's `Q̂₁(y_θ, λ) = 0`, where eq.
//! (39)'s `[Q̂₁]_{y_r} = 0` is meant; and its jump condition reads `∂Q̂₁/∂y(y_θ) = −exp(−λτ_rp) + H
//! exp(−λτ)`: at `y_θ` where the jump at `y_r` is meant, and with `e^{−λτ}` where eq. (39) has
//! `n₁(t − D)` — that is, `e^{−λD}`. Eq. (46) itself carries `e^{−λD}` in both places, and it is
//! (46), evaluated as printed, that reproduces Table 1 below. Two smaller inconsistencies do not
//! propagate: the homogeneous equation is printed `½φ″ + yφ′ + (1 − λ)φ = 0` on page 202 and with
//! `(1 − λτ)` on page 203, and the Wronskian (45) has `Γ(λ/2)` where `φ₂` (44) has `Γ(λτ/2)`; that
//! prefactor cancels from (46).
//!
//! # What Richardson leaves open, and what was measured instead
//!
//! ⚠ **The variance drive needs `∂P₀/∂V`, and the paper does not say how to compute it.** The
//! obvious choice is to read the slope off the steady-state equation (43),
//! `∂P₀/∂V = −G P₀ − τJ₀/σ²`, at each lattice point. For the EIF that is two nearly equal terms of
//! order `τr₀/σ²` whose difference is the true slope, of order `τr₀/(Δ_T ψ)`, and each term carries
//! the scheme's first-order error; where `ψ` is large their difference is dominated by that error.
//! Measured on Fig. 2's case i at 10 kHz, the EIF's variance response read that way is `−0.29 +
//! 9.23i` times Richardson's asymptote (51) at 10 µV, and its distance from the asymptote goes
//! 9.3, 4.2, 1.8, 0.77 as the step halves. This module differences the lattice density instead,
//! `(P₀⁽ᵏ⁾ − P₀⁽ᵏ⁻¹⁾)/Δ` over the cell the step integrates, which is within 1.1% of the asymptote at
//! every one of those steps: `the_variance_drive_is_differenced_not_read_off_the_ode`.
//!
//! ⚠ **The backward recursion overflows at high frequency, and the fix is exact.** Unscaled, the
//! response recursion's components pass `f64::MAX` just past the 10 kHz Richardson's figures reach,
//! and it returns `NaN`: on Fig. 1's case i at 1 µV above 10.8 kHz with the mean drive and 10.6 kHz
//! with the variance drive, and on Fig. 2's case i at 10 µV above 15.5 and 15.3 kHz. On the
//! negative real axis, which an eigenvalue search can visit, they underflow instead. The pairs
//! `(ĵ_r, p̂_r)` and `(ĵ_α, p̂_α)` never feed each other, so each is multiplied by `2⁻⁵⁰⁰` whenever
//! one of its components passes `2⁵⁰⁰` and by `2⁵⁰⁰` whenever all fall below `2⁻⁵⁰⁰`, its source
//! term is scaled to match, and the two counts are undone on the ratio at the end. A power of two
//! changes no mantissa, so where the unscaled recursion survives the rescaled one returns its bits:
//! `the_response_survives_where_the_recursion_would_overflow` and
//! `a_decaying_recursion_and_a_far_subthreshold_neuron_keep_their_digits`. The pairs need separate
//! scales: for a neuron 30σ below threshold, whose Siegert rate is `2.2 × 10⁻¹⁹³` Hz, the rate pair
//! passes `2⁵⁰⁰` while, at `λ = 10⁻³⁰⁰`, the drive pair is `4 × 10⁻²⁹⁶`, which divided by `2⁵⁰⁰` is
//! below the smallest subnormal.
//!
//! The complex division at the end has the same trap in miniature. Written as `z/w = z w̄/|w|²` it
//! overflows once the fluxes pass `√f64::MAX ≈ 1.3 × 10¹⁵⁴`, which on Fig. 2's case i they do
//! between 5 and 6 kHz — about a third of the frequency at which they themselves overflow — and
//! returns `NaN`; [`Complex`] divides by Smith's algorithm, which never forms `|w|²`.
//!
//! ⚠ **Three of Richardson's high-frequency driving terms are printed without their dimensions.**
//! In the EIF section he rewrites each `F_α` in the variable `m = ωτΔ_T/ψ` (49). The mean and
//! variance forms before eq. (50) are rates, as `F_α` must be. Eq. (50) prints
//! `F_g = m (g₁r₀/g₀ω)(…)`, which is dimensionless: the definition (33) gives `ωτ₀` where it has
//! `ω`. Eq. (53) prints `F_VT = (V_T1/Δ_T) r₀τ₀` — a spurious `τ₀` — and
//! `F_ΔT = (Δ_T1/Δ_T)(log ωτ₀ − log m − 1)`, with no `r₀` at all. The asymptotes derived from them,
//! eqs. (52) and (54), are the ones the correctly dimensioned terms give, and the lattice drive
//! near threshold is the corrected term to 0.48%: `eqs_50_and_53_are_printed_without_their_dimensions`.
//!
//! Eq. (54)'s `Δ_T` response, `−r₀(Δ_T1/Δ_T) log ωτ`, is by Richardson's own account "given here
//! to leading order" (p. 7), and Table I's caption says "the constant iπ/2 terms were dropped". The
//! next terms, `+ γ − 1 + iπ/2` — the ones (52) keeps — come from carrying (53) through his
//! eq. (46), and the dimensional slip in (53) changes only their common scale. They are what put
//! Fig. 3 Bii's dashed asymptote near −160° at 1 kHz, not at the leading order's −180°; this module
//! converges to the full form.
//!
//! # The network
//!
//! Brunel's sparse network of `C_E` excitatory, `C_I = γC_E` inhibitory and `C_E` external inputs
//! per neuron, each a `J`-volt jump, gives a neuron the white-noise drive of his eq. (20):
//! `μ = C_E Jτ[ν_ext + ν(1 − gγ)]`, `σ_B² = C_E J²τ[ν_ext + ν(1 + g²γ)]`. [`NetworkDensity`]
//! takes that drive from [`crate::meanfield::BrunelNetwork`], finds the self-consistent rate
//! `ν = φ(μ(ν), σ(ν))` with threshold integration as `φ`, and linearises the loop: a rate
//! perturbation `δν e^{λt}` arrives a delay `D` later as a perturbation of both the mean and the
//! variance, so the asynchronous state has an eigenvalue wherever
//!
//! ```text
//! 1 = e^{−λD} [ (∂μ/∂ν) r̂_E(λ) + (∂σ_B²/∂ν) r̂_{σ_B²}(λ) ]
//! ```
//!
//! This is the linear-stability problem of Brunel's section 5 and Appendix A.3, with the two
//! response functions taken from the lattice instead of from confluent hypergeometric functions.
//! At Table 1's parameters it finds the eigenvalues that eq. (46) itself has — 190.2 Hz, unstable,
//! at point B and 29.0 Hz, unstable, at point D, where the table prints 190 Hz and 29 Hz; and at
//! point C, the asynchronous state for which the table prints no frequency, the root with the
//! largest real part in the window the reference searched (growth rates −600 to 400 per second,
//! 5 Hz to 1 kHz) is stable. It puts the fast Hopf line at `g = 8` where Fig. 7 draws it for
//! `C_E = 1000`, `J = 0.1` mV, and where Fig. 2A draws it for Fig. 1's network, `C_E = 4000`,
//! `J = 0.2` mV — Fig. 2 prints no parameters of its own, and page 196 calls it the network "with
//! higher connectivity" — at the frequency Fig. 3 plots.
//!
//! Table 1 prints 38.0 Hz for point C, where eqs. (20)–(21) give 37.9497 Hz, which rounds to 37.9.
//! That is 0.0003 Hz — eight parts in a million — below the rounding boundary, within any
//! plausible error of the paper's own evaluation, so it is read as rounding and not counted as a
//! defect. Points B and D, 55.8 Hz and 6.5 Hz, round correctly.
//!
//! # What is checked
//!
//! - Appendix A worked by hand on four cells, for the LIF and the EIF.
//! - The lattice rate against [`SiegertInput::rate`] over 56 neurons, the order measured on
//!   halving `Δ`; [`SiegertInput::rate`] against an independent `SciPy` quadrature.
//! - The lattice density against Brunel's eq. (19) at every lattice point, converging at first
//!   order; eq. (19) against `SciPy`, its normalisation (12) and its boundary slopes (9) and (10).
//! - The four steady-state rates Richardson's Figs. 1 and 2 print, and the EIF's rate against an
//!   independent `SciPy` integration of the same equation by a different route.
//! - The LIF's mean and variance responses from 1 Hz to 10 kHz, with and without a refractory
//!   period, against Brunel's eq. (46) and Klett and Lindner's eq. (10), within `6.2 × 10⁻³` at
//!   1 µV and converging at first order.
//! - The EIF's five responses from 1 Hz to 10 kHz against Richardson's first-order equations
//!   integrated as ODEs by `SciPy`'s Radau — the same equations, none of the lattice — a route
//!   that reproduces eq. (46) on the LIF to `5 × 10⁻¹³`; within `9.0 × 10⁻³` at 2.5 µV and
//!   converging at first order.
//! - Every response's zero-frequency limit against the derivative of the stationary rate, the
//!   LIF's three against `SciPy`'s derivative of the Siegert rate; and its high-frequency limit
//!   against Richardson's eqs. (34)–(36), (51), (52) and (54), with the phases his captions print.
//! - The resonances, phase zeros and rate crossing his captions describe, located on the closed
//!   form (eq. 46) for the LIF and on the ODE route for the EIF, and the lattice's converging to
//!   them.
//! - Brunel's Table 1 rates and frequencies against eqs. (20)–(21) and the roots of eq. (46), the
//!   characteristic function at one point against eq. (46)'s, and the fast Hopf line at `g = 8` for
//!   two networks; his high-activity rate against eq. (23).
//! - The rescaled response recursion against the unscaled one, bit for bit wherever the unscaled one
//!   survives, on growing and decaying recursions; and Smith's division where `z w̄/|w|²` fails.
//! - Every refusal, by its rendered message.
//!
//! Richardson's own code is "available on request" (p. 13, after (A15)); this review did not locate
//! a public copy of it, so the external references are the authors' closed forms and independent
//! integrations, in `tools/density_reference.py`.
//!
//! # Not here
//!
//! Conductance-based noise (Richardson's section IV), whose variance depends on voltage; the
//! algebraic IF models (his section III D); Brunel's model B with two populations, his randomly
//! distributed delays (Appendix A.4) and his finite-size power spectrum (Appendix A.5); and the
//! rest of the phase diagram — the Hopf line is located at one `g`, not traced.

use crate::meanfield::{BrunelNetwork, MeanFieldError, SiegertInput, siegert_integral};
use core::fmt;
use core::ops::{Add, Div, Mul, Neg, Sub};

/// Why a density calculation was refused.
#[derive(Debug, Clone, PartialEq)]
pub enum DensityError {
    /// A value that must be finite was not.
    NonFinite {
        /// Which.
        what: &'static str,
        /// Its value.
        value: f64,
    },
    /// A value that must be finite and positive was not.
    NotPositive {
        /// Which.
        what: &'static str,
        /// Its value.
        value: f64,
    },
    /// A refractory period or a rate was negative.
    Negative {
        /// Which.
        what: &'static str,
        /// Its value.
        value: f64,
    },
    /// The lattice needs `v_lb < v_re < v_th`.
    Ordering {
        /// `v_lb`, volts.
        v_lb: f64,
        /// `v_re`, volts.
        v_re: f64,
        /// `v_th`, volts.
        v_th: f64,
    },
    /// A voltage that must be a lattice point is not: Richardson's Appendix A puts both the
    /// threshold and the reset on the lattice.
    OffLattice {
        /// Which voltage.
        what: &'static str,
        /// How many steps of `dv` it lies above `v_lb`.
        cells: f64,
    },
    /// More lattice cells than [`MAX_CELLS`].
    TooManyCells {
        /// How many were asked for.
        cells: f64,
    },
    /// A quantity overflowed `f64`.
    Overflow {
        /// Which.
        what: &'static str,
        /// Its value.
        value: f64,
    },
    /// A result cannot be represented in `f64` here: a lattice density that overflows (a noise far
    /// too small for the drive) or underflows to zero, or a response past `f64::MAX`.
    NotRepresentable {
        /// Which quantity.
        what: &'static str,
    },
    /// Brunel's closed form cannot be evaluated by its route, `ν₀ · e^{−y²} ∫ e^{u²} du`, at these
    /// parameters, although the density itself may be an ordinary number. A reduced voltage
    /// `(V − μ)/σ` is infinite; or `y_θ = (θ − μ)/σ` is past `√ln(f64::MAX/2) ≈ 26.63`, where
    /// `2e^{y_θ²}` overflows, the integral is infinite and [`SiegertInput::rate`] reports `ν₀` as
    /// zero. At `y_θ = 30` (`μ = 17` mV, `σ_B = 0.1` mV) `ν₀` is `1.2 × 10⁻³⁸⁸` Hz, below the
    /// smallest `f64`, and the density at 16 mV is `2.1 × 10⁻⁴⁰` per volt
    /// (`tools/density_reference.py` section 2b, `mpmath` at 60 digits).
    Unevaluable {
        /// Which quantity.
        what: &'static str,
    },
    /// A modulation of the spike-generating current was asked of a model without one, the
    /// closed form of the LIF was asked of an EIF, or a network's `ν_thr` is undefined.
    WrongModel {
        /// What was asked.
        what: &'static str,
    },
    /// `λ = 0`, where the two backward solutions both end at zero flux and their ratio is `0/0`.
    ZeroFrequency,
    /// The eigenvalue search did not settle, or left the finite plane.
    NoConvergence {
        /// Newton steps taken.
        steps: usize,
        /// The last point, real part, per second.
        re: f64,
        /// The last point, imaginary part, radians per second.
        im: f64,
    },
    /// [`crate::meanfield`] refused its part.
    MeanField(MeanFieldError),
}

impl fmt::Display for DensityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFinite { what, value } => write!(f, "{what} = {value} is not finite"),
            Self::NotPositive { what, value } => write!(f, "{what} = {value} must be finite and positive"),
            Self::Negative { what, value } => write!(f, "{what} = {value} must not be negative"),
            Self::Ordering { v_lb, v_re, v_th } => {
                write!(f, "the lattice needs v_lb < v_re < v_th, and was given {v_lb}, {v_re}, {v_th}")
            }
            Self::OffLattice { what, cells } => {
                write!(f, "{what} is {cells} steps above v_lb, which is not a lattice point")
            }
            Self::TooManyCells { cells } => {
                write!(f, "{cells} lattice cells is more than the {MAX_CELLS} this module will allocate")
            }
            Self::Overflow { what, value } => write!(f, "{what} overflowed to {value}"),
            Self::NotRepresentable { what } => write!(f, "{what} is not representable in f64 at these parameters"),
            Self::Unevaluable { what } => {
                write!(f, "{what} cannot be evaluated in f64 at these parameters: nu0, e^(-y^2) or the integral of e^(u^2) leaves the finite range")
            }
            Self::WrongModel { what } => write!(f, "{what} is not defined for this model"),
            Self::ZeroFrequency => f.write_str("the response at lambda = 0 is 0/0; ask for a small nonzero frequency"),
            Self::NoConvergence { steps, re, im } => {
                write!(f, "the eigenvalue search did not settle in {steps} steps; it stopped at lambda = ({re}, {im}) per second")
            }
            Self::MeanField(e) => write!(f, "meanfield: {e}"),
        }
    }
}

impl std::error::Error for DensityError {}

fn finite(what: &'static str, value: f64) -> Result<f64, DensityError> {
    if value.is_finite() { Ok(value) } else { Err(DensityError::NonFinite { what, value }) }
}

fn positive(what: &'static str, value: f64) -> Result<f64, DensityError> {
    if value.is_finite() && value > 0.0 { Ok(value) } else { Err(DensityError::NotPositive { what, value }) }
}

/// The most lattice cells [`WhiteNoiseIf::stationary`] will allocate. A [`Stationary`] holds five
/// vectors of this many `f64`, 400 MB, and building one needs two more; the finest lattice the
/// tests allocate has 200 000 cells (Fig. 1's 50 mV at 0.25 µV).
pub const MAX_CELLS: f64 = 1e7;

/// How far, in lattice steps, a voltage may sit from a lattice point and still count as on it.
///
/// Richardson's step of 10 µV divides his 50 mV and 100 mV ranges exactly in floating point, and
/// puts his −60 mV reset `4.5 × 10⁻¹³` of a step from its lattice point; a voltage off by a
/// millionth of a step is a different lattice, not rounding.
pub const LATTICE_TOLERANCE: f64 = 1e-6;

/// A complex number: the response functions are complex, and this crate has no dependencies.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Complex {
    /// Real part.
    pub re: f64,
    /// Imaginary part.
    pub im: f64,
}

impl Complex {
    /// `re + i im`.
    #[must_use]
    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    /// `|z|`.
    #[must_use]
    pub fn norm(self) -> f64 {
        self.re.hypot(self.im)
    }

    /// The argument, radians in `(−π, π]`.
    #[must_use]
    pub fn arg(self) -> f64 {
        self.im.atan2(self.re)
    }

    /// `e^z`.
    #[must_use]
    pub fn exp(self) -> Self {
        let m = self.re.exp();
        Self::new(m * self.im.cos(), m * self.im.sin())
    }

    /// `k z` for real `k`.
    #[must_use]
    pub fn scale(self, k: f64) -> Self {
        Self::new(k * self.re, k * self.im)
    }
}

impl Add for Complex {
    type Output = Self;
    fn add(self, o: Self) -> Self {
        Self::new(self.re + o.re, self.im + o.im)
    }
}

impl Sub for Complex {
    type Output = Self;
    fn sub(self, o: Self) -> Self {
        Self::new(self.re - o.re, self.im - o.im)
    }
}

impl Mul for Complex {
    type Output = Self;
    fn mul(self, o: Self) -> Self {
        Self::new(self.re * o.re - self.im * o.im, self.re * o.im + self.im * o.re)
    }
}

/// Smith's algorithm: it never forms `|o|²`, which underflows to zero for `|o|` below about
/// `10⁻¹⁵⁴` and overflows above `10¹⁵⁴`, where the quotient itself is perfectly representable.
impl Div for Complex {
    type Output = Self;
    fn div(self, o: Self) -> Self {
        if o.re.abs() >= o.im.abs() {
            let (t, d) = (o.im / o.re, o.re + o.im * (o.im / o.re));
            Self::new((self.re + self.im * t) / d, (self.im - self.re * t) / d)
        } else {
            let (t, d) = (o.re / o.im, o.im + o.re * (o.re / o.im));
            Self::new((self.re * t + self.im) / d, (self.im * t - self.re) / d)
        }
    }
}

impl Neg for Complex {
    type Output = Self;
    fn neg(self) -> Self {
        Self::new(-self.re, -self.im)
    }
}

/// The spike-generating current `ψ(V)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Model {
    /// The leaky integrate-and-fire neuron: `ψ = 0`, Richardson's eq. (25).
    Lif,
    /// The exponential integrate-and-fire neuron: `ψ = Δ_T exp((V − V_T)/Δ_T)`, eq. (48).
    Eif {
        /// `V_T`, volts: where the exponential term becomes significant.
        v_t: f64,
        /// `Δ_T`, volts: the sharpness of the spike.
        delta_t: f64,
    },
}

/// Which parameter is modulated, each per unit of its own amplitude `α₁`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Modulation {
    /// The resting potential `E` — equivalently a modulated input current. `F_E = −E₁P₀/τ`,
    /// eq. (33). The response is in hertz per volt.
    Mean,
    /// The noise variance `σ²`, in Richardson's convention. `F_σ² = (σ₁²/τ) ∂P₀/∂V`, eq. (33).
    /// Hertz per volt squared.
    Variance,
    /// The leak conductance, as `g₁/g₀`, at fixed `σ²/τ`. `F_g = (g₁/g₀)(V − E)P₀/τ`, eq. (33), for
    /// the EIF too, as eq. (50) takes it: the modulated conductance reverses at `E` and leaves the
    /// spike current `ψ/τ` alone. Hertz per unit of `g₁/g₀`.
    Conductance,
    /// The EIF's `V_T`. `F_VT = V_T1 ψP₀/(Δ_T τ)`, from definition (13). Hertz per volt.
    SpikeThreshold,
    /// The EIF's `Δ_T`. `F_ΔT = −Δ_T1 (ψ/(Δ_T τ))(1 − (V − V_T)/Δ_T) P₀`, from definition (13).
    /// Hertz per volt.
    SpikeWidth,
}

/// An integrate-and-fire neuron under Gaussian white noise, in Richardson's convention.
///
/// All voltages in volts, times in seconds. `sigma` is the standard deviation the free membrane
/// potential would have without a threshold, NOT Brunel's `σ`, which is `√2` times it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WhiteNoiseIf {
    /// The spike-generating current.
    pub model: Model,
    /// `τ`, seconds.
    pub tau: f64,
    /// `E`, volts: where the free membrane would settle, mean drive included.
    pub e0: f64,
    /// `σ`, volts, Richardson's: the free membrane's standard deviation.
    pub sigma: f64,
    /// `V_th`, volts: the absorbing threshold.
    pub v_th: f64,
    /// `V_re`, volts: the reset.
    pub v_re: f64,
    /// `V_lb`, volts: the reflecting lower bound of eq. (6).
    pub v_lb: f64,
    /// `τ_ref`, seconds: the absolute refractory period of section V B. Zero or more.
    pub t_ref: f64,
}

/// The lattice of Richardson's Appendix A: `V⁽ᵏ⁾ = v_lb + k dv`, `k = 0..=n`, with the threshold
/// at `k = n` and the reset at `k = k_re`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Lattice {
    /// Cells: `V⁽ⁿ⁾ = V_th`.
    pub n: usize,
    /// The reset's lattice index.
    pub k_re: usize,
    /// The step `Δ`, volts: `(V_th − V_lb)/n` exactly.
    pub dv: f64,
}

impl WhiteNoiseIf {
    /// Richardson's Fig. 1, case i: the LIF with `E₀ = −45` mV above a −50 mV threshold,
    /// `σ₀ = 1` mV, reset −60 mV, `V_lb = −100` mV, `τ₀ = 20` ms. The caption prints `r₀ = 46` Hz.
    pub const FIG1_CASE_I: Self = Self {
        model: Model::Lif,
        tau: 20e-3,
        e0: -45e-3,
        sigma: 1e-3,
        v_th: -50e-3,
        v_re: -60e-3,
        v_lb: -100e-3,
        t_ref: 0.0,
    };

    /// Richardson's Fig. 1, case ii: `E₀ = −60` mV, `σ₀ = 5` mV, noise-driven. The caption prints
    /// `r₀ = 4.8` Hz.
    pub const FIG1_CASE_II: Self = Self { e0: -60e-3, sigma: 5e-3, ..Self::FIG1_CASE_I };

    /// Richardson's Fig. 2, case i: the EIF with `V_T = −53` mV, `Δ_T = 3` mV, `V_th = 0` mV,
    /// `E₀ = −45` mV, `σ₀ = 2` mV. The caption prints `r₀ = 44` Hz.
    pub const FIG2_CASE_I: Self = Self {
        model: Model::Eif { v_t: -53e-3, delta_t: 3e-3 },
        tau: 20e-3,
        e0: -45e-3,
        sigma: 2e-3,
        v_th: 0.0,
        v_re: -60e-3,
        v_lb: -100e-3,
        t_ref: 0.0,
    };

    /// Richardson's Fig. 2, case ii: `E₀ = −60` mV, `σ₀ = 6` mV. The caption prints `r₀ = 5.6` Hz.
    pub const FIG2_CASE_II: Self = Self { e0: -60e-3, sigma: 6e-3, ..Self::FIG2_CASE_I };

    /// The integration step of Figs. 1 and 2, 10 µV.
    pub const FIG_STEP: f64 = 10e-6;

    /// Every parameter finite, the ones that must be positive positive, `τ_ref ≥ 0`,
    /// `v_lb < v_re < v_th`, and for the EIF a `ψ(V_th)` that does not overflow.
    ///
    /// # Errors
    ///
    /// [`DensityError::NonFinite`], [`DensityError::NotPositive`], [`DensityError::Negative`],
    /// [`DensityError::Ordering`] or [`DensityError::Overflow`], naming the first that fails.
    pub fn check(&self) -> Result<(), DensityError> {
        positive("tau", self.tau)?;
        positive("sigma", self.sigma)?;
        finite("e0", self.e0)?;
        finite("v_th", self.v_th)?;
        finite("v_re", self.v_re)?;
        finite("v_lb", self.v_lb)?;
        if finite("t_ref", self.t_ref)? < 0.0 {
            return Err(DensityError::Negative { what: "t_ref", value: self.t_ref });
        }
        if !(self.v_lb < self.v_re && self.v_re < self.v_th) {
            return Err(DensityError::Ordering { v_lb: self.v_lb, v_re: self.v_re, v_th: self.v_th });
        }
        if let Model::Eif { v_t, delta_t } = self.model {
            finite("v_t", v_t)?;
            positive("delta_t", delta_t)?;
            let top = self.psi(self.v_th);
            if !top.is_finite() {
                return Err(DensityError::Overflow { what: "psi(v_th)", value: top });
            }
        }
        Ok(())
    }

    /// `ψ(V)`, volts.
    #[must_use]
    pub fn psi(&self, v: f64) -> f64 {
        match self.model {
            Model::Lif => 0.0,
            Model::Eif { v_t, delta_t } => delta_t * ((v - v_t) / delta_t).exp(),
        }
    }

    /// The lattice with step as close to `dv` as puts the threshold on a lattice point, and the
    /// reset on one too.
    ///
    /// # Errors
    ///
    /// [`DensityError::NotPositive`] for a `dv` that is not; [`DensityError::TooManyCells`] past
    /// [`MAX_CELLS`]; [`DensityError::OffLattice`] if `V_th` or `V_re` is more than
    /// [`LATTICE_TOLERANCE`] steps from a lattice point, or the reset rounds onto an end; whatever
    /// [`WhiteNoiseIf::check`] refuses.
    pub fn lattice(&self, dv: f64) -> Result<Lattice, DensityError> {
        self.check()?;
        positive("dv", dv)?;
        let cells = (self.v_th - self.v_lb) / dv;
        if !(cells <= MAX_CELLS) {
            return Err(DensityError::TooManyCells { cells });
        }
        let n = cells.round();
        if (cells - n).abs() > LATTICE_TOLERANCE {
            return Err(DensityError::OffLattice { what: "v_th", cells });
        }
        let step = (self.v_th - self.v_lb) / n;
        let at_re = (self.v_re - self.v_lb) / step;
        let k_re = at_re.round();
        if (at_re - k_re).abs() > LATTICE_TOLERANCE || !(k_re >= 1.0 && k_re < n) {
            return Err(DensityError::OffLattice { what: "v_re", cells: at_re });
        }
        Ok(Lattice { n: n as usize, k_re: k_re as usize, dv: step })
    }

    /// The steady state by threshold integration: Richardson's (A5)–(A8), with the refractory
    /// normalisation of eq. (91).
    ///
    /// # Errors
    ///
    /// Whatever [`WhiteNoiseIf::lattice`] refuses; [`DensityError::NotRepresentable`] if the
    /// unnormalised density overflows, which a noise far too small for the drive does, or underflows
    /// to zero.
    pub fn stationary(&self, dv: f64) -> Result<Stationary, DensityError> {
        let lattice = self.lattice(dv)?;
        let Lattice { n, k_re, dv } = lattice;
        let s2 = self.sigma * self.sigma;
        let voltage: Vec<f64> = (0..=n).map(|k| self.v_lb + k as f64 * dv).collect();
        let (mut a, mut b) = (vec![0.0; n + 1], vec![0.0; n + 1]);
        let (mut p, mut j) = (vec![0.0; n + 1], vec![0.0; n + 1]);
        j[n] = 1.0;
        for k in (1..=n).rev() {
            let v = voltage[k];
            let x = dv * (v - self.e0 - self.psi(v)) / s2;
            a[k] = x.exp();
            b[k] = (if x == 0.0 { 1.0 } else { x.exp_m1() / x }) / s2;
            j[k - 1] = if k == k_re + 1 { j[k] - 1.0 } else { j[k] };
            p[k - 1] = p[k] * a[k] + dv * self.tau * j[k] * b[k];
        }
        let mass = p.iter().sum::<f64>() * dv;
        if !(mass.is_finite() && mass > 0.0) {
            return Err(DensityError::NotRepresentable { what: "the unnormalised density" });
        }
        let r0 = 1.0 / mass;
        let rate = r0 / (1.0 + self.t_ref * r0);
        let density = p.iter().map(|x| rate * x).collect();
        let flux = j.iter().map(|x| rate * x).collect();
        Ok(Stationary { neuron: *self, lattice, voltage, density, flux, a, b, rate })
    }

    /// The same neuron in Brunel's convention, `σ_B = √2 σ`, as [`crate::meanfield`] takes it.
    ///
    /// # Errors
    ///
    /// [`DensityError::WrongModel`] for an EIF, which has no Siegert formula; whatever
    /// [`WhiteNoiseIf::check`] refuses.
    pub fn siegert(&self) -> Result<SiegertInput, DensityError> {
        self.check()?;
        if self.model != Model::Lif {
            return Err(DensityError::WrongModel { what: "the Siegert rate of a non-leaky model" });
        }
        SiegertInput::new(self.tau, self.t_ref, self.v_th, self.v_re, self.e0, core::f64::consts::SQRT_2 * self.sigma)
            .map_err(DensityError::MeanField)
    }

    /// The LIF [`crate::meanfield`] describes, in Richardson's convention, `σ = σ_B/√2`, with the
    /// lattice closed at `v_lb`.
    #[must_use]
    pub fn from_siegert(input: &SiegertInput, v_lb: f64) -> Self {
        Self {
            model: Model::Lif,
            tau: input.tau_m,
            e0: input.mu,
            sigma: input.sigma / core::f64::consts::SQRT_2,
            v_th: input.v_th,
            v_re: input.v_reset,
            v_lb,
            t_ref: input.t_ref,
        }
    }
}

/// The response recursion is rescaled by `2⁻ᴺ` whenever a component passes `2ᴺ`, with this `N`: a
/// power of two, so that no mantissa changes.
const RESCALE_POWER: i32 = 500;

/// A steady state on its lattice, and the linear response about it.
#[derive(Debug, Clone, PartialEq)]
pub struct Stationary {
    neuron: WhiteNoiseIf,
    lattice: Lattice,
    voltage: Vec<f64>,
    density: Vec<f64>,
    flux: Vec<f64>,
    a: Vec<f64>,
    b: Vec<f64>,
    rate: f64,
}

impl Stationary {
    /// The firing rate, hertz: `r₀/(1 + τ_ref r₀)`, eq. (91).
    #[must_use]
    pub fn rate(&self) -> f64 {
        self.rate
    }

    /// The neuron this is the steady state of.
    #[must_use]
    pub fn neuron(&self) -> &WhiteNoiseIf {
        &self.neuron
    }

    /// The lattice.
    #[must_use]
    pub fn lattice(&self) -> Lattice {
        self.lattice
    }

    /// `V⁽ᵏ⁾`, volts.
    #[must_use]
    pub fn voltage(&self) -> &[f64] {
        &self.voltage
    }

    /// `P₀⁽ᵏ⁾`, per volt: the density of non-refractory neurons, which integrates to
    /// `1 − rate · τ_ref`.
    #[must_use]
    pub fn density(&self) -> &[f64] {
        &self.density
    }

    /// `J₀⁽ᵏ⁾`, hertz: the rate above the reset and zero at and below it.
    #[must_use]
    pub fn flux(&self) -> &[f64] {
        &self.flux
    }

    /// `Σₖ Δ P₀⁽ᵏ⁾`: the non-refractory fraction, `1 − rate · τ_ref`.
    #[must_use]
    pub fn mass(&self) -> f64 {
        self.density.iter().sum::<f64>() * self.lattice.dv
    }

    /// `(V_T, Δ_T)` for the EIF; `None` for the LIF, which has no spike current to modulate.
    fn eif(&self) -> Option<(f64, f64)> {
        match self.neuron.model {
            Model::Eif { v_t, delta_t } => Some((v_t, delta_t)),
            Model::Lif => None,
        }
    }

    /// The inhomogeneous term `f_α = F_α/α₁` on the cell `[V⁽ᵏ⁻¹⁾, V⁽ᵏ⁾]` that the step from `k`
    /// integrates, `k ≥ 1`, per unit of the modulated parameter. `None` for a spike-current
    /// modulation of the LIF.
    fn drive_at(&self, modulation: Modulation, k: usize) -> Option<f64> {
        let WhiteNoiseIf { tau, e0, .. } = self.neuron;
        let (v, rho) = (self.voltage[k], self.density[k]);
        Some(match modulation {
            Modulation::Mean => -rho / tau,
            Modulation::Variance => (rho - self.density[k - 1]) / (self.lattice.dv * tau),
            Modulation::Conductance => (v - e0) * rho / tau,
            Modulation::SpikeThreshold => {
                let (_, delta_t) = self.eif()?;
                self.neuron.psi(v) * rho / (delta_t * tau)
            }
            Modulation::SpikeWidth => {
                let (v_t, delta_t) = self.eif()?;
                -self.neuron.psi(v) / (delta_t * tau) * (1.0 - (v - v_t) / delta_t) * rho
            }
        })
    }

    /// The first-order rate modulation `r̂_α(λ)` per unit `α₁`, by Richardson's (A9)–(A13), with
    /// `iω` generalised to any complex `λ` and the refractory reset term `e^{−λτ_ref}` of eq. (92).
    ///
    /// For a modulation `α₀ + α₁ cos ωt` the rate is `r₀ + α₁|r̂| cos(ωt + arg r̂)` at `λ = iω`.
    ///
    /// # Errors
    ///
    /// [`DensityError::NonFinite`] for a `λ` that is not; [`DensityError::ZeroFrequency`] at
    /// `λ = 0`; [`DensityError::WrongModel`] for a spike-current modulation of a LIF;
    /// [`DensityError::NotRepresentable`] where the response itself is past `f64::MAX`, which off
    /// the imaginary axis, near a pole of it, it can be.
    pub fn response(&self, modulation: Modulation, lambda: Complex) -> Result<Complex, DensityError> {
        let r = self.integrate(lambda, 2f64.powi(RESCALE_POWER), |k| {
            self.drive_at(modulation, k).ok_or(DensityError::WrongModel { what: "a spike-current modulation of the LIF" })
        })?;
        if r.re.is_finite() && r.im.is_finite() { Ok(r) } else { Err(DensityError::NotRepresentable { what: "the response" }) }
    }

    /// The backward recursion (A9)–(A12) with drive `f(k)`, from `V_th` to `V_lb`.
    ///
    /// The pairs `(ĵ_r, p̂_r)` and `(ĵ_α, p̂_α)` never feed each other, so each carries its own
    /// power-of-two scale: a pair is divided by `rescale_at` whenever one of its components passes
    /// `rescale_at` in magnitude and multiplied by it whenever all of them fall below
    /// `1/rescale_at`, and its source term — the reset for the first pair, the drive for the second
    /// — is scaled to match. `rescale_at` must be a power of two, or `f64::INFINITY` for no
    /// rescaling.
    fn recur(
        &self,
        lambda: Complex,
        rescale_at: f64,
        f: impl Fn(usize) -> Result<f64, DensityError>,
    ) -> Result<Ends, DensityError> {
        finite("lambda.re", lambda.re)?;
        finite("lambda.im", lambda.im)?;
        if lambda.re == 0.0 && lambda.im == 0.0 {
            return Err(DensityError::ZeroFrequency);
        }
        let Lattice { n, k_re, dv } = self.lattice;
        let tau = self.neuron.tau;
        let reset = lambda.scale(-self.neuron.t_ref).exp();
        let step = lambda.scale(dv);
        let zero = Complex::new(0.0, 0.0);
        let (mut jr, mut pr, mut ja, mut pa) = (Complex::new(1.0, 0.0), zero, zero, zero);
        let (mut shift_r, mut shift_a) = (0i64, 0i64);
        for k in (1..=n).rev() {
            let w = dv * tau * self.b[k];
            let mut jr_next = jr + step * pr;
            if k == k_re + 1 {
                jr_next = jr_next - Complex::new(shifted(reset.re, rescale_at, shift_r), shifted(reset.im, rescale_at, shift_r));
            }
            let pr_next = pr.scale(self.a[k]) + jr.scale(w);
            let ja_next = ja + step * pa;
            let pa_next = pa.scale(self.a[k]) + (ja + Complex::new(shifted(f(k)?, rescale_at, shift_a), 0.0)).scale(w);
            (jr, pr, ja, pa) = (jr_next, pr_next, ja_next, pa_next);
            rebalance(&mut jr, &mut pr, &mut shift_r, rescale_at);
            rebalance(&mut ja, &mut pa, &mut shift_a, rescale_at);
        }
        Ok(Ends { ja, jr, shift_a, shift_r })
    }

    /// Richardson's (A13), `r̂ = −ĵ_α⁽⁰⁾/ĵ_r⁽⁰⁾`, from [`Stationary::recur`], with the two shifts
    /// undone on the ratio.
    fn integrate(
        &self,
        lambda: Complex,
        rescale_at: f64,
        f: impl Fn(usize) -> Result<f64, DensityError>,
    ) -> Result<Complex, DensityError> {
        let Ends { ja, jr, shift_a, shift_r } = self.recur(lambda, rescale_at, f)?;
        let ratio = -(ja / jr);
        let back = shift_r - shift_a;
        Ok(Complex::new(shifted(ratio.re, rescale_at, back), shifted(ratio.im, rescale_at, back)))
    }
}

/// The two fluxes at `V_lb` as [`Stationary::recur`] stores them, and their shifts: the true
/// `ĵ_α⁽⁰⁾` is `ja · rescale_at^shift_a`, and likewise `ĵ_r⁽⁰⁾`.
#[derive(Debug, Clone, Copy)]
struct Ends {
    ja: Complex,
    jr: Complex,
    shift_a: i64,
    shift_r: i64,
}

/// `x · base^(−shift)`, one multiplication at a time, which is exact for a power-of-two `base`
/// until the result leaves the normal range, and stopping once it has reached zero or infinity,
/// which further multiplications would not change: so a few steps, whatever `shift` is.
fn shifted(mut x: f64, base: f64, shift: i64) -> f64 {
    let by = if shift > 0 { 1.0 / base } else { base };
    for _ in 0..shift.unsigned_abs() {
        if x == 0.0 || x.is_infinite() {
            break;
        }
        x *= by;
    }
    x
}

/// Divide the pair `(j, p)` by `rescale_at` and count one up in `shift` if a component has passed
/// `rescale_at` in magnitude; multiply it and count one down if every component is below
/// `1/rescale_at`. At most once per step. (A pair that is exactly zero is multiplied too, which
/// changes nothing but the count.)
fn rebalance(j: &mut Complex, p: &mut Complex, shift: &mut i64, rescale_at: f64) {
    let big = [j.re, j.im, p.re, p.im].iter().fold(0.0f64, |m, x| m.max(x.abs()));
    if big > rescale_at {
        (*j, *p) = (j.scale(1.0 / rescale_at), p.scale(1.0 / rescale_at));
        *shift += 1;
    } else if big < 1.0 / rescale_at {
        (*j, *p) = (j.scale(rescale_at), p.scale(rescale_at));
        *shift -= 1;
    }
}

/// `e^{−y²} ∫_lo^hi e^{u²} du`, through `2e^{u²} = erfcx(u) + erfcx(−u)` and
/// [`siegert_integral`], for `lo ≤ hi`. `None` where [`siegert_integral`] refuses a limit, which it
/// does only for one that is not finite.
fn gauss_weighted(y: f64, lo: f64, hi: f64) -> Option<f64> {
    let twice = siegert_integral(lo, hi)? + siegert_integral(-hi, -lo)?;
    Some((-y * y).exp() * 0.5 * twice)
}

/// Validate a [`SiegertInput`] for the closed form: [`SiegertInput::new`]'s checks, `σ > 0`, and a
/// finite `v`. Returns the input and the reduced voltages `(y, y_r, y_θ)`, each `(· − μ)/σ`.
fn brunel_terms(input: &SiegertInput, v: f64) -> Result<(SiegertInput, f64, f64, f64), DensityError> {
    let i = SiegertInput::new(input.tau_m, input.t_ref, input.v_th, input.v_reset, input.mu, input.sigma)
        .map_err(DensityError::MeanField)?;
    positive("sigma", i.sigma)?;
    finite("v", v)?;
    Ok((i, (v - i.mu) / i.sigma, (i.v_reset - i.mu) / i.sigma, (i.v_th - i.mu) / i.sigma))
}

/// Brunel's stationary density, his eq. (19) with the Heaviside argument read as
/// `(V_r − μ)/σ`, per volt, in Brunel's convention (`input.sigma` is `σ_B`). Zero above
/// threshold. It integrates to `1 − ν₀τ_rp`.
///
/// # Errors
///
/// [`DensityError::MeanField`] for what [`SiegertInput::new`] refuses;
/// [`DensityError::NotPositive`] for `σ = 0`; [`DensityError::NonFinite`] for `v`;
/// [`DensityError::Unevaluable`] where a reduced voltage is infinite or `y_θ = (θ − μ)/σ` is past
/// `√ln(f64::MAX/2) ≈ 26.63`: there `ν₀` and the integral of `e^{u²}` leave the finite range,
/// though the density need not.
pub fn brunel_density(input: &SiegertInput, v: f64) -> Result<f64, DensityError> {
    let (i, y, yr, yth) = brunel_terms(input, v)?;
    let lo = y.max(yr).min(yth);
    let p = i.rate().zip(gauss_weighted(y, lo, yth)).map(|(nu, g)| 2.0 * nu * i.tau_m / i.sigma * g);
    match p {
        Some(p) if p.is_finite() => Ok(p),
        _ => Err(DensityError::Unevaluable { what: "brunel_density" }),
    }
}

/// `∂P₀/∂V` of [`brunel_density`], per volt squared: `(2ν₀τ/σ²)[−2y e^{−y²}I(y) − Θ(y − y_r)]`,
/// with Brunel's `Θ(0) = 0`, so at `V_r` it is the slope from below. Zero above threshold.
///
/// # Errors
///
/// As [`brunel_density`].
pub fn brunel_slope(input: &SiegertInput, v: f64) -> Result<f64, DensityError> {
    let (i, y, yr, yth) = brunel_terms(input, v)?;
    if v > i.v_th {
        return Ok(0.0);
    }
    let lo = y.max(yr).min(yth);
    let step = if y > yr { 1.0 } else { 0.0 };
    let d = i.rate().zip(gauss_weighted(y, lo, yth)).map(|(nu, g)| 2.0 * nu * i.tau_m / (i.sigma * i.sigma) * (-2.0 * y * g - step));
    match d {
        Some(d) if d.is_finite() => Ok(d),
        _ => Err(DensityError::Unevaluable { what: "brunel_slope" }),
    }
}

/// Brunel's sparse network with threshold integration as its transfer function.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NetworkDensity {
    /// The network: neuron, fan-in, `J`, `g`, `ν_ext/ν_thr` and the delay `D`.
    pub net: BrunelNetwork,
    /// `V_lb`, volts.
    pub v_lb: f64,
    /// The lattice step, volts.
    pub dv: f64,
}

impl NetworkDensity {
    /// The neuron one cell of the network is when every cell fires at `nu` hertz: eq. (20)'s drive
    /// through [`crate::meanfield::BalancedInput::siegert`], in Richardson's convention.
    ///
    /// # Errors
    ///
    /// [`DensityError::NonFinite`] or [`DensityError::Negative`] for `nu`;
    /// [`DensityError::WrongModel`] for a network whose `ν_thr` is undefined; whatever
    /// [`SiegertInput::new`] refuses.
    pub fn neuron(&self, nu: f64) -> Result<WhiteNoiseIf, DensityError> {
        if finite("nu", nu)? < 0.0 {
            return Err(DensityError::Negative { what: "nu", value: nu });
        }
        let input = self.net.input(nu).ok_or(DensityError::WrongModel { what: "nu_thr = (v_th - v_rest)/(c_exc j tau_m)" })?;
        let s = input.siegert(&self.net.neuron).map_err(DensityError::MeanField)?;
        Ok(WhiteNoiseIf::from_siegert(&s, self.v_lb))
    }

    /// The self-consistent rate `ν = φ(ν)`, hertz, by bisection on `[0, 1/τ_ref]`.
    ///
    /// The bracket is exact for threshold integration: the lattice rate is positive at `ν = 0` and
    /// below `1/τ_ref` everywhere, so `φ(ν) − ν` changes sign inside it. The root returned is the
    /// one bisection converges to; uniqueness is not claimed.
    ///
    /// # Errors
    ///
    /// [`DensityError::NotPositive`] for a network without a refractory period; whatever
    /// [`NetworkDensity::neuron`] and [`WhiteNoiseIf::stationary`] refuse at a bisection point.
    pub fn rate(&self) -> Result<f64, DensityError> {
        let t_ref = positive("t_ref", self.net.neuron.t_ref)?;
        let (mut lo, mut hi) = (0.0f64, 1.0 / t_ref);
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            if self.neuron(mid)?.stationary(self.dv)?.rate() > mid {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        Ok(0.5 * (lo + hi))
    }

    fn characteristic_on(&self, st: &Stationary, lambda: Complex) -> Result<Complex, DensityError> {
        let BrunelNetwork { c_exc, c_inh, j, g, delay, .. } = self.net;
        let tau = self.net.neuron.tau_m;
        let dmu = tau * j * (c_exc - g * c_inh);
        let dvar = tau * j * j * (c_exc + g * g * c_inh);
        let re = st.response(Modulation::Mean, lambda)?;
        let rv = st.response(Modulation::Variance, lambda)?;
        let loop_gain = (re.scale(dmu) + rv.scale(0.5 * dvar)) * lambda.scale(-delay).exp();
        Ok(Complex::new(1.0, 0.0) - loop_gain)
    }

    /// `1 − e^{−λD}[(∂μ/∂ν) r̂_E(λ) + (∂σ_B²/∂ν) r̂_{σ_B²}(λ)]` about the rate `nu0`: zero exactly at
    /// an eigenvalue of the asynchronous state. `r̂_{σ_B²}` is half Richardson's `r̂_σ²`, because
    /// `σ_B² = 2σ²`.
    ///
    /// # Errors
    ///
    /// Whatever [`NetworkDensity::neuron`], [`WhiteNoiseIf::stationary`] and
    /// [`Stationary::response`] refuse.
    pub fn characteristic(&self, nu0: f64, lambda: Complex) -> Result<Complex, DensityError> {
        let st = self.neuron(nu0)?.stationary(self.dv)?;
        self.characteristic_on(&st, lambda)
    }

    /// The eigenvalue nearest `guess` by Newton's method on [`NetworkDensity::characteristic`],
    /// with a forward-difference derivative of relative step [`NEWTON_DIFFERENCE`], stopped when a
    /// step moves the root by less than [`NEWTON_TOLERANCE`] of itself. The real part of
    /// [`Eigenvalue::lambda`] is the growth rate of a perturbation of the asynchronous state at rate
    /// `nu0`, per second; its imaginary part divided by `2π` is the frequency of the oscillation it
    /// grows into.
    ///
    /// # Errors
    ///
    /// [`DensityError::NoConvergence`] after [`NEWTON_STEPS`] steps, or at the first step that
    /// leaves the finite plane; whatever [`NetworkDensity::characteristic`] refuses.
    pub fn eigenvalue(&self, nu0: f64, guess: Complex) -> Result<Eigenvalue, DensityError> {
        let st = self.neuron(nu0)?.stationary(self.dv)?;
        let mut z = guess;
        for steps in 1..=NEWTON_STEPS {
            let f = self.characteristic_on(&st, z)?;
            let h = Complex::new(NEWTON_DIFFERENCE * z.norm(), 0.0);
            let slope = (self.characteristic_on(&st, z + h)? - f) / h;
            let dz = f / slope;
            z = z - dz;
            // A step off the finite plane is a failure, not a root: `hypot(NaN, ∞)` is `∞`, so
            // the test below would otherwise accept it.
            if !(z.re.is_finite() && z.im.is_finite()) {
                return Err(DensityError::NoConvergence { steps, re: z.re, im: z.im });
            }
            if dz.norm() <= NEWTON_TOLERANCE * z.norm() {
                return Ok(Eigenvalue { lambda: z, steps });
            }
        }
        Err(DensityError::NoConvergence { steps: NEWTON_STEPS, re: z.re, im: z.im })
    }
}

/// The most Newton steps [`NetworkDensity::eigenvalue`] takes.
pub const NEWTON_STEPS: usize = 50;

/// The forward-difference step of [`NetworkDensity::eigenvalue`]'s derivative, relative to `|λ|`.
/// The root does not depend on it; how fast Newton's method reaches the root does, which is why
/// [`Eigenvalue::steps`] is reported.
pub const NEWTON_DIFFERENCE: f64 = 1e-7;

/// [`NetworkDensity::eigenvalue`] stops when a step moves `λ` by less than this fraction of `|λ|`.
pub const NEWTON_TOLERANCE: f64 = 1e-12;

/// An eigenvalue of the asynchronous state, and how it was found.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Eigenvalue {
    /// `λ`, per second: growth rate `+ i` angular frequency.
    pub lambda: Complex,
    /// Newton steps taken.
    pub steps: usize,
}

#[cfg(test)]
mod tests {
    use super::{
        Complex, DensityError, Lattice, Model, Modulation, NetworkDensity, Stationary, WhiteNoiseIf,
        brunel_density, brunel_slope,
    };
    use crate::meanfield::{BrunelNetwork, SiegertInput};
    use crate::neuron::Lif;
    use core::f64::consts::{E, FRAC_PI_2, FRAC_PI_4, PI, SQRT_2};

    /// `λ = iω` at `f` hertz.
    fn at(f: f64) -> Complex {
        Complex::new(0.0, 2.0 * PI * f)
    }

    /// `|a − b|/|b|`.
    fn rel(a: Complex, b: Complex) -> f64 {
        (a - b).norm() / b.norm()
    }

    /// Richardson's `√(iωτ)` at `f` hertz, `e^{iπ/4}√(ωτ)`.
    fn sqrt_iwt(f: f64, tau: f64) -> Complex {
        let m = (2.0 * PI * f * tau).sqrt();
        Complex::new(m * FRAC_PI_4.cos(), m * FRAC_PI_4.sin())
    }

    /// Brunel's cell (p. 185): `τ = 20` ms, `θ = 20` mV, `V_r = 10` mV, `τ_rp = 2` ms, rest at 0.
    fn brunel_cell() -> Lif {
        Lif { tau_m: 20e-3, v_rest: 0.0, v_th: 20e-3, v_reset: 10e-3, r_m: 1.0, t_ref: 2e-3, v: 0.0, refractory: 0.0 }
    }

    /// Brunel's model A with `C_I = C_E/4` and `D = 1.5` ms.
    fn brunel_net(c_exc: f64, j: f64, g: f64, nu_ext_ratio: f64) -> BrunelNetwork {
        BrunelNetwork { neuron: brunel_cell(), c_exc, c_inh: c_exc / 4.0, j, g, nu_ext_ratio, delay: 1.5e-3 }
    }

    /// Appendix A on four cells, worked by hand.
    ///
    /// `τ = σ = 1`, `E = −1` V, lattice `−4, −3, −2, −1, 0` V with the reset at `−2`: then `ΔG⁽ᵏ⁾ =
    /// V⁽ᵏ⁾ + 1` and (A5)–(A6) give `p₀ = (e⁻², 1, e, e − 1, 0)` from the bottom up, the cell at
    /// `V = −1` passing through the `G = 0` branch where `B = 1/σ²`; so `r₀ = 1/(2e + e⁻²)` by
    /// (A7). The flux `j₀ = (0, 0, 0, 1, 1)`. A second case, `τ = 1/2`, `σ = 2`, puts `σ²` and `τ`
    /// where (A4) and (A6) put them, and an EIF with `ψ = e^{V+1}` puts `ψ` into `G`.
    #[test]
    fn appendix_a_worked_by_hand_on_four_cells() {
        let base = WhiteNoiseIf { model: Model::Lif, tau: 1.0, e0: -1.0, sigma: 1.0, v_th: 0.0, v_re: -2.0, v_lb: -4.0, t_ref: 0.0 };
        let st = base.stationary(1.0).unwrap();
        assert_eq!(st.lattice(), Lattice { n: 4, k_re: 2, dv: 1.0 });
        assert_eq!(st.voltage(), &[-4.0, -3.0, -2.0, -1.0, 0.0]);
        let r0 = 1.0 / (2.0 * E + (-2.0f64).exp());
        assert!((st.rate() / r0 - 1.0).abs() < 1e-15, "{} against {r0}", st.rate());
        let want = [(-2.0f64).exp(), 1.0, E, E - 1.0, 0.0];
        for (k, w) in want.iter().enumerate() {
            assert!((st.density()[k] - r0 * w).abs() < 1e-15, "k = {k}: {} against {}", st.density()[k], r0 * w);
        }
        assert_eq!(st.flux(), &[0.0, 0.0, 0.0, st.rate(), st.rate()]);

        // τ = 1/2, σ = 2: ΔG = (V + 1)/4, B = (e^{ΔG} − 1)/(4ΔG), and p⁽ᵏ⁻¹⁾ = p A + τ j B.
        let st = WhiteNoiseIf { tau: 0.5, sigma: 2.0, ..base }.stationary(1.0).unwrap();
        let q = 0.25f64;
        let p3 = 0.5 * q.exp_m1();
        let p2 = p3 + 0.5 * 0.25;
        let p1 = p2 * (-q).exp();
        let p0 = p1 * (-2.0 * q).exp();
        let r = 1.0 / (p0 + p1 + p2 + p3);
        for (k, w) in [p0, p1, p2, p3, 0.0].iter().enumerate() {
            assert!((st.density()[k] - r * w).abs() < 1e-15, "k = {k}");
        }

        // EIF with V_T = −1, Δ_T = 1: ψ(V) = e^{V+1}, so ΔG⁽ᵏ⁾ = V + 1 − e^{V+1}.
        let eif = WhiteNoiseIf { model: Model::Eif { v_t: -1.0, delta_t: 1.0 }, ..base };
        assert!((eif.psi(0.0) - E).abs() < 1e-15 && (eif.psi(-3.0) - (-2.0f64).exp()).abs() < 1e-16);
        let st = eif.stationary(1.0).unwrap();
        let g = |v: f64| v + 1.0 - (v + 1.0).exp();
        let b = |x: f64| x.exp_m1() / x;
        let p3 = b(g(0.0));
        let p2 = p3 * g(-1.0).exp() + b(g(-1.0));
        let p1 = p2 * g(-2.0).exp();
        let p0 = p1 * g(-3.0).exp();
        let r = 1.0 / (p0 + p1 + p2 + p3);
        for (k, w) in [p0, p1, p2, p3, 0.0].iter().enumerate() {
            assert!((st.density()[k] - r * w).abs() < 1e-15, "EIF k = {k}: {} against {}", st.density()[k], r * w);
        }
    }

    /// The flux is `r₀` above the reset and zero at and below it with no rounding at all; the density
    /// vanishes at threshold and integrates to one — to `1 − r τ_ref` with a refractory period,
    /// eq. (91).
    ///
    /// `j₀` starts at exactly 1 and has exactly 1 subtracted from it once, so it is exactly 1 or 0 and
    /// `J₀ = r j₀` is exactly `r` or 0. The mass is `r Σ Δp`, which is `r/r₀` up to the rounding of
    /// the sum: measured within `9.3 × 10⁻¹⁵` of `1 − r τ_ref` over the four figure cases and a
    /// refractory one.
    #[test]
    fn the_flux_is_exact_and_the_density_is_normalised() {
        let cases = [
            WhiteNoiseIf::FIG1_CASE_I,
            WhiteNoiseIf::FIG1_CASE_II,
            WhiteNoiseIf::FIG2_CASE_I,
            WhiteNoiseIf::FIG2_CASE_II,
            WhiteNoiseIf { t_ref: 5e-3, ..WhiteNoiseIf::FIG2_CASE_I },
        ];
        let mut worst = 0.0f64;
        for c in cases {
            let st = c.stationary(WhiteNoiseIf::FIG_STEP).unwrap();
            let Lattice { n, k_re, .. } = st.lattice();
            for (k, &j) in st.flux().iter().enumerate() {
                assert_eq!(j, if k > k_re { st.rate() } else { 0.0 }, "k = {k}");
            }
            assert_eq!(st.density()[n], 0.0);
            assert!(st.density().iter().all(|&p| p >= 0.0));
            worst = worst.max((st.mass() - (1.0 - st.rate() * c.t_ref)).abs());
        }
        assert!(worst < 5e-14, "{worst}");
        // The dead time changes the rate and not the shape: r = r₀/(1 + τ_ref r₀) with r₀ the rate of
        // the same neuron without it, and the non-refractory mass is 1/(1 + τ_ref r₀), 0.82 here.
        let free = cases[2].stationary(WhiteNoiseIf::FIG_STEP).unwrap().rate();
        let st = cases[4].stationary(WhiteNoiseIf::FIG_STEP).unwrap();
        assert!((st.rate() - free / (1.0 + 5e-3 * free)).abs() < 1e-12 * free, "{} {free}", st.rate());
        assert!((st.mass() - 1.0 / (1.0 + 5e-3 * free)).abs() < 5e-14, "{}", st.mass());
    }

    /// The lattice rate against [`SiegertInput::rate`] — the crate's Siegert formula, Brunel's
    /// eq. (21) — over 56 neurons: `E₀` from −70 to −30 mV about a −50 mV threshold, `σ` from 1 to
    /// 8 mV, with and without a 2 ms refractory period, on a −150 mV floor (Richardson's −100 mV is
    /// only 3.75σ below the mean at `E₀ = −70`, `σ = 8` mV, and the lattice then converges to the
    /// floored rate rather than to Siegert's).
    ///
    /// Measured: every one of the 56 errors halves when the step does — the order, `log₂` of the
    /// error ratio, lies in `[0.96, 1.03]` between 10 and 5 µV and in `[0.98, 1.015]` between 5 and
    /// 2.5 µV. First order, as the scheme's `O(Δ²)` per step says (Richardson, after eq. (A3)).
    ///
    /// The size of the error has a reading. Far below threshold the Siegert rate goes as
    /// `exp(−(V_th − E₀)²/2σ²)`, so shifting the threshold by half a cell changes it by the factor
    /// `½Δ(V_th − E₀)/σ²`; and for every neuron here at least 10σ below threshold the measured
    /// error is that, to within 6% (ratios 0.950 to 0.978). The worst case of the grid is the
    /// farthest: `E₀ = −70` mV, `σ = 1` mV, 20σ below, firing at `5.5 × 10⁻⁸⁵` Hz, where the 10 µV
    /// step is 9.5% off. Among the 44 neurons less than 5σ below threshold the worst is
    /// `4.9 × 10⁻³`.
    #[test]
    fn the_lattice_rate_converges_to_the_siegert_rate_at_first_order() {
        let (mut orders, mut near, mut far, mut count) = (Vec::new(), 0.0f64, Vec::new(), 0);
        // Millivolts as integers, so that "10σ below threshold" is decided exactly: in volts,
        // (−0.05 + 0.06)/0.001 is 9.999999999999995.
        for e0_mv in [-70i32, -60, -55, -50, -45, -40, -30] {
            for sigma_mv in [1i32, 2, 4, 8] {
                for t_ref in [0.0, 2e-3] {
                    let (e0, sigma) = (f64::from(e0_mv) * 1e-3, f64::from(sigma_mv) * 1e-3);
                    let n = WhiteNoiseIf { e0, sigma, t_ref, v_lb: -150e-3, ..WhiteNoiseIf::FIG1_CASE_I };
                    let exact = n.siegert().unwrap().rate().unwrap();
                    let err: Vec<f64> =
                        [10e-6, 5e-6, 2.5e-6].iter().map(|&dv| (n.stationary(dv).unwrap().rate() / exact - 1.0).abs()).collect();
                    orders.push(((err[0] / err[1]).log2(), (err[1] / err[2]).log2()));
                    let distance = f64::from(-50 - e0_mv) / f64::from(sigma_mv);
                    if distance < 5.0 {
                        near = near.max(err[0]);
                    }
                    if distance >= 10.0 {
                        far.push(err[0] / (0.5 * 10e-6 * (n.v_th - e0) / (sigma * sigma)));
                    }
                    count += 1;
                }
            }
        }
        assert_eq!(count, 56);
        let span = |f: fn(&(f64, f64)) -> f64| orders.iter().map(f).fold((f64::INFINITY, 0.0f64), |(a, b), x| (a.min(x), b.max(x)));
        let (first, second) = (span(|o| o.0), span(|o| o.1));
        assert!(first.0 > 0.96 && first.1 < 1.03, "{first:?}");
        assert!(second.0 > 0.98 && second.1 < 1.015, "{second:?}");
        assert!(near < 1e-2, "{near}");
        assert_eq!(far.len(), 6);
        assert!(far.iter().all(|&x| x > 0.94 && x <= 1.0), "{far:?}");
    }

    /// [`SiegertInput::rate`] against an independent `SciPy` quadrature of Brunel's eq. (21),
    /// `tools/density_reference.py` section 1 (`quad` of `erfcx(−u)` at relative tolerance `10⁻¹³`),
    /// through [`WhiteNoiseIf::siegert`], which is where Richardson's `σ` becomes Brunel's `√2σ`.
    /// Measured agreement `2 × 10⁻¹⁵`.
    ///
    /// The `√2` in the wrong place is barely visible in a mean-driven neuron — it moves the Fig. 1
    /// case i rate by 0.7% — and it is ruinous in a noise-driven one: 35% at `E₀ = −52` mV,
    /// `σ = 2` mV, 82% at Fig. 1's case ii, and 99.9% at `E₀ = −65` mV, `σ = 4` mV.
    #[test]
    fn the_siegert_rate_is_an_independent_quadrature() {
        let table = [
            (-45e-3, 1e-3, 0.0, 46.21557620398916),
            (-60e-3, 5e-3, 0.0, 4.794595042424083),
            (-52e-3, 2e-3, 0.0, 12.066593163147884),
            (-40e-3, 3e-3, 2e-3, 65.4588799894017),
            (-65e-3, 4e-3, 5e-3, 0.060584373848869624),
        ];
        let mut worst = 0.0f64;
        for (e0, sigma, t_ref, want) in table {
            let n = WhiteNoiseIf { e0, sigma, t_ref, ..WhiteNoiseIf::FIG1_CASE_I };
            let got = n.siegert().unwrap().rate().unwrap();
            worst = worst.max((got / want - 1.0).abs());
            let wrong = SiegertInput { sigma, ..n.siegert().unwrap() }.rate().unwrap();
            if e0 < n.v_th {
                assert!((wrong / want - 1.0).abs() > 0.25, "Richardson's σ used as Brunel's: {wrong} against {want}");
            }
        }
        assert!(worst < 1e-14, "{worst}");
    }

    /// The four steady-state rates Richardson's captions print, at his 10 µV step: Fig. 1, 46 Hz
    /// and 4.8 Hz; Fig. 2, 44 Hz and 5.6 Hz. Each rounds to the printed digits.
    ///
    /// The EIF has no Siegert formula, so its rate is also checked against an integration that
    /// shares nothing with threshold integration: `tools/density_reference.py` section 3 solves
    /// `k′ = 1 + φ′k` forward from `V_lb` with `SciPy`'s DOP853 and Radau and integrates `k` over
    /// `[V_re, V_th]`, giving 44.046 578 074 Hz, 5.643 154 731 Hz and 27.865 315 236 Hz. Measured
    /// relative errors at 10 µV: `2.9 × 10⁻⁴`, `1.2 × 10⁻⁴`, and `2.4 × 10⁻⁴` for the third case
    /// (`E₀ = −50`, `σ = 4` mV); each halves with the step (orders 1.018, 1.003, 1.005).
    #[test]
    fn the_figure_rates_are_the_captions_and_the_eif_rate_is_an_independent_integration() {
        let r = |c: WhiteNoiseIf| c.stationary(WhiteNoiseIf::FIG_STEP).unwrap().rate();
        assert_eq!(r(WhiteNoiseIf::FIG1_CASE_I).round(), 46.0);
        assert_eq!((r(WhiteNoiseIf::FIG1_CASE_II) * 10.0).round(), 48.0);
        assert_eq!(r(WhiteNoiseIf::FIG2_CASE_I).round(), 44.0);
        assert_eq!((r(WhiteNoiseIf::FIG2_CASE_II) * 10.0).round(), 56.0);
        let third = WhiteNoiseIf { e0: -50e-3, sigma: 4e-3, ..WhiteNoiseIf::FIG2_CASE_I };
        for (c, want) in [
            (WhiteNoiseIf::FIG2_CASE_I, 44.04657807442763),
            (WhiteNoiseIf::FIG2_CASE_II, 5.643154731400086),
            (third, 27.865315236139086),
        ] {
            let coarse = c.stationary(10e-6).unwrap().rate() / want - 1.0;
            let fine = c.stationary(5e-6).unwrap().rate() / want - 1.0;
            assert!(coarse.abs() < 6e-4, "{coarse}");
            assert!(((coarse / fine).log2() - 1.0).abs() < 0.05, "{coarse} {fine}");
        }
    }

    /// Brunel's parameter set for the closed-form checks: `μ = 15` mV, `σ_B = 5` mV below his
    /// 20 mV threshold, 10 mV reset, 20 ms, 2 ms dead time — [`crate::meanfield`]'s own example,
    /// firing at 9.46 Hz.
    fn brunel_input() -> SiegertInput {
        SiegertInput::new(20e-3, 2e-3, 20e-3, 10e-3, 15e-3, 5e-3).unwrap()
    }

    /// Composite Simpson on `[a, b]` with `n` (even) panels.
    fn simpson(f: impl Fn(f64) -> f64, a: f64, b: f64, n: usize) -> f64 {
        let h = (b - a) / n as f64;
        let inner: f64 = (1..n).map(|i| f(a + i as f64 * h) * if i % 2 == 1 { 4.0 } else { 2.0 }).sum();
        (f(a) + f(b) + inner) * h / 3.0
    }

    /// [`brunel_density`] and [`brunel_slope`] against an independent `SciPy` quadrature of eq. (19)
    /// and its derivative (`tools/density_reference.py` section 2, `quad` of `e^{u² − y²}` at relative
    /// tolerance `10⁻¹³`), at six voltages from 25 mV below the mean to a quarter millivolt under
    /// threshold. Measured agreement `2.2 × 10⁻¹⁶` relative for both.
    ///
    /// And the closed form satisfies the paper's own conditions: it integrates to `1 − ν₀τ_rp`, the
    /// normalisation (12), to `1.3 × 10⁻¹⁰` by Simpson's rule with a thousand panels on each side of
    /// the reset (the rule's own `h⁴` error); its slope at
    /// threshold is `−2ν₀τ/σ²`, eq. (9), to the last bit; the slope jumps by `−2ν₀τ/σ²` across the
    /// reset, eq. (10), to `3 × 10⁻¹⁴`, and the density is continuous there to `2 × 10⁻¹³`; it
    /// vanishes at and above threshold.
    #[test]
    fn brunels_closed_form_is_an_independent_quadrature_and_satisfies_his_boundary_conditions() {
        let input = brunel_input();
        let table = [
            (-5.0e-3, 2.4915927519264787e-05, 0.03986548403082366),
            (5.0e-3, 4.055186586408968, 3244.1492691271733),
            (10.0e-3, 81.45059991173257, 32580.239964693017),
            (12.5e-3, 118.33952191986937, 8530.624694759266),
            (17.5e-3, 54.091399446380734, -25955.55957849075),
            (19.75e-3, 3.973294791338557, -16647.131709923244),
        ];
        let mut worst = 0.0f64;
        for (v, p, d) in table {
            worst = worst.max((brunel_density(&input, v).unwrap() / p - 1.0).abs());
            worst = worst.max((brunel_slope(&input, v).unwrap() / d - 1.0).abs());
        }
        assert!(worst < 1e-15, "{worst}");

        let nu0 = input.rate().unwrap();
        assert!((nu0 - 9.460799805759121).abs() < 1e-12);
        let p = |v: f64| brunel_density(&input, v).unwrap();
        let mass = simpson(p, -45e-3, 10e-3, 1000) + simpson(p, 10e-3, 20e-3, 1000);
        assert!((mass - (1.0 - nu0 * 2e-3)).abs() < 5e-10, "{mass} against {}", 1.0 - nu0 * 2e-3);
        let edge = -2.0 * nu0 * input.tau_m / (input.sigma * input.sigma);
        assert!((brunel_slope(&input, 20e-3).unwrap() / edge - 1.0).abs() < 1e-14);
        let above = brunel_slope(&input, 10e-3 + 1e-15).unwrap();
        let below = brunel_slope(&input, 10e-3).unwrap();
        assert!(((above - below) / edge - 1.0).abs() < 1e-13, "jump {} against {edge}", above - below);
        assert_eq!(brunel_density(&input, 20e-3).unwrap(), 0.0);
        assert_eq!(brunel_density(&input, 25e-3).unwrap(), 0.0);
        assert_eq!(brunel_slope(&input, 25e-3).unwrap(), 0.0);
        assert!((brunel_density(&input, 10e-3 + 1e-15).unwrap() / p(10e-3) - 1.0).abs() < 1e-12, "continuous at the reset");
    }

    /// The lattice density is Brunel's closed form at every lattice point, converging at first order.
    ///
    /// Brunel's neuron above in Richardson's convention, `σ = 5/√2` mV, on a −40 mV floor (`11σ_B`
    /// below the mean). Measured worst `|P_lattice − P_Brunel|`, as a fraction of the peak: `4.75 ×
    /// 10⁻⁴` at 10 µV, `2.37 × 10⁻⁴` at 5 µV, `1.19 × 10⁻⁴` at 2.5 µV. The two densities are two
    /// different objects — one a recursion on a lattice closed at `V_lb`, the other an integral on
    /// `(−∞, θ]` — and they differ by the scheme's first-order error and nothing else.
    #[test]
    fn the_lattice_density_is_brunels_closed_form() {
        let input = brunel_input();
        let n = WhiteNoiseIf::from_siegert(&input, -40e-3);
        assert!((n.sigma - 5e-3 / SQRT_2).abs() < 1e-18 && n.e0 == 15e-3 && n.t_ref == 2e-3);
        let mut errors = Vec::new();
        for dv in [10e-6, 5e-6, 2.5e-6] {
            let st = n.stationary(dv).unwrap();
            let peak = st.density().iter().copied().fold(0.0, f64::max);
            let worst = st
                .voltage()
                .iter()
                .zip(st.density())
                .map(|(&v, &p)| (p - brunel_density(&input, v).unwrap()).abs())
                .fold(0.0, f64::max);
            errors.push(worst / peak);
        }
        assert!(errors[0] < 5e-4 && errors[0] > 4e-4, "{errors:?}");
        for w in errors.windows(2) {
            assert!(((w[0] / w[1]).log2() - 1.0).abs() < 0.01, "{errors:?}");
        }
    }

    /// ⚠ The Heaviside argument of eq. (19) as printed, `Θ(u − V_r)`, does not normalise; read as
    /// `(V_r − μ)/σ` it does.
    ///
    /// Each reading is integrated over `V` by Simpson's rule, for Brunel's neuron above, whose
    /// normalisation (12) requires `1 − ν₀τ_rp = 0.981`. `V_r = 10` read in millivolts puts the step
    /// ten noise units above the mean, past the threshold at `y_θ = 1`: the density vanishes and the
    /// mass is 0. `V_r = 0.01` read in volts puts the reset at `μ + 0.01σ`: measured mass 0.76. Only
    /// `(V_r − μ)/σ = −1`, the lower limit of his eq. (21) and the reset of his own eq. (36), gives
    /// 0.981.
    ///
    /// The millivolt reading empties the density wherever `y_θ ≤ 10`, which is so at Table 1's three
    /// points: at the `SciPy` rates of eqs. (20)–(21), `y_θ` is −0.380, −0.133 and 1.165 at B, C and
    /// D. It is not so everywhere in the paper: near `ν = 0` on Fig. 1's network (`C_E = 4000`,
    /// `J = 0.2` mV) `y_θ` is `10(1 − r)/√r` at `ν_ext = rν_thr`, past 10 for `r < 0.38` — 15 at
    /// `r = 1/4`, which the last line checks.
    #[test]
    fn the_printed_heaviside_argument_does_not_normalise() {
        let input = brunel_input();
        let nu0 = input.rate().unwrap();
        let (mu, sigma, tau) = (input.mu, input.sigma, input.tau_m);
        let yth = (input.v_th - mu) / sigma;
        let mass = |step_at: f64| {
            let p = |v: f64| {
                let y = (v - mu) / sigma;
                let lo = y.max(step_at).min(yth);
                2.0 * nu0 * tau / sigma * super::gauss_weighted(y, lo, yth).unwrap()
            };
            simpson(p, -45e-3, 10e-3, 1000) + simpson(p, 10e-3, 20e-3, 1000)
        };
        let (mv, volts, reduced) = (mass(10.0), mass(0.01), mass((input.v_reset - mu) / sigma));
        let want = 1.0 - nu0 * input.t_ref;
        assert_eq!(mv, 0.0);
        assert!((volts - 0.76).abs() < 0.01, "{volts}");
        assert!((reduced - want).abs() < 5e-10, "{reduced} against {want}");
        for ((g, ratio, _, rate), want) in TABLE_1.iter().zip([-0.380, -0.133, 1.165]) {
            let cell = NetworkDensity { net: brunel_net(1000.0, 0.1e-3, *g, *ratio), v_lb: -40e-3, dv: 10e-6 }.neuron(*rate).unwrap();
            let y_th = (cell.v_th - cell.e0) / (SQRT_2 * cell.sigma);
            assert!((y_th - want).abs() < 1e-3, "g = {g}: {y_th}");
        }
        let quiet = NetworkDensity { net: brunel_net(4000.0, 0.2e-3, 5.0, 0.25), v_lb: -40e-3, dv: 10e-6 }.neuron(0.0).unwrap();
        assert!(((quiet.v_th - quiet.e0) / (SQRT_2 * quiet.sigma) - 15.0).abs() < 1e-9, "{quiet:?}");
    }

    /// On page 188 the Siegert formula is eq. (21), and eq. (22) is its low-rate limit,
    /// `ν₀τ ≃ (θ − μ)/(σ√π) exp(−(θ − μ)²/σ²)`.
    ///
    /// [`SiegertInput::rate`] (eq. 21) against eq. (22) as `y_θ = (θ − μ)/σ` grows, with no dead
    /// time: the ratio approaches one from below as `1 − 1/(2y_θ²)`, the next term of the asymptotic
    /// series of `∫^{y} e^{u²} du`, and what is left falls as `y_θ⁻⁴`: measured `|ratio − (1 −
    /// 1/2y²)|` = `2.0 × 10⁻²`, `2.4 × 10⁻³`, `1.3 × 10⁻⁴`, `7.7 × 10⁻⁶` at `y_θ` = 2, 4, 8, 16.
    #[test]
    fn eq_22_is_the_low_rate_limit_of_eq_21() {
        let mut last = f64::INFINITY;
        for y in [2.0, 4.0, 8.0, 16.0] {
            let i = SiegertInput::new(20e-3, 0.0, 20e-3, 10e-3, 20e-3 - y * 5e-3, 5e-3).unwrap();
            let eq21 = i.rate().unwrap() * i.tau_m;
            let eq22 = y / PI.sqrt() * (-y * y).exp();
            let off = (eq21 / eq22 - (1.0 - 1.0 / (2.0 * y * y))).abs();
            assert!(eq21 < eq22);
            assert!(off < last && off * y.powi(4) < 0.65, "y = {y}: {off}");
            last = off;
        }
    }

    /// Every response's zero-frequency limit is the derivative of the stationary rate.
    ///
    /// At `λ = 10⁻⁴ i` per second. For the LIF (Fig. 1's two cases and Brunel's refractory neuron,
    /// at 1 µV) the derivative is the Siegert rate's, by central difference through
    /// [`WhiteNoiseIf::siegert`] — and that difference is itself checked against `SciPy`'s,
    /// `tools/density_reference.py` section 4, to `1.3 × 10⁻¹¹` in the mean, `1.3 × 10⁻⁹` in the
    /// variance and `2.7 × 10⁻⁹` in the conductance (section 4b: `τ/(1 + x)`, `σ²/(1 + x)`, `E`
    /// unchanged, for the leak `g₀(1 + x)`). The LIF's conductance response is within `1.2 × 10⁻⁴`,
    /// `6.9 × 10⁻⁴` and `1.24 × 10⁻³` of it at 2 µV and halves at 1 µV (ratios 1.9998 to 2.0000).
    /// For the EIF (Fig. 2's two cases, at 10 and 5 µV), which has no Siegert formula, it is the
    /// central difference of the lattice's own rate, for all five modulations. Measured: LIF mean
    /// and variance within `2.2 × 10⁻⁴` at 1 µV, EIF within `5.0 × 10⁻³` at 5 µV (the spike width
    /// in case ii; every other within `1.5 × 10⁻³`). Eight of the ten EIF errors halve with the
    /// step (ratios 1.999 to 2.097); the two variance errors fall faster, by 3.4 in case i and by
    /// 2.5 in case ii, where it is already `1.3 × 10⁻⁷`.
    ///
    /// The conductance derivative is taken the way Richardson's `F_g` modulates the model: his (33)
    /// scales the leak `(E − V)/τ` and nothing else, and (50) carries it unchanged into the EIF. So
    /// the EIF with leak conductance `g₀(1 + x)` is the EIF with `τ/(1 + x)`, `σ²/(1 + x)` — the
    /// noise current is held — and `V_T + Δ_T ln(1 + x)`, which is where `ψ/τ` goes. Moving `τ` and
    /// `σ²` without `V_T` — for the LIF, which has no `ψ`, the whole of the modulation — disagrees
    /// with the EIF's response by 38% in case i and a factor of 33 in case ii; that is a different
    /// modulation, not an error in this one.
    #[test]
    fn every_response_at_zero_frequency_is_the_derivative_of_the_rate() {
        let slow = Complex::new(0.0, 1e-4);
        let siegert = |n: WhiteNoiseIf| n.siegert().unwrap().rate().unwrap();
        let lif = [
            (WhiteNoiseIf::FIG1_CASE_I, 5400.950132887772, 674361.5301729733, 45.54121467847949),
            (WhiteNoiseIf::FIG1_CASE_II, 1549.118779564651, 332872.8448437346, -3.5272261120766757),
            (WhiteNoiseIf::from_siegert(&brunel_input(), -40e-3), 2907.29106300347, 689993.6693116615, 0.6568654669081297),
        ];
        for (c, scipy_e, scipy_var, scipy_g) in lif {
            let h = 1e-7;
            let de = (siegert(WhiteNoiseIf { e0: c.e0 + h, ..c }) - siegert(WhiteNoiseIf { e0: c.e0 - h, ..c })) / (2.0 * h);
            let (s2, hv) = (c.sigma * c.sigma, 1e-12);
            let dvar = (siegert(WhiteNoiseIf { sigma: (s2 + hv).sqrt(), ..c })
                - siegert(WhiteNoiseIf { sigma: (s2 - hv).sqrt(), ..c }))
                / (2.0 * hv);
            let leak = |x: f64| WhiteNoiseIf { tau: c.tau / (1.0 + x), sigma: (s2 / (1.0 + x)).sqrt(), ..c };
            let dg = (siegert(leak(1e-6)) - siegert(leak(-1e-6))) / 2e-6;
            assert!((de / scipy_e - 1.0).abs() < 1e-10 && (dvar / scipy_var - 1.0).abs() < 5e-9, "{de} {dvar}");
            assert!((dg / scipy_g - 1.0).abs() < 1e-8, "{dg}");
            let st = c.stationary(1e-6).unwrap();
            let e = rel(st.response(Modulation::Mean, slow).unwrap(), Complex::new(de, 0.0));
            let v = rel(st.response(Modulation::Variance, slow).unwrap(), Complex::new(dvar, 0.0));
            assert!(e < 5e-4 && v < 5e-4, "E0 = {}: {e} {v}", c.e0);
            let g = |dv: f64| rel(c.stationary(dv).unwrap().response(Modulation::Conductance, slow).unwrap(), Complex::new(dg, 0.0));
            let (coarse, fine) = (g(2e-6), g(1e-6));
            assert!(fine < 1.5e-3 && (coarse / fine - 2.0).abs() < 0.05, "E0 = {}: {coarse} {fine}", c.e0);
        }
        let mut worst = 0.0f64;
        for c in [WhiteNoiseIf::FIG2_CASE_I, WhiteNoiseIf::FIG2_CASE_II] {
            let Model::Eif { v_t, delta_t } = c.model else { unreachable!() };
            let eif = |v_t: f64, delta_t: f64| Model::Eif { v_t, delta_t };
            let s2 = c.sigma * c.sigma;
            let leak = |x: f64| WhiteNoiseIf {
                tau: c.tau / (1.0 + x),
                sigma: (s2 / (1.0 + x)).sqrt(),
                model: eif(v_t + delta_t * (1.0 + x).ln(), delta_t),
                ..c
            };
            let whole_tau = |x: f64| WhiteNoiseIf { tau: c.tau / (1.0 + x), sigma: (s2 / (1.0 + x)).sqrt(), ..c };
            let mut errors = [[0.0; 5]; 2];
            for (row, dv) in [10e-6, 5e-6].into_iter().enumerate() {
                let rate = |n: WhiteNoiseIf| n.stationary(dv).unwrap().rate();
                let diff = |up: WhiteNoiseIf, down: WhiteNoiseIf, h: f64| (rate(up) - rate(down)) / (2.0 * h);
                let (h, hv, x) = (1e-6, 1e-10, 1e-5);
                let derivatives = [
                    (Modulation::Mean, diff(WhiteNoiseIf { e0: c.e0 + h, ..c }, WhiteNoiseIf { e0: c.e0 - h, ..c }, h)),
                    (
                        Modulation::Variance,
                        diff(WhiteNoiseIf { sigma: (s2 + hv).sqrt(), ..c }, WhiteNoiseIf { sigma: (s2 - hv).sqrt(), ..c }, hv),
                    ),
                    (Modulation::Conductance, diff(leak(x), leak(-x), x)),
                    (
                        Modulation::SpikeThreshold,
                        diff(WhiteNoiseIf { model: eif(v_t + h, delta_t), ..c }, WhiteNoiseIf { model: eif(v_t - h, delta_t), ..c }, h),
                    ),
                    (
                        Modulation::SpikeWidth,
                        diff(WhiteNoiseIf { model: eif(v_t, delta_t + h), ..c }, WhiteNoiseIf { model: eif(v_t, delta_t - h), ..c }, h),
                    ),
                ];
                let st = c.stationary(dv).unwrap();
                for (i, (m, d)) in derivatives.into_iter().enumerate() {
                    errors[row][i] = rel(st.response(m, slow).unwrap(), Complex::new(d, 0.0));
                }
                if row == 0 {
                    let tau_only = diff(whole_tau(x), whole_tau(-x), x);
                    let g = st.response(Modulation::Conductance, slow).unwrap().re;
                    assert!((g / tau_only - 1.0).abs() > 0.3, "{g} {tau_only}");
                }
            }
            for i in 0..5 {
                worst = worst.max(errors[1][i]);
                let halving = errors[0][i] / errors[1][i];
                if i == 1 {
                    assert!(halving > 2.4 && errors[1][i] < 7e-4, "E0 = {}: {errors:?}", c.e0);
                } else {
                    assert!(halving > 1.95 && halving < 2.15, "E0 = {}, modulation {i}: {errors:?}", c.e0);
                }
            }
        }
        assert!(worst < 1e-2, "{worst}");
    }

    /// The LIF response is the closed form, at every frequency, with and without a refractory period.
    ///
    /// The references are `tools/density_reference.py` section 4. The mean and variance responses
    /// come from Brunel's own Appendix A.3: his eigenvalue equation (46), with `φ₂` of eq. (44) and
    /// the particular solution (42), evaluated with `mpmath`'s confluent hypergeometric `M` at up to
    /// 598 digits (below the mean `φ₂` is a difference of two terms `e^{y²}` times larger than it,
    /// and at 10 kHz the Gamma prefactors of (44) cost hundreds of digits more), read as the loop
    /// gain of a network with only a mean coupling (`G = 1, H = 0`) or only a variance coupling
    /// (`G = 0, H = 1`). The mean response is also Klett and Lindner's eq. (10)
    /// (arXiv:2503.07434, the parabolic-cylinder form they attribute to Lindner and
    /// Schimansky-Geier 2001), which the script checks against Brunel's to `10⁻⁹` at every
    /// frequency below before printing either. Klett and Lindner transform with `e^{+iωt}`; Richardson
    /// modulates with `e^{+iωt}`; the printed values are therefore their conjugate.
    ///
    /// Measured, at 1 µV, over 1 Hz to 10 kHz and the three neurons: the worst relative
    /// disagreement is `6.2 × 10⁻³`, at Fig. 1 case i's resonance at 46 Hz, and every one of the 36
    /// errors halves with the step — the order lies in `[0.990, 1.003]` between 2 and 1 µV.
    #[test]
    fn the_lif_response_is_the_closed_form_at_every_frequency() {
        let freqs = [1.0, 10.0, 46.0, 100.0, 1000.0, 10000.0];
        // Per frequency: Re r̂_E, Im r̂_E, Re r̂_σ², Im r̂_σ².
        type Row = (f64, f64, f64, f64);
        let cases: [(WhiteNoiseIf, [Row; 6]); 3] = [
            (
                WhiteNoiseIf::FIG1_CASE_I,
                [
                    (5401.207570209509, 49.538402354899986, 673017.5764097755, 95896.44579442713),
                    (5428.271229897468, 513.9827091090958, 534769.3997646779, 970873.4937236013),
                    (16102.41705057402, 1760.7279415628818, 525091.0128776801, 19065837.33938441),
                    (7946.140423634715, -2216.4099641661433, 9461818.03924554, 14478589.964963853),
                    (2863.683819159527, -2040.4973469729873, 31806098.27063443, 10822638.869706627),
                    (920.4038918550146, -831.3035178091997, 41610137.86487611, 4226404.351011047),
                ],
            ),
            (
                WhiteNoiseIf::FIG1_CASE_II,
                [
                    (1539.3121008871622, -109.57687701809544, 333786.93450180674, 5378.202341666322),
                    (1019.7976280316111, -617.2954709045075, 372810.60393616674, 4759.81347069474),
                    (346.31990389270396, -397.38326134764947, 325550.62429035373, -68477.8448341423),
                    (209.40254775770163, -254.7350335443918, 277901.9252932873, -64209.454063887475),
                    (60.92087698678947, -67.75319604122727, 216316.10939251515, -23852.41644274057),
                    (19.139832363680217, -19.879324196091524, 199445.56692268714, -7640.394597762752),
                ],
            ),
            (
                WhiteNoiseIf::from_siegert(&brunel_input(), -40e-3),
                [
                    (2900.723241466958, -121.35967921463485, 692954.2413681526, 38240.521314730206),
                    (2413.7494314023406, -936.8362469131649, 897933.7916096378, 243386.17103765198),
                    (971.4852574088097, -953.5553238160516, 1109168.6517905765, -61939.87246832339),
                    (586.4056969859387, -636.9498164886263, 988911.1403820085, -116570.47383546055),
                    (170.2739346349133, -182.49408309616055, 825354.0165733895, -60403.71482005186),
                    (53.42078480465637, -54.84006705878154, 778247.8102566492, -20714.411327915746),
                ],
            ),
        ];
        let (mut worst, mut lo, mut hi) = (0.0f64, f64::INFINITY, 0.0f64);
        for (c, table) in cases {
            let (coarse, fine) = (c.stationary(2e-6).unwrap(), c.stationary(1e-6).unwrap());
            for (f, (e_re, e_im, v_re, v_im)) in freqs.iter().zip(table) {
                for (m, want) in [(Modulation::Mean, Complex::new(e_re, e_im)), (Modulation::Variance, Complex::new(v_re, v_im))] {
                    let a = rel(coarse.response(m, at(*f)).unwrap(), want);
                    let b = rel(fine.response(m, at(*f)).unwrap(), want);
                    worst = worst.max(b);
                    lo = lo.min((a / b).log2());
                    hi = hi.max((a / b).log2());
                }
            }
        }
        assert!(worst < 1.3e-2, "{worst}");
        assert!(lo > 0.985 && hi < 1.005, "[{lo}, {hi}]");
    }

    /// The EIF's five responses are an independent integration of the same equations, at every
    /// frequency.
    ///
    /// The EIF has no closed form. The reference, `tools/density_reference.py` section 7, integrates
    /// Richardson's first-order system (17)–(24), `∂P̂/∂V = [(E − V + ψ)P̂ − τ(Ĵ + F_α)]/σ²` and
    /// `∂Ĵ/∂V = −λP̂`, backward from threshold as he does — but as ODEs, by `SciPy`'s Radau at
    /// relative tolerance `10⁻¹⁰`, with `∂P₀/∂V` carried as a state of its own, where this module
    /// steps the exponential lattice of (A3) and differences the density. It shares the equations
    /// and nothing else. On the LIF the same route reproduces Brunel's eq. (46) to `5 × 10⁻¹³`
    /// (Fig. 1's cases at 46 Hz and 1 kHz, Brunel's refractory neuron at 46 Hz); its EIF rates are
    /// section 3's to `5 × 10⁻¹⁴`; and the whole table moves by `2 × 10⁻¹⁰` at relative tolerance
    /// `10⁻⁸`.
    ///
    /// Measured, over Fig. 2's two cases, the five modulations and 1 Hz to 10 kHz, at 2.5 µV: the
    /// worst relative disagreement is `9.0 × 10⁻³`, case i's conductance response at 100 Hz, near
    /// the resonance, and up to 100 Hz every error halves with the step — the order lies in
    /// `[0.987, 1.081]` between 5 and 2.5 µV. From 1 kHz the errors are smaller, at most
    /// `1.15 × 10⁻³`, and not yet in their asymptotic regime: orders from 0.75 to 1.25.
    #[test]
    fn the_eif_response_is_an_independent_integration_at_every_frequency() {
        let freqs = [1.0, 10.0, 44.0, 100.0, 1000.0, 10000.0];
        let modulations =
            [Modulation::Mean, Modulation::Variance, Modulation::Conductance, Modulation::SpikeThreshold, Modulation::SpikeWidth];
        // Per case, per modulation in that order, per frequency: (Re r̂, Im r̂).
        type Row = [(f64, f64); 6];
        let cases: [(WhiteNoiseIf, [Row; 5]); 2] = [
            (
                WhiteNoiseIf::FIG2_CASE_I,
                [
                    [
                        (3172.9215172268887, -29.34478888621063), (3219.2453097113, -303.3684940988804),
                        (1394.4498920938154, -5415.766760417835), (61.79872958980582, -1312.4991001274518),
                        (-0.2806596077451579, -118.30457725698697), (-0.02393868258859302, -11.69828714594252),
                    ],
                    [
                        (-63536.141445004345, 21532.841426204974), (-25491.002641765266, 217463.04749934486),
                        (1886120.0447331264, -562794.9037451932), (230885.3350579345, -536442.0280345931),
                        (46.72610701497186, -40428.0709760322), (-21.102016663597915, -3909.1297357534477),
                    ],
                    [
                        (27.28529755487958, -0.6654836531337904), (27.2620732172182, -6.877545894668672),
                        (-44.403774148325596, -45.98864152069608), (-8.48122175590437, -2.328715387934225),
                        (-0.552541746768259, 0.6225498435818638), (-0.05476925018971069, 0.14215457244667032),
                    ],
                    [
                        (-5671.808361776076, -193.1174291434149), (-5628.822955925559, -2002.5645682238592),
                        (-26968.624014606652, -16079.94037855913), (-17201.419496700328, -1491.4945000241805),
                        (-14866.31097225634, 153.61251989247393), (-14700.477244227863, 42.17268450136965),
                    ],
                    [
                        (-2451.515036344496, -310.7707380682905), (-1919.5865475334954, -3198.0085289581903),
                        (-14039.317598066677, -54290.59927350945), (-33500.292173408176, -27935.742591980943),
                        (-65720.92739683714, -22626.784747917307), (-98746.78444225498, -22812.92260966809),
                    ],
                ],
            ),
            (
                WhiteNoiseIf::FIG2_CASE_II,
                [
                    [
                        (1479.9492605942037, -138.6380515008007), (807.6448788931957, -725.3554570376104),
                        (85.43762235651151, -362.2679442235788), (10.819947542799957, -162.1222578554578),
                        (-0.2011893982532138, -15.164837575006676), (-0.004781264391953719, -1.498777609025916),
                    ],
                    [
                        (161744.8732722763, 871.1507105752204), (183342.22026469046, -34798.489931981014),
                        (65115.57899716552, -99847.09796336392), (12896.505363615499, -56301.21823745015),
                        (-157.62187913850778, -5186.871064607467), (-4.416462223191957, -500.84971785032076),
                    ],
                    [
                        (-5.058834890756539, -0.11771960794466574), (-6.207097102099039, 0.6599298518077968),
                        (-2.901587657272134, 3.220043223272072), (-0.9788290342425932, 2.1074004877745995),
                        (-0.06757639530098468, 0.30808356462209613), (-0.006930403223726225, 0.040702593560556416),
                    ],
                    [
                        (-1626.3913947842675, -28.786060788047354), (-1749.977301323079, -197.60526191470345),
                        (-2066.860514924713, -124.81743446969404), (-2052.569857517486, 26.852210408794864),
                        (-1905.468504783344, 40.45206876540985), (-1883.4147090879467, 7.557334572666417),
                    ],
                    [
                        (-82.03258526865434, -165.64402737398507), (-593.3636134467403, -1303.7791067411592),
                        (-2674.857038756504, -2660.247015497946), (-4347.644970710636, -2811.525334330511),
                        (-8455.098074519943, -2800.4361961610934), (-12654.722205287357, -2907.6090427537047),
                    ],
                ],
            ),
        ];
        // Worst error and the span of the order, up to 100 Hz and from 1 kHz.
        let (mut low, mut high) = ((0.0f64, f64::INFINITY, 0.0f64), (0.0f64, f64::INFINITY, 0.0f64));
        for (c, rows) in cases {
            let (coarse, fine) = (c.stationary(5e-6).unwrap(), c.stationary(2.5e-6).unwrap());
            for (m, row) in modulations.iter().zip(rows) {
                for (f, (re, im)) in freqs.iter().zip(row) {
                    let want = Complex::new(re, im);
                    let a = rel(coarse.response(*m, at(*f)).unwrap(), want);
                    let b = rel(fine.response(*m, at(*f)).unwrap(), want);
                    let band = if *f < 1e3 { &mut low } else { &mut high };
                    *band = (band.0.max(b), band.1.min((a / b).log2()), band.2.max((a / b).log2()));
                }
            }
        }
        assert!(low.0 < 1.5e-2 && low.1 > 0.98 && low.2 < 1.09, "{low:?}");
        assert!(high.0 < 2.5e-3 && high.1 > 0.7 && high.2 < 1.3, "{high:?}");
    }

    /// The LIF's high-frequency limits are Richardson's eqs. (34)–(36), and the phases the Fig. 1
    /// caption prints.
    ///
    /// `r̂_E ≃ r₀/(σ√(iωτ))`, `r̂_σ² ≃ (r₀/σ²)(1 + (V_th − E₀)/(σ√(iωτ)))` and `r̂_g ≃ r₀(E₀ −
    /// V_th)/(σ√(iωτ))`, each per unit of its amplitude. What they omit is one power of `ω^{−½}`
    /// down, so the ratio of the lattice response to the asymptote closes on one as `ω^{−½}` for the
    /// mean and conductance and as `ω^{−1}` for the variance. At 0.25 µV, measured `|ratio − 1|` at
    /// 1 kHz, 10 kHz and 100 kHz: mean 0.213, 0.069, 0.022 in case i (each step `√10 = 3.16` within
    /// 4%) and conductance 0.19, 0.059, 0.019; variance 0.10, 0.0092, 0.0011; and case ii closes
    /// faster in all three. The phases: `−45°` for the mean in both cases; `−45°` for the conductance
    /// in case i, where `E₀ > V_th`, and `−225° = 135°` in case ii, where `E₀ < V_th`; and for the
    /// variance a phase that "vanishes for high frequency" (Fig. 1 Cii) as `ω^{−½}`: 18.8°, 5.8°,
    /// 1.8° at 1, 10 and 100 kHz in case i.
    #[test]
    fn the_lif_high_frequency_limits_are_eqs_34_to_36() {
        let deg = |z: Complex| z.arg().to_degrees();
        for (case, c) in [WhiteNoiseIf::FIG1_CASE_I, WhiteNoiseIf::FIG1_CASE_II].into_iter().enumerate() {
            let st = c.stationary(0.25e-6).unwrap();
            assert_eq!(st.lattice().n, 200_000);
            let (r0, s) = (st.rate(), c.sigma);
            let (mut gaps, mut phases) = ([[0.0; 3]; 3], [0.0; 3]);
            for (j, f) in [1e3, 1e4, 1e5].into_iter().enumerate() {
                let root = sqrt_iwt(f, c.tau);
                let mean = Complex::new(r0 / s, 0.0) / root;
                let var = (Complex::new(1.0, 0.0) + Complex::new((c.v_th - c.e0) / s, 0.0) / root).scale(r0 / (s * s));
                let cond = Complex::new(r0 * (c.e0 - c.v_th) / s, 0.0) / root;
                for (i, (m, want)) in [(Modulation::Mean, mean), (Modulation::Variance, var), (Modulation::Conductance, cond)]
                    .into_iter()
                    .enumerate()
                {
                    gaps[i][j] = rel(st.response(m, at(f)).unwrap(), want);
                }
                phases[j] = deg(st.response(Modulation::Variance, at(f)).unwrap());
                if j == 2 {
                    assert!((deg(st.response(Modulation::Mean, at(f)).unwrap()) + 45.0).abs() < 1.5);
                    let g = deg(st.response(Modulation::Conductance, at(f)).unwrap());
                    assert!((g - if case == 0 { -45.0 } else { 135.0 }).abs() < 1.5, "case {case}: {g}");
                }
            }
            // The variance's phase vanishes (Fig. 1 Cii) as ω^{−½}, the phase of its own asymptote.
            assert!(phases[2].abs() < 4.0 && (phases[1] / phases[2] / 10f64.sqrt() - 1.0).abs() < 0.05, "{phases:?}");
            if case == 0 {
                for (i, want) in [[0.213, 0.069, 0.022], [0.10, 0.0092, 0.0011], [0.19, 0.059, 0.019]].iter().enumerate() {
                    for j in 0..3 {
                        assert!((gaps[i][j] / want[j] - 1.0).abs() < 0.1, "modulation {i}: {:?}", gaps[i]);
                    }
                }
                for i in [0, 2] {
                    assert!(((gaps[i][0] / gaps[i][1]) / 10f64.sqrt() - 1.0).abs() < 0.04, "{:?}", gaps[i]);
                    assert!(((gaps[i][1] / gaps[i][2]) / 10f64.sqrt() - 1.0).abs() < 0.04, "{:?}", gaps[i]);
                }
                assert!(gaps[1][1] / gaps[1][2] > 8.0, "the variance closes as 1/ω: {:?}", gaps[1]);
            } else {
                assert!(gaps.iter().all(|g| g[2] < 0.01 && g[2] < g[1] && g[1] < g[0]), "{gaps:?}");
            }
        }
    }

    /// The EIF's high-frequency limits are Richardson's eqs. (51), (52) and (54), with the phases the
    /// Fig. 2 and Fig. 3 captions print.
    ///
    /// Per unit amplitude: `r̂_E ≃ r₀/(iωτΔ_T)` and `r̂_σ² ≃ r₀/(iωτΔ_T²)` (51), both `−90°`;
    /// `r̂_g ≃ (ir₀/ωτ)(log ωτ + (V_T − E₀)/Δ_T + iπ/2 + γ − 1)` (52), a lead that tends to the
    /// caption's "−270° equivalent to a 90° phase advance" only as `1/log ω` — it is 103.9° at
    /// 100 kHz in case i, and the lattice's phase is that; `r̂_VT ≃ −r₀/Δ_T` (54), "a constant with
    /// a 180° phase". Measured at 2.5 µV and 100 kHz: every response within `1.8 × 10⁻³` of its
    /// asymptote in both cases, the variance in case ii the farthest, and every phase within 0.06°
    /// of its asymptote's.
    ///
    /// Eq. (54) gives `r̂_ΔT ≃ −r₀(Δ_T1/Δ_T) log(ωτ)`, which Richardson says is "given here to
    /// leading order" (p. 7). Carrying (53)'s `F_ΔT` through his eq. (46) gives the next terms too,
    /// `−r₀(Δ_T1/Δ_T)(log ωτ + γ − 1 + iπ/2)` — the ones (52) keeps — and the lattice converges to
    /// that (within `1.5 × 10⁻³` at 100 kHz), not to the leading order, which is 17% away there. The
    /// extra `iπ/2` is visible in Fig. 3 Bii, whose dashed asymptote at 1 kHz sits near `−160°`: the
    /// full asymptote's phase there is `−160.4°`, the leading order's `−180°`, and this module
    /// computes `−161.7°` in case ii (`−161.0°` in case i). And "the response amplitude increases
    /// with increasing frequency" (Fig. 3 caption): measured `|r̂_ΔT|` grows from 1 kHz to 100 kHz
    /// in both cases.
    #[test]
    fn the_eif_high_frequency_limits_are_eqs_51_52_and_54() {
        const EULER_GAMMA: f64 = 0.577_215_664_901_532_9;
        for c in [WhiteNoiseIf::FIG2_CASE_I, WhiteNoiseIf::FIG2_CASE_II] {
            let Model::Eif { v_t, delta_t } = c.model else { unreachable!() };
            let st = c.stationary(2.5e-6).unwrap();
            let r0 = st.rate();
            let f = 1e5;
            let wt = 2.0 * PI * f * c.tau;
            let iwt = Complex::new(0.0, wt);
            let log_terms = Complex::new(wt.ln() + EULER_GAMMA - 1.0, FRAC_PI_2);
            let asymptotes = [
                (Modulation::Mean, Complex::new(r0 / delta_t, 0.0) / iwt),
                (Modulation::Variance, Complex::new(r0 / (delta_t * delta_t), 0.0) / iwt),
                (Modulation::Conductance, Complex::new(0.0, r0 / wt) * (log_terms + Complex::new((v_t - c.e0) / delta_t, 0.0))),
                (Modulation::SpikeThreshold, Complex::new(-r0 / delta_t, 0.0)),
                (Modulation::SpikeWidth, log_terms.scale(-r0 / delta_t)),
            ];
            for (m, want) in asymptotes {
                let got = st.response(m, at(f)).unwrap();
                assert!(rel(got, want) < 5e-3, "E0 = {}, {m:?}: {got:?} against {want:?}", c.e0);
            }
            let deg = |m| st.response(m, at(f)).unwrap().arg().to_degrees();
            assert!((deg(Modulation::Mean) + 90.0).abs() < 0.2 && (deg(Modulation::Variance) + 90.0).abs() < 0.2);
            assert!((deg(Modulation::SpikeThreshold).abs() - 180.0).abs() < 0.2);
            // (52)'s lead is +90° plus arctan(π/2 ÷ the logarithms): 103.9° at 100 kHz in case i.
            let lead = asymptotes[2].1.arg().to_degrees();
            assert!(lead > 90.0 && (deg(Modulation::Conductance) - lead).abs() < 0.05, "{} {lead}", deg(Modulation::Conductance));
            let leading = Complex::new(-r0 / delta_t * wt.ln(), 0.0);
            assert!(rel(st.response(Modulation::SpikeWidth, at(f)).unwrap(), leading) > 0.15);
            let width_1k = st.response(Modulation::SpikeWidth, at(1e3)).unwrap();
            assert!(st.response(Modulation::SpikeWidth, at(f)).unwrap().norm() > width_1k.norm());
            let wt_1k = 2.0 * PI * 1e3 * c.tau;
            let full_1k = Complex::new(wt_1k.ln() + EULER_GAMMA - 1.0, FRAC_PI_2).scale(-r0 / delta_t);
            assert!((full_1k.arg().to_degrees() + 160.4).abs() < 0.05, "{}", full_1k.arg().to_degrees());
            if c.e0 == WhiteNoiseIf::FIG2_CASE_II.e0 {
                assert!((width_1k.arg().to_degrees() + 161.7).abs() < 0.05, "{}", width_1k.arg().to_degrees());
            } else {
                assert!((width_1k.arg().to_degrees() + 161.0).abs() < 0.05, "{}", width_1k.arg().to_degrees());
            }
        }
    }

    /// ⚠ Eqs. (50) and (53) print three driving terms without their dimensions; the lattice's drive is
    /// the correctly dimensioned one.
    ///
    /// Every `F_α` is a rate: it is added to the flux in eq. (18). With `P₀ ≃ r₀τ/ψ` and
    /// `m = ωτΔ_T/ψ` (49) the definitions (13) and (33) give `F_VT = V_T1 r₀/Δ_T`, `F_ΔT =
    /// (Δ_T1 r₀/Δ_T)(log ωτ − log m − 1)` and `F_g = m (g₁r₀/g₀ωτ)(log ωτ + (V_T − E₀)/Δ_T − log m)`.
    /// Eq. (53) prints `F_VT = (V_T1/Δ_T) r₀τ₀`, a spurious `τ₀`, and `F_ΔT` with no `r₀` at all;
    /// eq. (50) prints `F_g` with `ω` for `ωτ₀`. So the printed terms are the corrected ones times
    /// `τ₀`, `1/r₀` and `τ₀`: that is algebra, and the test checks only that the transcriptions below
    /// say it — each printed term evaluated as printed, with `m` from (49) at 1 kHz, against the
    /// corrected one, to `10⁻¹²`. The asymptotes (52) and (54) are the ones the corrected terms give
    /// — the test above — so the typos are in the intermediate display only.
    ///
    /// What the lattice is checked on is the corrected form. On Fig. 2's case i, at the lattice
    /// points above −35 mV where `ψ` exceeds 1 V and `P₀ ≃ r₀τ/ψ` holds, the drive `F_α/α₁` is the
    /// corrected term to within `4.78 × 10⁻³` at all 3500 of them, for all three (of the order of the
    /// scheme's half-cell offset `Δ/Δ_T = 3.3 × 10⁻³`). The three errors are the same number, because
    /// each drive's ratio to its corrected term is `ψP₀/(τr₀)`: the whole error is `P₀ ≃ r₀τ/ψ`.
    #[test]
    fn eqs_50_and_53_are_printed_without_their_dimensions() {
        let c = WhiteNoiseIf::FIG2_CASE_I;
        let Model::Eif { v_t, delta_t } = c.model else { unreachable!() };
        let st = c.stationary(WhiteNoiseIf::FIG_STEP).unwrap();
        let (r0, tau, n) = (st.rate(), c.tau, st.lattice().n);
        let omega = 2.0 * PI * 1e3;
        let (mut worst, mut count) = ([0.0f64; 3], 0);
        for k in 1..n {
            let v = st.voltage()[k];
            if v < -35e-3 {
                continue;
            }
            let x = (v - v_t) / delta_t;
            let corrected = [
                (Modulation::SpikeThreshold, r0 / delta_t),
                (Modulation::SpikeWidth, r0 / delta_t * (x - 1.0)),
                (Modulation::Conductance, r0 * (v - c.e0) / c.psi(v)),
            ];
            // As printed, with eq. (49)'s m.
            let m49 = omega * tau * delta_t / c.psi(v);
            let printed = [
                r0 * tau / delta_t,
                ((omega * tau).ln() - m49.ln() - 1.0) / delta_t,
                m49 * r0 / omega * ((omega * tau).ln() + (v_t - c.e0) / delta_t - m49.ln()),
            ];
            let factor = [tau, 1.0 / r0, tau];
            for (i, (m, right)) in corrected.into_iter().enumerate() {
                assert!((printed[i] / (right * factor[i]) - 1.0).abs() < 1e-12, "{m:?} at {v}");
                worst[i] = worst[i].max((st.drive_at(m, k).unwrap() / right - 1.0).abs());
            }
            count += 1;
        }
        assert_eq!(count, 3500);
        assert!(worst.iter().all(|&w| w < 6e-3 && (w / worst[0] - 1.0).abs() < 1e-9), "{worst:?}");
    }

    /// ⚠ The variance drive needs `∂P₀/∂V`, which Richardson does not say how to compute; read off
    /// the steady-state equation (43) it is wrong near the EIF's threshold, differenced it is not.
    ///
    /// Fig. 2's case i at 10 kHz, against the asymptote (51). Differenced, as [`Stationary::response`]
    /// does: within `1.1 × 10⁻²` of the asymptote at every step from 10 to 1.25 µV (measured
    /// `1.05 × 10⁻²` to `6.8 × 10⁻³`). Read off (43), `∂P₀/∂V = −GP₀ − τJ₀/σ²`: the ratio to the
    /// asymptote is `−0.29 + 9.23i` at 10 µV — nine times too large and at the wrong phase — and
    /// `|ratio − 1|` goes 9.3, 4.2, 1.8, 0.77 as the step halves: first order, and at 10 µV about
    /// 880 times the differenced drive's distance from the asymptote (measured 9.316 against
    /// 0.01055, a factor of 883.0). On the LIF, where `G` stays small, both
    /// readings are usable: at Fig. 1 case i, 10 Hz, 10 µV, the read-off response is 3.0% from the
    /// closed form and the differenced one 0.21%.
    #[test]
    fn the_variance_drive_is_differenced_not_read_off_the_ode() {
        let read_off = |st: &Stationary, lambda: Complex| {
            let c = *st.neuron();
            let s2 = c.sigma * c.sigma;
            let (v, p, j) = (st.voltage(), st.density(), st.flux());
            st.integrate(lambda, 2f64.powi(500), |k| Ok((-(v[k] - c.e0 - c.psi(v[k])) / s2 * p[k] - c.tau * j[k] / s2) / c.tau)).unwrap()
        };
        let c = WhiteNoiseIf::FIG2_CASE_I;
        let (mut gaps, mut differenced) = (Vec::new(), Vec::new());
        for dv in [10e-6, 5e-6, 2.5e-6, 1.25e-6] {
            let st = c.stationary(dv).unwrap();
            let asymptote = Complex::new(st.rate(), 0.0) / Complex::new(0.0, 2.0 * PI * 1e4 * c.tau * 9e-6);
            differenced.push(rel(st.response(Modulation::Variance, at(1e4)).unwrap(), asymptote));
            gaps.push(rel(read_off(&st, at(1e4)), asymptote));
        }
        assert!(differenced.iter().all(|&d| d < 1.1e-2), "{differenced:?}");
        assert!(gaps[0] > 9.0 && (gaps[0] / differenced[0] - 883.0).abs() < 1.0, "{gaps:?} {differenced:?}");
        for w in gaps.windows(2) {
            assert!(w[0] / w[1] > 2.0 && w[0] / w[1] < 2.5, "{gaps:?}");
        }
        let lif = WhiteNoiseIf::FIG1_CASE_I.stationary(10e-6).unwrap();
        let closed = Complex::new(534769.3997646779, 970873.4937236013);
        let a = rel(read_off(&lif, at(10.0)), closed);
        let b = rel(lif.response(Modulation::Variance, at(10.0)).unwrap(), closed);
        assert!((a - 0.0297).abs() < 0.002 && (b - 0.00212).abs() < 0.0002, "{a} {b}");
    }

    /// ⚠ The backward recursion overflows at high frequency, and rescaling each pair by its own
    /// power of two is exact.
    ///
    /// Unscaled, the recursion's components pass `f64::MAX` and the response comes out `NaN`.
    /// Measured by bisection on the frequency, the first component to overflow does so above, with
    /// the mean drive and then the variance drive: 10.8 and 10.6 kHz on Fig. 1's case i at 1 µV,
    /// 83.2 and 81.0 kHz on its case ii; 15.5 and 15.3 kHz on Fig. 2's case i at 10 µV, 78.0 and
    /// 77.2 kHz on its case ii. (The rate pair alone lasts to 11.0, 83.7, 15.6 and 78.0 kHz.) The
    /// last lines check Fig. 1's case i on either side of both. Rescaled, wherever the unscaled
    /// recursion survives the result is the same bits — checked from 100 Hz to 10 kHz, with a rescale
    /// at every `2⁸` as well as every `2⁵⁰⁰` — and where it does not, the result is the response:
    /// at 20 kHz and 30 kHz, `2.7 × 10⁻³` and `2.3 × 10⁻³` from eq. (51).
    #[test]
    fn the_response_survives_where_the_recursion_would_overflow() {
        let c = WhiteNoiseIf::FIG2_CASE_I;
        let st = c.stationary(WhiteNoiseIf::FIG_STEP).unwrap();
        let drive = |k| Ok(st.drive_at(Modulation::Mean, k).unwrap());
        for f in [100.0, 1000.0, 3000.0, 1e4] {
            let plain = st.integrate(at(f), f64::INFINITY, drive).unwrap();
            assert!(plain.re.is_finite());
            assert_eq!(st.integrate(at(f), 2f64.powi(500), drive).unwrap(), plain);
            assert_eq!(st.integrate(at(f), 2f64.powi(8), drive).unwrap(), plain);
            assert_eq!(st.response(Modulation::Mean, at(f)).unwrap(), plain);
        }
        // With a dead time the reset term is complex, `e^{−iωτ_ref}`, and both of its parts are
        // scaled with the pair it enters.
        let dead = WhiteNoiseIf { t_ref: 2e-3, ..c }.stationary(WhiteNoiseIf::FIG_STEP).unwrap();
        for m in [Modulation::Mean, Modulation::Variance] {
            let drive = |k| Ok(dead.drive_at(m, k).unwrap());
            for f in [300.0, 3000.0] {
                let plain = dead.integrate(at(f), f64::INFINITY, drive).unwrap();
                assert_eq!(dead.integrate(at(f), 2f64.powi(8), drive).unwrap(), plain, "{m:?} {f}");
            }
        }
        for f in [2e4, 3e4] {
            assert!(st.integrate(at(f), f64::INFINITY, drive).unwrap().re.is_nan());
            let got = st.response(Modulation::Mean, at(f)).unwrap();
            let want = Complex::new(st.rate() / 3e-3, 0.0) / Complex::new(0.0, 2.0 * PI * f * c.tau);
            assert!(rel(got, want) < 8e-3, "{f}: {got:?} {want:?}");
            assert_eq!(st.integrate(at(f), 2f64.powi(8), drive).unwrap(), got);
        }
        let lif = WhiteNoiseIf::FIG1_CASE_I.stationary(1e-6).unwrap();
        for (m, below, above) in [(Modulation::Mean, 10.8e3, 10.9e3), (Modulation::Variance, 10.6e3, 10.7e3)] {
            let drive = |k| Ok(lif.drive_at(m, k).unwrap());
            let ends = |f: f64| lif.recur(at(f), f64::INFINITY, drive).unwrap();
            let finite = |z: Complex| z.re.is_finite() && z.im.is_finite();
            assert!(finite(ends(below).ja) && finite(ends(below).jr), "{m:?}");
            assert!(!finite(ends(above).ja) && finite(ends(above).jr), "{m:?}");
            assert!(lif.integrate(at(above), f64::INFINITY, drive).unwrap().re.is_nan());
            assert!(lif.integrate(at(below), f64::INFINITY, drive).unwrap().re.is_finite());
        }
    }

    /// Off the imaginary axis the pairs are multiplied up as well as down, and each pair keeps its own
    /// scale.
    ///
    /// On the negative real axis, which the eigenvalue search can visit, the components can grow or
    /// shrink. Unscaled, Fig. 1's case i at 1 µV returns `NaN` at `λ = −10⁷` per second, and Fig. 2's
    /// case i at `−10⁶`, from overflow; rescaled both are finite, and every value the unscaled
    /// recursion does return — at `−10⁴`, `−10⁵`, `−3 × 10⁵` and `−10⁶` for the LIF — comes back bit
    /// for bit, rescaled every `2⁸` as well as every `2⁵⁰⁰`. With the floor moved to −150 mV the
    /// flux `ĵ_r` decays past the normal range instead: at `−3 × 10⁴` the unscaled recursion ends on
    /// a subnormal with five significant bits and returns −681, where both rescaled schedules return
    /// the same `−6.925 × 10¹⁵⁸` — this lattice's response is enormous there, beside a pole — and at
    /// `−10⁵` the response is past `f64::MAX`, which [`Stationary::response`] refuses.
    ///
    /// The two pairs have separate scales because they can be astronomically far apart. A neuron
    /// 30σ below threshold (`E₀ = −68` mV, `σ = 0.6` mV) has the Siegert rate `2.208 × 10⁻¹⁹³` Hz
    /// (`tools/density_reference.py` section 1b) and the 10 µV lattice rate `1.720 × 10⁻¹⁹³` Hz, 22%
    /// low — the half-cell rule of the module doc in its exponential form, `e^{−½Δ(V_th − E)/σ²} =
    /// e^{−1/4}`, to `2.2 × 10⁻⁴`. Its unnormalised density reaches `3.9 × 10¹⁹⁵`, so its rate pair
    /// passes `2⁵⁰⁰` and is divided;
    /// at `λ = 10⁻³⁰⁰` its drive flux ends at `−4.4 × 10⁻²⁹⁶`, which the same division would put
    /// below the smallest subnormal. With separate scales the response is the same at `10⁻³⁰⁰`,
    /// `10⁻²⁵⁰`, `10⁻²⁰⁰` and `10⁻¹⁰⁰` per second, as a response must be at frequencies that low,
    /// and equal to the unscaled recursion's.
    #[test]
    fn a_decaying_recursion_and_a_far_subthreshold_neuron_keep_their_digits() {
        let lif = WhiteNoiseIf::FIG1_CASE_I.stationary(1e-6).unwrap();
        let drive = |k| Ok(lif.drive_at(Modulation::Mean, k).unwrap());
        for lambda in [-1e4, -1e5, -3e5, -1e6] {
            let z = Complex::new(lambda, 0.0);
            let plain = lif.integrate(z, f64::INFINITY, drive).unwrap();
            assert!(plain.re.is_finite(), "{lambda}");
            assert_eq!(lif.integrate(z, 2f64.powi(500), drive).unwrap(), plain);
            assert_eq!(lif.integrate(z, 2f64.powi(8), drive).unwrap(), plain);
        }
        let deep = Complex::new(-1e7, 0.0);
        assert!(lif.integrate(deep, f64::INFINITY, drive).unwrap().re.is_nan());
        let got = lif.response(Modulation::Mean, deep).unwrap();
        assert!(got.re.is_finite() && got.im == 0.0, "{got:?}");
        assert_eq!(lif.integrate(deep, 2f64.powi(8), drive).unwrap(), got);
        // With the floor at −150 mV the decay goes past the normal range: unscaled, ĵ_r ends as
        // a subnormal with five significant bits and the quotient is garbage; rescaled on two
        // schedules the answer is the same bits, and it is enormous — at −3 × 10⁴ per second, near a
        // pole of the lattice's response; at −10⁵ past f64::MAX, which is refused.
        let low = WhiteNoiseIf { v_lb: -150e-3, ..WhiteNoiseIf::FIG1_CASE_I }.stationary(1e-6).unwrap();
        let drive = |k| Ok(low.drive_at(Modulation::Mean, k).unwrap());
        let z = Complex::new(-3e4, 0.0);
        let garbage = low.integrate(z, f64::INFINITY, drive).unwrap();
        let got = low.response(Modulation::Mean, z).unwrap();
        assert!(garbage.re.abs() < 1e4 && (got.re / -6.925e158 - 1.0).abs() < 1e-3, "{garbage:?} {got:?}");
        assert_eq!(low.integrate(z, 2f64.powi(8), drive).unwrap(), got);
        assert_eq!(
            low.response(Modulation::Mean, Complex::new(-1e5, 0.0)).unwrap_err().to_string(),
            "the response is not representable in f64 at these parameters"
        );
        let eif = WhiteNoiseIf::FIG2_CASE_I.stationary(10e-6).unwrap();
        let deep = Complex::new(-1e6, 0.0);
        assert!(eif.integrate(deep, f64::INFINITY, |k| Ok(eif.drive_at(Modulation::Mean, k).unwrap())).unwrap().re.is_nan());
        assert!(eif.response(Modulation::Mean, deep).unwrap().re.is_finite());

        let far = WhiteNoiseIf { e0: -68e-3, sigma: 0.6e-3, ..WhiteNoiseIf::FIG1_CASE_I }.stationary(10e-6).unwrap();
        let siegert = 2.2080076369031977e-193;
        assert!((far.rate() / 1.72e-193 - 1.0).abs() < 0.01, "{}", far.rate());
        assert!((far.rate() / siegert / (-0.25f64).exp() - 1.0).abs() < 5e-4, "{}", far.rate());
        assert!((far.neuron().siegert().unwrap().rate().unwrap() / siegert - 1.0).abs() < 1e-12);
        let peak = far.density().iter().copied().fold(0.0, f64::max) / far.rate();
        assert!(peak > 2f64.powi(500) && (peak / 3.87e195 - 1.0).abs() < 0.01, "{peak:e}");
        let drive = |k| Ok(far.drive_at(Modulation::Mean, k).unwrap());
        let ends = far.recur(Complex::new(1e-300, 0.0), f64::INFINITY, drive).unwrap();
        assert!(ends.ja.re < 0.0 && ends.ja.norm() * 2f64.powi(-500) < 5e-324 && ends.ja.norm() > 1e-300, "{ends:?}");
        let low = far.response(Modulation::Mean, Complex::new(1e-100, 0.0)).unwrap();
        for lambda in [1e-300, 1e-250, 1e-200] {
            let z = Complex::new(lambda, 0.0);
            let got = far.response(Modulation::Mean, z).unwrap();
            assert!((got.re / low.re - 1.0).abs() < 1e-14, "{lambda:e}: {got:?} against {low:?}");
            assert_eq!(far.integrate(z, f64::INFINITY, drive).unwrap(), got);
        }
    }

    /// Frequencies in `[lo, lo + n·step]` at which `|r̂|` has a strict local maximum, and at which the
    /// phase crosses zero (`Im r̂` changes sign), on a grid of spacing `step`.
    fn peaks_and_zeros(st: &Stationary, m: Modulation, lo: f64, step: f64, n: usize) -> (Vec<f64>, Vec<f64>) {
        let f = |i: usize| lo + i as f64 * step;
        let r: Vec<Complex> = (0..=n).map(|i| st.response(m, at(f(i))).unwrap()).collect();
        let peaks = (1..n).filter(|&i| r[i].norm() > r[i - 1].norm() && r[i].norm() > r[i + 1].norm()).map(f).collect();
        let zeros = (0..n).filter(|&i| (r[i].im > 0.0) != (r[i + 1].im > 0.0)).map(f).collect();
        (peaks, zeros)
    }

    /// The maximum of `g` on `[a, b]` by golden section: sixty contractions by `(√5 − 1)/2`, which
    /// shrink the bracket by `3 × 10⁻¹³`.
    fn golden_max(g: impl Fn(f64) -> f64, mut a: f64, mut b: f64) -> f64 {
        let r = (5f64.sqrt() - 1.0) / 2.0;
        let (mut c, mut d) = (b - r * (b - a), a + r * (b - a));
        let (mut gc, mut gd) = (g(c), g(d));
        for _ in 0..60 {
            if gc > gd {
                (b, d, gd) = (d, c, gc);
                c = b - r * (b - a);
                gc = g(c);
            } else {
                (a, c, gc) = (c, d, gd);
                d = a + r * (b - a);
                gd = g(d);
            }
        }
        0.5 * (a + b)
    }

    /// The zero of `g` in `[a, b]`, across which it changes sign, by sixty bisections.
    fn bisect_zero(g: impl Fn(f64) -> f64, mut a: f64, mut b: f64) -> f64 {
        let below = g(a) > 0.0;
        assert!(below != (g(b) > 0.0), "no sign change in [{a}, {b}]");
        for _ in 0..60 {
            let mid = 0.5 * (a + b);
            if (g(mid) > 0.0) == below {
                a = mid;
            } else {
                b = mid;
            }
        }
        0.5 * (a + b)
    }

    /// What Richardson's captions say about the shapes of Figs. 1–3, located on independent
    /// references, and the lattice converging to them.
    ///
    /// The references are `tools/density_reference.py`: the peaks of `|r̂|` and phase zeros of Brunel's
    /// eq. (46) for the LIF (section 4c), and of the ODE route of section 7 for the EIF (7b), each
    /// found on a grid and refined by `minimize_scalar` or `brentq`. The lattice's are found on a grid
    /// over the same range — the same number of them, which pins that there is no other — and
    /// refined by golden section or bisection at two steps (2 and 1 µV for the LIF, 5 and 2.5 µV for
    /// the EIF). Every one converges to its reference at first order: at the finer step the LIF's
    /// are within 0.004 Hz of theirs, but for the broad third resonance of `|r̂_E|` (0.25 Hz) and
    /// the second of `|r̂_σ²|` (0.095 Hz), and the EIF's within 0.09 Hz; and every distance halves
    /// with the step, ratios 1.96 to 2.08.
    ///
    /// What the captions say, and what the references find:
    ///
    /// - Fig. 1 Bi and Ci, the LIF's case i, `r₀ = 46.2` Hz: "resonances are seen at the firing rate
    ///   and its harmonics" — `|r̂_E|` peaks at 46.44, 94.40 and 140.31 Hz, 0.5%, 2.1% and 1.2% from
    ///   `r₀`, `2r₀` and `3r₀`, and `|r̂_σ²|` at 46.82 and 98.44 Hz, 1.3% and 6.5% from `r₀` and `2r₀`.
    /// - Fig. 1 Cii, case ii: "a resonance and phase zero are seen near 10 Hz" — `|r̂_σ²|` peaks at
    ///   14.89 Hz and its phase crosses zero at 11.10 Hz.
    /// - Fig. 2 Bi and Fig. 3 Ai, the EIF's case i, `r₀ = 44.05` Hz: resonances "at the firing rate
    ///   ... and its harmonics" — `|r̂_E|` peaks at 42.21 and 80.15 Hz and `|r̂_VT|` at 44.69 and
    ///   95.02 Hz: within 4.2% of `r₀` and 9.1% of `2r₀`.
    /// - Fig. 3 Aii: "a broad resonance is seen near 100 Hz" — `|r̂_VT|` peaks at 58.09 Hz, and at
    ///   100 Hz it is 98.5% of its peak.
    /// - Fig. 2 Ci: "the modulation at low frequency is 180°" — `r̂_σ²` at 0.01 Hz is negative,
    ///   `−63 908` Hz/V² on the ODE route, because in case i noise lowers the rate.
    /// - Fig. 2A: "for `E₀` greater than about −47 mV increasing noise decreases the firing rate" —
    ///   the `σ = 2` and 6 mV rate curves cross at −46.497 mV (section 3b, `brentq` on section 3's
    ///   forward integration); the lattice's cross within 4.4 µV of it at 10 µV.
    ///
    /// One caption is loose: Fig. 2 Cii's "a resonance and phase zero are seen at ~10 Hz" is right
    /// about the resonance, 11.59 Hz, but the EIF's phase crosses zero at 3.27 Hz and is `−10.75°`
    /// at 10 Hz, on the ODE route and on the lattice alike; the words fit Fig. 1 Cii's LIF better.
    #[test]
    fn richardsons_captions_describe_what_the_lattice_computes() {
        struct Feature {
            neuron: WhiteNoiseIf,
            m: Modulation,
            grid: (f64, f64, usize),
            peaks: &'static [f64],
            zeros: &'static [f64],
        }
        let (lif_i, lif_ii) = (WhiteNoiseIf::FIG1_CASE_I, WhiteNoiseIf::FIG1_CASE_II);
        let (eif_i, eif_ii) = (WhiteNoiseIf::FIG2_CASE_I, WhiteNoiseIf::FIG2_CASE_II);
        let features = [
            Feature { neuron: lif_i, m: Modulation::Mean, grid: (20.0, 0.5, 260), peaks: &[46.44085270707605, 94.401631892767, 140.30951768159855], zeros: &[46.66242518233779] },
            Feature { neuron: lif_i, m: Modulation::Variance, grid: (20.0, 0.5, 260), peaks: &[46.82354213157343, 98.44337211575912], zeros: &[] },
            Feature { neuron: lif_ii, m: Modulation::Variance, grid: (0.5, 0.5, 399), peaks: &[14.893355253673116], zeros: &[11.099533154558593] },
            Feature { neuron: eif_i, m: Modulation::Mean, grid: (20.0, 0.5, 260), peaks: &[42.21221101958546, 80.1511229000994], zeros: &[] },
            Feature { neuron: eif_i, m: Modulation::SpikeThreshold, grid: (20.0, 0.5, 260), peaks: &[44.6929536298177, 95.01544711776714], zeros: &[49.321293039034636, 61.59345710479597] },
            Feature { neuron: eif_ii, m: Modulation::Variance, grid: (0.5, 0.5, 399), peaks: &[11.594721968414115], zeros: &[3.265507690016857] },
            Feature { neuron: eif_ii, m: Modulation::SpikeThreshold, grid: (2.0, 2.0, 199), peaks: &[58.087187248414004], zeros: &[80.6365515592297] },
        ];
        let mut worst = (0.0f64, f64::INFINITY, 0.0f64);
        for &Feature { neuron, m, grid: (lo, step, n), peaks, zeros } in &features {
            let steps = if neuron.model == Model::Lif { [2e-6, 1e-6] } else { [5e-6, 2.5e-6] };
            let (p, z) = peaks_and_zeros(&neuron.stationary(steps[0]).unwrap(), m, lo, step, n);
            assert!(p.len() == peaks.len() && z.len() == zeros.len(), "{m:?} at E0 = {}: {p:?} {z:?}", neuron.e0);
            let mut gaps = [Vec::new(), Vec::new()];
            for (row, dv) in steps.into_iter().enumerate() {
                let st = neuron.stationary(dv).unwrap();
                let r = |f: f64| st.response(m, at(f)).unwrap();
                for (&g, &want) in p.iter().zip(peaks) {
                    gaps[row].push(golden_max(|f| r(f).norm(), g - 2.0 * step, g + 2.0 * step) - want);
                }
                for (&g, &want) in z.iter().zip(zeros) {
                    gaps[row].push(bisect_zero(|f| r(f).im, g - 2.0 * step, g + 3.0 * step) - want);
                }
            }
            for (coarse, fine) in gaps[0].iter().zip(&gaps[1]) {
                worst = (worst.0.max(fine.abs()), worst.1.min(coarse / fine), worst.2.max(coarse / fine));
            }
        }
        assert!(worst.0 < 0.3 && worst.1 > 1.9 && worst.2 < 2.15, "{worst:?}");

        // The captions' words, against the references.
        let multiple = |f: f64, r0: f64| (f / (r0 * (f / r0).round()) - 1.0).abs();
        let (lif_r0, eif_r0) = (46.21557620398916, 44.04657807442763);
        let off: Vec<f64> = features[0].peaks.iter().chain(features[1].peaks).map(|&f| multiple(f, lif_r0)).collect();
        assert!(off.iter().zip([0.0049, 0.0213, 0.0120, 0.0132, 0.0650]).all(|(o, w)| (o - w).abs() < 1e-4), "{off:?}");
        for f in [features[3].peaks, features[4].peaks] {
            assert!(multiple(f[0], eif_r0) < 0.042 && multiple(f[1], eif_r0) < 0.091 && (f[1] / eif_r0).round() == 2.0, "{f:?}");
        }
        let eif_ii = eif_ii.stationary(5e-6).unwrap();
        let vt = |f: f64| eif_ii.response(Modulation::SpikeThreshold, at(f)).unwrap().norm();
        let broad = vt(100.0) / vt(golden_max(vt, 56.0, 60.0));
        assert!((broad - 0.9849489372074396).abs() < 1e-3 && broad > 0.98, "{broad}");
        let phase = eif_ii.response(Modulation::Variance, at(10.0)).unwrap().arg().to_degrees();
        assert!((phase + 10.74694797089162).abs() < 0.03, "{phase}");
        let low = eif_i.stationary(5e-6).unwrap().response(Modulation::Variance, at(0.01)).unwrap();
        assert!(low.re < 0.0 && rel(low, Complex::new(-63907.778277970436, 215.30767756486148)) < 5e-3, "{low:?}");
        let rate = |e0: f64, sigma: f64| WhiteNoiseIf { e0, sigma, ..eif_i }.stationary(10e-6).unwrap().rate();
        let (mut lo, mut hi) = (-55e-3, -40e-3);
        for _ in 0..40 {
            let mid = 0.5 * (lo + hi);
            if rate(mid, 2e-3) < rate(mid, 6e-3) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        assert!((lo + 0.046497033886355746).abs() < 2e-5, "{lo}");
    }

    /// Table 1's three parameter sets, Fig. 7's network (`C_E = 1000`, `J = 0.1` mV, `D = 1.5` ms):
    /// `(g, ν_ext/ν_thr)`, the printed theoretical rate, and `SciPy`'s solution of eqs. (20)–(21)
    /// (`tools/density_reference.py` section 5, `brentq` to `10⁻¹³` Hz).
    const TABLE_1: [(f64, f64, f64, f64); 3] =
        [(6.0, 4.0, 55.8, 55.841262376204384), (5.0, 2.0, 38.0, 37.94969708576336), (4.5, 0.9, 6.5, 6.516702268415083)];

    /// Brunel's Table 1 rates: the self-consistent `ν = φ(μ(ν), σ(ν))` of his network with threshold
    /// integration as `φ`, against [`BrunelNetwork::self_consistent_rate`] (the Siegert `φ`), `SciPy`'s
    /// solution of the same equations, and the printed digits.
    ///
    /// The Siegert solution agrees with `SciPy`'s to `10⁻¹²`; the lattice's converges to it at first
    /// order (measured at 5 µV: `3.6 × 10⁻⁵`, `9.2 × 10⁻⁵` and `1.3 × 10⁻³` relative, each halving
    /// with the step). Points B and D round to the printed 55.8 and 6.5 Hz.
    ///
    /// Point C prints 38.0 Hz where eqs. (20)–(21) give 37.9497 Hz, which rounds to 37.9: 0.0003 Hz,
    /// eight parts in a million, below the rounding boundary, and read as the paper's own evaluation
    /// precision rather than a defect. The test pins both numbers.
    #[test]
    fn brunels_table_1_rates() {
        for (g, ratio, printed, scipy) in TABLE_1 {
            let net = brunel_net(1000.0, 0.1e-3, g, ratio);
            let siegert = net.self_consistent_rate().unwrap();
            assert!((siegert / scipy - 1.0).abs() < 1e-12, "{siegert} {scipy}");
            let lattice = |dv: f64| NetworkDensity { net, v_lb: -40e-3, dv }.rate().unwrap();
            let (coarse, fine) = (lattice(10e-6) / scipy - 1.0, lattice(5e-6) / scipy - 1.0);
            assert!(fine.abs() < 3e-3 && (coarse / fine - 2.0).abs() < 0.05, "{coarse} {fine}");
            let rounded = (siegert * 10.0).round() / 10.0;
            if g == 5.0 {
                assert_eq!(rounded, 37.9);
                assert!((siegert - 37.95).abs() < 4e-4 && printed == 38.0);
            } else {
                assert_eq!(rounded, printed);
            }
        }
    }

    /// Brunel's Table 1 frequencies, and which of his points are unstable, from the eigenvalues of
    /// the linearised network.
    ///
    /// The references are the roots of Brunel's eq. (46) itself, `tools/density_reference.py`
    /// section 5: `mpmath` evaluated (46) on a grid over growth rates from −600 to 400 per second and
    /// frequencies from 5 Hz to 1 kHz, refined every local minimum of `|1 − e^{−λD}X(λ)|`, and
    /// printed each root. The root with the largest real part is, at B, `51.498 + 2πi · 190.209 Hz`;
    /// at C, `−134.878 + 2πi · 124.276 Hz`; at D, `7.930 + 2πi · 28.996 Hz`. The lattice's Newton
    /// iteration, started at the printed frequency (at 100 Hz for C, which prints none), converges
    /// to each in five or six steps, and on each at first order in the lattice step: measured at
    /// 2.5 µV within `2.6 × 10⁻⁴` in frequency and `2.8 × 10⁻⁴` in `λ`, both halving from 5 µV.
    ///
    /// So: B and D are unstable (positive real part), oscillating at 190 Hz and 29 Hz as printed —
    /// the SI states of Fig. 8 B and D. At C, the AI state, for which the table prints no
    /// frequency, the root with the largest real part in the reference's window (growth rates
    /// −600 to 400 per second, 5 Hz to 1 kHz) is stable; a real root, one below 5 Hz or one growing
    /// faster than 400 per second was not searched for. That root sits at 124 Hz: Fig. 9A's power
    /// spectrum at the same parameters peaks "near 100 Hz".
    #[test]
    fn brunels_table_1_frequencies_are_the_eigenvalues_of_eq_46() {
        let roots = [(51.49760792062444, 190.20921707058415, 190.0), (-134.87769013262533, 124.27564005005671, 100.0), (7.930187083252262, 28.99550151985892, 29.0)];
        for ((g, ratio, _, _), (re, f, guess)) in TABLE_1.iter().zip(roots) {
            let want = Complex::new(re, 2.0 * PI * f);
            let mut gaps = Vec::new();
            for dv in [5e-6, 2.5e-6] {
                let nd = NetworkDensity { net: brunel_net(1000.0, 0.1e-3, *g, *ratio), v_lb: -40e-3, dv };
                let e = nd.eigenvalue(nd.rate().unwrap(), at(guess)).unwrap();
                assert!(e.steps <= 6, "{}", e.steps);
                assert_eq!(e.lambda.re > 0.0, re > 0.0);
                gaps.push(rel(e.lambda, want));
                if dv == 2.5e-6 {
                    assert!((e.lambda.im / want.im - 1.0).abs() < 5e-4);
                }
            }
            assert!(gaps[1] < 6e-4 && (gaps[0] / gaps[1] - 2.0).abs() < 0.05, "{gaps:?}");
        }
    }

    /// One cell of the network is the neuron of Brunel's eq. (20), in Richardson's convention; and
    /// the characteristic function is `1 − e^{−λD}[(∂μ/∂ν) r̂_E + (∂σ_B²/∂ν) r̂_σ²/2]`, zero at an
    /// eigenvalue.
    ///
    /// `μ₀ = C_E Jτ[ν_ext + ν(1 − gγ)]` and `σ₀² = C_E J²τ[ν_ext + ν(1 + g²γ)]`, with `ν_thr =
    /// θ/(C_E Jτ) = 10` Hz (p. 188), so `E₀ = μ₀` above a zero rest and Richardson's `σ = σ₀/√2`.
    /// The factor ½ on the variance response is the same `√2`: `σ_B² = 2σ²`.
    ///
    /// The first check recomputes the loop from the two responses, so it pins the assembly — the
    /// ½, the signs, the delay — and not the idea. The idea is checked against Brunel's own
    /// characteristic function: `1 − e^{−λD}X(λ)` of eq. (46), `tools/density_reference.py`
    /// section 5b, for this cell at `λ = 30 + 2πi · 150` per second, with `G` and `H` built from
    /// the cell's own Siegert rate, is `−0.07475 − 0.60288i`. The lattice's is `3.0 × 10⁻³` from it
    /// at 10 µV and `1.5 × 10⁻³` at 5 µV: first order. (The eigenvalue test above is the other
    /// independent check: its roots are eq. (46)'s.)
    #[test]
    fn a_network_cell_is_eq_20_and_the_characteristic_is_the_delayed_loop() {
        let (g, ratio, nu) = (6.0, 4.0, 55.8);
        let net = brunel_net(1000.0, 0.1e-3, g, ratio);
        let nd = NetworkDensity { net, v_lb: -40e-3, dv: 10e-6 };
        let cell = nd.neuron(nu).unwrap();
        let (ce, j, tau, gamma) = (1000.0, 0.1e-3, 20e-3, 0.25);
        let nu_ext = ratio * 20e-3 / (ce * j * tau);
        assert!((nu_ext - 40.0).abs() < 1e-12);
        let mu = ce * j * tau * (nu_ext + nu * (1.0 - g * gamma));
        let var = ce * j * j * tau * (nu_ext + nu * (1.0 + g * g * gamma));
        assert!((cell.e0 - mu).abs() < 1e-15 && (cell.sigma * cell.sigma * 2.0 / var - 1.0).abs() < 1e-14, "{cell:?}");
        assert_eq!((cell.tau, cell.v_th, cell.v_re, cell.t_ref, cell.v_lb, cell.model), (tau, 20e-3, 10e-3, 2e-3, -40e-3, Model::Lif));

        let lambda = Complex::new(30.0, 2.0 * PI * 150.0);
        let st = cell.stationary(10e-6).unwrap();
        let (dmu, dvar) = (ce * j * tau * (1.0 - g * gamma), ce * j * j * tau * (1.0 + g * g * gamma));
        let gain = st.response(Modulation::Mean, lambda).unwrap().scale(dmu) + st.response(Modulation::Variance, lambda).unwrap().scale(dvar / 2.0);
        let want = Complex::new(1.0, 0.0) - gain * lambda.scale(-1.5e-3).exp();
        assert!(rel(nd.characteristic(nu, lambda).unwrap(), want) < 2e-15);
        let brunel = Complex::new(-0.07475008022803799, -0.6028752035974095);
        let off = |dv: f64| rel(NetworkDensity { dv, ..nd }.characteristic(nu, lambda).unwrap(), brunel);
        let (coarse, fine) = (off(10e-6), off(5e-6));
        assert!(fine < 3e-3 && (coarse / fine - 2.0).abs() < 0.05, "{coarse} {fine}");
        let root = nd.eigenvalue(nd.rate().unwrap(), at(190.0)).unwrap();
        assert!(nd.characteristic(nd.rate().unwrap(), root.lambda).unwrap().norm() < 5e-14);
    }

    /// The fast Hopf line at `g = 8`, where the asynchronous state loses stability to the fast
    /// oscillation — Brunel's Fig. 7 (`C_E = 1000`, `J = 0.1` mV) and Fig. 2A, both `D = 1.5` ms.
    /// Fig. 2's network is taken to be Fig. 1's, `C_E = 4000`, `J = 0.2` mV: Fig. 2 prints no
    /// parameters of its own, and page 196 calls it the network "with higher connectivity" than
    /// Fig. 7's. Fig. 3's curve bears the reading out (below).
    ///
    /// Found by bisection on `ν_ext/ν_thr` for the external drive at which the real part of the fast
    /// eigenvalue is zero. Reference: the same bisection on Brunel's eq. (46),
    /// `tools/density_reference.py` section 6: `2.4641` at 161.2 Hz for Fig. 7's network, `2.5407`
    /// at 130.7 Hz for Fig. 2A's.
    /// Measured at 5 µV: within `8.6 × 10⁻⁴` in the drive and `5.7 × 10⁻⁴` in the frequency, both.
    ///
    /// Against the figures, as far as they can be read: Fig. 7's line meets `g = 8` at
    /// `ν_ext/ν_thr ≈ 2.48`, read at 300 dpi to about ±0.05; Fig. 2A's at about 2.5; and Fig. 3,
    /// which plots the frequency on Fig. 2's instability line, starts its `D = 1.5` ms, `g = 8` curve
    /// at `ν_ext/ν_thr ≈ 2.5` and 130 Hz.
    #[test]
    fn the_fast_hopf_line_at_g_8_is_where_the_figures_draw_it() {
        for (c_exc, j, drive, freq, read) in [(1000.0, 0.1e-3, 2.4641183720398203, 161.19350409481368, 2.48), (4000.0, 0.2e-3, 2.5407292929246523, 130.73844945576784, 2.5)] {
            let (mut lo, mut hi, mut guess) = (2.0f64, 3.5f64, at(freq));
            for _ in 0..30 {
                let mid = 0.5 * (lo + hi);
                let nd = NetworkDensity { net: brunel_net(c_exc, j, 8.0, mid), v_lb: -60e-3, dv: 5e-6 };
                let e = nd.eigenvalue(nd.rate().unwrap(), guess).unwrap();
                guess = e.lambda;
                if e.lambda.re > 0.0 {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }
            let at_line = 0.5 * (lo + hi);
            let f = guess.im / (2.0 * PI);
            assert!((at_line / drive - 1.0).abs() < 2e-3 && (f / freq - 1.0).abs() < 1.5e-3, "{at_line} {f}");
            assert!((at_line - read).abs() < 0.05);
        }
    }

    /// The complex arithmetic the recursion runs on, against values known exactly.
    #[test]
    fn complex_arithmetic_is_exact_where_it_can_be() {
        let (a, b) = (Complex::new(3.0, 4.0), Complex::new(1.0, -2.0));
        // Smith's division on both of its branches, and where |b|² would overflow or underflow.
        assert_eq!(a / Complex::new(2.0, 1.0), Complex::new(2.0, 1.0));
        assert_eq!(Complex::new(3e200, 4e200) / Complex::new(1e200, -2e200), Complex::new(-1.0, 2.0));
        assert_eq!(Complex::new(1e-200, 1e-200) / Complex::new(2e-200, 0.0), Complex::new(0.5, 0.5));
        assert_eq!(Complex::new(1e-200, 3e-200) / Complex::new(0.0, 1e-200), Complex::new(3.0, -1.0));
        assert_eq!(a + b, Complex::new(4.0, 2.0));
        assert_eq!(a - b, Complex::new(2.0, 6.0));
        assert_eq!(a * b, Complex::new(11.0, -2.0));
        assert_eq!(a / b, Complex::new(-1.0, 2.0));
        assert_eq!(-a, Complex::new(-3.0, -4.0));
        assert_eq!(a.scale(0.5), Complex::new(1.5, 2.0));
        assert_eq!(a.norm(), 5.0);
        assert_eq!(Complex::new(-1.0, 0.0).arg(), PI);
        assert_eq!(Complex::new(0.0, -2.0).arg(), -FRAC_PI_2);
        assert_eq!(Complex::new(1.0, 1.0).arg(), FRAC_PI_4);
        let e = Complex::new(1.0, PI / 2.0).exp();
        assert!(e.re.abs() < 1e-15 && (e.im - E).abs() < 1e-15, "{e:?}");
        assert_eq!(Complex::new(0.0, 0.0).exp(), Complex::new(1.0, 0.0));
        let q = Complex::new(-2.0, 0.0).exp();
        assert_eq!((q.re, q.im), ((-2.0f64).exp(), 0.0));
    }

    /// Richardson's lattices, and the rules that make a lattice his: the threshold and the reset on
    /// lattice points (Appendix A), no more than [`MAX_CELLS`] cells.
    ///
    /// Fig. 1's neuron at 10 µV has 5000 cells with the reset at 4000; Fig. 2's, with its 0 mV
    /// threshold, 10 000 with the reset at 4000. A threshold [`LATTICE_TOLERANCE`]/2 of a step off a
    /// lattice point is snapped onto it (the step is recomputed so that it divides the range
    /// exactly), and twice that is refused; so is a reset that rounds onto an end of the lattice. The
    /// boundary tests use binary fractions, so that the tolerance is not decided by decimal rounding.
    #[test]
    fn the_lattice_is_richardsons() {
        let l = WhiteNoiseIf::FIG1_CASE_I.lattice(WhiteNoiseIf::FIG_STEP).unwrap();
        assert_eq!((l.n, l.k_re), (5000, 4000));
        assert!((l.dv - 1e-5).abs() < 1e-20);
        let l = WhiteNoiseIf::FIG2_CASE_II.lattice(WhiteNoiseIf::FIG_STEP).unwrap();
        assert_eq!((l.n, l.k_re), (10000, 4000));
        // In floating point the 10 µV step divides both ranges exactly and misses the reset by
        // 4.5 × 10⁻¹³ of a step: far inside the tolerance.
        for c in [WhiteNoiseIf::FIG1_CASE_I, WhiteNoiseIf::FIG2_CASE_I] {
            assert_eq!(((c.v_th - c.v_lb) / WhiteNoiseIf::FIG_STEP).fract(), 0.0);
            let at_reset = (c.v_re - c.v_lb) / c.lattice(WhiteNoiseIf::FIG_STEP).unwrap().dv;
            assert!((at_reset - 4000.0).abs() < 1e-12, "{at_reset}");
        }

        // Range 1 V on steps of 2⁻¹⁰ V: 1024 cells, reset at cell 256.
        let base = WhiteNoiseIf { model: Model::Lif, tau: 1.0, e0: -0.5, sigma: 0.1, v_th: 0.0, v_re: -0.75, v_lb: -1.0, t_ref: 0.0 };
        let step = 2f64.powi(-10);
        assert_eq!(base.lattice(step).unwrap(), Lattice { n: 1024, k_re: 256, dv: step });
        // Offsets in millionths of a step, written out rather than taken from the constant, so
        // that the test pins the tolerance instead of following it.
        let half = 0.5e-6 * step;
        let snapped = WhiteNoiseIf { v_th: half, ..base }.lattice(step).unwrap();
        assert_eq!((snapped.n, snapped.k_re), (1024, 256));
        assert!(snapped.dv > step && (snapped.dv - step) < 1e-9 * step);
        let off = WhiteNoiseIf { v_th: 4.0 * half, ..base }.lattice(step).unwrap_err();
        assert!(matches!(off, DensityError::OffLattice { what: "v_th", .. }), "{off:?}");
        // Just below a lattice point rounds up to it, for the threshold and for the reset.
        assert_eq!(WhiteNoiseIf { v_th: -half, ..base }.lattice(step).unwrap().n, 1024);
        assert_eq!(WhiteNoiseIf { v_re: -0.75 - half, ..base }.lattice(step).unwrap().k_re, 256);
        // The reset is placed on the recomputed step, not the one asked for: with the threshold
        // snapped 0.9 tolerances up, a reset 0.9 tolerances off its point on the new lattice is on
        // it, and would be 1.125 tolerances off on the old one.
        let up = WhiteNoiseIf { v_th: 0.9e-6 * step, ..base };
        let new_step = (up.v_th - up.v_lb) / 1024.0;
        let v_re = up.v_lb + (256.0 + 0.9e-6) * new_step;
        assert_eq!(WhiteNoiseIf { v_re, ..up }.lattice(step).unwrap().k_re, 256);
        assert!(WhiteNoiseIf { v_re: -0.75 + half, ..base }.lattice(step).is_ok());
        let off = WhiteNoiseIf { v_re: -0.75 + 4.0 * half, ..base }.lattice(step).unwrap_err();
        assert!(matches!(off, DensityError::OffLattice { what: "v_re", .. }), "{off:?}");
        for v_re in [-1.0 + half, -half] {
            let e = WhiteNoiseIf { v_re, ..base }.lattice(step).unwrap_err();
            assert!(matches!(e, DensityError::OffLattice { what: "v_re", .. }), "{v_re}: {e:?}");
        }
        // Exactly MAX_CELLS cells is allowed; one more is not. (The lattice is only laid out, never
        // allocated, so the test costs nothing.)
        let tiny = 2f64.powi(-30);
        let wide = WhiteNoiseIf { v_lb: -1e7 * tiny, v_re: -5e6 * tiny, e0: -2.5e6 * tiny, ..base };
        assert_eq!(wide.lattice(tiny).unwrap().n, 10_000_000);
        let wider = WhiteNoiseIf { v_lb: -(1e7 + 2.0) * tiny, ..wide };
        assert_eq!(wider.lattice(tiny), Err(DensityError::TooManyCells { cells: 1e7 + 2.0 }));
    }

    /// The LIF crosses between the two conventions exactly: [`WhiteNoiseIf::siegert`] multiplies
    /// Richardson's `σ` by `√2` and [`WhiteNoiseIf::from_siegert`] divides it back, and Brunel's
    /// `σ/√2` is [`SiegertInput::free_membrane_sd`] — the free membrane's standard deviation, which is
    /// what Richardson's `σ` is. `ψ` of the LIF is zero.
    #[test]
    fn the_two_noise_conventions_differ_by_root_two() {
        let n = WhiteNoiseIf { t_ref: 3e-3, ..WhiteNoiseIf::FIG1_CASE_II };
        let s = n.siegert().unwrap();
        assert_eq!((s.tau_m, s.t_ref, s.v_th, s.v_reset, s.mu), (n.tau, n.t_ref, n.v_th, n.v_re, n.e0));
        assert!((s.sigma / (SQRT_2 * n.sigma) - 1.0).abs() < 1e-16);
        assert!((s.free_membrane_sd() / n.sigma - 1.0).abs() < 1e-15);
        let back = WhiteNoiseIf::from_siegert(&s, n.v_lb);
        assert!((back.sigma / n.sigma - 1.0).abs() < 1e-15);
        assert_eq!(WhiteNoiseIf { sigma: n.sigma, ..back }, n);
        assert_eq!(n.psi(0.0), 0.0);
        assert_eq!(n.psi(1.0), 0.0);
    }

    /// Newton's method from far away: a guess at 1 MHz reaches a root in ten steps, which is what
    /// [`NEWTON_STEPS`] has to allow. A subnormal guess leaves the finite plane on its first step,
    /// and that is refused, not reported as a root; a guess at `10⁻²⁰⁰` per second creeps towards the
    /// degenerate `λ = 0` and runs out of steps.
    #[test]
    fn the_eigenvalue_search_reports_its_steps_and_refuses_to_diverge() {
        let net = brunel_net(1000.0, 0.1e-3, 6.0, 4.0);
        let nd = NetworkDensity { net, v_lb: -40e-3, dv: 10e-6 };
        let far = nd.eigenvalue(55.8, Complex::new(0.0, 1e6)).unwrap();
        assert_eq!(far.steps, 10);
        assert!(nd.characteristic(55.8, far.lambda).unwrap().norm() < 1e-12);
        // A subnormal guess: the difference step 10⁻⁷|λ| is zero, the slope 0/0, the step NaN.
        let lost = nd.eigenvalue(55.8, Complex::new(0.0, 1e-320)).unwrap_err();
        assert_eq!(lost.to_string(), "the eigenvalue search did not settle in 1 steps; it stopped at lambda = (NaN, NaN) per second");
        // A guess at 10⁻²⁰⁰ creeps towards λ = 0, where the response is 0/0, and runs out of steps.
        let slow = nd.eigenvalue(55.8, Complex::new(1e-200, 0.0)).unwrap_err();
        let DensityError::NoConvergence { steps, re, im } = slow else { panic!("{slow:?}") };
        assert!(steps == 50 && re > 0.0 && re < 1e-10 && im == 0.0, "{slow:?}");
        assert!(slow.to_string().starts_with("the eigenvalue search did not settle in 50 steps; it stopped at lambda = (0.0000000000"));
        // And a guess from which the characteristic itself is out of range is refused as that.
        let far_off = nd.eigenvalue(55.8, Complex::new(1e-3, 1e-3)).unwrap_err();
        assert_eq!(far_off.to_string(), "the response is not representable in f64 at these parameters");
    }

    /// Every refusal, by its rendered message: a default is not a fallback.
    #[test]
    fn every_refusal_names_what_it_refused() {
        let msg = |e: DensityError| e.to_string();
        let ok = WhiteNoiseIf::FIG2_CASE_I;
        let lif = WhiteNoiseIf::FIG1_CASE_I;
        let check = |n: WhiteNoiseIf| msg(n.check().unwrap_err());
        assert_eq!(check(WhiteNoiseIf { tau: 0.0, ..ok }), "tau = 0 must be finite and positive");
        assert_eq!(check(WhiteNoiseIf { sigma: f64::NAN, ..ok }), "sigma = NaN must be finite and positive");
        assert_eq!(check(WhiteNoiseIf { e0: f64::INFINITY, ..ok }), "e0 = inf is not finite");
        assert_eq!(check(WhiteNoiseIf { v_th: f64::NAN, ..ok }), "v_th = NaN is not finite");
        assert_eq!(check(WhiteNoiseIf { v_re: f64::NEG_INFINITY, ..ok }), "v_re = -inf is not finite");
        assert_eq!(check(WhiteNoiseIf { v_lb: f64::NAN, ..ok }), "v_lb = NaN is not finite");
        assert_eq!(check(WhiteNoiseIf { t_ref: f64::INFINITY, ..ok }), "t_ref = inf is not finite");
        assert_eq!(check(WhiteNoiseIf { t_ref: -1e-3, ..ok }), "t_ref = -0.001 must not be negative");
        assert!(WhiteNoiseIf { t_ref: 0.0, ..ok }.check().is_ok());
        assert_eq!(
            check(WhiteNoiseIf { v_re: -0.1, ..ok }),
            "the lattice needs v_lb < v_re < v_th, and was given -0.1, -0.1, 0"
        );
        assert_eq!(
            check(WhiteNoiseIf { v_re: 0.0, ..ok }),
            "the lattice needs v_lb < v_re < v_th, and was given -0.1, 0, 0"
        );
        let eif = |v_t: f64, delta_t: f64| WhiteNoiseIf { model: Model::Eif { v_t, delta_t }, ..ok };
        assert_eq!(check(eif(f64::NAN, 3e-3)), "v_t = NaN is not finite");
        assert_eq!(check(eif(-53e-3, 0.0)), "delta_t = 0 must be finite and positive");
        assert_eq!(check(eif(-53e-3, 5e-5)), "psi(v_th) overflowed to inf");
        assert!(eif(-53e-3, 5e-5).psi(-0.06).is_finite() && eif(-53e-3, 1e-4).check().is_ok());

        let lattice = |n: WhiteNoiseIf, dv: f64| msg(n.lattice(dv).unwrap_err());
        assert_eq!(lattice(ok, -1e-5), "dv = -0.00001 must be finite and positive");
        assert_eq!(lattice(ok, f64::INFINITY), "dv = inf must be finite and positive");
        assert_eq!(lattice(WhiteNoiseIf { tau: -1.0, ..ok }, 1e-5), "tau = -1 must be finite and positive");
        assert_eq!(lattice(ok, 1e-9), "100000000 lattice cells is more than the 10000000 this module will allocate");
        assert_eq!(lattice(ok, 3e-5), "v_th is 3333.3333333333335 steps above v_lb, which is not a lattice point");
        let binary = WhiteNoiseIf { model: Model::Lif, tau: 1.0, e0: -0.5, sigma: 0.1, v_th: 0.0, v_re: -0.75 + 2f64.powi(-11), v_lb: -1.0, t_ref: 0.0 };
        assert_eq!(lattice(binary, 2f64.powi(-10)), "v_re is 256.5 steps above v_lb, which is not a lattice point");

        let far = WhiteNoiseIf { e0: -70e-3, sigma: 0.5e-3, ..lif };
        assert_eq!(msg(far.stationary(10e-6).unwrap_err()), "the unnormalised density is not representable in f64 at these parameters");
        // ψ ≈ 10³⁰¹ on every cell above the reset and σ = 1 µV: ΔG = −∞ there, B = 0, and the
        // density underflows to exactly zero.
        let flat = WhiteNoiseIf {
            model: Model::Eif { v_t: -0.7, delta_t: 1e-3 },
            tau: 1.0,
            e0: 0.0,
            sigma: 1e-6,
            v_th: 8e-3,
            v_re: 0.0,
            v_lb: -8e-3,
            t_ref: 0.0,
        };
        assert!(flat.psi(flat.v_th).is_finite() && flat.psi(1e-3) > 1e300);
        assert_eq!(msg(flat.stationary(1e-3).unwrap_err()), "the unnormalised density is not representable in f64 at these parameters");
        assert_eq!(msg(ok.siegert().unwrap_err()), "the Siegert rate of a non-leaky model is not defined for this model");
        assert_eq!(msg(WhiteNoiseIf { sigma: 0.0, ..lif }.siegert().unwrap_err()), "sigma = 0 must be finite and positive");

        let st = lif.stationary(10e-6).unwrap();
        assert_eq!(msg(st.response(Modulation::Mean, Complex::new(0.0, 0.0)).unwrap_err()), "the response at lambda = 0 is 0/0; ask for a small nonzero frequency");
        assert_eq!(msg(st.response(Modulation::Mean, Complex::new(f64::NAN, 1.0)).unwrap_err()), "lambda.re = NaN is not finite");
        assert_eq!(msg(st.response(Modulation::Mean, Complex::new(0.0, f64::INFINITY)).unwrap_err()), "lambda.im = inf is not finite");
        assert!(st.response(Modulation::Mean, Complex::new(1.0, 0.0)).is_ok());
        assert!(st.response(Modulation::Mean, Complex::new(0.0, 1.0)).is_ok());
        for m in [Modulation::SpikeThreshold, Modulation::SpikeWidth] {
            assert_eq!(msg(st.response(m, at(10.0)).unwrap_err()), "a spike-current modulation of the LIF is not defined for this model");
        }

        let input = brunel_input();
        let bad = SiegertInput { v_reset: 30e-3, ..input };
        assert_eq!(msg(brunel_density(&bad, 0.0).unwrap_err()), "meanfield: v_th = 0.02 must exceed v_reset = 0.03");
        assert_eq!(msg(brunel_slope(&SiegertInput { sigma: 0.0, ..input }, 0.0).unwrap_err()), "sigma = 0 must be finite and positive");
        assert_eq!(msg(brunel_density(&input, f64::NAN).unwrap_err()), "v = NaN is not finite");
        // y_θ = 30: ν₀ is 1.2 × 10⁻³⁸⁸ Hz and the integral of e^{u²} is past f64::MAX, though the
        // density at 16 mV is 2.1 × 10⁻⁴⁰ per volt. The route fails from y_θ = √ln(f64::MAX/2), where
        // 2e^{y_θ²} overflows. A subnormal σ: every reduced voltage is infinite. A reset at −10³⁰⁸ V:
        // the reduced reset is (at −10³⁰⁰ V it is merely −2 × 10³⁰², and fine).
        let unevaluable = |w: &str| format!("{w} cannot be evaluated in f64 at these parameters: nu0, e^(-y^2) or the integral of e^(u^2) leaves the finite range");
        let narrow = |y: f64| SiegertInput { mu: 20e-3 - y * 1e-4, sigma: 1e-4, ..input };
        assert_eq!(msg(brunel_density(&narrow(30.0), 16e-3).unwrap_err()), unevaluable("brunel_density"));
        assert_eq!(msg(brunel_slope(&narrow(30.0), 16e-3).unwrap_err()), unevaluable("brunel_slope"));
        let edge = (f64::MAX / 2.0).ln().sqrt();
        assert!((edge - 26.6287).abs() < 1e-4);
        assert!(brunel_density(&narrow(edge - 1e-3), 19.9e-3).is_ok() && brunel_slope(&narrow(edge - 1e-3), 19.9e-3).is_ok());
        assert!(brunel_density(&narrow(edge + 1e-3), 19.9e-3).is_err() && brunel_slope(&narrow(edge + 1e-3), 19.9e-3).is_err());
        let subnormal = SiegertInput { sigma: 5e-324, ..input };
        assert_eq!(msg(brunel_density(&subnormal, 15e-3).unwrap_err()), unevaluable("brunel_density"));
        let deep = SiegertInput { v_reset: -1e308, ..input };
        assert_eq!(msg(brunel_slope(&deep, 15e-3).unwrap_err()), unevaluable("brunel_slope"));

        let net = brunel_net(1000.0, 0.1e-3, 5.0, 2.0);
        let nd = NetworkDensity { net, v_lb: -40e-3, dv: 10e-6 };
        assert_eq!(msg(nd.neuron(f64::NAN).unwrap_err()), "nu = NaN is not finite");
        assert_eq!(msg(nd.neuron(-1.0).unwrap_err()), "nu = -1 must not be negative");
        assert!(nd.neuron(0.0).is_ok());
        let silent = NetworkDensity { net: BrunelNetwork { j: 0.0, ..net }, ..nd };
        assert_eq!(msg(silent.neuron(1.0).unwrap_err()), "nu_thr = (v_th - v_rest)/(c_exc j tau_m) is not defined for this model");
        let inverted = NetworkDensity { net: BrunelNetwork { neuron: Lif { v_reset: 25e-3, ..brunel_cell() }, ..net }, ..nd };
        assert_eq!(msg(inverted.neuron(1.0).unwrap_err()), "meanfield: v_th = 0.02 must exceed v_reset = 0.025");
        let no_dead_time = NetworkDensity { net: BrunelNetwork { neuron: Lif { t_ref: 0.0, ..brunel_cell() }, ..net }, ..nd };
        assert_eq!(msg(no_dead_time.rate().unwrap_err()), "t_ref = 0 must be finite and positive");
        assert_eq!(msg(nd.eigenvalue(38.0, Complex::new(0.0, 0.0)).unwrap_err()), "the response at lambda = 0 is 0/0; ask for a small nonzero frequency");
        assert_eq!(msg(nd.characteristic(-2.0, at(10.0)).unwrap_err()), "nu = -2 must not be negative");
    }

    /// Brunel's high-activity states, where excitation dominates: the self-consistent rate near
    /// saturation, and his eq. (23), its leading order in `1/C_E`.
    ///
    /// Fig. 8's point A, `g = 3`, `ν_ext/ν_thr = 2`, and two more excitable networks. The lattice
    /// rate agrees with the Siegert one to `1.3 × 10⁻⁵` at 10 µV and halves with the step; both are
    /// above half the `1/τ_rp` bracket, which the bisection must therefore search in full.
    /// Eq. (23), `ν₀ = (1/τ_rp)[1 − (θ − V_r)/(C_E J(1 − gγ))]`, gives 300, 400 and 433.3 Hz; the
    /// self-consistent rates are 327.0, 405.8 and 435.8 Hz — 9.0%, 1.5% and 0.57% apart, closing
    /// as the excitation, and with it the mean drive the expansion needs to be large, grows.
    #[test]
    fn the_high_activity_rate_approaches_eq_23() {
        let mut last = f64::INFINITY;
        for (g, gap) in [(3.0, 0.090), (2.0, 0.015), (1.0, 0.0057)] {
            let net = brunel_net(1000.0, 0.1e-3, g, 2.0);
            let siegert = net.self_consistent_rate().unwrap();
            let coarse = NetworkDensity { net, v_lb: -40e-3, dv: 10e-6 }.rate().unwrap() / siegert - 1.0;
            let fine = NetworkDensity { net, v_lb: -40e-3, dv: 5e-6 }.rate().unwrap() / siegert - 1.0;
            assert!(coarse.abs() < 3e-5 && (coarse / fine - 2.0).abs() < 0.05, "{coarse} {fine}");
            assert!(siegert > 250.0);
            let eq23 = (1.0 / 2e-3) * (1.0 - 10e-3 / (1000.0 * 0.1e-3 * (1.0 - g / 4.0)));
            let off = siegert / eq23 - 1.0;
            assert!((off / gap - 1.0).abs() < 0.05 && off < last, "g = {g}: {siegert} against {eq23}");
            last = off;
        }
    }

    /// The division that ends the recursion, `−ĵ_α⁽⁰⁾/ĵ_r⁽⁰⁾`, written the textbook way as
    /// `z w̄/|w|²`, overflows long before the fluxes do: on Fig. 2's case i at 10 µV, unscaled, the
    /// two fluxes are `10¹⁴¹`–`10¹⁴²` at 5 kHz and `10¹⁶¹`–`10¹⁶²` at 6 kHz, past `√f64::MAX`. The
    /// textbook quotient agrees with Smith's at 5 kHz and is `NaN` at 6 kHz, where Smith's is the
    /// response — the rescaled recursion's, bit for bit.
    #[test]
    fn the_last_division_is_smiths() {
        let textbook = |z: Complex, w: Complex| {
            let m = w.re * w.re + w.im * w.im;
            Complex::new((z.re * w.re + z.im * w.im) / m, (z.im * w.re - z.re * w.im) / m)
        };
        let st = WhiteNoiseIf::FIG2_CASE_I.stationary(WhiteNoiseIf::FIG_STEP).unwrap();
        let drive = |k| Ok(st.drive_at(Modulation::Mean, k).unwrap());
        let ends = |f: f64| st.recur(at(f), f64::INFINITY, drive).unwrap();
        let (low, high) = (ends(5e3), ends(6e3));
        assert!(low.jr.norm() < 1e142 && low.ja.norm() < 1e143, "{low:?}");
        assert!(high.jr.norm() > f64::MAX.sqrt() && high.ja.norm() > f64::MAX.sqrt(), "{high:?}");
        assert_eq!((low.shift_a, low.shift_r, high.shift_a, high.shift_r), (0, 0, 0, 0));
        assert!(rel(textbook(low.ja, low.jr), low.ja / low.jr) < 1e-15);
        assert!(textbook(high.ja, high.jr).re.is_nan());
        assert_eq!(-(high.ja / high.jr), st.response(Modulation::Mean, at(6e3)).unwrap());
    }
}
