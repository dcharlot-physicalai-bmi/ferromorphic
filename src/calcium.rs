//! Graupner and Brunel's calcium-based plasticity: a bistable synaptic efficacy pushed up or down
//! while a calcium trace sits above one of two thresholds, the closed-form probabilities that a
//! protocol flips it, and a stochastic simulation — checked against the authors' own code.
//!
//! # The model
//!
//! Graupner and Brunel, *Calcium-based plasticity model explains sensitivity of synaptic changes to
//! spike pattern, rate, and dendritic location*, PNAS 109:3991–3996, 2012, eq. 1; its Correction,
//! PNAS 109:21551, 2012 (doi:10.1073/pnas.1220044110); and its SI Appendix, "Supporting Information
//! Corrected November 28, 2012". The efficacy `ρ` obeys
//!
//! ```text
//! τ dρ/dt = −ρ(1 − ρ)(ρ* − ρ) + γ_p (1 − ρ) Θ[c − θ_p] − γ_d ρ Θ[c − θ_d] + Noise(t)     (1)
//! Noise(t) = σ √τ √(Θ[c − θ_d] + Θ[c − θ_p]) η(t)                         (the Correction)
//! dc/dt = −c/τ_Ca + C_pre Σ_i δ(t − t_i − D) + C_post Σ_j δ(t − t_j)      (SI eq. 1)
//! ```
//!
//! with `Θ[x] = 0` for `x < 0` and `1` for `x ≥ 0` (p. 3992) and `η` unit Gaussian white noise. The
//! paper first printed the noise as `σ√τ Θ[c − min(θ_d, θ_p)] η(t)`; the Correction replaced it.
//! The cubic makes `ρ = 0` and `ρ = 1` stable and `ρ*` the boundary between their basins; calcium
//! above `θ_d` drives `ρ` down at rate `γ_d`, above `θ_p` up at rate `γ_p`; a presynaptic spike adds
//! `C_pre` to the calcium after a delay `D`, a postsynaptic one adds `C_post` at once, and both decay
//! with `τ_Ca`. [`Synapse`] holds the thirteen parameters, and its constants are the paper's sets:
//! Table S1's six STDP curves ([`Synapse::DP`] and siblings), Table S2's three fits to data
//! ([`Synapse::HIPPOCAMPAL_SLICES`], [`Synapse::HIPPOCAMPAL_CULTURES`], [`Synapse::CORTICAL_SLICES`]),
//! and the two complete columns of Table S3 ([`Synapse::FIG_S1_DPD_PRIME`], [`Synapse::FIG_S1_DP`]).
//! Table S1's BCM column and Table S3's heterogeneous one are not constants, because their calcium
//! amplitudes are printed as "varied" and "drawn".
//!
//! # The analysis the SI derives, and this module computes
//!
//! - The calcium trace itself, [`calcium`], which the tests hold to SI eqs. (19), (20) corrected,
//!   and the feet and tops of transients in single triplets (22)–(26), (28)–(32) and in pairs at a
//!   frequency (34)–(37), (40)–(43).
//! - The fraction of time the calcium spends above a threshold, SI eq. (6), for any periodic
//!   protocol: [`fraction_above`] sums the steady-state periodic trace as a geometric series and
//!   takes each inter-transient segment's crossing, `τ_Ca ln(c/θ)`, in closed form. It is the SI's
//!   case analyses — single pairs (21), triplets (27) and (33), pairs at a frequency (39) and (44) —
//!   for every pattern at once; [`Synapse::pair_fractions`] and [`Synapse::burst_fractions`] build
//!   the protocols the figures use.
//! - The Ornstein–Uhlenbeck reduction, SI eq. (7): `Γ_x = γ_x α_x`, [`Synapse::rho_bar`] (9),
//!   [`Synapse::sigma_rho_sq`] (10), [`Synapse::tau_eff`] (11), and the transition probabilities
//!   [`Synapse::up`] (13) and [`Synapse::down`] (15).
//! - The change in synaptic strength, as the main text's Methods print it (p. 3996) and the authors'
//!   `changeInSynapticStrength` computes it:
//!   `([(1 − 𝒰)β + 𝒟(1 − β)] + b[𝒰β + (1 − 𝒟)(1 − β)])/(β + (1 − β)b)`, [`Synapse::change`].
//! - The ratio `γ_p/γ_d` that makes single transients cancel, SI eqs. (16)–(18): [`Synapse::balance`].
//! - For an arbitrary spike train, the exact time above each threshold, [`Synapse::time_above`], and
//!   an Euler–Maruyama simulation of eq. 1 with either noise term, [`Synapse::simulate`].
//!
//! What is exact: the calcium trace is advanced in closed form between transients and every
//! threshold crossing is found as `τ_Ca ln(c/θ)`, so `Θ` switches at the true instants and the only
//! discretisation error in a simulation is that of Euler–Maruyama inside intervals where the drift
//! is a polynomial and the noise amplitude a constant. The OU reduction is written with
//! `φ(x) = (1 − e⁻ˣ)/x`, evaluated as `−expm1(−x)/x`, so it stays finite, and continuous, where
//! no calcium crosses either threshold and `Γ_p + Γ_d = 0`.
//!
//! # The reference is the authors' code
//!
//! The authors' repository, github.com/mgraupe/CalciumBasedPlasticityModel at commit `56503b0`
//! (2026-01-16), holds `synapticChange.py` (the parameter sets and the OU formulas),
//! `timeAboveThreshold/timeAboveThreshold.py` (the fractions, by case analysis of the SI's regions),
//! the scripts that draw Figs. 2, 3 and 4B, and the C++ that ran the paper's simulations with its
//! output. `tools/calcium_reference.py` fetches the Python at that commit, runs it unmodified, and
//! prints the tables the tests hold. Against them this module reproduces:
//!
//! - every value of Tables S1 and S2 as `choseParameterSet` holds it, bit for bit (the tables print
//!   `τ_Ca` and `D` in ms; the code, like this module, holds seconds);
//! - Fig. 2A for the six Table S1 sets on the script's own 2001-point grid, Fig. 3's pair and
//!   pre-spike–post-burst curves, and Fig. 4B at five frequencies: every sampled point and extreme
//!   within `9.4 × 10⁻¹⁵`, and every grid's sum within `1.3 × 10⁻¹⁴` of the authors' `math.fsum`,
//!   relatively; Fig. 4A's two offsets at eight frequencies within `1.2 × 10⁻¹⁴`;
//! - Fig. 4B's "potentiation only above 29 Hz for all Δt": the frequency where the lowest change on
//!   the grid reaches 1 is 29.1139 Hz in both codes, bisected to the same last step;
//! - `spikePairFrequency` and `preSpikePostPair`'s fractions of time above threshold within
//!   `3.7 × 10⁻¹⁶`, `changeInSynapticStrength`'s `ρ̄`, `σ_ρ²`, `τ_eff`, `𝒰`, `𝒟`, means and change
//!   within `4.7 × 10⁻¹⁶`, and `eventBasedIntegration` on a 400-event irregular train within
//!   `5.3 × 10⁻¹⁵` s.
//!
//! Reading that code, the C++ beside it and the SI turned up the following.
//!
//! ⚠ **The simulation output committed with the code fits the corrected noise, if its runs shared
//! one `σ`; the committed code computes the printed one.** The Correction does not say which term
//! the simulations of Fig. 2 (cyan) used. The committed `numericalSimulation/motif.cpp` computes
//! `sigma*sqrt(beta/dephos)*sqrt(tau_rho)*eta`, where `beta = dephos` when `Ca ≥ Ct_dephos` and 0
//! otherwise: that is `σ√τ √Θ[c − θ_d] η`, the PRINTED term wherever `θ_d ≤ θ_p` (every set has
//! it). The committed x86-64 binary, identical in every output directory, agrees: its
//! `pre_post_spikes::functions` calls `sqrt` on the one ratio `beta/dephos`, and the potentiation
//! rate appears only in the drift. None of the three alternatives left commented out beside that
//! line is the corrected term either: `σ(β/dephos)` is the printed term again, since `Θ² = Θ`;
//! `σ(β/dephos + α/phos)` has the factor 2 above both thresholds where the corrected term has `√2`;
//! and the third is an unconditional `σ`. But the output committed beside the code,
//! `output/{DP,P}_curve/final_camkII_state.dat` with 1000 runs per cell, fits the corrected term:
//! simulated here 1000 times per cell, the corrected term gives `χ² = 142.9` over the 164 cells,
//! the printed one 444.1 and the commented-out sum 526.4, against `164 ± 18` by chance. That
//! verdict rests on `σ`. On these two curves alone the printed term with `σ × 1.22` (148.7) and the
//! sum with `σ × 0.75` (128.8) fit as well. What excludes them is the D′ curve, where the three
//! terms coincide: its output fits the files' `σ` (92.1 over 82 cells, against `82 ± 13`) and
//! neither rescaled one (626.8, and 908.0 over the 80 cells not all UP or all DOWN in both
//! samples). So the output favours the corrected term only if the three runs shared `σ`, as their
//! parameter files say (0.4 in each) — files that are not a reliable record, since DPD′'s is DPD's
//! (below). Whichever term made it, the output was not produced by the code and parameter files as
//! committed, and nothing in the repository ties it to the data behind the published Fig. 2.
//! [`Noise`] offers the two terms;
//! `the_committed_simulation_output_fits_the_corrected_noise_if_the_runs_shared_sigma` holds the test.
//!
//! ⚠ **`σ_ρ²` is twice the variance of `ρ`, not its square.** The SI calls `σ_ρ` "the standard
//! deviation of `ρ`", but its pdf (8) is `exp(−(ρ − ρ̄ + …)²/(σ_ρ²(1 − e^{−2t/τ_eff})))` over
//! `√(π σ_ρ² (1 − e^{−2t/τ_eff}))`: a Gaussian of variance `σ_ρ²(1 − e^{−2t/τ_eff})/2`. The OU
//! process (7) has stationary variance `σ²(α_p + α_d)/(2(Γ_p + Γ_d))`, half of (10). Eqs. (13) and
//! (15) are consistent with (8), so the probabilities are right; the name is not.
//! [`Synapse::variance`] is the variance, and the simulation's spread matches it.
//!
//! ⚠ **SI eq. (20) puts `e^{Δt}` on the wrong amplitude.** For a pre–post pair it prints the trace
//! after the postsynaptic spike as `e^{−t}(C_pre e^{Δt} + C_post)`; the sum of the two transients is
//! `e^{−t}(C_pre + C_post e^{Δt})`, the form (19) has for the other order. As printed the trace would
//! jump at `Δt` by `C_pre(1 − e^{−Δt}) + C_post e^{−Δt}` instead of `C_post`. Nothing downstream uses
//! the printed form: the expressions of (21) and (44) are right for pre–post pairs.
//!
//! ⚠ **SI §3.6 misplaces a parenthesis in the change in strength.** It prints
//! `[(1 − 𝒰)β + 𝒟(1 − β)] + (b[𝒰β + (1 − 𝒟)(1 − β)])/(β + [1 − β]b)` (SI p. 25), which divides
//! only the UP term. Its own preceding sentence, `w₀[…] + w₁[…]` over `βw₀ + (1 − β)w₁`, the main
//! text's Methods and `changeInSynapticStrength` all divide both. As printed, a protocol that switches
//! no synapse would report `β + b(1 − β)/(β + (1 − β)b) = 4/3` for Table S1's `β = ½`, `b = 5`,
//! instead of 1 (`si_3_6_as_printed_divides_only_the_up_term`).
//!
//! ⚠ **SI eq. (39) prints two of its frequency intervals with their ends reversed.** For post–pre
//! pairs its regions II and IV require `f ∈ [−ln(1 − X/(θ + C_pre))⁻¹, −ln(1 − X/θ)⁻¹]`, with
//! `X = C_post e^{Δt} + C_pre`. The left end is where the pre transient's foot `C` reaches `θ`, the
//! right end where its top `E` does; since `θ + C_pre > θ` the left end is the LARGER wherever both
//! ends exist (`X < θ`), so the interval as printed is empty. Read as
//! `[−ln(1 − X/θ)⁻¹, −ln(1 − X/(θ + C_pre))⁻¹]` the five regions tile the plane, and (39)'s five
//! expressions are then the periodic trace's. Eq. (44), for pre–post pairs, has its interval ends in
//! order, though its intervals for II and IV lack the comma between their ends and III and V carry a
//! stray `]`.
//!
//! ⚠ **The authors' code returns `NaN` wherever no calcium crosses either threshold.** There
//! `Γ_p + Γ_d = 0` and `rhoBar = GammaP/(GammaP + GammaD)` is `0/0`. On Fig. 2's 2001-point grid that
//! is 1122 points of the DPD curve and 1838 of the D curve, and the script's plot leaves them out:
//! the committed `outputFigures/Graupner2012PNAS_Fig2.png` draws the analytic D curve only over
//! `|Δt| ≤ τ_Ca ln(C/(θ_d − C)) = 8.1` ms of its ±15 ms axis. Without drive and without noise `ρ`
//! stays where it started, so the change is exactly 1; that is what [`Synapse::change`] returns, and
//! the tests check it is 1 at exactly the points where the reference is `NaN`.
//!
//! ⚠ **The printed delays put the LTD-to-LTP transition near `Δt = 0`, not at it.** Table S1 says
//! `D` "is adjusted such that the transition from depression to potentiation occurs at `Δt = 0` ms".
//! With `β = ρ* = ½` the change is 1 exactly where `γ_p α_p = γ_d α_d`, so that `D` solves one
//! equation. For DP it has a closed form: at `D = τ_Ca ln(C_post/θ_d)` the post transient has decayed
//! to `θ_d = C_pre` when the pre transient lands, the sum restarts at `C_post`, and the two halves
//! balance with the ratio (17) — `D = 13.863` ms. Table S1 prints 13.7 and Table S3 prints 13.8 for
//! the same calcium and the same `γ_p/γ_d`, and neither is 13.863 rounded. The other roots are
//! 4.527 ms for DPD (printed 4.6), 2.667 ms for DPD′ (printed 2.2) and 4.400 ms for Table S3's DPD′
//! (printed 4.3), so the printed transitions sit at −0.16, −0.06, +0.07, −0.47 and −0.10 ms.
//!
//! ⚠ **P's printed `γ_p = 257.447` is one unit in the last place above eq. (18)'s ratio times `γ_d`,
//! rounded.**
//! P's amplitudes, `C_pre = C_post = 2 > θ_p = 1.3`, are (18)'s fourth case at its edge
//! `C_pre = C_post` (SI p. 16), whose ratio
//! `(ln(C_post/θ_d) + ln(C_pre/θ_d))/(ln(C_post/θ_p) + ln(C_pre/θ_p))` is here (17)'s `ln 2/ln(2/1.3)`.
//! `160 ln 2/ln(2/1.3) = 257.44649`, which rounds to 257.446. The value is consistent with that
//! product rounded twice: the committed P parameter file holds `phos = 5.14893`, the product over 50
//! to five decimals, and `50 × 5.14893 = 257.4465` rounds half up to the printed value. DP's `321.808`
//! and Table S3's `241.356` are the products rounded once.
//!
//! ⚠ **`spikePairFrequency` folds a long offset to the wrong side.** For `|Δt − D| > 1/f` it sets
//! `deltaT = −(|deltaT| − 1/f)`, which maps `+1.2/f` to `−0.2/f`; in a periodic train `+1.2/f` is
//! `+0.2/f`. At 5 Hz on the DP amplitudes it returns `α_d = 0.1062` for an offset of 0.23 s, the
//! value at −0.03 s, where 0.03 s gives 0.0799. No figure script reaches it (every offset there is
//! within one period); [`fraction_above`] reduces every offset modulo the period.
//!
//! ⚠ **The DPD′ simulation's recorded parameters are DPD's.** `output/DPDprime_curve/camkmotifscan.par`
//! is byte-for-byte `output/DPD_curve/camkmotifscan.par` (`C_pre = C_post = 0.9`, `θ_p = 1.3`,
//! delay 4.6 ms), while its `final_camkII_state.dat` is the DPD′ curve: its change in strength is
//! within 0.0087 root-mean-square of the analytic DPD′ curve and 0.077 of the DPD one. The other five
//! parameter files hold Table S1 with eq. 1 divided through by 50 (`τ = 3` s, `γ/50`, `σ/√50`, the
//! cubic multiplied by 0.02), which is the same dynamics to the printed digits — D's printed
//! `σ = 5.6568` is twice the rounded 2.8284, where its file's `0.8√50 = 5.65685`.
//!
//! # The simulation against the analysis
//!
//! Eq. (13) is exact for the OU process (7), not for eq. 1: (7) drops the cubic and spreads over the
//! period a drive that arrives in pulses. The tests carry the Gaussian of eq. 1 without its cubic
//! across the pulses exactly and measure both approximations on 60 DP pairs at 1 Hz. The averaging
//! costs (13) 0.006 in `𝒰` — 0.6440 against 0.6381 at `Δt = +10` ms, 0.2444 against 0.2385 at
//! `−20` ms — and 0.002 to 0.003 in the mean; the cubic then moves the noise-free end of the protocol
//! by `+5 × 10⁻⁴` and `−4 × 10⁻³`. Scaling `γ_p`, `γ_d` and `τ` by 100 and `σ` by 10 leaves both
//! Gaussians unchanged and shrinks the cubic a hundredfold, and there the simulation matches the
//! pulsed Gaussian within sampling error. Neither gap is visible at the 1000 runs the paper used,
//! where a standard error of `𝒰` is 0.015.
//!
//! # Not here
//!
//! The SI's nonlinear calcium model (§3.1.2: rise times, and an NMDA term that makes summation
//! supralinear), the amplitude distribution of Poisson trains (§3.5, eqs. 45–47), the BCM-like
//! sliding threshold of Fig. S5 and the fits of §3.7 are not implemented.
//!
//! # Units
//!
//! Seconds for every time (`τ_Ca`, `τ`, `D`, spike times, `dt`), hertz for frequencies. Calcium is
//! dimensionless with its resting value at zero, as in the SI. The fractions `α` are dimensionless.

use core::fmt;

use crate::device::normal;
use crate::rng::Rng;
use crate::surrogate::erf;

/// The most Euler–Maruyama steps [`Synapse::simulate`] will take, counted as `t_end/dt`.
///
/// A run of 60 s at a 1 µs step is 6 × 10⁷. A step small enough to exceed this is refused by name
/// rather than left to spin: at a subnormal `dt` the count of steps is not representable at all.
pub const MAX_STEPS: f64 = 1e8;

/// Why a calcium-plasticity question could not be answered.
#[derive(Debug, Clone, PartialEq)]
pub enum CalciumError {
    /// A quantity that must be finite and positive was not.
    NotPositive {
        /// Which quantity.
        what: &'static str,
        /// Its value.
        value: f64,
    },
    /// A quantity that must be finite and non-negative was not.
    Negative {
        /// Which quantity.
        what: &'static str,
        /// Its value.
        value: f64,
    },
    /// A quantity that must be finite was not.
    NonFinite {
        /// Which quantity.
        what: &'static str,
        /// Its value.
        value: f64,
    },
    /// A quantity outside the interval it must lie in.
    OutOfRange {
        /// Which quantity.
        what: &'static str,
        /// Its value.
        value: f64,
        /// The interval, as text: `"[0, 1]"` or `"(0, 1)"`.
        range: &'static str,
    },
    /// A simulation that would take more than [`MAX_STEPS`] steps.
    TooManySteps {
        /// The run's length, seconds.
        t_end: f64,
        /// The step, seconds.
        dt: f64,
    },
}

impl fmt::Display for CalciumError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotPositive { what, value } => write!(f, "{what} = {value} must be finite and positive"),
            Self::Negative { what, value } => write!(f, "{what} = {value} must be finite and non-negative"),
            Self::NonFinite { what, value } => write!(f, "{what} = {value} is not finite"),
            Self::OutOfRange { what, value, range } => write!(f, "{what} = {value} must lie in {range}"),
            Self::TooManySteps { t_end, dt } => {
                write!(f, "t_end = {t_end} s at dt = {dt} s is more than {MAX_STEPS} Euler-Maruyama steps")
            }
        }
    }
}

impl std::error::Error for CalciumError {}

fn finite(what: &'static str, value: f64) -> Result<f64, CalciumError> {
    if value.is_finite() { Ok(value) } else { Err(CalciumError::NonFinite { what, value }) }
}

fn positive(what: &'static str, value: f64) -> Result<f64, CalciumError> {
    if value.is_finite() && value > 0.0 { Ok(value) } else { Err(CalciumError::NotPositive { what, value }) }
}

fn non_negative(what: &'static str, value: f64) -> Result<f64, CalciumError> {
    if value.is_finite() && value >= 0.0 { Ok(value) } else { Err(CalciumError::Negative { what, value }) }
}

fn unit_closed(what: &'static str, value: f64) -> Result<f64, CalciumError> {
    if (0.0..=1.0).contains(&value) { Ok(value) } else { Err(CalciumError::OutOfRange { what, value, range: "[0, 1]" }) }
}

/// `φ(x) = (1 − e⁻ˣ)/x`, with `φ(0) = 1`: the factor that keeps the OU reduction finite as the
/// drive vanishes.
fn phi(x: f64) -> f64 {
    if x == 0.0 { 1.0 } else { -(-x).exp_m1() / x }
}

/// A calcium transient: a jump of `amplitude` at `time`, decaying with `τ_Ca` after it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Transient {
    /// When the calcium jumps, seconds. For a presynaptic spike this is the spike time plus `D`.
    pub time: f64,
    /// By how much, dimensionless: `C_pre` or `C_post`.
    pub amplitude: f64,
}

/// The fractions of time the calcium spends above the two thresholds, SI eq. (6).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Fractions {
    /// `α_d`: above `θ_d`. In `[0, 1]`.
    pub alpha_d: f64,
    /// `α_p`: above `θ_p`. In `[0, 1]`.
    pub alpha_p: f64,
}

/// Which noise term eq. 1 carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Noise {
    /// The Correction's `σ√τ √(Θ[c − θ_d] + Θ[c − θ_p]) η`: variance adds over the two thresholds,
    /// which is what SI eq. (7) and (10) assume.
    Corrected,
    /// The paper as first printed, `σ√τ Θ[c − min(θ_d, θ_p)] η` — and what the committed C++
    /// computes, as `√Θ[c − θ_d]`, for `θ_d ≤ θ_p`.
    Printed,
    /// Tests only: `σ(β/dephos + α/phos)`, the second of the three alternatives left commented out
    /// below the committed noise line of `motif.cpp` (its line 355), which is
    /// `σ√τ (Θ[c − θ_d] + Θ[c − θ_p]) η`.
    #[cfg(test)]
    CommentedSum,
}

/// Whether single, non-interacting transients can cancel, SI eqs. (16)–(18).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Balance {
    /// They cancel exactly when `γ_p/γ_d` equals this.
    Ratio(f64),
    /// No single transient crosses either threshold: any ratio (eq. 18's "arbitrary").
    Any,
    /// Single transients cross `θ_d` but never `θ_p`: no ratio balances them (the D′ curve).
    Impossible,
}

/// The end of a simulated run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Run {
    /// `ρ` at `t_end`.
    pub rho: f64,
    /// Seconds the calcium spent at or above `θ_d`.
    pub above_d: f64,
    /// Seconds the calcium spent at or above `θ_p`.
    pub above_p: f64,
}

/// The fraction of each period a periodic calcium trace spends above `theta`, SI eq. (6), in the
/// periodic steady state.
///
/// Every transient recurs once per `period`; its `time` is reduced modulo the period, so an offset
/// longer than one period is the same protocol as its remainder. Just after the transient at `s_k`
/// the calcium is `c_k = Σ_m a_m e^{−d_km/τ_Ca}/(1 − e^{−T/τ_Ca})`, with `d_km ∈ [0, T)` the time
/// since transient `m`'s last occurrence; it then decays until the next transient, and spends
/// `min(L_k, max(0, τ_Ca ln(c_k/θ)))` of that segment of length `L_k` above `θ`. This is the SI's
/// case analyses (21), (27), (33), (39) and (44) for every pattern at once. A trace that never falls
/// to `θ` gives exactly 1: the segments' lengths sum to the period only to rounding, and a fraction
/// of time is not allowed to exceed it.
///
/// The steady state's gain `1/(1 − e^{−T/τ_Ca})` is refused when it is not finite, which happens
/// once `T/τ_Ca` is below about `5.6 × 10⁻³⁰⁹`: there the periodic trace is not representable, and a
/// train of zero amplitude would be `0 · ∞`. With the gain finite, each top is a sum of finite
/// non-negative terms times it, so finite or `+∞`; its logarithm against a finite positive `θ` is
/// never `NaN`, and the clamp's bounds, 0 and the segment's length, are ordered and never `NaN`.
/// The cap at 1 therefore never meets a `NaN` and cannot turn one into "always above": it absorbs
/// only the amount by which the segments' lengths, whose exact sum is the period, overshoot it in
/// floating point.
///
/// # Errors
///
/// [`CalciumError::NotPositive`] for a `tau_ca`, `period` or `theta` that is not finite and positive;
/// [`CalciumError::NonFinite`] for a transient time that is not finite, and for a period so short
/// against `tau_ca` that the gain is not;
/// [`CalciumError::Negative`] for an amplitude that is negative or not finite.
pub fn fraction_above(tau_ca: f64, period: f64, transients: &[Transient], theta: f64) -> Result<f64, CalciumError> {
    positive("tau_ca", tau_ca)?;
    positive("period", period)?;
    positive("theta", theta)?;
    let mut at = Vec::with_capacity(transients.len());
    for tr in transients {
        let s = finite("transient time", tr.time)?.rem_euclid(period);
        at.push((s, non_negative("transient amplitude", tr.amplitude)?));
    }
    at.sort_by(|a, b| a.0.total_cmp(&b.0));
    let gain = finite("steady-state gain 1/(1 - exp(-period/tau_ca))", 1.0 / -(-period / tau_ca).exp_m1())?;
    let mut above = 0.0;
    for k in 0..at.len() {
        let s = at[k].0;
        let top: f64 = at.iter().map(|&(m, a)| a * (-(s - m).rem_euclid(period) / tau_ca).exp()).sum::<f64>() * gain;
        let end = if k + 1 < at.len() { at[k + 1].0 } else { at[0].0 + period };
        above += (tau_ca * (top / theta).ln()).clamp(0.0, end - s);
    }
    // At 50 Hz on the cortical-slice amplitudes the segments sum to `1 + 2.2e-16` periods.
    Ok((above / period).min(1.0))
}

/// The calcium at time `t`, SI eq. (1) from rest: `Σ C_i e^{−(t − t_i)/τ_Ca}` over the transients at
/// or before `t`. A transient at `t` itself is counted, so at a transient's own time the trace is at
/// its top — the convention under which `Θ[c − θ]` is 1 at `c = θ` (p. 3992) and a crossing lasts
/// `τ_Ca ln(c/θ)`.
///
/// # Errors
///
/// [`CalciumError::NotPositive`] for a `tau_ca` that is not finite and positive;
/// [`CalciumError::NonFinite`] for a `t` or a transient time that is not finite;
/// [`CalciumError::Negative`] for an amplitude that is negative or not finite.
pub fn calcium(tau_ca: f64, transients: &[Transient], t: f64) -> Result<f64, CalciumError> {
    positive("tau_ca", tau_ca)?;
    finite("t", t)?;
    let mut c = 0.0;
    for tr in transients {
        let (at, amplitude) = (finite("transient time", tr.time)?, non_negative("transient amplitude", tr.amplitude)?);
        if at <= t {
            c += amplitude * (-(t - at) / tau_ca).exp();
        }
    }
    Ok(c)
}

/// Graupner and Brunel's synapse: the calcium dynamics, the thresholds, eq. 1's rates and noise,
/// and the two numbers that turn transition probabilities into a change in strength.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Synapse {
    /// `τ_Ca`, the calcium decay time constant, seconds.
    pub tau_ca: f64,
    /// `C_pre`, the jump a presynaptic spike causes, after `D`.
    pub c_pre: f64,
    /// `C_post`, the jump a postsynaptic spike causes.
    pub c_post: f64,
    /// `θ_d`, the depression threshold.
    pub theta_d: f64,
    /// `θ_p`, the potentiation threshold.
    pub theta_p: f64,
    /// `γ_d`, the depression rate (in units of `1/τ`).
    pub gamma_d: f64,
    /// `γ_p`, the potentiation rate (in units of `1/τ`).
    pub gamma_p: f64,
    /// `σ`, the noise amplitude.
    pub sigma: f64,
    /// `τ`, the time constant of eq. 1, seconds.
    pub tau: f64,
    /// `ρ*`, the unstable fixed point between the two basins. In `(0, 1)`.
    pub rho_star: f64,
    /// `D`, the delay of the presynaptic transient, seconds.
    pub delay: f64,
    /// `β`, the fraction of synapses in the DOWN state before a protocol. In `[0, 1]`.
    pub beta: f64,
    /// `b = w₁/w₀`, the strength of the UP state relative to the DOWN state.
    pub b: f64,
}

impl Synapse {
    /// Table S1's DP curve (depression then potentiation, the classical STDP curve; Fig. 2).
    pub const DP: Self = Self {
        tau_ca: 0.020,
        c_pre: 1.0,
        c_post: 2.0,
        theta_d: 1.0,
        theta_p: 1.3,
        gamma_d: 200.0,
        gamma_p: 321.808,
        sigma: 2.8284,
        tau: 150.0,
        rho_star: 0.5,
        delay: 0.0137,
        beta: 0.5,
        b: 5.0,
    };

    /// Table S1's DPD curve.
    pub const DPD: Self = Self {
        tau_ca: 0.020,
        c_pre: 0.9,
        c_post: 0.9,
        theta_d: 1.0,
        theta_p: 1.3,
        gamma_d: 250.0,
        gamma_p: 550.0,
        sigma: 2.8284,
        tau: 150.0,
        rho_star: 0.5,
        delay: 0.0046,
        beta: 0.5,
        b: 5.0,
    };

    /// Table S1's DPD′ curve.
    pub const DPD_PRIME: Self = Self {
        tau_ca: 0.020,
        c_pre: 1.0,
        c_post: 2.0,
        theta_d: 1.0,
        theta_p: 2.5,
        gamma_d: 50.0,
        gamma_p: 600.0,
        sigma: 2.8284,
        tau: 150.0,
        rho_star: 0.5,
        delay: 0.0022,
        beta: 0.5,
        b: 5.0,
    };

    /// Table S1's P curve (potentiation only).
    pub const P: Self = Self {
        tau_ca: 0.020,
        c_pre: 2.0,
        c_post: 2.0,
        theta_d: 1.0,
        theta_p: 1.3,
        gamma_d: 160.0,
        gamma_p: 257.447,
        sigma: 2.8284,
        tau: 150.0,
        rho_star: 0.5,
        delay: 0.0,
        beta: 0.5,
        b: 5.0,
    };

    /// Table S1's D curve (depression only).
    pub const D: Self = Self {
        tau_ca: 0.020,
        c_pre: 0.6,
        c_post: 0.6,
        theta_d: 1.0,
        theta_p: 1.3,
        gamma_d: 500.0,
        gamma_p: 550.0,
        sigma: 5.6568,
        tau: 150.0,
        rho_star: 0.5,
        delay: 0.0,
        beta: 0.5,
        b: 5.0,
    };

    /// Table S1's D′ curve.
    pub const D_PRIME: Self = Self {
        tau_ca: 0.020,
        c_pre: 1.0,
        c_post: 2.0,
        theta_d: 1.0,
        theta_p: 3.5,
        gamma_d: 60.0,
        gamma_p: 600.0,
        sigma: 2.8284,
        tau: 150.0,
        rho_star: 0.5,
        delay: 0.0,
        beta: 0.5,
        b: 5.0,
    };

    /// Table S2's fit to hippocampal slices (Wittenberg and Wang 2006; Fig. 3).
    pub const HIPPOCAMPAL_SLICES: Self = Self {
        tau_ca: 0.0488373,
        c_pre: 1.0,
        c_post: 0.275865,
        theta_d: 1.0,
        theta_p: 1.3,
        gamma_d: 313.0965,
        gamma_p: 1645.59,
        sigma: 9.1844,
        tau: 688.355,
        rho_star: 0.5,
        delay: 0.0188008,
        beta: 0.7,
        b: 5.28145,
    };

    /// Table S2's fit to hippocampal cultures (Wang et al. 2005; Fig. S3).
    pub const HIPPOCAMPAL_CULTURES: Self = Self {
        tau_ca: 0.0119536,
        c_pre: 0.58156,
        c_post: 1.76444,
        theta_d: 1.0,
        theta_p: 1.3,
        gamma_d: 61.141,
        gamma_p: 113.6545,
        sigma: 2.5654,
        tau: 33.7596,
        rho_star: 0.5,
        delay: 0.01,
        beta: 0.5,
        b: 36.0263,
    };

    /// Table S2's fit to cortical slices (Sjöström et al. 2001; Figs. 4 and 5).
    pub const CORTICAL_SLICES: Self = Self {
        tau_ca: 0.0226936,
        c_pre: 0.5617539,
        c_post: 1.23964,
        theta_d: 1.0,
        theta_p: 1.3,
        gamma_d: 331.909,
        gamma_p: 725.085,
        sigma: 3.3501,
        tau: 346.3615,
        rho_star: 0.5,
        delay: 0.0046098,
        beta: 0.5,
        b: 5.40988,
    };

    /// Table S3's DPD′ curve, Fig. S1's orange triangle.
    pub const FIG_S1_DPD_PRIME: Self = Self {
        tau_ca: 0.020,
        c_pre: 1.0,
        c_post: 1.3,
        theta_d: 1.0,
        theta_p: 1.3,
        gamma_d: 150.0,
        gamma_p: 310.0,
        sigma: 2.8284,
        tau: 150.0,
        rho_star: 0.5,
        delay: 0.0043,
        beta: 0.5,
        b: 5.0,
    };

    /// Table S3's DP curve, Fig. S1's magenta square.
    pub const FIG_S1_DP: Self = Self {
        tau_ca: 0.020,
        c_pre: 1.0,
        c_post: 2.0,
        theta_d: 1.0,
        theta_p: 1.3,
        gamma_d: 150.0,
        gamma_p: 241.356,
        sigma: 2.8284,
        tau: 150.0,
        rho_star: 0.5,
        delay: 0.0138,
        beta: 0.5,
        b: 5.0,
    };

    /// Every parameter in its range: `τ_Ca`, `θ_d`, `θ_p`, `τ` and `b` finite and positive;
    /// `C_pre`, `C_post`, `γ_d`, `γ_p`, `σ` and `D` finite and non-negative (a blocked NMDA
    /// receptor is `C_pre = 0`, SI §3.1.2); `ρ*` in `(0, 1)`; `β` in `[0, 1]`.
    ///
    /// # Errors
    ///
    /// [`CalciumError::NotPositive`], [`CalciumError::Negative`] or [`CalciumError::OutOfRange`],
    /// naming the first parameter that fails.
    pub fn check(&self) -> Result<(), CalciumError> {
        positive("tau_ca", self.tau_ca)?;
        non_negative("c_pre", self.c_pre)?;
        non_negative("c_post", self.c_post)?;
        positive("theta_d", self.theta_d)?;
        positive("theta_p", self.theta_p)?;
        non_negative("gamma_d", self.gamma_d)?;
        non_negative("gamma_p", self.gamma_p)?;
        non_negative("sigma", self.sigma)?;
        positive("tau", self.tau)?;
        if !(self.rho_star > 0.0 && self.rho_star < 1.0) {
            return Err(CalciumError::OutOfRange { what: "rho_star", value: self.rho_star, range: "(0, 1)" });
        }
        non_negative("delay", self.delay)?;
        unit_closed("beta", self.beta)?;
        positive("b", self.b)?;
        Ok(())
    }

    /// The transients of a spike train: each presynaptic spike's `C_pre` at its time plus `D`, each
    /// postsynaptic spike's `C_post` at its time, in time order.
    ///
    /// # Errors
    ///
    /// [`CalciumError::NonFinite`] for a spike time that is not finite; whatever [`Synapse::check`]
    /// refuses.
    pub fn transients(&self, pre: &[f64], post: &[f64]) -> Result<Vec<Transient>, CalciumError> {
        self.check()?;
        let mut out = Vec::with_capacity(pre.len() + post.len());
        for &t in pre {
            out.push(Transient { time: finite("pre spike time", t)? + self.delay, amplitude: self.c_pre });
        }
        for &t in post {
            out.push(Transient { time: finite("post spike time", t)?, amplitude: self.c_post });
        }
        out.sort_by(|a, b| a.time.total_cmp(&b.time));
        Ok(out)
    }

    /// `n` pre–post pairs at `frequency`: presynaptic spikes at `(k + ½)/f`, each followed by a
    /// postsynaptic spike `dt` later (before it, for `dt < 0`), so the protocol fills `[0, n/f]`.
    ///
    /// # Errors
    ///
    /// [`CalciumError::NotPositive`] for a frequency that is not finite and positive;
    /// [`CalciumError::OutOfRange`] for `|dt|` beyond half a period; whatever
    /// [`Synapse::transients`] refuses.
    pub fn pairs(&self, dt: f64, frequency: f64, n: u32) -> Result<Vec<Transient>, CalciumError> {
        let period = 1.0 / positive("frequency", frequency)?;
        if !(dt.abs() <= 0.5 * period) {
            return Err(CalciumError::OutOfRange { what: "dt", value: dt, range: "[-1/(2f), 1/(2f)]" });
        }
        let pre: Vec<f64> = (0..n).map(|k| (f64::from(k) + 0.5) * period).collect();
        let post: Vec<f64> = pre.iter().map(|t| t + dt).collect();
        self.transients(&pre, &post)
    }

    /// The fractions of time above `θ_d` and `θ_p` for pre–post pairs repeated at `frequency`, `dt`
    /// the postsynaptic spike's time after the presynaptic one: the authors'
    /// `spikePairFrequency(dt − D, frequency)`.
    ///
    /// # Errors
    ///
    /// As [`Synapse::burst_fractions`].
    pub fn pair_fractions(&self, dt: f64, frequency: f64) -> Result<Fractions, CalciumError> {
        self.burst_fractions(dt, frequency, 1, 0.0)
    }

    /// The fractions of time above `θ_d` and `θ_p` for one presynaptic spike and a burst of `n_post`
    /// postsynaptic spikes `interval` apart, repeated at `frequency`; `dt` is the time from the
    /// presynaptic spike to the LAST spike of the burst, the convention of the authors'
    /// `preSpikePostPair(dt − D, frequency, interval)` and of Fig. 3C's drawing.
    ///
    /// # Errors
    ///
    /// [`CalciumError::NonFinite`] for a `dt` that is not finite; [`CalciumError::NotPositive`] for a
    /// frequency that is not finite and positive; [`CalciumError::Negative`] for a negative interval;
    /// whatever [`Synapse::check`] refuses.
    pub fn burst_fractions(&self, dt: f64, frequency: f64, n_post: u32, interval: f64) -> Result<Fractions, CalciumError> {
        self.check()?;
        finite("dt", dt)?;
        let period = 1.0 / positive("frequency", frequency)?;
        non_negative("burst interval", interval)?;
        let mut tr = vec![Transient { time: self.delay, amplitude: self.c_pre }];
        tr.extend((0..n_post).map(|j| Transient { time: dt - f64::from(j) * interval, amplitude: self.c_post }));
        Ok(Fractions {
            alpha_d: fraction_above(self.tau_ca, period, &tr, self.theta_d)?,
            alpha_p: fraction_above(self.tau_ca, period, &tr, self.theta_p)?,
        })
    }

    /// `(Γ_d, Γ_p) = (γ_d α_d, γ_p α_p)`: the average depression and potentiation rates.
    ///
    /// # Errors
    ///
    /// [`CalciumError::OutOfRange`] for a fraction outside `[0, 1]`; whatever [`Synapse::check`]
    /// refuses.
    pub fn rates(&self, f: Fractions) -> Result<(f64, f64), CalciumError> {
        self.check()?;
        unit_closed("alpha_d", f.alpha_d)?;
        unit_closed("alpha_p", f.alpha_p)?;
        Ok((self.gamma_d * f.alpha_d, self.gamma_p * f.alpha_p))
    }

    /// `ρ̄ = Γ_p/(Γ_p + Γ_d)`, SI eq. (9): where the OU process settles. `None` when
    /// `Γ_p + Γ_d = 0`, where it is `0/0` and nothing settles.
    ///
    /// # Errors
    ///
    /// As [`Synapse::rates`].
    pub fn rho_bar(&self, f: Fractions) -> Result<Option<f64>, CalciumError> {
        let (gd, gp) = self.rates(f)?;
        let g = gd + gp;
        Ok(if g > 0.0 { Some(gp / g) } else { None })
    }

    /// `σ_ρ² = σ²(α_p + α_d)/(Γ_p + Γ_d)`, SI eq. (10) — TWICE the stationary variance of `ρ`,
    /// whatever the SI calls it (see the module doc). `None` when `Γ_p + Γ_d = 0`.
    ///
    /// # Errors
    ///
    /// As [`Synapse::rates`].
    pub fn sigma_rho_sq(&self, f: Fractions) -> Result<Option<f64>, CalciumError> {
        let (gd, gp) = self.rates(f)?;
        let g = gd + gp;
        Ok(if g > 0.0 { Some(self.sigma * self.sigma * (f.alpha_p + f.alpha_d) / g) } else { None })
    }

    /// `τ_eff = τ/(Γ_p + Γ_d)`, SI eq. (11), seconds; `+∞` when nothing drives `ρ`.
    ///
    /// # Errors
    ///
    /// As [`Synapse::rates`].
    pub fn tau_eff(&self, f: Fractions) -> Result<f64, CalciumError> {
        let (gd, gp) = self.rates(f)?;
        Ok(self.tau / (gd + gp))
    }

    /// The mean of `ρ` after `t_total` seconds of the OU process (7), from `rho0`:
    /// `ρ̄ − (ρ̄ − ρ₀)e^{−t/τ_eff}`, written `ρ₀e^{−x} + Γ_p (t/τ) φ(x)` with `x = t/τ_eff` so that it
    /// is `ρ₀` when nothing drives `ρ`.
    ///
    /// # Errors
    ///
    /// [`CalciumError::Negative`] for a `t_total` that is negative or not finite;
    /// [`CalciumError::NonFinite`] for a `rho0` that is not finite; whatever [`Synapse::rates`]
    /// refuses.
    pub fn mean(&self, f: Fractions, t_total: f64, rho0: f64) -> Result<f64, CalciumError> {
        let (gd, gp) = self.rates(f)?;
        non_negative("t_total", t_total)?;
        finite("rho0", rho0)?;
        let x = t_total * (gd + gp) / self.tau;
        Ok(rho0 * (-x).exp() + gp * t_total / self.tau * phi(x))
    }

    /// The variance of `ρ` after `t_total` seconds of the OU process (7):
    /// `σ_ρ²(1 − e^{−2t/τ_eff})/2`, written `σ²(α_p + α_d)(t/τ) φ(2x)` so that it is the pure
    /// diffusion `σ²(α_p + α_d)t/τ` when `γ_p = γ_d = 0`.
    ///
    /// # Errors
    ///
    /// [`CalciumError::Negative`] for a `t_total` that is negative or not finite; whatever
    /// [`Synapse::rates`] refuses.
    pub fn variance(&self, f: Fractions, t_total: f64) -> Result<f64, CalciumError> {
        let (gd, gp) = self.rates(f)?;
        non_negative("t_total", t_total)?;
        let x = t_total * (gd + gp) / self.tau;
        Ok(self.sigma * self.sigma * (f.alpha_p + f.alpha_d) * t_total / self.tau * phi(2.0 * x))
    }

    /// The argument of the error function in (13) and (15): `(mean − ρ*)/√(2 variance)`, taken as 0
    /// when the mean sits exactly on `ρ*` so that a noiseless synapse there is split evenly rather
    /// than `0/0`.
    fn z(&self, f: Fractions, t_total: f64, rho0: f64) -> Result<f64, CalciumError> {
        let gap = self.mean(f, t_total, rho0)? - self.rho_star;
        let spread = (2.0 * self.variance(f, t_total)?).sqrt();
        Ok(if gap == 0.0 { 0.0 } else { gap / spread })
    }

    /// `𝒰`, SI eq. (13): the probability that `ρ` ends above `ρ*` after `t_total` seconds from
    /// `rho0` (the UP transition probability when `rho0 = 0`).
    ///
    /// # Errors
    ///
    /// As [`Synapse::mean`].
    pub fn up(&self, f: Fractions, t_total: f64, rho0: f64) -> Result<f64, CalciumError> {
        Ok(0.5 * (1.0 + erf(self.z(f, t_total, rho0)?)))
    }

    /// `𝒟`, SI eq. (15): the probability that `ρ` ends below `ρ*` after `t_total` seconds from
    /// `rho0` (the DOWN transition probability when `rho0 = 1`).
    ///
    /// # Errors
    ///
    /// As [`Synapse::mean`].
    pub fn down(&self, f: Fractions, t_total: f64, rho0: f64) -> Result<f64, CalciumError> {
        Ok(0.5 * (1.0 - erf(self.z(f, t_total, rho0)?)))
    }

    /// The change in synaptic strength after a protocol of `t_total` seconds, after/before:
    /// `([(1 − 𝒰)β + 𝒟(1 − β)] + b[𝒰β + (1 − 𝒟)(1 − β)])/(β + (1 − β)b)` with `𝒰` from `ρ = 0` and
    /// `𝒟` from `ρ = 1`, as the main text's Methods print it (p. 3996) and the authors'
    /// `changeInSynapticStrength` computes it. SI §3.6 prints it with only the UP term divided (see
    /// the module doc).
    ///
    /// # Errors
    ///
    /// As [`Synapse::mean`].
    pub fn change(&self, f: Fractions, t_total: f64) -> Result<f64, CalciumError> {
        let up = self.up(f, t_total, 0.0)?;
        let down = self.down(f, t_total, 1.0)?;
        let (beta, b) = (self.beta, self.b);
        Ok(((1.0 - up) * beta + down * (1.0 - beta) + b * (up * beta + (1.0 - down) * (1.0 - beta))) / (beta + (1.0 - beta) * b))
    }

    /// The ratio `γ_p/γ_d` at which single, non-interacting transients leave `ρ̄ = ½`, SI eqs.
    /// (16)–(18): `Σ ln⁺(C/θ_d) / Σ ln⁺(C/θ_p)` over `C ∈ {C_pre, C_post}`, with `ln⁺ = max(0, ln)`.
    /// Each isolated transient spends `τ_Ca ln(C/θ)` above a threshold it exceeds, so this is eq.
    /// (18)'s four cases in one expression, for either ordering of the amplitudes and thresholds.
    ///
    /// # Errors
    ///
    /// Whatever [`Synapse::check`] refuses.
    pub fn balance(&self) -> Result<Balance, CalciumError> {
        self.check()?;
        let above = |th: f64| (self.c_post / th).ln().max(0.0) + (self.c_pre / th).ln().max(0.0);
        let (d, p) = (above(self.theta_d), above(self.theta_p));
        Ok(if p > 0.0 {
            Balance::Ratio(d / p)
        } else if d > 0.0 {
            Balance::Impossible
        } else {
            Balance::Any
        })
    }

    /// Walk the calcium trace from `c = 0` at `t = 0` to `t_end`, handing `visit` each interval on
    /// which both indicators `Θ[c − θ_d]` and `Θ[c − θ_p]` are constant: `(length, above θ_d,
    /// above θ_p)`. The trace decays exactly between transients and each crossing is
    /// `τ_Ca ln(c/θ)` after the segment starts, kept as a duration rather than an absolute time so
    /// that forty seconds into a train it has the precision of the durations; transients at or after
    /// `t_end` never arrive. The caller has checked `t_end`.
    fn walk(&self, transients: &[Transient], t_end: f64, mut visit: impl FnMut(f64, bool, bool)) -> Result<(), CalciumError> {
        self.check()?;
        let mut ev = Vec::with_capacity(transients.len() + 1);
        for tr in transients {
            ev.push(Transient {
                time: non_negative("transient time", tr.time)?,
                amplitude: non_negative("transient amplitude", tr.amplitude)?,
            });
        }
        ev.retain(|e| e.time < t_end);
        ev.sort_by(|a, b| a.time.total_cmp(&b.time));
        ev.push(Transient { time: t_end, amplitude: 0.0 });
        let (lo, hi) = (self.theta_d.min(self.theta_p), self.theta_d.max(self.theta_p));
        let d_is_lo = self.theta_d <= self.theta_p;
        let (mut t, mut c) = (0.0_f64, 0.0_f64);
        for e in &ev {
            let gap = e.time - t;
            let cross = |th: f64| (self.tau_ca * (c / th).ln()).clamp(0.0, gap);
            let (b_hi, b_lo) = (cross(hi), cross(lo));
            visit(b_hi, true, true);
            visit(b_lo - b_hi, d_is_lo, !d_is_lo);
            visit(gap - b_lo, false, false);
            c = c * (-gap / self.tau_ca).exp() + e.amplitude;
            t = e.time;
        }
        Ok(())
    }

    /// The time the calcium spends at or above `θ_d` and at or above `θ_p` over `[0, t_end]`,
    /// seconds, starting from rest: exact, crossing by crossing.
    ///
    /// # Errors
    ///
    /// [`CalciumError::Negative`] for a `t_end`, transient time or amplitude that is negative or not
    /// finite; whatever [`Synapse::check`] refuses.
    pub fn time_above(&self, transients: &[Transient], t_end: f64) -> Result<(f64, f64), CalciumError> {
        non_negative("t_end", t_end)?;
        let (mut d, mut p) = (0.0, 0.0);
        self.walk(transients, t_end, |len, on_d, on_p| {
            if on_d {
                d += len;
            }
            if on_p {
                p += len;
            }
        })?;
        Ok((d, p))
    }

    /// Eq. 1 by Euler–Maruyama from `rho0` at `t = 0` to `t_end`, under `transients`.
    ///
    /// The calcium is exact and every interval on which `Θ[c − θ_d]` and `Θ[c − θ_p]` are constant
    /// is cut into `⌈L/dt⌉` equal steps, so no step straddles a crossing. Each step is
    /// `ρ ← ρ + (h/τ) f(ρ) + σ√(h/τ) g ξ`, with `f` the drift of eq. 1, `ξ` a standard normal and `g`
    /// the noise factor: `√(Θ_d + Θ_p)` for [`Noise::Corrected`], `Θ[c − min(θ_d, θ_p)]` for
    /// [`Noise::Printed`]. Below both thresholds `g = 0` and no random number is drawn, so a quiet
    /// stretch costs no randomness and a run with `σ = 0` is deterministic.
    ///
    /// # Errors
    ///
    /// [`CalciumError::Negative`] for a `t_end`, transient time or amplitude that is negative or not
    /// finite; [`CalciumError::NonFinite`] for a `rho0` that is not finite;
    /// [`CalciumError::NotPositive`] for a `dt` that is not finite and positive;
    /// [`CalciumError::TooManySteps`] when `t_end/dt` exceeds [`MAX_STEPS`]; whatever
    /// [`Synapse::check`] refuses.
    pub fn simulate(&self, transients: &[Transient], t_end: f64, rho0: f64, dt: f64, noise: Noise, rng: &mut Rng) -> Result<Run, CalciumError> {
        non_negative("t_end", t_end)?;
        finite("rho0", rho0)?;
        positive("dt", dt)?;
        if t_end / dt > MAX_STEPS {
            return Err(CalciumError::TooManySteps { t_end, dt });
        }
        let s = self;
        let (mut rho, mut above_d, mut above_p) = (rho0, 0.0, 0.0);
        self.walk(transients, t_end, |len, on_d, on_p| {
            let (td, tp) = (f64::from(u8::from(on_d)), f64::from(u8::from(on_p)));
            above_d += len * td;
            above_p += len * tp;
            let g = match noise {
                Noise::Corrected => (td + tp).sqrt(),
                Noise::Printed => td.max(tp),
                #[cfg(test)]
                Noise::CommentedSum => td + tp,
            };
            let n = (len / dt).ceil();
            let h = len / n;
            let kick = s.sigma * (h / s.tau).sqrt() * g;
            for _ in 0..n as u64 {
                let drift = -rho * (1.0 - rho) * (s.rho_star - rho) + s.gamma_p * (1.0 - rho) * tp - s.gamma_d * rho * td;
                rho += h / s.tau * drift;
                if on_d || on_p {
                    rho += kick * normal(rng);
                }
            }
        })?;
        Ok(Run { rho, above_d, above_p })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Table S1's six sets in the order `tools/calcium_reference.py` runs them.
    const S1: [Synapse; 6] = [Synapse::DP, Synapse::DPD, Synapse::DPD_PRIME, Synapse::P, Synapse::D, Synapse::D_PRIME];

    fn row(s: &Synapse) -> [f64; 13] {
        [s.tau_ca, s.c_pre, s.c_post, s.theta_d, s.theta_p, s.gamma_d, s.gamma_p, s.sigma, s.tau, s.rho_star, s.delay, s.beta, s.b]
    }

    /// `numpy.linspace(start, stop, 2001)` as numpy 1.26 computes it: `k·step + start` with
    /// `step = (stop − start)/2000`, and the last point set to `stop`.
    fn linspace(start: f64, stop: f64) -> Vec<f64> {
        let step = (stop - start) / 2000.0;
        let mut v: Vec<f64> = (0..2001).map(|k| f64::from(k) * step + start).collect();
        v[2000] = stop;
        v
    }

    /// The change in strength along a grid of `dt`, and whether any calcium crossed a threshold there
    /// (where the authors' code is finite).
    fn curve(s: &Synapse, dts: &[f64], frequency: f64, n: f64, burst: Option<f64>) -> (Vec<f64>, Vec<bool>) {
        let (mut ys, mut driven) = (Vec::new(), Vec::new());
        for &dt in dts {
            let f = match burst {
                None => s.pair_fractions(dt, frequency),
                Some(b) => s.burst_fractions(dt, frequency, 2, b),
            }
            .unwrap();
            driven.push(f.alpha_d + f.alpha_p > 0.0);
            ys.push(s.change(f, n / frequency).unwrap());
        }
        (ys, driven)
    }

    /// Compares a curve with the reference's samples and digest; returns the worst error of a sample
    /// or an extreme, and the relative error of the sum and of the sum of squares.
    fn against(ys: &[f64], driven: &[bool], samples: &[(usize, f64)], digest: (usize, f64, f64, f64, usize, f64, usize)) -> (f64, f64, f64) {
        let (nan, sum, sq, max, imax, min, imin) = digest;
        assert_eq!(driven.iter().filter(|d| !**d).count(), nan, "the points where the reference is NaN");
        let mut worst = 0.0_f64;
        for &(i, want) in samples {
            if want.is_nan() {
                assert!(!driven[i] && ys[i] == 1.0, "at {i}: {} where the reference is NaN", ys[i]);
            } else {
                assert!(driven[i], "at {i}");
                worst = worst.max((ys[i] - want).abs());
            }
        }
        let kept: Vec<f64> = ys.iter().zip(driven).filter(|(_, d)| **d).map(|(y, _)| *y).collect();
        let got_sum: f64 = kept.iter().sum();
        let got_sq: f64 = kept.iter().map(|y| y * y).sum();
        let got_max = kept.iter().fold(f64::NEG_INFINITY, |m, y| m.max(*y));
        let got_min = kept.iter().fold(f64::INFINITY, |m, y| m.min(*y));
        worst = worst.max((got_max - max).abs()).max((got_min - min).abs());
        worst = worst.max((ys[imax] - max).abs()).max((ys[imin] - min).abs());
        (worst, ((got_sum - sum) / sum).abs(), ((got_sq - sq) / sq).abs())
    }

    // Generated by tools/calcium_reference.py from mgraupe/CalciumBasedPlasticityModel@56503b0.
    // synapticChange.choseParameterSet, in the order DP, DPD, DPDprime, P, D, Dprime, 'hippocampal slices',
    // 'hippocampal cultures', 'cortical slices': tauCa, Cpre, Cpost, thetaD, thetaP, gammaD, gammaP, sigma, tau,
    // rhoStar, D, beta, b.
    const PARAMS: [[f64; 13]; 9] = [
        [0.02, 1.0, 2.0, 1.0, 1.3, 200.0, 321.808, 2.8284, 150.0, 0.5, 0.0137, 0.5, 5.0],
        [0.02, 0.9, 0.9, 1.0, 1.3, 250.0, 550.0, 2.8284, 150.0, 0.5, 0.0046, 0.5, 5.0],
        [0.02, 1.0, 2.0, 1.0, 2.5, 50.0, 600.0, 2.8284, 150.0, 0.5, 0.0022, 0.5, 5.0],
        [0.02, 2.0, 2.0, 1.0, 1.3, 160.0, 257.447, 2.8284, 150.0, 0.5, 0.0, 0.5, 5.0],
        [0.02, 0.6, 0.6, 1.0, 1.3, 500.0, 550.0, 5.6568, 150.0, 0.5, 0.0, 0.5, 5.0],
        [0.02, 1.0, 2.0, 1.0, 3.5, 60.0, 600.0, 2.8284, 150.0, 0.5, 0.0, 0.5, 5.0],
        [0.0488373, 1.0, 0.275865, 1.0, 1.3, 313.0965, 1645.59, 9.1844, 688.355, 0.5, 0.0188008, 0.7, 5.28145],
        [0.0119536, 0.58156, 1.76444, 1.0, 1.3, 61.141, 113.6545, 2.5654, 33.7596, 0.5, 0.01, 0.5, 36.0263],
        [0.0226936, 0.5617539, 1.23964, 1.0, 1.3, 331.909, 725.085, 3.3501, 346.3615, 0.5, 0.0046098, 0.5, 5.40988],
    ];
    const FIG2: [[(usize, f64); 31]; 6] = [
        // DP
        [(0, 0.9916803924784937), (100, 0.9863335521162101), (200, 0.9776050265107098), (300, 0.9634529646072405), (400, 0.9407779202670269), (500, 0.905199785976794), (600, 0.8514200091060561), (700, 0.7752903620768624), (800, 0.7642633081277915), (900, 0.881797996139262), (960, 0.9552155244739856), (968, 0.9647521377668227), (976, 0.9741924765550419), (984, 0.9835267623511164), (992, 0.9927459896496238), (1000, 1.0079035766658127), (1008, 1.0466586626592345), (1016, 1.0851497888556292), (1024, 1.123137344848394), (1032, 1.1603926487965668), (1040, 1.1967027969332353), (1100, 1.221361877347258), (1200, 1.1722102590628578), (1300, 1.1247177614638828), (1400, 1.0848979796888498), (1500, 1.0552433547787492), (1600, 1.0349216094220621), (1700, 1.0216973989133942), (1800, 1.0133466967756874), (1900, 1.008162578932034), (2000, 1.0049752521703084)],
        // DPD
        [(0, f64::NAN), (100, f64::NAN), (200, f64::NAN), (300, f64::NAN), (400, f64::NAN), (500, f64::NAN), (600, f64::NAN), (700, 0.99999999999643), (800, 0.9983421343655802), (900, 0.8591617539666864), (960, 0.8597949736293083), (968, 0.8770138301289157), (976, 0.8999325372333163), (984, 0.9281225807726284), (992, 0.9607273969378246), (1000, 0.9966085210427994), (1008, 1.0345151916138262), (1016, 1.0732359254528503), (1024, 1.1117078736505877), (1032, 1.1490774336593887), (1040, 1.1847180888798916), (1100, 0.9607273969378246), (1200, 0.8674892565348516), (1300, 0.9991895342953233), (1400, 0.9999999999999214), (1500, f64::NAN), (1600, f64::NAN), (1700, f64::NAN), (1800, f64::NAN), (1900, f64::NAN), (2000, f64::NAN)],
        // DPDprime
        [(0, 0.945356896102397), (100, 0.9435392122770753), (200, 0.9405302497661824), (300, 0.9355452736468447), (400, 0.9272963491670555), (500, 0.9137429262861128), (600, 0.8919280086971789), (700, 0.8583630847246831), (800, 0.81065802594769), (900, 0.7714045696159365), (960, 0.8069370891224521), (968, 0.826131949104686), (976, 0.867314726720318), (984, 0.9153727248504584), (992, 0.9728892531634181), (1000, 1.0412099593439255), (1008, 1.119413608585195), (1016, 1.2040280725916948), (1024, 1.2590742413287583), (1032, 1.221220369385177), (1040, 1.1838851924450497), (1100, 0.9712408925153699), (1200, 0.9003158550913524), (1300, 0.9190032373828142), (1400, 0.9305097074482869), (1500, 0.9374890365656866), (1600, 0.9417034373235541), (1700, 0.9442476849264715), (1800, 0.9457852282391243), (1900, 0.9467154342550578), (2000, 0.9472786962335532)],
        // P
        [(0, 1.002651776980531), (100, 1.0043466185077883), (200, 1.0071001269052309), (300, 1.011531897371839), (400, 1.0185577618802053), (500, 1.0294310774257425), (600, 1.0456403189030985), (700, 1.0684900534429551), (800, 1.0982751621391695), (900, 1.2298884845030253), (960, 1.2525233491880297), (968, 1.2496580642707158), (976, 1.2467366968317515), (984, 1.243759693040076), (992, 1.2407276484576153), (1000, 1.2376413150312648), (1008, 1.2407276484576153), (1016, 1.243759693040076), (1024, 1.2467366968317515), (1032, 1.2496580642707158), (1040, 1.2525233491880297), (1100, 1.2298884845030253), (1200, 1.0982751621391695), (1300, 1.0684900534429551), (1400, 1.0456403189030985), (1500, 1.0294310774257425), (1600, 1.0185577618802053), (1700, 1.011531897371839), (1800, 1.0071001269052304), (1900, 1.0043466185077883), (2000, 1.002651776980531)],
        // D
        [(0, f64::NAN), (100, f64::NAN), (200, f64::NAN), (300, f64::NAN), (400, f64::NAN), (500, f64::NAN), (600, f64::NAN), (700, f64::NAN), (800, f64::NAN), (900, f64::NAN), (960, 0.9648104927908445), (968, 0.9170592399156275), (976, 0.8521793021948761), (984, 0.7788381497608059), (992, 0.7052338821967962), (1000, 0.6370843848885008), (1008, 0.7052338821967962), (1016, 0.7788381497608059), (1024, 0.8521793021948761), (1032, 0.9170592399156275), (1040, 0.9648104927908445), (1100, f64::NAN), (1200, f64::NAN), (1300, f64::NAN), (1400, f64::NAN), (1500, f64::NAN), (1600, f64::NAN), (1700, f64::NAN), (1800, f64::NAN), (1900, f64::NAN), (2000, f64::NAN)],
        // Dprime
        [(0, 0.9215461747756003), (100, 0.9187664005429049), (200, 0.9141778325671681), (300, 0.9066138000241205), (400, 0.8942082241283315), (500, 0.8741455187274499), (600, 0.8427236219717592), (700, 0.7964680746746584), (800, 0.7348802511485886), (900, 0.7191584486053579), (960, 0.7594239742318903), (968, 0.7644653222919809), (976, 0.7694184914560392), (984, 0.7742806949998201), (992, 0.7790494354659652), (1000, 0.7837225050018399), (1008, 0.7882979838932421), (1016, 0.792774237387324), (1024, 0.7971499109129662), (1032, 0.8014239238194802), (1040, 0.8055954617645534), (1100, 0.8335979633530703), (1200, 0.8681403278783929), (1300, 0.8904403533065532), (1400, 0.9043026493797298), (1500, 0.9127730225392616), (1600, 0.9179149957540199), (1700, 0.9210304572815543), (1800, 0.9229176443114007), (1900, 0.9240610952677272), (2000, 0.9247541300093652)],
    ];
    const FIG2_DIGEST: [(usize, f64, f64, f64, usize, f64, usize); 6] = [
        (0, 1984.0257842854498, 1994.8820472678078, 1.24399923147461, 1051, 0.7220186180759401, 758),
        (1122, 853.0589352670748, 833.0197026366917, 1.2100586757538145, 1046, 0.8393799427933804, 935),
        (0, 1843.6276375598025, 1707.7563744554793, 1.268539944700088, 1022, 0.761132504411235, 883),
        (0, 2124.9968543455534, 2268.1820470780513, 1.2679027768574769, 914, 1.002651776980531, 0),
        (1838, 147.50839639855394, 135.66284340026385, 1.0, 919, 0.6370843848885008, 1000),
        (0, 1740.706799342567, 1522.3645436582835, 0.9247541300093652, 2000, 0.6917945576461729, 861),
    ];
    const PAIR_ALPHAS: [[(f64, f64, f64); 7]; 6] = [
        [(-0.1, 0.013998345562877702, 0.008615658321849084), (-0.02, 0.02017213031583294, 0.0096775597371333), (-0.005, 0.025453235980846353, 0.01495866540214671), (0.0, 0.027644581357717087, 0.01731295439021635), (0.005, 0.025310554786473684, 0.019978927818972946), (0.02, 0.020084485415475008, 0.014837200126125185), (0.1, 0.013996164424633164, 0.008748879135283341)],
        [(-0.1, 0.0, 0.0), (-0.02, 0.003021146347585237, 0.0), (-0.005, 0.007526287174758312, 0.0022790018854084893), (0.0, 0.00958769282091805, 0.004340407531568229), (0.005, 0.0115567332813762, 0.006309447992026378), (0.02, 0.005502750777676543, 0.0002554654883267193), (0.1, 0.0, 0.0)],
        [(-0.1, 0.014102941180169278, 0.0), (-0.02, 0.023988665390031826, 0.0), (-0.005, 0.024670609303532744, 0.0), (0.0, 0.022732787271673043, 0.002206972634189944), (0.005, 0.02108176786814962, 0.0027559532306665174), (0.02, 0.017597976141374802, 0.0), (0.1, 0.013938016760045253, 0.0)],
        [(-0.1, 0.027860194192180172, 0.017365623613480527), (-0.02, 0.03399112097276227, 0.023496550394062624), (-0.005, 0.030381732008775777, 0.02513444671942596), (0.0, 0.027725887222397813, 0.02247860193304799), (0.005, 0.030381732008775777, 0.02513444671942596), (0.02, 0.03399112097276227, 0.023496550394062624), (0.1, 0.027860194192180172, 0.017365623613480527)],
        [(-0.1, 0.0, 0.0), (-0.02, 0.0, 0.0), (-0.005, 0.0013022759222570607, 0.0), (0.0, 0.0036464311358790917, 0.0), (0.005, 0.0013022759222570607, 0.0), (0.02, 0.0, 0.0), (0.1, 0.0, 0.0)],
        [(-0.1, 0.014130661645627885, 0.0), (-0.02, 0.02489183788983993, 0.0), (-0.005, 0.023781398622238385, 0.0), (0.0, 0.021972245773362195, 0.0), (0.005, 0.02044038923277104, 0.0), (0.02, 0.017239896081165024, 0.0), (0.1, 0.013930209835641576, 0.0)],
    ];
    // spikePairFrequency at 20 Hz, 'cortical slices': (dt, alphaD, alphaP)
    const PAIR_ALPHAS_20HZ: [(f64, f64, f64); 5] = [
        (-0.02, 0.25762248270025406, 0.09419431937276107),
        (-0.01, 0.3326574467966828, 0.09449785991190456),
        (0.0, 0.3510574760726435, 0.19833164888426263),
        (0.01, 0.28929238483545805, 0.17021259139306888),
        (0.02, 0.2445752479822528, 0.12549545453986366),
    ];
    const BURST_ALPHAS: [(f64, f64, f64); 9] = [
        (-0.08, 0.019549528647215916, 0.0),
        (-0.03, 0.04489186688669914, 0.0),
        (-0.005, 0.06879465863303645, 0.004728847167645427),
        (0.0, 0.0748460038757514, 0.010780192410360406),
        (0.004, 0.08003280893846423, 0.015966997473073237),
        (0.0115, 0.09063206632164707, 0.02656625485625606),
        (0.015, 0.09598629212853038, 0.03192048066313936),
        (0.03, 0.12207376856584747, 0.0023166752657022083),
        (0.08, 0.011031932846238204, 0.0),
    ];
    const FIG3: [[(usize, f64); 21]; 3] = [
        [(0, 0.9812142539339858), (100, 0.9732604059224109), (200, 0.9624689972744257), (300, 0.9483061229810954), (400, 0.9303815206838424), (500, 0.9085495312154567), (600, 0.8829892288200035), (700, 0.8542434100725781), (800, 0.8232040616752472), (900, 0.7910412239650022), (1000, 0.7590808660448326), (1100, 0.7286464904404363), (1200, 0.7026817077045565), (1300, 0.8633657312301951), (1400, 0.997607639204221), (1500, 0.9967524090413612), (1600, 0.9954917585699162), (1700, 0.9936301247129963), (1800, 0.9908924594474295), (1900, 0.9869094547546305), (2000, 0.9812142539339858)],
        [(0, 0.9937910547279136), (100, 0.9892282327100765), (200, 0.9821732491615253), (300, 0.971899500772344), (400, 0.9577948712464642), (500, 0.9394972931061818), (600, 0.9169996401585889), (700, 0.8906974301269933), (800, 0.8613700506418717), (900, 0.8301017349741142), (1000, 1.1137170457141872), (1100, 1.5129700845758445), (1200, 1.6756560797928195), (1300, 0.7566049088229936), (1400, 0.8071946392486986), (1500, 0.9881484517440375), (1600, 0.9995204875159655), (1700, 0.999057122820076), (1800, 0.9981770449940705), (1900, 0.9965729247910645), (2000, 0.9937910547279136)],
        [(0, 0.9999990726944977), (100, 0.9999950144231486), (200, 0.999977158192573), (300, 0.9999110531267863), (400, 0.9997044771068948), (500, 0.9991549959524932), (600, 0.9978960790713389), (700, 0.9953770573106692), (800, 0.9909100083919937), (900, 0.9837870019512527), (1000, 1.01889867008289), (1100, 1.1443500352818399), (1200, 1.2645632793133614), (1300, 0.9334557104735675), (1400, 0.9767163877427866), (1500, 0.999993337921312), (1600, 0.9999999996842168), (1700, 0.9999999973455229), (1800, 0.9999999791068901), (1900, 0.9999998514606208), (2000, 0.9999990726944977)],
    ];
    const FIG3_DIGEST: [(usize, f64, f64, f64, usize, f64, usize); 3] = [
        (0, 1810.1598778696, 1655.5079230206602, 0.9978823881644348, 1357, 0.7014130036396699, 1198),
        (0, 2019.7821329729143, 2120.48593802748, 1.7676083122804436, 1188, 0.7065464281457383, 1316),
        (0, 2030.1077880894566, 2067.7011246992442, 1.325411365692359, 1188, 0.9180791301826547, 1316),
    ];
    // (frequency, the grid's half-width, samples)
    type Fig4bRow = (f64, f64, [(usize, f64); 21]);
    const FIG4B: [Fig4bRow; 5] = [
        (1.0, 0.1, [(0, 0.990795368385839), (100, 0.990795368385839), (200, 0.990795368385839), (300, 0.990795368385839), (400, 0.990795368385839), (500, 0.990795368385839), (600, 0.990795368385839), (700, 0.990795368385839), (800, 0.990795368385839), (900, 0.6907557602463055), (1000, 0.6830584092619681), (1100, 1.0613302014951376), (1200, 0.9262891228151011), (1300, 0.9132046421531077), (1400, 0.9323758977307702), (1500, 0.9501955978298734), (1600, 0.9662512716368054), (1700, 0.9773367326001042), (1800, 0.9830619882201459), (1900, 0.9861877865650484), (2000, 0.987978305861131)]),
        (20.0, 0.025, [(0, 1.0153568495420995), (100, 0.9302222667189206), (200, 0.8452819726678176), (300, 0.761044852306855), (400, 0.6792789312371074), (500, 0.6028016504242918), (600, 0.6348511752835353), (700, 0.707983257435073), (800, 0.7977325759108385), (900, 1.0045552238890443), (1000, 1.2324174898342046), (1100, 1.3622470806248603), (1200, 1.3285689353982686), (1300, 1.2919603849197763), (1400, 1.253017444341527), (1500, 1.2126144899013973), (1600, 1.171848248032059), (1700, 1.1319311145234885), (1800, 1.0940526000366304), (1900, 1.0592395805946935), (2000, 1.0153568495420995)]),
        (30.0, 0.016666666666666666, [(0, 1.0703913015632702), (100, 1.0943993027117063), (200, 1.119614702465801), (300, 1.1455893947430937), (400, 1.1863390595408407), (500, 1.2743760117801366), (600, 1.3561062810333222), (700, 1.4289360146000938), (800, 1.4890693888053999), (900, 1.4798683644568364), (1000, 1.4700787254817564), (1100, 1.4596875983381197), (1200, 1.4486884478865458), (1300, 1.42081291415677), (1400, 1.3316937256885975), (1500, 1.243946380649607), (1600, 1.191707831317299), (1700, 1.136161003231268), (1800, 1.077989556706713), (1900, 1.048035873910973), (2000, 1.0703913015632702)]),
        (40.0, 0.0125, [(0, 1.5685024865441193), (100, 1.5767598539059362), (200, 1.5850671409777741), (300, 1.589825323057599), (400, 1.5780163845784518), (500, 1.5652020876097665), (600, 1.55136892369559), (700, 1.5365144132703006), (800, 1.5209339737559173), (900, 1.5166755658755005), (1000, 1.5122794358969769), (1100, 1.5077467630094747), (1200, 1.5030794586177096), (1300, 1.4982802099815236), (1400, 1.4982510410092218), (1500, 1.5081648578157039), (1600, 1.516984662501715), (1700, 1.5357927390428656), (1800, 1.552628100474137), (1900, 1.5604162746359862), (2000, 1.5685024865441193)]),
        (50.0, 0.01, [(0, 1.636808856446775), (100, 1.636808856446775), (200, 1.636808856446775), (300, 1.636808856446775), (400, 1.636808856446775), (500, 1.636808856446775), (600, 1.636808856446775), (700, 1.636808856446775), (800, 1.636808856446775), (900, 1.636808856446775), (1000, 1.636808856446775), (1100, 1.636808856446775), (1200, 1.636808856446775), (1300, 1.636808856446775), (1400, 1.6332363055317338), (1500, 1.6342934699564926), (1600, 1.636808856446775), (1700, 1.636808856446775), (1800, 1.636808856446775), (1900, 1.636808856446775), (2000, 1.636808856446775)]),
    ];
    const FIG4B_DIGEST: [(usize, f64, f64, f64, usize, f64, usize); 5] = [
        (0, 1910.0177955252998, 1841.1677699792162, 1.1783589340753673, 1046, 0.5613106687244857, 929),
        (0, 2013.410665247919, 2141.242881005576, 1.3711025671331625, 1072, 0.590314089483715, 518),
        (0, 2547.7086230974664, 3292.3407886030345, 1.4893365018672333, 797, 1.0403628587951723, 1863),
        (0, 3079.828433580748, 4742.1640820167695, 1.5918370710734127, 282, 1.4949278886010287, 1369),
        (0, 3274.6012004011022, 5358.829604512016, 1.636808856446775, 0, 1.6305184983245906, 1461),
    ];
    const FIG4A: [(f64, f64, f64); 8] = [
        (0.1, 1.0613302014951276, 0.690755760246329),
        (1.0, 1.0613302014951376, 0.6907557602463055),
        (5.0, 1.0615915684593042, 0.6898755450223933),
        (10.0, 1.083003500587286, 0.62137587697656),
        (20.0, 1.2530174443415267, 0.6348511752835353),
        (30.0, 1.191707831317299, 1.1863390595408407),
        (40.0, 1.552628100474137, 1.5850671409777741),
        (50.0, 1.636808856446775, 1.636808856446775),
    ];
    // changeInSynapticStrength(T, 0.5, alphaD, alphaP) on the DP set and on 'hippocampal cultures'
    const OU: [(usize, f64, f64, f64, [f64; 8]); 6] = [
        (0, 0.02017213031583294, 0.0096775597371333, 60.0, [0.4356453280430729, 0.03340349020442513, 20.98271215487227, 0.2443902181299395, 0.5979952559382496, 0.4106839048284605, 0.46798148524035094, 0.7642633081277932]),
        (0, 0.013, 0.021, 60.0, [0.7221619052341278, 0.029065581656188607, 16.029120851877245, 0.9555757604999073, 0.028849622431026622, 0.7050621840350159, 0.7287406986107302, 1.6178174253792539]),
        (0, 0.05, 0.001, 10.0, [0.03117748363465005, 0.03952720052145903, 14.532337745480248, 3.357711408069619e-05, 0.44103689620237363, 0.015510185439310882, 0.5180298364456172, 0.7059977872744714]),
        (1, 0.02017213031583294, 0.0096775597371333, 60.0, [0.4714033131274151, 0.08419574435059231, 14.468963672888988, 0.4302520937754117, 0.539289542766472, 0.4639481486714556, 0.479762981211558, 0.8968522806870906]),
        (1, 0.013, 0.021, 60.0, [0.7501764454896981, 0.07033096740217705, 10.61096264353139, 0.9065990740112875, 0.0903224029122418, 0.747549612166246, 0.7510512328039579, 1.772184948669365]),
        (1, 0.05, 0.001, 10.0, [0.03584518834852002, 0.10585822020311259, 10.647349823990218, 0.011968748248293526, 0.6598049834907969, 0.021831865736205873, 0.4127719962793539, 0.3871570660186274]),
    ];
    // eventBasedIntegration on 400 events, t_k = t_(k-1) + 0.004 + 0.2*frac(k*0.6180339887498949), post when k % 3 == 0
    const EVENT_BASED: (f64, f64, f64) = (41.66517954831422, 2.420561726026117, 1.4085928681922237);
    // spikePairFrequency(x, 5.0) on the DP amplitudes at x = 0.23, -0.03, +0.03
    const WRAP: [(f64, f64); 3] = [(0.10623208468919358, 0.05375923179569536), (0.10623208468919358, 0.05375923179569536), (0.07994138407241892, 0.05365973213346051)];
    // output/DP_curve/final_camkII_state.dat: (Delta t ms, U = column 7, D = -column 9)
    const SIM_DP: [(f64, f64, f64); 41] = [
        (-100.0, 0.301, 0.33),
        (-95.0, 0.3, 0.325),
        (-90.0, 0.326, 0.333),
        (-85.0, 0.324, 0.348),
        (-80.0, 0.283, 0.338),
        (-75.0, 0.299, 0.343),
        (-70.0, 0.287, 0.357),
        (-65.0, 0.27, 0.357),
        (-60.0, 0.286, 0.384),
        (-55.0, 0.295, 0.42),
        (-50.0, 0.273, 0.436),
        (-45.0, 0.282, 0.426),
        (-40.0, 0.212, 0.471),
        (-35.0, 0.238, 0.51),
        (-30.0, 0.195, 0.544),
        (-25.0, 0.187, 0.608),
        (-20.0, 0.222, 0.582),
        (-15.0, 0.271, 0.571),
        (-10.0, 0.369, 0.542),
        (-5.0, 0.429, 0.517),
        (0.0, 0.458, 0.467),
        (5.0, 0.667, 0.312),
        (10.0, 0.649, 0.308),
        (15.0, 0.599, 0.332),
        (20.0, 0.601, 0.319),
        (25.0, 0.542, 0.313),
        (30.0, 0.528, 0.337),
        (35.0, 0.509, 0.326),
        (40.0, 0.469, 0.345),
        (45.0, 0.441, 0.33),
        (50.0, 0.414, 0.354),
        (55.0, 0.406, 0.345),
        (60.0, 0.378, 0.341),
        (65.0, 0.397, 0.325),
        (70.0, 0.354, 0.29),
        (75.0, 0.354, 0.313),
        (80.0, 0.334, 0.329),
        (85.0, 0.355, 0.334),
        (90.0, 0.339, 0.335),
        (95.0, 0.336, 0.32),
        (100.0, 0.329, 0.339),
    ];
    // output/P_curve/final_camkII_state.dat: (Delta t ms, U = column 7, D = -column 9)
    const SIM_P: [(f64, f64, f64); 41] = [
        (-100.0, 0.469, 0.5),
        (-95.0, 0.463, 0.447),
        (-90.0, 0.469, 0.453),
        (-85.0, 0.47, 0.469),
        (-80.0, 0.455, 0.455),
        (-75.0, 0.468, 0.445),
        (-70.0, 0.478, 0.478),
        (-65.0, 0.439, 0.439),
        (-60.0, 0.47, 0.43),
        (-55.0, 0.483, 0.462),
        (-50.0, 0.486, 0.457),
        (-45.0, 0.482, 0.431),
        (-40.0, 0.514, 0.434),
        (-35.0, 0.504, 0.443),
        (-30.0, 0.507, 0.406),
        (-25.0, 0.535, 0.432),
        (-20.0, 0.571, 0.428),
        (-15.0, 0.583, 0.435),
        (-10.0, 0.646, 0.309),
        (-5.0, 0.678, 0.293),
        (0.0, 0.662, 0.293),
        (5.0, 0.68, 0.292),
        (10.0, 0.627, 0.298),
        (15.0, 0.573, 0.406),
        (20.0, 0.528, 0.439),
        (25.0, 0.523, 0.421),
        (30.0, 0.517, 0.441),
        (35.0, 0.524, 0.424),
        (40.0, 0.491, 0.465),
        (45.0, 0.46, 0.443),
        (50.0, 0.484, 0.425),
        (55.0, 0.483, 0.454),
        (60.0, 0.461, 0.453),
        (65.0, 0.471, 0.451),
        (70.0, 0.473, 0.436),
        (75.0, 0.471, 0.453),
        (80.0, 0.456, 0.469),
        (85.0, 0.448, 0.472),
        (90.0, 0.453, 0.452),
        (95.0, 0.462, 0.453),
        (100.0, 0.447, 0.437),
    ];
    // output/Dprime_curve/final_camkII_state.dat: (Delta t ms, U = column 7, D = -column 9)
    const SIM_D_PRIME: [(f64, f64, f64); 41] = [
        (-100.0, 0.003, 0.092),
        (-95.0, 0.001, 0.11),
        (-90.0, 0.002, 0.134),
        (-85.0, 0.002, 0.132),
        (-80.0, 0.0, 0.098),
        (-75.0, 0.003, 0.13),
        (-70.0, 0.001, 0.122),
        (-65.0, 0.003, 0.136),
        (-60.0, 0.004, 0.15),
        (-55.0, 0.004, 0.161),
        (-50.0, 0.007, 0.16),
        (-45.0, 0.006, 0.195),
        (-40.0, 0.003, 0.243),
        (-35.0, 0.01, 0.258),
        (-30.0, 0.004, 0.313),
        (-25.0, 0.012, 0.344),
        (-20.0, 0.007, 0.374),
        (-15.0, 0.015, 0.466),
        (-10.0, 0.011, 0.403),
        (-5.0, 0.014, 0.362),
        (0.0, 0.008, 0.32),
        (5.0, 0.009, 0.283),
        (10.0, 0.007, 0.262),
        (15.0, 0.007, 0.219),
        (20.0, 0.004, 0.192),
        (25.0, 0.012, 0.19),
        (30.0, 0.004, 0.16),
        (35.0, 0.003, 0.139),
        (40.0, 0.003, 0.132),
        (45.0, 0.001, 0.115),
        (50.0, 0.001, 0.134),
        (55.0, 0.003, 0.111),
        (60.0, 0.001, 0.097),
        (65.0, 0.0, 0.124),
        (70.0, 0.001, 0.116),
        (75.0, 0.001, 0.103),
        (80.0, 0.002, 0.113),
        (85.0, 0.004, 0.097),
        (90.0, 0.002, 0.105),
        (95.0, 0.0, 0.101),
        (100.0, 0.002, 0.119),
    ];
    // output/DPDprime_curve/final_camkII_state.dat: (Delta t ms, change in strength = column 5, its error = column 6)
    const SIM_DPD_PRIME_CHANGE: [(f64, f64, f64); 41] = [
        (-100.0, 0.942, 0.0263367),
        (-95.0, 0.954, 0.0239424),
        (-90.0, 0.942667, 0.0253661),
        (-85.0, 0.944667, 0.0258325),
        (-80.0, 0.95, 0.0244797),
        (-75.0, 0.942, 0.0268797),
        (-70.0, 0.94, 0.0255969),
        (-65.0, 0.942, 0.0257798),
        (-60.0, 0.926667, 0.0287363),
        (-55.0, 0.918, 0.0305406),
        (-50.0, 0.918, 0.0316495),
        (-45.0, 0.910667, 0.0320087),
        (-40.0, 0.894, 0.033516),
        (-35.0, 0.879333, 0.0351862),
        (-30.0, 0.862, 0.0370978),
        (-25.0, 0.836667, 0.0402797),
        (-20.0, 0.826, 0.0411587),
        (-15.0, 0.791333, 0.0443238),
        (-10.0, 0.779333, 0.0434433),
        (-5.0, 0.823333, 0.0407768),
        (0.0, 1.034, 0.0454819),
        (5.0, 1.146, 0.0468536),
        (10.0, 0.971333, 0.0375948),
        (15.0, 0.892667, 0.0344529),
        (20.0, 0.889333, 0.0344534),
        (25.0, 0.904, 0.0324683),
        (30.0, 0.938, 0.0298142),
        (35.0, 0.920667, 0.0294421),
        (40.0, 0.946, 0.0258579),
        (45.0, 0.942, 0.0263367),
        (50.0, 0.953333, 0.0231444),
        (55.0, 0.938667, 0.0272126),
        (60.0, 0.956, 0.0231914),
        (65.0, 0.939333, 0.0260054),
        (70.0, 0.948667, 0.0250476),
        (75.0, 0.950667, 0.0237275),
        (80.0, 0.944, 0.0251036),
        (85.0, 0.950667, 0.0240366),
        (90.0, 0.949333, 0.0243166),
        (95.0, 0.95, 0.0241772),
        (100.0, 0.962, 0.0221383),
    ];
    // output/<case>_curve/camkmotifscan.par, in the order DP, DPD, DPDprime, P, D, Dprime: C_pre, C_post, tau_pre, Ct_dephos, Ct_phos, dephos, phos, sigma, tau_rho, epsilon, delay
    const PAR_FILES: [[f64; 11]; 6] = [
        [1.0, 2.0, 20.0, 1.0, 1.3, 4.0, 6.43616, 0.4, 3000.0, 0.02, 13.7],
        [0.9, 0.9, 20.0, 1.0, 1.3, 5.0, 11.0, 0.4, 3000.0, 0.02, 4.6],
        [0.9, 0.9, 20.0, 1.0, 1.3, 5.0, 11.0, 0.4, 3000.0, 0.02, 4.6],
        [2.0, 2.0, 20.0, 1.0, 1.3, 3.2, 5.14893, 0.4, 3000.0, 0.02, 0.0],
        [0.6, 0.6, 20.0, 1.0, 1.3, 10.0, 11.0, 0.8, 3000.0, 0.02, 0.0],
        [1.0, 2.0, 20.0, 1.0, 3.5, 1.2, 12.0, 0.4, 3000.0, 0.02, 0.0],
    ];
    // 'cortical slices', 75 pairs: the frequency above which every offset on the Fig. 4B grid potentiates, bisected
    const EVERY_OFFSET_POTENTIATES_HZ: (f64, f64) = (29.113918698440102, 29.11391869844465);

    /// Every value Tables S1 and S2 print (SI pp. 7–8) is the value the authors' `choseParameterSet`
    /// holds, bit for bit; Table S3's two complete columns (SI p. 9), which the code does not hold,
    /// are pinned as printed.
    ///
    /// The tables print `τ_Ca` and `D` in milliseconds and the code holds seconds; `13.7e-3` and
    /// `0.0137` are the same double, so the comparison is exact. Every set passes [`Synapse::check`].
    #[test]
    fn every_table_value_is_the_authors_code_bit_for_bit() {
        let sets = [
            Synapse::DP,
            Synapse::DPD,
            Synapse::DPD_PRIME,
            Synapse::P,
            Synapse::D,
            Synapse::D_PRIME,
            Synapse::HIPPOCAMPAL_SLICES,
            Synapse::HIPPOCAMPAL_CULTURES,
            Synapse::CORTICAL_SLICES,
        ];
        for (s, want) in sets.iter().zip(PARAMS) {
            assert_eq!(row(s), want);
            assert_eq!(s.check(), Ok(()));
        }
        assert_eq!(row(&Synapse::FIG_S1_DPD_PRIME), [20e-3, 1.0, 1.3, 1.0, 1.3, 150.0, 310.0, 2.8284, 150.0, 0.5, 4.3e-3, 0.5, 5.0]);
        assert_eq!(row(&Synapse::FIG_S1_DP), [20e-3, 1.0, 2.0, 1.0, 1.3, 150.0, 241.356, 2.8284, 150.0, 0.5, 13.8e-3, 0.5, 5.0]);
        assert_eq!(Synapse::FIG_S1_DP.check(), Ok(()));
        assert_eq!(Synapse::FIG_S1_DPD_PRIME.check(), Ok(()));
        // Table S1's DP delay in its printed unit, and Table S2's hippocampal-slice time constant.
        assert_eq!(Synapse::DP.delay, 13.7e-3);
        assert_eq!(Synapse::HIPPOCAMPAL_SLICES.tau_ca, 48.8373e-3);
    }

    /// Where single transients cross both thresholds, the printed `γ_p` is `γ_d` times eq. (18)'s
    /// ratio to the printed third decimal — for P consistently with a second rounding that the
    /// authors' own parameter file shows — and where they do not, eq. (18) has no ratio.
    ///
    /// DP and Table S3's DP (`C_pre = θ_d < θ_p < C_post`) sit where (18)'s second and third cases
    /// meet, both eq. (17)'s ratio since `ln(C_pre/θ_d) = 0`; P (`θ_d < θ_p < C_pre = C_post`) is its
    /// fourth, `(ln(C_post/θ_d) + ln(C_pre/θ_d))/(ln(C_post/θ_p) + ln(C_pre/θ_p))`, which with equal
    /// amplitudes is (17)'s ratio again. All three have
    /// `ln 2/ln(2/1.3) = 1.6090406`, and the exact products are `321.80811`, `257.44649` and
    /// `241.35608`. DP and Table S3's DP print them rounded, `321.808` and `241.356`. P prints
    /// `257.447`, one unit in the last place above `257.446`. That is consistent with rounding twice:
    /// the committed `output/P_curve/camkmotifscan.par` holds `phos = 5.14893` in its units scaled by
    /// 50, the product over 50 rounded to five decimals, and `50 × 5.14893 = 257.4465` rounds half up
    /// to the printed digit. (DP's file holds `6.43616`, which gives `321.808` either way.) DPD and D
    /// never cross either threshold alone (eq. 18's "arbitrary"); DPD′, D′ and Table S3's DPD′ cross
    /// `θ_d` alone but never `θ_p`, so no ratio balances them.
    #[test]
    fn the_printed_potentiation_rates_are_eq_18s_ratio_rounded() {
        let r = 2.0_f64.ln() / (2.0 / 1.3_f64).ln();
        let round = |x: f64, places: i32| (x * 10f64.powi(places)).round() / 10f64.powi(places);
        for (s, exact) in [(Synapse::DP, 321.808_11), (Synapse::P, 257.446_49), (Synapse::FIG_S1_DP, 241.356_08)] {
            let Balance::Ratio(got) = s.balance().unwrap() else { panic!("{s:?} balances") };
            assert!((got - r).abs() < 1e-15, "{got} against {r}");
            let product = s.gamma_d * got;
            assert!((product - exact).abs() < 5e-6, "{product}");
            assert!((product - s.gamma_p).abs() < 6e-4, "{product}");
            if s == Synapse::P {
                assert_eq!(round(product, 3), 257.446, "{product}");
                assert_eq!(round(product / 50.0, 5), 5.148_93);
                assert_eq!(round(50.0 * 5.148_93, 3), s.gamma_p);
            } else {
                assert_eq!(round(product, 3), s.gamma_p, "{product}");
            }
        }
        assert_eq!(round(Synapse::DP.gamma_d * r / 50.0, 5), 6.436_16);
        assert!((r - 1.609_040_6).abs() < 5e-8, "{r}");
        for s in [Synapse::DPD, Synapse::D] {
            assert_eq!(s.balance(), Ok(Balance::Any));
        }
        for s in [Synapse::DPD_PRIME, Synapse::D_PRIME, Synapse::FIG_S1_DPD_PRIME] {
            assert_eq!(s.balance(), Ok(Balance::Impossible));
        }
        // Pairs 5 s apart at 0.1 Hz are 250 calcium time constants apart and do not interact, and the
        // exact ratio leaves the strength unchanged to rounding, whatever the noise (measured
        // `5.6 × 10⁻¹⁷` in the rates and `3.3 × 10⁻¹⁶` in the change).
        let balanced = Synapse { gamma_p: Synapse::DP.gamma_d * r, ..Synapse::DP };
        let f = balanced.pair_fractions(balanced.delay + 5.0, 0.1).unwrap();
        assert!(f.alpha_p > 0.0 && f.alpha_d > f.alpha_p);
        assert!((balanced.gamma_p * f.alpha_p - balanced.gamma_d * f.alpha_d).abs() < 3e-16, "{f:?}");
        assert!((balanced.change(f, 600.0).unwrap() - 1.0).abs() < 1e-15);
    }

    /// `balance` is eq. (18) in each of its four cases, and in the orderings the SI leaves to the
    /// reader: the amplitudes swapped, and a potentiation threshold below the depression one.
    #[test]
    fn balance_is_eq_18_in_each_of_its_cases() {
        let with = |c_pre: f64, c_post: f64, theta_d: f64, theta_p: f64| Synapse { c_pre, c_post, theta_d, theta_p, ..Synapse::DP }.balance().unwrap();
        let ln = f64::ln;
        assert_eq!(with(0.5, 0.8, 1.0, 1.3), Balance::Any, "C_pre, C_post < θ_d, θ_p");
        let case2 = ln(2.5) / ln(2.5 / 1.3);
        let case3 = (ln(2.5) + ln(1.2)) / ln(2.5 / 1.3);
        let case4 = (ln(2.5) + ln(1.6)) / (ln(2.5 / 1.3) + ln(1.6 / 1.3));
        for (got, want) in [
            (with(0.7, 2.5, 1.0, 1.3), case2),
            (with(1.2, 2.5, 1.0, 1.3), case3),
            (with(1.6, 2.5, 1.0, 1.3), case4),
            (with(2.5, 0.7, 1.0, 1.3), case2),
            (with(2.5, 1.6, 1.0, 1.3), case4),
        ] {
            let Balance::Ratio(r) = got else { panic!("{got:?}") };
            assert!((r - want).abs() < 1e-15, "{r} against {want}");
        }
        assert_eq!(with(0.5, 1.2, 1.5, 1.0), Balance::Ratio(0.0), "only potentiation is reached: γ_p must be 0");
        assert_eq!(with(0.5, 1.2, 1.0, 1.3), Balance::Impossible, "only depression is reached");
        assert_eq!(with(0.0, 0.0, 1.0, 1.3), Balance::Any, "blocked calcium");
    }

    /// Fig. 2A: the six Table S1 curves, 60 pairs at 1 Hz, on the Fig. 2 script's own 2001-point
    /// grid, against the authors' code.
    ///
    /// Measured: every sampled point and each curve's extremes within `9.4 × 10⁻¹⁵`, and each grid's
    /// sum and sum of squares within `1.9 × 10⁻¹⁵` of the reference's `math.fsum`, relatively. Where
    /// the reference is `NaN` — 1122 DPD points and 1838 D points, where no calcium crosses a
    /// threshold — the change is exactly 1.
    #[test]
    fn fig_2a_is_the_authors_curve_on_their_grid() {
        let dts = linspace(-0.1, 0.1);
        let (mut worst, mut sums, mut squares) = (0.0_f64, 0.0_f64, 0.0_f64);
        for (k, s) in S1.iter().enumerate() {
            let (ys, driven) = curve(s, &dts, 1.0, 60.0, None);
            let (w, a, b) = against(&ys, &driven, &FIG2[k], FIG2_DIGEST[k]);
            (worst, sums, squares) = (worst.max(w), sums.max(a), squares.max(b));
        }
        assert!(worst < 3e-14 && sums < 6e-15 && squares < 6e-15, "{worst:e} {sums:e} {squares:e}");
    }

    /// Fig. 3: the hippocampal-slice set at 5 Hz — a pair 200 times, and a pre-spike and post-burst
    /// (11.5 ms) 100 and 30 times — against the authors' code on the Fig. 3 script's grid.
    ///
    /// Measured: samples and extremes within `1.2 × 10⁻¹⁵`, sums within `1.9 × 10⁻¹⁵` and sums of
    /// squares within `2.2 × 10⁻¹⁵`, relatively.
    #[test]
    fn fig_3_is_the_authors_curves() {
        let s = Synapse::HIPPOCAMPAL_SLICES;
        let dts = linspace(-0.1, 0.1);
        let (mut worst, mut sums, mut squares) = (0.0_f64, 0.0_f64, 0.0_f64);
        for (k, (n, burst)) in [(200.0, None), (100.0, Some(0.0115)), (30.0, Some(0.0115))].into_iter().enumerate() {
            let (ys, driven) = curve(&s, &dts, 5.0, n, burst);
            let (w, a, b) = against(&ys, &driven, &FIG3[k], FIG3_DIGEST[k]);
            (worst, sums, squares) = (worst.max(w), sums.max(a), squares.max(b));
        }
        assert!(worst < 4e-15 && sums < 6e-15 && squares < 7e-15, "{worst:e} {sums:e} {squares:e}");
    }

    /// Fig. 4: the cortical-slice set, 75 pairs, at 1, 20, 30, 40 and 50 Hz on the Fig. 4B script's
    /// grids (±100 ms at 1 Hz, half a period above), and at `Δt = ±10` ms from 0.1 to 50 Hz (Fig. 4A).
    ///
    /// Measured: samples and extremes within `3.0 × 10⁻¹⁵`, sums within `1.3 × 10⁻¹⁴` and squares
    /// within `8.0 × 10⁻¹⁵`, relatively, and the Fig. 4A points within `1.2 × 10⁻¹⁴`. At 50 Hz the
    /// calcium never falls below `θ_d`, and the fraction of time above it is exactly 1.
    #[test]
    fn fig_4_is_the_authors_curves() {
        let s = Synapse::CORTICAL_SLICES;
        let (mut worst, mut sums, mut squares) = (0.0_f64, 0.0_f64, 0.0_f64);
        for (k, &(f, end, ref samples)) in FIG4B.iter().enumerate() {
            let half = if f == 1.0 { 0.1 } else { 1.0 / f / 2.0 };
            assert_eq!(half, end, "the script's grid at {f} Hz");
            let (ys, driven) = curve(&s, &linspace(-half, half), f, 75.0, None);
            let (w, a, b) = against(&ys, &driven, samples, FIG4B_DIGEST[k]);
            (worst, sums, squares) = (worst.max(w), sums.max(a), squares.max(b));
        }
        let mut worst_a = 0.0_f64;
        for &(f, plus, minus) in &FIG4A {
            for (dt, want) in [(0.01, plus), (-0.01, minus)] {
                let got = s.change(s.pair_fractions(dt, f).unwrap(), 75.0 / f).unwrap();
                worst_a = worst_a.max((got - want).abs());
            }
        }
        assert!(worst < 1e-14 && sums < 4e-14 && squares < 3e-14, "{worst:e} {sums:e} {squares:e}");
        assert!(worst_a < 4e-14, "{worst_a:e}");
        assert_eq!(s.pair_fractions(0.0, 50.0).unwrap().alpha_d, 1.0);
    }

    /// Fig. 4B's caption: "potentiation only above 29 Hz for all Δt". Bisected on the Fig. 4B grid,
    /// 75 pairs, the lowest change over every offset reaches 1 at 29.1139 Hz, the frequency the
    /// authors' code gives to the last bisection step.
    #[test]
    fn every_offset_potentiates_above_29_hz() {
        let s = Synapse::CORTICAL_SLICES;
        let lowest = |f: f64| {
            let (ys, _) = curve(&s, &linspace(-0.5 / f, 0.5 / f), f, 75.0, None);
            ys.into_iter().fold(f64::INFINITY, f64::min)
        };
        let (mut lo, mut hi) = (25.0, 30.0);
        for _ in 0..40 {
            let mid = 0.5 * (lo + hi);
            if lowest(mid) < 1.0 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        assert_eq!((lo, hi), EVERY_OFFSET_POTENTIATES_HZ);
        assert!(lo > 29.0 && hi < 29.5);
        assert!(lowest(29.0) < 1.0 && lowest(29.5) > 1.0);
    }

    /// `spikePairFrequency` and `preSpikePostPair`'s fractions of time above threshold, against
    /// [`Synapse::pair_fractions`] and [`Synapse::burst_fractions`]: the six Table S1 sets at 1 Hz, the
    /// cortical-slice set at 20 Hz where successive transients interact, and the hippocampal-slice
    /// burst at 5 Hz across its three orderings (post–post–pre, post–pre–post, pre–post–post).
    ///
    /// The authors' code is a case analysis of the SI's regions; this module sums the periodic trace.
    /// Measured agreement: `4.7 × 10⁻¹⁷` at 1 Hz, `3.7 × 10⁻¹⁶` at 20 Hz, `5.6 × 10⁻¹⁷` for bursts.
    #[test]
    fn the_fractions_are_the_authors() {
        let (mut w1, mut w20, mut wb) = (0.0_f64, 0.0_f64, 0.0_f64);
        let close = |f: Fractions, d: f64, p: f64| (f.alpha_d - d).abs().max((f.alpha_p - p).abs());
        for (s, rows) in S1.iter().zip(&PAIR_ALPHAS) {
            for &(dt, d, p) in rows {
                w1 = w1.max(close(s.pair_fractions(dt, 1.0).unwrap(), d, p));
            }
        }
        for &(dt, d, p) in &PAIR_ALPHAS_20HZ {
            w20 = w20.max(close(Synapse::CORTICAL_SLICES.pair_fractions(dt, 20.0).unwrap(), d, p));
        }
        for &(dt, d, p) in &BURST_ALPHAS {
            wb = wb.max(close(Synapse::HIPPOCAMPAL_SLICES.burst_fractions(dt, 5.0, 2, 0.0115).unwrap(), d, p));
        }
        assert!(w1.max(w20).max(wb) < 1.2e-15, "{w1:e} {w20:e} {wb:e}");
    }

    /// The Ornstein–Uhlenbeck reduction against `changeInSynapticStrength`: `ρ̄`, `σ_ρ²`, `τ_eff`, `𝒰`,
    /// `𝒟`, the two means and the change, for three drives on each of two parameter sets.
    ///
    /// This module writes the mean and the spread with `φ(x) = −expm1(−x)/x` where the authors write
    /// `ρ̄ − (ρ̄ − ρ₀)e^{−T/τ_eff}` and `σ_ρ²(1 − e^{−2T/τ_eff})`. `𝒰` and `𝒟` are compared absolutely,
    /// because `1 ± erf` cancels in both codes alike, the rest relatively: measured, the worst
    /// disagreement is `4.7 × 10⁻¹⁶`.
    #[test]
    fn the_ou_reduction_is_the_authors() {
        let mut worst = 0.0_f64;
        for &(set, alpha_d, alpha_p, t, want) in &OU {
            let s = [Synapse::DP, Synapse::HIPPOCAMPAL_CULTURES][set];
            let f = Fractions { alpha_d, alpha_p };
            let got = [
                s.rho_bar(f).unwrap().unwrap(),
                s.sigma_rho_sq(f).unwrap().unwrap(),
                s.tau_eff(f).unwrap(),
                s.up(f, t, 0.0).unwrap(),
                s.down(f, t, 1.0).unwrap(),
                s.mean(f, t, 0.0).unwrap(),
                s.mean(f, t, 1.0).unwrap(),
                s.change(f, t).unwrap(),
            ];
            for (k, (g, w)) in got.iter().zip(want).enumerate() {
                let err = if k == 3 || k == 4 { (g - w).abs() } else { ((g - w) / w).abs() };
                worst = worst.max(err);
            }
            // The variance is half of `σ_ρ²(1 − e^{−2T/τ_eff})`, the authors' spread.
            let spread = want[1] * (1.0 - (-2.0 * t / want[2]).exp());
            worst = worst.max((s.variance(f, t).unwrap() / (spread / 2.0) - 1.0).abs());
        }
        assert!(worst < 1.5e-15, "{worst:e}");
    }

    /// [`Synapse::time_above`] against the authors' `eventBasedIntegration` on a 400-event irregular
    /// train of DP amplitudes: measured agreement `5.3 × 10⁻¹⁵` s on `θ_d` and exact on `θ_p`, out of
    /// 2.42 s and 1.41 s.
    #[test]
    fn time_above_is_the_authors_event_based_integration() {
        let s = Synapse::DP;
        let mut ev = Vec::new();
        let mut t = 0.0;
        for k in 1..=400u32 {
            t += 0.004 + 0.2 * ((f64::from(k) * 0.618_033_988_749_894_9) % 1.0);
            ev.push(Transient { time: t, amplitude: if k.is_multiple_of(3) { s.c_post } else { s.c_pre } });
        }
        assert_eq!(t, EVENT_BASED.0, "the same train");
        let (d, p) = s.time_above(&ev, t).unwrap();
        assert!((d - EVENT_BASED.1).abs() < 2e-14 && (p - EVENT_BASED.2).abs() < 2e-14, "{d} {p}");
        // A transient after `t_end` never arrives, and the order the train is given in does not matter.
        ev.push(Transient { time: t + 0.001, amplitude: 50.0 });
        assert_eq!(s.time_above(&ev, t).unwrap(), (d, p));
        ev.reverse();
        assert_eq!(s.time_above(&ev, t).unwrap(), (d, p));
    }

    /// An offset longer than one period is its remainder: at 5 Hz, 0.23 s is 0.03 s. The authors'
    /// `spikePairFrequency` folds it to −0.03 s instead.
    ///
    /// The reference returns `α_d = 0.10623` both at 0.23 s and at −0.03 s, and `0.07994` at 0.03 s;
    /// this module gives the 0.03 s value at 0.23 s, at 0.43 s and at 0.03 s, and agrees with the
    /// reference at ±0.03 s to the last bit. At −0.23 s both fold correctly, to −0.03 s.
    #[test]
    fn an_offset_longer_than_a_period_is_its_remainder() {
        let s = Synapse { delay: 0.0, ..Synapse::DP };
        let at = |x: f64| s.pair_fractions(x, 5.0).unwrap();
        let gap = |f: Fractions, (d, p): (f64, f64)| (f.alpha_d - d).abs().max((f.alpha_p - p).abs());
        assert_eq!(WRAP[0], WRAP[1], "the reference folds 0.23 s onto −0.03 s");
        assert!(gap(at(0.03), WRAP[2]) < 1e-16 && gap(at(-0.03), WRAP[1]) < 1e-16);
        for x in [0.23, 0.43] {
            assert!(gap(at(x), WRAP[2]) < 1e-15, "{x}: {:?}", at(x));
            assert!(gap(at(x), WRAP[0]) > 0.02);
        }
        assert!(gap(at(-0.23), WRAP[1]) < 1e-15);
    }

    /// The calcium trace is the sum of its decaying transients: SI eq. (19) for a post–pre pair and
    /// eq. (20), corrected, for a pre–post pair, in units of `τ_Ca`, on either side of each transient;
    /// at a transient's own time the jump has happened.
    ///
    /// Eq. (20) prints the trace after the postsynaptic spike as `e^{−t}(C_pre e^{Δt} + C_post)`. With
    /// the pre transient at 0 and the post one at `Δt` it is `C_pre e^{−t} + C_post e^{−(t − Δt)} =
    /// e^{−t}(C_pre + C_post e^{Δt})`: the factor `e^{Δt}` belongs to `C_post`, as it does to `C_post`
    /// in (19). As printed, the trace would jump at `Δt` by `C_pre(1 − e^{−Δt}) + C_post e^{−Δt}`
    /// rather than by `C_post`; this test finds the printed form off the trace by up to 0.92 on these
    /// four pairs, and the corrected one within `10⁻¹⁵` (measured `4.4 × 10⁻¹⁶`).
    #[test]
    fn calcium_is_eq_19_and_eq_20_corrected() {
        let (c_pre, c_post) = (0.8, 2.0_f64);
        let (mut worst, mut printed) = (0.0_f64, 0.0_f64);
        for dt in [-1.5, -0.25, 0.25, 1.5_f64] {
            let tr = [Transient { time: 0.0, amplitude: c_pre }, Transient { time: dt, amplitude: c_post }];
            for k in -40..=60 {
                let t = f64::from(k) * 0.1 + 0.0137;
                let got = calcium(1.0, &tr, t).unwrap();
                let want = if dt < 0.0 {
                    if t < dt {
                        0.0
                    } else if t < 0.0 {
                        c_post * (dt - t).exp()
                    } else {
                        (-t).exp() * (c_post * dt.exp() + c_pre)
                    }
                } else if t < 0.0 {
                    0.0
                } else if t < dt {
                    c_pre * (-t).exp()
                } else {
                    printed = printed.max((got - (-t).exp() * (c_pre * dt.exp() + c_post)).abs());
                    (-t).exp() * (c_pre + c_post * dt.exp())
                };
                worst = worst.max((got - want).abs());
            }
            // At the later transient's own time the trace is at its top, and just before at its foot.
            let (first, second) = if dt < 0.0 { (c_post, c_pre) } else { (c_pre, c_post) };
            let late = dt.max(0.0);
            let foot = first * (-dt.abs()).exp();
            assert!((calcium(1.0, &tr, late).unwrap() - (foot + second)).abs() < 1e-15);
            assert!((calcium(1.0, &tr, late - 1e-9).unwrap() - foot).abs() < 1e-8);
        }
        assert!(worst < 1e-15, "{worst:e}");
        assert!(printed > 0.9, "{printed}");
        // `τ_Ca` scales time, and nothing before the first transient is calcium.
        let tr = [Transient { time: 0.1, amplitude: 1.7 }];
        assert!((calcium(0.02, &tr, 0.1 + 0.02 * 3.0_f64.ln()).unwrap() - 1.7 / 3.0).abs() < 1e-15);
        assert_eq!(calcium(0.02, &tr, 0.099).unwrap(), 0.0);
        assert_eq!(calcium(0.02, &[], 5.0).unwrap(), 0.0);
    }

    /// SI eq. (21): a single pair, `C_pre < θ < C_post`, in units of `τ_Ca`, in each of its three
    /// regions, against [`fraction_above`] with a period so long that nothing interacts.
    ///
    /// The SI's example amplitudes (Fig. S7), `C_pre = 0.8`, `C_post = 2`, `θ = 1`, across
    /// `Δt ∈ [−6, 6]`: region I below `ln((θ − C_pre)/C_post) = −2.303`, III above `ln(θ/C_post) =
    /// −0.693` — and for positive `Δt` (pre first) III's expression still holds, since
    /// `ln(C_post/θ) + ln(1 + C_pre e^{−Δt}/C_post)` is the time after a post transient landing on a
    /// pre transient's tail.
    #[test]
    fn single_pairs_follow_si_eq_21() {
        let (c_pre, c_post, th) = (0.8, 2.0, 1.0_f64);
        let (edge1, edge2) = (((th - c_pre) / c_post).ln(), (th / c_post).ln());
        let (mut worst, mut seen) = (0.0_f64, [0; 3]);
        for k in -600..=600 {
            let dt = f64::from(k) / 100.0 + 0.003;
            let region = if dt < edge1 {
                0
            } else if dt <= edge2 {
                1
            } else {
                2
            };
            seen[region] += 1;
            let rise = c_post * dt.exp() + c_pre;
            let want = (c_post / th).ln()
                + match region {
                    0 => 0.0,
                    1 => (rise / th).ln(),
                    _ => (rise / (c_post * dt.exp())).ln(),
                };
            let period = 1e3;
            let tr = [Transient { time: 0.0, amplitude: c_pre }, Transient { time: dt, amplitude: c_post }];
            let got = fraction_above(1.0, period, &tr, th).unwrap() * period;
            worst = worst.max((got - want).abs());
        }
        assert!(seen.iter().all(|&n| n > 50), "{seen:?}");
        assert!(worst < 1e-13, "{worst:e}");
    }

    /// SI eqs. (22)–(33): single triplets, `C_pre < θ < C_post`, in units of `τ_Ca`. The feet and tops
    /// (22)–(26) and (28)–(32) are [`calcium`] just before and at each transient, and the times above
    /// threshold (27) for pre–post–pre and (33) for post–pre–post, in all three and all five of their
    /// regions, are [`fraction_above`] with a period so long that nothing interacts. Measured: the
    /// feet and tops exactly, the times within `5.1 × 10⁻¹⁴`; every region is visited at least 44
    /// times in the 1600 triplets of each kind.
    #[test]
    fn single_triplets_follow_si_eqs_22_to_33() {
        let (c_pre, c_post, th) = (0.8, 2.0, 1.0_f64);
        let period = 1e3;
        let at = |tr: &[Transient], k: usize| (calcium(1.0, &tr[..k], tr[k].time).unwrap(), calcium(1.0, &tr[..=k], tr[k].time).unwrap());
        let (mut worst_c, mut worst_a, mut ppp, mut pop) = (0.0_f64, 0.0_f64, [0; 3], [0; 5]);
        for i in 0..40 {
            let d1 = f64::from(i) * 0.1 + 0.017;
            let e1 = (-d1).exp();
            for j in 0..40 {
                let d2 = f64::from(j) * 0.1 + 0.023;
                let e2 = (-d2).exp();
                // Pre–post–pre: a pre `d1` before the post at 0, a pre `d2` after it.
                let tr = [
                    Transient { time: -d1, amplitude: c_pre },
                    Transient { time: 0.0, amplitude: c_post },
                    Transient { time: d2, amplitude: c_pre },
                ];
                let (big_a, big_b) = (c_pre * e1, c_pre * (-(d1 + d2)).exp() + c_post * e2);
                let (big_c, big_d, big_e) = (c_pre, big_a + c_post, big_b + c_pre);
                for (got, want) in [(at(&tr, 0).1, big_c), (at(&tr, 1).0, big_a), (at(&tr, 1).1, big_d), (at(&tr, 2).0, big_b), (at(&tr, 2).1, big_e)] {
                    worst_c = worst_c.max((got - want).abs());
                }
                let tail = c_post + c_pre * e1;
                let (region, alpha) = if d2 > (tail / (th - c_pre)).ln() {
                    (0, (big_d / th).ln())
                } else if d2 >= (tail / th).ln() {
                    (1, (big_d / th).ln() + (big_e / th).ln())
                } else {
                    (2, (big_e / th).ln() + d2)
                };
                ppp[region] += 1;
                worst_a = worst_a.max((fraction_above(1.0, period, &tr, th).unwrap() * period - alpha).abs());
                // Post–pre–post: a post `d1` before the pre at 0, a post `d2` after it.
                let tr = [
                    Transient { time: -d1, amplitude: c_post },
                    Transient { time: 0.0, amplitude: c_pre },
                    Transient { time: d2, amplitude: c_post },
                ];
                let (big_j, big_k) = (c_post * e1, c_post * (-(d1 + d2)).exp() + c_pre * e2);
                let (big_l, big_m, big_n) = (c_post, big_j + c_pre, big_k + c_post);
                for (got, want) in [(at(&tr, 0).1, big_l), (at(&tr, 1).0, big_j), (at(&tr, 1).1, big_m), (at(&tr, 2).0, big_k), (at(&tr, 2).1, big_n)] {
                    worst_c = worst_c.max((got - want).abs());
                }
                let (short1, short2) = (d1 <= (c_post / th).ln(), d2 <= ((c_post * e1 + c_pre) / th).ln());
                let (region, alpha) = if d1 > (c_post / (th - c_pre)).ln() {
                    (0, (big_l / th).ln() + (big_n / th).ln())
                } else if !short1 && !short2 {
                    (1, (big_l / th).ln() + (big_m / th).ln() + (big_n / th).ln())
                } else if short1 && !short2 {
                    (2, (big_m / th).ln() + d1 + (big_n / th).ln())
                } else if !short1 {
                    (3, (big_l / th).ln() + (big_n / th).ln() + d2)
                } else {
                    (4, (big_n / th).ln() + d1 + d2)
                };
                pop[region] += 1;
                worst_a = worst_a.max((fraction_above(1.0, period, &tr, th).unwrap() * period - alpha).abs());
            }
        }
        assert!(ppp.iter().chain(&pop).all(|&n| n > 10), "{ppp:?} {pop:?}");
        assert!(worst_c < 1e-15 && worst_a < 1.5e-13, "{worst_c:e} {worst_a:e}");
    }

    /// SI eqs. (34)–(44): pairs at a frequency `f` (in units of `1/τ_Ca`), `C_pre < θ < C_post`. The
    /// periodic feet and tops (34)–(37) and (40)–(43) are [`calcium`] once the start-up has decayed, and the times
    /// above threshold in all five regions of each ordering are [`fraction_above`] — with (39)'s
    /// regions II and IV read with their interval ends in order. `A(f) − 1` is formed as
    /// `1/(e^{1/f} − 1)`: at the lowest frequency here `1 − e^{−1/f}` is 1 in floating point.
    ///
    /// For post–pre pairs (39) bounds its regions by `f_E = −1/ln(1 − X/θ)`, where the pre transient's
    /// top `E` reaches `θ`, `f_C = −1/ln(1 − X/(θ + C_pre))`, where its foot `C` does, and `f_B`, where
    /// the post transient's foot `B` does. It prints II and IV as `f ∈ [f_C, f_E]`; this test finds
    /// `f_C > f_E` at every point where both exist, so that interval is empty, and reads it as
    /// `[f_E, f_C]`. Eq. (44), for pre–post pairs, has its interval ends in the right order (unlike
    /// (39)'s): `f_L`, where the pre transient's top `L` reaches `θ`, is below `f_K`, where the post
    /// transient's foot `K = L e^{−Δt}` does, at every point where both exist. Its intervals for II and IV lack
    /// the comma between their ends, though, and III and V carry a stray `]` (SI p. 22). Measured
    /// over 1457 offsets of each sign, every region visited at least 40 times, 488 post–pre points
    /// with `f_C > f_E` and all 1457 pre–post points with `f_L < f_K`: the feet and tops within
    /// `2.7 × 10⁻¹⁴` relatively, the times within `3.6 × 10⁻¹⁵`.
    #[test]
    fn pairs_at_a_frequency_follow_si_eqs_34_to_44() {
        let (c_pre, c_post, th) = (0.8, 2.0, 1.0);
        let bound = |x: f64| if x < 1.0 { -1.0 / (1.0 - x).ln() } else { 0.0 };
        let (mut worst_c, mut worst, mut post_pre, mut pre_post, mut reversed, mut in_order) = (0.0_f64, 0.0_f64, [0; 5], [0; 5], 0, 0);
        for j in 0..60 {
            let f = 0.0117 + f64::from(j) * 0.0197;
            let am1 = 1.0 / (1.0 / f).exp_m1();
            let a = 1.0 + am1;
            for k in 1..60 {
                let lag = f64::from(k) * 0.1 + 0.013;
                if lag >= 1.0 / f {
                    continue;
                }
                for dt in [-lag, lag] {
                    let (e_dt, e_mdt) = (dt.exp(), (-dt).exp());
                    let tr = [Transient { time: 0.0, amplitude: c_pre }, Transient { time: dt, amplitude: c_post }];
                    // The steady state's feet and tops, against the trace of a long train.
                    // Enough periods for the start-up to decay below `e^{−40}`, and no more, so that
                    // the times stay small and keep their precision.
                    let periods = 3 + (40.0 * f).ceil() as u32;
                    let train: Vec<Transient> = (0..periods)
                        .flat_map(|n| tr.map(|t| Transient { time: t.time + f64::from(n) / f, ..t }))
                        .collect();
                    let last = f64::from(periods - 1) / f;
                    let at = |time: f64| {
                        let before: Vec<Transient> = train.iter().copied().filter(|t| t.time < time).collect();
                        (calcium(1.0, &before, time).unwrap(), calcium(1.0, &train, time).unwrap())
                    };
                    let want = if dt < 0.0 {
                        let x = c_post * e_dt + c_pre;
                        let z = c_post + c_pre * e_mdt;
                        let (big_b, big_c) = (am1 * z, c_post * a * e_dt + am1 * c_pre);
                        let (big_d, big_e) = (a * c_post + am1 * c_pre * e_mdt, a * x);
                        let ((b0, d0), (c0, e0)) = (at(last + dt), at(last));
                        for (got, want) in [(b0, big_b), (c0, big_c), (d0, big_d), (e0, big_e)] {
                            worst_c = worst_c.max((got - want).abs() / want);
                        }
                        let (fe, fc, fb) = (bound(x / th), bound(x / (th + c_pre)), bound(z / (th + z)));
                        if x < th {
                            assert!(fc > fe, "(39) printed [f_C, f_E] = [{fc}, {fe}] is empty");
                            reversed += 1;
                        }
                        let (region, alpha) = if f < fe {
                            (0, (big_d / th).ln())
                        } else if f < fc && f < fb {
                            (1, (big_d / th).ln() + (big_e / th).ln())
                        } else if f >= fc && f < fb {
                            (2, (big_e / th).ln() + lag)
                        } else if f < fc {
                            (3, (big_d / th).ln() + 1.0 / f - lag)
                        } else {
                            (4, 1.0 / f)
                        };
                        post_pre[region] += 1;
                        alpha
                    } else {
                        let y = c_post * e_dt + c_pre;
                        let w = c_post + c_pre * e_mdt;
                        let (big_j, big_k) = (am1 * y, am1 * c_post + a * c_pre * e_mdt);
                        let (big_l, big_m) = (am1 * c_post * e_dt + a * c_pre, a * w);
                        let ((j0, l0), (k0, m0)) = (at(last), at(last + dt));
                        for (got, want) in [(j0, big_j), (k0, big_k), (l0, big_l), (m0, big_m)] {
                            worst_c = worst_c.max((got - want).abs() / want);
                        }
                        let (fl, fk, fj) = (bound(y / (th + c_post * e_dt)), bound(w / (th + c_post)), bound(y / (th + y)));
                        if y < th + c_post * e_dt && w < th + c_post {
                            assert!(fl < fk, "(44) printed [f_L, f_K] = [{fl}, {fk}] is in order");
                            in_order += 1;
                        }
                        let (region, alpha) = if f < fl {
                            (0, (big_m / th).ln())
                        } else if f < fk && f < fj {
                            (1, (big_l / th).ln() + (big_m / th).ln())
                        } else if f >= fk && f < fj {
                            (2, (big_m / th).ln() + lag)
                        } else if f < fk {
                            (3, (big_l / th).ln() + 1.0 / f - lag)
                        } else {
                            (4, 1.0 / f)
                        };
                        pre_post[region] += 1;
                        alpha
                    };
                    let got = fraction_above(1.0, 1.0 / f, &tr, th).unwrap() / f;
                    worst = worst.max((got - want).abs());
                }
            }
        }
        assert!(post_pre.iter().chain(&pre_post).all(|&n| n > 30), "{post_pre:?} {pre_post:?}");
        assert!(reversed > 400 && in_order > 1400, "{reversed} {in_order}");
        assert!(worst_c < 1e-13 && worst < 1.2e-14, "{worst_c:e} {worst:e}");
    }

    /// One isolated transient spends `τ_Ca ln(C/θ)` above `θ`; a train fast enough never drops below
    /// it and spends the whole period there; no transient, or one below `θ`, spends none. And the
    /// periodic sum does not care how the transients are listed or where the period starts.
    #[test]
    fn fraction_above_has_its_closed_forms_and_invariances() {
        let tr = [Transient { time: 0.3, amplitude: 2.7 }];
        let got = fraction_above(0.02, 10.0, &tr, 1.1).unwrap() * 10.0;
        assert!((got - 0.02 * (2.7_f64 / 1.1).ln()).abs() < 1e-17, "{got}");
        let fast = [Transient { time: 0.0, amplitude: 2.0 }];
        assert_eq!(fraction_above(0.02, 0.005, &fast, 1.3).unwrap(), 1.0, "at 200 Hz the calcium never falls to 1.3");
        assert_eq!(fraction_above(0.02, 1.0, &[], 1.0).unwrap(), 0.0);
        assert_eq!(fraction_above(0.02, 1.0, &[Transient { time: 0.1, amplitude: 0.9 }], 1.0).unwrap(), 0.0);
        let three = [
            Transient { time: 0.013, amplitude: 1.0 },
            Transient { time: 0.05, amplitude: 2.0 },
            Transient { time: 0.041, amplitude: 0.7 },
        ];
        let base = fraction_above(0.02, 0.1, &three, 1.3).unwrap();
        assert!(base > 0.0);
        let mut shuffled = three;
        shuffled.reverse();
        assert!((fraction_above(0.02, 0.1, &shuffled, 1.3).unwrap() - base).abs() < 1e-16);
        for shift in [0.037, 0.1, -0.29, 7.0] {
            let moved: Vec<Transient> = three.iter().map(|t| Transient { time: t.time + shift, ..*t }).collect();
            let got = fraction_above(0.02, 0.1, &moved, 1.3).unwrap();
            assert!((got - base).abs() < 1e-15, "shifted by {shift}: {got} against {base}");
        }
        // The gain `1/(1 − e^{−T/τ_Ca})` overflows where `T/τ_Ca` falls below `1/f64::MAX =
        // 5.56 × 10⁻³⁰⁹`, and is refused there rather than turned into `0 · ∞`; just above, a train of
        // zero amplitude spends no time above `θ` and one of positive amplitude all of it.
        let (zero, two) = ([Transient { time: 0.0, amplitude: 0.0 }], [Transient { time: 0.0, amplitude: 2.0 }]);
        assert!((1.0 / f64::MAX - 5.562_684_646e-309).abs() < 1e-318);
        assert_eq!((fraction_above(1.0, 5.7e-309, &zero, 1.0), fraction_above(1.0, 5.7e-309, &two, 1.0)), (Ok(0.0), Ok(1.0)));
        for tr in [zero, two] {
            assert!(matches!(fraction_above(1.0, 5.5e-309, &tr, 1.0), Err(CalciumError::NonFinite { value, .. }) if value == f64::INFINITY));
        }
    }

    /// The printed delays put the LTD-to-LTP transition near `Δt = 0`, and not at it.
    ///
    /// With `β = ρ* = ½` the change is exactly 1 where `γ_p α_p = γ_d α_d`, and the fractions depend
    /// on `Δt − D` alone, so the delay that puts the transition at `Δt = 0` solves one equation.
    /// Bisected, it is 13.8629 ms for DP — the closed form the module doc derives, `τ_Ca ln 2 =
    /// 13.8629436` ms, to `7.3 × 10⁻⁹` s, which is the printed `γ_p` rounded from eq. (17): with the
    /// unrounded ratio the root is the closed form to `2.4 × 10⁻¹⁷` s — and for Table S3's DP, 4.5271 ms for DPD,
    /// 2.6671 ms for DPD′ and 4.3996 ms for Table S3's DPD′. The printed 13.7, 13.8, 4.6, 2.2 and 4.3 ms
    /// are none of them the root rounded to their 0.1 ms, and with them the change is 1 at `Δt = D −
    /// root`: −0.16, −0.06, +0.07, −0.47 and −0.10 ms.
    #[test]
    fn the_printed_delays_are_near_the_balance_point_and_not_at_it() {
        let root = |s: Synapse, lo: f64, hi: f64| {
            let g = |d: f64| {
                let f = Synapse { delay: d, ..s }.pair_fractions(0.0, 1.0).unwrap();
                s.gamma_p * f.alpha_p - s.gamma_d * f.alpha_d
            };
            let (mut a, mut b) = (lo, hi);
            let below = g(a) < 0.0;
            assert_ne!(below, g(b) < 0.0, "{lo}..{hi} brackets the root");
            for _ in 0..80 {
                let m = 0.5 * (a + b);
                if (g(m) < 0.0) == below {
                    a = m;
                } else {
                    b = m;
                }
            }
            0.5 * (a + b)
        };
        let dp = root(Synapse::DP, 0.013, 0.0145);
        let closed = 0.02 * 2.0_f64.ln();
        let exact = root(Synapse { gamma_p: 200.0 * 2.0_f64.ln() / (2.0 / 1.3_f64).ln(), ..Synapse::DP }, 0.013, 0.0145);
        assert!((dp - closed).abs() < 2e-8, "{dp}");
        assert!((exact - closed).abs() < 1e-15, "{exact}");
        assert!((root(Synapse::FIG_S1_DP, 0.013, 0.0145) - dp).abs() < 1e-15, "the same calcium, the same ratio");
        for (s, lo, hi, want, shift) in [
            (Synapse::DP, 0.013, 0.0145, 13.8629e-3, -0.16e-3),
            (Synapse::FIG_S1_DP, 0.013, 0.0145, 13.8629e-3, -0.06e-3),
            (Synapse::DPD, 0.004, 0.005, 4.5271e-3, 0.07e-3),
            (Synapse::DPD_PRIME, 0.002, 0.003, 2.6671e-3, -0.47e-3),
            (Synapse::FIG_S1_DPD_PRIME, 0.004, 0.005, 4.3996e-3, -0.10e-3),
        ] {
            let r = root(s, lo, hi);
            assert!((r - want).abs() < 5e-8, "{r}");
            assert!((s.delay - r - shift).abs() < 5e-6, "{}", s.delay - r);
            assert_ne!((r * 1e4).round() / 1e4, s.delay, "the printed {} ms is the root rounded", s.delay * 1e3);
            let at = s.change(s.pair_fractions(s.delay - r, 1.0).unwrap(), 60.0).unwrap();
            assert!((at - 1.0).abs() < 1e-9, "{at}");
            let at_zero = s.change(s.pair_fractions(0.0, 1.0).unwrap(), 60.0).unwrap();
            assert!((at_zero - 1.0).abs() > 1e-4, "{at_zero}");
        }
    }

    /// Where no calcium crosses either threshold nothing drives `ρ` and nothing shakes it: `ρ̄` and
    /// `σ_ρ²` do not exist, `τ_eff` is infinite, the mean is where `ρ` started, the probabilities are 0
    /// — or ½ for a synapse that starts exactly on `ρ*` — and the change is exactly 1. The authors'
    /// code returns `NaN` for all of it.
    #[test]
    fn where_no_calcium_crosses_nothing_changes() {
        let s = Synapse::DP;
        let zero = Fractions { alpha_d: 0.0, alpha_p: 0.0 };
        assert_eq!(s.rho_bar(zero), Ok(None));
        assert_eq!(s.sigma_rho_sq(zero), Ok(None));
        assert_eq!(s.tau_eff(zero), Ok(f64::INFINITY));
        assert_eq!(s.mean(zero, 60.0, 0.3), Ok(0.3));
        assert_eq!(s.variance(zero, 60.0), Ok(0.0));
        assert_eq!((s.up(zero, 60.0, 0.0), s.down(zero, 60.0, 1.0)), (Ok(0.0), Ok(0.0)));
        assert_eq!(s.change(zero, 60.0), Ok(1.0));
        assert_eq!((s.up(zero, 60.0, 0.5), s.down(zero, 60.0, 0.5)), (Ok(0.5), Ok(0.5)));
        // With the calcium above threshold but no rates, ρ diffuses freely: the variance is
        // `σ²(α_p + α_d)t/τ`, the mean does not move — and a vanishing drive approaches that limit.
        let free = Synapse { gamma_d: 0.0, gamma_p: 0.0, ..s };
        let f = Fractions { alpha_d: 0.25, alpha_p: 0.125 };
        let diffusion = s.sigma * s.sigma * 0.375 * 60.0 / s.tau;
        assert!((free.variance(f, 60.0).unwrap() - diffusion).abs() < 1e-15);
        assert_eq!(free.mean(f, 60.0, 0.3), Ok(0.3));
        let faint = Synapse { gamma_d: 1e-9, gamma_p: 2e-9, ..s };
        assert!((faint.variance(f, 60.0).unwrap() / diffusion - 1.0).abs() < 1e-9);
        assert!((faint.mean(f, 60.0, 0.3).unwrap() - 0.3).abs() < 1e-9);
        // At `t = 0` nothing has happened yet.
        let g = Fractions { alpha_d: 0.02, alpha_p: 0.01 };
        assert_eq!((s.mean(g, 0.0, 0.3), s.variance(g, 0.0), s.up(g, 0.0, 0.0), s.change(g, 0.0)), (Ok(0.3), Ok(0.0), Ok(0.0), Ok(1.0)));
        // Fig. 2's D and DPD curves are driven only where the two transients' sum reaches `θ_d`:
        // `C(1 + e^{−|Δt − D|/τ_Ca}) ≥ θ_d`, `|Δt − D| ≤ τ_Ca ln(C/(θ_d − C))` — 8.11 ms for D, 43.94 ms
        // for DPD — and are exactly 1 outside.
        for (s, width) in [(Synapse::D, 8.109_302e-3), (Synapse::DPD, 43.944_492e-3)] {
            let edge = s.tau_ca * (s.c_pre / (s.theta_d - s.c_pre)).ln();
            assert!((edge - width).abs() < 1e-9, "{edge}");
            for x in [-1.001, -0.999, 0.999, 1.001] {
                let dt = s.delay + x * edge;
                let f = s.pair_fractions(dt, 1.0).unwrap();
                assert_eq!(f.alpha_d > 0.0, x.abs() < 1.0, "{dt}");
                assert_eq!(f.alpha_p, 0.0);
                if x.abs() > 1.0 {
                    assert_eq!(s.change(f, 60.0), Ok(1.0));
                }
            }
            assert!((s.change(s.pair_fractions(s.delay, 1.0).unwrap(), 60.0).unwrap() - 1.0).abs() > 0.1);
        }
    }

    /// `σ_ρ²` is twice the variance: after a long protocol [`Synapse::variance`] settles at `σ_ρ²/2`,
    /// and at any length it is `σ_ρ²(1 − e^{−2T/τ_eff})/2`. The mean settles at `ρ̄`, and `𝒰 + 𝒟 = 1`
    /// from any start.
    #[test]
    fn sigma_rho_squared_is_twice_the_variance() {
        let s = Synapse::HIPPOCAMPAL_CULTURES;
        let f = Fractions { alpha_d: 0.03, alpha_p: 0.017 };
        let (sr2, te, rb) = (s.sigma_rho_sq(f).unwrap().unwrap(), s.tau_eff(f).unwrap(), s.rho_bar(f).unwrap().unwrap());
        assert!((s.variance(f, 1e6).unwrap() / (sr2 / 2.0) - 1.0).abs() < 1e-14);
        assert!((s.mean(f, 1e6, 0.9).unwrap() - rb).abs() < 1e-14);
        for t in [0.5, 7.0, 60.0] {
            let want = sr2 * (1.0 - (-2.0 * t / te).exp()) / 2.0;
            assert!((s.variance(f, t).unwrap() / want - 1.0).abs() < 1e-14, "{t}");
            let m = rb - (rb - 0.2) * (-t / te).exp();
            assert!((s.mean(f, t, 0.2).unwrap() - m).abs() < 1e-15);
            for rho0 in [0.0, 0.2, 1.0] {
                assert!((s.up(f, t, rho0).unwrap() + s.down(f, t, rho0).unwrap() - 1.0).abs() < 1e-15);
            }
        }
    }

    /// The change in strength at its extremes, noise-free: every DOWN synapse switched up gives
    /// `b/(β + (1 − β)b)`, every UP one switched down `1/(β + (1 − β)b)` — `5/3` and `1/3` for
    /// Table S1, and `5.28145/2.284435` and `1/2.284435` at the hippocampal slices' `β = 0.7`.
    #[test]
    fn the_change_at_its_extremes() {
        for base in [Synapse::DP, Synapse::HIPPOCAMPAL_SLICES] {
            let s = Synapse { sigma: 0.0, ..base };
            let denom = s.beta + (1.0 - s.beta) * s.b;
            let potentiate = Fractions { alpha_d: 1.0, alpha_p: 1.0 };
            assert!(s.rho_bar(potentiate).unwrap().unwrap() > 0.5);
            assert_eq!((s.up(potentiate, 1e4, 0.0), s.down(potentiate, 1e4, 1.0)), (Ok(1.0), Ok(0.0)));
            assert!((s.change(potentiate, 1e4).unwrap() - s.b / denom).abs() < 1e-15);
            let depress = Fractions { alpha_d: 1.0, alpha_p: 0.0 };
            assert_eq!((s.up(depress, 1e4, 0.0), s.down(depress, 1e4, 1.0)), (Ok(0.0), Ok(1.0)));
            assert!((s.change(depress, 1e4).unwrap() - 1.0 / denom).abs() < 1e-15);
        }
        let dp = Synapse::DP;
        assert!((dp.b / (dp.beta + (1.0 - dp.beta) * dp.b) - 5.0 / 3.0).abs() < 1e-15);
        assert!((Synapse::HIPPOCAMPAL_SLICES.beta + 0.3 * Synapse::HIPPOCAMPAL_SLICES.b - 2.284_435).abs() < 1e-15);
    }

    /// SI §3.6 prints the change in strength as `[(1 − 𝒰)β + 𝒟(1 − β)] + (b[𝒰β + (1 − 𝒟)(1 − β)])/(β +
    /// [1 − β]b)` (SI p. 25), dividing only the UP term; the main text's Methods (p. 3996) and the
    /// authors' `changeInSynapticStrength` divide both. Where no synapse switches, `𝒰 = 𝒟 = 0`, the
    /// printed form gives `β + b(1 − β)/(β + (1 − β)b) = 4/3` for Table S1's `β = ½`, `b = 5`, and
    /// [`Synapse::change`] gives 1. On a driven protocol, 60 DP pairs at `Δt = +10` ms, the two
    /// forms differ too.
    #[test]
    fn si_3_6_as_printed_divides_only_the_up_term() {
        let s = Synapse::DP;
        let (bt, w) = (s.beta, s.b);
        let as_printed = |u: f64, d: f64| (1.0 - u) * bt + d * (1.0 - bt) + w * (u * bt + (1.0 - d) * (1.0 - bt)) / (bt + (1.0 - bt) * w);
        let zero = Fractions { alpha_d: 0.0, alpha_p: 0.0 };
        let (u, d) = (s.up(zero, 60.0, 0.0).unwrap(), s.down(zero, 60.0, 1.0).unwrap());
        assert_eq!((u, d), (0.0, 0.0));
        assert_eq!(s.change(zero, 60.0), Ok(1.0));
        assert!((as_printed(u, d) - 4.0 / 3.0).abs() < 1e-15);
        let f = s.pair_fractions(0.01, 1.0).unwrap();
        let (u, d) = (s.up(f, 60.0, 0.0).unwrap(), s.down(f, 60.0, 1.0).unwrap());
        assert!((as_printed(u, d) - s.change(f, 60.0).unwrap()).abs() > 0.1);
    }

    /// The transients of a spike train, and of the pair protocol: presynaptic ones delayed by `D`,
    /// all of them in time order.
    #[test]
    fn transients_are_delayed_and_ordered() {
        let s = Synapse::DP;
        let got = s.transients(&[0.5, 0.1], &[0.2]).unwrap();
        let want = [
            Transient { time: 0.1 + s.delay, amplitude: 1.0 },
            Transient { time: 0.2, amplitude: 2.0 },
            Transient { time: 0.5 + s.delay, amplitude: 1.0 },
        ];
        assert_eq!(got, want);
        let p = s.pairs(-0.02, 2.0, 3).unwrap();
        let times: Vec<f64> = p.iter().map(|t| t.time).collect();
        assert_eq!(times, [0.23, 0.25 + s.delay, 0.73, 0.75 + s.delay, 1.23, 1.25 + s.delay]);
        assert_eq!(p.iter().map(|t| t.amplitude).collect::<Vec<_>>(), [2.0, 1.0, 2.0, 1.0, 2.0, 1.0]);
        // Half a period either way is allowed, and fills the protocol exactly.
        assert_eq!(s.pairs(0.5, 1.0, 1).unwrap()[1].time, 1.0);
        assert_eq!(s.pairs(-0.5, 1.0, 1).unwrap()[0].time, 0.0);
    }

    /// Over a long periodic train the exact time above each threshold is the periodic fraction times
    /// the length — at 1 Hz from the first period, at 20 Hz once the start-up has decayed — with the
    /// thresholds in either order; and a simulation reports the same times. Measured: `1.4 × 10⁻¹⁶` at
    /// 1 Hz over 50 s, `2.5 × 10⁻¹⁴` at 20 Hz over the second 10 s.
    #[test]
    fn time_above_a_long_train_is_the_periodic_fraction() {
        let s = Synapse::DP;
        let tr = s.pairs(0.01, 1.0, 50).unwrap();
        let f = s.pair_fractions(0.01, 1.0).unwrap();
        let (d, p) = s.time_above(&tr, 50.0).unwrap();
        assert!((d / 50.0 - f.alpha_d).abs() < 5e-16 && (p / 50.0 - f.alpha_p).abs() < 5e-16, "{d} {p} {f:?}");
        for c in [Synapse::CORTICAL_SLICES, Synapse { theta_d: 1.3, theta_p: 1.0, ..Synapse::CORTICAL_SLICES }] {
            let tr = c.pairs(-0.01, 20.0, 400).unwrap();
            let f = c.pair_fractions(-0.01, 20.0).unwrap();
            let (d1, p1) = c.time_above(&tr, 10.0).unwrap();
            let (d2, p2) = c.time_above(&tr, 20.0).unwrap();
            assert!(((d2 - d1) / 10.0 - f.alpha_d).abs() < 1e-13 && ((p2 - p1) / 10.0 - f.alpha_p).abs() < 1e-13, "{f:?}");
            assert!(f.alpha_d != f.alpha_p);
        }
        let run = s.simulate(&tr, 50.0, 0.5, 1e-3, Noise::Corrected, &mut Rng::new(1)).unwrap();
        assert_eq!((run.above_d, run.above_p), (d, p));
    }

    /// One Euler–Maruyama step is eq. 1 term by term, with each noise term's factor: `√2` above both
    /// thresholds and 1 above one for the corrected term, 1 above either for the printed one, and no
    /// draw at all below both.
    ///
    /// A run shorter than the step and shorter than the first crossing is exactly one step, so its end
    /// is `ρ₀ + (h/τ)f(ρ₀) + σ√(h/τ) g ξ` with `ξ` the first normal a twin generator draws; `ρ* = 0.4`
    /// and `ρ₀ = 0.3` keep every term of the drift visible. It is bit for bit.
    #[test]
    fn one_step_is_eq_1_term_by_term() {
        let base = Synapse { rho_star: 0.4, ..Synapse::DP };
        let (rho0, t_end) = (0.3, 0.004);
        let tr = [Transient { time: 0.0, amplitude: 2.0 }];
        for (theta_d, theta_p, on_d, on_p) in [(1.0, 1.3, 1.0, 1.0), (1.0, 30.0, 1.0, 0.0), (30.0, 1.0, 0.0, 1.0), (30.0, 40.0, 0.0, 0.0)] {
            let s = Synapse { theta_d, theta_p, ..base };
            for (noise, g) in [(Noise::Corrected, f64::sqrt(on_d + on_p)), (Noise::Printed, (on_d + on_p).min(1.0))] {
                let mut rng = Rng::new(11);
                let got = s.simulate(&tr, t_end, rho0, 0.01, noise, &mut rng).unwrap().rho;
                let mut twin = Rng::new(11);
                let xi = if g > 0.0 { normal(&mut twin) } else { 0.0 };
                let cubic = -rho0 * (1.0 - rho0) * (s.rho_star - rho0);
                let drive = s.gamma_p * (1.0 - rho0) * on_p - s.gamma_d * rho0 * on_d;
                let want = rho0 + t_end / s.tau * (cubic + drive) + s.sigma * (t_end / s.tau).sqrt() * g * xi;
                assert_eq!(got, want, "θ_d = {theta_d}, θ_p = {theta_p}, {noise:?}");
                assert_eq!(rng, twin, "one draw above a threshold, none below both");
            }
        }
    }

    /// A simulation draws exactly one normal per step while the calcium is above a threshold, and
    /// none otherwise: a DP transient of 2 is above both thresholds for `0.02 ln(2/1.3) = 8.6` ms and
    /// above `θ_d` alone for `0.02 ln 1.3 = 5.2` ms, so at 1 ms steps a one-second run draws
    /// `9 + 6 = 15`, and a transient that stays below `θ_d` draws nothing.
    #[test]
    fn a_run_draws_one_normal_per_noisy_step() {
        let s = Synapse::DP;
        let tr = [Transient { time: 0.0, amplitude: 2.0 }];
        let steps = (0.02 * (2.0_f64 / 1.3).ln() / 1e-3).ceil() + (0.02 * 1.3_f64.ln() / 1e-3).ceil();
        assert_eq!(steps, 15.0);
        for noise in [Noise::Corrected, Noise::Printed] {
            let mut rng = Rng::new(3);
            s.simulate(&tr, 1.0, 0.2, 1e-3, noise, &mut rng).unwrap();
            let mut twin = Rng::new(3);
            for _ in 0..15 {
                let _ = normal(&mut twin);
            }
            assert_eq!(rng, twin);
            let mut quiet = Rng::new(3);
            s.simulate(&[Transient { time: 0.0, amplitude: 0.9 }], 1.0, 0.2, 1e-3, noise, &mut quiet).unwrap();
            assert_eq!(quiet, Rng::new(3));
        }
    }

    /// With no calcium, `ρ` follows the cubic alone, and Euler's method is first order against its
    /// closed forms.
    ///
    /// For `ρ* = ½` the substitution `w = (ρ − ½)⁻²` makes eq. 1 linear, `dw/dt = (4 − w)/(2τ)`, so
    /// `ρ(t) = ½ + sgn(ρ₀ − ½)/√(4 + ((ρ₀ − ½)⁻² − 4)e^{−t/(2τ)})`. For any `ρ*`, separating variables
    /// gives `G(ρ(t)) = G(ρ₀) − t/τ` with `G(ρ) = ln|ρ|/ρ* + ln|1 − ρ|/(1 − ρ*) −
    /// ln|ρ* − ρ|/(ρ*(1 − ρ*))`. From `ρ₀ = 0.3` over 300 s, steps of 4, 2 and 1 s leave errors in the
    /// ratios 2.004 and 2.002; at 1 ms the error is `2.2 × 10⁻⁸`, and `G` holds to `9.6 × 10⁻⁷` at
    /// `ρ* = 0.4`.
    #[test]
    fn without_calcium_rho_follows_the_cubic_and_euler_is_first_order() {
        let s = Synapse::DP;
        let (rho0, t) = (0.3, 300.0);
        let u0 = rho0 - 0.5_f64;
        let exact = 0.5 + u0.signum() / (4.0 + (1.0 / (u0 * u0) - 4.0) * (-t / (2.0 * s.tau)).exp()).sqrt();
        let run = |dt: f64, syn: &Synapse| syn.simulate(&[], t, rho0, dt, Noise::Corrected, &mut Rng::new(0)).unwrap().rho;
        let err: Vec<f64> = [4.0, 2.0, 1.0].iter().map(|&dt| (run(dt, &s) - exact).abs()).collect();
        let ratios: Vec<f64> = err.windows(2).map(|p| p[0] / p[1]).collect();
        let fine = (run(1e-3, &s) - exact).abs();
        let r4 = Synapse { rho_star: 0.4, ..s };
        let big_g = |r: f64| r.abs().ln() / 0.4 + (1.0 - r).abs().ln() / 0.6 - (0.4 - r).abs().ln() / 0.24;
        let end = run(1e-3, &r4);
        let resid = big_g(end) - big_g(rho0) + t / r4.tau;
        for ratio in ratios {
            assert!((1.99..2.01).contains(&ratio), "{err:?}");
        }
        assert!(fine < 7e-8, "{fine:e}");
        assert!(resid.abs() < 3e-6, "{resid:e}");
        // Starting on a fixed point, ρ stays there.
        for fixed in [0.0, 0.4, 1.0] {
            assert_eq!(r4.simulate(&[], t, fixed, 0.5, Noise::Corrected, &mut Rng::new(0)).unwrap().rho, fixed);
        }
    }

    /// With calcium and no noise, the simulation is first order too: twenty DP pairs at `Δt = 10` ms
    /// from `ρ = 0.3`, at steps of 4, 2 and 1 ms against a run at 1/64 ms, give errors of
    /// `7.1 × 10⁻⁴`, `3.4 × 10⁻⁴` and `1.8 × 10⁻⁴`, in the ratios 2.07 and 1.91.
    #[test]
    fn with_calcium_euler_is_first_order() {
        let s = Synapse { sigma: 0.0, ..Synapse::DP };
        let tr = s.pairs(0.01, 1.0, 20).unwrap();
        let run = |dt: f64| s.simulate(&tr, 20.0, 0.3, dt, Noise::Corrected, &mut Rng::new(0)).unwrap().rho;
        let reference = run(1e-3 / 64.0);
        let err: Vec<f64> = [4e-3, 2e-3, 1e-3].iter().map(|&dt| (run(dt) - reference).abs()).collect();
        let ratios: Vec<f64> = err.windows(2).map(|p| p[0] / p[1]).collect();
        for ratio in ratios {
            assert!((1.8..2.2).contains(&ratio), "{err:?}");
        }
        assert!((reference - 0.3).abs() > 0.01, "the pairs moved ρ: {reference}");
    }

    /// Runs `n` simulations from `rho0` and returns (fraction above `ρ*`, mean, variance) of `ρ(t_end)`.
    fn ensemble(s: &Synapse, tr: &[Transient], t_end: f64, rho0: f64, dt: f64, noise: Noise, n: u32, seed: u64) -> (f64, f64, f64) {
        let mut rng = Rng::new(seed);
        let (mut above, mut m1, mut m2) = (0u32, 0.0, 0.0);
        for _ in 0..n {
            let r = s.simulate(tr, t_end, rho0, dt, noise, &mut rng).unwrap().rho;
            above += u32::from(r > s.rho_star);
            m1 += r;
            m2 += r * r;
        }
        let nf = f64::from(n);
        (f64::from(above) / nf, m1 / nf, m2 / nf - (m1 / nf) * (m1 / nf))
    }

    /// [`ensemble`] split over `threads` seeds run in parallel, pooled.
    fn pooled(s: &Synapse, tr: &[Transient], rho0: f64, noise: Noise, runs: u32, threads: u32, seed: u64) -> (f64, f64, f64) {
        let parts: Vec<(f64, f64, f64)> = std::thread::scope(|scope| {
            let jobs: Vec<_> = (0..threads)
                .map(|w| scope.spawn(move || ensemble(s, tr, 60.0, rho0, 2e-3, noise, runs / threads, seed + u64::from(w))))
                .collect();
            jobs.into_iter().map(|j| j.join().unwrap()).collect()
        });
        let k = f64::from(threads);
        let up = parts.iter().map(|p| p.0).sum::<f64>() / k;
        let mean = parts.iter().map(|p| p.1).sum::<f64>() / k;
        let second = parts.iter().map(|p| p.2 + p.1 * p.1).sum::<f64>() / k;
        (up, mean, second - mean * mean)
    }

    /// The exact mean and variance of `ρ(t_end)` for eq. 1 WITHOUT its cubic, under the same
    /// transients and the corrected noise: between crossings the rates are constant, so the linear
    /// equation is an OU process on each interval, and its Gaussian is carried across them exactly.
    fn pulsed(s: &Synapse, tr: &[Transient], t_end: f64, rho0: f64) -> (f64, f64) {
        let (mut m, mut v) = (rho0, 0.0);
        s.walk(tr, t_end, |len, on_d, on_p| {
            let (gd, gp) = (if on_d { s.gamma_d } else { 0.0 }, if on_p { s.gamma_p } else { 0.0 });
            let q = s.sigma * s.sigma * (f64::from(u8::from(on_d)) + f64::from(u8::from(on_p))) / s.tau;
            let k = (gd + gp) / s.tau;
            if k > 0.0 {
                let target = gp / (gd + gp);
                let e = (-k * len).exp();
                m = target + (m - target) * e;
                v = v * e * e + q * (1.0 - e * e) / (2.0 * k);
            } else {
                v += q * len;
            }
        })
        .unwrap();
        (m, v)
    }

    /// Where the drive is constant, eq. 1 is the OU process (7) up to its cubic, and the simulation
    /// matches (13), (15), the mean and the variance within sampling error.
    ///
    /// One transient of `1.01 θ_p e^{T/τ_Ca}` keeps the calcium above both thresholds for the whole
    /// `T = 0.377` s run, so `α_d = α_p = 1` exactly, and the cubic is `10⁻⁴` of the DP drive. Over
    /// 10 000 runs each way, at 1 ms steps, measured: `𝒰` 0.3423 against 0.3392, `𝒟` 0.0329 against
    /// 0.0325, the mean 0.4509 against 0.4506 and the variance 0.01438 against 0.01422 — 0.65, 0.21,
    /// 0.30 and 0.83 standard errors. With the printed noise the variance halves, as `α_p + α_d = 2`
    /// becomes 1: 0.00707 against 0.00711. The variance is NOT `σ_ρ²(1 − e^{−2T/τ_eff})`, twice it.
    #[test]
    fn where_the_drive_is_constant_the_simulation_is_the_ou_process() {
        let s = Synapse::DP;
        let t = 0.377;
        let tr = [Transient { time: 0.0, amplitude: 1.01 * s.theta_p * (t / s.tau_ca).exp() }];
        let all = Fractions { alpha_d: 1.0, alpha_p: 1.0 };
        assert_eq!(s.time_above(&tr, t).unwrap(), (t, t), "above both thresholds throughout");
        let n = 10_000;
        let (up, mean, var) = ensemble(&s, &tr, t, 0.0, 1e-3, Noise::Corrected, n, 21);
        let (from_up, _, _) = ensemble(&s, &tr, t, 1.0, 1e-3, Noise::Corrected, n, 22);
        let (_, _, var_printed) = ensemble(&s, &tr, t, 0.0, 1e-3, Noise::Printed, n, 23);
        let nf = f64::from(n);
        let (want_up, want_down) = (s.up(all, t, 0.0).unwrap(), s.down(all, t, 1.0).unwrap());
        let (want_mean, want_var) = (s.mean(all, t, 0.0).unwrap(), s.variance(all, t).unwrap());
        let se = |p: f64| (p * (1.0 - p) / nf).sqrt();
        let se_var = want_var * (2.0 / nf).sqrt();
        assert!((up - want_up).abs() < 4.0 * se(want_up), "{up} {want_up}");
        assert!((1.0 - from_up - want_down).abs() < 4.0 * se(want_down), "{from_up} {want_down}");
        assert!((mean - want_mean).abs() < 4.0 * (want_var / nf).sqrt(), "{mean} {want_mean}");
        assert!((var - want_var).abs() < 4.0 * se_var, "{var} {want_var}");
        assert!((var_printed - want_var / 2.0).abs() < 4.0 * se_var / 2.0, "{var_printed}");
        assert!((var - 2.0 * want_var).abs() > 40.0 * se_var, "σ_ρ²(1 − e^(−2T/τ_eff)) is twice the variance");
    }

    /// Eq. (13) is exact for the OU process (7), which averages the drive over a period and drops
    /// the cubic. Both are measured here, on 60 DP pairs at 1 Hz.
    ///
    /// The drive arrives in pulses, and [`pulsed`] carries the Gaussian of eq. 1 without its cubic
    /// across them exactly. Scaling `γ_d`, `γ_p` and `τ` by 100 and `σ` by 10 leaves (7) and the pulsed
    /// process unchanged and shrinks the cubic a hundredfold, and there the simulation IS the pulsed
    /// process: 10 000 runs from each state at 2 ms steps put `𝒰`, `𝒟`, the mean and the variance
    /// within 3.5 standard errors of it at `Δt = −20` ms (measured 0.58, 0.81, 0.85 and 0.45).
    /// Against the pulsed process, (13) overestimates `𝒰` by 0.0059 at `Δt = +10` ms (0.6440 against
    /// 0.6381) and 0.0059 at `−20` ms (0.2444 against 0.2385), and the mean by 0.0019 and 0.0026. At
    /// the paper's own parameters the cubic then moves the noise-free end of the protocol by
    /// `+4.9 × 10⁻⁴` at `+10` ms, where `ρ̄ = 0.55` is near `ρ*`, and by `−3.95 × 10⁻³` at `−20` ms,
    /// towards `ρ = 0`; scaled, by `3 × 10⁻⁵` and `2 × 10⁻⁵`, which is the step.
    #[test]
    fn eq_13_averages_a_pulsed_drive_and_drops_the_cubic() {
        let base = Synapse::DP;
        let scaled = Synapse { gamma_d: base.gamma_d * 100.0, gamma_p: base.gamma_p * 100.0, tau: base.tau * 100.0, sigma: base.sigma * 10.0, ..base };
        let gauss_up = |(m, v): (f64, f64)| 0.5 * (1.0 + erf((m - 0.5) / (2.0 * v).sqrt()));
        let gauss_down = |(m, v): (f64, f64)| 0.5 * (1.0 - erf((m - 0.5) / (2.0 * v).sqrt()));
        for (dt, ou_up, pulsed_up, ou_gap, pull) in [(0.01, 0.6440, 0.6381, 0.0019, 4.9e-4), (-0.02, 0.2444, 0.2385, 0.0026, -3.95e-3)] {
            let tr = base.pairs(dt, 1.0, 60).unwrap();
            let f = base.pair_fractions(dt, 1.0).unwrap();
            for s in [base, scaled] {
                let (from_down, from_up) = (pulsed(&s, &tr, 60.0, 0.0), pulsed(&s, &tr, 60.0, 1.0));
                assert!((s.up(f, 60.0, 0.0).unwrap() - ou_up).abs() < 5e-5);
                assert!((gauss_up(from_down) - pulsed_up).abs() < 5e-5);
                assert!((s.mean(f, 60.0, 0.0).unwrap() - from_down.0 - ou_gap).abs() < 5e-5);
                assert!(gauss_down(from_up) > 0.3);
            }
            // Without noise the simulation follows the pulsed mean but for the cubic.
            let quiet = |s: Synapse| Synapse { sigma: 0.0, ..s }.simulate(&tr, 60.0, 0.0, 2.5e-4, Noise::Corrected, &mut Rng::new(0)).unwrap().rho;
            let (cubic_paper, cubic_scaled) = (quiet(base) - pulsed(&base, &tr, 60.0, 0.0).0, quiet(scaled) - pulsed(&scaled, &tr, 60.0, 0.0).0);
            assert!((cubic_paper - pull).abs() < 1e-5 && cubic_scaled.abs() < 5e-5, "{cubic_paper:e} {cubic_scaled:e}");
        }
        let tr = scaled.pairs(-0.02, 1.0, 60).unwrap();
        let (want_down, want_up) = (pulsed(&scaled, &tr, 60.0, 0.0), pulsed(&scaled, &tr, 60.0, 1.0));
        let n = 10_000;
        let (up, mean, var) = pooled(&scaled, &tr, 0.0, Noise::Corrected, n, 8, 40);
        let (from_up, _, _) = pooled(&scaled, &tr, 1.0, Noise::Corrected, n, 8, 50);
        let nf = f64::from(n);
        let se = |p: f64| (p * (1.0 - p) / nf).sqrt();
        let z = [
            (up - gauss_up(want_down)) / se(gauss_up(want_down)),
            (1.0 - from_up - gauss_down(want_up)) / se(gauss_down(want_up)),
            (mean - want_down.0) / (want_down.1 / nf).sqrt(),
            (var - want_down.1) / (want_down.1 * (2.0 / nf).sqrt()),
        ];
        assert!(z.iter().all(|z| z.abs() < 3.5), "{z:?}");
    }

    /// The paper's own protocol — 60 DP pairs at 1 Hz, `Δt = +10` ms — simulated 4000 times from each
    /// state lands within sampling error of (13) and (15), the gaps above notwithstanding.
    ///
    /// Measured at 2 ms steps: `𝒰` 0.6340 against 0.6440 and `𝒟` 0.3220 against 0.3119, 1.32 and
    /// 1.37 standard errors. At the 1000 runs the paper used, a standard error is 0.015.
    #[test]
    fn the_papers_protocol_is_within_sampling_error_of_eq_13() {
        let s = Synapse::DP;
        let tr = s.pairs(0.01, 1.0, 60).unwrap();
        let f = s.pair_fractions(0.01, 1.0).unwrap();
        let n = 4000;
        let (up, _, _) = pooled(&s, &tr, 0.0, Noise::Corrected, n, 4, 31);
        let (from_up, _, _) = pooled(&s, &tr, 1.0, Noise::Corrected, n, 4, 35);
        let (want_up, want_down) = (s.up(f, 60.0, 0.0).unwrap(), s.down(f, 60.0, 1.0).unwrap());
        let se = |p: f64| (p * (1.0 - p) / f64::from(n)).sqrt();
        assert!((up - want_up).abs() < 3.0 * se(want_up));
        assert!((1.0 - from_up - want_down).abs() < 3.0 * se(want_down));
    }

    /// The parameter files beside the authors' simulation output hold Table S1 with eq. 1 divided
    /// through by 50 — `τ = 3000` ms, `γ/50`, `σ/√50`, the cubic multiplied by `0.02` — in
    /// milliseconds; except DPD′'s, which is DPD's.
    ///
    /// To the printed digits: `50 × 5.14893 = 257.4465` for P's `γ_p = 257.447`, and D's `0.8√50 =
    /// 5.65685` for its `σ = 5.6568`, which is twice the other curves' rounded `2.8284`.
    #[test]
    fn the_authors_parameter_files_are_table_s1_scaled_by_50_but_dpd_primes_is_dpds() {
        for (k, (s, par)) in S1.iter().zip(PAR_FILES).enumerate() {
            let [c_pre, c_post, tau_pre, th_d, th_p, dephos, phos, sigma, tau_rho, eps, delay] = par;
            if k == 2 {
                assert_eq!(par, PAR_FILES[1], "output/DPDprime_curve holds DPD's parameters");
                assert!(s.c_pre != c_pre && s.c_post != c_post && s.theta_p != th_p && s.gamma_d != 50.0 * dephos);
                continue;
            }
            assert_eq!([c_pre, c_post, th_d, th_p], [s.c_pre, s.c_post, s.theta_d, s.theta_p], "{s:?}");
            assert!((tau_pre * 1e-3 - s.tau_ca).abs() < 1e-17 && (delay * 1e-3 - s.delay).abs() < 1e-17);
            assert_eq!((50.0 * dephos, 50.0 * tau_rho * 1e-3, 50.0 * eps), (s.gamma_d, s.tau, 1.0));
            assert!((50.0 * phos - s.gamma_p).abs() < 6e-4, "{phos}");
            assert!((sigma * 50f64.sqrt() - s.sigma).abs() < 6e-5, "{sigma}");
        }
    }

    /// The DPD′ simulation's output is the DPD′ curve although its parameter file is DPD's: its change
    /// in strength (column 5, 41 points) is within 0.0087 root-mean-square of the analytic DPD′ curve
    /// and 0.0775 of the DPD one.
    #[test]
    fn the_dpd_prime_output_is_the_dpd_prime_curve() {
        let rms = |s: Synapse| {
            let sq: f64 = SIM_DPD_PRIME_CHANGE.iter().map(|&(dt_ms, theirs, _)| (s.change(s.pair_fractions(dt_ms * 1e-3, 1.0).unwrap(), 60.0).unwrap() - theirs).powi(2)).sum();
            (sq / 41.0).sqrt()
        };
        let (prime, plain) = (rms(Synapse::DPD_PRIME), rms(Synapse::DPD));
        assert!(prime < 0.012 && plain > 0.06, "{prime} {plain}");
    }

    /// The simulation output committed with the code fits the corrected noise, not the printed noise
    /// the committed C++ computes — provided the DP, P and D′ runs shared one `σ`.
    ///
    /// `output/{DP,P}_curve/final_camkII_state.dat` hold `𝒰` (column 7) and `−𝒟` (column 9) at 41
    /// offsets each, 1000 runs of the C++ per cell at 0.1 ms steps. Its protocol is [`Synapse::pairs`]
    /// at 1 Hz moved `D` earlier — the first presynaptic transient at 500 ms, 60 pairs, the run ending
    /// at 60 s — with `ρ > 0.5` counted UP and `ρ ≤ 0.5` DOWN (`stdp_noisy.cpp`, `motif.cpp`).
    /// Simulated here 1000 times per cell at 2 ms steps, the two-sample `χ²` over the 164 cells, against
    /// 164 ± 18 by chance, is 142.9 for the corrected term, 444.1 for the printed one and
    /// 526.4 for `σ(β/dephos + α/phos)`, the second of the three alternatives commented out in
    /// `motif.cpp`.
    ///
    /// These two curves do not pin `σ`. The printed term with `σ × 1.22` gives 148.7 on them, and
    /// the sum with `σ × 0.75` gives 128.8: each fits as well as the corrected term. The D′ curve
    /// is what separates them. Its calcium never reaches `θ_p = 3.5` (`C_pre + C_post = 3`), so the
    /// three terms are one term there and draw the same numbers; at the files' `σ` its 82 cells
    /// give `χ² = 92.1` against 82 ± 13, at `σ × 1.22` 626.8 and at `σ × 0.75` 908.0 (over the 80
    /// cells where the two samples are not both all UP or all DOWN). The corrected term is
    /// therefore favoured only if the three runs shared `σ`, as their parameter files say (0.4 in
    /// each) — and those files are not a reliable record: DPD′'s is DPD's.
    #[test]
    fn the_committed_simulation_output_fits_the_corrected_noise_if_the_runs_shared_sigma() {
        // Two cheap preconditions first, so that a simulation broken in either way fails here in
        // milliseconds rather than after the 3 × 10¹⁰ steps below: over each 60 s protocol the calcium
        // spends 60 periods' worth of [`fraction_above`]'s time above each threshold (measured within
        // `3.9 × 10⁻¹⁴` s), and a minute without calcium draws no random number.
        let mut worst = 0.0_f64;
        for (s, table) in [(Synapse::DP, &SIM_DP[..]), (Synapse::P, &SIM_P[..]), (Synapse::D_PRIME, &SIM_D_PRIME[..])] {
            for &(dt_ms, _, _) in table {
                let (d, p) = s.time_above(&s.pairs(dt_ms * 1e-3, 1.0, 60).unwrap(), 60.0).unwrap();
                let f = s.pair_fractions(dt_ms * 1e-3, 1.0).unwrap();
                worst = worst.max((d - 60.0 * f.alpha_d).abs()).max((p - 60.0 * f.alpha_p).abs());
            }
        }
        assert!(worst < 1e-13, "{worst:e}");
        let mut quiet = Rng::new(0);
        Synapse::DP.simulate(&[], 60.0, 0.0, 2e-3, Noise::Corrected, &mut quiet).unwrap();
        assert_eq!(quiet, Rng::new(0), "a minute without calcium draws nothing");
        type Cell = (Synapse, f64, bool, f64);
        type Curve<'a> = (Synapse, &'a [(f64, f64, f64)]);
        let with_sigma = |s: Synapse, k: f64| Synapse { sigma: s.sigma * k, ..s };
        let cells = |k: f64, curves: &[Curve]| -> Vec<Cell> {
            curves
                .iter()
                .flat_map(|&(s, table)| table.iter().flat_map(move |&(dt_ms, up, down)| [(with_sigma(s, k), dt_ms, true, up), (with_sigma(s, k), dt_ms, false, down)]))
                .collect()
        };
        // One cell's contribution to the two-sample `χ²`, `None` where both samples are all UP or all
        // DOWN; cell `i` of every job draws from seed `1000 + i`.
        let term = |&(s, dt_ms, is_up, theirs): &Cell, noise: Noise, i: usize| -> Option<f64> {
            let tr = s.pairs(dt_ms * 1e-3, 1.0, 60).unwrap();
            let rho0 = if is_up { 0.0 } else { 1.0 };
            let (above, _, _) = ensemble(&s, &tr, 60.0, rho0, 2e-3, noise, 1000, 1000 + i as u64);
            let ours = if is_up { above } else { 1.0 - above };
            let p = 0.5 * (ours + theirs);
            (p > 0.0 && p < 1.0).then(|| (ours - theirs).powi(2) / (p * (1.0 - p) * 2.0 / 1000.0))
        };
        let main = |k: f64| cells(k, &[(Synapse::DP, &SIM_DP), (Synapse::P, &SIM_P)]);
        let control = |k: f64| cells(k, &[(Synapse::D_PRIME, &SIM_D_PRIME)]);
        let jobs: Vec<(Vec<Cell>, Noise)> = vec![
            (main(1.0), Noise::Corrected),
            (main(1.0), Noise::Printed),
            (main(1.0), Noise::CommentedSum),
            (main(1.22), Noise::Printed),
            (main(0.75), Noise::CommentedSum),
            (control(1.0), Noise::Corrected),
            (control(1.22), Noise::Printed),
            (control(0.75), Noise::CommentedSum),
        ];
        let tasks: Vec<(usize, usize)> = jobs.iter().enumerate().flat_map(|(j, (c, _))| (0..c.len()).map(move |i| (j, i))).collect();
        let threads = std::thread::available_parallelism().map_or(4, std::num::NonZeroUsize::get);
        let done: Vec<(usize, usize, Option<f64>)> = std::thread::scope(|scope| {
            let (jobs, tasks) = (&jobs, &tasks);
            let workers: Vec<_> = (0..threads)
                .map(|w| scope.spawn(move || tasks.iter().skip(w).step_by(threads).map(|&(j, i)| (j, i, term(&jobs[j].0[i], jobs[j].1, i))).collect::<Vec<_>>()))
                .collect();
            workers.into_iter().flat_map(|h| h.join().unwrap()).collect()
        });
        // Each job's cells in order, so that the sums do not depend on how many threads ran them.
        let per_cell: Vec<Vec<Option<f64>>> = (0..jobs.len())
            .map(|j| {
                let mut v = vec![None; jobs[j].0.len()];
                for &(_, i, x) in done.iter().filter(|d| d.0 == j) {
                    v[i] = x;
                }
                v
            })
            .collect();
        let chi2: Vec<(f64, usize)> = per_cell.iter().map(|v| (v.iter().flatten().sum(), v.iter().flatten().count())).collect();
        let [corrected, printed, sum, printed_122, sum_075, c_corrected, c_122, c_075] = chi2[..] else { unreachable!() };
        assert_eq!([corrected.1, printed.1, sum.1, printed_122.1, sum_075.1], [164; 5]);
        assert_eq!([c_corrected.1, c_122.1, c_075.1], [82, 82, 80]);
        let (fits, main_rejected) = (164.0 + 3.0 * 18.0, 164.0 + 10.0 * 18.0);
        let (control_fits, control_rejected) = (82.0 + 3.0 * 13.0, 82.0 + 10.0 * 13.0);
        assert!(corrected.0 < fits, "{corrected:?}");
        assert!(printed.0 > main_rejected && sum.0 > main_rejected, "{printed:?} {sum:?}");
        assert!(printed_122.0 < fits && sum_075.0 < fits, "{printed_122:?} {sum_075:?}");
        // On D′ no protocol reaches `θ_p`, so every noise factor is `Θ[c − θ_d]` and the terms coincide.
        for &(dt_ms, _, _) in &SIM_D_PRIME {
            let s = Synapse::D_PRIME;
            assert_eq!(s.time_above(&s.pairs(dt_ms * 1e-3, 1.0, 60).unwrap(), 60.0).unwrap().1, 0.0);
        }
        assert!(c_corrected.0 < control_fits, "{c_corrected:?}");
        assert!(c_122.0 > control_rejected && c_075.0 > control_rejected, "{c_122:?} {c_075:?}");
    }

    /// Every refusal, rendered.
    #[test]
    fn every_refusal_names_what_it_refused() {
        let s = Synapse::DP;
        let tr = [Transient { time: 0.0, amplitude: 2.0 }];
        let ok = Fractions { alpha_d: 0.1, alpha_p: 0.1 };
        let cases: Vec<(Result<f64, CalciumError>, &str)> = vec![
            (fraction_above(0.0, 1.0, &tr, 1.0), "tau_ca = 0 must be finite and positive"),
            (fraction_above(0.02, f64::NAN, &tr, 1.0), "period = NaN must be finite and positive"),
            (fraction_above(0.02, 1.0, &tr, -1.0), "theta = -1 must be finite and positive"),
            (fraction_above(0.02, 1.0, &[Transient { time: f64::INFINITY, amplitude: 1.0 }], 1.0), "transient time = inf is not finite"),
            (fraction_above(0.02, 1.0, &[Transient { time: 0.0, amplitude: -1.0 }], 1.0), "transient amplitude = -1 must be finite and non-negative"),
            (fraction_above(1e300, 1e-10, &[Transient { time: 0.0, amplitude: 0.0 }], 1.0), "steady-state gain 1/(1 - exp(-period/tau_ca)) = inf is not finite"),
            (calcium(-0.02, &tr, 1.0), "tau_ca = -0.02 must be finite and positive"),
            (calcium(0.02, &tr, f64::NAN), "t = NaN is not finite"),
            (calcium(0.02, &[Transient { time: f64::NEG_INFINITY, amplitude: 1.0 }], 1.0), "transient time = -inf is not finite"),
            (calcium(0.02, &[Transient { time: 2.0, amplitude: f64::INFINITY }], 1.0), "transient amplitude = inf must be finite and non-negative"),
            (s.pair_fractions(f64::NAN, 1.0).map(|f| f.alpha_d), "dt = NaN is not finite"),
            (s.pair_fractions(0.0, 0.0).map(|f| f.alpha_d), "frequency = 0 must be finite and positive"),
            (s.burst_fractions(0.0, 1.0, 2, -0.01).map(|f| f.alpha_d), "burst interval = -0.01 must be finite and non-negative"),
            (s.rho_bar(Fractions { alpha_d: 1.5, alpha_p: 0.0 }).map(|_| 0.0), "alpha_d = 1.5 must lie in [0, 1]"),
            (s.rates(Fractions { alpha_d: 0.0, alpha_p: -0.1 }).map(|r| r.0), "alpha_p = -0.1 must lie in [0, 1]"),
            (s.sigma_rho_sq(Fractions { alpha_d: f64::NAN, alpha_p: 0.0 }).map(|_| 0.0), "alpha_d = NaN must lie in [0, 1]"),
            (s.tau_eff(Fractions { alpha_d: 0.0, alpha_p: 2.0 }), "alpha_p = 2 must lie in [0, 1]"),
            (s.mean(ok, -1.0, 0.0), "t_total = -1 must be finite and non-negative"),
            (s.mean(ok, 1.0, f64::NAN), "rho0 = NaN is not finite"),
            (s.variance(ok, f64::INFINITY), "t_total = inf must be finite and non-negative"),
            (s.up(ok, 1.0, f64::INFINITY), "rho0 = inf is not finite"),
            (s.down(ok, -2.0, 1.0), "t_total = -2 must be finite and non-negative"),
            (s.change(ok, f64::NAN), "t_total = NaN must be finite and non-negative"),
            (s.time_above(&tr, -1.0).map(|r| r.0), "t_end = -1 must be finite and non-negative"),
            (s.time_above(&[Transient { time: -0.5, amplitude: 1.0 }], 1.0).map(|r| r.0), "transient time = -0.5 must be finite and non-negative"),
            (s.time_above(&[Transient { time: 0.5, amplitude: f64::NAN }], 1.0).map(|r| r.0), "transient amplitude = NaN must be finite and non-negative"),
            (s.transients(&[f64::NAN], &[]).map(|_| 0.0), "pre spike time = NaN is not finite"),
            (s.transients(&[], &[f64::NEG_INFINITY]).map(|_| 0.0), "post spike time = -inf is not finite"),
            (s.pairs(0.6, 1.0, 1).map(|_| 0.0), "dt = 0.6 must lie in [-1/(2f), 1/(2f)]"),
            (s.pairs(f64::NAN, 1.0, 1).map(|_| 0.0), "dt = NaN must lie in [-1/(2f), 1/(2f)]"),
            (s.pairs(0.0, -1.0, 1).map(|_| 0.0), "frequency = -1 must be finite and positive"),
        ];
        for (got, want) in cases {
            assert_eq!(got.unwrap_err().to_string(), want);
        }
        let mut rng = Rng::new(0);
        let sims = [
            (s.simulate(&tr, f64::NAN, 0.0, 1e-3, Noise::Corrected, &mut rng), "t_end = NaN must be finite and non-negative"),
            (s.simulate(&tr, 1.0, f64::INFINITY, 1e-3, Noise::Corrected, &mut rng), "rho0 = inf is not finite"),
            (s.simulate(&tr, 1.0, 0.0, 0.0, Noise::Corrected, &mut rng), "dt = 0 must be finite and positive"),
            (s.simulate(&tr, 1.5e8, 0.0, 1.0, Noise::Corrected, &mut rng), "t_end = 150000000 s at dt = 1 s is more than 100000000 Euler-Maruyama steps"),
            (s.simulate(&[Transient { time: -1.0, amplitude: 1.0 }], 1.0, 0.0, 1e-3, Noise::Corrected, &mut rng), "transient time = -1 must be finite and non-negative"),
        ];
        for (got, want) in sims {
            assert_eq!(got.unwrap_err().to_string(), want);
        }
        assert_eq!(rng, Rng::new(0), "a refused run draws nothing");
    }

    /// Every parameter is checked, by name, before anything is computed from it; and the boundaries
    /// are where the ranges say: `C_pre`, `σ`, `D` and the rates may be 0, `β` may be 0 or 1, `ρ*`
    /// may be neither.
    #[test]
    fn every_parameter_is_checked_and_named() {
        let base = Synapse::DP;
        let cases = [
            (Synapse { tau_ca: 0.0, ..base }, "tau_ca = 0 must be finite and positive"),
            (Synapse { c_pre: -0.1, ..base }, "c_pre = -0.1 must be finite and non-negative"),
            (Synapse { c_post: f64::NAN, ..base }, "c_post = NaN must be finite and non-negative"),
            (Synapse { theta_d: 0.0, ..base }, "theta_d = 0 must be finite and positive"),
            (Synapse { theta_p: f64::INFINITY, ..base }, "theta_p = inf must be finite and positive"),
            (Synapse { gamma_d: -1.0, ..base }, "gamma_d = -1 must be finite and non-negative"),
            (Synapse { gamma_p: f64::NAN, ..base }, "gamma_p = NaN must be finite and non-negative"),
            (Synapse { sigma: -2.0, ..base }, "sigma = -2 must be finite and non-negative"),
            (Synapse { tau: 0.0, ..base }, "tau = 0 must be finite and positive"),
            (Synapse { rho_star: 0.0, ..base }, "rho_star = 0 must lie in (0, 1)"),
            (Synapse { rho_star: 1.0, ..base }, "rho_star = 1 must lie in (0, 1)"),
            (Synapse { rho_star: f64::NAN, ..base }, "rho_star = NaN must lie in (0, 1)"),
            (Synapse { delay: -0.001, ..base }, "delay = -0.001 must be finite and non-negative"),
            (Synapse { beta: 1.5, ..base }, "beta = 1.5 must lie in [0, 1]"),
            (Synapse { b: 0.0, ..base }, "b = 0 must be finite and positive"),
        ];
        let tr = [Transient { time: 0.0, amplitude: 2.0 }];
        let f = Fractions { alpha_d: 0.1, alpha_p: 0.1 };
        for (s, want) in cases {
            assert_eq!(s.check().unwrap_err().to_string(), want);
            assert_eq!(s.pair_fractions(0.0, 1.0).unwrap_err().to_string(), want);
            assert_eq!(s.rates(f).unwrap_err().to_string(), want);
            assert_eq!(s.balance().unwrap_err().to_string(), want);
            assert_eq!(s.transients(&[0.0], &[]).unwrap_err().to_string(), want);
            assert_eq!(s.time_above(&tr, 1.0).unwrap_err().to_string(), want);
            assert_eq!(s.simulate(&tr, 1.0, 0.0, 1e-3, Noise::Printed, &mut Rng::new(0)).unwrap_err().to_string(), want);
        }
        let edges = [
            Synapse { c_pre: 0.0, sigma: 0.0, delay: 0.0, gamma_d: 0.0, gamma_p: 0.0, beta: 0.0, ..base },
            Synapse { beta: 1.0, rho_star: 1e-9, ..base },
            Synapse { rho_star: 1.0 - 1e-9, ..base },
        ];
        for s in edges {
            assert_eq!(s.check(), Ok(()), "{s:?}");
        }
        assert!(base.rates(Fractions { alpha_d: 1.0, alpha_p: 0.0 }).is_ok() && base.rates(Fractions { alpha_d: 0.0, alpha_p: 1.0 }).is_ok());
    }

    /// [`MAX_STEPS`] is a boundary and not an approximation of one: a run of exactly `10⁸` steps goes
    /// ahead (a quiet one, from `ρ = 0`, where the cubic is exactly 0), one step more is refused.
    #[test]
    fn max_steps_is_the_boundary() {
        let s = Synapse::DP;
        let dt = 1.0 / 1024.0;
        let t_end = MAX_STEPS * dt;
        assert_eq!(t_end / dt, MAX_STEPS);
        assert_eq!(s.simulate(&[], t_end, 0.0, dt, Noise::Corrected, &mut Rng::new(0)).unwrap().rho, 0.0);
        let over = (MAX_STEPS + 1.0) * dt;
        assert_eq!(s.simulate(&[], over, 0.0, dt, Noise::Corrected, &mut Rng::new(0)), Err(CalciumError::TooManySteps { t_end: over, dt }));
    }
}
