//! Electrical synapses between integrate-and-fire neurons: the ohmic junction, the spike it
//! carries across, and the phase-locked states the two decide between them — in Lewis and
//! Rinzel's weak-coupling phase model and in Chow and Kopell's exact spike-response theory, each
//! checked against its own closed forms, an independent integration, and its own figures.
//!
//! # The two papers
//!
//! Chow and Kopell, *Dynamics of spiking neurons with electrical coupling*, Neural Computation
//! 12:1643–1678, 2000, doi:10.1162/089976600300015295; and Lewis and Rinzel, *Dynamics of spiking
//! neurons connected by both inhibitory and electrical coupling*, Journal of Computational
//! Neuroscience 14:283–309, 2003, doi:10.1023/A:1023265027714. Both take leaky integrate-and-fire
//! cells in units where the reset is 0 and the threshold 1, `v = (V − V_reset)/(V_th − V_reset)`,
//! time in membrane time constants, and join them by an ohmic junction, `I_gap = g_c(v_j − v_i)`.
//! They differ in the one thing both titles are about: how the SPIKE crosses the junction.
//!
//! Lewis and Rinzel, eq. (5), for cell 1 (cell 2 by symmetry):
//!
//! ```text
//! dv₁/dt = −v₁ + I − g_s Σ s₁₂(t − t_{n,2}) + g_c[(v₂ − v₁) + β Σ δ(t − t_{n,2})]        (5)
//! ```
//!
//! The spike has no width: at threshold the cell resets to 0 at once, and its partner is kicked up
//! by `g_c β`, the charge `g_c ∫ V_spike dt` of a spike too brief to resolve (p. 286). The reset
//! reaches the partner only through the ohmic term, as the fall from 1 to 0. Inhibition is an
//! alpha function `s(t) = α²te^{−αt}` per spike ([`LewisRinzel`], [`LewisRinzelPair`]).
//!
//! Chow and Kopell, eqs. (2.3), (2.5) and (2.7):
//!
//! ```text
//! dv₁/dt = I₁ − v₁ − g(v₁ − v₂) + Σ A(t − t₁ˡ),   A(t) = v_A e^{ξt} on (0, Δ],  −(1 + v_M)δ(t − Δ)
//! ```
//!
//! The spike is a current of width `Δ` in the spiking cell; its partner feels it only through the
//! ohmic term, as the spiking cell's voltage climbs to its peak, and then feels the reset the same
//! way ([`ChowKopell`], [`ChowKopellNet`], `n` cells all to all in eq. 4.1). Lewis and Rinzel's
//! spikelet is a `δ` kick; Chow and Kopell's is a bump of width `Δ` followed by the reset's dip.
//! The first is the thin-spike limit of the second in spirit — "an important condition for this
//! to be a reasonable approximation is that the width of a spike is much smaller than the period"
//! (p. 303) — and it has none of Chow and Kopell's bifurcations at periods near `2Δ` (p. 294).
//!
//! Where the conventions differ. Both measure `φ` as the delay of cell 2 behind cell 1 in cycles
//! (Lewis and Rinzel's section 4; Chow and Kopell's eqs. 3.1–3.2). Lewis and Rinzel's `G(φ)` is the
//! velocity `dφ/dt` of eq. (8), so a locked state is stable where `G′ < 0`; Chow and Kopell's
//! `G(φ, T)` of eq. (3.4) is a locking condition whose slope must be POSITIVE for stability, a
//! necessary condition only (p. 1652). Lewis and Rinzel's `G` is the weak-coupling average,
//! exact as `g_c, g_s → 0` and linear in them; Chow and Kopell's is exact for existence at any
//! coupling and period. Lewis and Rinzel reset to 0; Chow and Kopell's analysis resets somewhere
//! else (the ⚠ below). Only Chow and Kopell treat `n` cells.
//!
//! # Lewis and Rinzel's phase model
//!
//! An uncoupled cell has period `T = ln(I/(I − 1))`, limit cycle `v_LC(t) = I(1 − e^{−t})` and
//! infinitesimal phase-resetting curve `Z(t) = eᵗ/(IT)` on `0 < t < T`, zero at the spike (p. 289
//! and the appendix). The interaction function of eq. (7), `H(ψ) = (1/T)∫₀ᵀ Z(t) P(t, t + ψT) dt`,
//! is here in closed form ([`LewisRinzel::h`]) — three terms with `p = ψ mod 1`:
//!
//! ```text
//! H(p) = (g_c/T)[1 − (1 − p)e^{−pT} − p e^{(1−p)T}]  +  g_c β e^{(1−p)T}/(IT²)  −  g_s J(p)/(IT²)
//! J(p) = ∫₀ᵀ eᵗ s_T(t + pT) dt = e^{−pT}[F(T) + (eᵀ − 1)F(pT)],   F(x) = ∫₀ˣ eᵘ s_T(u) du
//! ```
//!
//! with `s_T` the periodic sum of eq. (4) and `F` summed through `φ₁(y) = (eʸ − 1)/y` and
//! `φ₂(y) = ∫₀¹ ueʸᵘ du`, so that the formula stays finite at `α = 1`, where eq. (9) is `0/0`.
//! `G(φ) = H(−φ) − H(φ)`, eq. (8) ([`LewisRinzel::g`]), is then eq. (10) for the junction and eq. (9)
//! for the inhibition, and their sum, eq. (12). The kick makes `G` jump at `φ = 0` to
//! `G(0⁺) = g_c β(1 − eᵀ)/(IT²) < 0`, which is why synchrony is stable for every combination with
//! `β > 0` (p. 295). [`LewisRinzel::locked_states`] finds every zero with its stability,
//! [`LewisRinzel::probability_of_synchrony`] the width of synchrony's basin (the "10%" and "50%"
//! of pp. 290–291 and the grey scale of Fig. 10), [`LewisRinzel::antiphase_critical_drives`] the
//! drives where antiphase changes stability, and [`LewisRinzel::critical_drive_electrical`]
//! solves eq. (11) for electrical coupling alone. Beyond weak coupling, section 4's matching
//! conditions give the antiphase orbit ([`LewisRinzel::antiphase_orbit`]) and the synchronous
//! period ([`LewisRinzel::synchronous_period`]) exactly.
//!
//! # Chow and Kopell's spike-response theory
//!
//! The pair splits into a sum mode relaxing at rate 1 and a difference mode at `r = 1 + 2g` (`1 + ng`
//! for `n` cells), and each spike leaves the kernel `η_r` of eq. (2.14) in each mode
//! ([`ChowKopell::eta`]). A cell's own spike acts through `γ_s = ½(η₊ + η₋)` and its partner's
//! through `γ_c = ½(η₊ − η₋)`, eqs. (2.19)–(2.23). After the spike
//! `γ_c = −½[e^{−(t−Δ)} − δ_c e^{−r(t−Δ)}]`, and `δ_c = 1 + 2γ_c(Δ)` (2.22) is the coefficient of
//! its difference-mode term, the one relaxing at rate `r`. Locking with phase `φ` and period `T`
//! requires `G(φ, T) = (I₁ − I₂)/(1 + 2g)`, eq. (3.3), and `F(φ, T) = 1 − Ī`, eq. (3.5)
//! ([`ChowKopell::locking`], [`ChowKopell::period_function`]);
//! both are summed here in closed form for every `T > Δ`, overlapping spikes included, where the
//! paper evaluates them only for `T ≥ 2Δ`. The synchronous period is eq. (3.9)
//! ([`ChowKopell::sync_period`]); the antiphase one is the smallest root of eq. (3.11)
//! ([`ChowKopell::antiphase_period`]); the four critical periods of Fig. 4 are sign changes of
//! `∂G/∂φ` ([`ChowKopell::slope_switches`]), equal to the roots of eqs. (3.23) and (3.29) and to
//! `2Δ`; `t_min` is eq. (3.24); antiphase exists only while eq. (3.15) stays below 1
//! ([`ChowKopell::antiphase_partner_peak`]); and the splay state of `n` cells is eq. (4.9)
//! ([`ChowKopell::splay_drive`]).
//!
//! # Simulated exactly between events
//!
//! Both simulations are exact to rounding at any time between events, because both models are
//! linear there. [`LewisRinzelPair`] propagates the sum and difference modes in closed form
//! through the alpha currents; [`ChowKopellNet`] propagates the mean and every deviation from it
//! through the exponential spike currents of the cells in mid-spike. Only the threshold crossings
//! are searched for — the voltages are sampled on a grid and the first sample at or above 1 is
//! bisected until no float lies between, so a spike lands on the first float at which the closed-form
//! voltage reaches threshold — and the ends of Chow and Kopell's spikes are known in advance.
//! There is no integrator error to converge; instead a classical Runge–Kutta integration of the
//! same equations is shown to converge on each closed form at fourth order, and sampling eight
//! times more finely moves no spike by more than 3 × 10⁻¹⁴. A crossing that enters and leaves
//! threshold between two samples is not seen.
//!
//! # What is checked
//!
//! Every closed form against its definition: `H` against Simpson's rule on eq. (7) (1.7 × 10⁻¹⁴)
//! and against `SciPy`'s adaptive quadrature (2.2 × 10⁻¹⁶); `G` against eqs. (9) and (10) as printed
//! (7.4 × 10⁻¹⁶); eq. (11) against the sign change of `G′(½)` (4.5 × 10⁻¹⁶); the kernels against
//! quadrature of eq. (2.13) (1.2 × 10⁻¹⁴); `G(φ, T)` against eqs. (3.19)–(3.20) and a brute-force
//! sum of eq. (3.4) (7 × 10⁻¹⁶); the slopes against eqs. (3.22) and (3.27)–(3.28); `F` against eqs.
//! (3.8) and (3.11); the critical periods against eqs. (3.23) and (3.29). Every simulation against
//! `SciPy`'s DOP853 integration of eqs. (5) and (4.1) (`tools/gapjunction_reference.py`), spike for
//! spike to the twelve decimals it prints; Chow and Kopell's simulated voltages against their
//! spike-response form (4.5) (7.2 × 10⁻¹⁴); Lewis and Rinzel's section-4 orbit against the
//! simulation it predicts (1.2 × 10⁻¹²). The phase models against the full models: a pair coupled
//! at `ε = 0.004` and `0.002` drifts at the rate `G′` predicts with an error that halves with `ε`
//! (1.72%, 0.82%); a Chow–Kopell pair locks in synchrony at `T_S` and in antiphase at the period
//! eq. (3.11) gives, to 1.1 × 10⁻¹²; three cells settle into the splay state of eq. (4.9) to
//! 1.7 × 10⁻¹².
//!
//! And the figures, against the module at their printed parameters. Lewis and Rinzel's Figs. 4, 5,
//! 7, 8, 9 and 11 are digitised from the PDF and compared point by point; Figs. 3 and 6 by their
//! axes and the extremes of their curves; Fig. 10 by its caption's inequalities and the computed
//! end points of its curves; Figs. 1 and 2 through the runs they plot, spike for spike against
//! `SciPy`, not through their drawn traces. Chow and Kopell's Figs. 1, 2(b) and 3 are read from
//! their vector paths. Not checked: Lewis and Rinzel's Figs. 12 and 13 (see *Not here yet*), the
//! drawn curves and grey scale of their Fig. 10, and Chow and Kopell's Fig. 2(a); Chow and
//! Kopell's Figs. 4–6 have no numeric period or drive axis to read.
//!
//! # The reference
//!
//! This review did not locate code from either paper's authors: `ModelDB` lists no model for
//! either, Lewis's publication directory holds PDFs only, and the one program Chow and Kopell
//! name, `XPPAUT`, ran their conductance-based models, not the integrate-and-fire one. The
//! reference is therefore `tools/gapjunction_reference.py`, written from the papers' equations
//! with `SciPy` 1.13: it integrates eqs. (5) and (4.1) by DOP853 at `rtol = 10⁻¹³`, integrates
//! eq. (7) and section 4's matching conditions by adaptive quadrature, and reads the figures out
//! of the PDFs — Lewis and Rinzel's as 144 dpi rasters against their own tick marks, Chow and
//! Kopell's as vector paths, Fig. 3's fitted by least squares to its own term-by-term sum of
//! eq. (3.4). The tests embed its output with its provenance.
//!
//! # ⚠ Printed defects and inconsistencies, Lewis and Rinzel
//!
//! ⚠ **Section 4's matching conditions carry the inhibition with the wrong sign.** Every synaptic
//! bracket of the four equations on p. 299 is added with `+g_s`, where eq. (5) makes inhibition
//! `−g_s`; at Fig. 1's left panel they give an antiphase period of 1.631 404 against the 3.518 766
//! the simulation settles on and the corrected system gives. Their third equation's bracket also
//! subtracts one current where the first adds it, so at `φ = ½` they cannot both hold for `u₁ = u₂`.
//! And their kick terms are right only if `u` is the voltage AFTER the kick, while the text defines
//! it as the voltage "immediately before spike effects are added" (p. 299): with electrical coupling
//! alone they return the right period with `u` larger by exactly `g_c β`, and the text's own test
//! for spike capture, `u > 1 − g_c β`, then misplaces the boundary by `g_c β`.
//!
//! ⚠ **Fig. 7 is drawn at `β = 0.2`, not the caption's 0.1.** Its arrow marks `I*_c = 1.2588`, which
//! is eq. (11) at `β = 0.2` (1.2592); at `β = 0.1` it is 1.4942. Its unstable branch runs through
//! `β = 0.2`'s locked states to 0.008 in `φ`, and `β = 0.1`'s lie at half the phase.
//!
//! ⚠ **Fig. 11's caption gives `I*_c(β = 0.2) = 1.28`** (p. 298); the paper's own eq. (11) gives
//! 1.2592, and its text "I ∼ 1.26" (p. 297).
//!
//! ⚠ **Table 1 reverses the inequality that its Fig. 10 turns on.** It reads "Fast synapses, large
//! spike effect (`I*_c > I*_s`) — promotes synchrony" (p. 302); Fig. 10's caption, the text of p. 295
//! and the computation all have `I*_s > I*_c` there (1.484 against 1.165 at `β = 0.3`, `α = 4`).
//!
//! ⚠ **Fig. 9's combined branch is not the captioned one near its fork.** The light `ρ = 1` and
//! `ρ = 0` arrows sit where the computation puts them (1.2587 against 1.2592; 1.6618 against 1.6576),
//! but the black `ρ = 0.5` branch at `α = 5`, `β = 0.2` drifts away as it nears the fork — 0.295
//! against 0.345 at `I = 1.5` — and its arrow reads `I*_sc = 1.5746` against 1.5390. No nearby `ρ`,
//! `α` or `β` fits both; the discrepancy is measured, not explained.
//!
//! ⚠ **The appendix prints the period as `T = (ln(I/(I − 1)))⁻¹`** (p. 307): that is the
//! frequency, eq. (2), and the figures label their frequency axes with it. And it starts the
//! phase-resetting curve's linear fall at `ln((I − 1)/(I + ε(I/(I − 1 + ε))))/T` (p. 308), which is
//! negative for every `ε > 0`; the two branches of its own formula meet at `ln((I − 1 + ε)/(I − 1))/T`.
//!
//! Smaller things, recorded so nobody corrects the module towards them. Eq. (9)'s integral line
//! omits the `−g_s` its result line carries, and eq. (12)'s writes `+g_s` (p. 294); the result lines
//! are right. At `I = 1.2`, `α = 4` the unstable states the text puts at 0.05 and 0.95 with "only a
//! 10% chance" of synchrony (p. 290) are at 0.0635 and 0.9365, a 12.7% chance. "Cells fire at times
//! `(m + φ_j)T`" (p. 288) has the sign of `φ_j` reversed for `v_j ∼ v_LC(t + φ_jT)`. Fig. 1's caption
//! names both synaptic traces "the synaptic currents in cell 2 due to the firing of cell 1".
//!
//! # ⚠ Printed defects and inconsistencies, Chow and Kopell
//!
//! ⚠ **Eq. (2.6) does not solve eq. (2.3).** Printed as `v(t) = 1 + I(1 − e^{−t}) + v_A(e^{ξt} −
//! e^{−t})/(1 + ξ)`, its slope at `t = 0` is `I + v_A` where the equation gives `I − 1 + v_A`; the
//! solution from `v(0) = 1` has `I − 1` for `I` ([`ChowKopell::spike_voltage`]).
//!
//! ⚠ **The paper defines `v_M` three ways, and its figure and its analysis use different ones.**
//! Eq. (2.6) defines `v_M = v(Δ) − 1`, which resets an uncoupled cell to exactly 0 and depends on
//! the drive; p. 1648 says `v_M ≃ η₊(Δ)`; p. 1650 says `v_M ≡ η₊(Δ)`. The kernels (2.21) and (2.23)
//! and every period of section 3 are exact only for the last, under which an uncoupled cell lands
//! at `(I − 1)(1 − e^{−Δ})`: this module analyses and, by default, simulates that one
//! ([`Reset::Kernel`]). Fig. 1 was drawn with the first ([`Reset::ToZero`]): its spikes are 1.5655
//! and 1.538 apart in its vector paths, which is `Δ + ln(I/(I − 1))` — 1.5663 and 1.5361 — and not
//! eq. (3.9)'s 1.5441 and 1.3857.
//!
//! ⚠ **Fig. 1(d) is drawn at `v_A ≈ 0.2`, not the caption's 0.1**: its spikes peak at 7.19, where
//! `v_A = 0.1` peaks at 4.32 and `v_A = 0.2` at 7.41; panels (a)–(c) peak within 3% below their
//! computed values.
//!
//! ⚠ **Fig. 3 draws `½G`.** At its printed parameters `G(φ, T)` peaks at 0.039474, 0.050842, 0.017212
//! and 0.012492; the drawn curves peak at 0.019722, 0.025000 (clipped at the axis), 0.008583 and
//! 0.006222, and a least-squares fit of the whole drawn curve to `G` gives the scale 0.4994–0.4998
//! in every panel. Across `g` from 0.02 to 6 the four panels agree on one scale only near the
//! printed `g = 0.5`: on a grid of 75 values their scales differ by 0.07% there and by at least
//! 1.7% at every other `g`, 1.8% and 1.7% already at `g = 0.48` and 0.52; at `g = 0.25`, where
//! they straddle 1, they run from 0.91 to 1.13. Fig. 2, drawn with the same kernels, is at full
//! scale.
//!
//! ⚠ **Eq. (3.17) is misprinted, and eq. (3.18) inherits the misprint.** Substituting eq. (3.11) into
//! eq. (3.16) gives a last term `δ_c e^{rΔ}/(e^{rT/2} + 1)`; it is printed with `e^{rT/2} − 1`
//! (p. 1654). Solved for `δ_c`, the corrected (3.17) is an upper bound for every `T > 2Δ`, its
//! coefficient of `δ_c` being `1 − e^{−r(T/2 − Δ)} > 0`; the printed one only while its own
//! coefficient stays positive — at `T = 0.5` for Fig. 3's spike it is −2.02, so the printed
//! inequality holds for every `δ_c > 0`, where the corrected one requires `δ_c < 3.90`. At `Δ = 0` that coefficient is `2(sinh x − 1)/(eˣ − 1)`, `x = rT/2`: eq. (3.18),
//! `sinh(rT/2)/(sinh(rT/2) − 1)`, is the misprinted (3.17) at `Δ = 0` where `sinh(rT/2) > 1`,
//! while below `asinh 1` the misprinted inequality imposes no bound and (3.18) as printed is
//! negative, which would forbid antiphase outright. The correct limit is `coth(rT/4)` at every
//! `T`. For Fig. 3's spike the printed (3.17) puts the longest antiphase period at 3.4158, where
//! eq. (3.15) and the corrected (3.17) put it at 3.3688.
//!
//! ⚠ **The range of drives for synchrony is printed empty.** "For `1 < Ī ≤ 1 + (1 − e^Δ)⁻¹` there is
//! always a solution for `T_S`" (p. 1653): since `e^Δ > 1` the upper end is below 1. The period of
//! eq. (3.9) exceeds `Δ` exactly for `Ī < 1 + (1 − e^{−Δ})⁻¹` — 11.508 for `Δ = 0.1` — and
//! [`ChowKopell::sync_period`] refuses beyond it.
//!
//! ⚠ **The network kernels (4.6)–(4.7) omit `1/n`, and eq. (4.12) counts each cell as its own
//! partner.** Diagonalising eq. (4.1) gives `Γ_s = [η₊ + (n − 1)η₋]/n` and `Γ_c = (η₊ − η₋)/n`; as
//! printed they are twice the pair's own `γ_s` and `γ_c` at `n = 2`, and the simulated voltages
//! follow the normalised ones. Eq. (4.12) also sums `Γ_c` over every multiple of `T/n`, the cell's
//! own firing times included: for uncoupled cells it gives `I − 1 = n e^Δ/(eᵀ − 1)`, `n` times the
//! drive excess a single cell needs for the same period — for `n = 3` at `I = 1.3` and `Δ = 0.1`
//! it predicts `T = 2.489` against the 1.544 an uncoupled cell fires at. The correct condition is
//! in [`ChowKopell::splay_drive`].
//!
//! ⚠ **"For small r, increasing r will increase `T_C^{AS3}`" (p. 1659) is the reverse of the paper's
//! own eqs. (3.27)–(3.28)**: for every spike shape tried that has a `T_C^{AS3}`, it falls as `r` grows,
//! from `g = 0.001` up — for Fig. 3's spike from 0.115 430 to 0.114 055 at `g = 1` — so a stronger
//! junction widens the short-period window in which antiphase meets the necessary condition.
//!
//! Smaller things. Eq. (3.25)'s `t_min ≈ Δ + 1 + v_M/(1 + ξ)` misses its own `g → 0` limit by the
//! `v_A Δe^{−Δ}/(1 + ξ)` that eq. (2.25)'s fast-spike approximation drops (0.0018 for Fig. 3's spike).
//! "Bifurcations take place at critical points `T_C` that satisfy `G(φ, T_C) = 0`" (p. 1655) means
//! `∂G/∂φ`, since `G` vanishes at `φ = 0` and `½` for every `T`. Equations 3.24, 3.27 and 3.29 are cited
//! as 4.24, 4.27 and 4.29 on pp. 1658–1660, and section 3.3 as 3.1 on p. 1671.
//!
//! # Units
//!
//! Both papers work in dimensionless units, and this module keeps them, as [`crate::planar`] keeps
//! `FitzHugh`'s: voltages are fractions of the reset-to-threshold range, times are membrane time
//! constants, conductances are in units of the leak, and the drive is `I = (I_app + g_m(V_r −
//! V_reset))/(g_m(V_th − V_reset))` (Lewis and Rinzel, p. 286; Chow and Kopell's eq. 2.4). The
//! conversion to SI needs a membrane time constant and a voltage range that neither model carries.
//!
//! # Not here yet
//!
//! Asymmetric locked states beyond weak coupling — the tines of Lewis and Rinzel's Figs. 12 and 13,
//! which need a two-dimensional solve of section 4's system — and Chow and Kopell's two
//! conductance-based models (their Tables 1 and 2), which are not integrate-and-fire neurons.

use core::fmt;

/// Why a gap-junction question could not be answered.
#[derive(Debug, Clone, PartialEq)]
pub enum GapError {
    /// A parameter that must be finite and positive was not.
    NotPositive {
        /// Which parameter.
        what: &'static str,
        /// Its value.
        value: f64,
    },
    /// A parameter that must be finite and not negative was not.
    Negative {
        /// Which parameter.
        what: &'static str,
        /// Its value.
        value: f64,
    },
    /// A value that must be finite was not.
    NonFinite {
        /// Which value.
        what: &'static str,
        /// The value.
        value: f64,
    },
    /// A value outside the closed interval it must lie in.
    OutOfRange {
        /// Which value.
        what: &'static str,
        /// The value.
        value: f64,
        /// The least it may be.
        low: f64,
        /// The most it may be.
        high: f64,
    },
    /// A drive at or below the threshold 1: an uncoupled cell settles at `I` and never fires.
    Subthreshold {
        /// The dimensionless drive `I`.
        drive: f64,
    },
    /// A drive so large that the synchronous period of eq. (3.9) would not exceed the spike width.
    Overdriven {
        /// The dimensionless drive `I`.
        drive: f64,
        /// `1 + 1/(1 − e^{−Δ})`, which it must stay below.
        limit: f64,
    },
    /// A period no longer than the stretch the spike response sums need.
    PeriodTooShort {
        /// The period `T`.
        period: f64,
        /// What it must exceed: the spike width `Δ`, or `2Δ` for the antiphase peak.
        limit: f64,
    },
    /// A starting voltage at or above threshold, where the cell would have to fire before time 0.
    AboveThreshold {
        /// Which cell.
        cell: usize,
        /// Its voltage.
        value: f64,
    },
    /// Neither coupling is on, so `G ≡ 0`: every phase difference is locked and none is selected.
    Uncoupled,
    /// A count below its minimum.
    TooFew {
        /// What was counted.
        what: &'static str,
        /// The count.
        value: usize,
        /// The least it may be.
        min: usize,
    },
    /// An interval whose lower end is not below its upper end.
    Empty {
        /// Lower end.
        low: f64,
        /// Upper end.
        high: f64,
    },
    /// Two lists that must have one entry per cell do not.
    Mismatch {
        /// Entries in the first.
        drives: usize,
        /// Entries in the second.
        voltages: usize,
    },
    /// A simulation used up its bound on events before reaching the end of the run.
    Stalled {
        /// How far it got.
        time: f64,
    },
}

impl fmt::Display for GapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotPositive { what, value } => write!(f, "{what} = {value} must be finite and positive"),
            Self::Negative { what, value } => write!(f, "{what} = {value} must be finite and not negative"),
            Self::NonFinite { what, value } => write!(f, "{what} = {value} is not finite"),
            Self::OutOfRange { what, value, low, high } => write!(f, "{what} = {value} is outside [{low}, {high}]"),
            Self::Subthreshold { drive } => {
                write!(f, "drive I = {drive} does not exceed the threshold 1, so an uncoupled cell never fires")
            }
            Self::Overdriven { drive, limit } => {
                write!(f, "drive I = {drive} is not below {limit}, where the synchronous period reaches the spike width")
            }
            Self::PeriodTooShort { period, limit } => write!(f, "period T = {period} must exceed {limit}"),
            Self::AboveThreshold { cell, value } => write!(f, "cell {cell} starts at v = {value}, not below the threshold 1"),
            Self::Uncoupled => write!(f, "g_c = 0 and g_s = 0: with no coupling every phase difference is locked"),
            Self::TooFew { what, value, min } => write!(f, "{what} = {value} must be at least {min}"),
            Self::Empty { low, high } => write!(f, "the interval ({low}, {high}) is empty"),
            Self::Mismatch { drives, voltages } => {
                write!(f, "{drives} drives and {voltages} starting voltages: one of each per cell")
            }
            Self::Stalled { time } => write!(f, "the run used up its event budget at t = {time}; sample more finely"),
        }
    }
}

impl std::error::Error for GapError {}

fn finite(what: &'static str, value: f64) -> Result<f64, GapError> {
    if value.is_finite() { Ok(value) } else { Err(GapError::NonFinite { what, value }) }
}

fn positive(what: &'static str, value: f64) -> Result<f64, GapError> {
    if value.is_finite() && value > 0.0 { Ok(value) } else { Err(GapError::NotPositive { what, value }) }
}

fn non_negative(what: &'static str, value: f64) -> Result<f64, GapError> {
    if value.is_finite() && value >= 0.0 { Ok(value) } else { Err(GapError::Negative { what, value }) }
}

fn at_least(what: &'static str, value: usize, min: usize) -> Result<usize, GapError> {
    if value >= min { Ok(value) } else { Err(GapError::TooFew { what, value, min }) }
}

/// `φ₁(y) = (eʸ − 1)/y = ∫₀¹ e^{yu} du`, with its limit 1 at `y = 0`.
fn phi1(y: f64) -> f64 {
    if y == 0.0 { 1.0 } else { y.exp_m1() / y }
}

/// `φ₂(y) = (eʸ(y − 1) + 1)/y² = ∫₀¹ u e^{yu} du`, with its limit ½ at `y = 0`.
///
/// The closed form loses about `ε/|y|` to cancellation, so inside `|y| < ½` the Taylor series
/// `Σ (k + 1) yᵏ/(k + 2)!` is summed instead; eighteen terms leave a remainder below 10⁻²⁰ there.
fn phi2(y: f64) -> f64 {
    if y.abs() < 0.5 {
        let mut term = 0.5;
        let mut sum = 0.5;
        for k in 1..18 {
            term *= y / (k + 2) as f64;
            sum += (k + 1) as f64 * term;
        }
        sum
    } else {
        (y * y.exp() - y.exp_m1()) / (y * y)
    }
}

/// `∫₀ˢ e^{−κ(s−u)} (p + q u) e^{−αu} du`: a leaky membrane of rate `κ` driven for `s` by an
/// alpha-shaped current `(p + qu)e^{−αu}`, from rest. Exact.
///
/// With `d = κ − α` it is `e^{−κs} s [p φ₁(ds) + q s φ₂(ds)]`, which overflows once `ds` passes
/// about 709; for `d > 0` the identities `φ₁(y) = eʸ φ₁(−y)` and `φ₂(y) = eʸ[φ₁(−y) − φ₂(−y)]`
/// move the exponential onto `e^{−αs}`, so every `φ` here is evaluated at a non-positive argument.
fn alpha_response(kappa: f64, alpha: f64, s: f64, p: f64, q: f64) -> f64 {
    let d = kappa - alpha;
    if d > 0.0 {
        let (a, b) = (phi1(-d * s), phi2(-d * s));
        (-alpha * s).exp() * s * (p * a + q * s * (a - b))
    } else {
        (-kappa * s).exp() * s * (p * phi1(d * s) + q * s * phi2(d * s))
    }
}

/// The `T`-periodic synaptic current `s_T(u)` of Lewis and Rinzel's eq. (4) made dimensionless:
/// `α²e^{−αu}[u(1 − e^{−αT}) + Te^{−αT}]/(1 − e^{−αT})²` for `0 ≤ u ≤ T`.
fn s_periodic(alpha: f64, period: f64, u: f64) -> f64 {
    let q = (-alpha * period).exp();
    let one_q = -(-alpha * period).exp_m1();
    alpha * alpha / (one_q * one_q) * (-alpha * u).exp() * (u * one_q + period * q)
}

/// `∫₀^{span} e^{−κ(span − t)} s_T(t₀ + t) dt` for `0 ≤ t₀` and `t₀ + span ≤ T`: the voltage a
/// membrane relaxing at rate `κ` collects from the periodic synaptic current over one stretch.
fn leaky_s(alpha: f64, period: f64, kappa: f64, t0: f64, span: f64) -> f64 {
    let q = (-alpha * period).exp();
    let one_q = -(-alpha * period).exp_m1();
    let scale = alpha * alpha / (one_q * one_q) * (-alpha * t0).exp();
    alpha_response(kappa, alpha, span, scale * (t0 * one_q + period * q), scale * one_q)
}

/// Bisect `[lo, hi]`, on which `below(lo)` holds and `below(hi)` does not, until no float lies
/// between the two ends; returns `hi`, the first point found where `below` fails.
fn bisect(below: impl Fn(f64) -> bool, mut lo: f64, mut hi: f64) -> f64 {
    for _ in 0..1100 {
        let m = 0.5 * (lo + hi);
        if m <= lo || m >= hi {
            break;
        }
        if below(m) {
            lo = m;
        } else {
            hi = m;
        }
    }
    hi
}

/// Every sign change of `f` between consecutive points of the grid `a + (b − a)n/samples`,
/// `n = 0..=samples`, each bisected to the last bit. Two roots inside one grid step are missed.
fn roots(f: impl Fn(f64) -> f64, a: f64, b: f64, samples: usize) -> Vec<f64> {
    let mut out = Vec::new();
    let mut prev = (a, f(a));
    for n in 1..=samples {
        let x = a + (b - a) * n as f64 / samples as f64;
        let now = (x, f(x));
        if (prev.1 < 0.0) != (now.1 < 0.0) {
            let neg = prev.1 < 0.0;
            out.push(bisect(|m| (f(m) < 0.0) == neg, prev.0, now.0));
        }
        prev = now;
    }
    out
}

/// One spike of a simulated cell.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Spike {
    /// When, in membrane time constants from the start of the simulation.
    pub time: f64,
    /// Which cell.
    pub cell: usize,
}

/// A phase-locked state of the reduced pair: a zero of `G` and whether it attracts.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LockedState {
    /// The phase difference `φ* ∈ [0, 1)`, in cycles.
    pub phase: f64,
    /// Whether nearby phase differences move towards it.
    pub stable: bool,
}

/// An antiphase orbit of the pair beyond weak coupling (Lewis and Rinzel's section 4).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Orbit {
    /// The period `T` of each cell, membrane time constants.
    pub period: f64,
    /// Each cell's voltage immediately before its partner fires, before the kick `g_c β` lands.
    pub u: f64,
}

/// Lewis and Rinzel's cell pair: two identical leaky integrate-and-fire cells coupled by a gap
/// junction and by reciprocal alpha-function inhibition, in their dimensionless units (eq. 5).
///
/// Voltage is `(V − V_reset)/(V_th − V_reset)`, so the threshold is 1 and the reset 0; time is in
/// membrane time constants.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LewisRinzel {
    /// The applied current `I`; an uncoupled cell fires when `I > 1`.
    pub drive: f64,
    /// The electrical coupling `g_c`, not negative.
    pub g_c: f64,
    /// The spike effect `β`: each spike of a cell kicks its partner by `g_c β`. Not negative.
    pub beta: f64,
    /// The synaptic strength `g_s`: positive inhibits, negative excites (p. 292).
    pub g_s: f64,
    /// The synaptic rate `α`, the reciprocal of the alpha function's time constant. Positive.
    pub alpha: f64,
}

impl LewisRinzel {
    /// Fig. 1 (left): inhibition alone, `I = 1.1`, `g_s = 0.2`, `α = 3`, `g_c = 0`.
    pub const FIG1_LOW: Self = Self { drive: 1.1, g_c: 0.0, beta: 0.0, g_s: 0.2, alpha: 3.0 };
    /// Fig. 1 (right): the same coupling at `I = 1.6`.
    pub const FIG1_HIGH: Self = Self { drive: 1.6, ..Self::FIG1_LOW };
    /// Fig. 2 (left): electrical coupling alone, `I = 1.1`, `g_c = 0.2`, `β = 0.2`, `g_s = 0`.
    ///
    /// With `g_s = 0` the synaptic rate never enters; it is Fig. 1's 3 only because it must be
    /// positive.
    pub const FIG2_LOW: Self = Self { drive: 1.1, g_c: 0.2, beta: 0.2, g_s: 0.0, alpha: 3.0 };
    /// Fig. 2 (right): the same coupling at `I = 1.6`.
    pub const FIG2_HIGH: Self = Self { drive: 1.6, ..Self::FIG2_LOW };
    /// Fig. 3 (top): inhibition alone with `α = 4` at `I = 1.2`. The paper prints no `g_s`, which
    /// only scales `G`; 1 is the value that reproduces the figure's axes (see the tests).
    pub const FIG3_TOP: Self = Self { drive: 1.2, g_c: 0.0, beta: 0.0, g_s: 1.0, alpha: 4.0 };
    /// Fig. 3 (middle): `I = 1.4`.
    pub const FIG3_MIDDLE: Self = Self { drive: 1.4, ..Self::FIG3_TOP };
    /// Fig. 3 (bottom): `I = 1.6`.
    pub const FIG3_BOTTOM: Self = Self { drive: 1.6, ..Self::FIG3_TOP };
    /// Fig. 6 (bottom): electrical coupling alone, `β = 0.1`, `I = 1.15`, and `g_c = 1`, the value
    /// that reproduces the figure's axes. The synaptic rate is unused and set to 1.
    pub const FIG6_BOTTOM: Self = Self { drive: 1.15, g_c: 1.0, beta: 0.1, g_s: 0.0, alpha: 1.0 };

    /// A pair with every parameter checked.
    ///
    /// # Errors
    ///
    /// Whatever [`LewisRinzel::check`] refuses.
    pub fn new(drive: f64, g_c: f64, beta: f64, g_s: f64, alpha: f64) -> Result<Self, GapError> {
        let m = Self { drive, g_c, beta, g_s, alpha };
        m.check()?;
        Ok(m)
    }

    /// A pair with total coupling `g_tot = g_c + g_s`, a fraction `ρ = g_c/(g_c + g_s)` of it
    /// electrical (section 3.5): `g_c = ρ g_tot`, `g_s = (1 − ρ) g_tot`.
    ///
    /// # Errors
    ///
    /// [`GapError::OutOfRange`] for a `rho` outside `[0, 1]`; [`GapError::NotPositive`] for a
    /// `g_tot` that is not; whatever [`LewisRinzel::check`] refuses.
    pub fn combined(drive: f64, rho: f64, g_tot: f64, beta: f64, alpha: f64) -> Result<Self, GapError> {
        if !(0.0..=1.0).contains(&rho) {
            return Err(GapError::OutOfRange { what: "rho", value: rho, low: 0.0, high: 1.0 });
        }
        let g_tot = positive("g_tot", g_tot)?;
        Self::new(drive, rho * g_tot, beta, (1.0 - rho) * g_tot, alpha)
    }

    /// Every parameter in range: `I > 1`, `g_c ≥ 0`, `β ≥ 0`, `g_s` finite, `α > 0`.
    ///
    /// # Errors
    ///
    /// [`GapError::NonFinite`] or [`GapError::Subthreshold`] for the drive,
    /// [`GapError::Negative`] for `g_c` or `β`, [`GapError::NonFinite`] for `g_s`,
    /// [`GapError::NotPositive`] for `α`.
    pub fn check(&self) -> Result<(), GapError> {
        finite("I", self.drive)?;
        if self.drive <= 1.0 {
            return Err(GapError::Subthreshold { drive: self.drive });
        }
        non_negative("g_c", self.g_c)?;
        non_negative("beta", self.beta)?;
        finite("g_s", self.g_s)?;
        positive("alpha", self.alpha)?;
        Ok(())
    }

    /// The uncoupled period `T = ln(I/(I − 1))`.
    fn t(&self) -> f64 {
        (self.drive / (self.drive - 1.0)).ln()
    }

    /// The intrinsic period `T = ln(I/(I − 1))`, in membrane time constants: eq. (2) made
    /// dimensionless, and the line under eq. (9).
    ///
    /// # Errors
    ///
    /// Whatever [`LewisRinzel::check`] refuses.
    pub fn period(&self) -> Result<f64, GapError> {
        self.check()?;
        Ok(self.t())
    }

    /// The uncoupled voltage `v_LC(t) = I(1 − e^{−t})`, `t` taken modulo the period.
    ///
    /// # Errors
    ///
    /// [`GapError::NonFinite`] for `t`; whatever [`LewisRinzel::check`] refuses.
    pub fn limit_cycle(&self, t: f64) -> Result<f64, GapError> {
        self.check()?;
        let u = finite("t", t)?.rem_euclid(self.t());
        Ok(self.drive * -(-u).exp_m1())
    }

    /// The infinitesimal phase-resetting curve `Z(t) = eᵗ/(IT)` on `0 < t < T`, and 0 at the
    /// spike (`t ≡ 0`), as the paper takes it (p. 289). `t` is taken modulo the period.
    ///
    /// # Errors
    ///
    /// [`GapError::NonFinite`] for `t`; whatever [`LewisRinzel::check`] refuses.
    pub fn prc(&self, t: f64) -> Result<f64, GapError> {
        self.check()?;
        let period = self.t();
        let u = finite("t", t)?.rem_euclid(period);
        Ok(if u > 0.0 { u.exp() / (self.drive * period) } else { 0.0 })
    }

    /// The phase advance, in cycles, that a voltage kick `ε` delivered at time `t` after a spike
    /// produces: the appendix's `Δφ = ln(I/(I − εeᵗ))/T`, and `(T − t)/T` once the kick alone
    /// carries the cell to threshold.
    ///
    /// # Errors
    ///
    /// [`GapError::NonFinite`] for `eps` or `t`; whatever [`LewisRinzel::check`] refuses.
    pub fn phase_advance(&self, eps: f64, t: f64) -> Result<f64, GapError> {
        self.check()?;
        let period = self.t();
        let eps = finite("eps", eps)?;
        let u = finite("t", t)?.rem_euclid(period);
        let v = self.drive * -(-u).exp_m1() + eps;
        Ok(if v >= 1.0 { (period - u) / period } else { (self.drive / (self.drive - eps * u.exp())).ln() / period })
    }

    /// The periodic synaptic current `s_T(t)` a cell firing with its intrinsic period injects:
    /// eq. (4) made dimensionless, `α²e^{−αt}[t(1 − e^{−αT}) + Te^{−αT}]/(1 − e^{−αT})²` on
    /// `[0, T)`, `t` taken modulo `T`.
    ///
    /// # Errors
    ///
    /// [`GapError::NonFinite`] for `t`; whatever [`LewisRinzel::check`] refuses.
    pub fn synaptic_current(&self, t: f64) -> Result<f64, GapError> {
        self.check()?;
        let period = self.t();
        Ok(s_periodic(self.alpha, period, finite("t", t)?.rem_euclid(period)))
    }

    /// `H` at reduced phase `p ∈ [0, 1]`; `p = 1` stands for the limit from below.
    fn h_at(&self, p: f64) -> f64 {
        let (i, period) = (self.drive, self.t());
        let sub = self.g_c / period * (1.0 - (1.0 - p) * (-p * period).exp() - p * ((1.0 - p) * period).exp());
        let spike = if p > 0.0 { self.g_c * self.beta * ((1.0 - p) * period).exp() / (i * period * period) } else { 0.0 };
        sub + spike - self.g_s * self.j(p) / (i * period * period)
    }

    /// `J(p) = ∫₀ᵀ eᵗ s_T(t + pT) dt = e^{−pT}[F(T) + (eᵀ − 1)F(pT)]`, with
    /// `F(x) = ∫₀ˣ eᵘ s_T(u) du` in closed form.
    fn j(&self, p: f64) -> f64 {
        let (a, period) = (self.alpha, self.t());
        let q = (-a * period).exp();
        let one_q = -(-a * period).exp_m1();
        let k = a * a / (one_q * one_q);
        let f = |x: f64| k * (one_q * x * x * phi2((1.0 - a) * x) + period * q * x * phi1((1.0 - a) * x));
        (-p * period).exp() * (f(period) + period.exp_m1() * f(p * period))
    }

    /// `dH/dp` at reduced phase `p ∈ [0, 1]`, one-sided at the ends.
    fn h_slope_at(&self, p: f64) -> f64 {
        let (i, period) = (self.drive, self.t());
        let (e_lo, e_hi) = ((-p * period).exp(), ((1.0 - p) * period).exp());
        let sub = self.g_c / period * (e_lo + (1.0 - p) * period * e_lo - e_hi + p * period * e_hi);
        let spike = -self.g_c * self.beta * e_hi / (i * period);
        let dj = period * (period.exp_m1() * s_periodic(self.alpha, period, p * period) - self.j(p));
        sub + spike - self.g_s * dj / (i * period * period)
    }

    /// The interaction function `H(ψ) = (1/T)∫₀ᵀ Z(t) P(t, t + ψT) dt` of eq. (7), in closed form:
    /// the rate at which a partner leading by `ψ` cycles advances a cell's phase.
    ///
    /// Three terms: the ohmic current through the junction,
    /// `(g_c/T)[1 − (1 − p)e^{−pT} − p e^{(1−p)T}]`; the spike's kick, `g_c β e^{(1−p)T}/(IT²)`,
    /// zero at `p = 0` where `Z` is; and the inhibition, `−g_s J(p)/(IT²)`, with `p = ψ mod 1`.
    ///
    /// # Errors
    ///
    /// [`GapError::NonFinite`] for `psi`; whatever [`LewisRinzel::check`] refuses.
    pub fn h(&self, psi: f64) -> Result<f64, GapError> {
        self.check()?;
        Ok(self.h_at(finite("psi", psi)?.rem_euclid(1.0)))
    }

    /// `G(φ) = H(−φ) − H(φ)`, eq. (8): the rate of change of the phase difference `φ = φ₁ − φ₂`,
    /// in cycles per membrane time constant.
    ///
    /// # Errors
    ///
    /// [`GapError::NonFinite`] for `phi`; whatever [`LewisRinzel::check`] refuses.
    pub fn g(&self, phi: f64) -> Result<f64, GapError> {
        self.check()?;
        let phi = finite("phi", phi)?;
        Ok(self.h_at((-phi).rem_euclid(1.0)) - self.h_at(phi.rem_euclid(1.0)))
    }

    /// `dG/dφ = −H′(−φ) − H′(φ)`, from the closed forms. At `φ ≡ 0` it is the one-sided slope,
    /// the same on both sides; `G` itself jumps there when `g_c β > 0`.
    ///
    /// # Errors
    ///
    /// [`GapError::NonFinite`] for `phi`; whatever [`LewisRinzel::check`] refuses.
    pub fn g_slope(&self, phi: f64) -> Result<f64, GapError> {
        self.check()?;
        let p = finite("phi", phi)?.rem_euclid(1.0);
        let (lo, hi) = if p > 0.0 { (p, 1.0 - p) } else { (0.0, 1.0) };
        Ok(-self.h_slope_at(hi) - self.h_slope_at(lo))
    }

    /// Every phase-locked state of the reduced pair, in increasing phase, each with its stability.
    ///
    /// Synchrony `φ = 0` and antiphase `φ = ½` are zeros of `G` for every parameter set, because
    /// `G` is odd and 1-periodic. Any others come in pairs `φ*`, `1 − φ*`; they are found as sign
    /// changes of `G` on the interior points `½n/samples` of `(0, ½)`, each bisected to the last
    /// bit, and a pair of zeros closer than one grid step is missed. An interior zero is stable
    /// when `G′ < 0` there, and so is antiphase. Synchrony is judged from the side: the spike's
    /// kick makes `G` jump at `φ = 0`, to `G(0⁺) = g_c β(1 − eᵀ)/(IT²) < 0`, which attracts from
    /// both sides whatever the slope; without it, `G′(0) < 0` decides.
    ///
    /// # Errors
    ///
    /// [`GapError::Uncoupled`] when `g_c = g_s = 0`; [`GapError::TooFew`] for fewer than three
    /// samples; whatever [`LewisRinzel::check`] refuses.
    pub fn locked_states(&self, samples: usize) -> Result<Vec<LockedState>, GapError> {
        self.check()?;
        if self.g_c == 0.0 && self.g_s == 0.0 {
            return Err(GapError::Uncoupled);
        }
        let samples = at_least("samples", samples, 3)?;
        let step = 0.5 / samples as f64;
        let inner: Vec<LockedState> = roots(|x| self.h_at(1.0 - x) - self.h_at(x), step, 0.5 - step, samples - 2)
            .into_iter()
            .map(|z| LockedState { phase: z, stable: -self.h_slope_at(1.0 - z) - self.h_slope_at(z) < 0.0 })
            .collect();
        let sync = self.g_c * self.beta > 0.0 || -self.h_slope_at(1.0) - self.h_slope_at(0.0) < 0.0;
        let mut out = vec![LockedState { phase: 0.0, stable: sync }];
        out.extend(inner.iter().copied());
        out.push(LockedState { phase: 0.5, stable: -2.0 * self.h_slope_at(0.5) < 0.0 });
        out.extend(inner.iter().rev().map(|s| LockedState { phase: 1.0 - s.phase, stable: s.stable }));
        Ok(out)
    }

    /// The probability that a uniformly random initial phase difference ends in synchrony — the
    /// grey scale of Fig. 10 and the "10%" and "50%" of p. 290–291.
    ///
    /// It is 0 when synchrony is unstable, and otherwise the width `2φ_u` of synchrony's basin,
    /// `φ_u` the first unstable state above 0 in [`LewisRinzel::locked_states`] — antiphase
    /// itself when nothing lies between, which makes the probability 1.
    ///
    /// # Errors
    ///
    /// As [`LewisRinzel::locked_states`].
    pub fn probability_of_synchrony(&self, samples: usize) -> Result<f64, GapError> {
        let states = self.locked_states(samples)?;
        if !states[0].stable {
            return Ok(0.0);
        }
        Ok(2.0 * states[1..].iter().find(|s| !s.stable).map_or(0.5, |s| s.phase))
    }

    /// The drives in `[low, high]` at which antiphase changes stability — where `G′(½)` changes
    /// sign with every other parameter held — found on a grid of `samples` intervals and bisected.
    ///
    /// # Errors
    ///
    /// [`GapError::Empty`] unless `low < high`; [`GapError::TooFew`] for no samples; whatever
    /// [`LewisRinzel::check`] refuses at `low` or `high`.
    pub fn antiphase_critical_drives(&self, low: f64, high: f64, samples: usize) -> Result<Vec<f64>, GapError> {
        Self { drive: low, ..*self }.check()?;
        Self { drive: high, ..*self }.check()?;
        if low >= high {
            return Err(GapError::Empty { low, high });
        }
        let samples = at_least("samples", samples, 1)?;
        Ok(roots(|i| Self { drive: i, ..*self }.h_slope_at(0.5), low, high, samples))
    }

    /// Eq. (11) solved for the drive: the `I*_c` at which antiphase loses stability under
    /// electrical coupling alone, for spike effect `β`.
    ///
    /// Eq. (11) reads `β = (I − ½) ln(I/(I − 1)) − 1`. Since `I − ½ = ½ coth(T/2)`, it is
    /// `β = x coth x − 1` with `x = T/2`, and `x coth x` rises from 1 at `x = 0` without bound, so
    /// every `β > 0` has exactly one critical period, bisected here on `0 < x < 1 + β`; then
    /// `I = 1/(1 − e^{−2x})`. Below `x = 1` the left side is summed as `x C(x)/sinh x` with
    /// `C(x) = cosh x − sinh(x)/x = Σ 2n x^{2n}/(2n + 1)!`, a series of positive terms, because the
    /// direct `x/tanh x − 1` loses `ε/x²` of its digits to cancellation there. At `β = 0` there is
    /// no critical drive: antiphase is stable at every drive (p. 293).
    ///
    /// # Errors
    ///
    /// [`GapError::NotPositive`] for a `beta` that is not.
    pub fn critical_drive_electrical(beta: f64) -> Result<f64, GapError> {
        let beta = positive("beta", beta)?;
        let q = |x: f64| {
            if x < 1.0 {
                let (mut term, mut c) = (1.0, 0.0);
                for n in 1..=10 {
                    term *= x * x / ((2 * n) * (2 * n + 1)) as f64;
                    c += (2 * n) as f64 * term;
                }
                x * c / x.sinh()
            } else {
                x / x.tanh() - 1.0
            }
        };
        let x = bisect(|x| q(x) < beta, 0.0, 1.0 + beta);
        Ok(1.0 / -(-2.0 * x).exp_m1())
    }

    /// The antiphase orbit beyond weak coupling: section 4's matching conditions at `φ = ½`,
    /// solved for the period, or `None` when there is no physical one.
    ///
    /// Cell 1 fires at 0 and cell 2 at `T/2`; just before each spike the other cell sits at `u`,
    /// and the kick lifts it to `u + g_c β`. Integrating `v₊ = v₁ + v₂` (rate 1) and
    /// `v₋ = v₁ − v₂` (rate `μ = 1 + 2g_c`) across one half period gives
    ///
    /// ```text
    /// u + 1 = (u + g_c β)e^{−T/2} + 2I(1 − e^{−T/2}) − g_s ∫₀^{T/2} e^{−(T/2−t)}[s_T(t) + s_T(t + T/2)] dt
    /// u − 1 = −(u + g_c β)e^{−μT/2} − g_s ∫₀^{T/2} e^{−μ(T/2−t)}[s_T(t + T/2) − s_T(t)] dt
    /// ```
    ///
    /// The second gives `u` for each `T`; the first is then one equation in `T`, whose smallest
    /// root on a grid of `samples` intervals across `(0, t_max]` is bisected. The orbit is
    /// physical only if the kick does not capture the partner, `u + g_c β < 1` (p. 299).
    ///
    /// # Errors
    ///
    /// [`GapError::NotPositive`] for a `t_max` that is not; [`GapError::TooFew`] for no samples;
    /// whatever [`LewisRinzel::check`] refuses.
    pub fn antiphase_orbit(&self, t_max: f64, samples: usize) -> Result<Option<Orbit>, GapError> {
        self.check()?;
        let t_max = positive("t_max", t_max)?;
        let samples = at_least("samples", samples, 1)?;
        let (i, a, k, mu) = (self.drive, self.alpha, self.g_c * self.beta, 1.0 + 2.0 * self.g_c);
        let u_of = |t: f64| {
            let h = 0.5 * t;
            let d = leaky_s(a, t, mu, h, h) - leaky_s(a, t, mu, 0.0, h);
            (1.0 - k * (-mu * h).exp() - self.g_s * d) / (1.0 + (-mu * h).exp())
        };
        let residual = |t: f64| {
            let h = 0.5 * t;
            let s = leaky_s(a, t, 1.0, 0.0, h) + leaky_s(a, t, 1.0, h, h);
            let u = u_of(t);
            (u + k) * (-h).exp() - 2.0 * i * (-h).exp_m1() - self.g_s * s - (u + 1.0)
        };
        let Some(&period) = roots(residual, t_max / samples as f64, t_max, samples).first() else { return Ok(None) };
        let u = u_of(period);
        Ok(if u + k < 1.0 { Some(Orbit { period, u }) } else { None })
    }

    /// The synchronous period beyond weak coupling: both cells fire together, reset together,
    /// and never feel the junction, so `1 = I(1 − e^{−T}) − g_s ∫₀ᵀ e^{−(T−t)} s_T(t) dt`; the
    /// smallest root on a grid of `samples` intervals across `(0, t_max]`, or `None`.
    ///
    /// With `g_s = 0` this is the intrinsic period `ln(I/(I − 1))` whatever `g_c` and `β` are:
    /// "it does not affect the period of the synchronous state" (p. 301).
    ///
    /// # Errors
    ///
    /// As [`LewisRinzel::antiphase_orbit`].
    pub fn synchronous_period(&self, t_max: f64, samples: usize) -> Result<Option<f64>, GapError> {
        self.check()?;
        let t_max = positive("t_max", t_max)?;
        let samples = at_least("samples", samples, 1)?;
        let residual = |t: f64| -self.drive * (-t).exp_m1() - self.g_s * leaky_s(self.alpha, t, 1.0, 0.0, t) - 1.0;
        Ok(roots(residual, t_max / samples as f64, t_max, samples).first().copied())
    }
}

/// Lewis and Rinzel's eq. (5), integrated exactly between spikes.
///
/// Between spikes the pair is linear: the sum `v₁ + v₂` relaxes at rate 1 and the difference at
/// rate `1 + 2g_c`, each driven by the constant current and by the sum or difference of the two
/// alpha-function synaptic currents. Each synaptic current is `(p + q(t − t₀))e^{−α(t−t₀)}` — a
/// spike adds `α²` to `q` — and a membrane driven by such a current has a closed-form solution, so
/// every voltage below is exact to rounding at any time between events. Only the threshold
/// crossing is searched for: the voltages are sampled `samples` times per intrinsic period and the
/// first sample at or above 1 is bisected to the last bit. A crossing that enters and leaves
/// threshold between two samples is not seen.
///
/// At a spike the cell resets to 0 and its partner is kicked by `g_c β` (the `δ` term of eq. 5)
/// and starts receiving a new alpha current. A kick that carries the partner to threshold fires it
/// at the same instant — spike-capture synchrony (p. 299) — and that partner's kick back is not
/// applied to the cell that has just fired, since `Z = 0` at the spike.
#[derive(Debug, Clone, PartialEq)]
pub struct LewisRinzelPair {
    model: LewisRinzel,
    time: f64,
    v: [f64; 2],
    /// The synaptic current into each cell, as `[p, q]` in a frame starting at `time`.
    syn: [[f64; 2]; 2],
}

impl LewisRinzelPair {
    /// A pair at time 0 with voltages `v0` and no synaptic current, as Figs. 1 and 2 start.
    ///
    /// # Errors
    ///
    /// Whatever [`LewisRinzel::check`] refuses; [`GapError::NonFinite`] or
    /// [`GapError::AboveThreshold`] for a starting voltage.
    pub fn new(model: LewisRinzel, v0: [f64; 2]) -> Result<Self, GapError> {
        model.check()?;
        for (cell, &v) in v0.iter().enumerate() {
            finite("v0", v)?;
            if v >= 1.0 {
                return Err(GapError::AboveThreshold { cell, value: v });
            }
        }
        Ok(Self { model, time: 0.0, v: v0, syn: [[0.0; 2]; 2] })
    }

    /// The pair's parameters.
    #[must_use]
    pub fn model(&self) -> LewisRinzel {
        self.model
    }

    /// The current time, membrane time constants.
    #[must_use]
    pub fn time(&self) -> f64 {
        self.time
    }

    /// The two voltages now.
    #[must_use]
    pub fn voltages(&self) -> [f64; 2] {
        self.v
    }

    /// The voltages `s` after `time` if no cell fires, in closed form.
    fn after(&self, s: f64) -> [f64; 2] {
        let m = &self.model;
        let r = 1.0 + 2.0 * m.g_c;
        let (sum0, diff0) = (self.v[0] + self.v[1], self.v[0] - self.v[1]);
        let (p, q) = (self.syn[0][0] + self.syn[1][0], self.syn[0][1] + self.syn[1][1]);
        let (dp, dq) = (self.syn[0][0] - self.syn[1][0], self.syn[0][1] - self.syn[1][1]);
        let sum = 2.0 * m.drive + (sum0 - 2.0 * m.drive) * (-s).exp() - m.g_s * alpha_response(1.0, m.alpha, s, p, q);
        let diff = diff0 * (-r * s).exp() - m.g_s * alpha_response(r, m.alpha, s, dp, dq);
        [0.5 * (sum + diff), 0.5 * (sum - diff)]
    }

    /// Move `s` forward with no spike.
    fn advance(&mut self, s: f64) {
        self.v = self.after(s);
        let decay = (-self.model.alpha * s).exp();
        for syn in &mut self.syn {
            *syn = [(syn[0] + syn[1] * s) * decay, syn[1] * decay];
        }
        self.time += s;
    }

    /// Cell `j` fires now.
    fn fire(&mut self, j: usize, out: &mut Vec<Spike>) {
        let k = 1 - j;
        let (kick, rate) = (self.model.g_c * self.model.beta, self.model.alpha * self.model.alpha);
        self.v[j] = 0.0;
        self.v[k] += kick;
        self.syn[k][1] += rate;
        out.push(Spike { time: self.time, cell: j });
        if self.v[k] >= 1.0 {
            self.v[k] = 0.0;
            self.syn[j][1] += rate;
            out.push(Spike { time: self.time, cell: k });
        }
    }

    /// Run for `duration`, sampling the voltages `samples` times per intrinsic period to find each
    /// threshold crossing, and return the spikes in order.
    ///
    /// # Errors
    ///
    /// [`GapError::NotPositive`] for a `duration` that is not; [`GapError::TooFew`] for no samples;
    /// [`GapError::Stalled`] if the run needs more than seven events per sampling step on average,
    /// which only a coupling strong enough to fire a cell many times per step can cause.
    pub fn run(&mut self, duration: f64, samples: usize) -> Result<Vec<Spike>, GapError> {
        let duration = positive("duration", duration)?;
        let samples = at_least("samples", samples, 1)?;
        let steps = (duration * samples as f64 / self.model.t()).ceil() as usize;
        self.run_within(duration, samples, 8 * (steps + 1))
    }

    fn run_within(&mut self, duration: f64, samples: usize, budget: usize) -> Result<Vec<Spike>, GapError> {
        let h = self.model.t() / samples as f64;
        let end = self.time + duration;
        let mut out = Vec::new();
        for _ in 0..budget {
            let left = end - self.time;
            if left <= 0.0 {
                return Ok(out);
            }
            let s = h.min(left);
            let v = self.after(s);
            let mut first: Option<(f64, usize)> = None;
            for j in 0..2 {
                if v[j] >= 1.0 {
                    let t = bisect(|m| self.after(m)[j] < 1.0, 0.0, s);
                    if first.is_none_or(|(t0, _)| t < t0) {
                        first = Some((t, j));
                    }
                }
            }
            match first {
                None => self.advance(s),
                Some((t, j)) => {
                    self.advance(t);
                    self.fire(j, &mut out);
                }
            }
        }
        Err(GapError::Stalled { time: self.time })
    }
}

/// Chow and Kopell's integrate-and-fire neuron with a spike of finite width, coupled through gap
/// junctions: eqs. (2.3), (2.5) and (2.7)–(2.8), in their rescaled units.
///
/// When `v` reaches 1 the spike current `v_A e^{ξt}` switches on for a time `Δ`, and then a reset
/// current `−(1 + v_M)δ(t − Δ)` takes `1 + v_M` off the voltage. This module takes
/// `v_M ≡ η₊(Δ) = v_A(e^{ξΔ} − e^{−Δ})/(1 + ξ)`, the definition on p. 1650 that the kernels
/// (2.21) and (2.23) and the period (3.9) are exact for.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChowKopell {
    /// The spike amplitude scale `v_A`, not negative.
    pub v_a: f64,
    /// The spike's rise rate `ξ`, positive.
    pub xi: f64,
    /// The spike width `Δ`, from threshold to peak, positive.
    pub width: f64,
    /// The gap-junction conductance `g`, scaled by the leak; not negative.
    pub g: f64,
}

impl ChowKopell {
    /// Fig. 3: `Δ = 0.1`, `ξ = 50`, `g = 0.5`, `v_A = 1` — also Fig. 2's solid lines.
    pub const FIG3: Self = Self { v_a: 1.0, xi: 50.0, width: 0.1, g: 0.5 };
    /// Fig. 2's dashed lines: Fig. 3's spike with `g = 5`.
    pub const FIG2_STRONG: Self = Self { g: 5.0, ..Self::FIG3 };
    /// Fig. 1(a), one uncoupled cell at `I = 1.3`: `ξ = 50`, `Δ = 0.1`, `v_A = 1`.
    pub const FIG1_A: Self = Self { v_a: 1.0, xi: 50.0, width: 0.1, g: 0.0 };
    /// Fig. 1(b), `I = 1.3`: `ξ = 12`, `Δ = 0.1`, `v_A = 1`.
    pub const FIG1_B: Self = Self { xi: 12.0, ..Self::FIG1_A };
    /// Fig. 1(c), `I = 1.55`: `ξ = 12`, `Δ = 0.5`, `v_A = 1`.
    pub const FIG1_C: Self = Self { width: 0.5, ..Self::FIG1_B };
    /// Fig. 1(d) as its caption prints it, `I = 1.55`: `ξ = 12`, `Δ = 0.5`, `v_A = 0.1`.
    pub const FIG1_D: Self = Self { v_a: 0.1, ..Self::FIG1_C };

    /// A neuron and junction with every parameter checked.
    ///
    /// # Errors
    ///
    /// Whatever [`ChowKopell::check`] refuses.
    pub fn new(v_a: f64, xi: f64, width: f64, g: f64) -> Result<Self, GapError> {
        let m = Self { v_a, xi, width, g };
        m.check()?;
        Ok(m)
    }

    /// Every parameter in range, and a spike whose peak `v_M` is a finite number.
    ///
    /// # Errors
    ///
    /// [`GapError::Negative`] for `v_A` or `g`, [`GapError::NotPositive`] for `ξ` or `Δ`,
    /// [`GapError::NonFinite`] for a `v_M` that overflows.
    pub fn check(&self) -> Result<(), GapError> {
        non_negative("v_A", self.v_a)?;
        positive("xi", self.xi)?;
        positive("Delta", self.width)?;
        non_negative("g", self.g)?;
        finite("v_M", self.v_m())?;
        Ok(())
    }

    /// The spike-phase kernel `v_A(e^{ξt} − e^{−rt})/(r + ξ)`, `0 < t ≤ Δ`.
    fn spike_part(&self, t: f64, r: f64) -> f64 {
        self.v_a / (r + self.xi) * ((self.xi * t).exp() - (-r * t).exp())
    }

    /// `v_M ≡ η₊(Δ) = v_A(e^{ξΔ} − e^{−Δ})/(1 + ξ)` (p. 1650).
    #[must_use]
    pub fn v_m(&self) -> f64 {
        self.spike_part(self.width, 1.0)
    }

    /// The voltage of one uncoupled cell `t` into its spike, `0 ≤ t ≤ Δ`, having crossed threshold
    /// at `t = 0` under drive `I`: eq. (2.3) solved from `v(0) = 1`,
    /// `v(t) = 1 + (I − 1)(1 − e^{−t}) + v_A(e^{ξt} − e^{−t})/(1 + ξ)`. Its value at `Δ` is the
    /// spike's peak.
    ///
    /// # Errors
    ///
    /// [`GapError::NonFinite`] for the drive; [`GapError::OutOfRange`] for a `t` outside
    /// `[0, Δ]`; whatever [`ChowKopell::check`] refuses.
    pub fn spike_voltage(&self, drive: f64, t: f64) -> Result<f64, GapError> {
        self.check()?;
        let drive = finite("I", drive)?;
        if !(0.0..=self.width).contains(&t) {
            return Err(GapError::OutOfRange { what: "t", value: t, low: 0.0, high: self.width });
        }
        Ok(1.0 - (drive - 1.0) * (-t).exp_m1() + self.spike_part(t, 1.0))
    }

    /// The coefficient `−(1 + v_M − η_r(Δ))` of the kernel's tail `e^{−r(t−Δ)}`, eq. (2.14).
    fn tail(&self, r: f64) -> f64 {
        -(1.0 + self.v_m() - self.spike_part(self.width, r))
    }

    /// The response kernel `η_r(t) = ∫₀ᵗ e^{−r(t−t′)} A(t′) dt′`, eq. (2.14): the voltage a spike
    /// leaves in a mode that relaxes at rate `r`. Zero for `t ≤ 0`; `r = 1` is `η₊`, `r = 1 + 2g`
    /// is `η₋`. At `t = Δ` it takes the spike's value, the reset landing just after.
    ///
    /// # Errors
    ///
    /// [`GapError::NonFinite`] for `t`, [`GapError::NotPositive`] for `r`; whatever
    /// [`ChowKopell::check`] refuses.
    pub fn eta(&self, t: f64, r: f64) -> Result<f64, GapError> {
        self.check()?;
        Ok(self.eta_at(finite("t", t)?, positive("r", r)?))
    }

    fn eta_at(&self, t: f64, r: f64) -> f64 {
        if t <= 0.0 {
            0.0
        } else if t <= self.width {
            self.spike_part(t, r)
        } else {
            self.tail(r) * (-r * (t - self.width)).exp()
        }
    }

    /// `dη_r/dt` for `t ≥ 0`, from the left at `t = Δ` (the spike's cusp). At `t = 0` it is the
    /// limit from the right, `v_A(ξ + r)/(r + ξ) = v_A` in every mode — the quotient is formed
    /// first so that it is exactly `v_A` for every `r` — and so the slope of `γ_c = ½(η₊ − η₋)`
    /// there is exactly 0, as eq. (2.23) gives it (p. 1657). Every caller passes `t ≥ 0`: the
    /// cell's own spike at `c = φT ∈ [0, T]` and the lattice's first term at `T ± c ≥ 0`.
    fn eta_slope_at(&self, t: f64, r: f64) -> f64 {
        if t <= self.width {
            self.v_a * ((self.xi * (self.xi * t).exp() + r * (-r * t).exp()) / (r + self.xi))
        } else {
            -r * self.tail(r) * (-r * (t - self.width)).exp()
        }
    }

    /// `Σ_{l ≥ 1} η_r(lT + c)` (or of its slope), for `−T ≤ c ≤ T` and `T > Δ`: the first term is
    /// taken directly when its argument `T + c` is still inside the spike — only it can be, since
    /// `2T + c ≥ T > Δ` — and the exponential tails of the rest as a geometric series.
    fn lattice(&self, c: f64, period: f64, r: f64, slope: bool) -> f64 {
        let first = period + c;
        let (head, l0) = if first > self.width {
            (0.0, 1.0)
        } else if slope {
            (self.eta_slope_at(first, r), 2.0)
        } else {
            (self.eta_at(first, r), 2.0)
        };
        let coef = if slope { -r * self.tail(r) } else { self.tail(r) };
        head + coef * (-r * (l0 * period + c - self.width)).exp() / -(-r * period).exp_m1()
    }

    /// The rate of the difference modes in an all-to-all network of `n` cells, `1 + ng`.
    fn rate(&self, n: usize) -> f64 {
        1.0 + n as f64 * self.g
    }

    /// The spike-generation-and-reset kernel `γ_s = ½(η₊ + η₋)`, eq. (2.19), `r = 1 + 2g`.
    ///
    /// # Errors
    ///
    /// As [`ChowKopell::eta`].
    pub fn gamma_s(&self, t: f64) -> Result<f64, GapError> {
        self.network_gamma_s(2, t)
    }

    /// The coupling kernel `γ_c = ½(η₊ − η₋)`, eq. (2.20): the gap-junction analogue of a
    /// postsynaptic potential.
    ///
    /// # Errors
    ///
    /// As [`ChowKopell::eta`].
    pub fn gamma_c(&self, t: f64) -> Result<f64, GapError> {
        self.network_gamma_c(2, t)
    }

    /// `Γ_s = [η₊ + (n − 1)η₋]/n` with `r₋ = 1 + ng`: the voltage a cell's own spike leaves in it,
    /// in an all-to-all network of `n` cells. Eq. (4.6) prints it without the `1/n`.
    ///
    /// # Errors
    ///
    /// [`GapError::TooFew`] for `n = 0`; as [`ChowKopell::eta`].
    pub fn network_gamma_s(&self, n: usize, t: f64) -> Result<f64, GapError> {
        self.check()?;
        let t = finite("t", t)?;
        let nf = at_least("n", n, 1)? as f64;
        Ok((self.eta_at(t, 1.0) + (nf - 1.0) * self.eta_at(t, self.rate(n))) / nf)
    }

    /// `Γ_c = (η₊ − η₋)/n` with `r₋ = 1 + ng`: the voltage a spike leaves in each other cell of an
    /// all-to-all network of `n`. Eq. (4.7) prints it without the `1/n`.
    ///
    /// # Errors
    ///
    /// [`GapError::TooFew`] for `n = 0`; as [`ChowKopell::eta`].
    pub fn network_gamma_c(&self, n: usize, t: f64) -> Result<f64, GapError> {
        self.check()?;
        let t = finite("t", t)?;
        let nf = at_least("n", n, 1)? as f64;
        Ok((self.eta_at(t, 1.0) - self.eta_at(t, self.rate(n))) / nf)
    }

    /// `δ_c = 1 + 2γ_c(Δ)`, eq. (2.22): the coefficient of the difference-mode term, the one
    /// relaxing at rate `r = 1 + 2g`, in the coupling kernel's tail after the spike,
    /// `γ_c = −½[e^{−(t−Δ)} − δ_c e^{−r(t−Δ)}]` (2.23).
    ///
    /// # Errors
    ///
    /// Whatever [`ChowKopell::check`] refuses.
    pub fn delta_c(&self) -> Result<f64, GapError> {
        self.check()?;
        Ok(1.0 + self.v_m() - self.spike_part(self.width, self.rate(2)))
    }

    fn check_period(&self, period: f64) -> Result<f64, GapError> {
        self.check()?;
        let period = finite("T", period)?;
        if period > self.width { Ok(period) } else { Err(GapError::PeriodTooShort { period, limit: self.width }) }
    }

    /// `2G(φ, T)` at `c = φT`, in kernel sums.
    fn g_at(&self, c: f64, period: f64) -> f64 {
        let r = self.rate(2);
        let own = self.eta_at(c, 1.0) - self.eta_at(c, r);
        let ahead = self.lattice(c, period, 1.0, false) - self.lattice(c, period, r, false);
        let behind = self.lattice(-c, period, 1.0, false) - self.lattice(-c, period, r, false);
        own + ahead - behind
    }

    /// `(2/T)∂G/∂φ` at `c = φT`, in kernel sums.
    fn g_slope_at(&self, c: f64, period: f64) -> f64 {
        let r = self.rate(2);
        let own = self.eta_slope_at(c, 1.0) - self.eta_slope_at(c, r);
        let ahead = self.lattice(c, period, 1.0, true) - self.lattice(c, period, r, true);
        let behind = self.lattice(-c, period, 1.0, true) - self.lattice(-c, period, r, true);
        own + ahead + behind
    }

    /// The locking function `G(φ, T) = γ_c(φT) + Σ_{l≥1}[γ_c(lT + φT) − γ_c(lT − φT)]`, eq. (3.4):
    /// a pair can lock at phase difference `φ` with period `T` exactly when `G(φ, T)` equals
    /// `(I₁ − I₂)/(1 + 2g)`, eq. (3.3) — zero for identical cells. `φ` is taken modulo 1.
    ///
    /// # Errors
    ///
    /// [`GapError::PeriodTooShort`] unless `T > Δ`; [`GapError::NonFinite`] for `φ` or `T`;
    /// whatever [`ChowKopell::check`] refuses.
    pub fn locking(&self, phi: f64, period: f64) -> Result<f64, GapError> {
        let period = self.check_period(period)?;
        let c = finite("phi", phi)?.rem_euclid(1.0) * period;
        Ok(0.5 * self.g_at(c, period))
    }

    /// `∂G/∂φ = T[γ̇_c(φT) + Σ_{l≥1}(γ̇_c(lT + φT) + γ̇_c(lT − φT))]`, eq. (3.21) at any `φ`. A
    /// locked state with `∂G/∂φ < 0` is unstable; `> 0` is necessary for stability (§3.1).
    ///
    /// # Errors
    ///
    /// As [`ChowKopell::locking`].
    pub fn locking_slope(&self, phi: f64, period: f64) -> Result<f64, GapError> {
        let period = self.check_period(period)?;
        let c = finite("phi", phi)?.rem_euclid(1.0) * period;
        Ok(0.5 * period * self.g_slope_at(c, period))
    }

    /// The period function `F(φ, T) = ½γ_c(φT) + Σ_{l≥1}[γ_s(lT) + ½(γ_c(lT − φT) + γ_c(lT + φT))]`,
    /// eq. (3.6): a locked pair with mean drive `Ī` has period `T` where `F(φ, T) = 1 − Ī`, eq. (3.5).
    ///
    /// # Errors
    ///
    /// As [`ChowKopell::locking`].
    pub fn period_function(&self, phi: f64, period: f64) -> Result<f64, GapError> {
        let period = self.check_period(period)?;
        let c = finite("phi", phi)?.rem_euclid(1.0) * period;
        Ok(self.f_at(c, period))
    }

    fn f_at(&self, c: f64, period: f64) -> f64 {
        let r = self.rate(2);
        let own = self.eta_at(c, 1.0) - self.eta_at(c, r);
        let reset = self.lattice(0.0, period, 1.0, false) + self.lattice(0.0, period, r, false);
        let partner = self.lattice(c, period, 1.0, false) - self.lattice(c, period, r, false) + self.lattice(-c, period, 1.0, false)
            - self.lattice(-c, period, r, false);
        0.25 * own + 0.5 * reset + 0.25 * partner
    }

    /// The synchronous period, eq. (3.9): `T_S = ln((I − 1 + e^Δ)/(I − 1))` — the uncoupled period,
    /// since in synchrony no current crosses the junction.
    ///
    /// It exceeds the spike width, as a period must, only for `I < 1 + 1/(1 − e^{−Δ})`. The paper
    /// prints that bound as `1 + (1 − e^Δ)^{−1}` (p. 1653), which is below 1.
    ///
    /// # Errors
    ///
    /// [`GapError::NonFinite`] or [`GapError::Subthreshold`] for the drive,
    /// [`GapError::Overdriven`] for one at or above `1 + 1/(1 − e^{−Δ})`; whatever
    /// [`ChowKopell::check`] refuses.
    pub fn sync_period(&self, drive: f64) -> Result<f64, GapError> {
        self.check()?;
        if finite("I", drive)? <= 1.0 {
            return Err(GapError::Subthreshold { drive });
        }
        let limit = 1.0 + 1.0 / -(-self.width).exp_m1();
        if drive >= limit {
            return Err(GapError::Overdriven { drive, limit });
        }
        Ok(((drive - 1.0 + self.width.exp()) / (drive - 1.0)).ln())
    }

    /// The antiphase period at drive `I`: the smallest `T` in `[Δ, t_max]` with
    /// `F(½, T) = 1 − I`, eq. (3.5) at `φ = ½` — the "only physical solution" (p. 1653). Found on a
    /// grid of `samples` intervals and bisected; `None` when there is none on the interval.
    ///
    /// # Errors
    ///
    /// [`GapError::NonFinite`] for the drive or `t_max`; [`GapError::Empty`] unless `t_max > Δ`;
    /// [`GapError::TooFew`] for no samples; whatever [`ChowKopell::check`] refuses.
    pub fn antiphase_period(&self, drive: f64, t_max: f64, samples: usize) -> Result<Option<f64>, GapError> {
        self.check()?;
        let drive = finite("I", drive)?;
        let t_max = finite("t_max", t_max)?;
        if t_max <= self.width {
            return Err(GapError::Empty { low: self.width, high: t_max });
        }
        let samples = at_least("samples", samples, 1)?;
        Ok(roots(|t| self.f_at(0.5 * t, t) - (1.0 - drive), self.width, t_max, samples).first().copied())
    }

    /// The periods in `[low, high]` at which `∂G/∂φ` at phase `phi` changes sign — the critical
    /// periods of §3.4–3.6 — found on a grid of `samples` intervals and bisected. At `φ = 0` the
    /// one root is `T_C^S` of eq. (3.23); at `φ = ½` they are `T_C^{AS3}`, the cusp's
    /// `T_C^{AS2} = 2Δ` where the slope jumps sign rather than crossing zero, and `T_C^{AS1}` of
    /// eq. (3.29).
    ///
    /// # Errors
    ///
    /// [`GapError::PeriodTooShort`] unless `low > Δ`; [`GapError::Empty`] unless `low < high`;
    /// [`GapError::TooFew`] for no samples; as [`ChowKopell::locking`].
    pub fn slope_switches(&self, phi: f64, low: f64, high: f64, samples: usize) -> Result<Vec<f64>, GapError> {
        let low = self.check_period(low)?;
        let high = finite("high", high)?;
        let phi = finite("phi", phi)?.rem_euclid(1.0);
        if low >= high {
            return Err(GapError::Empty { low, high });
        }
        let samples = at_least("samples", samples, 1)?;
        Ok(roots(|t| self.g_slope_at(phi * t, t), low, high, samples))
    }

    /// `t_min = Δ + ln(δ_c(1 + 2g))/(2g)`, eq. (3.24): where the coupling kernel `γ_c` bottoms out,
    /// after which it rises for good. Firing separated by more than `t_min` satisfies Chow's
    /// sufficient condition for stable locking.
    ///
    /// # Errors
    ///
    /// [`GapError::NotPositive`] for `g = 0`, where the kernel is zero; whatever
    /// [`ChowKopell::check`] refuses.
    pub fn t_min(&self) -> Result<f64, GapError> {
        let dc = self.delta_c()?;
        let g = positive("g", self.g)?;
        Ok(self.width + (dc * (1.0 + 2.0 * g)).ln() / (2.0 * g))
    }

    /// The voltage `v₁(T/2 + Δ)` of one cell of an antiphase pair with period `T` at the peak of its
    /// partner's spike, eq. (3.15), with the drive the period needs from eq. (3.5). Antiphase
    /// exists only while this stays below 1 (§3.3).
    ///
    /// # Errors
    ///
    /// [`GapError::PeriodTooShort`] unless `T > 2Δ`; as [`ChowKopell::locking`].
    pub fn antiphase_partner_peak(&self, period: f64) -> Result<f64, GapError> {
        self.check()?;
        let period = finite("T", period)?;
        let limit = 2.0 * self.width;
        if period <= limit {
            return Err(GapError::PeriodTooShort { period, limit });
        }
        let r = self.rate(2);
        let drive = 1.0 - self.f_at(0.5 * period, period);
        let at = 0.5 * period + self.width;
        let own = self.eta_at(at, 1.0) + self.eta_at(at, r) + self.lattice(at, period, 1.0, false) + self.lattice(at, period, r, false);
        let partner = self.eta_at(self.width, 1.0) - self.eta_at(self.width, r) + self.lattice(self.width, period, 1.0, false)
            - self.lattice(self.width, period, r, false);
        Ok(drive + 0.5 * (own + partner))
    }

    /// The drive at which `n` identical all-to-all cells fire in the splay state with period `T`,
    /// cell `j` at `jT/n`: eq. (4.9), `1 = I + Σ_{l≥1} Γ_s(lT) + Σ_{j=1}^{n−1} Σ_{m≥1} Γ_c(mT − jT/n)`,
    /// with the kernels normalised as they must be ([`ChowKopell::network_gamma_s`]).
    ///
    /// # Errors
    ///
    /// [`GapError::TooFew`] for `n = 0`; [`GapError::PeriodTooShort`] unless `T > Δ`; whatever
    /// [`ChowKopell::check`] refuses.
    pub fn splay_drive(&self, n: usize, period: f64) -> Result<f64, GapError> {
        let period = self.check_period(period)?;
        let nf = at_least("n", n, 1)? as f64;
        let r = self.rate(n);
        let own = (self.lattice(0.0, period, 1.0, false) + (nf - 1.0) * self.lattice(0.0, period, r, false)) / nf;
        let mut others = 0.0;
        for j in 1..n {
            let c = -(j as f64) * period / nf;
            others += (self.lattice(c, period, 1.0, false) - self.lattice(c, period, r, false)) / nf;
        }
        Ok(1.0 - own - others)
    }
}

/// What the reset at the end of a Chow–Kopell spike takes off the voltage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reset {
    /// `1 + v_M` with `v_M ≡ η₊(Δ)` (p. 1650): the reset the kernels and every closed form of
    /// sections 3 and 4 assume. An uncoupled cell lands at `(I − 1)(1 − e^{−Δ})`, not 0.
    Kernel,
    /// `1 + v_M` with `v_M = v(Δ) − 1` of eq. (2.6), which adds `(I − 1)(1 − e^{−Δ})` for a cell of
    /// drive `I`: an uncoupled cell lands exactly at 0, as Fig. 1 is drawn.
    ToZero,
}

/// Chow and Kopell's eq. (4.1): `n` cells coupled all to all through gap junctions, each firing a
/// spike of width `Δ`, integrated exactly between events.
///
/// The network is linear between events: the mean voltage relaxes at rate 1 and every deviation
/// from it at rate `1 + ng`, each driven by its share of the drives and of the spike currents
/// `v_A e^{ξ(t − t₀)}` of the cells in mid-spike, all in closed form. The events are threshold
/// crossings, searched for by sampling every `step` and bisecting to the last bit, and spike ends,
/// which are known in advance: at `t₀ + Δ` the reset takes `1 + v_M` off the cell.
#[derive(Debug, Clone, PartialEq)]
pub struct ChowKopellNet {
    model: ChowKopell,
    time: f64,
    drives: Vec<f64>,
    v: Vec<f64>,
    /// When each cell's spike in progress ends, if one is.
    ends: Vec<Option<f64>>,
    /// When each cell's spike in progress began.
    onsets: Vec<f64>,
    reset: Reset,
}

impl ChowKopellNet {
    /// A network at time 0, cell `i` with drive `drives[i]` and voltage `v0[i]`, no cell in a spike.
    ///
    /// # Errors
    ///
    /// [`GapError::TooFew`] for no cells; [`GapError::Mismatch`] for lists of different lengths;
    /// [`GapError::NonFinite`] for a drive or voltage; [`GapError::AboveThreshold`] for a voltage
    /// not below 1; whatever [`ChowKopell::check`] refuses.
    pub fn new(model: ChowKopell, drives: Vec<f64>, v0: Vec<f64>) -> Result<Self, GapError> {
        model.check()?;
        let n = at_least("cells", drives.len(), 1)?;
        if v0.len() != n {
            return Err(GapError::Mismatch { drives: n, voltages: v0.len() });
        }
        for &d in &drives {
            finite("I", d)?;
        }
        for (cell, &v) in v0.iter().enumerate() {
            finite("v0", v)?;
            if v >= 1.0 {
                return Err(GapError::AboveThreshold { cell, value: v });
            }
        }
        Ok(Self { model, time: 0.0, drives, v: v0, ends: vec![None; n], onsets: vec![0.0; n], reset: Reset::Kernel })
    }

    /// The same network with the spike's reset chosen; [`Reset::Kernel`] unless this is called.
    #[must_use]
    pub fn with_reset(mut self, reset: Reset) -> Self {
        self.reset = reset;
        self
    }

    /// The current time, membrane time constants.
    #[must_use]
    pub fn time(&self) -> f64 {
        self.time
    }

    /// The voltages now.
    #[must_use]
    pub fn voltages(&self) -> &[f64] {
        &self.v
    }

    /// The voltages `s` after `time` if no event intervenes, in closed form.
    fn after(&self, s: f64) -> Vec<f64> {
        let m = &self.model;
        let n = self.v.len() as f64;
        let big_r = m.rate(self.v.len());
        let mean = |x: &[f64]| x.iter().sum::<f64>() / n;
        let a: Vec<f64> =
            self.ends.iter().zip(&self.onsets).map(|(e, &t0)| if e.is_some() { m.v_a * (m.xi * (self.time - t0)).exp() } else { 0.0 }).collect();
        let (i_bar, v_bar, a_bar) = (mean(&self.drives), mean(&self.v), mean(&a));
        let (e1, er, ex) = ((-s).exp(), (-big_r * s).exp(), (m.xi * s).exp());
        let (k1, kr) = ((ex - e1) / (m.xi + 1.0), (ex - er) / (m.xi + big_r));
        let common = i_bar + (v_bar - i_bar) * e1 + a_bar * k1;
        (0..self.v.len())
            .map(|i| {
                let fixed = (self.drives[i] - i_bar) / big_r;
                common + fixed + (self.v[i] - v_bar - fixed) * er + (a[i] - a_bar) * kr
            })
            .collect()
    }

    fn advance(&mut self, s: f64) {
        self.v = self.after(s);
        self.time += s;
    }

    /// Run for `duration`, sampling every `step` for threshold crossings, and return the spikes —
    /// each at its threshold crossing — in order.
    ///
    /// # Errors
    ///
    /// [`GapError::NotPositive`] for a `duration` or `step` that is not; [`GapError::Stalled`] only
    /// for a `step` too small to move the clock (below the spacing of floats near the current
    /// time). Otherwise the event budget is sufficient by construction: each cell's spikes are at
    /// least `Δ` apart, since none can start before the last one's reset, and every sampling step,
    /// crossing and reset costs one pass.
    pub fn run(&mut self, duration: f64, step: f64) -> Result<Vec<Spike>, GapError> {
        let duration = positive("duration", duration)?;
        let step = positive("step", step)?;
        let n = self.v.len();
        let steps = (duration / step).ceil() as usize;
        let spikes = n * ((duration / self.model.width).ceil() as usize + 1);
        self.run_within(duration, step, steps + 4 * spikes + 1)
    }

    fn run_within(&mut self, duration: f64, step: f64, budget: usize) -> Result<Vec<Spike>, GapError> {
        let width = self.model.width;
        let end = self.time + duration;
        let n = self.v.len();
        let mut out = Vec::new();
        for _ in 0..budget {
            for i in 0..n {
                if self.ends[i].is_none() && self.v[i] >= 1.0 {
                    self.ends[i] = Some(self.time + width);
                    self.onsets[i] = self.time;
                    out.push(Spike { time: self.time, cell: i });
                }
            }
            if self.time >= end {
                return Ok(out);
            }
            let horizon = self.ends.iter().flatten().fold(end, |a, &b| a.min(b));
            let target = (self.time + step).min(horizon);
            let s = target - self.time;
            let v = self.after(s);
            let mut first: Option<f64> = None;
            for i in 0..n {
                if self.ends[i].is_none() && v[i] >= 1.0 {
                    let t = bisect(|m| self.after(m)[i] < 1.0, 0.0, s);
                    if first.is_none_or(|t0| t < t0) {
                        first = Some(t);
                    }
                }
            }
            if let Some(t) = first {
                self.advance(t);
                continue;
            }
            self.advance(s);
            self.time = target;
            let drift = -(-width).exp_m1();
            for i in 0..n {
                if self.ends[i] == Some(target) {
                    let extra = if self.reset == Reset::ToZero { (self.drives[i] - 1.0) * drift } else { 0.0 };
                    self.v[i] -= 1.0 + self.model.v_m() + extra;
                    self.ends[i] = None;
                }
            }
        }
        Err(GapError::Stalled { time: self.time })
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ChowKopell, ChowKopellNet, GapError, LewisRinzel, LewisRinzelPair, LockedState, Reset, Spike, alpha_response, bisect,
        leaky_s, phi1, phi2, s_periodic,
    };

    /// Composite Simpson's rule on `n` (even) intervals.
    fn simpson(f: impl Fn(f64) -> f64, a: f64, b: f64, n: usize) -> f64 {
        let h = (b - a) / n as f64;
        let mut s = f(a) + f(b);
        for k in 1..n {
            s += if k % 2 == 1 { 4.0 } else { 2.0 } * f(a + h * k as f64);
        }
        s * h / 3.0
    }

    /// For each spike of cell 0 followed by another: its time, the interval to the next, and the
    /// delay to cell 1's next spike as a fraction of that interval — the phase difference `φ`.
    fn phases(spikes: &[Spike]) -> Vec<(f64, f64, f64)> {
        let t0: Vec<f64> = spikes.iter().filter(|s| s.cell == 0).map(|s| s.time).collect();
        let t1: Vec<f64> = spikes.iter().filter(|s| s.cell == 1).map(|s| s.time).collect();
        t0.windows(2)
            .filter_map(|w| t1.iter().find(|&&u| u >= w[0]).map(|&u| (w[0], w[1] - w[0], (u - w[0]) / (w[1] - w[0]))))
            .collect()
    }

    /// The appendix's derivation, step by step: the period, the limit cycle reaching threshold
    /// at it, `Z = 1/(T dv_LC/dt)`, and a finite kick's phase advance per unit kick converging on
    /// `Z` at first order with the second-order term the expansion predicts.
    ///
    /// `Δφ = ln(I/(I − εeᵗ))/T = εeᵗ/(IT) + ε²e^{2t}/(2I²T) + O(ε³)`, so `(Δφ/ε − Z)/ε` must tend
    /// to `e^{2t}/(2I²T)`; at `ε = 10⁻⁴` it does to a relative 2.1 × 10⁻⁴ (measured), the size of
    /// the next term. The appendix prints the period as `T = (ln(I/(I − 1)))⁻¹` (p. 307), which is
    /// eq. (2)'s FREQUENCY: Figs. 4 and 7 label their top axes with it, and 0.56, 0.80, 1.02,
    /// 1.23 and 1.44 at `I` = 1.2, 1.4, 1.6, 1.8 and 2 are `1/T` to the printed digits.
    #[test]
    fn the_appendix_gives_the_period_and_the_phase_resetting_curve() {
        let m = LewisRinzel::new(1.3, 0.0, 0.0, 1.0, 1.0).unwrap();
        let t = m.period().unwrap();
        assert_eq!(t, (1.3f64 / (1.3 - 1.0)).ln());
        assert!((m.limit_cycle(t * (1.0 - 1e-15)).unwrap() - 1.0).abs() < 1e-14, "v_LC reaches 1 at T");
        assert_eq!(m.limit_cycle(0.0).unwrap(), 0.0);
        assert!((m.limit_cycle(0.5 + 3.0 * t).unwrap() - 1.3 * (1.0 - (-0.5f64).exp())).abs() < 1e-14, "periodic");
        for k in 1..20 {
            let s = t * f64::from(k) / 20.0;
            let slope = 1.3 * (-s).exp();
            assert!((m.prc(s).unwrap() * t * slope - 1.0).abs() < 1e-14, "Z = 1/(T v') at {s}");
            let eps = 1e-4;
            let second = (m.phase_advance(eps, s).unwrap() / eps - m.prc(s).unwrap()) / eps;
            let want = (2.0 * s).exp() / (2.0 * 1.3 * 1.3 * t);
            assert!((second / want - 1.0).abs() < 4e-4, "t = {s}: {second} against {want}");
        }
        assert_eq!(m.prc(0.0).unwrap(), 0.0, "Z = 0 at the spike");
        assert_eq!(m.prc(2.0 * t).unwrap(), 0.0, "and at every later spike");
        for (i, f) in [(1.2, "0.56"), (1.4, "0.80"), (1.6, "1.02"), (1.8, "1.23"), (2.0, "1.44")] {
            let period = LewisRinzel { drive: i, ..m }.period().unwrap();
            assert_eq!(format!("{:.2}", 1.0 / period), f, "the axis label is 1/T");
        }
    }

    /// Past `t̃ = ln(I/(I − 1 + ε))` the kick alone carries the cell over threshold and the advance
    /// falls linearly to 0 at `T`. The appendix starts that fall at `ln((I − 1)/(I + ε(I/(I − 1 + ε))))/T`
    /// (p. 308), which is NEGATIVE for every `ε > 0`; the two branches of its own `Δφ` meet at
    /// `ln((I − 1 + ε)/(I − 1))/T`, the value `(T − t̃)/T` takes there.
    #[test]
    fn the_appendix_starts_its_linear_fall_at_a_negative_value() {
        let m = LewisRinzel::new(1.3, 0.0, 0.0, 1.0, 1.0).unwrap();
        let (i, t) = (1.3f64, m.period().unwrap());
        for eps in [0.01, 0.1, 0.25] {
            let edge = (i / (i - 1.0 + eps)).ln();
            let below = m.phase_advance(eps, edge * (1.0 - 1e-12)).unwrap();
            let above = m.phase_advance(eps, edge * (1.0 + 1e-12)).unwrap();
            let meet = ((i - 1.0 + eps) / (i - 1.0)).ln() / t;
            assert!((below - meet).abs() < 5e-12 && (above - meet).abs() < 5e-12, "{below} {above} {meet}");
            assert!((above - (t - edge * (1.0 + 1e-12)) / t).abs() < 1e-15, "the linear branch");
            let printed = ((i - 1.0) / (i + eps * (i / (i - 1.0 + eps)))).ln() / t;
            assert!(printed < 0.0, "ε = {eps}: the printed start {printed} is a delay");
        }
        // A kick of exactly 1 at the reset carries the cell exactly to threshold: a whole cycle.
        assert_eq!(m.phase_advance(1.0, 0.0).unwrap(), 1.0);
        // A kick that lands exactly on threshold fires the cell: the linear branch, to the bit. The
        // two branches agree there in exact arithmetic; at I = 1.5, t = 1/16 their roundings differ.
        let m15 = LewisRinzel { drive: 1.5, ..m };
        let (t15, u) = (m15.period().unwrap(), 0.0625);
        let eps = 1.0 - m15.limit_cycle(u).unwrap();
        assert_eq!(m15.limit_cycle(u).unwrap() + eps, 1.0);
        assert_eq!(m15.phase_advance(eps, u).unwrap(), (t15 - u) / t15);
        assert!(m.phase_advance(-0.1, 1.0).unwrap() < 0.0, "a hyperpolarising kick delays");
    }

    /// Eq. (4)'s periodic current is the sum of the alpha functions of every earlier spike, it is
    /// continuous across the spike (each alpha function starts at 0), and each spike injects unit
    /// charge — the normalisation p. 285 states — which the closed-form membrane integral
    /// [`leaky_s`] with no leak (`κ = 0`) returns to 2 × 10⁻¹⁶.
    #[test]
    fn the_synaptic_current_is_the_periodic_sum_of_alpha_functions() {
        for (a, i) in [(0.5, 1.1), (3.0, 1.6), (4.0, 1.2)] {
            let m = LewisRinzel { drive: i, g_c: 0.0, beta: 0.0, g_s: 1.0, alpha: a };
            let t = m.period().unwrap();
            for k in 0..10 {
                let u = t * f64::from(k) / 10.0;
                let direct: f64 = (0..600).map(|n| u + f64::from(n) * t).map(|x| a * a * x * (-a * x).exp()).sum();
                assert!((m.synaptic_current(u).unwrap() / direct - 1.0).abs() < 2e-15, "α = {a}, t = {u}");
            }
            assert!((s_periodic(a, t, 0.0) / s_periodic(a, t, t) - 1.0).abs() < 1e-15, "continuous across the spike");
            assert!((leaky_s(a, t, 0.0, 0.0, t) - 1.0).abs() < 1e-15, "unit charge: {}", leaky_s(a, t, 0.0, 0.0, t));
            let quad = simpson(|u| s_periodic(a, t, u), 0.0, t, 4000);
            assert!((quad - 1.0).abs() < 1e-12, "{quad}");
        }
    }

    /// The special functions and the membrane integral against their definitions: `φ₁`, `φ₂` by
    /// quadrature of `∫₀¹ uᵏ e^{yu} du` on both sides of the series branch at `|y| = ½`, and
    /// [`alpha_response`] against Simpson's rule on its defining integral on both sides of `κ = α`.
    #[test]
    fn the_special_functions_are_their_integrals() {
        for y in [-40.0, -3.0, -0.75, -0.5, -0.49, -0.25, -1e-9, 0.0, 1e-9, 0.25, 0.49, 0.5, 0.75, 3.0, 20.0] {
            let one = simpson(|u| (y * u).exp(), 0.0, 1.0, 20000);
            let two = simpson(|u| u * (y * u).exp(), 0.0, 1.0, 20000);
            // At |y| = 40 Simpson's rule on e^{yu} is itself the limit (measured 3.0 × 10⁻¹³); elsewhere
            // the residual is at most 1.4 × 10⁻¹⁴.
            let tol = if y.abs() > 10.0 { 1e-12 } else { 5e-14 };
            assert!((phi1(y) / one - 1.0).abs() < tol, "φ₁({y})");
            assert!((phi2(y) / two - 1.0).abs() < tol, "φ₂({y}): {} against {two}", phi2(y));
        }
        assert_eq!((phi1(0.0), phi2(0.0)), (1.0, 0.5));
        // A fast membrane (κ − α = 799.5): the direct form would overflow e^{(κ−α)s}; the
        // rearranged one stays finite and right.
        let fast = alpha_response(800.0, 0.5, 1.0, 1.0, 1.0);
        let want = simpson(|u| (-800.0 * (1.0 - u)).exp() * (1.0 + u) * (-0.5 * u).exp(), 0.0, 1.0, 200_000);
        assert!(fast.is_finite() && (fast / want - 1.0).abs() < 1e-9, "{fast} {want}");
        for (kappa, alpha) in [(1.0, 3.0), (3.0, 1.0), (1.4, 1.4), (2.0, 0.5), (0.0, 2.0)] {
            for s in [0.3, 1.7] {
                let (p, q) = (0.4, 2.5);
                let want = simpson(|u| (-kappa * (s - u)).exp() * (p + q * u) * (-alpha * u).exp(), 0.0, s, 4000);
                assert!((alpha_response(kappa, alpha, s, p, q) - want).abs() < 1e-14, "κ {kappa}, α {alpha}, s {s}");
            }
        }
    }

    /// `H(ψ)` in closed form is eq. (7)'s average of the phase-resetting curve times the coupling
    /// current, done here by Simpson's rule on the two smooth stretches either side of the
    /// partner's reset at `(1 − ψ)T`, the kick's `δ` adding `g_c β Z((1 − ψ)T)/T` — and against
    /// `SciPy`'s adaptive quadrature of the same integral (`tools/gapjunction_reference.py`, B).
    ///
    /// Simpson's residual here is 1.7 × 10⁻¹⁴ (measured); `SciPy`'s numbers agree to 2.2 × 10⁻¹⁶.
    #[test]
    fn h_is_the_average_of_the_prc_times_the_coupling_current() {
        let quadrature = |m: &LewisRinzel, psi: f64| {
            let (i, t, a) = (m.drive, m.period().unwrap(), m.alpha);
            let p = psi.rem_euclid(1.0);
            let z = |x: f64| x.exp() / (i * t);
            let v = |x: f64| i * (1.0 - (-x).exp());
            // Until the partner's reset at (1 − p)T its phase is x + pT; after it, x + pT − T.
            let before = |x: f64| z(x) * (-m.g_s * s_periodic(a, t, x + p * t) + m.g_c * (v(x + p * t) - v(x)));
            let after = |x: f64| z(x) * (-m.g_s * s_periodic(a, t, x + p * t - t) + m.g_c * (v(x + p * t - t) - v(x)));
            let smooth = simpson(before, 0.0, (1.0 - p) * t, 4000) + if p > 0.0 { simpson(after, (1.0 - p) * t, t, 4000) } else { 0.0 };
            let kick = if p > 0.0 { m.g_c * m.beta * z((1.0 - p) * t) } else { 0.0 };
            (smooth + kick) / t
        };
        let mut worst: f64 = 0.0;
        for m in [
            LewisRinzel::new(1.2, 0.3, 0.2, 0.7, 4.0).unwrap(),
            LewisRinzel::new(1.6, 1.0, 0.1, -0.4, 0.5).unwrap(),
            LewisRinzel::new(1.05, 0.0, 0.0, 1.0, 1.0).unwrap(),
        ] {
            for psi in [0.0, 0.1, 0.125, 0.3, 0.5, 0.7, 0.875, 0.95] {
                worst = worst.max((m.h(psi).unwrap() - quadrature(&m, psi)).abs());
            }
        }
        assert!(worst < 1e-13, "{worst}");
        let scipy = [
            ((1.2, 0.3, 0.2, 0.7, 4.0), [-0.3150676389953462, -0.4185919565527859, -0.65914369220735, -0.3874054603913334]),
            ((1.6, 1.0, 0.1, -0.4, 0.5), [0.4413055912199761, 0.5237324143972376, 0.422827569249319, 0.4725728012685708]),
            ((1.05, 0.0, 0.0, 1.0, 1.0), [-0.5500000000000002, -0.568138534218401, -0.7746735103377728, -0.6327086162670831]),
        ];
        for ((i, gc, beta, gs, a), want) in scipy {
            let m = LewisRinzel::new(i, gc, beta, gs, a).unwrap();
            for (psi, w) in [0.0, 0.125, 0.5, 0.875].into_iter().zip(want) {
                assert!((m.h(psi).unwrap() - w).abs() < 1e-15, "({i}, {gc}, {beta}, {gs}, {a}) H({psi})");
            }
        }
        // H(1⁻) is the limit from below, and H is 1-periodic.
        let m = LewisRinzel::new(1.2, 0.3, 0.2, 0.7, 4.0).unwrap();
        assert!((m.h(-1e-12).unwrap() - m.h(1.0 - 1e-12).unwrap()).abs() < 1e-11);
        assert!((m.h(2.3).unwrap() - m.h(0.3).unwrap()).abs() < 1e-14);
    }

    /// At `α = 1` the synaptic `H` at zero phase is `−g_s(I − ½)` exactly: `F(T) = T²(1 + q)/(2(1 − q)²)`
    /// with `q = e^{−T} = (I − 1)/I`. The closed form reaches it through the `y = 0` limits of
    /// `φ₁` and `φ₂`, where eq. (9) — with its `(1 − α)²` in a denominator — is `0/0`.
    #[test]
    fn at_alpha_one_the_synaptic_h_is_minus_g_s_times_i_minus_a_half() {
        for i in [1.05, 1.3, 2.0, 7.0] {
            let m = LewisRinzel { drive: i, g_c: 0.0, beta: 0.0, g_s: 0.8, alpha: 1.0 };
            assert!((m.h(0.0).unwrap() + 0.8 * (i - 0.5)).abs() < 3e-15 * i, "I = {i}: {}", m.h(0.0).unwrap());
        }
    }

    /// Eq. (10), `G_c`, and eq. (9), `G_s`, transcribed from the page, against `H(−φ) − H(φ)`.
    ///
    /// Both closed forms are right. Eq. (9)'s integral line omits the `−g_s` that eq. (5) puts on
    /// the inhibition, and eq. (12)'s writes `+g_s`; its result line carries the `−g_s`, and that
    /// is what matches. Eq. (9) cannot be evaluated at `α = 1`, where it is `0/0`; the module's
    /// `G` is continuous through it.
    #[test]
    fn g_is_eq_10_for_the_junction_and_eq_9_for_the_inhibition() {
        let eq10 = |phi: f64, i: f64, gc: f64, beta: f64| {
            let t = (i / (i - 1.0)).ln();
            gc * 2.0 / t * (phi * ((1.0 - phi) * t).sinh() - (1.0 - phi) * (phi * t).sinh())
                + gc * beta / (i * t * t) * ((phi * t).exp() - ((1.0 - phi) * t).exp())
        };
        let eq9 = |phi: f64, i: f64, gs: f64, a: f64| {
            let t = (i / (i - 1.0)).ln();
            let big_a = a * a / (1.0 - (-a * t).exp());
            let b = t * (-a * t).exp() / (1.0 - (-a * t).exp());
            let first = ((-a * t).exp() * ((t + b) * (1.0 - a) - 1.0) - (b * (1.0 - a) - 1.0)) * ((phi * t).exp() - ((1.0 - phi) * t).exp());
            let second = (1.0 - (-t).exp())
                * ((phi * t).exp() * ((1.0 - a) * (1.0 - phi) * t).exp() * (((1.0 - phi) * t + b) * (1.0 - a) - 1.0)
                    - ((1.0 - phi) * t).exp() * ((1.0 - a) * phi * t).exp() * ((phi * t + b) * (1.0 - a) - 1.0));
            -gs * big_a / (i * t * t * (1.0 - a) * (1.0 - a)) * (first + second)
        };
        let mut worst: f64 = 0.0;
        for i in [1.05, 1.15, 1.3, 1.6, 2.5] {
            for k in 1..40 {
                let phi = f64::from(k) / 40.0;
                let e = LewisRinzel::new(i, 0.7, 0.15, 0.0, 1.0).unwrap();
                worst = worst.max((e.g(phi).unwrap() - eq10(phi, i, 0.7, 0.15)).abs());
                for a in [0.5, 2.0, 4.0, 6.0] {
                    let s = LewisRinzel::new(i, 0.0, 0.0, 0.3, a).unwrap();
                    worst = worst.max((s.g(phi).unwrap() - eq9(phi, i, 0.3, a)).abs());
                }
            }
        }
        assert!(worst < 3e-15, "{worst}");
        assert!(eq9(0.3, 1.2, 0.3, 1.0).is_nan(), "eq. (9) is 0/0 at α = 1");
        let at = |a: f64| LewisRinzel::new(1.2, 0.0, 0.0, 0.3, a).unwrap().g(0.3).unwrap();
        assert!((at(1.0) - 0.5 * (at(1.0 + 1e-4) + at(1.0 - 1e-4))).abs() < 3e-10, "continuous through α = 1");
        // G(0) = G(½) = 0 exactly, and G is odd.
        let m = LewisRinzel::new(1.3, 0.4, 0.2, 0.3, 2.0).unwrap();
        assert_eq!((m.g(0.0).unwrap(), m.g(0.5).unwrap()), (0.0, 0.0));
        assert!((m.g(0.3).unwrap() + m.g(-0.3).unwrap()).abs() < 1e-15 && (m.g(0.3).unwrap() + m.g(0.7).unwrap()).abs() < 1e-15);
    }

    /// `G′` from the closed forms is the derivative of `G`: central differences at interior phases,
    /// and one-sided ones at `0⁺` and `1⁻`, where `G` jumps but its slope does not. The residual
    /// is the difference quotient's own `O(h²)`: measured 3.5 × 10⁻⁹ at `h = 10⁻⁵`.
    #[test]
    fn g_slope_is_the_derivative_of_g() {
        let h = 1e-5;
        for m in [
            LewisRinzel::new(1.2, 0.3, 0.2, 0.7, 4.0).unwrap(),
            LewisRinzel::new(1.6, 1.0, 0.1, -0.4, 0.5).unwrap(),
            LewisRinzel::new(1.1, 0.0, 0.0, 1.0, 1.0).unwrap(),
        ] {
            for phi in [0.05, 0.2, 0.37, 0.5, 0.66, 0.9] {
                let fd = (m.g(phi + h).unwrap() - m.g(phi - h).unwrap()) / (2.0 * h);
                assert!((m.g_slope(phi).unwrap() - fd).abs() < 2e-8, "φ = {phi}: {} against {fd}", m.g_slope(phi).unwrap());
            }
            // The forward difference at `h` estimates `G′(h) = G′(0⁺) + O(h)`.
            let forward = (-3.0 * m.g(h).unwrap() + 4.0 * m.g(2.0 * h).unwrap() - m.g(3.0 * h).unwrap()) / (2.0 * h);
            assert!((m.g_slope(0.0).unwrap() - forward).abs() < 1e-3, "{} against {forward}", m.g_slope(0.0).unwrap());
            assert!((m.g_slope(1.0).unwrap() - m.g_slope(0.0).unwrap()).abs() < 1e-14);
        }
    }


    /// Eq. (11) is where antiphase changes stability under electrical coupling alone: its solution
    /// for each `β` is the drive at which the closed-form `G′(½)` changes sign, found independently
    /// by [`LewisRinzel::antiphase_critical_drives`] — agreement to 4.5 × 10⁻¹⁶ (measured).
    ///
    /// At `β = 0.1`, `I*_c = 1.494 153`; the text's "if I is increased above I = 1.5, only the
    /// synchronous state is stable" (p. 293) is that, rounded. At `β = 0.2` it is `1.259 221`,
    /// which the text gives as "I ∼ 1.26" (p. 297) and Fig. 11's caption as "`I*_c(β = 0.2)` =
    /// 1.28" (p. 298). ⚠ The caption is off by 0.02 against the paper's own eq. (11). The solve
    /// stays exact for tiny `β`, where `x = T/2 ≈ √(3β)` and `I ≈ 1/(2x) + ½`: at `β = 10⁻²⁰⁰`
    /// the drive is `2.886 751 × 10⁹⁹`.
    #[test]
    fn eq_11_is_where_antiphase_changes_stability() {
        for beta in [0.05, 0.1, 0.2, 0.3, 0.5, 2.0] {
            let closed = LewisRinzel::critical_drive_electrical(beta).unwrap();
            let pair = LewisRinzel { drive: 1.5, g_c: 1.0, beta, g_s: 0.0, alpha: 1.0 };
            let found = pair.antiphase_critical_drives(1.0001, 50.0, 4000).unwrap();
            assert_eq!(found.len(), 1, "β = {beta}: {found:?}");
            assert!((found[0] - closed).abs() < 1e-15 * closed.max(1.0) * 4.0, "β = {beta}: {} against {closed}", found[0]);
            let eq11 = (closed - 0.5) * (1.0 / (closed - 1.0)).ln_1p() - 1.0;
            assert!((eq11 - beta).abs() < 1e-15 * 4.0 * (1.0 + beta), "β = {beta}: eq. (11) gives {eq11}");
        }
        assert!((LewisRinzel::critical_drive_electrical(0.1).unwrap() - 1.494_153_235_807_301).abs() < 1e-14);
        // One grid interval is enough when the root lies in it: the last interval is scanned too.
        let pair = LewisRinzel { drive: 1.5, g_c: 1.0, beta: 0.1, g_s: 0.0, alpha: 1.0 };
        let one = pair.antiphase_critical_drives(1.2, 1.6, 1).unwrap();
        assert!(one.len() == 1 && (one[0] - 1.494_153_235_807_301).abs() < 1e-14, "{one:?}");
        let i02 = LewisRinzel::critical_drive_electrical(0.2).unwrap();
        assert!((i02 - 1.259_221_127_332_138).abs() < 1e-14);
        assert_eq!(format!("{i02:.2}"), "1.26", "the text's 1.26, not the caption's 1.28");
        for beta in [1e-6, 1e-12, 1e-200] {
            let i = LewisRinzel::critical_drive_electrical(beta).unwrap();
            let x = (3.0 * beta).sqrt();
            let asymptote = 1.0 / (2.0 * x) + 0.5;
            assert!((i / asymptote - 1.0).abs() < beta.max(1e-15) * 2.0, "β = {beta}: {i} against {asymptote}");
        }
        // β increases, I*_c decreases (Fig. 8): a stronger spike effect promotes synchrony.
        let ics: Vec<f64> = [0.02, 0.05, 0.1, 0.2, 0.4].iter().map(|&b| LewisRinzel::critical_drive_electrical(b).unwrap()).collect();
        assert!(ics.windows(2).all(|w| w[0] > w[1]), "{ics:?}");
    }

    /// Figs. 3 and 4: inhibition alone at `α = 4`, where antiphase loses stability at
    /// `I*_s = 1.484 231`, printed as "1.48" (Fig. 5's caption).
    ///
    /// At `I = 1.2` the unstable states the text puts at `φ* = 0.05` and 0.95, "only a 10% chance"
    /// of synchrony (p. 290), are at `φ* = 0.063 517` and 0.936 483 — a 12.7% chance. At `I = 1.4`
    /// they are at 0.239 800, a 48.0% chance: the text's "about a 50% chance". At `I = 1.6` only
    /// synchrony (stable) and antiphase (unstable) remain. The paper prints no `g_s` for Fig. 3; at
    /// `g_s = 1` the extremes of `G_s` are 0.200, 0.0500 and 0.0671, and the figure's axes are
    /// ±0.25, ±0.06 and ±0.08 with the curves reaching near the top of each: `g_s = 1`.
    #[test]
    fn figures_3_and_4_inhibition_alone_at_alpha_4() {
        let is = LewisRinzel::FIG3_TOP.antiphase_critical_drives(1.01, 3.0, 400).unwrap();
        assert_eq!(is.len(), 1);
        assert!((is[0] - 1.484_231_096_457_75).abs() < 1e-13, "{}", is[0]);
        let states = |m: LewisRinzel| m.locked_states(2000).unwrap();
        let top = states(LewisRinzel::FIG3_TOP);
        assert_eq!(top.iter().map(|s| s.stable).collect::<Vec<_>>(), vec![true, false, true, false]);
        assert!((top[1].phase - 0.063_516_960_431_7).abs() < 1e-12 && (top[3].phase - 0.936_483_039_568_3).abs() < 1e-12);
        assert!((LewisRinzel::FIG3_TOP.probability_of_synchrony(2000).unwrap() - 0.127_033_920_863).abs() < 1e-11);
        let middle = states(LewisRinzel::FIG3_MIDDLE);
        assert_eq!(middle.iter().map(|s| s.stable).collect::<Vec<_>>(), vec![true, false, true, false]);
        assert!((middle[1].phase - 0.239_800_263_587_5).abs() < 1e-12);
        assert!((LewisRinzel::FIG3_MIDDLE.probability_of_synchrony(2000).unwrap() - 0.479_600_527_175).abs() < 1e-11);
        assert_eq!(states(LewisRinzel::FIG3_BOTTOM), vec![LockedState { phase: 0.0, stable: true }, LockedState { phase: 0.5, stable: false }]);
        assert_eq!(LewisRinzel::FIG3_BOTTOM.probability_of_synchrony(2000).unwrap(), 1.0);
        for (m, want, axis) in [(LewisRinzel::FIG3_TOP, 0.2003, 0.25), (LewisRinzel::FIG3_MIDDLE, 0.04998, 0.06), (LewisRinzel::FIG3_BOTTOM, 0.06707, 0.08)] {
            let peak = (1..1000).map(|k| m.g(f64::from(k) / 1000.0).unwrap().abs()).fold(0.0, f64::max);
            assert!((peak - want).abs() < 2e-4 && peak < axis && peak > 0.6 * axis, "{peak}");
        }
    }

    /// Fig. 6 at `g_c = 1`: the spike's part of `G_c` jumps to `G(0⁺) = β(1 − eᵀ)/(IT²)` —
    /// −0.2055, −0.1397 and −0.1193 at `I` = 1.05, 1.15 and 1.3, against a panel whose axis is
    /// ±0.25 — and the subthreshold part alone (`β = 0`, the middle panel, axis ±0.45) peaks at
    /// 0.4046 at `I = 1.05`, makes synchrony unstable and antiphase stable at every drive. The
    /// bottom panel, `β = 0.1` at `I = 1.15`, has both stable and the separatrix at `φ* = 0.088 43`.
    ///
    /// For `0 < φ` below about 1.1 × 10⁻¹⁶, `−φ mod 1` rounds to exactly 1, where `H` stands for its
    /// limit from below, kick included: `G(φ)` there is still the jump `G(0⁺)`, to 2.8 × 10⁻¹⁷
    /// (measured), and within 2.1 × 10⁻¹² of `G(10⁻¹²)`; `H(−10⁻¹⁷)` is within 5.3 × 10⁻¹³ of
    /// `H(1 − 10⁻¹²)` (both measured).
    #[test]
    fn figure_6_the_two_parts_of_the_electrical_g() {
        for (i, jump) in [(1.05, -0.205_495_510_596_66), (1.15, -0.139_726_632_037_12), (1.3, -0.119_252_575_078_97)] {
            let spike = LewisRinzel::new(i, 1.0, 0.1, 0.0, 1.0).unwrap();
            let sub = LewisRinzel { beta: 0.0, ..spike };
            let near = spike.g(1e-12).unwrap() - sub.g(1e-12).unwrap();
            assert!((near - jump).abs() < 1e-11, "I = {i}: {near}");
            for d in [1.02, 1.05, 1.15, 1.3, 1.6, 2.0, 4.0] {
                let s = LewisRinzel { drive: d, ..sub }.locked_states(400).unwrap();
                assert_eq!(s, vec![LockedState { phase: 0.0, stable: false }, LockedState { phase: 0.5, stable: true }], "I = {d}");
            }
        }
        let peak = (1..1000).map(|k| LewisRinzel::new(1.05, 1.0, 0.0, 0.0, 1.0).unwrap().g(f64::from(k) / 1000.0).unwrap()).fold(0.0, f64::max);
        assert!((peak - 0.4046).abs() < 1e-4 && peak < 0.45, "{peak}");
        let bottom = LewisRinzel::FIG6_BOTTOM.locked_states(1000).unwrap();
        assert_eq!(bottom.iter().map(|s| s.stable).collect::<Vec<_>>(), vec![true, false, true, false]);
        assert!((bottom[1].phase - 0.088_427_572_761_1).abs() < 1e-12, "{}", bottom[1].phase);
        assert_eq!(LewisRinzel { beta: 0.0, ..LewisRinzel::FIG6_BOTTOM }.probability_of_synchrony(1000).unwrap(), 0.0);
        let m = LewisRinzel::FIG6_BOTTOM;
        let (i, t) = (m.drive, m.period().unwrap());
        assert_eq!((-1e-17f64).rem_euclid(1.0), 1.0, "the wrap rounds to 1");
        let jump = m.g_c * m.beta * (1.0 - t.exp()) / (i * t * t);
        let tiny = m.g(1e-17).unwrap();
        assert!((tiny - jump).abs() < 1e-16 && (tiny - m.g(1e-12).unwrap()).abs() < 1e-11, "G(10⁻¹⁷) = {tiny} against {jump}");
        assert!((m.h(-1e-17).unwrap() - m.h(1.0 - 1e-12).unwrap()).abs() < 2e-12, "H(1⁻) keeps the kick");
    }

    /// Fig. 10 and Table 1: the combined critical drive `I*_sc` runs monotonically from `I*_s` at
    /// `ρ = 0` to `I*_c` at `ρ = 1` (p. 295), and which end is higher decides whether adding
    /// electrical coupling promotes synchrony.
    ///
    /// Left panel, "large spikes, fast synapses" (`β = 0.3`, `α = 4`): `I*_s = 1.484 > I*_c = 1.165`,
    /// `I*_sc` falls with `ρ`, and at `I = 1.3` the probability of synchrony rises from 0.262 to 1.
    /// Right panel, "small spikes, slow synapses" (`β = 0.1`, `α = 1.5`): `I*_s = 1.097 < I*_c =
    /// 1.494`, `I*_sc` rises, and the probability falls from 1 to 0.357. Fig. 10's caption says
    /// exactly this ("Is* > Ic*" on the left). ⚠ Table 1 prints the inequalities the other way
    /// round — "Fast synapses, large spike effect (I*_c > I*_s)", "Slow synapses, small spike
    /// effect (I*_c < I*_s)" (p. 302) — which its own figure, its text on p. 295 ("When the spike
    /// effect is relatively large and inhibition is relatively fast, I*_c < I*_s") and the
    /// computation all contradict.
    #[test]
    fn figure_10_and_table_1_disagree_on_which_critical_drive_is_higher() {
        let isc = |beta: f64, alpha: f64, rho: f64| {
            let m = LewisRinzel::combined(1.3, rho, 1.0, beta, alpha).unwrap();
            let r = m.antiphase_critical_drives(1.001, 3.0, 600).unwrap();
            assert_eq!(r.len(), 1, "one critical drive for every ρ: {r:?}");
            r[0]
        };
        let rhos = [0.0, 0.25, 0.5, 0.75, 1.0];
        let left: Vec<f64> = rhos.iter().map(|&r| isc(0.3, 4.0, r)).collect();
        let right: Vec<f64> = rhos.iter().map(|&r| isc(0.1, 1.5, r)).collect();
        assert!((left[0] - 1.484_231_096_457_75).abs() < 1e-12 && (left[4] - 1.164_842_660_495_38).abs() < 1e-12, "{left:?}");
        assert!((right[0] - 1.097_145_396_597_11).abs() < 1e-12 && (right[4] - 1.494_153_235_807_30).abs() < 1e-12, "{right:?}");
        assert!(left.windows(2).all(|w| w[0] > w[1]) && right.windows(2).all(|w| w[0] < w[1]), "monotone in ρ");
        let p = |beta: f64, alpha: f64, rho: f64| LewisRinzel::combined(1.3, rho, 1.0, beta, alpha).unwrap().probability_of_synchrony(4000).unwrap();
        let pl: Vec<f64> = rhos.iter().map(|&r| p(0.3, 4.0, r)).collect();
        let pr: Vec<f64> = rhos.iter().map(|&r| p(0.1, 1.5, r)).collect();
        assert!(pl.windows(2).all(|w| w[0] <= w[1]) && pr.windows(2).all(|w| w[0] >= w[1]), "{pl:?} {pr:?}");
        assert!((pl[0] - 0.262_052_677_937).abs() < 1e-10 && (pr[4] - 0.356_717_313_494).abs() < 1e-10, "{pl:?} {pr:?}");
        // Fig. 9's statement: I*_sc lies between I*_c and I*_s — here at α = 5, β = 0.2, ρ = ½.
        let (c, sc, s) = (isc(0.2, 5.0, 1.0), isc(0.2, 5.0, 0.5), isc(0.2, 5.0, 0.0));
        assert!(c < sc && sc < s, "{c} {sc} {s}");
        assert!((s - 1.657_562_776_000_96).abs() < 1e-12 && (sc - 1.539_002_974_094_90).abs() < 1e-12, "{sc} {s}");
    }

    /// Fig. 11: with electrical coupling present, antiphase can lose stability as the synapse
    /// slows and then REGAIN it at slower synapses still — "This behavior cannot occur for
    /// inhibitory coupling alone" (p. 298).
    ///
    /// At `I = 1.2`, `β = 0.2`, `G′(½)` changes sign twice as `α` runs down from 4 for every `ρ`
    /// below the dome's top, and not at all above it; the top is at `ρ = 0.654` (the figure draws
    /// it at about 0.65), and at `ρ = 0` the one change is at `α = 2.245`, where Fig. 5's curve puts
    /// `I*_s = 1.2`. At `I = 1.25` the dome's top is at `ρ = 0.938` (drawn: about 0.93). At `I = 1.3`,
    /// above `I*_c(0.2)`, there is one change for every `ρ` — at `α = 2.891` for `ρ = 0`.
    #[test]
    fn figure_11_antiphase_can_lose_and_regain_stability_as_alpha_falls() {
        let switches = |i: f64, rho: f64| {
            let f = |a: f64| LewisRinzel::combined(i, rho, 1.0, 0.2, a).unwrap().g_slope(0.5).unwrap();
            let mut out = Vec::new();
            let mut prev = (0.01, f(0.01));
            for k in 1..=2000 {
                let a = 0.01 + (4.0 - 0.01) * f64::from(k) / 2000.0;
                let now = (a, f(a));
                if (prev.1 < 0.0) != (now.1 < 0.0) {
                    out.push(bisect(|x| (f(x) < 0.0) == (prev.1 < 0.0), prev.0, now.0));
                }
                prev = now;
            }
            out
        };
        for rho in [0.1, 0.3, 0.5, 0.6, 0.65] {
            assert_eq!(switches(1.2, rho).len(), 2, "ρ = {rho}: {:?}", switches(1.2, rho));
        }
        assert!(switches(1.2, 0.66).is_empty() && switches(1.2, 0.9).is_empty());
        let foot = switches(1.2, 0.0);
        assert!(foot.len() == 1 && (foot[0] - 2.2454).abs() < 1e-3, "{foot:?}");
        let back = LewisRinzel::combined(1.3, 0.0, 1.0, 0.2, foot[0]).unwrap();
        assert!((LewisRinzel { drive: 1.2, ..back }.g_slope(0.5).unwrap()).abs() < 1e-12, "the same point as Fig. 5's curve");
        assert_eq!(switches(1.25, 0.93).len(), 2);
        assert!(switches(1.25, 0.94).is_empty());
        for rho in [0.0, 0.3, 0.5, 0.8] {
            assert_eq!(switches(1.3, rho).len(), 1, "ρ = {rho}");
        }
        assert!((switches(1.3, 0.0)[0] - 2.8908).abs() < 1e-3);
    }

    /// The structural claims of pp. 292–295: the coupling strengths only scale `G`, so they move
    /// no locked state; the combined `G` is the sum of the two (eq. 12); and "the zeros of `G(φ)`
    /// for excitatory synaptic connections (`g_s < 0`) are the same ... however the slopes at these
    /// zeros are opposite" — every stability flips.
    #[test]
    fn coupling_strength_scales_g_and_excitation_flips_every_stability() {
        let base = LewisRinzel::new(1.3, 0.0, 0.0, 0.5, 3.0).unwrap();
        let strong = LewisRinzel { g_s: 1.75, ..base };
        let excite = LewisRinzel { g_s: -0.5, ..base };
        for k in 1..20 {
            let phi = f64::from(k) / 20.0;
            assert!((strong.g(phi).unwrap() - 3.5 * base.g(phi).unwrap()).abs() < 1e-14);
            assert!((excite.g(phi).unwrap() + base.g(phi).unwrap()).abs() < 1e-15);
        }
        let (a, b) = (base.locked_states(1000).unwrap(), excite.locked_states(1000).unwrap());
        assert_eq!(a.len(), b.len());
        assert!(a.len() > 2, "an interior pair to flip: {a:?}");
        for (x, y) in a.iter().zip(&b) {
            assert!((x.phase - y.phase).abs() < 1e-14 && x.stable != y.stable, "{x:?} {y:?}");
        }
        for (x, y) in a.iter().zip(&strong.locked_states(1000).unwrap()) {
            assert!((x.phase - y.phase).abs() < 1e-14 && x.stable == y.stable, "{x:?} {y:?}");
        }
        let both = LewisRinzel::new(1.3, 0.4, 0.2, 0.5, 3.0).unwrap();
        let elec = LewisRinzel { g_s: 0.0, ..both };
        for k in 1..20 {
            let phi = f64::from(k) / 20.0;
            assert!((both.g(phi).unwrap() - elec.g(phi).unwrap() - base.g(phi).unwrap()).abs() < 1e-15, "eq. (12)");
        }
        // "the synchronous state φ* = 0, 1 is always stable for any combination" (p. 295) —
        // because the kick's jump attracts whatever the slope does.
        for (i, a) in [(1.05, 0.5), (1.3, 6.0), (2.5, 1.0)] {
            for rho in [0.1, 0.5, 1.0] {
                assert!(LewisRinzel::combined(i, rho, 1.0, 0.05, a).unwrap().locked_states(200).unwrap()[0].stable);
            }
        }
    }

    /// Every locked state [`LewisRinzel::locked_states`] reports is a zero of `G` whose stability
    /// is the sign of `G′` there, over a spread of parameters; and the mirror pairs are mirrors.
    #[test]
    fn every_locked_state_is_a_zero_with_the_slope_its_stability_says() {
        let mut interior = 0;
        for i in [1.02, 1.1, 1.25, 1.5, 2.0] {
            for (gc, beta, gs, a) in [(0.5, 0.1, 0.5, 1.0), (1.0, 0.05, 0.0, 1.0), (0.2, 0.3, 0.8, 5.0), (0.0, 0.0, 1.0, 0.7)] {
                let m = LewisRinzel::new(i, gc, beta, gs, a).unwrap();
                let s = m.locked_states(1000).unwrap();
                let n = s.len();
                assert_eq!(s[n / 2].phase, 0.5);
                for k in 1..n / 2 {
                    interior += 1;
                    assert!(m.g(s[k].phase).unwrap().abs() < 1e-14, "{:?}", s[k]);
                    assert_eq!(s[k].stable, m.g_slope(s[k].phase).unwrap() < 0.0);
                    assert!(s[n - k].phase + s[k].phase == 1.0 && s[n - k].stable == s[k].stable);
                }
                assert_eq!(s[n / 2].stable, m.g_slope(0.5).unwrap() < 0.0);
            }
        }
        assert!(interior >= 8, "{interior}");
    }

    /// Figs. 1 and 2 at their printed parameters and initial conditions (`v₁(0) = 0.4` and 0.59,
    /// `v₂(0) = 0`): the first eight spikes against `SciPy`'s DOP853 integration of eq. (5) at
    /// `rtol = 10⁻¹³` (`tools/gapjunction_reference.py`, A), agreeing to the twelve decimals it prints
    /// (worst difference 5.0 × 10⁻¹³, the rounding of the printed value), and
    /// the pattern each panel shows — antiphase at `I = 1.1`, synchrony at `I = 1.6`.
    #[test]
    fn figures_1_and_2_run_as_printed() {
        let scipy: [(LewisRinzel, [f64; 2], [f64; 8]); 4] = [
            (LewisRinzel::FIG1_LOW, [0.4, 0.0], [1.945910149055, 3.631531863923, 5.416534997091, 7.166665253953, 8.929453089036, 10.687587952513, 12.447428901689, 14.206643601457]),
            (LewisRinzel::FIG1_HIGH, [0.4, 0.0], [core::f64::consts::LN_2, 1.077150847397, 1.846057493550, 2.287906713711, 3.055967382571, 3.494544457268, 4.269653296142, 4.698235700557]),
            (LewisRinzel::FIG2_LOW, [0.59, 0.0], [1.897467817916, 2.853087527184, 4.478339345610, 5.634059861104, 7.116593302189, 8.371711998066, 9.784584375918, 11.087930693909]),
            (LewisRinzel::FIG2_HIGH, [0.59, 0.0], [0.579445496350, 0.958931492435, 1.545006503035, 1.917424290424, 2.510621739438, 2.875425611459, 3.476284937878, 3.832874254418]),
        ];
        for (m, v0, want) in scipy {
            let mut pair = LewisRinzelPair::new(m, v0).unwrap();
            let spikes = pair.run(300.0, 32).unwrap();
            for (k, (s, w)) in spikes.iter().zip(want).enumerate() {
                assert_eq!(s.cell, k % 2, "they alternate from cell 0");
                assert!((s.time - w).abs() < 1e-12, "I = {}: spike {k} at {} against {w}", m.drive, s.time);
            }
            let (_, _, lag) = *phases(&spikes).last().unwrap();
            if m.drive < 1.5 {
                assert!((lag - 0.5).abs() < 1e-9, "I = {}: antiphase, lag {lag}", m.drive);
            } else {
                assert!(lag < 1e-12 || lag > 1.0 - 1e-12, "I = {}: synchrony, lag {lag}", m.drive);
            }
        }
    }

    /// Section 4, beyond weak coupling: [`LewisRinzel::antiphase_orbit`] is the orbit the full
    /// simulation settles on — period and pre-kick voltage — and `SciPy`'s quadrature of the same
    /// matching conditions (reference C) agrees to 10⁻¹².
    ///
    /// ⚠ The four matching equations as printed (p. 299) carry `+g_s` on every synaptic bracket,
    /// where eq. (5) makes inhibition `−g_s`; at Fig. 1 (left) they give `T = 1.631 404` against
    /// the simulated 3.518 766. Their kick terms are right only if `u` is the voltage AFTER the
    /// kick, while the text defines it as before ("immediately before spike effects are added",
    /// p. 299): at Fig. 2 (left) they return the right period with `u` larger by exactly `g_c β =
    /// 0.04`. And their third equation's bracket subtracts where the first adds, so at `φ = ½`
    /// with inhibition the two cannot hold together for `u₁ = u₂`.
    #[test]
    fn section_4_antiphase_orbit_is_the_simulated_one_and_the_printed_system_is_not() {
        for (m, v0, t_ref, u_ref) in [
            (LewisRinzel::FIG1_LOW, [0.4, 0.0], 3.518_765_594_089, 0.908_941_879_914),
            (LewisRinzel::FIG2_LOW, [0.59, 0.0], 2.696_337_925_394, 0.863_201_489_830),
        ] {
            let orbit = m.antiphase_orbit(8.0, 400).unwrap().unwrap();
            assert!((orbit.period - t_ref).abs() < 1e-12 && (orbit.u - u_ref).abs() < 1e-12, "{orbit:?}");
            let mut pair = LewisRinzelPair::new(m, v0).unwrap();
            let spikes = pair.run(400.0, 32).unwrap();
            let (last, period, _) = *phases(&spikes).last().unwrap();
            assert!((period - orbit.period).abs() < 1e-11, "{period} against {}", orbit.period);
            // Stop just short of cell 0's next spike and read cell 1, which fired half a period ago.
            let mut probe = LewisRinzelPair::new(m, v0).unwrap();
            probe.run(last + orbit.period - 1e-9, 32).unwrap();
            assert!((probe.voltages()[1] - orbit.u).abs() < 1e-9, "{} against {}", probe.voltages()[1], orbit.u);
        }
        let printed = |m: LewisRinzel| {
            let (i, a, k, mu) = (m.drive, m.alpha, m.g_c * m.beta, 1.0 + 2.0 * m.g_c);
            let at = |t: f64| {
                let h = 0.5 * t;
                let sp = leaky_s(a, t, 1.0, h, h) + leaky_s(a, t, 1.0, 0.0, h);
                let sm = leaky_s(a, t, mu, h, h) - leaky_s(a, t, mu, 0.0, h);
                let u = (1.0 + k + m.g_s * sm) / (1.0 + (-mu * h).exp());
                (u - k + 1.0 - u * (-h).exp() - 2.0 * i * (1.0 - (-h).exp()) - m.g_s * sp, u)
            };
            let neg = at(0.3).0 < 0.0;
            let t = bisect(|t| (at(t).0 < 0.0) == neg, 0.3, 8.0);
            (t, at(t).1)
        };
        let (t1, _) = printed(LewisRinzel::FIG1_LOW);
        assert!((t1 - 1.631_403_909_057).abs() < 1e-11, "{t1}");
        let (t2, u2) = printed(LewisRinzel::FIG2_LOW);
        assert!((t2 - 2.696_337_925_394).abs() < 1e-11 && (u2 - (0.863_201_489_830 + 0.04)).abs() < 1e-11, "{t2} {u2}");
        let (a, t) = (3.0, 3.5);
        let (eq1, eq3) = (leaky_s(a, t, 1.0, 1.75, 1.75) + leaky_s(a, t, 1.0, 0.0, 1.75), leaky_s(a, t, 1.0, 0.0, 1.75) - leaky_s(a, t, 1.0, 1.75, 1.75));
        assert!((eq1 - eq3).abs() > 0.01, "the printed brackets of eqs. 1 and 3 differ at φ = ½: {eq1} {eq3}");
    }

    /// In synchrony no current crosses the junction: with electrical coupling alone the simulated
    /// period is the intrinsic `ln(I/(I − 1))` to 10⁻¹⁴ (Fig. 2, right, captured into exact
    /// synchrony by the kick), which also confirms the convention that a captured cell's kick is
    /// not returned. Inhibition lengthens it (p. 301): at Fig. 1 (right) the simulation settles on
    /// [`LewisRinzel::synchronous_period`], 1.176 394, which `SciPy`'s quadrature gives to 10⁻¹².
    #[test]
    fn the_synchronous_period_ignores_the_junction() {
        let mut pair = LewisRinzelPair::new(LewisRinzel::FIG2_HIGH, [0.59, 0.0]).unwrap();
        let spikes = pair.run(40.0, 32).unwrap();
        let (_, period, lag) = *phases(&spikes).last().unwrap();
        let t0 = LewisRinzel::FIG2_HIGH.period().unwrap();
        assert!((period - t0).abs() < 1e-13 && lag == 0.0, "{period} {lag}");
        assert_eq!(LewisRinzel::FIG2_HIGH.synchronous_period(3.0, 300).unwrap(), Some(bisect(|t| -1.6 * (-t).exp_m1() < 1.0, 0.01, 3.0)));
        let sync = LewisRinzel::FIG1_HIGH.synchronous_period(3.0, 300).unwrap().unwrap();
        assert!((sync - 1.176_393_888_224).abs() < 1e-12 && sync > LewisRinzel::FIG1_HIGH.period().unwrap(), "{sync}");
        let mut pair = LewisRinzelPair::new(LewisRinzel::FIG1_HIGH, [0.4, 0.0]).unwrap();
        let spikes = pair.run(300.0, 32).unwrap();
        let (_, period, _) = *phases(&spikes).last().unwrap();
        assert!((period - sync).abs() < 1e-11, "{period} against {sync}");
    }

    /// Spike capture with inhibition live: at `I = 1.6` with both couplings (`g_c = β = g_s = 0.2`,
    /// `α = 3`) the kick captures the partner from `t = 24.09` on, both cells then firing at one instant
    /// and each receiving the other's inhibition — so captured synchrony runs at the inhibition-only
    /// synchronous period, 1.176 394, whatever the junction, once the synaptic history from before
    /// capture has decayed as `e^{−αt}` (by `t = 45`, to 10⁻¹²). The spikes after capture match
    /// `SciPy`'s (reference A) to the twelve decimals it prints, cell order included.
    #[test]
    fn spike_capture_with_inhibition_runs_at_the_synchronous_period() {
        let m = LewisRinzel::new(1.6, 0.2, 0.2, 0.2, 3.0).unwrap();
        let spikes = LewisRinzelPair::new(m, [0.59, 0.0]).unwrap().run(45.0, 32).unwrap();
        let late: Vec<&Spike> = spikes.iter().filter(|s| s.time > 24.0).collect();
        let scipy = [
            (24.092900084636, 1),
            (24.092900084636, 0),
            (25.269860738724, 0),
            (25.269860738724, 1),
            (26.446259928423, 0),
            (26.446259928423, 1),
            (27.622653413188, 0),
            (27.622653413188, 1),
        ];
        for (s, (t, c)) in late.iter().zip(scipy) {
            assert!(s.cell == c && (s.time - t).abs() < 1e-12, "{s:?} against ({t}, {c})");
        }
        let n = late.len();
        assert!(late[n - 1].time == late[n - 2].time, "captured: the last two spikes share an instant");
        let sync = m.synchronous_period(3.0, 300).unwrap().unwrap();
        assert_eq!(sync, LewisRinzel::FIG1_HIGH.synchronous_period(3.0, 300).unwrap().unwrap(), "the junction drops out");
        assert!((late[n - 1].time - late[n - 3].time - sync).abs() < 1e-12, "{}", late[n - 1].time - late[n - 3].time);
    }

    /// The pair is exact between spikes: a classical Runge–Kutta integration of eq. (5) across one
    /// quiet stretch, with the synaptic currents live, converges on [`LewisRinzelPair`]'s closed
    /// form at fourth order — the error ratios on halving the step are 16.4 and 16.2 (measured),
    /// approaching 16. And a spike is placed at the first float at which the closed-form voltage
    /// reaches threshold.
    #[test]
    fn the_pair_is_exact_between_spikes() {
        let m = LewisRinzel::new(1.3, 0.35, 0.1, 0.4, 2.5).unwrap();
        let mut pair = LewisRinzelPair::new(m, [0.2, 0.6]).unwrap();
        pair.run(5.0, 16).unwrap();
        assert!(pair.syn.iter().all(|s| s[1] != 0.0), "both synaptic currents live: {:?}", pair.syn);
        let span = 0.25;
        let exact = pair.after(span);
        assert!(exact.iter().all(|&v| v < 1.0), "no spike in the stretch");
        let field = |t: f64, v: [f64; 2]| {
            let c = |j: usize| (pair.syn[j][0] + pair.syn[j][1] * t) * (-m.alpha * t).exp();
            [
                -v[0] + m.drive - m.g_s * c(0) + m.g_c * (v[1] - v[0]),
                -v[1] + m.drive - m.g_s * c(1) + m.g_c * (v[0] - v[1]),
            ]
        };
        let rk4 = |n: usize| {
            let h = span / n as f64;
            let mut v = pair.v;
            for k in 0..n {
                let t = h * k as f64;
                let k1 = field(t, v);
                let k2 = field(t + h / 2.0, [v[0] + h / 2.0 * k1[0], v[1] + h / 2.0 * k1[1]]);
                let k3 = field(t + h / 2.0, [v[0] + h / 2.0 * k2[0], v[1] + h / 2.0 * k2[1]]);
                let k4 = field(t + h, [v[0] + h * k3[0], v[1] + h * k3[1]]);
                v = [v[0] + h / 6.0 * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]), v[1] + h / 6.0 * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1])];
            }
            (v[0] - exact[0]).abs().max((v[1] - exact[1]).abs())
        };
        let (e1, e2, e3) = (rk4(8), rk4(16), rk4(32));
        assert!((e1 / e2 - 16.0).abs() < 0.6 && (e2 / e3 - 16.0).abs() < 0.6, "{e1} {e2} {e3}");
        // The first crossing, sampled once per period so it lies in the first step.
        let alone = LewisRinzel::new(1.3, 0.0, 0.0, 0.0, 1.0).unwrap();
        let fresh = LewisRinzelPair::new(alone, [0.0, 0.5]).unwrap();
        let mut run = fresh.clone();
        let spike = run.run(2.0, 1).unwrap()[0];
        assert_eq!(spike.cell, 1);
        assert!(fresh.after(spike.time)[1] >= 1.0 && fresh.after(f64::from_bits(spike.time.to_bits() - 1))[1] < 1.0);
        assert!((spike.time - (0.8f64 / 0.3).ln()).abs() < 1e-15, "ln((I − v₀)/(I − 1))");
    }

    /// Sampling more finely moves no spike: the crossings are bisected to the last bit whatever
    /// the step, so 8 and 64 samples per period give the same spikes to 10⁻¹³ over forty
    /// periods (measured 2.8 × 10⁻¹⁴).
    #[test]
    fn sampling_more_finely_moves_no_spike() {
        let m = LewisRinzel::new(1.2, 0.3, 0.2, 0.3, 3.0).unwrap();
        let coarse = LewisRinzelPair::new(m, [0.3, 0.0]).unwrap().run(80.0, 8).unwrap();
        let fine = LewisRinzelPair::new(m, [0.3, 0.0]).unwrap().run(80.0, 64).unwrap();
        assert_eq!(coarse.len(), fine.len());
        assert!(coarse.len() > 40);
        for (a, b) in coarse.iter().zip(&fine) {
            assert!(a.cell == b.cell && (a.time - b.time).abs() < 1e-13, "{a:?} {b:?}");
        }
    }

    /// Weak coupling, simulated: two cells coupled at strength `ε` drift in phase difference as
    /// `dφ/dt = G(φ)` says. Each run starts on one side of the separatrix and ends where `G` sends
    /// it; near a stable state the distance decays at the rate `G′(φ*)` (the slope of `ln|φ − φ*|`
    /// against time), and below the separatrix the whole approach to synchrony follows the phase
    /// model — whose jump `G(0⁺) < 0` brings it there in finite time, where the kick captures it.
    ///
    /// The phase model is the `ε → 0` limit, and the simulation approaches it at first order: the
    /// relative error of the antiphase rate is 1.72% at `ε = 0.004` and 0.82% at 0.002 (electrical
    /// coupling), 1.77% and 0.84% (inhibition), halving with `ε`. The synchrony rate under inhibition
    /// is read where `|φ| < 10⁻⁴` (measured within 0.22%), because near 0 `G` is far from linear:
    /// `G(0.02)` is 0.603 of `G′(0) × 0.02` at `α = 4`, the alpha current's slope jumping at the spike,
    /// and a fit reaching out to 0.02 reads the rate 7% low at every `ε`. The electrical approach to
    /// synchrony tracks the integrated phase model to 1.8 × 10⁻³ in `φ` (measured).
    #[test]
    fn weak_coupling_simulation_follows_the_phase_model() {
        let start = |m: LewisRinzel, phi0: f64, duration: f64| {
            let t = m.period().unwrap();
            let v1 = m.limit_cycle((1.0 - phi0) * t).unwrap();
            phases(&LewisRinzelPair::new(m, [0.0, v1]).unwrap().run(duration, 16).unwrap())
        };
        let rate = |trace: &[(f64, f64, f64)], target: f64, lo: f64, hi: f64| {
            let pts: Vec<(f64, f64)> =
                trace.iter().filter(|p| (p.2 - target).abs() < hi && (p.2 - target).abs() > lo).map(|p| (p.0, (p.2 - target).abs().ln())).collect();
            let n = pts.len() as f64;
            let (mx, my) = (pts.iter().map(|p| p.0).sum::<f64>() / n, pts.iter().map(|p| p.1).sum::<f64>() / n);
            pts.iter().map(|p| (p.0 - mx) * (p.1 - my)).sum::<f64>() / pts.iter().map(|p| (p.0 - mx) * (p.0 - mx)).sum::<f64>()
        };
        // Electrical coupling alone, β = 0.1, I = 1.15: separatrix at 0.0884.
        let anti = |eps: f64| {
            let m = LewisRinzel { g_c: eps, ..LewisRinzel::FIG6_BOTTOM };
            rate(&start(m, 0.45, 8.0 / eps), 0.5, 1e-3, 0.04) / m.g_slope(0.5).unwrap() - 1.0
        };
        let (e4, e2) = (anti(0.004), anti(0.002));
        assert!(e2.abs() < 0.01 && (e4 / e2 - 2.0).abs() < 0.3, "{e4} {e2}");
        // Below the separatrix, the whole approach to synchrony: the phase model integrated from the
        // first measured phase tracks the simulation until the kick captures it, which the jump
        // G(0⁺) < 0 lets happen in finite time.
        let elec = LewisRinzel { g_c: 0.004, ..LewisRinzel::FIG6_BOTTOM };
        let to_sync = start(elec, 0.06, 400.0);
        let (mut t, mut phi) = (to_sync[0].0, to_sync[0].2);
        let g = |p: f64| elec.g(p).unwrap();
        let mut worst: f64 = 0.0;
        for &(tn, _, pn) in &to_sync[1..] {
            if pn < 0.005 {
                break;
            }
            while t < tn {
                let h = (tn - t).min(0.25);
                let k1 = g(phi);
                let k2 = g(phi + h / 2.0 * k1);
                let k3 = g(phi + h / 2.0 * k2);
                let k4 = g(phi + h * k3);
                phi += h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
                t += h;
            }
            worst = worst.max((phi - pn).abs());
        }
        assert!(worst < 2e-3, "{worst}");
        assert!(to_sync.last().unwrap().2 == 0.0, "captured into exact synchrony");
        // Inhibition alone, α = 4, I = 1.2: separatrix at 0.0635.
        let inh = |eps: f64| LewisRinzel { g_s: eps, ..LewisRinzel::FIG3_TOP };
        let anti = |eps: f64| rate(&start(inh(eps), 0.45, 8.0 / eps), 0.5, 1e-3, 0.04) / inh(eps).g_slope(0.5).unwrap() - 1.0;
        let sync = |eps: f64| {
            let back = start(inh(eps), 0.04, 8.0 / eps);
            let tail: Vec<(f64, f64, f64)> = back.iter().map(|p| (p.0, p.1, if p.2 > 0.5 { p.2 - 1.0 } else { p.2 })).collect();
            rate(&tail, 0.0, 1e-6, 1e-4) / inh(eps).g_slope(0.0).unwrap() - 1.0
        };
        let (a4, a2, s4, s2) = (anti(0.004), anti(0.002), sync(0.004), sync(0.002));
        assert!(a2.abs() < 0.01 && (a4 / a2 - 2.0).abs() < 0.3, "inhibitory antiphase: {a4} {a2}");
        assert!(s4.abs() < 0.005 && s2.abs() < 0.005, "inhibitory synchrony: {s4} {s2}");
        // Why the synchrony rate is read so close to 0: there G is far from linear.
        let near = inh(0.002);
        let bend = near.g(0.02).unwrap() / (near.g_slope(0.0).unwrap() * 0.02);
        assert!((bend - 0.6033).abs() < 1e-3, "{bend}");
    }

    /// The event budget: a run that needs more events than it allows stops with an error naming
    /// how far it got, and the budget [`LewisRinzelPair::run`] grants carries a pair through fast
    /// excitatory firing sampled only once per intrinsic period.
    #[test]
    fn a_run_that_outgrows_its_budget_says_so() {
        let mut pair = LewisRinzelPair::new(LewisRinzel::FIG2_LOW, [0.59, 0.0]).unwrap();
        let err = pair.run_within(40.0, 8, 3).unwrap_err();
        assert!(matches!(err, GapError::Stalled { .. }));
        assert_eq!(err.to_string(), format!("the run used up its event budget at t = {}; sample more finely", pair.time()));
        let fast = LewisRinzel::new(1.1, 0.0, 0.0, -0.5, 3.0).unwrap();
        let mut pair = LewisRinzelPair::new(fast, [0.5, 0.0]).unwrap();
        let spikes = pair.run(100.0, 1).unwrap();
        let steps = (100.0 / fast.period().unwrap()).ceil();
        assert!(spikes.len() as f64 > steps, "more spikes ({}) than sampling steps ({steps})", spikes.len());
        assert!((pair.time() - 100.0).abs() < 1e-12);
    }

    /// The lower unstable branch `φ*(I)` of a bifurcation diagram: the smallest interior locked
    /// state at each drive.
    fn branch(m: LewisRinzel, i: f64) -> f64 {
        LewisRinzel { drive: i, ..m }.locked_states(2000).unwrap()[1].phase
    }

    /// The bifurcation diagrams, read from the PDF (`tools/gapjunction_reference.py`, E: 144 dpi
    /// rasters, one pixel ≈ 0.002 in `I` and 0.003 in `φ`), against the module at the printed
    /// parameters. Fig. 4 (`α = 4`): its arrow at `I = 1.4843` and seven points of its dashed branch
    /// agree to 0.005. ⚠ Fig. 7, captioned "`β = 0.1`": its arrow sits at `I = 1.2588` — eq. (11)'s
    /// `I*_c` for `β = 0.2`, where `β = 0.1` gives 1.4942 — and its branch runs through `β = 0.2`'s
    /// unstable states to 0.008 while `β = 0.1`'s are at half the phase (0.197 drawn at `I = 1.149`,
    /// 0.196 for `β = 0.2`, 0.088 for 0.1). Fig. 7 was drawn at `β = 0.2`; Fig. 6's bottom panel,
    /// `β = 0.1` at `I = 1.15`, has its separatrix at 0.088, consistent with the caption and not the
    /// drawing. ⚠ Fig. 9 (`α = 5`, `β = 0.2`, `ρ = 0.5`): the light `ρ = 1` and `ρ = 0` arrows agree
    /// (1.2587 against 1.2592; 1.6618 against 1.6576), but the black `ρ = 0.5` branch leaves the
    /// computed one as it nears its fork — 0.130 against 0.142 at `I = 1.29`, 0.295 against 0.345 at
    /// `I = 1.5` — and its arrow reads `I*_sc = 1.5746` against 1.5390. No single `ρ`, `α` or `β`
    /// near the caption's reproduces both the branch and the arrow; the discrepancy is recorded,
    /// not explained.
    #[test]
    fn the_bifurcation_diagrams_against_their_drawings() {
        let fig4 = LewisRinzel::FIG3_TOP;
        assert!((fig4.antiphase_critical_drives(1.01, 3.0, 400).unwrap()[0] - 1.4843).abs() < 0.001);
        for (i, drawn) in [(1.0961, 0.021), (1.1490, 0.040), (1.2490, 0.094), (1.2980, 0.129), (1.3451, 0.171), (1.4000, 0.241), (1.4490, 0.330)] {
            assert!((branch(fig4, i) - drawn).abs() < 0.005, "Fig. 4 at I = {i}: {} against {drawn}", branch(fig4, i));
        }
        let (b01, b02) = (LewisRinzel::critical_drive_electrical(0.1).unwrap(), LewisRinzel::critical_drive_electrical(0.2).unwrap());
        assert!((b02 - 1.2588).abs() < 0.002 && (b01 - 1.2588).abs() > 0.2, "the arrow is β = 0.2's");
        let (c01, c02) = (LewisRinzel { beta: 0.1, ..LewisRinzel::FIG6_BOTTOM }, LewisRinzel { beta: 0.2, ..LewisRinzel::FIG6_BOTTOM });
        for (i, drawn) in [(1.0196, 0.060), (1.0490, 0.088), (1.0941, 0.133), (1.1490, 0.197), (1.1961, 0.264), (1.2255, 0.329), (1.2490, 0.402)] {
            assert!((branch(c02, i) - drawn).abs() < 0.008, "Fig. 7 at I = {i}: β = 0.2 gives {}", branch(c02, i));
            assert!(drawn - branch(c01, i) > 0.02, "and β = 0.1 does not: {}", branch(c01, i));
        }
        let fig9 = |rho: f64| LewisRinzel::combined(1.3, rho, 1.0, 0.2, 5.0).unwrap();
        assert!((fig9(1.0).antiphase_critical_drives(1.01, 3.0, 400).unwrap()[0] - 1.2587).abs() < 0.001);
        assert!((fig9(0.0).antiphase_critical_drives(1.01, 3.0, 400).unwrap()[0] - 1.6618).abs() < 0.005);
        let sc = fig9(0.5).antiphase_critical_drives(1.01, 3.0, 400).unwrap()[0];
        assert!((sc - 1.5746).abs() > 0.03, "the drawn I*_sc is not ρ = 0.5's: {sc}");
        for (i, drawn, err) in [(1.0504, 0.040, 0.004), (1.1008, 0.055, 0.004), (1.1531, 0.072, 0.004), (1.2054, 0.091, 0.006), (1.2946, 0.130, 0.015), (1.3953, 0.192, 0.025), (1.5, 0.295, 0.055)] {
            assert!((branch(fig9(0.5), i) - drawn).abs() < err, "Fig. 9 at I = {i}: {}", branch(fig9(0.5), i));
        }
    }

    /// The two-parameter diagrams, read from the PDF (reference E; one pixel ≈ 0.0028 in `I`, 0.026
    /// in `α` for Fig. 5, 0.0035 in `I`, 0.0022 in `β` for Fig. 8, and 0.056 in `α`, 0.0068 in `ρ`
    /// for Fig. 11). Fig. 5's dashed curve is `I*_s(α)` to 0.0063 in `I` (measured). Every point read
    /// from Fig. 8 lies within 0.42 pixels (measured) of the module's `I*_c(β)`, eq. (11) solved by
    /// [`LewisRinzel::critical_drive_electrical`]; measured along `I` alone the point at the curve's
    /// flat end, `I = 1.81`, is 0.011 away, three pixels, for 0.0009 in `β`. And every point read
    /// from Fig. 11's three panels lies within 2.2 pixels (measured) of the zero set of the module's
    /// `G′(½)` in `(α, ρ)` at `β = 0.2`.
    #[test]
    fn the_two_parameter_diagrams_against_their_drawings() {
        for (i, alpha) in [(1.3028, 2.947), (1.4, 3.502), (1.4972, 4.057), (1.7, 5.247), (1.75, 5.524)] {
            let at = LewisRinzel::new(1.3, 0.0, 0.0, 1.0, alpha).unwrap().antiphase_critical_drives(1.01, 3.0, 400).unwrap()[0];
            assert!((at - i).abs() < 0.008, "Fig. 5: α = {alpha} gives I*_s = {at}, drawn at {i}");
        }
        // Fig. 8's frame: 289 pixels per unit of I, 458 per unit of β.
        let fig8: Vec<(f64, f64)> = (0..=4000)
            .map(|k| {
                let beta = 0.03 + 0.2 * f64::from(k) / 4000.0;
                (LewisRinzel::critical_drive_electrical(beta).unwrap(), beta)
            })
            .collect();
        for (i, beta) in [(1.3010, 0.1725), (1.4014, 0.1266), (1.5017, 0.0983), (1.5882, 0.0808), (1.8097, 0.0524)] {
            let px = fig8.iter().map(|&(ci, cb)| (((ci - i) * 289.0).powi(2) + ((cb - beta) * 458.0).powi(2)).sqrt()).fold(f64::MAX, f64::min);
            assert!(px < 1.0, "Fig. 8: ({i}, {beta}) is {px} pixels from I*_c(β)");
        }
        let slope = |i: f64, rho: f64, a: f64| LewisRinzel::combined(i, rho, 1.0, 0.2, a).unwrap().g_slope(0.5).unwrap();
        for (i, pts) in [
            (1.2, vec![(0.502, 0.367), (1.045, 0.616), (1.394, 0.653), (1.798, 0.595), (2.077, 0.388)]),
            (1.25, vec![(0.488, 0.748), (1.017, 0.912), (1.519, 0.939), (2.021, 0.922), (2.411, 0.789)]),
            (1.3, vec![(3.220, 0.738), (3.498, 0.840), (3.749, 0.884)]),
        ] {
            // The zero set of G′(½), traced along both axes so its steep flanks are sampled too.
            let mut curve = Vec::new();
            for k in 0..=400 {
                let a = 0.005 + 0.01 * f64::from(k);
                let f = |r: f64| slope(i, r, a);
                let mut prev = (0.0, f(0.0));
                for j in 1..=200 {
                    let now = (f64::from(j) / 200.0, f(f64::from(j) / 200.0));
                    if (prev.1 < 0.0) != (now.1 < 0.0) {
                        curve.push((a, bisect(|x| (f(x) < 0.0) == (prev.1 < 0.0), prev.0, now.0)));
                    }
                    prev = now;
                }
            }
            for j in 0..=200 {
                let r = f64::from(j) / 200.0;
                let f = |a: f64| slope(i, r, a);
                let mut prev = (0.01, f(0.01));
                for k in 1..=400 {
                    let a = 0.01 + 3.99 * f64::from(k) / 400.0;
                    let now = (a, f(a));
                    if (prev.1 < 0.0) != (now.1 < 0.0) {
                        curve.push((bisect(|x| (f(x) < 0.0) == (prev.1 < 0.0), prev.0, now.0), r));
                    }
                    prev = now;
                }
            }
            for (a, rho) in pts {
                let px = curve.iter().map(|&(ca, cr)| (((ca - a) * 71.75).powi(2) + ((cr - rho) * 147.0).powi(2)).sqrt()).fold(f64::MAX, f64::min);
                assert!(px < 2.5, "Fig. 11, I = {i}: ({a}, {rho}) is {px} pixels from the computed curve");
            }
        }
    }

    /// Every refusal of the Lewis–Rinzel side, each rendered.
    #[test]
    fn every_lewis_rinzel_refusal_names_what_it_refused() {
        let msg = |r: Result<LewisRinzel, GapError>| r.unwrap_err().to_string();
        assert_eq!(msg(LewisRinzel::new(f64::NAN, 0.1, 0.1, 0.1, 1.0)), "I = NaN is not finite");
        assert_eq!(msg(LewisRinzel::new(1.0, 0.1, 0.1, 0.1, 1.0)), "drive I = 1 does not exceed the threshold 1, so an uncoupled cell never fires");
        assert_eq!(msg(LewisRinzel::new(1.2, -0.1, 0.1, 0.1, 1.0)), "g_c = -0.1 must be finite and not negative");
        assert_eq!(msg(LewisRinzel::new(1.2, 0.1, f64::NAN, 0.1, 1.0)), "beta = NaN must be finite and not negative");
        assert_eq!(msg(LewisRinzel::new(1.2, 0.1, 0.1, f64::INFINITY, 1.0)), "g_s = inf is not finite");
        assert_eq!(msg(LewisRinzel::new(1.2, 0.1, 0.1, 0.1, 0.0)), "alpha = 0 must be finite and positive");
        assert_eq!(msg(LewisRinzel::new(1.2, 0.1, 0.1, 0.1, f64::INFINITY)), "alpha = inf must be finite and positive");
        assert_eq!(msg(LewisRinzel::new(1.2, 0.1, f64::INFINITY, 0.1, 1.0)), "beta = inf must be finite and not negative");
        assert!(LewisRinzel::new(1.2, 0.0, 0.0, -0.3, 1.0).is_ok(), "excitation is allowed");
        assert_eq!(msg(LewisRinzel::combined(1.2, 1.5, 1.0, 0.1, 1.0)), "rho = 1.5 is outside [0, 1]");
        assert_eq!(msg(LewisRinzel::combined(1.2, -0.25, 1.0, 0.1, 1.0)), "rho = -0.25 is outside [0, 1]");
        assert_eq!(msg(LewisRinzel::combined(1.2, 0.5, 0.0, 0.1, 1.0)), "g_tot = 0 must be finite and positive");
        let c = LewisRinzel::combined(1.2, 0.25, 2.0, 0.1, 1.0).unwrap();
        assert_eq!((c.g_c, c.g_s), (0.5, 1.5));
        let m = LewisRinzel::FIG2_LOW;
        assert_eq!(m.h(f64::NAN).unwrap_err().to_string(), "psi = NaN is not finite");
        assert_eq!(m.g(f64::INFINITY).unwrap_err().to_string(), "phi = inf is not finite");
        assert_eq!(m.g_slope(f64::NAN).unwrap_err().to_string(), "phi = NaN is not finite");
        assert_eq!(m.prc(f64::NAN).unwrap_err().to_string(), "t = NaN is not finite");
        assert_eq!(m.limit_cycle(f64::NEG_INFINITY).unwrap_err().to_string(), "t = -inf is not finite");
        assert_eq!(m.synaptic_current(f64::NAN).unwrap_err().to_string(), "t = NaN is not finite");
        assert_eq!(m.phase_advance(f64::NAN, 0.5).unwrap_err().to_string(), "eps = NaN is not finite");
        assert_eq!(m.phase_advance(0.1, f64::NAN).unwrap_err().to_string(), "t = NaN is not finite");
        let bad = LewisRinzel { drive: 0.5, ..m };
        for r in [bad.period(), bad.h(0.1), bad.g(0.1), bad.g_slope(0.1), bad.prc(0.1), bad.limit_cycle(0.1), bad.synaptic_current(0.1), bad.phase_advance(0.1, 0.1)] {
            assert_eq!(r.unwrap_err(), GapError::Subthreshold { drive: 0.5 });
        }
        assert_eq!(bad.locked_states(10).unwrap_err(), GapError::Subthreshold { drive: 0.5 });
        let none = LewisRinzel { g_c: 0.0, g_s: 0.0, ..m };
        assert_eq!(none.locked_states(10).unwrap_err().to_string(), "g_c = 0 and g_s = 0: with no coupling every phase difference is locked");
        assert_eq!(none.probability_of_synchrony(10).unwrap_err(), GapError::Uncoupled);
        assert!(LewisRinzel { g_s: 0.0, ..m }.locked_states(3).is_ok() && LewisRinzel { g_c: 0.0, g_s: 0.1, ..m }.locked_states(3).is_ok());
        assert_eq!(m.locked_states(2).unwrap_err().to_string(), "samples = 2 must be at least 3");
        assert_eq!(m.antiphase_critical_drives(1.0, 2.0, 10).unwrap_err(), GapError::Subthreshold { drive: 1.0 });
        assert_eq!(m.antiphase_critical_drives(1.5, f64::NAN, 10).unwrap_err().to_string(), "I = NaN is not finite");
        assert_eq!(m.antiphase_critical_drives(1.5, 1.5, 10).unwrap_err().to_string(), "the interval (1.5, 1.5) is empty");
        assert_eq!(m.antiphase_critical_drives(1.5, 1.6, 0).unwrap_err().to_string(), "samples = 0 must be at least 1");
        assert_eq!(LewisRinzel::critical_drive_electrical(0.0).unwrap_err().to_string(), "beta = 0 must be finite and positive");
        assert_eq!(LewisRinzel::critical_drive_electrical(f64::NAN).unwrap_err().to_string(), "beta = NaN must be finite and positive");
        assert_eq!(m.antiphase_orbit(0.0, 10).unwrap_err().to_string(), "t_max = 0 must be finite and positive");
        assert_eq!(m.antiphase_orbit(8.0, 0).unwrap_err().to_string(), "samples = 0 must be at least 1");
        assert_eq!(bad.antiphase_orbit(8.0, 10).unwrap_err(), GapError::Subthreshold { drive: 0.5 });
        assert_eq!(m.synchronous_period(f64::NAN, 10).unwrap_err().to_string(), "t_max = NaN must be finite and positive");
        assert_eq!(m.synchronous_period(3.0, 0).unwrap_err().to_string(), "samples = 0 must be at least 1");
        assert_eq!(bad.synchronous_period(3.0, 10).unwrap_err(), GapError::Subthreshold { drive: 0.5 });
        // No root on the interval, and a root the kick would not let happen.
        assert_eq!(m.antiphase_orbit(0.5, 10).unwrap(), None);
        assert_eq!(m.synchronous_period(0.5, 10).unwrap(), None);
        let captured = LewisRinzel { g_c: 0.5, beta: 1.5, ..m };
        let raw = LewisRinzel { beta: 0.0, ..captured }.antiphase_orbit(8.0, 400).unwrap().unwrap();
        assert!(raw.u + 0.75 >= 1.0, "{raw:?}");
        assert_eq!(captured.antiphase_orbit(8.0, 400).unwrap(), None, "u + g_c β ≥ 1: spike capture, not antiphase");
        let pair = |v0: [f64; 2]| LewisRinzelPair::new(m, v0).unwrap_err().to_string();
        assert_eq!(pair([f64::NAN, 0.0]), "v0 = NaN is not finite");
        assert_eq!(pair([0.0, 1.0]), "cell 1 starts at v = 1, not below the threshold 1");
        assert_eq!(LewisRinzelPair::new(bad, [0.0, 0.0]).unwrap_err(), GapError::Subthreshold { drive: 0.5 });
        let mut p = LewisRinzelPair::new(m, [0.2, 0.0]).unwrap();
        assert_eq!(p.run(0.0, 8).unwrap_err().to_string(), "duration = 0 must be finite and positive");
        assert_eq!(p.run(1.0, 0).unwrap_err().to_string(), "samples = 0 must be at least 1");
        assert_eq!((p.model(), p.time(), p.voltages()), (m, 0.0, [0.2, 0.0]));
    }

    /// The response kernels are eq. (2.13) integrated: Simpson's rule on the spike current plus the
    /// reset's `δ`, against eq. (2.14), agreeing to 1.2 × 10⁻¹⁴ (measured). The kernel jumps
    /// by the reset `−(1 + v_M)` at `Δ`, takes the spike's value AT `Δ`, and is 0 before the spike.
    /// `γ_s` and `γ_c` are eqs. (2.21) and (2.23) as printed, and `δ_c = 1 + 2γ_c(Δ)` (2.22).
    ///
    /// "For `γ_c` given by equation 2.23, `γ̇_c(0) = 0`" (p. 1657): every mode's kernel leaves the
    /// spike with slope `v_A`, so `γ_c(h) = v_A g h² / 2 + O(h³)`, which it is at `h = 10⁻⁶` to a
    /// relative 1.5 × 10⁻⁵ (measured, the size of the next term).
    #[test]
    fn the_kernels_are_eq_2_13_integrated() {
        let m = ChowKopell::FIG3;
        let vm = m.v_m();
        assert!((vm - 2.892_320_033_030_209).abs() < 1e-14, "{vm}");
        for r in [1.0, 2.0, 11.0] {
            for t in [0.02f64, 0.05, 0.1, 0.100_000_001, 0.3, 1.0, 3.0] {
                let spike = simpson(|u| (-r * (t - u)).exp() * m.v_a * (m.xi * u).exp(), 0.0, t.min(m.width), 20000);
                let reset = if t > m.width { (1.0 + vm) * (-r * (t - m.width)).exp() } else { 0.0 };
                assert!((m.eta(t, r).unwrap() - (spike - reset)).abs() < 5e-14, "r = {r}, t = {t}");
            }
            let jump = m.eta(m.width * (1.0 + 1e-15), r).unwrap() - m.eta(m.width, r).unwrap();
            assert!((jump + 1.0 + vm).abs() < 1e-12, "{jump}");
            assert_eq!((m.eta(0.0, r).unwrap(), m.eta(-0.5, r).unwrap()), (0.0, 0.0));
        }
        for k in [m, ChowKopell::FIG2_STRONG, ChowKopell::new(0.1, 12.0, 0.5, 0.3).unwrap()] {
            let h = 1e-6;
            let ratio = k.gamma_c(h).unwrap() / (h * h) / (k.v_a * k.g / 2.0);
            assert!((ratio - 1.0).abs() < 5e-5, "g = {}: γ_c(h)/(v_A g h²/2) = {ratio}", k.g);
        }
        let dc = m.delta_c().unwrap();
        assert!((dc - (1.0 + 2.0 * m.gamma_c(m.width).unwrap())).abs() < 1e-15 && (dc - 1.053_965_641_693_696).abs() < 1e-14);
        let r = 1.0 + 2.0 * m.g;
        for t in [0.01, 0.07, 0.1, 0.2, 0.9, 4.0] {
            let (gs, gc) = if t <= m.width {
                let a = ((m.xi * t).exp() - (-t).exp()) / (1.0 + m.xi);
                let b = ((m.xi * t).exp() - (-r * t).exp()) / (r + m.xi);
                (m.v_a / 2.0 * (a + b), m.v_a / 2.0 * (a - b))
            } else {
                let (a, b) = ((-(t - m.width)).exp(), dc * (-r * (t - m.width)).exp());
                (-0.5 * (a + b), -0.5 * (a - b))
            };
            assert!((m.gamma_s(t).unwrap() - gs).abs() < 1e-14 && (m.gamma_c(t).unwrap() - gc).abs() < 1e-14, "t = {t}");
        }
        // At exactly Δ the kernel is the spike's (a binary Δ puts t = Δ on the float).
        let b = ChowKopell { width: 0.125, ..m };
        assert_eq!(b.eta(0.125, 3.0).unwrap(), b.v_a / 53.0 * ((50.0f64 * 0.125).exp() - (-3.0f64 * 0.125).exp()));
    }

    /// `δ_c`'s two approximations. (2.24), for fast-rising spikes, drops the `e^{−Δ}` and `e^{−rΔ}`
    /// terms against `e^{ξΔ}` and takes `v_M ≃ v_A e^{ξΔ}/(1 + ξ)` (p. 1650): its relative error in
    /// `δ_c − 1` is measured at 3.70% for Fig. 3's spike (`ξΔ = 5`) and 7.0 × 10⁻⁵ at `ξ = 120`
    /// (`ξΔ = 12`), falling like `e^{−ξΔ}` times a prefactor that grows with `ξ` — 5.49 and 11.3
    /// there. (2.25), for weak coupling, tends as `g → 0` not to 0 but to 3.86%: 3.23 points of it
    /// are the `v_A Δ e^{−Δ}/(1 + ξ)` that the fast-spike expansion drops (the error left with the
    /// exact `v_M`), and the rest is `v_M ≃ v_A e^{ξΔ}/(1 + ξ)`. The limits are
    /// `v_AΔe^{−Δ}/(v_M − v_AΔe^{−Δ})` and `v_A e^{ξΔ}/((1 + ξ)(v_M − v_AΔe^{−Δ})) − 1`, reached at
    /// `g = 10⁻⁴` to 4 × 10⁻⁶ (measured).
    #[test]
    fn delta_c_approximations_2_24_and_2_25() {
        let fast = |m: ChowKopell| m.v_a * (m.xi * m.width).exp() / (1.0 + m.xi);
        let fig3 = ChowKopell::FIG3;
        let err24 = |m: ChowKopell| {
            let r = 1.0 + 2.0 * m.g;
            fast(m) * (1.0 - (1.0 + m.xi) / (r + m.xi)) / (m.delta_c().unwrap() - 1.0) - 1.0
        };
        let (e50, e120) = (err24(fig3), err24(ChowKopell { xi: 120.0, ..fig3 }));
        assert!((e50 - 0.037_007).abs() < 1e-6 && (e120 - 6.958e-5).abs() < 1e-8, "{e50} {e120}");
        let (p50, p120) = (e50 * 5.0f64.exp(), e120 * 12.0f64.exp());
        assert!((p50 - 5.492).abs() < 1e-3 && (p120 - 11.325).abs() < 1e-3, "prefactors {p50} {p120}");
        // (2.25) with the fast-spike v_M, as p. 1650 has it, and with the exact v_M = η₊(Δ).
        let err25 = |m: ChowKopell, vm: f64| 2.0 * m.g * vm / (1.0 + m.xi) / (m.delta_c().unwrap() - 1.0) - 1.0;
        let dropped = fig3.v_a * fig3.width * (-fig3.width).exp();
        let (as_printed, exact) = (fast(fig3) / (fig3.v_m() - dropped) - 1.0, dropped / (fig3.v_m() - dropped));
        assert!((as_printed - 0.038_627).abs() < 1e-6 && (exact - 0.032_294).abs() < 1e-6, "{as_printed} {exact}");
        for g in [1e-3, 1e-4] {
            let weak = ChowKopell { g, ..fig3 };
            let (a, b) = (err25(weak, fast(weak)), err25(weak, weak.v_m()));
            let tol = if g < 5e-4 { 1e-5 } else { 1e-4 };
            assert!((a - as_printed).abs() < tol && (b - exact).abs() < tol, "g = {g}: {a} against {as_printed}, {b} against {exact}");
        }
    }

    /// Fig. 2(b): `γ_c` bottoms out at `t_min` of eq. (3.24) — 0.8457 at `g = 0.5`, 0.3780 at `g = 5` —
    /// at −0.1186 and −0.3442. The figure's vector paths (reference E) draw the minima at −0.1189 and
    /// −0.3444 near `t` = 0.825 and 0.374 (its vertices are about 0.02 apart), and the peaks at the
    /// cusp at 0.0267 and 0.2324, where `γ_c(Δ) = (δ_c − 1)/2` is 0.0270 and 0.2324. As `g → 0`,
    /// `t_min` tends to `Δ + 1 + (v_M − v_A Δ e^{−Δ})/(1 + ξ)` — 1.15494 for Fig. 3's spike, reached to
    /// 10⁻⁴ at `g = 10⁻⁴` — and eq. (3.25), `Δ + 1 + v_M/(1 + ξ)` = 1.15671, misses that limit by
    /// the `v_A Δ e^{−Δ}/(1 + ξ)` = 0.0018 that (2.25)'s fast-spike approximation drops.
    #[test]
    fn figure_2_and_the_minimum_of_the_coupling_kernel() {
        for (g, tmin_want, low_drawn, high_drawn) in [(0.5, 0.845_707_032_134_218, -0.1189, 0.0267), (5.0, 0.377_959_646_094_614, -0.3444, 0.2324)] {
            let m = ChowKopell { g, ..ChowKopell::FIG3 };
            let tmin = m.t_min().unwrap();
            assert!((tmin - tmin_want).abs() < 1e-14, "{tmin}");
            let low = m.gamma_c(tmin).unwrap();
            for d in [-1e-4, 1e-4] {
                assert!(m.gamma_c(tmin + d).unwrap() > low, "t_min is the minimum");
            }
            let fd = (m.gamma_c(tmin + 1e-6).unwrap() - m.gamma_c(tmin - 1e-6).unwrap()) / 2e-6;
            assert!(fd.abs() < 1e-8, "{fd}");
            assert!((low - low_drawn).abs() < 5e-4, "drawn {low_drawn}, computed {low}");
            assert!((m.gamma_c(m.width).unwrap() - high_drawn).abs() < 5e-4);
        }
        let weak = ChowKopell { g: 1e-4, ..ChowKopell::FIG3 };
        let limit = weak.width + 1.0 + (weak.v_m() - weak.v_a * weak.width * (-weak.width).exp()) / (1.0 + weak.xi);
        let eq325 = weak.width + 1.0 + weak.v_m() / (1.0 + weak.xi);
        assert!((weak.t_min().unwrap() - limit).abs() < 2e-4, "{} {limit}", weak.t_min().unwrap());
        assert!((eq325 - limit - 0.001_774).abs() < 1e-6, "{eq325} {limit}");
    }

    /// `G(φ, T)` from the kernel sums is eqs. (3.19)–(3.20) for `T ≥ 2Δ`, and a brute-force sum of
    /// eq. (3.4) over two thousand periods at any `T > Δ`, including a lattice point that lands
    /// exactly on `Δ` (`Δ = 0.125`, `T = 0.5`, `φ = 0.75`: `T − φT = Δ`); it is odd about `φ = ½` and
    /// vanishes at 0 and ½.
    #[test]
    fn the_locking_function_is_eqs_3_19_and_3_20() {
        let closed = |m: ChowKopell, phi: f64, t: f64| {
            let (d, r, dc) = (m.width, 1.0 + 2.0 * m.g, m.delta_c().unwrap());
            let e = |x: f64| (-x).exp();
            let sum = |a: f64, b: f64| (e(a - d) - e(b - d)) / (1.0 - e(t)) - dc * (e(r * (a - d)) - e(r * (b - d))) / (1.0 - e(r * t));
            if phi * t <= d { m.gamma_c(phi * t).unwrap() - 0.5 * sum(t + phi * t, t - phi * t) } else { -0.5 * sum(phi * t, t - phi * t) }
        };
        let brute = |m: ChowKopell, phi: f64, t: f64| {
            let mut s = m.gamma_c(phi * t).unwrap();
            for l in 1..2000 {
                s += m.gamma_c(f64::from(l) * t + phi * t).unwrap() - m.gamma_c(f64::from(l) * t - phi * t).unwrap();
            }
            s
        };
        for m in [ChowKopell::FIG3, ChowKopell::FIG2_STRONG, ChowKopell::new(0.1, 12.0, 0.5, 0.3).unwrap()] {
            for t in [2.0 * m.width, 3.0 * m.width, 7.0 * m.width, 13.0 * m.width] {
                for k in 1..10 {
                    let phi = 0.05 * f64::from(k);
                    let g = m.locking(phi, t).unwrap();
                    assert!((g - closed(m, phi, t)).abs() < 3e-15, "T = {t}, φ = {phi}");
                    assert!((g - brute(m, phi, t)).abs() < 3e-15);
                    assert!((g + m.locking(1.0 - phi, t).unwrap()).abs() < 2e-15, "odd about ½");
                }
            }
            for t in [1.1 * m.width, 1.5 * m.width, 1.9 * m.width] {
                for phi in [0.1, 0.3, 0.45, 0.8] {
                    assert!((m.locking(phi, t).unwrap() - brute(m, phi, t)).abs() < 3e-15, "overlapping spikes, T = {t}");
                }
            }
            assert_eq!(m.locking(0.0, 3.0 * m.width).unwrap(), 0.0);
            assert!(m.locking(0.5, 3.0 * m.width).unwrap().abs() < 1e-15);
        }
        let b = ChowKopell { width: 0.125, ..ChowKopell::FIG3 };
        assert!((b.locking(0.75, 0.5).unwrap() - brute(b, 0.75, 0.5)).abs() < 1e-13, "{} {}", b.locking(0.75, 0.5).unwrap(), brute(b, 0.75, 0.5));
        assert!((b.locking(0.25, 0.5).unwrap() - brute(b, 0.25, 0.5)).abs() < 1e-13);
    }

    /// `∂G/∂φ` is the derivative of `G` (central differences, relative residual 6.5 × 10⁻⁹, measured),
    /// and at `φ = 0` and `½` it is eqs. (3.22) and (3.27)–(3.28), both branches of the latter and
    /// the cusp `T = 2Δ` from the left, to 1.8 × 10⁻¹⁵ and 8.9 × 10⁻¹⁶ (measured). Eq. (3.22) has no
    /// term from the cell's own spike, `γ̇_c(0) = 0` (p. 1657); nor does the slope at a phase just
    /// below 0, which wraps to exactly 1 and puts the partner's spike one period back at `lT − φT = 0`
    /// (agreement 4.4 × 10⁻¹⁶, measured).
    #[test]
    fn the_locking_slopes_are_eqs_3_22_and_3_27() {
        for m in [ChowKopell::FIG3, ChowKopell::FIG2_STRONG, ChowKopell::new(0.1, 12.0, 0.5, 0.3).unwrap()] {
            let (d, r, dc) = (m.width, 1.0 + 2.0 * m.g, m.delta_c().unwrap());
            for t in [1.2 * d, 1.7 * d, 2.0 * d, 2.5 * d, 1.0, 3.0] {
                let s0 = m.locking_slope(0.0, t).unwrap() / t;
                let eq322 = d.exp() / t.exp_m1() - dc * r * (r * d).exp() / (r * t).exp_m1();
                assert!((s0 - eq322).abs() < 5e-15 * eq322.abs().max(1.0), "T = {t}: {s0} {eq322}");
                let wrapped = m.locking_slope(-1e-17, t).unwrap() / t;
                assert!((wrapped - s0).abs() < 1e-15 * s0.abs().max(1.0), "T = {t}: {wrapped} {s0}");
                let s5 = m.locking_slope(0.5, t).unwrap() / t;
                let tail = |a: f64| (-(a - d)).exp() / -(-t).exp_m1() - dc * r * (-r * (a - d)).exp() / -(-r * t).exp_m1();
                let eq327 = if t <= 2.0 * d {
                    let dot = m.v_a / 2.0 * ((m.xi * (m.xi * t / 2.0).exp() + (-t / 2.0).exp()) / (1.0 + m.xi) - (m.xi * (m.xi * t / 2.0).exp() + r * (-r * t / 2.0).exp()) / (r + m.xi));
                    2.0 * dot + tail(1.5 * t)
                } else {
                    tail(0.5 * t)
                };
                assert!((s5 - eq327).abs() < 3e-15 * eq327.abs().max(1.0), "T = {t}: {s5} {eq327}");
                for phi in [0.13, 0.37, 0.61, 0.88] {
                    let h = 1e-7;
                    let fd = (m.locking(phi + h, t).unwrap() - m.locking(phi - h, t).unwrap()) / (2.0 * h);
                    assert!((m.locking_slope(phi, t).unwrap() - fd).abs() < 3e-8 * fd.abs().max(1.0), "T = {t}, φ = {phi}");
                }
            }
        }
    }

    /// The four critical periods of Fig. 4 for Fig. 3's spike: `T_C^S = 0.284 896` (eq. 3.23),
    /// `T_C^{AS1} = 1.133 045` (eq. 3.29), `T_C^{AS2} = 2Δ` where `∂G/∂φ(½)` jumps sign at the cusp,
    /// and `T_C^{AS3} = 0.114 737` below it — each a sign change of the closed-form slope found by
    /// [`ChowKopell::slope_switches`], and the first two also the roots of their printed equations.
    ///
    /// ⚠ "For small r, increasing r will increase `T_C^{AS3}`. Thus, the regime for stable AS at these
    /// short periods is reduced" (p. 1659). The paper's own (3.27)–(3.28) say the opposite: for every
    /// spike shape tried that has a `T_C^{AS3}`, it FALLS as `r` grows, from the smallest `g` up —
    /// 0.115 430 at `g = 0.001`, 0.114 737 at `g = 0.5`, 0.114 055 at `g = 1` for Fig. 3's spike —
    /// so a stronger junction widens the short-period window `(T_C^{AS3}, 2Δ]` in which antiphase
    /// meets the necessary condition.
    #[test]
    fn the_critical_periods_are_eqs_3_23_and_3_29() {
        let m = ChowKopell::FIG3;
        let (d, r, dc) = (m.width, 2.0, m.delta_c().unwrap());
        let s = m.slope_switches(0.0, 1.000_001 * d, 3.0, 3000).unwrap();
        let eq323 = bisect(|t| (r * t).exp_m1() / t.exp_m1() < r * dc * ((r - 1.0) * d).exp(), d, 3.0);
        assert!(s.len() == 1 && (s[0] - eq323).abs() < 1e-14 && (s[0] - 0.284_896_460_009_709).abs() < 1e-13, "{s:?} {eq323}");
        let a = m.slope_switches(0.5, 1.000_001 * d, 3.0, 3000).unwrap();
        let eq329 = bisect(|t| (r * t / 2.0).sinh() / (t / 2.0).sinh() < r * dc * ((r - 1.0) * d).exp(), 2.0 * d, 3.0);
        assert_eq!(a.len(), 3, "{a:?}");
        assert!((a[0] - 0.114_736_734_454_847).abs() < 1e-12, "T_C^AS3 = {}", a[0]);
        assert!((a[1] - 2.0 * d).abs() < 1e-15, "T_C^AS2 = 2Δ: {}", a[1]);
        assert!((a[2] - eq329).abs() < 1e-14 && (a[2] - 1.133_045_489_955_725).abs() < 1e-12, "{} {eq329}", a[2]);
        let as3 = |g: f64| ChowKopell { g, ..m }.slope_switches(0.5, 1.000_001 * d, 1.999 * d, 2000).unwrap()[0];
        let ladder: Vec<f64> = [0.001, 0.01, 0.05, 0.25, 0.5, 1.0].iter().map(|&g| as3(g)).collect();
        assert!(ladder.windows(2).all(|w| w[1] < w[0]), "{ladder:?}");
        assert!((ladder[0] - 0.115_430).abs() < 1e-6 && (ladder[5] - 0.114_055).abs() < 1e-6, "{ladder:?}");
        for (va, xi, w) in [(1.0, 12.0, 0.5), (0.1, 12.0, 0.5), (0.1, 50.0, 0.1), (5.0, 5.0, 0.3)] {
            let at = |g: f64| ChowKopell { v_a: va, xi, width: w, g }.slope_switches(0.5, 1.000_001 * w, 1.999 * w, 2000).unwrap()[0];
            assert!(at(0.01) < at(0.001) && at(0.2) < at(0.01), "v_A {va}, ξ {xi}, Δ {w}");
        }
    }

    /// Fig. 3's four panels, at their printed parameters: the zeros and slope signs are those the text
    /// describes — (a) `T = 2`: synchrony and antiphase with `∂G/∂φ > 0`, an unstable third mode at
    /// 0.1072; (b) `T = 1`: antiphase has lost it; (c) `T = 0.25`: synchrony has lost it, a stable
    /// third mode at 0.2920; (d) `T = 0.2 = 2Δ`: antiphase has regained it at the cusp.
    ///
    /// ⚠ The figure draws `½G`. Its vector paths (reference E) put the extremes at 0.019722, 0.025000
    /// (clipped at the axis, four points), 0.008583 and 0.006222; `G` of eq. (3.4) at the printed
    /// parameters has 0.039474, 0.050842, 0.017212 and 0.012492 — each twice the drawing, to the
    /// drawing's resolution, in all four panels. Reference E fits every drawn vertex by least
    /// squares: scales 0.4998, 0.4994, 0.4998 and 0.4997 at the printed `g = 0.5`, and on a grid of
    /// 75 values of `g` from 0.02 to 6 the panels agree on one scale only there (to 0.07%, and to no
    /// better than 1.7% at any other). `g = 0.25`, where the scales straddle 1, does not reproduce
    /// the drawing at full scale: they run from 0.91 to 1.13.
    ///
    /// Here the module's own `G` is fitted to the vertices nearest `φ = k/20` (reference E, 19 per
    /// panel): at `g = 0.5` every panel's scale is within 0.0019 of ½ (measured); at `g = 0.25`
    /// they run from 0.910 to 1.121; and at every other `g` tried the four disagree by more than
    /// 3%, where at `g = 0.5` they agree to 0.61% (both measured).
    #[test]
    fn figure_3_is_drawn_at_half_scale() {
        let m = ChowKopell::FIG3;
        let zeros = |t: f64| {
            let f = |p: f64| m.locking(p, t).unwrap();
            let mut z = Vec::new();
            for k in 1..2000 {
                let (a, b) = (f64::from(k) / 2000.0, f64::from(k + 1) / 2000.0);
                if (f(a) < 0.0) != (f(b) < 0.0) && b < 0.4995 {
                    z.push(bisect(|x| (f(x) < 0.0) == (f(a) < 0.0), a, b));
                }
            }
            z
        };
        let up = |p: f64, t: f64| m.locking_slope(p, t).unwrap() > 0.0;
        let a = zeros(2.0);
        assert!(up(0.0, 2.0) && up(0.5, 2.0) && a.len() == 1 && (a[0] - 0.1072).abs() < 1e-4 && !up(a[0], 2.0), "{a:?}");
        assert!(up(0.0, 1.0) && !up(0.5, 1.0) && zeros(1.0).is_empty());
        let c = zeros(0.25);
        assert!(!up(0.0, 0.25) && !up(0.5, 0.25) && (c[0] - 0.2920).abs() < 1e-4 && up(c[0], 0.25), "{c:?}");
        assert!(!up(0.0, 0.2) && up(0.5, 0.2) && zeros(0.2).is_empty());
        for (t, drawn, clipped) in [(2.0, 0.019722, false), (1.0, 0.025000, true), (0.25, 0.008583, false), (0.2, 0.006222, false)] {
            let peak = (1..4000).map(|k| m.locking(f64::from(k) / 4000.0, t).unwrap()).fold(f64::MIN, f64::max);
            if clipped {
                assert!(peak / 2.0 > drawn, "T = {t}: ½G = {} runs off the axis", peak / 2.0);
            } else {
                assert!((peak / 2.0 / drawn - 1.0).abs() < 0.01, "T = {t}: G peaks at {peak}, drawn {drawn}");
            }
        }
        let drawn: [(f64, [(f64, f64); 19]); 4] = [
            (2.0, [(0.050578, 0.019722), (0.099570, 0.002223), (0.150097, -0.009376), (0.200625, -0.015765), (0.251152, -0.018265), (0.298659, -0.017847), (0.349186, -0.015209), (0.399713, -0.010904), (0.450241, -0.005695), (0.500768, 0.000069), (0.549811, 0.005624), (0.600338, 0.010833), (0.650865, 0.015138), (0.699857, 0.017709), (0.750384, 0.018193), (0.800962, 0.015624), (0.849954, 0.009304), (0.900481, -0.002292), (0.949473, -0.019793)]),
            (1.0, [(0.050527, 0.006943), (0.098034, 0.024583), (0.148561, 0.018472), (0.200625, 0.012777), (0.249616, 0.008748), (0.300143, 0.005624), (0.349186, 0.003542), (0.399713, 0.002013), (0.450241, 0.000832), (0.499232, 0.000000), (0.551295, -0.000973), (0.600287, -0.002084), (0.649329, -0.003613), (0.699857, -0.005834), (0.750384, -0.008889), (0.799375, -0.012917), (0.849903, -0.018544), (0.900481, -0.023891), (0.949473, -0.006669)]),
            (0.25, [(0.050578, -0.000472), (0.099570, -0.000861), (0.151633, -0.001167), (0.203696, -0.001250), (0.249616, -0.000889), (0.300195, 0.000250), (0.349186, 0.002861), (0.399713, 0.008583), (0.450241, 0.004278), (0.500768, -0.000111), (0.551346, -0.004389), (0.600338, -0.008611), (0.650865, -0.002889), (0.699857, -0.000278), (0.750384, 0.000889), (0.794819, 0.001194), (0.852974, 0.001139), (0.900481, 0.000834), (0.951008, 0.000445)]),
            (0.2, [(0.050527, -0.001250), (0.099570, -0.002389), (0.148561, -0.003472), (0.200625, -0.004555), (0.249616, -0.005417), (0.300143, -0.006055), (0.352206, -0.006250), (0.401249, -0.005667), (0.450241, -0.003861), (0.499232, 0.000000), (0.549759, 0.003945), (0.600287, 0.005695), (0.644722, 0.006222), (0.699857, 0.006027), (0.750384, 0.005389), (0.800911, 0.004472), (0.852974, 0.003389), (0.898945, 0.002361), (0.951008, 0.001139)]),
        ];
        let scales = |g: f64| -> Vec<f64> {
            let k = ChowKopell { g, ..m };
            drawn
                .iter()
                .map(|(t, pts)| {
                    let (num, den) = pts.iter().fold((0.0, 0.0), |(a, b), &(p, y)| {
                        let model = k.locking(p, *t).unwrap();
                        (a + y * model, b + model * model)
                    });
                    num / den
                })
                .collect()
        };
        let spread = |s: &[f64]| s.iter().copied().fold(f64::MIN, f64::max) / s.iter().copied().fold(f64::MAX, f64::min);
        let half = scales(0.5);
        assert!(half.iter().all(|s| (s - 0.5).abs() < 0.005) && spread(&half) < 1.01, "g = 0.5: {half:?}");
        let quarter = scales(0.25);
        assert!((spread(&quarter) - 1.231).abs() < 0.01 && quarter.iter().any(|&s| s < 1.0) && quarter.iter().any(|&s| s > 1.0), "g = 0.25: {quarter:?}");
        for g in [0.2, 0.3, 0.4, 0.45, 0.55, 0.6, 0.8, 1.0, 2.0, 5.0] {
            assert!(spread(&scales(g)) > 1.03, "g = {g}: {:?}", scales(g));
        }
    }

    /// The period functions: `F(0, T)` is eq. (3.8), `e^Δ/(1 − eᵀ)`, and its root eq. (3.9); `F(½, T)`
    /// is eq. (3.11) for `T > 2Δ`; and `F(½, T) − F(0, T)` is `ψ(T)` of eq. (3.13), positive — the
    /// antiphase period the shorter — for large spikes and negative for small ones (p. 1654).
    ///
    /// ⚠ The synchronous period exceeds the spike width only for `Ī < 1 + 1/(1 − e^{−Δ})`, 11.508 for
    /// `Δ = 0.1`. The paper prints the range as `1 < Ī ≤ 1 + (1 − e^Δ)⁻¹` (p. 1653), whose upper end
    /// is −8.508: an empty range.
    #[test]
    fn the_period_functions_are_eqs_3_8_3_9_and_3_11() {
        for m in [ChowKopell::FIG3, ChowKopell::FIG2_STRONG, ChowKopell::new(1.0, 12.0, 0.5, 0.5).unwrap()] {
            let (d, r, dc) = (m.width, 1.0 + 2.0 * m.g, m.delta_c().unwrap());
            for t in [1.01 * d, 2.0 * d, 9.0 * d, 25.0 * d] {
                let eq38 = d.exp() / -t.exp_m1();
                assert!((m.period_function(0.0, t).unwrap() - eq38).abs() < 2e-15 * eq38.abs().max(1.0), "T = {t}");
            }
            for t in [2.01 * d, 3.0 * d, 6.0 * d, 20.0 * d] {
                let eq311 = dc / 2.0 * (r * d).exp() / ((r * t / 2.0).exp() + 1.0) - 0.5 * d.exp() / (t / 2.0).exp_m1();
                let f = m.period_function(0.5, t).unwrap();
                assert!((f - eq311).abs() < 2e-15 * eq311.abs().max(1.0), "T = {t}: {f} {eq311}");
                let psi = dc / 2.0 * (r * d).exp() / ((r * t / 2.0).exp() + 1.0) - 0.5 * d.exp() / ((t / 2.0).exp() + 1.0);
                assert!((f - m.period_function(0.0, t).unwrap() - psi).abs() < 1e-13, "ψ (3.13)");
            }
            for i in [1.05, 1.3, 2.0, 3.0] {
                let ts = m.sync_period(i).unwrap();
                assert!((m.period_function(0.0, ts).unwrap() - (1.0 - i)).abs() < 1e-13, "(3.9) solves (3.8)");
            }
        }
        let small = ChowKopell::FIG3;
        let i = 1.0 - small.period_function(0.5, 2.0).unwrap();
        assert!(small.sync_period(i).unwrap() < 2.0, "small spikes (δ_c = 1.054): T_S < T_AS");
        let large = ChowKopell::new(1.0, 12.0, 0.5, 0.5).unwrap();
        assert!(large.delta_c().unwrap() > 3.0);
        let i = 1.0 - large.period_function(0.5, 3.0).unwrap();
        assert!(large.sync_period(i).unwrap() > 3.0, "large spikes (δ_c = {}): T_S > T_AS", large.delta_c().unwrap());
        let limit = 1.0 + 1.0 / -(-0.1f64).exp_m1();
        assert!((limit - 11.508_331_944_775_33).abs() < 1e-12);
        assert!(small.sync_period(limit * (1.0 - 1e-15)).unwrap() > 0.1);
        assert_eq!(small.sync_period(limit).unwrap_err(), GapError::Overdriven { drive: limit, limit });
        assert!(1.0 + 1.0 / (1.0 - 0.1f64.exp()) < 1.0, "the printed bound is below threshold");
    }

    /// Antiphase exists only while the partner's spike leaves a cell below threshold: eq. (3.15), the
    /// voltage at `T/2 + Δ` from the kernel sums, is below 1 exactly when eq. (3.16) holds, and
    /// substituting (3.11) into (3.16) gives the module's (3.17) with `e^{rT/2} + 1`.
    ///
    /// ⚠ Eq. (3.17) is printed with `e^{rT/2} − 1` in that denominator (p. 1654). At `T = 0.5` and
    /// Fig. 3's `δ_c = 1.054` its right-hand side is 4.20 where (3.16)'s is 1.79; solved for `δ_c`,
    /// its coefficient of `δ_c` is −2.02, so it bounds nothing, where the corrected one requires
    /// `δ_c < 3.90`. The misprint is carried into eq. (3.18). At `Δ = 0`, with `x = rT/2`, the
    /// misprinted (3.17) reads `δ_c · 2(sinh x − 1)/(eˣ − 1) < 1 + e^{−x}`: where `sinh x > 1` that
    /// is `δ_c < sinh x/(sinh x − 1)`, eq. (3.18); below `x = asinh 1 = 0.881` the coefficient is
    /// negative and every `δ_c > 0` satisfies it, while (3.18) as printed is negative there and would
    /// forbid antiphase outright. The corrected (3.17) becomes `δ_c < coth(rT/4)` at every `x`. For
    /// Fig. 3's spike the longest antiphase period is 3.3688 by (3.15) and by the corrected (3.17),
    /// and 3.4158 by the printed one.
    #[test]
    fn antiphase_existence_eq_3_16_and_the_misprint_in_3_17() {
        let m = ChowKopell::FIG3;
        let (d, r, dc) = (m.width, 2.0, m.delta_c().unwrap());
        let eq316 = |t: f64, i: f64| (3.0 - 2.0 * i + 1.0 / (t / 2.0).exp_m1()) * (1.0 + (-r * t / 2.0).exp());
        let eq317 = |t: f64, minus: bool| {
            let den = (r * t / 2.0).exp() + if minus { -1.0 } else { 1.0 };
            (1.0 - (d.exp() - 1.0) / (t / 2.0).exp_m1() + dc * (r * d).exp() / den) * (1.0 + (-r * t / 2.0).exp())
        };
        for t in [0.5, 1.0, 2.0, 3.0, 4.6, 5.0] {
            let i = 1.0 - m.period_function(0.5, t).unwrap();
            let peak = m.antiphase_partner_peak(t).unwrap();
            assert_eq!(peak < 1.0, dc < eq316(t, i), "T = {t}");
            assert!((eq316(t, i) - eq317(t, false)).abs() < 1e-12, "(3.17) is (3.16) with (3.11)");
            // (3.16) is the peak condition rearranged: equal at the boundary, same side elsewhere.
            assert!(((peak - 1.0) * (dc - eq316(t, i))) >= 0.0);
        }
        assert!((eq317(0.5, true) - 4.199_637_960_844_586).abs() < 1e-9 && (eq317(0.5, false) - 1.792_450_200_540_32).abs() < 1e-9);
        // (3.17) is affine in δ_c: δ_c < c₀ + k δ_c, a bound δ_c < c₀/(1 − k) only while 1 − k > 0.
        let affine = |t: f64, minus: bool| {
            let c0 = eq317_at(m, t, minus, 0.0);
            (c0, eq317_at(m, t, minus, 1.0) - c0)
        };
        let ((c_p, k_p), (c_f, k_f)) = (affine(0.5, true), affine(0.5, false));
        assert!((1.0 - k_p + 2.024_75).abs() < 1e-5 && (c_f / (1.0 - k_f) - 3.903_26).abs() < 1e-5, "{} {}", 1.0 - k_p, c_f / (1.0 - k_f));
        assert!((1.0 - k_f + (-r * (0.25 - d)).exp_m1()).abs() < 1e-14, "the corrected coefficient is 1 − e^(−r(T/2 − Δ))");
        assert!(c_p > 0.0 && eq317_at(m, 0.5, true, 1e6) > 1e6, "the printed (3.17) holds for any δ_c");
        let t_true = bisect(|t| m.antiphase_partner_peak(t).unwrap() < 1.0, 1.0, 8.0);
        let t_fixed = bisect(|t| dc < eq317(t, false), 1.0, 8.0);
        let t_printed = bisect(|t| dc < eq317(t, true), 1.0, 8.0);
        assert!((t_true - t_fixed).abs() < 1e-12 && (t_true - 3.368_841_156).abs() < 1e-9 && (t_printed - 3.415_791_163).abs() < 1e-9, "{t_true} {t_fixed} {t_printed}");
        // At Δ = 0, with x = rT/2, (3.17) reads δ_c < (1 + δ_c/(eˣ ∓ 1))(1 + e^{−x}), the misprint
        // taking the minus sign.
        for x in [0.3, 0.5, 0.8, 1.5, 3.0] {
            let (sinh, coth) = (f64::sinh(x), 1.0 / (x / 2.0).tanh());
            let eq318 = sinh / (sinh - 1.0);
            let holds = |minus: bool, d: f64| d < (1.0 + d / (x.exp() + if minus { -1.0 } else { 1.0 })) * (1.0 + (-x).exp());
            let coefficient = 2.0 * (sinh - 1.0) / x.exp_m1();
            assert!((1.0 - (1.0 + (-x).exp()) / x.exp_m1() - coefficient).abs() < 1e-14, "x = {x}");
            assert_eq!(coefficient < 0.0, x < 1.0f64.asinh(), "x = {x}");
            if coefficient > 0.0 {
                // A bound, and it is (3.18): satisfied just below, violated just above.
                assert!(holds(true, eq318 * (1.0 - 1e-9)) && !holds(true, eq318 * (1.0 + 1e-9)), "x = {x}");
            } else {
                // No bound at all, where (3.18) as printed is negative.
                assert!(eq318 < 0.0 && [0.01, 1.0, 1e3, 1e9].iter().all(|&d| holds(true, d)), "x = {x}");
            }
            assert!(holds(false, coth * (1.0 - 1e-9)) && !holds(false, coth * (1.0 + 1e-9)), "x = {x}: coth(rT/4)");
        }
    }

    /// The right-hand side of eq. (3.17) for spike `m` at period `t` and a given `δ_c`, with the
    /// printed `e^{rT/2} − 1` (`minus`) or the derived `e^{rT/2} + 1` in its last denominator.
    fn eq317_at(m: ChowKopell, t: f64, minus: bool, dc: f64) -> f64 {
        let (d, r) = (m.width, 1.0 + 2.0 * m.g);
        let den = (r * t / 2.0).exp() + if minus { -1.0 } else { 1.0 };
        (1.0 - (d.exp() - 1.0) / (t / 2.0).exp_m1() + dc * (r * d).exp() / den) * (1.0 + (-r * t / 2.0).exp())
    }

    /// ⚠ Eq. (2.6) as printed, `v(t) = 1 + I(1 − e^{−t}) + v_A(e^{ξt} − e^{−t})/(1 + ξ)`, does not solve
    /// eq. (2.3) from `v(0) = 1`: its slope at 0 is `I + v_A` where the equation gives `I − 1 + v_A`.
    /// The solution has `I − 1` for `I` ([`ChowKopell::spike_voltage`], whose relative residual in
    /// eq. (2.3) is 4.2 × 10⁻¹⁰ by central differences, measured).
    ///
    /// ⚠ And Fig. 1 is drawn with a reset the analysis does not use. Its vector paths (reference E,
    /// p. 1647) space the spikes' peaks 1.5655 apart in panels (a) and (b) and 1.538 in (c) and (d):
    /// that is `Δ + ln(I/(I − 1))` — 1.5663 and 1.5361 — the period of a cell reset EXACTLY to 0,
    /// eq. (2.6)'s `v_M = v(Δ) − 1` ([`Reset::ToZero`]); eq. (3.9), which the analysis derives with
    /// `v_M ≡ η₊(Δ)` ([`Reset::Kernel`]), gives 1.5441 and 1.3857. The highest drawn peaks of (a), (b)
    /// and (c) are within 3% below the computed ones — 3.8890, 1.1854 and 31.943 against 3.921, 1.214
    /// and 32.20 — but (d), captioned `v_A = .1`, is drawn at 7.1887 where `v_A = 0.1` peaks at 4.32;
    /// `v_A = 0.2` peaks at 7.41.
    #[test]
    fn eq_2_6_as_printed_does_not_solve_2_3_and_figure_1_resets_to_zero() {
        for (m, i) in [(ChowKopell::FIG1_A, 1.3), (ChowKopell::FIG1_C, 1.55)] {
            let h = 1e-6;
            for k in 1..10 {
                let t = m.width * f64::from(k) / 10.0;
                let dv = (m.spike_voltage(i, t + h).unwrap() - m.spike_voltage(i, t - h).unwrap()) / (2.0 * h);
                let rhs = i - m.spike_voltage(i, t).unwrap() + m.v_a * (m.xi * t).exp();
                assert!((dv - rhs).abs() < 2e-9 * rhs.abs().max(1.0), "t = {t}: {dv} {rhs}");
            }
            assert_eq!(m.spike_voltage(i, 0.0).unwrap(), 1.0);
            let printed = |t: f64| 1.0 + i * -(-t).exp_m1() + m.v_a / (1.0 + m.xi) * ((m.xi * t).exp() - (-t).exp());
            let slope0 = (printed(h) - printed(0.0)) / h;
            assert!((slope0 - (i + m.v_a)).abs() < 1e-3 && (slope0 - (i - 1.0 + m.v_a)).abs() > 0.99, "{slope0}");
        }
        for (m, i, drawn, tol) in [
            (ChowKopell::FIG1_A, 1.3, 3.8890, 0.03),
            (ChowKopell::FIG1_B, 1.3, 1.1854, 0.03),
            (ChowKopell::FIG1_C, 1.55, 31.943, 0.03),
            (ChowKopell { v_a: 0.2, ..ChowKopell::FIG1_D }, 1.55, 7.1887, 0.04),
        ] {
            let peak = m.spike_voltage(i, m.width).unwrap();
            assert!(drawn < peak && peak < drawn * (1.0 + tol), "peak {peak}, drawn {drawn}");
        }
        assert!((ChowKopell::FIG1_D.spike_voltage(1.55, 0.5).unwrap() - 4.315).abs() < 1e-3, "the caption's v_A");
        for (m, i, drawn) in [(ChowKopell::FIG1_A, 1.3, 1.5655), (ChowKopell::FIG1_C, 1.55, 1.538)] {
            let period = |reset: Reset| {
                let mut net = ChowKopellNet::new(m, vec![i], vec![0.0]).unwrap().with_reset(reset);
                let s = net.run(12.0, 0.01).unwrap();
                s[s.len() - 1].time - s[s.len() - 2].time
            };
            let (zero, kernel) = (period(Reset::ToZero), period(Reset::Kernel));
            assert!((zero - (m.width + (i / (i - 1.0)).ln())).abs() < 1e-12, "{zero}");
            assert!((kernel - m.sync_period(i).unwrap()).abs() < 1e-12, "{kernel}");
            let quoted = if i < 1.5 { (1.5663, 1.5441) } else { (1.5361, 1.3857) };
            assert!((zero - quoted.0).abs() < 5e-5 && (kernel - quoted.1).abs() < 5e-5, "{zero} {kernel}");
            assert!((zero - drawn).abs() < 0.003 && (kernel - drawn).abs() > 0.02, "drawn {drawn}: ToZero {zero}, Kernel {kernel}");
        }
    }

    /// One uncoupled cell, simulated: under [`Reset::Kernel`] it lands at `(I − 1)(1 − e^{−Δ})` after
    /// each spike and fires with eq. (3.9)'s period; under [`Reset::ToZero`] it lands at 0.
    #[test]
    fn one_cell_resets_where_its_reset_rule_says() {
        let m = ChowKopell::FIG1_C;
        for (reset, land) in [(Reset::Kernel, 0.55 * -(-0.5f64).exp_m1()), (Reset::ToZero, 0.0)] {
            let mut net = ChowKopellNet::new(m, vec![1.55], vec![0.9]).unwrap().with_reset(reset);
            let first = net.run(2.0, 0.01).unwrap()[0].time;
            let mut probe = ChowKopellNet::new(m, vec![1.55], vec![0.9]).unwrap().with_reset(reset);
            probe.run(first + m.width + 1e-12, 0.01).unwrap();
            assert!((probe.voltages()[0] - land).abs() < 1e-11, "{reset:?}: {}", probe.voltages()[0]);
        }
    }

    /// A Chow–Kopell pair locks where `G` and `F` say: at the drive `I = 1.244 866` that makes
    /// antiphase's period exactly 2 by eq. (3.5), Fig. 3's spike satisfies Chow's sufficient
    /// conditions for both states (`T_S = 1.7072 > t_min` and `T/2 = 1 > t_min = 0.8457`), and runs
    /// from different starts end in synchrony with period `T_S` of eq. (3.9) or in antiphase with
    /// period 2, to 6.6 × 10⁻¹³ and 1.1 × 10⁻¹² (measured).
    #[test]
    fn a_pair_locks_where_g_and_f_say() {
        let m = ChowKopell::FIG3;
        let i = 1.0 - m.period_function(0.5, 2.0).unwrap();
        assert_eq!(m.antiphase_period(i, 5.0, 500).unwrap().map(|t| (t - 2.0).abs() < 1e-13), Some(true));
        let ts = m.sync_period(i).unwrap();
        let tmin = m.t_min().unwrap();
        assert!(ts > tmin && 1.0 > tmin && m.antiphase_partner_peak(2.0).unwrap() < 1.0);
        let settle = |v0: f64| {
            let mut net = ChowKopellNet::new(m, vec![i, i], vec![v0, 0.0]).unwrap();
            *phases(&net.run(200.0, 0.02).unwrap()).last().unwrap()
        };
        for v0 in [0.05, 0.3, 0.5] {
            let (_, period, lag) = settle(v0);
            assert!((period - ts).abs() < 5e-12 && lag == 0.0, "v0 = {v0}: {period} {lag}");
        }
        let (_, period, lag) = settle(0.7);
        assert!((period - 2.0).abs() < 5e-12 && (lag - 0.5).abs() < 5e-12, "{period} {lag}");
        assert_eq!(m.antiphase_period(i, 1.5, 100).unwrap(), None, "no antiphase period below 1.5");
    }

    /// The network simulation against `SciPy`'s DOP853 integration of eqs. (2.7)–(2.8) and (4.1) with
    /// the same spike currents and resets (reference D): every spike time to the twelve decimals it
    /// prints (worst difference 5.1 × 10⁻¹³), in a pair and in three heterogeneous cells.
    #[test]
    fn the_network_matches_an_independent_integration() {
        let pair = ChowKopellNet::new(ChowKopell::FIG3, vec![1.3, 1.3], vec![0.3, 0.0]).unwrap();
        let three = ChowKopellNet::new(ChowKopell { g: 0.2, ..ChowKopell::FIG3 }, vec![1.2, 1.25, 1.3], vec![0.5, 0.25, 0.0]).unwrap();
        let cases: [(ChowKopellNet, [(f64, usize); 8]); 2] = [
            (pair, [(1.307827248772, 0), (1.366814305513, 1), (2.869832865980, 0), (2.895163150793, 1), (4.421239748978, 0), (4.432219464006, 1), (5.968486766867, 0), (5.973267145564, 1)]),
            (three, [(1.384140473486, 2), (1.386292444012, 1), (1.389484868332, 0), (2.966353356952, 2), (3.054016458782, 1), (3.884279444800, 0), (4.602377982335, 2), (4.930459779416, 1)]),
        ];
        for (mut net, want) in cases {
            let spikes = net.run(6.0, 0.01).unwrap();
            assert!(spikes.len() >= 8);
            for (s, (t, c)) in spikes.iter().zip(want) {
                assert!(s.cell == c && (s.time - t).abs() < 1e-12, "{s:?} against ({t}, {c})");
            }
        }
    }

    /// The simulated voltages are the spike-response form (4.5) — `I` plus each cell's own spikes
    /// through `Γ_s` and every other cell's through `Γ_c` — with the kernels normalised by `1/n`, to
    /// 7.2 × 10⁻¹⁴ (measured) after forty membrane times. ⚠ Eqs. (4.6)–(4.7) print
    /// `Γ_s = η₊ + (n − 1)η₋` and `Γ_c = η₊ − η₋` without the `1/n`: for `n = 2` that is twice the
    /// pair's own `γ_s`, `γ_c` of eqs. (2.19)–(2.20), and the voltages it predicts miss by order 1.
    #[test]
    fn the_voltages_are_the_spike_response_form_with_normalised_kernels() {
        let m = ChowKopell { g: 0.2, ..ChowKopell::FIG3 };
        let n = 3;
        let mut net = ChowKopellNet::new(m, vec![1.3; n], vec![0.5, 0.25, 0.0]).unwrap();
        let spikes = net.run(40.0, 0.01).unwrap();
        let t = net.time();
        let r = m.rate(n);
        for i in 0..n {
            let mut normalised = 1.3;
            let mut printed = 1.3;
            for s in &spikes {
                let (own, other) = (m.network_gamma_s(n, t - s.time).unwrap(), m.network_gamma_c(n, t - s.time).unwrap());
                normalised += if s.cell == i { own } else { other };
                let (e1, er) = (m.eta(t - s.time, 1.0).unwrap(), m.eta(t - s.time, r).unwrap());
                printed += if s.cell == i { e1 + (n as f64 - 1.0) * er } else { e1 - er };
            }
            assert!((net.voltages()[i] - normalised).abs() < 3e-13, "cell {i}: {} against {normalised}", net.voltages()[i]);
            assert!((net.voltages()[i] - printed).abs() > 0.1, "cell {i}: the printed kernels give {printed}");
        }
        for t in [0.05, 0.1, 0.4, 2.0] {
            assert!((m.network_gamma_s(2, t).unwrap() - m.gamma_s(t).unwrap()).abs() < 1e-15);
            assert!((m.network_gamma_c(2, t).unwrap() - m.gamma_c(t).unwrap()).abs() < 1e-15);
        }
        assert_eq!(m.network_gamma_s(1, 0.7).unwrap(), m.eta(0.7, 1.0).unwrap(), "one cell: its own spike is η₊");
    }

    /// The splay state of eq. (4.9): with the kernels normalised, [`ChowKopell::splay_drive`] is the
    /// single-cell condition (3.8) at `n = 1`, the antiphase condition (3.5) at `n = 2`, and for
    /// uncoupled cells (`g = 0`) the single-cell period whatever `n`. Three cells at `g = 0.05` with
    /// `T/3 > t_min` — Chow's sufficient condition — settle into it: spikes `T/3` apart in cyclic
    /// order, to 1.7 × 10⁻¹² (measured).
    ///
    /// ⚠ Eq. (4.12) as printed, `I − 1 = e^Δ/(eᵀ − 1) + e^Δ/(e^{T/n} − 1) + (n − 1)δ_c e^{rΔ}/(e^{rT} − 1)
    /// − δ_c e^{rΔ}/(e^{rT/n} − 1)`, sums `Γ_c` over every multiple of `T/n`, the cell's own firing
    /// times included, with the unnormalised kernels: at `g = 0` it gives `I − 1 = n e^Δ/(eᵀ − 1)`,
    /// `n` times the drive excess an uncoupled cell needs for the same period. At a fixed drive the
    /// period it predicts is `ln(1 + n e^Δ/(I − 1))` against the single cell's `ln(1 + e^Δ/(I − 1))`
    /// — for `n = 3`, `I = 1.3`, `Δ = 0.1`, 2.489 against 1.544 — longer by a factor that tends to
    /// `n` only as `I → ∞`. The correct sum is
    /// `I − 1 = [e^Δ/(e^{T/n} − 1) + nδ_c e^{rΔ}/(e^{rT} − 1) − δ_c e^{rΔ}/(e^{rT/n} − 1)]/n`.
    #[test]
    fn the_splay_state_eq_4_9_and_the_misprint_in_4_12() {
        let m = ChowKopell::FIG3;
        for t in [0.3, 1.0, 2.5] {
            assert!((m.splay_drive(1, t).unwrap() - (1.0 - m.period_function(0.0, t).unwrap())).abs() < 1e-14);
            assert!((m.splay_drive(2, t).unwrap() - (1.0 - m.period_function(0.5, t).unwrap())).abs() < 1e-14);
            let alone = ChowKopell { g: 0.0, ..m };
            for n in 1..6 {
                let single = 1.0 + m.width.exp() / t.exp_m1();
                assert!((alone.splay_drive(n, t).unwrap() - single).abs() < 1e-13, "n = {n}");
                let printed = 1.0 + (n as f64) * m.width.exp() / t.exp_m1();
                assert!(((printed - 1.0) / (single - 1.0) - n as f64).abs() < 1e-12, "the printed (4.12) needs n times the drive");
            }
            for n in [3usize, 5] {
                let (d, r) = (m.width, m.rate(n));
                let dc = 1.0 + m.v_m() - m.v_a / (r + m.xi) * ((m.xi * d).exp() - (-r * d).exp());
                let nf = n as f64;
                let fixed = 1.0 + (d.exp() / (t / nf).exp_m1() + nf * dc * (r * d).exp() / (r * t).exp_m1() - dc * (r * d).exp() / (r * t / nf).exp_m1()) / nf;
                if t / nf > d {
                    assert!((m.splay_drive(n, t).unwrap() - fixed).abs() < 2e-15, "n = {n}, T = {t}");
                }
            }
        }
        // The periods at a fixed drive: the uncoupled splay state fires at the single cell's period,
        // and the printed (4.12) puts it at ln(1 + n e^Δ/(I − 1)).
        let alone = ChowKopell { g: 0.0, ..m };
        let single = bisect(|t| alone.splay_drive(3, t).unwrap() > 1.3, 0.2, 10.0);
        assert!((single - alone.sync_period(1.3).unwrap()).abs() < 1e-14 && (single - 1.544).abs() < 5e-4, "{single}");
        let printed = (3.0 * m.width.exp() / 0.3).ln_1p();
        assert!((printed - 2.489).abs() < 5e-4 && printed / single < 3.0, "{printed}");
        let m = ChowKopell { g: 0.05, ..m };
        let r3 = m.rate(3);
        let dc3 = 1.0 + m.v_m() - m.v_a / (r3 + m.xi) * ((m.xi * m.width).exp() - (-r3 * m.width).exp());
        let tmin3 = m.width + (dc3 * r3).ln() / (r3 - 1.0);
        let t = 3.0 * tmin3 + 0.3;
        let i = m.splay_drive(3, t).unwrap();
        let mut net = ChowKopellNet::new(m, vec![i; 3], vec![0.6, 0.3, 0.0]).unwrap();
        let spikes = net.run(300.0, 0.02).unwrap();
        let tail = &spikes[spikes.len() - 7..];
        for w in tail.windows(2) {
            assert!((w[1].time - w[0].time - t / 3.0).abs() < 1e-11, "{tail:?}");
            assert_eq!(w[1].cell, (w[0].cell + 1) % 3, "cyclic order");
        }
    }

    /// The network is exact between events: a Runge–Kutta integration of eq. (4.1) across a stretch
    /// in which one cell is mid-spike, its current `v_A e^{ξ(t − t₀)}` live, converges on the closed
    /// form at fourth order (error ratios 15.98 and 15.99 on halving, measured).
    #[test]
    fn the_network_is_exact_between_events() {
        let m = ChowKopell { g: 0.05, ..ChowKopell::FIG3 };
        let mut net = ChowKopellNet::new(m, vec![1.3, 1.2, 1.4], vec![0.999, 0.2, 0.5]).unwrap();
        net.run(0.01, 0.001).unwrap();
        let k = net.ends.iter().position(Option::is_some).expect("cell 0 is mid-spike");
        let t0 = net.onsets[k];
        let span = 0.04;
        let exact = net.after(span);
        let field = |t: f64, v: &[f64]| -> Vec<f64> {
            let n = v.len() as f64;
            let sum: f64 = v.iter().sum();
            (0..v.len())
                .map(|i| {
                    let a = if i == k { m.v_a * (m.xi * (net.time() + t - t0)).exp() } else { 0.0 };
                    net.drives[i] - v[i] - m.g * (n * v[i] - sum) + a
                })
                .collect()
        };
        let rk4 = |steps: usize| {
            let h = span / steps as f64;
            let mut v = net.v.clone();
            for s in 0..steps {
                let t = h * s as f64;
                let add = |a: &[f64], b: &[f64], c: f64| a.iter().zip(b).map(|(x, y)| x + c * y).collect::<Vec<f64>>();
                let k1 = field(t, &v);
                let k2 = field(t + h / 2.0, &add(&v, &k1, h / 2.0));
                let k3 = field(t + h / 2.0, &add(&v, &k2, h / 2.0));
                let k4 = field(t + h, &add(&v, &k3, h));
                v = (0..v.len()).map(|i| v[i] + h / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i])).collect();
            }
            v.iter().zip(&exact).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max)
        };
        let (e1, e2, e3) = (rk4(8), rk4(16), rk4(32));
        assert!((e1 / e2 - 16.0).abs() < 0.5 && (e2 / e3 - 16.0).abs() < 0.5, "{e1} {e2} {e3}");
    }

    /// The event budget of the network: sufficient by construction for [`ChowKopellNet::run`], and a
    /// run that outgrows a smaller one says so.
    #[test]
    fn the_network_budget_covers_steps_and_spikes() {
        let m = ChowKopell::FIG3;
        let mut coarse = ChowKopellNet::new(m, vec![1.3, 1.3], vec![0.3, 0.0]).unwrap();
        assert!(coarse.run(100.0, 10.0).unwrap().len() > 100, "many spikes, ten steps");
        let mut fine = ChowKopellNet::new(m, vec![1.3, 1.3], vec![0.3, 0.0]).unwrap();
        assert!(fine.run(1.0, 1e-4).unwrap().is_empty() && (fine.time() - 1.0).abs() < 1e-12, "ten thousand steps, no spike");
        let mut short = ChowKopellNet::new(m, vec![1.3, 1.3], vec![0.3, 0.0]).unwrap();
        let err = short.run_within(10.0, 0.1, 5).unwrap_err();
        assert_eq!(err.to_string(), format!("the run used up its event budget at t = {}; sample more finely", short.time()));
    }

    /// Every refusal of the Chow–Kopell side, each rendered.
    #[test]
    fn every_chow_kopell_refusal_names_what_it_refused() {
        let msg = |r: Result<ChowKopell, GapError>| r.unwrap_err().to_string();
        assert_eq!(msg(ChowKopell::new(-1.0, 50.0, 0.1, 0.5)), "v_A = -1 must be finite and not negative");
        assert_eq!(msg(ChowKopell::new(1.0, 0.0, 0.1, 0.5)), "xi = 0 must be finite and positive");
        assert_eq!(msg(ChowKopell::new(1.0, 50.0, f64::NAN, 0.5)), "Delta = NaN must be finite and positive");
        assert_eq!(msg(ChowKopell::new(1.0, 50.0, 0.1, -0.5)), "g = -0.5 must be finite and not negative");
        assert_eq!(msg(ChowKopell::new(1.0, 1000.0, 1.0, 0.5)), "v_M = inf is not finite");
        assert!(ChowKopell::new(0.0, 50.0, 0.1, 0.0).is_ok(), "no spike current and no junction are allowed");
        let m = ChowKopell::FIG3;
        assert_eq!(m.eta(f64::NAN, 1.0).unwrap_err().to_string(), "t = NaN is not finite");
        assert_eq!(m.eta(0.5, 0.0).unwrap_err().to_string(), "r = 0 must be finite and positive");
        assert_eq!(m.network_gamma_s(0, 0.5).unwrap_err().to_string(), "n = 0 must be at least 1");
        assert_eq!(m.network_gamma_c(0, 0.5).unwrap_err().to_string(), "n = 0 must be at least 1");
        assert_eq!(m.gamma_s(f64::INFINITY).unwrap_err().to_string(), "t = inf is not finite");
        assert_eq!(m.gamma_c(f64::NAN).unwrap_err().to_string(), "t = NaN is not finite");
        assert_eq!(m.locking(0.3, 0.1).unwrap_err().to_string(), "period T = 0.1 must exceed 0.1");
        assert_eq!(m.locking(f64::NAN, 1.0).unwrap_err().to_string(), "phi = NaN is not finite");
        assert_eq!(m.locking_slope(0.3, f64::NAN).unwrap_err().to_string(), "T = NaN is not finite");
        assert_eq!(m.period_function(0.3, 0.05).unwrap_err(), GapError::PeriodTooShort { period: 0.05, limit: 0.1 });
        assert_eq!(m.sync_period(1.0).unwrap_err().to_string(), "drive I = 1 does not exceed the threshold 1, so an uncoupled cell never fires");
        assert_eq!(m.sync_period(f64::NAN).unwrap_err().to_string(), "I = NaN is not finite");
        assert_eq!(
            m.sync_period(20.0).unwrap_err().to_string(),
            format!("drive I = 20 is not below {}, where the synchronous period reaches the spike width", 1.0 + 1.0 / -(-0.1f64).exp_m1())
        );
        assert_eq!(m.antiphase_period(1.3, 0.1, 10).unwrap_err().to_string(), "the interval (0.1, 0.1) is empty");
        assert_eq!(m.antiphase_period(1.3, 3.0, 0).unwrap_err().to_string(), "samples = 0 must be at least 1");
        assert_eq!(m.antiphase_period(f64::NAN, 3.0, 10).unwrap_err().to_string(), "I = NaN is not finite");
        assert_eq!(m.antiphase_period(1.3, f64::INFINITY, 10).unwrap_err().to_string(), "t_max = inf is not finite");
        assert_eq!(m.slope_switches(0.0, 0.1, 1.0, 10).unwrap_err(), GapError::PeriodTooShort { period: 0.1, limit: 0.1 });
        assert_eq!(m.slope_switches(0.0, 0.5, f64::NAN, 10).unwrap_err().to_string(), "high = NaN is not finite");
        assert_eq!(m.slope_switches(f64::NAN, 0.5, 1.0, 10).unwrap_err().to_string(), "phi = NaN is not finite");
        assert_eq!(m.slope_switches(0.0, 0.5, 0.5, 10).unwrap_err().to_string(), "the interval (0.5, 0.5) is empty");
        assert_eq!(m.slope_switches(0.0, 0.5, 1.0, 0).unwrap_err().to_string(), "samples = 0 must be at least 1");
        assert_eq!(ChowKopell { g: 0.0, ..m }.t_min().unwrap_err().to_string(), "g = 0 must be finite and positive");
        assert_eq!(m.antiphase_partner_peak(0.2).unwrap_err().to_string(), "period T = 0.2 must exceed 0.2");
        assert_eq!(m.antiphase_partner_peak(f64::NAN).unwrap_err().to_string(), "T = NaN is not finite");
        assert_eq!(m.splay_drive(0, 1.0).unwrap_err().to_string(), "n = 0 must be at least 1");
        assert_eq!(m.splay_drive(3, 0.1).unwrap_err(), GapError::PeriodTooShort { period: 0.1, limit: 0.1 });
        assert_eq!(m.spike_voltage(1.3, 0.2).unwrap_err().to_string(), "t = 0.2 is outside [0, 0.1]");
        assert_eq!(m.spike_voltage(1.3, -0.01).unwrap_err().to_string(), "t = -0.01 is outside [0, 0.1]");
        assert_eq!(m.spike_voltage(f64::NAN, 0.05).unwrap_err().to_string(), "I = NaN is not finite");
        let bad = ChowKopell { xi: -1.0, ..m };
        for r in [bad.eta(0.5, 1.0), bad.delta_c(), bad.t_min(), bad.locking(0.2, 1.0), bad.sync_period(1.2), bad.spike_voltage(1.2, 0.05)] {
            assert_eq!(r.unwrap_err(), GapError::NotPositive { what: "xi", value: -1.0 });
        }
        assert_eq!(bad.antiphase_period(1.3, 3.0, 10).unwrap_err(), GapError::NotPositive { what: "xi", value: -1.0 });
        assert_eq!(bad.antiphase_partner_peak(1.0).unwrap_err(), GapError::NotPositive { what: "xi", value: -1.0 });
        let net = |d: Vec<f64>, v: Vec<f64>| ChowKopellNet::new(m, d, v).unwrap_err().to_string();
        assert_eq!(net(vec![], vec![]), "cells = 0 must be at least 1");
        assert_eq!(net(vec![1.3, 1.3], vec![0.0, 0.0, 0.0]), "2 drives and 3 starting voltages: one of each per cell");
        assert_eq!(net(vec![1.3, f64::NAN], vec![0.0, 0.0]), "I = NaN is not finite");
        assert_eq!(net(vec![1.3], vec![f64::INFINITY]), "v0 = inf is not finite");
        assert_eq!(net(vec![1.3, 1.3], vec![1.5, 0.0]), "cell 0 starts at v = 1.5, not below the threshold 1");
        assert_eq!(net(vec![1.3, 1.3], vec![0.0, 1.0]), "cell 1 starts at v = 1, not below the threshold 1");
        assert_eq!(ChowKopellNet::new(bad, vec![1.3], vec![0.0]).unwrap_err(), GapError::NotPositive { what: "xi", value: -1.0 });
        let mut ok = ChowKopellNet::new(m, vec![1.3], vec![0.0]).unwrap();
        assert_eq!(ok.run(f64::NAN, 0.1).unwrap_err().to_string(), "duration = NaN must be finite and positive");
        assert_eq!(ok.run(1.0, 0.0).unwrap_err().to_string(), "step = 0 must be finite and positive");
        assert_eq!((ok.time(), ok.voltages()), (0.0, &[0.0][..]));
    }
}
