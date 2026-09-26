//! Morris and Lecar's barnacle muscle fibre: two conductances that do not inactivate, and the
//! plateaus, bistability and oscillations they produce between them — each claim checked against
//! the closed forms of the paper's own equations.
//!
//! # The paper
//!
//! Morris and Lecar, *Voltage oscillations in the barnacle giant muscle fiber*, Biophysical Journal
//! 35:193–213, 1981. A space-clamped patch of sarcolemma is a capacitor in parallel with a leak, a
//! calcium conductance and a potassium conductance (their Fig. 1), and neither conductance
//! inactivates. Eqs. 1 (p. 195) and 2 (p. 196), in their notation:
//!
//! ```text
//! I = C dV/dt + g_L(V − V_L) + g_Ca M (V − V_Ca) + g_K N (V − V_K)            (1)
//! dM/dt = λ_M(V) [M∞(V) − M]          dN/dt = λ_N(V) [N∞(V) − N]
//! M∞(V) = ½{1 + tanh[(V − V1)/V2]}    λ_M(V) = λ̄_M cosh[(V − V1)/2V2]         (2)
//! N∞(V) = ½{1 + tanh[(V − V3)/V4]}    λ_N(V) = λ̄_N cosh[(V − V3)/2V4]
//! ```
//!
//! — [`MorrisLecar::field`], with each gate a [`Gate`]. With one conductance alone it is Eq. 3
//! (p. 201), one gate `μ` in the `(V, μ)` plane: [`SingleConductance`], whose `V̇ = 0` nullcline is
//! the bilinear Eq. 4a running from `V(0) = (I + g_L V_L)/g_L` at `μ = 0` to Eq. 5's
//! `V(1) = (I + g_L V_L + g_i V_i)/(g_L + g_i)` at `μ = 1`, and whose `μ̇ = 0` nullcline is `μ∞(V)`
//! (Eq. 4b). With calcium so much faster than potassium that `M = M∞(V)` throughout, it is Eq. 9
//! (p. 205), the `V, N` reduced system that now carries the authors' names:
//! [`MorrisLecar::reduced_field`].
//!
//! # What is exact
//!
//! Everything about an equilibrium, because the equilibrium curve has a closed form. At rest
//! `M = M∞(V)` and `N = N∞(V)`, so the current that holds the membrane at `V` is
//!
//! ```text
//! I_ss(V) = g_L(V − V_L) + g_Ca M∞(V)(V − V_Ca) + g_K N∞(V)(V − V_K)
//! ```
//!
//! — [`MorrisLecar::steady_current`] — and the equilibria at a current `I` are the roots of
//! `I_ss(V) = I`, bisected to adjacent doubles by [`MorrisLecar::equilibria`]. On that curve:
//!
//! - Eq. 11's `N_s` is the `V̇ = 0` nullcline of Eq. 9 evaluated at `V_s`
//!   ([`MorrisLecar::v_nullcline`]);
//! - the determinant of Eq. 9's Jacobian is `λ_N(V) I_ss'(V)/C` exactly, so the folds at which
//!   equilibria appear in pairs are the extrema of `I_ss` ([`MorrisLecar::folds`]);
//! - Eq. 15's two inequalities are exactly `trace > 0` and `det > 0` of that Jacobian, with margins
//!   `C·trace` and `C·det/λ_N` ([`MorrisLecar::oscillation_margins`]);
//! - a Hopf current is `I_ss` at a zero of the trace, so it costs one root in `V` and one
//!   evaluation ([`MorrisLecar::trace_zeros`]). For the Fig. 9 parameters the two are
//!   `I = 289.651 573 716` and `465.104 978 128 µA/cm²`, at `ω = 0.241 843 5` and
//!   `0.203 453 5 rad/ms`. The paper prints neither number: Fig. 8 draws them as the two ends of its
//!   shaded loop.
//!
//! # Units: the paper's
//!
//! Millivolts, milliseconds, µA/cm², mmho/cm² and µF/cm², inside and at every argument, which is
//! what makes every constant below comparable with its caption. The frame is self-consistent:
//! µF·mV/ms and mmho·mV are both exactly µA. [`crate::neuron::Izhikevich`] keeps its paper's
//! frame INSIDE the model for the same reason and converts to SI at the [`crate::neuron::Neuron`]
//! boundary; this module has no such boundary, because it is a phase-plane analysis of one paper,
//! like [`crate::planar`], and not a spiking neuron — the paper defines no spike, threshold or
//! reset. Every argument names its unit.
//!
//! # What the page prints, and what its own equations need instead
//!
//! Every value below is kept as printed, next to the value that is used and the evidence; nothing is
//! corrected silently.
//!
//! ⚠ **Eq. 1 prints the leak as `g_L(V_L)`**, with no `V −`. Eqs. 3 and 9 print `g_L(V − V_L)`, and
//! Eq. 8 prints `g*_L(V − V_L)`; this module uses `g_L(V − V_L)`. Read as printed, the leak is a
//! constant current, and the all-K membrane of Fig. 2b would rest at −7.19 mV instead of −50.06 —
//! where the figure starts every trace.
//!
//! ⚠ **Eq. 9 prints `C dV̇/dt` and `dṄ/dt`** — a dot inside each derivative — and `g_k`, `V_k` in
//! lower case (p. 205). Read literally, both left sides are second derivatives, which would make
//! Eq. 9 fourth order and not the second-order system the text goes on to study; the text says what
//! it is, Eq. 1 with `M = M∞(V)`, and that is [`MorrisLecar::reduced_field`].
//!
//! ⚠ **The rate constants are listed in s⁻¹ and must be ms⁻¹.** The abbreviation list (p. 194)
//! gives `λ_M`, `λ_N` and their maxima in s⁻¹, but every model time axis except Fig. 3d's is in
//! milliseconds. Read as s⁻¹, Fig. 2b's `λ̄_N = 1/15` all but freezes `N` — from 0.0012 to 0.071
//! in the figure's 200 ms — and the `I = 400` response climbs to +78.9 mV and is still at +59.3 mV
//! at 200 ms, where the figure peaks near +20 mV within 10 ms and settles near −2 mV. Every `λ̄`
//! here is ms⁻¹.
//!
//! ⚠ **Fig. 9 prints `V2 = −15`; Figs. 7–10 need `+15`.** With `−15`, `M∞` falls with
//! depolarisation, and every equilibrium of Eq. 9 for `I` from 0 to 550 µA/cm² has two real negative
//! eigenvalues (largest −0.1139 ms⁻¹): nothing oscillates, and there is no Hopf point. With `+15`
//! the Hopf currents above bound the loop of Fig. 8, and Figs. 7, 9 and 10 follow.
//! [`MorrisLecar::FIG9`] uses `+15`; [`MorrisLecar::FIG9_PRINTED_V2`] keeps the printed value.
//!
//! ⚠ **Fig. 5a prints `V_L = 50`**, without the minus sign. Fig. 2b's caption gives −50 for the same
//! all-K system, and the drawn nullclines need −50: at `V = −50` the `I = 25` and `I = 100`
//! nullclines cross `N = 25/160 = 0.156` and `100/160 = 0.625`, and the `I = 400` nullcline reaches
//! `N = 1` at Eq. 5's `V(1) = −28.2 mV` (with `+50` it would be −0.9 mV).
//!
//! ⚠ **Eq. 11 prints the leak as `g_L(V_s − V_K)`.** From Eq. 9 it is `g_L(V_s − V_L)`; the printed
//! form misses `N∞(V_s)` by `g_L(V_K − V_L)/(g_K(V_s − V_K))`, which at Fig. 7's singular point is
//! 0.072 of an `N` of 0.503.
//!
//! ⚠ **Fig. 5b's caption names the `I = 100` singular point C.** The figure labels it C′, and so
//! does p. 203, whose account of the plateau needs the two apart: C′, at +43.05 mV, is where the
//! membrane goes while the current flows, and C, at +27.49 mV and `I = 0`, is where it stays when the
//! current stops.
//!
//! ⚠ **Fig. 7's "unstable node" is an unstable focus.** Its eigenvalues are
//! `+0.00364 ± 0.24206i` ms⁻¹ — complex, which is Eq. 17 — and the figure draws the spiral a focus
//! makes. `I = 300` is just past the Hopf current 289.65.
//!
//! ⚠ **Fig. 5a's caption calls the all-K singular point "always a stable node".** It is a node only
//! below `I = 2.117` and above 731.6 µA/cm²; in between — at every current the figure draws but 0 —
//! it is a stable focus, with eigenvalues `−0.1481 ± 0.0279i`, `−0.1277 ± 0.0923i` and
//! `−0.2022 ± 0.2099i` ms⁻¹ at `I = 25`, 100 and 400. Fig. 2b is consistent with that but cannot
//! show it. Its `I = 400` trace dips to −3.23 mV at 20.8 ms, below where it settles, and a node could
//! undershoot like that after the nonlinear swing down from its +21.7 mV peak. What only a focus does
//! is cross the final level again after the dip — near the equilibrium a node's
//! `a e^{λ₁t} + b e^{λ₂t}`, started with zero slope, has no later zero — and the computed response
//! does, to a second peak at 36.0 ms just 0.063 mV above its final −1.89 mV: about half a pixel of
//! the page at 600 dpi, where its scale is 9.3 pixels to the millivolt. The eigenvalues are the
//! evidence.
//!
//! ⚠ **Eq. 10's upper bound is a bound only up to `I = g_L(V_Ca − V_L)`.** `V_max` is where `V̇`
//! would vanish with `M = 1`, `N = 0`, but `M = M∞(V) < 1`. At `V_max` with `N = 0`,
//! `C V̇ = g_Ca(1 − M∞)(V_max − V_Ca)`, which is positive once `V_max > V_Ca` — above 300 µA/cm² for
//! the Fig. 9 set, which Fig. 9 draws up to 500. At `I = 400`, `V̇(V_max, 0) = +2.2 × 10⁻⁶` mV/ms.
//! Eq. 10 takes two corners of the `(M, N)` square; the interval that does trap every trajectory
//! takes all four, [`MorrisLecar::trapping_interval`], beside the printed
//! [`MorrisLecar::eq10_bounds`]. The Poincaré–Bendixson argument survives; the printed rectangle
//! does not.
//!
//! ⚠ **Fig. 9's caption says the oscillating currents "coincide" with those at which the nullclines
//! cross on the `V̇ = 0` nullcline's rising, negative-resistance limb. They do not.** At a singular
//! point the slope of Eq. 11's nullcline is `(G − ḡ)/(g_K(V_s − V_K))`, in the notation of Eq. 16,
//! so the limb is where `G > ḡ`; Eq. 15 needs `G > ḡ + Cλ_N`. The crossings are on the limb from
//! `I = 195.64` to 493.89 µA/cm²; Eq. 9 oscillates only from 289.65 to 465.10. The caption's own
//! `I = 200` trace crosses on the limb, at −6.94 mV against the limb's foot at −7.24 mV, and is
//! damped. The Fig. 7 caption has the logic right: the limb is where the crossing "must" be.
//!
//! ⚠ **The reduction to Eq. 9 changes the character of the singular point for the paper's own
//! `λ̄_M`.** The paper justifies Eq. 9 by Tikhonov's theorem, which "allows reduction of the dimension
//! of the phase space without changing the character of the singular point" (p. 206). That holds as
//! `λ̄_M → ∞`. With the `λ̄_M = 0.1` ms⁻¹ the Fig. 9 caption lists — 1.5 times `λ̄_N` — the full Eq. 1
//! has a STABLE equilibrium at every current where Eq. 9 oscillates. The Routh–Hurwitz conditions
//! on the full Jacobian ([`characteristic`], [`routh_hurwitz`]) hold along the whole equilibrium
//! curve from −60 to +60 mV for `λ̄_M` of 0.1, 0.2, 0.5 and 1, so there is no Hopf point — at 1 only
//! just: the least `a₂a₁ − a₀` on the curve is 0.0021 there, against 0.0148 at 0.1. Hopf points
//! appear at `λ̄_M = 1.0207` ms⁻¹, 15.3 times `λ̄_N`, and above it Eq. 1 fails between two Hopf
//! currents that close in on Eq. 9's as `λ̄_M` grows: 316.64 and 448.81 µA/cm² at `λ̄_M = 2`, 293.46
//! and 462.71 at 10. A stable equilibrium can still sit inside a cycle, as Fig. 6's does, so Eq. 1
//! is also run: from `V = −50 mV` with its gates at steady state, the start of the Fig. 10 tests, it
//! settles on its equilibrium at 300, 350, 400 and 450 µA/cm² with the Fig. 9 set and at 300 with
//! each of Fig. 10's four, and draws none of their oscillations. So Figs. 7–10 are properties of
//! Eq. 9, as their captions say, and not of Eq. 1 with the listed `λ̄_M`. The same holds for
//! Fig. 11, whose caption does not name the system: Eq. 1 with its `λ̄_M = 0.1` settles on its
//! equilibrium.
//!
//! ⚠ **Fig. 6's current is not printed, and its line labels are swapped.** Neither the caption nor
//! the text gives `I`. At `I = 50 µA/cm²`, from `V = −50 mV` with both gates at steady state, the
//! set printed for the broken line (`g_Ca = 6`, `g_K = 12`) draws the SOLID trace — sustained, peaks
//! at 30.80, 61.49, 91.03, 120.04, 148.75, 177.25 and 205.62 ms — and the set printed for the solid
//! line (`g_Ca = 4`, `g_K = 8`) draws the BROKEN one, damped from +21.1 mV at 40.8 ms to +9.0 mV at
//! 211.5 ms. The swap does not hang on the choice of 50: at every current from 30 to 120 µA/cm², in
//! steps of 10, the (6, 12) set's first peak comes earlier and higher than the (4, 8) set's, which at
//! 30 does not peak within 220 ms at all. At any of those currents, the set that draws the leading,
//! solid trace is (6, 12); and 50 is the one of them at which the first peaks fall where the figure
//! draws them, near 31 and 40 ms. Nor is the sustained trace what the eigenvalues predict: both
//! equilibria are stable under Eq. 1 at that current, their slowest eigenvalues `−0.0125 ± 0.3506i`
//! and `−0.0129 ± 0.2901i` ms⁻¹, and the solid trace is a limit cycle AROUND a stable equilibrium —
//! started 1 mV from it, the same model decays onto it. The Discussion's "one can vary parameters
//! and use the eigenvalues to predict the nature of the voltage behavior" (p. 211) cannot see that
//! cycle. The equilibrium the cycle surrounds is depolarised, at +7.24 mV, and not rest: what
//! coexists at the one current is a cycle and a depolarised steady state. That is not "the bistable
//! oscillation pattern" the same page counts among what the model cannot produce. P. 200 describes
//! that pattern after the stimulus: "at the end of the stimulus the voltage falls to a slightly
//! lower mean level, but continues to oscillate (usually for not >100-200 ms) and then falls to rest
//! (Fig. 4ai)". Nor does the model draw it at these parameters. Switched to `I = 0` at any of 31
//! moments a millisecond apart from 200 to 230 ms, which covers a whole cycle, the (6, 12) set fires
//! at most one more peak above 0 mV, the latest 26.47 ms after the switch, and is within 1 mV of rest
//! — −49.38 mV, by then its only stable equilibrium — 71.65 ms after it at the latest.
//!
//! ⚠ **Fig. 10b is not a limit cycle — by a hair.** Shifting `N∞` by +13 mV (`V3 = 12`) leaves a
//! stable focus, `−0.000117 ± 0.2672i` ms⁻¹: the oscillation decays with a time constant of 8.6 s,
//! which over the figure's 200 ms is invisible, and a minute from the start it is within 0.001 mV of
//! the equilibrium. The margin is tiny. With `V3 = 12` Eq. 9's upper Hopf current is 299.890 µA/cm²,
//! 0.11 below the figure's 300; at `I = 300` the Hopf point is at `V3 = 11.9933` mV, 0.0067 mV away;
//! and `λ̄_N = 0.066`, a 1% rounding of `1/15`, makes the one equilibrium an unstable focus,
//! `+0.000217 ± 0.2658i` ms⁻¹ — the only equilibrium inside the trapping interval, so
//! Poincaré–Bendixson puts a limit cycle round it. So the caption's "increases the frequency of the
//! oscillations and lowers the amplitude" is right to within its parameters' precision, and over the
//! figure's 200 ms the transient does both: a period of 24.03 ms against the control's 27.70, and a
//! swing of 11.8 mV against 21.5.
//!
//! ⚠ **Fig. 11's period matches `I = 35`, not the printed `I = 50`.** Eq. 9 with the caption's
//! parameters fires every 58.6 ms at 50 µA/cm²; the figure shows seven spikes about 145 ms apart.
//! At 35 µA/cm² the period is 145.4 ms. The onset is a fold — the equilibria at 33.30 µA/cm² merge
//! and vanish, [`MorrisLecar::folds`] — so the period grows without bound as `I` falls towards it,
//! which is why a misprinted digit moves it by a factor of 2.5. The amplitude agrees: digitised at
//! 600 dpi, the drawn troughs sit at about −39.1 mV and the peaks at about +24.9, where Eq. 9's cycle
//! runs from −39.24 to +25.39 mV at 35 µA/cm² and from −35.64 to +27.35 at 50.
//!
//! ⚠ **Both Fig. 12 maps are sketches.** In Fig. 12a (`I = 300`) the drawn limit-cycle band begins
//! near `g_Ca = 9`, and the Fig. 9 control (`g_K = 8`, `g_Ca = 4`) sits inside the band drawn as
//! damped, about 0.7 mmho/cm² above its lower edge (digitised at 600 dpi) — yet it is the unstable
//! focus of Fig. 7. Eq. 9 is unstable lower still: at `(g_K, g_Ca) = (5, 3)`, unshaded on the page
//! below the tip of the band, it is an unstable focus, `+0.0108 ± 0.1609i` ms⁻¹, against the
//! caption's "All such values fall in the crosshatched region". And the region drawn "NODE (K
//! RESTING)", of which the caption says the "unshaded regions have single equilibrium points which
//! are stable nodes", is a stable FOCUS at all 72 integer points from `g_K = 9` to 20 that lie at
//! least 0.5 mmho/cm² under the band — `−0.1649 ± 0.2850i` ms⁻¹ at (20, 1), `−0.0935 ± 0.3106i` at
//! (20, 5): damped oscillation that the map leaves unshaded.
//!
//! Fig. 12b (`I = 0`) draws its edge from `g_Ca ≈ 1.6` on the `g_K = 0` axis to `g_Ca = 20` near
//! `g_K = 14.4`; Eq. 9's three singular points begin at `g_Ca = 2.211` on that axis and end at
//! `g_K = 13.227` along the top. Digitised at 600 dpi, the drawn edge crosses the computed one near
//! `g_Ca = 5.3` and again near 15.7: between the crossings it runs up to 0.3 mmho/cm² to the left,
//! outside them to the right, by 0.9 to 1.0 at `g_Ca = 19.5`. Nor are the three always "two stable
//! nodes and a saddle point": at `g_K = 13`, `g_Ca = 20` the upper one is a stable FOCUS,
//! `−0.1504 ± 0.1515i` ms⁻¹, and for `g_Ca` above 5.644, where the fold meets a zero of the trace, a
//! sliver along the edge of the three-point region — 0.027 mmho/cm² wide at `g_Ca = 13`, 0.065 at
//! 19.5 — has it UNSTABLE. At `g_K = 12.8`, `g_Ca = 19.5`, about 1.0 mmho/cm² inside the drawn edge
//! (whose centre there is at `g_K ≈ 13.8`), the three are a stable node at −49.65 mV, a saddle at
//! 17.02 mV and an unstable focus at 20.70 mV (`+0.01365 ± 0.1463i` ms⁻¹), and a trajectory leaving
//! the focus ends at rest: one stable state, not two. The caption's "the system cannot oscillate"
//! holds there; its "bistable" does not.
//!
//! ⚠ **Eqs. 6–8, the nonlinear calcium current, are not implemented as printed.** Eq. 6 prints
//! `I_Ca = −g*_Ca M R`, and Eq. 7's `R` is negative wherever `[Ca]_i ≪ [Ca]_o`, so that current is
//! OUTWARD — it cannot hold the calcium plateau of Figs. 3b and 5b. [`Drive::Electrodiffusion`]
//! enters as `+g*_Ca M R` in Eq. 1's outward-positive convention: inward, and it reproduces Fig. 5b's
//! singular points A, B, C and C′ and Fig. 3b's plateaus. Eq. 8 is not implemented at all. Its
//! voltage equation multiplies `R` by an extra `V` (units of mmho·mV²/cm²), adds the calcium term
//! with a sign that makes it hyperpolarising above 0 mV, and uses a `g*_L` that is defined nowhere:
//! at Fig. 3d's start, `V = +28` mV with `[Ca]_i = 0.001` mM, it gives `dV/dt = −177.6` mV/ms, where
//! the inward Eq. 3 form gives −0.18 and the figure declines over seconds. Its accumulation law
//! `d[Ca]_i/dt ≃ K(CF)⁻¹ g*_Ca M∞ V R` does not close dimensionally: with `K` in cm⁻¹ as listed and
//! `C` the membrane capacitance, it comes out in mV²/(cm·ms) if `F` is a bare number and in
//! mV²·mol/(C·cm·ms) if it is in C/mol — neither a concentration per unit time — so no reading of
//! it is the paper's. One reading reproduces Fig. 3d: `d[Ca]_i/dt = K g*_Ca M∞(V) R` in mM/ms with
//! the caption's `K = −10⁻⁴`, which is `−1/(F × 1 µm)` in mM/ms per µA/cm² to 3.6% — the text's
//! "compartment of 1 µm", without the valence 2. It passes +16.00, +9.70 and +4.40 mV at 1, 2 and
//! 3 s and falls through −20 mV at 3.677 s. Four separate readings of the page, digitised at 600 dpi
//! against the figure's own tick marks (13.24 to 13.26 pixels to the millivolt, 230.0 to 230.2 to the
//! second), put the centre of the drawn trace at +15.96 to +16.06, +9.70 to +9.77 and +4.44 to
//! +4.52 mV at those times and its fall through −20 mV at 3.668 to 3.676 s. The stroke is 6 pixels
//! wide, so the page is good to about ±0.3 mV and ±0.03 s, and the model is inside that against
//! every reading. The tests hold that reading of the accumulation law; the module does not implement
//! it.
//!
//! # What is checked
//!
//! The tests hold the paper's printed numbers where the paper is self-consistent, and reference
//! numbers computed outside the crate by `tools/morrislecar_reference.py` — `SciPy` 1.13's DOP853 at
//! relative and absolute tolerance `10⁻¹²`, `brentq` roots and `NumPy` eigenvalues, on the same
//! equations written again from the paper: Fig. 2b's all-K responses, Fig. 3b's plateaus and
//! Fig. 5b's singular points, Fig. 6's peak times, its first peaks from 30 to 120 µA/cm² and its
//! fall to rest when the current is switched off, the Fig. 8 root locus and its Hopf currents, the
//! Fig. 10 and Fig. 11 periods and Fig. 11's amplitude, Fig. 10b's distance from its Hopf point,
//! Eq. 1's Hopf threshold in `λ̄_M` and its runs at the Fig. 9 and Fig. 10 parameters, and the
//! Fig. 12 maps at the points and edges named above. Figs. 5a, 7, 8 and 9 are checked at points
//! only — Fig. 5a's nullclines where they cross `V = −50` and `N = 1`, Fig. 8's locus at eight
//! currents and where it meets the real axis, Figs. 7 and 9 at their singular points — and none of
//! them is redrawn: not the nullclines of Figs. 5a and 9b, nor Fig. 7's spiral, nor Fig. 9a's
//! traces. Page readings are digitised from the published PDF rendered at 600 dpi, against each
//! figure's own axes and tick marks. The paper's own integrator (MLAB,
//! p. 195) is not available, and no code by the authors is known to this review. The closed forms
//! above are checked as identities, every Jacobian against central differences of its field, and
//! the integrator for fourth order. Each of the seventeen printed defects above has a test that
//! shows it, except the two that are notation — Eq. 9's dotted derivatives and Fig. 5b's C for C′
//! — which are read off the page; for the second, the tests hold the two points apart, C at
//! +27.49 mV and C′ at +43.05.

use core::fmt;

use crate::planar::Linearisation;

/// Eq. 7's voltage constant: the `12.5` in `exp(V/12.5)`, millivolts.
///
/// Printed as a bare number (p. 203). `RT/2F` at the experiments' 22 °C (p. 195) is 12.72 mV, so it
/// is read as millivolts.
pub const EQ7_SCALE: f64 = 12.5;

/// How far past a closed-form search window the equilibrium scans reach, millivolts.
///
/// Every equilibrium lies inside the window in exact arithmetic; the margin keeps one that lies
/// within rounding of an end bracketed, and gives a window of zero width — a passive membrane, whose
/// one equilibrium is `V_L + I/g_L` exactly — a width to bracket in.
const PAD: f64 = 1.0;

/// Why a Morris–Lecar question could not be answered.
#[derive(Debug, Clone, PartialEq)]
pub enum MorrisLecarError {
    /// A parameter or step that must be finite and positive was not.
    NotPositive {
        /// Which quantity.
        what: &'static str,
        /// Its value.
        value: f64,
    },
    /// A conductance that must be finite and non-negative was not.
    Negative {
        /// Which conductance.
        what: &'static str,
        /// Its value.
        value: f64,
    },
    /// A potential, state or current that is not a finite number.
    NonFinite {
        /// Which quantity.
        what: &'static str,
        /// Its value.
        value: f64,
    },
    /// A gate slope (`V2` or `V4`) of zero, which Eq. 2 divides by.
    Zero {
        /// Which slope.
        what: &'static str,
    },
    /// A value outside the range its meaning allows.
    OutOfRange {
        /// Which quantity.
        what: &'static str,
        /// Its value.
        value: f64,
        /// The least value allowed.
        lo: f64,
        /// The greatest value allowed.
        hi: f64,
    },
    /// A root search over no samples.
    NoSamples,
    /// A search window whose ends are not finite and increasing.
    EmptyWindow {
        /// The lower end, millivolts.
        lo: f64,
        /// The upper end, millivolts.
        hi: f64,
    },
    /// Calcium concentrations outside `0 ≤ [Ca]_i < [Ca]_o`, where Eq. 7's driving force is monotone
    /// and the equilibrium search is complete.
    Concentrations {
        /// `[Ca]_i`, mM.
        ca_in: f64,
        /// `[Ca]_o`, mM.
        ca_out: f64,
    },
    /// Eq. 4a asked of an electrodiffusion drive, for which the nullcline has no closed form `V(μ)`.
    NotLinear,
}

impl fmt::Display for MorrisLecarError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotPositive { what, value } => write!(f, "{what} = {value} must be finite and positive"),
            Self::Negative { what, value } => write!(f, "{what} = {value} must be finite and non-negative"),
            Self::NonFinite { what, value } => write!(f, "{what} = {value} is not finite"),
            Self::Zero { what } => write!(f, "{what} = 0 is a slope Eq. 2 divides by"),
            Self::OutOfRange { what, value, lo, hi } => write!(f, "{what} = {value} is outside [{lo}, {hi}]"),
            Self::NoSamples => write!(f, "a search over 0 samples brackets nothing"),
            Self::EmptyWindow { lo, hi } => write!(f, "the window [{lo}, {hi}] mV is not a finite, increasing interval"),
            Self::Concentrations { ca_in, ca_out } => write!(
                f,
                "[Ca]_i = {ca_in} mM, [Ca]_o = {ca_out} mM: Eq. 7 is used here only for 0 <= [Ca]_i < [Ca]_o, where its drive is monotone"
            ),
            Self::NotLinear => write!(f, "Eq. 4a is the nullcline of Eq. 3's linear drive; an electrodiffusion drive has no closed-form V(mu)"),
        }
    }
}

impl std::error::Error for MorrisLecarError {}

fn finite(what: &'static str, value: f64) -> Result<f64, MorrisLecarError> {
    if value.is_finite() { Ok(value) } else { Err(MorrisLecarError::NonFinite { what, value }) }
}

fn positive(what: &'static str, value: f64) -> Result<f64, MorrisLecarError> {
    if value.is_finite() && value > 0.0 { Ok(value) } else { Err(MorrisLecarError::NotPositive { what, value }) }
}

fn non_negative(what: &'static str, value: f64) -> Result<f64, MorrisLecarError> {
    if value.is_finite() && value >= 0.0 { Ok(value) } else { Err(MorrisLecarError::Negative { what, value }) }
}

/// One classical fourth-order Runge–Kutta step of `dx/dt = f(x)`.
fn rk4<const D: usize>(f: impl Fn([f64; D]) -> [f64; D], x: [f64; D], h: f64) -> [f64; D] {
    let at = |k: [f64; D], s: f64| -> [f64; D] { core::array::from_fn(|j| x[j] + s * k[j]) };
    let k1 = f(x);
    let k2 = f(at(k1, 0.5 * h));
    let k3 = f(at(k2, 0.5 * h));
    let k4 = f(at(k3, h));
    core::array::from_fn(|j| x[j] + h / 6.0 * (k1[j] + 2.0 * k2[j] + 2.0 * k3[j] + k4[j]))
}

/// Every sign change of `f` between consecutive points of an even grid of `samples` intervals on
/// `[lo, hi]`, each bisected until the bracket is two adjacent doubles.
///
/// Two roots closer than one grid interval are missed; the callers' tests check the COUNT against the
/// paper's figures.
fn roots(f: impl Fn(f64) -> f64, lo: f64, hi: f64, samples: usize) -> Result<Vec<f64>, MorrisLecarError> {
    if samples == 0 {
        return Err(MorrisLecarError::NoSamples);
    }
    if !(lo.is_finite() && hi.is_finite() && lo < hi) {
        return Err(MorrisLecarError::EmptyWindow { lo, hi });
    }
    let step = (hi - lo) / samples as f64;
    let mut out = Vec::new();
    let (mut a, mut fa) = (lo, f(lo));
    for k in 1..=samples {
        let b = lo + step * k as f64;
        let fb = f(b);
        if (fa < 0.0) != (fb < 0.0) {
            let (mut x0, mut x1, below) = (a, b, fa < 0.0);
            // A bracket of a millivolt or less is two adjacent doubles long before 200 halvings.
            for _ in 0..200 {
                let mid = 0.5 * (x0 + x1);
                if (f(mid) < 0.0) == below {
                    x0 = mid;
                } else {
                    x1 = mid;
                }
            }
            out.push(0.5 * (x0 + x1));
        }
        (a, fa) = (b, fb);
    }
    Ok(out)
}

/// One gate of Eq. 2: a steady state that is a `tanh` of voltage and a rate that is a `cosh`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Gate {
    /// `V1` (calcium) or `V3` (potassium): the potential at which the steady state is one half, mV.
    pub v_half: f64,
    /// `V2` or `V4`: the reciprocal slope of the steady state, mV. Non-zero; negative turns the
    /// steady state around, which is what Fig. 9's printed `V2 = −15` does.
    pub slope: f64,
    /// `λ̄_M` or `λ̄_N`: the rate at `v_half`, ms⁻¹ (the abbreviation list's s⁻¹ is a misprint; see
    /// the module doc).
    pub rate: f64,
}

impl Gate {
    /// The steady-state open fraction, `½{1 + tanh[(V − V_half)/V_slope]}`.
    #[must_use]
    pub fn steady(&self, v: f64) -> f64 {
        0.5 * (1.0 + ((v - self.v_half) / self.slope).tanh())
    }

    /// Its derivative in `V`, `sech²[(V − V_half)/V_slope]/(2 V_slope)`, per mV.
    #[must_use]
    pub fn steady_slope(&self, v: f64) -> f64 {
        let c = ((v - self.v_half) / self.slope).cosh();
        0.5 / (self.slope * c * c)
    }

    /// The rate constant `λ̄ cosh[(V − V_half)/2V_slope]`, ms⁻¹.
    #[must_use]
    pub fn rate_at(&self, v: f64) -> f64 {
        self.rate * ((v - self.v_half) / (2.0 * self.slope)).cosh()
    }

    /// Its derivative in `V`, `λ̄ sinh[(V − V_half)/2V_slope]/(2 V_slope)`, per ms per mV.
    #[must_use]
    pub fn rate_slope(&self, v: f64) -> f64 {
        self.rate * ((v - self.v_half) / (2.0 * self.slope)).sinh() / (2.0 * self.slope)
    }

    /// The gate's equation, `dx/dt = λ(V)[x∞(V) − x]`, per ms.
    #[must_use]
    pub fn relax(&self, v: f64, x: f64) -> f64 {
        self.rate_at(v) * (self.steady(v) - x)
    }

    /// `∂/∂V` of [`Gate::relax`] at fixed `x`: `λ'(V)[x∞(V) − x] + λ(V) x∞'(V)`, per ms per mV.
    #[must_use]
    pub fn relax_slope(&self, v: f64, x: f64) -> f64 {
        self.rate_slope(v) * (self.steady(v) - x) + self.rate_at(v) * self.steady_slope(v)
    }

    fn check(&self, names: [&'static str; 3]) -> Result<(), MorrisLecarError> {
        finite(names[0], self.v_half)?;
        if finite(names[1], self.slope)? == 0.0 {
            return Err(MorrisLecarError::Zero { what: names[1] });
        }
        positive(names[2], self.rate)?;
        Ok(())
    }
}

/// `x/(eˣ − 1)`, continuous through its removable singularity at `x = 0`, where it is 1.
fn bernoulli(x: f64) -> f64 {
    if x == 0.0 { 1.0 } else { x / x.exp_m1() }
}

/// The derivative of [`bernoulli`], `B(1 − B − x)/x`, which is −½ at `x = 0`.
fn bernoulli_slope(x: f64) -> f64 {
    if x == 0.0 {
        -0.5
    } else {
        let b = bernoulli(x);
        b * (1.0 - b - x) / x
    }
}

/// Eq. 7's electrodiffusion driving force `R`, millivolts, for `ratio = [Ca]_i/[Ca]_o`.
///
/// Printed as `R = V{1 − ([Ca]_i/[Ca]_o) exp(V/12.5)}/[1 − exp(V/12.5)]`, which is `0/0` at `V = 0`.
/// Written with `B(x) = x/(eˣ − 1)` and `x = V/12.5` it is the same function,
///
/// ```text
/// R(V) = ρV − (1 − ρ)·12.5·B(V/12.5),      ρ = [Ca]_i/[Ca]_o
/// ```
///
/// continuous through `V = 0`, where it is `−12.5(1 − ρ)`, and free of overflow at any finite `V`.
/// It vanishes at the reversal potential `12.5 ln(1/ρ)` and is negative below it. The tests check it
/// against the printed form.
#[must_use]
pub fn electrodiffusion(v: f64, ratio: f64) -> f64 {
    ratio * v - (1.0 - ratio) * EQ7_SCALE * bernoulli(v / EQ7_SCALE)
}

/// `dR/dV` of [`electrodiffusion`]: `ρ − (1 − ρ)B'(V/12.5)`, dimensionless, and between `ρ` and 1.
///
/// At `V = 0` it is `(1 + ρ)/2`. Within a few microvolts of zero, `B'` is computed with a
/// cancellation that costs about `4ε·12.5/|V|` of relative accuracy — `10⁻¹¹` at `|V| = 1.25 µV`.
#[must_use]
pub fn electrodiffusion_slope(v: f64, ratio: f64) -> f64 {
    ratio - (1.0 - ratio) * bernoulli_slope(v / EQ7_SCALE)
}

/// What a single conductance's current is proportional to: its driving force.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Drive {
    /// Eq. 3's linear driving force `V − V_i`, reversing at `v_rev` millivolts.
    Linear {
        /// `V_i`: `V_K` for the all-K system, `V_Ca` for the all-Ca one, mV.
        v_rev: f64,
    },
    /// Eq. 7's electrodiffusion driving force `R(V, [Ca]_i, [Ca]_o)`, entered with the sign that
    /// makes the calcium current INWARD (see the module doc on Eq. 6).
    Electrodiffusion {
        /// `[Ca]_i`, mM.
        ca_in: f64,
        /// `[Ca]_o`, mM.
        ca_out: f64,
    },
}

impl Drive {
    /// The driving force at `v`, mV.
    #[must_use]
    pub fn at(&self, v: f64) -> f64 {
        match *self {
            Self::Linear { v_rev } => v - v_rev,
            Self::Electrodiffusion { ca_in, ca_out } => electrodiffusion(v, ca_in / ca_out),
        }
    }

    /// Its derivative in `V`, dimensionless: 1 for the linear drive.
    #[must_use]
    pub fn slope(&self, v: f64) -> f64 {
        match *self {
            Self::Linear { .. } => 1.0,
            Self::Electrodiffusion { ca_in, ca_out } => electrodiffusion_slope(v, ca_in / ca_out),
        }
    }

    fn check(&self) -> Result<(), MorrisLecarError> {
        match *self {
            Self::Linear { v_rev } => finite("V_i", v_rev).map(|_| ()),
            Self::Electrodiffusion { ca_in, ca_out } => {
                if ca_out.is_finite() && 0.0 <= ca_in && ca_in < ca_out {
                    Ok(())
                } else {
                    Err(MorrisLecarError::Concentrations { ca_in, ca_out })
                }
            }
        }
    }
}

/// What kind of equilibrium a planar linearisation makes, from its trace and determinant.
///
/// Eq. 13's roots are complex exactly when `trace² < 4 det` (Eq. 17), and a planar equilibrium is
/// then a focus; with `det < 0` it is a saddle; otherwise a node. The two exact boundaries a
/// constructed matrix can reach are named rather than folded into a neighbour.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// Two real negative eigenvalues.
    StableNode,
    /// Complex eigenvalues with negative real part: damped oscillation (Eq. 17).
    StableFocus,
    /// Complex eigenvalues with positive real part: Poincaré–Bendixson's unstable point.
    UnstableFocus,
    /// Two real positive eigenvalues.
    UnstableNode,
    /// Real eigenvalues of opposite sign.
    Saddle,
    /// Purely imaginary eigenvalues: `trace = 0 < det`, a Hopf point.
    Centre,
    /// A zero eigenvalue: `det = 0`, a fold.
    Degenerate,
}

impl Kind {
    /// The kind of `l`'s equilibrium.
    #[must_use]
    pub fn of(l: &Linearisation) -> Self {
        let (t, d) = (l.trace(), l.determinant());
        if d < 0.0 {
            Self::Saddle
        } else if d == 0.0 {
            Self::Degenerate
        } else if t == 0.0 {
            Self::Centre
        } else if t * t < 4.0 * d {
            if t < 0.0 { Self::StableFocus } else { Self::UnstableFocus }
        } else if t < 0.0 {
            Self::StableNode
        } else {
            Self::UnstableNode
        }
    }
}

/// An equilibrium of Eq. 3 or Eq. 9.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Equilibrium {
    /// The membrane potential, mV.
    pub v: f64,
    /// The gate there, at its steady state: `μ∞(V)` for Eq. 3, `N∞(V)` for Eq. 9.
    pub x: f64,
    /// Its kind, from the planar linearisation.
    pub kind: Kind,
}

/// A point of Eq. 9's equilibrium curve, with the linearisation's trace and determinant there.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OnCurve {
    /// The equilibrium potential, mV.
    pub v: f64,
    /// The current that holds it, `I_ss(V)`, µA/cm².
    pub i: f64,
    /// The trace of Eq. 9's Jacobian there, ms⁻¹.
    pub trace: f64,
    /// Its determinant, ms⁻². At a zero of the trace, positive is a Hopf point with frequency
    /// `√det` rad/ms; negative is a neutral saddle.
    pub det: f64,
}

/// Eq. 3: one voltage-dependent conductance with its gate `μ`, and a leak.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SingleConductance {
    /// Membrane capacitance `C`, µF/cm².
    pub c: f64,
    /// Leak conductance `g_L`, mmho/cm². Positive.
    pub g_l: f64,
    /// Leak reversal potential `V_L`, mV.
    pub v_l: f64,
    /// The conductance `g_i` — or `g*_Ca`, "the conductance constant for nonlinear `I_Ca`" — mmho/cm².
    pub g: f64,
    /// Its driving force.
    pub drive: Drive,
    /// Its gate.
    pub gate: Gate,
}

impl SingleConductance {
    /// The all-K system of Fig. 2b (caption, p. 197): `g_K = 8`, `g_L = 3`, `V_K = −70`,
    /// `V_L = −50`, `λ̄_N = 1/15`, `C = 20`, `V3 = −1.0`, `V4 = 14.5`, and `g_Ca = 0`.
    ///
    /// Also Fig. 5a's, whose caption prints `V_L` as +50: [`SingleConductance::FIG5A_PRINTED_V_L`].
    pub const FIG2: Self = Self {
        c: 20.0,
        g_l: 3.0,
        v_l: -50.0,
        g: 8.0,
        drive: Drive::Linear { v_rev: -70.0 },
        gate: Gate { v_half: -1.0, slope: 14.5, rate: 1.0 / 15.0 },
    };

    /// `V_L` as the Fig. 5a caption prints it, mV. The figure is drawn with −50.
    pub const FIG5A_PRINTED_V_L: f64 = 50.0;

    /// The all-Ca system of Fig. 3b (caption, p. 198) with Eq. 7's driving force: `g_K = 0`,
    /// `g*_Ca = 40`, `g_L = 2`, `[Ca]_i = 0`, `[Ca]_o = 100`, `V_L = −35`, `λ̄_M = 0.1`, `C = 20`,
    /// `V1 = 10`, `V2 = 15`. Fig. 5b uses it too.
    pub const FIG3B: Self = Self {
        c: 20.0,
        g_l: 2.0,
        v_l: -35.0,
        g: 40.0,
        drive: Drive::Electrodiffusion { ca_in: 0.0, ca_out: 100.0 },
        gate: Gate { v_half: 10.0, slope: 15.0, rate: 0.1 },
    };

    /// Every parameter finite, `C` and `g_L` positive, `g` non-negative, `V_slope` non-zero,
    /// `λ̄` positive, and the drive's own conditions.
    ///
    /// # Errors
    ///
    /// [`MorrisLecarError::NotPositive`], [`MorrisLecarError::Negative`],
    /// [`MorrisLecarError::NonFinite`], [`MorrisLecarError::Zero`] or
    /// [`MorrisLecarError::Concentrations`], naming the first parameter that fails.
    pub fn check(&self) -> Result<(), MorrisLecarError> {
        positive("C", self.c)?;
        positive("g_L", self.g_l)?;
        finite("V_L", self.v_l)?;
        non_negative("g", self.g)?;
        self.drive.check()?;
        self.gate.check(["V_half", "V_slope", "lambda_bar"])
    }

    /// Eq. 3's vector field `(dV/dt, dμ/dt)` at `(V, μ)` under current `I` µA/cm², in mV/ms and
    /// ms⁻¹. Not checked: see [`SingleConductance::step`].
    #[must_use]
    pub fn field(&self, v: f64, mu: f64, i: f64) -> (f64, f64) {
        ((i - self.g_l * (v - self.v_l) - self.g * mu * self.drive.at(v)) / self.c, self.gate.relax(v, mu))
    }

    /// The current that holds the membrane at rest at `V`, µA/cm²: `g_L(V − V_L) + g μ∞(V) D(V)`.
    #[must_use]
    pub fn steady_current(&self, v: f64) -> f64 {
        self.g_l * (v - self.v_l) + self.g * self.gate.steady(v) * self.drive.at(v)
    }

    /// Its derivative in `V`, mmho/cm².
    #[must_use]
    pub fn steady_current_slope(&self, v: f64) -> f64 {
        self.g_l + self.g * (self.gate.steady_slope(v) * self.drive.at(v) + self.gate.steady(v) * self.drive.slope(v))
    }

    /// Eq. 4a, the `V̇ = 0` nullcline as the paper writes it: `V(μ) = (I + g_L V_L + g μ V_i)/(g_L + g μ)`.
    ///
    /// Bilinear in `μ`; `μ = 0` gives `V(0) = (I + g_L V_L)/g_L` and `μ = 1` gives Eq. 5.
    ///
    /// # Errors
    ///
    /// [`MorrisLecarError::NotLinear`] for an electrodiffusion drive;
    /// [`MorrisLecarError::OutOfRange`] for a `μ` outside `[0, 1]`; [`MorrisLecarError::NonFinite`]
    /// for a `μ` or `I` that is not finite; whatever [`SingleConductance::check`] refuses.
    pub fn v_nullcline(&self, mu: f64, i: f64) -> Result<f64, MorrisLecarError> {
        self.check()?;
        finite("I", i)?;
        if !(0.0..=1.0).contains(&finite("mu", mu)?) {
            return Err(MorrisLecarError::OutOfRange { what: "mu", value: mu, lo: 0.0, hi: 1.0 });
        }
        match self.drive {
            Drive::Linear { v_rev } => Ok((i + self.g_l * self.v_l + self.g * mu * v_rev) / (self.g_l + self.g * mu)),
            Drive::Electrodiffusion { .. } => Err(MorrisLecarError::NotLinear),
        }
    }

    /// The same nullcline as `μ` against `V` — how Figs. 5a and 5b draw it:
    /// `μ = (I − g_L(V − V_L))/(g D(V))`. `None` where `g D(V)` is zero. Not checked.
    #[must_use]
    pub fn mu_nullcline(&self, v: f64, i: f64) -> Option<f64> {
        let gd = self.g * self.drive.at(v);
        if gd == 0.0 { None } else { Some((i - self.g_l * (v - self.v_l)) / gd) }
    }

    /// The Jacobian of Eq. 3 at `(V, μ)`, in the paper's per-ms units. It does not depend on `I`.
    #[must_use]
    pub fn linearisation(&self, v: f64, mu: f64) -> Linearisation {
        Linearisation {
            jacobian: [
                [-(self.g_l + self.g * mu * self.drive.slope(v)) / self.c, -self.g * self.drive.at(v) / self.c],
                [self.gate.relax_slope(v, mu), -self.gate.rate_at(v)],
            ],
        }
    }

    /// Every equilibrium at current `I`, found on a grid of `samples` intervals and bisected.
    ///
    /// All of them lie within `g|D(V0)|/g_L` of `V0 = V_L + I/g_L`, the `μ = 0` end of the
    /// nullcline. At an equilibrium `g_L(V − V_L) − I = −μ g D(V)` with `0 < μ < 1`; the left side
    /// rises with slope `g_L` and `−g D` falls (both drives are increasing in `V`), so every root
    /// lies between `V0` and the `μ = 1` point, which is at most `g|D(V0)|/g_L` from `V0`. The window
    /// is that interval widened by 1 mV at each end.
    ///
    /// # Errors
    ///
    /// [`MorrisLecarError::NoSamples`] for `samples = 0`; [`MorrisLecarError::NonFinite`] for an `I`
    /// that is not finite; whatever [`SingleConductance::check`] refuses.
    pub fn equilibria(&self, i: f64, samples: usize) -> Result<Vec<Equilibrium>, MorrisLecarError> {
        self.check()?;
        finite("I", i)?;
        let v0 = self.v_l + i / self.g_l;
        let w = self.g * self.drive.at(v0).abs() / self.g_l + PAD;
        let vs = roots(|v| self.steady_current(v) - i, v0 - w, v0 + w, samples)?;
        Ok(vs
            .into_iter()
            .map(|v| {
                let mu = self.gate.steady(v);
                Equilibrium { v, x: mu, kind: Kind::of(&self.linearisation(v, mu)) }
            })
            .collect())
    }

    /// One fourth-order Runge–Kutta step of `h` ms from `(V, μ)` under constant `I`.
    ///
    /// # Errors
    ///
    /// [`MorrisLecarError::NotPositive`] for a step that is not finite and positive;
    /// [`MorrisLecarError::NonFinite`] for a state or current that is not finite; whatever
    /// [`SingleConductance::check`] refuses.
    pub fn step(&self, v: f64, mu: f64, i: f64, h: f64) -> Result<(f64, f64), MorrisLecarError> {
        self.check()?;
        positive("h", h)?;
        finite("V", v)?;
        finite("mu", mu)?;
        finite("I", i)?;
        let [v1, mu1] = rk4(
            |[v, mu]| {
                let (dv, dmu) = self.field(v, mu, i);
                [dv, dmu]
            },
            [v, mu],
            h,
        );
        Ok((v1, mu1))
    }
}

/// Eq. 1, the full `(V, M, N)` system, and Eq. 9, its `V, N` reduction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MorrisLecar {
    /// Membrane capacitance `C`, µF/cm².
    pub c: f64,
    /// Leak conductance `g_L`, mmho/cm². Positive.
    pub g_l: f64,
    /// Maximum calcium conductance `g_Ca`, mmho/cm². Non-negative.
    pub g_ca: f64,
    /// Maximum potassium conductance `g_K`, mmho/cm². Non-negative.
    pub g_k: f64,
    /// Leak reversal potential `V_L`, mV.
    pub v_l: f64,
    /// Calcium reversal potential `V_Ca`, mV.
    pub v_ca: f64,
    /// Potassium reversal potential `V_K`, mV.
    pub v_k: f64,
    /// The calcium gate `M`: `V1`, `V2`, `λ̄_M`.
    pub m: Gate,
    /// The potassium gate `N`: `V3`, `V4`, `λ̄_N`.
    pub n: Gate,
}

impl MorrisLecar {
    /// Fig. 9 (caption, p. 208), the set Figs. 7, 8, 10 and 12 vary: `g_L = 2`, `g_Ca = 4`,
    /// `g_K = 8`, `V_L = −50`, `V_Ca = 100`, `V_K = −70`, `λ̄_M = 0.1`, `λ̄_N = 1/15`, `V1 = 10`,
    /// `V3 = −1`, `V4 = 14.5`, `C = 20` — and `V2 = +15`, where the caption prints −15
    /// ([`MorrisLecar::FIG9_PRINTED_V2`]; the module doc has the evidence).
    pub const FIG9: Self = Self {
        c: 20.0,
        g_l: 2.0,
        g_ca: 4.0,
        g_k: 8.0,
        v_l: -50.0,
        v_ca: 100.0,
        v_k: -70.0,
        m: Gate { v_half: 10.0, slope: 15.0, rate: 0.1 },
        n: Gate { v_half: -1.0, slope: 14.5, rate: 1.0 / 15.0 },
    };

    /// `V2` as the Fig. 9 caption prints it, mV. No figure is reproduced with it.
    pub const FIG9_PRINTED_V2: f64 = -15.0;

    /// Fig. 10a (caption, p. 209): the Fig. 9 set with `g_Ca` doubled to 8.
    pub const FIG10A: Self = Self { g_ca: 8.0, ..Self::FIG9 };

    /// Fig. 10b: the Fig. 9 set with `N∞` shifted 13 mV positive, `V3 = 12`.
    pub const FIG10B: Self = Self { n: Gate { v_half: 12.0, ..Self::FIG9.n }, ..Self::FIG9 };

    /// Fig. 10c: the Fig. 9 set with `τ_K = λ̄_N⁻¹` doubled, `λ̄_N = 1/30`.
    pub const FIG10C: Self = Self { n: Gate { rate: 1.0 / 30.0, ..Self::FIG9.n }, ..Self::FIG9 };

    /// Fig. 10d: the Fig. 9 set with the leak halved, `g_L = 1`.
    pub const FIG10D: Self = Self { g_l: 1.0, ..Self::FIG9 };

    /// Fig. 11 (caption, p. 210), the pacemaker: the Fig. 9 conductances and potentials with
    /// `V1 = −1`, `V2 = 15`, `V3 = 10`, `V4 = 14.5` — calcium now activating below potassium.
    pub const FIG11: Self = Self {
        m: Gate { v_half: -1.0, slope: 15.0, rate: 0.1 },
        n: Gate { v_half: 10.0, slope: 14.5, rate: 1.0 / 15.0 },
        ..Self::FIG9
    };

    /// `I` as the Fig. 11 caption prints it, µA/cm². The figure's period is Eq. 9's at 35.
    pub const FIG11_PRINTED_I: f64 = 50.0;

    /// Fig. 6 (caption, p. 205), full Eq. 1: `g_L = 2`, `V_L = −50`, `V_Ca = 100`, `V_K = −70`,
    /// `λ̄_M = 1.0`, `λ̄_N = 0.1` (printed without its bar), `V1 = 0`, `V2 = 15`, `V3 = 10`,
    /// `V4 = 10`, `C = 20`, with `g_Ca = 6`, `g_K = 12` — printed for the broken line (+), and
    /// drawing the solid one.
    pub const FIG6_CA6_K12: Self = Self {
        c: 20.0,
        g_l: 2.0,
        g_ca: 6.0,
        g_k: 12.0,
        v_l: -50.0,
        v_ca: 100.0,
        v_k: -70.0,
        m: Gate { v_half: 0.0, slope: 15.0, rate: 1.0 },
        n: Gate { v_half: 10.0, slope: 10.0, rate: 0.1 },
    };

    /// Fig. 6 with `g_Ca = 4`, `g_K = 8` — printed for the solid line, and drawing the broken one.
    pub const FIG6_CA4_K8: Self = Self { g_ca: 4.0, g_k: 8.0, ..Self::FIG6_CA6_K12 };

    /// Every parameter finite, `C`, `g_L` and both `λ̄` positive, `g_Ca` and `g_K` non-negative,
    /// `V2` and `V4` non-zero.
    ///
    /// # Errors
    ///
    /// [`MorrisLecarError::NotPositive`], [`MorrisLecarError::Negative`],
    /// [`MorrisLecarError::NonFinite`] or [`MorrisLecarError::Zero`], naming the first parameter
    /// that fails.
    pub fn check(&self) -> Result<(), MorrisLecarError> {
        positive("C", self.c)?;
        positive("g_L", self.g_l)?;
        non_negative("g_Ca", self.g_ca)?;
        non_negative("g_K", self.g_k)?;
        for (what, value) in [("V_L", self.v_l), ("V_Ca", self.v_ca), ("V_K", self.v_k)] {
            finite(what, value)?;
        }
        self.m.check(["V1", "V2", "lambda_M"])?;
        self.n.check(["V3", "V4", "lambda_N"])
    }

    /// Eq. 1's vector field `[dV/dt, dM/dt, dN/dt]` at `(V, M, N)` under current `I` µA/cm²,
    /// with the leak `g_L(V − V_L)`. Not checked: see [`MorrisLecar::step_full`].
    #[must_use]
    pub fn field(&self, v: f64, m: f64, n: f64, i: f64) -> [f64; 3] {
        let ionic = self.g_l * (v - self.v_l) + self.g_ca * m * (v - self.v_ca) + self.g_k * n * (v - self.v_k);
        [(i - ionic) / self.c, self.m.relax(v, m), self.n.relax(v, n)]
    }

    /// Eq. 9's vector field `(dV/dt, dN/dt)`: Eq. 1 with `M = M∞(V)`. Not checked: see
    /// [`MorrisLecar::step_reduced`].
    #[must_use]
    pub fn reduced_field(&self, v: f64, n: f64, i: f64) -> (f64, f64) {
        let [dv, _, dn] = self.field(v, self.m.steady(v), n, i);
        (dv, dn)
    }

    /// `I_ss(V)`, the current that holds the membrane at rest at `V`, µA/cm²: the equilibrium curve
    /// of Eqs. 1 and 9 alike.
    #[must_use]
    pub fn steady_current(&self, v: f64) -> f64 {
        self.g_l * (v - self.v_l) + self.g_ca * self.m.steady(v) * (v - self.v_ca) + self.g_k * self.n.steady(v) * (v - self.v_k)
    }

    /// `I_ss'(V)`, mmho/cm². Zero at a fold; Eq. 9's determinant on the curve is `λ_N(V) I_ss'(V)/C`.
    #[must_use]
    pub fn steady_current_slope(&self, v: f64) -> f64 {
        let ca = self.m.steady_slope(v) * (v - self.v_ca) + self.m.steady(v);
        let k = self.n.steady_slope(v) * (v - self.v_k) + self.n.steady(v);
        self.g_l + self.g_ca * ca + self.g_k * k
    }

    /// Eq. 9's `V̇ = 0` nullcline as `N` against `V` — Eq. 11's right side, with the leak it needs:
    /// `N = [I − g_L(V − V_L) − g_Ca M∞(V)(V − V_Ca)]/[g_K(V − V_K)]`. `None` where the denominator
    /// is zero. Not checked.
    #[must_use]
    pub fn v_nullcline(&self, v: f64, i: f64) -> Option<f64> {
        let gk = self.g_k * (v - self.v_k);
        if gk == 0.0 {
            None
        } else {
            Some((i - self.g_l * (v - self.v_l) - self.g_ca * self.m.steady(v) * (v - self.v_ca)) / gk)
        }
    }

    /// Eq. 10 as printed: `(V_min, V_max) = ((g_L V_L + g_K V_K + I)/(g_L + g_K), (g_L V_L + g_Ca V_Ca + I)/(g_L + g_Ca))`,
    /// the potentials at which `V̇` vanishes with `(M, N) = (0, 1)` and `(1, 0)`. Not checked.
    ///
    /// A bound on `V` only for `g_L(V_K − V_L) ≤ I ≤ g_L(V_Ca − V_L)`: see
    /// [`MorrisLecar::trapping_interval`] and the module doc.
    #[must_use]
    pub fn eq10_bounds(&self, i: f64) -> (f64, f64) {
        (
            (self.g_l * self.v_l + self.g_k * self.v_k + i) / (self.g_l + self.g_k),
            (self.g_l * self.v_l + self.g_ca * self.v_ca + i) / (self.g_l + self.g_ca),
        )
    }

    /// The voltage interval that no trajectory of Eq. 1 or Eq. 9 leaves while its gates stay in
    /// `[0, 1]`, and that every trajectory outside it approaches, mV. Not checked.
    ///
    /// `C V̇ = (g_L + g_Ca M + g_K N)(V̄ − V)`, where `V̄` is the mean of `V0 = V_L + I/g_L`, `V_Ca` and
    /// `V_K` weighted by `g_L`, `g_Ca M` and `g_K N`. `V̄` is a ratio of two functions linear in
    /// `(M, N)`, so over the square `0 ≤ M, N ≤ 1` its extremes lie at the corners: `V0` at `(0, 0)`,
    /// Eq. 10's `V_max` at `(1, 0)` and `V_min` at `(0, 1)`, and
    /// `V_all = (g_L V_L + g_Ca V_Ca + g_K V_K + I)/(g_L + g_Ca + g_K)` at `(1, 1)`. The interval runs
    /// from the least of the four to the greatest: `V̇` points inward at both ends, and every
    /// equilibrium is inside. When `V_K ≤ V0 ≤ V_Ca` the other two lie between `V_min` and `V_max`,
    /// and the interval is Eq. 10's.
    #[must_use]
    pub fn trapping_interval(&self, i: f64) -> (f64, f64) {
        let (lo, hi) = self.eq10_bounds(i);
        let v0 = self.v_l + i / self.g_l;
        let all = (self.g_l * self.v_l + self.g_ca * self.v_ca + self.g_k * self.v_k + i) / (self.g_l + self.g_ca + self.g_k);
        (lo.min(hi).min(v0).min(all), lo.max(hi).max(v0).max(all))
    }

    /// Every equilibrium at current `I`, with its kind under Eq. 9, found on a grid of `samples`
    /// intervals across [`MorrisLecar::trapping_interval`] (widened by 1 mV at each end) and
    /// bisected. Eq. 1 has the same equilibria; its stability can differ — see [`routh_hurwitz`].
    ///
    /// # Errors
    ///
    /// [`MorrisLecarError::NoSamples`] for `samples = 0`; [`MorrisLecarError::NonFinite`] for an `I`
    /// that is not finite; whatever [`MorrisLecar::check`] refuses.
    pub fn equilibria(&self, i: f64, samples: usize) -> Result<Vec<Equilibrium>, MorrisLecarError> {
        self.check()?;
        finite("I", i)?;
        let (lo, hi) = self.trapping_interval(i);
        let vs = roots(|v| self.steady_current(v) - i, lo - PAD, hi + PAD, samples)?;
        Ok(vs
            .into_iter()
            .map(|v| {
                let n = self.n.steady(v);
                Equilibrium { v, x: n, kind: Kind::of(&self.reduced_linearisation(v, n)) }
            })
            .collect())
    }

    /// The Jacobian of Eq. 9 at `(V, N)` — `[[∂f₁/∂V, ∂f₁/∂N], [∂f₂/∂V, ∂f₂/∂N]]` in Eq. 12's
    /// notation — per ms. It does not depend on `I`.
    #[must_use]
    pub fn reduced_linearisation(&self, v: f64, n: f64) -> Linearisation {
        let m_part = self.m.steady_slope(v) * (v - self.v_ca) + self.m.steady(v);
        Linearisation {
            jacobian: [
                [-(self.g_l + self.g_ca * m_part + self.g_k * n) / self.c, -self.g_k * (v - self.v_k) / self.c],
                [self.n.relax_slope(v, n), -self.n.rate_at(v)],
            ],
        }
    }

    /// The Jacobian of Eq. 1 at `(V, M, N)`, rows and columns in the order `V, M, N`, per ms.
    #[must_use]
    pub fn full_jacobian(&self, v: f64, m: f64, n: f64) -> [[f64; 3]; 3] {
        [
            [
                -(self.g_l + self.g_ca * m + self.g_k * n) / self.c,
                -self.g_ca * (v - self.v_ca) / self.c,
                -self.g_k * (v - self.v_k) / self.c,
            ],
            [self.m.relax_slope(v, m), -self.m.rate_at(v), 0.0],
            [self.n.relax_slope(v, n), 0.0, -self.n.rate_at(v)],
        ]
    }

    /// Eq. 15's two inequalities at a singular point `V_s` (with `N_s = N∞(V_s)`), as margins that
    /// are positive when the inequality holds, mmho/cm².
    ///
    /// With `G = g_Ca (∂M∞/∂V)(V_Ca − V_s)`, the calcium current's negative dynamic conductance, and
    /// `ḡ = g_L + g_K N_s + g_Ca M∞(V_s)`, the equivalent conductance (p. 207):
    /// `(G − ḡ − C λ_N, ḡ + g_K (∂N∞/∂V)(V_s − V_K) − G)` — Eq. 16's two gaps. They are exactly
    /// `C·trace` and `C·det/λ_N` of Eq. 9's Jacobian, so both positive is Eq. 14, an unstable node or
    /// focus.
    #[must_use]
    pub fn oscillation_margins(&self, v: f64) -> (f64, f64) {
        let neg = self.g_ca * self.m.steady_slope(v) * (self.v_ca - v);
        let gbar = self.g_l + self.g_k * self.n.steady(v) + self.g_ca * self.m.steady(v);
        (neg - gbar - self.c * self.n.rate_at(v), gbar + self.g_k * self.n.steady_slope(v) * (v - self.v_k) - neg)
    }

    fn on_curve(&self, v: f64) -> OnCurve {
        let l = self.reduced_linearisation(v, self.n.steady(v));
        OnCurve { v, i: self.steady_current(v), trace: l.trace(), det: l.determinant() }
    }

    /// Every zero of Eq. 9's trace along the equilibrium curve for `V` in `[v_lo, v_hi]` mV, on a
    /// grid of `samples` intervals: the Hopf currents where `det > 0`.
    ///
    /// # Errors
    ///
    /// [`MorrisLecarError::EmptyWindow`] for a window that is not finite and increasing;
    /// [`MorrisLecarError::NoSamples`] for `samples = 0`; whatever [`MorrisLecar::check`] refuses.
    pub fn trace_zeros(&self, v_lo: f64, v_hi: f64, samples: usize) -> Result<Vec<OnCurve>, MorrisLecarError> {
        self.check()?;
        let vs = roots(|v| self.reduced_linearisation(v, self.n.steady(v)).trace(), v_lo, v_hi, samples)?;
        Ok(vs.into_iter().map(|v| self.on_curve(v)).collect())
    }

    /// Every fold of the equilibrium curve for `V` in `[v_lo, v_hi]` mV — a zero of `I_ss'`, where
    /// two equilibria meet — on a grid of `samples` intervals.
    ///
    /// # Errors
    ///
    /// As [`MorrisLecar::trace_zeros`].
    pub fn folds(&self, v_lo: f64, v_hi: f64, samples: usize) -> Result<Vec<OnCurve>, MorrisLecarError> {
        self.check()?;
        let vs = roots(|v| self.steady_current_slope(v), v_lo, v_hi, samples)?;
        Ok(vs.into_iter().map(|v| self.on_curve(v)).collect())
    }

    /// One fourth-order Runge–Kutta step of Eq. 9, `h` ms from `(V, N)` under constant `I`.
    ///
    /// # Errors
    ///
    /// [`MorrisLecarError::NotPositive`] for a step that is not finite and positive;
    /// [`MorrisLecarError::NonFinite`] for a state or current that is not finite; whatever
    /// [`MorrisLecar::check`] refuses.
    pub fn step_reduced(&self, v: f64, n: f64, i: f64, h: f64) -> Result<(f64, f64), MorrisLecarError> {
        self.check()?;
        positive("h", h)?;
        finite("V", v)?;
        finite("N", n)?;
        finite("I", i)?;
        let [v1, n1] = rk4(
            |[v, n]| {
                let (dv, dn) = self.reduced_field(v, n, i);
                [dv, dn]
            },
            [v, n],
            h,
        );
        Ok((v1, n1))
    }

    /// One fourth-order Runge–Kutta step of Eq. 1, `h` ms from `(V, M, N)` under constant `I`.
    ///
    /// # Errors
    ///
    /// As [`MorrisLecar::step_reduced`], with `M` checked too.
    pub fn step_full(&self, v: f64, m: f64, n: f64, i: f64, h: f64) -> Result<[f64; 3], MorrisLecarError> {
        self.check()?;
        positive("h", h)?;
        finite("V", v)?;
        finite("M", m)?;
        finite("N", n)?;
        finite("I", i)?;
        Ok(rk4(|[v, m, n]| self.field(v, m, n, i), [v, m, n], h))
    }
}

/// The coefficients `[a₂, a₁, a₀]` of a 3×3 matrix's characteristic polynomial
/// `det(pI − J) = p³ + a₂p² + a₁p + a₀`: `a₂ = −trace`, `a₁` the sum of the three principal 2×2
/// minors, `a₀ = −det`.
///
/// The 3×3 counterpart of Eq. 13's two brackets. On Eq. 1's equilibrium curve a Hopf point is a
/// zero of `a₂a₁ − a₀` with `a₁ > 0`, which is how the tests find Eq. 1's.
#[must_use]
pub fn characteristic(j: &[[f64; 3]; 3]) -> [f64; 3] {
    let a2 = -(j[0][0] + j[1][1] + j[2][2]);
    let a1 = (j[0][0] * j[1][1] - j[0][1] * j[1][0]) + (j[0][0] * j[2][2] - j[0][2] * j[2][0]) + (j[1][1] * j[2][2] - j[1][2] * j[2][1]);
    let det = j[0][0] * (j[1][1] * j[2][2] - j[1][2] * j[2][1]) - j[0][1] * (j[1][0] * j[2][2] - j[1][2] * j[2][0])
        + j[0][2] * (j[1][0] * j[2][1] - j[1][1] * j[2][0]);
    [a2, a1, -det]
}

/// Whether every eigenvalue of a 3×3 matrix has a negative real part: the Routh–Hurwitz conditions
/// `a₂ > 0`, `a₀ > 0` and `a₂a₁ > a₀` on its [`characteristic`] polynomial.
///
/// The 3×3 counterpart of Eq. 14, for Eq. 1's [`MorrisLecar::full_jacobian`].
#[must_use]
pub fn routh_hurwitz(j: &[[f64; 3]; 3]) -> bool {
    let [a2, a1, a0] = characteristic(j);
    a2 > 0.0 && a0 > 0.0 && a2 * a1 > a0
}

#[cfg(test)]
mod tests {
    use super::{
        Drive, EQ7_SCALE, Gate, Kind, MorrisLecar, MorrisLecarError, SingleConductance, characteristic, electrodiffusion,
        electrodiffusion_slope, rk4, roots, routh_hurwitz,
    };
    use crate::planar::Linearisation;

    /// The step every trajectory test uses, ms: the step of the `SciPy` reference's output grid.
    const H: f64 = 0.01;

    fn run_single(s: &SingleConductance, v0: f64, i: f64, steps: usize) -> Vec<f64> {
        let (mut v, mut mu) = (v0, s.gate.steady(v0));
        let mut out = vec![v];
        for _ in 0..steps {
            (v, mu) = s.step(v, mu, i, H).unwrap();
            out.push(v);
        }
        out
    }

    fn run_reduced(m: &MorrisLecar, v0: f64, n0: f64, i: f64, steps: usize) -> Vec<f64> {
        let (mut v, mut n) = (v0, n0);
        let mut out = vec![v];
        for _ in 0..steps {
            (v, n) = m.step_reduced(v, n, i, H).unwrap();
            out.push(v);
        }
        out
    }

    fn run_full(m: &MorrisLecar, v0: f64, i: f64, steps: usize) -> Vec<f64> {
        let mut y = [v0, m.m.steady(v0), m.n.steady(v0)];
        let mut out = vec![v0];
        for _ in 0..steps {
            y = m.step_full(y[0], y[1], y[2], i, H).unwrap();
            out.push(y[0]);
        }
        out
    }

    /// Local maxima `(t, V)` of a trace sampled every `H`: `V[k−1] < V[k] ≥ V[k+1]`, as the reference
    /// script finds them.
    fn peaks(v: &[f64]) -> Vec<(f64, f64)> {
        (1..v.len() - 1).filter(|&k| v[k] > v[k - 1] && v[k] >= v[k + 1]).map(|k| (k as f64 * H, v[k])).collect()
    }

    /// The reference script's period: over the second half of the trace, the mean interval between
    /// upward crossings of the level midway between its extremes, crossings placed by linear
    /// interpolation. Returns the period, the spread of the intervals, and the half's extremes.
    fn period(v: &[f64]) -> (f64, f64, f64, f64) {
        let half = v.len() / 2;
        let lo = v[half..].iter().copied().fold(f64::INFINITY, f64::min);
        let hi = v[half..].iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let level = 0.5 * (lo + hi);
        let ups: Vec<f64> = (0..v.len() - 1)
            .filter(|&k| v[k] < level && level <= v[k + 1])
            .map(|k| (k as f64 + (level - v[k]) / (v[k + 1] - v[k])) * H)
            .filter(|&t| t > half as f64 * H)
            .collect();
        let gaps: Vec<f64> = ups.windows(2).map(|w| w[1] - w[0]).collect();
        let mean = gaps.iter().sum::<f64>() / gaps.len() as f64;
        let spread = gaps.iter().fold(0.0_f64, |s, g| s.max((g - mean).abs()));
        (mean, spread, lo, hi)
    }

    /// The eigenvalue with the larger real part, `(re, im)`.
    fn top(l: &Linearisation) -> (f64, f64) {
        l.eigenvalues()[0]
    }

    /// The eigenvalues of a 3×3 matrix from its [`characteristic`] polynomial: one real root by
    /// bisection inside Cauchy's bound, then the quadratic that remains. Returns `(real, re, im)`
    /// for the real root and the pair `re ± i·im`; `im` is 0 when the pair is real.
    fn roots3(j: &[[f64; 3]; 3]) -> (f64, f64, f64) {
        let [a2, a1, a0] = characteristic(j);
        let p = |x: f64| ((x + a2) * x + a1) * x + a0;
        let bound = 1.0 + a2.abs().max(a1.abs()).max(a0.abs());
        let (mut lo, mut hi) = (-bound, bound);
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            if p(mid) < 0.0 { lo = mid } else { hi = mid }
        }
        let r = 0.5 * (lo + hi);
        let (b, c) = (a2 + r, a1 + r * (a2 + r));
        (r, -0.5 * b, (c - 0.25 * b * b).max(0.0).sqrt())
    }

    /// Central differences of a field in each coordinate, compared with a Jacobian entry by entry,
    /// relative to its largest entry. Returns the worst relative disagreement.
    fn fd_error<const D: usize>(f: &dyn Fn([f64; D]) -> [f64; D], x: [f64; D], jac: [[f64; D]; D]) -> f64 {
        let h = 1e-5;
        let scale = jac.iter().flatten().fold(0.0_f64, |m, v| m.max(v.abs()));
        let mut worst = 0.0_f64;
        for col in 0..D {
            let (mut up, mut down) = (x, x);
            up[col] += h;
            down[col] -= h;
            let (fu, fd) = (f(up), f(down));
            for row in 0..D {
                worst = worst.max(((fu[row] - fd[row]) / (2.0 * h) - jac[row][col]).abs() / scale);
            }
        }
        worst
    }

    /// Eq. 2's gates: the steady state is a logistic in `2(V − V_half)/V_slope`, the rate a `cosh`,
    /// and every derivative the module uses agrees with a central difference.
    ///
    /// `½{1 + tanh u} = 1/(1 + e^{−2u})` is an identity; measured agreement 2.2 × 10⁻¹⁶, one unit in
    /// the last place of ½, over ±100 mV. The derivatives agree with central differences of step
    /// 10⁻⁵ mV to 2 × 10⁻⁹ of the largest value (measured 5.9 × 10⁻¹⁰). A negative slope — Fig. 9's
    /// printed `V2` — turns the steady state around and leaves the rate unchanged, since `cosh` is
    /// even.
    #[test]
    fn the_gates_are_eq_2() {
        let gates = [MorrisLecar::FIG9.m, MorrisLecar::FIG9.n, MorrisLecar::FIG11.m, MorrisLecar::FIG6_CA6_K12.n];
        let (mut logistic, mut deriv) = (0.0_f64, 0.0_f64);
        for g in gates {
            assert_eq!(g.steady(g.v_half), 0.5);
            assert_eq!(g.rate_at(g.v_half), g.rate);
            assert_eq!(g.rate_slope(g.v_half), 0.0);
            let flipped = Gate { slope: -g.slope, ..g };
            for k in -40..=40 {
                let v = g.v_half + f64::from(k) * 2.5;
                let u = (v - g.v_half) / g.slope;
                logistic = logistic.max((g.steady(v) - 1.0 / (1.0 + (-2.0 * u).exp())).abs());
                assert!((g.rate_at(v) - g.rate * 0.5 * ((0.5 * u).exp() + (-0.5 * u).exp())).abs() <= 1e-15 * g.rate_at(v));
                assert!((g.steady(v) + flipped.steady(v) - 1.0).abs() < 1e-15, "a negative slope mirrors the gate");
                assert_eq!(flipped.rate_at(v), g.rate_at(v));
                let h = 1e-5;
                let d = |f: &dyn Fn(f64) -> f64| (f(v + h) - f(v - h)) / (2.0 * h);
                let x = 0.3;
                for (got, want, scale) in [
                    (g.steady_slope(v), d(&|w| g.steady(w)), 0.5 / g.slope.abs()),
                    (g.rate_slope(v), d(&|w| g.rate_at(w)), g.rate_at(v) / g.slope.abs()),
                    (g.relax_slope(v, x), d(&|w| g.relax(w, x)), g.rate_at(v) / g.slope.abs()),
                ] {
                    deriv = deriv.max((got - want).abs() / scale);
                }
                assert_eq!(g.relax(v, g.steady(v)), 0.0, "at steady state the gate does not move");
                assert!((g.relax(v, 0.0) - g.relax(v, 1.0) - g.rate_at(v)).abs() <= 4e-16 * g.rate_at(v));
            }
        }
        assert!(logistic < 7e-16, "{logistic}");
        assert!(deriv < 2e-9, "{deriv}");
    }

    /// Eq. 7's `R` is the printed quotient everywhere the quotient can be evaluated, and continuous
    /// through the `0/0` at `V = 0`, where it is `−12.5(1 − ρ)`.
    ///
    /// Against the printed `V{1 − ρe^{V/12.5}}/[1 − e^{V/12.5}]` at 400 potentials from −200 to +200 mV
    /// (zero excluded) and four ratios, the disagreement is at most 1.1 × 10⁻¹⁵ of the larger of `|R|`
    /// and 1 mV (measured). Its slope agrees with central differences of step 10⁻⁴ mV to 3.2 × 10⁻¹⁰
    /// (measured; bound 10⁻⁹), is `(1 + ρ)/2` at zero and stays between `ρ` and 1. Near zero it is
    /// the series `ρ − (1 − ρ)(−1/2 + x/6 − x³/180 + x⁵/5040)`, `x = V/12.5`, to within the
    /// cancellation bound [`electrodiffusion_slope`] states, `4ε/|x|` relative: measured 0.22 of that
    /// bound at worst, and 9.5 × 10⁻¹³ relative at `|V| = 1.25 µV`, against the stated 10⁻¹¹. It
    /// vanishes at the reversal potential `12.5 ln(1/ρ)`, and it is finite at ±9 V, where the printed
    /// form overflows.
    #[test]
    fn electrodiffusion_is_eq_7_through_its_removable_singularity() {
        let mut worst = 0.0_f64;
        let mut slope_err = 0.0_f64;
        let (mut cancellation, mut micro) = (0.0_f64, 0.0_f64);
        for rho in [0.0, 1e-5, 0.01, 0.1] {
            assert_eq!(electrodiffusion(0.0, rho), -12.5 * (1.0 - rho));
            assert_eq!(electrodiffusion_slope(0.0, rho), 0.5 * (1.0 + rho));
            for k in -200..=200 {
                if k == 0 {
                    continue;
                }
                let v = f64::from(k) + 0.25;
                let e = (v / 12.5).exp();
                let printed = v * (1.0 - rho * e) / (1.0 - e);
                let r = electrodiffusion(v, rho);
                worst = worst.max((r - printed).abs() / printed.abs().max(1.0));
                let h = 1e-4;
                let fd = (electrodiffusion(v + h, rho) - electrodiffusion(v - h, rho)) / (2.0 * h);
                let s = electrodiffusion_slope(v, rho);
                slope_err = slope_err.max((s - fd).abs());
                assert!(s > rho && s < 1.0, "the slope is between rho and 1: {s} at {v}");
            }
            // Continuity through the singularity, from both sides: R moves off its limit with slope
            // (1 + ρ)/2, and the slope is B's series −1/2 + x/6 − x³/180 + x⁵/5040 to the
            // cancellation bound the function documents, 4ε/|x| relative.
            for v in [1e-9, -1e-9, 1.25e-3, -1.25e-3, 1e-3, -1e-3, 0.125, -0.125] {
                let limit = -12.5 * (1.0 - rho);
                assert!((electrodiffusion(v, rho) - limit).abs() <= 0.5 * (1.0 + rho) * v.abs() * 1.01, "R({v}) = {}", electrodiffusion(v, rho));
                let x: f64 = v / 12.5;
                let want = rho - (1.0 - rho) * (-0.5 + x / 6.0 - x.powi(3) / 180.0 + x.powi(5) / 5040.0);
                let rel = ((electrodiffusion_slope(v, rho) - want) / want).abs();
                cancellation = cancellation.max(rel * x.abs() / (4.0 * f64::EPSILON));
                if x.abs() == 1e-4 {
                    micro = micro.max(rel);
                }
            }
            assert!(electrodiffusion(9000.0, rho).is_finite() && electrodiffusion(-9000.0, rho).is_finite());
            assert!((electrodiffusion(-9000.0, rho) + 9000.0).abs() < 1e-9, "far below reversal R is V");
            assert!((electrodiffusion(9000.0, rho) - 9000.0 * rho).abs() < 1e-9, "far above it R is rho V");
            if rho > 0.0 {
                let rev = 12.5 * (1.0 / rho).ln();
                assert!(electrodiffusion(rev, rho).abs() < 1e-12 * rev, "R vanishes at reversal: {}", electrodiffusion(rev, rho));
                assert!(electrodiffusion(rev - 1.0, rho) < 0.0 && electrodiffusion(rev + 1.0, rho) > 0.0);
            }
        }
        assert!(worst < 5e-15, "{worst}");
        assert!(slope_err < 1e-9, "{slope_err}");
        assert!(cancellation < 0.5, "{cancellation}");
        assert!(micro < 1e-11, "{micro}");
    }

    /// `12.5` is `RT/2F` at the experiments' 22 °C to within 2%: 12.72 mV with the CODATA 2018
    /// constants.
    #[test]
    fn eqs_7_constant_is_rt_over_2f_at_22_degrees() {
        let rt_2f: f64 = 8.314_462_618 * (273.15 + 22.0) / (2.0 * 96_485.332_12) * 1e3;
        assert!((rt_2f - 12.717).abs() < 1e-3, "{rt_2f}");
        assert_eq!(EQ7_SCALE, 12.5);
        assert!((EQ7_SCALE - rt_2f).abs() / rt_2f < 0.02);
    }

    /// Fig. 2b: the all-K system's graded responses to 25, 100 and 400 µA/cm² from rest at −50 mV.
    ///
    /// Reference: `SciPy`'s DOP853 at tolerance `10⁻¹²` on Eq. 3 with `V(0) = −50`, `N(0) = N∞(−50)`,
    /// sampled every 0.01 ms. The module's RK4 at the same step agrees at the peaks and at 200 ms to
    /// 5.1 × 10⁻¹⁰ mV (measured; bound 2 × 10⁻⁹), and its peaks fall on the same samples. Against the
    /// page — read off Fig. 2b (p. 197) — `I = 25` settles near −42 mV, `I = 100` peaks near −21 mV
    /// and settles near −22.5, and `I = 400` peaks near +20 mV inside 10 ms and settles near −2.
    #[test]
    fn figure_2b_the_all_k_responses() {
        let s = SingleConductance::FIG2;
        // (I, peak time, peak, V(200 ms)) from the reference; I = 25 has no early peak to speak of.
        let cases = [
            (25.0, None, -41.93020683328814),
            (100.0, Some((19.35, -21.155881556415686)), -22.69378272933245),
            (400.0, Some((7.01, 21.720206132231283)), -1.8927005495188385),
        ];
        let mut err = 0.0_f64;
        for (i, peak, v200) in cases {
            let v = run_single(&s, -50.0, i, 20_000);
            err = err.max((v[20_000] - v200).abs());
            if let Some((t, p)) = peak {
                let first = peaks(&v)[0];
                assert!((first.0 - t).abs() < 1e-9, "I = {i}: peak at {} ms, reference {t}", first.0);
                err = err.max((first.1 - p).abs());
            }
            // The run ends at the one equilibrium: within 2.5 × 10⁻¹⁰ mV (measured).
            let eq = s.equilibria(i, 2_000).unwrap();
            assert_eq!(eq.len(), 1);
            assert!((eq[0].v - v[20_000]).abs() < 1e-9, "{} vs {}", eq[0].v, v[20_000]);
        }
        assert!(err < 2e-9, "{err}");
        let v = run_single(&s, -50.0, 400.0, 20_000);
        let first = peaks(&v)[0];
        assert!(first.0 < 10.0 && (first.1 - 20.0).abs() < 3.0, "the page: about +20 mV inside 10 ms, {first:?}");
        assert!((v[20_000] + 2.0).abs() < 1.0);
        let v = run_single(&s, -50.0, 100.0, 20_000);
        assert!((peaks(&v)[0].1 + 21.0).abs() < 1.0 && (v[20_000] + 22.5).abs() < 1.0);
        let v = run_single(&s, -50.0, 25.0, 20_000);
        assert!((v[20_000] + 42.0).abs() < 1.0);
    }

    /// Read as the abbreviation list's s⁻¹, `λ̄_N = 1/15` cannot draw Fig. 2b.
    ///
    /// `1/15` s⁻¹ is `1/15 000` ms⁻¹. `N` then moves from 0.0012 to 0.071 in 200 ms (reference:
    /// 0.0011594832518865839 to 0.07097668393976922; RK4 agrees to 9.4 × 10⁻¹³), and the `I = 400`
    /// response climbs to +78.9 mV and is still at +59.3 mV at 200 ms (reference: DOP853 as above; RK4
    /// agrees to 8.3 × 10⁻¹⁰ mV, measured), where the figure peaks near +20 mV and settles near −2.
    #[test]
    fn rates_in_per_second_cannot_draw_figure_2b() {
        let s = SingleConductance { gate: Gate { rate: 1.0 / 15_000.0, ..SingleConductance::FIG2.gate }, ..SingleConductance::FIG2 };
        let v = run_single(&s, -50.0, 400.0, 20_000);
        let top = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        assert!((top - 78.90710200580936).abs() < 3e-9 && (v[20_000] - 59.34954875819219).abs() < 3e-9, "{top} {}", v[20_000]);
        let n0 = s.gate.steady(-50.0);
        let (mut x, mut n) = (-50.0, n0);
        for _ in 0..20_000 {
            (x, n) = s.step(x, n, 400.0, H).unwrap();
        }
        assert!((n0 - 0.0011594832518865839).abs() < 1e-15 && (n - 0.07097668393976922).abs() < 3e-12, "{n0} {n}");
    }

    /// Fig. 5a's nullclines are drawn with `V_L = −50`; the caption's printed +50 would move them.
    ///
    /// Eq. 4a at `μ = 0` is `V(0) = V_L + I/g_L`, and at `μ = 1` Eq. 5,
    /// `(I + g_L V_L + g_K V_K)/(g_L + g_K)`: for `I = 400` that is `−310/11 = −28.18 mV`, where the
    /// figure's `I = 400` nullcline meets `N = 1`. With `V_L = +50` it would be `−10/11 = −0.91 mV`. At
    /// `V = −50` the `I = 25` and `I = 100` nullclines are at `N = 25/160` and `100/160` exactly, as
    /// drawn. The two forms of the nullcline, Eq. 4a and `μ(V)`, invert each other.
    #[test]
    fn figure_5a_needs_v_l_minus_50() {
        let s = SingleConductance::FIG2;
        assert_eq!(s.mu_nullcline(-50.0, 25.0), Some(0.15625));
        assert_eq!(s.mu_nullcline(-50.0, 100.0), Some(0.625));
        assert!((s.v_nullcline(1.0, 400.0).unwrap() + 310.0 / 11.0).abs() < 1e-13);
        assert!((s.v_nullcline(0.0, 400.0).unwrap() - (-50.0 + 400.0 / 3.0)).abs() < 1e-13);
        let printed = SingleConductance { v_l: SingleConductance::FIG5A_PRINTED_V_L, ..s };
        assert!((printed.v_nullcline(1.0, 400.0).unwrap() + 10.0 / 11.0).abs() < 1e-13);
        for k in 0..=20 {
            let mu = f64::from(k) / 20.0;
            for i in [0.0, 25.0, 100.0, 400.0] {
                let v = s.v_nullcline(mu, i).unwrap();
                let back = s.mu_nullcline(v, i).unwrap();
                assert!((back - mu).abs() < 1e-14, "mu {mu} I {i}: {back}");
                assert!(s.field(v, mu, i).0.abs() < 1e-14, "Eq. 4a is where V does not move");
            }
        }
        assert_eq!(s.mu_nullcline(-70.0, 10.0), None, "at V_K the drive vanishes");
        assert_eq!(SingleConductance { g: 0.0, ..s }.mu_nullcline(-50.0, 10.0), None);
    }

    /// The all-K system has one singular point at every current from −60 to 800 µA/cm², always
    /// stable — and a FOCUS, not the caption's "always a stable node", from 2.117 to 731.6 µA/cm².
    ///
    /// On the equilibrium curve the discriminant `trace² − 4 det` changes sign at `V = −49.364 mV`,
    /// `I = 2.116 583 937 µA/cm²`, and again at `V = 11.208 mV`, `I = 731.559 625 µA/cm²` (reference:
    /// `brentq` on the discriminant, 2.1165839370827713 and 731.5596254817773; measured agreement
    /// 6.4 × 10⁻¹⁴ and 1.1 × 10⁻¹³), and nowhere else between −70 and +40 mV (`I` from −60 to above
    /// 1100): the kind is a node at every current of the grid below 2.117 and above 731.6, and a focus
    /// at every one between. The eigenvalues at 25, 100 and 400 are the reference's (`NumPy`) to
    /// 5.6 × 10⁻¹⁷ ms⁻¹ (measured). Fig. 2b's `I = 400` trace is consistent with the focus without
    /// showing it: it dips to −3.23 mV at 20.83 ms, 1.3 mV below where it settles, which a node could
    /// also do after the swing down from +21.7 mV, and the crossing only a focus makes — back above the
    /// final level, to a second peak at 35.97 ms — is 0.063 mV, half a pixel of the page at 600 dpi.
    #[test]
    fn the_all_k_point_is_unique_and_a_focus_above_two_microamps() {
        let s = SingleConductance::FIG2;
        for k in -12..=160 {
            let i = f64::from(k) * 5.0;
            let eq = s.equilibria(i, 2_000).unwrap();
            assert_eq!(eq.len(), 1, "I = {i}: {eq:?}");
            let want = if (5.0..=730.0).contains(&i) { Kind::StableFocus } else { Kind::StableNode };
            assert_eq!(eq[0].kind, want, "I = {i}: {eq:?}");
        }
        let mut worst = 0.0_f64;
        for (i, re, im) in [
            (25.0, -0.148128208911954, 0.02786130880094744),
            (100.0, -0.12765853638687963, 0.09229252206535025),
            (400.0, -0.20220033871495685, 0.2099032403418031),
        ] {
            let e = s.equilibria(i, 2_000).unwrap()[0];
            assert_eq!(e.kind, Kind::StableFocus, "I = {i}");
            let (r, m) = top(&s.linearisation(e.v, e.x));
            worst = worst.max((r - re).abs()).max((m - im).abs());
        }
        let disc = |v: f64| {
            let l = s.linearisation(v, s.gate.steady(v));
            l.trace() * l.trace() - 4.0 * l.determinant()
        };
        let signs = roots(disc, -70.0, 40.0, 11_000).unwrap();
        assert_eq!(signs.len(), 2, "{signs:?}");
        assert!(s.steady_current(-70.0) <= -60.0 && s.steady_current(40.0) > 1_100.0);
        let mut bound = 0.0_f64;
        for ((mut a, mut b), want) in [((-50.0, -45.0), 2.1165839370827713), ((20.0, 5.0), 731.5596254817773)] {
            assert!(disc(a) > 0.0 && disc(b) < 0.0, "a node at {a}, a focus at {b}");
            for _ in 0..100 {
                let mid = 0.5 * (a + b);
                if disc(mid) > 0.0 { a = mid } else { b = mid }
            }
            bound = bound.max((s.steady_current(a) - want).abs());
        }
        assert!(bound < 1e-12, "{bound}");
        assert_eq!(s.equilibria(750.0, 2_000).unwrap()[0].kind, Kind::StableNode, "a node again above 731.6");
        assert!(worst < 3e-16, "{worst}");
        let v = run_single(&s, -50.0, 400.0, 20_000);
        let (k_low, low) = (701..3000).map(|k| (k, v[k])).fold((0, f64::INFINITY), |m, p| if p.1 < m.1 { p } else { m });
        let second = peaks(&v)[1];
        assert!((low - (-3.2303382505481073)).abs() < 1e-8 && low < v[20_000] - 1.3, "{low}");
        assert_eq!(k_low, 2083, "the dip is at 20.83 ms");
        assert!((second.0 - 35.97).abs() < 1e-9 && (second.1 - (-1.82958603606778)).abs() < 1e-8, "{second:?}");
        let above = second.1 - v[20_000];
        assert!((above - 0.063).abs() < 5e-4, "the second crossing: {above} mV");
    }

    /// Fig. 3b and Fig. 5b: the all-Ca system, with Eq. 7's drive entered inward, has the paper's
    /// three singular points at `I = 0`, its one plateau at `I = 100`, and Fig. 3b's plateaus.
    ///
    /// At `I = 0`: A, a stable node near −33 mV with `M` near zero; B, a saddle near −9 mV at
    /// `M = 0.07`; C, a stable node near +27 mV at `M = 0.91`; and at `I = 100` only C′, a stable node
    /// near +43 mV — all as Fig. 5b draws and p. 203 describes them. Reference positions from `brentq`
    /// on the same equation; measured agreement 7.1 × 10⁻¹⁵ mV. Fig. 3b's plateaus at 15, 25 and 50
    /// µA/cm² are the upper equilibria, and the `I = 15` trace from `V = −35` crosses 0 mV at 180.156 ms
    /// (DOP853: 180.15576364570958; RK4 at 0.01 ms agrees to 1.3 × 10⁻¹² ms, measured). The drawn
    /// trace crosses near 180 ms: four separate readings at 600 dpi (5.31 to 5.33 pixels to the
    /// millisecond, 11.43 to 11.44 to the millivolt) put the stroke's centre on the 0 mV row at
    /// 179.94 to 180.2 ms, each good to about ±0.3 ms, the spread of the ticks about their straight
    /// line; the model is inside that against every reading.
    #[test]
    fn figures_3b_and_5b_the_all_ca_plateau() {
        let s = SingleConductance::FIG3B;
        let eq = s.equilibria(0.0, 20_000).unwrap();
        let kinds: Vec<Kind> = eq.iter().map(|e| e.kind).collect();
        assert_eq!(kinds, [Kind::StableNode, Kind::Saddle, Kind::StableNode], "A, B, C: {eq:?}");
        let mut worst = 0.0_f64;
        for (e, (v, mu)) in eq.iter().zip([
            (-32.607431709252616, 0.003398587853850943),
            (-8.841853471105942, 0.0750040581564983),
            (27.493682147539317, 0.9115324158864554),
        ]) {
            worst = worst.max((e.v - v).abs());
            assert!((e.x - mu).abs() < 5e-16);
        }
        let page = [(-33.0, 0.0), (-9.0, 0.07), (27.0, 0.91)];
        for (e, (v, mu)) in eq.iter().zip(page) {
            assert!((e.v - v).abs() < 1.0 && (e.x - mu).abs() < 0.02, "{e:?} against the page's {v}, {mu}");
        }
        let c_prime = s.equilibria(100.0, 20_000).unwrap();
        assert_eq!(c_prime.len(), 1);
        assert_eq!(c_prime[0].kind, Kind::StableNode);
        worst = worst.max((c_prime[0].v - 43.05472604277627).abs());
        for (i, plateau) in [(15.0, 29.57333528972675), (25.0, 30.976753465759106), (50.0, 34.62376361072389)] {
            let top = *s.equilibria(i, 20_000).unwrap().last().unwrap();
            worst = worst.max((top.v - plateau).abs());
        }
        assert!(worst < 3e-14, "{worst}");
        let v = run_single(&s, -35.0, 15.0, 20_000);
        let k = (1..v.len()).find(|&k| v[k - 1] < 0.0 && v[k] >= 0.0).unwrap();
        let t0 = (k as f64 - 1.0 + (0.0 - v[k - 1]) / (v[k] - v[k - 1])) * H;
        assert!((t0 - 180.15576364570958).abs() < 5e-12, "{t0}");
        for page in [179.94, 180.2] {
            assert!((t0 - page).abs() < 0.3, "against the page's {page}: {t0}");
        }
    }

    /// Fig. 5c: as calcium accumulates the upper singular point C and the saddle B move together and
    /// vanish, leaving the rest point A.
    ///
    /// The caption's five pairs `(g*_Ca, [Ca]_i)` at `I = 0`. Reference roots from `brentq` on the
    /// same equation; measured agreement 3.6 × 10⁻¹⁴ mV. The first three keep three singular points,
    /// the last two only A, and A itself moves left, as the text says the `V̇ = 0` nullcline does.
    #[test]
    fn figure_5c_accumulation_takes_the_plateau_away() {
        let cases: [(f64, f64, &[f64]); 5] = [
            (40.0, 0.001, &[-32.60743408935897, -8.84175444555902, 27.492121834160724]),
            (30.0, 0.1, &[-33.34375981751282, -2.742024786508113, 21.895107495269272]),
            (25.0, 0.5, &[-33.66808545202949, 2.2932814348011066, 16.77803333749492]),
            (20.0, 1.0, &[-33.969243713531505]),
            (10.0, 10.0, &[-34.51733670273706]),
        ];
        let mut rest = f64::INFINITY;
        let mut worst = 0.0_f64;
        for (g, ca_in, want) in cases {
            let s = SingleConductance { g, drive: Drive::Electrodiffusion { ca_in, ca_out: 100.0 }, ..SingleConductance::FIG3B };
            let got: Vec<f64> = s.equilibria(0.0, 20_000).unwrap().iter().map(|e| e.v).collect();
            assert_eq!(got.len(), want.len(), "({g}, {ca_in}): {got:?}");
            for (a, b) in got.iter().zip(want) {
                worst = worst.max((a - b).abs());
            }
            assert!(got[0] < rest, "A moves left");
            rest = got[0];
        }
        assert!(worst < 1e-13, "{worst}");
    }

    /// Eq. 1's leak as printed, `g_L(V_L)`, is a constant current, and puts rest near −7 mV.
    ///
    /// With it the all-K system of Fig. 2b has its one equilibrium at −7.19 mV at `I = 0`, and at
    /// −5.79, −2.15 and +11.32 mV at 25, 100 and 400 µA/cm², where the figure starts every trace from
    /// rest at −50 mV and settles them near −42, −22.5 and −2 mV; the Fig. 9 set would rest at
    /// −8.07 mV, where Fig. 9a's traces start from −50. Reference (`brentq`): −7.1935204608067895,
    /// −5.787400701865791, −2.1464186511247925, 11.319325796777974 and −8.072483744295793; measured
    /// agreement 3.6 × 10⁻¹⁵ mV. With `g_L(V − V_L)` the rest points are −50.06 and −49.99 mV.
    #[test]
    fn eq_1_as_printed_rests_near_minus_7_mv() {
        let bisect = |f: &dyn Fn(f64) -> f64, mut a: f64, mut b: f64| {
            assert!(f(a) < 0.0 && f(b) > 0.0, "a bracket");
            for _ in 0..200 {
                let mid = 0.5 * (a + b);
                if f(mid) < 0.0 { a = mid } else { b = mid }
            }
            0.5 * (a + b)
        };
        let s = SingleConductance::FIG2;
        let mut worst = 0.0_f64;
        for (i, want) in [(0.0, -7.1935204608067895), (25.0, -5.787400701865791), (100.0, -2.1464186511247925), (400.0, 11.319325796777974)] {
            let printed = |v: f64| s.g_l * s.v_l + s.g * s.gate.steady(v) * s.drive.at(v) - i;
            worst = worst.max((bisect(&printed, -40.0, 40.0) - want).abs());
        }
        let m = MorrisLecar::FIG9;
        let printed = |v: f64| m.g_l * m.v_l + m.g_ca * m.m.steady(v) * (v - m.v_ca) + m.g_k * m.n.steady(v) * (v - m.v_k);
        worst = worst.max((bisect(&printed, -40.0, 40.0) - (-8.072483744295793)).abs());
        assert!(worst < 1e-14, "{worst}");
        assert!((s.equilibria(0.0, 2_000).unwrap()[0].v - (-50.06113302466996)).abs() < 1e-13);
        assert!((m.equilibria(0.0, 10_000).unwrap()[0].v - (-49.99219020343545)).abs() < 1e-13);
    }

    /// Eq. 6 as printed, `I_Ca = −g*_Ca M R`, holds no plateau: with `R < 0` that current is outward.
    ///
    /// Scanning the printed form's steady current from −100 to +100 mV finds one rest point for each of
    /// Fig. 3b's currents, below `V_L + I/g_L` and far below 0 mV — where the figure shows plateaus
    /// above +29 mV.
    #[test]
    fn eq_6_as_printed_holds_no_plateau() {
        let s = SingleConductance::FIG3B;
        for i in [15.0, 25.0, 50.0] {
            let printed = |v: f64| s.g_l * (v - s.v_l) - s.g * s.gate.steady(v) * electrodiffusion(v, 0.0) - i;
            let grid: Vec<f64> = (-10_000..=10_000).map(|k| f64::from(k) / 100.0).collect();
            let roots: Vec<f64> = grid.windows(2).filter(|w| (printed(w[0]) < 0.0) != (printed(w[1]) < 0.0)).map(|w| w[0]).collect();
            assert_eq!(roots.len(), 1, "I = {i}: {roots:?}");
            assert!(roots[0] < s.v_l + i / s.g_l && roots[0] < -15.0, "I = {i}: {roots:?}");
            assert!(s.equilibria(i, 20_000).unwrap().last().unwrap().v > 29.0, "the inward reading has the plateau");
        }
    }

    /// Fig. 9's equilibria and their eigenvalues across Fig. 8's currents, against `NumPy`.
    ///
    /// Reference: `brentq` on `I_ss(V) = I` and `numpy.linalg.eigvals` of the analytic Jacobian.
    /// Measured agreement 7.1 × 10⁻¹⁵ mV and 1.4 × 10⁻¹⁶ ms⁻¹. Every current has one equilibrium,
    /// as p. 206 says ("The nullclines intersect once").
    #[test]
    fn figure_9_equilibria_and_eigenvalues() {
        let m = MorrisLecar::FIG9;
        let table = [
            (0.0, -49.99219020343545, -0.10201068483296108, 0.0, Kind::StableNode),
            (100.0, -16.34182341978526, -0.06888128626816645, 0.1469739415642651, Kind::StableFocus),
            (200.0, -6.938470173997106, -0.03253748464045732, 0.22142186512375084, Kind::StableFocus),
            (300.0, -0.9016140945101736, 0.00363834537147266, 0.24206060696293166, Kind::UnstableFocus),
            (400.0, 5.093435120824236, 0.02679289561806704, 0.22458148692988383, Kind::UnstableFocus),
            (475.0, 10.89147305597114, -0.01067402841194569, 0.20031674052407883, Kind::StableFocus),
            (500.0, 13.235103297041684, -0.04726156146572691, 0.19064748199019393, Kind::StableFocus),
            (550.0, 18.17201731038362, -0.14696023760484128, 0.1427867940779295, Kind::StableFocus),
        ];
        let (mut dv, mut dp) = (0.0_f64, 0.0_f64);
        for (i, v, re, im, kind) in table {
            let eq = m.equilibria(i, 10_000).unwrap();
            assert_eq!(eq.len(), 1, "I = {i}");
            assert_eq!(eq[0].kind, kind, "I = {i}");
            dv = dv.max((eq[0].v - v).abs());
            let (r, j) = top(&m.reduced_linearisation(eq[0].v, eq[0].x));
            dp = dp.max((r - re).abs()).max((j - im).abs());
        }
        assert!(dv < 3e-14 && dp < 5e-16, "{dv} {dp}");
    }

    /// Fig. 8's shaded loop ends at two Hopf currents, found here to the last bits.
    ///
    /// The trace of Eq. 9's Jacobian along the equilibrium curve vanishes at `V = −1.4907` and
    /// `V = 10.0191 mV`, where `I_ss` is 289.651 573 716 and 465.104 978 128 µA/cm² and the eigenvalues
    /// are `±0.241 843 5i` and `±0.203 453 5i` ms⁻¹. Reference: `brentq` on the same trace
    /// (289.65157371603095, 465.10497812788356): the lower is the same double, and the upper
    /// 1.1 × 10⁻¹³ away, from a root two doubles apart in `V` (measured). The paper
    /// prints neither number; the extraction's readers found 289.65 and 465.10. Just inside each end
    /// the equilibrium is an unstable focus, just outside a stable one.
    #[test]
    fn figure_8_hopf_currents() {
        let m = MorrisLecar::FIG9;
        let z = m.trace_zeros(-60.0, 60.0, 1_200).unwrap();
        assert_eq!(z.len(), 2, "{z:?}");
        let want = [(-1.4906662190930566, 289.65157371603095, 0.24184353805206926), (10.019050393136647, 465.10497812788356, 0.203453517183528)];
        let mut worst = 0.0_f64;
        for (p, (v, i, w)) in z.iter().zip(want) {
            worst = worst.max((p.i - i).abs());
            assert!((p.v - v).abs() < 2e-14, "{p:?}");
            assert!(p.trace.abs() < 1e-16 && p.det > 0.0, "{p:?}");
            assert!((p.det.sqrt() - w).abs() < 1e-15, "{p:?}");
        }
        assert!(worst < 3e-13, "{worst}");
        let kind = |i: f64| m.equilibria(i, 10_000).unwrap()[0].kind;
        assert_eq!([kind(289.6), kind(289.7), kind(465.0), kind(465.2)], [
            Kind::StableFocus,
            Kind::UnstableFocus,
            Kind::UnstableFocus,
            Kind::StableFocus
        ]);
    }

    /// Fig. 8's locus in order: real at `I = 0`, complex with negative real part at 100 and 200,
    /// inside the loop at 300 and 400, out again by 475 and 550, and real once more past 589.09.
    ///
    /// The locus leaves and rejoins the real axis where `trace² = 4 det`: at 14.059 299 5 and
    /// 589.088 101 5 µA/cm² (reference: `brentq`, 14.059299505851328 and 589.0881014566572; measured
    /// agreement 2.8 × 10⁻¹⁴).
    #[test]
    fn figure_8_root_locus() {
        let m = MorrisLecar::FIG9;
        let kinds: Vec<Kind> = [0.0, 100.0, 200.0, 300.0, 400.0, 475.0, 550.0, 600.0].iter().map(|&i| m.equilibria(i, 10_000).unwrap()[0].kind).collect();
        use Kind::{StableFocus as SF, StableNode as SN, UnstableFocus as UF};
        assert_eq!(kinds, [SN, SF, SF, UF, UF, SF, SF, SN]);
        let disc = |v: f64| {
            let l = m.reduced_linearisation(v, m.n.steady(v));
            l.trace() * l.trace() - 4.0 * l.determinant()
        };
        let mut worst = 0.0_f64;
        for ((mut a, mut b), want) in [((-45.0, -40.0), 14.059299505851328), ((20.0, 25.0), 589.0881014566572)] {
            let below = disc(a) < 0.0;
            for _ in 0..100 {
                let mid = 0.5 * (a + b);
                if (disc(mid) < 0.0) == below { a = mid } else { b = mid }
            }
            worst = worst.max((m.steady_current(a) - want).abs());
        }
        assert!(worst < 1e-13, "{worst}");
    }

    /// Fig. 7's "unstable node" at `I = 300` is an unstable focus: Eq. 17 holds, and Eq. 14 — both
    /// margins of Eq. 15 positive — holds too. The Fig. 9/10 control that Fig. 12a draws inside its
    /// damped band is this point.
    #[test]
    fn figure_7_is_an_unstable_focus() {
        let m = MorrisLecar::FIG9;
        let e = m.equilibria(300.0, 10_000).unwrap()[0];
        let l = m.reduced_linearisation(e.v, e.x);
        assert_eq!(e.kind, Kind::UnstableFocus);
        assert!(l.trace() * l.trace() - 4.0 * l.determinant() < 0.0, "Eq. 17: complex roots");
        let (a, b) = m.oscillation_margins(e.v);
        assert!(a > 0.0 && b > 0.0, "Eq. 15: {a} {b}");
        assert!((e.v - (-0.9016140945101736)).abs() < 2e-14 && (e.x - 0.5033925653669384).abs() < 1e-15);
    }

    /// Fig. 9's printed `V2 = −15` makes every equilibrium a stable node: nothing in Figs. 7–10 can
    /// be drawn with it.
    ///
    /// From 0 to 550 µA/cm² in steps of 10, one equilibrium each, both eigenvalues real and at most
    /// −0.1139 ms⁻¹ (reference: `NumPy`, largest real part −0.11390804649031287 over steps of 1), and the
    /// trace never vanishes on the curve between −100 and +100 mV.
    #[test]
    fn figure_9_as_printed_cannot_oscillate() {
        let printed = MorrisLecar { m: Gate { slope: MorrisLecar::FIG9_PRINTED_V2, ..MorrisLecar::FIG9.m }, ..MorrisLecar::FIG9 };
        let mut worst = f64::NEG_INFINITY;
        for k in 0..=55 {
            let i = f64::from(k) * 10.0;
            let eq = printed.equilibria(i, 10_000).unwrap();
            assert_eq!(eq.len(), 1, "I = {i}");
            assert_eq!(eq[0].kind, Kind::StableNode, "I = {i}");
            worst = worst.max(top(&printed.reduced_linearisation(eq[0].v, eq[0].x)).0);
        }
        assert!(worst <= -0.11390804649031287 + 1e-15, "{worst}");
        assert!(printed.trace_zeros(-100.0, 100.0, 2_000).unwrap().is_empty());
    }

    /// Eq. 11 is the `V̇ = 0` nullcline at the singular point, with the leak `g_L(V_s − V_L)`; as
    /// printed, with `V_K`, it misses `N∞(V_s)`.
    ///
    /// The printed form is off by `g_L(V_K − V_L)/(g_K(V_s − V_K))` exactly: at Fig. 7's point it gives
    /// 0.431 against `N∞ = 0.503`.
    #[test]
    fn eq_11_is_the_nullcline_and_its_printed_leak_is_not() {
        let m = MorrisLecar::FIG9;
        for i in [0.0, 100.0, 300.0, 475.0] {
            let e = m.equilibria(i, 10_000).unwrap()[0];
            let n = m.v_nullcline(e.v, i).unwrap();
            assert!((n - e.x).abs() < 5e-16, "I = {i}: {n} vs {}", e.x);
            let printed = (i - m.g_l * (e.v - m.v_k) - m.g_ca * m.m.steady(e.v) * (e.v - m.v_ca)) / (m.g_k * (e.v - m.v_k));
            let gap = m.g_l * (m.v_k - m.v_l) / (m.g_k * (e.v - m.v_k));
            assert!((printed - n - gap).abs() < 1e-16, "I = {i}");
            if i == 300.0 {
                assert!((printed - 0.431).abs() < 1e-3 && (n - 0.503).abs() < 1e-3, "{printed} {n}");
                assert!((gap + 0.072).abs() < 5e-4, "the printed form is 0.072 low: {gap}");
            }
        }
        assert_eq!(m.v_nullcline(m.v_k, 100.0), None);
        assert_eq!(MorrisLecar { g_k: 0.0, ..m }.v_nullcline(0.0, 100.0), None);
    }

    /// Eq. 15's margins are `C·trace` and `C·det/λ_N` of Eq. 9's Jacobian on the equilibrium curve,
    /// and that determinant is `λ_N I_ss'/C` — three identities, at 300 potentials in four parameter
    /// sets. Measured agreement 1.0 × 10⁻¹⁵ of the largest term.
    #[test]
    fn eq_15_is_the_trace_and_the_determinant() {
        let mut worst = 0.0_f64;
        for m in [MorrisLecar::FIG9, MorrisLecar::FIG11, MorrisLecar::FIG10A, MorrisLecar::FIG6_CA6_K12] {
            for k in -150..150 {
                let v = f64::from(k) * 0.5 + 0.125;
                let l = m.reduced_linearisation(v, m.n.steady(v));
                let (a, b) = m.oscillation_margins(v);
                let lam = m.n.rate_at(v);
                let scale = m.g_l + m.g_ca + m.g_k + m.c * lam;
                worst = worst.max((a - m.c * l.trace()).abs() / scale);
                worst = worst.max((b - m.c * l.determinant() / lam).abs() / scale);
                worst = worst.max((l.determinant() - lam * m.steady_current_slope(v) / m.c).abs() * m.c / (lam * scale));
            }
        }
        assert!(worst < 5e-15, "{worst}");
    }

    /// Where Eq. 10's rectangle holds, where it leaks, and that the trapping interval does not.
    ///
    /// Eq. 10's potentials are where Eq. 1's `V̇` vanishes with `(M, N) = (0, 1)` and `(1, 0)`. For the
    /// Fig. 9 set they bound `V` for `−40 ≤ I ≤ 300`, and there the trapping interval IS Eq. 10's; at
    /// `I = 300` the upper bound, `V_L + I/g_L` and `V_Ca` all equal 100 mV exactly. At `I = 400`,
    /// `V̇(V_max, N = 0) = +2.2 × 10⁻⁶` mV/ms (reference: 2.2194523424445833e-06; measured difference
    /// 1.4 × 10⁻¹⁵, the rounding of a difference of terms near 50 mV/ms): the rectangle
    /// leaks. On both edges of the trapping interval `V̇` points inward for every `N` in `[0, 1]`, at
    /// currents on both sides of both limits.
    ///
    /// The fourth corner, `(M, N) = (1, 1)`, matters when `g_K` is small and `I` large: with
    /// `g_Ca = 40`, `g_K = 2` and `I = 1000`, `V_min = 190`, `V_max = 350/3` and `V0 = 450` mV, and the
    /// one equilibrium sits near `V_all = 4760/44 = 108.2 mV`, below all three. An interval built from
    /// Eq. 10 and `V0` alone does not contain it, and a search inside that interval finds nothing.
    /// Each of the four corners is the lower end for some parameters and the upper end for others —
    /// `V_max` and `V_min` swap roles when `V_K > V_Ca` — and the cases below reach all eight.
    #[test]
    fn eq_10_holds_only_between_its_limits() {
        let m = MorrisLecar::FIG9;
        for i in [-40.0, 0.0, 150.0, 300.0, 400.0] {
            let (lo, hi) = m.eq10_bounds(i);
            assert!(m.field(lo, 0.0, 1.0, i)[0].abs() < 1e-14 && m.field(hi, 1.0, 0.0, i)[0].abs() < 1e-14, "I = {i}");
        }
        for i in [-40.0, 0.0, 150.0, 300.0] {
            assert_eq!(m.trapping_interval(i), m.eq10_bounds(i), "I = {i}");
        }
        assert_eq!(m.eq10_bounds(300.0).1, 100.0);
        let leak = m.reduced_field(m.eq10_bounds(400.0).1, 0.0, 400.0).0;
        assert!((leak - 2.2194523424445833e-06).abs() < 1e-14, "{leak}");
        assert_eq!(m.trapping_interval(400.0), (-26.0, 150.0));
        assert_eq!(m.trapping_interval(-100.0), (-100.0, (-100.0 + 400.0 - 100.0) / 6.0));
        assert!(m.reduced_field(m.eq10_bounds(-100.0).0, 0.0, -100.0).0 < 0.0, "below −40 the lower edge leaks too");
        let lean = MorrisLecar { g_ca: 40.0, g_k: 2.0, ..m };
        assert_eq!(lean.eq10_bounds(1000.0), (190.0, 4900.0 / 42.0));
        let (lo, hi) = lean.trapping_interval(1000.0);
        assert!((lo - 4760.0 / 44.0).abs() < 1e-13 && hi == 450.0, "({lo}, {hi})");
        let eq = lean.equilibria(1000.0, 10_000).unwrap();
        assert_eq!(eq.len(), 1, "{eq:?}");
        assert!(eq[0].v > lo && eq[0].v < lean.eq10_bounds(1000.0).1 - 1.0, "{eq:?}");
        assert!((lean.steady_current(eq[0].v) - 1000.0).abs() < 1e-11, "{eq:?}");
        let heavy = MorrisLecar { g_ca: 2.0, g_k: 40.0, ..m };
        let swapped = MorrisLecar { v_ca: -70.0, v_k: 100.0, ..lean };
        let swapped_heavy = MorrisLecar { v_ca: -70.0, v_k: 100.0, ..heavy };
        // (model, I, which corner is the lower end, which the upper): V_all is the upper end at
        // (-100 + 200 - 2800 - 1000)/44; V_max the lower end at (-100 - 2800 + 1000)/42; V_min the upper
        // end at (-100 + 4000 - 1000)/42.
        for (model, i, want) in [
            (heavy, -1000.0, (-550.0, -3700.0 / 44.0)),
            (swapped, 1000.0, (-1900.0 / 42.0, 450.0)),
            (swapped_heavy, -1000.0, (-550.0, 2900.0 / 42.0)),
        ] {
            let (lo, hi) = model.trapping_interval(i);
            assert!((lo - want.0).abs() < 1e-13 && (hi - want.1).abs() < 1e-13, "I = {i}: ({lo}, {hi}) against {want:?}");
            assert_eq!(model.equilibria(i, 10_000).unwrap().iter().filter(|e| (model.steady_current(e.v) - i).abs() < 1e-9).count(), model.equilibria(i, 10_000).unwrap().len());
            assert!(!model.equilibria(i, 10_000).unwrap().is_empty(), "I = {i}");
        }
        for (model, i) in [(m, -100.0), (m, -40.0), (m, 0.0), (m, 300.0), (m, 400.0), (m, 500.0), (lean, 1000.0), (lean, -500.0), (heavy, -1000.0), (swapped, 1000.0), (swapped_heavy, -1000.0)] {
            let (lo, hi) = model.trapping_interval(i);
            for k in 0..=10 {
                let n = f64::from(k) / 10.0;
                assert!(model.reduced_field(lo, n, i).0 >= 0.0 && model.reduced_field(hi, n, i).0 <= 0.0, "I = {i}, N = {n}");
                for mm in [0.0, 0.5, 1.0] {
                    // At the corner that sets an end, V̇ is zero in exact arithmetic; allow its rounding.
                    assert!(model.field(lo, mm, n, i)[0] >= -1e-12 && model.field(hi, mm, n, i)[0] <= 1e-12, "Eq. 1, I = {i}");
                }
            }
        }
    }

    /// Fig. 10: each parameter change against the `I = 300` control, as periods of Eq. 9's limit cycle.
    ///
    /// Reference: DOP853 over 3 s from `V = −50`, `N = N∞(−50)`, periods from the last 1.5 s: control
    /// 27.5888, `g_Ca` doubled 57.7297, `τ_K` doubled 41.7386, `g_L` halved 28.8060 ms. The module's
    /// RK4 at 0.01 ms agrees to 7.2 × 10⁻¹¹ ms (measured). The caption's claims (p. 209) follow:
    /// doubling `g_Ca` "decreases the frequency", and the figure draws a wider swing too; doubling
    /// `τ_K` "slows the oscillation"; halving `g_L` "increases amplitude" with "little or no change in
    /// frequency" (4%). And Fig. 8's caption, that the imaginary part of the eigenvalue approximates
    /// the frequency: `2π/0.24206 = 25.96` ms against 27.59. Fig. 10b (`V3 = 12`) is not a limit cycle:
    /// its equilibrium is a stable focus, `−0.000117 ± 0.2672i`. Its caption's "increases the frequency
    /// of the oscillations and lowers the amplitude" is what the transient does over the figure's 200
    /// ms: over the second half of that window its period is 24.03 ms and its swing 11.80 mV, against
    /// the control's 27.70 ms and 21.45 mV there (reference: DOP853 over 200 ms, 24.029891913634742,
    /// 11.804340677603337, 27.70031201033153 and 21.453418991480497; measured agreement 1.7 × 10⁻¹⁰).
    #[test]
    fn figure_10_periods() {
        let n0 = |m: &MorrisLecar| m.n.steady(-50.0);
        let mut got = Vec::new();
        let mut err = 0.0_f64;
        for (m, want) in [
            (MorrisLecar::FIG9, 27.588840721114877),
            (MorrisLecar::FIG10A, 57.72973795348801),
            (MorrisLecar::FIG10C, 41.73861899104295),
            (MorrisLecar::FIG10D, 28.80600530216603),
        ] {
            let (p, spread, lo, hi) = period(&run_reduced(&m, -50.0, n0(&m), 300.0, 300_000));
            err = err.max((p - want).abs());
            assert!(spread < 1e-5, "a limit cycle repeats: {spread}");
            got.push((p, hi - lo));
        }
        assert!(err < 3e-10, "{err}");
        let (control, a, c, d) = (got[0], got[1], got[2], got[3]);
        assert!(a.0 > 2.0 * control.0 && a.1 > 2.0 * control.1, "10a: slower and wider");
        assert!(c.0 > 1.4 * control.0, "10c: slower");
        assert!((d.0 - control.0).abs() < 0.05 * control.0 && d.1 > 1.5 * control.1, "10d");
        let e = MorrisLecar::FIG9.equilibria(300.0, 10_000).unwrap()[0];
        let omega = top(&MorrisLecar::FIG9.reduced_linearisation(e.v, e.x)).1;
        assert!((2.0 * std::f64::consts::PI / omega - control.0).abs() < 0.07 * control.0);
        let b = MorrisLecar::FIG10B.equilibria(300.0, 10_000).unwrap()[0];
        assert_eq!(b.kind, Kind::StableFocus);
        let (re, im) = top(&MorrisLecar::FIG10B.reduced_linearisation(b.v, b.x));
        assert!((re - (-0.00011693859158904)).abs() < 1e-15 && (im - 0.2671512881625305).abs() < 1e-15, "{re} {im}");
        let mut window = 0.0_f64;
        let mut seen = Vec::new();
        for (m, (p_want, swing_want)) in [(MorrisLecar::FIG9, (27.70031201033153, 21.453418991480497)), (MorrisLecar::FIG10B, (24.029891913634742, 11.804340677603337))] {
            let (p, _, lo, hi) = period(&run_reduced(&m, -50.0, n0(&m), 300.0, 20_000));
            window = window.max((p - p_want).abs()).max((hi - lo - swing_want).abs());
            seen.push((p, hi - lo));
        }
        assert!(window < 1e-9, "{window}");
        assert!(seen[1].0 < 0.9 * seen[0].0 && seen[1].1 < 0.6 * seen[0].1, "10b: faster and smaller, {seen:?}");
    }

    /// Fig. 11's pacemaker fires every 145 ms at `I = 35`, not at the printed `I = 50`, where Eq. 9
    /// fires every 58.6 ms; its amplitude is 35's too; and its onset is a fold at 33.30 µA/cm².
    ///
    /// Reference: DOP853 over 4 s, periods 58.6214442203586 (`I = 50`) and 145.44675094982938
    /// (`I = 35`), which RK4 at 0.01 ms matches to 7.4 × 10⁻¹² ms (measured), and the cycle's extremes
    /// over the last 2 s, −35.642509571751084 to 27.35366155979881 mV at 50 and −39.23633067261344 to
    /// 25.38535507754903 at 35, matched to 2.2 × 10⁻¹¹ mV (measured). Digitised at 600 dpi against the
    /// figure's 10 mV ticks (10.56 pixels to the millivolt), the drawn troughs are at −39.1 mV and the
    /// peaks at +24.9, stroke centres: within half a millivolt of the cycle at 35, while the troughs at
    /// 50 are 3.46 mV higher — 36.5 pixels. Spike times at 35 over
    /// the figure's first second 123.22, 268.66, 414.11, 559.56, 705.00, 850.45 and 995.90 ms. Read
    /// off Fig. 11 (p. 210): seven spikes about 145 ms apart, the first near 125 ms. The folds are the
    /// extrema of `I_ss` (reference: 33.30262757728779 at −24.4915 mV, −8.29086610218586 at
    /// −3.3738 mV; measured agreement 7.1 × 10⁻¹⁴), and between 33 and 34 µA/cm² the three
    /// equilibria become one. Eq. 1 with the caption's `λ̄_M = 0.1` does not fire at either current:
    /// its equilibrium is stable, and a run from −50 mV is within 1.2 × 10⁻¹² mV of it from 900 ms on
    /// (measured).
    #[test]
    fn figure_11_fires_at_35_not_the_printed_50() {
        let m = MorrisLecar::FIG11;
        let n0 = m.n.steady(-50.0);
        let mut err = 0.0_f64;
        let mut periods = Vec::new();
        let mut swing = 0.0_f64;
        let mut extremes = Vec::new();
        for (i, want, (lo_want, hi_want)) in [
            (MorrisLecar::FIG11_PRINTED_I, 58.6214442203586, (-35.642509571751084, 27.35366155979881)),
            (35.0, 145.44675094982938, (-39.23633067261344, 25.38535507754903)),
        ] {
            let (p, spread, lo, hi) = period(&run_reduced(&m, -50.0, n0, i, 400_000));
            err = err.max((p - want).abs());
            swing = swing.max((lo - lo_want).abs()).max((hi - hi_want).abs());
            assert!(spread < 1e-5, "{spread}");
            periods.push(p);
            extremes.push((lo, hi));
        }
        assert!(swing < 1e-10, "{swing}");
        let (page_trough, page_peak) = (-39.1, 24.9);
        let (at50, at35) = (extremes[0], extremes[1]);
        assert!((at35.0 - page_trough).abs() < 0.5 && (at35.1 - page_peak).abs() < 0.6, "35 is the page's: {at35:?}");
        assert!(at50.0 - page_trough > 3.0 && at50.1 - page_peak > 2.0, "50 is not: {at50:?}");
        assert!((periods[1] / periods[0] - 2.5).abs() < 0.05, "a factor of 2.5: {periods:?}");
        let v = run_reduced(&m, -50.0, n0, 35.0, 100_000);
        let spikes: Vec<f64> = peaks(&v).iter().map(|p| p.0).collect();
        let want = [123.22, 268.66, 414.11, 559.56, 705.0, 850.45, 995.9];
        assert_eq!(spikes.len(), 7, "{spikes:?}");
        for (s, w) in spikes.iter().zip(want) {
            assert!((s - w).abs() < 1e-9, "{spikes:?}");
        }
        for (s, page) in spikes.iter().zip([126.0, 270.0, 416.0, 561.0, 705.0, 851.0, 997.0]) {
            assert!((s - page).abs() < 4.0, "against the page: {spikes:?}");
        }
        assert!(err < 3e-11, "{err}");
        let folds = m.folds(-60.0, 40.0, 1_000).unwrap();
        assert_eq!(folds.len(), 2, "{folds:?}");
        assert!((folds[0].i - 33.30262757728779).abs() < 3e-13 && (folds[1].i - (-8.29086610218586)).abs() < 3e-13, "{folds:?}");
        assert!(folds.iter().all(|f| f.det.abs() < 3e-17), "{folds:?}");
        assert_eq!((m.equilibria(33.0, 10_000).unwrap().len(), m.equilibria(34.0, 10_000).unwrap().len()), (3, 1));
        for i in [35.0, 50.0] {
            let e = m.equilibria(i, 10_000).unwrap()[0];
            assert!(routh_hurwitz(&m.full_jacobian(e.v, m.m.steady(e.v), e.x)), "I = {i}");
            let v = run_full(&m, -50.0, i, 100_000);
            assert!(v[90_000..].iter().all(|x| (x - e.v).abs() < 1e-11), "I = {i}: Eq. 1 settles");
        }
    }

    /// Fig. 6: at `I = 50` the set printed for the broken line draws the solid one and the other way
    /// round.
    ///
    /// Full Eq. 1 from `V = −50` with both gates at steady state. Reference peak times (DOP853):
    /// `g_Ca = 6, g_K = 12` — 30.80, 61.49, 91.03, 120.04, 148.75, 177.25, 205.62 ms, all above
    /// +24.6 mV; `g_Ca = 4, g_K = 8` — 40.76, 70.04, 96.34, 120.90, 144.36, 167.11, 189.44, 211.50 ms,
    /// falling from +21.1 to +9.0 mV. RK4 at 0.01 ms peaks on the same samples, at the reference's
    /// values to 4.2 × 10⁻¹⁰ mV (measured). Read off Fig. 6 (p. 205): the solid line's first peak near
    /// 31 ms and its peaks steady near +24; the crosses' first peak near 40 ms and theirs falling to
    /// about +9.
    ///
    /// The swap does not depend on the current the caption leaves out. At every current from 30 to
    /// 120 µA/cm² in steps of 10, over 220 ms from the same start, the (6, 12) set's first peak is
    /// earlier and higher than the (4, 8) set's, and at 30 the (4, 8) set does not peak at all
    /// (reference first peaks, DOP853 as above: 43.28 ms at +26.105 mV against 68.88 ms at +19.991
    /// at 40 µA/cm², 24.64 against 30.65 ms at 60, 12.82 against 14.62 ms at 120, and the rest in
    /// the table below). In every case the module's first peak falls on the reference's sample,
    /// its voltage within 3.4 × 10⁻¹⁰ mV of the reference's (measured).
    #[test]
    fn figure_6_labels_are_swapped() {
        let solid = run_full(&MorrisLecar::FIG6_CA6_K12, -50.0, 50.0, 22_000);
        let broken = run_full(&MorrisLecar::FIG6_CA4_K8, -50.0, 50.0, 22_000);
        let (ps, pb) = (peaks(&solid), peaks(&broken));
        let ts = [30.8, 61.49, 91.03, 120.04, 148.75, 177.25, 205.62];
        let tb = [40.76, 70.04, 96.34, 120.9, 144.36, 167.11, 189.44, 211.5];
        assert_eq!((ps.len(), pb.len()), (ts.len(), tb.len()), "{ps:?} {pb:?}");
        for (p, t) in ps.iter().zip(ts).chain(pb.iter().zip(tb)) {
            assert!((p.0 - t).abs() < 1e-9, "{p:?} vs {t}");
        }
        assert!(ps.iter().all(|p| p.1 > 24.6), "sustained: {ps:?}");
        let vs = [26.833679646404335, 25.713667273866402, 25.256251013693745, 25.00384207220202, 24.84508528795985, 24.737626728055567, 24.661487308625727];
        let vb = [21.054990279014994, 18.10591205484432, 15.824141505625558, 13.876754463254324, 12.238978610346436, 10.904811792392712, 9.846729920222915, 9.022914073372936];
        let worst = ps.iter().zip(vs).chain(pb.iter().zip(vb)).fold(0.0_f64, |w, (p, v)| w.max((p.1 - v).abs()));
        assert!(worst < 2e-9, "{worst}");
        assert!(pb.windows(2).all(|w| w[1].1 < w[0].1), "damped: {pb:?}");
        assert!((ps[0].0 - 31.0).abs() < 2.0 && (pb[0].0 - 40.0).abs() < 2.0, "first peaks against the page");
        // (I, the (6, 12) set's first peak, the (4, 8) set's), from the reference.
        let scan = [
            (30.0, (93.33, 25.35670316688767), None),
            (40.0, (43.28, 26.105319186971563), Some((68.88, 19.991073511224066))),
            (50.0, (30.8, 26.833679646404335), Some((40.76, 21.054990279014994))),
            (60.0, (24.64, 27.544189632825354), Some((30.65, 22.073763974169783))),
            (70.0, (20.87, 28.238885616660195), Some((25.15, 23.055705439946944))),
            (80.0, (18.29, 28.919432840881697), Some((21.61, 24.006779969029676))),
            (90.0, (16.39, 29.58709969916148), Some((19.12, 24.931464103950308))),
            (100.0, (14.93, 30.24306736155913), Some((17.25, 25.832995027909174))),
            (110.0, (13.77, 30.888229319880875), Some((15.79, 26.71392290001153))),
            (120.0, (12.82, 31.523175686613357), Some((14.62, 27.5762476800778))),
        ];
        let mut first = 0.0_f64;
        for (i, want_a, want_b) in scan {
            let a = peaks(&run_full(&MorrisLecar::FIG6_CA6_K12, -50.0, i, 22_000));
            let b = peaks(&run_full(&MorrisLecar::FIG6_CA4_K8, -50.0, i, 22_000));
            assert!((a[0].0 - want_a.0).abs() < 1e-9, "I = {i}: {a:?}");
            first = first.max((a[0].1 - want_a.1).abs());
            match want_b {
                None => assert!(b.is_empty(), "I = {i}: {b:?}"),
                Some((t, v)) => {
                    assert!((b[0].0 - t).abs() < 1e-9, "I = {i}: {b:?}");
                    first = first.max((b[0].1 - v).abs());
                    assert!(a[0].0 < b[0].0 && a[0].1 > b[0].1, "I = {i}: (6, 12) leads, {:?} against {:?}", a[0], b[0]);
                }
            }
        }
        assert!(first < 1e-9, "{first}");
    }

    /// Both Fig. 6 sets have a STABLE equilibrium under Eq. 1 at `I = 50`; the sustained trace is a
    /// limit cycle around it, which the eigenvalues cannot see. Under Eq. 9 both are unstable foci.
    ///
    /// Reference (`NumPy`): the full Jacobians' eigenvalues are `−1.6416` and `−0.012 518 ± 0.3506i`
    /// for `(6, 12)`, `−1.4694` and `−0.012 925 ± 0.2901i` ms⁻¹ for `(4, 8)`; [`characteristic`]'s
    /// roots agree to 6.7 × 10⁻¹⁶ ms⁻¹ (measured). Started 1 mV above its equilibrium, the `(6, 12)`
    /// set decays onto it (within 2.2 × 10⁻⁹ mV by 1.5 s, measured). Started from `V = −50` with its
    /// gates at steady state it is still on the cycle from 1.4 to 1.5 s, swinging 37.9999 mV between
    /// −13.58 and +24.42 mV (reference, DOP853: 37.999878170975855; RK4 agrees to 1.9 × 10⁻⁹ mV,
    /// measured).
    #[test]
    fn figure_6_oscillates_around_a_stable_equilibrium() {
        let mut worst = 0.0_f64;
        for (m, v_eq, (real, re, im)) in [
            (MorrisLecar::FIG6_CA6_K12, 7.237218559214957, (-1.6415577113972197, -0.01251752124570056, 0.35056512912506477)),
            (MorrisLecar::FIG6_CA4_K8, 6.3853884324585595, (-1.4693961748221362, -0.01292515551294762, 0.2900723493906475)),
        ] {
            let e = m.equilibria(50.0, 10_000).unwrap();
            assert_eq!(e.len(), 1);
            assert!((e[0].v - v_eq).abs() < 4e-15, "{e:?}");
            assert_eq!(e[0].kind, Kind::UnstableFocus, "under Eq. 9");
            let j = m.full_jacobian(e[0].v, m.m.steady(e[0].v), e[0].x);
            assert!(routh_hurwitz(&j), "under Eq. 1");
            let got = roots3(&j);
            worst = worst.max((got.0 - real).abs()).max((got.1 - re).abs()).max((got.2 - im).abs());
        }
        assert!(worst < 3e-15, "{worst}");
        let m = MorrisLecar::FIG6_CA6_K12;
        let v_eq = m.equilibria(50.0, 10_000).unwrap()[0].v;
        let mut y = [v_eq + 1.0, m.m.steady(v_eq), m.n.steady(v_eq)];
        for _ in 0..150_000 {
            y = m.step_full(y[0], y[1], y[2], 50.0, H).unwrap();
        }
        assert!((y[0] - v_eq).abs() < 1e-8, "{}", y[0]);
        let v = run_full(&m, -50.0, 50.0, 150_000);
        let tail = &v[140_000..];
        let swing = tail.iter().copied().fold(f64::NEG_INFINITY, f64::max) - tail.iter().copied().fold(f64::INFINITY, f64::min);
        assert!((swing - 37.999878170975855).abs() < 1e-8, "{swing}");
    }

    /// Fig. 6's cycle does not outlast its stimulus: switched to `I = 0`, the `(6, 12)` set is back at
    /// rest within one more peak, where the fibre of Fig. 4ai "continues to oscillate (usually for
    /// not >100-200 ms) and then falls to rest" (p. 200).
    ///
    /// At `I = 0` the set has three equilibria (reference, `brentq` and `NumPy`): rest at
    /// −49.382 437 8 mV, stable under Eq. 1; a saddle at −16.409 mV; and an unstable focus at
    /// +5.827 mV, `+0.024 697 ± 0.315 538i` ms⁻¹. So rest is the only stable one. From `V = −50` at
    /// `I = 50`, the current is switched to 0 at each whole millisecond from 200 to 230 ms, which
    /// spans a whole cycle of the sustained trace (its peaks are 28.4 ms apart there), and the run
    /// goes on for 300 ms. Reference (DOP853 as for Fig. 6): after 11 of the 31 switches one peak
    /// above 0 mV follows, and after none of them two, the latest 26.47 ms after its switch; the last
    /// sample more than 1 mV from rest is 71.65 ms after a switch; and from 100 ms on nothing is
    /// further from rest than 0.0743 mV. RK4 at 0.01 ms puts every peak and the 71.65 on the
    /// reference's samples, and the 0.0743 within 1.8 × 10⁻⁸ mV (measured), which is the reference's
    /// own error there: the same RK4 written in `NumPy` gives the module's value to 10⁻¹³ at 0.01 ms
    /// and moves by only 1.5 × 10⁻⁹ mV at half the step. The worst case is the switch at
    /// 226 ms, whose run lingers longest before its one peak. The three equilibria and the focus's
    /// eigenvalues agree with the reference to 2.8 × 10⁻¹⁴, in mV and ms⁻¹ (measured).
    #[test]
    fn figure_6_does_not_outlast_its_stimulus() {
        let m = MorrisLecar::FIG6_CA6_K12;
        let eq = m.equilibria(0.0, 10_000).unwrap();
        let stable: Vec<bool> = eq.iter().map(|e| routh_hurwitz(&m.full_jacobian(e.v, m.m.steady(e.v), e.x))).collect();
        assert_eq!(stable, [true, false, false], "{eq:?}");
        let mut worst = 0.0_f64;
        for (e, want) in eq.iter().zip([-49.382437827323145, -16.409267653990838, 5.826935729379282]) {
            worst = worst.max((e.v - want).abs());
        }
        let focus = roots3(&m.full_jacobian(eq[2].v, m.m.steady(eq[2].v), eq[2].x));
        worst = worst.max((focus.1 - 0.02469675925370943).abs()).max((focus.2 - 0.3155381565059851).abs());
        assert!(worst < 1e-13, "{worst}");
        let rest = eq[0].v;
        let mut y = [-50.0, m.m.steady(-50.0), m.n.steady(-50.0)];
        let mut during = vec![y];
        for _ in 0..23_000 {
            y = m.step_full(y[0], y[1], y[2], 50.0, H).unwrap();
            during.push(y);
        }
        let (mut settle, mut late, mut after) = (0.0_f64, 0.0_f64, Vec::new());
        for off in 200..=230_usize {
            let mut y = during[off * 100];
            let mut v = vec![y[0]];
            for _ in 0..30_000 {
                y = m.step_full(y[0], y[1], y[2], 0.0, H).unwrap();
                v.push(y[0]);
            }
            let far = v.iter().rposition(|x| (x - rest).abs() > 1.0).unwrap();
            settle = settle.max(far as f64 * H);
            late = v[10_000..].iter().fold(late, |w, x| w.max((x - rest).abs()));
            after.extend(peaks(&v).into_iter().filter(|p| p.1 > 0.0).map(|p| (off, p.0)));
        }
        let want = [(200, 7.17), (201, 5.37), (202, 3.93), (203, 2.64), (204, 1.38), (205, 0.08), (226, 26.47), (227, 10.94), (228, 7.78), (229, 5.81), (230, 4.3)];
        assert_eq!(after.len(), want.len(), "{after:?}");
        for ((off, t), (w_off, w_t)) in after.iter().zip(want) {
            assert!(*off == w_off && (t - w_t).abs() < 1e-9, "{after:?}");
        }
        assert!((settle - 71.65).abs() < 1e-9, "{settle}");
        assert!((late - 0.07427316926551697).abs() < 5e-8, "{late}");
        assert!(settle < 100.0 && after.iter().all(|p| p.1 < 30.0), "no oscillation outlasts the stimulus by 100 ms");
    }

    /// The reduction to Eq. 9 changes the singular point's character at the paper's `λ̄_M`.
    ///
    /// Fig. 9's set under Eq. 1, with the caption's `λ̄_M = 0.1`: stable at every current from 0 to 600
    /// µA/cm² in steps of 25 — including 300 and 400, where Eq. 9 is an unstable focus. Along the whole
    /// equilibrium curve from −60 to +60 mV (`I_ss` from below 0 to above 600 µA/cm²), sampled every
    /// 0.025 mV, `a₂`, `a₀` and `a₂a₁ − a₀` stay positive for `λ̄_M` = 0.1, 0.2, 0.5 and 1: no Hopf
    /// point. At 1 it is close — the least `a₂a₁ − a₀` on that grid is 0.002 064 against 0.014 792 at
    /// 0.1 (reference: the same grid in `NumPy`, 0.0020640086763051776 and 0.014792228711851232;
    /// measured agreement 2.8 × 10⁻¹⁷) — and bisection in `λ̄_M` puts the first sign change at `λ̄_M =
    /// 1.020 689 5` ms⁻¹, 15.3 times `λ̄_N` (reference: `brentq` on the same least value,
    /// 1.0206895453806346; measured agreement 1.6 × 10⁻¹⁵). Above it the last condition changes sign
    /// twice, at Hopf currents 316.640 770 and 448.814 933 µA/cm² for `λ̄_M = 2` and 293.463 352 and
    /// 462.709 921 for `λ̄_M = 10`, closing in on Eq. 9's 289.65 and 465.10 (reference: `brentq` on
    /// `a₂a₁ − a₀`, 316.6407696450433, 448.8149326374532, 293.463352232487 and 462.7099206323318;
    /// measured agreement 2.8 × 10⁻¹³).
    ///
    /// A stable equilibrium can sit inside a cycle the eigenvalues do not see — Fig. 6's does — so
    /// Eq. 1 is also RUN, for 3 s from `V = −50` with both gates at steady state, the start of the
    /// Fig. 10 tests: with the Fig. 9 set at 300, 350, 400 and 450 µA/cm², where Eq. 9 is an unstable
    /// focus, and with each of Fig. 10's four sets at 300. Every run settles on its equilibrium: over
    /// the last second no sample is further from it than 4.9 × 10⁻¹² mV (measured), and the
    /// equilibria are the reference's to 1.8 × 10⁻¹⁴ mV. The reference
    /// (DOP853 at `10⁻¹²`, the same runs) is within 1.4 × 10⁻⁹ mV of its equilibria over that second,
    /// which is its own tolerance; its equilibria are the ones held here (`brentq`, e.g.
    /// −0.9016140945101736 mV at 300 and 17.057241458379945 mV for Fig. 10a).
    #[test]
    fn the_reduction_changes_the_singular_point() {
        let m = MorrisLecar::FIG9;
        let full = |m: &MorrisLecar, i: f64| {
            let e = m.equilibria(i, 10_000).unwrap()[0];
            routh_hurwitz(&m.full_jacobian(e.v, m.m.steady(e.v), e.x))
        };
        for k in 0..=24 {
            assert!(full(&m, f64::from(k) * 25.0), "I = {}", k * 25);
        }
        assert!(m.steady_current(-60.0) < 0.0 && m.steady_current(60.0) > 600.0);
        let at = |rate: f64| MorrisLecar { m: Gate { rate, ..m.m }, ..m };
        let hopf = |m: &MorrisLecar, v: f64| {
            let [a2, a1, a0] = characteristic(&m.full_jacobian(v, m.m.steady(v), m.n.steady(v)));
            (a2, a0, a2 * a1 - a0)
        };
        // The least a₂a₁ − a₀ on the curve, every 0.025 mV from −60 to +60, after checking a₂ and a₀.
        let least = |rate: f64| {
            let slow = at(rate);
            (-2_400..=2_400).fold(f64::INFINITY, |low, k| {
                let (a2, a0, h) = hopf(&slow, f64::from(k) * 0.025);
                assert!(a2 > 0.0 && a0 > 0.0, "lambda_M = {rate}, V = {}", f64::from(k) * 0.025);
                low.min(h)
            })
        };
        for rate in [0.1, 0.2, 0.5, 1.0] {
            assert!(least(rate) > 0.0, "lambda_M = {rate}: {}", least(rate));
        }
        let worst = (least(1.0) - 0.0020640086763051776).abs().max((least(0.1) - 0.014792228711851232).abs());
        assert!(worst < 1e-16, "{worst}");
        let (mut a, mut b) = (1.0, 2.0);
        assert!(least(a) > 0.0 && least(b) < 0.0);
        for _ in 0..50 {
            let mid = 0.5 * (a + b);
            if least(mid) > 0.0 { a = mid } else { b = mid }
        }
        assert!((a - 1.0206895453806346).abs() < 1e-14 && (a * 15.0 - 15.31).abs() < 5e-3, "{a}");
        let fast = at(10.0);
        assert_eq!([full(&fast, 275.0), full(&fast, 300.0), full(&fast, 400.0), full(&fast, 475.0)], [true, false, false, true]);
        assert_eq!([full(&at(2.0), 300.0), full(&at(2.0), 400.0), full(&at(2.0), 460.0)], [true, false, true]);
        let mut currents = 0.0_f64;
        for (rate, want) in [(2.0, [316.6407696450433, 448.8149326374532]), (10.0, [293.463352232487, 462.7099206323318])] {
            let slow = at(rate);
            let vs = roots(|v| hopf(&slow, v).2, -60.0, 60.0, 1_200).unwrap();
            assert_eq!(vs.len(), 2, "lambda_M = {rate}: {vs:?}");
            for (v, i) in vs.iter().zip(want) {
                currents = currents.max((slow.steady_current(*v) - i).abs());
            }
        }
        assert!(currents < 1e-12, "{currents}");
        let (mut settled, mut placed) = (0.0_f64, 0.0_f64);
        for (model, i, v_s, reduced) in [
            (m, 300.0, -0.9016140945101736, Kind::UnstableFocus),
            (m, 350.0, 1.9773355086398416, Kind::UnstableFocus),
            (m, 400.0, 5.093435120824236, Kind::UnstableFocus),
            (m, 450.0, 8.754751734146403, Kind::UnstableFocus),
            (MorrisLecar::FIG10A, 300.0, 17.057241458379945, Kind::UnstableFocus),
            (MorrisLecar::FIG10B, 300.0, 14.318221309310584, Kind::StableFocus),
            (MorrisLecar::FIG10C, 300.0, -0.9016140945101736, Kind::UnstableFocus),
            (MorrisLecar::FIG10D, 300.0, 2.1016824419373243, Kind::UnstableFocus),
        ] {
            let e = model.equilibria(i, 10_000).unwrap();
            assert!(e.len() == 1 && e[0].kind == reduced, "I = {i}: {e:?}");
            placed = placed.max((e[0].v - v_s).abs());
            let v = run_full(&model, -50.0, i, 300_000);
            settled = v[200_000..].iter().fold(settled, |w, x| w.max((x - e[0].v).abs()));
        }
        assert!(placed < 6e-14, "{placed}");
        assert!(settled < 2e-11, "Eq. 1 settles where Eq. 9 oscillates: {settled}");
    }

    /// Fig. 12b: three singular points where the map shades and one where it does not, an edge that
    /// the drawing places about 1 mmho/cm² too far right at the top — and, along the edge of the
    /// three-point region, a sliver where the upper singular point is UNSTABLE and only rest is left.
    ///
    /// Reference (`brentq`, `NumPy`, `I = 0`): at `(g_K, g_Ca) = (13, 20)` a stable node, a saddle and
    /// a stable focus, `−0.150 441 ± 0.151 475i`; at `(13.5, 20)` — shaded on the page, whose edge
    /// meets `g_Ca = 20` near `g_K = 14.4` — one node; the three-point region ends at `g_K = 13.227`
    /// for `g_Ca = 20`. The sliver: at `g_Ca = 13` the upper point's trace crosses zero at `g_K = 7.992
    /// 978` and the fold is at 8.020 246; at `g_Ca = 19.5`, 12.790 578 and 12.855 213. Bisection here
    /// finds all four to 9.6 × 10⁻¹⁴ (measured; the fold as the `g_K` at which the dip of `I_ss`
    /// between the saddle and the upper point rises through zero, reference by `SciPy`'s bounded
    /// minimiser). At `(12.8, 19.5)` — about 1.0 mmho/cm² inside the drawn edge, whose centre at `g_Ca
    /// = 19.5` is `g_K ≈ 13.8` (digitised at 600 dpi) — the three are a stable node at −49.646, a
    /// saddle at 17.018 and an unstable focus at 20.702 mV (`+0.013 648 ± 0.146 292i`), and runs
    /// started 0.01 mV either side of the focus end at rest, within 3.6 × 10⁻¹² mV by 3 s (measured).
    /// So do runs from `(8, 13)`, just inside the computed three-point region (its fold is at `g_K =
    /// 8.020`) and about 0.2 mmho/cm² right of the drawn edge (`g_K ≈ 7.8` at `g_Ca = 13`) — on the
    /// page's monostable side, which is behaviourally right there.
    #[test]
    fn figure_12b_is_not_everywhere_bistable() {
        let at = |g_k: f64, g_ca: f64| MorrisLecar { g_k, g_ca, ..MorrisLecar::FIG9 };
        let kinds = |m: &MorrisLecar| m.equilibria(0.0, 20_000).unwrap().iter().map(|e| e.kind).collect::<Vec<_>>();
        assert_eq!(kinds(&at(13.0, 20.0)), [Kind::StableNode, Kind::Saddle, Kind::StableFocus]);
        let e = *at(13.0, 20.0).equilibria(0.0, 20_000).unwrap().last().unwrap();
        let (re, im) = top(&at(13.0, 20.0).reduced_linearisation(e.v, e.x));
        let mut worst = (re - (-0.15044074611620467)).abs().max((im - 0.1514751621713861).abs());
        assert_eq!(kinds(&at(13.5, 20.0)), [Kind::StableNode]);
        assert_eq!((kinds(&at(13.22, 20.0)).len(), kinds(&at(13.235, 20.0)).len()), (3, 1));
        // The sliver's two edges, by bisection in g_K: where the upper point's trace changes sign,
        // and where the three points become one.
        let upper_trace = |m: &MorrisLecar| {
            let e = *m.equilibria(0.0, 20_000).unwrap().last().unwrap();
            m.reduced_linearisation(e.v, e.x).trace()
        };
        let mut edges = 0.0_f64;
        for (g_ca, (lo, hi), want) in [(13.0, (7.9, 8.01), (7.992977871824355, 8.02024567072917)), (19.5, (12.7, 12.8), (12.790578368592389, 12.85521332494709))] {
            let (mut a, mut b) = (lo, hi);
            assert!(upper_trace(&at(a, g_ca)) < 0.0 && upper_trace(&at(b, g_ca)) > 0.0);
            for _ in 0..40 {
                let mid = 0.5 * (a + b);
                if upper_trace(&at(mid, g_ca)) < 0.0 { a = mid } else { b = mid }
            }
            edges = edges.max((a - want.0).abs());
            // The fold: where the local minimum of I_ss that separates the saddle from the upper
            // point rises through zero. (Counting roots would find it early, once the two roots are
            // closer than one grid interval.)
            let dip = |g_k: f64| at(g_k, g_ca).folds(0.0, 40.0, 4_000).unwrap().last().unwrap().i;
            let (mut a, mut b) = (hi, hi + 0.2);
            assert!(dip(a) < 0.0 && dip(b) > 0.0);
            assert_eq!((kinds(&at(a, g_ca)).len(), kinds(&at(b, g_ca)).len()), (3, 1));
            for _ in 0..40 {
                let mid = 0.5 * (a + b);
                if dip(mid) < 0.0 { a = mid } else { b = mid }
            }
            edges = edges.max((a - want.1).abs());
        }
        assert!(edges < 5e-13, "{edges}");
        for (odd, want, (re_want, im_want)) in [
            (at(12.8, 19.5), [-49.645644932019934, 17.01782084968794, 20.70205452390926], (0.01364790087216278, 0.14629221663244857)),
            (at(8.0, 13.0), [-49.75997210422571, 17.623816758310536, 20.327002569973164], (0.01325660363099756, 0.10265590281986776)),
        ] {
            let eq = odd.equilibria(0.0, 20_000).unwrap();
            assert_eq!(eq.iter().map(|e| e.kind).collect::<Vec<_>>(), [Kind::StableNode, Kind::Saddle, Kind::UnstableFocus]);
            for (e, v) in eq.iter().zip(want) {
                worst = worst.max((e.v - v).abs());
            }
            let (re, im) = top(&odd.reduced_linearisation(eq[2].v, eq[2].x));
            worst = worst.max((re - re_want).abs()).max((im - im_want).abs());
            for dv in [0.01, -0.01] {
                let v = run_reduced(&odd, eq[2].v + dv, eq[2].x, 0.0, 300_000);
                assert!((v[300_000] - eq[0].v).abs() < 1e-10, "{}", v[300_000]);
            }
        }
        assert!(worst < 2e-13, "{worst}");
    }

    /// Fig. 12b's drawn edge against the computed one, at both ends and in between.
    ///
    /// The computed edge is the fold, where the dip of `I_ss` between the saddle and the upper point
    /// rises through zero; bisected in `g_K` at seven values of `g_Ca` (reference: `brentq` on
    /// `SciPy`'s bounded minimum, 0.5854557167184771, 1.6996171605900645, 2.814531145749783,
    /// 8.02024567072917, 9.50787931568575, 10.623637843040477 and 12.855213324947092; measured
    /// agreement 9.1 × 10⁻¹⁴). The drawn edge's centre, digitised at 600 dpi against the figure's `g_K`
    /// tick marks (47.6 pixels to the mmho/cm²), is at `g_K` = 0.82, 1.77, 2.74, 7.74, 9.42, 10.76 and
    /// 13.79 at those `g_Ca` — 3, 4.5, 6, 13, 15, 16.5 and 19.5 — so the drawing is right of the
    /// computation at 3 and 4.5, left of it from 6 to 15, and right again at 16.5 and 19.5: it crosses
    /// once between 4.5 and 6 and once between 15 and 16.5, and is 0.93 mmho/cm² right at 19.5. On the
    /// `g_K = 0` axis the drawn stroke meets the axis near `g_Ca = 1.6`, and the computed edge at `g_Ca
    /// = 2.210 886` (reference: 2.210886014618964; measured agreement 4.4 × 10⁻¹⁶). The sliver of
    /// unstable upper points begins where the fold meets a zero of the trace, the Bogdanov–Takens
    /// point: at the fold the upper point's trace is −0.001 141 for `g_Ca = 5.6` and +0.001 447 for
    /// 5.7, and bisection puts the change at `g_Ca = 5.644 095` (reference: the fold placed by `brentq`
    /// on the analytic `I_ss'`, −0.0011410024554361908, 0.0014474020201523802 and 5.6440945758495955;
    /// measured agreement 5.3 × 10⁻¹⁵ in the trace and 1.5 × 10⁻¹⁴ in `g_Ca`).
    #[test]
    fn figure_12b_edges_against_the_drawing() {
        let at = |g_k: f64, g_ca: f64| MorrisLecar { g_k, g_ca, ..MorrisLecar::FIG9 };
        let dip = |g_k: f64, g_ca: f64| *at(g_k, g_ca).folds(0.0, 40.0, 2_000).unwrap().last().unwrap();
        // The fold g_K at g_Ca, bisected from a bracket of ±0.05 around an estimate.
        let fold = |g_ca: f64, estimate: f64| {
            let (mut a, mut b) = (estimate - 0.05, estimate + 0.05);
            assert!(dip(a, g_ca).i < 0.0 && dip(b, g_ca).i > 0.0, "a bracket at g_Ca = {g_ca}");
            for _ in 0..40 {
                let mid = 0.5 * (a + b);
                if dip(mid, g_ca).i < 0.0 { a = mid } else { b = mid }
            }
            a
        };
        let drawn = [(3.0, 0.82), (4.5, 1.77), (6.0, 2.74), (13.0, 7.74), (15.0, 9.42), (16.5, 10.76), (19.5, 13.79)];
        let computed = [0.5854557167184771, 1.6996171605900645, 2.814531145749783, 8.02024567072917, 9.50787931568575, 10.623637843040477, 12.855213324947092];
        let mut worst = 0.0_f64;
        let mut side = Vec::new();
        for ((g_ca, page), want) in drawn.into_iter().zip(computed) {
            let got = fold(g_ca, want);
            worst = worst.max((got - want).abs());
            side.push(page > got);
        }
        assert!(worst < 5e-13, "{worst}");
        assert_eq!(side, [true, true, false, false, false, true, true], "which side of the computed edge the drawing is");
        assert!((drawn[6].1 - computed[6] - 0.93).abs() < 0.01);
        // The g_K = 0 axis: three points from g_Ca = 2.2109, drawn from about 1.6.
        let (mut a, mut b) = (2.0, 3.0);
        assert!(dip(0.0, a).i > 0.0 && dip(0.0, b).i < 0.0);
        for _ in 0..60 {
            let mid = 0.5 * (a + b);
            if dip(0.0, mid).i > 0.0 { a = mid } else { b = mid }
        }
        assert!((a - 2.210886014618964).abs() < 2e-15 && a - 1.6 > 0.6, "{a}");
        assert_eq!((at(0.0, 2.2).equilibria(0.0, 20_000).unwrap().len(), at(0.0, 2.22).equilibria(0.0, 20_000).unwrap().len()), (1, 3));
        // The Bogdanov–Takens point: the trace at the fold changes sign between g_Ca = 5.6 and 5.7.
        let trace_at_fold = |g_ca: f64| dip(fold(g_ca, 2.517176755007063 + 0.7433 * (g_ca - 5.6)), g_ca).trace;
        let (lo, hi) = (trace_at_fold(5.6), trace_at_fold(5.7));
        assert!((lo - (-0.0011410024554361908)).abs() < 2e-14 && (hi - 0.0014474020201523802).abs() < 2e-14, "{lo} {hi}");
        let (mut a, mut b) = (5.6, 5.7);
        for _ in 0..40 {
            let mid = 0.5 * (a + b);
            if trace_at_fold(mid) < 0.0 { a = mid } else { b = mid }
        }
        assert!((a - 5.6440945758495955).abs() < 1e-13, "{a}");
    }

    /// Fig. 12a (`I = 300`) shades neither all of Eq. 9's instability nor its damped oscillation.
    ///
    /// The Fig. 9 control, `(g_K, g_Ca) = (8, 4)`, sits about 0.7 mmho/cm² above the lower edge of the
    /// band drawn as damped (digitised at 600 dpi: the edge is at `g_Ca = 3.3` there) and is Fig. 7's
    /// unstable focus. `(5, 3)`, unshaded below the tip of the band, is an unstable focus too, `+0.010
    /// 825 ± 0.160 936i` ms⁻¹. The band's lower edge, digitised at `g_K` = 9, 10, …, 20, is at `g_Ca` =
    /// 3.79, 4.80, 5.59, 6.15, 6.62, 7.07, 7.48, 7.89, 8.29, 8.47, 8.86 and 9.05 (stroke centres; the
    /// figure's own ticks give 46.7 pixels to the mmho/cm² along `g_K`); at each of the 72 integer
    /// points at least 0.5 under it the one equilibrium is a stable FOCUS — the caption's unshaded
    /// "stable nodes" oscillate, damped: `−0.164 859 ± 0.285 020i` at (20, 1), `−0.133 208 ± 0.281
    /// 001i` at (15, 2) and `−0.093 483 ± 0.310 616i` at (20, 5) (reference, `NumPy`:
    /// −0.16485924106910063, 0.2850204142648936, −0.13320834223591116, 0.28100131071909745,
    /// −0.09348271777113232, 0.31061621399565803, and 0.0108246256260499, 0.16093615528544003 at (5,
    /// 3); measured agreement 1.1 × 10⁻¹⁶).
    #[test]
    fn figure_12a_is_a_sketch() {
        let at = |g_k: f64, g_ca: f64| MorrisLecar { g_k, g_ca, ..MorrisLecar::FIG9 };
        let one = |m: &MorrisLecar| {
            let eq = m.equilibria(300.0, 10_000).unwrap();
            assert_eq!(eq.len(), 1, "{eq:?}");
            (eq[0].kind, top(&m.reduced_linearisation(eq[0].v, eq[0].x)))
        };
        assert_eq!(one(&at(8.0, 4.0)).0, Kind::UnstableFocus);
        let edge = [3.79, 4.8, 5.59, 6.15, 6.62, 7.07, 7.48, 7.89, 8.29, 8.47, 8.86, 9.05];
        let mut points = 0;
        for (g_k, top_edge) in (9..=20).zip(edge) {
            for g_ca in (1..).take_while(|&g| f64::from(g) <= top_edge - 0.5) {
                assert_eq!(one(&at(f64::from(g_k), f64::from(g_ca))).0, Kind::StableFocus, "({g_k}, {g_ca})");
                points += 1;
            }
        }
        assert_eq!(points, 72);
        let mut worst = 0.0_f64;
        for ((g_k, g_ca), kind, (re, im)) in [
            ((20.0, 1.0), Kind::StableFocus, (-0.16485924106910063, 0.2850204142648936)),
            ((15.0, 2.0), Kind::StableFocus, (-0.13320834223591116, 0.28100131071909745)),
            ((20.0, 5.0), Kind::StableFocus, (-0.09348271777113232, 0.31061621399565803)),
            ((5.0, 3.0), Kind::UnstableFocus, (0.0108246256260499, 0.16093615528544003)),
        ] {
            let (k, (r, i)) = one(&at(g_k, g_ca));
            assert_eq!(k, kind, "({g_k}, {g_ca})");
            worst = worst.max((r - re).abs()).max((i - im).abs());
        }
        assert!(worst < 5e-16, "{worst}");
    }

    /// The Routh–Hurwitz verdict at each of its three boundaries, and against an independent root
    /// computation on 3000 pseudo-random matrices.
    ///
    /// The independent path: the characteristic polynomial by Faddeev–LeVerrier (traces of powers),
    /// a real root by bisection, and the remaining quadratic's roots by its own two signs. Matrices
    /// within `10⁻⁹` of a boundary are skipped; none of the rest disagrees. [`characteristic`]'s
    /// coefficients agree with Faddeev–LeVerrier's to 9.0 × 10⁻¹⁵ on entries in `[−2, 2]` (measured).
    #[test]
    fn routh_hurwitz_is_the_sign_of_every_root() {
        let d = |a: f64, b: f64, c: f64| [[a, 0.0, 0.0], [0.0, b, 0.0], [0.0, 0.0, c]];
        assert!(routh_hurwitz(&d(-1.0, -2.0, -3.0)));
        assert!(!routh_hurwitz(&d(-1.0, -2.0, 0.0)), "a zero root: a0 = 0");
        assert!(!routh_hurwitz(&d(-1.0, -2.0, 3.0)));
        assert!(!routh_hurwitz(&[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]), "±i: a2 a1 = a0");
        assert!(routh_hurwitz(&[[-0.001, -1.0, 0.0], [1.0, -0.001, 0.0], [0.0, 0.0, -1.0]]));
        // p³ − p² − 2p + 1: a0 > 0 and a2 a1 = 2 > 1, but a2 < 0 — two roots in the right half-plane.
        assert!(!routh_hurwitz(&[[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 2.0, 1.0]]));
        let mut seed = 0x2545_f491_4f6c_dd1d_u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f64 / (1u64 << 53) as f64 * 4.0 - 2.0
        };
        let (mut compared, mut coefficients) = (0, 0.0_f64);
        for _ in 0..3_000 {
            let j: [[f64; 3]; 3] = core::array::from_fn(|_| core::array::from_fn(|_| next()));
            let mul = |a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]| -> [[f64; 3]; 3] {
                core::array::from_fn(|r| core::array::from_fn(|c| (0..3).map(|k| a[r][k] * b[k][c]).sum()))
            };
            let tr = |a: &[[f64; 3]; 3]| a[0][0] + a[1][1] + a[2][2];
            let (j2, j3) = (mul(&j, &j), mul(&mul(&j, &j), &j));
            let c2 = -tr(&j);
            let c1 = (tr(&j) * tr(&j) - tr(&j2)) / 2.0;
            let c0 = -(tr(&j).powi(3) - 3.0 * tr(&j) * tr(&j2) + 2.0 * tr(&j3)) / 6.0;
            for (got, want) in characteristic(&j).iter().zip([c2, c1, c0]) {
                coefficients = coefficients.max((got - want).abs());
            }
            if c0.abs() < 1e-9 || (c2 * c1 - c0).abs() < 1e-9 || c2.abs() < 1e-9 {
                continue;
            }
            let p = |x: f64| ((x + c2) * x + c1) * x + c0;
            let bound = 1.0 + c2.abs().max(c1.abs()).max(c0.abs());
            let (mut lo, mut hi) = (-bound, bound);
            for _ in 0..200 {
                let mid = 0.5 * (lo + hi);
                if p(mid) < 0.0 { lo = mid } else { hi = mid }
            }
            let r = 0.5 * (lo + hi);
            // p(x) = (x − r)(x² + b x + c): the quadratic's roots are in the left half-plane iff b, c > 0.
            let (b, c) = (c2 + r, c1 + r * (c2 + r));
            if b.abs() < 1e-7 || c.abs() < 1e-7 || r.abs() < 1e-7 {
                continue;
            }
            assert_eq!(routh_hurwitz(&j), r < 0.0 && b > 0.0 && c > 0.0, "{j:?}");
            compared += 1;
        }
        assert!(compared > 2_900, "{compared}");
        assert!(coefficients < 3e-14, "{coefficients}");
    }

    /// Each kind at the boundary it shares with its neighbours, from exact matrices.
    ///
    /// A repeated real root (`trace² = 4 det`) is a node; `det = 0` is degenerate before anything
    /// else; `trace = 0 < det` is a centre.
    #[test]
    fn each_kind_at_its_boundary() {
        let k = |a: f64, b: f64, c: f64, d: f64| Kind::of(&Linearisation { jacobian: [[a, b], [c, d]] });
        assert_eq!(k(-1.0, 0.0, 0.0, -2.0), Kind::StableNode);
        assert_eq!(k(1.0, 0.0, 0.0, 2.0), Kind::UnstableNode);
        assert_eq!(k(-1.0, -4.0, 1.0, -1.0), Kind::StableFocus);
        assert_eq!(k(1.0, -4.0, 1.0, 1.0), Kind::UnstableFocus);
        assert_eq!(k(1.0, 0.0, 0.0, -1.0), Kind::Saddle);
        assert_eq!(k(0.0, 1.0, 1.0, 0.0), Kind::Saddle);
        assert_eq!(k(0.0, -1.0, 1.0, 0.0), Kind::Centre);
        assert_eq!(k(-1.0, 0.0, 0.0, 0.0), Kind::Degenerate);
        assert_eq!(k(0.0, 0.0, 0.0, 0.0), Kind::Degenerate);
        assert_eq!(k(-1.0, 0.0, 0.0, -1.0), Kind::StableNode, "trace² = 4 det: a repeated real root");
        assert_eq!(k(1.0, 0.0, 0.0, 1.0), Kind::UnstableNode);
    }

    /// Every Jacobian is the derivative of its field, by central differences at points away from
    /// every equilibrium, with parameters chosen so that no two coincide.
    ///
    /// Measured worst disagreement 4.3 × 10⁻¹¹ of the largest entry.
    #[test]
    fn the_jacobians_are_the_derivatives_of_their_fields() {
        let odd = MorrisLecar {
            c: 17.0,
            g_l: 1.7,
            g_ca: 5.3,
            g_k: 9.1,
            v_l: -48.0,
            v_ca: 110.0,
            v_k: -75.0,
            m: Gate { v_half: 7.0, slope: 13.0, rate: 0.3 },
            n: Gate { v_half: -3.0, slope: 16.0, rate: 0.07 },
        };
        let mut worst = 0.0_f64;
        for m in [MorrisLecar::FIG9, MorrisLecar::FIG11, odd] {
            for (v, mm, n) in [(-40.0, 0.2, 0.3), (-5.0, 0.7, 0.1), (20.0, 0.4, 0.8), (35.0, 0.95, 0.5)] {
                let f2 = |x: [f64; 2]| {
                    let (a, b) = m.reduced_field(x[0], x[1], 123.0);
                    [a, b]
                };
                worst = worst.max(fd_error(&f2, [v, n], m.reduced_linearisation(v, n).jacobian));
                let f3 = |x: [f64; 3]| m.field(x[0], x[1], x[2], 123.0);
                worst = worst.max(fd_error(&f3, [v, mm, n], m.full_jacobian(v, mm, n)));
            }
        }
        let ca = SingleConductance { drive: Drive::Electrodiffusion { ca_in: 3.0, ca_out: 100.0 }, ..SingleConductance::FIG3B };
        for s in [SingleConductance::FIG2, SingleConductance::FIG3B, ca] {
            for (v, mu) in [(-40.0, 0.2), (-5.0, 0.7), (20.0, 0.4), (35.0, 0.95)] {
                let f = |x: [f64; 2]| {
                    let (a, b) = s.field(x[0], x[1], 17.0);
                    [a, b]
                };
                worst = worst.max(fd_error(&f, [v, mu], s.linearisation(v, mu).jacobian));
            }
        }
        assert!(worst < 2e-10, "{worst}");
    }

    /// The integrator is fourth order: halving the step divides the error by sixteen.
    ///
    /// Eq. 9 with the Fig. 9 set at `I = 300` from `V = −50`, `N = N∞(−50)`, to 20 ms, with steps of
    /// 0.1, 0.05 and 0.025 ms against a run at 1/1280 ms. Measured ratios 16.8 and 16.4; the classic
    /// slip, a first midpoint slope taken a full step out, measures 2.0 — first order.
    #[test]
    fn the_integrator_is_fourth_order() {
        let m = MorrisLecar::FIG9;
        let run = |h: f64, steps: usize| {
            let (mut v, mut n) = (-50.0, m.n.steady(-50.0));
            for _ in 0..steps {
                (v, n) = m.step_reduced(v, n, 300.0, h).unwrap();
            }
            (v, n)
        };
        let (vr, nr) = run(1.0 / 1280.0, 25_600);
        let err: Vec<f64> = [(0.1, 200), (0.05, 400), (0.025, 800)]
            .iter()
            .map(|&(h, k)| {
                let (v, n) = run(h, k);
                (v - vr).abs() + 100.0 * (n - nr).abs()
            })
            .collect();
        for w in err.windows(2) {
            let ratio = w[0] / w[1];
            assert!((14.0..18.0).contains(&ratio), "{err:?}: {ratio}");
        }
    }

    /// The equilibrium searches find what the closed forms say is there, including at the edges of
    /// their windows.
    ///
    /// A passive membrane — no conductance but the leak — has one equilibrium, `V_L + I/g_L`, which is
    /// also the whole of its closed-form window; one sample interval is enough to find the all-K
    /// system's one point; and at every equilibrium the steady current is the current asked for.
    #[test]
    fn the_searches_find_what_the_closed_forms_place() {
        let passive = MorrisLecar { g_ca: 0.0, g_k: 0.0, ..MorrisLecar::FIG9 };
        // A root exactly on a grid point: at I = 0 the window is [−51, −49] and the middle sample is
        // −50, where I_ss is 0 exactly. It is found once, at −50, and not moved to a neighbour.
        assert_eq!(passive.equilibria(0.0, 2).unwrap().iter().map(|e| e.v).collect::<Vec<_>>(), [-50.0]);
        // With g_L = 3, V0 = V_L + I/g_L is rounded, and I_ss(V0) lands a few 10⁻¹⁵ either side of
        // zero; the 1 mV margins on BOTH sides of the (zero-width) window keep every root bracketed.
        let third = MorrisLecar { g_l: 3.0, ..passive };
        for k in 1..=12 {
            let i = f64::from(k);
            let e = third.equilibria(i, 3).unwrap();
            assert!(e.len() == 1 && (e[0].v - (-50.0 + i / 3.0)).abs() < 1e-13, "I = {i}: {e:?}");
        }
        for i in [-30.0, 0.0, 70.0] {
            let e = passive.equilibria(i, 3).unwrap();
            assert_eq!(e.len(), 1, "{e:?}");
            assert!((e[0].v - (-50.0 + i / 2.0)).abs() < 1e-13, "{e:?}");
            let s = SingleConductance { g: 0.0, ..SingleConductance::FIG2 };
            let e = s.equilibria(i, 3).unwrap();
            assert!(e.len() == 1 && (e[0].v - (-50.0 + i / 3.0)).abs() < 1e-13, "{e:?}");
        }
        let one = SingleConductance::FIG2.equilibria(100.0, 1).unwrap();
        assert!(one.len() == 1 && (one[0].v - (-22.693782729085434)).abs() < 3e-14, "{one:?}");
        for m in [MorrisLecar::FIG9, MorrisLecar::FIG11] {
            for i in [-200.0, -20.0, 30.0, 33.0, 250.0, 900.0] {
                for e in m.equilibria(i, 10_000).unwrap() {
                    assert!((m.steady_current(e.v) - i).abs() < 5e-15 * (1.0 + i.abs()), "I = {i}: {e:?}");
                    assert!(m.reduced_field(e.v, e.x, i).0.abs() < 5e-14);
                    let f = m.field(e.v, m.m.steady(e.v), e.x, i);
                    assert!(f.iter().all(|x| x.abs() < 5e-14), "Eq. 1 rests there too: {f:?}");
                }
            }
        }
        for i in [-100.0, 0.0, 500.0] {
            for e in SingleConductance::FIG3B.equilibria(i, 20_000).unwrap() {
                assert!((SingleConductance::FIG3B.steady_current(e.v) - i).abs() < 1e-13 * (1.0 + i.abs()));
                assert!(SingleConductance::FIG3B.field(e.v, e.x, i).0.abs() < 1e-14);
            }
        }
        // I_ss' against a central difference, for both systems.
        for v in [-60.0, -20.0, 0.5, 30.0] {
            let h = 1e-5;
            let m = MorrisLecar::FIG11;
            let fd = (m.steady_current(v + h) - m.steady_current(v - h)) / (2.0 * h);
            assert!((m.steady_current_slope(v) - fd).abs() < 1e-7, "{v}");
            let s = SingleConductance::FIG3B;
            let fd = (s.steady_current(v + h) - s.steady_current(v - h)) / (2.0 * h);
            assert!((s.steady_current_slope(v) - fd).abs() < 1e-7, "{v}");
        }
    }

    /// Every refusal names what it refused, and a default never stands in for a bad value.
    #[test]
    fn every_refusal_names_what_it_refused() {
        let m = MorrisLecar::FIG9;
        let bad_models: Vec<(MorrisLecar, &str)> = vec![
            (MorrisLecar { c: 0.0, ..m }, "C = 0 must be finite and positive"),
            (MorrisLecar { g_l: -1.0, ..m }, "g_L = -1 must be finite and positive"),
            (MorrisLecar { g_ca: -1.0, ..m }, "g_Ca = -1 must be finite and non-negative"),
            (MorrisLecar { g_ca: f64::INFINITY, ..m }, "g_Ca = inf must be finite and non-negative"),
            (MorrisLecar { g_k: f64::NAN, ..m }, "g_K = NaN must be finite and non-negative"),
            (MorrisLecar { v_l: f64::NAN, ..m }, "V_L = NaN is not finite"),
            (MorrisLecar { v_ca: f64::INFINITY, ..m }, "V_Ca = inf is not finite"),
            (MorrisLecar { v_k: f64::NEG_INFINITY, ..m }, "V_K = -inf is not finite"),
            (MorrisLecar { m: Gate { v_half: f64::NAN, ..m.m }, ..m }, "V1 = NaN is not finite"),
            (MorrisLecar { m: Gate { slope: 0.0, ..m.m }, ..m }, "V2 = 0 is a slope Eq. 2 divides by"),
            (MorrisLecar { m: Gate { slope: f64::INFINITY, ..m.m }, ..m }, "V2 = inf is not finite"),
            (MorrisLecar { m: Gate { rate: 0.0, ..m.m }, ..m }, "lambda_M = 0 must be finite and positive"),
            (MorrisLecar { n: Gate { v_half: f64::INFINITY, ..m.n }, ..m }, "V3 = inf is not finite"),
            (MorrisLecar { n: Gate { slope: -0.0, ..m.n }, ..m }, "V4 = 0 is a slope Eq. 2 divides by"),
            (MorrisLecar { n: Gate { rate: f64::NAN, ..m.n }, ..m }, "lambda_N = NaN must be finite and positive"),
        ];
        for (bad, want) in bad_models {
            assert_eq!(bad.check().unwrap_err().to_string(), want);
            assert_eq!(bad.equilibria(0.0, 10).unwrap_err().to_string(), want);
            assert_eq!(bad.step_reduced(0.0, 0.5, 0.0, 0.1).unwrap_err().to_string(), want);
            assert_eq!(bad.step_full(0.0, 0.5, 0.5, 0.0, 0.1).unwrap_err().to_string(), want);
            assert_eq!(bad.trace_zeros(-10.0, 10.0, 10).unwrap_err().to_string(), want);
            assert_eq!(bad.folds(-10.0, 10.0, 10).unwrap_err().to_string(), want);
        }
        assert_eq!(m.check(), Ok(()));
        assert!(MorrisLecar { g_ca: 0.0, g_k: 0.0, ..m }.check().is_ok(), "zero conductances are allowed");
        assert!(MorrisLecar { m: Gate { slope: -15.0, ..m.m }, ..m }.check().is_ok(), "a negative slope is allowed");
        let calls: Vec<(Result<(), MorrisLecarError>, &str)> = vec![
            (m.step_reduced(0.0, 0.5, 0.0, 0.0).map(|_| ()), "h = 0 must be finite and positive"),
            (m.step_reduced(f64::NAN, 0.5, 0.0, 0.1).map(|_| ()), "V = NaN is not finite"),
            (m.step_reduced(0.0, f64::INFINITY, 0.0, 0.1).map(|_| ()), "N = inf is not finite"),
            (m.step_reduced(0.0, 0.5, f64::NAN, 0.1).map(|_| ()), "I = NaN is not finite"),
            (m.step_full(0.0, 0.5, 0.5, 0.0, f64::INFINITY).map(|_| ()), "h = inf must be finite and positive"),
            (m.step_full(f64::NAN, 0.5, 0.5, 0.0, 0.1).map(|_| ()), "V = NaN is not finite"),
            (m.step_full(0.0, f64::NAN, 0.5, 0.0, 0.1).map(|_| ()), "M = NaN is not finite"),
            (m.step_full(0.0, 0.5, f64::NAN, 0.0, 0.1).map(|_| ()), "N = NaN is not finite"),
            (m.step_full(0.0, 0.5, 0.5, f64::INFINITY, 0.1).map(|_| ()), "I = inf is not finite"),
            (m.equilibria(f64::NAN, 10).map(|_| ()), "I = NaN is not finite"),
            (m.equilibria(0.0, 0).map(|_| ()), "a search over 0 samples brackets nothing"),
            (m.trace_zeros(1.0, 1.0, 10).map(|_| ()), "the window [1, 1] mV is not a finite, increasing interval"),
            (m.trace_zeros(2.0, 1.0, 10).map(|_| ()), "the window [2, 1] mV is not a finite, increasing interval"),
            (m.trace_zeros(f64::NEG_INFINITY, 1.0, 10).map(|_| ()), "the window [-inf, 1] mV is not a finite, increasing interval"),
            (m.folds(0.0, f64::INFINITY, 10).map(|_| ()), "the window [0, inf] mV is not a finite, increasing interval"),
            (m.folds(f64::NAN, 1.0, 10).map(|_| ()), "the window [NaN, 1] mV is not a finite, increasing interval"),
            (m.folds(-10.0, 10.0, 0).map(|_| ()), "a search over 0 samples brackets nothing"),
        ];
        for (got, want) in calls {
            assert_eq!(got.unwrap_err().to_string(), want);
        }
        let s = SingleConductance::FIG2;
        let e7 = |ca_in: f64, ca_out: f64| SingleConductance { drive: Drive::Electrodiffusion { ca_in, ca_out }, ..SingleConductance::FIG3B };
        let bad_singles: Vec<(SingleConductance, &str)> = vec![
            (SingleConductance { c: f64::NAN, ..s }, "C = NaN must be finite and positive"),
            (SingleConductance { g_l: 0.0, ..s }, "g_L = 0 must be finite and positive"),
            (SingleConductance { v_l: f64::INFINITY, ..s }, "V_L = inf is not finite"),
            (SingleConductance { g: -8.0, ..s }, "g = -8 must be finite and non-negative"),
            (SingleConductance { g: f64::INFINITY, ..s }, "g = inf must be finite and non-negative"),
            (SingleConductance { drive: Drive::Linear { v_rev: f64::NAN }, ..s }, "V_i = NaN is not finite"),
            (SingleConductance { gate: Gate { v_half: f64::NAN, ..s.gate }, ..s }, "V_half = NaN is not finite"),
            (SingleConductance { gate: Gate { slope: 0.0, ..s.gate }, ..s }, "V_slope = 0 is a slope Eq. 2 divides by"),
            (SingleConductance { gate: Gate { rate: -0.1, ..s.gate }, ..s }, "lambda_bar = -0.1 must be finite and positive"),
            (e7(100.0, 100.0), "[Ca]_i = 100 mM, [Ca]_o = 100 mM: Eq. 7 is used here only for 0 <= [Ca]_i < [Ca]_o, where its drive is monotone"),
            (e7(-1.0, 100.0), "[Ca]_i = -1 mM, [Ca]_o = 100 mM: Eq. 7 is used here only for 0 <= [Ca]_i < [Ca]_o, where its drive is monotone"),
            (e7(1.0, f64::INFINITY), "[Ca]_i = 1 mM, [Ca]_o = inf mM: Eq. 7 is used here only for 0 <= [Ca]_i < [Ca]_o, where its drive is monotone"),
            (e7(f64::NAN, 100.0), "[Ca]_i = NaN mM, [Ca]_o = 100 mM: Eq. 7 is used here only for 0 <= [Ca]_i < [Ca]_o, where its drive is monotone"),
        ];
        for (bad, want) in bad_singles {
            assert_eq!(bad.check().unwrap_err().to_string(), want);
            assert_eq!(bad.equilibria(0.0, 10).unwrap_err().to_string(), want);
            assert_eq!(bad.step(0.0, 0.5, 0.0, 0.1).unwrap_err().to_string(), want);
            assert_eq!(bad.v_nullcline(0.5, 0.0).unwrap_err().to_string(), want);
        }
        assert!(e7(0.0, 100.0).check().is_ok() && e7(99.0, 100.0).check().is_ok());
        let calls: Vec<(Result<(), MorrisLecarError>, &str)> = vec![
            (s.step(0.0, 0.5, 0.0, -1.0).map(|_| ()), "h = -1 must be finite and positive"),
            (s.step(f64::INFINITY, 0.5, 0.0, 0.1).map(|_| ()), "V = inf is not finite"),
            (s.step(0.0, f64::NAN, 0.0, 0.1).map(|_| ()), "mu = NaN is not finite"),
            (s.step(0.0, 0.5, f64::NAN, 0.1).map(|_| ()), "I = NaN is not finite"),
            (s.equilibria(f64::INFINITY, 10).map(|_| ()), "I = inf is not finite"),
            (s.equilibria(0.0, 0).map(|_| ()), "a search over 0 samples brackets nothing"),
            (s.v_nullcline(1.5, 0.0).map(|_| ()), "mu = 1.5 is outside [0, 1]"),
            (s.v_nullcline(-0.25, 0.0).map(|_| ()), "mu = -0.25 is outside [0, 1]"),
            (s.v_nullcline(f64::NAN, 0.0).map(|_| ()), "mu = NaN is not finite"),
            (s.v_nullcline(0.5, f64::NAN).map(|_| ()), "I = NaN is not finite"),
            (
                SingleConductance::FIG3B.v_nullcline(0.5, 0.0).map(|_| ()),
                "Eq. 4a is the nullcline of Eq. 3's linear drive; an electrodiffusion drive has no closed-form V(mu)",
            ),
        ];
        for (got, want) in calls {
            assert_eq!(got.unwrap_err().to_string(), want);
        }
        assert!(s.v_nullcline(0.0, 0.0).is_ok() && s.v_nullcline(1.0, 0.0).is_ok(), "both ends of [0, 1] are allowed");
    }

    /// Every figure's parameters, pinned by value against its caption.
    ///
    /// A transcription slip in most of these moves no count or kind checked elsewhere, so the values
    /// themselves are the check. `V2` is +15 in Fig. 9's set and printed −15; `I` is printed 50 in
    /// Fig. 11; `V_L` is printed 50 in Fig. 5a.
    #[test]
    fn the_figures_parameters_are_the_papers() {
        let row = |m: MorrisLecar| [m.c, m.g_l, m.g_ca, m.g_k, m.v_l, m.v_ca, m.v_k, m.m.v_half, m.m.slope, m.m.rate, m.n.v_half, m.n.slope, m.n.rate];
        let third = 1.0 / 15.0;
        assert_eq!(row(MorrisLecar::FIG9), [20.0, 2.0, 4.0, 8.0, -50.0, 100.0, -70.0, 10.0, 15.0, 0.1, -1.0, 14.5, third]);
        assert_eq!(row(MorrisLecar::FIG10A), [20.0, 2.0, 8.0, 8.0, -50.0, 100.0, -70.0, 10.0, 15.0, 0.1, -1.0, 14.5, third]);
        assert_eq!(row(MorrisLecar::FIG10B), [20.0, 2.0, 4.0, 8.0, -50.0, 100.0, -70.0, 10.0, 15.0, 0.1, 12.0, 14.5, third]);
        assert_eq!(row(MorrisLecar::FIG10C), [20.0, 2.0, 4.0, 8.0, -50.0, 100.0, -70.0, 10.0, 15.0, 0.1, -1.0, 14.5, 1.0 / 30.0]);
        assert_eq!(row(MorrisLecar::FIG10D), [20.0, 1.0, 4.0, 8.0, -50.0, 100.0, -70.0, 10.0, 15.0, 0.1, -1.0, 14.5, third]);
        assert_eq!(row(MorrisLecar::FIG11), [20.0, 2.0, 4.0, 8.0, -50.0, 100.0, -70.0, -1.0, 15.0, 0.1, 10.0, 14.5, third]);
        assert_eq!(row(MorrisLecar::FIG6_CA6_K12), [20.0, 2.0, 6.0, 12.0, -50.0, 100.0, -70.0, 0.0, 15.0, 1.0, 10.0, 10.0, 0.1]);
        assert_eq!(row(MorrisLecar::FIG6_CA4_K8), [20.0, 2.0, 4.0, 8.0, -50.0, 100.0, -70.0, 0.0, 15.0, 1.0, 10.0, 10.0, 0.1]);
        assert_eq!((MorrisLecar::FIG9_PRINTED_V2, MorrisLecar::FIG11_PRINTED_I), (-15.0, 50.0));
        let single = |s: SingleConductance| (s.c, s.g_l, s.v_l, s.g, s.drive, s.gate.v_half, s.gate.slope, s.gate.rate);
        assert_eq!(single(SingleConductance::FIG2), (20.0, 3.0, -50.0, 8.0, Drive::Linear { v_rev: -70.0 }, -1.0, 14.5, third));
        assert_eq!(
            single(SingleConductance::FIG3B),
            (20.0, 2.0, -35.0, 40.0, Drive::Electrodiffusion { ca_in: 0.0, ca_out: 100.0 }, 10.0, 15.0, 0.1)
        );
        assert_eq!(SingleConductance::FIG5A_PRINTED_V_L, 50.0);
    }

    /// Eq. 3's all-Ca system with its LINEAR driving force: the plateau p. 202 calls "unrealistically
    /// high", and the three singular points p. 201 allows when `V_Ca > V_L`.
    ///
    /// The Fig. 3b caption prints no `V_Ca`, because its drive is Eq. 7's; with the 100 mV of every
    /// other caption, the linear system has ONE singular point at `I = 0`, a stable node at
    /// +93.571 mV — within 10⁻⁴ mV of Eq. 5's `V(1) = (I − 70 + 4000)/42`, because `M∞` there is
    /// 0.99999 — so the membrane cannot rest at all; at `I = 15`, 25 and 50 it sits at +93.9, +94.2
    /// and +94.8 mV, more than 59 mV above Fig. 3b's plateaus with Eq. 7's drive and far above the
    /// fibre's "about +20 mV" (p. 197). Reference (`brentq`): 93.5713399275728, 93.9284916026632,
    /// 94.16659236777508, 94.76184313499334; measured agreement 1.4 × 10⁻¹⁴ mV. With Fig. 9's calcium
    /// gate, `g_Ca = 4`, `g_L = 2` and `V_L = −50`, the same linear system has three singular points
    /// at `I = 0` — a stable node at −49.898, a saddle at 2.516 and a stable node at 49.836 mV
    /// (reference: −49.898088342154075, 2.515779591076898, 49.83577575916638), all between Eq. 4a's
    /// ends `V(0) = −50` and `V(1) = +50` — and they are Eq. 1's equilibria with `g_K = 0`: the other
    /// search, on another grid, finds the same doubles (measured).
    #[test]
    fn the_linear_all_ca_system() {
        let lin = SingleConductance { drive: Drive::Linear { v_rev: 100.0 }, ..SingleConductance::FIG3B };
        let mut worst = 0.0_f64;
        for (i, want) in [(0.0, 93.5713399275728), (15.0, 93.9284916026632), (25.0, 94.16659236777508), (50.0, 94.76184313499334)] {
            let eq = lin.equilibria(i, 20_000).unwrap();
            assert_eq!(eq.len(), 1, "I = {i}: {eq:?}");
            assert_eq!(eq[0].kind, Kind::StableNode, "I = {i}");
            worst = worst.max((eq[0].v - want).abs());
            let v1 = lin.v_nullcline(1.0, i).unwrap();
            assert!((v1 - (i - 70.0 + 4000.0) / 42.0).abs() < 1e-13 && (eq[0].v - v1).abs() < 1e-4, "Eq. 5 at I = {i}: {v1}");
            if i > 0.0 {
                let plateau = SingleConductance::FIG3B.equilibria(i, 20_000).unwrap().last().unwrap().v;
                assert!(eq[0].v - plateau > 59.0, "I = {i}: {} against {plateau}", eq[0].v);
            }
        }
        let ca = SingleConductance { c: 20.0, g_l: 2.0, v_l: -50.0, g: 4.0, drive: Drive::Linear { v_rev: 100.0 }, gate: MorrisLecar::FIG9.m };
        let eq = ca.equilibria(0.0, 20_000).unwrap();
        assert_eq!(eq.iter().map(|e| e.kind).collect::<Vec<_>>(), [Kind::StableNode, Kind::Saddle, Kind::StableNode]);
        for (e, want) in eq.iter().zip([-49.898088342154075, 2.515779591076898, 49.83577575916638]) {
            worst = worst.max((e.v - want).abs());
        }
        assert!(worst < 5e-14, "{worst}");
        let (v0, v1) = (ca.v_nullcline(0.0, 0.0).unwrap(), ca.v_nullcline(1.0, 0.0).unwrap());
        assert_eq!((v0, v1), (-50.0, 50.0));
        assert!(eq.iter().all(|e| v0 < e.v && e.v < v1), "{eq:?}");
        let full = MorrisLecar { g_k: 0.0, ..MorrisLecar::FIG9 }.equilibria(0.0, 20_000).unwrap();
        assert_eq!(full.len(), 3);
        let agree = eq.iter().zip(&full).fold(0.0_f64, |w, (a, b)| w.max((a.v - b.v).abs()));
        assert!(agree < 1e-13, "{agree}");
    }

    /// Eq. 8 as printed: its voltage equation collapses Fig. 3d's plateau at once, its accumulation
    /// law does not close dimensionally, and one reading that does close draws Fig. 3d.
    ///
    /// The first two parts check the PAGE's arithmetic, not this module: the printed `dV/dt` is written
    /// out here from Eq. 8 (only [`Gate::steady`] and [`electrodiffusion`] are the module's), and the
    /// dimension exponents are constants of this test that no edit to the module can change. The third
    /// part is the module's: the reading runs on [`SingleConductance::field`], [`electrodiffusion`]
    /// and the module's own Runge–Kutta step, with `[Ca]_i` as a second state.
    ///
    /// At Fig. 3d's start (`V = +28`, `[Ca]_i = 0.001`, `[Ca]_o = 100`, `g*_Ca = 40`, `V_L = −35`,
    /// `C = 20`, and the caption's `g_L = 2` for the undefined `g*_L`), the printed `dV/dt` is
    /// −177.56 mV/ms; Eq. 3 with the inward drive gives −0.1835 (reference: −177.5609839321603 and
    /// −0.18353628813713171; measured agreement 2.8 × 10⁻¹⁴ and 6.9 × 10⁻¹⁶).
    ///
    /// Dimensions as exponents of ampere, volt, second, metre and mole: the printed
    /// `K(CF)⁻¹ g*_Ca V R` is mV²/(cm·ms) with a bare `F` and mV²·mol/(C·cm·ms) with `F` in C/mol,
    /// while a concentration per unit time is mol/(m³·s) — which `g*_Ca R/(F × depth)` is.
    /// `1/(F × 1 µm)`, with `F = 96 485.332 12 C/mol` (CODATA 2018), is 1.036 × 10⁻⁴ mM/ms per µA/cm²:
    /// the caption's `|K| = 10⁻⁴` to 3.6%.
    ///
    /// The reading `d[Ca]_i/dt = K g*_Ca M∞(V) R` in mM/ms, with `K = −10⁻⁴`, beside Eq. 3 with
    /// `M = M∞(V)` and the inward drive, integrated by the module's RK4 at 0.05 ms from Fig. 3d's
    /// initial conditions: +16.00, +9.70 and +4.40 mV at 1, 2 and 3 s, and through −20 mV at 3.677 s
    /// with `[Ca]_i = 33.95` mM (reference, DOP853: 15.9988952902707, 9.699378679530456,
    /// 4.396517543117574 mV, 3677.046115908112 ms, 33.95499385219301 mM; measured agreement
    /// 2.8 × 10⁻¹³ mV, and 1.8 × 10⁻⁶ ms for the crossing placed by linear interpolation inside one
    /// step). Four separate readings at 600 dpi against the figure's ticks (13.24 to 13.26 pixels to
    /// the millivolt, 230.0 to 230.2 to the second) put the centre of the drawn stroke at +15.96 to
    /// +16.06, +9.70 to +9.77 and +4.44 to +4.52 mV at 1, 2 and 3 s, and its fall through −20 mV at
    /// 3.668 to 3.676 s; the stroke is 6 pixels wide, 0.45 mV and 0.026 s, so each reading is good to
    /// about ±0.3 mV and ±0.03 s, and the model is inside that against both ends of every range.
    #[test]
    fn eq_8_as_printed_cannot_draw_figure_3d() {
        let s = SingleConductance { drive: Drive::Electrodiffusion { ca_in: 0.001, ca_out: 100.0 }, ..SingleConductance::FIG3B };
        let m = s.gate.steady(28.0);
        let printed = (0.0 - 2.0 * (28.0 + 35.0) + 40.0 * m * 28.0 * electrodiffusion(28.0, 0.001 / 100.0)) / 20.0;
        let inward = s.field(28.0, m, 0.0).0;
        assert!((printed - (-177.5609839321603)).abs() < 1e-13 && (inward - (-0.18353628813713171)).abs() < 3e-15, "{printed} {inward}");

        type Dim = [i32; 5];
        let times = |a: Dim, b: Dim| -> Dim { core::array::from_fn(|k| a[k] + b[k]) };
        let per = |a: Dim, b: Dim| -> Dim { core::array::from_fn(|k| a[k] - b[k]) };
        let (per_length, capacitance, conductance, volt): (Dim, Dim, Dim, Dim) =
            ([0, 0, 0, -1, 0], [1, -1, 1, -2, 0], [1, -1, 0, -2, 0], [0, 1, 0, 0, 0]);
        let (bare, faraday, length, rate_of_concentration): (Dim, Dim, Dim, Dim) = ([0; 5], [1, 0, 1, 0, -1], [0, 0, 0, 1, 0], [0, 0, -1, -3, 1]);
        let printed_law = |f: Dim| times(times(times(per(per_length, times(capacitance, f)), conductance), volt), volt);
        assert_eq!(printed_law(bare), [0, 2, -1, -1, 0], "mV²/(cm·ms)");
        assert_eq!(printed_law(faraday), [-1, 2, -2, -1, 1], "mV²·mol/(C·cm·ms)");
        assert_eq!(per(times(conductance, volt), times(faraday, length)), rate_of_concentration);
        let flux: f64 = 1e-6 / (96_485.332_12 * 1e-4) * 1e6 / 1e3;
        assert!((flux - 1.0364269656617729e-4).abs() < 1e-18 && (flux / 1e-4 - 1.036).abs() < 1e-3, "{flux}");

        let f = |[v, ca]: [f64; 2]| -> [f64; 2] {
            let s = SingleConductance { drive: Drive::Electrodiffusion { ca_in: ca, ca_out: 100.0 }, ..SingleConductance::FIG3B };
            let m = s.gate.steady(v);
            [s.field(v, m, 0.0).0, -1e-4 * s.g * m * electrodiffusion(v, ca / 100.0)]
        };
        let h = 0.05;
        let mut y = [28.0, 0.001];
        let mut at = Vec::new();
        let mut fall = None;
        for k in 1..=80_000 {
            let next = rk4(f, y, h);
            if fall.is_none() && y[0] >= -20.0 && next[0] < -20.0 {
                let x = (-20.0 - y[0]) / (next[0] - y[0]);
                fall = Some(((f64::from(k) - 1.0 + x) * h, y[1] + x * (next[1] - y[1])));
            }
            y = next;
            if k % 20_000 == 0 && k <= 60_000 {
                at.push(y[0]);
            }
        }
        let (t_fall, ca_fall) = fall.unwrap();
        let mut worst = 0.0_f64;
        for (got, want) in at.iter().zip([15.9988952902707, 9.699378679530456, 4.396517543117574]) {
            worst = worst.max((got - want).abs());
        }
        assert!(worst < 1e-12, "{at:?}: {worst}");
        assert!((t_fall - 3677.046115908112).abs() < 5e-6 && (ca_fall - 33.95499385219301).abs() < 3e-8, "{t_fall} {ca_fall}");
        for (got, (low, high)) in at.iter().zip([(15.96, 16.06), (9.70, 9.77), (4.44, 4.52)]) {
            assert!((got - low).abs() < 0.3 && (got - high).abs() < 0.3, "against the page: {at:?}");
        }
        assert!((t_fall - 3668.0).abs() < 30.0 && (t_fall - 3676.0).abs() < 30.0, "against the page: {t_fall}");
    }

    /// Fig. 9's caption: the oscillating currents do NOT coincide with those at which the nullclines
    /// cross on the `V̇ = 0` nullcline's rising, negative-resistance limb.
    ///
    /// On the equilibrium curve `G − ḡ` is Eq. 15's first margin plus `Cλ_N`, and it is the slope of
    /// Eq. 11's nullcline times `g_K(V_s − V_K)` — checked against central differences of
    /// [`MorrisLecar::v_nullcline`] to 1.0 × 10⁻⁹ of `g_L + g_Ca + g_K` (measured). It changes sign at
    /// `V = −7.2375` and `12.6461 mV`, where `I_ss` is 195.636 and 493.886 µA/cm² (reference: `brentq`,
    /// 195.63622207770723 and 493.88640851011166; measured agreement 5.7 × 10⁻¹⁴); Eq. 9's Hopf
    /// currents are 289.65 and 465.10. At `I = 200` the crossing, −6.94 mV, is on the limb and the
    /// equilibrium is a stable focus: Fig. 9a's damped `I = 200` trace.
    #[test]
    fn figure_9_oscillates_on_less_than_the_rising_limb() {
        let m = MorrisLecar::FIG9;
        let limb = |v: f64| m.oscillation_margins(v).0 + m.c * m.n.rate_at(v);
        let mut slope_err = 0.0_f64;
        for k in -100..=100 {
            let v = f64::from(k) * 0.5 + 0.25;
            let (i, h) = (m.steady_current(v), 1e-5);
            let slope = (m.v_nullcline(v + h, i).unwrap() - m.v_nullcline(v - h, i).unwrap()) / (2.0 * h);
            slope_err = slope_err.max((slope * m.g_k * (v - m.v_k) - limb(v)).abs() / (m.g_l + m.g_ca + m.g_k));
        }
        assert!(slope_err < 5e-9, "{slope_err}");
        let mut worst = 0.0_f64;
        for ((mut a, mut b), want) in [((-10.0, 0.0), 195.63622207770723), ((10.0, 20.0), 493.88640851011166)] {
            let below = limb(a) < 0.0;
            assert!(below != (limb(b) < 0.0));
            for _ in 0..100 {
                let mid = 0.5 * (a + b);
                if (limb(mid) < 0.0) == below { a = mid } else { b = mid }
            }
            worst = worst.max((m.steady_current(a) - want).abs());
        }
        assert!(worst < 3e-13, "{worst}");
        let hopf: Vec<f64> = m.trace_zeros(-60.0, 60.0, 1_200).unwrap().iter().map(|p| p.i).collect();
        assert!(195.63622207770723 < hopf[0] - 90.0 && hopf[1] + 25.0 < 493.88640851011166, "{hopf:?}");
        let e = m.equilibria(200.0, 10_000).unwrap()[0];
        assert!(limb(e.v) > 0.0 && (e.v - (-6.938470173997106)).abs() < 5e-15, "{e:?}");
        assert_eq!(e.kind, Kind::StableFocus);
    }

    /// Fig. 10b is a transient: with `V3 = 12` the oscillation decays onto a stable focus.
    ///
    /// RK4 at 0.05 ms for a minute from `V = −50`, `N = N∞(−50)` at `I = 300`. The largest
    /// `|V − V_s|` over the last 200 ms before 1, 3, 10, 30 and 60 s is 2.06, 0.972, 0.314, 0.0287 and
    /// 0.000 859 mV (reference: DOP853 sampled on the same grid, 2.057745171921461,
    /// 0.9724628802796467, 0.3135392052874426, 0.02869686669446736, 0.0008592083447371834; measured
    /// agreement 6.4 × 10⁻⁹ mV). From 30 s to 60 s it falls by a factor of 33.40, the linear rate's
    /// `exp(0.000 116 94 × 30 000) = 33.39` to 0.04% (measured): nothing but the equilibrium is left.
    ///
    /// By a hair. With `V3 = 12` the trace of Eq. 9's Jacobian vanishes on the curve at 78.110 685 and
    /// 299.890 483 µA/cm², both with `det > 0`: Hopf currents, the upper 0.11 µA/cm² below the figure's
    /// 300 (reference: `brentq`, 78.11068469014617 and 299.89048271329466; measured agreement 2.3 ×
    /// 10⁻¹³). Bisecting `V3` for the upper one to reach 300 gives `V3 = 11.993 278` mV (reference:
    /// 11.993277808156565; measured agreement 5.3 × 10⁻¹⁵), and at `λ̄_N = 0.066` the one equilibrium
    /// is an unstable focus, `+0.000 217 46 ± 0.265 812i`, at 0.0666 a stable one, `−0.000 083 50 ±
    /// 0.267 018i` (reference: `NumPy`, 0.00021746034072741, 0.2658121120417925,
    /// −8.349869835739564e-05, 0.26701769163308153; measured agreement 1.3 × 10⁻¹⁶).
    #[test]
    fn figure_10b_is_a_decaying_transient() {
        let m = MorrisLecar::FIG10B;
        let z = m.trace_zeros(-60.0, 60.0, 1_200).unwrap();
        assert_eq!(z.len(), 2, "{z:?}");
        assert!(z.iter().all(|p| p.det > 0.0), "{z:?}");
        let hopf = (z[0].i - 78.11068469014617).abs().max((z[1].i - 299.89048271329466).abs());
        assert!(hopf < 1e-12, "{hopf}");
        assert!((300.0 - z[1].i - 0.11).abs() < 0.01, "{z:?}");
        let upper = |v3: f64| MorrisLecar { n: Gate { v_half: v3, ..m.n }, ..m }.trace_zeros(-60.0, 60.0, 1_200).unwrap()[1].i;
        let (mut a, mut b) = (11.9, 12.1);
        assert!(upper(a) > 300.0 && upper(b) < 300.0);
        for _ in 0..50 {
            let mid = 0.5 * (a + b);
            if upper(mid) > 300.0 { a = mid } else { b = mid }
        }
        assert!((a - 11.993277808156565).abs() < 3e-14 && (12.0 - a - 0.0067).abs() < 1e-4, "{a}");
        let mut rounding = 0.0_f64;
        for (rate, kind, (re_want, im_want)) in [
            (0.066, Kind::UnstableFocus, (0.00021746034072741, 0.2658121120417925)),
            (0.0666, Kind::StableFocus, (-8.349869835739564e-05, 0.26701769163308153)),
        ] {
            let r = MorrisLecar { n: Gate { rate, ..m.n }, ..m };
            let eq = r.equilibria(300.0, 10_000).unwrap();
            assert_eq!(eq.len(), 1, "lambda_N = {rate}: {eq:?}");
            assert_eq!(eq[0].kind, kind, "lambda_N = {rate}");
            let (re, im) = top(&r.reduced_linearisation(eq[0].v, eq[0].x));
            rounding = rounding.max((re - re_want).abs()).max((im - im_want).abs());
        }
        assert!(rounding < 4e-16, "{rounding}");
        let e = m.equilibria(300.0, 10_000).unwrap()[0];
        let (re, _) = top(&m.reduced_linearisation(e.v, e.x));
        let h = 0.05;
        let (mut v, mut n) = (-50.0, m.n.steady(-50.0));
        let mut far = Vec::new();
        let mut window = 0.0_f64;
        for k in 1..=1_200_000_u32 {
            (v, n) = m.step_reduced(v, n, 300.0, h).unwrap();
            let t = f64::from(k) * h;
            if [1000.0, 3000.0, 10_000.0, 30_000.0, 60_000.0].iter().any(|end| t >= end - 200.0 - 1e-9 && t <= end + 1e-9) {
                window = window.max((v - e.v).abs());
            }
            if [20_000, 60_000, 200_000, 600_000, 1_200_000].contains(&k) {
                far.push(window);
                window = 0.0;
            }
        }
        let mut worst = 0.0_f64;
        for (got, want) in far.iter().zip([2.057745171921461, 0.9724628802796467, 0.3135392052874426, 0.02869686669446736, 0.0008592083447371834]) {
            worst = worst.max((got - want).abs());
        }
        assert!(worst < 2e-8, "{far:?}: {worst}");
        assert!(far.windows(2).all(|w| w[1] < w[0]), "{far:?}");
        let ratio = far[3] / far[4];
        assert!((ratio / (-re * 30_000.0).exp() - 1.0).abs() < 1e-3, "{ratio}");
    }
}
