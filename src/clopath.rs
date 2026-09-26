//! Voltage-based spike-timing-dependent plasticity with homeostasis: a synapse that reads the
//! postsynaptic membrane POTENTIAL rather than postsynaptic spike times — checked against the closed
//! forms its paper derives and, bit for bit, against a transcription of the authors' own code.
//!
//! # The rule
//!
//! Clopath, Büsing, Vasilaki and Gerstner, *Connectivity reflects coding: a model of voltage-based
//! STDP with homeostasis*, Nature Neuroscience 13:344–352, 2010, doi:10.1038/nn.2479. The model is in
//! its Online Methods, whose three pages carry no journal page numbers and are cited here as PDF
//! pp. 10–12. Eq. (3), PDF p. 10:
//!
//! ```text
//! dw_i/dt = −A_LTD(ū̄) X_i (ū_− − θ_−)_+  +  A_LTP x̄_i (u − θ_+)_+ (ū_+ − θ_−)_+        (3)
//! τ_− dū_−/dt = −ū_− + u,     τ_+ dū_+/dt = −ū_+ + u,     τ_x dx̄_i/dt = −x̄_i + X_i
//! A_LTD(ū̄) = A_LTD ū̄²/u²_ref,     w_min ≤ w_i ≤ w_max
//! ```
//!
//! `u` is the postsynaptic potential, `X_i = Σ_n δ(t − t_i^n)` the presynaptic spike train and
//! `(·)_+` rectification. Each presynaptic spike DEPRESSES by `A_LTD (ū_− − θ_−)_+` if the potential,
//! low-passed over `τ_−`, is above `θ_−`; POTENTIATION runs continuously while the momentary
//! potential is above `θ_+`, the potential low-passed over `τ_+` is above `θ_−`, and the presynaptic
//! trace `x̄` is non-zero. Because the trace is driven by a delta train, each spike raises it by
//! `1/τ_x` and its time integral per spike is exactly one — which is what makes `A_LTP` carry the
//! printed unit mV⁻². The `ū_+` equation is not printed as one: the text says `ū_+` "is similar to
//! `ū_−(t)` but has a shorter time constant `τ_+`" (PDF p. 10), and the middle line above is that
//! sentence.
//!
//! [`Rule`] holds one of Table 1b's parameter sets (journal p. 349): [`Rule::VISUAL_CORTEX`], the
//! paper's standard set; [`Rule::SOMATOSENSORY_CORTEX`]; [`Rule::HIPPOCAMPAL`]. [`Synapses`] runs
//! eq. (3) for a bank of synapses onto one neuron in discrete time.
//!
//! ⚠ **Table 1b's fifth column is headed "`A_LTD` (mV⁻²)", a second time.** Its unit, mV⁻², is the
//! unit of `A_LTP` in eq. (2), and the authors' code passes that column's 8 × 10⁻⁵ as `A_p`, "amplitude
//! for potentiation". It is `A_LTP` here.
//!
//! ⚠ **The hippocampal row leaves `τ_−` and `τ_+` blank**, although the Online Methods say those two
//! are fitted "to each data set" (PDF p. 11). They are `None` in [`Rule::HIPPOCAMPAL`], not a value
//! chosen here, and [`Synapses::new`] refuses the set by naming the blank cell. The row's only use in
//! the paper is the voltage-clamp fit of Fig. 1h, where every filtered voltage equals the clamp and
//! neither time constant enters — so the closed forms below accept it.
//!
//! ⚠ **The Online Methods give `τ_+` as "around 10 ms"** (PDF p. 10); Table 1b gives 7 ms and 5 ms,
//! and 10 ms is the visual set's `τ_−`. The table is used, and it is what the code uses.
//!
//! ⚠ **Eqs. (1) and (2) gate on the bounds; eq. (3) clips to them.** Depression is printed "if
//! `w_i > w_min`" and potentiation "if `w_i < w_max`" — a step that starts inside may still end
//! outside — while eq. (3) is "combined with the hard bounds `w_min ≤ w_i ≤ w_max`" (PDF p. 10).
//! [`Synapses`] clips after each step's whole update, so a weight never leaves its bounds, as the
//! authors' `RFdevelop.m` does (`w(find(w<0)) = 0`, `w(find(w>wmax)) = wmax`).
//!
//! `θ_+ = −45.3` mV is "the firing threshold of the `AdEx` model" (PDF p. 11), not its
//! `V_Trest = −50.4` mV, and the paper gives no derivation. It matches, to the printed precision, the
//! upper zero of the membrane current with no adaptation and no afterpotential,
//! `−g_L (u − E_L) + g_L Δ_T e^{(u − V_Trest)/Δ_T} = 0`, at −45.327 mV. That zero is what the paper the
//! Online Methods take the neuron's parameters from — ref. 49, Brette and Gerstner, *Adaptive
//! exponential integrate-and-fire model as an effective description of neuronal activity*, Journal of
//! Neurophysiology 94:3637–3642, 2005 — calls `V_s`: "the critical voltage `V_s`, above which an action
//! potential upstroke is generated", the right-hand zero of `f(V)` "in the nonadapted state (w = 0)"
//! (its Fig. 1A caption, p. 3639). Ref. 49 prints no value for it. This is the reading adopted here,
//! and the tests find that root; with the steady-state adaptation `a (u − E_L)` included it would be
//! −45.06 mV. `θ_− = −70.6` mV is `E_L`, "the resting potential".
//!
//! # The reference is the authors' code
//!
//! `ModelDB` accession 144566, "Voltage-based STDP synapse (Clopath et al. 2010)", mirrored at
//! github.com/ModelDBRepository/144566 (commit `8f336a6`). Its `VoTriCode/aEIF.m` (the neuron),
//! `VoTri.m` (the rule and the pairing protocol) and `FrequencyDependence.m` (the script that computes
//! Fig. 2b's model lines, at its five frequencies) each open "Code written by Claudia Clopath";
//! `FrequencyDependence.m` calls `VoTri` with `par = [0.00014 0.00008 7 15 10]`, the visual-cortex
//! row. The same accession carries a NEURON port, `stdp_cc.mod`, by Torben-Nielsen and Clopath; it
//! drives a Hodgkin–Huxley soma with its own parameters (`A_m = 0.00001`, `A_p = 0.00012`,
//! `tetam = -64.9`, `tetap = -35`, `tau_y = 114`) and is not the model of the paper's figures.
//!
//! No MATLAB or Octave was available to this review, so the `.m` files were not run. They were
//! transcribed line for line into `tools/clopath_reference.py`, which downloads them at that commit
//! (`aEIF.m`, `VoTri.m`, and the homeostat's statements from `RFdevelop.m`), refuses to run unless
//! every MATLAB statement it transcribes appears in them verbatim, and prints the table the tests
//! hold. [`votri_pairing`] is the same computation built from this module's parts, and it
//! reproduces that table BIT FOR BIT: the ten weights `FrequencyDependence.m` computes for Fig. 2b,
//! three post-pre weights around 35 Hz, three runs with one doubtful value changed, six with the
//! delay moved and eight with one of the code's train quirks undone, thirty trains of 150 000 to
//! 500 113 one-millisecond samples, each final weight the same `f64`; and [`Neuron`] fires on the
//! same steps as the transcribed `aEIF.m`.
//! (The transcription calls `CPython`'s `math.exp` and this module `f64::exp`, the same C library
//! function on the machine that made the table. MATLAB's own `exp` may differ from it in the last
//! place, and that does not move the table: with every `exp` result moved one unit in the last place
//! down, or up, all 34 of its entries come out bit for bit the same —
//! `tools/clopath_reference.py --perturb-exp`.) As normalized weights the ten are 100.0 %, 102.9 %,
//! 115.7 %, 146.3 % and 165.0 % for pre-post pairing at 0.1, 10, 20, 40 and 50 Hz, and 71.7 %,
//! 59.6 %, 60.5 %, 117.8 % and 173.6 % for post-pre.
//!
//! ⚠ **Fig. 2b as published is not exactly what this code computes.** Its model lines are vector
//! paths in the PDF, and `tools/clopath_figures.py` reads their vertices against the panel's own
//! axis ticks, whose three positions on each axis lie on a straight line to 0.004 Hz and 0.017 %. Each
//! line has 82 vertices, at 0.1 Hz and at every whole-millisecond period from 100 ms to 20 ms, where
//! `FrequencyDependence.m` computes five points. At the five frequencies they read 100.96, 104.35,
//! 117.27, 148.25 and 167.25 % pre-post, and 71.68, 59.59, 60.49, 118.77 and 175.90 % post-pre. The
//! post-pre line is the code's to 0.05 points at 0.1 Hz and at every period from 100 ms down to
//! 39 ms, 63 vertices, then rises above it: by 0.99 points at 40 Hz and 2.34 at 50 Hz. The pre-post
//! line lies above the code's at every vertex, by 0.96 points at 0.1 Hz and by 1.43 to 2.24 points
//! elsewhere, growing with the frequency. No read delay from zero to five steps closes the gap at
//! 0.1 Hz: with none that point is 129.9 %, and every delay from one to five steps leaves it at
//! 100.0 %. So the figure agrees with the code wherever each presynaptic spike is at least 29 ms
//! before the next postsynaptic one, and lies above it wherever a presynaptic spike comes sooner. A
//! depolarization following each presynaptic spike would do that, and `VoTri.m` models none (its
//! `I_ext` is zeros); the paper describes none either, and this module does not test that hypothesis.
//!
//! The code settles every value below that the paper does not print or prints doubtfully, and each
//! is recorded as coming from the code, not the page.
//!
//! # The neuron
//!
//! The postsynaptic cell is the adaptive exponential integrate-and-fire model with two additions,
//! Online Methods PDF p. 10: a depolarizing spike afterpotential current `z`, set to `I_sp` at a spike
//! and decaying with `τ_z`, and a threshold `V_T` that "starts at `V_Tmax` after a spike and decays to
//! `V_Trest`":
//!
//! ```text
//! C du/dt = −g_L (u − E_L) + g_L Δ_T exp((u − V_T)/Δ_T) − w_ad + z + I
//! τ_wad dw_ad/dt = a (u − E_L) − w_ad,     τ_z dz/dt = −z,     τ_VT dV_T/dt = −(V_T − V_Trest)
//! ```
//!
//! [`Membrane::TABLE_1A`] is Table 1a as printed; [`Membrane::AEIF_M`] is what `aEIF.m` runs; and
//! [`Neuron`] runs `aEIF.m` step for step — forward Euler at a fixed 1 ms.
//!
//! ⚠ **`V_Tmax` is printed as "30.4 mV", without the minus sign** that `E_L` and `V_Trest` carry in
//! the same column. `aEIF.m` sets `V_T = VT_jump + VT_rest` with `VT_jump = 20`: −30.4 mV. Fig. 1c
//! alone cannot decide the sign, because the step current that makes its neuron "fire at 50 Hz" is
//! not printed; the code decides it. The sign is not cosmetic: under the same 1 nA step the code's
//! neuron fires 48 times in its second second, and with the printed +30.4 mV — a threshold that
//! restarts 80.8 mV above `V_Trest` — 9 times, 101, 107 and 110 ms apart.
//!
//! ⚠ **`b` is printed as 0.805 pA, the code runs 0.0805 pA, and the source the Online Methods name
//! gives 80.5 pA.** "Parameters for the neuron model are taken from ref. 49 for the `AdEx` model"
//! (PDF p. 10), and ref. 49's Table 1 has `b` = 0.0805 nA (p. 3638) — "we took b = 80.5 pA"
//! (p. 3639). `aEIF.m` has `b = 0.0805` with the comment `[nA]`, and adds it to a current that the
//! same line's arithmetic keeps in pA (`g_L` in nS times mV, `w_jump = 400`): it runs the source's
//! number in the wrong unit, 0.0805 pA. That slip is what the code's Fig. 2b weights were computed
//! with. Ref. 49's 80.5 pA does not reproduce the figure — 143.7 % at 50 Hz pre-post, against the
//! code's 165.0 % and the figure's 167.25 % — and the printed 0.805 pA matches neither the source nor
//! the code, though Fig. 2b cannot tell it from the code's value (164.8 %). The code's value is
//! used; the printed one is kept in [`Membrane::TABLE_1A`].
//!
//! ⚠ **`V_reset`, the spike's detection level and its shape are not printed in this paper** — the
//! text names "the fixed value `V_reset`" and gives none. Ref. 49 prints two of them: a spike is
//! "triggered when the voltage reaches a threshold `V_peak` = 20 mV", and integration "is restarted
//! from a reset value `V_r`, with `V_r = E_L`" (p. 3638), −70.6 mV. `aEIF.m` detects a spike when `u`
//! exceeds the same 20 mV, holds `u` at 29.4 mV for that millisecond and at 29.4 + 3.462 mV for the
//! next (its comment: values computed once from a finely resolved spike so that the clamped
//! millisecond carries the same voltage integral), undoes the adaptation update of that second
//! millisecond, and resets not to `E_L` but to `E_L + 15 + 6.0984` = −49.5016 mV. Fig. 2h draws such
//! spikes: flat tops near +30 mV, about 2 ms wide by eye. [`Membrane::TABLE_1A`] leaves the three
//! `None`, and [`Neuron::new`] refuses it by naming the first.
//!
//! # A delay the paper does not print, and cannot do without
//!
//! `VoTri.m` reads the filtered voltages THREE steps late — `u_md(t-3)` and `u_mp(t-3)` — so that
//! "we want to read the value of the voltage trace before this spike" (its comment in `aEIF.m`).
//! [`Synapses::new`] takes that delay as a parameter; eq. (3) as printed has none.
//!
//! ⚠ **Read literally, eq. (3) cannot produce the paper's own Fig. 2b.** The main text says pre-post
//! pairing at 0.1 Hz "did not show any change in the model" (journal p. 346), because `ū_+` "decays
//! back to zero before the next impulse arrives" — and the eq. (6) step of the Online Methods needs
//! `Ȳ_+(t)` "not influenced by a possible spike at time t". A spike 2 ms wide is not a delta: with
//! no delay, `ū_+` in the spike's second millisecond already contains its first, and every isolated
//! pairing potentiates. The 0.1 Hz pre-post point moves from 100.0 % to 129.9 %.
//!
//! What the figure needs is SOME delay, not three steps: read 1, 2, 3, 4 or 5 steps late, 0.1 Hz
//! pre-post ends at 100.0 %. Three is `aEIF.m`'s choice — "the spike length (2ms) plus 1 time step
//! (1ms)". The NEURON port is configured with `delay_steps = 5.0 / DT` ("AP duration",
//! `claudia_pairing.py`), although in `stdp_cc.mod` the ring-buffer index it reads, a `LOCAL` of the
//! `DERIVATIVE` block, is read before it is set, so what delay it actually applies is not settled
//! by reading it, and NEURON was not run here. The delay moves the rest of the curve: 50 Hz pre-post
//! is 159.6, 162.2, 165.0 and 168.1 % for delays of 1, 2, 3 and 4 steps. The published post-pre line
//! does single out three: at 0.1, 10 and 20 Hz it is within 0.03 points of the code's with a delay
//! of 3 steps and at least 0.48 points from it with 1, 2, 4 or 5.
//!
//! ⚠ **"100 % normalized weight" is not one weight.** `VoTri.m` starts at `w = 0.5` and
//! `FrequencyDependence.m` plots `(w − 0.5)/0.5`, so Fig. 2b is normalized to 0.5. Fig. 1h is not:
//! its visual-cortex curve bottoms out at 91.1 % and reaches 250 % at −26.5 mV only if 100 % is a
//! weight of 1; normalized to 0.5 the same run bottoms out at 82.3 %. Its two lines, read from their
//! vector paths (`tools/clopath_figures.py`), follow `100 (1 + n g(u_c))` — 100 % a weight of one — to
//! 0.14 and 0.17 points at every vertex below the 250 % where the axis stops, 293 and 423 of their
//! 426.
//!
//! # Voltage clamp: Fig. 1h in closed form
//!
//! Under clamp "the actual value of the voltage `u` itself and the low-pass-filtered versions `ū`
//! are constant and equal to `u_clamp`" (PDF p. 11), and eq. (3) becomes linear in the presynaptic
//! rate: each spike changes the weight by `g(u_c) = −A_LTD (u_c − θ_−)_+ + A_LTP (u_c − θ_+)_+ (u_c − θ_−)_+`
//! ([`Rule::clamp_per_spike`], [`Rule::clamp_rate`]) — nothing below `θ_−`, depression only up to
//! `θ_+`, a quadratic above. LTP overtakes LTD at `θ_+ + A_LTD/A_LTP` ([`Rule::clamp_crossover`]),
//! and the curve is lowest at `max(θ_+, (θ_+ + θ_− + A_LTD/A_LTP)/2)` ([`Rule::clamp_minimum`]):
//! −30 mV exactly for the hippocampal set, the middle of the flat bottom Fig. 1h's red line draws
//! from −30.40 to −29.59 mV.
//!
//! ⚠ **The clamp protocol is printed twice, differently**: "2 Hz during 50 s" between −60 and 0 mV
//! (Fig. 1 caption) and "25 (blue line) or 100 pulses (red line) at 50 Hz" from −80 to 0 mV (PDF
//! p. 11). For a constant clamp inside the bounds the rate does not matter — the change is the number
//! of spikes times `g(u_c)` — and the tests check that 100 spikes at 2 Hz and at 50 Hz give the same
//! weight, as the Discussion predicts ("dependent on the voltage and the number of presynaptic spikes
//! but not on their exact timing", journal p. 351).
//!
//! # The rate limit: eq. (8) is BCM
//!
//! For a linear Poisson neuron, `u = u^s + βY + θ_−` with rate `ν_post = u^s/α` (eq. 4), averaging
//! eq. (3) over independent Poisson trains gives eq. (8), PDF p. 11:
//!
//! ```text
//! ⟨dw_i/dt⟩ = (α + β) β A_LTP ν_i^pre ν_post (ν_post − ϑ),     ϑ = A_LTD/(β A_LTP)
//! ```
//!
//! — the quadratic BCM rule with a threshold that slides with `A_LTD(ū̄)`
//! ([`Rule::bcm_drift`], [`Rule::bcm_threshold`]). Its step `⟨Y Ȳ_+⟩ = ν ν̄_+` needs `Ȳ_+` to exclude
//! the spike at `t` itself, and a discrete scheme honours that only if it reads `ū_+` before adding
//! the current step's voltage — which [`Synapses::step`] does. The tests simulate that neuron and
//! find eq. (8), below and above `ϑ`.
//!
//! ⚠ **`β` is printed as 1.2 mV; in eq. (4) it multiplies a delta train and is a voltage × time.**
//! The paper's illustration — a triangular spike of 120 mV and 1 ms at half height in a 100 ms record
//! — has an integral of 120 mV·ms; 1.2 mV is that spike's contribution to the 100 ms MEAN, `βν`. With
//! the visual set, `β = 120 mV·ms` puts `ϑ` at 14.6 Hz; read as 1.2 mV·ms, at 1458 Hz, a rate no
//! neuron reaches.
//!
//! # Homeostasis: what `ū̄` is
//!
//! For the network simulations `A_LTD` depends on "the mean depolarization … of the postsynaptic
//! neuron, averaged over a time scale of 1 s", as `A_LTD ū̄²/u²_ref` (PDF p. 10), with `u²_ref` 60,
//! 80 and 50 mV² for Figs. 5, 6 and 7 (PDF p. 12). [`Rule::a_ltd_homeostatic`] is that formula.
//!
//! ⚠ **The sentence that defines the average names the wrong variable**: "the mean depolarization
//! `ū_−`", with the single bar and minus sign of the 10 ms depression filter, while eq. (1), eq. (3)
//! and `A_LTD(ū̄)` carry the double bar and PDF p. 11 calls it "the long-term averaged potential
//! `ū̄`". The double bar is meant: a 10 ms filter is not a 1 s average.
//!
//! ⚠ **The code averages the SQUARE.** `RFdevelop.m` — the receptive-field script, with Fig. 5's
//! `u²_ref = 60` — low-passes `(u − E_L)²` with `tau_th = 1.2*1000.0` ms and divides by 60: the mean
//! of the squared depolarization over 1.2 s, not the square of the mean over 1 s. The two readings
//! differ in time constant and, far more, by the depolarization's variance, which for a spiking
//! neuron is large. [`Homeostat`] keeps both. On the code's neuron under a 1 nA step from rest, with
//! both filtered over 1.2 s, the mean of the square is 1.28 times the square of the mean after 5 s,
//! while the filters are still settling, and 1.26 times after 20 s; against the paper's 1 s mean the
//! ratio is 1.26 at both times. Rest-relative, as both readings are: an absolute potential of −70 mV
//! would square to 4 900 mV², eighty times `u²_ref`.
//!
//! # The pairing protocol
//!
//! [`votri_trains`] builds `VoTri.m`'s spike trains: five pairings at the pairing frequency, the
//! postsynaptic spike `lag` ms after the presynaptic one, each group of five padded with silence to
//! 10 s and repeated 15 times, at 0.1 Hz. When the pairing frequency is itself 0.1 Hz a group lasts
//! 50.011 s, is not padded, and is repeated 10 times, back to back. [`votri_pairing`] runs them.
//!
//! ⚠ **The paper's own protocol statements disagree.** Fig. 2a's caption — the timing window at
//! 20 Hz, not the frequency sweep of Fig. 2b reproduced here — says "60 pre-post-pairs at 20 Hz"; the
//! Online Methods' "STDP experiment and frequency dependence" give five pairings repeated 15 times,
//! 75, and the code runs 75 (50 at 0.1 Hz). The main text sends the reader to the Supplementary
//! Methods for this protocol ("Fig. 2 and Supplementary Methods", journal p. 345); they were not
//! consulted here. The Methods also say postsynaptic spikes were "advanced by +10 ms or delayed by
//! −10 ms", but in the figures and in the code +10 ms puts the postsynaptic spike AFTER the
//! presynaptic one: `lag` here is `t_post − t_pre`.
//!
//! ⚠ **The code's post-pre trains carry six postsynaptic spikes per group of five pairings**: `y` is
//! filled up to `end` and `x` only to `end-1`, so for a negative lag one more postsynaptic spike
//! fits, `|lag|` ms before the end of the group — at 50 Hz, 10 ms after the last presynaptic spike, a
//! pre-post pair. And its main loop starts at `t = 4`, so a spike in the first three samples is never
//! delivered: the very first postsynaptic spike of every post-pre run. Both are reproduced, because
//! the published post-pre line has them. With the code's trains it is within 0.03 points of the
//! model at 0.1, 10 and 20 Hz. Remove each group's sixth postsynaptic spike and the model moves
//! 2.17 points from the figure at 0.1 Hz and 0.89 at 20 Hz (and at 50 Hz, where that spike makes a
//! pre-post pair, 29.43); deliver the first one and it moves 0.52 points at each of the three.
//!
//! # What is checked
//!
//! - Table 1a and 1b cell by cell, printed and coded, and the blank cells refused by name; `θ_+`
//!   against the upper root of the `AdEx` membrane current, ref. 49's `V_s`.
//! - The voltage-clamp closed form and its crossover and minimum; Fig. 1h's 91.1 % at `θ_+` and 250 %
//!   at −26.5 mV (visual set, 25 spikes, 100 % = 1), its hippocampal minimum of 75.8 % at −30 mV and
//!   crossover at −19 mV (100 spikes), each against the figure's own vector paths.
//! - [`Synapses`] under clamp: the filters stay at the clamp and the weight follows the closed form
//!   for a finite window, under either integrator, at any presynaptic rate. The filters' and the
//!   trace's step responses in closed form, and the delay step by step. The bounds clip once, after
//!   the step's whole update — neither term by term nor, as eqs. (1) and (2) print, by gating each
//!   term on the weight the step starts from: a weight at `w_min` whose step depresses more than it
//!   potentiates stays there, as one at `w_max` does when its step potentiates more.
//! - Eq. (8) against a simulated linear Poisson neuron, on both sides of `ϑ`; the printed `β`.
//! - The time-stepped rule converges at first order, as the authors' scheme does: halving the step
//!   halves the error, and the two integrators, extrapolated to a zero step, reach the weight that
//!   eq. (3) gives in continuous time, computed in closed form up to one quadrature.
//! - The neuron against `aEIF.m`: its resting potential solves the steady-state current–voltage
//!   equation, its spike is the code's two clamped milliseconds, and under a 1 nA step for 2 s it
//!   fires on the same steps as the transcription, all 97 spikes with the code's `V_Tmax` and all 18
//!   with the printed one; firing with no presynaptic spike changes no weight, as Fig. 1c says.
//! - [`Homeostat`]: its filter in closed form, and both readings of `ū̄` on the code's neuron against
//!   the transcription's.
//! - Fig. 2b: all ten weights of `VoTri.m` bit for bit against the transcription; against the
//!   figure's own vector paths, the post-pre line at 0.1, 10 and 20 Hz and the pre-post line's gap;
//!   the main text's statements (no change at 0.1 Hz, LTP above 10 Hz, post-pre LTD at every period
//!   from 29 ms to 100 ms, LTP dominating at 50 Hz); each of the three doubtful choices above moved on
//!   its own; the delay moved to 1, 2 and 4 steps, bit for bit, with the figure singling out 3; and
//!   each of the two train quirks undone, bit for bit, with the figure telling each apart.
//!
//! # Units
//!
//! Every interface is SI: potentials in volts, currents in amperes, times in seconds, rates in
//! hertz, `α` and `β` in volt-seconds. The parameters stay as Tables 1a and 1b print them — mV, ms,
//! pA, pF, nS, mV⁻¹, mV⁻² — so that each can be read against its cell, and are converted at each
//! boundary, as [`crate::neuron::Izhikevich`] does. A weight is dimensionless.
//!
//! # Not here
//!
//! The network simulations of Figs. 4–7 — they need a synapse model, an inhibitory neuron and
//! current amplitudes the paper does not give. The protocols of Fig. 2c–i, whose depolarizing and
//! hyperpolarizing currents are not quantified. Fig. 3, whose two alternative parameter sets print
//! `A_LTP` as 50 and 67 × 10⁻⁴ mV⁻², 17 and 22 times Table 1b's somatosensory value, with no way here
//! to tell a genuine value from an exponent slip.

use core::fmt;
use std::collections::VecDeque;

/// Why the rule, the neuron or a protocol could not be built or run.
#[derive(Debug, Clone, PartialEq)]
pub enum ClopathError {
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
    /// A value that must be finite and not negative was not.
    Negative {
        /// Which.
        what: &'static str,
        /// Its value.
        value: f64,
    },
    /// A cell Table 1b leaves blank.
    Blank {
        /// Which parameter.
        what: &'static str,
    },
    /// A neuron value the paper does not print.
    Unprinted {
        /// Which value.
        what: &'static str,
    },
    /// `θ_−` above `θ_+`.
    Thresholds {
        /// `θ_−`, mV.
        theta_minus: f64,
        /// `θ_+`, mV.
        theta_plus: f64,
    },
    /// Weight bounds that do not form an interval.
    Bounds {
        /// The lower bound.
        w_min: f64,
        /// The upper bound.
        w_max: f64,
    },
    /// An initial weight outside the bounds.
    OutsideBounds {
        /// Which synapse.
        index: usize,
        /// Its weight.
        w: f64,
        /// The lower bound.
        w_min: f64,
        /// The upper bound.
        w_max: f64,
    },
    /// A presynaptic spike slice of the wrong length.
    Length {
        /// The number of synapses.
        expected: usize,
        /// The number of flags passed.
        got: usize,
    },
    /// A forward-Euler step longer than a time constant.
    StepTooLong {
        /// Which time constant.
        what: &'static str,
        /// The step, seconds.
        dt: f64,
        /// The time constant, seconds.
        tau: f64,
    },
    /// A pairing protocol outside the range [`votri_trains`] builds.
    Protocol {
        /// The pairing period, 1 ms steps.
        period: usize,
        /// `t_post − t_pre`, 1 ms steps.
        lag: i32,
    },
}

impl fmt::Display for ClopathError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFinite { what, value } => write!(f, "{what} = {value} is not finite"),
            Self::NotPositive { what, value } => write!(f, "{what} = {value} must be finite and positive"),
            Self::Negative { what, value } => write!(f, "{what} = {value} must be finite and not negative"),
            Self::Blank { what } => write!(f, "{what} is blank in Table 1b for this set; the paper prints no value"),
            Self::Unprinted { what } => {
                write!(f, "{what} is not printed in the paper; Membrane::AEIF_M takes it from the authors' aEIF.m")
            }
            Self::Thresholds { theta_minus, theta_plus } => {
                write!(f, "theta_minus = {theta_minus} mV lies above theta_plus = {theta_plus} mV")
            }
            Self::Bounds { w_min, w_max } => write!(f, "w_min = {w_min} and w_max = {w_max} do not bound an interval"),
            Self::OutsideBounds { index, w, w_min, w_max } => {
                write!(f, "weight {index} = {w} lies outside [{w_min}, {w_max}]")
            }
            Self::Length { expected, got } => write!(f, "{got} presynaptic flags for {expected} synapses"),
            Self::StepTooLong { what, dt, tau } => {
                write!(f, "dt = {dt} s exceeds {what} = {tau} s, and the forward-Euler factor 1 - dt/tau would be negative")
            }
            Self::Protocol { period, lag } => write!(
                f,
                "a period of {period} ms with a lag of {lag} ms: votri_trains takes 1 <= period <= 10000 and |lag| < period"
            ),
        }
    }
}

impl std::error::Error for ClopathError {}

fn finite(what: &'static str, value: f64) -> Result<f64, ClopathError> {
    if value.is_finite() { Ok(value) } else { Err(ClopathError::NonFinite { what, value }) }
}

fn positive(what: &'static str, value: f64) -> Result<f64, ClopathError> {
    if value.is_finite() && value > 0.0 { Ok(value) } else { Err(ClopathError::NotPositive { what, value }) }
}

fn non_negative(what: &'static str, value: f64) -> Result<f64, ClopathError> {
    if value.is_finite() && value >= 0.0 { Ok(value) } else { Err(ClopathError::Negative { what, value }) }
}

/// One of Table 1b's parameter sets, in the table's units.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rule {
    /// `θ_−`, mV: the depression threshold, and the level `ū_+` must exceed for potentiation.
    pub theta_minus: f64,
    /// `θ_+`, mV: the level the momentary potential must exceed for potentiation. Not below `θ_−`.
    pub theta_plus: f64,
    /// `A_LTD`, mV⁻¹: depression per presynaptic spike per millivolt of `ū_−` above `θ_−`.
    pub a_ltd: f64,
    /// `A_LTP`, mV⁻²: potentiation per unit of presynaptic trace integral per mV².
    pub a_ltp: f64,
    /// `τ_x`, ms: the presynaptic trace's time constant.
    pub tau_x: f64,
    /// `τ_−`, ms: the depression filter's time constant; `None` where Table 1b is blank.
    pub tau_minus: Option<f64>,
    /// `τ_+`, ms: the potentiation filter's time constant; `None` where Table 1b is blank.
    pub tau_plus: Option<f64>,
}

impl Rule {
    /// Table 1b, "Visual cortex", the standard set, fitted to Sjöström, Turrigiano and Nelson (2001).
    pub const VISUAL_CORTEX: Self = Self {
        theta_minus: -70.6,
        theta_plus: -45.3,
        a_ltd: 14e-5,
        a_ltp: 8e-5,
        tau_x: 15.0,
        tau_minus: Some(10.0),
        tau_plus: Some(7.0),
    };

    /// Table 1b, "Somatosensory cortex", fitted to Nevian and Sakmann (2006).
    pub const SOMATOSENSORY_CORTEX: Self = Self {
        theta_minus: -70.6,
        theta_plus: -45.3,
        a_ltd: 21e-5,
        a_ltp: 30e-5,
        tau_x: 30.0,
        tau_minus: Some(6.0),
        tau_plus: Some(5.0),
    };

    /// Table 1b, "Hippocampal", fitted to Ngezahayo, Schachner and Artola (2000), with `τ_x = 16` ms
    /// fixed rather than fitted and `τ_−`, `τ_+` blank.
    pub const HIPPOCAMPAL: Self = Self {
        theta_minus: -41.0,
        theta_plus: -38.0,
        a_ltd: 38e-5,
        a_ltp: 2e-5,
        tau_x: 16.0,
        tau_minus: None,
        tau_plus: None,
    };

    /// Every value finite, both amplitudes not negative, every time constant that is present
    /// positive, and `θ_− ≤ θ_+`.
    ///
    /// # Errors
    ///
    /// [`ClopathError::NonFinite`] for a threshold, [`ClopathError::Thresholds`] for `θ_− > θ_+`,
    /// [`ClopathError::Negative`] for an amplitude and [`ClopathError::NotPositive`] for a time
    /// constant, naming the first that fails.
    pub fn check(&self) -> Result<(), ClopathError> {
        finite("theta_minus", self.theta_minus)?;
        finite("theta_plus", self.theta_plus)?;
        if self.theta_minus > self.theta_plus {
            return Err(ClopathError::Thresholds { theta_minus: self.theta_minus, theta_plus: self.theta_plus });
        }
        non_negative("a_ltd", self.a_ltd)?;
        non_negative("a_ltp", self.a_ltp)?;
        positive("tau_x", self.tau_x)?;
        if let Some(tau) = self.tau_minus {
            positive("tau_minus", tau)?;
        }
        if let Some(tau) = self.tau_plus {
            positive("tau_plus", tau)?;
        }
        Ok(())
    }

    /// The weight change per presynaptic spike under a voltage clamp at `u_clamp` volts:
    /// `g = −A_LTD (u_c − θ_−)_+ + A_LTP (u_c − θ_+)_+ (u_c − θ_−)_+`, the clamp form of eq. (3) with
    /// the trace's integral of one per spike.
    ///
    /// # Errors
    ///
    /// [`ClopathError::NonFinite`] for the clamp; whatever [`Rule::check`] refuses.
    pub fn clamp_per_spike(&self, u_clamp: f64) -> Result<f64, ClopathError> {
        self.check()?;
        let u = finite("u_clamp", u_clamp)? * 1e3;
        let over_minus = (u - self.theta_minus).max(0.0);
        let over_plus = (u - self.theta_plus).max(0.0);
        Ok(-self.a_ltd * over_minus + self.a_ltp * over_plus * over_minus)
    }

    /// The mean rate of weight change under a clamp at `u_clamp` volts with presynaptic spikes at
    /// `nu_pre` hertz: `ν_pre g(u_c)`, per second. Linear in the rate — Fig. 1h's shape times `ν_pre`.
    ///
    /// # Errors
    ///
    /// [`ClopathError::Negative`] for the rate; whatever [`Rule::clamp_per_spike`] refuses.
    pub fn clamp_rate(&self, u_clamp: f64, nu_pre: f64) -> Result<f64, ClopathError> {
        let per_spike = self.clamp_per_spike(u_clamp)?;
        Ok(non_negative("nu_pre", nu_pre)? * per_spike)
    }

    /// The clamp voltage, volts, above which a presynaptic spike potentiates: `θ_+ + A_LTD/A_LTP`.
    ///
    /// # Errors
    ///
    /// [`ClopathError::NotPositive`] for an `A_LTP` of zero, which never overtakes; whatever
    /// [`Rule::check`] refuses.
    pub fn clamp_crossover(&self) -> Result<f64, ClopathError> {
        self.check()?;
        let a_ltp = positive("a_ltp", self.a_ltp)?;
        Ok((self.theta_plus + self.a_ltd / a_ltp) * 1e-3)
    }

    /// The clamp voltage, volts, at which the per-spike change is lowest:
    /// `max(θ_+, (θ_+ + θ_− + A_LTD/A_LTP)/2)`. Between `θ_−` and `θ_+` the change falls linearly;
    /// above `θ_+` it is the convex quadratic `(u − θ_−)(A_LTP(u − θ_+) − A_LTD)`, whose vertex this is
    /// when the vertex lies above `θ_+`.
    ///
    /// # Errors
    ///
    /// As [`Rule::clamp_crossover`].
    pub fn clamp_minimum(&self) -> Result<f64, ClopathError> {
        self.check()?;
        let a_ltp = positive("a_ltp", self.a_ltp)?;
        let vertex = 0.5 * (self.theta_plus + self.theta_minus + self.a_ltd / a_ltp);
        Ok(vertex.max(self.theta_plus) * 1e-3)
    }

    /// Eq. (8)'s threshold `ϑ = A_LTD/(β A_LTP)`, hertz, for a spike weight `beta` in volt-seconds.
    ///
    /// # Errors
    ///
    /// [`ClopathError::NotPositive`] for `beta` or an `A_LTP` of zero; whatever [`Rule::check`]
    /// refuses.
    pub fn bcm_threshold(&self, beta: f64) -> Result<f64, ClopathError> {
        self.check()?;
        let beta = positive("beta", beta)? * 1e6;
        let a_ltp = positive("a_ltp", self.a_ltp)?;
        Ok(self.a_ltd / (beta * a_ltp) * 1e3)
    }

    /// Eq. (8): the mean drift, per second, of a synapse whose presynaptic train fires at `nu_pre`
    /// hertz onto a linear Poisson neuron firing at `nu_post` hertz, with `alpha` (`u^s/ν_post`) and
    /// `beta` (the spike's voltage integral) in volt-seconds:
    /// `(α + β) β A_LTP ν_pre ν_post (ν_post − ϑ)`, computed as
    /// `(α + β) ν_pre ν_post (β A_LTP ν_post − A_LTD)` so that it holds for `A_LTP = 0` as well.
    ///
    /// # Errors
    ///
    /// [`ClopathError::Negative`] for a rate, [`ClopathError::NotPositive`] for `alpha` or `beta`;
    /// whatever [`Rule::check`] refuses.
    pub fn bcm_drift(&self, nu_pre: f64, nu_post: f64, alpha: f64, beta: f64) -> Result<f64, ClopathError> {
        self.check()?;
        let pre = non_negative("nu_pre", nu_pre)? * 1e-3;
        let post = non_negative("nu_post", nu_post)? * 1e-3;
        let alpha = positive("alpha", alpha)? * 1e6;
        let beta = positive("beta", beta)? * 1e6;
        Ok((alpha + beta) * pre * post * (beta * self.a_ltp * post - self.a_ltd) * 1e3)
    }

    /// The homeostatic amplitude `A_LTD(ū̄) = A_LTD ū̄²/u²_ref`, mV⁻¹, from `ubarbar_sq` and `u_ref_sq`
    /// in the same unit (V² at this crate's interface). At `ū̄² = u²_ref` it is `A_LTD` itself, which
    /// is how the paper fits the slice experiments.
    ///
    /// # Errors
    ///
    /// [`ClopathError::Negative`] for `ubarbar_sq`, [`ClopathError::NotPositive`] for `u_ref_sq`;
    /// whatever [`Rule::check`] refuses.
    pub fn a_ltd_homeostatic(&self, ubarbar_sq: f64, u_ref_sq: f64) -> Result<f64, ClopathError> {
        self.check()?;
        Ok(self.a_ltd * non_negative("ubarbar_sq", ubarbar_sq)? / positive("u_ref_sq", u_ref_sq)?)
    }
}

/// `u²_ref` for the rate-coding network of Fig. 5: 60 mV², in V².
pub const U_REF_SQ_FIG5: f64 = 60e-6;
/// `u²_ref` for the temporal-coding network of Fig. 6: 80 mV², in V².
pub const U_REF_SQ_FIG6: f64 = 80e-6;
/// `u²_ref` for the receptive-field simulations of Fig. 7: 50 mV², in V².
pub const U_REF_SQ_FIG7: f64 = 50e-6;

/// How [`Synapses`] advances the filters and the trace over one step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Integrator {
    /// Exact decay over the step for a potential held at the step's value: `ū ← u + (ū − u)e^{−h/τ}`,
    /// and the potentiation term integrates the decaying trace over the step exactly. Under a clamp
    /// the whole rule is then exact at any step.
    Exponential,
    /// `VoTri.m`'s forward Euler: `ū ← u h/τ + (1 − h/τ) ū`, the trace decays by `1 − h/τ_x`, and
    /// the potentiation term is the trace at the start of the step times `h`. At `h` = 1 ms this is
    /// the code's arithmetic, operation for operation.
    Euler,
}

/// A bank of plastic synapses onto one neuron, run by eq. (3) in steps of `dt`.
///
/// The postsynaptic filters `ū_−` and `ū_+` are the neuron's, shared by every synapse; each synapse
/// keeps its weight and its presynaptic trace. One call of [`Synapses::step`] is one step of
/// `VoTri.m`'s main loop.
#[derive(Debug, Clone, PartialEq)]
pub struct Synapses {
    rule: Rule,
    integrator: Integrator,
    h: f64,
    tau_minus: f64,
    tau_plus: f64,
    decay_x: f64,
    decay_minus: f64,
    decay_plus: f64,
    window: f64,
    w_min: f64,
    w_max: f64,
    ltd_scale: f64,
    w: Vec<f64>,
    x: Vec<f64>,
    now: (f64, f64),
    past: VecDeque<(f64, f64)>,
}

impl Synapses {
    /// Synapses under `rule`, stepped by `integrator` every `dt` seconds, reading the filtered
    /// voltages `delay` steps late (`VoTri.m`: 3; eq. (3) as printed: 0), with both filters starting
    /// at `u0` volts, the traces at zero, the given initial weights, and hard bounds
    /// `bounds = (w_min, w_max)` — infinite bounds are no bound, as in `VoTri.m`.
    ///
    /// # Errors
    ///
    /// [`ClopathError::Blank`] for a rule without `τ_−` or `τ_+`; [`ClopathError::NotPositive`] for
    /// `dt`; [`ClopathError::StepTooLong`] for an Euler step longer than a time constant;
    /// [`ClopathError::NonFinite`] for `u0` or a weight; [`ClopathError::Bounds`] and
    /// [`ClopathError::OutsideBounds`]; whatever [`Rule::check`] refuses.
    pub fn new(
        rule: Rule,
        integrator: Integrator,
        dt: f64,
        delay: usize,
        u0: f64,
        weights: Vec<f64>,
        bounds: (f64, f64),
    ) -> Result<Self, ClopathError> {
        rule.check()?;
        let tau_minus = rule.tau_minus.ok_or(ClopathError::Blank { what: "tau_minus" })?;
        let tau_plus = rule.tau_plus.ok_or(ClopathError::Blank { what: "tau_plus" })?;
        let h = positive("dt", dt)? * 1e3;
        let taus = [("tau_x", rule.tau_x), ("tau_minus", tau_minus), ("tau_plus", tau_plus)];
        let [decay_x, decay_minus, decay_plus] = match integrator {
            Integrator::Exponential => taus.map(|(_, tau)| (-h / tau).exp()),
            Integrator::Euler => {
                for (what, tau) in taus {
                    if h > tau {
                        return Err(ClopathError::StepTooLong { what, dt, tau: tau * 1e-3 });
                    }
                }
                taus.map(|(_, tau)| 1.0 - h / tau)
            }
        };
        let window = match integrator {
            Integrator::Exponential => -rule.tau_x * (-h / rule.tau_x).exp_m1(),
            Integrator::Euler => h,
        };
        let u0 = finite("u0", u0)? * 1e3;
        let (w_min, w_max) = bounds;
        if !(w_min <= w_max) {
            return Err(ClopathError::Bounds { w_min, w_max });
        }
        for (index, &w) in weights.iter().enumerate() {
            finite("weight", w)?;
            if w < w_min || w > w_max {
                return Err(ClopathError::OutsideBounds { index, w, w_min, w_max });
            }
        }
        let n = weights.len();
        Ok(Self {
            rule,
            integrator,
            h,
            tau_minus,
            tau_plus,
            decay_x,
            decay_minus,
            decay_plus,
            window,
            w_min,
            w_max,
            ltd_scale: 1.0,
            w: weights,
            x: vec![0.0; n],
            now: (u0, u0),
            past: std::iter::repeat_n((u0, u0), delay + 1).collect(),
        })
    }

    /// Scale depression by `A_LTD(ū̄)/A_LTD` — from [`Rule::a_ltd_homeostatic`] divided by
    /// `A_LTD`, or a [`Homeostat`]'s average over `u²_ref`. It starts at 1, the slice experiments'
    /// `ū̄ = u_ref`.
    ///
    /// # Errors
    ///
    /// [`ClopathError::Negative`] for a scale that is negative or not finite.
    pub fn set_ltd_scale(&mut self, scale: f64) -> Result<(), ClopathError> {
        self.ltd_scale = non_negative("ltd_scale", scale)?;
        Ok(())
    }

    /// One step with the postsynaptic potential at `u` volts and `pre[i]` true for each synapse whose
    /// presynaptic neuron spikes in it.
    ///
    /// The order is `VoTri.m`'s. A presynaptic spike depresses by `A_LTD (ū_− − θ_−)_+` with `ū_−`
    /// as it was `delay` steps before this one; potentiation over the step reads `u`, `ū_+` as it
    /// was `delay` steps before, and the trace BEFORE this step's spike; the weight is clipped to the
    /// bounds; then the trace takes the spike and decays and the filters take `u`.
    ///
    /// # Errors
    ///
    /// [`ClopathError::NonFinite`] for `u`; [`ClopathError::Length`] for a `pre` slice that is not
    /// one flag per synapse.
    pub fn step(&mut self, u: f64, pre: &[bool]) -> Result<(), ClopathError> {
        let u = finite("u", u)? * 1e3;
        if pre.len() != self.w.len() {
            return Err(ClopathError::Length { expected: self.w.len(), got: pre.len() });
        }
        self.step_mv(u, pre);
        Ok(())
    }

    /// [`Synapses::step`] with the potential already in millivolts and one flag per synapse, as
    /// `VoTri.m` holds them, so that [`votri_pairing`] runs the code's arithmetic without a round
    /// trip through volts.
    fn step_mv(&mut self, u: f64, pre: &[bool]) {
        let r = self.rule;
        let (lagged_minus, lagged_plus) = self.past[0];
        let ltd = r.a_ltd * self.ltd_scale * (lagged_minus - r.theta_minus).max(0.0);
        let gain = r.a_ltp * (u - r.theta_plus).max(0.0);
        let gate = (lagged_plus - r.theta_minus).max(0.0);
        let jump = 1.0 / r.tau_x;
        for (i, &spike) in pre.iter().enumerate() {
            let mut w = self.w[i];
            if spike {
                w -= ltd;
            }
            w += gain * self.x[i] * gate * self.window;
            self.w[i] = w.clamp(self.w_min, self.w_max);
            self.x[i] = if spike { jump } else { 0.0 } + self.decay_x * self.x[i];
        }
        let (m, p) = self.now;
        self.now = match self.integrator {
            Integrator::Exponential => (u + (m - u) * self.decay_minus, u + (p - u) * self.decay_plus),
            Integrator::Euler => {
                (u * self.h / self.tau_minus + self.decay_minus * m, u * self.h / self.tau_plus + self.decay_plus * p)
            }
        };
        self.past.push_back(self.now);
        self.past.pop_front();
    }

    /// The weights.
    #[must_use]
    pub fn weights(&self) -> &[f64] {
        &self.w
    }

    /// The presynaptic traces `x̄_i`, hertz: each spike adds `1/τ_x`.
    #[must_use]
    pub fn traces(&self) -> Vec<f64> {
        self.x.iter().map(|x| x * 1e3).collect()
    }

    /// `(ū_−, ū_+)` now, volts — not the delayed values the rule reads.
    #[must_use]
    pub fn filtered(&self) -> (f64, f64) {
        (self.now.0 * 1e-3, self.now.1 * 1e-3)
    }
}

/// The long-term average that drives homeostasis, kept both ways the sources define it.
///
/// The paper: `ū̄`, the mean depolarization over 1 s, squared. `RFdevelop.m`: the mean of the squared
/// depolarization, over 1.2 s. Both are low-passed here by the same exact exponential filter of the
/// depolarization `u − rest` (and of its square), starting from zero as the code's `theta` does, so
/// that [`Homeostat::mean_square`] ≥ [`Homeostat::mean`]² always. The gap is the filtered variance
/// plus a start-up term that vanishes as the filters settle: with the filter weights summing to
/// `W = 1 − e^{−t/τ}`, it is `W·var + W(1 − W)·m²`, where `m` and `var` are the mean and variance
/// under those weights divided by `W`. Held at one depolarization, `var` is zero and the gap is the
/// start-up term alone.
///
/// `RFdevelop.m`'s own statement is forward Euler, fed the potential one step late
/// (`theta = (1-dt/tau_th)*theta+dt/tau_th*((u1ss-E_L).^2)`). On the code's neuron under a 1 nA step
/// from rest, with the same 1.2 s, it is 1.00017 times [`Homeostat::mean_square`] after 5 s and
/// 1.00054 times after 20 s (`tools/clopath_reference.py`, on the transcribed `aEIF.m`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Homeostat {
    rest: f64,
    decay: f64,
    mean: f64,
    mean_sq: f64,
}

impl Homeostat {
    /// `RFdevelop.m`'s averaging time, `tau_th = 1.2*1000.0` ms, in seconds. The paper says 1 s.
    pub const TAU_CODE: f64 = 1.2;

    /// A homeostat averaging over `tau` seconds, stepped every `dt` seconds, measuring the
    /// depolarization from `rest` volts.
    ///
    /// # Errors
    ///
    /// [`ClopathError::NotPositive`] for `tau` or `dt`; [`ClopathError::NonFinite`] for `rest`.
    pub fn new(tau: f64, rest: f64, dt: f64) -> Result<Self, ClopathError> {
        let decay = (-positive("dt", dt)? / positive("tau", tau)?).exp();
        Ok(Self { rest: finite("rest", rest)? * 1e3, decay, mean: 0.0, mean_sq: 0.0 })
    }

    /// One step with the potential at `u` volts, held over the step.
    ///
    /// # Errors
    ///
    /// [`ClopathError::NonFinite`] for `u`.
    pub fn step(&mut self, u: f64) -> Result<(), ClopathError> {
        let d = finite("u", u)? * 1e3 - self.rest;
        self.mean = d + (self.mean - d) * self.decay;
        self.mean_sq = d * d + (self.mean_sq - d * d) * self.decay;
        Ok(())
    }

    /// The filtered depolarization, volts: the paper's `ū̄`.
    #[must_use]
    pub fn mean(&self) -> f64 {
        self.mean * 1e-3
    }

    /// The filtered squared depolarization, V²: what `RFdevelop.m` divides by `u²_ref`.
    #[must_use]
    pub fn mean_square(&self) -> f64 {
        self.mean_sq * 1e-6
    }
}

/// The neuron's parameters in Table 1a's units: pF, nS, mV, ms, pA.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Membrane {
    /// `C`, pF.
    pub c: f64,
    /// `g_L`, nS.
    pub g_l: f64,
    /// `E_L`, mV.
    pub e_l: f64,
    /// `Δ_T`, mV.
    pub delta_t: f64,
    /// `V_Trest`, mV: the threshold's resting value.
    pub v_t_rest: f64,
    /// `τ_wad`, ms.
    pub tau_w: f64,
    /// `a`, nS: subthreshold adaptation.
    pub a: f64,
    /// `b`, pA: spike-triggered adaptation.
    pub b: f64,
    /// `I_sp`, pA: the afterpotential current a spike sets.
    pub i_sp: f64,
    /// `τ_z`, ms.
    pub tau_z: f64,
    /// `τ_VT`, ms.
    pub tau_v_t: f64,
    /// `V_Tmax`, mV: the threshold a spike sets.
    pub v_t_max: f64,
    /// The potential above which a spike is detected, mV. Not printed.
    pub v_detect: Option<f64>,
    /// The potential held for each of the spike's two clamped milliseconds, mV. Not printed.
    pub peak: Option<[f64; 2]>,
    /// `V_reset`, mV. Named in the Online Methods, not printed.
    pub v_reset: Option<f64>,
}

impl Membrane {
    /// Table 1a as printed, journal p. 349: `b` = 0.805 pA and `V_Tmax` = 30.4 mV as they stand, and
    /// the three values the paper never prints left `None`.
    pub const TABLE_1A: Self = Self {
        c: 281.0,
        g_l: 30.0,
        e_l: -70.6,
        delta_t: 2.0,
        v_t_rest: -50.4,
        tau_w: 144.0,
        a: 4.0,
        b: 0.805,
        i_sp: 400.0,
        tau_z: 40.0,
        tau_v_t: 50.0,
        v_t_max: 30.4,
        v_detect: None,
        peak: None,
        v_reset: None,
    };

    /// `aEIF.m`: Table 1a with `b = 0.0805` pA, `V_Tmax = VT_jump + VT_rest = 20 − 50.4` mV, spike
    /// detection above `th = 20` mV, the clamped values 29.4 and 29.4 + 3.462 mV, and the reset
    /// `E_L + 15 + 6.0984` mV — each written as the code computes it.
    pub const AEIF_M: Self = Self {
        b: 0.0805,
        v_t_max: 20.0 + -50.4,
        v_detect: Some(20.0),
        peak: Some([29.4, 29.4 + 3.462]),
        v_reset: Some(-70.6 + 15.0 + 6.0984),
        ..Self::TABLE_1A
    };
}

/// Where the neuron is in `aEIF.m`'s spike: its `counter`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    /// `counter == 0`: integrating.
    Integrating,
    /// `counter == 1`: the millisecond the spike was detected in, held at the first peak value.
    Peak,
    /// `counter == 2`: held at the second peak value; the next step resets.
    Plateau,
}

/// The paper's neuron as `aEIF.m` integrates it: forward Euler at a fixed 1 ms, a spike detected
/// above `v_detect`, held for two milliseconds, then reset.
///
/// One call of [`Neuron::step`] is one call of `aEIF.m`, statement for statement. The step is fixed
/// because the spike's two clamped values were computed for it, and because it is the scheme the
/// figures were made with; a convergence test would be testing a different model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Neuron {
    m: Membrane,
    v_detect: f64,
    peak: [f64; 2],
    v_reset: f64,
    u: f64,
    w: f64,
    z: f64,
    v_t: f64,
    phase: Phase,
}

impl Neuron {
    /// The fixed step, seconds.
    pub const DT: f64 = 1e-3;

    /// A neuron at `E_L` with no adaptation, no afterpotential and the threshold at `V_Trest`.
    ///
    /// # Errors
    ///
    /// [`ClopathError::NotPositive`] for `C`, `g_L`, `Δ_T` or a time constant;
    /// [`ClopathError::StepTooLong`] for a time constant under the 1 ms step;
    /// [`ClopathError::NonFinite`] for any other value; [`ClopathError::Unprinted`] for a value left
    /// `None`, as [`Membrane::TABLE_1A`] leaves three.
    pub fn new(m: Membrane) -> Result<Self, ClopathError> {
        for (what, value) in [("c", m.c), ("g_l", m.g_l), ("delta_t", m.delta_t)] {
            positive(what, value)?;
        }
        for (what, tau) in [("tau_w", m.tau_w), ("tau_z", m.tau_z), ("tau_v_t", m.tau_v_t)] {
            if positive(what, tau)? < 1.0 {
                return Err(ClopathError::StepTooLong { what, dt: Self::DT, tau: tau * 1e-3 });
            }
        }
        for (what, value) in
            [("e_l", m.e_l), ("v_t_rest", m.v_t_rest), ("a", m.a), ("b", m.b), ("i_sp", m.i_sp), ("v_t_max", m.v_t_max)]
        {
            finite(what, value)?;
        }
        let v_detect = finite("v_detect", m.v_detect.ok_or(ClopathError::Unprinted { what: "v_detect" })?)?;
        let peak = m.peak.ok_or(ClopathError::Unprinted { what: "peak" })?;
        finite("peak[0]", peak[0])?;
        finite("peak[1]", peak[1])?;
        let v_reset = finite("v_reset", m.v_reset.ok_or(ClopathError::Unprinted { what: "v_reset" })?)?;
        Ok(Self { m, v_detect, peak, v_reset, u: m.e_l, w: 0.0, z: 0.0, v_t: m.v_t_rest, phase: Phase::Integrating })
    }

    /// One millisecond under an injected current of `i` amperes; `true` if a spike is detected in it.
    ///
    /// # Errors
    ///
    /// [`ClopathError::NonFinite`] for `i`.
    pub fn step(&mut self, i: f64) -> Result<bool, ClopathError> {
        Ok(self.step_pa(finite("i", i)? * 1e12))
    }

    /// [`Neuron::step`] with the current already in picoamperes, as `aEIF.m` takes it.
    fn step_pa(&mut self, i: f64) -> bool {
        let m = self.m;
        if self.phase == Phase::Plateau {
            self.u = self.v_reset;
            self.w += m.b;
            self.z = m.i_sp;
            self.phase = Phase::Integrating;
            self.v_t = m.v_t_max;
        }
        let exponential = m.g_l * m.delta_t * ((self.u - self.v_t) / m.delta_t).exp();
        let udot = 1.0 / m.c * (-m.g_l * (self.u - m.e_l) + exponential - self.w + self.z + i);
        let wdot = 1.0 / m.tau_w * (m.a * (self.u - m.e_l) - self.w);
        self.u += udot;
        self.w += wdot;
        self.z -= self.z / m.tau_z;
        self.v_t = m.v_t_rest / m.tau_v_t + (1.0 - 1.0 / m.tau_v_t) * self.v_t;
        if self.phase == Phase::Peak {
            self.phase = Phase::Plateau;
            self.u = self.peak[1];
            self.w -= wdot;
        }
        let fired = self.u > self.v_detect && self.phase == Phase::Integrating;
        if fired {
            self.u = self.peak[0];
            self.phase = Phase::Peak;
        }
        fired
    }

    /// The membrane potential `u`, volts.
    #[must_use]
    pub fn potential(&self) -> f64 {
        self.u * 1e-3
    }

    /// The adaptive threshold `V_T`, volts.
    #[must_use]
    pub fn threshold(&self) -> f64 {
        self.v_t * 1e-3
    }

    /// The adaptation current `w_ad`, amperes.
    #[must_use]
    pub fn adaptation(&self) -> f64 {
        self.w * 1e-12
    }

    /// The afterpotential current `z`, amperes.
    #[must_use]
    pub fn afterpotential(&self) -> f64 {
        self.z * 1e-12
    }
}

/// `VoTri.m`'s presynaptic and postsynaptic spike trains, one flag per 1 ms step: five pairings
/// `period` steps apart with the postsynaptic spike `lag` steps after the presynaptic one, the group
/// padded with silence to 10 s if it is shorter — a 0.1 Hz group is 50.011 s long and is not — and
/// repeated 15 times, or 10 times when `period` is 10 000 (0.1 Hz).
///
/// Built as the code builds them, including its sixth postsynaptic spike for a negative `lag` (see
/// the module doc).
///
/// The range is Fig. 2b's: a period of 10 000 ms is its lowest pairing frequency, 0.1 Hz, the rate
/// the groups themselves are repeated at, and a lag of a whole period or more would pair each
/// spike with the partner of the next pairing.
///
/// # Errors
///
/// [`ClopathError::Protocol`] unless `1 ≤ period ≤ 10 000` and `|lag| < period`.
pub fn votri_trains(period: usize, lag: i32) -> Result<(Vec<bool>, Vec<bool>), ClopathError> {
    let gap = lag.unsigned_abs() as usize;
    // A period of 0 fails `gap >= period` for every lag.
    if period > 10_000 || gap >= period {
        return Err(ClopathError::Protocol { period, lag });
    }
    let len = 5 * period + gap + 1;
    let block = len.max(10_000);
    let reps = if period == 10_000 { 10 } else { 15 };
    let (mut pre, mut post) = (vec![false; block * reps], vec![false; block * reps]);
    let first_post = if lag > 0 { 2 * gap } else { 0 };
    for rep in 0..reps {
        let base = rep * block;
        // x(abs(Dt)+1:f:end-1) = 1, 0-based: from |Dt| while at most len − 2.
        for t in (gap..len - 1).step_by(period) {
            pre[base + t] = true;
        }
        // y(abs(Dt)+1+Dt:f:end) = 1, 0-based: from |Dt| + Dt while at most len − 1.
        for t in (first_post..len).step_by(period) {
            post[base + t] = true;
        }
    }
    Ok((pre, post))
}

/// `VoTri.m` end to end: the neuron built from `membrane`, driven at each postsynaptic spike of
/// [`votri_trains`] by `VoTri.m`'s 10⁶ pA (`I_s = y*1000000`), one synapse under `rule` stepped by
/// [`Integrator::Euler`] at 1 ms reading its filters `delay` steps late, starting at `w = 0.5` with
/// no bounds, the loop starting at `t = 4` as the code's does. Returns the final weight; Fig. 2b plots
/// `(w − 0.5)/0.5`.
///
/// # Errors
///
/// Whatever [`votri_trains`], [`Neuron::new`] and [`Synapses::new`] refuse.
pub fn votri_pairing(rule: Rule, membrane: Membrane, delay: usize, period: usize, lag: i32) -> Result<f64, ClopathError> {
    let (pre, post) = votri_trains(period, lag)?;
    votri_run(rule, membrane, delay, &pre, &post)
}

/// [`votri_pairing`] on trains given as two flag slices of one length, so that the tests can run
/// the code's loop on trains with either of its quirks undone.
fn votri_run(rule: Rule, membrane: Membrane, delay: usize, pre: &[bool], post: &[bool]) -> Result<f64, ClopathError> {
    let mut neuron = Neuron::new(membrane)?;
    let unbounded = (f64::NEG_INFINITY, f64::INFINITY);
    let mut synapse = Synapses::new(rule, Integrator::Euler, Neuron::DT, delay, membrane.e_l * 1e-3, vec![0.5], unbounded)?;
    for t in 3..pre.len() {
        // `I_s = y*1000000` pA, and `u(t)` in mV, handed on as the code holds them.
        neuron.step_pa(if post[t] { 1e6 } else { 0.0 });
        synapse.step_mv(neuron.u, &pre[t..=t]);
    }
    Ok(synapse.w[0])
}

#[cfg(test)]
mod tests {
    use super::{
        ClopathError, Homeostat, Integrator, Membrane, Neuron, Rule, Synapses, U_REF_SQ_FIG5, U_REF_SQ_FIG6, U_REF_SQ_FIG7,
        votri_pairing, votri_run, votri_trains,
    };
    use crate::rng::Rng;

    const UNBOUNDED: (f64, f64) = (f64::NEG_INFINITY, f64::INFINITY);

    /// `tools/clopath_reference.py`: `VoTri(rho, Dt, par)` of the transcribed `VoTri.m` and `aEIF.m`
    /// with `par = [0.00014 0.00008 7 15 10]`, as `(period ms, lag ms, final w)`. Fig. 2b's ten
    /// points, then post-pre at 29, 28 and 27 ms.
    const REFERENCE: [(usize, i32, f64); 13] = [
        (10000, 10, 0.5000010253248656),
        (100, 10, 0.5145169131346606),
        (50, 10, 0.5785321888565067),
        (25, 10, 0.7313255793239102),
        (20, 10, 0.8250073071063487),
        (10000, -10, 0.3584295456866586),
        (100, -10, 0.2980880775388822),
        (50, -10, 0.30255459853267885),
        (25, -10, 0.5888818833532738),
        (20, -10, 0.867785888080795),
        (29, -10, 0.47058659846259776),
        (28, -10, 0.49459237300370246),
        (27, -10, 0.521940645642134),
    ];

    /// The same script, pre-post, with one value changed at a time: `b` = 0.805 and `b` = 80.5 at
    /// a period of 20 ms, and the filtered voltages read with no delay at 10 000 ms, as
    /// `(b, delay, period ms, final w)`.
    const DOUBTFUL: [(f64, usize, usize, f64); 3] =
        [(0.805, 3, 20, 0.824048792613747), (80.5, 3, 20, 0.7186127154955915), (0.0805, 0, 10_000, 0.6493608562588798)];

    /// The same script, pre-post, with the filtered voltages read 1, 2 and 4 steps late instead of
    /// the code's 3, as `(delay, period ms, final w)`.
    const DELAYED: [(usize, usize, f64); 6] = [
        (1, 10_000, 0.5000010268167033),
        (1, 20, 0.7978469004682445),
        (2, 10_000, 0.5000010261263226),
        (2, 20, 0.8108807441955364),
        (4, 10_000, 0.5000010243992341),
        (4, 20, 0.84036509869193),
    ];

    /// `tools/clopath_figures.py`: Fig. 2b's two model lines, read from the vector paths of the
    /// PDF against the panel's own axis ticks (three per axis, on a line to 0.017 %), at 0.1, 10, 20,
    /// 40 and 50 Hz, pre-post then post-pre, in %, unrounded as its last block prints them.
    const FIGURE_2B: [f64; 10] = [
        100.96362645344767,
        104.34776491858337,
        117.27307567060024,
        148.25236961856092,
        167.24552382992638,
        71.67640047884043,
        59.58794322763427,
        60.485133453187736,
        118.7657724979926,
        175.89999860964267,
    ];

    /// `tools/clopath_reference.py`: post-pre pairing, lag −10 ms, at periods of 10 000, 100, 50 and
    /// 20 ms with one of `VoTri.m`'s two train quirks undone, as `(period ms, final w)`: every group's
    /// last postsynaptic spike — the sixth — removed.
    const NO_SIXTH: [(usize, f64); 4] =
        [(10_000, 0.36923460636352695), (100, 0.2980603324521832), (50, 0.29799520059776513), (20, 0.7323261924840213)];

    /// The same, with both trains delayed by three samples so that the loop, which starts at `t = 4`,
    /// delivers the first postsynaptic spike.
    const FIRST_DELIVERED: [(usize, f64); 4] =
        [(10_000, 0.3557608698899583), (100, 0.2953541488314974), (50, 0.2998027038589923), (20, 0.8708415058404853)];

    /// `tools/clopath_reference.py`: the transcribed `aEIF.m` from rest under 1 000 pA for 2 000
    /// calls, the call of every spike, with the code's `VT_jump = 20` (`V_Tmax` = −30.4 mV).
    const SPIKES_CODE: [usize; 97] = [
        13, 30, 47, 65, 83, 101, 120, 139, 158, 177, 197, 217, 237, 257, 277, 297, 318, 339, 360, 381, 402, 423, 444,
        465, 486, 507, 528, 549, 570, 591, 612, 633, 654, 675, 696, 717, 738, 759, 780, 801, 822, 843, 864, 885, 906,
        927, 948, 969, 990, 1011, 1032, 1053, 1074, 1095, 1116, 1137, 1158, 1179, 1200, 1221, 1242, 1263, 1284, 1305,
        1326, 1347, 1368, 1389, 1410, 1431, 1452, 1473, 1494, 1515, 1536, 1557, 1578, 1599, 1620, 1641, 1662, 1683,
        1704, 1725, 1746, 1767, 1788, 1809, 1830, 1851, 1872, 1893, 1914, 1935, 1956, 1977, 1998,
    ];

    /// The same with `VT_jump = 80.8`, the jump Table 1a's unsigned `V_Tmax` of +30.4 mV needs.
    const SPIKES_PRINTED: [usize; 18] =
        [13, 114, 221, 331, 442, 554, 666, 778, 890, 1002, 1114, 1226, 1338, 1450, 1562, 1674, 1786, 1898];

    /// `tools/clopath_reference.py`: on the transcribed `aEIF.m` under 1 000 pA from rest, `u − E_L`
    /// and its square low-passed from zero by an exact exponential filter; after 5 000 and 20 000
    /// calls, the mean square over the square of the mean, both over 1.2 s, and the mean square over
    /// 1.2 s over the square of the mean over 1 s.
    const HOMEOSTAT: [[f64; 2]; 2] = [[1.2804601986224244, 1.2585834187537077], [1.2625372185933816, 1.262094964820008]];

    /// Fig. 2b's normalization, `VoTri.m`'s `w = 0.5` start: `100 + 100 (w − 0.5)/0.5`.
    fn percent(w: f64) -> f64 {
        100.0 + 100.0 * (w - 0.5) / 0.5
    }

    /// To the one decimal the module doc quotes.
    fn tenth(x: f64) -> f64 {
        (x * 10.0).round() / 10.0
    }

    fn fig_2b(period: usize, lag: i32) -> f64 {
        votri_pairing(Rule::VISUAL_CORTEX, Membrane::AEIF_M, 3, period, lag).unwrap()
    }

    /// Table 1b, cell by cell, and the code's `par` is its visual-cortex row.
    ///
    /// `FrequencyDependence.m` passes `par = [0.00014 0.00008 7 15 10]` and `VoTri.m` reads it as
    /// `A_m`, `A_p`, `tau_p`, `tau_r`, `tau_d`: `A_LTD`, `A_LTP` — the column Table 1b heads
    /// "`A_LTD` (mV⁻²)" — `τ_+`, `τ_x`, `τ_−`. The hippocampal row's blank `τ_−` and `τ_+` are refused by
    /// name, first the one, then the other.
    #[test]
    fn table_1b_cell_by_cell_with_its_blank_cells_refused() {
        let row = |r: Rule| (r.theta_minus, r.theta_plus, r.a_ltd, r.a_ltp, r.tau_x, r.tau_minus, r.tau_plus);
        assert_eq!(row(Rule::VISUAL_CORTEX), (-70.6, -45.3, 14e-5, 8e-5, 15.0, Some(10.0), Some(7.0)));
        assert_eq!(row(Rule::SOMATOSENSORY_CORTEX), (-70.6, -45.3, 21e-5, 30e-5, 30.0, Some(6.0), Some(5.0)));
        assert_eq!(row(Rule::HIPPOCAMPAL), (-41.0, -38.0, 38e-5, 2e-5, 16.0, None, None));
        let v = Rule::VISUAL_CORTEX;
        assert_eq!((v.a_ltd, v.a_ltp, v.tau_plus, v.tau_x, v.tau_minus), (0.00014, 0.00008, Some(7.0), 15.0, Some(10.0)));
        for r in [Rule::VISUAL_CORTEX, Rule::SOMATOSENSORY_CORTEX, Rule::HIPPOCAMPAL] {
            assert_eq!(r.check(), Ok(()));
        }
        let build = |r: Rule| Synapses::new(r, Integrator::Exponential, 1e-3, 0, -0.07, vec![1.0], UNBOUNDED);
        assert_eq!(
            build(Rule::HIPPOCAMPAL).unwrap_err().to_string(),
            "tau_minus is blank in Table 1b for this set; the paper prints no value"
        );
        assert_eq!(
            build(Rule { tau_minus: Some(10.0), ..Rule::HIPPOCAMPAL }).unwrap_err().to_string(),
            "tau_plus is blank in Table 1b for this set; the paper prints no value"
        );
        assert!(build(Rule { tau_minus: Some(10.0), tau_plus: Some(7.0), ..Rule::HIPPOCAMPAL }).is_ok());
        // u²_ref for Figs. 5, 6 and 7, PDF p. 12: 60, 80 and 50 mV², in V².
        let mv2 = [U_REF_SQ_FIG5, U_REF_SQ_FIG6, U_REF_SQ_FIG7].map(|v| v * 1e6);
        assert_eq!(mv2, [60.0, 80.0, 50.0]);
        // Fig. 3's alternative sets print A_LTP as 50 and 67 × 10⁻⁴ mV⁻²: 17 and 22 times this row's.
        let ratios = [50e-4, 67e-4].map(|a: f64| (a / Rule::SOMATOSENSORY_CORTEX.a_ltp).round());
        assert_eq!(ratios, [17.0, 22.0]);
    }

    /// Table 1a as printed and as `aEIF.m` runs it, and the printed table refused for what it lacks.
    ///
    /// The code's values, each as it computes them: `b = 0.0805`, `V_T = VT_jump + VT_rest = 20 − 50.4`,
    /// `th = 20`, `29.4` and `29.4 + 3.462`, `E_L + 15 + 6.0984` = −49.5016 mV.
    #[test]
    fn table_1a_as_printed_and_as_the_code_runs_it() {
        let cells = |m: Membrane| {
            [m.c, m.g_l, m.e_l, m.delta_t, m.v_t_rest, m.tau_w, m.a, m.b, m.i_sp, m.tau_z, m.tau_v_t, m.v_t_max]
        };
        let printed = Membrane::TABLE_1A;
        assert_eq!(cells(printed), [281.0, 30.0, -70.6, 2.0, -50.4, 144.0, 4.0, 0.805, 400.0, 40.0, 50.0, 30.4]);
        assert_eq!((printed.v_detect, printed.peak, printed.v_reset), (None, None, None));
        let code = Membrane::AEIF_M;
        assert_eq!(cells(code), [281.0, 30.0, -70.6, 2.0, -50.4, 144.0, 4.0, 0.0805, 400.0, 40.0, 50.0, -30.4]);
        assert_eq!((code.v_detect, code.peak), (Some(20.0), Some([29.4, 32.862])));
        assert!((code.v_reset.unwrap() - -49.5016).abs() < 3e-14, "{:?}", code.v_reset);
        let refused = |m: Membrane| Neuron::new(m).unwrap_err().to_string();
        let tail = "is not printed in the paper; Membrane::AEIF_M takes it from the authors' aEIF.m";
        assert_eq!(refused(printed), format!("v_detect {tail}"));
        assert_eq!(refused(Membrane { v_detect: code.v_detect, ..printed }), format!("peak {tail}"));
        assert_eq!(refused(Membrane { v_detect: code.v_detect, peak: code.peak, ..printed }), format!("v_reset {tail}"));
        assert!(Neuron::new(Membrane { v_detect: code.v_detect, peak: code.peak, v_reset: code.v_reset, ..printed }).is_ok());
    }

    /// `θ_+ = −45.3` mV, "the firing threshold of the `AdEx` model", matches where Table 1a's
    /// membrane current turns positive for good; the paper gives no derivation.
    ///
    /// With `w_ad = z = I = 0` and `V_T = V_Trest`, the current
    /// `−g_L (u − E_L) + g_L Δ_T e^{(u − V_Trest)/Δ_T}` has an upper zero — ref. 49's `V_s`, "above
    /// which an action potential upstroke is generated" — found here by bisection between `V_Trest`
    /// and 0 mV at −45.3268 mV: `θ_+` to the one decimal Table 1b prints. Counting the steady-state
    /// adaptation `a (u − E_L)` as well puts it at −45.0551 mV, which would print as −45.1. `θ_−` is
    /// `E_L`.
    #[test]
    fn theta_plus_matches_the_adex_currents_upper_zero() {
        let m = Membrane::TABLE_1A;
        let root = |a: f64| {
            let current = |u: f64| -(m.g_l + a) * (u - m.e_l) + m.g_l * m.delta_t * ((u - m.v_t_rest) / m.delta_t).exp();
            let (mut lo, mut hi) = (m.v_t_rest, 0.0);
            assert!(current(lo) < 0.0 && current(hi) > 0.0);
            for _ in 0..100 {
                let mid = 0.5 * (lo + hi);
                if current(mid) < 0.0 { lo = mid } else { hi = mid }
            }
            lo
        };
        let bare = root(0.0);
        let adapted = root(m.a);
        assert!((bare - -45.3268).abs() < 5e-5 && (adapted - -45.0551).abs() < 5e-5, "{bare} {adapted}");
        for r in [Rule::VISUAL_CORTEX, Rule::SOMATOSENSORY_CORTEX] {
            assert_eq!(tenth(bare), r.theta_plus);
            assert_eq!(r.theta_minus, m.e_l);
        }
        assert_eq!(tenth(adapted), -45.1);
    }

    /// The clamp form of eq. (3) regime by regime, with binary-fraction thresholds so that each
    /// boundary is met exactly: nothing below `θ_−`, depression alone up to `θ_+`, and the quadratic
    /// above — Fig. 1h's shape, times the presynaptic rate.
    #[test]
    fn under_clamp_the_rule_is_rectified_linear_then_quadratic() {
        let r = Rule {
            theta_minus: -62.5,
            theta_plus: -46.875,
            a_ltd: 1.0 / 1024.0,
            a_ltp: 1.0 / 4096.0,
            tau_x: 16.0,
            tau_minus: None,
            tau_plus: None,
        };
        let g = |u: f64| r.clamp_per_spike(u).unwrap();
        assert_eq!(g(-0.078125), 0.0, "below θ_−");
        assert_eq!(g(-0.0625), 0.0, "at θ_−");
        assert_eq!(g(-0.0546875), -7.8125 / 1024.0, "between: depression only");
        assert_eq!(g(-0.046875), -15.625 / 1024.0, "at θ_+");
        assert_eq!(g(-0.03125), -31.25 / 1024.0 + 15.625 * 31.25 / 4096.0, "above: both");
        for nu in [0.0, 2.0, 10.0, 50.0] {
            for u in [-0.078125, -0.0546875, -0.03125] {
                assert_eq!(r.clamp_rate(u, nu).unwrap(), nu * g(u));
            }
        }
    }

    /// Fig. 1h, dashed line: the visual-cortex set, 25 presynaptic spikes, 100 % a weight of one.
    ///
    /// At `θ_+` it is `100 (1 − 25 × 14 × 10⁻⁵ × 25.3)` = 91.145 %; it reaches the figure's 250 % cap at
    /// the root of `25 g(u) = 1.5`, the larger root of a quadratic in `u − θ_−`: −26.5312 mV (the
    /// figure: −26.5 mV). Bisection on [`Rule::clamp_per_spike`] finds the same root to 7.1 × 10⁻¹⁵ mV.
    /// Normalized to 0.5, as Fig. 2b is, the same curve would bottom out at 82.29 %.
    ///
    /// Against the figure's dashed blue line, read from its vector path by `tools/clopath_figures.py`:
    /// its lowest vertex, the one nearest `θ_+`, is at −45.3986 mV and 91.2075 %, where this curve is
    /// 0.028 points lower (the path steps down to it by 0.071 points a vertex); and it reaches the cap
    /// between its vertices at −26.6074 mV (249.20 %) and −26.3930 mV, around the root.
    #[test]
    fn figure_1h_visual_cortex() {
        let r = Rule::VISUAL_CORTEX;
        let pct = |u_mv: f64| 100.0 * (1.0 + 25.0 * r.clamp_per_spike(u_mv * 1e-3).unwrap());
        let at_theta_plus = pct(-45.3);
        assert!((at_theta_plus - 91.145).abs() < 3e-14, "{at_theta_plus}");
        assert!((pct(-45.398601840697125) - 91.20745904682812).abs() < 0.1, "{}", pct(-45.398601840697125));
        let (mut lo, mut hi) = (-45.3, 0.0);
        for _ in 0..100 {
            let mid = 0.5 * (lo + hi);
            if pct(mid) < 250.0 { lo = mid } else { hi = mid }
        }
        // A_LTP p² − (A_LTP (θ_+ − θ_−) + A_LTD) p − 1.5/25 = 0, p = u − θ_−.
        let b = r.a_ltp * (r.theta_plus - r.theta_minus) + r.a_ltd;
        let p = (b + (b * b + 4.0 * r.a_ltp * 0.06).sqrt()) / (2.0 * r.a_ltp);
        let root = p + r.theta_minus;
        assert!((lo - root).abs() < 3e-14, "{lo} {root}");
        assert!((root - -26.5312).abs() < 5e-5 && (root * 10.0).round() / 10.0 == -26.5, "{root}");
        assert!(-26.607379401091663 < root && root < -26.39304190783639 && pct(-26.607379401091663) < 250.0);
        let halved = 100.0 * (1.0 + 25.0 * r.clamp_per_spike(-0.0453).unwrap() / 0.5);
        assert!((halved - 82.29).abs() < 3e-14, "{halved}");
        assert!((r.clamp_crossover().unwrap() - -0.04355).abs() < 3e-17);
        assert_eq!(r.clamp_minimum().unwrap(), -0.0453, "the vertex, −57.075 mV, is below θ_+");
    }

    /// Fig. 1h, red line: the hippocampal set, 100 presynaptic spikes.
    ///
    /// Flat at 100 % up to its own `θ_−` = −41 mV; lowest at `(θ_+ + θ_− + A_LTD/A_LTP)/2` =
    /// `(−38 − 41 + 19)/2` = −30 mV exactly, at 75.8 %, where the figure's red line has its minimum;
    /// back to 100 % at `θ_+ + A_LTD/A_LTP` = −19 mV; 255.8 % at 0 mV, above the 250 % the figure's line
    /// ends at. The figure's red line, read from its vector path by `tools/clopath_figures.py`, is
    /// flat at its lowest, 75.8055 %, on its five vertices from −30.40 to −29.59 mV; at the middle one,
    /// −29.9933 mV, this curve is 0.0055 points lower.
    #[test]
    fn figure_1h_hippocampal_minimum_is_at_minus_30_mv() {
        let r = Rule::HIPPOCAMPAL;
        let pct = |u_mv: f64| 100.0 * (1.0 + 100.0 * r.clamp_per_spike(u_mv * 1e-3).unwrap());
        assert_eq!(pct(-45.0), 100.0);
        assert_eq!(pct(-41.5), 100.0);
        let min = r.clamp_minimum().unwrap();
        assert!((min - -0.030).abs() < 1e-17, "{min}");
        assert!((pct(-30.0) - 75.8).abs() < 3e-14);
        assert!((pct(-29.993279236045733) - 75.80554843741092).abs() < 0.02, "{}", pct(-29.993279236045733));
        assert!(pct(-30.5) > pct(-30.0) && pct(-29.5) > pct(-30.0));
        assert!((r.clamp_crossover().unwrap() - -0.019).abs() < 1e-17);
        assert!((pct(0.0) - 255.8).abs() < 1e-13);
    }

    /// Run `Synapses` clamped at `u_c` volts for `steps` steps with presynaptic spikes at the given
    /// steps; the final weight minus the initial one.
    fn clamped(integrator: Integrator, u_c: f64, spikes: &[usize], steps: usize) -> (f64, Synapses) {
        let mut s = Synapses::new(Rule::VISUAL_CORTEX, integrator, 1e-3, 0, u_c, vec![0.0], UNBOUNDED).unwrap();
        for k in 0..steps {
            s.step(u_c, &[spikes.contains(&k)]).unwrap();
        }
        (s.weights()[0], s)
    }

    /// Under a clamp the filters stay at the clamp and the weight is the closed form, for a finite
    /// window, under either integrator; the rate of the presynaptic spikes does not matter.
    ///
    /// A spike in step `k` depresses by `A_LTD (u_c − θ_−)_+` and then potentiates, by the end of step
    /// `K − 1`, by `A_LTP (u_c − θ_+)_+ (u_c − θ_−)_+` times the trace's integral over the steps after
    /// it: `1 − e^{−(K − 1 − k)h/τ_x}` for the exponential integrator and `1 − (1 − h/τ_x)^{K − 1 − k}`
    /// for Euler's. Measured against those, at −39.0625 mV with 25 spikes at 50 Hz over 800 ms:
    /// 5.6 × 10⁻¹⁷ and 3.3 × 10⁻¹⁶. At −78.125 mV, below `θ_−`, the weight does not move; at
    /// −54.6875 mV, between the thresholds, it falls by 25 `A_LTD (u_c − θ_−)`, to 2.8 × 10⁻¹⁷.
    /// And 100 spikes at 2 Hz and at 50 Hz, each followed by 450 ms, leave the same weight to
    /// 8.3 × 10⁻¹⁴, and 100 × `g(u_c)` to 6.9 × 10⁻¹⁵ — the two printed protocols' rates give the
    /// same weight (their clamp ranges still differ; see the module doc).
    #[test]
    fn under_clamp_the_weight_follows_the_closed_form_at_any_rate() {
        let u_c = -0.0390625;
        let r = Rule::VISUAL_CORTEX;
        let over_minus = -39.0625 - r.theta_minus;
        let over_plus = -39.0625 - r.theta_plus;
        let spikes: Vec<usize> = (0..25).map(|j| 20 * j).collect();
        let k_end = 800;
        for (integrator, left, bound) in [
            (Integrator::Exponential, Box::new(|n: f64| (-n / 15.0).exp()) as Box<dyn Fn(f64) -> f64>, 2e-16),
            (Integrator::Euler, Box::new(|n: f64| (1.0 - 1.0 / 15.0_f64).powf(n)), 1e-15),
        ] {
            let (dw, s) = clamped(integrator, u_c, &spikes, k_end);
            let want: f64 = spikes
                .iter()
                .map(|&k| -r.a_ltd * over_minus + r.a_ltp * over_plus * over_minus * (1.0 - left((k_end - 1 - k) as f64)))
                .sum();
            assert!((dw - want).abs() < bound, "{integrator:?}: {dw} against {want}");
            let (m, p) = s.filtered();
            assert!((m - u_c).abs() < 5e-17 && (p - u_c).abs() < 5e-17, "{integrator:?}: {:?}", s.filtered());
            // The other two regimes of Fig. 1h in the synapse itself: below θ_− nothing moves, and
            // between θ_− and θ_+ each spike depresses by A_LTD (u_c − θ_−) and nothing potentiates.
            assert_eq!(clamped(integrator, -0.078125, &spikes, k_end).0, 0.0, "{integrator:?} below θ_−");
            let between = clamped(integrator, -0.0546875, &spikes, k_end).0;
            let want = -25.0 * r.a_ltd * (-54.6875 - r.theta_minus);
            assert!((between - want).abs() < 1e-16, "{integrator:?}: {between} against {want}");
        }
        let (_, s) = clamped(Integrator::Exponential, u_c, &spikes, 3);
        assert_eq!(s.filtered(), (u_c, u_c), "the exponential filters hold the clamp to the bit");
        let slow: Vec<usize> = (0..100).map(|j| 500 * j).collect();
        let fast: Vec<usize> = (0..100).map(|j| 20 * j).collect();
        let (dw_slow, _) = clamped(Integrator::Exponential, u_c, &slow, 49_500 + 450);
        let (dw_fast, _) = clamped(Integrator::Exponential, u_c, &fast, 1_980 + 450);
        let per_spike = r.clamp_per_spike(u_c).unwrap();
        assert!((dw_slow - dw_fast).abs() < 3e-13, "{dw_slow} {dw_fast}");
        assert!((dw_fast - 100.0 * per_spike).abs() < 3e-14);
    }

    /// Each filter relaxes toward a held potential at its own time constant: `ū = u + (u₀ − u) e^{−nh/τ}`
    /// under the exponential integrator and `ū = u + (u₀ − u)(1 − h/τ)^n` under Euler's, with
    /// `τ_−` = 10 ms and `τ_+` = 7 ms and a step of 0.5 ms so that `h/τ` is not `1/τ`. From −70 mV
    /// toward −50 mV, over 40 steps; the worst residual is 2.8 × 10⁻¹⁴ mV.
    #[test]
    fn the_filters_relax_at_their_own_time_constants() {
        let (u0, u) = (-70.0, -50.0);
        let h = 0.5;
        for integrator in [Integrator::Exponential, Integrator::Euler] {
            let mut s = Synapses::new(Rule::VISUAL_CORTEX, integrator, h * 1e-3, 0, u0 * 1e-3, vec![], UNBOUNDED).unwrap();
            for n in 1..=40 {
                s.step(u * 1e-3, &[]).unwrap();
                let factor = |tau: f64| match integrator {
                    Integrator::Exponential => (-f64::from(n) * h / tau).exp(),
                    Integrator::Euler => (1.0 - h / tau).powi(n),
                };
                let (m, p) = s.filtered();
                let (m_want, p_want) = (u + (u0 - u) * factor(10.0), u + (u0 - u) * factor(7.0));
                assert!((m * 1e3 - m_want).abs() < 1e-13 && (p * 1e3 - p_want).abs() < 1e-13, "{integrator:?} {n}: {m} {p}");
            }
        }
    }

    /// Eq. (8) is the drift of a linear Poisson neuron, below and above its threshold.
    ///
    /// The neuron of eq. (4): `u = θ_− + u^s + βY`, with `Y` a Poisson train at `ν_post` and each
    /// spike a pulse of `β/h` for one step of `h` = 0.1 ms, `β` = 120 mV·ms. `u^s` is held at
    /// `θ_+ − θ_−`, so the subthreshold potential sits exactly at `θ_+` and `(u − θ_+)_+` is `β/h` at a
    /// spike and zero otherwise, as eq. (5) takes it; `α = u^s/ν_post`. An independent Poisson
    /// presynaptic train at 20 Hz. `ϑ` is 14.6 Hz; at 5 Hz the synapse must depress and at 40 Hz
    /// potentiate, each at eq. (8)'s rate. Measured over 1 000 s in 10 s blocks: the drift is 1.08 and
    /// 0.76 standard errors from eq. (8), each standard error 1.3 % and 1.6 % of the drift.
    #[test]
    fn eq_8_is_the_drift_of_a_linear_poisson_neuron() {
        let r = Rule::VISUAL_CORTEX;
        let (beta, h) = (120.0, 0.1);
        let us = r.theta_plus - r.theta_minus;
        let theta = r.bcm_threshold(beta * 1e-6).unwrap();
        for (nu_post, seed) in [(5.0, 81), (40.0, 82)] {
            let alpha = us / (nu_post * 1e-3);
            let want = r.bcm_drift(20.0, nu_post, alpha * 1e-6, beta * 1e-6).unwrap();
            let mut s = Synapses::new(r, Integrator::Exponential, h * 1e-3, 0, r.theta_plus * 1e-3, vec![0.0], UNBOUNDED)
                .unwrap();
            let mut rng = Rng::new(seed);
            let (blocks, per_block) = (100, 100_000);
            let mut drifts = Vec::with_capacity(blocks);
            for _ in 0..blocks {
                let w0 = s.weights()[0];
                for _ in 0..per_block {
                    let post = rng.next_f64() < nu_post * h * 1e-3;
                    let pre = rng.next_f64() < 20.0 * h * 1e-3;
                    let u = if post { r.theta_plus + beta / h } else { r.theta_plus };
                    s.step(u * 1e-3, &[pre]).unwrap();
                }
                drifts.push((s.weights()[0] - w0) / (per_block as f64 * h * 1e-3));
            }
            let mean = drifts.iter().sum::<f64>() / blocks as f64;
            let var = drifts.iter().map(|d| (d - mean) * (d - mean)).sum::<f64>() / (blocks - 1) as f64;
            let se = (var / blocks as f64).sqrt();
            assert!((mean - want).abs() < 3.0 * se, "ν_post = {nu_post}: {mean} against {want} ± {se}");
            assert!(se < 0.03 * want.abs(), "{se} {want}");
            assert_eq!(want > 0.0, nu_post > theta, "{want} at {nu_post} Hz, ϑ = {theta}");
        }
    }

    /// Eq. (8)'s two printed forms agree, it vanishes at `ϑ`, its depression is linear and its
    /// potentiation quadratic in `ν_post`, and without `A_LTP` it is pure depression.
    #[test]
    fn eq_8_has_the_bcm_shape() {
        let r = Rule::SOMATOSENSORY_CORTEX;
        let (alpha, beta) = (500e-6, 120e-6);
        let theta = r.bcm_threshold(beta).unwrap();
        let drift = |post: f64| r.bcm_drift(12.0, post, alpha, beta).unwrap();
        for post in [0.0, 3.0, 10.0, 25.0, 60.0] {
            let factored = (alpha + beta) * 1e6 * beta * 1e6 * r.a_ltp * 12e-3 * post * 1e-3 * (post - theta) * 1e-3 * 1e3;
            let unfactored = (-(alpha + beta) * 1e6 * r.a_ltd * 12e-3 * post * 1e-3
                + (alpha + beta) * 1e6 * r.a_ltp * 12e-3 * beta * 1e6 * post * 1e-3 * post * 1e-3)
                * 1e3;
            assert!((drift(post) - factored).abs() < 2e-15 && (drift(post) - unfactored).abs() < 2e-15, "{post}");
        }
        let ltd_term = (alpha + beta) * 1e6 * r.a_ltd * 12e-3 * theta * 1e-3 * 1e3;
        assert!(drift(theta).abs() < 5e-16 * ltd_term, "{}", drift(theta));
        assert!(drift(0.5 * theta) < 0.0 && drift(2.0 * theta) > 0.0);
        let second = drift(30.0) - 2.0 * drift(20.0) + drift(10.0);
        let want = 2.0 * (alpha + beta) * 1e6 * beta * 1e6 * r.a_ltp * 12e-3 * (10e-3 * 10e-3) * 1e3;
        assert!((second - want).abs() < 3e-15 * want, "{second} {want}");
        let ltd_only = Rule { a_ltp: 0.0, ..r };
        let d = ltd_only.bcm_drift(12.0, 7.0, alpha, beta).unwrap();
        let want = -(alpha + beta) * 1e6 * r.a_ltd * 12e-3 * 7e-3 * 1e3;
        assert!((d - want).abs() < 5e-16 * want.abs(), "{d} {want}");
        assert_eq!(r.bcm_drift(0.0, 30.0, alpha, beta).unwrap(), 0.0);
    }

    /// The printed `β = 1.2 mV` is a mean, not a spike weight.
    ///
    /// The Online Methods' triangular spike, 120 mV high and 1 ms wide at half height, has a base of
    /// 2 ms and an integral of 120 mV·ms; over their 100 ms record it adds 1.2 mV to the mean. With
    /// `β` = 120 mV·ms the visual set's threshold is `1.75/120` per ms = 14.58 Hz; read as 1.2 mV·ms it
    /// is 1 458 Hz.
    #[test]
    fn the_printed_beta_is_the_spikes_share_of_a_mean() {
        let integral = 0.5 * 2.0 * 120.0;
        assert_eq!((integral, integral / 100.0), (120.0, 1.2));
        let r = Rule::VISUAL_CORTEX;
        let good = r.bcm_threshold(integral * 1e-6).unwrap();
        let printed = r.bcm_threshold(1.2e-6).unwrap();
        assert!((good - 14.583333333333334).abs() < 1e-14 && (printed - 1458.3333333333333).abs() < 1e-12, "{good} {printed}");
    }

    /// `A_LTD(ū̄) = A_LTD ū̄²/u²_ref`: the printed amplitude at `ū̄ = u_ref`, four times it at twice
    /// the depolarization, and a BCM threshold that slides with `ū̄²`; and a synapse whose depression
    /// is scaled by it depresses by exactly that factor and potentiates as before.
    #[test]
    fn homeostasis_scales_depression_by_the_square() {
        let r = Rule::VISUAL_CORTEX;
        let at_ref = r.a_ltd_homeostatic(U_REF_SQ_FIG5, U_REF_SQ_FIG5).unwrap();
        let doubled = r.a_ltd_homeostatic(4.0 * U_REF_SQ_FIG6, U_REF_SQ_FIG6).unwrap();
        let slid = Rule { a_ltd: doubled, ..r }.bcm_threshold(120e-6).unwrap();
        assert!((at_ref - r.a_ltd).abs() < 1e-20);
        assert!((doubled - 4.0 * r.a_ltd).abs() < 1e-19);
        assert!((slid - 4.0 * r.bcm_threshold(120e-6).unwrap()).abs() < 3e-14);
        assert_eq!(r.a_ltd_homeostatic(0.0, U_REF_SQ_FIG7).unwrap(), 0.0);
        // In the depression-only band a clamped spike changes the weight by −scale·A_LTD(u − θ_−).
        let u_c = -0.05;
        let run = |scale: f64| {
            let mut s = Synapses::new(r, Integrator::Exponential, 1e-3, 0, u_c, vec![1.0], UNBOUNDED).unwrap();
            s.set_ltd_scale(scale).unwrap();
            s.step(u_c, &[true]).unwrap();
            s.weights()[0] - 1.0
        };
        let one = run(1.0);
        assert!((one - -r.a_ltd * (-50.0 - r.theta_minus)).abs() < 5e-18, "{one}");
        assert!((run(2.5) - 2.5 * one).abs() < 2e-16);
        assert_eq!(run(0.0), 0.0);
        // Above θ_+ potentiation is untouched by the scale.
        let above = |scale: f64| {
            let mut s = Synapses::new(r, Integrator::Euler, 1e-3, 0, -0.03, vec![0.0], UNBOUNDED).unwrap();
            s.set_ltd_scale(scale).unwrap();
            s.step(-0.03, &[true]).unwrap();
            let w1 = s.weights()[0];
            s.step(-0.03, &[false]).unwrap();
            s.weights()[0] - w1
        };
        assert!((above(0.0) - above(3.0)).abs() < 3e-18);
        assert!(above(1.0) > 0.0);
    }

    /// The homeostat's filter is exact, and its mean of the square is never below its square of the
    /// mean.
    ///
    /// Held at a depolarization `d` from zero, both averages follow `W = 1 − e^{−t/τ}` exactly — `d`
    /// and `d²` times it; the worst residuals over the 3 000 steps of 3 s are 2.3 × 10⁻¹⁶ V and
    /// 2.1 × 10⁻¹⁸ V² — and the mean square exceeds the squared mean by the start-up term alone,
    /// `d² W(1 − W)`, to 1.6 × 10⁻¹⁸ V². On the code's neuron under a 1 nA step from rest, with both
    /// readings filtered over `RFdevelop.m`'s 1.2 s, the mean of the square is at least the square of
    /// the mean at every step (they are the same weighted average), and their ratio is 1.2805 after
    /// 5 s, while the filters settle, and 1.2625 after 20 s. Against the paper's reading — the square
    /// of a 1 s mean — the code's is 1.2586 times as large after 5 s and 1.2621 after 20 s: the time
    /// constants barely matter once settled, the variance does. All four ratios are the transcription's
    /// ([`HOMEOSTAT`]), computed there in millivolts from the transcribed `aEIF.m`; the worst
    /// relative difference is 8.9 × 10⁻¹⁶, from this module's round trips through volts.
    #[test]
    fn the_homeostat_keeps_both_readings_of_u_bar_bar() {
        assert_eq!(Homeostat::TAU_CODE, 1.2);
        let mut h = Homeostat::new(Homeostat::TAU_CODE, -0.0706, 1e-3).unwrap();
        let mut worst_gap = 0.0_f64;
        for n in 1..=3000 {
            h.step(-0.0606).unwrap();
            let f = 1.0 - (-(n as f64) * 1e-3 / 1.2).exp();
            assert!((h.mean() - 0.010 * f).abs() < 1e-15 && (h.mean_square() - 1e-4 * f).abs() < 1e-17, "{n}: {h:?}");
            let gap = h.mean_square() - h.mean() * h.mean();
            worst_gap = worst_gap.max((gap - 1e-4 * f * (1.0 - f)).abs());
        }
        println!("start-up term residual {worst_gap:e}");
        assert!(worst_gap < 5e-18, "{worst_gap}");
        let mut n = Neuron::new(Membrane::AEIF_M).unwrap();
        let mut code = Homeostat::new(Homeostat::TAU_CODE, Membrane::AEIF_M.e_l * 1e-3, Neuron::DT).unwrap();
        let mut paper = Homeostat::new(1.0, Membrane::AEIF_M.e_l * 1e-3, Neuron::DT).unwrap();
        let mut ratios = Vec::new();
        for k in 1..=20_000 {
            n.step(1e-9).unwrap();
            code.step(n.potential()).unwrap();
            paper.step(n.potential()).unwrap();
            assert!(code.mean_square() >= code.mean() * code.mean(), "{code:?}");
            if k == 5000 || k == 20_000 {
                ratios.push(code.mean_square() / (code.mean() * code.mean()));
                ratios.push(code.mean_square() / (paper.mean() * paper.mean()));
            }
        }
        let want = [HOMEOSTAT[0][0], HOMEOSTAT[0][1], HOMEOSTAT[1][0], HOMEOSTAT[1][1]];
        let worst = ratios.iter().zip(&want).map(|(got, want)| (got / want - 1.0).abs()).fold(0.0, f64::max);
        println!("homeostat ratios {ratios:?}, worst relative difference {worst:e}");
        assert!(worst < 4e-15, "{ratios:?} against {want:?}");
        assert_eq!(want.map(|r| (r * 1e4).round() / 1e4), [1.2805, 1.2586, 1.2625, 1.2621]);
    }

    /// The neuron under a 1 nA step for 2 s, spike for spike against the transcribed `aEIF.m`.
    ///
    /// Every spike step equals the transcription's ([`SPIKES_CODE`], [`SPIKES_PRINTED`]). With the
    /// code's `V_Tmax` = −30.4 mV that is 97 spikes, 48 of them in the second second; with Table 1a's
    /// unsigned +30.4 mV, 18, the first four on steps 13, 114, 221 and 331 — 101, 107 and 110 ms
    /// apart — and 9 in the second second.
    #[test]
    fn under_a_step_the_neuron_fires_as_aeif_m_does_with_either_sign() {
        for (v_t_max, want) in [(-30.4, &SPIKES_CODE[..]), (30.4, &SPIKES_PRINTED[..])] {
            let mut n = Neuron::new(Membrane { v_t_max, ..Membrane::AEIF_M }).unwrap();
            let spikes: Vec<usize> = (0..2000).filter(|_| n.step(1e-9).unwrap()).collect();
            assert_eq!(spikes, want, "V_Tmax = {v_t_max}");
        }
        // The counts the module doc quotes, read off the transcription's lists.
        let late = |s: &[usize]| s.iter().filter(|&&t| t >= 1000).count();
        assert_eq!((late(&SPIKES_CODE), late(&SPIKES_PRINTED)), (48, 9));
        assert_eq!([1, 2, 3].map(|i| SPIKES_PRINTED[i] - SPIKES_PRINTED[i - 1]), [101, 107, 110]);
    }

    /// The spike is `aEIF.m`'s: detected, held at 29.4 mV, then at 32.862 mV with that millisecond's
    /// adaptation update undone, then reset with `w += b`, `z = I_sp`, `V_T = V_Tmax` — after which
    /// `z` and `V_T` relax geometrically, `z_n = I_sp (1 − 1/τ_z)^n` and
    /// `V_T − V_Trest = (V_Tmax − V_Trest)(1 − 1/τ_VT)^n`.
    #[test]
    fn the_spike_is_the_codes_two_clamped_milliseconds() {
        let m = Membrane::AEIF_M;
        let mut n = Neuron::new(m).unwrap();
        assert_eq!((n.potential(), n.threshold(), n.adaptation(), n.afterpotential()), (-0.0706, -0.0504, 0.0, 0.0));
        assert!(n.step(1e-6).unwrap(), "10⁶ pA fires in one step");
        assert_eq!(n.potential(), 0.0294);
        let w_detected = n.adaptation();
        assert!(!n.step(0.0).unwrap());
        assert_eq!(n.potential(), 0.032862);
        assert_eq!(n.adaptation(), w_detected, "the second millisecond's wdot is undone");
        // The reset step, computed from the code's statements.
        let (u, w) = (m.v_reset.unwrap(), w_detected * 1e12 + m.b);
        let udot = 1.0 / m.c * (-m.g_l * (u - m.e_l) + m.g_l * m.delta_t * ((u - m.v_t_max) / m.delta_t).exp() - w + m.i_sp);
        let wdot = 1.0 / m.tau_w * (m.a * (u - m.e_l) - w);
        assert!(!n.step(0.0).unwrap());
        assert_eq!((n.potential(), n.adaptation()), ((u + udot) * 1e-3, (w + wdot) * 1e-12));
        for k in 1..=200 {
            let (z, v_t) = (n.afterpotential(), n.threshold());
            let z_want = m.i_sp * (1.0 - 1.0 / m.tau_z).powi(k) * 1e-12;
            let v_want = (m.v_t_rest + (m.v_t_max - m.v_t_rest) * (1.0 - 1.0 / m.tau_v_t).powi(k)) * 1e-3;
            assert!((z - z_want).abs() < 1e-24 && (v_t - v_want).abs() < 3e-16, "step {k}: {z} {z_want} {v_t} {v_want}");
            assert!(!n.step(0.0).unwrap(), "the afterpotential alone does not fire it");
        }
    }

    /// The fixed point of the Euler map is the model's: the potential where
    /// `(g_L + a)(u − E_L) − g_L Δ_T e^{(u − V_Trest)/Δ_T} = I`, found here by bisection.
    ///
    /// At rest that is 7.25 × 10⁻⁵ mV above `E_L`; under 400 pA, 11.79 mV above it. After 5 s from
    /// `E_L` the neuron sits within 5.6 × 10⁻¹⁷ V of the root in both.
    #[test]
    fn the_resting_potential_solves_the_steady_state_equation() {
        let m = Membrane::AEIF_M;
        for (i_pa, above) in [(0.0, 7.25e-5), (400.0, 11.79)] {
            let f = |u: f64| (m.g_l + m.a) * (u - m.e_l) - m.g_l * m.delta_t * ((u - m.v_t_rest) / m.delta_t).exp() - i_pa;
            let (mut lo, mut hi) = (m.e_l - 1.0, m.v_t_rest);
            for _ in 0..200 {
                let mid = 0.5 * (lo + hi);
                if f(mid) < 0.0 { lo = mid } else { hi = mid }
            }
            let mut n = Neuron::new(m).unwrap();
            for _ in 0..5000 {
                assert!(!n.step(i_pa * 1e-12).unwrap());
            }
            assert!(((lo - m.e_l) / above - 1.0).abs() < 5e-3, "{i_pa} pA: {}", lo - m.e_l);
            assert!((n.potential() - lo * 1e-3).abs() < 2e-16, "{i_pa} pA: {} against {}", n.potential(), lo * 1e-3);
            assert!((n.adaptation() - m.a * (lo - m.e_l) * 1e-12).abs() < 1e-24);
        }
    }

    /// Firing alone changes no weight — Fig. 1c: "No weight change was observed". With no
    /// presynaptic spike there is no depression and the trace stays at zero, so the product that
    /// potentiates is zero whatever the potential does. Under the 1 nA step for 2 s — the
    /// transcription's 97 spikes, [`SPIKES_CODE`] — with the code's delay and without it, every weight
    /// ends exactly where it started; one presynaptic spike in the same run moves its own synapse and
    /// no other.
    #[test]
    fn firing_alone_changes_no_weight() {
        for (integrator, delay) in [(Integrator::Euler, 3), (Integrator::Exponential, 0)] {
            let mut n = Neuron::new(Membrane::AEIF_M).unwrap();
            let w0 = vec![0.25, 0.5, 1.0];
            let mut s = Synapses::new(Rule::VISUAL_CORTEX, integrator, Neuron::DT, delay, -0.0706, w0.clone(), UNBOUNDED).unwrap();
            let mut spikes = 0;
            for t in 0..2000 {
                spikes += usize::from(n.step(1e-9).unwrap());
                s.step(n.potential(), &[false, t == 1005, false]).unwrap();
            }
            assert_eq!(spikes, SPIKES_CODE.len(), "{integrator:?}");
            assert_eq!((s.weights()[0], s.weights()[2]), (w0[0], w0[2]), "{integrator:?}");
            assert!(s.weights()[1] != w0[1], "{integrator:?}");
        }
    }

    /// `VoTri.m`'s trains: five pairings a group, 15 groups of 10 s, 10 groups of 50.011 s at 0.1 Hz,
    /// and one extra postsynaptic spike per group when the lag is negative.
    #[test]
    fn the_protocol_is_votri_ms() {
        let count = |v: &[bool]| v.iter().filter(|&&b| b).count();
        let at = |v: &[bool]| v.iter().enumerate().filter(|&(_, &b)| b).map(|(t, _)| t).collect::<Vec<_>>();
        let (pre, post) = votri_trains(20, 10).unwrap();
        assert_eq!((pre.len(), count(&pre), count(&post)), (150_000, 75, 75));
        assert_eq!(at(&pre[..10_000]), [10, 30, 50, 70, 90]);
        assert_eq!(at(&post[..10_000]), [20, 40, 60, 80, 100]);
        let (pre, post) = votri_trains(20, -10).unwrap();
        assert_eq!((pre.len(), count(&pre), count(&post)), (150_000, 75, 90));
        assert_eq!(at(&pre[10_000..20_000]), [10, 30, 50, 70, 90]);
        assert_eq!(at(&post[10_000..20_000]), [0, 20, 40, 60, 80, 100], "the sixth: 10 ms after the last pre");
        let (pre, post) = votri_trains(10_000, 10).unwrap();
        assert_eq!((pre.len(), count(&pre), count(&post)), (500_110, 50, 50));
        assert_eq!(at(&pre[..50_011]), [10, 10_010, 20_010, 30_010, 40_010]);
        let (pre, post) = votri_trains(10_000, -10).unwrap();
        assert_eq!((pre.len(), count(&pre), count(&post)), (500_110, 50, 60));
        assert_eq!(at(&post[..50_011]), [0, 10_000, 20_000, 30_000, 40_000, 50_000]);
        let (pre, post) = votri_trains(1, 0).unwrap();
        assert_eq!((pre.len(), count(&pre), count(&post)), (150_000, 75, 90));
        for (period, lag) in [(0, 0), (10_001, 0), (20, 20), (20, -20)] {
            assert_eq!(
                votri_trains(period, lag).unwrap_err().to_string(),
                format!("a period of {period} ms with a lag of {lag} ms: votri_trains takes 1 <= period <= 10000 and |lag| < period")
            );
        }
        assert!(votri_trains(10_000, 9_999).is_ok() && votri_trains(20, -19).is_ok());
    }

    /// The `ModelDB` code's Fig. 2b weights, bit for bit against the transcription: all ten weights
    /// of the transcribed `VoTri.m`, and the post-pre weights at 29, 28 and 27 ms, reproduced by this
    /// module's parts BIT FOR BIT.
    ///
    /// As normalized weights they are 100.0/102.9/115.7/146.3/165.0 % pre-post and
    /// 71.7/59.6/60.5/117.8/173.6 % post-pre. The published figure is not this code's output
    /// throughout. Against its vector paths ([`FIGURE_2B`]) the post-pre line at 0.1, 10 and 20 Hz is
    /// the code's to within 0.03 points — measured, 0.010, 0.030 and 0.026 — and above it by 1.0 and
    /// 2.3 points at 40 and 50 Hz; the pre-post line is above the code's at all five, by 1.0, 1.4,
    /// 1.6, 2.0 and 2.2 points.
    #[test]
    fn votri_m_weights_bit_for_bit() {
        let quoted = [100.0, 102.9, 115.7, 146.3, 165.0, 71.7, 59.6, 60.5, 117.8, 173.6];
        let mut gap = [0.0; 10];
        for (k, &(period, lag, want)) in REFERENCE.iter().enumerate() {
            let got = fig_2b(period, lag);
            assert_eq!(got, want, "{period} ms, {lag:+} ms");
            if k < 10 {
                assert_eq!(tenth(percent(got)), quoted[k], "{period} ms, {lag:+} ms");
                gap[k] = FIGURE_2B[k] - percent(got);
            }
        }
        println!("figure minus model {gap:?}");
        assert!(gap[5..8].iter().all(|g| g.abs() < 0.1), "{gap:?}");
        assert_eq!(gap.map(tenth), [1.0, 1.4, 1.6, 2.0, 2.2, 0.0, 0.0, 0.0, 1.0, 2.3], "{gap:?}");
        assert_eq!(gap[5..8].iter().map(|g| (g.abs() * 1e3).round() / 1e3).collect::<Vec<_>>(), [0.010, 0.030, 0.026]);
    }

    /// Fig. 2b's post-pre line carries both of `VoTri.m`'s train quirks: undone one at a time, each
    /// moves the model off the figure where the code's own trains sit on it.
    ///
    /// Bit for bit against the transcription ([`NO_SIXTH`], [`FIRST_DELIVERED`]). With the code's
    /// trains the model is within 0.03 points of the figure at 0.1, 10 and 20 Hz
    /// ([`votri_m_weights_bit_for_bit`]). With every group's sixth postsynaptic spike removed, it is
    /// 2.17 points from the figure at 0.1 Hz, 0.02 at 10 Hz and 0.89 at 20 Hz — and 29.43 at 50 Hz,
    /// where that spike falls 10 ms after the group's last presynaptic one. With both trains delayed
    /// three samples, so that the first postsynaptic spike is delivered, it is 0.52 points from the
    /// figure at each of the three, and 1.73 at 50 Hz (the code's: 2.34).
    #[test]
    fn fig_2b_carries_both_train_quirks() {
        let run = |pre: &[bool], post: &[bool]| votri_run(Rule::VISUAL_CORTEX, Membrane::AEIF_M, 3, pre, post).unwrap();
        let mut gaps = [[0.0; 4]; 2];
        for (j, (&(period, no_sixth), &(also, first))) in NO_SIXTH.iter().zip(&FIRST_DELIVERED).enumerate() {
            assert_eq!(period, also);
            let figure = FIGURE_2B[[5, 6, 7, 9][j]];
            let (pre, mut post) = votri_trains(period, -10).unwrap();
            let delayed = |v: &[bool]| [false; 3].iter().chain(v).copied().collect::<Vec<bool>>();
            let got = run(&delayed(&pre), &delayed(&post));
            assert_eq!(got, first, "first delivered, {period} ms");
            gaps[1][j] = figure - percent(got);
            let groups = if period == 10_000 { 10 } else { 15 };
            for group in post.chunks_mut(pre.len() / groups) {
                let last = group.iter().rposition(|&b| b).unwrap();
                assert_eq!(last, 5 * period, "the sixth spike ends its group");
                group[last] = false;
            }
            assert_eq!(post.iter().filter(|&&b| b).count(), 5 * groups);
            let got = run(&pre, &post);
            assert_eq!(got, no_sixth, "no sixth, {period} ms");
            gaps[0][j] = figure - percent(got);
        }
        println!("figure minus model: no sixth {:?}, first delivered {:?}", gaps[0], gaps[1]);
        assert_eq!(gaps.map(|g| g.map(|x| (x * 100.0).round() / 100.0)), [[-2.17, -0.02, 0.89, 29.43], [0.52, 0.52, 0.52, 1.73]]);
        assert!(gaps.iter().all(|g| g[..3].iter().any(|x| x.abs() > 0.4)), "{gaps:?}");
    }

    /// The main text's statements about Fig. 2b, on the code's model.
    ///
    /// Pre-post at 0.1 Hz "did not show any change": 100.0002 %. Pre-post "above 10 Hz yielded LTP".
    /// Post-pre gave LTD "below 35 Hz" — at every whole-millisecond period Fig. 2b draws from 29 ms
    /// (34.5 Hz) to 100 ms (10 Hz), and at 0.1 Hz; the weakest is 94.1 % at 29 ms. It turns
    /// to LTP between 28 ms (98.9 %) and 27 ms (104.4 %). At 50 Hz
    /// post-pre is "nearly indistinguishable from a pre-post timing and LTP dominates": 173.6 % against
    /// 165.0 %.
    #[test]
    fn the_main_texts_statements_about_figure_2b_hold() {
        assert!((percent(fig_2b(10_000, 10)) - 100.0).abs() < 3e-4);
        for period in [100, 50, 25, 20] {
            assert!(percent(fig_2b(period, 10)) > 100.0, "{period}");
        }
        let post_pre: Vec<f64> = (29..=100).chain([10_000]).map(|period| percent(fig_2b(period, -10))).collect();
        let weakest = post_pre.iter().copied().fold(0.0, f64::max);
        assert!(weakest < 100.0 && tenth(weakest) == 94.1 && weakest == post_pre[0], "{post_pre:?}");
        assert_eq!((tenth(percent(REFERENCE[11].2)), tenth(percent(REFERENCE[12].2))), (98.9, 104.4));
        let (post_pre, pre_post) = (percent(fig_2b(20, -10)), percent(fig_2b(20, 10)));
        assert!(post_pre > pre_post && pre_post > 100.0, "{post_pre} {pre_post}");
    }

    /// Each doubtful value moved on its own, bit for bit against the transcribed code.
    ///
    /// `b` as Table 1a prints it, 0.805 pA: 50 Hz pre-post goes from 165.0 % to 164.8 %. `b` in the
    /// unit the code's comment names, 0.0805 nA = 80.5 pA — ref. 49's value, the source the Online
    /// Methods name: to 143.7 %, far from the figure's 167.25 % ([`FIGURE_2B`]).
    /// And the filtered voltages read with no delay, as eq. (3) prints them: 0.1 Hz pre-post goes from
    /// no change to 129.9 %.
    #[test]
    fn the_doubtful_values_moved_one_at_a_time() {
        for ((b, delay, period, want), quoted) in DOUBTFUL.into_iter().zip([164.8, 143.7, 129.9]) {
            let got = votri_pairing(Rule::VISUAL_CORTEX, Membrane { b, ..Membrane::AEIF_M }, delay, period, 10).unwrap();
            assert_eq!(got, want, "b = {b}, delay {delay}, {period} ms");
            assert_eq!(tenth(percent(got)), quoted);
        }
    }

    /// Fig. 2b needs a read delay, but not three steps: any delay from one step to five gives no change
    /// at 0.1 Hz pre-post.
    ///
    /// Read 0 steps late 0.1 Hz pre-post ends at 129.9 %; read 1, 2, 3, 4 or 5 steps late, at
    /// 100.0 % — for 1, 2 and 4 steps bit for bit against the transcription ([`DELAYED`]). The delay
    /// moves the rest of the curve: 50 Hz pre-post is 159.6, 162.2, 165.0 and 168.1 % for 1 to 4
    /// steps. The published post-pre line ([`FIGURE_2B`]) singles out the code's 3 at 0.1, 10 and
    /// 20 Hz: with a delay of 3 steps the model is within 0.03 points of it there, and with 1, 2, 4 or
    /// 5 steps at least 0.48 points from it at each of the three.
    #[test]
    fn a_read_delay_is_needed_and_three_is_the_codes_choice() {
        let run = |delay: usize, period: usize, lag: i32| votri_pairing(Rule::VISUAL_CORTEX, Membrane::AEIF_M, delay, period, lag).unwrap();
        for (delay, period, want) in DELAYED {
            assert_eq!(run(delay, period, 10), want, "delay {delay}, {period} ms");
        }
        assert_eq!([0, 1, 2, 3, 4, 5].map(|d| tenth(percent(run(d, 10_000, 10)))), [129.9, 100.0, 100.0, 100.0, 100.0, 100.0]);
        assert_eq!([1, 2, 3, 4].map(|d| tenth(percent(run(d, 20, 10)))), [159.6, 162.2, 165.0, 168.1]);
        let miss = |delay: usize| [(10_000, 5), (100, 6), (50, 7)].map(|(period, k)| (FIGURE_2B[k] - percent(run(delay, period, -10))).abs());
        assert!(miss(3).iter().all(|&m| m < 0.1), "{:?}", miss(3));
        for delay in [1, 2, 4, 5] {
            assert!(miss(delay).iter().all(|&m| m > 0.4), "delay {delay}: {:?}", miss(delay));
        }
    }

    /// The time-stepped rule converges at first order under either integrator: halving the step
    /// halves the error, and both converge to eq. (3)'s weight in continuous time.
    ///
    /// The potential `u(t) = −50 + 25 sin(ωt)` mV, `ω = 2π/40 ms`, crosses both thresholds;
    /// presynaptic spikes every 8 ms from `t = 0`, 25 in all; 200 ms. Against a run at `h` = 0.1/64 ms,
    /// the errors at 0.1, 0.05 and 0.025 ms shrink by measured ratios of 2.04 and 2.07 (exponential)
    /// and 2.09 and 2.09 (Euler). Each scheme's finest run is compared only with itself there, so the
    /// limit is checked apart, against eq. (3) itself. Both filters, started at −50 mV, have the
    /// closed form `ū(t) = −50 + 25 (sin ωt − ωτ cos ωt + ωτ e^{−t/τ})/(1 + ω²τ²)`, which never
    /// reaches `θ_−`; so the weight at 200 ms is
    /// `1 − A_LTD Σ_k (ū_−(t_k) − θ_−) + A_LTP ∫ x̄ (u − θ_+)_+ (ū_+ − θ_−) dt`, `x̄` the sum of the
    /// spikes' decaying `e^{−(t − t_k)/τ_x}/τ_x`, and the integral is taken by Simpson's rule between
    /// the spikes and the zeros of `u − θ_+`: 1.243799069, the same to 6.7 × 10⁻¹⁶ with 2 000 or
    /// 4 000 intervals a piece. The two finest runs still differ from each other by 1.3 × 10⁻⁵, but
    /// extrapolated to a zero step, `2 w(h) − w(2h)` at `h` = 0.1/64 ms, they reach it to
    /// 1.8 × 10⁻⁹ (exponential) and 1.9 × 10⁻¹⁰ (Euler).
    #[test]
    fn the_rule_converges_at_first_order() {
        let r = Rule::VISUAL_CORTEX;
        let omega = 2.0 * std::f64::consts::PI / 40.0;
        let run = |integrator: Integrator, h: f64| {
            let steps = (200.0 / h).round() as usize;
            let every = (8.0 / h).round() as usize;
            let u = |k: usize| -50.0 + 25.0 * (omega * k as f64 * h).sin();
            let mut s = Synapses::new(r, integrator, h * 1e-3, 0, -0.05, vec![1.0], UNBOUNDED).unwrap();
            for k in 0..steps {
                s.step(u(k) * 1e-3, &[k % every == 0]).unwrap();
            }
            s.weights()[0]
        };
        // Eq. (3) in continuous time for the same input.
        let filtered = |t: f64, tau: f64| {
            let wt = omega * tau;
            -50.0 + 25.0 * ((omega * t).sin() - wt * (omega * t).cos() + wt * (-t / tau).exp()) / (1.0 + wt * wt)
        };
        let (tau_minus, tau_plus) = (r.tau_minus.unwrap(), r.tau_plus.unwrap());
        // The transient term is not negative, so each filter stays above −50 − 25/√(1 + ω²τ²):
        // −63.4 and −66.8 mV, both above θ_−, and neither rectification of a filter ever acts.
        for tau in [tau_minus, tau_plus] {
            assert!(-50.0 - 25.0 / (1.0 + (omega * tau).powi(2)).sqrt() > r.theta_minus, "{tau}");
        }
        let spikes: Vec<f64> = (0..25).map(|k| 8.0 * f64::from(k)).collect();
        let depression: f64 = spikes.iter().map(|&t| r.a_ltd * (filtered(t, tau_minus) - r.theta_minus)).sum();
        // `u − θ_+` changes sign where sin ωt = (θ_+ + 50)/25.
        let root = ((r.theta_plus + 50.0) / 25.0).asin() / omega;
        let mut cuts: Vec<f64> = spikes.clone();
        for n in 0..5 {
            cuts.extend([root + 40.0 * f64::from(n), 20.0 - root + 40.0 * f64::from(n)]);
        }
        cuts.push(200.0);
        cuts.sort_by(f64::total_cmp);
        let potentiation = |intervals: usize| {
            let mut total = 0.0;
            for piece in cuts.windows(2) {
                let (a, b) = (piece[0], piece[1]);
                let fired: Vec<f64> = spikes.iter().copied().filter(|&t| t <= a).collect();
                let f = |t: f64| {
                    let trace: f64 = fired.iter().map(|&tk| (-(t - tk) / r.tau_x).exp() / r.tau_x).sum();
                    let momentary = (-50.0 + 25.0 * (omega * t).sin() - r.theta_plus).max(0.0);
                    trace * momentary * (filtered(t, tau_plus) - r.theta_minus)
                };
                let step = (b - a) / intervals as f64;
                let inner: f64 = (1..intervals).map(|i| if i % 2 == 1 { 4.0 } else { 2.0 } * f(a + i as f64 * step)).sum();
                total += (f(a) + inner + f(b)) * step / 3.0;
            }
            r.a_ltp * total
        };
        let exact = 1.0 - depression + potentiation(2000);
        let quadrature = (exact - (1.0 - depression + potentiation(4000))).abs();
        let (mut limits, mut finest) = (Vec::new(), Vec::new());
        for integrator in [Integrator::Exponential, Integrator::Euler] {
            let fine = run(integrator, 0.1 / 64.0);
            finest.push(fine);
            let err: Vec<f64> = [0.1, 0.05, 0.025].iter().map(|&h| (run(integrator, h) - fine).abs()).collect();
            let ratios: Vec<f64> = err.windows(2).map(|p| p[0] / p[1]).collect();
            println!("{integrator:?}: error ratios {ratios:?}");
            for ratio in &ratios {
                assert!((1.9..2.2).contains(ratio), "{integrator:?}: {err:?}");
            }
            limits.push(2.0 * fine - run(integrator, 0.1 / 32.0));
        }
        println!("exact {exact}, quadrature {quadrature:e}, limits {limits:?}, finest {:e}", finest[0] - finest[1]);
        assert!(quadrature < 3e-15, "{quadrature}");
        assert!((exact - 1.243799069).abs() < 5e-10, "{exact}");
        assert!(limits.iter().all(|l| (l - exact).abs() < 5e-9), "{limits:?} against {exact}");
    }

    /// The trace jumps by `1/τ_x` and, summed over the potentiation windows, integrates to one per
    /// spike under either integrator — the convention eq. (2)'s mV⁻² needs.
    ///
    /// Clamped at −30 mV with no depression, one spike potentiates by `A_LTP (u − θ_+)(u − θ_−)` × 1
    /// once the trace has died away: measured, to 1.4 × 10⁻¹⁷ (exponential) and 1.0 × 10⁻¹⁶ (Euler).
    #[test]
    fn the_trace_jumps_by_one_over_tau_x_and_integrates_to_one() {
        let r = Rule::VISUAL_CORTEX;
        for (integrator, decay) in [(Integrator::Exponential, (-0.5 / 15.0_f64).exp()), (Integrator::Euler, 1.0 - 0.5 / 15.0)] {
            let mut s = Synapses::new(r, integrator, 0.5e-3, 0, -0.07, vec![1.0], UNBOUNDED).unwrap();
            s.step(-0.07, &[true]).unwrap();
            assert!((s.traces()[0] - 1e3 / 15.0).abs() < 1e-13, "{integrator:?}: {:?}", s.traces());
            for k in 1..=40 {
                s.step(-0.07, &[false]).unwrap();
                let want = 1e3 / 15.0 * decay.powi(k);
                assert!((s.traces()[0] - want).abs() < 1e-13, "{integrator:?} step {k}");
            }
        }
        for (integrator, bound) in [(Integrator::Exponential, 5e-17), (Integrator::Euler, 4e-16)] {
            let rule = Rule { a_ltd: 0.0, ..r };
            let mut s = Synapses::new(rule, integrator, 0.5e-3, 0, -0.03, vec![0.0], UNBOUNDED).unwrap();
            s.step(-0.03, &[true]).unwrap();
            for _ in 0..4000 {
                s.step(-0.03, &[false]).unwrap();
            }
            let want = r.a_ltp * (-30.0 - r.theta_plus) * (-30.0 - r.theta_minus);
            assert!((s.weights()[0] - want).abs() < bound, "{integrator:?}: {}", s.weights()[0]);
        }
    }

    /// Hard bounds clip after the step's whole update, and a weight inside them moves freely.
    ///
    /// Clipping once is not clipping each term, nor gating each term on the weight the step starts
    /// from, as eqs. (1) and (2) print ("if `w_i > w_min`", "if `w_i < w_max`"). From `w_min` = 0 at
    /// −30 mV with presynaptic spikes in steps 0 and 1, step 1 depresses by `A_LTD (u − θ_−)` =
    /// 5.684 × 10⁻³ and potentiates by `A_LTP (u − θ_+)(u − θ_−)(1 − e^{−1/15})` = 3.205 × 10⁻³, and the
    /// weight stays at 0; clipping the depression alone, or skipping it at `w_min`, would leave
    /// 3.205 × 10⁻³. From `w_max` = 1 at 0 mV with spikes in steps 0 and 2, step 0 depresses to
    /// 1 − 14 × 10⁻⁵ × 70.6 = 0.990116, step 1 potentiates past 1, and in step 2 the potentiation
    /// outweighs the depression, so the weight stays at 1; skipping the potentiation because the step
    /// starts at `w_max` would leave 0.990116.
    #[test]
    fn hard_bounds_clip() {
        let r = Rule::VISUAL_CORTEX;
        let run = |w0: f64, bounds: (f64, f64), u: f64, spikes: usize| {
            let mut s = Synapses::new(r, Integrator::Exponential, 1e-3, 0, u, vec![w0], bounds).unwrap();
            for k in 0..spikes * 20 {
                s.step(u, &[k % 20 == 0]).unwrap();
            }
            s.weights()[0]
        };
        assert_eq!(run(0.5, (0.0, 0.5), -0.03, 5), 0.5, "LTP held at w_max");
        assert_eq!(run(0.0, (0.0, 0.5), -0.05, 5), 0.0, "LTD held at w_min");
        let free = run(0.25, (0.0, 0.5), -0.05, 5);
        assert!((free - (0.25 - 5.0 * r.a_ltd * 20.6)).abs() < 1e-16, "{free}");
        let up = run(0.25, (0.0, 0.5), -0.03, 5);
        assert!(up > 0.25 && up < 0.5);
        assert_eq!(run(0.5, (0.5, 0.5), -0.03, 2), 0.5);
        let trail = |w0: f64, u: f64, spikes: [usize; 2]| {
            let mut s = Synapses::new(r, Integrator::Exponential, 1e-3, 0, u, vec![w0], (0.0, 1.0)).unwrap();
            [0, 1, 2].map(|k| {
                s.step(u, &[spikes.contains(&k)]).unwrap();
                s.weights()[0]
            })
        };
        let (ltd, ltp) = (r.a_ltd * 40.6, r.a_ltp * 15.3 * 40.6 * -(-1.0 / 15.0_f64).exp_m1());
        assert!((ltd - 5.684e-3).abs() < 1e-17 && (ltp - 3.2049e-3).abs() < 1e-7 && ltp < ltd, "{ltd} {ltp}");
        assert_eq!(trail(0.0, -0.03, [0, 1])[1], 0.0);
        let top = trail(1.0, 0.0, [0, 2]);
        assert!((top[0] - 0.990116).abs() < 1e-15 && top[1..] == [1.0, 1.0], "{top:?}");
    }

    /// The rule reads the filters `delay` steps late: with a delay of 2, a spike in the third step
    /// still sees the starting filter, and one in the fourth sees the filter after the first step —
    /// `ū_−` for depression and `ū_+` for the potentiation gate alike.
    #[test]
    fn the_filters_are_read_delay_steps_late() {
        let r = Rule::VISUAL_CORTEX;
        let u0 = r.theta_minus * 1e-3;
        let lifted = -0.0606;
        let run = |delay: usize, spike_at: usize| {
            let mut s = Synapses::new(r, Integrator::Exponential, 1e-3, delay, u0, vec![0.0], UNBOUNDED).unwrap();
            let mut dw = 0.0;
            for k in 0..=spike_at {
                let before = s.weights()[0];
                s.step(lifted, &[k == spike_at]).unwrap();
                dw = s.weights()[0] - before;
            }
            dw
        };
        let after_one = lifted * 1e3 + (u0 * 1e3 - lifted * 1e3) * (-1.0 / 10.0_f64).exp();
        let after_two = lifted * 1e3 + (after_one - lifted * 1e3) * (-1.0 / 10.0_f64).exp();
        let ltd = |ubar: f64| -r.a_ltd * (ubar - r.theta_minus).max(0.0);
        assert_eq!(run(2, 0), 0.0);
        assert_eq!(run(2, 2), 0.0, "reads the state before step 0");
        assert!((run(2, 3) - ltd(after_one)).abs() < 1e-18, "{}", run(2, 3));
        assert!((run(2, 4) - ltd(after_two)).abs() < 1e-18);
        assert!((run(0, 2) - ltd(after_two)).abs() < 1e-18, "no delay reads the state at the step");
        assert!(run(0, 2) < run(2, 3) && run(2, 3) < 0.0);
        // The potentiation gate reads `ū_+` as late. One spike in step 0 at −30 mV with a delay of 2:
        // steps 1 and 2 still see the starting `ū_+`, and step 3 the `ū_+` after step 0, relaxed at
        // `τ_+`; from below `θ_−` the gate is shut, not negative.
        let gate_run = |start_mv: f64| {
            let rule = Rule { a_ltd: 0.0, ..r };
            let mut s = Synapses::new(rule, Integrator::Exponential, 1e-3, 2, start_mv * 1e-3, vec![0.0], UNBOUNDED).unwrap();
            let mut dw = [0.0; 4];
            for (k, d) in dw.iter_mut().enumerate() {
                let before = s.weights()[0];
                s.step(-0.03, &[k == 0]).unwrap();
                *d = s.weights()[0] - before;
            }
            dw
        };
        let window = -15.0 * (-1.0 / 15.0_f64).exp_m1();
        let ltp = |gate: f64, k: i32| r.a_ltp * (-30.0 - r.theta_plus) * (-f64::from(k - 1) / 15.0).exp() / 15.0 * gate * window;
        let start = r.theta_minus + 1.0;
        let relaxed = -30.0 + (start - -30.0) * (-1.0 / 7.0_f64).exp();
        let dw = gate_run(start);
        let want = [0.0, ltp(start - r.theta_minus, 1), ltp(start - r.theta_minus, 2), ltp(relaxed - r.theta_minus, 3)];
        assert_eq!(dw[0], 0.0, "the trace is still zero in the spike's own step");
        for k in 1..4 {
            assert!((dw[k] - want[k]).abs() < 1e-19, "{dw:?} against {want:?}");
        }
        assert_eq!(gate_run(-80.0), [0.0; 4], "ū_+ below θ_− shuts the gate");
    }

    /// Every refusal, rendered.
    #[test]
    fn every_refusal_names_what_it_refused() {
        let v = Rule::VISUAL_CORTEX;
        let bad_rules = [
            (Rule { theta_minus: f64::NAN, ..v }, "theta_minus = NaN is not finite"),
            (Rule { theta_plus: f64::INFINITY, ..v }, "theta_plus = inf is not finite"),
            (Rule { theta_minus: -40.0, ..v }, "theta_minus = -40 mV lies above theta_plus = -45.3 mV"),
            (Rule { a_ltd: -1e-5, ..v }, "a_ltd = -0.00001 must be finite and not negative"),
            (Rule { a_ltp: f64::INFINITY, ..v }, "a_ltp = inf must be finite and not negative"),
            (Rule { tau_x: 0.0, ..v }, "tau_x = 0 must be finite and positive"),
            (Rule { tau_x: f64::INFINITY, ..v }, "tau_x = inf must be finite and positive"),
            (Rule { tau_minus: Some(-1.0), ..v }, "tau_minus = -1 must be finite and positive"),
            (Rule { tau_plus: Some(f64::NAN), ..v }, "tau_plus = NaN must be finite and positive"),
        ];
        for (r, want) in bad_rules {
            assert_eq!(r.check().unwrap_err().to_string(), want);
            assert_eq!(r.clamp_per_spike(-0.05).unwrap_err().to_string(), want);
            assert_eq!(r.clamp_rate(-0.05, 1.0).unwrap_err().to_string(), want);
            assert_eq!(r.clamp_crossover().unwrap_err().to_string(), want);
            assert_eq!(r.clamp_minimum().unwrap_err().to_string(), want);
            assert_eq!(r.bcm_threshold(1e-4).unwrap_err().to_string(), want);
            assert_eq!(r.bcm_drift(1.0, 1.0, 1e-4, 1e-4).unwrap_err().to_string(), want);
            assert_eq!(r.a_ltd_homeostatic(1.0, 1.0).unwrap_err().to_string(), want);
            assert_eq!(Synapses::new(r, Integrator::Exponential, 1e-3, 0, -0.07, vec![], UNBOUNDED).unwrap_err().to_string(), want);
        }
        assert_eq!(Rule { theta_minus: -45.3, ..v }.check(), Ok(()), "θ_− = θ_+ is allowed");
        assert_eq!(Rule { a_ltd: 0.0, a_ltp: 0.0, ..v }.check(), Ok(()), "zero amplitudes are allowed");
        let e = |r: Result<f64, ClopathError>| r.unwrap_err().to_string();
        assert_eq!(e(v.clamp_per_spike(f64::NAN)), "u_clamp = NaN is not finite");
        assert_eq!(e(v.clamp_rate(-0.05, -1.0)), "nu_pre = -1 must be finite and not negative");
        let flat = Rule { a_ltp: 0.0, ..v };
        assert_eq!(e(flat.clamp_crossover()), "a_ltp = 0 must be finite and positive");
        assert_eq!(e(flat.clamp_minimum()), "a_ltp = 0 must be finite and positive");
        assert_eq!(e(flat.bcm_threshold(1e-4)), "a_ltp = 0 must be finite and positive");
        assert_eq!(e(v.bcm_threshold(0.0)), "beta = 0 must be finite and positive");
        assert_eq!(e(v.bcm_drift(-1.0, 1.0, 1e-4, 1e-4)), "nu_pre = -1 must be finite and not negative");
        assert_eq!(e(v.bcm_drift(1.0, f64::NAN, 1e-4, 1e-4)), "nu_post = NaN must be finite and not negative");
        assert_eq!(e(v.bcm_drift(1.0, 1.0, 0.0, 1e-4)), "alpha = 0 must be finite and positive");
        assert_eq!(e(v.bcm_drift(1.0, 1.0, 1e-4, -1e-4)), "beta = -0.0001 must be finite and positive");
        assert_eq!(e(v.a_ltd_homeostatic(-1.0, 1.0)), "ubarbar_sq = -1 must be finite and not negative");
        assert_eq!(e(v.a_ltd_homeostatic(1.0, 0.0)), "u_ref_sq = 0 must be finite and positive");
        let syn = |integrator, dt: f64, u0: f64, w: Vec<f64>, bounds| {
            Synapses::new(v, integrator, dt, 0, u0, w, bounds).unwrap_err().to_string()
        };
        assert_eq!(syn(Integrator::Exponential, 0.0, -0.07, vec![], UNBOUNDED), "dt = 0 must be finite and positive");
        assert_eq!(
            syn(Integrator::Euler, 0.008, -0.07, vec![], UNBOUNDED),
            "dt = 0.008 s exceeds tau_plus = 0.007 s, and the forward-Euler factor 1 - dt/tau would be negative"
        );
        assert_eq!(
            syn(Integrator::Euler, 0.012, -0.07, vec![], UNBOUNDED),
            "dt = 0.012 s exceeds tau_minus = 0.01 s, and the forward-Euler factor 1 - dt/tau would be negative"
        );
        assert_eq!(
            Synapses::new(Rule { tau_minus: Some(20.0), tau_plus: Some(20.0), ..v }, Integrator::Euler, 0.016, 0, -0.07, vec![], UNBOUNDED)
                .unwrap_err()
                .to_string(),
            "dt = 0.016 s exceeds tau_x = 0.015 s, and the forward-Euler factor 1 - dt/tau would be negative"
        );
        assert!(Synapses::new(v, Integrator::Euler, 0.007, 0, -0.07, vec![], UNBOUNDED).is_ok(), "dt = τ_+ is allowed");
        assert!(Synapses::new(v, Integrator::Exponential, 0.5, 0, -0.07, vec![], UNBOUNDED).is_ok(), "any step is exact");
        assert_eq!(syn(Integrator::Exponential, 1e-3, f64::NAN, vec![], UNBOUNDED), "u0 = NaN is not finite");
        assert_eq!(syn(Integrator::Exponential, 1e-3, -0.07, vec![], (1.0, 0.0)), "w_min = 1 and w_max = 0 do not bound an interval");
        assert_eq!(syn(Integrator::Exponential, 1e-3, -0.07, vec![], (f64::NAN, 1.0)), "w_min = NaN and w_max = 1 do not bound an interval");
        assert_eq!(syn(Integrator::Exponential, 1e-3, -0.07, vec![0.5, f64::INFINITY], UNBOUNDED), "weight = inf is not finite");
        assert_eq!(syn(Integrator::Exponential, 1e-3, -0.07, vec![0.5, 1.5], (0.0, 1.0)), "weight 1 = 1.5 lies outside [0, 1]");
        assert_eq!(syn(Integrator::Exponential, 1e-3, -0.07, vec![-0.5], (0.0, 1.0)), "weight 0 = -0.5 lies outside [0, 1]");
        assert!(Synapses::new(v, Integrator::Exponential, 1e-3, 0, -0.07, vec![0.0, 1.0], (0.0, 1.0)).is_ok());
        let mut s = Synapses::new(v, Integrator::Exponential, 1e-3, 0, -0.07, vec![0.5, 0.5], UNBOUNDED).unwrap();
        assert_eq!(s.step(f64::INFINITY, &[false, false]).unwrap_err().to_string(), "u = inf is not finite");
        assert_eq!(s.step(-0.07, &[false]).unwrap_err().to_string(), "1 presynaptic flags for 2 synapses");
        assert_eq!(s.set_ltd_scale(-1.0).unwrap_err().to_string(), "ltd_scale = -1 must be finite and not negative");
        assert_eq!(s.set_ltd_scale(f64::NAN).unwrap_err().to_string(), "ltd_scale = NaN must be finite and not negative");
        assert_eq!(Homeostat::new(0.0, -0.07, 1e-3).unwrap_err().to_string(), "tau = 0 must be finite and positive");
        assert_eq!(Homeostat::new(1.0, -0.07, -1e-3).unwrap_err().to_string(), "dt = -0.001 must be finite and positive");
        assert_eq!(Homeostat::new(1.0, f64::NAN, 1e-3).unwrap_err().to_string(), "rest = NaN is not finite");
        let mut h = Homeostat::new(1.0, -0.07, 1e-3).unwrap();
        assert_eq!(h.step(f64::NAN).unwrap_err().to_string(), "u = NaN is not finite");
        let m = Membrane::AEIF_M;
        let neuron = |m: Membrane| Neuron::new(m).unwrap_err().to_string();
        assert_eq!(neuron(Membrane { c: 0.0, ..m }), "c = 0 must be finite and positive");
        assert_eq!(neuron(Membrane { g_l: -1.0, ..m }), "g_l = -1 must be finite and positive");
        assert_eq!(neuron(Membrane { delta_t: f64::NAN, ..m }), "delta_t = NaN must be finite and positive");
        assert_eq!(neuron(Membrane { tau_w: 0.0, ..m }), "tau_w = 0 must be finite and positive");
        assert_eq!(
            neuron(Membrane { tau_z: 0.5, ..m }),
            "dt = 0.001 s exceeds tau_z = 0.0005 s, and the forward-Euler factor 1 - dt/tau would be negative"
        );
        assert_eq!(
            neuron(Membrane { tau_v_t: 0.25, ..m }),
            "dt = 0.001 s exceeds tau_v_t = 0.00025 s, and the forward-Euler factor 1 - dt/tau would be negative"
        );
        assert_eq!(
            neuron(Membrane { tau_w: 0.75, ..m }),
            "dt = 0.001 s exceeds tau_w = 0.00075 s, and the forward-Euler factor 1 - dt/tau would be negative"
        );
        assert!(Neuron::new(Membrane { tau_w: 1.0, tau_z: 1.0, tau_v_t: 1.0, ..m }).is_ok(), "τ = the step is allowed");
        for (bad, what) in [
            (Membrane { e_l: f64::NAN, ..m }, "e_l"),
            (Membrane { v_t_rest: f64::INFINITY, ..m }, "v_t_rest"),
            (Membrane { a: f64::NAN, ..m }, "a"),
            (Membrane { b: f64::NAN, ..m }, "b"),
            (Membrane { i_sp: f64::NAN, ..m }, "i_sp"),
            (Membrane { v_t_max: f64::NAN, ..m }, "v_t_max"),
            (Membrane { v_detect: Some(f64::NAN), ..m }, "v_detect"),
            (Membrane { peak: Some([f64::NAN, 0.0]), ..m }, "peak[0]"),
            (Membrane { peak: Some([0.0, f64::NAN]), ..m }, "peak[1]"),
            (Membrane { v_reset: Some(f64::NAN), ..m }, "v_reset"),
        ] {
            let got = neuron(bad);
            assert!(got == format!("{what} = NaN is not finite") || got == format!("{what} = inf is not finite"), "{got}");
        }
        let mut n = Neuron::new(m).unwrap();
        assert_eq!(n.step(f64::NAN).unwrap_err().to_string(), "i = NaN is not finite");
        assert_eq!(
            votri_pairing(Rule::HIPPOCAMPAL, m, 3, 20, 10).unwrap_err().to_string(),
            "tau_minus is blank in Table 1b for this set; the paper prints no value"
        );
        assert_eq!(votri_pairing(v, Membrane::TABLE_1A, 3, 20, 10).unwrap_err(), ClopathError::Unprinted { what: "v_detect" });
        assert_eq!(votri_pairing(v, m, 3, 20, 20).unwrap_err(), ClopathError::Protocol { period: 20, lag: 20 });
    }
}
