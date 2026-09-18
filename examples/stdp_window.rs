//! **A VERIFICATION GATE.** Exits non-zero when its check fails.
//!
//! Spike-timing-dependent plasticity in one picture: a synapse is strengthened when the presynaptic
//! spike arrives *before* the postsynaptic one and weakened when it arrives after, with the size of
//! the change falling off exponentially in the gap between them.
//!
//! Bi & Poo, *Synaptic modifications in cultured hippocampal neurons: dependence on spike timing,
//! synaptic strength, and postsynaptic cell type*, J. Neurosci. 18(24):10464–10472, 1998, for the
//! mechanism; Song, Miller & Abbott, *Competitive Hebbian learning through spike-timing-dependent
//! synaptic plasticity*, Nat. Neurosci. 3:919–926, 2000, for the parameter set used here.
//!
//! The Bi & Poo constructor in this crate deliberately takes its amplitudes from the caller, because
//! that paper reports a percentage change in EPSC amplitude with large scatter and a single number
//! presented as "Bi and Poo's A₊" would be indefensible. Song, Miller and Abbott do give a fully
//! specified set, so that is the one a gate can check.
//!
//! ```sh
//! cargo run --release --example stdp_window
//! ```
//!
//! # Why this is the sharpest check in the module
//!
//! The rule has a closed form for a single pair of spikes:
//!
//! ```text
//! Δw = +A₊ · exp(−Δt/τ₊)   for Δt > 0   (pre before post — potentiation)
//! Δw = −A₋ · exp(+Δt/τ₋)   for Δt < 0   (post before pre — depression)
//! ```
//!
//! So the simulated weight change can be compared against an expression written from the paper
//! rather than against a previous run. Anything that drifts — a trace that decays at the wrong rate,
//! an amplitude applied on the wrong side, a sign flip — moves the number immediately.
//!
//! The gate also checks the two properties the window has to have beyond its own arithmetic: it is
//! **causal** (the sign flips at Δt = 0) and it is **depression-dominated** — the area below the
//! axis exceeds the area above it, which is what stops a network driving every weight to its
//! ceiling.
//!
//! ⚠ A sign hazard worth knowing before transcribing anything: Song, Miller and Abbott write their
//! window with `Δt = t_pre − t_post`, the **opposite** of this crate's `lag = t_post − t_pre`.
//! The model is the same; the variable is not. Flip the axis before comparing to their figure.

use ferromorphic::plasticity::PairStdp;

/// Relative tolerance on a single pair's weight change.
const TOL: f64 = 1e-12;

fn main() {
    // A fully specified published set: tau± = 20 ms, A₊ = 0.005·g_max, A₋/A₊ = 1.05. That 5%
    // asymmetry is the entire stabilising mechanism in the paper, and it is why the net area below
    // is negative even though the two time constants are equal.
    const G_MAX: f64 = 1.0;
    let stdp = match PairStdp::song_abbott_2000(G_MAX) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("  FAIL: could not build the Song-Abbott window: {e}");
            std::process::exit(1);
        }
    };

    println!();
    println!("  SPIKE-TIMING-DEPENDENT PLASTICITY AGAINST ITS CLOSED FORM");
    println!("  Song, Miller & Abbott 2000 parameters, g_max = {G_MAX}");
    println!("  A+ {:.4}  tau+ {:.1} ms   A- {:.4}  tau- {:.1} ms",
             stdp.a_plus, stdp.tau_plus * 1e3, stdp.a_minus, stdp.tau_minus * 1e3);
    println!();
    println!("  {:>10} {:>16} {:>16} {:>12}", "dt (ms)", "closed form", "window()", "relative");
    println!("  {:->10} {:->16} {:->16} {:->12}", "", "", "", "");

    let mut worst: f64 = 0.0;
    let mut checked = 0u32;

    // ⛔ THE CONSTANTS BELOW ARE TRANSCRIBED FROM THE PAPER, NOT READ OFF THE STRUCT.
    //
    // The first version of this gate computed its "closed form" from `stdp.a_plus`, `stdp.tau_plus`
    // and friends — the same fields `window()` reads — so it compared the code against itself and
    // agreed to the last bit at every lag. A relative error of EXACTLY 0.00e0 across twelve values
    // is not a strong result, it is the tell.
    //
    // Song, Miller & Abbott, Nat. Neurosci. 3:919-926, 2000: tau_plus = tau_minus = 20 ms,
    // A_plus = 0.005 * g_max, A_minus / A_plus = 1.05. With those written out here, a constructor
    // that got any of them wrong fails this gate instead of passing it.
    const PAPER_TAU: f64 = 20e-3;
    const PAPER_A_PLUS: f64 = 0.005 * G_MAX;
    const PAPER_A_MINUS: f64 = 1.05 * PAPER_A_PLUS;

    // Both signs, spanning well inside and well outside the time constants.
    for &ms in &[-80.0, -40.0, -20.0, -10.0, -5.0, -1.0, 1.0, 5.0, 10.0, 20.0, 40.0, 80.0] {
        let dt = ms * 1e-3;
        let want = if dt > 0.0 {
            PAPER_A_PLUS * (-dt / PAPER_TAU).exp()
        } else {
            -PAPER_A_MINUS * (dt / PAPER_TAU).exp()
        };
        let got = stdp.window(dt);
        let rel = if want == 0.0 { (got - want).abs() } else { (got - want).abs() / want.abs() };
        worst = worst.max(rel);
        checked += 1;
        println!("  {ms:>10.1} {want:>16.9} {got:>16.9} {rel:>12.2e}");
    }

    println!();

    // Causality: the sign must flip exactly at zero, and zero lag must be exactly zero rather than
    // "very small". A rule that potentiates on a simultaneous pair has lost the thing it models.
    let at_zero = stdp.window(0.0);
    if at_zero != 0.0 {
        eprintln!("  FAIL: a simultaneous pair changed the weight by {at_zero}, not 0");
        std::process::exit(1);
    }
    let just_after = stdp.window(1e-9);
    let just_before = stdp.window(-1e-9);
    if !(just_after > 0.0 && just_before < 0.0) {
        eprintln!("  FAIL: the window is not causal — {just_before} before, {just_after} after");
        std::process::exit(1);
    }
    println!("  causal: {just_before:+.3e} just before, exactly {at_zero} at zero, {just_after:+.3e} just after");

    // Depression dominance over a wide window. `total_window_area` has its own closed form
    // (A+·tau+ - A-·tau-), so this is checked against arithmetic as well as reported.
    let area = stdp.total_window_area();
    let area_closed = PAPER_A_PLUS * PAPER_TAU - PAPER_A_MINUS * PAPER_TAU;
    let area_rel = (area - area_closed).abs() / area_closed.abs().max(f64::MIN_POSITIVE);
    if area_rel > 1e-12 {
        eprintln!("  FAIL: total area {area:.6e} disagrees with A+tau+ - A-tau- = {area_closed:.6e}");
        std::process::exit(1);
    }
    println!("  net area {area:+.4e} = A+tau+ - A-tau-, depression-dominated: {}",
             stdp.is_depression_dominated());
    println!();
    println!("  dw = +A+ exp(-dt/tau+) for dt > 0, -A- exp(+dt/tau-) for dt < 0.");
    println!("  Expected values transcribed from Song, Miller & Abbott 2000, not read off the struct.");
    println!();

    if worst > TOL {
        eprintln!("  FAIL: worst relative error {worst:.2e} exceeds {TOL:.0e}");
        std::process::exit(1);
    }
    println!("  PASS  {checked} lags, worst relative error {worst:.2e}, tolerance {TOL:.0e}");
    println!();
}
