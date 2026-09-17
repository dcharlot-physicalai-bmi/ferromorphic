//! **A VERIFICATION GATE.** Exits non-zero when its check fails.
//!
//! Runs the leaky integrate-and-fire model against the analytic inter-spike interval it is supposed
//! to reproduce, across a range of input currents, and prints the comparison. This is the claim the
//! rest of the crate rests on: if the neuron does not reproduce its own closed form, no spike train
//! produced here means anything.
//!
//! ```sh
//! cargo run --release --example lif_closed_form
//! ```
//!
//! The tolerance is 0.1% relative on the interval. That is not a tuned number: exponential Euler is
//! EXACT for constant input, so the only error is tick quantisation of the threshold crossing, which
//! at `dt = 1 µs` against intervals of 3–20 ms is a few parts in 10,000.

use ferromorphic::neuron::{Lif, Neuron};

/// Relative tolerance on the inter-spike interval.
const TOL: f64 = 1e-3;

fn main() {
    let proto = Lif::default();
    let dt = 1e-6;
    let seconds = 2.0;
    let steps = (seconds / dt) as u64;

    println!();
    println!("  LIF AGAINST ITS CLOSED FORM");
    println!("  tau_m {:.0} ms, v_rest {:.0} mV, v_th {:.0} mV, t_ref {:.0} ms, R {:.0} MOhm",
             proto.tau_m * 1e3, proto.v_rest * 1e3, proto.v_th * 1e3,
             proto.t_ref * 1e3, proto.r_m / 1e6);
    println!("  {seconds:.0} s per current at dt = {:.0} us", dt * 1e6);
    println!();
    println!("  {:>8} {:>10} {:>12} {:>12} {:>10} {:>8}",
             "I (nA)", "v_inf (mV)", "closed (ms)", "simulated", "rel", "spikes");
    println!("  {:->8} {:->10} {:->12} {:->12} {:->10} {:->8}", "", "", "", "", "", "");

    let mut worst: f64 = 0.0;
    let mut checked = 0u32;

    for &na in &[1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0] {
        let i = na * 1e-9;
        let v_inf = proto.v_inf(i);

        let Some(want) = proto.isi(i) else {
            // Sub-threshold. The model says it never fires; check that it does not, which is a
            // different and equally load-bearing claim.
            let mut n = proto;
            let mut fired = 0u32;
            for _ in 0..steps {
                if n.step(dt, i) {
                    fired += 1;
                }
            }
            println!("  {na:>8.1} {:>10.1} {:>12} {:>12} {:>10} {fired:>8}",
                     v_inf * 1e3, "never", "never", "-");
            if fired != 0 {
                eprintln!("\n  FAIL: a sub-threshold neuron fired {fired} times at {na} nA");
                std::process::exit(1);
            }
            continue;
        };

        let mut n = proto;
        let mut spikes = 0u32;
        let mut first = None;
        let mut last = 0.0;
        for k in 0..steps {
            if n.step(dt, i) {
                let t = k as f64 * dt;
                if first.is_none() {
                    first = Some(t);
                }
                last = t;
                spikes += 1;
            }
        }
        if spikes < 3 {
            eprintln!("\n  FAIL: {na} nA produced only {spikes} spikes in {seconds} s");
            std::process::exit(1);
        }

        // Measured from the FIRST spike. The interval before it starts from v_rest rather than from
        // v_reset and is a different quantity; including it would compare the wrong thing to the
        // closed form — and with the default's v_rest == v_reset it would pass anyway, which is how
        // a check like this comes to be both wrong and green.
        let got = (last - first.unwrap()) / f64::from(spikes - 1);
        let rel = (got - want).abs() / want;
        worst = worst.max(rel);
        checked += 1;

        println!("  {na:>8.1} {:>10.1} {:>12.4} {:>12.4} {rel:>10.2e} {spikes:>8}",
                 v_inf * 1e3, want * 1e3, got * 1e3);
    }

    println!();
    println!("  T = tau_m * ln((v_inf - v_reset) / (v_inf - v_th)), ISI = T + t_ref");
    println!("  exponential Euler is EXACT for constant input, so the residual here is tick");
    println!("  quantisation of the threshold crossing and nothing else.");
    println!();

    if worst > TOL {
        eprintln!("  FAIL: worst relative error {worst:.2e} exceeds {TOL:.0e}");
        std::process::exit(1);
    }
    println!("  PASS  {checked} currents, worst relative error {worst:.2e}, tolerance {TOL:.0e}");
    println!();
}
