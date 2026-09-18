//! **A VERIFICATION GATE.** Exits non-zero when its check fails.
//!
//! The case for neuromorphic hardware in one number: a spiking network does almost nothing almost
//! all of the time, so a machine that only pays for the "something" wins. That is an empirical claim
//! about a *workload*, and it is usually asserted rather than measured.
//!
//! ```sh
//! cargo run --release --example event_driven_saves
//! ```
//!
//! # What this gate establishes, in order
//!
//! 1. **The two modes agree.** [`Mode::Clocked`] updates every neuron on every tick;
//!    [`Mode::EventDriven`] updates one only when a spike reaches it and jumps it across the quiet
//!    interval. They must produce the **identical spike train**, compared spike for spike. A
//!    shortcut that changes the answer is not a shortcut, and if this check fails nothing below it
//!    means anything.
//! 2. **The saving is a count, not an adjective.** With the trains proved identical, the difference
//!    between the two modes is [`Ledger::idle_fraction`] — the share of a clocked run's membrane
//!    updates that land on a neuron nothing reached.
//! 3. **It is a property of the workload.** The same network under a dense drive and a sparse one
//!    gives very different answers, so the gate runs both and prints them side by side. Quoting one
//!    without the other is how "1000× more efficient" claims get made.
//!
//! # The safety property underneath
//!
//! Jumping a neuron across quiet ticks is only legal for a model whose state composes across an
//! interval — [`Neuron::EXACT_OVER_GAPS`]. `Sim::new` refuses `Mode::EventDriven` for a model that
//! lacks it rather than producing spike times that depend on which ticks happened to be quiet. The
//! gate exercises that refusal too, because a safety check nobody triggers is a safety check nobody
//! has tested.

use ferromorphic::ledger::Ledger;
use ferromorphic::net::{Net, NetBuilder};
use ferromorphic::neuron::{Izhikevich, Lif, Neuron};
use ferromorphic::sim::{Mode, Sim, SimError};

/// Neurons in the chain.
const N: usize = 8;
/// Ticks to run: 4,000 at 0.1 ms is 400 ms.
const TICKS: u64 = 4_000;
/// Tick length, seconds.
const DT: f64 = 1e-4;

fn chain() -> Net {
    let mut b = NetBuilder::new(N);
    for i in 0..N - 1 {
        b.connect(i as u32, i as u32 + 1, 22e-3, 30).expect("in range and finite");
    }
    b.build()
}

/// Run both modes and return `(clocked ledger, event-driven ledger, spikes)`, or exit if they
/// disagree — which is the one outcome that invalidates everything downstream.
fn both_modes(ext: &[f64], label: &str) -> (Ledger, Ledger, usize) {
    let proto = Lif::default();
    let mut a = Sim::new(chain(), vec![proto; N], DT, Mode::Clocked).expect("counts match");
    let mut b = Sim::new(chain(), vec![proto; N], DT, Mode::EventDriven).expect("Lif is exact over gaps");
    let ta = a.run(TICKS, ext);
    let tb = b.run(TICKS, ext);

    if ta.spikes() != tb.spikes() {
        eprintln!(
            "\n  FAIL [{label}]: the two modes disagree — {} spikes clocked, {} event-driven",
            ta.len(),
            tb.len()
        );
        std::process::exit(1);
    }
    (a.ledger, b.ledger, ta.len())
}

fn main() {
    println!();
    println!("  EVENT-DRIVEN SIMULATION: THE SAVING, AS A COUNT");
    println!("  {N} leaky integrate-and-fire cells in a chain, {} ms at {} ms per tick",
             TICKS as f64 * DT * 1e3, DT * 1e3);
    println!();

    // Two workloads on the SAME network. Sparse: a 3 ms pulse every 50 ms, so a wave runs down the
    // chain and the chain falls silent. Dense: the same current, never switched off.
    let mut dense = vec![0.0; N];
    dense[0] = 25e-9;

    println!("  {:<10} {:>8} {:>16} {:>16} {:>10}", "drive", "spikes", "clocked updates", "event-driven", "idle");
    println!("  {:-<10} {:->8} {:->16} {:->16} {:->10}", "", "", "", "", "");

    let mut rows = Vec::new();
    for (label, sparse) in [("sparse", true), ("dense", false)] {
        // The sparse arm gates the current on the tick, which `Sim::run` cannot express, so it is
        // stepped by hand. Same network, same neurons, same ledger.
        let proto = Lif::default();
        let (led_a, led_b, spikes) = if sparse {
            let mut a = Sim::new(chain(), vec![proto; N], DT, Mode::Clocked).expect("counts");
            let mut b = Sim::new(chain(), vec![proto; N], DT, Mode::EventDriven).expect("exact");
            let mut ext = vec![0.0; N];
            let (mut sa, mut sb) = (Vec::new(), Vec::new());
            for t in 0..TICKS {
                let ms = t as f64 * DT * 1e3;
                ext[0] = if ms % 50.0 < 3.0 { 25e-9 } else { 0.0 };
                for s in a.step(&ext) { sa.push((t, s)); }
                for s in b.step(&ext) { sb.push((t, s)); }
            }
            if sa != sb {
                eprintln!("\n  FAIL [sparse]: the two modes disagree — {} vs {} spikes", sa.len(), sb.len());
                std::process::exit(1);
            }
            (a.ledger, b.ledger, sa.len())
        } else {
            both_modes(&dense, "dense")
        };

        let idle = led_a.idle_fraction().expect("the clocked run updated something");
        if led_b.neuron_updates_idle != 0 {
            eprintln!("\n  FAIL [{label}]: the event-driven run updated {} idle neurons",
                      led_b.neuron_updates_idle);
            std::process::exit(1);
        }
        println!("  {label:<10} {spikes:>8} {:>16} {:>16} {:>9.1}%",
                 led_a.neuron_updates(), led_b.neuron_updates(), idle * 100.0);
        rows.push((label, idle, led_a.neuron_updates(), led_b.neuron_updates()));
    }

    println!();

    // The workload-dependence claim, asserted rather than said: the sparse drive must leave a
    // strictly larger share of the clocked run idle than the dense one. If it does not, the two
    // arms are not exercising different regimes and the table above is one number printed twice.
    let sparse_idle = rows[0].1;
    let dense_idle = rows[1].1;
    if sparse_idle <= dense_idle {
        eprintln!("  FAIL: the sparse drive ({:.1}%) is not idler than the dense one ({:.1}%) — the \
                   two arms are not in different regimes", sparse_idle * 100.0, dense_idle * 100.0);
        std::process::exit(1);
    }
    println!("  The saving is a property of the WORKLOAD: {:.1}% idle under a sparse drive against",
             sparse_idle * 100.0);
    println!("  {:.1}% under a dense one, on the same network with the same neurons. A figure quoted",
             dense_idle * 100.0);
    println!("  without its workload is not a figure about a chip.");
    println!();

    // The safety property. `Izhikevich` is quadratic under forward Euler, so one step of k·dt is not
    // k steps of dt and jumping a gap would change its spike times. `Sim::new` must refuse.
    let cells = vec![Izhikevich::regular_spiking(); N];
    if Izhikevich::EXACT_OVER_GAPS {
        eprintln!("  FAIL: Izhikevich claims to be exact over gaps; it is quadratic under forward Euler");
        std::process::exit(1);
    }
    match Sim::new(chain(), cells.clone(), DT, Mode::EventDriven) {
        Err(SimError::NotExactOverGaps) => {
            println!("  Refused, correctly: an event-driven Izhikevich is SimError::NotExactOverGaps");
        }
        Err(e) => {
            eprintln!("  FAIL: refused for the wrong reason: {e}");
            std::process::exit(1);
        }
        Ok(_) => {
            eprintln!("  FAIL: built an event-driven simulation of a model that cannot be jumped");
            std::process::exit(1);
        }
    }
    if Sim::new(chain(), cells, DT, Mode::Clocked).is_err() {
        eprintln!("  FAIL: the clocked mode must still accept it");
        std::process::exit(1);
    }

    println!();
    println!("  PASS  both modes agree spike for spike, the event-driven run touches no idle");
    println!("        neuron, the saving differs by workload, and the unsafe mode is refused.");
    println!();
}
