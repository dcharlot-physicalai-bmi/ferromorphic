//! Analytic, independent-numerical and convergence gates for the models this workstream re-checked.
//!
//! These are integration tests: they call the public API and they also carry a broken method
//! beside the real one, so a tolerance or an order window that can no longer fail is itself a
//! failure. Tolerances below are the ones the assertions use. Measured orders and errors are in
//! the assertion text when a bound is missed.

use ferromorphic::encode::RateEncoder;
use ferromorphic::ledger::{Evidence, LOIHI_2018, TRUENORTH_2014};
use ferromorphic::net::NetBuilder;
use ferromorphic::neuron::{IntegrateAndFire, Izhikevich, Lif, Neuron};
use ferromorphic::nir::{CubaState, Graph, Input, Lif as NirLif, Node, Output};
use ferromorphic::rng::Rng;
use ferromorphic::sim::{Mode, Sim, SimError};

const CLOSED_FORM_TOL: f64 = 1e-12;
const ORDER_MIN: f64 = 0.7;
const ORDER_MAX: f64 = 1.6;

fn closed_lif(v0: f64, v_rest: f64, tau: f64, t: f64) -> f64 {
    v_rest + (v0 - v_rest) * (-t / tau).exp()
}

fn forward_euler_lif(mut v: f64, v_rest: f64, tau: f64, dt: f64, steps: u32) -> f64 {
    for _ in 0..steps {
        v += dt * (-(v - v_rest) / tau);
    }
    v
}

fn orders(errors: &[f64]) -> Vec<f64> {
    errors.windows(2).map(|w| (w[0] / w[1]).log2()).collect()
}

fn order_accepted(order: f64) -> bool {
    (ORDER_MIN..=ORDER_MAX).contains(&order)
}

/// Free decay of [`Lif`] against the closed form, and a forward-Euler membrane that must miss it.
#[test]
fn lif_free_decay_matches_an_independent_closed_form_and_forward_euler_does_not() {
    // Subthreshold and not equal to rest, so the decay is observable and the cell does not fire.
    let cell = Lif {
        v: -60e-3,
        ..Lif::default()
    };
    assert!(cell.v < cell.v_th && cell.v != cell.v_rest);
    let (v0, v_rest, tau) = (cell.v, cell.v_rest, cell.tau_m);
    let t_end = 50e-3;
    let want = closed_lif(v0, v_rest, tau, t_end);

    for dt in [1e-2, 1e-5] {
        let mut stepped = cell;
        let steps = (t_end / dt).round() as u32;
        let mut spiked = false;
        for _ in 0..steps {
            spiked |= stepped.step(dt, 0.0);
        }
        assert!(!spiked, "a subthreshold decay fired at dt={dt}");
        let err = (stepped.v - want).abs();
        assert!(
            err < CLOSED_FORM_TOL,
            "Lif at dt={dt} differs from the closed form by {err}"
        );
    }

    let dt = 1e-3;
    let steps = (t_end / dt).round() as u32;
    let euler = forward_euler_lif(v0, v_rest, tau, dt, steps);
    let euler_err = (euler - want).abs();
    assert!(
        euler_err > CLOSED_FORM_TOL,
        "forward Euler error {euler_err} met {CLOSED_FORM_TOL}; widening the tolerance would hide it"
    );

    let mut euler_errors = Vec::new();
    for dt in [2e-3, 1e-3, 5e-4] {
        let steps = (t_end / dt).round() as u32;
        euler_errors.push((forward_euler_lif(v0, v_rest, tau, dt, steps) - want).abs());
    }
    let euler_orders = orders(&euler_errors);
    eprintln!("MEASURED lif_forward_euler_errors={euler_errors:?} orders={euler_orders:?}");
    for order in euler_orders {
        assert!(
            order_accepted(order),
            "forward Euler order {order} from errors {euler_errors:?} is outside [{ORDER_MIN}, {ORDER_MAX}]"
        );
    }

    // Same errors repeated: a nonconvergent residual. The order gate must reject it.
    let flat = vec![euler_errors[0]; 3];
    for order in orders(&flat) {
        assert!(
            !order_accepted(order),
            "a flat error sequence produced order {order}, and the gate accepted it"
        );
    }
}

/// The perfect integrator spikes on the tick the linear ramp predicts, and a wrong capacitance does not.
#[test]
fn integrate_and_fire_spikes_on_the_predicted_tick_and_a_wrong_capacitance_does_not() {
    let cell = IntegrateAndFire::default();
    let i = 1e-9;
    let dt = 1e-4;
    // Independent of `IntegrateAndFire::isi`: c * dV/dt = I, so T = c * (v_th - v) / I.
    let t_cross = cell.c * (cell.v_th - cell.v) / i;
    assert!(
        (t_cross - 15e-3).abs() < 1e-15,
        "the default ramp is 15 ms, got {t_cross}"
    );

    let mut stepped = cell;
    let mut ticks = 0u32;
    loop {
        ticks += 1;
        if stepped.step(dt, i) {
            break;
        }
        assert!(ticks < 10_000, "the ramp never reached threshold");
    }
    let simulated = f64::from(ticks) * dt;
    assert!(
        simulated + 1e-12 >= t_cross,
        "spike at {simulated} s is early of {t_cross}"
    );
    assert!(
        simulated - t_cross < 2.0 * dt,
        "spike at {simulated} s is more than one tick after {t_cross}"
    );

    // 1 pF instead of the 1 nF the ramp used. The crossing moves by the same factor.
    let mut v = 0.0;
    let mut broken_ticks = 0u32;
    let wrong_c = 1e-12;
    loop {
        broken_ticks += 1;
        v += i * dt / wrong_c;
        if v >= cell.v_th {
            break;
        }
        assert!(broken_ticks < 10_000, "the wrong capacitance never fired");
    }
    let broken_t = f64::from(broken_ticks) * dt;
    assert!(
        (broken_t - t_cross).abs() > dt,
        "a 1000x capacitance error still landed on the predicted tick ({broken_t} vs {t_cross})"
    );
}

/// Izhikevich's forward-Euler scheme against an independent classical RK4 of the paper's ODE.
///
/// The crate's `substeps = 1` update is semi-implicit: `v` advances with the old `u`, then `u`
/// advances with the new `v`. RK4 advances the pair together from the continuous vector field
/// `v' = 0.04 v^2 + 5 v + 140 - u + I`, `u' = a (b v - u)`, in the paper's milliseconds. Neither
/// update is copied from [`Izhikevich::step`].
#[test]
fn izhikevich_converges_to_an_independent_rk4_and_a_dropped_quadratic_does_not() {
    let (a, b, i_paper) = (0.02_f64, 0.2_f64, 0.0_f64);
    // Hand evaluation at the paper's start, not a replay of the update loop.
    // 0.04*4225 + 5*(-65) + 140 - (-13) = 169 - 325 + 140 + 13 = -3.
    let v0 = -65.0_f64;
    let u0 = -13.0_f64;
    let slope = 0.04 * v0 * v0 + 5.0 * v0 + 140.0 - u0 + i_paper;
    assert!(
        (slope + 3.0).abs() < 1e-12,
        "analytic slope is -3 mV/ms, got {slope}"
    );

    let mut one = Izhikevich {
        substeps: 1,
        ..Izhikevich::regular_spiking()
    };
    assert!(!one.step(0.5e-3, 0.0));
    assert!(
        (one.v - (v0 + 0.5 * slope)).abs() < 1e-12,
        "one 0.5 ms step landed at {}",
        one.v
    );

    let t_ms = 10.0;
    let reference = rk4_final(v0, u0, a, b, i_paper, 0.01, t_ms);
    assert!(
        reference.0 < -65.0 && reference.0 > -72.0,
        "RK4 left the subthreshold well: {reference:?}"
    );

    let mut errors = Vec::new();
    for h_ms in [0.5, 0.25, 0.125, 0.0625] {
        let (v, _) = crate_final(h_ms, t_ms);
        errors.push((v - reference.0).abs());
    }
    assert!(
        errors.iter().all(|e| e.is_finite() && *e > 0.0),
        "errors {errors:?}"
    );
    let measured = orders(&errors);
    eprintln!(
        "MEASURED izhikevich_errors_mV={errors:?} orders={measured:?} rk4_v={}",
        reference.0
    );
    for order in measured {
        assert!(
            order_accepted(order),
            "Izhikevich order {order} from errors {errors:?} mV is outside [{ORDER_MIN}, {ORDER_MAX}]"
        );
    }
    let fine = *errors.last().unwrap();
    let coarse = errors[0];
    assert!(
        coarse > fine * 4.0,
        "refining the step did not shrink the error: {errors:?}"
    );

    let broken = broken_quadratic_final(v0, u0, a, b, 0.0625, t_ms);
    let broken_err = (broken - reference.0).abs();
    assert!(
        !broken.is_finite() || broken_err > fine * 10.0,
        "dropping 0.04 v^2 left the trajectory {broken_err} mV from RK4; the crate's fine error is {fine}"
    );

    // The safety property, at the construction site.
    let net = NetBuilder::new(1).build();
    let refused = Sim::new(
        net,
        vec![Izhikevich::regular_spiking()],
        1e-4,
        Mode::EventDriven,
    );
    assert!(matches!(refused, Err(SimError::NotExactOverGaps)));
}

fn field(v: f64, u: f64, a: f64, b: f64, i_paper: f64) -> (f64, f64) {
    (
        0.04 * v * v + 5.0 * v + 140.0 - u + i_paper,
        a * (b * v - u),
    )
}

fn rk4_final(v0: f64, u0: f64, a: f64, b: f64, i_paper: f64, h: f64, t_ms: f64) -> (f64, f64) {
    let steps = (t_ms / h).round() as u32;
    let (mut v, mut u) = (v0, u0);
    for _ in 0..steps {
        let (k1v, k1u) = field(v, u, a, b, i_paper);
        let (k2v, k2u) = field(v + 0.5 * h * k1v, u + 0.5 * h * k1u, a, b, i_paper);
        let (k3v, k3u) = field(v + 0.5 * h * k2v, u + 0.5 * h * k2u, a, b, i_paper);
        let (k4v, k4u) = field(v + h * k3v, u + h * k3u, a, b, i_paper);
        v += (h / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v);
        u += (h / 6.0) * (k1u + 2.0 * k2u + 2.0 * k3u + k4u);
        assert!(
            v < 30.0,
            "the reference crossed the cutoff; the comparison would be a reset, not the flow"
        );
    }
    (v, u)
}

fn crate_final(h_ms: f64, t_ms: f64) -> (f64, f64) {
    let mut cell = Izhikevich {
        substeps: 1,
        ..Izhikevich::regular_spiking()
    };
    let steps = (t_ms / h_ms).round() as u32;
    let dt = h_ms * 1e-3;
    for _ in 0..steps {
        assert!(!cell.step(dt, 0.0), "subthreshold relaxation spiked");
    }
    (cell.v, cell.u)
}

fn broken_quadratic_final(v0: f64, u0: f64, a: f64, b: f64, h: f64, t_ms: f64) -> f64 {
    let steps = (t_ms / h).round() as u32;
    let (mut v, mut u) = (v0, u0);
    for _ in 0..steps {
        // The paper's linear terms only. The 0.04 v^2 term is the quadratic this model is.
        let dv = 5.0 * v + 140.0 - u;
        v += h * dv;
        u += h * a * (b * v - u);
        if !v.is_finite() {
            return v;
        }
    }
    v
}

/// A membrane that rests on its threshold never fires under NIR's strict crossing and fires under Lif.
#[test]
fn nir_strict_threshold_and_lif_non_strict_threshold_disagree_on_a_resting_membrane() {
    let mut g = Graph::new();
    g.push("in", Node::Input(Input { shape: vec![1] }));
    g.push(
        "n",
        Node::Lif(NirLif {
            shape: vec![1],
            tau: vec![20e-3],
            r: vec![0.0],
            v_leak: vec![-50e-3],
            v_threshold: vec![-50e-3],
            v_reset: vec![-65e-3],
        }),
    );
    g.push("out", Node::Output(Output { shape: vec![1] }));
    g.edge("in", "n");
    g.edge("n", "out");
    let text = g.to_text();
    assert_eq!(
        Graph::from_text(&text).expect("text round-trip").to_text(),
        text
    );

    let mut nir_side = CubaState {
        tau_syn: 0.0,
        tau_mem: 20e-3,
        r: 0.0,
        v_leak: -50e-3,
        v_threshold: -50e-3,
        v_reset: -65e-3,
        w_in: 1.0,
        i_syn: 0.0,
        v: -50e-3,
    };
    for _ in 0..20 {
        assert!(
            !nir_side.step_exact(1e-4, 0.0),
            "NIR fires on a strict crossing"
        );
    }

    let converted = g.to_net(1e-4).expect("a LIF node is inside the bridge");
    let mut sim =
        Sim::new(converted.net, converted.neurons, 1e-4, Mode::Clocked).expect("counts match");
    assert_eq!(
        sim.step(&[]),
        vec![0],
        "the bridge keeps Lif's non-strict test"
    );
}

/// Operation counts price only where a table has a price, and a pre-silicon figure is not a meter reading.
#[test]
fn an_operation_count_is_not_a_measured_joule() {
    let mut b = NetBuilder::new(3);
    b.connect(0, 1, 20e-3, 1).unwrap();
    b.connect(1, 2, 20e-3, 1).unwrap();
    let net = b.build();
    assert_eq!(net.n_syn, 2, "each synapse is stored once");
    let mut sim = Sim::new(net, vec![Lif::default(); 3], 1e-4, Mode::EventDriven).unwrap();
    let mut ext = vec![0.0; 3];
    ext[0] = 3e-9;
    let train = sim.run(2_000, &ext);
    assert!(
        !train.is_empty(),
        "the chain produced no spikes, so the bill would be vacuous"
    );
    assert!(
        sim.ledger.syn_ops > 0,
        "synaptic operations were not counted"
    );

    let bill = sim.ledger.bill(&TRUENORTH_2014);
    assert!(
        bill.total.is_none(),
        "TrueNorth's all-in table produced a total"
    );
    assert!(bill.unpriced.contains(&"synapse memory fetch"));
    let synops = sim.ledger.joules_synops_only(&TRUENORTH_2014);
    assert!(
        synops.is_some_and(|j| j > 0.0),
        "the synapse-only figure is how a count becomes a joule, and it is not the bill"
    );
    assert!(sim.ledger.joules(&TRUENORTH_2014).is_none());

    assert_eq!(LOIHI_2018.evidence, Evidence::Simulated);
    assert!(LOIHI_2018.e_syn_fetch.is_none());
    assert!(
        sim.ledger.joules(&LOIHI_2018).is_none(),
        "Loihi's pre-silicon pJ/SOP is not a complete bill"
    );
}

/// Same seed, same draws. A second seed is a different stream.
#[test]
fn the_same_seed_reproduces_the_spike_train() {
    let mut a = Rng::new(0x5EED_u64);
    let mut b = Rng::new(0x5EED_u64);
    let mut c = Rng::new(0x5EED_u64 ^ 0xFFFF);
    let (mut da, mut db, mut dc) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..8 {
        da.push(a.next_u32());
        db.push(b.next_u32());
        dc.push(c.next_u32());
    }
    assert_eq!(da, db);
    assert_ne!(da, dc);

    let encode = |seed| {
        let mut enc = RateEncoder::new(100.0, 1e-3, seed);
        enc.encode(&[0.8, 0.2], 200)
    };
    assert_eq!(encode(7), encode(7));
    assert_ne!(encode(7), encode(8));
}
