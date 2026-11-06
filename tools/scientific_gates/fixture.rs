//! Standalone specimen for the analytic, numerical and convergence gates.
//!
//! Compiled with `rustc --edition 2024 -O`. It does not link the crate: `tools/mutation_selftest.py`
//! mutates a copy of this file and requires the mutated program to fail. The same predicates are
//! applied to `ferromorphic` in `tests/scientific_gates.rs`.
//!
//! The membrane is the linear leak `dv/dt = -(v - v_rest) / tau`, whose solution is
//! `v_rest + (v0 - v_rest) exp(-t/tau)`. Exponential Euler is that solution over a step.
//! Forward Euler is first order. A stepper that ignores `dt` does not converge.

const CLOSED_FORM_TOL: f64 = 1e-12;
const ORDER_MIN: f64 = 0.8;
const ORDER_MAX: f64 = 1.5;

fn closed(v0: f64, v_rest: f64, tau: f64, t: f64) -> f64 {
    v_rest + (v0 - v_rest) * (-t / tau).exp()
}

fn exp_euler(v: f64, v_rest: f64, tau: f64, dt: f64) -> f64 {
    v_rest + (v - v_rest) * (-dt / tau).exp()
}

fn forward_euler(v: f64, v_rest: f64, tau: f64, dt: f64) -> f64 {
    v + dt * (-(v - v_rest) / tau)
}

/// Ignores the step. The error against the closed form does not shrink when `dt` does.
fn frozen(v: f64, _v_rest: f64, _tau: f64, _dt: f64) -> f64 {
    v
}

fn march(
    mut v: f64,
    v_rest: f64,
    tau: f64,
    dt: f64,
    steps: u32,
    step: fn(f64, f64, f64, f64) -> f64,
) -> f64 {
    for _ in 0..steps {
        v = step(v, v_rest, tau, dt);
    }
    v
}

fn abs_err(got: f64, want: f64) -> f64 {
    (got - want).abs()
}

fn orders(errors: &[f64]) -> Vec<f64> {
    errors.windows(2).map(|w| (w[0] / w[1]).log2()).collect()
}

fn order_accepted(order: f64) -> bool {
    order >= ORDER_MIN && order <= ORDER_MAX
}

fn main() {
    let (v0, v_rest, tau) = (-60e-3, -65e-3, 20e-3);
    assert!(
        v0 < -50e-3 && v0 > v_rest,
        "subthreshold, and not already at rest"
    );
    let t_end = 20e-3;
    let want = closed(v0, v_rest, tau, t_end);

    let dt_fine = 1e-4;
    let steps_fine = (t_end / dt_fine).round() as u32;
    let exact = march(v0, v_rest, tau, dt_fine, steps_fine, exp_euler);
    let exact_err = abs_err(exact, want);
    assert!(
        exact_err < CLOSED_FORM_TOL,
        "exponential Euler error {exact_err} exceeded {CLOSED_FORM_TOL}"
    );

    // The tolerance is not vacuous: coarse forward Euler misses it.
    let dt_coarse = 1e-3;
    let steps_coarse = (t_end / dt_coarse).round() as u32;
    let euler = march(v0, v_rest, tau, dt_coarse, steps_coarse, forward_euler);
    let euler_err = abs_err(euler, want);
    assert!(
        euler_err > CLOSED_FORM_TOL,
        "forward Euler error {euler_err} met {CLOSED_FORM_TOL}; the gate cannot fail"
    );

    // Measured order of forward Euler on a refinement where truncation dominates roundoff.
    let mut euler_errors = Vec::new();
    for dt in [1e-3, 5e-4, 2.5e-4] {
        let steps = (t_end / dt).round() as u32;
        let got = march(v0, v_rest, tau, dt, steps, forward_euler);
        euler_errors.push(abs_err(got, want));
    }
    let euler_orders = orders(&euler_errors);
    println!("MEASURED forward_euler_errors={euler_errors:?} orders={euler_orders:?}");
    for order in &euler_orders {
        assert!(
            order_accepted(*order),
            "forward Euler order {order} outside [{ORDER_MIN}, {ORDER_MAX}]"
        );
    }

    // A frozen stepper does not converge. The gate must reject it.
    let mut frozen_errors = Vec::new();
    for dt in [1e-3, 5e-4, 2.5e-4] {
        let steps = (t_end / dt).round() as u32;
        let got = march(v0, v_rest, tau, dt, steps, frozen);
        frozen_errors.push(abs_err(got, want));
    }
    let frozen_orders = orders(&frozen_errors);
    println!("MEASURED frozen_errors={frozen_errors:?} orders={frozen_orders:?}");
    for order in &frozen_orders {
        assert!(
            !order_accepted(*order),
            "nonconvergent order {order} was accepted; errors {frozen_errors:?}"
        );
    }

    // Independent check of the same closed form: the series of the exponential at a tiny time,
    // where the truncation of the series is smaller than the tolerance and no stepper is involved.
    let t = 1e-6;
    let x = -t / tau;
    let series = 1.0 + x + x * x / 2.0 + x * x * x / 6.0 + x * x * x * x / 24.0;
    let from_series = v_rest + (v0 - v_rest) * series;
    let from_exp = closed(v0, v_rest, tau, t);
    let series_gap = abs_err(from_series, from_exp);
    println!("MEASURED series_gap={series_gap}");
    assert!(
        series_gap < 1e-18,
        "series and exp diverged by {series_gap}"
    );
}
