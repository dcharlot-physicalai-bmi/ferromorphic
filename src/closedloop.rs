//! Closed-loop Mackey–Glass readout designs, with the grids fixed before any held-out score.
//!
//! The pinned `NeuroBench` comment `40.803375080880265` is the repeat-1 mean `sMAPE` over tau 17
//! through 30. It is the target these designs are judged against. It is not a number this module
//! searches for: every grid below is a constant, validation is the last fifth of the post-warmup
//! training window, and a caller scores the `NeuroBench` test split once per config.
//!
//! The reservoir matrices stay the caller's. What changes is how the readout row is fit:
//! a ridge penalty on every coefficient, a Tikhonov penalty on the state-block of that row only,
//! a second rollout whose drive blends the teacher into the first readout's own output, or
//! Gaussian state noise during the teacher-forced collection. [`Rng`] is the only randomness.

use crate::metrics::{smape, MetricError};
use crate::reservoir::{cholesky, SuppliedEcho, ReservoirError};
use crate::rng::Rng;

/// The repeat-1 comment in `NeuroBench` `examples/mackey_glass/esn_benchmark.py` at
/// `e521c28f5798cb40bb9423758bf066dad53fe070`. It is a 14-tau mean, not a tau-30 score.
pub const PUBLISHED_REPEAT1_SMAPE: f64 = 40.803375080880265;

/// Absolute gap against the reference trajectory that counts as a divergence.
pub const DIVERGE_ABS: f64 = 1e-8;

/// Absolute output that counts as a blow-up. The rollout still finishes when the value is finite.
pub const BLOWUP_ABS: f64 = 1.0e6;

/// Fraction of the post-warmup training window held out of the fit and scored autonomously.
pub const VAL_FRACTION: f64 = 0.2;

/// Power-iteration steps for the closed-loop Jacobian's spectral-radius estimate.
pub const JACOBIAN_ITERS: usize = 48;

/// Ridge penalties, including the tau-30 table value `1e-8`. Added to every Gram diagonal.
pub const RIDGE_ALPHA: [f64; 7] = [1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1.0, 10.0];

/// Tikhonov penalties on the state coefficients of the readout row. The `[1, u]` columns are not
/// penalised. The tau-30 table value `1e-8` is on the grid.
pub const STATE_ROW_ALPHA: [f64; 6] = [1e-8, 1e-6, 1e-4, 1e-2, 1.0, 10.0];

/// End of the linear blend from teacher forcing (`0`) to the model's own previous output (`1`).
pub const SCHEDULE_END: [f64; 5] = [0.0, 0.25, 0.5, 0.75, 1.0];

/// Standard deviation of the Gaussian added to each reservoir unit after warmup.
pub const NOISE_SIGMA: [f64; 5] = [0.0, 1e-4, 1e-3, 1e-2, 0.1];

/// Seeds for every positive [`NOISE_SIGMA`]. Sigma `0` does not draw.
pub const NOISE_SEEDS: [u64; 3] = [1, 2, 3];

/// Which diagonal of the readout Gram is penalised.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Penalty {
    /// `alpha` on every coefficient, including the constant and the input column.
    ///
    /// This is the `NeuroBench` `ridge_param * I` placement.
    Ridge {
        /// Non-negative penalty.
        alpha: f64,
    },
    /// `alpha` on the state block of the readout row only.
    StateRow {
        /// Non-negative penalty.
        alpha: f64,
    },
}

/// One predeclared configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Config {
    /// Readout penalty.
    pub penalty: Penalty,
    /// Linear scheduled-sampling end blend. `0` stays on the teacher.
    pub schedule_end: f64,
    /// State-noise standard deviation. `0` adds nothing.
    pub noise_sigma: f64,
    /// [`Rng`] seed used when `noise_sigma > 0`.
    pub noise_seed: u64,
    /// Fraction of the post-warmup window kept out of the fit. `0` fits every post-warmup step.
    pub val_fraction: f64,
}

/// Scores and stability for one config.
#[derive(Clone, Debug, PartialEq)]
pub struct Report {
    /// Validation autonomous sMAPE, percent. `None` when a prediction was not finite.
    pub val_smape: Option<f64>,
    /// Held-out autonomous sMAPE, percent. `None` when a prediction was not finite.
    pub held_smape: Option<f64>,
    /// First held-out index with `|prediction - reference| > `[`DIVERGE_ABS`].
    pub diverge_step: Option<usize>,
    /// Absolute gap at [`Report::diverge_step`].
    pub diverge_gap: Option<f64>,
    /// First held-out index with a non-finite prediction or `|prediction| > `[`BLOWUP_ABS`].
    pub blowup_step: Option<usize>,
    /// Spectral-radius estimate of the closed-loop Jacobian at the start of the held-out roll.
    pub radius_at_start: f64,
    /// The same estimate at the state that produced [`Report::diverge_step`], when that step exists.
    pub radius_at_diverge: Option<f64>,
    /// Euclidean norm of the state coefficients of the fitted readout row.
    pub state_row_norm: f64,
}

/// Train inputs, the target aligned with each input, and the held-out pair plus a reference roll.
#[derive(Clone, Debug, PartialEq)]
pub struct Task {
    /// Scalar drive for each training step.
    pub train_u: Vec<f64>,
    /// Target aligned with [`Task::train_u`]. Same length.
    pub train_y: Vec<f64>,
    /// Steps at the start of training that are rolled but not fit.
    pub warmup: usize,
    /// Held-out drives. Autonomous mode consumes only the first.
    pub test_u: Vec<f64>,
    /// Held-out targets, same length as [`Task::test_u`].
    pub test_y: Vec<f64>,
    /// Reference autonomous trajectory, same length as [`Task::test_u`].
    pub reference: Vec<f64>,
}

/// One standard normal from [`Rng`], Box–Muller, using two uniforms.
///
/// `next_f64` lies in `[0, 1)`. The log argument is `1 - u`, which lies in `(0, 1]`, so the
/// logarithm stays finite.
pub fn standard_normal(rng: &mut Rng) -> f64 {
    let u1 = 1.0 - rng.next_f64();
    let u2 = rng.next_f64();
    let radius = (-2.0 * u1.ln()).sqrt();
    radius * (std::f64::consts::TAU * u2).cos()
}

/// Closed-loop Jacobian of `(state, drive) -> (next state, readout)` at the current echo state.
///
/// Row-major, order `units + 1`. The last coordinate is the drive, and the readout is the value
/// that autonomous mode feeds back as the next drive. The estimate of its spectral radius is
/// [`spectral_radius`].
///
/// # Errors
///
/// [`ReservoirError::OutOfRange`] when the echo does not keep the bias and the input in the readout.
pub fn closed_loop_jacobian(echo: &SuppliedEcho, drive: f64) -> Result<Vec<f64>, ReservoirError> {
    if !echo.include_bias || !echo.include_input || echo.in_width != 2 || echo.n_out != 1 {
        return Err(ReservoirError::OutOfRange {
            what: "closed-loop feature layout",
            value: echo.in_width as f64,
            low: 2.0,
            high: 2.0,
        });
    }
    let n = echo.units;
    let mut phi = vec![0.0; n];
    for i in 0..n {
        let mut pre = echo.w_in[i * 2] + echo.w_in[i * 2 + 1] * drive;
        for j in 0..n {
            pre += echo.w[i * n + j] * echo.state[j];
        }
        let t = pre.tanh();
        phi[i] = echo.leakage * (1.0 - t * t);
    }
    let dim = n + 1;
    let mut jac = vec![0.0; dim * dim];
    for i in 0..n {
        for j in 0..n {
            let diag = if i == j { 1.0 - echo.leakage } else { 0.0 };
            jac[i * dim + j] = diag + phi[i] * echo.w[i * n + j];
        }
        jac[i * dim + n] = phi[i] * echo.w_in[i * 2 + 1];
    }
    // Readout: [1, drive, next state]. dy/dx = wx * dx'/dx, dy/du = w_u + wx * dx'/du.
    let wx = &echo.w_out[2..];
    for j in 0..n {
        let mut acc = 0.0;
        for i in 0..n {
            acc += wx[i] * jac[i * dim + j];
        }
        jac[n * dim + j] = acc;
    }
    let mut acc = echo.w_out[1];
    for i in 0..n {
        acc += wx[i] * jac[i * dim + n];
    }
    jac[n * dim + n] = acc;
    Ok(jac)
}

/// Magnitude of the dominant eigenvalue, by power iteration from the all-ones direction.
#[must_use]
pub fn spectral_radius(matrix: &[f64], n: usize, iters: usize) -> f64 {
    if n == 0 || matrix.len() != n * n || iters == 0 {
        return 0.0;
    }
    let mut v = vec![1.0 / (n as f64).sqrt(); n];
    let mut w = vec![0.0; n];
    for _ in 0..iters {
        apply(matrix, n, &v, &mut w);
        let norm = norm2(&w);
        if norm == 0.0 {
            return 0.0;
        }
        for i in 0..n {
            v[i] = w[i] / norm;
        }
    }
    apply(matrix, n, &v, &mut w);
    norm2(&w)
}

/// Fit a readout row on `rows` (each `[1, u, state…]`) against scalar `targets`.
///
/// # Errors
///
/// [`ReservoirError::Empty`] when there are no rows, [`ReservoirError::ShapeMismatch`] when a row
/// has a different width, [`ReservoirError::OutOfRange`] for a negative penalty, and
/// [`ReservoirError::IllConditioned`] when the penalised Gram is not usable.
pub fn fit_readout(rows: &[Vec<f64>], targets: &[f64], penalty: Penalty) -> Result<Vec<f64>, ReservoirError> {
    if rows.is_empty() {
        return Err(ReservoirError::Empty { what: "readout rows" });
    }
    if targets.len() != rows.len() {
            return Err(ReservoirError::ShapeMismatch {
                what: "readout targets",
                got: targets.len(),
                want: rows.len(),
            });
    }
    let p = rows[0].len();
    if p < 3 {
        return Err(ReservoirError::OutOfRange {
            what: "readout width",
            value: p as f64,
            low: 3.0,
            high: f64::INFINITY,
        });
    }
    let alpha = match penalty {
        Penalty::Ridge { alpha } | Penalty::StateRow { alpha } => alpha,
    };
    if !(alpha >= 0.0) || !alpha.is_finite() {
        return Err(ReservoirError::OutOfRange {
            what: "readout penalty",
            value: alpha,
            low: 0.0,
            high: f64::INFINITY,
        });
    }
    let mut gram = vec![0.0; p * p];
    let mut cross = vec![0.0; p];
    for (row, &y) in rows.iter().zip(targets) {
        if row.len() != p {
            return Err(ReservoirError::ShapeMismatch {
                what: "readout row",
                got: row.len(),
                want: p,
            });
        }
        for i in 0..p {
            cross[i] += row[i] * y;
            for j in 0..p {
                gram[i * p + j] += row[i] * row[j];
            }
        }
    }
    match penalty {
        Penalty::Ridge { alpha } => {
            for i in 0..p {
                gram[i * p + i] += alpha;
            }
        }
        Penalty::StateRow { alpha } => {
            for i in 2..p {
                gram[i * p + i] += alpha;
            }
        }
    }
    let factor = cholesky(&gram, p, 1e-12)?;
    factor.solve(&cross)
}

/// Roll `echo` from its current state and replace its readout with one config's fit.
///
/// `echo.w_out` is ignored on entry and replaced by the fit. The state on entry is the training
/// origin, which for the `NeuroBench` protocol is the zero vector. On return the state is where
/// training finished, and `w_out` is the fitted row.
///
/// # Errors
///
/// [`ReservoirError`] when the task lengths disagree, the windows are empty, or the fit is refused.
/// [`MetricError`] is mapped to [`ReservoirError::NonFinite`] when a scored series is not finite.
pub fn run(echo: &mut SuppliedEcho, task: &Task, config: Config) -> Result<Report, ReservoirError> {
    check_task(task)?;
    if !(config.schedule_end >= 0.0 && config.schedule_end <= 1.0) || !(config.noise_sigma >= 0.0) {
        return Err(ReservoirError::OutOfRange {
            what: "closed-loop config",
            value: config.schedule_end,
            low: 0.0,
            high: 1.0,
        });
    }
    let windows = windows(task.train_u.len(), task.warmup, config.val_fraction)?;
    // Pass 1 is always teacher-forced. A positive schedule refits on a second blended rollout.
    let first = collect(echo, task, &windows, None, config.noise_sigma, config.noise_seed, config.penalty)?;
    let fitted = if config.schedule_end == 0.0 {
        first
    } else {
        echo.state.fill(0.0);
        collect(
            echo,
            task,
            &windows,
            Some((&first.readout, config.schedule_end)),
            config.noise_sigma,
            config.noise_seed,
            config.penalty,
        )?
    };
    let readout = fitted.readout;
    echo.w_out.clone_from(&readout);
    let state_row_norm = norm2(&readout[2..]);

    let val_start = windows.fit_end;
    let mut val_net = echo.clone();
    val_net.state.clone_from(&fitted.state_at_fit_end);
    val_net.w_out.clone_from(&readout);
    let val_len = task.train_u.len() - val_start;
    let val_pred = if val_len == 0 {
        Vec::new()
    } else {
        val_net.autonomous(&task.train_u[val_start..])?
    };
    let val_smape = if val_len == 0 {
        None
    } else {
        scored(&val_pred, &task.train_y[val_start..])?
    };

    let radius_at_start = spectral_radius(
        &closed_loop_jacobian(echo, task.test_u[0])?,
        echo.units + 1,
        JACOBIAN_ITERS,
    );
    let held = autonomous_report(echo, task)?;
    let pred = held.pred;
    let held_smape = scored(&pred, &task.test_y)?;
    Ok(Report {
        val_smape,
        held_smape,
        diverge_step: held.diverge,
        diverge_gap: held.diverge_gap,
        blowup_step: held.blowup,
        radius_at_start,
        radius_at_diverge: held.radius_at_diverge,
        state_row_norm,
    })
}

fn scored(pred: &[f64], truth: &[f64]) -> Result<Option<f64>, ReservoirError> {
    if pred.iter().any(|v| !v.is_finite()) {
        return Ok(None);
    }
    match smape(pred, truth) {
        Ok(v) => Ok(Some(v)),
        Err(MetricError::NonFinite { .. }) => Ok(None),
        Err(_) => Err(ReservoirError::ShapeMismatch {
            what: "sMAPE",
            got: pred.len(),
            want: truth.len(),
        }),
    }
}

struct HeldRoll {
    pred: Vec<f64>,
    diverge: Option<usize>,
    diverge_gap: Option<f64>,
    blowup: Option<usize>,
    radius_at_diverge: Option<f64>,
}

fn autonomous_report(echo: &SuppliedEcho, task: &Task) -> Result<HeldRoll, ReservoirError> {
    let mut net = echo.clone();
    let mut drive = task.test_u[0];
    let mut pred = Vec::with_capacity(task.test_y.len());
    let mut diverge = None;
    let mut diverge_gap = None;
    let mut blowup = None;
    let mut radius_at_diverge = None;
    for i in 0..task.test_y.len() {
        let before = net.clone();
        let y = net.step(drive)?;
        pred.push(y);
        if diverge.is_none() && (y - task.reference[i]).abs() > DIVERGE_ABS {
            diverge = Some(i);
            diverge_gap = Some((y - task.reference[i]).abs());
            radius_at_diverge = Some(spectral_radius(
                &closed_loop_jacobian(&before, drive)?,
                before.units + 1,
                JACOBIAN_ITERS,
            ));
        }
        if blowup.is_none() && (!y.is_finite() || y.abs() > BLOWUP_ABS) {
            blowup = Some(i);
        }
        drive = y;
        if !drive.is_finite() {
            break;
        }
    }
    while pred.len() < task.test_y.len() {
        pred.push(f64::NAN);
    }
    Ok(HeldRoll { pred, diverge, diverge_gap, blowup, radius_at_diverge })
}

struct Collected {
    readout: Vec<f64>,
    /// Reservoir state after the last fit step, before the validation window is driven.
    state_at_fit_end: Vec<f64>,
}

struct Windows {
    warmup: usize,
    fit_end: usize,
}

fn windows(train_len: usize, warmup: usize, val_fraction: f64) -> Result<Windows, ReservoirError> {
    if warmup >= train_len {
        return Err(ReservoirError::OutOfRange {
            what: "warmup",
            value: warmup as f64,
            low: 0.0,
            high: train_len as f64,
        });
    }
    if !(0.0..1.0).contains(&val_fraction) {
        return Err(ReservoirError::OutOfRange {
            what: "val_fraction",
            value: val_fraction,
            low: 0.0,
            high: 1.0,
        });
    }
    let post = train_len - warmup;
    let val = ((post as f64) * val_fraction).floor() as usize;
    let fit_end = train_len - val;
    if fit_end <= warmup {
        return Err(ReservoirError::Empty { what: "fit window" });
    }
    Ok(Windows { warmup, fit_end })
}

fn collect(
    echo: &mut SuppliedEcho,
    task: &Task,
    windows: &Windows,
    blend: Option<(&[f64], f64)>,
    noise_sigma: f64,
    noise_seed: u64,
    penalty: Penalty,
) -> Result<Collected, ReservoirError> {
    echo.state.fill(0.0);
    if let Some((row, _)) = blend {
        echo.w_out.clone_from(&row.to_vec());
    }
    let mut rng = Rng::new(noise_seed);
    let mut rows = Vec::new();
    let mut targets = Vec::new();
    let mut prev = task.train_u[0];
    let mut state_at_fit_end = echo.state.clone();
    let fit_len = windows.fit_end - windows.warmup;
    for t in 0..task.train_u.len() {
        let teacher = task.train_u[t];
        let p = match blend {
            Some((_, end)) if t >= windows.warmup => {
                let j = t - windows.warmup;
                end * (j as f64) / (fit_len.saturating_sub(1).max(1) as f64)
            }
            _ => 0.0,
        };
        let drive = (1.0 - p) * teacher + p * prev;
        let y = echo.step(drive)?;
        if noise_sigma > 0.0 && t >= windows.warmup {
            for unit in &mut echo.state {
                *unit += noise_sigma * standard_normal(&mut rng);
            }
        }
        if t >= windows.warmup && t < windows.fit_end {
            let mut feat = Vec::with_capacity(2 + echo.units);
            feat.push(1.0);
            feat.push(drive);
            feat.extend_from_slice(&echo.state);
            rows.push(feat);
            targets.push(task.train_y[t]);
        }
        prev = if blend.is_some() { dot(&echo.w_out, &feature_now(echo, drive)) } else { y };
        if t + 1 == windows.fit_end {
            state_at_fit_end.clone_from(&echo.state);
        }
    }
    Ok(Collected { readout: fit_readout(&rows, &targets, penalty)?, state_at_fit_end })
}

fn feature_now(echo: &SuppliedEcho, drive: f64) -> Vec<f64> {
    let mut feat = Vec::with_capacity(2 + echo.units);
    feat.push(1.0);
    feat.push(drive);
    feat.extend_from_slice(&echo.state);
    feat
}

fn check_task(task: &Task) -> Result<(), ReservoirError> {
    if task.train_u.len() != task.train_y.len() || task.train_u.is_empty() {
        return Err(ReservoirError::ShapeMismatch {
            what: "train series",
            got: task.train_u.len(),
            want: task.train_y.len(),
        });
    }
    if task.test_u.is_empty() || task.test_u.len() != task.test_y.len() || task.test_y.len() != task.reference.len() {
        return Err(ReservoirError::ShapeMismatch {
            what: "held-out series",
            got: task.test_u.len(),
            want: task.test_y.len(),
        });
    }
    Ok(())
}

fn apply(matrix: &[f64], n: usize, v: &[f64], out: &mut [f64]) {
    for i in 0..n {
        let mut s = 0.0;
        for j in 0..n {
            s += matrix[i * n + j] * v[j];
        }
        out[i] = s;
    }
}

fn norm2(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn dot(w: &[f64], x: &[f64]) -> f64 {
    w.iter().zip(x).map(|(a, b)| a * b).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny() -> SuppliedEcho {
        SuppliedEcho::try_new(
            1,
            2,
            1,
            vec![0.0, 0.5],
            vec![0.2],
            vec![0.0, 0.0, 1.0],
            1.0,
            vec![0.1],
            true,
            true,
        )
        .unwrap()
    }

    #[test]
    fn the_same_seed_replays_one_standard_normal() {
        let mut a = Rng::new(7);
        let mut b = Rng::new(7);
        let x = standard_normal(&mut a);
        let y = standard_normal(&mut b);
        assert_eq!(x, y);
        assert!(x.is_finite());
        let mut rng = Rng::new(3);
        let mut acc = 0.0;
        let mut acc2 = 0.0;
        let n = 4_000.0;
        for _ in 0..4_000 {
            let z = standard_normal(&mut rng);
            acc += z;
            acc2 += z * z;
        }
        let mean = acc / n;
        let second = acc2 / n;
        assert!(mean.abs() < 0.08, "{mean}");
        assert!((second - 1.0).abs() < 0.15, "{second}");
    }

    #[test]
    fn a_one_unit_jacobian_matches_the_hand_derivative() {
        let echo = tiny();
        let drive = 0.3;
        let jac = closed_loop_jacobian(&echo, drive).unwrap();
        let pre = 0.2 * 0.1 + 0.5 * drive;
        let phi = 1.0 - pre.tanh() * pre.tanh();
        let dx_dx = phi * 0.2;
        let dx_du = phi * 0.5;
        assert!((jac[0] - dx_dx).abs() < 1e-12);
        assert!((jac[1] - dx_du).abs() < 1e-12);
        assert!((jac[2] - dx_dx).abs() < 1e-12, "y = next state, so dy/dx = dx'/dx");
        assert!((jac[3] - dx_du).abs() < 1e-12);
        let mut net = echo.clone();
        let y0 = net.step(drive).unwrap();
        let mut bumped = echo.clone();
        bumped.state[0] += 1e-6;
        let y1 = bumped.step(drive).unwrap();
        let empirical = (y1 - y0) / 1e-6;
        assert!((empirical - jac[2]).abs() < 1e-6, "{empirical} vs {}", jac[2]);
    }

    #[test]
    fn power_iteration_reads_the_larger_diagonal_entry() {
        let matrix = [3.0, 0.0, 0.0, -1.0];
        let radius = spectral_radius(&matrix, 2, 30);
        assert!((radius - 3.0).abs() < 1e-9, "{radius}");
    }

    #[test]
    fn ridge_on_a_line_recovers_the_slope_and_the_state_penalty_shrinks_it() {
        let rows = vec![vec![1.0, 0.2, 0.0], vec![1.0, -0.3, 1.0], vec![1.0, 0.5, 2.0]];
        let y = [1.0, 2.0, 3.0];
        let plain = fit_readout(&rows, &y, Penalty::Ridge { alpha: 0.0 }).unwrap();
        assert!((plain[0] - 1.0).abs() < 1e-6, "{plain:?}");
        assert!((plain[2] - 1.0).abs() < 1e-6, "{plain:?}");
        let shrunk = fit_readout(&rows, &y, Penalty::StateRow { alpha: 10.0 }).unwrap();
        assert!(shrunk[2].abs() < plain[2].abs());
    }

    #[test]
    fn a_zero_schedule_matches_teacher_forcing() {
        let mut echo = tiny();
        echo.w_out = vec![0.0, 0.0, 0.0];
        echo.state = vec![0.0];
        let task = Task {
            train_u: vec![0.2, -0.1, 0.4, 0.0, 0.3, -0.2],
            train_y: vec![0.1, 0.0, 0.2, -0.1, 0.15, 0.05],
            warmup: 1,
            test_u: vec![0.2, 0.0],
            test_y: vec![0.1, 0.0],
            reference: vec![0.0, 0.0],
        };
        let teacher = Config {
            penalty: Penalty::Ridge { alpha: 1e-6 },
            schedule_end: 0.0,
            noise_sigma: 0.0,
            noise_seed: 1,
            val_fraction: 0.0,
        };
        let mut a = echo.clone();
        let mut b = echo.clone();
        let ra = run(&mut a, &task, teacher).unwrap();
        let scheduled = Config { schedule_end: 0.0, ..teacher };
        let rb = run(&mut b, &task, scheduled).unwrap();
        assert_eq!(a.w_out, b.w_out);
        assert_eq!(ra.held_smape, rb.held_smape);
        assert!(ra.state_row_norm.is_finite());
    }
}
