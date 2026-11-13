//! Tau-30 closed-loop readout sweeps. The grids live in [`ferromorphic::closedloop`].
//!
//! ```text
//! closed_loop_sweep DIR
//! ```
//!
//! `DIR` holds `w_in.npy`, `w.npy`, `train_u.npy`, `train_y.npy`, `test_u.npy`, `test_y.npy`,
//! `reference.npy`, and `meta.txt` with `leakage`, `warmup`, and `ridge` lines. The `NeuroBench`
//! test split is scored once per predeclared config. Validation is the last fifth of the
//! post-warmup training window.

use ferromorphic::closedloop::{
    run, Config, Penalty, Task, BLOWUP_ABS, DIVERGE_ABS, NOISE_SEEDS, NOISE_SIGMA, PUBLISHED_REPEAT1_SMAPE,
    RIDGE_ALPHA, SCHEDULE_END, STATE_ROW_ALPHA, VAL_FRACTION,
};
use ferromorphic::npy;
use ferromorphic::reservoir::SuppliedEcho;
use std::env;
use std::fs;
use std::path::Path;
use std::process::ExitCode;

fn main() -> ExitCode {
    match run_dir() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("{err}");
            ExitCode::from(2)
        }
    }
}

fn run_dir() -> Result<(), String> {
    let dir = env::args().nth(1).ok_or("usage: closed_loop_sweep DIR")?;
    let dir = Path::new(&dir);
    let meta = fs::read_to_string(dir.join("meta.txt")).map_err(|e| e.to_string())?;
    let leakage = field(&meta, "leakage")?;
    let warmup = field(&meta, "warmup")? as usize;
    let ridge = field(&meta, "ridge")?;
    let w_in = load(dir, "w_in.npy")?;
    let w = load(dir, "w.npy")?;
    let units = w.len().isqrt();
    if units * units != w.len() || w_in.len() != units * 2 {
        return Err(format!("weight shapes units {units} w_in {}", w_in.len()));
    }
    let task = Task {
        train_u: load(dir, "train_u.npy")?,
        train_y: load(dir, "train_y.npy")?,
        warmup,
        test_u: load(dir, "test_u.npy")?,
        test_y: load(dir, "test_y.npy")?,
        reference: load(dir, "reference.npy")?,
    };
    println!("published_repeat1_smape {PUBLISHED_REPEAT1_SMAPE:.15}");
    println!("diverge_abs {DIVERGE_ABS:e} blowup_abs {BLOWUP_ABS:e} val_fraction {VAL_FRACTION}");
    println!("protocol_ridge {ridge:e} leakage {leakage} warmup {warmup} units {units}");
    println!("name\tval_smape\theld_smape\tdiverge\tdiverge_gap\tblowup\tradius_start\tradius_diverge\tstate_row_norm");
    let mut echo = SuppliedEcho::try_new(
        units, 2, 1, w_in, w, vec![0.0; 2 + units], leakage, vec![0.0; units], true, true,
    )
    .map_err(|e| e.to_string())?;
    let protocol = Config {
        penalty: Penalty::Ridge { alpha: ridge },
        schedule_end: 0.0,
        noise_sigma: 0.0,
        noise_seed: 0,
        val_fraction: 0.0,
    };
    emit(&mut echo, &task, "protocol_full_train", protocol)?;
    for alpha in RIDGE_ALPHA {
        let config = Config {
            penalty: Penalty::Ridge { alpha },
            schedule_end: 0.0,
            noise_sigma: 0.0,
            noise_seed: 0,
            val_fraction: VAL_FRACTION,
        };
        emit(&mut echo, &task, &format!("ridge_{alpha:e}"), config)?;
    }
    for alpha in STATE_ROW_ALPHA {
        let config = Config {
            penalty: Penalty::StateRow { alpha },
            schedule_end: 0.0,
            noise_sigma: 0.0,
            noise_seed: 0,
            val_fraction: VAL_FRACTION,
        };
        emit(&mut echo, &task, &format!("state_row_{alpha:e}"), config)?;
    }
    for end in SCHEDULE_END {
        let config = Config {
            penalty: Penalty::Ridge { alpha: ridge },
            schedule_end: end,
            noise_sigma: 0.0,
            noise_seed: 0,
            val_fraction: VAL_FRACTION,
        };
        emit(&mut echo, &task, &format!("schedule_{end}"), config)?;
    }
    emit(
        &mut echo,
        &task,
        "noise_0",
        Config {
            penalty: Penalty::Ridge { alpha: ridge },
            schedule_end: 0.0,
            noise_sigma: 0.0,
            noise_seed: 0,
            val_fraction: VAL_FRACTION,
        },
    )?;
    for sigma in NOISE_SIGMA.iter().copied().filter(|s| *s > 0.0) {
        for seed in NOISE_SEEDS {
            let config = Config {
                penalty: Penalty::Ridge { alpha: ridge },
                schedule_end: 0.0,
                noise_sigma: sigma,
                noise_seed: seed,
                val_fraction: VAL_FRACTION,
            };
            emit(&mut echo, &task, &format!("noise_{sigma:e}_seed_{seed}"), config)?;
        }
    }
    Ok(())
}

fn emit(echo: &mut SuppliedEcho, task: &Task, name: &str, config: Config) -> Result<(), String> {
    match run(echo, task, config) {
        Ok(report) => println!(
            "{name}\t{}\t{}\t{}\t{}\t{}\t{:.6e}\t{}\t{:.6e}",
            fmt_opt(report.val_smape),
            fmt_opt(report.held_smape),
            fmt_idx(report.diverge_step),
            fmt_opt(report.diverge_gap),
            fmt_idx(report.blowup_step),
            report.radius_at_start,
            fmt_opt(report.radius_at_diverge),
            report.state_row_norm,
        ),
        Err(err) => println!("{name}\trefused\t{err}"),
    }
    Ok(())
}

fn fmt_opt(v: Option<f64>) -> String {
    match v {
        Some(x) => format!("{x:.8}"),
        None => "none".to_string(),
    }
}

fn fmt_idx(v: Option<usize>) -> String {
    match v {
        Some(i) => i.to_string(),
        None => "none".to_string(),
    }
}

fn field(text: &str, key: &str) -> Result<f64, String> {
    for line in text.lines() {
        let mut parts = line.split_whitespace();
        if parts.next() == Some(key) {
            return parts.next().ok_or_else(|| format!("{key} missing"))?.parse().map_err(|_| format!("{key} not a number"));
        }
    }
    Err(format!("{key} not in meta.txt"))
}

fn load(dir: &Path, name: &str) -> Result<Vec<f64>, String> {
    let bytes = fs::read(dir.join(name)).map_err(|e| format!("{name}: {e}"))?;
    npy::parse(&bytes).and_then(|a| a.values()).map_err(|e| format!("{name}: {e}"))
}
