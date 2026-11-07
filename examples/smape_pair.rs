//! Symmetric mean absolute percentage error of two little-endian `f64` arrays.
//!
//! The arrays are the predictions and targets a caller already computed. This example does not
//! train a model and it does not read a dataset. `tools/neurobench_mackey.py` points it at the
//! held-out arrays from the pinned `NeuroBench` harness.

use std::env;
use std::fs;
use std::process::ExitCode;

use ferromorphic::metrics::smape;

fn read_f64(path: &str) -> Result<Vec<f64>, String> {
    let bytes = fs::read(path).map_err(|err| format!("{path}: {err}"))?;
    if bytes.len() % 8 != 0 {
        return Err(format!("{path} has {} bytes, not a whole number of f64", bytes.len()));
    }
    let mut out = Vec::with_capacity(bytes.len() / 8);
    let mut offset = 0;
    while offset < bytes.len() {
        let mut buf = [0u8; 8];
        buf.copy_from_slice(&bytes[offset..offset + 8]);
        offset += 8;
        let value = f64::from_le_bytes(buf);
        if !value.is_finite() {
            return Err(format!("{path} holds a non-finite value"));
        }
        out.push(value);
    }
    Ok(out)
}

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    let Some(pred_path) = args.next() else {
        eprintln!("usage: smape_pair PRED.bin LABEL.bin");
        return ExitCode::from(2);
    };
    let Some(label_path) = args.next() else {
        eprintln!("usage: smape_pair PRED.bin LABEL.bin");
        return ExitCode::from(2);
    };
    let predicted = match read_f64(&pred_path) {
        Ok(values) => values,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let truth = match read_f64(&label_path) {
        Ok(values) => values,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    match smape(&predicted, &truth) {
        Ok(value) => {
            println!("{value:.17}");
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("{err}");
            ExitCode::from(1)
        }
    }
}
