//! Two-column coefficient of determination.
//!
//! Each file is row-major `f64` pairs `(x, y)`, the layout of a 2-output prediction.
//! The value printed is the mean of [`ferromorphic::metrics::r_squared`] on each column.
//! That is the arithmetic in `NeuroBench` `R2.compute` at commit
//! `e521c28f5798cb40bb9423758bf066dad53fe070`: population variance, then the average of the
//! two axes. A constant column is an error here. The harness can return a non-finite value
//! for that column.

use std::env;
use std::fs;
use std::process::ExitCode;

use ferromorphic::metrics::r_squared;

fn read_pairs(path: &str) -> Result<(Vec<f64>, Vec<f64>), String> {
    let bytes = fs::read(path).map_err(|err| format!("{path}: {err}"))?;
    if bytes.len() % 16 != 0 {
        return Err(format!("{path} is not a sequence of f64 pairs"));
    }
    let mut x = Vec::new();
    let mut y = Vec::new();
    let mut offset = 0;
    while offset < bytes.len() {
        let mut a = [0u8; 8];
        let mut b = [0u8; 8];
        a.copy_from_slice(&bytes[offset..offset + 8]);
        b.copy_from_slice(&bytes[offset + 8..offset + 16]);
        offset += 16;
        let (xv, yv) = (f64::from_le_bytes(a), f64::from_le_bytes(b));
        if !xv.is_finite() || !yv.is_finite() {
            return Err(format!("{path} holds a non-finite value"));
        }
        x.push(xv);
        y.push(yv);
    }
    Ok((x, y))
}

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    let (Some(pred_path), Some(label_path)) = (args.next(), args.next()) else {
        eprintln!("usage: r2_columns PRED.bin LABEL.bin");
        return ExitCode::from(2);
    };
    let (px, py) = match read_pairs(&pred_path) {
        Ok(values) => values,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let (lx, ly) = match read_pairs(&label_path) {
        Ok(values) => values,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let x = match r_squared(&px, &lx) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("x: {err}");
            return ExitCode::from(1);
        }
    };
    let y = match r_squared(&py, &ly) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("y: {err}");
            return ExitCode::from(1);
        }
    };
    println!("{:.17}", (x + y) / 2.0);
    ExitCode::SUCCESS
}
