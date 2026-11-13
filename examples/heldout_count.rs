//! Fixed-count held-out inference.
//!
//! One process runs `--count` steps and exits. It does not read a power counter and it does not
//! repeat until a wall-clock window elapses. The baseline arm reads the same tables and writes
//! the constant control value. That arm is an attribution control, not a forecast.
//!
//! ```text
//! heldout_count --arm activity|baseline --task reaching|mackey-teacher|mackey-auto|mackey-replay \
//!   --weights FILE --inputs FILE --reference FILE --count N [--claim-quality]
//! ```

use std::env;
use std::fs;
use std::process::ExitCode;

use ferromorphic::heldout::{execute, receipt_text, Arm, HeldoutError, Request, Task};

fn main() -> ExitCode {
    match run() {
        Ok(text) => {
            print!("{text}");
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("{err}");
            ExitCode::from(2)
        }
    }
}

fn run() -> Result<String, String> {
    let mut args = env::args().skip(1);
    let mut arm = None;
    let mut task = None;
    let mut weights = None;
    let mut inputs = None;
    let mut reference = None;
    let mut count = None;
    let mut claim_quality = false;
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--claim-quality" => claim_quality = true,
            "--arm" | "--task" | "--weights" | "--inputs" | "--reference" | "--count" => {
                let value = args.next().ok_or_else(|| format!("{flag} needs a value"))?;
                match flag.as_str() {
                    "--arm" => arm = Some(value),
                    "--task" => task = Some(value),
                    "--weights" => weights = Some(value),
                    "--inputs" => inputs = Some(value),
                    "--reference" => reference = Some(value),
                    "--count" => count = Some(value),
                    _ => {}
                }
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    let arm = parse_arm(&arm.ok_or("--arm is required")?)?;
    let task = parse_task(&task.ok_or("--task is required")?)?;
    let count: usize = count.ok_or("--count is required")?.parse().map_err(|_| "--count is not an integer")?;
    let weights = fs::read(weights.ok_or("--weights is required")?).map_err(|err| err.to_string())?;
    let inputs = fs::read(inputs.ok_or("--inputs is required")?).map_err(|err| err.to_string())?;
    let reference = fs::read(reference.ok_or("--reference is required")?).map_err(|err| err.to_string())?;
    match execute(&Request { arm, task, count, weights: &weights, inputs: &inputs, reference: &reference, claim_quality }) {
        Ok(receipt) => Ok(receipt_text(&receipt)),
        Err(HeldoutError::Quality { max_abs, limit }) => {
            Err(format!("activity residual {max_abs} exceeds the gate {limit}"))
        }
        Err(err) => Err(err.to_string()),
    }
}

fn parse_arm(text: &str) -> Result<Arm, String> {
    match text {
        "activity" => Ok(Arm::Activity),
        "baseline" => Ok(Arm::Baseline),
        other => Err(format!("unknown arm {other}")),
    }
}

fn parse_task(text: &str) -> Result<Task, String> {
    match text {
        "reaching" => Ok(Task::Reaching),
        "mackey-teacher" => Ok(Task::MackeyTeacher),
        "mackey-auto" => Ok(Task::MackeyAuto),
        "mackey-replay" => Ok(Task::MackeyReplay),
        other => Err(format!("unknown task {other}")),
    }
}
