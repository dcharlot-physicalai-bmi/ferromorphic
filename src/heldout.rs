//! Fixed-count held-out inference, and a same-I/O control that does not forecast.
//!
//! One call performs `work_count` steps and returns. It does not read a power counter, it does
//! not integrate watts, and it does not repeat until a wall-clock window is long enough. A meter
//! that wants a longer window repeats this call. Package energy and core energy are not fields
//! here.
//!
//! The baseline arm reads the same weight table and the same input series as the activity arm,
//! and it folds every loaded byte through [`std::hint::black_box`] into `touch`. Its outputs are
//! the constant [`CONTROL_OUTPUT`]. That arm is an attribution control. It is not a second model
//! and it does not satisfy the task quality gate.

use crate::metrics::{r_squared, MetricError};
use crate::reservoir::{
    parse_ann_table, parse_echo_table, parse_f32_series, parse_f64_series, sha256, ReservoirError,
    SuppliedEcho, TableError,
};

/// Largest Mackey series this runner accepts. The pinned protocol uses 750.
pub const MACKEY_LEN_MAX: usize = 4096;
/// Largest reaching frame buffer, in `f32` values.
pub const REACH_FRAME_MAX: usize = 40_000_000;
/// Largest reaching prediction buffer, in `f32` values.
pub const REACH_PRED_MAX: usize = 400_000;
/// Absolute gate for a reaching activity arm, `f32`.
pub const REACH_MAX_ABS: f32 = 1e-5;
/// Absolute gate for teacher-forced or replayed Mackey, `f64`.
pub const TEACHER_MAX_ABS: f64 = 1e-12;
/// Index reported when an autonomous residual crosses this absolute gap.
pub const AUTO_SPLIT_ABS: f64 = 1e-8;
/// Finite value written by the baseline arm. It is not a forecast.
pub const CONTROL_OUTPUT: f64 = -1.0;

/// Which arm ran.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arm {
    /// The reservoir or the window stack.
    Activity,
    /// Same bytes in, constant control out. Not a forecast.
    Baseline,
}

/// Which pinned task the tables belong to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Task {
    /// `WindowAnn` on reaching frames. Reference is the upstream `f32` prediction.
    Reaching,
    /// Teacher-forced echo. Reference is the upstream teacher-forced series.
    MackeyTeacher,
    /// Autonomous echo. The residual is reported. Pointwise parity is not the gate.
    MackeyAuto,
    /// Drive the echo with the supplied inputs. Reference is the upstream output of those inputs.
    MackeyReplay,
}

/// Why a held-out call was refused.
#[derive(Debug)]
pub enum HeldoutError {
    /// The table, series, or hash check failed.
    Table(TableError),
    /// `count` is not the length of the input, or not the length of the reference.
    Count {
        /// What was counted.
        what: &'static str,
        /// Length found.
        got: usize,
        /// Length required.
        want: usize,
    },
    /// The baseline arm was asked to wear a quality pass.
    BaselineIsControl,
    /// An activity arm that has a quality gate missed it.
    Quality {
        /// Absolute residual.
        max_abs: f64,
        /// The gate.
        limit: f64,
    },
    /// A metric could not be computed.
    Metric(MetricError),
    /// The model step refused the series.
    Model,
}

impl core::fmt::Display for HeldoutError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Table(err) => write!(f, "{err}"),
            Self::Count { what, got, want } => {
                write!(f, "{what} has length {got} and the fixed count is {want}")
            }
            Self::BaselineIsControl => {
                f.write_str("the baseline arm is an attribution control and cannot claim task quality")
            }
            Self::Quality { max_abs, limit } => {
                write!(f, "activity residual {max_abs} exceeds the gate {limit}")
            }
            Self::Metric(err) => write!(f, "{err}"),
            Self::Model => f.write_str("the model refused the series"),
        }
    }
}

impl std::error::Error for HeldoutError {}

impl From<TableError> for HeldoutError {
    fn from(err: TableError) -> Self {
        Self::Table(err)
    }
}

impl From<MetricError> for HeldoutError {
    fn from(err: MetricError) -> Self {
        Self::Metric(err)
    }
}

/// One fixed-count run.
#[derive(Clone, Debug, PartialEq)]
pub struct Receipt {
    /// `activity` or `baseline`.
    pub arm: &'static str,
    /// Task token.
    pub task: &'static str,
    /// `inference` or `attribution-control`.
    pub role: &'static str,
    /// Whether this arm is a task-quality result.
    pub matched_quality: bool,
    /// Steps executed. Equal to the requested count.
    pub work_count: usize,
    /// Always 1. A meter repeats the process; this call does not.
    pub repetitions: u32,
    /// Max `|output - reference|`.
    pub max_abs: f64,
    /// First autonomous index whose absolute gap exceeds [`AUTO_SPLIT_ABS`], if any.
    pub parity_split: Option<usize>,
    /// SHA-256 of the weight-table bytes.
    pub weights_sha256: String,
    /// SHA-256 of the input-series bytes.
    pub inputs_sha256: String,
    /// SHA-256 of the reference-series bytes.
    pub reference_sha256: String,
    /// SHA-256 of the little-endian output payload.
    pub output_sha256: String,
    /// Mix of every loaded byte. Present so the baseline's reads cannot be deleted.
    pub touch_sha256: String,
    /// Column-mean `R²` for reaching, absent for Mackey.
    pub r2: Option<f64>,
}

/// Inputs of one call. The three buffers are the file bytes, already read.
pub struct Request<'a> {
    /// Activity or baseline.
    pub arm: Arm,
    /// Which tables these bytes are.
    pub task: Task,
    /// Required length. A mismatch is a refusal, not a shorter run.
    pub count: usize,
    /// Weight table.
    pub weights: &'a [u8],
    /// Input series table.
    pub inputs: &'a [u8],
    /// Reference series table.
    pub reference: &'a [u8],
    /// When true on a baseline arm, the call is refused.
    pub claim_quality: bool,
}

struct Produced {
    max_abs: f64,
    split: Option<usize>,
    r2: Option<f64>,
    output: Vec<u8>,
    touch: u64,
}

/// Run one fixed count.
///
/// # Errors
///
/// [`HeldoutError`] when a table is refused, the count does not match, the baseline is asked
/// to claim quality, or an activity arm with a numeric gate misses that gate.
pub fn execute(req: &Request<'_>) -> Result<Receipt, HeldoutError> {
    if req.arm == Arm::Baseline && req.claim_quality {
        return Err(HeldoutError::BaselineIsControl);
    }
    let weights_sha256 = hex(&sha256(req.weights));
    let inputs_sha256 = hex(&sha256(req.inputs));
    let reference_sha256 = hex(&sha256(req.reference));
    let produced = match req.task {
        Task::Reaching => run_reaching(req)?,
        Task::MackeyTeacher => run_mackey(req, MackeyMode::Teacher)?,
        Task::MackeyAuto => run_mackey(req, MackeyMode::Auto)?,
        Task::MackeyReplay => run_mackey(req, MackeyMode::Replay)?,
    };
    let max_abs = produced.max_abs;
    let matched_quality = match (req.arm, req.task) {
        (Arm::Baseline, _) => false,
        (Arm::Activity, Task::MackeyAuto) => false,
        (Arm::Activity, Task::Reaching) => max_abs <= f64::from(REACH_MAX_ABS),
        (Arm::Activity, Task::MackeyTeacher | Task::MackeyReplay) => max_abs <= TEACHER_MAX_ABS,
    };
    if req.arm == Arm::Activity
        && matches!(req.task, Task::Reaching | Task::MackeyTeacher | Task::MackeyReplay)
        && !matched_quality
    {
        let limit = if req.task == Task::Reaching { f64::from(REACH_MAX_ABS) } else { TEACHER_MAX_ABS };
        return Err(HeldoutError::Quality { max_abs, limit });
    }
    Ok(Receipt {
        arm: match req.arm {
            Arm::Activity => "activity",
            Arm::Baseline => "baseline",
        },
        task: match req.task {
            Task::Reaching => "reaching",
            Task::MackeyTeacher => "mackey-teacher",
            Task::MackeyAuto => "mackey-auto",
            Task::MackeyReplay => "mackey-replay",
        },
        role: match req.arm {
            Arm::Activity => "inference",
            Arm::Baseline => "attribution-control",
        },
        matched_quality,
        work_count: req.count,
        repetitions: 1,
        max_abs,
        parity_split: produced.split,
        weights_sha256,
        inputs_sha256,
        reference_sha256,
        output_sha256: hex(&sha256(&produced.output)),
        touch_sha256: hex(&produced.touch.to_le_bytes()),
        r2: produced.r2,
    })
}

enum MackeyMode {
    Teacher,
    Auto,
    Replay,
}

fn run_reaching(req: &Request<'_>) -> Result<Produced, HeldoutError> {
    let mut net = parse_ann_table(req.weights)?;
    let frames = parse_f32_series(req.inputs, "reaching-frames", REACH_FRAME_MAX)?;
    let reference = parse_f32_series(req.reference, "reaching-pred", REACH_PRED_MAX)?;
    let n = frames.len() / net.input_dim;
    if n != req.count {
        return Err(HeldoutError::Count { what: "frames", got: n, want: req.count });
    }
    if reference.len() != req.count * net.output_dim {
        return Err(HeldoutError::Count {
            what: "reference",
            got: reference.len() / net.output_dim.max(1),
            want: req.count,
        });
    }
    let touch = touch_bytes(req.weights, req.inputs);
    let pred = if req.arm == Arm::Activity {
        net.forward(&frames).map_err(model_err)?
    } else {
        let _ = std::hint::black_box(net.fc1_w.len());
        vec![CONTROL_OUTPUT as f32; reference.len()]
    };
    if pred.len() != reference.len() {
        return Err(HeldoutError::Count { what: "output", got: pred.len(), want: reference.len() });
    }
    let mut max_abs = 0.0f32;
    for (p, r) in pred.iter().zip(&reference) {
        max_abs = max_abs.max((p - r).abs());
    }
    let cols = net.output_dim;
    let mut score = 0.0;
    for col in 0..cols {
        let p: Vec<f64> = (0..n).map(|i| f64::from(pred[i * cols + col])).collect();
        let t: Vec<f64> = (0..n).map(|i| f64::from(reference[i * cols + col])).collect();
        score += r_squared(&p, &t)?;
    }
    score /= cols as f64;
    let mut raw = Vec::with_capacity(pred.len() * 4);
    for value in &pred {
        raw.extend_from_slice(&value.to_le_bytes());
    }
    Ok(Produced { max_abs: f64::from(max_abs), split: None, r2: Some(score), output: raw, touch })
}

fn run_mackey(req: &Request<'_>, mode: MackeyMode) -> Result<Produced, HeldoutError> {
    let mut echo = parse_echo_table(req.weights)?;
    let (drive_role, ref_role) = match mode {
        MackeyMode::Teacher => ("mackey-drive", "mackey-teacher"),
        MackeyMode::Auto => ("mackey-drive", "mackey-pred"),
        MackeyMode::Replay => ("mackey-replay-input", "mackey-replay-output"),
    };
    let drive = parse_f64_series(req.inputs, drive_role, MACKEY_LEN_MAX)?;
    let reference = parse_f64_series(req.reference, ref_role, MACKEY_LEN_MAX)?;
    if drive.len() != req.count {
        return Err(HeldoutError::Count { what: "drive", got: drive.len(), want: req.count });
    }
    if reference.len() != req.count {
        return Err(HeldoutError::Count { what: "reference", got: reference.len(), want: req.count });
    }
    let touch = touch_echo(&echo).wrapping_add(touch_bytes(req.weights, req.inputs));
    let pred = if req.arm == Arm::Activity {
        match mode {
            MackeyMode::Teacher | MackeyMode::Replay => echo.teacher(&drive).map_err(model_err)?,
            MackeyMode::Auto => echo.autonomous(&drive).map_err(model_err)?,
        }
    } else {
        let _ = std::hint::black_box(touch_echo(&echo));
        vec![CONTROL_OUTPUT; req.count]
    };
    let mut max_abs = 0.0f64;
    let mut split = None;
    for (i, (p, r)) in pred.iter().zip(&reference).enumerate() {
        let gap = (p - r).abs();
        if gap > max_abs {
            max_abs = gap;
        }
        if split.is_none() && gap > AUTO_SPLIT_ABS {
            split = Some(i);
        }
    }
    let mut raw = Vec::with_capacity(pred.len() * 8);
    for value in &pred {
        raw.extend_from_slice(&value.to_le_bytes());
    }
    Ok(Produced { max_abs, split, r2: None, output: raw, touch })
}

fn touch_echo(echo: &SuppliedEcho) -> u64 {
    let mut acc = 0u64;
    for (i, w) in echo.w_in.iter().chain(&echo.w).chain(&echo.w_out).chain(&echo.state).enumerate() {
        acc = acc.wrapping_add(std::hint::black_box(w).to_bits().wrapping_mul(i as u64 + 1));
    }
    acc
}

fn touch_bytes(weights: &[u8], inputs: &[u8]) -> u64 {
    let mut acc = 0u64;
    for (i, byte) in weights.iter().chain(inputs).enumerate() {
        acc = acc.wrapping_add(u64::from(std::hint::black_box(*byte)).wrapping_mul(i as u64 + 1));
    }
    acc
}

fn model_err(err: ReservoirError) -> HeldoutError {
    let _ = err;
    HeldoutError::Model
}

/// Machine-readable receipt. The field order is part of the text.
#[must_use]
pub fn receipt_text(receipt: &Receipt) -> String {
    let split = match receipt.parity_split {
        Some(i) => i.to_string(),
        None => "none".to_string(),
    };
    let r2 = match receipt.r2 {
        Some(v) => format!("{v:.16e}"),
        None => "none".to_string(),
    };
    format!(
        "ferromorphic-heldout 1\narm {arm}\ntask {task}\nrole {role}\nmatched_quality {matched}\nwork_count {work}\nrepetitions {reps}\nmax_abs {max:.16e}\nparity_split {split}\nr2 {r2}\nweights_sha256 {weights}\ninputs_sha256 {inputs}\nreference_sha256 {reference}\noutput_sha256 {output}\ntouch_sha256 {touch}\nmeter none\njoules none\n",
        arm = receipt.arm,
        task = receipt.task,
        role = receipt.role,
        matched = receipt.matched_quality,
        work = receipt.work_count,
        reps = receipt.repetitions,
        max = receipt.max_abs,
        weights = receipt.weights_sha256,
        inputs = receipt.inputs_sha256,
        reference = receipt.reference_sha256,
        output = receipt.output_sha256,
        touch = receipt.touch_sha256,
    )
}

fn hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reservoir::{
        ann_table_bytes, echo_table_bytes, series_f32_bytes, series_f64_bytes, SuppliedEcho, WindowAnn,
    };

    fn tiny_echo() -> SuppliedEcho {
        SuppliedEcho::try_new(
            1, 2, 1, vec![0.5, -0.25], vec![0.2], vec![0.0, 1.0, -1.0], 0.5, vec![0.1], true, true,
        )
        .unwrap()
    }

    #[test]
    fn activity_teacher_matches_and_the_baseline_is_not_quality() {
        let echo = tiny_echo();
        let mut teacher_net = echo.clone();
        let drive = [0.4, 0.4, 0.4];
        let reference = teacher_net.teacher(&drive).unwrap();
        let weights = echo_table_bytes(&echo);
        let inputs = series_f64_bytes("mackey-drive", &drive);
        let truth = series_f64_bytes("mackey-teacher", &reference);
        let got = execute(&Request {
            arm: Arm::Activity,
            task: Task::MackeyTeacher,
            count: 3,
            weights: &weights,
            inputs: &inputs,
            reference: &truth,
            claim_quality: false,
        })
        .unwrap();
        assert!(got.matched_quality);
        assert_eq!(got.work_count, 3);
        assert_eq!(got.repetitions, 1);
        assert_eq!(got.role, "inference");
        assert!(got.max_abs <= TEACHER_MAX_ABS);
        assert!(receipt_text(&got).contains("meter none\njoules none\n"));
        let base = execute(&Request {
            arm: Arm::Baseline,
            task: Task::MackeyTeacher,
            count: 3,
            weights: &weights,
            inputs: &inputs,
            reference: &truth,
            claim_quality: false,
        })
        .unwrap();
        assert!(!base.matched_quality);
        assert_eq!(base.role, "attribution-control");
        assert_eq!(base.work_count, 3);
        assert_ne!(base.output_sha256, got.output_sha256);
        assert!(base.max_abs > 0.1);
        assert!(execute(&Request {
            arm: Arm::Baseline,
            task: Task::MackeyTeacher,
            count: 3,
            weights: &weights,
            inputs: &inputs,
            reference: &truth,
            claim_quality: true,
        })
        .is_err());
        let mut bad = weights.clone();
        bad[0] = b'x';
        assert!(execute(&Request {
            arm: Arm::Activity,
            task: Task::MackeyTeacher,
            count: 3,
            weights: &bad,
            inputs: &inputs,
            reference: &truth,
            claim_quality: false,
        })
        .is_err());
        assert!(execute(&Request {
            arm: Arm::Activity,
            task: Task::MackeyTeacher,
            count: 2,
            weights: &weights,
            inputs: &inputs,
            reference: &truth,
            claim_quality: false,
        })
        .is_err());
    }

    #[test]
    fn a_wrong_reaching_window_misses_the_quality_gate() {
        let ann = WindowAnn::try_new(
            1, 1, 1, 1, 2, vec![1.0], vec![0.0], vec![1.0], vec![0.0], vec![0.0], vec![1.0],
            vec![1.0], vec![0.0], vec![1.0], vec![0.0], vec![0.0], vec![1.0], vec![1.0], vec![0.0],
            1e-5,
        )
        .unwrap();
        let mut net = ann.clone();
        let pred = net.forward(&[1.0, 2.0, 3.0]).unwrap();
        let weights = ann_table_bytes(&ann);
        let inputs = series_f32_bytes("reaching-frames", &[1.0, 2.0, 3.0]);
        let reference = series_f32_bytes("reaching-pred", &pred);
        let got = execute(&Request {
            arm: Arm::Activity,
            task: Task::Reaching,
            count: 3,
            weights: &weights,
            inputs: &inputs,
            reference: &reference,
            claim_quality: false,
        })
        .unwrap();
        assert!(got.matched_quality);
        assert_eq!(got.work_count, 3);
        let mut shifted = ann;
        shifted.window = 49;
        let wrong = shifted.forward(&[1.0, 2.0, 3.0]).unwrap();
        assert_ne!(wrong, pred, "window 49 is not the reset window of 2");
        let mut bad = pred.clone();
        bad[2] += 1.0;
        let wrong_ref = series_f32_bytes("reaching-pred", &bad);
        assert!(matches!(
            execute(&Request {
                arm: Arm::Activity,
                task: Task::Reaching,
                count: 3,
                weights: &weights,
                inputs: &inputs,
                reference: &wrong_ref,
                claim_quality: false,
            }),
            Err(HeldoutError::Quality { .. })
        ));
    }
}
