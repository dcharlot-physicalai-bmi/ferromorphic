//! Held-out inference through [`ferromorphic::reservoir::SuppliedEcho`] and
//! [`ferromorphic::reservoir::WindowAnn`].
//!
//! The tables are written by `tools/export_heldout_tables.py` from `NeuroBench`
//! `e521c28` and torch `2.7.1+cpu`. Training labels are not read here. Metric
//! replay (`examples/smape_pair`, `examples/r2_columns`) is a different check:
//! it scores bytes the harness already produced. This test runs the recurrence.

use std::fs;
use std::path::Path;
use std::time::Instant;

use ferromorphic::metrics::{r_squared, smape};
use ferromorphic::reservoir::{parse_ann_table, parse_echo_table, parse_f32_series, parse_f64_series};

/// One Mackey step against the upstream `f64` gemm. A wrong matrix fails this.
/// The full autonomous horizon is a chaotic recurrence and is reported separately.
const ECHO_STEP_MAX_ABS: f64 = 1e-12;
/// Teacher forcing drives every sample with the observation, so the error does
/// not feed back. This is the check of the recurrence itself.
const ECHO_TEACHER_MAX_ABS: f64 = 1e-12;
/// Reaching `f32` stack. The measured maximum on the six sessions is `1.34e-6`.
const ANN_MAX_ABS: f32 = 1e-5;
const MACKEY_SERIES_MAX: usize = 4096;
const REACH_FRAME_MAX: usize = 40_000_000;
const REACH_PRED_MAX: usize = 400_000;

fn max_abs_f64(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(p, q)| (p - q).abs()).fold(0.0, f64::max)
}

fn max_abs_f32(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(p, q)| (p - q).abs()).fold(0.0, f32::max)
}

fn sha256_file(path: &Path) -> String {
    ferromorphic::reservoir::sha256(&fs::read(path).unwrap())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

#[test]
fn held_out_echo_and_reaching_match_the_pinned_upstream_within_the_declared_tolerance() {
    let root = Path::new("target/verification/heldout_tables");
    let text = fs::read_to_string(root.join("manifest.json"))
        .expect("run tools/export_heldout_tables.py before this test");
    let manifest = parse_manifest(&text);
    assert_eq!(manifest.torch, "2.7.1+cpu");
    assert!(!manifest.fit_on_held_out);
    let mut report = Vec::new();
    let mut failures = Vec::new();
    for tau in &manifest.mackey {
        let echo_path = root.join(format!("tau{}.echo", tau.tau));
        let echo_sha = sha256_file(&echo_path);
        assert_eq!(echo_sha, tau.echo_sha);
        let echo = parse_echo_table(&fs::read(&echo_path).unwrap()).unwrap();
        let drive = parse_f64_series(
            &fs::read(root.join(format!("tau{}.drive", tau.tau))).unwrap(),
            "mackey-drive",
            MACKEY_SERIES_MAX,
        )
        .unwrap();
        let auto = parse_f64_series(
            &fs::read(root.join(format!("tau{}.auto", tau.tau))).unwrap(),
            "mackey-pred",
            MACKEY_SERIES_MAX,
        )
        .unwrap();
        let teacher = parse_f64_series(
            &fs::read(root.join(format!("tau{}.teacher", tau.tau))).unwrap(),
            "mackey-teacher",
            MACKEY_SERIES_MAX,
        )
        .unwrap();
        let label = parse_f64_series(
            &fs::read(root.join(format!("tau{}.label", tau.tau))).unwrap(),
            "mackey-label",
            MACKEY_SERIES_MAX,
        )
        .unwrap();
        assert_eq!(drive.len(), 750);
        assert_eq!(echo.state.len(), tau.units);
        let started = Instant::now();
        let mut autonomous_net = echo.clone();
        let got = autonomous_net.autonomous(&drive).unwrap();
        let mut forced = echo.clone();
        let got_teacher = forced.teacher(&drive).unwrap();
        let mut zeros = echo.clone();
        zeros.state.fill(0.0);
        let from_zero = zeros.autonomous(&drive).unwrap();
        let elapsed_ms = started.elapsed().as_secs_f64() * 1e3;
        let auto_err = max_abs_f64(&got, &auto);
        let step0 = (got[0] - auto[0]).abs();
        let teacher_err = max_abs_f64(&got_teacher, &teacher);
        let modes = max_abs_f64(&got, &got_teacher);
        let zero_err = max_abs_f64(&from_zero, &auto);
        let split_at = got.iter().zip(&auto).position(|(p, q)| (p - q).abs() > 1e-8);
        if step0 > ECHO_STEP_MAX_ABS {
            failures.push(format!("tau {} first step {step0}", tau.tau));
        }
        if teacher_err > ECHO_TEACHER_MAX_ABS {
            failures.push(format!("tau {} teacher {teacher_err}", tau.tau));
        }
        if modes <= 1e-3 {
            failures.push(format!("tau {} modes agreed", tau.tau));
        }
        if zero_err <= 1e-3 {
            failures.push(format!("tau {} zero state agreed", tau.tau));
        }
        let rust_smape = smape(&got, &label).unwrap();
        let upstream_smape = smape(&auto, &label).unwrap();
        report.push(format!(
            "tau {} step0 {step0:.3e} teacher_max_abs {teacher_err:.3e} auto_max_abs {auto_err:.3e} split_after_1e-8 {split_at:?} mode_gap {modes:.3e} zero_gap {zero_err:.3e} rust_smape {rust_smape:.16} upstream_smape {upstream_smape:.16} ms {elapsed_ms:.1} echo_sha {echo_sha}",
            tau.tau
        ));
    }
    for session in &manifest.reaching {
        let ann_path = root.join(format!("{}.ann", session.stem));
        let ann_sha = sha256_file(&ann_path);
        assert_eq!(ann_sha, session.ann_sha);
        let ann = parse_ann_table(&fs::read(&ann_path).unwrap()).unwrap();
        assert_eq!(ann.window, 50);
        let frames = parse_f32_series(
            &fs::read(root.join(format!("{}.frames", session.stem))).unwrap(),
            "reaching-frames",
            REACH_FRAME_MAX,
        )
        .unwrap();
        let pred = parse_f32_series(
            &fs::read(root.join(format!("{}.pred", session.stem))).unwrap(),
            "reaching-pred",
            REACH_PRED_MAX,
        )
        .unwrap();
        let label = parse_f32_series(
            &fs::read(root.join(format!("{}.label", session.stem))).unwrap(),
            "reaching-label",
            REACH_PRED_MAX,
        )
        .unwrap();
        let started = Instant::now();
        let mut net = ann.clone();
        let got = net.forward(&frames).unwrap();
        let mut shifted = ann.clone();
        shifted.window = 49;
        let wrong_window = shifted.forward(&frames).unwrap();
        let mut bare = ann.clone();
        bare.discard_buffer();
        let no_zero = bare.forward(&frames).unwrap();
        let elapsed_ms = started.elapsed().as_secs_f64() * 1e3;
        let err = max_abs_f32(&got, &pred);
        let err_at = got.iter().zip(&pred).enumerate().max_by(|p, q| {
            (p.1.0 - p.1.1).abs().partial_cmp(&(q.1.0 - q.1.1).abs()).unwrap()
        }).map(|(i, _)| i).unwrap();
        if err > ANN_MAX_ABS {
            failures.push(format!("{} reaching {err} at {err_at}", session.stem));
        }
        if max_abs_f32(&wrong_window, &pred) <= 1e-3 {
            failures.push(format!("{} window 49 matched", session.stem));
        }
        if max_abs_f32(&no_zero, &pred) <= 1e-3 {
            failures.push(format!("{} empty buffer matched", session.stem));
        }
        let cols = 2;
        let rows = got.len() / cols;
        let mut rust_r = 0.0;
        let mut upstream_r = 0.0;
        for col in 0..cols {
            let pred_c: Vec<f64> = (0..rows).map(|i| f64::from(got[i * cols + col])).collect();
            let up_c: Vec<f64> = (0..rows).map(|i| f64::from(pred[i * cols + col])).collect();
            let lab_c: Vec<f64> = (0..rows).map(|i| f64::from(label[i * cols + col])).collect();
            rust_r += r_squared(&pred_c, &lab_c).unwrap();
            upstream_r += r_squared(&up_c, &lab_c).unwrap();
        }
        rust_r /= cols as f64;
        upstream_r /= cols as f64;
        report.push(format!(
            "{} max_abs {err:.3e} rust_r2 {rust_r:.16} upstream_r2 {upstream_r:.16} n {rows} ms {elapsed_ms:.1} ann_sha {ann_sha} checkpoint_sha {}",
            session.stem, session.checkpoint_sha
        ));
    }
    let body = report.join("\n") + "\n" + &failures.join("\n");
    fs::write(root.join("rust_report.txt"), &body).unwrap();
    println!("{body}");
    assert!(failures.is_empty(), "{}", failures.join("; "));
}

struct Tau {
    tau: u32,
    units: usize,
    echo_sha: String,
}

struct Session {
    stem: String,
    ann_sha: String,
    checkpoint_sha: String,
}

struct Manifest {
    torch: String,
    fit_on_held_out: bool,
    mackey: Vec<Tau>,
    reaching: Vec<Session>,
}

/// The manifest is small and flat enough to read without a JSON crate.
fn parse_manifest(text: &str) -> Manifest {
    let torch = field(text, "torch");
    let states_fit_was_not_on_held_out = text.contains("\"fit_on_held_out\": false");
    let mut mackey = Vec::new();
    for tau in 17..=30 {
        let echo_sha = sha_after(text, &format!("tau{tau}.echo"));
        let units = usize_after_tau(text, tau);
        mackey.push(Tau { tau, units, echo_sha });
    }
    let mut reaching = Vec::new();
    for stem in [
        "indy_20160622_01",
        "indy_20160630_01",
        "indy_20170131_02",
        "loco_20170210_03",
        "loco_20170215_02",
        "loco_20170301_05",
    ] {
        reaching.push(Session {
            ann_sha: sha_after(text, &format!("{stem}.ann")),
            checkpoint_sha: checkpoint_before_ann(text, stem),
            stem: stem.to_string(),
        });
    }
    Manifest { torch, fit_on_held_out: !states_fit_was_not_on_held_out, mackey, reaching }
}

fn field(text: &str, key: &str) -> String {
    let pat = format!("\"{key}\": \"");
    let start = text.find(&pat).unwrap() + pat.len();
    let end = text[start..].find('"').unwrap();
    text[start..start + end].to_string()
}

fn sha_after(text: &str, filename: &str) -> String {
    let at = text.find(filename).unwrap();
    let slice = &text[at..];
    let key = "\"sha256\": \"";
    let start = slice.find(key).unwrap() + key.len();
    slice[start..start + 64].to_string()
}

fn usize_after_tau(text: &str, tau: u32) -> usize {
    let pat = format!("\"tau\": {tau}");
    let at = text.find(&pat).unwrap();
    let slice = &text[at..];
    let key = "\"reservoir_size\": ";
    let start = slice.find(key).unwrap() + key.len();
    let end = slice[start..].find([',', '\n']).unwrap();
    slice[start..start + end].trim().parse().unwrap()
}

fn checkpoint_before_ann(text: &str, stem: &str) -> String {
    let file = format!("\"file\": \"{stem}.mat\"");
    let at = text.find(&file).unwrap();
    let slice = &text[at..];
    let key = "\"checkpoint_sha256\": \"";
    let start = slice.find(key).unwrap() + key.len();
    slice[start..start + 64].to_string()
}
