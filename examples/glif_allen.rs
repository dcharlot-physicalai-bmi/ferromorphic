//! Run Allen Cell Types Database GLIF models from their served `neuron_config.json` files.
//!
//! ```text
//! cargo run --release --example glif_allen -- path/to/configs   # a directory of *.json
//! cargo run --release --example glif_allen -- one.json two.json
//! ```
//!
//! Each model gets 50 ms of nothing, 200 ms at twice its rheobase, 50 ms of nothing, at its own `dt`,
//! and one tab-separated line: `id  level  spikes  steps  cut  sum_v  sum_abs_v` — or
//! `id  level  ERROR  message` for a config this crate refuses. `tools/glif_sweep.py` produces the
//! same lines from the `AllenSDK` and compares the two.

use std::{fs, path::PathBuf};

use ferromorphic::{
    glif::Glif,
    json::{self, Json},
};

/// Neumaier's compensated sum, so a comparison with Python's exact `math.fsum` measures the traces.
fn sum(xs: impl Iterator<Item = f64>) -> f64 {
    let (mut s, mut c) = (0.0_f64, 0.0_f64);
    for x in xs {
        let t = s + x;
        c += if s.abs() >= x.abs() { (s - t) + x } else { (x - t) + s };
        s = t;
    }
    s + c
}

fn line(path: &PathBuf) -> String {
    let id = path.file_stem().and_then(|s| s.to_str()).unwrap_or("?").to_owned();
    let text = match fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) => return format!("{id}\t?\tERROR\t{e}"),
    };
    let model = match Glif::from_neuron_config(&text) {
        Ok(m) => m,
        Err(e) => return format!("{id}\t?\tERROR\t{e}"),
    };
    let level = model.level().map_or_else(|| "?".to_owned(), |l| l.to_string());
    let r_input = json::parse(&text).ok().and_then(|c| c.get("R_input").and_then(Json::as_f64)).unwrap_or(f64::NAN);
    let mut stim = vec![0.0; 6000];
    stim[1000..5000].fill(2.0 * (model.th_inf / r_input));
    match model.run(&stim) {
        Err(e) => format!("{id}\t{level}\tERROR\t{e}"),
        Ok(run) => {
            let steps: Vec<String> = run.spikes.iter().map(|s| s.step.to_string()).collect();
            let finite = || run.voltage.iter().copied().filter(|v| !v.is_nan());
            format!(
                "{id}\t{level}\t{}\t{}\t{}\t{:?}\t{:?}",
                run.spikes.len(),
                if steps.is_empty() { "-".to_owned() } else { steps.join(",") },
                run.voltage.iter().filter(|v| v.is_nan()).count(),
                sum(finite()),
                sum(finite().map(f64::abs)),
            )
        }
    }
}

fn main() {
    let mut paths: Vec<PathBuf> = Vec::new();
    for arg in std::env::args().skip(1) {
        let p = PathBuf::from(arg);
        if p.is_dir() {
            let mut inside: Vec<PathBuf> = fs::read_dir(&p)
                .expect("a readable directory")
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|q| q.extension().is_some_and(|x| x == "json"))
                .collect();
            inside.sort();
            paths.extend(inside);
        } else {
            paths.push(p);
        }
    }
    if paths.is_empty() {
        eprintln!("usage: glif_allen <config.json | directory> ...");
        std::process::exit(2);
    }
    for p in &paths {
        println!("{}", line(p));
    }
}
