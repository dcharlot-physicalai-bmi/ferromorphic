//! Read a ferromorphic NIR text graph and write its canonical text.
//!
//! The HDF5 `.nir` container stays in `tools/nir_hdf5_oracle.py`. This example is the Rust side of
//! that exchange: it accepts the text the oracle emits from an official file and prints the text
//! [`ferromorphic::nir::Graph::to_text`] produces. A second run of the same bytes is the identity.

use std::env;
use std::fs;
use std::io::{self, Read, Write};
use std::process::ExitCode;

use ferromorphic::nir::Graph;

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    let path = args.next();
    let text = match path.as_deref() {
        None | Some("-") => {
            let mut buf = String::new();
            if io::stdin().read_to_string(&mut buf).is_err() {
                eprintln!("could not read stdin");
                return ExitCode::from(1);
            }
            buf
        }
        Some(path) => match fs::read_to_string(path) {
            Ok(text) => text,
            Err(err) => {
                eprintln!("could not read {path}: {err}");
                return ExitCode::from(1);
            }
        },
    };
    let graph = match Graph::from_text(&text) {
        Ok(graph) => graph,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let canonical = graph.to_text();
    let again = match Graph::from_text(&canonical) {
        Ok(graph) => graph.to_text(),
        Err(err) => {
            eprintln!("canonical text did not re-parse: {err}");
            return ExitCode::from(1);
        }
    };
    if again != canonical {
        eprintln!("canonical text is not a fixed point");
        return ExitCode::from(1);
    }
    let mut out = io::stdout().lock();
    if out.write_all(canonical.as_bytes()).is_err() {
        return ExitCode::from(1);
    }
    ExitCode::SUCCESS
}
