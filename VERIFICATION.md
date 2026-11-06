# Verification

This file is the index for the scientific-verification workstream. Counts of modules, tests and mutations live in `README.md` and are recomputed by `python3 tools/readme_numbers.py`. The supported-model matrix is `docs/verification/MATRIX.md`. The exchange contract is `INTEGRATION.md`.

The crate stays std-only. Reference interpreters live in `tools/` and are pinned in `tools/reference/requirements.txt` and `tools/reference_manifest.json`. `rust-toolchain.toml` pins rustc 1.98.1 (48a229cea, channel date 2026-09-03), which is the release `Cargo.toml`'s `rust-version = "1.98"` names as verified here.

## Commands

| command | role |
| --- | --- |
| `zsh tools/gates.sh` | nine release gates: `cargo test --release`, clippy `-D warnings`, wasm32, three examples, prose, readme numbers, rustdoc |
| `python3 tools/scientific_verify.py` | manifest: in-tree commands must pass; missing primaries stay `INCOMPLETE` or `UNAVAILABLE` and are printed |
| `python3 tools/scientific_verify.py --require-all` | exit 2 unless every manifest entry is `PASS` |
| `python3 tools/scientific_verify.py --self-test` | the runner itself fails on a silent skip |
| `python3 tools/mutation_selftest.py` | five mutations of the closed-form specimen must be caught |
| `python3 tools/model_matrix.py --check` | the matrix matches the Neuron impls |
| `FERROMORPHIC_PYTHON=target/ref-venv/bin/python python3 tools/scientific_verify.py` | same runner on the pinned NumPy, SciPy, mpmath and PySpike |

`{python}` in the manifest is `FERROMORPHIC_PYTHON` or the interpreter that launched the runner. A report is written to `target/verification/report.json` with sha256 sums of each log. `target/` is gitignored.

## Tolerances re-executed here

`tests/scientific_gates.rs` and `tools/scientific_gates/fixture.rs` use these bounds.

- Lif free decay against `v_rest + (v0 - v_rest) exp(-t/tau)`, at 1e-2 s and 1e-5 s steps, must land inside 1e-12 V. The start is subthreshold.
- Forward Euler on that same decay, step 1e-3 s, must land outside 1e-12 V. Widening the tolerance to 1e-3 is one of the mutations the self-test requires to fail.
- Forward Euler's measured order on that decay is required to lie in [0.8, 1.5] in the specimen and in [0.7, 1.6] for the Izhikevich study below. A flat error sequence is rejected.
- Integrate-and-fire uses `T = c (v_th - v) / I`, independent of `IntegrateAndFire::isi`. The simulated spike must not be early, and must fall less than two ticks after `T`. A capacitance of 1 pF in place of 1 nF must miss that tick.
- Izhikevich, `substeps = 1`, zero current, 10 ms: the crate's semi-implicit Euler is compared with a classical RK4 of `v' = 0.04 v^2 + 5 v + 140 - u`, `u' = a (b v - u)`, reference step 0.01 ms. Crate steps 0.5, 0.25, 0.125 and 0.0625 ms. The one-step slope at the paper's start is -3 mV/ms by direct substitution. Dropping `0.04 v^2` must miss the RK4 trajectory.
- Equivalence binaries must print a disagreement count of 0. The regex is `(?<!\d)0 disagreements`, so a count of 10 does not pass.

On the release-test run of `tests/scientific_gates.rs` in this workstream, forward Euler on the Lif decay produced errors 5.1475999273561945e-5, 2.570011673592998e-5 and 1.2838543810331227e-5 V, with orders 1.0021250188353508 and 1.0012933359466971. The Izhikevich comparison against RK4 ended at v = -71.27536064594346 mV and produced errors 0.008258875643647912, 0.003473515201022792, 0.0015607992139337057 and 0.000735232184709389 mV, with orders 1.2495489767683985, 1.154111454486211 and 1.0860131295330195. The specimen's own forward-Euler errors were 4.6967593814506436e-5, 2.3235006417821058e-5 and 1.155648562160827e-5, orders 1.015365641786043 and 1.007597308748482. A frozen stepper measured order 0 and was rejected.

`tools/density_reference.py` called `brentq` on `[1e-6, 499]`. SciPy 1.13.1 refuses that bracket for the C_E = 4000, J = 0.2 mV, g = 8 cell because the Siegert integral is NaN at 499 Hz, while the sign change lies between 1 Hz and 10 Hz. The ceiling is now halved until the endpoint is finite. A ceiling that was already finite stays 499 Hz.
- NIR text round-trips. A membrane with `r = 0` and `v_leak = v_threshold` does not fire under `CubaState::step_exact` and fires on the first tick after `Graph::to_net`.
- `Ledger::joules` against `TRUENORTH_2014` and `LOIHI_2018` stays `None` on a workload whose `syn_ops` is nonzero. `LOIHI_2018.evidence` is `Evidence::Simulated`.

## What the manifest does with a missing primary

`tools/intrinsic_2007_reference.py` prints its quadratures and then exits 2 when `papers/triesch-2007.pdf` is absent. `tools/gapjunction_reference.py` prints sections A–D and exits 2 when either PDF is absent. The runner accepts that exit only together with `UNAVAILABLE` on stderr and a stdout marker that the quadrature actually ran. Exit 0 is not available for those figure sections while the PDF is missing. `--require-all` still exits non-zero, because an incomplete reading is not a pass.

`tools/glif_sweep.py` and `tools/clopath_figures.py` do not run until their inputs exist. The reason is stored on the manifest entry. Creating the missing paths and failing the command is a failure, not an unavailable result.

The historical mutation lists under `tools/mutations/` are not re-executed by this workstream. `tools/mutate.py` remains the harness. Re-running every recorded mutation rebuilds the crate once per edit; that sweep is a separate execution from the five-mutation specimen above.

## Release-suite result on rustc 1.98.1

`zsh tools/gates.sh` on x86_64-unknown-linux-gnu, rustc 1.98.1 (48a229cea 2026-09-01), release profile `opt-level = 3` and `lto = true`. The library harness ran 3,069 tests: 3,067 passed and 2 failed, in 84.41 s. `density::tests::the_eigenvalue_search_reports_its_steps_and_refuses_to_diverge` stopped at `NoConvergence { steps: 14, re: NaN, im: NaN }` (`src/density.rs`). `exponential::tests::the_theta_closed_form_solves_the_circle_equation_it_is_written_for` compared `canonical_y()` = 0.4999999999999999 with 0.5 (`src/exponential.rs`). A second run of those two tests on the same binary failed the same way. Clippy `-D warnings`, `wasm32-unknown-unknown`, `event_driven_saves`, `lif_closed_form`, `stdp_window`, the prose gate, `readme_numbers.py` and `cargo doc --no-deps` passed on the final tree. Because the library harness failed, that `cargo test --release` did not continue into doctests or `tests/scientific_gates.rs`. Run separately, 19 doctests passed and the scientific-gates harness passed 6. These two library failures were not edited into passing.

## Evidence limits

A passing `cargo test` is a statement about this simulator. It is not a statement about a chip. `Ledger` counts operations. `joules_synops_only` multiplies a count by a published price and is the figure the ledger's own documentation tells a caller not to defend as a complete bill. No device table in this crate prices `e_syn_fetch`. Hardware mapping, a NeuroBench-style held-out task, continual or embodied competence, and a calibrated whole-system energy are parallel objectives. None of them is executed here.

## Parallel workstreams

- Re-run `python3 tools/mutate.py` on the current tree, one module at a time, and keep a repair only after the mutation it addresses returns `caught`.
- Run `python3 tools/scientific_verify.py --require-all` on a machine that has the copyrighted PDFs and the Allen sweep cache, and archive `target/verification/report.json`.
- Put HDF5 `NIR` beside this crate, in a sibling that is allowed a dependency. This crate's text graph stays the std-only form.
- Measure a workload on a board, with the instrument and the boundary named, and grade that reading `Evidence::Metered`. Until that exists, a simulated joule stays a simulated joule.
- Hold out a NeuroBench-style task, and a continual or embodied competence task, on data this repository does not yet pin.
