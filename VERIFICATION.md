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

`tools/intrinsic_2007_reference.py` prints its quadratures and then exits 2 when `papers/triesch-2007.pdf` is absent. `tools/gapjunction_reference.py` prints sections A–D and exits 2 when either PDF is absent. The runner accepts that exit only together with `UNAVAILABLE` on stderr and a stdout marker that the quadrature actually ran. Exit 0 is not available for those figure sections while the PDF is missing. Statuses `INCOMPLETE`, `UNAVAILABLE` and `DELEGATED` are not passes. The report field `complete_reproduction` is true only when every entry is `PASS`.

`tools/glif_sweep.py` and `tools/clopath_figures.py` do not run until their inputs exist. The reason is stored on the manifest entry. Creating the missing paths and failing the command is a failure, not an unavailable result.

`python3 tools/scientific_verify.py --require-all`, with the pinned interpreter, was run on this tree. It exited 2. The log says `not a complete reproduction` and names `release-gates` as `DELEGATED`, `intrinsic-2007` and `gapjunction-reference` as `INCOMPLETE`, and `glif-sweep` and `clopath-figures` as `UNAVAILABLE`. Report sha256 `783812ed34a2638a978668e1503488f51c50dae42238e8ff748f1e2e657bda20`.

`tools/mutate.py` on a slim copy needs `rust-toolchain.toml` in that copy. Without it the copy's Cargo was 1.83.0, which cannot parse edition 2024, and every module's baseline was `BASELINE-caught` before any mutation ran. `tools/slim.py` now copies `rust-toolchain.toml`. On rustc 1.98.1 a mutation of the slim `istdp` copy takes about 12 s. At that rate the 10,579 recorded mutations are about 35 hours on one core, which was not available here. Completed slim sweeps, each `EXIT:0`: `istdp` 21 caught, `rng` 21, `crossover` 23, `force` 24, `net` 26, `forwardforward` 27, `encode` 28 caught and 1 equivalent, `decolle` 31, `voxel` 37, `polychron` 37, `biosignal` 39, `eventprop` 40, `sim` 40 caught and 1 equivalent, `spike` 41, `sdr` 41 caught and 4 equivalent, `ottt` 46, `srm` 49. That is 571 caught and 6 equivalent, 577 of 10,579. The other lists were not run to a verdict.

## Two release failures, and the repairs

On rustc 1.98.1 the release profile (`opt-level = 3`, `lto = true`) failed two library tests. Both assertions were kept. The constructions around them were wrong on this libm, not the tolerances.

`Theta::canonical_y` is `tan(θ/2)`. The unstable fixed point used by the circle test is `a = 0.5`, and `exact_theta_after` treats `u₀ = (y₀ − a)/(y₀ + a)` as exactly zero only when `y₀` is exactly `a`. Otherwise a long interval overflows `u₀ · e^{2as}`. `θ = 2·atan(0.5)` has `tan(θ/2) = 0.49999999999999994` here, and `(2·atan(a)).next_down()` is `0.4999999999999999`. One ulp above `2·atan(a)` has `tan(θ/2) = 0.5` exactly. The test walks at most four ulps and requires exact equality. One ulp below that phase is required to move within a second. One ulp above is required to spike once and settle at `y = −a`.

`NetworkDensity::eigenvalue` at `λ = 10⁻²⁰⁰` was dividing a forward difference that had become exact zero. The characteristic there is about 11.6, and `h = 10⁻⁷·|λ|` sits below an ulp of that value, so the quotient is rounding noise and then `0/h`. The Newton step `f/slope` is `0/0`, and the search reported `NoConvergence { steps: 14, re: NaN, im: NaN }`. A zero-length step, the subnormal guess, still divides, so that refusal stays a `NaN` after one step. A positive step whose two samples are the same `f64`, a zero-length difference, or a non-finite step returns `DensityError::NoConvergence` with that last finite iterate and the characteristic there. `Eigenvalue` is returned only when the step is at most `NEWTON_TOLERANCE` of `|λ|` and the residual norm is at most `NEWTON_RESIDUAL` (`1e-12`). The 1 MHz guess is unchanged: ten steps, characteristic norm under `1e-12`.

`zsh tools/gates.sh` on that repaired tree, same toolchain and release profile, exited 0. The library harness passed 3,069 tests in 82.71 s, the `scientific_gates` harness passed 6, and doctests passed 19. Clippy, the wasm32 build, the three examples, the prose gate, `readme_numbers.py` and rustdoc exited 0.

## Evidence limits

A passing `cargo test` is a statement about this simulator. It is not a statement about a chip. `Ledger` counts operations. `joules_synops_only` multiplies a count by a published price and is the figure the ledger's own documentation tells a caller not to defend as a complete bill. No device table in this crate prices `e_syn_fetch`. Hardware mapping, a NeuroBench-style held-out task, continual or embodied competence, and a calibrated whole-system energy are parallel objectives. None of them is executed here.

## Parallel workstreams

- Re-run `python3 tools/mutate.py` on the current tree, one module at a time, and keep a repair only after the mutation it addresses returns `caught`.
- Run `python3 tools/scientific_verify.py --require-all` on a machine that has the copyrighted PDFs and the Allen sweep cache, and archive `target/verification/report.json`.
- Put HDF5 `NIR` beside this crate, in a sibling that is allowed a dependency. This crate's text graph stays the std-only form.
- Measure a workload on a board, with the instrument and the boundary named, and grade that reading `Evidence::Metered`. Until that exists, a simulated joule stays a simulated joule.
- Hold out a NeuroBench-style task, and a continual or embodied competence task, on data this repository does not yet pin.
