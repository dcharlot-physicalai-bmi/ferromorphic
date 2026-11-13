# Toolchain receipt, rustc 1.100.0

Recorded after `channel-rust-stable.toml` on static.rust-lang.org, manifest date 2026-11-12.
The previous pin was rustc 1.98.1 (48a229cea, 2026-09-01), channel date 2026-09-03.

```text
rustc 1.100.0 (0ef1bc386 2026-11-09)
binary: rustc
commit-hash: 0ef1bc386059596dada8fb65b50425aab2ea637a
commit-date: 2026-11-09
host: x86_64-unknown-linux-gnu
release: 1.100.0
LLVM version: 23.1.5

cargo 1.100.0 (3acc79c0c 2026-11-05)
```

`Cargo.lock` sha256 `77b4efb658b5ca5ec0c1d61e4a9bf400cba44ceead417b74fd4c237a367f112a`.
`cargo update` locked 0 packages. `cargo upgrade --incompatible` (cargo-edit 0.13.14) found no
dependency to move. `[dependencies]` is empty and stays empty.

`cargo audit` 0.23.0 loaded 1509 advisories from the RustSec database and reported no
vulnerability. `cargo deny` 0.20.2 reported advisories, bans, licenses and sources ok.
`deny.toml` allows Apache-2.0 by name. No advisory is ignored. Nothing is upstream-blocked.

Clippy 1.100 denies `approx_constant` on two literals that were already the standard constants.
`f64::consts::GOLDEN_RATIO` negated is bits `bff9e3779b97f4a8`. `f64::consts::EULER_GAMMA` is
bits `3fe2788cfc6fb619`. The source now names the constants. The bits did not move.

On this compiler the 14-tau ridge-10 held-out scores match `tools/fixtures/mackey_14tau_ridge10.tsv`
on all 14 rows. The finite mean is 24.61700776. The held-out receipts are unchanged:
tau-17 teacher receipt `514578ad302a7b760782aeeeee844aae6cfc485a7b65255dbb579befc291400c`,
output `46c20eb42fd8541dbc81e05b503bd6b67ac04eebec89ac00f781f4271909a3b7`;
tau-30 autonomous output `87b88691290440b225adad93d5bbc67d0391ed251a7a1d7b3857c723df8830b3`;
`indy_20170131_02` output `5f388980124065752ef30f3e78745eac33ebbc0407a92e42c8c0e36defd7a4bc`.
The six reaching R² figures printed by `tests/heldout_tables.rs` match between rustc 1.98.1 and
1.100.0. Only the recorded millisecond timings differ.

The workflow's `dtolnay/rust-toolchain` pin `683b588c487a69d01f6c6ddf40f036ff682cc2c8` is the
latest commit on that action's master (2026-10-16). A full SHA pin does not select a compiler:
the action requires a `toolchain` input, which is why verify run
`43441561144` failed both jobs with `'toolchain' is a required input` before any gate ran.
The workflow now passes toolchain `1.100.0`, target `wasm32-unknown-unknown`, and components
`clippy`, `rustfmt`, `rust-docs`. `actions/checkout` `3d3c42e5aac5ba805825da76410c181273ba90b1`
is v7.0.1, `actions/upload-artifact` `cf430e030ddbb5b0abf93d22962f4752f3646cd9` is v7.0.2, and
`actions/setup-python` `6d4fd22b20465ccc0f7354476588e8df7c9ca0e0` is v7.1.0. Those are the current
release tags. No action SHA moved.

Python reference pins stay `numpy==1.26.4`, `scipy==1.13.1`, `mpmath==1.3.0`, `pyspike==0.9.0`.
PyPI's current releases are NumPy 2.5.4, SciPy 1.18.1, mpmath 1.4.1, and PySpike 0.9.0.
PySpike 0.9.0 is already current, and it builds against NumPy 2.5.4 when `Python.h` is present.
`jitcdde` 1.8.3 in the verification environment is the current PyPI release and stays the solver
for the pinned series. Torch `2.7.1+cpu`, NeuroBench `e521c28f5798cb40bb9423758bf066dad53fe070`,
NIR `563c58b45873429e3b1b6f71763ec334ceb981d5` and nirtorch `1ad5e67ce373ebfc82347209be321b7f1b7c44e6`
stay. NumPy 2.5.4 and SciPy 1.18.1 are not substituted into `tools/reference/requirements.txt`:
the reference scripts quote figures produced by 1.26.4 and 1.13.1, and those scripts were not
re-run to a verdict on the newer pair. A requirements change would send that unverified pair
through the scientific job.
