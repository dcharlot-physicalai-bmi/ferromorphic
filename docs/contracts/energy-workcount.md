# Work-count contract for a later meter

This note is the interface a meter can wrap. It is not a measurement. `ferromorphic` at this
revision reports `meter none` and `joules none`. No RAPL sample and no macmon sample is taken here.

`examples/heldout_count` runs one process for exactly `--count` steps. `repetitions` in the
receipt is 1. The process does not extend itself until two seconds have passed, and it does not
divide joules by a repetition count. A meter that wants a longer window starts the process again.
The work count of one start is the integer in the receipt, not a wall-clock duration.

## Arms

`activity` runs `WindowAnn` or `SuppliedEcho`. Reaching, teacher-forced Mackey, and replayed
Mackey may carry `matched_quality true` inside the gates in `heldout`. `mackey-auto` always has
`matched_quality false`. Its `parity_split` is the first index whose absolute gap exceeds `1e-8`,
or `none`. An autonomous end-point gap is not a quality pass.

`baseline` reads the same weight table and the same input series. Every loaded byte is mixed into
`touch_sha256` through `std::hint::black_box`. The outputs are the constant control value `-1`.
`role` is `attribution-control` and `matched_quality` is false. Passing `--claim-quality` with
`--arm baseline` is a refusal. The baseline is not a second model and it is not a matched-quality
alternative.

## What a later paired protocol would record

This crate does not run the protocol. The shape below follows ferrotherm `bench/thrml-tsp` at
`0d380e6d2ce8c57dcc716acef9e0f4cd62a15233` only as a list of fields a meter would fill. No joule
was taken here.

- Two sides, activity and baseline, in ABBA order: even repetitions activity then baseline, odd
  repetitions the reverse. An idle bracket before the pair and an idle bracket after it.
- Package and core counters as separate scopes. They are not added. A whole-SoC integral, if a
  meter defines one, is a third scope. This note does not define that integral. ferrotherm
  `0d380e6d` multiplies a mean of `sys_power` by seconds and labels the scope `whole-system`.
  That label and that integral are not copied here.
- Refusal of a negative incremental, of an incremental inside the meter's own noise band, and of
  a bracket whose ends drifted. Quiet and contended grades stay unpooled. `load1`, the pin, and
  other-core busy fractions belong on the meter's receipt.
- At least five repetitions before a Student-t interval. This CLI does not compute that interval.

`projected_prices_applied` stays false. A missing counter is unmetered, not a substituted price.
