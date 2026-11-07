# Ferromorphic step clock and a Ferromotion closed loop

Draft exchange. It records what ferromorphic implements, what was read in Ferromotion, and fields that are still negotiable. It does not add a dependency and it does not declare a function in the other repository.

Ferromotion was read at commit `5daca8aac1b513a18c54792383d021cf1d27af56` (2026-10-01, default branch `main` as of 2026-11-07). The repository states Apache-2.0 OR MIT. This crate was not linked against it. Ferromotion-core uses `nalgebra`, which this crate does not take as a dependency.

## Clocks that were actually found in Ferromotion

These symbols exist at that commit. They are not one clock.

| symbol | file | what it counts |
| --- | --- | --- |
| `DynamicsLab::step(&mut self, dt: f64)` and `DynamicsLab::time(&self) -> f64` | `crates/ferromotion-wasm/src/dynamics_lab.rs` | Accumulated `f64` seconds. The step adds `dt` after a semi-implicit Euler with internal substeps. `τ = 0` in that lab. |
| `Clock::now_seconds(&self) -> f64`, `StdClock` | `crates/ferromotion-bench/src/lib.rs` | Benchmark wall time. `StdClock` reads `std::time::Instant`. It is not a simulation step. |
| `ChunkClock` | `crates/ferromotion-policy/src/clock.rs` | Action-chunk queue. `control_period` is seconds. `executed`, `queued`, `frozen` and `chunks` are integers. `frozen_for` is `ceil(inference_time / control_period)`, capped at `chunk_len`. |
| `ode::integrate` | `crates/ferromotion-core/src/ode.rs` | Adaptive Dormand–Prince. Time and step size are `f64`. |
| `integrate_pos(..., dt: f64)` | `crates/ferromotion-core/src/mujoco_step.rs` | MuJoCo position update. The methods on that type are `pub(crate)`. |
| `causal::Scm` | `crates/ferromotion-core/src/causal.rs` | Linear-Gaussian structural causal model. Pearl's `do` operator. It does not order events. |

No integer epoch, and no shared step index, appeared in those files. An epoch field on a joint record is negotiable. The receipt below has none.

## What ferromorphic implements

`Sim::t` is a `u64`. It is the tick the next `Sim::step` will process. The method increments it at the end. A spike produced by that call is stored by `Sim::run` as `Spike { t, source }` with the value of `Sim::t` from before the increment. `Spike::t` is a tick index. Seconds are `step as f64 * dt`, one multiply, via `step_seconds`.

`dt` is seconds. `Sim::new` does not reject a non-positive `dt` on the clocked path. `Sim::with_exact_timing` returns `SimError::BadTick` when `dt` is not finite and positive. `same_dt` accepts two values only when both are finite and positive and `to_bits()` agrees. A Ferromotion `dt` of `0.1` and a ferromorphic `dt` of `0.1 + f64::EPSILON` are different time bases.

`StepReceipt::to_text` writes this record. The bytes are fixed by `a_step_receipt_records_the_tick_and_refuses_a_different_dt`.

```text
ferromorphic-step 1
step 4
dt_bits 3fb999999999999a
mode clocked
spike 4 0
spike 4 1
```

`dt_bits` is the little-endian IEEE-754 hex of `0.1_f64`. `mode` is `clocked` or `event`. Spike lines are `(t, source)` with `t` equal to `step`, in non-decreasing order. `StepReceipt::from_text` returns `TimebaseError::Unordered` otherwise.

### Phase inside one tick

`Sim::step` does these four things, in this order.

1. Take the deliveries whose slot is `t % ring_len` and add their weights into the postsynaptic injection. Each delivery counts one `syn_ops` and one `syn_fetches`.
2. Update neurons. Clocked mode walks neuron index `0..n`. Event-driven mode sorts the touched indices, catch-up jumps the quiet gap, then applies the same bump and step. External current is amperes, empty or length `n`.
3. Post each spike onto the ring at `(t + 1 + delay) % ring_len`.
4. Clear the injection and set `t = t + 1`.

Event-driven touched indices are sorted so `fired` is in neuron order, which is the order clocked mode emits. `Spike`'s derived `Ord` is `(t, source)`. Two spikes from one neuron in one tick under exact timing keep emission order. That is the tie-break this crate implements.

### Delay, late, future, reset, refractory

`delivery_step(step, delay)` is `step + 1 + delay`, checked against `u64` overflow (`TimebaseError::Overflow`). A delay of zero is delivered on the following tick. `a_delayed_synapse_delivers_when_it_says_it_will` pins delay 5 at `source.t + 6`. `Net::delay` is in ticks. The simulator sizes its ring to `max_delay + 1`, so a synapse cannot name a delay the ring cannot hold.

`Sim::step` still adds `t + 1 + delay` with the ordinary `+` operator. In debug that panics on overflow. In release it wraps. The checked helper is the contract. Changing `Sim::step` to return `TimebaseError` is negotiable and is not done here.

`accept_event_step(event, clock, max_delay)` accepts `clock..=clock+max_delay`. An earlier step is `TimebaseError::Late`. A later step is `TimebaseError::BeyondHorizon`. `Train::push` panics when a spike's tick is below the last stored tick. There is no late-event queue.

`Sim::reset_state` restores each neuron, clears the ring and the injection, and leaves `Sim::t` and `Ledger` unchanged. `a_reset_returns_every_neuron_to_rest_and_keeps_the_ledger` pins the ledger half. The receipt test pins the surviving `t`.

Refractory time is seconds on the neuron. Clocked mode holds a cell refractory for `ceil(t_ref / dt)` ticks. Event-driven catch-up rounds the refractory piece up to a tick boundary and walks exact-timing refractory tick by tick. `Neuron::EXACT_OVER_GAPS == false` makes `Sim::new(..., Mode::EventDriven)` return `SimError::NotExactOverGaps`.

NIR `LIF` has no refractory parameter. `Graph::from_net` returns `BridgeError::RefractoryNotRepresentable` when `Lif::t_ref != 0`. NIR fires on `v > v_threshold`. `neuron::Lif` fires on `v >= v_th`. The bridge copies the threshold unchanged.

`AerEvent::t` is microseconds on the wire. `AerEvent::to_event` keeps that integer as the tick. `TrainMap` is the conversion to another tick length. Mixing those microseconds with `Sim::t` without a stated tick length is a time-base mismatch. `same_dt` is the check once both sides are in seconds.

### Controller, sensor, actuator inside this crate

`control::SpikingPid::step(dt, error)` and `control::Pendulum::step(dt, torque_nm)` advance by an `f64` duration. They do not read `Sim::t`. `Pendulum::energy` is mechanical energy of that pendulum, joules of `T + U`. It is not a meter reading.

`MeterReceipt::unmetered` is the reading in this workspace. `measured_joules` is `None`. `MeterReceipt::synthetic_fixture` is labelled `synthetic` and also returns `None`. No wall, board, or SoC sample was taken.

## Proposed joint step

This schedule is a proposal. No function in either repository runs it.

Both sides hold the same `dt` bits, checked with `same_dt`, and the same integer `step`. Ferromotion would have to expose that integer. Today `DynamicsLab::time` is an accumulated `f64`, so a long run can drift from `step as f64 * dt`. Adopting the integer and multiplying once is the ferromorphic rule. Whether Ferromotion keeps the `f64` accumulator is negotiable.

At the start of step `k`:

1. Sense. Read plant state that is current at step `k`. Spikes stamped `k` do not exist yet.
2. Control. A command may use spikes with `t < k` and the plant state from (1). `ChunkClock`'s `frozen` count stays inside Ferromotion. This crate does not call it.
3. Actuate. Torque is constant across the one shared `dt`. Substeps inside `DynamicsLab::step` stay inside Ferromotion.
4. The spiking network runs `Sim::step` for tick `k`. Spikes are stamped `k` and become visible to the controller at step `k + 1`, because delivery is `k + 1 + delay`.

A sensor event that arrives with a step below `k` is late and is rejected. A command that names a step above `k + max_delay` is rejected. Zero-delay actuation means the torque of step `k` affects the plant during step `k`, and the spikes caused by that step are visible at `k + 1`. Making spikes visible inside step `k` would be a same-tick feedback loop. This crate's delivery rule does not do that, and changing it is negotiable only together with `the_two_modes_produce_the_same_spike_train`.

## NIR and NeuroBench status beside this draft

neuromorphs/NIR commit `563c58b45873429e3b1b6f71763ec334ceb981d5` is SPDX BSD-3-Clause, copyright 2023 NIR Team. `CubaLI` from that pin is stored in `nir::CubaLi`, round-trips in the text graph, and `Graph::to_net` returns `BridgeError::UnsupportedNode`. Official `.nir` files are HDF5. This crate does not read them. `tools/nir_hdf5_oracle.py` imports the pinned Python package when it is installed and exits 2 when it is not.

NeuroBench commit `e521c28f5798cb40bb9423758bf066dad53fe070` is Apache-2.0. `metrics::smape` follows that harness's `200 * mean(|p-y|/(|p|+|y|))` arithmetic, with a zero denominator contributing 200, on arrays the test builds. Those arrays are not a held-out score. `metrics::r_squared` is one-dimensional and returns `MetricError::ZeroVariance` when the target is constant. The harness's R2 averages two columns and can yield a non-finite value on a constant column. `metrics::accuracy` and `metrics::mean_squared_error` match the harness's means on equal-length finite samples. Empty input is an error here.

No official NeuroBench task was run. The harness downloads Mackey–Glass from `https://huggingface.co/datasets/NeuroBench/mackey_glass` and expects Primate Reaching `.mat` files from Zenodo record 583331 (`indy_20170131_02.mat`, `indy_20160630_01.mat`, `indy_20160622_01.mat`, `loco_20170301_05.mat`, `loco_20170215_02.mat`, `loco_20170210_03.mat`). Those files are not in this workspace. Speech Commands, MSWC and the other task families were not fetched. Dataset-by-dataset terms were not accepted. No account was created.

`src/nir/nirdata.rs` holds `TimeGridded`, `EventData` and `ValuedEventData` for the shapes in that NIR commit's `docs/source/nirdata.md`. `time_shift` must lie in `[0, dt)`. Valued events convert to a grid. `ValuedEventData::to_events` returns `NirDataError::Unsupported`.
