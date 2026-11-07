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
ferromorphic-step 2
epoch 0
step 4
dt_bits 3fb999999999999a
mode clocked
spike 4 0
spike 4 1
```

`dt_bits` is the little-endian IEEE-754 hex of `0.1_f64`. `mode` is `clocked` or `event`. Spike lines are `(t, source)` with `t` equal to `step`, in non-decreasing order. `StepReceipt::from_text` returns `TimebaseError::Unordered` otherwise. A record whose first line is `ferromorphic-step 1` is refused. That version had no epoch.

### Episode epoch, implemented here

`Sim::new` opens episode 0 at tick 0 with no scheduling period yet. `Sim::episode_start(origin_ns, period_ns)` is what installs the period and the origin, advances the local epoch, and sets `t` to 0. Their adapter has no monotonic episode counter: `epoch_tick0` is either unset or `Some(0)`. The local epoch counter is this crate's stale-event check. It is not their field. `Sim::reset_state` restores neurons and clears the ring and keeps `t`, the epoch, `origin_ns`, and `period_ns`. A membrane reset is not a new episode. `an_episode_starts_at_tick_zero_and_rejects_the_previous_epoch` pins that split. An event whose epoch is not the clock's epoch is `TimebaseError::StaleEpoch`. `episode_start` when the epoch is already `u64::MAX` returns `TimebaseError::EpochOverflow` and leaves the clock where it was.

### One tick, two authorities

Read at ferromotion `00fa031083b85a4a95815a3899faab2b3afefd7d`, file `docs/acceptance/contracts/ferromotion-clock.md`, and the adapter `crates/ferromotion-core/src/ferromorphic_clock.rs` at that commit. This crate was not linked to that repository. No joint loop was run. Their constant `EPISODE_ENVELOPE_INTEROPERABLE` is false. Their constant `JOINT_LOOP_EXECUTED` is false.

The pin does not require `dt` bits to equal `(period_ns as f64) * 1e-9`. `EpisodeEnvelope` has `dt_bits`, `period_ns`, and `origin_ns`. It has no physical-time ratio field. `nominal_seconds` is `(period_ns as f64) * 1e-9`, refused for a zero period or a product that is not finite and positive. The comment on that function, and the comment on `EpisodeEnvelope::period_ns`, say the nominal seconds are not required to equal `dt`. The test `one_microsecond_is_a_valid_period_and_the_grid_is_integer` calls `episode_start(0, 0.1, 1_000)` and `bind_v2_timestamps` accepts an envelope whose `dt_bits` are the bits of `0.1` and whose `period_ns` is 1000. Those are not the same duration. The same test checks that the bits of `nominal_seconds(1000)` equal the bits of `1000.0 * 1e-9` and are not the bits of `1000.0 * 1e-9 * 1e9`.

`Schedule::try_new` and `nominal_seconds` follow that arithmetic. `dt` bits stay the caller's. 1000 ns is valid. A 1000 ns period may be paired with `dt = 0.1`. Scheduling stamps are `origin_ns + tick * period_ns`, checked. Membrane time stays `step as f64 * dt`. Neither is produced by adding `dt` in a loop. `Sim::episode_start(origin_ns, period_ns)` stores those integers, sets `t` to 0, and does not rewrite `dt`. The operation ledger is kept. Their adapter clears a spike vector also named ledger. That is a different record, and this crate does not copy that clear.

`Sim::try_step` checks `t + 1 + max_delay` before it touches a neuron or a ring slot. The sum is checked addition. The ring slot is `tick % ring_len` computed in `u64`. That modulus names a slot. It is not a wrap of the tick. A delay stored above `max_delay` is `TimebaseError::DelayPastRing`. `a_step_that_would_overflow_changes_nothing` sets `t` to `u64::MAX - 5` on a delay of 5 and checks that the membrane, the ring, the ledger and `t` are unchanged. `Sim::step` panics with the same error for callers that already know the tick fits. Untrusted ticks use `try_step`.

`ticks_per_action(control_period, sim_dt)` accepts the ratio only when it is an integer `k >= 1` and `k as f64 * sim_dt` equals `control_period`. `0.5` seconds at `dt = 0.125` is 4 ticks. `0.3 / 0.1` is not an integer in `f64` and is `TimebaseError::AmbiguousRate`, as is a control period shorter than one tick. `action_tick(index, k)` is `index * k` from tick 0 of the episode, checked. Action 0 is tick 0. Ferromotion's `ChunkClock` was not called. Nothing in that repository reads this epoch.

### Phase inside one tick

`Sim::step` does these four things, in this order.

1. Take the deliveries whose slot is `t % ring_len` and add their weights into the postsynaptic injection. Each delivery counts one `syn_ops` and one `syn_fetches`.
2. Update neurons. Clocked mode walks neuron index `0..n`. Event-driven mode sorts the touched indices, catch-up jumps the quiet gap, then applies the same bump and step. External current is amperes, empty or length `n`.
3. Post each spike onto the ring at `(t + 1 + delay) % ring_len`.
4. Clear the injection and set `t = t + 1`.

Event-driven touched indices are sorted so `fired` is in neuron order, which is the order clocked mode emits. `Spike`'s derived `Ord` is `(t, source)`. Two spikes from one neuron in one tick under exact timing keep emission order. That is the tie-break this crate implements.

### Delay, late, future, reset, refractory

`delivery_step(step, delay)` is `step + 1 + delay`, checked against `u64` overflow (`TimebaseError::Overflow`). A delay of zero is delivered on the following tick. `a_delayed_synapse_delivers_when_it_says_it_will` pins delay 5 at `source.t + 6`. `Net::delay` is in ticks. The simulator sizes its ring to `max_delay + 1`. `try_step` refuses a delay above that cache before the step runs.

`accept_event_step` adds `max_delay` with checked addition. A horizon that does not fit in a `u64` is `TimebaseError::Overflow`.

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

neuromorphs/NIR commit `563c58b45873429e3b1b6f71763ec334ceb981d5` is SPDX BSD-3-Clause, copyright 2023 NIR Team. The installed build reports `1.0.9.dev4+g563c58b45`. PyPI `nir` 1.0.8 is tag `v1.0.8`, four commits behind that pin. `CubaLI` from the pin is stored in `nir::CubaLi`, round-trips in the text graph, and `Graph::to_net` returns `BridgeError::UnsupportedNode`. Official `.nir` files are HDF5. This crate does not read them. `tools/nir_hdf5_oracle.py` reads five paper fixtures from that commit through the `nir` package, emits text, and checks parameter bits against `examples/nir_exchange`. `lif_rockpool.nir` fails `nir.read(type_check=True)` because the LIF output shape is `[1]` and the Output node stores `[1, 1, 1]`. The text keeps `[1, 1, 1]`, and `Graph::validate` returns `ShapeMismatch` on that edge. `cnn_sinabs.nir` writes `Flatten` `end_dim=-1`. The reader stores the last axis, which is `2` for size `[8, 4, 4]`. `nirtorch` commit `1ad5e67ce373ebfc82347209be321b7f1b7c44e6` (reported `2.7.dev5+g1ad5e67ce`, BSD-3-Clause) maps an Affine, bias included, and rejects a LIF: the default map has no LIF entry.

NeuroBench commit `e521c28f5798cb40bb9423758bf066dad53fe070` is tag `2.3.0`, Apache-2.0. `metrics::smape` follows that harness's `200 * mean(|p-y|/(|p|+|y|))` arithmetic, with a zero denominator contributing 200. `metrics::r_squared` is one column. `examples/r2_columns` averages the two columns the way harness `R2.compute` does, using population variance. A constant column is `MetricError::ZeroVariance` here.

Mackey–Glass data is Hugging Face dataset `NeuroBench/mackey_glass`, card license CC-BY-4.0, file `data.tar.gz` sha256 `5e7c2b62a5135b744e7b49f103f58a2d74c70fe2dae67890cd5f93a6110fdf7c`. `tools/neurobench_mackey.py` runs the example echo-state network at `repeat=1`, `seed_id=0`, `start_offset=0`, tau 17 through 30, on the test index range. On torch `2.7.1+cpu` the harness mean sMAPE is `26.846464407283186`. Rust on the same prediction bytes agrees within `1e-13` per tau. The comment in `esn_benchmark.py` prints `40.803375080880265` for that repeat. This run did not reproduce that comment. The example scales the reservoir by the first value of `torch.linalg.eig`, so eigenvalue order can move the score. Speech Commands, MSWC and the other task families were not run.

Primate Reaching `indy_20170131_02.mat` is Zenodo record 583331, license CC-BY-4.0, md5 `2790b1c869564afaa7772dbf9e42d784`. The 2D ANN checkpoint from the same NeuroBench commit, test split at `train_ratio=0.5`, `bin_width=0.004`, gives harness R2 `0.6211199760437012`. Rust on the replayed predictions, after restoring the model's zero spike buffer, differs by `2.5e-8`. The other five official sessions were not downloaded. Their sizes on the Zenodo record are about 0.38 GB to 1.14 GB each.

`src/nir/nirdata.rs` holds `TimeGridded`, `EventData` and `ValuedEventData` for the shapes in that NIR commit's `docs/source/nirdata.md`. `time_shift` must lie in `[0, dt)`. Valued events convert to a grid. `ValuedEventData::to_events` returns `NirDataError::Unsupported`.
