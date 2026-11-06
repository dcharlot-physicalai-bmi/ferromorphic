# Event-model and NIR exchange

This contract is ferromorphic's side of a policy exchange. It states records this crate already produces and the checks it already enforces. It does not declare an API in any other repository, and this workstream does not modify one.

Four neighbouring efforts are named by the role they would fill. Their function names, types and file formats, beyond the records below, are outside this document.

| effort | role this contract assigns | record ferromorphic can already hand across |
| --- | --- | --- |
| bmi-3 | deterministic runtime and resource accounts | seed, `dt` in seconds, `sim::Mode`, spike train, ledger counts, `Bill` |
| bmi-4 | dynamics, contact, control, uncertainty | membrane state in SI units; controller outputs where `control` is used |
| bmi-1 | sampler and fabric interfaces, calibrated metrology | `aer::AerEvent`, NIR text graph |
| bmi-0 | site evidence, integrated | `Prices::source`, `Evidence`, the refusal from `Ledger::joules` |

## bmi-3 — runtime and resources

A run is determined by the network, the neuron values, `dt`, the mode, the external current, and any `rng::Rng` seed the caller passed. Two runs that share those inputs produce the same spike train. `Rng` is the only randomness in the library. `Sim::new` returns `SimError::NotExactOverGaps` when `Mode::EventDriven` is asked of a model whose `Neuron::EXACT_OVER_GAPS` is false.

`Ledger` counts `syn_ops`, `syn_fetches`, idle and driven membrane updates, `spikes_out` and `reads`. `Ledger::idle_fraction` is that idle share on a clocked run, and it is a property of the workload. `Ledger::bill` returns `Bill`, whose `total` is `None` when any term with a non-zero count has no price. `Bill::unpriced` names those terms. `Evidence` grades a price that was actually used.

A consumer that wants a deterministic resource account can store those fields. A consumer that wants a joule stores `total` only together with `unpriced` and `evidence`.

## bmi-4 — dynamics, contact, control, uncertainty

Public neuron interfaces take seconds, amperes and volts. `Izhikevich` keeps the 2003 paper's millisecond and millivolt constants inside the model and converts at `Neuron::step`. `control` states torque in newton-metres at the boundary it documents. `SigmaDeltaEncoder` is a controller component; the module states that it is not a `Neuron`.

This crate has no contact model, no rigid-body state and no posterior on a spike. A spike time is a simulated threshold crossing. Uncertainty on that crossing belongs to a dynamics consumer that has a noise model. Nothing in this library attaches one.

## bmi-1 — samplers, fabric, metrology

`aer` decodes vendor event streams into `AerEvent` values whose timestamps are microseconds, the unit those formats put on the wire. `TrainMap` converts to the crate's tick at a stated tick length. Coordinate orientation is the module's documented convention, including the `AEDAT` 2.0 row inversion.

`nir` holds an in-memory graph and a text serialisation. `Graph::to_text` and `Graph::from_text` round-trip. The module does not read HDF5 `.nir` files. Pedersen et al., Nature Communications 15:8122 (2024), https://doi.org/10.1038/s41467-024-52259-9, define that representation; the binary container stays outside this crate. A file in the wild is often stored in normalised units. The bridge treats numbers as SI, and a normalised graph yields normalised "volts" downstream.

`Graph::to_net` copies a threshold unchanged. `NIR` fires on `v > v_threshold`. `neuron::Lif` fires on `v >= v_th`. The resting-on-threshold case is asserted by `nir_strict_threshold_and_lif_non_strict_threshold_disagree_on_a_resting_membrane` in `tests/scientific_gates.rs` and by `the_bridge_turns_a_strict_crossing_into_a_non_strict_one` in `src/nir.rs`. `Net::n_syn` equals the number of stored synapses, once each.

Calibrated metrology of a sensor or a fabric is a measurement with an instrument. This crate's AER decoder and NIR bridge do not perform that measurement.

## bmi-0 — site evidence

`Prices::source` names the device and the publication the numbers came from. `TRUENORTH_2014` is an all-in 26 pJ figure from Merolla et al., Science 345:668–673 (2014), graded `Evidence::Measured`, and a bill against it still refuses a total. `LOIHI_2018` is the ~23.6 pJ/SOP from Davies et al., IEEE Micro 38(1) (2018), Table 2, captioned pre-silicon, graded `Evidence::Simulated`. `mapping::LOIHI_FABRIC` prices a tile hop from that same pre-silicon table. Every in-crate device table leaves `e_syn_fetch` unset, so `Ledger::joules` refuses when fetches were counted.

Site evidence is a reading from a place, with the boundary named. A ledger total computed from a price table is not that reading. A simulation is a simulation. Board execution is a separate record this repository does not produce.

## Marks that belong to other work

`NeuroBench` is the benchmark framework of Yik et al., Nature Communications 16:1545 (2025), https://doi.org/10.1038/s41467-025-56739-4. `NIR` is the neuromorphic intermediate representation of Pedersen and coauthors. `TrueNorth`, `Loihi`, `SpiNNaker`, `AllenSDK`, `PySpike`, `snnTorch` and `Lava` are their owners' names. This crate is an independent implementation that cites those sources. It does not incorporate their code, and the auxiliary scripts that fetch a reference do so beside the crate and record the licence in `tools/reference_manifest.json`.

## Held beside this contract

Hardware mapping onto a placed fabric, a NeuroBench-style held-out task, continual or embodied competence, and a calibrated whole-system energy are parallel objectives. They are not implied by a spike train, a NIR text file, or an operation count.
