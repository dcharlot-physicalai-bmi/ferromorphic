//! `NIRData`: recorded observables beside a [`crate::nir::Graph`].
//!
//! The shapes and conversion rules follow neuromorphs/NIR commit
//! `563c58b45873429e3b1b6f71763ec334ceb981d5`, file `docs/source/nirdata.md` and
//! `nir/data_ir/graph.py`, read as a comparison source. This file is an independent
//! implementation. It does not read HDF5. Official `.nir` files stay with the sibling oracle
//! `tools/nir_hdf5_oracle.py`.
//!
//! A time-gridded observable has shape `(samples, steps, neurons)` and a positive `dt` in
//! seconds. An event observable stores, per sample, a neuron index and a time in seconds.
//! Event times within a sample are kept in non-decreasing order. `time_shift` on the way from
//! a grid to events lies in `[0, dt)`. The inverse of a valued event is refused: the pinned
//! source implements valued-event to grid only.

use core::fmt;

use crate::nir::Graph;

/// Why a data record was refused.
#[derive(Debug, Clone, PartialEq)]
pub enum NirDataError {
    /// A length, a time, or a count was not a usable number.
    BadNumber {
        /// Which field.
        what: &'static str,
    },
    /// An array's length did not match the declared shape.
    BadShape {
        /// What was supplied.
        got: usize,
        /// What the shape required.
        want: usize,
    },
    /// An event time went backwards inside one sample, or a time fell outside `[0, t_max]`.
    BadTime {
        /// Sample index.
        sample: usize,
        /// Event index inside that sample.
        event: usize,
    },
    /// A neuron index was outside `0..neurons`.
    BadIndex {
        /// Sample index.
        sample: usize,
        /// Event index.
        event: usize,
    },
    /// `time_shift` was outside `[0, dt)`.
    TimeShift {
        /// The rejected shift, seconds.
        shift: f64,
        /// The grid step, seconds.
        dt: f64,
    },
    /// A conversion the pinned source does not define.
    Unsupported {
        /// What was asked.
        what: &'static str,
    },
    /// Two observables in one node share a key.
    DuplicateKey {
        /// The node name.
        node: String,
        /// The key.
        key: String,
    },
    /// A node named in the data record is not in the graph.
    UnknownNode {
        /// The name.
        name: String,
    },
}

impl fmt::Display for NirDataError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadNumber { what } => write!(f, "{what} is not a finite positive quantity"),
            Self::BadShape { got, want } => write!(f, "length {got} does not match shape {want}"),
            Self::BadTime { sample, event } => {
                write!(f, "sample {sample} event {event} is out of time order")
            }
            Self::BadIndex { sample, event } => {
                write!(f, "sample {sample} event {event} names a neuron that does not exist")
            }
            Self::TimeShift { shift, dt } => {
                write!(f, "time_shift {shift} is outside [0, {dt})")
            }
            Self::Unsupported { what } => write!(f, "{what} is not defined"),
            Self::DuplicateKey { node, key } => {
                write!(f, "node {node} repeats observable {key}")
            }
            Self::UnknownNode { name } => write!(f, "node {name} is not in the graph"),
        }
    }
}

impl std::error::Error for NirDataError {}

/// One sample of index/time events.
#[derive(Debug, Clone, PartialEq)]
pub struct EventSample {
    /// Neuron indices.
    pub idx: Vec<u32>,
    /// Event times, seconds, non-decreasing.
    pub time: Vec<f64>,
}

/// Binary events for one observable.
#[derive(Debug, Clone, PartialEq)]
pub struct EventData {
    /// One entry per sample.
    pub samples: Vec<EventSample>,
    /// Neuron count.
    pub neurons: usize,
    /// End of the recording, seconds. Event times lie in `[0, t_max]`.
    pub t_max: f64,
}

/// Events that also carry a value. The inverse conversion to [`EventData`] is refused.
#[derive(Debug, Clone, PartialEq)]
pub struct ValuedEventData {
    /// One entry per sample.
    pub samples: Vec<EventSample>,
    /// Parallel to each sample's events.
    pub value: Vec<Vec<f64>>,
    /// Neuron count.
    pub neurons: usize,
    /// End of the recording, seconds.
    pub t_max: f64,
}

/// A dense record of shape `(samples, steps, neurons)`.
#[derive(Debug, Clone, PartialEq)]
pub struct TimeGridded {
    /// Row-major `samples * steps * neurons` values.
    pub data: Vec<f64>,
    /// Sample count.
    pub samples: usize,
    /// Time steps.
    pub steps: usize,
    /// Neurons.
    pub neurons: usize,
    /// Step, seconds.
    pub dt: f64,
}

impl TimeGridded {
    /// Build a grid. `data.len()` must be `samples * steps * neurons`. `dt` must be finite and
    /// positive. Every value must be finite.
    ///
    /// # Errors
    ///
    /// [`NirDataError::BadNumber`] or [`NirDataError::BadShape`].
    pub fn try_new(
        data: Vec<f64>,
        samples: usize,
        steps: usize,
        neurons: usize,
        dt: f64,
    ) -> Result<Self, NirDataError> {
        if !(dt.is_finite() && dt > 0.0) || samples == 0 || steps == 0 {
            return Err(NirDataError::BadNumber { what: "grid" });
        }
        let want = samples.checked_mul(steps).and_then(|n| n.checked_mul(neurons)).ok_or(
            NirDataError::BadShape { got: data.len(), want: 0 },
        )?;
        if data.len() != want {
            return Err(NirDataError::BadShape { got: data.len(), want });
        }
        if data.iter().any(|v| !v.is_finite()) {
            return Err(NirDataError::BadNumber { what: "data" });
        }
        Ok(Self { data, samples, steps, neurons, dt })
    }

    fn at(&self, sample: usize, step: usize, neuron: usize) -> f64 {
        self.data[(sample * self.steps + step) * self.neurons + neuron]
    }

    /// Grid to binary events. Every value must be exactly `0.0` or `1.0`. `time_shift` must lie
    /// in `[0, dt)`. Event time is `step * dt + time_shift`. Order inside a sample is time, then
    /// neuron index.
    ///
    /// # Errors
    ///
    /// [`NirDataError::TimeShift`], [`NirDataError::Unsupported`], [`NirDataError::BadIndex`].
    pub fn to_events(&self, time_shift: f64) -> Result<EventData, NirDataError> {
        if !(time_shift.is_finite() && time_shift >= 0.0 && time_shift < self.dt) {
            return Err(NirDataError::TimeShift { shift: time_shift, dt: self.dt });
        }
        let mut samples = Vec::with_capacity(self.samples);
        for s in 0..self.samples {
            let mut idx = Vec::new();
            let mut time = Vec::new();
            for step in 0..self.steps {
                for n in 0..self.neurons {
                    let v = self.at(s, step, n);
                    if v == 0.0 {
                        continue;
                    }
                    if v != 1.0 {
                        return Err(NirDataError::Unsupported {
                            what: "a non-binary grid to EventData",
                        });
                    }
                    idx.push(u32::try_from(n).map_err(|_| NirDataError::BadIndex {
                        sample: s,
                        event: idx.len(),
                    })?);
                    time.push(step as f64 * self.dt + time_shift);
                }
            }
            samples.push(EventSample { idx, time });
        }
        Ok(EventData { samples, neurons: self.neurons, t_max: self.steps as f64 * self.dt })
    }
}

fn check_events(
    samples: &[EventSample],
    neurons: usize,
    t_max: f64,
) -> Result<(), NirDataError> {
    if !(t_max.is_finite() && t_max >= 0.0) || neurons == 0 {
        return Err(NirDataError::BadNumber { what: "events" });
    }
    for (s, sample) in samples.iter().enumerate() {
        if sample.idx.len() != sample.time.len() {
            return Err(NirDataError::BadShape {
                got: sample.idx.len(),
                want: sample.time.len(),
            });
        }
        let mut prev = f64::NEG_INFINITY;
        for (e, (&i, &t)) in sample.idx.iter().zip(&sample.time).enumerate() {
            if !(t.is_finite() && t >= 0.0 && t <= t_max) || t < prev {
                return Err(NirDataError::BadTime { sample: s, event: e });
            }
            if i as usize >= neurons {
                return Err(NirDataError::BadIndex { sample: s, event: e });
            }
            prev = t;
        }
    }
    Ok(())
}

impl EventData {
    /// Check order, indices and the time window.
    ///
    /// # Errors
    ///
    /// [`NirDataError::BadNumber`], [`NirDataError::BadShape`], [`NirDataError::BadTime`],
    /// [`NirDataError::BadIndex`].
    pub fn validate(&self) -> Result<(), NirDataError> {
        check_events(&self.samples, self.neurons, self.t_max)
    }

    /// Events to a binary grid. `step = floor(time / dt)`. A time that falls on or past the last
    /// bin is refused.
    ///
    /// # Errors
    ///
    /// [`NirDataError::BadNumber`], [`NirDataError::BadTime`], and the errors of [`EventData::validate`].
    pub fn to_time_gridded(&self, dt: f64, steps: usize) -> Result<TimeGridded, NirDataError> {
        self.validate()?;
        if !(dt.is_finite() && dt > 0.0) || steps == 0 {
            return Err(NirDataError::BadNumber { what: "dt" });
        }
        let mut data = vec![0.0; self.samples.len() * steps * self.neurons];
        for (s, sample) in self.samples.iter().enumerate() {
            for (&i, &t) in sample.idx.iter().zip(&sample.time) {
                let step = (t / dt).floor();
                if !(step >= 0.0 && step < steps as f64) {
                    return Err(NirDataError::BadTime { sample: s, event: 0 });
                }
                let step = step as usize;
                let at = (s * steps + step) * self.neurons + i as usize;
                data[at] = 1.0;
            }
        }
        TimeGridded::try_new(data, self.samples.len(), steps, self.neurons, dt)
    }
}

impl ValuedEventData {
    /// Check that values line up with events.
    ///
    /// # Errors
    ///
    /// [`NirDataError::BadShape`] and the errors of [`EventData::validate`].
    pub fn validate(&self) -> Result<(), NirDataError> {
        check_events(&self.samples, self.neurons, self.t_max)?;
        if self.value.len() != self.samples.len() {
            return Err(NirDataError::BadShape {
                got: self.value.len(),
                want: self.samples.len(),
            });
        }
        for (sample, values) in self.samples.iter().zip(&self.value) {
            if values.len() != sample.time.len() || values.iter().any(|v| !v.is_finite()) {
                return Err(NirDataError::BadShape {
                    got: values.len(),
                    want: sample.time.len(),
                });
            }
        }
        Ok(())
    }

    /// Valued events to a grid. A bin that receives two values keeps the later event. The pinned
    /// source defines this direction only.
    ///
    /// # Errors
    ///
    /// [`NirDataError::BadNumber`], [`NirDataError::BadTime`], and the errors of
    /// [`ValuedEventData::validate`].
    pub fn to_time_gridded(&self, dt: f64, steps: usize) -> Result<TimeGridded, NirDataError> {
        self.validate()?;
        if !(dt.is_finite() && dt > 0.0) || steps == 0 {
            return Err(NirDataError::BadNumber { what: "dt" });
        }
        let mut data = vec![0.0; self.samples.len() * steps * self.neurons];
        for (s, (sample, values)) in self.samples.iter().zip(&self.value).enumerate() {
            for (e, ((&i, &t), &v)) in
                sample.idx.iter().zip(&sample.time).zip(values).enumerate()
            {
                let step = (t / dt).floor();
                if !(step >= 0.0 && step < steps as f64) {
                    return Err(NirDataError::BadTime { sample: s, event: e });
                }
                let at = (s * steps + step as usize) * self.neurons + i as usize;
                data[at] = v;
            }
        }
        TimeGridded::try_new(data, self.samples.len(), steps, self.neurons, dt)
    }

    /// Refused. The pinned source does not define valued events back to binary events.
    ///
    /// # Errors
    ///
    /// [`NirDataError::Unsupported`] always.
    pub fn to_events(&self) -> Result<EventData, NirDataError> {
        Err(NirDataError::Unsupported { what: "ValuedEventData to EventData" })
    }
}

/// One named observable on one node.
#[derive(Debug, Clone, PartialEq)]
pub enum Observable {
    /// A dense grid.
    Gridded(TimeGridded),
    /// Binary events.
    Events(EventData),
    /// Valued events.
    Valued(ValuedEventData),
}

/// Observables attached to one graph node.
#[derive(Debug, Clone, PartialEq)]
pub struct NodeData {
    /// Node name, matching [`crate::nir::Named::name`].
    pub name: String,
    /// `(key, observable)` pairs. Keys are unique inside the node.
    pub observables: Vec<(String, Observable)>,
}

/// A data record for one [`Graph`].
#[derive(Debug, Clone, PartialEq)]
pub struct GraphData {
    /// Per-node observables.
    pub nodes: Vec<NodeData>,
}

impl GraphData {
    /// Every node name exists in `graph`, and no observable key repeats inside a node.
    ///
    /// # Errors
    ///
    /// [`NirDataError::UnknownNode`] or [`NirDataError::DuplicateKey`].
    pub fn validate(&self, graph: &Graph) -> Result<(), NirDataError> {
        for node in &self.nodes {
            if !graph.nodes.iter().any(|named| named.name == node.name) {
                return Err(NirDataError::UnknownNode { name: node.name.clone() });
            }
            let mut seen = Vec::new();
            for (key, _) in &node.observables {
                if key.is_empty() || seen.iter().any(|k: &String| k == key) {
                    return Err(NirDataError::DuplicateKey {
                        node: node.name.clone(),
                        key: key.clone(),
                    });
                }
                seen.push(key.clone());
            }
        }
        Ok(())
    }
}
