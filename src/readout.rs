//! Closed-loop maps between spike counts, torque, observations, and current.
//!
//! Both maps are generic in their dimensions. A robot's lengths and clamps are arguments, not
//! constants of this module. The Panda fixture that supplies one such argument list lives in
//! `tools/fixtures/panda_nohand_4d038b3f.txt` and is checked by the tests below.
//!
//! # Determinism
//!
//! The same inputs produce the same bits. Summation walks channel 0, then channel 1, and inside
//! a channel walks input 0, then input 1. A product is a separate multiply. The add is a separate
//! `+`. Nothing here calls [`f64::mul_add`], and the source does not write `a * b + c` as one
//! expression, so the compiler is not asked to contract the two operations. Bias is the starting
//! value of a readout row. An encoder offset is added after the scale multiply.

use crate::reservoir::sha256;

/// Largest axis a map will allocate. This is a construction bound, not a robot's dimension.
pub const MAP_AXIS_MAX: usize = 4096;

/// Which side of the interval a raw value was moved to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Clamp {
    /// The raw value was inside the interval.
    None,
    /// The raw value was below `lo`.
    Lo,
    /// The raw value was above `hi`.
    Hi,
}

/// Why a map was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MapError {
    /// A weight, bias, scale, offset, bound, or observation was not finite.
    NonFinite {
        /// Which array.
        what: &'static str,
        /// Index in that array.
        index: usize,
    },
    /// Two arrays that must share a length do not.
    Dimension {
        /// Which array.
        what: &'static str,
        /// Length found.
        got: usize,
        /// Length required.
        want: usize,
    },
    /// `lo` is above `hi` on one channel.
    Bound {
        /// Channel index.
        channel: usize,
    },
    /// A spike count is above `2^53` and is not kept as an exact `f64`.
    InexactCount {
        /// Input index.
        index: usize,
        /// The count that was refused.
        count: u64,
    },
    /// The window was zero ticks.
    Window,
    /// A text record is not the canonical line sequence.
    Text(TextFault),
    /// A canonical line's value does not parse.
    Value,
}

/// What is wrong with a map's text, before any numeric check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextFault {
    /// A line's key is not part of this record.
    Unknown,
    /// A required line is absent.
    Missing,
    /// A key appears twice.
    Duplicate,
    /// A key is not in the header order.
    Reordered,
}

impl core::fmt::Display for MapError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NonFinite { what, index } => write!(f, "{what}[{index}] is not finite"),
            Self::Dimension { what, got, want } => {
                write!(f, "{what} has length {got} and the map requires {want}")
            }
            Self::Bound { channel } => write!(f, "channel {channel} has lo above hi"),
            Self::InexactCount { index, count } => {
                write!(f, "count[{index}] = {count} is above 2^53")
            }
            Self::Window => f.write_str("the readout window is zero ticks"),
            Self::Text(TextFault::Unknown) => f.write_str("a map line is not a canonical key"),
            Self::Text(TextFault::Missing) => f.write_str("a canonical map line is missing"),
            Self::Text(TextFault::Duplicate) => f.write_str("a canonical map line is repeated"),
            Self::Text(TextFault::Reordered) => f.write_str("a canonical map line is out of order"),
            Self::Value => f.write_str("a canonical map value does not parse"),
        }
    }
}

impl std::error::Error for MapError {}

/// Linear map from per-input spike counts over `window_ticks` to one torque per channel.
///
/// `tau[channel] = bias[channel] + Σ_j weight[channel, j] * count[j]`, then clamped to
/// `[lo, hi]`. The sum is in increasing `j`. The unit of `tau` is whatever unit the weights
/// carry. For the Panda fixture that unit is N·m.
#[derive(Debug, Clone, PartialEq)]
pub struct SpikeReadout {
    window_ticks: u64,
    inputs: usize,
    weights: Vec<f64>,
    bias: Vec<f64>,
    lo: Vec<f64>,
    hi: Vec<f64>,
}

/// One evaluation of [`SpikeReadout`], including the pre-clamp value and which side fired.
#[derive(Debug, Clone, PartialEq)]
pub struct ReadoutTrace {
    /// Window the counts were gathered over, in ticks.
    pub window_ticks: u64,
    /// Counts in input order.
    pub counts: Vec<u64>,
    /// Row sums before the clamp.
    pub raw: Vec<f64>,
    /// Clamped outputs.
    pub output: Vec<f64>,
    /// Which side of the interval each channel used.
    pub clamp: Vec<Clamp>,
}

/// Affine map from an observation to input current, in amperes, then a per-channel clamp.
///
/// `ampere[channel] = scale[channel] * observation[channel] + offset[channel]`. The multiply
/// happens first. The add is the next statement.
#[derive(Debug, Clone, PartialEq)]
pub struct CurrentEncoder {
    scale: Vec<f64>,
    offset: Vec<f64>,
    lo: Vec<f64>,
    hi: Vec<f64>,
}

/// One evaluation of [`CurrentEncoder`].
#[derive(Debug, Clone, PartialEq)]
pub struct EncoderTrace {
    /// Observations in channel order.
    pub observation: Vec<f64>,
    /// Affine results before the clamp.
    pub raw: Vec<f64>,
    /// Clamped currents, amperes.
    pub output: Vec<f64>,
    /// Which side of the interval each channel used.
    pub clamp: Vec<Clamp>,
}

impl SpikeReadout {
    /// Build a readout. `weights` is row-major, one row per channel.
    ///
    /// # Errors
    ///
    /// [`MapError::Window`] when `window_ticks` is 0.
    /// [`MapError::Dimension`] when the slices disagree or an axis exceeds [`MAP_AXIS_MAX`].
    /// [`MapError::NonFinite`] when a weight, bias, or bound is not finite.
    /// [`MapError::Bound`] when `lo > hi` on a channel.
    pub fn try_new(
        window_ticks: u64,
        inputs: usize,
        weights: &[f64],
        bias: &[f64],
        lo: &[f64],
        hi: &[f64],
    ) -> Result<Self, MapError> {
        if window_ticks == 0 {
            return Err(MapError::Window);
        }
        let channels = bias.len();
        check_axis(channels, "channels")?;
        check_axis(inputs, "inputs")?;
        let cells = channels.checked_mul(inputs).ok_or(MapError::Dimension {
            what: "weights",
            got: weights.len(),
            want: 0,
        })?;
        if weights.len() != cells {
            return Err(MapError::Dimension { what: "weights", got: weights.len(), want: cells });
        }
        same_len("bias", bias.len(), channels)?;
        same_len("lo", lo.len(), channels)?;
        same_len("hi", hi.len(), channels)?;
        finite("weight", weights)?;
        finite("bias", bias)?;
        finite("lo", lo)?;
        finite("hi", hi)?;
        for channel in 0..channels {
            if lo[channel] > hi[channel] {
                return Err(MapError::Bound { channel });
            }
        }
        Ok(Self {
            window_ticks,
            inputs,
            weights: weights.to_vec(),
            bias: bias.to_vec(),
            lo: lo.to_vec(),
            hi: hi.to_vec(),
        })
    }

    /// Ticks the counts are defined over.
    #[must_use]
    pub fn window_ticks(&self) -> u64 {
        self.window_ticks
    }

    /// Channels of torque.
    #[must_use]
    pub fn channels(&self) -> usize {
        self.bias.len()
    }

    /// Spike-count inputs.
    #[must_use]
    pub fn inputs(&self) -> usize {
        self.inputs
    }

    /// Map counts to clamped outputs and record the clamp.
    ///
    /// # Errors
    ///
    /// [`MapError::Dimension`] when `counts.len()` is not the input count.
    /// [`MapError::InexactCount`] when a count is above `2^53`.
    pub fn apply(&self, counts: &[u64]) -> Result<ReadoutTrace, MapError> {
        if counts.len() != self.inputs {
            return Err(MapError::Dimension {
                what: "counts",
                got: counts.len(),
                want: self.inputs,
            });
        }
        let channels = self.channels();
        let mut raw = Vec::with_capacity(channels);
        let mut output = Vec::with_capacity(channels);
        let mut clamp = Vec::with_capacity(channels);
        for channel in 0..channels {
            let mut acc = self.bias[channel];
            for input in 0..self.inputs {
                let weight = self.weights[channel * self.inputs + input];
                let count_as_f64 = exact_count(input, counts[input])?;
                let term = weight * count_as_f64;
                acc += term;
            }
            let (clamped, side) = clamp_to(acc, self.lo[channel], self.hi[channel]);
            raw.push(acc);
            output.push(clamped);
            clamp.push(side);
        }
        Ok(ReadoutTrace {
            window_ticks: self.window_ticks,
            counts: counts.to_vec(),
            raw,
            output,
            clamp,
        })
    }

    /// Canonical text of the map, not of one evaluation.
    #[must_use]
    pub fn to_text(&self) -> String {
        let mut out = format!(
            "ferromorphic-readout 1\nwindow {}\nchannels {}\ninputs {}\n",
            self.window_ticks,
            self.channels(),
            self.inputs
        );
        for channel in 0..self.channels() {
            for input in 0..self.inputs {
                let bits = self.weights[channel * self.inputs + input].to_bits();
                out.push_str(&format!("weight {channel} {input} {bits:016x}\n"));
            }
        }
        push_channel_bits(&mut out, "bias", &self.bias);
        push_channel_bits(&mut out, "lo", &self.lo);
        push_channel_bits(&mut out, "hi", &self.hi);
        out
    }

    /// Parse [`SpikeReadout::to_text`].
    ///
    /// # Errors
    ///
    /// [`MapError::Text`] for a missing, duplicate, unknown, or reordered line.
    /// [`MapError::Value`] for a value that does not parse.
    /// The variants of [`SpikeReadout::try_new`] for a well-formed record the constructor refuses.
    pub fn from_text(text: &str) -> Result<Self, MapError> {
        let lines: Vec<&str> = text.lines().collect();
        let mut at = 0usize;
        expect_header(&lines, &mut at, "ferromorphic-readout 1")?;
        let window_ticks: u64 = field(&lines, &mut at, &["window", "channels", "inputs"], "window")?
            .parse()
            .map_err(|_| MapError::Value)?;
        let channels: usize = field(&lines, &mut at, &["channels", "inputs"], "channels")?
            .parse()
            .map_err(|_| MapError::Value)?;
        let inputs: usize = field(&lines, &mut at, &["inputs"], "inputs")?
            .parse()
            .map_err(|_| MapError::Value)?;
        check_axis(channels, "channels")?;
        check_axis(inputs, "inputs")?;
        let mut weights = Vec::with_capacity(channels * inputs);
        for channel in 0..channels {
            for input in 0..inputs {
                let rest = field(&lines, &mut at, &["weight", "bias", "lo", "hi"], "weight")?;
                let mut parts = rest.split(' ');
                let got_c: usize = parts.next().ok_or(MapError::Value)?.parse().map_err(|_| MapError::Value)?;
                let got_i: usize = parts.next().ok_or(MapError::Value)?.parse().map_err(|_| MapError::Value)?;
                let hex = parts.next().ok_or(MapError::Value)?;
                if parts.next().is_some() || got_c != channel || got_i != input {
                    return Err(MapError::Text(TextFault::Reordered));
                }
                weights.push(parse_bits(hex)?);
            }
        }
        let bias = channel_bits(&lines, &mut at, channels, "bias", &["bias", "lo", "hi"])?;
        let lo = channel_bits(&lines, &mut at, channels, "lo", &["lo", "hi"])?;
        let hi = channel_bits(&lines, &mut at, channels, "hi", &["hi"])?;
        if at != lines.len() {
            return Err(MapError::Text(TextFault::Unknown));
        }
        Self::try_new(window_ticks, inputs, &weights, &bias, &lo, &hi)
    }
}

impl ReadoutTrace {
    /// Canonical text of one evaluation.
    #[must_use]
    pub fn to_text(&self) -> String {
        let mut out = format!(
            "ferromorphic-readout-trace 1\nwindow {}\nchannels {}\ninputs {}\n",
            self.window_ticks,
            self.output.len(),
            self.counts.len()
        );
        for (index, count) in self.counts.iter().enumerate() {
            out.push_str(&format!("count {index} {count}\n"));
        }
        push_channel_bits(&mut out, "raw", &self.raw);
        push_channel_bits(&mut out, "output", &self.output);
        for (index, side) in self.clamp.iter().enumerate() {
            out.push_str(&format!("clamp {index} {}\n", clamp_word(*side)));
        }
        out
    }

    /// SHA-256 of [`ReadoutTrace::to_text`], lowercase hex.
    #[must_use]
    pub fn sha256_hex(&self) -> String {
        hex_sha(&self.to_text())
    }
}

impl CurrentEncoder {
    /// Build an encoder. Every slice is one entry per channel, in observation order.
    ///
    /// # Errors
    ///
    /// [`MapError::Dimension`] when the slices disagree or the channel count exceeds
    /// [`MAP_AXIS_MAX`].
    /// [`MapError::NonFinite`] when a scale, offset, or bound is not finite.
    /// [`MapError::Bound`] when `lo > hi` on a channel.
    pub fn try_new(scale: &[f64], offset: &[f64], lo: &[f64], hi: &[f64]) -> Result<Self, MapError> {
        let channels = scale.len();
        check_axis(channels, "channels")?;
        same_len("offset", offset.len(), channels)?;
        same_len("lo", lo.len(), channels)?;
        same_len("hi", hi.len(), channels)?;
        finite("scale", scale)?;
        finite("offset", offset)?;
        finite("lo", lo)?;
        finite("hi", hi)?;
        for channel in 0..channels {
            if lo[channel] > hi[channel] {
                return Err(MapError::Bound { channel });
            }
        }
        Ok(Self {
            scale: scale.to_vec(),
            offset: offset.to_vec(),
            lo: lo.to_vec(),
            hi: hi.to_vec(),
        })
    }

    /// Observation channels.
    #[must_use]
    pub fn channels(&self) -> usize {
        self.scale.len()
    }

    /// Map an observation to clamped amperes.
    ///
    /// # Errors
    ///
    /// [`MapError::Dimension`] when `observation.len()` is not the channel count.
    /// [`MapError::NonFinite`] when an observation entry is not finite.
    pub fn apply(&self, observation: &[f64]) -> Result<EncoderTrace, MapError> {
        if observation.len() != self.channels() {
            return Err(MapError::Dimension {
                what: "observation",
                got: observation.len(),
                want: self.channels(),
            });
        }
        finite("observation", observation)?;
        let mut raw = Vec::with_capacity(observation.len());
        let mut output = Vec::with_capacity(observation.len());
        let mut clamp = Vec::with_capacity(observation.len());
        for channel in 0..self.channels() {
            let product = self.scale[channel] * observation[channel];
            let amperes = product + self.offset[channel];
            let (clamped, side) = clamp_to(amperes, self.lo[channel], self.hi[channel]);
            raw.push(amperes);
            output.push(clamped);
            clamp.push(side);
        }
        Ok(EncoderTrace { observation: observation.to_vec(), raw, output, clamp })
    }

    /// Canonical text of the map.
    #[must_use]
    pub fn to_text(&self) -> String {
        let mut out = format!("ferromorphic-current 1\nchannels {}\n", self.channels());
        push_channel_bits(&mut out, "scale", &self.scale);
        push_channel_bits(&mut out, "offset", &self.offset);
        push_channel_bits(&mut out, "lo", &self.lo);
        push_channel_bits(&mut out, "hi", &self.hi);
        out
    }

    /// Parse [`CurrentEncoder::to_text`].
    ///
    /// # Errors
    ///
    /// [`MapError::Text`] for a missing, duplicate, unknown, or reordered line.
    /// [`MapError::Value`] for a value that does not parse.
    /// The variants of [`CurrentEncoder::try_new`] for a well-formed record the constructor refuses.
    pub fn from_text(text: &str) -> Result<Self, MapError> {
        let lines: Vec<&str> = text.lines().collect();
        let mut at = 0usize;
        expect_header(&lines, &mut at, "ferromorphic-current 1")?;
        let channels: usize = field(&lines, &mut at, &["channels"], "channels")?
            .parse()
            .map_err(|_| MapError::Value)?;
        check_axis(channels, "channels")?;
        let scale = channel_bits(&lines, &mut at, channels, "scale", &["scale", "offset", "lo", "hi"])?;
        let offset = channel_bits(&lines, &mut at, channels, "offset", &["offset", "lo", "hi"])?;
        let lo = channel_bits(&lines, &mut at, channels, "lo", &["lo", "hi"])?;
        let hi = channel_bits(&lines, &mut at, channels, "hi", &["hi"])?;
        if at != lines.len() {
            return Err(MapError::Text(TextFault::Unknown));
        }
        Self::try_new(&scale, &offset, &lo, &hi)
    }
}

impl EncoderTrace {
    /// Canonical text of one evaluation.
    #[must_use]
    pub fn to_text(&self) -> String {
        let mut out = format!("ferromorphic-current-trace 1\nchannels {}\n", self.output.len());
        push_channel_bits(&mut out, "obs", &self.observation);
        push_channel_bits(&mut out, "raw", &self.raw);
        push_channel_bits(&mut out, "output", &self.output);
        for (index, side) in self.clamp.iter().enumerate() {
            out.push_str(&format!("clamp {index} {}\n", clamp_word(*side)));
        }
        out
    }

    /// SHA-256 of [`EncoderTrace::to_text`], lowercase hex.
    #[must_use]
    pub fn sha256_hex(&self) -> String {
        hex_sha(&self.to_text())
    }
}

fn clamp_to(raw: f64, lo: f64, hi: f64) -> (f64, Clamp) {
    if raw < lo {
        (lo, Clamp::Lo)
    } else if raw > hi {
        (hi, Clamp::Hi)
    } else {
        (raw, Clamp::None)
    }
}

fn clamp_word(side: Clamp) -> &'static str {
    match side {
        Clamp::None => "none",
        Clamp::Lo => "lo",
        Clamp::Hi => "hi",
    }
}

fn exact_count(index: usize, count: u64) -> Result<f64, MapError> {
    if count > (1u64 << 53) {
        return Err(MapError::InexactCount { index, count });
    }
    Ok(count as f64)
}

fn check_axis(n: usize, what: &'static str) -> Result<(), MapError> {
    if n == 0 || n > MAP_AXIS_MAX {
        return Err(MapError::Dimension { what, got: n, want: MAP_AXIS_MAX });
    }
    Ok(())
}

fn same_len(what: &'static str, got: usize, want: usize) -> Result<(), MapError> {
    if got != want {
        return Err(MapError::Dimension { what, got, want });
    }
    Ok(())
}

fn finite(what: &'static str, values: &[f64]) -> Result<(), MapError> {
    for (index, value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(MapError::NonFinite { what, index });
        }
    }
    Ok(())
}

fn push_channel_bits(out: &mut String, key: &str, values: &[f64]) {
    for (index, value) in values.iter().enumerate() {
        let bits = value.to_bits();
        out.push_str(&format!("{key} {index} {bits:016x}\n"));
    }
}

fn hex_sha(text: &str) -> String {
    let digest = sha256(text.as_bytes());
    let mut out = String::with_capacity(64);
    for byte in digest {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

fn parse_bits(hex: &str) -> Result<f64, MapError> {
    if hex.len() != 16 || !hex.bytes().all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()) {
        return Err(MapError::Value);
    }
    let bits = u64::from_str_radix(hex, 16).map_err(|_| MapError::Value)?;
    Ok(f64::from_bits(bits))
}

fn expect_header(lines: &[&str], at: &mut usize, header: &str) -> Result<(), MapError> {
    let line = lines.get(*at).copied().ok_or(MapError::Text(TextFault::Missing))?;
    *at += 1;
    if line == header {
        Ok(())
    } else {
        Err(MapError::Text(TextFault::Unknown))
    }
}

fn field<'a>(
    lines: &[&'a str],
    at: &mut usize,
    upcoming: &[&str],
    key: &str,
) -> Result<&'a str, MapError> {
    let line = lines.get(*at).copied().ok_or(MapError::Text(TextFault::Missing))?;
    let (got, rest) = line.split_once(' ').ok_or(MapError::Text(TextFault::Unknown))?;
    if rest.is_empty() {
        return Err(MapError::Value);
    }
    if got == key {
        *at += 1;
        return Ok(rest);
    }
    if upcoming.contains(&got) {
        return Err(MapError::Text(TextFault::Reordered));
    }
    Err(MapError::Text(if lines[..*at].iter().any(|seen| seen.starts_with(got)) {
        TextFault::Duplicate
    } else {
        TextFault::Unknown
    }))
}

fn channel_bits(
    lines: &[&str],
    at: &mut usize,
    channels: usize,
    key: &str,
    upcoming: &[&str],
) -> Result<Vec<f64>, MapError> {
    let mut values = Vec::with_capacity(channels);
    for channel in 0..channels {
        let rest = field(lines, at, upcoming, key)?;
        let mut parts = rest.split(' ');
        let got: usize = parts.next().ok_or(MapError::Value)?.parse().map_err(|_| MapError::Value)?;
        let hex = parts.next().ok_or(MapError::Value)?;
        if parts.next().is_some() || got != channel {
            return Err(MapError::Text(TextFault::Reordered));
        }
        values.push(parse_bits(hex)?);
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::{CurrentEncoder, MapError, SpikeReadout, TextFault};

    fn readout() -> SpikeReadout {
        SpikeReadout::try_new(4, 2, &[0.5, -0.25, 1.0, 0.0], &[0.1, -0.1], &[-1.0, -0.5], &[1.0, 0.5])
            .unwrap()
    }

    #[test]
    fn the_readout_sums_in_index_order_and_records_the_clamp() {
        let map = readout();
        let trace = map.apply(&[2, 3]).unwrap();
        let acc0 = (0.1f64 + 0.5 * 2.0) + (-0.25 * 3.0);
        assert_eq!(trace.raw[0].to_bits(), acc0.to_bits());
        assert_eq!(trace.output[0], acc0);
        assert_eq!(trace.clamp[0], super::Clamp::None);
        let acc1 = (-0.1f64 + 1.0 * 2.0) + (0.0 * 3.0);
        assert!(acc1 > 0.5);
        assert_eq!(trace.output[1], 0.5);
        assert_eq!(trace.clamp[1], super::Clamp::Hi);
        assert_eq!(SpikeReadout::from_text(&map.to_text()).unwrap(), map);
        assert_eq!(
            trace.sha256_hex(),
            "9725e17835446423e762918c21af40c6fc26cb8cd6d702c8764eb1db4790233c"
        );
    }

    #[test]
    fn the_encoder_multiplies_then_adds_and_clamps() {
        let map = CurrentEncoder::try_new(&[2.0, -1.0], &[0.25, 0.0], &[-1.0, -0.5], &[1.0, 0.4]).unwrap();
        let trace = map.apply(&[0.5, 1.0]).unwrap();
        let product = 2.0f64 * 0.5;
        let amperes = product + 0.25;
        assert_eq!(trace.raw[0].to_bits(), amperes.to_bits());
        assert_eq!(trace.output[0], 1.0);
        assert_eq!(trace.clamp[0], super::Clamp::Hi);
        assert_eq!(trace.output[1], -0.5);
        assert_eq!(trace.clamp[1], super::Clamp::Lo);
        assert_eq!(CurrentEncoder::from_text(&map.to_text()).unwrap(), map);
        assert_eq!(
            trace.sha256_hex(),
            "789e251cca549828c64075a0a3552166e3376198d782611053c79209f3f4ccfd"
        );
    }

    #[test]
    fn construction_refuses_a_non_finite_a_bad_bound_and_a_bad_line() {
        assert!(matches!(
            SpikeReadout::try_new(1, 1, &[f64::NAN], &[0.0], &[0.0], &[1.0]),
            Err(MapError::NonFinite { what: "weight", index: 0 })
        ));
        assert!(matches!(
            SpikeReadout::try_new(0, 1, &[0.0], &[0.0], &[0.0], &[1.0]),
            Err(MapError::Window)
        ));
        assert!(matches!(
            CurrentEncoder::try_new(&[1.0], &[0.0], &[1.0], &[0.0]),
            Err(MapError::Bound { channel: 0 })
        ));
        assert!(matches!(
            readout().apply(&[1]),
            Err(MapError::Dimension { what: "counts", .. })
        ));
        assert!(matches!(
            readout().apply(&[(1u64 << 53) + 1, 0]),
            Err(MapError::InexactCount { index: 0, .. })
        ));
        let text = readout().to_text();
        let swapped = text.replacen("window 4\nchannels 2", "channels 2\nwindow 4", 1);
        assert_eq!(SpikeReadout::from_text(&swapped).unwrap_err(), MapError::Text(TextFault::Reordered));
        let duplicated = text.replacen("window 4\n", "window 4\nwindow 4\n", 1);
        assert_eq!(
            SpikeReadout::from_text(&duplicated).unwrap_err(),
            MapError::Text(TextFault::Duplicate)
        );
        let extra = format!("{text}note 1\n");
        assert_eq!(SpikeReadout::from_text(&extra).unwrap_err(), MapError::Text(TextFault::Unknown));
        let truncated = text.lines().take(2).collect::<Vec<_>>().join("\n") + "\n";
        assert_eq!(
            SpikeReadout::from_text(&truncated).unwrap_err(),
            MapError::Text(TextFault::Missing)
        );
    }

    #[test]
    fn the_panda_fixture_sets_torque_and_position_clamps() {
        let text = include_str!("../tools/fixtures/panda_nohand_4d038b3f.txt");
        let mut nq = 0usize;
        let mut torque_lo = Vec::new();
        let mut torque_hi = Vec::new();
        let mut pos_lo = Vec::new();
        let mut pos_hi = Vec::new();
        for line in text.lines() {
            if line.starts_with('#') || line.is_empty() {
                continue;
            }
            let mut parts = line.split_whitespace();
            let key = parts.next().unwrap();
            if key == "nq" {
                nq = parts.next().unwrap().parse().unwrap();
                continue;
            }
            if key == "joint" {
                let _index: usize = parts.next().unwrap().parse().unwrap();
                pos_lo.push(parts.next().unwrap().parse::<f64>().unwrap());
                pos_hi.push(parts.next().unwrap().parse::<f64>().unwrap());
                torque_lo.push(parts.next().unwrap().parse::<f64>().unwrap());
                torque_hi.push(parts.next().unwrap().parse::<f64>().unwrap());
            }
        }
        assert_eq!(nq, 7);
        assert_eq!(pos_lo.len(), 7);
        let weights = vec![100.0; 7];
        let readout = SpikeReadout::try_new(1, 1, &weights, &[0.0; 7], &torque_lo, &torque_hi).unwrap();
        let trace = readout.apply(&[1]).unwrap();
        assert_eq!(trace.output, vec![87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0]);
        assert!(trace.clamp.iter().all(|side| *side == super::Clamp::Hi));
        let scale = vec![1.0; 7];
        let encoder = CurrentEncoder::try_new(&scale, &[0.0; 7], &pos_lo, &pos_hi).unwrap();
        let mut observation = vec![0.0; 7];
        observation[0] = 3.0;
        observation[3] = 0.0;
        observation[5] = -1.0;
        let encoded = encoder.apply(&observation).unwrap();
        assert_eq!(encoded.output[0], 2.8973);
        assert_eq!(encoded.output[3], -0.0698);
        assert_eq!(encoded.output[5], -0.0175);
        assert_eq!(encoded.clamp[0], super::Clamp::Hi);
        assert_eq!(encoded.clamp[3], super::Clamp::Hi);
        assert_eq!(encoded.clamp[5], super::Clamp::Lo);
    }
}
