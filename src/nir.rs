//! `NIR`, the neuromorphic intermediate representation: the graph model a trained spiking network
//! travels in, a text serialisation of it that round-trips exactly, and a bridge into this crate.
//!
//! Pedersen et al., "Neuromorphic intermediate representation: A unified instruction set for
//! interoperable brain-inspired computing", *Nature Communications* 15:8122, 2024.
//!
//! # What the problem is
//!
//! A spiking network trained in `snnTorch` and a chip programmed through `Lava` share no file
//! format, no layer vocabulary and no discretisation. Every framework-to-hardware pair is its own
//! exporter, so a field with `N` frameworks and `M` targets writes `N * M` of them — and in
//! practice writes about six of them, badly, and the rest never get written at all. That is the
//! reason a model published with a paper usually cannot be run on the hardware the paper is about.
//!
//! `NIR`'s answer is the standard compiler answer: put an intermediate representation in the
//! middle, and the bill falls to `N + M`. What makes the neuromorphic version interesting is
//! **what the IR is made of**. It is not a list of layers with a timestep baked in. It is a graph
//! whose nodes are **continuous-time dynamical primitives** — `tau * dv/dt = (v_leak - v) + R * I`,
//! written as an ordinary differential equation, with no `dt` anywhere in the file. The target
//! chooses the discretisation, because the target is the only party that knows its own clock. A
//! mixed-signal chip integrates that equation in physics; a digital core integrates it in fixed
//! point at 1 ms; a `PyTorch` backend integrates it by exponential Euler at whatever step the
//! training used. They are all reading the same file.
//!
//! # What it buys, and what it costs
//!
//! It buys transport. The paper's support table lists on the order of ten frameworks, simulators
//! and hardware platforms — `Nengo`, `Norse`, `Rockpool`, `Sinabs`, `snnTorch`, `Spyx`, `Lava`,
//! `SpiNNaker2` and `Xylo` are among the names it carries, though the membership changes with each
//! release and **this implementation did not re-verify that table**.
//!
//! It costs two things, and both are worth saying beside the figure rather than under it.
//!
//! **The node set is a floor, not a ceiling.** A mechanism that is not in the vocabulary cannot
//! cross. There is no node for a learning rule, none for a dendritic nonlinearity, none for a
//! stochastic synapse. What travels is the trained forward model.
//!
//! **Transportable is not identical.** The paper's own cross-platform comparison is the honest
//! caveat: the same graph on different backends does not produce the same output, because each
//! backend discretises those differential equations its own way and a spike is a threshold crossing
//! — the one place where a small numerical difference becomes a large observable one. An IR makes a
//! model *portable*; nobody in this field currently claims *bit-identity* across fabrics, and this
//! implementation did not re-run the paper's comparison to put a number on the gap.
//!
//! # What this module is, and what it is not
//!
//! `NIR`'s reference serialisation is `HDF5`. This crate has zero dependencies and will keep them,
//! so **this module does not read or write `.nir` files** — saying otherwise would be the useful
//! lie. What it has instead is:
//!
//! - an in-memory graph over seventeen `NIR` node kinds ([`Node`]),
//! - a line-oriented **text** serialisation that round-trips the graph *exactly*, every float
//!   included ([`Graph::to_text`], [`Graph::from_text`]),
//! - validation that names what is wrong ([`Graph::validate`]),
//! - and a bridge to [`crate::net::Net`] + [`crate::neuron::Lif`] for the subset that maps
//!   ([`Graph::to_net`], [`Graph::from_net`]).
//!
//! ⚠ **The node set here was not re-verified against the reference implementation.** The naming
//! ladder `I -> IF` and `LI -> LIF` is complete, but `CubaLIF` has no `CubaLI` beside it — the
//! current-based leaky integrator with no threshold, the readout counterpart a systematic naming
//! scheme would predict, and the most likely gap in the seventeen. Whether `NIR` defines one was
//! not checked here, so what follows is what this module implements and not a claim about what
//! `NIR` contains.
//!
//! The node types are plain structs with public fields, so an `HDF5` reader living in a sibling
//! crate can populate them without this module changing. That is the intended division: the format
//! is somebody else's dependency, the graph is not.
//!
//! # Units
//!
//! `NIR`'s equations are dimensionally consistent when read in SI — `tau` in seconds, `r` in ohms,
//! `v_leak`, `v_threshold` and `v_reset` in volts, `delay` in seconds — and that is the reading
//! this module declares and the bridge assumes.
//!
//! ⚠ **A file in the wild is usually normalised instead.** A network trained in a surrogate-gradient
//! framework typically carries `v_threshold = 1`, `r = 1` and `tau` measured in timesteps, because
//! nothing in the training loop ever needed a volt. Such a graph is dimensionless; the bridge will
//! happily hand you neurons whose "volts" are those normalised units, and every energy figure
//! computed downstream inherits that. This implementation did not locate a statement in the `NIR`
//! specification that fixes the units, so the burden is on the person reporting the number.
//!
//! # What crosses the bridge
//!
//! | `NIR` node | `to_net` | `from_net` | lossless? |
//! |---|---|---|---|
//! | `LIF` | one neuron per element | yes | parameters yes; `NIR` has **no refractory period**, so [`crate::neuron::Lif::t_ref`] must be `0` |
//! | `Linear` | dense synapse block | yes, as one dense `n * n` node | weights **bit-exact**; exact zeros are dropped |
//! | `Affine` | only when every bias is exactly `0` | emitted as `Linear` | the bias is **not representable** |
//! | `Delay` | per-element tick delay | yes, when every synapse onto a neuron shares a delay | seconds to ticks is exact or refused |
//! | `Input`, `Output` | recorded in [`Conversion`] | emitted | they carry only a shape |
//! | `I`, `IF`, `LI`, `CubaLIF`, `Threshold`, `Scale`, `Conv1d`, `Conv2d`, `SumPool2d`, `AvgPool2d`, `Flatten` | **refused by name** | — | — |
//!
//! Refused by name, rather than converted approximately: a `CubaLIF` silently demoted to a `LIF`
//! is a network whose synaptic filter vanished, and it would still produce a plausible raster.
//!
//! ⚠ **A third loss, on the comparison itself.** `NIR` fires on `v > v_threshold`; this crate's
//! [`crate::neuron::Lif`] fires on `v >= v_th`. [`Graph::to_net`] copies the threshold across
//! unchanged rather than nudging it by an ulp, so the converted network uses the non-strict rule.
//! The two differ only on a membrane that lands **exactly** on its threshold — measure zero in
//! floating point, and not zero: a `LIF` with `r = 0` and `v_leak = v_threshold` rests on its
//! threshold, never fires under `NIR`, and fires on the first tick of the converted network.
//! `the_bridge_turns_a_strict_crossing_into_a_non_strict_one` pins that case, so the loss is a
//! disclosed one rather than a discovered one.
//!
//! # `CubaLIF` and the reduction that explains it
//!
//! `CubaLIF` is the current-based leaky integrate-and-fire pair,
//!
//! ```text
//! tau_syn * dI/dt = -I + w_in * u
//! tau_mem * dv/dt = (v_leak - v) + r * I
//! ```
//!
//! and its documented relationship to the plain `LIF` is a limit: **as `tau_syn` goes to zero the
//! synaptic current stops filtering**, `I` collapses onto `w_in * u`, and the membrane equation
//! becomes the `LIF`'s with resistance `r * w_in`. [`CubaLif::reduce_to_lif`] performs exactly that
//! substitution and refuses when `tau_syn` is not zero. [`CubaState`] integrates the pair with the
//! **exact propagator** of the linear system (Rotter & Diesmann, *Biological Cybernetics*
//! 81:381-402, 1999), which at `tau_syn = 0` degenerates — with no special case in the code — into
//! the exponential-Euler update [`crate::neuron::Lif`] already uses, bit for bit.
//!
//! # Example
//!
//! ```
//! use ferromorphic::nir::{Graph, Input, Lif, Linear, Node, Output, Rules};
//!
//! let mut g = Graph::new();
//! g.push("in", Node::Input(Input { shape: vec![2] }));
//! g.push("n", Node::Lif(Lif {
//!     shape: vec![2],
//!     tau: vec![20e-3; 2],
//!     r: vec![10e6; 2],
//!     v_leak: vec![-65e-3; 2],
//!     v_threshold: vec![-50e-3; 2],
//!     v_reset: vec![-65e-3; 2],
//! }));
//! g.push("w", Node::Linear(Linear { rows: 2, cols: 2, weight: vec![0.0, 1e-3, 2e-3, 0.0] }));
//! g.push("out", Node::Output(Output { shape: vec![2] }));
//! g.edge("in", "n");
//! g.edge("n", "w");
//! g.edge("w", "n"); // recurrent, which NIR allows and this validator allows by default
//! g.edge("n", "out");
//! g.validate(&Rules::default())?;
//!
//! // Text -> graph is the identity.
//! assert_eq!(Graph::from_text(&g.to_text())?, g);
//!
//! // And it becomes a runnable network: two neurons, two synapses (the exact zeros are dropped).
//! let c = g.to_net(1e-4)?;
//! assert_eq!(c.net.n, 2);
//! assert_eq!(c.net.n_syn, 2);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use crate::net::{Net, NetBuilder, NetError};
use core::fmt;

// ---------------------------------------------------------------------------------------------
// Node kinds
// ---------------------------------------------------------------------------------------------

/// Which `NIR` primitive a node is, without its parameters.
///
/// [`Kind::name`] returns the spelling the `NIR` specification uses (`LIF`, `CubaLIF`, `SumPool2d`),
/// which is also the token the text format writes, so a file is readable next to the paper.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// Graph boundary in: carries a shape and no dynamics.
    Input,
    /// Graph boundary out: carries a shape and no dynamics.
    Output,
    /// `y = W x + b`.
    Affine,
    /// `y = W x`, the bias-free `Affine`.
    Linear,
    /// One-dimensional convolution over a `(channels, length)` signal.
    Conv1d,
    /// Two-dimensional convolution over a `(channels, height, width)` signal.
    Conv2d,
    /// Two-dimensional pooling that **sums** its window, which is what a spike count wants.
    SumPool2d,
    /// Two-dimensional pooling that averages its window.
    AvgPool2d,
    /// Reshape that collapses a contiguous run of axes into one.
    Flatten,
    /// Ideal integrator: `dv/dt = r * I`, no leak and no threshold.
    I,
    /// Integrate-and-fire: an [`Kind::I`] with a threshold and a reset.
    If,
    /// Leaky integrator: `tau * dv/dt = (v_leak - v) + r * I`, no threshold.
    Li,
    /// Leaky integrate-and-fire: an [`Kind::Li`] with a threshold and a reset.
    Lif,
    /// Current-based leaky integrate-and-fire: a first-order synaptic filter in front of a
    /// [`Kind::Lif`].
    CubaLif,
    /// Emits a spike while its input exceeds a per-element threshold; carries no state.
    Threshold,
    /// Delays its input by a per-element number of **seconds**.
    Delay,
    /// Element-wise multiplication by a constant.
    Scale,
}

impl Kind {
    /// The `NIR` spelling, which is the token the text format uses.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Input => "Input",
            Self::Output => "Output",
            Self::Affine => "Affine",
            Self::Linear => "Linear",
            Self::Conv1d => "Conv1d",
            Self::Conv2d => "Conv2d",
            Self::SumPool2d => "SumPool2d",
            Self::AvgPool2d => "AvgPool2d",
            Self::Flatten => "Flatten",
            Self::I => "I",
            Self::If => "IF",
            Self::Li => "LI",
            Self::Lif => "LIF",
            Self::CubaLif => "CubaLIF",
            Self::Threshold => "Threshold",
            Self::Delay => "Delay",
            Self::Scale => "Scale",
        }
    }

    /// Parse a `NIR` spelling. `None` for anything not in the vocabulary — the reader refuses an
    /// unknown node rather than skipping it, because a skipped node leaves a graph that is smaller
    /// than the file and still runs.
    #[must_use]
    pub fn from_name(s: &str) -> Option<Self> {
        let k = match s {
            "Input" => Self::Input,
            "Output" => Self::Output,
            "Affine" => Self::Affine,
            "Linear" => Self::Linear,
            "Conv1d" => Self::Conv1d,
            "Conv2d" => Self::Conv2d,
            "SumPool2d" => Self::SumPool2d,
            "AvgPool2d" => Self::AvgPool2d,
            "Flatten" => Self::Flatten,
            "I" => Self::I,
            "IF" => Self::If,
            "LI" => Self::Li,
            "LIF" => Self::Lif,
            "CubaLIF" => Self::CubaLif,
            "Threshold" => Self::Threshold,
            "Delay" => Self::Delay,
            "Scale" => Self::Scale,
            _ => return None,
        };
        Some(k)
    }
}

impl fmt::Display for Kind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// What kind of signal travels on an edge.
///
/// ⚠ **`NIR` itself does not carry this distinction**: its type system is shapes, and a spike train
/// is an array of zeros and ones like any other. [`Signal`] is this crate's addition, checked
/// locally per edge and switchable off with [`Rules::check_signal_kind`], because the one thing it
/// catches — thresholding something that is already a spike train — is a modelling error that
/// produces a running network and a meaningless one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Signal {
    /// Binary events. Emitted by [`Kind::Lif`], [`Kind::If`], [`Kind::CubaLif`], [`Kind::Threshold`].
    Spikes,
    /// A real-valued signal — a current, a potential, a weighted sum.
    Continuous,
    /// Unconstrained: either the node passes through whatever it is given (pooling, `Flatten`,
    /// `Delay`, `Scale`) or it accepts both (every weight node, every integrator).
    Either,
}

impl fmt::Display for Signal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Spikes => "spikes",
            Self::Continuous => "continuous",
            Self::Either => "either",
        })
    }
}

// ---------------------------------------------------------------------------------------------
// Node parameter structs
// ---------------------------------------------------------------------------------------------

/// Graph input boundary.
#[derive(Debug, Clone, PartialEq)]
pub struct Input {
    /// Shape of the signal entering the graph, **without** a batch axis. Every dimension must be
    /// non-zero.
    pub shape: Vec<usize>,
}

/// Graph output boundary.
#[derive(Debug, Clone, PartialEq)]
pub struct Output {
    /// Shape of the signal leaving the graph, **without** a batch axis.
    pub shape: Vec<usize>,
}

/// `y = W x + b`.
#[derive(Debug, Clone, PartialEq)]
pub struct Affine {
    /// Output width: `W` has this many rows, and `bias` this many entries.
    pub rows: usize,
    /// Input width: `W` has this many columns.
    pub cols: usize,
    /// `W` in **row-major** order, `rows * cols` finite entries. Row `i` holds the weights that
    /// reach output `i`, so `weight[i * cols + j]` is the connection from input `j` to output `i` —
    /// the same orientation `NIR` and `PyTorch` use, and the one [`Graph::to_net`] relies on.
    pub weight: Vec<f64>,
    /// `b`, `rows` finite entries. **Not representable in [`crate::net::Net`]**, which has no
    /// per-neuron constant term; [`Graph::to_net`] accepts an `Affine` only when every entry here
    /// is exactly zero.
    pub bias: Vec<f64>,
}

/// `y = W x`: the bias-free `Affine`, and the only weight node that survives a round trip through
/// [`crate::net::Net`].
#[derive(Debug, Clone, PartialEq)]
pub struct Linear {
    /// Output width, in rows of `W`.
    pub rows: usize,
    /// Input width, in columns of `W`.
    pub cols: usize,
    /// `W` in row-major order, `rows * cols` finite entries, oriented as [`Affine::weight`].
    pub weight: Vec<f64>,
}

/// One-dimensional convolution.
///
/// Output length follows the published convolution arithmetic (Dumoulin & Visin, *A guide to
/// convolution arithmetic for deep learning*, `arXiv`:1603.07285, 2016; the same expression
/// `PyTorch` documents for `nn.Conv1d`):
/// `out = floor((length + 2*padding - dilation*(kernel - 1) - 1) / stride) + 1`.
#[derive(Debug, Clone, PartialEq)]
pub struct Conv1d {
    /// Input channel count.
    pub in_channels: usize,
    /// Output channel count, which is also the number of filters.
    pub out_channels: usize,
    /// Input signal length in samples.
    pub length: usize,
    /// Filter width in samples.
    pub kernel: usize,
    /// Step between filter applications, samples. Must be non-zero.
    pub stride: usize,
    /// Zeros added at **each** end before convolving, samples.
    pub padding: usize,
    /// Spacing between filter taps, samples; `1` is the ordinary dense filter.
    pub dilation: usize,
    /// Channel groups; must divide both channel counts. `1` is a full convolution, `in_channels`
    /// is depthwise.
    pub groups: usize,
    /// Filters in row-major `(out_channels, in_channels / groups, kernel)` order.
    pub weight: Vec<f64>,
    /// Per-output-channel bias, `out_channels` entries — or **empty**, which is how `NIR` encodes
    /// a convolution built with `bias=False`.
    pub bias: Vec<f64>,
}

/// Two-dimensional convolution, with the output shape computed per axis by the same published
/// formula as [`Conv1d`].
#[derive(Debug, Clone, PartialEq)]
pub struct Conv2d {
    /// Input channel count.
    pub in_channels: usize,
    /// Output channel count.
    pub out_channels: usize,
    /// Input spatial size as `[height, width]`.
    pub size: [usize; 2],
    /// Filter size as `[height, width]`.
    pub kernel: [usize; 2],
    /// Step per axis as `[height, width]`; neither may be zero.
    pub stride: [usize; 2],
    /// Zeros added at each end of each axis, `[height, width]`.
    pub padding: [usize; 2],
    /// Tap spacing per axis, `[height, width]`; `[1, 1]` is the dense filter.
    pub dilation: [usize; 2],
    /// Channel groups; must divide both channel counts.
    pub groups: usize,
    /// Filters in row-major `(out_channels, in_channels / groups, kernel_h, kernel_w)` order.
    pub weight: Vec<f64>,
    /// Per-output-channel bias, `out_channels` entries, or empty for none.
    pub bias: Vec<f64>,
}

/// Two-dimensional pooling, shared by [`Kind::SumPool2d`] and [`Kind::AvgPool2d`].
///
/// The two differ only by a constant: over the same window, `sum = avg * kernel_h * kernel_w`. They
/// are separate nodes in `NIR` because a spiking layer usually wants the **sum** — pooling spike
/// counts and then dividing throws away the quantity the next threshold is about to compare.
#[derive(Debug, Clone, PartialEq)]
pub struct Pool2d {
    /// Channel count; pooling never mixes channels, so this passes through unchanged.
    pub channels: usize,
    /// Input spatial size as `[height, width]`.
    pub size: [usize; 2],
    /// Window size as `[height, width]`.
    pub kernel: [usize; 2],
    /// Step per axis as `[height, width]`; neither may be zero.
    pub stride: [usize; 2],
    /// Zeros added at each end of each axis, `[height, width]`.
    pub padding: [usize; 2],
}

/// Collapse a contiguous run of axes into one.
///
/// ⚠ `NIR` inherits `PyTorch`'s convention in which axis `0` is the batch, so a file written by a
/// `PyTorch`-side exporter usually carries `start_dim = 1`. The shapes in this module carry **no
/// batch axis**, so [`Flatten::start_dim`] indexes [`Flatten::size`] directly and such a file needs
/// its dimensions decremented by one on the way in. This implementation did not locate a statement
/// in the `NIR` paper that fixes whether `input_type` includes the batch axis, so the convention is
/// stated here rather than assumed.
#[derive(Debug, Clone, PartialEq)]
pub struct Flatten {
    /// Input shape, at least one axis.
    pub size: Vec<usize>,
    /// First axis to collapse, indexing [`Flatten::size`].
    pub start_dim: usize,
    /// Last axis to collapse, **inclusive**; must be at least `start_dim` and inside the shape.
    pub end_dim: usize,
}

/// Ideal integrator: `dv/dt = r * I`. No leak, no threshold, no reset.
#[derive(Debug, Clone, PartialEq)]
pub struct I {
    /// Element shape; the product of its dimensions is the length of every parameter array below.
    pub shape: Vec<usize>,
    /// Input resistance per element, ohms.
    pub r: Vec<f64>,
}

/// Integrate-and-fire: [`I`] with a threshold and a reset, and no leak.
#[derive(Debug, Clone, PartialEq)]
pub struct If {
    /// Element shape; its product is the length of every parameter array below.
    pub shape: Vec<usize>,
    /// Input resistance per element, ohms.
    pub r: Vec<f64>,
    /// Firing threshold per element, volts. `NIR` spikes on `v > v_threshold`.
    pub v_threshold: Vec<f64>,
    /// Post-spike potential per element, volts.
    pub v_reset: Vec<f64>,
}

/// Leaky integrator: `tau * dv/dt = (v_leak - v) + r * I`. The sub-threshold membrane on its own,
/// which is what a readout layer usually wants.
#[derive(Debug, Clone, PartialEq)]
pub struct Li {
    /// Element shape; its product is the length of every parameter array below.
    pub shape: Vec<usize>,
    /// Membrane time constant per element, seconds. Must be strictly positive and finite.
    pub tau: Vec<f64>,
    /// Input resistance per element, ohms.
    pub r: Vec<f64>,
    /// Resting potential per element, volts.
    pub v_leak: Vec<f64>,
}

/// Leaky integrate-and-fire, the node that carries most of the field's models.
///
/// `tau * dv/dt = (v_leak - v) + r * I`, spike on `v > v_threshold`, then `v = v_reset`.
///
/// ⚠ **`NIR` version 1 hard-resets to zero**; later revisions of the specification carry an explicit
/// reset potential. This implementation carries [`Lif::v_reset`] explicitly, so a version-1 file is
/// reproduced by filling it with zeros. Where a reader's `NIR` release differs, this is the field
/// to check.
///
/// ⚠ **There is no refractory period in `NIR`.** [`crate::neuron::Lif`] has one, and
/// [`Graph::from_net`] therefore refuses a neuron whose `t_ref` is non-zero rather than dropping it.
#[derive(Debug, Clone, PartialEq)]
pub struct Lif {
    /// Element shape; its product is the length of every parameter array below.
    pub shape: Vec<usize>,
    /// Membrane time constant per element, seconds. Strictly positive and finite.
    pub tau: Vec<f64>,
    /// Input resistance per element, ohms.
    pub r: Vec<f64>,
    /// Resting potential per element, volts.
    pub v_leak: Vec<f64>,
    /// Firing threshold per element, volts.
    pub v_threshold: Vec<f64>,
    /// Post-spike potential per element, volts.
    pub v_reset: Vec<f64>,
}

/// Current-based leaky integrate-and-fire: a first-order synapse in front of a [`Lif`].
///
/// ```text
/// tau_syn * dI/dt = -I + w_in * u
/// tau_mem * dv/dt = (v_leak - v) + r * I
/// ```
///
/// The synaptic filter is what makes an arriving spike a **current that decays** rather than an
/// instantaneous jump in the membrane, which is the difference between a neuron that integrates
/// coincidences over a few milliseconds and one that only sees them within a single timestep.
///
/// [`CubaLif::reduce_to_lif`] is the documented relationship to the plain [`Lif`]: at
/// `tau_syn = 0` the filter is instantaneous, `I` equals `w_in * u`, and the pair collapses to a
/// `LIF` of resistance `r * w_in`.
#[derive(Debug, Clone, PartialEq)]
pub struct CubaLif {
    /// Element shape; its product is the length of every parameter array below.
    pub shape: Vec<usize>,
    /// Synaptic time constant per element, seconds. Finite and **non-negative**: exactly zero is
    /// legal and means the instantaneous-synapse limit, which is the `LIF` reduction.
    pub tau_syn: Vec<f64>,
    /// Membrane time constant per element, seconds. Strictly positive and finite.
    pub tau_mem: Vec<f64>,
    /// Input resistance per element, ohms.
    pub r: Vec<f64>,
    /// Resting potential per element, volts.
    pub v_leak: Vec<f64>,
    /// Firing threshold per element, volts.
    pub v_threshold: Vec<f64>,
    /// Post-spike potential per element, volts.
    pub v_reset: Vec<f64>,
    /// Gain applied to the input before the synaptic filter, dimensionless. It multiplies `r` in
    /// the reduction, which is why a graph can move gain between the two without changing anything.
    pub w_in: Vec<f64>,
}

/// Emits a spike while its input exceeds a per-element threshold. Stateless.
#[derive(Debug, Clone, PartialEq)]
pub struct Threshold {
    /// Element shape; its product is the length of [`Threshold::threshold`].
    pub shape: Vec<usize>,
    /// Comparison level per element, in the units of whatever arrives.
    pub threshold: Vec<f64>,
}

/// Delays its input by a per-element number of seconds.
///
/// Seconds, not ticks — a `NIR` file has no timestep in it. [`Graph::to_net`] converts to
/// [`crate::net::Net`]'s tick delays against the `dt` it is given, and **refuses** a delay that is
/// not a whole number of ticks rather than rounding one.
#[derive(Debug, Clone, PartialEq)]
pub struct Delay {
    /// Element shape; its product is the length of [`Delay::delay`].
    pub shape: Vec<usize>,
    /// Delay per element, seconds. Finite and non-negative.
    pub delay: Vec<f64>,
}

/// Element-wise multiplication by a constant.
#[derive(Debug, Clone, PartialEq)]
pub struct Scale {
    /// Element shape; its product is the length of [`Scale::scale`].
    pub shape: Vec<usize>,
    /// Multiplier per element, dimensionless.
    pub scale: Vec<f64>,
}

/// One `NIR` primitive with its parameters.
#[derive(Debug, Clone, PartialEq)]
pub enum Node {
    /// Graph boundary in.
    Input(Input),
    /// Graph boundary out.
    Output(Output),
    /// `y = W x + b`.
    Affine(Affine),
    /// `y = W x`.
    Linear(Linear),
    /// One-dimensional convolution.
    Conv1d(Conv1d),
    /// Two-dimensional convolution.
    Conv2d(Conv2d),
    /// Two-dimensional pooling that sums its window.
    SumPool2d(Pool2d),
    /// Two-dimensional pooling that averages its window.
    AvgPool2d(Pool2d),
    /// Axis-collapsing reshape.
    Flatten(Flatten),
    /// Ideal integrator.
    I(I),
    /// Integrate-and-fire.
    If(If),
    /// Leaky integrator.
    Li(Li),
    /// Leaky integrate-and-fire.
    Lif(Lif),
    /// Current-based leaky integrate-and-fire.
    CubaLif(CubaLif),
    /// Stateless threshold.
    Threshold(Threshold),
    /// Per-element delay in seconds.
    Delay(Delay),
    /// Element-wise constant multiply.
    Scale(Scale),
}

/// `floor((input + 2*padding - dilation*(kernel - 1) - 1) / stride) + 1`, saturating.
///
/// Returns `0` for a window that cannot be placed even once, and for a zero stride or kernel, so
/// that a shape query never panics on a malformed node; [`Graph::validate`] reports the malformed
/// hyperparameter first and the zero dimension second.
///
/// **Every** product here saturates, the doubling of the padding included: `2 * padding` written
/// plainly overflows for a padding past `usize::MAX / 2`, which panics in a debug build and wraps
/// in a release one — and a shape query that panics would make a malformed file a crash rather
/// than an error.
fn conv_dim(input: usize, padding: usize, dilation: usize, kernel: usize, stride: usize) -> usize {
    if stride == 0 || kernel == 0 {
        return 0;
    }
    let span = dilation.saturating_mul(kernel - 1).saturating_add(1);
    let padded = input.saturating_add(padding.saturating_mul(2));
    if padded < span {
        return 0;
    }
    (padded - span) / stride + 1
}

/// How many elements a shape describes, or `None` when that count does not fit in a `usize`.
///
/// A shape read from a file can name more elements than the machine can address — `[2^32, 2^32]`
/// is four tokens — and the plain product of it panics in a debug build and **wraps to zero** in a
/// release one. A wrapped zero is the dangerous half: a node declaring `2^64` elements would then
/// pass [`Graph::validate`] with empty parameter arrays, because zero entries is exactly what a
/// zero count expects. Every caller with an error channel uses this form.
fn checked_product(shape: &[usize]) -> Option<usize> {
    shape.iter().copied().try_fold(1usize, usize::checked_mul)
}

/// The same product, saturating at `usize::MAX`, for the shape queries that have no error channel.
///
/// [`Node::output_shape`] promises never to panic on a malformed node, so it cannot use the
/// checked form. Saturating leaves the answer wrong-but-enormous rather than wrong-and-zero, and
/// an enormous axis is what [`Graph::validate`] then refuses through [`checked_product`].
fn product(shape: &[usize]) -> usize {
    shape.iter().copied().fold(1usize, usize::saturating_mul)
}

impl Node {
    /// Which primitive this is.
    #[must_use]
    pub fn kind(&self) -> Kind {
        match self {
            Self::Input(_) => Kind::Input,
            Self::Output(_) => Kind::Output,
            Self::Affine(_) => Kind::Affine,
            Self::Linear(_) => Kind::Linear,
            Self::Conv1d(_) => Kind::Conv1d,
            Self::Conv2d(_) => Kind::Conv2d,
            Self::SumPool2d(_) => Kind::SumPool2d,
            Self::AvgPool2d(_) => Kind::AvgPool2d,
            Self::Flatten(_) => Kind::Flatten,
            Self::I(_) => Kind::I,
            Self::If(_) => Kind::If,
            Self::Li(_) => Kind::Li,
            Self::Lif(_) => Kind::Lif,
            Self::CubaLif(_) => Kind::CubaLif,
            Self::Threshold(_) => Kind::Threshold,
            Self::Delay(_) => Kind::Delay,
            Self::Scale(_) => Kind::Scale,
        }
    }

    /// The shape this node requires on its input, or `None` for [`Kind::Input`], which has no
    /// input port at all.
    ///
    /// Every node declares its own input shape rather than inferring one from the graph. That is a
    /// deliberate departure from a compiler's usual shape inference and it is what makes validation
    /// **local**: a recurrent graph has no topological order to infer along, and `NIR` nodes carry
    /// `input_type` for the same reason.
    #[must_use]
    pub fn input_shape(&self) -> Option<Vec<usize>> {
        let s = match self {
            Self::Input(_) => return None,
            Self::Output(x) => x.shape.clone(),
            Self::Affine(x) => vec![x.cols],
            Self::Linear(x) => vec![x.cols],
            Self::Conv1d(x) => vec![x.in_channels, x.length],
            Self::Conv2d(x) => vec![x.in_channels, x.size[0], x.size[1]],
            Self::SumPool2d(x) | Self::AvgPool2d(x) => vec![x.channels, x.size[0], x.size[1]],
            Self::Flatten(x) => x.size.clone(),
            Self::I(x) => x.shape.clone(),
            Self::If(x) => x.shape.clone(),
            Self::Li(x) => x.shape.clone(),
            Self::Lif(x) => x.shape.clone(),
            Self::CubaLif(x) => x.shape.clone(),
            Self::Threshold(x) => x.shape.clone(),
            Self::Delay(x) => x.shape.clone(),
            Self::Scale(x) => x.shape.clone(),
        };
        Some(s)
    }

    /// The shape this node emits, or `None` for [`Kind::Output`], which has no output port.
    ///
    /// For the convolutional and pooling nodes this is computed from the published arithmetic
    /// (see [`Conv1d`]); a window that cannot be placed yields a zero dimension, which
    /// [`Graph::validate`] reports rather than propagating.
    #[must_use]
    pub fn output_shape(&self) -> Option<Vec<usize>> {
        let s = match self {
            Self::Input(x) => x.shape.clone(),
            Self::Output(_) => return None,
            Self::Affine(x) => vec![x.rows],
            Self::Linear(x) => vec![x.rows],
            Self::Conv1d(x) => {
                vec![x.out_channels, conv_dim(x.length, x.padding, x.dilation, x.kernel, x.stride)]
            }
            Self::Conv2d(x) => vec![
                x.out_channels,
                conv_dim(x.size[0], x.padding[0], x.dilation[0], x.kernel[0], x.stride[0]),
                conv_dim(x.size[1], x.padding[1], x.dilation[1], x.kernel[1], x.stride[1]),
            ],
            Self::SumPool2d(x) | Self::AvgPool2d(x) => vec![
                x.channels,
                conv_dim(x.size[0], x.padding[0], 1, x.kernel[0], x.stride[0]),
                conv_dim(x.size[1], x.padding[1], 1, x.kernel[1], x.stride[1]),
            ],
            Self::Flatten(x) => {
                if x.start_dim > x.end_dim || x.end_dim >= x.size.len() {
                    // Malformed; `validate` reports the range error first. Passing the shape
                    // through unchanged keeps that error the one the caller sees.
                    x.size.clone()
                } else {
                    let mut out: Vec<usize> = x.size[..x.start_dim].to_vec();
                    out.push(product(&x.size[x.start_dim..=x.end_dim]));
                    out.extend_from_slice(&x.size[x.end_dim + 1..]);
                    out
                }
            }
            Self::I(x) => x.shape.clone(),
            Self::If(x) => x.shape.clone(),
            Self::Li(x) => x.shape.clone(),
            Self::Lif(x) => x.shape.clone(),
            Self::CubaLif(x) => x.shape.clone(),
            Self::Threshold(x) => x.shape.clone(),
            Self::Delay(x) => x.shape.clone(),
            Self::Scale(x) => x.shape.clone(),
        };
        Some(s)
    }

    /// What kind of signal this node puts on its outgoing edges. See [`Signal`] for the caveat
    /// that `NIR` itself does not type this.
    #[must_use]
    pub fn emits(&self) -> Signal {
        match self {
            Self::Lif(_) | Self::If(_) | Self::CubaLif(_) | Self::Threshold(_) => Signal::Spikes,
            Self::Affine(_)
            | Self::Linear(_)
            | Self::Conv1d(_)
            | Self::Conv2d(_)
            | Self::I(_)
            | Self::Li(_) => Signal::Continuous,
            _ => Signal::Either,
        }
    }

    /// What kind of signal this node requires on its incoming edges.
    ///
    /// Only [`Kind::Threshold`] constrains it: comparing a spike train against a level either
    /// reproduces it or annihilates it, and neither is what the author meant.
    #[must_use]
    pub fn accepts(&self) -> Signal {
        match self {
            Self::Threshold(_) => Signal::Continuous,
            _ => Signal::Either,
        }
    }
}

impl Affine {
    /// `W x + b`.
    ///
    /// `None` when `x.len()` is not [`Affine::cols`], when `weight` does not hold exactly
    /// `rows * cols` entries, when `bias` does not hold `rows`, or when `rows * cols` does not fit
    /// in a `usize` — the fields are public, so that product is checked rather than assumed.
    #[must_use]
    pub fn apply(&self, x: &[f64]) -> Option<Vec<f64>> {
        let cells = self.rows.checked_mul(self.cols)?;
        if x.len() != self.cols || self.weight.len() != cells || self.bias.len() != self.rows {
            return None;
        }
        let mut y = self.bias.clone();
        for i in 0..self.rows {
            let row = &self.weight[i * self.cols..(i + 1) * self.cols];
            let mut acc = y[i];
            for j in 0..self.cols {
                acc += row[j] * x[j];
            }
            y[i] = acc;
        }
        Some(y)
    }
}

impl Linear {
    /// `W x`.
    ///
    /// `None` when `x.len()` is not [`Linear::cols`], when the weight vector does not hold
    /// `rows * cols` entries, or when `rows * cols` does not fit in a `usize` — the fields are
    /// public, so that product is checked rather than assumed.
    #[must_use]
    pub fn apply(&self, x: &[f64]) -> Option<Vec<f64>> {
        let cells = self.rows.checked_mul(self.cols)?;
        if x.len() != self.cols || self.weight.len() != cells {
            return None;
        }
        let mut y = vec![0.0; self.rows];
        for i in 0..self.rows {
            let row = &self.weight[i * self.cols..(i + 1) * self.cols];
            let mut acc = 0.0;
            for j in 0..self.cols {
                acc += row[j] * x[j];
            }
            y[i] = acc;
        }
        Some(y)
    }
}

// ---------------------------------------------------------------------------------------------
// CubaLIF: the reduction, and the exact propagator that demonstrates it
// ---------------------------------------------------------------------------------------------

impl CubaLif {
    /// The [`Lif`] this node becomes in the instantaneous-synapse limit, or `None` when it does not
    /// take that limit.
    ///
    /// # The reduction
    ///
    /// With `tau_syn = 0` the synaptic equation `tau_syn * dI/dt = -I + w_in * u` has no dynamics
    /// left: `I = w_in * u` at every instant. Substituting into the membrane equation,
    ///
    /// ```text
    /// tau_mem * dv/dt = (v_leak - v) + r * w_in * u
    /// ```
    ///
    /// which is the `LIF` equation with resistance `r * w_in`. So the gain and the resistance are
    /// the same parameter in the limit, and this is where they get multiplied together.
    ///
    /// `None` unless **every** element has `tau_syn` exactly `0.0`: a small-but-non-zero synaptic
    /// constant is a filter, not the absence of one, and silently dropping it would change the
    /// network's coincidence window without changing anything a reader could see.
    #[must_use]
    pub fn reduce_to_lif(&self) -> Option<Lif> {
        if !self.tau_syn.iter().all(|&t| t == 0.0) {
            return None;
        }
        let n = self.tau_mem.len();
        if self.r.len() != n || self.w_in.len() != n {
            return None;
        }
        let mut r = Vec::with_capacity(n);
        for k in 0..n {
            r.push(self.r[k] * self.w_in[k]);
        }
        Some(Lif {
            shape: self.shape.clone(),
            tau: self.tau_mem.clone(),
            r,
            v_leak: self.v_leak.clone(),
            v_threshold: self.v_threshold.clone(),
            v_reset: self.v_reset.clone(),
        })
    }

    /// The state of element `k`, ready to integrate, starting at `v = v_leak` with no synaptic
    /// current.
    ///
    /// `None` when `k` is past the population or the parameter arrays are ragged. `NIR` says
    /// nothing about initial state — a graph carries parameters, not a trajectory — so the resting
    /// start is this implementation's choice and is stated here rather than assumed.
    #[must_use]
    pub fn state(&self, k: usize) -> Option<CubaState> {
        let n = self.tau_mem.len();
        if k >= n
            || self.tau_syn.len() != n
            || self.r.len() != n
            || self.v_leak.len() != n
            || self.v_threshold.len() != n
            || self.v_reset.len() != n
            || self.w_in.len() != n
        {
            return None;
        }
        Some(CubaState {
            tau_syn: self.tau_syn[k],
            tau_mem: self.tau_mem[k],
            r: self.r[k],
            v_leak: self.v_leak[k],
            v_threshold: self.v_threshold[k],
            v_reset: self.v_reset[k],
            w_in: self.w_in[k],
            i_syn: 0.0,
            v: self.v_leak[k],
        })
    }

    /// Sub-threshold membrane potential of element `k` at time `t`, in closed form, under a
    /// constant input `u` applied from rest.
    ///
    /// # The closed form
    ///
    /// From `v(0) = v_leak` and `I(0) = 0`, with `A = w_in * u`,
    ///
    /// ```text
    /// v(t) = v_leak + r*A * [ 1 - exp(-t/tau_mem)
    ///                           - tau_syn/(tau_syn - tau_mem) * (exp(-t/tau_syn) - exp(-t/tau_mem)) ]
    /// ```
    ///
    /// and at `tau_syn = tau_mem` the bracket degenerates to `1 - (1 + t/tau_mem)*exp(-t/tau_mem)`,
    /// the alpha function. At `tau_syn = 0` the third term vanishes and the whole thing is the
    /// `LIF`'s `1 - exp(-t/tau_mem)`, which is the reduction again, in closed form.
    ///
    /// **This ignores the threshold**: it is the sub-threshold solution, and above threshold the
    /// neuron would have fired and reset. That is what makes it usable as a check — it is the exact
    /// solution of a linear system, and [`CubaState::step_exact`] must reproduce it to
    /// floating-point noise rather than to a discretisation error.
    ///
    /// `None` for `k` past the population, for a non-finite or negative `t`, for a non-finite `u`,
    /// or for `tau_mem <= 0`.
    ///
    /// ⚠ **`t = 0` with `tau_syn = 0`** is the one place the expression above cannot be evaluated
    /// as written: `exp(-t/tau_syn)` is `exp(-0/0)`, which is `NaN`, and the cross term's factor of
    /// exactly zero does not rescue it — `0 * NaN` is `NaN`. The instantaneous-synapse limit of
    /// that exponential is `0` at every `t >= 0`, and that is what is substituted, so the value
    /// returned is `v_leak`, the rest the neuron starts from. This function is the oracle
    /// `the_cuba_propagator_matches_its_closed_form` compares the integrator against, so a `NaN`
    /// here is a `NaN` in the only independent check the propagator has.
    #[must_use]
    pub fn step_response(&self, k: usize, t: f64, u: f64) -> Option<f64> {
        let s = self.state(k)?;
        if !t.is_finite() || t < 0.0 || !u.is_finite() || !(s.tau_mem > 0.0) || s.tau_syn < 0.0 {
            return None;
        }
        let a = s.w_in * u;
        let em = (-t / s.tau_mem).exp();
        // See the `t = 0, tau_syn = 0` note above. For every `t > 0` this IS `(-t/0.0).exp()`,
        // bit for bit, so no other value moves.
        let es = if s.tau_syn == 0.0 { 0.0 } else { (-t / s.tau_syn).exp() };
        let bracket = if (s.tau_syn - s.tau_mem).abs() <= 1e-12 * s.tau_mem {
            1.0 - em - (t / s.tau_mem) * em
        } else {
            1.0 - em - (s.tau_syn / (s.tau_syn - s.tau_mem)) * (es - em)
        };
        Some(s.v_leak + s.r * a * bracket)
    }
}

/// One `CubaLIF` element being integrated, with the **exact** propagator of its linear system.
///
/// # Why exact, and what that buys
///
/// The pair `(I, v)` is a two-dimensional linear time-invariant system, so over a step of constant
/// input its solution is a matrix exponential — available in closed form, not as an approximation
/// (Rotter & Diesmann, "Exact digital simulation of time-invariant linear systems with applications
/// to neuronal modeling", *Biological Cybernetics* 81:381-402, 1999). Forward Euler on the same
/// pair has an error that grows with `dt / tau_syn`, and `tau_syn` is the *smallest* constant in
/// the model, so it is exactly the term that forces a small timestep and therefore the energy bill.
///
/// # ⚠ Exact does not mean jumpable: this model is NOT `EXACT_OVER_GAPS`
///
/// The propagator is exact and matrix exponentials compose, so the **flow** may be jumped: ten
/// steps of `dt` and one step of `10 dt` under zero input land in the same place to a part in
/// `1e14`. That is not the property [`crate::neuron::Neuron::EXACT_OVER_GAPS`] declares. That
/// constant is about the **hybrid** system — the flow *plus* the threshold and the reset — and the
/// threshold is where this model parts company with [`crate::neuron::Lif`].
///
/// Under zero input a `Lif`'s membrane is **monotone** toward `v_rest`, which sits below
/// threshold, so a `Lif` that is sub-threshold when a quiet gap opens is sub-threshold throughout
/// it and there is nothing for a jump to miss. A `CubaLIF`'s membrane is a **difference of two
/// exponentials** and keeps *rising* after its input stops whenever `i_syn` is non-zero — which is
/// the state this node exists to have. That late peak, several milliseconds after the last spike
/// arrived, is the entire reason `CubaLIF` is in `NIR`, and a jump does not delay the spike it
/// produces, it **deletes** it: [`crate::sim::Sim`] discards the return value of the step it jumps
/// a quiet gap with.
///
/// So [`crate::sim::Sim::new`] **refuses** a `CubaState` in [`crate::sim::Mode::EventDriven`], and
/// the model runs clocked. `the_cuba_hybrid_does_not_compose_across_a_quiet_gap` is the
/// measurement behind that refusal.
///
/// # The degenerate cases, both of them deliberate
///
/// At `tau_syn = 0` the synaptic factor is `exp(-dt/0) = exp(-inf) = 0` and the cross term's factor
/// `tau_syn / (tau_syn - tau_mem)` is `0`, so the update collapses onto the exponential-Euler
/// update [`crate::neuron::Lif`] uses. That is the `LIF` reduction falling out of the arithmetic
/// rather than being modelled separately, and a test pins it bit for bit.
///
/// The zero is substituted rather than divided for, for one reason: a **negative** zero is still
/// `== 0.0`, still passes [`Graph::validate`] (`-0.0 < 0.0` is false), and turns `-dt / tau_syn`
/// into `+inf`, so the division produced `exp(+inf)` and then a `NaN` membrane. The substituted
/// value is the same `0.0` the division gave for a positive zero, so nothing else moves.
///
/// At `tau_syn = tau_mem` the factor `tau_syn / (tau_syn - tau_mem)` is a division by zero that is
/// *not* a limit the arithmetic reaches; the analytic limit is `(dt/tau_mem) * exp(-dt/tau_mem)`
/// and there is a branch for it. The branch fires within a relative `1e-12` of equality, because
/// the general expression loses its significant digits to cancellation long before it divides by
/// zero.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CubaState {
    /// Synaptic time constant, seconds. Finite and **non-negative**; zero means the
    /// instantaneous-synapse limit. A negative value is a synapse that amplifies instead of
    /// decaying, and [`CubaState::step_exact`] refuses it rather than integrating it.
    pub tau_syn: f64,
    /// Membrane time constant, seconds. Strictly positive.
    pub tau_mem: f64,
    /// Input resistance, ohms.
    pub r: f64,
    /// Resting potential, volts.
    pub v_leak: f64,
    /// Firing threshold, volts. `NIR` spikes on `v > v_threshold`, strictly.
    pub v_threshold: f64,
    /// Post-spike potential, volts.
    pub v_reset: f64,
    /// Input gain, dimensionless.
    pub w_in: f64,
    /// Synaptic current, amperes. This is the state a `CubaLIF` synapse displaces; see
    /// [`CubaState::bump_current`].
    pub i_syn: f64,
    /// Membrane potential, volts.
    pub v: f64,
}

impl CubaState {
    /// Advance by `dt` seconds under a constant input `u`, returning `true` if the element spiked.
    ///
    /// The update is the exact solution over the step, not a discretisation of it. A non-finite or
    /// non-positive `dt`, or a non-finite `u`, is a **no-op returning `false`**: the alternative is
    /// a `NaN` in the membrane, which does not fail loudly — it propagates into every downstream
    /// spike time and the run completes and reports no spikes.
    ///
    /// A [`CubaState::tau_syn`] that is negative, infinite or `NaN` is a no-op for the same reason
    /// and one more. [`Graph::validate`] refuses a negative `tau_syn` at the graph boundary, but
    /// this struct's fields are public and [`CubaLif::state`] is not the only way to reach one: at
    /// `tau_syn = -5 ms` the synaptic factor `exp(-dt/tau_syn)` is greater than one and the current
    /// **grows without bound**, silently, with no `NaN` to notice. The precondition the validator
    /// enforces is enforced here too.
    pub fn step_exact(&mut self, dt: f64, u: f64) -> bool {
        if !(dt > 0.0)
            || !dt.is_finite()
            || !u.is_finite()
            || !(self.tau_mem > 0.0)
            || !(self.tau_syn >= 0.0)
            || !self.tau_syn.is_finite()
        {
            return false;
        }
        let a = self.w_in * u;
        let b = self.i_syn - a;
        let em = (-dt / self.tau_mem).exp();
        // `tau_syn == 0.0` is TRUE for a NEGATIVE ZERO, and `-dt / -0.0` is `+inf`, so the
        // instantaneous limit written as an ordinary division gave `exp(+inf) = inf` and then
        // `0 * inf = NaN` in the membrane — from a value `Graph::validate` accepts, because
        // `-0.0 < 0.0` is false. For a positive zero this substitutes the same `0.0` that
        // `exp(-dt/0.0)` already produced, so the `LIF` reduction stays bit for bit.
        let es = if self.tau_syn == 0.0 { 0.0 } else { (-dt / self.tau_syn).exp() };
        let cross = if (self.tau_syn - self.tau_mem).abs() <= 1e-12 * self.tau_mem {
            (dt / self.tau_mem) * em
        } else {
            (self.tau_syn / (self.tau_syn - self.tau_mem)) * (es - em)
        };
        // Grouped as `v_inf + (v - v_inf) * em + ...` rather than around `v_leak`, so that at
        // `tau_syn = 0` — where `cross` is exactly zero and the last term is exactly zero — the
        // arithmetic performed is the SAME arithmetic `Lif::step` performs, in the same order.
        let v_inf = self.v_leak + self.r * a;
        self.v = v_inf + (self.v - v_inf) * em + self.r * b * cross;
        self.i_syn = a + b * es;
        if self.v > self.v_threshold {
            self.v = self.v_reset;
            return true;
        }
        false
    }

    /// Displace the **synaptic current** by `di` amperes.
    ///
    /// This is what an arriving spike does to a `CubaLIF`, and it is why the node exists: the jump
    /// lands in `i_syn` and reaches the membrane filtered by `tau_syn`. The trait's
    /// [`crate::neuron::Neuron::bump`] displaces the membrane instead, which is the delta-synapse
    /// semantics the rest of this crate uses; both are here, and a caller wiring a `NIR` `CubaLIF`
    /// into [`crate::sim::Sim`] should know that `Sim` delivers through `bump` and therefore
    /// bypasses the filter.
    pub fn bump_current(&mut self, di: f64) {
        if di.is_finite() {
            self.i_syn += di;
        }
    }
}

impl crate::neuron::Neuron for CubaState {
    // FALSE, and the reason is the threshold rather than the propagator; see the type's own doc.
    //
    // The linear flow composes — Phi(a) * Phi(b) = Phi(a + b), held to 1e-14 relative by
    // `the_cuba_flow_composes_across_a_quiet_gap`. The hybrid system does not, because under zero
    // input this membrane is non-monotone whenever `i_syn` is non-zero and can cross threshold
    // inside a gap the jump skips over.
    //
    // Measured, `tau_syn = 5 ms`, `tau_mem = 20 ms`, `r = 10 MOhm`, `v_leak = -65 mV`,
    // `v_threshold = -50 mV`, one arrival of 12 nA and then silence: 600 steps of 0.1 ms spike
    // ONCE, at tick 41, and land at -63.9396 mV; one step of 60 ms spikes NOT AT ALL and lands at
    // -63.0088 mV. A `true` here would have let `crate::sim` produce the second answer while
    // reporting the first model.
    const EXACT_OVER_GAPS: bool = false;

    fn step(&mut self, dt: f64, i: f64) -> bool {
        self.step_exact(dt, i)
    }

    /// Displaces the MEMBRANE, per the trait's contract. For the `CubaLIF` path — a spike landing
    /// in the synaptic current — use [`CubaState::bump_current`].
    fn bump(&mut self, dv: f64) {
        if dv.is_finite() {
            self.v += dv;
        }
    }

    fn potential(&self) -> f64 {
        self.v
    }

    fn reset(&mut self) {
        self.v = self.v_leak;
        self.i_syn = 0.0;
    }
}

// ---------------------------------------------------------------------------------------------
// The graph
// ---------------------------------------------------------------------------------------------

/// A node with the name the graph refers to it by.
#[derive(Debug, Clone, PartialEq)]
pub struct Named {
    /// Unique within the graph. Must be non-empty and free of whitespace, `=` and `#`, because the
    /// text format is whitespace-separated and a name containing a space would read back as two
    /// tokens and a different graph.
    pub name: String,
    /// The primitive and its parameters.
    pub node: Node,
}

/// A directed edge. `NIR` graphs are directed and may be cyclic; a recurrent layer *is* a cycle.
#[derive(Debug, Clone, PartialEq)]
pub struct Edge {
    /// Name of the node the signal leaves.
    pub from: String,
    /// Name of the node the signal arrives at. A node with several incoming edges receives their
    /// **sum**, which is `NIR`'s execution semantics and the reason a fan-in shape check is a
    /// per-edge check.
    pub to: String,
}

/// A `NIR` computational graph: named nodes and directed edges between them.
///
/// Node order is **insertion order** and is preserved through the text format, so two graphs built
/// by the same sequence of calls serialise to byte-identical text. Nothing here uses a hash map:
/// `std`'s `HashMap` seeds itself from the operating system, which would make iteration order — and
/// therefore the file — differ between runs on the same machine.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Graph {
    /// Nodes in insertion order.
    pub nodes: Vec<Named>,
    /// Edges in insertion order.
    pub edges: Vec<Edge>,
}

/// Which structural rules a graph is being held to.
///
/// `NIR` permits recurrence, so [`Rules::allow_cycles`] defaults to `true`. A consumer that cannot
/// execute a cycle — a feedforward compiler, a layer-by-layer mapper — asks for
/// [`Rules::feedforward`] and gets the cycle named instead of a hang.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rules {
    /// Whether a directed cycle is legal. `true` by default: a recurrent spiking layer is a cycle
    /// and is the point of half the models this IR carries.
    pub allow_cycles: bool,
    /// Whether the graph must contain at least one [`Kind::Input`] node. `false` by default: a
    /// subgraph is a legal object and has no boundary of its own.
    pub require_input: bool,
    /// Whether the graph must contain at least one [`Kind::Output`] node. `false` by default, for
    /// the same reason.
    pub require_output: bool,
    /// Whether to apply the spike-versus-continuous check described on [`Signal`]. `true` by
    /// default. Turn it off when importing a graph whose author used [`Kind::Threshold`]
    /// deliberately on a spike train.
    pub check_signal_kind: bool,
}

impl Default for Rules {
    /// Permissive in the two places `NIR` is permissive — cycles are legal, boundaries are optional
    /// — and strict in the one place this crate adds a rule of its own.
    fn default() -> Self {
        Self {
            allow_cycles: true,
            require_input: false,
            require_output: false,
            check_signal_kind: true,
        }
    }
}

impl Rules {
    /// Cycles refused, boundaries required: what a mapper that walks the graph once needs.
    #[must_use]
    pub fn feedforward() -> Self {
        Self {
            allow_cycles: false,
            require_input: true,
            require_output: true,
            check_signal_kind: true,
        }
    }

    /// Cycles permitted, boundaries required.
    #[must_use]
    pub fn recurrent() -> Self {
        Self {
            allow_cycles: true,
            require_input: true,
            require_output: true,
            check_signal_kind: true,
        }
    }
}

/// What is wrong with a graph.
///
/// [`Graph::validate`] reports the **first** problem in a fixed order — names, then parameters,
/// then ports, then shapes, then signal kinds, then cycles, then boundaries — so that the same
/// broken graph always produces the same message. A validator that reported whichever error its
/// iteration order reached first would make a regression test on the message flaky.
#[derive(Debug, Clone, PartialEq)]
pub enum ValidationError {
    /// Two nodes share a name, so an edge naming it is ambiguous.
    DuplicateNode {
        /// The repeated name.
        name: String,
    },
    /// A name is empty or contains whitespace, `=` or `#`, none of which survive the text format.
    BadNodeName {
        /// The offending name, as given.
        name: String,
    },
    /// An edge names a node the graph does not contain.
    DanglingEdge {
        /// Source end of the edge.
        from: String,
        /// Target end of the edge.
        to: String,
        /// Whichever end is missing.
        missing: String,
    },
    /// An edge arrives at a [`Kind::Input`] node, which has no input port.
    NoInputPort {
        /// The node the edge leaves.
        from: String,
        /// The `Input` node it arrives at.
        into: String,
    },
    /// An edge leaves a [`Kind::Output`] node, which has no output port.
    NoOutputPort {
        /// The `Output` node the edge leaves.
        from: String,
        /// The node it arrives at.
        to: String,
    },
    /// The shape one node emits is not the shape the next one requires.
    ShapeMismatch {
        /// Source node.
        from: String,
        /// Target node.
        to: String,
        /// What the source emits.
        emitted: Vec<usize>,
        /// What the target requires.
        expected: Vec<usize>,
    },
    /// A shape contains a zero-length axis, or none at all. Usually a convolution whose window
    /// does not fit its input.
    EmptyDimension {
        /// The node whose shape is degenerate.
        node: String,
        /// Which axis, or `0` when the shape is empty.
        axis: usize,
    },
    /// A parameter array is not as long as the node's shape says it must be.
    RaggedParameter {
        /// The node.
        node: String,
        /// Which parameter, by its name in the text format.
        field: &'static str,
        /// The length found.
        len: usize,
        /// The length the shape implies.
        expected: usize,
    },
    /// A parameter is `NaN` or infinite. Rejected at the boundary: a non-finite weight does not
    /// fail loudly, it poisons one membrane and then every spike time downstream of it.
    NonFiniteParameter {
        /// The node.
        node: String,
        /// Which parameter.
        field: &'static str,
        /// Index within the array.
        index: usize,
    },
    /// A structural parameter is out of range — a zero stride, a `groups` that does not divide the
    /// channels, a `start_dim` past the end of the shape.
    BadHyperparameter {
        /// The node.
        node: String,
        /// Which parameter.
        field: &'static str,
        /// What the constraint is.
        why: &'static str,
    },
    /// A spike train reached a node that requires a continuous input.
    ///
    /// **One-directional, and it will stay that way while [`Node::accepts`] has one arm**: only
    /// [`Kind::Threshold`] requires anything, and what it requires is [`Signal::Continuous`], so no
    /// node in the vocabulary can demand spikes and the mirror-image error cannot arise. The
    /// variant is shaped to carry both ends anyway, so a future node that does require spikes
    /// needs no new vocabulary. See [`Signal`] for why this check is this crate's and not `NIR`'s.
    SignalMismatch {
        /// Source node.
        from: String,
        /// Target node.
        to: String,
        /// What the source emits.
        emitted: Signal,
        /// What the target requires.
        required: Signal,
    },
    /// The graph contains a directed cycle and [`Rules::allow_cycles`] was false.
    Cycle {
        /// Every node that is on a cycle or downstream of one, in graph order. Not the cycle
        /// itself: separating the two costs a second traversal and buys nothing a reader of this
        /// list cannot see.
        nodes: Vec<String>,
    },
    /// [`Rules::require_input`] was set and no [`Kind::Input`] node is present.
    MissingInputNode,
    /// [`Rules::require_output`] was set and no [`Kind::Output`] node is present.
    MissingOutputNode,
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateNode { name } => write!(f, "two nodes are named {name}"),
            Self::BadNodeName { name } => write!(
                f,
                "node name {name:?} is empty or holds whitespace, '=' or '#', which the text \
                 format cannot represent"
            ),
            Self::DanglingEdge { from, to, missing } => {
                write!(f, "edge {from} -> {to} names {missing}, which is not in the graph")
            }
            Self::NoInputPort { from, into } => {
                write!(f, "edge {from} -> {into}: an Input node has no input port")
            }
            Self::NoOutputPort { from, to } => {
                write!(f, "edge {from} -> {to}: an Output node has no output port")
            }
            Self::ShapeMismatch { from, to, emitted, expected } => write!(
                f,
                "edge {from} -> {to}: {from} emits {emitted:?} and {to} requires {expected:?}"
            ),
            Self::EmptyDimension { node, axis } => {
                write!(f, "node {node} has a zero-length axis {axis}")
            }
            Self::RaggedParameter { node, field, len, expected } => write!(
                f,
                "node {node}: parameter {field} has {len} entries where the shape implies {expected}"
            ),
            Self::NonFiniteParameter { node, field, index } => {
                write!(f, "node {node}: parameter {field}[{index}] is not finite")
            }
            Self::BadHyperparameter { node, field, why } => {
                write!(f, "node {node}: {field} {why}")
            }
            Self::SignalMismatch { from, to, emitted, required } => write!(
                f,
                "edge {from} -> {to}: {from} emits {emitted} and {to} requires {required}"
            ),
            Self::Cycle { nodes } => {
                write!(f, "the graph has a directed cycle involving {nodes:?}, which these rules forbid")
            }
            Self::MissingInputNode => f.write_str("the graph has no Input node"),
            Self::MissingOutputNode => f.write_str("the graph has no Output node"),
        }
    }
}

/// See the note on [`crate::net::NetError`]: a library error has to cross a `Box<dyn Error>`
/// boundary or its callers reach for `.unwrap()`.
impl std::error::Error for ValidationError {}

fn name_is_writable(s: &str) -> bool {
    !s.is_empty()
        && !s.chars().any(|c| c.is_whitespace() || c == '=' || c == '#' || c.is_control())
}

impl Graph {
    /// An empty graph.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a node. **Does not check anything** — [`Graph::validate`] does, in one place and in a
    /// fixed order, so that building a graph incrementally never fails halfway and leaves a
    /// half-built object behind.
    pub fn push(&mut self, name: &str, node: Node) -> &mut Self {
        self.nodes.push(Named { name: name.to_string(), node });
        self
    }

    /// Append an edge. Does not check that either end exists; [`Graph::validate`] reports a
    /// dangling edge by name.
    pub fn edge(&mut self, from: &str, to: &str) -> &mut Self {
        self.edges.push(Edge { from: from.to_string(), to: to.to_string() });
        self
    }

    /// Index of the node with this name, by linear scan.
    ///
    /// Linear rather than hashed, deliberately: a `NIR` graph has tens of nodes, and the hash map
    /// that would make this `O(1)` seeds itself from the operating system and would put a
    /// platform-dependent order into anything that iterated it.
    #[must_use]
    pub fn index_of(&self, name: &str) -> Option<usize> {
        self.nodes.iter().position(|n| n.name == name)
    }

    /// A topological order of the node indices, or `None` if the graph has a cycle.
    ///
    /// Kahn's algorithm with the ready set kept in **graph order**, so the order returned is a
    /// function of the graph alone and not of any iteration accident.
    #[must_use]
    pub fn topological_order(&self) -> Option<Vec<usize>> {
        let order = self.settled_prefix();
        if order.len() == self.nodes.len() { Some(order) } else { None }
    }

    /// Whether the graph is free of directed cycles.
    #[must_use]
    pub fn is_acyclic(&self) -> bool {
        self.topological_order().is_some()
    }

    /// The node indices Kahn's algorithm can settle: everything not on a cycle and not downstream
    /// of one. Equal to a full topological order when the graph is acyclic.
    fn settled_prefix(&self) -> Vec<usize> {
        let n = self.nodes.len();
        let mut indeg = vec![0usize; n];
        let mut out: Vec<Vec<usize>> = vec![Vec::new(); n];
        for e in &self.edges {
            let (Some(a), Some(b)) = (self.index_of(&e.from), self.index_of(&e.to)) else {
                continue; // a dangling edge constrains nothing; `validate` reports it separately
            };
            out[a].push(b);
            indeg[b] += 1;
        }
        let mut ready: Vec<usize> = (0..n).filter(|&i| indeg[i] == 0).collect();
        let mut head = 0;
        while head < ready.len() {
            let i = ready[head];
            head += 1;
            for &j in &out[i] {
                indeg[j] -= 1;
                if indeg[j] == 0 {
                    ready.push(j);
                }
            }
        }
        ready
    }

    /// Check the graph against `rules`.
    ///
    /// # Errors
    ///
    /// The first [`ValidationError`] in this fixed order: node names, node parameters, edge
    /// endpoints and ports, edge shapes, edge signal kinds, cycles, boundary nodes. The order is
    /// part of the contract so that a message can be asserted on.
    ///
    /// This function is **total**: it is the boundary a graph parsed by [`Graph::from_text`]
    /// crosses, so every count a shape or a kernel implies is computed with checked arithmetic and
    /// reported as [`ValidationError::BadHyperparameter`] rather than overflowing. It does not
    /// panic on any graph, however malformed.
    pub fn validate(&self, rules: &Rules) -> Result<(), ValidationError> {
        // 1. Names.
        for k in 0..self.nodes.len() {
            let name = &self.nodes[k].name;
            if !name_is_writable(name) {
                return Err(ValidationError::BadNodeName { name: name.clone() });
            }
            if self.nodes[..k].iter().any(|m| &m.name == name) {
                return Err(ValidationError::DuplicateNode { name: name.clone() });
            }
        }

        // 2. Parameters.
        for named in &self.nodes {
            check_node(&named.name, &named.node)?;
        }

        // 3. Edge endpoints and ports.
        for e in &self.edges {
            let Some(a) = self.index_of(&e.from) else {
                return Err(ValidationError::DanglingEdge {
                    from: e.from.clone(),
                    to: e.to.clone(),
                    missing: e.from.clone(),
                });
            };
            let Some(b) = self.index_of(&e.to) else {
                return Err(ValidationError::DanglingEdge {
                    from: e.from.clone(),
                    to: e.to.clone(),
                    missing: e.to.clone(),
                });
            };
            if self.nodes[b].node.kind() == Kind::Input {
                return Err(ValidationError::NoInputPort {
                    from: e.from.clone(),
                    into: e.to.clone(),
                });
            }
            if self.nodes[a].node.kind() == Kind::Output {
                return Err(ValidationError::NoOutputPort {
                    from: e.from.clone(),
                    to: e.to.clone(),
                });
            }
        }

        // 4. Shapes.
        for e in &self.edges {
            let (Some(a), Some(b)) = (self.index_of(&e.from), self.index_of(&e.to)) else {
                continue;
            };
            let emitted = self.nodes[a].node.output_shape().unwrap_or_default();
            let expected = self.nodes[b].node.input_shape().unwrap_or_default();
            if emitted != expected {
                return Err(ValidationError::ShapeMismatch {
                    from: e.from.clone(),
                    to: e.to.clone(),
                    emitted,
                    expected,
                });
            }
        }

        // 5. Signal kinds.
        if rules.check_signal_kind {
            for e in &self.edges {
                let (Some(a), Some(b)) = (self.index_of(&e.from), self.index_of(&e.to)) else {
                    continue;
                };
                let emitted = self.nodes[a].node.emits();
                let required = self.nodes[b].node.accepts();
                if emitted != Signal::Either && required != Signal::Either && emitted != required {
                    return Err(ValidationError::SignalMismatch {
                        from: e.from.clone(),
                        to: e.to.clone(),
                        emitted,
                        required,
                    });
                }
            }
        }

        // 6. Cycles.
        if !rules.allow_cycles && self.topological_order().is_none() {
            // Kahn's algorithm settles exactly the nodes that are NOT on or downstream of a cycle,
            // so re-running it and taking the complement names the offending region without a
            // second, different traversal that could disagree with the first.
            let settled = self.settled_prefix();
            let stuck: Vec<String> = (0..self.nodes.len())
                .filter(|i| !settled.contains(i))
                .map(|i| self.nodes[i].name.clone())
                .collect();
            return Err(ValidationError::Cycle { nodes: stuck });
        }

        // 7. Boundaries.
        if rules.require_input && !self.nodes.iter().any(|n| n.node.kind() == Kind::Input) {
            return Err(ValidationError::MissingInputNode);
        }
        if rules.require_output && !self.nodes.iter().any(|n| n.node.kind() == Kind::Output) {
            return Err(ValidationError::MissingOutputNode);
        }
        Ok(())
    }
}

/// Every float in an array must be finite, and the array must be as long as the shape says.
fn check_array(
    node: &str,
    field: &'static str,
    v: &[f64],
    expected: usize,
) -> Result<(), ValidationError> {
    if v.len() != expected {
        return Err(ValidationError::RaggedParameter {
            node: node.to_string(),
            field,
            len: v.len(),
            expected,
        });
    }
    for (index, x) in v.iter().enumerate() {
        if !x.is_finite() {
            return Err(ValidationError::NonFiniteParameter {
                node: node.to_string(),
                field,
                index,
            });
        }
    }
    Ok(())
}

/// A shape must have at least one axis, no zero axis, and an element count that fits in a `usize`.
///
/// The last of those is the one a file can violate on purpose: `shape=[4294967296,4294967296]` is
/// four tokens and names `2^64` elements. Before this check the plain product wrapped to zero in a
/// release build, `check_array` then expected zero entries, found zero, and the node validated.
fn check_shape(node: &str, shape: &[usize]) -> Result<(), ValidationError> {
    if shape.is_empty() {
        return Err(ValidationError::EmptyDimension { node: node.to_string(), axis: 0 });
    }
    for (axis, &d) in shape.iter().enumerate() {
        if d == 0 {
            return Err(ValidationError::EmptyDimension { node: node.to_string(), axis });
        }
    }
    if checked_product(shape).is_none() {
        return Err(ValidationError::BadHyperparameter {
            node: node.to_string(),
            field: "shape",
            why: "names more elements than a usize can count",
        });
    }
    Ok(())
}

fn check_positive(node: &str, field: &'static str, v: &[f64]) -> Result<(), ValidationError> {
    if v.iter().any(|&x| !(x > 0.0)) {
        return Err(ValidationError::BadHyperparameter {
            node: node.to_string(),
            field,
            why: "must be strictly positive at every element",
        });
    }
    Ok(())
}

fn check_node(name: &str, node: &Node) -> Result<(), ValidationError> {
    let bad = |field: &'static str, why: &'static str| ValidationError::BadHyperparameter {
        node: name.to_string(),
        field,
        why,
    };
    match node {
        Node::Input(x) => check_shape(name, &x.shape)?,
        Node::Output(x) => check_shape(name, &x.shape)?,
        Node::Affine(x) => {
            if x.rows == 0 || x.cols == 0 {
                return Err(bad("rows/cols", "must both be non-zero"));
            }
            let cells = x
                .rows
                .checked_mul(x.cols)
                .ok_or_else(|| bad("rows/cols", "name more cells than a usize can count"))?;
            check_array(name, "weight", &x.weight, cells)?;
            check_array(name, "bias", &x.bias, x.rows)?;
        }
        Node::Linear(x) => {
            if x.rows == 0 || x.cols == 0 {
                return Err(bad("rows/cols", "must both be non-zero"));
            }
            let cells = x
                .rows
                .checked_mul(x.cols)
                .ok_or_else(|| bad("rows/cols", "name more cells than a usize can count"))?;
            check_array(name, "weight", &x.weight, cells)?;
        }
        Node::Conv1d(x) => {
            if x.groups == 0 || x.in_channels % x.groups != 0 || x.out_channels % x.groups != 0 {
                return Err(bad("groups", "must be non-zero and divide both channel counts"));
            }
            if x.stride == 0 {
                return Err(bad("stride", "must be non-zero"));
            }
            if x.kernel == 0 || x.dilation == 0 {
                return Err(bad("kernel/dilation", "must both be non-zero"));
            }
            let want = x
                .out_channels
                .checked_mul(x.in_channels / x.groups)
                .and_then(|v| v.checked_mul(x.kernel))
                .ok_or_else(|| {
                    bad("weight", "would need more entries than a usize can count")
                })?;
            check_array(name, "weight", &x.weight, want)?;
            if !x.bias.is_empty() {
                check_array(name, "bias", &x.bias, x.out_channels)?;
            }
            check_shape(name, &node.input_shape().unwrap_or_default())?;
            check_shape(name, &node.output_shape().unwrap_or_default())?;
        }
        Node::Conv2d(x) => {
            if x.groups == 0 || x.in_channels % x.groups != 0 || x.out_channels % x.groups != 0 {
                return Err(bad("groups", "must be non-zero and divide both channel counts"));
            }
            if x.stride[0] == 0 || x.stride[1] == 0 {
                return Err(bad("stride", "must be non-zero on both axes"));
            }
            if x.kernel[0] == 0 || x.kernel[1] == 0 || x.dilation[0] == 0 || x.dilation[1] == 0 {
                return Err(bad("kernel/dilation", "must be non-zero on both axes"));
            }
            let want = x
                .out_channels
                .checked_mul(x.in_channels / x.groups)
                .and_then(|v| v.checked_mul(x.kernel[0]))
                .and_then(|v| v.checked_mul(x.kernel[1]))
                .ok_or_else(|| {
                    bad("weight", "would need more entries than a usize can count")
                })?;
            check_array(name, "weight", &x.weight, want)?;
            if !x.bias.is_empty() {
                check_array(name, "bias", &x.bias, x.out_channels)?;
            }
            check_shape(name, &node.input_shape().unwrap_or_default())?;
            check_shape(name, &node.output_shape().unwrap_or_default())?;
        }
        Node::SumPool2d(x) | Node::AvgPool2d(x) => {
            if x.stride[0] == 0 || x.stride[1] == 0 {
                return Err(bad("stride", "must be non-zero on both axes"));
            }
            if x.kernel[0] == 0 || x.kernel[1] == 0 {
                return Err(bad("kernel", "must be non-zero on both axes"));
            }
            check_shape(name, &node.input_shape().unwrap_or_default())?;
            check_shape(name, &node.output_shape().unwrap_or_default())?;
        }
        Node::Flatten(x) => {
            check_shape(name, &x.size)?;
            if x.start_dim > x.end_dim {
                return Err(bad("start_dim", "must not exceed end_dim"));
            }
            if x.end_dim >= x.size.len() {
                return Err(bad("end_dim", "must index the input shape"));
            }
        }
        Node::I(x) => {
            check_shape(name, &x.shape)?;
            check_array(name, "r", &x.r, product(&x.shape))?;
        }
        Node::If(x) => {
            // `check_shape` before the product: it is what refuses a shape whose element count
            // does not fit in a `usize`, and `product` saturates rather than reporting.
            check_shape(name, &x.shape)?;
            let n = product(&x.shape);
            check_array(name, "r", &x.r, n)?;
            check_array(name, "v_threshold", &x.v_threshold, n)?;
            check_array(name, "v_reset", &x.v_reset, n)?;
        }
        Node::Li(x) => {
            // `check_shape` before the product: it is what refuses a shape whose element count
            // does not fit in a `usize`, and `product` saturates rather than reporting.
            check_shape(name, &x.shape)?;
            let n = product(&x.shape);
            check_array(name, "tau", &x.tau, n)?;
            check_positive(name, "tau", &x.tau)?;
            check_array(name, "r", &x.r, n)?;
            check_array(name, "v_leak", &x.v_leak, n)?;
        }
        Node::Lif(x) => {
            // `check_shape` before the product: it is what refuses a shape whose element count
            // does not fit in a `usize`, and `product` saturates rather than reporting.
            check_shape(name, &x.shape)?;
            let n = product(&x.shape);
            check_array(name, "tau", &x.tau, n)?;
            check_positive(name, "tau", &x.tau)?;
            check_array(name, "r", &x.r, n)?;
            check_array(name, "v_leak", &x.v_leak, n)?;
            check_array(name, "v_threshold", &x.v_threshold, n)?;
            check_array(name, "v_reset", &x.v_reset, n)?;
        }
        Node::CubaLif(x) => {
            // `check_shape` before the product: it is what refuses a shape whose element count
            // does not fit in a `usize`, and `product` saturates rather than reporting.
            check_shape(name, &x.shape)?;
            let n = product(&x.shape);
            check_array(name, "tau_syn", &x.tau_syn, n)?;
            if x.tau_syn.iter().any(|&t| t < 0.0) {
                // Zero IS legal, and means the instantaneous-synapse limit; see
                // `CubaLif::reduce_to_lif`. Negative is a synapse that grows without bound.
                return Err(bad("tau_syn", "must be non-negative (zero is the LIF limit)"));
            }
            check_array(name, "tau_mem", &x.tau_mem, n)?;
            check_positive(name, "tau_mem", &x.tau_mem)?;
            check_array(name, "r", &x.r, n)?;
            check_array(name, "v_leak", &x.v_leak, n)?;
            check_array(name, "v_threshold", &x.v_threshold, n)?;
            check_array(name, "v_reset", &x.v_reset, n)?;
            check_array(name, "w_in", &x.w_in, n)?;
        }
        Node::Threshold(x) => {
            check_shape(name, &x.shape)?;
            check_array(name, "threshold", &x.threshold, product(&x.shape))?;
        }
        Node::Delay(x) => {
            check_shape(name, &x.shape)?;
            check_array(name, "delay", &x.delay, product(&x.shape))?;
            if x.delay.iter().any(|&d| d < 0.0) {
                return Err(bad("delay", "must be non-negative"));
            }
        }
        Node::Scale(x) => {
            check_shape(name, &x.shape)?;
            check_array(name, "scale", &x.scale, product(&x.shape))?;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Text serialisation
// ---------------------------------------------------------------------------------------------

/// The first line of every file this module writes, and the only one it will read.
pub const MAGIC: &str = "ferromorphic-nir 1";

/// Why a text file could not be read as a graph.
///
/// Every variant carries the **1-based line number**, because the first thing anyone does with a
/// parse error is open the file at that line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TextError {
    /// The first meaningful line was not [`MAGIC`].
    WrongMagic {
        /// What was there instead, trimmed.
        found: String,
    },
    /// A line began with something other than `node` or `edge`.
    UnknownDirective {
        /// 1-based line number.
        line: usize,
        /// The first word of the line.
        word: String,
    },
    /// A line ran out of tokens before its mandatory ones were read.
    ShortLine {
        /// 1-based line number.
        line: usize,
        /// The shape the line should have had.
        want: &'static str,
    },
    /// A node named a type that is not in the `NIR` vocabulary. Refused rather than skipped: a
    /// skipped node leaves a graph smaller than its file, and it still runs.
    UnknownNodeType {
        /// 1-based line number.
        line: usize,
        /// The type token found.
        found: String,
    },
    /// A node name that could not survive a round trip through this format.
    UnwritableName {
        /// 1-based line number.
        line: usize,
        /// The name found.
        name: String,
    },
    /// A parameter token had no `=` in it.
    MalformedField {
        /// 1-based line number.
        line: usize,
        /// The token.
        token: String,
    },
    /// The same parameter appeared twice on one line, so which one wins would be an accident.
    RepeatedField {
        /// 1-based line number.
        line: usize,
        /// The parameter name.
        key: String,
    },
    /// A parameter this node type requires was not present.
    AbsentField {
        /// 1-based line number.
        line: usize,
        /// The node type.
        kind: &'static str,
        /// The parameter name.
        key: &'static str,
    },
    /// A parameter this node type does not have was present. Refused rather than ignored: an
    /// ignored field is how a file written by a newer version reads as a valid older graph.
    SurplusField {
        /// 1-based line number.
        line: usize,
        /// The node type.
        kind: &'static str,
        /// The parameter name.
        key: String,
    },
    /// A value did not parse as the number or array its parameter needs.
    BadValue {
        /// 1-based line number.
        line: usize,
        /// The parameter name.
        key: &'static str,
        /// The text that failed.
        text: String,
    },
    /// A float parsed as `NaN` or an infinity. Rejected here rather than downstream, because a
    /// non-finite parameter does not fail loudly later.
    NonFiniteValue {
        /// 1-based line number.
        line: usize,
        /// The parameter name.
        key: &'static str,
        /// The text that produced it.
        text: String,
    },
    /// A fixed-length parameter had the wrong number of entries.
    BadArity {
        /// 1-based line number.
        line: usize,
        /// The parameter name.
        key: &'static str,
        /// How many entries were required.
        want: usize,
        /// How many were found.
        got: usize,
    },
}

impl fmt::Display for TextError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WrongMagic { found } => {
                write!(f, "first line is {found:?}, not {MAGIC:?}")
            }
            Self::UnknownDirective { line, word } => {
                write!(f, "line {line}: {word:?} is not 'node' or 'edge'")
            }
            Self::ShortLine { line, want } => write!(f, "line {line}: expected {want}"),
            Self::UnknownNodeType { line, found } => {
                write!(f, "line {line}: {found:?} is not a NIR node type")
            }
            Self::UnwritableName { line, name } => {
                write!(f, "line {line}: node name {name:?} cannot be written back out")
            }
            Self::MalformedField { line, token } => {
                write!(f, "line {line}: {token:?} is not a key=value pair")
            }
            Self::RepeatedField { line, key } => write!(f, "line {line}: {key} appears twice"),
            Self::AbsentField { line, kind, key } => {
                write!(f, "line {line}: a {kind} node needs {key}")
            }
            Self::SurplusField { line, kind, key } => {
                write!(f, "line {line}: a {kind} node has no field {key}")
            }
            Self::BadValue { line, key, text } => {
                write!(f, "line {line}: {key}={text} did not parse")
            }
            Self::NonFiniteValue { line, key, text } => {
                write!(f, "line {line}: {key} holds the non-finite value {text}")
            }
            Self::BadArity { line, key, want, got } => {
                write!(f, "line {line}: {key} needs {want} entries and has {got}")
            }
        }
    }
}

/// See the note on [`crate::net::NetError`].
impl std::error::Error for TextError {}

/// Format a float so that parsing it back gives the **same bits**.
///
/// Rust's `{:?}` already emits the shortest decimal that round-trips, and this **checks** that
/// property per value rather than trusting it, falling back to [`fmt_f64_long`] when it does not
/// hold. The check costs one parse per number written and removes a class of silent corruption from
/// the format's central claim.
///
/// ⚠ The fallback has never fired, and on a conforming Rust it never will — `{:?}` round-trips by
/// construction. It is kept because the cost is one parse and the alternative is trusting a
/// property the format's whole claim rests on. [`fmt_f64_long`] is a named function rather than an
/// inline `else` precisely so that the branch can be tested on its own; an untested fallback is a
/// fallback that fails the first time it is needed.
fn fmt_f64(x: f64) -> String {
    let s = format!("{x:?}");
    if s.parse::<f64>().map(f64::to_bits) == Ok(x.to_bits()) { s } else { fmt_f64_long(x) }
}

/// The long form [`fmt_f64`] falls back to: **eighteen** significant digits, one before the point
/// and seventeen after.
///
/// Seventeen significant digits is the number that always round-trips an `f64`; `{:.17e}` prints
/// eighteen, which is one more than needed and therefore also round-trips. The count is stated
/// exactly because an earlier doc called this "17 significant digits", and a reader checking the
/// format's guarantee against the standard figure would have found the two off by one.
fn fmt_f64_long(x: f64) -> String {
    format!("{x:.17e}")
}

fn w_usize(out: &mut String, key: &str, v: usize) {
    out.push(' ');
    out.push_str(key);
    out.push('=');
    out.push_str(&v.to_string());
}

fn w_dims(out: &mut String, key: &str, v: &[usize]) {
    out.push(' ');
    out.push_str(key);
    out.push_str("=[");
    for (k, d) in v.iter().enumerate() {
        if k > 0 {
            out.push(',');
        }
        out.push_str(&d.to_string());
    }
    out.push(']');
}

fn w_floats(out: &mut String, key: &str, v: &[f64]) {
    out.push(' ');
    out.push_str(key);
    out.push_str("=[");
    for (k, x) in v.iter().enumerate() {
        if k > 0 {
            out.push(',');
        }
        out.push_str(&fmt_f64(*x));
    }
    out.push(']');
}

/// The `key=value` pairs on one node line, consumed by name so that a missing field and a surplus
/// field are both caught rather than one of them being caught by accident.
struct Fields<'a> {
    line: usize,
    kind: &'static str,
    items: Vec<(&'a str, &'a str, bool)>,
}

impl<'a> Fields<'a> {
    fn new(line: usize, kind: &'static str, toks: &[&'a str]) -> Result<Self, TextError> {
        let mut items: Vec<(&'a str, &'a str, bool)> = Vec::with_capacity(toks.len());
        for t in toks {
            let Some((k, v)) = t.split_once('=') else {
                return Err(TextError::MalformedField { line, token: (*t).to_string() });
            };
            if items.iter().any(|(existing, _, _)| *existing == k) {
                return Err(TextError::RepeatedField { line, key: k.to_string() });
            }
            items.push((k, v, false));
        }
        Ok(Self { line, kind, items })
    }

    fn take(&mut self, key: &'static str) -> Result<&'a str, TextError> {
        for (k, v, used) in &mut self.items {
            if *k == key {
                *used = true;
                return Ok(*v);
            }
        }
        Err(TextError::AbsentField { line: self.line, kind: self.kind, key })
    }

    fn usize_of(&mut self, key: &'static str) -> Result<usize, TextError> {
        let raw = self.take(key)?;
        raw.parse::<usize>().map_err(|_| TextError::BadValue {
            line: self.line,
            key,
            text: raw.to_string(),
        })
    }

    fn dims(&mut self, key: &'static str) -> Result<Vec<usize>, TextError> {
        let raw = self.take(key)?;
        let inner = raw
            .strip_prefix('[')
            .and_then(|r| r.strip_suffix(']'))
            .ok_or_else(|| TextError::BadValue {
                line: self.line,
                key,
                text: raw.to_string(),
            })?;
        if inner.is_empty() {
            return Ok(Vec::new());
        }
        let mut v = Vec::new();
        for part in inner.split(',') {
            v.push(part.parse::<usize>().map_err(|_| TextError::BadValue {
                line: self.line,
                key,
                text: part.to_string(),
            })?);
        }
        Ok(v)
    }

    fn pair(&mut self, key: &'static str) -> Result<[usize; 2], TextError> {
        let v = self.dims(key)?;
        if v.len() == 2 {
            Ok([v[0], v[1]])
        } else {
            Err(TextError::BadArity { line: self.line, key, want: 2, got: v.len() })
        }
    }

    fn floats(&mut self, key: &'static str) -> Result<Vec<f64>, TextError> {
        let raw = self.take(key)?;
        let inner = raw
            .strip_prefix('[')
            .and_then(|r| r.strip_suffix(']'))
            .ok_or_else(|| TextError::BadValue {
                line: self.line,
                key,
                text: raw.to_string(),
            })?;
        if inner.is_empty() {
            return Ok(Vec::new());
        }
        let mut v = Vec::new();
        for part in inner.split(',') {
            let x = part.parse::<f64>().map_err(|_| TextError::BadValue {
                line: self.line,
                key,
                text: part.to_string(),
            })?;
            if !x.is_finite() {
                return Err(TextError::NonFiniteValue {
                    line: self.line,
                    key,
                    text: part.to_string(),
                });
            }
            v.push(x);
        }
        Ok(v)
    }

    fn done(self) -> Result<(), TextError> {
        for (k, _, used) in &self.items {
            if !used {
                return Err(TextError::SurplusField {
                    line: self.line,
                    kind: self.kind,
                    key: (*k).to_string(),
                });
            }
        }
        Ok(())
    }
}

impl Graph {
    /// Write the graph as text.
    ///
    /// # The format
    ///
    /// ```text
    /// ferromorphic-nir 1
    /// node in Input shape=[2]
    /// node w Linear rows=2 cols=2 weight=[0.0,0.001,0.002,0.0]
    /// edge in w
    /// ```
    ///
    /// One node or edge per line, whitespace-separated, `key=value` parameters, arrays in brackets
    /// with **no spaces**. Blank lines and lines starting with `#` are accepted by the reader and
    /// are **not** reproduced by the writer, so the identity this format guarantees is over the
    /// graph and over the writer's own canonical text, not over an arbitrary hand-edited file.
    ///
    /// Floats are written by [`fmt_f64`], which verifies per value that the decimal it emits parses
    /// back to the same bits. `graph -> text -> graph` is therefore the identity on every float,
    /// including negative zero and subnormals.
    ///
    /// This does **not** validate. A graph holding a `NaN` will be written with `NaN` in it, and
    /// [`Graph::from_text`] will then refuse to read it back by name. Call [`Graph::validate`]
    /// first if you want the refusal at the point the mistake was made.
    #[must_use]
    pub fn to_text(&self) -> String {
        let mut out = String::new();
        out.push_str(MAGIC);
        out.push('\n');
        for named in &self.nodes {
            out.push_str("node ");
            out.push_str(&named.name);
            out.push(' ');
            out.push_str(named.node.kind().name());
            match &named.node {
                Node::Input(x) => w_dims(&mut out, "shape", &x.shape),
                Node::Output(x) => w_dims(&mut out, "shape", &x.shape),
                Node::Affine(x) => {
                    w_usize(&mut out, "rows", x.rows);
                    w_usize(&mut out, "cols", x.cols);
                    w_floats(&mut out, "weight", &x.weight);
                    w_floats(&mut out, "bias", &x.bias);
                }
                Node::Linear(x) => {
                    w_usize(&mut out, "rows", x.rows);
                    w_usize(&mut out, "cols", x.cols);
                    w_floats(&mut out, "weight", &x.weight);
                }
                Node::Conv1d(x) => {
                    w_usize(&mut out, "in_channels", x.in_channels);
                    w_usize(&mut out, "out_channels", x.out_channels);
                    w_usize(&mut out, "length", x.length);
                    w_usize(&mut out, "kernel", x.kernel);
                    w_usize(&mut out, "stride", x.stride);
                    w_usize(&mut out, "padding", x.padding);
                    w_usize(&mut out, "dilation", x.dilation);
                    w_usize(&mut out, "groups", x.groups);
                    w_floats(&mut out, "weight", &x.weight);
                    w_floats(&mut out, "bias", &x.bias);
                }
                Node::Conv2d(x) => {
                    w_usize(&mut out, "in_channels", x.in_channels);
                    w_usize(&mut out, "out_channels", x.out_channels);
                    w_dims(&mut out, "size", &x.size);
                    w_dims(&mut out, "kernel", &x.kernel);
                    w_dims(&mut out, "stride", &x.stride);
                    w_dims(&mut out, "padding", &x.padding);
                    w_dims(&mut out, "dilation", &x.dilation);
                    w_usize(&mut out, "groups", x.groups);
                    w_floats(&mut out, "weight", &x.weight);
                    w_floats(&mut out, "bias", &x.bias);
                }
                Node::SumPool2d(x) | Node::AvgPool2d(x) => {
                    w_usize(&mut out, "channels", x.channels);
                    w_dims(&mut out, "size", &x.size);
                    w_dims(&mut out, "kernel", &x.kernel);
                    w_dims(&mut out, "stride", &x.stride);
                    w_dims(&mut out, "padding", &x.padding);
                }
                Node::Flatten(x) => {
                    w_dims(&mut out, "size", &x.size);
                    w_usize(&mut out, "start_dim", x.start_dim);
                    w_usize(&mut out, "end_dim", x.end_dim);
                }
                Node::I(x) => {
                    w_dims(&mut out, "shape", &x.shape);
                    w_floats(&mut out, "r", &x.r);
                }
                Node::If(x) => {
                    w_dims(&mut out, "shape", &x.shape);
                    w_floats(&mut out, "r", &x.r);
                    w_floats(&mut out, "v_threshold", &x.v_threshold);
                    w_floats(&mut out, "v_reset", &x.v_reset);
                }
                Node::Li(x) => {
                    w_dims(&mut out, "shape", &x.shape);
                    w_floats(&mut out, "tau", &x.tau);
                    w_floats(&mut out, "r", &x.r);
                    w_floats(&mut out, "v_leak", &x.v_leak);
                }
                Node::Lif(x) => {
                    w_dims(&mut out, "shape", &x.shape);
                    w_floats(&mut out, "tau", &x.tau);
                    w_floats(&mut out, "r", &x.r);
                    w_floats(&mut out, "v_leak", &x.v_leak);
                    w_floats(&mut out, "v_threshold", &x.v_threshold);
                    w_floats(&mut out, "v_reset", &x.v_reset);
                }
                Node::CubaLif(x) => {
                    w_dims(&mut out, "shape", &x.shape);
                    w_floats(&mut out, "tau_syn", &x.tau_syn);
                    w_floats(&mut out, "tau_mem", &x.tau_mem);
                    w_floats(&mut out, "r", &x.r);
                    w_floats(&mut out, "v_leak", &x.v_leak);
                    w_floats(&mut out, "v_threshold", &x.v_threshold);
                    w_floats(&mut out, "v_reset", &x.v_reset);
                    w_floats(&mut out, "w_in", &x.w_in);
                }
                Node::Threshold(x) => {
                    w_dims(&mut out, "shape", &x.shape);
                    w_floats(&mut out, "threshold", &x.threshold);
                }
                Node::Delay(x) => {
                    w_dims(&mut out, "shape", &x.shape);
                    w_floats(&mut out, "delay", &x.delay);
                }
                Node::Scale(x) => {
                    w_dims(&mut out, "shape", &x.shape);
                    w_floats(&mut out, "scale", &x.scale);
                }
            }
            out.push('\n');
        }
        for e in &self.edges {
            out.push_str("edge ");
            out.push_str(&e.from);
            out.push(' ');
            out.push_str(&e.to);
            out.push('\n');
        }
        out
    }

    /// Read a graph back from [`Graph::to_text`]'s output.
    ///
    /// Blank lines and `#` comments are skipped. Node and edge order is the order they appear in,
    /// which is what makes the round trip an identity rather than a permutation.
    ///
    /// # Errors
    ///
    /// A [`TextError`] naming the 1-based line: a wrong magic line, an unknown directive or node
    /// type, a missing, repeated or surplus parameter, a value that does not parse, a non-finite
    /// float, or a fixed-length parameter of the wrong arity. Nothing is skipped and nothing is
    /// defaulted — a file this reader accepts holds exactly the graph it returns.
    pub fn from_text(text: &str) -> Result<Self, TextError> {
        let mut g = Self::new();
        let mut seen_magic = false;
        for (k, raw) in text.lines().enumerate() {
            let line = k + 1;
            let trimmed = raw.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            if !seen_magic {
                if trimmed != MAGIC {
                    return Err(TextError::WrongMagic { found: trimmed.to_string() });
                }
                seen_magic = true;
                continue;
            }
            let toks: Vec<&str> = trimmed.split_whitespace().collect();
            match toks[0] {
                "node" => {
                    let (name, node) = parse_node(line, &toks)?;
                    g.nodes.push(Named { name, node });
                }
                "edge" => {
                    if toks.len() != 3 {
                        return Err(TextError::ShortLine { line, want: "edge <from> <to>" });
                    }
                    g.edges.push(Edge { from: toks[1].to_string(), to: toks[2].to_string() });
                }
                other => {
                    return Err(TextError::UnknownDirective { line, word: other.to_string() });
                }
            }
        }
        if seen_magic {
            Ok(g)
        } else {
            Err(TextError::WrongMagic { found: String::new() })
        }
    }
}

fn parse_node(line: usize, toks: &[&str]) -> Result<(String, Node), TextError> {
    if toks.len() < 3 {
        return Err(TextError::ShortLine { line, want: "node <name> <Type> [key=value ...]" });
    }
    let name = toks[1].to_string();
    if !name_is_writable(&name) {
        return Err(TextError::UnwritableName { line, name });
    }
    let Some(kind) = Kind::from_name(toks[2]) else {
        return Err(TextError::UnknownNodeType { line, found: toks[2].to_string() });
    };
    let mut f = Fields::new(line, kind.name(), &toks[3..])?;
    let node = match kind {
        Kind::Input => Node::Input(Input { shape: f.dims("shape")? }),
        Kind::Output => Node::Output(Output { shape: f.dims("shape")? }),
        Kind::Affine => Node::Affine(Affine {
            rows: f.usize_of("rows")?,
            cols: f.usize_of("cols")?,
            weight: f.floats("weight")?,
            bias: f.floats("bias")?,
        }),
        Kind::Linear => Node::Linear(Linear {
            rows: f.usize_of("rows")?,
            cols: f.usize_of("cols")?,
            weight: f.floats("weight")?,
        }),
        Kind::Conv1d => Node::Conv1d(Conv1d {
            in_channels: f.usize_of("in_channels")?,
            out_channels: f.usize_of("out_channels")?,
            length: f.usize_of("length")?,
            kernel: f.usize_of("kernel")?,
            stride: f.usize_of("stride")?,
            padding: f.usize_of("padding")?,
            dilation: f.usize_of("dilation")?,
            groups: f.usize_of("groups")?,
            weight: f.floats("weight")?,
            bias: f.floats("bias")?,
        }),
        Kind::Conv2d => Node::Conv2d(Conv2d {
            in_channels: f.usize_of("in_channels")?,
            out_channels: f.usize_of("out_channels")?,
            size: f.pair("size")?,
            kernel: f.pair("kernel")?,
            stride: f.pair("stride")?,
            padding: f.pair("padding")?,
            dilation: f.pair("dilation")?,
            groups: f.usize_of("groups")?,
            weight: f.floats("weight")?,
            bias: f.floats("bias")?,
        }),
        Kind::SumPool2d | Kind::AvgPool2d => {
            let p = Pool2d {
                channels: f.usize_of("channels")?,
                size: f.pair("size")?,
                kernel: f.pair("kernel")?,
                stride: f.pair("stride")?,
                padding: f.pair("padding")?,
            };
            if kind == Kind::SumPool2d { Node::SumPool2d(p) } else { Node::AvgPool2d(p) }
        }
        Kind::Flatten => Node::Flatten(Flatten {
            size: f.dims("size")?,
            start_dim: f.usize_of("start_dim")?,
            end_dim: f.usize_of("end_dim")?,
        }),
        Kind::I => Node::I(I { shape: f.dims("shape")?, r: f.floats("r")? }),
        Kind::If => Node::If(If {
            shape: f.dims("shape")?,
            r: f.floats("r")?,
            v_threshold: f.floats("v_threshold")?,
            v_reset: f.floats("v_reset")?,
        }),
        Kind::Li => Node::Li(Li {
            shape: f.dims("shape")?,
            tau: f.floats("tau")?,
            r: f.floats("r")?,
            v_leak: f.floats("v_leak")?,
        }),
        Kind::Lif => Node::Lif(Lif {
            shape: f.dims("shape")?,
            tau: f.floats("tau")?,
            r: f.floats("r")?,
            v_leak: f.floats("v_leak")?,
            v_threshold: f.floats("v_threshold")?,
            v_reset: f.floats("v_reset")?,
        }),
        Kind::CubaLif => Node::CubaLif(CubaLif {
            shape: f.dims("shape")?,
            tau_syn: f.floats("tau_syn")?,
            tau_mem: f.floats("tau_mem")?,
            r: f.floats("r")?,
            v_leak: f.floats("v_leak")?,
            v_threshold: f.floats("v_threshold")?,
            v_reset: f.floats("v_reset")?,
            w_in: f.floats("w_in")?,
        }),
        Kind::Threshold => Node::Threshold(Threshold {
            shape: f.dims("shape")?,
            threshold: f.floats("threshold")?,
        }),
        Kind::Delay => Node::Delay(Delay { shape: f.dims("shape")?, delay: f.floats("delay")? }),
        Kind::Scale => Node::Scale(Scale { shape: f.dims("shape")?, scale: f.floats("scale")? }),
    };
    f.done()?;
    Ok((name, node))
}

// ---------------------------------------------------------------------------------------------
// The bridge into this crate
// ---------------------------------------------------------------------------------------------

/// Where a `LIF` node's elements ended up in the flat [`crate::net::Net`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Block {
    /// The `NIR` node name.
    pub node: String,
    /// Index of this population's first neuron in the network.
    pub base: usize,
    /// How many neurons it holds: the product of the node's shape. **Rank is lost here** — a `LIF`
    /// of shape `[2, 3]` becomes six flat neurons and comes back as shape `[6]`.
    pub len: usize,
}

/// The result of [`Graph::to_net`]: a runnable network plus the map back to the graph.
#[derive(Debug, Clone)]
pub struct Conversion {
    /// The connectivity, with one synapse per non-zero weight.
    pub net: Net,
    /// One neuron per element of every `LIF` node, in block order. Each starts at rest with no
    /// refractory countdown, because `NIR` carries parameters and not a trajectory.
    pub neurons: Vec<crate::neuron::Lif>,
    /// Where each `LIF` node's elements live, in graph order.
    pub blocks: Vec<Block>,
    /// `(Input node name, block index)` for every boundary that drives a population. The external
    /// current vector [`crate::sim::Sim::step`] takes is indexed by neuron, so this is how a caller
    /// knows which entries to fill.
    pub inputs: Vec<(String, usize)>,
    /// `(Output node name, block index)` for every boundary that reads a population.
    pub outputs: Vec<(String, usize)>,
}

/// Why a graph could not be converted, in either direction.
#[derive(Debug, Clone, PartialEq)]
pub enum BridgeError {
    /// The graph does not validate, so there is nothing to convert. Conversion runs
    /// [`Graph::validate`] first on purpose: half of what could go wrong here is already named
    /// there, and naming it twice, differently, is how two error vocabularies drift apart.
    Invalid(ValidationError),
    /// A node type that has no representation in [`crate::net::Net`], named rather than
    /// approximated.
    UnsupportedNode {
        /// The node's name in the graph.
        name: String,
        /// Its type.
        kind: Kind,
    },
    /// An [`Affine`] whose bias is not exactly zero. A `NIR` bias is a constant added to the
    /// pre-synaptic sum on every timestep; `Net` has no per-neuron constant term, and the nearest
    /// thing — a constant external current — is in amperes and would need a timestep and a
    /// resistance to convert. This implementation declines to pick them.
    BiasNotRepresentable {
        /// The `Affine` node.
        name: String,
        /// The first non-zero entry.
        index: usize,
    },
    /// A weight node with other than exactly one incoming and one outgoing edge. `Net` stores a
    /// synapse block between two named populations; a weight matrix feeding two places is two
    /// blocks and the graph should say so.
    WeightFanout {
        /// The weight node.
        name: String,
        /// Incoming edge count.
        incoming: usize,
        /// Outgoing edge count.
        outgoing: usize,
    },
    /// A weight node whose source or target is not a `LIF` population.
    WeightNotBetweenNeurons {
        /// The weight node.
        name: String,
        /// The neighbour that is not a `LIF`.
        neighbour: String,
        /// That neighbour's type.
        kind: Kind,
    },
    /// A [`Delay`] node that is not sitting between a weight node and a `LIF` population, or that
    /// two weight nodes share.
    DelayOutOfPlace {
        /// The `Delay` node.
        name: String,
    },
    /// A delay in seconds that is not a whole number of ticks at this `dt`. Refused rather than
    /// rounded: a rounded delay changes a coincidence-detection window, and a coincidence detector
    /// that stops coinciding still runs.
    DelayNotWholeTicks {
        /// The `Delay` node.
        name: String,
        /// The delay that failed, seconds.
        seconds: f64,
        /// The tick length it was measured against, seconds.
        dt: f64,
    },
    /// A delay that is a whole number of ticks but more of them than a `u32` can index.
    ///
    /// [`crate::net::Net`] stores a synaptic delay as a `u32`, so `u32::MAX` ticks is the
    /// representational ceiling and this is where it is enforced. Before this variant existed the
    /// cast `rounded as u32` **saturated**, which is what a Rust float-to-int cast does: a delay of
    /// `1e9` s at `dt = 1e-4` (a true count of `1e13` ticks) was accepted as `4294967295`, nine
    /// orders of magnitude wrong and silently so.
    ///
    /// ⚠ A delay that *is* inside the ceiling can still be expensive: [`crate::sim::Sim`] allocates
    /// `max_delay + 1` delivery buckets, so a network whose longest delay is a billion ticks asks
    /// for a billion of them. That cost is disclosed rather than capped, because the tick count a
    /// caller can afford is not this module's to choose.
    DelayTooManyTicks {
        /// The `Delay` node.
        name: String,
        /// The delay that failed, seconds.
        seconds: f64,
        /// The tick length it was measured against, seconds.
        dt: f64,
        /// The whole number of ticks it works out to, which does not fit in a `u32`.
        ticks: f64,
    },
    /// More neurons than [`crate::net::Net`]'s `u32` indices can address.
    ///
    /// `Net` names a neuron with a `u32`. A graph whose `LIF` elements outnumber `u32::MAX`, or a
    /// network whose dense `n * n` matrix would overflow a `usize`, has no `Net` or no `NIR` form.
    /// Named rather than clamped: `u32::try_from(..).unwrap_or(u32::MAX)` would have built a
    /// plausible-looking synapse onto neuron `4294967295`, and [`crate::net::NetBuilder::connect`]
    /// accepts that index whenever the network really is that large.
    TooManyNeurons {
        /// How many neurons were asked for.
        count: usize,
    },
    /// An edge between two node types the bridge has no rule for — two populations wired together
    /// with no weights between them, for instance, which `Net` cannot express because a synapse
    /// without a weight is not a synapse.
    UnsupportedEdge {
        /// Source node.
        from: String,
        /// Target node.
        to: String,
    },
    /// `dt` was not a positive finite number of seconds.
    BadTimeStep {
        /// The value supplied.
        dt: f64,
    },
    /// The graph holds no `LIF` node, so there are no neurons to build.
    NoNeurons,
    /// [`Graph::from_net`] was handed a neuron list that does not match the network.
    WrongNeuronCount {
        /// How many neurons were supplied.
        got: usize,
        /// How many the network has.
        want: usize,
    },
    /// [`Graph::from_net`] was handed a neuron with an absolute refractory period. `NIR` has no
    /// such parameter, so the only honest options are to refuse or to silently change the model's
    /// maximum firing rate.
    RefractoryNotRepresentable {
        /// Index of the neuron.
        neuron: usize,
        /// Its refractory period, seconds.
        t_ref: f64,
    },
    /// A neuron parameter that is not finite.
    NonFiniteNeuron {
        /// Index of the neuron.
        neuron: usize,
        /// Which parameter.
        field: &'static str,
    },
    /// Two synapses onto the same neuron carry different delays. `NIR`'s [`Delay`] node delays a
    /// signal **per element**, not per synapse, so this network has no `NIR` form. Named rather
    /// than resolved: picking one of the two would change spike times.
    SplitDelay {
        /// The postsynaptic neuron.
        post: u32,
        /// One delay seen, ticks.
        a: u32,
        /// The other, ticks.
        b: u32,
    },
    /// Two synapses share a presynaptic and a postsynaptic neuron. A dense matrix has one cell for
    /// them and summing them would be a modelling decision this function is not entitled to make.
    DuplicateSynapse {
        /// Presynaptic neuron.
        pre: u32,
        /// Postsynaptic neuron.
        post: u32,
    },
    /// The network builder refused. Unreachable after [`Graph::validate`] — the indices are
    /// constructed in range and the weights are already known finite — and carried rather than
    /// unwrapped, because a library that unwraps an impossible error turns its own bug into a crash
    /// in somebody else's process.
    Net(NetError),
}

impl fmt::Display for BridgeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(e) => write!(f, "the graph does not validate: {e}"),
            Self::UnsupportedNode { name, kind } => write!(
                f,
                "node {name} is a {kind}, which has no representation in a ferromorphic Net"
            ),
            Self::BiasNotRepresentable { name, index } => write!(
                f,
                "node {name}: bias[{index}] is non-zero, and a Net has no per-neuron constant term"
            ),
            Self::WeightFanout { name, incoming, outgoing } => write!(
                f,
                "node {name} has {incoming} incoming and {outgoing} outgoing edges; a weight node \
                 must have exactly one of each"
            ),
            Self::WeightNotBetweenNeurons { name, neighbour, kind } => write!(
                f,
                "node {name} is wired to {neighbour}, a {kind}; weights must sit between two LIF \
                 populations"
            ),
            Self::DelayOutOfPlace { name } => write!(
                f,
                "node {name} is a Delay that is not between a weight node and a LIF population"
            ),
            Self::DelayNotWholeTicks { name, seconds, dt } => write!(
                f,
                "node {name}: a delay of {seconds} s is not a whole number of {dt} s ticks"
            ),
            Self::DelayTooManyTicks { name, seconds, dt, ticks } => write!(
                f,
                "node {name}: a delay of {seconds} s is {ticks} ticks of {dt} s, and a Net stores \
                 a delay in a u32"
            ),
            Self::TooManyNeurons { count } => {
                write!(f, "{count} neurons is past what a Net's u32 indices can address")
            }
            Self::UnsupportedEdge { from, to } => {
                write!(f, "edge {from} -> {to} has no representation in a Net")
            }
            Self::BadTimeStep { dt } => write!(f, "dt {dt} is not a positive finite number of seconds"),
            Self::NoNeurons => f.write_str("the graph holds no LIF node, so it has no neurons"),
            Self::WrongNeuronCount { got, want } => {
                write!(f, "{got} neurons supplied for a network of {want}")
            }
            Self::RefractoryNotRepresentable { neuron, t_ref } => write!(
                f,
                "neuron {neuron} has a {t_ref} s refractory period and NIR has no such parameter"
            ),
            Self::NonFiniteNeuron { neuron, field } => {
                write!(f, "neuron {neuron}: {field} is not finite")
            }
            Self::SplitDelay { post, a, b } => write!(
                f,
                "neuron {post} receives synapses delayed {a} and {b} ticks, and NIR delays per \
                 element rather than per synapse"
            ),
            Self::DuplicateSynapse { pre, post } => {
                write!(f, "synapse {pre} -> {post} appears twice, and a dense matrix has one cell")
            }
            Self::Net(e) => write!(f, "the network builder refused: {e}"),
        }
    }
}

/// See the note on [`crate::net::NetError`].
impl std::error::Error for BridgeError {}

impl From<ValidationError> for BridgeError {
    fn from(e: ValidationError) -> Self {
        Self::Invalid(e)
    }
}

/// The weight matrix a weight node carries, whichever of the two kinds it is.
fn weights_of(node: &Node, name: &str) -> Result<(usize, usize, Vec<f64>), BridgeError> {
    match node {
        Node::Linear(x) => Ok((x.rows, x.cols, x.weight.clone())),
        Node::Affine(x) => {
            if let Some(index) = x.bias.iter().position(|&b| b != 0.0) {
                return Err(BridgeError::BiasNotRepresentable { name: name.to_string(), index });
            }
            Ok((x.rows, x.cols, x.weight.clone()))
        }
        other => Err(BridgeError::UnsupportedNode {
            name: name.to_string(),
            kind: other.kind(),
        }),
    }
}

impl Graph {
    /// Convert the `LIF` + weights subset of this graph into a runnable network.
    ///
    /// # The shape a graph has to have
    ///
    /// ```text
    /// Input? -> LIF -> (Linear | zero-bias Affine) -> Delay? -> LIF -> Output?
    /// ```
    ///
    /// Every `LIF` node becomes a contiguous block of neurons; every weight node becomes a dense
    /// block of synapses from its source population to its target population, one synapse per
    /// **non-zero** entry. Any other node type, and any other wiring, is refused **by name**.
    ///
    /// # Two places this is not information-preserving, both deliberate
    ///
    /// A weight of exactly `0.0` is **dropped** rather than stored. A stored zero synapse changes
    /// no spike time and is charged for by [`crate::ledger`] on every spike that crosses it, so
    /// keeping it would inflate an energy figure while doing nothing. [`Graph::from_net`] cannot
    /// tell a dropped zero from an absent connection.
    ///
    /// Rank is flattened: a `LIF` of shape `[2, 3]` becomes six neurons in a network that has no
    /// notion of rank. [`Block`] records what was flattened.
    ///
    /// # Errors
    ///
    /// [`BridgeError::Invalid`] if the graph does not validate; [`BridgeError::BadTimeStep`] for a
    /// `dt` that is not positive and finite; [`BridgeError::NoNeurons`] for a graph with no `LIF`;
    /// [`BridgeError::UnsupportedNode`] naming any node type outside the subset;
    /// [`BridgeError::BiasNotRepresentable`], [`BridgeError::WeightFanout`],
    /// [`BridgeError::WeightNotBetweenNeurons`], [`BridgeError::DelayOutOfPlace`],
    /// [`BridgeError::DelayNotWholeTicks`], [`BridgeError::DelayTooManyTicks`],
    /// [`BridgeError::UnsupportedEdge`], [`BridgeError::TooManyNeurons`] or [`BridgeError::Net`]
    /// as described on each.
    pub fn to_net(&self, dt: f64) -> Result<Conversion, BridgeError> {
        self.validate(&Rules::default())?;
        if !(dt > 0.0) || !dt.is_finite() {
            return Err(BridgeError::BadTimeStep { dt });
        }

        // 1. Every node must be a kind the subset covers, checked BEFORE the edge sweep below.
        //    Order matters for the message and only for the message: step 2 reports an edge, so a
        //    CubaLIF that is actually wired to something — which is every CubaLIF in a real file —
        //    used to come back as `UnsupportedEdge { from: "w", to: "cu" }` with its kind nowhere
        //    in it, and `UnsupportedNode` was reachable only by a node with no edges at all. The
        //    doc's promise is that an unsupported node is refused BY NAME, with its kind; this is
        //    where that promise is kept.
        for named in &self.nodes {
            let kind = named.node.kind();
            if !matches!(
                kind,
                Kind::Input | Kind::Output | Kind::Affine | Kind::Linear | Kind::Lif | Kind::Delay
            ) {
                return Err(BridgeError::UnsupportedNode { name: named.name.clone(), kind });
            }
        }

        // 2. Lay the LIF populations out in graph order.
        let mut blocks: Vec<Block> = Vec::new();
        let mut block_of: Vec<Option<usize>> = vec![None; self.nodes.len()];
        let mut total = 0usize;
        for (i, named) in self.nodes.iter().enumerate() {
            if let Node::Lif(x) = &named.node {
                let len = product(&x.shape);
                block_of[i] = Some(blocks.len());
                blocks.push(Block { node: named.name.clone(), base: total, len });
                total += len;
            }
        }
        if blocks.is_empty() {
            return Err(BridgeError::NoNeurons);
        }

        // 3. Every edge must be one of the shapes the subset allows. Checked before anything is
        //    built, so an unsupported wiring is reported rather than half-converted.
        for e in &self.edges {
            let (Some(a), Some(b)) = (self.index_of(&e.from), self.index_of(&e.to)) else {
                continue; // validate() already refused a dangling edge
            };
            let (ka, kb) = (self.nodes[a].node.kind(), self.nodes[b].node.kind());
            let ok = matches!(
                (ka, kb),
                (Kind::Input, Kind::Lif)
                    | (Kind::Lif, Kind::Output)
                    | (Kind::Lif, Kind::Affine | Kind::Linear)
                    | (Kind::Affine | Kind::Linear, Kind::Lif | Kind::Delay)
                    | (Kind::Delay, Kind::Lif)
            );
            if !ok {
                return Err(BridgeError::UnsupportedEdge {
                    from: e.from.clone(),
                    to: e.to.clone(),
                });
            }
        }

        // 4. Walk the nodes, consuming Delay nodes from the weight node in front of them.
        let mut builder = NetBuilder::new(total);
        let mut delay_used = vec![false; self.nodes.len()];
        let mut inputs = Vec::new();
        let mut outputs = Vec::new();

        for named in &self.nodes {
            match named.node.kind() {
                Kind::Lif | Kind::Delay => {}
                Kind::Input => {
                    for e in self.edges.iter().filter(|e| e.from == named.name) {
                        let Some(t) = self.index_of(&e.to) else { continue };
                        if let Some(bi) = block_of[t] {
                            inputs.push((named.name.clone(), bi));
                        }
                    }
                }
                Kind::Output => {
                    for e in self.edges.iter().filter(|e| e.to == named.name) {
                        let Some(s) = self.index_of(&e.from) else { continue };
                        if let Some(bi) = block_of[s] {
                            outputs.push((named.name.clone(), bi));
                        }
                    }
                }
                Kind::Affine | Kind::Linear => {
                    let (rows, cols, w) = weights_of(&named.node, &named.name)?;
                    let ins: Vec<&Edge> =
                        self.edges.iter().filter(|e| e.to == named.name).collect();
                    let outs: Vec<&Edge> =
                        self.edges.iter().filter(|e| e.from == named.name).collect();
                    if ins.len() != 1 || outs.len() != 1 {
                        return Err(BridgeError::WeightFanout {
                            name: named.name.clone(),
                            incoming: ins.len(),
                            outgoing: outs.len(),
                        });
                    }
                    let src = self.index_of(&ins[0].from).ok_or_else(|| {
                        BridgeError::UnsupportedEdge {
                            from: ins[0].from.clone(),
                            to: named.name.clone(),
                        }
                    })?;
                    let src_block = block_of[src].ok_or_else(|| {
                        BridgeError::WeightNotBetweenNeurons {
                            name: named.name.clone(),
                            neighbour: self.nodes[src].name.clone(),
                            kind: self.nodes[src].node.kind(),
                        }
                    })?;

                    // Follow an optional Delay node to the target population.
                    let mid = self.index_of(&outs[0].to).ok_or_else(|| {
                        BridgeError::UnsupportedEdge {
                            from: named.name.clone(),
                            to: outs[0].to.clone(),
                        }
                    })?;
                    let (dst, delay_ticks) = if let Node::Delay(d) = &self.nodes[mid].node {
                        if delay_used[mid] {
                            return Err(BridgeError::DelayOutOfPlace {
                                name: self.nodes[mid].name.clone(),
                            });
                        }
                        delay_used[mid] = true;
                        let douts: Vec<&Edge> = self
                            .edges
                            .iter()
                            .filter(|e| e.from == self.nodes[mid].name)
                            .collect();
                        if douts.len() != 1 {
                            return Err(BridgeError::DelayOutOfPlace {
                                name: self.nodes[mid].name.clone(),
                            });
                        }
                        let after = self.index_of(&douts[0].to).ok_or_else(|| {
                            BridgeError::DelayOutOfPlace { name: self.nodes[mid].name.clone() }
                        })?;
                        let ticks = ticks_of(&self.nodes[mid].name, &d.delay, dt)?;
                        (after, ticks)
                    } else {
                        (mid, vec![0u32; rows])
                    };
                    let dst_block = block_of[dst].ok_or_else(|| {
                        BridgeError::WeightNotBetweenNeurons {
                            name: named.name.clone(),
                            neighbour: self.nodes[dst].name.clone(),
                            kind: self.nodes[dst].node.kind(),
                        }
                    })?;

                    // `validate` has already matched [cols] against the source's output shape and
                    // [rows] against the target's input shape, so these products agree — but this
                    // function is `pub` and the agreement is an argument, not a type. A short delay
                    // array silently became zero delay here, and an out-of-range neuron index
                    // silently became a synapse onto neuron 4294967295, which
                    // `NetBuilder::connect` accepts whenever the network really is that large.
                    if delay_ticks.len() != rows {
                        return Err(BridgeError::Invalid(ValidationError::RaggedParameter {
                            node: self.nodes[mid].name.clone(),
                            field: "delay",
                            len: delay_ticks.len(),
                            expected: rows,
                        }));
                    }
                    let src_base = blocks[src_block].base;
                    let dst_base = blocks[dst_block].base;
                    for row in 0..rows {
                        let d = delay_ticks[row];
                        for col in 0..cols {
                            let weight = w[row * cols + col];
                            if weight == 0.0 {
                                continue; // dropped, and the doc says so
                            }
                            let pre = u32::try_from(src_base + col)
                                .map_err(|_| BridgeError::TooManyNeurons { count: total })?;
                            let post = u32::try_from(dst_base + row)
                                .map_err(|_| BridgeError::TooManyNeurons { count: total })?;
                            builder
                                .connect(pre, post, weight, d)
                                .map_err(BridgeError::Net)?;
                        }
                    }
                }
                other => {
                    return Err(BridgeError::UnsupportedNode {
                        name: named.name.clone(),
                        kind: other,
                    });
                }
            }
        }

        // Every Delay node must have been consumed by the weight node in front of it.
        for (i, named) in self.nodes.iter().enumerate() {
            if named.node.kind() == Kind::Delay && !delay_used[i] {
                return Err(BridgeError::DelayOutOfPlace { name: named.name.clone() });
            }
        }

        // 5. Build the neurons, one per element, parameters copied bit for bit.
        let mut neurons = Vec::with_capacity(total);
        for (i, named) in self.nodes.iter().enumerate() {
            if block_of[i].is_none() {
                continue;
            }
            let Node::Lif(x) = &named.node else { continue };
            for k in 0..product(&x.shape) {
                neurons.push(crate::neuron::Lif {
                    tau_m: x.tau[k],
                    v_rest: x.v_leak[k],
                    v_th: x.v_threshold[k],
                    v_reset: x.v_reset[k],
                    r_m: x.r[k],
                    // NIR has no refractory period. Zero is the faithful value, not a default.
                    t_ref: 0.0,
                    v: x.v_leak[k],
                    refractory: 0.0,
                });
            }
        }

        Ok(Conversion { net: builder.build(), neurons, blocks, inputs, outputs })
    }

    /// Rebuild a `NIR` graph from a network and its neurons.
    ///
    /// The graph produced is always the same five-node shape — `input -> neurons`,
    /// `neurons -> weights -> [delay ->] neurons`, `neurons -> output` — with **one dense
    /// `n * n` weight matrix**, because `NIR`'s [`Linear`] node has no sparse form. That is a
    /// property of the IR and not of this implementation, and it is the reason a 10,000-neuron
    /// network becomes an 800 MB node. The block structure a [`Conversion`] recorded is not
    /// recovered; the weights are, bit for bit.
    ///
    /// ⚠ **The graph returned is cyclic, whatever the network was.** Collapsing every population
    /// into one `neurons` node turns `neurons -> weights -> neurons` into a self-loop through the
    /// weight node, so a feedforward chain comes back as a cycle. The graph validates under
    /// [`Rules::default`] and **never** under [`Rules::feedforward`], which fails it with
    /// [`ValidationError::Cycle`] naming `["neurons", "weights", "output"]`. Since
    /// `Rules::feedforward` exists for the layer-by-layer consumers, no output of this function is
    /// consumable by one. Recovering the layering would need the block structure this function
    /// does not have.
    ///
    /// # Errors
    ///
    /// [`BridgeError::WrongNeuronCount`], [`BridgeError::BadTimeStep`], [`BridgeError::NoNeurons`],
    /// [`BridgeError::RefractoryNotRepresentable`] for a neuron `NIR` cannot describe,
    /// [`BridgeError::NonFiniteNeuron`], [`BridgeError::SplitDelay`] where a neuron's incoming
    /// synapses disagree about their delay, [`BridgeError::DuplicateSynapse`], or
    /// [`BridgeError::TooManyNeurons`].
    pub fn from_net(
        net: &Net,
        neurons: &[crate::neuron::Lif],
        dt: f64,
    ) -> Result<Self, BridgeError> {
        if neurons.len() != net.n {
            return Err(BridgeError::WrongNeuronCount { got: neurons.len(), want: net.n });
        }
        if !(dt > 0.0) || !dt.is_finite() {
            return Err(BridgeError::BadTimeStep { dt });
        }
        let n = net.n;
        if n == 0 {
            return Err(BridgeError::NoNeurons);
        }
        for (k, neuron) in neurons.iter().enumerate() {
            if neuron.t_ref != 0.0 {
                return Err(BridgeError::RefractoryNotRepresentable {
                    neuron: k,
                    t_ref: neuron.t_ref,
                });
            }
            for (field, value) in [
                ("tau", neuron.tau_m),
                ("r", neuron.r_m),
                ("v_leak", neuron.v_rest),
                ("v_threshold", neuron.v_th),
                ("v_reset", neuron.v_reset),
            ] {
                if !value.is_finite() {
                    return Err(BridgeError::NonFiniteNeuron { neuron: k, field });
                }
            }
        }

        // `n * n` is the dense matrix NIR's Linear node forces; an `n` past a `u32` also has no
        // Net-side index, so both are refused here rather than wrapping or clamping.
        let cells = n.checked_mul(n).ok_or(BridgeError::TooManyNeurons { count: n })?;
        if u32::try_from(n - 1).is_err() {
            return Err(BridgeError::TooManyNeurons { count: n });
        }
        let mut dense = vec![0.0f64; cells];
        let mut filled = vec![false; cells];
        let mut delay_of: Vec<Option<u32>> = vec![None; n];
        for pre in 0..n {
            for (post, w, d) in net.out_of(pre) {
                let cell = post as usize * n + pre;
                if filled[cell] {
                    // `pre` is below `n` and `n` fits a `u32` by the check above, so this cast is
                    // exact rather than clamped.
                    let pre = u32::try_from(pre)
                        .map_err(|_| BridgeError::TooManyNeurons { count: n })?;
                    return Err(BridgeError::DuplicateSynapse { pre, post });
                }
                filled[cell] = true;
                dense[cell] = w;
                match delay_of[post as usize] {
                    None => delay_of[post as usize] = Some(d),
                    Some(seen) if seen != d => {
                        return Err(BridgeError::SplitDelay { post, a: seen, b: d });
                    }
                    Some(_) => {}
                }
            }
        }

        let shape = vec![n];
        let mut g = Self::new();
        g.push("input", Node::Input(Input { shape: shape.clone() }));
        g.push(
            "neurons",
            Node::Lif(Lif {
                shape: shape.clone(),
                tau: neurons.iter().map(|x| x.tau_m).collect(),
                r: neurons.iter().map(|x| x.r_m).collect(),
                v_leak: neurons.iter().map(|x| x.v_rest).collect(),
                v_threshold: neurons.iter().map(|x| x.v_th).collect(),
                v_reset: neurons.iter().map(|x| x.v_reset).collect(),
            }),
        );
        g.push("weights", Node::Linear(Linear { rows: n, cols: n, weight: dense }));
        let any_delay = delay_of.iter().any(|d| d.is_some_and(|t| t > 0));
        if any_delay {
            g.push(
                "delay",
                Node::Delay(Delay {
                    shape: shape.clone(),
                    delay: delay_of
                        .iter()
                        .map(|d| f64::from(d.unwrap_or(0)) * dt)
                        .collect(),
                }),
            );
        }
        g.push("output", Node::Output(Output { shape }));
        g.edge("input", "neurons");
        g.edge("neurons", "weights");
        if any_delay {
            g.edge("weights", "delay");
            g.edge("delay", "neurons");
        } else {
            g.edge("weights", "neurons");
        }
        g.edge("neurons", "output");
        Ok(g)
    }
}

/// Seconds to whole ticks, refusing anything in between and anything past a `u32`.
///
/// The second refusal is not decoration. `rounded as u32` is a saturating cast in Rust, so without
/// the bound a delay of `1e30` s at `dt = 1e-4` — `1e34` ticks — became `4294967295` with no error
/// at all, and `crate::sim::Sim` then tried to allocate that many delivery buckets. The `NIR` Units
/// section says why such a number reaches this function in practice: a graph normalised for
/// training carries `delay` in **timesteps**, and reading timesteps as seconds multiplies the tick
/// count by `1 / dt`.
fn ticks_of(name: &str, delay: &[f64], dt: f64) -> Result<Vec<u32>, BridgeError> {
    let mut out = Vec::with_capacity(delay.len());
    for &seconds in delay {
        let exact = seconds / dt;
        let rounded = exact.round();
        // A relative tolerance, because `ticks * dt` is not exactly `seconds` in binary floating
        // point for any realistic dt — 2e-3 / 1e-4 is 19.999999999999996, not 20.
        if !(rounded >= 0.0) || (exact - rounded).abs() > 1e-9 * rounded.max(1.0) {
            return Err(BridgeError::DelayNotWholeTicks {
                name: name.to_string(),
                seconds,
                dt,
            });
        }
        // `u32::MAX as f64` is exact, and `rounded` is whole, so this comparison is exact and the
        // cast below cannot saturate.
        if rounded > f64::from(u32::MAX) {
            return Err(BridgeError::DelayTooManyTicks {
                name: name.to_string(),
                seconds,
                dt,
                ticks: rounded,
            });
        }
        out.push(rounded as u32);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::{
        Affine, Block, BridgeError, Conv1d, Conv2d, CubaLif, CubaState, Delay, Flatten,
        Graph, I, If, Input, Kind, Li, Lif, Linear, MAGIC, Named, Node, Output, Pool2d, Rules,
        Scale, Signal, TextError, Threshold, ValidationError, checked_product, fmt_f64,
        fmt_f64_long, product,
    };
    use crate::neuron::Neuron;

    /// Floats chosen to break a lazy codec: a negative zero that `PartialEq` cannot see, the
    /// smallest subnormal, the largest finite value, and two decimals with no exact binary form.
    const AWKWARD: [f64; 8] = [
        -0.0,
        5e-324,
        f64::MIN_POSITIVE,
        f64::MAX,
        -1e300,
        0.1,
        core::f64::consts::PI,
        -2.5e-7,
    ];

    /// One graph holding **every** node type this module implements, wired where the shapes line
    /// up and left unconnected where they do not. It validates, so the round-trip test is over a
    /// legal graph rather than a bag of structs.
    fn every_node_type() -> Graph {
        let mut g = Graph::new();
        g.push("in", Node::Input(Input { shape: vec![2] }));
        g.push(
            "aff",
            Node::Affine(Affine {
                rows: 3,
                cols: 2,
                weight: vec![5e-324, -1e300, 0.1, core::f64::consts::PI, f64::MAX, -2.5e-7],
                bias: vec![-0.0, 0.0, 1e-3],
            }),
        );
        g.push(
            "n1",
            Node::Lif(Lif {
                shape: vec![3],
                tau: vec![f64::MIN_POSITIVE, 20e-3, 5e-3],
                r: vec![10e6, 1.0, 1e9],
                v_leak: vec![-65e-3, -0.0, 0.0],
                v_threshold: vec![-50e-3, 1.0, 0.5],
                v_reset: vec![-65e-3, 0.0, -0.0],
            }),
        );
        g.push("out", Node::Output(Output { shape: vec![3] }));
        // Everything below is legal on its own and deliberately unconnected: the point is that the
        // writer and reader agree about every node type, not that one chain uses them all.
        g.push(
            "lin",
            Node::Linear(Linear { rows: 2, cols: 2, weight: vec![0.0, -0.0, 1.5, -1.5] }),
        );
        g.push(
            "c1",
            Node::Conv1d(Conv1d {
                in_channels: 2,
                out_channels: 3,
                length: 8,
                kernel: 3,
                stride: 2,
                padding: 1,
                dilation: 1,
                groups: 1,
                weight: (0..18).map(|k| f64::from(k) * 0.125).collect(),
                bias: vec![0.0, -0.0, 1e-9],
            }),
        );
        g.push(
            "c2",
            Node::Conv2d(Conv2d {
                in_channels: 1,
                out_channels: 2,
                size: [5, 5],
                kernel: [3, 3],
                stride: [1, 1],
                padding: [1, 1],
                dilation: [1, 1],
                groups: 1,
                weight: (0..18).map(|k| -f64::from(k) * 0.0625).collect(),
                // Empty, which is how NIR encodes a convolution built with bias=False.
                bias: Vec::new(),
            }),
        );
        g.push(
            "sp",
            Node::SumPool2d(Pool2d {
                channels: 2,
                size: [4, 4],
                kernel: [2, 2],
                stride: [2, 2],
                padding: [0, 0],
            }),
        );
        g.push(
            "ap",
            Node::AvgPool2d(Pool2d {
                channels: 2,
                size: [4, 4],
                kernel: [2, 2],
                stride: [2, 2],
                padding: [0, 0],
            }),
        );
        g.push("fl", Node::Flatten(Flatten { size: vec![2, 3, 4], start_dim: 1, end_dim: 2 }));
        g.push("ii", Node::I(I { shape: vec![2], r: vec![1e6, 2e6] }));
        g.push(
            "iff",
            Node::If(If {
                shape: vec![2],
                r: vec![1e6, 2e6],
                v_threshold: vec![1.0, 0.5],
                v_reset: vec![0.0, -0.0],
            }),
        );
        g.push(
            "li",
            Node::Li(Li {
                shape: vec![2],
                tau: vec![10e-3, 1.0],
                r: vec![1.0, 2.0],
                v_leak: vec![0.0, -65e-3],
            }),
        );
        g.push(
            "cu",
            Node::CubaLif(CubaLif {
                shape: vec![2],
                // Zero is legal and means the instantaneous-synapse limit.
                tau_syn: vec![0.0, 5e-3],
                tau_mem: vec![20e-3, 20e-3],
                r: vec![10e6, 10e6],
                v_leak: vec![-65e-3, -0.0],
                v_threshold: vec![-50e-3, 1.0],
                v_reset: vec![-65e-3, 0.0],
                w_in: vec![1.0, 2.5],
            }),
        );
        g.push("th", Node::Threshold(Threshold { shape: vec![2], threshold: vec![0.5, -0.0] }));
        g.push("dl", Node::Delay(Delay { shape: vec![2], delay: vec![0.0, 2e-3] }));
        g.push(
            "sc",
            Node::Scale(Scale { shape: vec![2], scale: vec![0.1, core::f64::consts::PI] }),
        );
        g.edge("in", "aff");
        g.edge("aff", "n1");
        g.edge("n1", "out");
        g
    }

    /// **The module's central check.** Every node type, every awkward float, both directions.
    ///
    /// Graph equality alone would not be enough: `-0.0 == 0.0` in `PartialEq`, so a codec that lost
    /// the sign of zero would pass it. Comparing the two texts catches that, because the writer
    /// prints `-0.0` and `0.0` differently.
    #[test]
    fn graph_to_text_to_graph_is_exactly_the_identity() {
        let g = every_node_type();
        g.validate(&Rules::default()).expect("the fixture graph must be legal");

        // The premise, asserted with its numbers: all seventeen kinds are present and the awkward
        // floats really are in the text. A round-trip test over a graph missing a node type is
        // green and worthless.
        assert_eq!(g.nodes.len(), 17, "the fixture must carry every node type");
        let kinds: Vec<Kind> = g.nodes.iter().map(|n| n.node.kind()).collect();
        for k in [
            Kind::Input,
            Kind::Output,
            Kind::Affine,
            Kind::Linear,
            Kind::Conv1d,
            Kind::Conv2d,
            Kind::SumPool2d,
            Kind::AvgPool2d,
            Kind::Flatten,
            Kind::I,
            Kind::If,
            Kind::Li,
            Kind::Lif,
            Kind::CubaLif,
            Kind::Threshold,
            Kind::Delay,
            Kind::Scale,
        ] {
            assert!(kinds.contains(&k), "{k} is missing from the fixture");
        }

        let first = g.to_text();
        assert!(first.contains("-0.0"), "the fixture lost its negative zero before the codec saw it");
        assert!(first.contains("5e-324"), "the fixture lost its subnormal");
        assert!(first.contains("1.7976931348623157e308"), "the fixture lost f64::MAX");

        let back = Graph::from_text(&first).expect("its own output must read back");
        assert_eq!(back, g, "graph -> text -> graph changed the graph");

        let second = back.to_text();
        assert_eq!(first, second, "text -> graph -> text changed a byte");
    }

    /// The codec's claim, on values chosen to break it, checked on **bits** rather than on `==`.
    #[test]
    fn the_float_codec_round_trips_bit_for_bit() {
        let mut extra = AWKWARD.to_vec();
        extra.extend_from_slice(&[0.0, 1.0, -1.0, 1e-300, f64::MIN, 1.0 / 3.0, 1e16 + 1.0]);
        for x in extra {
            let s = fmt_f64(x);
            let back: f64 = s.parse().expect("the codec must emit something parseable");
            assert_eq!(back.to_bits(), x.to_bits(), "{x:?} round-tripped as {s} -> {back:?}");
        }
    }

    /// Negative zero survives, which is the case `assert_eq!` on floats cannot see.
    #[test]
    fn negative_zero_survives_the_text_format() {
        let mut g = Graph::new();
        g.push("s", Node::Scale(Scale { shape: vec![2], scale: vec![-0.0, 0.0] }));
        let back = Graph::from_text(&g.to_text()).expect("round trip");
        let Node::Scale(s) = &back.nodes[0].node else { panic!("wrong node type came back") };
        assert!(s.scale[0].is_sign_negative(), "-0.0 came back as +0.0");
        assert!(s.scale[1].is_sign_positive(), "+0.0 came back as -0.0");
    }

    #[test]
    fn a_non_finite_value_in_a_file_is_refused_by_name() {
        let text = "ferromorphic-nir 1\nnode s Scale shape=[2] scale=[1.0,NaN]\n";
        match Graph::from_text(text) {
            Err(TextError::NonFiniteValue { line, key, text }) => {
                assert_eq!(line, 2);
                assert_eq!(key, "scale");
                assert_eq!(text, "NaN");
            }
            other => panic!("expected a non-finite refusal, got {other:?}"),
        }
        let inf = "ferromorphic-nir 1\nnode s Scale shape=[1] scale=[inf]\n";
        assert!(matches!(Graph::from_text(inf), Err(TextError::NonFiniteValue { .. })));
    }

    /// An unknown node type is refused rather than skipped: a skipped node leaves a graph smaller
    /// than its file, and it still runs.
    #[test]
    fn an_unknown_node_type_is_named_rather_than_skipped() {
        let text = "ferromorphic-nir 1\nnode x Izhikevich shape=[1]\n";
        match Graph::from_text(text) {
            Err(TextError::UnknownNodeType { line, found }) => {
                assert_eq!(line, 2);
                assert_eq!(found, "Izhikevich");
            }
            other => panic!("expected an unknown-type refusal, got {other:?}"),
        }
    }

    #[test]
    fn a_missing_a_surplus_and_a_repeated_field_are_all_refused() {
        let missing = "ferromorphic-nir 1\nnode s Scale shape=[1]\n";
        assert!(matches!(
            Graph::from_text(missing),
            Err(TextError::AbsentField { key: "scale", .. })
        ));

        let surplus = "ferromorphic-nir 1\nnode s Scale shape=[1] scale=[1.0] tau=[1.0]\n";
        match Graph::from_text(surplus) {
            Err(TextError::SurplusField { kind, key, .. }) => {
                assert_eq!(kind, "Scale");
                assert_eq!(key, "tau");
            }
            other => panic!("expected a surplus-field refusal, got {other:?}"),
        }

        let repeated = "ferromorphic-nir 1\nnode s Scale shape=[1] scale=[1.0] scale=[2.0]\n";
        assert!(matches!(Graph::from_text(repeated), Err(TextError::RepeatedField { .. })));
    }

    #[test]
    fn a_wrong_magic_line_and_a_wrong_directive_are_refused() {
        assert!(matches!(Graph::from_text("nir 2\n"), Err(TextError::WrongMagic { .. })));
        assert!(matches!(Graph::from_text(""), Err(TextError::WrongMagic { .. })));
        let bad = "ferromorphic-nir 1\nvertex s Scale\n";
        match Graph::from_text(bad) {
            Err(TextError::UnknownDirective { line, word }) => {
                assert_eq!(line, 2);
                assert_eq!(word, "vertex");
            }
            other => panic!("expected an unknown-directive refusal, got {other:?}"),
        }
    }

    #[test]
    fn a_fixed_length_field_of_the_wrong_arity_is_refused() {
        let text =
            "ferromorphic-nir 1\nnode p SumPool2d channels=1 size=[4,4] kernel=[2] stride=[2,2] padding=[0,0]\n";
        match Graph::from_text(text) {
            Err(TextError::BadArity { key, want, got, .. }) => {
                assert_eq!((key, want, got), ("kernel", 2, 1));
            }
            other => panic!("expected an arity refusal, got {other:?}"),
        }
    }

    /// Comments and blank lines are read and are not written back, which the doc states and this
    /// pins: the identity is over the graph and over the writer's canonical text.
    #[test]
    fn comments_are_read_and_are_not_reproduced() {
        let text = "# a note\n\nferromorphic-nir 1\n# another\nnode s Scale shape=[1] scale=[2.0]\n";
        let g = Graph::from_text(text).expect("comments are legal");
        assert_eq!(g.nodes.len(), 1);
        assert!(!g.to_text().contains('#'));
    }

    // -- validation -----------------------------------------------------------------------------

    /// A two-population chain, used as the base for the validation failures below.
    fn chain() -> Graph {
        let mut g = Graph::new();
        g.push("in", Node::Input(Input { shape: vec![2] }));
        g.push("a", Node::Lif(lif_params(2, 20e-3)));
        g.push("w", Node::Linear(Linear { rows: 3, cols: 2, weight: vec![1e-3; 6] }));
        g.push("b", Node::Lif(lif_params(3, 20e-3)));
        g.push("out", Node::Output(Output { shape: vec![3] }));
        g.edge("in", "a");
        g.edge("a", "w");
        g.edge("w", "b");
        g.edge("b", "out");
        g
    }

    fn lif_params(n: usize, tau: f64) -> Lif {
        Lif {
            shape: vec![n],
            tau: vec![tau; n],
            r: vec![10e6; n],
            v_leak: vec![-65e-3; n],
            v_threshold: vec![-50e-3; n],
            v_reset: vec![-65e-3; n],
        }
    }

    #[test]
    fn the_base_chain_validates_under_every_rule_set() {
        let g = chain();
        g.validate(&Rules::default()).expect("default");
        g.validate(&Rules::feedforward()).expect("feedforward: this chain has no cycle");
        g.validate(&Rules::recurrent()).expect("recurrent");
    }

    #[test]
    fn validation_catches_a_dangling_edge() {
        let mut g = chain();
        g.edge("b", "nowhere");
        match g.validate(&Rules::default()) {
            Err(ValidationError::DanglingEdge { from, to, missing }) => {
                assert_eq!((from.as_str(), to.as_str(), missing.as_str()), ("b", "nowhere", "nowhere"));
            }
            other => panic!("expected a dangling edge, got {other:?}"),
        }
    }

    #[test]
    fn validation_catches_a_shape_mismatch() {
        let mut g = chain();
        // The weight node now emits four values into a three-element population.
        g.nodes[2].node = Node::Linear(Linear { rows: 4, cols: 2, weight: vec![1e-3; 8] });
        match g.validate(&Rules::default()) {
            Err(ValidationError::ShapeMismatch { from, to, emitted, expected }) => {
                assert_eq!((from.as_str(), to.as_str()), ("w", "b"));
                assert_eq!(emitted, vec![4]);
                assert_eq!(expected, vec![3]);
            }
            other => panic!("expected a shape mismatch, got {other:?}"),
        }
    }

    /// A type error rather than a shape error: an `Input` node has no input port at all, so the
    /// edge is not merely wrong about widths.
    #[test]
    fn validation_catches_an_edge_into_a_node_with_no_input_port() {
        let mut g = chain();
        g.edge("b", "in");
        match g.validate(&Rules::default()) {
            Err(ValidationError::NoInputPort { from, into }) => {
                assert_eq!((from.as_str(), into.as_str()), ("b", "in"));
            }
            other => panic!("expected a port error, got {other:?}"),
        }
    }

    #[test]
    fn validation_catches_an_edge_out_of_a_node_with_no_output_port() {
        let mut g = chain();
        g.edge("out", "b");
        assert!(matches!(g.validate(&Rules::default()), Err(ValidationError::NoOutputPort { .. })));
    }

    /// The signal-kind check: a `Threshold` fed by a spiking population. This is the rule `NIR`
    /// itself does not have, so it is also the rule a caller can switch off.
    #[test]
    fn validation_catches_a_signal_kind_mismatch_and_can_be_told_not_to() {
        let mut g = chain();
        g.push("th", Node::Threshold(Threshold { shape: vec![3], threshold: vec![0.5; 3] }));
        g.edge("b", "th");
        match g.validate(&Rules::default()) {
            Err(ValidationError::SignalMismatch { from, to, emitted, required }) => {
                assert_eq!((from.as_str(), to.as_str()), ("b", "th"));
                assert_eq!(emitted, Signal::Spikes);
                assert_eq!(required, Signal::Continuous);
            }
            other => panic!("expected a signal mismatch, got {other:?}"),
        }
        let lenient = Rules { check_signal_kind: false, ..Rules::default() };
        g.validate(&lenient).expect("the check is switchable, and the graph is otherwise legal");
    }

    /// Recurrence is legal in `NIR` and legal here by default; only a caller that says it cannot
    /// execute a cycle gets the refusal, and it gets the nodes named.
    #[test]
    fn a_cycle_is_legal_by_default_and_named_when_it_is_not() {
        let mut g = chain();
        g.push("rec", Node::Linear(Linear { rows: 2, cols: 3, weight: vec![1e-3; 6] }));
        g.edge("b", "rec");
        g.edge("rec", "a");
        g.validate(&Rules::default()).expect("a recurrent graph is a legal NIR graph");
        assert!(!g.is_acyclic());
        match g.validate(&Rules::feedforward()) {
            Err(ValidationError::Cycle { nodes }) => {
                for name in ["a", "w", "b", "rec"] {
                    assert!(nodes.iter().any(|n| n == name), "{name} missing from {nodes:?}");
                }
                assert!(!nodes.iter().any(|n| n == "in"), "the Input node is not on the cycle");
            }
            other => panic!("expected a cycle refusal, got {other:?}"),
        }
    }

    #[test]
    fn a_topological_order_respects_every_edge() {
        let g = chain();
        let order = g.topological_order().expect("the chain is acyclic");
        assert_eq!(order.len(), g.nodes.len());
        let position = |name: &str| {
            order.iter().position(|&i| g.nodes[i].name == name).expect("every node is ordered")
        };
        for e in &g.edges {
            assert!(position(&e.from) < position(&e.to), "edge {} -> {} is out of order", e.from, e.to);
        }
    }

    #[test]
    fn validation_catches_a_ragged_parameter_and_a_non_finite_one() {
        let mut g = chain();
        let Node::Lif(x) = &mut g.nodes[1].node else { panic!("fixture changed") };
        x.r.pop();
        match g.validate(&Rules::default()) {
            Err(ValidationError::RaggedParameter { node, field, len, expected }) => {
                assert_eq!((node.as_str(), field, len, expected), ("a", "r", 1, 2));
            }
            other => panic!("expected a ragged parameter, got {other:?}"),
        }

        let mut h = chain();
        let Node::Lif(x) = &mut h.nodes[1].node else { panic!("fixture changed") };
        x.v_threshold[1] = f64::NAN;
        match h.validate(&Rules::default()) {
            Err(ValidationError::NonFiniteParameter { node, field, index }) => {
                assert_eq!((node.as_str(), field, index), ("a", "v_threshold", 1));
            }
            other => panic!("expected a non-finite refusal, got {other:?}"),
        }
    }

    #[test]
    fn a_duplicate_name_and_an_unwritable_name_are_refused() {
        let mut g = chain();
        g.push("a", Node::Lif(lif_params(1, 20e-3)));
        assert!(matches!(g.validate(&Rules::default()), Err(ValidationError::DuplicateNode { .. })));

        let mut h = Graph::new();
        h.nodes.push(Named { name: "two words".to_string(), node: Node::Input(Input { shape: vec![1] }) });
        assert!(matches!(h.validate(&Rules::default()), Err(ValidationError::BadNodeName { .. })));
    }

    #[test]
    fn a_tau_of_zero_is_refused_but_a_cuba_tau_syn_of_zero_is_not() {
        let mut g = Graph::new();
        g.push("bad", Node::Lif(lif_params(1, 0.0)));
        match g.validate(&Rules::default()) {
            Err(ValidationError::BadHyperparameter { node, field, .. }) => {
                assert_eq!((node.as_str(), field), ("bad", "tau"));
            }
            other => panic!("expected a hyperparameter refusal, got {other:?}"),
        }

        let mut h = Graph::new();
        h.push(
            "cu",
            Node::CubaLif(CubaLif {
                shape: vec![1],
                tau_syn: vec![0.0],
                tau_mem: vec![20e-3],
                r: vec![1.0],
                v_leak: vec![0.0],
                v_threshold: vec![1.0],
                v_reset: vec![0.0],
                w_in: vec![1.0],
            }),
        );
        h.validate(&Rules::default()).expect("tau_syn = 0 is the LIF limit, and it is legal");
    }

    // -- shapes ---------------------------------------------------------------------------------

    /// Against the published convolution arithmetic (Dumoulin & Visin, `arXiv`:1603.07285, 2016;
    /// the same expression `PyTorch` documents), with every case computed by hand from
    /// `floor((i + 2p - d(k-1) - 1)/s) + 1` rather than from a previous run of this code.
    #[test]
    fn conv2d_output_shapes_match_the_published_formula() {
        let base = Conv2d {
            in_channels: 1,
            out_channels: 4,
            size: [28, 28],
            kernel: [3, 3],
            stride: [1, 1],
            padding: [0, 0],
            dilation: [1, 1],
            groups: 1,
            weight: vec![0.0; 4 * 9],
            bias: Vec::new(),
        };
        // (28 + 0 - 1*2 - 1)/1 + 1 = 26
        assert_eq!(Node::Conv2d(base.clone()).output_shape(), Some(vec![4, 26, 26]));

        // "same" padding: (28 + 2 - 2 - 1)/1 + 1 = 28
        let same = Conv2d { padding: [1, 1], ..base.clone() };
        assert_eq!(Node::Conv2d(same).output_shape(), Some(vec![4, 28, 28]));

        // stride 2 with padding 1: (28 + 2 - 2 - 1)/2 + 1 = floor(27/2) + 1 = 13 + 1 = 14
        let strided = Conv2d { padding: [1, 1], stride: [2, 2], ..base.clone() };
        assert_eq!(Node::Conv2d(strided).output_shape(), Some(vec![4, 14, 14]));

        // dilation 2, no padding: (28 + 0 - 2*2 - 1)/1 + 1 = 24. The term the naive formula drops.
        let dilated = Conv2d { dilation: [2, 2], ..base.clone() };
        assert_eq!(Node::Conv2d(dilated).output_shape(), Some(vec![4, 24, 24]));

        // Asymmetric, all four knobs different per axis:
        // h: (28 + 4 - 3*(5-1) - 1)/3 + 1 = (28 + 4 - 12 - 1)/3 + 1 = floor(19/3) + 1 = 7
        // w: (28 + 0 - 1*(2-1) - 1)/4 + 1 = 26/4 + 1 = 6 + 1 = 7
        let mixed = Conv2d {
            kernel: [5, 2],
            stride: [3, 4],
            padding: [2, 0],
            dilation: [3, 1],
            weight: vec![0.0; 4 * 5 * 2],
            ..base.clone()
        };
        assert_eq!(Node::Conv2d(mixed).output_shape(), Some(vec![4, 7, 7]));

        // A window that cannot be placed gives a zero dimension, which `validate` then reports.
        let too_big = Conv2d { kernel: [31, 31], weight: vec![0.0; 4 * 31 * 31], ..base };
        assert_eq!(Node::Conv2d(too_big.clone()).output_shape(), Some(vec![4, 0, 0]));
        let mut g = Graph::new();
        g.push("c", Node::Conv2d(too_big));
        assert!(matches!(g.validate(&Rules::default()), Err(ValidationError::EmptyDimension { .. })));
    }

    #[test]
    fn conv1d_output_length_matches_the_published_formula() {
        let c = Conv1d {
            in_channels: 2,
            out_channels: 3,
            length: 8,
            kernel: 3,
            stride: 2,
            padding: 1,
            dilation: 1,
            groups: 1,
            weight: vec![0.0; 18],
            bias: Vec::new(),
        };
        // (8 + 2 - 1*2 - 1)/2 + 1 = floor(7/2) + 1 = 4
        assert_eq!(Node::Conv1d(c).output_shape(), Some(vec![3, 4]));
    }

    /// Sum and average pooling differ only by a constant, so their shapes must be identical —
    /// and the pooling formula is the convolution one with dilation fixed at 1.
    #[test]
    fn the_two_pools_have_the_same_shape_and_the_pooling_formula() {
        let p = Pool2d {
            channels: 3,
            size: [7, 7],
            kernel: [2, 2],
            stride: [2, 2],
            padding: [0, 0],
        };
        // (7 + 0 - 2)/2 + 1 = floor(5/2) + 1 = 3: the last column is dropped, as PyTorch drops it
        // without ceil_mode.
        assert_eq!(Node::SumPool2d(p.clone()).output_shape(), Some(vec![3, 3, 3]));
        assert_eq!(
            Node::SumPool2d(p.clone()).output_shape(),
            Node::AvgPool2d(p).output_shape(),
            "sum and average pooling differ by a constant, never by a shape"
        );
    }

    /// The invariant a reshape must satisfy: it moves elements, it does not create or destroy them.
    #[test]
    fn flatten_preserves_the_element_count() {
        let size = vec![2, 3, 4];
        for (start, end, want) in [
            (0usize, 0usize, vec![2, 3, 4]),
            (0, 1, vec![6, 4]),
            (1, 2, vec![2, 12]),
            (0, 2, vec![24]),
        ] {
            let n = Node::Flatten(Flatten { size: size.clone(), start_dim: start, end_dim: end });
            let out = n.output_shape().expect("Flatten has an output port");
            assert_eq!(out, want, "flatten {start}..={end}");
            assert_eq!(product(&out), product(&size), "flatten changed the element count");
        }
    }

    #[test]
    fn a_flatten_outside_its_own_shape_is_refused() {
        let mut g = Graph::new();
        g.push("f", Node::Flatten(Flatten { size: vec![2, 3], start_dim: 1, end_dim: 5 }));
        match g.validate(&Rules::default()) {
            Err(ValidationError::BadHyperparameter { field, .. }) => assert_eq!(field, "end_dim"),
            other => panic!("expected a hyperparameter refusal, got {other:?}"),
        }
    }

    /// `Affine` against the matrix product, computed by hand, and against the identity.
    #[test]
    fn affine_and_linear_apply_the_matrix_product() {
        let a = Affine {
            rows: 2,
            cols: 3,
            weight: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            bias: vec![0.5, -0.5],
        };
        // [1 2 3; 4 5 6] * [1, 0, -1] = [1 - 3, 4 - 6] = [-2, -2]; plus the bias.
        assert_eq!(a.apply(&[1.0, 0.0, -1.0]), Some(vec![-1.5, -2.5]));
        assert_eq!(a.apply(&[1.0, 0.0]), None, "a width mismatch has no answer");

        let identity = Linear { rows: 3, cols: 3, weight: vec![
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ] };
        let x = vec![core::f64::consts::PI, -0.25, 1e300];
        assert_eq!(identity.apply(&x), Some(x.clone()), "the identity matrix must be the identity");
    }

    // -- the bridge -----------------------------------------------------------------------------

    /// An `Affine` + `LIF` graph whose weights include a subnormal, a huge value and an exact zero.
    fn bridge_fixture() -> Graph {
        let mut g = Graph::new();
        g.push("in", Node::Input(Input { shape: vec![2] }));
        g.push("a", Node::Lif(lif_params(2, 20e-3)));
        g.push(
            "w",
            Node::Affine(Affine {
                rows: 3,
                cols: 2,
                weight: vec![5e-324, -1e300, 0.0, 0.1, core::f64::consts::PI, -2.5e-7],
                bias: vec![0.0, 0.0, 0.0],
            }),
        );
        g.push("b", Node::Lif(lif_params(3, 5e-3)));
        g.push("out", Node::Output(Output { shape: vec![3] }));
        g.edge("in", "a");
        g.edge("a", "w");
        g.edge("w", "b");
        g.edge("b", "out");
        g
    }

    /// **Requirement (c).** Weights cross the bridge bit for bit, checked by round-tripping back
    /// and comparing `to_bits`, not by comparing the code against itself.
    #[test]
    fn the_bridge_preserves_weights_bit_exactly() {
        let g = bridge_fixture();
        let c = g.to_net(1e-4).expect("the fixture is inside the supported subset");

        assert_eq!(c.net.n, 5, "two plus three neurons");
        assert_eq!(c.net.n_syn, 5, "six matrix entries, one of them an exact zero, so five synapses");
        assert_eq!(
            c.blocks,
            vec![
                Block { node: "a".to_string(), base: 0, len: 2 },
                Block { node: "b".to_string(), base: 2, len: 3 },
            ]
        );
        assert_eq!(c.inputs, vec![("in".to_string(), 0)]);
        assert_eq!(c.outputs, vec![("out".to_string(), 1)]);

        // Straight out of the network: weight[row][col] must be the synapse (2 + row) <- (0 + col).
        let want = [
            (0u32, 2u32, 5e-324),
            (1, 2, -1e300),
            (1, 3, 0.1),
            (0, 4, core::f64::consts::PI),
            (1, 4, -2.5e-7),
        ];
        for (pre, post, w) in want {
            let found = c
                .net
                .out_of(pre as usize)
                .find(|&(p, _, _)| p == post)
                .unwrap_or_else(|| panic!("synapse {pre} -> {post} is missing"));
            assert_eq!(found.1.to_bits(), w.to_bits(), "synapse {pre} -> {post} changed value");
        }
        assert!(
            c.net.out_of(0).all(|(p, _, _)| p != 3),
            "the exact-zero weight should have been dropped, not stored"
        );

        // And back: the dense matrix must hold the same bits in the same places.
        let h = Graph::from_net(&c.net, &c.neurons, 1e-4).expect("a delay-free network converts");
        let Node::Linear(dense) = &h.nodes[2].node else { panic!("expected a Linear node") };
        assert_eq!((dense.rows, dense.cols), (5, 5));
        for (pre, post, w) in want {
            let cell = dense.weight[post as usize * 5 + pre as usize];
            assert_eq!(cell.to_bits(), w.to_bits(), "dense[{post}][{pre}] changed value");
        }
        // The documented loss: a dropped zero and an absent connection are the same cell now.
        assert_eq!(dense.weight[3 * 5], 0.0);

        // Neuron parameters survive too, and the whole thing closes.
        let c2 = h.to_net(1e-4).expect("the rebuilt graph is in the subset");
        assert_eq!(c2.net, c.net, "Net -> NIR -> Net changed the network");
        assert_eq!(c2.neurons, c.neurons, "Net -> NIR -> Net changed a neuron");
    }

    /// **Requirement (e).** An unsupported node is named, not partially converted.
    #[test]
    fn the_bridge_refuses_an_unsupported_node_by_name() {
        let mut g = bridge_fixture();
        g.push(
            "cu",
            Node::CubaLif(CubaLif {
                shape: vec![3],
                tau_syn: vec![5e-3; 3],
                tau_mem: vec![20e-3; 3],
                r: vec![10e6; 3],
                v_leak: vec![-65e-3; 3],
                v_threshold: vec![-50e-3; 3],
                v_reset: vec![-65e-3; 3],
                w_in: vec![1.0; 3],
            }),
        );
        match g.to_net(1e-4) {
            Err(BridgeError::UnsupportedNode { name, kind }) => {
                assert_eq!(name, "cu");
                assert_eq!(kind, Kind::CubaLif);
                // The message has to carry the name, because that is what a user greps for.
                let text = BridgeError::UnsupportedNode { name, kind }.to_string();
                assert!(text.contains("cu") && text.contains("CubaLIF"), "{text}");
            }
            other => panic!("expected an unsupported-node refusal, got {other:?}"),
        }
    }

    #[test]
    fn the_bridge_refuses_a_non_zero_bias_rather_than_dropping_it() {
        let mut g = bridge_fixture();
        let Node::Affine(a) = &mut g.nodes[2].node else { panic!("fixture changed") };
        a.bias[1] = 1e-9;
        match g.to_net(1e-4) {
            Err(BridgeError::BiasNotRepresentable { name, index }) => {
                assert_eq!((name.as_str(), index), ("w", 1));
            }
            other => panic!("expected a bias refusal, got {other:?}"),
        }
        // Exactly zero is fine, and -0.0 counts as zero because it adds nothing.
        let Node::Affine(a) = &mut g.nodes[2].node else { panic!("fixture changed") };
        a.bias[1] = -0.0;
        g.to_net(1e-4).expect("a zero bias is representable");
    }

    #[test]
    fn a_delay_node_becomes_tick_delays_and_survives_the_round_trip() {
        let mut g = Graph::new();
        g.push("a", Node::Lif(lif_params(2, 20e-3)));
        g.push("w", Node::Linear(Linear { rows: 2, cols: 2, weight: vec![1e-3, 2e-3, 3e-3, 4e-3] }));
        g.push("d", Node::Delay(Delay { shape: vec![2], delay: vec![0.0, 2e-3] }));
        g.push("b", Node::Lif(lif_params(2, 20e-3)));
        g.edge("a", "w");
        g.edge("w", "d");
        g.edge("d", "b");
        let c = g.to_net(1e-4).expect("a delay of 2 ms is exactly twenty 0.1 ms ticks");
        assert_eq!(c.net.max_delay, 20);
        for pre in 0..2 {
            for (post, _, ticks) in c.net.out_of(pre) {
                let want = if post == 2 { 0 } else { 20 };
                assert_eq!(ticks, want, "synapse {pre} -> {post}");
            }
        }

        let h = Graph::from_net(&c.net, &c.neurons, 1e-4).expect("uniform per-neuron delays convert");
        // `from_net` flattens the two populations into one, so the delay node covers all four
        // neurons: the two source cells receive nothing, neuron 2 is undelayed and neuron 3 holds
        // the twenty ticks.
        let Node::Delay(d) = &h.nodes[3].node else { panic!("expected a Delay node") };
        assert_eq!(d.delay.len(), 4);
        assert_eq!(&d.delay[..3], &[0.0, 0.0, 0.0]);
        assert!((d.delay[3] - 2e-3).abs() < 1e-15, "delay came back as {}", d.delay[3]);
        let c2 = h.to_net(1e-4).expect("and converts back");
        assert_eq!(c2.net, c.net, "the delays did not survive Net -> NIR -> Net");
    }

    /// Refused rather than rounded: a rounded delay changes a coincidence window, and a
    /// coincidence detector that stops coinciding still runs.
    #[test]
    fn a_delay_that_is_not_a_whole_number_of_ticks_is_refused() {
        let mut g = Graph::new();
        g.push("a", Node::Lif(lif_params(1, 20e-3)));
        g.push("w", Node::Linear(Linear { rows: 1, cols: 1, weight: vec![1e-3] }));
        g.push("d", Node::Delay(Delay { shape: vec![1], delay: vec![1.5e-4] }));
        g.push("b", Node::Lif(lif_params(1, 20e-3)));
        g.edge("a", "w");
        g.edge("w", "d");
        g.edge("d", "b");
        match g.to_net(1e-4) {
            Err(BridgeError::DelayNotWholeTicks { name, seconds, dt }) => {
                assert_eq!(name, "d");
                assert!((seconds - 1.5e-4).abs() < 1e-18);
                assert!((dt - 1e-4).abs() < 1e-18);
            }
            other => panic!("expected a whole-ticks refusal, got {other:?}"),
        }
        g.to_net(5e-5).expect("at a 0.05 ms tick the same delay is exactly three ticks");
    }

    #[test]
    fn from_net_refuses_what_nir_cannot_say() {
        let g = bridge_fixture();
        let c = g.to_net(1e-4).expect("fixture");

        let mut refractory = c.neurons.clone();
        refractory[2].t_ref = 2e-3;
        match Graph::from_net(&c.net, &refractory, 1e-4) {
            Err(BridgeError::RefractoryNotRepresentable { neuron, t_ref }) => {
                assert_eq!(neuron, 2);
                assert!((t_ref - 2e-3).abs() < 1e-18);
            }
            other => panic!("expected a refractory refusal, got {other:?}"),
        }

        assert!(matches!(
            Graph::from_net(&c.net, &c.neurons[..4], 1e-4),
            Err(BridgeError::WrongNeuronCount { got: 4, want: 5 })
        ));
        assert!(matches!(
            Graph::from_net(&c.net, &c.neurons, 0.0),
            Err(BridgeError::BadTimeStep { .. })
        ));

        // Two synapses onto one neuron with different delays: NIR delays per element, not per
        // synapse, so this network has no NIR form and the refusal names the neuron.
        let mut b = crate::net::NetBuilder::new(2);
        b.connect(0, 1, 1e-3, 1).expect("in range");
        b.connect(1, 1, 1e-3, 3).expect("in range");
        let split = b.build();
        let two = vec![c.neurons[0], c.neurons[0]];
        match Graph::from_net(&split, &two, 1e-4) {
            Err(BridgeError::SplitDelay { post, a, b }) => {
                assert_eq!(post, 1);
                assert_eq!([a, b], [1, 3]);
            }
            other => panic!("expected a split-delay refusal, got {other:?}"),
        }
    }

    #[test]
    fn the_bridge_refuses_wiring_it_cannot_express() {
        // Two populations wired straight together: a synapse without a weight is not a synapse.
        let mut g = Graph::new();
        g.push("a", Node::Lif(lif_params(2, 20e-3)));
        g.push("b", Node::Lif(lif_params(2, 20e-3)));
        g.edge("a", "b");
        assert!(matches!(g.to_net(1e-4), Err(BridgeError::UnsupportedEdge { .. })));

        // A weight matrix feeding two places is two synapse blocks, and the graph should say so.
        let mut h = Graph::new();
        h.push("a", Node::Lif(lif_params(2, 20e-3)));
        h.push("w", Node::Linear(Linear { rows: 2, cols: 2, weight: vec![1e-3; 4] }));
        h.push("b", Node::Lif(lif_params(2, 20e-3)));
        h.push("c", Node::Lif(lif_params(2, 20e-3)));
        h.edge("a", "w");
        h.edge("w", "b");
        h.edge("w", "c");
        match h.to_net(1e-4) {
            Err(BridgeError::WeightFanout { name, incoming, outgoing }) => {
                assert_eq!((name.as_str(), incoming, outgoing), ("w", 1, 2));
            }
            other => panic!("expected a fan-out refusal, got {other:?}"),
        }

        // And a graph with no LIF at all has no neurons to build.
        let mut k = Graph::new();
        k.push("in", Node::Input(Input { shape: vec![1] }));
        assert!(matches!(k.to_net(1e-4), Err(BridgeError::NoNeurons)));
    }

    /// An invalid graph is refused by the bridge with the validator's own message, rather than a
    /// second vocabulary for the same mistake.
    #[test]
    fn the_bridge_refuses_an_invalid_graph_in_the_validators_words() {
        let mut g = bridge_fixture();
        g.edge("a", "nowhere");
        match g.to_net(1e-4) {
            Err(BridgeError::Invalid(ValidationError::DanglingEdge { missing, .. })) => {
                assert_eq!(missing, "nowhere");
            }
            other => panic!("expected a wrapped validation error, got {other:?}"),
        }
    }

    /// The point of the bridge: a `NIR` graph ends up running in this crate's simulator.
    #[test]
    fn a_converted_graph_runs_in_the_simulator() {
        let mut g = Graph::new();
        g.push("in", Node::Input(Input { shape: vec![1] }));
        g.push("a", Node::Lif(lif_params(1, 20e-3)));
        g.push("w", Node::Linear(Linear { rows: 1, cols: 1, weight: vec![20e-3] }));
        g.push("b", Node::Lif(lif_params(1, 20e-3)));
        g.edge("in", "a");
        g.edge("a", "w");
        g.edge("w", "b");
        let c = g.to_net(1e-4).expect("in the subset");

        let base = c.blocks[c.inputs[0].1].base;
        let mut ext = vec![0.0; c.net.n];
        ext[base] = 3e-9; // v_inf = -35 mV, comfortably over the -50 mV threshold
        let mut sim = crate::sim::Sim::new(c.net, c.neurons, 1e-4, crate::sim::Mode::Clocked)
            .expect("the neuron count matches by construction");
        let train = sim.run(2_000, &ext); // 200 ms

        let driven = train.of(u32::try_from(base).expect("small")).len();
        let downstream = train.of(u32::try_from(c.blocks[1].base).expect("small")).len();
        // NIR has no refractory period, so the closed-form rate is 1 / (tau * ln(30/15)) = 72.1 Hz,
        // which is 14 spikes in 200 ms. Checked against `Lif::isi` rather than against a constant.
        let expected = crate::neuron::Lif { t_ref: 0.0, ..crate::neuron::Lif::default() }
            .rate(3e-9)
            .expect("supra-threshold");
        let measured = driven as f64 / 0.2;
        assert!(
            (measured - expected).abs() / expected < 0.05,
            "driven cell fired at {measured} Hz against a closed form of {expected} Hz"
        );
        assert!(downstream > 0, "the 20 mV synapse delivered nothing");
    }

    // -- CubaLIF --------------------------------------------------------------------------------

    fn cuba_node(tau_syn: f64, tau_mem: f64, w_in: f64) -> CubaLif {
        CubaLif {
            shape: vec![1],
            tau_syn: vec![tau_syn],
            tau_mem: vec![tau_mem],
            r: vec![10e6],
            v_leak: vec![-65e-3],
            // Far out of reach, so the sub-threshold closed form is what is being compared.
            v_threshold: vec![1e3],
            v_reset: vec![-65e-3],
            w_in: vec![w_in],
        }
    }

    /// The exact propagator against the exact solution, stepped two thousand times.
    ///
    /// This passes at floating-point noise rather than at a discretisation tolerance BECAUSE the
    /// update is the analytic solution over the step. Forward Euler would fail it by four orders of
    /// magnitude at this `dt`.
    #[test]
    fn the_cuba_propagator_matches_its_closed_form() {
        for (tau_syn, tau_mem) in [(5e-3, 20e-3), (20e-3, 5e-3), (1e-3, 1e-3), (0.0, 20e-3)] {
            let node = cuba_node(tau_syn, tau_mem, 1.0);
            let u = 1e-9;
            let mut s = node.state(0).expect("one element");
            let dt = 1e-5;
            // `t = 0` FIRST, before any step. The loop used to start at k = 1, so the sample where
            // `exp(-t/tau_syn)` is `exp(-0/0)` was never taken and the oracle's `NaN` at
            // `tau_syn = 0` never surfaced — in the one function every other CubaLIF check is
            // measured against.
            let want0 = node.step_response(0, 0.0, u).expect("t = 0 is in range");
            assert!(want0.is_finite(), "the oracle is {want0} at t = 0 for tau_syn {tau_syn}");
            assert_eq!(
                want0.to_bits(),
                (-65e-3f64).to_bits(),
                "at t = 0 the step response is the rest it starts from, not {want0}"
            );
            assert!((s.v - want0).abs() < 1e-13, "tau_syn {tau_syn} step 0: {} vs {want0}", s.v);
            for k in 1..=2_000u32 {
                assert!(!s.step_exact(dt, u), "the threshold is 1 kV; nothing should fire");
                let t = f64::from(k) * dt;
                let want = node.step_response(0, t, u).expect("in range");
                assert!(
                    (s.v - want).abs() < 1e-13,
                    "tau_syn {tau_syn} step {k}: propagator {} vs closed form {want}",
                    s.v
                );
            }
        }
    }

    /// Two hand-computable values of the closed form, so the formula itself is pinned and not only
    /// its agreement with the integrator.
    #[test]
    fn the_closed_form_hits_its_hand_computed_values() {
        let r = 10e6;
        let u = 1e-9;
        let tau = 20e-3;

        // tau_syn = 0: the LIF response. At t = tau the bracket is 1 - 1/e.
        let lif_limit = cuba_node(0.0, tau, 1.0);
        let want = -65e-3 + r * u * (1.0 - core::f64::consts::E.recip());
        let got = lif_limit.step_response(0, tau, u).expect("in range");
        assert!((got - want).abs() < 1e-15, "{got} vs 1 - 1/e giving {want}");

        // tau_syn = tau_mem: the alpha function. At t = tau the bracket is 1 - 2/e.
        let alpha = cuba_node(tau, tau, 1.0);
        let want = -65e-3 + r * u * (1.0 - 2.0 * core::f64::consts::E.recip());
        let got = alpha.step_response(0, tau, u).expect("in range");
        assert!((got - want).abs() < 1e-15, "{got} vs 1 - 2/e giving {want}");
    }

    /// **Requirement (d), the sharp form.** At `tau_syn = 0` the `CubaLIF` propagator is not merely
    /// close to the `LIF`'s — it performs the same arithmetic in the same order, so the two
    /// trajectories agree **bit for bit**. Held sub-threshold, because `NIR` spikes on `>` and
    /// `Lif` on `>=`.
    #[test]
    fn cuba_at_zero_tau_syn_is_bit_for_bit_the_lif() {
        let mut cuba = CubaState {
            tau_syn: 0.0,
            tau_mem: 20e-3,
            r: 10e6,
            v_leak: -65e-3,
            v_threshold: -50e-3,
            v_reset: -65e-3,
            w_in: 1.0,
            i_syn: 0.0,
            v: -65e-3,
        };
        let mut lif = crate::neuron::Lif {
            tau_m: 20e-3,
            v_rest: -65e-3,
            v_th: -50e-3,
            v_reset: -65e-3,
            r_m: 10e6,
            t_ref: 0.0,
            v: -65e-3,
            refractory: 0.0,
        };
        // 1 nA puts v_inf at -55 mV, five millivolts below threshold: no spike, so the comparison
        // is over the integrator and not over two reset rules.
        for k in 0..1_000 {
            let a = cuba.step_exact(1e-4, 1e-9);
            let b = lif.step(1e-4, 1e-9);
            assert!(!a && !b, "step {k} fired; this trajectory is sub-threshold by construction");
            assert_eq!(cuba.v.to_bits(), lif.v.to_bits(), "step {k}: {} vs {}", cuba.v, lif.v);
        }
        assert!(cuba.v > -56e-3 && cuba.v < -54.9e-3, "the trajectory went nowhere: {}", cuba.v);
    }

    /// The reduction as a parameter substitution: the gain folds into the resistance, exactly.
    #[test]
    fn reduce_to_lif_folds_the_gain_into_the_resistance() {
        let node = CubaLif {
            shape: vec![2],
            tau_syn: vec![0.0, 0.0],
            tau_mem: vec![20e-3, 5e-3],
            r: vec![1e6, 2e6],
            v_leak: vec![-65e-3, -0.0],
            v_threshold: vec![-50e-3, 1.0],
            v_reset: vec![-65e-3, 0.0],
            w_in: vec![2.5, 4.0],
        };
        let reduced = node.reduce_to_lif().expect("tau_syn is zero at every element");
        assert_eq!(reduced.shape, vec![2]);
        assert_eq!(reduced.tau, node.tau_mem);
        assert_eq!(reduced.r[0].to_bits(), 2.5e6f64.to_bits());
        assert_eq!(reduced.r[1].to_bits(), 8e6f64.to_bits());
        assert_eq!(reduced.v_leak, node.v_leak);

        // And the reduced node's trajectory is the two-variable model's, to rounding — not bit for
        // bit this time, because `r * (w_in * u)` and `(r * w_in) * u` round differently.
        let u = 1e-9;
        let mut s = node.state(0).expect("element 0");
        let mut lif = crate::neuron::Lif {
            tau_m: reduced.tau[0],
            v_rest: reduced.v_leak[0],
            v_th: 1e3,
            v_reset: reduced.v_reset[0],
            r_m: reduced.r[0],
            t_ref: 0.0,
            v: reduced.v_leak[0],
            refractory: 0.0,
        };
        for k in 0..500 {
            s.step_exact(1e-4, u);
            lif.step(1e-4, u);
            let rel = (s.v - lif.v).abs() / lif.v.abs();
            assert!(rel < 1e-15, "step {k}: {} vs {} (relative {rel})", s.v, lif.v);
        }
    }

    #[test]
    fn reduce_to_lif_refuses_a_synapse_that_actually_filters() {
        let node = cuba_node(1e-12, 20e-3, 1.0);
        assert!(
            node.reduce_to_lif().is_none(),
            "a picosecond synapse is still a filter, and dropping it is a modelling decision"
        );
        assert!(cuba_node(0.0, 20e-3, 1.0).reduce_to_lif().is_some());
    }

    /// The reduction is a **limit**, so the error must go to zero at first order: halve `tau_syn`
    /// and the gap to the `LIF` response halves. A test that only asserted "the gap is small" would
    /// pass for a model that converged to the wrong thing.
    #[test]
    fn the_cuba_response_converges_to_the_lif_response_at_first_order() {
        let tau_mem = 20e-3;
        let t = tau_mem;
        let u = 1e-9;
        let target = cuba_node(0.0, tau_mem, 1.0).step_response(0, t, u).expect("the LIF limit");

        let mut errs = Vec::new();
        let mut tau_syn = 4e-3;
        for _ in 0..5 {
            let v = cuba_node(tau_syn, tau_mem, 1.0).step_response(0, t, u).expect("in range");
            errs.push((v - target).abs());
            tau_syn /= 2.0;
        }
        for w in errs.windows(2) {
            let ratio = w[0] / w[1];
            assert!(
                (1.9..=2.3).contains(&ratio),
                "halving tau_syn changed the error by {ratio}x, not ~2x: {errs:?}"
            );
            assert!(w[1] < w[0], "the error did not shrink: {errs:?}");
        }
        // And it really is converging to the LIF and not merely settling: the last error is small.
        assert!(errs[4] < 0.02 * (target + 65e-3).abs(), "{errs:?}");
    }

    /// The `tau_syn == tau_mem` branch exists because the general expression divides by zero there.
    /// It has to agree with the general expression arbitrarily close to that point.
    #[test]
    fn the_alpha_branch_is_continuous_with_the_general_one() {
        let tau = 20e-3;
        let u = 1e-9;
        for t in [1e-3, 5e-3, 20e-3, 100e-3] {
            let exact = cuba_node(tau, tau, 1.0).step_response(0, t, u).expect("in range");
            let nearby = cuba_node(tau * (1.0 + 1e-9), tau, 1.0)
                .step_response(0, t, u)
                .expect("in range");
            let rel = (exact - nearby).abs() / exact.abs();
            assert!(rel < 1e-6, "at t = {t} the branches disagree by {rel} relative");
        }
    }

    /// The threshold and reset still work, and `NIR`'s strict `>` is what is implemented.
    #[test]
    fn a_cuba_element_spikes_and_resets_on_a_strict_crossing() {
        let node = CubaLif {
            shape: vec![1],
            tau_syn: vec![1e-3],
            tau_mem: vec![10e-3],
            r: vec![10e6],
            v_leak: vec![0.0],
            v_threshold: vec![10e-3],
            v_reset: vec![-1e-3],
            w_in: vec![1.0],
        };
        let mut s = node.state(0).expect("one element");
        let mut spikes = 0u32;
        for _ in 0..10_000 {
            if s.step_exact(1e-5, 5e-9) {
                spikes += 1;
                assert!((s.v - (-1e-3)).abs() < 1e-18, "reset went to {}", s.v);
            }
            assert!(s.v.is_finite() && s.i_syn.is_finite());
        }
        assert!(spikes > 5, "a 50 mV steady state against a 10 mV threshold gave {spikes} spikes");

        // Exactly at the threshold is NOT a spike: NIR writes `v > v_threshold`.
        let mut edge = node.state(0).expect("one element");
        edge.v = 10e-3;
        assert!(!edge.step_exact(0.0, 0.0), "a zero step is a no-op, threshold or not");
        edge.v = 10e-3;
        edge.tau_syn = 0.0;
        edge.r = 0.0;
        edge.v_leak = 10e-3;
        assert!(!edge.step_exact(1e-9, 0.0), "sitting exactly on the threshold is not a crossing");
    }

    /// A non-finite step or input is a no-op rather than a `NaN` in the membrane.
    #[test]
    fn a_non_finite_step_leaves_the_state_alone() {
        let node = cuba_node(5e-3, 20e-3, 1.0);
        let start = node.state(0).expect("one element");
        for (dt, u) in [(f64::NAN, 1e-9), (0.0, 1e-9), (-1e-4, 1e-9), (1e-4, f64::NAN), (f64::INFINITY, 1e-9)] {
            let mut s = start;
            assert!(!s.step_exact(dt, u));
            assert_eq!(s, start, "dt {dt}, u {u} moved the state");
        }
    }

    /// The trait implementation is the cross-fabric path: a `CubaLIF` element is a
    /// [`crate::neuron::Neuron`], so it can be put in a [`crate::sim::Sim`].
    #[test]
    fn a_cuba_element_is_a_neuron() {
        let node = cuba_node(5e-3, 20e-3, 1.0);
        let mut s = node.state(0).expect("one element");
        assert!((s.potential() - (-65e-3)).abs() < 1e-18);
        s.bump(5e-3);
        assert!((s.potential() - (-60e-3)).abs() < 1e-15, "bump displaces the membrane");
        s.bump_current(2e-9);
        assert!((s.i_syn - 2e-9).abs() < 1e-18, "bump_current displaces the synapse");
        s.reset();
        assert_eq!(s.i_syn, 0.0);
        assert!((s.potential() - (-65e-3)).abs() < 1e-18);
        assert_eq!(s.refractory_left(), 0.0, "NIR has no refractory period");

        // The enforcement point for EXACT_OVER_GAPS. This test USED to assert that the simulator
        // ACCEPTED a CubaState in event-driven mode, which pinned the wrong constant in place as a
        // regression test. It is refused, for the reason
        // `the_cuba_hybrid_does_not_compose_across_a_quiet_gap` measures.
        let net = crate::net::NetBuilder::new(1).build();
        assert!(matches!(
            crate::sim::Sim::new(net.clone(), vec![s], 1e-4, crate::sim::Mode::EventDriven),
            Err(crate::sim::SimError::NotExactOverGaps)
        ));
        crate::sim::Sim::new(net.clone(), vec![s], 1e-4, crate::sim::Mode::Clocked)
            .expect("clocked must still work; the model is fine, the jump is not");
        assert!(matches!(
            crate::sim::Sim::new(
                net,
                vec![crate::neuron::Izhikevich::regular_spiking()],
                1e-4,
                crate::sim::Mode::EventDriven
            ),
            Err(crate::sim::SimError::NotExactOverGaps)
        ));
    }

    /// The **linear flow** composes: a matrix exponential composes, so ten steps of `dt` and one
    /// step of `10 dt` over a quiet interval land in the same place.
    ///
    /// ⚠ Read what this fixture is before reading this as a licence to jump gaps.
    /// [`cuba_node`] sets `v_threshold` to one kilovolt, so the threshold logic is structurally
    /// out of reach and what is measured here is the propagator alone. That is worth measuring and
    /// it is **not** [`crate::neuron::Neuron::EXACT_OVER_GAPS`], which is a claim about the flow
    /// *and* the threshold together. For years this test carried that claim; the hybrid half is
    /// `the_cuba_hybrid_does_not_compose_across_a_quiet_gap`, and it fails.
    ///
    /// To rounding, not bit for bit — `exp(-a)*exp(-b)` and `exp(-(a+b))` differ in the last place,
    /// which is the same caveat [`crate::neuron::Lif`]'s scalar exponential carries.
    #[test]
    fn the_cuba_flow_composes_across_a_quiet_gap() {
        let node = cuba_node(5e-3, 20e-3, 1.0);
        let mut fine = node.state(0).expect("one element");
        fine.i_syn = 3e-9;
        fine.v = -60e-3;
        let mut coarse = fine;
        for _ in 0..10 {
            assert!(!fine.step_exact(1e-4, 0.0));
        }
        assert!(!coarse.step_exact(1e-3, 0.0));
        assert!(
            (fine.v - coarse.v).abs() / coarse.v.abs() < 1e-14,
            "ten steps gave {} and one gap gave {}",
            fine.v,
            coarse.v
        );
        assert!(
            (fine.i_syn - coarse.i_syn).abs() / coarse.i_syn.abs() < 1e-14,
            "the synaptic current disagreed: {} vs {}",
            fine.i_syn,
            coarse.i_syn
        );
        // The premise: the state actually moved, so this is not two copies of the initial value.
        assert!((coarse.i_syn - 3e-9).abs() > 1e-10, "nothing decayed");
    }
    // -- the repairs the audit asked for, each with the measurement behind it --------------------

    /// **The hybrid system does NOT compose across a quiet gap**, which is why
    /// [`CubaState`]'s [`crate::neuron::Neuron::EXACT_OVER_GAPS`] is `false`.
    ///
    /// `the_cuba_flow_composes_across_a_quiet_gap` measures the propagator with the threshold held
    /// structurally out of reach at one kilovolt. Put the threshold back where a `CubaLIF` actually
    /// has it and the two runs disagree about a whole spike — because under zero input this
    /// membrane is a difference of two exponentials and keeps climbing after the input stops, and
    /// the jumped step reads only the endpoint.
    #[test]
    fn the_cuba_hybrid_does_not_compose_across_a_quiet_gap() {
        let node = CubaLif {
            shape: vec![1],
            tau_syn: vec![5e-3],
            tau_mem: vec![20e-3],
            r: vec![10e6],
            v_leak: vec![-65e-3],
            v_threshold: vec![-50e-3], // reachable, unlike `cuba_node`'s kilovolt
            v_reset: vec![-65e-3],
            w_in: vec![1.0],
        };
        let dt = 1e-4;
        let steps = 600u32; // 60 ms, every tick of it quiet

        let mut fine = node.state(0).expect("one element");
        fine.bump_current(12e-9);
        let mut fine_spikes = 0u32;
        let mut first = None;
        for k in 0..steps {
            if fine.step_exact(dt, 0.0) {
                fine_spikes += 1;
                first.get_or_insert(k);
            }
        }

        let mut jumped = node.state(0).expect("one element");
        jumped.bump_current(12e-9);
        let jumped_spiked = jumped.step_exact(dt * f64::from(steps), 0.0);

        // The premise, asserted rather than assumed: the fine run really does cross, INSIDE the
        // interval that has no input in it at all.
        assert_eq!(fine_spikes, 1, "the fine run must spike or this test proves nothing");
        let at = first.expect("one spike means one first spike");
        assert!(
            (20..80).contains(&at),
            "the crossing is a few ms into the quiet interval, measured at tick {at}"
        );

        // And the jump does not delay that spike, it deletes it.
        assert!(!jumped_spiked, "one step over the whole gap saw no crossing");
        assert!(
            (fine.v - jumped.v).abs() > 5e-4,
            "the two runs landed at {} V and {} V; if those agreed, EXACT_OVER_GAPS could be true",
            fine.v,
            jumped.v
        );
        // A compile-time assertion, as `sim.rs` makes for `Izhikevich`: the constant is the whole
        // safety property, and a spike this model produces inside a quiet gap would be discarded
        // rather than delayed if it were `true`.
        const {
            assert!(!CubaState::EXACT_OVER_GAPS);
        }

        // End to end. The clocked run is the model; the event-driven run a `true` constant would
        // have permitted is refused at construction instead of being produced and believed.
        let net = crate::net::NetBuilder::new(1).build();
        let cell = node.state(0).expect("one element");
        let mut sim = crate::sim::Sim::new(net.clone(), vec![cell], dt, crate::sim::Mode::Clocked)
            .expect("clocked is always legal");
        let mut ext = vec![60e-9];
        let mut clocked = 0usize;
        for _ in 0..30 {
            clocked += sim.step(&ext).len(); // 3 ms of drive
        }
        ext[0] = 0.0;
        for _ in 0..570 {
            clocked += sim.step(&ext).len(); // 57 ms of silence
        }
        assert!(clocked > 1, "the clocked run fired {clocked} times; the jumped one fires once");
        assert!(matches!(
            crate::sim::Sim::new(net, vec![cell], dt, crate::sim::Mode::EventDriven),
            Err(crate::sim::SimError::NotExactOverGaps)
        ));
    }

    /// A shape can name more elements than a `usize` can count, and four tokens is all it takes.
    ///
    /// Before the repair the plain product **wrapped to zero** in a release build — so a node
    /// declaring `2^64` elements validated, because `check_array` expects zero entries for a zero
    /// count and empty arrays supply them — and **panicked** in a debug build, inside the boundary
    /// function whose job is to reject malformed nodes, on input straight from
    /// [`Graph::from_text`].
    #[test]
    fn a_shape_that_overflows_a_usize_is_refused_rather_than_wrapped() {
        let big = usize::MAX / 2 + 1;
        let text = format!("{MAGIC}\nnode s Scale shape=[{big},2] scale=[]\n");
        let g = Graph::from_text(&text).expect("the reader takes it; the validator is the boundary");
        assert_eq!(
            g.validate(&Rules::default()),
            Err(ValidationError::BadHyperparameter {
                node: "s".to_string(),
                field: "shape",
                why: "names more elements than a usize can count",
            })
        );

        // The same for a node with dynamics, whose parameter arrays are the ones a wrapped zero
        // would have certified as correctly empty.
        let mut h = Graph::new();
        h.push(
            "n",
            Node::Lif(Lif {
                shape: vec![big, 2],
                tau: vec![],
                r: vec![],
                v_leak: vec![],
                v_threshold: vec![],
                v_reset: vec![],
            }),
        );
        assert!(matches!(
            h.validate(&Rules::default()),
            Err(ValidationError::BadHyperparameter { field: "shape", .. })
        ));

        // And the two products themselves: one reports, one saturates, neither wraps.
        assert_eq!(checked_product(&[big, 2]), None);
        assert_eq!(checked_product(&[3, 4, 5]), Some(60));
        assert_eq!(product(&[big, 2]), usize::MAX, "the saturating form must not wrap to zero");
        assert_eq!(product(&[3, 4, 5]), 60);
    }

    /// A weight count is a product too, and `rows * cols` is public arithmetic on public fields.
    #[test]
    fn an_overflowing_weight_count_is_refused_rather_than_wrapped() {
        let big = usize::MAX / 2 + 1;

        let lin = Linear { rows: big, cols: 2, weight: vec![] };
        let mut g = Graph::new();
        g.push("w", Node::Linear(lin.clone()));
        assert_eq!(
            g.validate(&Rules::default()),
            Err(ValidationError::BadHyperparameter {
                node: "w".to_string(),
                field: "rows/cols",
                why: "name more cells than a usize can count",
            })
        );
        // `apply` is `#[must_use]` on public fields and must answer rather than panic.
        assert_eq!(lin.apply(&[1.0, 2.0]), None);
        assert_eq!(
            Affine { rows: big, cols: 2, weight: vec![], bias: vec![] }.apply(&[1.0, 2.0]),
            None
        );

        // out_channels * (in_channels / groups) * kernel, overflowing on any pointer width.
        let mut h = Graph::new();
        h.push(
            "c1",
            Node::Conv1d(Conv1d {
                in_channels: 1 << 20,
                out_channels: 1 << 20,
                length: 16,
                kernel: 1 << 24,
                stride: 1,
                padding: 0,
                dilation: 1,
                groups: 1,
                weight: vec![],
                bias: vec![],
            }),
        );
        assert!(matches!(
            h.validate(&Rules::default()),
            Err(ValidationError::BadHyperparameter { field: "weight", .. })
        ));
    }

    /// [`conv_dim`]'s doc promises a shape query never panics on a malformed node. It did:
    /// `2 * padding` overflowed before `saturating_add` ever saw the result.
    #[test]
    fn a_shape_query_never_panics_on_a_malformed_node() {
        let c = Node::Conv2d(Conv2d {
            in_channels: 1,
            out_channels: 1,
            size: [8, 8],
            kernel: [3, 3],
            stride: [1, 1],
            padding: [usize::MAX / 2 + 1, 0],
            dilation: [1, 1],
            groups: 1,
            weight: vec![0.0; 9],
            bias: vec![],
        });
        let shape = c.output_shape().expect("a Conv2d has an output port");
        assert_eq!(shape[0], 1, "the channel axis is untouched");
        assert_eq!(shape[2], 6, "the well-formed axis still follows the published arithmetic");
        assert_eq!(shape[1], usize::MAX - 2, "the malformed axis saturates rather than wrapping");

        // And the validator names it instead of overflowing on it.
        let mut g = Graph::new();
        g.push("c", c);
        assert!(matches!(
            g.validate(&Rules::default()),
            Err(ValidationError::BadHyperparameter { .. })
        ));

        // A Flatten whose collapsed run overflows answers too.
        let f = Node::Flatten(Flatten {
            size: vec![usize::MAX / 2 + 1, 2, 3],
            start_dim: 0,
            end_dim: 1,
        });
        assert_eq!(f.output_shape().expect("a Flatten has an output port"), vec![usize::MAX, 3]);
    }

    /// A delay in seconds is cast to a tick count, and a Rust float-to-int cast **saturates**.
    ///
    /// The `Delay` node's own doc says a delay is refused rather than rounded, because a rounded
    /// delay changes a coincidence window. A delay of `1e30` s silently became `4294967295` ticks:
    /// the same failure, four orders of magnitude past the most extreme form the doc imagined, and
    /// then `crate::sim::Sim` asks for that many delivery buckets.
    #[test]
    fn a_delay_past_a_u32_is_refused_rather_than_saturated() {
        for seconds in [1e9f64, 1e30] {
            let mut g = Graph::new();
            g.push("a", Node::Lif(lif_params(1, 20e-3)));
            g.push("w", Node::Linear(Linear { rows: 1, cols: 1, weight: vec![1e-3] }));
            g.push("d", Node::Delay(Delay { shape: vec![1], delay: vec![seconds] }));
            g.push("b", Node::Lif(lif_params(1, 20e-3)));
            g.edge("a", "w");
            g.edge("w", "d");
            g.edge("d", "b");
            match g.to_net(1e-4) {
                Err(BridgeError::DelayTooManyTicks { name, ticks, .. }) => {
                    assert_eq!(name, "d");
                    assert!(
                        ticks > f64::from(u32::MAX) * 1e3,
                        "the error must carry the TRUE count, not the clamped one: {ticks}"
                    );
                }
                other => panic!("a delay of {seconds} s gave {other:?}"),
            }
        }

        // The boundary from both sides, at a `dt` of one second so the division is exact.
        let mut g = Graph::new();
        g.push("a", Node::Lif(lif_params(1, 20e-3)));
        g.push("w", Node::Linear(Linear { rows: 1, cols: 1, weight: vec![1e-3] }));
        g.push("d", Node::Delay(Delay { shape: vec![1], delay: vec![f64::from(u32::MAX)] }));
        g.push("b", Node::Lif(lif_params(1, 20e-3)));
        g.edge("a", "w");
        g.edge("w", "d");
        g.edge("d", "b");
        let c = g.to_net(1.0).expect("u32::MAX ticks is exactly representable");
        assert_eq!(c.net.max_delay, u32::MAX, "the last legal tick count survives unclamped");

        let Node::Delay(d) = &mut g.nodes[2].node else { panic!("fixture changed") };
        d.delay[0] = f64::from(u32::MAX) + 1.0;
        assert!(matches!(g.to_net(1.0), Err(BridgeError::DelayTooManyTicks { .. })));
    }

    /// **Requirement (e), on a graph that can actually occur.** The disconnected fixture in
    /// `the_bridge_refuses_an_unsupported_node_by_name` was the only shape that reached the
    /// by-name refusal: the edge sweep ran first, so every unsupported node WIRED to anything came
    /// back as an edge error with its kind nowhere in the message.
    #[test]
    fn the_bridge_names_a_wired_unsupported_node_by_its_kind() {
        let mut g = Graph::new();
        g.push("in", Node::Input(Input { shape: vec![1] }));
        g.push("a", Node::Lif(lif_params(1, 20e-3)));
        g.push("w", Node::Linear(Linear { rows: 1, cols: 1, weight: vec![1e-3] }));
        g.push(
            "cu",
            Node::CubaLif(CubaLif {
                shape: vec![1],
                tau_syn: vec![5e-3],
                tau_mem: vec![20e-3],
                r: vec![10e6],
                v_leak: vec![-65e-3],
                v_threshold: vec![-50e-3],
                v_reset: vec![-65e-3],
                w_in: vec![1.0],
            }),
        );
        g.edge("in", "a");
        g.edge("a", "w");
        g.edge("w", "cu");
        // The premise: this is a perfectly legal NIR graph, not a broken one.
        g.validate(&Rules::default()).expect("a weighted LIF -> CubaLIF chain is legal NIR");
        match g.to_net(1e-4) {
            Err(BridgeError::UnsupportedNode { name, kind }) => {
                assert_eq!((name.as_str(), kind), ("cu", Kind::CubaLif));
            }
            other => panic!("expected the node named with its kind, got {other:?}"),
        }
    }

    /// A `tau_syn` the graph validator refuses must not integrate, because the struct's fields are
    /// public and [`CubaLif::state`] is not the only way to reach one.
    ///
    /// At `tau_syn = -5 ms` the synaptic factor `exp(-dt/tau_syn)` exceeds one and the current
    /// grows every step, with no error, no `NaN` and nothing in the result to notice. And the
    /// value the validator *accepts* — a **negative zero**, since `-0.0 < 0.0` is false — used to
    /// put a `NaN` straight into the membrane.
    #[test]
    fn a_tau_syn_outside_its_range_cannot_grow_a_synapse_or_poison_a_membrane() {
        let base = CubaState {
            tau_syn: 0.0,
            tau_mem: 20e-3,
            r: 10e6,
            v_leak: -65e-3,
            v_threshold: 1e3,
            v_reset: -65e-3,
            w_in: 1.0,
            i_syn: 5e-9,
            v: -65e-3,
        };
        for tau_syn in [-5e-3, f64::NEG_INFINITY, f64::INFINITY, f64::NAN] {
            let start = CubaState { tau_syn, ..base };
            let mut s = start;
            for k in 0..5 {
                assert!(!s.step_exact(1e-4, 0.0), "tau_syn {tau_syn} step {k} claimed a spike");
            }
            assert_eq!(s.i_syn.to_bits(), start.i_syn.to_bits(), "tau_syn {tau_syn} moved i_syn");
            assert_eq!(s.v.to_bits(), start.v.to_bits(), "tau_syn {tau_syn} moved v");
        }

        // A negative zero IS the instantaneous-synapse limit and must behave as one, bit for bit.
        let mut plus = CubaState { tau_syn: 0.0, ..base };
        let mut minus = CubaState { tau_syn: -0.0, ..base };
        for k in 0..50 {
            assert_eq!(plus.step_exact(1e-4, 1e-9), minus.step_exact(1e-4, 1e-9));
            assert!(minus.v.is_finite(), "step {k}: a negative zero put {} in the membrane", minus.v);
            assert_eq!(plus.v.to_bits(), minus.v.to_bits(), "step {k}: {} vs {}", plus.v, minus.v);
            assert_eq!(plus.i_syn.to_bits(), minus.i_syn.to_bits());
        }
        assert!(minus.v > -65e-3, "the trajectory went nowhere: {}", minus.v);
    }

    /// The documented loss on the comparison itself, pinned so that it stays documented.
    ///
    /// `NIR` fires on `v > v_threshold`; [`crate::neuron::Lif`] fires on `v >= v_th`. A neuron with
    /// `r = 0` and `v_leak = v_threshold` rests exactly on its threshold: under `NIR` it never
    /// fires, and the converted network fires on every tick. Measure zero in floating point, and
    /// not zero — this module's own `a_cuba_element_spikes_and_resets_on_a_strict_crossing` builds
    /// the same fixed point to pin `>` on the `NIR` side.
    #[test]
    fn the_bridge_turns_a_strict_crossing_into_a_non_strict_one() {
        let mut g = Graph::new();
        g.push("in", Node::Input(Input { shape: vec![1] }));
        g.push(
            "n",
            Node::Lif(Lif {
                shape: vec![1],
                tau: vec![20e-3],
                r: vec![0.0],
                v_leak: vec![-50e-3],
                v_threshold: vec![-50e-3],
                v_reset: vec![-65e-3],
            }),
        );
        g.push("out", Node::Output(Output { shape: vec![1] }));
        g.edge("in", "n");
        g.edge("n", "out");
        let c = g.to_net(1e-4).expect("a plain LIF is inside the subset");

        // `NIR`'s rule, on this module's own `NIR`-side integrator: never a crossing.
        let mut nir_side = CubaState {
            tau_syn: 0.0,
            tau_mem: 20e-3,
            r: 0.0,
            v_leak: -50e-3,
            v_threshold: -50e-3,
            v_reset: -65e-3,
            w_in: 1.0,
            i_syn: 0.0,
            v: -50e-3,
        };
        for k in 0..100 {
            assert!(!nir_side.step_exact(1e-4, 0.0), "step {k}: NIR fires on `>`, strictly");
        }
        assert_eq!(nir_side.v.to_bits(), (-50e-3f64).to_bits(), "it really did sit on the line");

        // The converted network's rule: fires on the first tick, and the doc says so.
        let mut sim = crate::sim::Sim::new(c.net, c.neurons, 1e-4, crate::sim::Mode::Clocked)
            .expect("the neuron count matches by construction");
        assert_eq!(sim.step(&[]), vec![0], "the `>` to `>=` loss the bridge table now discloses");
    }

    /// [`fmt_f64`]'s fallback had never executed — `{:?}` round-trips by construction, so the
    /// check that guards it always passes. Exercise the fallback directly, on the same awkward
    /// floats, so that "17 significant digits" being eighteen is a fact about tested code.
    #[test]
    fn the_float_codecs_long_form_round_trips_bit_for_bit() {
        for x in AWKWARD {
            let long = fmt_f64_long(x);
            assert_eq!(
                long.parse::<f64>().map(f64::to_bits),
                Ok(x.to_bits()),
                "the fallback lost {x} as {long}"
            );
            // One digit before the point and seventeen after: eighteen significant digits.
            let mantissa = long.split('e').next().expect("an exponent form");
            let digits = mantissa.chars().filter(char::is_ascii_digit).count();
            assert_eq!(digits, 18, "{long} has {digits} significant digits, not 18");
        }
        // And the two forms agree on value even where they disagree on spelling.
        for x in AWKWARD {
            assert_eq!(
                fmt_f64(x).parse::<f64>().map(f64::to_bits),
                fmt_f64_long(x).parse::<f64>().map(f64::to_bits)
            );
        }
    }

    /// [`Graph::from_net`] returns a **cyclic** graph whatever the network was, so no consumer that
    /// asked for [`Rules::feedforward`] can take its output. The doc now says so; this pins it.
    #[test]
    fn a_rebuilt_graph_is_cyclic_and_no_feedforward_consumer_takes_it() {
        let mut b = crate::net::NetBuilder::new(2);
        b.connect(0, 1, 1e-3, 0).expect("in range");
        let net = b.build();
        let cells = vec![crate::neuron::Lif { t_ref: 0.0, ..crate::neuron::Lif::default() }; 2];
        let g = Graph::from_net(&net, &cells, 1e-4).expect("a delay-free network converts");

        g.validate(&Rules::default()).expect("NIR allows cycles and so does the default");
        assert!(!g.is_acyclic(), "the collapsed `neurons -> weights -> neurons` IS a cycle");
        match g.validate(&Rules::feedforward()) {
            Err(ValidationError::Cycle { nodes }) => {
                assert_eq!(nodes, vec!["neurons", "weights", "output"]);
            }
            other => panic!("expected a cycle refusal, got {other:?}"),
        }
    }
}
