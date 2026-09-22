//! Spiking convolutional networks: the architecture almost all deployed spiking vision runs.
//!
//! # What a spiking convolution is, and what it is not
//!
//! A convolutional layer is not an arithmetic operation. It is a **weight-sharing pattern**: one
//! small kernel, `k_h * k_w * C_in` numbers, is applied at every spatial position of the input, so
//! a layer that would need `C_out * H * W * C_in * H * W` weights as a dense matrix needs
//! `C_out * C_in * k_h * k_w`. Nothing about that pattern is specific to real-valued activations,
//! which is why it transfers to spiking networks unchanged and why every spiking vision result
//! anyone has published on `CIFAR-10`, `ImageNet` or the `DVS` gesture sets is convolutional.
//!
//! What changes in the spiking case is the other half. A conventional convolution multiplies each
//! input by a weight; a spiking one is handed a **binary** input, so the multiply degenerates into
//! a conditional add, which is the whole energy argument (see [`crate::metrics::ActivationKind`]
//! and [`crate::ledger`]). And the layer acquires **state**: one membrane potential per output
//! position per output channel, carried from one timestep to the next. That state is the cost
//! nobody puts in the parameter table. A `64`-channel layer on a `128x128` map holds
//! `64 * 128 * 128 = 1_048_576` membrane potentials — at `f32`, 4 MiB of state for a layer whose
//! *weights*, at `3x3` from 64 channels, are `36_864` numbers, 144 KiB. **The state is 28.4x the
//! weights**, and a [`crate::neuron::Lif`] carries a refractory counter beside the potential, so
//! the deployed figure is twice that again. [`SpikingConv2d::footprint`] reports both, because a
//! figure that reports only the parameters is describing a different chip than the one the model
//! has to run on — and `the_membrane_state_dominates_the_parameter_count` recomputes the three
//! numbers in this paragraph rather than leaving them as prose.
//!
//! # What this module calls "convolution" is cross-correlation
//!
//! As in every deep-learning framework this review checked, and unlike the signal-processing
//! definition: the kernel is **not** flipped. `out[y][x] = sum_{ky,kx} w[ky][kx] * in[y+ky][x+kx]`.
//! For a learned kernel the distinction is a relabelling and nobody flips; for a *transcribed*
//! kernel — an edge detector from a textbook, a Gabor filter from a paper — it is the difference
//! between the published figure and its mirror image. Stated here because
//! `the_convolution_is_cross_correlation_and_the_kernel_is_not_flipped` is a test rather than a
//! remark.
//!
//! # Units
//!
//! Weights are in **volts per input spike**: a synapse delivers an instantaneous membrane
//! displacement through [`crate::neuron::Neuron::bump`], which is the delta-synapse convention the
//! rest of this crate uses and the only one under which halving the time step does not halve every
//! synaptic influence. Biases are volts per timestep. [`SpikingConv2d::dt`] is seconds.
//!
//! [`TdBn`] is the exception, and it is exactly the exception [`crate::neuron::Izhikevich`] is: the
//! paper's constants are dimensionless because they are ratios against the firing threshold, and
//! they are kept in that frame inside the model with the threshold `v_th` supplied in volts at the
//! boundary, so a reader can compare the expressions line by line against Zheng et al. 2021.
//!
//! # The four mechanisms here, and what each costs
//!
//! **Convolution and pooling** ([`Conv2d`], [`avg_pool`], [`SpikingMaxPool`]). Pooling over spikes
//! is where the literature quietly disagrees with itself; [`MaxPolicy`] states the disagreement
//! rather than picking a side silently.
//!
//! **tdBN** ([`TdBn`], Zheng, Wu, Deng, Yan & Li, *Going Deeper with Directly-Trained Larger
//! Spiking Neural Networks*, `AAAI` 2021). Ordinary batch normalisation targets unit variance.
//! A spiking layer does not care about unit variance, it cares about variance **relative to the
//! firing threshold**: too small and the population never fires, too large and every unit
//! saturates and the surrogate gradient vanishes on both sides. tdBN targets a standard deviation
//! of `alpha * V_th` and normalises over the time axis as well as the batch. It is the single
//! change that took directly-trained spiking networks from roughly ten layers to fifty.
//!
//! **Residual connections** ([`ResidualBlock`], [`ResidualStyle`]; Fang, Yu, Chen, Huang, Masquelier
//! & Tian, *Deep Residual Learning in Spiking Neural Networks*, `NeurIPS` 2021). See
//! [`ResidualStyle`] for what `SEW` changes and why. The short version: the naive spiking residual
//! **cannot represent the identity**, for any choice of shortcut gain, because the neuron it passes
//! the sum through has a reset and a refractory period. This module demonstrates that rather than
//! asserting it — `the_naive_residual_cannot_be_the_identity_even_with_a_tuned_gain` runs both.
//!
//! **Initialisation** ([`Init`]). Fan-in scaling, with the fan-in stated as
//! `C_in * k_h * k_w` and the threshold-aware variant derived in [`Init::ThresholdScaled`] from a
//! second moment rather than transcribed.
//!
//! # What is verified, and against what
//!
//! Every convolution result in the tests is a **hand-computed integer**, compared at exactly zero
//! tolerance — the arithmetic of a small-integer convolution has no error term, so a tolerance
//! would only be hiding something. The shape formula is checked against an independent *count* of
//! valid window positions rather than against itself. Average pooling of a dyadic constant is
//! exactly that constant. A zero-weight `SEW`-ADD block is bit-exactly the identity. The synaptic
//! operation counts are hand-calculated for a `3x3` input, and the loop's tap count is
//! cross-checked against a separable closed form across a sweep.
//!
//! # What this module does not do
//!
//! It does not train. [`crate::surrogate`] carries the gradient machinery for a recurrent `LIF`
//! layer and this module exposes the two places a convolutional backward pass would touch —
//! [`SpikingConv2d::surrogate_mask`] and [`identity_path_gain`] — but there is no convolutional
//! backward pass here, and this review did not locate one that could be checked against finite
//! differences without a good deal more machinery than one module should carry.

use crate::ledger::Ledger;
use crate::metrics::{ActivationKind, Footprint, MetricError, SynOps};
use crate::neuron::Neuron;
use crate::rng::Rng;
use crate::surrogate::Surrogate;

/// Everything a layer in this module refuses to do, with the numbers that made it refuse.
///
/// Every variant names the offending quantity. A shape error that says only "bad shape" costs the
/// caller the debugging session that the two numbers would have ended.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConvError {
    /// A dimension, count or window that was zero where a positive number is required.
    Zero {
        /// Which quantity: `"kernel_h"`, `"stride_w"`, `"in_channels"`, and so on.
        what: &'static str,
    },
    /// A parameter, weight or activation that was `NaN` or infinite.
    ///
    /// Rejected at the boundary because one non-finite weight poisons every membrane potential
    /// downstream of it and the resulting spike train is silent about where it came from.
    NonFinite {
        /// Which quantity was non-finite.
        what: &'static str,
        /// Flat index into the offending array, or `0` for a scalar parameter.
        index: usize,
        /// The offending value itself.
        value: f64,
    },
    /// An array whose length did not match the shape it was declared with.
    BadShape {
        /// Which array.
        what: &'static str,
        /// The length supplied.
        got: usize,
        /// The length the declared shape requires.
        want: usize,
    },
    /// An activation that was neither exactly `0.0` nor exactly `1.0` where a binary spike is
    /// required.
    ///
    /// `SEW`-ADD blocks emit values above one on purpose ([`ResidualStyle::SewAdd`]), so this is a
    /// real condition a real pipeline hits rather than a defensive check — and the honest response
    /// is to say that the accumulate-only energy argument no longer applies, not to round.
    NotBinary {
        /// Flat index of the offending activation.
        index: usize,
        /// The offending value.
        value: f64,
    },
    /// A kernel whose dilated extent does not fit inside the padded input, so the standard output
    /// formula yields nothing.
    ///
    /// Refused rather than clamped to zero or to one: a layer that silently produced a `0x0` map
    /// would propagate an empty tensor through the rest of the network and fail somewhere else.
    ImpossibleShape {
        /// `"height"` or `"width"`.
        axis: &'static str,
        /// Input extent along that axis, elements.
        n: usize,
        /// Kernel extent along that axis, taps.
        kernel: usize,
        /// Padding added to **each** side along that axis, elements.
        pad: usize,
        /// Stride along that axis, elements.
        stride: usize,
        /// Dilation along that axis: the spacing between taps, `1` meaning adjacent.
        dilation: usize,
    },
    /// Two tensors or two layers whose shapes must agree and did not.
    ShapeMismatch {
        /// What was being matched, e.g. `"residual shortcut"`.
        what: &'static str,
        /// The first shape, as `(channels, height, width)`.
        a: (usize, usize, usize),
        /// The second shape, as `(channels, height, width)`.
        b: (usize, usize, usize),
    },
    /// A pooling window that contained no input element at all — every tap fell on padding.
    ///
    /// There is no maximum of an empty set and no mean of zero elements. Returning `0.0` would be
    /// an answer, and a wrong one: it would be indistinguishable from a window that really did see
    /// only silence.
    EmptyWindow {
        /// Output row whose window was empty.
        y: usize,
        /// Output column whose window was empty.
        x: usize,
    },
    /// A count that exceeded `u64`.
    Overflow {
        /// Which count.
        what: &'static str,
    },
}

impl core::fmt::Display for ConvError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Zero { what } => write!(f, "{what} must be at least 1, got 0"),
            Self::NonFinite { what, index, value } => {
                write!(f, "{what}[{index}] is not finite: {value}")
            }
            Self::BadShape { what, got, want } => {
                write!(f, "{what} has length {got}, shape requires {want}")
            }
            Self::NotBinary { index, value } => {
                write!(f, "activation[{index}] = {value} is not a binary spike (0.0 or 1.0)")
            }
            Self::ImpossibleShape { axis, n, kernel, pad, stride, dilation } => write!(
                f,
                "{axis}: dilated kernel extent {} exceeds padded input {} \
                 (n={n}, k={kernel}, p={pad}, s={stride}, d={dilation})",
                dilation * kernel.saturating_sub(1) + 1,
                n + 2 * pad
            ),
            Self::ShapeMismatch { what, a, b } => {
                write!(f, "{what}: {a:?} does not match {b:?}")
            }
            Self::EmptyWindow { y, x } => {
                write!(f, "pooling window at ({y}, {x}) contains no input element")
            }
            Self::Overflow { what } => write!(f, "{what} exceeded u64"),
        }
    }
}

impl std::error::Error for ConvError {}

impl From<MetricError> for ConvError {
    /// Only the overflow path can occur: this module validates shapes, finiteness and binarity
    /// itself before handing anything to [`crate::metrics`], so the other variants are unreachable
    /// from here and are folded into an overflow that names where it came from rather than being
    /// silently discarded.
    fn from(e: MetricError) -> Self {
        match e {
            MetricError::Overflow { what } => Self::Overflow { what },
            _ => Self::Overflow { what: "synaptic operation accounting" },
        }
    }
}

/// A dense channels-height-width tensor, the shape a convolution consumes and produces.
///
/// One timestep. A spiking run is a sequence of these, and the layer rather than the tensor holds
/// the state that connects them — which is what lets a caller stream frames from
/// [`crate::aer`] through a layer without ever materialising the whole recording.
///
/// Layout is row-major with channel slowest: element `(c, y, x)` is at
/// `c * height * width + y * width + x`. Stated because a transposed tensor of the same length is
/// still a valid tensor and produces a plausible wrong answer.
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor3 {
    /// Channels. At least one.
    pub channels: usize,
    /// Rows. At least one.
    pub height: usize,
    /// Columns. At least one.
    pub width: usize,
    /// `channels * height * width` values, in the layout above. Volts for a membrane or a
    /// pre-activation, dimensionless `0.0`/`1.0` for a spike map.
    pub data: Vec<f64>,
}

impl Tensor3 {
    /// Wrap `data` in a shape, checking the length and that every value is finite.
    ///
    /// # Errors
    ///
    /// [`ConvError::Zero`] for a zero dimension, [`ConvError::BadShape`] when the length does not
    /// match, [`ConvError::NonFinite`] for a `NaN` or infinite element.
    pub fn new(channels: usize, height: usize, width: usize, data: Vec<f64>) -> Result<Self, ConvError> {
        let want = dims_len(channels, height, width)?;
        if data.len() != want {
            return Err(ConvError::BadShape { what: "tensor data", got: data.len(), want });
        }
        check_finite(&data, "tensor element")?;
        Ok(Self { channels, height, width, data })
    }

    /// An all-zero tensor of the given shape.
    ///
    /// # Errors
    ///
    /// [`ConvError::Zero`] for a zero dimension, [`ConvError::Overflow`] if the element count does
    /// not fit in a `usize`.
    pub fn zeros(channels: usize, height: usize, width: usize) -> Result<Self, ConvError> {
        let n = dims_len(channels, height, width)?;
        Ok(Self { channels, height, width, data: vec![0.0; n] })
    }

    /// A tensor with every element set to `value`.
    ///
    /// # Errors
    ///
    /// As [`Tensor3::zeros`], plus [`ConvError::NonFinite`] for a non-finite `value`.
    pub fn filled(channels: usize, height: usize, width: usize, value: f64) -> Result<Self, ConvError> {
        if !value.is_finite() {
            return Err(ConvError::NonFinite { what: "fill value", index: 0, value });
        }
        let n = dims_len(channels, height, width)?;
        Ok(Self { channels, height, width, data: vec![value; n] })
    }

    /// Flat index of element `(c, y, x)`, without bounds checking.
    ///
    /// Use this rather than the arithmetic: the layout is documented on [`Tensor3`] and this is the
    /// single place it is written down.
    #[must_use]
    pub fn index(&self, c: usize, y: usize, x: usize) -> usize {
        (c * self.height + y) * self.width + x
    }

    /// The value at `(c, y, x)`, or `None` if any coordinate is out of range.
    #[must_use]
    pub fn at(&self, c: usize, y: usize, x: usize) -> Option<f64> {
        if c >= self.channels || y >= self.height || x >= self.width {
            return None;
        }
        self.data.get(self.index(c, y, x)).copied()
    }

    /// Shape as `(channels, height, width)`, for comparison and error messages.
    #[must_use]
    pub fn shape(&self) -> (usize, usize, usize) {
        (self.channels, self.height, self.width)
    }

    /// `true` when every element is exactly `0.0` or exactly `1.0`.
    ///
    /// Exact comparison, deliberately: an activation of `0.999` is not a spike that needs rounding,
    /// it is evidence that something upstream emitted a graded value, and the accumulate-only
    /// energy argument does not survive it.
    #[must_use]
    pub fn is_binary(&self) -> bool {
        self.data.iter().all(|v| *v == 0.0 || *v == 1.0)
    }

    /// How many elements are non-zero — the spike count for a spike map.
    #[must_use]
    pub fn count_nonzero(&self) -> usize {
        self.data.iter().filter(|v| **v != 0.0).count()
    }

    /// Check that every element is exactly `0.0` or exactly `1.0`.
    ///
    /// # Errors
    ///
    /// [`ConvError::NotBinary`], naming the flat index and the value.
    pub fn require_binary(&self) -> Result<(), ConvError> {
        for (i, v) in self.data.iter().enumerate() {
            if *v != 0.0 && *v != 1.0 {
                return Err(ConvError::NotBinary { index: i, value: *v });
            }
        }
        Ok(())
    }
}

fn dims_len(channels: usize, height: usize, width: usize) -> Result<usize, ConvError> {
    for (what, n) in [("channels", channels), ("height", height), ("width", width)] {
        if n == 0 {
            return Err(ConvError::Zero { what });
        }
    }
    channels
        .checked_mul(height)
        .and_then(|a| a.checked_mul(width))
        .ok_or(ConvError::Overflow { what: "tensor element count" })
}

fn check_finite(values: &[f64], what: &'static str) -> Result<(), ConvError> {
    for (i, v) in values.iter().enumerate() {
        if !v.is_finite() {
            return Err(ConvError::NonFinite { what, index: i, value: *v });
        }
    }
    Ok(())
}

/// The standard convolution output length: `(n + 2p - d(k - 1) - 1) / s + 1`, floored.
///
/// `None` — never a truncated or clamped answer — when `k`, `s` or `d` is zero, or when the
/// **dilated extent** `d(k - 1) + 1` exceeds the padded input `n + 2p`, which is the case where no
/// window fits at all.
///
/// The formula counts the window start positions `0, s, 2s, ...` whose last tap
/// `start + d(k - 1)` is still inside `[0, n + 2p)`. That reading is what
/// `the_shape_formula_agrees_with_a_count_of_valid_window_positions` checks it against — a count,
/// not the formula written twice.
#[must_use]
pub fn out_dim(n: usize, kernel: usize, pad: usize, stride: usize, dilation: usize) -> Option<usize> {
    if kernel == 0 || stride == 0 || dilation == 0 {
        return None;
    }
    let padded = n.checked_add(pad.checked_mul(2)?)?;
    let extent = dilation.checked_mul(kernel - 1)?.checked_add(1)?;
    if extent > padded {
        return None;
    }
    Some((padded - extent) / stride + 1)
}

/// The shape of a convolution: kernel, stride, padding and dilation, per axis.
///
/// Per axis rather than a single number for both, because a spiking audio-spectrogram front end is
/// routinely `k_h != k_w`, and a spec that forced them equal would push that asymmetry into the
/// caller's indexing where it cannot be checked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Conv2dSpec {
    /// Input channels. At least one.
    pub in_channels: usize,
    /// Output channels, one membrane per output channel per position. At least one.
    pub out_channels: usize,
    /// Kernel taps along the row axis. At least one.
    pub kernel_h: usize,
    /// Kernel taps along the column axis. At least one.
    pub kernel_w: usize,
    /// Row stride, elements. At least one.
    pub stride_h: usize,
    /// Column stride, elements. At least one.
    pub stride_w: usize,
    /// Zero padding added to **each** side of the row axis, elements. The total added is `2 * p`.
    pub pad_h: usize,
    /// Zero padding added to **each** side of the column axis, elements.
    pub pad_w: usize,
    /// Row dilation: the spacing between taps, `1` meaning adjacent taps.
    pub dilation_h: usize,
    /// Column dilation.
    pub dilation_w: usize,
}

impl Conv2dSpec {
    /// A spec with stride 1, dilation 1 and no padding, validated.
    ///
    /// # Errors
    ///
    /// [`ConvError::Zero`] naming the first zero field.
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_h: usize,
        kernel_w: usize,
    ) -> Result<Self, ConvError> {
        let s = Self {
            in_channels,
            out_channels,
            kernel_h,
            kernel_w,
            stride_h: 1,
            stride_w: 1,
            pad_h: 0,
            pad_w: 0,
            dilation_h: 1,
            dilation_w: 1,
        };
        s.validate()?;
        Ok(s)
    }

    /// A shape-preserving spec: stride 1, dilation 1, and padding `(k - 1) / 2` on each side.
    ///
    /// Exactly shape-preserving only for an **odd** kernel. For an even kernel, `(k - 1) / 2`
    /// floors and the output is one element short on each axis; this constructor returns
    /// [`ConvError::Zero`] with `what = "odd kernel_h"` rather than returning a spec whose name
    /// lies about what it does. Residual blocks need this property exactly
    /// ([`ResidualBlock::new`] refuses a branch that changes shape), which is why it is a named
    /// constructor rather than a comment at the call site.
    ///
    /// # Errors
    ///
    /// [`ConvError::Zero`] for a zero or even kernel extent, or a zero channel count.
    pub fn same_padding(
        in_channels: usize,
        out_channels: usize,
        kernel_h: usize,
        kernel_w: usize,
    ) -> Result<Self, ConvError> {
        if kernel_h == 0 || kernel_h.is_multiple_of(2) {
            return Err(ConvError::Zero { what: "odd kernel_h" });
        }
        if kernel_w == 0 || kernel_w.is_multiple_of(2) {
            return Err(ConvError::Zero { what: "odd kernel_w" });
        }
        let mut s = Self::new(in_channels, out_channels, kernel_h, kernel_w)?;
        s.pad_h = (kernel_h - 1) / 2;
        s.pad_w = (kernel_w - 1) / 2;
        Ok(s)
    }

    /// Check that no field is zero where a positive number is required.
    ///
    /// # Errors
    ///
    /// [`ConvError::Zero`] naming the first offending field, in declaration order.
    pub fn validate(&self) -> Result<(), ConvError> {
        for (what, n) in [
            ("in_channels", self.in_channels),
            ("out_channels", self.out_channels),
            ("kernel_h", self.kernel_h),
            ("kernel_w", self.kernel_w),
            ("stride_h", self.stride_h),
            ("stride_w", self.stride_w),
            ("dilation_h", self.dilation_h),
            ("dilation_w", self.dilation_w),
        ] {
            if n == 0 {
                return Err(ConvError::Zero { what });
            }
        }
        Ok(())
    }

    /// Output `(height, width)` for an input of `(in_h, in_w)`, or `None` if no window fits.
    #[must_use]
    pub fn out_shape(&self, in_h: usize, in_w: usize) -> Option<(usize, usize)> {
        Some((
            out_dim(in_h, self.kernel_h, self.pad_h, self.stride_h, self.dilation_h)?,
            out_dim(in_w, self.kernel_w, self.pad_w, self.stride_w, self.dilation_w)?,
        ))
    }

    /// Same as [`Conv2dSpec::out_shape`] but naming the axis that failed.
    ///
    /// # Errors
    ///
    /// [`ConvError::ImpossibleShape`] carrying every parameter of the failing axis.
    pub fn require_out_shape(&self, in_h: usize, in_w: usize) -> Result<(usize, usize), ConvError> {
        self.validate()?;
        let oh = out_dim(in_h, self.kernel_h, self.pad_h, self.stride_h, self.dilation_h).ok_or(
            ConvError::ImpossibleShape {
                axis: "height",
                n: in_h,
                kernel: self.kernel_h,
                pad: self.pad_h,
                stride: self.stride_h,
                dilation: self.dilation_h,
            },
        )?;
        let ow = out_dim(in_w, self.kernel_w, self.pad_w, self.stride_w, self.dilation_w).ok_or(
            ConvError::ImpossibleShape {
                axis: "width",
                n: in_w,
                kernel: self.kernel_w,
                pad: self.pad_w,
                stride: self.stride_w,
                dilation: self.dilation_w,
            },
        )?;
        Ok((oh, ow))
    }

    /// Connections into one output unit: `in_channels * kernel_h * kernel_w`.
    ///
    /// **This is the fan-in every initialiser in [`Init`] scales by**, and it counts padded taps —
    /// a unit at the border has fewer real inputs than this, which is a second-order effect the
    /// initialisation literature ignores and this doc declines to pretend it handles.
    #[must_use]
    pub fn fan_in(&self) -> usize {
        self.in_channels.saturating_mul(self.kernel_h).saturating_mul(self.kernel_w)
    }

    /// Number of kernel weights: `out_channels * fan_in()`.
    #[must_use]
    pub fn n_weights(&self) -> usize {
        self.out_channels.saturating_mul(self.fan_in())
    }
}

/// Weight initialisation for a spiking convolution, each variant a variance target.
///
/// All three draw **uniformly**, because [`crate::rng::Rng`] provides a uniform exactly and a
/// Gaussian would need a transform whose tails are a second thing to verify. A uniform on
/// `[-a, a]` has variance `a^2 / 3`, so the half-width is `a = sigma * sqrt(3)` — that conversion
/// is [`Init::std_dev`] feeding [`Conv2d::init`] and it is where a factor of `sqrt(3)` would hide
/// if it were wrong.
///
/// Biases are initialised to zero in every case.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Init {
    /// `sigma^2 = 2 / fan_in`, the rectifier-aware scaling of He, Zhang, Ren & Sun, *Delving Deep
    /// into Rectifiers*, `ICCV` 2015. Half-width `sqrt(6 / fan_in)`.
    ///
    /// The factor 2 is there because a `ReLU` zeroes half the distribution and so halves the
    /// variance passed forward. **A spiking neuron is not a `ReLU`**: what it passes forward is a
    /// binary spike whose second moment is its firing rate, not a rectified copy of its input. So
    /// this is the right *form* and an unexamined constant for a spiking layer, and it is here
    /// because it is what every framework defaults to and therefore what most published spiking
    /// networks were actually initialised with.
    KaimingUniform,
    /// `sigma^2 = 1 / fan_in`, `LeCun` scaling: preserves the variance of a **linear** layer.
    ///
    /// Half-width `sqrt(3 / fan_in)`. The conservative choice, and the one whose derivation makes
    /// no assumption about the nonlinearity at all.
    LecunUniform,
    /// `sigma = v_th / sqrt(fan_in * rate)`: the pre-activation's standard deviation is one
    /// threshold when the inputs are `Bernoulli(rate)` spikes.
    ///
    /// **Derived here, not transcribed.** Let `z = sum_i w_i s_i` with `w_i` independent, zero-mean,
    /// variance `sigma^2`, and `s_i` independent `Bernoulli(r)` spikes independent of the weights.
    /// Then `E[z] = 0` and
    ///
    /// ```text
    /// E[z^2] = sum_i E[w_i^2] E[s_i^2] = fan_in * sigma^2 * r        (s_i^2 == s_i for a binary spike)
    /// ```
    ///
    /// so setting `E[z^2] = v_th^2` gives `sigma = v_th / sqrt(fan_in * r)`. That is the whole
    /// derivation, and `the_threshold_scaled_init_hits_its_second_moment` measures both sides.
    ///
    /// The spirit — scale the initialisation against the threshold rather than against unity — is
    /// the initialisation half of the argument Zheng et al. 2021 make for [`TdBn`] and that Rathi &
    /// Roy make in *`DIET-SNN`* (`IEEE TNNLS`, 2021) for learning the threshold instead. This exact
    /// expression is an elementary second moment rather than any one paper's formula, and it is
    /// stated that way so nobody cites it to a source that does not contain it.
    ThresholdScaled {
        /// Firing threshold measured **from the resting potential**, volts. For
        /// [`crate::neuron::Lif::default`] that is `v_th - v_rest = 15 mV`, not the bare `v_th` of
        /// `-50 mV` — the absolute potential is not a scale, and using it would size every weight
        /// by the arbitrary offset of the voltage origin.
        v_th: f64,
        /// Expected input firing rate, spikes per unit per timestep, in `(0, 1]`.
        ///
        /// This is the parameter nobody knows before training. `0.1`-`0.3` is the range the sparse
        /// spiking literature reports for hidden layers; a value that is too low sizes the weights
        /// too large and the layer saturates on its first batch.
        rate: f64,
    },
    /// An explicit standard deviation, volts, ignoring fan-in.
    ///
    /// For reproducing a published configuration that states its own number, and for tests.
    Fixed {
        /// Standard deviation of the weight draw, volts per spike. Finite and non-negative; zero is
        /// allowed and gives a silent layer, which [`ResidualBlock`]'s identity test needs.
        std_dev: f64,
    },
}

impl Init {
    /// The target standard deviation for this fan-in, volts per spike.
    ///
    /// `None` for a zero fan-in, a non-finite or negative `v_th` or `std_dev`, or a `rate` outside
    /// `(0, 1]` — each of which would otherwise produce an infinity or a negative width and a layer
    /// full of `NaN` some hundreds of timesteps later.
    #[must_use]
    pub fn std_dev(&self, fan_in: usize) -> Option<f64> {
        if fan_in == 0 {
            return None;
        }
        let n = fan_in as f64;
        match *self {
            Self::KaimingUniform => Some((2.0 / n).sqrt()),
            Self::LecunUniform => Some((1.0 / n).sqrt()),
            Self::ThresholdScaled { v_th, rate } => {
                if !v_th.is_finite() || v_th < 0.0 || !rate.is_finite() || rate <= 0.0 || rate > 1.0
                {
                    return None;
                }
                Some(v_th / (n * rate).sqrt())
            }
            Self::Fixed { std_dev } => {
                if !std_dev.is_finite() || std_dev < 0.0 {
                    return None;
                }
                Some(std_dev)
            }
        }
    }
}

/// A convolution kernel and its biases. No state, no neurons — just the weight-sharing pattern.
///
/// Weight layout is `[out_channel][in_channel][ky][kx]`, row-major, which is
/// [`Conv2d::weight_index`]. That is the layout `PyTorch` uses and the one a reader transcribing a
/// published kernel will assume; a transposed kernel of the same length is still a valid kernel and
/// gives a plausible wrong answer, which is why
/// `the_convolution_is_cross_correlation_and_the_kernel_is_not_flipped` uses an asymmetric kernel
/// whose transpose gives a different number.
#[derive(Debug, Clone, PartialEq)]
pub struct Conv2d {
    /// Shape of the convolution.
    pub spec: Conv2dSpec,
    /// `spec.n_weights()` kernel weights, volts per input spike, in the layout above.
    pub weights: Vec<f64>,
    /// `spec.out_channels` biases, volts per timestep, added to every position of their channel.
    pub bias: Vec<f64>,
}

impl Conv2d {
    /// Wrap weights and biases in a spec, checking lengths and finiteness.
    ///
    /// # Errors
    ///
    /// [`ConvError::Zero`] from [`Conv2dSpec::validate`], [`ConvError::BadShape`] for a wrong
    /// length, [`ConvError::NonFinite`] for a `NaN` or infinite weight or bias.
    pub fn new(spec: Conv2dSpec, weights: Vec<f64>, bias: Vec<f64>) -> Result<Self, ConvError> {
        spec.validate()?;
        let want_w = spec.n_weights();
        if weights.len() != want_w {
            return Err(ConvError::BadShape {
                what: "conv weights",
                got: weights.len(),
                want: want_w,
            });
        }
        if bias.len() != spec.out_channels {
            return Err(ConvError::BadShape {
                what: "conv bias",
                got: bias.len(),
                want: spec.out_channels,
            });
        }
        check_finite(&weights, "conv weight")?;
        check_finite(&bias, "conv bias")?;
        Ok(Self { spec, weights, bias })
    }

    /// An all-zero kernel: the layer that passes nothing.
    ///
    /// Not a degenerate case to be tolerated but the one [`ResidualBlock`]'s identity argument
    /// needs — a residual branch that emits no spikes is what "identity mapping" means in
    /// Fang et al. 2021, and a block that is not exactly the identity under it is not a residual
    /// block in their sense.
    ///
    /// # Errors
    ///
    /// [`ConvError::Zero`] from [`Conv2dSpec::validate`].
    pub fn zeros(spec: Conv2dSpec) -> Result<Self, ConvError> {
        spec.validate()?;
        Ok(Self { spec, weights: vec![0.0; spec.n_weights()], bias: vec![0.0; spec.out_channels] })
    }

    /// Draw weights from `init`, biases zero, deterministically from `seed`.
    ///
    /// Draw order is flat over [`Conv2d::weight_index`], so the same seed gives the same kernel on
    /// every platform and a change of layout would change the weights — which is a property worth
    /// having, because it makes a layout change visible in a regression rather than silent.
    ///
    /// # Errors
    ///
    /// [`ConvError::Zero`] from [`Conv2dSpec::validate`], and [`ConvError::NonFinite`] with
    /// `what = "init std_dev"` when [`Init::std_dev`] refuses — a non-finite `v_th`, a `rate`
    /// outside `(0, 1]`, or a negative explicit width.
    pub fn init(spec: Conv2dSpec, init: Init, seed: u64) -> Result<Self, ConvError> {
        spec.validate()?;
        let sigma = init.std_dev(spec.fan_in()).ok_or(ConvError::NonFinite {
            what: "init std_dev",
            index: 0,
            value: f64::NAN,
        })?;
        let half_width = sigma * 3.0_f64.sqrt();
        let mut rng = Rng::new(seed);
        let weights =
            (0..spec.n_weights()).map(|_| (2.0 * rng.next_f64() - 1.0) * half_width).collect();
        Ok(Self { spec, weights, bias: vec![0.0; spec.out_channels] })
    }

    /// Flat index of weight `[oc][ic][ky][kx]`.
    ///
    /// No bounds checking; the layout is documented on [`Conv2d`] and this is the only place it is
    /// written down.
    #[must_use]
    pub fn weight_index(&self, oc: usize, ic: usize, ky: usize, kx: usize) -> usize {
        (((oc * self.spec.in_channels + ic) * self.spec.kernel_h) + ky) * self.spec.kernel_w + kx
    }

    /// Learned scalars in this layer: weights plus biases.
    #[must_use]
    pub fn n_params(&self) -> usize {
        self.weights.len() + self.bias.len()
    }

    /// The textbook tap count, `k_h * k_w * C_in * C_out * H_out * W_out`, **including taps that
    /// land on padding**.
    ///
    /// This is the figure the `FLOP`-counting literature reports, and it is not the number of
    /// synapses: a tap on padding multiplies a weight by a structural zero. For a `3x3` kernel with
    /// padding 1 on a `3x3` input it is `81` against `49` real connections — a 65% overstatement,
    /// which shrinks as the map grows and never reaches zero. Both numbers are exposed so a report
    /// can say which one it used. [`Conv2d::synops`] uses [`Conv2d::in_bounds_taps`].
    ///
    /// `None` on overflow or when no window fits.
    #[must_use]
    pub fn padded_taps(&self, in_h: usize, in_w: usize) -> Option<u64> {
        let (oh, ow) = self.spec.out_shape(in_h, in_w)?;
        let per = (self.spec.fan_in() as u64).checked_mul(self.spec.out_channels as u64)?;
        per.checked_mul(oh as u64)?.checked_mul(ow as u64)
    }

    /// Real connections exercised per timestep: taps whose input coordinate is inside the array.
    ///
    /// Computed separably — the in-bounds tap count factorises as
    /// `(sum over output rows of in-bounds ky) * (sum over output cols of in-bounds kx) * C_in *
    /// C_out`, because the row and column conditions are independent. That factorisation is a
    /// second, independent route to the same number as the convolution loop's own count, and
    /// `the_separable_tap_count_agrees_with_the_loop` checks them against each other across a
    /// sweep of shapes.
    ///
    /// `None` on overflow, when no window fits, or when the count would take more than
    /// [`MAX_TAP_SWEEP`] iterations on either axis — the sweep is `outputs × taps` per axis, and
    /// nothing bounds `pad`, so a `pad_h` of `5e7` on a `1x1` input used to spend 2.6 s here with
    /// no allocation to fail first.
    #[must_use]
    pub fn in_bounds_taps(&self, in_h: usize, in_w: usize) -> Option<u64> {
        let (oh, ow) = self.spec.out_shape(in_h, in_w)?;
        let rows = axis_in_bounds(oh, in_h, self.spec.kernel_h, self.spec.pad_h, self.spec.stride_h, self.spec.dilation_h)?;
        let cols = axis_in_bounds(ow, in_w, self.spec.kernel_w, self.spec.pad_w, self.spec.stride_w, self.spec.dilation_w)?;
        rows.checked_mul(cols)?
            .checked_mul(self.spec.in_channels as u64)?
            .checked_mul(self.spec.out_channels as u64)
    }

    /// The convolution, plus the counts the same pass can produce for free.
    ///
    /// Returns `(output, in_bounds_taps, effective_taps)` where an effective tap is one whose input
    /// value is non-zero **and** whose weight is non-zero — `NeuroBench`'s conjunction, the reason
    /// activation sparsity and connection sparsity compound instead of adding.
    fn convolve(&self, input: &Tensor3) -> Result<(Tensor3, u64, u64), ConvError> {
        if input.channels != self.spec.in_channels {
            return Err(ConvError::ShapeMismatch {
                what: "conv input channels",
                a: input.shape(),
                b: (self.spec.in_channels, input.height, input.width),
            });
        }
        let want = dims_len(input.channels, input.height, input.width)?;
        if input.data.len() != want {
            return Err(ConvError::BadShape { what: "conv input", got: input.data.len(), want });
        }
        check_finite(&input.data, "conv input")?;
        let (oh, ow) = self.spec.require_out_shape(input.height, input.width)?;

        let mut out = Tensor3::zeros(self.spec.out_channels, oh, ow)?;
        let mut dense: u64 = 0;
        let mut effective: u64 = 0;
        for oc in 0..self.spec.out_channels {
            for oy in 0..oh {
                for ox in 0..ow {
                    // Fixed accumulation order — in channel, then ky, then kx — so the sum is
                    // reproducible bit for bit. Floating-point addition is not associative and a
                    // parallel reduction over the same taps would give a different last bit.
                    let mut acc = self.bias[oc];
                    for ic in 0..self.spec.in_channels {
                        for ky in 0..self.spec.kernel_h {
                            let iy = match src_coord(oy, self.spec.stride_h, ky, self.spec.dilation_h, self.spec.pad_h, input.height) {
                                Some(v) => v,
                                None => continue,
                            };
                            for kx in 0..self.spec.kernel_w {
                                let ix = match src_coord(ox, self.spec.stride_w, kx, self.spec.dilation_w, self.spec.pad_w, input.width) {
                                    Some(v) => v,
                                    None => continue,
                                };
                                dense += 1;
                                let a = input.data[input.index(ic, iy, ix)];
                                let w = self.weights[self.weight_index(oc, ic, ky, kx)];
                                if a != 0.0 && w != 0.0 {
                                    effective += 1;
                                    acc += w * a;
                                }
                            }
                        }
                    }
                    let idx = out.index(oc, oy, ox);
                    out.data[idx] = acc;
                }
            }
        }
        Ok((out, dense, effective))
    }

    /// The convolution of `input`, volts.
    ///
    /// Zero padding, cross-correlation, fixed accumulation order. Positions whose window lies
    /// wholly on padding receive the bias alone, which is correct and is not the same as receiving
    /// nothing.
    ///
    /// # Errors
    ///
    /// [`ConvError::ShapeMismatch`] for the wrong channel count, [`ConvError::BadShape`] for a
    /// tensor whose data length contradicts its shape, [`ConvError::NonFinite`] for a `NaN` or
    /// infinite input, [`ConvError::ImpossibleShape`] when no window fits.
    pub fn forward(&self, input: &Tensor3) -> Result<Tensor3, ConvError> {
        Ok(self.convolve(input)?.0)
    }

    /// Synaptic operations this layer performs on `input`, in [`crate::metrics`]'s terms.
    ///
    /// `dense` is [`Conv2d::in_bounds_taps`] — real connections, not the padded figure. The
    /// effective count goes to `effective_acs` for [`ActivationKind::Spiking`] (a binary input
    /// needs no multiplier) and to `effective_macs` for [`ActivationKind::RealValued`].
    ///
    /// # Errors
    ///
    /// As [`Conv2d::forward`], plus [`ConvError::NotBinary`] when `kind` is
    /// [`ActivationKind::Spiking`] and an input is neither `0.0` nor `1.0` — which is exactly the
    /// condition a [`ResidualStyle::SewAdd`] output puts the next layer in.
    pub fn synops(&self, input: &Tensor3, kind: ActivationKind) -> Result<SynOps, ConvError> {
        if kind == ActivationKind::Spiking {
            input.require_binary()?;
        }
        let (_, dense, effective) = self.convolve(input)?;
        Ok(match kind {
            ActivationKind::Spiking => {
                SynOps { dense, effective_macs: 0, effective_acs: effective }
            }
            ActivationKind::RealValued => {
                SynOps { dense, effective_macs: effective, effective_acs: 0 }
            }
        })
    }
}

/// Input coordinate for output index `o`, tap `k`, or `None` when the tap falls on padding.
fn src_coord(o: usize, stride: usize, k: usize, dilation: usize, pad: usize, n: usize) -> Option<usize> {
    let p = o * stride + k * dilation;
    if p < pad {
        return None;
    }
    let i = p - pad;
    if i >= n { None } else { Some(i) }
}

/// The most `outputs × taps` iterations [`Conv2d::in_bounds_taps`] will sweep on one axis: `2^26`.
pub const MAX_TAP_SWEEP: u64 = 1 << 26;

/// In-bounds taps summed over all output indices on one axis.
fn axis_in_bounds(o_n: usize, n: usize, kernel: usize, pad: usize, stride: usize, dilation: usize) -> Option<u64> {
    if (o_n as u64).checked_mul(kernel as u64)? > MAX_TAP_SWEEP {
        return None;
    }
    let mut total: u64 = 0;
    for o in 0..o_n {
        for k in 0..kernel {
            if src_coord(o, stride, k, dilation, pad, n).is_some() {
                total = total.checked_add(1)?;
            }
        }
    }
    Some(total)
}

/// Shape of a pooling window: kernel, stride and padding per axis, no dilation.
///
/// No dilation because this review did not locate a dilated pooling layer in any spiking vision
/// architecture it examined, and an untested parameter is a liability rather than a feature.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PoolSpec {
    /// Window rows. At least one.
    pub kernel_h: usize,
    /// Window columns. At least one.
    pub kernel_w: usize,
    /// Row stride, elements. At least one. Equal to `kernel_h` for non-overlapping pooling.
    pub stride_h: usize,
    /// Column stride, elements. At least one.
    pub stride_w: usize,
    /// Zero padding on each side of the row axis, elements.
    pub pad_h: usize,
    /// Zero padding on each side of the column axis, elements.
    pub pad_w: usize,
}

impl PoolSpec {
    /// A non-overlapping window: stride equal to kernel, no padding.
    ///
    /// # Errors
    ///
    /// [`ConvError::Zero`] for a zero extent.
    pub fn new(kernel_h: usize, kernel_w: usize) -> Result<Self, ConvError> {
        if kernel_h == 0 {
            return Err(ConvError::Zero { what: "kernel_h" });
        }
        if kernel_w == 0 {
            return Err(ConvError::Zero { what: "kernel_w" });
        }
        Ok(Self { kernel_h, kernel_w, stride_h: kernel_h, stride_w: kernel_w, pad_h: 0, pad_w: 0 })
    }

    /// Output `(height, width)` for an input of `(in_h, in_w)`, or `None` if no window fits.
    #[must_use]
    pub fn out_shape(&self, in_h: usize, in_w: usize) -> Option<(usize, usize)> {
        Some((
            out_dim(in_h, self.kernel_h, self.pad_h, self.stride_h, 1)?,
            out_dim(in_w, self.kernel_w, self.pad_w, self.stride_w, 1)?,
        ))
    }

    /// Same as [`PoolSpec::out_shape`] but naming the axis that failed.
    ///
    /// # Errors
    ///
    /// [`ConvError::ImpossibleShape`] carrying the failing axis's parameters.
    pub fn require_out_shape(&self, in_h: usize, in_w: usize) -> Result<(usize, usize), ConvError> {
        let oh = out_dim(in_h, self.kernel_h, self.pad_h, self.stride_h, 1).ok_or(
            ConvError::ImpossibleShape {
                axis: "height",
                n: in_h,
                kernel: self.kernel_h,
                pad: self.pad_h,
                stride: self.stride_h,
                dilation: 1,
            },
        )?;
        let ow = out_dim(in_w, self.kernel_w, self.pad_w, self.stride_w, 1).ok_or(
            ConvError::ImpossibleShape {
                axis: "width",
                n: in_w,
                kernel: self.kernel_w,
                pad: self.pad_w,
                stride: self.stride_w,
                dilation: 1,
            },
        )?;
        Ok((oh, ow))
    }
}

/// Average pooling, per channel, in a fixed accumulation order.
///
/// `count_include_pad` decides the divisor at the border, and it is an argument rather than a
/// default because the two conventions give different numbers and both are in use: `PyTorch`'s
/// `AvgPool2d` defaults to `true`, and its `functional.avg_pool2d` takes the same flag.
/// With `false` the divisor is the number of **real** elements in the window, so a constant input
/// pools to exactly that constant everywhere including the border; with `true` the padded zeros are
/// counted and the border is pulled toward zero, which is a real and usually unintended shrinkage.
///
/// # Exactness
///
/// The window is summed in row-major order and divided once. For an input whose value `c` is a
/// dyadic rational small enough that every partial sum `m * c` is representable, the sum is exact
/// and `m * c / m` is exactly `c` — no tolerance, and
/// `average_pooling_a_constant_returns_exactly_that_constant` asserts bit equality across a sweep
/// of windows, strides and constants. For a general real constant the result is within a rounding
/// of `c` and this module does not claim more.
///
/// # Errors
///
/// [`ConvError::Zero`] for a zero window extent, [`ConvError::ImpossibleShape`] when no window
/// fits, [`ConvError::NonFinite`] for a non-finite input, [`ConvError::EmptyWindow`] when
/// `count_include_pad` is `false` and some window contains no real element.
pub fn avg_pool(input: &Tensor3, spec: &PoolSpec, count_include_pad: bool) -> Result<Tensor3, ConvError> {
    let (oh, ow) = pool_prepare(input, spec)?;
    let mut out = Tensor3::zeros(input.channels, oh, ow)?;
    for c in 0..input.channels {
        for oy in 0..oh {
            for ox in 0..ow {
                let mut sum = 0.0;
                let mut real: usize = 0;
                for ky in 0..spec.kernel_h {
                    let iy = src_coord(oy, spec.stride_h, ky, 1, spec.pad_h, input.height);
                    for kx in 0..spec.kernel_w {
                        let ix = src_coord(ox, spec.stride_w, kx, 1, spec.pad_w, input.width);
                        if let (Some(iy), Some(ix)) = (iy, ix) {
                            sum += input.data[input.index(c, iy, ix)];
                            real += 1;
                        }
                    }
                }
                let n = if count_include_pad { spec.kernel_h * spec.kernel_w } else { real };
                if n == 0 {
                    return Err(ConvError::EmptyWindow { y: oy, x: ox });
                }
                let idx = out.index(c, oy, ox);
                out.data[idx] = sum / n as f64;
            }
        }
    }
    Ok(out)
}

/// Max pooling over **real-valued** maps — membrane potentials, `tdBN` outputs, rate estimates.
///
/// Unambiguous precisely because the input is not a spike train: the maximum of a set of reals is a
/// number. This is the function to use when the architecture pools membranes rather than spikes,
/// which is one of the two defensible readings of "spiking max pooling"; the other is
/// [`SpikingMaxPool`].
///
/// # Errors
///
/// As [`avg_pool`], plus [`ConvError::EmptyWindow`] for a window wholly on padding — there is no
/// maximum of an empty set and `0.0` would be indistinguishable from a window of real zeros.
pub fn max_pool_values(input: &Tensor3, spec: &PoolSpec) -> Result<Tensor3, ConvError> {
    let (oh, ow) = pool_prepare(input, spec)?;
    let mut out = Tensor3::zeros(input.channels, oh, ow)?;
    for c in 0..input.channels {
        for oy in 0..oh {
            for ox in 0..ow {
                let mut best: Option<f64> = None;
                for ky in 0..spec.kernel_h {
                    let iy = src_coord(oy, spec.stride_h, ky, 1, spec.pad_h, input.height);
                    for kx in 0..spec.kernel_w {
                        let ix = src_coord(ox, spec.stride_w, kx, 1, spec.pad_w, input.width);
                        if let (Some(iy), Some(ix)) = (iy, ix) {
                            let v = input.data[input.index(c, iy, ix)];
                            best = Some(match best {
                                Some(b) if b >= v => b,
                                _ => v,
                            });
                        }
                    }
                }
                let idx = out.index(c, oy, ox);
                out.data[idx] = best.ok_or(ConvError::EmptyWindow { y: oy, x: ox })?;
            }
        }
    }
    Ok(out)
}

fn pool_prepare(input: &Tensor3, spec: &PoolSpec) -> Result<(usize, usize), ConvError> {
    for (what, n) in [
        ("kernel_h", spec.kernel_h),
        ("kernel_w", spec.kernel_w),
        ("stride_h", spec.stride_h),
        ("stride_w", spec.stride_w),
    ] {
        if n == 0 {
            return Err(ConvError::Zero { what });
        }
    }
    let want = dims_len(input.channels, input.height, input.width)?;
    if input.data.len() != want {
        return Err(ConvError::BadShape { what: "pool input", got: input.data.len(), want });
    }
    check_finite(&input.data, "pool input")?;
    spec.require_out_shape(input.height, input.width)
}

/// How to take a maximum over spikes — the operation that has no unambiguous definition.
///
/// # Why this is a real problem and not a detail
///
/// Max pooling over a real-valued map means "keep the strongest response and discard the rest".
/// Over a **binary** map at a single timestep, every response is `0` or `1`, so the maximum is a
/// logical OR: it says *some* unit in the window fired and forgets which, how many, and by how
/// much. Whatever the pooling was for, that is not it.
///
/// The literature's responses, as far as this review located them, are three. Rueckauer, Lungu, Hu,
/// Pfeiffer & Liu (*Frontiers in Neuroscience* 11:682, 2017) gate the output onto the input unit
/// with the highest **running firing-rate estimate**, which is [`MaxPolicy::RateGated`]. Others
/// pool the membrane potential rather than the spikes, which is [`max_pool_values`] and is not a
/// spiking operation at all — it needs the analogue state that event-driven routing was supposed to
/// avoid sending. And a large fraction of the field simply uses average pooling instead, which is
/// well defined over spikes because the mean of a binary window is its firing fraction; that is
/// [`avg_pool`], and it is the default this module would suggest.
///
/// None of the three is the max pooling of a conventional network. Choosing one is an architectural
/// decision with a measurable cost, which is why this enum has no `Default`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaxPolicy {
    /// Logical OR over the window at this timestep: `1.0` if any unit in the window spiked.
    ///
    /// Stateless, exact, and lossy in a specific way: a window where four units fire and a window
    /// where one fires are indistinguishable, so pooling **raises** the output firing rate relative
    /// to every input unit and the layer above sees a denser code than the layer below produced.
    Instant,
    /// Rueckauer et al. 2017: pass through the spikes of whichever unit in the window has fired
    /// most so far, and ignore the others.
    ///
    /// The estimate is a cumulative spike count per input unit, updated **after** the output is
    /// produced, so the gate is strictly causal — at timestep `t` it depends only on timesteps
    /// `< t`. Including the current timestep would make the first output an OR and would leak
    /// information backwards in a way an event-driven implementation could not reproduce.
    ///
    /// Ties, including the all-zero counts at the first timestep, go to the **lowest flat index**
    /// in the window. Deterministic and arbitrary; a random tie-break would be neither.
    ///
    /// The cost is visible and sharp: once one unit is ahead, spikes from a unit that has just
    /// become more active are dropped until the cumulative count catches up, which can take as many
    /// timesteps as the lead was long. `rate_gated_max_pooling_drops_the_newly_active_unit`
    /// measures exactly that.
    RateGated,
}

/// Max pooling over spike maps, carrying whatever state the policy needs.
///
/// Fixed to one input shape at construction because [`MaxPolicy::RateGated`] holds one counter per
/// input element and a shape change would silently reinterpret them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpikingMaxPool {
    /// Window shape.
    pub spec: PoolSpec,
    /// How the maximum is resolved. See [`MaxPolicy`].
    pub policy: MaxPolicy,
    /// Input channels this pool was built for.
    pub channels: usize,
    /// Input rows this pool was built for.
    pub height: usize,
    /// Input columns this pool was built for.
    pub width: usize,
    /// Cumulative spikes per input element, in [`Tensor3`] layout. All zero for
    /// [`MaxPolicy::Instant`], which never reads them.
    pub counts: Vec<u64>,
}

impl SpikingMaxPool {
    /// Build for a fixed input shape, checking that a window fits.
    ///
    /// # Errors
    ///
    /// [`ConvError::Zero`] for a zero dimension or extent, [`ConvError::ImpossibleShape`] when no
    /// window fits, [`ConvError::Overflow`] on an element count past `usize`.
    pub fn new(
        spec: PoolSpec,
        policy: MaxPolicy,
        channels: usize,
        height: usize,
        width: usize,
    ) -> Result<Self, ConvError> {
        let n = dims_len(channels, height, width)?;
        spec.require_out_shape(height, width)?;
        Ok(Self { spec, policy, channels, height, width, counts: vec![0; n] })
    }

    /// Output `(channels, height, width)`.
    ///
    /// # Errors
    ///
    /// [`ConvError::ImpossibleShape`] — cannot occur after a successful [`SpikingMaxPool::new`],
    /// and is returned rather than unwrapped because the fields are public and a caller may have
    /// changed them.
    pub fn out_shape(&self) -> Result<(usize, usize, usize), ConvError> {
        let (oh, ow) = self.spec.require_out_shape(self.height, self.width)?;
        Ok((self.channels, oh, ow))
    }

    /// Pool one timestep of spikes.
    ///
    /// # Errors
    ///
    /// [`ConvError::ShapeMismatch`] for an input of a different shape than this pool was built for,
    /// [`ConvError::NotBinary`] for a non-spike input, [`ConvError::EmptyWindow`] for a window
    /// wholly on padding, [`ConvError::ImpossibleShape`] if the public fields have been changed to
    /// an impossible combination.
    pub fn step(&mut self, spikes: &Tensor3) -> Result<Tensor3, ConvError> {
        if spikes.shape() != (self.channels, self.height, self.width) {
            return Err(ConvError::ShapeMismatch {
                what: "spiking max pool input",
                a: spikes.shape(),
                b: (self.channels, self.height, self.width),
            });
        }
        let want = dims_len(self.channels, self.height, self.width)?;
        if spikes.data.len() != want {
            return Err(ConvError::BadShape { what: "spiking max pool input", got: spikes.data.len(), want });
        }
        spikes.require_binary()?;
        let (oh, ow) = self.spec.require_out_shape(self.height, self.width)?;
        let mut out = Tensor3::zeros(self.channels, oh, ow)?;
        for c in 0..self.channels {
            for oy in 0..oh {
                for ox in 0..ow {
                    let mut value: Option<f64> = None;
                    let mut best_count: u64 = 0;
                    let mut seen = false;
                    for ky in 0..self.spec.kernel_h {
                        let iy = src_coord(oy, self.spec.stride_h, ky, 1, self.spec.pad_h, self.height);
                        for kx in 0..self.spec.kernel_w {
                            let ix = src_coord(ox, self.spec.stride_w, kx, 1, self.spec.pad_w, self.width);
                            let (Some(iy), Some(ix)) = (iy, ix) else { continue };
                            let idx = spikes.index(c, iy, ix);
                            let s = spikes.data[idx];
                            match self.policy {
                                MaxPolicy::Instant => {
                                    value = Some(value.unwrap_or(0.0).max(s));
                                }
                                MaxPolicy::RateGated => {
                                    let n = self.counts[idx];
                                    // Strictly greater: the first element scanned wins a tie, which
                                    // is the lowest flat index because the scan is in layout order.
                                    if !seen || n > best_count {
                                        best_count = n;
                                        value = Some(s);
                                    }
                                }
                            }
                            seen = true;
                        }
                    }
                    let idx = out.index(c, oy, ox);
                    out.data[idx] = value.ok_or(ConvError::EmptyWindow { y: oy, x: ox })?;
                }
            }
        }
        if self.policy == MaxPolicy::RateGated {
            for (n, s) in self.counts.iter_mut().zip(spikes.data.iter()) {
                if *s != 0.0 {
                    *n = n.saturating_add(1);
                }
            }
        }
        Ok(out)
    }

    /// Forget the rate estimates. The window shape is untouched.
    pub fn reset(&mut self) {
        self.counts.fill(0);
    }
}

/// Threshold-dependent batch normalisation — Zheng, Wu, Deng, Yan & Li, `AAAI` 2021.
///
/// # The lesson
///
/// Batch normalisation standardises a layer's pre-activation to zero mean and unit variance. In a
/// rectifier network unit variance is the right target because the nonlinearity's interesting
/// region is around zero and is scale-free. In a spiking network it is the wrong target, because
/// the nonlinearity's interesting region is a **threshold** at a particular height: a pre-activation
/// distribution much narrower than `V_th` produces a layer that never fires, and one much wider
/// produces a layer where every unit fires on every timestep. Both are dead ends for a surrogate
/// gradient, which is largest for units near threshold and vanishes for units far from it in either
/// direction.
///
/// tdBN normalises to standard deviation `alpha * V_th` instead of to `1`, and takes its statistics
/// over the **time** axis as well as the batch and the two spatial axes — a spiking layer's
/// activations at different timesteps are the same layer, and treating them as separate
/// distributions would normalise away the temporal structure the network exists to use.
///
/// ```text
/// x_hat = alpha * V_th * (x - mean) / sqrt(var + eps)
/// y     = gain * x_hat + shift
/// ```
///
/// with `mean` and `var` taken per channel over `(sample, timestep, row, column)`. `gain` and
/// `shift` are the learnable pair every batch-norm has, initialised to `1` and `0`.
///
/// # Units, and where the conversion happens
///
/// The paper's expression is dimensionless because `x / V_th` is a ratio. Here `v_th` is supplied
/// in **volts measured from the resting potential** and the products are volts, so the whole
/// expression is in the same units as a [`Conv2d`] pre-activation and can be handed straight to
/// [`crate::neuron::Neuron::bump`]. That is the boundary conversion; nothing inside the expression
/// is rescaled, so it can be read line by line against the paper.
///
/// # What this implementation is unsure of
///
/// `alpha` defaults to `1.0`, which is the value the paper's main experiments use. The paper also
/// discusses scaling the shortcut branch of a residual block by `1 / sqrt(2)` so that the summed
/// branches keep the target variance — the elementary reason being that two independent variables
/// of variance `s^2` sum to `2 s^2`. **This review did not locate an unambiguous statement of which
/// blocks take which `alpha` in every architecture the paper reports**, so `1 / sqrt(2)` is offered
/// as [`TdBn::residual_alpha`] with that caveat attached rather than applied silently.
#[derive(Debug, Clone, PartialEq)]
pub struct TdBn {
    /// Channels, which is the number of independent statistics kept.
    pub channels: usize,
    /// The paper's `alpha`, dimensionless. The target standard deviation is `alpha * v_th`.
    pub alpha: f64,
    /// Firing threshold measured from rest, **volts**. See the struct doc on units.
    pub v_th: f64,
    /// Added to the variance before the square root, in volts squared.
    ///
    /// Defaults to `1e-5` as in every batch-norm implementation this review checked. **`0.0` is
    /// permitted**, and is what makes the normalisation hit its target statistic exactly rather
    /// than within a factor of `sqrt(var / (var + eps))`; the caller then owns the case of a
    /// channel with zero variance, which divides by zero.
    pub eps: f64,
    /// Learnable scale per channel, dimensionless. Initialised to `1.0`.
    pub gain: Vec<f64>,
    /// Learnable offset per channel, volts. Initialised to `0.0`.
    pub shift: Vec<f64>,
    /// Running mean per channel, volts, for [`TdBn::eval_forward`].
    pub running_mean: Vec<f64>,
    /// Running variance per channel, volts squared. Initialised to `1.0` so that a layer used for
    /// inference before it has ever seen a batch scales by `1` rather than dividing by zero.
    pub running_var: Vec<f64>,
    /// Exponential-average rate for the running statistics, in `[0, 1]`. `0.1` by default, matching
    /// `PyTorch`'s `momentum`; `1.0` replaces the running statistics with the last batch's.
    pub momentum: f64,
}

impl TdBn {
    /// The residual-branch `alpha` of `1 / sqrt(2)`, offered with the caveat in the struct doc.
    ///
    /// Two independent variables of variance `s^2` sum to variance `2 s^2`, so scaling each by
    /// `1 / sqrt(2)` restores the target. That much is arithmetic. Whether it is what the paper
    /// applies to every block is what this review could not settle.
    #[must_use]
    pub fn residual_alpha() -> f64 {
        1.0 / 2.0_f64.sqrt()
    }

    /// A layer with `alpha = 1`, `eps = 1e-5`, `momentum = 0.1`, unit gain and zero shift.
    ///
    /// # Errors
    ///
    /// [`ConvError::Zero`] for zero channels, [`ConvError::NonFinite`] for a non-finite or negative
    /// `v_th`.
    pub fn new(channels: usize, v_th: f64) -> Result<Self, ConvError> {
        if channels == 0 {
            return Err(ConvError::Zero { what: "channels" });
        }
        if !v_th.is_finite() || v_th < 0.0 {
            return Err(ConvError::NonFinite { what: "v_th", index: 0, value: v_th });
        }
        Ok(Self {
            channels,
            alpha: 1.0,
            v_th,
            eps: 1e-5,
            gain: vec![1.0; channels],
            shift: vec![0.0; channels],
            running_mean: vec![0.0; channels],
            running_var: vec![1.0; channels],
            momentum: 0.1,
        })
    }

    /// The standard deviation this layer normalises to, `alpha * v_th` volts.
    #[must_use]
    pub fn target_sigma(&self) -> f64 {
        self.alpha * self.v_th
    }

    /// Normalise a batch with the batch's own statistics and update the running ones.
    ///
    /// `batch` is **every (sample, timestep) feature map**, flattened into one slice. That
    /// flattening is what makes this tdBN rather than plain batch normalisation: pass one timestep
    /// at a time and the statistics are per-timestep, which is the thing the paper changed.
    ///
    /// # Errors
    ///
    /// [`ConvError::Zero`] for an empty batch or a channel count of zero,
    /// [`ConvError::ShapeMismatch`] if the maps do not all share a shape or do not match
    /// `self.channels`, [`ConvError::BadShape`] for a `gain`, `shift` or running buffer of the
    /// wrong length, [`ConvError::NonFinite`] for a non-finite input or parameter.
    pub fn train_forward(&mut self, batch: &[Tensor3]) -> Result<Vec<Tensor3>, ConvError> {
        self.check_params()?;
        let (mean, var, count) = channel_moments(batch, self.channels)?;
        let sigma = self.target_sigma();
        let mut out = Vec::with_capacity(batch.len());
        for t in batch {
            let mut o = t.clone();
            for c in 0..self.channels {
                let denom = (var[c] + self.eps).sqrt();
                let scale = self.gain[c] * sigma / denom;
                for y in 0..t.height {
                    for x in 0..t.width {
                        let i = t.index(c, y, x);
                        o.data[i] = (t.data[i] - mean[c]) * scale + self.shift[c];
                    }
                }
            }
            out.push(o);
        }
        // PyTorch tracks the UNBIASED variance in the running buffer while normalising with the
        // biased one. Reproduced here rather than simplified, because a model whose running buffer
        // is off by n/(n-1) gives a different inference result than the framework it was trained
        // in, on the same weights. With a single element there is no unbiased estimate and the
        // biased one is carried through.
        let n = count as f64;
        let correction = if count > 1 { n / (n - 1.0) } else { 1.0 };
        for c in 0..self.channels {
            self.running_mean[c] = (1.0 - self.momentum) * self.running_mean[c] + self.momentum * mean[c];
            self.running_var[c] =
                (1.0 - self.momentum) * self.running_var[c] + self.momentum * var[c] * correction;
        }
        Ok(out)
    }

    /// Normalise one map with the running statistics, leaving them unchanged.
    ///
    /// # Errors
    ///
    /// As [`TdBn::train_forward`], for one map.
    pub fn eval_forward(&self, x: &Tensor3) -> Result<Tensor3, ConvError> {
        self.check_params()?;
        if x.channels != self.channels {
            return Err(ConvError::ShapeMismatch {
                what: "tdBN input channels",
                a: x.shape(),
                b: (self.channels, x.height, x.width),
            });
        }
        let want = dims_len(x.channels, x.height, x.width)?;
        if x.data.len() != want {
            return Err(ConvError::BadShape { what: "tdBN input", got: x.data.len(), want });
        }
        check_finite(&x.data, "tdBN input")?;
        let sigma = self.target_sigma();
        let mut o = x.clone();
        for c in 0..self.channels {
            let denom = (self.running_var[c] + self.eps).sqrt();
            let scale = self.gain[c] * sigma / denom;
            for y in 0..x.height {
                for xx in 0..x.width {
                    let i = x.index(c, y, xx);
                    o.data[i] = (x.data[i] - self.running_mean[c]) * scale + self.shift[c];
                }
            }
        }
        Ok(o)
    }

    fn check_params(&self) -> Result<(), ConvError> {
        if self.channels == 0 {
            return Err(ConvError::Zero { what: "channels" });
        }
        for (what, v) in [
            ("gain", &self.gain),
            ("shift", &self.shift),
            ("running_mean", &self.running_mean),
            ("running_var", &self.running_var),
        ] {
            if v.len() != self.channels {
                return Err(ConvError::BadShape { what, got: v.len(), want: self.channels });
            }
            check_finite(v, what)?;
        }
        for (what, v) in
            [("alpha", self.alpha), ("v_th", self.v_th), ("eps", self.eps), ("momentum", self.momentum)]
        {
            if !v.is_finite() {
                return Err(ConvError::NonFinite { what, index: 0, value: v });
            }
        }
        if self.eps < 0.0 {
            return Err(ConvError::NonFinite { what: "eps", index: 0, value: self.eps });
        }
        // `momentum` documents a range and had no check: at -1 the running variance goes negative
        // and every inference output is NaN, with `Ok` on every call. Same for a running variance
        // set negative through the public field.
        if !(0.0..=1.0).contains(&self.momentum) {
            return Err(ConvError::NonFinite { what: "momentum (outside [0, 1])", index: 0, value: self.momentum });
        }
        if let Some((i, &v)) = self.running_var.iter().enumerate().find(|(_, v)| **v < 0.0) {
            return Err(ConvError::NonFinite { what: "running_var (negative)", index: i, value: v });
        }
        Ok(())
    }
}

/// Per-channel mean and **population** variance over a whole batch of maps, plus the element count
/// each statistic was taken over.
///
/// Two passes — mean, then squared deviations from it — rather than the `E[x^2] - E[x]^2` shortcut,
/// which cancels catastrophically when the mean is large compared with the spread and can return a
/// negative variance that then produces a `NaN` at the square root. That failure is silent and
/// looks like a bad initialisation.
///
/// The variance is the biased one, dividing by `n` rather than `n - 1`: it is what the
/// normalisation divides by, in tdBN and in every batch-norm implementation this review checked.
/// [`TdBn::train_forward`] applies Bessel's correction where it stores the running buffer, and only
/// there.
///
/// # Errors
///
/// [`ConvError::Zero`] for an empty batch or zero channels, [`ConvError::ShapeMismatch`] for maps
/// that do not all share a shape or do not have `channels` channels, [`ConvError::NonFinite`] for a
/// non-finite element.
pub fn channel_moments(batch: &[Tensor3], channels: usize) -> Result<(Vec<f64>, Vec<f64>, usize), ConvError> {
    if batch.is_empty() {
        return Err(ConvError::Zero { what: "batch length" });
    }
    if channels == 0 {
        return Err(ConvError::Zero { what: "channels" });
    }
    let shape = batch[0].shape();
    for t in batch {
        if t.shape() != shape {
            return Err(ConvError::ShapeMismatch { what: "batch member", a: t.shape(), b: shape });
        }
        if t.channels != channels {
            return Err(ConvError::ShapeMismatch {
                what: "batch channels",
                a: t.shape(),
                b: (channels, t.height, t.width),
            });
        }
        let want = dims_len(t.channels, t.height, t.width)?;
        if t.data.len() != want {
            return Err(ConvError::BadShape { what: "batch member", got: t.data.len(), want });
        }
        check_finite(&t.data, "batch element")?;
    }
    let per_channel = shape.1 * shape.2;
    let count = per_channel * batch.len();
    if count == 0 {
        return Err(ConvError::Zero { what: "elements per channel" });
    }
    let mut mean = vec![0.0; channels];
    let mut var = vec![0.0; channels];
    for c in 0..channels {
        let mut sum = 0.0;
        for t in batch {
            for y in 0..t.height {
                for x in 0..t.width {
                    sum += t.data[t.index(c, y, x)];
                }
            }
        }
        let m = sum / count as f64;
        let mut ss = 0.0;
        for t in batch {
            for y in 0..t.height {
                for x in 0..t.width {
                    let d = t.data[t.index(c, y, x)] - m;
                    ss += d * d;
                }
            }
        }
        mean[c] = m;
        var[c] = ss / count as f64;
    }
    Ok((mean, var, count))
}

/// A convolution with one spiking neuron per output unit, stepped one timestep at a time.
///
/// The layer holds the membranes; the caller holds the frames. Per timestep, for every output unit:
/// deliver the convolution's pre-activation through [`crate::neuron::Neuron::bump`], then advance
/// the membrane with [`crate::neuron::Neuron::step`] under **zero** external current. That is the
/// same order [`crate::sim`] uses, and it is the delta-synapse convention: a weight is a voltage
/// displacement, not a current, so halving `dt` does not halve every synaptic influence.
#[derive(Debug, Clone, PartialEq)]
pub struct SpikingConv2d<N: Neuron> {
    /// The kernel.
    pub conv: Conv2d,
    /// Input rows this layer was built for.
    pub in_height: usize,
    /// Input columns this layer was built for.
    pub in_width: usize,
    /// Output rows, from [`Conv2dSpec::out_shape`].
    pub out_height: usize,
    /// Output columns.
    pub out_width: usize,
    /// Timestep, **seconds**. Finite and strictly positive.
    pub dt: f64,
    /// One neuron per output unit, in [`Tensor3`] layout over `(out_channel, row, column)`.
    ///
    /// A `Vec<N>` rather than `Vec<Box<dyn Neuron>>` for the reason [`crate::neuron::Neuron`]
    /// gives: membrane state scattered across the heap costs a pointer chase per unit per timestep,
    /// and on a `128x128` map with 64 channels that is a million of them.
    pub units: Vec<N>,
    /// Exact counters for [`crate::ledger`], accumulated across every [`SpikingConv2d::step`] since
    /// the last [`SpikingConv2d::clear_counts`]. Not cleared by [`SpikingConv2d::reset`].
    ///
    /// `neuron_updates_idle` means the unit's **summed drive was exactly zero** this step, not that
    /// it received nothing: two deliveries of `+20 mV` and `-20 mV` cancel to an idle update while
    /// `syn_ops` counts both, and a bias of one picovolt makes every unit driven every step. The
    /// idle fraction is therefore a statement about membrane arithmetic an event-driven core could
    /// skip; the synaptic traffic is in `syn_ops` and `syn_fetches`, which do not cancel.
    pub ledger: Ledger,
    /// `NeuroBench` synaptic-operation counts, accumulated alongside the ledger.
    pub ops: SynOps,
}

impl<N: Neuron> SpikingConv2d<N> {
    /// Build for a fixed input shape, cloning `proto` into every output unit.
    ///
    /// # Errors
    ///
    /// [`ConvError::Zero`] for a zero dimension, [`ConvError::ImpossibleShape`] when no window
    /// fits, [`ConvError::NonFinite`] for a `dt` that is not finite and strictly positive.
    pub fn new(conv: Conv2d, in_height: usize, in_width: usize, proto: N, dt: f64) -> Result<Self, ConvError> {
        if !dt.is_finite() || dt <= 0.0 {
            return Err(ConvError::NonFinite { what: "dt", index: 0, value: dt });
        }
        let (out_height, out_width) = conv.spec.require_out_shape(in_height, in_width)?;
        let n = dims_len(conv.spec.out_channels, out_height, out_width)?;
        Ok(Self {
            conv,
            in_height,
            in_width,
            out_height,
            out_width,
            dt,
            units: vec![proto; n],
            ledger: Ledger::default(),
            ops: SynOps::default(),
        })
    }

    /// Output `(channels, height, width)`.
    #[must_use]
    pub fn out_shape(&self) -> (usize, usize, usize) {
        (self.conv.spec.out_channels, self.out_height, self.out_width)
    }

    /// Advance one timestep and return the output spike map.
    ///
    /// `kind` says how to charge the input: [`ActivationKind::Spiking`] requires a binary input and
    /// charges accumulates, [`ActivationKind::RealValued`] accepts graded input — which is what a
    /// [`ResidualStyle::SewAdd`] block upstream produces — and counts multiply-accumulates. The
    /// argument is not inferred from the data, because a batch that happens to contain only zeros
    /// and ones would then be charged the cheaper rate and the layer's cost would depend on its
    /// input rather than on its design.
    ///
    /// ⛔ **A multiply-accumulate is not charged to the ledger's `syn_ops`.** That counter is what
    /// [`crate::ledger::Prices::e_syn_op`] prices — a spike arriving at a synapse and
    /// *accumulating* — and a MAC on a graded input is a different operation with no price in this
    /// crate. The first version charged both at the accumulate rate, so a graded and a binary
    /// workload produced byte-identical ledgers: it supplied, silently, exactly the AC:MAC ratio
    /// [`crate::metrics::SynOps::effective_total`] declines to supply. MACs are counted in
    /// [`SpikingConv2d::ops`] where [`crate::ledger::Ledger::bill`] cannot see them, which is the
    /// honest state of the accounting. The weight **fetch** is charged either way, because the
    /// weight is read from memory before either operation can happen.
    ///
    /// # Errors
    ///
    /// [`ConvError::ShapeMismatch`] for an input of the wrong shape, [`ConvError::NotBinary`] for a
    /// graded input under [`ActivationKind::Spiking`], [`ConvError::NonFinite`] for a non-finite
    /// input, [`ConvError::Overflow`] on a count past `u64`.
    pub fn step(&mut self, input: &Tensor3, kind: ActivationKind) -> Result<Tensor3, ConvError> {
        if (input.channels, input.height, input.width)
            != (self.conv.spec.in_channels, self.in_height, self.in_width)
        {
            return Err(ConvError::ShapeMismatch {
                what: "spiking conv input",
                a: input.shape(),
                b: (self.conv.spec.in_channels, self.in_height, self.in_width),
            });
        }
        if kind == ActivationKind::Spiking {
            input.require_binary()?;
        }
        // The fields are public. A caller who changed `conv.spec` after construction would
        // otherwise index a drive of one size with a unit array of another: an out-of-bounds
        // panic when the drive shrank, a silently scrambled spatial map when it grew.
        let (oh, ow) = self.conv.spec.require_out_shape(self.in_height, self.in_width)?;
        let now = (self.conv.spec.out_channels, oh, ow);
        if now != self.out_shape() || self.units.len() != dims_len(now.0, now.1, now.2)? {
            return Err(ConvError::ShapeMismatch { what: "spiking conv units", a: now, b: self.out_shape() });
        }
        // `conv.weights` and `conv.bias` are public too. One NaN weight, set after construction,
        // poisons every membrane downstream with `Ok` on every step — the exact failure the
        // `NonFinite` variant's doc describes — so they are re-checked here, once per step.
        check_finite(&self.conv.weights, "conv weights")?;
        check_finite(&self.conv.bias, "conv bias")?;
        let (drive, dense, effective) = self.conv.convolve(input)?;
        let step_ops = match kind {
            ActivationKind::Spiking => SynOps { dense, effective_macs: 0, effective_acs: effective },
            ActivationKind::RealValued => {
                SynOps { dense, effective_macs: effective, effective_acs: 0 }
            }
        };
        self.ops.add(step_ops)?;
        if kind == ActivationKind::Spiking {
            self.ledger.syn_ops = self
                .ledger
                .syn_ops
                .checked_add(effective)
                .ok_or(ConvError::Overflow { what: "syn_ops" })?;
        }
        // One fetch per delivery. A layer that cached the kernel across positions would fetch far
        // fewer, which is exactly why the ledger counts fetches separately from operations instead
        // of assuming a ratio — see `Ledger::syn_fetches`.
        self.ledger.syn_fetches = self
            .ledger
            .syn_fetches
            .checked_add(effective)
            .ok_or(ConvError::Overflow { what: "syn_fetches" })?;

        let mut out = Tensor3::zeros(self.conv.spec.out_channels, self.out_height, self.out_width)?;
        for (j, unit) in self.units.iter_mut().enumerate() {
            let dv = drive.data[j];
            if dv == 0.0 {
                self.ledger.neuron_updates_idle += 1;
            } else {
                self.ledger.neuron_updates_driven += 1;
            }
            unit.bump(dv);
            if unit.step(self.dt, 0.0) {
                out.data[j] = 1.0;
                self.ledger.spikes_out += 1;
            }
        }
        Ok(out)
    }

    /// Return every membrane to rest. The counters are **not** cleared; see
    /// [`SpikingConv2d::clear_counts`].
    pub fn reset(&mut self) {
        for u in &mut self.units {
            u.reset();
        }
    }

    /// Zero the ledger and the synaptic-operation counts, leaving the membranes alone.
    ///
    /// Separate from [`SpikingConv2d::reset`] because the two are wanted at different moments: the
    /// membranes are reset between samples, the counters between benchmarks.
    pub fn clear_counts(&mut self) {
        self.ledger = Ledger::default();
        self.ops = SynOps::default();
    }

    /// The surrogate derivative at each output unit's current membrane — the factor a backward pass
    /// would multiply that unit's gradient by.
    ///
    /// `v_th` is the firing threshold in volts and `scale` is **the number of volts that one
    /// dimensionless unit of [`crate::surrogate`] width corresponds to**. That second argument
    /// exists because the surrogate families are parameterised in units of a threshold set to `1`,
    /// while membranes here are in volts: pass a `scale` of `1.0` and every unit in a millivolt-
    /// scale network sits within `0.02` of threshold in the surrogate's frame, where every family
    /// is near its peak, and the dead-neuron problem the surrogate exists to expose disappears.
    /// The natural choice for [`crate::neuron::Lif`] is `v_th - v_rest`.
    ///
    /// Zero for a unit in its refractory period is **not** special-cased: the membrane is clamped at
    /// `v_reset` there and the surrogate reports what it reports at that offset, which is what a
    /// framework that ignores refractoriness in the backward pass also does. Whether that is right
    /// is an open question this module does not settle.
    ///
    /// # Errors
    ///
    /// [`ConvError::NonFinite`] for a non-finite `v_th`, or a `scale` that is not finite and
    /// strictly positive.
    pub fn surrogate_mask(&self, s: &dyn Surrogate, v_th: f64, scale: f64) -> Result<Tensor3, ConvError> {
        if !v_th.is_finite() {
            return Err(ConvError::NonFinite { what: "v_th", index: 0, value: v_th });
        }
        if !scale.is_finite() || scale <= 0.0 {
            return Err(ConvError::NonFinite { what: "scale", index: 0, value: scale });
        }
        let mut out = Tensor3::zeros(self.conv.spec.out_channels, self.out_height, self.out_width)?;
        for (j, unit) in self.units.iter().enumerate() {
            out.data[j] = s.backward((unit.potential() - v_th) / scale);
        }
        Ok(out)
    }

    /// Parameters and per-unit state, in bits, as [`crate::metrics::Footprint`].
    ///
    /// `state_per_unit` is how many scalars the neuron model carries — `2` for
    /// [`crate::neuron::Lif`] (membrane and refractory counter), `3` for
    /// [`crate::neuron::AdaptiveLif`], `2` for [`crate::neuron::Izhikevich`]. It is an argument
    /// because the [`crate::neuron::Neuron`] trait does not expose a state count, and inventing one
    /// would be a guess about someone else's model.
    ///
    /// # Errors
    ///
    /// [`ConvError::Overflow`] on a bit count past `u64` or a bit width outside `1..=64`.
    pub fn footprint(&self, param_bits: u32, state_per_unit: u64, state_bits: u32) -> Result<Footprint, ConvError> {
        let params = self.conv.n_params() as u64;
        let states = (self.units.len() as u64)
            .checked_mul(state_per_unit)
            .ok_or(ConvError::Overflow { what: "state scalars" })?;
        Ok(Footprint::new(params, param_bits, states, state_bits)?)
    }
}

/// How a residual block joins its branch to its shortcut — the choice Fang et al. `NeurIPS` 2021
/// showed is the difference between a spiking network that gets deeper and one that stops.
///
/// # What `SEW` changes
///
/// A **naive** spiking residual block follows the rectifier recipe literally: add the shortcut to
/// the residual branch's pre-activation, then pass the sum through the block's last spiking neuron.
/// `out = SN(F(x) + g * x)`. Two things go wrong.
///
/// *It cannot represent the identity.* For `out` to equal `x` the neuron must fire exactly where
/// the shortcut spikes — but the neuron resets after firing and holds a refractory period, so a run
/// of consecutive input spikes comes out as one spike followed by silence. No choice of `g` fixes
/// this, because the failure is in the state, not the gain.
/// `the_naive_residual_cannot_be_the_identity_even_with_a_tuned_gain` runs it: a shortcut spiking on
/// every timestep comes out at one third of its rate under
/// [`crate::neuron::Lif::default`] at `dt = 1 ms`.
///
/// *The shortcut gradient is attenuated per block.* Differentiating through the neuron gives
/// `d out / d x = sigma'(u - theta) * (g + d F / d x)`, so the shortcut path carries a factor
/// `sigma'(u - theta) * g` at **every** block and a stack of `L` blocks multiplies it `L` times.
/// With a surrogate whose peak is below `1 / g` this decays geometrically — which is precisely the
/// vanishing-gradient problem the residual connection was invented to remove.
///
/// `SEW` (spike-element-wise) moves the neuron **inside** the branch and merges two spike tensors:
/// `out = g(SN(F(x)), x)`. A silent branch now leaves `x` untouched under [`ResidualStyle::SewAdd`],
/// and the derivative along the shortcut is exactly `1` per block, for any surrogate.
/// [`identity_path_gain`] computes both.
///
/// # What it costs
///
/// [`ResidualStyle::SewAdd`]'s output is **not binary**. One block gives values in `{0, 1, 2}` and
/// `L` stacked blocks reach `L + 1`. Everything downstream is then a multiply-accumulate rather
/// than an accumulate, and a fabric that routes one-bit events cannot carry it without a graded
/// spike — which `Loihi 2` has and most do not. `AND` and `IAND` stay binary and pay elsewhere:
/// `AND` zeroes the shortcut whenever the branch is silent, so it cannot represent the identity at
/// all, and `IAND` inverts the branch. Fang et al. report `ADD` as the best-performing variant,
/// which is the one that leaves the binary regime — a result worth reporting alongside the accuracy
/// rather than under it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidualStyle {
    /// `out = SN(F(x) + g * x)`: the shortcut joins in volts, before the block's last neuron.
    ///
    /// The baseline the `SEW` paper argues against, implemented here so the argument can be run
    /// rather than quoted. `g` is [`ResidualBlock::shortcut_gain`], volts per shortcut spike.
    Naive,
    /// `out = SN(F(x)) + x`, element-wise. Represents the identity exactly; leaves `{0, 1}`.
    SewAdd,
    /// `out = SN(F(x)) * x`, element-wise. Stays binary; cannot represent the identity, because a
    /// silent branch zeroes the output.
    SewAnd,
    /// `out = (1 - SN(F(x))) * x`, element-wise. Stays binary and **does** represent the identity
    /// under a silent branch — the branch gates the shortcut off rather than on.
    SewIand,
}

impl ResidualStyle {
    /// The element-wise merge of one branch spike and one shortcut spike.
    ///
    /// `None` for [`ResidualStyle::Naive`], which has no element-wise merge at all: its shortcut
    /// joins the pre-activation in volts, before the neuron, so there is no pair of spikes to
    /// combine. Returning `branch + identity` there would be an answer to a question that was not
    /// asked.
    #[must_use]
    pub fn merge(self, branch: f64, identity: f64) -> Option<f64> {
        match self {
            Self::Naive => None,
            Self::SewAdd => Some(branch + identity),
            Self::SewAnd => Some(branch * identity),
            Self::SewIand => Some((1.0 - branch) * identity),
        }
    }

    /// Whether the merge can produce a value above `1.0` from binary inputs.
    ///
    /// `true` only for [`ResidualStyle::SewAdd`]. A `true` here means every layer downstream is
    /// charged multiply-accumulates rather than accumulates.
    #[must_use]
    pub fn leaves_binary_regime(self) -> bool {
        matches!(self, Self::SewAdd)
    }
}

/// The factor a gradient picks up along the **shortcut** of `depth` stacked residual blocks whose
/// residual branches are silent.
///
/// This is the closed form of the argument in [`ResidualStyle`]'s doc, with the branch contribution
/// set to zero so that only the identity path remains:
///
/// ```text
/// SewAdd   out = 0 + x            d out / d x = 1              -> 1
/// SewIand  out = (1 - 0) * x      d out / d x = 1              -> 1
/// SewAnd   out = 0 * x            d out / d x = 0              -> 0
/// Naive    out = Theta(g x - th)  d out / d x = sigma'(off)*g  -> (sigma'(off) * g)^depth
/// ```
///
/// `offset` is the membrane's distance from threshold in the surrogate's dimensionless frame — see
/// [`SpikingConv2d::surrogate_mask`] on why that frame is not volts. `shortcut_gain` is the naive
/// block's `g`, dimensionless in the same frame.
///
/// `None` for a non-finite `offset` or `shortcut_gain`, or a `depth` past `i32`.
#[must_use]
pub fn identity_path_gain(
    style: ResidualStyle,
    surrogate: &dyn Surrogate,
    offset: f64,
    shortcut_gain: f64,
    depth: u32,
) -> Option<f64> {
    if !offset.is_finite() || !shortcut_gain.is_finite() {
        return None;
    }
    let d = i32::try_from(depth).ok()?;
    Some(match style {
        ResidualStyle::SewAdd | ResidualStyle::SewIand => 1.0,
        ResidualStyle::SewAnd => {
            if depth == 0 {
                1.0
            } else {
                0.0
            }
        }
        ResidualStyle::Naive => (surrogate.backward(offset) * shortcut_gain).powi(d),
    })
}

/// A two-convolution residual block, in any of the four styles.
///
/// Shape-preserving by construction: both convolutions must map `(channels, height, width)` to
/// itself, which [`ResidualBlock::new`] checks and refuses. A block that changed shape would need a
/// projection on the shortcut, which is a third convolution and a fifth design decision; this
/// review implemented the identity-shortcut case, which is the one the `SEW` argument is about.
#[derive(Debug, Clone, PartialEq)]
pub struct ResidualBlock<N: Neuron> {
    /// First convolution with its spiking neurons — `SN(conv1(x))` in every style.
    pub first: SpikingConv2d<N>,
    /// Second convolution's kernel. Its neurons are [`ResidualBlock::units`], held here rather than
    /// in a [`SpikingConv2d`] because [`ResidualStyle::Naive`] must add the shortcut **before**
    /// stepping them and the merge styles must step them before merging.
    pub second: Conv2d,
    /// Neurons after the second convolution, in [`Tensor3`] layout.
    pub units: Vec<N>,
    /// How the branch and the shortcut are joined.
    pub style: ResidualStyle,
    /// Volts of membrane displacement per shortcut spike, used only by [`ResidualStyle::Naive`].
    ///
    /// For a naive block to fire on a lone shortcut spike this must exceed
    /// `(v_th - v_rest) / exp(-dt / tau_m)` — the decay applies after the bump, so the bare
    /// threshold distance is not enough. The other three styles ignore it entirely.
    pub shortcut_gain: f64,
    /// Timestep, seconds.
    pub dt: f64,
    /// Channels in and out, which are equal.
    pub channels: usize,
    /// Rows in and out.
    pub height: usize,
    /// Columns in and out.
    pub width: usize,
    /// Synaptic operations of the **second** convolution, accumulated. The first convolution's are
    /// in `first.ops`.
    pub ops: SynOps,
    /// Exact counters for the **second** stage — its accumulates, its fetches, its neuron updates
    /// and its spikes — accumulated across every [`ResidualBlock::step`]. The first stage's are in
    /// `first.ledger`; the block's total is the sum of the two.
    ///
    /// ⛔ The first version had no ledger here at all. The second stage's units and deliveries
    /// touched no counter, so a `SEW-ResNet` priced from `first.ledger` undercounted neuron
    /// updates and spikes by exactly half and synaptic operations by the whole second convolution.
    pub ledger: Ledger,
}

impl<N: Neuron> ResidualBlock<N> {
    /// Build a block, checking that both convolutions preserve `(channels, height, width)`.
    ///
    /// # Errors
    ///
    /// [`ConvError::ShapeMismatch`] when either convolution changes the shape or the channel counts
    /// do not line up, [`ConvError::NonFinite`] for a non-finite `dt` or `shortcut_gain` or a `dt`
    /// that is not strictly positive, plus anything [`SpikingConv2d::new`] refuses.
    pub fn new(
        first: Conv2d,
        second: Conv2d,
        height: usize,
        width: usize,
        proto: N,
        dt: f64,
        style: ResidualStyle,
        shortcut_gain: f64,
    ) -> Result<Self, ConvError> {
        if !shortcut_gain.is_finite() {
            return Err(ConvError::NonFinite { what: "shortcut_gain", index: 0, value: shortcut_gain });
        }
        let channels = first.spec.in_channels;
        let a = first.spec.require_out_shape(height, width)?;
        if (first.spec.out_channels, a.0, a.1) != (channels, height, width) {
            return Err(ConvError::ShapeMismatch {
                what: "residual first convolution",
                a: (first.spec.out_channels, a.0, a.1),
                b: (channels, height, width),
            });
        }
        let b = second.spec.require_out_shape(height, width)?;
        if (second.spec.in_channels, second.spec.out_channels, b.0, b.1)
            != (channels, channels, height, width)
        {
            return Err(ConvError::ShapeMismatch {
                what: "residual second convolution",
                a: (second.spec.out_channels, b.0, b.1),
                b: (channels, height, width),
            });
        }
        let first = SpikingConv2d::new(first, height, width, proto.clone(), dt)?;
        let n = dims_len(channels, height, width)?;
        Ok(Self {
            first,
            second,
            units: vec![proto; n],
            style,
            shortcut_gain,
            dt,
            channels,
            height,
            width,
            ops: SynOps::default(),
            ledger: Ledger::default(),
        })
    }

    /// Advance the block one timestep.
    ///
    /// `kind` charges the **input** to the first convolution, exactly as in
    /// [`SpikingConv2d::step`]: pass [`ActivationKind::RealValued`] when the block is downstream of
    /// a [`ResidualStyle::SewAdd`] block, whose output is not binary.
    ///
    /// # Errors
    ///
    /// [`ConvError::ShapeMismatch`] for the wrong input shape, [`ConvError::NotBinary`] under
    /// [`ActivationKind::Spiking`] with a graded input, [`ConvError::Overflow`] on a count past
    /// `u64`.
    pub fn step(&mut self, x: &Tensor3, kind: ActivationKind) -> Result<Tensor3, ConvError> {
        if x.shape() != (self.channels, self.height, self.width) {
            return Err(ConvError::ShapeMismatch {
                what: "residual block input",
                a: x.shape(),
                b: (self.channels, self.height, self.width),
            });
        }
        let s1 = self.first.step(x, kind)?;
        check_finite(&self.second.weights, "conv weights")?;
        check_finite(&self.second.bias, "conv bias")?;
        // The first layer's output is always binary — it came from a spiking neuron — so the second
        // convolution is charged accumulates regardless of what the block's own input was.
        let (drive, dense, effective) = self.second.convolve(&s1)?;
        self.ops.add(SynOps { dense, effective_macs: 0, effective_acs: effective })?;
        self.ledger.syn_ops =
            self.ledger.syn_ops.checked_add(effective).ok_or(ConvError::Overflow { what: "syn_ops" })?;
        self.ledger.syn_fetches = self
            .ledger
            .syn_fetches
            .checked_add(effective)
            .ok_or(ConvError::Overflow { what: "syn_fetches" })?;
        // Same guard as `SpikingConv2d::step`: `second` is a public field.
        if drive.data.len() != self.units.len() {
            return Err(ConvError::ShapeMismatch {
                what: "residual second convolution units",
                a: drive.shape(),
                b: (self.channels, self.height, self.width),
            });
        }

        let mut out = Tensor3::zeros(self.channels, self.height, self.width)?;
        for (j, unit) in self.units.iter_mut().enumerate() {
            let shortcut = x.data[j];
            let dv = match self.style {
                ResidualStyle::Naive => drive.data[j] + self.shortcut_gain * shortcut,
                ResidualStyle::SewAdd | ResidualStyle::SewAnd | ResidualStyle::SewIand => drive.data[j],
            };
            if dv == 0.0 {
                self.ledger.neuron_updates_idle += 1;
            } else {
                self.ledger.neuron_updates_driven += 1;
            }
            unit.bump(dv);
            let fired = unit.step(self.dt, 0.0);
            if fired {
                self.ledger.spikes_out += 1;
            }
            match self.style {
                ResidualStyle::Naive => {
                    if fired {
                        out.data[j] = 1.0;
                    }
                }
                // The three merge styles are listed rather than wildcarded, so a style added
                // later is a compile error here instead of falling through to the branch output.
                // `merge` returns `None` only for `Naive`, which the arm above already handled.
                ResidualStyle::SewAdd | ResidualStyle::SewAnd | ResidualStyle::SewIand => {
                    let branch = if fired { 1.0 } else { 0.0 };
                    out.data[j] = self.style.merge(branch, shortcut).unwrap_or(branch);
                }
            }
        }
        Ok(out)
    }

    /// Zero both stages' counters, leaving every membrane where it is.
    pub fn clear_counts(&mut self) {
        self.first.clear_counts();
        self.ops = SynOps::default();
        self.ledger = Ledger::default();
    }

    /// Return every membrane in the block to rest.
    pub fn reset(&mut self) {
        self.first.reset();
        for u in &mut self.units {
            u.reset();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        avg_pool, channel_moments, identity_path_gain, max_pool_values, out_dim, Conv2d, Conv2dSpec,
        ConvError, Init, MaxPolicy, PoolSpec, ResidualBlock, ResidualStyle, SpikingConv2d,
        SpikingMaxPool, TdBn, Tensor3, MAX_TAP_SWEEP,
    };
    use crate::metrics::{ActivationKind, SynOps};
    use crate::neuron::{Lif, Neuron};
    use crate::surrogate::{ArcTan, FastSigmoid, SigmoidDeriv, Surrogate};

    fn t(c: usize, h: usize, w: usize, d: &[f64]) -> Tensor3 {
        Tensor3::new(c, h, w, d.to_vec()).expect("test tensor is well formed")
    }

    fn conv1(k: &[f64], kh: usize, kw: usize, pad: usize, stride: usize, dil: usize) -> Conv2d {
        let mut spec = Conv2dSpec::new(1, 1, kh, kw).expect("valid spec");
        spec.pad_h = pad;
        spec.pad_w = pad;
        spec.stride_h = stride;
        spec.stride_w = stride;
        spec.dilation_h = dil;
        spec.dilation_w = dil;
        Conv2d::new(spec, k.to_vec(), vec![0.0]).expect("valid kernel")
    }

    /// (a) A hand-computed convolution, at ZERO tolerance, with an ASYMMETRIC kernel.
    ///
    /// The kernel `[[1,2],[3,4]]` on `[[1..9]]` gives four distinct numbers, so a transposed
    /// kernel, a flipped kernel or a swapped row/column index all produce a different answer. The
    /// first draft of this test used `[[1,0],[0,-1]]`, whose output is the constant `-4` at every
    /// position — it passed under a transposed kernel and under a swapped index, and would have
    /// gone on passing.
    #[test]
    fn the_convolution_is_cross_correlation_and_the_kernel_is_not_flipped() {
        let input = t(1, 3, 3, &[1., 2., 3., 4., 5., 6., 7., 8., 9.]);
        let c = conv1(&[1., 2., 3., 4.], 2, 2, 0, 1, 1);
        let out = c.forward(&input).expect("fits");
        assert_eq!(out.shape(), (1, 2, 2));
        assert_eq!(out.data, vec![37.0, 47.0, 67.0, 77.0]);

        // The flipped (true-convolution) kernel gives a DIFFERENT answer, which is what makes the
        // assertion above a statement about the convention rather than about the arithmetic.
        let flipped = conv1(&[4., 3., 2., 1.], 2, 2, 0, 1, 1);
        let f = flipped.forward(&input).expect("fits");
        assert_ne!(f.data, out.data);
        assert_eq!(f.data, vec![23.0, 33.0, 53.0, 63.0]);
    }

    /// (a) Padding, hand-computed on a padded `2x2`, at zero tolerance.
    ///
    /// The kernel `[[1,2],[3,5]]` is deliberately NOT the input `[[1,2],[3,4]]`: with those two
    /// equal the output comes out invariant under a 180-degree rotation, and a test asserting it
    /// would pass under an implementation that indexed the padded window backwards. The `5` breaks
    /// that symmetry, and the nine values below are all reachable only from the intended indexing.
    #[test]
    fn a_padded_convolution_matches_the_hand_computed_answer() {
        let input = t(1, 2, 2, &[1., 2., 3., 4.]);
        let c = conv1(&[1., 2., 3., 5.], 2, 2, 1, 1, 1);
        let out = c.forward(&input).expect("fits");
        assert_eq!(out.shape(), (1, 3, 3));
        // Padded input is [[0,0,0,0],[0,1,2,0],[0,3,4,0],[0,0,0,0]]; the nine windows give
        // 5, 13, 6 / 17, 34, 14 / 6, 11, 4 against weights w00=1 w01=2 w10=3 w11=5.
        assert_eq!(out.data, vec![5.0, 13.0, 6.0, 17.0, 34.0, 14.0, 6.0, 11.0, 4.0]);
        let mut rev = out.data.clone();
        rev.reverse();
        assert_ne!(rev, out.data, "the answer must not be rotation-symmetric, or it proves nothing");
    }

    /// (a) Stride and dilation, hand-computed on `1..25`.
    #[test]
    fn strided_and_dilated_convolutions_match_the_hand_computed_answer() {
        let d: Vec<f64> = (1..=25).map(f64::from).collect();
        let input = t(1, 5, 5, &d);

        // Stride 2 with the trace kernel: windows at rows {0,2} and columns {0,2}.
        let s = conv1(&[1., 0., 0., 1.], 2, 2, 0, 2, 1);
        let o = s.forward(&input).expect("fits");
        assert_eq!(o.shape(), (1, 2, 2));
        assert_eq!(o.data, vec![8.0, 12.0, 28.0, 32.0]);

        // Dilation 2: taps at (y,x), (y,x+2), (y+2,x), (y+2,x+2).
        let dl = conv1(&[1., 2., 3., 4.], 2, 2, 0, 1, 2);
        let o = dl.forward(&input).expect("fits");
        assert_eq!(o.shape(), (1, 3, 3));
        assert_eq!(o.data, vec![92.0, 102.0, 112.0, 142.0, 152.0, 162.0, 192.0, 202.0, 212.0]);
    }

    /// (a) Channels and bias, hand-computed through a `1x1` kernel that is a pure channel mix.
    ///
    /// A `1x1` kernel removes the spatial arithmetic entirely, so what remains is only the channel
    /// indexing and the bias — which is where a `[oc][ic]` / `[ic][oc]` transposition lives.
    #[test]
    fn channel_mixing_and_bias_match_the_hand_computed_answer() {
        let input = t(2, 2, 2, &[1., 2., 3., 4., 5., 6., 7., 8.]);
        let spec = Conv2dSpec::new(2, 2, 1, 1).expect("valid");
        let c = Conv2d::new(spec, vec![1., 10., -1., 2.], vec![100., 0.]).expect("valid");
        let out = c.forward(&input).expect("fits");
        assert_eq!(out.shape(), (2, 2, 2));
        assert_eq!(out.data, vec![151., 162., 173., 184., 9., 10., 11., 12.]);
    }

    /// (b) The shape formula against an independent COUNT of valid window positions.
    ///
    /// The count is not the formula written twice: it enumerates the start positions `0, s, 2s, ...`
    /// and asks whether the last tap is still inside the padded input. If the formula had a `+1`
    /// too many, or floored the wrong quantity, this disagrees.
    #[test]
    fn the_shape_formula_agrees_with_a_count_of_valid_window_positions() {
        let mut checked = 0;
        let mut impossible = 0;
        for n in 1..=9usize {
            for k in 1..=5usize {
                for p in 0..=3usize {
                    for s in 1..=3usize {
                        for d in 1..=3usize {
                            let padded = n + 2 * p;
                            let extent = d * (k - 1) + 1;
                            let mut count = 0usize;
                            let mut start = 0usize;
                            while start + extent <= padded {
                                count += 1;
                                start += s;
                            }
                            let got = out_dim(n, k, p, s, d);
                            if count == 0 {
                                assert_eq!(got, None, "n={n} k={k} p={p} s={s} d={d}");
                                impossible += 1;
                            } else {
                                assert_eq!(got, Some(count), "n={n} k={k} p={p} s={s} d={d}");
                                checked += 1;
                            }
                        }
                    }
                }
            }
        }
        // Both branches must actually be exercised, or the sweep is pinned to the side where it
        // cannot break.
        assert!(checked > 500, "only {checked} feasible cases");
        assert!(impossible > 20, "only {impossible} infeasible cases");
    }

    /// (b) An impossible shape is REFUSED, at every entry point, rather than truncated to zero.
    #[test]
    fn an_impossible_shape_is_refused_rather_than_truncated() {
        assert_eq!(out_dim(3, 5, 0, 1, 1), None);
        assert_eq!(out_dim(3, 3, 0, 1, 2), None); // dilated extent 5 > 3
        assert_eq!(out_dim(3, 2, 0, 0, 1), None); // zero stride
        assert_eq!(out_dim(3, 0, 0, 1, 1), None); // zero kernel

        let input = t(1, 3, 3, &[0.; 9]);
        let c = conv1(&[0.; 25], 5, 5, 0, 1, 1);
        match c.forward(&input) {
            Err(ConvError::ImpossibleShape { axis, n, kernel, .. }) => {
                assert_eq!((axis, n, kernel), ("height", 3, 5));
            }
            other => panic!("expected a refusal, got {other:?}"),
        }
        let p = PoolSpec { kernel_h: 4, kernel_w: 2, stride_h: 1, stride_w: 1, pad_h: 0, pad_w: 0 };
        assert!(matches!(
            avg_pool(&input, &p, false),
            Err(ConvError::ImpossibleShape { axis: "height", .. })
        ));
    }

    /// The separable in-bounds tap count against the convolution loop's own count, over a sweep.
    #[test]
    fn the_separable_tap_count_agrees_with_the_loop() {
        let mut seen_gap = 0;
        for n in 2..=6usize {
            for k in 1..=3usize {
                for p in 0..=2usize {
                    for s in 1..=2usize {
                        for d in 1..=2usize {
                            if out_dim(n, k, p, s, d).is_none() {
                                continue;
                            }
                            let c = conv1(&vec![1.0; k * k], k, k, p, s, d);
                            let input =
                                Tensor3::filled(1, n, n, 1.0).expect("valid");
                            let (_, dense, _) = c.convolve(&input).expect("fits");
                            assert_eq!(
                                Some(dense),
                                c.in_bounds_taps(n, n),
                                "n={n} k={k} p={p} s={s} d={d}"
                            );
                            let padded = c.padded_taps(n, n).expect("fits");
                            assert!(padded >= dense);
                            if padded > dense {
                                seen_gap += 1;
                            }
                        }
                    }
                }
            }
        }
        // The padded and in-bounds counts must actually DIFFER somewhere, or the assertion above is
        // comparing a number against itself.
        assert!(seen_gap > 20, "padded and in-bounds counts never differed ({seen_gap})");
    }

    /// (e) Synaptic operations against a hand calculation, with both kinds of zero present.
    ///
    /// `3x3` binary input on the diagonal, `2x2` kernel, two output channels — one dense, one with
    /// two zero weights. Dense taps: 4 positions * 2 channels * 4 taps = 32. Effective, counted by
    /// hand position by position: (2+2) + (1+0) + (1+0) + (2+2) = 10.
    #[test]
    fn the_synaptic_operation_count_matches_a_hand_calculation() {
        let input = t(1, 3, 3, &[1., 0., 0., 0., 1., 0., 0., 0., 1.]);
        let spec = Conv2dSpec::new(1, 2, 2, 2).expect("valid");
        let c = Conv2d::new(spec, vec![1., 1., 1., 1., 1., 0., 0., 1.], vec![0., 0.]).expect("valid");
        let ops = c.synops(&input, ActivationKind::Spiking).expect("fits");
        assert_eq!(ops.dense, 32);
        assert_eq!(ops.effective_acs, 10);
        assert_eq!(ops.effective_macs, 0);
        assert_eq!(ops.effective_total(), Some(10));

        // The same input charged as real-valued moves every operation to the MAC column, and the
        // dense count does not move. That is the whole point of keeping the two apart.
        let ops = c.synops(&input, ActivationKind::RealValued).expect("fits");
        assert_eq!((ops.dense, ops.effective_acs, ops.effective_macs), (32, 0, 10));

        // A non-binary activation is refused under the spiking charge rather than rounded.
        let graded = t(1, 3, 3, &[2., 0., 0., 0., 1., 0., 0., 0., 1.]);
        assert!(matches!(
            c.synops(&graded, ActivationKind::Spiking),
            Err(ConvError::NotBinary { index: 0, .. })
        ));
    }

    /// (e) The padded tap count overstates the connections, by exactly the hand-computed amount.
    #[test]
    fn the_padded_tap_count_overstates_the_connections_by_a_known_amount() {
        let c = conv1(&[1.; 9], 3, 3, 1, 1, 1);
        // 9 output positions * 9 taps = 81 padded; in-bounds factorises as (2+3+2)^2 = 49.
        assert_eq!(c.padded_taps(3, 3), Some(81));
        assert_eq!(c.in_bounds_taps(3, 3), Some(49));
    }

    /// (c) Average pooling of a constant is EXACTLY that constant, bit for bit, across a sweep.
    ///
    /// Dyadic constants only, and the module doc says why: `m * c` is then exact for every partial
    /// sum and `m * c / m` is exactly `c`. A wrong divisor, an off-by-one window bound or a
    /// mis-indexed channel all break this at the first element.
    #[test]
    fn average_pooling_a_constant_returns_exactly_that_constant() {
        for &c in &[0.25_f64, 0.5, 0.75, 1.0, 0.125, 2.0, 3.0, -0.5] {
            for kh in 1..=3usize {
                for kw in 1..=3usize {
                    for stride in 1..=2usize {
                        let spec = PoolSpec {
                            kernel_h: kh,
                            kernel_w: kw,
                            stride_h: stride,
                            stride_w: stride,
                            pad_h: 0,
                            pad_w: 0,
                        };
                        let input = Tensor3::filled(2, 4, 4, c).expect("valid");
                        let out = avg_pool(&input, &spec, true).expect("fits");
                        for v in &out.data {
                            assert_eq!(*v, c, "c={c} kh={kh} kw={kw} s={stride}");
                        }
                    }
                }
            }
        }
    }

    /// (c) With padding, the two divisor conventions give DIFFERENT answers, and only one of them
    /// preserves a constant. A test that exercised just one would pass under an implementation that
    /// ignored the flag.
    #[test]
    fn the_padding_divisor_convention_changes_the_border_and_only_one_preserves_a_constant() {
        let spec = PoolSpec { kernel_h: 3, kernel_w: 3, stride_h: 1, stride_w: 1, pad_h: 1, pad_w: 1 };
        let input = Tensor3::filled(1, 3, 3, 1.0).expect("valid");

        let excl = avg_pool(&input, &spec, false).expect("fits");
        for v in &excl.data {
            assert_eq!(*v, 1.0);
        }

        let incl = avg_pool(&input, &spec, true).expect("fits");
        // Corner window sees 4 of 9 real elements; edge 6 of 9; centre 9 of 9.
        assert_eq!(incl.data[0], 4.0 / 9.0);
        assert_eq!(incl.data[1], 6.0 / 9.0);
        assert_eq!(incl.data[4], 1.0);

        // A window wholly on padding has no mean, and is refused rather than returned as zero.
        // ALL THREE pooling paths refuse it: this was found by mutation, and before the two lines
        // below, replacing the maximum's refusal with `unwrap_or(0.0)` left every test green.
        let far = PoolSpec { kernel_h: 2, kernel_w: 2, stride_h: 1, stride_w: 1, pad_h: 3, pad_w: 3 };
        assert!(matches!(avg_pool(&input, &far, false), Err(ConvError::EmptyWindow { y: 0, x: 0 })));
        assert!(matches!(max_pool_values(&input, &far), Err(ConvError::EmptyWindow { y: 0, x: 0 })));
        let mut p = SpikingMaxPool::new(far, MaxPolicy::Instant, 1, 3, 3).expect("valid");
        assert!(matches!(
            p.step(&Tensor3::zeros(1, 3, 3).expect("valid")),
            Err(ConvError::EmptyWindow { y: 0, x: 0 })
        ));
        // And a zero is a real answer everywhere else: the same maximum over a window of genuine
        // zeros returns 0.0 rather than refusing, which is the distinction the refusal preserves.
        let near = PoolSpec { kernel_h: 2, kernel_w: 2, stride_h: 1, stride_w: 1, pad_h: 0, pad_w: 0 };
        let zeros = Tensor3::zeros(1, 3, 3).expect("valid");
        assert_eq!(max_pool_values(&zeros, &near).expect("fits").data, vec![0.0; 4]);
    }

    /// Average pooling of a non-constant map, hand-computed.
    ///
    /// **Two channels, carrying different data**: the constant-input sweep above cannot see a
    /// channel-index mutation, because every channel holds the same number there.
    #[test]
    fn average_pooling_matches_the_hand_computed_mean() {
        let mut d: Vec<f64> = (1..=16).map(f64::from).collect();
        d.extend((1..=16).rev().map(f64::from));
        let input = t(2, 4, 4, &d);
        let spec = PoolSpec::new(2, 2).expect("valid");
        let out = avg_pool(&input, &spec, true).expect("fits");
        // Channel 0: (1+2+5+6)/4 = 3.5, (3+4+7+8)/4 = 5.5, 11.5, 13.5.
        // Channel 1 is the reverse: (16+15+12+11)/4 = 13.5, 11.5, 5.5, 3.5.
        assert_eq!(out.data, vec![3.5, 5.5, 11.5, 13.5, 13.5, 11.5, 5.5, 3.5]);
        let mx = max_pool_values(&input, &spec).expect("fits");
        assert_eq!(mx.data, vec![6.0, 8.0, 14.0, 16.0, 16.0, 14.0, 8.0, 6.0]);
    }

    /// Instant max pooling over spikes is exactly a logical OR — and that is the honest description
    /// of what it computes, not an approximation of a maximum.
    #[test]
    fn instant_max_pooling_over_spikes_is_a_logical_or() {
        let spec = PoolSpec::new(2, 2).expect("valid");
        let mut p = SpikingMaxPool::new(spec, MaxPolicy::Instant, 1, 2, 2).expect("valid");
        for (bits, want) in [
            ([0., 0., 0., 0.], 0.0),
            ([1., 0., 0., 0.], 1.0),
            ([0., 0., 0., 1.], 1.0),
            ([1., 1., 1., 1.], 1.0),
        ] {
            let out = p.step(&t(1, 2, 2, &bits)).expect("fits");
            assert_eq!(out.data, vec![want], "bits {bits:?}");
        }
        // Rows two and four of that table are the loss the MaxPolicy doc names: one firing unit and
        // four firing units produce the same output, so pooling raises the output rate above every
        // input unit's. And a graded activation is refused rather than thresholded.
        assert!(matches!(p.step(&t(1, 2, 2, &[0.5, 0., 0., 0.])), Err(ConvError::NotBinary { .. })));
    }

    /// Rate-gated max pooling DROPS the newly active unit, for exactly as long as the old leader's
    /// lead. The published cost of Rueckauer et al.'s gate, measured.
    #[test]
    fn rate_gated_max_pooling_drops_the_newly_active_unit() {
        let spec = PoolSpec::new(2, 2).expect("valid");
        let mut gated = SpikingMaxPool::new(spec, MaxPolicy::RateGated, 1, 2, 2).expect("valid");
        let mut or = SpikingMaxPool::new(spec, MaxPolicy::Instant, 1, 2, 2).expect("valid");

        // Unit (0,0) fires on ticks 0-2; unit (1,1) fires on ticks 3-5.
        let frames = [
            [1., 0., 0., 0.],
            [1., 0., 0., 0.],
            [1., 0., 0., 0.],
            [0., 0., 0., 1.],
            [0., 0., 0., 1.],
            [0., 0., 0., 1.],
        ];
        let mut g = Vec::new();
        let mut o = Vec::new();
        for f in &frames {
            g.push(gated.step(&t(1, 2, 2, f)).expect("fits").data[0]);
            o.push(or.step(&t(1, 2, 2, f)).expect("fits").data[0]);
        }
        assert_eq!(o, vec![1., 1., 1., 1., 1., 1.], "the OR sees every spike");
        assert_eq!(g, vec![1., 1., 1., 0., 0., 0.], "the gate is still pointed at the old leader");

        // Reset forgets the lead. It does NOT rescue the dropped spikes: the tie at zero counts
        // goes to the lowest flat index, which is still the silent (0,0), so the sixth frame still
        // does not pass. Only a different policy changes that.
        gated.reset();
        assert!(gated.counts.iter().all(|c| *c == 0));
        assert_eq!(gated.step(&t(1, 2, 2, &frames[5])).expect("fits").data[0], 0.0);
    }

    /// The rate gate is STRICTLY CAUSAL: the counts it reads exclude the current timestep.
    ///
    /// `rate_gated_max_pooling_drops_the_newly_active_unit` cannot see this — moving the count
    /// update before the gate gives the identical train there. It takes a tie that the current
    /// timestep would break: two units level at one spike each, and only the second firing now.
    /// Causally the tie goes to the lower index and the output is silent; update-first would hand
    /// the gate to the unit that just fired and emit a spike.
    #[test]
    fn the_rate_gate_is_strictly_causal() {
        let spec = PoolSpec::new(2, 2).expect("valid");
        let mut gated = SpikingMaxPool::new(spec, MaxPolicy::RateGated, 1, 2, 2).expect("valid");
        // Tick 0: both (0,0) and (1,1) fire, leaving the counts level at 1.
        assert_eq!(gated.step(&t(1, 2, 2, &[1., 0., 0., 1.])).expect("fits").data[0], 1.0);
        assert_eq!(gated.counts, vec![1, 0, 0, 1]);
        // Tick 1: only (1,1) fires. The gate still reads the level counts and picks (0,0).
        assert_eq!(gated.step(&t(1, 2, 2, &[0., 0., 0., 1.])).expect("fits").data[0], 0.0);
        assert_eq!(gated.counts, vec![1, 0, 0, 2]);
        // Tick 2: (1,1) is now ahead, so the gate moves and its spike comes through.
        assert_eq!(gated.step(&t(1, 2, 2, &[0., 0., 0., 1.])).expect("fits").data[0], 1.0);
    }

    /// (f) tdBN leaves the per-channel statistic EXACTLY at its target, at zero tolerance.
    ///
    /// Each channel is a batch of values `m_c +/- d_c` with dyadic `m_c` and `d_c`, so the mean is
    /// exactly `m_c`, the population variance is exactly `d_c^2`, and with `eps = 0` the normalised
    /// values are exactly `+/- 1`. The output is therefore exactly `+/- alpha * v_th`, elementwise,
    /// and the channels are given DIFFERENT means and spreads so that a layer using pooled
    /// statistics instead of per-channel ones fails.
    #[test]
    fn tdbn_puts_the_channel_statistic_exactly_on_its_target() {
        let mut bn = TdBn::new(2, 0.5).expect("valid");
        bn.eps = 0.0;
        assert_eq!(bn.target_sigma(), 0.5);

        // Channel 0: mean 4, spread 0.5. Channel 1: mean -8, spread 2.
        let frames = vec![
            t(2, 1, 2, &[4.5, 3.5, -6.0, -10.0]),
            t(2, 1, 2, &[3.5, 4.5, -10.0, -6.0]),
        ];
        let out = bn.train_forward(&frames).expect("valid");
        assert_eq!(out[0].data, vec![0.5, -0.5, 0.5, -0.5]);
        assert_eq!(out[1].data, vec![-0.5, 0.5, -0.5, 0.5]);

        // The statistic of the OUTPUT, computed inline here rather than through the module's own
        // moment function, so the check does not share code with the thing it checks.
        for c in 0..2 {
            let vals: Vec<f64> = out.iter().flat_map(|o| [o.data[c * 2], o.data[c * 2 + 1]]).collect();
            let mean = vals.iter().sum::<f64>() / vals.len() as f64;
            let var = vals.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / vals.len() as f64;
            assert_eq!(mean, 0.0, "channel {c} mean");
            assert_eq!(var, 0.25, "channel {c} variance is exactly (alpha * v_th)^2");
        }

        // alpha moves the target and nothing else.
        //
        // ⛔ AGAINST A LITERAL, not against `residual_alpha()` itself. The first version computed
        // `want` from the function under test, so `42.0` passed — and so did `sqrt(2)`, the
        // reciprocal, which is the one wrong answer a reader is likely to write.
        const ONE_OVER_ROOT_TWO: f64 = 0.707_106_781_186_547_5;
        assert!((TdBn::residual_alpha() - ONE_OVER_ROOT_TWO).abs() < 1e-16);
        assert!((TdBn::residual_alpha() * TdBn::residual_alpha() - 0.5).abs() < 2e-16);
        bn.alpha = TdBn::residual_alpha();
        let out = bn.train_forward(&frames).expect("valid");
        let want = 0.5 * ONE_OVER_ROOT_TWO;
        assert!((out[0].data[0] - want).abs() < 1e-15);
    }

    /// `eps`, `gain` and `shift` each move the answer by the amount their expression says.
    ///
    /// Every assertion above runs at `eps = 0`, `gain = 1`, `shift = 0` — the three values at which
    /// those terms disappear from the arithmetic. Deleting any of the three from the implementation
    /// would leave all of them green.
    #[test]
    fn the_epsilon_gain_and_shift_terms_each_move_the_answer() {
        // Population variance of -3,-1,1,3 is exactly 5, and the batch mean is exactly 0.
        let frames = vec![t(1, 1, 4, &[-3., -1., 1., 3.])];

        let mut bare = TdBn::new(1, 1.0).expect("valid");
        bare.eps = 0.0;
        let a = bare.train_forward(&frames).expect("valid");
        assert!((a[0].data[3] - 3.0 / 5.0_f64.sqrt()).abs() < 1e-15);

        // eps = 5 doubles the quantity under the root, so every output shrinks by exactly sqrt(2).
        let mut with_eps = TdBn::new(1, 1.0).expect("valid");
        with_eps.eps = 5.0;
        let b = with_eps.train_forward(&frames).expect("valid");
        assert!((b[0].data[3] - 3.0 / 10.0_f64.sqrt()).abs() < 1e-15);
        assert!((a[0].data[3] / b[0].data[3] - 2.0_f64.sqrt()).abs() < 1e-14);

        // gain scales and shift translates, in that order.
        let mut affine = TdBn::new(1, 1.0).expect("valid");
        affine.eps = 0.0;
        affine.gain[0] = 3.0;
        affine.shift[0] = 7.0;
        let c = affine.train_forward(&frames).expect("valid");
        assert!((c[0].data[3] - (3.0 * 3.0 / 5.0_f64.sqrt() + 7.0)).abs() < 1e-14);
        // The mean of the output is now the shift, not zero — which is what a shift is for.
        let mean = c[0].data.iter().sum::<f64>() / 4.0;
        assert!((mean - 7.0).abs() < 1e-14);
    }

    /// The running statistics are an exponential average, not a replacement.
    ///
    /// `the_running_variance_is_the_unbiased_estimate` runs at `momentum = 1.0`, where the
    /// `(1 - momentum) * old` term is multiplied by zero and cannot be seen.
    #[test]
    fn the_running_statistics_blend_rather_than_replace() {
        let mut bn = TdBn::new(1, 1.0).expect("valid");
        bn.eps = 0.0;
        bn.momentum = 0.5;
        assert_eq!((bn.running_mean[0], bn.running_var[0]), (0.0, 1.0));
        // Mean 2, population variance 5, unbiased variance 20/3.
        let frames = vec![t(1, 1, 4, &[-1., 1., 3., 5.])];
        bn.train_forward(&frames).expect("valid");
        assert!((bn.running_mean[0] - 1.0).abs() < 1e-15, "got {}", bn.running_mean[0]);
        let want = 0.5 * 1.0 + 0.5 * (20.0 / 3.0);
        assert!((bn.running_var[0] - want).abs() < 1e-14, "got {}", bn.running_var[0]);
    }

    /// tdBN's running buffer carries Bessel's correction, and `eval_forward` uses it.
    #[test]
    fn the_running_variance_is_the_unbiased_estimate() {
        let mut bn = TdBn::new(1, 1.0).expect("valid");
        bn.eps = 0.0;
        bn.momentum = 1.0; // replace, so the buffer is exactly this batch's statistic
        // Four values: -3, -1, 1, 3. Mean 0, population variance 5, unbiased 20/3.
        let frames = vec![t(1, 1, 4, &[-3., -1., 1., 3.])];
        let (mean, var, count) = channel_moments(&frames, 1).expect("valid");
        assert_eq!((mean[0], var[0], count), (0.0, 5.0, 4));
        bn.train_forward(&frames).expect("valid");
        assert_eq!(bn.running_mean[0], 0.0);
        assert!((bn.running_var[0] - 20.0 / 3.0).abs() < 1e-15, "got {}", bn.running_var[0]);

        // eval_forward divides by sqrt(running_var), which is NOT the batch's own sigma — so an
        // implementation that reused the batch statistic at inference gives a different number.
        let out = bn.eval_forward(&frames[0]).expect("valid");
        assert!((out.data[3] - 3.0 / (20.0_f64 / 3.0).sqrt()).abs() < 1e-15);
    }

    /// `channel_moments` against a hand-computed mean and variance, so the moment code itself is
    /// checked against arithmetic rather than against the normalisation that uses it.
    #[test]
    fn channel_moments_match_a_hand_calculation() {
        let frames = vec![t(2, 1, 2, &[1., 3., 10., 10.]), t(2, 1, 2, &[5., 7., 10., 14.])];
        let (mean, var, count) = channel_moments(&frames, 2).expect("valid");
        assert_eq!(count, 4);
        // Channel 0: 1,3,5,7 -> mean 4, deviations -3,-1,1,3 -> variance 20/4 = 5.
        assert_eq!((mean[0], var[0]), (4.0, 5.0));
        // Channel 1: 10,10,10,14 -> mean 11, deviations -1,-1,-1,3 -> variance 12/4 = 3.
        assert_eq!((mean[1], var[1]), (11.0, 3.0));
        assert!(matches!(channel_moments(&[], 2), Err(ConvError::Zero { what: "batch length" })));
    }

    /// (d) A zero-weight branch makes a `SEW`-ADD block bit-exactly the identity, for every
    /// timestep of a run — and `AND` and `IAND` behave exactly as their doc says under the same
    /// branch. Three different answers from one construction, so the test cannot pass by accident.
    #[test]
    fn a_zero_weight_sew_block_is_exactly_the_identity() {
        let spec = Conv2dSpec::same_padding(1, 1, 3, 3).expect("valid");
        let frames = [
            [1., 0., 0., 1.],
            [1., 1., 0., 0.],
            [0., 0., 1., 1.],
            [1., 1., 1., 1.],
            [0., 0., 0., 0.],
        ];
        for (style, want) in [
            (ResidualStyle::SewAdd, Some(())),
            (ResidualStyle::SewIand, Some(())),
            (ResidualStyle::SewAnd, None),
        ] {
            let mut b = ResidualBlock::new(
                Conv2d::zeros(spec).expect("valid"),
                Conv2d::zeros(spec).expect("valid"),
                2,
                2,
                Lif::default(),
                1e-3,
                style,
                0.0,
            )
            .expect("shape preserving");
            for f in &frames {
                let x = t(1, 2, 2, f);
                let out = b.step(&x, ActivationKind::Spiking).expect("fits");
                match want {
                    Some(()) => assert_eq!(out.data, x.data, "{style:?} on {f:?}"),
                    None => assert_eq!(out.data, vec![0.0; 4], "{style:?} on {f:?}"),
                }
            }
            // The block's own synaptic accounting, hand-computed. A 3x3 kernel with padding 1 on a
            // 2x2 map: each output row sees 2 of its 3 taps in bounds and likewise each column, so
            // 4 * 4 = 16 real connections a timestep, over five frames. Every weight is zero, so
            // not one of them is effective — which is the arithmetic statement of "silent branch".
            assert_eq!(b.second.in_bounds_taps(2, 2), Some(16));
            assert_eq!(b.ops.dense, 80, "{style:?}");
            assert_eq!(b.ops.effective_acs, 0, "{style:?}");
            assert_eq!(b.first.ops.dense, 80, "{style:?}");

            // reset returns every membrane in BOTH stages to rest.
            b.first.units[0].v = b.first.units[0].v_th;
            b.units[0].refractory = 1e-3;
            b.reset();
            assert_eq!(b.first.units[0].potential(), Lif::default().v_rest);
            assert_eq!(b.units[0].refractory_left(), 0.0);
        }
    }

    /// (d) The naive residual CANNOT be the identity, for any shortcut gain — the Fang et al. 2021
    /// argument, run rather than quoted.
    ///
    /// The gain is chosen so a single shortcut spike fires the neuron from rest with margin: the
    /// bump is applied before the step's decay, so the threshold gain is
    /// `(v_th - v_rest) / exp(-dt / tau_m) = 15 mV / 0.9512 = 15.77 mV`, and 20 mV clears it. The
    /// block still fails to reproduce a constant input, because a 2 ms refractory period at
    /// `dt = 1 ms` costs two timesteps after every spike.
    #[test]
    fn the_naive_residual_cannot_be_the_identity_even_with_a_tuned_gain() {
        let spec = Conv2dSpec::same_padding(1, 1, 3, 3).expect("valid");
        let lif = Lif::default();
        let critical = (lif.v_th - lif.v_rest) / (-1e-3 / lif.tau_m).exp();
        assert!(critical > 15e-3 && critical < 16e-3, "critical gain {critical}");

        let mut naive = ResidualBlock::new(
            Conv2d::zeros(spec).expect("valid"),
            Conv2d::zeros(spec).expect("valid"),
            1,
            1,
            lif,
            1e-3,
            ResidualStyle::Naive,
            20e-3,
        )
        .expect("shape preserving");
        let mut sew = ResidualBlock::new(
            Conv2d::zeros(spec).expect("valid"),
            Conv2d::zeros(spec).expect("valid"),
            1,
            1,
            lif,
            1e-3,
            ResidualStyle::SewAdd,
            20e-3,
        )
        .expect("shape preserving");

        let on = t(1, 1, 1, &[1.0]);
        let mut naive_out = Vec::new();
        let mut sew_out = Vec::new();
        for _ in 0..9 {
            naive_out.push(naive.step(&on, ActivationKind::Spiking).expect("fits").data[0]);
            sew_out.push(sew.step(&on, ActivationKind::Spiking).expect("fits").data[0]);
        }
        // SEW-ADD reproduces the shortcut at every timestep.
        assert_eq!(sew_out, vec![1.0; 9]);
        // The naive block fires once every three timesteps: one spike, then two refractory.
        assert_eq!(naive_out, vec![1., 0., 0., 1., 0., 0., 1., 0., 0.]);
        // And the gain cannot fix it: a hundred times the critical gain gives the same train,
        // because the failure is in the reset and the refractory period, not in the drive.
        naive.shortcut_gain = 2.0;
        naive.reset();
        let mut retry = Vec::new();
        for _ in 0..9 {
            retry.push(naive.step(&on, ActivationKind::Spiking).expect("fits").data[0]);
        }
        assert_eq!(retry, naive_out);
    }

    /// The shortcut gradient: exactly 1 per block for `SEW`-ADD at any depth, geometric decay for
    /// the naive block, against a closed form computed independently.
    #[test]
    fn the_identity_path_gain_is_one_for_sew_and_geometric_for_naive() {
        let s = FastSigmoid::default();
        for depth in [0u32, 1, 10, 100] {
            assert_eq!(identity_path_gain(ResidualStyle::SewAdd, &s, -3.0, 1.0, depth), Some(1.0));
            assert_eq!(identity_path_gain(ResidualStyle::SewIand, &s, -3.0, 1.0, depth), Some(1.0));
        }
        // A gain chosen so the per-block factor is exactly 0.5, checked against powi.
        let offset = -1.0;
        let g = 0.5 / s.backward(offset);
        for depth in [1u32, 5, 20, 100] {
            let got = identity_path_gain(ResidualStyle::Naive, &s, offset, g, depth).expect("finite");
            let want = 0.5_f64.powi(depth as i32);
            assert!((got - want).abs() <= 1e-12 * want, "depth {depth}: {got} vs {want}");
        }
        // At depth 100 the naive shortcut has lost 30 orders of magnitude and SEW has lost nothing.
        let naive = identity_path_gain(ResidualStyle::Naive, &s, offset, g, 100).expect("finite");
        assert!(naive < 1e-30, "naive gain {naive}");
        assert!(identity_path_gain(ResidualStyle::Naive, &s, f64::NAN, 1.0, 1).is_none());

        // SewAnd zeroes the shortcut at the first block, and is 1 at depth 0 because a stack of no
        // blocks is the identity whatever the style. Neither value is reachable from the other
        // branches, so a mutation that merged the arms is caught.
        assert_eq!(identity_path_gain(ResidualStyle::SewAnd, &s, -1.0, 1.0, 0), Some(1.0));
        for depth in [1u32, 2, 50] {
            assert_eq!(identity_path_gain(ResidualStyle::SewAnd, &s, -1.0, 1.0, depth), Some(0.0));
        }

        // The heavy-tailed surrogate does not rescue it: the decay is geometric either way, only
        // the ratio changes.
        let a = ArcTan::default();
        let heavy = identity_path_gain(ResidualStyle::Naive, &a, -5.0, 1.0, 50).expect("finite");
        assert!(heavy < 1e-40, "arctan gain {heavy}");
    }

    /// `merge` refuses for the naive style rather than inventing an element-wise rule for a block
    /// that has none.
    #[test]
    fn the_naive_style_has_no_element_wise_merge() {
        assert_eq!(ResidualStyle::Naive.merge(1.0, 1.0), None);
        assert_eq!(ResidualStyle::SewAdd.merge(1.0, 1.0), Some(2.0));
        assert_eq!(ResidualStyle::SewAnd.merge(1.0, 1.0), Some(1.0));
        assert_eq!(ResidualStyle::SewAnd.merge(0.0, 1.0), Some(0.0));
        assert_eq!(ResidualStyle::SewIand.merge(1.0, 1.0), Some(0.0));
        assert_eq!(ResidualStyle::SewIand.merge(0.0, 1.0), Some(1.0));
        assert!(ResidualStyle::SewAdd.leaves_binary_regime());
        for st in [ResidualStyle::Naive, ResidualStyle::SewAnd, ResidualStyle::SewIand] {
            assert!(!st.leaves_binary_regime(), "{st:?}");
        }
    }

    /// `SEW`-ADD really does leave the binary regime, and the next layer really does refuse to
    /// charge it as accumulates. The cost in the doc, demonstrated.
    #[test]
    fn a_sew_add_block_emits_values_above_one_and_the_next_layer_refuses_the_cheap_charge() {
        let spec = Conv2dSpec::same_padding(1, 1, 1, 1).expect("valid");
        let mut conv = Conv2d::zeros(spec).expect("valid");
        conv.bias[0] = 1.0; // every unit driven hard past threshold every timestep
        let mut b = ResidualBlock::new(
            conv,
            Conv2d::new(spec, vec![0.0], vec![1.0]).expect("valid"),
            1,
            1,
            Lif::default(),
            1e-3,
            ResidualStyle::SewAdd,
            0.0,
        )
        .expect("shape preserving");
        let on = t(1, 1, 1, &[1.0]);
        let out = b.step(&on, ActivationKind::Spiking).expect("fits");
        assert_eq!(out.data, vec![2.0], "branch spike plus shortcut spike");
        assert!(!out.is_binary());

        let next = conv1(&[1.0], 1, 1, 0, 1, 1);
        assert!(matches!(
            next.synops(&out, ActivationKind::Spiking),
            Err(ConvError::NotBinary { value, .. }) if value == 2.0
        ));
        assert!(next.synops(&out, ActivationKind::RealValued).is_ok());
    }

    /// The threshold-scaled initialiser hits the second moment its doc derives, measured two ways:
    /// the weight variance directly, and the pre-activation's second moment under Bernoulli input.
    ///
    /// Both are Monte Carlo against a CLOSED FORM, not against a previous run. A wrong constant —
    /// `sqrt(2/fan_in)` instead, or a missing `sqrt(3)` in the uniform half-width — moves the
    /// measured value by much more than the tolerance.
    #[test]
    fn the_threshold_scaled_init_hits_its_second_moment() {
        let v_th = 15e-3;
        let rate = 0.2;
        let spec = Conv2dSpec::new(8, 1, 8, 8).expect("valid"); // fan_in = 512
        let fan_in = spec.fan_in();
        assert_eq!(fan_in, 512);
        let init = Init::ThresholdScaled { v_th, rate };
        let sigma = init.std_dev(fan_in).expect("valid");
        assert!((sigma - v_th / ((fan_in as f64) * rate).sqrt()).abs() < 1e-18);

        let c = Conv2d::init(spec, init, 12345).expect("valid");
        let ms = c.weights.iter().map(|w| w * w).sum::<f64>() / fan_in as f64;
        let rel = (ms - sigma * sigma).abs() / (sigma * sigma);
        assert!(rel < 0.10, "weight second moment off by {rel}");

        // E[z^2] = fan_in * sigma^2 * rate = v_th^2, over redrawn weights AND spikes.
        let mut rng = crate::rng::Rng::new(7);
        let mut acc = 0.0;
        let samples: u32 = 2000;
        for s in 0..samples {
            let c = Conv2d::init(spec, init, 1000 + u64::from(s)).expect("valid");
            let mut z = 0.0;
            for w in &c.weights {
                if rng.next_f64() < rate {
                    z += w;
                }
            }
            acc += z * z;
        }
        let measured = acc / f64::from(samples);
        let rel = (measured - v_th * v_th).abs() / (v_th * v_th);
        assert!(rel < 0.12, "pre-activation second moment {measured} vs {} (rel {rel})", v_th * v_th);

        // The Kaiming and LeCun variants differ by exactly sqrt(2), which is the whole content of
        // the rectifier correction.
        let k = Init::KaimingUniform.std_dev(fan_in).expect("valid");
        let l = Init::LecunUniform.std_dev(fan_in).expect("valid");
        assert!((k / l - 2.0_f64.sqrt()).abs() < 1e-15);
        assert_eq!(Init::ThresholdScaled { v_th, rate: 0.0 }.std_dev(fan_in), None);
        assert_eq!(Init::ThresholdScaled { v_th, rate: 1.5 }.std_dev(fan_in), None);
        assert_eq!(Init::KaimingUniform.std_dev(0), None);
    }

    /// A spiking convolution fires where the arithmetic says it must, and its ledger counts what
    /// the hand calculation says. One driven unit, one silent one, over three timesteps.
    #[test]
    fn a_spiking_convolution_fires_and_bills_what_it_did() {
        // 1x1 kernel of 20 mV: a single input spike clears the 15 mV threshold after decay.
        let spec = Conv2dSpec::new(1, 1, 1, 1).expect("valid");
        let conv = Conv2d::new(spec, vec![20e-3], vec![0.0]).expect("valid");
        let mut layer = SpikingConv2d::new(conv, 1, 2, Lif::default(), 1e-3).expect("valid");
        assert_eq!(layer.out_shape(), (1, 1, 2));

        let x = t(1, 1, 2, &[1.0, 0.0]);
        let out = layer.step(&x, ActivationKind::Spiking).expect("fits");
        assert_eq!(out.data, vec![1.0, 0.0], "only the driven unit fires");

        // Two units, one tap each: 2 dense taps, 1 effective (the silent unit's input is zero).
        assert_eq!(layer.ops.dense, 2);
        assert_eq!(layer.ops.effective_acs, 1);
        assert_eq!(layer.ledger.syn_ops, 1);
        assert_eq!(layer.ledger.syn_fetches, 1);
        assert_eq!(layer.ledger.spikes_out, 1);
        assert_eq!(layer.ledger.neuron_updates_driven, 1);
        assert_eq!(layer.ledger.neuron_updates_idle, 1);

        // The idle fraction is the whole event-driven argument, and here it is exactly one half.
        assert_eq!(layer.ledger.idle_fraction(), Some(0.5));

        // The driven unit is refractory for the next two ticks, so the same input does not fire.
        for _ in 0..2 {
            let out = layer.step(&x, ActivationKind::Spiking).expect("fits");
            assert_eq!(out.data, vec![0.0, 0.0]);
        }
        assert_eq!(layer.ledger.spikes_out, 1);
        assert_eq!(layer.ledger.syn_ops, 3);
        // The counts ACCUMULATE across timesteps rather than being overwritten: 2 dense taps a
        // timestep, three timesteps.
        assert_eq!(layer.ops.dense, 6);
        assert_eq!(layer.ops.effective_acs, 3);

        // Determinism: a second identical layer produces the identical train.
        let conv = Conv2d::new(spec, vec![20e-3], vec![0.0]).expect("valid");
        let mut again = SpikingConv2d::new(conv, 1, 2, Lif::default(), 1e-3).expect("valid");
        let mut a = Vec::new();
        for _ in 0..3 {
            a.push(again.step(&x, ActivationKind::Spiking).expect("fits").data.clone());
        }
        assert_eq!(a[0], vec![1.0, 0.0]);
        assert_eq!(a[1], vec![0.0, 0.0]);
        assert_eq!(again.ledger, layer.ledger);

        // A fourth step: the refractory period has expired, so the same input fires again and
        // leaves a counter that is distinguishable from rest — which is what makes the next two
        // assertions able to fail.
        let out = layer.step(&x, ActivationKind::Spiking).expect("fits");
        assert_eq!(out.data, vec![1.0, 0.0]);
        assert_eq!(layer.units[0].refractory_left(), Lif::default().t_ref);

        // `reset` leaves the counters alone, as its doc promises; adding a ledger clear to it was
        // green because the only test called `clear_counts` first.
        let counted = layer.ledger;
        let ops = layer.ops;
        layer.reset();
        assert_eq!(layer.ledger, counted, "reset cleared the ledger");
        assert_eq!(layer.ops.dense, ops.dense);
        assert_eq!(layer.units[0].refractory_left(), 0.0);
        let _ = layer.step(&x, ActivationKind::Spiking).expect("fits");

        layer.clear_counts();
        assert_eq!(layer.ledger.syn_ops, 0);
        assert_eq!(layer.ops.dense, 0);
        assert_eq!(
            layer.units[0].refractory_left(),
            Lif::default().t_ref,
            "clear_counts zeroes the counters and leaves the membranes alone"
        );
        layer.reset();
        assert_eq!(layer.units[0].refractory_left(), 0.0);
        assert_eq!(layer.units[0].potential(), Lif::default().v_rest);
    }

    /// The layer's state is the footprint nobody reports, and here it is against the weights.
    #[test]
    fn the_membrane_state_dominates_the_parameter_count() {
        let spec = Conv2dSpec::same_padding(4, 8, 3, 3).expect("valid");
        let conv = Conv2d::init(spec, Init::KaimingUniform, 3).expect("valid");
        let layer = SpikingConv2d::new(conv, 16, 16, Lif::default(), 1e-3).expect("valid");
        // 8 * 4 * 3 * 3 = 288 weights + 8 biases = 296 parameters; 8 * 16 * 16 = 2048 units.
        assert_eq!(layer.conv.n_params(), 296);
        assert_eq!(layer.units.len(), 2048);
        let f = layer.footprint(32, 2, 32).expect("valid");
        assert_eq!(f.parameter_bits, 296 * 32);
        assert_eq!(f.state_bits, 2048 * 2 * 32);
        // ⛔ Distinct widths, because `footprint(32, 2, 32)` cannot see them swapped — and int8
        // weights with f32 membranes is exactly the configuration the 28.4x argument is about.
        let g = layer.footprint(8, 2, 32).expect("valid");
        assert_eq!(g.parameter_bits, 296 * 8);
        assert_eq!(g.state_bits, 2048 * 2 * 32);
        assert!(f.state_bits > 13 * f.parameter_bits, "state is {}x the parameters", f.state_bits / f.parameter_bits);

        // The three figures the module doc quotes, recomputed. A doc number nothing checks is a
        // doc number that drifts: this paragraph said "29x" until this assertion was written.
        let big = Conv2dSpec::same_padding(64, 64, 3, 3).expect("valid");
        assert_eq!(big.out_shape(128, 128), Some((128, 128)), "the layer must be shape-preserving");
        assert_eq!(big.n_weights(), 36_864);
        let units = 64usize * 128 * 128;
        assert_eq!(units, 1_048_576);
        assert_eq!(4 * units, 4 * 1024 * 1024, "4 MiB of f32 membranes");
        assert_eq!(4 * big.n_weights(), 144 * 1024, "144 KiB of f32 weights");
        let ratio = units as f64 / big.n_weights() as f64;
        assert!((ratio - 28.4).abs() < 0.05, "state:weight ratio is {ratio}, not 28.4");
    }

    /// The surrogate mask is the surrogate's own peak, exactly, for a unit sitting at threshold —
    /// and the scale argument is the thing that decides whether the units look near threshold at
    /// all.
    #[test]
    fn the_surrogate_mask_is_exact_at_threshold_and_depends_on_the_scale() {
        let spec = Conv2dSpec::new(1, 1, 1, 1).expect("valid");
        let conv = Conv2d::zeros(spec).expect("valid");
        let mut layer = SpikingConv2d::new(conv, 1, 2, Lif::default(), 1e-3).expect("valid");
        let s = FastSigmoid::default();
        let lif = Lif::default();
        let scale = lif.v_th - lif.v_rest;

        layer.units[0].v = lif.v_th;
        let m = layer.surrogate_mask(&s, lif.v_th, scale).expect("valid");
        assert_eq!(m.data[0], s.peak(), "a unit exactly at threshold gets the peak, exactly");

        // The resting unit is one full threshold below, in the surrogate's frame.
        assert!((m.data[1] - s.backward(-1.0)).abs() < 1e-18);
        assert!(m.data[1] < m.data[0]);

        // With the correct scale the resting unit passes 1/(1+beta)^2 = 1/121 of the peak. With
        // scale = 1 volt its offset is 15 millivolts in the surrogate's frame, it passes 76%, and
        // the mask says almost nothing about which units are near threshold — the unit error the
        // doc warns about, as a factor of 91 in the reported gradient.
        let right_ratio = m.data[1] / m.data[0];
        let wrong = layer.surrogate_mask(&s, lif.v_th, 1.0).expect("valid");
        let wrong_ratio = wrong.data[1] / wrong.data[0];
        assert!((right_ratio - 1.0 / 121.0).abs() < 1e-15, "right ratio {right_ratio}");
        assert!(wrong_ratio > 0.7, "wrong ratio {wrong_ratio}");
        assert!(wrong_ratio / right_ratio > 80.0, "the two scales must disagree sharply");
        assert!(layer.surrogate_mask(&s, lif.v_th, 0.0).is_err());
        assert!(layer.surrogate_mask(&s, f64::NAN, scale).is_err());
    }

    /// Shapes, lengths, finiteness and channel counts are refused at the boundary, each naming what
    /// was wrong.
    #[test]
    fn bad_inputs_are_refused_by_name() {
        assert!(matches!(
            Tensor3::new(1, 2, 2, vec![0.0; 3]),
            Err(ConvError::BadShape { got: 3, want: 4, .. })
        ));
        assert!(matches!(
            Tensor3::new(1, 1, 1, vec![f64::NAN]),
            Err(ConvError::NonFinite { index: 0, .. })
        ));
        assert!(matches!(Tensor3::zeros(0, 1, 1), Err(ConvError::Zero { what: "channels" })));
        assert!(matches!(
            Conv2dSpec::new(0, 1, 1, 1),
            Err(ConvError::Zero { what: "in_channels" })
        ));
        assert!(matches!(
            Conv2dSpec::same_padding(1, 1, 2, 3),
            Err(ConvError::Zero { what: "odd kernel_h" })
        ));
        let spec = Conv2dSpec::new(1, 1, 2, 2).expect("valid");
        assert!(matches!(
            Conv2d::new(spec, vec![1.0; 3], vec![0.0]),
            Err(ConvError::BadShape { what: "conv weights", got: 3, want: 4 })
        ));
        assert!(matches!(
            Conv2d::new(spec, vec![1.0; 4], vec![0.0, 0.0]),
            Err(ConvError::BadShape { what: "conv bias", .. })
        ));
        let c = Conv2d::zeros(spec).expect("valid");
        assert!(matches!(
            c.forward(&Tensor3::zeros(2, 3, 3).expect("valid")),
            Err(ConvError::ShapeMismatch { what: "conv input channels", .. })
        ));
        assert!(matches!(
            SpikingConv2d::new(c.clone(), 3, 3, Lif::default(), 0.0),
            Err(ConvError::NonFinite { what: "dt", .. })
        ));
        // A residual block whose branch changes shape is refused rather than silently cropped.
        let narrowing = Conv2dSpec::new(1, 1, 3, 3).expect("valid");
        assert!(matches!(
            ResidualBlock::new(
                Conv2d::zeros(narrowing).expect("valid"),
                Conv2d::zeros(narrowing).expect("valid"),
                4,
                4,
                Lif::default(),
                1e-3,
                ResidualStyle::SewAdd,
                0.0,
            ),
            Err(ConvError::ShapeMismatch { what: "residual first convolution", .. })
        ));
        // The SECOND convolution's shape check, which only the first's exercised.
        let same = Conv2dSpec::same_padding(1, 1, 3, 3).expect("valid");
        assert!(matches!(
            ResidualBlock::new(
                Conv2d::zeros(same).expect("valid"),
                Conv2d::zeros(narrowing).expect("valid"),
                4,
                4,
                Lif::default(),
                1e-3,
                ResidualStyle::SewAdd,
                0.0,
            ),
            Err(ConvError::ShapeMismatch { what: "residual second convolution", .. })
        ));
        assert!(matches!(
            ResidualBlock::new(
                Conv2d::zeros(same).expect("valid"),
                Conv2d::zeros(same).expect("valid"),
                4,
                4,
                Lif::default(),
                1e-3,
                ResidualStyle::Naive,
                f64::NAN,
            ),
            Err(ConvError::NonFinite { what: "shortcut_gain", .. })
        ));
        // Non-finite fill and non-finite tdBN threshold.
        assert!(Tensor3::filled(1, 1, 1, f64::INFINITY).is_err());
        assert!(TdBn::new(1, f64::NAN).is_err());
        assert!(TdBn::new(0, 1.0).is_err());
        // A documented range with, until now, no check.
        let mut bn = TdBn::new(1, 1.0).expect("valid");
        bn.momentum = -1.0;
        assert!(matches!(bn.train_forward(&[t(1, 1, 1, &[1.0])]), Err(ConvError::NonFinite { what: "momentum (outside [0, 1])", .. })));
        let mut bn = TdBn::new(1, 1.0).expect("valid");
        bn.running_var[0] = -4.0;
        assert!(matches!(bn.eval_forward(&t(1, 1, 1, &[1.0])), Err(ConvError::NonFinite { what: "running_var (negative)", .. })));
        // Two entry points that validated the shape and the finiteness but not the length.
        let mut short = t(1, 2, 2, &[0.0; 4]);
        short.data.truncate(2);
        let mut pool = SpikingMaxPool::new(PoolSpec::new(2, 2).expect("valid"), MaxPolicy::Instant, 1, 2, 2).expect("valid");
        assert!(matches!(pool.step(&short), Err(ConvError::BadShape { what: "spiking max pool input", got: 2, want: 4 })));
        let bn = TdBn::new(1, 1.0).expect("valid");
        assert!(matches!(bn.eval_forward(&short), Err(ConvError::BadShape { what: "tdBN input", got: 2, want: 4 })));
    }

    /// The corners of the public surface that no other test reaches: `Init::Fixed`, the tensor
    /// accessors, the pool's own shape report, the batch-shape refusals, and the error conversion.
    ///
    /// Collected here rather than left uncovered — every one of them could be deleted or inverted
    /// without any other test in this module noticing.
    #[test]
    fn the_remaining_public_surface_behaves() {
        assert_eq!(Init::Fixed { std_dev: 0.25 }.std_dev(999), Some(0.25));
        assert_eq!(Init::Fixed { std_dev: 0.0 }.std_dev(4), Some(0.0));
        assert_eq!(Init::Fixed { std_dev: -1.0 }.std_dev(4), None);
        assert_eq!(Init::Fixed { std_dev: f64::NAN }.std_dev(4), None);
        // A zero width really does give a silent kernel, which is what the identity test relies on.
        let spec = Conv2dSpec::new(1, 1, 2, 2).expect("valid");
        let c = Conv2d::init(spec, Init::Fixed { std_dev: 0.0 }, 9).expect("valid");
        assert_eq!(c.weights, vec![0.0; 4]);
        assert_eq!(c.n_params(), 5);
        assert!(Conv2d::init(spec, Init::Fixed { std_dev: -1.0 }, 9).is_err());

        let x = t(2, 1, 3, &[0., 1., 1., 0., 0., 1.]);
        assert_eq!(x.at(0, 0, 1), Some(1.0));
        assert_eq!(x.at(1, 0, 2), Some(1.0));
        assert_eq!(x.at(1, 0, 0), Some(0.0));
        assert_eq!(x.at(2, 0, 0), None);
        assert_eq!(x.at(0, 1, 0), None);
        assert_eq!(x.at(0, 0, 3), None);
        assert_eq!(x.count_nonzero(), 3);
        assert!(x.is_binary());
        assert!(!t(1, 1, 1, &[0.5]).is_binary());
        assert!(x.require_binary().is_ok());

        let p = SpikingMaxPool::new(PoolSpec::new(2, 2).expect("valid"), MaxPolicy::Instant, 3, 4, 6)
            .expect("valid");
        assert_eq!(p.out_shape().expect("fits"), (3, 2, 3));
        let mut p = p;
        assert!(matches!(
            p.step(&Tensor3::zeros(3, 4, 4).expect("valid")),
            Err(ConvError::ShapeMismatch { what: "spiking max pool input", .. })
        ));

        // channel_moments refuses a ragged batch and a channel count that does not match.
        let a = t(1, 1, 2, &[1., 2.]);
        let b = t(1, 1, 3, &[1., 2., 3.]);
        assert!(matches!(
            channel_moments(&[a.clone(), b], 1),
            Err(ConvError::ShapeMismatch { what: "batch member", .. })
        ));
        assert!(matches!(
            channel_moments(&[a], 2),
            Err(ConvError::ShapeMismatch { what: "batch channels", .. })
        ));
        assert!(matches!(channel_moments(&[], 1), Err(ConvError::Zero { .. })));

        // The metrics error folds into an overflow that keeps the name it came with.
        let e: ConvError = crate::metrics::MetricError::Overflow { what: "dense ops" }.into();
        assert_eq!(e, ConvError::Overflow { what: "dense ops" });
        let e: ConvError = crate::metrics::MetricError::Empty { what: "activations" }.into();
        assert!(matches!(e, ConvError::Overflow { .. }));

        // Conv2dSpec::same_padding really does preserve the shape it names, on both axes and for
        // every odd kernel this module is likely to see.
        for k in [1usize, 3, 5, 7] {
            let s = Conv2dSpec::same_padding(2, 2, k, k).expect("valid");
            assert_eq!(s.out_shape(9, 11), Some((9, 11)), "k={k}");
            assert_eq!(s.fan_in(), 2 * k * k);
            assert_eq!(s.n_weights(), 2 * 2 * k * k);
        }
        // The documented zero-dilation refusal, which the claim list left out.
        assert_eq!(out_dim(5, 3, 0, 1, 0), None);
        assert_eq!(out_dim(5, 3, 0, 1, 1), Some(3));
        // The defaults `TdBn::new` documents as PyTorch's, pinned.
        let bn = TdBn::new(3, 0.2).expect("valid");
        assert_eq!((bn.eps, bn.momentum, bn.alpha), (1e-5, 0.1, 1.0));
    }

    /// The error messages carry the numbers. A message that says only "bad shape" costs the caller
    /// the debugging session the two numbers would have ended.
    #[test]
    fn error_messages_name_the_numbers() {
        let e = ConvError::ImpossibleShape {
            axis: "height",
            n: 3,
            kernel: 5,
            pad: 0,
            stride: 1,
            dilation: 1,
        };
        let s = e.to_string();
        assert!(s.contains("extent 5"), "{s}");
        assert!(s.contains("padded input 3"), "{s}");
        // With dilation and padding, so both arithmetic terms in the message are live: the extent
        // is d*(k-1)+1 = 7 and the padded input is n + 2p = 6.
        let e = ConvError::ImpossibleShape { axis: "width", n: 4, kernel: 3, pad: 1, stride: 1, dilation: 3 };
        let s = e.to_string();
        assert!(s.contains("extent 7") && s.contains("padded input 6"), "{s}");
        let s = ConvError::NotBinary { index: 7, value: 2.0 }.to_string();
        assert!(s.contains('7') && s.contains('2'), "{s}");
    }

    /// ⛔ A LIVE BLOCK. The AC/MAC column assertion in the identity test is made on an all-zero
    /// block where both columns are zero, so swapping them was green. Here both stages fire, the
    /// second stage's accumulates land in the right column, and the block's own ledger — new; the
    /// block had none — carries the second stage, so the block's total is twice what `first`
    /// alone reports.
    #[test]
    fn a_live_residual_block_counts_its_second_stage_in_the_right_column_and_in_a_ledger() {
        let spec = Conv2dSpec::new(1, 1, 1, 1).expect("valid");
        let hot = || Conv2d::new(spec, vec![20e-3], vec![0.0]).expect("valid");
        let mut b =
            ResidualBlock::new(hot(), hot(), 2, 2, Lif::default(), 1e-3, ResidualStyle::SewAdd, 0.0)
                .expect("shape preserving");
        let x = t(1, 2, 2, &[1., 0., 1., 1.]);
        let out = b.step(&x, ActivationKind::Spiking).expect("fits");
        // Stage one fires on every driven unit; stage two is driven by those spikes and fires
        // too; SEW-ADD then adds the shortcut: 2 where the input was 1, 0 elsewhere.
        assert_eq!(out.data, vec![2., 0., 2., 2.]);
        assert_eq!(b.first.ledger.spikes_out, 3);
        assert_eq!((b.ops.dense, b.ops.effective_macs, b.ops.effective_acs), (4, 0, 3));
        assert_eq!(b.ledger.syn_ops, 3);
        assert_eq!(b.ledger.syn_fetches, 3);
        assert_eq!(b.ledger.spikes_out, 3);
        assert_eq!(b.ledger.neuron_updates_driven, 3);
        assert_eq!(b.ledger.neuron_updates_idle, 1);
        assert_eq!(b.first.ledger.spikes_out + b.ledger.spikes_out, 6, "eight units fired, six spikes");
        b.clear_counts();
        assert_eq!(b.ledger, crate::ledger::Ledger::default());
        assert_eq!(b.first.ledger, crate::ledger::Ledger::default());
        assert_eq!(b.ops.dense, 0);
    }

    /// ⛔ A graded input is charged fetches but NOT accumulates. The same layer and the same
    /// numbers in, once as spikes and once as real values: the ops columns swap, the fetches
    /// agree, and the ledger's synaptic operations — the accumulate count, the thing
    /// `Prices::e_syn_op` prices — are 1 and 0. The first version produced identical ledgers.
    #[test]
    fn a_graded_input_is_charged_fetches_but_not_accumulates() {
        let spec = Conv2dSpec::new(1, 1, 1, 1).expect("valid");
        let mk = || {
            SpikingConv2d::new(
                Conv2d::new(spec, vec![20e-3], vec![0.0]).expect("valid"),
                1,
                2,
                Lif::default(),
                1e-3,
            )
            .expect("valid")
        };
        let x = t(1, 1, 2, &[1.0, 0.0]);
        let (mut spiking, mut graded) = (mk(), mk());
        let a = spiking.step(&x, ActivationKind::Spiking).expect("fits");
        let b = graded.step(&x, ActivationKind::RealValued).expect("fits");
        assert_eq!(a.data, b.data, "the membrane arithmetic is the same; only the charge differs");
        assert_eq!((spiking.ops.effective_acs, spiking.ops.effective_macs), (1, 0));
        assert_eq!((graded.ops.effective_acs, graded.ops.effective_macs), (0, 1));
        assert_eq!((spiking.ledger.syn_fetches, graded.ledger.syn_fetches), (1, 1));
        assert_eq!(spiking.ledger.syn_ops, 1);
        assert_eq!(graded.ledger.syn_ops, 0, "a multiply-accumulate is not an accumulate");
        assert_ne!(spiking.ledger, graded.ledger);
        // And a genuinely graded input under `Spiking` is refused, not rounded.
        let y = t(1, 1, 2, &[2.0, 0.0]);
        assert!(matches!(
            mk().step(&y, ActivationKind::Spiking),
            Err(ConvError::NotBinary { index: 0, .. })
        ));
        assert!(mk().step(&y, ActivationKind::RealValued).is_ok());
        let ops = SynOps { dense: 2, effective_macs: 1, effective_acs: 0 };
        assert_eq!(mk().step(&y, ActivationKind::RealValued).map(|_| ()), Ok(()));
        assert_eq!(ops.effective_macs, 1);
    }

    /// ⛔ A layer whose public spec was changed after construction is refused rather than indexed
    /// out of bounds (padding shrunk: the drive is 3x3 under 5x5 units — a panic) or read through
    /// a scrambled map (padding grown: a 7x7 drive under 5x5 units, `Ok`, wrong).
    #[test]
    fn a_layer_whose_public_spec_was_changed_is_refused_rather_than_indexed_out_of_bounds() {
        let spec = Conv2dSpec::same_padding(1, 1, 3, 3).expect("valid");
        let mut l = SpikingConv2d::new(Conv2d::zeros(spec).expect("valid"), 5, 5, Lif::default(), 1e-3)
            .expect("valid");
        let x = Tensor3::zeros(1, 5, 5).expect("valid");
        assert!(l.step(&x, ActivationKind::Spiking).is_ok());
        l.conv.spec.pad_h = 0;
        l.conv.spec.pad_w = 0;
        assert!(matches!(
            l.step(&x, ActivationKind::Spiking),
            Err(ConvError::ShapeMismatch { what: "spiking conv units", .. })
        ));
        l.conv.spec.pad_h = 2;
        l.conv.spec.pad_w = 2;
        assert!(
            matches!(l.step(&x, ActivationKind::Spiking), Err(ConvError::ShapeMismatch { .. })),
            "a drive larger than the unit array must be refused, not read through"
        );
        // The same guard on the block's second convolution.
        let mut b = ResidualBlock::new(
            Conv2d::zeros(spec).expect("valid"),
            Conv2d::zeros(spec).expect("valid"),
            5,
            5,
            Lif::default(),
            1e-3,
            ResidualStyle::SewAdd,
            0.0,
        )
        .expect("valid");
        b.second.spec.pad_h = 0;
        b.second.spec.pad_w = 0;
        assert!(matches!(
            b.step(&x, ActivationKind::Spiking),
            Err(ConvError::ShapeMismatch { what: "residual second convolution units", .. })
        ));
    }

    /// `eval_forward` subtracts the running MEAN. The unbiased-variance test uses a batch whose
    /// mean is exactly zero, so dropping the subtraction was green — the same class of gap that
    /// was found and closed for `eps`, `gain`, `shift` and `momentum`, one field over.
    #[test]
    fn eval_forward_subtracts_the_running_mean() {
        let mut bn = TdBn::new(1, 1.0).expect("valid");
        bn.eps = 0.0;
        bn.momentum = 1.0;
        // 7, 9, 11, 13: mean 10, population variance 5, unbiased 20/3.
        bn.train_forward(&[t(1, 1, 4, &[7., 9., 11., 13.])]).expect("valid");
        assert_eq!(bn.running_mean[0], 10.0);
        let out = bn.eval_forward(&t(1, 1, 1, &[10.0])).expect("valid");
        assert_eq!(out.data, vec![0.0], "a sample at the running mean normalises to exactly zero");
        let out = bn.eval_forward(&t(1, 1, 1, &[13.0])).expect("valid");
        assert!((out.data[0] - 3.0 / (20.0_f64 / 3.0).sqrt()).abs() < 1e-15);
    }

    /// ⛔ THE UNIT CONVENTION, PINNED. Weights are volts per input spike, delivered by `bump`;
    /// reading them as a current through `r_m` instead was green, because every test used a 20 mV
    /// weight, which is supra-threshold under both readings. At 10 mV the two give different
    /// trains: two bumps, each decayed by `exp(-1/20)` after arrival, clear the 15 mV threshold on
    /// the second tick, the 2 ms refractory period holds two ticks, and then again — [0,1,0,0,0,1]
    /// — where a 10 mV current through 10 MΩ would fire on the first tick and every third.
    #[test]
    fn a_weight_is_a_membrane_displacement_not_a_current() {
        let spec = Conv2dSpec::new(1, 1, 1, 1).expect("valid");
        let conv = Conv2d::new(spec, vec![10e-3], vec![0.0]).expect("valid");
        let mut layer = SpikingConv2d::new(conv, 1, 1, Lif::default(), 1e-3).expect("valid");
        let x = t(1, 1, 1, &[1.0]);
        let train: Vec<f64> =
            (0..6).map(|_| layer.step(&x, ActivationKind::Spiking).expect("fits").data[0]).collect();
        assert_eq!(train, vec![0., 1., 0., 0., 0., 1.]);
    }

    /// `neuron_updates_idle` is "zero net drive", and the doc now says so: two cancelling
    /// deliveries are two synaptic operations and one idle update. Pinned so the semantics cannot
    /// drift silently in either direction.
    #[test]
    fn idle_means_zero_net_drive_not_no_deliveries() {
        let spec = Conv2dSpec::new(2, 1, 1, 1).expect("valid");
        let conv = Conv2d::new(spec, vec![20e-3, -20e-3], vec![0.0]).expect("valid");
        let mut layer = SpikingConv2d::new(conv, 1, 1, Lif::default(), 1e-3).expect("valid");
        let both = t(2, 1, 1, &[1.0, 1.0]);
        let out = layer.step(&both, ActivationKind::Spiking).expect("fits");
        assert_eq!(out.data, vec![0.0]);
        assert_eq!(layer.ledger.syn_ops, 2, "two deliveries happened");
        assert_eq!(layer.ledger.neuron_updates_idle, 1, "and they cancelled to an idle update");
        assert_eq!(layer.ledger.neuron_updates_driven, 0);
        // A bias of one picovolt is a drive.
        let conv = Conv2d::new(spec, vec![0.0, 0.0], vec![1e-12]).expect("valid");
        let mut biased = SpikingConv2d::new(conv, 1, 1, Lif::default(), 1e-3).expect("valid");
        biased.step(&t(2, 1, 1, &[0.0, 0.0]), ActivationKind::Spiking).expect("fits");
        assert_eq!((biased.ledger.neuron_updates_driven, biased.ledger.syn_ops), (1, 0));
    }

    /// A NaN written into a public weight after construction is refused at the next step, not
    /// propagated as `Ok`.
    #[test]
    fn a_weight_poisoned_after_construction_is_refused_at_the_next_step() {
        let spec = Conv2dSpec::same_padding(1, 1, 3, 3).expect("valid");
        let mut l = SpikingConv2d::new(Conv2d::zeros(spec).expect("valid"), 3, 3, Lif::default(), 1e-3)
            .expect("valid");
        let x = Tensor3::zeros(1, 3, 3).expect("valid");
        assert!(l.step(&x, ActivationKind::Spiking).is_ok());
        l.conv.weights[4] = f64::NAN;
        assert!(matches!(
            l.step(&x, ActivationKind::Spiking),
            Err(ConvError::NonFinite { what: "conv weights", index: 4, .. })
        ));
        let mut b = ResidualBlock::new(
            Conv2d::zeros(spec).expect("valid"),
            Conv2d::zeros(spec).expect("valid"),
            3,
            3,
            Lif::default(),
            1e-3,
            ResidualStyle::SewAdd,
            0.0,
        )
        .expect("valid");
        b.second.bias[0] = f64::INFINITY;
        assert!(matches!(
            b.step(&x, ActivationKind::Spiking),
            Err(ConvError::NonFinite { what: "conv bias", .. })
        ));
        // And the block's own input-shape check, which only the layer's was exercised.
        let mut ok = ResidualBlock::new(
            Conv2d::zeros(spec).expect("valid"),
            Conv2d::zeros(spec).expect("valid"),
            3,
            3,
            Lif::default(),
            1e-3,
            ResidualStyle::SewAdd,
            0.0,
        )
        .expect("valid");
        assert!(matches!(
            ok.step(&Tensor3::zeros(1, 4, 4).expect("valid"), ActivationKind::Spiking),
            Err(ConvError::ShapeMismatch { what: "residual block input", .. })
        ));
    }

    /// The tap sweep is bounded: a padding of 5e7 on a 1x1 input answers `None` at once instead of
    /// spending seconds, and a `SpikingMaxPool` whose window cannot fit is refused at construction.
    ///
    /// ⛔ THE FIXTURE BELOW LEFT `pad_w` AT 0, AND THAT MADE THIS TEST VACUOUS. With `pad_w = 0` a
    /// 3-tap kernel does not fit a 1-element row at all, so `out_shape` returned `None` on the
    /// WIDTH axis before `in_bounds_taps` ever reached the sweep bound — deleting the bound
    /// entirely left the assertion green. `pad_w = 1` makes the width axis fit exactly, so the
    /// only thing left that can refuse is the bound this test is named after.
    #[test]
    fn unbounded_geometry_is_refused_rather_than_swept() {
        let mut spec = Conv2dSpec::new(1, 1, 3, 3).expect("valid");
        spec.pad_h = 50_000_000;
        spec.pad_w = 1;
        let c = Conv2d::zeros(spec).expect("valid");
        assert_eq!(spec.out_shape(1, 1), Some((99_999_999, 1)), "both axes must admit a window");
        assert_eq!(c.in_bounds_taps(1, 1), None);

        // The bound is `outputs * taps` on ONE axis, and it is where the constant says: a geometry
        // whose row sweep is 67_108_863 iterations is answered and one of 67_108_869 is refused,
        // either side of MAX_TAP_SWEEP = 2^26. Both are the same 1x1 input and the same 3x3 kernel;
        // only the padding moves. A bound read off the wrong quantity — the tap count, the output
        // count alone, the product over both axes — lands somewhere else entirely.
        assert_eq!(MAX_TAP_SWEEP, 1 << 26);
        let mut inside = Conv2dSpec::new(1, 1, 3, 3).expect("valid");
        inside.pad_h = 11_184_811;
        inside.pad_w = 1;
        let o_n = inside.out_shape(1, 1).expect("fits").0;
        assert_eq!(o_n as u64 * 3, MAX_TAP_SWEEP - 1);
        // Three row taps land on the single real element, one column tap does: 3 * 1 * 1 * 1.
        assert_eq!(Conv2d::zeros(inside).expect("valid").in_bounds_taps(1, 1), Some(3));
        let mut outside = inside;
        outside.pad_h = 11_184_812;
        assert_eq!(outside.out_shape(1, 1).expect("fits").0 as u64 * 3, MAX_TAP_SWEEP + 5);
        assert_eq!(Conv2d::zeros(outside).expect("valid").in_bounds_taps(1, 1), None);
        let mut small = Conv2dSpec::new(1, 1, 3, 3).expect("valid");
        small.pad_h = 1;
        small.pad_w = 1;
        assert_eq!(Conv2d::zeros(small).expect("valid").in_bounds_taps(2, 2), Some(16));
        assert!(matches!(
            SpikingMaxPool::new(PoolSpec::new(5, 5).expect("valid"), MaxPolicy::Instant, 1, 4, 4),
            Err(ConvError::ImpossibleShape { .. })
        ));
        let mut bn = TdBn::new(1, 1.0).expect("valid");
        bn.gain = vec![1.0; 3];
        assert!(matches!(
            bn.eval_forward(&t(1, 1, 1, &[1.0])),
            Err(ConvError::BadShape { what: "gain", got: 3, want: 1 })
        ));
    }

    /// The two-pass variance, on the input where the one-pass shortcut cancels catastrophically:
    /// values of 1e8 ± 1 have a variance of exactly 1, and `E[x²] − E[x]²` in f64 does not get it.
    /// Also the fixed accumulation order: over three input channels with weights `[1e16, -1e16, 1]`
    /// the channel-major sum is `(1e16 − 1e16) + 1 = 1`; reversed it is `(1 − 1e16) + 1e16 = 0`.
    #[test]
    fn the_numerics_the_docs_promise_are_the_numerics_that_run() {
        let frames = vec![t(1, 1, 2, &[1e8 + 1.0, 1e8 - 1.0])];
        let (mean, var, _) = channel_moments(&frames, 1).expect("valid");
        assert_eq!(mean[0], 1e8);
        assert!((var[0] - 1.0).abs() < 1e-9, "variance {} — the shortcut would cancel", var[0]);
        let spec = Conv2dSpec::new(3, 1, 1, 1).expect("valid");
        let conv = Conv2d::new(spec, vec![1e16, -1e16, 1.0], vec![0.0]).expect("valid");
        let out = conv.forward(&t(3, 1, 1, &[1.0, 1.0, 1.0])).expect("valid");
        assert_eq!(out.data, vec![1.0], "the accumulation order is channel-major, bit for bit");
    }

    /// The spike count counts the units that FIRED, not the units that stayed silent.
    ///
    /// `the_remaining_public_surface_behaves` asserts `count_nonzero() == 3` on
    /// `[0, 1, 1, 0, 0, 1]` — a map with exactly three spikes AND exactly three silences, so
    /// counting the wrong side of the predicate gives the same 3. Every tensor here has a
    /// different number of each, and one of them is graded, because "non-zero" and "equal to one"
    /// are the same predicate on a binary map and only differ off it.
    #[test]
    fn the_spike_count_counts_the_firing_units_not_the_silent_ones() {
        assert_eq!(t(1, 1, 3, &[0., 1., 1.]).count_nonzero(), 2);
        assert_eq!(t(1, 1, 3, &[1., 0., 0.]).count_nonzero(), 1);
        assert_eq!(t(1, 1, 4, &[1., 1., 1., 0.]).count_nonzero(), 3);
        assert_eq!(Tensor3::zeros(2, 2, 2).expect("valid").count_nonzero(), 0);
        assert_eq!(Tensor3::filled(2, 2, 2, 1.0).expect("valid").count_nonzero(), 8);
        // A graded map: the count is of non-zero elements, not of elements equal to one, and a
        // negative element is a non-zero one.
        assert_eq!(t(1, 1, 4, &[0.5, 0., -1., 0.]).count_nonzero(), 2);
    }

    /// `same_padding` pads each axis by ITS OWN kernel, which is the whole reason `Conv2dSpec`
    /// carries per-axis fields.
    ///
    /// Every other call in this module passes `kernel_h == kernel_w` — the one asymmetric call,
    /// `same_padding(1, 1, 2, 3)`, is refused for the even row kernel before the column padding is
    /// ever computed — so `pad_w` could be derived from `kernel_h` and every test stayed green.
    /// A `3x5` kernel is the spiking audio-spectrogram front end the `Conv2dSpec` doc names.
    #[test]
    fn same_padding_pads_each_axis_by_its_own_kernel() {
        let tall = Conv2dSpec::same_padding(1, 1, 5, 3).expect("valid");
        assert_eq!((tall.pad_h, tall.pad_w), (2, 1));
        let wide = Conv2dSpec::same_padding(1, 1, 3, 5).expect("valid");
        assert_eq!((wide.pad_h, wide.pad_w), (1, 2));
        assert_eq!(wide.fan_in(), 15);

        // And the name is kept on a NON-SQUARE map, so a swap of the two paddings shows up as a
        // shape change rather than cancelling against a square input.
        assert_eq!(tall.out_shape(7, 9), Some((7, 9)));
        assert_eq!(wide.out_shape(7, 9), Some((7, 9)));
        // The two specs really are different objects: swapping the kernel axes swaps the padding.
        assert_ne!(tall.pad_w, wide.pad_w);

        // Run one of them, so the claim is about a convolution and not only about a struct field.
        let c = Conv2d::zeros(wide).expect("valid");
        let out = c.forward(&Tensor3::zeros(1, 7, 9).expect("valid")).expect("fits");
        assert_eq!(out.shape(), (1, 7, 9));
    }

    /// A non-finite weight or bias is refused by `Conv2d::new`, at the boundary, naming which one
    /// and where.
    ///
    /// Every kernel in this module arrives through `Conv2d::zeros`, `Conv2d::init` or a literal
    /// list of finite numbers, so the constructor's own finiteness check had no fixture at all —
    /// only the per-step re-check in `SpikingConv2d::step` did, and that is a different call.
    #[test]
    fn a_non_finite_weight_or_bias_is_refused_at_construction() {
        let spec = Conv2dSpec::new(1, 1, 2, 2).expect("valid");
        assert!(matches!(
            Conv2d::new(spec, vec![1.0, f64::NAN, 1.0, 1.0], vec![0.0]),
            Err(ConvError::NonFinite { what: "conv weight", index: 1, .. })
        ));
        assert!(matches!(
            Conv2d::new(spec, vec![1.0, 1.0, 1.0, f64::INFINITY], vec![0.0]),
            Err(ConvError::NonFinite { what: "conv weight", index: 3, .. })
        ));
        assert!(matches!(
            Conv2d::new(spec, vec![1.0; 4], vec![f64::NEG_INFINITY]),
            Err(ConvError::NonFinite { what: "conv bias", index: 0, .. })
        ));
        // The same four weights, finite, are accepted — so the refusals above are about the value
        // and not about the shape.
        assert!(Conv2d::new(spec, vec![1.0; 4], vec![0.0]).is_ok());
    }

    /// The initialiser's stream is the seed's own.
    ///
    /// `Conv2d::init` had NO golden vector: the zero-width case pins an all-zero kernel, which is
    /// the one width at which the generator cannot be seen at all, and the two second-moment
    /// checks are Monte Carlo over redrawn seeds, which any seed mapping reproduces. So the seed
    /// could be mixed with a constant on its way to `Rng::new` and every test stayed green, while
    /// "the same seed gives the same kernel on every platform" quietly became false for anyone
    /// comparing against a stream they started themselves.
    ///
    /// The first assertion recomputes the draw from a stream this test starts, at BIT equality —
    /// `(2u - 1) * sigma * sqrt(3)` is the same operations in the same order, so there is no error
    /// term to allow for. The two literals after it are MEASURED: they are what this
    /// implementation's generator produces for this seed, recorded so that a change to the stream
    /// is a visible regression rather than a silent one.
    #[test]
    fn the_initialiser_draws_its_weights_from_the_seed_it_was_handed() {
        let spec = Conv2dSpec::new(1, 2, 2, 2).expect("valid");
        let seed = 20_260_920_u64;
        let c = Conv2d::init(spec, Init::Fixed { std_dev: 1.0 }, seed).expect("valid");
        assert_eq!(c.weights.len(), 8);
        assert_eq!(c.bias, vec![0.0; 2], "biases are zero in every case");

        // `Init::Fixed { std_dev: 1.0 }`, so the target sigma is 1 and the uniform half-width is
        // sqrt(3) exactly. ⛔ Written on its own line rather than as `sigma * 3.0_f64.sqrt()`:
        // that spelling is character for character the implementation's, and a mutation harness
        // that finds its anchor text twice cannot apply the edit at all.
        let half = 3.0_f64.sqrt();
        let mut stream = crate::rng::Rng::new(seed);
        let want: Vec<f64> = (0..8).map(|_| (2.0 * stream.next_f64() - 1.0) * half).collect();
        assert_eq!(c.weights, want, "the kernel is this seed's stream, draw for draw");

        assert_eq!(c.weights[0].to_bits(), 0xbfd2_ec40_f65e_8faf);
        assert_eq!(c.weights[7].to_bits(), 0x3feb_54f5_2bca_1b75);

        // The same seed gives the same kernel; a neighbouring seed does not. Neither statement
        // alone can see a seed that was mixed with a constant — both hold for any injection — so
        // they are here for what they do say, and the two assertions above carry the claim.
        let again = Conv2d::init(spec, Init::Fixed { std_dev: 1.0 }, seed).expect("valid");
        assert_eq!(again.weights, c.weights);
        let neighbour = Conv2d::init(spec, Init::Fixed { std_dev: 1.0 }, seed + 1).expect("valid");
        assert_ne!(neighbour.weights, c.weights);

        // A non-unit width scales the whole kernel by exactly that width, which is what makes the
        // `sqrt(3)` above the only constant in the draw.
        let half = Conv2d::init(spec, Init::Fixed { std_dev: 0.5 }, seed).expect("valid");
        for (h, w) in half.weights.iter().zip(&c.weights) {
            assert_eq!(*h, 0.5 * w);
        }
    }

    /// Both tap counts carry every factor they name, on a convolution that is asymmetric in all
    /// four of them.
    ///
    /// ⛔ EVERY tap-count fixture in this module was `conv1`: one input channel, one output
    /// channel, a square input and identical kernel, padding, stride and dilation on both axes.
    /// A factor of 1 dropped from a product is invisible, and a row sweep handed the column
    /// geometry reads the same numbers back. Here `C_in = 2`, `C_out = 3`, the kernel is `3x2`,
    /// the padding is 1 on rows and 0 on columns, and the input is `4x5`, so each of those four
    /// mistakes gives a different answer.
    ///
    /// Hand-computed. `out_h = (4 + 2 - 3) + 1 = 4`, `out_w = (5 + 0 - 2) + 1 = 4`.
    /// Padded: `fan_in * C_out * out_h * out_w = (2*3*2) * 3 * 4 * 4 = 576`.
    /// In bounds, separably: rows `2 + 3 + 3 + 2 = 10`, columns `2 + 2 + 2 + 2 = 8`, so
    /// `10 * 8 * 2 * 3 = 480`.
    #[test]
    fn both_tap_counts_carry_every_factor_of_an_asymmetric_convolution() {
        let mut spec = Conv2dSpec::new(2, 3, 3, 2).expect("valid");
        spec.pad_h = 1;
        spec.pad_w = 0;
        assert_eq!(spec.fan_in(), 12);
        assert_eq!(spec.n_weights(), 36);
        assert_eq!(spec.out_shape(4, 5), Some((4, 4)));
        let c = Conv2d::new(spec, vec![1.0; 36], vec![0.0; 3]).expect("valid");

        assert_eq!(c.padded_taps(4, 5), Some(576));
        assert_eq!(c.in_bounds_taps(4, 5), Some(480));

        // The convolution loop counts the same in-bounds taps, one at a time, with no
        // factorisation and no channel arithmetic — the independent route to the same number.
        let input = Tensor3::filled(2, 4, 5, 1.0).expect("valid");
        let (out, dense, effective) = c.convolve(&input).expect("fits");
        assert_eq!(dense, 480);
        assert_eq!(effective, 480, "every weight and every input element is non-zero here");
        assert_eq!(out.shape(), (3, 4, 4));

        // Each factor, removed one at a time, lands somewhere else: 192 without the output
        // channels, 240 without the input channels, 336 with the row sweep handed the column
        // geometry (7 row taps instead of 10). None of them is 576 or 480.
        assert_ne!(c.padded_taps(4, 5), Some(192));
        assert_ne!(c.in_bounds_taps(4, 5), Some(240));
        assert_ne!(c.in_bounds_taps(4, 5), Some(336));

        // And the padded figure overstates the real connections by 20% here — the gap the
        // `padded_taps` doc is about, on a shape where it is neither zero nor the 65% of the
        // square fixture.
        let overstatement = 576.0 / 480.0;
        assert_eq!(overstatement, 1.2);
    }

    /// A non-finite element in a convolution's INPUT is refused rather than multiplied into every
    /// output of its window.
    ///
    /// Every tensor in this module reaches a convolution through `Tensor3::new`, which refuses a
    /// `NaN` first, so `convolve`'s own check had no fixture. A `Tensor3::zeros` whose public
    /// `data` is written afterwards is the path a real pipeline takes — it is how the layer's
    /// weight re-check is reached too — and one `NaN` there poisons every output position whose
    /// window contains it, with `Ok` on the call.
    #[test]
    fn a_non_finite_convolution_input_is_refused_rather_than_propagated() {
        let c = conv1(&[1.0; 4], 2, 2, 0, 1, 1);
        let mut poisoned = Tensor3::zeros(1, 3, 3).expect("valid");
        poisoned.data[5] = f64::NAN;
        assert!(matches!(
            c.forward(&poisoned),
            Err(ConvError::NonFinite { what: "conv input", index: 5, .. })
        ));
        poisoned.data[5] = f64::INFINITY;
        assert!(matches!(
            c.forward(&poisoned),
            Err(ConvError::NonFinite { what: "conv input", index: 5, .. })
        ));
        // The same tensor with a finite element there is accepted, so the refusal is about the
        // value. Four windows of four unit taps over a single 1.0 at the centre: every output is 1.
        poisoned.data[5] = 1.0;
        assert_eq!(c.forward(&poisoned).expect("fits").data, vec![0., 1., 0., 1.]);
        // The synaptic-operation path goes through the same convolution and refuses it too.
        poisoned.data[5] = f64::NAN;
        assert!(matches!(
            c.synops(&poisoned, ActivationKind::RealValued),
            Err(ConvError::NonFinite { what: "conv input", .. })
        ));
    }

    /// `PoolSpec::out_shape` does not dilate its window, and agrees with the shape the pooling
    /// functions actually produce.
    ///
    /// ⛔ `PoolSpec::out_shape` is called from NOWHERE inside this module — `avg_pool`,
    /// `max_pool_values` and `SpikingMaxPool` all go through `require_out_shape` — so it was dead
    /// to every test and could have said anything. A window has no dilation (the `PoolSpec` doc
    /// says why), and a dilation of 2 on the row axis of a 3-tap window turns an extent of 3 into
    /// an extent of 5: on a 5-row input that is 1 output row instead of 3.
    #[test]
    fn the_pool_shape_formula_does_not_dilate_its_window() {
        let spec = PoolSpec { kernel_h: 3, kernel_w: 2, stride_h: 1, stride_w: 1, pad_h: 0, pad_w: 0 };
        assert_eq!(spec.out_shape(5, 4), Some((3, 3)));
        assert_eq!(spec.require_out_shape(5, 4).expect("fits"), (3, 3));
        // The number the pooling functions actually produce, which is what the formula claims to
        // predict. The axes are deliberately unequal, so a transposed answer is a different one.
        let input = Tensor3::filled(1, 5, 4, 1.0).expect("valid");
        let pooled = avg_pool(&input, &spec, false).expect("fits");
        assert_eq!((pooled.height, pooled.width), spec.out_shape(5, 4).expect("fits"));
        assert_eq!(max_pool_values(&input, &spec).expect("fits").shape(), (1, 3, 3));

        // With padding and a stride the two routes still agree, and `None` is returned for the
        // window that cannot fit rather than a clamped answer.
        let padded = PoolSpec { kernel_h: 3, kernel_w: 2, stride_h: 2, stride_w: 1, pad_h: 1, pad_w: 0 };
        assert_eq!(padded.out_shape(5, 4), Some((3, 3)));
        let too_big = PoolSpec { kernel_h: 6, kernel_w: 2, stride_h: 1, stride_w: 1, pad_h: 0, pad_w: 0 };
        assert_eq!(too_big.out_shape(5, 4), None);
    }

    /// Each channel is scaled by ITS OWN gain and shifted by its own shift, in training and at
    /// inference.
    ///
    /// `the_epsilon_gain_and_shift_terms_each_move_the_answer` is the only test that sets either,
    /// and it sets them on a ONE-channel layer, where `gain[c]` and `gain[0]` are the same number.
    /// The two-channel fixture leaves both at their defaults, where the gain is 1 and the shift is
    /// 0 and neither term is visible. So "learnable scale per channel" was untested as a per-
    /// channel claim.
    ///
    /// Exact arithmetic, no tolerance. Channel 0 is `1, -1`: mean 0, population variance 1.
    /// Channel 1 is `6, 2`: mean 4, population variance 4. With `eps = 0` and `alpha * v_th = 1`
    /// both channels normalise to exactly `+/-1`, so the outputs are exactly `gain +/- shift`.
    #[test]
    fn tdbn_scales_and_shifts_each_channel_by_its_own_parameter() {
        let mut bn = TdBn::new(2, 1.0).expect("valid");
        bn.eps = 0.0;
        bn.gain = vec![2.0, 3.0];
        bn.shift = vec![0.5, -0.5];
        let frames = vec![t(2, 1, 2, &[1., -1., 6., 2.])];
        let out = bn.train_forward(&frames).expect("valid");
        assert_eq!(out[0].data, vec![2.5, -1.5, 2.5, -3.5]);
        // Channel 1 under channel 0's gain would be 1.5 and -2.5; under channel 0's shift, 3.5 and
        // -2.5. Neither is in the vector above.
        assert_ne!(out[0].data[2], 1.5);
        assert_ne!(out[0].data[2], 3.5);

        // The same per-channel parameters at inference, against running statistics set by hand so
        // the two paths are compared on the same numbers rather than on each other's output.
        let mut ev = TdBn::new(2, 1.0).expect("valid");
        ev.eps = 0.0;
        ev.gain = vec![2.0, 3.0];
        ev.shift = vec![0.5, -0.5];
        ev.running_mean = vec![0.0, 4.0];
        ev.running_var = vec![1.0, 4.0];
        let got = ev.eval_forward(&t(2, 1, 2, &[1., -1., 6., 2.])).expect("valid");
        assert_eq!(got.data, vec![2.5, -1.5, 2.5, -3.5]);
    }

    /// A map whose channel count is not the layer's is refused rather than read past the end of
    /// its own buffer.
    ///
    /// The `"tdBN input channels"` arm had no fixture: the truncated-input test reaches the LENGTH
    /// check one line further down instead, because a short `data` fails `dims_len` first. A map
    /// with the right length and the wrong shape — one channel of `1x4` handed to a two-channel
    /// layer — passes the length check and reaches this one.
    #[test]
    fn tdbn_refuses_a_map_whose_channel_count_is_not_its_own() {
        let bn = TdBn::new(2, 1.0).expect("valid");
        let one_channel = t(1, 1, 4, &[1., 2., 3., 4.]);
        assert_eq!(one_channel.data.len(), 4, "the length check cannot be what refuses this");
        assert!(matches!(
            bn.eval_forward(&one_channel),
            Err(ConvError::ShapeMismatch { what: "tdBN input channels", .. })
        ));
        // Three channels of the same total length is refused for the same reason, in the other
        // direction.
        let three = t(3, 1, 2, &[1., 2., 3., 4., 5., 6.]);
        assert!(matches!(
            bn.eval_forward(&three),
            Err(ConvError::ShapeMismatch { what: "tdBN input channels", .. })
        ));
        // And the shape it WAS built for is accepted, so the refusals are about the channel count.
        assert!(bn.eval_forward(&t(2, 1, 2, &[1., 2., 3., 4.])).is_ok());
    }

    /// A layer is handed an input of the wrong shape and refuses it by name.
    ///
    /// The `"spiking conv input"` arm had no fixture either. `a_layer_whose_public_spec_was_changed`
    /// reaches the UNITS guard, which is a different check three lines later, and
    /// `bad_inputs_are_refused_by_name` exercises `Conv2d::forward`'s channel check, which is
    /// inside the convolution. With this guard removed a `4x4` frame through a layer built for
    /// `3x3` returns `Ok` and a `3x3` map assembled from the wrong nine of its sixteen drives.
    #[test]
    fn a_spiking_layer_refuses_an_input_of_the_wrong_shape() {
        let spec = Conv2dSpec::same_padding(1, 1, 3, 3).expect("valid");
        let mut l = SpikingConv2d::new(Conv2d::zeros(spec).expect("valid"), 3, 3, Lif::default(), 1e-3)
            .expect("valid");
        assert!(l.step(&Tensor3::zeros(1, 3, 3).expect("valid"), ActivationKind::Spiking).is_ok());

        // Bigger in both axes: every downstream check still passes, so only this guard can refuse.
        assert!(matches!(
            l.step(&Tensor3::zeros(1, 4, 4).expect("valid"), ActivationKind::Spiking),
            Err(ConvError::ShapeMismatch { what: "spiking conv input", .. })
        ));
        // Smaller, and transposed, and with the wrong channel count — each named by the LAYER's
        // guard rather than by the convolution's, which is a different message with a different
        // shape pair in it.
        assert!(matches!(
            l.step(&Tensor3::zeros(1, 2, 2).expect("valid"), ActivationKind::Spiking),
            Err(ConvError::ShapeMismatch { what: "spiking conv input", .. })
        ));
        let mut wide = SpikingConv2d::new(Conv2d::zeros(spec).expect("valid"), 3, 5, Lif::default(), 1e-3)
            .expect("valid");
        assert!(matches!(
            wide.step(&Tensor3::zeros(1, 5, 3).expect("valid"), ActivationKind::Spiking),
            Err(ConvError::ShapeMismatch { what: "spiking conv input", .. })
        ));
        assert!(matches!(
            l.step(&Tensor3::zeros(2, 3, 3).expect("valid"), ActivationKind::Spiking),
            Err(ConvError::ShapeMismatch { what: "spiking conv input", .. })
        ));
    }

    /// A bias poisoned after construction is refused at the next step, and so are the block's
    /// second convolution's WEIGHTS.
    ///
    /// `a_weight_poisoned_after_construction_is_refused_at_the_next_step` poisons the layer's
    /// `weights` and the block's second `bias` — which leaves the layer's own `bias` check and the
    /// block's own `weights` check, the other diagonal of the same four, with no fixture at all.
    /// A `NaN` bias is the worse of the two: it reaches the accumulator as its STARTING value, so
    /// it poisons every output position whether or not a single input spike arrived.
    #[test]
    fn a_bias_poisoned_after_construction_is_refused_at_the_next_step() {
        let spec = Conv2dSpec::same_padding(1, 1, 3, 3).expect("valid");
        let mut l = SpikingConv2d::new(Conv2d::zeros(spec).expect("valid"), 3, 3, Lif::default(), 1e-3)
            .expect("valid");
        let x = Tensor3::zeros(1, 3, 3).expect("valid");
        assert!(l.step(&x, ActivationKind::Spiking).is_ok());
        l.conv.bias[0] = f64::NAN;
        assert!(matches!(
            l.step(&x, ActivationKind::Spiking),
            Err(ConvError::NonFinite { what: "conv bias", index: 0, .. })
        ));

        let mut b = ResidualBlock::new(
            Conv2d::zeros(spec).expect("valid"),
            Conv2d::zeros(spec).expect("valid"),
            3,
            3,
            Lif::default(),
            1e-3,
            ResidualStyle::SewAdd,
            0.0,
        )
        .expect("valid");
        assert!(b.step(&x, ActivationKind::Spiking).is_ok());
        b.second.weights[4] = f64::NAN;
        assert!(matches!(
            b.step(&x, ActivationKind::Spiking),
            Err(ConvError::NonFinite { what: "conv weights", index: 4, .. })
        ));
        // The weights are checked BEFORE the bias, so a block with both poisoned names the
        // weights — which is what makes the assertion above about the weight check and not about
        // whichever check happens to be reached.
        b.second.bias[0] = f64::NAN;
        assert!(matches!(
            b.step(&x, ActivationKind::Spiking),
            Err(ConvError::NonFinite { what: "conv weights", .. })
        ));
    }

    /// ⛔ THE `SEW` STRUCTURE ITSELF: the second convolution reads the FIRST STAGE'S SPIKES, not
    /// the block's input. `out = g(SN(F(x)), x)`, and `F` is two convolutions deep.
    ///
    /// Not one existing fixture could tell the two apart. In `a_live_residual_block...` the first
    /// stage's spikes EQUAL the input — every driven unit fires, so `s1 == x` element for element
    /// — and in every other block fixture the second kernel is all zeros, where `convolve(x)` and
    /// `convolve(&s1)` give the same drive AND the same effective count. Feeding the second stage
    /// the block's input rather than the branch's output is the difference between a residual
    /// block and two independent convolutions sharing an adder, and it was invisible.
    ///
    /// The first convolution here is an INVERTER: weight `-20 mV`, bias `+20 mV`, so a unit whose
    /// input spiked is driven by exactly `0` and one whose input was silent is driven by `20 mV`
    /// and fires. On `x = [1, 0, 1, 1]` the branch's first stage emits `s1 = [0, 1, 0, 0]`, which
    /// shares not one element with `x`.
    #[test]
    fn the_second_convolution_reads_the_first_stages_spikes_not_the_blocks_input() {
        let spec = Conv2dSpec::new(1, 1, 1, 1).expect("valid");
        let inverter = || Conv2d::new(spec, vec![-20e-3], vec![20e-3]).expect("valid");
        let hot = || Conv2d::new(spec, vec![20e-3], vec![0.0]).expect("valid");
        let x = t(1, 2, 2, &[1., 0., 1., 1.]);

        // The first stage, run on its own, so the fixture's premise is measured rather than
        // assumed: its spikes are the complement of the input, not a copy of it.
        let mut stage_one =
            SpikingConv2d::new(inverter(), 2, 2, Lif::default(), 1e-3).expect("valid");
        let s1 = stage_one.step(&x, ActivationKind::Spiking).expect("fits");
        assert_eq!(s1.data, vec![0., 1., 0., 0.]);
        assert_ne!(s1.data, x.data, "the two candidate inputs must differ, or this proves nothing");

        let mut b = ResidualBlock::new(
            inverter(),
            hot(),
            2,
            2,
            Lif::default(),
            1e-3,
            ResidualStyle::SewAdd,
            0.0,
        )
        .expect("shape preserving");
        let out = b.step(&x, ActivationKind::Spiking).expect("fits");
        // Branch spike where s1 fired, plus the shortcut: [0,1,0,0] + [1,0,1,1] = [1,1,1,1].
        // Fed the block's input instead, the branch would be [1,0,1,1] and the sum [2,0,2,2].
        assert_eq!(out.data, vec![1., 1., 1., 1.]);

        // The second stage's accounting says the same thing in integers: ONE of its four taps had
        // a non-zero activation, because `s1` carries one spike. Fed `x` it would be three.
        assert_eq!((b.ops.dense, b.ops.effective_acs), (4, 1));
        assert_eq!(b.ledger.syn_ops, 1);
        assert_eq!(b.ledger.syn_fetches, 1);
        assert_eq!(b.ledger.spikes_out, 1);
        assert_eq!((b.ledger.neuron_updates_driven, b.ledger.neuron_updates_idle), (1, 3));
        // The first stage saw three effective taps on the same input, which is the count the
        // second stage would have reported had it been handed `x`.
        assert_eq!(b.first.ops.effective_acs, 3);
        assert_eq!(b.first.ledger.spikes_out, 1);
    }

    /// The naive shortcut drives a unit only where the shortcut SPIKED — `g * x`, not `g`.
    ///
    /// `the_naive_residual_cannot_be_the_identity_even_with_a_tuned_gain` runs on a `1x1` map whose
    /// shortcut is `1.0` at every timestep, and `g * 1` is the same number as `g`. So dropping the
    /// shortcut from the product left the naive block firing everywhere on every timestep with
    /// every test green — a block that ignores its own input and calls itself a residual
    /// connection.
    ///
    /// Both convolutions are silent here, so the drive is the shortcut term alone: `20 mV` where
    /// the input spiked and exactly `0` where it did not, which is also the difference between a
    /// driven neuron update and an idle one.
    #[test]
    fn the_naive_shortcut_drives_only_the_units_whose_shortcut_spiked() {
        let spec = Conv2dSpec::new(1, 1, 1, 1).expect("valid");
        let mut b = ResidualBlock::new(
            Conv2d::zeros(spec).expect("valid"),
            Conv2d::zeros(spec).expect("valid"),
            2,
            2,
            Lif::default(),
            1e-3,
            ResidualStyle::Naive,
            20e-3,
        )
        .expect("shape preserving");
        let x = t(1, 2, 2, &[1., 0., 1., 1.]);
        let out = b.step(&x, ActivationKind::Spiking).expect("fits");
        assert_eq!(out.data, vec![1., 0., 1., 1.], "the silent shortcut leaves its unit silent");
        assert_eq!(b.ledger.spikes_out, 3);
        assert_eq!((b.ledger.neuron_updates_driven, b.ledger.neuron_updates_idle), (3, 1));
        // A gain of zero silences the whole block, which is the other end of the same product and
        // is not reachable from a drive that ignores the shortcut.
        let mut off = ResidualBlock::new(
            Conv2d::zeros(spec).expect("valid"),
            Conv2d::zeros(spec).expect("valid"),
            2,
            2,
            Lif::default(),
            1e-3,
            ResidualStyle::Naive,
            0.0,
        )
        .expect("shape preserving");
        assert_eq!(off.step(&x, ActivationKind::Spiking).expect("fits").data, vec![0.0; 4]);
        assert_eq!(off.ledger.neuron_updates_idle, 4);
    }

    /// The surrogate offset is measured FROM the threshold TO the membrane, and the sign of that
    /// subtraction is observable. The recorded argument for the swapped-operand mutation says every
    /// surrogate family here is an even function of its argument; [`SigmoidDeriv`] is not, in
    /// binary64. Its `logistic` branches on the sign of `z` deliberately — for `z >= 0` it is
    /// `1/(1 + exp(-z))`, which is EXACTLY 1.0 once `exp(-z) < 2^-53`, i.e. `z >= 53 ln 2 =
    /// 36.7368`, so `s * (1 - s)` is exactly 0.0 there; for `z < 0` it is `e/(1 + e)`, a small
    /// positive number. The algebraic identity `s(-z) = 1 - s(z)` holds as real numbers and is lost
    /// to rounding well before that: at `|z| = 0.2` the two sides are 0.24751657271185995 and
    /// 0.24751657271185998.
    ///
    /// MEASURED at a resting unit one full threshold below firing, `beta = 40`, so `z = -40.0`
    /// exactly: this module answers `1.6993417021166355e-16`, and with the subtraction reversed it
    /// answers exactly `0.0` — the dead-neuron failure the module's own documentation warns about,
    /// from a sign rather than from the physics.
    ///
    /// Why the suite could not see it: `the_surrogate_mask_is_exact_at_threshold_and_depends_on_the_scale`
    /// is the only fixture that calls `surrogate_mask`, and it passes a [`FastSigmoid`], whose
    /// backward is written on `x.abs()` and therefore cannot see the sign at all.
    #[test]
    fn the_surrogate_offset_runs_from_the_threshold_to_the_membrane_and_the_sign_is_visible() {
        let spec = Conv2dSpec::new(1, 1, 1, 1).expect("valid");
        let conv = Conv2d::zeros(spec).expect("valid");
        let layer = SpikingConv2d::new(conv, 1, 2, Lif::default(), 1e-3).expect("valid");
        let lif = Lif::default();
        let scale = lif.v_th - lif.v_rest;
        // The offset is exact: `v_rest - v_th` and `v_th - v_rest` are the same subtraction with
        // the operands swapped, so their quotient is -1.0 to the last bit, and `beta * -1.0` is
        // -40.0 to the last bit. No tolerance is needed anywhere below.
        assert_eq!((lif.v - lif.v_th) / scale, -1.0);
        let s = SigmoidDeriv::new(40.0).expect("a positive sharpness");
        let m = layer.surrogate_mask(&s, lif.v_th, scale).expect("valid");
        assert_eq!(m.data[0], s.backward(-1.0));
        assert_eq!(m.data[1], s.backward(-1.0));
        // The far side of the threshold is where this family saturates to an exact zero, and it is
        // the answer the reversed subtraction would give: 1.0 / (1.0 + exp(-40)) rounds to 1.0, so
        // beta * s * (1 - s) is beta * 1.0 * 0.0.
        assert_eq!(s.backward(1.0), 0.0, "the family is not saturating, so this fixture is blunt");
        assert!(
            m.data[0] > 0.0,
            "a unit one threshold below firing got no gradient at all: {}",
            m.data[0]
        );
        // And the two sides are not merely unequal in the last place: one is zero and the other is
        // 40 * exp(-40), which is what `SigmoidDeriv::logistic`'s negative branch computes exactly.
        assert_eq!(m.data[0], 40.0 * (-40.0_f64).exp() * (1.0 - (-40.0_f64).exp()));
    }

}
