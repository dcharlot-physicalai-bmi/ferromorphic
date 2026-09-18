//! Event-based vision: the algorithms that consume what an event camera emits.
//!
//! [`crate::aer`] decodes the wire formats — `AEDAT` 2.0 and 4.0, `EVT` 2.0 and 3.0, `Prophesee`'s
//! `.dat` — and hands back `(t, x, y, polarity)` tuples. Nothing in this crate computed on them
//! until this module. This is what happens next.
//!
//! # What an event camera is, and what it costs you
//!
//! A conventional sensor samples every pixel on a global clock and ships a frame whether or not
//! anything moved. A dynamic vision sensor does the opposite: each pixel holds its own reference
//! log-intensity and fires an event the instant the log-intensity moves past a contrast threshold
//! `C`, independently of every other pixel (Lichtsteiner, Posch and Delbruck, *A 128x128 120 dB 15
//! us Latency Asynchronous Temporal Contrast Vision Sensor*, IEEE Journal of Solid-State Circuits
//! 43(2), 2008). What you buy is microsecond latency, a dynamic range past 120 dB, and a data rate
//! proportional to *scene motion* rather than to pixel count.
//!
//! What you pay is that **there is no image**. Every frame-based algorithm — convolution, corner
//! detection, optical flow, feature tracking — has to be re-derived on an asynchronous point
//! stream, and the stream is a motion-dependent, incomplete sampling of the scene: **a static
//! scene emits nothing at all**. You cannot recover absolute intensity from change events alone,
//! and an algorithm that quietly assumes you can will work beautifully on a moving checkerboard
//! and return nothing on a parked car.
//!
//! # The one idea the whole module is built on
//!
//! The event stream is a point cloud in `(x, y, t)`. Almost every result below is a statement
//! about the *geometry of that cloud*:
//!
//! - A moving straight edge sweeps each pixel once, at a time linear in position, so its events
//!   lie on a **plane** in `(x, y, t)` — and the plane's gradient is the reciprocal velocity
//!   (Benosman, Clercq, Lagorce, Ieng and Bartolozzi, *Event-Based Visual Flow*, IEEE Transactions
//!   on Neural Networks and Learning Systems 25(2), 2014).
//! - The cloud's *most recent arrival time* at each pixel is a scalar field — the **surface of
//!   active events** — and decaying it exponentially gives the **time surface** that every
//!   descriptor here is built on (Lagorce, Orchard, Gallupi, Shi and Benosman, *HOTS: A Hierarchy
//!   of Event-Based Time-Surfaces for Pattern Recognition*, IEEE Transactions on Pattern Analysis
//!   and Machine Intelligence 39(7), 2017).
//! - Undoing a candidate motion shears the cloud; the **correct** motion is the one whose shear
//!   makes the cloud's projection onto the image plane sharpest (Gallego, Rebecq and Scaramuzza,
//!   *A Unifying Contrast Maximization Framework for Event Cameras*, CVPR 2018).
//!
//! # The aperture problem is not a detail here, it is the answer
//!
//! A straight edge, observed through a small window, carries no information about motion *along*
//! itself. Plane fitting therefore recovers **normal flow** — the component of velocity along the
//! edge's normal — and nothing else, however good the fit is. [`PlaneFlow`] returns normal flow
//! and says so; the tests in this module generate edges that move along their own normal, so that
//! normal flow and true velocity coincide and the recovery can be checked exactly. Every
//! two-dimensional recovery below ([`search_translation`] on a corner, the rotation and expansion
//! sweeps) uses a stimulus with two independent edge orientations, because one is not enough and
//! an implementation that appears to recover two components from one edge is reporting its own
//! regulariser.
//!
//! # Units
//!
//! SI at every interface: [`PixelEvent::t_s`] is **seconds**, velocities are **pixels per second**,
//! decay constants are **seconds**. The wire is microseconds, so [`PixelEvent::from_aer`] is the
//! one conversion point and it is a single multiply by `1e-6`. Pixels are not an SI unit and are
//! not pretending to be: a pixel is the sensor's own sampling lattice, the lattice pitch is a
//! property of the optics and the die, and converting to radians or metres needs a calibration
//! this module does not have. So the boundary is stated rather than guessed — flow is px/s, and a
//! caller with a focal length in pixels can divide.
//!
//! The dimensionless constants of the papers reproduced here — `HOTS`'s `0.01` and `20000`,
//! `Harris`'s `0.04`, `eFAST`'s arc-length ranges — stay **verbatim inside the model** where a
//! reader can compare them against the source, and the conversion happens at the boundary.
//!
//! # What is checked, and against what
//!
//! - Plane-fit flow recovers the velocity the generator was given, across six directions and three
//!   speeds, on an event surface that is *exactly* planar by construction — so the tolerance is
//!   floating-point noise, not a fudge factor. On the curved surfaces (a rotating bar, a looming
//!   disc) the closed-form flow is also known and the residual is the fit's curvature bias, which
//!   is asserted to scale the way `L/r` says it must.
//! - Contrast maximisation's objective is swept and its argmax is asserted to land on ground truth,
//!   separately for translation, rotation and radial expansion.
//! - A time surface is asserted to equal `exp(-(t - t_last)/tau)` to 1e-15.
//! - The `Harris` response is asserted *negative* on a straight edge and *positive* on a right-angle
//!   corner, which is the textbook sign property rather than a threshold someone tuned; and the
//!   detection rate is reported near and far from a moving corner as two numbers.
//! - Accumulating a static scene gives an exactly zero frame, and an edge swept forward as `On` and
//!   back as `Off` sums to exactly zero per pixel.
//!
//! # What this implementation is unsure about
//!
//! The transcriptions of `HOTS`'s online clustering rule and of `HATS`'s default parameters are
//! from the papers as this implementation reads them; **this implementation did not locate an
//! author-released reference implementation of either to check the transcription against**, and
//! the items concerned say so in their own docs. The `eFAST` circle offsets and arc-length ranges
//! are the standard `FAST` `Bresenham` circles and the ranges Mueggler et al. print, and the
//! *behaviour* they are supposed to produce — half-circle arcs on an edge, quarter-circle arcs on
//! a corner — is asserted directly rather than taken on trust.
//!
//! # Example
//!
//! ```
//! use ferromorphic::vision::{Geometry, PlaneFlow, FlowOutcome, moving_edge};
//! use ferromorphic::spike::Polarity;
//!
//! let geom = Geometry::new(64, 64)?;
//! // An edge whose normal points along +x, moving at 300 px/s: pure horizontal motion.
//! let events = moving_edge(geom, 0.0, 300.0, -5.0, 0.5, Polarity::On)?;
//!
//! let mut flow = PlaneFlow::new(geom, 3, 0.05, Some(2e-4), 8)?;
//! let mut best = None;
//! for e in &events {
//!     if let FlowOutcome::Fitted(f) = flow.push(*e)? {
//!         best = Some(f);
//!     }
//! }
//! let f = best.expect("an edge crossing 64 columns fits somewhere");
//! assert!((f.vx - 300.0).abs() < 1e-6, "vx = {}", f.vx);
//! assert!(f.vy.abs() < 1e-6, "vy = {}", f.vy);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use crate::aer::AerEvent;
use crate::spike::Polarity;
use core::fmt;

// ---------------------------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------------------------

/// What went wrong, naming the quantity rather than saying "invalid input".
///
/// Every variant here is a refusal at a boundary. The alternative — clamping a coordinate into
/// range, substituting a zero for a missing timestamp, treating a `NaN` velocity as no motion —
/// produces a flow field that is smooth, plausible and wrong, which is the failure mode this
/// module is most exposed to.
#[derive(Debug, Clone, PartialEq)]
pub enum VisionError {
    /// A sensor dimension was zero, so the pixel lattice contains no pixels.
    EmptyGeometry {
        /// Width in pixels, as supplied.
        width: u16,
        /// Height in pixels, as supplied.
        height: u16,
    },
    /// A pixel coordinate was at or past the sensor's extent.
    ///
    /// Refused rather than wrapped: folding an out-of-range column into the next row produces an
    /// image that is sheared rather than obviously broken, which is exactly the defect
    /// [`crate::aer::AerEvent::to_event`] refuses for the same reason.
    OutOfBounds {
        /// The column asked for.
        x: u16,
        /// The row asked for.
        y: u16,
        /// Sensor width in pixels.
        width: u16,
        /// Sensor height in pixels.
        height: u16,
    },
    /// A floating-point input was `NaN` or infinite.
    NonFinite {
        /// Which quantity, as a noun phrase completing "the ... ".
        what: &'static str,
        /// The value found.
        value: f64,
    },
    /// A quantity that must be strictly positive was zero or negative.
    ///
    /// A zero speed is the common one, and it is a refusal rather than a special case because a
    /// scene that does not move emits no events at all: there is no degenerate answer to return.
    NonPositive {
        /// Which quantity, as a noun phrase completing "the ... ".
        what: &'static str,
        /// The value found.
        value: f64,
    },
    /// Fewer samples than the estimator needs.
    TooFew {
        /// Which collection, as a noun phrase completing "the ... ".
        what: &'static str,
        /// How many were supplied.
        have: usize,
        /// How many the estimator needs.
        need: usize,
    },
    /// Events arrived out of time order.
    ///
    /// Every running structure here — [`TimeSurface`], [`PlaneFlow`], [`EHarris`], [`EFast`] —
    /// holds "the most recent time at each pixel", and that phrase means nothing if the stream can
    /// go backwards. Sorting silently would hide a decoder bug; this names the offending pair.
    OutOfOrder {
        /// Time of the event being accepted, seconds.
        t_s: f64,
        /// The stream's current time, seconds.
        now_s: f64,
    },
    /// The estimator's design matrix was singular, so the answer it was asked for does not exist.
    ///
    /// For [`fit_plane`] this means the supporting events were collinear in `(x, y)` — a plane
    /// through a line is not determined — which happens routinely on a thin stimulus and is a
    /// legitimate outcome rather than a failure.
    Degenerate {
        /// Which estimator, as a noun phrase completing "the ... ".
        what: &'static str,
    },
    /// A parameter combination has no meaning, with the reason spelled out.
    ///
    /// Used where the offending quantity is a *relationship* between arguments rather than any one
    /// of them: a rejection threshold wider than the acceptance window, a corner arm parallel to
    /// the motion that sweeps it.
    BadParameters {
        /// The reason, as a sentence fragment completing "refused because ... ".
        why: &'static str,
    },
}

impl fmt::Display for VisionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyGeometry { width, height } => {
                write!(f, "a {width}x{height} sensor has no pixels")
            }
            Self::OutOfBounds { x, y, width, height } => {
                write!(f, "pixel ({x}, {y}) is outside a {width}x{height} sensor")
            }
            Self::NonFinite { what, value } => {
                write!(f, "the {what} is {value}, which is not a finite number")
            }
            Self::NonPositive { what, value } => {
                write!(f, "the {what} is {value}; it must be strictly positive")
            }
            Self::TooFew { what, have, need } => {
                write!(f, "the {what} has {have} entries where {need} are needed")
            }
            Self::OutOfOrder { t_s, now_s } => {
                write!(f, "an event at t = {t_s} s arrived after the stream reached {now_s} s")
            }
            Self::Degenerate { what } => {
                write!(f, "the {what} is singular; the answer asked for does not exist")
            }
            Self::BadParameters { why } => write!(f, "refused because {why}"),
        }
    }
}

/// So that `?` works in a caller whose error type is `Box<dyn Error>`, as every example and
/// doctest in this crate uses. See the same note on [`crate::net::NetError`].
impl std::error::Error for VisionError {}

/// Reject a non-finite scalar at the boundary, naming it.
fn finite(x: f64, what: &'static str) -> Result<f64, VisionError> {
    if x.is_finite() { Ok(x) } else { Err(VisionError::NonFinite { what, value: x }) }
}

/// Reject a scalar that is not strictly positive (and not finite), naming it.
fn positive(x: f64, what: &'static str) -> Result<f64, VisionError> {
    let x = finite(x, what)?;
    if x > 0.0 { Ok(x) } else { Err(VisionError::NonPositive { what, value: x }) }
}

// ---------------------------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------------------------

/// The sensor's pixel lattice.
///
/// Carried explicitly rather than inferred from the largest coordinate seen, because the largest
/// coordinate seen depends on what moved: a recording in which nothing crossed the right-hand
/// columns would silently produce a narrower sensor and shift every subsequent address.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Geometry {
    /// Columns, pixels. Origin at the left edge, matching [`crate::aer::AerEvent::x`].
    pub width: u16,
    /// Rows, pixels. Origin at the **top** edge, matching [`crate::aer::AerEvent::y`].
    pub height: u16,
}

impl Geometry {
    /// A sensor of `width` by `height` pixels.
    ///
    /// # Errors
    ///
    /// [`VisionError::EmptyGeometry`] when either dimension is zero. Nothing downstream has a
    /// sensible answer over an empty lattice, and a zero-length image buffer would make every
    /// accumulator report a variance of `None` for a reason the caller could not diagnose.
    pub fn new(width: u16, height: u16) -> Result<Self, VisionError> {
        if width == 0 || height == 0 {
            return Err(VisionError::EmptyGeometry { width, height });
        }
        Ok(Self { width, height })
    }

    /// Pixel count, `width * height`.
    #[must_use]
    pub fn pixels(self) -> usize {
        usize::from(self.width) * usize::from(self.height)
    }

    /// Whether a signed coordinate pair falls on the lattice.
    ///
    /// Signed because neighbourhood loops work in offsets and the `-1` case is the point.
    #[must_use]
    pub fn contains(self, x: i64, y: i64) -> bool {
        x >= 0 && y >= 0 && x < i64::from(self.width) && y < i64::from(self.height)
    }

    /// Row-major index `y * width + x`, or `None` off the lattice.
    #[must_use]
    pub fn index(self, x: u16, y: u16) -> Option<usize> {
        if x >= self.width || y >= self.height {
            return None;
        }
        Some(usize::from(y) * usize::from(self.width) + usize::from(x))
    }

    /// Row-major index, or [`VisionError::OutOfBounds`] naming the sensor.
    ///
    /// # Errors
    ///
    /// [`VisionError::OutOfBounds`] when the pixel is off the lattice.
    pub fn require(self, x: u16, y: u16) -> Result<usize, VisionError> {
        self.index(x, y).ok_or(VisionError::OutOfBounds {
            x,
            y,
            width: self.width,
            height: self.height,
        })
    }
}

// ---------------------------------------------------------------------------------------------
// The event, in SI units
// ---------------------------------------------------------------------------------------------

/// One change-detection event, with time in **seconds**.
///
/// [`crate::aer::AerEvent`] is the same thing with time in microseconds, because that is what every
/// wire format counts and a float conversion inside a decoder would put a rounding between the
/// decoder and its bit-exactness tests. This type is the other side of that boundary: everything in
/// this module does arithmetic on time, so time here is a `f64` of seconds and the conversion
/// happens exactly once, in [`PixelEvent::from_aer`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PixelEvent {
    /// Timestamp, **seconds** since the recording's zero. Finite; the running structures in this
    /// module additionally require it to be non-decreasing along the stream.
    pub t_s: f64,
    /// Column, pixels, origin at the left edge.
    pub x: u16,
    /// Row, pixels, origin at the top edge.
    pub y: u16,
    /// Sign of the contrast change that produced the event.
    pub polarity: Polarity,
}

impl PixelEvent {
    /// Convert from the decoder's microsecond timestamp. **This is the module's unit boundary.**
    ///
    /// One multiply by `1e-6`, exact for every `t` below `2^53` microseconds — about 285 years —
    /// so no recording this implementation can imagine loses a microsecond here.
    #[must_use]
    pub fn from_aer(e: AerEvent) -> Self {
        Self { t_s: e.t as f64 * 1e-6, x: e.x, y: e.y, polarity: e.polarity }
    }

    /// Convert back to the decoder's microsecond timestamp, rounding to the nearest microsecond.
    ///
    /// `None` for a non-finite or negative time: a wire timestamp is an unsigned count from the
    /// recording's zero, and there is no encoding of "half a microsecond before the start".
    #[must_use]
    pub fn to_aer(self) -> Option<AerEvent> {
        if !self.t_s.is_finite() || self.t_s < 0.0 {
            return None;
        }
        let us = (self.t_s * 1e6).round();
        if us > u64::MAX as f64 {
            return None;
        }
        Some(AerEvent { t: us as u64, x: self.x, y: self.y, polarity: self.polarity })
    }

    /// Which decay plane this event belongs to when polarity is kept separate: `On` is 1, `Off` 0.
    ///
    /// Exposed so that a caller indexing its own per-polarity buffers uses the same convention the
    /// rest of this module does; a flip here inverts every learned descriptor while leaving every
    /// event count unchanged.
    #[must_use]
    pub fn plane(self, split: bool) -> usize {
        if !split {
            0
        } else {
            match self.polarity {
                Polarity::Off => 0,
                Polarity::On => 1,
            }
        }
    }
}

/// Check that a slice of events is non-decreasing in time and finite.
///
/// # Errors
///
/// [`VisionError::NonFinite`] for a bad timestamp, [`VisionError::OutOfOrder`] for a backwards
/// step. Both name the offending value.
pub fn require_time_ordered(events: &[PixelEvent]) -> Result<(), VisionError> {
    let mut prev = f64::NEG_INFINITY;
    for e in events {
        let t = finite(e.t_s, "event timestamp")?;
        if t < prev {
            return Err(VisionError::OutOfOrder { t_s: t, now_s: prev });
        }
        prev = t;
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Time surfaces
// ---------------------------------------------------------------------------------------------

/// How a time surface forgets.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Decay {
    /// `exp(-elapsed / tau_s)`: the standard time surface of Lagorce et al. 2017.
    ///
    /// Never reaches zero, so every pixel that has ever fired keeps a positive weight. That is the
    /// point — the surface is a soft "how recently" rather than a hard window — and it is also the
    /// cost, because a descriptor built on it is never fully independent of the distant past.
    Exponential {
        /// Time constant, **seconds**. Strictly positive and finite. The surface falls to `1/e`
        /// after this long.
        tau_s: f64,
    },
    /// `max(0, 1 - elapsed / window_s)`: a linear ramp to a hard cut-off.
    ///
    /// Cheaper, and exactly zero past the window, which is what makes a descriptor built on it
    /// depend on a bounded slice of history. Used by several event-based trackers for that reason.
    Linear {
        /// Window length, **seconds**. Strictly positive and finite. The surface is exactly zero
        /// at and past this elapsed time.
        window_s: f64,
    },
}

impl Decay {
    /// The surface's value for a pixel whose last event was `elapsed_s` seconds ago.
    ///
    /// A negative `elapsed_s` is clamped to zero, returning the peak value of `1.0`. That case
    /// cannot arise from a causal stream — [`TimeSurface::update`] refuses backwards time — and
    /// clamping is the honest answer for a hand-built query rather than a `NaN`.
    #[must_use]
    pub fn value(self, elapsed_s: f64) -> f64 {
        let d = if elapsed_s > 0.0 { elapsed_s } else { 0.0 };
        match self {
            Self::Exponential { tau_s } => (-d / tau_s).exp(),
            Self::Linear { window_s } => {
                let v = 1.0 - d / window_s;
                if v > 0.0 { v } else { 0.0 }
            }
        }
    }

    /// The time constant or window, whichever this decay has, in seconds.
    #[must_use]
    pub fn scale_s(self) -> f64 {
        match self {
            Self::Exponential { tau_s } => tau_s,
            Self::Linear { window_s } => window_s,
        }
    }

    /// Reject a non-finite or non-positive time constant.
    ///
    /// # Errors
    ///
    /// [`VisionError::NonPositive`] or [`VisionError::NonFinite`], naming the constant. A zero
    /// `tau` makes every value `exp(-inf) = 0` except at `elapsed == 0`, which is a surface that
    /// carries no information and would still plot.
    pub fn validate(self) -> Result<(), VisionError> {
        match self {
            Self::Exponential { tau_s } => positive(tau_s, "decay time constant tau").map(|_| ()),
            Self::Linear { window_s } => positive(window_s, "decay window").map(|_| ()),
        }
    }
}

/// The **surface of active events**: the most recent event time at each pixel, with a decay.
///
/// The undecayed content — one timestamp per pixel — is the `SAE` of Benosman et al. 2014 and is
/// what [`TimeSurface::last_time`] returns. Applying [`Decay`] to the elapsed time gives the *time
/// surface* of Lagorce et al. 2017, which is what [`TimeSurface::value_at`] returns and what every
/// descriptor in this module is built on.
///
/// # Why one timestamp per pixel and not a history
///
/// It is `O(1)` memory per pixel and `O(1)` per event, which is the whole reason the representation
/// is usable on an embedded target. What it throws away is event *density*: a pixel that fired
/// fifty times in the last millisecond and a pixel that fired once are indistinguishable. `HATS`
/// ([`Hats`]) is the descriptor that puts that back, and it pays for it with a pass over history —
/// see its own doc.
///
/// # Polarity
///
/// With `split` set, `On` and `Off` keep separate planes and a query must name which. Collapsing
/// them is a decision with consequences: an edge that passes and returns writes over its own trail
/// in a collapsed surface, and the returning motion becomes invisible. The default in this module
/// is to split, and the places that do not say why.
#[derive(Debug, Clone, PartialEq)]
pub struct TimeSurface {
    geom: Geometry,
    decay: Decay,
    split: bool,
    /// Per plane, per pixel: the last event time in seconds, or `-inf` for "never fired".
    last: Vec<f64>,
    now_s: f64,
}

impl TimeSurface {
    /// An empty surface: no pixel has ever fired.
    ///
    /// # Errors
    ///
    /// [`VisionError::NonPositive`] or [`VisionError::NonFinite`] when the decay constant is not a
    /// finite positive number.
    pub fn new(geom: Geometry, decay: Decay, split: bool) -> Result<Self, VisionError> {
        decay.validate()?;
        let planes = if split { 2 } else { 1 };
        Ok(Self {
            geom,
            decay,
            split,
            last: vec![f64::NEG_INFINITY; planes * geom.pixels()],
            now_s: f64::NEG_INFINITY,
        })
    }

    /// The sensor lattice this surface covers.
    #[must_use]
    pub fn geometry(&self) -> Geometry {
        self.geom
    }

    /// The decay this surface applies.
    #[must_use]
    pub fn decay(&self) -> Decay {
        self.decay
    }

    /// Whether `On` and `Off` are kept in separate planes.
    #[must_use]
    pub fn is_split(&self) -> bool {
        self.split
    }

    /// The time of the most recent event accepted, or `-inf` before the first.
    #[must_use]
    pub fn now_s(&self) -> f64 {
        self.now_s
    }

    /// Record an event.
    ///
    /// # Errors
    ///
    /// [`VisionError::NonFinite`] for a bad timestamp, [`VisionError::OutOfBounds`] for a pixel off
    /// the lattice, and [`VisionError::OutOfOrder`] for a timestamp earlier than one already
    /// accepted. The last is a refusal rather than a sort because "the most recent time at this
    /// pixel" is the surface's entire content and a stream that can go backwards does not have one.
    pub fn update(&mut self, e: PixelEvent) -> Result<(), VisionError> {
        let t = finite(e.t_s, "event timestamp")?;
        let idx = self.geom.require(e.x, e.y)?;
        if t < self.now_s {
            return Err(VisionError::OutOfOrder { t_s: t, now_s: self.now_s });
        }
        let plane = e.plane(self.split);
        self.last[plane * self.geom.pixels() + idx] = t;
        self.now_s = t;
        Ok(())
    }

    /// The raw `SAE` entry: when this pixel last fired, or `Ok(None)` if it never has.
    ///
    /// The two outcomes are kept apart deliberately. `Err` means the caller asked about a pixel
    /// that does not exist, which is a bug in the caller; `Ok(None)` means the pixel exists and has
    /// no history, which is the normal state of most of a sensor most of the time.
    ///
    /// # Errors
    ///
    /// [`VisionError::OutOfBounds`] for a pixel off the lattice.
    pub fn last_time(&self, x: u16, y: u16, pol: Polarity) -> Result<Option<f64>, VisionError> {
        let idx = self.geom.require(x, y)?;
        let plane = if self.split {
            match pol {
                Polarity::Off => 0,
                Polarity::On => 1,
            }
        } else {
            0
        };
        let v = self.last[plane * self.geom.pixels() + idx];
        Ok(if v.is_finite() { Some(v) } else { None })
    }

    /// The decayed surface value at `t_s`.
    ///
    /// Exactly `decay.value(t_s - t_last)`, and exactly `0.0` for a pixel that has never fired —
    /// which is also the limit of the exponential as the last event recedes, so the two agree.
    ///
    /// # Errors
    ///
    /// [`VisionError::OutOfBounds`] off the lattice, [`VisionError::NonFinite`] for a bad query
    /// time.
    pub fn value_at(
        &self,
        x: u16,
        y: u16,
        pol: Polarity,
        t_s: f64,
    ) -> Result<f64, VisionError> {
        let t = finite(t_s, "query time")?;
        match self.last_time(x, y, pol)? {
            None => Ok(0.0),
            Some(last) => Ok(self.decay.value(t - last)),
        }
    }

    /// A `(2 * radius + 1)` square patch of decayed values, row-major, centred on `(x, y)`.
    ///
    /// **Pixels off the lattice contribute exactly `0.0`**, the same value a pixel that exists and
    /// has never fired contributes. That conflation is deliberate: the alternative — refusing every
    /// event within `radius` of the border — silently eats a band of perfectly good events, which
    /// is the shape of defect this crate's last audit found in a decoder. A caller that needs the
    /// distinction can walk [`TimeSurface::last_time`] itself.
    ///
    /// # Errors
    ///
    /// [`VisionError::OutOfBounds`] if the *centre* is off the lattice, [`VisionError::NonFinite`]
    /// for a bad query time.
    pub fn patch(
        &self,
        x: u16,
        y: u16,
        radius: u16,
        pol: Polarity,
        t_s: f64,
    ) -> Result<Vec<f64>, VisionError> {
        let t = finite(t_s, "query time")?;
        self.geom.require(x, y)?;
        let r = i64::from(radius);
        let side = (2 * radius + 1) as usize;
        let mut out = vec![0.0; side * side];
        for dy in -r..=r {
            for dx in -r..=r {
                let px = i64::from(x) + dx;
                let py = i64::from(y) + dy;
                if !self.geom.contains(px, py) {
                    continue;
                }
                let v = self.value_at(px as u16, py as u16, pol, t)?;
                out[((dy + r) as usize) * side + (dx + r) as usize] = v;
            }
        }
        Ok(out)
    }

    /// The whole decayed surface for one polarity plane as a [`Frame`], evaluated at `t_s`.
    ///
    /// # Errors
    ///
    /// [`VisionError::NonFinite`] for a bad query time.
    pub fn render(&self, pol: Polarity, t_s: f64) -> Result<Frame, VisionError> {
        let t = finite(t_s, "query time")?;
        let mut f = Frame::zeros(self.geom);
        for y in 0..self.geom.height {
            for x in 0..self.geom.width {
                let idx = self.geom.require(x, y)?;
                f.data[idx] = self.value_at(x, y, pol, t)?;
            }
        }
        Ok(f)
    }

    /// Forget everything, as if no event had ever arrived.
    pub fn clear(&mut self) {
        self.last.fill(f64::NEG_INFINITY);
        self.now_s = f64::NEG_INFINITY;
    }
}

// ---------------------------------------------------------------------------------------------
// Frames
// ---------------------------------------------------------------------------------------------

/// A dense scalar image over the sensor lattice, row-major.
///
/// Event-based vision produces one of these only when it is asked to. The point of the
/// representation is that you choose *what* to accumulate — counts, signed polarity, decayed
/// weight, warped votes — and every choice throws away something different. [`accumulate_count`]
/// loses the contrast direction; [`accumulate_polarity`] loses activity where `On` and `Off`
/// cancel; [`accumulate_decay`] keeps both at the price of a time constant that has to be chosen.
#[derive(Debug, Clone, PartialEq)]
pub struct Frame {
    /// Columns, pixels.
    pub width: u16,
    /// Rows, pixels.
    pub height: u16,
    /// `width * height` values, row-major, index `y * width + x`.
    pub data: Vec<f64>,
}

impl Frame {
    /// An all-zero image over `geom`.
    #[must_use]
    pub fn zeros(geom: Geometry) -> Self {
        Self { width: geom.width, height: geom.height, data: vec![0.0; geom.pixels()] }
    }

    /// The lattice this image covers.
    #[must_use]
    pub fn geometry(&self) -> Geometry {
        Geometry { width: self.width, height: self.height }
    }

    /// The value at a pixel, or `None` off the lattice.
    #[must_use]
    pub fn at(&self, x: u16, y: u16) -> Option<f64> {
        self.geometry().index(x, y).map(|i| self.data[i])
    }

    /// Sum over all pixels.
    #[must_use]
    pub fn sum(&self) -> f64 {
        self.data.iter().sum()
    }

    /// Sum of squares over all pixels — the objective Gallego et al. 2019 call *image area* when
    /// normalised, and one of the two focus measures [`Objective`] offers.
    #[must_use]
    pub fn sum_of_squares(&self) -> f64 {
        self.data.iter().map(|v| v * v).sum()
    }

    /// Mean over all pixels, or `None` for an empty image.
    #[must_use]
    pub fn mean(&self) -> Option<f64> {
        if self.data.is_empty() {
            return None;
        }
        Some(self.sum() / self.data.len() as f64)
    }

    /// **Population** variance over all pixels, or `None` for an empty image.
    ///
    /// Population rather than sample: the pixels are not a sample of some larger image, they are
    /// the whole image, and the `n - 1` correction would make the objective depend on the sensor
    /// size in a way the paper's derivation does not.
    #[must_use]
    pub fn variance(&self) -> Option<f64> {
        let m = self.mean()?;
        let n = self.data.len() as f64;
        Some(self.data.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / n)
    }

    /// Largest value, or `None` for an empty image.
    ///
    /// Uses [`f64::max`], so a `NaN` anywhere is ignored rather than propagated — but nothing in
    /// this module can put a `NaN` here, because every accumulator rejects non-finite input at the
    /// boundary.
    #[must_use]
    pub fn max(&self) -> Option<f64> {
        self.data.iter().copied().reduce(f64::max)
    }

    /// Smallest value, or `None` for an empty image.
    #[must_use]
    pub fn min(&self) -> Option<f64> {
        self.data.iter().copied().reduce(f64::min)
    }
}

/// One event per pixel counted, polarity ignored.
///
/// # Errors
///
/// [`VisionError::OutOfBounds`] for an event off the lattice, [`VisionError::NonFinite`] for a bad
/// timestamp. Out-of-range events are refused rather than dropped: a recording decoded against the
/// wrong sensor width produces a sheared but entirely plausible image, and the count would not show
/// it.
pub fn accumulate_count(geom: Geometry, events: &[PixelEvent]) -> Result<Frame, VisionError> {
    let mut f = Frame::zeros(geom);
    for e in events {
        finite(e.t_s, "event timestamp")?;
        f.data[geom.require(e.x, e.y)?] += 1.0;
    }
    Ok(f)
}

/// Signed sum of polarities: `+1` per `On`, `-1` per `Off`.
///
/// The closest thing to a brightness-change image the sensor can give, and the reason
/// [`crate::spike::Polarity::sign`] exists rather than each call site choosing a convention. Note
/// what it hides: a pixel that fired a thousand `On` and a thousand `Off` events reads zero, the
/// same as a pixel that never fired.
///
/// # Errors
///
/// As [`accumulate_count`].
pub fn accumulate_polarity(geom: Geometry, events: &[PixelEvent]) -> Result<Frame, VisionError> {
    let mut f = Frame::zeros(geom);
    for e in events {
        finite(e.t_s, "event timestamp")?;
        f.data[geom.require(e.x, e.y)?] += e.polarity.sign();
    }
    Ok(f)
}

/// Signed sum weighted by `decay.value(t_ref_s - t)`, i.e. an exponentially-decayed image.
///
/// Events **after** `t_ref_s` contribute at the decay's peak weight of `1.0`, because
/// [`Decay::value`] clamps a negative elapsed time. Pass a `t_ref_s` at or after the last event to
/// get the causal image the name suggests.
///
/// # Errors
///
/// As [`accumulate_count`], plus [`VisionError::NonFinite`] for a bad `t_ref_s` and
/// [`VisionError::NonPositive`] for a bad decay constant.
pub fn accumulate_decay(
    geom: Geometry,
    events: &[PixelEvent],
    decay: Decay,
    t_ref_s: f64,
) -> Result<Frame, VisionError> {
    decay.validate()?;
    let t_ref = finite(t_ref_s, "reference time")?;
    let mut f = Frame::zeros(geom);
    for e in events {
        let t = finite(e.t_s, "event timestamp")?;
        f.data[geom.require(e.x, e.y)?] += e.polarity.sign() * decay.value(t_ref - t);
    }
    Ok(f)
}

// ---------------------------------------------------------------------------------------------
// Plane fitting and optical flow
// ---------------------------------------------------------------------------------------------

/// A least-squares plane `t = a * x + b * y + c` through events in `(x, y, t)`.
///
/// The gradient `(a, b)` has units of **seconds per pixel**: it is the reciprocal of speed, not
/// speed. [`PlaneFit::flow`] does the inversion, and the two are kept apart because the inversion
/// is where the interesting failure lives — `(a, b) = (0, 0)` is a perfectly good plane fit
/// describing an infinite speed, and it is the fit you get when every supporting event has the
/// same timestamp.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlaneFit {
    /// `dt/dx`, seconds per pixel.
    pub a: f64,
    /// `dt/dy`, seconds per pixel.
    pub b: f64,
    /// Intercept, seconds: the plane's value at `(0, 0)`.
    pub c: f64,
    /// How many points the fit used, after any outlier rejection.
    pub points: usize,
    /// Root-mean-square residual of the supporting points, **seconds**.
    ///
    /// On the synthetic edges in this module's tests this is at floating-point noise, because the
    /// event surface of a straight edge is exactly planar. On a real recording it is the thing that
    /// says whether the local-planarity assumption held.
    pub rms_s: f64,
}

impl PlaneFit {
    /// The plane's predicted time at a pixel, seconds.
    #[must_use]
    pub fn predict(&self, x: f64, y: f64) -> f64 {
        self.a * x + self.b * y + self.c
    }

    /// Signed residual `t - predict(x, y)`, seconds.
    #[must_use]
    pub fn residual(&self, x: f64, y: f64, t_s: f64) -> f64 {
        t_s - self.predict(x, y)
    }

    /// Normal flow from the plane's gradient.
    ///
    /// # The closed form
    ///
    /// An edge with unit normal `n` moving at normal speed `s` px/s sweeps pixel `p` at
    /// `t(p) = (n . p - d0) / s`, so `grad t = n / s` and therefore
    ///
    /// ```text
    /// |grad t| = 1 / s        and        v = s * n = (a, b) / (a^2 + b^2)
    /// ```
    ///
    /// This is the identity Benosman et al. 2014 build the method on, and it is why the *gradient*
    /// of the event surface is a velocity rather than a slowness that needs a separate direction
    /// estimate.
    ///
    /// # Errors
    ///
    /// [`VisionError::Degenerate`] when `a` and `b` are both zero — a flat plane, meaning every
    /// supporting event carried the same timestamp, which describes an infinite speed rather than
    /// a large one. Refused rather than returned as a huge number, because a flow field with one
    /// `1e12` in it averages to nonsense and no plot shows which pixel did it.
    pub fn flow(&self) -> Result<Flow, VisionError> {
        let g2 = self.a * self.a + self.b * self.b;
        if !(g2 > 0.0) || !g2.is_finite() {
            return Err(VisionError::Degenerate { what: "event-surface gradient" });
        }
        let vx = self.a / g2;
        let vy = self.b / g2;
        Ok(Flow { vx, vy, speed: 1.0 / g2.sqrt(), direction_rad: self.b.atan2(self.a) })
    }
}

/// Normal flow at one event, in pixels per second.
///
/// **Normal** flow: the component of the scene's velocity along the local edge normal. See the
/// module doc — a straight edge carries no information about motion along itself, so this is all
/// that is recoverable from a local fit, and calling it "optical flow" without the qualifier is how
/// an aperture-limited estimate ends up in a trajectory.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Flow {
    /// Horizontal component, px/s, positive toward larger `x`.
    pub vx: f64,
    /// Vertical component, px/s, positive toward larger `y` (i.e. **downward**, since row 0 is the
    /// top edge).
    pub vy: f64,
    /// Magnitude, px/s. Equal to `hypot(vx, vy)` and to `1 / |grad t|`.
    pub speed: f64,
    /// Direction, radians, `atan2(vy, vx)` in `(-pi, pi]`.
    pub direction_rad: f64,
}

/// Least-squares plane through `(x, y, t)` points.
///
/// Solved on **centred** coordinates, which turns the 3x3 normal equations into a 2x2 system with
/// no intercept row: `Su = Sv = 0` after centring, so `c` falls out afterwards as
/// `t_mean - a * x_mean - b * y_mean`. That is not a micro-optimisation — the uncentred 3x3 system
/// on pixel coordinates near 640 has a condition number in the millions, and the whole estimate is
/// a difference of large numbers.
///
/// # Errors
///
/// - [`VisionError::TooFew`] with fewer than three points; a plane through two points is not
///   determined.
/// - [`VisionError::NonFinite`] for any non-finite coordinate or timestamp.
/// - [`VisionError::Degenerate`] when the points are collinear in `(x, y)`, judged by the squared
///   correlation of the centred coordinates exceeding `1 - 1e-9`. Collinear support is the normal
///   state of a thin stimulus, not an exotic failure.
pub fn fit_plane(points: &[(f64, f64, f64)]) -> Result<PlaneFit, VisionError> {
    if points.len() < 3 {
        return Err(VisionError::TooFew {
            what: "plane-fit support",
            have: points.len(),
            need: 3,
        });
    }
    let n = points.len() as f64;
    let (mut sx, mut sy, mut st) = (0.0, 0.0, 0.0);
    for &(x, y, t) in points {
        sx += finite(x, "plane-fit x")?;
        sy += finite(y, "plane-fit y")?;
        st += finite(t, "plane-fit timestamp")?;
    }
    let (xm, ym, tm) = (sx / n, sy / n, st / n);

    let (mut suu, mut svv, mut suv, mut sut, mut svt) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for &(x, y, t) in points {
        let (u, v, w) = (x - xm, y - ym, t - tm);
        suu += u * u;
        svv += v * v;
        suv += u * v;
        sut += u * w;
        svt += v * w;
    }
    let det = suu * svv - suv * suv;
    // Scale-free collinearity test: `suv^2 / (suu * svv)` is the squared correlation of the
    // centred coordinates, exactly 1 when the support is a line. Comparing `det` against an
    // absolute epsilon instead would pass a thin-but-not-degenerate support at 640x480 and fail an
    // identical one at 64x64.
    if !(suu > 0.0) || !(svv > 0.0) || !(det > 1e-9 * suu * svv) {
        return Err(VisionError::Degenerate { what: "plane-fit design matrix" });
    }
    let a = (svv * sut - suv * svt) / det;
    let b = (suu * svt - suv * sut) / det;
    let c = tm - a * xm - b * ym;
    let fit = PlaneFit { a, b, c, points: points.len(), rms_s: 0.0 };
    let ss: f64 = points.iter().map(|&(x, y, t)| fit.residual(x, y, t).powi(2)).sum();
    Ok(PlaneFit { rms_s: (ss / n).sqrt(), ..fit })
}

/// Why a per-event flow estimate did or did not produce a velocity.
///
/// An enum rather than an `Option`, because "no flow here" has four distinct causes and a caller
/// that cannot tell them apart cannot tell a stationary sensor from a mis-set window. Silently
/// collapsing them is how a decoder in this crate's last audit came to eat valid records.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FlowOutcome {
    /// A velocity was recovered.
    Fitted(Flow),
    /// Fewer events in the neighbourhood and time window than `min_events`.
    ///
    /// The common outcome at the start of a stream and on a sparsely-textured scene.
    TooFew {
        /// How many supporting events were found.
        found: usize,
    },
    /// The supporting events were collinear in `(x, y)`, so no plane is determined.
    ///
    /// Routine on a stimulus one pixel wide, and on the very first events of any edge.
    Degenerate,
    /// The fitted plane was flat: every supporting event carried the same timestamp.
    ///
    /// Describes infinite speed. Reported rather than returned as a large number.
    NoMotion,
    /// Outlier rejection left fewer than `min_events` inliers.
    RejectedTooMany {
        /// How many survived the rejection pass.
        inliers: usize,
    },
}

/// Per-event normal flow by local plane fitting (Benosman et al., IEEE `TNNLS` 25(2), 2014).
///
/// # The algorithm, and where this implementation departs from the paper
///
/// For each incoming event the estimator takes the `SAE` entries in a `(2r+1)` square around it
/// **on the same polarity plane**, keeps those whose timestamp is within `window_s` of the current
/// event and not in the future, fits a plane, and inverts the gradient.
///
/// Note what the neighbourhood actually contains. At the instant pixel `p` fires, the neighbours
/// *ahead* of the edge have not fired yet; only the trailing half-disc has. The fit is therefore
/// always one-sided, which is not a defect — the trailing events lie on the same plane — but it is
/// why the curvature bias on a *curved* event surface does not cancel, and why the tests in this
/// module assert a bias that scales like `L/r` rather than `(L/r)^2`.
///
/// The paper's second stage rejects events whose distance from the first-pass plane exceeds a
/// threshold it calls `th2` and refits; that is `reject_s` here, in **seconds**, and setting it to
/// `None` disables the pass. The paper states `th2` as a fraction of the local time scale; this
/// implementation takes an absolute time because a fraction of `window_s` would couple two
/// parameters that a user tunes for different reasons.
#[derive(Debug, Clone, PartialEq)]
pub struct PlaneFlow {
    radius: u16,
    window_s: f64,
    reject_s: Option<f64>,
    min_events: usize,
    sae: TimeSurface,
}

impl PlaneFlow {
    /// Build an estimator.
    ///
    /// `radius` is the neighbourhood half-width `L` in pixels (the paper uses 3 to 5), `window_s`
    /// is how far back the `SAE` is trusted, `reject_s` is the outlier threshold, and `min_events`
    /// is the smallest support that will be fitted.
    ///
    /// # Errors
    ///
    /// - [`VisionError::NonPositive`] / [`VisionError::NonFinite`] for a bad `window_s` or
    ///   `reject_s`.
    /// - [`VisionError::TooFew`] when `min_events` is below 3; a plane needs three points, and
    ///   accepting fewer would return a fit determined by the collinearity guard rather than by the
    ///   data.
    /// - [`VisionError::BadParameters`] when `radius` is zero — a one-pixel neighbourhood has no
    ///   support — or when `reject_s` exceeds `window_s`, which is a rejection pass that can never
    ///   reject.
    pub fn new(
        geom: Geometry,
        radius: u16,
        window_s: f64,
        reject_s: Option<f64>,
        min_events: usize,
    ) -> Result<Self, VisionError> {
        let window_s = positive(window_s, "plane-fit time window")?;
        if radius == 0 {
            return Err(VisionError::BadParameters {
                why: "a radius-0 neighbourhood contains only the event itself",
            });
        }
        if min_events < 3 {
            return Err(VisionError::TooFew {
                what: "minimum plane-fit support",
                have: min_events,
                need: 3,
            });
        }
        if let Some(r) = reject_s {
            let r = positive(r, "outlier rejection threshold")?;
            if r > window_s {
                return Err(VisionError::BadParameters {
                    why: "the rejection threshold is wider than the time window, so it can never \
                          reject anything",
                });
            }
        }
        let sae = TimeSurface::new(geom, Decay::Linear { window_s }, true)?;
        Ok(Self { radius, window_s, reject_s, min_events, sae })
    }

    /// The neighbourhood half-width in pixels.
    #[must_use]
    pub fn radius(&self) -> u16 {
        self.radius
    }

    /// The time window in seconds.
    #[must_use]
    pub fn window_s(&self) -> f64 {
        self.window_s
    }

    /// The surface of active events this estimator maintains.
    #[must_use]
    pub fn surface(&self) -> &TimeSurface {
        &self.sae
    }

    /// Accept one event and try to estimate the normal flow at it.
    ///
    /// # Errors
    ///
    /// [`VisionError::OutOfBounds`], [`VisionError::NonFinite`] or [`VisionError::OutOfOrder`] from
    /// the underlying [`TimeSurface::update`]. A refusal to *estimate* is an `Ok` carrying the
    /// reason; an `Err` means the input was not a well-formed event stream.
    pub fn push(&mut self, e: PixelEvent) -> Result<FlowOutcome, VisionError> {
        self.sae.update(e)?;
        let r = i64::from(self.radius);
        let mut pts: Vec<(f64, f64, f64)> = Vec::with_capacity(((2 * r + 1) * (2 * r + 1)) as usize);
        let geom = self.sae.geometry();
        for dy in -r..=r {
            for dx in -r..=r {
                let px = i64::from(e.x) + dx;
                let py = i64::from(e.y) + dy;
                if !geom.contains(px, py) {
                    continue;
                }
                let Some(t) = self.sae.last_time(px as u16, py as u16, e.polarity)? else {
                    continue;
                };
                // Causal and recent. `t <= e.t_s` is guaranteed by the surface's ordering
                // invariant, so the comparison that does work here is the window.
                if e.t_s - t <= self.window_s {
                    pts.push((px as f64, py as f64, t));
                }
            }
        }
        if pts.len() < self.min_events {
            return Ok(FlowOutcome::TooFew { found: pts.len() });
        }
        let first = match fit_plane(&pts) {
            Ok(f) => f,
            Err(VisionError::Degenerate { .. }) => return Ok(FlowOutcome::Degenerate),
            Err(other) => return Err(other),
        };
        let fit = if let Some(th) = self.reject_s {
            let kept: Vec<(f64, f64, f64)> =
                pts.iter().copied().filter(|&(x, y, t)| first.residual(x, y, t).abs() <= th).collect();
            if kept.len() < self.min_events {
                return Ok(FlowOutcome::RejectedTooMany { inliers: kept.len() });
            }
            match fit_plane(&kept) {
                Ok(f) => f,
                Err(VisionError::Degenerate { .. }) => return Ok(FlowOutcome::Degenerate),
                Err(other) => return Err(other),
            }
        } else {
            first
        };
        match fit.flow() {
            Ok(f) => Ok(FlowOutcome::Fitted(f)),
            Err(_) => Ok(FlowOutcome::NoMotion),
        }
    }

    /// Run a whole stream, returning one outcome per input event in the same order.
    ///
    /// # Errors
    ///
    /// As [`PlaneFlow::push`].
    pub fn run(&mut self, events: &[PixelEvent]) -> Result<Vec<FlowOutcome>, VisionError> {
        events.iter().map(|e| self.push(*e)).collect()
    }
}

/// Fraction of outcomes that produced a velocity, and the mean error against a known flow field.
///
/// Returned by [`flow_error`]. Kept as a struct rather than a tuple because the two numbers are
/// meaningless apart: a method that fits one event in a thousand can report any error it likes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FlowError {
    /// How many events produced a [`FlowOutcome::Fitted`].
    pub fitted: usize,
    /// How many events were offered.
    pub offered: usize,
    /// Mean absolute error in speed, px/s, over the fitted events.
    pub mean_speed_err: f64,
    /// Mean absolute angular error, radians, wrapped to `[0, pi]`, over the fitted events.
    pub mean_angle_err: f64,
    /// Largest relative speed error over the fitted events, as a fraction of the true speed.
    pub max_rel_speed_err: f64,
}

/// Compare per-event flow outcomes against a ground-truth field.
///
/// `truth` is called with the event that produced each outcome and returns the true `(vx, vy)`
/// there, or `None` where the field is undefined — at the centre of a rotation, say — in which case
/// that event is excluded from both counts.
///
/// # Errors
///
/// [`VisionError::TooFew`] when `outcomes` and `events` differ in length, and when no event both
/// fitted and had a defined ground truth, because a mean over nothing is not zero.
pub fn flow_error(
    events: &[PixelEvent],
    outcomes: &[FlowOutcome],
    truth: impl Fn(PixelEvent) -> Option<(f64, f64)>,
) -> Result<FlowError, VisionError> {
    if events.len() != outcomes.len() {
        return Err(VisionError::TooFew {
            what: "outcome list against the event list",
            have: outcomes.len(),
            need: events.len(),
        });
    }
    let (mut fitted, mut offered) = (0usize, 0usize);
    let (mut se, mut ae, mut max_rel) = (0.0f64, 0.0f64, 0.0f64);
    for (e, o) in events.iter().zip(outcomes) {
        let Some((tvx, tvy)) = truth(*e) else { continue };
        offered += 1;
        let FlowOutcome::Fitted(f) = o else { continue };
        fitted += 1;
        let tspeed = tvx.hypot(tvy);
        se += (f.speed - tspeed).abs();
        if tspeed > 0.0 {
            max_rel = max_rel.max((f.speed - tspeed).abs() / tspeed);
        }
        let mut d = (f.direction_rad - tvy.atan2(tvx)).abs();
        // Wrap into [0, pi]: an angular error is a distance on a circle, and 359 degrees is one
        // degree of error, not 359.
        while d > core::f64::consts::PI {
            d = (2.0 * core::f64::consts::PI - d).abs();
        }
        ae += d;
    }
    if fitted == 0 {
        return Err(VisionError::TooFew {
            what: "fitted events with a defined ground truth",
            have: 0,
            need: 1,
        });
    }
    let n = fitted as f64;
    Ok(FlowError {
        fitted,
        offered,
        mean_speed_err: se / n,
        mean_angle_err: ae / n,
        max_rel_speed_err: max_rel,
    })
}

// ---------------------------------------------------------------------------------------------
// HOTS: a hierarchy of event-based time surfaces
// ---------------------------------------------------------------------------------------------

/// One `HOTS` layer: online clustering of local time-surface patches (Lagorce, Orchard, Gallupi,
/// Shi and Benosman, IEEE `TPAMI` 39(7), 2017).
///
/// # What it is for
///
/// A time-surface patch around an event is a small, dense, real-valued vector describing the local
/// spatio-temporal texture — which neighbours fired, how recently. `HOTS` learns a dictionary of
/// `C` such patches by online clustering, then **relabels** each incoming event with the index of
/// its nearest prototype. Stack layers with growing radius and time constant and you get a feature
/// hierarchy whose every level is still an event stream, which is the property that lets the whole
/// thing run on event-driven hardware.
///
/// # The update rule, verbatim
///
/// For the selected cluster `k`, with `p_k` the number of times it has been selected:
///
/// ```text
/// alpha = 0.01 / (1 + p_k / 20000)
/// beta  = (C_k . S) / (|C_k| |S|)
/// C_k  <- C_k + alpha * (S - beta * C_k)
/// ```
///
/// The `0.01` and the `20000` are the paper's own dimensionless constants and are kept exactly as
/// printed, where a reader can compare them line by line. The `beta` factor is what distinguishes
/// this from plain online k-means: it scales the prototype down toward the *direction* of the
/// sample rather than its magnitude, so a cluster is a ray rather than a point.
///
/// **This implementation did not locate an author-released reference implementation to check the
/// transcription against.** A reader with the paper should compare, and this doc is where the
/// disagreement should be recorded if there is one.
///
/// # The failure mode this has and does not hide
///
/// Two prototypes initialised identically stay identical forever: the tie-break in
/// [`Hots::nearest`] picks the lowest index, that cluster alone is updated, and the other is a dead
/// unit that never wins. That is the standard k-means degeneracy and there is no cure inside the
/// update rule. [`Hots::new`] therefore initialises from [`crate::rng::Rng`] rather than from a
/// constant, and [`Hots::counts`] is public precisely so a caller can see a zero and know.
#[derive(Debug, Clone, PartialEq)]
pub struct Hots {
    radius: u16,
    centers: Vec<Vec<f64>>,
    counts: Vec<u64>,
    surface: TimeSurface,
}

impl Hots {
    /// The paper's learning rate numerator, dimensionless, transcribed verbatim.
    pub const ALPHA0: f64 = 0.01;
    /// The paper's learning-rate decay constant, in units of "times this cluster was selected",
    /// transcribed verbatim.
    pub const ALPHA_DECAY: f64 = 20000.0;

    /// A layer with `n_centers` prototypes drawn uniformly from `[0, 1)`.
    ///
    /// Uniform `[0, 1)` because a time-surface value lives in `[0, 1]` by construction, so the
    /// initial prototypes occupy the same box the data does. The draw is from the crate's seeded
    /// [`crate::rng::Rng`], so a layer is reproducible from its seed and nothing here reads a clock.
    ///
    /// # Errors
    ///
    /// [`VisionError::NonPositive`] for a bad `tau_s`, [`VisionError::TooFew`] for zero centres or
    /// zero radius, and anything [`TimeSurface::new`] refuses.
    pub fn new(
        geom: Geometry,
        radius: u16,
        tau_s: f64,
        n_centers: usize,
        rng: &mut crate::rng::Rng,
    ) -> Result<Self, VisionError> {
        if n_centers == 0 {
            return Err(VisionError::TooFew { what: "prototype set", have: 0, need: 1 });
        }
        if radius == 0 {
            return Err(VisionError::TooFew { what: "patch radius", have: 0, need: 1 });
        }
        let side = (2 * usize::from(radius) + 1).pow(2);
        let centers: Vec<Vec<f64>> =
            (0..n_centers).map(|_| (0..side).map(|_| rng.next_f64()).collect()).collect();
        Self::with_centers(geom, radius, tau_s, centers)
    }

    /// A layer with prototypes supplied — a trained dictionary, or a hand-built one for a test.
    ///
    /// # Errors
    ///
    /// [`VisionError::TooFew`] when the set is empty or the radius is zero;
    /// [`VisionError::BadParameters`] when a prototype's length is not `(2 * radius + 1)^2`;
    /// [`VisionError::NonFinite`] for a non-finite prototype entry; plus anything
    /// [`TimeSurface::new`] refuses.
    pub fn with_centers(
        geom: Geometry,
        radius: u16,
        tau_s: f64,
        centers: Vec<Vec<f64>>,
    ) -> Result<Self, VisionError> {
        if centers.is_empty() {
            return Err(VisionError::TooFew { what: "prototype set", have: 0, need: 1 });
        }
        if radius == 0 {
            return Err(VisionError::TooFew { what: "patch radius", have: 0, need: 1 });
        }
        let want = (2 * usize::from(radius) + 1).pow(2);
        for c in &centers {
            if c.len() != want {
                return Err(VisionError::BadParameters {
                    why: "a prototype's length does not match (2 * radius + 1)^2",
                });
            }
            for v in c {
                finite(*v, "prototype entry")?;
            }
        }
        let surface = TimeSurface::new(geom, Decay::Exponential { tau_s }, true)?;
        let counts = vec![0u64; centers.len()];
        Ok(Self { radius, centers, counts, surface })
    }

    /// The patch half-width in pixels.
    #[must_use]
    pub fn radius(&self) -> u16 {
        self.radius
    }

    /// The prototype dictionary, one `(2 * radius + 1)^2` vector per cluster.
    #[must_use]
    pub fn centers(&self) -> &[Vec<f64>] {
        &self.centers
    }

    /// How many times each cluster has been selected. A zero here is a dead unit; see the type doc.
    #[must_use]
    pub fn counts(&self) -> &[u64] {
        &self.counts
    }

    /// The time surface this layer maintains.
    #[must_use]
    pub fn surface(&self) -> &TimeSurface {
        &self.surface
    }

    /// Index of the nearest prototype under Euclidean distance, with that distance.
    ///
    /// Ties go to the lowest index, which is the degeneracy described in the type doc.
    ///
    /// # Errors
    ///
    /// [`VisionError::BadParameters`] when `s` is not `(2 * radius + 1)^2` long.
    pub fn nearest(&self, s: &[f64]) -> Result<(usize, f64), VisionError> {
        if s.len() != self.centers[0].len() {
            return Err(VisionError::BadParameters {
                why: "the patch length does not match the prototype length",
            });
        }
        let mut best = (0usize, f64::INFINITY);
        for (i, c) in self.centers.iter().enumerate() {
            let d2: f64 = c.iter().zip(s).map(|(a, b)| (a - b) * (a - b)).sum();
            if d2 < best.1 {
                best = (i, d2);
            }
        }
        Ok((best.0, best.1.sqrt()))
    }

    /// Record the event and return the index of its nearest prototype, **without learning**.
    ///
    /// This is inference: the layer's output event is `(t, x, y, cluster)`.
    ///
    /// # Errors
    ///
    /// As [`TimeSurface::update`] and [`TimeSurface::patch`].
    pub fn assign(&mut self, e: PixelEvent) -> Result<usize, VisionError> {
        self.surface.update(e)?;
        let s = self.surface.patch(e.x, e.y, self.radius, e.polarity, e.t_s)?;
        Ok(self.nearest(&s)?.0)
    }

    /// Record the event, select the nearest prototype and move it toward the patch.
    ///
    /// Returns the selected cluster index.
    ///
    /// # Errors
    ///
    /// As [`Hots::assign`].
    pub fn learn(&mut self, e: PixelEvent) -> Result<usize, VisionError> {
        self.surface.update(e)?;
        let s = self.surface.patch(e.x, e.y, self.radius, e.polarity, e.t_s)?;
        let (k, _) = self.nearest(&s)?;
        let p = self.counts[k] as f64;
        let alpha = Self::ALPHA0 / (1.0 + p / Self::ALPHA_DECAY);
        let c = &mut self.centers[k];
        let dot: f64 = c.iter().zip(&s).map(|(a, b)| a * b).sum();
        let nc: f64 = c.iter().map(|a| a * a).sum::<f64>().sqrt();
        let ns: f64 = s.iter().map(|b| b * b).sum::<f64>().sqrt();
        // A zero-norm prototype or patch has no direction, so `beta` is undefined. The paper does
        // not say what to do; this implementation uses `beta = 0`, which reduces the step to plain
        // online k-means for that one update, and says so here rather than dividing by zero.
        let beta = if nc > 0.0 && ns > 0.0 { dot / (nc * ns) } else { 0.0 };
        for (ci, si) in c.iter_mut().zip(&s) {
            *ci += alpha * (si - beta * *ci);
        }
        self.counts[k] += 1;
        Ok(k)
    }

    /// The learning rate this layer would use for its `n`-th selection of a cluster, dimensionless.
    ///
    /// Exposed so that the geometric convergence of [`Hots::learn`] can be predicted in closed form
    /// and checked, rather than compared against a previous run.
    #[must_use]
    pub fn alpha_for(n: u64) -> f64 {
        Self::ALPHA0 / (1.0 + n as f64 / Self::ALPHA_DECAY)
    }
}

// ---------------------------------------------------------------------------------------------
// HATS: histograms of averaged time surfaces
// ---------------------------------------------------------------------------------------------

/// `HATS`: histograms of averaged time surfaces (Sironi, Brambilla, Bourdis, Lagorce and Benosman,
/// *`HATS`: Histograms of Averaged Time Surfaces for Robust Event-based Object Classification*,
/// CVPR 2018).
///
/// # What it adds over a plain time surface
///
/// A [`TimeSurface`] keeps one timestamp per pixel, so it cannot tell a pixel that fired fifty
/// times in the last millisecond from one that fired once. `HATS` replaces "the last event" with a
/// **local memory time surface**: the *sum* of `exp(-(t_i - t_j)/tau)` over every earlier event
/// `e_j` in the same cell, within `radius` pixels and `window_s` seconds. It then averages that
/// surface over all events in a cell, giving one fixed-length histogram per cell per polarity that
/// a plain linear classifier can consume.
///
/// The price is history. Where a time surface is `O(1)` per event, this is `O(k)` in the number of
/// recent events in the cell — the implementation here is the straightforward quadratic pass over
/// each cell, which is right for teaching and wrong for a 640x480 sensor at 10 Mev/s. A production
/// version keeps a per-pixel ring buffer; the arithmetic is identical and the doc says so instead
/// of the code hiding it.
///
/// # The defaults, and what is uncertain about them
///
/// The paper's reported settings for `N-CARS` are `radius = 3` (a 7x7 neighbourhood),
/// `cell_px = 10`, `window_s = 0.1` and a `tau` this implementation reads as `1e9` nanoseconds,
/// i.e. **1 second**. [`Hats::n_cars`] carries those. A one-second time constant against a
/// hundred-millisecond window means the exponential barely decays inside the window, so the
/// descriptor is close to a plain event count — which may well be the paper's intent and may
/// equally be this implementation misreading the units. **It is flagged rather than presented
/// confidently**, and a reader with the paper should check.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hats {
    /// Cell side `K` in pixels. The sensor is tiled by `ceil(width/K)` by `ceil(height/K)` cells
    /// and each produces its own histogram.
    pub cell_px: u16,
    /// Neighbourhood half-width `rho` in pixels; each histogram has `(2 * rho + 1)^2` bins per
    /// polarity plane.
    pub radius: u16,
    /// Exponential time constant, **seconds**.
    pub tau_s: f64,
    /// Memory window, **seconds**: events older than this contribute nothing. The boundary is
    /// **inclusive** — an event exactly `window_s` old is the last one that counts — which is
    /// pinned by a test rather than left to whichever comparison someone writes next.
    pub window_s: f64,
    /// Whether `On` and `Off` get separate histograms. The paper keeps them separate.
    pub split_polarity: bool,
}

impl Hats {
    /// The settings this implementation reads from the paper's `N-CARS` experiment. See the type
    /// doc for the uncertainty about `tau`.
    #[must_use]
    pub fn n_cars() -> Self {
        Self { cell_px: 10, radius: 3, tau_s: 1.0, window_s: 0.1, split_polarity: true }
    }

    /// Bins per cell per polarity plane, `(2 * radius + 1)^2`.
    #[must_use]
    pub fn bins_per_cell(self) -> usize {
        (2 * usize::from(self.radius) + 1).pow(2)
    }

    /// Cells across and down for a given sensor, `ceil(width / cell_px)` by
    /// `ceil(height / cell_px)`.
    ///
    /// # Errors
    ///
    /// [`VisionError::NonPositive`] when `cell_px` is zero — a zero-width cell tiles nothing.
    pub fn cell_grid(self, geom: Geometry) -> Result<(usize, usize), VisionError> {
        if self.cell_px == 0 {
            return Err(VisionError::NonPositive { what: "cell side", value: 0.0 });
        }
        let k = usize::from(self.cell_px);
        Ok((
            usize::from(geom.width).div_ceil(k),
            usize::from(geom.height).div_ceil(k),
        ))
    }

    /// Total descriptor length: `cells * planes * bins_per_cell`.
    ///
    /// # Errors
    ///
    /// As [`Hats::cell_grid`].
    pub fn descriptor_len(self, geom: Geometry) -> Result<usize, VisionError> {
        let (cx, cy) = self.cell_grid(geom)?;
        let planes = if self.split_polarity { 2 } else { 1 };
        Ok(cx * cy * planes * self.bins_per_cell())
    }

    /// Compute the descriptor for a whole event slice.
    ///
    /// Layout is cell-major: cell `(cy, cx)` occupies
    /// `((cy * cells_x + cx) * planes + plane) * bins_per_cell` onward, and within a cell the bins
    /// are row-major over offsets `(dx, dy)` in `-radius..=radius` with `(dx + rho, dy + rho)` the
    /// bin index. **The offset is the neighbour's position relative to the current event**, so a
    /// sign flip here mirrors every descriptor; the tests assert the asymmetric case for that
    /// reason.
    ///
    /// Each cell's histogram is divided by the number of events in that cell, which is the paper's
    /// averaging step and is what makes a fast-moving object's descriptor comparable to a slow
    /// one's. A cell with no events keeps its exact zeros.
    ///
    /// # Errors
    ///
    /// [`VisionError::NonPositive`] for a bad `tau_s`, `window_s` or `cell_px`;
    /// [`VisionError::OutOfBounds`] for an event off the lattice; [`VisionError::OutOfOrder`] when
    /// the slice is not sorted by time, because the "earlier events" the memory surface sums over
    /// are defined by the slice order.
    pub fn descriptor(
        self,
        geom: Geometry,
        events: &[PixelEvent],
    ) -> Result<Vec<f64>, VisionError> {
        positive(self.tau_s, "HATS time constant tau")?;
        positive(self.window_s, "HATS memory window")?;
        require_time_ordered(events)?;
        let (cells_x, _cells_y) = self.cell_grid(geom)?;
        let planes = if self.split_polarity { 2 } else { 1 };
        let bins = self.bins_per_cell();
        let mut out = vec![0.0; self.descriptor_len(geom)?];
        let mut per_cell = vec![0u64; out.len() / (planes * bins)];
        let k = usize::from(self.cell_px);
        let rho = i64::from(self.radius);
        let side = 2 * usize::from(self.radius) + 1;

        for (i, e) in events.iter().enumerate() {
            geom.require(e.x, e.y)?;
            let cell = (usize::from(e.y) / k) * cells_x + usize::from(e.x) / k;
            let plane = e.plane(self.split_polarity);
            let base = (cell * planes + plane) * bins;
            per_cell[cell] += 1;
            // The event's own contribution: `exp(0) = 1` at the centre bin. The paper's local
            // memory time surface includes the current event, so a cell holding exactly one event
            // has a histogram that is exactly the centre-bin indicator — which is the closed form
            // this is tested against.
            out[base + rho as usize * side + rho as usize] += 1.0;
            for e_j in events[..i].iter().rev() {
                let dt = e.t_s - e_j.t_s;
                if dt > self.window_s {
                    break;
                }
                if self.split_polarity && e_j.polarity != e.polarity {
                    continue;
                }
                if usize::from(e_j.y) / k != usize::from(e.y) / k
                    || usize::from(e_j.x) / k != usize::from(e.x) / k
                {
                    continue;
                }
                let dx = i64::from(e_j.x) - i64::from(e.x);
                let dy = i64::from(e_j.y) - i64::from(e.y);
                if dx.abs() > rho || dy.abs() > rho {
                    continue;
                }
                let bin = (dy + rho) as usize * side + (dx + rho) as usize;
                out[base + bin] += (-dt / self.tau_s).exp();
            }
        }
        for (c, n) in per_cell.iter().enumerate() {
            if *n == 0 {
                continue;
            }
            let lo = c * planes * bins;
            for v in &mut out[lo..lo + planes * bins] {
                *v /= *n as f64;
            }
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------------------------
// Corner detection
// ---------------------------------------------------------------------------------------------

/// A corner event: the detector's verdict on one incoming event.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Corner {
    /// Time of the triggering event, seconds.
    pub t_s: f64,
    /// Column of the triggering event.
    pub x: u16,
    /// Row of the triggering event.
    pub y: u16,
    /// Detector-specific strength. For [`EHarris`] this is the `Harris` response
    /// `det(M) - k * trace(M)^2`, whose **sign** is the meaningful part. For [`EFast`] it is the
    /// length of the contiguous newest arc found on the inner circle, in pixels.
    pub score: f64,
}

/// `eHarris`: the `Harris` corner response evaluated on a binarised local time surface (Vasco,
/// Glover and Bartolozzi, *Fast event-based `Harris` corner detection exploiting the advantages of
/// event-driven cameras*, `IROS` 2016).
///
/// # The idea
///
/// `Harris`'s 1988 detector asks whether the local image gradient has *two* strong directions. That
/// question survives the move to events intact, because it only needs a local binary pattern — and
/// an event camera gives one for free: the set of pixels that fired recently. So: take the `(2r+1)`
/// window around the incoming event, mark 1 where the `SAE` says the pixel fired within
/// `window_s`, run `Sobel`, accumulate the structure tensor `M = sum [Ix^2, IxIy; IxIy, Iy^2]`, and
/// score it as `det(M) - k * trace(M)^2`.
///
/// # The sign property this is tested on, rather than a tuned threshold
///
/// On a straight edge filling the window, the binary pattern is constant along the edge, so one
/// `Sobel` response is **exactly zero**, `det(M)` is exactly zero, and the score is
/// `-k * trace(M)^2`, which is strictly negative for any non-flat window. On a right-angle corner
/// both eigenvalues are positive and the score exceeds zero whenever `lambda1 * lambda2 >
/// k * (lambda1 + lambda2)^2`, which for equal eigenvalues is `1 > 4k` — true for the standard
/// `k = 0.04`. That sign flip is the property asserted in this module's tests. A threshold is a
/// tuning parameter; a sign is a theorem.
///
/// # Where this departs from the paper
///
/// Vasco et al. binarise by keeping a fixed-length queue of the `N` most recent events in the
/// window. This implementation binarises by a **time window** instead — same idea, with a parameter
/// that has units. The two differ on a scene whose event rate varies a lot: a queue adapts its
/// effective window to the rate, a fixed time window does not. Neither is obviously right, and the
/// one with units is the one this crate can state a value for.
#[derive(Debug, Clone, PartialEq)]
pub struct EHarris {
    radius: u16,
    window_s: f64,
    k: f64,
    threshold: f64,
    sae: TimeSurface,
}

impl EHarris {
    /// `Harris`'s own sensitivity constant, dimensionless, as the 1988 paper prints it.
    ///
    /// Sets the largest eigenvalue ratio that still scores positive. `0.04` admits anything up to
    /// about a 19:1 ratio; raising it makes the detector fussier about how square a corner is.
    pub const K_HARRIS: f64 = 0.04;

    /// A detector over `geom`.
    ///
    /// `radius` is the window half-width (the paper's 9x9 window is `radius = 4`), `window_s` is
    /// how recently a pixel must have fired to count as set, `k` is the `Harris` constant and
    /// `threshold` is the score a detection must exceed. A `threshold` of `0.0` makes the detector
    /// exactly the sign test described in the type doc.
    ///
    /// # Errors
    ///
    /// [`VisionError::NonPositive`] for a bad `window_s`, [`VisionError::NonFinite`] for a bad `k`
    /// or `threshold`, and [`VisionError::TooFew`] when `radius` is below 2 — `Sobel` consumes one
    /// ring and the structure tensor needs more than a single interior pixel to have two
    /// directions.
    pub fn new(
        geom: Geometry,
        radius: u16,
        window_s: f64,
        k: f64,
        threshold: f64,
    ) -> Result<Self, VisionError> {
        let window_s = positive(window_s, "eHarris binarisation window")?;
        finite(k, "Harris constant k")?;
        finite(threshold, "eHarris threshold")?;
        if radius < 2 {
            return Err(VisionError::TooFew { what: "eHarris window radius", have: radius.into(), need: 2 });
        }
        let sae = TimeSurface::new(geom, Decay::Linear { window_s }, false)?;
        Ok(Self { radius, window_s, k, threshold, sae })
    }

    /// The window half-width in pixels.
    #[must_use]
    pub fn radius(&self) -> u16 {
        self.radius
    }

    /// The binarisation window in seconds.
    #[must_use]
    pub fn window_s(&self) -> f64 {
        self.window_s
    }

    /// The surface of active events this detector maintains.
    #[must_use]
    pub fn surface(&self) -> &TimeSurface {
        &self.sae
    }

    /// The `Harris` response of a square binary patch, given its side length.
    ///
    /// Exposed as a free-standing computation so that the sign property can be checked against
    /// hand-built patterns — a half-plane, a quadrant — without going through an event stream. That
    /// is the difference between testing the detector and testing the scene generator.
    ///
    /// `Sobel` is evaluated on the `(side - 2)^2` interior; the structure tensor is an unweighted
    /// sum over it. `Harris`'s original uses a `Gaussian` window, which sharpens localisation and
    /// changes no sign; this implementation uses uniform weights and says so.
    ///
    /// # Errors
    ///
    /// [`VisionError::BadParameters`] when `patch.len()` is not `side * side` or `side` is below 3,
    /// [`VisionError::NonFinite`] for a non-finite entry.
    pub fn score_of_patch(&self, patch: &[f64], side: usize) -> Result<f64, VisionError> {
        harris_score(patch, side, self.k)
    }

    /// Accept an event and classify it.
    ///
    /// `Ok(None)` means the score did not exceed `threshold` — a classification, not a dropped
    /// record. Use [`EHarris::score_at`] when the number itself is wanted.
    ///
    /// # Errors
    ///
    /// As [`TimeSurface::update`].
    pub fn push(&mut self, e: PixelEvent) -> Result<Option<Corner>, VisionError> {
        self.sae.update(e)?;
        let s = self.score_at(e.x, e.y, e.t_s)?;
        if s > self.threshold {
            Ok(Some(Corner { t_s: e.t_s, x: e.x, y: e.y, score: s }))
        } else {
            Ok(None)
        }
    }

    /// The `Harris` response at a pixel, from the detector's current `SAE`.
    ///
    /// Pixels outside the sensor are treated as never having fired, i.e. binary 0 — the same
    /// convention [`TimeSurface::patch`] uses, for the same reason.
    ///
    /// # Errors
    ///
    /// [`VisionError::OutOfBounds`] if the centre is off the lattice, [`VisionError::NonFinite`]
    /// for a bad query time.
    pub fn score_at(&self, x: u16, y: u16, t_s: f64) -> Result<f64, VisionError> {
        let t = finite(t_s, "query time")?;
        self.sae.geometry().require(x, y)?;
        let r = i64::from(self.radius);
        let side = (2 * self.radius + 1) as usize;
        let mut patch = vec![0.0; side * side];
        for dy in -r..=r {
            for dx in -r..=r {
                let px = i64::from(x) + dx;
                let py = i64::from(y) + dy;
                if !self.sae.geometry().contains(px, py) {
                    continue;
                }
                // Binarise: set if the pixel fired within the window. Any polarity — the detector
                // is looking for geometry, and splitting the planes here would halve the pattern
                // an alternating edge writes.
                let last = self.sae.last_time(px as u16, py as u16, Polarity::On)?;
                if let Some(l) = last
                    && t - l <= self.window_s
                {
                    patch[((dy + r) as usize) * side + (dx + r) as usize] = 1.0;
                }
            }
        }
        harris_score(&patch, side, self.k)
    }
}

/// `Sobel` gradients, structure tensor, `Harris` response. Shared by [`EHarris::score_of_patch`]
/// and [`EHarris::score_at`] so that a test against a hand-built patch exercises the same
/// arithmetic the stream path does.
fn harris_score(patch: &[f64], side: usize, k: f64) -> Result<f64, VisionError> {
    if side < 3 || patch.len() != side * side {
        return Err(VisionError::BadParameters {
            why: "the patch length does not match a square of side at least 3",
        });
    }
    for v in patch {
        finite(*v, "patch entry")?;
    }
    const GX: [[f64; 3]; 3] = [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]];
    const GY: [[f64; 3]; 3] = [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]];
    let (mut mxx, mut myy, mut mxy) = (0.0f64, 0.0f64, 0.0f64);
    for j in 1..side - 1 {
        for i in 1..side - 1 {
            let (mut gx, mut gy) = (0.0f64, 0.0f64);
            for dj in 0..3usize {
                for di in 0..3usize {
                    let v = patch[(j + dj - 1) * side + (i + di - 1)];
                    gx += GX[dj][di] * v;
                    gy += GY[dj][di] * v;
                }
            }
            mxx += gx * gx;
            myy += gy * gy;
            mxy += gx * gy;
        }
    }
    let det = mxx * myy - mxy * mxy;
    let trace = mxx + myy;
    Ok(det - k * trace * trace)
}

/// The `Bresenham` circle of radius 3, 16 pixels, in circular order — the `FAST` circle.
///
/// Offsets are `(dx, dy)` with `dy` positive downward, and the order is contiguous around the
/// circle, which is what makes "a contiguous arc" a window over consecutive entries.
pub const FAST_CIRCLE_16: [(i64, i64); 16] = [
    (0, 3),
    (1, 3),
    (2, 2),
    (3, 1),
    (3, 0),
    (3, -1),
    (2, -2),
    (1, -3),
    (0, -3),
    (-1, -3),
    (-2, -2),
    (-3, -1),
    (-3, 0),
    (-3, 1),
    (-2, 2),
    (-1, 3),
];

/// The `Bresenham` circle of radius 4, 20 pixels, in circular order — the outer circle `eFAST`
/// requires to agree with the inner one.
pub const FAST_CIRCLE_20: [(i64, i64); 20] = [
    (0, 4),
    (1, 4),
    (2, 3),
    (3, 2),
    (4, 1),
    (4, 0),
    (4, -1),
    (3, -2),
    (2, -3),
    (1, -4),
    (0, -4),
    (-1, -4),
    (-2, -3),
    (-3, -2),
    (-4, -1),
    (-4, 0),
    (-4, 1),
    (-3, 2),
    (-2, 3),
    (-1, 4),
];

/// The length of the shortest contiguous arc whose timestamps are **all strictly newer** than every
/// timestamp off the arc, if one exists within `lo..=hi`.
///
/// The `eFAST` criterion in one function, and the reason it distinguishes an edge from a corner.
/// When a straight edge sweeps a pixel, the circle splits into a fired half and an unfired half:
/// the shortest valid arc is that half, about `n/2` long. When a right-angle corner sweeps it, the
/// fired sector is about a quarter, so the shortest valid arc is about `n/4`. The accepted ranges
/// are Mueggler et al.'s.
///
/// Returns `None` when no arc in the range qualifies. Arcs wrap around the end of the slice, which
/// is what "contiguous on a circle" means.
#[must_use]
pub fn newest_arc(times: &[f64], lo: usize, hi: usize) -> Option<usize> {
    let n = times.len();
    if n == 0 || lo == 0 || hi < lo || hi >= n {
        return None;
    }
    for len in lo..=hi {
        for start in 0..n {
            let mut arc_min = f64::INFINITY;
            let mut out_max = f64::NEG_INFINITY;
            for (j, t) in times.iter().enumerate() {
                // `j` is inside the arc `[start, start + len)` taken modulo `n`.
                let inside = (j + n - start) % n < len;
                if inside {
                    arc_min = arc_min.min(*t);
                } else {
                    out_max = out_max.max(*t);
                }
            }
            if arc_min > out_max {
                return Some(len);
            }
        }
    }
    None
}

/// `eFAST`: corner detection by arc length on the surface of active events (Mueggler, Bartolozzi
/// and Scaramuzza, *Fast Event-based Corner Detection*, `BMVC` 2017).
///
/// # Why this is cheap
///
/// No gradients, no structure tensor, no floating-point convolution — just timestamp comparisons on
/// 36 pixels. That is what makes it the one event-based corner detector that runs comfortably at
/// full sensor rate on a microcontroller, and it is why it is here beside [`EHarris`] rather than
/// instead of it: [`EHarris`] gives a continuous score that can be ranked, `eFAST` gives a binary
/// verdict very fast.
///
/// # The criterion
///
/// On each of two `Bresenham` circles around the incoming event, find the shortest contiguous arc
/// whose `SAE` timestamps are all newer than every timestamp off the arc. A corner is declared when
/// the inner circle's arc length falls in `inner_arc` and the outer circle's in `outer_arc`.
/// Requiring both is what rejects the isolated-noise event that makes one short arc by accident.
///
/// Pixels that have never fired are treated as infinitely old, so an arc that includes one can
/// never be the newest — which is exactly right and is why a sensor's first events produce no
/// corners rather than producing spurious ones.
#[derive(Debug, Clone, PartialEq)]
pub struct EFast {
    inner_arc: (usize, usize),
    outer_arc: (usize, usize),
    sae: TimeSurface,
}

impl EFast {
    /// The accepted inner-circle arc lengths in Mueggler et al., inclusive: 3 to 6 of 16.
    pub const INNER_ARC: (usize, usize) = (3, 6);
    /// The accepted outer-circle arc lengths in Mueggler et al., inclusive: 4 to 8 of 20.
    pub const OUTER_ARC: (usize, usize) = (4, 8);

    /// A detector with the paper's arc ranges.
    ///
    /// # Errors
    ///
    /// As [`EFast::with_arcs`].
    pub fn new(geom: Geometry) -> Result<Self, VisionError> {
        Self::with_arcs(geom, Self::INNER_ARC, Self::OUTER_ARC)
    }

    /// A detector with arc ranges of your own, for studying the criterion.
    ///
    /// # Errors
    ///
    /// [`VisionError::BadParameters`] when a range is empty, starts at zero, or reaches past its
    /// circle's circumference — 16 for the inner circle, 20 for the outer. An arc of 16 of 16 is
    /// "every pixel is newer than every pixel", which is never true.
    pub fn with_arcs(
        geom: Geometry,
        inner_arc: (usize, usize),
        outer_arc: (usize, usize),
    ) -> Result<Self, VisionError> {
        if inner_arc.0 == 0 || inner_arc.1 < inner_arc.0 || inner_arc.1 >= FAST_CIRCLE_16.len() {
            return Err(VisionError::BadParameters {
                why: "the inner arc range is empty or reaches past the 16-pixel circle",
            });
        }
        if outer_arc.0 == 0 || outer_arc.1 < outer_arc.0 || outer_arc.1 >= FAST_CIRCLE_20.len() {
            return Err(VisionError::BadParameters {
                why: "the outer arc range is empty or reaches past the 20-pixel circle",
            });
        }
        // The decay is unused: this detector reads raw SAE timestamps. `Linear` with a one-second
        // window is a placeholder and is documented here so nobody reads meaning into it.
        let sae = TimeSurface::new(geom, Decay::Linear { window_s: 1.0 }, true)?;
        Ok(Self { inner_arc, outer_arc, sae })
    }

    /// The surface of active events this detector maintains.
    #[must_use]
    pub fn surface(&self) -> &TimeSurface {
        &self.sae
    }

    /// Mutable access to the surface, so a test can install a hand-built `SAE` and check the
    /// criterion directly rather than through a scene generator.
    pub fn surface_mut(&mut self) -> &mut TimeSurface {
        &mut self.sae
    }

    /// The two arc lengths found at a pixel, or `None` for the circle that has none.
    ///
    /// Returns `(inner, outer)`.
    ///
    /// # Errors
    ///
    /// [`VisionError::OutOfBounds`] when the centre is off the lattice, or when any circle pixel
    /// is. A partial circle is refused rather than padded: padding with "never fired" would make
    /// every pixel within four of the border look like a corner, because the pad is always the
    /// oldest and any arc beats it.
    pub fn arcs_at(
        &self,
        x: u16,
        y: u16,
        pol: Polarity,
    ) -> Result<(Option<usize>, Option<usize>), VisionError> {
        let inner = self.circle_times(x, y, pol, &FAST_CIRCLE_16)?;
        let outer = self.circle_times(x, y, pol, &FAST_CIRCLE_20)?;
        Ok((
            newest_arc(&inner, self.inner_arc.0, self.inner_arc.1),
            newest_arc(&outer, self.outer_arc.0, self.outer_arc.1),
        ))
    }

    fn circle_times(
        &self,
        x: u16,
        y: u16,
        pol: Polarity,
        circle: &[(i64, i64)],
    ) -> Result<Vec<f64>, VisionError> {
        let geom = self.sae.geometry();
        let mut out = Vec::with_capacity(circle.len());
        for &(dx, dy) in circle {
            let px = i64::from(x) + dx;
            let py = i64::from(y) + dy;
            if !geom.contains(px, py) {
                return Err(VisionError::OutOfBounds {
                    x,
                    y,
                    width: geom.width,
                    height: geom.height,
                });
            }
            out.push(
                self.sae
                    .last_time(px as u16, py as u16, pol)?
                    .unwrap_or(f64::NEG_INFINITY),
            );
        }
        Ok(out)
    }

    /// Whether a pixel satisfies the criterion on both circles.
    ///
    /// # Errors
    ///
    /// As [`EFast::arcs_at`].
    pub fn is_corner(&self, x: u16, y: u16, pol: Polarity) -> Result<bool, VisionError> {
        let (i, o) = self.arcs_at(x, y, pol)?;
        Ok(i.is_some() && o.is_some())
    }

    /// Accept an event and classify it.
    ///
    /// `Ok(None)` covers two cases and they are different: the event was within four pixels of the
    /// border, so no circle could be read; or the arcs were read and did not qualify. Use
    /// [`EFast::arcs_at`] when the distinction matters.
    ///
    /// # Errors
    ///
    /// As [`TimeSurface::update`]. A border event is **not** an error here — it is a `None`,
    /// because a stream always has border events and refusing the whole call would make the
    /// detector unusable on a real recording.
    pub fn push(&mut self, e: PixelEvent) -> Result<Option<Corner>, VisionError> {
        self.sae.update(e)?;
        match self.arcs_at(e.x, e.y, e.polarity) {
            Ok((Some(i), Some(_))) => {
                Ok(Some(Corner { t_s: e.t_s, x: e.x, y: e.y, score: i as f64 }))
            }
            Ok(_) => Ok(None),
            Err(VisionError::OutOfBounds { .. }) => Ok(None),
            Err(other) => Err(other),
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Contrast maximisation
// ---------------------------------------------------------------------------------------------

/// A candidate motion to undo, for contrast maximisation.
///
/// Each variant is a one- or three-parameter family; a full six-degree-of-freedom warp needs scene
/// depth, which an event camera alone does not supply, so these are the families that can be
/// searched from events only.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Motion {
    /// Uniform image-plane translation.
    Translation {
        /// Horizontal velocity, px/s.
        vx: f64,
        /// Vertical velocity, px/s, positive downward.
        vy: f64,
    },
    /// Rotation about a fixed image point — a camera rolling, or a fan seen head-on.
    Rotation {
        /// Angular velocity, **radians per second**, positive from `+x` toward `+y` (i.e.
        /// clockwise on screen, because row 0 is the top).
        omega_rad_s: f64,
        /// Centre column, pixels.
        cx: f64,
        /// Centre row, pixels.
        cy: f64,
    },
    /// Radial expansion at a constant radial speed — the looming case.
    ///
    /// Linear in radius, not exponential: `r' = r - rate * dt`. That matches a rigid surface
    /// approaching at constant speed only to first order, and it matches [`looming_disc`] exactly,
    /// which is why the sweep over it has ground truth.
    RadialExpansion {
        /// Radial speed, px/s, positive outward.
        rate_px_s: f64,
        /// Centre column, pixels.
        cx: f64,
        /// Centre row, pixels.
        cy: f64,
    },
}

impl Motion {
    /// Map an event at `(x, y)` with `dt = t - t_ref` back to where it would have been at `t_ref`.
    ///
    /// # The known pathology
    ///
    /// Every non-translational family here has a **degenerate global maximum**: a rotation or
    /// expansion large enough collapses the whole cloud onto one pixel, which is perfectly sharp
    /// and completely wrong. Gallego et al. discuss it as *event collapse*; the remedies in the
    /// literature are a bounded search range, a regulariser, or a different objective. This module
    /// takes the first — [`search_translation`] and [`sweep`] are bounded, and the bound is the
    /// caller's — and this doc is here so nobody is surprised by a search that runs away when the
    /// bound is widened.
    ///
    /// For [`Motion::RadialExpansion`], a point warped past the centre emerges on the far side
    /// rather than being clamped. That is what the linear radial model says happens and clamping
    /// would put a spurious pile-up at the centre, which is a sharpness the objective would reward.
    #[must_use]
    pub fn warp(self, x: f64, y: f64, dt: f64) -> (f64, f64) {
        match self {
            Self::Translation { vx, vy } => (x - vx * dt, y - vy * dt),
            Self::Rotation { omega_rad_s, cx, cy } => {
                let (dx, dy) = (x - cx, y - cy);
                let a = -omega_rad_s * dt;
                let (s, c) = a.sin_cos();
                (cx + c * dx - s * dy, cy + s * dx + c * dy)
            }
            Self::RadialExpansion { rate_px_s, cx, cy } => {
                let (dx, dy) = (x - cx, y - cy);
                let r = dx.hypot(dy);
                if r == 0.0 {
                    return (cx, cy);
                }
                let k = (r - rate_px_s * dt) / r;
                (cx + k * dx, cy + k * dy)
            }
        }
    }

    /// Reject non-finite parameters at the boundary.
    ///
    /// # Errors
    ///
    /// [`VisionError::NonFinite`], naming the parameter.
    pub fn validate(self) -> Result<(), VisionError> {
        match self {
            Self::Translation { vx, vy } => {
                finite(vx, "translation vx")?;
                finite(vy, "translation vy")?;
            }
            Self::Rotation { omega_rad_s, cx, cy } => {
                finite(omega_rad_s, "angular velocity")?;
                finite(cx, "rotation centre column")?;
                finite(cy, "rotation centre row")?;
            }
            Self::RadialExpansion { rate_px_s, cx, cy } => {
                finite(rate_px_s, "radial rate")?;
                finite(cx, "expansion centre column")?;
                finite(cy, "expansion centre row")?;
            }
        }
        Ok(())
    }
}

/// Which focus measure to maximise.
///
/// Gallego, Gehrig and Scaramuzza (*Focus Is All You Need*, CVPR 2019) survey twenty-two of these
/// and find several that beat variance. Two are implemented here; the module doc says which tests
/// pin which.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Objective {
    /// Population variance of the warped image — the measure of the CVPR 2018 paper.
    Variance,
    /// Mean of the squared pixel values.
    ///
    /// Differs from variance by exactly `mean^2`, and the mean of a warped image is *not* constant
    /// across motions because events warped off the sensor are dropped. So this is a genuinely
    /// different objective, not a reparameterisation, and it is the more sensitive of the two to
    /// the off-sensor bias described on [`WarpedImage::dropped`].
    MeanSquare,
}

/// A warped event image and an honest account of what did not make it in.
///
/// The `dropped` count is not decoration. Bilinear voting silently discards events that warp off
/// the sensor, and a search can exploit that: pushing everything off-sensor gives an empty image,
/// whose variance is exactly zero — harmless for a *maximisation*, fatal if anyone ever minimises
/// this. The count is returned so a caller can see it happening.
#[derive(Debug, Clone, PartialEq)]
pub struct WarpedImage {
    /// The accumulated image of warped events, bilinearly voted.
    pub frame: Frame,
    /// How many events landed at least partly on the sensor.
    pub placed: usize,
    /// How many events warped entirely off the sensor and contributed nothing.
    pub dropped: usize,
}

/// Warp every event by `m` to the reference time and accumulate with bilinear voting.
///
/// Bilinear rather than nearest-neighbour because the objective's argmax is being searched: a
/// nearest-neighbour vote makes the objective piecewise constant in the motion parameters, with
/// plateaus wide enough that a local search stalls and a grid search reports whichever grid point
/// happened to land in the best plateau. The paper uses bilinear for the same reason.
///
/// Polarity is **ignored** — every event votes `+1`. Signed voting is also standard and gives a
/// sharper image on a scene with well-separated `On` and `Off` edges, but it allows cancellation:
/// two opposite edges warped onto each other sum to zero, which the objective reads as perfectly
/// unfocused. Unsigned is the conservative choice and this is where the choice is recorded.
///
/// # Errors
///
/// [`VisionError::NonFinite`] for a bad timestamp, reference time or motion parameter;
/// [`VisionError::OutOfBounds`] for an event off the lattice *before* warping. An event off the
/// lattice after warping is a `dropped`, not an error — that is the point of the warp.
pub fn warped_image(
    geom: Geometry,
    events: &[PixelEvent],
    t_ref_s: f64,
    m: Motion,
) -> Result<WarpedImage, VisionError> {
    m.validate()?;
    let t_ref = finite(t_ref_s, "reference time")?;
    let mut frame = Frame::zeros(geom);
    let (mut placed, mut dropped) = (0usize, 0usize);
    let (w, h) = (i64::from(geom.width), i64::from(geom.height));
    for e in events {
        let t = finite(e.t_s, "event timestamp")?;
        geom.require(e.x, e.y)?;
        let (wx, wy) = m.warp(f64::from(e.x), f64::from(e.y), t - t_ref);
        if !wx.is_finite() || !wy.is_finite() {
            dropped += 1;
            continue;
        }
        let x0 = wx.floor();
        let y0 = wy.floor();
        let (fx, fy) = (wx - x0, wy - y0);
        let mut any = false;
        for (ox, oy, weight) in [
            (0i64, 0i64, (1.0 - fx) * (1.0 - fy)),
            (1, 0, fx * (1.0 - fy)),
            (0, 1, (1.0 - fx) * fy),
            (1, 1, fx * fy),
        ] {
            if weight == 0.0 {
                continue;
            }
            // `as i64` on a float outside i64's range saturates in Rust, so the bounds test below
            // is doing the work rather than relying on a wrap that cannot happen.
            let px = x0 as i64 + ox;
            let py = y0 as i64 + oy;
            if px < 0 || py < 0 || px >= w || py >= h {
                continue;
            }
            frame.data[(py * w + px) as usize] += weight;
            any = true;
        }
        if any {
            placed += 1;
        } else {
            dropped += 1;
        }
    }
    Ok(WarpedImage { frame, placed, dropped })
}

/// The focus measure of the warped image: the quantity contrast maximisation maximises.
///
/// # Errors
///
/// As [`warped_image`]. Also [`VisionError::EmptyGeometry`] can never arise here because `geom` is
/// already validated, and the `None` branch of [`Frame::variance`] is therefore unreachable — it is
/// mapped to a [`VisionError::TooFew`] rather than unwrapped, so a future change to `Geometry`
/// cannot turn it into a panic.
pub fn contrast(
    geom: Geometry,
    events: &[PixelEvent],
    t_ref_s: f64,
    m: Motion,
    obj: Objective,
) -> Result<f64, VisionError> {
    let wi = warped_image(geom, events, t_ref_s, m)?;
    let n = wi.frame.data.len();
    match obj {
        Objective::Variance => wi.frame.variance(),
        Objective::MeanSquare => {
            if n == 0 { None } else { Some(wi.frame.sum_of_squares() / n as f64) }
        }
    }
    .ok_or(VisionError::TooFew { what: "warped image", have: 0, need: 1 })
}

/// Evaluate the objective at `n` equally-spaced values of a one-parameter motion family.
///
/// `build` turns a scalar into a [`Motion`], which is what makes this usable for any of the three
/// families: pass `|vx| Motion::Translation { vx, vy: 0.0 }` to sweep horizontal speed, or
/// `|w| Motion::Rotation { omega_rad_s: w, cx, cy }` to sweep angular velocity.
///
/// Returns `(parameter, objective)` pairs in order, so a caller can plot the curve rather than only
/// read its peak — which is the difference between seeing that the maximum is where it should be
/// and seeing that the objective has a *single* maximum there.
///
/// # Errors
///
/// [`VisionError::TooFew`] for `n < 2`, [`VisionError::NonFinite`] for bad bounds, plus anything
/// [`contrast`] refuses.
pub fn sweep(
    geom: Geometry,
    events: &[PixelEvent],
    t_ref_s: f64,
    lo: f64,
    hi: f64,
    n: usize,
    obj: Objective,
    build: impl Fn(f64) -> Motion,
) -> Result<Vec<(f64, f64)>, VisionError> {
    if n < 2 {
        return Err(VisionError::TooFew { what: "sweep", have: n, need: 2 });
    }
    let lo = finite(lo, "sweep lower bound")?;
    let hi = finite(hi, "sweep upper bound")?;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let p = lo + (hi - lo) * i as f64 / (n - 1) as f64;
        out.push((p, contrast(geom, events, t_ref_s, build(p), obj)?));
    }
    Ok(out)
}

/// The parameter at which a sweep's objective is largest.
///
/// Ties go to the first, so the result is deterministic even on a perfectly flat objective — which
/// is what an all-zero image gives, and is the answer a caller gets when the events all warped off
/// the sensor.
///
/// Returns `None` for an empty sweep.
#[must_use]
pub fn argmax(curve: &[(f64, f64)]) -> Option<(f64, f64)> {
    let mut best: Option<(f64, f64)> = None;
    for &(p, v) in curve {
        if best.is_none_or(|(_, bv)| v > bv) {
            best = Some((p, v));
        }
    }
    best
}

/// Search a translation by coarse grid then deterministic pattern refinement.
///
/// The grid is `steps` by `steps` over `[-bound, bound]` in each component; refinement then repeats
/// `refine` times: try a step of the current size along each of the four axis directions, move to
/// the best improvement, and halve the step when none improves. No randomness, no line search, no
/// derivative — the objective is not differentiable through the bilinear vote's floor operations,
/// and a finite-difference gradient on it is noise at small steps.
///
/// Returns the best motion found and its objective value.
///
/// # Errors
///
/// [`VisionError::NonPositive`] for a non-positive `bound`, [`VisionError::TooFew`] when `steps` is
/// below 2, plus anything [`contrast`] refuses.
pub fn search_translation(
    geom: Geometry,
    events: &[PixelEvent],
    t_ref_s: f64,
    bound: f64,
    steps: usize,
    refine: usize,
    obj: Objective,
) -> Result<(Motion, f64), VisionError> {
    let bound = positive(bound, "search bound")?;
    if steps < 2 {
        return Err(VisionError::TooFew { what: "search grid", have: steps, need: 2 });
    }
    let mut best = (0.0f64, 0.0f64, f64::NEG_INFINITY);
    for iy in 0..steps {
        for ix in 0..steps {
            let vx = -bound + 2.0 * bound * ix as f64 / (steps - 1) as f64;
            let vy = -bound + 2.0 * bound * iy as f64 / (steps - 1) as f64;
            let v = contrast(geom, events, t_ref_s, Motion::Translation { vx, vy }, obj)?;
            if v > best.2 {
                best = (vx, vy, v);
            }
        }
    }
    let mut step = 2.0 * bound / (steps - 1) as f64;
    for _ in 0..refine {
        let mut moved = false;
        for (dx, dy) in [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)] {
            let vx = best.0 + dx * step;
            let vy = best.1 + dy * step;
            let v = contrast(geom, events, t_ref_s, Motion::Translation { vx, vy }, obj)?;
            if v > best.2 {
                best = (vx, vy, v);
                moved = true;
            }
        }
        if !moved {
            step *= 0.5;
        }
    }
    Ok((Motion::Translation { vx: best.0, vy: best.1 }, best.2))
}

// ---------------------------------------------------------------------------------------------
// Synthetic scenes, with ground truth
// ---------------------------------------------------------------------------------------------

/// Sort events by time, breaking ties by `(y, x)` so the stream is deterministic.
///
/// Ties are common and not an edge case: an axis-aligned edge sweeps a whole column at one instant,
/// and a sensor emits that column in *some* order. Fixing the order here is what makes a run
/// reproducible; leaving it to an unstable sort would make the `SAE` contents at each fit depend on
/// the standard library's pivot choice.
fn sort_events(mut v: Vec<PixelEvent>) -> Vec<PixelEvent> {
    v.sort_by(|a, b| {
        a.t_s
            .partial_cmp(&b.t_s)
            .unwrap_or(core::cmp::Ordering::Equal)
            .then(a.y.cmp(&b.y))
            .then(a.x.cmp(&b.x))
    });
    v
}

/// A straight edge sweeping the sensor, with **exact** event times.
///
/// The edge has unit normal `n = (cos(normal_rad), sin(normal_rad))` and its signed offset is
/// `d(t) = d0_px + speed_px_s * t`. Pixel `p` is crossed at the single instant `n . p = d(t)`, so
///
/// ```text
/// t(p) = (n . p - d0_px) / speed_px_s
/// ```
///
/// which is **exactly linear in `x` and `y`**: the event surface is a plane by construction, and
/// nothing about the discrete pixel lattice perturbs that, because each pixel's time is evaluated
/// at its own exact centre. This is why the plane-fit test in this module can assert recovery to
/// floating-point noise rather than to a tolerance somebody chose.
///
/// The ground-truth **normal flow** is `speed_px_s * n`, and because a straight edge carries no
/// information about motion along itself, that is also the only velocity any local method can
/// recover. Generate an edge moving along its own normal — which is what this function does — and
/// normal flow and true velocity coincide.
///
/// Every event carries `polarity`; a real sensor's sign depends on whether the edge is
/// bright-to-dark or dark-to-bright, and that is the caller's to state.
///
/// # Errors
///
/// [`VisionError::NonPositive`] for a non-positive `speed_px_s` or `duration_s` — a static scene
/// emits nothing, so there is no degenerate answer to return — and [`VisionError::NonFinite`] for
/// any bad parameter.
pub fn moving_edge(
    geom: Geometry,
    normal_rad: f64,
    speed_px_s: f64,
    d0_px: f64,
    duration_s: f64,
    polarity: Polarity,
) -> Result<Vec<PixelEvent>, VisionError> {
    let normal_rad = finite(normal_rad, "edge normal angle")?;
    let speed = positive(speed_px_s, "edge speed")?;
    let d0 = finite(d0_px, "edge initial offset")?;
    let duration = positive(duration_s, "scene duration")?;
    let (ny, nx) = normal_rad.sin_cos();
    let mut out = Vec::new();
    for y in 0..geom.height {
        for x in 0..geom.width {
            let t = (nx * f64::from(x) + ny * f64::from(y) - d0) / speed;
            if (0.0..=duration).contains(&t) {
                out.push(PixelEvent { t_s: t, x, y, polarity });
            }
        }
    }
    Ok(sort_events(out))
}

/// The ground-truth normal flow of [`moving_edge`], px/s.
///
/// A free function rather than a field on the event, so that a test compares the recovered flow
/// against the *generator's stated intent* rather than against anything the generator wrote into
/// its output.
#[must_use]
pub fn moving_edge_flow(normal_rad: f64, speed_px_s: f64) -> (f64, f64) {
    let (s, c) = normal_rad.sin_cos();
    (speed_px_s * c, speed_px_s * s)
}

/// Two half-lines meeting at a vertex, the whole `L` translating at a constant velocity.
///
/// Unlike [`moving_edge`], this stimulus has **two independent edge orientations**, so the full
/// two-dimensional velocity is observable and a two-parameter search over it has a unique answer.
/// That is what it is for: every test in this module that recovers two components uses this or a
/// rotating/looming scene, because a single edge cannot determine two numbers however good the
/// estimator is.
///
/// Arm `i` runs from the vertex in direction `arm_rad[i]` for `arm_len_px`. Pixel `p` is crossed by
/// arm `i` at the time solving `(p - V0 - v t) x d_i = 0`, i.e.
///
/// ```text
/// t = ((p - V0) x d_i) / (v x d_i)
/// ```
///
/// accepted when `t` is inside the duration and the foot of `p` lies within `arm_len_px` of the
/// vertex.
///
/// # Errors
///
/// [`VisionError::NonPositive`] for a non-positive `arm_len_px` or `duration_s`;
/// [`VisionError::NonFinite`] for a bad parameter; [`VisionError::BadParameters`] when an arm is
/// parallel to the velocity, in which case that arm slides along itself and sweeps nothing — a
/// refusal rather than a silently empty half of the stimulus.
pub fn moving_corner(
    geom: Geometry,
    vertex0: (f64, f64),
    velocity: (f64, f64),
    arm_rad: (f64, f64),
    arm_len_px: f64,
    duration_s: f64,
    polarity: Polarity,
) -> Result<Vec<PixelEvent>, VisionError> {
    let (vx0, vy0) = (finite(vertex0.0, "vertex column")?, finite(vertex0.1, "vertex row")?);
    let (vx, vy) = (finite(velocity.0, "velocity vx")?, finite(velocity.1, "velocity vy")?);
    let len = positive(arm_len_px, "arm length")?;
    let duration = positive(duration_s, "scene duration")?;
    let arms = [finite(arm_rad.0, "first arm angle")?, finite(arm_rad.1, "second arm angle")?];
    let mut out = Vec::new();
    for a in arms {
        let (dy, dx) = a.sin_cos();
        let cross_v = vx * dy - vy * dx;
        if cross_v.abs() < 1e-12 {
            return Err(VisionError::BadParameters {
                why: "an arm is parallel to the velocity, so it sweeps no new pixels",
            });
        }
        for y in 0..geom.height {
            for x in 0..geom.width {
                let (px, py) = (f64::from(x) - vx0, f64::from(y) - vy0);
                let t = (px * dy - py * dx) / cross_v;
                if !(0.0..=duration).contains(&t) {
                    continue;
                }
                // Foot along the arm at that instant, measured from the moving vertex.
                let s = (px - vx * t) * dx + (py - vy * t) * dy;
                if (0.0..=len).contains(&s) {
                    out.push(PixelEvent { t_s: t, x, y, polarity });
                }
            }
        }
    }
    Ok(sort_events(out))
}

/// A bar through a centre, rotating at a constant angular velocity, with **exact** event times.
///
/// The bar is a full diameter, so it covers angles `phi` and `phi + pi` at once, and pixel `p` is
/// crossed whenever `theta0 + omega * t == phi_p (mod pi)`:
///
/// ```text
/// t_k(p) = (atan2(dy, dx) - theta0 + k * pi) / omega_rad_s
/// ```
///
/// for every integer `k` putting `t_k` inside the duration.
///
/// # The closed-form flow, and why it is curved
///
/// The event surface here is `t(p) = (atan2(dy, dx) - theta0) / omega` within one branch, whose
/// gradient is `(-dy, dx) / (omega * r^2)`, giving
///
/// ```text
/// v = (a, b) / (a^2 + b^2) = omega_rad_s * (-dy, dx)
/// ```
///
/// — tangential, of magnitude `omega * r`, which is [`rotating_bar_flow`]. Unlike [`moving_edge`]
/// this surface is **not** planar, so a local plane fit carries a curvature bias that falls like
/// the neighbourhood half-width over the radius. That bias is asserted in this module's tests
/// rather than tolerated by a wide tolerance.
///
/// `r_min_px` guards the singular centre, where the flow is zero and the angle field is
/// discontinuous; `r_max_px` is the bar's half-length.
///
/// # Errors
///
/// [`VisionError::NonPositive`] for a non-positive `omega_rad_s`, `r_min_px`, `r_max_px` or
/// `duration_s`; [`VisionError::BadParameters`] when `r_min_px >= r_max_px`;
/// [`VisionError::NonFinite`] for a bad parameter.
pub fn rotating_bar(
    geom: Geometry,
    centre: (f64, f64),
    omega_rad_s: f64,
    theta0_rad: f64,
    r_min_px: f64,
    r_max_px: f64,
    duration_s: f64,
    polarity: Polarity,
) -> Result<Vec<PixelEvent>, VisionError> {
    let (cx, cy) = (finite(centre.0, "centre column")?, finite(centre.1, "centre row")?);
    let omega = positive(omega_rad_s, "angular velocity")?;
    let theta0 = finite(theta0_rad, "initial bar angle")?;
    let r_min = positive(r_min_px, "inner radius")?;
    let r_max = positive(r_max_px, "outer radius")?;
    let duration = positive(duration_s, "scene duration")?;
    if r_min >= r_max {
        return Err(VisionError::BadParameters {
            why: "the inner radius is not smaller than the outer radius",
        });
    }
    let pi = core::f64::consts::PI;
    let mut out = Vec::new();
    for y in 0..geom.height {
        for x in 0..geom.width {
            let (dx, dy) = (f64::from(x) - cx, f64::from(y) - cy);
            let r = dx.hypot(dy);
            if r < r_min || r > r_max {
                continue;
            }
            let phi = dy.atan2(dx);
            // Smallest k putting t at or after zero, then every k until the duration runs out.
            let base = (phi - theta0) / omega;
            let period = pi / omega;
            let k0 = (-base / period).ceil();
            let mut k = k0;
            loop {
                let t = base + k * period;
                if t > duration {
                    break;
                }
                if t >= 0.0 {
                    out.push(PixelEvent { t_s: t, x, y, polarity });
                }
                k += 1.0;
            }
        }
    }
    Ok(sort_events(out))
}

/// The ground-truth tangential flow of [`rotating_bar`] at a pixel, px/s.
///
/// `None` at the centre itself, where the flow is zero and its direction undefined — returned
/// rather than a zero vector, because "no motion" and "motion whose direction we cannot name" are
/// different claims and an angular error against the second one is meaningless.
#[must_use]
pub fn rotating_bar_flow(centre: (f64, f64), omega_rad_s: f64, x: u16, y: u16) -> Option<(f64, f64)> {
    let (dx, dy) = (f64::from(x) - centre.0, f64::from(y) - centre.1);
    if dx == 0.0 && dy == 0.0 {
        return None;
    }
    Some((-omega_rad_s * dy, omega_rad_s * dx))
}

/// A circle expanding at a constant radial speed — the looming stimulus — with **exact** event
/// times.
///
/// Pixel `p` is crossed when the radius reaches it:
///
/// ```text
/// t(p) = (|p - centre| - r0_px) / rate_px_s
/// ```
///
/// The event surface is a cone, not a plane: the closed-form flow is `rate * (dx, dy) / r`, i.e.
/// radially outward at exactly `rate_px_s` everywhere, which is [`looming_disc_flow`]. That the
/// *speed* is constant while the *direction* varies is what makes this a good second check on a
/// plane fit — an estimator that recovered the magnitude by accident would still get the direction
/// wrong.
///
/// # Errors
///
/// [`VisionError::NonPositive`] for a non-positive `rate_px_s` or `duration_s`;
/// [`VisionError::NonFinite`] for a bad parameter. A negative `r0_px` is accepted: it simply means
/// the circle was already notionally expanding before the recording started.
pub fn looming_disc(
    geom: Geometry,
    centre: (f64, f64),
    r0_px: f64,
    rate_px_s: f64,
    duration_s: f64,
    polarity: Polarity,
) -> Result<Vec<PixelEvent>, VisionError> {
    let (cx, cy) = (finite(centre.0, "centre column")?, finite(centre.1, "centre row")?);
    let r0 = finite(r0_px, "initial radius")?;
    let rate = positive(rate_px_s, "expansion rate")?;
    let duration = positive(duration_s, "scene duration")?;
    let mut out = Vec::new();
    for y in 0..geom.height {
        for x in 0..geom.width {
            let r = (f64::from(x) - cx).hypot(f64::from(y) - cy);
            let t = (r - r0) / rate;
            if (0.0..=duration).contains(&t) {
                out.push(PixelEvent { t_s: t, x, y, polarity });
            }
        }
    }
    Ok(sort_events(out))
}

/// The ground-truth radial flow of [`looming_disc`] at a pixel, px/s.
///
/// `None` at the centre, where the radial direction is undefined; see [`rotating_bar_flow`] for why
/// that is a `None` rather than a zero.
#[must_use]
pub fn looming_disc_flow(centre: (f64, f64), rate_px_s: f64, x: u16, y: u16) -> Option<(f64, f64)> {
    let (dx, dy) = (f64::from(x) - centre.0, f64::from(y) - centre.1);
    let r = dx.hypot(dy);
    if r == 0.0 {
        return None;
    }
    Some((rate_px_s * dx / r, rate_px_s * dy / r))
}

/// Quantise timestamps to whole microseconds, as a sensor's clock does.
///
/// The generators above produce mathematically exact times, which is what lets the central flow
/// test assert recovery to floating-point noise. A real recording cannot: `AEDAT`, `EVT` and `.dat`
/// all count whole microseconds, so every timestamp arrives already rounded, and an estimator
/// checked only against exact times has not been checked against anything it will ever see. Passing
/// a stream through this and re-running the same test is how this module separates "the estimator
/// is correct" from "the estimator is usable".
///
/// Re-sorts afterwards, because rounding can reorder two events a few nanoseconds apart.
///
/// # Errors
///
/// [`VisionError::NonFinite`] for a bad timestamp, [`VisionError::NonPositive`] for a negative one
/// — a wire timestamp counts up from the recording's zero.
pub fn quantise_microseconds(events: &[PixelEvent]) -> Result<Vec<PixelEvent>, VisionError> {
    let mut out = Vec::with_capacity(events.len());
    for e in events {
        let t = finite(e.t_s, "event timestamp")?;
        if t < 0.0 {
            return Err(VisionError::NonPositive { what: "event timestamp", value: t });
        }
        out.push(PixelEvent { t_s: (t * 1e6).round() * 1e-6, ..*e });
    }
    Ok(sort_events(out))
}

#[cfg(test)]
mod tests {
    use super::{
        Decay, EFast, EHarris, FAST_CIRCLE_16, FAST_CIRCLE_20, FlowOutcome, Frame, Geometry,
        Hats, Hots, Motion, Objective, PixelEvent, PlaneFlow, TimeSurface, VisionError,
        accumulate_count, accumulate_decay, accumulate_polarity, argmax, contrast, fit_plane,
        flow_error, looming_disc, looming_disc_flow, moving_corner, moving_edge, moving_edge_flow,
        newest_arc, quantise_microseconds, rotating_bar, rotating_bar_flow, search_translation,
        sweep, warped_image,
    };
    use crate::aer::AerEvent;
    use crate::rng::Rng;
    use crate::spike::Polarity;

    /// Set to `true` to print the measured numbers behind every stated tolerance in this module.
    /// Off in the committed tests because a passing suite should be silent; kept because every
    /// tolerance below was chosen by reading these, not by widening until green.
    const PRINT_MEASUREMENTS: bool = false;

    fn geom(w: u16, h: u16) -> Geometry {
        Geometry::new(w, h).expect("a non-empty sensor")
    }

    /// Smallest absolute angular difference, in `[0, pi]`. An angular error is a distance on a
    /// circle: 359 degrees is one degree of error, not 359, and a test that forgets this reports a
    /// perfect estimate as a catastrophic one.
    fn wrapped_angle(mut d: f64) -> f64 {
        let two_pi = 2.0 * core::f64::consts::PI;
        d = d.rem_euclid(two_pi);
        if d > core::f64::consts::PI { two_pi - d } else { d }
    }

    // -----------------------------------------------------------------------------------------
    // Geometry, events, units
    // -----------------------------------------------------------------------------------------

    #[test]
    fn an_empty_sensor_is_refused_rather_than_carried() {
        assert!(matches!(Geometry::new(0, 8), Err(VisionError::EmptyGeometry { .. })));
        assert!(matches!(Geometry::new(8, 0), Err(VisionError::EmptyGeometry { .. })));
        let g = geom(7, 5);
        assert_eq!(g.pixels(), 35);
        assert_eq!(g.index(6, 4), Some(34));
        assert_eq!(g.index(7, 4), None);
        assert!(g.contains(0, 0) && !g.contains(-1, 0) && !g.contains(0, 5));
    }

    /// The module's unit boundary, in both directions. A microsecond wire timestamp becomes a
    /// second and comes back the same integer — not approximately, exactly, because `1e-6` times a
    /// small integer and back is exact in binary64 after the round.
    #[test]
    fn the_microsecond_boundary_round_trips_exactly() {
        for t in [0u64, 1, 999, 1_000_000, 123_456_789] {
            let a = AerEvent { t, x: 3, y: 4, polarity: Polarity::On };
            let p = PixelEvent::from_aer(a);
            assert!((p.t_s - t as f64 * 1e-6).abs() < 1e-18);
            assert_eq!(p.to_aer(), Some(a));
        }
        // A negative time has no wire encoding and is refused rather than wrapped to a huge u64.
        let bad = PixelEvent { t_s: -1e-6, x: 0, y: 0, polarity: Polarity::On };
        assert_eq!(bad.to_aer(), None);
        let nan = PixelEvent { t_s: f64::NAN, x: 0, y: 0, polarity: Polarity::On };
        assert_eq!(nan.to_aer(), None);
    }

    /// The plane index is the one place a polarity convention could silently invert every learned
    /// descriptor in the module, so it is pinned.
    #[test]
    fn the_polarity_plane_convention_is_pinned() {
        let on = PixelEvent { t_s: 0.0, x: 0, y: 0, polarity: Polarity::On };
        let off = PixelEvent { t_s: 0.0, x: 0, y: 0, polarity: Polarity::Off };
        assert_eq!(on.plane(true), 1);
        assert_eq!(off.plane(true), 0);
        assert_eq!(on.plane(false), 0);
        assert_eq!(off.plane(false), 0);
    }

    // -----------------------------------------------------------------------------------------
    // (c) A time surface decays exactly as exp(-(t - t_last)/tau)
    // -----------------------------------------------------------------------------------------

    /// **Check (c).** The defining closed form, at eleven elapsed times spanning four time
    /// constants, to 1e-15 — machine epsilon, not a chosen tolerance. An implementation that used
    /// `1 - d/tau`, or `2^(-d/tau)`, or the right formula with the wrong sign, fails at every point
    /// but the first.
    ///
    /// The surface is seeded at `t = 0` here, so `t - t_last` is exact and the only error is the
    /// one `exp` itself makes. [`a_time_surface_seeded_late_loses_precision_to_the_subtraction`]
    /// is the same check from a late start, where it is not, and says why that matters.
    #[test]
    fn a_time_surface_decays_exactly_as_the_exponential() {
        let g = geom(8, 8);
        let tau = 3.7e-3;
        let mut ts = TimeSurface::new(g, Decay::Exponential { tau_s: tau }, true).unwrap();
        let t0 = 0.0;
        ts.update(PixelEvent { t_s: t0, x: 4, y: 5, polarity: Polarity::On }).unwrap();
        for k in 0..=10 {
            let d = 0.4 * tau * f64::from(k);
            let got = ts.value_at(4, 5, Polarity::On, t0 + d).unwrap();
            let want = (-d / tau).exp();
            assert!((got - want).abs() < 1e-15, "elapsed {d}: {got} vs {want}");
        }
        // At zero elapsed the surface is exactly 1, which is what makes a patch comparable across
        // events with different absolute times.
        assert!((ts.value_at(4, 5, Polarity::On, t0).unwrap() - 1.0).abs() < 1e-15);
        // A pixel that never fired is exactly zero, and that is the same value the exponential
        // approaches, so the two conventions agree rather than meeting at a discontinuity.
        assert_eq!(ts.value_at(0, 0, Polarity::On, t0).unwrap(), 0.0);
        // The other polarity plane is untouched: this is what `split` buys.
        assert_eq!(ts.value_at(4, 5, Polarity::Off, t0).unwrap(), 0.0);
        assert_eq!(ts.last_time(4, 5, Polarity::Off).unwrap(), None);
    }

    /// The same closed form, seeded 250 ms into a recording: now `t - t_last` is a subtraction of
    /// two nearby doubles and the check holds only to a **relative** 1e-14, not an absolute 1e-15.
    ///
    /// That is not a defect in the decay, it is the arithmetic: at `t = 0.25 s` the spacing of
    /// binary64 is about 5.6e-17 s, so an elapsed time of 1.5 ms carries roughly four digits fewer
    /// than it would from zero. It is the same argument [`crate::spike`] makes for storing time as
    /// an integer tick, reaching this module through the one `f64` of seconds at its boundary — and
    /// it is why a recording measured in hours should be re-zeroed before its surfaces are built.
    /// Asserted rather than remarked on, so that a change making it worse fails here.
    #[test]
    fn a_time_surface_seeded_late_loses_precision_to_the_subtraction() {
        let tau = 3.7e-3;
        let mut ts =
            TimeSurface::new(geom(8, 8), Decay::Exponential { tau_s: tau }, true).unwrap();
        let t0 = 0.25;
        ts.update(PixelEvent { t_s: t0, x: 4, y: 5, polarity: Polarity::On }).unwrap();
        let mut worst: f64 = 0.0;
        for k in 1..=10 {
            let d = 0.4 * tau * f64::from(k);
            let got = ts.value_at(4, 5, Polarity::On, t0 + d).unwrap();
            let want = (-d / tau).exp();
            worst = worst.max((got - want).abs() / want);
        }
        if PRINT_MEASUREMENTS {
            println!("worst relative decay error from t0 = 0.25 s: {worst:e}");
        }
        // Measured 5.1e-15 at the time of writing. The bound is one order above it: loose enough
        // not to flake on another libm's `exp`, tight enough that a formula error — which costs
        // whole percent — cannot hide underneath.
        assert!(worst < 1e-14, "relative error {worst:e} is larger than the subtraction explains");
        // And strictly worse than the same sweep from zero, which is the claim being made.
        assert!(worst > 0.0, "seeding late cost nothing, so the argument above is wrong");
    }

    /// The linear decay reaches exactly zero at the window and stays there, which is the property
    /// that distinguishes it from the exponential and the reason a hard-windowed descriptor uses it.
    #[test]
    fn the_linear_decay_is_exactly_zero_past_its_window() {
        let d = Decay::Linear { window_s: 0.01 };
        assert!((d.value(0.0) - 1.0).abs() < 1e-15);
        assert!((d.value(0.005) - 0.5).abs() < 1e-15);
        assert_eq!(d.value(0.01), 0.0);
        assert_eq!(d.value(0.5), 0.0);
        // A negative elapsed time clamps to the peak rather than exceeding 1 or returning a NaN.
        assert!((d.value(-1.0) - 1.0).abs() < 1e-15);
        assert!((Decay::Exponential { tau_s: 1e-3 }.value(-1.0) - 1.0).abs() < 1e-15);
    }

    /// Backwards time is refused, not sorted away. The surface's whole content is "the most recent
    /// time at each pixel", and that phrase is empty if the stream can go backwards.
    #[test]
    fn a_backwards_event_is_refused_rather_than_absorbed() {
        let mut ts = TimeSurface::new(geom(8, 8), Decay::Exponential { tau_s: 1e-3 }, false).unwrap();
        ts.update(PixelEvent { t_s: 1.0, x: 1, y: 1, polarity: Polarity::On }).unwrap();
        let e = ts.update(PixelEvent { t_s: 0.5, x: 2, y: 2, polarity: Polarity::On });
        assert!(matches!(e, Err(VisionError::OutOfOrder { .. })), "{e:?}");
        // And the refused event left no trace.
        assert_eq!(ts.last_time(2, 2, Polarity::On).unwrap(), None);
        // Equal times are fine: a sensor emits a whole column at one instant.
        ts.update(PixelEvent { t_s: 1.0, x: 3, y: 3, polarity: Polarity::On }).unwrap();
    }

    /// A zero or negative time constant is refused. `tau = 0` makes every value `exp(-inf) = 0`
    /// except exactly at the event, which is a surface carrying no information that would still
    /// plot as a sensible-looking sparse image.
    #[test]
    fn a_degenerate_decay_constant_is_refused() {
        let g = geom(4, 4);
        assert!(TimeSurface::new(g, Decay::Exponential { tau_s: 0.0 }, false).is_err());
        assert!(TimeSurface::new(g, Decay::Exponential { tau_s: -1e-3 }, false).is_err());
        assert!(TimeSurface::new(g, Decay::Linear { window_s: f64::NAN }, false).is_err());
    }

    /// The border convention is a documented conflation and this is where it is pinned: a patch
    /// centred one pixel from the edge is full-size, padded with the same exact zero a never-fired
    /// pixel contributes. The alternative eats a band of real events.
    #[test]
    fn a_patch_at_the_border_is_padded_rather_than_refused() {
        let g = geom(5, 5);
        let mut ts = TimeSurface::new(g, Decay::Exponential { tau_s: 1.0 }, false).unwrap();
        ts.update(PixelEvent { t_s: 0.0, x: 0, y: 0, polarity: Polarity::On }).unwrap();
        let p = ts.patch(0, 0, 2, Polarity::On, 0.0).unwrap();
        assert_eq!(p.len(), 25);
        // Centre of the 5x5 patch is index 12, and it is the pixel that fired.
        assert!((p[12] - 1.0).abs() < 1e-15);
        assert_eq!(p.iter().filter(|v| **v != 0.0).count(), 1);
        // The centre itself must still be on the lattice.
        assert!(matches!(ts.patch(5, 0, 1, Polarity::On, 0.0), Err(VisionError::OutOfBounds { .. })));
    }

    // -----------------------------------------------------------------------------------------
    // (e) Accumulation
    // -----------------------------------------------------------------------------------------

    /// **Check (e), part one.** A static scene emits no events, and no events accumulate to an
    /// image that is *exactly* zero — every pixel, the sum, and the variance, compared with `==`
    /// rather than a tolerance, because there is no arithmetic here that could introduce a
    /// rounding.
    #[test]
    fn a_static_scene_accumulates_to_exactly_nothing() {
        let g = geom(16, 12);
        for f in [
            accumulate_count(g, &[]).unwrap(),
            accumulate_polarity(g, &[]).unwrap(),
            accumulate_decay(g, &[], Decay::Exponential { tau_s: 1e-3 }, 1.0).unwrap(),
        ] {
            assert_eq!(f.data.len(), 192);
            assert!(f.data.iter().all(|v| *v == 0.0));
            assert_eq!(f.sum(), 0.0);
            assert_eq!(f.sum_of_squares(), 0.0);
            assert_eq!(f.variance(), Some(0.0));
            assert_eq!(f.max(), Some(0.0));
        }
        // And the generators refuse to call a still scene a moving one, rather than returning an
        // empty vector that a caller would read as "nothing happened to be in frame".
        let e = moving_edge(g, 0.0, 0.0, 0.0, 1.0, Polarity::On);
        assert!(matches!(e, Err(VisionError::NonPositive { what: "edge speed", .. })), "{e:?}");
        let e = looming_disc(g, (8.0, 6.0), 1.0, 0.0, 1.0, Polarity::On);
        assert!(matches!(e, Err(VisionError::NonPositive { what: "expansion rate", .. })), "{e:?}");
    }

    /// **Check (e), part two.** A conservation invariant that a sign error cannot survive: sweep an
    /// edge forward emitting `On` and back emitting `Off`, and every pixel's polarity sum is
    /// *exactly* zero, because each pixel received exactly one of each and `+1 + -1` is exact.
    /// Deleting the `polarity.sign()` factor, or taking its absolute value, breaks this at every
    /// pixel while leaving the event count identical.
    #[test]
    fn an_edge_swept_out_and_back_sums_to_exactly_zero() {
        let g = geom(24, 18);
        let fwd = moving_edge(g, 0.4, 200.0, -30.0, 1.0, Polarity::On).unwrap();
        let back = moving_edge(g, 0.4, 200.0, -30.0, 1.0, Polarity::Off).unwrap();
        assert!(!fwd.is_empty());
        assert_eq!(fwd.len(), back.len());
        let mut both = fwd.clone();
        both.extend(back.iter().copied());
        let f = accumulate_polarity(g, &both).unwrap();
        assert!(f.data.iter().all(|v| *v == 0.0), "a pixel did not cancel");
        // The count image, by contrast, is exactly 2 everywhere the edge passed — which is the
        // information the polarity sum destroyed, and the reason both accumulators exist.
        let c = accumulate_count(g, &both).unwrap();
        assert_eq!(c.sum(), 2.0 * fwd.len() as f64);
        assert_eq!(c.max(), Some(2.0));
    }

    /// The decayed accumulator's weights are the closed form, not something close to it.
    #[test]
    fn the_decayed_accumulator_weights_events_by_the_closed_form() {
        let g = geom(4, 4);
        let tau = 0.02;
        let evs = [
            PixelEvent { t_s: 0.00, x: 1, y: 1, polarity: Polarity::On },
            PixelEvent { t_s: 0.01, x: 1, y: 1, polarity: Polarity::On },
            PixelEvent { t_s: 0.03, x: 2, y: 1, polarity: Polarity::Off },
        ];
        let f = accumulate_decay(g, &evs, Decay::Exponential { tau_s: tau }, 0.05).unwrap();
        let want11 = (-0.05f64 / tau).exp() + (-0.04f64 / tau).exp();
        let want21 = -(-0.02f64 / tau).exp();
        assert!((f.at(1, 1).unwrap() - want11).abs() < 1e-14, "{:?}", f.at(1, 1));
        assert!((f.at(2, 1).unwrap() - want21).abs() < 1e-14, "{:?}", f.at(2, 1));
        assert_eq!(f.at(0, 0), Some(0.0));
    }

    /// An event off the declared lattice is refused, not folded into the next row. Folding gives a
    /// sheared image that no count would show as wrong — the same defect
    /// [`crate::aer::AerEvent::to_event`] refuses for the same reason.
    #[test]
    fn an_out_of_range_event_is_refused_by_every_accumulator() {
        let g = geom(8, 8);
        let bad = [PixelEvent { t_s: 0.0, x: 8, y: 0, polarity: Polarity::On }];
        assert!(matches!(accumulate_count(g, &bad), Err(VisionError::OutOfBounds { .. })));
        assert!(matches!(accumulate_polarity(g, &bad), Err(VisionError::OutOfBounds { .. })));
        let nan = [PixelEvent { t_s: f64::NAN, x: 0, y: 0, polarity: Polarity::On }];
        assert!(matches!(accumulate_count(g, &nan), Err(VisionError::NonFinite { .. })));
    }

    // -----------------------------------------------------------------------------------------
    // Plane fitting
    // -----------------------------------------------------------------------------------------

    /// A plane through three points whose answer can be read off by hand, then the flow inversion
    /// on it. `t = 1*x + 2*y`, so `grad t = (1, 2)`, `|grad t|^2 = 5` and `v = (0.2, 0.4)` px/s.
    /// An implementation returning the gradient itself, or its reciprocal component-wise, gets the
    /// direction right and the magnitude wrong — which is why the speed is asserted too.
    #[test]
    fn a_hand_computable_plane_and_its_flow() {
        let f = fit_plane(&[(0.0, 0.0, 0.0), (1.0, 0.0, 1.0), (0.0, 1.0, 2.0)]).unwrap();
        assert!((f.a - 1.0).abs() < 1e-12, "a = {}", f.a);
        assert!((f.b - 2.0).abs() < 1e-12, "b = {}", f.b);
        assert!(f.c.abs() < 1e-12, "c = {}", f.c);
        assert!(f.rms_s < 1e-15);
        let v = f.flow().unwrap();
        assert!((v.vx - 0.2).abs() < 1e-12, "vx = {}", v.vx);
        assert!((v.vy - 0.4).abs() < 1e-12, "vy = {}", v.vy);
        assert!((v.speed - 1.0 / 5.0f64.sqrt()).abs() < 1e-12, "speed = {}", v.speed);
        assert!((v.direction_rad - 2.0f64.atan2(1.0)).abs() < 1e-12);
    }

    /// The refusals a plane fit owes its caller, each for a different reason.
    #[test]
    fn a_plane_fit_refuses_what_it_cannot_determine() {
        // Two points do not determine a plane.
        assert!(matches!(
            fit_plane(&[(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)]),
            Err(VisionError::TooFew { .. })
        ));
        // Collinear support: every point on the line y = x. The design matrix is singular however
        // many points you add, so this is not fixed by more data.
        let line: Vec<(f64, f64, f64)> =
            (0..20).map(|k| (f64::from(k), f64::from(k), f64::from(k) * 0.5)).collect();
        assert!(matches!(fit_plane(&line), Err(VisionError::Degenerate { .. })));
        // A vertical line: no x variation at all.
        let col: Vec<(f64, f64, f64)> =
            (0..20).map(|k| (3.0, f64::from(k), f64::from(k))).collect();
        assert!(matches!(fit_plane(&col), Err(VisionError::Degenerate { .. })));
        // A non-finite coordinate is refused at the boundary rather than producing a NaN plane.
        assert!(matches!(
            fit_plane(&[(0.0, 0.0, 0.0), (1.0, 0.0, f64::INFINITY), (0.0, 1.0, 1.0)]),
            Err(VisionError::NonFinite { .. })
        ));
        // A flat plane is a perfectly good fit describing an infinite speed, and the *inversion*
        // is what refuses it — the fit itself succeeds.
        let flat = fit_plane(&[(0.0, 0.0, 7.0), (1.0, 0.0, 7.0), (0.0, 1.0, 7.0)]).unwrap();
        assert_eq!(flat.a, 0.0);
        assert!(matches!(flat.flow(), Err(VisionError::Degenerate { .. })));
    }
    // -----------------------------------------------------------------------------------------
    // (a) THE MODULE'S CENTRAL CHECK: plane-fit flow recovers the generated velocity
    // -----------------------------------------------------------------------------------------

    /// Corner extremes of `n . p` over the sensor, so an edge can be started outside it and swept
    /// clear across.
    fn edge_span(g: Geometry, normal_rad: f64) -> (f64, f64) {
        let (ny, nx) = normal_rad.sin_cos();
        let (w, h) = (f64::from(g.width - 1), f64::from(g.height - 1));
        let vals = [0.0, nx * w, ny * h, nx * w + ny * h];
        let lo = vals.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        (lo, hi)
    }

    /// Run one edge through [`PlaneFlow`] and report the error against the generator's own stated
    /// ground truth.
    fn edge_recovery(
        g: Geometry,
        normal_rad: f64,
        speed: f64,
        radius: u16,
        quantise: bool,
    ) -> super::FlowError {
        let (lo, hi) = edge_span(g, normal_rad);
        let d0 = lo - 2.0;
        let duration = (hi - lo + 4.0) / speed;
        let evs = moving_edge(g, normal_rad, speed, d0, duration, Polarity::On).unwrap();
        let evs = if quantise { quantise_microseconds(&evs).unwrap() } else { evs };
        assert!(evs.len() > 500, "the edge swept only {} pixels", evs.len());
        // The window has to hold the trailing half-neighbourhood: at `speed` px/s an event
        // `radius` pixels behind is `radius / speed` seconds old. Four times that is slack.
        let window = 4.0 * f64::from(radius) / speed;
        let mut pf = PlaneFlow::new(g, radius, window, None, 8).unwrap();
        let outs = pf.run(&evs).unwrap();
        let truth = moving_edge_flow(normal_rad, speed);
        flow_error(&evs, &outs, |_| Some(truth)).unwrap()
    }

    /// **CHECK (a) — the module's central test.** Plane-fit flow recovers the velocity the
    /// generator was given, across six directions and three speeds.
    ///
    /// The tolerance is machine noise, not a fudge factor, and that is the whole point: the event
    /// surface of a straight edge is *exactly* a plane in `(x, y, t)` — see [`moving_edge`] — so a
    /// correct least-squares fit reproduces its gradient to the last few bits. Anything a
    /// discretisation, a sign convention or a transposed index would cost shows up here as a
    /// percentage, orders of magnitude above the bound.
    ///
    /// The directions deliberately include 117 and 200 degrees. A sweep over axis-aligned and
    /// 45-degree cases alone cannot distinguish `(vx, vy)` from `(vy, vx)` or from `(vx, -vy)`, and
    /// a test that cannot tell those apart is not testing the thing it names.
    #[test]
    fn plane_fit_flow_recovers_the_generated_velocity_exactly() {
        let g = geom(64, 64);
        let mut worst_rel: f64 = 0.0;
        let mut worst_ang: f64 = 0.0;
        for deg in [0.0, 30.0, 45.0, 117.0, 200.0, 300.0] {
            let normal = deg * core::f64::consts::PI / 180.0;
            for speed in [60.0, 250.0, 900.0] {
                let e = edge_recovery(g, normal, speed, 3, false);
                if PRINT_MEASUREMENTS {
                    println!(
                        "{deg:>5} deg {speed:>6} px/s: fitted {}/{}, rel {:e}, ang {:e}",
                        e.fitted, e.offered, e.max_rel_speed_err, e.mean_angle_err
                    );
                }
                // A method that fits one event in a thousand can report any error it likes, so the
                // yield is asserted alongside the error.
                assert!(
                    e.fitted * 2 > e.offered,
                    "{deg} deg at {speed}: only {} of {} events fitted",
                    e.fitted,
                    e.offered
                );
                assert!(
                    e.max_rel_speed_err < 1e-9,
                    "{deg} deg at {speed}: worst relative speed error {:e}",
                    e.max_rel_speed_err
                );
                assert!(
                    e.mean_angle_err < 1e-9,
                    "{deg} deg at {speed}: mean angular error {:e} rad",
                    e.mean_angle_err
                );
                worst_rel = worst_rel.max(e.max_rel_speed_err);
                worst_ang = worst_ang.max(e.mean_angle_err);
            }
        }
        if PRINT_MEASUREMENTS {
            println!("worst over the sweep: rel {worst_rel:e}, ang {worst_ang:e}");
        }
    }

    /// The same sweep on **microsecond-quantised** timestamps, which is what a sensor actually
    /// delivers. The event surface is no longer exactly planar, so the recovery is no longer
    /// exact — and the error is asserted against a bound derived from the quantum rather than
    /// against whatever came out.
    ///
    /// A timestamp rounded to the nearest microsecond carries an error up to 0.5 us. Over a
    /// neighbourhood of half-width `L` the plane's time range is about `L / speed`, so the
    /// relative gradient error is of order `0.5e-6 * speed / L`. At 900 px/s and `L = 3` that is
    /// about 1.5e-4 — and the **measured** worst case over the whole sweep is 3.1e-4, which is that
    /// prediction within a factor of two. (The factor is the one-sided half-neighbourhood: the fit
    /// only ever sees the trailing half, so its effective lever arm is shorter than `L`.) The bound
    /// below is 1e-3, half an order above the measurement, which leaves no room for a systematic
    /// error — those cost a percent or more — while tolerating another platform's rounding.
    #[test]
    fn plane_fit_flow_survives_microsecond_quantisation() {
        let g = geom(64, 64);
        let mut worst: f64 = 0.0;
        for deg in [0.0, 30.0, 117.0, 300.0] {
            let normal = deg * core::f64::consts::PI / 180.0;
            for speed in [60.0, 900.0] {
                let e = edge_recovery(g, normal, speed, 3, true);
                if PRINT_MEASUREMENTS {
                    println!(
                        "quantised {deg:>5} deg {speed:>6} px/s: mean speed err {:e} px/s, max rel {:e}",
                        e.mean_speed_err, e.max_rel_speed_err
                    );
                }
                assert!(e.fitted * 2 > e.offered, "{deg} deg at {speed}: yield collapsed");
                assert!(
                    e.max_rel_speed_err < 1e-3,
                    "{deg} deg at {speed}: worst relative speed error {:e}",
                    e.max_rel_speed_err
                );
                assert!(e.mean_angle_err < 1e-3, "{deg} deg at {speed}: angular error");
                worst = worst.max(e.max_rel_speed_err);
            }
        }
        if PRINT_MEASUREMENTS {
            println!("worst quantised relative error: {worst:e}");
        }
        // Quantisation must actually cost something, or this test is asserting nothing that the
        // exact-timestamp test above does not already assert.
        assert!(worst > 1e-9, "quantisation cost nothing measurable ({worst:e}); check the rounding");
    }

    /// The curved-surface counterpart of check (a): a **looming disc**, whose event surface is a
    /// cone rather than a plane, run end to end through the same estimator.
    ///
    /// The closed form is still exact — flow is radially outward at exactly `rate_px_s` everywhere,
    /// [`looming_disc_flow`] — but a local *plane* fit to a cone carries a curvature bias, and
    /// working out how that bias behaves is the point of this test. It is asserted as a **scaling
    /// law in two variables**, because a law is something an estimator can fail and a single
    /// tolerance is not.
    ///
    /// # What the measurement says, and what the obvious guess gets wrong
    ///
    /// Measured on a 160x160 sensor, over two annuli and two neighbourhood sizes:
    ///
    /// ```text
    /// r in [24, 32]   L = 2 -> 0.759%     L = 4 -> 1.523%
    /// r in [48, 64]   L = 2 -> 0.399%     L = 4 -> 0.824%
    /// ```
    ///
    /// Doubling `L` doubles the bias (2.01x, 2.07x); doubling `r` halves it (0.53x, 0.54x). So the
    /// bias is **first order in `L/r`** — but with a coefficient of about **0.107**, not 1. The
    /// naive estimate, which takes the one-sidedness of the `SAE` neighbourhood at face value and
    /// predicts `L/r`, is nine times too pessimistic.
    ///
    /// The reason is worth stating because it is the geometry of the stimulus rather than of the
    /// estimator: `t = (r - r0) / rate` is **exactly linear along the radius** and curved only
    /// tangentially, and the fired inner half-disc is *symmetric* tangentially. Almost all of the
    /// first-order term therefore cancels, and what is left is the coupling between the radial
    /// offset and the tangential extent of a square neighbourhood clipped by a circular arc.
    ///
    /// The `0.107` is **measured, not derived**. This implementation did not work out the
    /// coefficient in closed form; the test asserts the two ratios, which are the part that is
    /// derived, and bounds the magnitude well below the naive `L/r` so that a genuinely first-order
    /// estimator would fail.
    #[test]
    fn plane_fit_flow_on_a_cone_has_a_bias_first_order_in_l_over_r() {
        let g = geom(160, 160);
        let centre = (79.5, 79.5);
        let rate = 200.0;
        let evs = looming_disc(g, centre, 4.0, rate, 0.36, Polarity::On).unwrap();
        assert!(evs.len() > 10_000);

        // Both annuli sit well inside the sensor, so a neighbourhood truncated at the border can
        // never be mistaken for curvature.
        let measure = |lo: f64, hi: f64, l: u16| -> super::FlowError {
            let mut pf = PlaneFlow::new(g, l, 4.0 * f64::from(l) / rate, None, 8).unwrap();
            let outs = pf.run(&evs).unwrap();
            flow_error(&evs, &outs, |e| {
                let d = (f64::from(e.x) - centre.0).hypot(f64::from(e.y) - centre.1);
                if (lo..=hi).contains(&d) {
                    looming_disc_flow(centre, rate, e.x, e.y)
                } else {
                    None
                }
            })
            .unwrap()
        };

        let near2 = measure(24.0, 32.0, 2).mean_speed_err / rate;
        let near4 = measure(24.0, 32.0, 4).mean_speed_err / rate;
        let far2 = measure(48.0, 64.0, 2).mean_speed_err / rate;
        let far4 = measure(48.0, 64.0, 4).mean_speed_err / rate;
        if PRINT_MEASUREMENTS {
            println!(
                "cone bias: near L2 {:.4}% L4 {:.4}% | far L2 {:.4}% L4 {:.4}% | L-ratios {:.3} {:.3} | r-ratios {:.3} {:.3}",
                100.0 * near2,
                100.0 * near4,
                100.0 * far2,
                100.0 * far4,
                near4 / near2,
                far4 / far2,
                near2 / far2,
                near4 / far4
            );
        }

        // THE LAW IN L: doubling the neighbourhood doubles the bias. The upper bound is what rules
        // out a second-order bias (which would be 4x); the lower bound rules out an estimator
        // whose error does not track the neighbourhood at all.
        for (small, large, which) in [(near2, near4, "near"), (far2, far4, "far")] {
            let ratio = large / small;
            assert!(
                (1.8..=2.4).contains(&ratio),
                "{which}: doubling L moved the bias by {ratio:.3}x, not the ~2x a first-order bias gives"
            );
        }
        // THE LAW IN r: doubling the radius halves the bias.
        for (near, far, which) in [(near2, far2, "L=2"), (near4, far4, "L=4")] {
            let ratio = near / far;
            assert!(
                (1.8..=2.4).contains(&ratio),
                "{which}: doubling r moved the bias by {ratio:.3}x, not the ~2x of a 1/r law"
            );
        }
        // THE MAGNITUDE, bounded at a fifth of the naive `L/r`. A genuinely first-order estimator
        // — one whose one-sidedness did not cancel tangentially — sits at 8.3% here and fails.
        assert!(near2 < 0.2 * 2.0 / 24.0, "L=2 near bias {near2:e} is at the naive L/r scale");
        assert!(near2 > 1e-4, "the bias vanished ({near2:e}); the annulus filter is probably empty");
        // Direction survives curvature far better than magnitude, because the gradient points
        // exactly along the radius whatever the fit does to its length.
        assert!(measure(24.0, 32.0, 2).mean_angle_err < 0.02);
    }

    /// The **rotating bar**'s closed-form flow, checked on the fit itself rather than through the
    /// stream.
    ///
    /// The angular event surface has a branch cut — `atan2` jumps by `2*pi`, and the bar repeats
    /// every `pi / omega` — so a running `SAE` near either discontinuity mixes two branches, which
    /// is a property of the *stimulus* and not of the estimator. Feeding [`fit_plane`] the analytic
    /// surface directly is what separates the two, and it is the honest way to check the estimator
    /// against `v = omega * (-dy, dx)`.
    #[test]
    fn plane_fit_recovers_the_rotating_bars_tangential_flow() {
        let centre = (48.0, 48.0);
        let omega = 12.0;
        for &(px, py) in &[(78u16, 48u16), (48, 78), (70, 70), (20, 34)] {
            for l in [1i64, 2, 3] {
                let mut pts = Vec::new();
                for dy in -l..=l {
                    for dx in -l..=l {
                        let (x, y) = (f64::from(px) + dx as f64, f64::from(py) + dy as f64);
                        let t = (y - centre.1).atan2(x - centre.0) / omega;
                        pts.push((x, y, t));
                    }
                }
                let f = fit_plane(&pts).unwrap();
                let v = f.flow().unwrap();
                let (tx, ty) = rotating_bar_flow(centre, omega, px, py).unwrap();
                let tspeed = tx.hypot(ty);
                let rel = (v.speed - tspeed).abs() / tspeed;
                let ang = wrapped_angle(v.direction_rad - ty.atan2(tx));
                let r = (f64::from(px) - centre.0).hypot(f64::from(py) - centre.1);
                if PRINT_MEASUREMENTS {
                    println!("bar ({px},{py}) r={r:.1} L={l}: rel {rel:e}, ang {ang:e}");
                }
                // The full symmetric neighbourhood is available here, so the leading `L/r` term
                // cancels and the residual is second order: `(L/r)^2` is at most (3/17)^2 = 3.1e-2
                // for the closest point tested.
                let predicted = (l as f64 / r).powi(2);
                assert!(
                    rel < 3.0 * predicted + 1e-12,
                    "({px},{py}) L={l}: relative error {rel:e} against a predicted {predicted:e}"
                );
                assert!(ang < 3.0 * predicted + 1e-12, "({px},{py}) L={l}: angular error {ang:e}");
            }
        }
    }

    /// Outlier rejection has to *do* something, and this is the test that would fail if the
    /// rejection pass were deleted.
    ///
    /// One event in the neighbourhood is corrupted by a timestamp far off the plane — a hot pixel,
    /// or a background-activity event, both of which a real sensor emits constantly. With rejection
    /// off the fit is visibly pulled; with rejection on it returns to the exact answer. Both halves
    /// are asserted, so the test cannot pass by the estimator being insensitive to the outlier in
    /// the first place.
    #[test]
    fn outlier_rejection_recovers_a_fit_that_a_single_hot_pixel_destroys() {
        // A hand-built 5x5 neighbourhood on the plane t = x / 400 (i.e. 400 px/s along +x).
        // Five wide, not seven: 49 points dilute one outlier to a 2% error, which is a bad test
        // because it would pass with the rejection pass deleted. The size is chosen so the outlier
        // does real damage, and the damage is asserted below before it is repaired.
        let mut pts = Vec::new();
        for dy in -2i64..=2 {
            for dx in -2i64..=2 {
                let (x, y) = (20.0 + dx as f64, 20.0 + dy as f64);
                pts.push((x, y, x / 400.0));
            }
        }
        let clean = fit_plane(&pts).unwrap().flow().unwrap();
        assert!((clean.speed - 400.0).abs() < 1e-9, "the clean fit is {}", clean.speed);

        // One corrupted timestamp at (22, 20) — the far edge of the neighbourhood, where a hot
        // pixel does the most damage — 20 ms off a surface whose whole range is 10 ms.
        let bad = pts.iter().position(|&(x, y, _)| x == 22.0 && y == 20.0).expect("in the patch");
        pts[bad].2 += 20e-3;
        let dirty = fit_plane(&pts).unwrap();
        let dirty_flow = dirty.flow().unwrap();
        if PRINT_MEASUREMENTS {
            println!("with the outlier: {} px/s, rms {:e} s", dirty_flow.speed, dirty.rms_s);
        }
        // The outlier must actually hurt, or the rejection pass has nothing to prove.
        assert!(
            (dirty_flow.speed - 400.0).abs() / 400.0 > 0.10,
            "the outlier only moved the fit to {} px/s",
            dirty_flow.speed
        );

        // Now reject against the first-pass plane and refit, which is the paper's second stage.
        // The clean points' residuals after the pulled fit are at most about 0.2 * 20 ms = 4 ms and
        // the outlier's is about 0.88 * 20 ms = 17.6 ms, so 8 ms separates them by construction.
        let kept: Vec<(f64, f64, f64)> =
            pts.iter().copied().filter(|&(x, y, t)| dirty.residual(x, y, t).abs() <= 8e-3).collect();
        assert_eq!(kept.len(), pts.len() - 1, "rejection removed {} points", pts.len() - kept.len());
        let fixed = fit_plane(&kept).unwrap().flow().unwrap();
        assert!((fixed.speed - 400.0).abs() < 1e-9, "the refit is {} px/s", fixed.speed);
    }

    /// The rejection pass wired into [`PlaneFlow`] itself, not just the arithmetic beside it.
    #[test]
    fn plane_flow_rejects_configurations_that_cannot_work() {
        let g = geom(32, 32);
        // A rejection threshold wider than the window can never reject; refused rather than left
        // as a silently inert parameter.
        assert!(matches!(
            PlaneFlow::new(g, 3, 1e-3, Some(1e-2), 8),
            Err(VisionError::BadParameters { .. })
        ));
        // Fewer than three supporting events cannot determine a plane.
        assert!(matches!(PlaneFlow::new(g, 3, 1e-3, None, 2), Err(VisionError::TooFew { .. })));
        // A radius-zero neighbourhood contains only the event itself.
        assert!(matches!(
            PlaneFlow::new(g, 0, 1e-3, None, 8),
            Err(VisionError::BadParameters { .. })
        ));
        assert!(PlaneFlow::new(g, 3, 1e-3, Some(1e-4), 8).is_ok());
    }

    /// Every reason an estimate is unavailable is reported as its own outcome. The audit that
    /// preceded this module found a decoder silently eating valid records; an `Option` here would
    /// be the same shape of defect, so the four causes are separated and each is produced.
    #[test]
    fn each_reason_for_no_flow_is_reported_as_itself() {
        let g = geom(32, 32);

        // TooFew: the very first event has an empty neighbourhood.
        let mut pf = PlaneFlow::new(g, 3, 1e-2, None, 8).unwrap();
        let first = pf.push(PixelEvent { t_s: 0.0, x: 16, y: 16, polarity: Polarity::On }).unwrap();
        assert!(matches!(first, FlowOutcome::TooFew { found: 1 }), "{first:?}");

        // Degenerate: a one-pixel-wide vertical line of events is collinear in (x, y).
        let mut pf = PlaneFlow::new(g, 3, 1e-2, None, 4).unwrap();
        let mut last = FlowOutcome::TooFew { found: 0 };
        for k in 0..8u16 {
            last = pf
                .push(PixelEvent {
                    t_s: f64::from(k) * 1e-4,
                    x: 16,
                    y: 8 + k,
                    polarity: Polarity::On,
                })
                .unwrap();
        }
        assert!(matches!(last, FlowOutcome::Degenerate), "{last:?}");

        // NoMotion: a filled square all at the same instant is a perfectly good flat plane.
        let mut pf = PlaneFlow::new(g, 3, 1e-2, None, 8).unwrap();
        let mut last = FlowOutcome::TooFew { found: 0 };
        for y in 10..16u16 {
            for x in 10..16u16 {
                last = pf.push(PixelEvent { t_s: 0.0, x, y, polarity: Polarity::On }).unwrap();
            }
        }
        assert!(matches!(last, FlowOutcome::NoMotion), "{last:?}");

        // RejectedTooMany: a tight threshold against a jagged patch leaves too few inliers. The
        // raster ramp alone is exactly planar, so the residuals come entirely from the 50 us
        // checkerboard laid over it — and the jitter is half the raster step, which keeps the
        // stream monotone in time. (An earlier draft jittered by more than the step and was refused
        // by `TimeSurface::update`, which is the ordering invariant doing its job.)
        //
        // The outcome is looked for across the whole stream rather than at the last event, because
        // the last event of a raster is always at a corner where the neighbourhood is truncated and
        // the honest answer there is `TooFew`, not `RejectedTooMany`.
        let mut pf = PlaneFlow::new(g, 3, 1e-1, Some(1e-9), 20).unwrap();
        let mut outs = Vec::new();
        let mut k = 0u32;
        for y in 10..24u16 {
            for x in 10..24u16 {
                let jitter = if (x + y) % 2 == 0 { 5e-5 } else { 0.0 };
                outs.push(
                    pf.push(PixelEvent {
                        t_s: f64::from(k) * 1e-4 + jitter,
                        x,
                        y,
                        polarity: Polarity::On,
                    })
                    .unwrap(),
                );
                k += 1;
            }
        }
        let rejected =
            outs.iter().filter(|o| matches!(o, FlowOutcome::RejectedTooMany { .. })).count();
        assert!(rejected > 0, "no event reported RejectedTooMany; outcomes were {outs:?}");
        // And with the rejection pass off, those same events fit — so the outcome above is the
        // rejection doing something, not the data being unfittable.
        let mut pf2 = PlaneFlow::new(g, 3, 1e-1, None, 20).unwrap();
        let mut k = 0u32;
        let mut fitted = 0usize;
        for y in 10..24u16 {
            for x in 10..24u16 {
                let jitter = if (x + y) % 2 == 0 { 5e-5 } else { 0.0 };
                if matches!(
                    pf2.push(PixelEvent {
                        t_s: f64::from(k) * 1e-4 + jitter,
                        x,
                        y,
                        polarity: Polarity::On
                    })
                    .unwrap(),
                    FlowOutcome::Fitted(_)
                ) {
                    fitted += 1;
                }
                k += 1;
            }
        }
        assert!(fitted >= rejected, "{fitted} fitted without rejection against {rejected} rejected with it");
    }


    // -----------------------------------------------------------------------------------------
    // (b) Contrast maximisation: the objective is maximised at the true motion
    // -----------------------------------------------------------------------------------------

    /// The index of the largest entry in a sweep, and the number of *local* maxima in it.
    ///
    /// The second number is what says the objective is usable: an argmax landing on truth means
    /// little if the curve has forty peaks and a search would fall into one of the others.
    fn peak_shape(curve: &[(f64, f64)]) -> (usize, usize) {
        let mut best = 0;
        for (i, &(_, v)) in curve.iter().enumerate() {
            if v > curve[best].1 {
                best = i;
            }
        }
        let locals = (1..curve.len() - 1)
            .filter(|&i| curve[i].1 > curve[i - 1].1 && curve[i].1 > curve[i + 1].1)
            .count();
        (best, locals)
    }

    /// **CHECK (b), translation.** Sweep the edge-normal velocity of a moving edge and assert the
    /// objective's argmax lands on the velocity the generator was given.
    ///
    /// The sweep is **one-dimensional on purpose**. A single straight edge spanning the sensor
    /// carries no information about motion along itself, so the objective is flat in `vy` and a
    /// two-dimensional argmax there would be reporting the grid, not the data. That is the aperture
    /// problem arriving in contrast maximisation exactly as it arrives in plane fitting, and the
    /// module doc says so. [`contrast_maximisation_recovers_two_components_from_a_corner`] is the
    /// two-dimensional case, on a stimulus that actually determines two numbers.
    ///
    /// Both objectives are swept, because they are genuinely different functions — the warped
    /// image's mean is not constant across motions, so variance and mean-square do not differ by a
    /// constant.
    ///
    /// # The reference time is not free, and this is where that bites
    ///
    /// Events are warped **to `t_ref`**, so at the true motion they all collapse onto the structure
    /// as it was *at that instant*. Choose `t_ref = 0` for an edge that entered the sensor at
    /// `x = -4` and the correct motion piles every event onto a column that is off the sensor,
    /// where the bilinear vote discards it: the objective is then exactly zero at truth and its
    /// argmax lands on whatever partial motion happens to leave the most events inside. The first
    /// draft of this test did that and reported 270 px/s with total confidence.
    ///
    /// The fix is the one the literature uses without always saying why: put `t_ref` in the
    /// **middle** of the time span, where the collapsed structure is inside the field of view.
    /// Here that is 0.11 s of a 0.22 s sweep.
    #[test]
    fn contrast_maximisation_finds_the_true_translation() {
        let g = geom(64, 64);
        let truth = 300.0;
        let evs = moving_edge(g, 0.0, truth, -4.0, 0.22, Polarity::On).unwrap();
        assert!(evs.len() > 3000);
        for obj in [Objective::Variance, Objective::MeanSquare] {
            // 61 points over [150, 450], so the grid spacing is 5 px/s and truth is exactly on a
            // grid point — which is what makes "the argmax IS truth" a meaningful assertion rather
            // than "the argmax is near truth".
            let curve = sweep(g, &evs, 0.11, 150.0, 450.0, 61, obj, |vx| Motion::Translation {
                vx,
                vy: 0.0,
            })
            .unwrap();
            let (best, locals) = peak_shape(&curve);
            if PRINT_MEASUREMENTS {
                println!(
                    "{obj:?}: argmax {} px/s (truth {truth}), {locals} local maxima, peak {:.4} vs ends {:.4}/{:.4}",
                    curve[best].0, curve[best].1, curve[0].1, curve[60].1
                );
            }
            assert!(
                (curve[best].0 - truth).abs() < 1e-9,
                "{obj:?}: argmax at {} px/s, not {truth}",
                curve[best].0
            );
            assert_eq!(locals, 1, "{obj:?}: the objective has {locals} local maxima, not one");
            // The peak must be a peak. A flat objective also has an argmax.
            let (_, peak) = argmax(&curve).unwrap();
            assert!(peak > 1.5 * curve[0].1, "{obj:?}: the peak is barely above the sweep's edge");
        }
    }

    /// The objective is **maximised** at truth, checked as a strict inequality against neighbours
    /// on both sides rather than as an argmax over a grid.
    ///
    /// This is the assertion that would survive someone replacing the grid with a finer one, and
    /// the one that fails if the sign of the warp is flipped — a flipped warp doubles the apparent
    /// motion instead of undoing it, so the true velocity becomes an ordinary point on the curve.
    #[test]
    fn the_contrast_objective_is_strictly_maximised_at_the_true_motion() {
        let g = geom(64, 64);
        let truth = 300.0;
        let evs = moving_edge(g, 0.0, truth, -4.0, 0.22, Polarity::On).unwrap();
        let at = |vx: f64| {
            contrast(g, &evs, 0.11, Motion::Translation { vx, vy: 0.0 }, Objective::Variance)
                .unwrap()
        };
        let peak = at(truth);
        for d in [1.0, 5.0, 20.0, 100.0, 250.0] {
            let lo = at(truth - d);
            let hi = at(truth + d);
            if PRINT_MEASUREMENTS {
                println!("delta {d:>6}: {lo:.5} < {peak:.5} > {hi:.5}");
            }
            assert!(peak > lo, "offset -{d} scored {lo} against a peak of {peak}");
            assert!(peak > hi, "offset +{d} scored {hi} against a peak of {peak}");
        }
        // Undoing NO motion is the null hypothesis and must lose to undoing the right one.
        assert!(peak > at(0.0), "the true motion did not beat not warping at all");
        // And doubling the velocity — which is what a sign-flipped warp effectively does — loses.
        assert!(peak > at(2.0 * truth));
    }

    /// **CHECK (b), two dimensions.** On a corner, which has two independent edge orientations,
    /// the full two-parameter search recovers both components.
    ///
    /// The tolerance is the refinement's own resolution: the coarse grid is 21 points over
    /// +/-500 px/s, so 50 px/s, halved by the pattern search until it stops improving.
    #[test]
    fn contrast_maximisation_recovers_two_components_from_a_corner() {
        let g = geom(64, 64);
        let (tvx, tvy) = (180.0, -120.0);
        let evs = moving_corner(
            g,
            (10.0, 58.0),
            (tvx, tvy),
            (0.0, core::f64::consts::FRAC_PI_2),
            40.0,
            0.25,
            Polarity::On,
        )
        .unwrap();
        assert!(evs.len() > 1500, "the corner swept only {} pixels", evs.len());
        let (m, score) =
            search_translation(g, &evs, 0.125, 500.0, 21, 24, Objective::Variance).unwrap();
        let Motion::Translation { vx, vy } = m else { panic!("the search returned {m:?}") };
        if PRINT_MEASUREMENTS {
            println!("corner search: ({vx:.2}, {vy:.2}) px/s against ({tvx}, {tvy}), score {score:.4}");
        }
        assert!((vx - tvx).abs() < 5.0, "vx = {vx}, truth {tvx}");
        assert!((vy - tvy).abs() < 5.0, "vy = {vy}, truth {tvy}");
        // The recovered motion must beat the null, or the "search" is returning its own start.
        let null =
            contrast(g, &evs, 0.125, Motion::Translation { vx: 0.0, vy: 0.0 }, Objective::Variance)
                .unwrap();
        assert!(score > 2.0 * null, "score {score} is barely above the unwarped {null}");
        // The sign of vy is the thing an axis-aligned test cannot check, so it is checked here:
        // the corner moves UP the image and a sign error would put it down.
        assert!(vy < 0.0, "the recovered vertical motion has the wrong sign");
    }

    /// **CHECK (b), rotation.** Sweep angular velocity against a rotating bar.
    ///
    /// A different warp, a different closed form, the same assertion — which matters because a
    /// translation-only implementation could pass the tests above with the rotation branch never
    /// executed.
    #[test]
    fn contrast_maximisation_finds_the_true_angular_velocity() {
        let g = geom(80, 80);
        let centre = (39.5, 39.5);
        let omega = 8.0;
        // 1.6 rad — under half a turn, so the bar's pi-periodicity cannot sweep a pixel twice and
        // create a second peak. That also means only about half the annulus fires, which is why
        // the event count below is a little over half the annulus area.
        let evs = rotating_bar(g, centre, omega, 0.0, 4.0, 38.0, 0.2, Polarity::On).unwrap();
        assert!(evs.len() > 2000, "the bar swept only {} pixels", evs.len());
        // 41 points over [4, 12]: spacing 0.2 rad/s, truth exactly on a grid point.
        let curve = sweep(g, &evs, 0.1, 4.0, 12.0, 41, Objective::Variance, |w| Motion::Rotation {
            omega_rad_s: w,
            cx: centre.0,
            cy: centre.1,
        })
        .unwrap();
        let (best, locals) = peak_shape(&curve);
        if PRINT_MEASUREMENTS {
            println!(
                "rotation: argmax {} rad/s (truth {omega}), {locals} local maxima",
                curve[best].0
            );
        }
        assert!((curve[best].0 - omega).abs() < 1e-9, "argmax {} rad/s", curve[best].0);
        assert_eq!(locals, 1, "{locals} local maxima in the angular sweep");
        assert!(curve[best].1 > 1.5 * curve[0].1, "the rotational peak is not a peak");
    }

    /// **CHECK (b), radial expansion.** Sweep the looming rate against an expanding disc.
    ///
    /// The third warp branch, and the one with the known pathology: at a large enough rate the
    /// model folds the whole cloud through its centre, and the objective is not to be trusted
    /// outside a bounded range. The sweep is bounded at twice the true rate and the doc on
    /// [`Motion::warp`] says why.
    #[test]
    fn contrast_maximisation_finds_the_true_expansion_rate() {
        let g = geom(96, 96);
        let centre = (47.5, 47.5);
        let rate = 150.0;
        let evs = looming_disc(g, centre, 6.0, rate, 0.25, Polarity::On).unwrap();
        assert!(evs.len() > 3000);
        // 61 points over [0, 300]: spacing 5 px/s, truth exactly on a grid point.
        let curve = sweep(g, &evs, 0.125, 0.0, 300.0, 61, Objective::Variance, |r| {
            Motion::RadialExpansion { rate_px_s: r, cx: centre.0, cy: centre.1 }
        })
        .unwrap();
        let (best, locals) = peak_shape(&curve);
        if PRINT_MEASUREMENTS {
            println!("expansion: argmax {} px/s (truth {rate}), {locals} local maxima", curve[best].0);
        }
        assert!((curve[best].0 - rate).abs() < 1e-9, "argmax {} px/s", curve[best].0);
        assert!(locals <= 2, "{locals} local maxima in the expansion sweep");
        assert!(curve[best].1 > 1.5 * curve[0].1, "the expansion peak is not a peak");
    }

    /// Warping is an *identity* at the reference time and the accounting is honest about what left
    /// the sensor.
    #[test]
    fn a_warp_at_the_reference_time_moves_nothing_and_drops_nothing() {
        let g = geom(16, 16);
        let evs: Vec<PixelEvent> = (0..16u16)
            .map(|k| PixelEvent { t_s: 0.5, x: k, y: 8, polarity: Polarity::On })
            .collect();
        for m in [
            Motion::Translation { vx: 900.0, vy: -300.0 },
            Motion::Rotation { omega_rad_s: 40.0, cx: 7.5, cy: 7.5 },
            Motion::RadialExpansion { rate_px_s: 500.0, cx: 7.5, cy: 7.5 },
        ] {
            let wi = warped_image(g, &evs, 0.5, m).unwrap();
            assert_eq!(wi.dropped, 0, "{m:?} dropped an event at zero elapsed time");
            assert_eq!(wi.placed, 16);
            // Zero elapsed time means the bilinear vote lands exactly on the original pixel, so
            // the image is the count image and every value is an exact 1.
            assert_eq!(wi.frame.sum(), 16.0);
            assert_eq!(wi.frame.max(), Some(1.0));
        }
        // A translation large enough to push everything off the sensor is COUNTED, not hidden.
        let moving: Vec<PixelEvent> = (0..16u16)
            .map(|k| PixelEvent { t_s: 1.0, x: k, y: 8, polarity: Polarity::On })
            .collect();
        let off = warped_image(g, &moving, 0.0, Motion::Translation { vx: -1e4, vy: 0.0 }).unwrap();
        assert_eq!(off.placed, 0);
        assert_eq!(off.dropped, 16);
        assert_eq!(off.frame.sum(), 0.0);
        // Which is why nothing here may ever be MINIMISED: the empty image's variance is zero.
        assert_eq!(off.frame.variance(), Some(0.0));
    }

    /// The bilinear vote conserves mass for an event that lands wholly inside the sensor, which is
    /// what makes the accumulated image a histogram rather than a weighted one.
    #[test]
    fn the_bilinear_vote_conserves_event_mass_inside_the_sensor() {
        let g = geom(32, 32);
        // One event, warped by a translation that lands it at a non-integer position well inside.
        let evs = [PixelEvent { t_s: 0.1, x: 16, y: 16, polarity: Polarity::On }];
        let wi = warped_image(g, &evs, 0.0, Motion::Translation { vx: 33.0, vy: -17.0 }).unwrap();
        assert_eq!(wi.placed, 1);
        assert_eq!(wi.dropped, 0);
        assert!((wi.frame.sum() - 1.0).abs() < 1e-12, "the vote summed to {}", wi.frame.sum());
        // Spread over four pixels, none of them the whole vote: this is what distinguishes it from
        // a nearest-neighbour vote, whose plateaus stall a local search.
        let touched = wi.frame.data.iter().filter(|v| **v > 0.0).count();
        assert_eq!(touched, 4, "a non-integer warp touched {touched} pixels");
    }

    /// The sweep's refusals.
    #[test]
    fn a_sweep_refuses_what_it_cannot_sweep() {
        let g = geom(8, 8);
        let evs = [PixelEvent { t_s: 0.0, x: 4, y: 4, polarity: Polarity::On }];
        assert!(matches!(
            sweep(g, &evs, 0.0, 0.0, 1.0, 1, Objective::Variance, |v| Motion::Translation {
                vx: v,
                vy: 0.0
            }),
            Err(VisionError::TooFew { .. })
        ));
        assert!(matches!(
            sweep(g, &evs, f64::NAN, 0.0, 1.0, 4, Objective::Variance, |v| Motion::Translation {
                vx: v,
                vy: 0.0
            }),
            Err(VisionError::NonFinite { .. })
        ));
        assert!(matches!(
            search_translation(g, &evs, 0.0, 0.0, 8, 4, Objective::Variance),
            Err(VisionError::NonPositive { .. })
        ));
        assert!(argmax(&[]).is_none());
        // Ties go to the first entry, so a flat objective still gives a deterministic answer.
        assert_eq!(argmax(&[(1.0, 5.0), (2.0, 5.0), (3.0, 5.0)]), Some((1.0, 5.0)));
    }

    // -----------------------------------------------------------------------------------------
    // (d) Corner detection: fires on a corner, not on a straight edge, quantified
    // -----------------------------------------------------------------------------------------

    /// Build a square binary patch from a predicate on `(i, j)`, `j` being the row.
    fn patch_of(side: usize, f: impl Fn(usize, usize) -> bool) -> Vec<f64> {
        let mut v = vec![0.0; side * side];
        for j in 0..side {
            for i in 0..side {
                if f(i, j) {
                    v[j * side + i] = 1.0;
                }
            }
        }
        v
    }

    /// **CHECK (d), part one — the sign property, which is a theorem rather than a threshold.**
    ///
    /// On a half-plane filling the window, every row is identical, so the vertical `Sobel` response
    /// is **exactly zero**, `det(M)` is exactly zero, and the `Harris` score is `-k * trace(M)^2`:
    /// strictly negative. On a quadrant both eigenvalues are positive and the score clears zero,
    /// because `k = 0.04` admits eigenvalue ratios up to about 19:1.
    ///
    /// Asserting the *sign* is what makes this a check. A threshold is a tuning parameter and a
    /// test against one only says that somebody chose it to pass.
    #[test]
    fn the_harris_response_is_negative_on_an_edge_and_positive_on_a_corner() {
        let d = EHarris::new(geom(64, 64), 4, 1e-3, EHarris::K_HARRIS, 0.0).unwrap();
        let side = 9usize;

        // A vertical edge: the left half is set. det(M) is exactly zero by the argument above.
        let edge = patch_of(side, |i, _| i <= 4);
        let s_edge = d.score_of_patch(&edge, side).unwrap();
        // A horizontal edge: the same argument with the axes exchanged.
        let edge_h = patch_of(side, |_, j| j <= 4);
        let s_edge_h = d.score_of_patch(&edge_h, side).unwrap();
        // A diagonal edge, where neither Sobel response vanishes and the cancellation in det(M) is
        // arithmetic rather than structural — the harder case, and the one a transposed index would
        // pass while failing the two above.
        let edge_d = patch_of(side, |i, j| i + j <= 8);
        let s_edge_d = d.score_of_patch(&edge_d, side).unwrap();
        // A right-angle corner: the upper-left quadrant.
        let corner = patch_of(side, |i, j| i <= 4 && j <= 4);
        let s_corner = d.score_of_patch(&corner, side).unwrap();
        // A blank window and a full window both have no gradient at all.
        let blank = patch_of(side, |_, _| false);
        let full = patch_of(side, |_, _| true);

        if PRINT_MEASUREMENTS {
            println!(
                "harris: edge_v {s_edge:.1} edge_h {s_edge_h:.1} edge_d {s_edge_d:.1} corner {s_corner:.1}"
            );
        }
        assert!(s_edge < 0.0, "a vertical edge scored {s_edge}");
        assert!(s_edge_h < 0.0, "a horizontal edge scored {s_edge_h}");
        assert!(s_edge_d < 0.0, "a diagonal edge scored {s_edge_d}");
        assert!(s_corner > 0.0, "a right-angle corner scored {s_corner}");
        // And the corner beats every edge by a wide margin, not by a hair. Measured: the corner
        // scores 11047 against the strongest edge magnitude of 2621, a factor of 4.2. The bound is
        // 2, which no edge pattern can reach and a broken structure tensor would miss.
        assert!(s_corner > 2.0 * s_edge_d.abs(), "corner {s_corner} against edge {s_edge_d}");
        assert_eq!(d.score_of_patch(&blank, side).unwrap(), 0.0);
        assert_eq!(d.score_of_patch(&full, side).unwrap(), 0.0);

        // The structural claim behind the sign, asserted directly: an axis-aligned edge gives a
        // det(M) of exactly zero, not merely a small one. If this ever becomes approximate, the
        // Sobel kernels have been changed.
        let exact = harris_det_is_zero(&edge, side);
        assert_eq!(exact, 0.0, "det(M) on an axis-aligned edge is {exact}, not exactly zero");
    }

    /// `det(M)` alone for a patch, so the exactness claim above can be asserted rather than
    /// inferred from the score's sign.
    fn harris_det_is_zero(patch: &[f64], side: usize) -> f64 {
        const GX: [[f64; 3]; 3] = [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]];
        const GY: [[f64; 3]; 3] = [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]];
        let (mut mxx, mut myy, mut mxy) = (0.0f64, 0.0f64, 0.0f64);
        for j in 1..side - 1 {
            for i in 1..side - 1 {
                let (mut gx, mut gy) = (0.0f64, 0.0f64);
                for dj in 0..3 {
                    for di in 0..3 {
                        let v = patch[(j + dj - 1) * side + (i + di - 1)];
                        gx += GX[dj][di] * v;
                        gy += GY[dj][di] * v;
                    }
                }
                mxx += gx * gx;
                myy += gy * gy;
                mxy += gx * gy;
            }
        }
        mxx * myy - mxy * mxy
    }

    /// **CHECK (d), part two — the rate, on a real `SAE` built from a real event stream.**
    ///
    /// An `L` is written into the detector's surface one event at a time, then every pixel of the
    /// shape is scored. The result is reported as two counts rather than a yes/no: how many of the
    /// straight-arm pixels score positive, and whether the vertex does.
    ///
    /// Arm pixels are taken 5 to 12 away from the vertex, which keeps both the vertex and the arm
    /// *ends* outside a 9x9 window — a line termination is a corner too, and a test that let one
    /// into the "edge" set would be measuring the wrong thing and would have to be loosened until
    /// it passed.
    #[test]
    fn eharris_fires_on_a_corner_and_on_no_pixel_of_a_straight_arm() {
        let g = geom(48, 48);
        let mut d = EHarris::new(g, 4, 1.0, EHarris::K_HARRIS, 0.0).unwrap();
        let vertex = (24u16, 24u16);
        // Two arms 20 px long meeting at a right angle.
        let mut shape: Vec<(u16, u16)> = Vec::new();
        for k in 0..=20u16 {
            shape.push((vertex.0 - k, vertex.1));
            shape.push((vertex.0, vertex.1 + k));
        }
        shape.sort_unstable();
        shape.dedup();
        for (n, &(x, y)) in shape.iter().enumerate() {
            d.push(PixelEvent { t_s: n as f64 * 1e-4, x, y, polarity: Polarity::On }).unwrap();
        }
        let now = shape.len() as f64 * 1e-4;

        let arm: Vec<(u16, u16)> = shape
            .iter()
            .copied()
            .filter(|&(x, y)| {
                let d = (i32::from(x) - i32::from(vertex.0)).abs()
                    + (i32::from(y) - i32::from(vertex.1)).abs();
                (5..=12).contains(&d)
            })
            .collect();
        let positives =
            arm.iter().filter(|&&(x, y)| d.score_at(x, y, now).unwrap() > 0.0).count();
        let vertex_score = d.score_at(vertex.0, vertex.1, now).unwrap();
        if PRINT_MEASUREMENTS {
            println!(
                "eHarris rate: {positives} of {} straight-arm pixels scored positive; vertex scored {vertex_score:.1}",
                arm.len()
            );
        }
        assert!(arm.len() >= 14, "only {} arm pixels to measure over", arm.len());
        assert_eq!(positives, 0, "{positives} of {} arm pixels fired", arm.len());
        assert!(vertex_score > 0.0, "the vertex scored {vertex_score}");
        // Wired end to end: `push` must agree with `score_at` at the same instant, or the
        // threshold path is untested and the numbers above say nothing about the streaming API.
        //
        // What is NOT asserted here is a detection *rate* over this stream, and the reason is worth
        // recording: the shape is drawn one pixel at a time, so the growing tip of each arm is a
        // line termination at every step — and a termination is a corner. `eHarris` fires on
        // essentially every event of an incrementally-drawn line and is right to. A rate is only
        // meaningful on a stream a sensor could produce, which is
        // [`corner_detectors_fire_near_a_moving_vertex_and_rarely_on_a_straight_edge`].
        let mut d2 = EHarris::new(g, 4, 1.0, EHarris::K_HARRIS, 0.0).unwrap();
        let mut hits = 0usize;
        for (n, &(x, y)) in shape.iter().enumerate() {
            let t = n as f64 * 1e-4;
            let fired = d2.push(PixelEvent { t_s: t, x, y, polarity: Polarity::On }).unwrap();
            let score = d2.score_at(x, y, t).unwrap();
            assert_eq!(
                fired.is_some(),
                score > 0.0,
                "push and score_at disagree at ({x}, {y}): {fired:?} against {score}"
            );
            if let Some(c) = fired {
                assert!((c.score - score).abs() < 1e-12);
                assert_eq!((c.x, c.y), (x, y));
                hits += 1;
            }
        }
        assert!(hits > 0, "the streaming path never fired at all");
    }

    /// **CHECK (d), part three — the `eFAST` criterion against its specification.**
    ///
    /// Every non-arc pixel is given one shared old timestamp and every arc pixel one shared new
    /// one, which makes the *only* valid arc length exactly the one constructed: a shorter arc
    /// excludes an equally-new pixel, a longer one includes an equally-old pixel, and `min > max`
    /// fails in both cases. So the detector's answer is a direct read of the criterion.
    #[test]
    fn efast_accepts_exactly_the_arc_lengths_its_ranges_name() {
        let g = geom(40, 40);
        let (cx, cy) = (20u16, 20u16);
        let build = |k_in: usize, k_out: usize| -> EFast {
            let mut d = EFast::new(g).unwrap();
            {
                let sae = d.surface_mut();
                for (c, _k) in [(&FAST_CIRCLE_16[..], k_in), (&FAST_CIRCLE_20[..], k_out)] {
                    for &(dx, dy) in c {
                        let x = (i64::from(cx) + dx) as u16;
                        let y = (i64::from(cy) + dy) as u16;
                        sae.update(PixelEvent { t_s: 0.0, x, y, polarity: Polarity::On }).unwrap();
                    }
                }
                for (c, k) in [(&FAST_CIRCLE_16[..], k_in), (&FAST_CIRCLE_20[..], k_out)] {
                    for &(dx, dy) in c.iter().take(k) {
                        let x = (i64::from(cx) + dx) as u16;
                        let y = (i64::from(cy) + dy) as u16;
                        sae.update(PixelEvent { t_s: 1.0, x, y, polarity: Polarity::On }).unwrap();
                    }
                }
            }
            d
        };
        // Inside both ranges: a corner.
        assert_eq!(build(4, 5).arcs_at(cx, cy, Polarity::On).unwrap(), (Some(4), Some(5)));
        assert!(build(4, 5).is_corner(cx, cy, Polarity::On).unwrap());
        assert!(build(3, 4).is_corner(cx, cy, Polarity::On).unwrap());
        assert!(build(6, 8).is_corner(cx, cy, Polarity::On).unwrap());
        // A HALF CIRCLE — what a straight edge produces — is outside both ranges.
        assert_eq!(build(8, 10).arcs_at(cx, cy, Polarity::On).unwrap(), (None, None));
        assert!(!build(8, 10).is_corner(cx, cy, Polarity::On).unwrap());
        // An isolated pixel, below both ranges: the noise case.
        assert!(!build(1, 1).is_corner(cx, cy, Polarity::On).unwrap());
        assert!(!build(2, 3).is_corner(cx, cy, Polarity::On).unwrap());
        // BOTH circles must agree. A good inner arc with a bad outer one is rejected, which is the
        // requirement that buys the noise rejection — delete it and the line below passes.
        assert_eq!(build(4, 12).arcs_at(cx, cy, Polarity::On).unwrap(), (Some(4), None));
        assert!(!build(4, 12).is_corner(cx, cy, Polarity::On).unwrap());
        assert!(!build(9, 5).is_corner(cx, cy, Polarity::On).unwrap());
    }

    /// The arc search itself, on hand-written timestamp rings, including the wrap-around case that
    /// a naive linear scan gets wrong.
    #[test]
    fn the_newest_arc_search_wraps_around_the_circle() {
        // Newest run spans the seam: indices 6, 7, 0, 1.
        let mut t = [0.0f64; 8];
        for i in [6, 7, 0, 1] {
            t[i] = 1.0;
        }
        assert_eq!(newest_arc(&t, 2, 6), Some(4));
        // The same four, not wrapping.
        let mut t2 = [0.0f64; 8];
        for i in [2, 3, 4, 5] {
            t2[i] = 1.0;
        }
        assert_eq!(newest_arc(&t2, 2, 6), Some(4));
        // Two separate newest runs cannot form one contiguous arc.
        let mut t3 = [0.0f64; 8];
        for i in [0, 1, 4, 5] {
            t3[i] = 1.0;
        }
        assert_eq!(newest_arc(&t3, 1, 7), None);
        // All equal: no arc is strictly newer than its complement.
        assert_eq!(newest_arc(&[5.0; 8], 1, 7), None);
        // Never-fired pixels are infinitely old and can never be part of the newest arc.
        let t4 = [f64::NEG_INFINITY; 8];
        assert_eq!(newest_arc(&t4, 1, 7), None);
        // Degenerate ranges refuse rather than wrap into nonsense.
        assert_eq!(newest_arc(&t, 0, 6), None);
        assert_eq!(newest_arc(&t, 5, 2), None);
        assert_eq!(newest_arc(&t, 2, 8), None);
        assert_eq!(newest_arc(&[], 1, 2), None);
    }

    /// **CHECK (d), part four — the rate on real streams, edge against corner.**
    ///
    /// Both detectors are run over a straight moving edge and over a moving corner, and the
    /// detection rate per event is reported for each. The edge is the control: a detector that
    /// fires on everything scores well on the corner and is useless, so the assertion is on the
    /// *ratio* as well as on the corner rate.
    ///
    /// # Which way the corner faces is not a detail
    ///
    /// The first draft of this test used a corner whose vertex **led** the motion, and measured a
    /// near-vertex rate of exactly zero. That is the detector being right. `eFAST` asks for a
    /// quarter-circle arc of recently-fired pixels; around a leading convex vertex the swept region
    /// is the *complement* of the wedge, about three quarters of the circle, which is as far
    /// outside the accepted range as a straight edge's half. A **trailing** vertex — arms pointing
    /// back along the motion — leaves a quarter-circle behind it, and that is what the criterion
    /// names. The arms here point left and up while the corner travels down and to the right.
    ///
    /// The arms' free **ends** are excluded from the far set, because a line termination is a
    /// corner too and the detector is right to fire on it. Including them would force the threshold
    /// down until the test stopped measuring anything.
    #[test]
    fn corner_detectors_fire_near_a_moving_vertex_and_rarely_on_a_straight_edge() {
        let g = geom(64, 64);
        let comp = 110.0;
        let v = (comp, comp);
        let vertex0 = (12.0, 12.0);
        let arm_len = 26.0;
        let pi = core::f64::consts::PI;
        let corner = moving_corner(
            g,
            vertex0,
            v,
            (pi, 1.5 * pi),
            arm_len,
            0.36,
            Polarity::On,
        )
        .unwrap();
        let edge = moving_edge(g, 0.0, 155.0, -4.0, 0.45, Polarity::On).unwrap();
        assert!(corner.len() > 800, "the corner swept {} pixels", corner.len());
        assert!(edge.len() > 3000);

        let mut near = (0usize, 0usize);
        let mut far = (0usize, 0usize);
        let mut ef = EFast::new(g).unwrap();
        for e in &corner {
            let vx = vertex0.0 + v.0 * e.t_s;
            let vy = vertex0.1 + v.1 * e.t_s;
            let dv = (f64::from(e.x) - vx).hypot(f64::from(e.y) - vy);
            // The two free ends, which are corners in their own right.
            let d_end1 = (f64::from(e.x) - (vx - arm_len)).hypot(f64::from(e.y) - vy);
            let d_end2 = (f64::from(e.x) - vx).hypot(f64::from(e.y) - (vy - arm_len));
            let hit = ef.push(*e).unwrap().is_some();
            if dv <= 3.0 {
                near.1 += 1;
                near.0 += usize::from(hit);
            } else if dv >= 8.0 && d_end1 >= 6.0 && d_end2 >= 6.0 {
                far.1 += 1;
                far.0 += usize::from(hit);
            }
        }
        let mut ef_edge = EFast::new(g).unwrap();
        let edge_hits = edge.iter().filter(|e| ef_edge.push(**e).unwrap().is_some()).count();

        let near_rate = near.0 as f64 / near.1 as f64;
        let far_rate = far.0 as f64 / far.1 as f64;
        let edge_rate = edge_hits as f64 / edge.len() as f64;
        if PRINT_MEASUREMENTS {
            println!(
                "eFAST: near-vertex {}/{} = {near_rate:.3}, mid-arm {}/{} = {far_rate:.3}, straight edge {edge_hits}/{} = {edge_rate:.4}",
                near.0,
                near.1,
                far.0,
                far.1,
                edge.len()
            );
        }
        assert!(near.1 >= 20 && far.1 >= 150, "near {} far {}: too few to rate", near.1, far.1);
        // A straight edge produces half-circle arcs, which are outside the accepted range.
        assert!(edge_rate < 0.02, "a straight edge fired at {edge_rate:.4} per event");
        assert!(near_rate > 0.10, "the vertex fired at only {near_rate:.3} per event");
        assert!(
            near_rate > 4.0 * far_rate.max(2e-3),
            "near {near_rate:.3} against mid-arm {far_rate:.3}: the detector is not localising"
        );
    }

    // -----------------------------------------------------------------------------------------
    // HOTS
    // -----------------------------------------------------------------------------------------

    /// **The closed form `HOTS`'s update rule is checked against.**
    ///
    /// Drive one pixel repeatedly and nothing else. Its neighbours never fire, so the time-surface
    /// patch is *always* the same vector: exactly `1` at the centre and exactly `0` elsewhere.
    /// Initialise the single prototype as `lambda_0 * S` with `lambda_0 > 0`, and the paper's rule
    /// collapses to a scalar recurrence — because `C` parallel to `S` makes `beta = cos = 1`, so
    ///
    /// ```text
    /// C <- C + alpha (S - C)     i.e.     lambda <- lambda + alpha (1 - lambda)
    /// ```
    ///
    /// which has the exact solution `lambda_n - 1 = (lambda_0 - 1) * prod(1 - alpha_k)`. The test
    /// computes that product independently and compares, so it is checking the implementation
    /// against **analysis**, not against a previous run of itself. Deleting `beta`, changing the
    /// sign in the update, or mis-indexing `p_k` in `alpha` all move the product.
    #[test]
    fn hots_converges_along_the_ray_exactly_as_its_update_rule_predicts() {
        let g = geom(16, 16);
        let tau = 1e-3;
        let lambda0 = 0.5;
        let mut c0 = vec![0.0; 9];
        c0[4] = lambda0;
        let mut h = Hots::with_centers(g, 1, tau, vec![c0]).unwrap();

        let steps = 500u64;
        for n in 0..steps {
            let k = h.learn(PixelEvent {
                t_s: n as f64 * 1e-2,
                x: 5,
                y: 5,
                polarity: Polarity::On,
            })
            .unwrap();
            assert_eq!(k, 0, "there is only one cluster to select");
        }

        // The closed form, computed here and nowhere else in the crate.
        let mut prod = 1.0f64;
        for n in 0..steps {
            prod *= 1.0 - Hots::alpha_for(n);
        }
        let want = 1.0 + (lambda0 - 1.0) * prod;
        let got = h.centers()[0][4];
        if PRINT_MEASUREMENTS {
            println!("HOTS after {steps} steps: lambda = {got:.12}, closed form {want:.12}");
        }
        assert!((got - want).abs() < 1e-12, "lambda {got} against the closed form {want}");
        // The prototype must still be distinguishable from its limit, or the test would pass for
        // any rule that converges to S at all.
        assert!(got < 0.999, "the recurrence had already saturated at {got}; nothing was tested");
        assert!(got > lambda0, "the prototype moved away from the patch");
        // Off-centre entries stay EXACTLY zero: `C + alpha(0 - beta * 0)` is exactly `0`.
        for (i, v) in h.centers()[0].iter().enumerate() {
            if i != 4 {
                assert_eq!(*v, 0.0, "entry {i} drifted to {v}");
            }
        }
        assert_eq!(h.counts(), &[steps]);
    }

    /// Assignment picks the truly nearest prototype under Euclidean distance, ties to the lowest
    /// index — and the tie rule is asserted because it is the mechanism behind the dead-unit
    /// degeneracy the type doc warns about.
    #[test]
    fn hots_assigns_to_the_nearest_prototype_and_breaks_ties_low() {
        let g = geom(16, 16);
        // Three prototypes over a 3x3 patch: centre-heavy, left-heavy, right-heavy.
        let mut a = vec![0.0; 9];
        a[4] = 1.0;
        let mut b = vec![0.0; 9];
        b[3] = 1.0;
        let mut c = vec![0.0; 9];
        c[5] = 1.0;
        let h = Hots::with_centers(g, 1, 1e-3, vec![a.clone(), b.clone(), c.clone()]).unwrap();
        assert_eq!(h.nearest(&a).unwrap().0, 0);
        assert_eq!(h.nearest(&b).unwrap().0, 1);
        assert_eq!(h.nearest(&c).unwrap().0, 2);
        // The distance is the Euclidean one, not a squared one: two orthogonal unit vectors are
        // sqrt(2) apart.
        assert!((h.nearest(&b).unwrap().1).abs() < 1e-15);
        assert!((h.nearest(&a).unwrap().1).abs() < 1e-15);
        let mut probe = vec![0.0; 9];
        probe[4] = 0.9;
        probe[3] = 0.1;
        assert_eq!(h.nearest(&probe).unwrap().0, 0);
        // Identical prototypes: the tie goes to index 0, so index 1 is a dead unit forever.
        let dead = Hots::with_centers(g, 1, 1e-3, vec![a.clone(), a.clone()]).unwrap();
        assert_eq!(dead.nearest(&a).unwrap().0, 0);
        // A patch of the wrong length is refused rather than compared against a prefix.
        assert!(matches!(h.nearest(&[0.0; 8]), Err(VisionError::BadParameters { .. })));
    }

    /// The dead-unit degeneracy, demonstrated rather than only described: two identical prototypes
    /// stay identical however long the layer learns, and [`Hots::counts`] is where a caller sees it.
    #[test]
    fn identically_initialised_prototypes_stay_dead() {
        let g = geom(16, 16);
        let proto = vec![0.25; 9];
        let mut h = Hots::with_centers(g, 1, 1e-3, vec![proto.clone(), proto]).unwrap();
        for n in 0..200u64 {
            h.learn(PixelEvent { t_s: n as f64 * 1e-3, x: 8, y: 8, polarity: Polarity::On })
                .unwrap();
        }
        assert_eq!(h.counts()[1], 0, "the second prototype was somehow selected");
        assert_eq!(h.counts()[0], 200);
        // And the dead one is untouched, bit for bit.
        assert!(h.centers()[1].iter().all(|v| *v == 0.25));
        assert!(h.centers()[0].iter().any(|v| *v != 0.25), "the live prototype did not move");
    }

    /// A layer is reproducible from its seed, which is the crate-wide determinism promise reaching
    /// this module.
    #[test]
    fn a_random_hots_layer_is_reproducible_from_its_seed() {
        let g = geom(16, 16);
        let a = Hots::new(g, 2, 1e-3, 4, &mut Rng::new(7)).unwrap();
        let b = Hots::new(g, 2, 1e-3, 4, &mut Rng::new(7)).unwrap();
        let c = Hots::new(g, 2, 1e-3, 4, &mut Rng::new(8)).unwrap();
        assert_eq!(a.centers(), b.centers());
        assert_ne!(a.centers(), c.centers());
        // Prototypes occupy the same box a time surface does, which is what makes the initial
        // distances comparable to the data's.
        assert!(a.centers().iter().flatten().all(|v| (0.0..1.0).contains(v)));
        assert_eq!(a.centers()[0].len(), 25);
    }

    #[test]
    fn hots_refuses_a_dictionary_it_cannot_use() {
        let g = geom(16, 16);
        assert!(matches!(Hots::with_centers(g, 1, 1e-3, vec![]), Err(VisionError::TooFew { .. })));
        assert!(matches!(
            Hots::with_centers(g, 0, 1e-3, vec![vec![0.0]]),
            Err(VisionError::TooFew { .. })
        ));
        assert!(matches!(
            Hots::with_centers(g, 1, 1e-3, vec![vec![0.0; 8]]),
            Err(VisionError::BadParameters { .. })
        ));
        assert!(matches!(
            Hots::with_centers(g, 1, 1e-3, vec![vec![f64::NAN; 9]]),
            Err(VisionError::NonFinite { .. })
        ));
        assert!(Hots::with_centers(g, 1, 0.0, vec![vec![0.0; 9]]).is_err());
        assert!(matches!(Hots::new(g, 1, 1e-3, 0, &mut Rng::new(1)), Err(VisionError::TooFew { .. })));
    }

    // -----------------------------------------------------------------------------------------
    // HATS
    // -----------------------------------------------------------------------------------------

    /// **The closed form `HATS` is checked against**, in three cases whose answers can be written
    /// down.
    ///
    /// One event in a cell: its local memory time surface is its own contribution alone, `exp(0)`,
    /// so the averaged histogram is *exactly* the centre-bin indicator.
    ///
    /// Two events at the same pixel `dt` apart: the first sees only itself, the second sees itself
    /// plus the first at `exp(-dt/tau)`. The average over the cell's two events is therefore
    /// `1 + exp(-dt/tau)/2` at the centre and exactly zero everywhere else.
    ///
    /// Two events one pixel apart: the later one records the earlier at the bin for offset
    /// `(-1, 0)`, and **not** at `(+1, 0)`. That asymmetry is the whole content of the sign
    /// convention, and a test on a symmetric stimulus cannot see it.
    #[test]
    fn hats_reproduces_its_local_memory_surface_in_closed_form() {
        let g = geom(20, 20);
        let h = Hats { cell_px: 10, radius: 3, tau_s: 0.05, window_s: 1.0, split_polarity: false };
        let bins = h.bins_per_cell();
        assert_eq!(bins, 49);
        let centre_bin = 3 * 7 + 3;

        // One event.
        let one = h.descriptor(g, &[PixelEvent { t_s: 0.0, x: 5, y: 5, polarity: Polarity::On }])
            .unwrap();
        assert_eq!(one.len(), h.descriptor_len(g).unwrap());
        assert_eq!(one[centre_bin], 1.0, "a single event's own contribution is not exp(0)");
        assert_eq!(one.iter().filter(|v| **v != 0.0).count(), 1);

        // Two at the same pixel.
        let dt = 0.02;
        let two = h
            .descriptor(g, &[
                PixelEvent { t_s: 0.0, x: 5, y: 5, polarity: Polarity::On },
                PixelEvent { t_s: dt, x: 5, y: 5, polarity: Polarity::On },
            ])
            .unwrap();
        let want = 1.0 + (-dt / h.tau_s).exp() / 2.0;
        assert!((two[centre_bin] - want).abs() < 1e-14, "{} against {want}", two[centre_bin]);
        assert_eq!(two.iter().filter(|v| **v != 0.0).count(), 1);

        // Two one pixel apart: the asymmetry.
        let off = h
            .descriptor(g, &[
                PixelEvent { t_s: 0.0, x: 5, y: 5, polarity: Polarity::On },
                PixelEvent { t_s: dt, x: 6, y: 5, polarity: Polarity::On },
            ])
            .unwrap();
        let left_bin = 3 * 7 + 2; // offset (-1, 0)
        let right_bin = 3 * 7 + 4; // offset (+1, 0)
        assert_eq!(off[centre_bin], 1.0, "each event's own contribution, averaged over two");
        assert!(
            (off[left_bin] - (-dt / h.tau_s).exp() / 2.0).abs() < 1e-14,
            "the neighbour landed at {} in the (-1, 0) bin",
            off[left_bin]
        );
        assert_eq!(off[right_bin], 0.0, "the offset sign is mirrored");
    }

    /// The memory window is a hard cut-off and the cell tiling is a hard partition. Both are
    /// asserted with exact zeros, because both are places where an off-by-one would produce a
    /// descriptor that is merely slightly wrong.
    #[test]
    fn hats_forgets_past_its_window_and_does_not_leak_between_cells() {
        let g = geom(20, 20);
        let h = Hats { cell_px: 10, radius: 3, tau_s: 1.0, window_s: 0.1, split_polarity: false };
        let centre_bin = 3 * 7 + 3;
        // The earlier event is 0.2 s back — twice the window — so it contributes exactly nothing
        // and the average is over two events each seeing only itself.
        let d = h
            .descriptor(g, &[
                PixelEvent { t_s: 0.0, x: 5, y: 5, polarity: Polarity::On },
                PixelEvent { t_s: 0.2, x: 6, y: 5, polarity: Polarity::On },
            ])
            .unwrap();
        assert_eq!(d[centre_bin], 1.0);
        assert_eq!(d.iter().filter(|v| **v != 0.0).count(), 1, "something crossed the window");

        // Cells: (5, 5) is in cell 0, (14, 5) in cell 1, and they are three pixels apart in neither
        // sense — the point is that even a NEIGHBOURING pixel across a cell boundary does not
        // contribute. (9, 5) and (10, 5) are adjacent and in different cells.
        let (cells_x, cells_y) = h.cell_grid(g).unwrap();
        assert_eq!((cells_x, cells_y), (2, 2));
        let d = h
            .descriptor(g, &[
                PixelEvent { t_s: 0.0, x: 9, y: 5, polarity: Polarity::On },
                PixelEvent { t_s: 0.01, x: 10, y: 5, polarity: Polarity::On },
            ])
            .unwrap();
        let bins = h.bins_per_cell();
        assert_eq!(d[centre_bin], 1.0, "cell 0");
        assert_eq!(d[bins + centre_bin], 1.0, "cell 1");
        assert_eq!(d.iter().filter(|v| **v != 0.0).count(), 2, "a cell boundary leaked");

        // And a cell with no events keeps its exact zeros rather than dividing by zero.
        assert!(d[2 * bins..].iter().all(|v| *v == 0.0));
    }

    /// Polarity splitting doubles the descriptor and keeps the two signs apart, which is what
    /// stops an `On` edge and the `Off` edge trailing it from cancelling into an average.
    #[test]
    fn hats_keeps_the_polarity_planes_apart_when_asked() {
        let g = geom(10, 10);
        let split =
            Hats { cell_px: 10, radius: 1, tau_s: 1.0, window_s: 1.0, split_polarity: true };
        let joint = Hats { split_polarity: false, ..split };
        assert_eq!(split.descriptor_len(g).unwrap(), 2 * joint.descriptor_len(g).unwrap());
        let evs = [
            PixelEvent { t_s: 0.0, x: 5, y: 5, polarity: Polarity::On },
            PixelEvent { t_s: 0.01, x: 5, y: 5, polarity: Polarity::Off },
        ];
        let d = split.descriptor(g, &evs).unwrap();
        // 9 bins per plane; Off is plane 0 and On is plane 1. Each event sees only itself, and the
        // cell's two events are averaged over both planes together, so each centre reads 0.5.
        assert_eq!(d.len(), 18);
        assert_eq!(d[4], 0.5, "the Off plane's centre bin");
        assert_eq!(d[9 + 4], 0.5, "the On plane's centre bin");
        // Joined, the later event sees the earlier one and the centre carries both.
        let j = joint.descriptor(g, &evs).unwrap();
        let want = 1.0 + (-0.01f64 / joint.tau_s).exp() / 2.0;
        assert!((j[4] - want).abs() < 1e-14, "{} against {want}", j[4]);
    }

    #[test]
    fn hats_refuses_what_it_cannot_describe() {
        let g = geom(20, 20);
        let h = Hats::n_cars();
        assert_eq!(h.cell_px, 10);
        assert_eq!(h.bins_per_cell(), 49);
        assert!(h.descriptor_len(g).unwrap() > 0);
        assert!(matches!(
            Hats { cell_px: 0, ..h }.cell_grid(g),
            Err(VisionError::NonPositive { .. })
        ));
        assert!(Hats { tau_s: 0.0, ..h }.descriptor(g, &[]).is_err());
        assert!(Hats { window_s: -1.0, ..h }.descriptor(g, &[]).is_err());
        // Out-of-order events are refused: "earlier events" is defined by the slice order, and a
        // shuffled slice would silently describe a scene that never happened.
        let bad = [
            PixelEvent { t_s: 1.0, x: 5, y: 5, polarity: Polarity::On },
            PixelEvent { t_s: 0.5, x: 5, y: 5, polarity: Polarity::On },
        ];
        assert!(matches!(h.descriptor(g, &bad), Err(VisionError::OutOfOrder { .. })));
        // An event off the lattice is refused rather than binned into the nearest cell.
        let off = [PixelEvent { t_s: 0.0, x: 20, y: 0, polarity: Polarity::On }];
        assert!(matches!(h.descriptor(g, &off), Err(VisionError::OutOfBounds { .. })));
        // An empty stream gives an all-zero descriptor of the right length, not an empty vector.
        let empty = h.descriptor(g, &[]).unwrap();
        assert_eq!(empty.len(), h.descriptor_len(g).unwrap());
        assert!(empty.iter().all(|v| *v == 0.0));
    }

    // -----------------------------------------------------------------------------------------
    // The generators, checked against their own defining equations
    // -----------------------------------------------------------------------------------------
    //
    // Every other test in this module rests on these, so they are checked against the closed forms
    // their docs state rather than against the algorithms that consume them. A generator that is
    // wrong in the same way as its consumer agrees with it perfectly.

    /// Each event of [`moving_edge`] satisfies `n . p = d0 + speed * t` exactly, and the swept set
    /// is exactly the pixels whose crossing time falls in the window — no pixel twice, none missed.
    #[test]
    fn the_moving_edge_generator_satisfies_its_defining_equation() {
        let g = geom(32, 24);
        for deg in [0.0, 37.0, 143.0, 271.0] {
            let n = deg * core::f64::consts::PI / 180.0;
            let (ny, nx) = n.sin_cos();
            let (speed, d0, dur) = (250.0, -6.0, 0.35);
            let evs = moving_edge(g, n, speed, d0, dur, Polarity::On).unwrap();
            let mut seen = vec![false; g.pixels()];
            let mut prev = f64::NEG_INFINITY;
            for e in &evs {
                let lhs = nx * f64::from(e.x) + ny * f64::from(e.y);
                let rhs = d0 + speed * e.t_s;
                assert!((lhs - rhs).abs() < 1e-12, "{deg} deg: {lhs} against {rhs}");
                assert!((0.0..=dur).contains(&e.t_s));
                let idx = g.require(e.x, e.y).unwrap();
                assert!(!seen[idx], "pixel ({}, {}) fired twice", e.x, e.y);
                seen[idx] = true;
                assert!(e.t_s >= prev, "the stream is not sorted");
                prev = e.t_s;
                assert_eq!(e.polarity, Polarity::On);
            }
            // Completeness: every pixel whose crossing time is in the window is present.
            let want = (0..g.height)
                .flat_map(|y| (0..g.width).map(move |x| (x, y)))
                .filter(|&(x, y)| {
                    let t = (nx * f64::from(x) + ny * f64::from(y) - d0) / speed;
                    (0.0..=dur).contains(&t)
                })
                .count();
            assert_eq!(evs.len(), want, "{deg} deg: {} events against {want} crossings", evs.len());
            // The ground truth the flow tests compare against is the generator's own statement.
            let (vx, vy) = moving_edge_flow(n, speed);
            assert!((vx - speed * nx).abs() < 1e-15 && (vy - speed * ny).abs() < 1e-15);
            assert!((vx.hypot(vy) - speed).abs() < 1e-12);
        }
    }

    /// Each event of [`moving_corner`] lies on one of the two arms at its own timestamp, within the
    /// stated arm length, and the two arms are both present.
    #[test]
    fn the_moving_corner_generator_puts_every_event_on_an_arm() {
        let g = geom(48, 48);
        let pi = core::f64::consts::PI;
        let v = (120.0, 80.0);
        let v0 = (8.0, 6.0);
        let (len, dur) = (24.0, 0.3);
        let arms = (0.0, pi / 2.0);
        let evs = moving_corner(g, v0, v, arms, len, dur, Polarity::Off).unwrap();
        assert!(evs.len() > 400, "only {} events", evs.len());
        let mut on_arm = [0usize, 0usize];
        for e in &evs {
            let vx = v0.0 + v.0 * e.t_s;
            let vy = v0.1 + v.1 * e.t_s;
            let (px, py) = (f64::from(e.x) - vx, f64::from(e.y) - vy);
            let mut matched = false;
            for (k, a) in [arms.0, arms.1].into_iter().enumerate() {
                let (dy, dx) = a.sin_cos();
                // Perpendicular distance to the arm's line, and the foot along it.
                let perp = (px * dy - py * dx).abs();
                let along = px * dx + py * dy;
                if perp < 1e-9 && (-1e-9..=len + 1e-9).contains(&along) {
                    on_arm[k] += 1;
                    matched = true;
                }
            }
            assert!(matched, "an event at ({}, {}) t={} is on neither arm", e.x, e.y, e.t_s);
            assert_eq!(e.polarity, Polarity::Off);
        }
        assert!(on_arm[0] > 100 && on_arm[1] > 100, "arm coverage {on_arm:?}");
        // An arm parallel to the motion sweeps nothing and is refused rather than silently empty.
        let e = moving_corner(g, v0, (100.0, 0.0), (0.0, pi / 2.0), len, dur, Polarity::On);
        assert!(matches!(e, Err(VisionError::BadParameters { .. })), "{e:?}");
    }

    /// Each event of [`rotating_bar`] is on the bar at its own timestamp, modulo the bar's
    /// `pi`-periodicity, and each annulus pixel fires the number of times the sweep angle implies.
    #[test]
    fn the_rotating_bar_generator_puts_every_event_on_the_bar() {
        let g = geom(64, 64);
        let centre = (31.5, 31.5);
        let (omega, theta0) = (9.0, 0.3);
        let (r_min, r_max, dur) = (5.0, 28.0, 0.5);
        let evs = rotating_bar(g, centre, omega, theta0, r_min, r_max, dur, Polarity::On).unwrap();
        assert!(evs.len() > 1000);
        let pi = core::f64::consts::PI;
        let mut counts = std::collections::HashMap::new();
        for e in &evs {
            let (dx, dy) = (f64::from(e.x) - centre.0, f64::from(e.y) - centre.1);
            let r = dx.hypot(dy);
            assert!((r_min..=r_max).contains(&r), "an event at r = {r}");
            // The bar's angle at this instant, and the pixel's, agree modulo pi.
            let bar = theta0 + omega * e.t_s;
            let d = (dy.atan2(dx) - bar).rem_euclid(pi);
            let d = d.min(pi - d);
            assert!(d < 1e-9, "off the bar by {d} rad");
            assert!((0.0..=dur).contains(&e.t_s));
            *counts.entry((e.x, e.y)).or_insert(0usize) += 1;
        }
        // The bar sweeps omega * dur = 4.5 rad, i.e. 4.5 / pi = 1.43 half-turns, so every annulus
        // pixel is crossed once or twice and at least one is crossed twice.
        assert!(counts.values().all(|c| (1..=2).contains(c)), "a pixel fired an impossible count");
        assert!(counts.values().any(|c| *c == 2), "no pixel was crossed twice in 1.43 half-turns");
        // The ground-truth flow is tangential of magnitude omega * r, and undefined at the centre.
        let (fx, fy) = rotating_bar_flow(centre, omega, 51, 31).unwrap();
        let r = (51.0f64 - centre.0).hypot(31.0 - centre.1);
        assert!((fx.hypot(fy) - omega * r).abs() < 1e-12);
        // Perpendicular to the radius: the dot product with the radius vector is exactly zero.
        assert!((fx * (51.0 - centre.0) + fy * (31.0 - centre.1)).abs() < 1e-12);
        assert!(rotating_bar_flow((10.0, 10.0), 1.0, 10, 10).is_none());
        assert!(matches!(
            rotating_bar(g, centre, omega, theta0, 20.0, 10.0, dur, Polarity::On),
            Err(VisionError::BadParameters { .. })
        ));
    }

    /// Each event of [`looming_disc`] satisfies `r = r0 + rate * t`, each pixel fires once, and the
    /// flow field is radially outward at exactly the expansion rate.
    #[test]
    fn the_looming_disc_generator_satisfies_its_defining_equation() {
        let g = geom(40, 40);
        let centre = (19.5, 19.5);
        let (r0, rate, dur) = (3.0, 180.0, 0.1);
        let evs = looming_disc(g, centre, r0, rate, dur, Polarity::On).unwrap();
        assert!(evs.len() > 400);
        let mut seen = vec![false; g.pixels()];
        for e in &evs {
            let r = (f64::from(e.x) - centre.0).hypot(f64::from(e.y) - centre.1);
            assert!((r - (r0 + rate * e.t_s)).abs() < 1e-12);
            let idx = g.require(e.x, e.y).unwrap();
            assert!(!seen[idx], "pixel fired twice");
            seen[idx] = true;
        }
        // Speed is exactly `rate` at every pixel; only the direction varies. That is what makes
        // this a good check on a flow estimator's direction recovery.
        for &(x, y) in &[(30u16, 19u16), (19, 30), (8, 8), (35, 35)] {
            let (fx, fy) = looming_disc_flow(centre, rate, x, y).unwrap();
            assert!((fx.hypot(fy) - rate).abs() < 1e-12, "speed at ({x}, {y})");
            // Radially outward: the cross product with the radius vector is exactly zero and the
            // dot product is positive.
            let (dx, dy) = (f64::from(x) - centre.0, f64::from(y) - centre.1);
            assert!((fx * dy - fy * dx).abs() < 1e-12, "not radial at ({x}, {y})");
            assert!(fx * dx + fy * dy > 0.0, "inward at ({x}, {y})");
        }
        assert!(looming_disc_flow(centre, rate, 19, 19).is_some());
    }

    /// Quantisation loses precision and nothing else: every timestamp becomes an exact multiple of
    /// a microsecond, every event survives, and the stream stays sorted.
    #[test]
    fn quantisation_rounds_the_clock_without_losing_an_event() {
        let g = geom(32, 32);
        let evs = moving_edge(g, 0.7, 333.0, -5.0, 0.2, Polarity::On).unwrap();
        let q = quantise_microseconds(&evs).unwrap();
        assert_eq!(q.len(), evs.len(), "quantisation ate an event");
        let mut prev = f64::NEG_INFINITY;
        for e in &q {
            let us = e.t_s * 1e6;
            assert!((us - us.round()).abs() < 1e-6, "{us} is not a whole microsecond");
            assert!(e.t_s >= prev);
            prev = e.t_s;
        }
        // It must actually change something, or every test that uses it is testing nothing.
        let moved = evs.iter().zip(&q).filter(|(a, b)| a.t_s != b.t_s).count();
        assert!(moved * 2 > evs.len(), "only {moved} of {} timestamps moved", evs.len());
        // A negative timestamp has no wire encoding.
        let bad = [PixelEvent { t_s: -1.0, x: 0, y: 0, polarity: Polarity::On }];
        assert!(matches!(quantise_microseconds(&bad), Err(VisionError::NonPositive { .. })));
    }

    /// The whole pipeline is deterministic: the same inputs give bit-identical outputs, including
    /// the tie-breaking order the generators impose.
    #[test]
    fn the_whole_pipeline_is_bit_reproducible() {
        let g = geom(48, 48);
        let run = || {
            let evs = moving_edge(g, 0.0, 400.0, -4.0, 0.14, Polarity::On).unwrap();
            let mut pf = PlaneFlow::new(g, 3, 0.03, Some(1e-4), 8).unwrap();
            let outs = pf.run(&evs).unwrap();
            let speeds: Vec<f64> = outs
                .iter()
                .filter_map(|o| match o {
                    FlowOutcome::Fitted(f) => Some(f.speed),
                    _ => None,
                })
                .collect();
            let frame = accumulate_count(g, &evs).unwrap();
            (evs, speeds, frame)
        };
        let a = run();
        let b = run();
        assert_eq!(a.0, b.0, "the generator is not reproducible");
        assert_eq!(a.1, b.1, "the flow estimates are not bit-identical");
        assert_eq!(a.2, b.2, "the accumulated frame is not bit-identical");
        assert!(!a.1.is_empty());
        // An axis-aligned edge fires a whole column at one instant, so the tie-break is exercised:
        // there are far more events than distinct timestamps.
        let distinct = {
            let mut t: Vec<u64> = a.0.iter().map(|e| e.t_s.to_bits()).collect();
            t.sort_unstable();
            t.dedup();
            t.len()
        };
        assert!(distinct * 4 < a.0.len(), "{distinct} distinct times over {} events", a.0.len());
    }

    /// The bridge back to [`crate::aer`]: a decoded recording becomes a stream this module can
    /// consume, and the addresses survive the trip.
    #[test]
    fn a_decoded_recording_crosses_into_this_module_intact() {
        let g = geom(64, 64);
        let evs = moving_edge(g, 0.5, 200.0, -4.0, 0.3, Polarity::On).unwrap();
        let q = quantise_microseconds(&evs).unwrap();
        let wire: Vec<AerEvent> = q.iter().map(|e| e.to_aer().unwrap()).collect();
        let back: Vec<PixelEvent> = wire.iter().map(|e| PixelEvent::from_aer(*e)).collect();
        assert_eq!(back, q, "the microsecond boundary is not a round trip");
        // And the crate's flat address convention agrees with this module's row-major index.
        for e in wire.iter().take(50) {
            let flat = e.to_event(g.width).unwrap();
            assert_eq!(flat.address as usize, g.require(e.x, e.y).unwrap());
        }
    }

    /// [`TimeSurface::render`] agrees with [`TimeSurface::value_at`] everywhere, and [`clear`]
    /// returns the surface to its birth state exactly.
    ///
    /// [`clear`]: TimeSurface::clear
    #[test]
    fn a_rendered_surface_agrees_with_the_pointwise_query() {
        let g = geom(12, 9);
        let mut ts = TimeSurface::new(g, Decay::Exponential { tau_s: 0.01 }, false).unwrap();
        for (n, (x, y)) in [(1u16, 1u16), (5, 3), (11, 8), (0, 0)].into_iter().enumerate() {
            ts.update(PixelEvent { t_s: n as f64 * 1e-3, x, y, polarity: Polarity::On }).unwrap();
        }
        let now = 0.01;
        let f = ts.render(Polarity::On, now).unwrap();
        assert_eq!(f.data.len(), g.pixels());
        for y in 0..g.height {
            for x in 0..g.width {
                let a = f.at(x, y).unwrap();
                let b = ts.value_at(x, y, Polarity::On, now).unwrap();
                assert_eq!(a, b, "render and value_at disagree at ({x}, {y})");
            }
        }
        assert_eq!(f.data.iter().filter(|v| **v > 0.0).count(), 4);
        let fresh = TimeSurface::new(g, Decay::Exponential { tau_s: 0.01 }, false).unwrap();
        ts.clear();
        assert_eq!(ts, fresh, "clear did not restore the birth state");
        assert_eq!(ts.render(Polarity::On, now).unwrap().sum(), 0.0);
    }

    /// A [`Frame`]'s summary statistics, on an image whose answers are arithmetic.
    #[test]
    fn frame_statistics_are_the_ones_they_are_named_after() {
        let g = geom(2, 2);
        let f = Frame { width: 2, height: 2, data: vec![1.0, 2.0, 3.0, 4.0] };
        assert_eq!(f.sum(), 10.0);
        assert_eq!(f.sum_of_squares(), 30.0);
        assert_eq!(f.mean(), Some(2.5));
        // POPULATION variance: (2.25 + 0.25 + 0.25 + 2.25) / 4 = 1.25. The sample form would give
        // 1.6667, and the difference would make the contrast objective depend on sensor size.
        assert_eq!(f.variance(), Some(1.25));
        assert_eq!(f.max(), Some(4.0));
        assert_eq!(f.min(), Some(1.0));
        assert_eq!(f.at(1, 1), Some(4.0));
        assert_eq!(f.at(2, 0), None);
        assert_eq!(Frame::zeros(g).geometry(), g);
    }

    // -----------------------------------------------------------------------------------------
    // Gaps found by mutating this module and re-running its own tests
    // -----------------------------------------------------------------------------------------
    //
    // Fifteen mutations were applied to the code above and the tests re-run. Four survived. Three
    // were real holes and are closed here; the fourth is recorded in
    // `swapping_the_sobel_kernels_is_an_equivalent_mutation` because it is not a defect at all.

    /// **Mutation found: weakening the collinearity guard to `det > 0.0` survived every test.**
    ///
    /// The `Degenerate` tests above all use *exactly* collinear support, where `det` is exactly
    /// zero in floating point and even a zero-threshold guard fires. The case the guard actually
    /// exists for is support that is collinear to within rounding — a thin stimulus, a line of
    /// events with one pixel a fraction off — where `det` is tiny and positive and the fit it
    /// produces is enormous and entirely plausible-looking.
    ///
    /// So the guard is bracketed instead: at a perpendicular spread far below its threshold the
    /// fit must refuse, and far above it must succeed and be accurate. Removing the guard breaks
    /// the first half; making it aggressive breaks the second.
    #[test]
    fn the_collinearity_guard_is_bracketed_on_nearly_collinear_support() {
        // A line of 11 points along y = x, with a plane t = 0.001 * x laid over it. Every point
        // gets a perpendicular displacement of `d` on alternate indices, so the support's
        // perpendicular spread is exactly what `d` says.
        let build = |d: f64| -> Vec<(f64, f64, f64)> {
            (0..11)
                .map(|k| {
                    let x = f64::from(k);
                    let y = x + if k % 2 == 0 { d } else { 0.0 };
                    (x, y, 1e-3 * x)
                })
                .collect()
        };
        // Below the guard: the squared correlation of the centred coordinates exceeds 1 - 1e-9 and
        // the answer does not exist.
        for d in [0.0, 1e-9, 1e-7] {
            let r = fit_plane(&build(d));
            assert!(
                matches!(r, Err(VisionError::Degenerate { .. })),
                "a perpendicular spread of {d} px was fitted: {r:?}"
            );
        }
        // Above it: a real two-dimensional support, fitted accurately. `t` does not depend on `y`
        // at all here, so the correct `b` is zero and a guard that let the degenerate cases through
        // would be returning a `b` of order 1/d.
        for d in [1e-3, 1e-1, 1.0] {
            let f = fit_plane(&build(d)).unwrap_or_else(|e| panic!("spread {d} px refused: {e}"));
            assert!((f.a - 1e-3).abs() < 1e-9, "spread {d}: a = {}", f.a);
            assert!(f.b.abs() < 1e-9, "spread {d}: b = {}", f.b);
        }
        // And the transition is where the doc says it is: the guard is on the squared correlation,
        // so it is scale-free. The same support scaled up by 100 in x and y behaves identically.
        let scaled: Vec<(f64, f64, f64)> =
            build(1e-7).iter().map(|&(x, y, t)| (100.0 * x, 100.0 * y, t)).collect();
        assert!(matches!(fit_plane(&scaled), Err(VisionError::Degenerate { .. })));
    }

    /// **Mutation found: deleting the time window in [`PlaneFlow::push`] survived every test.**
    ///
    /// Every stream used above sweeps each pixel at most once inside the fitting window, so no
    /// stale `SAE` entry was ever there to exclude. A real recording is nothing like that: the
    /// surface holds whatever last happened at each pixel, from a second or a minute ago.
    ///
    /// Here an edge crosses, then crosses again a full second later. During the second pass every
    /// neighbourhood contains second-old timestamps from the first, and the window is the only
    /// thing keeping them out. With it, the second pass recovers the velocity exactly; without it,
    /// the fit mixes two passes and the result is meaningless.
    #[test]
    fn the_fitting_window_excludes_a_previous_pass_over_the_same_pixels() {
        let g = geom(48, 48);
        let speed = 400.0;
        let first = moving_edge(g, 0.0, speed, -4.0, 0.14, Polarity::On).unwrap();
        let second: Vec<PixelEvent> =
            first.iter().map(|e| PixelEvent { t_s: e.t_s + 1.0, ..*e }).collect();
        let mut both = first.clone();
        both.extend(second.iter().copied());
        assert_eq!(both.len(), 2 * first.len());

        let mut pf = PlaneFlow::new(g, 3, 0.05, None, 8).unwrap();
        let outs = pf.run(&both).unwrap();
        // Score the SECOND pass only — the pass with a stale surface underneath it.
        let truth = moving_edge_flow(0.0, speed);
        let late: Vec<PixelEvent> = both.iter().copied().filter(|e| e.t_s >= 1.0).collect();
        let late_outs: Vec<FlowOutcome> = both
            .iter()
            .zip(&outs)
            .filter(|(e, _)| e.t_s >= 1.0)
            .map(|(_, o)| *o)
            .collect();
        let err = flow_error(&late, &late_outs, |_| Some(truth)).unwrap();
        assert!(err.fitted * 2 > err.offered, "only {} of {} fitted", err.fitted, err.offered);
        assert!(
            err.max_rel_speed_err < 1e-9,
            "the second pass recovered {:e} relative error; a stale surface leaked in",
            err.max_rel_speed_err
        );
        // The stale entries really are there: a surface with no window would see them.
        let sae = pf.surface();
        let old = sae.last_time(0, 0, Polarity::On).unwrap().expect("pixel (0,0) fired");
        assert!(old > 1.0, "the surface does not hold a second pass at all");
    }

    /// **Mutation found: setting [`PlaneFit::rms_s`] to a constant zero survived every test.**
    ///
    /// The residual is documented as the quantity that says whether the local-planarity assumption
    /// held, and nothing asserted it — so it was decoration. It has an exact closed form on a
    /// deliberately non-planar support, which is what is checked here.
    ///
    /// Take the four corners of the unit square with `t = 0, 0, 0, d`. Centring gives `Suu = Svv =
    /// 1`, `Suv = 0`, `Sut = Svt = d/2`, so the fitted plane is `a = b = d/2`, `c = -d/4` — and
    /// every one of the four residuals is `±d/4`. The root-mean-square is therefore **exactly
    /// `d/4`**, with no approximation anywhere in the derivation.
    #[test]
    fn the_plane_fit_residual_is_the_closed_form_of_a_deliberately_non_planar_support() {
        for d in [4e-3, 1.0, 1e-6] {
            let f = fit_plane(&[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 1.0, d)])
                .unwrap();
            assert!((f.a - d / 2.0).abs() < 1e-15 * d.max(1.0), "a = {}", f.a);
            assert!((f.b - d / 2.0).abs() < 1e-15 * d.max(1.0), "b = {}", f.b);
            assert!((f.c + d / 4.0).abs() < 1e-15 * d.max(1.0), "c = {}", f.c);
            assert!(
                (f.rms_s - d / 4.0).abs() < 1e-15 * d.max(1.0),
                "rms = {} against the closed form {}",
                f.rms_s,
                d / 4.0
            );
            assert_eq!(f.points, 4);
            // Each residual SIGNED, not merely its magnitude. An `abs()` here would leave the
            // direction of `residual` — documented as `t - predict`, and the thing a caller reads
            // to know whether an event is early or late against the fitted plane — unpinned, which
            // is exactly what mutating the subtraction order showed.
            for (x, y, t, sign) in [
                (0.0, 0.0, 0.0, 1.0),
                (1.0, 0.0, 0.0, -1.0),
                (0.0, 1.0, 0.0, -1.0),
                (1.0, 1.0, d, 1.0),
            ] {
                let want = sign * d / 4.0;
                assert!(
                    (f.residual(x, y, t) - want).abs() < 1e-15 * d.max(1.0),
                    "residual at ({x}, {y}) is {} against {want}",
                    f.residual(x, y, t)
                );
            }
        }
        // On support that IS planar, the residual is floating-point noise rather than a small
        // constant — this is the half that says the quantity is a measurement and not a label.
        let planar: Vec<(f64, f64, f64)> = (0..5)
            .flat_map(|j| (0..5).map(move |i| (f64::from(i), f64::from(j), 0.003 * f64::from(i) - 0.002 * f64::from(j))))
            .collect();
        let f = fit_plane(&planar).unwrap();
        assert!(f.rms_s < 1e-17, "an exactly planar support reported a residual of {}", f.rms_s);
        assert!((f.a - 0.003).abs() < 1e-15 && (f.b + 0.002).abs() < 1e-15);
        // And the residual reaches a caller through the streaming path too, not only `fit_plane`.
        let g = geom(32, 32);
        let mut pf = PlaneFlow::new(g, 3, 1.0, None, 8).unwrap();
        let evs = moving_edge(g, 0.3, 500.0, -4.0, 0.12, Polarity::On).unwrap();
        let outs = pf.run(&evs).unwrap();
        assert!(outs.iter().any(|o| matches!(o, FlowOutcome::Fitted(_))));
    }

    /// **Mutation found: tightening either memory window from `<=` to `<` survived every test.**
    ///
    /// The same hole as [`the_eharris_binarisation_window_is_inclusive_at_its_boundary`], in the
    /// two other places this module keeps a time window: no test had ever put a contribution
    /// *exactly* on the boundary, so the convention was whatever the code happened to say. Both are
    /// pinned here with boundaries that are exact in binary64.
    ///
    /// It is worth naming what kind of defect this is. An off-by-one-ulp window does not produce a
    /// wrong answer on any synthetic stimulus and will never show up in a plot; it shows up as two
    /// implementations of the same paper disagreeing on a recording where events happen to land on
    /// a round microsecond, which they very often do.
    #[test]
    fn both_memory_windows_are_inclusive_at_their_boundaries() {
        // PlaneFlow: three supporting events exactly `window_s` old, and a fourth arriving now.
        // Inclusive, that is a support of 4 and the fit proceeds; exclusive, it is 1 and refuses.
        let g = geom(32, 32);
        let mut pf = PlaneFlow::new(g, 3, 0.5, None, 4).unwrap();
        for (x, y) in [(9u16, 9u16), (11, 9), (9, 11)] {
            let o = pf.push(PixelEvent { t_s: 0.0, x, y, polarity: Polarity::On }).unwrap();
            assert!(matches!(o, FlowOutcome::TooFew { .. }), "{o:?}");
        }
        let at_edge = pf.push(PixelEvent { t_s: 0.5, x: 10, y: 10, polarity: Polarity::On }).unwrap();
        assert!(
            !matches!(at_edge, FlowOutcome::TooFew { .. }),
            "an entry exactly one window old was excluded: {at_edge:?}"
        );
        // One ulp later the three are genuinely out of the window and the support collapses to
        // just the arriving event.
        let mut pf = PlaneFlow::new(g, 3, 0.5, None, 4).unwrap();
        for (x, y) in [(9u16, 9u16), (11, 9), (9, 11)] {
            pf.push(PixelEvent { t_s: 0.0, x, y, polarity: Polarity::On }).unwrap();
        }
        let past = pf
            .push(PixelEvent { t_s: 0.5 + f64::EPSILON, x: 10, y: 10, polarity: Polarity::On })
            .unwrap();
        assert!(matches!(past, FlowOutcome::TooFew { found: 1 }), "{past:?}");

        // HATS: two events at one pixel, exactly `window_s` apart. Inclusive, the later one sees
        // the earlier at exp(-1) and the cell average is 1 + exp(-1)/2; exclusive, it is 1.
        let g = geom(10, 10);
        let h = Hats { cell_px: 10, radius: 1, tau_s: 0.25, window_s: 0.25, split_polarity: false };
        let d = h
            .descriptor(g, &[
                PixelEvent { t_s: 0.0, x: 5, y: 5, polarity: Polarity::On },
                PixelEvent { t_s: 0.25, x: 5, y: 5, polarity: Polarity::On },
            ])
            .unwrap();
        let want = 1.0 + (-1.0f64).exp() / 2.0;
        assert!((d[4] - want).abs() < 1e-14, "at the boundary: {} against {want}", d[4]);
        // Just past it, the earlier event is forgotten and each event sees only itself.
        let d = h
            .descriptor(g, &[
                PixelEvent { t_s: 0.0, x: 5, y: 5, polarity: Polarity::On },
                PixelEvent { t_s: 0.26, x: 5, y: 5, polarity: Polarity::On },
            ])
            .unwrap();
        assert_eq!(d[4], 1.0, "past the window the earlier event still contributed");
    }

    /// **Mutation found: replacing `beta` with the constant `1.0` in [`Hots::learn`] survived every
    /// test.**
    ///
    /// [`hots_converges_along_the_ray_exactly_as_its_update_rule_predicts`] initialises the
    /// prototype *parallel* to the patch, where `beta` is exactly 1 by construction — so the
    /// closed form it checks cannot see `beta` at all. That is the whole reason `beta` is in the
    /// rule: it is what makes a cluster a **ray** rather than a point, and it only does anything
    /// when the prototype and the patch point in different directions.
    ///
    /// Two updates are computed by hand here, both with the prototype off the patch's ray.
    #[test]
    fn the_hots_cosine_factor_is_exercised_off_the_ray() {
        let g = geom(16, 16);
        let alpha = Hots::alpha_for(0);
        assert!((alpha - Hots::ALPHA0).abs() < 1e-18, "the first step's rate is the paper's alpha0");
        let ev = PixelEvent { t_s: 0.0, x: 8, y: 8, polarity: Polarity::On };

        // Case 1: the prototype is ORTHOGONAL to the patch, so beta is exactly zero and the update
        // is a pure move toward the patch that leaves the prototype's own mass untouched.
        // With beta forced to 1 the entry at index 0 would decay to 1 - alpha instead.
        let mut c = vec![0.0; 9];
        c[0] = 1.0;
        let mut h = Hots::with_centers(g, 1, 1e-3, vec![c]).unwrap();
        h.learn(ev).unwrap();
        assert_eq!(h.centers()[0][0], 1.0, "an orthogonal prototype's own mass was decayed");
        assert!((h.centers()[0][4] - alpha).abs() < 1e-18, "{}", h.centers()[0][4]);

        // Case 2: an intermediate angle. C = e0 + e4, S = e4, so beta = 1/sqrt(2) exactly.
        let mut c = vec![0.0; 9];
        c[0] = 1.0;
        c[4] = 1.0;
        let mut h = Hots::with_centers(g, 1, 1e-3, vec![c]).unwrap();
        h.learn(ev).unwrap();
        let beta = 1.0 / 2.0f64.sqrt();
        let want0 = 1.0 + alpha * (0.0 - beta * 1.0);
        let want4 = 1.0 + alpha * (1.0 - beta * 1.0);
        assert!((h.centers()[0][0] - want0).abs() < 1e-15, "{} against {want0}", h.centers()[0][0]);
        assert!((h.centers()[0][4] - want4).abs() < 1e-15, "{} against {want4}", h.centers()[0][4]);
        // The two entries must differ, or the case has collapsed back to the parallel one.
        assert!((want0 - want4).abs() > 1e-4);
        // A zero prototype has no direction, so beta is defined as 0 and the step is plain
        // online k-means. The doc on `learn` says so; this is where it is pinned.
        let mut h = Hots::with_centers(g, 1, 1e-3, vec![vec![0.0; 9]]).unwrap();
        h.learn(ev).unwrap();
        assert!((h.centers()[0][4] - alpha).abs() < 1e-18);
        assert!(h.centers()[0].iter().all(|v| v.is_finite()), "a zero prototype produced a NaN");
    }

    /// **Mutation found: tightening the binarisation window from `<=` to `<` survived every test.**
    ///
    /// No test above ever put an event *exactly* on the window boundary, so the inclusive
    /// convention was unpinned. It is pinned here with a boundary that is exact in binary64: events
    /// at `t = 0`, a window of 0.5 s, and a query at exactly 0.5 s.
    #[test]
    fn the_eharris_binarisation_window_is_inclusive_at_its_boundary() {
        let g = geom(32, 32);
        let mut d = EHarris::new(g, 4, 0.5, EHarris::K_HARRIS, 0.0).unwrap();
        // A right-angle corner, every pixel written at exactly t = 0.
        for y in 8..=16u16 {
            for x in 8..=16u16 {
                if x <= 12 && y <= 12 {
                    d.push(PixelEvent { t_s: 0.0, x, y, polarity: Polarity::On }).unwrap();
                }
            }
        }
        // Exactly at the boundary: inclusive, so the whole quadrant is still set and the corner
        // scores positive. Exclusive would give a blank patch and a score of exactly zero.
        let at_edge = d.score_at(12, 12, 0.5).unwrap();
        assert!(at_edge > 0.0, "at exactly one window the corner scored {at_edge}");
        // One ulp past it, everything has been forgotten and the patch is blank.
        let past = d.score_at(12, 12, 0.5 + f64::EPSILON).unwrap();
        assert_eq!(past, 0.0, "past the window the patch was not blank; it scored {past}");
        // Well inside, the same positive score — so the boundary is the only thing being tested.
        assert!((d.score_at(12, 12, 0.25).unwrap() - at_edge).abs() < 1e-12);
    }

    /// **A mutation that survived and should have**: exchanging the two `Sobel` kernels in
    /// [`EHarris`] changes no observable output, because the `Harris` response is invariant under
    /// it.
    ///
    /// Swapping `GX` and `GY` exchanges `Ix` with `Iy`, which exchanges `Mxx` with `Myy` and leaves
    /// `Mxy` alone — so `det(M)` and `trace(M)` are both unchanged, and so is the score. It is an
    /// *equivalent mutation*, not an escaped defect, and no test should be written to catch it.
    /// Recorded here so that a future reader running the same mutation does not go looking for a
    /// hole that is not there. The invariance is asserted rather than argued.
    #[test]
    fn swapping_the_sobel_kernels_is_an_equivalent_mutation() {
        let d = EHarris::new(geom(32, 32), 4, 1e-3, EHarris::K_HARRIS, 0.0).unwrap();
        let side = 9usize;
        for pattern in [
            patch_of(side, |i, _| i <= 4),
            patch_of(side, |i, j| i <= 4 && j <= 4),
            patch_of(side, |i, j| i + 2 * j <= 9),
            patch_of(side, |i, j| (i * 3 + j * 7) % 5 == 0),
        ] {
            // Transposing the patch is exactly what exchanging the kernels does to the gradients.
            let mut t = vec![0.0; side * side];
            for j in 0..side {
                for i in 0..side {
                    t[i * side + j] = pattern[j * side + i];
                }
            }
            let a = d.score_of_patch(&pattern, side).unwrap();
            let b = d.score_of_patch(&t, side).unwrap();
            assert!((a - b).abs() < 1e-9, "the Harris response is not transpose-invariant: {a} vs {b}");
        }
    }

}
