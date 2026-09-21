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

/// Reject a neighbourhood half-width that no patch over `geom` could usefully carry.
///
/// A patch of half-width `r` centred on any pixel of the lattice already covers **every** column
/// and row once `r` reaches `max(width, height) - 1`, so every larger `r` adds nothing but the
/// zero padding [`TimeSurface::patch`] documents, while the buffer grows as `(2r + 1)^2`. Leaving
/// `r` unbounded is also an arithmetic hazard rather than merely a wasteful one: `2 * radius + 1`
/// overflows `u16` at `radius = 32768`, and `EHarris::new(geom(64, 64), 32768, ..)` used to be
/// accepted and then panic on the first event. The bound is stated at every constructor that
/// stores a radius, so the refusal arrives before the allocation rather than during it.
fn radius_within(geom: Geometry, radius: u16) -> Result<usize, VisionError> {
    let r = usize::from(radius);
    if r + 1 > usize::from(geom.width.max(geom.height)) {
        return Err(VisionError::BadParameters {
            why: "the neighbourhood half-width reaches past the sensor's larger dimension, \
                  beyond which every further ring is padding",
        });
    }
    Ok(r)
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
    /// for a bad query time, and [`VisionError::BadParameters`] when `radius` reaches the sensor's
    /// larger dimension — past that point the patch is all padding and `2 * radius + 1` is no
    /// longer a `u16`. See the note on the private `radius_within`.
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
        let side = 2 * radius_within(self.geom, radius)? + 1;
        let r = i64::from(radius);
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

    /// Whether `data` holds exactly `width * height` entries.
    ///
    /// The fields are public so that an image can be written down as a literal — which is what
    /// makes the statistics below checkable against arithmetic — and the price is that the type
    /// cannot enforce its own shape in a constructor. Everything this module *produces* is
    /// consistent; a hand-built `Frame` that is not reports `false` here, and the accessors say
    /// what they do in that case rather than panicking: [`Frame::at`] returns `None` for a pixel
    /// whose datum is missing, and [`Frame::sum`], [`Frame::mean`] and [`Frame::variance`] are
    /// over `data` as it stands rather than over the lattice `width` and `height` claim.
    #[must_use]
    pub fn is_consistent(&self) -> bool {
        self.data.len() == self.geometry().pixels()
    }

    /// The value at a pixel, or `None` off the lattice — or when `data` is shorter than the
    /// lattice claims, for which see [`Frame::is_consistent`].
    #[must_use]
    pub fn at(&self, x: u16, y: u16) -> Option<f64> {
        self.geometry().index(x, y).and_then(|i| self.data.get(i).copied())
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

    /// Mean over `data`, or `None` for an empty image.
    ///
    /// Over `data` rather than over `width * height`: see [`Frame::is_consistent`] for the one
    /// case where those differ.
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
///
/// `reject_s` is **inclusive**, exactly as the two memory windows in this module are: a point
/// whose residual is *exactly* `reject_s` is the last one kept. Which way that comparison faces
/// is not a detail on a real recording — events land on round microseconds, so residuals land on
/// exact multiples of the quantum and the boundary case is the common case, not the rare one —
/// and it is pinned by a test rather than left to whichever comparison someone writes next.
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
    ///   support — when `radius` reaches the sensor's larger dimension, past which the
    ///   neighbourhood loop is all off-lattice and a `u16` radius costs `(2r+1)^2` iterations per
    ///   event, or when `reject_s` exceeds `window_s`, which is a rejection pass that can never
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
        radius_within(geom, radius)?;
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
    /// zero radius, [`VisionError::BadParameters`] for a radius at or past the sensor's larger
    /// dimension, and anything [`TimeSurface::new`] refuses.
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
        // Bounded before the draw, not after: `(2 * radius + 1)^2` is the length of every vector
        // about to be allocated, and on a 32-bit target it overflows `usize` for a large `u16`.
        let side = (2 * radius_within(geom, radius)? + 1).pow(2);
        let centers: Vec<Vec<f64>> =
            (0..n_centers).map(|_| (0..side).map(|_| rng.next_f64()).collect()).collect();
        Self::with_centers(geom, radius, tau_s, centers)
    }

    /// A layer with prototypes supplied — a trained dictionary, or a hand-built one for a test.
    ///
    /// # Errors
    ///
    /// [`VisionError::TooFew`] when the set is empty or the radius is zero;
    /// [`VisionError::BadParameters`] when a prototype's length is not `(2 * radius + 1)^2`, or
    /// when `radius` reaches the sensor's larger dimension — beyond that the patch
    /// [`Hots::learn`] takes is all padding, and `(2 * radius + 1)^2` overflows a 32-bit `usize`
    /// before the padding runs out; [`VisionError::NonFinite`] for a non-finite prototype entry;
    /// plus anything [`TimeSurface::new`] refuses.
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
        let want = (2 * radius_within(geom, radius)? + 1).pow(2);
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
    ///
    /// Every field of this type is public, so `radius` can be any `u16`, and at the top of that
    /// range the square overflows a 32-bit `usize` — silently, in a release `wasm32` build. It is
    /// therefore saturating here and refused outright by [`Hats::descriptor_len`], which is the
    /// fallible entry point every consumer goes through.
    #[must_use]
    pub fn bins_per_cell(self) -> usize {
        let side = 2 * usize::from(self.radius) + 1;
        side.saturating_mul(side)
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
    /// As [`Hats::cell_grid`], plus [`VisionError::BadParameters`] when the product does not fit
    /// this target's `usize`. That is not a theoretical branch: the fields are public, so a
    /// `radius` near `u16::MAX` is constructible, and `(2 * radius + 1)^2` passes `u32` at
    /// `radius = 32767`. Refusing here is what keeps [`Hats::descriptor`]'s `bins` and its own
    /// `side * side` the same number, which is what keeps its writes in bounds.
    pub fn descriptor_len(self, geom: Geometry) -> Result<usize, VisionError> {
        let (cx, cy) = self.cell_grid(geom)?;
        let planes = if self.split_polarity { 2 } else { 1 };
        let side = 2 * usize::from(self.radius) + 1;
        let too_long = || VisionError::BadParameters {
            why: "the HATS descriptor length does not fit this target's usize",
        };
        let bins = side.checked_mul(side).ok_or_else(too_long)?;
        cx.checked_mul(cy)
            .and_then(|c| c.checked_mul(planes))
            .and_then(|c| c.checked_mul(bins))
            .ok_or_else(too_long)
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
    /// # The divisor is the cell's total, not the plane's — and that is a choice
    ///
    /// With `split_polarity` set, **both** planes of a cell are divided by that cell's *total*
    /// event count, not by the count of their own polarity. The consequence is worth stating
    /// plainly, because it is not what "the average of the plane's time surfaces" would mean: an
    /// `On`-dominated cell and a balanced cell with **identical `On` texture** produce different
    /// `On` planes, scaled by the polarity mix. [`Hats::descriptor`]'s own test asserts exactly
    /// that — one `On` and one `Off` event in a cell give `0.5`, not `1.0`, in each plane's centre
    /// bin.
    ///
    /// **This implementation did not locate an author-released reference implementation to check
    /// that reading against**, and it is a place where the arithmetic is a decision rather than a
    /// derivation — unlike the memory surface itself, which the paper writes out. It is flagged
    /// here for the same reason the `tau` reading is flagged on the type doc: a reader with the
    /// paper should check, and this is where a disagreement should be recorded.
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
    /// `Harris`'s own sensitivity constant, dimensionless, as the 1988 paper prints it: `0.04`.
    ///
    /// # What the number decides
    ///
    /// It sets the largest eigenvalue ratio that still scores positive. Writing `rho = l1 / l2`
    /// with `l1 >= l2 > 0`, the response `l1 * l2 - k * (l1 + l2)^2` is positive exactly when
    /// `rho > k * (1 + rho)^2`, so the admitted ratios run up to
    ///
    /// ```text
    /// rho_max(k) = ((1 - 2k) + sqrt(1 - 4k)) / (2k)
    /// ```
    ///
    /// which is `22.956...` at `k = 0.04` and `14.598...` at `k = 0.06`. Raising `k` makes the
    /// detector fussier about how square a corner is, and `k = 0.25` admits nothing at all.
    ///
    /// An earlier revision of this doc said `0.04` admits "about a 19:1 ratio". That was wrong —
    /// 19:1 is what `k = 0.0475` admits — and the claim now has a test: a patch whose structure
    /// tensor has eigenvalues in exactly the ratio `20.5:1`, which sits between the two bounds
    /// above, scores positive at `0.04` and negative at `0.06`.
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
    /// directions. [`VisionError::BadParameters`] when `radius` reaches the sensor's larger
    /// dimension: the window is then all padding, and `2 * radius + 1` stops being a `u16`, which
    /// used to be accepted here and panic inside [`EHarris::score_at`] on the first event.
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
        radius_within(geom, radius)?;
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
        // `usize`, not `u16`: `2 * radius + 1` wraps a `u16` at radius 32768. `EHarris::new`
        // refuses any radius that large, and the arithmetic here does not rely on it having.
        let side = 2 * usize::from(self.radius) + 1;
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

    /// The criterion itself, in one place: a corner is declared when **both** circles produced an
    /// arc, and the verdict carries the inner arc's length because that is what [`Corner::score`]
    /// reports.
    ///
    /// [`EFast::is_corner`] and [`EFast::push`] both route through this rather than each spelling
    /// out `inner.is_some() && outer.is_some()`. Two encodings of one rule is how the inner-only
    /// weakening — which changes the detection count on a moving corner by about 12% — came to
    /// survive a suite that asserts the `AND` through `is_corner` and never through `push`.
    fn verdict(arcs: (Option<usize>, Option<usize>)) -> Option<usize> {
        match arcs {
            (Some(inner), Some(_outer)) => Some(inner),
            _ => None,
        }
    }

    /// Whether a pixel satisfies the criterion on both circles.
    ///
    /// # Errors
    ///
    /// As [`EFast::arcs_at`].
    pub fn is_corner(&self, x: u16, y: u16, pol: Polarity) -> Result<bool, VisionError> {
        Ok(Self::verdict(self.arcs_at(x, y, pol)?).is_some())
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
            Ok(arcs) => Ok(Self::verdict(arcs)
                .map(|i| Corner { t_s: e.t_s, x: e.x, y: e.y, score: i as f64 })),
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

/// The largest number of events [`rotating_bar`] will produce before refusing: ten million.
///
/// The generator emits one event per annulus pixel per half turn, so the count is
/// `annulus_pixels * (1 + omega_rad_s * duration_s / pi)` and **nothing in the parameter list
/// bounds it**: `positive()` accepts any finite `omega_rad_s`, and on a 3x3 sensor the measured
/// counts run 1274, 12734, 127324, 1273240 for `omega` of 1e3 to 1e6 — linear, with no ceiling.
/// At `omega = 1e12` over one second the request is ~1.3e12 events, about 30 TB.
///
/// Ten million is where this crate draws the line: 240 MB of `PixelEvent`, well past any stimulus
/// the module's own tests build (a few thousand) and well short of a machine. A caller that wants
/// more can call the generator in slices of `duration_s`.
pub const MAX_ROTATING_BAR_EVENTS: usize = 10_000_000;

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
/// `duration_s`; [`VisionError::BadParameters`] when `r_min_px >= r_max_px`, or when the
/// parameters ask for more than [`MAX_ROTATING_BAR_EVENTS`] events;
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
    let period = pi / omega;
    // How many times the bar can sweep one pixel inside the recording. Each pixel of the annulus
    // fires once per half turn, so this is `floor(duration / period) + 1` at most, whatever the
    // phase. Nothing in the signature bounds it: at `omega = 1e12` and one second this asks for
    // ~1.3e12 events, and once `period` falls below `duration / 2^53` the crossing counter `k`
    // stops advancing in binary64 and the loop below would never end at all. So it is counted
    // first and refused here, rather than discovered as a hang.
    let per_pixel = (duration / period).floor() + 1.0;
    let annulus = (0..geom.height)
        .flat_map(|y| (0..geom.width).map(move |x| (x, y)))
        .filter(|&(x, y)| {
            let r = (f64::from(x) - cx).hypot(f64::from(y) - cy);
            (r_min..=r_max).contains(&r)
        })
        .count();
    if !per_pixel.is_finite() || per_pixel * annulus as f64 > MAX_ROTATING_BAR_EVENTS as f64 {
        return Err(VisionError::BadParameters {
            why: "the angular velocity and duration ask for more rotating-bar events than \
                  MAX_ROTATING_BAR_EVENTS allows",
        });
    }
    // Safe: `per_pixel` is finite, at least 1, and bounded by the check above.
    let sweeps = per_pixel as u64 + 1;
    let mut out = Vec::with_capacity(
        annulus.saturating_mul(sweeps as usize).min(MAX_ROTATING_BAR_EVENTS),
    );
    for y in 0..geom.height {
        for x in 0..geom.width {
            let (dx, dy) = (f64::from(x) - cx, f64::from(y) - cy);
            let r = dx.hypot(dy);
            if r < r_min || r > r_max {
                continue;
            }
            let phi = dy.atan2(dx);
            // Smallest k putting t at or after zero, then every k until the duration runs out.
            // The iteration count is the bound established above; the `t > duration` test is what
            // still decides, so the emitted set is exactly what the closed form names.
            let base = (phi - theta0) / omega;
            let k0 = (-base / period).ceil();
            for i in 0..sweeps {
                let t = base + (k0 + i as f64) * period;
                if t > duration {
                    break;
                }
                if t >= 0.0 {
                    out.push(PixelEvent { t_s: t, x, y, polarity });
                }
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
        Decay, EFast, EHarris, FAST_CIRCLE_16, FAST_CIRCLE_20, Flow, FlowOutcome, Frame, Geometry,
        Hats, Hots, MAX_ROTATING_BAR_EVENTS, Motion, Objective, PixelEvent, PlaneFlow, TimeSurface,
        VisionError, accumulate_count, accumulate_decay, accumulate_polarity, argmax, contrast,
        fit_plane, flow_error, looming_disc, looming_disc_flow, moving_corner, moving_edge,
        moving_edge_flow, newest_arc, quantise_microseconds, require_time_ordered, rotating_bar,
        rotating_bar_flow, search_translation, sweep, warped_image,
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
    /// because `k = 0.04` admits eigenvalue ratios up to `22.956...`; the closed form is on
    /// [`EHarris::K_HARRIS`] and the ratio itself is checked in
    /// `the_harris_constant_sets_the_eigenvalue_ratio_its_doc_names`.
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

        // **And the same criterion through `push`**, which is the streaming API every consumer
        // uses and which carried its own second copy of the rule until this audit. Every
        // assertion above goes through `arcs_at`/`is_corner`; weakening `push` to accept an inner
        // arc alone survived all of them, and it is not an equivalent mutant — on this module's
        // own moving-corner stream the AND fires 231 times and inner-only 258.
        //
        // The centre pixel is on neither circle, so `push`'s own `update` cannot change the
        // fixture; what it returns is the verdict and nothing else.
        let fire = |mut d: EFast| {
            d.push(PixelEvent { t_s: 1.0, x: cx, y: cy, polarity: Polarity::On }).unwrap()
        };
        // The score a corner reports is the INNER arc length, which is what makes these two
        // different assertions rather than one repeated.
        assert_eq!(fire(build(4, 5)).map(|c| c.score), Some(4.0));
        assert_eq!(fire(build(3, 4)).map(|c| c.score), Some(3.0));
        assert_eq!(fire(build(6, 8)).map(|c| c.score), Some(6.0));
        assert!(fire(build(8, 10)).is_none(), "push fired on a half circle");
        assert!(fire(build(1, 1)).is_none(), "push fired on an isolated pixel");
        // Good inner arc, bad outer one: the case the AND exists for.
        assert!(fire(build(4, 12)).is_none(), "push accepted an inner arc with no outer one");
        // And the mirror image, so the assertion is about the conjunction rather than about the
        // outer circle alone.
        assert!(fire(build(9, 5)).is_none(), "push accepted an outer arc with no inner one");
    }

    /// **The two `Bresenham` circles, against the tables `FAST` prints** — as literals, because
    /// every other `eFAST` test in this file writes its fixture *through* these same tables and so
    /// cancels their geometry out of both sides of its own assertion.
    ///
    /// Changing `FAST_CIRCLE_16[2]` from `(2, 2)` to `(2, 1)`, or duplicating an offset so the
    /// "circle" has a repeated pixel and a missing one, used to survive all 60 tests.
    ///
    /// The literals are the standard `FAST` circle of radius 3 (Rosten and Drummond, *Machine
    /// Learning for High-Speed Corner Detection*, ECCV 2006) and the radius-4 circle `eFAST`'s
    /// second test uses. Below them the three properties that make the arc search meaningful are
    /// asserted from the geometry rather than from the table: every offset sits on its circle,
    /// every offset is distinct, consecutive offsets are adjacent pixels, and the polar angle
    /// advances monotonically so that "a window over consecutive entries" really is an arc.
    #[test]
    fn the_fast_circles_are_the_bresenham_rings_their_docs_name() {
        assert_eq!(
            FAST_CIRCLE_16,
            [
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
            ]
        );
        assert_eq!(
            FAST_CIRCLE_20,
            [
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
            ]
        );

        for (circle, radius) in [(&FAST_CIRCLE_16[..], 3usize), (&FAST_CIRCLE_20[..], 4)] {
            let n = circle.len();
            let r = radius as f64;
            assert_eq!(n, 4 * radius + 4, "the ring has the wrong circumference");
            for (i, &(dx, dy)) in circle.iter().enumerate() {
                // On the circle to within half a pixel, which is what rasterising it means.
                let d = (dx as f64).hypot(dy as f64);
                assert!((d - r).abs() <= 0.5, "offset {i} = ({dx}, {dy}) is {d} from the centre");
                // Distinct: a repeated pixel would let one timestamp vote twice in the arc test.
                for (j, &other) in circle.iter().enumerate() {
                    assert!(i == j || (dx, dy) != other, "offsets {i} and {j} are the same pixel");
                }
                // Adjacent to its successor around the ring, including across the seam. This is
                // what makes a contiguous window over the slice a contiguous arc on the sensor.
                let (nx, ny) = circle[(i + 1) % n];
                assert_eq!(
                    (nx - dx).abs().max((ny - dy).abs()),
                    1,
                    "offsets {i} and {} are not neighbouring pixels",
                    (i + 1) % n
                );
                // The centre is not on the ring.
                assert!((dx, dy) != (0, 0));
            }
            // The polar angle decreases monotonically over exactly one turn, so the ordering is
            // circular rather than merely a permutation that happens to start and end together.
            let mut prev = (circle[0].1 as f64).atan2(circle[0].0 as f64);
            let mut wraps = 0;
            for &(dx, dy) in &circle[1..] {
                let a = (dy as f64).atan2(dx as f64);
                if a > prev {
                    wraps += 1;
                    assert!(wraps <= 1, "the ring changes direction at ({dx}, {dy})");
                }
                prev = a;
            }
            assert_eq!(wraps, 1, "the ring does not cover exactly one turn");
        }
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

        // The closed form, computed here and nowhere else in the crate — and out of the PAPER'S
        // two constants written as literals, not out of `Hots::alpha_for`. An earlier revision
        // built the product from the code's own rate, so the constants cancelled from both sides
        // and `ALPHA_DECAY: 20000.0 -> 10000.0` moved the layer's whole learning schedule while
        // this assertion held to 1e-12.
        let mut prod = 1.0f64;
        for n in 0..steps {
            prod *= 1.0 - 0.01 / (1.0 + n as f64 / 20000.0);
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
        // A prototype is exactly zero from itself. That is the easy half, and on its own it is no
        // test at all: sqrt(0) == 0, so deleting the `.sqrt()` in `nearest` survived it.
        assert_eq!(h.nearest(&b).unwrap().1, 0.0);
        assert_eq!(h.nearest(&a).unwrap().1, 0.0);
        // The half that was missing. The distance is the EUCLIDEAN one, not the squared one, and
        // the probe has to be a vector the dictionary does not contain for the two to differ.
        // `c` is orthogonal to both `a` and `b`, so both squared distances are exactly 2 and both
        // Euclidean distances are sqrt(2) = 1.41421356... Returning the square would give 2.
        let ab = Hots::with_centers(g, 1, 1e-3, vec![a.clone(), b.clone()]).unwrap();
        let (k, dist) = ab.nearest(&c).unwrap();
        assert_eq!(k, 0, "the tie between two equidistant prototypes did not go to the low index");
        assert!(
            (dist - core::f64::consts::SQRT_2).abs() < 1e-15,
            "two orthogonal unit vectors came out {dist} apart, not sqrt(2)"
        );
        // Named against the squared distance too, because that is the mutation: 2.0, not 1.414.
        assert!((dist - 2.0).abs() > 0.5, "the distance returned was the SQUARED one");
        // And an asymmetric probe, so the answer is not a special value of the symmetric case:
        // 0.5 * c is 0.5 from c, and sqrt(1 + 0.25) = 1.118... from a. The square would be 1.25.
        let mut half = vec![0.0; 9];
        half[5] = 0.5;
        let (k2, d2) = h.nearest(&half).unwrap();
        assert_eq!(k2, 2, "the nearest prototype to half of c is c");
        assert!((d2 - 0.5).abs() < 1e-15, "distance {d2}, not 0.5");
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


    // -----------------------------------------------------------------------------------------
    // Gaps found by a SECOND mutation audit of this module
    // -----------------------------------------------------------------------------------------
    //
    // Ninety-six further mutations were applied to the code above and the suite re-run. The
    // twenty-five the previous round claimed to catch were all caught; thirty-eight more escaped,
    // and they cluster into the tests below. The pattern is the same one this crate keeps finding:
    // the CENTRAL arithmetic of each mechanism is well tested, and its periphery — the constant
    // table, the second copy of a criterion, the measuring instrument itself, the parameters every
    // test leaves at their defaults, the refusal branches — is not.

    /// **The instrument the module's central check reads its angular error from.**
    ///
    /// `plane_fit_flow_recovers_the_generated_velocity_exactly` asserts `mean_angle_err < 1e-9`,
    /// and the accumulation behind that number had no test of its own: `ae += d` weakened to
    /// `ae += 0.0 * d` survived all 60 tests, as did deleting the wrap loop whose doc argues that
    /// "359 degrees is one degree of error, not 359". The module guards the other two error fields
    /// against exactly this — the cone test on `mean_speed_err`, the quantised test on
    /// `max_rel_speed_err` — and this is the guard that was never written for the third.
    ///
    /// Everything below is arithmetic on hand-built outcomes, so every expected value is a literal.
    #[test]
    fn flow_error_measures_the_angle_it_names_and_wraps_it_onto_the_circle() {
        let ev = |t: f64| PixelEvent { t_s: t, x: 1, y: 1, polarity: Polarity::On };
        let fit = |speed: f64, dir: f64| {
            FlowOutcome::Fitted(Flow {
                vx: speed * dir.cos(),
                vy: speed * dir.sin(),
                speed,
                direction_rad: dir,
            })
        };

        // One event, one fit, against a truth pointing along +x at 100 px/s. The angular error is
        // exactly 0.25 rad, the speed error exactly 10 px/s, the relative one exactly 0.1.
        let e = flow_error(&[ev(0.0)], &[fit(110.0, 0.25)], |_| Some((100.0, 0.0))).unwrap();
        assert_eq!((e.fitted, e.offered), (1, 1));
        assert!((e.mean_angle_err - 0.25).abs() < 1e-15, "angle {}", e.mean_angle_err);
        assert!((e.mean_speed_err - 10.0).abs() < 1e-12, "speed {}", e.mean_speed_err);
        assert!((e.max_rel_speed_err - 0.1).abs() < 1e-15, "relative {}", e.max_rel_speed_err);

        // THE WRAP. An estimate at -(pi - 0.01) against a truth at +(pi - 0.01) is 0.02 rad out,
        // not 2 pi - 0.02 = 6.2631853. Without the wrap loop this reads the second number, and the
        // tests' own `wrapped_angle` helper is a SEPARATE implementation used only by the rotating
        // bar, so `flow_error`'s own wrap had never run on an input that needed it.
        let pi = core::f64::consts::PI;
        let near = pi - 0.01;
        let e = flow_error(&[ev(0.0)], &[fit(100.0, -near)], |_| {
            Some((100.0 * near.cos(), 100.0 * near.sin()))
        })
        .unwrap();
        assert!((e.mean_angle_err - 0.02).abs() < 1e-12, "wrapped to {}", e.mean_angle_err);
        // The unwrapped value, named, so the assertion above cannot be read as approximate.
        assert!(e.mean_angle_err < 0.1, "the angular error was not wrapped at all");

        // The mean is a mean: errors of 0.0 and 0.5 over two events give 0.25.
        let e = flow_error(
            &[ev(0.0), ev(1.0)],
            &[fit(100.0, 0.0), fit(100.0, 0.5)],
            |_| Some((100.0, 0.0)),
        )
        .unwrap();
        assert_eq!(e.fitted, 2);
        assert!((e.mean_angle_err - 0.25).abs() < 1e-15, "mean {}", e.mean_angle_err);

        // An event that did not fit counts as offered and not as fitted; one whose ground truth is
        // undefined counts in neither. Those are different exclusions and both are asserted.
        let e = flow_error(
            &[ev(0.0), ev(1.0)],
            &[fit(100.0, 0.0), FlowOutcome::Degenerate],
            |_| Some((100.0, 0.0)),
        )
        .unwrap();
        assert_eq!((e.fitted, e.offered), (1, 2));
        let e = flow_error(&[ev(0.0), ev(1.0)], &[fit(100.0, 0.0), fit(100.0, 0.0)], |e| {
            (e.t_s == 0.0).then_some((100.0, 0.0))
        })
        .unwrap();
        assert_eq!((e.fitted, e.offered), (1, 1));

        // A zero-speed ground truth has no relative error to report — the division would be
        // infinite — but still has an absolute one.
        let e = flow_error(&[ev(0.0)], &[fit(3.0, 0.0)], |_| Some((0.0, 0.0))).unwrap();
        assert_eq!(e.max_rel_speed_err, 0.0);
        assert!((e.mean_speed_err - 3.0).abs() < 1e-15);

        // A length mismatch is a refusal, not a `zip` that silently truncates — which is precisely
        // the "decoder quietly eating valid records" failure this module's docs keep citing.
        assert!(matches!(
            flow_error(&[ev(0.0), ev(1.0)], &[fit(100.0, 0.0)], |_| Some((100.0, 0.0))),
            Err(VisionError::TooFew { .. })
        ));
        assert!(matches!(
            flow_error(&[ev(0.0)], &[fit(100.0, 0.0), fit(100.0, 0.0)], |_| Some((100.0, 0.0))),
            Err(VisionError::TooFew { .. })
        ));
        // And a mean over nothing is refused rather than reported as zero.
        assert!(matches!(
            flow_error(&[ev(0.0)], &[FlowOutcome::Degenerate], |_| Some((100.0, 0.0))),
            Err(VisionError::TooFew { .. })
        ));
    }

    /// **[`Objective::Variance`] and [`Objective::MeanSquare`] are different functions, and the
    /// suite could not tell them apart in either direction.**
    ///
    /// `contrast_maximisation_finds_the_true_translation` loops over both and asks only whether
    /// the argmax lands on truth, which both satisfy — so replacing the `MeanSquare` arm with
    /// `variance()` survived, and so did replacing the `Variance` arm with `sum_of_squares / n`.
    /// On that test's own stimulus the ratio between the two objectives is 64.0 at `vx = 0` and
    /// 1.016 at truth, so they are nowhere near each other; the test simply never looked.
    ///
    /// Here both are read off an image whose four pixels are written down, so both expected values
    /// are literals — and the `MeanSquare` doc's claim that it "differs from variance by exactly
    /// `mean^2`" becomes the third assertion rather than a sentence.
    #[test]
    fn the_two_focus_measures_are_different_functions_of_the_same_image() {
        let g = geom(2, 2);
        let evs = [PixelEvent { t_s: 0.0, x: 0, y: 0, polarity: Polarity::On }];
        let m = Motion::Translation { vx: 0.0, vy: 0.0 };
        // One event warped by nothing: the image is exactly [1, 0, 0, 0].
        let wi = warped_image(g, &evs, 0.0, m).unwrap();
        assert_eq!(wi.frame.data, vec![1.0, 0.0, 0.0, 0.0]);
        assert_eq!((wi.placed, wi.dropped), (1, 0));

        // sum_of_squares = 1 over n = 4 pixels.
        assert_eq!(contrast(g, &evs, 0.0, m, Objective::MeanSquare).unwrap(), 0.25);
        // mean = 0.25, so the population variance is (0.5625 + 3 * 0.0625) / 4.
        assert_eq!(contrast(g, &evs, 0.0, m, Objective::Variance).unwrap(), 0.1875);
        // And the stated difference: mean^2 = 0.0625.
        let ms = contrast(g, &evs, 0.0, m, Objective::MeanSquare).unwrap();
        let var = contrast(g, &evs, 0.0, m, Objective::Variance).unwrap();
        assert_eq!(ms - var, 0.0625);
        assert_eq!(ms - var, wi.frame.mean().unwrap().powi(2));

        // A second image, so neither number is a special value of the one-event case: two events
        // on one pixel give [2, 0, 0, 0], mean 0.5, MeanSquare 1.0, variance 0.75.
        let two = [
            PixelEvent { t_s: 0.0, x: 0, y: 0, polarity: Polarity::On },
            PixelEvent { t_s: 0.0, x: 0, y: 0, polarity: Polarity::Off },
        ];
        assert_eq!(contrast(g, &two, 0.0, m, Objective::MeanSquare).unwrap(), 1.0);
        assert_eq!(contrast(g, &two, 0.0, m, Objective::Variance).unwrap(), 0.75);
    }

    /// **The two `HOTS` constants against the paper, written out as literals.**
    ///
    /// [`Hots::ALPHA0`] and [`Hots::ALPHA_DECAY`] are transcriptions, and the only way to check a
    /// transcription is to put the source's number beside it. The convergence test above checks
    /// the *recurrence structure*, which is a different and worthwhile claim — but it held to
    /// 1e-12 with `ALPHA_DECAY` halved, because it built its closed form out of `alpha_for`.
    ///
    /// Lagorce, Orchard, Gallupi, Shi and Benosman, IEEE `TPAMI` 39(7), 2017, print
    /// `alpha = 0.01 / (1 + p_k / 20000)`.
    #[test]
    fn the_hots_learning_rate_is_the_papers_two_constants() {
        assert_eq!(Hots::ALPHA0, 0.01);
        assert_eq!(Hots::ALPHA_DECAY, 20000.0);
        // The rule at three points, each from the paper's expression rather than read back off it.
        assert_eq!(Hots::alpha_for(0), 0.01);
        assert_eq!(Hots::alpha_for(20000), 0.01 / 2.0);
        assert_eq!(Hots::alpha_for(60000), 0.01 / 4.0);
        // It decays, and it never reaches zero: a cluster slows down, it does not stop learning.
        let mut prev = f64::INFINITY;
        for n in [0u64, 1, 100, 20000, 1_000_000] {
            let a = Hots::alpha_for(n);
            assert!(a > 0.0 && a < prev, "alpha_for({n}) = {a} against {prev}");
            prev = a;
        }
    }

    /// **The patch layout, on a neighbourhood that is not symmetric.**
    ///
    /// [`TimeSurface::patch`]'s doc says row-major, and transposing the write survived all 60
    /// tests: every `HOTS` test drives a single pixel, which makes the patch the centre indicator
    /// and therefore symmetric, and the border test uses the corner `(0, 0)`, whose result is
    /// symmetric too.
    ///
    /// Three pixels here fire at three different times, so the three non-zero entries are three
    /// different numbers and a transpose exchanges two of them.
    #[test]
    fn a_patch_is_row_major_and_an_asymmetric_neighbourhood_shows_it() {
        let g = geom(16, 16);
        let mut ts = TimeSurface::new(g, Decay::Exponential { tau_s: 1e-3 }, false).unwrap();
        for (t, x, y) in [(0.0, 5u16, 5u16), (1e-3, 6, 5), (2e-3, 5, 6)] {
            ts.update(PixelEvent { t_s: t, x, y, polarity: Polarity::On }).unwrap();
        }
        let p = ts.patch(5, 5, 1, Polarity::On, 2e-3).unwrap();
        assert_eq!(p.len(), 9);
        // Row-major over offsets (dx, dy) in -1..=1: the index is (dy + 1) * 3 + (dx + 1).
        // (0, 0) is the centre, index 4, and fired two time constants ago.
        assert!((p[4] - (-2.0f64).exp()).abs() < 1e-15, "the centre holds {}", p[4]);
        // (+1, 0) — one COLUMN right, same row — is index 5, and fired one time constant ago.
        assert!((p[5] - (-1.0f64).exp()).abs() < 1e-15, "dx = +1 landed at {}", p[5]);
        // (0, +1) — one ROW down, same column — is index 7, and fired just now.
        assert_eq!(p[7], 1.0, "dy = +1 landed at {}", p[7]);
        // Transposing the write exchanges 5 and 7, and those two hold different numbers.
        assert!(p[5] != p[7], "the two off-centre entries are indistinguishable");
        for i in [0usize, 1, 2, 3, 6, 8] {
            assert_eq!(p[i], 0.0, "entry {i} is not zero");
        }
    }

    /// **[`Hots::assign`] — the inference path, the thing the type doc calls the layer's output —
    /// appeared exactly once in this file: its own definition.** Replacing its body with `Ok(0)`
    /// survived all 60 tests.
    ///
    /// It makes two claims and both are checked: it returns the *nearest* prototype, and it does
    /// so **without learning**.
    #[test]
    fn hots_assign_reports_the_cluster_without_moving_it() {
        let g = geom(16, 16);
        let ev = PixelEvent { t_s: 0.0, x: 8, y: 8, polarity: Polarity::On };
        // Driving one pixel makes the patch exactly the centre indicator, so the nearest prototype
        // is known by hand: `near` is parallel to it at half the length, `far` is orthogonal.
        let mut near = vec![0.0; 9];
        near[4] = 0.5;
        let mut far = vec![0.0; 9];
        far[3] = 1.0;

        // Ordered one way the answer is 0 ...
        let mut a = Hots::with_centers(g, 1, 1e-3, vec![near.clone(), far.clone()]).unwrap();
        assert_eq!(a.assign(ev).unwrap(), 0);
        // ... and ordered the other way it is 1, which a hardcoded `Ok(0)` cannot produce.
        let mut b = Hots::with_centers(g, 1, 1e-3, vec![far.clone(), near.clone()]).unwrap();
        assert_eq!(b.assign(ev).unwrap(), 1);

        // Nothing moved: `assign` is inference. Assigning again changes neither counter.
        assert_eq!(b.assign(PixelEvent { t_s: 1e-3, ..ev }).unwrap(), 1);
        assert_eq!(b.counts(), &[0, 0]);
        assert_eq!(b.centers()[1], near);
        // And `learn` on the same event DOES move it, so the comparison above is a real one.
        assert_eq!(b.learn(PixelEvent { t_s: 2e-3, ..ev }).unwrap(), 1);
        assert_eq!(b.counts(), &[0, 1]);
        assert_ne!(b.centers()[1], near);
        assert_eq!(b.centers()[0], far, "learning moved a prototype that did not win");
    }

    /// **[`PlaneFlow`] takes its neighbourhood from the event's own polarity plane, and no stream
    /// in the suite ever had two polarities in it.**
    ///
    /// Hardcoding the read to `Polarity::On` survived all 60 tests, and it can only show up on an
    /// `Off` event — whose neighbourhood would then be read off the `On` edge's timestamps. So the
    /// stimulus is two edges at once, in different directions, on opposite polarities. Each is
    /// exactly planar on its own plane, so each recovery is exact; reading the wrong plane costs a
    /// right angle.
    #[test]
    fn plane_flow_fits_on_the_events_own_polarity_plane() {
        let g = geom(64, 64);
        let (on_deg, on_speed) = (0.0f64, 300.0f64);
        let (off_deg, off_speed) = (117.0f64, 200.0f64);
        let make = |deg: f64, speed: f64, pol: Polarity| {
            let normal = deg * core::f64::consts::PI / 180.0;
            let (lo, hi) = edge_span(g, normal);
            moving_edge(g, normal, speed, lo - 2.0, (hi - lo + 4.0) / speed, pol).unwrap()
        };
        let mut evs = make(on_deg, on_speed, Polarity::On);
        evs.extend(make(off_deg, off_speed, Polarity::Off));
        evs.sort_by(|a, b| {
            a.t_s
                .partial_cmp(&b.t_s)
                .unwrap_or(core::cmp::Ordering::Equal)
                .then(a.y.cmp(&b.y))
                .then(a.x.cmp(&b.x))
        });
        assert!(evs.iter().filter(|e| e.polarity == Polarity::Off).count() > 1000);
        assert!(evs.iter().filter(|e| e.polarity == Polarity::On).count() > 1000);

        // The window has to hold the trailing half-neighbourhood of the SLOWER edge.
        let mut pf = PlaneFlow::new(g, 3, 4.0 * 3.0 / off_speed, None, 8).unwrap();
        let outs = pf.run(&evs).unwrap();
        let on_truth = moving_edge_flow(on_deg * core::f64::consts::PI / 180.0, on_speed);
        let off_truth = moving_edge_flow(off_deg * core::f64::consts::PI / 180.0, off_speed);

        for (pol, truth, name) in
            [(Polarity::On, on_truth, "On"), (Polarity::Off, off_truth, "Off")]
        {
            let e = flow_error(&evs, &outs, |ev| (ev.polarity == pol).then_some(truth)).unwrap();
            assert!(e.fitted * 2 > e.offered, "{name}: {} of {} fitted", e.fitted, e.offered);
            assert!(e.max_rel_speed_err < 1e-9, "{name}: speed error {:e}", e.max_rel_speed_err);
            assert!(e.mean_angle_err < 1e-9, "{name}: angular error {:e}", e.mean_angle_err);
        }

        // The two edges really are far apart, so the assertions above are not a coincidence:
        // scoring the Off events against the On edge's truth is wrong by more than a radian.
        let crossed =
            flow_error(&evs, &outs, |ev| (ev.polarity == Polarity::Off).then_some(on_truth))
                .unwrap();
        assert!(
            crossed.mean_angle_err > 1.0,
            "the two stimuli are not distinguishable: {:e}",
            crossed.mean_angle_err
        );
    }

    /// **[`EFast`] builds its `SAE` with `split: true` deliberately, and no test ever gave it an
    /// `Off` event.** Hardcoding `EFast::push`'s plane to `Polarity::On` survived all 60 tests.
    ///
    /// The fixture writes a quarter-circle entirely onto the `Off` plane, so the `On` plane holds
    /// nothing at all and the two answers are as far apart as they can be.
    #[test]
    fn efast_reads_the_circle_on_the_events_own_polarity_plane() {
        let g = geom(40, 40);
        let (cx, cy) = (20u16, 20u16);
        let mut d = EFast::new(g).unwrap();
        {
            let sae = d.surface_mut();
            for c in [&FAST_CIRCLE_16[..], &FAST_CIRCLE_20[..]] {
                for &(dx, dy) in c {
                    let x = (i64::from(cx) + dx) as u16;
                    let y = (i64::from(cy) + dy) as u16;
                    sae.update(PixelEvent { t_s: 0.0, x, y, polarity: Polarity::Off }).unwrap();
                }
            }
            for (c, k) in [(&FAST_CIRCLE_16[..], 4usize), (&FAST_CIRCLE_20[..], 5usize)] {
                for &(dx, dy) in c.iter().take(k) {
                    let x = (i64::from(cx) + dx) as u16;
                    let y = (i64::from(cy) + dy) as u16;
                    sae.update(PixelEvent { t_s: 1.0, x, y, polarity: Polarity::Off }).unwrap();
                }
            }
        }
        // Read on the plane the events were written to: a corner.
        assert_eq!(d.arcs_at(cx, cy, Polarity::Off).unwrap(), (Some(4), Some(5)));
        assert!(d.is_corner(cx, cy, Polarity::Off).unwrap());
        // Read on the other plane: nothing has ever fired there, every entry is infinitely old,
        // and no arc can be newer than its complement.
        assert_eq!(d.arcs_at(cx, cy, Polarity::On).unwrap(), (None, None));
        assert!(!d.is_corner(cx, cy, Polarity::On).unwrap());
        // And through `push`, which is where the plane is taken from the event rather than named.
        let mut off = d.clone();
        let hit = off.push(PixelEvent { t_s: 1.0, x: cx, y: cy, polarity: Polarity::Off }).unwrap();
        assert_eq!(hit.map(|c| c.score), Some(4.0), "an Off event did not read the Off plane");
        let mut on = d.clone();
        assert!(
            on.push(PixelEvent { t_s: 1.0, x: cx, y: cy, polarity: Polarity::On })
                .unwrap()
                .is_none(),
            "an On event read the Off plane's corner"
        );
    }

    /// **Mutation found: `reject_s`'s `<=` weakened to `<` survived every test.**
    ///
    /// The module pins the inclusive convention on both memory windows and on the `eHarris`
    /// binarisation window; the outlier threshold is the fourth comparison of the same class and
    /// it had neither a test nor a doc sentence. It is the same defect the module's own docs
    /// describe as "two implementations of the same paper disagreeing on a recording whose events
    /// land on round microseconds".
    ///
    /// # Why the boundary here is exact
    ///
    /// Eight ring pixels at `t = 0` and the centre at `t = 9 / 1024` s. The support is symmetric
    /// in `(x, y)`, so both gradient sums vanish **exactly** and the first-pass fit is the flat
    /// plane `t = c` with `c = (9 / 1024) / 9 = 1 / 1024`. Residuals are therefore exactly
    /// `1 / 1024` on the ring and `8 / 1024` at the centre — all powers of two, so no rounding
    /// enters — and setting `reject_s` to exactly `1 / 1024` decides the whole ring at once.
    #[test]
    fn the_outlier_rejection_threshold_is_inclusive_at_its_boundary() {
        let g = geom(32, 32);
        let t_centre = 9.0 / 1024.0;
        let ring_residual = 1.0 / 1024.0;
        assert_eq!(t_centre / 9.0, ring_residual, "the fixture's arithmetic is not exact");
        // `min_events` is nine, the whole neighbourhood, so the outcome REPORTS the surviving
        // count: the boundary decides eight points at once and the number comes back in the
        // `RejectedTooMany` payload rather than having to be inferred from a fit.
        let run = |reject_s: Option<f64>| {
            let mut pf = PlaneFlow::new(g, 1, 0.05, reject_s, 9).unwrap();
            for dy in -1i64..=1 {
                for dx in -1i64..=1 {
                    if (dx, dy) == (0, 0) {
                        continue;
                    }
                    pf.push(PixelEvent {
                        t_s: 0.0,
                        x: (5 + dx) as u16,
                        y: (5 + dy) as u16,
                        polarity: Polarity::On,
                    })
                    .unwrap();
                }
            }
            pf.push(PixelEvent { t_s: t_centre, x: 5, y: 5, polarity: Polarity::On }).unwrap()
        };
        // EXACTLY at the threshold the eight ring points are KEPT and only the centre is rejected.
        // Under `<` the same call keeps nothing at all, so this one assertion is the whole test.
        let at = run(Some(ring_residual));
        assert!(
            matches!(at, FlowOutcome::RejectedTooMany { inliers: 8 }),
            "at the boundary: {at:?}"
        );
        // A hair above, the same eight.
        let above = run(Some(ring_residual * 1.0001));
        assert!(
            matches!(above, FlowOutcome::RejectedTooMany { inliers: 8 }),
            "above the boundary: {above:?}"
        );
        // A hair below, the ring goes too and nothing survives — which is what says the eight
        // above are the boundary's doing and not simply everything that was there.
        let below = run(Some(ring_residual * 0.9999));
        assert!(
            matches!(below, FlowOutcome::RejectedTooMany { inliers: 0 }),
            "below the boundary: {below:?}"
        );
        // Wide enough to reject nothing, all nine survive and the flat fit is reported as
        // `NoMotion` — so the outcomes above are the rejection pass and not the support's shape.
        let wide = run(Some(0.05));
        assert!(matches!(wide, FlowOutcome::NoMotion), "with nothing rejected: {wide:?}");
        // And with the pass disabled entirely, the same answer, which is the control.
        assert!(matches!(run(None), FlowOutcome::NoMotion), "{:?}", run(None));
    }

    /// **Mutation found: `.round()` weakened to `.floor()` survived every test.**
    ///
    /// `quantisation_rounds_the_clock_without_losing_an_event` asserts only that the timestamps
    /// became whole microseconds and that more than half of them moved, and truncation satisfies
    /// both. But the module's stated error bound — "a timestamp rounded to the nearest microsecond
    /// carries an error up to 0.5 us", which is what predicts the 1.5e-4 relative gradient error
    /// at 900 px/s — is a claim about round-to-nearest. Under truncation the quantum is one-sided
    /// with a systematic bias and the prediction roughly doubles, while the measured 3.1e-4
    /// against a 1e-3 assertion would not notice.
    ///
    /// So the rule is pinned at four points chosen to separate `round` from `floor`, from `trunc`,
    /// from `ceil` and from round-half-to-even.
    #[test]
    fn quantisation_rounds_to_nearest_rather_than_truncating() {
        let at = |t_s: f64| {
            quantise_microseconds(&[PixelEvent { t_s, x: 0, y: 0, polarity: Polarity::On }])
                .unwrap()[0]
                .t_s
        };
        // 2.7 us goes UP. `floor` and `trunc` would give 2.
        assert!((at(2.7e-6) - 3e-6).abs() < 1e-15, "2.7 us became {} s", at(2.7e-6));
        // 2.2 us goes DOWN. `ceil` would give 3.
        assert!((at(2.2e-6) - 2e-6).abs() < 1e-15, "2.2 us became {} s", at(2.2e-6));
        // An exact half goes AWAY FROM ZERO, which is what `f64::round` documents. Round-half-to-
        // even would send 0.5 to 0 and 2.5 to 2; both products below are exact in binary64.
        assert!((at(0.5e-6) - 1e-6).abs() < 1e-15, "0.5 us became {} s", at(0.5e-6));
        assert!((at(2.5e-6) - 3e-6).abs() < 1e-15, "2.5 us became {} s", at(2.5e-6));
        // Zero is a fixed point rather than a special case, and so is a whole microsecond.
        assert_eq!(at(0.0), 0.0);
        assert!((at(4e-6) - 4e-6).abs() < 1e-18);
        // The error a round introduces is at most half a quantum, which is the claim the module's
        // quantised-recovery bound is derived from. Truncation would break this at 0.5 us.
        for k in 0..200 {
            let t = f64::from(k) * 0.37e-6;
            assert!((at(t) - t).abs() <= 0.5e-6 + 1e-18, "{t} moved to {}", at(t));
        }
    }

    /// **Mutation found: `for len in lo..=hi` reversed to `(lo..=hi).rev()` — the LONGEST valid
    /// arc instead of the shortest — survived every test.**
    ///
    /// Every arc fixture in this module is two-level by construction, one shared old timestamp and
    /// one shared new one, so exactly one length is ever valid and the choice is invisible. A real
    /// `SAE` has three or more distinct levels at every pixel, and then the two answers genuinely
    /// differ — and the returned length is what is compared against `inner_arc` and `outer_arc`, so
    /// it decides detections on real data.
    #[test]
    fn the_newest_arc_returned_is_the_shortest_valid_one() {
        // Three levels: 3, 3, 2, then five zeros.
        let t = [3.0f64, 3.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        // The shortest valid arc is {3, 3}: strictly newer than max(2, 0, ..) = 2.
        assert_eq!(newest_arc(&t, 1, 7), Some(2));
        // Length 3 is ALSO valid — {3, 3, 2} beats max(0, ..) = 0 — so the search is choosing
        // between two right answers rather than finding the only one.
        assert_eq!(newest_arc(&t, 3, 3), Some(3));
        // Length 1 is not: no single entry is strictly newer than all seven others.
        assert_eq!(newest_arc(&t, 1, 1), None);
        // Asked for a range that starts above the shortest answer, it returns the shortest one IN
        // the range.
        assert_eq!(newest_arc(&t, 2, 7), Some(2));

        // The same three-level structure across the seam, so the preference survives wrapping.
        let w = [2.0f64, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 3.0];
        assert_eq!(newest_arc(&w, 1, 7), Some(2));
        assert_eq!(newest_arc(&w, 3, 3), Some(3));
    }

    /// **The paper's own sensor is the untested case.**
    ///
    /// `cell_grid`'s two `div_ceil` calls had one assertion on them — a 20x20 sensor with
    /// `cell_px = 10`, where floor and ceil agree — so replacing both with `/ k` survived all 60
    /// tests. [`Hats::n_cars`] is the `N-CARS`/`ATIS` configuration and that sensor is **304x240**:
    /// 304 / 10 is 30.4, and under floor division the right-hand column of events writes into the
    /// *next row's* cell and, in the bottom-right corner, past the end of `per_cell`.
    #[test]
    fn hats_tiles_a_sensor_whose_width_is_not_a_multiple_of_the_cell() {
        let h = Hats::n_cars();
        let atis = geom(304, 240);
        // ceil(304 / 10) = 31 and ceil(240 / 10) = 24, not 30 and 24.
        assert_eq!(h.cell_grid(atis).unwrap(), (31, 24));
        assert_eq!(h.descriptor_len(atis).unwrap(), 31 * 24 * 2 * 49);
        assert_eq!(h.descriptor_len(atis).unwrap(), 72912);
        // A remainder in each direction, and the exactly-divisible case beside them.
        assert_eq!(h.cell_grid(geom(25, 25)).unwrap(), (3, 3));
        assert_eq!(h.cell_grid(geom(30, 30)).unwrap(), (3, 3));
        assert_eq!(h.cell_grid(geom(31, 30)).unwrap(), (4, 3));
        assert_eq!(h.cell_grid(geom(30, 31)).unwrap(), (3, 4));
        assert_eq!(h.cell_grid(geom(1, 1)).unwrap(), (1, 1));

        // And the last column really is addressed by the last cell: an event at (303, 239) lands
        // in cell (23, 30) of a 31-wide grid, inside the descriptor rather than past its end.
        let last = [PixelEvent { t_s: 0.0, x: 303, y: 239, polarity: Polarity::On }];
        let d = h.descriptor(atis, &last).unwrap();
        assert_eq!(d.len(), 72912);
        let cell = 23 * 31 + 30;
        let centre = (cell * 2 + 1) * 49 + 3 * 7 + 3;
        assert_eq!(d[centre], 1.0, "the bottom-right event did not reach the bottom-right cell");
        assert_eq!(d.iter().filter(|v| **v != 0.0).count(), 1);
        // The cell to its left is a different cell, which is what "30.4 cells" has to mean.
        let left = [PixelEvent { t_s: 0.0, x: 299, y: 239, polarity: Polarity::On }];
        let dl = h.descriptor(atis, &left).unwrap();
        let centre_left = ((23 * 31 + 29) * 2 + 1) * 49 + 3 * 7 + 3;
        assert_eq!(dl[centre_left], 1.0);
        assert_eq!(dl[centre], 0.0, "x = 299 and x = 303 landed in the same cell");
    }

    /// **Mutation found: the neighbourhood test `|dx| > rho || |dy| > rho` widened to `>=`
    /// survived every test.**
    ///
    /// That mutation permanently zeroes every bin on the *edge* of the neighbourhood: for the
    /// paper's `rho = 3` that is 24 of the 49 bins per cell, roughly half the descriptor. It
    /// survived because no test ever placed a neighbour further than one pixel from the event —
    /// the radius-3 tests use offsets of 0 and +/-1, and the radius-1 tests use offset 0 only.
    #[test]
    fn hats_fills_the_outer_ring_of_its_neighbourhood() {
        let g = geom(20, 20);
        let h = Hats { cell_px: 10, radius: 3, tau_s: 0.05, window_s: 1.0, split_polarity: false };
        let dt = 0.02;
        let decay = (-dt / h.tau_s).exp();
        let centre_bin = 3 * 7 + 3;
        // Every pixel named below is inside cell 0, which spans x and y in 0..=9.
        let pair = |x0: u16, y0: u16, x1: u16, y1: u16| {
            h.descriptor(g, &[
                PixelEvent { t_s: 0.0, x: x0, y: y0, polarity: Polarity::On },
                PixelEvent { t_s: dt, x: x1, y: y1, polarity: Polarity::On },
            ])
            .unwrap()
        };
        // Straight out along -x, exactly on the ring: offset (-3, 0), bin (0 + 3) * 7 + 0 = 21.
        let side = pair(1, 4, 4, 4);
        assert!((side[21] - decay / 2.0).abs() < 1e-14, "the (-3, 0) bin holds {}", side[21]);
        assert_eq!(side[centre_bin], 1.0);
        assert_eq!(side.iter().filter(|v| **v != 0.0).count(), 2);
        // The near corner of the ring, offset (-3, -3): bin 0, the very first entry of the cell.
        let corner = pair(1, 1, 4, 4);
        assert!((corner[0] - decay / 2.0).abs() < 1e-14, "the (-3, -3) bin holds {}", corner[0]);
        // The far corner, offset (+3, +3): bin 6 * 7 + 6 = 48, the very last.
        let opposite = pair(7, 7, 4, 4);
        assert!(
            (opposite[48] - decay / 2.0).abs() < 1e-14,
            "the (+3, +3) bin holds {}",
            opposite[48]
        );
        // Straight down, offset (0, +3): bin 6 * 7 + 3 = 45.
        let down = pair(4, 7, 4, 4);
        assert!((down[45] - decay / 2.0).abs() < 1e-14, "the (0, +3) bin holds {}", down[45]);
        // And one pixel further out is OUTSIDE the neighbourhood and contributes nothing, which is
        // what stops this test from passing for a detector that simply counts everything.
        let outside = pair(0, 4, 4, 4);
        assert_eq!(outside[centre_bin], 1.0);
        assert_eq!(
            outside.iter().filter(|v| **v != 0.0).count(),
            1,
            "a neighbour at |dx| = 4 was counted"
        );
    }

    /// **The `HATS` divisor is the cell's TOTAL event count, not the plane's, and the consequence
    /// is asserted rather than only described.**
    ///
    /// An `On`-dominated cell and a balanced cell with **identical `On` texture** produce different
    /// `On` planes. That follows from the averaging step as this implementation reads it, it could
    /// not be checked against an author-released reference, and it is now flagged on
    /// [`Hats::descriptor`] for the same reason the `tau` reading is flagged on the type. With a
    /// per-plane divisor every number below would be 1.0.
    #[test]
    fn hats_normalises_a_polarity_plane_by_the_cells_total() {
        let g = geom(10, 10);
        let h = Hats { cell_px: 10, radius: 1, tau_s: 1.0, window_s: 1.0, split_polarity: true };
        // Plane 1 is `On`, nine bins per plane, centre of a 3x3 neighbourhood is index 4.
        let on_centre = 9 + 4;
        let on = PixelEvent { t_s: 0.0, x: 5, y: 5, polarity: Polarity::On };
        let off = |t: f64, x: u16, y: u16| PixelEvent { t_s: t, x, y, polarity: Polarity::Off };

        // One `On` event alone in the cell: its own contribution over a divisor of one.
        assert_eq!(h.descriptor(g, &[on]).unwrap()[on_centre], 1.0);
        // The SAME `On` texture with one `Off` event added elsewhere in the cell. The `On` plane's
        // content has not changed; its divisor has.
        let mixed = h.descriptor(g, &[on, off(0.01, 8, 8)]).unwrap();
        assert_eq!(mixed[on_centre], 0.5, "the On plane was divided by its own count");
        // Three `Off` events, and the same `On` plane is divided by four.
        let mixed3 = h
            .descriptor(g, &[on, off(0.01, 8, 8), off(0.02, 8, 0), off(0.03, 0, 8)])
            .unwrap();
        assert_eq!(mixed3[on_centre], 0.25);
        // The `Off` plane of that last descriptor carries three own-contributions over four.
        assert_eq!(mixed3[4], 0.75);
    }

    /// **Every `# Errors` branch below had no test at all.** Each of these mutations survived all
    /// 60 tests: replacing `Motion::validate` with `Ok(())`; dropping `require_time_ordered`'s
    /// finiteness check; dropping `warped_image`'s pre-warp bounds check; dropping either of
    /// `EFast::with_arcs`'s guards, which were dead under test because no test ever called it with
    /// a custom range; and dropping `Motion::RadialExpansion`'s `r == 0` guard, which no test could
    /// reach because every expansion centre in this module is half-integer.
    #[test]
    fn the_refusals_the_error_docs_name_are_reachable() {
        let g = geom(16, 16);
        let ev = [PixelEvent { t_s: 0.0, x: 4, y: 4, polarity: Polarity::On }];
        let still = Motion::Translation { vx: 0.0, vy: 0.0 };

        // `Motion::validate`, on its own and through the two public entry points that call it.
        for m in [
            Motion::Translation { vx: f64::NAN, vy: 0.0 },
            Motion::Translation { vx: 0.0, vy: f64::INFINITY },
            Motion::Rotation { omega_rad_s: f64::NAN, cx: 0.0, cy: 0.0 },
            Motion::Rotation { omega_rad_s: 1.0, cx: f64::NAN, cy: 0.0 },
            Motion::Rotation { omega_rad_s: 1.0, cx: 0.0, cy: f64::NEG_INFINITY },
            Motion::RadialExpansion { rate_px_s: f64::NAN, cx: 0.0, cy: 0.0 },
            Motion::RadialExpansion { rate_px_s: 1.0, cx: f64::NAN, cy: 0.0 },
            Motion::RadialExpansion { rate_px_s: 1.0, cx: 0.0, cy: f64::NAN },
        ] {
            assert!(matches!(m.validate(), Err(VisionError::NonFinite { .. })), "{m:?} validated");
            assert!(
                matches!(warped_image(g, &ev, 0.0, m), Err(VisionError::NonFinite { .. })),
                "{m:?} warped"
            );
            assert!(matches!(
                contrast(g, &ev, 0.0, m, Objective::Variance),
                Err(VisionError::NonFinite { .. })
            ));
        }
        // Finite motions of all three families pass, so the loop is not refusing everything.
        for m in [
            still,
            Motion::Rotation { omega_rad_s: 1.0, cx: 8.0, cy: 8.0 },
            Motion::RadialExpansion { rate_px_s: 1.0, cx: 8.0, cy: 8.0 },
        ] {
            assert!(m.validate().is_ok(), "{m:?} was refused");
        }

        // `warped_image`'s PRE-warp bounds check: an event off the lattice is an error ...
        let off_lattice = [PixelEvent { t_s: 0.0, x: 16, y: 0, polarity: Polarity::On }];
        assert!(matches!(
            warped_image(g, &off_lattice, 0.0, still),
            Err(VisionError::OutOfBounds { .. })
        ));
        // ... while an event that warps off the lattice is a `dropped`, which is the distinction
        // the two branches exist to keep.
        let far = warped_image(g, &ev, 1.0, Motion::Translation { vx: 1e6, vy: 0.0 }).unwrap();
        assert_eq!((far.placed, far.dropped), (0, 1));
        // And a bad timestamp or reference time is refused before either.
        let nan_t = [PixelEvent { t_s: f64::NAN, x: 4, y: 4, polarity: Polarity::On }];
        assert!(matches!(
            warped_image(g, &nan_t, 0.0, still),
            Err(VisionError::NonFinite { .. })
        ));
        assert!(matches!(
            warped_image(g, &ev, f64::NAN, still),
            Err(VisionError::NonFinite { .. })
        ));

        // `require_time_ordered`, both halves, directly and through `Hats::descriptor` — where
        // without the finiteness check a single NaN silently poisons the whole descriptor.
        let nan_stream = [ev[0], PixelEvent { t_s: f64::NAN, x: 5, y: 4, polarity: Polarity::On }];
        assert!(matches!(
            require_time_ordered(&nan_stream),
            Err(VisionError::NonFinite { .. })
        ));
        assert!(matches!(
            Hats::n_cars().descriptor(g, &nan_stream),
            Err(VisionError::NonFinite { .. })
        ));
        let backwards = [
            PixelEvent { t_s: 1.0, x: 4, y: 4, polarity: Polarity::On },
            PixelEvent { t_s: 0.5, x: 5, y: 4, polarity: Polarity::On },
        ];
        assert!(matches!(
            require_time_ordered(&backwards),
            Err(VisionError::OutOfOrder { .. })
        ));
        assert!(require_time_ordered(&ev).is_ok());
        assert!(require_time_ordered(&[]).is_ok());

        // `EFast::with_arcs`: both `BadParameters` branches, in both directions.
        for arcs in [(0usize, 6usize), (7, 3), (3, 16), (3, 20)] {
            assert!(
                matches!(
                    EFast::with_arcs(g, arcs, EFast::OUTER_ARC),
                    Err(VisionError::BadParameters { .. })
                ),
                "inner range {arcs:?} was accepted"
            );
        }
        for arcs in [(0usize, 8usize), (9, 4), (4, 20), (4, 25)] {
            assert!(
                matches!(
                    EFast::with_arcs(g, EFast::INNER_ARC, arcs),
                    Err(VisionError::BadParameters { .. })
                ),
                "outer range {arcs:?} was accepted"
            );
        }
        // A legal custom range builds, so neither loop is refusing everything.
        assert!(EFast::with_arcs(g, (2, 5), (3, 7)).is_ok());
        // And the paper's ranges, as literals: 3 to 6 of 16 and 4 to 8 of 20.
        assert_eq!(EFast::INNER_ARC, (3, 6));
        assert_eq!(EFast::OUTER_ARC, (4, 8));

        // `Motion::RadialExpansion`'s centre guard. Without it `k = (r - rate * dt) / r` is 0/0 at
        // the centre, the event warps to NaN, and it is silently counted as `dropped`.
        let m = Motion::RadialExpansion { rate_px_s: 50.0, cx: 4.0, cy: 4.0 };
        assert_eq!(m.warp(4.0, 4.0, 0.01), (4.0, 4.0));
        assert_eq!(m.warp(4.0, 4.0, -0.01), (4.0, 4.0));
        let at_centre = warped_image(g, &ev, 0.01, m).unwrap();
        assert_eq!((at_centre.placed, at_centre.dropped), (1, 0));
        assert_eq!(at_centre.frame.at(4, 4), Some(1.0));
        // A pixel one away from the centre DOES move, so the guard is a special case of something.
        let (wx, wy) = m.warp(5.0, 4.0, -0.01);
        assert!((wx - 5.5).abs() < 1e-12 && (wy - 4.0).abs() < 1e-12, "({wx}, {wy})");
    }

    /// **Six public accessors that no test called.** Each could be replaced by a wrong constant and
    /// all 60 tests still passed; [`Decay::scale_s`] is not called anywhere in the crate at all.
    ///
    /// Each is checked against the value its owner was *built* with, and where the accessor names
    /// a quantity with a definition, against the definition too — the exponential is exactly `1/e`
    /// at `scale_s`, the linear ramp exactly zero there.
    #[test]
    fn the_accessors_report_what_their_owners_were_built_with() {
        assert_eq!(Decay::Exponential { tau_s: 0.007 }.scale_s(), 0.007);
        assert_eq!(Decay::Linear { window_s: 0.13 }.scale_s(), 0.13);
        let e = Decay::Exponential { tau_s: 0.007 };
        assert!((e.value(e.scale_s()) - (-1.0f64).exp()).abs() < 1e-15);
        let l = Decay::Linear { window_s: 0.13 };
        assert_eq!(l.value(l.scale_s()), 0.0);
        assert!(l.value(l.scale_s() / 2.0) > 0.0);

        let g = geom(12, 9);
        let decay = Decay::Exponential { tau_s: 0.004 };
        let mut ts = TimeSurface::new(g, decay, true).unwrap();
        assert_eq!(ts.decay(), decay);
        assert_eq!(ts.decay().scale_s(), 0.004);
        assert!(ts.is_split());
        assert!(!TimeSurface::new(g, decay, false).unwrap().is_split());
        // `now_s` is -inf before the first event, the last accepted time afterwards, and -inf
        // again after `clear`.
        assert_eq!(ts.now_s(), f64::NEG_INFINITY);
        ts.update(PixelEvent { t_s: 0.25, x: 1, y: 1, polarity: Polarity::On }).unwrap();
        assert_eq!(ts.now_s(), 0.25);
        ts.update(PixelEvent { t_s: 0.75, x: 2, y: 1, polarity: Polarity::Off }).unwrap();
        assert_eq!(ts.now_s(), 0.75);
        ts.clear();
        assert_eq!(ts.now_s(), f64::NEG_INFINITY);

        let pf = PlaneFlow::new(g, 4, 0.021, Some(1e-3), 9).unwrap();
        assert_eq!(pf.radius(), 4);
        assert_eq!(pf.window_s(), 0.021);
        // The surface it keeps is the LINEAR one its own window defines, and it is split — which
        // is the wiring between the accessors rather than each one on its own.
        assert_eq!(pf.surface().decay(), Decay::Linear { window_s: 0.021 });
        assert!(pf.surface().is_split(), "the plane fit's SAE must keep the polarities apart");
        assert_eq!(pf.surface().geometry(), g);
        // A second estimator with different numbers, so neither reading is a coincidence.
        let pf2 = PlaneFlow::new(g, 2, 0.05, None, 3).unwrap();
        assert_eq!((pf2.radius(), pf2.window_s()), (2, 0.05));

        let eh = EHarris::new(g, 3, 0.011, 0.05, 1.5).unwrap();
        assert_eq!((eh.radius(), eh.window_s()), (3, 0.011));
        assert_eq!(eh.surface().decay(), Decay::Linear { window_s: 0.011 });
        assert!(!eh.surface().is_split(), "eHarris binarises without regard to polarity");
        assert!(EFast::new(g).unwrap().surface().is_split());

        let hots = Hots::with_centers(g, 2, 0.003, vec![vec![0.25; 25]]).unwrap();
        assert_eq!(hots.radius(), 2);
        assert_eq!(hots.centers().len(), 1);
        assert_eq!(hots.counts(), &[0]);
        assert_eq!(hots.surface().decay(), Decay::Exponential { tau_s: 0.003 });
    }

    /// **A panic reachable from the public API on an input the constructor accepted.**
    ///
    /// `EHarris::new(geom(64, 64), 32768, ..)` used to succeed, and the next `push` panicked at
    /// `(2 * self.radius + 1) as usize` — that is `u16` arithmetic, so `2 * 32768` overflows in
    /// debug and wraps to `0` in release, after which a length-1 buffer is indexed at 65536.
    /// `TimeSurface::patch(4, 4, 32768, ..)` was the same defect with a caller-supplied radius and
    /// no validation at all, and `Hots::learn` reached it through the same call. All three were
    /// reproduced before this test was written.
    ///
    /// The arithmetic is `usize` now, and every constructor that stores a radius states a bound.
    #[test]
    fn a_neighbourhood_wider_than_the_sensor_is_refused_rather_than_overflowing() {
        let g = geom(64, 64);
        assert!(matches!(
            EHarris::new(g, 32768, 1e-3, EHarris::K_HARRIS, 0.0),
            Err(VisionError::BadParameters { .. })
        ));
        let mut ts = TimeSurface::new(g, Decay::Exponential { tau_s: 1e-3 }, true).unwrap();
        ts.update(PixelEvent { t_s: 0.0, x: 4, y: 4, polarity: Polarity::On }).unwrap();
        assert!(matches!(
            ts.patch(4, 4, 32768, Polarity::On, 0.0),
            Err(VisionError::BadParameters { .. })
        ));
        assert!(matches!(
            Hots::with_centers(g, 32768, 1e-3, vec![vec![0.0; 9]]),
            Err(VisionError::BadParameters { .. })
        ));
        assert!(matches!(
            Hots::new(g, 32768, 1e-3, 2, &mut Rng::new(1)),
            Err(VisionError::BadParameters { .. })
        ));
        assert!(matches!(
            PlaneFlow::new(g, 32768, 1e-2, None, 8),
            Err(VisionError::BadParameters { .. })
        ));

        // The bound is exactly the sensor's larger dimension, not a round number someone picked:
        // one below it works and gives a full-size patch, and at it the call is refused.
        let tall = geom(8, 20);
        let mut ts = TimeSurface::new(tall, Decay::Linear { window_s: 1.0 }, false).unwrap();
        ts.update(PixelEvent { t_s: 0.0, x: 3, y: 3, polarity: Polarity::On }).unwrap();
        let p = ts.patch(3, 3, 19, Polarity::On, 0.0).unwrap();
        assert_eq!(p.len(), 39 * 39);
        // At that radius the patch already covers every pixel of the sensor from any centre, which
        // is the whole reason the bound is where it is: one pixel has fired and the rest is pad.
        assert_eq!(p.iter().filter(|v| **v != 0.0).count(), 1);
        assert!(matches!(
            ts.patch(3, 3, 20, Polarity::On, 0.0),
            Err(VisionError::BadParameters { .. })
        ));
        // And the ordinary radii the rest of this module uses are untouched.
        assert!(ts.patch(3, 3, 2, Polarity::On, 0.0).is_ok());
        assert!(EHarris::new(g, 63, 1e-3, EHarris::K_HARRIS, 0.0).is_ok());
        assert!(EHarris::new(g, 64, 1e-3, EHarris::K_HARRIS, 0.0).is_err());
        assert!(PlaneFlow::new(g, 63, 1e-2, None, 8).is_ok());
        assert!(PlaneFlow::new(g, 64, 1e-2, None, 8).is_err());

        // `Hats` is the fourth member of the family and reaches the same hazard by a different
        // road: its fields are public, so `(2 * radius + 1)^2` can be made to overflow a 32-bit
        // `usize`, which on a release wasm32 build would wrap silently and then index out of
        // bounds. `bins_per_cell` saturates and `descriptor_len` refuses.
        let huge =
            Hats { cell_px: 1, radius: 65535, tau_s: 1.0, window_s: 0.1, split_polarity: true };
        let wide_sensor = geom(65535, 65535);
        assert!(matches!(
            huge.descriptor_len(wide_sensor),
            Err(VisionError::BadParameters { .. })
        ));
        assert!(matches!(
            huge.descriptor(wide_sensor, &[]),
            Err(VisionError::BadParameters { .. })
        ));
        // `bins_per_cell` itself saturates rather than wrapping, checked in `u128` so the
        // assertion is the same arithmetic on a 64-bit host and on `wasm32`.
        let side = 2u128 * 65535 + 1;
        assert_eq!(huge.bins_per_cell() as u128, (side * side).min(usize::MAX as u128));
        // The paper's own configuration is nowhere near any of this.
        assert_eq!(Hats::n_cars().bins_per_cell(), 49);
        assert_eq!(Hats::n_cars().descriptor_len(geom(304, 240)).unwrap(), 72912);
    }

    /// **An unbounded loop reachable from the public API.**
    ///
    /// `rotating_bar` emitted one event per annulus pixel per half turn with nothing bounding the
    /// count. Measured on a 3x3 sensor with four annulus pixels, `omega` of 1e3 to 1e6 gave 2548,
    /// 25466, 254648 and 2546480 events — linear, with no ceiling — and `omega = 1e12` over one
    /// second asks for about 1.3e12, roughly 30 TB. Worse, once `pi / omega` falls below
    /// `duration / 2^53` the crossing counter `k` stops advancing in binary64 and the loop
    /// **never terminates**: reproduced at `omega = 1e17`, which ran until it was killed.
    ///
    /// The count is computed first and refused against [`MAX_ROTATING_BAR_EVENTS`].
    #[test]
    fn rotating_bar_refuses_a_sweep_it_cannot_finish() {
        let g = geom(3, 3);
        let bar = |omega: f64, duration: f64| {
            rotating_bar(g, (1.0, 1.0), omega, 0.0, 0.5, 1.5, duration, Polarity::On)
        };
        // The counts measured before the fix, unchanged: the cap refuses, it does not truncate.
        assert_eq!(bar(1e3, 1.0).unwrap().len(), 2548);
        assert_eq!(bar(1e4, 1.0).unwrap().len(), 25466);
        assert_eq!(bar(1e5, 1.0).unwrap().len(), 254648);
        // The cap bites exactly where it says, and this pair is deliberately the FIRST refusal
        // asserted: eight annulus pixels at 127324 crossings each is 1018592 events and is
        // produced, and 3183099 each is 25464792 and is refused. A disabled cap returns `Ok` here,
        // in about half a second, which is what catches the mutation before anything downstream
        // asks for the 30 TB the 1e12 case below would need.
        assert_eq!(bar(4e6, 0.1).unwrap().len(), 1018592);
        assert!(matches!(bar(1e7, 1.0), Err(VisionError::BadParameters { .. })));
        // Far past the cap: a named refusal instead of an allocation nobody has the memory for.
        assert!(matches!(bar(1e12, 1.0), Err(VisionError::BadParameters { .. })));
        // The case that used to spin forever now returns, in bounded time.
        assert!(matches!(bar(1e17, 1.0), Err(VisionError::BadParameters { .. })));
        assert!(matches!(bar(f64::MAX, 1.0), Err(VisionError::BadParameters { .. })));
        // The cap is on the EVENT count, not on omega: the same omega over a short enough
        // recording is fine, and a modest omega over a large enough sensor is refused.
        assert!(bar(1e6, 1e-3).is_ok());
        assert!(matches!(
            rotating_bar(geom(512, 512), (256.0, 256.0), 1e5, 0.0, 1.0, 400.0, 1.0, Polarity::On),
            Err(VisionError::BadParameters { .. })
        ));
        // Right at the boundary the answer is still exact rather than clipped: four annulus pixels
        // at one crossing each.
        let one_turn = bar(1.0, 1e-9).unwrap();
        assert!(one_turn.len() <= 4, "{} events in a billionth of a second", one_turn.len());
        // The cap, as its doc states it.
        assert_eq!(MAX_ROTATING_BAR_EVENTS, 10_000_000);
    }

    /// **A panic on a [`Frame`] the type lets you build.**
    ///
    /// All three fields are public — this module's own tests write images down as literals, which
    /// is what makes their statistics checkable against arithmetic — so
    /// `Frame { width: 4, height: 4, data: Vec::new() }` is constructible, and `at(0, 0)` on it
    /// panicked with "index out of bounds", from a method documented as returning `None` off the
    /// lattice. Reproduced before this test was written.
    #[test]
    fn a_frame_whose_data_is_short_reports_none_rather_than_panicking() {
        let empty = Frame { width: 4, height: 4, data: Vec::new() };
        assert!(!empty.is_consistent());
        assert_eq!(empty.at(0, 0), None);
        assert_eq!(empty.at(3, 3), None);
        assert_eq!(empty.at(4, 0), None);
        assert_eq!(empty.sum(), 0.0);
        assert_eq!(empty.mean(), None);
        assert_eq!(empty.variance(), None);
        assert_eq!(empty.max(), None);
        assert_eq!(empty.min(), None);

        // A partial image: the pixels that exist read back, the rest are `None` rather than a panic
        // or a zero somebody might plot.
        let short = Frame { width: 4, height: 4, data: vec![1.0, 2.0, 3.0] };
        assert!(!short.is_consistent());
        assert_eq!(short.at(0, 0), Some(1.0));
        assert_eq!(short.at(2, 0), Some(3.0));
        assert_eq!(short.at(3, 0), None);
        assert_eq!(short.at(0, 1), None);

        // And everything this module PRODUCES is consistent, which is the invariant the `None` is
        // standing in for.
        let g = geom(7, 5);
        assert!(Frame::zeros(g).is_consistent());
        let evs = moving_edge(g, 0.0, 100.0, -2.0, 0.2, Polarity::On).unwrap();
        assert!(!evs.is_empty());
        assert!(accumulate_count(g, &evs).unwrap().is_consistent());
        assert!(accumulate_polarity(g, &evs).unwrap().is_consistent());
        assert!(
            accumulate_decay(g, &evs, Decay::Exponential { tau_s: 0.01 }, 0.2)
                .unwrap()
                .is_consistent()
        );
        let ts = TimeSurface::new(g, Decay::Exponential { tau_s: 0.01 }, false).unwrap();
        assert!(ts.render(Polarity::On, 0.0).unwrap().is_consistent());
        assert!(
            warped_image(g, &evs, 0.1, Motion::Translation { vx: 100.0, vy: 0.0 })
                .unwrap()
                .frame
                .is_consistent()
        );
    }

    /// **The tie-break `sort_events` documents, against a stream that has ties in two dimensions.**
    ///
    /// Its doc says ties break by `(y, x)` and that "fixing the order here is what makes a run
    /// reproducible", but `the_whole_pipeline_is_bit_reproducible` only compares a run against
    /// itself, so swapping the tie-break to `(x, y)` survived every test. An axis-aligned edge
    /// cannot see the difference either: all its ties share one coordinate, and the two orders
    /// agree there. Microsecond quantisation on a diagonal edge is what creates ties across
    /// genuinely different pixels.
    #[test]
    fn the_stream_order_is_the_one_sort_events_documents() {
        let g = geom(48, 48);
        let evs = quantise_microseconds(
            &moving_edge(g, 0.7, 1_000_000.0, -5.0, 1e-4, Polarity::On).unwrap(),
        )
        .unwrap();
        assert_eq!(evs.len(), g.pixels(), "the edge did not sweep the whole sensor");
        let mut two_dimensional_ties = 0usize;
        for w in evs.windows(2) {
            let (a, b) = (w[0], w[1]);
            assert!(a.t_s <= b.t_s, "the stream is not sorted by time");
            if a.t_s == b.t_s {
                assert!(
                    (a.y, a.x) < (b.y, b.x),
                    "a tie put ({}, {}) before ({}, {})",
                    a.y,
                    a.x,
                    b.y,
                    b.x
                );
                if a.x != b.x && a.y != b.y {
                    two_dimensional_ties += 1;
                }
            }
        }
        // Without ties that move BOTH coordinates the assertion above cannot tell `(y, x)` from
        // `(x, y)`, so the stimulus is required to produce them.
        assert!(
            two_dimensional_ties > 50,
            "only {two_dimensional_ties} ties could see the tie-break at all"
        );
    }

    /// **`warped_image`'s doc says "Polarity is **ignored** — every event votes `+1`", with a
    /// paragraph justifying unsigned voting, and no mixed-polarity stream was ever warped.**
    ///
    /// Making the warp itself polarity-dependent — `m.warp(x, y, (t - t_ref) * e.polarity.sign())`
    /// — survived all 60 tests.
    #[test]
    fn warping_ignores_the_polarity_of_the_events_it_warps() {
        let g = geom(32, 32);
        let mixed: Vec<PixelEvent> = (0..24u16)
            .map(|k| PixelEvent {
                t_s: f64::from(k) * 1e-3,
                x: 4 + k,
                y: 8 + k / 2,
                polarity: if k % 3 == 0 { Polarity::Off } else { Polarity::On },
            })
            .collect();
        assert!(mixed.iter().any(|e| e.polarity == Polarity::Off));
        assert!(mixed.iter().any(|e| e.polarity == Polarity::On));
        let recolour = |p: Polarity| -> Vec<PixelEvent> {
            mixed.iter().map(|e| PixelEvent { polarity: p, ..*e }).collect()
        };
        let all_on = recolour(Polarity::On);
        let all_off = recolour(Polarity::Off);
        let t_ref = 0.012;
        let still = Motion::Translation { vx: 0.0, vy: 0.0 };
        let unwarped = warped_image(g, &mixed, t_ref, still).unwrap();

        // Displacements below stay under two pixels, so nothing leaves the sensor and the image's
        // total mass is exactly the number of events — which is what "every event votes +1" means.
        for m in [
            Motion::Translation { vx: 100.0, vy: 50.0 },
            Motion::Rotation { omega_rad_s: 5.0, cx: 15.5, cy: 15.5 },
            Motion::RadialExpansion { rate_px_s: 30.0, cx: 15.5, cy: 15.5 },
        ] {
            let a = warped_image(g, &mixed, t_ref, m).unwrap();
            let b = warped_image(g, &all_on, t_ref, m).unwrap();
            let c = warped_image(g, &all_off, t_ref, m).unwrap();
            assert_eq!(a.frame, b.frame, "{m:?}: the warp depends on polarity");
            assert_eq!(a.frame, c.frame, "{m:?}: the warp depends on polarity");
            assert_eq!((a.placed, a.dropped), (b.placed, b.dropped));
            assert_eq!((a.placed, a.dropped), (24, 0), "{m:?}: events left the sensor");
            assert!(
                (a.frame.sum() - 24.0).abs() < 1e-9,
                "{m:?}: mass {} against 24 events",
                a.frame.sum()
            );
            // And the warp is doing something, so the equalities above are not the identity.
            assert_ne!(a.frame, unwarped.frame, "{m:?}: the warp moved nothing");
        }
        // The same for the accumulator that DOES read polarity, so the contrast above is real.
        assert_ne!(
            accumulate_polarity(g, &mixed).unwrap(),
            accumulate_polarity(g, &all_on).unwrap()
        );
    }

    /// **`k` and [`EHarris::K_HARRIS`] were both inert under test.** Hardcoding `0.04` inside
    /// `harris_score` survived, and so did `K_HARRIS: 0.04 -> 0.06`. The sign theorem the module
    /// asserts holds for any `k < 0.25`, which is exactly why it is a good test and exactly why it
    /// cannot pin the constant.
    ///
    /// # The patch, and why its structure tensor is known without computing it
    ///
    /// Take `v(i, j) = i * j + i` over a 5x5 patch, `i` the column and `j` the row. `Sobel` is
    /// separable, so on the 3x3 interior `gx = 8 * (j + 1)` and `gy = 8 * i`, and summing over
    /// `i, j` in `1..=3` gives, exactly and in integers,
    ///
    /// ```text
    /// Mxx = 64 * 3 * (4 + 9 + 16) = 5568      Myy = 64 * 3 * (1 + 4 + 9) = 2688
    /// Mxy = 64 * 6 * 9            = 3456
    /// det = 3022848                           trace = 8256
    /// ```
    ///
    /// whose eigenvalues are `4128 +/- sqrt(1440^2 + 3456^2) = 4128 +/- 3744`, i.e. `7872` and
    /// `384`: a ratio of exactly **20.5:1**. That sits between `rho_max(0.06) = 14.598` and
    /// `rho_max(0.04) = 22.956`, so the response is positive at the paper's constant and negative
    /// at the other end of `Harris`'s stated range — which is the doc's quantitative claim, as a
    /// test. The doc used to say `0.04` admits "about 19:1"; that was wrong, and this is the check
    /// that would have caught it.
    #[test]
    fn the_harris_constant_sets_the_eigenvalue_ratio_its_doc_names() {
        let g = geom(32, 32);
        let side = 5usize;
        let mut patch = vec![0.0; side * side];
        for j in 0..side {
            for i in 0..side {
                patch[j * side + i] = (i * j + i) as f64;
            }
        }
        let score = |k: f64| {
            EHarris::new(g, 4, 1e-3, k, 0.0).unwrap().score_of_patch(&patch, side).unwrap()
        };
        let trace_sq = 8256.0f64 * 8256.0;

        // `k = 0` leaves det(M) alone, so the structure tensor is read off exactly.
        assert_eq!(score(0.0), 3022848.0, "det(M) is not the value the derivation gives");
        // The response is affine in `k` with slope -trace(M)^2, so a second point names the trace.
        assert!((score(0.5) - (3022848.0 - 0.5 * trace_sq)).abs() < 1e-6, "{}", score(0.5));

        // THE CLAIM. A 20.5:1 corner is a corner at 0.04 and is not one at 0.06.
        assert_eq!(EHarris::K_HARRIS, 0.04, "not the constant the 1988 paper prints");
        let at_04 = score(EHarris::K_HARRIS);
        let at_06 = score(0.06);
        assert!(at_04 > 0.0, "a 20.5:1 corner scored {at_04} at k = 0.04");
        assert!(at_06 < 0.0, "a 20.5:1 corner scored {at_06} at k = 0.06");
        // The literal values too, so a `k` that is used but transcribed wrongly is caught as well
        // as a `k` that is ignored.
        assert!((at_04 - (3022848.0 - 0.04 * trace_sq)).abs() < 1e-6, "{at_04}");
        assert!((at_06 - (3022848.0 - 0.06 * trace_sq)).abs() < 1e-6, "{at_06}");
        assert!((at_04 - at_06).abs() > 1.0, "score_of_patch ignored the detector's own k");

        // And the boundary itself. `rho_max(k) = ((1 - 2k) + sqrt(1 - 4k)) / (2k)` inverts to
        // `k_crit = rho / (1 + rho)^2`, which for 20.5 is 20.5 / 21.5^2 = 0.04434829637641969.
        let k_crit = 20.5 / (21.5 * 21.5);
        assert!(score(k_crit * 0.999) > 0.0, "{}", score(k_crit * 0.999));
        assert!(score(k_crit * 1.001) < 0.0, "{}", score(k_crit * 1.001));
        assert!(k_crit > 0.04 && k_crit < 0.06, "k_crit = {k_crit} is outside the bracket");
        // The two ends of the closed form the doc prints, as numbers.
        let rho_max = |k: f64| ((1.0 - 2.0 * k) + (1.0 - 4.0 * k).sqrt()) / (2.0 * k);
        assert!((rho_max(0.04) - 22.9564392373896).abs() < 1e-12, "{}", rho_max(0.04));
        assert!((rho_max(0.06) - 14.598164905901124).abs() < 1e-12, "{}", rho_max(0.06));
        assert!(20.5 < rho_max(0.04) && 20.5 > rho_max(0.06), "the bracket does not contain 20.5");
    }

    /// **Three `eHarris` parameters were inert under test.**
    ///
    /// Every test constructs the detector with `threshold = 0.0`, so `if s > self.threshold`
    /// weakened to `if s > 0.0` survived — a documented constructor parameter could have been
    /// deleted. And the binarisation writes `1.0`; mutating it to `2.0` survived too, so
    /// "binarised" was unpinned in the streaming path, where it scales the response by sixteen.
    #[test]
    fn the_eharris_threshold_and_binarisation_are_both_used() {
        let g = geom(24, 24);
        // A right-angle corner: the quadrant x >= 12 and y >= 12, inside the radius-2 window
        // around (12, 12).
        let shape: Vec<(u16, u16)> = (10..=14u16)
            .flat_map(|y| (10..=14u16).map(move |x| (x, y)))
            .filter(|&(x, y)| x >= 12 && y >= 12)
            .collect();
        let stream: Vec<PixelEvent> = shape
            .iter()
            .enumerate()
            .map(|(n, &(x, y))| PixelEvent { t_s: n as f64 * 1e-4, x, y, polarity: Polarity::On })
            .chain(core::iter::once(PixelEvent {
                t_s: 1e-2,
                x: 12,
                y: 12,
                polarity: Polarity::On,
            }))
            .collect();

        // What the streaming path scores at the vertex ...
        let mut probe = EHarris::new(g, 2, 1.0, EHarris::K_HARRIS, 0.0).unwrap();
        for e in &stream {
            probe.push(*e).unwrap();
        }
        let s = probe.score_at(12, 12, 1e-2).unwrap();
        assert!(s > 0.0, "the corner scored {s}");

        // ... is EXACTLY the response of the same quadrant as a 0/1 patch. This is where the value
        // the binarisation writes is pinned: writing 2.0 scales every gradient by two and the
        // response by sixteen, and nothing else in the module would notice.
        let expect = probe.score_of_patch(&patch_of(5, |i, j| i >= 2 && j >= 2), 5).unwrap();
        assert_eq!(s, expect, "the binarised patch is not 0/1 valued");

        // The threshold is used, and the comparison is a STRICT `>` as the doc says: a score equal
        // to the threshold does not fire.
        let fire = |threshold: f64| {
            let mut d = EHarris::new(g, 2, 1.0, EHarris::K_HARRIS, threshold).unwrap();
            let mut last = None;
            for e in &stream {
                last = d.push(*e).unwrap();
            }
            last
        };
        assert!(fire(0.0).is_some(), "at threshold 0 the corner did not fire");
        assert!(fire(s * 0.5).is_some(), "at half the score the corner did not fire");
        assert!(fire(s).is_none(), "the score must EXCEED the threshold, not merely reach it");
        assert!(fire(s * 2.0).is_none(), "at twice the score the corner fired anyway");
        assert_eq!(fire(0.0).map(|c| (c.x, c.y)), Some((12, 12)));
        assert!((fire(0.0).unwrap().score - s).abs() < 1e-12);
    }


    /// **The shared `positive` boundary names a `NaN` as non-finite, and refuses an infinity
    /// outright.**
    ///
    /// `positive` is `finite` followed by `x > 0.0`, and deleting the `finite` call is invisible to
    /// every test above: [`a_degenerate_decay_constant_is_refused`] asserts `is_err()` and nothing
    /// finer, and `NaN > 0.0` is false, so a `NaN` still comes back as *an* error — under the
    /// variant that says "zero or negative", which is a different defect from the one that
    /// happened. An **infinity** is where the two differ in kind rather than in wording:
    /// `inf > 0.0` is true, so without the finiteness check an infinite time constant, speed or
    /// window is accepted, and an infinite `tau` makes `exp(-d/tau)` exactly `1` at every elapsed
    /// time — a surface that has forgotten nothing, plotted as if it had.
    #[test]
    fn the_positive_boundary_refuses_an_infinity_and_names_a_nan_as_non_finite() {
        assert_eq!(
            Decay::Exponential { tau_s: f64::INFINITY }.validate(),
            Err(VisionError::NonFinite { what: "decay time constant tau", value: f64::INFINITY })
        );
        assert!(matches!(
            Decay::Exponential { tau_s: f64::NAN }.validate(),
            Err(VisionError::NonFinite { what: "decay time constant tau", value }) if value.is_nan()
        ));
        // What an accepted infinity would build: a surface whose every value is exactly 1.
        assert_eq!(Decay::Exponential { tau_s: f64::INFINITY }.value(1.0), 1.0);
        assert!(TimeSurface::new(geom(4, 4), Decay::Exponential { tau_s: f64::INFINITY }, false).is_err());
        // A second entry point, so the check is pinned at `positive` rather than at one caller.
        assert_eq!(
            moving_edge(geom(8, 8), 0.0, f64::INFINITY, 0.0, 0.1, Polarity::On).err(),
            Some(VisionError::NonFinite { what: "edge speed", value: f64::INFINITY })
        );
        // And the rejection is still by SIGN where the value is finite, naming that instead.
        assert_eq!(
            Decay::Exponential { tau_s: -1e-3 }.validate(),
            Err(VisionError::NonPositive { what: "decay time constant tau", value: -1e-3 })
        );
    }

    /// **The linear decay's window is strictly positive, exactly like the exponential's `tau`.**
    ///
    /// [`a_degenerate_decay_constant_is_refused`] covers `Exponential { tau_s: 0.0 }` and
    /// `Exponential { tau_s: -1e-3 }`, but the only `Linear` it passes is a `NaN` — which the
    /// finiteness half of `positive` refuses on its own. So the sign half of the `Linear` arm was
    /// never reached, and weakening it to `finite` left a zero window accepted.
    ///
    /// A zero window is not a harmless parameter: `1 - 0/0` is `NaN`, the `v > 0.0` test reads that
    /// as zero, and the surface is then identically zero *including at the event itself* — the
    /// linear twin of the exp(-inf) surface the `tau` doc describes.
    #[test]
    fn the_linear_decays_window_must_be_strictly_positive_like_the_exponentials_tau() {
        assert_eq!(
            Decay::Linear { window_s: 0.0 }.validate(),
            Err(VisionError::NonPositive { what: "decay window", value: 0.0 })
        );
        assert_eq!(
            Decay::Linear { window_s: -1e-3 }.validate(),
            Err(VisionError::NonPositive { what: "decay window", value: -1e-3 })
        );
        // The surface a zero window would carry: zero at the event, not one.
        assert_eq!(Decay::Linear { window_s: 0.0 }.value(0.0), 0.0);
        assert!(TimeSurface::new(geom(4, 4), Decay::Linear { window_s: 0.0 }, false).is_err());
        assert!(matches!(
            Decay::Linear { window_s: f64::NAN }.validate(),
            Err(VisionError::NonFinite { what: "decay window", value }) if value.is_nan()
        ));
    }

    /// **A time past the wire clock's range has no encoding and is refused, not saturated.**
    ///
    /// [`the_microsecond_boundary_round_trips_exactly`] covers the negative and `NaN` refusals and
    /// five small timestamps, none of them within twenty orders of magnitude of the limit. Dropping
    /// the range check is therefore invisible: a float-to-integer cast in Rust *saturates*, so the
    /// mutant hands back `u64::MAX` — a wire timestamp of 584,942 years — for any time past it,
    /// silently, with the same `Some` a good conversion has.
    #[test]
    fn a_time_past_the_wire_clocks_range_has_no_encoding_rather_than_a_saturated_one() {
        let at = |t_s: f64| PixelEvent { t_s, x: 1, y: 2, polarity: Polarity::On };
        // 1e13 s is 1e19 us, inside `u64::MAX = 1.8446744e19`, and 1e19 is exactly a binary64.
        assert_eq!(at(1e13).to_aer().map(|a| a.t), Some(10_000_000_000_000_000_000));
        // 1e14 s is 1e20 us, past it. The saturating cast would give 18446744073709551615.
        assert_eq!(at(1e14).to_aer(), None);
        assert_eq!(at(f64::MAX).to_aer(), None);
    }

    /// **A stream may hold simultaneous events; only a step backwards is refused.**
    ///
    /// [`require_time_ordered`]'s doc says "non-decreasing", and [`TimeSurface::update`] has an
    /// explicit equal-times case ("a sensor emits a whole column at one instant") — but the free
    /// function had no test of its own at all, so tightening its `<` to `<=` broke the documented
    /// tolerance of simultaneity without failing anything. [`Hats::descriptor`] is its only caller
    /// in this crate, and that is where the breakage would have surfaced, on a real sensor, as a
    /// refusal to describe a frame.
    #[test]
    fn a_stream_may_hold_simultaneous_events_and_only_a_step_backwards_is_refused() {
        let at = |t_s: f64, x: u16| PixelEvent { t_s, x, y: 0, polarity: Polarity::On };
        assert_eq!(require_time_ordered(&[at(1.0, 0), at(1.0, 1), at(1.0, 2)]), Ok(()));
        assert_eq!(require_time_ordered(&[at(0.0, 0), at(1.0, 1), at(1.0, 2), at(2.0, 3)]), Ok(()));
        assert_eq!(
            require_time_ordered(&[at(1.0, 0), at(0.5, 1)]),
            Err(VisionError::OutOfOrder { t_s: 0.5, now_s: 1.0 })
        );

        // Through the consumer, with the arithmetic a simultaneous pair produces. One cell, one
        // plane, nine bins: each event writes 1.0 into the centre bin (index 4), and the second
        // event also sees the first at `dt = 0`, one pixel to its left — bin `(0 + 1) * 3 + 0 = 3`
        // — weighted `exp(-0 / tau) = 1`. The cell's two events then divide both.
        let g = geom(10, 10);
        let h = Hats { cell_px: 10, radius: 1, tau_s: 1.0, window_s: 1.0, split_polarity: false };
        let d = h.descriptor(g, &[at(1.0, 4), at(1.0, 5)]).expect("simultaneous events describe");
        assert_eq!(d[4], 1.0, "each event's own centre contribution, over a divisor of two");
        assert_eq!(d[3], 0.5, "the simultaneous neighbour at full weight, over a divisor of two");
        assert_eq!(d.iter().filter(|v| **v != 0.0).count(), 2);
    }

    /// **The decayed accumulator validates its decay before it weights anything.**
    ///
    /// Every other consumer of [`Decay`] goes through [`TimeSurface::new`], whose own test covers
    /// the refusal; [`accumulate_decay`] is the one that calls [`Decay::validate`] directly, and
    /// deleting that call leaves no error and no panic. A zero `tau` gives `exp(-elapsed / 0)`,
    /// which is `0` for every event but one and `1` at the reference time exactly — a sparse image
    /// that looks like a recording of a nearly static scene, which is the shape of wrong answer
    /// this module refuses at its boundaries rather than returns.
    #[test]
    fn the_decayed_accumulator_validates_its_decay_before_it_weights_anything() {
        let g = geom(8, 8);
        let ev = [PixelEvent { t_s: 0.0, x: 1, y: 1, polarity: Polarity::On }];
        assert_eq!(
            accumulate_decay(g, &ev, Decay::Exponential { tau_s: 0.0 }, 1.0),
            Err(VisionError::NonPositive { what: "decay time constant tau", value: 0.0 })
        );
        assert_eq!(
            accumulate_decay(g, &ev, Decay::Linear { window_s: -1.0 }, 1.0),
            Err(VisionError::NonPositive { what: "decay window", value: -1.0 })
        );
        assert!(matches!(
            accumulate_decay(g, &ev, Decay::Exponential { tau_s: f64::NAN }, 1.0),
            Err(VisionError::NonFinite { what: "decay time constant tau", .. })
        ));
        // What the mutant would have accumulated instead, which is why the refusal belongs at the
        // boundary: a zero `tau` weights every earlier event by exactly zero and the event at the
        // reference time by `exp(-0 / 0)`, which is a `NaN` — one poisoned pixel in an image whose
        // every other pixel is a plausible zero.
        assert_eq!(Decay::Exponential { tau_s: 0.0 }.value(1.0), 0.0);
        assert!(Decay::Exponential { tau_s: 0.0 }.value(0.0).is_nan());
    }

    /// **The frame statistics divide by the data present, not by the lattice claimed.**
    ///
    /// [`Frame::mean`]'s doc says "over `data` rather than over `width * height`", and
    /// [`a_frame_whose_data_is_short_reports_none_rather_than_panicking`] builds exactly the frame
    /// where those differ — three values on a 4x4 lattice — but asserts `mean()` only for the
    /// *empty* frame, where both divisors give `None`. Everything else in the module produces a
    /// consistent frame, so `data.len()` and `pixels()` are the same number in every other fixture.
    #[test]
    fn the_frame_statistics_divide_by_the_data_present_not_by_the_lattice_claimed() {
        let short = Frame { width: 4, height: 4, data: vec![1.0, 2.0, 3.0] };
        assert!(!short.is_consistent());
        assert_eq!(short.geometry().pixels(), 16);
        assert_eq!(short.sum(), 6.0);
        // 6 / 3, not 6 / 16. Exact: both divisors are small integers and the quotient is exact.
        assert_eq!(short.mean(), Some(2.0));
        // Population variance over the same three values: ((1-2)^2 + 0 + (3-2)^2) / 3.
        assert_eq!(short.variance(), Some(2.0 / 3.0));
        assert_eq!(short.max(), Some(3.0));
        assert_eq!(short.min(), Some(1.0));
        // A consistent frame is the case where the two divisors agree, and it still holds there.
        let full = Frame { width: 2, height: 2, data: vec![1.0, 2.0, 3.0, 4.0] };
        assert!(full.is_consistent());
        assert_eq!(full.mean(), Some(2.5));
    }

    /// **The second pass fits the points that survived rejection, not the ones that went in.**
    ///
    /// [`outlier_rejection_recovers_a_fit_that_a_single_hot_pixel_destroys`] does the paper's two
    /// stages *beside* [`PlaneFlow`], by calling [`fit_plane`] twice and filtering in the test
    /// itself, so the estimator's own second pass had no test at all: refitting on `pts` instead of
    /// `kept` returns the first-pass plane and every outcome stays a `Fitted`, with a velocity that
    /// is wrong by the outlier's pull and by nothing that shows.
    ///
    /// The neighbourhood here is the same 5x5 on `t = x / 400`, driven through the estimator as a
    /// stream. The hot pixel is at `dx = +2` from the centre: an outlier *at* the centre would move
    /// only the intercept `c`, because its centred coordinates are zero, and the flow would not
    /// notice it at all.
    #[test]
    fn the_plane_fits_second_pass_uses_the_points_that_survived_rejection() {
        let g = geom(32, 32);
        // The clean support: t = x / 400, i.e. 400 px/s along +x, over x in 16..=20, y in 16..=20.
        let mut stream: Vec<PixelEvent> = Vec::new();
        for x in 16..=20u16 {
            for y in 16..=20u16 {
                stream.push(PixelEvent {
                    t_s: f64::from(x) / 400.0,
                    x,
                    y,
                    polarity: Polarity::On,
                });
            }
        }
        // A hot pixel re-firing 10 ms off the plane at (20, 18), then the event whose flow is read,
        // at the neighbourhood's centre. Both are later than every clean event, which is what lets
        // a causal stream carry an outlier at all.
        stream.push(PixelEvent { t_s: 0.060, x: 20, y: 18, polarity: Polarity::On });
        stream.push(PixelEvent { t_s: 0.065, x: 18, y: 18, polarity: Polarity::On });

        let run = |reject_s: Option<f64>| -> f64 {
            let mut pf = PlaneFlow::new(g, 2, 0.03, reject_s, 8).unwrap();
            let outcomes = pf.run(&stream).unwrap();
            match outcomes.last().expect("one outcome per event") {
                FlowOutcome::Fitted(f) => f.speed,
                other => panic!("the last event produced {other:?}"),
            }
        };

        // Without the rejection pass the hot pixel pulls dt/dx from 1/400 to 0.0029 s/px, so the
        // reported speed is 1 / 0.0029 = 344.83 px/s. The damage is asserted first, or the repair
        // below would prove nothing.
        let pulled = run(None);
        assert!(
            (pulled - 400.0).abs() / 400.0 > 0.10,
            "the outlier only moved the fit to {pulled} px/s"
        );

        // With it, the two displaced points are the only residuals past 4 ms — the clean ones
        // reach 2.0 ms against the pulled plane — so 23 of the 25 survive and the refit is the
        // exact plane they lie on.
        let repaired = run(Some(4e-3));
        assert!(
            (repaired - 400.0).abs() < 1e-6,
            "the refit reported {repaired} px/s, not the 400 px/s its 23 inliers lie on"
        );
    }

    /// **The learning rate counts the selected cluster's own selections, not the layer's total.**
    ///
    /// The paper writes `alpha = 0.01 / (1 + p_k / 20000)` with `p_k` the number of times cluster
    /// `k` has been selected, and [`the_hots_learning_rate_is_the_papers_two_constants`] pins the
    /// expression. What no test could see is *which count is substituted for `p_k`*: every `HOTS`
    /// fixture in this module drives a layer with **one** cluster, and with one cluster the layer's
    /// total and that cluster's own count are the same number.
    ///
    /// Two clusters here, and one of them is dragged to exactly [`Hots::ALPHA_DECAY`] selections
    /// while the other is never chosen — so the rate the second one finally learns at is either
    /// `alpha0` (its own count is zero) or `alpha0 / 2` (the layer's total is 20000), a factor of
    /// two apart rather than a rounding apart.
    #[test]
    fn the_hots_learning_rate_counts_its_own_clusters_selections_not_the_layers() {
        let g = geom(16, 16);
        // `hot` is exactly the centre indicator, which is the patch an isolated event produces, so
        // it wins every one of the first 20000 events. It is also parallel to that patch and of
        // unit length, so `beta` is exactly 1 and its own update is exactly zero: it stays the
        // nearest prototype, unchanged, for the whole run.
        let mut hot = vec![0.0; 9];
        hot[4] = 1.0;
        // `cold` is nearer the patch that a SIMULTANEOUS neighbour one pixel up and to the left
        // produces, and nothing else. It is not parallel to it, so its one update is visible.
        let mut cold = vec![0.0; 9];
        cold[0] = 1.0;
        cold[4] = 0.5;
        let mut h = Hots::with_centers(g, 1, 1e-3, vec![hot.clone(), cold.clone()]).unwrap();

        let selections = Hots::ALPHA_DECAY as u64;
        for n in 0..selections {
            let k = h
                .learn(PixelEvent { t_s: n as f64 * 1e-3, x: 8, y: 8, polarity: Polarity::On })
                .unwrap();
            assert_eq!(k, 0, "the centre-indicator prototype must win an isolated event");
        }
        assert_eq!(h.counts(), &[selections, 0]);
        assert_eq!(h.centers()[0], hot, "a prototype parallel to the patch does not move");

        // One event whose patch is `e0 + e4`: the neighbour is recorded with `assign`, which reads
        // the surface without touching a count or a centre, so the layer's totals are undisturbed.
        let t = selections as f64 * 1e-3;
        h.assign(PixelEvent { t_s: t, x: 7, y: 7, polarity: Polarity::On }).unwrap();
        let k = h.learn(PixelEvent { t_s: t, x: 8, y: 8, polarity: Polarity::On }).unwrap();
        assert_eq!(k, 1, "the patch e0 + e4 is at distance 0.5 from `cold` and 1.0 from `hot`");
        assert_eq!(h.counts(), &[selections, 1]);

        // The paper's rule, written out here from its own two constants at `p_k = 0`.
        let alpha = 0.01 / (1.0 + 0.0 / 20000.0);
        let beta = 1.5 / (1.25f64.sqrt() * 2.0f64.sqrt());
        let want0 = 1.0 + alpha * (1.0 - beta * 1.0);
        let want4 = 0.5 + alpha * (1.0 - beta * 0.5);
        assert!((h.centers()[1][0] - want0).abs() < 1e-18, "{}", h.centers()[1][0]);
        assert!((h.centers()[1][4] - want4).abs() < 1e-18, "{}", h.centers()[1][4]);
        // And the two readings are a factor of two apart, so the assertion above is not a
        // tolerance question: at the layer's total the step would be half this one.
        let halved = 0.5 + 0.005 * (1.0 - beta * 0.5);
        assert!((want4 - halved).abs() > 1e-3, "the two readings differ by {}", want4 - halved);
    }

    /// **The two `N-CARS` numbers nothing read.**
    ///
    /// [`hats_refuses_what_it_cannot_describe`] checks `cell_px` and `bins_per_cell` (which is
    /// `radius`), and the descriptor tests build their own [`Hats`] literals — so `tau_s` and
    /// `window_s` of [`Hats::n_cars`] were read by no assertion anywhere, and the one call that
    /// used the constructor handed it a stream that errored before either number was reached.
    ///
    /// They are pinned here **through the arithmetic**, not as field reads: a neighbour 0.05 s back
    /// is weighted `exp(-0.05 / tau)`, which is 0.951 at the one-second `tau` this implementation
    /// reads and 0.607 at a hundred-millisecond one; and a neighbour 0.45 s back is outside a
    /// hundred-millisecond memory window and inside a one-second one.
    ///
    /// The type doc flags the `tau` reading as uncertain and asks a reader with the paper to check
    /// it; this review did not locate an author-released reference implementation to check the
    /// transcription against. What this test pins is that changing the transcription is a visible
    /// change rather than a silent one.
    #[test]
    fn the_n_cars_settings_are_the_ones_this_implementation_reads_from_the_paper() {
        let h = Hats::n_cars();
        assert_eq!(h.cell_px, 10);
        assert_eq!(h.radius, 3);
        assert_eq!(h.tau_s, 1.0, "the N-CARS time constant this implementation reads");
        assert_eq!(h.window_s, 0.1, "the N-CARS memory window this implementation reads");
        assert!(h.split_polarity);

        // One 10x10 cell, 7x7 = 49 bins per plane, two planes; the `On` plane is the second, so
        // its base is 49 and its centre bin is 3 * 7 + 3 = 24.
        let g = geom(10, 10);
        let on = |t_s: f64, x: u16| PixelEvent { t_s, x, y: 4, polarity: Polarity::On };
        let d = h.descriptor(g, &[on(0.0, 4), on(0.05, 5), on(0.5, 6)]).unwrap();
        assert_eq!(d.len(), 98);
        let (base, centre) = (49usize, 3 * 7 + 3);
        // Three events in the cell, each contributing its own `exp(0) = 1` at the centre.
        assert_eq!(d[base + centre], 1.0);
        // The second event's neighbour is one pixel to its left and 0.05 s back: bin 3 * 7 + 2.
        let want = (-0.05f64 / 1.0).exp() / 3.0;
        assert!((d[base + centre - 1] - want).abs() < 1e-15, "{}", d[base + centre - 1]);
        // The third event's neighbours are 0.45 s and 0.5 s back, outside a 0.1 s memory window.
        // Two pixels to its left is bin 3 * 7 + 1, and it is exactly zero.
        assert_eq!(d[base + centre - 2], 0.0, "an event 0.5 s back reached inside the window");
        assert_eq!(d.iter().filter(|v| **v != 0.0).count(), 2);
    }

    /// **The averaging step divides each cell's own block, and a cell's block is `planes * bins`
    /// long.**
    ///
    /// Every `HATS` fixture above tiles the sensor into exactly **one** cell, and for cell zero
    /// `c * planes * bins` and `c * bins` are both zero — so the stride the averaging loop walks
    /// was unpinned, and dropping `planes` from it makes cell one's divisor land half a block
    /// early: it would divide cell zero's `On` plane by cell one's event count and leave cell one's
    /// own `On` plane undivided.
    ///
    /// Two cells here, with different event counts, and `On` content in both.
    #[test]
    fn each_cells_averaging_divides_that_cells_own_two_planes() {
        let g = geom(20, 10);
        let h = Hats { cell_px: 10, radius: 1, tau_s: 1.0, window_s: 2.0, split_polarity: true };
        let ev = |t_s: f64, x: u16, polarity: Polarity| PixelEvent { t_s, x, y: 5, polarity };
        // Cell 0 (x < 10) holds one `On` event; cell 1 holds one `On` and two `Off`.
        let d = h
            .descriptor(g, &[
                ev(0.0, 5, Polarity::On),
                ev(0.25, 15, Polarity::On),
                ev(0.5, 16, Polarity::Off),
                ev(0.75, 17, Polarity::Off),
            ])
            .unwrap();
        assert_eq!(d.len(), 2 * 2 * 9);
        // Block base is `(cell * planes + plane) * bins`, `Off` is plane 0 and `On` is plane 1,
        // and the centre of a 3x3 neighbourhood is bin 4.
        let block = |cell: usize, plane: usize| (cell * 2 + plane) * 9;
        assert_eq!(d[block(0, 1) + 4], 1.0, "cell 0's single On event, over a divisor of one");
        assert_eq!(d[block(1, 1) + 4], 1.0 / 3.0, "cell 1's On plane, over cell 1's three events");
        assert_eq!(d[block(1, 0) + 4], 2.0 / 3.0, "cell 1's two Off events, over three");
        // The later `Off` event sees the earlier one, one pixel to its left, 0.25 s back.
        assert_eq!(d[block(1, 0) + 3], (-0.25f64).exp() / 3.0);
        assert_eq!(d[block(0, 0) + 4], 0.0, "cell 0 has no Off event");
        assert_eq!(d.iter().filter(|v| **v != 0.0).count(), 4);
    }

    /// **The `eHarris` constructor's two unexercised refusals: a window too small to hold a
    /// structure tensor, and a non-finite `Harris` constant.**
    ///
    /// Every detector built in this module is built with `radius = 2` or more and with
    /// [`EHarris::K_HARRIS`], so `radius < 2` was never reached from below and `k` was never
    /// anything but a finite constant. Both weakenings are silent rather than loud: at `radius = 1`
    /// the `Sobel` interior of a 3x3 window is the single centre pixel, so the structure tensor is
    /// a rank-one matrix whose determinant is exactly zero and whose score is therefore
    /// `-k * trace^2` — negative, plausible, and blind to corners by construction. A `NaN` `k`
    /// makes every score a `NaN`, and `NaN > threshold` is false, so the detector simply never
    /// fires again.
    #[test]
    fn the_eharris_constructor_refuses_a_window_below_two_and_a_non_finite_constant() {
        let g = geom(32, 32);
        assert_eq!(
            EHarris::new(g, 1, 1e-3, EHarris::K_HARRIS, 0.0).err(),
            Some(VisionError::TooFew { what: "eHarris window radius", have: 1, need: 2 })
        );
        assert_eq!(
            EHarris::new(g, 0, 1e-3, EHarris::K_HARRIS, 0.0).err(),
            Some(VisionError::TooFew { what: "eHarris window radius", have: 0, need: 2 })
        );
        // Two is accepted, so the boundary is pinned from both sides rather than only from below.
        assert!(EHarris::new(g, 2, 1e-3, EHarris::K_HARRIS, 0.0).is_ok());

        assert!(matches!(
            EHarris::new(g, 2, 1e-3, f64::NAN, 0.0),
            Err(VisionError::NonFinite { what: "Harris constant k", .. })
        ));
        assert_eq!(
            EHarris::new(g, 2, 1e-3, f64::INFINITY, 0.0).err(),
            Some(VisionError::NonFinite { what: "Harris constant k", value: f64::INFINITY })
        );
        // The threshold's own check is beside it and stays reachable.
        assert!(matches!(
            EHarris::new(g, 2, 1e-3, EHarris::K_HARRIS, f64::NAN),
            Err(VisionError::NonFinite { what: "eHarris threshold", .. })
        ));
        // What a radius-1 window would have scored: a rank-one structure tensor over a one-pixel
        // interior, whose determinant is exactly zero whatever the pattern in it.
        let d = EHarris::new(g, 2, 1e-3, EHarris::K_HARRIS, 0.0).unwrap();
        let corner = d.score_of_patch(&patch_of(3, |i, j| i >= 1 && j >= 1), 3).unwrap();
        assert!(corner < 0.0, "a 3x3 window scored {corner} on a corner, so it can see one");
    }

    /// **The `Harris` scorer's own two guards, neither of which any patch in this module reached.**
    ///
    /// [`patch_of`] builds squares of side 3, 5 and 7 from a boolean predicate, so `side < 3` and
    /// a non-finite entry were both unreachable from every existing fixture. The first is not a
    /// tidiness check: with the side guard dropped, `side = 2` makes both `Sobel` loops
    /// (`1..side - 1`) empty, the structure tensor is all zeros, and the function returns a
    /// confident `Ok(0.0)` — a score exactly on the sign boundary the whole detector is built on.
    /// At `side = 0` the same expression underflows and the function indexes an empty slice.
    #[test]
    fn the_harris_scorer_refuses_a_patch_too_small_for_its_kernel_or_carrying_a_nan() {
        let d = EHarris::new(geom(32, 32), 2, 1e-3, EHarris::K_HARRIS, 0.0).unwrap();
        for (patch, side) in [
            (vec![], 0usize),
            (vec![1.0], 1),
            (vec![1.0, 0.0, 0.0, 1.0], 2),
        ] {
            assert_eq!(
                d.score_of_patch(&patch, side).err(),
                Some(VisionError::BadParameters {
                    why: "the patch length does not match a square of side at least 3",
                }),
                "side {side} was scored"
            );
        }
        // Side 3 is the smallest the kernel fits, and it is accepted.
        assert!(d.score_of_patch(&[0.0; 9], 3).is_ok());
        // A non-finite entry is refused rather than propagated into the response, where it would
        // read as "no corner here" at every threshold.
        let mut poisoned = patch_of(5, |i, j| i >= 2 && j >= 2);
        poisoned[7] = f64::NAN;
        assert!(matches!(
            d.score_of_patch(&poisoned, 5),
            Err(VisionError::NonFinite { what: "patch entry", .. })
        ));
        poisoned[7] = f64::NEG_INFINITY;
        assert_eq!(
            d.score_of_patch(&poisoned, 5).err(),
            Some(VisionError::NonFinite { what: "patch entry", value: f64::NEG_INFINITY })
        );
    }

    /// **A circle that runs off the sensor is refused, and the refusal names the pixel the caller
    /// asked about.**
    ///
    /// [`corner_detectors_fire_near_a_moving_vertex_and_rarely_on_a_straight_edge`] drives streams
    /// whose border events come back as `Ok(None)` through [`EFast::push`], which maps *any*
    /// [`VisionError::OutOfBounds`] to `None` — so the bounds test inside `circle_times` was
    /// covered only in the sense that something, somewhere, refused. Delete it and a negative
    /// circle coordinate becomes `(-1i64) as u16 = 65535`, which the surface refuses on its own:
    /// the same `Err` variant, carrying row 65535 of a sixteen-row sensor, for a pixel no caller
    /// named. The distinction the error exists to make — *which* pixel is off the lattice — is the
    /// thing the mutation destroys, so it is the thing asserted.
    #[test]
    fn a_circle_running_off_the_sensor_is_refused_by_the_centre_the_caller_named() {
        let g = geom(16, 16);
        let d = EFast::new(g).unwrap();
        // Each centre is itself ON the lattice; it is the radius-4 circle around it that is not.
        for (x, y) in [(3u16, 2u16), (2, 3), (12, 13), (13, 12), (0, 0), (15, 15)] {
            assert!(g.require(x, y).is_ok(), "({x}, {y}) is a real pixel");
            assert_eq!(
                d.arcs_at(x, y, Polarity::On),
                Err(VisionError::OutOfBounds { x, y, width: 16, height: 16 }),
                "the circle around ({x}, {y})"
            );
        }
        // Four pixels in from every edge, the circle fits and the call succeeds.
        assert_eq!(d.arcs_at(4, 4, Polarity::On), Ok((None, None)));
        assert_eq!(d.arcs_at(11, 11, Polarity::On), Ok((None, None)));
        // And a border event is still a `None` from `push`, not an error: a real recording always
        // has border events.
        let mut d = EFast::new(g).unwrap();
        assert_eq!(d.push(PixelEvent { t_s: 0.0, x: 0, y: 0, polarity: Polarity::On }), Ok(None));
    }

    /// **The bilinear vote's four weights, and the stride the warped image is written with.**
    ///
    /// [`the_bilinear_vote_conserves_event_mass_inside_the_sensor`] asserts the *sum* of the four
    /// weights and how many pixels they touch, and both are invariant under exchanging the two
    /// off-diagonal weights — `fx(1-fy)` and `(1-fx)fy` sum to the same thing whichever corner
    /// gets which. Every warp fixture in this module also runs on a **square** sensor, where
    /// `py * width` and `py * height` are the same index, so the one hardcoded row-major write in
    /// the module was unpinned.
    ///
    /// One event, a 16x8 sensor, and a warp landing at `(7.25, 4.5)`: the two off-diagonal weights
    /// are then 0.125 and 0.375 rather than equal, and the row is 4 rather than 0. Every number
    /// here is an exact binary64, so these are equalities and not tolerances.
    #[test]
    fn the_bilinear_vote_puts_each_weight_on_its_own_pixel_of_a_non_square_sensor() {
        let g = geom(16, 8);
        let evs = [PixelEvent { t_s: 0.1, x: 6, y: 3, polarity: Polarity::On }];
        // warp(x, y, dt) = (x - vx dt, y - vy dt) with dt = 0.1: (6 + 1.25, 3 + 1.5).
        let wi = warped_image(g, &evs, 0.0, Motion::Translation { vx: -12.5, vy: -15.0 }).unwrap();
        assert_eq!(wi.placed, 1);
        assert_eq!(wi.dropped, 0);
        assert_eq!(wi.frame.sum(), 1.0);
        // fx = 0.25, fy = 0.5, and the four corners are (7, 4), (8, 4), (7, 5), (8, 5).
        assert_eq!(wi.frame.at(7, 4), Some(0.75 * 0.5), "the (0, 0) corner");
        assert_eq!(wi.frame.at(8, 4), Some(0.25 * 0.5), "the (+1, 0) corner takes fx * (1 - fy)");
        assert_eq!(wi.frame.at(7, 5), Some(0.75 * 0.5), "the (0, +1) corner takes (1 - fx) * fy");
        assert_eq!(wi.frame.at(8, 5), Some(0.25 * 0.5), "the (+1, +1) corner");
        // The two off-diagonal weights must be different numbers, or exchanging them is invisible.
        assert!(wi.frame.at(8, 4) != wi.frame.at(7, 5));
        // Nothing landed anywhere else — which is what a wrong stride would show as.
        assert_eq!(wi.frame.data.iter().filter(|v| **v != 0.0).count(), 4);
    }

    /// **The sweep validates both of its bounds, and the search needs two grid points, not one.**
    ///
    /// [`a_sweep_refuses_what_it_cannot_sweep`] passes a `NaN` *reference time* and a zero
    /// *bound*, and asserts the variant only — so dropping `finite` from the upper bound, or
    /// admitting a one-point search grid, still produced an `Err` and still matched. Both mutants
    /// fail further in and under the wrong name: a `NaN` bound reaches [`Motion::validate`] as a
    /// `NaN` velocity ("translation vx"), and a one-point grid divides by `steps - 1 = 0`, which
    /// makes every candidate velocity a `NaN` and arrives at the same place. A caller reading the
    /// error would go looking at the motion family for a defect in the argument it passed.
    #[test]
    fn the_sweep_and_the_search_name_the_argument_that_was_wrong() {
        let g = geom(8, 8);
        let evs = [PixelEvent { t_s: 0.0, x: 4, y: 4, polarity: Polarity::On }];
        let translate = |v: f64| Motion::Translation { vx: v, vy: 0.0 };
        assert!(matches!(
            sweep(g, &evs, 0.0, 0.0, f64::NAN, 4, Objective::Variance, translate),
            Err(VisionError::NonFinite { what: "sweep upper bound", .. })
        ));
        assert_eq!(
            sweep(g, &evs, 0.0, 0.0, f64::INFINITY, 4, Objective::Variance, translate).err(),
            Some(VisionError::NonFinite { what: "sweep upper bound", value: f64::INFINITY })
        );
        assert!(matches!(
            sweep(g, &evs, 0.0, f64::NAN, 1.0, 4, Objective::Variance, translate),
            Err(VisionError::NonFinite { what: "sweep lower bound", .. })
        ));
        // A two-point sweep is the smallest that is a sweep, and it is accepted.
        assert_eq!(sweep(g, &evs, 0.0, 0.0, 1.0, 2, Objective::Variance, translate).unwrap().len(), 2);

        // The search grid: one point is not a grid, and it is refused by name rather than divided
        // by zero.
        assert_eq!(
            search_translation(g, &evs, 0.0, 10.0, 1, 0, Objective::Variance).err(),
            Some(VisionError::TooFew { what: "search grid", have: 1, need: 2 })
        );
        assert_eq!(
            search_translation(g, &evs, 0.0, 10.0, 0, 0, Objective::Variance).err(),
            Some(VisionError::TooFew { what: "search grid", have: 0, need: 2 })
        );
        assert!(search_translation(g, &evs, 0.0, 10.0, 2, 0, Objective::Variance).is_ok());
    }

    /// **The coarse grid covers the whole square `[-bound, bound]^2`, in both components
    /// independently.**
    ///
    /// [`contrast_maximisation_recovers_two_components_from_a_corner`] runs the search with 24
    /// refinement rounds, and the pattern search walks far enough to repair a grid that covers
    /// only half the square or only its diagonal — it arrives within 5 px/s of truth either way,
    /// which is all that test asserts. So the grid itself was untested.
    ///
    /// Here `refine` is **zero**, which makes the answer exactly a grid point, and the true
    /// velocity `(180, -120)` is on the grid but is neither in the half-square `vx <= 0` nor on
    /// the diagonal `vy = vx`. The assertion is an equality on both components.
    #[test]
    fn the_coarse_search_grid_covers_the_whole_square_and_not_only_its_diagonal() {
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
        // 13 points over +/-360 px/s is a spacing of 60, so truth is exactly on a grid point and
        // "the search returns truth" is an equality rather than a tolerance.
        let (m, score) = search_translation(g, &evs, 0.125, 360.0, 13, 0, Objective::Variance).unwrap();
        assert_eq!(
            m,
            Motion::Translation { vx: tvx, vy: tvy },
            "the coarse grid's best point, with no refinement, was {m:?}"
        );
        assert_eq!(score, contrast(g, &evs, 0.125, m, Objective::Variance).unwrap());
    }

    /// **A recording of zero length is refused, in every generator that takes a duration.**
    ///
    /// The four generators all call `positive(duration_s, "scene duration")`, and no test passed
    /// any of them a zero: weakened to `finite`, a zero duration emits the single instant `t = 0`
    /// — for [`moving_edge`] that is one whole column of the sensor, a perfectly plausible frame
    /// of events describing a scene that did not move. A negative duration emits an empty vector,
    /// which is the same answer a static scene gives and is indistinguishable from it.
    #[test]
    fn a_recording_of_zero_or_negative_length_is_refused_by_every_generator() {
        let g = geom(16, 16);
        let pi = core::f64::consts::PI;
        for bad in [0.0, -1e-3] {
            assert_eq!(
                moving_edge(g, 0.0, 100.0, 3.0, bad, Polarity::On).err(),
                Some(VisionError::NonPositive { what: "scene duration", value: bad })
            );
            assert_eq!(
                moving_corner(g, (2.0, 2.0), (100.0, 50.0), (0.0, pi / 2.0), 8.0, bad, Polarity::On)
                    .err(),
                Some(VisionError::NonPositive { what: "scene duration", value: bad })
            );
            assert_eq!(
                rotating_bar(g, (8.0, 8.0), 9.0, 0.0, 1.0, 6.0, bad, Polarity::On).err(),
                Some(VisionError::NonPositive { what: "scene duration", value: bad })
            );
            assert_eq!(
                looming_disc(g, (8.0, 8.0), 1.0, 100.0, bad, Polarity::On).err(),
                Some(VisionError::NonPositive { what: "scene duration", value: bad })
            );
        }
        // What a zero-duration edge would have produced: the whole column it crosses at t = 0.
        let column = moving_edge(g, 0.0, 100.0, 3.0, 1e-9, Polarity::On).unwrap();
        assert_eq!(column.len(), 16, "the instant t = 0 is a full column of events");
    }

    /// **The corner's arms point along `(cos, sin)` of their angles, not `(sin, cos)`.**
    ///
    /// [`the_moving_corner_generator_puts_every_event_on_an_arm`] uses the arm pair
    /// `(0, pi/2)`, whose two direction vectors are `(1, 0)` and `(0, 1)` — and exchanging the
    /// components of both maps that pair onto **itself**. The stimulus is identical, so the test
    /// that checks every event against the closed form agrees with the mutant perfectly. Any arm
    /// angle that is not a multiple of `pi/2` breaks the symmetry, and the pair here is also not
    /// symmetric about the `45`-degree line, where the exchange is a reflection.
    #[test]
    fn the_corners_arms_run_along_the_cosine_and_sine_of_their_angles() {
        let g = geom(48, 48);
        let v0 = (8.0, 6.0);
        let v = (120.0, 80.0);
        let arms = (0.3, 1.9);
        let (len, dur) = (24.0, 0.3);
        let evs = moving_corner(g, v0, v, arms, len, dur, Polarity::On).unwrap();
        assert!(evs.len() > 300, "only {} events", evs.len());
        let mut on_arm = [0usize, 0usize];
        for e in &evs {
            // The vertex at this instant, and the event's offset from it.
            let (px, py) = (
                f64::from(e.x) - (v0.0 + v.0 * e.t_s),
                f64::from(e.y) - (v0.1 + v.1 * e.t_s),
            );
            let mut matched = false;
            for (k, a) in [arms.0, arms.1].into_iter().enumerate() {
                // The arm's direction is (cos a, sin a): the angle is measured from +x, as every
                // other angle in this module is.
                let (dx, dy) = (a.cos(), a.sin());
                let perp = (px * dy - py * dx).abs();
                let along = px * dx + py * dy;
                if perp < 1e-9 && (-1e-9..=len + 1e-9).contains(&along) {
                    on_arm[k] += 1;
                    matched = true;
                }
            }
            assert!(matched, "an event at ({}, {}) t={} is on neither arm", e.x, e.y, e.t_s);
        }
        assert!(on_arm[0] > 50 && on_arm[1] > 50, "arm coverage {on_arm:?}");
        // Neither arm is parallel to the velocity under either reading of `sin_cos`, so the
        // generator is producing a stimulus in both cases rather than refusing in one of them.
        for a in [arms.0, arms.1] {
            for (dx, dy) in [(a.cos(), a.sin()), (a.sin(), a.cos())] {
                assert!((v.0 * dy - v.1 * dx).abs() > 1.0, "arm {a} is nearly parallel");
            }
        }
    }

    /// **The event budget counts the crossing every annulus pixel gets for free.**
    ///
    /// A pixel is crossed `floor(duration / period) + 1` times at most — the `+ 1` is the crossing
    /// that happens before the first full half-turn completes, and it is the only one there is
    /// when the recording is shorter than a half-turn. [`rotating_bar_refuses_a_sweep_it_cannot_finish`]
    /// drives the cap with a **large** `omega`, where `floor(duration / period)` is in the
    /// millions and one crossing either way is a rounding; the case that can see the `+ 1` is the
    /// opposite one, where `floor(duration / period)` is **zero** and dropping the term takes the
    /// estimate from "one event per annulus pixel" to "none at all", which passes any cap.
    ///
    /// Ten million and four thousand pixels, each of which could fire once: past
    /// [`MAX_ROTATING_BAR_EVENTS`], and refused. The sensor is large and the scan over it is the
    /// cost of the test — 19 ms measured here — because the refusal has to be about the *count*,
    /// and the count is what a large sensor has.
    #[test]
    fn the_rotating_bars_budget_counts_the_first_crossing_of_every_pixel() {
        // 4000 x 2501 = 10,004,000 pixels, every one of them in the annulus, against a cap of
        // 10,000,000.
        let g = geom(4000, 2501);
        assert_eq!(g.pixels(), 10_004_000);
        assert!(g.pixels() > MAX_ROTATING_BAR_EVENTS);
        // A period of one second and a recording of a microsecond: `floor(duration / period)` is
        // zero, so the budget is exactly "one crossing per annulus pixel".
        let bar = |duration: f64| {
            rotating_bar(g, (0.5, 0.5), core::f64::consts::PI, 0.0, 0.5, 1e4, duration, Polarity::On)
        };
        assert_eq!(
            bar(1e-6).err(),
            Some(VisionError::BadParameters {
                why: "the angular velocity and duration ask for more rotating-bar events than \
                      MAX_ROTATING_BAR_EVENTS allows",
            })
        );
        // And the same budget on a sensor one pixel under the cap is produced rather than refused,
        // so the refusal above is the count and not the sensor.
        let small = geom(4000, 2500);
        assert_eq!(small.pixels(), 10_000_000);
        assert!(
            rotating_bar(small, (0.5, 0.5), core::f64::consts::PI, 0.0, 0.5, 1e4, 1e-6, Polarity::On)
                .is_ok()
        );
    }

    /// **The annulus is closed at both ends: a pixel exactly at `r_min` is on the bar.**
    ///
    /// Every rotating-bar fixture in this module centres the bar on a **half-integer** pixel
    /// centre, where `r^2 = (i + 0.5)^2 + (j + 0.5)^2` always ends in `.5` and is never the square
    /// of an integer radius — so no pixel ever sits exactly on `r_min` or `r_max`, and both
    /// comparisons could face either way unnoticed. An integer centre puts four pixels exactly on
    /// each boundary.
    ///
    /// The emitted pixel set is compared against the closed form `r_min <= r <= r_max`, which is
    /// the inclusive range the generator's own annulus count uses; the two disagreeing is how a
    /// pixel comes to be counted against the event budget and then never fired.
    #[test]
    fn the_rotating_bars_annulus_is_inclusive_at_both_of_its_radii() {
        let g = geom(33, 33);
        let centre = (16.0, 16.0);
        let (r_min, r_max) = (4.0, 8.0);
        // 1.43 half-turns, so every annulus pixel is crossed at least once whatever its angle.
        let evs = rotating_bar(g, centre, 9.0, 0.3, r_min, r_max, 0.5, Polarity::On).unwrap();

        let radius = |x: u16, y: u16| (f64::from(x) - centre.0).hypot(f64::from(y) - centre.1);
        let mut want: Vec<(u16, u16)> = Vec::new();
        let (mut on_inner, mut on_outer) = (0usize, 0usize);
        for y in 0..33u16 {
            for x in 0..33u16 {
                let r = radius(x, y);
                if (r_min..=r_max).contains(&r) {
                    want.push((x, y));
                }
                if r == r_min {
                    on_inner += 1;
                }
                if r == r_max {
                    on_outer += 1;
                }
            }
        }
        // The fixture is not vacuous: four pixels sit exactly on each boundary.
        assert_eq!(on_inner, 4, "no pixel is exactly at r_min, so the boundary is untested");
        assert_eq!(on_outer, 4, "no pixel is exactly at r_max, so the boundary is untested");

        let mut got: Vec<(u16, u16)> = evs.iter().map(|e| (e.x, e.y)).collect();
        got.sort_unstable();
        got.dedup();
        want.sort_unstable();
        assert_eq!(got, want, "the fired set is not the closed annulus r_min <= r <= r_max");
        // Named explicitly, so the failure message says which boundary moved.
        for (x, y) in [(20u16, 16u16), (12, 16), (16, 20), (16, 12)] {
            assert!(evs.iter().any(|e| (e.x, e.y) == (x, y)), "({x}, {y}) is at exactly r_min");
        }
        for (x, y) in [(24u16, 16u16), (8, 16), (16, 24), (16, 8)] {
            assert!(evs.iter().any(|e| (e.x, e.y) == (x, y)), "({x}, {y}) is at exactly r_max");
        }
    }

    /// **The looming disc's radius is measured from the centre's column AND its row.**
    ///
    /// Every looming fixture in this module puts the centre on the diagonal — `(19.5, 19.5)`,
    /// `(7.5, 7.5)` — where `cx` and `cy` are the same number and reading one in place of the
    /// other changes nothing at all. Off the diagonal it is a different stimulus: the disc expands
    /// about `(cx, cx)` instead, and every crossing time is wrong by the difference.
    #[test]
    fn the_looming_discs_radius_is_measured_from_both_centre_components() {
        let g = geom(40, 40);
        let centre = (12.0, 27.0);
        let (r0, rate, dur) = (2.0, 150.0, 0.12);
        let evs = looming_disc(g, centre, r0, rate, dur, Polarity::On).unwrap();
        assert!(evs.len() > 400, "only {} events", evs.len());

        // Every event satisfies r = r0 + rate * t with r measured from (cx, cy) ...
        for e in &evs {
            let r = (f64::from(e.x) - centre.0).hypot(f64::from(e.y) - centre.1);
            assert!(
                (r - (r0 + rate * e.t_s)).abs() < 1e-12,
                "({}, {}) at t = {} is at r = {r}",
                e.x,
                e.y,
                e.t_s
            );
        }
        // ... and the fired set is exactly the pixels whose crossing time is inside the recording.
        let mut want = 0usize;
        for y in 0..40u16 {
            for x in 0..40u16 {
                let r = (f64::from(x) - centre.0).hypot(f64::from(y) - centre.1);
                if (0.0..=dur).contains(&((r - r0) / rate)) {
                    want += 1;
                }
            }
        }
        assert_eq!(evs.len(), want);
        // The two components are 15 pixels apart, so the pixel directly below the centre and the
        // pixel directly right of it fire at times that differ by the asymmetry the mutation
        // erases: reading the column twice would put both on the same circle.
        let time_at = |x: u16, y: u16| evs.iter().find(|e| e.x == x && e.y == y).map(|e| e.t_s);
        let right = time_at(22, 27).expect("ten pixels right of the centre");
        let below = time_at(12, 32).expect("five pixels below the centre");
        assert!((right - (10.0 - r0) / rate).abs() < 1e-12);
        assert!((below - (5.0 - r0) / rate).abs() < 1e-12);
        assert!(right > below, "the nearer pixel must fire first");
    }

    /// **The rotating bar emits exactly the crossings its closed form names — every one of them,
    /// and no others.**
    ///
    /// [`the_rotating_bar_generator_puts_every_event_on_the_bar`] checks that each emitted event is
    /// *on* the bar and that no pixel fires more than twice in 1.43 half-turns; neither statement
    /// can see a crossing that was never emitted, and "at least one pixel fired twice" is satisfied
    /// by one pixel out of several hundred. The generator's loop is bounded by a slot count rather
    /// than by the duration alone, so "did the loop have enough slots" is a question only a
    /// per-pixel comparison against `t_k = base + k * period` can answer.
    ///
    /// The comparison is an exact equality, value for value: the test evaluates the same closed
    /// form with the same three quantities in the same order, so every emitted time is bit-for-bit
    /// the one the form names or the assertion fails.
    #[test]
    fn every_crossing_the_rotating_bars_closed_form_names_is_emitted_and_no_other() {
        let g = geom(21, 21);
        let centre = (10.0, 10.0);
        let (omega, theta0) = (9.0, 0.3);
        let (r_min, r_max, dur) = (3.0, 9.0, 0.5);
        let evs = rotating_bar(g, centre, omega, theta0, r_min, r_max, dur, Polarity::On).unwrap();
        assert!(evs.len() > 300, "only {} events", evs.len());

        let period = core::f64::consts::PI / omega;
        let mut checked = 0usize;
        let mut total = 0usize;
        for y in 0..21u16 {
            for x in 0..21u16 {
                let (dx, dy) = (f64::from(x) - centre.0, f64::from(y) - centre.1);
                let r = dx.hypot(dy);
                let mut got: Vec<f64> =
                    evs.iter().filter(|e| e.x == x && e.y == y).map(|e| e.t_s).collect();
                if !(r_min..=r_max).contains(&r) {
                    assert!(got.is_empty(), "({x}, {y}) is off the annulus and fired {got:?}");
                    continue;
                }
                let base = (dy.atan2(dx) - theta0) / omega;
                // Every k the closed form admits, scanned two slots past each end so that a
                // generator emitting too many would be caught as well as one emitting too few.
                let lo = ((0.0 - base) / period).floor() - 2.0;
                let hi = ((dur - base) / period).ceil() + 2.0;
                let mut want: Vec<f64> = Vec::new();
                let mut k = lo;
                while k <= hi {
                    let t = base + k * period;
                    if (0.0..=dur).contains(&t) {
                        want.push(t);
                    }
                    k += 1.0;
                }
                got.sort_by(|a, b| a.partial_cmp(b).unwrap());
                assert_eq!(got, want, "pixel ({x}, {y})");
                assert!(!want.is_empty(), "an annulus pixel with no crossing in 1.43 half-turns");
                checked += 1;
                total += want.len();
            }
        }
        assert_eq!(total, evs.len());
        assert!(checked > 200, "only {checked} annulus pixels were compared");
    }
}
