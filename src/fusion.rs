//! Multimodal fusion: more than one sense, in spikes.
//!
//! A robot with a camera, a microphone, an inertial unit and skin has four clocks, four sampling
//! rates and four notions of "now". Fusing them is where event-based sensing is at its strongest,
//! because asynchronous streams at unrelated rates are exactly what a spiking substrate handles
//! natively and what a frame-based one has to resample onto a common grid first — and resampling is
//! where the timing information that makes fusion work gets thrown away.
//!
//! # The lesson, in the order the problems arrive
//!
//! **1. Nothing is aligned.** Each sensor timestamps in its own clock. Two clocks differ by an
//! *offset* (their epochs are not the same instant) and by a *drift* (their seconds are not the
//! same length). A 200 ppm crystal — an ordinary part — moves 4 ms in 20 seconds, which is larger
//! than every temporal window in the rest of this module. So the first thing a fusion stack does is
//! estimate the offset, and the estimator is [`OffsetEstimator`]: a coarse histogram of pairwise
//! timestamp differences to find the peak, then the **median** of the differences near that peak.
//! The median rather than the mean because dropouts and background events are outliers, and a mean
//! is not robust to them. This is the same shape as the robust clock estimators in the network
//! measurement literature — Paxson, *On Calibrating Measurements of Packet Transit Times*,
//! `SIGMETRICS` 1998, and Moon, Skelly & Towsley, *Estimation and Removal of Clock Skew from
//! Network Delay Measurements*, IEEE `INFOCOM` 1999 — rather than a transcription of either.
//!
//! ⚠ **A constant transport lag and a clock offset are not separable from coincidence data alone.**
//! If the microphone's clock is 5 ms ahead and sound takes 5 ms to arrive, the observations are
//! identical to a clock 10 ms ahead with instant arrival. [`OffsetEstimator`] recovers the **sum**,
//! and it says so rather than pretending to attribute it. Separating them needs a second
//! measurement the events do not contain — a known geometry, or a round trip in the style of
//! Cristian, *Probabilistic Clock Synchronization*, Distributed Computing 3:146–158, 1989, and
//! Mills, *Internet Time Synchronization: the Network Time Protocol*, IEEE Trans. Comm.
//! 39(10):1482–1493, 1991.
//!
//! **2. Binding is a window, and the window is a parameter.** Once the streams share a timeline,
//! "these two events are about the same thing" is a statement about *synchrony*: the binding-by-
//! synchrony hypothesis of von der Malsburg, *The Correlation Theory of Brain Function*, MPI
//! internal report 81-2, 1981, and Singer & Gray, Annu. Rev. Neurosci. 18:555–586, 1995. The
//! biology fixes the tolerance empirically: Meredith, Nemitz & Stein (J. Neurosci. 7:3215–3229,
//! 1987) mapped the temporal window of multisensory neurons in the cat superior colliculus and
//! found response enhancement over offsets of roughly 100 ms, widest near stimulus onset alignment.
//! So [`CoincidenceDetector::half_window_s`] is a field with a unit, not a constant in the code.
//!
//! And a window has a **false-positive rate that is computable before you run anything**. Two
//! independent Poisson streams will look bound by accident; for a partner at rate `r` and a
//! half-window `w`, the expected partners per anchor is exactly `2*w*r` and the probability of at
//! least one is exactly `1 - exp(-2*w*r)`. [`chance_partners_per_anchor`] and
//! [`chance_binding_probability`] are those two lines, and the tests assert the detector against
//! them rather than against a plot.
//!
//! **3. Which sense wins is a precision question.** Ernst & Banks (Nature 415:429–433, 2002) showed
//! that humans combine visual and haptic size estimates with weights proportional to the inverse of
//! each channel's variance, which is the maximum-likelihood combination, and Alais & Burr (Curr.
//! Biol. 14:257–262, 2004) showed the ventriloquist effect is the same rule with vision degraded
//! until hearing wins. [`PrecisionGate`] is that rule: weights `∝ 1/σ²`, fused variance
//! `1/Σ(1/σ²)`. It is also the honest answer to "what happens when a modality drops out" —
//! the fused variance rises by exactly `Σ_all(1/σ²) / Σ_kept(1/σ²)` and nothing else changes. Two
//! equally good senses, one lost: the variance doubles. That is degradation, not failure, and it is
//! a number rather than an adjective.
//!
//! **4. Early and late fusion are not two implementations of one idea.** *Late* fusion decides per
//! modality and combines the decisions; *early* fusion decides once from the joint pattern. The
//! difference is not style. A late branch can only see features of its own stream, so **it cannot
//! see whether two streams fired together** in any way its combiner can express — and on a task
//! whose label is exactly that, late fusion is at chance no matter how good each branch is.
//! [`SynchronyTask`] is such a task, and the test asserts the direction. The boundary is sharper
//! than "late fusion cannot see across streams": a combiner that sums per-branch scores is a
//! *linear* function of both streams and can reconstruct a linear cross-modal quantity such as a
//! signed interval. What it cannot reconstruct is a **nonlinear** one such as `|t_a - t_b|`, and
//! that type's doc records the measurement that forced the distinction. The price is symmetric and this module measures that too: early
//! fusion's joint features are undefined when a stream is missing, so the same task that early
//! fusion wins intact, it loses entirely when one modality drops. [`RedundantRateTask`] is the
//! opposite case, where the label is in each modality separately and the gap closes.
//!
//! # Units
//!
//! Every interface is SI. Timestamps inside a [`Stream`] are `u64` ticks in that stream's **own**
//! clock, multiplied by [`Clock::dt_s`] seconds at the point of use, for the reason
//! [`crate::spike`] gives. Once aligned, [`AlignedEvent::t_s`] is a real number of seconds: a clock
//! offset is not an integer number of anybody's ticks, and rounding the aligned timeline back onto
//! a grid would reintroduce the quantisation the alignment just removed. A consumer that needs
//! ticks quantises explicitly, at a `dt` it chooses.
//!
//! # Quickstart: two clocks, one world
//!
//! ```
//! use ferromorphic::{
//!     fusion::{
//!         Aligned, CoincidenceDetector, ModalitySpec, MultimodalSource, OffsetEstimator,
//!         chance_binding_probability,
//!     },
//!     rng::Rng,
//! };
//!
//! // An event camera ticking at 100 us and a microphone at 1 ms, watching the same events.
//! // The microphone's clock is 7 ms ahead of the camera's, which nothing downstream is told.
//! let camera = ModalitySpec {
//!     dt_s: 1e-4,
//!     lag_s: 0.0,
//!     clock_offset_s: 0.050,
//!     jitter_s: 2e-4,
//!     dropout: 0.0,
//!     background_hz: 3.0,
//!     drift: 0.0,
//! };
//! let microphone = ModalitySpec { dt_s: 1e-3, clock_offset_s: 0.057, ..camera };
//! let source = MultimodalSource {
//!     source_rate_hz: 20.0,
//!     duration_s: 30.0,
//!     addresses: 8,
//!     modalities: vec![camera, microphone],
//!     event_cap: 1_000_000,
//! };
//! let rec = source.generate(&mut Rng::new(7))?;
//!
//! // 1. Recover the offset from the event times alone, and check it against the truth.
//! let est = OffsetEstimator::new(0.050, 1e-3, 6e-3)?;
//! let fit = est.estimate(&rec.streams[0], &rec.streams[1])?;
//! assert!((fit.offset_s - 7e-3).abs() < 1e-3, "recovered {}", fit.offset_s);
//!
//! // 2. Put the microphone on the camera's timeline.
//! let mut mic = rec.streams[1].clone();
//! mic.clock.offset_s = fit.offset_s;
//! let aligned = Aligned::merge(&[rec.streams[0].clone(), mic])?;
//!
//! // 3. Bind by synchrony — and price the answer against what the window binds by accident.
//! let detector = CoincidenceDetector::new(3e-3, 0, vec![1])?;
//! let report = detector.detect(&aligned, 0.0, 30.0)?;
//! let mic_rate = aligned.rate_hz_in(1, 0.0, 30.0).expect("a window with length");
//! let by_chance = chance_binding_probability(3e-3, &[mic_rate])?;
//! assert!(report.binding_rate().unwrap() > 2.0 * by_chance);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # What this module does not have
//!
//! This is a synthetic-generator module, not a dataset loader. Real multimodal event corpora exist
//! — `N-TIDIGITS18` (Anumula, Neil, Delbruck & Liu, Front. Neurosci. 12:23, 2018) pairs spoken
//! digits with silicon-cochlea spikes, and `MVSEC` (Zhu et al., IEEE RA-L 3(3):2032–2039, 2018)
//! pairs event cameras with lidar, `IMU` and `GPS` — and decoding them belongs in [`crate::aer`] and
//! [`crate::tasks`], not here. This review **did not locate** an open benchmark that scores
//! cross-modal *spike timing* fusion specifically, as opposed to accuracy on a task that happens to
//! have two input streams; if one exists, the generator here should be replaced by it.
//!
//! The estimators here also assume each sensor's jitter is **symmetric about its true time**. A
//! sensor whose latency distribution has a tail on one side — which is the normal case for a
//! network-attached camera — biases the median by roughly the tail's asymmetry, and this
//! implementation does not correct for it and does not detect it.

use core::fmt;

use crate::metrics::{self, MetricError};
use crate::plasticity::{PairStdp, PlasticityError};
use crate::rng::Rng;
use crate::spike::{Event, Polarity};

// =================================================================================================
// Errors
// =================================================================================================

/// What a fusion call refuses on, and what was wrong.
#[derive(Debug, Clone, PartialEq)]
pub enum FusionError {
    /// A parameter was `NaN` or infinite. Rejected at the boundary: a non-finite timestamp sorts
    /// unpredictably and then poisons every median, centroid and weight downstream of it.
    NonFinite {
        /// Which parameter carried it.
        what: &'static str,
    },
    /// A quantity that must be strictly positive — a tick length, a window width — was not.
    NotPositive {
        /// Which parameter.
        what: &'static str,
        /// The value supplied.
        value: f64,
    },
    /// A quantity that must be non-negative — a jitter half-width, a rate — was negative.
    Negative {
        /// Which parameter.
        what: &'static str,
        /// The value supplied.
        value: f64,
    },
    /// A probability was outside `0.0..=1.0`.
    NotAProbability {
        /// Which parameter.
        what: &'static str,
        /// The value supplied.
        value: f64,
    },
    /// A fusion of no modalities was requested. The fused estimate of nothing is not zero.
    NoModalities,
    /// A modality index named no stream.
    UnknownModality {
        /// The index asked for.
        modality: usize,
        /// How many there are.
        count: usize,
    },
    /// A sensor's own clock would have to report a time before its zero.
    ///
    /// The generator refuses rather than clipping, because clipping deletes exactly the earliest
    /// events and so biases every offset estimate computed from the result.
    NegativeEpoch {
        /// Which modality.
        modality: usize,
        /// The combined lag plus clock offset that pushed it negative, seconds.
        shift_s: f64,
    },
    /// A stream's events were not in non-decreasing tick order.
    ///
    /// The invariant is load-bearing: the pair sweep and every binary search in this module assume
    /// it, and an unsorted stream produces a smaller pair count rather than an error.
    Unsorted {
        /// Index of the first event that went backwards.
        index: usize,
    },
    /// Fewer timestamp pairs fell in the search window than the estimator was told to require.
    TooFewPairs {
        /// How many were found.
        found: usize,
        /// How many were required.
        needed: usize,
    },
    /// The pairwise sweep exceeded its cap.
    ///
    /// Pair enumeration is `O(n_a * r_b * 2W)`; on two 1 `MHz` event streams with a wide search
    /// window it is a memory exhaustion rather than a slow run, so it is capped and reported.
    TooManyPairs {
        /// The cap that was hit.
        cap: usize,
    },
    /// A Poisson draw produced more events than its cap allowed, which means the rate and duration
    /// asked for more memory than the caller can have meant.
    EventCap {
        /// The cap that was hit.
        cap: usize,
    },
    /// A tick index would not fit in a `u64` at the stream's `dt`.
    TickOverflow {
        /// The reported time that overflowed, seconds.
        t_s: f64,
    },
    /// Two arrays that must be the same length were not.
    LengthMismatch {
        /// Length of the first.
        a: usize,
        /// Length of the second.
        b: usize,
    },
    /// A classifier or a statistic was asked of an empty set.
    Empty {
        /// Which set.
        what: &'static str,
    },
    /// A classifier was fitted on trials that did not all have the same feature layout.
    RaggedFeatures {
        /// The trial index that disagreed.
        index: usize,
    },
    /// A classifier was fitted on a single class, so there is nothing to discriminate.
    SingleClass {
        /// The only label present.
        label: u32,
    },
    /// Every feature was masked out, so there is no space left to measure a distance in.
    AllFeaturesDropped,
    /// A drift estimate needs two separated halves and the record did not provide them.
    NoLeverArm {
        /// The separation between the two half-record centroids, seconds.
        span_s: f64,
    },
    /// A plasticity call refused. Forwarded rather than flattened so the original reason survives.
    Plasticity(PlasticityError),
    /// A metric call refused. Forwarded for the same reason.
    Metric(MetricError),
}

impl fmt::Display for FusionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFinite { what } => write!(f, "{what} was not finite"),
            Self::NotPositive { what, value } => write!(f, "{what} must be positive, got {value}"),
            Self::Negative { what, value } => {
                write!(f, "{what} must be non-negative, got {value}")
            }
            Self::NotAProbability { what, value } => {
                write!(f, "{what} must be in 0..=1, got {value}")
            }
            Self::NoModalities => f.write_str("no modalities to fuse"),
            Self::UnknownModality { modality, count } => {
                write!(f, "modality {modality} named, only {count} present")
            }
            Self::NegativeEpoch { modality, shift_s } => write!(
                f,
                "modality {modality} would report a time before its own clock zero (shift {shift_s} s)"
            ),
            Self::Unsorted { index } => write!(f, "event {index} went backwards in time"),
            Self::TooFewPairs { found, needed } => {
                write!(f, "only {found} timestamp pairs in the window, {needed} required")
            }
            Self::TooManyPairs { cap } => write!(f, "pair sweep exceeded its cap of {cap}"),
            Self::EventCap { cap } => write!(f, "Poisson draw exceeded its cap of {cap} events"),
            Self::TickOverflow { t_s } => write!(f, "reported time {t_s} s does not fit in a tick"),
            Self::LengthMismatch { a, b } => write!(f, "length {a} does not match {b}"),
            Self::Empty { what } => write!(f, "{what} was empty"),
            Self::RaggedFeatures { index } => {
                write!(f, "trial {index} has a different feature layout from trial 0")
            }
            Self::SingleClass { label } => {
                write!(f, "every trial carries label {label}; there is nothing to discriminate")
            }
            Self::AllFeaturesDropped => f.write_str("every feature was masked out"),
            Self::NoLeverArm { span_s } => {
                write!(f, "the two half-records are {span_s} s apart; drift needs a lever arm")
            }
            Self::Plasticity(e) => write!(f, "plasticity refused: {e}"),
            Self::Metric(e) => write!(f, "metric refused: {e}"),
        }
    }
}

impl std::error::Error for FusionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Plasticity(e) => Some(e),
            Self::Metric(e) => Some(e),
            _ => None,
        }
    }
}

impl From<PlasticityError> for FusionError {
    fn from(e: PlasticityError) -> Self {
        Self::Plasticity(e)
    }
}

impl From<MetricError> for FusionError {
    fn from(e: MetricError) -> Self {
        Self::Metric(e)
    }
}

fn finite(what: &'static str, x: f64) -> Result<f64, FusionError> {
    if x.is_finite() { Ok(x) } else { Err(FusionError::NonFinite { what }) }
}

fn positive(what: &'static str, x: f64) -> Result<f64, FusionError> {
    let x = finite(what, x)?;
    if x > 0.0 { Ok(x) } else { Err(FusionError::NotPositive { what, value: x }) }
}

fn non_negative(what: &'static str, x: f64) -> Result<f64, FusionError> {
    let x = finite(what, x)?;
    if x >= 0.0 { Ok(x) } else { Err(FusionError::Negative { what, value: x }) }
}

fn probability(what: &'static str, x: f64) -> Result<f64, FusionError> {
    let x = finite(what, x)?;
    if (0.0..=1.0).contains(&x) {
        Ok(x)
    } else {
        Err(FusionError::NotAProbability { what, value: x })
    }
}

/// A standard normal deviate, by the Box–Muller transform.
///
/// Mean 0, variance 1. Deterministic in the seed like everything else in this crate, and present
/// because modelling a sensor for [`PrecisionGate`] needs a noise source with a *stated* variance
/// and [`crate::rng`] supplies only uniforms. One draw costs two uniforms and one `cos`; the second
/// Box–Muller output is discarded rather than cached, so that the number of uniforms consumed per
/// call is constant and a caller can reason about stream position.
///
/// The `1.0 - u` is not cosmetic: [`Rng::next_f64`] returns `[0, 1)`, and `ln(0)` is an infinity.
#[must_use]
pub fn gaussian(rng: &mut Rng) -> f64 {
    let u1 = 1.0 - rng.next_f64();
    let u2 = rng.next_f64();
    (-2.0 * u1.ln()).sqrt() * (core::f64::consts::TAU * u2).cos()
}

/// Homogeneous Poisson arrival times in `[t0, t1)`, by exponential inter-arrivals.
///
/// Continuous time, not a per-tick Bernoulli draw: the closed forms this module asserts against
/// are continuous-time Poisson results, and a Bernoulli approximation would differ from them by
/// `O(r*dt)` and turn a sharp test into a loose one.
fn poisson_times(
    rng: &mut Rng,
    rate_hz: f64,
    t0: f64,
    t1: f64,
    cap: usize,
) -> Result<Vec<f64>, FusionError> {
    let rate = non_negative("rate_hz", rate_hz)?;
    let mut out = Vec::new();
    if rate == 0.0 || !(t1 > t0) {
        return Ok(out);
    }
    let mut t = t0;
    loop {
        // -ln(1 - u) with u in [0, 1) is Exp(1); u == 0 gives a zero gap, which advances nothing
        // but cannot recur indefinitely. The cap is the guard against a caller's rate, not against
        // the draw.
        t += -(1.0 - rng.next_f64()).ln() / rate;
        if t >= t1 {
            return Ok(out);
        }
        if out.len() >= cap {
            return Err(FusionError::EventCap { cap });
        }
        out.push(t);
    }
}

/// Median of a finite slice, by sorting a copy. `None` for an empty slice.
fn median(v: &[f64]) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    let n = s.len();
    Some(if n % 2 == 1 { s[n / 2] } else { 0.5 * (s[n / 2 - 1] + s[n / 2]) })
}

// =================================================================================================
// Clocks and streams
// =================================================================================================

/// One sensor's clock, as a map from its own ticks onto the common timeline.
///
/// The model, stated once so every estimator in this module can be checked against it:
///
/// ```text
/// t_reported = t_common * (1 + drift) + offset_s
/// ```
///
/// `t_reported` is `tick * dt_s` seconds. A stream chosen as the reference has `offset_s = 0.0` and
/// `drift = 0.0`, and then its reported time *is* common time.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Clock {
    /// Seconds per tick of this sensor's own counter. Strictly positive.
    pub dt_s: f64,
    /// Where this clock's zero sits on the common timeline, seconds. Positive means this sensor
    /// reports later than common time; it absorbs both the clock epoch and any constant transport
    /// lag, which coincidence data cannot separate.
    pub offset_s: f64,
    /// Fractional rate error, seconds per second. `2e-4` is a 200 ppm crystal. Strictly greater
    /// than `-1.0`, since `1 + drift` is a divisor and a clock that runs backwards is not modelled.
    pub drift: f64,
}

impl Clock {
    /// A clock with the given tick length, no offset and no drift — i.e. the reference.
    ///
    /// # Errors
    ///
    /// [`FusionError::NotPositive`] for a `dt_s` that is not strictly positive, and
    /// [`FusionError::NonFinite`] for a non-finite one.
    pub fn new(dt_s: f64) -> Result<Self, FusionError> {
        Ok(Self { dt_s: positive("dt_s", dt_s)?, offset_s: 0.0, drift: 0.0 })
    }

    /// A clock with a known offset and drift.
    ///
    /// # Errors
    ///
    /// As [`Clock::new`], plus [`FusionError::NonFinite`] for a non-finite offset or drift and
    /// [`FusionError::NotPositive`] on `1 + drift` when `drift <= -1.0`.
    pub fn with_offset(dt_s: f64, offset_s: f64, drift: f64) -> Result<Self, FusionError> {
        let dt_s = positive("dt_s", dt_s)?;
        let offset_s = finite("offset_s", offset_s)?;
        let drift = finite("drift", drift)?;
        positive("1 + drift", 1.0 + drift)?;
        Ok(Self { dt_s, offset_s, drift })
    }

    /// This sensor's reported time for a tick, in its own frame, seconds.
    #[must_use]
    pub fn reported_s(&self, tick: u64) -> f64 {
        tick as f64 * self.dt_s
    }

    /// A tick mapped onto the common timeline, seconds.
    ///
    /// The inverse of the model on the type: `(tick*dt - offset) / (1 + drift)`. With the default
    /// offset and drift this is exactly `tick * dt_s` — no rounding, no scaling — which is what
    /// makes alignment of an unshifted stream the identity rather than approximately the identity.
    #[must_use]
    pub fn to_common_s(&self, tick: u64) -> f64 {
        if self.offset_s == 0.0 && self.drift == 0.0 {
            return self.reported_s(tick);
        }
        (self.reported_s(tick) - self.offset_s) / (1.0 + self.drift)
    }
}

/// One modality's event stream, timestamped in its own clock.
#[derive(Debug, Clone, PartialEq)]
pub struct Stream {
    /// Which sense this is. Indices are the caller's; [`MultimodalSource`] numbers them from zero
    /// in the order the specs were given.
    pub modality: u16,
    /// The clock that turns this stream's ticks into common-timeline seconds.
    pub clock: Clock,
    /// Events in non-decreasing tick order — an invariant [`Stream::new`] checks and every sweep in
    /// this module depends on.
    events: Vec<Event>,
}

impl Stream {
    /// Build a stream, checking the sort invariant.
    ///
    /// # Errors
    ///
    /// [`FusionError::Unsorted`] naming the first event whose tick went backwards.
    pub fn new(modality: u16, clock: Clock, events: Vec<Event>) -> Result<Self, FusionError> {
        for i in 1..events.len() {
            if events[i].t < events[i - 1].t {
                return Err(FusionError::Unsorted { index: i });
            }
        }
        Ok(Self { modality, clock, events })
    }

    /// The events, in tick order.
    #[must_use]
    pub fn events(&self) -> &[Event] {
        &self.events
    }

    /// How many events the stream holds.
    #[must_use]
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Whether the stream is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Reported times in this sensor's own frame, seconds, in order.
    ///
    /// This is what an offset estimator works in: the whole point is that the common timeline is
    /// not yet known.
    #[must_use]
    pub fn reported_s(&self) -> Vec<f64> {
        self.events.iter().map(|e| self.clock.reported_s(e.t)).collect()
    }

    /// A sub-stream of the events whose index is in `range`, sharing this stream's clock.
    fn slice(&self, from: usize, to: usize) -> Self {
        Self {
            modality: self.modality,
            clock: self.clock,
            events: self.events[from..to].to_vec(),
        }
    }
}

// =================================================================================================
// Offset and drift estimation
// =================================================================================================

/// Recovers the constant offset between two streams' clocks from their event times alone.
///
/// # How
///
/// Every pair of timestamps `(t_a, t_b)` whose difference falls inside `±search_half_s` votes into
/// a histogram of bin width `bin_s`. Events that came from the same physical cause pile into one
/// bin; accidental pairs spread out. The tallest bin is the coarse estimate. The fine estimate is
/// the **median** of the differences within `±refine_half_s` of it.
///
/// # Choosing the three widths
///
/// - `search_half_s` must exceed the true offset, or the peak is outside the histogram and the
///   answer is background.
/// - `bin_s` must be wider than the true pairs' spread or the peak splits across bins, and narrower
///   than `refine_half_s` or the coarse step buys nothing.
/// - `refine_half_s` must cover the true cluster — roughly the summed jitter half-widths plus
///   `bin_s/2` — and no more, because everything it admits beyond the cluster is background.
///
/// # What tolerance to expect
///
/// For symmetric jitter with half-widths `J_a`, `J_b`, the pairwise difference has density
/// `1/(2*max(J_a, J_b))` at its centre, so the standard error of the median over `n` true pairs is
/// `max(J_a, J_b) / sqrt(n)`. Tick quantisation adds `dt/2` to each half-width. Symmetric
/// background contamination inflates the spread but does not bias the median, which is the reason
/// for the median in the first place.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OffsetEstimator {
    /// Half-width of the offsets searched, seconds. Strictly positive.
    pub search_half_s: f64,
    /// Coarse histogram bin width, seconds. Strictly positive, at most `search_half_s`.
    pub bin_s: f64,
    /// Half-width of the median refinement around the coarse peak, seconds.
    pub refine_half_s: f64,
    /// Refuse below this many differences in the refinement window. A median of three pairs is a
    /// number, not an estimate.
    pub min_pairs: usize,
    /// Cap on the pair sweep, as a guard against an enumeration that is quadratic in the event rate.
    pub max_pairs: usize,
}

/// What [`OffsetEstimator::estimate`] found.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OffsetFit {
    /// The estimated offset, seconds: **b's reported time minus a's reported time** for the same
    /// physical event. Feeds [`Clock::offset_s`] for b when a is the reference.
    pub offset_s: f64,
    /// The tallest histogram bin's centre, seconds — the estimate before refinement.
    pub coarse_s: f64,
    /// How many differences the median was taken over.
    pub pairs: usize,
    /// Median absolute deviation of those differences, seconds. A robust spread: the cluster's
    /// width, not the search window's.
    pub mad_s: f64,
    /// How many distinct `a` events contributed to those differences — the number of independent
    /// things the median was estimated from, which is smaller than [`OffsetFit::pairs`] whenever a
    /// neighbouring source event also fell inside the refinement window.
    pub events: usize,
}

impl OffsetFit {
    /// Standard error of the offset, seconds: `2 · mad_s / sqrt(events)`.
    ///
    /// # The closed form, and its bound
    ///
    /// The asymptotic standard error of a median over `n` independent samples is
    /// `1 / (2 · f(0) · √n)`, which needs the density at the centre. `2 · mad_s` stands in for
    /// `1 / (2 f(0))`, and the substitution is exact for a uniform difference and conservative for
    /// the peaked shapes: relative to the exact expression it is **1.00** (uniform), **1.17**
    /// (triangular — two uniform jitters), **1.08** (normal). It never under-reports for a symmetric
    /// shape, which is the direction an error bar must err in.
    ///
    /// `events`, not `pairs`, is the `n`: the extra differences a neighbouring source event
    /// contributes are consequences of the same events, not independent draws.
    ///
    /// # What it measured as, and why it is not exactly 1
    ///
    /// Under the protocol that found the previous formula wrong — 20 Hz Poisson source, 1 ms
    /// jitter, 0.1 ms ticks, 60 s, 5 ms lag, 300 seeds, backgrounds of 0, 1 and 5 Hz — this figure
    /// is **1.2 to 1.3 times** the true seed-to-seed scatter of the estimate. The excess is
    /// identified: the measured `mad_s` (7.8e-4 s) is 1.27 × the pure-triangular value
    /// `(2 − √2)·J`, inflated by the cross-pair differences that share the window. Over by that
    /// much, for that reason, in the safe direction. The calibration test asserts the band.
    ///
    /// # What shipped before
    ///
    /// `mad_s / sqrt(pairs)`, with a doc claiming `mad_s ≈ 1/(2 f(0))` "to about 25%". For a
    /// triangular difference `mad_s = 0.586 J` against an exact `1.0 J` — 41% low before anything
    /// else — and `pairs` over-counted the independent sample by ~32%. Measured: **0.37** of the true
    /// scatter, and nothing in the suite could see the formula. `None` when no event contributed.
    #[must_use]
    pub fn standard_error_s(&self) -> Option<f64> {
        if self.events == 0 { None } else { Some(2.0 * self.mad_s / (self.events as f64).sqrt()) }
    }
}

/// What [`OffsetEstimator::estimate_drift`] found: the two-parameter clock fit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DriftFit {
    /// Offset extrapolated back to common time zero, seconds. Feeds [`Clock::offset_s`].
    pub offset_s: f64,
    /// Fractional rate error of b against a, seconds per second. Feeds [`Clock::drift`].
    pub drift: f64,
    /// Offset measured on the first half of the record, seconds.
    pub first_half_s: f64,
    /// Offset measured on the second half, seconds.
    pub second_half_s: f64,
    /// Separation between the two halves' centroid times, seconds — the lever arm the drift
    /// estimate divides by, and therefore the thing that sets its precision.
    pub span_s: f64,
}

impl OffsetEstimator {
    /// Build an estimator, checking the three widths against each other.
    ///
    /// `min_pairs` defaults to 16 and `max_pairs` to 20 million; set the fields to change them.
    ///
    /// # Errors
    ///
    /// [`FusionError::NotPositive`] or [`FusionError::NonFinite`] for any width that is not a
    /// positive finite number, and [`FusionError::NotPositive`] on `search_half_s - bin_s` or
    /// `search_half_s - refine_half_s` when a sub-width exceeds the search window it lives in.
    pub fn new(search_half_s: f64, bin_s: f64, refine_half_s: f64) -> Result<Self, FusionError> {
        let search_half_s = positive("search_half_s", search_half_s)?;
        let bin_s = positive("bin_s", bin_s)?;
        let refine_half_s = positive("refine_half_s", refine_half_s)?;
        if bin_s > search_half_s {
            return Err(FusionError::NotPositive {
                what: "search_half_s - bin_s",
                value: search_half_s - bin_s,
            });
        }
        if refine_half_s > search_half_s {
            return Err(FusionError::NotPositive {
                what: "search_half_s - refine_half_s",
                value: search_half_s - refine_half_s,
            });
        }
        Ok(Self { search_half_s, bin_s, refine_half_s, min_pairs: 16, max_pairs: 20_000_000 })
    }

    /// Every pairwise difference `t_b - t_a` inside the search window, in sweep order.
    fn differences(&self, a: &[f64], b: &[f64]) -> Result<Vec<f64>, FusionError> {
        Ok(self.differences_indexed(a, b)?.into_iter().map(|(_, d)| d).collect())
    }

    /// As [`Self::differences`], each difference tagged with the index of the `a` event that
    /// produced it. Contiguous by that index, because the sweep emits every partner of one `a`
    /// event before moving to the next.
    ///
    /// The tag is what makes an honest error bar possible. One source event yields one true pair
    /// and, whenever a neighbouring source event falls inside the refinement window — for a
    /// Poisson source at 20 Hz and an 8 ms window, about 15% of the time in each direction — one
    /// or more cross-pairs as well. Those extra differences are not independent draws; they are
    /// deterministic consequences of the same events, and any resampling that treats them as
    /// independent under-reports the median's scatter. Resampling by `a` index resamples the
    /// events, which is the unit that actually varies from recording to recording.
    fn differences_indexed(&self, a: &[f64], b: &[f64]) -> Result<Vec<(usize, f64)>, FusionError> {
        let mut out = Vec::new();
        let (mut lo, mut hi) = (0usize, 0usize);
        for (i, &ta) in a.iter().enumerate() {
            while lo < b.len() && b[lo] < ta - self.search_half_s {
                lo += 1;
            }
            if hi < lo {
                hi = lo;
            }
            while hi < b.len() && b[hi] <= ta + self.search_half_s {
                hi += 1;
            }
            if out.len() + (hi - lo) > self.max_pairs {
                return Err(FusionError::TooManyPairs { cap: self.max_pairs });
            }
            for &tb in &b[lo..hi] {
                out.push((i, tb - ta));
            }
        }
        Ok(out)
    }

    /// Estimate the offset of `b`'s clock relative to `a`'s.
    ///
    /// Both streams are read in their **own reported frames**, so any offset already recorded in
    /// their [`Clock`]s is ignored: this is the measurement that produces such an offset, and
    /// letting it consume one would make the call idempotent in the wrong direction.
    ///
    /// # Errors
    ///
    /// [`FusionError::Empty`] if either stream has no events, [`FusionError::TooManyPairs`] if the
    /// sweep exceeds its cap, and [`FusionError::TooFewPairs`] if the refinement window holds fewer
    /// than `min_pairs` differences — which is the honest answer when two streams share no visible
    /// structure, and is what distinguishes "no common source" from "offset zero".
    pub fn estimate(&self, a: &Stream, b: &Stream) -> Result<OffsetFit, FusionError> {
        if a.is_empty() {
            return Err(FusionError::Empty { what: "stream a" });
        }
        if b.is_empty() {
            return Err(FusionError::Empty { what: "stream b" });
        }
        let ta = a.reported_s();
        let tb = b.reported_s();
        let diffs = self.differences(&ta, &tb)?;
        if diffs.len() < self.min_pairs {
            return Err(FusionError::TooFewPairs { found: diffs.len(), needed: self.min_pairs });
        }

        // Coarse: the tallest histogram bin, ties to the lowest index so the answer is
        // deterministic rather than dependent on iteration order.
        let bins = ((2.0 * self.search_half_s / self.bin_s).ceil() as usize).max(1);
        let mut hist = vec![0usize; bins];
        for &d in &diffs {
            let k = ((d + self.search_half_s) / self.bin_s).floor();
            if k >= 0.0 && (k as usize) < bins {
                hist[k as usize] += 1;
            }
        }
        let peak = hist.iter().enumerate().fold((0usize, 0usize), |best, (i, &c)| {
            if c > best.1 { (i, c) } else { best }
        });
        let coarse_s = -self.search_half_s + (peak.0 as f64 + 0.5) * self.bin_s;

        // Fine: the median of the differences near the peak.
        let near: Vec<f64> =
            diffs.iter().copied().filter(|d| (d - coarse_s).abs() <= self.refine_half_s).collect();
        if near.len() < self.min_pairs {
            return Err(FusionError::TooFewPairs { found: near.len(), needed: self.min_pairs });
        }
        // `min_pairs` is a `pub` field and may legally be zero, so an empty refinement window is
        // reachable and is refused here rather than unwrapped.
        let (Some(first), true) = (median(&near), !near.is_empty()) else {
            return Err(FusionError::TooFewPairs { found: near.len(), needed: 1 });
        };

        // ⛔ RE-CENTRE, AND TAKE THE MEDIAN AGAIN. The window above is centred on a histogram
        // bin, so it sits up to `bin_s / 2` off the true lag by an amount that depends on which
        // bin the peak happened to land in. That is harmless for the cluster itself, which is far
        // narrower than the window, but the cross-pair differences from neighbouring source events
        // are spread across the whole window, and an off-centre window admits more of them on one
        // side than the other — biasing the median by an amount that varies from recording to
        // recording. Measured under the calibration protocol below: re-centring cuts the
        // seed-to-seed scatter of the estimate to 0.67 (no background), 0.60 (1 Hz) and 0.56
        // (5 Hz) of what a single pass gives, and brings it within 19% of the closed form
        // `J / √n`. The improvement grows with contamination because the mechanism is
        // contamination.
        let indexed = self.differences_indexed(&ta, &tb)?;
        let near_ix: Vec<(usize, f64)> =
            indexed.into_iter().filter(|(_, d)| (d - first).abs() <= self.refine_half_s).collect();
        let vals: Vec<f64> = near_ix.iter().map(|(_, d)| *d).collect();
        let Some(offset_s) = median(&vals) else {
            return Err(FusionError::TooFewPairs { found: 0, needed: 1 });
        };
        let dev: Vec<f64> = vals.iter().map(|d| (d - offset_s).abs()).collect();
        let mad_s = median(&dev).unwrap_or(0.0);
        // Contiguous by `a` index, so a change of index is a new event.
        let mut events = 0usize;
        let mut last: Option<usize> = None;
        for (i, _) in &near_ix {
            if last != Some(*i) {
                events += 1;
                last = Some(*i);
            }
        }
        Ok(OffsetFit { offset_s, coarse_s, pairs: vals.len(), mad_s, events })
    }

    /// Estimate offset **and** drift by measuring the offset on each half of the record.
    ///
    /// The offset observed at common time `t` is `offset + drift * t`, so two measurements at known
    /// times give the slope. The record is split at `a`'s median event index; each half's centroid
    /// is the mean of the `a` reported times it contains.
    ///
    /// Precision is the thing to watch: the drift's standard error is about `sqrt(2)` times an
    /// offset's, divided by the lever arm. Halving the record length doubles the error in the
    /// drift; doubling the event rate only reduces it as `1/sqrt(n)`.
    ///
    /// # Errors
    ///
    /// [`FusionError::Empty`] for an empty stream, [`FusionError::NoLeverArm`] when the two halves'
    /// centroids are not separated (a degenerate record, e.g. every event at one instant), and
    /// whatever [`OffsetEstimator::estimate`] returns for either half.
    pub fn estimate_drift(&self, a: &Stream, b: &Stream) -> Result<DriftFit, FusionError> {
        if a.len() < 2 {
            return Err(FusionError::Empty { what: "stream a (needs two halves)" });
        }
        let mid = a.len() / 2;
        let (first, second) = (a.slice(0, mid), a.slice(mid, a.len()));
        let f1 = self.estimate(&first, b)?;
        let f2 = self.estimate(&second, b)?;
        let c1 = first.reported_s().iter().sum::<f64>() / first.len() as f64;
        let c2 = second.reported_s().iter().sum::<f64>() / second.len() as f64;
        let span_s = c2 - c1;
        if !(span_s > 0.0) {
            return Err(FusionError::NoLeverArm { span_s });
        }
        let drift = (f2.offset_s - f1.offset_s) / span_s;
        Ok(DriftFit {
            offset_s: f1.offset_s - drift * c1,
            drift,
            first_half_s: f1.offset_s,
            second_half_s: f2.offset_s,
            span_s,
        })
    }
}

// =================================================================================================
// Alignment
// =================================================================================================

/// One event on the common timeline.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AlignedEvent {
    /// Common-timeline time, seconds. Real-valued because a clock offset is not an integer number
    /// of anybody's ticks; see the module doc.
    pub t_s: f64,
    /// Which sense it came from.
    pub modality: u16,
    /// Which element of that sense fired.
    pub address: u32,
    /// The sign the sensor reported, carried through untouched. Nothing in this module reads it —
    /// it is preserved because discarding polarity halves the information at no visible cost.
    pub polarity: Polarity,
}

/// Several streams merged onto one timeline, sorted by `(t_s, modality, address)`.
#[derive(Debug, Clone, PartialEq)]
pub struct Aligned {
    events: Vec<AlignedEvent>,
}

impl Aligned {
    /// Merge streams through their own clocks.
    ///
    /// Each stream's [`Clock`] is applied as it stands. A stream whose clock is the default is
    /// mapped by the identity, so aligning a single unshifted stream returns its own reported times
    /// **bit for bit** — the property test (e) in this module's suite asserts.
    ///
    /// # Errors
    ///
    /// [`FusionError::NoModalities`] for an empty slice, and [`FusionError::NonFinite`] if a clock
    /// maps a tick to a non-finite time, which needs a `dt_s` written into the `pub` field after
    /// construction.
    pub fn merge(streams: &[Stream]) -> Result<Self, FusionError> {
        if streams.is_empty() {
            return Err(FusionError::NoModalities);
        }
        let mut events = Vec::new();
        for s in streams {
            for e in &s.events {
                let t_s = finite("aligned time", s.clock.to_common_s(e.t))?;
                events.push(AlignedEvent {
                    t_s,
                    modality: s.modality,
                    address: e.address,
                    polarity: e.polarity,
                });
            }
        }
        events.sort_by(|x, y| {
            x.t_s
                .total_cmp(&y.t_s)
                .then(x.modality.cmp(&y.modality))
                .then(x.address.cmp(&y.address))
        });
        Ok(Self { events })
    }

    /// The merged events, in time order.
    #[must_use]
    pub fn events(&self) -> &[AlignedEvent] {
        &self.events
    }

    /// How many events there are in total.
    #[must_use]
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Whether nothing was aligned.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Times of one modality's events, seconds, in order.
    #[must_use]
    pub fn times_of(&self, modality: u16) -> Vec<f64> {
        self.events.iter().filter(|e| e.modality == modality).map(|e| e.t_s).collect()
    }

    /// Mean rate of one modality over an explicit observation window, hertz.
    ///
    /// The window is a parameter rather than inferred from the first and last event, because the
    /// closed forms in [`chance_binding_probability`] are statements about a *stated* observation
    /// interval, and inferring one from the data conditions on the extremes and biases the rate
    /// upward. `None` when the window has no length.
    #[must_use]
    pub fn rate_hz_in(&self, modality: u16, t0: f64, t1: f64) -> Option<f64> {
        if !(t1 > t0) {
            return None;
        }
        let n = self.events.iter().filter(|e| e.modality == modality && e.t_s >= t0 && e.t_s < t1).count();
        Some(n as f64 / (t1 - t0))
    }
}

/// Signed difference of the closest cross-stream pair within a half-window, or `None`.
///
/// Both slices must be sorted ascending. Returns `t_b - t_a` for the pair minimising `|t_b - t_a|`,
/// so the **sign is the temporal order** — which is the one quantity no single-modality branch can
/// ever observe, and therefore the whole difference between early and late fusion.
///
/// Ties in `|t_b - t_a|` resolve to the earlier `a`, which makes the answer deterministic.
#[must_use]
pub fn nearest_cross_pair(a: &[f64], b: &[f64], half_window_s: f64) -> Option<f64> {
    let mut best: Option<f64> = None;
    for &ta in a {
        let k = b.partition_point(|&x| x < ta);
        for idx in [k.checked_sub(1), (k < b.len()).then_some(k)].into_iter().flatten() {
            let d = b[idx] - ta;
            if d.abs() <= half_window_s && best.is_none_or(|bd| d.abs() < bd.abs()) {
                best = Some(d);
            }
        }
    }
    best
}

// =================================================================================================
// Coincidence detection — binding by synchrony
// =================================================================================================

/// Binds an anchor modality's events to partner modalities that fired within a tolerance window.
///
/// This is binding by synchrony as a countable operation. An anchor event is **bound** when every
/// listed partner modality has at least one event within `±half_window_s` of it — a conjunction,
/// not a disjunction, which is what makes three-way binding stricter than two-way rather than
/// looser.
#[derive(Debug, Clone, PartialEq)]
pub struct CoincidenceDetector {
    /// Tolerance half-width, seconds; the window is `[t - w, t + w]`, **inclusive at both ends**.
    /// Meredith, Nemitz & Stein (1987) measured multisensory enhancement in the cat superior
    /// colliculus over offsets of order 100 ms, so biology's own window is wide; this field exists
    /// so the number is the caller's and is visible in their code.
    pub half_window_s: f64,
    /// The modality whose events are the anchors — the one being asked "did anything else fire
    /// with you?".
    pub anchor: u16,
    /// Modalities that must **all** be present inside the window for the anchor to count as bound.
    pub partners: Vec<u16>,
}

/// What a coincidence sweep counted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CoincidenceReport {
    /// Anchors inside the interior `[t0 + w, t1 - w]`, i.e. those whose window fitted entirely
    /// inside the observation interval. Anchors near the edges are excluded, because a truncated
    /// window has a lower binding probability and would bias the rate below its closed form.
    pub anchors_considered: usize,
    /// Anchors for which every listed partner was present.
    pub bound: usize,
    /// Total partner events falling inside anchors' windows, summed over partners and anchors.
    /// Counts multiplicity, so its expectation is the sum of `2*w*r` over partners.
    pub partner_events: u64,
}

impl CoincidenceReport {
    /// Bound anchors as a fraction of anchors considered. `None` when nothing was considered.
    #[must_use]
    pub fn binding_rate(&self) -> Option<f64> {
        if self.anchors_considered == 0 {
            None
        } else {
            Some(self.bound as f64 / self.anchors_considered as f64)
        }
    }

    /// Mean partner events per anchor considered. `None` when nothing was considered.
    #[must_use]
    pub fn mean_partners(&self) -> Option<f64> {
        if self.anchors_considered == 0 {
            None
        } else {
            Some(self.partner_events as f64 / self.anchors_considered as f64)
        }
    }
}

/// Expected partner events inside one anchor's window under independence: `2 * w * r`, exactly.
///
/// A homogeneous Poisson partner at rate `r` puts, in expectation, `2*w*r` events into a window of
/// half-width `w`, with no approximation. This is the sharper of the two closed forms here, because
/// it is a mean rather than a probability and so a detector that is off by a factor of two in the
/// window width fails it immediately.
///
/// # Errors
///
/// [`FusionError::Negative`] or [`FusionError::NonFinite`] for a half-window or rate that is not a
/// finite non-negative number.
pub fn chance_partners_per_anchor(half_window_s: f64, rate_hz: f64) -> Result<f64, FusionError> {
    let w = non_negative("half_window_s", half_window_s)?;
    let r = non_negative("rate_hz", rate_hz)?;
    Ok(2.0 * w * r)
}

/// Probability that an anchor is bound **by accident** to every listed partner:
/// `prod_m (1 - exp(-2 * w * r_m))`.
///
/// Each factor is the probability that a homogeneous Poisson partner at rate `r_m` puts at least
/// one event in the window — one minus the Poisson zero term. The product is the conjunction across
/// independent partners. This is the number a coincidence detector's output has to be compared
/// against before any of it means anything: with a 10 ms half-window and two partners at 100 Hz,
/// **33% of anchors bind by chance alone**, and a detector reporting 40% has found almost nothing.
///
/// # Errors
///
/// As [`chance_partners_per_anchor`], for the half-window and each rate.
pub fn chance_binding_probability(
    half_window_s: f64,
    partner_rates_hz: &[f64],
) -> Result<f64, FusionError> {
    let w = non_negative("half_window_s", half_window_s)?;
    if partner_rates_hz.is_empty() {
        return Err(FusionError::Empty { what: "partner rates" });
    }
    let mut p = 1.0;
    for &r in partner_rates_hz {
        let r = non_negative("rate_hz", r)?;
        p *= 1.0 - (-2.0 * w * r).exp();
    }
    Ok(p)
}

impl CoincidenceDetector {
    /// Build a detector.
    ///
    /// # Errors
    ///
    /// [`FusionError::NotPositive`] or [`FusionError::NonFinite`] for a half-window that is not a
    /// positive finite number, and [`FusionError::Empty`] when no partner modality was named —
    /// binding an anchor to nothing would report every anchor as bound.
    pub fn new(half_window_s: f64, anchor: u16, partners: Vec<u16>) -> Result<Self, FusionError> {
        let half_window_s = positive("half_window_s", half_window_s)?;
        if partners.is_empty() {
            return Err(FusionError::Empty { what: "partner modalities" });
        }
        Ok(Self { half_window_s, anchor, partners })
    }

    /// Sweep an aligned recording over the stated observation interval.
    ///
    /// # Errors
    ///
    /// [`FusionError::NonFinite`] for a non-finite bound, and [`FusionError::NotPositive`] when the
    /// interval is not longer than the window it has to contain — a run shorter than `2w` has no
    /// interior and so no anchor whose window fits.
    pub fn detect(
        &self,
        aligned: &Aligned,
        t0: f64,
        t1: f64,
    ) -> Result<CoincidenceReport, FusionError> {
        let t0 = finite("t0", t0)?;
        let t1 = finite("t1", t1)?;
        let w = self.half_window_s;
        if !(t1 - t0 > 2.0 * w) {
            return Err(FusionError::NotPositive {
                what: "observation interval minus 2 * half_window_s",
                value: t1 - t0 - 2.0 * w,
            });
        }
        let partner_times: Vec<Vec<f64>> =
            self.partners.iter().map(|&m| aligned.times_of(m)).collect();
        let mut report =
            CoincidenceReport { anchors_considered: 0, bound: 0, partner_events: 0 };
        for e in &aligned.events {
            if e.modality != self.anchor || e.t_s < t0 + w || e.t_s > t1 - w {
                continue;
            }
            report.anchors_considered += 1;
            let mut all = true;
            for times in &partner_times {
                let lo = times.partition_point(|&x| x < e.t_s - w);
                let hi = times.partition_point(|&x| x <= e.t_s + w);
                let n = hi - lo;
                report.partner_events += n as u64;
                if n == 0 {
                    all = false;
                }
            }
            if all {
                report.bound += 1;
            }
        }
        Ok(report)
    }
}

// =================================================================================================
// Attention: precision-weighted gating
// =================================================================================================

/// Routes influence between modalities by their precision, and says what a dropout costs.
///
/// Ernst & Banks (2002) and Alais & Burr (2004) both report that the nervous system weights a
/// sensory channel by its **reliability**, `1/σ²`, which is also the maximum-likelihood combination
/// of unbiased Gaussian estimates. That is one formula doing two jobs here: it decides which
/// modality dominates, and it prices the loss when one goes away.
///
/// ```text
/// weight_m       = (1/σ_m²) / Σ_k (1/σ_k²)
/// fused variance = 1 / Σ_k (1/σ_k²)
/// ```
///
/// Both sums run over **available** modalities only. Dropping one does not renormalise away its
/// cost: the weights still sum to one, but the fused variance rises by exactly the ratio of the
/// full precision sum to the remaining one. Two equally reliable senses, one lost — the variance
/// doubles and the error grows by `sqrt(2)`. Quantified degradation, not failure.
#[derive(Debug, Clone, PartialEq)]
pub struct PrecisionGate {
    variances: Vec<f64>,
    available: Vec<bool>,
}

impl PrecisionGate {
    /// Build from per-modality variances, in the squared unit of whatever is being estimated.
    ///
    /// # Errors
    ///
    /// [`FusionError::NoModalities`] for an empty slice, and [`FusionError::NotPositive`] or
    /// [`FusionError::NonFinite`] for a variance that is not a positive finite number — a
    /// zero-variance channel has infinite precision and would silence every other sense, which is
    /// a claim no sensor can support.
    pub fn new(variances: Vec<f64>) -> Result<Self, FusionError> {
        if variances.is_empty() {
            return Err(FusionError::NoModalities);
        }
        for &v in &variances {
            positive("variance", v)?;
        }
        let n = variances.len();
        Ok(Self { variances, available: vec![true; n] })
    }

    /// Build by measuring each modality's variance from its own residuals.
    ///
    /// `residuals[m]` are modality `m`'s estimate-minus-truth samples. The variance is the **sample**
    /// variance with `n-1` in the denominator, because these are a sample; using `n` would report a
    /// channel as more precise than it is, which in an inverse-variance gate means over-trusting it.
    ///
    /// # Errors
    ///
    /// [`FusionError::NoModalities`] for no modalities, [`FusionError::Empty`] for a modality with
    /// fewer than two residuals, [`FusionError::NonFinite`] for a non-finite residual, and
    /// [`FusionError::NotPositive`] if a modality's residuals are all identical.
    pub fn from_residuals(residuals: &[Vec<f64>]) -> Result<Self, FusionError> {
        if residuals.is_empty() {
            return Err(FusionError::NoModalities);
        }
        let mut variances = Vec::with_capacity(residuals.len());
        for r in residuals {
            if r.len() < 2 {
                return Err(FusionError::Empty { what: "residuals (need at least two)" });
            }
            for &x in r {
                finite("residual", x)?;
            }
            let n = r.len() as f64;
            let mean = r.iter().sum::<f64>() / n;
            let var = r.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (n - 1.0);
            variances.push(positive("measured variance", var)?);
        }
        Self::new(variances)
    }

    /// How many modalities the gate knows about.
    #[must_use]
    pub fn len(&self) -> usize {
        self.variances.len()
    }

    /// Whether the gate knows about no modalities. Unreachable through [`PrecisionGate::new`],
    /// which refuses an empty slice; present because clippy is right that a `len` without an
    /// `is_empty` is a trap for a caller.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.variances.is_empty()
    }

    /// Mark a modality present or absent — the dropout switch.
    ///
    /// # Errors
    ///
    /// [`FusionError::UnknownModality`] for an index past the end.
    pub fn set_available(&mut self, modality: usize, ok: bool) -> Result<(), FusionError> {
        if modality >= self.available.len() {
            return Err(FusionError::UnknownModality {
                modality,
                count: self.available.len(),
            });
        }
        self.available[modality] = ok;
        Ok(())
    }

    /// Sum of `1/σ²` over available modalities, or `None` when none are available.
    #[must_use]
    pub fn total_precision(&self) -> Option<f64> {
        let s: f64 = self
            .variances
            .iter()
            .zip(&self.available)
            .filter(|&(_, &ok)| ok)
            .map(|(v, _)| 1.0 / v)
            .sum();
        if s > 0.0 { Some(s) } else { None }
    }

    /// Per-modality weights, summing to one over the available modalities and `0.0` for the rest.
    /// `None` when nothing is available.
    #[must_use]
    pub fn weights(&self) -> Option<Vec<f64>> {
        let total = self.total_precision()?;
        Some(
            self.variances
                .iter()
                .zip(&self.available)
                .map(|(v, &ok)| if ok { (1.0 / v) / total } else { 0.0 })
                .collect(),
        )
    }

    /// Variance of the fused estimate, `1 / Σ(1/σ²)`. `None` when nothing is available.
    #[must_use]
    pub fn fused_variance(&self) -> Option<f64> {
        self.total_precision().map(|p| 1.0 / p)
    }

    /// The modality carrying the largest weight — which sense is currently dominating. Ties resolve
    /// to the lowest index. `None` when nothing is available.
    #[must_use]
    pub fn dominant(&self) -> Option<usize> {
        let mut best: Option<(usize, f64)> = None;
        for (i, (v, &ok)) in self.variances.iter().zip(&self.available).enumerate() {
            if ok && best.is_none_or(|(_, bv)| 1.0 / v > bv) {
                best = Some((i, 1.0 / v));
            }
        }
        best.map(|(i, _)| i)
    }

    /// Combine per-modality estimates into one. `None` when nothing is available.
    ///
    /// # Errors
    ///
    /// [`FusionError::LengthMismatch`] if the estimate count does not match the modality count, and
    /// [`FusionError::NonFinite`] for a non-finite estimate — including from a modality currently
    /// marked unavailable, because a caller passing `NaN` for a dropped sensor and a caller passing
    /// `NaN` by mistake are indistinguishable here and only one of them wants silence.
    pub fn fuse(&self, estimates: &[f64]) -> Result<Option<f64>, FusionError> {
        if estimates.len() != self.variances.len() {
            return Err(FusionError::LengthMismatch {
                a: estimates.len(),
                b: self.variances.len(),
            });
        }
        for &x in estimates {
            finite("estimate", x)?;
        }
        let Some(w) = self.weights() else { return Ok(None) };
        Ok(Some(w.iter().zip(estimates).map(|(wi, xi)| wi * xi).sum()))
    }

    /// By what factor the fused variance would grow if `modality` were lost, given what is
    /// currently available.
    ///
    /// `Some(1.0)` if it is already unavailable — losing it again costs nothing — and `None` if it
    /// is the last one standing, because the variance after is infinite and a ratio to infinity is
    /// not a degradation figure, it is a total loss and should be reported as one.
    ///
    /// # Errors
    ///
    /// [`FusionError::UnknownModality`] for an index past the end.
    pub fn degradation_from_dropping(
        &self,
        modality: usize,
    ) -> Result<Option<f64>, FusionError> {
        if modality >= self.variances.len() {
            return Err(FusionError::UnknownModality {
                modality,
                count: self.variances.len(),
            });
        }
        if !self.available[modality] {
            return Ok(Some(1.0));
        }
        let Some(before) = self.total_precision() else { return Ok(None) };
        let after = before - 1.0 / self.variances[modality];
        if after <= 0.0 {
            return Ok(None);
        }
        Ok(Some(before / after))
    }
}

// =================================================================================================
// Cross-modal association learning
// =================================================================================================

/// Learns which element of one modality goes with which element of another, by spike timing.
///
/// A camera pixel and a microphone channel that report the same physical event fire together, over
/// and over, with a consistent sign. That is exactly the statistic spike-timing-dependent
/// plasticity measures, so the correspondence can be learned without a label: every cross-modal
/// event pair within `max_lag_s` is played through [`PairStdp::apply_pair`] on the weight of the
/// matching `(pre address, post address)` cell.
///
/// The sign matters and is the thing to check. With `lag = t_post - t_pre > 0` the pair potentiates;
/// with the modalities' roles reversed the same physical correlation depresses. A correspondence
/// matrix that came out right with the arguments the wrong way round would be a matrix of
/// depressions with the correct **arg-min**, which plots identically to a correct one.
///
/// # Cost
///
/// The sweep is `O(n_pre * partners-within-the-window)`. It uses [`PairStdp::apply_pair`], which
/// clears the traces per pair — the isolated-pair protocol of Bi & Poo (1998), not a running
/// all-to-all trace. That makes the result exactly the sum of [`PairStdp::window`] over the pairs
/// (with an additive rule, away from the bounds), which is what lets a test assert it in closed
/// form; it also means the rule cannot express triplet or higher-order effects here. For those,
/// drive [`crate::plasticity::TripletStdp`] directly.
#[derive(Debug, Clone, PartialEq)]
pub struct CrossModalAssociator {
    rule: PairStdp,
    pre_modality: u16,
    post_modality: u16,
    n_pre: usize,
    n_post: usize,
    max_lag_s: f64,
    w: Vec<f64>,
}

impl CrossModalAssociator {
    /// Build an all-to-all cross-modal weight matrix at a uniform initial weight.
    ///
    /// # Errors
    ///
    /// [`FusionError::Empty`] for a zero address count on either side, [`FusionError::NotPositive`]
    /// or [`FusionError::NonFinite`] for a `max_lag_s` that is not a positive finite number, and
    /// [`FusionError::NonFinite`] for a non-finite initial weight.
    pub fn new(
        rule: PairStdp,
        pre_modality: u16,
        n_pre: usize,
        post_modality: u16,
        n_post: usize,
        w0: f64,
        max_lag_s: f64,
    ) -> Result<Self, FusionError> {
        if n_pre == 0 || n_post == 0 {
            return Err(FusionError::Empty { what: "address space" });
        }
        let w0 = finite("initial weight", w0)?;
        let max_lag_s = positive("max_lag_s", max_lag_s)?;
        Ok(Self {
            rule,
            pre_modality,
            post_modality,
            n_pre,
            n_post,
            max_lag_s,
            w: vec![w0; n_pre * n_post],
        })
    }

    /// Play every cross-modal pair within `max_lag_s` through the rule. Returns the pair count.
    ///
    /// Pairs at exactly zero lag are included and potentiate by `A_plus`, which is what
    /// [`PairStdp::apply_pair`] does at that point; its doc explains why that differs from
    /// [`PairStdp::window`] there. Addresses at or past the declared counts are **skipped**, not
    /// refused, and are not counted — an aligned recording may legitimately carry a wider address
    /// space than the associator was sized for.
    ///
    /// # Errors
    ///
    /// Whatever [`PairStdp::apply_pair`] returns, forwarded as [`FusionError::Plasticity`].
    pub fn observe(&mut self, aligned: &Aligned) -> Result<u64, FusionError> {
        let pre: Vec<(f64, u32)> = aligned
            .events
            .iter()
            .filter(|e| e.modality == self.pre_modality)
            .map(|e| (e.t_s, e.address))
            .collect();
        let post: Vec<(f64, u32)> = aligned
            .events
            .iter()
            .filter(|e| e.modality == self.post_modality)
            .map(|e| (e.t_s, e.address))
            .collect();
        let mut pairs = 0u64;
        let (mut lo, mut hi) = (0usize, 0usize);
        for &(tp, ap) in &pre {
            if ap as usize >= self.n_pre {
                continue;
            }
            while lo < post.len() && post[lo].0 < tp - self.max_lag_s {
                lo += 1;
            }
            if hi < lo {
                hi = lo;
            }
            while hi < post.len() && post[hi].0 <= tp + self.max_lag_s {
                hi += 1;
            }
            for &(tq, aq) in &post[lo..hi] {
                if aq as usize >= self.n_post {
                    continue;
                }
                let k = ap as usize * self.n_post + aq as usize;
                self.w[k] = self.rule.apply_pair(self.w[k], tq - tp)?;
                pairs += 1;
            }
        }
        Ok(pairs)
    }

    /// The learned weight for one correspondence, or `None` for an out-of-range address.
    #[must_use]
    pub fn weight(&self, pre: usize, post: usize) -> Option<f64> {
        if pre >= self.n_pre || post >= self.n_post {
            return None;
        }
        Some(self.w[pre * self.n_post + post])
    }

    /// The whole matrix, row-major, `n_pre` rows of `n_post`.
    #[must_use]
    pub fn weights(&self) -> &[f64] {
        &self.w
    }

    /// The post-modality address most strongly associated with `pre`.
    ///
    /// `None` for an out-of-range `pre`, **and `None` on a tie** — two correspondences with the
    /// same weight is a refusal to guess, not a coin flip, because the arg-max of an untrained
    /// matrix is otherwise always address zero and looks like a learned answer.
    #[must_use]
    pub fn best_partner(&self, pre: usize) -> Option<usize> {
        if pre >= self.n_pre {
            return None;
        }
        let row = &self.w[pre * self.n_post..(pre + 1) * self.n_post];
        let mut best = (0usize, row[0]);
        let mut tied = false;
        for (j, &x) in row.iter().enumerate().skip(1) {
            if x > best.1 {
                best = (j, x);
                tied = false;
            } else if x == best.1 {
                tied = true;
            }
        }
        if tied { None } else { Some(best.0) }
    }
}

// =================================================================================================
// Early and late fusion
// =================================================================================================

/// One trial, as the two fusion architectures each see it.
///
/// The split is the definition. A **late** branch sees `per_modality[m]` and nothing else — it
/// cannot reach the other streams, so no feature it computes can depend on two of them. An
/// **early** classifier sees every block concatenated, `cross_modal` included, so it can use
/// quantities that only exist jointly. Building both from one struct is what makes the comparison
/// an experiment rather than two different programs.
#[derive(Debug, Clone, PartialEq)]
pub struct TrialFeatures {
    /// One feature vector per modality, each computed from that stream alone.
    pub per_modality: Vec<Vec<f64>>,
    /// Features that require more than one stream to compute — a cross-modal interval, a binding
    /// count. Invisible to late fusion by construction, and undefined when a stream is missing.
    pub cross_modal: Vec<f64>,
    /// The class this trial belongs to.
    pub label: u32,
}

/// A nearest-centroid classifier on standardised features.
///
/// Deliberately the simplest thing that can learn: per-class feature means, Euclidean distance, no
/// covariance and no iteration. That is the point — the early/late comparison must not turn into a
/// comparison of two optimisers, so **both paths use this same classifier** and differ only in
/// which features reach it.
///
/// Features are standardised with the **training** mean and standard deviation, which matters more
/// than it looks: an event count is `O(10)` and an inter-modal interval is `O(0.01)` seconds, so an
/// unstandardised distance would be a count classifier with a rounding error attached. A feature
/// with zero training variance gets a scale of `1.0` and therefore contributes an identical
/// constant to every class's distance, which cancels — a constant feature is ignored rather than
/// dividing by zero.
#[derive(Debug, Clone, PartialEq)]
pub struct NearestCentroid {
    classes: Vec<u32>,
    centroids: Vec<Vec<f64>>,
    mean: Vec<f64>,
    scale: Vec<f64>,
}

impl NearestCentroid {
    /// Fit on rows of features and their labels.
    ///
    /// # Errors
    ///
    /// [`FusionError::Empty`] for no rows or zero-width rows, [`FusionError::LengthMismatch`] if
    /// the label count differs from the row count, [`FusionError::RaggedFeatures`] for a row of a
    /// different width, [`FusionError::NonFinite`] for a non-finite feature, and
    /// [`FusionError::SingleClass`] when every label is the same.
    pub fn fit(x: &[Vec<f64>], y: &[u32]) -> Result<Self, FusionError> {
        if x.is_empty() {
            return Err(FusionError::Empty { what: "training set" });
        }
        if x.len() != y.len() {
            return Err(FusionError::LengthMismatch { a: x.len(), b: y.len() });
        }
        let d = x[0].len();
        if d == 0 {
            return Err(FusionError::Empty { what: "feature vector" });
        }
        for (i, row) in x.iter().enumerate() {
            if row.len() != d {
                return Err(FusionError::RaggedFeatures { index: i });
            }
            for &v in row {
                finite("feature", v)?;
            }
        }
        let n = x.len() as f64;
        let mut mean = vec![0.0; d];
        for row in x {
            for j in 0..d {
                mean[j] += row[j];
            }
        }
        for m in &mut mean {
            *m /= n;
        }
        let mut scale = vec![0.0; d];
        for row in x {
            for j in 0..d {
                let e = row[j] - mean[j];
                scale[j] += e * e;
            }
        }
        for s in &mut scale {
            let sd = (*s / n).sqrt();
            *s = if sd > 0.0 { sd } else { 1.0 };
        }

        // Accumulated by label rather than looked up in a pre-built class list: a lookup that
        // "cannot fail" is a panic path, and this shape has none. Sorted afterwards so the class
        // order is ascending and reproducible whatever order the labels arrived in.
        let mut acc: Vec<(u32, Vec<f64>, usize)> = Vec::new();
        for (row, &lab) in x.iter().zip(y) {
            let c = match acc.iter().position(|e| e.0 == lab) {
                Some(c) => c,
                None => {
                    acc.push((lab, vec![0.0; d], 0));
                    acc.len() - 1
                }
            };
            acc[c].2 += 1;
            for j in 0..d {
                acc[c].1[j] += (row[j] - mean[j]) / scale[j];
            }
        }
        if acc.len() < 2 {
            return Err(FusionError::SingleClass { label: acc[0].0 });
        }
        acc.sort_by_key(|e| e.0);
        let classes: Vec<u32> = acc.iter().map(|e| e.0).collect();
        let centroids: Vec<Vec<f64>> = acc
            .iter()
            .map(|(_, sum, cnt)| sum.iter().map(|v| v / *cnt as f64).collect())
            .collect();
        Ok(Self { classes, centroids, mean, scale })
    }

    /// The class labels, ascending — the order [`NearestCentroid::score`] reports in.
    #[must_use]
    pub fn classes(&self) -> &[u32] {
        &self.classes
    }

    /// Negated squared distance to each class centroid, in class order; larger is better.
    ///
    /// Only features whose `mask` entry is `true` contribute. A mask of all `true` is the ordinary
    /// case; a mask is how a dropped modality is expressed without refitting.
    ///
    /// # Errors
    ///
    /// [`FusionError::LengthMismatch`] if the row or mask is the wrong width,
    /// [`FusionError::NonFinite`] for a non-finite feature, and [`FusionError::AllFeaturesDropped`]
    /// when the mask leaves nothing.
    pub fn score(&self, x: &[f64], mask: &[bool]) -> Result<Vec<f64>, FusionError> {
        let d = self.mean.len();
        if x.len() != d {
            return Err(FusionError::LengthMismatch { a: x.len(), b: d });
        }
        if mask.len() != d {
            return Err(FusionError::LengthMismatch { a: mask.len(), b: d });
        }
        if !mask.iter().any(|&b| b) {
            return Err(FusionError::AllFeaturesDropped);
        }
        let mut z = vec![0.0; d];
        for j in 0..d {
            if mask[j] {
                z[j] = (finite("feature", x[j])? - self.mean[j]) / self.scale[j];
            }
        }
        Ok(self
            .centroids
            .iter()
            .map(|c| {
                -(0..d)
                    .filter(|&j| mask[j])
                    .map(|j| (z[j] - c[j]) * (z[j] - c[j]))
                    .sum::<f64>()
            })
            .collect())
    }

    /// The highest-scoring class. Ties resolve to the lowest label, deterministically.
    ///
    /// # Errors
    ///
    /// As [`NearestCentroid::score`].
    pub fn predict(&self, x: &[f64], mask: &[bool]) -> Result<u32, FusionError> {
        let s = self.score(x, mask)?;
        let mut best = 0usize;
        for i in 1..s.len() {
            if s[i] > s[best] {
                best = i;
            }
        }
        Ok(self.classes[best])
    }
}

/// A score vector with its mean removed, so that only the differences between classes survive.
fn centred(s: &[f64]) -> Vec<f64> {
    let mean = s.iter().sum::<f64>() / s.len() as f64;
    s.iter().map(|v| v - mean).collect()
}

/// Flatten a trial into the early-fusion feature vector, and record which block each feature is in.
fn early_layout(t: &TrialFeatures) -> (Vec<f64>, Vec<Option<usize>>) {
    let mut x = Vec::new();
    let mut group = Vec::new();
    for (m, block) in t.per_modality.iter().enumerate() {
        for &v in block {
            x.push(v);
            group.push(Some(m));
        }
    }
    for &v in &t.cross_modal {
        x.push(v);
        group.push(None);
    }
    (x, group)
}

fn check_layout(trials: &[TrialFeatures]) -> Result<(), FusionError> {
    if trials.is_empty() {
        return Err(FusionError::Empty { what: "trials" });
    }
    let shape: Vec<usize> = trials[0].per_modality.iter().map(Vec::len).collect();
    let cross = trials[0].cross_modal.len();
    for (i, t) in trials.iter().enumerate() {
        if t.per_modality.len() != shape.len() || t.cross_modal.len() != cross {
            return Err(FusionError::RaggedFeatures { index: i });
        }
        for (b, block) in t.per_modality.iter().enumerate() {
            if block.len() != shape[b] {
                return Err(FusionError::RaggedFeatures { index: i });
            }
        }
    }
    Ok(())
}

/// One decision from every stream at once.
///
/// Every block is concatenated, `cross_modal` included, and one classifier sees the lot. This is
/// the architecture that can use the interval between two modalities, and the one whose joint
/// features stop meaning anything the moment a stream goes missing.
#[derive(Debug, Clone, PartialEq)]
pub struct EarlyFusion {
    clf: NearestCentroid,
    group: Vec<Option<usize>>,
}

impl EarlyFusion {
    /// Fit on labelled trials.
    ///
    /// # Errors
    ///
    /// [`FusionError::Empty`] for no trials, [`FusionError::RaggedFeatures`] for a trial whose
    /// layout differs from the first, and whatever [`NearestCentroid::fit`] returns.
    pub fn fit(trials: &[TrialFeatures]) -> Result<Self, FusionError> {
        check_layout(trials)?;
        let mut x = Vec::with_capacity(trials.len());
        let mut y = Vec::with_capacity(trials.len());
        let mut group = Vec::new();
        for t in trials {
            let (row, g) = early_layout(t);
            if group.is_empty() {
                group = g;
            }
            x.push(row);
            y.push(t.label);
        }
        Ok(Self { clf: NearestCentroid::fit(&x, &y)?, group })
    }

    /// The mask this architecture uses when `dropped` modalities are missing.
    ///
    /// A dropped modality's own features go, **and so does every cross-modal feature** — all of
    /// them, not just the ones nominally involving that stream, because a joint feature computed
    /// from a stream that produced nothing is not a degraded measurement, it is a different
    /// quantity. That is the structural cost of early fusion, and it is why this method exists
    /// rather than zeroing the missing block and pretending.
    #[must_use]
    pub fn mask_for(&self, dropped: &[usize]) -> Vec<bool> {
        self.group
            .iter()
            .map(|g| match g {
                Some(m) => !dropped.contains(m),
                None => dropped.is_empty(),
            })
            .collect()
    }

    /// Classify one trial with the listed modalities missing.
    ///
    /// # Errors
    ///
    /// [`FusionError::AllFeaturesDropped`] when nothing survives the mask, plus whatever
    /// [`NearestCentroid::score`] returns.
    pub fn predict(&self, t: &TrialFeatures, dropped: &[usize]) -> Result<u32, FusionError> {
        let (x, _) = early_layout(t);
        self.clf.predict(&x, &self.mask_for(dropped))
    }

    /// Accuracy over a test set with the listed modalities missing.
    ///
    /// # Errors
    ///
    /// As [`EarlyFusion::predict`], plus [`FusionError::Metric`] from [`metrics::accuracy`] on an
    /// empty test set.
    pub fn accuracy(
        &self,
        trials: &[TrialFeatures],
        dropped: &[usize],
    ) -> Result<f64, FusionError> {
        let mut pred = Vec::with_capacity(trials.len());
        let mut truth = Vec::with_capacity(trials.len());
        for t in trials {
            pred.push(self.predict(t, dropped)? as usize);
            truth.push(t.label as usize);
        }
        Ok(metrics::accuracy(&pred, &truth)?)
    }
}

/// One decision per stream, then a vote.
///
/// Each modality gets its own [`NearestCentroid`] on its own block. Their per-class scores are
/// centred, divided by a **per-branch scale measured once on the training set**, and summed with a
/// **log-odds vote weight** `ln(p / (1 - p))` taken from the branch's own training accuracy `p`.
/// That is the classical weighted-majority weight — Nitzan & Paroush, Econometrica 50(3):683–688,
/// 1982, and Shapley & Grofman, Public Choice 43:329–343, 1984, with the derivation set out in
/// Kuncheva, *Combining Pattern Classifiers*, Wiley 2004 — and it is the Bayes-optimal gain for
/// independent binary voters. It is [`PrecisionGate`]'s rule in the discrete case: trust a sense in
/// proportion to the evidence it carries, which for a coin-flip branch is none.
///
/// ⚠ The log-odds derivation is a **binary** result. With more than two classes it is applied here
/// as a scalar gain on the score vector, which is a heuristic rather than a theorem, and this
/// implementation did not locate a closed-form multiclass equivalent that does not need the full
/// confusion matrix.
///
/// ⛔ Why not weight by accuracy directly, which is the obvious thing: a branch at chance then
/// still votes with weight 0.5 against an informative branch's 0.93, and **measurably drags the
/// result down** — 5.5 accuracy points, in the test that now guards this. Why not clamp the weight
/// at zero for a below-chance branch: a set of branches all clamped to zero makes the combiner a
/// constant predictor, and then "late fusion is at chance" is unfalsifiable rather than measured.
/// The log-odds weight is small for a useless branch, negative for an anti-correlated one, and
/// never exactly zero.
///
/// ⛔ **The scale is per branch, not per trial, and that distinction is the whole architecture.**
/// An earlier version of this type normalised each trial's score vector to unit length, which looks
/// like the same idea and is not: for two classes it collapses every branch's output to `±1`,
/// discarding how confident the branch was, and soft voting degenerates into hard voting. With two
/// branches and two classes that is provably no better than one branch — the pair either agree,
/// which one of them would have got anyway, or tie. It was caught by the test that asserts two
/// senses beat one on [`RedundantRateTask`], and that test is the regression guard.
///
/// The accuracy it is built from is in-sample and therefore optimistic; on a task where one branch
/// overfits, a held-out estimate belongs here instead.
#[derive(Debug, Clone, PartialEq)]
pub struct LateFusion {
    branches: Vec<NearestCentroid>,
    reliability: Vec<f64>,
    weight: Vec<f64>,
    scale: Vec<f64>,
    classes: Vec<u32>,
}

impl LateFusion {
    /// Fit one branch per modality and weight each by its in-sample accuracy.
    ///
    /// # Errors
    ///
    /// As [`EarlyFusion::fit`]; additionally [`FusionError::NoModalities`] when the trials carry no
    /// per-modality blocks at all.
    pub fn fit(trials: &[TrialFeatures]) -> Result<Self, FusionError> {
        check_layout(trials)?;
        let m = trials[0].per_modality.len();
        if m == 0 {
            return Err(FusionError::NoModalities);
        }
        let y: Vec<u32> = trials.iter().map(|t| t.label).collect();
        let mut branches = Vec::with_capacity(m);
        let mut reliability = Vec::with_capacity(m);
        let mut weight = Vec::with_capacity(m);
        let mut scale = Vec::with_capacity(m);
        let truth: Vec<usize> = y.iter().map(|&l| l as usize).collect();
        for b in 0..m {
            let x: Vec<Vec<f64>> = trials.iter().map(|t| t.per_modality[b].clone()).collect();
            let clf = NearestCentroid::fit(&x, &y)?;
            let mask = vec![true; x[0].len()];
            let mut pred = Vec::with_capacity(x.len());
            let mut energy = 0.0;
            for row in &x {
                let s = centred(&clf.score(row, &mask)?);
                energy += s.iter().map(|v| v * v).sum::<f64>();
                pred.push(clf.predict(row, &mask)? as usize);
            }
            let acc = metrics::accuracy(&pred, &truth)?;
            reliability.push(acc);
            // Clamped to the finest resolution the training set can actually express, so that a
            // branch which happens to be perfect in sample gets a large weight rather than an
            // infinite one.
            let guard = 0.5 / x.len() as f64;
            let p = acc.clamp(guard, 1.0 - guard);
            weight.push((p / (1.0 - p)).ln());
            // Root-mean-square length of this branch's centred score vector across the training
            // set. One number per branch, so a trial where the branch was unusually sure stays
            // unusually sure at the vote.
            let rms = (energy / x.len() as f64).sqrt();
            scale.push(if rms > 0.0 { rms } else { 1.0 });
            branches.push(clf);
        }
        let classes = branches[0].classes().to_vec();
        Ok(Self { branches, reliability, weight, scale, classes })
    }

    /// Each branch's in-sample accuracy, in `0.0..=1.0`. A diagnostic in its own right: a branch
    /// near chance is telling you its modality does not carry the label alone.
    #[must_use]
    pub fn reliability(&self) -> &[f64] {
        &self.reliability
    }

    /// Each branch's log-odds vote weight, `ln(p / (1 - p))` for its accuracy `p`.
    ///
    /// Zero at chance, positive above it, **negative below** — a branch that is reliably wrong is
    /// evidence, and flipping it is the right thing to do with it. Unbounded in principle and
    /// bounded in practice by the training-set resolution, so a branch that is perfect on `n`
    /// trials weighs `ln(2n - 1)` rather than infinity.
    #[must_use]
    pub fn vote_weights(&self) -> &[f64] {
        &self.weight
    }

    /// Classify one trial with the listed modalities missing.
    ///
    /// A dropped branch simply does not vote. Nothing else changes, which is the architectural
    /// advantage this path has and the reason it survives a sensor failure that destroys early
    /// fusion.
    ///
    /// # Errors
    ///
    /// [`FusionError::AllFeaturesDropped`] when every branch is dropped, [`FusionError::UnknownModality`]
    /// for a dropped index past the end, plus whatever [`NearestCentroid::score`] returns.
    pub fn predict(&self, t: &TrialFeatures, dropped: &[usize]) -> Result<u32, FusionError> {
        for &d in dropped {
            if d >= self.branches.len() {
                return Err(FusionError::UnknownModality {
                    modality: d,
                    count: self.branches.len(),
                });
            }
        }
        if t.per_modality.len() != self.branches.len() {
            return Err(FusionError::LengthMismatch {
                a: t.per_modality.len(),
                b: self.branches.len(),
            });
        }
        let mut total = vec![0.0; self.classes.len()];
        let mut voted = false;
        for (b, clf) in self.branches.iter().enumerate() {
            if dropped.contains(&b) {
                continue;
            }
            let mask = vec![true; t.per_modality[b].len()];
            let s = centred(&clf.score(&t.per_modality[b], &mask)?);
            for (k, v) in s.iter().enumerate() {
                total[k] += self.weight[b] * v / self.scale[b];
            }
            voted = true;
        }
        if !voted {
            return Err(FusionError::AllFeaturesDropped);
        }
        let mut best = 0usize;
        for i in 1..total.len() {
            if total[i] > total[best] {
                best = i;
            }
        }
        Ok(self.classes[best])
    }

    /// Accuracy over a test set with the listed modalities missing.
    ///
    /// # Errors
    ///
    /// As [`LateFusion::predict`], plus [`FusionError::Metric`] on an empty test set.
    pub fn accuracy(
        &self,
        trials: &[TrialFeatures],
        dropped: &[usize],
    ) -> Result<f64, FusionError> {
        let mut pred = Vec::with_capacity(trials.len());
        let mut truth = Vec::with_capacity(trials.len());
        for t in trials {
            pred.push(self.predict(t, dropped)? as usize);
            truth.push(t.label as usize);
        }
        Ok(metrics::accuracy(&pred, &truth)?)
    }
}

// =================================================================================================
// Synthetic multimodal generator
// =================================================================================================

/// One sensor's parameters in [`MultimodalSource`].
///
/// The generator's model, written out so every estimator in this module can be checked against it:
/// a source event at common time `t` is reported by this sensor at
///
/// ```text
/// t_reported = t * (1 + drift) + lag_s + clock_offset_s + U(-jitter_s, +jitter_s)
/// ```
///
/// quantised to `round(t_reported / dt_s)` ticks. The drift multiplies elapsed common time only;
/// `lag_s` and `clock_offset_s` are added in the sensor's own frame. That keeps the algebra exact:
/// the offset observed at common time `t` is `lag_s + clock_offset_s + drift * t`, so an offset
/// estimator must recover the **sum** of the first two, and a drift estimator must recover the
/// slope.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModalitySpec {
    /// Tick length of this sensor, seconds. Strictly positive; different sensors having different
    /// values is the "different rates" half of the problem.
    pub dt_s: f64,
    /// Physical transport lag, seconds. May be negative — a sensor can report a common cause before
    /// another one does — provided `lag_s + clock_offset_s` stays non-negative.
    pub lag_s: f64,
    /// Clock epoch offset, seconds. Physically distinct from `lag_s` and **observationally
    /// identical to it**; both are here so a caller can express the distinction they mean, and the
    /// estimators recover only the sum.
    pub clock_offset_s: f64,
    /// Half-width of the uniform detection jitter, seconds. Non-negative; `0.0` makes the sensor
    /// exact up to tick quantisation.
    pub jitter_s: f64,
    /// Probability that this sensor misses a given source event, in `0.0..=1.0`. This is the
    /// dropout that a fusion stack has to survive.
    pub dropout: f64,
    /// Independent background event rate for this sensor, hertz, over the reported interval. The
    /// clutter every coincidence detector's false-positive rate is computed against.
    pub background_hz: f64,
    /// Fractional clock rate error, seconds per second. `2e-4` is 200 ppm. Greater than `-1.0`.
    pub drift: f64,
}

/// A source that drives several senses from one train of underlying events.
///
/// Every modality sees the **same** Poisson source, each through its own lag, jitter, dropout,
/// clock and tick grid, plus its own independent background. That is the structure real multimodal
/// data has and the reason an offset estimate is recoverable at all: the shared cause is the only
/// thing the streams have in common.
#[derive(Debug, Clone, PartialEq)]
pub struct MultimodalSource {
    /// Rate of the shared underlying event train, hertz. Non-negative.
    pub source_rate_hz: f64,
    /// Length of the recording on the common timeline, seconds. Strictly positive.
    pub duration_s: f64,
    /// How many addresses the source can fire on; every modality uses the same address for a given
    /// source event, so a caller wanting a non-trivial correspondence permutes one side afterwards.
    pub addresses: u32,
    /// One entry per sense, numbered from zero in this order.
    pub modalities: Vec<ModalitySpec>,
    /// Cap on events drawn per Poisson process, as a guard against a rate-duration product the
    /// caller did not intend.
    pub event_cap: usize,
}

/// What [`MultimodalSource::generate`] produced, with its own ground truth.
#[derive(Debug, Clone, PartialEq)]
pub struct Generated {
    /// One stream per modality, in spec order, each with a **default** clock: the offsets are what
    /// an estimator is supposed to find, so handing them back inside the clock would make every
    /// alignment test circular.
    pub streams: Vec<Stream>,
    /// The shared source events' common-timeline times, seconds, ascending.
    pub source_times_s: Vec<f64>,
    /// The shared source events' addresses, in the same order.
    pub source_addresses: Vec<u32>,
    /// Per modality, how many source events survived dropout and were emitted.
    pub emitted: Vec<usize>,
}

impl MultimodalSource {
    /// Draw one recording.
    ///
    /// The draw order is fixed — source train first, then each modality's dropout, jitter and
    /// background in spec order — so a given seed reproduces a given recording exactly, and adding
    /// a modality changes only what follows it.
    ///
    /// # Errors
    ///
    /// [`FusionError::NoModalities`] for an empty spec list, [`FusionError::Empty`] for a zero
    /// address count, [`FusionError::NotPositive`], [`FusionError::Negative`],
    /// [`FusionError::NotAProbability`] or [`FusionError::NonFinite`] for an out-of-range
    /// parameter, [`FusionError::NegativeEpoch`] when a sensor's combined shift would make it
    /// report before its own clock zero, [`FusionError::EventCap`] when a Poisson draw exceeds the
    /// cap, and [`FusionError::TickOverflow`] for a time that does not fit in a `u64` tick.
    pub fn generate(&self, rng: &mut Rng) -> Result<Generated, FusionError> {
        if self.modalities.is_empty() {
            return Err(FusionError::NoModalities);
        }
        if self.addresses == 0 {
            return Err(FusionError::Empty { what: "address space" });
        }
        let duration = positive("duration_s", self.duration_s)?;
        let rate = non_negative("source_rate_hz", self.source_rate_hz)?;
        for (i, m) in self.modalities.iter().enumerate() {
            positive("dt_s", m.dt_s)?;
            finite("lag_s", m.lag_s)?;
            finite("clock_offset_s", m.clock_offset_s)?;
            non_negative("jitter_s", m.jitter_s)?;
            probability("dropout", m.dropout)?;
            non_negative("background_hz", m.background_hz)?;
            positive("1 + drift", 1.0 + finite("drift", m.drift)?)?;
            let shift = m.lag_s + m.clock_offset_s;
            if shift - m.jitter_s < 0.0 {
                return Err(FusionError::NegativeEpoch { modality: i, shift_s: shift });
            }
        }

        let source_times_s = poisson_times(rng, rate, 0.0, duration, self.event_cap)?;
        let source_addresses: Vec<u32> =
            source_times_s.iter().map(|_| rng.below(self.addresses)).collect();

        let mut streams = Vec::with_capacity(self.modalities.len());
        let mut emitted = Vec::with_capacity(self.modalities.len());
        for (i, m) in self.modalities.iter().enumerate() {
            let mut events: Vec<Event> = Vec::new();
            let mut n = 0usize;
            for (k, &t) in source_times_s.iter().enumerate() {
                let keep = rng.next_f64() >= m.dropout;
                let j = (2.0 * rng.next_f64() - 1.0) * m.jitter_s;
                let pol = if rng.next_u32() & 1 == 0 { Polarity::Off } else { Polarity::On };
                if !keep {
                    continue;
                }
                let t_rep = t * (1.0 + m.drift) + m.lag_s + m.clock_offset_s + j;
                if t_rep < 0.0 {
                    continue;
                }
                events.push(Event { t: to_tick(t_rep, m.dt_s)?, address: source_addresses[k], polarity: pol });
                n += 1;
            }
            let bg_end = duration * (1.0 + m.drift) + m.lag_s + m.clock_offset_s;
            for t in poisson_times(rng, m.background_hz, 0.0, bg_end.max(0.0), self.event_cap)? {
                let a = rng.below(self.addresses);
                let pol = if rng.next_u32() & 1 == 0 { Polarity::Off } else { Polarity::On };
                events.push(Event { t: to_tick(t, m.dt_s)?, address: a, polarity: pol });
            }
            events.sort_unstable();
            streams.push(Stream::new(
                u16::try_from(i).map_err(|_| FusionError::UnknownModality {
                    modality: i,
                    count: usize::from(u16::MAX),
                })?,
                Clock::new(m.dt_s)?,
                events,
            )?);
            emitted.push(n);
        }
        Ok(Generated { streams, source_times_s, source_addresses, emitted })
    }
}

fn to_tick(t_s: f64, dt_s: f64) -> Result<u64, FusionError> {
    let ticks = (t_s / dt_s).round();
    if !(ticks >= 0.0) || ticks > 9.0e18 {
        return Err(FusionError::TickOverflow { t_s });
    }
    Ok(ticks as u64)
}

// =================================================================================================
// Two tasks, built so early and late fusion must disagree on one and agree on the other
// =================================================================================================

/// The label is whether the two modalities fired **together**, and nothing else.
///
/// Each trial places one event per modality. On a *bound* trial they coincide; on an *unbound*
/// trial they are `async_s` apart, **with the sign drawn at random**. Each modality also emits
/// independent background, so neither stream's own statistics carry the label: the counts are
/// identical, and the timing of either event alone is the shared nuisance onset plus a
/// zero-mean displacement.
///
/// # Why the sign is randomised, which is the whole point
///
/// ⛔ An earlier version of this task used the signed *order* — which modality came first — and
/// late fusion reached **0.625**, well above chance, on a label it was supposed to be unable to
/// see. The reason is worth more than the task: a late combiner that sums per-branch scores is a
/// *linear* function of both streams, and the signed interval `t_a - t_b` is also linear, so the
/// combiner can reconstruct it from two marginal branches even though neither branch alone sees
/// anything. **Late fusion is not blind to cross-modal structure; it is blind to cross-modal
/// structure its combiner cannot express.**
///
/// Randomising the sign makes the discriminative quantity `|t_a - t_b|`, which is not linear in
/// the two streams and has no representation in any weighted sum of per-branch outputs. That is the
/// construction that forces the gap, and it is also the binding-by-synchrony question itself:
/// *were these two events about the same thing?*
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SynchronyTask {
    /// How many trials to draw.
    pub trials: usize,
    /// Length of each trial, seconds.
    pub trial_s: f64,
    /// Separation between the two modalities on an unbound trial, seconds; zero on a bound one.
    pub async_s: f64,
    /// Uniform half-width of each event's timing jitter, seconds. Must stay below `async_s / 2` or
    /// bound and unbound trials overlap.
    pub jitter_s: f64,
    /// Uniform spread of the pair's centre within the trial, seconds — the nuisance that hides the
    /// label from any single stream. Wide compared with `async_s` is what keeps the marginal leak
    /// small.
    pub onset_span_s: f64,
    /// Independent background rate per modality, hertz. Every background event is a chance to bind
    /// the wrong pair, so this is the knob that sets how far below 1.0 early fusion lands.
    pub background_hz: f64,
    /// Half-window the cross-modal pair search uses, seconds. Must exceed `async_s`, or an unbound
    /// pair is never found and the task becomes trivially separable on "no pair at all".
    pub match_half_window_s: f64,
}

impl Default for SynchronyTask {
    /// 400 trials of 400 ms, a 40 ms asynchrony with 1.5 ms jitter, a 200 ms onset spread, 1 Hz
    /// background per modality and a 55 ms match window.
    ///
    /// Round numbers chosen so the analysis holds with margin, not a fit to anything. At 1 Hz the
    /// expected clutter pair closer than 40 ms is about 0.2 per trial, which caps early fusion
    /// near 0.91 rather than at 1.0 — a ceiling below perfection matters, because a task both
    /// architectures saturate proves nothing about either.
    fn default() -> Self {
        Self {
            trials: 400,
            trial_s: 0.4,
            async_s: 40e-3,
            jitter_s: 1.5e-3,
            onset_span_s: 0.2,
            background_hz: 1.0,
            match_half_window_s: 55e-3,
        }
    }
}

/// Per-modality summary features: count, mean event time, first event time.
///
/// The richest single-stream summary at this level of description, and the point is that on
/// [`SynchronyTask`] all three have the same expectation under both labels.
fn block_features(times: &[f64], trial_s: f64) -> Vec<f64> {
    let n = times.len() as f64;
    let mean = if times.is_empty() { trial_s * 0.5 } else { times.iter().sum::<f64>() / n };
    let first = times.first().copied().unwrap_or(trial_s);
    vec![n, mean, first]
}

impl SynchronyTask {
    /// Validate the parameters against each other.
    ///
    /// # Errors
    ///
    /// [`FusionError::Empty`] for zero trials, [`FusionError::NotPositive`],
    /// [`FusionError::Negative`] or [`FusionError::NonFinite`] for an out-of-range parameter, and
    /// [`FusionError::NotPositive`] on `async_s/2 - jitter_s` when the jitter can erase the
    /// asynchrony, on `match_half_window_s - async_s` when an unbound pair could never be found,
    /// or on `trial_s - onset_span_s - async_s` when the pair could fall outside the trial.
    pub fn validate(&self) -> Result<(), FusionError> {
        if self.trials == 0 {
            return Err(FusionError::Empty { what: "trials" });
        }
        positive("trial_s", self.trial_s)?;
        positive("async_s", self.async_s)?;
        non_negative("jitter_s", self.jitter_s)?;
        non_negative("onset_span_s", self.onset_span_s)?;
        non_negative("background_hz", self.background_hz)?;
        positive("match_half_window_s", self.match_half_window_s)?;
        positive("async_s / 2 - jitter_s", 0.5 * self.async_s - self.jitter_s)?;
        positive("match_half_window_s - async_s", self.match_half_window_s - self.async_s)?;
        positive(
            "trial_s - onset_span_s - async_s",
            self.trial_s - self.onset_span_s - self.async_s,
        )?;
        Ok(())
    }

    /// Draw the trials.
    ///
    /// # Errors
    ///
    /// Whatever [`SynchronyTask::validate`] returns, plus [`FusionError::EventCap`] from the
    /// background draw.
    pub fn generate(&self, rng: &mut Rng) -> Result<Vec<TrialFeatures>, FusionError> {
        self.validate()?;
        let cap = ((10.0 * self.background_hz * self.trial_s) as usize).max(1024);
        let mut out = Vec::with_capacity(self.trials);
        for _ in 0..self.trials {
            let label = u32::from(rng.next_u32() & 1 == 1);
            // The SIGN is drawn whatever the label, so the two draws stay in step and the signed
            // interval carries no information under either.
            let sign = if rng.next_u32() & 1 == 1 { 1.0 } else { -1.0 };
            let gap = if label == 1 { 0.0 } else { self.async_s };
            let centre =
                0.5 * (self.trial_s - self.onset_span_s) + rng.next_f64() * self.onset_span_s;
            let ja = (2.0 * rng.next_f64() - 1.0) * self.jitter_s;
            let jb = (2.0 * rng.next_f64() - 1.0) * self.jitter_s;
            let ta = centre - 0.5 * sign * gap + ja;
            let tb = centre + 0.5 * sign * gap + jb;

            let mut a = poisson_times(rng, self.background_hz, 0.0, self.trial_s, cap)?;
            let mut b = poisson_times(rng, self.background_hz, 0.0, self.trial_s, cap)?;
            a.push(ta);
            b.push(tb);
            a.sort_by(f64::total_cmp);
            b.sort_by(f64::total_cmp);

            let d = nearest_cross_pair(&a, &b, self.match_half_window_s);
            out.push(TrialFeatures {
                per_modality: vec![
                    block_features(&a, self.trial_s),
                    block_features(&b, self.trial_s),
                ],
                // The signed interval is kept alongside the absolute one precisely because it is
                // uninformative here: a feature block hand-picked for the task would make the
                // early/late comparison a comparison of feature engineering.
                cross_modal: vec![
                    d.unwrap_or(0.0),
                    d.map_or(self.match_half_window_s, f64::abs),
                ],
                label,
            });
        }
        Ok(out)
    }
}

/// The label is each modality's own **rate**, redundantly, in both streams.
///
/// The complement of [`SynchronyTask`]: here every stream carries the label on its own, nothing is in
/// the interval between them, and the cross-modal features are pure clutter. Late fusion has
/// everything it needs, and the gap that the temporal task opens collapses.
///
/// Having both is what stops "early beats late" from being read as a ranking. It is a statement
/// about **where the information is**, and this type is the control that says so.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RedundantRateTask {
    /// How many trials to draw.
    pub trials: usize,
    /// Length of each trial, seconds.
    pub trial_s: f64,
    /// Event rate of both modalities for label 0, hertz.
    pub rate_lo_hz: f64,
    /// Event rate of both modalities for label 1, hertz. Must exceed `rate_lo_hz`.
    pub rate_hi_hz: f64,
    /// Half-window the (uninformative) cross-modal pair search uses, seconds.
    pub match_half_window_s: f64,
}

impl Default for RedundantRateTask {
    /// 1200 trials of 400 ms at 20 Hz against 40 Hz, with a 25 ms match window.
    ///
    /// The rates are chosen to leave a measurable gap between one sense and two at this trial
    /// count. Counts of 8 and 16 with Poisson spreads of 2.83 and 4.00 give a discriminability of
    /// about `8 / 3.4 = 2.35` for one stream and `2.35 * sqrt(2) = 3.3` for two — both plainly
    /// above chance, and separated by more than the sampling error of 1200 trials. A larger gap
    /// would saturate both at 1.0 and make the dropout comparison vacuous.
    fn default() -> Self {
        Self {
            trials: 1200,
            trial_s: 0.4,
            rate_lo_hz: 20.0,
            rate_hi_hz: 40.0,
            match_half_window_s: 25e-3,
        }
    }
}

impl RedundantRateTask {
    /// Validate the parameters.
    ///
    /// # Errors
    ///
    /// [`FusionError::Empty`] for zero trials, and [`FusionError::NotPositive`],
    /// [`FusionError::Negative`] or [`FusionError::NonFinite`] for an out-of-range parameter,
    /// including on `rate_hi_hz - rate_lo_hz` when the two classes are not separated at all.
    pub fn validate(&self) -> Result<(), FusionError> {
        if self.trials == 0 {
            return Err(FusionError::Empty { what: "trials" });
        }
        positive("trial_s", self.trial_s)?;
        non_negative("rate_lo_hz", self.rate_lo_hz)?;
        positive("rate_hi_hz - rate_lo_hz", self.rate_hi_hz - self.rate_lo_hz)?;
        positive("match_half_window_s", self.match_half_window_s)?;
        Ok(())
    }

    /// Draw the trials.
    ///
    /// # Errors
    ///
    /// Whatever [`RedundantRateTask::validate`] returns, plus [`FusionError::EventCap`].
    pub fn generate(&self, rng: &mut Rng) -> Result<Vec<TrialFeatures>, FusionError> {
        self.validate()?;
        let cap = ((20.0 * self.rate_hi_hz * self.trial_s) as usize).max(1024);
        let mut out = Vec::with_capacity(self.trials);
        for _ in 0..self.trials {
            let label = u32::from(rng.next_u32() & 1 == 1);
            let rate = if label == 1 { self.rate_hi_hz } else { self.rate_lo_hz };
            let a = poisson_times(rng, rate, 0.0, self.trial_s, cap)?;
            let b = poisson_times(rng, rate, 0.0, self.trial_s, cap)?;
            let d = nearest_cross_pair(&a, &b, self.match_half_window_s);
            out.push(TrialFeatures {
                per_modality: vec![
                    block_features(&a, self.trial_s),
                    block_features(&b, self.trial_s),
                ],
                cross_modal: vec![
                    d.unwrap_or(0.0),
                    d.map_or(self.match_half_window_s, f64::abs),
                ],
                label,
            });
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Aligned, Clock, CoincidenceDetector, CoincidenceReport, CrossModalAssociator, EarlyFusion,
        FusionError,
        Generated, LateFusion, SynchronyTask, ModalitySpec, MultimodalSource, NearestCentroid,
        OffsetEstimator, PrecisionGate, RedundantRateTask, Stream, TrialFeatures,
        chance_binding_probability, chance_partners_per_anchor, gaussian, nearest_cross_pair,
        poisson_times,
    };
    use crate::plasticity::{Bounds, PairStdp, WeightRule};
    use crate::rng::Rng;
    use crate::spike::{Event, Polarity};

    // ------------------------------------------------------------------------------------------
    // Fixtures
    // ------------------------------------------------------------------------------------------

    /// Two sensors on one source. Modality 0 is the reference; modality 1 carries `lag_b`.
    ///
    /// Both sensors sit on a 100 ms clock offset so that a NEGATIVE relative lag is still a legal
    /// recording — a sensor cannot report before its own clock zero, and without the common
    /// pedestal half of the sweep below would be unreachable.
    fn two_sensors(
        lag_b: f64,
        jitter: f64,
        dropout: f64,
        background_hz: f64,
        duration_s: f64,
    ) -> MultimodalSource {
        let base = ModalitySpec {
            dt_s: 1e-4,
            lag_s: 0.0,
            clock_offset_s: 0.100,
            jitter_s: jitter,
            dropout,
            background_hz,
            drift: 0.0,
        };
        MultimodalSource {
            source_rate_hz: 20.0,
            duration_s,
            addresses: 4,
            modalities: vec![base, ModalitySpec { lag_s: lag_b, ..base }],
            event_cap: 1_000_000,
        }
    }

    fn estimator() -> OffsetEstimator {
        OffsetEstimator::new(0.120, 1e-3, 8e-3).expect("widths are consistent")
    }

    fn generate(src: &MultimodalSource, seed: u64) -> Generated {
        src.generate(&mut Rng::new(seed)).expect("parameters are in range")
    }

    // ------------------------------------------------------------------------------------------
    // Determinism and the generator's own contract
    // ------------------------------------------------------------------------------------------

    #[test]
    fn the_same_seed_gives_the_same_recording() {
        let src = two_sensors(7e-3, 1e-3, 0.1, 3.0, 5.0);
        assert_eq!(generate(&src, 99), generate(&src, 99));
        assert_ne!(generate(&src, 99), generate(&src, 100));
    }

    /// Dropout is a probability, not a decoration: the emitted count has to track it. Without this
    /// a generator that ignored `dropout` entirely would pass every other test in this module,
    /// because every downstream estimator is supposed to survive dropout.
    #[test]
    fn dropout_removes_about_the_fraction_it_names() {
        let src = two_sensors(0.0, 0.0, 0.35, 0.0, 60.0);
        let g = generate(&src, 3);
        let n = g.source_times_s.len() as f64;
        assert!(n > 800.0, "only {n} source events; the test has no power");
        let kept = g.emitted[1] as f64 / n;
        // 4 binomial standard errors at this n.
        let se = (0.35 * 0.65 / n).sqrt();
        assert!((kept - 0.65).abs() < 4.0 * se, "kept {kept}, expected 0.65 +- {:.4}", 4.0 * se);
        // Modality 0 drew its own dropout mask: the two must not be the same events.
        assert_ne!(g.emitted[0], 0);
    }

    #[test]
    fn a_sensor_that_would_report_before_its_own_clock_zero_is_refused() {
        let mut src = two_sensors(0.0, 0.0, 0.0, 0.0, 1.0);
        src.modalities[1].lag_s = -0.5;
        let e = src.generate(&mut Rng::new(1)).unwrap_err();
        assert!(matches!(e, FusionError::NegativeEpoch { modality: 1, .. }), "{e}");
    }

    #[test]
    fn a_stream_whose_events_go_backwards_is_refused() {
        let e = Stream::new(
            0,
            Clock::new(1e-3).unwrap(),
            vec![
                Event { t: 10, address: 0, polarity: Polarity::On },
                Event { t: 9, address: 0, polarity: Polarity::On },
            ],
        )
        .unwrap_err();
        assert!(matches!(e, FusionError::Unsorted { index: 1 }), "{e}");
    }

    // ------------------------------------------------------------------------------------------
    // (e) Alignment with nothing to correct is the identity
    // ------------------------------------------------------------------------------------------

    /// ⭐ Verification (e). With zero lag and zero jitter the estimated offset is **exactly** zero
    /// and the aligned timeline is the reported one, bit for bit — not to a tolerance.
    #[test]
    fn with_no_lag_and_no_jitter_alignment_is_the_identity() {
        let src = two_sensors(0.0, 0.0, 0.0, 0.0, 20.0);
        let g = generate(&src, 11);
        let fit = estimator().estimate(&g.streams[0], &g.streams[1]).expect("a peak exists");
        assert_eq!(fit.offset_s, 0.0, "identical sensors disagreed by {}", fit.offset_s);
        assert_eq!(fit.mad_s, 0.0, "identical sensors had spread {}", fit.mad_s);

        let aligned = Aligned::merge(&g.streams).expect("two streams");
        let reported: Vec<f64> = g.streams[0].reported_s();
        let back: Vec<f64> = aligned.times_of(0);
        assert_eq!(reported, back, "the identity clock moved a timestamp");
    }

    /// The other half of (e): a clock that *does* carry an offset moves every aligned time by
    /// exactly that offset, so the alignment is a pure translation and not a resampling.
    #[test]
    fn a_recorded_offset_translates_the_timeline_and_nothing_else() {
        let src = two_sensors(0.0, 0.0, 0.0, 0.0, 5.0);
        let g = generate(&src, 12);
        let plain = g.streams[1].reported_s();
        let shifted = Stream::new(
            1,
            Clock::with_offset(1e-4, 0.037, 0.0).unwrap(),
            g.streams[1].events().to_vec(),
        )
        .unwrap();
        let aligned = Aligned::merge(&[shifted]).unwrap().times_of(1);
        assert_eq!(plain.len(), aligned.len());
        for (p, a) in plain.iter().zip(&aligned) {
            assert!((p - 0.037 - a).abs() < 1e-12, "{p} - 0.037 != {a}");
        }
    }

    // ------------------------------------------------------------------------------------------
    // (a) ⭐ The estimated offset recovers the one the generator used
    // ------------------------------------------------------------------------------------------

    /// ⭐ Verification (a). Swept across lags of both signs, four jitters and two dropouts.
    ///
    /// The tolerance is not a guess. For symmetric jitter of half-width `J` the pairwise difference
    /// has density `1/(2J)` at its centre, so the median's standard error over `n` true pairs is
    /// `J/sqrt(n)`; tick quantisation adds `dt/2` to `J`. The bound asserted is **six** of those
    /// standard errors plus half a tick, computed from the generator's own parameters at each sweep
    /// point rather than fixed.
    #[test]
    fn the_estimated_offset_recovers_the_generators_across_lags_and_jitter() {
        let duration = 50.0;
        let est = estimator();
        let mut worst: f64 = 0.0;
        for &lag in &[-40e-3, -9e-3, 0.0, 3.5e-3, 40e-3] {
            for &jitter in &[0.0, 0.2e-3, 1e-3, 3e-3] {
                for &dropout in &[0.0, 0.4] {
                    let src = two_sensors(lag, jitter, dropout, 1.0, duration);
                    let g = generate(&src, 2_024);
                    let fit = est.estimate(&g.streams[0], &g.streams[1]).expect("a peak exists");

                    let n_true = duration * src.source_rate_hz * (1.0 - dropout) * (1.0 - dropout);
                    let j_eff = jitter + 0.5 * src.modalities[0].dt_s;
                    let tol = 6.0 * j_eff / n_true.sqrt() + 0.5 * src.modalities[0].dt_s;
                    let err = (fit.offset_s - lag).abs();
                    worst = worst.max(err / tol);
                    assert!(
                        err < tol,
                        "lag {lag}, jitter {jitter}, dropout {dropout}: \
                         estimated {} off by {err:.3e}, tolerance {tol:.3e}",
                        fit.offset_s
                    );
                }
            }
        }
        // The sweep must be a real test at its tightest point, not a wide net: if the worst point
        // used under a twentieth of its budget the tolerances are too loose to catch anything.
        assert!(worst > 0.05, "the tightest sweep point used only {worst:.3} of its tolerance");
    }

    /// Catches an estimator that returns a constant, which the sweep alone cannot: shifting every
    /// tick of one stream by a known amount must shift the estimate by exactly that amount.
    #[test]
    fn the_offset_estimate_is_equivariant_under_a_known_extra_shift() {
        let src = two_sensors(5e-3, 1e-3, 0.0, 1.0, 30.0);
        let g = generate(&src, 77);
        let est = estimator();
        let base = est.estimate(&g.streams[0], &g.streams[1]).unwrap().offset_s;
        let bump = 170u64; // ticks, = 17 ms at dt = 1e-4
        let moved: Vec<Event> =
            g.streams[1].events().iter().map(|e| Event { t: e.t + bump, ..*e }).collect();
        let shifted = Stream::new(1, Clock::new(1e-4).unwrap(), moved).unwrap();
        let after = est.estimate(&g.streams[0], &shifted).unwrap().offset_s;
        assert!(
            (after - base - 17e-3).abs() < 1e-9,
            "shifting by 17 ms moved the estimate by {}",
            after - base
        );
    }

    /// ⚠ The limitation, asserted rather than only documented: a transport lag and a clock offset
    /// are observationally identical, and the estimator recovers their sum. Two recordings that
    /// split the same total differently must give the same answer.
    #[test]
    fn lag_and_clock_offset_are_recovered_only_as_their_sum() {
        let est = estimator();
        let mut src = two_sensors(0.0, 0.5e-3, 0.0, 0.0, 30.0);
        src.modalities[1].lag_s = 9e-3;
        src.modalities[1].clock_offset_s = 0.100;
        let a = est
            .estimate(&generate(&src, 5).streams[0], &generate(&src, 5).streams[1])
            .unwrap()
            .offset_s;
        src.modalities[1].lag_s = 0.0;
        src.modalities[1].clock_offset_s = 0.109;
        let b = est
            .estimate(&generate(&src, 5).streams[0], &generate(&src, 5).streams[1])
            .unwrap()
            .offset_s;
        assert!((a - 9e-3).abs() < 2e-4, "sum not recovered: {a}");
        assert!((a - b).abs() < 1e-12, "the split changed the answer: {a} vs {b}");
    }

    /// The drift half of the clock model: a 100 ppm crystal over 40 s, recovered from the
    /// difference between the two half-records.
    #[test]
    fn the_drift_estimate_recovers_a_hundred_ppm_clock() {
        // A NON-ZERO offset as well as a drift: with a zero offset, reporting the first half's
        // measurement instead of extrapolating back to time zero is wrong by only drift * c1,
        // which the tolerance would have absorbed.
        let mut src = two_sensors(5e-3, 0.5e-3, 0.0, 0.5, 40.0);
        src.source_rate_hz = 30.0;
        src.modalities[1].drift = 1e-4;
        let g = generate(&src, 31);
        let est = OffsetEstimator::new(0.02, 5e-4, 5e-3).unwrap();
        let fit = est.estimate_drift(&g.streams[0], &g.streams[1]).expect("both halves resolve");

        assert!(fit.span_s > 15.0, "lever arm was only {} s", fit.span_s);
        assert!(
            (fit.drift - 1e-4).abs() < 3e-5,
            "drift {:.3e} against a true 1.0e-4",
            fit.drift
        );
        assert!(
            (fit.offset_s - 5e-3).abs() < 3e-4,
            "offset at common time zero was {} against a true 5.0e-3",
            fit.offset_s
        );
        // Reporting the first half's measurement unextrapolated would give 5 ms + drift*c1 ~ 6 ms,
        // which the bound above rejects — and this asserts that the two really do differ here.
        assert!(
            (fit.first_half_s - fit.offset_s).abs() > 5e-4,
            "the extrapolation moved nothing: {} vs {}",
            fit.first_half_s,
            fit.offset_s
        );
        // The two halves must actually disagree, or the slope came from nothing.
        assert!(
            fit.second_half_s - fit.first_half_s > 1e-3,
            "the halves differed by only {}",
            fit.second_half_s - fit.first_half_s
        );
    }

    /// A zero-drift recording must not grow a drift out of estimator noise.
    #[test]
    fn a_stable_clock_reports_no_drift() {
        let mut src = two_sensors(3e-3, 0.5e-3, 0.0, 0.5, 40.0);
        src.source_rate_hz = 30.0;
        let g = generate(&src, 32);
        let est = OffsetEstimator::new(0.02, 5e-4, 5e-3).unwrap();
        let fit = est.estimate_drift(&g.streams[0], &g.streams[1]).unwrap();
        assert!(fit.drift.abs() < 2e-5, "invented a drift of {:.3e}", fit.drift);
        assert!((fit.offset_s - 3e-3).abs() < 3e-4, "offset {}", fit.offset_s);
    }

    /// The spread is the confidence signal. Two streams with a shared cause give a cluster whose
    /// median absolute deviation is the jitter; two independent streams give a flat background
    /// filling the refinement window, and the estimator's `offset_s` is then meaningless even
    /// though it is a number.
    /// ⛔ THE ERROR BAR AGAINST THE SCATTER IT CLAIMS TO DESCRIBE. This is the test the previous
    /// formula never had: `mad_s / sqrt(pairs)` reported 0.37 of the true seed-to-seed standard
    /// deviation of the estimate, and the only assertion touching it was scale-free in `n`, so
    /// deleting the `sqrt` entirely passed 52/52.
    ///
    /// The protocol is the one that found it: 20 Hz Poisson source, 1 ms jitter, 0.1 ms ticks,
    /// 60 s, 5 ms lag, backgrounds of 0 and 5 Hz. The reported standard error, averaged over
    /// seeds, is compared with the actual standard deviation of the reported offsets across those
    /// seeds. The band is `[1.0, 1.5]`: below 1 is under-reporting, which the shape bound says
    /// cannot happen for a symmetric difference and which every previous defect produced; above
    /// 1.5 is further than the identified mechanism (MAD inflated 1.27× by cross-pairs) can carry
    /// it. Measured at 1.2–1.3 when written. 100 seeds put the ratio's own uncertainty near 7%.
    ///
    /// Not asserted at 240 s: with 4,800 events the median locks to the 0.1 ms tick grid and the
    /// seed-to-seed scatter collapses to roundoff, which is a fact about the protocol's
    /// quantisation and not about the estimator.
    #[test]
    fn the_standard_error_matches_the_scatter_it_describes() {
        let est = estimator();
        for &bg in &[0.0f64, 5.0] {
            let src = two_sensors(5e-3, 1e-3, 0.0, bg, 60.0);
            let seeds = 100u64;
            let mut offsets = Vec::with_capacity(seeds as usize);
            let mut se_sum = 0.0;
            for seed in 0..seeds {
                let g = generate(&src, 1000 + seed);
                let fit = est.estimate(&g.streams[0], &g.streams[1]).unwrap();
                offsets.push(fit.offset_s);
                se_sum += fit.standard_error_s().unwrap();
                // The independent-event count is the true event count, not the pair count.
                let n_true = 60.0 * src.source_rate_hz;
                assert!(
                    (fit.events as f64) < 1.2 * n_true && (fit.events as f64) > 0.8 * n_true,
                    "seed {seed}: {} events against {n_true} source events", fit.events
                );
                assert!(fit.pairs >= fit.events, "pairs {} < events {}", fit.pairs, fit.events);
            }
            let k = seeds as f64;
            let mean = offsets.iter().sum::<f64>() / k;
            let sd = (offsets.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (k - 1.0)).sqrt();
            let ratio = (se_sum / k) / sd;
            println!("CALIBRATION bg={bg:.0}Hz: reported {:.3e}, actual {sd:.3e}, ratio {ratio:.3}", se_sum / k);
            assert!(
                (1.0..=1.5).contains(&ratio),
                "bg {bg} Hz: reported SE is {ratio:.3} of the actual scatter; the band is [1.0, 1.5]"
            );
        }
    }

    #[test]
    fn independent_streams_are_distinguishable_by_their_spread() {
        let src = two_sensors(0.0, 1e-3, 0.0, 0.0, 60.0);
        let correlated = generate(&src, 41);
        let est = estimator();
        let good = est.estimate(&correlated.streams[0], &correlated.streams[1]).unwrap();

        // Two recordings from different seeds share no source event at all.
        let other = generate(&src, 42);
        let bad = est.estimate(&correlated.streams[0], &other.streams[1]).unwrap();

        // A cluster's spread is set by the JITTER; a background's is set by the REFINEMENT
        // WINDOW, which for a flat distribution over +-8 ms has a median absolute deviation of
        // 4 ms. Those are the two references, so neither side of the comparison is a number
        // chosen after seeing the output.
        assert!(good.mad_s < 1.5e-3, "a real cluster had spread {} against a 1 ms jitter", good.mad_s);
        assert!(
            bad.mad_s > 0.35 * est.refine_half_s,
            "independent streams spread only {}; a flat window would give {}",
            bad.mad_s,
            0.5 * est.refine_half_s
        );
        assert!(bad.mad_s > 3.0 * good.mad_s, "{} vs {}", bad.mad_s, good.mad_s);
        // And therefore the reported standard error separates them too, which is the number a
        // caller actually reads.
        let (se_good, se_bad) = (good.standard_error_s().unwrap(), bad.standard_error_s().unwrap());
        assert!(se_bad > 2.0 * se_good, "standard errors {se_good:.2e} and {se_bad:.2e}");
    }

    #[test]
    fn too_few_pairs_is_a_refusal_and_not_a_number() {
        let src = two_sensors(0.0, 0.0, 0.0, 0.0, 0.3);
        let g = generate(&src, 51);
        let mut est = estimator();
        est.min_pairs = 100_000;
        let e = est.estimate(&g.streams[0], &g.streams[1]).unwrap_err();
        assert!(matches!(e, FusionError::TooFewPairs { needed: 100_000, .. }), "{e}");
    }

    #[test]
    fn the_pair_sweep_is_capped_rather_than_exhausting_memory() {
        let src = two_sensors(0.0, 0.0, 0.0, 0.0, 30.0);
        let g = generate(&src, 52);
        let mut est = estimator();
        est.max_pairs = 8;
        let e = est.estimate(&g.streams[0], &g.streams[1]).unwrap_err();
        assert!(matches!(e, FusionError::TooManyPairs { cap: 8 }), "{e}");
    }

    #[test]
    fn inconsistent_estimator_widths_are_refused() {
        assert!(OffsetEstimator::new(1e-3, 2e-3, 1e-3).is_err());
        assert!(OffsetEstimator::new(1e-3, 1e-3, 2e-3).is_err());
        assert!(OffsetEstimator::new(0.0, 1e-3, 1e-3).is_err());
        assert!(OffsetEstimator::new(f64::NAN, 1e-3, 1e-3).is_err());
        assert!(OffsetEstimator::new(1e-1, 1e-3, 1e-3).is_ok());
    }

    // ------------------------------------------------------------------------------------------
    // (b) ⭐ Coincidence detection against its closed form
    // ------------------------------------------------------------------------------------------

    /// Independent Poisson streams on one timeline, at a tick fine enough that quantisation is
    /// three orders below the narrowest window under test.
    fn independent_streams(rng: &mut Rng, rates_hz: &[f64], t_end: f64) -> Aligned {
        let dt = 1e-7;
        let streams: Vec<Stream> = rates_hz
            .iter()
            .enumerate()
            .map(|(m, &r)| {
                let times = poisson_times(rng, r, 0.0, t_end, 10_000_000).expect("in range");
                let events: Vec<Event> = times
                    .iter()
                    .map(|&t| Event {
                        t: (t / dt).round() as u64,
                        address: 0,
                        polarity: Polarity::On,
                    })
                    .collect();
                Stream::new(m as u16, Clock::new(dt).unwrap(), events).expect("sorted")
            })
            .collect();
        Aligned::merge(&streams).expect("streams exist")
    }

    /// ⭐ Verification (b). Under independence the detector's output is predicted exactly by the
    /// window width and the partner rate, and both closed forms are asserted:
    ///
    /// - mean partners per anchor = `2*w*r`, exactly;
    /// - probability of at least one partner = `1 - exp(-2*w*r)`, exactly.
    ///
    /// Tolerances are four standard errors computed from the Poisson and binomial variances at the
    /// anchor count actually observed — not a fixed fraction.
    #[test]
    fn chance_coincidence_matches_the_closed_form_across_windows_and_rates() {
        let t_end = 200.0;
        for (k, &(w, r_b)) in
            [(1e-3, 50.0), (1e-3, 150.0), (5e-3, 50.0), (5e-3, 150.0)].iter().enumerate()
        {
            let mut rng = Rng::new(900 + k as u64);
            let aligned = independent_streams(&mut rng, &[30.0, r_b], t_end);
            let measured_r = aligned.rate_hz_in(1, 0.0, t_end).expect("a window with length");
            let det = CoincidenceDetector::new(w, 0, vec![1]).unwrap();
            let rep = det.detect(&aligned, 0.0, t_end).expect("the run outlasts the window");
            let n = rep.anchors_considered as f64;
            assert!(n > 4_000.0, "only {n} anchors; the test has no power");

            // Closed form 1: the mean. Poisson count variance equals its mean, so the standard
            // error of the average over n anchors is sqrt(2*w*r / n).
            let want_mean = chance_partners_per_anchor(w, measured_r).unwrap();
            let got_mean = rep.mean_partners().unwrap();
            let se_mean = (want_mean / n).sqrt();
            assert!(
                (got_mean - want_mean).abs() < 4.0 * se_mean,
                "w={w} r={measured_r:.2}: mean partners {got_mean:.4} against 2wr={want_mean:.4}                  (+-{:.4})",
                4.0 * se_mean
            );

            // Closed form 2: the probability.
            let want_p = chance_binding_probability(w, &[measured_r]).unwrap();
            let got_p = rep.binding_rate().unwrap();
            let se_p = (want_p * (1.0 - want_p) / n).sqrt();
            assert!(
                (got_p - want_p).abs() < 4.0 * se_p,
                "w={w} r={measured_r:.2}: binding rate {got_p:.4} against                  1-exp(-2wr)={want_p:.4} (+-{:.4})",
                4.0 * se_p
            );

            // ⛔ THE ANTI-VACUITY CLAUSE. The commonest way to get this wrong is to use the
            // window's half-width where its full width belongs, which is a factor of two in the
            // exponent. The assertions above must REJECT that answer, or they are decoration.
            let half = chance_binding_probability(0.5 * w, &[measured_r]).unwrap();
            assert!(
                (got_p - half).abs() > 4.0 * se_p,
                "a half-width window would also have passed: {got_p:.4} vs {half:.4}"
            );
        }
    }

    /// Binding to several partners is a CONJUNCTION. A detector that took the disjunction would
    /// report 0.51 where the product is 0.09, and would look more sensitive rather than wrong.
    #[test]
    fn three_way_binding_is_the_product_of_the_factors_and_not_their_union() {
        let t_end = 200.0;
        let mut rng = Rng::new(913);
        let aligned = independent_streams(&mut rng, &[30.0, 90.0, 60.0], t_end);
        let (w, t0, t1) = (2e-3, 0.0, t_end);
        let r1 = aligned.rate_hz_in(1, t0, t1).unwrap();
        let r2 = aligned.rate_hz_in(2, t0, t1).unwrap();
        let det = CoincidenceDetector::new(w, 0, vec![1, 2]).unwrap();
        let rep = det.detect(&aligned, t0, t1).unwrap();
        let n = rep.anchors_considered as f64;

        let want = chance_binding_probability(w, &[r1, r2]).unwrap();
        let got = rep.binding_rate().unwrap();
        let se = (want * (1.0 - want) / n).sqrt();
        assert!((got - want).abs() < 4.0 * se, "three-way {got:.4} against {want:.4}");

        let union = 1.0 - (1.0 - (1.0 - (-2.0 * w * r1).exp())) * (1.0 - (1.0 - (-2.0 * w * r2).exp()));
        assert!(
            (got - union).abs() > 4.0 * se,
            "a disjunction would also have passed: {got:.4} vs {union:.4}"
        );
        // And the conjunction must be strictly harder than either pairwise binding.
        let pair = CoincidenceDetector::new(w, 0, vec![1]).unwrap().detect(&aligned, t0, t1).unwrap();
        assert!(rep.bound < pair.bound, "{} vs {}", rep.bound, pair.bound);
    }

    /// A correlated pair binds far above chance — otherwise the closed form above would be the
    /// whole story and the detector would be measuring nothing but its own window.
    #[test]
    fn a_shared_source_binds_far_above_the_chance_rate() {
        let src = two_sensors(2e-3, 0.5e-3, 0.0, 5.0, 60.0);
        let g = generate(&src, 55);
        let aligned = Aligned::merge(&g.streams).unwrap();
        let (t0, t1) = (0.2, 59.0);
        let r1 = aligned.rate_hz_in(1, t0, t1).unwrap();
        let det = CoincidenceDetector::new(4e-3, 0, vec![1]).unwrap();
        let rep = det.detect(&aligned, t0, t1).unwrap();
        let chance = chance_binding_probability(4e-3, &[r1]).unwrap();
        let got = rep.binding_rate().unwrap();
        assert!(chance < 0.35, "the chance floor was already {chance:.3}");
        assert!(got > 0.75, "a shared source bound only {got:.3}");
        assert!(got > 2.0 * chance, "{got:.3} against a chance rate of {chance:.3}");
    }

    #[test]
    fn a_detector_refuses_an_interval_shorter_than_its_own_window() {
        let mut rng = Rng::new(7);
        let aligned = independent_streams(&mut rng, &[50.0, 50.0], 1.0);
        let det = CoincidenceDetector::new(1.0, 0, vec![1]).unwrap();
        assert!(det.detect(&aligned, 0.0, 1.0).is_err());
        assert!(det.detect(&aligned, 0.0, f64::NAN).is_err());
        assert!(det.detect(&aligned, 0.0, 3.0).is_ok());
    }

    #[test]
    fn binding_to_no_partner_at_all_is_refused() {
        assert!(CoincidenceDetector::new(1e-3, 0, vec![]).is_err());
        assert!(CoincidenceDetector::new(0.0, 0, vec![1]).is_err());
        assert!(chance_binding_probability(1e-3, &[]).is_err());
        assert!(chance_binding_probability(-1e-3, &[10.0]).is_err());
        assert!(chance_partners_per_anchor(1e-3, f64::NAN).is_err());
    }

    /// The closed forms themselves, at the two limits where they are exactly known.
    #[test]
    fn the_chance_formulae_are_right_at_their_limits() {
        // Zero window: no partner can be inside it.
        assert_eq!(chance_partners_per_anchor(0.0, 1e6).unwrap(), 0.0);
        assert_eq!(chance_binding_probability(0.0, &[1e6]).unwrap(), 0.0);
        // Zero rate: nothing to be inside it.
        assert_eq!(chance_binding_probability(1.0, &[0.0]).unwrap(), 0.0);
        // A very large product saturates at one, without overflowing.
        assert!(chance_binding_probability(1.0, &[1e4]).unwrap() > 1.0 - 1e-12);
        // And the mean is linear in both arguments, exactly.
        let a = chance_partners_per_anchor(3e-3, 40.0).unwrap();
        assert!((a - 0.24).abs() < 1e-15, "{a}");
    }

    // ------------------------------------------------------------------------------------------
    // (c) Attention, and what a dropout costs
    // ------------------------------------------------------------------------------------------

    /// The inverse-variance rule, against the algebra. Ernst & Banks (2002) in three lines.
    #[test]
    fn precision_weights_and_fused_variance_match_the_closed_form() {
        let v = vec![4.0, 1.0, 16.0];
        let gate = PrecisionGate::new(v.clone()).unwrap();
        let total: f64 = v.iter().map(|x| 1.0 / x).sum();
        let w = gate.weights().unwrap();
        for (i, wi) in w.iter().enumerate() {
            assert!((wi - (1.0 / v[i]) / total).abs() < 1e-15, "weight {i} = {wi}");
        }
        assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-15);
        assert!((gate.fused_variance().unwrap() - 1.0 / total).abs() < 1e-15);
        // The most precise sense dominates, and it is the one with the SMALLEST variance.
        assert_eq!(gate.dominant(), Some(1));
        // The fused variance must beat every single channel, which is the whole reason to fuse.
        assert!(gate.fused_variance().unwrap() < v.iter().copied().fold(f64::MAX, f64::min));
    }

    /// The textbook result stated as a number: two equally reliable senses give `sigma/sqrt(2)`.
    #[test]
    fn two_equally_reliable_senses_give_sigma_over_root_two() {
        let sigma = 0.7;
        let gate = PrecisionGate::new(vec![sigma * sigma, sigma * sigma]).unwrap();
        let fused = gate.fused_variance().unwrap().sqrt();
        assert!((fused - sigma / 2.0f64.sqrt()).abs() < 1e-15, "{fused}");
        assert_eq!(gate.weights().unwrap(), vec![0.5, 0.5]);
    }

    /// ⭐ Verification (c), analytically. A lost sense multiplies the fused variance by exactly the
    /// ratio of the precision sums — a finite number, which is what "degrades rather than
    /// destroys" means when it is written down.
    #[test]
    fn a_dropped_modality_costs_exactly_the_precision_it_carried() {
        // Two equal senses: losing one doubles the variance, so the error grows by sqrt(2).
        let mut two = PrecisionGate::new(vec![1.0, 1.0]).unwrap();
        assert_eq!(two.degradation_from_dropping(0).unwrap(), Some(2.0));
        let before = two.fused_variance().unwrap();
        two.set_available(0, false).unwrap();
        assert!((two.fused_variance().unwrap() / before - 2.0).abs() < 1e-15);
        assert_eq!(two.weights().unwrap(), vec![0.0, 1.0]);
        // Losing it again is free; losing the last one is not a degradation at all.
        assert_eq!(two.degradation_from_dropping(0).unwrap(), Some(1.0));
        assert_eq!(two.degradation_from_dropping(1).unwrap(), None);
        two.set_available(1, false).unwrap();
        assert!(two.fused_variance().is_none());
        assert!(two.weights().is_none());
        assert!(two.fuse(&[1.0, 2.0]).unwrap().is_none());

        // Three equal senses: 3/2, exactly.
        let three = PrecisionGate::new(vec![2.0, 2.0, 2.0]).unwrap();
        assert!((three.degradation_from_dropping(2).unwrap().unwrap() - 1.5).abs() < 1e-15);

        // An unequal trio: the expensive loss is the precise sense, and the cheap one is not free.
        let mixed = PrecisionGate::new(vec![0.01, 1.0, 1.0]).unwrap();
        let costly = mixed.degradation_from_dropping(0).unwrap().unwrap();
        let cheap = mixed.degradation_from_dropping(1).unwrap().unwrap();
        assert!(costly > 30.0, "losing the precise sense cost only {costly}");
        assert!(cheap > 1.0 && cheap < 1.02, "losing a weak sense cost {cheap}");
    }

    /// The analytic variance has to be the variance the fused estimate actually has. Monte Carlo
    /// against the closed form, because a weight formula can be right and a fusion loop still
    /// apply it to the wrong channel.
    #[test]
    fn the_fused_estimate_has_the_variance_the_gate_predicts() {
        let v = [0.25, 1.0, 4.0];
        let gate = PrecisionGate::new(v.to_vec()).unwrap();
        let predicted = gate.fused_variance().unwrap();
        let mut rng = Rng::new(4242);
        let n = 200_000;
        let (mut s, mut s2) = (0.0f64, 0.0f64);
        for _ in 0..n {
            let est: Vec<f64> = v.iter().map(|sig2| gaussian(&mut rng) * sig2.sqrt()).collect();
            let f = gate.fuse(&est).unwrap().unwrap();
            s += f;
            s2 += f * f;
        }
        let mean = s / f64::from(n);
        let var = s2 / f64::from(n) - mean * mean;
        // The relative standard error of a sample variance is sqrt(2/n) = 0.32% here; 3% is ten of
        // them, tight enough to reject the next-simplest wrong answer (an unweighted mean, which
        // would give (0.25+1+4)/9 = 0.583 against 0.190).
        assert!(mean.abs() < 0.01, "the fusion is biased: {mean}");
        assert!(
            (var / predicted - 1.0).abs() < 0.03,
            "empirical {var:.5} against predicted {predicted:.5}"
        );
        let unweighted = v.iter().sum::<f64>() / 9.0;
        assert!((var - unweighted).abs() > 0.1, "an unweighted mean would also have passed");
    }

    /// The gate can learn its own weights from how wrong each sense has been.
    #[test]
    fn the_gate_recovers_the_variances_it_was_shown() {
        let truth: [f64; 2] = [0.09, 2.25];
        let mut rng = Rng::new(17);
        let residuals: Vec<Vec<f64>> = truth
            .iter()
            .map(|s2| (0..20_000).map(|_| gaussian(&mut rng) * s2.sqrt()).collect())
            .collect();
        let gate = PrecisionGate::from_residuals(&residuals).unwrap();
        let w = gate.weights().unwrap();
        let want0 = (1.0 / truth[0]) / (1.0 / truth[0] + 1.0 / truth[1]);
        assert!((w[0] - want0).abs() < 0.01, "learned weight {} against {want0}", w[0]);
        assert_eq!(gate.dominant(), Some(0));
    }

    /// ⛔ THE ABSOLUTE CALIBRATION OF `from_residuals`, which `the_gate_recovers_the_variances`
    /// above cannot see: it checks a weight (a ratio) and `dominant()` (an ordering), both
    /// scale-invariant, so multiplying every measured variance by ten passed it — and so did
    /// dividing by `n` instead of `n − 1`, despite that choice having its own paragraph of
    /// justification. Two checks: the fused variance against its closed form on a large sample,
    /// and `n` against `n − 1` on the smallest sample where they differ by a factor of two.
    #[test]
    fn from_residuals_measures_the_variance_it_was_shown_in_absolute_terms() {
        let truth: [f64; 2] = [0.09, 2.25];
        let mut rng = Rng::new(17);
        let residuals: Vec<Vec<f64>> = truth
            .iter()
            .map(|s2| (0..20_000).map(|_| gaussian(&mut rng) * s2.sqrt()).collect())
            .collect();
        let gate = PrecisionGate::from_residuals(&residuals).unwrap();
        // 1/(1/0.09 + 1/2.25) = 1/11.5556 = 0.086538..., written out rather than recomputed
        // through the gate. 20,000 samples put the sample variance within ~1% of truth.
        let want = 0.086_538_461_538;
        let got = gate.fused_variance().unwrap();
        assert!((got - want).abs() / want < 0.02, "fused variance {got} against {want}");
        let tp = gate.total_precision().unwrap();
        assert!((1.0 / tp - want).abs() / want < 0.02, "1/total_precision {} against {want}", 1.0 / tp);

        // n − 1, not n. Two residuals at ±1 have sample variance exactly 2 (n − 1 = 1) and
        // population variance exactly 1 (n = 2): the two conventions differ by a factor of two
        // here, and only one of them is what the doc promises.
        let tiny = PrecisionGate::from_residuals(&[vec![-1.0, 1.0]]).unwrap();
        assert_eq!(tiny.fused_variance().unwrap(), 2.0, "sample variance with n − 1 must be exactly 2");
    }

    #[test]
    fn a_gate_refuses_what_it_cannot_price() {
        assert!(PrecisionGate::new(vec![]).is_err());
        assert!(PrecisionGate::new(vec![1.0, 0.0]).is_err());
        assert!(PrecisionGate::new(vec![1.0, -1.0]).is_err());
        assert!(PrecisionGate::new(vec![f64::NAN]).is_err());
        let mut g = PrecisionGate::new(vec![1.0, 1.0]).unwrap();
        assert!(!g.is_empty() && g.len() == 2);
        assert!(g.set_available(2, false).is_err());
        assert!(g.degradation_from_dropping(9).is_err());
        assert!(g.fuse(&[1.0]).is_err());
        assert!(g.fuse(&[1.0, f64::NAN]).is_err());
        assert!(PrecisionGate::from_residuals(&[vec![1.0]]).is_err());
        assert!(PrecisionGate::from_residuals(&[vec![1.0, 1.0, 1.0]]).is_err());
        assert!(PrecisionGate::from_residuals(&[]).is_err());
    }

    // ------------------------------------------------------------------------------------------
    // Cross-modal association learning
    // ------------------------------------------------------------------------------------------

    /// Song, Miller & Abbott's window shape on a wide, additive, unclamped weight so that the
    /// learned matrix is a plain sum of window values and can be checked in closed form.
    fn assoc_rule() -> PairStdp {
        PairStdp::new(1.0, 1.05, 16.8e-3, 33.7e-3, WeightRule::Additive, Bounds::wide())
            .expect("parameters are in range")
    }

    fn aligned_from(pairs: &[(u16, u64, u32)], dt: f64) -> Aligned {
        let mut streams: Vec<Stream> = Vec::new();
        for m in 0..=pairs.iter().map(|p| p.0).max().unwrap_or(0) {
            let mut ev: Vec<Event> = pairs
                .iter()
                .filter(|p| p.0 == m)
                .map(|p| Event { t: p.1, address: p.2, polarity: Polarity::On })
                .collect();
            ev.sort_unstable();
            streams.push(Stream::new(m, Clock::new(dt).unwrap(), ev).unwrap());
        }
        Aligned::merge(&streams).unwrap()
    }

    /// The closed form the whole associator rests on: one isolated cross-modal pair moves the
    /// weight by exactly [`PairStdp::window`] at that lag, to the last bit.
    #[test]
    fn an_isolated_pair_moves_the_weight_by_exactly_the_stdp_window() {
        let rule = assoc_rule();
        for &(lag_ticks, sign) in &[(5_000u64, 1.0f64), (1u64, 1.0)] {
            let dt = 1e-6;
            let lag = lag_ticks as f64 * dt;
            let aligned = aligned_from(&[(0, 0, 0), (1, lag_ticks, 0)], dt);
            let mut a = CrossModalAssociator::new(rule, 0, 1, 1, 1, 0.25, 0.1).unwrap();
            assert_eq!(a.observe(&aligned).unwrap(), 1);
            let want = 0.25 + sign * rule.window(lag);
            let got = a.weight(0, 0).unwrap();
            assert!((got - want).abs() < 1e-15, "lag {lag}: {got} against {want}");
        }
        // And the reversed order depresses by exactly the other lobe.
        let dt = 1e-6;
        let aligned = aligned_from(&[(1, 0, 0), (0, 5_000, 0)], dt);
        let mut a = CrossModalAssociator::new(assoc_rule(), 0, 1, 1, 1, 0.25, 0.1).unwrap();
        assert_eq!(a.observe(&aligned).unwrap(), 1);
        let want = 0.25 + assoc_rule().window(-5e-3);
        assert!((a.weight(0, 0).unwrap() - want).abs() < 1e-15);
        assert!(want < 0.25, "the reversed pair did not depress");
    }

    /// The real job: recover which element of one sense goes with which element of another, when
    /// the correspondence is a permutation rather than the identity.
    ///
    /// The identity would be recovered by a broken implementation that simply returned its own row
    /// index, so the permutation is what makes the assertion mean anything.
    #[test]
    fn the_associator_recovers_a_permuted_cross_modal_correspondence() {
        let n = 5u32;
        let perm = |k: u32| (k * 3 + 1) % n; // a derangement-free permutation of 0..5
        let mut src = two_sensors(5e-3, 0.5e-3, 0.0, 0.0, 60.0);
        src.addresses = n;
        let g = generate(&src, 601);

        // Modality 1 speaks a permuted address space, as a microphone channel does against a pixel.
        let permuted: Vec<Event> = g.streams[1]
            .events()
            .iter()
            .map(|e| Event { address: perm(e.address), ..*e })
            .collect();
        let b = Stream::new(1, g.streams[1].clock, permuted).unwrap();
        let aligned = Aligned::merge(&[g.streams[0].clone(), b]).unwrap();

        let mut a = CrossModalAssociator::new(
            assoc_rule(),
            0,
            n as usize,
            1,
            n as usize,
            0.0,
            20e-3,
        )
        .unwrap();
        let pairs = a.observe(&aligned).unwrap();
        assert!(pairs > 1_000, "only {pairs} cross-modal pairs; the test has no power");

        for i in 0..n as usize {
            let want = perm(i as u32) as usize;
            assert_eq!(a.best_partner(i), Some(want), "row {i} chose the wrong partner");
            let w = a.weight(i, want).unwrap();
            let runner_up = (0..n as usize)
                .filter(|&j| j != want)
                .map(|j| a.weight(i, j).unwrap())
                .fold(f64::MIN, f64::max);
            assert!(w > 0.0, "the correspondence was not potentiated: {w}");
            assert!(w > 3.0 * runner_up.abs(), "row {i}: {w} against a runner-up of {runner_up}");
        }
    }

    /// The sign is not decoration. Swapping which modality is presynaptic turns every correct
    /// correspondence from the row's maximum into its minimum, because the same physical lag is
    /// now on the depressing side of the window.
    #[test]
    fn swapping_the_modality_roles_depresses_where_it_potentiated() {
        let mut src = two_sensors(5e-3, 0.5e-3, 0.0, 0.0, 40.0);
        src.addresses = 3;
        let g = generate(&src, 602);
        let aligned = Aligned::merge(&g.streams).unwrap();

        let mut fwd = CrossModalAssociator::new(assoc_rule(), 0, 3, 1, 3, 0.0, 20e-3).unwrap();
        fwd.observe(&aligned).unwrap();
        let mut rev = CrossModalAssociator::new(assoc_rule(), 1, 3, 0, 3, 0.0, 20e-3).unwrap();
        rev.observe(&aligned).unwrap();

        for i in 0..3usize {
            assert_eq!(fwd.best_partner(i), Some(i));
            let w_rev = rev.weight(i, i).unwrap();
            assert!(w_rev < 0.0, "the reversed association potentiated: {w_rev}");
            let row_min = (0..3usize).map(|j| rev.weight(i, j).unwrap()).fold(f64::MAX, f64::min);
            assert!((w_rev - row_min).abs() < 1e-12, "row {i}: {w_rev} was not the minimum");
            assert_ne!(rev.best_partner(i), Some(i), "arg-max found a depressed cell");
        }
    }

    /// An untrained matrix has no answer, and saying "address zero" would look exactly like one.
    #[test]
    fn a_tied_row_refuses_to_name_a_partner() {
        let a = CrossModalAssociator::new(assoc_rule(), 0, 3, 1, 4, 0.5, 1e-3).unwrap();
        assert_eq!(a.best_partner(0), None);
        assert_eq!(a.best_partner(9), None);
        assert_eq!(a.weight(0, 0), Some(0.5));
        assert_eq!(a.weight(3, 0), None);
        assert_eq!(a.weights().len(), 12);
        assert!(CrossModalAssociator::new(assoc_rule(), 0, 0, 1, 2, 0.0, 1e-3).is_err());
        assert!(CrossModalAssociator::new(assoc_rule(), 0, 2, 1, 2, 0.0, 0.0).is_err());
        assert!(CrossModalAssociator::new(assoc_rule(), 0, 2, 1, 2, f64::NAN, 1e-3).is_err());
    }

    // ------------------------------------------------------------------------------------------
    // nearest_cross_pair, the primitive both tasks are built on
    // ------------------------------------------------------------------------------------------

    #[test]
    fn the_nearest_cross_pair_keeps_its_sign_and_respects_its_window() {
        // b after a is positive, b before a is negative.
        assert_eq!(nearest_cross_pair(&[1.0], &[1.25], 1.0), Some(0.25));
        assert_eq!(nearest_cross_pair(&[1.0], &[0.75], 1.0), Some(-0.25));
        // Outside the window there is no pair, and that is not the same as a pair at zero.
        assert_eq!(nearest_cross_pair(&[1.0], &[3.0], 1.0), None);
        assert_eq!(nearest_cross_pair(&[], &[1.0], 1.0), None);
        assert_eq!(nearest_cross_pair(&[1.0], &[], 1.0), None);
        // The window is inclusive at its edge.
        assert_eq!(nearest_cross_pair(&[1.0], &[2.0], 1.0), Some(1.0));
        // The closest wins, not the first — this is the one a naive sweep gets wrong. Compared
        // with a tolerance because 4.9 - 5.0 is not -0.1 in binary floating point.
        let d = nearest_cross_pair(&[0.0, 5.0], &[4.9, 9.0], 1.0).expect("a pair inside 1 s");
        assert!((d + 0.1).abs() < 1e-12, "{d}");
        // It searches both sides of the insertion point.
        assert_eq!(nearest_cross_pair(&[5.0], &[4.0, 5.5], 1.0), Some(0.5));
    }

    // ------------------------------------------------------------------------------------------
    // (d) Early against late fusion, on a task where they MUST differ
    // ------------------------------------------------------------------------------------------

    /// ⭐ Verification (d). On [`SynchronyTask`] the label is `|t_a - t_b|`. Early fusion sees it;
    /// late fusion cannot express it, because a weighted sum of per-branch scores is linear in the
    /// two streams and an absolute difference is not.
    ///
    /// Both paths use the **same** classifier on the **same** trials and differ only in which
    /// features reach them, so the gap is architectural rather than a difference in fitting.
    #[test]
    fn a_nonlinear_cross_modal_label_is_invisible_to_late_fusion() {
        let task = SynchronyTask::default();
        let train = task.generate(&mut Rng::new(70)).unwrap();
        let test = task.generate(&mut Rng::new(71)).unwrap();

        let early = EarlyFusion::fit(&train).unwrap();
        let late = LateFusion::fit(&train).unwrap();
        let ea = early.accuracy(&test, &[]).unwrap();
        let la = late.accuracy(&test, &[]).unwrap();

        assert!(ea > 0.85, "early fusion only reached {ea:.3}");
        assert!(la < 0.60, "late fusion reached {la:.3}; the task leaks into a single stream");
        // The cross-modal block is what carries the label, so early fusion must lose it too when
        // it is masked away. Without this, "early fusion is better" could be about the extra
        // features rather than about the joint ones.
        let blind = EarlyFusion::fit(&train).unwrap().accuracy(&test, &[1]).unwrap();
        assert!(blind < 0.62, "early fusion kept {blind:.3} with the joint features masked");
        assert!(ea - la > 0.30, "the gap was only {:.3}", ea - la);

        // The branches' own reliabilities say the same thing from the other side: neither stream
        // carries the label alone, even in sample.
        for (b, r) in late.reliability().iter().enumerate() {
            assert!(*r < 0.62, "branch {b} reached {r:.3} in sample on a label it cannot see");
        }

        // ⛔ AND IT IS AT CHANCE BY VOTING, NOT BY DEGENERATING. A combiner whose weights had all
        // been clamped to zero would emit one class for every trial and score 0.5 too, which would
        // make the assertion above unfalsifiable. Both labels must appear.
        let emitted: Vec<u32> = test.iter().map(|t| late.predict(t, &[]).unwrap()).collect();
        let ones = emitted.iter().filter(|&&c| c == 1).count();
        assert!(
            ones > test.len() / 5 && ones < 4 * test.len() / 5,
            "late fusion emitted class 1 on {ones} of {} trials; it is not voting",
            test.len()
        );
    }

    /// The price of early fusion, measured. The architecture that wins on the temporal task loses
    /// everything when one of its two streams goes away, because its joint features stop existing;
    /// the late path, which was at chance anyway, does not move.
    #[test]
    fn early_fusion_collapses_on_a_dropout_and_late_fusion_does_not() {
        let task = SynchronyTask::default();
        let train = task.generate(&mut Rng::new(72)).unwrap();
        let test = task.generate(&mut Rng::new(73)).unwrap();
        let early = EarlyFusion::fit(&train).unwrap();
        let late = LateFusion::fit(&train).unwrap();

        let e_full = early.accuracy(&test, &[]).unwrap();
        let e_drop = early.accuracy(&test, &[1]).unwrap();
        let l_full = late.accuracy(&test, &[]).unwrap();
        let l_drop = late.accuracy(&test, &[1]).unwrap();

        assert!(e_full - e_drop > 0.30, "early fusion lost only {:.3}", e_full - e_drop);
        assert!(e_drop < 0.62, "early fusion still reached {e_drop:.3} without its partner");
        assert!((l_full - l_drop).abs() < 0.15, "late fusion moved by {:.3}", l_full - l_drop);

        // Losing everything is a refusal, not a guess.
        assert!(matches!(early.predict(&test[0], &[0, 1]), Err(FusionError::AllFeaturesDropped)));
        assert!(matches!(late.predict(&test[0], &[0, 1]), Err(FusionError::AllFeaturesDropped)));
        assert!(late.predict(&test[0], &[9]).is_err());
    }

    /// ⭐ Verification (c), empirically. On a task where each sense carries the label on its own, a
    /// lost sense **degrades** late fusion: measurably below the intact figure, and measurably
    /// above chance. Both bounds are asserted, which is what makes it a quantity rather than a
    /// reassurance.
    ///
    /// It is also the control for (d): the large early/late gap of the temporal task is a statement
    /// about where the information is, and here, where it is in the marginals, the gap closes.
    #[test]
    fn a_dropped_sense_degrades_redundant_fusion_without_destroying_it() {
        let task = RedundantRateTask::default();
        let train = task.generate(&mut Rng::new(80)).unwrap();
        let test = task.generate(&mut Rng::new(81)).unwrap();
        let early = EarlyFusion::fit(&train).unwrap();
        let late = LateFusion::fit(&train).unwrap();

        let l_full = late.accuracy(&test, &[]).unwrap();
        let l_drop = late.accuracy(&test, &[1]).unwrap();
        let e_full = early.accuracy(&test, &[]).unwrap();

        assert!(l_full > 0.80, "two senses only reached {l_full:.3}");
        assert!(l_drop > 0.68, "one sense fell to {l_drop:.3}; that is destruction, not degradation");
        // ⛔ REGRESSION GUARD. This is the assertion that caught the late combiner discarding its
        // branches' confidence: with per-trial normalisation, two senses bought -0.011.
        assert!(l_full - l_drop > 0.02, "the second sense bought only {:.3}", l_full - l_drop);
        assert!(e_full > 0.80, "early fusion only reached {e_full:.3}");

        // The control: the architectures no longer disagree the way they did on the temporal task.
        let temporal = SynchronyTask::default();
        let t_train = temporal.generate(&mut Rng::new(82)).unwrap();
        let t_test = temporal.generate(&mut Rng::new(83)).unwrap();
        let t_gap = EarlyFusion::fit(&t_train).unwrap().accuracy(&t_test, &[]).unwrap()
            - LateFusion::fit(&t_train).unwrap().accuracy(&t_test, &[]).unwrap();
        let r_gap = (e_full - l_full).abs();
        assert!(
            r_gap < 0.4 * t_gap,
            "redundant gap {r_gap:.3} against a temporal gap of {t_gap:.3}"
        );
    }

    // ------------------------------------------------------------------------------------------
    // The classifier itself, and the refusals on both fusion paths
    // ------------------------------------------------------------------------------------------

    /// Standardisation is load-bearing, not hygiene. With a count feature at O(10) and an interval
    /// at O(0.01) s, an unstandardised Euclidean distance is a count classifier: the informative
    /// feature here is the small one, and only scaling makes it visible.
    #[test]
    fn standardisation_is_what_lets_a_small_feature_decide() {
        let mut rng = Rng::new(95);
        let mut x = Vec::new();
        let mut y = Vec::new();
        for _ in 0..400 {
            let lab = u32::from(rng.next_u32() & 1 == 1);
            let big = 10.0 + gaussian(&mut rng) * 3.0; // no label information, huge scale
            let small = if lab == 1 { 0.01 } else { -0.01 } + gaussian(&mut rng) * 0.002;
            x.push(vec![big, small]);
            y.push(lab);
        }
        let clf = NearestCentroid::fit(&x, &y).unwrap();
        let mask = [true, true];
        let hits = x
            .iter()
            .zip(&y)
            .filter(|&(row, &lab)| clf.predict(row, &mask).unwrap() == lab)
            .count();
        assert!(hits as f64 / 400.0 > 0.95, "only {hits}/400 with the small feature scaled");
        // And masking the informative feature away must destroy it, or the first number proved
        // nothing about which feature was used.
        let blind = [true, false];
        let blind_hits = x
            .iter()
            .zip(&y)
            .filter(|&(row, &lab)| clf.predict(row, &blind).unwrap() == lab)
            .count();
        assert!(blind_hits as f64 / 400.0 < 0.60, "the big feature alone got {blind_hits}/400");
        assert_eq!(clf.classes(), &[0, 1]);
    }

    /// A feature that never varies must be ignored rather than divide by zero.
    #[test]
    fn a_constant_feature_neither_divides_by_zero_nor_decides_anything() {
        let x = vec![
            vec![7.0, 0.0],
            vec![7.0, 1.0],
            vec![7.0, 0.1],
            vec![7.0, 0.9],
        ];
        let y = vec![0u32, 1, 0, 1];
        let clf = NearestCentroid::fit(&x, &y).unwrap();
        for (row, &lab) in x.iter().zip(&y) {
            assert_eq!(clf.predict(row, &[true, true]).unwrap(), lab);
        }
        // The constant column alone separates nothing, so every score is identical and the tie
        // resolves to the lowest label.
        let s = clf.score(&x[1], &[true, false]).unwrap();
        assert!((s[0] - s[1]).abs() < 1e-15, "{s:?}");
        assert_eq!(clf.predict(&x[1], &[true, false]).unwrap(), 0);
    }

    #[test]
    fn the_classifiers_refuse_what_they_cannot_fit() {
        assert!(NearestCentroid::fit(&[], &[]).is_err());
        assert!(NearestCentroid::fit(&[vec![1.0]], &[]).is_err());
        assert!(NearestCentroid::fit(&[vec![]], &[0]).is_err());
        assert!(NearestCentroid::fit(&[vec![1.0], vec![2.0]], &[0, 0]).is_err());
        assert!(NearestCentroid::fit(&[vec![1.0], vec![1.0, 2.0]], &[0, 1]).is_err());
        assert!(NearestCentroid::fit(&[vec![f64::NAN], vec![2.0]], &[0, 1]).is_err());

        let clf = NearestCentroid::fit(&[vec![0.0], vec![1.0]], &[0, 1]).unwrap();
        assert!(clf.predict(&[0.0, 0.0], &[true]).is_err());
        assert!(clf.predict(&[0.0], &[true, true]).is_err());
        assert!(clf.predict(&[0.0], &[false]).is_err());
        assert!(clf.predict(&[f64::INFINITY], &[true]).is_err());

        let one = TrialFeatures { per_modality: vec![vec![1.0]], cross_modal: vec![], label: 0 };
        let two = TrialFeatures { per_modality: vec![vec![2.0], vec![3.0]], cross_modal: vec![], label: 1 };
        assert!(EarlyFusion::fit(&[]).is_err());
        assert!(matches!(
            EarlyFusion::fit(&[one.clone(), two.clone()]),
            Err(FusionError::RaggedFeatures { index: 1 })
        ));
        let nomod = TrialFeatures { per_modality: vec![], cross_modal: vec![1.0], label: 0 };
        let nomod2 = TrialFeatures { per_modality: vec![], cross_modal: vec![2.0], label: 1 };
        assert!(matches!(
            LateFusion::fit(&[nomod, nomod2]),
            Err(FusionError::NoModalities)
        ));
    }

    /// The early-fusion mask is the architecture's structural cost written as code: a dropped
    /// stream takes every joint feature with it, not just its own block.
    #[test]
    fn dropping_a_stream_invalidates_every_joint_feature() {
        let trials: Vec<TrialFeatures> = (0..8)
            .map(|k| TrialFeatures {
                per_modality: vec![vec![k as f64], vec![(k % 3) as f64]],
                cross_modal: vec![(k % 2) as f64],
                label: u32::from(k % 2 == 0),
            })
            .collect();
        let early = EarlyFusion::fit(&trials).unwrap();
        assert_eq!(early.mask_for(&[]), vec![true, true, true]);
        assert_eq!(early.mask_for(&[0]), vec![false, true, false]);
        assert_eq!(early.mask_for(&[1]), vec![true, false, false]);
        assert_eq!(early.mask_for(&[0, 1]), vec![false, false, false]);
    }

    // ------------------------------------------------------------------------------------------
    // Mutation-audit additions: each of these was written because deleting the thing it checks
    // left every other test in this module green.
    // ------------------------------------------------------------------------------------------

    /// `to_common_s` divides by `1 + drift`, and nothing else in the suite exercises that divisor:
    /// the drift ESTIMATOR works in reported frames and the alignment tests all use `drift = 0`.
    /// Deleting the division survived every other test here.
    #[test]
    fn a_drifting_clock_is_rescaled_and_not_merely_shifted() {
        let c = Clock::with_offset(1e-3, 0.25, 2e-3).unwrap();
        // Reported time at tick 10_000 is 10 s; common time is (10 - 0.25) / 1.002.
        let want = (10.0 - 0.25) / 1.002;
        assert!((c.to_common_s(10_000) - want).abs() < 1e-12, "{}", c.to_common_s(10_000));
        // The error a missing divisor makes is 19.5 ms at this tick — far above any window here.
        assert!((c.to_common_s(10_000) - (10.0 - 0.25)).abs() > 1e-2);
        // A drifting clock is NOT an affine shift: two ticks an hour apart disagree by more than
        // any constant offset could explain.
        let a = c.to_common_s(0) - (0.0 - 0.25);
        let b = c.to_common_s(3_600_000) - (3600.0 - 0.25);
        assert!((a - b).abs() > 1.0, "the map was affine after all: {a} vs {b}");
        // And the zero-drift path is the exact identity it claims to be.
        let plain = Clock::new(1e-3).unwrap();
        assert_eq!(plain.to_common_s(123_456_789), 123_456_789.0 * 1e-3);
        assert_eq!(plain.to_common_s(7), plain.reported_s(7));
    }

    /// The generator's tick placement, checked against arithmetic rather than against a downstream
    /// estimator. Rounding versus truncation shifts every event by half a tick in BOTH streams and
    /// therefore cancels in every offset estimate — it survives the whole sweep.
    #[test]
    fn a_source_event_lands_on_the_tick_the_arithmetic_names() {
        let mut src = two_sensors(0.0, 0.0, 0.0, 0.0, 3.0);
        src.modalities[0].dt_s = 1e-3;
        src.modalities[0].clock_offset_s = 0.0;
        src.modalities[1].dt_s = 1e-3;
        src.modalities[1].clock_offset_s = 0.0;
        let g = generate(&src, 606);
        assert_eq!(g.streams[0].len(), g.source_times_s.len());
        for (k, &t) in g.source_times_s.iter().enumerate() {
            let e = g.streams[0].events()[k];
            assert_eq!(e.t, (t / 1e-3).round() as u64, "event {k} at {t}");
            assert_eq!(e.address, g.source_addresses[k]);
        }
        // Truncation instead of rounding would move at least one of these; it cannot move none.
        let trunc_differs = g
            .source_times_s
            .iter()
            .any(|&t| (t / 1e-3).round() as u64 != (t / 1e-3).floor() as u64);
        assert!(trunc_differs, "the fixture cannot distinguish rounding from truncation");
    }

    /// The Poisson draw itself. Every closed form in this module measures the partner rate from the
    /// data, so a generator drawing uniform or Bernoulli arrivals at the right MEAN would pass all
    /// of them. The coefficient of variation is what separates exponential gaps from uniform ones.
    #[test]
    fn the_arrival_process_is_poisson_and_not_merely_the_right_rate() {
        let mut rng = Rng::new(3_141);
        let (rate, t_end) = (200.0, 400.0);
        let t = poisson_times(&mut rng, rate, 0.0, t_end, 1_000_000).unwrap();
        let n = t.len() as f64;
        // Count: mean and variance both rate*T, so the standard error is sqrt(rate*T).
        let want = rate * t_end;
        assert!((n - want).abs() < 4.0 * want.sqrt(), "{n} arrivals against {want}");
        // Gaps: exponential has CV exactly 1; a uniform-spaced process has 0, a regular one 0.
        let gaps: Vec<f64> = t.windows(2).map(|w| w[1] - w[0]).collect();
        let m = gaps.iter().sum::<f64>() / gaps.len() as f64;
        let var = gaps.iter().map(|g| (g - m) * (g - m)).sum::<f64>() / (gaps.len() as f64 - 1.0);
        let cv = var.sqrt() / m;
        assert!((cv - 1.0).abs() < 0.06, "coefficient of variation {cv}");
        assert!((m - 1.0 / rate).abs() < 4.0 * m / n.sqrt(), "mean gap {m}");
        // A zero rate is silence, not a division by zero, and a backwards interval is empty.
        assert!(poisson_times(&mut rng, 0.0, 0.0, 10.0, 10).unwrap().is_empty());
        assert!(poisson_times(&mut rng, 10.0, 5.0, 1.0, 10).unwrap().is_empty());
        assert!(poisson_times(&mut rng, 1e6, 0.0, 10.0, 8).is_err());
        assert!(poisson_times(&mut rng, -1.0, 0.0, 1.0, 8).is_err());
    }

    /// `gaussian` on its own terms, because everything downstream of it only ever compares a
    /// variance against another variance and would survive a generator with the wrong scale.
    #[test]
    fn the_normal_draw_has_the_moments_it_claims() {
        let mut rng = Rng::new(2_718);
        let n = 400_000;
        let (mut s1, mut s2, mut s4) = (0.0f64, 0.0f64, 0.0f64);
        for _ in 0..n {
            let x = gaussian(&mut rng);
            s1 += x;
            s2 += x * x;
            s4 += x * x * x * x;
        }
        let nf = f64::from(n);
        let mean = s1 / nf;
        let var = s2 / nf - mean * mean;
        assert!(mean.abs() < 0.01, "mean {mean}");
        assert!((var - 1.0).abs() < 0.015, "variance {var}");
        // Kurtosis 3 is what separates a normal from a uniform (1.8) or a two-point sign (1.0).
        let kurt = s4 / nf;
        assert!((kurt - 3.0).abs() < 0.12, "kurtosis {kurt}");
    }

    /// The interior restriction, pinned on an interval short enough for the edges to matter. On the
    /// 200 s runs above they are a third of a percent of the anchors and deleting them changes
    /// nothing any assertion can see.
    #[test]
    fn anchors_whose_window_would_run_off_the_end_are_excluded() {
        let dt = 1e-3;
        // Anchors at 0.05, 0.30, 0.50, 0.70 and 0.95 s; the interval is [0, 1] and w is 0.2 s, so
        // exactly three of the five are interior.
        let aligned = aligned_from(
            &[(0, 50, 0), (0, 300, 0), (0, 500, 0), (0, 700, 0), (0, 950, 0), (1, 500, 0)],
            dt,
        );
        let det = CoincidenceDetector::new(0.2, 0, vec![1]).unwrap();
        let rep = det.detect(&aligned, 0.0, 1.0).unwrap();
        assert_eq!(rep.anchors_considered, 3, "the edge anchors were not excluded");
        // The single partner at 0.5 s is inside the windows of the anchors at 0.3, 0.5 and 0.7.
        assert_eq!(rep.bound, 3);
        assert_eq!(rep.partner_events, 3);
        assert_eq!(rep.binding_rate(), Some(1.0));
        assert_eq!(rep.mean_partners(), Some(1.0));
        // An empty sweep has no rate rather than a zero one.
        let none = CoincidenceReport { anchors_considered: 0, bound: 0, partner_events: 0 };
        assert_eq!(none.binding_rate(), None);
        assert_eq!(none.mean_partners(), None);
    }

    /// The window's endpoints are inclusive, which continuous random data can never test: the
    /// probability of a partner landing exactly on the boundary is zero.
    #[test]
    fn the_coincidence_window_includes_both_of_its_endpoints() {
        let dt = 1e-3;
        // Anchor at 0.5 s; partners at exactly 0.49, 0.51 (on the edge of a 10 ms window) and at
        // 0.489 and 0.511 (just outside it).
        let aligned = aligned_from(
            &[(0, 500, 0), (1, 489, 0), (1, 490, 0), (1, 510, 0), (1, 511, 0)],
            dt,
        );
        let rep = CoincidenceDetector::new(10e-3, 0, vec![1])
            .unwrap()
            .detect(&aligned, 0.0, 2.0)
            .unwrap();
        assert_eq!(rep.anchors_considered, 1);
        assert_eq!(rep.partner_events, 2, "the endpoints were not both inclusive");
    }

    /// The gate's dominance and fusion under a dropout, both of which the analytic tests reach only
    /// through `weights` and would survive a `fuse` that summed over unavailable channels.
    #[test]
    fn a_dropped_channel_is_absent_from_the_fused_estimate_entirely() {
        let mut g = PrecisionGate::new(vec![1.0, 4.0]).unwrap();
        // With both present the answer is the precision-weighted mean, 0.8*10 + 0.2*0.
        let both = g.fuse(&[10.0, 0.0]).unwrap().unwrap();
        assert!((both - 8.0).abs() < 1e-15, "{both}");
        g.set_available(0, false).unwrap();
        // With channel 0 gone the answer is channel 1's value EXACTLY, whatever channel 0 says.
        let one = g.fuse(&[10.0, 3.0]).unwrap().unwrap();
        assert_eq!(one, 3.0, "a dropped channel still contributed");
        assert_eq!(g.dominant(), Some(1));
        // Ties resolve to the lowest index, deterministically.
        let tied = PrecisionGate::new(vec![2.0, 2.0, 2.0]).unwrap();
        assert_eq!(tied.dominant(), Some(0));
    }

    /// The associator's window and its address bounds, neither of which the recovery test can see:
    /// a wider window admits more clutter without changing the arg-max, and a wider address space
    /// simply never appears in the fixtures.
    #[test]
    fn the_associator_ignores_pairs_outside_its_window_and_addresses_outside_its_matrix() {
        let dt = 1e-6;
        // One pair 30 ms apart, against a 10 ms window.
        let far = aligned_from(&[(0, 0, 0), (1, 30_000, 0)], dt);
        let mut a = CrossModalAssociator::new(assoc_rule(), 0, 1, 1, 1, 0.5, 10e-3).unwrap();
        assert_eq!(a.observe(&far).unwrap(), 0, "a pair outside the window was used");
        assert_eq!(a.weight(0, 0), Some(0.5), "an unused pair moved the weight");

        // An address past the declared matrix is skipped rather than folded into cell zero.
        let wide = aligned_from(&[(0, 0, 5), (1, 1_000, 5), (0, 2_000, 0), (1, 3_000, 0)], dt);
        let mut b = CrossModalAssociator::new(assoc_rule(), 0, 1, 1, 1, 0.5, 10e-3).unwrap();
        assert_eq!(b.observe(&wide).unwrap(), 1, "an out-of-range address was folded in");
        assert!(b.weight(0, 0).unwrap() > 0.5);
    }

    /// The reliability weight is a measured quantity, and every task above gives its two branches
    /// the SAME reliability — so replacing the measurement with a constant survives all of them.
    #[test]
    fn a_branch_that_cannot_see_the_label_reports_that_it_cannot() {
        let mut rng = Rng::new(555);
        let trials: Vec<TrialFeatures> = (0..600)
            .map(|_| {
                let label = u32::from(rng.next_u32() & 1 == 1);
                let signal = f64::from(label) * 3.0 + gaussian(&mut rng);
                let noise = gaussian(&mut rng);
                TrialFeatures {
                    per_modality: vec![vec![signal], vec![noise]],
                    cross_modal: vec![0.0],
                    label,
                }
            })
            .collect();
        let late = LateFusion::fit(&trials).unwrap();
        let r = late.reliability();
        assert!(r[0] > 0.85, "the informative branch reported {:.3}", r[0]);
        assert!(r[1] < 0.60, "the noise branch reported {:.3}", r[1]);
        assert!(r[0] > r[1] + 0.25, "{r:?}");
        // ⛔ REGRESSION GUARD for the vote weight. With the branch's ACCURACY as its weight the
        // chance branch voted at 0.5 against the informative branch's 0.93 and cost 5.5 accuracy
        // points here; with the log-odds weight it votes at 0.08 against 2.6 and costs almost
        // nothing. The bound is what forced the change.
        let fused = late.accuracy(&trials, &[]).unwrap();
        let alone = late.accuracy(&trials, &[1]).unwrap();
        assert!(fused > alone - 0.015, "the noise branch cost {:.3}", alone - fused);

        // The weights themselves: log-odds, so a chance branch weighs almost nothing and an
        // informative one weighs a lot, and neither is exactly zero.
        let w = late.vote_weights();
        assert!(w[0] > 1.5, "the informative branch weighed {:.3}", w[0]);
        assert!(w[1].abs() < 0.5, "the chance branch weighed {:.3}", w[1]);
        assert!(w[1] != 0.0, "a branch was silenced outright");
        // ln(p/(1-p)) against the reported accuracy, exactly.
        for (b, (&wi, &ri)) in w.iter().zip(r).enumerate() {
            assert!((wi - (ri / (1.0 - ri)).ln()).abs() < 1e-12, "branch {b}: {wi} vs {ri}");
        }
    }

    /// Centring the branch scores before scaling them, which a mutation sweep found the rest of
    /// this suite could not see: adding the same constant to every class does not move an arg-max,
    /// so the *decision* is unchanged — what changes is the per-branch **scale**, which without
    /// centring is dominated by each branch's absolute distance rather than by how far apart its
    /// classes are.
    ///
    /// The fixture is the case that separates them: a strong branch whose one informative feature
    /// is buried in fifteen noise dimensions, so its raw distances are large and its between-class
    /// difference is not, against a weak branch with one feature and small distances. Without
    /// centring the strong branch is divided by about fifteen and the weak one wins the vote.
    #[test]
    fn centring_stops_a_branchs_absolute_distance_from_silencing_it() {
        let mut rng = Rng::new(1_234);
        let mut trials = Vec::new();
        for _ in 0..800 {
            let label = u32::from(rng.next_u32() & 1 == 1);
            let mut strong = vec![f64::from(label) * 3.0 + gaussian(&mut rng)];
            for _ in 0..15 {
                strong.push(gaussian(&mut rng));
            }
            let weak = vec![f64::from(label) * 0.8 + gaussian(&mut rng)];
            trials.push(TrialFeatures {
                per_modality: vec![strong, weak],
                cross_modal: vec![0.0],
                label,
            });
        }
        let late = LateFusion::fit(&trials).unwrap();
        let r = late.reliability();
        assert!(r[0] > 0.88, "the strong branch reported {:.3}", r[0]);
        assert!(r[1] > 0.58 && r[1] < 0.78, "the weak branch reported {:.3}", r[1]);

        let fused = late.accuracy(&trials, &[]).unwrap();
        let strong_alone = late.accuracy(&trials, &[1]).unwrap();
        let weak_alone = late.accuracy(&trials, &[0]).unwrap();
        assert!(
            fused > weak_alone + 0.15,
            "the weak branch dominated the vote: fused {fused:.3} against weak {weak_alone:.3}"
        );
        assert!(
            fused > strong_alone - 0.03,
            "the strong branch was diluted: fused {fused:.3} against strong {strong_alone:.3}"
        );
    }

    /// The coarse histogram step, which the sweep's tolerance is too loose to pin: the median
    /// refinement recovers from a peak one bin out, so a histogram that is systematically wrong by
    /// a few bins still passes.
    #[test]
    fn the_coarse_peak_lands_in_the_bin_that_holds_the_answer() {
        for &lag in &[-40e-3, 0.0, 17e-3] {
            let src = two_sensors(lag, 0.5e-3, 0.0, 1.0, 40.0);
            let g = generate(&src, 88);
            let est = estimator();
            let fit = est.estimate(&g.streams[0], &g.streams[1]).unwrap();
            assert!(
                (fit.coarse_s - lag).abs() <= 1.5 * est.bin_s,
                "lag {lag}: coarse peak at {} is more than a bin away",
                fit.coarse_s
            );
            // And the coarse estimate is a BIN CENTRE, not the refined answer copied across.
            let k = (fit.coarse_s + est.search_half_s) / est.bin_s - 0.5;
            assert!((k - k.round()).abs() < 1e-9, "coarse {} is not a bin centre", fit.coarse_s);
        }
    }

    /// `rate_hz_in` is half-open, `[t0, t1)`, and every closed form above is a statement about a
    /// stated interval — so an off-by-one at the boundary quietly changes the rate the comparison
    /// is made against.
    #[test]
    fn a_measured_rate_counts_its_left_edge_and_not_its_right() {
        let aligned = aligned_from(&[(0, 0, 0), (0, 500, 0), (0, 1_000, 0)], 1e-3);
        assert_eq!(aligned.rate_hz_in(0, 0.0, 1.0), Some(2.0));
        assert_eq!(aligned.rate_hz_in(0, 0.0, 2.0), Some(1.5));
        assert_eq!(aligned.rate_hz_in(0, 0.5, 1.0), Some(2.0));
        assert_eq!(aligned.rate_hz_in(1, 0.0, 1.0), Some(0.0));
        assert_eq!(aligned.rate_hz_in(0, 1.0, 1.0), None);
        assert_eq!(aligned.rate_hz_in(0, 2.0, 1.0), None);
        assert_eq!(aligned.len(), 3);
        assert!(!aligned.is_empty());
        assert!(Aligned::merge(&[]).is_err());
    }

    #[test]
    fn a_clock_that_runs_backwards_is_refused() {
        assert!(Clock::with_offset(1e-3, 0.0, -1.0).is_err());
        assert!(Clock::with_offset(1e-3, 0.0, -1.5).is_err());
        assert!(Clock::with_offset(1e-3, 0.0, -0.5).is_ok());
        assert!(Clock::new(0.0).is_err());
        assert!(Clock::new(f64::INFINITY).is_err());
    }
}
