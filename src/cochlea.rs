//! The silicon cochlea: sound in, spikes out, with every stage checked against its closed form.
//!
//! # What the cochlea is, and why a chip copies it
//!
//! A microphone gives you one number per sample. The ear gives you a few thousand parallel
//! channels, each tuned to a narrow band, each already half-wave rectified and compressed, each
//! reporting in spikes. The conversion happens mechanically: the basilar membrane is a graded
//! resonator, stiff and narrow at the base where high frequencies peak, floppy and wide at the apex
//! where low ones do, so **position along the membrane is frequency**. That is tonotopy, and it is
//! the reason a neuromorphic audio front end is a filterbank rather than a transform: a bandpass
//! channel is a handful of multiply-accumulates per sample that runs forever, where a spectrogram
//! is a block transform that must buffer, window and wake up.
//!
//! What it buys: latency that does not wait for a frame, a representation that is already sparse
//! (most channels are quiet most of the time), and a front end whose cost scales with *how much the
//! sound changes* rather than with the sample rate. What it costs: the filterbank runs continuously
//! — every channel multiplies whether or not anything happened — so the sparsity appears only after
//! the spiking stage, and the analogue front ends that avoid even that (Lyon and Mead's original
//! silicon cochleae, and the `AER` ear chips since) pay in mismatch, drift and calibration.
//!
//! Lyon's 1982 cascade model (Lyon, *A computational model of filtering, detection and compression
//! in the cochlea*, Proc. `ICASSP` 1982, pp. 1282–1285) is the ancestor of every neuromorphic audio
//! front end, including the ones [`crate::hardware`] describes. Its modern form is `CARFAC` (Lyon,
//! *Cascades of two-pole–two-zero asymmetric resonators are good models of peripheral auditory
//! function*, `JASA` 130(6):3893–3904 (2011), doi:10.1121/1.3658470; Lyon, *Human and Machine
//! Hearing*, Cambridge, 2017). **This module did not port the cascade, the 1982 model's gain stages
//! or `CARFAC`'s coupled four-stage `AGC`.** ⛔ This paragraph used to call `CARFAC` the 1982
//! model's `AGC`: "did not port Lyon's cascade or its coupled four-stage `AGC` (the `CARFAC`
//! model)". The four-stage `AGC` is `CARFAC`'s. The 1982 paper describes "a cascade of three stages
//! of bilinear elements (simple multipliers)" that set gains after the filterbank and detector, the
//! slowest optionally moved in front of it (p. 1284). What is here is the parallel gammatone bank
//! that the same literature uses as the standard reference filterbank, plus a deliberately simpler
//! feed-forward `AGC` whose steady state is exactly computable — and the doc on [`Agc`] describes
//! both of Lyon's gain controls and says which is which.
//!
//! # The four stages, and the closed form each is checked against
//!
//! 1. **[`Gammatone`] filterbank**, centre frequencies on the `ERB` scale (Glasberg & Moore,
//!    *Derivation of auditory filter shapes from notched-noise data*, Hearing Research 47, 1990).
//!    Checked by driving each channel with a pure-tone sweep: the peak must land on the stated
//!    `f_c` and the −3 dB width must match `0.887 · ERB(f_c)`, the width of a fourth-order
//!    gammatone that follows from the published table (Holdsworth, Nimmo-Smith, Patterson & Rice,
//!    *Implementing a `GammaTone` Filter Bank*, Annex C of the `SVOS` Final Report, Part A: The
//!    Auditory Filterbank, `MRC` Applied Psychology Unit, 26 Feb 1988, Table 1: `c_4 = 0.870`, so
//!    the −3 dB width is `0.870 × 1.019 = 0.887` `ERB`; the annex's worked example gives 113.59 Hz
//!    at an `ERB` of 128.14 Hz). ⛔ This line used to cite "`APU` report 2341, 1988". This review
//!    did not locate that number on either annex scan; [`PATTERSON_B`] says where it comes from.
//! 2. **Rectification and compression**, [`Compression`]. Half-wave rectification is what the inner
//!    hair cell does — it depolarises when the stereocilia bend one way and not the other — and the
//!    compression that follows is the reason a 120 dB input range fits into a 40 dB firing-rate
//!    range. Checked by an exact identity: a power law with exponent `e` maps an input range of
//!    `D` dB onto exactly `e · D` dB.
//! 3. **[`Meddis`] inner-hair-cell transduction** (Meddis, *Simulation of mechanical to neural
//!    transduction in the auditory receptor*, `JASA` 79(3), 1986). A three-pool transmitter model
//!    whose output is a firing probability. Checked against the steady state its own equations
//!    imply, `c∞ = k·y·M / (y(l + r) + k·l)`, derived in [`Meddis::steady_state_cleft`].
//! 4. **Spiking output**, one [`Lif`] per channel, so the front end's product is a
//!    [`Train`] the rest of this crate already speaks.
//!
//! Plus **onset and offset channels**, because a neuromorphic front end reports change. These are a
//! fast-minus-slow leaky difference ([`Onset`]) whose step response `e^{-t/τ_s} − e^{-t/τ_f}` has a
//! peak time and height in closed form.
//!
//! ## One design choice worth stating, because getting it wrong is invisible
//!
//! The onset path sees the **envelope** of each channel, not the rectified carrier. Running a 1 ms
//! differencer on a half-wave-rectified 4 kHz carrier produces a detector that fires all the way
//! through a steady tone — an "onset" channel that reports no onset. The sustained path sees the
//! carrier, because that is where phase locking lives. The two paths therefore carry different
//! units, and [`Filterbank::onset_drive`] is documented separately from [`Filterbank::drive`] for
//! exactly that reason.
//!
//! # Units
//!
//! `SI` at every interface: seconds, hertz, amperes, volts. Sample rates and centre frequencies are
//! hertz; time constants are seconds; the current handed to a neuron is amperes.
//!
//! **The exception is deliberate and confined.** Meddis's constants — `A = 5`, `B = 300`,
//! `g = 2000`, `y = 5.05`, `l = 2500`, `x = 66.31`, `r = 6580`, `h = 50000` — are printed here
//! exactly as the 1986 paper prints them, in its own units of reciprocal seconds and arbitrary
//! transmitter quanta, so a reader can compare them to the source line by line. They are converted
//! at the boundary: [`Meddis::step`] takes `dt` in seconds and returns a rate in spikes per second.
//! Rescaling them into anything else would make them unrecognisable against the paper.
//!
//! # Quickstart
//!
//! ```
//! use ferromorphic::cochlea::{Compression, Filterbank, Transduction, tone};
//!
//! let fs = 48_000.0;
//! let x = tone(fs, 1_000.0, 1.0, 0.2)?;
//! let mut bank = Filterbank::erb_bank(
//!     fs,
//!     200.0,
//!     8_000.0,
//!     24,
//!     Transduction::HalfWave(Compression::Power { exponent: 0.4 }),
//! )?;
//! let train = bank.spike_train(&x)?;
//! let gram = bank.cochleagram(&train, x.len() as u64);
//!
//! // The place code: the busiest channel is the one tuned nearest the tone.
//! let best = gram.best_channel().expect("a 1 kHz tone drives some channel");
//! assert!((gram.f_c[best] - 1_000.0).abs() < 100.0);
//! # Ok::<(), ferromorphic::cochlea::CochleaError>(())
//! ```
//!
//! # What this module does not claim
//!
//! A gammatone bank is **linear and level-independent**. Real cochlear tuning sharpens at low level
//! and broadens at high level, and two-tone suppression — a tone reducing the response to another
//! tone that is not in its passband — does not fall out of a linear filter at all. This
//! implementation did not locate those effects in what is here, and did not port the compressive
//! dual-resonance models that do produce them (Lopez-Poveda & Meddis, `JASA` 110, 2001). The
//! masking this module demonstrates is *energetic*: a loud masker wins the place code. That is a
//! real effect and it is not the whole of masking.

use crate::neuron::{Lif, Neuron};
use crate::rng::Rng;
use crate::spike::{Spike, Train};

use core::f64::consts::PI;

/// Everything that can be refused, with the offending value attached.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CochleaError {
    /// A quantity was `NaN` or infinite. Rejected at the boundary because a non-finite sample
    /// poisons a recursive filter's state permanently — the channel is dead from that sample on and
    /// nothing downstream reports it.
    NotFinite {
        /// Which quantity, e.g. `"sample rate"` or `"signal sample"`.
        what: &'static str,
        /// Index within the offending slice, or `0` for a scalar.
        index: usize,
    },
    /// A quantity that must be strictly positive was zero or negative.
    NotPositive {
        /// Which quantity, e.g. `"duration"` or `"tau"`.
        what: &'static str,
        /// The value supplied.
        value: f64,
    },
    /// A centre frequency fell outside the band this module will filter at the given sample rate.
    ///
    /// The upper limit is [`NYQUIST_GUARD`] times the sample rate, not half of it: the
    /// negative-frequency image that the complex gammatone's closed form leaves out aliases back
    /// toward `f_c` as `f_c` approaches Nyquist, and the guarantee that the response peaks exactly
    /// at `f_c` stops holding before the sampling theorem does.
    CentreFrequency {
        /// The centre frequency asked for, hertz.
        f_c: f64,
        /// The sample rate, hertz.
        fs: f64,
        /// The highest centre frequency allowed at that sample rate, hertz.
        limit: f64,
    },
    /// A frequency band was empty or inverted: `lo` must be strictly below `hi`.
    Band {
        /// Lower edge as supplied, hertz.
        lo: f64,
        /// Upper edge as supplied, hertz.
        hi: f64,
    },
    /// A filterbank was asked for a number of channels it cannot build.
    Channels {
        /// The count requested.
        n: usize,
    },
    /// A gammatone order outside `1..=`[`MAX_ORDER`].
    ///
    /// The upper bound is a policy, not physiology and not arithmetic. This doc used to say the
    /// passband gain `(1 − r)^order` underflows at high order; measured at the narrowest realistic
    /// case, order 8 at 100 Hz and 48 kHz, the gain is `4.95e-19` against an `f64` floor of
    /// `2.2e-308`, so nothing is within 289 orders of magnitude of underflowing. What the bound
    /// actually buys: the per-sample cost is `order` complex multiplies per channel, and the
    /// closed forms this module ships are driven and measured at orders 2 to 6 and constructed at
    /// 8. Above that nothing here has been checked.
    Order {
        /// The order requested.
        order: usize,
    },
    /// A compression exponent outside `(0, 1]`.
    ///
    /// Above 1 is expansion, not compression, and is refused because the name would then be a lie;
    /// at or below 0 the map is not monotone on `[0, ∞)` and the place code stops meaning anything.
    Exponent {
        /// The exponent requested.
        exponent: f64,
    },
    /// An onset detector's fast constant was not strictly faster than its slow one.
    ///
    /// With `tau_fast >= tau_slow` the difference is zero or inverted, so the detector reports an
    /// offset at every onset — a sign error that looks like a working detector wired backwards.
    OnsetTaus {
        /// Fast time constant as supplied, seconds.
        fast: f64,
        /// Slow time constant as supplied, seconds.
        slow: f64,
    },
    /// A time step too coarse for the forward-Euler integrator it was handed to.
    ///
    /// [`Meddis`] has a transmitter-return rate of 6580 s⁻¹ in its published parameter set, and
    /// forward Euler past the stability bound does not degrade gracefully: it oscillates and then
    /// diverges, producing a firing rate that looks like a burst.
    TimeStep {
        /// The step supplied, seconds.
        dt: f64,
        /// The largest step the integrator is accepted at, seconds.
        bound: f64,
    },
    /// A window into a signal ran past the signal's own duration.
    Window {
        /// Where the window starts, seconds.
        onset: f64,
        /// How long it lasts, seconds.
        duration: f64,
        /// The total signal duration, seconds.
        total: f64,
    },
    /// A slice did not match the length the callee required.
    LengthMismatch {
        /// Length the callee required.
        expected: usize,
        /// Length it received.
        got: usize,
    },
}

impl core::fmt::Display for CochleaError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotFinite { what, index } => {
                write!(f, "{what} at index {index} was not a finite number")
            }
            Self::NotPositive { what, value } => {
                write!(f, "{what} must be strictly positive, was {value}")
            }
            Self::CentreFrequency { f_c, fs, limit } => write!(
                f,
                "centre frequency {f_c} Hz is outside (0, {limit}] Hz at a sample rate of {fs} Hz"
            ),
            Self::Band { lo, hi } => write!(f, "band [{lo}, {hi}] Hz is empty or inverted"),
            Self::Channels { n } => write!(f, "a filterbank of {n} channels cannot be built"),
            Self::Order { order } => {
                write!(f, "gammatone order {order} is outside 1..={MAX_ORDER}")
            }
            Self::Exponent { exponent } => {
                write!(f, "compression exponent {exponent} is outside (0, 1]")
            }
            Self::OnsetTaus { fast, slow } => write!(
                f,
                "onset needs tau_fast < tau_slow, got {fast} s and {slow} s"
            ),
            Self::TimeStep { dt, bound } => write!(
                f,
                "time step {dt} s exceeds the integrator's stability bound of {bound} s"
            ),
            Self::Window {
                onset,
                duration,
                total,
            } => write!(
                f,
                "a window starting at {onset} s for {duration} s does not fit in {total} s"
            ),
            Self::LengthMismatch { expected, got } => {
                write!(f, "expected {expected} values, got {got}")
            }
        }
    }
}

/// So `?` works in a caller whose error type is `Box<dyn Error>`, as every example here uses.
impl std::error::Error for CochleaError {}

/// The highest gammatone order this module will build. See [`CochleaError::Order`].
pub const MAX_ORDER: usize = 8;

/// The highest centre frequency, as a fraction of the sample rate, that a [`Gammatone`] accepts.
///
/// `0.40`, not `0.5`. See [`CochleaError::CentreFrequency`] for why the guard is below Nyquist.
///
/// ⛔ **Measured, not argued.** This constant was `0.45`, justified as "conservative by argument".
/// Driving the filter and locating its peak by ternary search: at `0.40·fs` the peak sits
/// `0.0005 ERB` below `f_c`, at `0.44·fs` `0.0014 ERB`, at `0.445·fs` `0.03 ERB`, and at the old
/// guard itself, `0.45·fs`, **`0.0375 ERB` — 1.9× the tolerance the peak test holds every other
/// channel to** (`Gammatone::new(21600, 48000)` was accepted and peaked 88 Hz low). The
/// negative-frequency image, which the complex output's closed form leaves out, stops being
/// negligible between 0.44 and 0.445; 0.40 is the value with margin, and the peak test now runs at
/// the guard itself.
pub const NYQUIST_GUARD: f64 = 0.40;

/// Patterson's bandwidth factor: the gammatone's exponential decay rate is `2π · b · ERB(f_c)`.
///
/// `1.019` is the value that makes a **fourth-order** gammatone's equivalent rectangular bandwidth
/// equal `ERB(f_c)`. It is order-specific — the factor for a third- or fifth-order filter is
/// different — and [`Gammatone::with_shape`] lets a caller supply their own rather than silently
/// reusing this one at an order it was not derived for. Source: Holdsworth, Nimmo-Smith, Patterson
/// & Rice, *Implementing a `GammaTone` Filter Bank*, Annex C of the `SVOS` Final Report, Part A:
/// The Auditory Filterbank (`MRC` Applied Psychology Unit, 26 Feb 1988), Table 1, row `n = 4`:
/// `a_4 = 0.982` and `1/a_4 = 1.019`, where `a_n` is the order-`n` gammatone's `ERB` in units of
/// `b`. In closed form `a_4 = 5π/16 = 0.98175`, so `1/a_4 = 1.0186`, which the table rounds to
/// 1.019; `patterson_b_is_row_four_of_annex_c_table_1` recomputes the row.
///
/// ⛔ This doc used to credit the value to Patterson et al., *An efficient auditory filterbank
/// based on the gammatone function*, "`APU` report 2341, 1988". That title is Annex B of the same
/// report (Patterson, Nimmo-Smith, Holdsworth & Rice, December 1987), and Annex B states only the
/// rounded figure: "for order 4, the gammatone `ERB` is 1.02b" (p. 7). 1.019 is printed in Annex C.
/// The number 2341 comes from later citations of the report (for example the `AMT` 0.9.8
/// gammatone documentation: "`APU` report, 2341, 1987"); this review did not locate it on either
/// annex scan.
pub const PATTERSON_B: f64 = 1.019;

/// Default amperes per unit of half-wave-rectified, compressed basilar-membrane displacement.
///
/// **This is an interface parameter, not a physiological constant.** It exists because the
/// transduction stage produces a dimensionless number and [`Lif`] wants amperes, and somebody has
/// to choose the scale. `2e-8` puts a unit-amplitude tone at a channel's own centre frequency
/// comfortably above [`Lif`]'s default threshold without saturating it at the refractory bound.
pub const DEFAULT_DRIVE_HALF_WAVE: f64 = 2e-8;

/// Default amperes per spike-per-second of [`Meddis`] output. An interface parameter, as
/// [`DEFAULT_DRIVE_HALF_WAVE`] is; the units differ because the transduction stages do.
pub const DEFAULT_DRIVE_MEDDIS: f64 = 1e-10;

/// Default amperes per unit of compressed envelope on the onset and offset paths.
///
/// Larger than [`DEFAULT_DRIVE_HALF_WAVE`] because the onset signal is a *difference* of two leaky
/// integrators and so is smaller than the signal it is computed from — at the default time
/// constants a step of height `h` produces a peak of about `0.81 · h`, and the mean of a
/// rectified carrier is well below its peak.
pub const DEFAULT_DRIVE_ONSET: f64 = 6e-8;

fn positive(what: &'static str, v: f64) -> Result<f64, CochleaError> {
    if !v.is_finite() {
        return Err(CochleaError::NotFinite { what, index: 0 });
    }
    if !(v > 0.0) {
        return Err(CochleaError::NotPositive { what, value: v });
    }
    Ok(v)
}

fn finite(what: &'static str, v: f64) -> Result<f64, CochleaError> {
    if v.is_finite() {
        Ok(v)
    } else {
        Err(CochleaError::NotFinite { what, index: 0 })
    }
}

fn finite_slice(what: &'static str, xs: &[f64]) -> Result<(), CochleaError> {
    for (i, &x) in xs.iter().enumerate() {
        if !x.is_finite() {
            return Err(CochleaError::NotFinite { what, index: i });
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// The ERB scale
// ---------------------------------------------------------------------------------------------

/// Equivalent rectangular bandwidth of the auditory filter centred at `f` hertz, in hertz.
///
/// `ERB(f) = 24.7 · (4.37 · f/1000 + 1)`, which is Glasberg & Moore (Hearing Research 47, 1990)
/// equation 3, fitted to notched-noise masking data from 100 Hz to 10 kHz. Outside that range it is
/// an extrapolation and this module says so rather than pretending otherwise: at `f = 0` it returns
/// 24.7 Hz, which is a fitted intercept and not a measurement.
///
/// The `ERB` is *not* the filter's −3 dB width. It is the width of the rectangular filter that
/// passes the same power for flat input, which for a fourth-order gammatone is about 13 % wider
/// than the −3 dB width — see [`Gammatone::bandwidth_3db_continuous`].
///
/// Returns `NaN` for a non-finite argument rather than refusing, because it is an algebraic
/// identity used inside inner loops; every constructor in this module checks its inputs before
/// calling it.
#[must_use]
pub fn erb_hz(f: f64) -> f64 {
    24.7 * (4.37 * f / 1000.0 + 1.0)
}

/// Position of `f` hertz on the `ERB`-rate scale, in `ERB` numbers.
///
/// `E(f) = 21.4 · log10(4.37 · f/1000 + 1)`, Glasberg & Moore (1990) equation 4. It is the integral
/// of `1/ERB(f)` up to a constant, so one unit of `E` is one auditory filter width, and equal steps
/// in `E` are what "`ERB`-spaced" means. Roughly 39 `ERB`s fit below 15 kHz.
///
/// `None` for `f <= -1000/4.37 ≈ -228.8` Hz, where the logarithm's argument is not positive, and
/// for a non-finite argument. Negative frequencies are accepted above that bound because the scale
/// is defined there and refusing would make the inverse's round-trip test impossible to write.
#[must_use]
pub fn erb_rate(f: f64) -> Option<f64> {
    if !f.is_finite() {
        return None;
    }
    let arg = 4.37 * f / 1000.0 + 1.0;
    if !(arg > 0.0) {
        return None;
    }
    Some(21.4 * arg.log10())
}

/// Inverse of [`erb_rate`]: the frequency in hertz at `ERB`-number `e`.
///
/// `f = 1000 · (10^(e/21.4) − 1) / 4.37`. `None` for a non-finite argument or one large enough to
/// overflow the power of ten.
#[must_use]
pub fn erb_rate_to_hz(e: f64) -> Option<f64> {
    if !e.is_finite() {
        return None;
    }
    let f = 1000.0 * (10f64.powf(e / 21.4) - 1.0) / 4.37;
    if f.is_finite() { Some(f) } else { None }
}

/// `n` centre frequencies from `lo` to `hi` hertz, equally spaced on the `ERB`-rate scale.
///
/// The endpoints are exact: `out[0] == lo` and `out[n-1] == hi` to within the round trip through
/// [`erb_rate`] and back. For `n == 1` the single channel sits at `lo`.
///
/// This is what a cochlea's tonotopy looks like: channels crowd together at low frequency, where
/// the filters are narrow, and spread out at high frequency, where they are wide. A linear spacing
/// would put twenty channels inside one auditory filter at 200 Hz and leave gaps at 8 kHz.
///
/// # Errors
///
/// [`CochleaError::NotFinite`] or [`CochleaError::NotPositive`] for a bad edge,
/// [`CochleaError::Band`] for `lo >= hi`, [`CochleaError::Channels`] for `n == 0`.
pub fn erb_space(lo: f64, hi: f64, n: usize) -> Result<Vec<f64>, CochleaError> {
    let lo = positive("band lower edge", lo)?;
    let hi = positive("band upper edge", hi)?;
    if !(lo < hi) {
        return Err(CochleaError::Band { lo, hi });
    }
    if n == 0 {
        return Err(CochleaError::Channels { n });
    }
    // Both edges are strictly positive and finite, so `erb_rate` cannot refuse them; the
    // `ok_or` arms are unreachable and are written as refusals rather than unwraps so that a
    // future change to `erb_rate`'s domain surfaces as an error instead of a panic.
    let e_lo = erb_rate(lo).ok_or(CochleaError::Band { lo, hi })?;
    let e_hi = erb_rate(hi).ok_or(CochleaError::Band { lo, hi })?;
    if n == 1 {
        return Ok(vec![lo]);
    }
    let step = (e_hi - e_lo) / (n - 1) as f64;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let e = e_lo + step * i as f64;
        out.push(erb_rate_to_hz(e).ok_or(CochleaError::Band { lo, hi })?);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------------------------
// The gammatone filter
// ---------------------------------------------------------------------------------------------

/// A gammatone bandpass channel: the standard model of one point on the basilar membrane.
///
/// # The filter
///
/// The gammatone's impulse response is a gamma envelope times a tone,
///
/// ```text
/// g(t) = t^(n-1) · e^(-2π·b·ERB(f_c)·t) · cos(2π·f_c·t)
/// ```
///
/// fitted by de Boer (1975) to reverse-correlation measurements of cat auditory nerve fibres and
/// made the standard filterbank by Patterson and colleagues (the `SVOS` Final Report, Part A: The
/// Auditory Filterbank, `MRC` Applied Psychology Unit, 1987/88, whose Annexes B and C are cited at
/// [`PATTERSON_B`]). `n` is the order, almost always 4.
///
/// # How it is implemented, and why that matters for the tests
///
/// This is the **complex (analytic) gammatone**: a cascade of `n` identical one-pole
/// complex-coefficient sections, `y[k] = x[k] + p·y[k-1]` with `p = e^{(-β + i·2π·f_c)/fs}` and
/// `β = 2π·b·ERB(f_c)` (Darling, *Properties and implementation of the gammatone filter: a
/// tutorial*, `UCL` Speech Hearing and Language: Work in Progress 5:43–61 (1991); Lyon 1996, cited
/// below, names it the "complex" (or "analytic") gammatone filter on p. 16 and credits the
/// construction to Darling on p. 17). Its complex output is all-pole, with no pole at the
/// conjugate of `p`, so its transfer function leaves out the negative-frequency image, which at
/// these centre frequencies is five orders of magnitude down. [`Gammatone::step`] returns the real
/// part. That reintroduces zeros on the real axis and makes the output a gammatone again:
/// exactly in continuous time ("Taking the real part of the output introduces spurious zeros on
/// the real axis in the Laplace domain, and converts it back exactly to the GTF", Lyon 1996,
/// p. 16), and approximately here, where the impulse response carries the binomial
/// `C(m+n−1, n−1)` in place of the sampled `t^(n-1)`.
///
/// ⛔ This paragraph used to call the filter "the **all-pole gammatone** (Lyon, *The all-pole
/// gammatone filter and auditory models*, Forum Acusticum, 1996)". That paper's all-pole gammatone
/// (`APGF`) is a different, real-valued filter that keeps both poles of each pair: "the order-N
/// `APGF` is the Nth power of a filter with a complex-conjugate pair of poles", implemented as "a
/// cascade of N identical two-pole filter stages" (abstract of the draft of 5 Feb. 1996,
/// `dicklyon.com/tech/Hearing/APGF_Lyon_1996.pdf`). This filter keeps one pole of each pair and
/// takes the real part at the end, which is Darling's construction (Lyon 1996, p. 17: "a
/// complex-coefficient IIR filter to place N poles directly at the complex location corresponding
/// to CF ... then converted back to a gamma-tone by taking the real part"). No code changed.
///
/// The payoff is that the complex output's transfer function is exact and short:
/// `|H(f)|² = |1 − p·e^{-i·2π·f/fs}|^{-2n}`, whose minimum denominator is at `f = f_c`
/// **exactly**, for any sample rate. So [`Gammatone::peak_frequency`] is not an approximation, and
/// below [`NYQUIST_GUARD`]`·fs` a measured peak that misses `f_c` is a real defect rather than a
/// discretisation artefact — above it the dropped image is no longer negligible and the *driven*
/// peak drifts even though the denominator's minimum does not, which is why the guard sits where
/// it does. The −3 dB width follows in closed form too, in [`Gammatone::bandwidth_3db`].
///
/// # The two outputs
///
/// [`Gammatone::step`] returns the real part — the membrane displacement, which carries the
/// carrier and therefore the phase locking. [`Gammatone::envelope`] returns the magnitude of the
/// same complex state, which is the analytic envelope for free, with no Hilbert transform and no
/// extra state. The envelope is what the onset path uses; see the module doc for why.
///
/// # Gain
///
/// Normalised so that a unit-amplitude tone **at `f_c`** produces a unit-amplitude output. That
/// makes [`Gammatone::magnitude_response`] a ratio against the channel's own peak, which is the
/// quantity a filter shape is usually plotted in.
#[derive(Debug, Clone, PartialEq)]
pub struct Gammatone {
    /// Centre frequency, hertz. The response peaks here exactly; see the struct doc.
    f_c: f64,
    /// Sample rate, hertz.
    fs: f64,
    /// Number of cascaded one-pole sections, `1..=`[`MAX_ORDER`].
    order: usize,
    /// Patterson's bandwidth factor; `β = 2π · b_factor · ERB(f_c)` in reciprocal seconds.
    b_factor: f64,
    /// Pole magnitude `r = e^{-β/fs}`, dimensionless, in `(0, 1)`.
    r: f64,
    /// Real part of the pole `p = r·e^{i·2π·f_c/fs}`.
    p_re: f64,
    /// Imaginary part of the pole.
    p_im: f64,
    /// Input scale that normalises the peak gain to unity.
    gain: f64,
    /// Real parts of the `order` cascade states.
    state_re: Vec<f64>,
    /// Imaginary parts of the `order` cascade states.
    state_im: Vec<f64>,
}

impl Gammatone {
    /// A fourth-order channel at `f_c` hertz, sampled at `fs` hertz, with [`PATTERSON_B`].
    ///
    /// # Errors
    ///
    /// [`CochleaError::NotFinite`] or [`CochleaError::NotPositive`] for a bad sample rate, and
    /// [`CochleaError::CentreFrequency`] for one outside `(0, `[`NYQUIST_GUARD`]` · fs]`.
    pub fn new(f_c: f64, fs: f64) -> Result<Self, CochleaError> {
        Self::with_shape(f_c, fs, 4, PATTERSON_B)
    }

    /// A channel with an explicit order and bandwidth factor.
    ///
    /// [`PATTERSON_B`] was derived for order 4; supplying a different order without a matching
    /// factor gives a filter whose equivalent rectangular bandwidth is no longer `ERB(f_c)`, which
    /// is why the factor is a parameter here rather than a constant reached for silently.
    ///
    /// # Errors
    ///
    /// As [`Gammatone::new`], plus [`CochleaError::Order`] for an order outside
    /// `1..=`[`MAX_ORDER`] and [`CochleaError::NotPositive`] for a non-positive `b_factor`.
    pub fn with_shape(
        f_c: f64,
        fs: f64,
        order: usize,
        b_factor: f64,
    ) -> Result<Self, CochleaError> {
        let fs = positive("sample rate", fs)?;
        let f_c = finite("centre frequency", f_c)?;
        let b_factor = positive("bandwidth factor", b_factor)?;
        let limit = NYQUIST_GUARD * fs;
        if !(f_c > 0.0) || f_c > limit {
            return Err(CochleaError::CentreFrequency { f_c, fs, limit });
        }
        if order == 0 || order > MAX_ORDER {
            return Err(CochleaError::Order { order });
        }
        let beta = 2.0 * PI * b_factor * erb_hz(f_c);
        let r = (-beta / fs).exp();
        let theta = 2.0 * PI * f_c / fs;
        // A unit-amplitude REAL tone at f_c splits into two conjugate half-amplitude exponentials;
        // only the positive one is passed, so the factor of 2 restores unit amplitude at the peak.
        let gain = 2.0 * (1.0 - r).powi(order as i32);
        Ok(Self {
            f_c,
            fs,
            order,
            b_factor,
            r,
            p_re: r * theta.cos(),
            p_im: r * theta.sin(),
            gain,
            state_re: vec![0.0; order],
            state_im: vec![0.0; order],
        })
    }

    /// Centre frequency, hertz.
    #[must_use]
    pub fn f_c(&self) -> f64 {
        self.f_c
    }

    /// Sample rate, hertz.
    #[must_use]
    pub fn fs(&self) -> f64 {
        self.fs
    }

    /// Cascade order.
    #[must_use]
    pub fn order(&self) -> usize {
        self.order
    }

    /// Bandwidth factor as supplied; `β = 2π · b_factor · ERB(f_c)`.
    #[must_use]
    pub fn b_factor(&self) -> f64 {
        self.b_factor
    }

    /// The exponential decay rate `β` of the impulse response envelope, reciprocal seconds.
    #[must_use]
    pub fn decay_rate(&self) -> f64 {
        2.0 * PI * self.b_factor * erb_hz(self.f_c)
    }

    /// The frequency at which this channel's response is largest, hertz.
    ///
    /// Exactly `f_c`, and that is a theorem about this filter rather than a restatement of the
    /// field: `|1 − p·e^{-iωT}|² = 1 − 2r·cos((ω_c − ω)T) + r²` is minimised where the cosine is 1.
    /// The method exists so a test can assert against something computed independently of the
    /// constructor's arithmetic.
    #[must_use]
    pub fn peak_frequency(&self) -> f64 {
        self.f_c
    }

    /// `|H(f)| / |H(f_c)|`, dimensionless, from the closed-form transfer function.
    ///
    /// `1.0` at `f_c` by construction. Drops as `[1 + ((f − f_c)/b)²]^{-n/2}` in the continuous
    /// limit, so the skirts fall at `6n` dB per octave of detuning — 24 dB for a fourth-order
    /// filter, which is the number a filterbank plot should show.
    ///
    /// Returns `NaN` for a non-finite argument; it is an algebraic identity, not an interface.
    #[must_use]
    pub fn magnitude_response(&self, f: f64) -> f64 {
        let d = 2.0 * PI * (self.f_c - f) / self.fs;
        let denom = 1.0 - 2.0 * self.r * d.cos() + self.r * self.r;
        let peak = (1.0 - self.r) * (1.0 - self.r);
        (peak / denom).powf(self.order as f64 / 2.0)
    }

    /// The −3 dB bandwidth of this discrete filter, hertz, in closed form.
    ///
    /// Solving `|H(f)|/|H(f_c)| = 1/√2` gives
    /// `Δf = (fs/π) · arccos( (1 + r² − (1−r)²·2^{1/n}) / (2r) )` for the full width.
    ///
    /// `None` when the arccos argument leaves `[−1, 1]`, which happens for a filter so wide that it
    /// never falls 3 dB anywhere inside the sampled band. That is a refusal rather than a clamp: a
    /// clamped answer would report a bandwidth of exactly the sample rate and look plausible.
    #[must_use]
    pub fn bandwidth_3db(&self) -> Option<f64> {
        let r = self.r;
        let two_pow = 2f64.powf(1.0 / self.order as f64);
        let arg = (1.0 + r * r - (1.0 - r) * (1.0 - r) * two_pow) / (2.0 * r);
        if !(-1.0..=1.0).contains(&arg) {
            return None;
        }
        Some(arg.acos() * self.fs / PI)
    }

    /// The −3 dB bandwidth of the *continuous* gammatone this filter discretises, hertz.
    ///
    /// `2 · b_factor · ERB(f_c) · √(2^{1/n} − 1)`. For `n = 4` and [`PATTERSON_B`] the numeric
    /// factor is `0.8865`, the fourth-order gammatone's −3 dB width in `ERB` from the published
    /// table (Holdsworth et al., Annex C of the `SVOS` Final Report Part A, 1988, Table 1:
    /// `c_4 = 0.870` times `1/a_4 = 1.019`, which is `0.88653`, the 0.887 `ERB` the width test
    /// uses). The annex prints the two factors, not their product. ⛔ This line used to quote
    /// "−3 dB width is 0.887 `ERB`" as published and credit it to "Patterson et al., 1988", which
    /// is Annex B; this review did not locate the product there, only the rounded "1.02b".
    /// The discrete and continuous values agree to a few parts in ten thousand at audio sample
    /// rates, and the pair is kept separate so a test can check one against the other rather than
    /// against itself.
    #[must_use]
    pub fn bandwidth_3db_continuous(&self) -> f64 {
        let factor = (2f64.powf(1.0 / self.order as f64) - 1.0).sqrt();
        2.0 * self.b_factor * erb_hz(self.f_c) * factor
    }

    /// Latency of the impulse-response envelope peak, seconds — the travelling-wave delay.
    ///
    /// Exact for the discrete cascade. Its impulse response is `C(m+n−1, n−1)·r^m`, and
    /// `a[m]/a[m−1] = ((m + n − 1)/m)·r ≥ 1` exactly while `m ≤ (n − 1)·r/(1 − r)`, so the peak is
    /// at `m* = ⌊(n − 1)·r/(1 − r)⌋` and the latency is `m*/fs`. In the continuous limit
    /// `r → 1 − β/fs` this tends to `(n − 1)/β`, which is [`Gammatone::peak_latency_continuous`];
    /// the two differ by about `n − 1` samples, which is 5 % of the latency at 4 kHz and 0.4 % at
    /// 200 Hz.
    ///
    /// The point of the quantity: `β` grows with `ERB(f_c)`, so **low-frequency channels ring
    /// later**. That is the travelling wave, and a front end that ignored it would align the apex
    /// and the base of the cochlea at the same instant, which no ear does.
    ///
    /// Tie convention: when `(n − 1)·r/(1 − r)` is an exact integer `m`, samples `m − 1` and `m`
    /// hold the same value and this returns the **later** one. A measure-zero case over real
    /// centre frequencies, named because a test that takes the first argmax would disagree by one
    /// sample there.
    #[must_use]
    pub fn peak_latency(&self) -> f64 {
        let m = (self.order as f64 - 1.0) * self.r / (1.0 - self.r);
        m.max(0.0).floor() / self.fs
    }

    /// Latency of the *continuous* gammatone's envelope peak, `(n − 1)/β` seconds.
    ///
    /// Kept beside [`Gammatone::peak_latency`] for the same reason the two bandwidths are kept
    /// apart: one is the discretisation, one is the model it discretises, and agreement between
    /// them is evidence rather than tautology.
    #[must_use]
    pub fn peak_latency_continuous(&self) -> f64 {
        (self.order as f64 - 1.0) / self.decay_rate()
    }

    /// Advance one sample and return the basilar-membrane displacement, in input units.
    ///
    /// **Does not check `x`.** This is the inner loop of the whole module — order multiplies per
    /// channel per sample — and a non-finite sample is rejected once, at the boundary, by
    /// [`Gammatone::filter`], [`Gammatone::envelopes`] and [`Filterbank::spike_train`]. A caller
    /// driving `step` directly owns that check; a `NaN` here is permanent, because the state is
    /// recursive and nothing downstream will tell you which sample did it.
    pub fn step(&mut self, x: f64) -> f64 {
        let mut re = x * self.gain;
        let mut im = 0.0;
        for k in 0..self.order {
            // y[k][t] = x[k][t] + p · y[k][t-1], with the stored state being y[k][t-1].
            let yr = re + self.p_re * self.state_re[k] - self.p_im * self.state_im[k];
            let yi = im + self.p_re * self.state_im[k] + self.p_im * self.state_re[k];
            self.state_re[k] = yr;
            self.state_im[k] = yi;
            re = yr;
            im = yi;
        }
        re
    }

    /// Magnitude of the final cascade stage: the analytic envelope, in input units.
    ///
    /// Reads the state left by the last [`Gammatone::step`] and does not advance anything.
    #[must_use]
    pub fn envelope(&self) -> f64 {
        let k = self.order - 1;
        self.state_re[k].hypot(self.state_im[k])
    }

    /// Clear the filter state. The coefficients are unchanged.
    pub fn reset(&mut self) {
        for k in 0..self.order {
            self.state_re[k] = 0.0;
            self.state_im[k] = 0.0;
        }
    }

    /// Filter a whole buffer, returning the displacement per sample. Resets first.
    ///
    /// # Errors
    ///
    /// [`CochleaError::NotFinite`], naming the index of the first offending sample.
    pub fn filter(&mut self, signal: &[f64]) -> Result<Vec<f64>, CochleaError> {
        finite_slice("signal sample", signal)?;
        self.reset();
        Ok(signal.iter().map(|&x| self.step(x)).collect())
    }

    /// Filter a whole buffer, returning the analytic envelope per sample. Resets first.
    ///
    /// # Errors
    ///
    /// [`CochleaError::NotFinite`], naming the index of the first offending sample.
    pub fn envelopes(&mut self, signal: &[f64]) -> Result<Vec<f64>, CochleaError> {
        finite_slice("signal sample", signal)?;
        self.reset();
        Ok(signal
            .iter()
            .map(|&x| {
                self.step(x);
                self.envelope()
            })
            .collect())
    }
}

// ---------------------------------------------------------------------------------------------
// Compression
// ---------------------------------------------------------------------------------------------

/// A static, memoryless compression applied after half-wave rectification.
///
/// The ear's input range spans about 120 dB and an auditory nerve fibre's rate range spans about
/// 30. Something has to compress, and most of it happens mechanically, in the outer hair cells'
/// active feedback: basilar-membrane displacement grows roughly as the 0.2–0.5 power of pressure
/// near the characteristic frequency (Ruggero et al., `JASA` 101, 1997, measured in chinchilla).
/// That exponent is the default suggested here.
///
/// These are *static* laws, applied sample by sample with no state. Real compression is dynamic —
/// it has attack and release — and that part lives in [`Agc`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Compression {
    /// No compression. The identity, kept as a variant so a caller can turn compression off
    /// without a sentinel exponent of `1.0` that reads like a parameter choice.
    Linear,
    /// `y = x^e` for `x ≥ 0`, with `e` in `(0, 1]`.
    ///
    /// The exact property this is checked against: a power law maps an input range of `D` decibels
    /// onto exactly `e · D` decibels, because `20·log10(x^e) = e · 20·log10(x)`. So `e = 0.4` turns
    /// 100 dB of input into 40 dB of output, which is the compression the whole stage exists for.
    Power {
        /// The exponent, in `(0, 1]`. Around `0.3`–`0.5` for basilar-membrane compression.
        exponent: f64,
    },
    /// `y = ln(1 + x/knee)`, a logarithm with a linear region below `knee`.
    ///
    /// Preferred over a bare logarithm because `ln(x)` diverges at silence, and a front end whose
    /// quietest possible input is `−∞` cannot be represented in any fixed point. For `x ≪ knee`
    /// this reduces to `x/knee` exactly to first order, which is the limit its test checks.
    Log {
        /// Input value at which the law leaves its linear region, in the same units as the input.
        knee: f64,
    },
}

impl Default for Compression {
    /// `Power { exponent: 0.4 }`, in the middle of the measured basilar-membrane range. Stated here
    /// so a figure made with the default is reproducible from the documentation alone.
    fn default() -> Self {
        Self::Power { exponent: 0.4 }
    }
}

impl Compression {
    /// Reject a law whose parameters are outside the range it is defined on.
    ///
    /// # Errors
    ///
    /// [`CochleaError::Exponent`] for an exponent outside `(0, 1]`,
    /// [`CochleaError::NotPositive`] for a non-positive knee, [`CochleaError::NotFinite`] for
    /// either being non-finite.
    pub fn validate(self) -> Result<Self, CochleaError> {
        match self {
            Self::Linear => Ok(self),
            Self::Power { exponent } => {
                if !exponent.is_finite() {
                    return Err(CochleaError::NotFinite {
                        what: "compression exponent",
                        index: 0,
                    });
                }
                if !(exponent > 0.0) || exponent > 1.0 {
                    return Err(CochleaError::Exponent { exponent });
                }
                Ok(self)
            }
            Self::Log { knee } => {
                positive("compression knee", knee)?;
                Ok(self)
            }
        }
    }

    /// Apply the law to a non-negative input. Negative inputs are rectified to zero first, so this
    /// is safe to call on a raw displacement.
    ///
    /// The **law** is not checked here — this is the inner loop. For one [`Compression::validate`]
    /// would refuse the result is meaningless: `Log { knee: -1.0 }` returns `NaN` for inputs past
    /// the knee and `Power { exponent: -1.0 }` returns `inf` at zero. A [`Filterbank`] validates at
    /// construction; a caller using `apply` directly owns the check.
    #[must_use]
    pub fn apply(self, x: f64) -> f64 {
        let x = x.max(0.0);
        match self {
            Self::Linear => x,
            Self::Power { exponent } => x.powf(exponent),
            Self::Log { knee } => (1.0 + x / knee).ln(),
        }
    }
}

/// `max(x, 0)` that passes `NaN` through instead of replacing it with zero.
fn floor_at_zero(x: f64) -> f64 {
    if x < 0.0 { 0.0 } else { x }
}

/// Half-wave rectification: `max(x, 0)`.
///
/// What the inner hair cell does. Its stereocilia open transduction channels when they bend toward
/// the tallest one and close them when they bend the other way, so the receptor potential follows
/// only one half of the membrane's motion. Losing this — by taking `|x|` instead — doubles the
/// apparent carrier frequency and destroys the phase locking that carries pitch below about 4 kHz.
#[must_use]
pub fn half_wave(x: f64) -> f64 {
    x.max(0.0)
}

// ---------------------------------------------------------------------------------------------
// Automatic gain control
// ---------------------------------------------------------------------------------------------

/// A one-pole feed-forward automatic gain control.
///
/// **This is not Lyon's `AGC`, and the difference is worth stating.** Lyon's `CARFAC` model (Lyon,
/// *Cascades of two-pole–two-zero asymmetric resonators are good models of peripheral auditory
/// function*, `JASA` 130(6):3893–3904 (2011), doi:10.1121/1.3658470; *Human and Machine Hearing*,
/// 2017) feeds a four-stage coupled `AGC` smoothing network back into the pole damping of its
/// filter cascade. The stages have time constants of 2, 8, 32 and 128 ms in the reference code
/// (`google/carfac`, `matlab/AGC_params_default.m`: `'n_stages', 4` and
/// `'time_constants', 0.002 * 4.^(0:3)`), and each spreads laterally across neighbouring channels
/// (`'AGC1_scales', 1.0 * sqrt(2).^(0:3)`, "in units of channels"), so that a loud sound at one
/// place turns the gain down at nearby places too. That coupling is most of what makes the model
/// behave like an ear. Strictly it is one loop whose smoothing filter has four coupled stages, not
/// four loops.
///
/// ⛔ This paragraph used to say that "Lyon's cochlear model closes four coupled `AGC` loops", and
/// the module doc gave that `AGC` to the 1982 model. The 1982 model has three stages, not four,
/// and they set a gain after the filterbank and detector instead of damping the filters: "a
/// cascade of three stages of bilinear elements (simple multipliers), with possibly separate
/// control signals, time constants, and degrees of coupling on each" (Lyon, `ICASSP` 1982,
/// p. 1284). The slowest could optionally be moved in front of the filterbank, "like the stapedial
/// reflex". Lyon (2011) draws the line himself: the coupled smoothing network "descends from one
/// first described by Lyon (1982); in that work, the loop filter directly controlled a
/// post-filterbank gain rather than a pole damping as it does in more recent versions." No code
/// changed; this type is neither model.
///
/// What is here is a single uncoupled loop per channel whose level detector watches the **input**
/// rather than the output. That makes it feed-forward, which costs the stability argument a real
/// `AGC` needs and buys an exactly computable steady state: with a constant input `x`, the level
/// settles on `x` and the gain on `1/(1 + target·x)`, which is [`Agc::steady_state_gain`] and is
/// what its test checks. A feedback loop's steady state is the root of a fixed-point equation and
/// is checkable too, just less sharply.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Agc {
    /// Level-detector time constant, seconds. Attack and release are the same here; a real `AGC`
    /// makes release much slower than attack, and this one does not.
    pub tau: f64,
    /// Sensitivity, in reciprocal input units. The gain is halved when the level reaches
    /// `1/target`, so `target` sets where compression begins.
    pub target: f64,
    /// Current smoothed input level, in input units.
    level: f64,
}

impl Agc {
    /// A loop with the given time constant and sensitivity, starting from a level of zero.
    ///
    /// # Errors
    ///
    /// [`CochleaError::NotFinite`] or [`CochleaError::NotPositive`] for a non-positive `tau` or
    /// `target`.
    pub fn new(tau: f64, target: f64) -> Result<Self, CochleaError> {
        let tau = positive("agc tau", tau)?;
        let target = positive("agc target", target)?;
        Ok(Self {
            tau,
            target,
            level: 0.0,
        })
    }

    /// The gain the loop is currently applying, dimensionless, in `(0, 1]`.
    #[must_use]
    pub fn gain(&self) -> f64 {
        1.0 / (1.0 + self.target * self.level)
    }

    /// Current smoothed level, in input units.
    #[must_use]
    pub fn level(&self) -> f64 {
        self.level
    }

    /// The gain this loop settles on under a constant input `x`: `1/(1 + target·x)`.
    #[must_use]
    pub fn steady_state_gain(&self, x: f64) -> f64 {
        1.0 / (1.0 + self.target * x.max(0.0))
    }

    /// Apply the current gain to `x`, then advance the level detector by `dt` seconds.
    ///
    /// The gain is applied **before** the update, so the loop cannot react to a sample within the
    /// same sample — an `AGC` that did would be an algebraic loop and would suppress the very
    /// transient it is supposed to pass.
    ///
    /// The detector integrates by exponential Euler, which for a piecewise-constant input is the
    /// exact solution, so the step response is exactly `1 − e^{-t/τ}` at every sample rate.
    pub fn step(&mut self, dt: f64, x: f64) -> f64 {
        let y = x * self.gain();
        let alpha = 1.0 - (-dt / self.tau).exp();
        self.level += (x.max(0.0) - self.level) * alpha;
        y
    }

    /// Return the level to zero.
    pub fn reset(&mut self) {
        self.level = 0.0;
    }
}

// ---------------------------------------------------------------------------------------------
// The Meddis inner hair cell
// ---------------------------------------------------------------------------------------------

/// Meddis's inner-hair-cell transduction model: displacement in, firing probability out.
///
/// # What it models
///
/// The hair cell holds a pool of transmitter `q`. Sound opens a permeability `k` that lets
/// transmitter into the synaptic cleft `c`, where it drives the auditory nerve fibre. Cleft
/// transmitter is either lost at rate `l` or recovered into a reprocessing store `w` at rate `r`,
/// from which it returns to the pool at rate `x`; the pool is also replenished from an unlimited
/// supply at rate `y` toward its maximum `M`.
///
/// ```text
/// dq/dt = y·(M − q) + x·w − k·q
/// dc/dt = k·q − l·c − r·c
/// dw/dt = r·c − x·w
/// k(s)  = g·(s + A)/(s + A + B)   for s + A > 0, else 0
/// ```
///
/// and the instantaneous firing rate is `h·c`.
///
/// **This is the mechanism behind auditory adaptation.** A tone that switches on drains the pool
/// faster than it refills, so the firing rate overshoots hugely and then decays to a much lower
/// plateau over tens of milliseconds. No amount of level increase raises the plateau past
/// [`Meddis::max_steady_rate`], because the pool cannot be refilled faster than `y·M`. That is the
/// closed-form limit this model is checked against.
///
/// Source: Meddis, *Simulation of mechanical to neural transduction in the auditory receptor*,
/// `JASA` 79(3), 1986, 702–711.
///
/// # On the constants
///
/// The default set is the one quoted for the 1986 paper: `A = 5`, `B = 300`, `g = 2000`,
/// `y = 5.05`, `l = 2500`, `x = 66.31`, `r = 6580`, `M = 1`, `h = 50000`. Meddis published a
/// revised set in 1988 (`A = 10`, `B = 3000`, `g = 1000`, the rest unchanged) with different
/// spontaneous-rate behaviour, and implementations differ over which they ship. **This
/// implementation did not verify the 1986 numbers against a scanned copy of the paper**; they are
/// transcribed from the values that circulate in the reference implementations, and a reader who
/// has the paper should check them. What *is* verified here is that these constants imply a
/// spontaneous rate of 64.77 spikes per second and a saturating steady rate of 101 spikes per
/// second, and that the integrator reproduces both.
///
/// The constants are dimensionally reciprocal seconds (rates) and arbitrary transmitter quanta,
/// in the paper's own frame; they are kept verbatim so they can be compared to the source, and the
/// conversion to `SI` happens at the boundary in [`Meddis::step`], which takes seconds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Meddis {
    /// `A`: displacement offset inside the permeability law, in stimulus units. Sets the
    /// spontaneous rate, since `k(0) = g·A/(A + B)`.
    pub a: f64,
    /// `B`: half-saturation constant of the permeability law, in stimulus units.
    pub b: f64,
    /// `g`: maximum permeability, reciprocal seconds. `k → g` as the stimulus grows.
    pub g: f64,
    /// `y`: replenishment rate of the free transmitter pool, reciprocal seconds.
    pub y: f64,
    /// `l`: loss rate of cleft transmitter, reciprocal seconds. Transmitter lost this way does not
    /// come back, which is what bounds the sustained rate.
    pub l: f64,
    /// `x`: return rate from the reprocessing store to the pool, reciprocal seconds.
    pub x: f64,
    /// `r`: uptake rate from the cleft into the reprocessing store, reciprocal seconds.
    pub r: f64,
    /// `M`: maximum free transmitter in the pool, in arbitrary quanta. `1.0` in the paper, which
    /// makes every other quantity a fraction of it.
    pub m: f64,
    /// `h`: spikes per second per unit of cleft transmitter. Converts `c` into a firing rate.
    pub h: f64,
    /// Free transmitter in the pool, quanta, in `[0, M]`.
    pub q: f64,
    /// Transmitter in the synaptic cleft, quanta, non-negative. The firing rate is `h·c`.
    pub c: f64,
    /// Transmitter in the reprocessing store, quanta, non-negative.
    pub w: f64,
}

impl Default for Meddis {
    /// The 1986 parameter set, started at its own silent steady state.
    ///
    /// Starting at the steady state rather than at `q = M, c = 0` matters: the latter produces a
    /// large artificial onset transient in the first 50 ms of every simulation, which looks exactly
    /// like the adaptation the model is famous for and is not it.
    fn default() -> Self {
        let mut m = Self {
            a: 5.0,
            b: 300.0,
            g: 2000.0,
            y: 5.05,
            l: 2500.0,
            x: 66.31,
            r: 6580.0,
            m: 1.0,
            h: 50000.0,
            q: 0.0,
            c: 0.0,
            w: 0.0,
        };
        m.reset();
        m
    }
}

impl Meddis {
    /// Permeability under a steady stimulus `s`, reciprocal seconds.
    ///
    /// `g·(s + A)/(s + A + B)` where `s + A > 0`, else zero — the paper's own clamp, which exists
    /// because a large enough inward displacement shuts the transduction channels entirely.
    #[must_use]
    pub fn permeability(&self, s: f64) -> f64 {
        let num = s + self.a;
        if num > 0.0 {
            self.g * num / (num + self.b)
        } else {
            0.0
        }
    }

    /// Cleft transmitter at equilibrium under a constant permeability `k`, quanta.
    ///
    /// Setting all three derivatives to zero and eliminating `q` and `w`:
    ///
    /// ```text
    /// c∞ = k·y·M / ( y·(l + r) + k·l )
    /// ```
    ///
    /// This is the analytic solution the integrator is checked against, and its `k → ∞` limit
    /// `y·M/l` is [`Meddis::max_steady_rate`] divided by `h`. Both are derived from the same three
    /// equations printed on this struct, so a reader can rederive them.
    #[must_use]
    pub fn steady_state_cleft(&self, k: f64) -> f64 {
        k * self.y * self.m / (self.y * (self.l + self.r) + k * self.l)
    }

    /// Free transmitter at equilibrium under a constant permeability `k`, quanta.
    ///
    /// `q∞ = y·M / (y + k·l/(l + r))`.
    #[must_use]
    pub fn steady_state_pool(&self, k: f64) -> f64 {
        self.y * self.m / (self.y + k * self.l / (self.l + self.r))
    }

    /// Steady firing rate under a constant stimulus `s`, spikes per second.
    #[must_use]
    pub fn steady_state_rate(&self, s: f64) -> f64 {
        self.h * self.steady_state_cleft(self.permeability(s))
    }

    /// Firing rate in silence, spikes per second.
    ///
    /// Non-zero, and that is the model being right rather than the implementation leaking: an
    /// auditory nerve fibre fires in a quiet room. For the 1986 constants it is 64.77 spikes per
    /// second, which is in the range reported for a high-spontaneous-rate fibre. **This
    /// implementation did not locate a single published figure to compare that digit against**; the
    /// check that exists is that the integrator agrees with the algebra.
    #[must_use]
    pub fn spontaneous_rate(&self) -> f64 {
        self.steady_state_rate(0.0)
    }

    /// The firing rate no stimulus can exceed in steady state, `h·y·M/l` spikes per second.
    ///
    /// The `k → ∞` limit of [`Meddis::steady_state_cleft`]: once every available transmitter is
    /// being released the instant it is made, the rate is set by the replenishment rate `y·M` and
    /// the loss rate `l`, not by the sound. 101 spikes per second for the 1986 constants.
    ///
    /// The *onset* rate is far higher, because the pool starts full. Saturation is a statement
    /// about the plateau only.
    ///
    /// **No stimulus actually reaches it**, because the permeability saturates at `g` before `k`
    /// reaches infinity: the largest plateau any sound can produce is
    /// [`Meddis::max_achievable_rate`], 100.08 spikes per second for the 1986 constants against
    /// this bound of 101. Two ceilings, and the lower one is the physical answer.
    #[must_use]
    pub fn max_steady_rate(&self) -> f64 {
        self.h * self.y * self.m / self.l
    }

    /// The largest plateau rate any stimulus can produce, `h·c∞(g)` spikes per second.
    ///
    /// The permeability law tops out at `g`, so this — not [`Meddis::max_steady_rate`] — is the
    /// rate a sound can actually drive the fibre to. 100.08 spikes per second for the 1986
    /// constants.
    #[must_use]
    pub fn max_achievable_rate(&self) -> f64 {
        self.h * self.steady_state_cleft(self.g)
    }

    /// The largest time step this forward-Euler integrator is accepted at, seconds.
    ///
    /// `1/(y + l + r + x + g)`: the reciprocal of a conservative upper bound on the sum of the
    /// system's rates. The binding eigenvalue is `−(l + r) = −9080 s⁻¹`, so the true forward-Euler
    /// limit is about 220 µs; this bound is 90 µs, deliberately tighter, because accuracy degrades
    /// well before stability does. It admits every audio sample rate at or above about 11.2 kHz.
    #[must_use]
    pub fn stable_dt_bound(&self) -> f64 {
        1.0 / (self.y + self.l + self.r + self.x + self.g)
    }

    /// Advance by `dt` seconds under stimulus `s`, returning the instantaneous firing rate `h·c`
    /// in spikes per second.
    ///
    /// `s` is the half-wave-rectified basilar-membrane displacement in the paper's own stimulus
    /// units; `dt` is `SI` seconds. Integrated by forward Euler, as the paper's own implementation
    /// is. The three pools are floored at zero after each update, which the paper does not say to
    /// do: forward Euler can push a pool a few parts in `10^15` negative near equilibrium, and a
    /// negative cleft would produce a negative firing rate that a downstream rectifier would hide.
    ///
    /// Does not check `dt` or `s`; [`Meddis::step_checked`] does, and every constructor in this
    /// module checks `dt` once at build time instead of once per sample per channel.
    pub fn step(&mut self, dt: f64, s: f64) -> f64 {
        let k = self.permeability(s);
        let dq = self.y * (self.m - self.q) + self.x * self.w - k * self.q;
        let dc = k * self.q - self.l * self.c - self.r * self.c;
        let dw = self.r * self.c - self.x * self.w;
        // `f64::max(NaN, 0.0)` is `0.0`: it would launder a poisoned pool into a silent, healthy
        // looking cell. This floor keeps a NaN a NaN, so it is visible downstream.
        self.q = floor_at_zero(self.q + dq * dt);
        self.c = floor_at_zero(self.c + dc * dt);
        self.w = floor_at_zero(self.w + dw * dt);
        self.h * self.c
    }

    /// Check every constant and state variable, naming the first that cannot run.
    ///
    /// Every field is public, so a caller can build `Meddis { b: -300.0, ..Meddis::default() }` —
    /// which has a *negative* steady-state pool, a sustained rate of 228 spikes/s past both
    /// documented ceilings, and until this check existed was accepted by [`Filterbank`].
    ///
    /// # Errors
    ///
    /// [`CochleaError::NotFinite`] or [`CochleaError::NotPositive`] naming the constant; the
    /// state variables `q`, `c` and `w` may be zero but not negative or non-finite.
    pub fn validate(&self) -> Result<(), CochleaError> {
        for (what, v) in [
            ("meddis a", self.a),
            ("meddis b", self.b),
            ("meddis g", self.g),
            ("meddis y", self.y),
            ("meddis l", self.l),
            ("meddis x", self.x),
            ("meddis r", self.r),
            ("meddis m", self.m),
            ("meddis h", self.h),
        ] {
            positive(what, v)?;
        }
        for (what, v) in [("meddis q", self.q), ("meddis c", self.c), ("meddis w", self.w)] {
            let v = finite(what, v)?;
            if v < 0.0 {
                positive(what, v)?;
            }
        }
        Ok(())
    }

    /// [`Meddis::step`] with its arguments checked.
    ///
    /// # Errors
    ///
    /// [`CochleaError::NotFinite`] for a non-finite `dt` or `s`, [`CochleaError::NotPositive`] for
    /// a non-positive `dt`, and [`CochleaError::TimeStep`] for one past
    /// [`Meddis::stable_dt_bound`].
    pub fn step_checked(&mut self, dt: f64, s: f64) -> Result<f64, CochleaError> {
        let dt = positive("time step", dt)?;
        let s = finite("stimulus", s)?;
        let bound = self.stable_dt_bound();
        if dt > bound {
            return Err(CochleaError::TimeStep { dt, bound });
        }
        Ok(self.step(dt, s))
    }

    /// Probability that this fibre spikes during a step of `dt` seconds: `h·c·dt`.
    ///
    /// `None` when that exceeds 1. A probability greater than one is not a probability, and the
    /// usual fix — clamping it — silently converts the model from a rate code into a "fires every
    /// tick" saturation whose apparent rate is `1/dt` and depends on the simulator, not the sound.
    #[must_use]
    pub fn spike_probability(&self, dt: f64) -> Option<f64> {
        let p = self.h * self.c * dt;
        if p.is_finite() && (0.0..=1.0).contains(&p) {
            Some(p)
        } else {
            None
        }
    }

    /// Draw a spike for this step from `rng`.
    ///
    /// `None` when [`Meddis::spike_probability`] refuses. Deterministic given the seed, as
    /// everything in this crate is.
    pub fn draw(&self, rng: &mut Rng, dt: f64) -> Option<bool> {
        let p = self.spike_probability(dt)?;
        Some(rng.next_f64() < p)
    }

    /// Return the three pools to the silent steady state, computed in closed form.
    ///
    /// Not to `q = M, c = 0, w = 0`: see the note on [`Meddis::default`].
    pub fn reset(&mut self) {
        let k = self.permeability(0.0);
        self.c = self.steady_state_cleft(k);
        self.q = self.steady_state_pool(k);
        self.w = if self.x > 0.0 {
            self.r * self.c / self.x
        } else {
            0.0
        };
    }
}

// ---------------------------------------------------------------------------------------------
// Onset and offset
// ---------------------------------------------------------------------------------------------

/// A fast-minus-slow leaky difference: one detector that reports both onsets and offsets.
///
/// # Why a front end needs one
///
/// A sustained channel tells you a sound is present. An onset channel tells you it *started*, in a
/// few milliseconds, with a spike burst that is over before a rate code has finished integrating.
/// Onset responses are everywhere in the auditory brainstem — octopus and bushy cells in the
/// cochlear nucleus are built for them — and every neuromorphic audio front end that claims to be
/// event-driven is claiming this: that a steady sound costs almost nothing to report.
///
/// # Onset and offset are one mechanism
///
/// The detector's output is `fast − slow`. Positive means the signal is rising; negative means it
/// is falling. Splitting them into two channels is a rectification, not a second algorithm, and
/// keeping it in one place is what prevents the two from drifting apart in sign convention.
///
/// # The closed form
///
/// Both stages are one-pole exponential-Euler smoothers, which for piecewise-constant input are
/// exact. For a unit step applied from rest the output is exactly
///
/// ```text
/// d(t) = e^{-t/τ_s} − e^{-t/τ_f}
/// ```
///
/// whose peak is at `t* = τ_f·τ_s·ln(τ_s/τ_f)/(τ_s − τ_f)`. That time and the height at it are
/// [`Onset::step_peak_time`] and [`Onset::step_peak_height`], and they are what the test checks —
/// against the formula, not against a recorded run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Onset {
    /// Fast smoother time constant, seconds. Must be strictly below `tau_slow`.
    pub tau_fast: f64,
    /// Slow smoother time constant, seconds. Sets how long after an onset the detector goes quiet
    /// again — about `5·tau_slow`.
    pub tau_slow: f64,
    /// Fast smoother state, in input units.
    fast: f64,
    /// Slow smoother state, in input units.
    slow: f64,
}

impl Default for Onset {
    /// 1 ms and 20 ms, giving a peak 3.15 ms after a step at 81 % of its height. Round numbers in
    /// the range brainstem onset cells work over, not a fit to any recording.
    fn default() -> Self {
        Self {
            tau_fast: 1e-3,
            tau_slow: 20e-3,
            fast: 0.0,
            slow: 0.0,
        }
    }
}

impl Onset {
    /// A detector with the given time constants, starting from zero.
    ///
    /// # Errors
    ///
    /// [`CochleaError::NotFinite`] or [`CochleaError::NotPositive`] for a bad constant, and
    /// [`CochleaError::OnsetTaus`] when the fast one is not strictly faster.
    pub fn new(tau_fast: f64, tau_slow: f64) -> Result<Self, CochleaError> {
        let tau_fast = positive("tau_fast", tau_fast)?;
        let tau_slow = positive("tau_slow", tau_slow)?;
        if !(tau_fast < tau_slow) {
            return Err(CochleaError::OnsetTaus {
                fast: tau_fast,
                slow: tau_slow,
            });
        }
        Ok(Self {
            tau_fast,
            tau_slow,
            fast: 0.0,
            slow: 0.0,
        })
    }

    /// Time of the peak of the unit-step response, seconds.
    ///
    /// `τ_f·τ_s·ln(τ_s/τ_f)/(τ_s − τ_f)`, from `d/dt (e^{-t/τ_s} − e^{-t/τ_f}) = 0`.
    #[must_use]
    pub fn step_peak_time(&self) -> f64 {
        self.tau_fast * self.tau_slow * (self.tau_slow / self.tau_fast).ln()
            / (self.tau_slow - self.tau_fast)
    }

    /// Height of the peak of the unit-step response, dimensionless, in `(0, 1)`.
    #[must_use]
    pub fn step_peak_height(&self) -> f64 {
        let t = self.step_peak_time();
        (-t / self.tau_slow).exp() - (-t / self.tau_fast).exp()
    }

    /// Advance by `dt` seconds and return `fast − slow` in input units: positive on a rise,
    /// negative on a fall, zero for anything steady.
    pub fn step(&mut self, dt: f64, x: f64) -> f64 {
        let af = 1.0 - (-dt / self.tau_fast).exp();
        let as_ = 1.0 - (-dt / self.tau_slow).exp();
        self.fast += (x - self.fast) * af;
        self.slow += (x - self.slow) * as_;
        self.fast - self.slow
    }

    /// Set both smoothers to `x`, so the detector reports nothing until the input moves.
    ///
    /// Used when a channel's resting output is not zero — [`Meddis`] has a spontaneous rate — so
    /// that a simulation does not open with an onset burst produced by its own initial conditions.
    pub fn prime(&mut self, x: f64) {
        self.fast = x;
        self.slow = x;
    }

    /// Return both smoothers to zero.
    pub fn reset(&mut self) {
        self.fast = 0.0;
        self.slow = 0.0;
    }
}

// ---------------------------------------------------------------------------------------------
// The filterbank
// ---------------------------------------------------------------------------------------------

/// How a channel's basilar-membrane displacement becomes a drive for its neuron.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Transduction {
    /// Half-wave rectification followed by a static [`Compression`]. Output is dimensionless.
    ///
    /// Lyon's own path, and the cheap one: two operations per channel per sample, no state. Silence
    /// produces exactly zero, which is what makes "no sound, no spikes" checkable as an equality
    /// rather than as a threshold.
    HalfWave(Compression),
    /// The [`Meddis`] inner-hair-cell model. Output is spikes per second.
    ///
    /// The prototype is cloned once per channel, since each hair cell has its own transmitter
    /// pools. Costs three state variables and about a dozen operations per channel per sample, and
    /// buys adaptation — and a spontaneous rate, so silence here is *not* zero.
    Meddis(Meddis),
}

/// Which of the three parallel outputs a spike came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ChannelKind {
    /// Driven by the rectified carrier: reports that a sound is present, and phase-locks to it.
    Sustained,
    /// Driven by the rising part of the envelope difference: reports that a sound started.
    Onset,
    /// Driven by the falling part: reports that a sound stopped.
    Offset,
}

/// A bank of [`Gammatone`] channels with transduction, spiking and change detection.
///
/// # Spike addresses
///
/// With `n` channels, [`Train`] sources are laid out as `0..n` sustained, `n..2n` onset,
/// `2n..3n` offset, so `source % n` is the channel and `source / n` is the kind. Use
/// [`Filterbank::source`] and [`Filterbank::decode_source`] rather than reproducing that
/// arithmetic; a front end and its decoder disagreeing about the layout produces a raster that
/// looks fine and means something else.
///
/// # Cost
///
/// Every channel filters every sample. A 32-channel bank at 48 kHz does about 1.5 million complex
/// multiply-accumulates per channel-second, whatever the input — the sparsity appears at the
/// spiking stage, not before it. [`crate::ledger`] is where that distinction gets priced.
#[derive(Debug, Clone)]
pub struct Filterbank {
    /// Sample rate, hertz.
    fs: f64,
    /// Centre frequencies, hertz, ascending.
    f_c: Vec<f64>,
    /// One gammatone per channel.
    filters: Vec<Gammatone>,
    /// One hair cell per channel; empty unless the transduction is [`Transduction::Meddis`].
    hair: Vec<Meddis>,
    /// One gain loop per channel; empty unless one was configured.
    agc: Vec<Agc>,
    /// One change detector per channel, running on the compressed envelope.
    onset: Vec<Onset>,
    /// Sustained output cells.
    sustained: Vec<Lif>,
    /// Onset output cells.
    onset_cells: Vec<Lif>,
    /// Offset output cells.
    offset_cells: Vec<Lif>,
    /// The transduction in force.
    transduction: Transduction,
    /// Compression applied to the envelope before the change detector.
    envelope_compression: Compression,
    /// Amperes per unit of transduced output.
    drive: f64,
    /// Amperes per unit of compressed envelope difference.
    onset_drive: f64,
}

impl Filterbank {
    /// A bank of `n` channels spanning `lo` to `hi` hertz, `ERB`-spaced.
    ///
    /// Uses [`Lif::default`] for all three output populations, [`Onset::default`] for the change
    /// detectors, [`Compression::default`] on the envelope path, and the drive constants
    /// [`DEFAULT_DRIVE_HALF_WAVE`] / [`DEFAULT_DRIVE_MEDDIS`] and [`DEFAULT_DRIVE_ONSET`].
    ///
    /// # Errors
    ///
    /// Everything [`erb_space`] and [`Gammatone::new`] can refuse, plus
    /// [`CochleaError::TimeStep`] when `1/fs` is past [`Meddis::stable_dt_bound`] and the
    /// transduction is [`Transduction::Meddis`], and whatever [`Compression::validate`] refuses.
    pub fn erb_bank(
        fs: f64,
        lo: f64,
        hi: f64,
        n: usize,
        transduction: Transduction,
    ) -> Result<Self, CochleaError> {
        let fs = positive("sample rate", fs)?;
        let f_c = erb_space(lo, hi, n)?;
        Self::from_centre_frequencies(fs, &f_c, transduction)
    }

    /// A bank at explicitly chosen centre frequencies, in the order given.
    ///
    /// The order is preserved rather than sorted: a caller who wants a descending bank, or two
    /// channels at the same place, is making a choice and silently reordering it would break the
    /// address layout they are about to decode.
    ///
    /// # Errors
    ///
    /// As [`Filterbank::erb_bank`], plus [`CochleaError::Channels`] for an empty slice.
    pub fn from_centre_frequencies(
        fs: f64,
        f_c: &[f64],
        transduction: Transduction,
    ) -> Result<Self, CochleaError> {
        let fs = positive("sample rate", fs)?;
        if f_c.is_empty() {
            return Err(CochleaError::Channels { n: 0 });
        }
        let mut filters = Vec::with_capacity(f_c.len());
        for &f in f_c {
            filters.push(Gammatone::new(f, fs)?);
        }
        let (hair, drive) = match transduction {
            Transduction::HalfWave(c) => {
                c.validate()?;
                (Vec::new(), DEFAULT_DRIVE_HALF_WAVE)
            }
            Transduction::Meddis(m) => {
                m.validate()?;
                let dt = 1.0 / fs;
                let bound = m.stable_dt_bound();
                if dt > bound {
                    return Err(CochleaError::TimeStep { dt, bound });
                }
                (vec![m; f_c.len()], DEFAULT_DRIVE_MEDDIS)
            }
        };
        let n = f_c.len();
        let mut bank = Self {
            fs,
            f_c: f_c.to_vec(),
            filters,
            hair,
            agc: Vec::new(),
            onset: vec![Onset::default(); n],
            sustained: vec![Lif::default(); n],
            onset_cells: vec![Lif::default(); n],
            offset_cells: vec![Lif::default(); n],
            transduction,
            envelope_compression: Compression::default(),
            drive,
            onset_drive: DEFAULT_DRIVE_ONSET,
        };
        bank.reset();
        Ok(bank)
    }

    /// Replace the amperes-per-transduced-unit scale.
    ///
    /// # Errors
    ///
    /// [`CochleaError::NotFinite`] or [`CochleaError::NotPositive`].
    pub fn with_drive(mut self, drive: f64) -> Result<Self, CochleaError> {
        self.drive = positive("drive", drive)?;
        Ok(self)
    }

    /// Replace the amperes-per-envelope-unit scale on the onset and offset paths.
    ///
    /// # Errors
    ///
    /// [`CochleaError::NotFinite`] or [`CochleaError::NotPositive`].
    pub fn with_onset_drive(mut self, drive: f64) -> Result<Self, CochleaError> {
        self.onset_drive = positive("onset drive", drive)?;
        Ok(self)
    }

    /// Replace the change detector on every channel.
    ///
    /// # Errors
    ///
    /// Whatever [`Onset::new`] refuses. `Onset::tau_fast` and `tau_slow` are public, so a detector
    /// [`Onset::new`] would have refused — the fast path slower than the slow one — can be built by
    /// assignment; the first version of this builder took it as given, and the result was a
    /// detector that labelled the tone's OFFSET as its onset (41 onset spikes in the offset window,
    /// none at the onset), which is exactly the failure [`CochleaError::OnsetTaus`] exists to stop.
    pub fn with_onset(mut self, onset: Onset) -> Result<Self, CochleaError> {
        let onset = Onset::new(onset.tau_fast, onset.tau_slow)?;
        self.onset = vec![onset; self.f_c.len()];
        self.reset();
        Ok(self)
    }

    /// Replace the compression applied to the envelope before the change detector.
    ///
    /// # Errors
    ///
    /// Whatever [`Compression::validate`] refuses.
    pub fn with_envelope_compression(mut self, c: Compression) -> Result<Self, CochleaError> {
        self.envelope_compression = c.validate()?;
        Ok(self)
    }

    /// Give every channel its own copy of `agc`, or remove the loops with `None`.
    ///
    /// # Errors
    ///
    /// Whatever [`Agc::new`] refuses. `Agc::tau` and `target` are public; a negative `tau` drives
    /// the level to `-1e18` and the gain to a negative number, which inverts the signal against a
    /// doc promising `(0, 1]`.
    pub fn with_agc(mut self, agc: Option<Agc>) -> Result<Self, CochleaError> {
        self.agc = match agc {
            Some(a) => vec![Agc::new(a.tau, a.target)?; self.f_c.len()],
            None => Vec::new(),
        };
        self.reset();
        Ok(self)
    }

    /// Replace the output cell used for all three populations.
    #[must_use]
    pub fn with_cell(mut self, cell: Lif) -> Self {
        let n = self.f_c.len();
        self.sustained = vec![cell; n];
        self.onset_cells = vec![cell; n];
        self.offset_cells = vec![cell; n];
        self.reset();
        self
    }

    /// Sample rate, hertz.
    #[must_use]
    pub fn fs(&self) -> f64 {
        self.fs
    }

    /// Simulation time step, seconds — exactly `1/fs`.
    #[must_use]
    pub fn dt(&self) -> f64 {
        1.0 / self.fs
    }

    /// Centre frequencies, hertz, in channel order.
    #[must_use]
    pub fn centre_frequencies(&self) -> &[f64] {
        &self.f_c
    }

    /// Number of frequency channels. The [`Train`] carries three times this many addresses.
    #[must_use]
    pub fn channels(&self) -> usize {
        self.f_c.len()
    }

    /// Amperes per unit of transduced output. Dimensionless units for
    /// [`Transduction::HalfWave`], spikes per second for [`Transduction::Meddis`].
    #[must_use]
    pub fn drive(&self) -> f64 {
        self.drive
    }

    /// Amperes per unit of compressed envelope difference, on the onset and offset paths. Always in
    /// envelope units, whichever transduction the sustained path uses.
    #[must_use]
    pub fn onset_drive(&self) -> f64 {
        self.onset_drive
    }

    /// The [`Train`] address for a given output, or `None` if the channel does not exist.
    #[must_use]
    pub fn source(&self, kind: ChannelKind, channel: usize) -> Option<u32> {
        let n = self.f_c.len();
        if channel >= n {
            return None;
        }
        let base = match kind {
            ChannelKind::Sustained => 0,
            ChannelKind::Onset => n,
            ChannelKind::Offset => 2 * n,
        };
        u32::try_from(base + channel).ok()
    }

    /// The output a [`Train`] address refers to, or `None` if it is outside this bank.
    #[must_use]
    pub fn decode_source(&self, source: u32) -> Option<(ChannelKind, usize)> {
        let n = self.f_c.len();
        let s = source as usize;
        if s >= 3 * n {
            return None;
        }
        let kind = match s / n {
            0 => ChannelKind::Sustained,
            1 => ChannelKind::Onset,
            _ => ChannelKind::Offset,
        };
        Some((kind, s % n))
    }

    /// Return every stage to its resting state.
    ///
    /// The change detectors are primed with their own resting input, which is the compressed
    /// envelope of a silent gammatone and is therefore zero whichever transduction is in force —
    /// so a run does not open with an onset burst manufactured by its initial conditions. The
    /// priming is written out rather than assumed because [`Onset::prime`] is what a caller running
    /// the detector on a [`Meddis`] channel's own output would need, where the resting value is the
    /// spontaneous rate rather than zero.
    pub fn reset(&mut self) {
        for f in &mut self.filters {
            f.reset();
        }
        for h in &mut self.hair {
            h.reset();
        }
        for a in &mut self.agc {
            a.reset();
        }
        // The onset path watches the compressed ENVELOPE, whose resting value is zero regardless
        // of transduction, because a silent gammatone has a zero envelope.
        let rest = self.envelope_compression.apply(0.0);
        for o in &mut self.onset {
            o.prime(rest);
        }
        for c in &mut self.sustained {
            c.reset();
        }
        for c in &mut self.onset_cells {
            c.reset();
        }
        for c in &mut self.offset_cells {
            c.reset();
        }
    }

    /// Per-channel analytic envelopes: `out[channel][sample]`. Resets first.
    ///
    /// Allocates `channels × signal.len()` floats. At 32 channels and one second of 48 kHz audio
    /// that is 12 MB, which is fine for a figure and wrong for a deployment; the streaming path is
    /// [`Filterbank::spike_train`].
    ///
    /// # Errors
    ///
    /// [`CochleaError::NotFinite`], naming the index of the first offending sample.
    pub fn envelopes(&mut self, signal: &[f64]) -> Result<Vec<Vec<f64>>, CochleaError> {
        finite_slice("signal sample", signal)?;
        self.reset();
        let mut out = vec![vec![0.0; signal.len()]; self.filters.len()];
        for (ch, f) in self.filters.iter_mut().enumerate() {
            f.reset();
            for (i, &x) in signal.iter().enumerate() {
                f.step(x);
                out[ch][i] = f.envelope();
            }
        }
        Ok(out)
    }

    /// Per-channel transduced output: `out[channel][sample]`. Resets first.
    ///
    /// Dimensionless for [`Transduction::HalfWave`], spikes per second for
    /// [`Transduction::Meddis`]. Same allocation warning as [`Filterbank::envelopes`].
    ///
    /// # Errors
    ///
    /// [`CochleaError::NotFinite`], naming the index of the first offending sample.
    pub fn transduce(&mut self, signal: &[f64]) -> Result<Vec<Vec<f64>>, CochleaError> {
        finite_slice("signal sample", signal)?;
        self.reset();
        let dt = self.dt();
        let n = self.filters.len();
        let mut out = vec![vec![0.0; signal.len()]; n];
        for (i, &x) in signal.iter().enumerate() {
            for ch in 0..n {
                let bm = self.filters[ch].step(x);
                out[ch][i] = self.transduce_one(ch, bm, dt);
            }
        }
        Ok(out)
    }

    /// Advance one channel's change detector from the filter state that channel is already
    /// holding, and return `fast − slow`.
    ///
    /// **The one place the onset path's input is chosen**, and it chooses the gammatone's analytic
    /// envelope rather than the carrier — see the module doc for why that distinction is invisible
    /// in a spike count. It is a method rather than two copies of three lines because
    /// [`Filterbank::spike_train`] and [`Filterbank::onset_signal`] must be the same front end: a
    /// test that measures the change signal through one and a deployment that spikes through the
    /// other would otherwise be checking something the deployment does not run.
    fn step_change_detector(&mut self, ch: usize, dt: f64) -> f64 {
        let e = self.envelope_compression.apply(self.filters[ch].envelope());
        self.onset[ch].step(dt, e)
    }

    /// One channel's transduction for one sample. The `AGC`, where present, acts on the rectified
    /// displacement before transduction, which is where a gain control belongs: after the
    /// nonlinearity it would compress an already-compressed signal twice.
    fn transduce_one(&mut self, ch: usize, bm: f64, dt: f64) -> f64 {
        let mut rect = half_wave(bm);
        if let Some(a) = self.agc.get_mut(ch) {
            rect = a.step(dt, rect);
        }
        match self.transduction {
            Transduction::HalfWave(c) => c.apply(rect),
            Transduction::Meddis(_) => self.hair[ch].step(dt, rect),
        }
    }

    /// Run the whole front end and return every spike it produced. Resets first.
    ///
    /// Tick `i` of the returned [`Train`] is sample `i` of `signal`, so tick times multiply by
    /// [`Filterbank::dt`] to get seconds.
    ///
    /// # Errors
    ///
    /// [`CochleaError::NotFinite`], naming the index of the first offending sample. The check runs
    /// once over the buffer rather than once per channel per sample; see [`Gammatone::step`].
    pub fn spike_train(&mut self, signal: &[f64]) -> Result<Train, CochleaError> {
        finite_slice("signal sample", signal)?;
        self.reset();
        let dt = self.dt();
        let n = self.filters.len();
        let mut train = Train::new();
        // Allocated once, not once per sample: the change signal computed in pass 2 and rectified
        // the other way in pass 3, so the two passes read ONE value rather than each deriving it.
        // The annotation is load-bearing for the mutation audit, not for the compiler's benefit
        // here: with it removed, the element type of `change` is inferred from the one place a
        // `f64` is written into it, so an edit that writes a literal instead makes `(-change[ch])`
        // an ambiguous `{float}` and the mutant does not compile — a mutation that says nothing
        // either way rather than one this suite is shown to catch.
        let mut change: Vec<f64> = vec![0.0; n];
        for (i, &x) in signal.iter().enumerate() {
            let t = i as u64;
            // Pass 1: the sustained path. Each channel's filter is stepped exactly once per
            // sample, here; the envelope the onset path needs is read back out of that same
            // state in pass 2, so nothing is buffered and nothing is filtered twice.
            for ch in 0..n {
                let bm = self.filters[ch].step(x);
                let drive = self.transduce_one(ch, bm, dt) * self.drive;
                if self.sustained[ch].step(dt, drive)
                    && let Some(source) = self.source(ChannelKind::Sustained, ch)
                {
                    train.push(Spike { t, source });
                }
            }
            // Pass 2: onsets, from the compressed envelope rather than the carrier.
            for ch in 0..n {
                let d = self.step_change_detector(ch, dt);
                change[ch] = d;
                let i_on = d.max(0.0) * self.onset_drive;
                if self.onset_cells[ch].step(dt, i_on)
                    && let Some(source) = self.source(ChannelKind::Onset, ch)
                {
                    train.push(Spike { t, source });
                }
            }
            // Pass 3: offsets, the other rectified half of the same difference.
            for ch in 0..n {
                let i_off = (-change[ch]).max(0.0) * self.onset_drive;
                if self.offset_cells[ch].step(dt, i_off)
                    && let Some(source) = self.source(ChannelKind::Offset, ch)
                {
                    train.push(Spike { t, source });
                }
            }
        }
        Ok(train)
    }

    /// Per-channel change signal `fast − slow`: `out[channel][sample]`. Resets first.
    ///
    /// Positive during a rise, negative during a fall, and **exactly zero under any steady
    /// input**, which is the property the whole onset stage depends on and the one that fails
    /// silently if the detector is wired to the carrier instead of the envelope. Exposed so that
    /// property can be measured rather than inferred from a spike count that happens to be zero.
    ///
    /// Same allocation warning as [`Filterbank::envelopes`].
    ///
    /// # Errors
    ///
    /// [`CochleaError::NotFinite`], naming the index of the first offending sample.
    pub fn onset_signal(&mut self, signal: &[f64]) -> Result<Vec<Vec<f64>>, CochleaError> {
        finite_slice("signal sample", signal)?;
        self.reset();
        let dt = self.dt();
        let n = self.filters.len();
        let mut out = vec![vec![0.0; signal.len()]; n];
        for (i, &x) in signal.iter().enumerate() {
            for ch in 0..n {
                self.filters[ch].step(x);
                out[ch][i] = self.step_change_detector(ch, dt);
            }
        }
        Ok(out)
    }

    /// Summarise a [`Train`] this bank produced as per-channel firing rates.
    ///
    /// `ticks` is the number of samples the train covers, which is `signal.len()`. Addresses
    /// outside this bank are ignored rather than counted into channel zero.
    ///
    /// A run of zero ticks reports every rate as `0.0` rather than dividing by its own zero
    /// duration. A `NaN` rate is worse than a wrong one: it compares false against every bound a
    /// caller might put it under, so a summary built out of one reads as "nothing exceeded the
    /// threshold" rather than as a refusal.
    #[must_use]
    pub fn cochleagram(&self, train: &Train, ticks: u64) -> Cochleagram {
        let n = self.f_c.len();
        let seconds = ticks as f64 * self.dt();
        let mut sustained = vec![0u64; n];
        let mut onset = vec![0u64; n];
        let mut offset = vec![0u64; n];
        for s in train.spikes() {
            if let Some((kind, ch)) = self.decode_source(s.source) {
                match kind {
                    ChannelKind::Sustained => sustained[ch] += 1,
                    ChannelKind::Onset => onset[ch] += 1,
                    ChannelKind::Offset => offset[ch] += 1,
                }
            }
        }
        let to_hz = |c: &[u64]| -> Vec<f64> {
            if seconds > 0.0 {
                c.iter().map(|&k| k as f64 / seconds).collect()
            } else {
                vec![0.0; c.len()]
            }
        };
        Cochleagram {
            f_c: self.f_c.clone(),
            sustained_hz: to_hz(&sustained),
            onset_hz: to_hz(&onset),
            offset_hz: to_hz(&offset),
            seconds,
        }
    }
}

/// Per-channel firing rates over a whole run: the place code, summarised.
#[derive(Debug, Clone, PartialEq)]
pub struct Cochleagram {
    /// Centre frequencies, hertz, in channel order.
    pub f_c: Vec<f64>,
    /// Sustained-channel rates, spikes per second.
    pub sustained_hz: Vec<f64>,
    /// Onset-channel rates, spikes per second, averaged over the *whole* run — an onset burst
    /// divided by a long recording is a small number, and that is the honest average rather than a
    /// peak.
    pub onset_hz: Vec<f64>,
    /// Offset-channel rates, spikes per second, averaged over the whole run.
    pub offset_hz: Vec<f64>,
    /// Duration the rates are averaged over, seconds.
    pub seconds: f64,
}

impl Cochleagram {
    /// Index of the busiest sustained channel.
    ///
    /// `None` when every channel is silent — which is the answer, not `0`. A silent cochlea has no
    /// best frequency, and returning the first channel would put a phantom peak at the apex of
    /// every quiet recording.
    #[must_use]
    pub fn best_channel(&self) -> Option<usize> {
        let mut best: Option<(usize, f64)> = None;
        for (i, &r) in self.sustained_hz.iter().enumerate() {
            if r > 0.0 && best.is_none_or(|(_, b)| r > b) {
                best = Some((i, r));
            }
        }
        best.map(|(i, _)| i)
    }

    /// Centre frequency of the busiest sustained channel, hertz. `None` for a silent bank.
    #[must_use]
    pub fn best_frequency(&self) -> Option<f64> {
        self.best_channel().map(|i| self.f_c[i])
    }

    /// Total sustained spikes per second summed over channels.
    #[must_use]
    pub fn total_sustained_hz(&self) -> f64 {
        self.sustained_hz.iter().sum()
    }
}

// ---------------------------------------------------------------------------------------------
// Synthetic signals
// ---------------------------------------------------------------------------------------------

/// How a chirp's frequency moves between its endpoints.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sweep {
    /// Frequency linear in time. Total phase is `2π·T·(f0 + f1)/2` — the mean frequency times the
    /// duration, exactly.
    Linear,
    /// Frequency geometric in time, so the sweep covers equal *octaves* per second and therefore
    /// roughly equal numbers of auditory filters per second. Total phase is
    /// `2π·f0·T·(r − 1)/ln r` with `r = f1/f0`.
    Exponential,
}

fn sample_count(fs: f64, seconds: f64) -> Result<usize, CochleaError> {
    let n = (fs * seconds).round();
    if !(n >= 1.0) || n > (usize::MAX as f64) {
        return Err(CochleaError::NotPositive {
            what: "signal length in samples",
            value: n,
        });
    }
    Ok(n as usize)
}

fn check_audio_frequency(f: f64, fs: f64) -> Result<f64, CochleaError> {
    let f = finite("frequency", f)?;
    let limit = fs / 2.0;
    if !(f > 0.0) || f >= limit {
        return Err(CochleaError::CentreFrequency { f_c: f, fs, limit });
    }
    Ok(f)
}

/// `seconds` of silence at `fs` hertz: the control condition.
///
/// # Errors
///
/// [`CochleaError::NotFinite`] or [`CochleaError::NotPositive`] for a bad rate or duration.
pub fn silence(fs: f64, seconds: f64) -> Result<Vec<f64>, CochleaError> {
    let fs = positive("sample rate", fs)?;
    let seconds = positive("duration", seconds)?;
    Ok(vec![0.0; sample_count(fs, seconds)?])
}

/// A pure tone: `amplitude · sin(2π·f·t)`.
///
/// Starts at zero phase so there is no step discontinuity at sample zero — a cosine here would
/// begin with a click, and the click's onset response would be indistinguishable from the tone's.
///
/// # Errors
///
/// [`CochleaError::NotFinite`] or [`CochleaError::NotPositive`] for a bad rate or duration,
/// [`CochleaError::CentreFrequency`] for a frequency outside `(0, fs/2)`.
pub fn tone(fs: f64, f: f64, amplitude: f64, seconds: f64) -> Result<Vec<f64>, CochleaError> {
    let fs = positive("sample rate", fs)?;
    let f = check_audio_frequency(f, fs)?;
    let amplitude = finite("amplitude", amplitude)?;
    let seconds = positive("duration", seconds)?;
    let n = sample_count(fs, seconds)?;
    let w = 2.0 * PI * f / fs;
    Ok((0..n).map(|i| amplitude * (w * i as f64).sin()).collect())
}

/// Two tones summed: a masker and a probe, the classic two-tone experiment.
///
/// What it is for: a masker close in frequency to a probe raises the response of the probe's own
/// channel, so the probe stops being visible in the place code. See the module doc for what kind of
/// masking this does and does not reproduce.
///
/// # Errors
///
/// As [`tone`], for either component.
pub fn two_tone(
    fs: f64,
    f_probe: f64,
    a_probe: f64,
    f_masker: f64,
    a_masker: f64,
    seconds: f64,
) -> Result<Vec<f64>, CochleaError> {
    let a = tone(fs, f_probe, a_probe, seconds)?;
    let b = tone(fs, f_masker, a_masker, seconds)?;
    if a.len() != b.len() {
        return Err(CochleaError::LengthMismatch {
            expected: a.len(),
            got: b.len(),
        });
    }
    Ok(a.iter().zip(b.iter()).map(|(x, y)| x + y).collect())
}

/// A frequency sweep from `f0` to `f1` hertz over `seconds`.
///
/// The instrument for testing tonotopy: a rising chirp should excite channels in order of centre
/// frequency, and if it does not, the bank's frequency map is wrong in a way no single-tone test
/// would show.
///
/// Starts at zero phase, for the reason [`tone`] does: a waveform that began mid-cycle would open
/// with a step, and the click's onset response is exactly what a tonotopy measurement reads.
///
/// # Errors
///
/// As [`tone`] for either endpoint, plus [`CochleaError::NotPositive`] on a
/// [`Sweep::Exponential`] whose endpoints are not both positive (which `check_audio_frequency`
/// already guarantees, so the arm is defensive).
pub fn chirp(
    fs: f64,
    f0: f64,
    f1: f64,
    amplitude: f64,
    seconds: f64,
    sweep: Sweep,
) -> Result<Vec<f64>, CochleaError> {
    let fs = positive("sample rate", fs)?;
    let f0 = check_audio_frequency(f0, fs)?;
    let f1 = check_audio_frequency(f1, fs)?;
    let amplitude = finite("amplitude", amplitude)?;
    let seconds = positive("duration", seconds)?;
    let n = sample_count(fs, seconds)?;
    let t_end = seconds;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f64 / fs;
        let phase = match sweep {
            Sweep::Linear => 2.0 * PI * (f0 * t + (f1 - f0) * t * t / (2.0 * t_end)),
            Sweep::Exponential => {
                let ratio = f1 / f0;
                if (ratio - 1.0).abs() < 1e-12 {
                    2.0 * PI * f0 * t
                } else {
                    2.0 * PI * f0 * t_end * (ratio.powf(t / t_end) - 1.0) / ratio.ln()
                }
            }
        };
        out.push(amplitude * phase.sin());
    }
    Ok(out)
}

/// Total phase in radians a [`chirp`] with these parameters accumulates.
///
/// Exposed so a test can check the generated waveform against the integral of its own instantaneous
/// frequency rather than against a stored copy of itself.
///
/// # Errors
///
/// [`CochleaError::NotFinite`] or [`CochleaError::NotPositive`] for a bad argument.
pub fn chirp_total_phase(
    f0: f64,
    f1: f64,
    seconds: f64,
    sweep: Sweep,
) -> Result<f64, CochleaError> {
    let f0 = positive("chirp start frequency", f0)?;
    let f1 = positive("chirp end frequency", f1)?;
    let seconds = positive("duration", seconds)?;
    Ok(match sweep {
        Sweep::Linear => 2.0 * PI * seconds * (f0 + f1) / 2.0,
        Sweep::Exponential => {
            let ratio = f1 / f0;
            if (ratio - 1.0).abs() < 1e-12 {
                2.0 * PI * f0 * seconds
            } else {
                2.0 * PI * f0 * seconds * (ratio - 1.0) / ratio.ln()
            }
        }
    })
}

/// A single-sample impulse: the widest-band stimulus that exists at this sample rate.
///
/// Every channel receives it at the same instant, so the times at which they respond are the
/// filterbank's own travelling-wave delays and nothing else.
///
/// # Errors
///
/// [`CochleaError::NotFinite`] or [`CochleaError::NotPositive`] for a bad rate or duration, and
/// [`CochleaError::Window`] when `at` falls outside the buffer.
pub fn click(fs: f64, amplitude: f64, seconds: f64, at: f64) -> Result<Vec<f64>, CochleaError> {
    let fs = positive("sample rate", fs)?;
    let amplitude = finite("amplitude", amplitude)?;
    let seconds = positive("duration", seconds)?;
    let at = finite("click time", at)?;
    let n = sample_count(fs, seconds)?;
    let idx = (at * fs).round();
    if !(idx >= 0.0) || idx >= n as f64 {
        return Err(CochleaError::Window {
            onset: at,
            duration: 0.0,
            total: seconds,
        });
    }
    let mut out = vec![0.0; n];
    out[idx as usize] = amplitude;
    Ok(out)
}

/// A raised-cosine gate: `0` outside the window, `1` inside, with `ramp` seconds of smooth edge.
///
/// `ramp = 0` gives a hard gate, whose own switching transient is a click with a click's spectrum.
/// Any burst meant to test frequency selectivity wants a ramp; any burst meant to test onset
/// latency may not.
fn gate(fs: f64, n: usize, onset: f64, duration: f64, ramp: f64) -> Vec<f64> {
    let mut g = vec![0.0; n];
    let start = onset * fs;
    let stop = (onset + duration) * fs;
    let ramp_n = (ramp * fs).max(0.0);
    for (i, slot) in g.iter_mut().enumerate() {
        let t = i as f64;
        if t < start || t >= stop {
            continue;
        }
        let rise = if ramp_n > 0.0 {
            ((t - start) / ramp_n).min(1.0)
        } else {
            1.0
        };
        let fall = if ramp_n > 0.0 {
            ((stop - t) / ramp_n).min(1.0)
        } else {
            1.0
        };
        let shape = rise.min(fall);
        *slot = 0.5 * (1.0 - (PI * shape).cos());
    }
    g
}

/// A tone switched on for part of a longer buffer, with raised-cosine edges.
///
/// # Errors
///
/// As [`tone`], plus [`CochleaError::Window`] when the burst does not fit inside `seconds` and
/// [`CochleaError::NotFinite`] for a non-finite window parameter.
pub fn tone_burst(
    fs: f64,
    f: f64,
    amplitude: f64,
    seconds: f64,
    onset: f64,
    duration: f64,
    ramp: f64,
) -> Result<Vec<f64>, CochleaError> {
    let carrier = tone(fs, f, amplitude, seconds)?;
    let onset = finite("burst onset", onset)?;
    let duration = positive("burst duration", duration)?;
    let ramp = finite("burst ramp", ramp)?;
    if !(onset >= 0.0) || !(ramp >= 0.0) || onset + duration > seconds {
        return Err(CochleaError::Window {
            onset,
            duration,
            total: seconds,
        });
    }
    let g = gate(fs, carrier.len(), onset, duration, ramp);
    Ok(carrier.iter().zip(g.iter()).map(|(x, w)| x * w).collect())
}

/// A burst of uniform white noise inside a longer buffer, with raised-cosine edges.
///
/// Samples are drawn uniformly from `[-amplitude, amplitude]`, so the burst's root-mean-square
/// level is `amplitude/√3` — a closed form, which is what its test checks. Uniform rather than
/// Gaussian because it needs one draw per sample instead of the Box-Muller pair, and the auditory
/// front end sees only the filtered result, which is close to Gaussian either way by the central
/// limit theorem.
///
/// Deterministic given `rng`'s seed.
///
/// # Errors
///
/// As [`tone_burst`], without the carrier-frequency checks.
pub fn noise_burst(
    rng: &mut Rng,
    fs: f64,
    amplitude: f64,
    seconds: f64,
    onset: f64,
    duration: f64,
    ramp: f64,
) -> Result<Vec<f64>, CochleaError> {
    let fs = positive("sample rate", fs)?;
    let amplitude = finite("amplitude", amplitude)?;
    let seconds = positive("duration", seconds)?;
    let onset = finite("burst onset", onset)?;
    let duration = positive("burst duration", duration)?;
    let ramp = finite("burst ramp", ramp)?;
    if !(onset >= 0.0) || !(ramp >= 0.0) || onset + duration > seconds {
        return Err(CochleaError::Window {
            onset,
            duration,
            total: seconds,
        });
    }
    let n = sample_count(fs, seconds)?;
    let g = gate(fs, n, onset, duration, ramp);
    Ok((0..n)
        .map(|i| amplitude * (2.0 * rng.next_f64() - 1.0) * g[i])
        .collect())
}

/// Amplitude of the component at `f` hertz in `samples`, by a single-bin discrete Fourier
/// transform.
///
/// `2·|Σ x[n]·e^{-i·2π·f·n/fs}| / N`. For a pure tone spanning a whole number of cycles this is
/// exact; otherwise the negative-frequency image leaks in at order `1/N`, so a measurement wants
/// either many cycles or a length chosen to make `f·N/fs` an integer.
///
/// One bin rather than a full transform because this module has no `FFT` and does not need one: a
/// filterbank *is* the spectral analysis, and this exists to check the filterbank.
///
/// # Errors
///
/// [`CochleaError::NotFinite`] for a bad sample, rate or frequency, [`CochleaError::NotPositive`]
/// for a non-positive rate, and [`CochleaError::LengthMismatch`] for an empty buffer.
pub fn bin_amplitude(samples: &[f64], fs: f64, f: f64) -> Result<f64, CochleaError> {
    let fs = positive("sample rate", fs)?;
    let f = finite("frequency", f)?;
    finite_slice("signal sample", samples)?;
    if samples.is_empty() {
        return Err(CochleaError::LengthMismatch {
            expected: 1,
            got: 0,
        });
    }
    let w = 2.0 * PI * f / fs;
    let (mut re, mut im) = (0.0, 0.0);
    for (n, &x) in samples.iter().enumerate() {
        let p = w * n as f64;
        re += x * p.cos();
        im -= x * p.sin();
    }
    Ok(2.0 * re.hypot(im) / samples.len() as f64)
}

/// Root-mean-square level of a buffer, in input units.
///
/// # Errors
///
/// [`CochleaError::NotFinite`] for a bad sample and [`CochleaError::LengthMismatch`] for an empty
/// buffer, which has no mean of anything.
pub fn rms(samples: &[f64]) -> Result<f64, CochleaError> {
    finite_slice("signal sample", samples)?;
    if samples.is_empty() {
        return Err(CochleaError::LengthMismatch {
            expected: 1,
            got: 0,
        });
    }
    let s: f64 = samples.iter().map(|x| x * x).sum();
    Ok((s / samples.len() as f64).sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    const FS: f64 = 48_000.0;

    // -------------------------------------------------------------------------------------
    // Test instruments. These measure the filter the way an experimenter would — by driving it
    // and reading the output — so that no test compares the implementation against the same
    // arithmetic that produced it.
    // -------------------------------------------------------------------------------------

    /// Mean analytic envelope of `f_c`'s channel under a steady unit tone at `f`, after the
    /// transient has decayed by `e^{-14}`.
    fn steady_envelope(f_c: f64, fs: f64, f: f64) -> f64 {
        steady_envelope_at_order(f_c, fs, 4, f)
    }

    /// As `steady_envelope`, for a cascade of any order.
    fn steady_envelope_at_order(f_c: f64, fs: f64, order: usize, f: f64) -> f64 {
        let mut g = Gammatone::with_shape(f_c, fs, order, PATTERSON_B).expect("valid channel");
        let settle = (14.0 / g.decay_rate() * fs).ceil() as usize;
        // Average over whole ripple periods so the residual negative-frequency image cancels.
        let window = ((fs / f).ceil() as usize) * 8;
        let w = 2.0 * PI * f / fs;
        for n in 0..settle {
            g.step((w * n as f64).sin());
        }
        let mut sum = 0.0;
        for n in settle..settle + window {
            g.step((w * n as f64).sin());
            sum += g.envelope();
        }
        sum / window as f64
    }

    /// Ternary-search the measured response for its peak frequency.
    fn measured_peak(f_c: f64, fs: f64) -> f64 {
        let (mut a, mut b) = (f_c * 0.5, f_c * 1.5);
        while b - a > 1e-3 {
            let m1 = a + (b - a) / 3.0;
            let m2 = b - (b - a) / 3.0;
            if steady_envelope(f_c, fs, m1) < steady_envelope(f_c, fs, m2) {
                a = m1;
            } else {
                b = m2;
            }
        }
        0.5 * (a + b)
    }

    /// Bisect the measured response for the frequency where it falls to `target`, searching
    /// upward from `f_c` if `up`, downward otherwise.
    fn measured_crossing(f_c: f64, fs: f64, target: f64, up: bool) -> f64 {
        measured_crossing_at_order(f_c, fs, 4, target, up)
    }

    fn measured_crossing_at_order(f_c: f64, fs: f64, order: usize, target: f64, up: bool) -> f64 {
        let span = 6.0 * erb_hz(f_c);
        let (mut lo, mut hi) = if up {
            (f_c, f_c + span)
        } else {
            (f_c, (f_c - span).max(1.0))
        };
        for _ in 0..50 {
            let m = 0.5 * (lo + hi);
            if steady_envelope_at_order(f_c, fs, order, m) > target {
                lo = m;
            } else {
                hi = m;
            }
        }
        0.5 * (lo + hi)
    }

    fn argmax(xs: &[f64]) -> usize {
        let mut best = 0;
        for (i, &x) in xs.iter().enumerate() {
            if x > xs[best] {
                best = i;
            }
        }
        best
    }

    // -------------------------------------------------------------------------------------
    // (b) The ERB scale, against Glasberg & Moore's own equations
    // -------------------------------------------------------------------------------------

    /// Both 1990 equations at frequencies whose values are quoted in the literature. The numbers
    /// on the right are typed from the published formulae, not read off this implementation.
    #[test]
    fn erb_scale_reproduces_glasberg_and_moore() {
        // ERB(f) = 24.7 (4.37 f/1000 + 1).
        assert!((erb_hz(0.0) - 24.7).abs() < 1e-12, "{}", erb_hz(0.0));
        assert!(
            (erb_hz(1000.0) - 132.639).abs() < 1e-3,
            "ERB(1 kHz) = {}",
            erb_hz(1000.0)
        );
        assert!(
            (erb_hz(4000.0) - 456.456).abs() < 1e-3,
            "ERB(4 kHz) = {}",
            erb_hz(4000.0)
        );
        // E(f) = 21.4 log10(4.37 f/1000 + 1): ~15.6 ERBs at 1 kHz, ~39 below 15 kHz.
        let e1k = erb_rate(1000.0).expect("1 kHz is on the scale");
        assert!((e1k - 15.6215).abs() < 1e-3, "E(1 kHz) = {e1k}");
        let e15k = erb_rate(15000.0).expect("15 kHz is on the scale");
        assert!((e15k - 39.0154).abs() < 1e-3, "E(15 kHz) = {e15k}");
        // Round trip.
        for &f in &[50.0, 250.0, 1000.0, 4000.0, 16000.0] {
            let back = erb_rate_to_hz(erb_rate(f).expect("on scale")).expect("invertible");
            assert!((back - f).abs() < 1e-8, "{f} -> {back}");
        }
    }

    /// The two Glasberg & Moore equations are not independent: `E` is the integral of `1/ERB`, so
    /// `ERB(f) · dE/df` must be the constant `24.7·21.4·4.37/(1000·ln 10) = 1.003173`, at every
    /// frequency. A mutation of a constant in either equation alone breaks this; this is the test
    /// that makes the pair check each other rather than themselves.
    #[test]
    fn the_two_erb_equations_share_their_constant() {
        for &f in &[100.0, 500.0, 1000.0, 4000.0, 10000.0] {
            let h = f * 1e-6;
            let d = (erb_rate(f + h).expect("on scale") - erb_rate(f - h).expect("on scale"))
                / (2.0 * h);
            let product = erb_hz(f) * d;
            assert!(
                (product - 1.003173).abs() < 1e-5,
                "at {f} Hz the product was {product}, not 1.003173"
            );
        }
    }

    #[test]
    fn erb_space_steps_are_equal_on_the_erb_rate_axis() {
        let f = erb_space(100.0, 8000.0, 16).expect("a valid band");
        assert_eq!(f.len(), 16);
        assert!((f[0] - 100.0).abs() < 1e-8);
        assert!((f[15] - 8000.0).abs() < 1e-6, "top channel at {}", f[15]);
        let e: Vec<f64> = f.iter().map(|&x| erb_rate(x).expect("on scale")).collect();
        let step = e[1] - e[0];
        for i in 1..e.len() {
            assert!(
                (e[i] - e[i - 1] - step).abs() < 1e-9,
                "step {i} was {}, not {step}",
                e[i] - e[i - 1]
            );
        }
        // Strictly ascending in hertz, which the address layout depends on.
        for i in 1..f.len() {
            assert!(f[i] > f[i - 1], "channel {i} is not above its predecessor");
        }
    }

    /// A bank whose `ERB`-rate step is exactly 1 must have hertz spacings equal to the local `ERB`,
    /// to within the second-order error of the linearisation. This is the same cross-check as
    /// `the_two_erb_equations_share_their_constant`, taken through the public spacing function.
    #[test]
    fn one_erb_of_spacing_is_one_erb_wide_in_hertz() {
        let (lo, hi) = (200.0, 6000.0);
        let span = erb_rate(hi).expect("on scale") - erb_rate(lo).expect("on scale");
        let n = span.round() as usize + 1;
        let f = erb_space(lo, hi, n).expect("a valid band");
        let step = span / (n - 1) as f64;
        assert!((step - 1.0).abs() < 0.02, "step was {step} ERB");
        for i in 1..f.len() {
            let mid = 0.5 * (f[i] + f[i - 1]);
            let gap = f[i] - f[i - 1];
            let expected = step * erb_hz(mid) / 1.003173;
            assert!(
                (gap - expected).abs() / expected < 0.01,
                "gap at {mid} Hz was {gap}, expected about {expected}"
            );
        }
    }

    #[test]
    fn erb_space_refuses_what_it_cannot_lay_out() {
        assert!(matches!(
            erb_space(1000.0, 100.0, 4),
            Err(CochleaError::Band { .. })
        ));
        assert!(matches!(
            erb_space(100.0, 1000.0, 0),
            Err(CochleaError::Channels { n: 0 })
        ));
        assert!(matches!(
            erb_space(-1.0, 1000.0, 4),
            Err(CochleaError::NotPositive { .. })
        ));
        assert!(matches!(
            erb_space(f64::NAN, 1000.0, 4),
            Err(CochleaError::NotFinite { .. })
        ));
        assert_eq!(erb_space(440.0, 880.0, 1).expect("one channel"), vec![440.0]);
        assert!(erb_rate(-500.0).is_none(), "off the bottom of the scale");
        assert!(erb_rate(f64::INFINITY).is_none());
    }

    // -------------------------------------------------------------------------------------
    // (a) The gammatone's centre frequency and bandwidth are what they claim
    // -------------------------------------------------------------------------------------

    /// Drive each channel with a pure-tone sweep and find where it responds most. The peak must
    /// land on the stated `f_c`; the tolerance below is 0.02 `ERB`, which at 250 Hz is 1.0 Hz and
    /// is set by the ternary search's own 1 mHz bracket plus the residual negative-frequency
    /// image, not by any slack in the filter.
    #[test]
    fn a_gammatone_peaks_at_its_stated_centre_frequency() {
        // ⛔ The last entry is the guard itself. At the old guard of 0.45·fs this test fails by
        // 1.9x its own tolerance; the constant is now pinned to a measurement, not an argument.
        for &f_c in &[250.0, 1000.0, 4000.0, FS * NYQUIST_GUARD] {
            let peak = measured_peak(f_c, FS);
            let tol = 0.02 * erb_hz(f_c);
            assert!(
                (peak - f_c).abs() < tol,
                "channel at {f_c} Hz peaked at {peak} Hz, off by {} Hz (tolerance {tol})",
                peak - f_c
            );
            let g = Gammatone::new(f_c, FS).expect("valid channel");
            assert_eq!(g.peak_frequency(), f_c);
        }
    }

    /// The gain normalisation: a unit tone at `f_c` must come out at unit amplitude, so that the
    /// magnitude response is a ratio against the channel's own peak.
    #[test]
    fn a_unit_tone_at_the_centre_frequency_comes_out_at_unit_amplitude() {
        for &f_c in &[250.0, 1000.0, 4000.0] {
            let a = steady_envelope(f_c, FS, f_c);
            assert!((a - 1.0).abs() < 2e-3, "channel at {f_c} Hz peaked at {a}");
        }
    }

    /// The −3 dB width, measured by bisecting the driven response, against the **published**
    /// table for a fourth-order gammatone: 0.887 `ERB`. The literal on the right is the product
    /// `0.870 × 1.019` of two entries of Holdsworth et al., Annex C of the `SVOS` Final Report
    /// Part A (1988), Table 1, which prints the factors rather than the product (this doc used to
    /// credit the number to "Patterson et al. (1988)"). It does not come from
    /// [`Gammatone::bandwidth_3db_continuous`], so mutating [`PATTERSON_B`] or the order breaks
    /// this test.
    #[test]
    fn the_three_db_width_matches_the_published_erb_relation() {
        for &f_c in &[250.0, 1000.0, 4000.0] {
            let peak = steady_envelope(f_c, FS, f_c);
            let target = peak / 2f64.sqrt();
            let hi = measured_crossing(f_c, FS, target, true);
            let lo = measured_crossing(f_c, FS, target, false);
            let measured = hi - lo;
            let published = 0.887 * erb_hz(f_c);
            assert!(
                (measured - published).abs() / published < 0.01,
                "channel at {f_c} Hz measured {measured} Hz wide, published says {published} Hz"
            );
            // And the module's own two derivations agree with the measurement and each other.
            let g = Gammatone::new(f_c, FS).expect("valid channel");
            let discrete = g.bandwidth_3db().expect("a narrow enough filter");
            let continuous = g.bandwidth_3db_continuous();
            assert!(
                (discrete - continuous).abs() / continuous < 1e-3,
                "discrete {discrete} vs continuous {continuous}"
            );
            assert!(
                (measured - discrete).abs() / discrete < 0.01,
                "measured {measured} vs closed form {discrete}"
            );
        }
    }

    /// [`PATTERSON_B`] against the table row it is transcribed from: Holdsworth, Nimmo-Smith,
    /// Patterson & Rice, Annex C of the `SVOS` Final Report Part A (1988), Table 1, `n = 4`, which
    /// prints `a_4 = 0.982`, `1/a_4 = 1.019`, `c_4 = 0.870` and `1/c_4 = 1.149`. `a_n` is the
    /// order-`n` gammatone's `ERB` over `b`, the integral of `(1 + u²)^{-n}` along the whole line;
    /// with `u = tan θ` that is the integral of `cos^{2n−2} θ` over one period, which the midpoint
    /// rule integrates exactly (a trigonometric polynomial of degree 3 in `2θ`, sampled 64 times),
    /// and whose closed form at `n = 4` is `5π/16`, the annex's own eqn (6),
    /// `π·(2n−2)!·2^{−(2n−2)}/((n−1)!)²`, at `n = 4`. `c_n` is the −3 dB width over `b`,
    /// `2√(2^{1/n} − 1)`. Each printed entry must be the three-decimal rounding of the recomputed
    /// value, and the 0.887 `ERB` the width test uses must be the three-decimal rounding of the
    /// product of the two printed entries, as the module doc says it is.
    #[test]
    fn patterson_b_is_row_four_of_annex_c_table_1() {
        let round3 = |v: f64| (v * 1000.0).round() / 1000.0;
        let steps = 64;
        let h = PI / f64::from(steps);
        let a4: f64 = (0..steps)
            .map(|k| (-PI / 2.0 + (f64::from(k) + 0.5) * h).cos().powi(6) * h)
            .sum();
        assert!((a4 - 5.0 * PI / 16.0).abs() < 1e-12, "a_4 integrated to {a4}");
        // Eqn (6) at n = 4: π · 6! · 2^-6 / (3!)².
        let eqn6 = PI * 720.0 / 64.0 / 36.0;
        assert!((a4 - eqn6).abs() < 1e-12, "eqn (6) gives {eqn6}, the integral {a4}");
        let c4 = 2.0 * (2f64.powf(0.25) - 1.0).sqrt();
        assert_eq!(round3(a4), 0.982, "a_4 = {a4}");
        assert_eq!(round3(1.0 / a4), PATTERSON_B, "1/a_4 = {}", 1.0 / a4);
        assert_eq!(round3(c4), 0.870, "c_4 = {c4}");
        assert_eq!(round3(1.0 / c4), 1.149, "1/c_4 = {}", 1.0 / c4);
        assert_eq!(round3(0.870 * 1.019), 0.887, "the published width is c_4 times 1/a_4");
    }

    /// The whole shape, not just its two landmarks: the driven response at a spread of detunings
    /// must match the closed-form transfer function, including 50 dB down the skirt.
    #[test]
    fn the_driven_response_follows_the_closed_form_transfer_function() {
        let f_c = 1000.0;
        let g = Gammatone::new(f_c, FS).expect("valid channel");
        let erb = erb_hz(f_c);
        for &k in &[-4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0] {
            let f = f_c + k * erb;
            let measured = steady_envelope(f_c, FS, f);
            let predicted = g.magnitude_response(f);
            assert!(
                (measured - predicted).abs() / predicted < 0.01,
                "at {k} ERB detuning: measured {measured}, closed form {predicted}"
            );
        }
        // The skirt slope: 6n dB per octave of detuning for large detuning. Between 4 and 8 ERB
        // the response must fall by close to a factor of 2^4 = 16 in amplitude.
        let a4 = g.magnitude_response(f_c + 4.0 * erb);
        let a8 = g.magnitude_response(f_c + 8.0 * erb);
        let ratio = a4 / a8;
        assert!(
            (10.0..24.0).contains(&ratio),
            "fourth-order skirt fell by {ratio}x over one doubling of detuning"
        );
    }

    #[test]
    fn a_gammatone_refuses_what_it_cannot_filter() {
        assert!(matches!(
            Gammatone::new(FS * 0.5, FS),
            Err(CochleaError::CentreFrequency { .. })
        ));
        assert!(matches!(
            Gammatone::new(FS * 0.46, FS),
            Err(CochleaError::CentreFrequency { .. })
        ));
        assert!(matches!(
            Gammatone::new(FS * 0.41, FS),
            Err(CochleaError::CentreFrequency { .. })
        ), "0.41·fs is past the measured guard of 0.40");
        assert!(Gammatone::new(FS * 0.40, FS).is_ok());
        assert!(Gammatone::new(FS * NYQUIST_GUARD, FS).is_ok());
        // The order bound, as a literal: 8 builds, 9 does not.
        assert!(Gammatone::with_shape(1000.0, FS, 8, PATTERSON_B).is_ok());
        assert!(matches!(
            Gammatone::with_shape(1000.0, FS, 9, PATTERSON_B),
            Err(CochleaError::Order { order: 9 })
        ));
        assert!(matches!(
            Gammatone::new(-100.0, FS),
            Err(CochleaError::CentreFrequency { .. })
        ));
        assert!(matches!(
            Gammatone::new(1000.0, 0.0),
            Err(CochleaError::NotPositive { .. })
        ));
        assert!(matches!(
            Gammatone::with_shape(1000.0, FS, 0, PATTERSON_B),
            Err(CochleaError::Order { order: 0 })
        ));
        assert!(matches!(
            Gammatone::with_shape(1000.0, FS, MAX_ORDER + 1, PATTERSON_B),
            Err(CochleaError::Order { .. })
        ));
        let mut g = Gammatone::new(1000.0, FS).expect("valid channel");
        assert!(matches!(
            g.filter(&[0.0, f64::NAN, 0.0]),
            Err(CochleaError::NotFinite { index: 1, .. })
        ));
    }

    /// Order is a real parameter, not decoration: a third-order filter's −3 dB width uses
    /// `2^{1/3}` and a fifth-order one uses `2^{1/5}`, so the widths must differ in a stated
    /// direction.
    ///
    /// ⛔ This compares algebra with algebra — `bandwidth_3db` against a retyped
    /// `bandwidth_3db_continuous` — and drives no filter; the comment that stood here said it
    /// "catches a cascade that silently runs a fixed number of stages", and it cannot: `step`
    /// capped at four stages passed it. `every_order_is_driven_and_measured` is the test that
    /// does what this one claimed.
    #[test]
    fn changing_the_order_changes_the_shape_the_way_the_algebra_says() {
        let f_c = 1000.0;
        let mut previous = f64::INFINITY;
        for order in 2..=6 {
            let g = Gammatone::with_shape(f_c, FS, order, PATTERSON_B).expect("valid channel");
            let bw = g.bandwidth_3db().expect("narrow enough");
            let expected = 2.0
                * PATTERSON_B
                * erb_hz(f_c)
                * (2f64.powf(1.0 / order as f64) - 1.0).sqrt();
            assert!(
                (bw - expected).abs() / expected < 1e-3,
                "order {order}: {bw} vs {expected}"
            );
            assert!(bw < previous, "order {order} was not narrower than {order} - 1");
            previous = bw;
        }
    }

    /// ⛔ EVERY ORDER, DRIVEN. Four of the module's closed forms carry an `n` no test had ever run:
    /// `step` capped at four stages, `with_shape`'s gain at `powi(4)`, the continuous bandwidth at
    /// `2^{1/4}` and the magnitude response at `powf(2.0)` all passed 58 tests, because every
    /// driven test used the default fourth order. Here orders 2 to 6 are each measured four ways:
    /// the impulse-envelope peak lands exactly on `peak_latency`, a unit tone at `f_c` comes out at
    /// unit amplitude, the −3 dB width bisected from the driven response matches the continuous
    /// closed form, and the driven response one ERB above `f_c` matches `magnitude_response`.
    #[test]
    fn every_order_is_driven_and_measured() {
        let f_c = 1000.0;
        for order in 2..=6 {
            let mut g = Gammatone::with_shape(f_c, FS, order, PATTERSON_B).expect("valid channel");
            let n = (0.05 * FS) as usize;
            let mut env = Vec::with_capacity(n);
            for i in 0..n {
                g.step(if i == 0 { 1.0 } else { 0.0 });
                env.push(g.envelope());
            }
            let want = (g.peak_latency() * FS).round() as usize;
            assert!(want > 0, "order {order}: latency rounds to zero samples");
            assert_eq!(argmax(&env), want, "order {order}: impulse-envelope peak");

            let a = steady_envelope_at_order(f_c, FS, order, f_c);
            assert!((a - 1.0).abs() < 3e-3, "order {order}: unit tone at f_c came out at {a}");

            let target = 1.0 / 2f64.sqrt();
            let hi = measured_crossing_at_order(f_c, FS, order, target, true);
            let lo = measured_crossing_at_order(f_c, FS, order, target, false);
            let width = hi - lo;
            let want_bw = g.bandwidth_3db_continuous();
            assert!(
                (width - want_bw).abs() / want_bw < 1e-2,
                "order {order}: measured -3 dB width {width} against {want_bw}"
            );

            let f1 = f_c + erb_hz(f_c);
            let driven = steady_envelope_at_order(f_c, FS, order, f1);
            let closed = g.magnitude_response(f1);
            assert!(
                (driven - closed).abs() < 2e-3,
                "order {order}: driven {driven} vs magnitude_response {closed} at +1 ERB"
            );
        }
    }

    // -------------------------------------------------------------------------------------
    // (d) The travelling-wave delay
    // -------------------------------------------------------------------------------------

    /// A click excites every channel at the same instant; the latency of each channel's envelope
    /// peak is its own ringing time. Checked against the exact discrete formula
    /// `⌊(n − 1)·r/(1 − r)⌋/fs` and, separately, against the continuous `(n − 1)/β`.
    #[test]
    fn a_click_produces_the_predicted_travelling_wave_delay() {
        let x = click(FS, 1.0, 0.2, 0.0).expect("a valid click");
        for &f_c in &[200.0, 500.0, 1000.0, 4000.0] {
            let mut g = Gammatone::new(f_c, FS).expect("valid channel");
            let env = g.envelopes(&x).expect("finite signal");
            let measured = argmax(&env) as f64 / FS;
            let discrete = g.peak_latency();
            assert!(
                (measured - discrete).abs() < 0.5 / FS,
                "channel at {f_c} Hz peaked at sample {}, the exact discrete formula says {}",
                measured * FS,
                discrete * FS
            );
            // The continuum is a different object and is allowed to differ by the n-1 samples the
            // discretisation costs: 0.4 % of the latency at 200 Hz, 4.6 % at 4 kHz. The tolerance
            // is 6 % because that is where the arithmetic puts it, not to make room for slack.
            let continuous = g.peak_latency_continuous();
            assert!(
                (measured - continuous).abs() / continuous < 0.06,
                "channel at {f_c} Hz peaked at {measured} s, continuum says {continuous} s"
            );
            assert!(
                measured < continuous,
                "the discrete peak must fall BELOW the continuous one, by about (n-1)/fs"
            );
        }
    }

    /// The delay must grow toward the apex — strictly, channel by channel — and the low end must
    /// be far slower than the high end. This is the property the cochlea's mechanics produce, and
    /// a bank that got its `ERB` map backwards would fail it in the ordering rather than in any
    /// single number.
    #[test]
    fn the_travelling_wave_delay_grows_toward_the_apex() {
        let f_c = erb_space(150.0, 6000.0, 20).expect("a valid band");
        let x = click(FS, 1.0, 0.25, 0.0).expect("a valid click");
        let mut previous = f64::INFINITY;
        let mut first = 0.0;
        let mut last = 0.0;
        for (i, &f) in f_c.iter().enumerate() {
            let mut g = Gammatone::new(f, FS).expect("valid channel");
            let env = g.envelopes(&x).expect("finite signal");
            let latency = argmax(&env) as f64 / FS;
            assert!(
                latency < previous,
                "channel {i} at {f} Hz peaked at {latency} s, not before its lower neighbour's \
                 {previous} s"
            );
            if i == 0 {
                first = latency;
            }
            last = latency;
            previous = latency;
        }
        assert!(
            first > 10.0 * last,
            "apex latency {first} s was not much longer than base latency {last} s"
        );
    }

    // -------------------------------------------------------------------------------------
    // (c) Tonotopy
    // -------------------------------------------------------------------------------------

    /// **The property the whole architecture exists for.** A rising chirp must sweep through the
    /// channels in strictly increasing order of centre frequency. Asserted as a strict ordering
    /// over every adjacent pair, not as a correlation, because a correlation of 0.99 is compatible
    /// with two channels being swapped.
    #[test]
    fn a_rising_chirp_excites_channels_in_increasing_order_of_centre_frequency() {
        let f_c = erb_space(300.0, 4000.0, 14).expect("a valid band");
        let x = chirp(FS, 250.0, 5000.0, 1.0, 1.2, Sweep::Linear).expect("a valid chirp");
        let mut peaks = Vec::with_capacity(f_c.len());
        for &f in &f_c {
            let mut g = Gammatone::new(f, FS).expect("valid channel");
            let env = g.envelopes(&x).expect("finite signal");
            peaks.push(argmax(&env) as f64 / FS);
        }
        for i in 1..peaks.len() {
            assert!(
                peaks[i] > peaks[i - 1],
                "channel {i} ({} Hz) peaked at {} s, not after channel {} ({} Hz) at {} s",
                f_c[i],
                peaks[i],
                i - 1,
                f_c[i - 1],
                peaks[i - 1]
            );
        }
        // And a FALLING chirp must reverse the order. Without this, a test that merely rewarded
        // "peaks spread out in time" would pass on a bank with no frequency map at all.
        let down = chirp(FS, 5000.0, 250.0, 1.0, 1.2, Sweep::Linear).expect("a valid chirp");
        let mut down_peaks = Vec::with_capacity(f_c.len());
        for &f in &f_c {
            let mut g = Gammatone::new(f, FS).expect("valid channel");
            let env = g.envelopes(&down).expect("finite signal");
            down_peaks.push(argmax(&env) as f64 / FS);
        }
        for i in 1..down_peaks.len() {
            assert!(
                down_peaks[i] < down_peaks[i - 1],
                "falling chirp: channel {i} peaked at {} s, not before {} s",
                down_peaks[i],
                down_peaks[i - 1]
            );
        }
    }

    /// The place code from the spiking bank itself: a tone's busiest channel is the one tuned
    /// nearest it, across the whole span of the bank.
    #[test]
    fn a_tone_drives_the_channel_tuned_nearest_it() {
        let mut bank = Filterbank::erb_bank(
            FS,
            200.0,
            6000.0,
            28,
            Transduction::HalfWave(Compression::default()),
        )
        .expect("a valid bank");
        for &f in &[400.0, 1000.0, 3000.0] {
            let x = tone(FS, f, 1.0, 0.25).expect("a valid tone");
            let train = bank.spike_train(&x).expect("finite signal");
            let gram = bank.cochleagram(&train, x.len() as u64);
            let best = gram.best_frequency().expect("some channel fired");
            let nearest = bank
                .centre_frequencies()
                .iter()
                .copied()
                .fold(f64::INFINITY, |acc, c| {
                    if (c - f).abs() < (acc - f).abs() {
                        c
                    } else {
                        acc
                    }
                });
            assert!(
                (best - nearest).abs() < 1e-9,
                "a {f} Hz tone's busiest channel was {best} Hz, nearest is {nearest} Hz"
            );
        }
    }

    // -------------------------------------------------------------------------------------
    // Compression
    // -------------------------------------------------------------------------------------

    /// The exact identity a power law is for: `D` dB in becomes `e·D` dB out, to floating point.
    #[test]
    fn power_compression_scales_a_decibel_range_exactly() {
        for &e in &[0.2, 0.4, 0.5, 1.0] {
            let c = Compression::Power { exponent: e }.validate().expect("valid");
            let (lo, hi) = (1e-5f64, 1.0f64);
            let in_db = 20.0 * (hi / lo).log10();
            let out_db = 20.0 * (c.apply(hi) / c.apply(lo)).log10();
            assert!(
                (out_db - e * in_db).abs() < 1e-9,
                "exponent {e}: {in_db} dB in became {out_db} dB out, expected {}",
                e * in_db
            );
        }
        // Exponent 1 is the identity, exactly, and therefore equals Linear.
        let unit = Compression::Power { exponent: 1.0 };
        for &x in &[0.0, 0.3, 1.0, 17.0] {
            assert_eq!(unit.apply(x), Compression::Linear.apply(x));
        }
        // Rectification happens inside apply, so a negative input is zero, not NaN from a
        // fractional power of a negative number.
        assert_eq!(Compression::default().apply(-3.0), 0.0);
    }

    /// The logarithm's small-signal limit: `ln(1 + u) → u`, with a second-order error of `u²/2`.
    #[test]
    fn log_compression_is_linear_below_its_knee() {
        let c = Compression::Log { knee: 1.0 }.validate().expect("valid");
        for &x in &[1e-6, 1e-4, 1e-3] {
            let y = c.apply(x);
            assert!(
                (y - x).abs() < x * x,
                "ln(1 + {x}) = {y}, which is not within x^2 of x"
            );
        }
        // And it compresses: a decade of input must become less than a decade of output.
        let decade = c.apply(10.0) / c.apply(1.0);
        assert!(decade < 10.0, "a decade became {decade}x");
        assert!(decade > 1.0, "compression inverted the order");
        assert_eq!(c.apply(0.0), 0.0, "silence must compress to silence");

        // ⛔ The knee divides. At `knee = 1`, the only value above, `x / knee` and `x` are the
        // same expression, so a `Log` that ignored its knee was green. At knee 0.01 a 1e-4 input
        // is one hundredth of the way to the knee: ln(1.01) = 0.00995, not 1e-4.
        let c = Compression::Log { knee: 0.01 }.validate().expect("valid");
        let y = c.apply(1e-4);
        assert!((y - 1e-2).abs() < 1e-4, "ln(1 + 1e-4 / 0.01) = {y}, not near 1e-2");
    }

    #[test]
    fn compression_refuses_an_exponent_that_is_not_compression() {
        assert!(matches!(
            Compression::Power { exponent: 1.5 }.validate(),
            Err(CochleaError::Exponent { .. })
        ));
        assert!(matches!(
            Compression::Power { exponent: 0.0 }.validate(),
            Err(CochleaError::Exponent { .. })
        ));
        assert!(matches!(
            Compression::Power { exponent: -0.4 }.validate(),
            Err(CochleaError::Exponent { .. })
        ));
        assert!(matches!(
            Compression::Power {
                exponent: f64::NAN
            }
            .validate(),
            Err(CochleaError::NotFinite { .. })
        ));
        assert!(matches!(
            Compression::Log { knee: 0.0 }.validate(),
            Err(CochleaError::NotPositive { .. })
        ));
        assert!(Compression::Linear.validate().is_ok());
    }

    #[test]
    fn half_wave_rectification_keeps_one_side_only() {
        assert_eq!(half_wave(1.0), 1.0);
        assert_eq!(half_wave(-1.0), 0.0);
        assert_eq!(half_wave(0.0), 0.0);
        // The mean of a rectified sine is A/pi, which is the DC term a hair cell actually sees.
        let x = tone(FS, 1000.0, 1.0, 0.048).expect("a valid tone");
        let mean: f64 = x.iter().map(|&v| half_wave(v)).sum::<f64>() / x.len() as f64;
        assert!(
            (mean - 1.0 / PI).abs() < 1e-3,
            "rectified mean was {mean}, not 1/pi"
        );
    }

    // -------------------------------------------------------------------------------------
    // AGC
    // -------------------------------------------------------------------------------------

    #[test]
    fn the_agc_settles_on_its_closed_form_gain() {
        let dt = 1.0 / FS;
        for &x in &[0.1, 1.0, 10.0] {
            let mut a = Agc::new(10e-3, 2.0).expect("valid loop");
            for _ in 0..(FS as usize / 5) {
                a.step(dt, x);
            }
            let expected = a.steady_state_gain(x);
            assert!(
                (a.gain() - expected).abs() < 1e-9,
                "at x = {x} the gain settled on {}, not {expected}",
                a.gain()
            );
            // And it is really compressing: gain falls as level rises.
            assert!(a.gain() < 1.0);
        }
        let quiet = Agc::new(10e-3, 2.0).expect("valid loop");
        assert_eq!(quiet.gain(), 1.0, "a loop at rest must not attenuate");
    }

    /// The detector's step response is exactly `1 − e^{-t/τ}` at every sample, because exponential
    /// Euler is the exact solution for a constant input — not an approximation that happens to be
    /// close.
    #[test]
    fn the_agc_level_follows_the_exact_one_pole_step_response() {
        let dt = 1.0 / FS;
        let tau = 5e-3;
        let mut a = Agc::new(tau, 1e-9).expect("valid loop");
        for n in 1..=2000 {
            a.step(dt, 1.0);
            let t = n as f64 * dt;
            let expected = 1.0 - (-t / tau).exp();
            assert!(
                (a.level() - expected).abs() < 1e-12,
                "sample {n}: level {} vs {expected}",
                a.level()
            );
        }
    }

    /// **A gain control must not react to a sample within that sample.** The loop applies the gain
    /// it already had and only then updates its level, so the first sample of a step passes at
    /// unity gain — exactly. Reversing those two lines makes an algebraic loop that attenuates the
    /// transient it exists to let through, and no steady-state measurement can see the difference,
    /// because both orderings settle on the same fixed point.
    #[test]
    fn the_agc_passes_the_first_sample_of_a_transient_ungained() {
        let dt = 1.0 / FS;
        let mut a = Agc::new(1e-3, 50.0).expect("valid loop");
        let first = a.step(dt, 1.0);
        assert_eq!(
            first, 1.0,
            "the first sample came out at {first}; the loop reacted to it within itself"
        );
        // And by the second sample it HAS reacted, so the ordering is not simply a delay line.
        let second = a.step(dt, 1.0);
        assert!(
            second < 1.0,
            "the loop never engaged: the second sample was still {second}"
        );
        // The second sample's gain is the one the level reached after exactly one step.
        let alpha = 1.0 - (-dt / 1e-3f64).exp();
        let expected = 1.0 / (1.0 + 50.0 * alpha);
        assert!(
            (second - expected).abs() < 1e-12,
            "second sample {second} vs the one-step closed form {expected}"
        );
    }

    #[test]
    fn the_agc_refuses_a_degenerate_loop() {
        assert!(matches!(
            Agc::new(0.0, 1.0),
            Err(CochleaError::NotPositive { .. })
        ));
        assert!(matches!(
            Agc::new(1e-3, -1.0),
            Err(CochleaError::NotPositive { .. })
        ));
        assert!(matches!(
            Agc::new(f64::INFINITY, 1.0),
            Err(CochleaError::NotFinite { .. })
        ));
    }

    // -------------------------------------------------------------------------------------
    // Meddis
    // -------------------------------------------------------------------------------------

    /// Integrate the three coupled equations to equilibrium and compare against the algebraic
    /// solution derived from the same three equations. This is what makes the integrator
    /// checkable: the steady state has a closed form even though the transient does not.
    #[test]
    fn meddis_settles_on_its_closed_form_steady_state() {
        let dt = 1.0 / FS;
        for &s in &[0.0, 1.0, 10.0, 100.0, 1000.0] {
            let mut m = Meddis::default();
            // Six seconds: the slowest mode of the coupled system relaxes on the order of 1/y,
            // about 200 ms, so a one-second run is still 4 parts in 10^6 short of equilibrium.
            for _ in 0..(6 * FS as usize) {
                m.step(dt, s);
            }
            let k = m.permeability(s);
            let expected_c = m.steady_state_cleft(k);
            let expected_q = m.steady_state_pool(k);
            assert!(
                (m.c - expected_c).abs() / expected_c < 1e-6,
                "s = {s}: cleft {} vs {expected_c}",
                m.c
            );
            assert!(
                (m.q - expected_q).abs() / expected_q < 1e-6,
                "s = {s}: pool {} vs {expected_q}",
                m.q
            );
            // The invariant the three equations imply at equilibrium: y(M - q) == l·c.
            let lhs = m.y * (m.m - m.q);
            let rhs = m.l * m.c;
            assert!(
                (lhs - rhs).abs() / rhs < 1e-6,
                "s = {s}: replenishment {lhs} does not balance loss {rhs}"
            );
        }
    }

    /// The published parameter set implies a spontaneous rate of 64.77 spikes per second and a
    /// saturating steady rate of 101. Both numbers are typed here from hand arithmetic on the
    /// constants printed in the struct doc, so mutating any of `A`, `B`, `g`, `y`, `l`, `r` or `h`
    /// breaks this test even though the closed form and the integrator would still agree.
    #[test]
    fn the_meddis_constants_imply_the_rates_this_module_documents() {
        let m = Meddis::default();
        assert!(
            (m.permeability(0.0) - 10000.0 / 305.0).abs() < 1e-9,
            "k(0) = {}",
            m.permeability(0.0)
        );
        assert!(
            (m.spontaneous_rate() - 64.768).abs() < 0.02,
            "spontaneous rate was {}",
            m.spontaneous_rate()
        );
        assert!(
            (m.max_steady_rate() - 101.0).abs() < 1e-9,
            "saturating rate was {}",
            m.max_steady_rate()
        );
        // The k -> infinity limit of the closed form really is h·y·M/l.
        let huge = m.h * m.steady_state_cleft(1e12);
        assert!(
            (huge - m.max_steady_rate()).abs() / m.max_steady_rate() < 1e-6,
            "the limit was {huge}"
        );
        // Permeability saturates at g, and is monotone.
        assert!(m.permeability(1e12) < m.g);
        assert!((m.permeability(1e12) - m.g).abs() / m.g < 1e-6);
        assert!(m.permeability(10.0) > m.permeability(1.0));
    }

    /// Saturation as a property of the model, not of one number: however loud the stimulus, the
    /// *sustained* rate cannot pass `h·y·M/l`.
    #[test]
    fn the_meddis_sustained_rate_saturates_however_loud_the_sound() {
        let m = Meddis::default();
        let transmitter_cap = m.max_steady_rate();
        let achievable = m.max_achievable_rate();
        // The two ceilings are distinct, and the achievable one is strictly the lower.
        assert!(
            achievable < transmitter_cap,
            "permeability saturation must bind first: {achievable} vs {transmitter_cap}"
        );
        assert!(
            (achievable - 100.082).abs() < 0.01,
            "the achievable ceiling was {achievable}"
        );
        let mut previous = 0.0;
        for &s in &[1.0, 10.0, 100.0, 1e3, 1e6, 1e12] {
            let r = m.steady_state_rate(s);
            assert!(r < achievable, "s = {s} gave {r}, past the ceiling of {achievable}");
            assert!(r > previous, "s = {s} did not raise the rate");
            previous = r;
        }
        assert!(
            (previous - achievable).abs() / achievable < 1e-6,
            "the loudest stimulus reached {previous}, not the ceiling {achievable}"
        );
    }

    /// Adaptation: the model's whole reason for existing. A tone switched on from the silent
    /// steady state must produce a rate that overshoots its own plateau by a large factor and then
    /// decays back to it.
    #[test]
    fn meddis_adapts_overshooting_its_own_steady_state() {
        let dt = 1.0 / FS;
        let s = 200.0;
        let mut m = Meddis::default();
        let plateau = m.steady_state_rate(s);
        let mut peak: f64 = 0.0;
        let mut early = 0.0;
        for i in 0..(FS as usize / 2) {
            let r = m.step(dt, s);
            peak = peak.max(r);
            if i == (FS as usize / 1000) {
                early = r;
            }
        }
        assert!(
            peak > 3.0 * plateau,
            "onset peaked at {peak}, only {}x the plateau {plateau}",
            peak / plateau
        );
        assert!(early > plateau, "1 ms in, the rate was already at plateau");
        let settled = m.h * m.c;
        assert!(
            (settled - plateau).abs() / plateau < 1e-3,
            "after 500 ms the rate was {settled}, not the plateau {plateau}"
        );
    }

    #[test]
    fn meddis_refuses_a_step_its_integrator_cannot_hold() {
        let mut m = Meddis::default();
        let bound = m.stable_dt_bound();
        assert!(
            (bound - 1.0 / (5.05 + 2500.0 + 6580.0 + 66.31 + 2000.0)).abs() < 1e-15,
            "bound was {bound}"
        );
        assert!(matches!(
            m.step_checked(1.0 / 8000.0, 0.0),
            Err(CochleaError::TimeStep { .. })
        ));
        assert!(m.step_checked(1.0 / 48000.0, 0.0).is_ok());
        assert!(matches!(
            m.step_checked(-1.0, 0.0),
            Err(CochleaError::NotPositive { .. })
        ));
        assert!(matches!(
            m.step_checked(1e-5, f64::NAN),
            Err(CochleaError::NotFinite { .. })
        ));
        // And the filterbank refuses to be built at a rate the hair cell cannot be stepped at.
        assert!(matches!(
            Filterbank::erb_bank(
                8000.0,
                200.0,
                3000.0,
                8,
                Transduction::Meddis(Meddis::default())
            ),
            Err(CochleaError::TimeStep { .. })
        ));
    }

    /// A probability above one is not a probability. The refusal names the condition rather than
    /// clamping it into a "fires every tick" saturation.
    #[test]
    fn meddis_refuses_a_spike_probability_it_cannot_honour() {
        let m = Meddis::default();
        // At the spontaneous rate of ~65 Hz, a 20 us step is fine and a 1 s step is not.
        assert!(m.spike_probability(1.0 / 48000.0).is_some());
        assert!(m.spike_probability(1.0).is_none());
        assert!(m.spike_probability(f64::NAN).is_none());
        let p = m.spike_probability(1e-3).expect("a valid step");
        assert!(
            (p - m.h * m.c * 1e-3).abs() < 1e-15,
            "probability was {p}, not h·c·dt"
        );
        // Drawing is deterministic given the seed.
        let mut r1 = Rng::new(11);
        let mut r2 = Rng::new(11);
        for _ in 0..500 {
            assert_eq!(m.draw(&mut r1, 1e-4), m.draw(&mut r2, 1e-4));
        }
        assert!(m.draw(&mut r1, 10.0).is_none());
    }

    /// The draw really is Bernoulli at the stated probability, not a stuck value.
    #[test]
    fn meddis_draws_at_the_rate_it_advertises() {
        let m = Meddis::default();
        let dt = 1e-4;
        let p = m.spike_probability(dt).expect("a valid step");
        let mut rng = Rng::new(2026);
        let n = 200_000;
        let fired = (0..n)
            .filter(|_| m.draw(&mut rng, dt).expect("a valid step"))
            .count();
        let observed = fired as f64 / n as f64;
        // Three sigma of a binomial with this n is about 0.0017.
        assert!(
            (observed - p).abs() < 0.002,
            "drew {observed}, advertised {p}"
        );
    }

    #[test]
    fn meddis_resets_to_its_silent_equilibrium_and_stays_there() {
        let dt = 1.0 / FS;
        let mut m = Meddis::default();
        let start = m.h * m.c;
        for _ in 0..(FS as usize / 10) {
            m.step(dt, 0.0);
        }
        let end = m.h * m.c;
        assert!(
            (end - start).abs() / start < 1e-9,
            "silence moved the rate from {start} to {end}"
        );
        assert!(
            (start - m.spontaneous_rate()).abs() / start < 1e-12,
            "reset did not land on the spontaneous rate"
        );
    }

    // -------------------------------------------------------------------------------------
    // Onset and offset
    // -------------------------------------------------------------------------------------

    /// The step response of a fast-minus-slow difference is `e^{-t/τ_s} − e^{-t/τ_f}` **exactly**
    /// at every sample, because both stages integrate by exponential Euler. So the peak time and
    /// height are checkable against the differentiated closed form, not against a recording.
    #[test]
    fn the_onset_step_response_matches_its_two_exponential_closed_form() {
        let dt = 1.0 / FS;
        let o = Onset::default();
        let mut d = Onset::default();
        let n = (0.2 * FS) as usize;
        let mut trace = Vec::with_capacity(n);
        for _ in 0..n {
            trace.push(d.step(dt, 1.0));
        }
        // Sample-by-sample against the closed form.
        for (i, &v) in trace.iter().enumerate() {
            let t = (i + 1) as f64 * dt;
            let expected = (-t / o.tau_slow).exp() - (-t / o.tau_fast).exp();
            assert!(
                (v - expected).abs() < 1e-12,
                "sample {i}: {v} vs closed form {expected}"
            );
        }
        // The peak lands where the derivative says it does.
        let peak_i = argmax(&trace);
        let peak_t = (peak_i + 1) as f64 * dt;
        assert!(
            (peak_t - o.step_peak_time()).abs() <= dt,
            "peak at {peak_t} s, formula says {} s",
            o.step_peak_time()
        );
        // The residual here is the curvature of the peak over the half sample between the grid
        // and the true maximum, not any error in either the filter or the formula; the
        // sample-by-sample comparison above is the exact one, at 1e-12.
        assert!(
            (trace[peak_i] - o.step_peak_height()).abs() < 1e-5,
            "peak height {} vs {}",
            trace[peak_i],
            o.step_peak_height()
        );
        // The documented defaults: 3.15 ms, 81 % of the step.
        assert!(
            (o.step_peak_time() - 3.153e-3).abs() < 1e-5,
            "peak time {}",
            o.step_peak_time()
        );
        assert!(
            (o.step_peak_height() - 0.8115).abs() < 1e-3,
            "peak height {}",
            o.step_peak_height()
        );
    }

    /// An offset is a negative onset, and the detector must be symmetric about it: the response to
    /// a step down from 1 to 0 is the exact negative of the response to a step up.
    #[test]
    fn an_offset_is_the_exact_negative_of_an_onset() {
        let dt = 1.0 / FS;
        let mut up = Onset::default();
        let mut down = Onset::default();
        down.prime(1.0);
        for i in 0..2000 {
            let a = up.step(dt, 1.0);
            let b = down.step(dt, 0.0);
            assert!((a + b).abs() < 1e-14, "sample {i}: {a} and {b} do not cancel");
        }
        // Priming makes a constant input produce exactly nothing.
        let mut steady = Onset::default();
        steady.prime(7.0);
        for _ in 0..1000 {
            assert_eq!(steady.step(dt, 7.0), 0.0);
        }
    }

    #[test]
    fn onset_refuses_a_fast_path_that_is_not_faster() {
        assert!(matches!(
            Onset::new(20e-3, 20e-3),
            Err(CochleaError::OnsetTaus { .. })
        ));
        assert!(matches!(
            Onset::new(30e-3, 20e-3),
            Err(CochleaError::OnsetTaus { .. })
        ));
        assert!(matches!(
            Onset::new(0.0, 20e-3),
            Err(CochleaError::NotPositive { .. })
        ));
        assert!(Onset::new(1e-3, 20e-3).is_ok());
    }

    // -------------------------------------------------------------------------------------
    // (e) Silence, and the spiking front end
    // -------------------------------------------------------------------------------------

    /// **Exactly zero.** Not "few", not "below a threshold": a half-wave front end in silence has
    /// a transduced output of exactly `0.0`, a drive current of exactly `0.0`, and a [`Lif`]
    /// resting below its threshold, so the spike count is an integer and that integer is nought.
    #[test]
    fn silence_produces_exactly_zero_spikes() {
        let mut bank = Filterbank::erb_bank(
            FS,
            100.0,
            8000.0,
            32,
            Transduction::HalfWave(Compression::default()),
        )
        .expect("a valid bank");
        let x = silence(FS, 0.5).expect("a valid buffer");
        let train = bank.spike_train(&x).expect("finite signal");
        assert_eq!(train.len(), 0, "silence produced {} spikes", train.len());
        let gram = bank.cochleagram(&train, x.len() as u64);
        assert!(gram.best_channel().is_none(), "a silent bank has no best frequency");
        assert_eq!(gram.total_sustained_hz(), 0.0);
        // And it is not zero because the bank cannot fire: the same bank on a tone does.
        let loud = tone(FS, 1000.0, 1.0, 0.5).expect("a valid tone");
        assert!(!bank.spike_train(&loud).expect("finite signal").is_empty());
    }

    /// With [`Meddis`] the answer is different, and that is the model being right: a nerve fibre
    /// fires in a quiet room. The sustained rate must match what [`Lif`]'s own closed form
    /// predicts from the spontaneous drive, and the change channels must still be exactly silent.
    #[test]
    fn silence_with_meddis_gives_the_closed_form_spontaneous_rate() {
        let m = Meddis::default();
        let mut bank =
            Filterbank::erb_bank(FS, 300.0, 3000.0, 6, Transduction::Meddis(m)).expect("valid");
        let x = silence(FS, 1.0).expect("a valid buffer");
        let train = bank.spike_train(&x).expect("finite signal");
        let gram = bank.cochleagram(&train, x.len() as u64);

        let current = m.spontaneous_rate() * bank.drive();
        let expected = Lif::default()
            .rate(current)
            .expect("the spontaneous drive is above threshold");
        for (ch, &r) in gram.sustained_hz.iter().enumerate() {
            assert!(
                (r - expected).abs() / expected < 0.02,
                "channel {ch} fired at {r} Hz, closed form says {expected} Hz"
            );
        }
        for (ch, (&on, &off)) in gram
            .onset_hz
            .iter()
            .zip(gram.offset_hz.iter())
            .enumerate()
        {
            assert_eq!(on, 0.0, "channel {ch} reported an onset in silence");
            assert_eq!(off, 0.0, "channel {ch} reported an offset in silence");
        }
    }

    /// Onset channels fire when a sound starts, offset channels when it stops, and neither fires
    /// during the steady middle. The middle window is the assertion that matters: a detector
    /// running on the rectified carrier instead of the envelope passes the first two and fails
    /// this one.
    #[test]
    fn onset_and_offset_channels_report_the_edges_and_not_the_middle() {
        let mut bank = Filterbank::erb_bank(
            FS,
            400.0,
            3000.0,
            10,
            Transduction::HalfWave(Compression::default()),
        )
        .expect("a valid bank");
        let x = tone_burst(FS, 1000.0, 1.0, 0.6, 0.1, 0.3, 2e-3).expect("a valid burst");
        let train = bank.spike_train(&x).expect("finite signal");

        let n = bank.channels();
        let mut on_start = 0;
        let mut on_middle = 0;
        let mut off_before = 0;
        let mut off_after = 0;
        for s in train.spikes() {
            let t = s.t as f64 * bank.dt();
            match bank.decode_source(s.source) {
                Some((ChannelKind::Onset, _)) => {
                    if (0.10..0.17).contains(&t) {
                        on_start += 1;
                    } else if (0.25..0.39).contains(&t) {
                        on_middle += 1;
                    }
                }
                Some((ChannelKind::Offset, _)) => {
                    if t < 0.39 {
                        off_before += 1;
                    } else if (0.40..0.47).contains(&t) {
                        off_after += 1;
                    }
                }
                _ => {}
            }
        }
        assert_eq!(n, 10);
        assert!(on_start > 0, "no onset spikes when the tone began");
        assert_eq!(on_middle, 0, "{on_middle} onset spikes during the steady tone");
        assert!(off_after > 0, "no offset spikes when the tone ended");
        assert_eq!(off_before, 0, "{off_before} offset spikes before the tone ended");
    }

    #[test]
    fn spike_addresses_round_trip_and_stay_inside_the_bank() {
        let bank = Filterbank::erb_bank(
            FS,
            200.0,
            4000.0,
            7,
            Transduction::HalfWave(Compression::Linear),
        )
        .expect("a valid bank");
        let n = bank.channels();
        for kind in [ChannelKind::Sustained, ChannelKind::Onset, ChannelKind::Offset] {
            for ch in 0..n {
                let s = bank.source(kind, ch).expect("an in-range channel");
                assert_eq!(bank.decode_source(s), Some((kind, ch)));
            }
        }
        assert!(bank.source(ChannelKind::Sustained, n).is_none());
        assert!(bank.decode_source(3 * n as u32).is_none());
        // A cochleagram ignores foreign addresses rather than folding them into channel zero.
        let alien = Train::from_spikes(vec![Spike {
            t: 0,
            source: 3 * n as u32 + 5,
        }]);
        let gram = bank.cochleagram(&alien, 100);
        assert_eq!(gram.total_sustained_hz(), 0.0);
    }

    #[test]
    fn the_filterbank_refuses_what_it_cannot_run() {
        let mut bank = Filterbank::erb_bank(
            FS,
            200.0,
            4000.0,
            4,
            Transduction::HalfWave(Compression::default()),
        )
        .expect("a valid bank");
        assert!(matches!(
            bank.spike_train(&[0.0, 0.1, f64::INFINITY]),
            Err(CochleaError::NotFinite { index: 2, .. })
        ));
        assert!(matches!(
            bank.envelopes(&[f64::NAN]),
            Err(CochleaError::NotFinite { index: 0, .. })
        ));
        assert!(matches!(
            bank.transduce(&[0.0, f64::NAN]),
            Err(CochleaError::NotFinite { index: 1, .. })
        ));
        assert!(matches!(
            Filterbank::from_centre_frequencies(
                FS,
                &[],
                Transduction::HalfWave(Compression::Linear)
            ),
            Err(CochleaError::Channels { n: 0 })
        ));
        assert!(matches!(
            Filterbank::erb_bank(
                FS,
                200.0,
                FS * 0.49,
                4,
                Transduction::HalfWave(Compression::Linear)
            ),
            Err(CochleaError::CentreFrequency { .. })
        ));
        assert!(matches!(
            Filterbank::erb_bank(
                FS,
                200.0,
                4000.0,
                4,
                Transduction::HalfWave(Compression::Power { exponent: 3.0 })
            ),
            Err(CochleaError::Exponent { .. })
        ));
        assert!(
            bank.clone().with_drive(0.0).is_err()
                && bank.clone().with_onset_drive(f64::NAN).is_err()
        );
    }

    /// The whole front end, twice, bit for bit — including the stochastic path, which is seeded.
    #[test]
    fn the_front_end_is_deterministic() {
        let mut rng = Rng::new(7);
        let x = noise_burst(&mut rng, FS, 0.5, 0.3, 0.05, 0.2, 5e-3).expect("a valid burst");
        let mut a = Filterbank::erb_bank(
            FS,
            200.0,
            5000.0,
            12,
            Transduction::Meddis(Meddis::default()),
        )
        .expect("a valid bank");
        let mut b = a.clone();
        assert_eq!(
            a.spike_train(&x).expect("finite"),
            b.spike_train(&x).expect("finite")
        );
        // And the noise itself reproduces from its seed.
        let mut rng2 = Rng::new(7);
        let y = noise_burst(&mut rng2, FS, 0.5, 0.3, 0.05, 0.2, 5e-3).expect("a valid burst");
        assert_eq!(x, y);
    }

    /// Resetting must really return the bank to its initial state: running the same signal twice
    /// through one bank gives the same train as running it once through two.
    #[test]
    fn a_bank_resets_between_runs() {
        let mut bank = Filterbank::erb_bank(
            FS,
            300.0,
            3000.0,
            8,
            Transduction::HalfWave(Compression::default()),
        )
        .expect("a valid bank");
        let x = tone(FS, 900.0, 0.8, 0.15).expect("a valid tone");
        let first = bank.spike_train(&x).expect("finite");
        let second = bank.spike_train(&x).expect("finite");
        assert_eq!(first, second);
    }

    // -------------------------------------------------------------------------------------
    // Signals
    // -------------------------------------------------------------------------------------

    /// A one-bin transform recovers a tone's amplitude exactly when the buffer spans a whole
    /// number of cycles, and reports essentially nothing at a neighbouring bin. The second half is
    /// what makes the first half a measurement rather than a coincidence.
    #[test]
    fn a_single_bin_transform_recovers_a_tone_and_rejects_its_neighbour() {
        let x = tone(FS, 1000.0, 0.7, 1.0).expect("a valid tone");
        let here = bin_amplitude(&x, FS, 1000.0).expect("a valid buffer");
        assert!((here - 0.7).abs() < 1e-10, "recovered {here}, not 0.7");
        let there = bin_amplitude(&x, FS, 2000.0).expect("a valid buffer");
        assert!(there < 1e-10, "a neighbouring bin held {there}");
        assert!(matches!(
            bin_amplitude(&[], FS, 1000.0),
            Err(CochleaError::LengthMismatch { .. })
        ));
        assert!(matches!(
            bin_amplitude(&[f64::NAN], FS, 1000.0),
            Err(CochleaError::NotFinite { .. })
        ));
    }

    /// A sinusoid's root-mean-square level is its amplitude over root two, exactly, over a whole
    /// number of cycles.
    #[test]
    fn a_tone_has_the_root_mean_square_of_a_sinusoid() {
        let x = tone(FS, 1200.0, 1.5, 0.5).expect("a valid tone");
        let r = rms(&x).expect("a non-empty buffer");
        assert!(
            (r - 1.5 / 2f64.sqrt()).abs() < 1e-9,
            "rms was {r}, not A/sqrt(2)"
        );
        assert!(matches!(rms(&[]), Err(CochleaError::LengthMismatch { .. })));
    }

    /// The generated chirp is checked against the integral of its own instantaneous frequency: a
    /// waveform `sin(φ(t))` with `φ` rising monotonically from 0 to `Φ` crosses zero `⌊Φ/π⌋` times.
    #[test]
    fn a_chirp_accumulates_the_phase_its_formula_predicts() {
        for sweep in [Sweep::Linear, Sweep::Exponential] {
            let (f0, f1, secs) = (200.0, 4000.0, 0.7);
            let x = chirp(FS, f0, f1, 1.0, secs, sweep).expect("a valid chirp");
            let phi = chirp_total_phase(f0, f1, secs, sweep).expect("valid parameters");
            let expected = (phi / PI).floor();
            let crossings = x
                .windows(2)
                .filter(|w| (w[0] < 0.0) != (w[1] < 0.0))
                .count() as f64;
            assert!(
                (crossings - expected).abs() <= 2.0,
                "{sweep:?}: counted {crossings} zero crossings, phase predicts {expected}"
            );
        }
        // The two sweeps really are different: the exponential one spends longer low, so it
        // accumulates strictly less phase between the same endpoints.
        let lin = chirp_total_phase(200.0, 4000.0, 1.0, Sweep::Linear).expect("valid");
        let exp = chirp_total_phase(200.0, 4000.0, 1.0, Sweep::Exponential).expect("valid");
        assert!(exp < lin, "exponential {exp} was not below linear {lin}");
        // A degenerate sweep is a tone.
        let flat = chirp_total_phase(440.0, 440.0, 1.0, Sweep::Exponential).expect("valid");
        assert!((flat - 2.0 * PI * 440.0).abs() < 1e-6, "flat sweep gave {flat}");
    }

    #[test]
    fn a_click_is_one_nonzero_sample_where_it_was_asked_for() {
        let x = click(FS, 2.0, 0.1, 0.05).expect("a valid click");
        assert_eq!(x.len(), 4800);
        let nonzero: Vec<usize> = x
            .iter()
            .enumerate()
            .filter(|&(_, &v)| v != 0.0)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(nonzero, vec![2400]);
        assert_eq!(x[2400], 2.0);
        assert!(matches!(
            click(FS, 1.0, 0.1, 0.2),
            Err(CochleaError::Window { .. })
        ));
        assert!(matches!(
            click(FS, 1.0, 0.1, -0.01),
            Err(CochleaError::Window { .. })
        ));
    }

    /// Uniform draws on `[-a, a]` have variance `a²/3`, so the burst's root-mean-square level is
    /// `a/√3`. Outside the gate it is exactly zero, which is the part a windowing bug would break.
    #[test]
    fn a_noise_burst_has_the_level_and_the_silence_it_claims() {
        let mut rng = Rng::new(31);
        let x = noise_burst(&mut rng, FS, 1.0, 1.0, 0.25, 0.5, 0.0).expect("a valid burst");
        assert_eq!(x.len(), 48_000);
        for (i, &v) in x.iter().enumerate() {
            let t = i as f64 / FS;
            if t < 0.25 || t >= 0.75 {
                assert_eq!(v, 0.0, "sample {i} at {t} s was outside the gate but nonzero");
            }
        }
        let inside = &x[(0.25 * FS) as usize..(0.75 * FS) as usize];
        let r = rms(inside).expect("a non-empty window");
        assert!(
            (r - 1.0 / 3f64.sqrt()).abs() < 0.01,
            "burst rms was {r}, uniform says {}",
            1.0 / 3f64.sqrt()
        );
        // ⛔ Bipolar, which the rms cannot see: uniform on [0, a] has E[x²] = a²/3 exactly like
        // uniform on [-a, a]. A DC-offset burst would put a step into every gammatone.
        let mean = inside.iter().sum::<f64>() / inside.len() as f64;
        assert!(mean.abs() < 0.02, "burst mean {mean}; the doc says [-amplitude, amplitude]");
        let min = inside.iter().cloned().fold(f64::INFINITY, f64::min);
        assert!(min < -0.9, "burst minimum {min}; the samples never went negative");
        assert!(matches!(
            noise_burst(&mut rng, FS, 1.0, 0.2, 0.15, 0.1, 0.0),
            Err(CochleaError::Window { .. })
        ));
    }

    /// A raised-cosine ramp must actually ramp: the gate rises smoothly rather than stepping.
    #[test]
    fn a_ramped_burst_rises_smoothly() {
        let x = tone_burst(FS, 1000.0, 1.0, 0.1, 0.02, 0.05, 5e-3).expect("a valid burst");
        let env: Vec<f64> = x.iter().map(|v| v.abs()).collect();
        // One carrier period of probe window: long enough that the carrier reaches 99.8 % of its
        // peak inside it, short enough that the 5 ms gate only moves a fifth of the way through.
        let period = (FS / 1000.0) as usize;
        let peak_near = |i: usize| env[i..i + period].iter().cloned().fold(0.0f64, f64::max);
        let start = (0.020 * FS) as usize; // gate opens here
        let mid = (0.022 * FS) as usize; // 40 % through a 5 ms ramp
        let full = (0.030 * FS) as usize; // 5 ms past the end of the ramp
        assert!(
            peak_near(start) < 0.15,
            "the gate opened too fast: {} in the first period",
            peak_near(start)
        );
        assert!(
            (0.30..0.70).contains(&peak_near(mid)),
            "mid-ramp amplitude was {}, expected the raised cosine's 0.35-0.65",
            peak_near(mid)
        );
        assert!(
            peak_near(full) > 0.98,
            "the gate never fully opened: {}",
            peak_near(full)
        );
        // A zero ramp really is a hard gate: full amplitude within one period of the onset.
        let hard = tone_burst(FS, 1000.0, 1.0, 0.1, 0.02, 0.05, 0.0).expect("a valid burst");
        let hard_peak = hard[start..start + period]
            .iter()
            .map(|v| v.abs())
            .fold(0.0f64, f64::max);
        assert!(hard_peak > 0.98, "a hard gate opened at only {hard_peak}");
        assert!(matches!(
            tone_burst(FS, 1000.0, 1.0, 0.1, 0.09, 0.05, 0.0),
            Err(CochleaError::Window { .. })
        ));
    }

    /// Two tones are present at their stated amplitudes, which is what makes a masking experiment
    /// interpretable at all.
    #[test]
    fn a_two_tone_signal_carries_both_components() {
        let x = two_tone(FS, 1000.0, 0.3, 1200.0, 0.9, 1.0).expect("a valid pair");
        let probe = bin_amplitude(&x, FS, 1000.0).expect("a valid buffer");
        let masker = bin_amplitude(&x, FS, 1200.0).expect("a valid buffer");
        assert!((probe - 0.3).abs() < 1e-9, "probe came out at {probe}");
        assert!((masker - 0.9).abs() < 1e-9, "masker came out at {masker}");
    }

    /// Energetic masking in a linear bank: a loud near neighbour takes the place code away from
    /// the probe. Stated for what it is in the module doc — this is not the suppressive masking a
    /// real cochlea shows.
    #[test]
    fn a_loud_masker_takes_the_place_code_from_the_probe() {
        let mut bank = Filterbank::erb_bank(
            FS,
            300.0,
            4000.0,
            24,
            Transduction::HalfWave(Compression::default()),
        )
        .expect("a valid bank");
        let secs = 0.3;
        let alone = tone(FS, 1000.0, 0.2, secs).expect("a valid tone");
        let train = bank.spike_train(&alone).expect("finite");
        let best_alone = bank
            .cochleagram(&train, alone.len() as u64)
            .best_frequency()
            .expect("the probe fired");
        assert!(
            (best_alone - 1000.0).abs() < 130.0,
            "the probe alone peaked at {best_alone} Hz"
        );

        let masked = two_tone(FS, 1000.0, 0.2, 1800.0, 1.0, secs).expect("a valid pair");
        let train = bank.spike_train(&masked).expect("finite");
        let best_masked = bank
            .cochleagram(&train, masked.len() as u64)
            .best_frequency()
            .expect("the masker fired");
        assert!(
            (best_masked - 1800.0).abs() < 220.0,
            "with the masker present the peak was at {best_masked} Hz, not near 1800"
        );
        assert!(
            best_masked > best_alone,
            "the masker did not move the place code"
        );
    }

    #[test]
    fn the_signal_generators_refuse_out_of_band_and_non_finite_arguments() {
        assert!(matches!(
            tone(FS, FS / 2.0, 1.0, 0.1),
            Err(CochleaError::CentreFrequency { .. })
        ));
        assert!(matches!(
            tone(FS, 0.0, 1.0, 0.1),
            Err(CochleaError::CentreFrequency { .. })
        ));
        assert!(matches!(
            tone(FS, 1000.0, f64::NAN, 0.1),
            Err(CochleaError::NotFinite { .. })
        ));
        assert!(matches!(
            tone(FS, 1000.0, 1.0, 0.0),
            Err(CochleaError::NotPositive { .. })
        ));
        assert!(matches!(
            tone(FS, 1000.0, 1.0, 1e-9),
            Err(CochleaError::NotPositive { .. })
        ));
        assert!(matches!(
            chirp(FS, 100.0, FS, 1.0, 0.1, Sweep::Linear),
            Err(CochleaError::CentreFrequency { .. })
        ));
        assert!(matches!(
            silence(0.0, 1.0),
            Err(CochleaError::NotPositive { .. })
        ));
        assert!(matches!(
            chirp_total_phase(0.0, 100.0, 1.0, Sweep::Linear),
            Err(CochleaError::NotPositive { .. })
        ));
    }

    /// Every error variant prints something that names the offending value, so a refusal read from
    /// a log says what to change.
    #[test]
    fn every_refusal_says_what_was_wrong() {
        let cases: Vec<CochleaError> = vec![
            CochleaError::NotFinite {
                what: "sample rate",
                index: 3,
            },
            CochleaError::NotPositive {
                what: "duration",
                value: -1.0,
            },
            CochleaError::CentreFrequency {
                f_c: 30000.0,
                fs: 48000.0,
                limit: 21600.0,
            },
            CochleaError::Band {
                lo: 100.0,
                hi: 50.0,
            },
            CochleaError::Channels { n: 0 },
            CochleaError::Order { order: 99 },
            CochleaError::Exponent { exponent: 2.0 },
            CochleaError::OnsetTaus {
                fast: 5e-3,
                slow: 1e-3,
            },
            CochleaError::TimeStep {
                dt: 1e-3,
                bound: 9e-5,
            },
            CochleaError::Window {
                onset: 0.9,
                duration: 0.5,
                total: 1.0,
            },
            CochleaError::LengthMismatch {
                expected: 4,
                got: 7,
            },
        ];
        for e in cases {
            let s = e.to_string();
            assert!(s.len() > 12, "{e:?} printed only {s:?}");
            assert!(
                s.chars().any(|c| c.is_ascii_digit()),
                "{e:?} printed {s:?} with no value in it"
            );
        }
    }

    /// **The property the module doc claims and a spike count does not prove.** Under a steady
    /// tone the change signal must fall to nothing — because the envelope is steady even though
    /// the carrier is not. A detector wired to the rectified carrier instead ripples at the
    /// carrier rate forever, worst at low frequency where a 1 ms smoother barely attenuates it, so
    /// this is checked at 300 Hz where that failure is largest.
    #[test]
    fn the_change_signal_dies_under_a_steady_tone() {
        let mut bank = Filterbank::erb_bank(
            FS,
            200.0,
            2000.0,
            8,
            Transduction::HalfWave(Compression::default()),
        )
        .expect("a valid bank");
        let x = tone_burst(FS, 300.0, 1.0, 0.5, 0.05, 0.35, 2e-3).expect("a valid burst");
        let d = bank.onset_signal(&x).expect("finite signal");
        let peak_in = |trace: &[f64], from: f64, to: f64| -> f64 {
            let (a, b) = ((from * FS) as usize, (to * FS) as usize);
            trace[a..b].iter().fold(0.0f64, |m, v| m.max(v.abs()))
        };
        // Normalised against the BANK's onset response, not each channel's own. A channel 30 dB
        // off band has an envelope that is not perfectly ripple-free: the complex gammatone drops
        // the negative-frequency image, and far from f_c that image is only 6 dB down rather than
        // 100, so the recovered envelope ripples at twice the stimulus frequency by a few percent
        // OF ITS OWN TINY VALUE. Measured here, under a 300 Hz tone: the worst channel is the one
        // at 2000 Hz, which swings 3.5 % of its own onset response and 0.24 % of the bank's; the
        // 1532 Hz channel swings 2.3 % and 0.19 %. That is a property of the complex
        // gammatone, stated rather than tuned around, and it leaves an eightfold margin under
        // the bound below — a margin a carrier-driven detector overruns by two orders of
        // magnitude, which is what the mutation check confirmed.
        let best = d
            .iter()
            .map(|t| peak_in(t, 0.05, 0.15))
            .fold(0.0f64, f64::max);
        assert!(best > 1e-3, "no channel responded to the onset at all");
        for (ch, trace) in d.iter().enumerate() {
            let in_middle = peak_in(trace, 0.25, 0.39);
            assert!(
                in_middle < 0.02 * best,
                "channel {ch} at {} Hz still swung {in_middle} mid-tone against the bank's \
                 {best} at onset: the detector is following the carrier, not the envelope",
                bank.centre_frequencies()[ch]
            );
        }
        let falling = d
            .iter()
            .map(|t| t[(0.40 * FS) as usize..(0.46 * FS) as usize].iter().fold(0.0f64, |m, v| m.min(*v)))
            .fold(0.0f64, f64::min);
        assert!(falling < -1e-3, "no channel reported the offset as a negative swing");
    }

    /// An impulse arrives whole on the first sample and entirely in the real part: `y[0] = gain`
    /// with zero imaginary component, because no stage has had a previous output to rotate yet.
    /// This is what pins [`Gammatone::step`] to the real part rather than the quadrature one — a
    /// swap no amplitude measurement can see, since the two have identical magnitude spectra.
    #[test]
    fn an_impulse_arrives_whole_and_entirely_real_on_the_first_sample() {
        for &f_c in &[250.0, 1000.0, 4000.0] {
            let mut g = Gammatone::new(f_c, FS).expect("valid channel");
            let r = (-g.decay_rate() / FS).exp();
            let expected = 2.0 * (1.0 - r).powi(4);
            let y0 = g.step(1.0);
            assert!(
                (y0 - expected).abs() < 1e-18,
                "first sample was {y0}, gain says {expected}"
            );
            assert!(
                (g.envelope() - expected).abs() < 1e-18,
                "the envelope disagreed with the real part at t = 0, so the output is not the \
                 real part of the cascade"
            );
        }
    }

    /// The output carries the input's carrier, not some other frequency: a steady tone through a
    /// channel crosses zero at exactly twice the tone frequency.
    #[test]
    fn the_filtered_output_keeps_the_input_carrier() {
        let f_c = 1000.0;
        let f = 1100.0;
        let secs = 0.25;
        let x = tone(FS, f, 1.0, secs).expect("a valid tone");
        let mut g = Gammatone::new(f_c, FS).expect("valid channel");
        let y = g.filter(&x).expect("finite signal");
        let settled = &y[(0.05 * FS) as usize..];
        let crossings = settled
            .windows(2)
            .filter(|w| (w[0] < 0.0) != (w[1] < 0.0))
            .count() as f64;
        let expected = 2.0 * f * (settled.len() as f64 / FS);
        assert!(
            (crossings - expected).abs() <= 2.0,
            "counted {crossings} zero crossings, the {f} Hz carrier predicts {expected}"
        );
    }

    #[test]
    fn a_gammatone_resets_between_buffers() {
        let x = tone(FS, 900.0, 1.0, 0.05).expect("a valid tone");
        let mut g = Gammatone::new(1000.0, FS).expect("valid channel");
        let first = g.filter(&x).expect("finite");
        let second = g.filter(&x).expect("finite");
        assert_eq!(first, second, "the filter carried state across buffers");
        let e1 = g.envelopes(&x).expect("finite");
        let e2 = g.envelopes(&x).expect("finite");
        assert_eq!(e1, e2);
    }

    /// Half-wave, not full-wave: with linear compression the transduced output of a channel driven
    /// at its own centre frequency is **exactly zero for half of every cycle**. A full-wave
    /// rectifier doubles the mean and leaves the place code untouched, so nothing about a spike
    /// count would notice; the zero fraction does.
    #[test]
    fn the_transduction_keeps_only_one_half_of_the_cycle() {
        let mut bank = Filterbank::from_centre_frequencies(
            FS,
            &[1000.0],
            Transduction::HalfWave(Compression::Linear),
        )
        .expect("a valid bank");
        let x = tone(FS, 1000.0, 1.0, 0.3).expect("a valid tone");
        let out = bank.transduce(&x).expect("finite");
        let settled = &out[0][(0.05 * FS) as usize..];
        let zeros = settled.iter().filter(|&&v| v == 0.0).count() as f64;
        let fraction = zeros / settled.len() as f64;
        assert!(
            (fraction - 0.5).abs() < 0.01,
            "{fraction} of the transduced samples were zero, half-wave rectification says 0.5"
        );
    }

    /// The gate is a raised cosine, not a straight line. Recovered exactly by combining samples a
    /// quarter of a carrier period apart — at 4 kHz and 48 kHz that is 3 samples, with no
    /// interpolation — and compared against `0.5·(1 − cos(π·s))` at three points along the ramp,
    /// where a linear ramp differs by 0.10.
    #[test]
    fn the_burst_gate_is_the_raised_cosine_it_claims() {
        let (onset, ramp) = (0.020, 5e-3);
        let x = tone_burst(FS, 4000.0, 1.0, 0.1, onset, 0.05, ramp).expect("a valid burst");
        let quarter = (FS / 4000.0 / 4.0) as usize;
        assert_eq!(quarter, 3, "the quarter-period trick needs an integer offset");
        let gate_at = |shape: f64| -> f64 {
            let n = ((onset + shape * ramp) * FS).round() as usize;
            x[n].hypot(x[n + quarter])
        };
        for &shape in &[0.25, 0.5, 0.75] {
            let measured = gate_at(shape);
            let raised_cosine = 0.5 * (1.0 - (PI * shape).cos());
            assert!(
                (measured - raised_cosine).abs() < 0.02,
                "at {shape} of the ramp the gate was {measured}, raised cosine says \
                 {raised_cosine} and a straight line would say {shape}"
            );
        }
    }

    /// The `AGC`, wired into a bank, does what its name says: a ten-fold louder tone does not
    /// produce a ten-fold larger transduced output.
    #[test]
    fn a_bank_with_agc_compresses_a_level_change() {
        let build = |agc: Option<Agc>| {
            Filterbank::erb_bank(
                FS,
                500.0,
                2000.0,
                6,
                Transduction::HalfWave(Compression::Linear),
            )
            .expect("a valid bank")
            .with_agc(agc)
            .expect("a valid loop")
        };
        let quiet = tone(FS, 1000.0, 0.1, 0.3).expect("a valid tone");
        let loud = tone(FS, 1000.0, 1.0, 0.3).expect("a valid tone");
        let peak = |bank: &mut Filterbank, x: &[f64]| -> f64 {
            let out = bank.transduce(x).expect("finite");
            let tail = out[3].len() * 2 / 3;
            out[3][tail..].iter().cloned().fold(0.0f64, f64::max)
        };
        let mut plain = build(None);
        let ratio_plain = peak(&mut plain, &loud) / peak(&mut plain, &quiet);
        assert!(
            (ratio_plain - 10.0).abs() < 0.5,
            "without AGC the ratio was {ratio_plain}, not the linear 10"
        );
        let mut gained = build(Some(Agc::new(20e-3, 20.0).expect("valid loop")));
        let ratio_gained = peak(&mut gained, &loud) / peak(&mut gained, &quiet);
        assert!(
            ratio_gained < 0.7 * ratio_plain,
            "with AGC the ratio was {ratio_gained}, barely below the linear {ratio_plain}"
        );
        assert!(ratio_gained > 1.0, "the AGC inverted the level order");
    }

    /// The tone starts at zero phase, as its doc says and as nothing measured: `.cos()` in place of
    /// `.sin()` was green because rms, a single-bin transform and zero-crossing counts are all
    /// phase-blind. A cosine begins with a click.
    #[test]
    fn a_tone_starts_at_zero_phase_and_rises() {
        let x = tone(FS, 1000.0, 1.0, 0.01).expect("a valid tone");
        assert_eq!(x[0], 0.0, "sample zero is not zero: the tone begins with a step");
        assert!(x[1] > 0.0 && x[1] < 0.2, "sample one is {}, not a small positive value", x[1]);
    }

    /// ⛔ FIVE OF SIX BUILDERS COULD BE REDUCED TO VALIDATE-ONLY NO-OPS with every test green:
    /// `with_onset`, `with_cell` and `with_envelope_compression` were never called; `with_drive`
    /// and `with_onset_drive` were called only to see them refuse. Each is shown here to change
    /// the front end's output in the direction it claims.
    #[test]
    fn every_builder_changes_the_output_it_says_it_does() {
        let build = || {
            Filterbank::erb_bank(FS, 400.0, 3000.0, 6, Transduction::HalfWave(Compression::default()))
                .expect("a valid bank")
        };
        let x = tone_burst(FS, 1000.0, 1.0, 0.4, 0.1, 0.2, 2e-3).expect("a valid burst");
        // (sustained, onset, offset) spike counts.
        let count = |bank: &mut Filterbank| -> (usize, usize, usize) {
            let train = bank.spike_train(&x).expect("finite");
            let (mut s, mut on, mut off) = (0, 0, 0);
            for sp in train.spikes() {
                match bank.decode_source(sp.source) {
                    Some((ChannelKind::Sustained, _)) => s += 1,
                    Some((ChannelKind::Onset, _)) => on += 1,
                    Some((ChannelKind::Offset, _)) => off += 1,
                    None => panic!("a spike with no address"),
                }
            }
            (s, on, off)
        };
        let (base_s, base_on, base_off) = count(&mut build());
        assert!(base_s > 0 && base_on > 0 && base_off > 0, "{base_s} {base_on} {base_off}");

        let half = build().drive() * 0.5;
        let (s, _, _) = count(&mut build().with_drive(half).expect("valid"));
        assert!(s < base_s, "half the drive, sustained {s} against {base_s}");

        let quarter = build().onset_drive() * 0.25;
        let (_, on, off) = count(&mut build().with_onset_drive(quarter).expect("valid"));
        assert!(on < base_on && off < base_off, "a quarter of the onset drive: {on} {off}");

        let stiff = Lif { v_th: Lif::default().v_th + 10e-3, ..Lif::default() };
        let (s, on, _) = count(&mut build().with_cell(stiff));
        assert!(s < base_s && on < base_on, "a 10 mV higher threshold: {s} {on}");

        let fast = build().onset_signal(&x).expect("finite");
        // A slower detector with a SMALLER tau ratio: the step-response peak height depends only
        // on tau_slow / tau_fast (0.81 at 20:1, 0.70 at 10:1), so 5 ms / 100 ms would peak at the
        // same height as the default 1 ms / 20 ms and only later. 5 ms / 50 ms peaks later AND lower.
        let slow_detector = Onset::new(5e-3, 50e-3).expect("valid");
        assert!(slow_detector.step_peak_height() < Onset::default().step_peak_height());
        let mut b = build().with_onset(slow_detector).expect("valid");
        let slow = b.onset_signal(&x).expect("finite");
        let peak = |trace: &[f64]| trace.iter().fold(0.0f64, |m, v| m.max(*v));
        let arg = |trace: &[f64]| argmax(trace);
        assert!(arg(&slow[2]) > arg(&fast[2]), "a slower detector must peak later");
        assert!(peak(&slow[2]) < peak(&fast[2]), "a detector with a smaller tau ratio must peak lower");

        let mut b = build().with_envelope_compression(Compression::Linear).expect("valid");
        let lin = b.onset_signal(&x).expect("finite");
        assert_ne!(lin[2], fast[2], "linear and power-law compression gave the same onset signal");
        // The compression stage is in the onset PATH, not only stored: a power law compresses a
        // unit-amplitude envelope's rise less than linear passes it, so the peaks differ.
        assert!((peak(&lin[2]) - peak(&fast[2])).abs() > 1e-3 * peak(&fast[2]));

        let with_loop = build().with_agc(Some(Agc::new(20e-3, 20.0).expect("valid"))).expect("valid");
        let (s, _, _) = count(&mut { with_loop });
        assert!(s < base_s, "an AGC on a unit tone must reduce the sustained count: {s} vs {base_s}");
    }

    /// ⛔ PUBLIC FIELDS ARE NOT A BACK DOOR PAST THE CONSTRUCTORS. `Onset::tau_fast`, `Agc::tau`
    /// and every Meddis constant are public; each can be set to a value its constructor refuses,
    /// and the first version of every builder took the struct as given. Now each re-validates.
    #[test]
    fn a_builder_refuses_a_struct_its_constructor_would_have_refused() {
        let bank = || {
            Filterbank::erb_bank(FS, 400.0, 3000.0, 6, Transduction::HalfWave(Compression::Linear))
                .expect("a valid bank")
        };
        let o = Onset { tau_fast: 30e-3, tau_slow: 1e-3, ..Onset::default() };
        assert!(matches!(bank().with_onset(o), Err(CochleaError::OnsetTaus { .. })));
        let mut a = Agc::new(20e-3, 20.0).expect("valid");
        a.tau = -1e-3;
        assert!(matches!(bank().with_agc(Some(a)), Err(CochleaError::NotPositive { .. })));
        let m = Meddis { b: -300.0, ..Meddis::default() };
        assert!(matches!(
            Filterbank::erb_bank(FS, 400.0, 3000.0, 6, Transduction::Meddis(m)),
            Err(CochleaError::NotPositive { .. })
        ));
        assert!(Meddis::default().validate().is_ok());
        let dead = Meddis { l: 0.0, ..Meddis::default() };
        assert!(dead.validate().is_err(), "a zero loss rate has an infinite steady-state cleft");
        // And a NaN in the pool is not laundered into a healthy zero.
        let mut m = Meddis { q: f64::NAN, ..Meddis::default() };
        m.step(1e-4, 0.0);
        assert!(m.q.is_nan(), "a NaN pool came back as {}", m.q);
    }

    /// The AGC on the Meddis arm. No test combined `Transduction::Meddis` with a loop, so the AGC
    /// could be deleted from that arm with every test green. A loud tone through the hair cell
    /// with and without the loop: the loop must lower the sustained count.
    #[test]
    fn the_agc_acts_on_the_meddis_arm_too() {
        let build = |agc: Option<Agc>| {
            Filterbank::erb_bank(FS, 500.0, 2000.0, 6, Transduction::Meddis(Meddis::default()))
                .expect("a valid bank")
                .with_agc(agc)
                .expect("a valid loop")
        };
        // Loud in the hair cell's own units: the Meddis permeability half-saturates at B = 300, so
        // a unit-amplitude tone barely leaves the spontaneous rate and an AGC on it changes nothing
        // (measured: 246 sustained spikes with and without). At 300 the loop's gain of
        // 1/(1 + 20·level) pulls the cell off its saturated branch.
        let loud = tone(FS, 1000.0, 300.0, 0.3).expect("a valid tone");
        let sustained = |bank: &mut Filterbank| -> usize {
            let train = bank.spike_train(&loud).expect("finite");
            train
                .spikes()
                .iter()
                .filter(|s| matches!(bank.decode_source(s.source), Some((ChannelKind::Sustained, _))))
                .count()
        };
        let plain = sustained(&mut build(None));
        let gained = sustained(&mut build(Some(Agc::new(20e-3, 20.0).expect("valid"))));
        assert!(plain > 0, "the hair cell was silent under a unit tone");
        assert!(gained < plain, "with the AGC the Meddis arm fired {gained} against {plain} without");
    }

    // -------------------------------------------------------------------------------------
    // ⛔ The mutation backfill. Each test below closes a mutation that SURVIVED the recorded
    // list: the claim was in a doc and in nothing that could fail.
    // -------------------------------------------------------------------------------------

    /// Each transduction's default drive must land its output cell where the cell can still move:
    /// above [`Lif`]'s threshold in silence, and far enough below the refractory ceiling that the
    /// hair cell's own sustained range still shows up in the firing rate.
    ///
    /// ⛔ Three constants had no test. `silence_with_meddis_gives_the_closed_form_spontaneous_rate`
    /// builds its expectation out of `bank.drive()` itself, so both sides of that comparison move
    /// together when [`DEFAULT_DRIVE_MEDDIS`] moves; which of the two scales the Meddis arm picks
    /// is chosen inside `from_centre_frequencies` and was never read back; and
    /// [`DEFAULT_DRIVE_ONSET`] was only ever compared against itself. Measured here with
    /// [`DEFAULT_DRIVE_MEDDIS`]: the spontaneous rate of 64.77 spikes/s drives the cell at
    /// 137.6 Hz and the ceiling plateau of 100.08 spikes/s drives it at 190.6 Hz — a 38 % rise,
    /// against a refractory bound of 500 Hz. A drive a hundred times larger pins the cell against
    /// that bound instead, 488.7 Hz against 492.6 Hz, and every *relative* measurement stays green.
    #[test]
    fn each_transduction_drives_its_cell_into_its_own_working_range() {
        let m = Meddis::default();
        let bank =
            Filterbank::erb_bank(FS, 300.0, 3000.0, 6, Transduction::Meddis(m)).expect("valid");
        assert_eq!(
            bank.drive(),
            DEFAULT_DRIVE_MEDDIS,
            "the Meddis arm was built with the wrong drive scale"
        );
        let half = Filterbank::erb_bank(
            FS,
            300.0,
            3000.0,
            6,
            Transduction::HalfWave(Compression::default()),
        )
        .expect("valid");
        assert_eq!(half.drive(), DEFAULT_DRIVE_HALF_WAVE);

        let cell = Lif::default();
        let ceiling = 1.0 / cell.t_ref;
        let quiet = cell
            .rate(m.spontaneous_rate() * bank.drive())
            .expect("a nerve fibre fires in a quiet room");
        let loudest = cell
            .rate(m.max_achievable_rate() * bank.drive())
            .expect("the ceiling plateau is above threshold");
        assert!(
            loudest < 0.5 * ceiling,
            "the loudest sustained sound drives the cell to {loudest} Hz, past half the \
             refractory ceiling of {ceiling} Hz: the drive has saturated the cell"
        );
        // The hair cell's whole sustained range, 64.77 to 100.08 spikes/s, has to survive the
        // conversion to a current: if it does not, the sustained channel reports the same rate
        // whatever it hears.
        assert!(
            loudest / quiet > 1.2,
            "silence drives the cell at {quiet} Hz and the loudest plateau at {loudest} Hz, a \
             rise of {}x: the drive has left the cell no room to move",
            loudest / quiet
        );

        // The onset path's drive is a different scale, and the doc says which way it differs.
        // Read back off a built bank rather than off the two constants, so this is a measurement
        // of what a default bank runs rather than an assertion the compiler can fold away.
        let build = || {
            Filterbank::erb_bank(
                FS,
                400.0,
                3000.0,
                6,
                Transduction::HalfWave(Compression::default()),
            )
            .expect("a valid bank")
        };
        let defaults = build();
        assert!(
            defaults.onset_drive() > defaults.drive(),
            "a default bank's onset drive is {} against a sustained drive of {}: a difference of \
             two leaky integrators is smaller than the signal it is computed from, so the onset \
             path needs the larger scale",
            defaults.onset_drive(),
            defaults.drive()
        );
        let x = tone_burst(FS, 1000.0, 1.0, 0.4, 0.1, 0.2, 2e-3).expect("a valid burst");
        let onsets = |bank: &mut Filterbank| -> usize {
            let train = bank.spike_train(&x).expect("finite");
            train
                .spikes()
                .iter()
                .filter(|s| matches!(bank.decode_source(s.source), Some((ChannelKind::Onset, _))))
                .count()
        };
        let at_onset_scale = onsets(&mut build());
        let at_sustained_scale = onsets(
            &mut build()
                .with_onset_drive(DEFAULT_DRIVE_HALF_WAVE)
                .expect("valid"),
        );
        // Measured: 23 onset spikes at the onset drive, 7 at the sustained one.
        assert!(
            at_sustained_scale < at_onset_scale,
            "the burst produced {at_sustained_scale} onset spikes at the SUSTAINED drive and \
             {at_onset_scale} at the onset drive: the two scales are the same number"
        );
    }

    /// The `ERB`-rate scale's logarithm needs `4.37·f/1000 + 1 > 0`, which is `f > −228.8` Hz.
    ///
    /// ⛔ The suite's only off-scale probe was −500 Hz, where the argument is −1.185. A guard
    /// loosened to `arg < −1.0` still refuses that one and hands back `Some(NaN)` for every
    /// frequency between the pole and −457.7 Hz — a position on the scale that compares false
    /// against every bound and propagates straight into [`erb_space`].
    #[test]
    fn erb_rate_refuses_every_frequency_below_its_pole() {
        // 4.37·f/1000 + 1 == 0 at f = −1000/4.37 = −228.83 Hz.
        let pole = -1000.0 / 4.37;
        assert!(
            erb_rate(pole).is_none(),
            "at the pole the logarithm's argument is zero, not positive"
        );
        for &f in &[-229.0, -250.0, -300.0, -400.0, -457.0, -500.0, -1e6] {
            assert!(
                erb_rate(f).is_none(),
                "{f} Hz is off the bottom of the scale; got {:?}",
                erb_rate(f)
            );
        }
        // Just above the pole the scale IS defined, and large and negative — negative frequencies
        // are kept so the inverse's round trip can be written at all.
        let e = erb_rate(-228.0).expect("just above the pole");
        assert!(e < -20.0, "E(-228 Hz) came out at {e}");
        let back = erb_rate_to_hz(e).expect("invertible");
        assert!((back + 228.0).abs() < 1e-6, "round trip gave {back}");
    }

    /// [`Gammatone::with_shape`]'s doc says the bandwidth factor is a parameter "rather than a
    /// constant reached for silently".
    ///
    /// ⛔ Every construction in this suite passed [`PATTERSON_B`], so the argument and the constant
    /// were the same number and the pole could be built from either. Here the factor is doubled
    /// and the filter is DRIVEN: `β = 2π·b·ERB(f_c)` doubles, so the impulse-envelope peak must
    /// land at half the latency and the −3 dB width must double. Measured at 1 kHz: the peak moves
    /// from sample 168 (3.500 ms) to sample 83 (1.729 ms), against the continuum's 1.766 ms.
    #[test]
    fn a_channel_built_with_its_own_bandwidth_factor_is_really_that_wide() {
        let f_c = 1000.0;
        let wide_b = 2.0 * PATTERSON_B;
        let narrow = Gammatone::with_shape(f_c, FS, 4, PATTERSON_B).expect("valid channel");
        let mut wide = Gammatone::with_shape(f_c, FS, 4, wide_b).expect("valid channel");
        assert_eq!(wide.b_factor(), wide_b);
        // β is linear in b, and scaling by two is exact in binary floating point.
        assert_eq!(wide.decay_rate(), 2.0 * narrow.decay_rate());

        let n = (0.05 * FS) as usize;
        let mut env = Vec::with_capacity(n);
        for i in 0..n {
            wide.step(if i == 0 { 1.0 } else { 0.0 });
            env.push(wide.envelope());
        }
        let measured = argmax(&env) as f64 / FS;
        assert_eq!(
            measured,
            wide.peak_latency(),
            "the driven envelope peak is not where the discrete formula puts it"
        );
        // Against the continuum, which is computed from `b_factor` and not from the pole. The
        // tolerance is the n−1 samples the discretisation costs, 2.1 % here, as in
        // `a_click_produces_the_predicted_travelling_wave_delay`.
        let continuous = wide.peak_latency_continuous();
        assert!(
            (measured - continuous).abs() / continuous < 0.06,
            "the doubled-bandwidth channel rang for {measured} s, its own continuum says \
             {continuous} s: the pole was built from PATTERSON_B, not from the factor supplied"
        );
        // Same statement in the frequency domain: the width read out of the pole against the width
        // read out of `b_factor`.
        let from_pole = wide.bandwidth_3db().expect("narrow enough to have a width");
        let from_factor = wide.bandwidth_3db_continuous();
        assert!(
            (from_pole - from_factor).abs() / from_factor < 1e-3,
            "the pole says {from_pole} Hz wide, the factor says {from_factor} Hz"
        );
        assert_eq!(from_factor, 2.0 * narrow.bandwidth_3db_continuous());
    }

    /// [`Gammatone::bandwidth_3db`] refuses rather than clamping, and the refusal is a statement
    /// about the filter: it never falls 3 dB anywhere inside the sampled band.
    ///
    /// ⛔ No test here had ever built one that wide, so the domain check could be deleted and the
    /// method would hand back `Some(NaN)` — a bandwidth that prints as a number and compares false
    /// against every bound. Measured: at `b_factor` 200 the pole magnitude is 0.031 and the
    /// response at Nyquist is 0.781 of the peak, above `1/√2`, so there is nothing to report.
    #[test]
    fn a_filter_with_no_half_power_point_refuses_to_report_one() {
        let half_power = 1.0 / 2f64.sqrt();
        let too_wide = Gammatone::with_shape(1000.0, FS, 4, 200.0).expect("valid channel");
        let at_nyquist = too_wide.magnitude_response(FS / 2.0);
        assert!(
            at_nyquist > half_power,
            "the response at Nyquist was {at_nyquist}, below the half-power line: this filter DOES \
             have a width and the test is no longer probing the refusal"
        );
        assert!(
            too_wide.bandwidth_3db().is_none(),
            "reported a width of {:?} for a filter that never falls 3 dB",
            too_wide.bandwidth_3db()
        );
        // Half as wide, and it crosses the half-power line before Nyquist, so it has a width.
        let narrower = Gammatone::with_shape(1000.0, FS, 4, 100.0).expect("valid channel");
        assert!(narrower.magnitude_response(FS / 2.0) < half_power);
        let w = narrower
            .bandwidth_3db()
            .expect("a filter that crosses the half-power line has a width");
        assert!(w.is_finite() && w > 0.0, "the width came out at {w}");
    }

    /// [`Compression::default`]'s doc states the exponent "so a figure made with the default is
    /// reproducible from the documentation alone", and nothing read it back.
    ///
    /// ⛔ 0.4 is the number that turns the 100 dB of input range the whole stage exists for into
    /// 40 dB of output. 0.5 — the other end of the measured basilar-membrane range — makes it 50,
    /// and every test that used the default compared it only against itself.
    #[test]
    fn the_default_compression_is_the_exponent_the_documentation_prints() {
        assert_eq!(Compression::default(), Compression::Power { exponent: 0.4 });
        let c = Compression::default();
        let (lo, hi) = (1e-5f64, 1.0f64);
        let in_db = 20.0 * (hi / lo).log10();
        assert!((in_db - 100.0).abs() < 1e-9, "the probe span was {in_db} dB");
        let out_db = 20.0 * (c.apply(hi) / c.apply(lo)).log10();
        assert!(
            (out_db - 40.0).abs() < 1e-9,
            "the default turned {in_db} dB of input into {out_db} dB of output, not 40"
        );
    }

    /// [`Agc::reset`] must put the loop back where [`Agc::new`] starts it.
    ///
    /// ⛔ `Agc::new` sets the level directly, so the only caller of `reset` inside this module is
    /// [`Filterbank::reset`], and every bank test measures a SETTLED loop whose starting level it
    /// has already forgotten. A reset that left the level at 1.0 would open every run attenuating
    /// by `1/(1 + target)` — a factor of 21 at the loop these tests use — and suppress exactly the
    /// transient an `AGC` exists to pass.
    #[test]
    fn a_gain_loop_returns_to_rest_and_not_to_some_other_level() {
        let dt = 1.0 / FS;
        let mut a = Agc::new(10e-3, 2.0).expect("valid loop");
        for _ in 0..(FS as usize / 10) {
            a.step(dt, 5.0);
        }
        assert!(a.level() > 4.9, "the loop never settled: level {}", a.level());
        a.reset();
        assert_eq!(a.level(), 0.0, "a reset loop's level is {}", a.level());
        assert_eq!(a.gain(), 1.0, "a reset loop attenuates by {}", a.gain());
        // A reset loop is the loop the constructor makes, coefficients and state alike.
        assert_eq!(a, Agc::new(10e-3, 2.0).expect("valid loop"));
        // And the first sample after a reset passes ungained, as it does from the constructor.
        assert_eq!(a.step(dt, 1.0), 1.0);
    }

    /// [`Filterbank::reset`] clears five kinds of state; two of them had no test.
    ///
    /// ⛔ `a_bank_resets_between_runs` runs the half-wave path with no gain loops, so the loops
    /// that reset the hair cells and the `AGC`s could each be reduced to a no-op with every test
    /// green: nothing ran a [`Transduction::Meddis`] bank or an [`Agc`] twice through one
    /// [`Filterbank`]. Both stages carry state across a run — a drained transmitter pool and a
    /// settled level — so an unreset bank's second run is a different recording.
    #[test]
    fn a_bank_resets_every_stage_it_owns_between_runs() {
        let build = || {
            Filterbank::erb_bank(
                FS,
                500.0,
                2000.0,
                5,
                Transduction::Meddis(Meddis::default()),
            )
            .expect("a valid bank")
            .with_agc(Some(Agc::new(20e-3, 20.0).expect("valid loop")))
            .expect("a valid loop")
        };
        // Loud in the hair cell's own units — the permeability half-saturates at B = 300 — so both
        // the transmitter pool and the gain loop really move.
        let loud = tone(FS, 1000.0, 300.0, 0.15).expect("a valid tone");
        let mut bank = build();
        let first = bank.spike_train(&loud).expect("finite");
        let second = bank.spike_train(&loud).expect("finite");
        assert!(!first.is_empty(), "the loud tone produced no spikes at all");
        assert_eq!(
            first, second,
            "the second run through one bank differed from the first"
        );
        let mut fresh = build();
        assert_eq!(
            first,
            fresh.spike_train(&loud).expect("finite"),
            "two runs through one bank differed from one run through two"
        );
        // And the run really does leave those two stages away from rest, so the equality above is
        // a reset being performed rather than a state that never moved.
        let rest = Meddis::default();
        assert!(
            bank.hair.iter().any(|h| (h.q - rest.q).abs() > 1e-6),
            "no hair cell's pool moved during the run"
        );
        assert!(
            bank.agc.iter().any(|a| a.level() > 1e-6),
            "no gain loop's level moved during the run"
        );
    }

    /// The displacement is half-wave rectified **before** it reaches the transduction stage.
    ///
    /// ⛔ That call is invisible on the [`Transduction::HalfWave`] arm, because
    /// [`Compression::apply`] rectifies its own input; only a [`Transduction::Meddis`] bank can
    /// tell, and the one test that drove the hair cell hard enough
    /// (`the_agc_acts_on_the_meddis_arm_too`) compares two banks that move together. Handing the
    /// raw displacement to [`Meddis`] does not merely pass the negative half through: the
    /// permeability law shuts the channels wherever `s + A < 0`, so the fibre falls nearly silent
    /// for half of every cycle. Measured under a 30-unit tone at the channel's own centre
    /// frequency, over the settled last quarter: the rectified path's instantaneous rate never
    /// falls below 31.4 spikes/s, the raw path's falls to 1.2.
    #[test]
    fn the_transduction_stage_sees_the_rectified_displacement_and_not_the_raw_one() {
        let dt = 1.0 / FS;
        let f_c = 1000.0;
        let x = tone(FS, f_c, 30.0, 0.4).expect("a valid tone");
        let mut bank =
            Filterbank::from_centre_frequencies(FS, &[f_c], Transduction::Meddis(Meddis::default()))
                .expect("a valid bank");
        let out = bank.transduce(&x).expect("finite");

        // The documented pipeline, rebuilt from this module's own public pieces in the same order
        // and with the same operations, so this is an equality and not a tolerance.
        let mut g = Gammatone::new(f_c, FS).expect("valid channel");
        let mut cell = Meddis::default();
        let want: Vec<f64> = x
            .iter()
            .map(|&s| cell.step(dt, half_wave(g.step(s))))
            .collect();
        assert_eq!(
            out[0], want,
            "the bank's Meddis arm is not gammatone, half_wave, hair cell"
        );

        // What the rectifier buys, computed here so this test's margin is measured rather than
        // asserted: the same cell fed the raw displacement shuts during the negative half-cycle.
        let mut g_raw = Gammatone::new(f_c, FS).expect("valid channel");
        let mut cell_raw = Meddis::default();
        let raw: Vec<f64> = x.iter().map(|&s| cell_raw.step(dt, g_raw.step(s))).collect();
        let tail = x.len() * 3 / 4;
        let floor_of = |v: &[f64]| v[tail..].iter().cloned().fold(f64::INFINITY, f64::min);
        assert!(
            floor_of(&out[0]) > 10.0 * floor_of(&raw),
            "the rectified path's rate floor is {} spikes/s and the raw path's is {}: the two \
             are not far enough apart for this test to be measuring the rectifier",
            floor_of(&out[0]),
            floor_of(&raw)
        );
    }

    /// `k(s) = g·(s + A)/(s + A + B)` for `s + A > 0` and **zero** below it — the paper's own
    /// clamp, which exists because a large enough inward displacement shuts the transduction
    /// channels entirely.
    ///
    /// ⛔ Nothing in this suite evaluated the law there, because `transduce_one` only ever hands it
    /// a rectified value. Without the clamp a displacement of −10 units returns a permeability of
    /// −33.9 s⁻¹ and a steady rate of 220 spikes/s — past both documented ceilings, from a sound
    /// pushing the stereocilia the wrong way.
    #[test]
    fn the_permeability_law_shuts_the_channels_at_large_inward_displacement() {
        let m = Meddis::default();
        for &s in &[-5.0, -5.5, -10.0, -300.0, -1e9] {
            assert_eq!(m.permeability(s), 0.0, "k({s}) was {}", m.permeability(s));
            assert_eq!(
                m.steady_state_rate(s),
                0.0,
                "a shut channel has no steady rate, got {}",
                m.steady_state_rate(s)
            );
        }
        // The clamp sits exactly at `s + A == 0`, and just above it the law is small and positive.
        let just_open = m.permeability(-4.9);
        assert!(
            just_open > 0.0 && just_open < 1.0,
            "k(-4.9) was {just_open}, not the small positive value the law gives at s + A = 0.1"
        );
        // Non-negative and monotone across the clamp: the law has no inward-opening branch.
        let mut previous = -1.0;
        for i in 0..400 {
            let s = -20.0 + i as f64 * 0.1;
            let k = m.permeability(s);
            assert!(k >= 0.0, "k({s}) = {k}: the channels opened inward");
            assert!(k >= previous, "k is not monotone at s = {s}: {k} after {previous}");
            previous = k;
        }
    }

    /// `M` is the pool maximum, and it appears in the equilibrium cleft as a factor.
    ///
    /// ⛔ It is `1.0` in the published parameter set and in every fixture here, and a factor of one
    /// is the identity — so `k·y·M / (y(l+r) + k·l)` could drop its `M` and the integrator would
    /// still agree with the closed form. `M` is the only forcing term in the three equations, so
    /// the whole system is linear in it: doubling the pool maximum must double the equilibrium
    /// cleft, pool and reprocessing store.
    #[test]
    fn the_steady_state_closed_forms_carry_the_pool_maximum() {
        let dt = 1.0 / FS;
        let s = 50.0;
        let one = Meddis::default();
        let mut big = Meddis {
            m: 2.0,
            ..Meddis::default()
        };
        big.reset();
        big.validate().expect("a doubled pool maximum is a valid cell");
        // Six seconds, as `meddis_settles_on_its_closed_form_steady_state` uses: the slowest mode
        // relaxes on the order of 1/y.
        for _ in 0..(6 * FS as usize) {
            big.step(dt, s);
        }
        let k = big.permeability(s);
        let want_c = big.steady_state_cleft(k);
        let want_q = big.steady_state_pool(k);
        assert!(
            (big.c - want_c).abs() / want_c < 1e-6,
            "with M = 2 the integrated cleft was {} and the closed form says {want_c}",
            big.c
        );
        assert!(
            (big.q - want_q).abs() / want_q < 1e-6,
            "with M = 2 the integrated pool was {} and the closed form says {want_q}",
            big.q
        );
        // And the closed form is linear in M. Scaling by two is exact in binary floating point and
        // the two denominators are the same expression, so these are equalities.
        assert_eq!(big.steady_state_cleft(k), 2.0 * one.steady_state_cleft(k));
        assert_eq!(big.max_steady_rate(), 2.0 * one.max_steady_rate());
    }

    /// Forward Euler evaluates all three derivatives at the state the step **starts** from.
    ///
    /// ⛔ Writing the pool back before the cleft's derivative reads it turns the method into a
    /// Gauss-Seidel sweep, and that is invisible to everything here: all three derivatives are
    /// zero at equilibrium, so `meddis_settles_on_its_closed_form_steady_state` cannot see it, and
    /// the transient it does bend has no closed form that any test pins. Checked as the arithmetic
    /// it is — one step, from a known state, against the three updates written out by hand in the
    /// same order, so these are equalities.
    #[test]
    fn the_hair_cell_integrates_by_forward_euler_from_one_state() {
        let dt = 1.0 / FS;
        for &s in &[0.0, 200.0, 1e4] {
            let mut m = Meddis::default();
            let (q0, c0, w0) = (m.q, m.c, m.w);
            let k = m.permeability(s);
            let dq = m.y * (m.m - q0) + m.x * w0 - k * q0;
            let dc = k * q0 - m.l * c0 - m.r * c0;
            let dw = m.r * c0 - m.x * w0;
            let rate = m.step(dt, s);
            assert_eq!(m.q, q0 + dq * dt, "s = {s}: the pool");
            assert_eq!(m.c, c0 + dc * dt, "s = {s}: the cleft");
            assert_eq!(m.w, w0 + dw * dt, "s = {s}: the reprocessing store");
            assert_eq!(rate, m.h * m.c, "s = {s}: the rate returned");
        }
        // And the step really moves, so an update applied twice is a different number: from the
        // silent equilibrium under a 200-unit stimulus the pool's derivative measures
        // −279.5 quanta per second.
        let m = Meddis::default();
        let k = m.permeability(200.0);
        let dq = m.y * (m.m - m.q) + m.x * m.w - k * m.q;
        assert!(
            dq.abs() > 1.0,
            "the probe state was already at equilibrium: dq = {dq}"
        );
    }

    /// [`Meddis::validate`]'s doc says the state variables "may be zero but not negative".
    ///
    /// ⛔ Every fixture handed it a state built by [`Meddis::reset`], which is strictly positive,
    /// so the state loop's comparison was never driven from either side. A guard loosened to
    /// `v < -1.0` still refuses a pool of −2 and accepts one of −1e−6, and a negative cleft is a
    /// negative firing rate that every rectifier downstream hides.
    #[test]
    fn a_hair_cell_with_a_negative_pool_is_refused() {
        for m in [
            Meddis {
                q: -1e-6,
                ..Meddis::default()
            },
            Meddis {
                c: -1e-9,
                ..Meddis::default()
            },
            Meddis {
                w: -0.5,
                ..Meddis::default()
            },
        ] {
            assert!(
                matches!(m.validate(), Err(CochleaError::NotPositive { .. })),
                "a negative state variable was accepted: {m:?}"
            );
            assert!(
                matches!(
                    Filterbank::erb_bank(FS, 400.0, 3000.0, 4, Transduction::Meddis(m)),
                    Err(CochleaError::NotPositive { .. })
                ),
                "a bank was built on a cell with a negative state variable"
            );
        }
        // Zero is a state, not an error: an emptied cell is one the model can start from.
        assert!(
            Meddis {
                q: 0.0,
                c: 0.0,
                w: 0.0,
                ..Meddis::default()
            }
            .validate()
            .is_ok()
        );
        // And a non-finite one is refused as non-finite rather than as non-positive.
        assert!(matches!(
            Meddis {
                c: f64::NAN,
                ..Meddis::default()
            }
            .validate(),
            Err(CochleaError::NotFinite { .. })
        ));
    }

    /// One `get_mut(ch)` in `transduce_one` is the whole of the per-channel gain wiring.
    ///
    /// ⛔ Every `AGC` test measured one channel, so `get_mut(0)` would have every channel gained by
    /// whatever channel zero happens to hear and nothing would fail. Driven from both ends of a
    /// three-channel bank: the tone's own channel must be compressed, and the channels that cannot
    /// hear the tone must be left alone. Measured under a 2 kHz tone: the 2 kHz channel goes from
    /// 1.000 to 0.137 and the 500 Hz channel from 9.17e−6 to 9.17e−6, its own loop never having
    /// left rest.
    #[test]
    fn each_channel_runs_its_own_gain_loop() {
        let build = |agc: Option<Agc>| {
            Filterbank::from_centre_frequencies(
                FS,
                &[500.0, 1000.0, 2000.0],
                Transduction::HalfWave(Compression::Linear),
            )
            .expect("a valid bank")
            .with_agc(agc)
            .expect("a valid loop")
        };
        let fresh_loop = || Some(Agc::new(20e-3, 20.0).expect("valid loop"));
        let peak = |bank: &mut Filterbank, x: &[f64], ch: usize| -> f64 {
            let out = bank.transduce(x).expect("finite");
            let tail = out[ch].len() * 2 / 3;
            out[ch][tail..].iter().cloned().fold(0.0f64, f64::max)
        };
        for &(driven, deaf) in &[(2usize, 0usize), (0, 2)] {
            let f = [500.0, 1000.0, 2000.0][driven];
            let x = tone(FS, f, 1.0, 0.3).expect("a valid tone");
            let own_plain = peak(&mut build(None), &x, driven);
            let own_gained = peak(&mut build(fresh_loop()), &x, driven);
            let far_plain = peak(&mut build(None), &x, deaf);
            let far_gained = peak(&mut build(fresh_loop()), &x, deaf);
            assert!(
                own_gained < 0.3 * own_plain,
                "a {f} Hz tone left channel {driven} at {own_gained} against {own_plain} \
                 ungained: its own loop is not acting on it"
            );
            assert!(
                far_gained > 0.99 * far_plain,
                "a {f} Hz tone left channel {deaf} at {far_gained} against {far_plain} ungained: \
                 it is being gained by a loop that is not its own"
            );
        }
    }

    /// The three output populations are three independent sets of neurons.
    ///
    /// ⛔ The offset pass steps `offset_cells[ch]`; stepping `onset_cells[ch]` there instead still
    /// puts an offset spike at the right instant, because the two currents are the two rectified
    /// halves of one difference and are never both non-zero on the same sample — so every edge
    /// test stayed green while one whole population was never stepped at all. There is no public
    /// builder that changes one population only ([`Filterbank::with_cell`] sets all three), so the
    /// cells are reached directly here. Measured on a 200 ms burst: 36 sustained, 23 onset,
    /// 23 offset, and a 10 mV stiffer offset cell drops the offsets to 13 and moves nothing else.
    #[test]
    fn the_three_populations_are_three_independent_sets_of_neurons() {
        let build = || {
            Filterbank::erb_bank(
                FS,
                400.0,
                3000.0,
                6,
                Transduction::HalfWave(Compression::default()),
            )
            .expect("a valid bank")
        };
        let x = tone_burst(FS, 1000.0, 1.0, 0.4, 0.1, 0.2, 2e-3).expect("a valid burst");
        let count = |bank: &mut Filterbank| -> (usize, usize, usize) {
            let train = bank.spike_train(&x).expect("finite");
            let (mut s, mut on, mut off) = (0, 0, 0);
            for sp in train.spikes() {
                match bank.decode_source(sp.source) {
                    Some((ChannelKind::Sustained, _)) => s += 1,
                    Some((ChannelKind::Onset, _)) => on += 1,
                    Some((ChannelKind::Offset, _)) => off += 1,
                    None => panic!("a spike with no address"),
                }
            }
            (s, on, off)
        };
        let (base_s, base_on, base_off) = count(&mut build());
        assert!(
            base_s > 0 && base_on > 0 && base_off > 0,
            "{base_s} {base_on} {base_off}"
        );

        let mut stiff_offset = build();
        for c in &mut stiff_offset.offset_cells {
            c.v_th += 10e-3;
        }
        let (s, on, off) = count(&mut stiff_offset);
        assert_eq!(
            (s, on),
            (base_s, base_on),
            "raising the OFFSET threshold moved the other two populations"
        );
        assert!(
            off < base_off,
            "a 10 mV stiffer offset cell left the offsets at {off} against {base_off}: they are \
             not being produced by the offset cells"
        );

        let mut stiff_onset = build();
        for c in &mut stiff_onset.onset_cells {
            c.v_th += 10e-3;
        }
        let (s, on, off) = count(&mut stiff_onset);
        assert_eq!(
            (s, off),
            (base_s, base_off),
            "raising the ONSET threshold moved the other two populations"
        );
        assert!(
            on < base_on,
            "a 10 mV stiffer onset cell left the onsets at {on} against {base_on}"
        );
    }

    /// [`Filterbank::onset_signal`] must be the same front end [`Filterbank::spike_train`] runs.
    ///
    /// ⛔ That is what `step_change_detector`'s doc claims — "a test that measures the change
    /// signal through one and a deployment that spikes through the other would otherwise be
    /// checking something the deployment does not run" — and nothing checked it. Advancing the
    /// detector before the filter that feeds it lags `onset_signal`'s trace by exactly one sample
    /// and leaves `spike_train` alone, and the steady-tone test that reads the trace holds an
    /// eightfold margin. Here the trace is replayed through a fresh [`Lif`] and must reproduce the
    /// bank's own onset spikes, tick for tick.
    #[test]
    fn the_change_signal_a_bank_reports_is_the_one_its_onset_cells_are_driven_by() {
        let mut bank = Filterbank::erb_bank(
            FS,
            400.0,
            3000.0,
            6,
            Transduction::HalfWave(Compression::default()),
        )
        .expect("a valid bank");
        let x = tone_burst(FS, 1000.0, 1.0, 0.4, 0.1, 0.2, 2e-3).expect("a valid burst");
        let dt = bank.dt();
        let drive = bank.onset_drive();
        let trace = bank.onset_signal(&x).expect("finite");
        let train = bank.spike_train(&x).expect("finite");

        let mut replayed: Vec<(u64, usize)> = Vec::new();
        for (ch, t) in trace.iter().enumerate() {
            let mut cell = Lif::default();
            cell.reset();
            for (i, &d) in t.iter().enumerate() {
                if cell.step(dt, d.max(0.0) * drive) {
                    replayed.push((i as u64, ch));
                }
            }
        }
        replayed.sort_unstable();
        let mut reported: Vec<(u64, usize)> = train
            .spikes()
            .iter()
            .filter_map(|s| match bank.decode_source(s.source) {
                Some((ChannelKind::Onset, ch)) => Some((s.t, ch)),
                _ => None,
            })
            .collect();
        reported.sort_unstable();
        assert!(
            !reported.is_empty(),
            "the burst produced no onset spikes to compare against"
        );
        assert_eq!(
            replayed, reported,
            "the trace onset_signal reports does not drive the bank's own onset cells"
        );
    }

    /// A [`Cochleagram`]'s three rows each hold their own population's spikes.
    ///
    /// ⛔ The rows are only ever read one at a time, and in silence all three are zero — so
    /// counting the onset spikes into the offset row, or summing the wrong row in
    /// [`Cochleagram::total_sustained_hz`], passed every test here. Checked against the train the
    /// bank itself produced, spike by spike. Measured on a 200 ms burst in a 400 ms buffer: the
    /// onset row's first channel holds 2.5 Hz where the offset row's holds 0.0, so the two rows
    /// are distinguishable rather than accidentally equal.
    #[test]
    fn a_cochleagram_counts_each_population_into_its_own_row() {
        let mut bank = Filterbank::erb_bank(
            FS,
            400.0,
            3000.0,
            6,
            Transduction::HalfWave(Compression::default()),
        )
        .expect("a valid bank");
        let x = tone_burst(FS, 1000.0, 1.0, 0.4, 0.1, 0.2, 2e-3).expect("a valid burst");
        let train = bank.spike_train(&x).expect("finite");
        let gram = bank.cochleagram(&train, x.len() as u64);
        let n = bank.channels();
        let (mut sus, mut on, mut off) = (vec![0u64; n], vec![0u64; n], vec![0u64; n]);
        for s in train.spikes() {
            match bank.decode_source(s.source) {
                Some((ChannelKind::Sustained, ch)) => sus[ch] += 1,
                Some((ChannelKind::Onset, ch)) => on[ch] += 1,
                Some((ChannelKind::Offset, ch)) => off[ch] += 1,
                None => panic!("a spike with no address"),
            }
        }
        assert_eq!(gram.seconds, x.len() as f64 * bank.dt());
        for ch in 0..n {
            // `k as f64 / seconds` is the operation the cochleagram performs, and it is repeated
            // here, so these are equalities.
            assert_eq!(
                gram.sustained_hz[ch],
                sus[ch] as f64 / gram.seconds,
                "channel {ch}: the sustained row"
            );
            assert_eq!(
                gram.onset_hz[ch],
                on[ch] as f64 / gram.seconds,
                "channel {ch}: the onset row"
            );
            assert_eq!(
                gram.offset_hz[ch],
                off[ch] as f64 / gram.seconds,
                "channel {ch}: the offset row"
            );
        }
        // The rows really differ here, so a comparison that swapped two of them would fail rather
        // than compare a row against itself.
        assert_ne!(gram.onset_hz, gram.offset_hz);
        assert_ne!(gram.sustained_hz, gram.onset_hz);
        // The total is the sustained row's own sum, in the same order, so this is an equality.
        let want: f64 = sus.iter().map(|&k| k as f64 / gram.seconds).sum();
        assert_eq!(
            gram.total_sustained_hz(),
            want,
            "the total came to {} against {want}",
            gram.total_sustained_hz()
        );
        assert_ne!(
            gram.total_sustained_hz(),
            gram.onset_hz.iter().sum::<f64>(),
            "the sustained total and the onset total are the same number here"
        );
    }

    /// A run of zero ticks has no duration to divide by.
    ///
    /// ⛔ Every call in this suite passes `signal.len()`, so the guard was never driven. With the
    /// comparison loosened to `>= 0.0` an empty train gives `0/0` and a non-empty one gives `k/0`,
    /// so every rate comes back `NaN` or infinite — and a `NaN` rate compares false against every
    /// bound a caller might put it under.
    #[test]
    fn a_run_of_no_length_reports_nothing_rather_than_dividing_by_zero() {
        let bank = Filterbank::erb_bank(
            FS,
            400.0,
            3000.0,
            4,
            Transduction::HalfWave(Compression::Linear),
        )
        .expect("a valid bank");
        let populated = Train::from_spikes(vec![
            Spike { t: 0, source: 0 },
            Spike { t: 0, source: 5 },
            Spike { t: 0, source: 9 },
        ]);
        for train in [Train::new(), populated] {
            let gram = bank.cochleagram(&train, 0);
            assert_eq!(gram.seconds, 0.0);
            for ch in 0..bank.channels() {
                assert_eq!(
                    gram.sustained_hz[ch], 0.0,
                    "channel {ch} sustained rate was {}",
                    gram.sustained_hz[ch]
                );
                assert_eq!(
                    gram.onset_hz[ch], 0.0,
                    "channel {ch} onset rate was {}",
                    gram.onset_hz[ch]
                );
                assert_eq!(
                    gram.offset_hz[ch], 0.0,
                    "channel {ch} offset rate was {}",
                    gram.offset_hz[ch]
                );
            }
            assert_eq!(gram.total_sustained_hz(), 0.0);
            assert!(
                gram.best_channel().is_none(),
                "a run of no length has no best channel"
            );
        }
    }

    /// Both sweeps start at zero phase, as [`tone`] does and for the same reason: a waveform that
    /// began mid-cycle opens with a step, and the click's onset response is exactly what a
    /// tonotopy measurement reads.
    ///
    /// ⛔ Nothing here could see it. The zero-crossing count that checks a chirp's total phase is
    /// a count of DIFFERENCES and is blind to a constant offset — and the exponential sweep's own
    /// `− 1` is that offset. Dropping it puts 293.6 radians of phase on sample zero and moves the
    /// crossing count by less than the test's own tolerance of two.
    #[test]
    fn both_chirps_start_at_zero_phase_and_rise() {
        for sweep in [Sweep::Linear, Sweep::Exponential] {
            let x = chirp(FS, 200.0, 4000.0, 1.0, 0.7, sweep).expect("a valid chirp");
            assert_eq!(
                x[0], 0.0,
                "{sweep:?}: sample zero is {}, so the chirp begins with a step",
                x[0]
            );
            // One sample in, the phase is the start frequency's: 2*pi*200/48000 = 0.02618 rad. The
            // tolerance is the sweep's own frequency change across that one sample — 0.113 Hz for
            // the linear sweep, which moves sample one by 7.4e-6.
            let want = (2.0 * PI * 200.0 / FS).sin();
            assert!(
                (x[1] - want).abs() < 2e-5,
                "{sweep:?}: sample one was {}, the 200 Hz start says {want}",
                x[1]
            );
        }
        // A degenerate exponential sweep really is the tone it says it degenerates to.
        let flat = chirp(FS, 440.0, 440.0, 0.7, 0.05, Sweep::Exponential).expect("a valid chirp");
        let plain = tone(FS, 440.0, 0.7, 0.05).expect("a valid tone");
        assert_eq!(flat.len(), plain.len());
        for (i, (&a, &b)) in flat.iter().zip(plain.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-12,
                "sample {i}: the flat sweep gave {a}, the tone {b}"
            );
        }
    }

    /// A noise burst carries the amplitude it was asked for.
    ///
    /// ⛔ The only burst this suite MEASURES has an amplitude of 1.0, and multiplying by one is
    /// the identity; the determinism test compares two runs that would both be missing the factor.
    /// Two bursts from the same seed at `a` and `2a`: scaling by two is exact in binary floating
    /// point, so the second is the first doubled, sample for sample.
    #[test]
    fn a_noise_burst_carries_the_amplitude_it_was_asked_for() {
        let mut r1 = Rng::new(31);
        let quiet = noise_burst(&mut r1, FS, 1.0, 0.2, 0.05, 0.1, 0.0).expect("a valid burst");
        let mut r2 = Rng::new(31);
        let loud = noise_burst(&mut r2, FS, 2.0, 0.2, 0.05, 0.1, 0.0).expect("a valid burst");
        assert_eq!(quiet.len(), loud.len());
        for (i, (&q, &l)) in quiet.iter().zip(loud.iter()).enumerate() {
            assert_eq!(l, 2.0 * q, "sample {i}: {l} is not twice {q}");
        }
        // And the level is the doc's own closed form at an amplitude that is not one: uniform on
        // [-a, a] has an rms of a/sqrt(3), which is 1.1547 at a = 2.
        let inside = &loud[(0.05 * FS) as usize..(0.15 * FS) as usize];
        let r = rms(inside).expect("a non-empty window");
        assert!(
            (r - 2.0 / 3f64.sqrt()).abs() < 0.02,
            "the burst's rms was {r}, uniform on [-2, 2] says {}",
            2.0 / 3f64.sqrt()
        );
        let extreme = inside.iter().cloned().fold(0.0f64, |m, v| m.max(v.abs()));
        assert!(
            extreme > 1.9 && extreme <= 2.0,
            "the burst's largest sample was {extreme}, not near the amplitude of 2 it was asked for"
        );
    }

    /// The pole sits in the UPPER half plane, and the derived `Debug` prints its sign.
    ///
    /// Pins the imaginary part of `p = r·e^{i·2π·f_c/fs}` as an observable. The cascade is
    /// conjugation-symmetric — `step` returns the real part and `envelope` returns the hypot of
    /// the two parts — so every driven measurement in this suite is bit-for-bit unchanged when
    /// `p_im` flips sign, and that is the shape of the hole: no fixture reads the pole itself.
    /// `Gammatone` derives `Debug`, which prints the private field by name, so the sign IS
    /// caller-visible. The expectation is rebuilt from the public accessors with the
    /// constructor's own operations in the constructor's own order, so equality is exact rather
    /// than toleranced.
    #[test]
    fn the_gammatone_pole_is_in_the_upper_half_plane_and_its_debug_shows_it() {
        let g = Gammatone::new(1000.0, 16_000.0).expect("a valid channel");
        let r = (-g.decay_rate() / g.fs()).exp();
        let theta = 2.0 * PI * g.f_c() / g.fs();
        // `f_c` is in (0, NYQUIST_GUARD·fs], so theta is in (0, 0.8·π] where the sine is
        // strictly positive: this holds for every admissible channel, not just this one.
        let p_im = r * theta.sin();
        assert!(p_im > 0.0, "measured imaginary part of the pole: {p_im}");
        let shown = format!("{g:?}");
        assert!(shown.contains(&format!("p_im: {p_im:?}")), "measured Debug: {shown}");
    }

    /// `permeability` takes its clamp BRANCH at `s + A == 0`, rather than an equal-valued ratio.
    ///
    /// Pins that the comparison is strict. Every field of `Meddis` is public and `permeability`
    /// calls no validator — `Meddis::validate` is a separate method whose own doc advertises this
    /// hazard — so `b` is whatever the caller put there. The hole is that every `Meddis` in this
    /// suite arrives through `Meddis::default` or `Filterbank` with the 1986 `b = 300`, where the
    /// two branches do agree at `num == 0`; at `b == 0` the taken branch is `0.0/0.0`, and at
    /// `b < 0` it is a negative zero that propagates into the rate.
    #[test]
    fn the_permeability_clamp_is_a_branch_and_not_an_algebraic_coincidence() {
        // num = -5.0 + A = -5.0 + 5.0 = +0.0 exactly, so the ratio would be 2000.0·0.0/(0.0+0.0).
        let shut = Meddis { b: 0.0, ..Meddis::default() };
        assert_eq!(shut.permeability(-5.0), 0.0, "the clamp must not evaluate the ratio");
        let inverted = Meddis { b: -300.0, ..Meddis::default() };
        assert!(
            inverted.permeability(-5.0).is_sign_positive(),
            "the clamp returns +0.0; the ratio returns 2000.0·0.0/(0.0 - 300.0) = -0.0"
        );
        assert!(
            inverted.steady_state_rate(-5.0).is_sign_positive(),
            "a negative zero permeability propagates through the cleft into the rate"
        );
    }
}
