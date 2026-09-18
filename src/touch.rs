//! Touch in spikes: the three mechanoreceptor channels of the glabrous skin as spiking afferents,
//! each driven by the feature of the contact it actually encodes, and a fingertip's worth of them
//! decoding where the contact is — checked against the closed forms of this crate's own neuron.
//!
//! # What the mechanism is
//!
//! The fingertip does not send "pressure". It sends three spike trains that encode three
//! different derivatives of the same indentation (Johansson and Flanagan, *Coding and use of
//! tactile signals from the fingertips in object manipulation tasks*, Nature Reviews Neuroscience
//! 10(5):345–359, 2009):
//!
//! | afferent | receptor | encodes | fires during |
//! |---|---|---|---|
//! | **SA1** (slowly adapting, type I) | Merkel cell | indentation **depth**, with slow adaptation | contact, for as long as it lasts |
//! | **RA** (rapidly adapting, type I) | Meissner corpuscle | indentation **velocity** | the making and breaking of contact, and slip |
//! | **PC** (Pacinian) | Pacinian corpuscle | **vibration**, 40–400 Hz, peaking near 250 Hz | texture and the transients of impact |
//!
//! Saal, Delhaye, Rayhaun and Bensmaia, *Simulating tactile signals from the whole hand with
//! millisecond precision*, PNAS 114(28):E5693–E5702, 2017 (`TouchSim`), fit integrate-and-fire
//! afferents of each class to recorded spike trains, with the drive of each a linear combination
//! of depth, velocity and acceleration. This module keeps that architecture — a feature, a linear
//! drive, this crate's [`crate::neuron::Lif`] — and **does not carry their fitted constants**:
//! every gain here is a round number stated at the constructor, and every claim is about the
//! shape of the response (which channel fires when) and the arithmetic of the neuron (the rate a
//! given drive produces, from [`crate::neuron::Lif::isi`]), not about matching a recording.
//!
//! # Why it is in a neuromorphic crate
//!
//! Touch is the sense a robot hand spends its energy on and the one the neuromorphic literature
//! has the least of. Three things make it the right shape for spikes, and each is a number this
//! module produces: an RA afferent is **silent during a steady hold** — a grasp costs nothing to
//! monitor until something moves; a PC afferent detects a vibration the SA1 channel cannot see at
//! all, so texture and slip arrive on a channel of their own; and a fingertip's population
//! locates a contact by the **centroid** of its SA1 activity, which is a decode a spiking core
//! performs with one accumulate per spike. The sense ledger this Institute keeps says sensing is
//! the unpriced term of a task's energy; [`Fingertip`] counts every afferent spike a contact
//! costs so that it can be priced.
//!
//! # The closed forms this module is checked against
//!
//! - Under a constant depth the SA1 rate is `1 / Lif::isi(I)` for the drive current, exactly (to
//!   the interval quantisation), and it falls as the adaptation decays the drive with the stated
//!   time constant.
//! - Under a ramp-and-hold, the RA channel fires during the ramps and **exactly zero** times
//!   during the hold; the PC channel fires exactly zero times for a slow ramp; the SA1 channel
//!   fires throughout.
//! - A small 250 Hz vibration drives the PC channel and neither of the others; the PC's band-pass
//!   passes 250 Hz with more gain than 25 Hz or 2500 Hz, by the filter's own transfer function.
//! - The SA1 population centroid recovers a contact position to within the afferent spacing,
//!   and the RA population fires during a slip and not before it.
//!
//! # What this module has NOT reproduced
//!
//! - `TouchSim`'s fitted parameters, its skin mechanics (the propagation of strain from a contact
//!   to a distant receptor is a Gaussian here, not a continuum model), or any recorded rate.
//! - SA2 afferents (Ruffini endings, skin stretch), which the whole-hand model has and this
//!   module leaves out, saying so.
//! - Any tactile sensor's own transfer function. The input is indentation depth in metres; what a
//!   sensor returns is the sensor's business.

use core::fmt;

use crate::neuron::{Lif, Neuron};

/// What went wrong, named rather than guessed around.
#[derive(Debug, Clone, PartialEq)]
pub enum TouchError {
    /// A parameter outside its range.
    OutOfRange {
        /// Which parameter.
        what: &'static str,
        /// Value supplied.
        value: f64,
        /// Lowest admissible.
        low: f64,
        /// Highest admissible.
        high: f64,
    },
    /// A `NaN` or infinity.
    NonFinite {
        /// Which quantity.
        what: &'static str,
        /// Position in the offending array, `0` for a scalar.
        index: usize,
    },
    /// A count of zero where at least one is needed.
    Empty {
        /// What was empty.
        what: &'static str,
    },
    /// An array of the wrong length.
    Dimension {
        /// Which object.
        what: &'static str,
        /// Length supplied.
        got: usize,
        /// Length required.
        want: usize,
    },
}

impl fmt::Display for TouchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfRange { what, value, low, high } => {
                write!(f, "{what} = {value} is outside [{low}, {high}]")
            }
            Self::NonFinite { what, index } => write!(f, "{what} is not finite at {index}"),
            Self::Empty { what } => write!(f, "{what} is empty"),
            Self::Dimension { what, got, want } => write!(f, "{what} has {got} entries, needs {want}"),
        }
    }
}

impl std::error::Error for TouchError {}

fn positive(what: &'static str, v: f64) -> Result<f64, TouchError> {
    if v.is_finite() && v > 0.0 {
        Ok(v)
    } else {
        Err(TouchError::OutOfRange { what, value: v, low: f64::MIN_POSITIVE, high: f64::INFINITY })
    }
}

fn non_negative(what: &'static str, v: f64) -> Result<f64, TouchError> {
    if v.is_finite() && v >= 0.0 {
        Ok(v)
    } else {
        Err(TouchError::OutOfRange { what, value: v, low: 0.0, high: f64::INFINITY })
    }
}

fn finite_slice(what: &'static str, v: &[f64]) -> Result<(), TouchError> {
    if let Some(i) = v.iter().position(|x| !x.is_finite()) {
        return Err(TouchError::NonFinite { what, index: i });
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Contact features
// ---------------------------------------------------------------------------------------------

/// The three features an indentation trace carries at one instant.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Contact {
    /// Indentation depth, metres, non-negative (zero is no contact).
    pub depth: f64,
    /// Indentation velocity, metres per second, signed.
    pub velocity: f64,
    /// Indentation acceleration, metres per second squared, signed.
    pub acceleration: f64,
}

/// Differentiate a depth trace sampled at `fs` into contact features, by central differences
/// (one-sided at the ends).
///
/// # Errors
///
/// [`TouchError::Empty`] for fewer than three samples, [`TouchError::OutOfRange`] for a
/// non-positive `fs` or a negative depth, [`TouchError::NonFinite`].
pub fn features(depth: &[f64], fs: f64) -> Result<Vec<Contact>, TouchError> {
    let fs = positive("fs", fs)?;
    if depth.len() < 3 {
        return Err(TouchError::Empty { what: "depth trace (needs three samples)" });
    }
    finite_slice("depth", depth)?;
    if let Some(i) = depth.iter().position(|d| *d < 0.0) {
        return Err(TouchError::OutOfRange { what: "depth", value: depth[i], low: 0.0, high: f64::INFINITY });
    }
    let n = depth.len();
    let dt = 1.0 / fs;
    let vel = |i: usize| -> f64 {
        if i == 0 {
            (depth[1] - depth[0]) / dt
        } else if i == n - 1 {
            (depth[n - 1] - depth[n - 2]) / dt
        } else {
            (depth[i + 1] - depth[i - 1]) / (2.0 * dt)
        }
    };
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let acc = if i == 0 || i == n - 1 {
            0.0
        } else {
            (depth[i + 1] - 2.0 * depth[i] + depth[i - 1]) / (dt * dt)
        };
        out.push(Contact { depth: depth[i], velocity: vel(i), acceleration: acc });
    }
    Ok(out)
}

// ---------------------------------------------------------------------------------------------
// Afferents
// ---------------------------------------------------------------------------------------------

/// Which receptor class an afferent belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Class {
    /// Slowly adapting type I: depth, with adaptation.
    Sa1,
    /// Rapidly adapting type I: speed.
    Ra,
    /// Pacinian: vibration, through a band-pass.
    Pc,
}

/// A second-order band-pass, two cascaded one-pole sections: a high-pass at `f_lo` into a
/// low-pass at `f_hi`. The Pacinian's tuning, in the simplest form that has a peak.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BandPass {
    /// Lower corner, hertz.
    pub f_lo: f64,
    /// Upper corner, hertz.
    pub f_hi: f64,
    hp_prev_in: f64,
    hp_state: f64,
    lp_state: f64,
}

impl BandPass {
    /// A band from `f_lo` to `f_hi`.
    ///
    /// # Errors
    ///
    /// [`TouchError::OutOfRange`] for non-positive corners or `f_lo ≥ f_hi`.
    pub fn new(f_lo: f64, f_hi: f64) -> Result<Self, TouchError> {
        let f_lo = positive("f_lo", f_lo)?;
        let f_hi = positive("f_hi", f_hi)?;
        if f_lo >= f_hi {
            return Err(TouchError::OutOfRange { what: "f_lo (must be below f_hi)", value: f_lo, low: 0.0, high: f_hi });
        }
        Ok(Self { f_lo, f_hi, hp_prev_in: 0.0, hp_state: 0.0, lp_state: 0.0 })
    }

    /// One sample in, one out, at sample rate `fs`.
    pub fn step(&mut self, fs: f64, x: f64) -> f64 {
        let dt = 1.0 / fs;
        let a_hp = 1.0 / (1.0 + core::f64::consts::TAU * self.f_lo * dt);
        self.hp_state = a_hp * (self.hp_state + x - self.hp_prev_in);
        self.hp_prev_in = x;
        let a_lp = 1.0 - (-core::f64::consts::TAU * self.f_hi * dt).exp();
        self.lp_state += (self.hp_state - self.lp_state) * a_lp;
        self.lp_state
    }

    /// The continuous-time magnitude response at `f` hertz:
    /// `(f/f_lo) / √(1 + (f/f_lo)²) · 1 / √(1 + (f/f_hi)²)`.
    #[must_use]
    pub fn gain(&self, f: f64) -> f64 {
        let r_lo = f / self.f_lo;
        let r_hi = f / self.f_hi;
        r_lo / (1.0 + r_lo * r_lo).sqrt() / (1.0 + r_hi * r_hi).sqrt()
    }

    /// Return the filter to rest.
    pub fn reset(&mut self) {
        self.hp_prev_in = 0.0;
        self.hp_state = 0.0;
        self.lp_state = 0.0;
    }
}

/// One afferent: a feature, a gain, an adaptation, a cell.
#[derive(Debug, Clone, PartialEq)]
pub struct Afferent {
    /// Receptor class.
    pub class: Class,
    /// Amperes per metre of depth (SA1), per metre-per-second of speed (RA), or per unit of
    /// band-passed acceleration in metres per second squared (PC).
    pub gain: f64,
    /// SA1 only: the fraction of the depth drive that adapts away, in `[0, 1)`.
    pub adaptation: f64,
    /// SA1 only: adaptation time constant, seconds.
    pub tau_adapt: f64,
    /// The cell.
    pub cell: Lif,
    /// PC only: the band-pass.
    pub band: Option<BandPass>,
    /// SA1 only: the adapted fraction currently in force, in `[0, adaptation]`.
    adapted: f64,
    /// Spikes emitted since the last reset.
    pub spikes: u64,
}

impl Afferent {
    /// An SA1 afferent: `gain` amperes per metre, a fraction `adaptation` of which decays away
    /// with time constant `tau_adapt` while the depth is held.
    ///
    /// # Errors
    ///
    /// [`TouchError::OutOfRange`] for a non-positive gain or time constant, or an adaptation
    /// outside `[0, 1)`.
    pub fn sa1(gain: f64, adaptation: f64, tau_adapt: f64, cell: Lif) -> Result<Self, TouchError> {
        positive("gain", gain)?;
        if !(0.0..1.0).contains(&adaptation) {
            return Err(TouchError::OutOfRange { what: "adaptation", value: adaptation, low: 0.0, high: 1.0 });
        }
        positive("tau_adapt", tau_adapt)?;
        Ok(Self { class: Class::Sa1, gain, adaptation, tau_adapt, cell, band: None, adapted: 0.0, spikes: 0 })
    }

    /// An RA afferent: `gain` amperes per metre per second of indentation speed, either direction.
    ///
    /// # Errors
    ///
    /// [`TouchError::OutOfRange`] for a non-positive gain.
    pub fn ra(gain: f64, cell: Lif) -> Result<Self, TouchError> {
        positive("gain", gain)?;
        Ok(Self { class: Class::Ra, gain, adaptation: 0.0, tau_adapt: 1.0, cell, band: None, adapted: 0.0, spikes: 0 })
    }

    /// A PC afferent: `gain` amperes per metre per second squared of band-passed acceleration,
    /// the band `40–400 Hz` by default.
    ///
    /// # Errors
    ///
    /// [`TouchError::OutOfRange`] for a non-positive gain, plus [`BandPass::new`]'s.
    pub fn pc(gain: f64, cell: Lif) -> Result<Self, TouchError> {
        positive("gain", gain)?;
        Ok(Self { class: Class::Pc, gain, adaptation: 0.0, tau_adapt: 1.0, cell, band: Some(BandPass::new(40.0, 400.0)?), adapted: 0.0, spikes: 0 })
    }

    /// The drive current this afferent would produce from `c` at this instant, amperes, with the
    /// adaptation and the band-pass advanced by one sample at `fs`.
    fn drive(&mut self, fs: f64, c: &Contact) -> f64 {
        match self.class {
            Class::Sa1 => {
                // The adapted fraction relaxes toward `adaptation` while in contact and back to
                // zero when the skin is released, with one time constant for both.
                let target = if c.depth > 0.0 { self.adaptation } else { 0.0 };
                let a = 1.0 - (-1.0 / (fs * self.tau_adapt)).exp();
                self.adapted += (target - self.adapted) * a;
                self.gain * c.depth * (1.0 - self.adapted)
            }
            Class::Ra => self.gain * c.velocity.abs(),
            Class::Pc => {
                let filtered = self.band.as_mut().map_or(0.0, |b| b.step(fs, c.acceleration));
                self.gain * filtered.abs()
            }
        }
    }

    /// Advance one sample at `fs` under contact `c`, returning whether the afferent fired.
    ///
    /// # Errors
    ///
    /// [`TouchError::OutOfRange`] for a non-positive `fs` or a negative depth,
    /// [`TouchError::NonFinite`] for a non-finite feature.
    pub fn step(&mut self, fs: f64, c: &Contact) -> Result<bool, TouchError> {
        let fs = positive("fs", fs)?;
        non_negative("depth", c.depth)?;
        if !c.velocity.is_finite() {
            return Err(TouchError::NonFinite { what: "velocity", index: 0 });
        }
        if !c.acceleration.is_finite() {
            return Err(TouchError::NonFinite { what: "acceleration", index: 0 });
        }
        let i = self.drive(fs, c);
        let fired = self.cell.step(1.0 / fs, i);
        if fired {
            self.spikes += 1;
        }
        Ok(fired)
    }

    /// The steady rate this afferent settles to under a constant contact, hertz, from
    /// [`Lif::isi`]: `None` if the drive is sub-threshold. For SA1 the adaptation is taken as
    /// complete; the transient rate is higher.
    #[must_use]
    pub fn steady_rate(&self, c: &Contact) -> Option<f64> {
        let i = match self.class {
            Class::Sa1 => self.gain * c.depth * (1.0 - self.adaptation),
            Class::Ra => self.gain * c.velocity.abs(),
            Class::Pc => 0.0,
        };
        self.cell.isi(i).map(|t| 1.0 / t)
    }

    /// Return the cell, the adaptation, the filter and the counter to rest.
    pub fn reset(&mut self) {
        self.cell.reset();
        self.adapted = 0.0;
        if let Some(b) = &mut self.band {
            b.reset();
        }
        self.spikes = 0;
    }
}

// ---------------------------------------------------------------------------------------------
// A fingertip
// ---------------------------------------------------------------------------------------------

/// A line of afferents of one class along one axis of skin, each with a Gaussian receptive
/// field, plus the spike counts a contact costs.
#[derive(Debug, Clone, PartialEq)]
pub struct Fingertip {
    /// Afferent positions along the skin, metres.
    pub positions: Vec<f64>,
    /// The afferents, one per position.
    pub afferents: Vec<Afferent>,
    /// Receptive-field width, metres: the depth an afferent sees is the contact depth times
    /// `exp(−(x − x_c)² / (2 σ²))`.
    pub sigma: f64,
    /// Samples stepped since the last reset.
    pub ticks: u64,
}

impl Fingertip {
    /// `n` afferents of one prototype, evenly spaced over `[0, length]` metres.
    ///
    /// # Errors
    ///
    /// [`TouchError::Empty`] for `n = 0`, [`TouchError::OutOfRange`] for a non-positive length or
    /// width.
    pub fn line(n: usize, length: f64, sigma: f64, proto: &Afferent) -> Result<Self, TouchError> {
        if n == 0 {
            return Err(TouchError::Empty { what: "afferents" });
        }
        positive("length", length)?;
        positive("sigma", sigma)?;
        let positions: Vec<f64> = (0..n).map(|k| if n == 1 { 0.5 * length } else { length * k as f64 / (n - 1) as f64 }).collect();
        Ok(Self { positions, afferents: vec![proto.clone(); n], sigma, ticks: 0 })
    }

    /// The afferent spacing, metres — the resolution floor of a centroid decode.
    #[must_use]
    pub fn spacing(&self) -> f64 {
        if self.positions.len() < 2 { 0.0 } else { self.positions[1] - self.positions[0] }
    }

    /// Advance one sample: a contact of features `c` centred at `x_c`, seen by every afferent
    /// through its receptive field. Returns the indices that fired.
    ///
    /// # Errors
    ///
    /// As [`Afferent::step`], plus [`TouchError::NonFinite`] for a non-finite `x_c`.
    pub fn step(&mut self, fs: f64, c: &Contact, x_c: f64) -> Result<Vec<usize>, TouchError> {
        if !x_c.is_finite() {
            return Err(TouchError::NonFinite { what: "x_c", index: 0 });
        }
        let mut fired = Vec::new();
        for (k, a) in self.afferents.iter_mut().enumerate() {
            let d = self.positions[k] - x_c;
            let w = (-d * d / (2.0 * self.sigma * self.sigma)).exp();
            let local = Contact { depth: c.depth * w, velocity: c.velocity * w, acceleration: c.acceleration * w };
            if a.step(fs, &local)? {
                fired.push(k);
            }
        }
        self.ticks += 1;
        Ok(fired)
    }

    /// The spike-count centroid, metres: `Σ_k x_k n_k / Σ_k n_k`. `None` with no spikes.
    #[must_use]
    pub fn centroid(&self) -> Option<f64> {
        let total: u64 = self.afferents.iter().map(|a| a.spikes).sum();
        if total == 0 {
            return None;
        }
        Some(self.positions.iter().zip(&self.afferents).map(|(x, a)| x * a.spikes as f64).sum::<f64>() / total as f64)
    }

    /// Total spikes since the last reset: what the contact cost the nerve.
    #[must_use]
    pub fn total_spikes(&self) -> u64 {
        self.afferents.iter().map(|a| a.spikes).sum()
    }

    /// Return every afferent and the counters to rest.
    pub fn reset(&mut self) {
        for a in &mut self.afferents {
            a.reset();
        }
        self.ticks = 0;
    }
}

/// A ramp-and-hold indentation: zero, a linear ramp to `depth` over `ramp_s`, a hold for
/// `hold_s`, a linear release over `ramp_s`, then zero — the standard tactile stimulus.
///
/// # Errors
///
/// [`TouchError::OutOfRange`] for a non-positive `fs`, depth or duration.
pub fn ramp_and_hold(fs: f64, depth: f64, ramp_s: f64, hold_s: f64, pad_s: f64) -> Result<Vec<f64>, TouchError> {
    let fs = positive("fs", fs)?;
    positive("depth", depth)?;
    positive("ramp_s", ramp_s)?;
    positive("hold_s", hold_s)?;
    non_negative("pad_s", pad_s)?;
    let total = 2.0 * pad_s + 2.0 * ramp_s + hold_s;
    let n = (total * fs).round() as usize;
    Ok((0..n)
        .map(|i| {
            let t = i as f64 / fs;
            if t < pad_s {
                0.0
            } else if t < pad_s + ramp_s {
                depth * (t - pad_s) / ramp_s
            } else if t < pad_s + ramp_s + hold_s {
                depth
            } else if t < pad_s + 2.0 * ramp_s + hold_s {
                depth * (1.0 - (t - pad_s - ramp_s - hold_s) / ramp_s)
            } else {
                0.0
            }
        })
        .collect())
}

/// As [`ramp_and_hold`] with raised-cosine ramps — continuous velocity, bounded acceleration —
/// so that the only transients are the ramps themselves and not their corners. A linear ramp's
/// corner is a one-sample acceleration impulse, which is a tap, and a Pacinian answers a tap.
///
/// # Errors
///
/// As [`ramp_and_hold`].
pub fn cosine_ramp_and_hold(fs: f64, depth: f64, ramp_s: f64, hold_s: f64, pad_s: f64) -> Result<Vec<f64>, TouchError> {
    let linear = ramp_and_hold(fs, depth, ramp_s, hold_s, pad_s)?;
    Ok(linear
        .iter()
        .map(|&d| {
            // Map the linear fraction through the raised cosine: ½(1 − cos(π f)).
            let f = d / depth;
            depth * 0.5 * (1.0 - (core::f64::consts::PI * f).cos())
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::{
        Afferent, BandPass, Class, Contact, Fingertip, TouchError, cosine_ramp_and_hold, features,
        ramp_and_hold,
    };
    use crate::neuron::Lif;

    fn cell() -> Lif {
        Lif::default()
    }

    /// Under a held depth the SA1 rate is `1 / Lif::isi` of the adapted drive, to the interval
    /// quantisation, and the early rate is higher than the late one by the adaptation.
    #[test]
    fn the_sa1_rate_is_the_neurons_own_closed_form_and_adapts() {
        let fs = 10_000.0;
        // 4 nA per millimetre: 1 mm gives 4 nA, adapted by 30% to 2.8 nA, both above the 1.5 nA
        // rheobase of the default cell (15 mV over 10 MΩ). At 2 nA per millimetre the adapted
        // drive was 1.4 nA and the afferent went silent, which is what the first run found.
        let mut a = Afferent::sa1(4e-6, 0.3, 0.2, cell()).unwrap();
        let c = Contact { depth: 1e-3, velocity: 0.0, acceleration: 0.0 };
        let early_end = (0.05 * fs) as usize;
        let late_start = (2.0 * fs) as usize;
        let late_end = (3.0 * fs) as usize;
        let (mut early, mut late) = (0u64, 0u64);
        for k in 0..late_end {
            let f = a.step(fs, &c).unwrap();
            if k < early_end && f {
                early += 1;
            }
            if k >= late_start && f {
                late += 1;
            }
        }
        let late_rate = late as f64 / 1.0;
        let want = a.steady_rate(&c).unwrap();
        assert!((late_rate - want).abs() <= 1.0 + want * 1e-4 / (1.0 / want), "late {late_rate} Hz vs isi {want} Hz");
        let early_rate = early as f64 / 0.05;
        let unadapted = 1.0 / cell().isi(4e-9).unwrap();
        assert!(early_rate > late_rate + 5.0, "no adaptation: early {early_rate}, late {late_rate}");
        assert!((early_rate - unadapted).abs() < 25.0, "early {early_rate} vs unadapted {unadapted}");
        assert!(a.spikes >= early + late);
        // No contact, no spikes.
        a.reset();
        for _ in 0..(fs as usize) {
            assert!(!a.step(fs, &Contact { depth: 0.0, velocity: 0.0, acceleration: 0.0 }).unwrap());
        }
        assert_eq!(a.steady_rate(&Contact { depth: 0.0, velocity: 0.0, acceleration: 0.0 }), None);
    }

    /// Ramp-and-hold: RA fires during the ramps and EXACTLY zero times during the hold; PC fires
    /// exactly zero times anywhere for a slow ramp; SA1 fires throughout the hold.
    #[test]
    fn a_ramp_and_hold_separates_the_three_channels() {
        let fs = 10_000.0;
        let depth = ramp_and_hold(fs, 1e-3, 0.05, 0.5, 0.05).unwrap();
        let feats = features(&depth, fs).unwrap();
        let mut sa1 = Afferent::sa1(4e-6, 0.3, 0.2, cell()).unwrap();
        // 20 mm/s ramp: 1 mm over 50 ms. RA gain 2e-7 A per m/s gives 4 nA on the ramp.
        let mut ra = Afferent::ra(2e-7, cell()).unwrap();
        let mut pc = Afferent::pc(1e-9, cell()).unwrap();
        let hold = ((0.10 * fs) as usize)..((0.60 * fs) as usize);
        let ramp_in = ((0.05 * fs) as usize)..((0.10 * fs) as usize);
        let (mut sa_hold, mut ra_hold, mut ra_ramp, mut pc_total, mut sa_ramp) = (0, 0, 0, 0, 0);
        for (k, c) in feats.iter().enumerate() {
            let s = sa1.step(fs, c).unwrap();
            let r = ra.step(fs, c).unwrap();
            let p = pc.step(fs, c).unwrap();
            if hold.contains(&k) {
                sa_hold += u32::from(s);
                ra_hold += u32::from(r);
            }
            if ramp_in.contains(&k) {
                ra_ramp += u32::from(r);
                sa_ramp += u32::from(s);
            }
            pc_total += u32::from(p);
        }
        assert!(sa_hold > 15, "SA1 fired {sa_hold} times during a 500 ms hold");
        assert!(ra_ramp > 0, "RA did not fire during the ramp");
        assert_eq!(ra_hold, 0, "RA fired {ra_hold} times during a steady hold");
        // The RELEASE ramp too: the RA channel is sign-blind, and a rectified drive would answer
        // the making of contact and not its breaking.
        let release = ((0.60 * fs) as usize)..((0.65 * fs) as usize);
        let mut ra_out = Afferent::ra(2e-7, cell()).unwrap();
        let mut ra_release = 0;
        for (k, c) in feats.iter().enumerate() {
            if ra_out.step(fs, c).unwrap() && release.contains(&k) {
                ra_release += 1;
            }
        }
        assert!(ra_release > 0, "RA did not fire while the contact was released");
        // A linear ramp's corners are one-sample acceleration impulses of 200 m/s². Through the
        // band-pass (0.22 of it reaches the cell) and the cell's 20 ms integration that deposits
        // about 9 mV, under the 15 mV threshold, so the Pacinian is silent for this ramp too — the
        // first draft asserted it would answer the corner as a tap, and the arithmetic said no.
        // The bounded-acceleration claim is made on the raised-cosine stimulus below.
        assert_eq!(pc_total, 0, "PC fired {pc_total} times at the corners of a 20 mm/s ramp");
        assert!(sa_ramp < sa_hold, "SA1 fired more during the 50 ms ramp than the 500 ms hold");
        let smooth = cosine_ramp_and_hold(fs, 1e-3, 0.05, 0.5, 0.05).unwrap();
        let sfeats = features(&smooth, fs).unwrap();
        let mut pc2 = Afferent::pc(1e-9, cell()).unwrap();
        let mut ra2 = Afferent::ra(2e-7, cell()).unwrap();
        let (mut pc_smooth, mut ra_smooth_hold, mut ra_smooth_ramp) = (0, 0, 0);
        for (k, c) in sfeats.iter().enumerate() {
            pc_smooth += u32::from(pc2.step(fs, c).unwrap());
            let r = ra2.step(fs, c).unwrap();
            if hold.contains(&k) {
                ra_smooth_hold += u32::from(r);
            }
            if ramp_in.contains(&k) {
                ra_smooth_ramp += u32::from(r);
            }
        }
        assert_eq!(pc_smooth, 0, "PC fired {pc_smooth} times for a smooth 50 ms ramp");
        assert_eq!(ra_smooth_hold, 0);
        assert!(ra_smooth_ramp > 0, "RA did not fire during a smooth ramp");
        let peak_acc = sfeats.iter().map(|c| c.acceleration.abs()).fold(0.0f64, f64::max);
        assert!(peak_acc < 5.0, "the smooth ramp's acceleration peaked at {peak_acc} m/s²");
        // The velocity feature during the hold is exactly zero, which is why the RA count is.
        assert!(feats[hold.start + 10..hold.end - 10].iter().all(|c| c.velocity == 0.0 && c.acceleration == 0.0));
        assert!((feats[ramp_in.start + 10].velocity - 0.02).abs() < 1e-9, "20 mm/s ramp");
    }

    /// A small 250 Hz vibration drives the PC channel and neither of the others; the band-pass
    /// passes 250 Hz more than 25 Hz or 2500 Hz by its own transfer function, and the measured
    /// steady amplitude matches that function to 5% at the three frequencies.
    #[test]
    fn a_vibration_is_a_pacinian_signal_and_nothing_else() {
        let fs = 20_000.0;
        let mut band = BandPass::new(40.0, 400.0).unwrap();
        for f in [25.0, 250.0, 2500.0] {
            band.reset();
            let w = core::f64::consts::TAU * f;
            let mut peak = 0.0f64;
            let steps = (0.5 * fs) as usize;
            for k in 0..steps {
                let t = k as f64 / fs;
                let y = band.step(fs, (w * t).sin());
                if t > 0.3 {
                    peak = peak.max(y.abs());
                }
            }
            let want = band.gain(f);
            assert!((peak - want).abs() < 0.05 * want + 0.01, "{f} Hz: measured {peak} vs gain {want}");
        }
        // Two one-pole sections are a shallow band — 6 dB per octave each side — so the peak is
        // 1.6x the gain a decade below and 5x a decade above; the real Pacinian's tuning is
        // steeper, and this is the simplest form that has a peak at all.
        assert!(band.gain(250.0) > 1.5 * band.gain(25.0) && band.gain(250.0) > 3.0 * band.gain(2500.0));

        // 3 µm at 250 Hz on top of a 1 mm hold: velocity amplitude 2π·250·3e-6 = 4.7 mm/s, which
        // at 0.2 µA per m/s is 0.94 nA — under the 1.5 nA rheobase, so the RA channel stays
        // silent; acceleration amplitude (2π·250)²·3e-6 = 7.4 m/s², which through the band and a
        // 1 nA per m/s² gain is 6 nA — four times rheobase.
        let hold = 1e-3;
        let amp = 3e-6;
        let w = core::f64::consts::TAU * 250.0;
        let n = (1.0 * fs) as usize;
        let depth: Vec<f64> = (0..n).map(|k| hold + amp * (w * k as f64 / fs).sin()).collect();
        let feats = features(&depth, fs).unwrap();
        let mut sa1 = Afferent::sa1(4e-6, 0.3, 0.2, cell()).unwrap();
        let mut ra = Afferent::ra(2e-7, cell()).unwrap();
        let mut pc = Afferent::pc(1e-9, cell()).unwrap();
        let (mut s_count, mut r_count, mut p_count) = (0, 0, 0);
        for c in &feats {
            s_count += u32::from(sa1.step(fs, c).unwrap());
            r_count += u32::from(ra.step(fs, c).unwrap());
            p_count += u32::from(pc.step(fs, c).unwrap());
        }
        assert!(p_count > 50, "PC fired only {p_count} times under a 250 Hz vibration");
        assert_eq!(r_count, 0, "RA fired {r_count} times under a 4.7 mm/s peak vibration");
        // The band-pass is IN the Pacinian's path: a 5 Hz, 5 mm sway on a 6 mm hold has a raw
        // acceleration amplitude of (2π·5)²·5e-3 = 4.9 m/s² — 4.9 nA raw, which fires — but the
        // band passes 0.12 of it, 0.6 nA, under rheobase. Feeding the raw acceleration to the
        // cell survived the first mutation sweep because nothing slow and large had been tried.
        let w_slow = core::f64::consts::TAU * 5.0;
        let slow: Vec<f64> = (0..n).map(|k| 6e-3 + 5e-3 * (w_slow * k as f64 / fs).sin()).collect();
        let sfeats = features(&slow, fs).unwrap();
        let mut pc_slow = Afferent::pc(1e-9, cell()).unwrap();
        let mut ra_slow = Afferent::ra(2e-7, cell()).unwrap();
        let (mut p_slow, mut r_slow) = (0, 0);
        for c in &sfeats {
            p_slow += u32::from(pc_slow.step(fs, c).unwrap());
            r_slow += u32::from(ra_slow.step(fs, c).unwrap());
        }
        assert_eq!(p_slow, 0, "PC fired {p_slow} times under a 5 Hz sway the band should reject");
        assert!(r_slow > 0, "a 157 mm/s sway is an RA signal and RA did not fire");
        // The SA1 keeps firing for the hold, vibration or not; the PC channel is the one that
        // added something.
        assert!(s_count > 20);
        assert_eq!(pc.class, Class::Pc);
        assert_eq!(sa1.steady_rate(&feats[0]).map(|r| r > 0.0), Some(true));
    }

    /// A fingertip locates a contact by the SA1 spike-count centroid to within the afferent
    /// spacing, at three positions, and its RA population is silent during a hold and fires when
    /// the contact slips.
    #[test]
    fn a_fingertip_locates_a_contact_and_its_ra_population_reports_a_slip() {
        let fs = 10_000.0;
        let sa1 = Afferent::sa1(4e-6, 0.3, 0.2, cell()).unwrap();
        let mut tip = Fingertip::line(21, 20e-3, 2e-3, &sa1).unwrap();
        assert!((tip.spacing() - 1e-3).abs() < 1e-15);
        let c = Contact { depth: 1e-3, velocity: 0.0, acceleration: 0.0 };
        for x_c in [5e-3, 10.3e-3, 14.9e-3] {
            tip.reset();
            for _ in 0..(0.5 * fs) as usize {
                tip.step(fs, &c, x_c).unwrap();
            }
            let got = tip.centroid().unwrap();
            assert!((got - x_c).abs() < tip.spacing(), "contact at {x_c} decoded at {got}");
            assert!(tip.total_spikes() > 100);
        }
        assert_eq!(Fingertip::line(3, 1e-2, 1e-3, &sa1).unwrap().centroid(), None);

        // Slip: a hold with zero velocity, then the contact moves across the skin at 50 mm/s —
        // each afferent sees a rising then falling local depth, which is a velocity.
        let ra = Afferent::ra(2e-7, cell()).unwrap();
        let mut ra_tip = Fingertip::line(21, 20e-3, 2e-3, &ra).unwrap();
        let hold_steps = (0.3 * fs) as usize;
        for _ in 0..hold_steps {
            ra_tip.step(fs, &c, 8e-3).unwrap();
        }
        let during_hold = ra_tip.total_spikes();
        assert_eq!(during_hold, 0, "the RA population fired {during_hold} times during a steady hold");
        // The slip: the contact centre moves; each afferent's local depth changes at a rate the
        // Gaussian profile sets, and the population must report it.
        let mut x = 8e-3;
        let dx = 0.05 / fs;
        let mut prev_local = [0.0f64; 21];
        for (k, p) in ra_tip.positions.iter().enumerate() {
            let d = p - 8e-3;
            prev_local[k] = c.depth * (-d * d / (2.0 * 4e-6)).exp();
        }
        for _ in 0..(0.1 * fs) as usize {
            x += dx;
            // The features each afferent sees are its OWN local derivatives, so the slip is
            // presented through per-afferent contacts rather than one global velocity.
            for (k, a) in ra_tip.afferents.iter_mut().enumerate() {
                let d = ra_tip.positions[k] - x;
                let local = c.depth * (-d * d / (2.0 * 4e-6)).exp();
                let v = (local - prev_local[k]) * fs;
                prev_local[k] = local;
                a.step(fs, &Contact { depth: local, velocity: v, acceleration: 0.0 }).unwrap();
            }
        }
        assert!(ra_tip.total_spikes() > 0, "a 50 mm/s slip produced no RA spikes");
    }

    /// The feature derivatives and the stimulus generator against hand values.
    #[test]
    fn features_and_the_stimulus_are_their_definitions() {
        let fs = 1000.0;
        let depth: Vec<f64> = (0..5).map(|k| 1e-3 * k as f64).collect(); // 1 m/s ramp
        let f = features(&depth, fs).unwrap();
        assert!((f[2].velocity - 1.0).abs() < 1e-9);
        assert!(f[2].acceleration.abs() < 1e-6);
        // On a parabola the central difference is exact and a one-sided one is not: depth k² at
        // unit rate gives velocity 2t = 4 at t = 2 (one-sided: 5) and acceleration exactly 2.
        let parabola = [0.0, 1.0, 4.0, 9.0, 16.0];
        let g = features(&parabola, 1.0).unwrap();
        assert_eq!(g[2].velocity, 4.0);
        assert_eq!(g[2].acceleration, 2.0);
        let d = ramp_and_hold(fs, 2e-3, 0.1, 0.2, 0.05).unwrap();
        assert_eq!(d.len(), 500);
        assert_eq!(d[0], 0.0);
        assert!((d[100] - 1e-3).abs() < 1e-12, "halfway up the ramp");
        assert_eq!(d[200], 2e-3);
        assert!(d[499] == 0.0);
        assert!(matches!(features(&[0.0, 1.0], fs), Err(TouchError::Empty { .. })));
        assert!(matches!(features(&[0.0, -1.0, 0.0], fs), Err(TouchError::OutOfRange { what: "depth", .. })));
        assert!(matches!(BandPass::new(400.0, 40.0), Err(TouchError::OutOfRange { .. })));
        assert!(matches!(Afferent::sa1(1.0, 1.0, 1.0, cell()), Err(TouchError::OutOfRange { what: "adaptation", .. })));
        let mut a = Afferent::ra(1.0, cell()).unwrap();
        assert!(matches!(a.step(fs, &Contact { depth: -1.0, velocity: 0.0, acceleration: 0.0 }), Err(TouchError::OutOfRange { .. })));
        assert!(matches!(a.step(fs, &Contact { depth: 0.0, velocity: f64::NAN, acceleration: 0.0 }), Err(TouchError::NonFinite { what: "velocity", .. })));
        assert!(matches!(Fingertip::line(0, 1.0, 1.0, &a), Err(TouchError::Empty { .. })));
        for e in [
            TouchError::OutOfRange { what: "w", value: 9.0, low: 0.0, high: 1.0 },
            TouchError::NonFinite { what: "z", index: 0 },
            TouchError::Empty { what: "x" },
            TouchError::Dimension { what: "y", got: 1, want: 2 },
        ] {
            assert!(!e.to_string().is_empty());
        }
    }
}
