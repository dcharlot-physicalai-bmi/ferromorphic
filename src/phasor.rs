//! Phasor symbols and the spike times that carry them: the complex-valued vector symbolic
//! architecture whose elements are phases, so that a symbol is a pattern of spike timings within
//! a rhythm and binding is adding delays — with the algebra checked exactly and the associative
//! memory checked against its own crosstalk.
//!
//! # What the mechanism is
//!
//! In [`crate::vsa`] a symbol is a vector of signs. Make each element a **unit complex number**
//! `e^{iφ}` instead and three things happen at once. Binding becomes element-wise multiplication,
//! which is *adding phases* — exact, and exactly undone by the conjugate. Bundling becomes a
//! complex sum whose phase is a circular mean. And a symbol becomes something a spiking network
//! can hold with no numbers at all: one spike per element per cycle of a background rhythm, at the
//! time the phase says. Binding two symbols is then delaying one train by the other, and reading
//! a similarity is counting coincidences. That mapping is Frady and Sommer, *Robust computation
//! with rhythmic spike patterns*, PNAS 116(36):18050–18058, 2019, and it is the reason a
//! hyperdimensional computer can live on a chip that emits only spike times.
//!
//! The same paper's memory is the **threshold phasor associative memory**: store patterns as a
//! complex Hebbian matrix `W = Σ_μ ξ^μ ξ^μ†`, and retrieve by iterating `z ← ph(W z)`, each element
//! re-normalised to unit magnitude if its field is above a threshold and silenced otherwise. A
//! stored pattern is a fixed point because its own term of the field is `D ξ^ν` and the others are
//! crosstalk of magnitude `√((P − 1) D)` — the same ratio that governs [`crate::hopfield`], now in
//! phase.
//!
//! # The closed forms this module is checked against
//!
//! - `unbind(bind(a, b), b) = a` to 1e-12 in every element; the similarity of a symbol with itself
//!   is exactly 1; two random symbols have a similarity of standard deviation `1/√(2D)`.
//! - A sum-bundle of `k` symbols has similarity `1` to each member in expectation, with noise of
//!   standard deviation `√((k − 1)/(2D))` — measured.
//! - Binding by spike-time delays equals binding by phases, modulo the cycle, to 1e-12.
//! - Stored patterns are fixed points of the memory to the similarity the crosstalk predicts
//!   (`1 − σ²/2` with `σ` the per-element phase error), and a cue with a fifth of its phases
//!   randomised converges to its pattern.
//!
//! # What this module has NOT reproduced
//!
//! - The paper's spiking neuron model, its resonate-and-fire implementation of the phasor unit,
//!   or its capacity curves. The spike-time mapping here is the arithmetic of the representation;
//!   the neuron that would produce those spikes is [`crate::resonate::ResonateAndFire`], and
//!   wiring the two is not done here.
//! - Sparse phasor codes. Every element carries a phase; the paper's sparse variant zeroes most.

use core::fmt;

use crate::rng::Rng;

/// What went wrong, named rather than guessed around.
#[derive(Debug, Clone, PartialEq)]
pub enum PhasorError {
    /// A dimension of zero, or a set with no members.
    Empty {
        /// What was empty.
        what: &'static str,
    },
    /// Two vectors of different dimensions were combined.
    Dimension {
        /// The first.
        a: usize,
        /// The second.
        b: usize,
    },
    /// A phase, period or magnitude that is not finite.
    NonFinite {
        /// Position of the first offending element, `0` for a scalar.
        index: usize,
    },
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
}

impl fmt::Display for PhasorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty { what } => write!(f, "{what} is empty"),
            Self::Dimension { a, b } => write!(f, "dimension {a} against dimension {b}"),
            Self::NonFinite { index } => write!(f, "element {index} is not finite"),
            Self::OutOfRange { what, value, low, high } => {
                write!(f, "{what} = {value} is outside [{low}, {high}]")
            }
        }
    }
}

impl std::error::Error for PhasorError {}

/// A phasor symbol: `D` phases in radians, each standing for the unit complex number `e^{iφ}`.
#[derive(Debug, Clone, PartialEq)]
pub struct Phasor {
    /// Phases, radians, each in `[0, 2π)`.
    pub phase: Vec<f64>,
}

fn wrap(p: f64) -> f64 {
    let t = core::f64::consts::TAU;
    let r = p.rem_euclid(t);
    if r >= t { 0.0 } else { r }
}

fn check(v: &[f64]) -> Result<(), PhasorError> {
    if let Some(i) = v.iter().position(|x| !x.is_finite()) {
        return Err(PhasorError::NonFinite { index: i });
    }
    Ok(())
}

fn same(a: usize, b: usize) -> Result<(), PhasorError> {
    if a == b { Ok(()) } else { Err(PhasorError::Dimension { a, b }) }
}

impl Phasor {
    /// A symbol from explicit phases, wrapped into `[0, 2π)`.
    ///
    /// # Errors
    ///
    /// [`PhasorError::Empty`] for no phases, [`PhasorError::NonFinite`].
    pub fn new(phase: &[f64]) -> Result<Self, PhasorError> {
        if phase.is_empty() {
            return Err(PhasorError::Empty { what: "phase" });
        }
        check(phase)?;
        Ok(Self { phase: phase.iter().map(|&p| wrap(p)).collect() })
    }

    /// A random symbol of dimension `dim`: phases uniform on `[0, 2π)`.
    ///
    /// # Errors
    ///
    /// [`PhasorError::Empty`] for `dim = 0`.
    pub fn random(dim: usize, rng: &mut Rng) -> Result<Self, PhasorError> {
        if dim == 0 {
            return Err(PhasorError::Empty { what: "dimension" });
        }
        Ok(Self { phase: (0..dim).map(|_| core::f64::consts::TAU * rng.next_f64()).collect() })
    }

    /// Dimension.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.phase.len()
    }

    /// `a ⊗ b`: phases add. Exactly invertible by [`Phasor::unbind`].
    ///
    /// # Errors
    ///
    /// [`PhasorError::Dimension`].
    pub fn bind(&self, other: &Self) -> Result<Self, PhasorError> {
        same(self.dim(), other.dim())?;
        Ok(Self { phase: self.phase.iter().zip(&other.phase).map(|(a, b)| wrap(a + b)).collect() })
    }

    /// `(a ⊗ b) ⊗ b* = a`: phases subtract.
    ///
    /// # Errors
    ///
    /// [`PhasorError::Dimension`].
    pub fn unbind(&self, other: &Self) -> Result<Self, PhasorError> {
        same(self.dim(), other.dim())?;
        Ok(Self { phase: self.phase.iter().zip(&other.phase).map(|(a, b)| wrap(a - b)).collect() })
    }

    /// The conjugate `a*`: every phase negated. `bind(a, conjugate(a))` is the identity (all
    /// phases zero).
    #[must_use]
    pub fn conjugate(&self) -> Self {
        Self { phase: self.phase.iter().map(|&p| wrap(-p)).collect() }
    }

    /// Cyclic shift right by `k`, the permutation.
    #[must_use]
    pub fn permute(&self, k: i64) -> Self {
        let d = self.dim() as i64;
        let mut phase = self.phase.clone();
        phase.rotate_right(k.rem_euclid(d) as usize);
        Self { phase }
    }

    /// The similarity `Re(⟨a, b⟩)/D = (1/D) Σ_k cos(φ_k − ψ_k)`, in `[−1, 1]`, exactly `1` for
    /// identical symbols.
    ///
    /// # Errors
    ///
    /// [`PhasorError::Dimension`].
    pub fn similarity(&self, other: &Self) -> Result<f64, PhasorError> {
        same(self.dim(), other.dim())?;
        Ok(self.phase.iter().zip(&other.phase).map(|(a, b)| (a - b).cos()).sum::<f64>() / self.dim() as f64)
    }

    /// The spike times, seconds, of this symbol on a rhythm of `period` seconds: element `k`
    /// fires at `period · φ_k / 2π` after the cycle's start.
    ///
    /// # Errors
    ///
    /// [`PhasorError::OutOfRange`] for a non-positive period.
    pub fn spike_times(&self, period: f64) -> Result<Vec<f64>, PhasorError> {
        if !(period > 0.0) || !period.is_finite() {
            return Err(PhasorError::OutOfRange { what: "period", value: period, low: f64::MIN_POSITIVE, high: f64::INFINITY });
        }
        Ok(self.phase.iter().map(|p| period * p / core::f64::consts::TAU).collect())
    }

    /// The symbol whose phases are these spike times on a rhythm of `period` seconds — the inverse
    /// of [`Phasor::spike_times`], with times outside one cycle wrapped.
    ///
    /// # Errors
    ///
    /// [`PhasorError::Empty`], [`PhasorError::NonFinite`], [`PhasorError::OutOfRange`].
    pub fn from_spike_times(times: &[f64], period: f64) -> Result<Self, PhasorError> {
        if !(period > 0.0) || !period.is_finite() {
            return Err(PhasorError::OutOfRange { what: "period", value: period, low: f64::MIN_POSITIVE, high: f64::INFINITY });
        }
        if times.is_empty() {
            return Err(PhasorError::Empty { what: "times" });
        }
        check(times)?;
        Ok(Self { phase: times.iter().map(|t| wrap(core::f64::consts::TAU * t / period)).collect() })
    }
}

/// A complex sum of symbols, kept as `(re, im)` per element so that magnitudes carry.
#[derive(Debug, Clone, PartialEq)]
pub struct Bundle {
    /// Real parts.
    pub re: Vec<f64>,
    /// Imaginary parts.
    pub im: Vec<f64>,
}

impl Bundle {
    /// The sum of `members`.
    ///
    /// # Errors
    ///
    /// [`PhasorError::Empty`] for no members, [`PhasorError::Dimension`] if they disagree.
    pub fn sum(members: &[Phasor]) -> Result<Self, PhasorError> {
        if members.is_empty() {
            return Err(PhasorError::Empty { what: "bundle" });
        }
        let d = members[0].dim();
        let (mut re, mut im) = (vec![0.0; d], vec![0.0; d]);
        for m in members {
            same(m.dim(), d)?;
            for (k, &p) in m.phase.iter().enumerate() {
                re[k] += p.cos();
                im[k] += p.sin();
            }
        }
        Ok(Self { re, im })
    }

    /// `Re(⟨bundle, symbol⟩)/D`: `1` in expectation for a member, `0` for a stranger.
    ///
    /// # Errors
    ///
    /// [`PhasorError::Dimension`].
    pub fn similarity(&self, symbol: &Phasor) -> Result<f64, PhasorError> {
        same(self.re.len(), symbol.dim())?;
        let d = self.re.len() as f64;
        Ok(symbol
            .phase
            .iter()
            .enumerate()
            .map(|(k, &p)| self.re[k] * p.cos() + self.im[k] * p.sin())
            .sum::<f64>()
            / d)
    }

    /// The phases of the sum, as a symbol: the circular mean per element. An element whose sum
    /// has zero magnitude gets phase zero.
    #[must_use]
    pub fn phases(&self) -> Phasor {
        Phasor { phase: self.re.iter().zip(&self.im).map(|(r, i)| wrap(i.atan2(*r))).collect() }
    }
}

/// The standard deviation of the similarity between two random symbols: `1/√(2D)` — the average
/// of `D` cosines of uniform phases, each of variance `½`.
#[must_use]
pub fn noise_sd(dim: usize) -> f64 {
    if dim == 0 { f64::NAN } else { 1.0 / (2.0 * dim as f64).sqrt() }
}

// ---------------------------------------------------------------------------------------------
// Threshold phasor associative memory
// ---------------------------------------------------------------------------------------------

/// The threshold phasor associative memory: complex Hebbian storage, phase-normalising retrieval.
#[derive(Debug, Clone, PartialEq)]
pub struct Tpam {
    /// Dimension.
    pub dim: usize,
    /// Stored patterns.
    pub patterns: Vec<Phasor>,
    /// Elements whose field magnitude falls below `threshold · D` are silenced (phase set to zero
    /// and reported as silent) rather than normalised out of noise.
    pub threshold: f64,
}

/// One retrieval step's outcome.
#[derive(Debug, Clone, PartialEq)]
pub struct Retrieval {
    /// The state.
    pub state: Phasor,
    /// Iterations run.
    pub iterations: usize,
    /// Whether the last iteration moved no phase by more than `1e-9`.
    pub converged: bool,
    /// The largest phase movement on the last iteration, radians: how far from a fixed point the
    /// state still is when `converged` is false.
    pub last_move: f64,
    /// Elements silenced by the threshold on the last iteration. A silenced element is reported
    /// with phase zero, because [`Phasor`] carries phases only; a caller reading a similarity off
    /// a state with silenced elements should discount up to `2 · silent / D` for them.
    pub silent: usize,
}

impl Tpam {
    /// An empty memory of dimension `dim` with the given threshold, a fraction of `D`.
    ///
    /// # Errors
    ///
    /// [`PhasorError::Empty`] for `dim = 0`, [`PhasorError::OutOfRange`] for a threshold outside
    /// `[0, 1)`.
    pub fn new(dim: usize, threshold: f64) -> Result<Self, PhasorError> {
        if dim == 0 {
            return Err(PhasorError::Empty { what: "dimension" });
        }
        if !(0.0..1.0).contains(&threshold) {
            return Err(PhasorError::OutOfRange { what: "threshold", value: threshold, low: 0.0, high: 1.0 });
        }
        Ok(Self { dim, patterns: Vec::new(), threshold })
    }

    /// Store a pattern.
    ///
    /// # Errors
    ///
    /// [`PhasorError::Dimension`].
    pub fn store(&mut self, pattern: &Phasor) -> Result<(), PhasorError> {
        same(pattern.dim(), self.dim)?;
        self.patterns.push(pattern.clone());
        Ok(())
    }

    /// The field `W z = Σ_μ ξ^μ ⟨ξ^μ, z⟩` at `z`, as `(re, im)` per element, computed through the
    /// patterns in `O(P D)` rather than through the `D × D` matrix.
    ///
    /// # Errors
    ///
    /// [`PhasorError::Dimension`], [`PhasorError::Empty`] with nothing stored.
    pub fn field(&self, z: &Phasor) -> Result<Bundle, PhasorError> {
        if self.patterns.is_empty() {
            return Err(PhasorError::Empty { what: "patterns" });
        }
        same(z.dim(), self.dim)?;
        let d = self.dim;
        let (mut re, mut im) = (vec![0.0; d], vec![0.0; d]);
        for p in &self.patterns {
            // ⟨ξ, z⟩ = Σ_k e^{i(z_k − ξ_k)} — the overlap, a complex number.
            let (mut or, mut oi) = (0.0, 0.0);
            for (a, b) in p.phase.iter().zip(&z.phase) {
                or += (b - a).cos();
                oi += (b - a).sin();
            }
            for (k, &a) in p.phase.iter().enumerate() {
                // ξ_k · overlap
                let (c, s) = (a.cos(), a.sin());
                re[k] += c * or - s * oi;
                im[k] += s * or + c * oi;
            }
        }
        Ok(Bundle { re, im })
    }

    /// Iterate `z ← ph(W z)` from `cue` until the phases stop moving or `max_iters`.
    ///
    /// # Errors
    ///
    /// As [`Tpam::field`].
    pub fn retrieve(&self, cue: &Phasor, max_iters: usize) -> Result<Retrieval, PhasorError> {
        let mut state = cue.clone();
        let floor = self.threshold * self.dim as f64;
        let mut iterations = 0;
        let mut converged = false;
        let mut silent = 0;
        let mut last_move = f64::INFINITY;
        while iterations < max_iters {
            iterations += 1;
            let f = self.field(&state)?;
            let mut moved = 0.0f64;
            silent = 0;
            let mut next = Vec::with_capacity(self.dim);
            for k in 0..self.dim {
                let mag = (f.re[k] * f.re[k] + f.im[k] * f.im[k]).sqrt();
                let p = if mag >= floor {
                    wrap(f.im[k].atan2(f.re[k]))
                } else {
                    silent += 1;
                    0.0
                };
                let delta = wrap(p - state.phase[k]);
                let delta = delta.min(core::f64::consts::TAU - delta);
                moved = moved.max(delta);
                next.push(p);
            }
            state = Phasor { phase: next };
            last_move = moved;
            if moved < 1e-9 {
                converged = true;
                break;
            }
        }
        Ok(Retrieval { state, iterations, converged, last_move, silent })
    }

    /// The crosstalk-to-signal ratio a stored pattern's field carries: `√((P − 1)/D)`. Below
    /// about `0.3` the stored patterns are fixed points to within a few degrees.
    #[must_use]
    pub fn crosstalk_ratio(&self) -> f64 {
        ((self.patterns.len().saturating_sub(1)) as f64 / self.dim as f64).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::{Bundle, Phasor, PhasorError, Tpam, noise_sd};
    use crate::rng::Rng;
    use core::f64::consts::{PI, TAU};

    fn phase_error(a: &Phasor, b: &Phasor) -> f64 {
        a.phase
            .iter()
            .zip(&b.phase)
            .map(|(x, y)| {
                let d = (x - y).rem_euclid(TAU);
                d.min(TAU - d)
            })
            .fold(0.0, f64::max)
    }

    /// The algebra: binding and unbinding are exact to 1e-12, the conjugate is the inverse, the
    /// permutation undoes itself, self-similarity is exactly 1, and strangers are `1/√(2D)`.
    #[test]
    fn binding_is_exactly_invertible_and_strangers_are_orthogonal_to_the_stated_noise() {
        let mut rng = Rng::new(1);
        let (a, b) = (Phasor::random(1000, &mut rng).unwrap(), Phasor::random(1000, &mut rng).unwrap());
        let ab = a.bind(&b).unwrap();
        assert!(phase_error(&ab.unbind(&b).unwrap(), &a) < 1e-12);
        assert!(phase_error(&ab.bind(&b.conjugate()).unwrap(), &a) < 1e-12);
        assert!(phase_error(&a.permute(7).permute(-7), &a) < 1e-15);
        assert_eq!(a.similarity(&a).unwrap(), 1.0);
        assert!(a.similarity(&ab).unwrap().abs() < 5.0 * noise_sd(1000), "a bound pair resembles its input");
        let identity = a.bind(&a.conjugate()).unwrap();
        assert!(identity.phase.iter().all(|p| *p < 1e-12 || (TAU - p) < 1e-12));
        // The noise floor, measured over five hundred pairs at three dimensions.
        for d in [64usize, 256, 1024] {
            let sims: Vec<f64> = (0..500)
                .map(|_| Phasor::random(d, &mut rng).unwrap().similarity(&Phasor::random(d, &mut rng).unwrap()).unwrap())
                .collect();
            let mean = sims.iter().sum::<f64>() / 500.0;
            let sd = (sims.iter().map(|s| (s - mean) * (s - mean)).sum::<f64>() / 499.0).sqrt();
            assert!((sd / noise_sd(d) - 1.0).abs() < 0.12, "D {d}: sd {sd} vs 1/sqrt(2D) = {}", noise_sd(d));
            assert!(mean.abs() < 4.0 * noise_sd(d) / (500f64).sqrt() * 2.0 + 1e-3);
        }
        assert!(noise_sd(0).is_nan());
    }

    /// A sum-bundle is similar to each member by `1` in expectation with noise `√((k−1)/(2D))`,
    /// and to a stranger by `0` — measured at D = 4000 over the members of a 7-bundle.
    #[test]
    fn a_bundle_is_similar_to_each_member_by_one_with_the_stated_noise() {
        let mut rng = Rng::new(2);
        let d = 4000;
        let mut sims = Vec::new();
        for _ in 0..12 {
            let members: Vec<Phasor> = (0..7).map(|_| Phasor::random(d, &mut rng).unwrap()).collect();
            let bundle = Bundle::sum(&members).unwrap();
            for m in &members {
                sims.push(bundle.similarity(m).unwrap());
            }
            let stranger = Phasor::random(d, &mut rng).unwrap();
            assert!(bundle.similarity(&stranger).unwrap().abs() < 5.0 * (7.0f64 / (2.0 * d as f64)).sqrt());
        }
        let mean = sims.iter().sum::<f64>() / sims.len() as f64;
        let sd = (sims.iter().map(|s| (s - mean) * (s - mean)).sum::<f64>() / (sims.len() as f64 - 1.0)).sqrt();
        let want_sd = (6.0f64 / (2.0 * d as f64)).sqrt();
        assert!((mean - 1.0).abs() < 4.0 * want_sd / (sims.len() as f64).sqrt() + 1e-3, "mean {mean}");
        assert!((sd / want_sd - 1.0).abs() < 0.25, "sd {sd} vs {want_sd}");
        // The bundle's phases read back a member with a similarity well above the noise.
        let members: Vec<Phasor> = (0..3).map(|_| Phasor::random(d, &mut rng).unwrap()).collect();
        let read = Bundle::sum(&members).unwrap().phases();
        assert!(read.similarity(&members[0]).unwrap() > 0.4);
        assert!(matches!(Bundle::sum(&[]), Err(PhasorError::Empty { what: "bundle" })));
    }

    /// Binding by spike-time delays is binding by phases: delay each element's spike by the
    /// other symbol's spike time, wrap into the cycle, and the result is the phase product.
    #[test]
    fn binding_by_spike_delays_is_binding_by_phases() {
        let mut rng = Rng::new(3);
        let period = 25e-3; // a 40 Hz rhythm
        let (a, b) = (Phasor::random(500, &mut rng).unwrap(), Phasor::random(500, &mut rng).unwrap());
        let ta = a.spike_times(period).unwrap();
        let tb = b.spike_times(period).unwrap();
        assert!(ta.iter().all(|t| (0.0..period).contains(t)));
        let delayed: Vec<f64> = ta.iter().zip(&tb).map(|(x, y)| (x + y) % period).collect();
        let via_spikes = Phasor::from_spike_times(&delayed, period).unwrap();
        let via_phase = a.bind(&b).unwrap();
        assert!(phase_error(&via_spikes, &via_phase) < 1e-9);
        // And the times round-trip.
        assert!(phase_error(&Phasor::from_spike_times(&ta, period).unwrap(), &a) < 1e-12);
        assert!(matches!(a.spike_times(0.0), Err(PhasorError::OutOfRange { what: "period", .. })));
        assert!(matches!(Phasor::from_spike_times(&[], period), Err(PhasorError::Empty { .. })));
        assert!(matches!(Phasor::from_spike_times(&[f64::NAN], period), Err(PhasorError::NonFinite { index: 0 })));
    }

    /// The memory: with `√((P−1)/D) = 0.15` every stored pattern is a fixed point to within the
    /// phase error the crosstalk predicts (a few degrees), a cue with a fifth of its phases
    /// randomised converges to its pattern, and a random cue lands nowhere near any pattern.
    #[test]
    fn stored_patterns_are_fixed_points_and_a_corrupted_cue_converges() {
        let mut rng = Rng::new(4);
        let (d, p) = (400usize, 10usize);
        let mut mem = Tpam::new(d, 0.1).unwrap();
        let patterns: Vec<Phasor> = (0..p).map(|_| Phasor::random(d, &mut rng).unwrap()).collect();
        for q in &patterns {
            mem.store(q).unwrap();
        }
        assert!((mem.crosstalk_ratio() - 0.15).abs() < 1e-12);
        // A stored pattern's field: D ξ^ν plus crosstalk of magnitude about sqrt((P−1) D) per
        // element — a ratio of 0.15, so a per-element phase error of about 0.1 rad at one sigma.
        // The similarity to the pattern is the mean cosine of those errors, 1 − σ²/2 ≈ 0.99; the
        // WORST element over four hundred is several sigma out (0.67 rad was measured), so the
        // bound on it is loose and the bound on the mean is tight.
        for (mu, q) in patterns.iter().enumerate() {
            let r = mem.retrieve(q, 50).unwrap();
            let sim = r.state.similarity(q).unwrap();
            assert!(sim > 0.95, "pattern {mu} settled at similarity {sim} ({} iterations)", r.iterations);
            let err = phase_error(&r.state, q);
            assert!(err < 1.2, "pattern {mu}: worst element moved by {err} rad");
            // The map contracts slowly near the fixed point and is not required to reach 1e-9 in
            // fifty iterations; it is required to be nearly still.
            assert!(r.last_move < 1e-3, "pattern {mu} was still moving by {} rad", r.last_move);
            assert_eq!(r.silent, 0);
        }
        // A corrupted cue: eighty elements re-drawn.
        let mut cue = patterns[3].clone();
        for k in 0..80 {
            cue.phase[k] = TAU * rng.next_f64();
        }
        let before = cue.similarity(&patterns[3]).unwrap();
        let r = mem.retrieve(&cue, 50).unwrap();
        let after = r.state.similarity(&patterns[3]).unwrap();
        assert!(before < 0.85, "the corruption did nothing: {before}");
        assert!(after > 0.95, "retrieval reached similarity {after} from {before}");
        for (mu, q) in patterns.iter().enumerate() {
            if mu != 3 {
                assert!(r.state.similarity(q).unwrap() < 0.3, "converged toward pattern {mu} instead");
            }
        }
        // A random cue at a low threshold lands in SOME basin — an associative memory with ten
        // patterns in four hundred dimensions has no empty space; the first draft asserted it
        // would resemble nothing and it retrieved a pattern at 0.956. Rejecting a cue that matches
        // nothing is the THRESHOLD's job: a random cue's field has magnitude about √(P·D) = 63 per
        // element (0.16 D) with a Rayleigh tail, a stored pattern's about D minus that same
        // crosstalk, so a threshold of 0.35 D silences all but a few elements of the one and none
        // of the other. (0.5 D was tried first: one element of a stored pattern fell under it —
        // the crosstalk's tail is fatter than a single-sigma estimate, because each pattern pair's
        // overlap is itself Rayleigh.) The fractions are asserted, not the exact counts.
        let noise = Phasor::random(d, &mut rng).unwrap();
        let r = mem.retrieve(&noise, 50).unwrap();
        let best = patterns.iter().map(|q| r.state.similarity(q).unwrap()).fold(f64::NEG_INFINITY, f64::max);
        assert!(best > 0.9, "at threshold 0.1 a random cue must fall into a basin: best similarity {best}");
        let strict = Tpam { threshold: 0.35, ..mem.clone() };
        let r = strict.retrieve(&noise, 10).unwrap();
        assert!(r.silent >= d * 95 / 100, "a threshold of 0.35 D let {} elements of a random cue through", d - r.silent);
        let r = strict.retrieve(&patterns[0], 50).unwrap();
        assert!(r.silent <= d / 100, "a stored pattern's field is near D, and {} elements fell under 0.35 D", r.silent);
        assert!(r.state.similarity(&patterns[0]).unwrap() > 0.95);
        let r = strict.retrieve(&cue, 50).unwrap();
        assert!(r.silent <= d / 100, "a cue with 80% of its phases right has a field near 0.8 D; {} elements fell under", r.silent);
        // A silenced element has no phase; this type carries none, so it reads as phase zero and
        // costs the similarity up to 2/D each. The bound is the arithmetic of that, not a knob.
        let sim = r.state.similarity(&patterns[3]).unwrap();
        assert!(sim > 0.95 - 2.0 * r.silent as f64 / d as f64, "similarity {sim} with {} silenced", r.silent);
        assert!(matches!(Tpam::new(d, 1.0), Err(PhasorError::OutOfRange { what: "threshold", .. })));
        assert!(matches!(Tpam::new(d, 0.5).unwrap().field(&noise), Err(PhasorError::Empty { what: "patterns" })));
    }

    /// Every refusal names the problem.
    #[test]
    fn the_refusals_name_the_problem() {
        assert!(matches!(Phasor::new(&[]), Err(PhasorError::Empty { what: "phase" })));
        assert!(matches!(Phasor::new(&[0.0, f64::INFINITY]), Err(PhasorError::NonFinite { index: 1 })));
        let p = Phasor::new(&[7.0, -1.0]).unwrap();
        assert!((p.phase[0] - (7.0 - TAU)).abs() < 1e-12 && (p.phase[1] - (TAU - 1.0)).abs() < 1e-12, "phases wrap into [0, 2π)");
        let mut rng = Rng::new(1);
        assert!(matches!(Phasor::random(0, &mut rng), Err(PhasorError::Empty { .. })));
        let q = Phasor::random(3, &mut rng).unwrap();
        assert!(matches!(p.bind(&q), Err(PhasorError::Dimension { a: 2, b: 3 })));
        assert!(matches!(p.similarity(&q), Err(PhasorError::Dimension { .. })));
        let mut mem = Tpam::new(2, 0.1).unwrap();
        assert!(matches!(mem.store(&q), Err(PhasorError::Dimension { .. })));
        mem.store(&p).unwrap();
        assert!(matches!(mem.retrieve(&q, 5), Err(PhasorError::Dimension { .. })));
        assert!((Phasor::new(&[PI]).unwrap().similarity(&Phasor::new(&[0.0]).unwrap()).unwrap() + 1.0).abs() < 1e-15);
        for e in [
            PhasorError::Empty { what: "x" },
            PhasorError::Dimension { a: 1, b: 2 },
            PhasorError::NonFinite { index: 0 },
            PhasorError::OutOfRange { what: "w", value: 9.0, low: 0.0, high: 1.0 },
        ] {
            assert!(!e.to_string().is_empty());
        }
    }


    /// `rem_euclid` of a tiny negative angle rounds UP to a full turn, which is not in `[0, 2π)`;
    /// the guard that folds it to zero was removed by the second mutation sweep and nothing
    /// noticed.
    #[test]
    fn wrap_never_returns_a_full_turn() {
        let tau = core::f64::consts::TAU;
        assert_eq!((-1e-20f64).rem_euclid(tau), tau, "the premise: rem_euclid does round up");
        assert_eq!(super::wrap(-1e-20), 0.0);
        assert_eq!(super::wrap(tau), 0.0);
        assert_eq!(super::wrap(-core::f64::consts::PI), core::f64::consts::PI);
        assert_eq!(super::wrap(0.25), 0.25);
    }

    /// Unbinding refuses two symbols of different dimensions instead of working on the shorter.
    ///
    /// Why the suite could not see it: the dimension refusals it checks are `bind`'s and
    /// `similarity`'s, and every `unbind` call it makes is on a pair built from the same
    /// dimension, so the check could be deleted and `zip` would quietly truncate to the shorter of
    /// the two — returning a symbol of the wrong width with no complaint.
    #[test]
    fn unbinding_refuses_two_symbols_of_different_dimensions() {
        let two = Phasor::new(&[0.3, 0.4]).unwrap();
        let three = Phasor::new(&[0.3, 0.4, 0.5]).unwrap();
        assert!(matches!(two.unbind(&three), Err(PhasorError::Dimension { a: 2, b: 3 })));
        assert!(matches!(three.unbind(&two), Err(PhasorError::Dimension { a: 3, b: 2 })));
    }

    /// The permutation is the cyclic shift to the RIGHT by the shift it is given: element `k` of
    /// the result is element `k − shift` of the input, indices taken around the ring.
    ///
    /// Why the suite could not see it: its only permutation assertion is that `permute(7)` and
    /// `permute(-7)` undo each other. That holds for a shift to the left, and it holds for a
    /// permutation that moves nothing at all, so neither direction nor magnitude was pinned.
    #[test]
    fn the_permutation_shifts_right_by_the_shift_it_is_given() {
        let p = Phasor::new(&[0.1, 0.2, 0.3, 0.4]).unwrap();
        assert_eq!(p.permute(1).phase, vec![0.4, 0.1, 0.2, 0.3], "one place to the right");
        assert_eq!(p.permute(-1).phase, vec![0.2, 0.3, 0.4, 0.1], "a negative shift goes left");
        assert_eq!(p.permute(5).phase, vec![0.4, 0.1, 0.2, 0.3], "shifts are taken around the ring");
        assert_eq!(p.permute(4).phase, p.phase, "a whole turn is the identity");
        assert_eq!(p.permute(0).phase, p.phase);
    }

    /// Reading spike times back refuses a period that is not a rhythm — zero, negative, infinite
    /// or not a number — exactly as writing them out does.
    ///
    /// Why the suite could not see it: the only period refusal it asserts is on `spike_times`, the
    /// forward direction. `from_spike_times` is called three times, always with the same 40 Hz
    /// period, and its other two refusals (an empty train, a time that is not finite) are reached
    /// after the period check, so they still fired with the period guard removed.
    #[test]
    fn reading_spike_times_back_refuses_a_period_that_is_not_a_rhythm() {
        for bad in [0.0, -25e-3, f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
            let built = Phasor::from_spike_times(&[1e-3, 2e-3], bad);
            assert!(
                matches!(built, Err(PhasorError::OutOfRange { what: "period", .. })),
                "a period of {bad} was accepted as a rhythm"
            );
        }
        // The guard runs BEFORE the emptiness and finiteness checks, so a bad period is reported
        // as a bad period even when the train is also wrong.
        assert!(matches!(Phasor::from_spike_times(&[], 0.0), Err(PhasorError::OutOfRange { .. })));
    }

    /// A bundle refuses members of another dimension, and refuses to be compared with a symbol of
    /// another dimension, rather than summing or scoring whatever overlaps.
    ///
    /// Why the suite could not see it: every bundle it builds is built from symbols drawn at one
    /// dimension and compared against symbols of that same dimension. The only `Bundle` refusal it
    /// asserts is the empty-set one. A dropped dimension check is then invisible on the shorter
    /// side (the loop simply stops early) and would panic on the longer, which no test reaches.
    #[test]
    fn a_bundle_refuses_members_and_probes_of_another_dimension() {
        let two = Phasor::new(&[0.3, 0.4]).unwrap();
        let three = Phasor::new(&[0.3, 0.4, 0.5]).unwrap();
        assert!(matches!(Bundle::sum(&[three.clone(), two.clone()]), Err(PhasorError::Dimension { a: 2, b: 3 })));
        assert!(matches!(Bundle::sum(&[two.clone(), three.clone()]), Err(PhasorError::Dimension { a: 3, b: 2 })));
        let wide = Bundle::sum(core::slice::from_ref(&three)).unwrap();
        let narrow = Bundle::sum(core::slice::from_ref(&two)).unwrap();
        assert!(matches!(wide.similarity(&two), Err(PhasorError::Dimension { a: 3, b: 2 })));
        assert!(matches!(narrow.similarity(&three), Err(PhasorError::Dimension { a: 2, b: 3 })));
    }

    /// A memory refuses a dimension of zero and a threshold outside `[0, 1)`, and keeps the
    /// threshold it was given — which is what makes it silence anything.
    ///
    /// Why the suite could not see it: it builds memories only through dimensions and thresholds
    /// that are already admissible, its one threshold refusal is at the upper end (`1.0`), and the
    /// one memory whose silencing it exercises is assembled by a struct literal
    /// (`Tpam { threshold: 0.35, .. }`) rather than by the constructor — so a constructor that
    /// threw the threshold away silenced nothing and no assertion moved.
    #[test]
    fn a_memory_refuses_a_zero_dimension_and_an_out_of_range_threshold_and_keeps_the_one_it_took() {
        assert!(matches!(Tpam::new(0, 0.1), Err(PhasorError::Empty { what: "dimension" })));
        assert!(matches!(Tpam::new(4, -0.5), Err(PhasorError::OutOfRange { what: "threshold", .. })));
        assert!(matches!(Tpam::new(4, -1e-300), Err(PhasorError::OutOfRange { what: "threshold", .. })));
        assert!(matches!(Tpam::new(4, 1.0), Err(PhasorError::OutOfRange { what: "threshold", .. })));
        assert_eq!(Tpam::new(4, 0.35).unwrap().threshold, 0.35, "the memory kept a different threshold");
        assert_eq!(Tpam::new(4, 0.0).unwrap().threshold, 0.0);
        // And the threshold the constructor kept is the one that silences: one pattern in
        // sixty-four dimensions gives a random cue a field of magnitude about √D = 8, far under a
        // floor of 0.5 D = 32, so every element is silenced.
        let mut rng = Rng::new(21);
        let mut strict = Tpam::new(64, 0.5).unwrap();
        strict.store(&Phasor::random(64, &mut rng).unwrap()).unwrap();
        let r = strict.retrieve(&Phasor::random(64, &mut rng).unwrap(), 1).unwrap();
        assert_eq!(r.silent, 64, "a constructed threshold of 0.5 D silenced nothing");
    }

    /// The overlap `⟨ξ, z⟩` is a COMPLEX number and the field carries both of its parts, so the
    /// field turns with the cue's phase: a memory holding one pattern reproduces a cue that is
    /// that pattern rotated bodily, rather than snapping back to the pattern itself.
    ///
    /// Why the suite could not see it: at a stored pattern the overlap is real by construction
    /// (`Σ sin(0) = 0`), and the crosstalk terms of the ten-pattern memory carry imaginary parts
    /// small enough that dropping them leaves every similarity inside the tolerances the fixed
    /// point test allows. Nothing in the suite called `field` and read a number out of it.
    #[test]
    fn the_field_carries_the_imaginary_part_of_the_overlap() {
        // A two-element memory holding the all-zero pattern. The overlap with a cue at phases
        // (0, π/2) is 1 + e^{iπ/2}, whose imaginary part is exactly 1, and the field of a pattern
        // whose own phases are zero is that overlap unrotated.
        let mut mem = Tpam::new(2, 0.0).unwrap();
        mem.store(&Phasor::new(&[0.0, 0.0]).unwrap()).unwrap();
        let f = mem.field(&Phasor::new(&[0.0, PI / 2.0]).unwrap()).unwrap();
        assert_eq!(f.im, vec![1.0, 1.0], "the imaginary part of the overlap was dropped");
        // And the consequence: a bodily rotation of the one stored pattern is a fixed point,
        // because the overlap is D·e^{iδ} and the field is the pattern turned by δ.
        let mut rng = Rng::new(31);
        let xi = Phasor::random(64, &mut rng).unwrap();
        let mut one = Tpam::new(64, 0.1).unwrap();
        one.store(&xi).unwrap();
        let delta = 0.7;
        let turned = Phasor::new(&xi.phase.iter().map(|p| p + delta).collect::<Vec<f64>>()).unwrap();
        let r = one.retrieve(&turned, 20).unwrap();
        assert!(r.state.similarity(&turned).unwrap() > 1.0 - 1e-12, "a turned pattern is a fixed point");
        let back = r.state.similarity(&xi).unwrap();
        assert!((back - delta.cos()).abs() < 1e-9, "similarity to the unturned pattern {back} vs cos(0.7)");
    }

    /// `max_iters` is a cap on the iterations RUN, and a state that has stopped moving is reported
    /// as converged.
    ///
    /// Why the suite could not see it: every retrieval it runs is given fifty iterations and
    /// stopped by the tolerance long before the cap, so one iteration more or fewer changed
    /// nothing; and the ten-pattern memory it uses never reaches the 1e-9 movement that sets the
    /// flag, so `converged` is false in every retrieval the suite performs and is never asserted.
    #[test]
    fn the_cap_counts_the_iterations_run_and_a_fixed_point_reports_convergence() {
        let mut rng = Rng::new(41);
        let xi = Phasor::random(64, &mut rng).unwrap();
        let mut mem = Tpam::new(64, 0.1).unwrap();
        mem.store(&xi).unwrap();
        let idle = mem.retrieve(&xi, 0).unwrap();
        assert_eq!(idle.iterations, 0, "a cap of zero iterations ran one anyway");
        assert!(!idle.converged);
        assert_eq!(idle.last_move, f64::INFINITY);
        assert_eq!(idle.state, xi, "a cap of zero must return the cue untouched");
        let one = mem.retrieve(&xi, 1).unwrap();
        assert_eq!(one.iterations, 1, "a cap of one iteration ran a different number");
        let r = mem.retrieve(&xi, 20).unwrap();
        assert!(r.converged, "the one stored pattern of a one-pattern memory is an exact fixed point");
        assert!(r.last_move < 1e-9, "last move {}", r.last_move);
    }

    /// The dimension refusal names the two dimensions in the order the call met them, and its
    /// message prints them that way round.
    ///
    /// Why the suite could not see it: every assertion about this variant destructures it — and
    /// the one that binds its fields at all, `Dimension { a: 2, b: 3 }`, is reached through a call
    /// whose arguments are symmetric to the reader. The message itself is only ever checked for
    /// being non-empty, so one that named the dimensions the wrong way round read as covered.
    #[test]
    fn the_dimension_refusal_names_the_two_dimensions_in_the_order_it_met_them() {
        let two = Phasor::new(&[0.3, 0.4]).unwrap();
        let three = Phasor::new(&[0.3, 0.4, 0.5]).unwrap();
        let Err(forward) = two.bind(&three) else { panic!("a 2 against a 3 must be refused") };
        assert_eq!(forward, PhasorError::Dimension { a: 2, b: 3 });
        assert_eq!(forward.to_string(), "dimension 2 against dimension 3");
        let Err(backward) = three.bind(&two) else { panic!("a 3 against a 2 must be refused") };
        assert_eq!(backward.to_string(), "dimension 3 against dimension 2");
    }
}
