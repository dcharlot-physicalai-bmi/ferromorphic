//! Grid cells: a position held as a set of phases — a hexagonal firing map, path integration
//! that is exact in phase space, and a modular code whose range is the PRODUCT of its periods
//! while its size is their sum. Each is checked against the arithmetic it claims.
//!
//! # What the mechanism is
//!
//! A grid cell in the medial entorhinal cortex fires at the vertices of a triangular lattice that
//! tiles the floor (Hafting, Fyhn, Molden, Moser and Moser, *Microstructure of a spatial map in the
//! entorhinal cortex*, Nature 436(7052):801–806, 2005). Cells come in **modules**: within one, every
//! cell shares a spacing and an orientation and differs only in the offset of its lattice, so the
//! module as a whole reports the animal's position MODULO one lattice cell — a two-dimensional
//! phase. Different modules have different spacings. The standard idealisation of one cell's map
//! is the sum of three plane waves 60° apart, [`GridModule::rate`].
//!
//! Two things follow, and both are arithmetic.
//!
//! - **Path integration is exact in phase space.** Moving by `v dt` advances the phase along wave
//!   vector `k` by `k · v dt`, whatever the path. The module never needs to know where it is, only
//!   how it moved ([`PhaseIntegrator`]).
//! - **The code is a residue number system** (Fiete, Burak and Brookings, *What grid cells convey
//!   about rat location*, Journal of Neuroscience 28(27):6858–6871, 2008). Several modules of
//!   periods `λ_1 … λ_M` together identify a position uniquely up to the least common multiple of
//!   the periods — their product, when coprime — using only `Σ λ_i` cells. A place code, one cell
//!   per position, needs as many cells as positions ([`ModularCode`]).
//!
//! # Why it is in a neuromorphic crate
//!
//! It is the brain's answer to a question every mobile robot has — how to carry a position estimate
//! forward from self-motion alone — in a form that is a handful of counters with wrap-around,
//! which is exactly what neuromorphic hardware has. It is also the companion of
//! [`crate::attractor`]: each module is, in the usual model, a continuous attractor on a torus,
//! and this module is what such attractors compute when they work.
//!
//! # The closed forms this module is checked against
//!
//! - **The map.** Rate `r_max` exactly at every lattice vertex `x₀ + n a₁ + m a₂`, `|a| = λ`, 60°
//!   apart; invariant under translation by a lattice vector and under rotation by 60° about a
//!   vertex; never negative; mean over a unit cell exactly `r_max/3`.
//! - **Path integration.** After any path the integrated phases equal the phases of the true
//!   position, so the decoded position differs from the true one by a whole lattice vector — the
//!   coefficients of that difference in the lattice basis are integers.
//! - **The modular code.** Range `lcm(λ_i)`; with periods 7, 9, 11, 13 that is 9009 positions
//!   from 40 cells. Encode–decode is the identity on the whole range (exhaustively); stepping the
//!   phases by any signed displacement equals re-encoding the moved position; phases that no
//!   position produces are refused (the generalised Chinese remainder condition).
//!
//! - **Error correction.** Use only a fraction of the range and the code becomes redundant: a
//!   position known to lie below the product of all but the two largest periods survives ANY
//!   corruption of ANY one module's phase, and [`ModularCode::correct`] recovers it — checked
//!   exhaustively, every position against every wrong phase of every module. With one spare
//!   module instead of two a single error is always noticed and never silently mis-corrected.
//!   This is the redundant-residue reading of Sreenivasan and Fiete (*Grid cells generate an
//!   analog error-correcting code for singularly precise neural computation*, Nature Neuroscience
//!   14(10):1330–1337, 2011), in the integer case where the guarantee is a theorem.
//!
//! # What this module has NOT reproduced
//!
//! - The attractor network that would HOLD these phases, or its noise. The phases here are
//!   numbers; drift, and the ANALOG error correction of Sreenivasan and Fiete — small phase noise
//!   on every module at once, corrected by the geometry of the code — are not modelled. What is
//!   here is the discrete case: one module arbitrarily wrong.
//! - Decoding two-dimensional position from several modules with real-valued spacings, where the
//!   range is set by noise rather than by a least common multiple.
//! - Any account of how grids form, anchor to landmarks, or distort in real enclosures.

use core::f64::consts::{PI, TAU};
use core::fmt;

/// What went wrong, named rather than guessed around.
#[derive(Debug, Clone, PartialEq)]
pub enum GridError {
    /// A count of zero where at least one is needed.
    Empty {
        /// What was empty.
        what: &'static str,
    },
    /// Two lengths that had to agree.
    Dimension {
        /// Which array.
        what: &'static str,
        /// Length supplied.
        got: usize,
        /// Length required.
        want: usize,
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
    /// A `NaN` or infinity.
    NonFinite {
        /// Which quantity.
        what: &'static str,
    },
    /// The least common multiple of the periods does not fit in a `u64`.
    RangeOverflow,
    /// Phases that no position produces.
    Inconsistent {
        /// Index of the first module whose phase contradicts the ones before it.
        module: usize,
    },
}

impl fmt::Display for GridError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty { what } => write!(f, "{what} is empty"),
            Self::Dimension { what, got, want } => write!(f, "{what} has length {got}, expected {want}"),
            Self::OutOfRange { what, value, low, high } => {
                write!(f, "{what} = {value} is outside [{low}, {high}]")
            }
            Self::NonFinite { what } => write!(f, "{what} is not finite"),
            Self::RangeOverflow => write!(f, "the least common multiple of the periods overflows u64"),
            Self::Inconsistent { module } => write!(f, "the phase of module {module} contradicts the modules before it"),
        }
    }
}

impl std::error::Error for GridError {}

fn finite2(what: &'static str, v: [f64; 2]) -> Result<[f64; 2], GridError> {
    if v[0].is_finite() && v[1].is_finite() { Ok(v) } else { Err(GridError::NonFinite { what }) }
}

// ---------------------------------------------------------------------------------------------
// One module's firing map
// ---------------------------------------------------------------------------------------------

/// A grid module: spacing, orientation, and the offset of one cell's lattice.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GridModule {
    /// Distance between neighbouring firing fields, metres.
    pub spacing: f64,
    /// Direction of the first lattice vector, radians.
    pub orientation: f64,
    /// A lattice vertex `x₀`, metres.
    pub offset: [f64; 2],
    /// Peak rate, hertz.
    pub rate_max: f64,
}

impl GridModule {
    /// Build.
    ///
    /// # Errors
    ///
    /// [`GridError::OutOfRange`] for a non-positive spacing or peak rate, [`GridError::NonFinite`]
    /// for a non-finite orientation or offset.
    pub fn new(spacing: f64, orientation: f64, offset: [f64; 2], rate_max: f64) -> Result<Self, GridError> {
        for (what, v) in [("spacing", spacing), ("rate_max", rate_max)] {
            if !(v > 0.0) || !v.is_finite() {
                return Err(GridError::OutOfRange { what, value: v, low: f64::MIN_POSITIVE, high: f64::INFINITY });
            }
        }
        if !orientation.is_finite() {
            return Err(GridError::NonFinite { what: "orientation" });
        }
        Ok(Self { spacing, orientation, offset: finite2("offset", offset)?, rate_max })
    }

    /// The two lattice vectors `a₁, a₂`: length `spacing`, at `orientation` and 60° past it.
    #[must_use]
    pub fn lattice_vectors(&self) -> [[f64; 2]; 2] {
        let at = |angle: f64| [self.spacing * angle.cos(), self.spacing * angle.sin()];
        [at(self.orientation), at(self.orientation + PI / 3.0)]
    }

    /// The three wave vectors, rad/m: magnitude `4π/(√3 λ)`, perpendicular to the three lattice
    /// directions, so that `k_i · a_j` is a whole number of turns for every pair.
    #[must_use]
    pub fn wave_vectors(&self) -> [[f64; 2]; 3] {
        let k = 4.0 * PI / (3.0f64.sqrt() * self.spacing);
        let at = |angle: f64| [k * angle.cos(), k * angle.sin()];
        let base = self.orientation + PI / 6.0;
        [at(base), at(base + PI / 3.0), at(base + 2.0 * PI / 3.0)]
    }

    /// The firing rate at `x`: `r_max · (2/3)(⅓ Σ_i cos(k_i · (x − x₀)) + ½)`, hertz, in
    /// `[0, r_max]`.
    ///
    /// # Errors
    ///
    /// [`GridError::NonFinite`] for a non-finite position.
    pub fn rate(&self, x: [f64; 2]) -> Result<f64, GridError> {
        let x = finite2("position", x)?;
        let d = [x[0] - self.offset[0], x[1] - self.offset[1]];
        let sum: f64 = self.wave_vectors().iter().map(|k| (k[0] * d[0] + k[1] * d[1]).cos()).sum();
        Ok(self.rate_max * (2.0 / 3.0) * (sum / 3.0 + 0.5))
    }
}

// ---------------------------------------------------------------------------------------------
// Path integration in phase space
// ---------------------------------------------------------------------------------------------

/// A module's position estimate as two phases, advanced by self-motion alone.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhaseIntegrator {
    /// The module whose first two wave vectors define the phases.
    pub module: GridModule,
    /// Phases `k₁·(x − x₀)` and `k₂·(x − x₀)`, radians, each wrapped into `[0, 2π)`.
    pub phases: [f64; 2],
}

fn wrap_tau(p: f64) -> f64 {
    let r = p.rem_euclid(TAU);
    if r >= TAU { 0.0 } else { r }
}

impl PhaseIntegrator {
    /// Start at the position `x`.
    ///
    /// # Errors
    ///
    /// [`GridError::NonFinite`] for a non-finite position.
    pub fn at(module: GridModule, x: [f64; 2]) -> Result<Self, GridError> {
        let x = finite2("position", x)?;
        let k = module.wave_vectors();
        let d = [x[0] - module.offset[0], x[1] - module.offset[1]];
        let phase = |i: usize| wrap_tau(k[i][0] * d[0] + k[i][1] * d[1]);
        Ok(Self { module, phases: [phase(0), phase(1)] })
    }

    /// Move with velocity `v` (m/s) for `dt` seconds: each phase advances by `k_i · v dt`.
    ///
    /// # Errors
    ///
    /// [`GridError::NonFinite`] for a non-finite velocity, [`GridError::OutOfRange`] for a
    /// non-positive `dt`.
    pub fn step(&mut self, v: [f64; 2], dt: f64) -> Result<(), GridError> {
        let v = finite2("velocity", v)?;
        if !(dt > 0.0) || !dt.is_finite() {
            return Err(GridError::OutOfRange { what: "dt", value: dt, low: f64::MIN_POSITIVE, high: f64::INFINITY });
        }
        let k = self.module.wave_vectors();
        for i in 0..2 {
            self.phases[i] = wrap_tau(self.phases[i] + (k[i][0] * v[0] + k[i][1] * v[1]) * dt);
        }
        Ok(())
    }

    /// The position the phases imply, inside the lattice cell whose corner is the module's offset:
    /// the solution of `k₁·d = φ₁`, `k₂·d = φ₂`, plus `x₀`. The true position is this plus some
    /// whole lattice vector — which one, a single module cannot say.
    #[must_use]
    pub fn position_in_cell(&self) -> [f64; 2] {
        let k = self.module.wave_vectors();
        let det = k[0][0] * k[1][1] - k[0][1] * k[1][0];
        let (p, q) = (self.phases[0], self.phases[1]);
        [
            self.module.offset[0] + (p * k[1][1] - q * k[0][1]) / det,
            self.module.offset[1] + (q * k[0][0] - p * k[1][0]) / det,
        ]
    }
}

/// The coefficients `(n, m)` of a displacement in a module's lattice basis, `d = n a₁ + m a₂`.
/// Whole numbers exactly when `d` is a lattice vector.
#[must_use]
pub fn lattice_coefficients(module: &GridModule, d: [f64; 2]) -> [f64; 2] {
    let a = module.lattice_vectors();
    let det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
    [(d[0] * a[1][1] - d[1] * a[1][0]) / det, (d[1] * a[0][0] - d[0] * a[0][1]) / det]
}

// ---------------------------------------------------------------------------------------------
// The modular code
// ---------------------------------------------------------------------------------------------

/// A one-dimensional position held as residues: module `i` keeps `x mod periods[i]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModularCode {
    /// Periods, in steps of the position's resolution; each at least 2.
    pub periods: Vec<u64>,
    /// The residues, `phases[i] < periods[i]`.
    pub phases: Vec<u64>,
}

fn gcd(a: u128, b: u128) -> u128 {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// The inverse of `a` modulo `n` by the extended Euclidean algorithm; `a` and `n` coprime, `n ≥ 1`.
fn mod_inverse(a: u128, n: u128) -> u128 {
    let (mut r0, mut r1) = (n as i128, (a % n) as i128);
    let (mut t0, mut t1) = (0i128, 1i128);
    while r1 != 0 {
        let q = r0 / r1;
        (r0, r1) = (r1, r0 - q * r1);
        (t0, t1) = (t1, t0 - q * t1);
    }
    t0.rem_euclid(n as i128) as u128
}

impl ModularCode {
    /// A code at position zero.
    ///
    /// # Errors
    ///
    /// [`GridError::Empty`] for no modules, [`GridError::OutOfRange`] for a period below 2 — a
    /// module of period 1 has one phase and says nothing — and [`GridError::RangeOverflow`] if
    /// the least common multiple of the periods does not fit in a `u64`.
    pub fn new(periods: Vec<u64>) -> Result<Self, GridError> {
        if periods.is_empty() {
            return Err(GridError::Empty { what: "modules" });
        }
        if let Some(&p) = periods.iter().find(|p| **p < 2) {
            return Err(GridError::OutOfRange { what: "period", value: p as f64, low: 2.0, high: u64::MAX as f64 });
        }
        let n = periods.len();
        let code = Self { periods, phases: vec![0; n] };
        code.range()?;
        Ok(code)
    }

    /// Positions the code distinguishes: the least common multiple of the periods.
    ///
    /// # Errors
    ///
    /// [`GridError::RangeOverflow`] past `u64` (only for a code whose public fields were edited
    /// after construction).
    pub fn range(&self) -> Result<u64, GridError> {
        let mut l: u128 = 1;
        for &p in &self.periods {
            let p = u128::from(p);
            l = l / gcd(l, p) * p;
            if l > u128::from(u64::MAX) {
                return Err(GridError::RangeOverflow);
            }
        }
        Ok(l as u64)
    }

    /// Cells the code uses, one per phase of every module: the sum of the periods. A place code
    /// for the same range needs [`ModularCode::range`] cells.
    #[must_use]
    pub fn cells(&self) -> u64 {
        self.periods.iter().fold(0u64, |a, p| a.saturating_add(*p))
    }

    /// Set the phases to those of position `x`.
    pub fn encode(&mut self, x: u64) {
        for (phase, p) in self.phases.iter_mut().zip(&self.periods) {
            *phase = x % p;
        }
    }

    /// Path-integrate a signed displacement: every module advances its own phase, with wrap-around,
    /// and none of them consults the others or the position.
    pub fn step(&mut self, dx: i64) {
        for (phase, p) in self.phases.iter_mut().zip(&self.periods) {
            let moved = i128::from(*phase) + i128::from(dx);
            *phase = moved.rem_euclid(i128::from(*p)) as u64;
        }
    }

    /// The position in `[0, range)` whose residues are the current phases, by the generalised
    /// Chinese remainder theorem.
    ///
    /// # Errors
    ///
    /// [`GridError::Dimension`] if the phases and periods differ in length,
    /// [`GridError::OutOfRange`] for a phase not below its period, [`GridError::Inconsistent`] if
    /// no position has these phases (possible only when two periods share a factor), and
    /// [`GridError::RangeOverflow`] as [`ModularCode::range`].
    pub fn decode(&self) -> Result<u64, GridError> {
        if self.phases.len() != self.periods.len() {
            return Err(GridError::Dimension { what: "phases", got: self.phases.len(), want: self.periods.len() });
        }
        self.range()?;
        let (mut x, mut m): (u128, u128) = (0, 1);
        for (i, (&phase, &p)) in self.phases.iter().zip(&self.periods).enumerate() {
            if phase >= p {
                return Err(GridError::OutOfRange { what: "phase", value: phase as f64, low: 0.0, high: (p - 1) as f64 });
            }
            let (a, n) = (u128::from(phase), u128::from(p));
            let g = gcd(m, n);
            if x % g != a % g {
                return Err(GridError::Inconsistent { module: i });
            }
            // x + m·t ≡ a (mod n)  ⇔  (m/g)·t ≡ (a − x)/g (mod n/g), and m/g is invertible there.
            // Every factor is below 2⁶⁴, so the products fit in a u128, and x stays below
            // lcm(m, n) ≤ u64::MAX.
            let reduced = n / g;
            let need = ((a + n - x % n) % n) / g;
            let t = need % reduced * mod_inverse(m / g % reduced, reduced) % reduced;
            x += m * t;
            m = m / g * n;
        }
        Ok(x as u64)
    }
}

/// What [`ModularCode::correct`] concluded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Correction {
    /// The position, below the legal range.
    pub position: u64,
    /// The module whose phase was wrong and has been disregarded; `None` if all agreed.
    pub repaired: Option<usize>,
}

impl ModularCode {
    /// The largest legal range over which ANY single-module error can be corrected: the product of
    /// all periods but the two largest — two spare modules give the code a minimum distance of
    /// three. `None` unless the periods are pairwise coprime and there are at least three.
    #[must_use]
    pub fn single_error_range(&self) -> Option<u64> {
        let k = self.periods.len();
        if k < 3 {
            return None;
        }
        for i in 0..k {
            for j in (i + 1)..k {
                if gcd(u128::from(self.periods[i]), u128::from(self.periods[j])) != 1 {
                    return None;
                }
            }
        }
        let mut sorted = self.periods.clone();
        sorted.sort_unstable();
        sorted[..k - 2].iter().try_fold(1u64, |acc, p| acc.checked_mul(*p))
    }

    fn without(&self, skip: usize) -> Self {
        let keep = |v: &[u64]| v.iter().enumerate().filter(|(i, _)| *i != skip).map(|(_, x)| *x).collect();
        Self { periods: keep(&self.periods), phases: keep(&self.phases) }
    }

    /// Decode a position known to lie below `legal_range`, tolerating one module whose phase is
    /// arbitrarily wrong. If the phases as they stand decode to a legal position, that is the
    /// answer; otherwise each module is set aside in turn, and the answer is the legal position
    /// the others agree on — if exactly one exists.
    ///
    /// # Errors
    ///
    /// [`GridError::OutOfRange`] for a legal range of zero or past the code's range, or a phase not
    /// below its period; [`GridError::Dimension`] for mismatched lengths;
    /// [`GridError::Inconsistent`] (naming module `0`) when no single module can be blamed, or
    /// more than one can — the corruption is then detected and NOT guessed at.
    pub fn correct(&self, legal_range: u64) -> Result<Correction, GridError> {
        let range = self.range()?;
        if legal_range == 0 || legal_range > range {
            return Err(GridError::OutOfRange { what: "legal_range", value: legal_range as f64, low: 1.0, high: range as f64 });
        }
        match self.decode() {
            Ok(x) if x < legal_range => return Ok(Correction { position: x, repaired: None }),
            Ok(_) | Err(GridError::Inconsistent { .. }) => {}
            Err(e) => return Err(e),
        }
        let mut found: Option<Correction> = None;
        for skip in 0..self.periods.len() {
            if self.periods.len() < 2 {
                break;
            }
            if let Ok(x) = self.without(skip).decode()
                && x < legal_range
            {
                match found {
                    Some(c) if c.position != x => return Err(GridError::Inconsistent { module: 0 }),
                    Some(_) => {}
                    None => found = Some(Correction { position: x, repaired: Some(skip) }),
                }
            }
        }
        found.ok_or(GridError::Inconsistent { module: 0 })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    fn module() -> GridModule {
        GridModule::new(0.45, 0.2, [0.13, -0.08], 12.0).unwrap()
    }

    #[test]
    fn the_map_peaks_on_a_triangular_lattice_of_the_stated_spacing() {
        let m = module();
        let a = m.lattice_vectors();
        for v in &a {
            assert!((v[0].hypot(v[1]) - 0.45).abs() < 1e-15);
        }
        let between = (a[0][0] * a[1][0] + a[0][1] * a[1][1]) / (0.45 * 0.45);
        assert!((between - 0.5).abs() < 1e-15, "the lattice vectors are 60° apart: cos = {between}");
        let k = m.wave_vectors();
        assert!((k[0][0].hypot(k[0][1]) - 4.0 * PI / (3.0f64.sqrt() * 0.45)).abs() < 1e-12);
        // Every wave vector turns a whole number of times along every lattice vector.
        for ki in &k {
            for aj in &a {
                let turns = (ki[0] * aj[0] + ki[1] * aj[1]) / TAU;
                assert!((turns - turns.round()).abs() < 1e-12, "k·a = {turns} turns");
            }
        }
        for (n, mm) in [(0, 0), (1, 0), (0, 1), (-2, 3), (5, -4)] {
            let x = [m.offset[0] + f64::from(n) * a[0][0] + f64::from(mm) * a[1][0], m.offset[1] + f64::from(n) * a[0][1] + f64::from(mm) * a[1][1]];
            assert!((m.rate(x).unwrap() - 12.0).abs() < 1e-11, "vertex ({n}, {mm}) fires at {}", m.rate(x).unwrap());
        }
        // Translation by a lattice vector and rotation by 60° about a vertex change nothing.
        let mut rng = Rng::new(6);
        let (c, s) = ((PI / 3.0).cos(), (PI / 3.0).sin());
        let mut lowest = f64::INFINITY;
        for _ in 0..500 {
            let d = [rng.next_f64() - 0.5, rng.next_f64() - 0.5];
            let x = [m.offset[0] + d[0], m.offset[1] + d[1]];
            let r = m.rate(x).unwrap();
            let shifted = m.rate([x[0] + a[1][0] - 2.0 * a[0][0], x[1] + a[1][1] - 2.0 * a[0][1]]).unwrap();
            let turned = m.rate([m.offset[0] + c * d[0] - s * d[1], m.offset[1] + s * d[0] + c * d[1]]).unwrap();
            assert!((r - shifted).abs() < 1e-11 && (r - turned).abs() < 1e-11);
            assert!((-1e-12..=12.0 + 1e-12).contains(&r), "rate {r} outside [0, r_max]");
            lowest = lowest.min(r);
        }
        assert!(lowest < 0.5, "the map does go quiet between fields: lowest seen {lowest}");
        // The mean over one lattice cell is a third of the peak: the three cosines average to zero.
        let grid = 300;
        let mut mean = 0.0;
        for i in 0..grid {
            for j in 0..grid {
                let (u, v) = ((f64::from(i) + 0.5) / f64::from(grid), (f64::from(j) + 0.5) / f64::from(grid));
                mean += m.rate([m.offset[0] + u * a[0][0] + v * a[1][0], m.offset[1] + u * a[0][1] + v * a[1][1]]).unwrap();
            }
        }
        assert!((mean / f64::from(grid * grid) - 4.0).abs() < 1e-9);
    }

    #[test]
    fn path_integration_lands_on_the_true_position_modulo_the_lattice() {
        let m = module();
        let mut rng = Rng::new(15);
        let mut x = [0.31, 0.27];
        let mut est = PhaseIntegrator::at(m, x).unwrap();
        let dt = 0.02;
        for step in 0..5000 {
            let v = [1.2 * (rng.next_f64() - 0.5), 1.2 * (rng.next_f64() - 0.5)];
            x = [x[0] + v[0] * dt, x[1] + v[1] * dt];
            est.step(v, dt).unwrap();
            if step % 250 == 0 || step == 4999 {
                let seen = est.position_in_cell();
                let coeff = lattice_coefficients(&m, [x[0] - seen[0], x[1] - seen[1]]);
                for c in coeff {
                    // Each phase has been added to 5000 times at a size of a few radians.
                    assert!((c - c.round()).abs() < 1e-9, "step {step}: off the lattice by {}", c - c.round());
                }
                // And the phases ARE the phases of the true position.
                let direct = PhaseIntegrator::at(m, x).unwrap();
                for i in 0..2 {
                    let gap = (est.phases[i] - direct.phases[i]).abs();
                    assert!(gap.min(TAU - gap) < 1e-9);
                }
            }
        }
        assert!(x[0].hypot(x[1]) > 0.45, "the walk never left the first lattice cell, so the wrap was never exercised");
        // A pure lattice translation leaves the phases where they were.
        let a = m.lattice_vectors();
        let mut there = PhaseIntegrator::at(m, [0.2, 0.1]).unwrap();
        let before = there.phases;
        there.step([a[0][0] + 2.0 * a[1][0], a[0][1] + 2.0 * a[1][1]], 1.0).unwrap();
        for i in 0..2 {
            let gap = (there.phases[i] - before[i]).abs();
            assert!(gap.min(TAU - gap) < 1e-12);
        }
        assert_eq!(lattice_coefficients(&m, [2.0 * a[0][0] - 3.0 * a[1][0], 2.0 * a[0][1] - 3.0 * a[1][1]]).map(f64::round), [2.0, -3.0]);
    }

    #[test]
    fn four_small_modules_distinguish_nine_thousand_positions() {
        let mut code = ModularCode::new(vec![7, 9, 11, 13]).unwrap();
        assert_eq!(code.range().unwrap(), 9009);
        assert_eq!(code.cells(), 40);
        for x in 0..9009u64 {
            code.encode(x);
            assert_eq!(code.decode().unwrap(), x);
        }
        // One past the range is indistinguishable from zero: that is what a range is.
        code.encode(9009);
        assert_eq!(code.decode().unwrap(), 0);
        // Path integration: each module steps alone, and together they track the position.
        let mut rng = Rng::new(21);
        let mut x: i64 = 4000;
        code.encode(4000);
        for _ in 0..2000 {
            let dx = i64::from(rng.below(601)) - 300;
            x += dx;
            code.step(dx);
            assert_eq!(code.decode().unwrap(), x.rem_euclid(9009) as u64);
        }
        code.step(i64::MIN);
        code.step(i64::MAX);
        code.step(1);
        assert_eq!(code.decode().unwrap(), x.rem_euclid(9009) as u64, "extreme steps cancel without overflow");
    }

    #[test]
    fn periods_that_share_a_factor_have_a_shorter_range_and_impossible_phases() {
        let mut code = ModularCode::new(vec![4, 6]).unwrap();
        assert_eq!(code.range().unwrap(), 12, "lcm, not the product 24");
        for x in 0..12u64 {
            code.encode(x);
            assert_eq!(code.decode().unwrap(), x);
        }
        // x ≡ 1 (mod 4) is odd and x ≡ 2 (mod 6) is even: no position has these phases.
        code.phases = vec![1, 2];
        assert_eq!(code.decode(), Err(GridError::Inconsistent { module: 1 }));
        // Exactly half of the 24 phase pairs are reachable.
        let mut reachable = 0;
        for a in 0..4 {
            for b in 0..6 {
                code.phases = vec![a, b];
                reachable += u32::from(code.decode().is_ok());
            }
        }
        assert_eq!(reachable, 12);
        code.phases = vec![4, 0];
        assert!(matches!(code.decode(), Err(GridError::OutOfRange { what: "phase", .. })));
        code.phases = vec![0];
        assert!(matches!(code.decode(), Err(GridError::Dimension { what: "phases", got: 1, want: 2 })));
    }

    #[test]
    fn two_spare_modules_correct_any_single_wrong_phase() {
        let mut code = ModularCode::new(vec![13, 7, 11, 9]).unwrap();
        // All but the two largest: 7 · 9.
        assert_eq!(code.single_error_range(), Some(63));
        let mut cases = 0;
        for x in 0..63u64 {
            code.encode(x);
            assert_eq!(code.correct(63).unwrap(), Correction { position: x, repaired: None });
            let clean = code.phases.clone();
            for module in 0..4 {
                for wrong in 0..code.periods[module] {
                    if wrong == clean[module] {
                        continue;
                    }
                    code.phases[module] = wrong;
                    assert_eq!(code.correct(63).unwrap(), Correction { position: x, repaired: Some(module) }, "x = {x}, module {module} set to {wrong}");
                    cases += 1;
                }
                code.phases[module] = clean[module];
            }
        }
        assert_eq!(cases, 63 * (12 + 6 + 10 + 8));
        // The legal range is half-open: position 63 itself is OUTSIDE it, and an uncorrupted code
        // word that says 63 is refused rather than returned.
        code.encode(63);
        assert_eq!(code.correct(63), Err(GridError::Inconsistent { module: 0 }));
        assert_eq!(code.correct(64).unwrap(), Correction { position: 63, repaired: None });
        // ONE spare module (legal range 7·9·11 = 693) always NOTICES a single error and never
        // returns a wrong position for it — but can seldom say which position was meant: setting
        // the period-13 module aside always leaves a legal-looking position, so almost every
        // error has two suspects. (The first draft of this test expected corrections to be
        // common, sampled 396 cases and found none.)
        let (mut fixed, mut refused) = (0, 0);
        for x in 0..693u64 {
            code.encode(x);
            let clean = code.phases.clone();
            for module in 0..4 {
                let wrong = (clean[module] + 1) % code.periods[module];
                code.phases[module] = wrong;
                match code.correct(693) {
                    Ok(c) => {
                        assert_eq!(c.position, x, "a single error was silently mis-corrected");
                        assert_eq!(c.repaired, Some(module), "and the corruption was not noticed");
                        fixed += 1;
                    }
                    Err(GridError::Inconsistent { .. }) => refused += 1,
                    Err(e) => panic!("{e}"),
                }
                code.phases[module] = clean[module];
            }
        }
        assert_eq!(fixed + refused, 693 * 4);
        assert!(refused > 10 * fixed, "with one spare module {fixed} errors were fixed and {refused} only detected");
        // No spare modules: every phase vector is a legal position, and nothing can be noticed.
        code.encode(100);
        code.phases[0] = (code.phases[0] + 1) % 13;
        assert_eq!(code.correct(9009).unwrap().repaired, None);
        assert_ne!(code.correct(9009).unwrap().position, 100);
        assert!(matches!(code.correct(0), Err(GridError::OutOfRange { what: "legal_range", .. })));
        assert!(matches!(code.correct(9010), Err(GridError::OutOfRange { what: "legal_range", .. })));
        assert_eq!(ModularCode::new(vec![4, 6, 7]).unwrap().single_error_range(), None, "4 and 6 share a factor");
        assert_eq!(ModularCode::new(vec![5, 7]).unwrap().single_error_range(), None);
        assert_eq!(ModularCode::new(vec![5, 7, 9]).unwrap().single_error_range(), Some(5));
    }

    #[test]
    fn bad_arguments_are_refused() {
        assert!(matches!(GridModule::new(0.0, 0.0, [0.0, 0.0], 1.0), Err(GridError::OutOfRange { what: "spacing", .. })));
        assert!(matches!(GridModule::new(1.0, 0.0, [0.0, 0.0], 0.0), Err(GridError::OutOfRange { what: "rate_max", .. })));
        assert!(matches!(GridModule::new(1.0, f64::NAN, [0.0, 0.0], 1.0), Err(GridError::NonFinite { what: "orientation" })));
        assert!(matches!(GridModule::new(1.0, 0.0, [f64::INFINITY, 0.0], 1.0), Err(GridError::NonFinite { what: "offset" })));
        let m = module();
        assert!(matches!(m.rate([f64::NAN, 0.0]), Err(GridError::NonFinite { what: "position" })));
        assert!(matches!(PhaseIntegrator::at(m, [0.0, f64::NAN]), Err(GridError::NonFinite { what: "position" })));
        let mut est = PhaseIntegrator::at(m, [0.0, 0.0]).unwrap();
        assert!(matches!(est.step([f64::NAN, 0.0], 0.1), Err(GridError::NonFinite { what: "velocity" })));
        assert!(matches!(est.step([0.1, 0.0], 0.0), Err(GridError::OutOfRange { what: "dt", .. })));
        // At the module's own vertex both phases are zero and the decoded position is the vertex.
        let home = PhaseIntegrator::at(m, m.offset).unwrap();
        assert_eq!(home.phases, [0.0, 0.0]);
        assert_eq!(home.position_in_cell(), m.offset);
        assert_eq!(ModularCode::new(vec![]), Err(GridError::Empty { what: "modules" }));
        assert!(matches!(ModularCode::new(vec![5, 1]), Err(GridError::OutOfRange { what: "period", .. })));
        // Coprime periods near 2³² multiply past u64.
        assert_eq!(ModularCode::new(vec![4_294_967_291, 4_294_967_279, 4_294_967_231]), Err(GridError::RangeOverflow));
        // Two such periods are fine, and decoding them is arithmetic, not a walk of 2³² steps.
        let mut big = ModularCode::new(vec![4_294_967_291, 4_294_967_279]).unwrap();
        let far = 4_294_967_291u64 * 4_294_967_279 - 12_345;
        big.encode(far);
        assert_eq!(big.decode().unwrap(), far);
        big.step(12_345);
        assert_eq!(big.decode().unwrap(), 0);
        assert_eq!(wrap_tau(-1e-20), 0.0);
        assert_eq!(wrap_tau(TAU + 0.5), 0.5);
    }

    /// Every error's text names its quantity AND says which way round the mismatch runs.
    ///
    /// Why the suite could not see it: every refusal in this module is checked with `matches!` or
    /// against the enum value, and nothing formats one. A `Display` that swapped `got` for `want`
    /// or dropped the "not" from "is not finite" would print a fluent, plausible sentence that is
    /// the exact opposite of what happened, and no assertion in the module reads a message at all.
    #[test]
    fn each_error_prints_the_quantity_it_found_and_the_one_it_wanted() {
        assert_eq!(
            GridError::Dimension { what: "phases", got: 1, want: 2 }.to_string(),
            "phases has length 1, expected 2"
        );
        assert_eq!(
            GridError::NonFinite { what: "orientation" }.to_string(),
            "orientation is not finite"
        );
        assert_eq!(GridError::Empty { what: "modules" }.to_string(), "modules is empty");
        assert_eq!(
            GridError::OutOfRange { what: "dt", value: 0.0, low: 1.0, high: 2.0 }.to_string(),
            "dt = 0 is outside [1, 2]"
        );
        assert_eq!(
            GridError::Inconsistent { module: 3 }.to_string(),
            "the phase of module 3 contradicts the modules before it"
        );
        assert_eq!(
            GridError::RangeOverflow.to_string(),
            "the least common multiple of the periods overflows u64"
        );
        // And the message a real mismatch produces, so the two numbers are not merely a literal
        // in this test: a three-module code carrying two phases.
        let mut code = ModularCode::new(vec![3, 5, 7]).unwrap();
        code.phases = vec![0, 0];
        assert_eq!(code.decode().unwrap_err().to_string(), "phases has length 2, expected 3");
    }

    /// An INFINITE spacing, peak rate, orientation or time step is refused, exactly as a `NaN` is.
    ///
    /// Why the suite could not see it: the refusal fixtures use `0.0` for the positive quantities
    /// and `NaN` for the orientation, and both of those are caught by the `!(v > 0.0)` half of the
    /// guard alone (`NaN > 0.0` is false). Infinity is the one value that passes the positivity
    /// test and is still not a number a metre or a second can take — and an infinite spacing makes
    /// every wave vector zero, so the map fires everywhere at `2/3 r_max` and never refuses.
    #[test]
    fn an_infinite_parameter_is_refused_as_firmly_as_a_nan() {
        for bad in [f64::INFINITY, f64::NAN] {
            assert!(
                matches!(
                    GridModule::new(bad, 0.0, [0.0, 0.0], 1.0),
                    Err(GridError::OutOfRange { what: "spacing", .. })
                ),
                "spacing {bad}"
            );
            assert!(
                matches!(
                    GridModule::new(1.0, 0.0, [0.0, 0.0], bad),
                    Err(GridError::OutOfRange { what: "rate_max", .. })
                ),
                "rate_max {bad}"
            );
            assert_eq!(
                GridModule::new(1.0, bad, [0.0, 0.0], 1.0),
                Err(GridError::NonFinite { what: "orientation" }),
                "orientation {bad}"
            );
        }
        // Negative infinity too, on the orientation, which `is_nan()` also lets through.
        assert_eq!(
            GridModule::new(1.0, f64::NEG_INFINITY, [0.0, 0.0], 1.0),
            Err(GridError::NonFinite { what: "orientation" })
        );
        // And the time step: an infinite `dt` advances every phase to `NaN` and the integrator
        // never recovers, because `wrap_tau` of a `NaN` is a `NaN`.
        let mut est = PhaseIntegrator::at(module(), [0.0, 0.0]).unwrap();
        let before = est.phases;
        for bad in [f64::INFINITY, f64::NAN, 0.0, -1.0] {
            assert!(
                matches!(
                    est.step([0.1, 0.0], bad),
                    Err(GridError::OutOfRange { what: "dt", .. })
                ),
                "dt {bad}"
            );
        }
        assert_eq!(est.phases, before, "a refused step must not have moved the phases");
    }

    /// The wave vectors are rotated `+30°` off the lattice, not `−30°`, and which way round decides
    /// WHICH reciprocal vector each phase is.
    ///
    /// Why the suite could not see it: `rate` sums `cos(k_i · d)` and the cosine is even, so `k`
    /// and `−k` are the same term; rotating the triple the other way maps it onto itself as a set
    /// of LINES (`θ−30 ≡ θ+150 mod 180`), leaving the firing map bit for bit unchanged. The
    /// existing lattice test only asks that every `k_i · a_j` be a whole number of turns, which
    /// holds for both triples. What moves is the assignment: `k[1] · a[0]` is 0 turns for the real
    /// triple and 1 turn for the mirrored one, so `PhaseIntegrator`'s two phases are different
    /// coordinates of the same cell.
    #[test]
    fn the_wave_vectors_are_the_reciprocal_basis_and_not_its_mirror() {
        let m = module();
        let a = m.lattice_vectors();
        let k = m.wave_vectors();
        // Turns of `k_i` along `a_j`, exactly: [[1, 1], [0, 1], [-1, 0]].
        let want = [[1.0, 1.0], [0.0, 1.0], [-1.0, 0.0]];
        for i in 0..3 {
            for j in 0..2 {
                let turns = (k[i][0] * a[j][0] + k[i][1] * a[j][1]) / TAU;
                assert!(
                    (turns - want[i][j]).abs() < 1e-12,
                    "k[{i}]·a[{j}] = {turns} turns, expected {}",
                    want[i][j]
                );
            }
        }
        // Stated the other way: the first wave vector points 30° anticlockwise of the lattice's
        // first vector, and the third is 120° beyond that.
        for (i, offset) in [(0usize, PI / 6.0), (1, PI / 2.0), (2, 5.0 * PI / 6.0)] {
            let angle = k[i][1].atan2(k[i][0]);
            let want = m.orientation + offset;
            assert!((angle - want).abs() < 1e-12, "k[{i}] at {angle} rad, expected {want}");
        }
    }

    /// The starting phases are wrapped into one turn, like every phase the integrator produces
    /// afterwards.
    ///
    /// Why the suite could not see it: both path-integration fixtures start within the first
    /// lattice cell of the module's own offset, where the raw phases are already inside `[0, 2π)`
    /// and wrapping is the identity. The comparisons that follow are all differences taken modulo
    /// a turn (`gap.min(TAU - gap)`), and `position_in_cell` inverts the same linear map it was
    /// given, so an unwrapped start is self-consistent — it just reports a phase of 20.7 radians
    /// and a position several cells away from the cell it claims to be in.
    #[test]
    fn the_starting_phases_are_wrapped_into_one_turn() {
        let m = module();
        let a = m.lattice_vectors();
        let near = [
            m.offset[0] + 0.2 * a[0][0] + 0.1 * a[1][0],
            m.offset[1] + 0.2 * a[0][1] + 0.1 * a[1][1],
        ];
        // Seven cells one way and four the other: the same phases, seven and four turns along.
        let far = [
            near[0] + 7.0 * a[0][0] - 4.0 * a[1][0],
            near[1] + 7.0 * a[0][1] - 4.0 * a[1][1],
        ];
        let p_near = PhaseIntegrator::at(m, near).unwrap().phases;
        let p_far = PhaseIntegrator::at(m, far).unwrap().phases;
        for i in 0..2 {
            assert!((0.0..TAU).contains(&p_near[i]), "phase {i} of the near start is {}", p_near[i]);
            assert!((0.0..TAU).contains(&p_far[i]), "phase {i} of the far start is {}", p_far[i]);
            assert!((p_near[i] - p_far[i]).abs() < 1e-12, "{} vs {}", p_near[i], p_far[i]);
        }
        // The documented range is the same one `step` maintains, so a start and a walk to the
        // same place agree.
        let mut walked = PhaseIntegrator::at(m, near).unwrap();
        walked.step([7.0 * a[0][0] - 4.0 * a[1][0], 7.0 * a[0][1] - 4.0 * a[1][1]], 1.0).unwrap();
        for i in 0..2 {
            assert!((0.0..TAU).contains(&walked.phases[i]));
            assert!((walked.phases[i] - p_far[i]).abs() < 1e-12);
        }
    }

    /// A period below 2 is refused, and the refusal names 2 as the floor — the number the doc gives
    /// and the reason it gives: a module of period 1 has one phase and says nothing.
    ///
    /// Why the suite could not see it: the existing check is `matches!(..., OutOfRange { what:
    /// "period", .. })`, which ignores every field but the name. A floor printed as 1 would refuse
    /// exactly the same inputs and tell the reader that the value they supplied, 1, is inside the
    /// range they were refused for.
    #[test]
    fn the_refused_periods_floor_is_two_and_the_message_says_so() {
        assert_eq!(
            ModularCode::new(vec![5, 1]),
            Err(GridError::OutOfRange {
                what: "period",
                value: 1.0,
                low: 2.0,
                high: u64::MAX as f64
            })
        );
        assert_eq!(
            ModularCode::new(vec![0, 5]),
            Err(GridError::OutOfRange {
                what: "period",
                value: 0.0,
                low: 2.0,
                high: u64::MAX as f64
            })
        );
        // Two itself is legal, which is what makes the floor a floor.
        assert_eq!(ModularCode::new(vec![2, 5]).unwrap().range().unwrap(), 10);
    }

    /// A range of exactly `u64::MAX` fits in a `u64` and is accepted; the guard is `>`, not `>=`.
    ///
    /// Why the suite could not see it: the overflow fixtures are a triple whose product is about
    /// `2^96`, which both forms of the guard refuse. Only a code whose least common multiple lands
    /// exactly on `u64::MAX` separates them, and `u64::MAX = (2^32 − 1)(2^32 + 1)` with the two
    /// factors coprime is one.
    #[test]
    fn a_range_of_exactly_the_largest_u64_is_accepted() {
        let code = ModularCode::new(vec![4_294_967_295, 4_294_967_297]).unwrap();
        assert_eq!(code.range().unwrap(), u64::MAX);
        assert_eq!(4_294_967_295u128 * 4_294_967_297, u128::from(u64::MAX), "the fixture's arithmetic");
        // One step past it is refused: 3 · 2^63 is the smallest multiple of 2^63 above u64::MAX.
        assert_eq!(ModularCode::new(vec![1u64 << 63, 3]), Err(GridError::RangeOverflow));
    }

    /// The overflow guard is INSIDE the loop, so the running least common multiple is checked
    /// before it is multiplied again.
    ///
    /// Why the suite could not see it: the existing overflow fixture is three periods near `2^32`,
    /// whose product is about `2^96` — far inside a `u128`, so a guard moved to after the loop sees
    /// the same enormous number and refuses it too. The guard only matters when the running value
    /// wraps a `u128` on the way, and then it can wrap back DOWN: these three periods have a true
    /// least common multiple of about `2^190`, and `(2^63 · q · r) mod 2^128` is exactly `2^63`
    /// because `q · r ≡ 1 (mod 2^65)`. A checked-only-at-the-end guard therefore returns
    /// `Ok(9223372036854775808)` for a code whose range does not fit in 190 bits.
    #[test]
    fn the_overflow_guard_runs_inside_the_loop_so_a_wrapped_lcm_cannot_pass() {
        let q: u64 = 9_223_372_036_854_775_805;
        let r: u64 = 15_372_286_728_091_293_013;
        // The property that makes the wrap land on a small number, stated as arithmetic.
        assert_eq!(u128::from(q) * u128::from(r) % (1u128 << 65), 1);
        let periods = vec![1u64 << 63, q, r];
        assert_eq!(ModularCode::new(periods.clone()), Err(GridError::RangeOverflow));
        let code = ModularCode { periods, phases: vec![0, 0, 0] };
        assert_eq!(code.range(), Err(GridError::RangeOverflow));
        // A guard that ran only after the loop would have returned `Ok(9223372036854775808)`
        // here — a range of 2^63 for a code whose modules cannot be held in one.
    }

    /// `decode` checks that the range fits in a `u64` before it starts, because `ModularCode`'s
    /// fields are public and a code can be assembled without going through `new`.
    ///
    /// Why the suite could not see it: the overflow fixture calls `new`, which refuses the code, so
    /// no test ever holds one whose range does not fit. Assembled directly, such a code decodes
    /// without complaint: the accumulator is a `u128`, the answer is truncated by `x as u64` at the
    /// end, and for phases that are all zero the answer even looks right.
    #[test]
    fn decode_refuses_a_code_whose_range_does_not_fit_in_a_u64() {
        let periods = vec![4_294_967_291u64, 4_294_967_279, 4_294_967_231];
        assert_eq!(ModularCode::new(periods.clone()), Err(GridError::RangeOverflow));
        let code = ModularCode { periods: periods.clone(), phases: vec![0, 0, 0] };
        assert_eq!(code.decode(), Err(GridError::RangeOverflow));
        // Including the shape that reads as a success: every phase zero decodes to 0.
        let ones = ModularCode { periods, phases: vec![1, 1, 1] };
        assert_eq!(ones.decode(), Err(GridError::RangeOverflow));
        // And `correct`, which decodes too, refuses it at the same place.
        assert_eq!(code.correct(10), Err(GridError::RangeOverflow));
    }

    /// The running modulus inside `decode` is the least common multiple of the periods so far, not
    /// their product, and the two differ exactly when the periods share a factor.
    ///
    /// Why the suite could not see it: the only shared-factor fixture is a pair, `[4, 6]`, and the
    /// running modulus is written for the LAST time on the last module and then never read. With
    /// two modules the difference between `m·n/g` and `m·n` is invisible by construction. A third
    /// module reads it — through `gcd(m, n)` and through `mod_inverse(m/g, n/g)` — and then
    /// `[4, 6, 9]` decodes 12 of its 36 positions to the wrong answer, each of them larger than the
    /// range.
    #[test]
    fn the_running_modulus_is_the_lcm_of_the_periods_and_not_their_product() {
        let mut code = ModularCode::new(vec![4, 6, 9]).unwrap();
        assert_eq!(code.range().unwrap(), 36, "lcm, not the product 216");
        for x in 0..36u64 {
            code.encode(x);
            assert_eq!(code.decode().unwrap(), x, "position {x}");
        }
        // The first position where a product-shaped modulus diverges, called out so the fixture's
        // discriminating power is on the record rather than implied by the loop.
        code.encode(13);
        assert_eq!(code.phases, vec![1, 1, 4]);
        assert_eq!(code.decode().unwrap(), 13);
    }

    /// A MALFORMED code word — a phase that is not below its period, or a phase vector of the wrong
    /// length — is reported as what it is, not folded into the corruption story.
    ///
    /// Why the suite could not see it: every corruption fixture sets a phase to another LEGAL phase
    /// of the same module, which is the case `correct` exists for. A malformed word reaches the
    /// same `decode`, and swallowing its error leaves the repair loop to run on a code the type's
    /// own invariant does not hold for — which usually ends in `Inconsistent`, a verdict that tells
    /// the caller their hardware is faulty when their array is the wrong shape.
    #[test]
    fn a_malformed_code_word_is_named_as_malformed_and_not_as_corrupted() {
        let mut code = ModularCode::new(vec![7, 9, 11, 13]).unwrap();
        code.encode(30);
        code.phases[2] = 11; // not below its period: a phase no module could hold
        assert_eq!(
            code.correct(63),
            Err(GridError::OutOfRange { what: "phase", value: 11.0, low: 0.0, high: 10.0 })
        );
        // Setting module 2 aside WOULD have decoded to 30, so the repair loop has an answer ready;
        // it must not be reached.
        let mut without_two = ModularCode::new(vec![7, 9, 13]).unwrap();
        without_two.encode(30);
        assert_eq!(without_two.decode().unwrap(), 30);

        // The other malformed shape: a phase vector of the wrong length.
        code.encode(30);
        code.phases.pop();
        assert_eq!(
            code.correct(63),
            Err(GridError::Dimension { what: "phases", got: 3, want: 4 })
        );
    }

    /// An IMPOSSIBLE code word — phases no position produces — is repaired, not returned as an
    /// error. That is the case `correct` was written for, one module arbitrarily wrong.
    ///
    /// Why the suite could not see it: every correction fixture uses pairwise coprime periods,
    /// where the generalised Chinese remainder condition is vacuous and `decode` never reports
    /// `Inconsistent` however a phase is corrupted — it just returns the wrong position, which the
    /// legal-range test then rejects. Periods that share a factor are the case where corruption is
    /// detected by `decode` itself, and there the refusal has to be swallowed and the repair tried.
    #[test]
    fn an_impossible_code_word_is_repaired_rather_than_refused() {
        // Position 7 in periods 5, 6, 9, with module 1's phase knocked from 1 to 0. Modules 1 and
        // 2 share the factor 3, so the phases now contradict each other outright.
        let mut code = ModularCode::new(vec![5, 6, 9]).unwrap();
        code.encode(7);
        assert_eq!(code.phases, vec![2, 1, 7]);
        code.phases[1] = 0;
        assert_eq!(code.decode(), Err(GridError::Inconsistent { module: 2 }));
        assert_eq!(
            code.correct(10).unwrap(),
            Correction { position: 7, repaired: Some(1) }
        );
        // The other two suspects do not produce a legal position, which is why the answer is
        // unique: dropping module 0 leaves a contradiction, dropping module 2 leaves 12.
        let mut dropped_zero = ModularCode::new(vec![6, 9]).unwrap();
        dropped_zero.phases = vec![0, 7];
        assert_eq!(dropped_zero.decode(), Err(GridError::Inconsistent { module: 1 }));
        let mut dropped_two = ModularCode::new(vec![5, 6]).unwrap();
        dropped_two.phases = vec![2, 0];
        assert_eq!(dropped_two.decode().unwrap(), 12);
    }

    /// A code of ONE module cannot correct itself: setting its only module aside leaves no modules
    /// at all, and an empty code decodes to 0 — a legal-looking position that is evidence of
    /// nothing.
    ///
    /// Why the suite could not see it: every `correct` fixture has four modules, and with four the
    /// guard `periods.len() < 2` never fires. The guard only matters for a one-module code whose
    /// single phase decodes outside the legal range, and then the difference is between refusing
    /// and answering 0 while blaming the only module there is.
    #[test]
    fn a_lone_module_cannot_correct_itself_away() {
        let mut lone = ModularCode::new(vec![7]).unwrap();
        lone.encode(5);
        assert_eq!(lone.decode().unwrap(), 5);
        // 5 is outside a legal range of 3, and there is no second module to appeal to.
        assert_eq!(lone.correct(3), Err(GridError::Inconsistent { module: 0 }));
        // The empty code that dropping the only module would leave decodes to zero, which is the
        // answer the guard exists to refuse.
        let empty = ModularCode { periods: vec![], phases: vec![] };
        assert_eq!(empty.decode().unwrap(), 0);
        // Inside the legal range the same code answers, unrepaired.
        lone.encode(2);
        assert_eq!(lone.correct(3).unwrap(), Correction { position: 2, repaired: None });
    }
}
