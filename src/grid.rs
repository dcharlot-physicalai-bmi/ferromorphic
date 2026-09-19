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
//! # What this module has NOT reproduced
//!
//! - The attractor network that would HOLD these phases, or its noise. The phases here are
//!   numbers; drift, and the error-correcting reading of the code (Sreenivasan and Fiete, 2011),
//!   are not modelled.
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
}
