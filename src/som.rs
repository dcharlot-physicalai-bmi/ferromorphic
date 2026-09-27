//! The self-organising map: Kohonen's 1982 rule as he first printed it, and the standard map of his
//! 1990 review — each step checked against its equation, the 1990 map against the Kohonen lab's own
//! `SOM_PAK` to the bit, and both against what the papers say they do.
//!
//! # The two papers
//!
//! Kohonen, *Self-organized formation of topologically correct feature maps*, Biological
//! Cybernetics 43:59–69 (1982), doi:10.1007/BF00337288, introduced the rule. Kohonen, *The
//! self-organizing map*, Proceedings of the IEEE 78(9):1464–1480 (1990), doi:10.1109/5.58325, gave
//! the form the field adopted and the lab's `SOM_PAK` implements.
//!
//! # The 1982 rule: [`Kohonen1982`]
//!
//! Each unit forms the inner product of its weight vector with its input, the largest response
//! wins, and the winner and its neighbours turn towards the input and are renormalised:
//!
//! ```text
//! η_i = Σ_j μ_ij ξ_j = m_iᵀ x                          (1)
//! η_k = max_i {η_i}                                    (2)
//! m_i(t + 1) = (m_i(t) + αx(t)) / ‖m_i(t) + αx(t)‖_E    (3)
//! ```
//!
//! for "unit k and all the eight of its nearest neighbours (except at the edges of the array where
//! the number of neighbours was different)" (p. 61): [`Reach::Nearest`], the ring at Chebyshev
//! distance one, which on a chain is Simulation 4's "two nearest neighbours" (p. 62). The
//! neighbourhood never shrinks. Simulation 3 adds the 16 units around it at gain `α/4`
//! ([`Reach::Rings`]). The gain is "a function of the iteration step, e.g., proportional to 1/t"
//! (p. 61), with no constant printed ([`Schedule::InverseT`]). Sect. 2.3 lets each unit read its own
//! "non-identical but coherent" inputs ([`Kohonen1982::step_coherent`]). The 1990 paper keeps the
//! rule as its eqs. (9)–(10), with a shrinking `N_c` and `α'(t) = 100/t` ([`Reach::Radius`]).
//!
//! # The 1990 map: [`Som`]
//!
//! The winner is the nearest weight vector, eq. (2'), `‖x − m_c‖ = minᵢ ‖x − m_i‖`, and every unit
//! moves towards the sample by its neighbourhood function, eq. (7),
//! `m_i(t + 1) = m_i(t) + h_ci(t)[x(t) − m_i(t)]`: [`Kernel::Bubble`] is eq. (6), `h_ci = α(t)` inside
//! `N_c(t)` and zero outside, and [`Kernel::Gaussian`] is eq. (8), `h_ci = h₀ exp(−‖r_c − r_i‖²/σ²)`.
//! Sect. II-D's hints (pp. 1468–1469) give a two-phase recipe: [`Phase::ordering_1990`], 1000 steps
//! with `α(t) = 0.9(1 − t/1000)` and a radius that starts wide and shrinks linearly to one; then
//! [`Phase::convergence_1990`], "at least 500 times the number of network units" steps with `α`
//! "of the order of or less than .01" and `N_c` still holding "the nearest neighbors of cell c".
//! Table 1's worked example (Sect. II-E) is [`Phase::table1_1990`]. The 1990 paper states no batch
//! form of the map — its batch algorithm is `k`-means (p. 1466), vector quantisation with no
//! neighbourhood — so this module has none.
//!
//! One step of either rule is its equation, checked by evaluating the equation in the test. The
//! lattice distances are exact in floating point, so the edge of `N_c` falls where the paper puts
//! it: a hexagonal neighbour is at `1`, not `0.9999999999999999`.
//!
//! # The reference is the lab's own program
//!
//! `SOM_PAK` 3.1 (7 April 1995), by the SOM Programming Team of the Helsinki University of
//! Technology, is still served at the lab's old address; `tools/som_reference.py` downloads it,
//! checks its SHA-256, builds it, runs it and prints the constants the tests hold. Built in
//! `double` and run on the same data, initial codebook and presentation order, it and
//! [`Som::train`] agree BIT FOR BIT through both phases of three bubble cases — a chain, a
//! rectangular sheet, a hexagonal sheet — quantisation errors included. The Gaussian case agrees to
//! `1.1e-16`, because `SOM_PAK` writes eq. (8) as `exp(−d²/2r²)` and `σ = r√2` squares back to `2r²`
//! only to rounding.
//!
//! ⚠ **On two of the four cases rounding picks the map, and `SOM_PAK`'s two builds part.** Where
//! the `float` build, the package as written, takes the `double` build's path, the two agree to
//! float rounding, `4.5e-7`. In `rect3` and `hexbubble` they part, and the cause is those cases,
//! not the `float` build: both open with a neighbourhood spanning all or most of the map at
//! `α = 0.5`, which pulls the codebook together until its units are near-ties that rounding
//! decides. At step 32 of `rect3`, 18 of the 20 units lie within `4.8e-10` of the nearest in
//! squared distance, units 14 and 4 `1.3e-12` apart, and the `float` build takes unit 1, `1.5e-10`
//! behind the `double` build's unit 14. At step 31 of `hexbubble`, sample 31 is `7.7e-9` nearer to
//! unit `(4, 3)` than to unit `(4, 1)`, mirror images across a map the wide neighbourhood has kept
//! nearly symmetric, and `float` takes `(4, 1)`; the map it ends with is the `double` map upside
//! down, to `4.3e-7`. Any change of precision can flip ties that close, so on these two cases
//! neither build is a reference for the other, and against the `float` build a correct
//! implementation shows an error of 0.7. The steps and units are the builds' own, printed by
//! `tools/som_reference.py`.
//!
//! # What the papers claim, measured
//!
//! - **Ordering.** Over 1000 seeds, a chain of 20 random scalars ends the 1990 ordering phase
//!   strictly monotone every time when its radius starts at 10, and 245 times when it starts at
//!   one — hint 3's "If the neighborhood is too small to start with, the map will not be ordered
//!   globally" (p. 1469). A chain of 10 with no neighbourhood never orders.
//! - **Magnification.** Ritter and Schulten, *On the stationary state of Kohonen's
//!   self-organizing sensory mapping*, Biological Cybernetics 54:99–106 (1986), derive from their
//!   equilibrium condition, eq. (3), `⟨h(r − s)(v − w(s))⟩ = 0`, that a chain's magnification is
//!   `M(w) ∝ P(w)^{2/3}`, eq. (11) — "for sufficiently many units and all sufficiently small d"
//!   (p. 104), a neighbourhood narrow against the map but many units wide. The tests solve eq. (3)
//!   for a chain on `p(x) = 2x` and fit the exponent over the middle three fifths of its gaps. On
//!   100 units it is 0.33332 with no neighbourhood at all — the `p^{1/3}` of vector quantisation,
//!   `[p(x)]^{n/(n+r)}` in the 1990 paper (p. 1466) — 0.59996 under the 1990 recipe's final bubble
//!   of radius one, and 0.64108, 0.66021 and 0.66633 under Gaussians of `σ` = 2, 4 and 8 units. The
//!   last is a boundary effect of the short chain: at `σ = 8` the exponent is 0.66533 on 200 units
//!   and 0.66505 on 400, `1.6e-3` below `2/3`, where `σ = 2` moves only to 0.64104 on 200. A
//!   vanishing neighbourhood in their sense is narrow, not absent. A map trained from their Fig. 9's
//!   starting curve `w(r) = √r`, exponent 1, comes to within `8.8e-4` of the solved state, with
//!   exponent 0.66449. All of this is the 1990 rule, eq. (7), on a chain.
//!
//!   The 1982 paper's Sect. 4.1 finds "the magnification factor is approximately proportional to
//!   the above frequency" (p. 67), an exponent of one, but on a two-dimensional array trained on an
//!   area whose two halves drew different fractions of the samples (Fig. 11, after Kohonen 1981,
//!   which this review did not read). That experiment is not rerun here, and the chain's law does
//!   not decide it: in two dimensions Ritter and Schulten conclude that "no general local
//!   expression in terms of the probability density can be given" (p. 105), and find `P^{2/3}` for
//!   a product density on a rectangular map, eq. (11), but `M ∝ P` for a map given by an analytic
//!   function, eq. (15). The "intuitive, but incorrect expectation `M(w) ∝ P(w)`" is the one
//!   "suggested in (Kohonen 1984)", a book, and `P^{2/3}` "may be in contrast" to it; the simple
//!   function of `P` they deny in two dimensions they find "implied in (Kohonen 1982c, 1984)",
//!   1982c being a different paper, Biological Cybernetics 44:135–140.
//! - **Table 1 (1990).** Rerun at its printed parameters on its 10 × 7 hexagonal array, 100 seeds
//!   put the ends of the minimal spanning tree's 31 edges 1.2619 units apart on average, against
//!   1.2357 in Fig. 6; 43 runs are at least as tight as the printed map, and 28 give every item a
//!   unit of its own, as Fig. 6 does. The run is shorter than hint 1's rule of thumb, "at least 500
//!   times the number of network units" steps (p. 1469), 35 000 for these 70 units against the
//!   11 000 given; but the hint goes on, "for 'fast learning,' e.g., in speech recognition, 10 000
//!   steps and even less may sometimes be enough", and 32 items "must be recycled" for any number.
//!   Stretched to 35 000 steps, the reruns come out no tighter: 1.2891 on average, 28 as tight as
//!   Fig. 6, 30 with every item apart. Its starting gain, 0.5, is below hint 2's "close to unity",
//!   which the paper notes: "the initial value could have been closer to unity, say, .9" (p. 1470).
//! - **Simulations 1–3 (1982).** With the 1990 paper's `α'(t) = 100/t`, an 8 × 8 array whose `N_c`
//!   shrinks orders fully in 50 of 50 runs by 10 000 steps; the 1982 paper's fixed ring of eight
//!   orders 13 of 50. Of Simulation 3's second ring p. 62 says "ordering seems to proceed more
//!   quickly and more reliably; on the other hand, the final result is perhaps not as good as
//!   before", and each clause holds by its own measure. More quickly: 26.9 of the 64 test vectors
//!   home at 500 steps, against 13.4 for the single ring. More reliably: at 10 000 steps a mean of
//!   47.6 home against 35.0, and 42 of 50 runs with at least half home against 23. Not as good: no
//!   run with even 60 home, where 17 single-ring runs reach Fig. 3c's 62 or more and 13 reach 64.
//! - **Quantisation against topographic error.** The mean distance from each sample to its winner
//!   (`SOM_PAK`'s `qerror`), and the fraction of samples whose two best units are not adjacent —
//!   the SOM Toolbox's `som_quality`, from the same laboratory, which cites Kiviluoto, *Topology
//!   preservation in self-organizing maps*, Proc. ICNN 1996, pp. 294–299 (this review did not read
//!   Kiviluoto's paper). On 20 maps of the unit square, dropping the neighbourhood after the 1990
//!   recipe lowers every map's quantisation error (0.0404 to 0.0361 on average) and raises every
//!   map's topographic error (0.0205 to 0.0453).
//!
//! # What the papers print that does not hold, or does not suffice
//!
//! ⚠ **Simulation 4 (1982, Table 1 on p. 63) cannot be rerun from its caption alone.** The caption
//! names the resonators, 20 "second-order filters with quality factor Q=2.5" tuned at random in
//! `[1, 2]`, and the training frequencies, drawn from `[0.5, 1]`: every one below every resonance.
//! A second-order band-pass gain rises all the way up to resonance, so every one of the fifty
//! inputs — ten units, "Five inputs to each array unit" picked from the 20 resonators (p. 63) —
//! rises across the whole training band. The paper does not give the filters' gain normalisation
//! (the tests take unit gain at resonance), the gain constant of eq. (3), the initial weights, or
//! whether inputs were normalised. It does say what the table holds: "those test frequencies to
//! which each processing unit became most sensitive", one frequency per unit. Read literally, with
//! raw amplitudes, every unit is most sensitive to the top test frequency, 1.00, in 200 of 200
//! runs, and one unit wins all 51 test frequencies in 175 (Sect. 4.5's "focusing"). With each
//! unit's five inputs normalised, the ten preferred frequencies run monotonically along the chain
//! in 117 of 200 runs, 97 of them strictly, and take 9.52 distinct values on average; the 51
//! per-frequency winners, a different measure, run monotonically in 41 runs and use 9.365 of the
//! ten units. Nor is the table itself an ordered list: Experiment 1's tenth unit (0.83) falls back
//! below its ninth (0.98), repeating its seventh, and Experiment 2 repeats 0.98 — disorders the
//! text accepts as like "natural maps". The reruns fold the same way: seed 0's rises from 0.58 to
//! 0.92 at its seventh unit and falls back over the last three, 0.86, 0.81, 0.80, where Experiment
//! 1 falls back at its last one.
//!
//! ⚠ **Simulations 1–3 print no gain constant and no patch size, and a distribution that can be
//! read two ways.** Fig. 3's caption: "The distribution had edges, each of which contained as many
//! vectors as the inside". Either the density was uniform up to sharp edges, or each edge was a
//! strip holding as many vectors as the interior; Fig. 3a's stipple looks uniform, which favours
//! the first, and the tests read it so. The patch is drawn to scale inside the sphere's outline,
//! and measured on the page (400 dpi) its half-width is 215.5 pixels against a radius of 302,
//! 0.714: the inscribed square, half-width `1/√2`, which the tests use, with Fig. 3b's test
//! vectors at the centres of its 8 × 8 tiling. The points are uniform in the front view and lifted
//! onto the sphere, so their density per unit area of the sphere falls as `z` towards the corners.
//! The gain constant is borrowed from the 1990 paper's eq. (10). The numbers above are for these
//! readings. Fig. 3c's own map puts 62 of its 64 test vectors on their own units.
//!
//! ⚠ **Fig. 2's neighbourhoods are hexagons; `SOM_PAK`'s are discs.** On the hexagonal lattice the
//! disc of whole radius `r` holds the same units as the hexagon of `r` steps for every `r` up to
//! six. At seven the disc takes 18 more, and between whole radii — which a shrinking radius passes
//! through — the disc is the larger, 13 units against 7 at radius 1.8. This module follows
//! `SOM_PAK`.
//!
//! ⚠ **Eq. (8)'s `σ` is not `SOM_PAK`'s radius.** The paper's Gaussian is `exp(−d²/σ²)` and
//! `SOM_PAK`'s is `exp(−d²/2r²)`, so a radius copied from one to the other narrows or widens the
//! kernel by `√2`.
//!
//! ⚠ **Eq. (3) keeps a length only once it is one.** "Notice that the process of Eq. (3) does not
//! change the length of `m_i` but only rotates `m_i` towards x" (1982, p. 61) — of a unit vector. The
//! initial values are "random numbers", and the first step through eq. (3) sets a unit's length to
//! one whatever it was.
//!
//! ⚠ **`α'(t) = 100/t` needs a count from one.** The 1990 paper counts `t = 1, 2, 3, …` in Sect.
//! I-C (p. 1465) and `t = 0, 1, 2, …` for its delta rule (p. 1466); at `t = 0` eq. (10)'s gain is
//! infinite. [`Schedule::InverseT`] counts from one.
//!
//! # What is checked
//!
//! One step of eq. (7) with each kernel, of eq. (3) with each reach and with coherent inputs, each
//! against its equation; the lattice's distances and neighbours, exactly; the schedules against the
//! printed laws, to `1.1e-16`; `SOM_PAK` 3.1 on four cases, bit for bit where the arithmetic is the
//! same; the near-ties at the steps where its two builds part; the ordering counts, the
//! magnification exponents, Table 1 against Fig. 6, Simulations 1–4, and the two errors on the
//! same maps, each as stated above; and every refusal, by the message it renders.

use core::fmt;

use crate::rng::Rng;

/// Why a map could not be built, trained or measured.
#[derive(Debug, Clone, PartialEq)]
pub enum SomError {
    /// A parameter that must be finite and positive was not.
    NotPositive {
        /// Which parameter.
        what: &'static str,
        /// Its value.
        value: f64,
    },
    /// A value that must be finite was not.
    NonFinite {
        /// Which quantity.
        what: &'static str,
        /// Its value.
        value: f64,
    },
    /// A finite value outside the closed interval it must lie in.
    OutOfRange {
        /// Which parameter.
        what: &'static str,
        /// Its value.
        value: f64,
        /// The lower end, included.
        low: f64,
        /// The upper end, included.
        high: f64,
    },
    /// An interval `[low, high)` to draw from that holds no value, because `high` is not above `low`.
    EmptyInterval {
        /// The lower end.
        low: f64,
        /// The upper end.
        high: f64,
    },
    /// A slice whose length is not the one the map needs.
    Length {
        /// Which slice.
        what: &'static str,
        /// The length it must have (or be a whole multiple of, for a data set).
        expected: usize,
        /// The length it has.
        got: usize,
    },
    /// Fewer items than the operation needs.
    TooFew {
        /// Of what.
        what: &'static str,
        /// The least number that works.
        need: usize,
        /// The number supplied.
        got: usize,
    },
    /// A 1982 neighbourhood that the paper defines only on a rectangular array.
    Lattice {
        /// Which neighbourhood.
        what: &'static str,
    },
    /// Eq. (3) of the 1982 paper divides by `‖m_i + αx‖`, and for this unit it was zero.
    ZeroNorm {
        /// The unit's index.
        unit: usize,
    },
}

impl fmt::Display for SomError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotPositive { what, value } => write!(f, "{what} = {value} must be finite and positive"),
            Self::NonFinite { what, value } => write!(f, "{what} = {value} is not finite"),
            Self::OutOfRange { what, value, low, high } => write!(f, "{what} = {value} is outside [{low}, {high}]"),
            Self::EmptyInterval { low, high } => write!(f, "[{low}, {high}) holds no value to draw"),
            Self::Length { what, expected, got } => write!(f, "{what} has length {got}; it must be {expected}"),
            Self::TooFew { what, need, got } => write!(f, "{what}: {got} supplied, at least {need} needed"),
            Self::Lattice { what } => write!(f, "{what} is defined on a rectangular array, not a hexagonal one"),
            Self::ZeroNorm { unit } => write!(f, "unit {unit}: m + alpha x is the zero vector and eq. (3) cannot normalise it"),
        }
    }
}

impl std::error::Error for SomError {}

fn finite(what: &'static str, value: f64) -> Result<f64, SomError> {
    if value.is_finite() { Ok(value) } else { Err(SomError::NonFinite { what, value }) }
}

fn positive(what: &'static str, value: f64) -> Result<f64, SomError> {
    if value.is_finite() && value > 0.0 { Ok(value) } else { Err(SomError::NotPositive { what, value }) }
}

fn within(what: &'static str, value: f64, low: f64, high: f64) -> Result<f64, SomError> {
    finite(what, value)?;
    if value >= low && value <= high { Ok(value) } else { Err(SomError::OutOfRange { what, value, low, high }) }
}

fn all_finite(what: &'static str, v: &[f64]) -> Result<(), SomError> {
    for &value in v {
        finite(what, value)?;
    }
    Ok(())
}

/// A uniform index below `n` from 64 random bits, `(2³²·hi + lo) mod n`. The modulo bias moves each
/// index's probability by less than `1/2⁶⁴`, and no length conversion can fail on the way, as one
/// to the `u32` of [`Rng::below`] could.
fn draw(rng: &mut Rng, n: usize) -> usize {
    let bits = (u64::from(rng.next_u32()) << 32) | u64::from(rng.next_u32());
    (bits % n as u64) as usize
}

/// How the units of a map are arranged in the plane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lattice {
    /// A rectangular array: unit `(x, y)` at `(x, y)`. With a height of one it is a chain.
    Rect,
    /// A hexagonal array: unit `(x, y)` at `(x + ½·(y mod 2), y·√3/2)`, so every interior unit has
    /// six neighbours at distance one — the 1990 paper's Fig. 2 and its "neuron c and its six
    /// neighbors" (p. 1470). Odd rows are the shifted ones, as in `SOM_PAK`.
    Hex,
}

/// The array of units: its lattice, its width and its height. Unit `i` sits at column `i % width`,
/// row `i / width`, the order `SOM_PAK` stores a codebook in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Grid {
    lattice: Lattice,
    width: usize,
    height: usize,
}

impl Grid {
    /// A `width × height` array.
    ///
    /// # Errors
    ///
    /// [`SomError::TooFew`] for a width or height of zero.
    pub fn new(lattice: Lattice, width: usize, height: usize) -> Result<Self, SomError> {
        if width == 0 {
            return Err(SomError::TooFew { what: "grid width", need: 1, got: 0 });
        }
        if height == 0 {
            return Err(SomError::TooFew { what: "grid height", need: 1, got: 0 });
        }
        Ok(Self { lattice, width, height })
    }

    /// A chain of `n` units: a rectangular array one unit high, whose lattice distance is `|i − j|`.
    ///
    /// # Errors
    ///
    /// [`SomError::TooFew`] for `n = 0`.
    pub fn chain(n: usize) -> Result<Self, SomError> {
        Self::new(Lattice::Rect, n, 1)
    }

    /// The lattice.
    #[must_use]
    pub fn lattice(&self) -> Lattice {
        self.lattice
    }

    /// Units per row.
    #[must_use]
    pub fn width(&self) -> usize {
        self.width
    }

    /// Rows.
    #[must_use]
    pub fn height(&self) -> usize {
        self.height
    }

    /// The number of units, `width × height`.
    #[must_use]
    pub fn len(&self) -> usize {
        self.width * self.height
    }

    /// Never true: a grid has at least one unit. Present because a `len` without it is a lint.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Column and row of unit `i`.
    #[must_use]
    pub fn coords(&self, i: usize) -> (usize, usize) {
        (i % self.width, i / self.width)
    }

    /// The squared distance between units `a` and `b` in the plane of the array.
    ///
    /// Computed as `SOM_PAK`'s `hexa_dist` and `rect_dist` compute it — `Δx² + ¾Δy²` on the
    /// hexagonal lattice, with `Δx` carrying the half-unit row offset — and therefore EXACT: every
    /// term is a half-integer squared or three quarters of an integer squared, so a hexagonal
    /// neighbour is at squared distance `1` exactly and not `0.9999999999999999`, which is the
    /// difference between inside a unit radius and outside it.
    #[must_use]
    pub fn distance2(&self, a: usize, b: usize) -> f64 {
        let (ax, ay) = self.coords(a);
        let (bx, by) = self.coords(b);
        let mut dx = ax as f64 - bx as f64;
        let dy = ay as f64 - by as f64;
        match self.lattice {
            Lattice::Rect => dx * dx + dy * dy,
            Lattice::Hex => {
                if ay % 2 != by % 2 {
                    dx += if ay % 2 == 0 { -0.5 } else { 0.5 };
                }
                dx * dx + 0.75 * dy * dy
            }
        }
    }

    /// The distance between units `a` and `b` in the plane of the array, `‖r_a − r_b‖`.
    #[must_use]
    pub fn distance(&self, a: usize, b: usize) -> f64 {
        self.distance2(a, b).sqrt()
    }

    /// Adjacent units: lattice distance exactly one — the four edge neighbours on a rectangular
    /// array, the six on a hexagonal one, as the SOM Toolbox's `som_unit_neighs` defines them.
    /// This is the adjacency [`Som::topographic_error`] counts.
    #[must_use]
    pub fn adjacent(&self, a: usize, b: usize) -> bool {
        self.distance2(a, b) == 1.0
    }

    /// The larger of the column and row offsets between two units: the 1982 paper's rings, in which
    /// "the eight of its nearest neighbours" are the units at one.
    #[must_use]
    pub fn chebyshev(&self, a: usize, b: usize) -> usize {
        let (ax, ay) = self.coords(a);
        let (bx, by) = self.coords(b);
        ax.abs_diff(bx).max(ay.abs_diff(by))
    }
}

/// The neighbourhood function `h_ci` of the 1990 paper.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kernel {
    /// Eqs. (6)–(7): `h_ci = α` for every unit within lattice distance `radius` of the winner (the
    /// set `N_c`), and zero outside it. `SOM_PAK` calls it "bubble".
    Bubble,
    /// Eq. (8): `h_ci = h₀ exp(−‖r_c − r_i‖²/σ²)`. ⚠ `SOM_PAK` evaluates `exp(−d²/2r²)`; its `r` is
    /// this `σ` divided by `√2`.
    Gaussian,
}

impl Kernel {
    /// `h_ci` at lattice distance `d` from the winner, for gain `alpha` and width `width` (the
    /// radius of `N_c` for [`Kernel::Bubble`], `σ` for [`Kernel::Gaussian`]).
    #[must_use]
    pub fn h(&self, d: f64, alpha: f64, width: f64) -> f64 {
        match self {
            Self::Bubble => {
                if d <= width {
                    alpha
                } else {
                    0.0
                }
            }
            Self::Gaussian => alpha * (-(d * d) / (width * width)).exp(),
        }
    }

    /// The width a kernel accepts: a radius may be zero (`N_c = {c}`, "simple competitive
    /// learning", p. 1467); `σ` may not, because eq. (8) divides by it.
    fn check_width(&self, width: f64) -> Result<f64, SomError> {
        match self {
            Self::Bubble => within("radius", width, 0.0, f64::INFINITY),
            Self::Gaussian => positive("sigma", width),
        }
    }
}

/// A value that changes over a training phase, indexed by the step `t = 0, 1, …`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Schedule {
    /// The same value at every step.
    Constant(f64),
    /// `end + (start − end)(T − t)/T`: `start` at `t = 0`, falling linearly to `end` at `t = T` and
    /// held there after it. With `end = 0` this is the 1990 paper's `α(t) = 0.9(1 − t/1000)`
    /// (p. 1469); with `end = 1` it is its radius "shrinking linearly to one". It is written this way
    /// round because it is how `SOM_PAK` writes `trad` and `linear_alpha`, and a reordering of the
    /// same line changes the last bit.
    Linear {
        /// The value at `t = 0`.
        start: f64,
        /// The value at `t = T` and after.
        end: f64,
        /// `T`, the length of the fall.
        steps: usize,
    },
    /// `start (end/start)^(t/T)`, held at `end` after `t = T`: the semantic map's
    /// `σ(t) = σ_i(σ_f/σ_i)^{t/t_max}` (p. 1475).
    Exponential {
        /// `σ_i`.
        start: f64,
        /// `σ_f`.
        end: f64,
        /// `t_max`.
        steps: usize,
    },
    /// `c/t` with the paper's count `t = 1, 2, 3, …`, so step `0` gets `c`: the 1990 paper's
    /// `α'(t) = 100/t` for eq. (10), and the 1982 paper's gain "proportional to 1/t" (p. 61), which
    /// that paper gives no constant for.
    InverseT {
        /// `c`.
        c: f64,
    },
}

impl Schedule {
    /// The value at step `t`.
    ///
    /// # Errors
    ///
    /// Whatever [`Schedule::check`] refuses: a schedule that cannot be evaluated has no value to
    /// return, and `0/0` from a linear fall of zero steps is not one.
    pub fn at(&self, t: usize) -> Result<f64, SomError> {
        self.check()?;
        Ok(match *self {
            Self::Constant(v) => v,
            Self::Linear { start, end, steps } => end + (start - end) * (steps - t.min(steps)) as f64 / steps as f64,
            Self::Exponential { start, end, steps } => start * (end / start).powf(t.min(steps) as f64 / steps as f64),
            Self::InverseT { c } => c / (t as f64 + 1.0),
        })
    }

    /// Finite parameters, a positive length, and for [`Schedule::Exponential`] and
    /// [`Schedule::InverseT`] positive values.
    ///
    /// # Errors
    ///
    /// [`SomError::NonFinite`] for a value that is not finite; [`SomError::TooFew`] for a fall of
    /// zero steps; [`SomError::NotPositive`] for an exponential end or an inverse-`t` constant that
    /// is not finite and positive.
    pub fn check(&self) -> Result<(), SomError> {
        match *self {
            Self::Constant(v) => finite("schedule value", v).map(drop),
            Self::Linear { start, end, steps } => {
                finite("schedule start", start)?;
                finite("schedule end", end)?;
                if steps == 0 {
                    return Err(SomError::TooFew { what: "schedule steps", need: 1, got: 0 });
                }
                Ok(())
            }
            Self::Exponential { start, end, steps } => {
                positive("schedule start", start)?;
                positive("schedule end", end)?;
                if steps == 0 {
                    return Err(SomError::TooFew { what: "schedule steps", need: 1, got: 0 });
                }
                Ok(())
            }
            Self::InverseT { c } => positive("schedule constant", c).map(drop),
        }
    }
}

/// One training phase: a number of steps, the gain and width schedules, and the kernel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Phase {
    /// Steps in the phase.
    pub steps: usize,
    /// `α(t)` — or `h₀(t)` for the Gaussian kernel.
    pub alpha: Schedule,
    /// The radius of `N_c(t)`, or `σ(t)`.
    pub width: Schedule,
    /// Which `h_ci`.
    pub kernel: Kernel,
}

impl Phase {
    /// The 1990 paper's ordering phase (Sect. II-D, hints 2 and 3, p. 1469): 1000 steps,
    /// `α(t) = 0.9(1 − t/1000)`, and a bubble neighbourhood whose radius starts at `radius` — "can
    /// even be more than half the diameter of the network" — and shrinks linearly to one.
    #[must_use]
    pub fn ordering_1990(radius: f64) -> Self {
        Self {
            steps: 1000,
            alpha: Schedule::Linear { start: 0.9, end: 0.0, steps: 1000 },
            width: Schedule::Linear { start: radius, end: 1.0, steps: 1000 },
            kernel: Kernel::Bubble,
        }
    }

    /// The 1990 paper's convergence phase for `grid`, from its hints 1 to 3 (p. 1469) with the
    /// choices it leaves open made and named: `500 × units` steps — hint 1's "at least 500 times the
    /// number of network units", given to this phase alone so that the two phases together clear
    /// it — with `α` falling linearly from 0.01, hint 2's "of the order of or less than .01", to
    /// zero, and `N_c` holding "the nearest neighbors of cell c", a radius of one.
    #[must_use]
    pub fn convergence_1990(grid: Grid) -> Self {
        let steps = 500 * grid.len();
        Self {
            steps,
            alpha: Schedule::Linear { start: 0.01, end: 0.0, steps },
            width: Schedule::Constant(1.0),
            kernel: Kernel::Bubble,
        }
    }

    /// The two phases that made Fig. 6 from Table 1 (p. 1470): 1000 steps with `α` falling linearly
    /// from 0.5 to 0.04 and the radius from six to one, then 10 000 steps with `α` falling from 0.04
    /// to zero at radius one.
    #[must_use]
    pub fn table1_1990() -> [Self; 2] {
        [
            Self {
                steps: 1000,
                alpha: Schedule::Linear { start: 0.5, end: 0.04, steps: 1000 },
                width: Schedule::Linear { start: 6.0, end: 1.0, steps: 1000 },
                kernel: Kernel::Bubble,
            },
            Self {
                steps: 10_000,
                alpha: Schedule::Linear { start: 0.04, end: 0.0, steps: 10_000 },
                width: Schedule::Constant(1.0),
                kernel: Kernel::Bubble,
            },
        ]
    }
}

/// How a phase picks the next training vector.
#[derive(Debug)]
pub enum Order<'a> {
    /// The data in order, from the start again after the last one: what `SOM_PAK`'s `vsom` does
    /// without `-rand`, and so the order the reference comparison uses.
    Cyclic,
    /// Uniformly at random with replacement: "The vectors x(t) were drawn from this density function
    /// independently and at random" (1990, p. 1467).
    Random(&'a mut Rng),
}

/// A self-organising map in its 1990 form: Euclidean winner, eqs. (2') and (6)–(8).
#[derive(Debug, Clone, PartialEq)]
pub struct Som {
    grid: Grid,
    dim: usize,
    weights: Vec<f64>,
}

/// The number of samples in a flat data slice, refusing a length that is not a whole number of
/// samples, a slice with none, and a datum that is not finite.
fn samples(dim: usize, data: &[f64]) -> Result<usize, SomError> {
    if !data.len().is_multiple_of(dim) {
        return Err(SomError::Length { what: "data (a multiple of the dimension)", expected: dim, got: data.len() });
    }
    if data.is_empty() {
        return Err(SomError::TooFew { what: "data samples", need: 1, got: 0 });
    }
    all_finite("data", data)?;
    Ok(data.len() / dim)
}

impl Som {
    /// A map on `grid` whose unit `i` has weight vector `weights[i·dim..(i + 1)·dim]`.
    ///
    /// # Errors
    ///
    /// [`SomError::TooFew`] for a dimension of zero; [`SomError::Length`] unless there are exactly
    /// `grid.len() × dim` weights; [`SomError::NonFinite`] for a weight that is not finite.
    pub fn new(grid: Grid, dim: usize, weights: Vec<f64>) -> Result<Self, SomError> {
        if dim == 0 {
            return Err(SomError::TooFew { what: "dimension", need: 1, got: 0 });
        }
        if weights.len() != grid.len() * dim {
            return Err(SomError::Length { what: "weights", expected: grid.len() * dim, got: weights.len() });
        }
        all_finite("weight", &weights)?;
        Ok(Self { grid, dim, weights })
    }

    /// A map whose every weight is drawn uniformly from `[low, high)`: the 1990 paper's "random,
    /// initial values for the `m_i(0)`, the only restriction being that they should be different"
    /// (p. 1468).
    ///
    /// # Errors
    ///
    /// [`SomError::NonFinite`] for a bound that is not finite; [`SomError::EmptyInterval`] for
    /// `high <= low`; [`SomError::TooFew`] for a dimension of zero.
    pub fn random(grid: Grid, dim: usize, low: f64, high: f64, rng: &mut Rng) -> Result<Self, SomError> {
        finite("low", low)?;
        finite("high", high)?;
        if !(high > low) {
            return Err(SomError::EmptyInterval { low, high });
        }
        let weights = (0..grid.len() * dim).map(|_| low + (high - low) * rng.next_f64()).collect();
        Self::new(grid, dim, weights)
    }

    /// The grid.
    #[must_use]
    pub fn grid(&self) -> Grid {
        self.grid
    }

    /// The input dimension `n`.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Every weight, unit after unit.
    #[must_use]
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Unit `i`'s weight vector `m_i`.
    ///
    /// # Panics
    ///
    /// If `i` is not a unit of the grid.
    #[must_use]
    pub fn unit(&self, i: usize) -> &[f64] {
        &self.weights[i * self.dim..(i + 1) * self.dim]
    }

    fn check_x(&self, x: &[f64]) -> Result<(), SomError> {
        if x.len() != self.dim {
            return Err(SomError::Length { what: "x", expected: self.dim, got: x.len() });
        }
        all_finite("x", x)
    }

    fn dist2(&self, i: usize, x: &[f64]) -> f64 {
        self.unit(i).iter().zip(x).map(|(m, v)| (v - m) * (v - m)).sum()
    }

    /// The two best-matching units, nearest first, and the nearest one's squared distance. Ties go
    /// to the lower index, as they do in `SOM_PAK`'s strict `<`.
    fn best_two(&self, x: &[f64]) -> (usize, usize, f64) {
        let (mut c, mut dc) = (0, f64::INFINITY);
        let (mut s, mut ds) = (0, f64::INFINITY);
        for i in 0..self.grid.len() {
            let d = self.dist2(i, x);
            if d < dc {
                (s, ds) = (c, dc);
                (c, dc) = (i, d);
            } else if d < ds {
                (s, ds) = (i, d);
            }
        }
        (c, s, dc)
    }

    /// The winner `c`, eq. (2') of the 1990 paper: `‖x − m_c‖ = minᵢ ‖x − m_i‖`, the lower index on
    /// a tie.
    ///
    /// # Errors
    ///
    /// [`SomError::Length`] for an `x` of the wrong dimension; [`SomError::NonFinite`] for one with a
    /// component that is not finite.
    pub fn winner(&self, x: &[f64]) -> Result<usize, SomError> {
        self.check_x(x)?;
        Ok(self.best_two(x).0)
    }

    /// One step of eq. (7), `m_i(t + 1) = m_i(t) + h_ci(t)[x(t) − m_i(t)]`, with `h_ci` from `kernel`
    /// at gain `alpha` and width `width`. Returns the winner.
    ///
    /// # Errors
    ///
    /// [`SomError::OutOfRange`] for an `alpha` outside `[0, 1]` — the paper's `0 < α(t) < 1`, with
    /// both ends admitted: `α = 0` is where a linear schedule ends, and `α = 1` puts the winner on
    /// the sample, which is what a running mean does at its first sample — or a negative radius;
    /// [`SomError::NotPositive`] for a `σ` that is not finite and positive; whatever
    /// [`Som::winner`] refuses.
    pub fn step(&mut self, x: &[f64], kernel: Kernel, alpha: f64, width: f64) -> Result<usize, SomError> {
        self.check_x(x)?;
        within("alpha", alpha, 0.0, 1.0)?;
        kernel.check_width(width)?;
        let (c, _, _) = self.best_two(x);
        for i in 0..self.grid.len() {
            let h = kernel.h(self.grid.distance(c, i), alpha, width);
            let m = &mut self.weights[i * self.dim..(i + 1) * self.dim];
            for (mj, xj) in m.iter_mut().zip(x) {
                *mj += h * (xj - *mj);
            }
        }
        Ok(c)
    }

    /// Run one phase over `data` (samples of length [`Som::dim`], laid end to end), taking samples
    /// in `order`. Every step's gain and width are checked before the first step is taken, so a
    /// refused phase leaves the map as it was.
    ///
    /// # Errors
    ///
    /// [`SomError::Length`] or [`SomError::TooFew`] for a data slice that is not a whole, non-zero
    /// number of samples; [`SomError::NonFinite`] for a datum that is not finite; whatever
    /// [`Schedule::at`] refuses; and what [`Som::step`] refuses of any step's gain or width.
    pub fn train(&mut self, data: &[f64], phase: &Phase, order: Order<'_>) -> Result<(), SomError> {
        let n = samples(self.dim, data)?;
        for t in 0..phase.steps {
            within("alpha", phase.alpha.at(t)?, 0.0, 1.0)?;
            phase.kernel.check_width(phase.width.at(t)?)?;
        }
        let mut order = order;
        for t in 0..phase.steps {
            let k = match &mut order {
                Order::Cyclic => t % n,
                Order::Random(rng) => draw(rng, n),
            };
            let x = &data[k * self.dim..(k + 1) * self.dim];
            self.step(x, phase.kernel, phase.alpha.at(t)?, phase.width.at(t)?)?;
        }
        Ok(())
    }

    /// The mean quantisation error `(1/N) Σ ‖x − m_c‖` over `data`: the average distance from each
    /// sample to its winner, which is what `SOM_PAK`'s `qerror` prints "per sample".
    ///
    /// # Errors
    ///
    /// As [`Som::train`], for the data.
    pub fn quantization_error(&self, data: &[f64]) -> Result<f64, SomError> {
        let n = samples(self.dim, data)?;
        let sum: f64 = data.chunks_exact(self.dim).map(|x| self.best_two(x).2.sqrt()).sum();
        Ok(sum / n as f64)
    }

    /// The topographic error: the fraction of `data` whose best- and second-best-matching units are
    /// not adjacent on the grid ([`Grid::adjacent`]).
    ///
    /// # Errors
    ///
    /// [`SomError::TooFew`] for a map of one unit, which has no second-best match; as
    /// [`Som::train`], for the data.
    pub fn topographic_error(&self, data: &[f64]) -> Result<f64, SomError> {
        if self.grid.len() < 2 {
            return Err(SomError::TooFew { what: "units for a second-best match", need: 2, got: self.grid.len() });
        }
        let n = samples(self.dim, data)?;
        let apart = data
            .chunks_exact(self.dim)
            .filter(|x| {
                let (c, s, _) = self.best_two(x);
                !self.grid.adjacent(c, s)
            })
            .count();
        Ok(apart as f64 / n as f64)
    }

    /// Whether a chain of scalars is ordered: its weights strictly increasing or strictly decreasing
    /// along the chain. This is the 1982 paper's "one-dimensional ordered mapping" (p. 60) read off
    /// the weights: for scalar inputs, a strictly monotone chain sends increasing inputs to units
    /// in increasing (or decreasing) order.
    ///
    /// # Errors
    ///
    /// [`SomError::Length`] for a map whose weights are not scalars or whose grid is not one unit
    /// high.
    pub fn is_ordered_chain(&self) -> Result<bool, SomError> {
        if self.dim != 1 {
            return Err(SomError::Length { what: "weight vector of a scalar chain", expected: 1, got: self.dim });
        }
        if self.grid.height != 1 {
            return Err(SomError::Length { what: "height of a chain", expected: 1, got: self.grid.height });
        }
        let w = &self.weights;
        Ok(w.windows(2).all(|p| p[0] < p[1]) || w.windows(2).all(|p| p[0] > p[1]))
    }
}

/// Which units a step of the 1982 rule updates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Reach {
    /// The winner and "all the eight of its nearest neighbours (except at the edges of the array
    /// where the number of neighbours was different)" — Simulations 1 and 2 (p. 61) — which on a
    /// chain are its "two nearest neighbours" (Simulation 4, p. 62). Every unit at ring distance at
    /// most one ([`Grid::chebyshev`]), at the full gain.
    Nearest,
    /// Simulation 3 (p. 62): eq. (3) "as such to the selected unit and its nearest eight neighbours
    /// while using an adaptation gain value of α/4 for those 16 units which surrounded the previous
    /// ones" — ring one at `α`, ring two at `α/4`.
    Rings,
    /// The 1990 paper's inner-product variant, eqs. (9)–(10): the same normalised step for every
    /// unit within lattice distance `radius` of the winner, the set `N_c` of its eq. (6).
    Radius(f64),
}

/// Kohonen's rule as the 1982 paper prints it: inner-product winner and normalised updates.
///
/// ```text
/// η_i = Σ_j μ_ij ξ_j = m_iᵀ x                         (1)
/// η_k = max_i {η_i}                                   (2)
/// m_i(t + 1) = (m_i(t) + αx(t)) / ‖m_i(t) + αx(t)‖_E   (3)
/// ```
///
/// for `k` and its neighbours ([`Reach`]); every other unit is left as it is.
#[derive(Debug, Clone, PartialEq)]
pub struct Kohonen1982 {
    grid: Grid,
    dim: usize,
    weights: Vec<f64>,
}

impl Kohonen1982 {
    /// A map on `grid` with the given weights, unit after unit. The paper's initial values are
    /// "random numbers" (p. 61); they need not be normalised, since the first step through eq. (3)
    /// normalises every unit it touches.
    ///
    /// # Errors
    ///
    /// As [`Som::new`].
    pub fn new(grid: Grid, dim: usize, weights: Vec<f64>) -> Result<Self, SomError> {
        let s = Som::new(grid, dim, weights)?;
        Ok(Self { grid: s.grid, dim: s.dim, weights: s.weights })
    }

    /// Every weight, unit after unit.
    #[must_use]
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// The grid.
    #[must_use]
    pub fn grid(&self) -> Grid {
        self.grid
    }

    /// Unit `i`'s weight vector `m_i`.
    ///
    /// # Panics
    ///
    /// If `i` is not a unit of the grid.
    #[must_use]
    pub fn unit(&self, i: usize) -> &[f64] {
        &self.weights[i * self.dim..(i + 1) * self.dim]
    }

    /// Unit `i`'s response to `x`, eq. (1): `η_i = m_iᵀ x`.
    ///
    /// # Panics
    ///
    /// If `i` is not a unit of the grid.
    #[must_use]
    pub fn response(&self, i: usize, x: &[f64]) -> f64 {
        self.unit(i).iter().zip(x).map(|(m, v)| m * v).sum()
    }

    fn check_inputs(&self, inputs: &[f64], units: usize) -> Result<(), SomError> {
        if inputs.len() != units * self.dim {
            return Err(SomError::Length { what: "inputs", expected: units * self.dim, got: inputs.len() });
        }
        all_finite("x", inputs)
    }

    /// The winner of eq. (2) for an input `x` that reaches every unit, the lower index on a tie.
    ///
    /// # Errors
    ///
    /// [`SomError::Length`] for an `x` of the wrong dimension; [`SomError::NonFinite`] for one with a
    /// component that is not finite.
    pub fn winner(&self, x: &[f64]) -> Result<usize, SomError> {
        self.check_inputs(x, 1)?;
        Ok(self.winner_of(|_| x))
    }

    /// The winner of eq. (2) when unit `i` reads `inputs[i·dim..(i + 1)·dim]` (Sect. 2.3).
    ///
    /// # Errors
    ///
    /// [`SomError::Length`] unless there is one input per unit; [`SomError::NonFinite`] for a
    /// component that is not finite.
    pub fn winner_coherent(&self, inputs: &[f64]) -> Result<usize, SomError> {
        self.check_inputs(inputs, self.grid.len())?;
        let dim = self.dim;
        Ok(self.winner_of(|i| &inputs[i * dim..(i + 1) * dim]))
    }

    fn winner_of<'x>(&self, input: impl Fn(usize) -> &'x [f64]) -> usize {
        let (mut k, mut best) = (0, f64::NEG_INFINITY);
        for i in 0..self.grid.len() {
            let eta = self.response(i, input(i));
            if eta > best {
                (k, best) = (i, eta);
            }
        }
        k
    }

    /// The gain eq. (3) applies to unit `i` when `k` won, or `None` outside the reach.
    fn gain(&self, k: usize, i: usize, alpha: f64, reach: Reach) -> Option<f64> {
        match reach {
            Reach::Nearest => (self.grid.chebyshev(k, i) <= 1).then_some(alpha),
            Reach::Rings => match self.grid.chebyshev(k, i) {
                0 | 1 => Some(alpha),
                2 => Some(alpha / 4.0),
                _ => None,
            },
            Reach::Radius(r) => (self.grid.distance(k, i) <= r).then_some(alpha),
        }
    }

    fn check_step(&self, alpha: f64, reach: Reach) -> Result<(), SomError> {
        positive("alpha", alpha)?;
        match reach {
            Reach::Radius(r) => within("radius", r, 0.0, f64::INFINITY).map(drop),
            Reach::Nearest | Reach::Rings if self.grid.lattice == Lattice::Hex => {
                Err(SomError::Lattice { what: "the 1982 paper's ring of nearest neighbours" })
            }
            Reach::Nearest | Reach::Rings => Ok(()),
        }
    }

    fn apply<'x>(&mut self, input: impl Fn(usize) -> &'x [f64], alpha: f64, reach: Reach) -> Result<usize, SomError> {
        let k = self.winner_of(&input);
        // Every new vector is computed before any is stored, so a zero norm refuses the whole step
        // rather than leaving half the neighbourhood updated.
        let mut new = Vec::new();
        for i in 0..self.grid.len() {
            if let Some(a) = self.gain(k, i, alpha, reach) {
                let v: Vec<f64> = self.unit(i).iter().zip(input(i)).map(|(m, x)| m + a * x).collect();
                let norm = v.iter().map(|c| c * c).sum::<f64>().sqrt();
                if !(norm > 0.0) {
                    return Err(SomError::ZeroNorm { unit: i });
                }
                new.push((i, v.into_iter().map(|c| c / norm).collect::<Vec<f64>>()));
            }
        }
        for (i, v) in new {
            self.weights[i * self.dim..(i + 1) * self.dim].copy_from_slice(&v);
        }
        Ok(k)
    }

    /// One step of eq. (3) with the same input `x` at every unit (Sect. 2.2). Returns the winner.
    ///
    /// # Errors
    ///
    /// [`SomError::NotPositive`] for an `alpha` that is not finite and positive — eq. (10) of the
    /// 1990 paper states `0 < α'(t) < ∞`; [`SomError::OutOfRange`] for a negative radius;
    /// [`SomError::Lattice`] for [`Reach::Nearest`] or [`Reach::Rings`] on a hexagonal grid;
    /// [`SomError::ZeroNorm`] when `m_i + αx` vanishes; whatever [`Kohonen1982::winner`] refuses.
    pub fn step(&mut self, x: &[f64], alpha: f64, reach: Reach) -> Result<usize, SomError> {
        self.check_inputs(x, 1)?;
        self.check_step(alpha, reach)?;
        self.apply(|_| x, alpha, reach)
    }

    /// One step of eq. (3) with "non-identical but coherent inputs" (Sect. 2.3, Simulation 4): unit
    /// `i` reads `inputs[i·dim..(i + 1)·dim]`, its own selection of signals caused by the same event,
    /// and the winner is the unit whose response to ITS input is largest. Returns the winner.
    ///
    /// # Errors
    ///
    /// [`SomError::Length`] unless there is one input per unit; otherwise as [`Kohonen1982::step`].
    pub fn step_coherent(&mut self, inputs: &[f64], alpha: f64, reach: Reach) -> Result<usize, SomError> {
        self.check_inputs(inputs, self.grid.len())?;
        self.check_step(alpha, reach)?;
        let dim = self.dim;
        self.apply(|i| &inputs[i * dim..(i + 1) * dim], alpha, reach)
    }
}

#[cfg(test)]
mod tests {
    use super::{Grid, Kernel, Kohonen1982, Lattice, Order, Phase, Reach, Schedule, Som, SomError};
    use crate::rng::Rng;

    // ----------------------------------------------------------------------------------------------
    // The reference: SOM_PAK 3.1, run by `tools/som_reference.py`, which prints this block verbatim.
    // ----------------------------------------------------------------------------------------------

    // chain: (1, 'rect', 8, 1, 'bubble', 16, [(200, 0.5, 4.0), (800, 0.03125, 1.0)], 1990)
    const CHAIN_DATA: [u16; 16] = [26, 696, 173, 268, 93, 53, 293, 187, 6, 409, 960, 139, 932, 383, 1022, 774];
    const CHAIN_CODE: [u16; 8] = [584, 583, 101, 642, 270, 36, 913, 454];
    const CHAIN_PHASE1_D: [f64; 8] = [0.881687501711597, 0.832632127245044, 0.5678412595229725, 0.3956372725558799, 0.2852181997097638, 0.18827934050204617, 0.09678009267840423, 0.06679178532335894];
    const CHAIN_PHASE2_D: [f64; 8] = [0.8990602547065836, 0.8548156561549437, 0.5537562913021832, 0.3998488589156004, 0.2791232888098727, 0.1876248248855892, 0.09440956636509551, 0.06199655858832643];
    const CHAIN_PHASE1_F: [f64; 8] = [0.881687522, 0.832632124, 0.567841232, 0.395637274, 0.285218209, 0.188279331, 0.0967800841, 0.0667917803];
    const CHAIN_PHASE2_F: [f64; 8] = [0.89906019, 0.854815543, 0.553756177, 0.399848819, 0.279123396, 0.187624797, 0.0944096074, 0.0619965494];
    const CHAIN_QE: [f64; 4] = [0.03914686872259059, 0.03722968869377302, 0.0391468629, 0.0372296907];
    const CHAIN_FLOAT_PARTS: Option<(usize, usize, usize, usize)> = None;
    // hexgauss: (2, 'hexa', 4, 3, 'gaussian', 24, [(300, 0.5, 3.0), (600, 0.0625, 1.0)], 1982)
    const HEXGAUSS_DATA: [u16; 48] = [160, 30, 514, 106, 931, 850, 574, 1004, 755, 798, 496, 764, 909, 467, 601, 628, 991, 695, 506, 362, 790, 303, 240, 998, 165, 6, 619, 180, 986, 653, 831, 597, 995, 205, 1022, 602, 1009, 258, 467, 1023, 951, 90, 869, 202, 524, 604, 602, 83];
    const HEXGAUSS_CODE: [u16; 24] = [849, 58, 1022, 420, 916, 456, 9, 1012, 616, 263, 407, 439, 169, 481, 658, 435, 497, 936, 688, 240, 239, 38, 955, 482];
    const HEXGAUSS_PHASE1_D: [f64; 24] = [0.5514761432188203, 0.21360978689830348, 0.5684377046257861, 0.3639309424011489, 0.5917548306549165, 0.5865539163618491, 0.6054541841039996, 0.7280120861039431, 0.6506682840558851, 0.2802997818039973, 0.6695526406407324, 0.4718053802562374, 0.6889873197745877, 0.6505369325788414, 0.6978552805775656, 0.7309373411934846, 0.7458569489083702, 0.24108861690822078, 0.7542816087475287, 0.3676852257788466, 0.7715026289353202, 0.5524613478537124, 0.7834321798018843, 0.6662342699930216];
    const HEXGAUSS_PHASE2_D: [f64; 24] = [0.4878304750802045, 0.14869544634247842, 0.5245623502776462, 0.3198484426424179, 0.5579707010655514, 0.6284705601217367, 0.5614573817840706, 0.7904869199808577, 0.6441703633265473, 0.22504911428995722, 0.6694599693033809, 0.45427087250671105, 0.6940690548895188, 0.6808774131286004, 0.7018740816407207, 0.7617369456523574, 0.7938627722409857, 0.20132249791672852, 0.7859306645389867, 0.32292778966986807, 0.8111870834325667, 0.5521314426998135, 0.8348930272284857, 0.6755066201634868];
    const HEXGAUSS_PHASE1_F: [f64; 24] = [0.551476181, 0.21360974, 0.568437755, 0.363930941, 0.591754854, 0.58655405, 0.605454266, 0.728012323, 0.650668323, 0.280299783, 0.669552624, 0.471805423, 0.688987434, 0.650537074, 0.697855175, 0.730937183, 0.745857, 0.241088554, 0.75428158, 0.367685288, 0.771502674, 0.552461386, 0.783432186, 0.666234434];
    const HEXGAUSS_PHASE2_F: [f64; 24] = [0.48783049, 0.148695454, 0.524562359, 0.319848448, 0.557970643, 0.628470361, 0.561457098, 0.790486872, 0.644170225, 0.225049183, 0.669459879, 0.454270661, 0.694069028, 0.680877864, 0.701874018, 0.761737168, 0.793862581, 0.201322436, 0.785930634, 0.322927833, 0.811187088, 0.552131236, 0.834892809, 0.67550689];
    const HEXGAUSS_QE: [f64; 4] = [0.18595677640140185, 0.1425249258308842, 0.185956761, 0.142524943];
    const HEXGAUSS_FLOAT_PARTS: Option<(usize, usize, usize, usize)> = None;
    // rect3: (3, 'rect', 5, 4, 'bubble', 40, [(500, 0.5, 5.0), (1500, 0.03125, 1.0)], 1995)
    const RECT3_DATA: [u16; 120] = [930, 358, 40, 928, 918, 730, 605, 761, 297, 297, 846, 453, 967, 395, 764, 847, 946, 449, 771, 551, 531, 532, 995, 583, 525, 276, 156, 576, 837, 241, 358, 631, 957, 609, 650, 261, 902, 341, 508, 513, 148, 509, 354, 624, 231, 786, 728, 444, 760, 446, 843, 223, 165, 386, 597, 596, 212, 983, 947, 415, 257, 342, 283, 777, 805, 974, 495, 595, 655, 303, 717, 241, 737, 52, 361, 466, 265, 745, 560, 305, 934, 152, 909, 539, 710, 633, 760, 829, 999, 696, 432, 316, 741, 118, 209, 704, 129, 577, 522, 146, 265, 146, 698, 143, 20, 500, 299, 980, 767, 383, 96, 704, 934, 945, 263, 241, 484, 751, 719, 679];
    const RECT3_CODE: [u16; 60] = [637, 569, 621, 293, 684, 454, 181, 953, 561, 638, 72, 733, 152, 833, 78, 341, 664, 70, 960, 38, 527, 588, 41, 538, 39, 310, 419, 277, 1002, 829, 464, 384, 600, 140, 912, 769, 509, 763, 60, 281, 189, 399, 26, 713, 108, 1019, 261, 398, 590, 456, 745, 242, 358, 593, 484, 923, 2, 181, 179, 530];
    const RECT3_PHASE1_D: [f64; 60] = [0.7686495730125463, 0.831636924310383, 0.6440025026243265, 0.7524104526645238, 0.7197578567810272, 0.6889047804479144, 0.6688394392014072, 0.5307832467486688, 0.7662462730731688, 0.5678361131327514, 0.3575991129511434, 0.7448011959834308, 0.46783091442749675, 0.2628825072423544, 0.7097735616927539, 0.6919873915537771, 0.8382421355391177, 0.5967484856092448, 0.6413213027277741, 0.7450414840777378, 0.6438295846785441, 0.5755472035046967, 0.5338862254456751, 0.6846932064923269, 0.4768293162964689, 0.32604410098116554, 0.6593760218272396, 0.4390532991393612, 0.2369333117549748, 0.6220752723447103, 0.5407999075795448, 0.7584483155587931, 0.40200970627073895, 0.4396139872987032, 0.7061787893565427, 0.43173616013907473, 0.3889223592908231, 0.5400803930065537, 0.4367427822852978, 0.4375070847064017, 0.3105101094180379, 0.3302397085143519, 0.49455665194169324, 0.24029721842980079, 0.29608810804366015, 0.4871931146624058, 0.736396367365094, 0.3271895365328034, 0.4405005080590571, 0.703171644667869, 0.3270275881703258, 0.35045858626175347, 0.5445805489007887, 0.319075933323212, 0.4682660175009544, 0.3214792571610986, 0.22807909133263635, 0.5188940086887958, 0.258529886291136, 0.21157764138974927];
    const RECT3_PHASE2_D: [f64; 60] = [0.8107422901181106, 0.8730513968778597, 0.6793240703633081, 0.7342328720560614, 0.7976368354942999, 0.721008994850544, 0.7016558249054244, 0.5890633904405221, 0.7442189106753249, 0.7112895550196291, 0.40794671405313276, 0.7427909140133193, 0.5687858687080329, 0.31134179747781526, 0.7565446356607175, 0.7950994917731022, 0.8989099826123432, 0.541425520407112, 0.5758233768660449, 0.8116538778291126, 0.648657634206447, 0.49644464062806415, 0.6364478164594893, 0.6551802076756106, 0.4375540466835988, 0.34199447615307815, 0.5860316287576092, 0.41903663136468217, 0.24847334985352845, 0.7397332254303942, 0.5918134581296637, 0.7834213963513973, 0.3597189721989187, 0.3554627942896627, 0.7741226164558248, 0.45206810746550524, 0.2629667273124835, 0.5074014595459178, 0.4546661669588649, 0.28720897482213326, 0.28291959369745506, 0.3445701546435634, 0.4529700362197025, 0.23534429789890357, 0.36391114352047665, 0.5602034086376834, 0.6987921740857164, 0.2758261070058155, 0.42582267318847533, 0.724720364436329, 0.30241850415172217, 0.3204500401832502, 0.5384718257942956, 0.2854045084216278, 0.4793515321771436, 0.27842837923472996, 0.21521995430810983, 0.6772696084870294, 0.22559864448466072, 0.18404616156279277];
    const RECT3_PHASE1_F: [f64; 60] = [0.499056607, 0.243246675, 0.246152252, 0.466977149, 0.328766674, 0.241013721, 0.326002777, 0.582697868, 0.315317065, 0.400061548, 0.723502874, 0.349914879, 0.462444037, 0.776897728, 0.356791347, 0.475627273, 0.250914693, 0.361347497, 0.460600287, 0.325354129, 0.360282063, 0.429515928, 0.566299438, 0.370778233, 0.527763963, 0.720192671, 0.391059905, 0.557143629, 0.80304563, 0.383018047, 0.405413836, 0.310480267, 0.679737747, 0.527612269, 0.378822953, 0.680684388, 0.700792491, 0.563405454, 0.606548965, 0.749546468, 0.719631195, 0.56576097, 0.740103841, 0.829978347, 0.532942653, 0.436589062, 0.346774161, 0.785590589, 0.580271304, 0.435934037, 0.768776059, 0.740336478, 0.565320313, 0.715104938, 0.79293859, 0.717398584, 0.672334909, 0.799010754, 0.849080026, 0.632666886];
    const RECT3_PHASE2_F: [f64; 60] = [0.666160524, 0.222716972, 0.193191454, 0.435479879, 0.336572945, 0.240879998, 0.311056584, 0.559472561, 0.289285839, 0.401956528, 0.731092751, 0.347978562, 0.346933812, 0.868591785, 0.472103059, 0.453522801, 0.235537469, 0.365238339, 0.319208413, 0.248612732, 0.326984614, 0.460198849, 0.494045019, 0.350305527, 0.605867088, 0.71051234, 0.301388651, 0.566133797, 0.814292252, 0.375348359, 0.41212374, 0.299026102, 0.77204591, 0.459540159, 0.339831889, 0.555559397, 0.791677713, 0.522393882, 0.57457, 0.739114761, 0.697184384, 0.500484586, 0.829099238, 0.888893306, 0.541029871, 0.449171126, 0.371999353, 0.801556468, 0.577748835, 0.466815889, 0.827745497, 0.736094654, 0.585751355, 0.724720597, 0.775557339, 0.777311862, 0.729421973, 0.809813261, 0.874756992, 0.680048645];
    const RECT3_QE: [f64; 4] = [0.21748133569808661, 0.1620312225235439, 0.21283789, 0.166318938];
    const RECT3_FLOAT_PARTS: Option<(usize, usize, usize, usize)> = Some((1, 32, 14, 1));
    // hexbubble: (2, 'hexa', 6, 5, 'bubble', 50, [(1000, 0.5, 4.0), (3000, 0.03125, 1.0)], 2026)
    const HEXBUBBLE_DATA: [u16; 100] = [243, 654, 210, 457, 861, 1005, 903, 491, 5, 165, 226, 588, 200, 920, 23, 1004, 643, 430, 813, 515, 712, 730, 770, 156, 697, 183, 598, 589, 936, 288, 632, 50, 755, 744, 944, 864, 185, 816, 1020, 238, 871, 1005, 815, 529, 860, 981, 51, 459, 264, 100, 210, 879, 962, 291, 987, 468, 249, 935, 961, 514, 823, 630, 980, 823, 886, 414, 602, 550, 18, 67, 306, 938, 1002, 737, 409, 532, 994, 955, 1018, 889, 68, 594, 648, 937, 201, 897, 530, 927, 374, 871, 981, 131, 545, 1020, 758, 197, 398, 842, 299, 920];
    const HEXBUBBLE_CODE: [u16; 60] = [28, 188, 0, 924, 170, 407, 507, 247, 91, 549, 731, 316, 10, 641, 1007, 276, 456, 968, 268, 595, 172, 855, 327, 277, 626, 843, 799, 304, 402, 770, 28, 217, 797, 150, 105, 710, 723, 52, 52, 974, 671, 621, 921, 667, 248, 1019, 955, 349, 315, 423, 888, 688, 939, 477, 142, 817, 69, 400, 355, 92];
    const HEXBUBBLE_PHASE1_D: [f64; 60] = [0.20530877182259977, 0.8843475024186983, 0.2841387038961082, 0.8924286234060866, 0.41366486942692643, 0.8946237007306673, 0.6988223230190089, 0.8888958349697234, 0.8325249836483843, 0.8784924636836626, 0.9196484964010287, 0.8820064310734668, 0.22537005395115778, 0.8113626129850569, 0.35462299180443635, 0.8137624966126542, 0.5514604670111589, 0.7662986394951632, 0.7462423431194705, 0.7340507086032773, 0.8691779627171776, 0.7641489062255985, 0.949043827863386, 0.787265945576358, 0.17839050407484744, 0.5759078858803409, 0.25705134748799235, 0.6257070291009201, 0.4728935482792141, 0.6262215471759576, 0.6813210973246545, 0.5716932327734612, 0.7783012792139176, 0.561323468179116, 0.8895113334403241, 0.5609746706919683, 0.15069619096016285, 0.38489713286336413, 0.33846511079901676, 0.36366210628801826, 0.6334403819262953, 0.3212952848594033, 0.757427213158993, 0.3847729149780875, 0.893497010913551, 0.4018735606676474, 0.926095906923811, 0.3848577837328778, 0.11674811469877608, 0.27299335490179744, 0.19294531602712772, 0.226265497231931, 0.5515146108887806, 0.1816558927728593, 0.7238851859665731, 0.19184741397086855, 0.8565860038963405, 0.2580856982646188, 0.9297408130754579, 0.3279469790062587];
    const HEXBUBBLE_PHASE2_D: [f64; 60] = [0.2041042107039416, 0.89231077364683, 0.28616895448657304, 0.8927586752582811, 0.4250512393987664, 0.9008299905739496, 0.7062295024316263, 0.8972382142606296, 0.8314431403681088, 0.8803443807023944, 0.919335162199777, 0.8862054885144922, 0.22372671707729694, 0.8158172383202981, 0.34100043168379596, 0.793082170664368, 0.544012778727135, 0.7565798284064973, 0.7419720990902738, 0.7294099017753514, 0.8670570064731471, 0.7432734367228364, 0.9490589266823175, 0.7363731816251935, 0.16002927481618293, 0.5805026083398543, 0.23449146715456307, 0.6304130602043788, 0.42700917879365297, 0.6167787919848016, 0.6692698566515435, 0.569897655024966, 0.7680186401820018, 0.557183256281105, 0.8867053129220933, 0.5562327025030286, 0.132573283529541, 0.3762577170158815, 0.3274953178464994, 0.39789047847859665, 0.6233838562591424, 0.32776628474165437, 0.7441955151980073, 0.3842010126694059, 0.8953880656202464, 0.40072524197466625, 0.9282784133265485, 0.3778259618529864, 0.10707567531256033, 0.2432024465933816, 0.1918075047468046, 0.21074846103161418, 0.6114509140708708, 0.18070681259807247, 0.7285183649306963, 0.18611302065452928, 0.856034922886001, 0.23211079697538228, 0.9392750788497093, 0.2984183580646005];
    const HEXBUBBLE_PHASE1_F: [f64; 60] = [0.116748109, 0.272993326, 0.192945302, 0.226265505, 0.551514566, 0.181655869, 0.723885059, 0.191847399, 0.856586039, 0.258085698, 0.929740846, 0.327946961, 0.150696188, 0.384897113, 0.338465124, 0.363662124, 0.633440495, 0.321295321, 0.757427216, 0.384772867, 0.89349699, 0.401873499, 0.926095903, 0.384857774, 0.178390518, 0.575907946, 0.257051378, 0.62570715, 0.472893536, 0.626221657, 0.681321084, 0.571693182, 0.778301239, 0.561323464, 0.889511347, 0.560974717, 0.225370064, 0.811362565, 0.35462296, 0.813762546, 0.551460505, 0.766298711, 0.746242344, 0.734050691, 0.869178116, 0.764148891, 0.94904387, 0.787265956, 0.205308795, 0.884347558, 0.284138709, 0.892428756, 0.413664848, 0.894623756, 0.698822379, 0.88889575, 0.832525074, 0.878492415, 0.919648528, 0.882006407];
    const HEXBUBBLE_PHASE2_F: [f64; 60] = [0.107075676, 0.243202418, 0.191807479, 0.210748538, 0.61145097, 0.180706799, 0.728518009, 0.186112985, 0.856034696, 0.232110724, 0.939275026, 0.298418403, 0.132573366, 0.376257688, 0.327495486, 0.397890449, 0.62338388, 0.32776621, 0.744195461, 0.38420099, 0.895388186, 0.400725365, 0.928278685, 0.377825975, 0.160029247, 0.58050245, 0.234491423, 0.630412936, 0.427009135, 0.616778493, 0.669269919, 0.569897771, 0.768018901, 0.557183504, 0.886705279, 0.556232631, 0.223726809, 0.815817535, 0.341000527, 0.793082297, 0.544013202, 0.756579936, 0.741972208, 0.729409635, 0.867057085, 0.743273616, 0.94905901, 0.736373305, 0.2041042, 0.892310858, 0.286168903, 0.892758548, 0.425051242, 0.900830388, 0.706229508, 0.897238195, 0.831443191, 0.880344808, 0.919335306, 0.886205375];
    const HEXBUBBLE_QE: [f64; 4] = [0.07667008556129436, 0.07211245801556332, 0.0766700804, 0.0721124113];
    const HEXBUBBLE_FLOAT_PARTS: Option<(usize, usize, usize, usize)> = Some((1, 31, 22, 10));

    /// `SOM_PAK`'s two `vsom` runs as phases: `rlen` steps each, `α` falling linearly from `alpha`
    /// to zero, the radius from `radius` to one — times `√2` for the Gaussian, whose `exp(−d²/2r²)`
    /// is eq. (8) with `σ = r√2`.
    fn sompak_phases(steps: [usize; 2], alpha: [f64; 2], radius: [f64; 2], kernel: Kernel) -> [Phase; 2] {
        let s = if kernel == Kernel::Gaussian { 2.0_f64.sqrt() } else { 1.0 };
        core::array::from_fn(|p| Phase {
            steps: steps[p],
            alpha: Schedule::Linear { start: alpha[p], end: 0.0, steps: steps[p] },
            width: Schedule::Linear { start: radius[p] * s, end: s, steps: steps[p] },
            kernel,
        })
    }

    /// The reference's data and initial codebook as `f64`: every value is `k/1024`, exact in `float`
    /// and in `double`, so both builds and this module start from the same bits.
    fn from_1024(v: &[u16]) -> Vec<f64> {
        v.iter().map(|&k| f64::from(k) / 1024.0).collect()
    }

    /// The two phases run from the reference's initial codebook, cyclically through its data: the
    /// codebook after each, and the quantisation error after each.
    fn sompak_run(grid: Grid, dim: usize, data: &[u16], code: &[u16], phases: &[Phase; 2]) -> (Vec<Vec<f64>>, Vec<f64>) {
        let x = from_1024(data);
        let mut som = Som::new(grid, dim, from_1024(code)).unwrap();
        let mut outs = Vec::new();
        let mut qe = Vec::new();
        for p in phases {
            som.train(&x, p, Order::Cyclic).unwrap();
            outs.push(som.weights().to_vec());
            qe.push(som.quantization_error(&x).unwrap());
        }
        (outs, qe)
    }

    fn max_diff(a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(a.len(), b.len());
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
    }

    /// The four reference cases: name, grid, dimension, data, initial codebook, phases, the
    /// `double` build's codebooks after each phase, the `float` build's, and the four quantisation
    /// errors (`double` after each phase, then `float`).
    #[allow(clippy::type_complexity)]
    fn sompak_cases() -> Vec<(&'static str, Grid, usize, &'static [u16], &'static [u16], [Phase; 2], [&'static [f64]; 2], [&'static [f64]; 2], [f64; 4])> {
        vec![
            ("chain", Grid::chain(8).unwrap(), 1, &CHAIN_DATA, &CHAIN_CODE,
             sompak_phases([200, 800], [0.5, 0.03125], [4.0, 1.0], Kernel::Bubble),
             [&CHAIN_PHASE1_D, &CHAIN_PHASE2_D], [&CHAIN_PHASE1_F, &CHAIN_PHASE2_F], CHAIN_QE),
            ("hexgauss", Grid::new(Lattice::Hex, 4, 3).unwrap(), 2, &HEXGAUSS_DATA, &HEXGAUSS_CODE,
             sompak_phases([300, 600], [0.5, 0.0625], [3.0, 1.0], Kernel::Gaussian),
             [&HEXGAUSS_PHASE1_D, &HEXGAUSS_PHASE2_D], [&HEXGAUSS_PHASE1_F, &HEXGAUSS_PHASE2_F], HEXGAUSS_QE),
            ("rect3", Grid::new(Lattice::Rect, 5, 4).unwrap(), 3, &RECT3_DATA, &RECT3_CODE,
             sompak_phases([500, 1500], [0.5, 0.03125], [5.0, 1.0], Kernel::Bubble),
             [&RECT3_PHASE1_D, &RECT3_PHASE2_D], [&RECT3_PHASE1_F, &RECT3_PHASE2_F], RECT3_QE),
            ("hexbubble", Grid::new(Lattice::Hex, 6, 5).unwrap(), 2, &HEXBUBBLE_DATA, &HEXBUBBLE_CODE,
             sompak_phases([1000, 3000], [0.5, 0.03125], [4.0, 1.0], Kernel::Bubble),
             [&HEXBUBBLE_PHASE1_D, &HEXBUBBLE_PHASE2_D], [&HEXBUBBLE_PHASE1_F, &HEXBUBBLE_PHASE2_F], HEXBUBBLE_QE),
        ]
    }

    /// Against `SOM_PAK` built in `double`, on the same data, initial codebook and presentation
    /// order: the three bubble cases agree BIT FOR BIT after both phases — codebooks and
    /// quantisation errors — because every line of the update is the same arithmetic in the same
    /// order. The Gaussian case agrees to `1.1e-16` (measured) and not exactly, because eq. (8)'s
    /// `σ² = (r√2)²` is `2r²` only to rounding; its quantisation errors agree exactly anyway.
    #[test]
    fn the_map_is_sompaks_to_the_bit() {
        for (name, grid, dim, data, code, phases, dbl, _, qe) in sompak_cases() {
            let (outs, q) = sompak_run(grid, dim, data, code, &phases);
            for p in 0..2 {
                if phases[p].kernel == Kernel::Bubble {
                    assert_eq!(outs[p], dbl[p], "{name}, phase {}", p + 1);
                } else {
                    let d = max_diff(&outs[p], dbl[p]);
                    assert!(d > 0.0 && d < 4e-16, "{name}, phase {}: {d:e}", p + 1);
                }
                assert_eq!(q[p], qe[p], "{name}, phase {}: quantisation error", p + 1);
            }
        }
    }

    /// Against `SOM_PAK` as written, in `float`: where the two builds take the same path the
    /// codebooks agree to float rounding — `4.5e-7` at most, measured, and the quantisation errors
    /// to `4.7e-8` — but in two of the four cases they do not take the same path. In `hexbubble`
    /// the `float` map is the `double` map turned upside down, row `r` for row `4 − r`, to `4.3e-7`;
    /// in `rect3` it is a different map altogether, `0.59` away. The next test shows why.
    #[test]
    fn the_float_build_agrees_to_float_rounding_or_takes_a_mirror_path() {
        let cases = sompak_cases();
        for (name, grid, dim, data, code, phases, _, flt, qe) in &cases[..2] {
            let (outs, q) = sompak_run(*grid, *dim, data, code, phases);
            for p in 0..2 {
                let d = max_diff(&outs[p], flt[p]);
                assert!(d < 1e-6, "{name}, phase {}: {d:e}", p + 1);
                assert!((q[p] - qe[p + 2]).abs() < 1e-7, "{name}, phase {}", p + 1);
            }
        }
        let (_, grid, dim, data, code, phases, _, flt, qe) = &cases[3];
        let (outs, q) = sompak_run(*grid, *dim, data, code, phases);
        for p in 0..2 {
            let flipped: Vec<f64> = (0..grid.len())
                .flat_map(|i| {
                    let (x, y) = grid.coords(i);
                    flt[p][((grid.height() - 1 - y) * grid.width() + x) * dim..][..*dim].to_vec()
                })
                .collect();
            let (direct, mirrored) = (max_diff(&outs[p], flt[p]), max_diff(&outs[p], &flipped));
            assert!(direct > 0.5 && mirrored < 1e-6, "hexbubble, phase {}: {direct:e}, mirrored {mirrored:e}", p + 1);
            assert!((q[p] - qe[p + 2]).abs() < 1e-7, "a mirror image quantises equally well");
        }
        let (_, grid, dim, data, code, phases, _, flt, _) = &cases[2];
        let (outs, _) = sompak_run(*grid, *dim, data, code, phases);
        assert!(max_diff(&outs[1], flt[1]) > 0.5, "rect3 parts from the float build");
    }

    /// Why the two builds part: at the step where their winners first differ — which the reference
    /// script finds by printing both builds' winner at every step — the codebook is a near-tie that
    /// a change of precision can flip. The wide opening neighbourhood of `rect3` and `hexbubble`
    /// has pulled their units together. In `rect3`, at step 32, 18 of the 20 units lie within
    /// `4.8e-10` of the nearest in squared distance (measured), units 14 and 4 `1.3e-12` apart,
    /// and the unit the `float` build took, unit 1, is `1.5e-10` behind the `double` build's
    /// unit 14. In `hexbubble`, at step 31, sample 31 is `7.7e-9` nearer to unit 22 = `(4, 3)` than
    /// to unit 10 = `(4, 1)`, the `float` build's choice — the same column, rows mirrored about the
    /// middle row of a map the neighbourhood has kept nearly symmetric — and the orientation of the
    /// whole map follows. Relative to the distance the two gaps are `1.4e-9` and `1.1e-7`: far
    /// below, and about, the `6e-8` that one rounding in `float` can move a value by. This module
    /// takes the `double` build's unit both times; the `chain` and `hexgauss` cases, whose builds
    /// never part, have no such step.
    #[test]
    fn near_ties_decide_where_the_two_builds_part() {
        assert_eq!((CHAIN_FLOAT_PARTS, HEXGAUSS_FLOAT_PARTS), (None, None));
        let cases = sompak_cases();
        let mut gaps = Vec::new();
        for (case, parts) in [(&cases[2], RECT3_FLOAT_PARTS), (&cases[3], HEXBUBBLE_FLOAT_PARTS)] {
            let (name, grid, dim, data, code, phases, _, _, _) = case;
            let (phase, step, dbl, flt) = parts.unwrap();
            assert_eq!(phase, 1, "{name}");
            let x = from_1024(data);
            let mut som = Som::new(*grid, *dim, from_1024(code)).unwrap();
            som.train(&x, &Phase { steps: step, ..phases[0] }, Order::Cyclic).unwrap();
            let k = step % (x.len() / dim);
            let sample = &x[k * dim..(k + 1) * dim];
            assert_eq!(som.winner(sample).unwrap(), dbl, "{name}: the double build's winner");
            let d: Vec<f64> = (0..grid.len()).map(|i| som.unit(i).iter().zip(sample).map(|(m, v)| (m - v) * (m - v)).sum()).collect();
            let near = d[dbl];
            assert!(d.iter().all(|&e| e >= near), "{name}");
            let within = |eps: f64| d.iter().filter(|&&e| e - near <= eps).count();
            gaps.push(((d[flt] - near) / near, d[flt] - near, within(4.8e-10), within(1.4e-12)));
        }
        let [(r3, g3, n3, t3), (r5, g5, n5, t5)] = gaps[..] else { unreachable!() };
        assert!(g3 > 1.5e-10 && g3 < 1.55e-10 && (r3 - 1.4e-9).abs() < 5e-11 && (n3, t3) == (18, 2), "rect3: {g3:e} {r3:e} {n3} {t3}");
        assert!(g5 > 7.65e-9 && g5 < 7.7e-9 && (r5 - 1.1e-7).abs() < 5e-9 && (n5, t5) == (1, 1), "hexbubble: {g5:e} {r5:e} {n5} {t5}");
        let (_, grid, _, _, _, _, _, _, _) = &cases[3];
        assert_eq!((grid.coords(22), grid.coords(10)), ((4, 3), (4, 1)));
    }

    // ----------------------------------------------------------------------------------------------
    // Closed forms: the lattice, the kernels, the schedules, one step of each rule.
    // ----------------------------------------------------------------------------------------------

    /// Unit `i` at column `i % width`, row `i / width`; rectangular distance `Δx² + Δy²`; hexagonal
    /// neighbours at squared distance exactly one — six of them for an interior unit of either row
    /// parity, four on a rectangular array — and every distance symmetric.
    #[test]
    fn the_lattice_is_sompaks_and_its_neighbours_are_exact() {
        let r = Grid::new(Lattice::Rect, 5, 4).unwrap();
        assert_eq!((r.lattice(), r.width(), r.height(), r.len(), r.is_empty()), (Lattice::Rect, 5, 4, 20, false));
        assert_eq!(r.coords(13), (3, 2));
        assert_eq!(r.distance2(0, 19), 16.0 + 9.0);
        assert_eq!(r.distance(0, 19), 5.0);
        let h = Grid::new(Lattice::Hex, 5, 5).unwrap();
        assert_eq!(h.lattice(), Lattice::Hex);
        let at = |x: usize, y: usize| y * 5 + x;
        // Odd rows are shifted half a unit to the right.
        assert_eq!(h.distance2(at(0, 0), at(0, 1)), 1.0);
        assert_eq!(h.distance2(at(0, 1), at(0, 2)), 1.0);
        assert_eq!(h.distance2(at(1, 0), at(0, 1)), 0.25 + 0.75);
        assert_eq!(h.distance2(at(0, 0), at(1, 1)), 2.25 + 0.75);
        assert_eq!(h.distance2(at(1, 1), at(0, 2)), 2.25 + 0.75);
        assert_eq!(h.distance2(at(0, 0), at(0, 2)), 3.0);
        assert_eq!(h.distance2(at(4, 0), at(0, 3)), 3.5 * 3.5 + 0.75 * 9.0);
        let even = [at(1, 2), at(3, 2), at(1, 1), at(2, 1), at(1, 3), at(2, 3)];
        let odd = [at(1, 1), at(3, 1), at(2, 0), at(3, 0), at(2, 2), at(3, 2)];
        for (c, want) in [(at(2, 2), even), (at(2, 1), odd)] {
            let mut got: Vec<usize> = (0..25).filter(|&i| h.adjacent(c, i)).collect();
            let mut want = want.to_vec();
            got.sort_unstable();
            want.sort_unstable();
            assert_eq!(got, want, "the six neighbours of unit {c}");
        }
        assert_eq!((0..20).filter(|&i| r.adjacent(7, i)).count(), 4, "an interior unit of the rectangle");
        assert_eq!((0..20).filter(|&i| r.adjacent(0, i)).count(), 2, "a corner of the rectangle");
        assert!(!r.adjacent(0, 6), "a diagonal is not adjacent");
        for g in [r, h] {
            for a in 0..g.len() {
                assert!(!g.adjacent(a, a));
                for b in 0..g.len() {
                    assert_eq!(g.distance2(a, b), g.distance2(b, a));
                }
            }
        }
        let c = Grid::chain(7).unwrap();
        assert_eq!((c.width(), c.height(), c.distance(1, 6)), (7, 1, 5.0));
        let s = Grid::new(Lattice::Rect, 8, 8).unwrap();
        let ring = |k: usize| (0..64).filter(|&i| s.chebyshev(27, i) == k).count();
        assert_eq!((ring(0), ring(1), ring(2)), (1, 8, 16), "the eight nearest and the 16 around them");
        assert_eq!(s.chebyshev(0, 63), 7);
    }

    /// Fig. 2 of the 1990 paper draws `N_c` on the hexagonal lattice as hexagons: the units within
    /// `r` steps of the winner, `1 + 3r(r + 1)` of them. `SOM_PAK`'s bubble, and this module's, is a
    /// disc of Euclidean radius `r` instead. Counted on a 41 × 41 lattice, the two hold the same
    /// units for every whole radius up to six and part at seven, where the disc takes 18 units eight
    /// steps out — ring `k` comes as close as `k√3/2`, and `8√3/2 = 6.93 < 7` — and at eight, 24
    /// units nine steps out. Between whole radii, which a linearly shrinking radius passes through,
    /// the disc is the larger: at 1.8 it holds 13 units to the hexagon's 7, the six at `√3`.
    #[test]
    fn the_bubble_is_fig_2s_hexagon_up_to_radius_six() {
        let g = Grid::new(Lattice::Hex, 41, 41).unwrap();
        let c = 20 * 41 + 20;
        let mut steps = vec![usize::MAX; g.len()];
        steps[c] = 0;
        let mut frontier = vec![c];
        for k in 1..=10 {
            let mut next = Vec::new();
            for &u in &frontier {
                for v in 0..g.len() {
                    if steps[v] == usize::MAX && g.adjacent(u, v) {
                        steps[v] = k;
                        next.push(v);
                    }
                }
            }
            frontier = next;
        }
        for r in 1..=8usize {
            let hexagon: Vec<usize> = (0..g.len()).filter(|&i| steps[i] <= r).collect();
            let disc: Vec<usize> = (0..g.len()).filter(|&i| Kernel::Bubble.h(g.distance(c, i), 1.0, r as f64) == 1.0).collect();
            assert_eq!(hexagon.len(), 1 + 3 * r * (r + 1));
            if r <= 6 {
                assert_eq!(disc, hexagon, "radius {r}");
            } else {
                assert!(hexagon.iter().all(|i| disc.contains(i)));
                assert_eq!(disc.len() - hexagon.len(), if r == 7 { 18 } else { 24 }, "radius {r}");
            }
        }
        let disc = (0..g.len()).filter(|&i| g.distance(c, i) <= 1.8).count();
        assert_eq!((disc, (0..g.len()).filter(|&i| steps[i] <= 1).count()), (13, 7));
    }

    /// Eqs. (6)–(8): the bubble is `α` inside `N_c`, its edge included, and nothing outside; the
    /// Gaussian is `α exp(−d²/σ²)`.
    #[test]
    fn the_kernels_are_eqs_6_to_8() {
        assert_eq!(Kernel::Bubble.h(2.0, 0.25, 2.0), 0.25);
        assert_eq!(Kernel::Bubble.h(2.0 + 1e-12, 0.25, 2.0), 0.0);
        assert_eq!(Kernel::Bubble.h(0.0, 0.25, 0.0), 0.25, "N_c = {{c}}");
        assert_eq!(Kernel::Gaussian.h(0.0, 0.5, 3.0), 0.5);
        assert_eq!(Kernel::Gaussian.h(3.0, 0.5, 3.0), 0.5 * (-1.0_f64).exp());
        assert_eq!(Kernel::Gaussian.h(2.0, 0.5, 4.0), 0.5 * (-0.25_f64).exp());
        assert_eq!(Kernel::Gaussian.h(1.5, 1.0, 0.5), (-9.0_f64).exp());
    }

    /// The schedules against the laws the papers print. The linear fall is `0.9(1 − t/1000)` to
    /// within one rounding (`1.1e-16`, measured) at every step, is `start` at `t = 0` and `end` from
    /// `t = T` on; the exponential is `4(0.5/4)^{t/2000}`, `√2` halfway; the inverse is `c/t` with
    /// the paper's count starting at one.
    #[test]
    fn the_schedules_are_the_printed_laws() {
        let a = Schedule::Linear { start: 0.9, end: 0.0, steps: 1000 };
        let mut worst = 0.0_f64;
        for t in 0..=1000 {
            worst = worst.max((a.at(t).unwrap() - 0.9 * (1.0 - t as f64 / 1000.0)).abs());
        }
        assert!(worst < 2.3e-16, "{worst:e}");
        assert_eq!((a.at(0).unwrap(), a.at(1000).unwrap(), a.at(5000).unwrap()), (0.9, 0.0, 0.0));
        let r = Schedule::Linear { start: 6.0, end: 1.0, steps: 1000 };
        assert_eq!((r.at(0).unwrap(), r.at(500).unwrap(), r.at(1000).unwrap(), r.at(1001).unwrap()), (6.0, 3.5, 1.0, 1.0));
        assert_eq!(Schedule::Linear { start: 1.0, end: 3.0, steps: 4 }.at(1).unwrap(), 1.5, "a rise is allowed");
        let s = Schedule::Exponential { start: 4.0, end: 0.5, steps: 2000 };
        assert_eq!(s.at(0).unwrap(), 4.0);
        assert!((s.at(1000).unwrap() - 2.0_f64.sqrt()).abs() < 1e-15);
        assert!((s.at(2000).unwrap() - 0.5).abs() < 1e-15);
        assert_eq!(s.at(2000).unwrap(), s.at(9000).unwrap(), "held at the end");
        assert!((s.at(500).unwrap() - 4.0 * 0.125_f64.powf(0.25)).abs() < 1e-15);
        let i = Schedule::InverseT { c: 100.0 };
        assert_eq!((i.at(0).unwrap(), i.at(1).unwrap(), i.at(99).unwrap()), (100.0, 50.0, 1.0));
        assert_eq!(Schedule::Constant(0.3).at(12345).unwrap(), 0.3);
    }

    /// The 1990 paper's phases with its printed numbers.
    #[test]
    fn the_phases_carry_the_1990_papers_numbers() {
        let o = Phase::ordering_1990(5.0);
        assert_eq!((o.steps, o.kernel), (1000, Kernel::Bubble));
        assert_eq!((o.alpha.at(0).unwrap(), o.alpha.at(250).unwrap()), (0.9, 0.675));
        assert_eq!((o.width.at(0).unwrap(), o.width.at(500).unwrap(), o.width.at(1000).unwrap()), (5.0, 3.0, 1.0));
        let c = Phase::convergence_1990(Grid::new(Lattice::Hex, 10, 7).unwrap());
        assert_eq!((c.steps, c.kernel, c.width), (35_000, Kernel::Bubble, Schedule::Constant(1.0)));
        assert_eq!((c.alpha.at(0).unwrap(), c.alpha.at(17_500).unwrap(), c.alpha.at(35_000).unwrap()), (0.01, 0.005, 0.0));
        let [p1, p2] = Phase::table1_1990();
        assert_eq!((p1.steps, p2.steps, p1.kernel, p2.kernel), (1000, 10_000, Kernel::Bubble, Kernel::Bubble));
        assert_eq!((p1.alpha.at(0).unwrap(), p1.alpha.at(1000).unwrap(), p2.alpha.at(0).unwrap(), p2.alpha.at(10_000).unwrap()), (0.5, 0.04, 0.04, 0.0));
        assert!((p1.alpha.at(500).unwrap() - 0.27).abs() < 1e-16);
        assert_eq!((p1.width.at(0).unwrap(), p1.width.at(1000).unwrap(), p2.width.at(7).unwrap()), (6.0, 1.0, 1.0));
    }

    /// One step of eq. (7) with the bubble: the winner is the nearest unit (eq. 2'), every unit
    /// within the radius moves `α` of the way to `x`, and every unit outside keeps its bits. With
    /// `α = 1` the winner lands on the sample. A tie goes to the lower index.
    #[test]
    fn one_bubble_step_is_eq_7() {
        let w = vec![0.0, 0.25, 0.5, 0.75, 1.0, 1.25];
        let mut som = Som::new(Grid::chain(6).unwrap(), 1, w.clone()).unwrap();
        assert_eq!((som.dim(), som.grid(), som.unit(3)), (1, Grid::chain(6).unwrap(), &[0.75][..]));
        assert_eq!(som.winner(&[0.625]).unwrap(), 2, "equidistant from units 2 and 3");
        assert_eq!(som.winner(&[0.3]).unwrap(), 1);
        assert_eq!(som.winner(&[-7.0]).unwrap(), 0);
        assert_eq!(som.winner(&[7.0]).unwrap(), 5);
        let c = som.step(&[0.8125], Kernel::Bubble, 0.5, 1.0).unwrap();
        assert_eq!(c, 3);
        let want = [0.0, 0.25, 0.5 + 0.5 * 0.3125, 0.75 + 0.5 * 0.0625, 1.0 - 0.5 * 0.1875, 1.25];
        assert_eq!(som.weights(), &want);
        let c = som.step(&[0.0625], Kernel::Bubble, 1.0, 0.0).unwrap();
        assert_eq!(c, 0);
        assert_eq!(som.weights()[0], 0.0625);
        assert_eq!(&som.weights()[1..], &want[1..], "radius zero moves the winner only");
        let mut two = Som::new(Grid::new(Lattice::Rect, 2, 2).unwrap(), 2, vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
        two.step(&[0.25, 0.25], Kernel::Bubble, 0.5, 1.0).unwrap();
        assert_eq!(two.weights(), &[0.125, 0.125, 0.625, 0.125, 0.125, 0.625, 1.0, 1.0], "the diagonal unit is outside");
    }

    /// One step with the Gaussian: every unit moves by `h_ci = α exp(−‖r_c − r_i‖²/σ²)` of the way,
    /// on a hexagonal array, against the closed form evaluated here.
    #[test]
    fn one_gaussian_step_is_eq_8() {
        let g = Grid::new(Lattice::Hex, 3, 3).unwrap();
        let w: Vec<f64> = (0..18).map(|k| k as f64 / 16.0).collect();
        let mut som = Som::new(g, 2, w.clone()).unwrap();
        let x = [0.5, 0.5625];
        let c = som.step(&x, Kernel::Gaussian, 0.75, 1.5).unwrap();
        assert_eq!(c, 4);
        for i in 0..9 {
            let h = 0.75 * (-g.distance2(4, i) / 2.25).exp();
            for j in 0..2 {
                let want = w[2 * i + j] + h * (x[j] - w[2 * i + j]);
                assert!((som.weights()[2 * i + j] - want).abs() < 2e-16, "unit {i}");
            }
        }
    }

    /// Eq. (3): the updated vector is `(m + αx)/‖m + αx‖`, of unit length; the winner is the largest
    /// inner product (eq. 2), the lower index on a tie; units outside the reach keep their bits.
    #[test]
    fn one_1982_step_is_eq_3() {
        let g = Grid::new(Lattice::Rect, 4, 4).unwrap();
        let w: Vec<f64> = (0..32).map(|k| (1 + (k * 7) % 11) as f64 / 8.0).collect();
        let mut map = Kohonen1982::new(g, 2, w.clone()).unwrap();
        assert_eq!((map.grid(), map.unit(3)), (g, &w[6..8]));
        let x = [0.6, 0.8];
        let k = map.winner(&x).unwrap();
        let eta: Vec<f64> = (0..16).map(|i| w[2 * i] * x[0] + w[2 * i + 1] * x[1]).collect();
        assert_eq!(map.response(5, &x), eta[5]);
        let best = eta.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        assert_eq!(k, eta.iter().position(|&e| e == best).unwrap());
        assert_eq!(map.step(&x, 0.5, Reach::Nearest).unwrap(), k);
        let (kx, ky) = g.coords(k);
        for i in 0..16 {
            let (ix, iy) = g.coords(i);
            let u = &map.weights()[2 * i..2 * i + 2];
            if ix.abs_diff(kx) <= 1 && iy.abs_diff(ky) <= 1 {
                let v = [w[2 * i] + 0.5 * x[0], w[2 * i + 1] + 0.5 * x[1]];
                let n = (v[0] * v[0] + v[1] * v[1]).sqrt();
                assert!((u[0] - v[0] / n).abs() < 1e-16 && (u[1] - v[1] / n).abs() < 1e-16, "unit {i}");
                assert!((u[0] * u[0] + u[1] * u[1] - 1.0).abs() < 1e-15);
            } else {
                assert_eq!(u, &w[2 * i..2 * i + 2], "unit {i} is outside the reach");
            }
        }
        // A tie in eq. (2) goes to the lower index.
        let tie = Kohonen1982::new(Grid::chain(3).unwrap(), 2, vec![0.0, 1.0, 1.0, 0.0, 1.0, 0.0]).unwrap();
        assert_eq!(tie.winner(&[1.0, 0.0]).unwrap(), 1);
        assert_eq!(tie.winner(&[0.0, 1.0]).unwrap(), 0);
        assert_eq!(tie.winner(&[-1.0, -1.0]).unwrap(), 0, "every response negative: the largest still wins");
        let neg = Kohonen1982::new(Grid::chain(3).unwrap(), 1, vec![-3.0, -1.0, -2.0]).unwrap();
        assert_eq!(neg.winner(&[1.0]).unwrap(), 1);
    }

    /// Simulation 3's rings: eq. (3) at `α` for the winner and its eight neighbours, at `α/4` for the
    /// sixteen around them, nothing beyond. At a corner the rings are cut by the edge — "the number of
    /// neighbours was different" — so a corner winner updates four units at `α` and five at `α/4`.
    #[test]
    fn the_rings_of_simulation_3() {
        let g = Grid::new(Lattice::Rect, 6, 6).unwrap();
        let w: Vec<f64> = (0..36).flat_map(|k| [1.0, k as f64 / 36.0]).collect();
        let x = [0.0, 1.0];
        let mut map = Kohonen1982::new(g, 2, w.clone()).unwrap();
        assert_eq!(map.step(&x, 2.0, Reach::Rings).unwrap(), 35);
        let mut counts = [0usize; 3];
        for i in 0..36 {
            let u = &map.weights()[2 * i..2 * i + 2];
            let a = match g.chebyshev(35, i) {
                0 | 1 => 2.0,
                2 => 0.5,
                _ => {
                    assert_eq!(u, &w[2 * i..2 * i + 2]);
                    counts[2] += 1;
                    continue;
                }
            };
            counts[usize::from(a < 1.0)] += 1;
            let v = [w[2 * i], w[2 * i + 1] + a];
            let n = (v[0] * v[0] + v[1] * v[1]).sqrt();
            assert!((u[0] - v[0] / n).abs() < 1e-16 && (u[1] - v[1] / n).abs() < 1e-16, "unit {i}");
        }
        assert_eq!(counts, [4, 5, 27]);
        let mut near = Kohonen1982::new(g, 2, w.clone()).unwrap();
        near.step(&x, 2.0, Reach::Nearest).unwrap();
        assert_eq!((0..36).filter(|&i| near.unit(i) != &w[2 * i..2 * i + 2]).count(), 4);
        let mut disc = Kohonen1982::new(g, 2, w).unwrap();
        disc.step(&x, 2.0, Reach::Radius(2.0)).unwrap();
        assert_eq!((0..36).filter(|&i| g.distance(35, i) <= 2.0).count(), 6);
        assert_eq!((0..36).filter(|&i| disc.unit(i)[0] != 1.0).count(), 6, "eq. (10)'s N_c is the disc");
    }

    /// Simulation 4's coherent inputs: each unit reads its own input, and the winner is the unit whose
    /// response to ITS input is largest — not the unit that would win a shared input.
    #[test]
    fn coherent_inputs_are_read_unit_by_unit() {
        let g = Grid::chain(3).unwrap();
        let w = vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0];
        let inputs = [0.25, 0.0, 0.0, 0.5, 0.375, 0.0];
        let mut map = Kohonen1982::new(g, 2, w).unwrap();
        assert_eq!(map.winner_coherent(&inputs).unwrap(), 1);
        assert_eq!(map.winner(&[0.25, 0.0]).unwrap(), 0);
        assert_eq!(map.step_coherent(&inputs, 1.0, Reach::Radius(0.0)).unwrap(), 1);
        assert_eq!(map.weights(), &[1.0, 0.0, 0.0, 1.0, 1.0, 0.0], "it already pointed at its input");
        assert_eq!(map.step_coherent(&inputs, 1.0, Reach::Nearest).unwrap(), 1);
        assert_eq!(map.weights(), &[1.0, 0.0, 0.0, 1.0, 1.0, 0.0], "units 0 and 2 read inputs along their weights");
        // Unit 2 wins on its own input (0, 2); its neighbour, unit 1, moves towards ITS input (0, 1):
        // along the winner's input it would have gone to (1, 1)/√2.
        let mut third = Kohonen1982::new(g, 2, vec![0.0, 1.0, 1.0, 0.0, 0.0, 1.0]).unwrap();
        assert_eq!(third.step_coherent(&[1.0, 0.0, 0.0, 1.0, 0.0, 2.0], 0.5, Reach::Nearest).unwrap(), 2);
        assert_eq!((third.unit(0), third.unit(2)), (&[0.0, 1.0][..], &[0.0, 1.0][..]));
        let n = 1.25_f64.sqrt();
        assert_eq!(third.unit(1), &[1.0 / n, 0.5 / n]);
    }

    // ----------------------------------------------------------------------------------------------
    // Measures, orders, refusals.
    // ----------------------------------------------------------------------------------------------

    /// Quantisation and topographic error on maps small enough to work by hand. A straight chain at
    /// `0, 1, 2`: every sample's two best units are neighbours. The same weights folded, `0, 2, 1`:
    /// a sample at `0.375` has unit 0 first and unit 2 — two steps away on the chain — second.
    #[test]
    fn the_two_errors_by_hand() {
        let data = [0.25, 1.5, 1.875, 0.375];
        let straight = Som::new(Grid::chain(3).unwrap(), 1, vec![0.0, 1.0, 2.0]).unwrap();
        assert_eq!(straight.quantization_error(&data).unwrap(), (0.25 + 0.5 + 0.125 + 0.375) / 4.0);
        assert_eq!(straight.topographic_error(&data).unwrap(), 0.0);
        let folded = Som::new(Grid::chain(3).unwrap(), 1, vec![0.0, 2.0, 1.0]).unwrap();
        assert_eq!(folded.quantization_error(&data).unwrap(), (0.25 + 0.5 + 0.125 + 0.375) / 4.0, "the same codebook");
        assert_eq!(folded.topographic_error(&data).unwrap(), 0.5, "0.25 and 0.375 have units 0 and 2 first and second");
        let flat = Som::new(Grid::new(Lattice::Rect, 2, 2).unwrap(), 2, vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
        let d = [0.125, 0.0, 0.5, 0.5, 0.875, 0.875];
        // (0.5, 0.5) is equidistant from all four: units 0 and 1, adjacent. (0.875, 0.875) has
        // unit 3 first and unit 1 second, adjacent; (0.125, 0) has 0 then 1.
        assert_eq!(flat.topographic_error(&d).unwrap(), 0.0);
        let diag = Som::new(Grid::new(Lattice::Rect, 2, 2).unwrap(), 2, vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.25, 0.25]).unwrap();
        // (0.125, 0.125) is as near unit 3 as unit 0: unit 0 first by index, unit 3 second — the
        // diagonal. (1, 0.25) has unit 1 first and unit 3, its neighbour, second.
        let d = [0.125, 0.125, 1.0, 0.25];
        assert_eq!(diag.topographic_error(&d).unwrap(), 0.5, "units 0 and 3 are diagonal, not adjacent");
        let q = diag.quantization_error(&d).unwrap();
        assert!((q - (0.125 * 2.0_f64.sqrt() + 0.25) / 2.0).abs() < 1e-16);
    }

    /// The 1982 paper's ordered mapping read off a chain of scalars: strictly monotone either way.
    #[test]
    fn an_ordered_chain_is_strictly_monotone() {
        let chain = |w: Vec<f64>| Som::new(Grid::chain(w.len()).unwrap(), 1, w).unwrap().is_ordered_chain().unwrap();
        assert!(chain(vec![0.0, 0.25, 0.5, 2.0]));
        assert!(chain(vec![3.0, 0.25, -0.5, -2.0]));
        assert!(!chain(vec![0.0, 0.5, 0.25, 2.0]));
        assert!(!chain(vec![3.0, 0.25, 0.5, -2.0]));
        assert!(!chain(vec![0.0, 0.5, 0.5, 2.0]), "a tie is not an order");
        assert!(!chain(vec![1.0, 1.0]));
        assert!(chain(vec![1.0]));
    }

    /// A map drawn from a seed is the same map every time, bit for bit, with every weight in
    /// `[low, high)`; and a phase in random order draws every sample, each about equally often.
    #[test]
    fn random_draws_are_seeded_and_uniform() {
        let g = Grid::new(Lattice::Hex, 4, 3).unwrap();
        let a = Som::random(g, 3, -2.0, 0.5, &mut Rng::new(9)).unwrap();
        let b = Som::random(g, 3, -2.0, 0.5, &mut Rng::new(9)).unwrap();
        assert_eq!(a, b);
        assert!(a.weights().iter().all(|&w| (-2.0..0.5).contains(&w)));
        assert!(a.weights().iter().any(|&w| w < -1.75) && a.weights().iter().any(|&w| w > 0.25), "the whole interval is used");
        let data = [0.0, 1.0, 2.0, 3.0, 4.0];
        let mut rng = Rng::new(1990);
        let mut counts = [0usize; 5];
        let one = Phase { steps: 1, alpha: Schedule::Constant(1.0), width: Schedule::Constant(0.0), kernel: Kernel::Bubble };
        let mut som = Som::new(Grid::chain(1).unwrap(), 1, vec![0.5]).unwrap();
        for _ in 0..50_000 {
            som.train(&data, &one, Order::Random(&mut rng)).unwrap();
            counts[som.weights()[0] as usize] += 1;
        }
        // Each count is binomial(50 000, 1/5): mean 10 000, standard deviation 89.
        assert!(counts.iter().all(|&c| c.abs_diff(10_000) < 400), "{counts:?}");
        let run = |seed| {
            let mut rng = Rng::new(seed);
            let mut s = Som::random(g, 2, 0.0, 1.0, &mut rng).unwrap();
            let d: Vec<f64> = (0..40).map(|_| rng.next_f64()).collect();
            s.train(&d, &Phase::ordering_1990(3.0), Order::Random(&mut rng)).unwrap();
            s
        };
        assert_eq!(run(4), run(4));
        assert_ne!(run(4), run(5));
    }

    /// A phase is checked whole before its first step: a gain schedule that leaves `[0, 1]` at step
    /// 6 is refused with the map untouched.
    #[test]
    fn a_refused_phase_leaves_the_map_as_it_was() {
        let mut som = Som::new(Grid::chain(3).unwrap(), 1, vec![0.0, 0.5, 1.0]).unwrap();
        let bad = Phase { steps: 10, alpha: Schedule::Linear { start: 0.5, end: -0.5, steps: 10 }, width: Schedule::Constant(1.0), kernel: Kernel::Bubble };
        assert_eq!(bad.alpha.at(5).unwrap(), 0.0);
        assert_eq!(som.train(&[0.25, 0.75], &bad, Order::Cyclic).unwrap_err().to_string(), "alpha = -0.09999999999999998 is outside [0, 1]");
        assert_eq!(som.weights(), &[0.0, 0.5, 1.0]);
        let wide = Phase { steps: 10, alpha: Schedule::Constant(0.5), width: Schedule::Linear { start: 1.0, end: -1.0, steps: 8 }, kernel: Kernel::Bubble };
        assert_eq!(som.train(&[0.25, 0.75], &wide, Order::Cyclic).unwrap_err().to_string(), "radius = -0.25 is outside [0, inf]");
        let narrow = Phase { steps: 3, alpha: Schedule::Constant(0.5), width: Schedule::Linear { start: 1.0, end: 0.0, steps: 2 }, kernel: Kernel::Gaussian };
        assert_eq!(som.train(&[0.25, 0.75], &narrow, Order::Cyclic).unwrap_err().to_string(), "sigma = 0 must be finite and positive");
        assert_eq!(som.weights(), &[0.0, 0.5, 1.0]);
        let ok = Phase { steps: 2, ..narrow };
        som.train(&[0.25, 0.75], &ok, Order::Cyclic).unwrap();
        assert_ne!(som.weights(), &[0.0, 0.5, 1.0]);
    }

    /// Every refusal, by the message it renders.
    #[test]
    fn every_refusal_names_what_it_refused() {
        let msg = |e: SomError| e.to_string();
        assert_eq!(msg(Grid::new(Lattice::Rect, 0, 3).unwrap_err()), "grid width: 0 supplied, at least 1 needed");
        assert_eq!(msg(Grid::new(Lattice::Hex, 3, 0).unwrap_err()), "grid height: 0 supplied, at least 1 needed");
        assert_eq!(msg(Grid::chain(0).unwrap_err()), "grid width: 0 supplied, at least 1 needed");
        let g = Grid::chain(3).unwrap();
        assert_eq!(msg(Som::new(g, 0, vec![]).unwrap_err()), "dimension: 0 supplied, at least 1 needed");
        assert_eq!(msg(Som::new(g, 2, vec![0.0; 5]).unwrap_err()), "weights has length 5; it must be 6");
        assert_eq!(msg(Som::new(g, 1, vec![0.0, f64::NAN, 0.0]).unwrap_err()), "weight = NaN is not finite");
        let mut rng = Rng::new(1);
        assert_eq!(msg(Som::random(g, 1, f64::NEG_INFINITY, 1.0, &mut rng).unwrap_err()), "low = -inf is not finite");
        assert_eq!(msg(Som::random(g, 1, 0.0, f64::NAN, &mut rng).unwrap_err()), "high = NaN is not finite");
        assert_eq!(msg(Som::random(g, 1, 1.0, 1.0, &mut rng).unwrap_err()), "[1, 1) holds no value to draw");
        assert_eq!(msg(Som::random(g, 1, 2.0, 1.0, &mut rng).unwrap_err()), "[2, 1) holds no value to draw");
        assert_eq!(msg(Som::random(g, 0, 0.0, 1.0, &mut rng).unwrap_err()), "dimension: 0 supplied, at least 1 needed");
        let mut som = Som::new(g, 1, vec![0.0, 0.5, 1.0]).unwrap();
        assert_eq!(msg(som.winner(&[0.0, 1.0]).unwrap_err()), "x has length 2; it must be 1");
        assert_eq!(msg(som.winner(&[f64::INFINITY]).unwrap_err()), "x = inf is not finite");
        assert_eq!(msg(som.step(&[0.5], Kernel::Bubble, 1.5, 1.0).unwrap_err()), "alpha = 1.5 is outside [0, 1]");
        assert_eq!(msg(som.step(&[0.5], Kernel::Bubble, -0.25, 1.0).unwrap_err()), "alpha = -0.25 is outside [0, 1]");
        assert_eq!(msg(som.step(&[0.5], Kernel::Bubble, f64::NAN, 1.0).unwrap_err()), "alpha = NaN is not finite");
        assert_eq!(msg(som.step(&[0.5], Kernel::Bubble, 0.5, -1.0).unwrap_err()), "radius = -1 is outside [0, inf]");
        assert_eq!(msg(som.step(&[0.5], Kernel::Bubble, 0.5, f64::NAN).unwrap_err()), "radius = NaN is not finite");
        assert_eq!(msg(som.step(&[0.5], Kernel::Gaussian, 0.5, 0.0).unwrap_err()), "sigma = 0 must be finite and positive");
        assert_eq!(msg(som.step(&[0.5], Kernel::Gaussian, 0.5, f64::INFINITY).unwrap_err()), "sigma = inf must be finite and positive");
        assert_eq!(msg(som.step(&[0.5, 0.5], Kernel::Bubble, 0.5, 1.0).unwrap_err()), "x has length 2; it must be 1");
        assert_eq!(msg(som.step(&[f64::NAN], Kernel::Bubble, 0.5, 1.0).unwrap_err()), "x = NaN is not finite");
        assert_eq!(som.weights(), &[0.0, 0.5, 1.0], "no refused step moved a unit");
        let two = Som::new(Grid::chain(2).unwrap(), 2, vec![0.0; 4]).unwrap();
        let p = Phase::ordering_1990(1.0);
        assert_eq!(msg(two.clone().train(&[0.0; 3], &p, Order::Cyclic).unwrap_err()), "data (a multiple of the dimension) has length 3; it must be 2");
        assert_eq!(msg(two.clone().train(&[], &p, Order::Cyclic).unwrap_err()), "data samples: 0 supplied, at least 1 needed");
        assert_eq!(msg(two.clone().train(&[0.0, f64::NAN], &p, Order::Cyclic).unwrap_err()), "data = NaN is not finite");
        assert_eq!(msg(two.quantization_error(&[0.0]).unwrap_err()), "data (a multiple of the dimension) has length 1; it must be 2");
        assert_eq!(msg(two.topographic_error(&[]).unwrap_err()), "data samples: 0 supplied, at least 1 needed");
        let lone = Som::new(Grid::chain(1).unwrap(), 1, vec![0.0]).unwrap();
        assert_eq!(msg(lone.topographic_error(&[0.0]).unwrap_err()), "units for a second-best match: 1 supplied, at least 2 needed");
        assert_eq!(msg(two.is_ordered_chain().unwrap_err()), "weight vector of a scalar chain has length 2; it must be 1");
        let sheet = Som::new(Grid::new(Lattice::Rect, 2, 2).unwrap(), 1, vec![0.0; 4]).unwrap();
        assert_eq!(msg(sheet.is_ordered_chain().unwrap_err()), "height of a chain has length 2; it must be 1");
        // Schedules.
        assert_eq!(msg(Schedule::Constant(f64::NAN).at(0).unwrap_err()), "schedule value = NaN is not finite");
        assert_eq!(msg(Schedule::Linear { start: f64::INFINITY, end: 0.0, steps: 5 }.at(0).unwrap_err()), "schedule start = inf is not finite");
        assert_eq!(msg(Schedule::Linear { start: 1.0, end: f64::NAN, steps: 5 }.at(0).unwrap_err()), "schedule end = NaN is not finite");
        assert_eq!(msg(Schedule::Linear { start: 1.0, end: 0.0, steps: 0 }.at(0).unwrap_err()), "schedule steps: 0 supplied, at least 1 needed");
        assert_eq!(msg(Schedule::Exponential { start: 0.0, end: 0.5, steps: 5 }.at(0).unwrap_err()), "schedule start = 0 must be finite and positive");
        assert_eq!(msg(Schedule::Exponential { start: 4.0, end: -0.5, steps: 5 }.at(0).unwrap_err()), "schedule end = -0.5 must be finite and positive");
        assert_eq!(msg(Schedule::Exponential { start: 4.0, end: 0.5, steps: 0 }.at(0).unwrap_err()), "schedule steps: 0 supplied, at least 1 needed");
        assert_eq!(msg(Schedule::InverseT { c: 0.0 }.at(0).unwrap_err()), "schedule constant = 0 must be finite and positive");
        assert_eq!(msg(Schedule::InverseT { c: f64::NAN }.check().unwrap_err()), "schedule constant = NaN must be finite and positive");
        let bad_phase = Phase { steps: 4, alpha: Schedule::Constant(0.5), width: Schedule::Linear { start: 1.0, end: 0.0, steps: 0 }, kernel: Kernel::Bubble };
        assert_eq!(msg(Som::new(g, 1, vec![0.0; 3]).unwrap().train(&[0.0], &bad_phase, Order::Cyclic).unwrap_err()), "schedule steps: 0 supplied, at least 1 needed");
        let bad_alpha = Phase { alpha: Schedule::Exponential { start: 1.0, end: 0.0, steps: 3 }, ..Phase::ordering_1990(1.0) };
        assert_eq!(msg(Som::new(g, 1, vec![0.0; 3]).unwrap().train(&[0.0], &bad_alpha, Order::Cyclic).unwrap_err()), "schedule end = 0 must be finite and positive");
        // The 1982 rule.
        let rect = Grid::new(Lattice::Rect, 2, 2).unwrap();
        let mut map = Kohonen1982::new(rect, 2, vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, -1.0, 0.0]).unwrap();
        assert_eq!(msg(Kohonen1982::new(rect, 2, vec![0.0; 7]).unwrap_err()), "weights has length 7; it must be 8");
        assert_eq!(msg(map.step(&[1.0, 0.0], 0.0, Reach::Nearest).unwrap_err()), "alpha = 0 must be finite and positive");
        assert_eq!(msg(map.step(&[1.0, 0.0], f64::INFINITY, Reach::Nearest).unwrap_err()), "alpha = inf must be finite and positive");
        assert_eq!(msg(map.step(&[1.0, 0.0], 1.0, Reach::Radius(-0.5)).unwrap_err()), "radius = -0.5 is outside [0, inf]");
        assert_eq!(msg(map.step(&[1.0], 1.0, Reach::Nearest).unwrap_err()), "inputs has length 1; it must be 2");
        assert_eq!(msg(map.step(&[1.0, f64::NAN], 1.0, Reach::Nearest).unwrap_err()), "x = NaN is not finite");
        assert_eq!(msg(map.winner(&[1.0, 0.0, 0.0]).unwrap_err()), "inputs has length 3; it must be 2");
        assert_eq!(msg(map.winner_coherent(&[1.0, 0.0]).unwrap_err()), "inputs has length 2; it must be 8");
        assert_eq!(msg(map.winner_coherent(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, f64::NEG_INFINITY]).unwrap_err()), "x = -inf is not finite");
        assert_eq!(msg(map.step_coherent(&[1.0, 0.0], 1.0, Reach::Nearest).unwrap_err()), "inputs has length 2; it must be 8");
        assert_eq!(msg(map.step_coherent(&[1.0; 8], -1.0, Reach::Nearest).unwrap_err()), "alpha = -1 must be finite and positive");
        // (1, 0) wins with η = 1 and its whole 2 × 2 neighbourhood updates; unit 3 is (−1, 0), and
        // (−1, 0) + 1·(1, 0) is the zero vector.
        assert_eq!(msg(map.step(&[1.0, 0.0], 1.0, Reach::Nearest).unwrap_err()), "unit 3: m + alpha x is the zero vector and eq. (3) cannot normalise it");
        assert_eq!(map.weights(), &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0, -1.0, 0.0], "nothing stored from the refused step");
        assert_eq!(map.step(&[1.0, 0.0], 1.0, Reach::Radius(0.0)).unwrap(), 0, "(1, 1) has η = 1 too; the lower index wins");
        let hex = Kohonen1982::new(Grid::new(Lattice::Hex, 2, 2).unwrap(), 1, vec![1.0; 4]).unwrap();
        for reach in [Reach::Nearest, Reach::Rings] {
            assert_eq!(msg(hex.clone().step(&[1.0], 1.0, reach).unwrap_err()), "the 1982 paper's ring of nearest neighbours is defined on a rectangular array, not a hexagonal one");
            assert_eq!(msg(hex.clone().step_coherent(&[1.0; 4], 1.0, reach).unwrap_err()), "the 1982 paper's ring of nearest neighbours is defined on a rectangular array, not a hexagonal one");
        }
        assert_eq!(hex.clone().step(&[1.0], 1.0, Reach::Radius(1.0)).unwrap(), 0, "a disc is defined on either");
        assert!(std::error::Error::source(&SomError::ZeroNorm { unit: 0 }).is_none());
    }

    // ----------------------------------------------------------------------------------------------
    // What the papers claim, run.
    // ----------------------------------------------------------------------------------------------

    /// How many of 1000 chains of `n` random scalars in `[0, 1)` come out strictly monotone after the
    /// 1990 ordering phase — 1000 steps, `α = 0.9(1 − t/1000)`, the radius shrinking linearly from
    /// `radius` to one — each step drawing a fresh sample uniform on `[0, 1)`.
    fn ordered_of_1000(n: usize, radius: f64) -> usize {
        (0..1000u64)
            .filter(|&seed| {
                let mut rng = Rng::new(seed);
                let mut som = Som::random(Grid::chain(n).unwrap(), 1, 0.0, 1.0, &mut rng).unwrap();
                let data: Vec<f64> = (0..1000).map(|_| rng.next_f64()).collect();
                som.train(&data, &Phase::ordering_1990(radius), Order::Cyclic).unwrap();
                som.is_ordered_chain().unwrap()
            })
            .count()
    }

    /// A chain orders itself from random weights, and hint 3 of the 1990 paper (p. 1469) holds: "If
    /// the neighborhood is too small to start with, the map will not be ordered globally." Measured
    /// over 1000 seeds each: a chain of 20 whose radius starts at 10, half its length, ends ordered
    /// every time; started at radius one it ends ordered 245 times; a chain of 10 with no
    /// neighbourhood at all — competitive learning, `N_c = {c}` — never, which is what chance
    /// predicts (a random order of 10 is monotone with probability `2/10!`).
    #[test]
    fn a_chain_orders_itself_when_its_neighbourhood_starts_wide() {
        assert_eq!(ordered_of_1000(20, 10.0), 1000);
        assert_eq!(ordered_of_1000(20, 1.0), 245);
        assert_eq!(ordered_of_1000(10, 0.0), 0);
    }

    /// The stationary state of the expected update of a chain of `n` units on `p(x) = 2x` over
    /// `[0, 1]`, for a kernel `h(|c − i|)` that vanishes beyond `reach`: Ritter and Schulten's
    /// equilibrium condition, their eq. (3), `⟨h(r − s)(v − w(s))⟩ = 0`. Solved by iterating
    /// `m_i ← Σ_c h(c, i) ∫_{V_c} x p dx / Σ_c h(c, i) ∫_{V_c} p dx` over the Voronoi intervals `V_c`
    /// of the ordered chain — for which `∫ p = x²` and `∫ x p = 2x³/3` in closed form — until no unit
    /// moves by `1e-13`. Returns the chain and the number of iterations.
    fn stationary(n: usize, h: impl Fn(usize) -> f64, reach: usize) -> (Vec<f64>, usize) {
        let mut m: Vec<f64> = (0..n).map(|i| (i as f64 + 0.5) / n as f64).collect();
        let hs: Vec<f64> = (0..=reach).map(h).collect();
        for iter in 1..=200_000 {
            let mut edges = vec![0.0];
            edges.extend(m.windows(2).map(|w| 0.5 * (w[0] + w[1])));
            edges.push(1.0);
            let mass: Vec<f64> = edges.windows(2).map(|e| e[1] * e[1] - e[0] * e[0]).collect();
            let moment: Vec<f64> = edges.windows(2).map(|e| 2.0 * (e[1] * e[1] * e[1] - e[0] * e[0] * e[0]) / 3.0).collect();
            let new: Vec<f64> = (0..n)
                .map(|i| {
                    let (mut a, mut b) = (0.0, 0.0);
                    for c in i.saturating_sub(reach)..(i + reach + 1).min(n) {
                        a += hs[c.abs_diff(i)] * moment[c];
                        b += hs[c.abs_diff(i)] * mass[c];
                    }
                    a / b
                })
                .collect();
            let moved = max_diff(&new, &m);
            m = new;
            if moved < 1e-13 {
                return (m, iter);
            }
        }
        panic!("no stationary state in 200 000 iterations");
    }

    /// The least-squares slope of `ln(unit density)` against `ln p(x)` over a chain on `p(x) = 2x`,
    /// the `cut` gaps at each end left out: the density of units is `1/(m_{i+1} − m_i)`, at the
    /// midpoint, where `p` is `m_i + m_{i+1}`. The exponent of a magnification law `M ∝ p^e`.
    fn magnification_exponent(m: &[f64], cut: usize) -> f64 {
        let pts: Vec<(f64, f64)> = m.windows(2).map(|w| ((w[0] + w[1]).ln(), -(w[1] - w[0]).ln())).collect();
        let pts = &pts[cut..pts.len() - cut];
        let k = pts.len() as f64;
        let (mx, my) = (pts.iter().map(|p| p.0).sum::<f64>() / k, pts.iter().map(|p| p.1).sum::<f64>() / k);
        let sxy: f64 = pts.iter().map(|p| (p.0 - mx) * (p.1 - my)).sum();
        let sxx: f64 = pts.iter().map(|p| (p.0 - mx) * (p.0 - mx)).sum();
        sxy / sxx
    }

    /// The magnification law, from the stationary states of a chain on `p(x) = 2x`, exponents
    /// fitted over the middle three fifths of its gaps — `n/5` left out at each end, the middle 59
    /// of the 99 gaps of 100 units. With no neighbourhood the rule is vector quantisation, whose
    /// point density the 1990 paper gives as `[p(x)]^{n/(n+r)}` (p. 1466) — `p^{1/3}` for `n = 1`,
    /// `r = 2` — and the measured exponent on 100 units is 0.33332 (78 530 iterations). The 1990
    /// recipe's final bubble of radius one settles at 0.59996. With a Gaussian neighbourhood the
    /// exponent climbs as the kernel widens: 0.64108, 0.66021 and 0.66633 at `σ` = 2, 4 and 8 units.
    /// The last is inflated by the chain's ends: at `σ = 8` the exponent falls to 0.66533 on 200
    /// units and 0.66505 on 400, `1.6e-3` below Ritter and Schulten's `2/3` (their eq. 11 holds for
    /// "all sufficiently small d", p. 104, and `σ = 8` is not small), while at `σ = 2` it moves only
    /// from 0.64108 to 0.64104. The Gaussian's tail past `8σ`, below `e^{−64}`, is dropped on the
    /// longer chains, and the values quoted are to `3e-5` (measured residuals below `1e-5`).
    #[test]
    fn the_stationary_chain_magnifies_as_p_to_the_two_thirds() {
        let gauss = |s: f64| move |d: usize| (-((d * d) as f64) / (s * s)).exp();
        let exponent = |n: usize, s: f64| {
            let (m, _) = stationary(n, gauss(s), (8.0 * s) as usize);
            magnification_exponent(&m, n / 5)
        };
        let (vq, it) = stationary(100, |d| if d == 0 { 1.0 } else { 0.0 }, 0);
        assert!(it > 1000, "{it}");
        let e = magnification_exponent(&vq, 20);
        assert!((e - 1.0 / 3.0).abs() < 3e-5, "{e}");
        let (b1, _) = stationary(100, |_| 1.0, 1);
        let e1 = magnification_exponent(&b1, 20);
        assert!((e1 - 0.6).abs() < 1e-4, "{e1}");
        let mut last = e1 - 0.1;
        for (s, want) in [(2.0, 0.64108), (4.0, 0.66021), (8.0, 0.66633)] {
            let (m, _) = stationary(100, gauss(s), 99);
            let e = magnification_exponent(&m, 20);
            assert!((e - want).abs() < 3e-5 && e > last, "σ = {s}: {e}");
            last = e;
        }
        let (e200, e400) = (exponent(200, 8.0), exponent(400, 8.0));
        assert!((e200 - 0.66533).abs() < 3e-5 && (e400 - 0.66505).abs() < 3e-5, "{e200} {e400}");
        assert!(e400 < e200 && e200 < last && 2.0 / 3.0 - e400 > 1.5e-3, "the long chain's exponent falls below 2/3");
        let e2 = exponent(200, 2.0);
        assert!((e2 - 0.64104).abs() < 3e-5 && (e2 - 0.64108).abs() < 1e-4, "{e2}");
    }

    /// And the map itself goes there. Ritter and Schulten's own test (their Fig. 9) started a chain
    /// at `w(r) = √r`, whose magnification is proportional to `p` — exponent 1.0000 here — and
    /// trained it on `p(x) = 2x`. Here: 100 units, a Gaussian of `σ = 8`, `α = 0.01` held for
    /// a million steps, the weights averaged over the second half. The average lies within
    /// `8.8e-4` (measured; units are about `0.01` apart) of the stationary state of the test above,
    /// and its exponent is 0.66449 against the stationary 0.66633: the proportional map is not
    /// where the rule rests.
    #[test]
    fn the_trained_chain_settles_where_eq_3_says() {
        let (n, sigma, steps) = (100, 8.0, 1_000_000);
        let (star, _) = stationary(n, |d| (-((d * d) as f64) / (sigma * sigma)).exp(), n - 1);
        let w0: Vec<f64> = (0..n).map(|i| ((i as f64 + 0.5) / n as f64).sqrt()).collect();
        assert!((magnification_exponent(&w0, 20) - 1.0).abs() < 1e-3);
        let mut som = Som::new(Grid::chain(n).unwrap(), 1, w0).unwrap();
        let mut rng = Rng::new(1986);
        let mut mean = vec![0.0; n];
        for t in 0..steps {
            som.step(&[rng.next_f64().sqrt()], Kernel::Gaussian, 0.01, sigma).unwrap();
            if t >= steps / 2 {
                for (a, w) in mean.iter_mut().zip(som.weights()) {
                    *a += w;
                }
            }
        }
        for a in &mut mean {
            *a /= (steps / 2) as f64;
        }
        let d = max_diff(&mean, &star);
        assert!(d < 2e-3, "{d:e}");
        let e = magnification_exponent(&mean, 20);
        assert!((e - magnification_exponent(&star, 20)).abs() < 5e-3 && e < 0.67, "{e}");
    }

    /// Table 1 of the 1990 paper (p. 1469), item by item `A`–`Z`, `1`–`6`: the five attributes.
    const TABLE1: [[f64; 5]; 32] = {
        let a1 = [1, 2, 3, 4, 5, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3];
        let a2 = [0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3];
        let a3 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 3, 3, 3, 3, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6];
        let a4 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 1, 2, 3, 4, 2, 2, 2, 2, 2, 2];
        let a5 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6];
        let mut t = [[0.0; 5]; 32];
        let mut i = 0;
        while i < 32 {
            t[i] = [a1[i] as f64, a2[i] as f64, a3[i] as f64, a4[i] as f64, a5[i] as f64];
            i += 1;
        }
        t
    };

    /// Fig. 6 of the 1990 paper (p. 1469) as printed: ten units wide, seven high, `*` for a unit no
    /// item labels, the first, third, fifth and seventh rows shifted half a unit right.
    const FIG6: [&str; 7] = ["BCDE*QR*YZ", "A****P**X*", "*F*NO*W**1", "*G*M****2*", "HKL*TU*3**", "*I******4*", "*J*S**V*56"];

    const ITEMS: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ123456";

    /// The item pairs of Table 1 at distance one: exactly the 31 edges of Fig. 7's minimal spanning
    /// tree, since every other pair is further apart.
    fn table1_edges() -> Vec<(usize, usize)> {
        let mut e = Vec::new();
        for a in 0..32 {
            for b in a + 1..32 {
                let d2: f64 = (0..5).map(|k| (TABLE1[a][k] - TABLE1[b][k]).powi(2)).sum();
                if d2 == 1.0 {
                    e.push((a, b));
                }
            }
        }
        e
    }

    /// Table 1's taxonomy map, rerun at the printed parameters: the hexagonal 10 × 7 array, 1000
    /// steps with `α` from 0.5 to 0.04 and the radius from six to one, then 10 000 with `α` from 0.04
    /// to zero at radius one, items drawn at random, weights starting uniform in `[0, 1)` (the paper
    /// says only "random"). Fig. 6's own map puts the two ends of the tree's 31 edges 1.2357 units
    /// apart on average, 23 of them on adjacent units, and gives every item a unit of its own.
    /// Over 100 seeds the reruns average 1.2619 with 21.2 adjacent edges, 43 are at least as tight
    /// as Fig. 6, and 28 give all 32 items distinct units: the printed map is a typical run, a
    /// little better than the median, and a lucky one in keeping every item apart. On the same
    /// maps the quantisation error runs from 0.165 to 0.332 (median 0.255) and the topographic error
    /// from 0 to 0.125 (median 1/32, one item).
    ///
    /// Hint 1's rule of thumb (p. 1469) asks for 35 000 steps for 70 units, and the same hint allows
    /// that "for 'fast learning' ... 10 000 steps and even less may sometimes be enough". With the
    /// second phase stretched to 34 000 steps, `α` still falling linearly from 0.04 to zero, and
    /// the same seeds, the reruns average 1.2891, 28 are as tight as Fig. 6 and 30 keep every item
    /// apart: the longer run does not tighten the map.
    #[test]
    fn table_1_maps_like_fig_6() {
        let edges = table1_edges();
        assert_eq!(edges.len(), 31);
        let tall = Grid::new(Lattice::Hex, 10, 8).unwrap();
        let mut at = [usize::MAX; 32];
        for (r, row) in FIG6.iter().enumerate() {
            for (c, ch) in row.chars().enumerate() {
                if let Some(k) = ITEMS.find(ch) {
                    // Printed row r is row r + 1 of a grid whose odd rows are the shifted ones.
                    at[k] = (r + 1) * 10 + c;
                }
            }
        }
        assert!(at.iter().all(|&u| u != usize::MAX));
        let fig: f64 = edges.iter().map(|&(a, b)| tall.distance(at[a], at[b])).sum::<f64>() / 31.0;
        assert!((fig - 1.2357).abs() < 1e-4, "{fig}");
        assert_eq!(edges.iter().filter(|&&(a, b)| tall.adjacent(at[a], at[b])).count(), 23);
        let data: Vec<f64> = TABLE1.iter().flatten().copied().collect();
        let grid = Grid::new(Lattice::Hex, 10, 7).unwrap();
        let [first, second] = Phase::table1_1990();
        let stretch = |steps: usize| Phase { steps, alpha: Schedule::Linear { start: 0.04, end: 0.0, steps }, ..second };
        assert_eq!(stretch(10_000), second, "the printed second phase is the 10 000-step stretch");
        let rerun = |last: Phase| {
            let (mut tight, mut distinct, mut sum, mut adjacent) = (0, 0, 0.0, 0);
            let (mut qe, mut te) = (Vec::new(), Vec::new());
            for seed in 0..100u64 {
                let mut rng = Rng::new(seed);
                let mut som = Som::random(grid, 5, 0.0, 1.0, &mut rng).unwrap();
                for p in [first, last] {
                    som.train(&data, &p, Order::Random(&mut rng)).unwrap();
                }
                let img: Vec<usize> = TABLE1.iter().map(|x| som.winner(x).unwrap()).collect();
                let mean = edges.iter().map(|&(a, b)| grid.distance(img[a], img[b])).sum::<f64>() / 31.0;
                sum += mean;
                tight += usize::from(mean <= fig);
                adjacent += edges.iter().filter(|&&(a, b)| grid.adjacent(img[a], img[b])).count();
                let mut used = img.clone();
                used.sort_unstable();
                used.dedup();
                distinct += usize::from(used.len() == 32);
                qe.push(som.quantization_error(&data).unwrap());
                te.push(som.topographic_error(&data).unwrap());
            }
            qe.sort_by(f64::total_cmp);
            te.sort_by(f64::total_cmp);
            (sum / 100.0, adjacent as f64 / 100.0, tight, distinct, qe, te)
        };
        let (mean, adjacent, tight, distinct, qe, te) = rerun(second);
        assert!((mean - 1.2619).abs() < 1e-4 && (adjacent - 21.24).abs() < 1e-9, "{mean} {adjacent}");
        assert_eq!((tight, distinct), (43, 28));
        assert!((qe[50] - 0.2550).abs() < 1e-3 && te[50] == 1.0 / 32.0, "{} {}", qe[50], te[50]);
        assert!((qe[0] - 0.1653).abs() < 1e-3 && (qe[99] - 0.3319).abs() < 1e-3 && te[0] == 0.0 && te[99] == 0.125);
        let (long, _, tight, distinct, _, _) = rerun(stretch(34_000));
        assert!((long - 1.2891).abs() < 1e-4 && long > mean, "{long}");
        assert_eq!((tight, distinct), (28, 30));
    }

    /// The two errors measure different things, and one buys the other. Twenty rectangular 10 × 10
    /// maps are trained by the 1990 recipe — ordering from radius ten, then `500 × 100` steps at
    /// radius one — on 2000 points uniform in the unit square; then 50 000 further steps with NO
    /// neighbourhood, the radius-zero "simple competitive learning" of p. 1467 with `α` from 0.01
    /// to zero, lower the quantisation error of every one of the 20 (0.0404 to 0.0361 on average)
    /// and raise its topographic error (0.0205 to 0.0453): the neighbourhood buys order with
    /// resolution.
    #[test]
    fn dropping_the_neighbourhood_trades_order_for_resolution() {
        let (mut qa, mut qb, mut ta, mut tb) = (0.0, 0.0, 0.0, 0.0);
        let grid = Grid::new(Lattice::Rect, 10, 10).unwrap();
        for seed in 0..20u64 {
            let mut rng = Rng::new(seed);
            let mut som = Som::random(grid, 2, 0.0, 1.0, &mut rng).unwrap();
            let data: Vec<f64> = (0..4000).map(|_| rng.next_f64()).collect();
            som.train(&data, &Phase::ordering_1990(10.0), Order::Random(&mut rng)).unwrap();
            som.train(&data, &Phase::convergence_1990(grid), Order::Random(&mut rng)).unwrap();
            let (q0, t0) = (som.quantization_error(&data).unwrap(), som.topographic_error(&data).unwrap());
            let vq = Phase { steps: 50_000, alpha: Schedule::Linear { start: 0.01, end: 0.0, steps: 50_000 }, width: Schedule::Constant(0.0), kernel: Kernel::Bubble };
            som.train(&data, &vq, Order::Random(&mut rng)).unwrap();
            let (q1, t1) = (som.quantization_error(&data).unwrap(), som.topographic_error(&data).unwrap());
            assert!(q1 < q0 && t1 > t0, "seed {seed}: {q0} {t0} -> {q1} {t1}");
            (qa, ta, qb, tb) = (qa + q0 / 20.0, ta + t0 / 20.0, qb + q1 / 20.0, tb + t1 / 20.0);
        }
        assert!((qa - 0.04044).abs() < 1e-4 && (qb - 0.03610).abs() < 1e-4, "{qa} {qb}");
        assert!((ta - 0.0205).abs() < 1e-3 && (tb - 0.0453).abs() < 1e-3, "{ta} {tb}");
    }

    /// The half-width of Fig. 3a's square in the front view, as a fraction of the sphere's radius:
    /// the inscribed square's `1/√2`. Measured on the page at 400 dpi, the drawn square is 431
    /// pixels wide inside a circle 604 across, 0.714.
    const PATCH: f64 = core::f64::consts::FRAC_1_SQRT_2;

    /// A point of the sphere above front-view position `(x, y)`. The `max` keeps a corner of the
    /// inscribed square, where `x² + y²` can round to just above one, on the sphere.
    fn lift(x: f64, y: f64) -> [f64; 3] {
        [x, y, (1.0 - x * x - y * y).max(0.0).sqrt()]
    }

    /// A training vector of Simulations 1–3 as read here: a point uniform on the square
    /// `[−PATCH, PATCH]²` of the front view, lifted onto the unit sphere. Uniform in the front view,
    /// its density per unit area of the sphere is proportional to `z`.
    fn sphere_square(rng: &mut Rng) -> [f64; 3] {
        lift(PATCH * (2.0 * rng.next_f64() - 1.0), PATCH * (2.0 * rng.next_f64() - 1.0))
    }

    /// How many of the 8 × 8 test vectors of Fig. 3b — the centres of an 8 × 8 tiling of the square —
    /// land on their own unit, under the best of the square's eight symmetries (the paper's "eight
    /// equally probable symmetrical alternatives", p. 62). Fig. 3c's map scores 62.
    fn fig3_score(map: &Kohonen1982) -> usize {
        let centre = |i: usize| PATCH * ((2 * i + 1) as f64 / 8.0 - 1.0);
        let image: Vec<(usize, usize)> = (0..64).map(|k| map.grid().coords(map.winner(&lift(centre(k % 8), centre(k / 8))).unwrap())).collect();
        let turn = |s: usize, (i, j): (usize, usize)| {
            let (i, j) = if s & 4 == 0 { (i, j) } else { (j, i) };
            (if s & 1 == 0 { i } else { 7 - i }, if s & 2 == 0 { j } else { 7 - j })
        };
        (0..8).map(|s| (0..64).filter(|&k| image[k] == turn(s, (k % 8, k / 8))).count()).max().unwrap()
    }

    /// Simulations 1–3 of the 1982 paper, rerun: an 8 × 8 array, eq. (3), the gain "proportional to
    /// 1/t" (p. 61) — the constant, which the paper does not give, taken from the 1990 paper's
    /// `α'(t) = 100/t` for the same normalised rule — weights starting uniform in `[0, 1)³`, and
    /// training vectors uniform on the inscribed square of the front view, lifted onto the sphere.
    /// Fifty runs of each, scored by `fig3_score` at 500 and at 10 000 steps:
    ///
    /// - the 1990 form, eq. (10) with `N_c` shrinking from radius 8 to 1 over 2000 steps, orders
    ///   every run completely (64 of 64 test vectors home in 50 of 50 runs);
    /// - the 1982 paper's fixed ring of eight neighbours orders 13 of 50 completely, and 17 reach
    ///   Fig. 3c's 62;
    /// - Simulation 3's second ring at `α/4` makes each clause of p. 62's "ordering seems to proceed
    ///   more quickly and more reliably; on the other hand, the final result is perhaps not as good
    ///   as before" true by its own measure. More quickly: a mean of 26.94 of the 64 home at 500
    ///   steps, against 13.36 for the single ring. More reliably: a mean of 47.64 at 10 000 steps
    ///   against 35.04, and 42 runs of 50 with at least half home, against 23. Not as good: no run
    ///   with even 60 home, where the single ring has 17 at 62 or more.
    #[test]
    fn simulations_1_to_3_of_1982() {
        let run = |reach: &dyn Fn(usize) -> Reach| {
            let (mut early, mut last) = (0, Vec::new());
            for seed in 0..50u64 {
                let mut rng = Rng::new(seed);
                let w: Vec<f64> = (0..192).map(|_| rng.next_f64()).collect();
                let mut map = Kohonen1982::new(Grid::new(Lattice::Rect, 8, 8).unwrap(), 3, w).unwrap();
                for t in 0..10_000 {
                    let x = sphere_square(&mut rng);
                    map.step(&x, Schedule::InverseT { c: 100.0 }.at(t).unwrap(), reach(t)).unwrap();
                    if t == 499 {
                        early += fig3_score(&map);
                    }
                }
                last.push(fig3_score(&map));
            }
            let at_least = |k: usize| last.iter().filter(|&&s| s >= k).count();
            (early as f64 / 50.0, last.iter().sum::<usize>() as f64 / 50.0, [at_least(32), at_least(60), at_least(62), at_least(64)])
        };
        let shrinking = run(&|t| Reach::Radius(Schedule::Linear { start: 8.0, end: 1.0, steps: 2000 }.at(t).unwrap()));
        let nearest = run(&|_| Reach::Nearest);
        let rings = run(&|_| Reach::Rings);
        assert_eq!(shrinking.2, [50; 4]);
        assert_eq!((nearest.2, rings.2), ([23, 17, 17, 13], [42, 0, 0, 0]));
        assert!((rings.0 - 26.94).abs() < 1e-9 && (nearest.0 - 13.36).abs() < 1e-9, "{rings:?} {nearest:?}");
        assert!((rings.1 - 47.64).abs() < 1e-9 && (nearest.1 - 35.04).abs() < 1e-9, "{rings:?} {nearest:?}");
    }

    /// Simulation 4's resonator as read here: a second-order band-pass of quality `q` tuned to `f0`,
    /// its gain at frequency `f`, one at resonance.
    fn bandpass(f: f64, f0: f64, q: f64) -> f64 {
        let (r, b) = (f / f0, f / (q * f0));
        b / ((1.0 - r * r).powi(2) + b * b).sqrt()
    }

    /// One run of Simulation 4 (1982, Sect. 2.3, Table 1) at its printed numbers — 20 resonators of
    /// `Q = 2.5` tuned at random in `[1, 2]`, ten units each reading five of them chosen at random,
    /// 2000 training frequencies uniform in `[0.5, 1]`, the two-neighbour chain of eq. (3) — with the
    /// gain `100/t`, weights starting uniform in `[0, 1)`, and each unit's five inputs normalised to
    /// unit length or not. The test frequencies are 0.50, 0.51, …, 1, numbered 0 to 50. Returns the
    /// winning unit at each test frequency, and each unit's preferred test frequency: the one its
    /// response `η_i` is largest at, the lowest on a tie — Table 1's "test frequencies to which each
    /// processing unit became most sensitive".
    fn simulation_4(seed: u64, normalised: bool) -> (Vec<usize>, Vec<usize>) {
        let mut rng = Rng::new(seed);
        let f0: Vec<f64> = (0..20).map(|_| 1.0 + rng.next_f64()).collect();
        let taps: Vec<Vec<usize>> = (0..10)
            .map(|_| {
                let mut pool: Vec<usize> = (0..20).collect();
                (0..5).map(|_| pool.remove(rng.below(pool.len() as u32) as usize)).collect()
            })
            .collect();
        let inputs = |f: f64| -> Vec<f64> {
            taps.iter()
                .flat_map(|t| {
                    let v: Vec<f64> = t.iter().map(|&j| bandpass(f, f0[j], 2.5)).collect();
                    let n = if normalised { v.iter().map(|a| a * a).sum::<f64>().sqrt() } else { 1.0 };
                    v.into_iter().map(move |a| a / n)
                })
                .collect()
        };
        let w: Vec<f64> = (0..50).map(|_| rng.next_f64()).collect();
        let mut map = Kohonen1982::new(Grid::chain(10).unwrap(), 5, w).unwrap();
        for t in 0..2000 {
            let f = 0.5 + 0.5 * rng.next_f64();
            map.step_coherent(&inputs(f), 100.0 / (t as f64 + 1.0), Reach::Nearest).unwrap();
        }
        let tests: Vec<Vec<f64>> = (0..=50).map(|k| inputs(0.5 + 0.01 * f64::from(k))).collect();
        let winners = tests.iter().map(|x| map.winner_coherent(x).unwrap()).collect();
        let preferred = (0..10)
            .map(|i| {
                let eta: Vec<f64> = tests.iter().map(|x| map.response(i, &x[i * 5..(i + 1) * 5])).collect();
                let top = eta.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                eta.iter().position(|&e| e == top).unwrap()
            })
            .collect();
        (winners, preferred)
    }

    /// Simulation 4 does not come out of its caption, and the numbers below are why. Every
    /// resonance lies in `[1, 2]` and every training frequency in `[0.5, 1]`, and a band-pass gain
    /// rises all the way up to resonance — so all fifty inputs, ten units × five taps on the 20
    /// resonators, rise together across the training band (checked below on a grid of 101
    /// resonances). Read literally, with raw amplitudes as inputs, every unit is most sensitive to
    /// the top test frequency, 1.00, in 200 of 200 runs, and one unit takes everything — Sect. 4.5's
    /// "focusing" — winning all 51 test frequencies in 175. Normalise each unit's five inputs, as
    /// Simulation 1 normalised its training vectors, and a frequency map forms. Table 1's own
    /// statistic, each unit's preferred frequency, runs monotonically along the chain in 117 of 200
    /// runs, strictly in 97, with 9.52 distinct values of the ten on average; seed 0 rises from
    /// 0.58 to 0.92 at unit 7 and folds back over the last three units, as Experiment 1 folds at its
    /// tenth, and seed 1 falls from 0.93 to 0.56, as Experiment 2 does. A different measure, the
    /// winner at each of the 51 test frequencies, runs monotonically in 41 of 200 runs and uses 9.365
    /// of the ten units on average.
    #[test]
    fn simulation_4_orders_only_with_normalised_inputs() {
        for k in 0..=100 {
            let f0 = 1.0 + 0.01 * f64::from(k);
            for j in 0..50 {
                let (f, g) = (0.5 + 0.01 * f64::from(j), 0.51 + 0.01 * f64::from(j));
                assert!(bandpass(g, f0, 2.5) > bandpass(f, f0, 2.5), "f0 = {f0}, f = {f}");
            }
        }
        assert!((bandpass(1.5, 1.5, 2.5) - 1.0).abs() < 1e-15);
        let monotone = |v: &[usize]| v.windows(2).all(|p| p[0] <= p[1]) || v.windows(2).all(|p| p[0] >= p[1]);
        let strict = |v: &[usize]| v.windows(2).all(|p| p[0] < p[1]) || v.windows(2).all(|p| p[0] > p[1]);
        let distinct = |v: &[usize]| {
            let mut u = v.to_vec();
            u.sort_unstable();
            u.dedup();
            u.len()
        };
        let (mut one, mut top) = (0, 0);
        for s in 0..200 {
            let (w, p) = simulation_4(s, false);
            one += usize::from(distinct(&w) == 1);
            top += usize::from(p.iter().all(|&k| k == 50));
        }
        assert_eq!((one, top), (175, 200));
        let (mut ordered, mut used, mut mono, mut strictly, mut values) = (0, 0, 0, 0, 0);
        for s in 0..200 {
            let (w, p) = simulation_4(s, true);
            ordered += usize::from(monotone(&w));
            used += distinct(&w);
            mono += usize::from(monotone(&p));
            strictly += usize::from(strict(&p));
            values += distinct(&p);
            match s {
                0 => assert_eq!(p, [8, 10, 17, 24, 36, 41, 42, 36, 31, 30]),
                1 => assert_eq!(p, [43, 38, 35, 31, 27, 22, 19, 14, 8, 6]),
                _ => {}
            }
        }
        assert_eq!((mono, strictly, values, ordered, used), (117, 97, 1904, 41, 1873));
    }
}
