//! Vector symbolic architectures — hyperdimensional computing — with every capacity claim checked
//! against its closed form.
//!
//! # What the mechanism is
//!
//! Represent every symbol as a random vector in a space of thousands of dimensions, and then do
//! symbolic computation with three algebraic operations that keep you in that space:
//!
//! - **binding** `a ⊗ b`: a new vector *dissimilar* to both inputs, from which either can be
//!   recovered given the other — a key-value pair, a role-filler pair, a variable bound to a value;
//! - **bundling** `a + b + c`: a vector *similar* to each input — a set, a superposition;
//! - **permutation** `ρ(a)`: a vector dissimilar to `a` and exactly invertible — position, order,
//!   time.
//!
//! What makes this work is a fact about high dimensions and nothing else: two random vectors in
//! `D = 10,000` dimensions are almost exactly orthogonal, with a similarity of order `1/√D`. So a
//! bundle of seven symbols is still recognisably similar to each of them (0.31, closed form below)
//! and dissimilar to every other symbol in a codebook of thousands (`±0.01`). Reading a symbol out
//! of a composite is a nearest-neighbour search against the codebook, which is why the operation
//! count is the whole story of what it costs.
//!
//! Kanerva, *Hyperdimensional computing: an introduction to computing in distributed representation
//! with high-dimensional random vectors*, Cognitive Computation 1(2):139–159, 2009, is the
//! introduction; Plate, *Holographic reduced representations*, IEEE Transactions on Neural Networks
//! 6(3):623–641, 1995, is the real-valued original; Gayler's multiply-add-permute model and
//! Kanerva's binary spatter code are the two discrete ones. Kleyko, Rachkovskij, Osipov and Rahimi,
//! *A survey on hyperdimensional computing aka vector symbolic architectures*, ACM Computing Surveys
//! 55(6) and 55(9), 2022–2023, is the map of the field.
//!
//! # Why it is in a neuromorphic crate
//!
//! Three reasons, each of which this module makes concrete rather than asserts.
//!
//! 1. **The operations are what a neuromorphic core does natively.** A binding is `D`
//!    element-wise operations with no multiplier for the discrete models; a bundle is a sum and a
//!    threshold; a nearest-neighbour readout is `M · D` accumulates. There is no back-propagation
//!    and no gradient — learning a class is bundling its examples. [`Meter`] counts every one of
//!    those operations so the cost is a number rather than an adjective, in the same units
//!    [`crate::ledger`] prices.
//! 2. **Noise tolerance is a closed form, not a hope.** A bundle of `k` random bipolar vectors has
//!    a similarity to each member of exactly `C(k−1, (k−1)/2) / 2^(k−1)` for odd `k` — 0.5 at three,
//!    0.375 at five, 0.3125 at seven — and the noise from every other symbol is `1/√D`. So the
//!    number of dimensions a task needs is arithmetic ([`bundle_similarity`]), which is what makes a
//!    low-precision, faulty, analog substrate acceptable: a flipped bit in a 10,000-dimensional
//!    vector is 1e-4 of a similarity.
//! 3. **Factorising a bound composite is the operation every scene-understanding demonstration on
//!    neuromorphic hardware runs**, and it has an algorithm: the resonator network of Frady, Kent,
//!    Olshausen and Sommer, *Resonator networks, 1: an efficient solution for factoring
//!    high-dimensional, distributed representations of data structures*, Neural Computation
//!    32(12):2311–2331, 2020. [`Resonator`] is that algorithm; its test factorises 8,000 candidate
//!    triples by searching all three codebooks at once, and its doc carries the measured success
//!    rate rather than the paper's, because the two differ.
//!
//! # The three models
//!
//! | model | vector | binding | bundling | similarity |
//! |---|---|---|---|---|
//! | [`Bipolar`] (multiply-add-permute) | `{−1, +1}^D` | element-wise product, **self-inverse** | sum, then sign | cosine, `= dot / D` |
//! | [`Binary`] (binary spatter code) | `{0, 1}^D` | XOR, **self-inverse** | majority | `1 − 2·Hamming / D` |
//! | [`Hrr`] (holographic reduced) | `N(0, 1/D)^D` | circular convolution | sum | dot product |
//!
//! The two discrete models are exact where the real one is approximate: `(a ⊗ b) ⊗ b = a` bit for
//! bit, and a permutation is a cyclic shift, exactly undone by the opposite shift. Holographic
//! reduced representations bind by circular convolution and unbind by correlation with the
//! *involution* `a*[i] = a[−i mod D]`, which is only the approximate inverse: `(a ⊛ b) ⊛ b*` is
//! `a` plus noise of order `1/√D` per element. Both facts are tested, the exact ones exactly.
//!
//! # Ties
//!
//! A majority over an even number of bipolar vectors can tie. This module breaks a tie toward
//! `+1` (and toward `1` in the binary model) — deterministically, so the same bundle is the same
//! vector on every platform — and says so here rather than adding a random tie-breaker vector,
//! which is Kanerva's suggestion and which would make a bundle depend on a seed. The closed-form
//! similarity is stated for odd `k`, where there is no tie to break; for even `k` the tie rule
//! biases the bundle toward `+1` by an amount the test measures.
//!
//! # What this module has NOT reproduced
//!
//! - The phasor / complex-valued model that maps onto spike *timing* (Frady and Sommer, PNAS
//!   116(36):18050–18058, 2019) — the one that runs on Loihi as a rhythmic spike pattern. This
//!   review located the mapping and did not implement it; the three models here are the ones the
//!   survey treats as canonical, and the spiking mapping deserves its own closed forms.
//! - Sparse block codes and the other 2020s variants in the survey's Part II. Named, not built.
//! - Any learned encoder from raw sensor data to a hypervector. The "random projection then
//!   bundle" classifiers in the literature are two lines on top of this module and their accuracy
//!   is a property of the dataset, not of the algebra, so no number is claimed here.

use core::fmt;

use crate::rng::Rng;

/// What went wrong, named rather than guessed around.
#[derive(Debug, Clone, PartialEq)]
pub enum VsaError {
    /// A dimension of zero, or a codebook or bundle with no members.
    Empty {
        /// What was empty: `"dimension"`, `"codebook"`, `"bundle"`, `"sequence"`.
        what: &'static str,
    },
    /// Two vectors of different dimensions were combined.
    Dimension {
        /// The first dimension.
        a: usize,
        /// The second.
        b: usize,
    },
    /// A real vector carried a `NaN` or an infinity.
    NonFinite {
        /// Position of the first offending element.
        index: usize,
    },
    /// A bipolar vector carried something other than `−1` or `+1`, or a binary one something other
    /// than `0` or `1`.
    NotSymbolic {
        /// Position of the first offending element.
        index: usize,
        /// The value found there.
        value: f64,
    },
    /// A resonator was asked to factor with a number of codebooks its state does not match.
    Factors {
        /// Codebooks supplied.
        got: usize,
        /// Codebooks the resonator was built with.
        want: usize,
    },
}

impl fmt::Display for VsaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty { what } => write!(f, "{what} is empty"),
            Self::Dimension { a, b } => write!(f, "dimension {a} against dimension {b}"),
            Self::NonFinite { index } => write!(f, "element {index} is not finite"),
            Self::NotSymbolic { index, value } => {
                write!(f, "element {index} is {value}, not a symbol of this model")
            }
            Self::Factors { got, want } => {
                write!(f, "{got} codebooks handed to a resonator built for {want}")
            }
        }
    }
}

impl std::error::Error for VsaError {}

fn same_dim(a: usize, b: usize) -> Result<(), VsaError> {
    if a == b { Ok(()) } else { Err(VsaError::Dimension { a, b }) }
}

fn non_empty(n: usize, what: &'static str) -> Result<(), VsaError> {
    if n == 0 { Err(VsaError::Empty { what }) } else { Ok(()) }
}

// ---------------------------------------------------------------------------------------------
// Operation counts
// ---------------------------------------------------------------------------------------------

/// Element-wise operations a sequence of VSA calls performed, by kind.
///
/// Every operation in this module that touches `D` elements adds `D` to one of these. A bind in a
/// discrete model is `D` operations with **no multiplier** — an XOR or a sign flip — and is counted
/// as [`Meter::binds`]; a similarity is `D` multiply-accumulates (or `D` XOR-and-popcount steps in
/// the binary model) and a codebook search is `M` of them. The counts are what a neuromorphic
/// implementation is billed for, in the same spirit as [`crate::ledger::Ledger`]: an integer, not
/// an estimate.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Meter {
    /// Element-wise binding operations (products, XORs, or convolution multiply-adds for HRR,
    /// which are `D²` per bind and counted as such).
    pub binds: u64,
    /// Element-wise additions in bundles, including the threshold pass.
    pub bundles: u64,
    /// Element moves in permutations.
    pub permutes: u64,
    /// Multiply-accumulates (or XOR-popcount steps) in similarity computations.
    pub similarities: u64,
}

impl Meter {
    /// Every count summed, or `None` on overflow.
    #[must_use]
    pub fn total(&self) -> Option<u64> {
        self.binds
            .checked_add(self.bundles)?
            .checked_add(self.permutes)?
            .checked_add(self.similarities)
    }

}

/// Add `n` element operations to one of a [`Meter`]'s counters, saturating.
fn tally(field: &mut u64, n: usize) {
    *field = field.saturating_add(n as u64);
}

// ---------------------------------------------------------------------------------------------
// Bipolar: multiply-add-permute
// ---------------------------------------------------------------------------------------------

/// The multiply-add-permute model over `{−1, +1}^D` (Gayler, 1998).
///
/// Values are stored as `f64` so that a bundle's integer sums and a similarity's cosine share a
/// type with the other models; every entry of a vector this type produces is exactly `−1.0` or
/// `+1.0`, and every entry of a vector it accepts is checked to be.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bipolar {
    /// Dimension `D`. At least one.
    pub dim: usize,
}

impl Bipolar {
    /// A model of dimension `dim`.
    ///
    /// # Errors
    ///
    /// [`VsaError::Empty`] for a zero dimension.
    pub fn new(dim: usize) -> Result<Self, VsaError> {
        non_empty(dim, "dimension")?;
        Ok(Self { dim })
    }

    /// A random symbol: each element `±1` with equal probability.
    #[must_use]
    pub fn random(&self, rng: &mut Rng) -> Vec<f64> {
        (0..self.dim).map(|_| if rng.next_u32() & 1 == 1 { 1.0 } else { -1.0 }).collect()
    }

    /// Check that `v` is a symbol of this model.
    ///
    /// # Errors
    ///
    /// [`VsaError::Dimension`] for the wrong length, [`VsaError::NotSymbolic`] for an entry that is
    /// not exactly `±1`.
    pub fn check(&self, v: &[f64]) -> Result<(), VsaError> {
        same_dim(v.len(), self.dim)?;
        for (i, &x) in v.iter().enumerate() {
            if x != 1.0 && x != -1.0 {
                return Err(VsaError::NotSymbolic { index: i, value: x });
            }
        }
        Ok(())
    }

    /// `a ⊗ b`: the element-wise product. Self-inverse: `bind(bind(a, b), b) == a` exactly.
    ///
    /// # Errors
    ///
    /// As [`Bipolar::check`], on either argument.
    pub fn bind(&self, a: &[f64], b: &[f64], meter: &mut Meter) -> Result<Vec<f64>, VsaError> {
        self.check(a)?;
        self.check(b)?;
        tally(&mut meter.binds, self.dim);
        Ok(a.iter().zip(b).map(|(x, y)| x * y).collect())
    }

    /// The majority of `members`: element-wise sum, then sign, ties toward `+1`.
    ///
    /// # Errors
    ///
    /// [`VsaError::Empty`] for no members, plus [`Bipolar::check`] on each.
    pub fn bundle(&self, members: &[Vec<f64>], meter: &mut Meter) -> Result<Vec<f64>, VsaError> {
        non_empty(members.len(), "bundle")?;
        let mut sum = vec![0.0f64; self.dim];
        for m in members {
            self.check(m)?;
            for (s, &x) in sum.iter_mut().zip(m) {
                *s += x;
            }
        }
        tally(&mut meter.bundles, self.dim * (members.len() + 1));
        Ok(sum.iter().map(|&s| if s >= 0.0 { 1.0 } else { -1.0 }).collect())
    }

    /// `ρ^k(a)`: cyclic shift right by `k`. `permute(permute(a, k), −k) == a` exactly, for which
    /// pass a negative `k` to shift left.
    ///
    /// # Errors
    ///
    /// As [`Bipolar::check`].
    pub fn permute(&self, a: &[f64], k: i64, meter: &mut Meter) -> Result<Vec<f64>, VsaError> {
        self.check(a)?;
        tally(&mut meter.permutes, self.dim);
        Ok(rotate(a, k))
    }

    /// Cosine similarity, `dot / D`, in `[−1, 1]`. Exactly `1` for identical vectors.
    ///
    /// # Errors
    ///
    /// As [`Bipolar::check`], on either argument.
    pub fn similarity(&self, a: &[f64], b: &[f64], meter: &mut Meter) -> Result<f64, VsaError> {
        self.check(a)?;
        self.check(b)?;
        tally(&mut meter.similarities, self.dim);
        Ok(a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>() / self.dim as f64)
    }
}

// ---------------------------------------------------------------------------------------------
// Binary: the binary spatter code
// ---------------------------------------------------------------------------------------------

/// The binary spatter code over `{0, 1}^D` (Kanerva, 1996; 2009).
///
/// Stored as `f64` entries of exactly `0.0` or `1.0`, for the same reason [`Bipolar`] is. The two
/// models are isomorphic under `x ↦ 1 − 2x`, which [`Binary::to_bipolar`] and
/// [`Binary::from_bipolar`] implement, and the isomorphism is tested: an XOR bind maps to a
/// product bind exactly, and the normalised Hamming similarity `1 − 2h/D` maps to the cosine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Binary {
    /// Dimension `D`. At least one.
    pub dim: usize,
}

impl Binary {
    /// A model of dimension `dim`.
    ///
    /// # Errors
    ///
    /// [`VsaError::Empty`] for a zero dimension.
    pub fn new(dim: usize) -> Result<Self, VsaError> {
        non_empty(dim, "dimension")?;
        Ok(Self { dim })
    }

    /// A random symbol: each bit `0` or `1` with equal probability.
    #[must_use]
    pub fn random(&self, rng: &mut Rng) -> Vec<f64> {
        (0..self.dim).map(|_| f64::from(rng.next_u32() & 1)).collect()
    }

    /// Check that `v` is a symbol of this model.
    ///
    /// # Errors
    ///
    /// [`VsaError::Dimension`] for the wrong length, [`VsaError::NotSymbolic`] for an entry that is
    /// not exactly `0` or `1`.
    pub fn check(&self, v: &[f64]) -> Result<(), VsaError> {
        same_dim(v.len(), self.dim)?;
        for (i, &x) in v.iter().enumerate() {
            if x != 0.0 && x != 1.0 {
                return Err(VsaError::NotSymbolic { index: i, value: x });
            }
        }
        Ok(())
    }

    /// `a ⊕ b`: element-wise XOR. Self-inverse.
    ///
    /// # Errors
    ///
    /// As [`Binary::check`], on either argument.
    pub fn bind(&self, a: &[f64], b: &[f64], meter: &mut Meter) -> Result<Vec<f64>, VsaError> {
        self.check(a)?;
        self.check(b)?;
        tally(&mut meter.binds, self.dim);
        Ok(a.iter().zip(b).map(|(x, y)| if x == y { 0.0 } else { 1.0 }).collect())
    }

    /// The majority bit per position, ties toward `1`.
    ///
    /// # Errors
    ///
    /// [`VsaError::Empty`] for no members, plus [`Binary::check`] on each.
    pub fn bundle(&self, members: &[Vec<f64>], meter: &mut Meter) -> Result<Vec<f64>, VsaError> {
        non_empty(members.len(), "bundle")?;
        let mut ones = vec![0usize; self.dim];
        for m in members {
            self.check(m)?;
            for (c, &x) in ones.iter_mut().zip(m) {
                if x == 1.0 {
                    *c += 1;
                }
            }
        }
        tally(&mut meter.bundles, self.dim * (members.len() + 1));
        let half = members.len();
        Ok(ones.iter().map(|&c| if 2 * c >= half { 1.0 } else { 0.0 }).collect())
    }

    /// Cyclic shift right by `k` (left for negative `k`).
    ///
    /// # Errors
    ///
    /// As [`Binary::check`].
    pub fn permute(&self, a: &[f64], k: i64, meter: &mut Meter) -> Result<Vec<f64>, VsaError> {
        self.check(a)?;
        tally(&mut meter.permutes, self.dim);
        Ok(rotate(a, k))
    }

    /// Hamming distance: positions where the two differ.
    ///
    /// # Errors
    ///
    /// As [`Binary::check`], on either argument.
    pub fn hamming(&self, a: &[f64], b: &[f64], meter: &mut Meter) -> Result<usize, VsaError> {
        self.check(a)?;
        self.check(b)?;
        tally(&mut meter.similarities, self.dim);
        Ok(a.iter().zip(b).filter(|(x, y)| x != y).count())
    }

    /// `1 − 2·Hamming / D`, in `[−1, 1]`: the cosine of the bipolar image.
    ///
    /// # Errors
    ///
    /// As [`Binary::hamming`].
    pub fn similarity(&self, a: &[f64], b: &[f64], meter: &mut Meter) -> Result<f64, VsaError> {
        let h = self.hamming(a, b, meter)?;
        Ok(1.0 - 2.0 * h as f64 / self.dim as f64)
    }

    /// The bipolar image `1 − 2x`: a `0` becomes `+1`, a `1` becomes `−1`.
    ///
    /// # Errors
    ///
    /// As [`Binary::check`].
    pub fn to_bipolar(&self, a: &[f64]) -> Result<Vec<f64>, VsaError> {
        self.check(a)?;
        Ok(a.iter().map(|&x| 1.0 - 2.0 * x).collect())
    }

    /// The inverse of [`Binary::to_bipolar`].
    ///
    /// # Errors
    ///
    /// [`Bipolar::check`]'s refusals, on a model of the same dimension.
    pub fn from_bipolar(&self, a: &[f64]) -> Result<Vec<f64>, VsaError> {
        Bipolar { dim: self.dim }.check(a)?;
        Ok(a.iter().map(|&x| (1.0 - x) / 2.0).collect())
    }
}

// ---------------------------------------------------------------------------------------------
// HRR: holographic reduced representations
// ---------------------------------------------------------------------------------------------

/// Holographic reduced representations (Plate, 1995): real vectors with circular convolution as
/// binding.
///
/// Symbols are drawn from `N(0, 1/D)` per element so that their expected squared norm is `1`, which
/// is what makes the dot product a similarity without a division. Binding is the circular
/// convolution `(a ⊛ b)[k] = Σ_i a[i]·b[(k − i) mod D]`, computed directly in `O(D²)` — no FFT,
/// because this crate has no dependencies and the sizes a test or a classroom uses do not need one.
/// A `D = 1024` bind is a million multiply-adds.
///
/// Unbinding uses the involution `a*[i] = a[(−i) mod D]`, the approximate inverse:
/// `(a ⊛ b) ⊛ b* ≈ a`, with noise of order `1/√D` per element. The exact inverse exists only in
/// the frequency domain and can be badly conditioned, which is why Plate recommends the involution
/// and why this module offers nothing else.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Hrr {
    /// Dimension `D`. At least one.
    pub dim: usize,
}

impl Hrr {
    /// A model of dimension `dim`.
    ///
    /// # Errors
    ///
    /// [`VsaError::Empty`] for a zero dimension.
    pub fn new(dim: usize) -> Result<Self, VsaError> {
        non_empty(dim, "dimension")?;
        Ok(Self { dim })
    }

    /// A random symbol, `N(0, 1/D)` per element by Box-Muller on the crate's generator.
    #[must_use]
    pub fn random(&self, rng: &mut Rng) -> Vec<f64> {
        let sigma = 1.0 / (self.dim as f64).sqrt();
        let mut v = Vec::with_capacity(self.dim);
        while v.len() < self.dim {
            let u1 = rng.next_f64().max(1e-300);
            let u2 = rng.next_f64();
            let r = (-2.0 * u1.ln()).sqrt();
            let (s, c) = (core::f64::consts::TAU * u2).sin_cos();
            v.push(sigma * r * c);
            if v.len() < self.dim {
                v.push(sigma * r * s);
            }
        }
        v
    }

    /// Check that `v` is a finite vector of this dimension.
    ///
    /// # Errors
    ///
    /// [`VsaError::Dimension`], [`VsaError::NonFinite`].
    pub fn check(&self, v: &[f64]) -> Result<(), VsaError> {
        same_dim(v.len(), self.dim)?;
        if let Some(i) = v.iter().position(|x| !x.is_finite()) {
            return Err(VsaError::NonFinite { index: i });
        }
        Ok(())
    }

    /// `a ⊛ b`, circular convolution. Commutative and associative; the identity is the unit
    /// impulse `[1, 0, …, 0]`.
    ///
    /// # Errors
    ///
    /// As [`Hrr::check`], on either argument.
    pub fn bind(&self, a: &[f64], b: &[f64], meter: &mut Meter) -> Result<Vec<f64>, VsaError> {
        self.check(a)?;
        self.check(b)?;
        let d = self.dim;
        tally(&mut meter.binds, d * d);
        let mut out = vec![0.0; d];
        for (k, o) in out.iter_mut().enumerate() {
            let mut acc = 0.0;
            for (i, &ai) in a.iter().enumerate() {
                let j = if k >= i { k - i } else { k + d - i };
                acc += ai * b[j];
            }
            *o = acc;
        }
        Ok(out)
    }

    /// The involution `a*[i] = a[(−i) mod D]`, Plate's approximate inverse.
    ///
    /// # Errors
    ///
    /// As [`Hrr::check`].
    pub fn involution(&self, a: &[f64]) -> Result<Vec<f64>, VsaError> {
        self.check(a)?;
        let d = self.dim;
        Ok((0..d).map(|i| a[(d - i) % d]).collect())
    }

    /// `(a ⊛ b) ⊛ b*`: recover `a` from a bound pair, approximately.
    ///
    /// # Errors
    ///
    /// As [`Hrr::bind`].
    pub fn unbind(&self, ab: &[f64], b: &[f64], meter: &mut Meter) -> Result<Vec<f64>, VsaError> {
        let inv = self.involution(b)?;
        // The involution is a PERMUTATION — `a[(d - i) % d]` — and it moves `D` elements. Every
        // other element move in this module is billed (`Bipolar::permute`, `Binary::permute`,
        // `Hrr::permute` all `tally(.., self.dim)`), and this one was not, so an unbind was
        // charged the bind's `D²` and nothing for the `D` moves it also does. An under-count of
        // `D` out of `D² + D` is 0.4% at `D = 256` — small, systematic, and in the direction that
        // flatters the operation. `Hrr::involution` itself stays unmetered because it is a public
        // helper a caller may use on its own; `unbind` is the metered entry point, so it bills.
        tally(&mut meter.permutes, self.dim);
        self.bind(ab, &inv, meter)
    }

    /// Element-wise sum, optionally rescaled to unit norm.
    ///
    /// # Errors
    ///
    /// [`VsaError::Empty`] for no members, plus [`Hrr::check`] on each.
    pub fn bundle(
        &self,
        members: &[Vec<f64>],
        normalise: bool,
        meter: &mut Meter,
    ) -> Result<Vec<f64>, VsaError> {
        non_empty(members.len(), "bundle")?;
        let mut sum = vec![0.0f64; self.dim];
        for m in members {
            self.check(m)?;
            for (s, &x) in sum.iter_mut().zip(m) {
                *s += x;
            }
        }
        tally(&mut meter.bundles, self.dim * (members.len() + usize::from(normalise)));
        if normalise {
            let n = sum.iter().map(|x| x * x).sum::<f64>().sqrt();
            if n > 0.0 {
                for s in &mut sum {
                    *s /= n;
                }
            }
        }
        Ok(sum)
    }

    /// Cyclic shift right by `k` (left for negative `k`).
    ///
    /// # Errors
    ///
    /// As [`Hrr::check`].
    pub fn permute(&self, a: &[f64], k: i64, meter: &mut Meter) -> Result<Vec<f64>, VsaError> {
        self.check(a)?;
        tally(&mut meter.permutes, self.dim);
        Ok(rotate(a, k))
    }

    /// The dot product, which is the cosine for unit-norm symbols.
    ///
    /// # Errors
    ///
    /// As [`Hrr::check`], on either argument.
    pub fn similarity(&self, a: &[f64], b: &[f64], meter: &mut Meter) -> Result<f64, VsaError> {
        self.check(a)?;
        self.check(b)?;
        tally(&mut meter.similarities, self.dim);
        Ok(a.iter().zip(b).map(|(x, y)| x * y).sum())
    }

    /// The exact cosine, `dot / (|a|·|b|)`, for vectors that are not unit norm. `None` if either
    /// norm is zero.
    ///
    /// # Errors
    ///
    /// As [`Hrr::check`], on either argument.
    pub fn cosine(&self, a: &[f64], b: &[f64], meter: &mut Meter) -> Result<Option<f64>, VsaError> {
        let dot = self.similarity(a, b, meter)?;
        let na = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let nb = b.iter().map(|x| x * x).sum::<f64>().sqrt();
        if na == 0.0 || nb == 0.0 {
            return Ok(None);
        }
        Ok(Some(dot / (na * nb)))
    }
}

/// Cyclic shift of `a` right by `k` positions (left for negative `k`), for any model.
fn rotate(a: &[f64], k: i64) -> Vec<f64> {
    let d = a.len() as i64;
    let shift = k.rem_euclid(d) as usize;
    let mut out = a.to_vec();
    out.rotate_right(shift);
    out
}

// ---------------------------------------------------------------------------------------------
// Closed forms
// ---------------------------------------------------------------------------------------------

/// The expected cosine between a majority bundle of `k` random bipolar vectors and any one of its
/// members, for odd `k`: `C(k−1, (k−1)/2) / 2^(k−1)`.
///
/// Derivation: at each position the member is `+1` without loss of generality and the other
/// `k − 1` members are fair coins; the majority agrees with the member unless strictly more than
/// half of the others disagree. `P(agree) = 1/2 + P(exactly (k−1)/2 of the others agree)/2`, the
/// second term being the tie-at-the-member case that the member itself breaks, and the cosine is
/// `2·P(agree) − 1 = C(k−1, (k−1)/2) / 2^(k−1)`. Three members: `2/4 = 0.5`. Five: `6/16 = 0.375`.
/// Seven: `20/64 = 0.3125`. Nine: `70/256 = 0.2734`. It falls as `√(2/(πk))`.
///
/// Also the expected `1 − 2·Hamming/D` for the binary spatter code, by the isomorphism.
///
/// `None` for an even `k` (a tie rule enters) or `k = 0`.
#[must_use]
pub fn bundle_similarity(k: usize) -> Option<f64> {
    if k == 0 || k.is_multiple_of(2) {
        return None;
    }
    let n = k - 1;
    let half = n / 2;
    // C(n, half) / 2^n, accumulated as a product of ratios so that it never overflows.
    let mut p = 1.0f64;
    for i in 0..half {
        p *= (n - i) as f64 / (i + 1) as f64;
    }
    Some(p / 2f64.powi(n as i32))
}

/// The standard deviation of the cosine between two independent random symbols: `1/√D` for the
/// discrete models (a sum of `D` fair `±1` products), and the same to leading order for HRR.
#[must_use]
pub fn noise_sd(dim: usize) -> f64 {
    if dim == 0 { f64::NAN } else { 1.0 / (dim as f64).sqrt() }
}

/// The dimension at which a member of a `k`-bundle is separated from the nearest of `m` distractors
/// by `z` standard deviations of the noise: `D = (z / bundle_similarity(k))²`, with the largest of
/// `m` noise cosines approximated by `z_m·noise_sd` where `z_m = √(2·ln m)`.
///
/// Returns `None` for an even or zero `k`, or `m < 2`. A rule of thumb with the derivation shown,
/// not a guarantee: the extreme-value approximation `√(2 ln m)` is the leading term.
#[must_use]
pub fn dimension_for(k: usize, m: usize, z: f64) -> Option<usize> {
    if m < 2 || !(z > 0.0) {
        return None;
    }
    let s = bundle_similarity(k)?;
    let z_m = (2.0 * (m as f64).ln()).sqrt();
    let d = ((z + z_m) / s).powi(2);
    Some(d.ceil() as usize)
}

// ---------------------------------------------------------------------------------------------
// Codebooks and cleanup
// ---------------------------------------------------------------------------------------------

/// A set of named symbols, and the nearest-neighbour readout every VSA computation ends with.
///
/// Storing `M` symbols of `D` elements is `M·D` values; one readout is `M·D` multiply-accumulates.
/// That product is the whole cost of a symbolic query on a neuromorphic substrate, and it is what
/// [`Meter::similarities`] accumulates when [`Codebook::nearest`] runs.
#[derive(Debug, Clone, PartialEq)]
pub struct Codebook {
    /// The symbols, in the order they were added.
    pub symbols: Vec<Vec<f64>>,
    /// Dimension every symbol has.
    pub dim: usize,
}

impl Codebook {
    /// `m` random symbols of the bipolar model.
    ///
    /// # Errors
    ///
    /// [`VsaError::Empty`] for `m = 0`.
    pub fn random_bipolar(model: &Bipolar, m: usize, rng: &mut Rng) -> Result<Self, VsaError> {
        non_empty(m, "codebook")?;
        Ok(Self { symbols: (0..m).map(|_| model.random(rng)).collect(), dim: model.dim })
    }

    /// `m` random symbols of the binary model.
    ///
    /// # Errors
    ///
    /// [`VsaError::Empty`] for `m = 0`.
    pub fn random_binary(model: &Binary, m: usize, rng: &mut Rng) -> Result<Self, VsaError> {
        non_empty(m, "codebook")?;
        Ok(Self { symbols: (0..m).map(|_| model.random(rng)).collect(), dim: model.dim })
    }

    /// `m` random symbols of the holographic model.
    ///
    /// # Errors
    ///
    /// [`VsaError::Empty`] for `m = 0`.
    pub fn random_hrr(model: &Hrr, m: usize, rng: &mut Rng) -> Result<Self, VsaError> {
        non_empty(m, "codebook")?;
        Ok(Self { symbols: (0..m).map(|_| model.random(rng)).collect(), dim: model.dim })
    }

    /// How many symbols.
    #[must_use]
    pub fn len(&self) -> usize {
        self.symbols.len()
    }

    /// Whether there are none — never, for a codebook built by this module.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.symbols.is_empty()
    }

    /// The index and dot-product similarity of the symbol nearest to `query`, over every symbol.
    ///
    /// Uses the plain dot product divided by `D`, which is the cosine for every model's symbols
    /// and a monotone proxy for it on an un-normalised query (a bundle sum), since every symbol
    /// has the same norm. Ties go to the lower index.
    ///
    /// # Errors
    ///
    /// [`VsaError::Dimension`] for a query of the wrong length, [`VsaError::NonFinite`] for one that
    /// is not finite.
    pub fn nearest(&self, query: &[f64], meter: &mut Meter) -> Result<(usize, f64), VsaError> {
        same_dim(query.len(), self.dim)?;
        if let Some(i) = query.iter().position(|x| !x.is_finite()) {
            return Err(VsaError::NonFinite { index: i });
        }
        tally(&mut meter.similarities, self.dim * self.symbols.len());
        let mut best = (0usize, f64::NEG_INFINITY);
        for (i, s) in self.symbols.iter().enumerate() {
            let dot = s.iter().zip(query).map(|(x, y)| x * y).sum::<f64>() / self.dim as f64;
            if dot > best.1 {
                best = (i, dot);
            }
        }
        Ok(best)
    }

    /// Every symbol's similarity to `query`, in codebook order.
    ///
    /// # Errors
    ///
    /// As [`Codebook::nearest`].
    pub fn similarities(&self, query: &[f64], meter: &mut Meter) -> Result<Vec<f64>, VsaError> {
        same_dim(query.len(), self.dim)?;
        if let Some(i) = query.iter().position(|x| !x.is_finite()) {
            return Err(VsaError::NonFinite { index: i });
        }
        tally(&mut meter.similarities, self.dim * self.symbols.len());
        Ok(self
            .symbols
            .iter()
            .map(|s| s.iter().zip(query).map(|(x, y)| x * y).sum::<f64>() / self.dim as f64)
            .collect())
    }
}

// ---------------------------------------------------------------------------------------------
// Sequences by permutation
// ---------------------------------------------------------------------------------------------

/// Encode a sequence of bipolar symbols as `ρ^(n−1)(s_0) + ρ^(n−2)(s_1) + … + s_{n−1}`: earlier
/// items are shifted further. The result is a bundle, so it is similar to each shifted item and
/// each is recovered by shifting the whole back and reading the nearest symbol.
///
/// # Errors
///
/// [`VsaError::Empty`] for an empty sequence, plus [`Bipolar::bundle`]'s refusals.
pub fn encode_sequence(
    model: &Bipolar,
    items: &[Vec<f64>],
    meter: &mut Meter,
) -> Result<Vec<f64>, VsaError> {
    non_empty(items.len(), "sequence")?;
    let n = items.len();
    let mut shifted = Vec::with_capacity(n);
    for (i, item) in items.iter().enumerate() {
        shifted.push(model.permute(item, (n - 1 - i) as i64, meter)?);
    }
    model.bundle(&shifted, meter)
}

/// Read item `position` of a sequence of length `n` encoded by [`encode_sequence`] back out of
/// `code`, as the nearest codebook symbol and its similarity.
///
/// # Errors
///
/// As [`Bipolar::permute`] and [`Codebook::nearest`].
pub fn decode_sequence_item(
    model: &Bipolar,
    code: &[f64],
    n: usize,
    position: usize,
    book: &Codebook,
    meter: &mut Meter,
) -> Result<(usize, f64), VsaError> {
    let back = model.permute(code, -((n - 1 - position) as i64), meter)?;
    book.nearest(&back, meter)
}

// ---------------------------------------------------------------------------------------------
// Resonator networks
// ---------------------------------------------------------------------------------------------

/// A resonator network (Frady, Kent, Olshausen and Sommer, 2020): factor a bipolar composite
/// `s = a ⊗ b ⊗ c` into one symbol from each of several codebooks.
///
/// The naive search over `F` codebooks of `M` symbols is `M^F` binds. The resonator instead keeps
/// an estimate per factor, and updates each estimate by unbinding the others' estimates from `s`
/// and projecting onto its codebook:
///
/// ```text
/// â ← sign( X_A · X_Aᵀ · (s ⊗ b̂ ⊗ ĉ) )
/// ```
///
/// where `X_A` is the codebook matrix. Each estimate starts as the bundle of its whole codebook —
/// every hypothesis superposed — and the superposition collapses onto the answer, typically in a
/// few iterations, because a wrong estimate cancels in the product and the right one reinforces.
/// The paper's finding is that the search capacity scales far better than the `M^F` it replaces;
/// the test here factors `20³ = 8,000` candidate triples in `D = 1,000` from every one of twenty
/// random composites, and reports the iteration count rather than asserting a number for it.
///
/// Convergence is detected as a fixed point (no estimate changed) and bounded by `max_iters`,
/// after which the current estimates are returned with `converged = false` — a resonator can
/// oscillate between two states, and that is reported rather than looped on forever.
///
/// # What this implementation measures, stated rather than assumed
///
/// A resonator can also converge to a **wrong** fixed point — a stable superposition that is not
/// a codebook triple — and it does so more often than the paper's accuracy curves suggest at
/// comparable sizes. Measured here, three codebooks of `M` symbols, forty random composites each,
/// correct factorings out of forty:
///
/// | `D` | `M` | synchronous | sequential |
/// |---|---|---|---|
/// | 1000 | 20 | 14 | 26 |
/// | 2000 | 20 | 32 | 36 |
/// | 4000 | 20 | 32 | 36 |
/// | 1000 | 10 | 32 | 36 |
/// | 1000 | 40 | 7 | 21 |
///
/// The sequential update — each factor reads the others' freshest estimates — is the better of
/// the two on every row and is the default; the paper's synchronous form is the `synchronous`
/// field. A second draw of codebooks at `D = 4000, M = 20` factored forty of forty, so the rate
/// also depends on the instance. This review did not locate the cause of the gap to the paper's
/// reported capacity; what is claimed is what the test asserts, and every failure carries a
/// [`Factoring::fidelity`] near zero so a caller can tell.
#[derive(Debug, Clone, PartialEq)]
pub struct Resonator {
    /// The model.
    pub model: Bipolar,
    /// One codebook per factor.
    pub books: Vec<Codebook>,
    /// Iterations to try before reporting non-convergence.
    pub max_iters: usize,
    /// Update every estimate from the previous iteration's estimates (the paper's form, `true`)
    /// or from the freshest ones in sequence (`false`, the default: measured better on every row
    /// of the table in the struct doc).
    pub synchronous: bool,
}

/// What a [`Resonator`] found.
#[derive(Debug, Clone, PartialEq)]
pub struct Factoring {
    /// The chosen symbol index in each codebook.
    pub indices: Vec<usize>,
    /// Iterations run, including the one that detected the fixed point.
    pub iterations: usize,
    /// Whether a fixed point was reached inside `max_iters`.
    pub converged: bool,
    /// Cosine between the composite rebuilt from `indices` and the input: `1.0` for an exact
    /// factoring, and the number to read when `converged` is false.
    pub fidelity: f64,
}

impl Resonator {
    /// Build over `books`, all of one dimension.
    ///
    /// # Errors
    ///
    /// [`VsaError::Empty`] with no codebooks or an empty one, [`VsaError::Dimension`] if they
    /// disagree with the model.
    pub fn new(model: Bipolar, books: Vec<Codebook>, max_iters: usize) -> Result<Self, VsaError> {
        non_empty(books.len(), "codebook")?;
        for b in &books {
            non_empty(b.len(), "codebook")?;
            same_dim(b.dim, model.dim)?;
            for s in &b.symbols {
                model.check(s)?;
            }
        }
        Ok(Self { model, books, max_iters: max_iters.max(1), synchronous: false })
    }

    /// Factor `composite`.
    ///
    /// # Errors
    ///
    /// As [`Bipolar::check`] on the composite.
    pub fn factor(&self, composite: &[f64], meter: &mut Meter) -> Result<Factoring, VsaError> {
        self.model.check(composite)?;
        let f = self.books.len();
        let d = self.model.dim;
        // Initial estimates: the superposition of every hypothesis in each codebook.
        let mut estimates: Vec<Vec<f64>> = Vec::with_capacity(f);
        for b in &self.books {
            estimates.push(self.model.bundle(&b.symbols, meter)?);
        }
        let mut iterations = 0;
        let mut converged = false;
        while iterations < self.max_iters {
            iterations += 1;
            let mut changed = false;
            let previous = estimates.clone();
            for i in 0..f {
                // Unbind every other factor's estimate from the composite.
                let source = if self.synchronous { &previous } else { &estimates };
                let mut residual = composite.to_vec();
                for (j, e) in source.iter().enumerate() {
                    if j != i {
                        residual = self.model.bind(&residual, e, meter)?;
                    }
                }
                // Project onto codebook i: X Xᵀ r, then sign.
                let sims = self.books[i].similarities(&residual, meter)?;
                let mut proj = vec![0.0f64; d];
                for (s, sym) in sims.iter().zip(&self.books[i].symbols) {
                    for (p, &x) in proj.iter_mut().zip(sym) {
                        *p += s * x;
                    }
                }
                tally(&mut meter.bundles, d * self.books[i].len());
                let next: Vec<f64> = proj.iter().map(|&p| if p >= 0.0 { 1.0 } else { -1.0 }).collect();
                if next != estimates[i] {
                    changed = true;
                    estimates[i] = next;
                }
            }
            if !changed {
                converged = true;
                break;
            }
        }
        let mut indices = Vec::with_capacity(f);
        let mut rebuilt = vec![1.0f64; d];
        for (i, e) in estimates.iter().enumerate() {
            let (idx, _) = self.books[i].nearest(e, meter)?;
            indices.push(idx);
            rebuilt = self.model.bind(&rebuilt, &self.books[i].symbols[idx], meter)?;
        }
        let fidelity = self.model.similarity(&rebuilt, composite, meter)?;
        Ok(Factoring { indices, iterations, converged, fidelity })
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Binary, Bipolar, Codebook, Hrr, Meter, Resonator, VsaError, bundle_similarity,
        decode_sequence_item, dimension_for, encode_sequence, noise_sd,
    };
    use crate::rng::Rng;

    fn mean_sd(xs: &[f64]) -> (f64, f64) {
        let n = xs.len() as f64;
        let m = xs.iter().sum::<f64>() / n;
        let v = xs.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (n - 1.0);
        (m, v.sqrt())
    }

    // ---- exactness of the discrete algebra ----

    /// Binding is self-inverse BIT FOR BIT in both discrete models, and a permutation is exactly
    /// undone by its opposite. No tolerance anywhere.
    #[test]
    fn the_discrete_bindings_and_permutations_are_exactly_invertible() {
        let mut rng = Rng::new(1);
        let mut meter = Meter::default();
        let bp = Bipolar::new(1000).unwrap();
        let (a, b) = (bp.random(&mut rng), bp.random(&mut rng));
        let ab = bp.bind(&a, &b, &mut meter).unwrap();
        assert_eq!(bp.bind(&ab, &b, &mut meter).unwrap(), a);
        assert_eq!(bp.bind(&ab, &a, &mut meter).unwrap(), b);
        assert_ne!(ab, a);
        assert_eq!(bp.permute(&bp.permute(&a, 7, &mut meter).unwrap(), -7, &mut meter).unwrap(), a);
        assert_eq!(bp.permute(&a, 1000, &mut meter).unwrap(), a, "a full turn is the identity");
        assert_eq!(bp.permute(&a, 1, &mut meter).unwrap()[1], a[0], "shift right moves index 0 to 1");

        let bn = Binary::new(1000).unwrap();
        let (x, y) = (bn.random(&mut rng), bn.random(&mut rng));
        let xy = bn.bind(&x, &y, &mut meter).unwrap();
        assert_eq!(bn.bind(&xy, &y, &mut meter).unwrap(), x);
        assert_eq!(bn.hamming(&x, &x, &mut meter).unwrap(), 0);
        assert_eq!(bn.similarity(&x, &x, &mut meter).unwrap(), 1.0);
        assert_eq!(bp.similarity(&a, &a, &mut meter).unwrap(), 1.0);
        assert_eq!(bp.similarity(&a, &bp.bind(&a, &vec![-1.0; 1000], &mut meter).unwrap(), &mut meter).unwrap(), -1.0);
    }

    /// The bipolar/binary isomorphism `x ↦ 1 − 2x`: XOR maps to the product exactly, and
    /// `1 − 2·Hamming/D` maps to the cosine exactly.
    #[test]
    fn the_binary_model_is_the_bipolar_model_under_one_minus_two_x() {
        let mut rng = Rng::new(2);
        let mut meter = Meter::default();
        let bn = Binary::new(512).unwrap();
        let bp = Bipolar::new(512).unwrap();
        let (x, y) = (bn.random(&mut rng), bn.random(&mut rng));
        let (a, b) = (bn.to_bipolar(&x).unwrap(), bn.to_bipolar(&y).unwrap());
        assert_eq!(bn.to_bipolar(&bn.bind(&x, &y, &mut meter).unwrap()).unwrap(), bp.bind(&a, &b, &mut meter).unwrap());
        assert_eq!(bn.similarity(&x, &y, &mut meter).unwrap(), bp.similarity(&a, &b, &mut meter).unwrap());
        assert_eq!(bn.from_bipolar(&a).unwrap(), x);
        // And the majority bundle maps too, tie rule included: three members, no ties.
        let z = bn.random(&mut rng);
        let c = bn.to_bipolar(&z).unwrap();
        assert_eq!(
            bn.to_bipolar(&bn.bundle(&[x.clone(), y.clone(), z.clone()], &mut meter).unwrap()).unwrap(),
            bp.bundle(&[a, b, c], &mut meter).unwrap()
        );
    }

    // ---- the closed forms ----

    /// `bundle_similarity` against the literals derived in its doc, and against a MEASURED bundle
    /// at D = 10,000 where the standard error of the measurement is about 0.01.
    #[test]
    fn a_bundle_is_as_similar_to_each_member_as_the_binomial_says() {
        assert_eq!(bundle_similarity(1), Some(1.0));
        assert_eq!(bundle_similarity(3), Some(0.5));
        assert_eq!(bundle_similarity(5), Some(0.375));
        assert_eq!(bundle_similarity(7), Some(0.3125));
        assert!((bundle_similarity(9).unwrap() - 70.0 / 256.0).abs() < 1e-15);
        assert_eq!(bundle_similarity(0), None);
        assert_eq!(bundle_similarity(4), None, "even k needs a tie rule; no closed form is offered");
        // The large-k limit sqrt(2/(pi k)) is approached from above.
        let k = 101;
        let want = (2.0 / (core::f64::consts::PI * k as f64)).sqrt();
        let got = bundle_similarity(k).unwrap();
        assert!(got > want && (got / want - 1.0) < 0.01, "{got} vs {want}");

        let mut rng = Rng::new(3);
        let mut meter = Meter::default();
        let bp = Bipolar::new(10_000).unwrap();
        for k in [3usize, 5, 7, 9] {
            let members: Vec<Vec<f64>> = (0..k).map(|_| bp.random(&mut rng)).collect();
            let bundle = bp.bundle(&members, &mut meter).unwrap();
            let sims: Vec<f64> = members.iter().map(|m| bp.similarity(&bundle, m, &mut meter).unwrap()).collect();
            let (mean, _) = mean_sd(&sims);
            let want = bundle_similarity(k).unwrap();
            // Each similarity is a mean of 10,000 ±1 terms with p(agree) = (1 + want)/2, so its sd
            // is 2·sqrt(p(1−p)/D) ≈ 0.01; the mean over k members is tighter still. 0.03 is 3+ sd.
            assert!((mean - want).abs() < 0.03, "k = {k}: measured {mean}, closed form {want}");
            // And a non-member is at noise level: 1/sqrt(D) = 0.01.
            let stranger = bp.random(&mut rng);
            let s = bp.similarity(&bundle, &stranger, &mut meter).unwrap();
            assert!(s.abs() < 5.0 * noise_sd(10_000), "stranger at {s}");
        }
        // The binary model, by the isomorphism, at the same closed form.
        let bn = Binary::new(10_000).unwrap();
        let members: Vec<Vec<f64>> = (0..7).map(|_| bn.random(&mut rng)).collect();
        let bundle = bn.bundle(&members, &mut meter).unwrap();
        let sims: Vec<f64> = members.iter().map(|m| bn.similarity(&bundle, m, &mut meter).unwrap()).collect();
        assert!((mean_sd(&sims).0 - 0.3125).abs() < 0.03);
    }

    /// Two random symbols have a cosine of mean zero and standard deviation `1/√D`, measured over
    /// a thousand pairs at three dimensions — the noise floor every capacity argument divides by.
    #[test]
    fn strangers_are_orthogonal_to_one_over_root_d() {
        let mut rng = Rng::new(4);
        let mut meter = Meter::default();
        for d in [100usize, 1_000, 10_000] {
            let bp = Bipolar::new(d).unwrap();
            let sims: Vec<f64> =
                (0..1000).map(|_| bp.similarity(&bp.random(&mut rng), &bp.random(&mut rng), &mut meter).unwrap()).collect();
            let (mean, sd) = mean_sd(&sims);
            assert!(mean.abs() < 4.0 * noise_sd(d) / (1000f64).sqrt() * 3.0 + 1e-3, "d {d}: mean {mean}");
            assert!((sd / noise_sd(d) - 1.0).abs() < 0.1, "d {d}: sd {sd} vs 1/sqrt(d) = {}", noise_sd(d));
        }
        // HRR symbols: unit norm in expectation, cosine noise at the same order.
        let hrr = Hrr::new(1000).unwrap();
        let norms: Vec<f64> = (0..200).map(|_| hrr.random(&mut rng).iter().map(|x| x * x).sum::<f64>()).collect();
        assert!((mean_sd(&norms).0 - 1.0).abs() < 0.02, "squared norm {}", mean_sd(&norms).0);
        let sims: Vec<f64> =
            (0..500).map(|_| hrr.similarity(&hrr.random(&mut rng), &hrr.random(&mut rng), &mut meter).unwrap()).collect();
        assert!((mean_sd(&sims).1 / noise_sd(1000) - 1.0).abs() < 0.15);
    }

    /// The tie rule is stated and pinned: an even bundle of two opposite vectors is all `+1`.
    #[test]
    fn an_even_tie_breaks_toward_plus_one_deterministically() {
        let mut meter = Meter::default();
        let bp = Bipolar::new(8).unwrap();
        let a = vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
        let b: Vec<f64> = a.iter().map(|x| -x).collect();
        assert_eq!(bp.bundle(&[a.clone(), b.clone()], &mut meter).unwrap(), vec![1.0; 8]);
        let bn = Binary::new(4).unwrap();
        assert_eq!(bn.bundle(&[vec![0., 0., 1., 1.], vec![1., 1., 0., 0.]], &mut meter).unwrap(), vec![1.0; 4]);
        // A one-member bundle is the member.
        assert_eq!(bp.bundle(std::slice::from_ref(&a), &mut meter).unwrap(), a);
    }

    // ---- HRR ----

    /// Circular convolution: commutative and associative to rounding, the unit impulse is the
    /// identity EXACTLY, and the involution unbinds to within the `1/√D` noise the doc states.
    #[test]
    fn circular_convolution_has_the_algebra_plate_describes() {
        let mut rng = Rng::new(5);
        let mut meter = Meter::default();
        let hrr = Hrr::new(256).unwrap();
        let (a, b, c) = (hrr.random(&mut rng), hrr.random(&mut rng), hrr.random(&mut rng));
        let ab = hrr.bind(&a, &b, &mut meter).unwrap();
        let ba = hrr.bind(&b, &a, &mut meter).unwrap();
        for (x, y) in ab.iter().zip(&ba) {
            assert!((x - y).abs() < 1e-15);
        }
        let abc = hrr.bind(&ab, &c, &mut meter).unwrap();
        let a_bc = hrr.bind(&a, &hrr.bind(&b, &c, &mut meter).unwrap(), &mut meter).unwrap();
        for (x, y) in abc.iter().zip(&a_bc) {
            assert!((x - y).abs() < 1e-14);
        }
        let mut delta = vec![0.0; 256];
        delta[0] = 1.0;
        assert_eq!(hrr.bind(&a, &delta, &mut meter).unwrap(), a, "the unit impulse is the identity");
        // Unbinding: (a ⊛ b) ⊛ b* ≈ a. The recovered vector's cosine with a is high, its cosine
        // with a stranger is at noise level, and the bound pair is dissimilar to both inputs.
        let rec = hrr.unbind(&ab, &b, &mut meter).unwrap();
        let cos = hrr.cosine(&rec, &a, &mut meter).unwrap().unwrap();
        assert!(cos > 0.6, "unbinding recovered a at cosine {cos}");
        let stranger = hrr.random(&mut rng);
        assert!(hrr.cosine(&rec, &stranger, &mut meter).unwrap().unwrap().abs() < 0.25);
        assert!(hrr.cosine(&ab, &a, &mut meter).unwrap().unwrap().abs() < 0.25, "a bound pair resembles neither input");
        // The involution is its own inverse.
        assert_eq!(hrr.involution(&hrr.involution(&a).unwrap()).unwrap(), a);
        assert_eq!(hrr.cosine(&vec![0.0; 256], &a, &mut meter).unwrap(), None);
    }

    // ---- codebooks, sequences, retrieval ----

    /// Retrieval from a bundle across a codebook: exact at a dimension the closed form says is
    /// safe, and failing at one it says is not — the same test that would pass at every dimension
    /// is not a capacity test.
    #[test]
    fn retrieval_from_a_bundle_succeeds_where_the_dimension_rule_says_and_fails_below_it() {
        let mut rng = Rng::new(6);
        let mut meter = Meter::default();
        let (k, m) = (7usize, 1000usize);
        let safe = dimension_for(k, m, 4.0).unwrap();
        // ((4 + sqrt(2 ln 1000)) / 0.3125)² = 610.
        assert_eq!(safe, 610, "rule of thumb gave D = {safe}");
        let mut run = |d: usize| -> f64 {
            let bp = Bipolar::new(d).unwrap();
            let book = Codebook::random_bipolar(&bp, m, &mut rng).unwrap();
            let mut correct = 0usize;
            let mut total = 0usize;
            for trial in 0..10 {
                let chosen: Vec<usize> = (0..k).map(|i| (trial * 97 + i * 131) % m).collect();
                let members: Vec<Vec<f64>> = chosen.iter().map(|&i| book.symbols[i].clone()).collect();
                let bundle = bp.bundle(&members, &mut meter).unwrap();
                for &i in &chosen {
                    // Query with the bundle: the nearest symbol should be each member in turn only
                    // if we remove the others, so instead read all k as the top-k similarities.
                    let sims = book.similarities(&bundle, &mut meter).unwrap();
                    let mut order: Vec<usize> = (0..m).collect();
                    order.sort_by(|&a, &b| sims[b].partial_cmp(&sims[a]).unwrap());
                    if order[..k].contains(&i) {
                        correct += 1;
                    }
                    total += 1;
                }
            }
            correct as f64 / total as f64
        };
        assert_eq!(run(safe), 1.0, "at the rule's dimension every member is in the top k");
        let low = run(64);
        assert!(low < 1.0, "at D = 64 with 1000 distractors retrieval cannot be perfect: {low}");
        assert_eq!(dimension_for(4, 10, 2.0), None);
        assert_eq!(dimension_for(3, 1, 2.0), None);
    }

    /// A sequence encoded by permutation reads back exactly at every position.
    #[test]
    fn a_permuted_sequence_reads_back_in_order() {
        let mut rng = Rng::new(7);
        let mut meter = Meter::default();
        let bp = Bipolar::new(2000).unwrap();
        let book = Codebook::random_bipolar(&bp, 50, &mut rng).unwrap();
        let order = [3usize, 41, 17, 3, 8];
        let items: Vec<Vec<f64>> = order.iter().map(|&i| book.symbols[i].clone()).collect();
        let code = encode_sequence(&bp, &items, &mut meter).unwrap();
        for (pos, &want) in order.iter().enumerate() {
            let (got, sim) = decode_sequence_item(&bp, &code, order.len(), pos, &book, &mut meter).unwrap();
            assert_eq!(got, want, "position {pos}");
            assert!(sim > 0.2, "position {pos} read back at similarity {sim}");
        }
        // Reading a position with the wrong shift lands on nothing in particular.
        let (_, wrong) = decode_sequence_item(&bp, &code, order.len() + 3, 0, &book, &mut meter).unwrap();
        assert!(wrong < 0.15, "a mis-shifted read still matched at {wrong}");
        assert!(matches!(encode_sequence(&bp, &[], &mut meter), Err(VsaError::Empty { what: "sequence" })));
    }

    /// The resonator factors a three-way composite from 20³ candidates at D = 4000 in at least 36
    /// of 40 trials — and every trial it gets wrong it REPORTS wrong, with a fidelity near zero,
    /// because the composite rebuilt from its answer is compared with the input. The success rate
    /// is a MEASURED property of this implementation (see the struct doc for the table), not a
    /// claim about the paper's; a spurious fixed point is a real outcome of the dynamics and the
    /// module's job is to make it visible. The iteration count is printed, not asserted.
    #[test]
    fn a_resonator_factors_a_three_way_composite() {
        let mut rng = Rng::new(8);
        let mut meter = Meter::default();
        let bp = Bipolar::new(4000).unwrap();
        let books: Vec<Codebook> = (0..3).map(|_| Codebook::random_bipolar(&bp, 20, &mut rng).unwrap()).collect();
        let res = Resonator::new(bp, books.clone(), 100).unwrap();
        assert!(!res.synchronous, "the measured-better update order is the default");
        let mut worst_iters = 0;
        let mut right = 0;
        for trial in 0..40usize {
            let want = [trial % 20, (trial * 7 + 3) % 20, (trial * 13 + 5) % 20];
            let mut s = books[0].symbols[want[0]].clone();
            s = bp.bind(&s, &books[1].symbols[want[1]], &mut meter).unwrap();
            s = bp.bind(&s, &books[2].symbols[want[2]], &mut meter).unwrap();
            let f = res.factor(&s, &mut meter).unwrap();
            if f.indices == want {
                right += 1;
                assert_eq!(f.fidelity, 1.0, "trial {trial}: the right answer rebuilds the input exactly");
                assert!(f.converged);
            } else {
                println!("resonator: trial {trial} answered {:?} for {want:?}, fidelity {:.3}, converged {}", f.indices, f.fidelity, f.converged);
                assert!(f.fidelity < 0.5, "trial {trial}: a wrong factoring reported fidelity {}", f.fidelity);
            }
            worst_iters = worst_iters.max(f.iterations);
        }
        println!("resonator: {right}/40 factored at D = 4000, worst case {worst_iters} iterations over 8000 candidates");
        assert!(right >= 36, "only {right} of 40 factored");
        // A composite that is NOT a product of codebook symbols does not reach fidelity 1.
        let noise = bp.random(&mut rng);
        let f = res.factor(&noise, &mut meter).unwrap();
        assert!(f.fidelity < 0.3, "a random vector factored at fidelity {}", f.fidelity);
        assert!(matches!(Resonator::new(bp, vec![], 10), Err(VsaError::Empty { .. })));
        let other = Bipolar::new(999).unwrap();
        assert!(matches!(Resonator::new(other, books, 10), Err(VsaError::Dimension { .. })));
    }

    // ---- the meter ----

    /// Every operation is counted in the unit the doc says: a bind is D, an HRR bind is D², a
    /// codebook search is M·D.
    #[test]
    fn the_meter_counts_what_each_operation_touches() {
        let mut rng = Rng::new(9);
        let bp = Bipolar::new(100).unwrap();
        let (a, b) = (bp.random(&mut rng), bp.random(&mut rng));
        let mut meter = Meter::default();
        bp.bind(&a, &b, &mut meter).unwrap();
        assert_eq!(meter.binds, 100);
        bp.bundle(&[a.clone(), b.clone(), a.clone()], &mut meter).unwrap();
        assert_eq!(meter.bundles, 400, "three members plus the threshold pass");
        bp.permute(&a, 3, &mut meter).unwrap();
        assert_eq!(meter.permutes, 100);
        bp.similarity(&a, &b, &mut meter).unwrap();
        assert_eq!(meter.similarities, 100);
        let book = Codebook::random_bipolar(&bp, 7, &mut rng).unwrap();
        book.nearest(&a, &mut meter).unwrap();
        assert_eq!(meter.similarities, 800);
        let hrr = Hrr::new(50).unwrap();
        let mut hm = Meter::default();
        hrr.bind(&hrr.random(&mut rng), &hrr.random(&mut rng), &mut hm).unwrap();
        assert_eq!(hm.binds, 2500, "a direct circular convolution is D² multiply-adds");
        assert_eq!(meter.total(), Some(100 + 400 + 100 + 800));
    }

    /// Every refusal names what was wrong.
    #[test]
    fn the_refusals_name_the_problem() {
        let mut meter = Meter::default();
        let bp = Bipolar::new(4).unwrap();
        assert!(matches!(Bipolar::new(0), Err(VsaError::Empty { what: "dimension" })));
        assert!(matches!(
            bp.bind(&[1.0, 1.0, 1.0, 1.0], &[1.0, 1.0, 1.0], &mut meter),
            Err(VsaError::Dimension { a: 3, b: 4 })
        ));
        assert!(matches!(
            bp.bind(&[1.0, 0.5, 1.0, 1.0], &[1.0; 4], &mut meter),
            Err(VsaError::NotSymbolic { index: 1, value }) if value == 0.5
        ));
        assert!(matches!(bp.bundle(&[], &mut meter), Err(VsaError::Empty { what: "bundle" })));
        let bn = Binary::new(2).unwrap();
        assert!(matches!(bn.check(&[0.0, 2.0]), Err(VsaError::NotSymbolic { index: 1, .. })));
        let hrr = Hrr::new(2).unwrap();
        assert!(matches!(hrr.check(&[0.0, f64::NAN]), Err(VsaError::NonFinite { index: 1 })));
        let book = Codebook { symbols: vec![vec![1.0, 1.0]], dim: 2 };
        assert!(matches!(book.nearest(&[1.0], &mut meter), Err(VsaError::Dimension { .. })));
        // A tie goes to the LOWER index, as the doc says; `>=` in the search survived the first
        // mutation sweep because no codebook held two equal symbols.
        let twins = Codebook { symbols: vec![vec![1.0, -1.0], vec![1.0, 1.0], vec![1.0, 1.0]], dim: 2 };
        assert_eq!(twins.nearest(&[1.0, 1.0], &mut meter).unwrap(), (1, 1.0));
        assert!(matches!(book.nearest(&[1.0, f64::INFINITY], &mut meter), Err(VsaError::NonFinite { index: 1 })));
        assert!(!book.is_empty());
        let e = VsaError::Factors { got: 2, want: 3 };
        assert!(e.to_string().contains("2 codebooks") && e.to_string().contains("3"));
        for e in [
            VsaError::Empty { what: "codebook" },
            VsaError::Dimension { a: 1, b: 2 },
            VsaError::NonFinite { index: 9 },
            VsaError::NotSymbolic { index: 3, value: 7.5 },
        ] {
            assert!(!e.to_string().is_empty());
        }
        assert!(noise_sd(0).is_nan());
    }


    /// Two things the second mutation sweep found untested: a resonator asked for zero iterations
    /// still runs one, and the HRR bundle's `normalise` flag does what it says — which no test
    /// had called with either value.
    #[test]
    fn a_resonator_runs_at_least_once_and_the_hrr_bundle_honours_its_flag() {
        let mut rng = Rng::new(4);
        let bp = Bipolar::new(64).unwrap();
        let books = vec![Codebook::random_bipolar(&bp, 3, &mut rng).unwrap(), Codebook::random_bipolar(&bp, 3, &mut rng).unwrap()];
        assert_eq!(Resonator::new(bp, books, 0).unwrap().max_iters, 1);
        let hrr = Hrr::new(4).unwrap();
        let mut meter = Meter::default();
        let (a, b) = (vec![3.0, 0.0, 0.0, 0.0], vec![0.0, 4.0, 0.0, 0.0]);
        assert_eq!(hrr.bundle(&[a.clone(), b.clone()], false, &mut meter).unwrap(), vec![3.0, 4.0, 0.0, 0.0]);
        assert_eq!(meter.bundles, 8, "two members of dimension four");
        assert_eq!(hrr.bundle(&[a, b], true, &mut meter).unwrap(), vec![0.6, 0.8, 0.0, 0.0]);
        assert_eq!(meter.bundles, 8 + 12, "and the rescaling is a third pass");
    }

    // ---- what the third mutation sweep found: guards, bookkeeping, billing, bounds ----

    /// The bipolar alphabet is exactly `{−1, +1}`, both at the model's own guard and at the
    /// resonator's check of every codebook symbol — a `0.0` is refused by both.
    ///
    /// Why the suite could not see it: every vector it handed to `check` came either from
    /// `random` or from a literal it had already decided was symbolic, and its one non-symbolic
    /// literal was `0.5`. A guard widened to admit `0.0` — the value a zeroed buffer, a cleared
    /// accumulator or an un-thresholded sum actually carries, and the one value that makes a
    /// bind silently return zeros — passed every assertion. The resonator's per-symbol check had
    /// no test at all: the only codebook it was ever handed came from `random_bipolar`.
    #[test]
    fn a_zero_is_not_a_bipolar_symbol_at_the_guard_or_inside_a_resonator_codebook() {
        let mut meter = Meter::default();
        let bp = Bipolar::new(3).unwrap();
        assert!(matches!(
            bp.check(&[1.0, 0.0, -1.0]),
            Err(VsaError::NotSymbolic { index: 1, value }) if value == 0.0
        ));
        assert!(matches!(
            bp.bind(&[1.0, 1.0, 1.0], &[1.0, 0.0, -1.0], &mut meter),
            Err(VsaError::NotSymbolic { index: 1, .. })
        ));
        assert!(matches!(
            bp.bundle(&[vec![0.0, 0.0, 0.0]], &mut meter),
            Err(VsaError::NotSymbolic { index: 0, .. })
        ));
        assert!(matches!(bp.permute(&[1.0, 1.0, 0.0], 1, &mut meter), Err(VsaError::NotSymbolic { index: 2, .. })));
        // A resonator checks the CONTENTS of every codebook, not just its shape: a zeroed symbol
        // and a fractional one are both refused, and the well-formed codebook is accepted.
        let zeroed = Codebook { symbols: vec![vec![1.0, -1.0, 1.0], vec![0.0, 0.0, 0.0]], dim: 3 };
        assert!(matches!(
            Resonator::new(bp, vec![zeroed], 10),
            Err(VsaError::NotSymbolic { index: 0, value }) if value == 0.0
        ));
        let fractional = Codebook { symbols: vec![vec![1.0, 0.5, 1.0]], dim: 3 };
        assert!(matches!(
            Resonator::new(bp, vec![fractional], 10),
            Err(VsaError::NotSymbolic { index: 1, value }) if value == 0.5
        ));
        let clean = Codebook { symbols: vec![vec![1.0, -1.0, 1.0], vec![-1.0, -1.0, 1.0]], dim: 3 };
        assert!(Resonator::new(bp, vec![clean], 10).is_ok());
    }

    /// Every guard that says "not finite" refuses an INFINITY as well as a `NaN`, at both of the
    /// codebook's readouts, and the dimension rule refuses a `NaN` separation rather than
    /// answering with a dimension.
    ///
    /// Why the suite could not see it: `Hrr::check` was only ever handed a `NaN` and
    /// `Codebook::nearest` only ever an infinity, so a guard narrowed from `!x.is_finite()` to
    /// `x.is_nan()` kept one of the two green and nothing exercised the other;
    /// `Codebook::similarities` was never handed a non-finite query at all; and `dimension_for`
    /// was never called with a non-finite `z`, so rewriting its NaN-rejecting `!(z > 0.0)` as the
    /// NaN-accepting `z <= 0.0` changed no assertion. A `NaN` separation then yields
    /// `NaN.ceil() as usize`, which is `0`: a capacity rule answering "zero dimensions".
    #[test]
    fn every_non_finite_guard_refuses_the_value_it_is_named_for() {
        let mut meter = Meter::default();
        let hrr = Hrr::new(2).unwrap();
        assert!(matches!(hrr.check(&[0.0, f64::INFINITY]), Err(VsaError::NonFinite { index: 1 })));
        assert!(matches!(hrr.check(&[f64::NEG_INFINITY, 0.0]), Err(VsaError::NonFinite { index: 0 })));
        assert!(matches!(hrr.check(&[0.0, f64::NAN]), Err(VsaError::NonFinite { index: 1 })));
        assert!(matches!(hrr.bind(&[1.0, f64::INFINITY], &[1.0, 1.0], &mut meter), Err(VsaError::NonFinite { index: 1 })));
        assert!(matches!(hrr.bundle(&[vec![f64::INFINITY, 0.0]], false, &mut meter), Err(VsaError::NonFinite { index: 0 })));

        let book = Codebook { symbols: vec![vec![1.0, 1.0], vec![1.0, -1.0]], dim: 2 };
        assert!(matches!(book.similarities(&[1.0, f64::NAN], &mut meter), Err(VsaError::NonFinite { index: 1 })));
        assert!(matches!(book.similarities(&[f64::INFINITY, 1.0], &mut meter), Err(VsaError::NonFinite { index: 0 })));
        assert!(matches!(book.nearest(&[1.0, f64::NAN], &mut meter), Err(VsaError::NonFinite { index: 1 })));

        // The dimension rule: a NaN separation is not a separation.
        assert_eq!(dimension_for(7, 1000, f64::NAN), None);
        assert_eq!(dimension_for(7, 1000, f64::INFINITY), Some(usize::MAX), "an infinite separation saturates rather than wrapping");
        assert_eq!(dimension_for(7, 1000, 0.0), None);
        assert_eq!(dimension_for(7, 1000, -1.0), None);
    }

    /// The binary and holographic models bill every element their operations touch: a Hamming
    /// distance and a similarity are `D` steps each, a permutation is `D` element moves, an HRR
    /// dot product is `D` multiply-accumulates, and a codebook's full similarity vector is `M·D`.
    ///
    /// Why the suite could not see it: the meter test counted the BIPOLAR model's four operations
    /// and the HRR bind, and nothing else. `Binary::hamming`, `Binary::permute`, `Hrr::similarity`
    /// and `Codebook::similarities` each ran inside other tests with a meter whose counters were
    /// never read, so halving any of their tallies changed no assertion — the module's whole
    /// argument is that an operation count is a number it is honest about, and half of `D` is the
    /// failure that reads as a plausible optimisation.
    #[test]
    fn the_binary_and_holographic_meters_bill_every_element_the_operation_touches() {
        let mut rng = Rng::new(11);
        let bn = Binary::new(16).unwrap();
        let (x, y) = (bn.random(&mut rng), bn.random(&mut rng));
        let mut m = Meter::default();
        bn.bind(&x, &y, &mut m).unwrap();
        assert_eq!(m.binds, 16, "an XOR bind is D element operations");
        bn.bundle(&[x.clone(), y.clone(), x.clone()], &mut m).unwrap();
        assert_eq!(m.bundles, 64, "three members plus the threshold pass");
        bn.permute(&x, 5, &mut m).unwrap();
        assert_eq!(m.permutes, 16, "a rotation moves every element, not half of them");
        bn.hamming(&x, &y, &mut m).unwrap();
        assert_eq!(m.similarities, 16, "a Hamming distance compares every position");
        bn.similarity(&x, &y, &mut m).unwrap();
        assert_eq!(m.similarities, 32, "and the normalised similarity is one more pass over D");
        assert_eq!(m.total(), Some(16 + 64 + 16 + 32));

        let hrr = Hrr::new(8).unwrap();
        let (a, b) = (hrr.random(&mut rng), hrr.random(&mut rng));
        let mut h = Meter::default();
        hrr.similarity(&a, &b, &mut h).unwrap();
        assert_eq!(h.similarities, 8, "a dot product is D multiply-accumulates");
        hrr.cosine(&a, &b, &mut h).unwrap().unwrap();
        assert_eq!(h.similarities, 16, "a cosine is a dot product plus two norms");
        hrr.permute(&a, 3, &mut h).unwrap();
        assert_eq!(h.permutes, 8);

        // A codebook's full similarity vector is M·D, the same M·D its nearest-neighbour search is.
        let bp = Bipolar::new(20).unwrap();
        let book = Codebook::random_bipolar(&bp, 6, &mut rng).unwrap();
        let q = bp.random(&mut rng);
        let mut c = Meter::default();
        book.similarities(&q, &mut c).unwrap();
        assert_eq!(c.similarities, 120);
        book.nearest(&q, &mut c).unwrap();
        assert_eq!(c.similarities, 240);
    }

    /// A codebook reports the dimension its symbols actually have, the count it actually holds,
    /// and emptiness truthfully — for all three constructors.
    ///
    /// Why the suite could not see it: every codebook it built was read back through `symbols`
    /// directly, so `len()` was never called; its one `is_empty` assertion was `!book.is_empty()`
    /// on a codebook with a symbol in it, which a hard-coded `false` also satisfies; and
    /// `Codebook::random_binary` appears in no other test at all, so a `dim` field one larger
    /// than the symbols it describes was never compared with anything — until a query of the
    /// model's dimension reaches it and is refused for the wrong length.
    #[test]
    fn a_codebook_reports_its_own_dimension_symbol_count_and_emptiness() {
        let mut rng = Rng::new(12);
        let mut meter = Meter::default();
        let bp = Bipolar::new(8).unwrap();
        let bn = Binary::new(8).unwrap();
        let hr = Hrr::new(8).unwrap();
        let books = [
            Codebook::random_bipolar(&bp, 5, &mut rng).unwrap(),
            Codebook::random_binary(&bn, 5, &mut rng).unwrap(),
            Codebook::random_hrr(&hr, 5, &mut rng).unwrap(),
        ];
        for (which, book) in books.iter().enumerate() {
            assert_eq!(book.dim, 8, "codebook {which} declares the model's dimension");
            assert_eq!(book.len(), 5, "codebook {which} holds what it was asked for");
            assert_eq!(book.len(), book.symbols.len(), "codebook {which}: len() is the symbol count");
            assert!(!book.is_empty());
            for s in &book.symbols {
                assert_eq!(s.len(), book.dim, "codebook {which}: a symbol is as long as the declared dimension");
            }
            // A declared dimension that does not match the symbols makes every readout a
            // Dimension refusal; a correct one answers with one score per symbol.
            let sims = book.similarities(&[1.0; 8], &mut meter).unwrap();
            assert_eq!(sims.len(), 5);
        }
        // Binary symbols are 0/1, bipolar are ±1, and each model accepts its own codebook.
        for s in &books[1].symbols {
            assert!(bn.check(s).is_ok());
        }
        for s in &books[0].symbols {
            assert!(bp.check(s).is_ok());
        }
        let empty = Codebook { symbols: Vec::new(), dim: 8 };
        assert!(empty.is_empty(), "a codebook with no symbols is empty");
        assert_eq!(empty.len(), 0);
    }

    /// A random codebook's symbols are DISTINCT and near-orthogonal — the property every capacity
    /// claim in this module divides by — in all three models.
    ///
    /// Why the suite could not see it: a codebook of one repeated vector answers every `nearest`
    /// and every `similarities` call perfectly well, and the tests that use codebooks read a
    /// symbol back by index or ask for a ranking, never for the relationship BETWEEN two symbols.
    /// `Codebook::random_hrr` is used by no other test at all. So a constructor that drew one
    /// vector and cloned it `M` times — which collapses the whole representation onto a single
    /// symbol — passed the entire suite.
    ///
    /// The bound is arithmetic, in each model's own similarity: two independent symbols score a
    /// mean of `D` fair `±1` products, whose standard deviation is `1/√D = 1/16` at `D = 256`, so
    /// `0.375` is six standard deviations and the 28 pairs of an eight-symbol codebook clear it
    /// with probability better than `1 − 1e-7`. A symbol against ITSELF scores exactly `1`, which
    /// is what two clones would score against each other.
    #[test]
    fn a_random_codebook_draws_distinct_near_orthogonal_symbols() {
        let mut rng = Rng::new(13);
        let mut meter = Meter::default();
        let d = 256usize;
        let bound = 6.0 * noise_sd(d);
        let bp = Bipolar::new(d).unwrap();
        let bn = Binary::new(d).unwrap();
        let hr = Hrr::new(d).unwrap();
        let bipolar = Codebook::random_bipolar(&bp, 8, &mut rng).unwrap();
        let binary = Codebook::random_binary(&bn, 8, &mut rng).unwrap();
        let holographic = Codebook::random_hrr(&hr, 8, &mut rng).unwrap();
        for (which, book) in [("bipolar", &bipolar), ("binary", &binary), ("hrr", &holographic)] {
            for i in 0..book.len() {
                for j in 0..book.len() {
                    let (x, y) = (&book.symbols[i], &book.symbols[j]);
                    let s = match which {
                        "bipolar" => bp.similarity(x, y, &mut meter).unwrap(),
                        "binary" => bn.similarity(x, y, &mut meter).unwrap(),
                        _ => hr.cosine(x, y, &mut meter).unwrap().unwrap(),
                    };
                    if i == j {
                        assert!((s - 1.0).abs() < 1e-12, "{which}: symbol {i} scores {s} against itself");
                    } else {
                        assert!(s.abs() < bound, "{which}: symbols {i} and {j} score {s} — not independent draws");
                        assert_ne!(x, y, "{which}: symbols {i} and {j} are the same vector");
                    }
                }
            }
        }
    }

    /// The codebook readout is the COSINE — the dot product divided by `D` — and the search is
    /// initialised below every score it could see, so a query whose every match is negative still
    /// reports the best of them rather than a default.
    ///
    /// Why the suite could not see either: every readout it made was of a bundle or a permuted
    /// code that CONTAINED the answer, so the winning score was positive and a search initialised
    /// at `0.0` found it anyway; and every assertion on a similarity was a threshold comparison
    /// (`> 0.2`, `< 0.15`, `> best`) that an undivided dot product — `D` times too large —
    /// passes at least as easily. Here the scores are exact small integers over `D = 4`.
    #[test]
    fn the_codebook_readout_is_divided_by_d_and_ranks_an_all_negative_query() {
        let mut meter = Meter::default();
        let book = Codebook { symbols: vec![vec![1.0, 1.0, 1.0, 1.0], vec![1.0, 1.0, 1.0, -1.0]], dim: 4 };
        // Dot products −4 and −2 over D = 4: the cosines are −1.0 and −0.5, both negative.
        let all_negative = vec![-1.0, -1.0, -1.0, -1.0];
        assert_eq!(book.similarities(&all_negative, &mut meter).unwrap(), vec![-1.0, -0.5]);
        assert_eq!(book.nearest(&all_negative, &mut meter).unwrap(), (1, -0.5), "the best of two negative matches is still the answer");
        // The score `nearest` reports is the entry `similarities` computes, to the last bit: the
        // two are the same expression over the same operands in the same order.
        let (idx, score) = book.nearest(&all_negative, &mut meter).unwrap();
        assert_eq!(score, book.similarities(&all_negative, &mut meter).unwrap()[idx]);
        // And a symbol is its own nearest neighbour at a cosine of exactly 1.
        assert_eq!(book.similarities(&book.symbols[0], &mut meter).unwrap(), vec![1.0, 0.5]);
        assert_eq!(book.nearest(&book.symbols[0], &mut meter).unwrap(), (0, 1.0));
    }

    /// A normalised holographic bundle whose members cancel exactly is returned AS the zero
    /// vector, not divided by its own zero norm.
    ///
    /// Why the suite could not see it: the only normalised bundle it ever built was
    /// `[3,0,0,0] + [0,4,0,0]`, whose norm is 5, so the `n > 0.0` guard was never reached with a
    /// zero. Relaxing it to `n >= 0.0` makes the result `0.0 / 0.0`, a vector of `NaN`s that no
    /// assertion in the module compared against anything — and that a downstream `Hrr::check`
    /// would only reject much later, at a call site that had already lost the cause.
    #[test]
    fn a_holographic_bundle_that_cancels_normalises_to_zero_rather_than_nan() {
        let mut meter = Meter::default();
        let hrr = Hrr::new(4).unwrap();
        let a = vec![1.0, -2.0, 0.5, 0.0];
        let minus_a: Vec<f64> = a.iter().map(|x| -x).collect();
        let out = hrr.bundle(&[a, minus_a], true, &mut meter).unwrap();
        assert_eq!(out, vec![0.0; 4], "a zero sum stays zero under the normalise flag");
        assert!(out.iter().all(|x| x.is_finite()));
        assert!(hrr.check(&out).is_ok(), "and the result is still a symbol of the model");
        assert_eq!(meter.bundles, 12, "two members plus the rescaling pass over D = 4");
        // The all-zero member list is the same case reached the other way.
        assert_eq!(hrr.bundle(&[vec![0.0; 4], vec![0.0; 4]], true, &mut meter).unwrap(), vec![0.0; 4]);
    }

    /// A resonator checks each codebook's declared dimension against the MODEL's, not against
    /// itself.
    ///
    /// Why the suite could not see it: its one dimension refusal handed a 999-dimensional model
    /// codebooks of 4000-dimensional symbols, so the per-symbol `check` raised
    /// `VsaError::Dimension` as well and a `matches!` on the variant passed with the codebook
    /// check disabled. The separating case is a codebook whose `dim` field disagrees with its own
    /// symbols: the symbols then satisfy the model and only the declared dimension is wrong —
    /// and that field is what `Codebook::nearest` and `Codebook::similarities` divide by inside
    /// the resonator's own loop.
    #[test]
    fn a_resonator_checks_each_codebook_against_the_models_dimension_not_its_own() {
        let bp = Bipolar::new(4).unwrap();
        let mislabelled = Codebook { symbols: vec![vec![1.0, -1.0, 1.0, -1.0]], dim: 8 };
        assert!(matches!(
            Resonator::new(bp, vec![mislabelled], 10),
            Err(VsaError::Dimension { a: 8, b: 4 })
        ));
        let honest = Codebook { symbols: vec![vec![1.0, -1.0, 1.0, -1.0]], dim: 4 };
        assert!(Resonator::new(bp, vec![honest.clone()], 10).is_ok());
        // Mixed: the second codebook is the wrong one, and it is still caught.
        let wrong = Codebook { symbols: vec![vec![1.0, 1.0, 1.0, 1.0]], dim: 2 };
        assert!(matches!(
            Resonator::new(bp, vec![honest, wrong], 10),
            Err(VsaError::Dimension { a: 2, b: 4 })
        ));
    }

    /// `max_iters` bounds the iterations RUN — a resonator given one iteration runs exactly one —
    /// and the `synchronous` flag selects the paper's update order rather than the sequential one.
    ///
    /// Why the suite could not see either: the only resonator test ran to convergence with
    /// `max_iters = 100`, so the loop bound was never the thing that stopped it and running one
    /// iteration too many changed nothing it asserted; and `synchronous` was only ever read
    /// (`assert!(!res.synchronous)`), never SET, so the two branches of the update were never
    /// compared and swapping them was invisible.
    ///
    /// The iteration counts are MEASURED on this instance (`D = 512`, three codebooks of eight,
    /// seed 23, factors 1, 4 and 7) and are deterministic by seed: sequential reaches its fixed
    /// point in 3 iterations, the synchronous form in 4 — consistent with the struct doc's table,
    /// where the sequential update is the better of the two on every row.
    #[test]
    fn a_resonator_runs_exactly_max_iters_and_the_synchronous_flag_picks_the_papers_update() {
        let mut rng = Rng::new(23);
        let mut meter = Meter::default();
        let bp = Bipolar::new(512).unwrap();
        let books: Vec<Codebook> = (0..3).map(|_| Codebook::random_bipolar(&bp, 8, &mut rng).unwrap()).collect();
        let want = [1usize, 4, 7];
        let mut s = vec![1.0f64; 512];
        for (b, &i) in books.iter().zip(&want) {
            s = bp.bind(&s, &b.symbols[i], &mut meter).unwrap();
        }

        // One iteration means one iteration: the fixed point is not reached in it, so the bound
        // is what stops the loop and the count it reports is the bound itself.
        let capped = Resonator::new(bp, books.clone(), 1).unwrap();
        let f = capped.factor(&s, &mut meter).unwrap();
        assert_eq!(f.iterations, 1, "a resonator bounded at one iteration ran {}", f.iterations);
        assert!(!f.converged, "no fixed point was detected inside the bound");

        let sequential = Resonator::new(bp, books.clone(), 100).unwrap();
        let mut synchronous = sequential.clone();
        synchronous.synchronous = true;
        let seq = sequential.factor(&s, &mut meter).unwrap();
        let syn = synchronous.factor(&s, &mut meter).unwrap();
        assert_eq!(seq.indices, want);
        assert_eq!(syn.indices, want);
        assert_eq!(seq.fidelity, 1.0);
        assert_eq!(syn.fidelity, 1.0);
        assert!(seq.converged && syn.converged);
        assert_eq!(seq.iterations, 3, "measured: the sequential update reaches the fixed point in 3");
        assert_eq!(syn.iterations, 4, "measured: the paper's synchronous update needs 4 on the same instance");
    }

    /// The binary draw returns the generator's bit, not its complement.
    ///
    /// Why the suite could not see it: the complement of a fair coin is a fair coin, so every
    /// statistical property the suite checks — bundle similarity, orthogonality, Hamming
    /// distance, the isomorphism to the bipolar model — holds exactly as well on the complemented
    /// stream, and no test compared a drawn symbol against anything the generator had said. This
    /// pins it against the bipolar draw, which reads the same low bit of the same stream: from
    /// one seed the two models' symbols agree position by position under `b ↦ 2b − 1`, so a
    /// complemented binary draw disagrees at EVERY position.
    #[test]
    fn the_binary_draw_is_the_generators_bit_and_not_its_complement() {
        let bn = Binary::new(64).unwrap();
        let bp = Bipolar::new(64).unwrap();
        let mut one = Rng::new(31);
        let mut two = Rng::new(31);
        let bits = bn.random(&mut one);
        let signs = bp.random(&mut two);
        let from_signs: Vec<f64> = signs.iter().map(|&s| (s + 1.0) / 2.0).collect();
        assert_eq!(bits, from_signs, "the two discrete models read the same bit with the same orientation");
        assert!(bn.check(&bits).is_ok());
        assert!(bits.contains(&1.0) && bits.contains(&0.0), "the draw is not constant");
    }
    /// An unbind is billed for the involution it performs as well as for the convolution.
    ///
    /// `Hrr::unbind` is `involution` then `bind`. The bind is `D²` multiply-accumulates and was
    /// billed; the involution is `D` element moves — the reversal permutation `a[(d − i) % d]` —
    /// and was billed nothing, although every other element move in this module is counted
    /// (`Bipolar::permute`, `Binary::permute` and `Hrr::permute` all `tally(.., self.dim)`).
    ///
    /// Measured: the under-count is `D` out of `D² + D`, which is 0.4% at `D = 256`. Small, but
    /// systematic and in the direction that flatters the operation — and this crate's whole
    /// argument is that it prices what it does. No fixture read an unbind's meter at all, which is
    /// why correcting it changed no existing test.
    #[test]
    fn an_unbind_is_billed_for_the_involution_as_well_as_the_convolution() {
        let d = 64;
        let hrr = Hrr::new(d).unwrap();
        let mut rng = Rng::new(5);
        let a = hrr.random(&mut rng);
        let b = hrr.random(&mut rng);
        let mut meter = Meter::default();
        let ab = hrr.bind(&a, &b, &mut meter).unwrap();
        assert_eq!((meter.binds, meter.permutes), ((d * d) as u64, 0), "a bind moves nothing");
        let mut meter = Meter::default();
        let _ = hrr.unbind(&ab, &b, &mut meter).unwrap();
        assert_eq!(meter.binds, (d * d) as u64, "the convolution, as before");
        assert_eq!(meter.permutes, d as u64, "and the D element moves the involution makes");
        assert_eq!(meter.total(), Some((d * d + d) as u64));
    }

}
