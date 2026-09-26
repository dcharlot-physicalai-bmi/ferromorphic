//! Sparse distributed representations: why a few active bits out of many are almost impossible to
//! confuse, and exactly how almost.
//!
//! # What the mechanism is
//!
//! Cortex represents things with a small fraction of a large population active — a few hundred of
//! tens of thousands. Ahmad and Hawkins (*Properties of sparse distributed representations and
//! their application to hierarchical temporal memory*, arXiv:1503.07469, 2015) work out what that
//! buys, and the answers are combinatorial rather than empirical.
//!
//! A representation is `n` bits of which `w` are on. Two independent ones overlap in `b` bits with
//! the **hypergeometric** probability
//!
//! ```text
//! P(overlap = b) = C(w, b) C(n − w, w − b) / C(n, w)
//! ```
//!
//! so a classifier that fires when the overlap reaches `θ` has a false-positive rate of the tail
//! of that distribution — and the tail falls off so fast that, at cortical sizes, matching
//! **half** a pattern is already astronomically unlikely to happen by chance. That is the whole
//! argument for sparse codes: not that they are efficient, but that they are unconfusable, and
//! that this survives losing most of the bits.
//!
//! Three consequences follow, and all three are computed here rather than asserted:
//!
//! - **Subsampling.** Keep only `s` of a pattern's `w` bits and the match still works: the
//!   false-positive rate against a random pattern is the same tail with `s` in place of `w`.
//! - **Noise.** Flip some bits and the overlap with the original falls by a known amount, which
//!   sets how much noise a threshold tolerates.
//! - **Unions.** Store `m` patterns by OR-ing them, and a random pattern matches the union with a
//!   probability that grows with `m` — which is the capacity of the store, and it is enormous
//!   before it is exceeded.
//!
//! # Why it is in a neuromorphic crate
//!
//! A sparse binary vector with a threshold is a spiking population with a coincidence detector,
//! and this is the arithmetic that says how many patterns such a detector can tell apart. It is
//! the counting argument behind [`crate::vsa`]'s binding, [`crate::delays`]'s pattern selectivity
//! and [`crate::polychron`]'s groups; and on hardware it is the reason a sparse code can be
//! subsampled — which is to say, why a synapse budget can be small.
//!
//! # The closed forms this module is checked against
//!
//! - **The distribution is a distribution.** `Σ_b P(overlap = b) = 1` over the whole support, and
//!   the mean overlap is `w²/n` — both checked against the computed values, and the mean against
//!   an independent Monte Carlo draw.
//! - **Small cases by hand.** For `n = 4`, `w = 2` the overlaps are `1/6, 4/6, 1/6`, which is
//!   written out in the test rather than computed from the same formula.
//! - **The tail is the sum of the terms**, and it falls monotonically in the threshold.
//! - **The numbers this module computes, stated as its own.** At the sizes usually quoted for
//!   cortex — `n = 2048`, `w = 40` — matching half a pattern by chance is `2.5 × 10⁻²⁶`; a
//!   dendrite sampling twenty of the forty bits and requiring all twenty is `2.2 × 10⁻³⁷`;
//!   requiring only half of that subsample, which is what robustness to noise costs, is still
//!   `3.9 × 10⁻¹³` — the paper's own example (§II.E), which it states as "better than 1 in 10¹²".
//! - **The paper's own tables.** Its worked examples (§II.B–II.H) and its "SDR Actuarial Tables"
//!   (Appendix A) are in the open arXiv full text, arXiv:1503.07469v1. This module's overlap tail
//!   reproduces all 16 rows of Table 1 (exact matches) and all 16 rows of Table 2 (inexact matches)
//!   to within half a unit of the last printed digit, as does [`union_false_positive_rate`] for
//!   the nine rows of Table 4 with `θ = w`, and the text's "one in 14,587" (§II.D),
//!   "one in 3,142" and "1 in 2.5 million" (§II.E). Earlier releases said this review had not
//!   located those numbers; they are checked now.
//! - **Every probability is in `[0, 1]`**, including where the binomials overflow a `u64` — the
//!   computation is done in logarithms, and the test drives it to `n = 100_000` where a factorial
//!   would have no chance.
//! - **Overlap, noise and union counts** are checked against directly enumerated bit vectors, so
//!   the combinatorics and the bits agree.
//!
//! # What this module has NOT reproduced
//!
//! - Hierarchical Temporal Memory itself: the spatial pooler, the temporal memory, boosting, and
//!   the learning rules. This is the arithmetic those rest on, not the algorithm.
//! - The paper's treatment of unions with noise together, and its analysis of classifier capacity
//!   under learning.
//! - Any claim about cortex. The sizes quoted are the paper's, and they are quoted as its
//!   assumptions.

use core::fmt;

/// The largest population this module will compute for.
pub const MAX_BITS: usize = 1 << 22;

/// What went wrong, named rather than guessed around.
#[derive(Debug, Clone, PartialEq)]
pub enum SdrError {
    /// `n` is zero or past [`MAX_BITS`], or `w` is zero or larger than `n`.
    BadShape {
        /// The population size.
        n: usize,
        /// The number of active bits.
        w: usize,
    },
    /// An index past the end of the population.
    OutOfRange {
        /// The index.
        index: usize,
        /// The population size.
        n: usize,
    },
    /// Two representations of different population sizes.
    Mismatched {
        /// The first.
        a: usize,
        /// The second.
        b: usize,
    },
}

impl fmt::Display for SdrError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadShape { n, w } => write!(f, "{w} active bits out of {n}"),
            Self::OutOfRange { index, n } => write!(f, "bit {index} of a population of {n}"),
            Self::Mismatched { a, b } => write!(f, "populations of {a} and {b} bits"),
        }
    }
}

impl std::error::Error for SdrError {}

/// `ln(k!)`, by the exact sum for small `k` and by Stirling's series above it.
fn ln_factorial(k: u64) -> f64 {
    if k < 2 {
        return 0.0;
    }
    if k < 256 {
        return (2..=k).map(|i| (i as f64).ln()).sum();
    }
    // Stirling with the first two correction terms: the error is below 1e-14 relative at k = 256.
    let x = k as f64;
    x * x.ln() - x + 0.5 * (core::f64::consts::TAU * x).ln() + 1.0 / (12.0 * x) - 1.0 / (360.0 * x * x * x)
}

/// `ln C(n, k)`, or `None` for `k > n`.
#[must_use]
pub fn ln_choose(n: u64, k: u64) -> Option<f64> {
    if k > n {
        return None;
    }
    Some(ln_factorial(n) - ln_factorial(k) - ln_factorial(n - k))
}

/// The probability that two independent representations of `n` bits, with `w` and `s` active,
/// overlap in exactly `b` bits.
///
/// The hypergeometric law `C(s, b) C(n − s, w − b) / C(n, w)`. Zero where the overlap is
/// impossible. `None` for a shape that makes no sense.
#[must_use]
pub fn overlap_probability(n: usize, w: usize, s: usize, b: usize) -> Option<f64> {
    if n == 0 || n > MAX_BITS || w > n || s > n {
        return None;
    }
    if b > w.min(s) || w + s > n + b {
        return Some(0.0);
    }
    let (n, w, s, b) = (n as u64, w as u64, s as u64, b as u64);
    let ln = ln_choose(s, b)? + ln_choose(n - s, w - b)? - ln_choose(n, w)?;
    Some(ln.exp().clamp(0.0, 1.0))
}

/// The chance that a random representation with `w` active bits overlaps a fixed set of `s` bits
/// in at least `theta` places — the false-positive rate of a coincidence detector.
///
/// `s == w` is the whole-pattern case; `s < w` is the subsampled one. `None` for a bad shape.
#[must_use]
pub fn false_positive_rate(n: usize, w: usize, s: usize, theta: usize) -> Option<f64> {
    if n == 0 || n > MAX_BITS || w > n || s > n {
        return None;
    }
    if theta > w.min(s) {
        return Some(0.0);
    }
    let total: f64 = (theta..=w.min(s)).filter_map(|b| overlap_probability(n, w, s, b)).sum();
    Some(total.clamp(0.0, 1.0))
}

/// The mean overlap of two independent representations with `w` and `s` active bits: `w s / n`.
/// `None` for a bad shape.
#[must_use]
pub fn mean_overlap(n: usize, w: usize, s: usize) -> Option<f64> {
    if n == 0 || n > MAX_BITS || w > n || s > n {
        return None;
    }
    Some((w * s) as f64 / n as f64)
}

/// A sparse distributed representation: the indices of its active bits, sorted and distinct.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sdr {
    /// The population size.
    pub n: usize,
    /// The active bits, ascending and without repeats.
    pub on: Vec<usize>,
}

impl Sdr {
    /// Build from a list of indices, which is sorted and de-duplicated.
    ///
    /// # Errors
    ///
    /// [`SdrError::BadShape`] for a population of zero or past [`MAX_BITS`];
    /// [`SdrError::OutOfRange`] for an index past the end.
    pub fn new(n: usize, mut on: Vec<usize>) -> Result<Self, SdrError> {
        if n == 0 || n > MAX_BITS {
            return Err(SdrError::BadShape { n, w: on.len() });
        }
        if let Some(&index) = on.iter().find(|&&i| i >= n) {
            return Err(SdrError::OutOfRange { index, n });
        }
        on.sort_unstable();
        on.dedup();
        Ok(Self { n, on })
    }

    /// A random representation with `w` active bits, by a partial shuffle — so the bits are
    /// distinct and every set of `w` is equally likely.
    ///
    /// # Errors
    ///
    /// [`SdrError::BadShape`] for `w` of zero, `w` larger than `n`, or a bad population.
    pub fn random(n: usize, w: usize, rng: &mut crate::rng::Rng) -> Result<Self, SdrError> {
        if n == 0 || n > MAX_BITS || w == 0 || w > n {
            return Err(SdrError::BadShape { n, w });
        }
        let mut pool: Vec<usize> = (0..n).collect();
        for k in 0..w {
            let pick = k + rng.below((n - k) as u32) as usize;
            pool.swap(k, pick);
        }
        Self::new(n, pool[..w].to_vec())
    }

    /// How many bits are active.
    #[must_use]
    pub fn w(&self) -> usize {
        self.on.len()
    }

    /// How many bits two representations share.
    ///
    /// # Errors
    ///
    /// [`SdrError::Mismatched`] for different population sizes.
    pub fn overlap(&self, other: &Self) -> Result<usize, SdrError> {
        if self.n != other.n {
            return Err(SdrError::Mismatched { a: self.n, b: other.n });
        }
        let (mut i, mut j, mut count) = (0, 0, 0);
        while i < self.on.len() && j < other.on.len() {
            match self.on[i].cmp(&other.on[j]) {
                core::cmp::Ordering::Less => i += 1,
                core::cmp::Ordering::Greater => j += 1,
                core::cmp::Ordering::Equal => {
                    count += 1;
                    i += 1;
                    j += 1;
                }
            }
        }
        Ok(count)
    }

    /// A subsample keeping the first `s` active bits in index order.
    ///
    /// # Errors
    ///
    /// [`SdrError::BadShape`] if `s` is larger than the number active.
    pub fn subsample(&self, s: usize) -> Result<Self, SdrError> {
        if s > self.w() {
            return Err(SdrError::BadShape { n: self.n, w: s });
        }
        Self::new(self.n, self.on[..s].to_vec())
    }

    /// The bitwise union of several representations.
    ///
    /// # Errors
    ///
    /// [`SdrError::Mismatched`] if they are not all the same size, or [`SdrError::BadShape`] for
    /// an empty list.
    pub fn union(parts: &[Self]) -> Result<Self, SdrError> {
        let Some(first) = parts.first() else {
            return Err(SdrError::BadShape { n: 0, w: 0 });
        };
        if let Some(other) = parts.iter().find(|p| p.n != first.n) {
            return Err(SdrError::Mismatched { a: first.n, b: other.n });
        }
        Self::new(first.n, parts.iter().flat_map(|p| p.on.iter().copied()).collect())
    }

    /// A copy with `flips` of the active bits moved to inactive positions — noise that keeps the
    /// number of active bits the same.
    ///
    /// # Errors
    ///
    /// [`SdrError::BadShape`] if there are not `flips` bits to move, or nowhere to move them.
    pub fn with_noise(&self, flips: usize, rng: &mut crate::rng::Rng) -> Result<Self, SdrError> {
        if flips > self.w() || self.w() + flips > self.n {
            return Err(SdrError::BadShape { n: self.n, w: flips });
        }
        let mut kept = self.on.clone();
        for k in 0..flips {
            let pick = k + rng.below((kept.len() - k) as u32) as usize;
            kept.swap(k, pick);
        }
        let mut out: Vec<usize> = kept[flips..].to_vec();
        let mut off: Vec<usize> = (0..self.n).filter(|i| !self.on.contains(i)).collect();
        for k in 0..flips {
            let pick = k + rng.below((off.len() - k) as u32) as usize;
            off.swap(k, pick);
            out.push(off[k]);
        }
        Self::new(self.n, out)
    }
}

/// The chance that a random representation with `w` active bits reaches an overlap of `theta`
/// with a union of `m` independent patterns of `w` bits each.
///
/// The union has an expected `n(1 − (1 − w/n)^m)` bits on, and this treats it as a fixed set of
/// that many (rounded to an integer) — the paper's eqs. (14)–(15), an approximation, and stated as
/// one.
///
/// ⚠ **The paper's own noisy-union numbers do not follow its eq. (14).** Its Table 4 rows with
/// `θ < w` and the §II.H figures ("1 in 123 million", "1 in 4 million", "1 in 223 billion")
/// reproduce to six figures as `C(w̃, b) · C(n − w, w − b) / C(n, w)`, which is not a probability —
/// it reaches 2.03 at the Table 4 row the paper prints as 1. And its "1 in 5 billion" for `θ = w`
/// (§II.H) comes from eq. (13) (1 in 5.45 × 10⁹), where its Table 4 prints 1.2532 × 10⁻¹⁰ (1 in
/// 7.98 × 10⁹); this function gives the latter. Only the `θ = w` rows of Table 4 are a fixture for
/// it.
///
/// `None` for a bad shape.
#[must_use]
pub fn union_false_positive_rate(n: usize, w: usize, m: usize, theta: usize) -> Option<f64> {
    if n == 0 || n > MAX_BITS || w > n || m == 0 {
        return None;
    }
    let expected = n as f64 * (1.0 - (1.0 - w as f64 / n as f64).powi(i32::try_from(m).ok()?));
    let s = (expected.round() as usize).min(n);
    false_positive_rate(n, w, s, theta)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    #[test]
    fn the_overlap_law_is_a_distribution_with_the_mean_it_claims() {
        // By hand: n = 4, w = s = 2. The pairs sharing 0, 1, 2 bits are 1, 4, 1 of the 6.
        let by_hand = [1.0 / 6.0, 4.0 / 6.0, 1.0 / 6.0];
        for (b, want) in by_hand.iter().enumerate() {
            let got = overlap_probability(4, 2, 2, b).unwrap();
            assert!((got - want).abs() < 1e-15, "overlap {b}: {got} against {want}");
        }
        assert_eq!(overlap_probability(4, 2, 2, 3), Some(0.0));
        // An overlap that is forced: w + s > n means they must share at least w + s − n.
        assert_eq!(overlap_probability(4, 3, 3, 1), Some(0.0));
        assert!(overlap_probability(4, 3, 3, 2).unwrap() > 0.0);

        for &(n, w, s) in &[(4usize, 2usize, 2usize), (2048, 40, 40), (2048, 40, 20), (1000, 100, 7), (64, 64, 3)] {
            let mass: f64 = (0..=w.min(s)).map(|b| overlap_probability(n, w, s, b).unwrap()).sum();
            assert!((mass - 1.0).abs() < 1e-9, "n={n} w={w} s={s}: the probabilities sum to {mass}");
            let mean: f64 = (0..=w.min(s)).map(|b| b as f64 * overlap_probability(n, w, s, b).unwrap()).sum();
            let want = mean_overlap(n, w, s).unwrap();
            assert!((mean - want).abs() < 1e-9, "n={n} w={w} s={s}: mean {mean} against {want}");
        }
        assert!(overlap_probability(0, 0, 0, 0).is_none() && overlap_probability(4, 5, 2, 0).is_none());
        assert!(overlap_probability(4, 2, 5, 0).is_none() && overlap_probability(MAX_BITS + 1, 2, 2, 0).is_none());
        assert!(mean_overlap(0, 1, 1).is_none() && mean_overlap(4, 5, 1).is_none());
    }

    #[test]
    fn the_law_agrees_with_actually_drawing_the_bits() {
        // The combinatorics and the bits are two different computations; here they meet.
        let (n, w, draws) = (200usize, 10usize, 40_000usize);
        let mut rng = Rng::new(5);
        let fixed = Sdr::random(n, w, &mut rng).unwrap();
        let mut counts = vec![0usize; w + 1];
        for _ in 0..draws {
            let other = Sdr::random(n, w, &mut rng).unwrap();
            counts[fixed.overlap(&other).unwrap()] += 1;
        }
        let measured_mean: f64 = counts.iter().enumerate().map(|(b, c)| b as f64 * *c as f64).sum::<f64>() / draws as f64;
        let want = mean_overlap(n, w, w).unwrap();
        assert!((measured_mean - want).abs() < 0.02, "drew a mean overlap of {measured_mean} for {want}");
        for b in 0..=3 {
            let seen = counts[b] as f64 / draws as f64;
            let law = overlap_probability(n, w, w, b).unwrap();
            assert!((seen - law).abs() < 0.01, "overlap {b}: drew {seen}, law says {law}");
        }
        // Every draw has exactly w bits, distinct and sorted.
        let mut smallest = w;
        for _ in 0..2_000 {
            let one = Sdr::random(n, w, &mut rng).unwrap();
            smallest = smallest.min(one.w());
            assert!(one.on.windows(2).all(|p| p[0] < p[1]));
            assert_eq!(one.overlap(&one).unwrap(), one.w());
        }
        assert_eq!(smallest, w, "some draw came back with only {smallest} of {w} bits");
        // And every bit is equally likely to be drawn. This is not implied by the above: the
        // partial shuffle swaps, so it can never produce a repeat however the index is chosen,
        // and a sampler that drew the swap partner from the WHOLE pool each time — the classic
        // naive shuffle — would still hand back w distinct bits every time while badly skewing
        // which ones. MEASURED at a size where the skew is unmissable: correct sampling holds
        // every marginal within a percent of w/n, the naive version misses by more than half.
        let (small, few, draws) = (20usize, 5usize, 40_000usize);
        let mut seen = vec![0usize; small];
        for _ in 0..draws {
            for b in Sdr::random(small, few, &mut rng).unwrap().on {
                seen[b] += 1;
            }
        }
        let expected = (draws * few) as f64 / small as f64;
        let worst = seen.iter().map(|c| (*c as f64 / expected - 1.0).abs()).fold(0.0f64, f64::max);
        assert!(worst < 0.1, "the marginals are skewed by {worst}: {seen:?}");
        assert_eq!(seen.iter().sum::<usize>(), draws * few);
    }

    #[test]
    fn the_false_positive_rate_is_the_tail_and_it_falls_with_the_threshold() {
        let (n, w) = (2048usize, 40usize);
        let mut last = f64::INFINITY;
        for theta in 1..=w {
            let tail = false_positive_rate(n, w, w, theta).unwrap();
            let by_sum: f64 = (theta..=w).map(|b| overlap_probability(n, w, w, b).unwrap()).sum();
            assert!((tail - by_sum).abs() < 1e-18 * by_sum.max(1e-300), "theta {theta}");
            assert!(tail < last, "theta {theta} did not fall: {tail} against {last}");
            assert!((0.0..=1.0).contains(&tail));
            last = tail;
        }
        // An overlap of zero or more is certain, up to the rounding of forty-one terms each
        // computed through an exponential.
        assert!((false_positive_rate(n, w, w, 0).unwrap() - 1.0).abs() < 1e-9);
        assert_eq!(false_positive_rate(n, w, w, w + 1), Some(0.0));

        // The three quantities this module's documentation quotes, MEASURED here and attributed
        // to nothing but this arithmetic.
        let half = false_positive_rate(n, w, w, 20).unwrap();
        assert!(half > 2.4e-26 && half < 2.6e-26, "matching half a whole pattern: {half}");
        let all_of_a_subsample = false_positive_rate(n, w, 20, 20).unwrap();
        assert!(all_of_a_subsample > 2.1e-37 && all_of_a_subsample < 2.3e-37, "all of a 20-bit subsample: {all_of_a_subsample}");
        let half_of_a_subsample = false_positive_rate(n, w, 20, 10).unwrap();
        assert!(half_of_a_subsample > 3.8e-13 && half_of_a_subsample < 4.0e-13, "half of a subsample: {half_of_a_subsample}");
        // The ordering is the point, and it is not the obvious one: demanding ALL of a twenty-bit
        // subsample is rarer than demanding ANY twenty of the forty, because there are fewer ways
        // to do it. Relaxing the subsample's threshold to ten is what costs the robustness.
        assert!(all_of_a_subsample < half && half < half_of_a_subsample);
    }

    /// Half a unit in the last digit of a printed decimal: the tolerance a printed number allows.
    /// The tables are compared at `half_unit · (1 + 10⁻⁵)`, because one row sits ON a rounding
    /// boundary: `1/1024 = 0.0009765625` is printed `0.000976563`, exactly half a unit away, and the
    /// log-domain arithmetic lands `8 × 10⁻¹³` relative below the exact value — `8 × 10⁻¹⁶` absolute,
    /// which is `1.6 × 10⁻⁶` of that half unit.
    fn half_unit(printed: &str) -> f64 {
        let upper = printed.to_uppercase();
        let (mantissa, exponent) = upper.split_once('E').unwrap_or((upper.as_str(), "0"));
        let decimals = mantissa.split_once('.').map_or(0, |(_, d)| d.len());
        0.5 * 10f64.powi(exponent.parse::<i32>().unwrap() - decimals as i32)
    }

    /// The paper's Appendix A, Tables 1 and 2 (arXiv:1503.07469v1, pp. 15–16), every row, Table 4's
    /// nine rows with `θ = w`, and the
    /// three "one in N" figures of its text: each is this module's tail to within half a unit of
    /// the last digit printed. Table 1 is the exact match, `1/C(n, w)`; Table 2 is the whole-pattern
    /// tail at threshold `t`.
    #[test]
    fn the_papers_actuarial_tables_are_this_tail() {
        let table1: [(usize, usize, &str); 16] = [
            (64, 1, "0.015625"), (64, 3, "2.40015E-05"), (64, 5, "1.31156E-07"), (64, 7, "1.60975E-09"),
            (64, 9, "3.631E-11"), (64, 11, "1.34482E-12"), (512, 1, "0.001953125"), (512, 3, "4.49666E-08"),
            (512, 5, "3.47807E-12"), (512, 7, "5.69416E-16"), (512, 9, "1.61079E-19"), (1024, 1, "0.000976563"),
            (1024, 3, "5.60434E-09"), (1024, 5, "1.07629E-13"), (1024, 7, "4.35769E-18"), (1024, 9, "3.03651E-22"),
        ];
        for (n, w, printed) in table1 {
            let got = false_positive_rate(n, w, w, w).unwrap();
            let want: f64 = printed.parse().unwrap();
            assert!((got - want).abs() <= half_unit(printed) * (1.0 + 1e-5), "Table 1, n {n} w {w}: {got} against {printed}");
        }
        let table2: [(usize, usize, usize, &str); 16] = [
            (64, 4, 4, "1.57387E-06"), (64, 4, 3, "0.000379303"), (64, 4, 2, "0.017093815"), (64, 4, 1, "0.232525308"),
            (64, 8, 8, "2.25929E-10"), (64, 8, 7, "1.01442E-07"), (64, 8, 6, "9.84351E-06"), (64, 8, 5, "0.000360558"),
            (64, 8, 4, "0.006169265"), (64, 32, 32, "5.45666E-19"), (64, 32, 24, "6.70223E-05"), (64, 32, 16, "0.59857385"),
            (1024, 20, 20, "1.82484E-42"), (1024, 20, 17, "3.50023E-31"), (1024, 20, 14, "9.93621E-23"), (1024, 20, 10, "9.32924E-14"),
        ];
        for (n, w, t, printed) in table2 {
            let got = false_positive_rate(n, w, w, t).unwrap();
            let want: f64 = printed.parse().unwrap();
            assert!((got - want).abs() <= half_unit(printed) * (1.0 + 1e-5), "Table 2, n {n} w {w} t {t}: {got} against {printed}");
        }
        // §II.D: n = 1024, w = 4, θ = 2 — "one in 14,587".
        assert_eq!((1.0 / false_positive_rate(1024, 4, 4, 2).unwrap()).round(), 14_587.0);
        // §II.E: eight bits, a subsample of four, θ = 2 — "one in 3,142"; twenty, ten, five — "1 in
        // 2.5 million" (2,529,618 by this arithmetic).
        assert_eq!((1.0 / false_positive_rate(1024, 8, 4, 2).unwrap()).round(), 3_142.0);
        let million = 1.0 / false_positive_rate(1024, 20, 10, 5).unwrap();
        assert!((2.5e6..2.55e6).contains(&million), "{million}");
        // Table 4 (p. 17), its nine rows with θ = w: the union treated as a fixed set of its
        // expected size, eqs. (14)–(15). Its rows with θ < w do not follow eq. (14) — see
        // `union_false_positive_rate` — and are not fixtures.
        let table4: [(usize, usize, usize, &str); 9] = [
            (64, 4, 10, "0.043131941"), (64, 8, 10, "0.07104513"), (1024, 20, 20, "1.2532E-10"),
            (1024, 20, 30, "7.76674E-08"), (8192, 20, 60, "4.33389E-18"), (8192, 40, 80, "2.15567E-20"),
            (65536, 40, 80, "1.06052E-53"), (65536, 40, 1000, "2.4446E-14"), (65536, 40, 600, "2.86956E-21"),
        ];
        for (n, w, m, printed) in table4 {
            let got = union_false_positive_rate(n, w, m, w).unwrap();
            let want: f64 = printed.parse().unwrap();
            assert!((got - want).abs() <= half_unit(printed) * (1.0 + 1e-5), "Table 4, n {n} w {w} M {m}: {got} against {printed}");
        }
        // The half-unit helper itself, on the three shapes the tables use.
        assert_eq!(half_unit("0.015625"), 0.5e-6);
        assert!((half_unit("2.40015E-05") - 0.5e-10).abs() < 1e-25);
        assert!((half_unit("3.631E-11") - 0.5e-14).abs() < 1e-29);
    }

    #[test]
    fn the_arithmetic_holds_at_sizes_where_a_factorial_would_not() {
        // 100,000 bits: C(n, w) is past 10^2000, so this is done in logarithms or not at all.
        let (n, w) = (100_000usize, 2_000usize);
        // The mean overlap is w²/n = 40 and the spread about six, so the support is exhausted
        // well inside 120 — but not to the last bit: the far terms underflow, and 4e-10 of the
        // mass is lost with them. That is the price of computing each term through a logarithm,
        // and it is stated rather than hidden in a loose tolerance.
        let mass: f64 = (0..=120).map(|b| overlap_probability(n, w, w, b).unwrap()).sum();
        assert!((mass - 1.0).abs() < 1e-8 && mass < 1.0, "the distribution sums to {mass}");
        for b in [0usize, 40, 200, 2_000] {
            let p = overlap_probability(n, w, w, b).unwrap();
            assert!((0.0..=1.0).contains(&p) && p.is_finite(), "overlap {b} gave {p}");
        }
        assert_eq!(overlap_probability(n, w, w, w).unwrap(), 0.0, "an exact match by chance underflows to zero");
        let rate = false_positive_rate(n, w, w, 100).unwrap();
        assert!(rate.is_finite() && (0.0..=1.0).contains(&rate));
        // The logarithm of a binomial, against the exact small case and against symmetry.
        assert!((ln_choose(10, 3).unwrap() - 120f64.ln()).abs() < 1e-12);
        assert!((ln_choose(52, 5).unwrap() - 2_598_960f64.ln()).abs() < 1e-9);
        // C(30, 15), against the integer, to a tolerance that says where the exact branch has to
        // end. Stirling with two correction terms is short by about 1/(1260 k⁵) — 3e-11 at k = 30,
        // 7e-16 at k = 256 — so the cut-over is an accuracy decision, and this is the assertion
        // that holds it: summing thirty logarithms lands within 2e-14, Stirling would not.
        assert!((ln_choose(30, 15).unwrap() - 155_117_520f64.ln()).abs() < 1e-12);
        assert!((ln_choose(200, 3).unwrap() - 1_313_400f64.ln()).abs() < 1e-11);
        assert_eq!(ln_choose(5, 0), Some(0.0));
        for (n, k) in [(300u64, 7u64), (5000, 2500), (1_000_000, 3)] {
            let a = ln_choose(n, k).unwrap();
            let b = ln_choose(n, n - k).unwrap();
            assert!((a - b).abs() < 1e-9 * a.abs().max(1.0), "C({n},{k}) and its mirror differ");
        }
        assert_eq!(ln_choose(3, 4), None);
    }

    #[test]
    fn subsampling_and_noise_move_the_overlap_by_the_amounts_they_should() {
        let (n, w) = (2048usize, 40usize);
        let mut rng = Rng::new(11);
        let pattern = Sdr::random(n, w, &mut rng).unwrap();
        // A subsample of the pattern still matches the pattern completely.
        for s in [1usize, 10, 20, 40] {
            let part = pattern.subsample(s).unwrap();
            assert_eq!(part.w(), s);
            // The documented choice is the FIRST s bits in index order — both ends would satisfy
            // the two properties below, so the choice itself is what is pinned here.
            assert_eq!(part.on, pattern.on[..s], "a subsample must be the first s bits");
            assert_eq!(part.overlap(&pattern).unwrap(), s, "a subsample must match its own pattern");
            // And a random pattern matches it that rarely.
            let rate = false_positive_rate(n, w, s, s).unwrap();
            assert!(rate <= 1.0 && (s < 10 || rate < 1e-4), "s = {s} gives {rate}");
        }
        assert!(pattern.subsample(41).is_err());
        // Noise moves exactly `flips` bits: the overlap with the original falls by that much.
        for flips in [0usize, 1, 10, 40] {
            let noisy = pattern.with_noise(flips, &mut rng).unwrap();
            assert_eq!(noisy.w(), w, "noise must not change how many bits are on");
            assert_eq!(pattern.overlap(&noisy).unwrap(), w - flips, "{flips} flips");
        }
        // Every noisy copy has exactly w bits — again over many draws, because drawing the
        // replacements with replacement collides only a few percent of the time and the
        // constructor would quietly de-duplicate the collision away.
        let mut smallest = w;
        for _ in 0..2_000 {
            smallest = smallest.min(pattern.with_noise(10, &mut rng).unwrap().w());
        }
        assert_eq!(smallest, w, "a noisy copy came back with only {smallest} of {w} bits");
        // And the bit that is dropped is drawn UNIFORMLY from the pattern, which is what makes
        // this noise rather than a rule. With one flip, each of the forty bits should go about
        // fifty times in two thousand draws; a scheme that dropped whichever bit happened to be
        // last would send one of them nineteen hundred times.
        let mut dropped = vec![0usize; w];
        for _ in 0..2_000 {
            let noisy = pattern.with_noise(1, &mut rng).unwrap();
            let gone = pattern.on.iter().position(|b| !noisy.on.contains(b)).expect("nothing was dropped");
            dropped[gone] += 1;
        }
        let (low, high) = (*dropped.iter().min().unwrap(), *dropped.iter().max().unwrap());
        assert!(high < 3 * (2_000 / w) && low > 2_000 / w / 3, "the dropped bit is not uniform: {low} to {high} of an expected {}", 2_000 / w);
        assert!(pattern.with_noise(41, &mut rng).is_err());
        // Nowhere to move them to: a fully dense representation cannot be made noisy.
        let dense = Sdr::new(4, vec![0, 1, 2, 3]).unwrap();
        assert!(dense.with_noise(1, &mut rng).is_err());
    }

    #[test]
    fn a_union_holds_many_patterns_before_it_stops_telling_them_apart() {
        let (n, w) = (2048usize, 40usize);
        let mut rng = Rng::new(3);
        let patterns: Vec<Sdr> = (0..20).map(|_| Sdr::random(n, w, &mut rng).unwrap()).collect();
        let stored = Sdr::union(&patterns).unwrap();
        // Every stored pattern matches the union completely — that is what storing means.
        for p in &patterns {
            assert_eq!(stored.overlap(p).unwrap(), w);
        }
        // The union's size is close to the expected n(1 − (1 − w/n)^m).
        let expected = n as f64 * (1.0 - (1.0 - w as f64 / n as f64).powi(20));
        assert!((stored.w() as f64 - expected).abs() < 3.0 * expected.sqrt(), "{} bits against {expected}", stored.w());
        // And the false-positive rate grows with how much is stored, from negligible to certain.
        let mut last = 0.0;
        for m in [1usize, 10, 50, 200, 1000] {
            let rate = union_false_positive_rate(n, w, m, 20).unwrap();
            assert!(rate >= last, "m = {m} lowered the rate to {rate}");
            last = rate;
        }
        // MEASURED, and the interesting part is how sharply it turns: one pattern 2.5e-26, five
        // 2.8e-11, ten 3.0e-6, twenty 1.6e-2, fifty 0.97. The store is not gradually degraded, it
        // is safe and then it is not.
        assert!(union_false_positive_rate(n, w, 1, 20).unwrap() < 1e-24);
        assert!(union_false_positive_rate(n, w, 5, 20).unwrap() < 1e-10, "five patterns should still be safe");
        assert!(union_false_positive_rate(n, w, 10, 20).unwrap() < 1e-5);
        assert!(union_false_positive_rate(n, w, 50, 20).unwrap() > 0.9, "fifty should not be");
        assert!(union_false_positive_rate(n, w, 0, 20).is_none());
        assert!(union_false_positive_rate(0, 1, 1, 1).is_none());
    }

    #[test]
    fn bad_shapes_are_refused() {
        assert_eq!(Sdr::new(0, vec![]), Err(SdrError::BadShape { n: 0, w: 0 }));
        assert!(Sdr::new(MAX_BITS + 1, vec![]).is_err());
        assert_eq!(Sdr::new(4, vec![0, 4]), Err(SdrError::OutOfRange { index: 4, n: 4 }));
        // Indices are sorted and de-duplicated, so the same bit twice is one bit.
        let s = Sdr::new(8, vec![5, 1, 5, 3]).unwrap();
        assert_eq!(s.on, vec![1, 3, 5]);
        assert_eq!(s.w(), 3);
        let mut rng = Rng::new(1);
        assert_eq!(Sdr::random(8, 0, &mut rng), Err(SdrError::BadShape { n: 8, w: 0 }));
        assert_eq!(Sdr::random(8, 9, &mut rng), Err(SdrError::BadShape { n: 8, w: 9 }));
        assert!(Sdr::random(0, 1, &mut rng).is_err());
        let other = Sdr::new(9, vec![1]).unwrap();
        assert_eq!(s.overlap(&other), Err(SdrError::Mismatched { a: 8, b: 9 }));
        assert_eq!(Sdr::union(&[]), Err(SdrError::BadShape { n: 0, w: 0 }));
        assert_eq!(Sdr::union(&[s.clone(), other]), Err(SdrError::Mismatched { a: 8, b: 9 }));
        assert_eq!(Sdr::union(std::slice::from_ref(&s)).unwrap(), s);
        assert!(false_positive_rate(0, 1, 1, 1).is_none() && false_positive_rate(4, 5, 1, 1).is_none());
        assert!(SdrError::BadShape { n: 4, w: 5 }.to_string().contains("5 active bits out of 4"));
        assert!(SdrError::Mismatched { a: 8, b: 9 }.to_string().contains("8 and 9"));
        assert!(SdrError::OutOfRange { index: 4, n: 4 }.to_string().contains("bit 4"));
    }

    /// [`Sdr::overlap`] counts the size of the multiset intersection: a repeated index in the
    /// RECEIVER cannot be matched twice against one index in the other.
    ///
    /// The hole: every `overlap` fixture in this module reaches the walk through `Sdr::new`,
    /// `random`, `subsample`, `union` or `with_noise`, and all five end in `Sdr::new`, which
    /// `sort_unstable`s and `dedup`s. So the whole suite only ever hands the walk strictly
    /// increasing lists, and "after a match the left value is strictly greater than the right" is
    /// a property of the CONSTRUCTOR rather than of the type. `Sdr` is
    /// `pub struct Sdr { pub n: usize, pub on: Vec<usize> }` — the ascending-and-distinct promise
    /// on the field is the constructor's, not the type's — and `overlap` re-derives nothing, so a
    /// struct literal or a `push` after construction is one line of safe Rust away.
    #[test]
    fn overlap_cannot_match_one_bit_twice_against_a_repeat_in_the_receiver() {
        let once = Sdr::new(8, vec![3]).unwrap();
        let repeated = Sdr { n: 8, on: vec![3, 3] };
        assert_eq!(repeated.w(), 2, "the fixture is not the two-entry list it is meant to be");
        assert_eq!(repeated.overlap(&once).unwrap(), 1, "one bit of `once` was matched twice");
        // The same list reached the other way, by pushing onto the public field after a
        // constructor that would have de-duplicated it.
        let mut grown = Sdr::new(8, vec![3]).unwrap();
        grown.on.push(3);
        assert_eq!(grown.overlap(&once).unwrap(), 1);
        // The direction matters, and saying so is what makes the two lines above a measurement of
        // the walk rather than of the fixture: a repeat in the ARGUMENT alone cannot overcount,
        // because the cursor a match leaves behind is then the one the `Greater` branch advances.
        assert_eq!(once.overlap(&repeated).unwrap(), 1);
        assert_eq!(Sdr::new(8, vec![3, 3]).unwrap().on, vec![3], "the constructor stopped de-duplicating");
        // And the walk still terminates on a longer pair with repeats on both sides.
        let a = Sdr { n: 8, on: vec![1, 3, 3, 5, 5, 5, 7] };
        let b = Sdr { n: 8, on: vec![0, 3, 5, 5, 6] };
        assert_eq!(a.overlap(&b).unwrap(), 3, "3 once and 5 twice is the multiset intersection");
    }

    /// The upper end of the clamp in [`overlap_probability`] is unreachable, and this pins the
    /// two facts that make it so rather than the conclusion.
    ///
    /// The hole: `every_probability_is_in_range_at_cortical_sizes` asserts `(0.0..=1.0)`, which
    /// the clamp itself makes true — so no existing assertion can see whether it ever fires. What
    /// decides that is arithmetic, in two halves.
    ///
    /// (a) Where the support is a single point the three log-binomials cancel EXACTLY: `s == n`
    /// and `w == 0` force `b` and leave `X + 0 - X` with the same `X` from the same three
    /// [`ln_factorial`] calls, and `w == n` and `s == 0` leave `0 + 0 - 0`. The sum is `+0.0`,
    /// `exp` is 1.0, and there is nothing to round up.
    ///
    /// (b) Everywhere else the mode's probability is at most `1 - 1/n`. For any distribution
    /// `Var <= diam² · (1 - P(mode))`, because `Var <= E[(B - mode)²]`; the hypergeometric's
    /// variance is `w s (n-s)(n-w) / (n²(n-1))` and its support's diameter is at most
    /// `min(w, s, n-w, n-s)`, so that ratio is at least `1/(4(n-1))`. [`MAX_BITS`] caps `n` at
    /// `2²²`, so the true `ln` is at most `-5.96e-8` — and the exact minimum, by enumeration in
    /// rationals, is `1 - P = 1/n` attained at `w = s = 1`, i.e. `ln <= -2.38e-7`. Against that
    /// the computed `ln` carries the rounding of `x · ln(x)` at `x <= 2²²`, whose product lands
    /// in `[2²⁵, 2²⁶)` — one ulp is `2⁻²⁷ = 7.45e-9`. Measured at `n = MAX_BITS`, `w = s = 1`:
    /// computed `-2.30967998504638672e-7` against an exact `-2.38418607523276430e-7`, an error of
    /// exactly one `2⁻²⁷`, 32 times inside the margin.
    #[test]
    fn the_overlap_probability_never_reaches_the_clamps_upper_end() {
        // (a) the degenerate families, where the answer is 1 and the cancellation is exact.
        for n in [2u64, 3, 255, 256, 257, 65536, 1_000_000, MAX_BITS as u64] {
            for (w, s, b) in [(1, n, 1), (n, 1, 1), (0, n / 2, 0), (n / 2, 0, 0), (n, n, n)] {
                let ln = ln_choose(s, b).unwrap() + ln_choose(n - s, w - b).unwrap() - ln_choose(n, w).unwrap();
                assert_eq!(ln, 0.0, "n = {n}, w = {w}, s = {s}: a single-point support gave ln = {ln:e}");
                assert!(ln.is_sign_positive(), "n = {n}: the cancellation produced a negative zero");
                let p = overlap_probability(usize::try_from(n).unwrap(), usize::try_from(w).unwrap(), usize::try_from(s).unwrap(), usize::try_from(b).unwrap());
                assert_eq!(p, Some(1.0), "n = {n}, w = {w}, s = {s}, b = {b}");
            }
        }
        // (b) the extremal non-degenerate configuration, w = s = 1, where 1 - P is exactly 1/n
        // and every other configuration is further from 1.
        let rounding = (2.0f64).powi(-27);
        for n in [256u64, 65536, 1_000_000, MAX_BITS as u64] {
            let ln = ln_choose(1, 0).unwrap() + ln_choose(n - 1, 1).unwrap() - ln_choose(n, 1).unwrap();
            assert!(ln < -0.8 / n as f64, "n = {n}: ln = {ln:e} is above the bound -1/n");
            assert!(ln.abs() > 25.0 * rounding, "n = {n}: ln = {ln:e} is within 25 roundings of x ln x, {rounding:e}");
            let p = overlap_probability(usize::try_from(n).unwrap(), 1, 1, 0).unwrap();
            assert!(p < 1.0, "n = {n}: the mode's probability rounded up to {p}");
            assert_eq!(p, ln.exp(), "the clamp moved a value it is not supposed to reach");
        }
    }
}
