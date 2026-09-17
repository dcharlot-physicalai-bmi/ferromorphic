//! Deterministic pseudorandom numbers, because a spike train has to be reproducible.
//!
//! A spiking network is a stochastic object almost everywhere: Poisson encoders draw spikes,
//! initial membrane potentials are jittered, synaptic transmission is probabilistic on some
//! hardware. If any of that reaches for the operating system's entropy, two runs of the same
//! experiment produce different spike trains and no result in this crate can be checked against
//! any other. So there is exactly one source of randomness here, it takes a seed, and it produces
//! the same stream on every platform this crate compiles for — including `wasm32`, which has no
//! OS entropy to reach for in the first place.
//!
//! # What this is
//!
//! PCG-XSH-RR 64/32, from O'Neill, *PCG: A Family of Simple Fast Space-Efficient Statistically
//! Good Algorithms for Random Number Generation* (2014), <https://www.pcg-random.org/paper.html>.
//! The multiplier `6364136223846793005` is the constant as the paper prints it, and it is left
//! unformatted so it can be compared against the paper by eye.
//!
//! It is not cryptographic and does not claim to be. What it claims is a long period, a passing
//! `TestU01` `BigCrush` record in the source literature, and — the property this crate actually
//! needs — bit-identical output for a given seed on every target.

/// A seeded PCG32 stream.
///
/// Same seed, same sequence, every platform and every release of this crate. That last clause is a
/// promise about the algorithm, not just the implementation: changing the generator would change
/// every spike train in every downstream experiment, so it is a breaking change and is treated as
/// one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rng {
    state: u64,
    inc: u64,
}

impl Rng {
    /// PCG's multiplier, transcribed from the paper.
    const MULT: u64 = 6364136223846793005;

    /// Start a stream from `seed`.
    ///
    /// Two streams from different seeds are independent for every practical purpose; two from the
    /// same seed are identical, which is the point. The odd `inc` is required by the algorithm —
    /// an even increment collapses the period — and is derived from the seed so that a caller
    /// cannot accidentally supply a degenerate one.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        // The stream selector must be odd. `| 1` on a value derived from the seed by an
        // odd-multiplier mix keeps distinct seeds in distinct streams without asking the caller to
        // know any of this.
        let inc = (seed.wrapping_mul(0x9E3779B97F4A7C15) ^ 0xDA3E_39CB_94B9_5BDB) | 1;
        let mut r = Self { state: 0, inc };
        r.next_u32();
        r.state = r.state.wrapping_add(seed);
        r.next_u32();
        r
    }

    /// The next 32 bits.
    pub fn next_u32(&mut self) -> u32 {
        let old = self.state;
        self.state = old.wrapping_mul(Self::MULT).wrapping_add(self.inc);
        // XSH-RR: xorshift the high bits down, then rotate by a count taken from the top 5 bits.
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        let rot = (old >> 59) as u32;
        xorshifted.rotate_right(rot)
    }

    /// A uniform draw in `[0, 1)`.
    ///
    /// Built from 53 bits so that every representable `f64` in the interval is reachable and the
    /// result is never exactly `1.0` — which matters because `-ln(1 - u)` is the inverse-CDF path
    /// used by the Poisson encoder, and `u == 1.0` there is an infinity.
    pub fn next_f64(&mut self) -> f64 {
        let hi = u64::from(self.next_u32()) >> 5; // 27 bits
        let lo = u64::from(self.next_u32()) >> 6; // 26 bits
        ((hi << 26) | lo) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// A uniform integer in `[0, n)`, without modulo bias.
    ///
    /// # Panics
    ///
    /// Never for `n >= 1`. `n == 0` has no value to return and panics rather than inventing one:
    /// there is no "empty" neuron index, and returning 0 would silently address a real neuron.
    pub fn below(&mut self, n: u32) -> u32 {
        assert!(n > 0, "below(0) has no value to return");
        // Rejection against the largest multiple of n that fits, which is the standard debiasing
        // and is cheap because rejection is rare for the n this crate uses (neuron counts).
        let zone = u32::MAX - (u32::MAX % n) - (n - 1);
        loop {
            let v = self.next_u32();
            if v <= zone {
                return v % n;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Rng;

    /// The property the whole crate rests on.
    #[test]
    fn the_same_seed_gives_the_same_stream() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        for _ in 0..1000 {
            assert_eq!(a.next_u32(), b.next_u32());
        }
    }

    #[test]
    fn different_seeds_give_different_streams() {
        let mut a = Rng::new(1);
        let mut b = Rng::new(2);
        let diff = (0..64).filter(|_| a.next_u32() != b.next_u32()).count();
        assert!(diff > 60, "only {diff} of 64 draws differed");
    }

    /// `next_f64` feeds `-ln(1 - u)`, where `u == 1.0` is an infinity. 53-bit construction from
    /// two 32-bit draws cannot reach 1.0, and this asserts it over a long run rather than assuming
    /// it from the arithmetic.
    #[test]
    fn uniforms_stay_inside_the_half_open_unit_interval() {
        let mut r = Rng::new(7);
        for _ in 0..200_000 {
            let u = r.next_f64();
            assert!((0.0..1.0).contains(&u), "u = {u} escaped [0, 1)");
        }
    }

    #[test]
    fn uniforms_have_about_the_right_mean_and_variance() {
        let mut r = Rng::new(9);
        let n = 200_000;
        let (mut s, mut s2) = (0.0f64, 0.0f64);
        for _ in 0..n {
            let u = r.next_f64();
            s += u;
            s2 += u * u;
        }
        let mean = s / f64::from(n);
        let var = s2 / f64::from(n) - mean * mean;
        // Uniform[0,1) has mean 1/2 and variance 1/12. The tolerance is ~6 standard errors of the
        // mean at this n, which is loose enough never to flake and tight enough to catch a
        // generator that is actually broken.
        assert!((mean - 0.5).abs() < 0.004, "mean {mean}");
        assert!((var - 1.0 / 12.0).abs() < 0.004, "var {var}");
    }

    /// Modulo bias is the classic silent defect here: it would skew which neurons get drawn, and
    /// a skew of a few percent is invisible in any plot anyone would make.
    #[test]
    fn below_is_unbiased_across_a_non_power_of_two() {
        let mut r = Rng::new(11);
        let n = 7u32;
        let mut counts = [0u32; 7];
        let draws = 140_000;
        for _ in 0..draws {
            counts[r.below(n) as usize] += 1;
        }
        let expect = f64::from(draws) / f64::from(n);
        for (i, &c) in counts.iter().enumerate() {
            let rel = (f64::from(c) - expect).abs() / expect;
            assert!(rel < 0.05, "bucket {i} off by {:.1}%", rel * 100.0);
        }
    }

    #[test]
    #[should_panic(expected = "below(0)")]
    fn below_zero_panics_rather_than_addressing_neuron_zero() {
        Rng::new(1).below(0);
    }
}
