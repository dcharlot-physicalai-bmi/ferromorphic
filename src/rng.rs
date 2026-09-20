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
        // Rejection against the largest multiple of `n` that fits in 32 bits: the accepted range
        // is `0..=u32::MAX - 2^32 % n`, whose SIZE is exactly a multiple of `n`, and that exactness
        // is what makes `v % n` unbiased rather than nearly so.
        //
        // `2^32 % n` cannot be written directly in `u32`, so it is built from `u32::MAX % n`. An
        // earlier version of this line subtracted `n - 1` instead, which left an accepted count of
        // `(k − 1)n + 2` — two values too many for every `n > 2`, and, worse, an accepted count of
        // exactly TWO once `n` passed `2^31`, where the loop below then spun about two billion
        // times per draw. Nothing in the crate called it with an `n` that large, which is the only
        // reason it was never seen.
        let excess = (u32::MAX % n).wrapping_add(1) % n; // 2^32 mod n
        let zone = u32::MAX - excess;
        // The accepted region is never smaller than half the word — `2^32 % n` is below `n`, and
        // below `2^31` either way — so a hundred rejections in a row has probability under
        // `2^-100` and cannot happen to a correct zone. It CAN happen to a wrong one, and the
        // wrong one this replaced would have spun about two billion times per draw at large `n`.
        // A loop that cannot say why it is not finishing is worse than one that stops.
        for _ in 0..100 {
            let v = self.next_u32();
            if v <= zone {
                return v % n;
            }
        }
        panic!("below({n}) rejected a hundred draws in a row: the acceptance zone is wrong")
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

    /// The promise this module's own documentation opens with — "same seed, same sequence, every
    /// platform and every release" — is the one property no other test here can see. Every other
    /// test asks whether the stream is WELL BEHAVED, and a different generator would be well
    /// behaved too: a changed multiplier, a changed rotation, a changed seeding mix would all pass
    /// them. These are the actual values, computed by a separate implementation of PCG32 XSH-RR
    /// written from the algorithm rather than from this code. They are what makes changing the
    /// generator a visible, breaking change instead of a silent one.
    #[test]
    fn the_stream_is_this_exact_sequence_and_not_merely_a_well_behaved_one() {
        for (seed, want) in [
            (0u64, [3_469_696_627u32, 1_581_262_666, 1_615_719_374, 3_412_491_734]),
            (42, [86_690_733, 2_594_090_985, 4_127_782_882, 867_463_249]),
            (1, [1_296_297_154, 1_900_379_881, 40_361_581, 826_615_500]),
        ] {
            let mut r = Rng::new(seed);
            let got: [u32; 4] = core::array::from_fn(|_| r.next_u32());
            assert_eq!(got, want, "seed {seed}");
        }
        // And the uniform built from two of those draws, to the last bit it carries.
        let mut r = Rng::new(42);
        for want in [0.020_184_260_636_447_85, 0.961_074_346_318_632_5, 0.331_237_497_802_463_5] {
            let got = r.next_f64();
            assert!((got - want).abs() < 1e-16, "{got} against {want}");
        }
    }

    /// `below` rejects down to a whole number of blocks of `n`. Two things hide whether it does:
    /// at the small `n` this crate actually uses the leftover bias is about `2n/2^32` — a few parts
    /// per billion, which no counting test can see — and a bound that rejects TOO much looks
    /// correct rather than slow. A large `n` exposes both at once: here the accepted region is
    /// three quarters of the word, one third of the outputs would be drawn twice as often if the
    /// rejection were dropped, and a bound that accepted only two values would take about two
    /// billion draws to return even once.
    #[test]
    fn below_rejects_to_a_whole_number_of_blocks_even_when_n_is_most_of_the_word() {
        let n = 3u32 << 30; // 3,221,225,472 — two thirds of the u32 range is a single block
        let mut r = Rng::new(3);
        let third = n / 3;
        let mut counts = [0u32; 3];
        let draws = 60_000;
        for _ in 0..draws {
            let v = r.below(n);
            assert!(v < n, "below({n}) returned {v}");
            counts[(v / third) as usize] += 1;
        }
        let expect = f64::from(draws) / 3.0;
        for (i, &c) in counts.iter().enumerate() {
            let rel = (f64::from(c) - expect).abs() / expect;
            assert!(rel < 0.05, "third {i} off by {:.1}% — {c} of {draws}", rel * 100.0);
        }
        // The extremes of the range are reachable at all, which a bound that collapsed to a couple
        // of values would not manage.
        let mut r = Rng::new(4);
        let (mut low, mut high) = (false, false);
        for _ in 0..10_000 {
            let v = r.below(u32::MAX);
            low |= v < u32::MAX / 4;
            high |= v > u32::MAX / 4 * 3;
        }
        assert!(low && high, "below(u32::MAX) did not cover its range");
        // n = 1 has one answer, and n = 2 splits evenly.
        let mut r = Rng::new(5);
        assert!((0..100).all(|_| r.below(1) == 0));
        let ones = (0..20_000).filter(|_| r.below(2) == 1).count();
        assert!((9_600..10_400).contains(&ones), "{ones} of 20,000");
    }

    #[test]
    #[should_panic(expected = "below(0)")]
    fn below_zero_panics_rather_than_addressing_neuron_zero() {
        Rng::new(1).below(0);
    }
}
