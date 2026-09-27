// Both versions of gapjunction::bisect, with and without the no-progress break, on 1,000,000
// random brackets [lo, hi] and monotone predicates `x < threshold` that meet the module's contract
// (below(lo) holds, below(hi) does not). Brackets span 1e-300..1e300 in scale, both signs, and a
// fifth of the thresholds sit within one ulp of an end.
fn with_break(below: impl Fn(f64) -> bool, mut lo: f64, mut hi: f64) -> f64 {
    for _ in 0..1100 {
        let m = 0.5 * (lo + hi);
        if m <= lo || m >= hi {
            break;
        }
        if below(m) { lo = m; } else { hi = m; }
    }
    hi
}
fn without_break(below: impl Fn(f64) -> bool, mut lo: f64, mut hi: f64) -> f64 {
    for _ in 0..1100 {
        let m = 0.5 * (lo + hi);
        if below(m) { lo = m; } else { hi = m; }
    }
    hi
}
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
    fn unit(&mut self) -> f64 { (self.next() >> 11) as f64 / (1u64 << 53) as f64 }
}
fn main() {
    let mut rng = Rng(0x9E3779B97F4A7C15);
    let (mut n, mut bad) = (0u64, 0u64);
    while n < 1_000_000 {
        let scale = 10f64.powf(rng.unit() * 600.0 - 300.0);
        let a = (rng.unit() * 2.0 - 1.0) * scale;
        let b = a + rng.unit() * scale * 2.0 + f64::MIN_POSITIVE;
        if !(b > a) || !a.is_finite() || !b.is_finite() { continue; }
        let th = match rng.next() % 5 {
            0 => f64::from_bits(a.to_bits().wrapping_add(if a >= 0.0 { 1 } else { u64::MAX })), // one ulp above lo
            1 => b,                                                                            // exactly hi
            _ => a + (b - a) * rng.unit(),
        };
        // the contract: below(lo) true, below(hi) false
        let below = |x: f64| x < th;
        if !(below(a) && !below(b)) { continue; }
        n += 1;
        if with_break(below, a, b).to_bits() != without_break(below, a, b).to_bits() { bad += 1; }
    }
    println!("{n} brackets, {bad} disagreements");
}
