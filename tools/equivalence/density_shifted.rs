// Run: rustc -O tools/equivalence/density_shifted.rs -o /tmp/density_shifted && /tmp/density_shifted
// Backs the "equivalent" argument of the mutation "shifted early exit removed" in
// tools/mutations/density.json. Prints: 2000012 inputs, 0 disagreements.
// Fixture: `shifted` with and without its early exit, bit for bit, over the domain `integrate`
// calls it on: a finite power-of-two base with any shift, or f64::INFINITY with shift 0.
fn with_exit(mut x: f64, base: f64, shift: i64) -> f64 {
    let by = if shift > 0 { 1.0 / base } else { base };
    for _ in 0..shift.unsigned_abs() {
        if x == 0.0 || x.is_infinite() {
            break;
        }
        x *= by;
    }
    x
}
fn without_exit(x: f64, base: f64, shift: i64) -> f64 {
    let by = if shift > 0 { 1.0 / base } else { base };
    (0..shift.unsigned_abs()).fold(x, |x, _| x * by)
}
fn same(a: f64, b: f64) -> bool {
    (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits()
}
fn main() {
    // xorshift64*
    let mut s: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = || {
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        s.wrapping_mul(0x2545_F491_4F6C_DD1D)
    };
    let specials = [0.0, -0.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN, 5e-324, -5e-324, f64::MIN_POSITIVE, f64::MAX, -f64::MAX, 1.0, -1.0];
    let bases: Vec<f64> = [1, 8, 100, 500, 1000, 1023].iter().map(|&k| 2f64.powi(k)).collect();
    let (mut count, mut bad) = (0u64, 0u64);
    for i in 0..2_000_000u64 {
        let x = if i % 7 == 0 {
            specials[(next() % specials.len() as u64) as usize]
        } else {
            // any f64 bit pattern: every sign, exponent and mantissa, subnormals and NaNs included
            f64::from_bits(next())
        };
        let base = bases[(next() % bases.len() as u64) as usize];
        let shift = (next() % 4001) as i64 - 2000;
        count += 1;
        if !same(with_exit(x, base, shift), without_exit(x, base, shift)) {
            bad += 1;
        }
    }
    for &x in &specials {
        count += 1;
        if !same(with_exit(x, f64::INFINITY, 0), without_exit(x, f64::INFINITY, 0)) {
            bad += 1;
        }
    }
    println!("{count} inputs, {bad} disagreements");
}
