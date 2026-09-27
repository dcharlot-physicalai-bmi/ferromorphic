// ChowKopell::eta_at as written (`t <= 0.0` returns 0) and mutated (`t < 0.0`), compared bit for
// bit on 1,000,000 random (t, r, v_A, xi, Delta); a quarter of the t are exactly +0.0 or -0.0.
struct K { v_a: f64, xi: f64, width: f64 }
impl K {
    fn spike_part(&self, t: f64, r: f64) -> f64 { self.v_a / (r + self.xi) * ((self.xi * t).exp() - (-r * t).exp()) }
    fn v_m(&self) -> f64 { self.spike_part(self.width, 1.0) }
    fn tail(&self, r: f64) -> f64 { -(1.0 + self.v_m() - self.spike_part(self.width, r)) }
    fn as_written(&self, t: f64, r: f64) -> f64 {
        if t <= 0.0 { 0.0 } else if t <= self.width { self.spike_part(t, r) } else { self.tail(r) * (-r * (t - self.width)).exp() }
    }
    fn mutated(&self, t: f64, r: f64) -> f64 {
        if t < 0.0 { 0.0 } else if t <= self.width { self.spike_part(t, r) } else { self.tail(r) * (-r * (t - self.width)).exp() }
    }
}
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
    fn unit(&mut self) -> f64 { (self.next() >> 11) as f64 / (1u64 << 53) as f64 }
}
fn main() {
    let mut rng = Rng(0xD1B54A32D192ED03);
    let mut bad = 0u64;
    let n = 1_000_000u64;
    for _ in 0..n {
        let k = K { v_a: rng.unit() * 10.0, xi: 0.01 + rng.unit() * 200.0, width: 1e-4 + rng.unit() * 2.0 };
        let r = 0.01 + rng.unit() * 50.0;
        let t = match rng.next() % 8 { 0 => 0.0, 1 => -0.0, _ => (rng.unit() * 2.0 - 0.5) * 3.0 * k.width };
        let (a, b) = (k.as_written(t, r), k.mutated(t, r));
        if a.to_bits() != b.to_bits() && !(a.is_nan() && b.is_nan()) { bad += 1; }
    }
    println!("{n} samples, {bad} disagreements");
}
