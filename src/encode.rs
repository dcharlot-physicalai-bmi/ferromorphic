//! Turning numbers into spikes, and back.
//!
//! A spiking network cannot read a float. Something has to decide how a measurement becomes a
//! pattern of events, and that decision is not a detail — it fixes how much information survives,
//! how long the network must wait before it can answer, and how many synaptic operations the answer
//! costs. The three schemes here span that trade-off rather than implementing one and calling it
//! the encoding.
//!
//! | scheme | spikes for one value | latency | what it preserves |
//! |---|---|---|---|
//! | [`RateEncoder`] | many, Poisson | a whole window | magnitude, in the mean |
//! | [`LatencyEncoder`] | exactly one | one spike | magnitude, in the timing |
//! | [`DeltaEncoder`] | one per threshold crossing | immediate | change, not level |
//!
//! **Rate coding is the expensive one and the default in most of the literature.** Representing a
//! value to one part in a hundred needs on the order of `10^4` spikes, because a Poisson count's
//! relative error falls as `1/sqrt(n)` — so a rate-coded input layer generates the synaptic
//! operations that a neuromorphic energy figure is then divided by. [`LatencyEncoder`] emits one
//! spike per value and carries the magnitude in *when* it arrives, which is the scheme that makes
//! the energy argument work and the scheme that is hardest to train.
//!
//! **[`DeltaEncoder`] is what an event camera actually does.** It reports crossings of a relative
//! threshold and emits nothing at all for a static input, which is why an event sensor staring at a
//! still scene costs nothing and why it cannot tell you what it is looking at until something moves.

use crate::rng::Rng;
use crate::spike::{Event, Polarity, Spike, Train};

/// Poisson rate coding: a value becomes a firing rate, and the spikes are drawn.
///
/// The value `x` in `[0, 1]` maps linearly to a rate in `[0, max_hz]`, and each tick emits a spike
/// with probability `1 - exp(-rate * dt)`.
///
/// # Why the exponential rather than `rate * dt`
///
/// `rate * dt` is the first term of that expansion and is the form most implementations use. It is
/// fine while `rate * dt` is small and wrong when it is not: at `rate * dt = 1.5` it is a
/// probability greater than one, which either panics or silently clamps to "spike every tick", and
/// the encoder's rate then saturates at `1/dt` while reporting that it is producing `max_hz`. The
/// exact form is one `exp` and cannot do that.
#[derive(Debug, Clone)]
pub struct RateEncoder {
    /// Firing rate in hertz corresponding to an input of `1.0`.
    pub max_hz: f64,
    /// Tick length in seconds.
    pub dt: f64,
    rng: Rng,
}

impl RateEncoder {
    /// Build with a seed, so the same input produces the same spikes.
    #[must_use]
    pub fn new(max_hz: f64, dt: f64, seed: u64) -> Self {
        Self { max_hz, dt, rng: Rng::new(seed) }
    }

    /// Probability of a spike in one tick for input `x`, clamped to `[0, 1]`.
    #[must_use]
    pub fn p_spike(&self, x: f64) -> f64 {
        let rate = self.max_hz * x.clamp(0.0, 1.0);
        1.0 - (-rate * self.dt).exp()
    }

    /// Encode a vector of values into a train over `ticks` ticks.
    ///
    /// Value `i` drives source `i`.
    pub fn encode(&mut self, x: &[f64], ticks: u64) -> Train {
        let mut out = Vec::new();
        for t in 0..ticks {
            for (i, &v) in x.iter().enumerate() {
                if self.rng.next_f64() < self.p_spike(v) {
                    out.push(Spike { t, source: i as u32 });
                }
            }
        }
        Train::from_spikes(out)
    }

    /// The number of ticks needed for a rate estimate to reach relative standard error `rel`.
    ///
    /// A Poisson count of mean `n` has standard deviation `sqrt(n)`, so `rel = 1/sqrt(n)` and
    /// `n = 1/rel^2` spikes are needed — then the ticks follow from the rate. `None` for an input
    /// of zero, which never produces a spike and so never reaches any precision at all.
    ///
    /// This is exposed because it is the number that decides whether rate coding is affordable, and
    /// it is almost never stated beside a rate-coded result.
    #[must_use]
    pub fn ticks_for_precision(&self, x: f64, rel: f64) -> Option<u64> {
        let rate = self.max_hz * x.clamp(0.0, 1.0);
        if rate <= 0.0 || rel <= 0.0 {
            return None;
        }
        let n = 1.0 / (rel * rel);
        Some((n / (rate * self.dt)).ceil() as u64)
    }
}

/// Time-to-first-spike coding: one spike per value, and the value is in the delay.
///
/// A larger input spikes sooner. `x = 1.0` spikes at tick 0 and `x = 0.0` does not spike within the
/// window at all, which is the honest encoding of "no evidence" — a value of zero has nothing to
/// say and saying nothing is cheaper than saying it late.
///
/// # The cost, stated
///
/// One spike carries the value, so the synaptic-operation count of an input layer falls by whatever
/// factor the rate encoder's window would have been — typically three or four orders of magnitude.
/// What is paid for it is that the code is not differentiable in any convenient way and is
/// sensitive to jitter: a spike delayed by one tick is a different number, where in a rate code it
/// is noise.
#[derive(Debug, Clone, Copy)]
pub struct LatencyEncoder {
    /// Ticks in the coding window. `x = 1.0` lands at 0 and `x` just above 0 lands at `window - 1`.
    pub window: u64,
}

impl LatencyEncoder {
    /// Build with a window length in ticks.
    #[must_use]
    pub fn new(window: u64) -> Self {
        Self { window }
    }

    /// The tick at which `x` spikes, or `None` when it does not spike in the window.
    #[must_use]
    pub fn tick_of(&self, x: f64) -> Option<u64> {
        if !(x > 0.0) || self.window == 0 {
            return None;
        }
        let x = x.min(1.0);
        let t = ((1.0 - x) * (self.window - 1) as f64).round() as u64;
        Some(t.min(self.window - 1))
    }

    /// Encode a vector; value `i` drives source `i`.
    #[must_use]
    pub fn encode(&self, x: &[f64]) -> Train {
        let mut out = Vec::new();
        for (i, &v) in x.iter().enumerate() {
            if let Some(t) = self.tick_of(v) {
                out.push(Spike { t, source: i as u32 });
            }
        }
        Train::from_spikes(out)
    }

    /// Recover the value a spike at `t` encodes — the exact inverse of [`LatencyEncoder::tick_of`]
    /// up to the rounding that tick quantisation imposes.
    #[must_use]
    pub fn value_of(&self, t: u64) -> f64 {
        if self.window <= 1 {
            return 1.0;
        }
        1.0 - (t as f64) / (self.window - 1) as f64
    }
}

/// Delta modulation: report changes, not levels. What an event camera does.
///
/// The encoder holds a reference level per channel and emits an [`Polarity::On`] event when the
/// signal rises `threshold` above it, an [`Polarity::Off`] event when it falls `threshold` below,
/// and moves the reference to the crossing each time. A constant input produces nothing after the
/// first sample, which is the whole point and also the whole limitation.
///
/// # A large step emits every crossing it passed
///
/// A jump of `5 * threshold` emits five events, not one. This is what the hardware does and it
/// matters: an implementation that emitted one event per sample regardless of size would silently
/// compress large transients, which is exactly the part of the signal a change detector exists to
/// report. The behaviour is bounded by [`DeltaEncoder::max_events_per_sample`] so that a
/// discontinuity cannot emit an unbounded burst and stall a downstream network.
#[derive(Debug, Clone)]
pub struct DeltaEncoder {
    /// Change required to emit one event.
    pub threshold: f64,
    /// The most events one sample may emit, however large its jump.
    ///
    /// A real sensor has a finite readout bandwidth and drops what will not fit; a simulation with
    /// no cap turns a step edge into an arbitrarily long burst at one timestamp, which is not what
    /// any sensor does and which propagates as a single enormous current into the first layer.
    pub max_events_per_sample: u32,
    reference: Vec<f64>,
    started: bool,
}

impl DeltaEncoder {
    /// Build for `channels` channels.
    #[must_use]
    pub fn new(channels: usize, threshold: f64, max_events_per_sample: u32) -> Self {
        Self {
            threshold,
            max_events_per_sample,
            reference: vec![0.0; channels],
            started: false,
        }
    }

    /// Feed one sample of every channel at tick `t`, returning the events it produced.
    ///
    /// The first sample sets the reference and emits nothing: there is no change to report against
    /// a level nobody has seen before, and emitting the absolute value would make the first frame
    /// the only one that carried a level.
    pub fn sample(&mut self, t: u64, x: &[f64]) -> Vec<Event> {
        let mut out = Vec::new();
        if !self.started {
            self.reference.copy_from_slice(x);
            self.started = true;
            return out;
        }
        for (c, &v) in x.iter().enumerate() {
            let mut n = 0u32;
            while n < self.max_events_per_sample {
                let d = v - self.reference[c];
                if d >= self.threshold {
                    self.reference[c] += self.threshold;
                    out.push(Event { t, address: c as u32, polarity: Polarity::On });
                } else if d <= -self.threshold {
                    self.reference[c] -= self.threshold;
                    out.push(Event { t, address: c as u32, polarity: Polarity::Off });
                } else {
                    break;
                }
                n += 1;
            }
        }
        out
    }

    /// The reference level of each channel — the encoder's reconstruction of the signal.
    #[must_use]
    pub fn reference(&self) -> &[f64] {
        &self.reference
    }
}

/// Recover a value from a spike count over a window.
///
/// The inverse of [`RateEncoder`], and deliberately a free function rather than a `Decoder` type:
/// there is no state to carry, and a type would suggest otherwise.
#[must_use]
pub fn rate_decode(count: u64, ticks: u64, dt: f64, max_hz: f64) -> Option<f64> {
    if ticks == 0 || dt <= 0.0 || max_hz <= 0.0 {
        return None;
    }
    Some((count as f64 / (ticks as f64 * dt)) / max_hz)
}

#[cfg(test)]
mod tests {
    use super::{DeltaEncoder, LatencyEncoder, RateEncoder, rate_decode};
    use crate::spike::Polarity;

    /// The encoder must produce the rate it advertises, and the round trip must recover the value.
    #[test]
    fn rate_coding_round_trips_within_its_own_error_bar() {
        let dt = 1e-4;
        let max_hz = 200.0;
        let ticks = 200_000; // 20 s, long enough that the Poisson error is well under the tolerance
        for &x in &[0.1, 0.25, 0.5, 0.9] {
            let mut e = RateEncoder::new(max_hz, dt, 17);
            let tr = e.encode(&[x], ticks);
            let got = rate_decode(tr.len() as u64, ticks, dt, max_hz).unwrap();
            // Expected count, and a six-sigma band on a Poisson count of that mean.
            let n = x * max_hz * ticks as f64 * dt;
            let tol = 6.0 * n.sqrt() / n;
            assert!((got - x).abs() / x < tol, "x {x}: decoded {got}, tolerance {tol}");
        }
    }

    /// The defect the exact form exists to avoid: at a high rate and a coarse tick, `rate * dt`
    /// exceeds one and the encoder saturates while claiming it did not.
    /// The defect avoided, and the arithmetic that shows it.
    ///
    /// At `rate * dt = 100` the naive `rate * dt` form yields the probability 100, which either
    /// panics against a uniform draw or clamps to "spike every tick" while the encoder goes on
    /// reporting that it produces `max_hz`. The exact form cannot exceed one.
    ///
    /// It DOES reach exactly 1.0, and that is correct rather than a bug: `1 - exp(-100)` is
    /// `1 - 3.7e-44`, and the nearest `f64` to that is 1.0. A spike every tick is the right answer
    /// when the rate is a hundred times the tick frequency. The first draft of this test asserted
    /// `p < 1.0` and was asserting a floating-point accident, not a property.
    #[test]
    fn the_spike_probability_never_exceeds_one_however_fast_the_rate() {
        let e = RateEncoder::new(10_000.0, 1e-2, 1);
        let naive = e.max_hz * e.dt;
        assert!(naive > 1.0, "the test is not exercising the defect: naive form gave {naive}");
        let p = e.p_spike(1.0);
        assert!(p <= 1.0, "p = {p} is not a probability");
        assert!(p > 0.999_999, "p should be essentially certain, was {p}");
        // And at a sane operating point the two forms agree to FIRST order, which is why the naive
        // one survives in so much published code. They differ at second order by `(rate*dt)^2 / 2`,
        // which at rate*dt = 0.01 is 5e-5 — a 0.5% relative error in the spike probability, quietly
        // present in every rate-coded result that uses the naive form. Asserted at exactly that
        // size rather than "small", so the number is on the page.
        let sane = RateEncoder::new(100.0, 1e-4, 1);
        let x = sane.max_hz * sane.dt;
        let exact = sane.p_spike(1.0);
        // `x - (1 - e^-x) = x^2/2 - x^3/6 + x^4/24 - ...`. Two terms of that, asserted at 1e-9,
        // where the first neglected term is `x^4/24 = 4.2e-10`. The single-term version of this
        // assertion failed at 1e-7 because `x^3/6 = 1.67e-7` is exactly the residual — which is a
        // better demonstration of the point than the test that was trying to make it.
        let series = x * x / 2.0 - x * x * x / 6.0;
        assert!(
            ((x - exact) - series).abs() < 1e-9,
            "naive {x} minus exact {exact} did not match the expansion {series}"
        );
        // The headline: the naive form is 0.5% high at a perfectly ordinary operating point.
        let rel = (x - exact) / exact;
        assert!((rel - 0.005).abs() < 5e-4, "relative overstatement was {rel}");
    }

    /// The number that decides whether rate coding is affordable.
    #[test]
    fn the_precision_cost_of_rate_coding_is_reported_and_is_large() {
        let e = RateEncoder::new(100.0, 1e-3, 1);
        // 1% relative error needs 10,000 spikes; at 100 Hz that is 100 s, which at 1 ms ticks is
        // 100,000 ticks. This is the arithmetic that makes latency coding interesting.
        let ticks = e.ticks_for_precision(1.0, 0.01).unwrap();
        assert_eq!(ticks, 100_000, "got {ticks}");
        assert!(e.ticks_for_precision(0.0, 0.01).is_none(), "zero input reaches no precision");
    }

    #[test]
    fn rate_coding_is_deterministic_for_a_seed() {
        let mut a = RateEncoder::new(300.0, 1e-3, 99);
        let mut b = RateEncoder::new(300.0, 1e-3, 99);
        assert_eq!(a.encode(&[0.4, 0.7], 500).spikes(), b.encode(&[0.4, 0.7], 500).spikes());
    }

    /// Latency coding: one spike per value, monotone, and invertible.
    #[test]
    fn latency_coding_emits_exactly_one_spike_per_positive_value() {
        let e = LatencyEncoder::new(100);
        let tr = e.encode(&[1.0, 0.5, 0.0, 0.25]);
        assert_eq!(tr.len(), 3, "the zero must not spike");
        assert_eq!(e.tick_of(1.0), Some(0), "the largest value is the earliest");
        assert_eq!(e.tick_of(0.0), None);
        // Monotone: a bigger value never spikes later.
        let a = e.tick_of(0.8).unwrap();
        let b = e.tick_of(0.3).unwrap();
        assert!(a < b, "0.8 spiked at {a}, 0.3 at {b}");
    }

    #[test]
    fn latency_coding_inverts_to_within_one_tick() {
        let e = LatencyEncoder::new(1_000);
        for &x in &[0.05, 0.3, 0.5, 0.77, 1.0] {
            let t = e.tick_of(x).unwrap();
            let back = e.value_of(t);
            assert!((back - x).abs() <= 1.0 / 999.0 + 1e-12, "x {x} -> tick {t} -> {back}");
        }
    }

    /// A change detector staring at a still scene must cost nothing. This is the property the whole
    /// event-sensor energy argument rests on.
    /// Seven mutations of this module survived its first recorded audit, and all seven lived in
    /// the edges: what an out-of-range input does, which source a value drives, and where the two
    /// ends of the latency window fall. Those are the cases here.
    #[test]
    fn the_coding_windows_two_ends_and_what_lies_outside_them() {
        let e = LatencyEncoder::new(10);
        // The brightest value lands on the first tick, and the dimmest that still spikes lands on
        // the LAST tick of the window, not one past it.
        assert_eq!(e.tick_of(1.0), Some(0));
        assert_eq!(e.tick_of(1e-12), Some(9));
        assert_eq!(e.tick_of(0.5), Some(5));
        // The interior is where the span shows. A window of ten has NINE gaps, so a quarter-bright
        // value sits at round(0.75 * 9) = 7; spreading it over ten gaps would put it at 8, and the
        // two ends alone cannot tell those apart because the clamp hides the difference there.
        assert_eq!(e.tick_of(0.25), Some(7));
        assert_eq!(e.tick_of(0.1), Some(8));
        assert_eq!(e.tick_of(0.9), Some(1));
        // Nothing may land outside the window, whatever it is handed.
        for x in [1.0, 2.0, 1e9, f64::INFINITY] {
            let t = e.tick_of(x).expect("a positive value spikes");
            assert!(t < 10, "x = {x} landed on tick {t} of a ten-tick window");
        }
        // And a value above one is CLIPPED, not extrapolated: it is as bright as bright gets.
        assert_eq!(e.tick_of(2.0), e.tick_of(1.0));
        assert_eq!(e.tick_of(0.0), None);
        assert_eq!(e.tick_of(-1.0), None);
        assert_eq!(LatencyEncoder::new(0).tick_of(0.5), None);
        // Decoding spans the same ends: the last tick decodes to zero, not to 1/window.
        assert_eq!(e.value_of(0), 1.0);
        assert_eq!(e.value_of(9), 0.0);
        // The interior divides by `window - 1`, so a ten-tick window's tick 5 is 1 - 5/9, and an
        // ELEVEN-tick window puts the midpoint exactly at one half.
        assert!((e.value_of(5) - (1.0 - 5.0 / 9.0)).abs() < 1e-15, "{}", e.value_of(5));
        assert!((LatencyEncoder::new(11).value_of(5) - 0.5).abs() < 1e-15);
        assert_eq!(LatencyEncoder::new(1).value_of(0), 1.0);
    }

    #[test]
    fn value_i_drives_source_i_and_an_input_past_one_is_no_faster_than_one() {
        // Two values an order of magnitude apart, so which source carries which is unmistakable.
        let (dt, max_hz, ticks) = (1e-4, 400.0, 40_000u64);
        let mut e = RateEncoder::new(max_hz, dt, 5);
        let tr = e.encode(&[0.05, 0.8], ticks);
        let (slow, fast) = (tr.of(0).len(), tr.of(1).len());
        assert!(fast > 8 * slow, "source 0 fired {slow} times and source 1 {fast}");
        assert_eq!(slow + fast, tr.len(), "every spike belongs to one of the two sources");
        assert!(slow > 10, "source 0 never fired, so the comparison is vacuous");
        // The rate saturates at max_hz: an input above one is clamped before it sets the rate, so
        // it cannot ask for a probability the tick cannot deliver.
        assert_eq!(e.p_spike(2.0), e.p_spike(1.0));
        assert_eq!(e.p_spike(1e9), e.p_spike(1.0));
        assert_eq!(e.p_spike(-1.0), e.p_spike(0.0));
        assert_eq!(e.p_spike(0.0), 0.0);
        // The precision cost rounds UP: asking for a tick and a half of counting means two ticks,
        // because a fractional tick buys nothing.
        let slow_e = RateEncoder::new(1.0, 1.0, 1);
        assert_eq!(slow_e.ticks_for_precision(1.0, 1.0), Some(1));
        // 1/rel^2 = 2.25 spikes at one spike a tick: three ticks, not two.
        assert_eq!(slow_e.ticks_for_precision(1.0, 1.0 / 1.5), Some(3));
        assert_eq!(slow_e.ticks_for_precision(0.0, 0.1), None);
        assert_eq!(slow_e.ticks_for_precision(1.0, 0.0), None);
    }

    #[test]
    fn a_static_signal_produces_no_events_at_all() {
        let mut e = DeltaEncoder::new(2, 0.1, 8);
        assert!(e.sample(0, &[0.5, -0.5]).is_empty(), "the first sample only sets the reference");
        for t in 1..1_000 {
            assert!(e.sample(t, &[0.5, -0.5]).is_empty(), "a still input emitted an event at {t}");
        }
    }

    /// A large jump emits every crossing it passed, up to the cap.
    #[test]
    fn a_large_step_emits_one_event_per_threshold_crossed() {
        let mut e = DeltaEncoder::new(1, 0.1, 100);
        e.sample(0, &[0.0]);
        let ev = e.sample(1, &[0.55]);
        assert_eq!(ev.len(), 5, "0.55 / 0.1 is five whole crossings, got {}", ev.len());
        assert!(ev.iter().all(|x| x.polarity == Polarity::On));
        // The reference tracks the crossings, not the raw signal: 5 * 0.1, with 0.05 left over.
        assert!((e.reference()[0] - 0.5).abs() < 1e-12, "reference {}", e.reference()[0]);
    }

    #[test]
    fn a_fall_emits_off_events() {
        let mut e = DeltaEncoder::new(1, 0.25, 100);
        e.sample(0, &[1.0]);
        let ev = e.sample(1, &[0.4]);
        assert_eq!(ev.len(), 2);
        assert!(ev.iter().all(|x| x.polarity == Polarity::Off));
    }

    /// The cap is what stops a discontinuity becoming an unbounded burst.
    #[test]
    fn the_per_sample_cap_bounds_a_discontinuity() {
        let mut e = DeltaEncoder::new(1, 0.01, 4);
        e.sample(0, &[0.0]);
        let ev = e.sample(1, &[1_000.0]);
        assert_eq!(ev.len(), 4, "the cap did not hold");
    }

    #[test]
    fn decoding_refuses_a_window_of_no_length() {
        assert!(rate_decode(10, 0, 1e-3, 100.0).is_none());
        assert!(rate_decode(10, 5, 0.0, 100.0).is_none());
        assert!(rate_decode(10, 5, 1e-3, 0.0).is_none());
    }
}
