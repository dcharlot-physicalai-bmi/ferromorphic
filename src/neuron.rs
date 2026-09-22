//! Spiking neuron models, each checked against the closed form it is supposed to reproduce.
//!
//! The models here are old and open. Lapicque described integrate-and-fire in 1907; the leaky
//! version is the standard textbook object (Gerstner & Kistler, *Spiking Neuron Models*, 2002);
//! Izhikevich's two-variable model is from *Simple Model of Spiking Neurons*, IEEE Transactions on
//! Neural Networks 14(6), 2003; adaptive thresholds are the mechanism behind long short-term memory
//! in spiking networks (Bellec et al., 2018). None of it is proprietary and none of it is new. What
//! a neuromorphic chip accelerates is exactly this loop, and what it charges for is the memory
//! traffic around it — so both belong in the open commons.
//!
//! # Units are SI at every interface
//!
//! [`Neuron::step`] takes `dt` in **seconds**, currents in **amperes**, potentials in **volts**.
//! This matters more than it looks: the spiking literature is written in milliseconds and
//! millivolts, and a crate that exposed both conventions would eventually integrate a millisecond
//! time step as a second and report a firing rate a thousand times wrong — in the right shape, with
//! the right units printed beside it.
//!
//! [`Izhikevich`] is the exception that proves it. Its constants — `0.04`, `5`, `140`, the 30 mV
//! cutoff — are dimensionally meaningless outside the paper's own millisecond/millivolt frame, so
//! they are kept **exactly as the paper prints them** where a reader can compare them line by line,
//! and the conversion happens at the boundary in [`Izhikevich::step`]. The alternative, rescaling
//! the constants into SI, would make them unrecognisable against the source.
//!
//! # What "verified" means here
//!
//! Every model in this module has a test that runs it against an analytic solution, not against a
//! previous run of itself. [`Lif`] integrates by **exponential Euler**, which is not an
//! approximation for piecewise-constant input — it is the exact solution of the membrane equation
//! over the step — so the simulated potential agrees with the closed form to floating-point noise
//! rather than to a discretisation error, and [`Lif::isi`] gives the inter-spike interval in closed
//! form for the constant-current case so a simulated firing rate can be checked against it.

/// A neuron that can be stepped forward in time.
///
/// One trait rather than an enum because a network holds one model type throughout: the memory
/// layout of a spiking simulation is the thing that determines its energy, and a `Vec<Box<dyn
/// Neuron>>` would scatter membrane state across the heap and charge a pointer chase per neuron per
/// step. See [`crate::net`], which stores state in parallel arrays for exactly this reason.
pub trait Neuron: Clone {
    /// Whether one step of length `k * dt` with zero input gives exactly the same state as `k`
    /// steps of `dt` with zero input.
    ///
    /// **This is the property that makes event-driven simulation legal**, and it is a constant on
    /// the trait rather than a comment because [`crate::sim`] refuses to run a model that lacks it
    /// in [`crate::sim::Mode::EventDriven`]. Skipping over quiet ticks is only free if the model
    /// can be jumped across them without changing the answer.
    ///
    /// True for [`Lif`], [`AdaptiveLif`] and [`IntegrateAndFire`]: the first two integrate by
    /// exponential Euler, which is the exact solution over any interval of constant input, and
    /// exponentials compose — `exp(-a/τ) · exp(-b/τ) = exp(-(a+b)/τ)`. The third is linear and does
    /// not move at all under zero current.
    ///
    /// **False for [`Izhikevich`]**, whose `v` equation is quadratic and is integrated by forward
    /// Euler. Two half-steps and one whole step of that give different answers, so jumping a gap
    /// would silently change the spike times — which is exactly the kind of error that produces a
    /// plausible raster plot and an unreproducible result.
    const EXACT_OVER_GAPS: bool;

    /// Whether [`Neuron::step_counted`] resolves spike times INSIDE the step, so that the number
    /// of spikes it reports does not depend on how time was cut into steps.
    ///
    /// True for [`Lif`] and [`AdaptiveLif`], whose threshold crossings have closed forms. False by
    /// default, and [`crate::sim::Sim::with_exact_timing`] refuses a model that lacks it: a
    /// simulation that says it keeps exact spike counts has to be running a model that can.
    const EXACT_TIMING: bool = false;

    /// Advance by `dt` seconds under input current `i` amperes. Returns `true` if the neuron
    /// spiked during this step.
    fn step(&mut self, dt: f64, i: f64) -> bool;

    /// Advance by `dt` seconds under input current `i` amperes and return HOW MANY spikes fell in
    /// the step. `None` if the model refuses the step (a non-finite current, say), in which case
    /// its state has not moved.
    ///
    /// The default is [`Neuron::step`] counted: at most one spike, recorded at the end of the
    /// step. A model with [`Neuron::EXACT_TIMING`] overrides it with its exact solver — more than
    /// one spike can fall in a step, and none is rounded up to the next.
    fn step_counted(&mut self, dt: f64, i: f64) -> Option<u32> {
        Some(u32::from(self.step(dt, i)))
    }

    /// Apply an instantaneous displacement of `dv` volts to the membrane.
    ///
    /// This is how a synapse delivers. The alternative — converting a weight into a current and
    /// passing it to `step` — makes the effect of a spike depend on `dt`, so halving the time step
    /// would halve every synaptic influence in the network while every parameter stayed the same.
    /// A delta synapse displaces the membrane by a voltage, and that is what this does.
    ///
    /// A refractory neuron IGNORES the displacement. That is what absolute refractoriness means,
    /// and a model that accumulated input during it would fire the instant the period ended,
    /// turning the refractory period into a delay line rather than a rate bound.
    fn bump(&mut self, dv: f64);

    /// Seconds of absolute refractory period still to run, or `0.0` for a model without one.
    ///
    /// Exposed because [`crate::sim`] has to land exactly on the end of a refractory period when it
    /// jumps a neuron across quiet ticks. Without it, an event-driven run and a clocked run of the
    /// same network disagree about when a neuron resumed integrating — by less than one tick, which
    /// is small enough never to be noticed and large enough to change a spike time.
    fn refractory_left(&self) -> f64 {
        0.0
    }

    /// Membrane potential in volts.
    fn potential(&self) -> f64;

    /// Return to the resting state, forgetting any refractory countdown and any adaptation.
    fn reset(&mut self);
}

/// Leaky integrate-and-fire.
///
/// `tau_m dV/dt = -(V - v_rest) + r_m * I`, with a spike and a reset when `V` reaches `v_th`, and
/// an absolute refractory period during which the potential is clamped at `v_reset`.
///
/// # The closed form this is checked against
///
/// Under constant current the membrane relaxes toward `v_inf = v_rest + r_m * I`. Starting from
/// `v_reset` it reaches threshold after
///
/// ```text
/// T = tau_m * ln( (v_inf - v_reset) / (v_inf - v_th) )
/// ```
///
/// and the inter-spike interval is `T + t_ref`. That expression is [`Lif::isi`], and
/// `examples/lif_closed_form.rs` compares it against a simulated spike train.
///
/// If `v_inf <= v_th` the neuron never fires however long you wait, and `isi` returns `None` rather
/// than a large number — the difference between "fires rarely" and "does not fire" is a difference
/// a rate-coded readout cannot recover later.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Lif {
    /// Membrane time constant, seconds. Sets how fast the potential forgets its input.
    pub tau_m: f64,
    /// Resting potential, volts. The value the membrane relaxes to with no input.
    pub v_rest: f64,
    /// Firing threshold, volts. A spike is emitted when the potential reaches it.
    pub v_th: f64,
    /// Post-spike potential, volts.
    pub v_reset: f64,
    /// Membrane resistance, ohms. Converts input current to the steady-state potential offset.
    pub r_m: f64,
    /// Absolute refractory period, seconds. The potential is held at `v_reset` for this long after
    /// a spike, which is what bounds the firing rate at `1 / t_ref`.
    pub t_ref: f64,
    /// Current membrane potential, volts.
    pub v: f64,
    /// Seconds remaining in the refractory period; zero when the neuron is free to integrate.
    pub refractory: f64,
}

impl Default for Lif {
    /// A cortical-ish default: 20 ms membrane constant, −65 mV rest, −50 mV threshold, 2 ms
    /// refractory, 10 MΩ. These are round numbers from the textbook range rather than a fit to any
    /// particular cell, and they are stated here so that a figure produced with the default is
    /// reproducible from the documentation alone.
    fn default() -> Self {
        Self {
            tau_m: 20e-3,
            v_rest: -65e-3,
            v_th: -50e-3,
            v_reset: -65e-3,
            r_m: 10e6,
            t_ref: 2e-3,
            v: -65e-3,
            refractory: 0.0,
        }
    }
}

impl Lif {
    /// The steady-state potential under constant current `i`, volts.
    #[must_use]
    pub fn v_inf(&self, i: f64) -> f64 {
        self.v_rest + self.r_m * i
    }

    /// Inter-spike interval under constant current `i`, in seconds, in closed form.
    ///
    /// `None` when `v_inf(i) <= v_th`: the neuron is sub-threshold and never fires, which is a
    /// different statement from a long interval and is kept different here.
    #[must_use]
    pub fn isi(&self, i: f64) -> Option<f64> {
        let v_inf = self.v_inf(i);
        if v_inf <= self.v_th {
            return None;
        }
        let t = self.tau_m * ((v_inf - self.v_reset) / (v_inf - self.v_th)).ln();
        Some(t + self.t_ref)
    }

    /// Steady-state firing rate under constant current `i`, in hertz.
    ///
    /// `None` for the sub-threshold case, for the same reason as [`Lif::isi`].
    #[must_use]
    pub fn rate(&self, i: f64) -> Option<f64> {
        self.isi(i).map(|t| 1.0 / t)
    }

    /// Advance by `dt` under constant current `i` with EXACT spike timing, returning the number of
    /// spikes emitted during the tick.
    ///
    /// [`Neuron::step`] decides once per tick whether the potential has reached threshold, so a
    /// spike is always recorded at the END of the tick in which the crossing happened, and the
    /// integration restarts from the next tick boundary. The time between the crossing and that
    /// boundary is lost — every interval is rounded UP to a whole number of ticks. A cell whose
    /// true interval is 3.3 ticks fires every 4: its rate is 17% low, and the error is systematic,
    /// so a population's decoded value is biased however many neurons it has. (`nef`'s spiking
    /// LMU measured the consequence: a state error of 47% at a 1 ms tick against 4.6% at 0.1 ms.)
    ///
    /// This method solves for the crossing time inside the tick,
    /// `t* = τ ln((V_∞ − V)/(V_∞ − V_th))`, resets there, serves the refractory period, and goes on
    /// integrating with what is left of the tick — so the spike COUNT over any run of constant
    /// current does not depend on the tick at all, and the rate is `1/isi` exactly. More than one
    /// spike can fall in a tick; hence the count.
    ///
    /// `None` for a `dt` that is not positive and finite, a non-finite current, or a cell that
    /// would fire infinitely often (`v_reset ≥ v_th` with no refractory period).
    #[must_use]
    pub fn step_exact(&mut self, dt: f64, i: f64) -> Option<u32> {
        self.exact(dt, i, |_| {})
    }

    /// [`Lif::step_exact`], also appending each spike's TIME within the tick — seconds from the
    /// tick's start, in order — to `offsets`. These are the spike times the membrane equation
    /// gives, not tick boundaries: under constant current they are `isi − t_ref`, then one every
    /// `isi`, whatever the tick.
    #[must_use]
    pub fn step_exact_times(&mut self, dt: f64, i: f64, offsets: &mut Vec<f64>) -> Option<u32> {
        self.exact(dt, i, |t| offsets.push(t))
    }

    fn exact(&mut self, dt: f64, i: f64, mut spike_at: impl FnMut(f64)) -> Option<u32> {
        if !(dt > 0.0) || !dt.is_finite() || !i.is_finite() || (self.v_reset >= self.v_th && !(self.t_ref > 0.0)) {
            return None;
        }
        let v_inf = self.v_inf(i);
        let mut left = dt;
        let mut spikes = 0u32;
        while left > 0.0 {
            if self.refractory > 0.0 {
                let served = self.refractory.min(left);
                self.refractory -= served;
                left -= served;
                self.v = self.v_reset;
                continue;
            }
            // Time to threshold from here; zero if a bump has already put the cell over it.
            let crossing = if self.v >= self.v_th {
                0.0
            } else if v_inf > self.v_th {
                self.tau_m * ((v_inf - self.v) / (v_inf - self.v_th)).ln()
            } else {
                f64::INFINITY
            };
            if crossing <= left {
                spikes = spikes.saturating_add(1);
                left -= crossing;
                spike_at(dt - left);
                self.v = self.v_reset;
                self.refractory = self.t_ref;
            } else {
                self.v = v_inf + (self.v - v_inf) * (-left / self.tau_m).exp();
                left = 0.0;
            }
        }
        Some(spikes)
    }
}

impl Neuron for Lif {
    // Exponential Euler is the exact solution over any interval of constant input, and exponentials
    // compose across concatenated intervals, so a gap may be jumped in one step.
    const EXACT_OVER_GAPS: bool = true;
    const EXACT_TIMING: bool = true;

    fn step_counted(&mut self, dt: f64, i: f64) -> Option<u32> {
        self.step_exact(dt, i)
    }

    fn step(&mut self, dt: f64, i: f64) -> bool {
        if self.refractory > 0.0 {
            // Clamped, not merely un-integrated: a refractory neuron on real hardware holds its
            // reset potential rather than drifting, and the distinction shows up as an offset in
            // the first interval after a burst.
            self.refractory -= dt;
            self.v = self.v_reset;
            return false;
        }
        // EXPONENTIAL EULER, which for constant `i` over the step is not an approximation but the
        // exact solution of `tau dV/dt = -(V - v_inf)`. Forward Euler would be the obvious choice
        // and is wrong here in a specific way: its error grows with `dt / tau_m` and it goes
        // unstable above `dt = 2 * tau_m`, so a user who coarsened the time step to save energy
        // would get a firing rate that drifted and then exploded, with no warning at either point.
        let v_inf = self.v_inf(i);
        let decay = (-dt / self.tau_m).exp();
        self.v = v_inf + (self.v - v_inf) * decay;

        if self.v >= self.v_th {
            self.v = self.v_reset;
            self.refractory = self.t_ref;
            return true;
        }
        false
    }

    fn bump(&mut self, dv: f64) {
        if self.refractory > 0.0 {
            return;
        }
        self.v += dv;
    }

    fn refractory_left(&self) -> f64 {
        self.refractory.max(0.0)
    }

    fn potential(&self) -> f64 {
        self.v
    }

    fn reset(&mut self) {
        self.v = self.v_rest;
        self.refractory = 0.0;
    }
}

/// Non-leaky integrate-and-fire: a perfect integrator.
///
/// `c * dV/dt = I`. No leak, so the potential is an exact linear ramp and the spike times are
/// exactly `k * c * (v_th - v_reset) / I`. That makes it the sharpest available test of a
/// simulator's timing — any drift in the loop shows up immediately as a drift in a quantity that
/// has no error term at all.
///
/// It is also the model several digital neuromorphic cores actually implement, because a leak costs
/// a multiply per neuron per tick and an integrator does not.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IntegrateAndFire {
    /// Membrane capacitance, farads.
    pub c: f64,
    /// Firing threshold, volts.
    pub v_th: f64,
    /// Post-spike potential, volts.
    pub v_reset: f64,
    /// Current membrane potential, volts.
    pub v: f64,
}

impl Default for IntegrateAndFire {
    fn default() -> Self {
        Self { c: 1e-9, v_th: 15e-3, v_reset: 0.0, v: 0.0 }
    }
}

impl IntegrateAndFire {
    /// Exact inter-spike interval under constant current `i`, seconds.
    ///
    /// `None` for `i <= 0`, where the potential never rises to threshold.
    #[must_use]
    pub fn isi(&self, i: f64) -> Option<f64> {
        if i <= 0.0 {
            return None;
        }
        Some(self.c * (self.v_th - self.v_reset) / i)
    }
}

impl Neuron for IntegrateAndFire {
    // Linear in `i * dt`, and motionless at `i = 0`: a gap changes nothing at all.
    const EXACT_OVER_GAPS: bool = true;

    fn step(&mut self, dt: f64, i: f64) -> bool {
        self.v += i * dt / self.c;
        if self.v >= self.v_th {
            self.v = self.v_reset;
            return true;
        }
        false
    }

    fn bump(&mut self, dv: f64) {
        self.v += dv;
    }

    fn potential(&self) -> f64 {
        self.v
    }

    fn reset(&mut self) {
        self.v = self.v_reset;
    }
}

/// Leaky integrate-and-fire with an adapting threshold.
///
/// The threshold relaxes toward `theta_0` with time constant `tau_a` and steps up by `beta` on
/// every spike, so a neuron driven hard fires quickly and then slows down. This is the mechanism
/// that gives spiking networks a memory longer than their membrane constant, and it is what
/// "adaptive LIF" or "ALIF" refers to in the surrogate-gradient literature.
///
/// The adaptation is on the THRESHOLD rather than as an adaptation current. Both forms exist; this
/// one is chosen because its state is a scalar that can be read directly, which makes the
/// closed-form check below possible.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdaptiveLif {
    /// The membrane, which behaves exactly as [`Lif`] does.
    pub lif: Lif,
    /// Resting threshold, volts — the value `theta` decays back to.
    pub theta_0: f64,
    /// Threshold adaptation time constant, seconds.
    pub tau_a: f64,
    /// Threshold increment per spike, volts.
    pub beta: f64,
    /// Current threshold, volts.
    pub theta: f64,
}

impl AdaptiveLif {
    /// Build from a membrane and an adaptation, taking the resting threshold from the membrane's
    /// own `v_th` so the two cannot disagree.
    #[must_use]
    pub fn new(lif: Lif, tau_a: f64, beta: f64) -> Self {
        Self { theta_0: lif.v_th, tau_a, beta, theta: lif.v_th, lif }
    }

    /// Threshold `n` spikes into a burst with no time to decay between them, volts.
    ///
    /// `theta_0 + n * beta`, which is trivial arithmetic and is exposed anyway because it is the
    /// quantity the test checks the simulator against — and because a reader looking for "how much
    /// does this thing adapt" should find the answer rather than derive it.
    #[must_use]
    pub fn theta_after(&self, n: u32) -> f64 {
        self.theta_0 + f64::from(n) * self.beta
    }

    /// The interspike interval the cell ADAPTS TO under constant current `i`, seconds: the period
    /// `T` of the regime in which the threshold's decay over one interval exactly undoes the
    /// increment of one spike. Just before each spike the threshold is then
    /// `θ₀ + β/(e^{T/τ_a} − 1)`, and `T` is where the membrane, charging for `T − t_ref` from reset,
    /// meets it:
    ///
    /// `V_∞ + (V_reset − V_∞) e^{−(T − t_ref)/τ_m} = θ₀ + β/(e^{T/τ_a} − 1)`,
    ///
    /// solved by bisection (the left side rises in `T` and the right side falls). `None` when
    /// `V_∞ ≤ θ₀`: the cell stops firing once it has adapted, or never starts.
    #[must_use]
    pub fn adapted_isi(&self, i: f64) -> Option<f64> {
        let v_inf = self.lif.v_inf(i);
        if !(v_inf > self.theta_0) || !v_inf.is_finite() || !(self.tau_a > 0.0) || !(self.beta >= 0.0) {
            return None;
        }
        let gap = |t: f64| {
            let charged = v_inf + (self.lif.v_reset - v_inf) * (-(t - self.lif.t_ref) / self.lif.tau_m).exp();
            charged - self.theta_0 - self.beta / (t / self.tau_a).exp_m1()
        };
        let mut lo = self.lif.t_ref.max(f64::MIN_POSITIVE);
        let mut hi = (lo + self.lif.tau_m).max(1e-6);
        let mut doublings = 0;
        while gap(hi) < 0.0 {
            hi *= 2.0;
            doublings += 1;
            if doublings > 200 {
                return None;
            }
        }
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            if gap(mid) < 0.0 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        Some(0.5 * (lo + hi))
    }

    /// [`Lif::step_exact`] for the adapting cell: the crossing of the rising membrane and the
    /// falling threshold has no closed form, so it is found inside the tick by bisection to the
    /// last bit, on a bracket chosen so that it holds exactly one crossing — the difference of two
    /// exponentials has at most one turning point, and the bracket ends there if it is the first
    /// place the membrane is over the threshold. Returns the spikes emitted in the tick.
    ///
    /// `None` for a `dt` that is not positive and finite, a non-finite current, or a cell that
    /// would fire infinitely often (no refractory period, no adaptation, reset at or above
    /// threshold).
    #[must_use]
    pub fn step_exact(&mut self, dt: f64, i: f64) -> Option<u32> {
        self.exact(dt, i, |_| {})
    }

    /// [`AdaptiveLif::step_exact`], also appending each spike's time within the tick to `offsets`.
    #[must_use]
    pub fn step_exact_times(&mut self, dt: f64, i: f64, offsets: &mut Vec<f64>) -> Option<u32> {
        self.exact(dt, i, |t| offsets.push(t))
    }

    fn exact(&mut self, dt: f64, i: f64, mut spike_at: impl FnMut(f64)) -> Option<u32> {
        let endless = !(self.lif.t_ref > 0.0) && !(self.beta > 0.0) && self.lif.v_reset >= self.theta.min(self.theta_0);
        if !(dt > 0.0) || !dt.is_finite() || !i.is_finite() || !(self.tau_a > 0.0) || endless {
            return None;
        }
        let v_inf = self.lif.v_inf(i);
        let (tau_m, tau_a, theta_0) = (self.lif.tau_m, self.tau_a, self.theta_0);
        let mut left = dt;
        let mut spikes = 0u32;
        while left > 0.0 {
            if self.lif.refractory > 0.0 {
                let served = self.lif.refractory.min(left);
                self.lif.refractory -= served;
                left -= served;
                self.lif.v = self.lif.v_reset;
                self.theta = theta_0 + (self.theta - theta_0) * (-served / tau_a).exp();
                continue;
            }
            let (v0, th0) = (self.lif.v, self.theta);
            // Membrane minus threshold, t seconds from here.
            let over = |t: f64| v_inf + (v0 - v_inf) * (-t / tau_m).exp() - theta_0 - (th0 - theta_0) * (-t / tau_a).exp();
            let crossing = if over(0.0) >= 0.0 {
                Some(0.0)
            } else {
                // Its one possible turning point: where the two exponentials' slopes are equal.
                let (a, b) = ((v0 - v_inf) / tau_m, (th0 - theta_0) / tau_a);
                let turn = if a * b > 0.0 && tau_m != tau_a { (a / b).ln() / (1.0 / tau_m - 1.0 / tau_a) } else { f64::NAN };
                let end = if turn > 0.0 && turn < left && over(turn) >= 0.0 {
                    Some(turn)
                } else if over(left) >= 0.0 {
                    Some(left)
                } else {
                    None
                };
                end.map(|mut hi| {
                    let mut lo = 0.0;
                    for _ in 0..200 {
                        let mid = 0.5 * (lo + hi);
                        if over(mid) < 0.0 {
                            lo = mid;
                        } else {
                            hi = mid;
                        }
                    }
                    hi
                })
            };
            let advance = crossing.unwrap_or(left);
            self.theta = theta_0 + (th0 - theta_0) * (-advance / tau_a).exp();
            if crossing.is_some() {
                spikes = spikes.saturating_add(1);
                left -= advance;
                spike_at(dt - left);
                self.lif.v = self.lif.v_reset;
                self.lif.refractory = self.lif.t_ref;
                self.theta += self.beta;
            } else {
                self.lif.v = v_inf + (v0 - v_inf) * (-left / tau_m).exp();
                left = 0.0;
            }
            if spikes > 10_000_000 {
                return None;
            }
        }
        self.lif.v_th = self.theta;
        Some(spikes)
    }
}

impl Neuron for AdaptiveLif {
    // Both state variables are exponential relaxations, and both compose across a gap.
    const EXACT_OVER_GAPS: bool = true;
    const EXACT_TIMING: bool = true;

    fn step_counted(&mut self, dt: f64, i: f64) -> Option<u32> {
        self.step_exact(dt, i)
    }

    fn step(&mut self, dt: f64, i: f64) -> bool {
        // The threshold decays whether or not the membrane is refractory. Adaptation is a slow
        // variable and freezing it during the refractory period would make the adaptation time
        // constant depend on the firing rate, which is not what the model says.
        self.theta = self.theta_0 + (self.theta - self.theta_0) * (-dt / self.tau_a).exp();
        self.lif.v_th = self.theta;
        let fired = self.lif.step(dt, i);
        if fired {
            self.theta += self.beta;
            self.lif.v_th = self.theta;
        }
        fired
    }

    fn bump(&mut self, dv: f64) {
        self.lif.bump(dv);
    }

    fn refractory_left(&self) -> f64 {
        self.lif.refractory_left()
    }

    fn potential(&self) -> f64 {
        self.lif.v
    }

    fn reset(&mut self) {
        self.lif.reset();
        self.theta = self.theta_0;
        self.lif.v_th = self.theta_0;
    }
}

/// Izhikevich's two-variable model.
///
/// From *Simple Model of Spiking Neurons*, IEEE Transactions on Neural Networks 14(6):1569–1572,
/// 2003:
///
/// ```text
/// v' = 0.04 v^2 + 5 v + 140 - u + I
/// u' = a (b v - u)
/// if v >= 30 mV:  v <- c,  u <- u + d
/// ```
///
/// with `v` in millivolts and `t` in milliseconds. The constants above are transcribed from the
/// paper and deliberately not rewritten: `0.04`, `5` and `140` have no meaning outside that frame,
/// and the only way to check them is to read them against the source.
///
/// # Why the substeps
///
/// The `v` equation is quadratic and its solution blows up in finite time, which is the mechanism
/// that produces the spike upstroke. Integrating it with one forward-Euler step of 1 ms overshoots
/// badly near the upstroke, so the paper's own reference code takes two half-steps of `v` per step
/// of `u`. [`Izhikevich::substeps`] carries that, defaulting to 2, and is exposed because a user
/// who wants tighter timing can raise it and one who wants it cheap on a microcontroller can see
/// exactly what they are trading.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Izhikevich {
    /// Recovery time scale, the paper's `a`, in reciprocal milliseconds.
    pub a: f64,
    /// Recovery sensitivity to `v`, the paper's `b`, dimensionless.
    pub b: f64,
    /// Post-spike reset value of `v`, millivolts.
    pub c: f64,
    /// Post-spike increment of `u`, the paper's `d`.
    pub d: f64,
    /// Membrane potential, millivolts.
    pub v: f64,
    /// Recovery variable.
    pub u: f64,
    /// Forward-Euler substeps of `v` per call to [`Neuron::step`]. See the type doc.
    pub substeps: u32,
}

impl Izhikevich {
    /// Regular spiking, the paper's Figure 2 parameters: `a = 0.02, b = 0.2, c = -65, d = 8`.
    ///
    /// The most common cortical excitatory type, and the one to reach for if you do not have a
    /// reason to reach for another.
    #[must_use]
    pub fn regular_spiking() -> Self {
        Self::new(0.02, 0.2, -65.0, 8.0)
    }

    /// Fast spiking, the paper's inhibitory interneuron: `a = 0.1, b = 0.2, c = -65, d = 2`.
    #[must_use]
    pub fn fast_spiking() -> Self {
        Self::new(0.1, 0.2, -65.0, 2.0)
    }

    /// Chattering / bursting, the paper's `a = 0.02, b = 0.2, c = -50, d = 2`.
    #[must_use]
    pub fn chattering() -> Self {
        Self::new(0.02, 0.2, -50.0, 2.0)
    }

    /// Build from the paper's four parameters, starting at rest.
    ///
    /// The initial state is the paper's own: `v = c`, `u = b * v`.
    #[must_use]
    pub fn new(a: f64, b: f64, c: f64, d: f64) -> Self {
        Self { a, b, c, d, v: c, u: b * c, substeps: 2 }
    }
}

impl Neuron for Izhikevich {
    // FALSE, and load-bearing. The `v` equation is quadratic and integrated by forward Euler, so
    // one step of `2h` and two steps of `h` do not agree. `sim` refuses to run this model
    // event-driven rather than producing spike times that depend on which ticks happened to be
    // quiet.
    const EXACT_OVER_GAPS: bool = false;

    /// `dt` is in **seconds** and `i` in amperes, like every other model here; both are converted
    /// at this boundary into the paper's milliseconds and its dimensionless current.
    ///
    /// The current conversion is the honest weak point of putting this model behind an SI trait.
    /// The paper's `I` is not an ampere — it is a number that produces the figures in the paper —
    /// so the scale chosen here is 1 nA to 1 unit of the paper's `I`, which makes `regular_spiking`
    /// fire at roughly the published rate for the published inputs. It is a CONVENTION, stated so
    /// that a reader comparing against the paper knows exactly what was done, and nothing in this
    /// crate depends on it being the only defensible choice.
    fn step(&mut self, dt: f64, i: f64) -> bool {
        let dt_ms = dt * 1e3;
        let i_paper = i * 1e9;
        let n = self.substeps.max(1);
        let h = dt_ms / f64::from(n);

        for _ in 0..n {
            self.v += h * (0.04 * self.v * self.v + 5.0 * self.v + 140.0 - self.u + i_paper);
            // Checked INSIDE the substep loop. Checking only after the last one lets `v` run past
            // the cutoff and into the quadratic's blow-up, where the next substep produces an
            // infinity and then a NaN that propagates silently through every downstream spike.
            if self.v >= 30.0 {
                self.v = self.c;
                self.u += self.d;
                return true;
            }
        }
        self.u += dt_ms * self.a * (self.b * self.v - self.u);
        false
    }

    /// `dv` is volts, converted here into the model's millivolts.
    fn bump(&mut self, dv: f64) {
        self.v += dv * 1e3;
    }

    /// Volts, converted from the model's millivolts so that the trait's contract holds.
    fn potential(&self) -> f64 {
        self.v * 1e-3
    }

    fn reset(&mut self) {
        self.v = self.c;
        self.u = self.b * self.c;
    }
}

#[cfg(test)]
mod counted {
    use super::{AdaptiveLif, Izhikevich, Lif, Neuron};

    /// [`Neuron::step_counted`] is the hook [`crate::sim`] runs on; the mutations that break it
    /// live in this file, so the test that catches them does too.
    #[test]
    fn the_counted_step_is_the_exact_solver_where_there_is_one_and_step_itself_where_there_is_not() {
        const { assert!(Lif::EXACT_TIMING && AdaptiveLif::EXACT_TIMING && !Izhikevich::EXACT_TIMING) };
        // The LIF: more than one spike in a tick, and the same state as `step_exact` throughout.
        let (mut counted, mut exact) = (Lif::default(), Lif::default());
        let mut total = 0;
        for _ in 0..50 {
            let n = counted.step_counted(20e-3, 4e-9).unwrap();
            assert_eq!(n, exact.step_exact(20e-3, 4e-9).unwrap());
            assert_eq!(counted, exact);
            total += n;
        }
        // DERIVED, not typed: from rest the first crossing is τ ln((V∞ − V_rest)/(V∞ − V_th))
        // and every one after it is an `isi`, so a tick of length `T` holds ⌊(T − first)/isi⌋ + 1.
        let (isi, first) = (Lif::default().isi(4e-9).unwrap(), 20e-3 * (40.0f64 / 25.0).ln());
        let in_one_tick = ((20e-3 - first) / isi).floor() as u32 + 1;
        assert!((total as f64 - 50.0 * 20e-3 / isi).abs() < 2.0, "{total} spikes in a second of ticks at an isi of {isi}");
        assert!(total > 50, "{total} spikes in 50 ticks: the fixture must fire more than once a tick");
        assert_eq!(Lif::default().step_counted(20e-3, 4e-9), Some(in_one_tick));
        assert_eq!(Lif::default().step_counted(20e-3, f64::NAN), None, "a refused step is None, not zero");
        // The adaptive cell, likewise.
        let proto = AdaptiveLif::new(Lif::default(), 100e-3, 2e-3);
        let (mut counted, mut exact) = (proto, proto);
        let mut total = 0;
        for _ in 0..50 {
            let n = counted.step_counted(20e-3, 4e-9).unwrap();
            assert_eq!(n, exact.step_exact(20e-3, 4e-9).unwrap());
            total += n;
        }
        assert!(total > 50, "{total} adaptive spikes in 50 ticks");
        // A model without EXACT_TIMING gets the default: `step`, counted, never more than one.
        let (mut a, mut b) = (Izhikevich::regular_spiking(), Izhikevich::regular_spiking());
        let mut spikes = 0;
        for _ in 0..2_000 {
            let n = a.step_counted(1e-3, 10.0).unwrap();
            assert!(n <= 1 && n == u32::from(b.step(1e-3, 10.0)));
            spikes += n;
        }
        assert!(spikes > 3, "the default hook was never seen to report a spike");
    }
}

#[cfg(test)]
mod tests {
    use super::{AdaptiveLif, IntegrateAndFire, Izhikevich, Lif, Neuron};

    /// Exact spike TIMES: under constant current they are `isi − t_ref`, then one every `isi` —
    /// the same list whatever the tick they were found in.
    #[test]
    fn exact_spike_times_are_the_closed_forms_at_any_tick() {
        let cell = Lif::default();
        let i = 4e-9;
        let isi = cell.isi(i).unwrap();
        // Ticks that divide the half second exactly, so that every run covers the same interval.
        for dt in [2e-4, 1e-3, 6.25e-3, 0.25] {
            let mut c = cell;
            let mut times = Vec::new();
            let mut offsets = Vec::new();
            for k in 0..(0.5f64 / dt).round() as usize {
                offsets.clear();
                let n = c.step_exact_times(dt, i, &mut offsets).unwrap();
                assert_eq!(n as usize, offsets.len());
                assert!(offsets.iter().all(|o| *o > 0.0 && *o <= dt) && offsets.windows(2).all(|p| p[0] < p[1]));
                times.extend(offsets.iter().map(|o| k as f64 * dt + o));
            }
            assert_eq!(times.len(), 44, "dt = {dt}");
            for (k, t) in times.iter().enumerate() {
                // Each time is rebuilt from a tick index and an offset: a few ulps of half a second.
                assert!((t - (isi - cell.t_ref + k as f64 * isi)).abs() < 1e-13, "dt = {dt}, spike {k}: {t}");
            }
        }
    }

    /// The adapting cell with exact timing settles into the interval its self-consistency equation
    /// gives, at any tick — and the equation is checked by substituting its root back.
    #[test]
    fn the_adapting_cell_settles_into_its_self_consistent_interval() {
        let cell = AdaptiveLif::new(Lif::default(), 0.1, 3e-3);
        let i = 4e-9;
        let t = cell.adapted_isi(i).unwrap();
        // Substituted back: the membrane after T − t_ref of charging meets θ₀ + β/(e^{T/τ_a} − 1).
        let v_inf = cell.lif.v_inf(i);
        let charged = v_inf + (cell.lif.v_reset - v_inf) * (-(t - 2e-3) / 20e-3).exp();
        assert!((charged - (-50e-3 + 3e-3 / ((t / 0.1).exp() - 1.0))).abs() < 1e-16);
        // Adaptation makes it SLOWER than the plain cell, and with no adaptation it is the plain cell.
        let plain = cell.lif.isi(i).unwrap();
        assert!(t > 1.5 * plain, "adapted interval {t}, unadapted {plain}");
        assert!((AdaptiveLif::new(Lif::default(), 0.1, 0.0).adapted_isi(i).unwrap() - plain).abs() < 1e-15);
        assert_eq!(cell.adapted_isi(1e-9), None, "V∞ = −55 mV never reaches θ₀");
        for dt in [1e-4, 1e-3, 6e-3] {
            let mut c = cell;
            let mut times = Vec::new();
            let mut offsets = Vec::new();
            for k in 0..(3.0f64 / dt).round() as usize {
                offsets.clear();
                c.step_exact_times(dt, i, &mut offsets).unwrap();
                times.extend(offsets.iter().map(|o| k as f64 * dt + o));
            }
            // The first interval is the plain cell's; thirty time constants later they are T.
            assert!((times[0] - (plain - 2e-3)).abs() < 1e-13, "dt = {dt}");
            let last: Vec<f64> = times.windows(2).rev().take(5).map(|p| p[1] - p[0]).collect();
            for gap in &last {
                assert!((gap - t).abs() < 1e-10, "dt = {dt}: late interval {gap}, the equation says {t}");
            }
            assert!(times.windows(2).take(6).all(|p| p[1] - p[0] < t), "the early intervals are shorter");
            // The threshold the struct reports is kept in step with the membrane's.
            assert_eq!(c.lif.v_th, c.theta);
        }
        // The tick-based step on the same cell at 1 ms settles on a LONGER interval: every one of
        // its intervals is a whole number of ticks.
        let mut ticked = cell;
        let mut spikes = Vec::new();
        for k in 0..3000 {
            if ticked.step(1e-3, i) {
                spikes.push(f64::from(k));
            }
        }
        let late = (spikes[spikes.len() - 1] - spikes[spikes.len() - 6]) / 5.0 * 1e-3;
        assert!(late > t + 0.3e-3, "tick-based late interval {late} against {t}");
        // A cell can fire on the REBOUND of its threshold: relaxing from −45 mV toward a
        // sub-threshold −52 mV while the threshold falls from −40 mV to −50 mV five times faster,
        // the membrane is over the threshold only for a while in the middle of a 50 ms tick —
        // membrane minus threshold is −5 mV at its start and −1.4 mV at its end. The crossing is
        // found from the turning point, not from the tick's end.
        let mut rebound = AdaptiveLif { lif: Lif { v: -45e-3, ..Lif::default() }, theta_0: -50e-3, tau_a: 5e-3, beta: 3e-3, theta: -40e-3 };
        let over = |t: f64| -2e-3 + 7e-3 * (-t / 20e-3f64).exp() - 10e-3 * (-t / 5e-3f64).exp();
        assert!(over(0.0) < 0.0 && over(10e-3) > 0.0 && over(50e-3) < 0.0);
        let (mut lo, mut hi) = (0.0f64, 10e-3f64);
        for _ in 0..100 {
            let mid = 0.5 * (lo + hi);
            if over(mid) < 0.0 { lo = mid } else { hi = mid }
        }
        let mut when = Vec::new();
        assert_eq!(rebound.step_exact_times(50e-3, 1.3e-9, &mut when), Some(1));
        assert!((when[0] - hi).abs() < 1e-15, "fired at {}, the crossing is at {hi}", when[0]);
        // Refusals.
        let mut bad = cell;
        assert_eq!(bad.step_exact(0.0, i), None);
        assert_eq!(bad.step_exact(1e-3, f64::NAN), None);
        let mut endless = AdaptiveLif::new(Lif { v_reset: -50e-3, t_ref: 0.0, ..Lif::default() }, 0.1, 0.0);
        assert_eq!(endless.step_exact(1e-3, i), None);
        // With adaptation the same reset-at-threshold cell is finite: each spike raises the bar.
        let mut climbing = AdaptiveLif::new(Lif { v_reset: -50e-3, t_ref: 0.0, ..Lif::default() }, 0.1, 3e-3);
        let n = climbing.step_exact(50e-3, i).unwrap();
        assert!((2..50).contains(&n), "{n} spikes");
    }

    /// With exact spike timing the spike count over a run of constant current does not depend on
    /// the tick, and is the closed form's: the first spike at `isi − t_ref`, then one every `isi`.
    /// The tick-based step, on the same run, loses spikes as the tick grows.
    #[test]
    fn exact_spike_timing_makes_the_count_independent_of_the_tick() {
        let cell = Lif::default();
        let i = 4e-9; // V_∞ = −25 mV: well over threshold
        let isi = cell.isi(i).unwrap();
        assert!((isi - (20e-3 * (40.0f64 / 25.0).ln() + 2e-3)).abs() < 1e-15);
        let run = 2.0;
        let want = ((run - (isi - cell.t_ref)) / isi).floor() as u32 + 1;
        assert_eq!(want, 175, "11.4 ms intervals for 2 s");
        let mut ticked_counts = Vec::new();
        let charge = isi - cell.t_ref;
        for dt in [1e-4, 3.7e-4, 1e-3, 2.5e-3, 8e-3, 0.5] {
            let steps = (run / dt).round() as usize;
            let (mut exact, mut ticked) = (cell, cell);
            let (mut n_exact, mut n_ticked) = (0u32, 0u32);
            for _ in 0..steps {
                n_exact += exact.step_exact(dt, i).unwrap();
                n_ticked += u32::from(ticked.step(dt, i));
            }
            // `steps · dt` is `run` only to rounding, and no spike falls within a microsecond of it.
            assert_eq!(n_exact, want, "dt = {dt}");
            // The tick-based step rounds BOTH phases of the interval up to whole ticks: charging
            // takes ⌈9.40007 ms / dt⌉ of them and the refractory period ⌈2 ms / dt⌉.
            let first = (charge / dt).ceil() as usize;
            let per = first + (cell.t_ref / dt - 1e-9).ceil() as usize;
            assert_eq!(n_ticked as usize, (steps - first) / per + 1, "dt = {dt}");
            ticked_counts.push(n_ticked);
        }
        // Which is 175 spikes at no tick at all — 9.40007 ms is 95 ticks of 0.1 ms, not 94 — and
        // half of them at 8 ms. (The first draft of this test TYPED these six numbers from a rough
        // idea of the rounding and got four of them wrong.)
        assert_eq!(ticked_counts, vec![174, 169, 166, 160, 83, 2]);
        // The two agree on the potential between spikes: one exact step of a sub-threshold current
        // is the same exponential.
        let (mut a, mut b) = (cell, cell);
        assert_eq!(a.step_exact(3e-3, 1e-9), Some(0));
        b.step(3e-3, 1e-9);
        assert_eq!(a.v, b.v);
        // A cell RELAXING toward a sub-threshold level from above it must not fire: from −52 mV
        // toward −60 mV the crossing-time formula's logarithm is of a number below one, a negative
        // "time", and only the check that V_∞ is over threshold keeps it from being taken as a
        // spike. (With the cell starting BELOW V_∞ the same formula gives NaN, which compares false
        // and hides the missing check — that mutation survived.)
        let mut relaxing = Lif { v: -52e-3, ..cell };
        assert_eq!(relaxing.step_exact(5e-3, 0.5e-9), Some(0));
        assert!((relaxing.v - (-60e-3 + 8e-3 * (-0.25f64).exp())).abs() < 1e-17);
        // A tick long enough to hold several spikes reports them all, and leaves the cell where the
        // closed form says: 50 ms holds spikes at 9.4, 20.8, 32.2 and 43.6 ms, then 2 ms of
        // refractoriness and 4.4 ms of charging.
        let mut burst = cell;
        assert_eq!(burst.step_exact(50e-3, i), Some(4));
        let since = 50e-3 - (4.0 * isi - cell.t_ref) - cell.t_ref;
        let v_inf = cell.v_inf(i);
        assert!((burst.v - (v_inf + (cell.v_reset - v_inf) * (-since / cell.tau_m).exp())).abs() < 1e-15);
        assert_eq!(burst.refractory, 0.0);
        // A bump over threshold fires at once, and the refusals.
        let mut bumped = cell;
        bumped.bump(20e-3);
        assert_eq!(bumped.step_exact(1e-4, 0.0), Some(1));
        assert_eq!(bumped.refractory, cell.t_ref - 1e-4);
        assert_eq!(Lif::default().step_exact(0.0, i), None);
        assert_eq!(Lif::default().step_exact(f64::NAN, i), None);
        assert_eq!(Lif::default().step_exact(1e-3, f64::INFINITY), None);
        let mut runaway = Lif { v_reset: -50e-3, t_ref: 0.0, ..Lif::default() };
        assert_eq!(runaway.step_exact(1e-3, i), None, "reset at threshold with no refractory period fires for ever");
        // The same cell WITH a refractory period is legal: from rest it first fires at 9.4 ms, and
        // then, reset onto its threshold, every 2 ms — 9.4, 11.4, … 19.4 ms in a 20 ms tick.
        let mut held = Lif { v_reset: -50e-3, ..Lif::default() };
        assert_eq!(held.step_exact(20e-3, i), Some(6));
    }

    /// The free membrane has an exact solution. This is the tightest check in the module, and it
    /// passes at floating-point noise rather than at a discretisation tolerance BECAUSE the
    /// integrator is exponential Euler — which is the property the comment in `step` claims.
    #[test]
    fn free_decay_matches_the_exponential_exactly() {
        // -55 mV: above rest so there is something to decay, and BELOW the -50 mV threshold so the
        // neuron does not fire on its first step. The first draft of this test started at -40 mV,
        // which is over threshold, and the neuron spiked and reset before decaying at all — the
        // test then compared a reset potential against an exponential and failed for the right
        // reason by luck.
        let mut n = Lif { v: -55e-3, ..Lif::default() };
        let v0 = n.v;
        let dt = 1e-4;
        for k in 1..=500 {
            n.step(dt, 0.0);
            let t = f64::from(k) * dt;
            let want = n.v_rest + (v0 - n.v_rest) * (-t / n.tau_m).exp();
            assert!(
                (n.v - want).abs() < 1e-12,
                "step {k}: simulated {} vs closed form {want}",
                n.v
            );
        }
    }

    /// A coarse step and a fine step must agree, because the exponential is exact at both. This is
    /// the property forward Euler would fail, and it is what lets a caller trade time resolution
    /// for energy without silently changing the answer.
    #[test]
    fn the_exponential_integrator_is_step_size_independent_on_free_decay() {
        let run = |dt: f64, steps: u32| {
            let mut n = Lif { v: -55e-3, ..Lif::default() };
            for _ in 0..steps {
                n.step(dt, 0.0);
            }
            n.v
        };
        let fine = run(1e-5, 5_000);
        let coarse = run(1e-2, 5);
        assert!((fine - coarse).abs() < 1e-12, "fine {fine} vs coarse {coarse}");
    }

    /// The simulated firing rate against `Lif::isi`, over a range of currents.
    #[test]
    fn the_firing_rate_matches_the_closed_form_interval() {
        let proto = Lif::default();
        for &na in &[2.0, 3.0, 5.0, 10.0, 20.0] {
            let i = na * 1e-9;
            let want = proto.isi(i).expect("supra-threshold by construction");
            let mut n = proto;
            let dt = 1e-6;
            let steps = 2_000_000; // 2 s
            let mut spikes = 0u32;
            let mut first = None;
            let mut last = 0.0;
            for k in 0..steps {
                if n.step(dt, i) {
                    let t = f64::from(k) * dt;
                    if first.is_none() {
                        first = Some(t);
                    }
                    last = t;
                    spikes += 1;
                }
            }
            assert!(spikes > 2, "{na} nA produced {spikes} spikes");
            // Measured from the FIRST spike, not from t = 0. The interval before the first spike
            // starts from `v_rest` rather than from `v_reset`, so including it would compare a
            // different quantity to the closed form — and with v_rest == v_reset in the default it
            // would pass anyway, which is how a check like this comes to be wrong and green.
            let got = (last - first.unwrap()) / f64::from(spikes - 1);
            let rel = (got - want).abs() / want;
            assert!(rel < 1e-3, "{na} nA: simulated ISI {got} vs closed form {want}");
        }
    }

    /// Sub-threshold means never, and `None` is the only honest way to say it.
    #[test]
    fn a_subthreshold_current_has_no_interval_rather_than_a_long_one() {
        let n = Lif::default();
        // v_inf = -65 mV + 10 MOhm * 1 nA = -55 mV, below the -50 mV threshold.
        assert!(n.isi(1e-9).is_none());
        assert!(n.rate(1e-9).is_none());
        let mut m = n;
        let mut spikes = 0u32;
        for _ in 0..1_000_000 {
            if m.step(1e-5, 1e-9) {
                spikes += 1;
            }
        }
        assert_eq!(spikes, 0, "sub-threshold neuron fired {spikes} times in 10 s");
    }

    /// The refractory period is what bounds the rate. Drive the neuron absurdly hard and the rate
    /// must converge to `1 / t_ref` rather than to the time step.
    #[test]
    fn the_refractory_period_caps_the_firing_rate() {
        let proto = Lif::default();
        let huge = 1e-3; // 1 mA; v_inf is 10 kV, which is nonsense physically and exactly the point
        let want = 1.0 / proto.t_ref;
        let got = proto.rate(huge).expect("supra-threshold");
        assert!((got - want).abs() / want < 0.02, "rate {got} vs cap {want}");
    }

    /// A perfect integrator has no error term at all, so this catches timing drift that the leaky
    /// model's tolerance would hide.
    #[test]
    fn the_perfect_integrator_spikes_at_exactly_the_predicted_times() {
        let proto = IntegrateAndFire::default();
        let i = 3e-9;
        let want = proto.isi(i).expect("positive current");
        // The interval here is 5 ms. `dt` has to be fine enough that tick quantisation sits far
        // below the tolerance AND coarse enough that several intervals fit in a test-sized run:
        // 1e-7 gives 50,000 ticks per interval and four intervals in 200,000 steps. The first
        // draft used 1e-9 and ran 2e6 steps, which is 2 ms — less than one interval — and the test
        // failed with "only 0 spikes" rather than with a timing error.
        let dt = 1e-7;
        let mut n = proto;
        let mut times = Vec::new();
        for k in 0..250_000u32 {
            if n.step(dt, i) {
                times.push(f64::from(k) * dt);
            }
        }
        assert!(times.len() >= 3, "only {} spikes", times.len());
        for w in times.windows(2) {
            let got = w[1] - w[0];
            assert!((got - want).abs() < 2.0 * dt, "interval {got} vs exact {want}");
        }
    }

    /// Constructing a neuron already at or above threshold is legal and fires on the first step.
    /// Found by a test that did it accidentally; pinned here so the behaviour is decided rather
    /// than incidental.
    #[test]
    fn a_neuron_built_above_threshold_fires_immediately() {
        let mut n = Lif { v: -40e-3, ..Lif::default() };
        assert!(n.step(1e-6, 0.0), "a supra-threshold membrane did not fire");
        assert!((n.v - n.v_reset).abs() < 1e-15);
    }

    #[test]
    fn a_perfect_integrator_with_no_current_never_fires() {
        let n = IntegrateAndFire::default();
        assert!(n.isi(0.0).is_none());
        assert!(n.isi(-1e-9).is_none());
    }

    /// Adaptation must actually adapt, and by the amount the model says.
    #[test]
    fn the_adaptive_threshold_rises_by_beta_per_spike() {
        let lif = Lif { t_ref: 0.0, ..Lif::default() };
        // tau_a far longer than the burst, so decay between spikes is negligible and the closed
        // form `theta_0 + n * beta` is the thing being tested.
        let mut n = AdaptiveLif::new(lif, 10.0, 2e-3);
        let theta0 = n.theta_0;
        let mut spikes = 0u32;
        for _ in 0..200_000 {
            if n.step(1e-6, 50e-9) {
                spikes += 1;
                if spikes == 4 {
                    break;
                }
            }
        }
        assert_eq!(spikes, 4, "expected four spikes, saw {spikes}");
        let want = n.theta_after(4);
        assert!(
            (n.theta - want).abs() < 1e-5,
            "theta {} vs theta_0 + 4 beta = {want} (theta_0 {theta0})",
            n.theta
        );
    }

    /// Adaptation must slow the neuron down; if it does not, the mechanism is decorative.
    #[test]
    fn adaptation_lengthens_successive_intervals() {
        let lif = Lif { t_ref: 0.0, ..Lif::default() };
        let mut n = AdaptiveLif::new(lif, 50e-3, 1e-3);
        let dt = 1e-6;
        let mut times = Vec::new();
        for k in 0..500_000u32 {
            if n.step(dt, 30e-9) {
                times.push(f64::from(k) * dt);
                if times.len() == 5 {
                    break;
                }
            }
        }
        assert_eq!(times.len(), 5);
        let iv: Vec<f64> = times.windows(2).map(|w| w[1] - w[0]).collect();
        for w in iv.windows(2) {
            assert!(w[1] > w[0], "intervals did not lengthen: {iv:?}");
        }
    }

    #[test]
    fn resetting_an_adaptive_neuron_forgets_its_adaptation() {
        let mut n = AdaptiveLif::new(Lif { t_ref: 0.0, ..Lif::default() }, 10.0, 2e-3);
        for _ in 0..50_000 {
            n.step(1e-6, 50e-9);
        }
        assert!(n.theta > n.theta_0);
        n.reset();
        assert!((n.theta - n.theta_0).abs() < 1e-15);
        assert!((n.lif.v_th - n.theta_0).abs() < 1e-15, "the membrane kept a stale threshold");
    }

    /// Izhikevich's regular-spiking cell must fire, must stay finite, and must never leave a NaN
    /// behind — the quadratic blow-up makes the last of those a real failure mode rather than a
    /// hypothetical one.
    #[test]
    fn izhikevich_regular_spiking_fires_and_stays_finite() {
        let mut n = Izhikevich::regular_spiking();
        let mut spikes = 0u32;
        for _ in 0..10_000 {
            if n.step(1e-4, 10e-9) {
                spikes += 1;
            }
            assert!(n.v.is_finite() && n.u.is_finite(), "v {} u {}", n.v, n.u);
        }
        assert!(spikes > 5, "regular spiking produced {spikes} spikes in 1 s");
    }

    /// The three published types must be distinguishable. Fast spiking is defined by firing faster
    /// than regular spiking at the same drive, and a model that lost that has lost the point of
    /// carrying four parameters.
    #[test]
    fn fast_spiking_fires_faster_than_regular_spiking_at_equal_drive() {
        let count = |mut n: Izhikevich| {
            let mut s = 0u32;
            for _ in 0..10_000 {
                if n.step(1e-4, 10e-9) {
                    s += 1;
                }
            }
            s
        };
        let rs = count(Izhikevich::regular_spiking());
        let fs = count(Izhikevich::fast_spiking());
        assert!(fs > rs, "fast spiking {fs} did not exceed regular spiking {rs}");
    }

    /// The cutoff check lives inside the substep loop precisely so that this cannot happen. Drive
    /// the cell far past anything reasonable and assert nothing became non-finite.
    #[test]
    fn a_violent_drive_does_not_produce_a_nan() {
        let mut n = Izhikevich::regular_spiking();
        for _ in 0..5_000 {
            n.step(1e-3, 1e-6);
            assert!(n.v.is_finite(), "v went non-finite");
            assert!(n.u.is_finite(), "u went non-finite");
        }
    }

    /// An adapting cell whose steady state only TOUCHES the resting threshold has no interspike
    /// interval, and [`AdaptiveLif::adapted_isi`] must refuse it from its `V∞ > θ₀` guard.
    ///
    /// The hole this fills: the suite's only sub-threshold case is `V∞ = −55 mV` against
    /// `θ₀ = −50 mV`, STRICTLY below, where `gap(t) → V∞ − θ₀ < 0` and the 200-doubling escape
    /// really is what refuses. At equality that limit is `0`, not negative, so the escape is
    /// unreachable: `gap` is exactly `+0.0` as soon as `e^{t/τ_a} − 1` overflows, `0.0 < 0.0` is
    /// false, the doubling loop stops after eight steps and 200 bisections hand back the overflow
    /// boundary as if it were an interval. Measured: with only the finiteness half of the guard
    /// left, this call returns `Some(3.548_913_564_466_92)` — 3.55 s of "interval" for a cell
    /// whose membrane never reaches its threshold.
    #[test]
    fn a_cell_whose_steady_state_only_touches_the_threshold_has_no_adapted_interval() {
        let cell = AdaptiveLif { lif: Lif::default(), theta_0: -65e-3, tau_a: 5e-3, beta: 3e-3, theta: -65e-3 };
        // V∞(0) = v_rest + r_m·0 = −65 mV, bit for bit θ₀: approached, never crossed.
        assert_eq!(cell.lif.v_inf(0.0), cell.theta_0);
        assert_eq!(cell.adapted_isi(0.0), None, "V∞ = θ₀ is reached only in the limit");
        // And the bracket search cannot be what refuses. `hi` starts at t_ref + τ_m and doubles;
        // the eighth doubling is past 3.549 s, where t/τ_a exceeds ln(f64::MAX) and expm1 is +∞.
        // v_reset = V∞ here, so the charging term is exactly zero and gap is +0.0 — not `< 0.0`.
        let hi = (cell.lif.t_ref + cell.lif.tau_m) * 256.0;
        let v_inf = cell.lif.v_inf(0.0);
        let charged = v_inf + (cell.lif.v_reset - v_inf) * (-(hi - cell.lif.t_ref) / cell.lif.tau_m).exp();
        assert_eq!(charged, v_inf);
        assert_eq!((hi / cell.tau_a).exp_m1(), f64::INFINITY, "hi = {hi}");
        assert_eq!(charged - cell.theta_0 - cell.beta / (hi / cell.tau_a).exp_m1(), 0.0);
    }

}
