//! Synaptic plasticity: the learning rules, each with the closed form it is checked against.
//!
//! A synapse that cannot change is a wire. Everything a brain-like system learns, it learns by
//! changing the number that multiplies a spike on its way from one cell to another — so a
//! neuromorphic library that ships neurons and no plasticity ships the inference half of the idea
//! and calls it the idea.
//!
//! # The lesson
//!
//! **Hebb's postulate** (Hebb, *The Organization of Behavior*, Wiley, 1949) is the whole field in
//! one sentence: cells that fire together, wire together. Written down, that is
//! `dw/dt = eta * x * y` — presynaptic activity times postsynaptic activity. It is also, written
//! down, **unstable**. The rule is positive feedback with no opposing term, so the weight vector's
//! length grows exponentially and every synapse saturates. [`Hebbian`] here is deliberately left
//! that way so the divergence is visible; it refuses with an error when it overflows rather than
//! returning an infinity.
//!
//! The field has three separate answers to that instability, and this module implements all three,
//! because they are not interchangeable:
//!
//! 1. **Normalise the weight vector.** [`Oja`] (Oja, J. Math. Biol. 15:267–273, 1982) subtracts
//!    `y^2 * w` from the Hebbian term. The subtraction is exactly the first-order expansion of
//!    dividing by the norm, so the weight vector converges to **unit length** and to the
//!    **principal eigenvector of the input correlation matrix** — Hebbian learning is principal
//!    component analysis, and Oja's is the line of algebra that proves it.
//! 2. **Move the goalposts.** [`Bcm`] (Bienenstock, Cooper & Munro, J. Neurosci. 2:32–48, 1982)
//!    keeps a *sliding threshold* `theta` that tracks the recent mean-square output. Output above
//!    `theta` potentiates, below it depresses, and `theta` chases `y^2` — so a cell that is too
//!    active raises its own bar. This is what produces **selectivity**: presented with several
//!    patterns, a `BCM` cell ends up responding to exactly one.
//! 3. **Rescale everything, slowly.** [`SynapticScaling`] (Turrigiano, Leslie, Desai, Rutherford &
//!    Nelson, Nature 391:892–896, 1998) multiplies *all* of a cell's inputs by one common factor
//!    driven by the difference between its firing rate and a target. Because it is
//!    **multiplicative**, it changes the scale without changing the ratios — so whatever the fast
//!    Hebbian rule learned survives the slow homeostatic correction. That is the experimental
//!    finding, not a modelling convenience: the mEPSC amplitude distribution scaled, it did not
//!    shift.
//!
//! # From rates to spikes
//!
//! Hebb, Oja and `BCM` are written in firing *rates*. Real synapses see spike *times*, and the
//! order matters: Bi & Poo (J. Neurosci. 18:10464–10472, 1998) showed that a presynaptic spike a
//! few milliseconds *before* a postsynaptic one potentiates, and the same pair in the other order
//! depresses, with the size of the change falling off exponentially in the lag. That is
//! **spike-timing-dependent plasticity**, and [`PairStdp`] is it:
//!
//! ```text
//! dw = +A_plus  * exp(-lag / tau_plus)     for lag > 0  (pre before post)
//! dw = -A_minus * exp( lag / tau_minus)    for lag < 0  (post before pre)
//! ```
//!
//! The exponential is not decoration — it is what the rule *is*, so a single isolated pair at a
//! known lag must reproduce it to the last bit. That is the sharpest test in this module and it
//! passes exactly, because [`PairStdp`] carries the window in two decaying traces whose product
//! with the amplitude is the same arithmetic as the closed form.
//!
//! # What the pair rule gets wrong
//!
//! The pair rule has one pre trace and one post trace, so the change caused by a pair depends on
//! the lag and, through residual traces, weakly on how often pairs arrive. Experiment says the
//! dependence on frequency is not weak. Sjöström, Turrigiano & Nelson (Neuron 32:1149–1164, 2001)
//! repeated the *same* pre-before-post pair at rates from 0.1 Hz to 50 Hz in visual cortex and saw
//! almost nothing at the bottom of that range and strong potentiation at the top. The pair rule,
//! fitted to the low-frequency window, goes the **wrong way**: at high frequency the accumulated
//! post trace makes every presynaptic spike depress, and the net change falls. The test
//! `the_pair_rule_cannot_reproduce_the_frequency_dependence_the_triplet_rule_can` in this module
//! shows it doing exactly that.
//!
//! [`TripletStdp`] (Pfister & Gerstner, J. Neurosci. 26:9673–9682, 2006) is the minimal repair: a
//! *second*, slower trace on each side, so that potentiation is driven by the pre trace multiplied
//! by how much postsynaptic activity there has *recently* been. One extra trace per side buys the
//! frequency dependence, and setting the triplet amplitudes to zero recovers the pair rule
//! **exactly** — an equivalence this module checks bit for bit rather than approximately.
//!
//! # The third factor
//!
//! Both rules above are *local*: they see one synapse's two spike trains and nothing else. An
//! animal rewarded a second after the action that earned it has a credit-assignment problem that
//! no local rule can solve, because the relevant spikes are long gone. Izhikevich (Cereb. Cortex
//! 17:2443–2452, 2007) named this the **distal reward problem** and gave the standard answer:
//! `STDP` does not write the weight, it writes a slowly decaying **eligibility trace**, and the
//! weight moves only when a global neuromodulator multiplies that trace. [`RewardStdp`] implements
//! it, and the consequence is worth stating plainly: **with zero reward the weight does not move
//! at all**, however much correlated activity the synapse sees. The synapse remembers; it does not
//! yet commit.
//!
//! # What this costs on hardware
//!
//! Inference reads a weight. Learning reads it, changes it and writes it back. On any fabric where
//! the weight lives off-core — which is every fabric large enough to matter — that turns one memory
//! transaction per synaptic event into at least two, and the transaction is the term
//! [`crate::ledger::Prices::e_syn_fetch`] prices and that every published device table in this
//! crate leaves unstated. The traces are worse per synapse than per neuron: [`PairStdp`] needs two
//! scalars, [`TripletStdp`] needs four, and the ones in this module are stored **per synapse**
//! because that is where the literature puts them. An implementation that shares traces per neuron
//! is a different model with different answers, and this module does not pretend otherwise.
//!
//! # Units
//!
//! Time is in **seconds** everywhere, rates in **hertz**. Weights are dimensionless multipliers
//! here rather than volts, because the same rule is applied to conductances, to volt-per-spike
//! synapses as in [`crate::net`], and to normalised weights in `[0, 1]`; the amplitudes `A_plus`
//! and `A_minus` therefore carry whatever unit the weight carries. Where a paper's amplitudes are
//! fractional changes in `EPSC` amplitude — Bi & Poo's are — the doc on the constructor says so
//! rather than pretending they are volts.
//!
//! # What is verified here
//!
//! A single pair against `A_plus * exp(-lag / tau_plus)` bit for bit, both signs, six lags; the
//! window's analytic integral against midpoint quadrature of the window itself; the triplet rule
//! against the pair rule as an exact equality; Oja's rule against the scalar map its own dynamics
//! reduce to and against the principal eigenvector of a correlation matrix built to have a known
//! one; `BCM`'s selective fixed point against `n_patterns * y_0`, which is where the sliding
//! threshold must settle; the distal-reward integral against
//! `c * d * tau_c * tau_d / (tau_c + tau_d)`; and every bounded rule against its bounds under
//! adversarial amplitudes.

/// Why a plasticity rule refused an input.
///
/// Every variant names the quantity, because a learning rule that silently accepts a `NaN`
/// amplitude produces weights that are all `NaN` after one update and a network that reports zero
/// spikes — which reads as "the model did not learn" rather than as "the input was malformed".
#[derive(Debug, Clone, PartialEq)]
pub enum PlasticityError {
    /// A parameter or input was infinite or `NaN`.
    NonFinite {
        /// Which quantity, by the name it has in the paper or in this API.
        what: &'static str,
        /// The offending value, so the caller can see whether it was an infinity or a `NaN`.
        value: f64,
    },
    /// A quantity that must be strictly positive — a time constant, a target rate — was not.
    NotPositive {
        /// Which quantity.
        what: &'static str,
        /// The offending value.
        value: f64,
    },
    /// A quantity that must not be negative — an amplitude, a time step — was.
    Negative {
        /// Which quantity.
        what: &'static str,
        /// The offending value.
        value: f64,
    },
    /// The weight floor was not below the weight ceiling, so no weight satisfies the bound.
    BoundsInverted {
        /// The proposed floor.
        w_min: f64,
        /// The proposed ceiling.
        w_max: f64,
    },
    /// An input vector did not match the weight vector it multiplies.
    LengthMismatch {
        /// Which vector arrived with the wrong length.
        what: &'static str,
        /// The length supplied.
        got: usize,
        /// The length required.
        want: usize,
    },
    /// A weight vector was empty, so the rule has nothing to learn on.
    Empty {
        /// Which vector.
        what: &'static str,
    },
    /// The rule's own state left the finite numbers, which for [`Hebbian`] is the documented
    /// instability arriving rather than a defect.
    Diverged {
        /// Which quantity went non-finite.
        what: &'static str,
        /// The value it reached.
        value: f64,
    },
}

impl core::fmt::Display for PlasticityError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NonFinite { what, value } => write!(f, "{what} is not finite ({value})"),
            Self::NotPositive { what, value } => {
                write!(f, "{what} must be strictly positive, got {value}")
            }
            Self::Negative { what, value } => write!(f, "{what} must not be negative, got {value}"),
            Self::BoundsInverted { w_min, w_max } => {
                write!(f, "weight floor {w_min} is not below ceiling {w_max}")
            }
            Self::LengthMismatch { what, got, want } => {
                write!(f, "{what} has {got} entries, expected {want}")
            }
            Self::Empty { what } => write!(f, "{what} is empty"),
            Self::Diverged { what, value } => {
                write!(f, "{what} left the finite numbers ({value}); the rule diverged")
            }
        }
    }
}

/// See the note on [`crate::net::NetError`]: a library error has to be able to cross a
/// `Box<dyn Error>` boundary or its callers reach for `.unwrap()`.
impl std::error::Error for PlasticityError {}

fn finite(what: &'static str, v: f64) -> Result<f64, PlasticityError> {
    if v.is_finite() { Ok(v) } else { Err(PlasticityError::NonFinite { what, value: v }) }
}

fn positive(what: &'static str, v: f64) -> Result<f64, PlasticityError> {
    let v = finite(what, v)?;
    if v > 0.0 { Ok(v) } else { Err(PlasticityError::NotPositive { what, value: v }) }
}

fn non_negative(what: &'static str, v: f64) -> Result<f64, PlasticityError> {
    let v = finite(what, v)?;
    if v >= 0.0 { Ok(v) } else { Err(PlasticityError::Negative { what, value: v }) }
}

/// A closed interval a weight is never allowed to leave.
///
/// The clamp is applied on **every** update by every bounded rule in this module, including the
/// ones whose weight dependence already vanishes at the edge. That is deliberate: a soft bound
/// makes the step small near the boundary, it does not make it zero, and one oversized amplitude
/// steps straight through. The guarantee a caller needs is "the weight is in range", and only an
/// unconditional clamp provides it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bounds {
    /// Weight floor, in whatever unit the weight carries. Inclusive.
    pub w_min: f64,
    /// Weight ceiling, same unit. Inclusive, and strictly above `w_min`.
    pub w_max: f64,
}

impl Bounds {
    /// Build a bound, rejecting an inverted or non-finite one.
    ///
    /// # Errors
    ///
    /// [`PlasticityError::NonFinite`] if either endpoint is infinite or `NaN`, or
    /// [`PlasticityError::BoundsInverted`] if the floor is not strictly below the ceiling. An
    /// infinite ceiling is refused rather than accepted as "no bound", because
    /// [`WeightRule::SoftBound`] divides by the span and an infinite span makes every factor zero.
    pub fn new(w_min: f64, w_max: f64) -> Result<Self, PlasticityError> {
        let w_min = finite("weight floor", w_min)?;
        let w_max = finite("weight ceiling", w_max)?;
        if !(w_min < w_max) {
            return Err(PlasticityError::BoundsInverted { w_min, w_max });
        }
        Ok(Self { w_min, w_max })
    }

    /// The `[0, 1]` normalised interval most of the `STDP` literature works in.
    ///
    /// Gütig, Aharonov, Rotter & Sompolinsky (J. Neurosci. 23:3697–3714, 2003) state their
    /// power-law weight dependence on exactly this interval, so [`WeightRule::SoftBound`]'s
    /// normalisation is the identity here and the exponent `mu` means what the paper says it means.
    #[must_use]
    pub const fn normalised() -> Self {
        Self { w_min: 0.0, w_max: 1.0 }
    }

    /// A finite stand-in for "no bound": `-1e12` to `1e12`.
    ///
    /// Finite on purpose. An infinite ceiling turns [`WeightRule::SoftBound`]'s span into an
    /// infinity and every factor into zero, so the rule would silently stop learning instead of
    /// running unbounded. `1e12` is past any weight a simulation reaches and still leaves 296
    /// powers of two of headroom below overflow.
    #[must_use]
    pub const fn wide() -> Self {
        Self { w_min: -1.0e12, w_max: 1.0e12 }
    }

    /// The weight, moved into range if it was outside.
    #[must_use]
    pub fn clamp(self, w: f64) -> f64 {
        // Written as two comparisons rather than `f64::clamp` because `f64::clamp` panics on a NaN
        // bound and propagates a NaN input; here a NaN weight would pass both comparisons unchanged
        // and reach `contains`, which reports it as out of range.
        if w < self.w_min {
            self.w_min
        } else if w > self.w_max {
            self.w_max
        } else {
            w
        }
    }

    /// Whether the weight is inside the interval. A `NaN` weight is **not** inside.
    #[must_use]
    pub fn contains(self, w: f64) -> bool {
        w >= self.w_min && w <= self.w_max
    }

    /// Ceiling minus floor, always strictly positive by construction.
    #[must_use]
    pub fn span(self) -> f64 {
        self.w_max - self.w_min
    }
}

/// Which spike pairs a trace-based rule lets interact.
///
/// Morrison, Diesmann & Gerstner (Biol. Cybern. 98:459–478, 2008) catalogue several inequivalent
/// nearest-neighbour schemes; the one here is the symmetric reduction, where a spike **resets** its
/// own side's traces to one rather than incrementing them. The choice changes the high-frequency
/// behaviour of every rule in this module and is therefore explicit rather than implied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pairing {
    /// Every presynaptic spike interacts with every postsynaptic spike: traces **accumulate**.
    ///
    /// The usual default, and the scheme under which the frequency dependence of the triplet model
    /// was fitted.
    AllToAll,
    /// Only the most recent partner on each side counts: a spike **sets** its traces to one.
    ///
    /// Saturating by construction, so the per-pair change stops growing with frequency. Cheaper on
    /// hardware — one trace value, no accumulator overflow — and a different model, not an
    /// optimisation of the other one.
    NearestNeighbour,
}

/// How large a step is, given where the weight already is.
///
/// The three forms below are the standard ones and they differ in what the *equilibrium weight
/// distribution* looks like, which is the thing experiments can see. Additive `STDP` drives weights
/// to the two bounds and leaves a bimodal distribution; multiplicative forms leave a unimodal one.
/// Van Rossum, Bi & Turrigiano (J. Neurosci. 20:8812–8821, 2000) is the argument for the second;
/// Gütig, Aharonov, Rotter & Sompolinsky (J. Neurosci. 23:3697–3714, 2003) show the crossover
/// between the two regimes is controlled by a single exponent, which is [`WeightRule::SoftBound`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WeightRule {
    /// The step does not depend on the weight at all; only the hard clamp stops it.
    ///
    /// Produces a bimodal steady-state distribution piled at both bounds, and competition between
    /// synapses — which is the property Song, Miller & Abbott (Nat. Neurosci. 3:919–926, 2000)
    /// wanted from it.
    Additive,
    /// Potentiation is weight-independent, depression is proportional to `w - w_min`.
    ///
    /// Van Rossum, Bi & Turrigiano (J. Neurosci. 20:8812–8821, 2000), whose `w_min` is zero so
    /// their depression term reads simply `w`. A weight already at the floor cannot be depressed
    /// further, so the floor is reached asymptotically rather than by clamping.
    MultiplicativeDepression,
    /// Both directions scale as a power of the **normalised** distance to the bound they approach.
    ///
    /// `mu = 0` is exactly [`WeightRule::Additive`] — `x.powf(0.0)` is one for every `x`, including
    /// zero — and `mu = 1` is linear soft bounds. Gütig et al. (2003) sweep `mu` and find the
    /// bimodal-to-unimodal transition inside `0 < mu < 1`.
    SoftBound {
        /// Exponent, dimensionless, normally in `0..=1`. Must not be negative: a negative exponent
        /// makes the factor *diverge* at the bound it is supposed to stop the weight reaching.
        mu: f64,
    },
}

impl WeightRule {
    /// The multiplier on a potentiating step for a synapse currently at `w`.
    ///
    /// Never negative, so potentiation can never be turned into depression by the weight
    /// dependence — which is what a negative factor would do and is never what a paper means.
    #[must_use]
    pub fn potentiation_factor(self, w: f64, bounds: Bounds) -> f64 {
        match self {
            Self::Additive | Self::MultiplicativeDepression => 1.0,
            Self::SoftBound { mu } => ((bounds.w_max - w) / bounds.span()).max(0.0).powf(mu),
        }
    }

    /// The multiplier on a depressing step for a synapse currently at `w`.
    #[must_use]
    pub fn depression_factor(self, w: f64, bounds: Bounds) -> f64 {
        match self {
            Self::Additive => 1.0,
            Self::MultiplicativeDepression => (w - bounds.w_min).max(0.0),
            Self::SoftBound { mu } => ((w - bounds.w_min) / bounds.span()).max(0.0).powf(mu),
        }
    }

    fn validate(self) -> Result<Self, PlasticityError> {
        if let Self::SoftBound { mu } = self {
            non_negative("soft-bound exponent mu", mu)?;
        }
        Ok(self)
    }
}

/// One exponentially decaying spike trace: the state every timing-based rule here is built from.
///
/// A trace is a running, leaky count of recent spikes. Its value `dt` seconds after a single spike
/// is `exp(-dt / tau)`, which is exactly the `STDP` window's shape — so a rule that multiplies an
/// amplitude by a trace at the moment of the partner spike *is* the exponential window, evaluated
/// online, with no table and no search back through spike history. That equivalence is what makes
/// [`PairStdp::apply_pair`] agree with [`PairStdp::window`] to the last bit rather than to a
/// tolerance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Trace {
    /// Decay time constant, seconds. The window's width, and the only parameter.
    pub tau: f64,
    /// Current value, dimensionless. One immediately after an isolated spike under either pairing
    /// scheme; unbounded above under [`Pairing::AllToAll`] at high rates.
    pub x: f64,
}

impl Trace {
    /// A trace at rest with time constant `tau` seconds.
    ///
    /// # Errors
    ///
    /// [`PlasticityError::NotPositive`] or [`PlasticityError::NonFinite`] for a `tau` that is not a
    /// strictly positive finite number. A zero `tau` is refused rather than treated as "instant
    /// decay": the division would produce an infinity and then, at `x = 0`, a `NaN`.
    pub fn new(tau: f64) -> Result<Self, PlasticityError> {
        Ok(Self { tau: positive("trace time constant tau", tau)?, x: 0.0 })
    }

    /// Decay forward by `dt` seconds, exactly.
    ///
    /// Exponential rather than `x -= x * dt / tau`, for the same reason [`crate::neuron::Lif`]
    /// integrates exponentially: the closed form is available, costs one `exp`, and composes across
    /// steps, so a coarse step and a fine step give the same trace.
    ///
    /// # Errors
    ///
    /// [`PlasticityError::Negative`] or [`PlasticityError::NonFinite`] for a `dt` that is not a
    /// finite non-negative number. Negative time is refused because it would *amplify* the trace.
    pub fn advance(&mut self, dt: f64) -> Result<(), PlasticityError> {
        let dt = non_negative("dt", dt)?;
        self.x *= (-dt / self.tau).exp();
        Ok(())
    }

    /// Register a spike on this trace's own side.
    pub fn fire(&mut self, pairing: Pairing) {
        match pairing {
            Pairing::AllToAll => self.x += 1.0,
            Pairing::NearestNeighbour => self.x = 1.0,
        }
    }

    /// The value the trace would have `dt` seconds from now with no further spikes.
    ///
    /// The closed form, exposed so a test can compare an online run against it rather than against
    /// a previous online run.
    #[must_use]
    pub fn after(self, dt: f64) -> f64 {
        self.x * (-dt / self.tau).exp()
    }

    /// Back to zero, forgetting every spike.
    pub fn clear(&mut self) {
        self.x = 0.0;
    }
}

/// Pair-based spike-timing-dependent plasticity.
///
/// Bi & Poo, J. Neurosci. 18:10464–10472, 1998. One presynaptic spike and one postsynaptic spike
/// separated by `lag = t_post - t_pre` change the weight by
///
/// ```text
/// dw = +A_plus  * exp(-lag / tau_plus)     lag > 0,  pre before post  -> potentiation
/// dw = -A_minus * exp( lag / tau_minus)    lag < 0,  post before pre  -> depression
/// ```
///
/// which is [`PairStdp::window`], and which [`PairStdp::apply_pair`] reproduces **exactly** rather
/// than approximately. The implementation is the standard online one: a pre trace `x` with time
/// constant `tau_plus`, a post trace `y` with time constant `tau_minus`, each bumped by its own
/// side's spikes and read by the other side's.
///
/// # Stability
///
/// The integral of the window over all lags is `A_plus * tau_plus - A_minus * tau_minus`. Song,
/// Miller & Abbott (Nat. Neurosci. 3:919–926, 2000) point out that this must be **negative** for
/// the rule to be stable against uncorrelated input — otherwise every synapse walks to the ceiling
/// and the competition that makes `STDP` interesting never happens. That is
/// [`PairStdp::is_depression_dominated`], and it is a one-line check worth running on any parameter
/// set taken from a paper, including the ones in this module.
///
/// # At zero lag
///
/// The window is discontinuous at `lag = 0` and the experiment has no measurement there —
/// simultaneous spikes are not a condition anyone can impose to sub-millisecond accuracy in a slice.
/// [`PairStdp::window`] returns `0.0` at exactly zero, which is a stated convention, not a
/// measurement. The online path never hits it: a pre and a post spike delivered in that order with
/// no [`PairStdp::advance`] between them potentiate by the full `A_plus`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PairStdp {
    /// Potentiation amplitude at zero lag, in the weight's own unit. Non-negative.
    pub a_plus: f64,
    /// Depression amplitude at zero lag, same unit, stored as a **positive** magnitude; the sign is
    /// applied by the rule, so a caller cannot accidentally make depression potentiate.
    pub a_minus: f64,
    /// Potentiation window width, seconds. Bi & Poo's fit is 16.8 ms.
    pub tau_plus: f64,
    /// Depression window width, seconds. Bi & Poo's fit is 33.7 ms — wider than potentiation, which
    /// is why the rule can be depression-dominated with comparable amplitudes.
    pub tau_minus: f64,
    /// How the step scales with the current weight.
    pub rule: WeightRule,
    /// The interval the weight may never leave.
    pub bounds: Bounds,
    /// Which spike pairs interact.
    pub pairing: Pairing,
    /// Presynaptic trace, decaying with `tau_plus`; read at postsynaptic spikes.
    pub pre_trace: Trace,
    /// Postsynaptic trace, decaying with `tau_minus`; read at presynaptic spikes.
    pub post_trace: Trace,
}

impl PairStdp {
    /// Build from the four window parameters.
    ///
    /// `pairing` defaults to [`Pairing::AllToAll`]; set the field afterwards to change it.
    ///
    /// # Errors
    ///
    /// [`PlasticityError::Negative`] for a negative amplitude, [`PlasticityError::NotPositive`] for
    /// a non-positive time constant, [`PlasticityError::NonFinite`] for any non-finite parameter,
    /// and whatever [`WeightRule`] validation returns for a negative soft-bound exponent.
    pub fn new(
        a_plus: f64,
        a_minus: f64,
        tau_plus: f64,
        tau_minus: f64,
        rule: WeightRule,
        bounds: Bounds,
    ) -> Result<Self, PlasticityError> {
        Ok(Self {
            a_plus: non_negative("A_plus", a_plus)?,
            a_minus: non_negative("A_minus", a_minus)?,
            tau_plus: positive("tau_plus", tau_plus)?,
            tau_minus: positive("tau_minus", tau_minus)?,
            rule: rule.validate()?,
            bounds,
            pairing: Pairing::AllToAll,
            pre_trace: Trace::new(tau_plus)?,
            post_trace: Trace::new(tau_minus)?,
        })
    }

    /// Bi & Poo's time constants — 16.8 ms potentiation, 33.7 ms depression — with amplitudes the
    /// caller supplies.
    ///
    /// The time constants are the widely reproduced fit from Bi & Poo (J. Neurosci.
    /// 18:10464–10472, 1998) and are reused unchanged by Pfister & Gerstner (2006). **The
    /// amplitudes are not supplied here on purpose**: the paper reports them as percentage changes
    /// in `EPSC` amplitude with large scatter across synapses, so a single number presented as
    /// "Bi & Poo's `A_plus`" would be a figure this implementation could not defend. Choose them for
    /// the weight unit you are using, and check
    /// [`PairStdp::is_depression_dominated`] afterwards.
    ///
    /// # Errors
    ///
    /// As [`PairStdp::new`].
    pub fn bi_poo_1998(
        a_plus: f64,
        a_minus: f64,
        rule: WeightRule,
        bounds: Bounds,
    ) -> Result<Self, PlasticityError> {
        Self::new(a_plus, a_minus, 16.8e-3, 33.7e-3, rule, bounds)
    }

    /// The parameter set from Song, Miller & Abbott, Nat. Neurosci. 3:919–926, 2000.
    ///
    /// `tau_plus = tau_minus = 20 ms`, `A_plus = 0.005 * g_max`, and `A_minus / A_plus = 1.05` —
    /// the 5% asymmetry is the entire stabilising mechanism in that paper, and it is what makes the
    /// window integral negative when the two time constants are equal. Additive, hard bounds
    /// `[0, g_max]`, which is what produces the bimodal weight distribution the paper reports.
    ///
    /// # Errors
    ///
    /// [`PlasticityError::NotPositive`] if `g_max` is not strictly positive, plus anything
    /// [`PairStdp::new`] returns.
    pub fn song_abbott_2000(g_max: f64) -> Result<Self, PlasticityError> {
        let g_max = positive("g_max", g_max)?;
        let a_plus = 0.005 * g_max;
        Self::new(
            a_plus,
            1.05 * a_plus,
            20e-3,
            20e-3,
            WeightRule::Additive,
            Bounds::new(0.0, g_max)?,
        )
    }

    /// The window in closed form: the weight change a single isolated pair at lag `lag` seconds
    /// would cause on a synapse whose weight dependence is one.
    ///
    /// `lag = t_post - t_pre`. Positive means pre fired first. Returns `0.0` at exactly zero lag by
    /// the convention stated on the type; `NaN` in gives `NaN` out is impossible here because a
    /// `NaN` lag satisfies neither comparison and falls through to the zero branch — which is a
    /// silent answer to a malformed question, so prefer [`PairStdp::apply_pair`], which refuses.
    #[must_use]
    pub fn window(&self, lag: f64) -> f64 {
        if lag > 0.0 {
            self.a_plus * (-lag / self.tau_plus).exp()
        } else if lag < 0.0 {
            -self.a_minus * (lag / self.tau_minus).exp()
        } else {
            0.0
        }
    }

    /// The integral of the window over `-half_width ..= half_width`, in closed form.
    ///
    /// ```text
    /// A_plus * tau_plus * (1 - exp(-T / tau_plus)) - A_minus * tau_minus * (1 - exp(-T / tau_minus))
    /// ```
    ///
    /// Units are the weight's unit times seconds. The quantity matters because it is the mean drift
    /// a synapse experiences under Poisson pre and post activity with no correlation between them:
    /// negative means the rule is self-limiting, positive means it is not.
    ///
    /// # Errors
    ///
    /// [`PlasticityError::Negative`] or [`PlasticityError::NonFinite`] for a `half_width` that is
    /// not a finite non-negative number.
    pub fn window_integral(&self, half_width: f64) -> Result<f64, PlasticityError> {
        let t = non_negative("half_width", half_width)?;
        let pot = self.a_plus * self.tau_plus * (1.0 - (-t / self.tau_plus).exp());
        let dep = self.a_minus * self.tau_minus * (1.0 - (-t / self.tau_minus).exp());
        Ok(pot - dep)
    }

    /// The `half_width -> infinity` limit of [`PairStdp::window_integral`]:
    /// `A_plus * tau_plus - A_minus * tau_minus`.
    #[must_use]
    pub fn total_window_area(&self) -> f64 {
        self.a_plus * self.tau_plus - self.a_minus * self.tau_minus
    }

    /// Whether the total window area is negative, which is Song, Miller & Abbott's (2000)
    /// stability condition against uncorrelated activity.
    ///
    /// Not a proof of stability — the condition is necessary for the drift argument they make, and
    /// correlated input can still drive a synapse to a bound, which is the point of the rule. It is
    /// a cheap check that a transcribed parameter set is on the right side of the line.
    #[must_use]
    pub fn is_depression_dominated(&self) -> bool {
        self.total_window_area() < 0.0
    }

    /// The weight change a presynaptic spike would cause **right now**, without applying it.
    ///
    /// Negative or zero. Exposed because [`RewardStdp`] needs the number to put into an eligibility
    /// trace rather than into the weight, and because a caller pricing plasticity wants to know the
    /// step before paying for the write.
    #[must_use]
    pub fn pre_increment(&self, w: f64) -> f64 {
        -(self.a_minus * self.post_trace.x) * self.rule.depression_factor(w, self.bounds)
    }

    /// The weight change a postsynaptic spike would cause right now, without applying it.
    ///
    /// Positive or zero.
    #[must_use]
    pub fn post_increment(&self, w: f64) -> f64 {
        (self.a_plus * self.pre_trace.x) * self.rule.potentiation_factor(w, self.bounds)
    }

    /// Register a presynaptic spike in the traces without touching any weight.
    pub fn note_pre(&mut self) {
        self.pre_trace.fire(self.pairing);
    }

    /// Register a postsynaptic spike in the traces without touching any weight.
    pub fn note_post(&mut self) {
        self.post_trace.fire(self.pairing);
    }

    /// Decay both traces by `dt` seconds.
    ///
    /// # Errors
    ///
    /// As [`Trace::advance`].
    pub fn advance(&mut self, dt: f64) -> Result<(), PlasticityError> {
        self.pre_trace.advance(dt)?;
        self.post_trace.advance(dt)
    }

    /// A presynaptic spike arrives: depress by the post trace, then register the spike.
    ///
    /// Returns the new weight, clamped into [`PairStdp::bounds`].
    ///
    /// # Errors
    ///
    /// [`PlasticityError::NonFinite`] if the weight handed in is not finite. The weight is an input
    /// here rather than state because a rule object is shared across many synapses in every
    /// realistic use, and storing one weight inside it would quietly make that wrong.
    pub fn on_pre(&mut self, w: f64) -> Result<f64, PlasticityError> {
        let w = finite("weight", w)?;
        let out = self.bounds.clamp(w + self.pre_increment(w));
        self.note_pre();
        Ok(out)
    }

    /// A postsynaptic spike arrives: potentiate by the pre trace, then register the spike.
    ///
    /// Returns the new weight, clamped into [`PairStdp::bounds`].
    ///
    /// # Errors
    ///
    /// As [`PairStdp::on_pre`].
    pub fn on_post(&mut self, w: f64) -> Result<f64, PlasticityError> {
        let w = finite("weight", w)?;
        let out = self.bounds.clamp(w + self.post_increment(w));
        self.note_post();
        Ok(out)
    }

    /// Play one isolated pre/post pair at lag `lag` seconds on a synapse at weight `w`, from a
    /// cleared state, and return the new weight.
    ///
    /// This is the protocol the 1998 experiment ran, and with [`WeightRule::Additive`] and a weight
    /// away from its bounds the change it produces is [`PairStdp::window`] to the last bit — the
    /// two paths compute the same product of the same amplitude and the same exponential.
    ///
    /// **At exactly `lag = 0` this returns `w + A_plus`, not `w`.** Two spikes with no time between
    /// them are delivered pre-then-post here, so the pre trace is at one when the post spike reads
    /// it. [`PairStdp::window`] returns zero at that point instead. The disagreement is the
    /// window's discontinuity, it is one isolated point, and neither answer is a measurement.
    ///
    /// # Errors
    ///
    /// [`PlasticityError::NonFinite`] for a non-finite `w` or `lag`.
    pub fn apply_pair(&mut self, w: f64, lag: f64) -> Result<f64, PlasticityError> {
        let w = finite("weight", w)?;
        let lag = finite("lag", lag)?;
        self.clear();
        if lag >= 0.0 {
            let w = self.on_pre(w)?;
            self.advance(lag)?;
            self.on_post(w)
        } else {
            let w = self.on_post(w)?;
            self.advance(-lag)?;
            self.on_pre(w)
        }
    }

    /// Forget all spike history. Parameters are untouched.
    pub fn clear(&mut self) {
        self.pre_trace.clear();
        self.post_trace.clear();
    }
}

/// Triplet spike-timing-dependent plasticity.
///
/// Pfister & Gerstner, J. Neurosci. 26:9673–9682, 2006. Two traces per side instead of one:
///
/// | trace | side | time constant | read by |
/// |---|---|---|---|
/// | `r1` | pre | `tau_plus` | post spikes, as the pair term |
/// | `r2` | pre | `tau_x` (slow) | pre spikes, as the triplet term |
/// | `o1` | post | `tau_minus` | pre spikes, as the pair term |
/// | `o2` | post | `tau_y` (slow) | post spikes, as the triplet term |
///
/// ```text
/// on pre  spike:  dw = -o1 * (A2_minus + A3_minus * r2(just before this spike))
/// on post spike:  dw = +r1 * (A2_plus  + A3_plus  * o2(just before this spike))
/// ```
///
/// *Just before* is load-bearing: the spike's own contribution to its own slow trace is excluded,
/// which is why the fast traces are read and the slow traces are incremented in that order.
///
/// # What the pair rule gets wrong, precisely
///
/// [`PairStdp`] has one number to describe how much recent postsynaptic activity there has been —
/// the post trace — and that number is used only for *depression*. So the amount of potentiation a
/// pre-before-post pair produces cannot grow when the pairs are repeated faster. Experiment says it
/// does grow, and strongly: Sjöström, Turrigiano & Nelson (Neuron 32:1149–1164, 2001) ran the same
/// `+10 ms` pair at 0.1 Hz through 50 Hz in visual cortex and found the outcome swing from
/// essentially nothing to robust `LTP`. Worse for the pair rule, its *residual post trace* makes
/// every presynaptic spike in a fast train depress, so a pair rule fitted to the low-frequency
/// window produces **less** potentiation at 40 Hz than at 1 Hz — the opposite of the measurement.
/// `A3_plus` is the repair: potentiation is now proportional to how much postsynaptic activity
/// preceded the post spike, which is exactly the quantity that grows with frequency.
///
/// The pair rule is recovered **exactly** by setting both triplet amplitudes to zero, and that
/// equivalence is checked here as a bit-for-bit equality over a random spike train rather than as a
/// tolerance.
///
/// # On the constants
///
/// The four time constants — 16.8, 101, 33.7 and 125 ms — are the widely reproduced values for the
/// visual-cortex fit, with `tau_plus` and `tau_minus` inherited from Bi & Poo (1998). The
/// **amplitudes** in the named constructors below were transcribed from Table 3 of the paper as
/// reported in the secondary literature; this implementation did not verify the digits against the
/// printed table, and anyone reproducing a published figure should check them at the source before
/// citing a result. They are stated here rather than omitted because a rule with no parameters
/// teaches nothing, and stated with this caveat rather than confidently because they might be wrong
/// in the last digit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TripletStdp {
    /// Pair potentiation amplitude, weight units. Zero in the *minimal* model, which is the point
    /// of calling it minimal: all potentiation is then a triplet effect.
    pub a2_plus: f64,
    /// Triplet potentiation amplitude, weight units, multiplying the slow post trace `o2`.
    pub a3_plus: f64,
    /// Pair depression amplitude, weight units, stored as a positive magnitude.
    pub a2_minus: f64,
    /// Triplet depression amplitude, weight units, multiplying the slow pre trace `r2`. Zero in the
    /// minimal visual-cortex model.
    pub a3_minus: f64,
    /// Fast presynaptic trace `r1`, time constant `tau_plus`.
    pub r1: Trace,
    /// Slow presynaptic trace `r2`, time constant `tau_x`, 101 ms in the visual-cortex fit.
    pub r2: Trace,
    /// Fast postsynaptic trace `o1`, time constant `tau_minus`.
    pub o1: Trace,
    /// Slow postsynaptic trace `o2`, time constant `tau_y`, 125 ms in the visual-cortex fit.
    pub o2: Trace,
    /// How the step scales with the current weight.
    pub rule: WeightRule,
    /// The interval the weight may never leave.
    pub bounds: Bounds,
    /// Which spike pairs interact. The published amplitudes are fitted **per scheme**, so changing
    /// this without changing the amplitudes changes the model's predictions.
    pub pairing: Pairing,
}

impl TripletStdp {
    /// Build from the four amplitudes and the four time constants, all seconds.
    ///
    /// # Errors
    ///
    /// [`PlasticityError::Negative`] for a negative amplitude, [`PlasticityError::NotPositive`] for
    /// a non-positive time constant, [`PlasticityError::NonFinite`] for a non-finite parameter.
    pub fn new(
        a2_plus: f64,
        a3_plus: f64,
        a2_minus: f64,
        a3_minus: f64,
        tau_plus: f64,
        tau_x: f64,
        tau_minus: f64,
        tau_y: f64,
        rule: WeightRule,
        bounds: Bounds,
    ) -> Result<Self, PlasticityError> {
        Ok(Self {
            a2_plus: non_negative("A2_plus", a2_plus)?,
            a3_plus: non_negative("A3_plus", a3_plus)?,
            a2_minus: non_negative("A2_minus", a2_minus)?,
            a3_minus: non_negative("A3_minus", a3_minus)?,
            r1: Trace::new(tau_plus)?,
            r2: Trace::new(tau_x)?,
            o1: Trace::new(tau_minus)?,
            o2: Trace::new(tau_y)?,
            rule: rule.validate()?,
            bounds,
            pairing: Pairing::AllToAll,
        })
    }

    /// The **minimal** triplet model fitted to visual-cortex data (Sjöström et al. 2001), all-to-all
    /// pairing: `A2_plus = 0`, `A3_plus = 6.5e-3`, `A2_minus = 7.1e-3`, `A3_minus = 0`.
    ///
    /// Minimal because two of the four amplitudes are zero: depression is purely a pair effect and
    /// potentiation is purely a triplet effect. This is the parameter set that reproduces the
    /// frequency dependence the pair rule cannot, and the one the test in this module uses.
    ///
    /// See the caveat on the type: the amplitudes are transcribed, not verified against the printed
    /// table.
    ///
    /// # Errors
    ///
    /// As [`TripletStdp::new`].
    pub fn visual_cortex_minimal(bounds: Bounds) -> Result<Self, PlasticityError> {
        Self::new(
            0.0,
            6.5e-3,
            7.1e-3,
            0.0,
            16.8e-3,
            101e-3,
            33.7e-3,
            125e-3,
            WeightRule::Additive,
            bounds,
        )
    }

    /// The **full** triplet model fitted to hippocampal culture data, nearest-spike pairing:
    /// `A2_plus = 5.3e-3`, `A3_plus = 8.0e-3`, `A2_minus = 3.5e-3`, `A3_minus = 1.0e-3`, with the
    /// slow time constants `tau_x = 946 ms` and `tau_y = 27 ms`.
    ///
    /// Note how different the slow constants are from the visual-cortex fit — 946 ms against
    /// 101 ms. The triplet model is not one model with one parameter set; it is a form that two
    /// preparations fill in differently, and reporting a result with the wrong preparation's
    /// numbers is a category error the shared function name makes easy.
    ///
    /// Sets [`Pairing::NearestNeighbour`], because that is the scheme these amplitudes were fitted
    /// under. Same transcription caveat as the type doc.
    ///
    /// # Errors
    ///
    /// As [`TripletStdp::new`].
    pub fn hippocampal_full(bounds: Bounds) -> Result<Self, PlasticityError> {
        let mut t = Self::new(
            5.3e-3,
            8.0e-3,
            3.5e-3,
            1.0e-3,
            16.8e-3,
            946e-3,
            33.7e-3,
            27e-3,
            WeightRule::Additive,
            bounds,
        )?;
        t.pairing = Pairing::NearestNeighbour;
        Ok(t)
    }

    /// The weight change a presynaptic spike would cause right now, without applying it.
    #[must_use]
    pub fn pre_increment(&self, w: f64) -> f64 {
        -(self.o1.x * (self.a2_minus + self.a3_minus * self.r2.x))
            * self.rule.depression_factor(w, self.bounds)
    }

    /// The weight change a postsynaptic spike would cause right now, without applying it.
    #[must_use]
    pub fn post_increment(&self, w: f64) -> f64 {
        (self.r1.x * (self.a2_plus + self.a3_plus * self.o2.x))
            * self.rule.potentiation_factor(w, self.bounds)
    }

    /// Decay all four traces by `dt` seconds.
    ///
    /// # Errors
    ///
    /// As [`Trace::advance`].
    pub fn advance(&mut self, dt: f64) -> Result<(), PlasticityError> {
        self.r1.advance(dt)?;
        self.r2.advance(dt)?;
        self.o1.advance(dt)?;
        self.o2.advance(dt)
    }

    /// Register a presynaptic spike in the traces without touching any weight.
    pub fn note_pre(&mut self) {
        self.r1.fire(self.pairing);
        self.r2.fire(self.pairing);
    }

    /// Register a postsynaptic spike in the traces without touching any weight.
    pub fn note_post(&mut self) {
        self.o1.fire(self.pairing);
        self.o2.fire(self.pairing);
    }

    /// A presynaptic spike arrives. Returns the new weight, clamped.
    ///
    /// # Errors
    ///
    /// [`PlasticityError::NonFinite`] for a non-finite weight.
    pub fn on_pre(&mut self, w: f64) -> Result<f64, PlasticityError> {
        let w = finite("weight", w)?;
        let out = self.bounds.clamp(w + self.pre_increment(w));
        self.note_pre();
        Ok(out)
    }

    /// A postsynaptic spike arrives. Returns the new weight, clamped.
    ///
    /// # Errors
    ///
    /// [`PlasticityError::NonFinite`] for a non-finite weight.
    pub fn on_post(&mut self, w: f64) -> Result<f64, PlasticityError> {
        let w = finite("weight", w)?;
        let out = self.bounds.clamp(w + self.post_increment(w));
        self.note_post();
        Ok(out)
    }

    /// Forget all spike history.
    pub fn clear(&mut self) {
        self.r1.clear();
        self.r2.clear();
        self.o1.clear();
        self.o2.clear();
    }
}

/// Plain Hebbian learning with optional weight decay, left unstable on purpose.
///
/// `dw_i = eta * (y * x_i - decay * w_i)` with `y = w . x`. Hebb, *The Organization of Behavior*,
/// Wiley, 1949, as usually written down.
///
/// With `decay = 0` this diverges. Under a constant input `x = u` with `|u| = 1` the weight along
/// `u` is multiplied by `1 + eta` every sample, so after `n` samples it is `(1 + eta)^n` times what
/// it started as — an exact geometric closed form, and the test that checks it is also the test that
/// demonstrates the instability. **There are no bounds on this type.** Clamping it would hide the
/// only thing it has to teach, which is why Oja's and `BCM`'s extra terms exist.
///
/// [`Hebbian::update`] refuses once the weights leave the finite numbers, so the divergence arrives
/// as a named error rather than as a vector of `NaN` that poisons everything downstream.
#[derive(Debug, Clone, PartialEq)]
pub struct Hebbian {
    /// Learning rate per sample, dimensionless. The growth factor under unit input is `1 + eta`.
    pub eta: f64,
    /// Linear weight decay per sample, dimensionless. Zero is pure Hebb; a positive value bounds
    /// the norm at `sqrt(eta * lambda_1 / decay)`-ish but does **not** normalise it.
    pub decay: f64,
    /// The weight vector, dimensionless multipliers, one per input channel.
    pub w: Vec<f64>,
}

impl Hebbian {
    /// Build from a starting weight vector.
    ///
    /// # Errors
    ///
    /// [`PlasticityError::Empty`] for an empty weight vector, [`PlasticityError::NonFinite`] for a
    /// non-finite weight or learning rate, [`PlasticityError::Negative`] for a negative `eta` or
    /// `decay`.
    pub fn new(w: Vec<f64>, eta: f64, decay: f64) -> Result<Self, PlasticityError> {
        if w.is_empty() {
            return Err(PlasticityError::Empty { what: "weight vector" });
        }
        for &wi in &w {
            finite("weight", wi)?;
        }
        Ok(Self { eta: non_negative("eta", eta)?, decay: non_negative("decay", decay)?, w })
    }

    /// The linear output `w . x`.
    ///
    /// # Errors
    ///
    /// [`PlasticityError::LengthMismatch`] if `x` is not as long as the weight vector,
    /// [`PlasticityError::NonFinite`] for a non-finite input.
    pub fn output(&self, x: &[f64]) -> Result<f64, PlasticityError> {
        dot(&self.w, x)
    }

    /// One sample. Returns the output `y` that drove the update.
    ///
    /// # Errors
    ///
    /// As [`Hebbian::output`], plus [`PlasticityError::Diverged`] when the update takes a weight
    /// out of the finite numbers — which is this rule working as advertised, not failing.
    pub fn update(&mut self, x: &[f64]) -> Result<f64, PlasticityError> {
        let y = self.output(x)?;
        for i in 0..self.w.len() {
            self.w[i] += self.eta * (y * x[i] - self.decay * self.w[i]);
            if !self.w[i].is_finite() {
                return Err(PlasticityError::Diverged { what: "Hebbian weight", value: self.w[i] });
            }
        }
        Ok(y)
    }

    /// Euclidean length of the weight vector.
    #[must_use]
    pub fn norm(&self) -> f64 {
        norm(&self.w)
    }
}

/// Oja's rule: Hebbian learning that normalises itself.
///
/// Oja, J. Math. Biol. 15:267–273, 1982:
///
/// ```text
/// dw_i = eta * y * (x_i - y * w_i),    y = w . x
/// ```
///
/// The subtracted term is the first-order expansion of explicitly renormalising `w` after each
/// Hebbian step, which is why it does the same job without a square root in the loop.
///
/// # Two closed forms it is checked against
///
/// **The scalar reduction.** Under a constant unit input `x = u`, write `w = c * u`. Then `y = c`
/// and the vector rule collapses exactly to the one-dimensional map `c <- c + eta * c * (1 - c^2)`,
/// whose only stable fixed point is `c = 1`. The test iterates that scalar map independently and
/// compares it to `|w|` step by step.
///
/// **The eigenvector.** For zero-mean input with correlation matrix `C = E[x x^T]`, the stable
/// equilibria of Oja's rule are `w = +-v_1`, the unit principal eigenvector of `C`. The test builds
/// a two-dimensional input whose principal direction is known by construction, runs the rule, and
/// checks both the direction and that `|w|` went to one. Hebbian learning *is* principal component
/// analysis; this is the line of algebra that says so.
#[derive(Debug, Clone, PartialEq)]
pub struct Oja {
    /// Learning rate per sample, dimensionless. The angle to the principal direction decays at
    /// roughly `eta * (lambda_1 - lambda_2)` per sample, so a small `eta` converges slowly and a
    /// large one leaves a residual fluctuation of order `eta` around the fixed point.
    pub eta: f64,
    /// The weight vector, dimensionless, converging to unit length.
    pub w: Vec<f64>,
}

impl Oja {
    /// Build from a starting weight vector, which must not be the zero vector.
    ///
    /// The zero vector is a fixed point — an unstable one — and starting there means never moving,
    /// so it is refused rather than silently producing a flat learning curve.
    ///
    /// # Errors
    ///
    /// [`PlasticityError::Empty`] for an empty or all-zero weight vector,
    /// [`PlasticityError::NonFinite`] for a non-finite weight, [`PlasticityError::Negative`] for a
    /// negative learning rate.
    pub fn new(w: Vec<f64>, eta: f64) -> Result<Self, PlasticityError> {
        if w.is_empty() {
            return Err(PlasticityError::Empty { what: "weight vector" });
        }
        for &wi in &w {
            finite("weight", wi)?;
        }
        if norm(&w) == 0.0 {
            return Err(PlasticityError::Empty { what: "weight vector (all zero, a fixed point)" });
        }
        Ok(Self { eta: non_negative("eta", eta)?, w })
    }

    /// The linear output `w . x`.
    ///
    /// # Errors
    ///
    /// As [`Hebbian::output`].
    pub fn output(&self, x: &[f64]) -> Result<f64, PlasticityError> {
        dot(&self.w, x)
    }

    /// One sample. Returns the output `y` that drove the update.
    ///
    /// # Errors
    ///
    /// As [`Oja::output`], plus [`PlasticityError::Diverged`] if a weight leaves the finite numbers
    /// — possible only with a learning rate large enough to overshoot the fixed point outward,
    /// which is a parameter error and is reported as one rather than clamped.
    pub fn update(&mut self, x: &[f64]) -> Result<f64, PlasticityError> {
        let y = self.output(x)?;
        for i in 0..self.w.len() {
            self.w[i] += self.eta * y * (x[i] - y * self.w[i]);
            if !self.w[i].is_finite() {
                return Err(PlasticityError::Diverged { what: "Oja weight", value: self.w[i] });
            }
        }
        Ok(y)
    }

    /// Euclidean length of the weight vector, which converges to one.
    #[must_use]
    pub fn norm(&self) -> f64 {
        norm(&self.w)
    }
}

/// The `BCM` rule with a sliding modification threshold.
///
/// Bienenstock, Cooper & Munro, J. Neurosci. 2:32–48, 1982, in the form of Intrator & Cooper
/// (Neural Networks 5:3–17, 1992):
///
/// ```text
/// y      = w . x                            postsynaptic rate, hertz
/// dw_i   = eta * dt * x_i * y * (y - theta)
/// theta -> y^2 / y_0    with time constant tau_theta
/// ```
///
/// Output above `theta` potentiates, below it depresses, and `theta` chases the recent mean square
/// of the output — so a cell that fires too much raises its own bar and stops potentiating. That
/// feedback is what makes the rule stable *and* what makes it selective.
///
/// # The selective fixed point, in closed form
///
/// Present `n` mutually orthogonal patterns with equal probability. At the selective equilibrium the
/// cell responds to exactly one of them, at rate `y*`, and to the rest at zero. The threshold then
/// settles at the average `E[y^2] / y_0 = y*^2 / (n * y_0)`, and the selected pattern stops moving
/// when `y* = theta`. Solving the two gives
///
/// ```text
/// y* = n * y_0
/// ```
///
/// which is [`Bcm::selective_fixed_point`] — the response of the winning pattern is the target rate
/// multiplied by the number of patterns competing. A test drives four orthogonal patterns and checks
/// the winner lands there, the other three collapse, and which one wins is decided by the initial
/// weights rather than by the rule.
#[derive(Debug, Clone, PartialEq)]
pub struct Bcm {
    /// Learning rate, units of inverse hertz squared, so that `eta * dt * x * y * (y - theta)` is a
    /// dimensionless weight change.
    pub eta: f64,
    /// Time constant of the sliding threshold, seconds. Must be **slow** relative to the interval
    /// between pattern presentations and **fast** relative to the weight dynamics; if the threshold
    /// moves faster than the weights the cell chases its own tail and never becomes selective.
    pub tau_theta: f64,
    /// Target output scale `y_0`, hertz. Sets where the selective fixed point lands:
    /// `n_patterns * y_0`.
    pub y_0: f64,
    /// Current modification threshold, hertz. Output above it potentiates, below it depresses.
    pub theta: f64,
    /// The interval the weights may never leave; normally floored at zero, since a `BCM` cell's
    /// inputs are excitatory rates.
    pub bounds: Bounds,
    /// The weight vector, dimensionless, one per input channel.
    pub w: Vec<f64>,
}

impl Bcm {
    /// Build from a starting weight vector and a starting threshold.
    ///
    /// # Errors
    ///
    /// [`PlasticityError::Empty`] for an empty weight vector, [`PlasticityError::NotPositive`] for
    /// a non-positive `tau_theta` or `y_0`, [`PlasticityError::Negative`] for a negative `eta` or
    /// `theta`, [`PlasticityError::NonFinite`] for any non-finite parameter or weight.
    pub fn new(
        w: Vec<f64>,
        eta: f64,
        tau_theta: f64,
        y_0: f64,
        theta: f64,
        bounds: Bounds,
    ) -> Result<Self, PlasticityError> {
        if w.is_empty() {
            return Err(PlasticityError::Empty { what: "weight vector" });
        }
        for &wi in &w {
            finite("weight", wi)?;
        }
        Ok(Self {
            eta: non_negative("eta", eta)?,
            tau_theta: positive("tau_theta", tau_theta)?,
            y_0: positive("y_0", y_0)?,
            theta: non_negative("theta", theta)?,
            bounds,
            w,
        })
    }

    /// The postsynaptic rate `w . x`, hertz.
    ///
    /// # Errors
    ///
    /// As [`Hebbian::output`].
    pub fn output(&self, x: &[f64]) -> Result<f64, PlasticityError> {
        dot(&self.w, x)
    }

    /// The winning pattern's response at the selective equilibrium, hertz: `n_patterns * y_0`.
    ///
    /// `None` for zero patterns, where the equilibrium is not defined — there is nothing to be
    /// selective *among*, and returning `y_0` or zero would both be inventions.
    ///
    /// Valid only for mutually orthogonal, equiprobable patterns. With overlapping patterns the
    /// threshold still slides but the equilibrium is a fixed point of a coupled system with no such
    /// one-line solution, and this function does not apply.
    #[must_use]
    pub fn selective_fixed_point(&self, n_patterns: usize) -> Option<f64> {
        if n_patterns == 0 {
            return None;
        }
        Some(n_patterns as f64 * self.y_0)
    }

    /// Present one pattern for `dt` seconds. Returns the output rate `y` in hertz.
    ///
    /// The threshold relaxes toward `y^2 / y_0` **exponentially** rather than by a forward-Euler
    /// step, for the reason [`crate::neuron::Lif`] does the same: the closed form is available, it
    /// costs one `exp`, and it cannot go unstable when `dt` approaches `tau_theta`.
    ///
    /// # Errors
    ///
    /// As [`Bcm::output`], plus [`PlasticityError::Negative`] for a negative `dt`.
    pub fn update(&mut self, x: &[f64], dt: f64) -> Result<f64, PlasticityError> {
        let dt = non_negative("dt", dt)?;
        let y = self.output(x)?;
        let phi = y * (y - self.theta);
        for i in 0..self.w.len() {
            self.w[i] = self.bounds.clamp(self.w[i] + self.eta * dt * x[i] * phi);
        }
        let target = y * y / self.y_0;
        self.theta = target + (self.theta - target) * (-dt / self.tau_theta).exp();
        Ok(y)
    }
}

fn dot(w: &[f64], x: &[f64]) -> Result<f64, PlasticityError> {
    if x.len() != w.len() {
        return Err(PlasticityError::LengthMismatch {
            what: "input vector",
            got: x.len(),
            want: w.len(),
        });
    }
    let mut y = 0.0;
    for i in 0..w.len() {
        finite("input", x[i])?;
        y += w[i] * x[i];
    }
    Ok(y)
}

fn norm(w: &[f64]) -> f64 {
    let mut s = 0.0;
    for &wi in w {
        s += wi * wi;
    }
    s.sqrt()
}

/// Reward-modulated `STDP`: the three-factor rule, for the distal reward problem.
///
/// Izhikevich, Cereb. Cortex 17:2443–2452, 2007. The two local factors are the pre and post spike
/// trains; the third is a global neuromodulator signal, which in the paper stands for dopamine.
///
/// ```text
/// dc/dt = -c / tau_c  +  STDP window at each spike     eligibility trace
/// dd/dt = -d / tau_d  +  reward deliveries             modulator concentration
/// dw/dt =  c * d                                       the weight moves only when both are on
/// ```
///
/// **The problem this solves.** A reward arrives a second after the spike pattern that earned it.
/// By then the spike traces have decayed and every synapse in the network has fired many more
/// times, so a rule that writes the weight at spike time has already forgotten which synapse to
/// credit, and a rule that waits has nothing to read. The eligibility trace is the answer: `STDP`
/// tags the synapse and the tag decays over about a second — long enough to still be there when the
/// reward arrives, short enough that unrelated later activity is not credited.
///
/// **The consequence worth stating.** With `d = 0` the product is zero and the weight does not move
/// **at all**, however much correlated activity the synapse sees. `c` still evolves. That is not a
/// numerical approximation; it is exactly zero, and the test asserts bit-for-bit equality of the
/// weight before and after.
///
/// # The closed form
///
/// Deliver one pre/post pair, wait, then deliver an impulse of modulator `d_0` at a moment when the
/// eligibility trace stands at `c`. Integrating `c * d` from there to infinity gives, exactly,
///
/// ```text
/// delta_w = c * d_0 * tau_c * tau_d / (tau_c + tau_d)
/// ```
///
/// [`RewardStdp::advance`] integrates that product **in closed form over each step** rather than by
/// forward Euler, so the accumulated total matches the expression above to floating-point noise
/// regardless of the step size — which is what the test checks.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RewardStdp {
    /// The window that writes the eligibility trace. Its own bounds and weight rule are used for
    /// the weight dependence of the tag; the weight itself is clamped by
    /// [`RewardStdp::bounds`].
    pub stdp: PairStdp,
    /// Eligibility trace decay, seconds. Izhikevich uses about 1 s, chosen to match the behavioural
    /// delay between action and reward rather than any membrane time constant.
    pub tau_c: f64,
    /// Modulator decay, seconds. Izhikevich uses about 0.2 s.
    pub tau_d: f64,
    /// Current eligibility, in weight units. Signed: a post-before-pre pair tags the synapse for
    /// depression, and a reward then depresses it.
    pub c: f64,
    /// Current modulator concentration, dimensionless. Signed, so a negative delivery is a
    /// punishment and reverses every tagged change.
    pub d: f64,
    /// The interval the weight may never leave.
    pub bounds: Bounds,
}

impl RewardStdp {
    /// Build from an `STDP` window and the two time constants.
    ///
    /// # Errors
    ///
    /// [`PlasticityError::NotPositive`] for a non-positive time constant,
    /// [`PlasticityError::NonFinite`] for a non-finite one.
    pub fn new(
        stdp: PairStdp,
        tau_c: f64,
        tau_d: f64,
        bounds: Bounds,
    ) -> Result<Self, PlasticityError> {
        Ok(Self {
            stdp,
            tau_c: positive("tau_c", tau_c)?,
            tau_d: positive("tau_d", tau_d)?,
            c: 0.0,
            d: 0.0,
            bounds,
        })
    }

    /// The effective time constant of the `c * d` product: `tau_c * tau_d / (tau_c + tau_d)`.
    ///
    /// Always shorter than either, which is the reason a reward delivered late is worth less than
    /// the same reward delivered early even though neither trace has vanished.
    #[must_use]
    pub fn tau_effective(&self) -> f64 {
        self.tau_c * self.tau_d / (self.tau_c + self.tau_d)
    }

    /// The total weight change that would follow from the present `c` and `d` if nothing else ever
    /// happened: `c * d * tau_effective`.
    ///
    /// Exposed because it is the closed form the test compares an integrated run against, and
    /// because it answers "how much is this tag worth" without running anything.
    #[must_use]
    pub fn pending_change(&self) -> f64 {
        self.c * self.d * self.tau_effective()
    }

    /// A presynaptic spike: tag the synapse for depression, weighted by the current weight
    /// dependence. The weight itself does not move.
    ///
    /// # Errors
    ///
    /// [`PlasticityError::NonFinite`] for a non-finite weight.
    pub fn on_pre(&mut self, w: f64) -> Result<(), PlasticityError> {
        let w = finite("weight", w)?;
        self.c += self.stdp.pre_increment(w);
        self.stdp.note_pre();
        Ok(())
    }

    /// A postsynaptic spike: tag the synapse for potentiation. The weight does not move.
    ///
    /// # Errors
    ///
    /// [`PlasticityError::NonFinite`] for a non-finite weight.
    pub fn on_post(&mut self, w: f64) -> Result<(), PlasticityError> {
        let w = finite("weight", w)?;
        self.c += self.stdp.post_increment(w);
        self.stdp.note_post();
        Ok(())
    }

    /// Deliver an impulse of modulator. Negative is a punishment.
    ///
    /// # Errors
    ///
    /// [`PlasticityError::NonFinite`] for a non-finite amount.
    pub fn reward(&mut self, amount: f64) -> Result<(), PlasticityError> {
        self.d += finite("reward", amount)?;
        Ok(())
    }

    /// Advance `dt` seconds: apply the exact integral of `c * d` to the weight, then decay both
    /// traces and the `STDP` window's traces.
    ///
    /// Returns the new weight, clamped into [`RewardStdp::bounds`].
    ///
    /// # Errors
    ///
    /// [`PlasticityError::NonFinite`] for a non-finite weight, [`PlasticityError::Negative`] or
    /// [`PlasticityError::NonFinite`] for a `dt` that is not finite and non-negative.
    pub fn advance(&mut self, w: f64, dt: f64) -> Result<f64, PlasticityError> {
        let w = finite("weight", w)?;
        let dt = non_negative("dt", dt)?;
        // The exact integral of c(0) d(0) exp(-s / tau_eff) over [0, dt]. Forward Euler would be
        // `c * d * dt`, which over-counts by half a step every step and therefore over-credits
        // exactly the synapses that were tagged hardest.
        let tau_eff = self.tau_effective();
        let dw = self.c * self.d * tau_eff * (1.0 - (-dt / tau_eff).exp());
        let out = self.bounds.clamp(w + dw);
        self.c *= (-dt / self.tau_c).exp();
        self.d *= (-dt / self.tau_d).exp();
        self.stdp.advance(dt)?;
        Ok(out)
    }

    /// Forget the tag, the modulator and the spike history. Parameters untouched.
    pub fn clear(&mut self) {
        self.c = 0.0;
        self.d = 0.0;
        self.stdp.clear();
    }
}

/// Homeostatic synaptic scaling: multiplicative, slow, and ratio-preserving.
///
/// Turrigiano, Leslie, Desai, Rutherford & Nelson, Nature 391:892–896, 1998. Cortical neurons
/// deprived of activity for two days scaled all their excitatory inputs **up**, and the measured
/// `mEPSC` amplitude distribution *scaled* — it did not shift. Every synapse was multiplied by the
/// same factor, so the relative differences between them survived.
///
/// That is the mechanism implemented here, in the form of van Rossum, Bi & Turrigiano (J. Neurosci.
/// 20:8812–8821, 2000):
///
/// ```text
/// w_i <- w_i * exp( dt * (y_target - rate) / (y_target * tau) )
/// ```
///
/// with `rate` a low-pass estimate of the cell's own firing rate. A silent cell doubles its weights
/// in `tau * ln 2` seconds; a cell at twice its target halves them in the same time.
///
/// # Why multiplicative matters
///
/// Hebbian rules are unstable and homeostasis is the brake — but a brake that pulled every weight
/// toward a common value would erase what the Hebbian rule learned. Multiplicative scaling changes
/// the gain without changing the pattern, so selectivity survives. [`SynapticScaling::scale`]
/// preserves every pairwise ratio to floating-point noise, and the test asserts that.
///
/// **The one caveat, stated beside the claim:** the ratio guarantee holds only while no weight hits
/// a bound. A clamped weight has been multiplied by a different factor than its neighbours, and the
/// ratios are then gone. [`SynapticScaling::scale`] returns the factor it applied so a caller can
/// tell; there is no way to have both an unconditional bound and exact multiplicativity, and this
/// implementation chooses the bound.
///
/// # Time scale
///
/// In the 1998 experiment `tau` is on the order of **days**. Simulations routinely shorten it by
/// four orders of magnitude to make a run finish, which is a legitimate thing to do and a dishonest
/// thing to leave unstated, because the separation of time scales between Hebbian learning and
/// homeostasis is the assumption the whole arrangement rests on. State the value you used.
#[derive(Debug, Clone, PartialEq)]
pub struct SynapticScaling {
    /// Target firing rate, hertz. The cell scales until its own rate reaches this.
    pub target_hz: f64,
    /// Homeostatic time constant, seconds. A silent cell's weights e-fold in `tau` seconds.
    pub tau: f64,
    /// Time constant of the firing-rate estimator, seconds. Must be long compared with the
    /// inter-spike interval or the estimate is a spike train rather than a rate.
    pub rate_tau: f64,
    /// Current firing-rate estimate, hertz.
    pub rate: f64,
    /// The interval the weights may never leave. A clamp here breaks multiplicativity; see the type
    /// doc.
    pub bounds: Bounds,
    /// The incoming weights of one postsynaptic cell. Scaling is a **per-cell** operation: it is the
    /// cell's own rate that drives it, so a weight vector here is one neuron's afferents.
    pub w: Vec<f64>,
}

impl SynapticScaling {
    /// Build from a cell's incoming weights.
    ///
    /// # Errors
    ///
    /// [`PlasticityError::Empty`] for an empty weight vector, [`PlasticityError::NotPositive`] for
    /// a non-positive `target_hz`, `tau` or `rate_tau`, [`PlasticityError::Negative`] for a
    /// negative starting rate, [`PlasticityError::NonFinite`] for a non-finite weight.
    pub fn new(
        w: Vec<f64>,
        target_hz: f64,
        tau: f64,
        rate_tau: f64,
        bounds: Bounds,
    ) -> Result<Self, PlasticityError> {
        if w.is_empty() {
            return Err(PlasticityError::Empty { what: "weight vector" });
        }
        for &wi in &w {
            finite("weight", wi)?;
        }
        Ok(Self {
            target_hz: positive("target_hz", target_hz)?,
            tau: positive("tau", tau)?,
            rate_tau: positive("rate_tau", rate_tau)?,
            rate: 0.0,
            bounds,
            w,
        })
    }

    /// Fold `spikes` spikes observed over `dt` seconds into the rate estimate. Returns the new
    /// estimate, hertz.
    ///
    /// The estimator relaxes exponentially toward the instantaneous rate `spikes / dt`, so under a
    /// constant input rate `r` it follows `r + (rate_0 - r) * exp(-t / rate_tau)` exactly — which is
    /// the closed form the test checks it against.
    ///
    /// # Errors
    ///
    /// [`PlasticityError::NotPositive`] for a `dt` of zero or less: a spike count over no elapsed
    /// time has no rate, and dividing would produce an infinity.
    pub fn observe(&mut self, spikes: u32, dt: f64) -> Result<f64, PlasticityError> {
        let dt = positive("dt", dt)?;
        let instant = f64::from(spikes) / dt;
        self.rate = instant + (self.rate - instant) * (-dt / self.rate_tau).exp();
        Ok(self.rate)
    }

    /// The factor [`SynapticScaling::scale`] would apply over `dt` seconds at the present rate
    /// estimate. Above one when the cell is below target.
    #[must_use]
    pub fn factor(&self, dt: f64) -> f64 {
        (dt * (self.target_hz - self.rate) / (self.target_hz * self.tau)).exp()
    }

    /// Multiply every weight by [`SynapticScaling::factor`] and clamp. Returns the factor applied.
    ///
    /// # Errors
    ///
    /// [`PlasticityError::Negative`] or [`PlasticityError::NonFinite`] for a `dt` that is not
    /// finite and non-negative, and [`PlasticityError::Diverged`] if the factor itself is not
    /// finite — which happens only for a rate estimate so far from target that the exponent
    /// overflows, and is reported rather than silently turned into an infinity.
    pub fn scale(&mut self, dt: f64) -> Result<f64, PlasticityError> {
        let dt = non_negative("dt", dt)?;
        let g = self.factor(dt);
        if !g.is_finite() {
            return Err(PlasticityError::Diverged { what: "scaling factor", value: g });
        }
        for i in 0..self.w.len() {
            self.w[i] = self.bounds.clamp(self.w[i] * g);
        }
        Ok(g)
    }

    /// Sum of the weights: the cell's total synaptic drive, which is what scaling regulates.
    #[must_use]
    pub fn total(&self) -> f64 {
        self.w.iter().sum()
    }

    /// Whether the rate estimate is within `tol_hz` of the target.
    ///
    /// Settled means the *estimate* has arrived, not that the cell has. With a `rate_tau` short
    /// compared with the inter-spike interval this returns `true` and `false` alternately on every
    /// spike, which is a statement about the estimator rather than about the homeostat.
    #[must_use]
    pub fn is_settled(&self, tol_hz: f64) -> bool {
        (self.rate - self.target_hz).abs() <= tol_hz
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Bcm, Bounds, Hebbian, Oja, PairStdp, Pairing, PlasticityError, RewardStdp, SynapticScaling,
        Trace, TripletStdp, WeightRule, norm,
    };
    use crate::rng::Rng;

    fn wide_pair() -> PairStdp {
        PairStdp::new(0.1, 0.105, 16.8e-3, 33.7e-3, WeightRule::Additive, Bounds::wide()).unwrap()
    }

    /// (a) THE SHARPEST CHECK IN THE MODULE. One isolated pair at a known lag must reproduce the
    /// window `A_plus * exp(-lag / tau_plus)` — not to a tolerance, to the last bit, because the
    /// online trace path and the closed form are the same two floating-point operations in the same
    /// order. Both signs, six lags each.
    ///
    /// The weight starts at exactly zero so that `w + dw` is `dw` with no rounding of its own; at
    /// `w = 0.37` the addition would round and the comparison would be testing `f64` addition
    /// rather than the rule.
    #[test]
    fn a_single_pair_reproduces_the_exponential_window_exactly() {
        let mut s = wide_pair();
        for &lag_ms in &[0.5, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0] {
            let lag = lag_ms * 1e-3;

            let got = s.apply_pair(0.0, lag).unwrap();
            let want = s.window(lag);
            assert_eq!(got, want, "pre->post at {lag_ms} ms: {got} vs closed form {want}");
            assert!(want > 0.0, "pre before post must potentiate at {lag_ms} ms");

            let got = s.apply_pair(0.0, -lag).unwrap();
            let want = s.window(-lag);
            assert_eq!(got, want, "post->pre at -{lag_ms} ms: {got} vs closed form {want}");
            assert!(want < 0.0, "post before pre must depress at -{lag_ms} ms");
        }
    }

    /// The window's shape, independent of the online path: potentiation must fall off with
    /// `tau_plus` and depression with `tau_minus`, and one time constant out must leave exactly
    /// `1/e` of the peak.
    #[test]
    fn the_window_decays_by_one_e_fold_per_time_constant() {
        let s = wide_pair();
        let e = std::f64::consts::E;
        // RELATIVE, at a few ulp. `a_minus / e` is about 0.0386, whose ulp is 6.9e-18, so the
        // first draft's absolute 1e-18 was asking for better than the representation allows and
        // failed on the depression side alone -- which reads as an asymmetry bug and is not one.
        assert!((s.window(s.tau_plus) - s.a_plus / e).abs() < 1e-15 * s.a_plus);
        assert!((s.window(-s.tau_minus) + s.a_minus / e).abs() < 1e-15 * s.a_minus);
        // And the two sides are genuinely different widths, which is the asymmetry Bi & Poo saw.
        assert!(s.tau_minus > s.tau_plus);
    }

    /// (b) The analytic integral against midpoint quadrature of the window itself. The grid has an
    /// even number of cells so no sample lands on the discontinuity at zero, where the window's
    /// value is a convention rather than a measurement.
    #[test]
    fn the_window_integral_matches_numerical_quadrature() {
        let s = wide_pair();
        for &t_ms in &[10.0, 50.0, 200.0, 1000.0] {
            let t = t_ms * 1e-3;
            let n = 2_000_000u32;
            let h = 2.0 * t / f64::from(n);
            let mut acc = 0.0;
            for k in 0..n {
                let x = -t + (f64::from(k) + 0.5) * h;
                acc += s.window(x);
            }
            acc *= h;
            let want = s.window_integral(t).unwrap();
            assert!(
                (acc - want).abs() < 1e-9 * want.abs().max(1e-6),
                "half-width {t_ms} ms: quadrature {acc} vs closed form {want}"
            );
        }
        // And the limit the finite integral approaches.
        let far = s.window_integral(100.0).unwrap();
        assert!((far - s.total_window_area()).abs() < 1e-15);
    }

    /// Song, Miller & Abbott's stability condition, on their own parameter set. Their whole
    /// mechanism is a 5% amplitude asymmetry with equal time constants, so if the ratio were
    /// transcribed as 1.00 the check would flip — which is exactly what this is here to catch.
    #[test]
    fn the_song_abbott_parameter_set_is_depression_dominated() {
        let s = PairStdp::song_abbott_2000(1.0).unwrap();
        assert!(s.is_depression_dominated(), "area {}", s.total_window_area());
        assert!((s.a_minus / s.a_plus - 1.05).abs() < 1e-12);
        assert_eq!(s.tau_plus, s.tau_minus);
        assert_eq!(s.bounds, Bounds::new(0.0, 1.0).unwrap());
        // A rule with equal amplitudes and equal time constants is exactly on the line, and "not
        // negative" is the honest verdict there rather than "stable".
        let flat = PairStdp::new(0.1, 0.1, 20e-3, 20e-3, WeightRule::Additive, Bounds::wide())
            .unwrap();
        assert!(!flat.is_depression_dominated());
        assert_eq!(flat.total_window_area(), 0.0);
    }

    /// `x.powf(0.0)` is one for every `x`, so a soft bound with a zero exponent is not merely close
    /// to the additive rule — it is the same arithmetic. Bit-for-bit over a random spike train.
    #[test]
    fn soft_bounds_with_exponent_zero_are_bit_identical_to_the_additive_rule() {
        let b = Bounds::new(-2.0, 2.0).unwrap();
        let mut add =
            PairStdp::new(0.05, 0.052, 16.8e-3, 33.7e-3, WeightRule::Additive, b).unwrap();
        let mut soft =
            PairStdp::new(0.05, 0.052, 16.8e-3, 33.7e-3, WeightRule::SoftBound { mu: 0.0 }, b)
                .unwrap();
        let mut rng = Rng::new(0xB177_0000_C0DE);
        let (mut wa, mut ws) = (0.3, 0.3);
        for _ in 0..4_000 {
            let dt = rng.next_f64() * 20e-3;
            add.advance(dt).unwrap();
            soft.advance(dt).unwrap();
            if rng.next_f64() < 0.5 {
                wa = add.on_pre(wa).unwrap();
                ws = soft.on_pre(ws).unwrap();
            } else {
                wa = add.on_post(wa).unwrap();
                ws = soft.on_post(ws).unwrap();
            }
            assert_eq!(wa, ws, "additive {wa} vs soft-bound mu=0 {ws}");
        }
        // Two identical constants are also equal. Assert the run went somewhere.
        assert!((wa - 0.3).abs() > 0.01, "the equivalence test never moved the weight ({wa})");
    }

    /// (e) Bounds under adversarial input: amplitudes far larger than the interval, alternating
    /// spikes with no time between them, every weight rule. The clamp is unconditional precisely so
    /// that this cannot fail.
    #[test]
    fn weight_bounds_are_never_violated_under_adversarial_input() {
        let b = Bounds::new(-0.25, 0.75).unwrap();
        let rules = [
            WeightRule::Additive,
            WeightRule::MultiplicativeDepression,
            WeightRule::SoftBound { mu: 0.0 },
            WeightRule::SoftBound { mu: 0.5 },
            WeightRule::SoftBound { mu: 1.0 },
            WeightRule::SoftBound { mu: 3.0 },
        ];
        let mut rng = Rng::new(4_242);
        for rule in rules {
            // Amplitudes 1000x the span of the interval: one spike should overshoot both bounds.
            let mut s = PairStdp::new(1000.0, 900.0, 5e-3, 7e-3, rule, b).unwrap();
            let mut w = 0.0;
            let (mut hit_floor, mut hit_ceiling) = (false, false);
            for _ in 0..20_000 {
                // Zero gaps included on purpose: they leave the traces un-decayed and maximise the
                // step, which is the adversarial case.
                s.advance(if rng.next_f64() < 0.3 { 0.0 } else { rng.next_f64() * 1e-3 }).unwrap();
                w = if rng.next_f64() < 0.5 { s.on_pre(w).unwrap() } else { s.on_post(w).unwrap() };
                assert!(b.contains(w), "{rule:?} left the bound at w = {w}");
                hit_floor |= w == b.w_min;
                hit_ceiling |= w == b.w_max;
            }
            // A bound nothing ever reaches is a bound nothing ever tested. `MultiplicativeDepression`
            // approaches its floor asymptotically by construction, so it is the one rule allowed to
            // miss it; every other rule has to have pinned both ends.
            assert!(hit_ceiling, "{rule:?} never reached the ceiling, so the clamp was never tried");
            assert!(
                hit_floor || rule == WeightRule::MultiplicativeDepression,
                "{rule:?} never reached the floor"
            );
        }
        // The triplet rule has to hold the same line.
        let mut t = TripletStdp::new(
            500.0, 500.0, 500.0, 500.0, 5e-3, 50e-3, 7e-3, 60e-3, WeightRule::Additive, b,
        )
        .unwrap();
        let mut w = 0.5;
        for _ in 0..20_000 {
            t.advance(rng.next_f64() * 1e-3).unwrap();
            w = if rng.next_f64() < 0.5 { t.on_pre(w).unwrap() } else { t.on_post(w).unwrap() };
            assert!(b.contains(w), "triplet left the bound at w = {w}");
        }
    }

    /// Multiplicative depression must actually slow down near the floor, or the name is decorative.
    /// Van Rossum et al. (2000) is the reason the steady-state weight distribution is unimodal
    /// rather than piled at both bounds.
    #[test]
    fn multiplicative_depression_shrinks_as_the_weight_approaches_the_floor() {
        let b = Bounds::new(0.0, 1.0).unwrap();
        let mut s = PairStdp::new(
            0.1,
            0.1,
            16.8e-3,
            33.7e-3,
            WeightRule::MultiplicativeDepression,
            b,
        )
        .unwrap();
        let step_at = |s: &mut PairStdp, w: f64| {
            let after = s.apply_pair(w, -5e-3).unwrap();
            w - after
        };
        let near_top = step_at(&mut s, 0.9);
        let mid = step_at(&mut s, 0.5);
        let near_floor = step_at(&mut s, 0.01);
        assert!(near_top > mid && mid > near_floor, "{near_top} {mid} {near_floor}");
        // Ratio is the weight ratio, exactly: the factor is linear in `w - w_min`.
        assert!((near_top / mid - 1.8).abs() < 1e-12);
        // Potentiation is weight-independent in this rule, which is the asymmetry that defines it.
        let p_low = s.apply_pair(0.01, 5e-3).unwrap() - 0.01;
        let p_high = s.apply_pair(0.9, 5e-3).unwrap() - 0.9;
        assert!((p_low - p_high).abs() < 1e-15, "{p_low} vs {p_high}");
    }

    /// (f) EXACT EQUIVALENCE. With both triplet amplitudes zero, the triplet rule's inner factor is
    /// `A2 + 0.0 * trace`, which is `A2` exactly, and the remaining product is the pair rule's with
    /// its operands swapped — and IEEE multiplication is commutative. So this is an equality, not a
    /// tolerance. If it ever needs a tolerance, the two rules have stopped being the same model.
    #[test]
    fn the_triplet_rule_reduces_exactly_to_the_pair_rule_when_the_triplet_amplitudes_are_zero() {
        let b = Bounds::new(-1.0, 1.0).unwrap();
        for pairing in [Pairing::AllToAll, Pairing::NearestNeighbour] {
            let mut pair =
                PairStdp::new(0.017, 0.023, 16.8e-3, 33.7e-3, WeightRule::Additive, b).unwrap();
            pair.pairing = pairing;
            let mut trip = TripletStdp::new(
                0.017, 0.0, 0.023, 0.0, 16.8e-3, 101e-3, 33.7e-3, 125e-3, WeightRule::Additive, b,
            )
            .unwrap();
            trip.pairing = pairing;

            let mut rng = Rng::new(0x7A1B_5EED);
            let (mut wp, mut wt) = (0.0, 0.0);
            for k in 0..20_000 {
                let dt = rng.next_f64() * 30e-3;
                pair.advance(dt).unwrap();
                trip.advance(dt).unwrap();
                if rng.next_f64() < 0.5 {
                    wp = pair.on_pre(wp).unwrap();
                    wt = trip.on_pre(wt).unwrap();
                } else {
                    wp = pair.on_post(wp).unwrap();
                    wt = trip.on_post(wt).unwrap();
                }
                assert_eq!(wp, wt, "{pairing:?} diverged at event {k}: pair {wp}, triplet {wt}");
            }
            // And the run has to have gone somewhere, or the equality is between two zeros.
            assert!(wp.abs() > 1e-6, "the reduction test never moved the weight ({wp})");
        }
    }

    /// Replay one pre/post pair at `freq` hertz, `n` times, at lag `lag` seconds, and return the
    /// mean weight change per pair. Event-driven, so the timing carries no discretisation error.
    fn pairing_protocol<F>(mut on_event: F, freq: f64, lag: f64, n: u32) -> f64
    where
        F: FnMut(f64, bool, f64) -> f64,
    {
        let mut events: Vec<(f64, bool)> = Vec::with_capacity(2 * n as usize);
        for k in 0..n {
            let t = f64::from(k) / freq;
            events.push((t, true));
            events.push((t + lag, false));
        }
        events.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let mut w = 0.0;
        let mut last = 0.0;
        for (t, is_pre) in events {
            w = on_event(t - last, is_pre, w);
            last = t;
        }
        w / f64::from(n)
    }

    /// The headline claim of Pfister & Gerstner (2006), reproduced. Sjöström, Turrigiano & Nelson
    /// (Neuron 32:1149–1164, 2001) repeated a `+10 ms` pre-before-post pair from 0.1 Hz to 50 Hz and
    /// found potentiation that GROWS strongly with frequency.
    ///
    /// The pair rule goes the other way: its residual post trace makes every presynaptic spike in a
    /// fast train depress, so the net change at 40 Hz is a fraction of the change at 1 Hz. The
    /// minimal triplet rule, whose potentiation is proportional to how much postsynaptic activity
    /// preceded the post spike, grows by more than three orders of magnitude over the same range and
    /// is monotone in frequency.
    #[test]
    fn the_pair_rule_cannot_reproduce_the_frequency_dependence_the_triplet_rule_can() {
        let b = Bounds::wide();
        let lag = 10e-3;
        let freqs = [1.0, 5.0, 10.0, 20.0, 40.0, 50.0];

        let pair_dw = |f: f64| {
            let mut s = PairStdp::new(0.005, 0.00525, 20e-3, 20e-3, WeightRule::Additive, b)
                .unwrap();
            pairing_protocol(
                |dt, is_pre, w| {
                    s.advance(dt).unwrap();
                    if is_pre { s.on_pre(w).unwrap() } else { s.on_post(w).unwrap() }
                },
                f,
                lag,
                60,
            )
        };
        let trip_dw = |f: f64| {
            let mut s = TripletStdp::visual_cortex_minimal(b).unwrap();
            pairing_protocol(
                |dt, is_pre, w| {
                    s.advance(dt).unwrap();
                    if is_pre { s.on_pre(w).unwrap() } else { s.on_post(w).unwrap() }
                },
                f,
                lag,
                60,
            )
        };

        let p: Vec<f64> = freqs.iter().map(|&f| pair_dw(f)).collect();
        let t: Vec<f64> = freqs.iter().map(|&f| trip_dw(f)).collect();

        // Measured here, per pair, over 60 pairs:
        //
        //     1 Hz    pair  +3.033e-3    triplet  +1.183e-6
        //    10 Hz    pair  +2.995e-3    triplet  +2.335e-3
        //    40 Hz    pair  +8.274e-4    triplet  +1.057e-2
        //    50 Hz    pair  -1.536e-4    triplet  +1.497e-2
        //
        // The pair rule's change SHRINKS with frequency and CHANGES SIGN at 50 Hz -- it predicts
        // depression exactly where Sjostrom et al. measured the strongest potentiation. That is the
        // failure, stated as an assertion rather than as prose.
        assert!(p[0] > 0.0, "pair rule should potentiate at 1 Hz, got {}", p[0]);
        assert!(p[4] < 0.5 * p[0], "pair rule at 40 Hz {} vs at 1 Hz {}", p[4], p[0]);
        assert!(p[5] < 0.0, "pair rule at 50 Hz {} did not cross into depression", p[5]);

        // The triplet rule's grows, and grows monotonically.
        for k in 1..t.len() {
            assert!(t[k] > t[k - 1], "triplet not monotone at {} Hz: {t:?}", freqs[k]);
        }
        assert!(t[4] > 1_000.0 * t[0], "triplet 40 Hz {} vs 1 Hz {}", t[4], t[0]);
        assert!(t[4] > 0.0 && t[0] > 0.0);
    }

    /// (Hebb) The exact geometric closed form. Under a constant unit input the weight along that
    /// input is multiplied by `1 + eta` every sample, so after `n` samples the norm is
    /// `|w_0| * (1 + eta)^n` — no approximation anywhere in the derivation.
    #[test]
    fn hebbian_growth_matches_the_exact_geometric_closed_form() {
        let u = [0.6, 0.8];
        let eta = 0.01;
        let w0 = 0.1;
        let mut h = Hebbian::new(vec![w0 * u[0], w0 * u[1]], eta, 0.0).unwrap();
        let n = 200u32;
        for _ in 0..n {
            h.update(&u).unwrap();
        }
        let want = w0 * (1.0 + eta).powi(n as i32);
        let got = h.norm();
        assert!((got - want).abs() / want < 1e-9, "norm {got} vs closed form {want}");
        // 7.3x in 200 samples, and it does not stop. That is the instability, not a tuning problem.
        assert!(got > 7.0 * w0);
    }

    /// And it really does run away: pure Hebb refuses with a named divergence rather than handing
    /// back an infinity.
    #[test]
    fn pure_hebbian_learning_diverges_where_oja_does_not() {
        let u = [0.6, 0.8];
        let mut h = Hebbian::new(vec![0.06, 0.08], 0.5, 0.0).unwrap();
        let mut hit = None;
        for k in 0..5_000u32 {
            if let Err(e) = h.update(&u) {
                hit = Some((k, e));
                break;
            }
        }
        let (k, e) = hit.expect("pure Hebbian learning did not diverge in 5000 samples");
        assert!(matches!(e, PlasticityError::Diverged { .. }), "{e}");
        assert!(k > 100, "diverged implausibly fast, at sample {k}");

        // Same input, same starting point, Oja's rule: bounded at unit norm forever.
        let mut o = Oja::new(vec![0.06, 0.08], 0.5).unwrap();
        for _ in 0..5_000 {
            o.update(&u).unwrap();
        }
        assert!((o.norm() - 1.0).abs() < 1e-9, "Oja norm {}", o.norm());
    }

    /// (c, part 1) Oja's rule under a constant unit input collapses exactly to the scalar map
    /// `c <- c + eta * c * (1 - c^2)`. The test iterates that map independently, in one dimension,
    /// and compares it against the two-dimensional rule's norm at every step.
    #[test]
    fn oja_reduces_to_the_scalar_map_whose_stable_fixed_point_is_unit_norm() {
        let u = [0.6, 0.8];
        let eta = 0.05;
        let mut o = Oja::new(vec![0.1 * u[0], 0.1 * u[1]], eta).unwrap();
        let mut c = 0.1;
        for k in 0..400u32 {
            o.update(&u).unwrap();
            c += eta * c * (1.0 - c * c);
            assert!(
                (o.norm() - c).abs() < 1e-12,
                "step {k}: vector rule {} vs scalar map {c}",
                o.norm()
            );
        }
        assert!((c - 1.0).abs() < 1e-9, "the scalar map did not reach its fixed point: {c}");
    }

    /// The unit principal direction is a fixed point: presented with its own input, the rule does
    /// not move. This is what "converges to the eigenvector" means before any statistics are
    /// involved.
    #[test]
    fn the_unit_principal_direction_is_a_fixed_point_of_ojas_rule() {
        let u = [0.6, 0.8];
        let mut o = Oja::new(vec![u[0], u[1]], 0.3).unwrap();
        let before = o.w.clone();
        for _ in 0..1_000 {
            o.update(&[5.0 * u[0], 5.0 * u[1]]).unwrap();
        }
        for i in 0..2 {
            assert!((o.w[i] - before[i]).abs() < 1e-14, "moved: {:?} from {before:?}", o.w);
        }
    }

    /// (c, part 2) THE EIGENVECTOR CHECK. Build a two-dimensional input whose correlation matrix is
    /// `C = (a^2/3) u u^T + (b^2/3) v v^T` with `u` and `v` orthonormal and `a > b`, so the
    /// principal direction is `u` by construction and the eigenvalues are known. Oja's rule must
    /// find `u` and must arrive at unit norm.
    ///
    /// The learning rate is dropped part-way through: the residual fluctuation about the fixed
    /// point scales with `eta`, so a single rate either converges slowly or lands noisily. That is a
    /// property of stochastic approximation, not of this implementation, and saying so is cheaper
    /// than a loose tolerance.
    #[test]
    fn oja_converges_to_the_principal_eigenvector_and_to_unit_norm() {
        let u = [0.6, 0.8];
        let v = [-0.8, 0.6];
        let (a, b) = (1.0, 0.3);
        let (l1, l2) = (a * a / 3.0, b * b / 3.0);

        // Start deliberately near the MINOR direction, so a test that merely failed to move would
        // fail rather than pass.
        let mut o = Oja::new(vec![0.05 * v[0], 0.05 * v[1]], 0.02).unwrap();
        let start_align = (o.w[0] * u[0] + o.w[1] * u[1]).abs() / o.norm();
        assert!(start_align < 0.01, "the test started aligned already ({start_align})");

        let mut rng = Rng::new(0x0A51_9EED);
        let sample = |rng: &mut Rng| {
            let s = a * (2.0 * rng.next_f64() - 1.0);
            let n = b * (2.0 * rng.next_f64() - 1.0);
            [s * u[0] + n * v[0], s * u[1] + n * v[1]]
        };
        for _ in 0..60_000 {
            let x = sample(&mut rng);
            o.update(&x).unwrap();
        }
        o.eta = 5e-4;
        for _ in 0..120_000 {
            let x = sample(&mut rng);
            o.update(&x).unwrap();
        }

        let n = o.norm();
        assert!((n - 1.0).abs() < 5e-3, "norm {n} did not converge to 1");
        let align = (o.w[0] * u[0] + o.w[1] * u[1]).abs() / n;
        assert!(align > 0.999, "alignment with the principal direction {align}");

        // And the eigenvalue equation itself: C w must be parallel to w, with the Rayleigh
        // quotient at the leading eigenvalue rather than the trailing one.
        let cw = [
            l1 * u[0] * (u[0] * o.w[0] + u[1] * o.w[1]) + l2 * v[0] * (v[0] * o.w[0] + v[1] * o.w[1]),
            l1 * u[1] * (u[0] * o.w[0] + u[1] * o.w[1]) + l2 * v[1] * (v[0] * o.w[0] + v[1] * o.w[1]),
        ];
        let rayleigh = cw[0] * o.w[0] + cw[1] * o.w[1];
        let resid = norm(&[cw[0] - rayleigh * o.w[0], cw[1] - rayleigh * o.w[1]]);
        assert!(resid < 1e-2 * l1, "C w is not parallel to w: residual {resid}");
        assert!((rayleigh - l1).abs() < 1e-2 * l1, "Rayleigh quotient {rayleigh} vs lambda_1 {l1}");
    }

    /// (d) BCM SELECTIVITY, against its closed-form fixed point. Four mutually orthogonal patterns
    /// are presented in random order. The rule must end up responding to exactly one of them, and
    /// the winner's response must land on `n_patterns * y_0` — the value where the sliding threshold
    /// and the output meet, derived on [`Bcm::selective_fixed_point`].
    ///
    /// The other three must be DEPRESSED, not merely smaller: selectivity is a statement about the
    /// losers as much as the winner.
    #[test]
    fn bcm_becomes_selective_for_one_pattern_and_depresses_the_rest() {
        let n_patterns = 4usize;
        let drive = 50.0; // hertz, the active channel of each pattern
        let y_0 = 10.0; // hertz
        let bounds = Bounds::new(0.0, 5.0).unwrap();
        // Asymmetric start: which pattern wins is decided here, not by the rule, and the test says
        // so by predicting the winner from the initial weights.
        let w0 = vec![0.010, 0.014, 0.011, 0.012];
        let favourite = 1usize;
        let mut b = Bcm::new(w0, 2.0e-6, 2.0, y_0, 0.01, bounds).unwrap();

        let mut rng = Rng::new(0x0BC0_5EED);
        let dt = 0.01;
        for _ in 0..200_000 {
            let k = rng.below(n_patterns as u32) as usize;
            let mut x = vec![0.0; n_patterns];
            x[k] = drive;
            b.update(&x, dt).unwrap();
        }

        let responses: Vec<f64> = (0..n_patterns).map(|k| b.w[k] * drive).collect();
        let want = b.selective_fixed_point(n_patterns).unwrap();
        assert!((want - 40.0).abs() < 1e-12, "fixed point {want}");

        let winner = responses
            .iter()
            .enumerate()
            .max_by(|a, c| a.1.partial_cmp(c.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        assert_eq!(winner, favourite, "responses {responses:?}");
        assert!(
            (responses[winner] - want).abs() / want < 0.05,
            "winner response {} vs closed-form fixed point {want} (all {responses:?})",
            responses[winner]
        );
        for k in 0..n_patterns {
            if k != winner {
                assert!(
                    responses[k] < 0.02 * want,
                    "pattern {k} was not depressed: {responses:?}"
                );
            }
        }
        // The threshold must sit AT the winner's response on average: that equality is what stops
        // the weight moving, and it is the mechanism rather than a coincidence.
        //
        // AVERAGED, not sampled. `theta` is a low-pass of a signal that is zero on three
        // presentations out of four, so the instantaneous value swings several percent either side
        // of its mean and a single read lands wherever the last pattern left it -- 43.7 against a
        // mean of 40.8 on the first draft of this test, which is the estimator moving, not the
        // model.
        let mut theta_bar = 0.0;
        let tail = 40_000u32;
        for _ in 0..tail {
            let k = rng.below(n_patterns as u32) as usize;
            let mut x = vec![0.0; n_patterns];
            x[k] = drive;
            b.update(&x, dt).unwrap();
            theta_bar += b.theta;
        }
        theta_bar /= f64::from(tail);
        assert!(
            (theta_bar - responses[winner]).abs() / want < 0.05,
            "mean theta {theta_bar} vs winner response {}",
            responses[winner]
        );
    }

    /// The sliding threshold is a low-pass of `y^2 / y_0`, and under a constant output it must
    /// relax to exactly that, exponentially. Checked against the closed form at every step.
    #[test]
    fn the_bcm_threshold_relaxes_exponentially_to_the_mean_square_output() {
        let bounds = Bounds::new(0.0, 5.0).unwrap();
        // eta = 0 freezes the weights, so `y` is constant and the threshold's own dynamics are the
        // only thing under test. A non-zero eta would move `y` and the comparison would be against
        // a moving target.
        let mut b = Bcm::new(vec![0.5, 0.0], 0.0, 1.5, 10.0, 0.0, bounds).unwrap();
        let x = [40.0, 0.0];
        let y = 20.0;
        let target = y * y / b.y_0;
        let dt = 0.01;
        for k in 1..=600u32 {
            let got_y = b.update(&x, dt).unwrap();
            assert!((got_y - y).abs() < 1e-12);
            let t = f64::from(k) * dt;
            let want = target * (1.0 - (-t / b.tau_theta).exp());
            assert!((b.theta - want).abs() < 1e-9, "step {k}: theta {} vs {want}", b.theta);
        }
        assert!((b.theta - target).abs() / target < 0.02);
    }

    /// (g) ZERO REWARD, ZERO WEIGHT CHANGE — exactly. The product `c * d` is zero because `d` is
    /// zero, so the weight is the same `f64` it started as, not a very close one. Meanwhile the
    /// eligibility trace is non-zero and decaying, which is the whole point of the mechanism: the
    /// synapse remembers, it has not committed.
    #[test]
    fn reward_modulated_stdp_with_no_reward_leaves_the_weight_bit_identical() {
        let mut r =
            RewardStdp::new(wide_pair(), 1.0, 0.2, Bounds::new(0.0, 1.0).unwrap()).unwrap();
        let w0 = 0.37;
        let mut w = w0;
        r.on_pre(w).unwrap();
        r.advance(w, 10e-3).unwrap();
        r.on_post(w).unwrap();
        let c_after_pair = r.c;
        assert!(c_after_pair > 0.0, "the pair left no eligibility");

        let mut seen = Vec::new();
        for _ in 0..2_000 {
            w = r.advance(w, 1e-3).unwrap();
            assert_eq!(w, w0, "an unrewarded synapse moved");
            seen.push(r.c);
        }
        // The trace evolved, and downward.
        assert!(r.c < c_after_pair, "eligibility did not decay: {} vs {c_after_pair}", r.c);
        assert!(r.c > 0.0, "eligibility vanished entirely");
        for k in 1..seen.len() {
            assert!(seen[k] < seen[k - 1], "eligibility not monotone at {k}");
        }
        // 2 s at tau_c = 1 s: exp(-2) of what it was, in closed form.
        let want = c_after_pair * (-2.0f64).exp();
        assert!((r.c - want).abs() / want < 1e-9, "eligibility {} vs closed form {want}", r.c);
    }

    /// THE DISTAL REWARD INTEGRAL, in closed form. Tag the synapse, wait half a second with no
    /// modulator, then deliver an impulse. The total weight change from there to the end of time is
    /// `c * d_0 * tau_c * tau_d / (tau_c + tau_d)`, which is a different expression from the
    /// per-step integral the implementation composes — the agreement between the two is the check.
    #[test]
    fn the_distal_reward_integral_matches_its_closed_form() {
        for &(tau_c, tau_d) in &[(1.0, 0.2), (0.5, 0.5), (2.0, 0.05)] {
            let mut r =
                RewardStdp::new(wide_pair(), tau_c, tau_d, Bounds::wide()).unwrap();
            let w0 = 0.0;
            let mut w = w0;

            // One pre-before-post pair at +10 ms.
            r.on_pre(w).unwrap();
            w = r.advance(w, 10e-3).unwrap();
            r.on_post(w).unwrap();

            // Half a second of silence: no reward, so no change, and the tag decays.
            for _ in 0..500 {
                w = r.advance(w, 1e-3).unwrap();
            }
            assert_eq!(w, w0);

            let d0 = 0.8;
            r.reward(d0).unwrap();
            let want = r.pending_change();
            assert!((want - r.c * d0 * tau_c * tau_d / (tau_c + tau_d)).abs() < 1e-18);

            // Integrate far past both time constants. 30 effective constants leaves 1e-13 of the
            // total unspent, which is below the tolerance below.
            let horizon = 30.0 * r.tau_effective();
            let steps = 20_000u32;
            for _ in 0..steps {
                w = r.advance(w, horizon / f64::from(steps)).unwrap();
            }
            let got = w - w0;
            assert!(
                (got - want).abs() / want.abs() < 1e-9,
                "tau_c {tau_c}, tau_d {tau_d}: integrated {got} vs closed form {want}"
            );
            assert!(got > 0.0, "a rewarded pre-before-post pair must potentiate");
        }
    }

    /// The step size must not change the answer, because the per-step integral is exact rather than
    /// a forward-Euler approximation. Forward Euler would over-credit by half a step every step.
    #[test]
    fn the_rewarded_weight_change_is_step_size_independent() {
        let run = |dt: f64, steps: u32| {
            let mut r = RewardStdp::new(wide_pair(), 1.0, 0.2, Bounds::wide()).unwrap();
            let mut w = 0.0;
            r.on_pre(w).unwrap();
            w = r.advance(w, 10e-3).unwrap();
            r.on_post(w).unwrap();
            r.reward(1.0).unwrap();
            for _ in 0..steps {
                w = r.advance(w, dt).unwrap();
            }
            w
        };
        let fine = run(1e-5, 500_000);
        let coarse = run(1e-2, 500);
        // Measured residual is 1.9e-12 relative across a 1000x change in step size, which is the
        // accumulated rounding of 500,000 compositions and not a discretisation error -- forward
        // Euler at these steps would differ in the fourth digit, not the twelfth.
        assert!((fine - coarse).abs() / fine < 1e-10, "fine {fine} vs coarse {coarse}");
    }

    /// A punishment reverses the sign of every tagged change, and by exactly the same magnitude.
    /// That symmetry is what makes the third factor a teaching signal rather than a gate.
    #[test]
    fn a_negative_reward_reverses_the_sign_of_the_weight_change() {
        let run = |amount: f64| {
            let mut r = RewardStdp::new(wide_pair(), 1.0, 0.2, Bounds::wide()).unwrap();
            let mut w = 0.0;
            r.on_pre(w).unwrap();
            w = r.advance(w, 10e-3).unwrap();
            r.on_post(w).unwrap();
            r.reward(amount).unwrap();
            for _ in 0..50_000 {
                w = r.advance(w, 1e-4).unwrap();
            }
            w
        };
        let up = run(1.0);
        let down = run(-1.0);
        assert!(up > 0.0 && down < 0.0, "up {up} down {down}");
        assert!((up + down).abs() / up < 1e-12, "not antisymmetric: {up} and {down}");
    }

    /// TURRIGIANO'S ACTUAL FINDING: the distribution SCALED, it did not shift. Every pairwise ratio
    /// survives an arbitrary sequence of scaling steps, to floating-point noise.
    #[test]
    fn synaptic_scaling_preserves_every_weight_ratio() {
        let w0 = vec![0.1, 0.37, 0.9, 2.5, 0.0004];
        let mut s =
            SynapticScaling::new(w0.clone(), 5.0, 100.0, 1.0, Bounds::new(0.0, 1e6).unwrap())
                .unwrap();
        let mut rng = Rng::new(0x5CA1_AB1E);
        for _ in 0..3_000 {
            // Rates all over the place, above and below target, so the factor swings both ways.
            s.observe(rng.below(4), 0.05).unwrap();
            let g = s.scale(0.05).unwrap();
            assert!(g.is_finite() && g > 0.0);
        }
        // The weights moved, or the invariant is vacuous.
        assert!((s.w[0] / w0[0] - 1.0).abs() > 0.1, "nothing scaled: {:?}", s.w);
        for i in 1..w0.len() {
            let want = w0[i] / w0[0];
            let got = s.w[i] / s.w[0];
            assert!((got - want).abs() / want < 1e-12, "ratio {i}: {got} vs {want}");
        }
    }

    /// The closed form for a silent cell: `w(t) = w_0 * exp(t / tau)`, so after exactly `tau`
    /// seconds the weights are `e` times what they were — regardless of how many steps it took.
    #[test]
    fn synaptic_scaling_of_a_silent_cell_follows_the_exponential_closed_form() {
        for &steps in &[10u32, 1_000, 100_000] {
            let mut s =
                SynapticScaling::new(vec![0.25, 0.5], 5.0, 20.0, 1.0, Bounds::new(0.0, 1e6).unwrap())
                    .unwrap();
            // Rate pinned at zero: a cell hearing nothing, which is the deprivation condition in the
            // 1998 experiment.
            s.rate = 0.0;
            let dt = s.tau / f64::from(steps);
            for _ in 0..steps {
                s.scale(dt).unwrap();
                s.rate = 0.0;
            }
            let want = 0.25 * std::f64::consts::E;
            // 1e-10, because 100,000 composed exponentials accumulate about 1e-11 of relative
            // rounding and the residual was measured before the tolerance was chosen: 9.7e-12 at
            // 100,000 steps, 1e-15 at ten. The claim is step-size independence, not exactness.
            assert!((s.w[0] - want).abs() / want < 1e-10, "{steps} steps: {} vs {want}", s.w[0]);
        }
        // And a cell at exactly its target does not move at all: the factor is exp(0) == 1.
        let mut s =
            SynapticScaling::new(vec![0.25], 5.0, 20.0, 1.0, Bounds::new(0.0, 1e6).unwrap())
                .unwrap();
        s.rate = 5.0;
        assert_eq!(s.scale(1.0).unwrap(), 1.0);
        assert_eq!(s.w[0], 0.25);
    }

    /// The rate estimator against its own closed form, and the bound the whole homeostat needs:
    /// weights never leave their interval even when the estimate is absurd.
    #[test]
    fn the_rate_estimator_relaxes_exponentially_and_the_bounds_hold() {
        let bounds = Bounds::new(0.05, 4.0).unwrap();
        let mut s = SynapticScaling::new(vec![0.2, 1.0], 5.0, 10.0, 0.5, bounds).unwrap();
        let dt = 0.01;
        let r = 20.0; // hertz, held constant
        let spikes_per_step = (r * dt) as u32; // 0 with dt = 10 ms, so drive the estimate directly
        assert_eq!(spikes_per_step, 0, "this test drives the estimator with a rate, not a count");
        for k in 1..=2_000u32 {
            // One spike every fifth step is exactly 20 Hz on average, but the estimator's closed
            // form is for a CONSTANT instantaneous rate, so feed it that: 1 spike per 50 ms step.
            s.observe(1, 0.05).unwrap();
            let t = f64::from(k) * 0.05;
            let want = r * (1.0 - (-t / s.rate_tau).exp());
            assert!((s.rate - want).abs() < 1e-9, "step {k}: rate {} vs {want}", s.rate);
        }
        assert!((s.rate - r).abs() / r < 1e-6);

        // Now scale hard in BOTH directions and check that BOTH clamps actually bind. The first
        // draft of this block used dt = 0.1 s for the upward excursion, whose factor is 1.01 -- it
        // never reached the ceiling, so half the assertion was decorative while reading as coverage.
        let (mut hit_floor, mut hit_ceiling) = (false, false);
        for _ in 0..200 {
            // exp(200 * (5 - 1000) / (5 * 10)) underflows to zero: the floor is the only thing
            // between these weights and nothing.
            s.rate = 1000.0;
            s.scale(200.0).unwrap();
            for &wi in &s.w {
                assert!(bounds.contains(wi), "scaled below the floor: {wi}");
                hit_floor |= wi == bounds.w_min;
            }
            // exp(200 * 5 / 50) is 4.9e8, which would take 0.05 past the ceiling by eight orders.
            s.rate = 0.0;
            s.scale(200.0).unwrap();
            for &wi in &s.w {
                assert!(bounds.contains(wi), "scaled above the ceiling: {wi}");
                hit_ceiling |= wi == bounds.w_max;
            }
        }
        assert!(hit_floor && hit_ceiling, "floor {hit_floor}, ceiling {hit_ceiling}");

        // A factor that is not finite is refused by name rather than written into a weight.
        s.rate = 0.0;
        assert!(matches!(
            s.scale(1e6),
            Err(PlasticityError::Diverged { what: "scaling factor", .. })
        ));
    }

    /// Nearest-neighbour pairing saturates by construction; all-to-all accumulates. That difference
    /// IS the model choice, and a library that made it silently would make every high-frequency
    /// result unreproducible.
    #[test]
    fn nearest_neighbour_pairing_saturates_where_all_to_all_accumulates() {
        let mk = |p| {
            let mut t = Trace::new(20e-3).unwrap();
            for _ in 0..50 {
                t.fire(p);
                t.advance(1e-3).unwrap();
            }
            t.x
        };
        let all = mk(Pairing::AllToAll);
        let near = mk(Pairing::NearestNeighbour);
        assert!(near < 1.0 && near > 0.9, "nearest-neighbour trace {near}");
        assert!(all > 15.0, "all-to-all trace {all} did not accumulate");
        // The saturating value is the geometric sum in closed form, evaluated one step after the
        // last spike: exp(-dt/tau) for nearest neighbour.
        assert!((near - (-1e-3f64 / 20e-3).exp()).abs() < 1e-15);
        let q = (-1e-3f64 / 20e-3).exp();
        let want = q * (1.0 - q.powi(50)) / (1.0 - q);
        assert!((all - want).abs() / want < 1e-12, "all-to-all {all} vs geometric sum {want}");
    }

    /// Every boundary refuses by name rather than propagating a non-finite number into a weight.
    #[test]
    fn non_finite_and_out_of_range_inputs_are_refused_by_name() {
        assert!(matches!(
            Trace::new(0.0),
            Err(PlasticityError::NotPositive { what: "trace time constant tau", .. })
        ));
        assert!(matches!(
            Trace::new(f64::NAN),
            Err(PlasticityError::NonFinite { what: "trace time constant tau", .. })
        ));
        assert!(matches!(
            Bounds::new(1.0, 1.0),
            Err(PlasticityError::BoundsInverted { .. })
        ));
        assert!(matches!(
            Bounds::new(0.0, f64::INFINITY),
            Err(PlasticityError::NonFinite { what: "weight ceiling", .. })
        ));
        assert!(matches!(
            PairStdp::new(-1.0, 0.1, 1e-2, 1e-2, WeightRule::Additive, Bounds::wide()),
            Err(PlasticityError::Negative { what: "A_plus", .. })
        ));
        assert!(matches!(
            PairStdp::new(
                0.1,
                0.1,
                1e-2,
                1e-2,
                WeightRule::SoftBound { mu: -0.5 },
                Bounds::wide()
            ),
            Err(PlasticityError::Negative { what: "soft-bound exponent mu", .. })
        ));

        let mut s = wide_pair();
        assert!(matches!(
            s.on_pre(f64::NAN),
            Err(PlasticityError::NonFinite { what: "weight", .. })
        ));
        assert!(matches!(s.advance(-1e-3), Err(PlasticityError::Negative { what: "dt", .. })));
        assert!(matches!(
            s.window_integral(-1.0),
            Err(PlasticityError::Negative { what: "half_width", .. })
        ));

        let o = Oja::new(vec![0.0, 0.0], 0.1);
        assert!(matches!(o, Err(PlasticityError::Empty { .. })));
        let mut o = Oja::new(vec![0.3, 0.4], 0.1).unwrap();
        assert!(matches!(
            o.update(&[1.0]),
            Err(PlasticityError::LengthMismatch { got: 1, want: 2, .. })
        ));
        assert!(matches!(
            o.update(&[1.0, f64::NAN]),
            Err(PlasticityError::NonFinite { what: "input", .. })
        ));
        assert!(Hebbian::new(vec![], 0.1, 0.0).is_err());
        assert!(Bcm::new(vec![0.1], 1e-6, 0.0, 10.0, 0.0, Bounds::wide()).is_err());
        assert!(SynapticScaling::new(vec![0.1], 0.0, 1.0, 1.0, Bounds::wide()).is_err());
        let mut sc =
            SynapticScaling::new(vec![0.1], 5.0, 1.0, 1.0, Bounds::wide()).unwrap();
        assert!(matches!(sc.observe(3, 0.0), Err(PlasticityError::NotPositive { what: "dt", .. })));

        // And a NaN weight is not "inside" any interval, which is what stops the clamp laundering
        // it into a plausible number.
        assert!(!Bounds::wide().contains(f64::NAN));
    }


    /// THE "JUST BEFORE THIS SPIKE" RULE, in closed form. A spike must not see its own contribution
    /// to its own side's slow trace: the triplet term at a presynaptic spike is `A3_minus * r2` with
    /// `r2` read *before* this spike bumps it.
    ///
    /// Found by mutation. Moving `note_pre` ahead of `pre_increment` survives the pair-reduction
    /// test, because that test sets `A3_minus = 0` and multiplies the slow trace away — so the
    /// reduction check, which is exact and which does its own job perfectly, is structurally blind
    /// to this ordering. It needs a fixture where the triplet amplitude is not zero.
    ///
    /// Protocol: one post spike, then two pre spikes. The FIRST pre spike has no earlier pre spike
    /// to its name, so `r2` is zero and the change must be the pair term alone. The SECOND sees
    /// `r2 = exp(-gap / tau_x)` from the first. Both are closed forms with no free parameters.
    #[test]
    fn a_spike_does_not_see_its_own_contribution_to_its_own_slow_trace() {
        let b = Bounds::wide();
        let (a2m, a3m) = (7.1e-3, 4.3e-3);
        let (tau_minus, tau_x) = (33.7e-3, 101e-3);
        let mut t = TripletStdp::new(
            0.0, 0.0, a2m, a3m, 16.8e-3, tau_x, tau_minus, 125e-3, WeightRule::Additive, b,
        )
        .unwrap();

        let lag = 12e-3; // post -> first pre
        let gap = 25e-3; // first pre -> second pre

        let mut w = 0.0;
        t.on_post(w).unwrap();
        t.advance(lag).unwrap();
        let w1 = t.on_pre(w).unwrap();

        // First pre spike: no earlier pre spike, so the triplet term contributes NOTHING and the
        // change is the pair term alone. If the spike saw its own bump, this would be
        // -o1 * (a2m + a3m) instead, which is 61% larger.
        let o1 = (-lag / tau_minus).exp();
        let want1 = -o1 * a2m;
        assert_eq!(w1 - w, want1, "first pre: {} vs closed form {want1}", w1 - w);
        let self_seeing = -o1 * (a2m + a3m);
        assert!((want1 - self_seeing).abs() / want1.abs() > 0.5, "the fixture cannot tell them apart");

        w = w1;
        t.advance(gap).unwrap();
        let w2 = t.on_pre(w).unwrap();

        // Second pre spike: r2 carries exactly one decayed spike, o1 has decayed by lag + gap.
        let o1b = (-(lag + gap) / tau_minus).exp();
        let r2 = (-gap / tau_x).exp();
        let want2 = -o1b * (a2m + a3m * r2);
        // A few ulp rather than an equality: `w` is no longer zero, so `w2 - w` is a floating
        // SUBTRACTION of two nearby numbers and need not return the increment bit for bit. The
        // first spike above, starting from exactly zero, is the one that can be an equality.
        assert!(
            (w2 - w - want2).abs() <= 8.0 * f64::EPSILON * want2.abs(),
            "second pre: {} vs closed form {want2}",
            w2 - w
        );
        // And the triplet term is actually doing work here, or the check above is the pair rule's.
        assert!(a3m * r2 > 0.3 * a2m, "the triplet term is negligible in this fixture");
    }

    /// The same rule on the postsynaptic side: `A3_plus * o2` is read before the post spike bumps
    /// `o2`. One pre spike, then two post spikes.
    #[test]
    fn a_post_spike_does_not_see_its_own_contribution_to_the_slow_post_trace() {
        let b = Bounds::wide();
        let (a2p, a3p) = (5.3e-3, 8.0e-3);
        let (tau_plus, tau_y) = (16.8e-3, 125e-3);
        let mut t = TripletStdp::new(
            a2p, a3p, 0.0, 0.0, tau_plus, 101e-3, 33.7e-3, tau_y, WeightRule::Additive, b,
        )
        .unwrap();

        let lag = 9e-3;
        let gap = 30e-3;
        let mut w = 0.0;
        t.on_pre(w).unwrap();
        t.advance(lag).unwrap();
        let w1 = t.on_post(w).unwrap();
        let r1 = (-lag / tau_plus).exp();
        assert_eq!(w1 - w, r1 * a2p, "first post saw a slow trace it should not have");

        w = w1;
        t.advance(gap).unwrap();
        let w2 = t.on_post(w).unwrap();
        let r1b = (-(lag + gap) / tau_plus).exp();
        let o2 = (-gap / tau_y).exp();
        let want = r1b * (a2p + a3p * o2);
        // Same reason as the pre-side test: a subtraction from a non-zero weight.
        assert!(
            (w2 - w - want).abs() <= 8.0 * f64::EPSILON * want.abs(),
            "second post: {} vs closed form {want}",
            w2 - w
        );
        assert!(a3p * o2 > 0.5 * a2p, "the triplet term is negligible in this fixture");
    }

    /// A rule with no weight-dependent factor and no clamping in range must be exactly additive
    /// over many pairs: `n` pairs at the same lag give `n` times the single-pair change. This is
    /// what "additive" means and it is what the multiplicative rules deliberately break.
    #[test]
    fn the_additive_rule_is_linear_in_the_number_of_isolated_pairs() {
        let mut s = wide_pair();
        let lag = 12e-3;
        let one = s.apply_pair(0.0, lag).unwrap();
        let mut w = 0.0;
        for _ in 0..100 {
            // A full second between pairs: exp(-1000 / 33.7) is 1e-13, so the pairs are isolated.
            s.clear();
            w = s.on_pre(w).unwrap();
            s.advance(lag).unwrap();
            w = s.on_post(w).unwrap();
            s.advance(1.0).unwrap();
        }
        assert!((w - 100.0 * one).abs() / w < 1e-12, "{w} vs 100 x {one}");
    }
}
