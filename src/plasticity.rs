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
//! 2. **Move the goalposts.** [`Bcm`] keeps a *sliding threshold* `theta`. Output above `theta`
//!    potentiates, below it depresses, and `theta` rises with the cell's recent activity — so a
//!    cell that is too active raises its own bar. This is what produces **selectivity**: presented
//!    with several patterns, a `BCM` cell ends up responding to exactly one. Bienenstock, Cooper &
//!    Munro, *Theory for the development of neuron selectivity: orientation specificity and
//!    binocular interaction in visual cortex*, J. Neurosci. 2:32–48 (1982), make the threshold a
//!    superlinear function of the time-averaged output `c_bar`: `theta_M = (c_bar / c_0)^p * c_bar`
//!    (their eq. 7), which at `p = 1` is `c_bar^2 / c_0`, the square of the mean. The threshold
//!    implemented here tracks the mean of the square instead, `theta -> E[y^2] / y_0`. That is the
//!    reformulation of Intrator & Cooper (Neural Networks 5:3–17, 1992), and the selective fixed
//!    point `n_patterns * y_0` this module checks belongs to it. This paragraph used to credit the
//!    mean-square threshold to the 1982 paper. That paper builds its threshold from "the average
//!    value of the postsynaptic firing rate" (p. 35), and the only mean square this review located
//!    in it is the norm of the noise in its Appendix C.
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
//! few milliseconds *before* a postsynaptic one potentiates and the same pair in the other order
//! depresses, each inside "a time window of 20 msec", with an effect that "decreases rapidly as
//! the absolute value of spike timing increases". That is **spike-timing-dependent plasticity**,
//! and [`PairStdp`] is it, in the exponential form fitted to those data afterwards:
//!
//! ```text
//! dw = +A_plus  * exp(-lag / tau_plus)     for lag > 0  (pre before post)
//! dw = -A_minus * exp( lag / tau_minus)    for lag < 0  (post before pre)
//! ```
//!
//! The two widths, `tau_plus = 16.8 ms` and `tau_minus = 33.7 ms`, come from a later fit.
//! Pfister & Gerstner (2006) take them "from Bi and Poo (2001)" (Table 4 caption), which is Bi &
//! Poo, *Synaptic Modification by Correlated Activity: Hebb's Postulate Revisited*, Annu. Rev.
//! Neurosci. 24:139–166 (2001), doi:10.1146/annurev.neuro.24.1.139. This paragraph used to say the
//! 1998 paper found the change "falling off exponentially in the lag". Its Fig. 7 is a scatter of
//! single experiments with no curve through it, and this review did not locate an exponential or
//! a time constant anywhere in its text.
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
//! A single pair against `A_plus * exp(-lag / tau_plus)` bit for bit, both signs, seven lags; the
//! window's analytic integral against midpoint quadrature of the window itself; the triplet rule
//! against the pair rule as an exact equality; every named triplet parameter set against the
//! printed Table 3 and Table 4 of Pfister & Gerstner (2006), field by field; the eligibility trace
//! against [`PairStdp::window`] itself, so the three-factor rule's headline claim is checked
//! against the window and not only against its own decay algebra; Oja's rule against the scalar map
//! its own dynamics reduce to and against the principal eigenvector of a correlation matrix built
//! to have a known one; `BCM`'s selective fixed point against `n_patterns * y_0`, which is where
//! the sliding threshold must settle, and its single-presentation step against the closed form that
//! fixes *when* the threshold moves; the distal-reward integral against
//! `c * d * tau_c * tau_d / (tau_c + tau_d)`; the soft bound's exponent against Gütig's normalised
//! distance at a span that is not one, so the normalisation is not the identity in the fixture; and
//! every bounded rule against its bounds under adversarial amplitudes, `BCM` included.

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
    ///
    /// Also what every timing rule here returns when the weight it just computed is not finite.
    /// That cannot happen from a weight inside the bounds with the parameters a constructor
    /// accepts, but every field on every rule in this module is `pub` — a caller who writes
    /// `s.a_minus = f64::NAN` after construction has bypassed the constructor's validation, and the
    /// answer to that is a named refusal rather than a `NaN` returned as if it were a weight.
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

/// A value a rule just *computed*, refused if it left the finite numbers.
///
/// Distinct from [`finite`], which guards an input: this guards an output, and reports
/// [`PlasticityError::Diverged`] rather than [`PlasticityError::NonFinite`] because the caller's
/// argument was fine and the rule's own arithmetic was not. The clamp cannot do this job — a `NaN`
/// satisfies neither of its comparisons and passes through unchanged, which is exactly how a `NaN`
/// weight would otherwise be laundered into a plausible-looking return value.
fn computed(what: &'static str, v: f64) -> Result<f64, PlasticityError> {
    if v.is_finite() { Ok(v) } else { Err(PlasticityError::Diverged { what, value: v }) }
}

/// A closed interval a weight is never allowed to leave.
///
/// The clamp is applied on **every** update by every bounded rule in this module, including the
/// ones whose weight dependence already vanishes at the edge. That is deliberate: a soft bound
/// makes the step small near the boundary, it does not make it zero, and one oversized amplitude
/// steps straight through. The guarantee a caller needs is "the weight is in range", and only an
/// unconditional clamp provides it.
///
/// # The invariant, and who establishes it
///
/// `w_min < w_max`, both finite. [`Bounds::new`] is the only thing that checks it; the fields are
/// `pub`, so `Bounds { w_min: 5.0, w_max: -5.0 }` is a legal expression and nothing here will stop
/// it. An inverted pair is a programming error rather than a supported input, and it is a
/// *detectable* one: [`Bounds::contains`] then returns `false` for every weight without exception
/// and [`Bounds::span`] is negative, so a caller that checks either finds out immediately. What it
/// is not is silently absorbed — nothing in this module reorders the endpoints for you, because
/// doing so would turn a typo into a different model that still runs.
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
    /// **decades** of headroom below `f64::MAX` — about 984 powers of two, since `f64::MAX / 1e12`
    /// is `1.8e296`.
    #[must_use]
    pub const fn wide() -> Self {
        Self { w_min: -1.0e12, w_max: 1.0e12 }
    }

    /// The weight, moved into range if it was outside.
    ///
    /// A `NaN` weight comes back as `NaN`: it satisfies neither comparison, so there is no range to
    /// move it into. **The clamp is therefore not the guard against a non-finite weight** — the
    /// rules that call it check their own result afterwards and return
    /// [`PlasticityError::Diverged`], because a `NaN` returned from here would otherwise arrive at
    /// the caller wearing the shape of a weight.
    #[must_use]
    pub fn clamp(self, w: f64) -> f64 {
        // Written as two comparisons rather than `f64::clamp`, which panics on a NaN bound. A NaN
        // weight passes both comparisons unchanged; see the doc above for what catches it.
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
/// nearest-neighbour schemes. The one here is their **symmetric** scheme (Fig. 7a), where a spike
/// **resets** its own side's trace to one rather than incrementing it: it "can be implemented by
/// pre- and postsynaptic traces that reset to 1, rather than incrementing by 1". It is not their
/// *reduced* symmetric interpretation (Fig. 7c), which pairs only immediate neighbours and needs
/// "doubly resetting" traces. This line used to call the scheme here "the symmetric reduction",
/// which reads as Fig. 7c; the mechanism it described and the code implement are Fig. 7a's. The
/// choice changes the high-frequency behaviour of every rule in this module and is therefore
/// explicit rather than implied.
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
    ///
    /// # Outside the bounds
    ///
    /// `w` is evaluated at [`Bounds::clamp`] of itself, so a weight outside the interval is priced
    /// as if it were at the nearest bound. Two reasons, and the first is the one that matters:
    /// without it `((w - w_min) / span).powf(mu)` is an **infinity** for a large `w` and a large
    /// `mu`, and an infinity multiplied by a trace that happens to be zero is a `NaN` weight
    /// returned as `Ok`. Second, extrapolating is wrong anyway — Gütig's factor is a function of
    /// the *normalised* distance to a bound, and at `w = -1` in `[0, 1]` the raw expression makes a
    /// soft bound **amplify** potentiation by 1.414 at `mu = 0.5` rather than damp it. No weight
    /// this module produces is ever outside its bounds, so this changes nothing for a weight that
    /// came from a rule in this module; it changes the answer for one the caller invented.
    #[must_use]
    pub fn potentiation_factor(self, w: f64, bounds: Bounds) -> f64 {
        match self {
            Self::Additive | Self::MultiplicativeDepression => 1.0,
            Self::SoftBound { mu } => {
                ((bounds.w_max - bounds.clamp(w)) / bounds.span()).max(0.0).powf(mu)
            }
        }
    }

    /// The multiplier on a depressing step for a synapse currently at `w`.
    ///
    /// `w` is clamped into `bounds` first, for the reasons on
    /// [`WeightRule::potentiation_factor`]. Note that
    /// [`WeightRule::MultiplicativeDepression`]'s factor is `w - w_min` in the weight's **own
    /// unit** and is therefore not normalised to one at the ceiling: van Rossum et al. write the
    /// depression term as a fraction of the weight itself, and their `w_min` is zero.
    #[must_use]
    pub fn depression_factor(self, w: f64, bounds: Bounds) -> f64 {
        match self {
            Self::Additive => 1.0,
            Self::MultiplicativeDepression => (bounds.clamp(w) - bounds.w_min).max(0.0),
            Self::SoftBound { mu } => {
                ((bounds.clamp(w) - bounds.w_min) / bounds.span()).max(0.0).powf(mu)
            }
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
/// The experiment is Bi & Poo, J. Neurosci. 18:10464–10472, 1998; the exponential window and its
/// two widths are the later fit that Pfister & Gerstner (2006) credit to Bi & Poo (2001), see
/// [`PairStdp::bi_poo_2001`]. One presynaptic spike and one postsynaptic spike separated by
/// `lag = t_post - t_pre` change the weight by
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
    /// Potentiation window width, seconds. Bi & Poo's (2001) fit is 16.8 ms.
    pub tau_plus: f64,
    /// Depression window width, seconds. Bi & Poo's (2001) fit is 33.7 ms — wider than
    /// potentiation, which is why the rule can be depression-dominated with comparable amplitudes.
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
    /// The time constants are the exponential fit that Pfister & Gerstner (J. Neurosci.
    /// 26:9673–9682, 2006) attribute to Bi & Poo, *Synaptic Modification by Correlated Activity:
    /// Hebb's Postulate Revisited*, Annu. Rev. Neurosci. 24:139–166 (2001),
    /// doi:10.1146/annurev.neuro.24.1.139, and reuse unchanged in every row of their tables:
    /// "The additional parameters τ+ = 16.8 ms and τ− = 33.7 ms are taken from Bi and Poo (2001)
    /// and kept fixed for all models and data sets" (Table 4 caption). The 2001 review itself was
    /// not read for this correction, so the attribution is Pfister & Gerstner's.
    ///
    /// **The amplitudes are not supplied here on purpose**: the 1998 experiment reports its changes
    /// as percentages of the `EPSC` amplitude with large scatter across synapses, so a single
    /// number presented as "Bi & Poo's `A_plus`" would be a figure this implementation could not
    /// defend. Choose them for the weight unit you are using, and check
    /// [`PairStdp::is_depression_dominated`] afterwards.
    ///
    /// # Called `bi_poo_1998` through version 0.22.0
    ///
    /// That name, and this doc, credited the two widths to the 1998 experiment (J. Neurosci.
    /// 18:10464–10472) as "the widely reproduced fit" from it. That paper's Fig. 7 plots each
    /// experiment's change in `EPSC` amplitude against spike timing with no curve through it, the
    /// text reports "a time window of 20 msec" on both sides, and this review did not locate 16.8,
    /// 33.7, an exponential or a time constant anywhere in it. The numbers are unchanged; the name
    /// and the citation moved to the source the numbers are credited to.
    ///
    /// # Errors
    ///
    /// As [`PairStdp::new`].
    pub fn bi_poo_2001(
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
    /// # A naming hazard, not a disagreement
    ///
    /// The paper writes its window as `F(Delta t)` with `Delta t = t_pre - t_post`, which is the
    /// **opposite** sign to this module's `lag = t_post - t_pre`, and it assigns `Delta t = 0` to
    /// depression where [`PairStdp::window`] returns `0.0` by convention. The model is the same
    /// model; the variable is not the same variable. Anyone transcribing a figure from that paper
    /// into this API has to flip the axis first.
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
    ///
    /// [`PlasticityError::Diverged`] if the weight this call computed is not finite, which needs a
    /// parameter written into a `pub` field after construction — the constructor rejects every
    /// amplitude and time constant that could do it. The spike is **not** registered when either
    /// refusal fires, so a refused call leaves the traces exactly as it found them.
    pub fn on_pre(&mut self, w: f64) -> Result<f64, PlasticityError> {
        let w = finite("weight", w)?;
        let out = computed("updated weight", self.bounds.clamp(w + self.pre_increment(w)))?;
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
        let out = computed("updated weight", self.bounds.clamp(w + self.post_increment(w)))?;
        self.note_post();
        Ok(out)
    }

    /// Play one isolated pre/post pair at lag `lag` seconds on a synapse at weight `w`, from a
    /// cleared state, and return the new weight.
    ///
    /// With [`WeightRule::Additive`] and a weight away from its bounds the change it produces is
    /// [`PairStdp::window`] to the last bit — the two paths compute the same product of the same
    /// amplitude and the same exponential.
    ///
    /// One pair is the per-pair reduction of the 1998 protocol, not the protocol. Bi & Poo applied
    /// "60 pulses at 1 Hz" (Fig. 7 caption), and Morrison, Diesmann & Gerstner (2008) note that in
    /// such experiments "a single pair has no effect". At 1 Hz almost nothing of one pairing's
    /// traces survives to the next — `exp(-1 s / 33.7 ms)` is about 1e-13 — so under
    /// [`WeightRule::Additive`] the 60-pairing protocol is, to that precision, sixty independent
    /// copies of this change. This paragraph used to call a single pair "the protocol the 1998
    /// experiment ran".
    ///
    /// **At exactly `lag = 0` this returns `w + A_plus`, not `w`.** Two spikes with no time between
    /// them are delivered pre-then-post here, so the pre trace is at one when the post spike reads
    /// it. [`PairStdp::window`] returns zero at that point instead. The disagreement is the
    /// window's discontinuity, it is one isolated point, and neither answer is a measurement.
    ///
    /// # Errors
    ///
    /// [`PlasticityError::NonFinite`] for a non-finite `w` or `lag`, plus whatever
    /// [`PairStdp::on_pre`] and [`PairStdp::on_post`] return.
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
/// `tau_plus = 16.8 ms` and `tau_minus = 33.7 ms` are inherited from Bi & Poo (2001, see
/// [`PairStdp::bi_poo_2001`]) and held fixed for every row of both fitted tables. Everything else
/// is fitted per data set **and per pairing scheme**, and the four named constructors below are
/// four different rows, transcribed from the printed Table 3 and Table 4:
///
/// | constructor | data set | scheme | `A2+` | `A3+` | `A2-` | `A3-` | `tau_x` | `tau_y` |
/// |---|---|---|---|---|---|---|---|---|
/// | [`TripletStdp::visual_cortex_minimal`] | visual cortex | all-to-all | 0 | 6.5e-3 | 7.1e-3 | 0 | (inert) | 114 ms |
/// | [`TripletStdp::visual_cortex_full`] | visual cortex | all-to-all | 5e-10 | 6.2e-3 | 7e-3 | 2.3e-4 | (101 ms) | 125 ms |
/// | [`TripletStdp::hippocampal_full`] | hippocampal culture | all-to-all | 6.1e-3 | 6.7e-3 | 1.6e-3 | 1.4e-3 | 946 ms | 27 ms |
/// | [`TripletStdp::hippocampal_full_nearest_spike`] | hippocampal culture | nearest-spike | 4.6e-3 | 9.1e-3 | 3e-3 | 7.5e-9 | 575 ms | 47 ms |
///
/// **A row is a package.** The amplitudes, the two slow time constants and the pairing scheme were
/// fitted together, so taking the time constants from one row and the amplitudes from another
/// produces a model that appears in no table and reproduces no measurement. A previous release of
/// this module did exactly that in `hippocampal_full` — see the note on that constructor — which is
/// why the table above is printed here and why a test compares every field of every constructor
/// against it.
///
/// Parentheses in the `tau_x` column are the paper's own notation, "to indicate that the error
/// function is insensitive to the exact value of τx in those cases". It prints the full
/// visual-cortex fit's 101 ms that way, and this table extends the notation to the minimal row,
/// whose cell the paper leaves blank: the fit is insensitive to `tau_x` wherever `A3-` is zero,
/// because zero multiplies the slow pre trace away. See [`TripletStdp::visual_cortex_minimal`] for
/// what this implementation puts there and for the test that shows the choice cannot change an
/// answer.
///
/// Source: Pfister & Gerstner, J. Neurosci. 26:9673–9682, 2006, Tables 3 and 4, read from the
/// published article rather than from a secondary account of it.
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
    /// Fast presynaptic trace `r1`, time constant `tau_plus`, 16.8 ms in every fitted row.
    pub r1: Trace,
    /// Slow presynaptic trace `r2`, time constant `tau_x`: 101 ms in the full visual-cortex fit,
    /// which prints it in parentheses because the fit is insensitive to it, 946 ms in the full
    /// hippocampal one, and unidentifiable wherever `A3_minus` is zero.
    pub r2: Trace,
    /// Fast postsynaptic trace `o1`, time constant `tau_minus`, 33.7 ms in every fitted row.
    pub o1: Trace,
    /// Slow postsynaptic trace `o2`, time constant `tau_y`: 114 ms in the *minimal* visual-cortex
    /// fit, 125 ms in the *full* one, 27 ms in the full hippocampal one. The two visual-cortex
    /// numbers are different fits of different models and are not interchangeable.
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
    /// pairing: `A2_plus = 0`, `A3_plus = 6.5e-3`, `A2_minus = 7.1e-3`, `A3_minus = 0`,
    /// `tau_y = 114 ms`. Table 3, "All-to-All / Min." row.
    ///
    /// Minimal because two of the four amplitudes are zero: depression is purely a pair effect and
    /// potentiation is purely a triplet effect. This is the parameter set that reproduces the
    /// frequency dependence the pair rule cannot, and the one the frequency test in this module
    /// uses.
    ///
    /// # `tau_y` is 114 ms here and 125 ms in the full model
    ///
    /// They are two fits of two different models to the same data and the paper prints them on two
    /// different rows. 125 ms belongs to [`TripletStdp::visual_cortex_full`]. It matters more here
    /// than anywhere else in this module, because with `A2_plus = 0` the slow post trace is the
    /// **only** thing the minimal model's potentiation depends on: moving 114 to 125 multiplies the
    /// per-pair change at 1 Hz by 2.16 and at 40 Hz by 1.19.
    ///
    /// # `tau_x` is unidentifiable here
    ///
    /// `A3_minus = 0` multiplies the slow pre trace `r2` out of the depression term, so no value of
    /// `tau_x` changes any output of this model and the paper leaves that cell of the table blank.
    /// A [`Trace`] still needs a positive time constant, so this constructor puts the full
    /// visual-cortex fit's 101 ms there and
    /// `the_minimal_triplet_models_slow_pre_trace_cannot_change_an_answer` proves the choice is
    /// inert by running the model twice with `tau_x` two orders apart and comparing bit for bit.
    /// The trace is still fired and decayed on every event, and
    /// [`crate::ledger`] charges for it, which is the honest cost of a shared implementation.
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
            114e-3,
            WeightRule::Additive,
            bounds,
        )
    }

    /// The **full** triplet model fitted to visual-cortex data, all-to-all pairing:
    /// `A2_plus = 5e-10`, `A3_plus = 6.2e-3`, `A2_minus = 7e-3`, `A3_minus = 2.3e-4`,
    /// `tau_x = 101 ms`, `tau_y = 125 ms`. Table 3, "All-to-All / Full" row.
    ///
    /// This is where 101 ms and 125 ms come from, but only 125 ms is identified by this fit.
    /// Table 3 prints the `tau_x` as "(101)", and the paper's parentheses mark "that the error
    /// function is insensitive to the exact value of τx in those cases" (caption on p. 9677). The
    /// visual-cortex row with an identified `tau_x` is the nearest-spike full fit, `tau_x = 714 ms`
    /// and `tau_y = 40 ms`, which has no constructor here. This paragraph used to call the
    /// all-to-all full row "the only visual-cortex row where either is identifiable"; 101 ms is the
    /// printed value and is kept, but nothing should lean on it as a measured time constant.
    ///
    /// `A2_plus = 5e-10` is the fit's way of saying "zero": the optimiser put the pair potentiation
    /// term at the floor of its search range, which is the observation the minimal model turns into
    /// a structural assumption.
    ///
    /// # Errors
    ///
    /// As [`TripletStdp::new`].
    pub fn visual_cortex_full(bounds: Bounds) -> Result<Self, PlasticityError> {
        Self::new(
            5e-10,
            6.2e-3,
            7e-3,
            2.3e-4,
            16.8e-3,
            101e-3,
            33.7e-3,
            125e-3,
            WeightRule::Additive,
            bounds,
        )
    }

    /// The **full** triplet model fitted to hippocampal culture data, all-to-all pairing:
    /// `A2_plus = 6.1e-3`, `A3_plus = 6.7e-3`, `A2_minus = 1.6e-3`, `A3_minus = 1.4e-3`, with the
    /// slow time constants `tau_x = 946 ms` and `tau_y = 27 ms`. Table 4, "All-to-All / Full" row.
    ///
    /// Note how different the slow constants are from the visual-cortex fit — 946 ms against
    /// 101 ms, 27 ms against 125 ms, and the two swapped in rank. The post side is the firmer half
    /// of that comparison: the visual-cortex 101 ms is printed in parentheses, a value the fit is
    /// insensitive to (see [`TripletStdp::visual_cortex_full`]). The triplet model is not one
    /// model with one parameter set; it is a form that two preparations fill in differently, and
    /// reporting a result with the wrong preparation's numbers is a category error the shared
    /// function name makes easy.
    ///
    /// # What this constructor used to return, and why it is worth saying
    ///
    /// Through version 0.4.0 it returned `A2_plus = 5.3e-3`, `A3_plus = 8.0e-3`,
    /// `A2_minus = 3.5e-3`, `A3_minus = 1.0e-3` under [`Pairing::NearestNeighbour`]. Those first
    /// three amplitudes are the hippocampal **all-to-all minimal** row, whose `A3_minus` is zero
    /// and whose `tau_y` is 40 ms; the `1.0e-3` appears in no row of either table; and the
    /// nearest-spike scheme belongs to a row with different amplitudes again. It was three rows and
    /// an invention in one object, and it had no test. If you have a result from that version,
    /// it was produced by a model that is in no paper.
    ///
    /// For the nearest-spike hippocampal fit, which is a genuine published row, use
    /// [`TripletStdp::hippocampal_full_nearest_spike`].
    ///
    /// # Errors
    ///
    /// As [`TripletStdp::new`].
    pub fn hippocampal_full(bounds: Bounds) -> Result<Self, PlasticityError> {
        Self::new(
            6.1e-3,
            6.7e-3,
            1.6e-3,
            1.4e-3,
            16.8e-3,
            946e-3,
            33.7e-3,
            27e-3,
            WeightRule::Additive,
            bounds,
        )
    }

    /// The full hippocampal fit under **nearest-spike** pairing: `A2_plus = 4.6e-3`,
    /// `A3_plus = 9.1e-3`, `A2_minus = 3e-3`, `A3_minus = 7.5e-9`, `tau_x = 575 ms`,
    /// `tau_y = 47 ms`. Table 4, "Nearest-Spike / Full" row, and it sets
    /// [`Pairing::NearestNeighbour`] to match.
    ///
    /// The same data as [`TripletStdp::hippocampal_full`], refitted under a different interaction
    /// scheme — and **every number moved**, which is the concrete reason the pairing scheme is a
    /// field of the model here rather than an implementation detail. The fitting error is the same
    /// to two figures (2.9 for both), so the data do not choose between them; the scheme is an
    /// assumption you make and then have to report.
    ///
    /// `A3_minus = 7.5e-9` is the fit's zero, exactly as `A2_plus = 5e-10` is in
    /// [`TripletStdp::visual_cortex_full`]. It is kept at the printed value rather than rounded to
    /// zero because rounding it would silently make `tau_x` unidentifiable, which is a different
    /// model — the paper's own parentheses around 575 ms say the error surface is already almost
    /// flat in it.
    ///
    /// # Errors
    ///
    /// As [`TripletStdp::new`].
    pub fn hippocampal_full_nearest_spike(bounds: Bounds) -> Result<Self, PlasticityError> {
        let mut t = Self::new(
            4.6e-3,
            9.1e-3,
            3e-3,
            7.5e-9,
            16.8e-3,
            575e-3,
            33.7e-3,
            47e-3,
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
    /// [`PlasticityError::NonFinite`] for a non-finite weight, [`PlasticityError::Diverged`] for a
    /// non-finite computed weight. As [`PairStdp::on_pre`], the spike is not registered when either
    /// refusal fires.
    pub fn on_pre(&mut self, w: f64) -> Result<f64, PlasticityError> {
        let w = finite("weight", w)?;
        let out = computed("updated weight", self.bounds.clamp(w + self.pre_increment(w)))?;
        self.note_pre();
        Ok(out)
    }

    /// A postsynaptic spike arrives. Returns the new weight, clamped.
    ///
    /// # Errors
    ///
    /// As [`TripletStdp::on_pre`].
    pub fn on_post(&mut self, w: f64) -> Result<f64, PlasticityError> {
        let w = finite("weight", w)?;
        let out = computed("updated weight", self.bounds.clamp(w + self.post_increment(w)))?;
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
    /// Linear weight decay per sample, dimensionless. Zero is pure Hebb.
    ///
    /// **A positive value does not bound the norm at anything.** The term is linear in `w`, so it
    /// subtracts `decay` from every eigenvalue of the input correlation matrix `C` and leaves the
    /// rule linear: the component along the principal direction is multiplied by
    /// `1 + eta * (lambda_1 - decay)` every sample, exactly, and the norm is therefore geometric in
    /// the sample count with no fixed point anywhere except zero. `decay < lambda_1` still diverges
    /// — more slowly. `decay > lambda_1` collapses to zero — not to a smaller weight vector, to no
    /// weight vector. `decay == lambda_1` is a knife edge no simulation lands on.
    ///
    /// Measured, with `C = u u^T` so `lambda_1 = 1`, `eta = 0.01`, 20,000 samples from
    /// `|w_0| = 0.1`: `decay = 0.05` reaches a norm of 1.3e81, `decay = 0.5` reaches 2.1e42, and
    /// `decay = 2.0` reaches 5e-89. Bounding the norm needs a term that is **not** linear in `w`,
    /// which is precisely what [`Oja`]'s `-y^2 w` and [`Bcm`]'s sliding threshold are.
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
/// feedback is what makes the rule stable *and* what makes it selective. The mean square is
/// Intrator & Cooper's; the 1982 threshold is a power of the time-averaged output,
/// `(c_bar / c_0)^p * c_bar` (its eq. 7), and the fixed point below is derived for the form here.
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
    ///
    /// # It is the averaged dynamics' fixed point, and nothing here checks the averaging
    ///
    /// The derivation replaces `theta` by its mean over presentations, which is only the same
    /// system when `tau_theta` is short compared with the time the weights take to move. Outside
    /// that regime the cell is still selective and still stable — it just settles somewhere else,
    /// and this function has no way to know. Measured on the fixture in
    /// `bcm_becomes_selective_for_one_pattern_and_depresses_the_rest`, which predicts 40 Hz: at
    /// `tau_theta = 2 s` the winner lands at 40.4 Hz, and at `tau_theta = 20 s` — still four orders
    /// of magnitude faster than the biological estimate — it lands at **51.5 Hz**, 29% high, with
    /// no error and no warning. `n_patterns * y_0` is what the averaged equations say, not a
    /// measurement of the simulation you are about to run.
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
    /// # Which `theta` the weight step uses
    ///
    /// **The one from before this presentation.** Both derivatives are evaluated at the state at
    /// the start of the interval — `phi = y * (y - theta_before)` moves the weights, and only then
    /// does `theta` relax toward this presentation's `y^2 / y_0`. That is the standard convention
    /// for stepping a coupled system forward and it is the one the published `BCM` equations are
    /// written in, where `dw/dt` and `dtheta/dt` are both functions of the same instantaneous
    /// state.
    ///
    /// Stated because it is invisible in the equations and load-bearing in the code: absorbing
    /// `y^2` first and then stepping the weights is a different model, and on the selectivity
    /// fixture in this module it moves the winner's settled response by 0.6 Hz — inside the 5% band
    /// that test allows, which is why
    /// `the_bcm_weight_step_uses_the_threshold_from_before_this_presentation` pins the single-step
    /// arithmetic against the closed form instead of relying on the long run to notice.
    ///
    /// # Errors
    ///
    /// As [`Bcm::output`], plus [`PlasticityError::Negative`] for a negative `dt`.
    pub fn update(&mut self, x: &[f64], dt: f64) -> Result<f64, PlasticityError> {
        let dt = non_negative("dt", dt)?;
        let y = self.output(x)?;
        // ORDER IS THE MODEL. `theta` here is the threshold as it stood before this presentation;
        // the relaxation below must not run first. See the doc above.
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
    /// **The tag written here is the `STDP` window, not a proxy for it.** After an isolated
    /// post-before-pre pair from a cleared state, `c` equals [`PairStdp::window`] at that lag
    /// exactly — the same amplitude times the same exponential, times the weight dependence — which
    /// is what makes "`STDP` does not write the weight, it writes an eligibility trace" a statement
    /// about this rule rather than about a decaying number. `the_eligibility_trace_is_the_stdp_window_itself`
    /// asserts that equality bit for bit on both signs.
    ///
    /// # Errors
    ///
    /// [`PlasticityError::NonFinite`] for a non-finite weight, [`PlasticityError::Diverged`] if the
    /// tag this call computed is not finite. The spike is not registered when either fires.
    pub fn on_pre(&mut self, w: f64) -> Result<(), PlasticityError> {
        let w = finite("weight", w)?;
        self.c = computed("eligibility trace", self.c + self.stdp.pre_increment(w))?;
        self.stdp.note_pre();
        Ok(())
    }

    /// A postsynaptic spike: tag the synapse for potentiation. The weight does not move.
    ///
    /// As [`RewardStdp::on_pre`], the tag is [`PairStdp::window`] itself at the pair's lag.
    ///
    /// # Errors
    ///
    /// As [`RewardStdp::on_pre`].
    pub fn on_post(&mut self, w: f64) -> Result<(), PlasticityError> {
        let w = finite("weight", w)?;
        self.c = computed("eligibility trace", self.c + self.stdp.post_increment(w))?;
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
    /// The last of those is easy to forget and changes the model: without it the pre and post
    /// traces never decay between spikes, every pair in a train reads a trace of one, and the tag
    /// stops depending on timing at all. `the_eligibility_trace_decays_the_stdp_window_between_spikes`
    /// is the test that notices.
    ///
    /// Returns the new weight, clamped into [`RewardStdp::bounds`].
    ///
    /// # Errors
    ///
    /// [`PlasticityError::NonFinite`] for a non-finite weight, [`PlasticityError::Negative`] or
    /// [`PlasticityError::NonFinite`] for a `dt` that is not finite and non-negative, and
    /// [`PlasticityError::Diverged`] for a computed weight that is not finite.
    pub fn advance(&mut self, w: f64, dt: f64) -> Result<f64, PlasticityError> {
        let w = finite("weight", w)?;
        let dt = non_negative("dt", dt)?;
        // The exact integral of c(0) d(0) exp(-s / tau_eff) over [0, dt]. Forward Euler would be
        // `c * d * dt`, which over-counts by half a step every step and therefore over-credits
        // exactly the synapses that were tagged hardest.
        let tau_eff = self.tau_effective();
        let dw = self.c * self.d * tau_eff * (1.0 - (-dt / tau_eff).exp());
        let out = computed("updated weight", self.bounds.clamp(w + dw))?;
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
    /// order. Both signs, seven lags each.
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
        // And the two sides are genuinely different widths, as in Bi & Poo's 2001 exponential fit
        // (33.7 ms against 16.8 ms). The 1998 paper reports a 20-ms window on each side, and the
        // asymmetry it describes is the sign flip with spike order. This comment used to call the
        // width difference "the asymmetry Bi & Poo saw".
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
    ///
    /// `g_max` is swept rather than left at 1.0. At `g_max = 1.0` the paper's `A_plus = 0.005 *
    /// g_max` is numerically indistinguishable from a bare `0.005` and `Bounds::new(0.0, g_max)`
    /// from `Bounds::normalised()`, so the one fixture that reads like coverage of the scaling is
    /// the one fixture that has none. The paper's sentence is "the value A+ = 0.005 thus
    /// corresponds to a change of 0.5% of the maximum synaptic strength per spike pair" (p. 920) —
    /// a FRACTION of `g_max`, which is only visible when `g_max` is not one.
    #[test]
    fn the_song_abbott_parameter_set_is_depression_dominated() {
        for &g_max in &[0.25f64, 1.0, 2.5, 400.0] {
            let s = PairStdp::song_abbott_2000(g_max).unwrap();
            assert_eq!(s.a_plus, 0.005 * g_max, "A_plus at g_max {g_max}");
            assert_eq!(s.bounds, Bounds::new(0.0, g_max).unwrap(), "bounds at g_max {g_max}");
            assert!(s.is_depression_dominated(), "g_max {g_max}, area {}", s.total_window_area());
            // 0.5% of the ceiling per pair, which is the sentence the paper writes.
            assert!(
                (s.a_plus / g_max - 0.005).abs() < 1e-15,
                "A_plus {} is not 0.5% of g_max {g_max}",
                s.a_plus
            );
        }
        // A non-positive ceiling has no fraction to take.
        assert!(matches!(
            PairStdp::song_abbott_2000(0.0),
            Err(PlasticityError::NotPositive { what: "g_max", .. })
        ));

        let s = PairStdp::song_abbott_2000(1.0).unwrap();
        assert!(s.is_depression_dominated(), "area {}", s.total_window_area());
        assert!((s.a_minus / s.a_plus - 1.05).abs() < 1e-12);
        assert_eq!(s.tau_plus, s.tau_minus);
        assert_eq!(s.tau_plus, 20e-3);
        assert_eq!(s.rule, WeightRule::Additive);
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
    ///
    /// The interval's span is deliberately **not** one. The first draft used `[-0.25, 0.75]`, whose
    /// span is exactly 1.0, and a span of one makes [`WeightRule::SoftBound`]'s division by
    /// `bounds.span()` the identity — so the whole soft-bound arm of this sweep was running a rule
    /// with its normalisation deleted and reporting six rules' worth of coverage.
    #[test]
    fn weight_bounds_are_never_violated_under_adversarial_input() {
        let b = Bounds::new(-0.6, 1.9).unwrap();
        assert!((b.span() - 2.5).abs() < 1e-15, "the fixture must not have a unit span");
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
        // Ratio is the weight ratio, exactly: the factor is linear in `w - w_min`. With w_min = 0
        // that is 0.9 / 0.5.
        assert!((near_top / mid - 1.8).abs() < 1e-12);

        // AND AGAIN WITH A FLOOR THAT IS NOT ZERO, because `w - w_min` and `w` are the same
        // expression when `w_min` is zero, and every van Rossum fixture in the literature has
        // `w_min = 0`. At `w_min = 0.25` the ratio is (0.9 - 0.25) / (0.5 - 0.25) = 2.6, which is
        // a different number from 1.8 only if the subtraction is really there.
        let raised = Bounds::new(0.25, 1.0).unwrap();
        let mut r = PairStdp::new(
            0.1,
            0.1,
            16.8e-3,
            33.7e-3,
            WeightRule::MultiplicativeDepression,
            raised,
        )
        .unwrap();
        let hi = step_at(&mut r, 0.9);
        let lo = step_at(&mut r, 0.5);
        assert!((hi / lo - 2.6).abs() < 1e-12, "raised floor: {hi} / {lo}");
        // A weight sitting exactly on the raised floor cannot be depressed at all.
        assert_eq!(step_at(&mut r, 0.25), 0.0, "depression at the floor must vanish");
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

        // Measured here, per pair, over 60 pairs, with the minimal model's PUBLISHED
        // `tau_y = 114 ms`:
        //
        //     1 Hz    pair  +3.0327e-3    triplet  +5.4648e-7
        //    10 Hz    pair  +2.9951e-3    triplet  +1.9774e-3
        //    40 Hz    pair  +8.2742e-4    triplet  +8.8685e-3
        //    50 Hz    pair  -1.5359e-4    triplet  +1.2712e-2
        //
        // Through version 0.4.0 this constructor carried the FULL model's `tau_y = 125 ms`, which
        // gives +1.1827e-6 and +1.0573e-2 at 1 Hz and 40 Hz -- a factor of 2.16 at the bottom of
        // the range. The qualitative assertions below pass either way, which is why they are now
        // accompanied by a closed-form test of the constructor's own parameters
        // (`the_minimal_visual_cortex_triplet_model_reproduces_its_published_window`).
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

        // The numbers in the comment above, pinned. A regression table nobody asserts is a
        // regression table that drifts: `tau_y` moved under exactly this test once already and it
        // did not notice, because "monotone and large" is true of both models.
        let want_pair = [3.0327e-3, 2.9951e-3, 8.2742e-4, -1.5359e-4];
        let want_trip = [5.4648e-7, 1.9774e-3, 8.8685e-3, 1.2712e-2];
        for (k, &i) in [0usize, 2, 4, 5].iter().enumerate() {
            assert!(
                (p[i] - want_pair[k]).abs() <= 1e-4 * want_pair[k].abs(),
                "pair at {} Hz: {} vs recorded {}",
                freqs[i],
                p[i],
                want_pair[k]
            );
            assert!(
                (t[i] - want_trip[k]).abs() <= 1e-4 * want_trip[k],
                "triplet at {} Hz: {} vs recorded {}",
                freqs[i],
                t[i],
                want_trip[k]
            );
        }
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
        let start: Vec<f64> = w0.iter().map(|w| w * drive).collect();
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
        // DEPRESSED, not merely small. The bar used to be `0.02 * want`, which is 0.8 Hz -- and the
        // initial responses are [0.5, 0.70, 0.55, 0.60] Hz, so all three losers START below it and
        // a rule that never touched them would have passed. The two bars below cannot be met by
        // standing still: each loser must have fallen by at least six orders of magnitude from
        // where it began, and must end essentially at zero in absolute terms.
        for k in 0..n_patterns {
            if k != winner {
                assert!(
                    responses[k] < 1e-6 * start[k],
                    "pattern {k} was not depressed: {} Hz from {} Hz (all {responses:?})",
                    responses[k],
                    start[k]
                );
                assert!(
                    responses[k] < 1e-6,
                    "pattern {k} did not collapse: {responses:?}"
                );
            }
        }
        // And the winner GREW, so "selective" is not "everything died but one".
        assert!(
            responses[winner] > 10.0 * start[winner],
            "the winner did not grow: {} Hz from {} Hz",
            responses[winner],
            start[winner]
        );
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
            // The step size varies too, so neither `observe` nor `scale` is ever fed the one `dt`
            // that would let a hard-coded constant impersonate a division.
            let dt = 0.01 + rng.next_f64() * 0.19;
            s.observe(rng.below(4), dt).unwrap();
            let g = s.scale(dt).unwrap();
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
        let r = 20.0; // hertz, held constant
        // THREE spikes per 150 ms step, not one per 50 ms. Both are 20 Hz, and that is the point:
        // every `observe` in the first draft of this suite used `dt = 0.05`, where `spikes / dt` is
        // indistinguishable from `spikes * 20.0` and the estimator's use of the elapsed time is
        // untested. With (3, 0.15) a hard-coded 20 would read 60 Hz and this loop would fail on its
        // first step.
        let (spikes, step) = (3u32, 0.15f64);
        assert!(
            (f64::from(spikes) / step - r).abs() < 1e-12,
            "the fixture must deliver exactly {r} Hz"
        );
        for k in 1..=2_000u32 {
            // The estimator's closed form is for a CONSTANT instantaneous rate, so feed it that
            // rather than a bursty train that averages to the same thing.
            s.observe(spikes, step).unwrap();
            let t = f64::from(k) * step;
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

    /// EVERY NAMED TRIPLET PARAMETER SET AGAINST THE PRINTED TABLE — Pfister & Gerstner (2006),
    /// Tables 3 and 4, read from the published article rather than from a secondary account.
    ///
    /// This exists because the one constructor whose numbers were wrong was the one constructor
    /// with no test at all. `hippocampal_full` shipped the hippocampal **all-to-all minimal** row's
    /// first three amplitudes, an `A3_minus` that appears in no row of either table, the
    /// **all-to-all full** row's slow time constants, and the **nearest-spike** pairing scheme —
    /// three rows and an invention in one object — and replacing all four of its amplitudes with
    /// `1.0, 2.0, 3.0, 4.0` left the module's other tests green.
    ///
    /// A row is a package: amplitudes, slow time constants and pairing scheme were fitted
    /// together. So the assertion is every field of every constructor, not a spot check.
    #[test]
    fn the_named_triplet_parameter_sets_match_the_printed_table() {
        let b = Bounds::wide();
        // One printed row: the model the constructor returns, then the six fitted numbers and the
        // pairing scheme it must agree with, then the table and row it came from.
        type Row = (TripletStdp, f64, f64, f64, f64, f64, f64, Pairing, &'static str);
        let rows: [Row; 4] = [
            (
                TripletStdp::visual_cortex_minimal(b).unwrap(),
                0.0,
                6.5e-3,
                7.1e-3,
                0.0,
                101e-3,
                114e-3,
                Pairing::AllToAll,
                "Table 3, All-to-All / Min.",
            ),
            (
                TripletStdp::visual_cortex_full(b).unwrap(),
                5e-10,
                6.2e-3,
                7e-3,
                2.3e-4,
                101e-3,
                125e-3,
                Pairing::AllToAll,
                "Table 3, All-to-All / Full",
            ),
            (
                TripletStdp::hippocampal_full(b).unwrap(),
                6.1e-3,
                6.7e-3,
                1.6e-3,
                1.4e-3,
                946e-3,
                27e-3,
                Pairing::AllToAll,
                "Table 4, All-to-All / Full",
            ),
            (
                TripletStdp::hippocampal_full_nearest_spike(b).unwrap(),
                4.6e-3,
                9.1e-3,
                3e-3,
                7.5e-9,
                575e-3,
                47e-3,
                Pairing::NearestNeighbour,
                "Table 4, Nearest-Spike / Full",
            ),
        ];
        for (t, a2p, a3p, a2m, a3m, tx, ty, pairing, name) in rows {
            assert_eq!(t.a2_plus, a2p, "{name}: A2+");
            assert_eq!(t.a3_plus, a3p, "{name}: A3+");
            assert_eq!(t.a2_minus, a2m, "{name}: A2-");
            assert_eq!(t.a3_minus, a3m, "{name}: A3-");
            assert_eq!(t.r2.tau, tx, "{name}: tau_x");
            assert_eq!(t.o2.tau, ty, "{name}: tau_y");
            assert_eq!(t.pairing, pairing, "{name}: pairing scheme");
            // tau_plus and tau_minus are Bi & Poo's (2001) and are held FIXED for every row of both
            // tables — the paper says so in the caption, and that is why they are not swept.
            assert_eq!(t.r1.tau, 16.8e-3, "{name}: tau_plus");
            assert_eq!(t.o1.tau, 33.7e-3, "{name}: tau_minus");
            // The rest of the object is the module's own default, not the paper's.
            assert_eq!(t.rule, WeightRule::Additive, "{name}: weight rule");
            assert_eq!(t.bounds, b, "{name}: bounds");
            assert_eq!((t.r1.x, t.r2.x, t.o1.x, t.o2.x), (0.0, 0.0, 0.0, 0.0), "{name}: at rest");

            // And each one actually runs. Pre, post at +10 ms, post again at +40 ms: every row
            // potentiates by a finite, non-zero amount. Two post spikes rather than one because
            // the minimal row's `A2_plus` is zero, so a single pair leaves it at exactly 0.0 —
            // which is that row's whole content and not a defect. A constructor returning a model
            // with every amplitude zero would satisfy all the field assertions above and fail here.
            let mut s = t;
            s.on_pre(0.0).unwrap();
            s.advance(10e-3).unwrap();
            let after_one = s.on_post(0.0).unwrap();
            s.advance(30e-3).unwrap();
            let dw = s.on_post(after_one).unwrap();
            assert!(dw.is_finite(), "{name}: non-finite step {dw}");
            assert!(dw > 0.0, "{name}: a pre-post-post triplet must potentiate, got {dw}");
            // The minimal row is the only one whose FIRST pair does nothing, because it is the only
            // one with `A2_plus = 0`. That is a prediction of the table, so assert it as one.
            assert_eq!(
                after_one == 0.0,
                a2p == 0.0,
                "{name}: the first pair's change disagrees with A2+ = {a2p}"
            );
        }

        // The two visual-cortex rows are two fits of two models to ONE data set, and they differ in
        // exactly the place that is easiest to get wrong: `tau_y`. 125 ms is the full model's.
        let vmin = TripletStdp::visual_cortex_minimal(b).unwrap();
        let vfull = TripletStdp::visual_cortex_full(b).unwrap();
        assert_eq!(vmin.o2.tau, 114e-3);
        assert_eq!(vfull.o2.tau, 125e-3);
        assert_ne!(vmin.o2.tau, vfull.o2.tau, "the two visual-cortex rows are not interchangeable");
        // Minimal means two amplitudes are exactly zero; full means neither is.
        assert_eq!((vmin.a2_plus, vmin.a3_minus), (0.0, 0.0));
        assert!(vfull.a2_plus > 0.0 && vfull.a3_minus > 0.0);
        // The two hippocampal rows are the SAME data refitted under a different pairing scheme, and
        // every fitted number moved — which is the concrete content of "fitted per scheme".
        let hall = TripletStdp::hippocampal_full(b).unwrap();
        let hnear = TripletStdp::hippocampal_full_nearest_spike(b).unwrap();
        assert_ne!(hall.pairing, hnear.pairing);
        for (l, r, what) in [
            (hall.a2_plus, hnear.a2_plus, "A2+"),
            (hall.a3_plus, hnear.a3_plus, "A3+"),
            (hall.a2_minus, hnear.a2_minus, "A2-"),
            (hall.a3_minus, hnear.a3_minus, "A3-"),
            (hall.r2.tau, hnear.r2.tau, "tau_x"),
            (hall.o2.tau, hnear.o2.tau, "tau_y"),
        ] {
            assert_ne!(l, r, "{what} is the same in both hippocampal rows, which it is not");
        }
        // And the two preparations are not the same model either: 946 ms against 101 ms on the pre
        // side, 27 ms against 125 ms on the post side, with the rank of the two REVERSED. The pre
        // side compares printed values only: Table 3 prints the 101 ms in parentheses, its mark
        // for a time constant the fit is insensitive to.
        assert!(hall.r2.tau > 9.0 * vfull.r2.tau);
        assert!(hall.o2.tau < vfull.o2.tau / 4.0);
        assert!(vfull.r2.tau < vfull.o2.tau && hall.r2.tau > hall.o2.tau);
    }

    /// THE MINIMAL VISUAL-CORTEX MODEL AGAINST ITS OWN PUBLISHED WINDOW, in closed form, with the
    /// table's numbers written out as literals rather than read back off the struct it is checking.
    ///
    /// With `A2_plus = 0` the slow post trace is the **only** thing this model's potentiation
    /// depends on, so `A3_plus` and `tau_y` are both load-bearing — and neither was bound by
    /// anything. The frequency test asserts "monotone and large", which is true of `tau_y = 114 ms`
    /// and of 125 ms alike, and multiplying `A3_plus` by ten left the whole suite green.
    #[test]
    fn the_minimal_visual_cortex_triplet_model_reproduces_its_published_window() {
        let mut t = TripletStdp::visual_cortex_minimal(Bounds::wide()).unwrap();

        // DEPRESSION IS A PAIR EFFECT ONLY. One post spike, one pre spike `lag` later:
        // `dw = -o1 * A2_minus`, with no triplet term at all because `A3_minus` is zero.
        let lag = 14e-3;
        t.clear();
        t.on_post(0.0).unwrap();
        t.advance(lag).unwrap();
        let dep = t.on_pre(0.0).unwrap();
        assert_eq!(dep, -((-lag / 33.7e-3f64).exp() * 7.1e-3), "pair depression: {dep}");

        // POTENTIATION IS A TRIPLET EFFECT ONLY. Pre, then post, then post. The FIRST post spike
        // finds `o2 = 0` and therefore changes the weight by exactly nothing — that is what
        // `A2_plus = 0` means, and it is the structural claim the word "minimal" is making.
        let (lag1, gap) = (9e-3, 120e-3);
        t.clear();
        t.on_pre(0.0).unwrap();
        t.advance(lag1).unwrap();
        let w1 = t.on_post(0.0).unwrap();
        assert_eq!(w1, 0.0, "the first post spike potentiated, so A2_plus is not zero");

        // The SECOND post spike reads `o2 = exp(-gap / 114 ms)` from the first, and `r1` has
        // decayed for the whole interval.
        t.advance(gap).unwrap();
        let w2 = t.on_post(w1).unwrap();
        let r1 = (-(lag1 + gap) / 16.8e-3f64).exp();
        let o2 = (-gap / 114e-3f64).exp();
        assert_eq!(w2, r1 * (6.5e-3 * o2), "triplet potentiation: {w2}");
        assert!(w2 > 0.0);

        // The fixture can tell 114 ms from 125 ms: the same protocol with the FULL model's `tau_y`
        // gives an answer 9% larger, which is a hundred million times any rounding here.
        let with_full_tau_y = r1 * (6.5e-3 * (-gap / 125e-3f64).exp());
        assert!(
            (with_full_tau_y - w2) / w2 > 0.05,
            "the fixture cannot distinguish tau_y = 114 ms from 125 ms: {w2} vs {with_full_tau_y}"
        );
        // And it can tell 6.5e-3 from anything else: the amplitude is a literal factor above.
        assert!((w2 / (r1 * o2) - 6.5e-3).abs() < 1e-18, "A3_plus is not 6.5e-3");
    }

    /// `tau_x` IS UNIDENTIFIABLE IN THE MINIMAL MODEL, and this is the test that says so out loud
    /// instead of pretending to bind a number nothing can bind.
    ///
    /// `A3_minus = 0` multiplies the slow pre trace out of the only term it ever appears in, so no
    /// value of `tau_x` changes any output — the paper leaves that cell of Table 3 blank for
    /// exactly this reason. A [`Trace`] still needs a positive time constant, so the constructor
    /// puts the full fit's 101 ms there; the honest check is that the choice cannot matter, run
    /// twice with `tau_x` two orders apart over the same 200-event train and compared bit for bit.
    #[test]
    fn the_minimal_triplet_models_slow_pre_trace_cannot_change_an_answer() {
        let run = |tau_x: f64, a3_minus: f64| {
            let mut t = TripletStdp::new(
                0.0,
                6.5e-3,
                7.1e-3,
                a3_minus,
                16.8e-3,
                tau_x,
                33.7e-3,
                114e-3,
                WeightRule::Additive,
                Bounds::wide(),
            )
            .unwrap();
            let mut rng = Rng::new(0xDEAD_BEEF);
            let mut w = 0.0;
            for _ in 0..200 {
                t.advance(rng.next_f64() * 30e-3).unwrap();
                w = if rng.next_f64() < 0.5 {
                    t.on_pre(w).unwrap()
                } else {
                    t.on_post(w).unwrap()
                };
            }
            w
        };
        // The published model: A3_minus = 0, so tau_x is inert across two orders of magnitude.
        let at_101ms = run(101e-3, 0.0);
        let at_5s = run(5.0, 0.0);
        assert_eq!(at_101ms, at_5s, "tau_x changed an answer in a model where A3_minus is zero");
        assert!(at_101ms.abs() > 1e-6, "the train never moved the weight, so the equality is vacuous");
        // And the constructor really does set A3_minus to zero, which is the whole mechanism.
        assert_eq!(TripletStdp::visual_cortex_minimal(Bounds::wide()).unwrap().a3_minus, 0.0);

        // THE CONTRAST, so this is a statement about `A3_minus = 0` and not about `tau_x` being
        // ignored everywhere: give the same model a non-zero triplet depression amplitude and the
        // same two time constants now disagree.
        let live_101ms = run(101e-3, 4.3e-3);
        let live_5s = run(5.0, 4.3e-3);
        assert_ne!(live_101ms, live_5s, "tau_x is inert even where A3_minus is not zero");
        assert!(
            (live_101ms - live_5s).abs() / live_101ms.abs() > 0.05,
            "the contrast is within rounding: {live_101ms} vs {live_5s}"
        );
    }

    /// `TripletStdp::clear` MUST FORGET ALL FOUR TRACES, not the two fast ones. A rule that cleared
    /// only `r1` and `o1` would leave `r2` and `o2` carrying spike history across a reset, and
    /// nothing in the suite looked: the reduction test sets both triplet amplitudes to zero, which
    /// multiplies the slow traces away.
    #[test]
    fn clearing_a_triplet_rule_forgets_all_four_traces() {
        let b = Bounds::wide();
        let mut t = TripletStdp::hippocampal_full(b).unwrap();
        t.note_pre();
        t.note_post();
        t.advance(5e-3).unwrap();
        assert!(t.r1.x > 0.0 && t.r2.x > 0.0 && t.o1.x > 0.0 && t.o2.x > 0.0, "the fixture is empty");
        t.clear();
        assert_eq!((t.r1.x, t.r2.x, t.o1.x, t.o2.x), (0.0, 0.0, 0.0, 0.0));

        // Behaviourally, where it bites. After the clear, one post spike sets `o1` and `o2`, and a
        // pre spike then reads `o1 * (A2_minus + A3_minus * r2)`. With `r2` properly cleared that
        // is the pair term alone; with a surviving `r2` it is 87% larger for this parameter set,
        // so the difference is not subtle.
        t.note_post();
        let dw = t.on_pre(0.0).unwrap();
        assert_eq!(dw, -(1.0 * t.a2_minus), "a cleared slow pre trace still contributed");
        let if_r2_survived = -(1.0 * (t.a2_minus + t.a3_minus));
        assert!(
            (dw - if_r2_survived).abs() / dw.abs() > 0.5,
            "the fixture cannot tell a cleared r2 from a surviving one"
        );
        // The parameters are untouched by a clear, which is the other half of the contract.
        assert_eq!(t.a3_plus, 6.7e-3);
        assert_eq!(t.r2.tau, 946e-3);
        assert_eq!(t.pairing, Pairing::AllToAll);
    }

    /// THE THIRD FACTOR'S HEADLINE, CHECKED AGAINST THE WINDOW RATHER THAN AGAINST ITSELF.
    ///
    /// "`STDP` does not write the weight, it writes an eligibility trace" is a claim about **what**
    /// gets written. Every other assertion about `c` in this module is relative to `c` — it is
    /// positive, it decreases, it is `c * exp(-2)` two seconds later, `pending_change()` is
    /// `c * d * tau_eff` — and all of them hold just as well if the tag is a constant. The tag is
    /// the window: same amplitude, same exponential, same weight dependence, bit for bit.
    #[test]
    fn the_eligibility_trace_is_the_stdp_window_itself() {
        let w = 0.37;
        for &lag in &[1e-3, 10e-3, 40e-3] {
            // Pre before post: the tag is the potentiating half of the window.
            let mut r = RewardStdp::new(wide_pair(), 1.0, 0.2, Bounds::wide()).unwrap();
            r.on_pre(w).unwrap();
            r.advance(w, lag).unwrap();
            r.on_post(w).unwrap();
            assert_eq!(r.c, r.stdp.window(lag), "pre->post at {lag} s tagged {}", r.c);
            assert!(r.c > 0.0);

            // Post before pre: the depressing half, sign and all.
            let mut r = RewardStdp::new(wide_pair(), 1.0, 0.2, Bounds::wide()).unwrap();
            r.on_post(w).unwrap();
            r.advance(w, lag).unwrap();
            r.on_pre(w).unwrap();
            assert_eq!(r.c, r.stdp.window(-lag), "post->pre at {lag} s tagged {}", r.c);
            assert!(r.c < 0.0);
        }

        // AND THE WEIGHT DEPENDENCE the doc promises. A soft bound at `mu = 1` makes the tag linear
        // in the distance to the ceiling, so the same pair tags a synapse at 0.25 exactly 0.75 as
        // hard as one at 0.0 — and a tag that ignored `w` would give them the same number.
        let tag_at = |w0: f64| {
            let s = PairStdp::new(
                0.05,
                0.05,
                16.8e-3,
                33.7e-3,
                WeightRule::SoftBound { mu: 1.0 },
                Bounds::normalised(),
            )
            .unwrap();
            let mut r = RewardStdp::new(s, 1.0, 0.2, Bounds::normalised()).unwrap();
            r.on_pre(w0).unwrap();
            r.advance(w0, 10e-3).unwrap();
            r.on_post(w0).unwrap();
            r.c
        };
        let at_floor = tag_at(0.0);
        let at_quarter = tag_at(0.25);
        let at_ceiling = tag_at(1.0);
        assert!(at_floor > 0.0);
        assert_eq!(at_quarter, 0.75 * at_floor, "the tag is not linear in the distance to the ceiling");
        assert_eq!(at_ceiling, 0.0, "a synapse at the ceiling cannot be tagged for potentiation");
    }

    /// `RewardStdp::advance` MUST DECAY THE WINDOW'S OWN TRACES, and nothing noticed that it does.
    ///
    /// Every fixture in this module ran a single pair through this rule, so deleting
    /// `self.stdp.advance(dt)` changed nothing anyone checked. Without it the pre trace stands at
    /// one forever, every post spike in a train tags the full amplitude, and the eligibility trace
    /// stops depending on spike timing at all — which is the one thing it is for.
    #[test]
    fn the_eligibility_trace_decays_the_stdp_window_between_spikes() {
        let mut r = RewardStdp::new(wide_pair(), 1.0, 0.2, Bounds::wide()).unwrap();
        let w = 0.0;
        r.on_pre(w).unwrap();
        // Fifty milliseconds, in five calls, so the decay has to COMPOSE across `advance`.
        for _ in 0..5 {
            r.advance(w, 10e-3).unwrap();
        }
        r.on_post(w).unwrap();
        let want = r.stdp.window(50e-3);
        assert!(
            (r.c - want).abs() / want < 1e-12,
            "tag {} vs the window at 50 ms {want}",
            r.c
        );
        // Un-decayed it would be the full amplitude — twenty times larger, which is the size of the
        // error this test exists to catch.
        assert!(r.c < 0.1 * r.stdp.a_plus, "the pre trace did not decay: {} of {}", r.c, r.stdp.a_plus);
        // The traces themselves, against the same closed form.
        assert!((r.stdp.pre_trace.x - (-50e-3f64 / 16.8e-3).exp()).abs() < 1e-15);
    }

    /// `RewardStdp::clear` forgets the tag, the modulator and the spike history, and keeps the
    /// parameters. A cleared rule cannot move a weight however long it runs, which is the property
    /// a caller reusing one object across synapses depends on.
    #[test]
    fn clearing_a_reward_rule_forgets_the_tag_the_modulator_and_the_spike_history() {
        let mut r = RewardStdp::new(wide_pair(), 1.0, 0.2, Bounds::wide()).unwrap();
        r.on_pre(0.0).unwrap();
        r.advance(0.0, 10e-3).unwrap();
        r.on_post(0.0).unwrap();
        r.reward(0.9).unwrap();
        assert!(r.c > 0.0 && r.d > 0.0 && r.pending_change() > 0.0, "the fixture is empty");

        r.clear();
        assert_eq!(r.c, 0.0);
        assert_eq!(r.d, 0.0);
        assert_eq!(r.pending_change(), 0.0);
        assert_eq!(r.stdp.pre_trace.x, 0.0);
        assert_eq!(r.stdp.post_trace.x, 0.0);

        let mut w = 0.37;
        for _ in 0..1_000 {
            w = r.advance(w, 1e-3).unwrap();
        }
        assert_eq!(w, 0.37, "a cleared rule moved a weight");

        // Parameters survived.
        assert_eq!(r.tau_c, 1.0);
        assert_eq!(r.tau_d, 0.2);
        assert_eq!(r.stdp.a_plus, 0.1);
        assert!((r.tau_effective() - 1.0 * 0.2 / 1.2).abs() < 1e-15);
        // A non-finite reward is refused by name rather than poisoning the modulator.
        assert!(matches!(
            r.reward(f64::NAN),
            Err(PlasticityError::NonFinite { what: "reward", .. })
        ));
        assert_eq!(r.d, 0.0, "a refused reward still changed the modulator");
    }

    /// WHICH `theta` THE `BCM` WEIGHT STEP READS, pinned against the closed form for a single
    /// presentation.
    ///
    /// The rule couples two variables and the code has to evaluate them in some order. It uses the
    /// threshold from **before** this presentation — both derivatives at the state at the start of
    /// the interval, which is how the published equations are written. Swapping the two blocks
    /// leaves every other test in this module green: the selectivity run's winner moves from
    /// 40.42 Hz to 39.82 Hz and the band there is 5%, and the threshold test sets `eta = 0`, which
    /// multiplies the weight step away exactly as `A3 = 0` multiplied the slow trace away in the
    /// triplet rule's "just before this spike" defect. Same shape, same blindness.
    #[test]
    fn the_bcm_weight_step_uses_the_threshold_from_before_this_presentation() {
        let bounds = Bounds::new(0.0, 5.0).unwrap();
        let (eta, tau_theta, y_0, theta0, dt) = (2.0e-4, 0.5, 10.0, 7.0, 0.05);
        let mut b = Bcm::new(vec![0.5, 0.0], eta, tau_theta, y_0, theta0, bounds).unwrap();
        let x = [40.0, 0.0];
        let y = 20.0;
        assert_eq!(b.update(&x, dt).unwrap(), y, "the fixture's output is not what it claims");

        // The weight step, in closed form, with `theta` AS IT WAS.
        let want_w = 0.5 + eta * dt * x[0] * (y * (y - theta0));
        assert_eq!(b.w[0], want_w, "the weight step did not use the threshold from before");
        // A channel with no input cannot move, whatever the threshold says.
        assert_eq!(b.w[1], 0.0);
        // And the threshold relaxed AFTERWARDS, toward this presentation's own `y^2 / y_0`.
        let want_theta = {
            let target = y * y / y_0;
            target + (theta0 - target) * (-dt / tau_theta).exp()
        };
        assert_eq!(b.theta, want_theta, "the threshold did not relax to its closed form");

        // The fixture can tell the two orders apart. Using the RELAXED threshold instead would make
        // this step 24% smaller, which is far outside anything rounding could explain.
        let swapped = 0.5 + eta * dt * x[0] * (y * (y - want_theta));
        assert!(
            (want_w - swapped).abs() / (want_w - 0.5).abs() > 0.15,
            "the fixture cannot distinguish the two orders: {want_w} vs {swapped}"
        );
        assert!(want_theta > theta0, "the threshold must have moved at all");
    }

    /// (e, continued) THE BOUNDS BATTERY, EXTENDED TO `Bcm` — the one bounded rule it skipped.
    ///
    /// `Bcm::update` clamps every channel on every presentation and nothing ever asked it to. In
    /// the selectivity fixture the winner settles at `w = 0.8` against a ceiling of 5.0 and the
    /// losers approach the floor asymptotically from above, so removing the ceiling — or removing
    /// the clamp outright — changed nothing any test could see. The house rule is stated in
    /// `weight_bounds_are_never_violated_under_adversarial_input` and it applies here: a bound
    /// nothing ever reaches is a bound nothing ever tested.
    #[test]
    fn bcm_weights_never_leave_their_bounds_and_both_clamps_bind() {
        // A floor ABOVE zero, so a clamped cell still has an output and the run keeps moving; with
        // a floor at zero a depressed cell is silent forever and the ceiling is never revisited.
        // `eta` is six orders above the selectivity fixture's, so one presentation overshoots both
        // ends of the interval and the rule spends the whole run pinned alternately to each.
        let bounds = Bounds::new(0.2, 1.0).unwrap();
        let mut b = Bcm::new(vec![0.5, 0.3], 1.0, 0.05, 10.0, 100.0, bounds).unwrap();
        let mut rng = Rng::new(0x0BC0_B0DE);
        let (mut hit_floor, mut hit_ceiling) = (false, false);
        for k in 0..20_000u32 {
            let mut x = vec![0.0; 2];
            x[rng.below(2) as usize] = 40.0;
            b.update(&x, 0.01).unwrap();
            for &wi in &b.w {
                assert!(bounds.contains(wi), "step {k}: BCM left the bound at w = {wi}");
                hit_floor |= wi == bounds.w_min;
                hit_ceiling |= wi == bounds.w_max;
            }
        }
        assert!(hit_floor, "the floor never bound, so it was never tested");
        assert!(hit_ceiling, "the ceiling never bound, so it was never tested");

        // The refusals on the same call, which nothing exercised either.
        assert!(matches!(
            b.update(&[1.0, 0.0], -1e-3),
            Err(PlasticityError::Negative { what: "dt", .. })
        ));
        assert!(matches!(
            b.update(&[1.0], 0.01),
            Err(PlasticityError::LengthMismatch { got: 1, want: 2, .. })
        ));
        assert!(matches!(
            b.update(&[f64::NAN, 0.0], 0.01),
            Err(PlasticityError::NonFinite { what: "input", .. })
        ));
        // Zero patterns has no selective equilibrium, and inventing one would be a lie.
        assert_eq!(b.selective_fixed_point(0), None);
        assert_eq!(b.selective_fixed_point(1), Some(10.0));
        assert_eq!(b.selective_fixed_point(7), Some(70.0));
    }

    /// GÜTIG'S EXPONENT ACTS ON THE **NORMALISED** DISTANCE, checked at a span that is not one.
    ///
    /// [`Bounds::normalised`]'s doc claims the normalisation is the identity on `[0, 1]` and that
    /// `mu` therefore means what the paper says it means. That is only worth saying if the division
    /// is there, and both soft-bound fixtures in this module were blind to it: one used
    /// `Bounds::new(-0.25, 0.75)`, whose span is exactly 1.0, and the other used `mu = 0.0`, where
    /// `powf` returns one whatever its argument. Deleting the division entirely left them green.
    #[test]
    fn the_soft_bound_exponent_acts_on_the_normalised_distance_to_the_bound() {
        // The invariant: the factor depends on WHERE IN THE INTERVAL the weight sits, not on how
        // wide the interval happens to be. Without the division it would scale as `span^mu`.
        let narrow = Bounds::normalised();
        let wide = Bounds::new(-3.0, 7.0).unwrap(); // span 10, so a missing division is 10^mu out
        assert_eq!(wide.span(), 10.0);
        for &mu in &[0.25, 0.5, 1.0, 2.0] {
            let r = WeightRule::SoftBound { mu };
            for &f in &[0.0, 0.1, 0.5, 0.9, 1.0] {
                let a = narrow.w_min + f * narrow.span();
                let b = wide.w_min + f * wide.span();
                assert!(
                    (r.potentiation_factor(a, narrow) - r.potentiation_factor(b, wide)).abs() < 1e-15,
                    "potentiation at {f} of the way, mu {mu}: {} vs {}",
                    r.potentiation_factor(a, narrow),
                    r.potentiation_factor(b, wide)
                );
                assert!(
                    (r.depression_factor(a, narrow) - r.depression_factor(b, wide)).abs() < 1e-15,
                    "depression at {f} of the way, mu {mu}"
                );
            }
            // Half way down the wide interval is one half, not five.
            assert!((r.potentiation_factor(2.0, wide) - 0.5f64.powf(mu)).abs() < 1e-15);
        }

        // One literal, so the arithmetic is pinned and not only its invariance: 40% of the way down
        // from the ceiling of `[-1, 4]` — span 5 — with Gütig's `mu = 1/2`, is `sqrt(0.4)`.
        let b = Bounds::new(-1.0, 4.0).unwrap();
        let half = WeightRule::SoftBound { mu: 0.5 };
        assert_eq!(half.potentiation_factor(2.0, b), 0.632_455_532_033_675_9);
        assert_eq!(half.depression_factor(1.0, b), 0.632_455_532_033_675_9);
        // `mu = 0` is the additive rule everywhere, including at the bounds themselves, because
        // `x.powf(0.0)` is one for every `x` including zero.
        let flat = WeightRule::SoftBound { mu: 0.0 };
        for &w in &[-1.0, 0.0, 2.5, 4.0] {
            assert_eq!(flat.potentiation_factor(w, b), 1.0);
            assert_eq!(flat.depression_factor(w, b), 1.0);
        }
        // The two rules with no weight dependence at all say so for every weight and every bound.
        for &w in &[-1.0, 1.5, 4.0] {
            assert_eq!(WeightRule::Additive.potentiation_factor(w, b), 1.0);
            assert_eq!(WeightRule::Additive.depression_factor(w, b), 1.0);
            assert_eq!(WeightRule::MultiplicativeDepression.potentiation_factor(w, b), 1.0);
        }
    }

    /// (Hebb, with decay) THE TERM THAT DOES NOT DO WHAT THE DOC USED TO SAY IT DID.
    ///
    /// `decay` was documented as bounding the norm at `sqrt(eta * lambda_1 / decay)`-ish. It cannot:
    /// the term is **linear** in `w`, so it shifts every eigenvalue of the input correlation matrix
    /// down by `decay` and leaves the rule linear. The norm stays geometric and the only fixed
    /// point anywhere is zero. Nothing caught the claim because no test ever set `decay` to
    /// anything but zero — flipping its sign and deleting it outright were equally invisible.
    #[test]
    fn hebbian_decay_shifts_the_eigenvalue_and_never_bounds_the_norm() {
        let u = [0.6, 0.8]; // unit, so C = u u^T and lambda_1 = 1
        let eta = 0.01;
        let w0 = 0.1;

        // The exact geometric closed form, with the decay in it: one multiplication by
        // `1 + eta * (lambda_1 - decay)` per sample. Both sides of `lambda_1`, and on it.
        for &decay in &[0.0, 0.05, 0.5, 1.0, 2.0] {
            let mut h = Hebbian::new(vec![w0 * u[0], w0 * u[1]], eta, decay).unwrap();
            for _ in 0..200 {
                h.update(&u).unwrap();
            }
            let want = w0 * (1.0 + eta * (1.0 - decay)).powi(200);
            assert!(
                (h.norm() - want).abs() / want < 1e-9,
                "decay {decay}: norm {} vs closed form {want}",
                h.norm()
            );
        }
        // `decay == lambda_1` is the knife edge: the rule stands exactly still.
        let mut h = Hebbian::new(vec![w0 * u[0], w0 * u[1]], eta, 1.0).unwrap();
        for _ in 0..5_000 {
            h.update(&u).unwrap();
        }
        assert!((h.norm() - w0).abs() / w0 < 1e-9, "at decay = lambda_1 the norm moved: {}", h.norm());

        // THE OLD DOC'S CLAIM, REFUTED AT ITS OWN NUMBERS. `sqrt(eta * lambda_1 / decay)` with
        // `eta = 0.01` and `decay = 0.05` is 0.447. The rule blows past it and keeps climbing at a
        // CONSTANT ratio, which is what "not bounded" means: every 5,000 samples multiply the norm
        // by the same factor, forever.
        let mut h = Hebbian::new(vec![w0 * u[0], w0 * u[1]], eta, 0.05).unwrap();
        let start = h.norm();
        for _ in 0..5_000 {
            h.update(&u).unwrap();
        }
        let at_5k = h.norm();
        for _ in 0..5_000 {
            h.update(&u).unwrap();
        }
        let at_10k = h.norm();
        assert!(at_5k > 100.0 * 0.447, "the claimed bound held: {at_5k}");
        let (first, second) = (at_5k / start, at_10k / at_5k);
        assert!(
            (second / first - 1.0).abs() < 1e-6,
            "the growth ratio changed, so something bounded it: {first} then {second}"
        );

        // A decay ABOVE lambda_1 does not settle the weights anywhere either — it deletes them.
        let mut h = Hebbian::new(vec![w0 * u[0], w0 * u[1]], eta, 2.0).unwrap();
        for _ in 0..20_000 {
            h.update(&u).unwrap();
        }
        assert!(h.norm() < 1e-80, "decay above lambda_1 left a norm of {}", h.norm());

        // And the SIGN of the term: a decay must slow growth, never accelerate it.
        let grow = |decay: f64| {
            let mut h = Hebbian::new(vec![w0 * u[0], w0 * u[1]], eta, decay).unwrap();
            for _ in 0..200 {
                h.update(&u).unwrap();
            }
            h.norm()
        };
        assert!(grow(0.2) < grow(0.0), "decay accelerated growth, so its sign is wrong");
        assert!(grow(0.4) < grow(0.2));
    }

    /// THE ZERO-LAG CONVENTION, both halves, because the type doc puts a claim about it in bold.
    ///
    /// [`PairStdp::window`] returns `0.0` at exactly zero lag — a stated convention where the
    /// experiment has no measurement — while [`PairStdp::apply_pair`] at zero lag delivers
    /// pre-then-post with nothing between them and pays the full `A_plus`. The two answers disagree
    /// on purpose, and neither returning `a_plus` from `window` nor sending zero lag down the
    /// post-first branch was visible to anything.
    #[test]
    fn the_window_and_the_online_path_disagree_at_exactly_zero_lag() {
        let mut s = wide_pair();
        assert_eq!(s.window(0.0), 0.0, "the closed form's convention at zero lag");
        assert_eq!(s.window(-0.0), 0.0);
        assert_eq!(s.apply_pair(0.0, 0.0).unwrap(), s.a_plus, "the online path at zero lag");
        // Negative zero satisfies `lag >= 0.0` in IEEE, so it takes the same branch. The code
        // relies on that; asserting it is cheaper than rediscovering it.
        assert_eq!(s.apply_pair(0.0, -0.0).unwrap(), s.a_plus);
        assert!(s.a_plus > 0.0, "the disagreement is a real one, not two zeros");

        // Either side of zero the two paths agree again exactly, arbitrarily close in — so the
        // disagreement is the window's discontinuity and not a bug in the neighbourhood.
        for &lag in &[1e-12, 1e-9, 1e-6] {
            assert_eq!(s.apply_pair(0.0, lag).unwrap(), s.window(lag), "at +{lag} s");
            assert_eq!(s.apply_pair(0.0, -lag).unwrap(), s.window(-lag), "at -{lag} s");
            assert!(s.window(lag) > 0.0 && s.window(-lag) < 0.0);
        }
        // A `NaN` lag falls through `window`'s comparisons to the zero branch — a silent answer to
        // a malformed question, which is why the doc says to prefer `apply_pair`, which refuses.
        assert_eq!(s.window(f64::NAN), 0.0);
        assert!(matches!(
            s.apply_pair(0.0, f64::NAN),
            Err(PlasticityError::NonFinite { what: "lag", .. })
        ));
        assert!(matches!(
            s.apply_pair(f64::INFINITY, 1e-3),
            Err(PlasticityError::NonFinite { what: "weight", .. })
        ));
    }

    /// `Trace::after` IS THE CLOSED FORM `Trace::advance` STEPS ALONG. Its doc says it is exposed
    /// so a test can compare an online run against it rather than against a previous online run,
    /// and then nothing in the crate ever called it.
    #[test]
    fn the_trace_closed_form_is_the_online_decay() {
        let mut t = Trace::new(20e-3).unwrap();
        assert_eq!(t.x, 0.0, "a new trace is at rest");
        assert_eq!(t.after(1.0), 0.0, "and a trace at rest stays there");

        t.fire(Pairing::AllToAll);
        assert_eq!(t.x, 1.0, "one immediately after an isolated spike");
        for &dt in &[0.0, 1e-4, 5e-3, 20e-3, 1.0] {
            let want = t.after(dt);
            let mut online = t;
            online.advance(dt).unwrap();
            assert_eq!(online.x, want, "after({dt}) and advance({dt}) are not the same arithmetic");
        }
        // `after` is a question, not a step: it does not move the trace.
        let before = t.x;
        let _ = t.after(5.0);
        assert_eq!(t.x, before);

        // One time constant out leaves exactly 1/e, which is the identity the whole window rests
        // on — the trace read at a partner spike IS `exp(-lag / tau)`.
        let mut u = Trace::new(20e-3).unwrap();
        u.fire(Pairing::NearestNeighbour);
        assert!((u.after(20e-3) - 1.0 / std::f64::consts::E).abs() < 1e-16);
        // And `clear` returns it to rest without touching `tau`.
        u.advance(7e-3).unwrap();
        assert!(u.x > 0.0);
        u.clear();
        assert_eq!(u.x, 0.0);
        assert_eq!(u.tau, 20e-3);
        // Negative time would AMPLIFY the trace, so it is refused rather than run.
        assert!(matches!(u.advance(-1e-9), Err(PlasticityError::Negative { what: "dt", .. })));
    }

    /// `Bounds::normalised` IS THE INTERVAL GÜTIG'S EXPONENT IS DEFINED ON, and `Bounds::wide` is
    /// the finite stand-in for "no bound". Both had docs making claims and neither had a test.
    #[test]
    fn the_normalised_interval_is_where_the_soft_bound_normalisation_is_the_identity() {
        let n = Bounds::normalised();
        assert_eq!(n, Bounds::new(0.0, 1.0).unwrap());
        assert_eq!(n.span(), 1.0);
        assert!(n.contains(0.0) && n.contains(1.0), "the interval is closed at both ends");
        assert!(!n.contains(-1e-12) && !n.contains(1.0 + 1e-12));
        // The identity the doc claims: dividing by a span of one is a no-op, so the factor is the
        // raw distance to the bound and `mu` is the paper's exponent on the paper's quantity.
        for &mu in &[0.3, 1.0, 2.0] {
            let r = WeightRule::SoftBound { mu };
            for &w in &[0.0, 0.2, 0.75, 1.0] {
                assert_eq!(r.potentiation_factor(w, n), (1.0 - w).powf(mu), "potentiation at {w}");
                assert_eq!(r.depression_factor(w, n), w.powf(mu), "depression at {w}");
            }
        }

        let w = Bounds::wide();
        assert_eq!(w.span(), 2.0e12);
        assert!(w.span().is_finite(), "an infinite span makes every soft-bound factor zero");
        assert!(w.contains(0.0) && w.contains(1e11) && w.contains(-1e12));
        assert!(!w.contains(1e13) && !w.contains(f64::NAN));
        assert_eq!(w.clamp(1e13), 1.0e12);
        assert_eq!(w.clamp(-1e13), -1.0e12);
        assert_eq!(w.clamp(3.5), 3.5);
        // The headroom the doc claims, as arithmetic rather than as prose: 296 DECADES, which is
        // about 984 powers of two, not 296 of them.
        assert!(((f64::MAX / 1e12).log10() - 296.0).abs() < 1.0);
        assert!(((f64::MAX / 1e12).log2() - 984.0).abs() < 1.0);
        // A NaN weight is not inside any interval, which is what stops the clamp laundering it.
        assert!(!n.contains(f64::NAN));
        assert!(w.clamp(f64::NAN).is_nan(), "the clamp cannot repair a NaN and does not pretend to");
    }

    /// AN INVERTED `Bounds` BUILT BY LITERAL REPORTS ITSELF rather than pretending to work. The
    /// fields are `pub`, so `Bounds::new`'s check can be bypassed; what must not happen is that the
    /// bypass looks like a working interval.
    #[test]
    fn an_inverted_bounds_literal_reports_itself_rather_than_pretending_to_work() {
        let bad = Bounds { w_min: 5.0, w_max: -5.0 };
        for &w in &[-1e9, -5.0, 0.0, 5.0, 1e9] {
            assert!(!bad.contains(w), "{w} reported as inside an empty interval");
        }
        assert!(bad.span() < 0.0, "a negative span is the second tell");
        // And the constructor refuses to build one, at inversion and at equality alike — a zero
        // span would make every soft-bound factor a division by zero.
        assert!(matches!(
            Bounds::new(5.0, -5.0),
            Err(PlasticityError::BoundsInverted { w_min: 5.0, w_max: -5.0 })
        ));
        assert!(matches!(Bounds::new(2.0, 2.0), Err(PlasticityError::BoundsInverted { .. })));
        assert!(matches!(
            Bounds::new(f64::NAN, 1.0),
            Err(PlasticityError::NonFinite { what: "weight floor", .. })
        ));
        assert!(matches!(
            Bounds::new(f64::NEG_INFINITY, 1.0),
            Err(PlasticityError::NonFinite { what: "weight floor", .. })
        ));
    }

    /// A WEIGHT OUTSIDE ITS BOUNDS CAN NEVER PRODUCE A NON-FINITE STEP.
    ///
    /// `PairStdp::new(0.05, 0.05, .., SoftBound { mu: 200.0 }, Bounds::normalised())` followed by
    /// `on_pre(1e6)` used to return **`Ok(NaN)`**: the soft bound's `((w - w_min) / span).powf(200)`
    /// is an infinity at `w = 1e6`, the post trace was `0.0`, and `0.0 * inf` is `NaN`. The clamp
    /// cannot catch that — a `NaN` satisfies neither of its comparisons — so it came back wearing
    /// the shape of a weight, which is the exact failure [`PlasticityError`] exists to prevent.
    #[test]
    fn a_weight_outside_its_bounds_can_never_produce_a_non_finite_step() {
        let b = Bounds::normalised();
        for &mu in &[0.5, 1.0, 200.0] {
            for &w in &[-1e6, -1.0, -1e-9, 1.0 + 1e-9, 1e6] {
                let rule = WeightRule::SoftBound { mu };
                let mut s = PairStdp::new(0.05, 0.05, 16.8e-3, 33.7e-3, rule, b).unwrap();
                let out = s.on_pre(w).unwrap();
                assert!(out.is_finite() && b.contains(out), "on_pre({w}) at mu {mu} gave {out}");
                let mut s = PairStdp::new(0.05, 0.05, 16.8e-3, 33.7e-3, rule, b).unwrap();
                let out = s.on_post(w).unwrap();
                assert!(out.is_finite() && b.contains(out), "on_post({w}) at mu {mu} gave {out}");
                // The factors themselves, which is where the infinity lived.
                assert!(rule.potentiation_factor(w, b).is_finite(), "potentiation factor at {w}");
                assert!(rule.depression_factor(w, b).is_finite(), "depression factor at {w}");
            }
            // The factor SATURATES at the nearest bound rather than extrapolating past it. The raw
            // expression at `w = -1` in `[0, 1]` gives `2^mu`, so a soft bound would AMPLIFY
            // potentiation outside its own interval instead of damping it.
            let rule = WeightRule::SoftBound { mu };
            assert_eq!(rule.potentiation_factor(-1.0, b), rule.potentiation_factor(0.0, b));
            assert_eq!(rule.potentiation_factor(-1.0, b), 1.0);
            assert_eq!(rule.depression_factor(2.0, b), rule.depression_factor(1.0, b));
            assert_eq!(rule.depression_factor(2.0, b), 1.0);
        }
        // Multiplicative depression saturates for the same reason, rather than growing without
        // bound in the weight's own unit.
        assert_eq!(WeightRule::MultiplicativeDepression.depression_factor(1e12, b), 1.0);
        assert_eq!(WeightRule::MultiplicativeDepression.depression_factor(-1e12, b), 0.0);
    }

    /// [`WeightRule::MultiplicativeDepression`]'s factor is the weight in its **own unit**, not a
    /// fraction of the span.
    ///
    /// Every other fixture in this module bounds weights to `[0, 1]`, whose span is exactly one,
    /// and a factor divided by one is the same factor. The one test that varies the floor,
    /// `multiplicative_depression_shrinks_as_the_weight_approaches_the_floor`, asserts only RATIOS
    /// of two steps, and a common normalisation cancels out of a ratio. So dividing by the span —
    /// which is what the rule's doc says van Rossum et al. do NOT do — passed the whole module.
    /// A span that is not one, compared against [`WeightRule::Additive`] at exactly one unit above
    /// the floor, is what pins the scale.
    #[test]
    fn the_multiplicative_depression_factor_is_the_weight_itself_not_a_fraction_of_the_span() {
        let b = Bounds::new(0.0, 4.0).unwrap();
        let step = |rule: WeightRule, w: f64| {
            let mut s = PairStdp::new(0.05, 0.05, 16.8e-3, 33.7e-3, rule, b).unwrap();
            w - s.apply_pair(w, -5e-3).unwrap()
        };
        let mult = step(WeightRule::MultiplicativeDepression, 1.0);
        let add = step(WeightRule::Additive, 1.0);
        assert!(mult > 0.0, "the fixture must actually depress");
        // At `w - w_min = 1` the factor is exactly 1.0 and the two rules take the same step to the
        // last bit. Normalised by the span of 4 it would be a quarter of it.
        assert_eq!(mult, add, "at one unit above the floor the factor is exactly one");
        // Linear in that unit and not in the fraction: two units above the floor is twice the step.
        let twice = step(WeightRule::MultiplicativeDepression, 2.0);
        assert!((twice / mult - 2.0).abs() < 1e-12, "{twice} is not twice {mult}");
    }

    /// The eligibility trace ACCUMULATES over pairs and the modulator ACCUMULATES over rewards.
    ///
    /// Both are `+=` in the source and both read as `=` under any test that delivers exactly one
    /// pair and exactly one reward — which was every test in this module. A tag that is overwritten
    /// rather than accumulated turns a burst into its last pair alone, which is precisely the
    /// quantity a three-factor rule exists to carry across the gap to the reward.
    #[test]
    fn the_tag_and_the_modulator_both_accumulate_rather_than_overwrite() {
        let b = Bounds::normalised();
        let lag = 6e-3;
        let one_pair = |r: &mut RewardStdp| {
            r.on_pre(0.5).expect("pre");
            r.advance(0.5, lag).expect("advance");
            r.on_post(0.5).expect("post");
        };

        let mut single = RewardStdp::new(wide_pair(), 1.0, 0.2, b).unwrap();
        one_pair(&mut single);
        let one = single.c;
        assert!(one > 0.0, "a pre-before-post pair tags for potentiation");

        let mut r = RewardStdp::new(wide_pair(), 1.0, 0.2, b).unwrap();
        one_pair(&mut r);
        r.advance(0.5, 0.5).expect("gap");
        // The WINDOW's traces are cleared between the pairs, and only those: `clear` on the inner
        // `PairStdp` leaves `c` and `d` untouched. That makes the second pair arithmetically
        // identical to the first, so the expected tag is an exact expression rather than a
        // tolerance around one.
        r.stdp.clear();
        let carried = r.c;
        assert!(carried > 0.0 && carried < one, "the tag decayed over the gap: {carried}");
        one_pair(&mut r);
        let want = carried * (-lag / r.tau_c).exp() + one;
        assert!(
            (r.c - want).abs() <= 1e-15 * want.abs(),
            "the tag is {}, not the carried {} plus a second pair {}",
            r.c,
            carried,
            one
        );
        assert!(r.c > one, "two pairs must tag more than one");

        // The modulator, the same way: two impulses sum, and a punishment on top of a reward
        // cancels it, which is the same arithmetic and the reason it cannot be an assignment.
        let mut r = RewardStdp::new(wide_pair(), 1.0, 0.2, b).unwrap();
        r.reward(1.0).unwrap();
        r.reward(1.0).unwrap();
        assert_eq!(r.d, 2.0, "two reward impulses must sum");
        r.reward(-2.0).unwrap();
        assert_eq!(r.d, 0.0, "a punishment of equal size cancels the reward");
    }

    /// A PARAMETER WRITTEN INTO A `pub` FIELD AFTER CONSTRUCTION is refused by name rather than
    /// written into a weight. Every field on every rule here is public and every constructor
    /// validation is therefore advisory; the guard that is not advisory is the check on the value
    /// the rule just computed.
    #[test]
    fn a_rule_poisoned_after_construction_refuses_rather_than_returning_a_nan_weight() {
        let b = Bounds::normalised();
        let mut s = PairStdp::new(0.05, 0.05, 16.8e-3, 33.7e-3, WeightRule::Additive, b).unwrap();
        s.note_post();
        s.a_minus = f64::NAN;
        assert!(matches!(
            s.on_pre(0.5),
            Err(PlasticityError::Diverged { what: "updated weight", .. })
        ));
        // The refusal left the traces exactly as it found them, so a caller who fixes the parameter
        // can carry on rather than having silently lost a spike. BOTH traces, and the PRE trace is
        // the load-bearing one: `on_pre` writes the pre trace and reads the post trace, so an
        // assertion on the post trace alone is equally true when the spike is registered BEFORE the
        // refusal instead of after it. Moving `note_pre` above the guard passed this test as it
        // stood.
        assert_eq!(s.pre_trace.x, 0.0, "a refused pre spike registered itself anyway");
        assert_eq!(s.post_trace.x, 1.0, "a refused pre spike disturbed the partner trace");
        s.a_minus = 0.05;
        assert!(s.on_pre(0.5).unwrap().is_finite());

        // And the mirror call, because `on_pre` and `on_post` are two pieces of code and only one
        // of them was exercised here: a refused POST spike must not have registered itself either.
        let mut s = PairStdp::new(0.05, 0.05, 16.8e-3, 33.7e-3, WeightRule::Additive, b).unwrap();
        s.note_pre();
        s.a_plus = f64::NAN;
        assert!(matches!(
            s.on_post(0.5),
            Err(PlasticityError::Diverged { what: "updated weight", .. })
        ));
        assert_eq!(s.post_trace.x, 0.0, "a refused post spike registered itself anyway");
        assert_eq!(s.pre_trace.x, 1.0, "a refused post spike disturbed the partner trace");

        let mut t = TripletStdp::visual_cortex_minimal(b).unwrap();
        t.note_pre();
        t.a3_plus = f64::INFINITY;
        assert!(matches!(
            t.on_post(0.5),
            Err(PlasticityError::Diverged { what: "updated weight", .. })
        ));
        // Again the side the call would WRITE, not the side it reads: `note_post` fires `o1` and
        // `o2`, and `r1` is the partner trace the spike was never going to touch.
        assert_eq!(t.o1.x, 0.0, "a refused post spike registered itself anyway");
        assert_eq!(t.o2.x, 0.0, "a refused post spike registered itself anyway");
        assert_eq!(t.r1.x, 1.0, "a refused post spike disturbed the partner trace");

        // The same guard on the three-factor rule's tag and on its weight.
        let mut r = RewardStdp::new(wide_pair(), 1.0, 0.2, b).unwrap();
        r.stdp.post_trace.x = f64::INFINITY;
        r.stdp.a_minus = 0.0;
        assert!(matches!(
            r.on_pre(0.5),
            Err(PlasticityError::Diverged { what: "eligibility trace", .. })
        ));
        let mut r = RewardStdp::new(wide_pair(), 1.0, 0.2, b).unwrap();
        r.c = f64::NAN;
        r.d = 1.0;
        assert!(matches!(
            r.advance(0.5, 1e-3),
            Err(PlasticityError::Diverged { what: "updated weight", .. })
        ));
    }

    /// `PairStdp::bi_poo_2001` CARRIES THE TIME CONSTANTS AND NOT THE AMPLITUDES, on purpose, and
    /// nothing checked either half of that.
    #[test]
    fn the_bi_poo_constructor_carries_the_published_widths_and_the_callers_amplitudes() {
        let b = Bounds::normalised();
        let s = PairStdp::bi_poo_2001(0.008, 0.009, WeightRule::Additive, b).unwrap();
        assert_eq!(s.tau_plus, 16.8e-3);
        assert_eq!(s.tau_minus, 33.7e-3);
        assert_eq!(s.a_plus, 0.008, "the amplitudes are the caller's, unchanged");
        assert_eq!(s.a_minus, 0.009);
        assert_eq!(s.pre_trace.tau, 16.8e-3, "the traces carry the same widths as the window");
        assert_eq!(s.post_trace.tau, 33.7e-3);
        assert_eq!(s.pairing, Pairing::AllToAll, "all-to-all is the documented default");
        assert_eq!((s.pre_trace.x, s.post_trace.x), (0.0, 0.0));
        // Depression is the WIDER window in Bi & Poo's 2001 exponential fit, and that is the
        // reason a rule with nearly equal amplitudes can still be depression-dominated. The 1998
        // paper reports 20-ms windows on both sides, and its "asymmetry" is the sign flip with the
        // order of the spikes. This comment used to call the width "the asymmetry Bi & Poo
        // measured".
        assert!(s.tau_minus > s.tau_plus);
        assert!(s.is_depression_dominated(), "area {}", s.total_window_area());
        // The same two constants are what every row of both Pfister & Gerstner tables inherits.
        let t = TripletStdp::hippocampal_full(b).unwrap();
        assert_eq!((t.r1.tau, t.o1.tau), (s.tau_plus, s.tau_minus));
        // A negative amplitude is refused rather than silently made positive by the sign convention.
        assert!(matches!(
            PairStdp::bi_poo_2001(0.008, -0.009, WeightRule::Additive, b),
            Err(PlasticityError::Negative { what: "A_minus", .. })
        ));
    }

    /// `Oja` REPORTS DIVERGENCE RATHER THAN RETURNING AN INFINITY. The rule normalises itself,
    /// which is exactly why this path is easy to leave untested — but a learning rate large enough
    /// to overshoot the fixed point outward makes the correction larger than the thing it is
    /// correcting, and the weights leave the finite numbers in a handful of samples.
    #[test]
    fn ojas_rule_reports_divergence_rather_than_returning_an_infinity() {
        let mut o = Oja::new(vec![0.1, 0.2], 1.0).unwrap();
        let mut hit = None;
        for k in 0..1_000u32 {
            if let Err(e) = o.update(&[3.0, 4.0]) {
                hit = Some((k, e));
                break;
            }
        }
        let (k, e) = hit.expect("eta = 1 with |x| = 5 did not diverge in 1000 samples");
        assert!(matches!(e, PlasticityError::Diverged { what: "Oja weight", .. }), "{e}");
        assert!(k < 50, "diverged at sample {k}, which is not the overshoot path");

        // The same input at a sane rate does not diverge: this is a parameter error and is reported
        // as one rather than clamped away.
        let mut o = Oja::new(vec![0.1, 0.2], 1e-3).unwrap();
        for _ in 0..20_000 {
            o.update(&[3.0, 4.0]).unwrap();
        }
        assert!((o.norm() - 1.0).abs() < 1e-6, "norm {}", o.norm());
        // The zero vector is a fixed point and is refused rather than producing a flat curve.
        assert!(matches!(Oja::new(vec![0.0, 0.0, 0.0], 0.1), Err(PlasticityError::Empty { .. })));
        assert!(matches!(Oja::new(vec![], 0.1), Err(PlasticityError::Empty { .. })));
        assert!(matches!(
            Oja::new(vec![0.1, f64::NAN], 0.1),
            Err(PlasticityError::NonFinite { what: "weight", .. })
        ));
        assert!(matches!(
            Oja::new(vec![0.1], -1e-3),
            Err(PlasticityError::Negative { what: "eta", .. })
        ));
    }

    /// THE LINEAR OUTPUT EVERY RATE RULE SHARES, against the dot product written out, plus the two
    /// refusals it makes. Three public `output` methods, none of them called directly by a test.
    #[test]
    fn the_rate_rules_linear_output_is_the_dot_product() {
        let w = vec![0.5, -0.25, 2.0];
        let x = [4.0, 8.0, 0.5];
        let want = 0.5 * 4.0 + (-0.25) * 8.0 + 2.0 * 0.5; // 2 - 2 + 1, all exact in binary
        assert_eq!(want, 1.0);

        let h = Hebbian::new(w.clone(), 0.01, 0.0).unwrap();
        let o = Oja::new(w.clone(), 0.01).unwrap();
        let b = Bcm::new(w.clone(), 1e-6, 1.0, 10.0, 0.0, Bounds::wide()).unwrap();
        assert_eq!(h.output(&x).unwrap(), want);
        assert_eq!(o.output(&x).unwrap(), want);
        assert_eq!(b.output(&x).unwrap(), want);
        // `update` returns the same number it used to drive itself.
        let mut b2 = b.clone();
        assert_eq!(b2.update(&x, 0.01).unwrap(), want);

        // The norm is the Euclidean length, not the sum of the weights.
        let len = (0.25 + 0.0625 + 4.0f64).sqrt();
        assert!((h.norm() - len).abs() < 1e-15, "{} vs {len}", h.norm());
        assert!((o.norm() - len).abs() < 1e-15);
        assert!(
            (h.norm() - w.iter().sum::<f64>()).abs() > 0.1,
            "the norm is the Euclidean length, not the sum: {} vs {}",
            h.norm(),
            w.iter().sum::<f64>()
        );

        // Both refusals, on all three rules.
        for r in [
            h.output(&[1.0, 2.0]),
            o.output(&[1.0, 2.0]),
            b.output(&[1.0, 2.0]),
        ] {
            assert!(matches!(r, Err(PlasticityError::LengthMismatch { got: 2, want: 3, .. })));
        }
        for r in [
            h.output(&[1.0, 2.0, f64::INFINITY]),
            o.output(&[1.0, 2.0, f64::INFINITY]),
            b.output(&[1.0, 2.0, f64::INFINITY]),
        ] {
            assert!(matches!(r, Err(PlasticityError::NonFinite { what: "input", .. })));
        }
    }

    /// `SynapticScaling` REPORTS ITS TOTAL DRIVE AND WHETHER ITS ESTIMATOR HAS SETTLED. Two public
    /// methods with docs making claims — that `total` is what scaling regulates, and that
    /// `is_settled` is a statement about the estimate rather than about the cell — and no test.
    #[test]
    fn synaptic_scaling_reports_the_total_drive_and_whether_its_estimator_has_settled() {
        let bounds = Bounds::new(0.0, 1e6).unwrap();
        let w0 = vec![0.1, 0.25, 0.65];
        let mut s = SynapticScaling::new(w0.clone(), 5.0, 20.0, 1.0, bounds).unwrap();
        assert!((s.total() - 1.0).abs() < 1e-15, "total {}", s.total());

        // Multiplicative means the TOTAL scales by exactly the factor the call reports, which is
        // the sense in which this rule regulates total synaptic drive.
        s.rate = 0.0;
        let g = s.scale(2.0).unwrap();
        assert!((g - (2.0f64 * 5.0 / (5.0 * 20.0)).exp()).abs() < 1e-15, "factor {g}");
        assert!(g > 1.0, "a silent cell must scale UP, got {g}");
        assert!((s.total() - g * 1.0).abs() < 1e-12, "total {} vs {g} times 1.0", s.total());
        // `factor` is the same number without applying it.
        assert_eq!(s.factor(2.0), g);
        // A cell exactly at target neither grows nor shrinks, and the factor is exactly one.
        s.rate = 5.0;
        let before = s.total();
        assert_eq!(s.scale(3.0).unwrap(), 1.0);
        assert_eq!(s.total(), before);

        // Settledness is about the ESTIMATE, and the tolerance is inclusive and symmetric.
        assert!(s.is_settled(0.0), "exactly at target must be settled at any tolerance");
        s.rate = 5.5;
        assert!(!s.is_settled(0.4));
        assert!(s.is_settled(0.5), "the tolerance is inclusive at exactly tol_hz");
        s.rate = 4.5;
        assert!(s.is_settled(0.5), "and symmetric below the target");
        assert!(!s.is_settled(0.4));
        // An empty cell has nothing to scale and is refused at construction.
        assert!(matches!(
            SynapticScaling::new(vec![], 5.0, 1.0, 1.0, bounds),
            Err(PlasticityError::Empty { .. })
        ));
    }

    /// THE `BCM` SELECTIVE FIXED POINT IS THE **AVERAGED** DYNAMICS', AND THE AVERAGING IS AN
    /// ASSUMPTION THE FUNCTION CANNOT CHECK.
    ///
    /// [`Bcm::selective_fixed_point`] answers `n_patterns * y_0` for any `tau_theta` at all,
    /// because the derivation replaced the sliding threshold by its mean over presentations. Run
    /// the same protocol with a threshold that is not fast compared with the weights and the cell
    /// is still selective, still stable, and settles somewhere else entirely — with no error and no
    /// warning. This is the measurement behind the caveat on that function.
    #[test]
    fn the_bcm_selective_fixed_point_holds_only_while_the_threshold_is_the_fast_variable() {
        let settle = |tau_theta: f64| {
            let (n_patterns, drive, y_0) = (4usize, 50.0, 10.0);
            let mut b = Bcm::new(
                vec![0.010, 0.014, 0.011, 0.012],
                2.0e-6,
                tau_theta,
                y_0,
                0.01,
                Bounds::new(0.0, 5.0).unwrap(),
            )
            .unwrap();
            let mut rng = Rng::new(0x0BC0_5EED);
            for _ in 0..200_000 {
                let mut x = vec![0.0; n_patterns];
                x[rng.below(n_patterns as u32) as usize] = drive;
                b.update(&x, 0.01).unwrap();
            }
            (0..n_patterns).map(|k| b.w[k] * drive).collect::<Vec<f64>>()
        };
        let want = 40.0; // n_patterns * y_0

        let fast = settle(2.0);
        assert!(
            (fast[1] - want).abs() / want < 0.05,
            "with a fast threshold the closed form should hold: {fast:?}"
        );

        // 20 s is still four orders of magnitude faster than any biological estimate of the
        // sliding threshold, and the winner lands 29% high.
        let slow = settle(20.0);
        assert!(
            slow[1] > 1.25 * want,
            "a slow threshold did not overshoot the closed form: {slow:?}"
        );
        assert!(slow[1] < 2.0 * want, "and it does not run away either: {slow:?}");

        // Both are still SELECTIVE, and for the same pattern. The closed form is wrong about
        // WHERE the winner lands, not about whether there is one.
        for r in [&fast, &slow] {
            for (k, &resp) in r.iter().enumerate() {
                if k != 1 {
                    assert!(resp < 1e-6, "pattern {k} was not depressed: {r:?}");
                }
            }
        }
    }

    /// `pre_increment` AND `post_increment` PREVIEW THE STEP WITHOUT TAKING IT. Their docs say
    /// they exist so a caller pricing plasticity can see the change before paying for the write,
    /// and so [`RewardStdp`] can put the number into a tag instead of into the weight — but no test
    /// ever called either of them directly, so "the same number" was an assumption rather than a
    /// checked property, on both rules.
    ///
    /// Each preview is pinned against a closed form written out with the fixture's own literals.
    /// The first draft of this test asserted only that `on_pre` applies `w + pre_increment(w)`,
    /// which is a comparison of the function against ITSELF and therefore true under any mutation
    /// of it: dropping the weight dependence, reading the other side's trace, and doubling the
    /// triplet term all survived it. That is finding 3's shape, reproduced here by accident, and
    /// the fix is the same one — compare the number to the model, not to the code that produced it.
    #[test]
    fn the_increment_preview_is_exactly_the_step_the_spike_would_take() {
        let b = Bounds::new(0.0, 1.0).unwrap();
        let w = 0.4;
        for rule in [
            WeightRule::Additive,
            WeightRule::MultiplicativeDepression,
            WeightRule::SoftBound { mu: 0.6 },
        ] {
            let mut s = PairStdp::new(0.05, 0.06, 16.8e-3, 33.7e-3, rule, b).unwrap();
            s.note_pre();
            s.note_post();
            s.advance(7e-3).unwrap();
            let dep = s.pre_increment(w);
            let pot = s.post_increment(w);

            // The closed form, from the fixture's literals. The weight dependence at `w = 0.4` in
            // `[0, 1]`: additive is one either way, multiplicative depression is `w - w_min` on the
            // depressing side only, and Gütig's soft bound is the normalised distance to whichever
            // bound the step approaches, raised to `mu`.
            let (dep_factor, pot_factor) = match rule {
                WeightRule::Additive => (1.0, 1.0),
                WeightRule::MultiplicativeDepression => (0.4f64 - 0.0, 1.0),
                WeightRule::SoftBound { mu } => {
                    (((0.4f64 - 0.0) / 1.0).powf(mu), ((1.0 - 0.4f64) / 1.0).powf(mu))
                }
            };
            // Depression reads the POST trace, decaying with `tau_minus`; potentiation reads the
            // PRE trace, decaying with `tau_plus`. Swapping them is the classic defect and the two
            // time constants differ by a factor of two, so this fixture sees it.
            let want_dep = -(0.06 * (-7e-3f64 / 33.7e-3).exp()) * dep_factor;
            let want_pot = (0.05 * (-7e-3f64 / 16.8e-3).exp()) * pot_factor;
            assert_eq!(dep, want_dep, "{rule:?}: pre_increment vs closed form");
            assert_eq!(pot, want_pot, "{rule:?}: post_increment vs closed form");
            // The sign is the rule's, not the caller's: `a_minus` is stored as a magnitude and the
            // rule applies the sign, so no amplitude a caller supplies can make depression
            // potentiate.
            assert!(dep < 0.0 && pot > 0.0, "{rule:?}: signs {dep} {pot}");

            // And the preview is bit for bit what the call then applies.
            let mut taken = s;
            assert_eq!(taken.on_pre(w).unwrap(), b.clamp(w + want_dep), "{rule:?}: pre");
            let mut taken = s;
            assert_eq!(taken.on_post(w).unwrap(), b.clamp(w + want_pot), "{rule:?}: post");
            // Previewing does not register a spike or move a trace.
            let before = (s.pre_trace.x, s.post_trace.x);
            let _ = s.pre_increment(w);
            let _ = s.post_increment(w);
            assert_eq!((s.pre_trace.x, s.post_trace.x), before, "{rule:?}: a preview fired");
        }

        // The triplet rule's pair, same contract, with both slow traces in play so the previews
        // carry the triplet terms and not only the pair ones. Table 4's all-to-all full row.
        let mut t = TripletStdp::hippocampal_full(b).unwrap();
        t.note_pre();
        t.note_post();
        t.advance(5e-3).unwrap();
        let (dep, pot) = (t.pre_increment(w), t.post_increment(w));
        let want_dep =
            -((-5e-3f64 / 33.7e-3).exp() * (1.6e-3 + 1.4e-3 * (-5e-3f64 / 946e-3).exp()));
        let want_pot = (-5e-3f64 / 16.8e-3).exp() * (6.1e-3 + 6.7e-3 * (-5e-3f64 / 27e-3).exp());
        assert_eq!(dep, want_dep, "triplet pre_increment vs closed form");
        assert_eq!(pot, want_pot, "triplet post_increment vs closed form");
        assert!(dep < 0.0 && pot > 0.0, "triplet previews {dep} {pot}");
        // Both triplet terms are doing real work here, or the closed forms above are the pair
        // rule's wearing four time constants.
        assert!(
            1.4e-3 * (-5e-3f64 / 946e-3).exp() > 0.5 * 1.6e-3,
            "the triplet depression term is negligible in this fixture"
        );
        assert!(
            6.7e-3 * (-5e-3f64 / 27e-3).exp() > 0.5 * 6.1e-3,
            "the triplet potentiation term is negligible in this fixture"
        );
        let mut taken = t;
        assert_eq!(taken.on_pre(w).unwrap(), b.clamp(w + want_dep));
        let mut taken = t;
        assert_eq!(taken.on_post(w).unwrap(), b.clamp(w + want_pot));
        let before = (t.r1.x, t.r2.x, t.o1.x, t.o2.x);
        let _ = t.pre_increment(w);
        let _ = t.post_increment(w);
        assert_eq!((t.r1.x, t.r2.x, t.o1.x, t.o2.x), before, "a triplet preview fired");
    }

    /// EVERY REFUSAL NAMES ITS QUANTITY WHEN PRINTED, and crosses a `Box<dyn Error>` boundary. A
    /// library error that says "invalid input" and nothing else sends its reader back to the
    /// source, which is the opposite of the job the type exists to do.
    #[test]
    fn every_refusal_names_its_quantity_when_printed() {
        let cases: [(PlasticityError, &str); 7] = [
            (PlasticityError::NonFinite { what: "A_plus", value: f64::NAN }, "A_plus"),
            (PlasticityError::NotPositive { what: "tau_c", value: 0.0 }, "tau_c"),
            (PlasticityError::Negative { what: "dt", value: -1.0 }, "dt"),
            (PlasticityError::BoundsInverted { w_min: 5.0, w_max: -5.0 }, "5"),
            (
                PlasticityError::LengthMismatch { what: "input vector", got: 1, want: 2 },
                "input vector",
            ),
            (PlasticityError::Empty { what: "weight vector" }, "weight vector"),
            (PlasticityError::Diverged { what: "Oja weight", value: f64::INFINITY }, "Oja weight"),
        ];
        for (e, needle) in cases {
            let printed = e.to_string();
            assert!(printed.contains(needle), "{printed:?} does not name {needle:?}");
            assert!(!printed.is_empty());
        }
        // The value is printed too, so a reader can tell an infinity from a NaN without a debugger.
        assert!(
            PlasticityError::NonFinite { what: "weight", value: f64::NEG_INFINITY }
                .to_string()
                .contains("inf")
        );
        let boxed: Box<dyn std::error::Error> =
            Box::new(PlasticityError::Empty { what: "weight vector" });
        assert!(boxed.to_string().contains("weight vector"));
    }
}
