//! Neuromorphic olfaction: the olfactory bulb's external plexiform layer, learning an odour from
//! one presentation.
//!
//! # What the mechanism is, and why it exists
//!
//! A nose has a hard problem. Its receptors are broadly tuned — every odorant drives many receptor
//! types and every receptor type responds to many odorants — so the signal arriving at the brain is
//! a dense, heavily overlapping vector across a few hundred to a couple of thousand channels, drifting
//! with concentration, temperature and the age of the sensor, and buried in whatever else is in the
//! air. There is no clean training set: an animal meets an odour once and must recognise it later,
//! in a different mixture, at a different concentration. Backpropagation over a labelled corpus is
//! not on offer.
//!
//! The olfactory bulb's answer is a **recurrent inhibitory loop**. Receptor axons converge on
//! *glomeruli*, one per receptor type; each glomerulus drives a *mitral cell*, the bulb's output
//! neuron. Mitral cells extend long lateral dendrites through the **external plexiform layer**
//! (EPL), where they meet the apical dendrites of *granule cells* — axonless interneurons — at
//! **reciprocal dendrodendritic synapses**. One synaptic site, two directions: the mitral dendrite
//! releases glutamate onto the granule spine, and the granule spine releases GABA straight back onto
//! the same mitral dendrite. Rall and Shepherd identified this arrangement in 1968 (*Journal of
//! Neurophysiology* 31:884–915) from field-potential reconstruction, before anyone could record from
//! the cells. The loop rings: mitral firing recruits granule inhibition, which silences the mitral
//! cells, which releases the granule cells, which stop inhibiting — at a **gamma-band** rate of
//! roughly 40 Hz, the oscillation Adrian recorded from the bulb in 1942 and which every subsequent
//! study of the EPL has had to account for.
//!
//! ⛔ **The gamma rhythm this module produces is imposed, not emergent.** [`EplParams::mitral_duty`]
//! gates the mitral cells to a fixed fraction of each cycle, so the population histogram is
//! periodic by construction — `the_gamma_period_measured_from_the_train_is_the_one_the_parameters_set`
//! measures the identical period, bit-exact, with every granule cell removed. The loop is real and
//! does shape the pattern; the clock is what sets the rhythm. Adrian's 1942 recording is the
//! reason for the band, not something this circuit reproduces.
//!
//! What the loop buys is three things at once:
//!
//! 1. **Activity control.** Broad granule inhibition scales with total mitral activity: over a
//!    five-fold concentration step the broad pool's own output rises from 3 spikes to 13 while the
//!    open-loop mitral total goes from 50 to 95. ⚠ But in this implementation that limb is
//!    **subtractive, not divisive**, so it does not preserve ratios — it removes a near-constant
//!    21 spikes across that five-fold range (0.2x to 1.0x), and the settled ratio (2.52) comes out
//!    *larger* than the open-loop one (1.90). ⛔ Not ten-fold, which this line used to say: from
//!    0.1x the broad pool is below its own rheobase at the low end, removes nothing there, and the
//!    ratio claim inverts (open 2.44, settled 1.87). The subtractive claim needs the pool above its
//!    rheobase at both ends of the range. Divisive normalisation is a glomerular-layer
//!    mechanism (Cleland and colleagues) that this module does not implement, and the phrase "gain
//!    control" is avoided here for that reason. The numbers are from
//!    `the_broad_pool_answers_concentration_and_not_shape`.
//! 2. **Decorrelation.** Two odours whose receptor patterns overlap heavily leave the EPL less
//!    similar than they entered it, because subtracting a common inhibitory term and passing the
//!    remainder through a spiking threshold is an expansive operation on the differences.
//!    ⚠ Read that sentence literally: the broad limb is ONE scalar, `i_broad * a_broad`, applied
//!    identically to every mitral cell (see [`Epl::present`]). There is no cell-to-cell term in
//!    it, so what this measures is a common threshold shift, not lateral inhibition — the same
//!    overlap drop is reachable with no granule cell in the circuit by lowering `i_bias` by a
//!    constant. It is a true property of the circuit and a weaker one than the name suggests.
//! 3. **A place to write a memory.** The dendrodendritic synapse is plastic, so a single
//!    presentation can carve a granule ensemble that is tuned to one odour — and because the
//!    ensemble is inhibitory, recalling the odour means *suppressing everything that is not it*.
//!
//! What it costs is the loop itself: every gamma cycle moves spikes in both directions across a
//! dense dendrodendritic field, and that traffic is the energy bill. [`Epl`] counts it into
//! [`crate::ledger::Ledger`], which then refuses to price it, for the reason that module states.
//!
//! # The work this module implements
//!
//! Nabil Imam and Thomas A. Cleland, **"Rapid online learning and robust recall in a neuromorphic
//! olfactory circuit"**, *Nature Machine Intelligence* 2:181–191 (2020), built this circuit on
//! Intel's `Loihi` and, per its title and abstract, learned odours from single presentations and
//! recalled them under impaired conditions. It is the clearest published case of a spiking network
//! doing something a conventional network finds awkward, which is why it is in this library.
//!
//! ## ⚠ What this implementation has NOT reproduced, stated plainly
//!
//! **This review could not reach the paper's figures.** The source was behind a paywall. So the
//! following are *not* claimed, not transcribed and not checked here:
//!
//! - **The paper's accuracy-versus-baseline numbers.** No figure from it is reproduced. Nothing in
//!   this module should be read as agreeing or disagreeing with one.
//! - **Any energy figure for the work.** This review did not locate a joules-per-inference number
//!   for the `Loihi` implementation that it could check, so none is quoted.
//! - **The paper's own similarity measure.** This module uses [`cosine_similarity`] on per-cycle
//!   mitral spike counts, and [`tanimoto`] is provided beside it; which measure the paper reports
//!   is not known here. A threshold quoted in this module's docs is a threshold *measured in this
//!   module's tests*, on this module's synthetic odours.
//! - **The paper's parameters**: cell counts, membrane constants, gamma frequency, plasticity rule
//!   and its constants, and the chemosensor dataset. Every constant in [`EplParams`] is a default
//!   chosen *here* to make a 40 Hz gamma cycle carry a usable spike count, and it is stated rather
//!   than fitted. Where a value could be wrong, the field doc says so.
//! - **The `Loihi` deployment.** This is a pure-Rust simulation on a clock.
//!
//! What *is* implemented is the circuit and its mechanism, and what is verified is what is
//! internally checkable: the oscillation's period against the parameter that sets it, the granule
//! cell's dendritic sum against the closed-form cosine it is supposed to compute, one-shot recall
//! against a measured false-positive rate, monotone degradation under occlusion, and the fall in
//! pattern overlap the inhibitory loop is supposed to produce.
//!
//! # The circuit, as built here
//!
//! ⚠ **The EPL here is dense: every granule cell contacts every mitral cell.** The real layer is
//! spatially local — a granule cell reaches the lateral dendrites that pass near it — and that
//! locality is most of why the biological circuit is affordable. The dense form is kept because it
//! makes the matched filter exact and the arithmetic checkable, and it means the synaptic-operation
//! counts this module puts in the ledger are an **upper bound** on a sparse implementation, not an
//! estimate of one.
//!
//! One **gamma cycle** of `1 / gamma_hz` seconds is split in two by [`EplParams::mitral_duty`]:
//!
//! - **Mitral phase.** Each mitral cell integrates `i_bias + i_gain * c[m] - inh[m]` amperes, where
//!   `c[m]` is the receptor activation of glomerulus `m` and `inh[m]` is the inhibition computed in
//!   the *previous* cycle. The odour is a **current**, not a spike train: it is sensory transduction,
//!   not a synapse. Both are work the cell does — an event-driven implementation still has to
//!   integrate the sensory current toward threshold — so every mitral update with a non-zero
//!   current is charged as driven, and what [`crate::ledger::Ledger::idle_fraction`] reports over
//!   a run is the granule cells sitting below their thresholds. ⛔ The first version charged a
//!   mitral tick as idle whenever no inhibition arrived, and reported an idle fraction of 1.0 for
//!   a mitral layer that had fired 498 times.
//! - **Granule phase.** Each granule cell forms a dendritic sum over the mitral spike counts `r` of
//!   this cycle, is biased **exactly to its own rheobase** `(v_th - v_rest) / r_m`, and is driven by
//!   `granule_gain * (drive - theta)`. Biasing to rheobase is what makes `theta` mean what its name
//!   says: the drive at which the cell starts to fire.
//!
//! The two granule populations differ in one deliberate way, and getting it wrong is a trap worth
//! naming:
//!
//! - The **broad pool** computes an **unnormalised** drive, `sum(r) / (n_mitral * r_max)`. This limb
//!   is a magnitude feedback loop: more mitral activity must produce more inhibition. Dividing it by
//!   `||r||` — which looks like tidying up — would make it measure the *shape* of the mitral pattern
//!   instead of its size, and the gain control would silently stop existing while every plot still
//!   looked right.
//! - Each **learned ensemble** computes a **normalised** drive, `dot(template, r) / ||r||`, because
//!   it is a matched filter for shape and must not care about concentration. That quantity is
//!   exactly the cosine similarity between the mitral pattern and the stored template.
//!   `granule_drive_equals_the_cosine_by_construction` asserts that identity to 1e-12. ⛔ But
//!   [`Epl::recall`] computes the score **beside** the circuit, from the settled pattern, and does
//!   not read the ensemble's spikes: set every ensemble threshold above 1 so that no ensemble cell
//!   can ever fire and clean recall still identifies every stored odour
//!   (`recall_scores_beside_the_circuit_and_does_not_need_the_ensembles_to_fire`). The dendrite
//!   computes the same quantity; the decision does not consume it.
//!
//! Inhibition delivered to mitral cell `m` next cycle is
//!
//! ```text
//! inh[m] = i_broad * a_broad  +  i_specific * ( sum_k a[k] * mask[k][m] ) / sum_k a[k]
//! ```
//!
//! where `a` is a granule population's normalised activity in `[0, 1]` and `mask[k][m] = 1 -
//! r[m]/max(r)` is the **complement** of the stored template. The complement is the point: an
//! inhibitory ensemble recalls an odour by suppressing the channels that do **not** belong to it.
//! The specific term is an activity-weighted average, so it is bounded by `i_specific` however many
//! odours are stored — and when several ensembles are co-active, which is what interference *is*,
//! the average is a blur of several masks and the cleaning degrades. That is the capacity mechanism,
//! and [`capacity_curve`] measures it.
//!
//! # One-shot learning
//!
//! [`Epl::learn`] presents the odour for [`EplParams::learn_cycles`] gamma cycles — one presentation
//! — and writes the settled mitral spike-count vector as a unit-norm template plus its complementary
//! mask. There is no gradient, no second epoch and no replay. Existing ensembles are left active
//! during learning, so a new odour is stored on top of the representation the loop has already
//! decorrelated.
//!
//! # Units
//!
//! SI at every interface: seconds, amperes, volts, hertz. Receptor activations are **dimensionless
//! in `[0, 1]`** — a normalised sensor reading, not a concentration — and [`EplParams::i_gain`] is
//! the ampere-per-unit conversion at the boundary. Spike counts are dimensionless. Thresholds
//! ([`EplParams::broad_theta_lo`] and friends) are in the dimensionless units of the drive they cut.
//!
//! # Example
//!
//! ```
//! use ferromorphic::olfaction::{Epl, EplParams, OdourGenerator};
//!
//! let mut params = EplParams::default();
//! params.mitral = 48;
//! params.granule_per_odour = 4;
//! params.broad_granule = 4;
//! params.max_odours = 4;
//!
//! let mut source = OdourGenerator::new(9, params.mitral, 0.35)?;
//! let target = source.next_odour()?;
//! let stranger = source.next_odour()?;
//!
//! let mut epl = Epl::new(params)?;
//! let id = epl.learn(&target)?;              // ONE presentation
//! assert_eq!(id, 0);
//!
//! let hit = epl.recall(&target)?;
//! assert_eq!(hit.identified, Some(0));
//!
//! let miss = epl.recall(&stranger)?;
//! assert_eq!(miss.identified, None);         // refuses rather than guessing
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use crate::ledger::Ledger;
use crate::neuron::{Lif, Neuron};
use crate::rng::Rng;
use crate::spike::{Spike, Train};
use core::fmt;

/// What went wrong, named rather than guessed around.
#[derive(Debug, Clone, PartialEq)]
pub enum OlfactionError {
    /// An odour with no channels. There is no width to compare against and no norm to divide by.
    EmptyOdour,
    /// Channel `index` was NaN or infinite. Rejected at the boundary: a non-finite receptor reading
    /// poisons the membrane potential, then the spike count, then every similarity downstream.
    NotFinite {
        /// Zero-based channel index that failed.
        index: usize,
    },
    /// Channel `index` was negative. A receptor activation is an occupancy fraction and cannot be.
    Negative {
        /// Zero-based channel index that failed.
        index: usize,
    },
    /// Every channel was zero, so the odour has no direction and no cosine against anything.
    ZeroOdour,
    /// An odour of the wrong width was handed to a circuit built for a different one.
    Width {
        /// Channels the circuit expects, equal to `EplParams::mitral`.
        expected: usize,
        /// Channels the odour actually carried.
        found: usize,
    },
    /// A parameter is outside the range the circuit can run in.
    Parameter {
        /// Field name as it appears in [`EplParams`].
        field: &'static str,
        /// What the constraint is, in words a caller can act on.
        reason: &'static str,
    },
    /// No more granule ensembles: the circuit is at [`EplParams::max_odours`].
    Full {
        /// The configured limit.
        max: usize,
    },
    /// The final gamma cycle produced no mitral spikes, so there is nothing to store or to score.
    /// Usually means the drive is below rheobase for every channel, or that the inhibition has
    /// grown strong enough to silence the layer.
    Silent,
    /// [`Epl::recall`] was called before anything was learned.
    Untrained,
    /// [`OdourGenerator::next_odour`] drew `tries` all-zero patterns in a row and stopped.
    ///
    /// A bound rather than a retry-forever loop: at a sparsity of `1e-18` the first version never
    /// returned, and [`OdourGenerator::pair_with_overlap`]'s own "fixed retry count rather than an
    /// unbounded loop" guarantee was void one call deep.
    NoActiveChannel {
        /// Draws attempted before giving up.
        tries: usize,
    },
    /// [`OdourGenerator::pair_with_overlap`] could not reach the requested overlap.
    ///
    /// Two independent non-negative patterns already share a floor of similarity; asking for less
    /// than that floor has no answer, and inventing one by allowing negative channels would make
    /// the odours unphysical.
    Unreachable {
        /// The overlap that was asked for.
        requested: f64,
        /// The lowest overlap this pair of draws can express.
        floor: f64,
    },
}

impl fmt::Display for OlfactionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyOdour => write!(f, "odour has no channels"),
            Self::NotFinite { index } => write!(f, "channel {index} is not finite"),
            Self::Negative { index } => write!(f, "channel {index} is negative"),
            Self::ZeroOdour => write!(f, "odour is all zeros and has no direction"),
            Self::Width { expected, found } => {
                write!(f, "odour has {found} channels, circuit expects {expected}")
            }
            Self::Parameter { field, reason } => write!(f, "parameter {field}: {reason}"),
            Self::Full { max } => write!(f, "circuit already holds its maximum of {max} odours"),
            Self::Silent => write!(f, "mitral layer emitted no spikes"),
            Self::Untrained => write!(f, "nothing has been learned yet"),
            Self::NoActiveChannel { tries } => {
                write!(f, "no channel came up active in {tries} draws; the sparsity is too low")
            }
            Self::Unreachable { requested, floor } => {
                write!(f, "overlap {requested} is below this pair's floor of {floor}")
            }
        }
    }
}

impl std::error::Error for OlfactionError {}

/// Cosine similarity of two equal-length vectors, or `None` when it does not exist.
///
/// `None` for differing lengths, for a zero vector (no direction), and for any non-finite entry.
/// Returning `0.0` for a zero vector — the common shortcut — would report "completely dissimilar"
/// for a case where the question has no answer, and a sweep would plot it as data.
///
/// For non-negative vectors the result is in `[0, 1]`.
#[must_use]
pub fn cosine_similarity(a: &[f64], b: &[f64]) -> Option<f64> {
    if a.len() != b.len() || a.is_empty() {
        return None;
    }
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for k in 0..a.len() {
        if !a[k].is_finite() || !b[k].is_finite() {
            return None;
        }
        dot += a[k] * b[k];
        na += a[k] * a[k];
        nb += b[k] * b[k];
    }
    if na <= 0.0 || nb <= 0.0 {
        return None;
    }
    Some(dot / (na.sqrt() * nb.sqrt()))
}

/// Tanimoto (extended Jaccard) coefficient, `dot / (|a|^2 + |b|^2 - dot)`.
///
/// The measure chemoinformatics uses for molecular fingerprints, given here because an olfaction
/// result reported under one similarity measure is not comparable with one reported under another,
/// and both appear in this literature. For binary vectors it reduces exactly to the Jaccard index
/// `|A ∩ B| / |A ∪ B|`, which `tanimoto_reduces_to_jaccard_on_binary_vectors` checks.
///
/// `None` on the same conditions as [`cosine_similarity`], plus a vanishing denominator.
#[must_use]
pub fn tanimoto(a: &[f64], b: &[f64]) -> Option<f64> {
    if a.len() != b.len() || a.is_empty() {
        return None;
    }
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for k in 0..a.len() {
        if !a[k].is_finite() || !b[k].is_finite() {
            return None;
        }
        dot += a[k] * b[k];
        na += a[k] * a[k];
        nb += b[k] * b[k];
    }
    let den = na + nb - dot;
    if den.abs() <= 0.0 {
        return None;
    }
    Some(dot / den)
}

/// A receptor activation pattern: one dimensionless value in `[0, 1]` per glomerulus.
///
/// Validated on construction so that the circuit never sees a NaN, an infinity or a negative
/// occupancy. Values above 1 are allowed — a saturating sensor reading is a real thing — but the
/// default gain in [`EplParams`] is scaled for `[0, 1]`.
#[derive(Debug, Clone, PartialEq)]
pub struct Odour {
    channels: Vec<f64>,
}

impl Odour {
    /// Build from receptor activations, rejecting anything the circuit cannot integrate.
    ///
    /// # Errors
    ///
    /// [`OlfactionError::EmptyOdour`], [`OlfactionError::NotFinite`], [`OlfactionError::Negative`]
    /// or [`OlfactionError::ZeroOdour`].
    pub fn new(channels: Vec<f64>) -> Result<Self, OlfactionError> {
        if channels.is_empty() {
            return Err(OlfactionError::EmptyOdour);
        }
        let mut total = 0.0;
        for (index, &c) in channels.iter().enumerate() {
            if !c.is_finite() {
                return Err(OlfactionError::NotFinite { index });
            }
            if c < 0.0 {
                return Err(OlfactionError::Negative { index });
            }
            total += c;
        }
        if total <= 0.0 {
            return Err(OlfactionError::ZeroOdour);
        }
        Ok(Self { channels })
    }

    /// The receptor activations, one per glomerulus.
    #[must_use]
    pub fn channels(&self) -> &[f64] {
        &self.channels
    }

    /// Number of glomeruli, which must equal [`EplParams::mitral`] of the circuit it is fed to.
    #[must_use]
    pub fn len(&self) -> usize {
        self.channels.len()
    }

    /// Always `false`: an [`Odour`] with no channels cannot be constructed.
    ///
    /// Present because a type with `len` and no `is_empty` is a trap for a caller who assumes one,
    /// and the honest answer here is a constant.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.channels.is_empty()
    }
}

/// How an odour is corrupted before recall.
///
/// Kept as an enum rather than three functions so a sweep can hold the model fixed and vary only
/// its parameter, which is what [`capacity_curve`] and the occlusion tests do.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Occlusion {
    /// A fraction of channels driven to exactly zero: sensor dropout, or a nose held shut.
    ///
    /// Inhibition cannot repair this — nothing in the EPL adds excitation to a silent channel — so
    /// the measured degradation under `Dropout` is the architecture's floor, not its claim.
    Dropout {
        /// Fraction of channels zeroed, in `[0, 1]`. Rounded to a whole number of channels.
        fraction: f64,
    },
    /// A fraction of channels overwritten by another odour's values: an interfering smell.
    ///
    /// This is the case the EPL is built for. The learned ensemble's complementary mask suppresses
    /// the channels that do not belong to the target, so contamination *outside* the target's
    /// support is removed and contamination inside it is not.
    Interferent {
        /// Fraction of channels replaced by the background odour's values, in `[0, 1]`.
        fraction: f64,
    },
    /// Every channel perturbed by a uniform draw in `[-amplitude, +amplitude]`, then rectified.
    Noise {
        /// Half-width of the uniform perturbation, in the same dimensionless units as a channel.
        amplitude: f64,
    },
}

impl Occlusion {
    /// Apply the corruption, drawing from `rng`.
    ///
    /// `background` is read only by [`Occlusion::Interferent`] and ignored by the others; it is a
    /// required argument so that a sweep over occlusion models needs one call site rather than a
    /// match.
    ///
    /// # Errors
    ///
    /// [`OlfactionError::Width`] if the two odours differ in width, [`OlfactionError::Parameter`]
    /// for a fraction outside `[0, 1]` or a non-finite amplitude, and [`OlfactionError::ZeroOdour`]
    /// if the corruption leaves nothing at all.
    pub fn apply(
        &self,
        target: &Odour,
        background: &Odour,
        rng: &mut Rng,
    ) -> Result<Odour, OlfactionError> {
        if target.len() != background.len() {
            return Err(OlfactionError::Width {
                expected: target.len(),
                found: background.len(),
            });
        }
        let n = target.len();
        let mut out = target.channels.clone();
        match *self {
            Self::Dropout { fraction } | Self::Interferent { fraction } => {
                if !(fraction >= 0.0) || fraction > 1.0 {
                    return Err(OlfactionError::Parameter {
                        field: "fraction",
                        reason: "must be finite and in [0, 1]",
                    });
                }
                let k = (fraction * n as f64).round() as usize;
                let mut idx: Vec<usize> = (0..n).collect();
                // Fisher-Yates from the seeded stream: which channels are hit must be reproducible.
                for i in (1..n).rev() {
                    let j = rng.below((i + 1) as u32) as usize;
                    idx.swap(i, j);
                }
                for &i in idx.iter().take(k.min(n)) {
                    out[i] = match *self {
                        Self::Interferent { .. } => background.channels[i],
                        _ => 0.0,
                    };
                }
            }
            Self::Noise { amplitude } => {
                if !(amplitude >= 0.0) || !amplitude.is_finite() {
                    return Err(OlfactionError::Parameter {
                        field: "amplitude",
                        reason: "must be finite and non-negative",
                    });
                }
                for v in &mut out {
                    *v = (*v + amplitude * (2.0 * rng.next_f64() - 1.0)).max(0.0);
                }
            }
        }
        Odour::new(out)
    }
}

/// Synthetic odours with controllable overlap.
///
/// Each channel is active with probability `sparsity` and, when active, takes a uniform draw in
/// `[0.2, 1.0]`. The floor of 0.2 is deliberate: an "active" channel at 0.001 is indistinguishable
/// from an inactive one after the mitral threshold, and a generator that produced them would make
/// the requested sparsity a fiction.
#[derive(Debug, Clone)]
pub struct OdourGenerator {
    rng: Rng,
    n: usize,
    sparsity: f64,
}

impl OdourGenerator {
    /// A generator of `n`-channel odours with the given active fraction.
    ///
    /// # Errors
    ///
    /// [`OlfactionError::Parameter`] for `n == 0` or a sparsity outside `(0, 1]`.
    pub fn new(seed: u64, n: usize, sparsity: f64) -> Result<Self, OlfactionError> {
        if n == 0 {
            return Err(OlfactionError::Parameter {
                field: "n",
                reason: "needs at least one channel",
            });
        }
        if !(sparsity > 0.0) || sparsity > 1.0 {
            return Err(OlfactionError::Parameter {
                field: "sparsity",
                reason: "must be finite and in (0, 1]",
            });
        }
        Ok(Self { rng: Rng::new(seed), n, sparsity })
    }

    /// Draw one odour.
    ///
    /// Retries until at least one channel is active, up to [`OdourGenerator::MAX_TRIES`] draws.
    /// For any sensible sparsity the first draw succeeds; the bound exists because the constructor
    /// accepts any sparsity in `(0, 1]`, and at `1e-18` an unbounded loop is a hang.
    ///
    /// # Errors
    ///
    /// [`OlfactionError::NoActiveChannel`] after [`OdourGenerator::MAX_TRIES`] all-zero draws.
    pub fn next_odour(&mut self) -> Result<Odour, OlfactionError> {
        for _ in 0..Self::MAX_TRIES {
            let mut c = vec![0.0; self.n];
            for v in &mut c {
                if self.rng.next_f64() < self.sparsity {
                    *v = 0.2 + 0.8 * self.rng.next_f64();
                }
            }
            if let Ok(o) = Odour::new(c) {
                return Ok(o);
            }
        }
        Err(OlfactionError::NoActiveChannel { tries: Self::MAX_TRIES })
    }

    /// Draws [`OdourGenerator::next_odour`] makes before refusing: 4096. At that many all-zero
    /// draws the expected number of active channels per draw is below `1/4096`, which is not an
    /// odour generator, it is a coin that never comes up.
    pub const MAX_TRIES: usize = 4096;

    /// `k` independent odours.
    ///
    /// # Errors
    ///
    /// As [`OdourGenerator::next_odour`].
    pub fn library(&mut self, k: usize) -> Result<Vec<Odour>, OlfactionError> {
        (0..k).map(|_| self.next_odour()).collect()
    }

    /// A pair whose cosine overlap is `target`, plus the overlap actually achieved.
    ///
    /// Draws two independent odours `x` and `y`, then returns `(x, mix)` with
    /// `mix = (1 - lam) * y + lam * x` for the `lam` found by bisection. The overlap is monotone
    /// non-decreasing in `lam` for non-negative vectors, running from `cos(x, y)` at `lam = 0` to
    /// exactly 1 at `lam = 1`, which is what makes bisection sound here.
    ///
    /// **The achieved overlap is returned rather than assumed.** Bisection on a function this
    /// module has not proved monotone could in principle stall, and a caller that wants the
    /// requested value to hold must read the third element — which is why it is not thrown away.
    ///
    /// Two independent non-negative draws share a floor of similarity that varies from draw to
    /// draw, so up to 64 pairs are drawn looking for one whose floor is below `target`. If none is,
    /// the call refuses and reports the **lowest** floor it saw, which is the number a caller needs
    /// to pick a reachable target — a fixed retry count rather than an unbounded loop, because an
    /// impossible target must end in a refusal and not in a hang.
    ///
    /// # Errors
    ///
    /// [`OlfactionError::Unreachable`] when no draw got below `target`, and
    /// [`OlfactionError::Parameter`] for a target outside `[0, 1]`.
    pub fn pair_with_overlap(
        &mut self,
        target: f64,
    ) -> Result<(Odour, Odour, f64), OlfactionError> {
        if !(target >= 0.0) || target > 1.0 {
            return Err(OlfactionError::Parameter {
                field: "target",
                reason: "must be finite and in [0, 1]",
            });
        }
        let mut lowest = f64::INFINITY;
        let mut pair = None;
        for _ in 0..64 {
            let x = self.next_odour()?;
            let y = self.next_odour()?;
            let floor =
                cosine_similarity(x.channels(), y.channels()).ok_or(OlfactionError::ZeroOdour)?;
            if floor < lowest {
                lowest = floor;
            }
            if floor <= target {
                pair = Some((x, y));
                break;
            }
        }
        let Some((x, y)) = pair else {
            return Err(OlfactionError::Unreachable {
                requested: target,
                floor: lowest,
            });
        };
        let mix = |lam: f64| -> Vec<f64> {
            (0..x.len())
                .map(|i| (1.0 - lam) * y.channels()[i] + lam * x.channels()[i])
                .collect()
        };
        let (mut lo, mut hi) = (0.0f64, 1.0f64);
        // 60 halvings takes the bracket below 1e-18, far under f64 resolution on [0, 1]: the loop
        // is bounded by construction rather than by a tolerance that might never be met.
        for _ in 0..60 {
            let mid = 0.5 * (lo + hi);
            let c = cosine_similarity(x.channels(), &mix(mid)).unwrap_or(1.0);
            if c < target { lo = mid; } else { hi = mid; }
        }
        let best = mix(0.5 * (lo + hi));
        let got = cosine_similarity(x.channels(), &best).ok_or(OlfactionError::ZeroOdour)?;
        Ok((x, Odour::new(best)?, got))
    }
}

/// One learned odour: a granule ensemble's stored weights.
///
/// The two vectors are the two limbs of the reciprocal dendrodendritic synapse, and they are *not*
/// the same numbers. The excitatory limb (mitral to granule) is the template, so the ensemble
/// recognises. The inhibitory limb (granule to mitral) is its complement, so the ensemble recalls
/// by suppressing what does not belong. Storing one and negating it would give a circuit that
/// silenced exactly the odour it recognised.
#[derive(Debug, Clone, PartialEq)]
pub struct Ensemble {
    /// Unit-norm mitral spike-count pattern from the single learning presentation. Dimensionless.
    pub template: Vec<f64>,
    /// Complementary inhibitory mask, `1 - r[m] / max(r)`, in `[0, 1]`: 0 for the most active
    /// mitral cell of the odour, 1 for one that was silent.
    pub mask: Vec<f64>,
    /// Per-cell firing thresholds on the dimensionless drive, spread over
    /// `[EplParams::ensemble_theta_lo, EplParams::ensemble_theta_hi]`, which gives the ensemble a
    /// graded population response to match strength instead of a single all-or-nothing cell.
    pub thresholds: Vec<f64>,
}

/// Parameters of the EPL circuit. Every default is chosen here, not transcribed from the paper.
#[derive(Debug, Clone, PartialEq)]
pub struct EplParams {
    /// Mitral cells, one per glomerulus, and therefore the odour width. The mammalian bulb has
    /// roughly 1000–2000 glomeruli; 64 is a default that runs in a test.
    pub mitral: usize,
    /// Granule cells recruited per learned odour. One would work; several give the ensemble a
    /// graded response through the threshold spread.
    pub granule_per_odour: usize,
    /// Granule cells in the untrained broad pool that provides gain control and decorrelation.
    ///
    /// Zero is legal and disables the broad pool entirely; with the specific term also at zero the
    /// mitral layer runs open-loop, which is the configuration the drive-map test uses to compare
    /// the circuit against a bare [`crate::neuron::Lif`].
    pub broad_granule: usize,
    /// Ensembles the circuit will allocate before [`Epl::learn`] refuses.
    pub max_odours: usize,
    /// Simulation time step, seconds. Must divide the gamma period into at least four ticks.
    pub dt: f64,
    /// Gamma oscillation frequency, hertz. ~40 Hz is the mammalian bulb's band (Adrian, 1942); the
    /// frequency the paper used is not known here.
    pub gamma_hz: f64,
    /// Fraction of each gamma cycle in which mitral cells are free to fire, in `(0, 1)`. The
    /// remainder is the granule phase.
    pub mitral_duty: f64,
    /// Mitral cell model. Its `t_ref` must be positive: the broad pool's normaliser divides by the
    /// maximum spikes per window, which `t_ref` sets.
    pub mitral_cell: Lif,
    /// Granule cell model. Biased to its own rheobase at run time, so `v_th > v_rest` and
    /// `r_m > 0` are required.
    pub granule_cell: Lif,
    /// Constant mitral drive, amperes, before the odour. Sets the resting spike count and therefore
    /// how much common mode the broad pool has to remove; at zero the circuit is sparser and the
    /// broad pool has almost nothing to do.
    pub i_bias: f64,
    /// Amperes per unit of receptor activation: the dimensionless-to-SI conversion at the boundary.
    pub i_gain: f64,
    /// Peak broad inhibition, amperes. This is the gain-control and decorrelation term.
    pub i_broad: f64,
    /// Peak odour-specific inhibition, amperes, bounded however many ensembles are co-active
    /// because the specific term is an activity-weighted average.
    pub i_specific: f64,
    /// Amperes per unit of granule drive above threshold.
    pub granule_gain: f64,
    /// Lowest broad-pool threshold, in units of mean mitral rate normalised by the maximum.
    pub broad_theta_lo: f64,
    /// Highest broad-pool threshold, same units. The loop settles with the mean rate near this
    /// band, so lowering it sparsifies the mitral layer.
    pub broad_theta_hi: f64,
    /// Lowest ensemble threshold, in cosine units: an ensemble cell at this threshold fires for any
    /// mitral pattern whose cosine with its template exceeds it.
    pub ensemble_theta_lo: f64,
    /// Highest ensemble threshold, cosine units.
    pub ensemble_theta_hi: f64,
    /// Gamma cycles in the single learning presentation. "One-shot" is about presentations, not
    /// cycles: a sniff lasts several gamma cycles in any animal.
    pub learn_cycles: usize,
    /// Gamma cycles allowed for the recall loop to settle.
    pub recall_cycles: usize,
    /// Cosine at or above which [`Epl::recall`] reports an identification.
    ///
    /// **The default 0.85 is measured in this module's tests, not taken from the paper**, and it is
    /// a property of the layer's WIDTH rather than of the circuit: two random patterns in 64
    /// dimensions overlap less than two in 32, so the same threshold buys different safety. On the
    /// 32-cell layer the tests use, `one_shot_learning_recalls_and_the_false_positive_rate_is_measured`
    /// finds a false-positive rate of exactly zero at 0.85 over 120 unlearned odours; the review's
    /// wider sweep of that layer (60 seeds × 120 unlearned odours at sparsity 0.35) found unlearned
    /// scores as high as **0.8465** — a margin of 0.0035 under the threshold, which is thin. ⛔ An
    /// earlier version of this doc quoted three numbers from a sweep nobody could reproduce; they
    /// are gone. Re-measure with [`Epl::acceptance_rate`] on your own odours before trusting the
    /// default; a threshold quoted without a measured false-positive rate is an unfalsifiable number.
    pub recall_threshold: f64,
}

impl Default for EplParams {
    fn default() -> Self {
        // Faster than a textbook cortical cell (20 ms) so that a 12.5 ms gamma window carries a
        // spike count with useful resolution — roughly 1 to 8 spikes across the input range. This
        // is a modelling choice made HERE for code resolution, not a fit to mitral-cell recordings,
        // and a reader who needs biophysical fidelity should replace it and re-measure everything
        // downstream.
        let mitral_cell = Lif { tau_m: 4e-3, t_ref: 0.5e-3, ..Lif::default() };
        let granule_cell = Lif { tau_m: 8e-3, t_ref: 1e-3, ..Lif::default() };
        Self {
            mitral: 64,
            granule_per_odour: 8,
            broad_granule: 8,
            max_odours: 32,
            dt: 1e-4,
            gamma_hz: 40.0,
            mitral_duty: 0.5,
            mitral_cell,
            granule_cell,
            i_bias: 1.6e-9,
            i_gain: 6.0e-9,
            i_broad: 3.0e-9,
            i_specific: 2.0e-9,
            granule_gain: 12.0e-9,
            broad_theta_lo: 0.02,
            broad_theta_hi: 0.10,
            ensemble_theta_lo: 0.30,
            ensemble_theta_hi: 0.75,
            learn_cycles: 4,
            recall_cycles: 8,
            recall_threshold: 0.85,
        }
    }
}

/// Evenly spaced thresholds, with the one-cell case answered rather than divided by zero.
fn spread(lo: f64, hi: f64, n: usize) -> Vec<f64> {
    match n {
        0 => Vec::new(),
        1 => vec![lo],
        _ => (0..n)
            .map(|j| lo + (hi - lo) * (j as f64) / ((n - 1) as f64))
            .collect(),
    }
}

/// What one presentation produced.
#[derive(Debug, Clone, PartialEq)]
pub struct Presentation {
    /// Inhibition each mitral cell integrated during the **final** cycle, amperes — what the loop
    /// had settled on when the counts in `mitral` were produced. `i_broad * a_broad` plus
    /// `i_specific` times the activity-weighted mask average; zero throughout the first cycle.
    ///
    /// Exposed because it is the only direct readout of which stored mask the specific limb
    /// delivered: spike counts saturate at the resting spike and cannot tell one mask from another.
    pub inhibition: Vec<f64>,
    /// Mitral spike counts in the final gamma cycle: the settled representation.
    pub mitral: Vec<f64>,
    /// Mitral spike counts in the **first** cycle, before any inhibition computed from this
    /// presentation has arrived. The open-loop response, kept so that the loop's effect can be
    /// measured against the same input rather than against a different run.
    pub mitral_open_loop: Vec<f64>,
    /// Each ensemble's normalised dendritic drive in the final cycle, which equals the cosine
    /// similarity between [`Presentation::mitral`] and that ensemble's template.
    pub granule_drive: Vec<f64>,
    /// Every spike, mitral and granule. Addresses: `0..mitral` are mitral cells, then
    /// `broad_granule` broad cells, then `granule_per_odour` cells per learned ensemble in
    /// learning order.
    pub train: Train,
    /// Gamma cycles run.
    pub cycles: usize,
}

/// The outcome of a recall attempt.
#[derive(Debug, Clone, PartialEq)]
pub struct Recall {
    /// The identified odour, or `None` when the best score fell below
    /// [`EplParams::recall_threshold`]. A refusal, not a nearest neighbour.
    ///
    /// The threshold is for CLEAN recall. Under occlusion the score falls below it well before the
    /// best match becomes the wrong one, so an occluded probe is better judged by the argmax over
    /// [`Recall::scores`] — which is what [`capacity_curve`] scores and why it does not use this
    /// field.
    pub identified: Option<usize>,
    /// Cosine between the settled mitral pattern and the best-matching template.
    pub score: f64,
    /// The second-best template's cosine, or 0 when only one odour is stored.
    pub runner_up: f64,
    /// `score - runner_up`: how much room the decision had. This is the quantity that shrinks as
    /// odours accumulate, well before the identification itself becomes wrong.
    pub margin: f64,
    /// Cosine against every stored template, in learning order.
    pub scores: Vec<f64>,
    /// The settled mitral spike-count pattern the scores were computed from.
    pub mitral: Vec<f64>,
}

/// The external plexiform layer: mitral cells, granule cells and the loop between them.
#[derive(Debug, Clone)]
pub struct Epl {
    params: EplParams,
    ensembles: Vec<Ensemble>,
    broad_thresholds: Vec<f64>,
    ticks_per_cycle: usize,
    mitral_ticks: usize,
    /// Counts of everything the circuit did, for [`crate::ledger::Prices`] to refuse to price.
    pub ledger: Ledger,
}

impl Epl {
    /// Build the circuit, or refuse and name the parameter that is wrong.
    ///
    /// # Errors
    ///
    /// [`OlfactionError::Parameter`] naming the first field that fails. The checks that matter:
    /// the gamma period must hold at least four ticks and both phases at least one; the mitral
    /// cell's refractory period must be positive (the broad pool divides by the maximum rate it
    /// implies); the granule cell must have `v_th > v_rest` and `r_m > 0` so its rheobase exists;
    /// every current and threshold must be finite.
    pub fn new(params: EplParams) -> Result<Self, OlfactionError> {
        let p = &params;
        let bad = |field, reason| OlfactionError::Parameter { field, reason };
        if p.mitral == 0 {
            return Err(bad("mitral", "needs at least one cell"));
        }
        if p.granule_per_odour == 0 {
            return Err(bad("granule_per_odour", "needs at least one cell"));
        }
        if !(p.dt > 0.0) || !p.dt.is_finite() {
            return Err(bad("dt", "must be finite and positive"));
        }
        if !(p.gamma_hz > 0.0) || !p.gamma_hz.is_finite() {
            return Err(bad("gamma_hz", "must be finite and positive"));
        }
        if !(p.mitral_duty > 0.0) || p.mitral_duty >= 1.0 {
            return Err(bad("mitral_duty", "must be finite and in (0, 1)"));
        }
        for (field, v) in [
            ("i_bias", p.i_bias),
            ("i_gain", p.i_gain),
            ("i_broad", p.i_broad),
            ("i_specific", p.i_specific),
            ("granule_gain", p.granule_gain),
            ("broad_theta_lo", p.broad_theta_lo),
            ("broad_theta_hi", p.broad_theta_hi),
            ("ensemble_theta_lo", p.ensemble_theta_lo),
            ("ensemble_theta_hi", p.ensemble_theta_hi),
            ("recall_threshold", p.recall_threshold),
        ] {
            if !v.is_finite() {
                return Err(bad(field, "must be finite"));
            }
        }
        if p.broad_theta_hi < p.broad_theta_lo {
            return Err(bad("broad_theta_hi", "must not be below broad_theta_lo"));
        }
        if p.ensemble_theta_hi < p.ensemble_theta_lo {
            return Err(bad(
                "ensemble_theta_hi",
                "must not be below ensemble_theta_lo",
            ));
        }
        if !(p.mitral_cell.t_ref > 0.0) {
            return Err(bad(
                "mitral_cell.t_ref",
                "must be positive; the broad pool normalises by the rate it implies",
            ));
        }
        // ⛔ The mitral cell gets the same check as the granule cell. With `v_th` below `v_rest`
        // every mitral cell saturates identically, the template becomes a constant vector, and the
        // circuit identifies every odour in the world as odour 0 at a cosine of 0.9999999999999996:
        // a 100% false-positive rate from a parameter set the first version accepted.
        if !(p.mitral_cell.tau_m > 0.0)
            || !p.mitral_cell.tau_m.is_finite()
            || !(p.mitral_cell.r_m > 0.0)
            || !p.mitral_cell.r_m.is_finite()
            || !(p.mitral_cell.v_th > p.mitral_cell.v_rest)
        {
            return Err(bad(
                "mitral_cell",
                "needs finite positive tau_m and r_m, and v_th > v_rest",
            ));
        }
        if !(p.granule_cell.r_m > 0.0) || !(p.granule_cell.v_th > p.granule_cell.v_rest) {
            return Err(bad(
                "granule_cell",
                "needs r_m > 0 and v_th > v_rest for a rheobase to exist",
            ));
        }
        if p.learn_cycles == 0 || p.recall_cycles == 0 {
            return Err(bad("learn_cycles", "both cycle counts must be at least one"));
        }
        let ticks = (1.0 / (p.gamma_hz * p.dt)).round();
        if !(ticks >= 4.0) || ticks > 1e7 {
            return Err(bad(
                "gamma_hz",
                "the gamma period must span between 4 and 1e7 time steps",
            ));
        }
        let ticks_per_cycle = ticks as usize;
        let mitral_ticks = (ticks * p.mitral_duty).round() as usize;
        if mitral_ticks == 0 || mitral_ticks >= ticks_per_cycle {
            return Err(bad(
                "mitral_duty",
                "must leave at least one time step in each phase",
            ));
        }
        let broad_thresholds = spread(p.broad_theta_lo, p.broad_theta_hi, p.broad_granule);
        Ok(Self {
            params,
            ensembles: Vec::new(),
            broad_thresholds,
            ticks_per_cycle,
            mitral_ticks,
            ledger: Ledger::default(),
        })
    }

    /// The parameters this circuit was built with.
    #[must_use]
    pub fn params(&self) -> &EplParams {
        &self.params
    }

    /// The learned ensembles, in learning order.
    #[must_use]
    pub fn ensembles(&self) -> &[Ensemble] {
        &self.ensembles
    }

    /// Time steps in one gamma cycle, `round(1 / (gamma_hz * dt))`.
    ///
    /// The rounding is why [`Epl::gamma_period`] is not exactly `1 / gamma_hz` for every parameter
    /// choice, and why the tests compare a measured period against this rather than against the
    /// requested frequency.
    #[must_use]
    pub fn ticks_per_cycle(&self) -> usize {
        self.ticks_per_cycle
    }

    /// The gamma period the circuit actually runs at, seconds: `ticks_per_cycle * dt`.
    #[must_use]
    pub fn gamma_period(&self) -> f64 {
        self.ticks_per_cycle as f64 * self.params.dt
    }

    /// Refractory-limited upper bound on the spikes one mitral cell can emit in one mitral phase:
    /// `floor(window / t_ref)`, at least 1. This is the broad pool's normaliser.
    ///
    /// It is an UPPER BOUND, not the achieved maximum. A refractory countdown subtracts `dt` from a
    /// float, and for the default 0.5 ms period at a 0.1 ms step the remainder can land a hair
    /// above zero, costing one extra time step per spike — a saturated default mitral cell emits
    /// 18 spikes in a window this bound puts at 25.
    #[must_use]
    pub fn mitral_ceiling(&self) -> f64 {
        let window = self.mitral_ticks as f64 * self.params.dt;
        (window / self.params.mitral_cell.t_ref).floor().max(1.0)
    }

    /// The same bound for a granule cell over the granule phase, used to normalise ensemble and
    /// broad-pool activity into `[0, 1]` before it is scaled into amperes.
    ///
    /// Falls back to the tick count when `t_ref` is zero, which is a legal granule cell even though
    /// it is not a legal mitral one — only the mitral limb's normaliser needs a finite rate.
    #[must_use]
    pub fn granule_ceiling(&self) -> f64 {
        let ticks = self.ticks_per_cycle - self.mitral_ticks;
        let window = ticks as f64 * self.params.dt;
        if self.params.granule_cell.t_ref > 0.0 {
            (window / self.params.granule_cell.t_ref).floor().max(1.0)
        } else {
            ticks as f64
        }
    }

    /// The current a mitral cell integrates, amperes: `i_bias + i_gain * activation - inhibition`.
    ///
    /// Public because it is the only place dimensionless receptor activation becomes SI, and a
    /// reader checking this circuit against another implementation needs to see the conversion
    /// rather than infer it from a firing rate. [`Epl::present`] uses this exact expression, which
    /// `the_circuit_uses_the_drive_map_it_documents` asserts by driving a bare
    /// [`crate::neuron::Lif`] with it and comparing spike counts.
    #[must_use]
    pub fn mitral_current(&self, activation: f64, inhibition: f64) -> f64 {
        self.params.i_bias + self.params.i_gain * activation - inhibition
    }

    /// Run the circuit on an odour for `cycles` gamma cycles.
    ///
    /// # Errors
    ///
    /// [`OlfactionError::Width`] for an odour of the wrong width, [`OlfactionError::Parameter`] for
    /// `cycles == 0`, and [`OlfactionError::Silent`] if the final cycle produced no mitral spikes.
    pub fn present(&mut self, odour: &Odour, cycles: usize) -> Result<Presentation, OlfactionError> {
        if odour.len() != self.params.mitral {
            return Err(OlfactionError::Width {
                expected: self.params.mitral,
                found: odour.len(),
            });
        }
        if cycles == 0 {
            return Err(OlfactionError::Parameter {
                field: "cycles",
                reason: "must be at least one",
            });
        }
        let n = self.params.mitral;
        let n_broad = self.params.broad_granule;
        let n_ens = self.ensembles.len();
        let per = self.params.granule_per_odour;
        let dt = self.params.dt;
        let granule_ticks = self.ticks_per_cycle - self.mitral_ticks;
        let ceiling = self.mitral_ceiling();
        // A granule cell's own rheobase: the current at which its steady state just touches
        // threshold. Biasing to it is what makes `theta` a threshold on the DRIVE.
        let g = &self.params.granule_cell;
        let rheobase = (g.v_th - g.v_rest) / g.r_m;
        let granule_ceiling = self.granule_ceiling();

        let mut mitral = vec![self.params.mitral_cell; n];
        let mut broad = vec![*g; n_broad];
        let mut ens_cells = vec![*g; n_ens * per];
        let mut inh = vec![0.0f64; n];
        let mut counts = vec![0.0f64; n];
        let mut open_loop = vec![0.0f64; n];
        let mut drive = vec![0.0f64; n_ens];
        let mut train = Train::new();
        let mut t: u64 = 0;
        let granule_base = n as u32;

        let mut delivered = vec![0.0f64; n];
        for cycle in 0..cycles {
            for cell in &mut mitral {
                cell.reset();
            }
            delivered.copy_from_slice(&inh);
            counts.fill(0.0);
            for _ in 0..self.mitral_ticks {
                for m in 0..n {
                    let i = self.mitral_current(odour.channels[m], inh[m]);
                    // A cell integrating a non-zero current toward threshold is doing work an
                    // event-driven implementation cannot skip, whether the current is the odour
                    // or the inhibition. The first version charged "idle" whenever no inhibition
                    // had arrived, and reported an idle fraction of 1.0 for a layer that fired 498
                    // times — see the module doc.
                    if i != 0.0 {
                        self.ledger.neuron_updates_driven += 1;
                    } else {
                        self.ledger.neuron_updates_idle += 1;
                    }
                    if mitral[m].step(dt, i) {
                        counts[m] += 1.0;
                        train.push(Spike { t, source: m as u32 });
                        self.ledger.spikes_out += 1;
                    }
                }
                t += 1;
            }
            if cycle == 0 {
                open_loop.copy_from_slice(&counts);
            }

            // Dendritic sums. Every mitral spike reaches every granule cell in this dense EPL, so
            // the SOP count is an upper bound on a sparse implementation, which is what the real
            // layer is.
            let total: f64 = counts.iter().sum();
            let norm = counts.iter().map(|x| x * x).sum::<f64>().sqrt();
            let n_granule = (n_broad + n_ens * per) as u64;
            self.ledger.syn_ops += (total as u64) * n_granule;
            self.ledger.syn_fetches += (total as u64) * n_granule;
            let broad_drive = total / (n as f64 * ceiling);
            for (k, e) in self.ensembles.iter().enumerate() {
                drive[k] = if norm > 0.0 {
                    (0..n).map(|m| e.template[m] * counts[m]).sum::<f64>() / norm
                } else {
                    0.0
                };
            }

            // Granule phase.
            let mut broad_spikes = vec![0.0f64; n_broad];
            let mut ens_spikes = vec![0.0f64; n_ens * per];
            for cell in &mut broad {
                cell.reset();
            }
            for cell in &mut ens_cells {
                cell.reset();
            }
            for _ in 0..granule_ticks {
                for j in 0..n_broad {
                    let over = broad_drive - self.broad_thresholds[j];
                    let i = rheobase + self.params.granule_gain * over;
                    if over > 0.0 {
                        self.ledger.neuron_updates_driven += 1;
                    } else {
                        self.ledger.neuron_updates_idle += 1;
                    }
                    if broad[j].step(dt, i) {
                        broad_spikes[j] += 1.0;
                        train.push(Spike { t, source: granule_base + j as u32 });
                        self.ledger.spikes_out += 1;
                    }
                }
                for k in 0..n_ens {
                    for j in 0..per {
                        let idx = k * per + j;
                        let over = drive[k] - self.ensembles[k].thresholds[j];
                        let i = rheobase + self.params.granule_gain * over;
                        if over > 0.0 {
                            self.ledger.neuron_updates_driven += 1;
                        } else {
                            self.ledger.neuron_updates_idle += 1;
                        }
                        if ens_cells[idx].step(dt, i) {
                            ens_spikes[idx] += 1.0;
                            train.push(Spike {
                                t,
                                source: granule_base + (n_broad + idx) as u32,
                            });
                            self.ledger.spikes_out += 1;
                        }
                    }
                }
                t += 1;
            }

            let granule_total: f64 = broad_spikes.iter().chain(ens_spikes.iter()).sum();
            self.ledger.syn_ops += (granule_total as u64) * n as u64;
            self.ledger.syn_fetches += (granule_total as u64) * n as u64;

            let a_broad = if n_broad == 0 {
                0.0
            } else {
                broad_spikes.iter().sum::<f64>() / (n_broad as f64 * granule_ceiling)
            };
            let mut act = vec![0.0f64; n_ens];
            for k in 0..n_ens {
                act[k] = (0..per).map(|j| ens_spikes[k * per + j]).sum::<f64>()
                    / (per as f64 * granule_ceiling);
            }
            let act_total: f64 = act.iter().sum();
            for m in 0..n {
                // The specific term is an ACTIVITY-WEIGHTED AVERAGE of the stored masks, so it is
                // bounded by i_specific however many ensembles exist — and when several are
                // co-active it is a blur of several masks, which is what interference is.
                let specific = if act_total > 0.0 {
                    (0..n_ens).map(|k| act[k] * self.ensembles[k].mask[m]).sum::<f64>() / act_total
                } else {
                    0.0
                };
                inh[m] = self.params.i_broad * a_broad + self.params.i_specific * specific;
            }
        }

        if counts.iter().all(|&c| c == 0.0) {
            return Err(OlfactionError::Silent);
        }
        Ok(Presentation {
            inhibition: delivered,
            mitral: counts,
            mitral_open_loop: open_loop,
            granule_drive: drive,
            train,
            cycles,
        })
    }

    /// Learn an odour from **one** presentation and return its index.
    ///
    /// Runs [`EplParams::learn_cycles`] gamma cycles with every existing ensemble active, then
    /// writes the settled mitral pattern as a unit-norm template and its complement as the
    /// inhibitory mask. No gradient, no second epoch, no replay.
    ///
    /// # Errors
    ///
    /// [`OlfactionError::Full`] at [`EplParams::max_odours`], plus anything [`Epl::present`]
    /// refuses.
    pub fn learn(&mut self, odour: &Odour) -> Result<usize, OlfactionError> {
        if self.ensembles.len() >= self.params.max_odours {
            return Err(OlfactionError::Full {
                max: self.params.max_odours,
            });
        }
        let cycles = self.params.learn_cycles;
        let shown = self.present(odour, cycles)?;
        let norm = shown.mitral.iter().map(|x| x * x).sum::<f64>().sqrt();
        if !(norm > 0.0) {
            return Err(OlfactionError::Silent);
        }
        let template: Vec<f64> = shown.mitral.iter().map(|x| x / norm).collect();
        let peak = shown.mitral.iter().copied().fold(0.0f64, f64::max);
        let mask: Vec<f64> = shown.mitral.iter().map(|x| 1.0 - x / peak).collect();
        let thresholds = spread(
            self.params.ensemble_theta_lo,
            self.params.ensemble_theta_hi,
            self.params.granule_per_odour,
        );
        self.ensembles.push(Ensemble {
            template,
            mask,
            thresholds,
        });
        Ok(self.ensembles.len() - 1)
    }

    /// Present an odour for [`EplParams::recall_cycles`] cycles and score it against every stored
    /// template.
    ///
    /// # Errors
    ///
    /// [`OlfactionError::Untrained`] before anything is learned, plus anything [`Epl::present`]
    /// refuses.
    pub fn recall(&mut self, odour: &Odour) -> Result<Recall, OlfactionError> {
        if self.ensembles.is_empty() {
            return Err(OlfactionError::Untrained);
        }
        let cycles = self.params.recall_cycles;
        let shown = self.present(odour, cycles)?;
        let mut scores = Vec::with_capacity(self.ensembles.len());
        for e in &self.ensembles {
            scores.push(cosine_similarity(&shown.mitral, &e.template).unwrap_or(0.0));
        }
        let mut best = 0usize;
        for (k, &v) in scores.iter().enumerate() {
            if v > scores[best] {
                best = k;
            }
        }
        let score = scores[best];
        let runner_up = scores
            .iter()
            .enumerate()
            .filter(|(k, _)| *k != best)
            .map(|(_, v)| *v)
            .fold(0.0f64, f64::max);
        Ok(Recall {
            identified: (score >= self.params.recall_threshold).then_some(best),
            score,
            runner_up,
            margin: score - runner_up,
            scores,
            mitral: shown.mitral,
        })
    }

    /// Fraction of `odours` the circuit accepts as one of its stored odours.
    ///
    /// Measure it on odours the circuit has *not* learned and it is the false-positive rate. It is
    /// a method rather than a note in the docs because a recall threshold quoted without one is an
    /// unfalsifiable number.
    ///
    /// # Errors
    ///
    /// Anything [`Epl::recall`] refuses.
    pub fn acceptance_rate(&mut self, odours: &[Odour]) -> Result<f64, OlfactionError> {
        if odours.is_empty() {
            return Err(OlfactionError::Parameter {
                field: "odours",
                reason: "needs at least one odour to measure a rate over",
            });
        }
        let mut hits = 0usize;
        for o in odours {
            if self.recall(o)?.identified.is_some() {
                hits += 1;
            }
        }
        Ok(hits as f64 / odours.len() as f64)
    }
}

/// One point on a capacity curve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CapacityPoint {
    /// Odours stored when this point was measured.
    pub learned: usize,
    /// Fraction of stored odours whose occluded probe picked the right template as the best match.
    pub accuracy: f64,
    /// Mean cosine against the probe's own template.
    pub mean_self: f64,
    /// Mean cosine against the best *wrong* template. This is the interference term, and it rises
    /// with the number stored because the maximum of more draws is larger.
    pub mean_best_other: f64,
    /// `mean_self - mean_best_other`. The decision's headroom, which shrinks long before the
    /// identification itself becomes wrong — which is why a capacity figure reported only as
    /// accuracy hides how close the circuit is to failing.
    pub mean_margin: f64,
}

/// Learn odours one at a time and measure recall at each of `probe_at`.
///
/// Each probe presents every stored odour once, corrupted by `occlusion` against a fresh
/// interfering odour, and records what the circuit does. Deterministic in `seed`.
///
/// # Errors
///
/// [`OlfactionError::Parameter`] for an empty or unsorted `probe_at`, or one asking for more
/// odours than [`EplParams::max_odours`]; plus anything [`Epl::learn`] or [`Epl::recall`] refuses.
pub fn capacity_curve(
    params: &EplParams,
    seed: u64,
    sparsity: f64,
    probe_at: &[usize],
    occlusion: Occlusion,
) -> Result<Vec<CapacityPoint>, OlfactionError> {
    if probe_at.is_empty() || probe_at[0] == 0 {
        return Err(OlfactionError::Parameter {
            field: "probe_at",
            reason: "needs at least one probe point and none may be zero",
        });
    }
    if probe_at.windows(2).any(|w| w[1] <= w[0]) {
        return Err(OlfactionError::Parameter {
            field: "probe_at",
            reason: "must be strictly increasing",
        });
    }
    let top = probe_at[probe_at.len() - 1];
    if top > params.max_odours {
        return Err(OlfactionError::Full {
            max: params.max_odours,
        });
    }
    let mut epl = Epl::new(params.clone())?;
    let mut source = OdourGenerator::new(seed, params.mitral, sparsity)?;
    // ⛔ The interfering backgrounds have their OWN generator. The first version drew them from
    // `source`, so asking for an extra probe point consumed draws and changed every odour learned
    // afterwards: the same `(params, seed)` gave accuracy 0.875 at 24 odours with five probe
    // points and 0.833 with one. "Deterministic in `seed`" was true and misleading.
    // And both the background generator and the occlusion draws are re-seeded from `(seed, k)` at
    // each probe point, so the point at `k` is a function of `(params, seed, sparsity, k)` alone
    // and not of how many earlier points were asked for.
    let mut stored: Vec<Odour> = Vec::new();
    let mut out = Vec::with_capacity(probe_at.len());
    for k in 1..=top {
        let o = source.next_odour()?;
        epl.learn(&o)?;
        stored.push(o);
        if !probe_at.contains(&k) {
            continue;
        }
        let (mut hits, mut self_sum, mut other_sum) = (0usize, 0.0f64, 0.0f64);
        let salt = (k as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let mut background =
            OdourGenerator::new(seed ^ 0x5EED_0BAC_0000_0007 ^ salt, params.mitral, sparsity)?;
        let mut probe_rng = Rng::new(seed ^ 0xA5A5_5A5A_1234_9876 ^ salt);
        for (i, s) in stored.iter().enumerate() {
            let interferent = background.next_odour()?;
            let probe = occlusion.apply(s, &interferent, &mut probe_rng)?;
            let r = epl.recall(&probe)?;
            let mine = r.scores[i];
            let other = r
                .scores
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, v)| *v)
                .fold(0.0f64, f64::max);
            if mine >= other {
                hits += 1;
            }
            self_sum += mine;
            other_sum += other;
        }
        let n = stored.len() as f64;
        out.push(CapacityPoint {
            learned: k,
            accuracy: hits as f64 / n,
            mean_self: self_sum / n,
            mean_best_other: other_sum / n,
            mean_margin: (self_sum - other_sum) / n,
        });
    }
    Ok(out)
}

/// The longest window [`population_period`] will analyse: `2^18` ticks, about 26 s at the default
/// time step and about 15 s of computation at the measured quadratic cost.
pub const MAX_PERIOD_TICKS: u64 = 1 << 18;

/// Period of a spike train's population rhythm, in seconds, measured from the train itself.
///
/// Builds the population histogram over `ticks` time steps from spikes whose `source` is below
/// `source_max` — pass [`EplParams::mitral`] to measure the mitral layer's rhythm — removes its
/// mean, walks forward to the first lag where the autocorrelation is non-positive, and returns the
/// lag of the largest autocorrelation from there on.
///
/// **The autocorrelation is deliberately left unnormalised by the overlap count.** At lag `k*T`
/// only `n - k*T` sample pairs overlap, so the fundamental outranks its own harmonics purely by
/// having more of the window to work with. Dividing by `n - lag`, which looks like the correct
/// thing to do, makes `2T` tie with `T`, and the estimator then reports a period twice too long on
/// about half its inputs — a failure that draws a perfectly plausible raster.
///
/// `None` when there are no spikes in range, when `ticks` is under 4 or over
/// [`MAX_PERIOD_TICKS`], when `dt` is not finite and positive, or when the autocorrelation never
/// goes non-positive inside the search range — that last case being a window with no resolvable
/// rhythm, which is a different statement from a long period and is kept different.
///
/// # Cost
///
/// Quadratic in `ticks`: every lag up to `ticks / 2` is a dot product over the window. Measured in
/// release: 20,000 ticks in 0.1 s, 50,000 in 0.7 s, 100,000 in 3.3 s, 200,000 in 10.5 s. The bound
/// refuses what would take minutes; the first version's only guard was `ticks > usize::MAX`, which
/// on a 64-bit machine never fires, and a caller asking about 2e12 ticks was handed a 16 TB
/// allocation attempt instead of a `None`.
#[must_use]
pub fn population_period(train: &Train, source_max: u32, ticks: u64, dt: f64) -> Option<f64> {
    if !(4..=MAX_PERIOD_TICKS).contains(&ticks) || !(dt > 0.0) || !dt.is_finite() {
        return None;
    }
    let n = ticks as usize;
    let mut h = vec![0.0f64; n];
    let mut total = 0.0f64;
    for s in train.spikes() {
        if s.source < source_max && s.t < ticks {
            h[s.t as usize] += 1.0;
            total += 1.0;
        }
    }
    if total <= 0.0 {
        return None;
    }
    let mean = total / n as f64;
    for v in &mut h {
        *v -= mean;
    }
    let max_lag = n / 2;
    if max_lag < 2 {
        return None;
    }
    let corr = |lag: usize| -> f64 { (0..n - lag).map(|t| h[t] * h[t + lag]).sum() };
    let mut lag = 1usize;
    while lag <= max_lag && corr(lag) > 0.0 {
        lag += 1;
    }
    if lag > max_lag {
        return None;
    }
    let (mut best_lag, mut best) = (0usize, 0.0f64);
    for l in lag..=max_lag {
        let c = corr(l);
        if c > best {
            best = c;
            best_lag = l;
        }
    }
    if best_lag == 0 {
        return None;
    }
    Some(best_lag as f64 * dt)
}

#[cfg(test)]
mod tests {
    use super::{
        CapacityPoint, Epl, EplParams, Occlusion, Odour, OdourGenerator, OlfactionError,
        capacity_curve, cosine_similarity, population_period, spread, tanimoto,
    };
    use crate::ledger::{LOIHI_2018, Ledger};
    use crate::neuron::{Lif, Neuron};
    use crate::rng::Rng;
    use crate::spike::{Spike, Train};

    /// A layer small enough to sweep in a test and wide enough to separate odours.
    fn small() -> EplParams {
        EplParams {
            mitral: 32,
            granule_per_odour: 4,
            broad_granule: 4,
            max_odours: 24,
            ..EplParams::default()
        }
    }

    // ---- the measures ----

    /// `None` is the answer where a cosine does not exist. Returning `0.0` for a zero vector would
    /// report "orthogonal", which a sweep would then plot as data.
    #[test]
    fn cosine_similarity_refuses_where_the_answer_does_not_exist() {
        assert_eq!(cosine_similarity(&[1.0, 2.0], &[1.0]), None);
        assert_eq!(cosine_similarity(&[], &[]), None);
        assert_eq!(cosine_similarity(&[0.0, 0.0], &[1.0, 1.0]), None);
        assert_eq!(cosine_similarity(&[f64::NAN, 1.0], &[1.0, 1.0]), None);
        assert_eq!(cosine_similarity(&[f64::INFINITY, 1.0], &[1.0, 1.0]), None);
        // And where it does exist it is the closed form: identical directions give exactly 1,
        // orthogonal ones exactly 0, and scale does not matter.
        let a = [3.0, 4.0];
        assert!((cosine_similarity(&a, &[6.0, 8.0]).unwrap() - 1.0).abs() < 1e-15);
        assert!(cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]).unwrap().abs() < 1e-15);
        // cos((1,0),(1,1)) = 1/sqrt(2), a value neither endpoint could produce by accident.
        let mixed = cosine_similarity(&[1.0, 0.0], &[1.0, 1.0]).unwrap();
        assert!((mixed - 0.5f64.sqrt()).abs() < 1e-15, "{mixed}");
    }

    /// The closed form: for 0/1 vectors the Tanimoto coefficient IS the Jaccard index. Checked on
    /// every pair of subsets of a four-element set, against `|A ∩ B| / |A ∪ B|` counted directly.
    #[test]
    fn tanimoto_reduces_to_jaccard_on_binary_vectors() {
        let mut checked = 0;
        for a in 0u32..16 {
            for b in 0u32..16 {
                if a == 0 || b == 0 {
                    continue;
                }
                let va: Vec<f64> = (0..4).map(|k| f64::from((a >> k) & 1)).collect();
                let vb: Vec<f64> = (0..4).map(|k| f64::from((b >> k) & 1)).collect();
                let inter = f64::from((a & b).count_ones());
                let union = f64::from((a | b).count_ones());
                let got = tanimoto(&va, &vb).unwrap();
                assert!((got - inter / union).abs() < 1e-15, "a={a} b={b} got={got}");
                checked += 1;
            }
        }
        assert_eq!(checked, 225, "the sweep must actually run every non-empty pair");
        assert_eq!(tanimoto(&[0.0, 0.0], &[0.0, 0.0]), None);
        assert_eq!(tanimoto(&[1.0], &[1.0, 2.0]), None);
    }

    // ---- boundary refusals ----

    #[test]
    fn odour_rejects_what_the_circuit_cannot_integrate() {
        assert_eq!(Odour::new(vec![]), Err(OlfactionError::EmptyOdour));
        assert_eq!(
            Odour::new(vec![0.1, f64::NAN]),
            Err(OlfactionError::NotFinite { index: 1 })
        );
        assert_eq!(
            Odour::new(vec![0.1, f64::INFINITY]),
            Err(OlfactionError::NotFinite { index: 1 })
        );
        assert_eq!(
            Odour::new(vec![-0.5, 0.1]),
            Err(OlfactionError::Negative { index: 0 })
        );
        assert_eq!(Odour::new(vec![0.0, 0.0]), Err(OlfactionError::ZeroOdour));
        let ok = Odour::new(vec![0.0, 0.5]).unwrap();
        assert_eq!(ok.len(), 2);
        assert!(!ok.is_empty());
    }

    /// Every constructor refusal names the field, so a caller can act on it without reading the
    /// source. The list is the one `Epl::new` documents.
    #[test]
    fn parameters_that_cannot_run_are_refused_by_name() {
        let field = |p: EplParams| match Epl::new(p) {
            Err(OlfactionError::Parameter { field, .. }) => field,
            other => panic!("expected a named parameter refusal, got {other:?}"),
        };
        assert_eq!(field(EplParams { mitral: 0, ..small() }), "mitral");
        assert_eq!(
            field(EplParams { granule_per_odour: 0, ..small() }),
            "granule_per_odour"
        );
        assert_eq!(field(EplParams { dt: 0.0, ..small() }), "dt");
        assert_eq!(field(EplParams { dt: f64::NAN, ..small() }), "dt");
        assert_eq!(field(EplParams { gamma_hz: -1.0, ..small() }), "gamma_hz");
        // A gamma period shorter than four time steps has no phases to split.
        assert_eq!(field(EplParams { gamma_hz: 1e5, ..small() }), "gamma_hz");
        assert_eq!(field(EplParams { mitral_duty: 1.0, ..small() }), "mitral_duty");
        assert_eq!(field(EplParams { mitral_duty: 0.0, ..small() }), "mitral_duty");
        assert_eq!(field(EplParams { i_gain: f64::NAN, ..small() }), "i_gain");
        // Crossed threshold bounds: the defect class that panicked a `clamp` in an earlier module.
        assert_eq!(
            field(EplParams { broad_theta_hi: -1.0, ..small() }),
            "broad_theta_hi"
        );
        assert_eq!(
            field(EplParams { ensemble_theta_hi: 0.0, ensemble_theta_lo: 0.5, ..small() }),
            "ensemble_theta_hi"
        );
        let no_ref = Lif { t_ref: 0.0, ..Lif::default() };
        assert_eq!(
            field(EplParams { mitral_cell: no_ref, ..small() }),
            "mitral_cell.t_ref"
        );
        // A granule cell whose threshold is at or below rest has no rheobase to bias to.
        let flat = Lif { v_th: -70e-3, ..Lif::default() };
        assert_eq!(
            field(EplParams { granule_cell: flat, ..small() }),
            "granule_cell"
        );
        assert_eq!(field(EplParams { learn_cycles: 0, ..small() }), "learn_cycles");
        // And the good one builds.
        assert!(Epl::new(small()).is_ok());
    }

    /// `n - 1` in a denominator is a division by zero for a one-cell ensemble, which is a
    /// configuration the constructor accepts. It answers instead.
    #[test]
    fn a_one_cell_ensemble_does_not_divide_by_zero() {
        assert_eq!(spread(0.3, 0.7, 0), Vec::<f64>::new());
        assert_eq!(spread(0.3, 0.7, 1), vec![0.3]);
        let three = spread(0.0, 1.0, 3);
        assert_eq!(three, vec![0.0, 0.5, 1.0]);
        assert!(three.iter().all(|v| v.is_finite()));

        let p = EplParams {
            mitral: 16,
            granule_per_odour: 1,
            broad_granule: 1,
            max_odours: 2,
            ..EplParams::default()
        };
        let mut src = OdourGenerator::new(6, 16, 0.4).unwrap();
        let mut epl = Epl::new(p).unwrap();
        let odour = src.next_odour().unwrap();
        assert_eq!(epl.learn(&odour).unwrap(), 0);
        assert_eq!(epl.ensembles()[0].thresholds.len(), 1);
        assert_eq!(epl.recall(&odour).unwrap().identified, Some(0));
    }

    // ---- the mitral limb, against the closed form ----

    /// Two claims in one: the circuit integrates exactly `mitral_current`, and that current's
    /// inter-spike interval is the one `Lif::isi` predicts in closed form.
    ///
    /// The open-loop configuration (no broad pool, no inhibition of either kind) is what isolates
    /// the mitral limb; with the loop closed the counts are supposed to differ, which is the point
    /// of the rest of the module.
    #[test]
    fn the_circuit_uses_the_drive_map_it_documents() {
        // A fine time step so that the discretisation bound on a measured interval is small
        // compared with the interval itself: 0.02 ms steps against a ~1.4 ms interval.
        let p = EplParams {
            mitral: 8,
            broad_granule: 0,
            granule_per_odour: 1,
            max_odours: 1,
            dt: 2e-5,
            i_broad: 0.0,
            i_specific: 0.0,
            ..EplParams::default()
        };
        let channels: Vec<f64> = (0..8).map(|k| f64::from(k) / 7.0).collect();
        let odour = Odour::new(channels.clone()).unwrap();
        let mut epl = Epl::new(p.clone()).unwrap();
        let shown = epl.present(&odour, 3).unwrap();

        // Arm 1: a bare Lif driven by the documented current reproduces the circuit's counts
        // EXACTLY, cell for cell. A different bias, gain or sign would move at least one of them.
        let mitral_ticks = (f64::from(u32::try_from(epl.ticks_per_cycle()).unwrap())
            * p.mitral_duty)
            .round() as usize;
        for m in 0..8 {
            let i = epl.mitral_current(channels[m], 0.0);
            assert!((i - (p.i_bias + p.i_gain * channels[m])).abs() < 1e-24);
            let mut cell = p.mitral_cell;
            let mut count = 0.0;
            for _ in 0..mitral_ticks {
                if cell.step(p.dt, i) {
                    count += 1.0;
                }
            }
            assert_eq!(count, shown.mitral[m], "cell {m}");
        }
        // The counts must actually vary across the sweep, or the comparison above is vacuous.
        assert!(shown.mitral[7] > shown.mitral[0] + 2.0, "{:?}", shown.mitral);

        // Arm 2: the closed form. `Lif::isi` is the analytic interval under constant current; the
        // train's measured intervals must match it to within the simulation's own quantisation.
        let top = epl.mitral_current(channels[7], 0.0);
        let closed = p.mitral_cell.isi(top).unwrap();
        let intervals = shown.train.intervals(7, p.dt);
        assert!(intervals.len() >= 5, "need several intervals to average");
        let tol = 2.0 * p.dt;
        let mut within = 0;
        for &iv in &intervals {
            if (iv - closed).abs() <= tol {
                within += 1;
            }
        }
        // Intervals that straddle a gamma-cycle boundary are longer by the granule phase, so the
        // claim is about the intervals inside a burst, which are the great majority.
        assert!(
            within >= intervals.len() - 2,
            "closed form {closed:.6e}, intervals {intervals:?}"
        );
        // The tolerance cannot accept the NEIGHBOURING current's answer: it bites. Channel 6 is
        // the neighbour of channel 7 (this guard used to compare against channel 5, two away, and
        // claim "the neighbouring channel"); its interval is 3.18 tolerances off.
        let neighbour = p.mitral_cell.isi(epl.mitral_current(channels[6], 0.0)).unwrap();
        assert!(
            (neighbour - closed).abs() > 3.0 * tol,
            "tolerance {tol:e} is loose enough to accept the wrong current"
        );
        // And a sub-rheobase current has no interval at all, rather than a large one.
        assert_eq!(p.mitral_cell.isi(0.0), None);
    }

    // ---- the granule limb ----

    /// The recognition score is not computed beside the circuit: the granule cell's dendritic sum
    /// IS the cosine similarity, because the template is unit-norm and the sum is divided by the
    /// mitral pattern's norm. Asserted to 1e-12 against `cosine_similarity` computed independently.
    #[test]
    fn granule_drive_equals_the_cosine_by_construction() {
        let mut src = OdourGenerator::new(21, 32, 0.35).unwrap();
        let mut epl = Epl::new(small()).unwrap();
        for o in &src.library(5).unwrap() {
            epl.learn(o).unwrap();
        }
        let probe = src.next_odour().unwrap();
        let shown = epl.present(&probe, 6).unwrap();
        assert_eq!(shown.granule_drive.len(), 5);
        for (k, e) in epl.ensembles().iter().enumerate() {
            let want = cosine_similarity(&shown.mitral, &e.template).unwrap();
            assert!(
                (shown.granule_drive[k] - want).abs() < 1e-12,
                "k={k} drive={} cosine={want}",
                shown.granule_drive[k]
            );
        }
        // Non-vacuous: the drives are not all equal, so matching them one by one says something.
        let hi = shown.granule_drive.iter().copied().fold(f64::MIN, f64::max);
        let lo = shown.granule_drive.iter().copied().fold(f64::MAX, f64::min);
        assert!(hi - lo > 0.05, "{:?}", shown.granule_drive);
    }

    /// The two limbs of the reciprocal dendrodendritic synapse carry different numbers. Storing one
    /// and negating it would give a circuit that silenced exactly the odour it recognised.
    #[test]
    fn the_two_limbs_of_the_reciprocal_synapse_are_complements() {
        let mut src = OdourGenerator::new(8, 32, 0.35).unwrap();
        let mut epl = Epl::new(small()).unwrap();
        let odour = src.next_odour().unwrap();
        epl.learn(&odour).unwrap();
        let e = &epl.ensembles()[0];
        // The template is a unit vector, exactly as the drive-equals-cosine identity needs.
        let norm2: f64 = e.template.iter().map(|x| x * x).sum();
        assert!((norm2 - 1.0).abs() < 1e-12, "{norm2}");
        // The mask is the complement: exactly 0 at the most active cell, exactly 1 at a silent one,
        // and inside [0, 1] everywhere.
        let peak = e
            .template
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(e.mask[peak], 0.0);
        assert!(e.mask.iter().all(|m| (0.0..=1.0).contains(m)));
        let silent: Vec<usize> = (0..e.template.len()).filter(|&m| e.template[m] == 0.0).collect();
        assert!(!silent.is_empty(), "the test needs a silent cell to check");
        for m in silent {
            assert_eq!(e.mask[m], 1.0);
        }
        // And the two limbs point in opposite directions rather than the same one.
        let overlap = cosine_similarity(&e.template, &e.mask).unwrap();
        assert!(overlap < 0.35, "template and mask are too alike: {overlap}");
    }

    /// The template is the **settled** mitral pattern, not the first cycle's open-loop response.
    /// Both are plausible things to store and they differ, so the choice has to be pinned by a test
    /// rather than left to the doc: storing the open-loop pattern would throw away exactly the
    /// decorrelation the loop performed, and every downstream number would still look reasonable.
    #[test]
    fn the_template_is_the_settled_pattern_not_the_open_loop_one() {
        let unit = |v: &[f64]| -> Vec<f64> {
            let n: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            v.iter().map(|x| x / n).collect()
        };
        let mut src = OdourGenerator::new(90, 32, 0.35).unwrap();
        let odour = src.next_odour().unwrap();
        let mut epl = Epl::new(small()).unwrap();
        epl.learn(&odour).unwrap();
        let template = epl.ensembles()[0].template.clone();

        // The same presentation on a circuit that has learned nothing: its FINAL cycle is what
        // `learn` is supposed to have stored.
        let mut fresh = Epl::new(small()).unwrap();
        let shown = fresh.present(&odour, small().learn_cycles).unwrap();
        let settled = unit(&shown.mitral);
        let open = unit(&shown.mitral_open_loop);
        for m in 0..template.len() {
            assert!((template[m] - settled[m]).abs() < 1e-12, "cell {m}");
        }
        // And the two candidates really differ, or the assertion above asserts nothing.
        let same = cosine_similarity(&settled, &open).unwrap();
        assert!(
            same < 0.999,
            "the loop changed nothing in {} cycles, so this test is vacuous (cos {same})",
            small().learn_cycles
        );
    }

    /// The broad pool is a **magnitude** feedback loop, and the module doc says that dividing its
    /// dendritic sum by `||r||` — which looks like tidying up, and which the learned ensembles
    /// genuinely do — would make it measure the mitral pattern's shape instead of its size.
    ///
    /// That claim survived every other test in this module: the mutation was applied and all twenty
    /// passed. So it is asserted directly. The broad pool's own spike count must rise with
    /// concentration, across a five-fold step, because its drive is the unnormalised population
    /// rate; a shape-normalised drive is scale-invariant and pins the pool at one output.
    ///
    /// The second assertion is the honest counterpart and is pinned so it cannot be quietly
    /// overclaimed later: the limb is **subtractive**, so it does not preserve ratios. It removes a
    /// near-constant number of spikes and the settled ratio therefore comes out LARGER than the
    /// open-loop one. Divisive normalisation is a glomerular-layer mechanism this module does not
    /// implement.
    #[test]
    fn the_broad_pool_answers_concentration_and_not_shape() {
        let measure = |gain: f64, scale: f64| {
            let p = EplParams { i_broad: gain, ..small() };
            let mut src = OdourGenerator::new(64, p.mitral, 0.35).unwrap();
            let base = src.next_odour().unwrap();
            let scaled = Odour::new(base.channels().iter().map(|c| c * scale).collect()).unwrap();
            let mut epl = Epl::new(p.clone()).unwrap();
            let shown = epl.present(&scaled, 6).unwrap();
            let first = u32::try_from(p.mitral).unwrap();
            let last = first + u32::try_from(p.broad_granule).unwrap();
            let broad = shown
                .train
                .spikes()
                .iter()
                .filter(|s| s.source >= first && s.source < last)
                .count();
            (
                broad,
                shown.mitral.iter().sum::<f64>(),
                shown.mitral_open_loop.iter().sum::<f64>(),
            )
        };

        let scales = [0.2f64, 0.5, 1.0];
        let on: Vec<_> = scales.iter().map(|&s| measure(3e-9, s)).collect();
        for w in on.windows(2) {
            assert!(
                w[1].0 >= w[0].0,
                "broad output fell as concentration rose: {:?}",
                on.iter().map(|x| x.0).collect::<Vec<_>>()
            );
        }
        assert!(
            on[2].0 >= on[0].0 * 3,
            "the broad pool barely answered a five-fold concentration step: {} -> {}",
            on[0].0,
            on[2].0
        );
        // The open-loop response must really have changed, or the sweep asserts nothing.
        assert!(on[2].2 > on[0].2 * 1.5, "open-loop totals {} and {}", on[0].2, on[2].2);

        let open_ratio = on[2].2 / on[0].2;
        let settled_ratio = on[2].1 / on[0].1;
        assert!(
            settled_ratio > open_ratio,
            "subtractive inhibition should expand the ratio, not compress it: \
             open {open_ratio:.4}, settled {settled_ratio:.4}"
        );
        assert!(on[0].2 - on[0].1 > 0.0, "the loop removed nothing at the low concentration");

        // The control isolates the DELIVERY, not the pool: at `i_broad = 0` the broad cells still
        // fire — their drive is unchanged — but nothing reaches a mitral dendrite, so there is no
        // path by which the settled response could differ from the open-loop one and the two must
        // be bit-identical.
        for &scale in &scales {
            let off = measure(0.0, scale);
            assert!(off.0 > 0, "the control arm must still have a firing pool at {scale}x");
            assert!((off.1 - off.2).abs() < 1e-12, "no inhibition delivered, yet the pattern moved");
        }
    }

    /// Both normalisers are the refractory bound in closed form, and both are genuine upper bounds
    /// on what a saturated cell can emit. The gap between the bound and the achieved maximum is
    /// asserted too, because it is real and surprising: a floating-point refractory countdown can
    /// cost one extra time step per spike, and the mitral bound of 25 is reached as 18.
    #[test]
    fn the_rate_ceilings_are_the_refractory_bound_and_bound_a_saturated_cell() {
        let p = small();
        let epl = Epl::new(p.clone()).unwrap();
        let cycle_ticks = epl.ticks_per_cycle();
        let mitral_ticks = (cycle_ticks as f64 * p.mitral_duty).round() as usize;
        let granule_ticks = cycle_ticks - mitral_ticks;
        assert_eq!(cycle_ticks, 250);
        assert_eq!(mitral_ticks, 125);

        let want_m = (mitral_ticks as f64 * p.dt / p.mitral_cell.t_ref).floor();
        let want_g = (granule_ticks as f64 * p.dt / p.granule_cell.t_ref).floor();
        assert_eq!(epl.mitral_ceiling(), want_m);
        assert_eq!(epl.granule_ceiling(), want_g);
        assert_eq!(want_m, 25.0);
        assert_eq!(want_g, 12.0);

        // A saturated bare cell of each kind stays under its bound.
        let saturate = |cell: Lif, ticks: usize| {
            let mut c = cell;
            let mut n = 0.0;
            for _ in 0..ticks {
                if c.step(p.dt, 1e-6) {
                    n += 1.0;
                }
            }
            n
        };
        let sat_m = saturate(p.mitral_cell, mitral_ticks);
        let sat_g = saturate(p.granule_cell, granule_ticks);
        assert!(sat_m <= epl.mitral_ceiling(), "{sat_m} spikes over a bound of {want_m}");
        assert!(sat_g <= epl.granule_ceiling(), "{sat_g} spikes over a bound of {want_g}");
        assert_eq!(sat_m, 18.0, "the mitral gap moved; the doc quotes this number");
        assert_eq!(sat_g, 12.0, "the granule bound is tight at these parameters");
        // A granule cell with no refractory period is legal and falls back to the tick count.
        let free = EplParams { granule_cell: Lif { t_ref: 0.0, ..p.granule_cell }, ..p };
        assert_eq!(Epl::new(free).unwrap().granule_ceiling(), granule_ticks as f64);
    }

    /// How many channels an occlusion fraction actually hits, to the channel. `floor` instead of
    /// `round` differs by one for any fraction whose product is not an integer, and one channel in
    /// thirty-two is invisible in any averaged sweep.
    #[test]
    fn an_occlusion_fraction_hits_the_rounded_number_of_channels() {
        let mut rng = Rng::new(3);
        let dense = Odour::new(vec![0.5; 3]).unwrap();
        let other = Odour::new(vec![0.9; 3]).unwrap();
        // 0.5 * 3 = 1.5: round gives 2, floor gives 1.
        let hit = Occlusion::Dropout { fraction: 0.5 }
            .apply(&dense, &other, &mut rng)
            .unwrap();
        assert_eq!(hit.channels().iter().filter(|c| **c == 0.0).count(), 2);
        // 0.3 * 3 = 0.9: round gives 1, floor gives 0 — the difference between corrupting and not.
        let one = Occlusion::Dropout { fraction: 0.3 }
            .apply(&dense, &other, &mut rng)
            .unwrap();
        assert_eq!(one.channels().iter().filter(|c| **c == 0.0).count(), 1);
        let none = Occlusion::Dropout { fraction: 0.1 }
            .apply(&dense, &other, &mut rng)
            .unwrap();
        assert_eq!(none, dense, "0.1 * 3 rounds to zero channels");
        // And the same arithmetic on the interferent model.
        let mixed = Occlusion::Interferent { fraction: 0.5 }
            .apply(&dense, &other, &mut rng)
            .unwrap();
        assert_eq!(mixed.channels().iter().filter(|c| **c == 0.9).count(), 2);
    }

    // ---- the oscillation ----

    /// The estimator is checked against a period it was not told, before it is used on the circuit.
    /// Without this, a gamma test would be checking the circuit against an estimator that could
    /// itself be reporting the parameter it was handed.
    #[test]
    fn population_period_recovers_a_period_it_was_not_told() {
        for period in [17usize, 37, 64] {
            let mut spikes = Vec::new();
            for cycle in 0..9u64 {
                for k in 0..5u64 {
                    spikes.push(Spike {
                        t: cycle * period as u64 + k,
                        source: u32::try_from(k).unwrap(),
                    });
                }
            }
            let train = Train::from_spikes(spikes);
            let ticks = 9 * period as u64;
            let got = population_period(&train, 8, ticks, 1e-4).unwrap();
            assert!(
                (got - period as f64 * 1e-4).abs() < 0.5e-4,
                "period {period}: got {got:e}"
            );
        }
        // The refusals, each for its own reason.
        assert_eq!(population_period(&Train::new(), 8, 100, 1e-4), None);
        let one = Train::from_spikes(vec![Spike { t: 3, source: 0 }]);
        assert_eq!(population_period(&one, 8, 2, 1e-4), None, "ticks under 4");
        assert_eq!(population_period(&one, 8, 100, 0.0), None, "dt not positive");
        assert_eq!(population_period(&one, 0, 100, 1e-4), None, "no source in range");
    }

    /// The gamma period the circuit runs at is the one its parameters set, measured from the spikes
    /// rather than read back off the parameter. Swept over a factor of four in frequency, so a
    /// circuit that oscillated at some fixed rate of its own would fail at two of the three points.
    /// ⛔ And the same period, bit-exact, with the loop deleted. This test used to read as
    /// evidence that "the loop rings"; the rhythm is the `mitral_duty` gate, and the control arm
    /// says so. See the module doc.
    #[test]
    fn the_gamma_period_measured_from_the_train_is_the_one_the_parameters_set() {
        for hz in [20.0f64, 40.0, 80.0] {
            let p = EplParams { gamma_hz: hz, ..small() };
            let open = EplParams { broad_granule: 0, i_broad: 0.0, i_specific: 0.0, ..p.clone() };
            let odour = OdourGenerator::new(3, p.mitral, 0.35).unwrap().next_odour().unwrap();
            let mut measured = Vec::new();
            for params in [p.clone(), open] {
                let mut epl = Epl::new(params.clone()).unwrap();
                let shown = epl.present(&odour, 8).unwrap();
                let ticks = (epl.ticks_per_cycle() * 8) as u64;
                let got = population_period(&shown.train, p.mitral as u32, ticks, p.dt).unwrap();
                let want = epl.gamma_period();
                assert!((want - 1.0 / hz).abs() < p.dt, "{hz} Hz rounds badly");
                assert!(
                    (got - want).abs() < p.dt,
                    "{hz} Hz: measured {got:e}, parameters say {want:e}"
                );
                measured.push(got);
            }
            assert_eq!(measured[0], measured[1], "{hz} Hz: the loop changed the period; the doc says it cannot");
        }
        // The analysis window is bounded, and refuses rather than allocating.
        let mut train = crate::spike::Train::new();
        train.push(Spike { t: 0, source: 0 });
        assert_eq!(population_period(&train, 8, super::MAX_PERIOD_TICKS + 1, 1e-4), None);
        assert_eq!(population_period(&train, 8, 2_000_000_000_000, 1e-4), None);
    }

    // ---- one-shot learning and recall ----

    /// One presentation each, then every odour is recalled and the false-positive rate on 120
    /// unlearned odours is MEASURED. The second arm lowers only the threshold and requires the rate
    /// to rise: without it, a broken `acceptance_rate` that always answered zero would pass.
    #[test]
    fn one_shot_learning_recalls_and_the_false_positive_rate_is_measured() {
        let p = small();
        let mut src = OdourGenerator::new(19, p.mitral, 0.35).unwrap();
        let library = src.library(8).unwrap();
        let novel = src.library(120).unwrap();

        let mut epl = Epl::new(p.clone()).unwrap();
        for o in &library {
            epl.learn(o).unwrap();
        }
        let mut worst_self = 1.0f64;
        for (k, o) in library.iter().enumerate() {
            let r = epl.recall(o).unwrap();
            assert_eq!(r.identified, Some(k), "odour {k} scored {:?}", r.scores);
            worst_self = worst_self.min(r.score);
            assert!(r.margin > 0.0);
        }
        assert!(worst_self > p.recall_threshold, "worst self score {worst_self}");
        let fp = epl.acceptance_rate(&novel).unwrap();
        assert_eq!(fp, 0.0, "false positives at threshold {}", p.recall_threshold);

        // The control. Same odours, same circuit, only the threshold moved down; if the measurement
        // works, strangers now get through.
        let loose = EplParams { recall_threshold: 0.5, ..p };
        let mut lax = Epl::new(loose).unwrap();
        for o in &library {
            lax.learn(o).unwrap();
        }
        let fp_loose = lax.acceptance_rate(&novel).unwrap();
        assert!(fp_loose > 0.1, "a threshold of 0.5 let nothing through: {fp_loose}");
        assert_eq!(lax.acceptance_rate(&library).unwrap(), 1.0);

        // The boundary is INCLUSIVE, as `EplParams::recall_threshold` says: "at or above". The
        // circuit is deterministic, so a threshold set to exactly a score it produces is a case
        // that can be constructed, and `>` instead of `>=` flips it.
        let exact = epl.recall(&library[0]).unwrap().score;
        let mut edge = Epl::new(EplParams { recall_threshold: exact, ..p }).unwrap();
        for o in &library {
            edge.learn(o).unwrap();
        }
        let at_edge = edge.recall(&library[0]).unwrap();
        assert_eq!(at_edge.score, exact, "determinism broke, so this case is not on the edge");
        assert_eq!(at_edge.identified, Some(0), "a score exactly at the threshold was rejected");
    }

    #[test]
    fn an_untrained_circuit_refuses_and_a_full_one_refuses() {
        let p = EplParams { max_odours: 2, ..small() };
        let mut src = OdourGenerator::new(31, p.mitral, 0.35).unwrap();
        let mut epl = Epl::new(p).unwrap();
        let odour = src.next_odour().unwrap();
        assert_eq!(epl.recall(&odour), Err(OlfactionError::Untrained));
        assert_eq!(epl.learn(&odour).unwrap(), 0);
        assert_eq!(epl.learn(&src.next_odour().unwrap()).unwrap(), 1);
        assert_eq!(
            epl.learn(&src.next_odour().unwrap()),
            Err(OlfactionError::Full { max: 2 })
        );
        // Wrong width is refused with both numbers, not silently truncated.
        let narrow = Odour::new(vec![0.5; 4]).unwrap();
        assert_eq!(
            epl.present(&narrow, 1),
            Err(OlfactionError::Width { expected: 32, found: 4 })
        );
        assert!(matches!(
            epl.present(&odour, 0),
            Err(OlfactionError::Parameter { field: "cycles", .. })
        ));
        assert!(epl.acceptance_rate(&[]).is_err());
    }

    // ---- occlusion ----

    /// Graceful degradation is the architecture's claim, so it is asserted as monotonicity over a
    /// sweep — for both occlusion models — and the endpoints are required to differ, or a circuit
    /// that ignored its input entirely would pass.
    #[test]
    fn recall_degrades_monotonically_under_both_occlusion_models() {
        let p = small();
        let mut src = OdourGenerator::new(29, p.mitral, 0.35).unwrap();
        let mut epl = Epl::new(p).unwrap();
        let target = src.next_odour().unwrap();
        epl.learn(&target).unwrap();
        let mut rng = Rng::new(101);

        for dropout in [false, true] {
            let mut curve = Vec::new();
            // Dropout stops at 0.6: past that it can erase every active channel, and
            // `Occlusion::apply` then refuses rather than returning a zero odour.
            let sweep: &[f64] = if dropout {
                &[0.0, 0.15, 0.3, 0.45, 0.6]
            } else {
                &[0.0, 0.2, 0.4, 0.6, 0.8]
            };
            for &fraction in sweep {
                let model = if dropout {
                    Occlusion::Dropout { fraction }
                } else {
                    Occlusion::Interferent { fraction }
                };
                let (mut sum, mut n) = (0.0f64, 0.0f64);
                for _ in 0..25 {
                    let background = src.next_odour().unwrap();
                    let probe = model.apply(&target, &background, &mut rng).unwrap();
                    sum += epl.recall(&probe).unwrap().score;
                    n += 1.0;
                }
                curve.push(sum / n);
            }
            for w in curve.windows(2) {
                assert!(w[1] <= w[0], "not monotone (dropout={dropout}): {curve:?}");
            }
            assert!(
                curve[0] - curve[curve.len() - 1] > 0.25,
                "curve is flat, so monotonicity asserts nothing: {curve:?}"
            );
            assert!(curve[0] > 0.95, "clean recall should be near-perfect: {curve:?}");
        }
    }

    /// The learned inhibitory mask has to earn its place: the same odour, the same occlusion, with
    /// and without the odour-specific limb. Dropout is included as the honest counterpart — nothing
    /// in an inhibitory circuit can put back a channel that was set to zero.
    #[test]
    fn the_learned_mask_earns_its_place_under_interferent_occlusion() {
        let p = small();
        let mut src = OdourGenerator::new(43, p.mitral, 0.35).unwrap();
        let target = src.next_odour().unwrap();
        let mut with = Epl::new(p.clone()).unwrap();
        let mut without = Epl::new(EplParams { i_specific: 0.0, ..p }).unwrap();
        with.learn(&target).unwrap();
        without.learn(&target).unwrap();

        let mut rng = Rng::new(7);
        let mut rng_off = Rng::new(7); // same stream, so both see the same corruptions
        let (mut on, mut off) = (0.0f64, 0.0f64);
        for _ in 0..40 {
            let background = src.next_odour().unwrap();
            let model = Occlusion::Interferent { fraction: 0.6 };
            let a = model.apply(&target, &background, &mut rng).unwrap();
            let b = model.apply(&target, &background, &mut rng_off).unwrap();
            assert_eq!(a, b, "the two arms must see identical probes");
            on += with.recall(&a).unwrap().score;
            off += without.recall(&b).unwrap().score;
        }
        assert!(
            on > off * 1.05,
            "the specific limb bought nothing: with={:.4} without={:.4}",
            on / 40.0,
            off / 40.0
        );
    }

    // ---- decorrelation ----

    /// The property the EPL is famous for, measured inside one presentation: the overlap between
    /// two odours' mitral patterns in the first gamma cycle, before any inhibition has arrived,
    /// against the overlap once the loop has settled.
    ///
    /// The control is the whole test. With `i_broad = 0` the two numbers must be EXACTLY equal —
    /// there is no other path by which the settled pattern could differ from the open-loop one — so
    /// deleting the broad limb turns the first assertion into a failure rather than into a smaller
    /// effect that might go unnoticed.
    #[test]
    fn the_inhibitory_loop_lowers_pattern_overlap_and_removing_it_stops_that() {
        for target in [0.4f64, 0.6, 0.8] {
            let mut measured = Vec::new();
            for gain in [3e-9f64, 0.0] {
                let p = EplParams { i_broad: gain, ..small() };
                let mut epl = Epl::new(p.clone()).unwrap();
                let mut src = OdourGenerator::new(17, p.mitral, 0.35).unwrap();
                let (mut open, mut settled, mut n) = (0.0f64, 0.0f64, 0.0f64);
                for _ in 0..12 {
                    let (x, y, got) = src.pair_with_overlap(target).unwrap();
                    assert!((got - target).abs() < 1e-6, "generator missed: {got}");
                    let px = epl.present(&x, 6).unwrap();
                    let py = epl.present(&y, 6).unwrap();
                    open += cosine_similarity(&px.mitral_open_loop, &py.mitral_open_loop).unwrap();
                    settled += cosine_similarity(&px.mitral, &py.mitral).unwrap();
                    n += 1.0;
                }
                measured.push((open / n, settled / n));
            }
            let (open_on, settled_on) = measured[0];
            let (open_off, settled_off) = measured[1];
            assert!(
                settled_on < open_on - 0.05,
                "overlap {target}: loop did not decorrelate ({open_on:.4} -> {settled_on:.4})"
            );
            assert!(
                (settled_off - open_off).abs() < 1e-12,
                "overlap {target}: with no broad inhibition nothing may change \
                 ({open_off:.6} -> {settled_off:.6})"
            );
            assert!((open_on - open_off).abs() < 1e-12, "the open-loop arm must be identical");
        }
    }

    // ---- capacity ----

    /// Capacity, reported as a curve rather than a single number, because the identification stays
    /// right long after the decision has stopped having any room in it.
    ///
    /// Two assertions, both about the direction the curve must move: the best WRONG template's
    /// score rises as odours accumulate — the maximum of more random overlaps is larger — and the
    /// margin therefore collapses. Accuracy is recorded but not asserted on, because in this
    /// implementation it has not fallen far enough over the range tested to assert anything
    /// honestly, and saying so is the finding.
    #[test]
    fn capacity_interference_rises_and_the_margin_collapses() {
        let curve = capacity_curve(
            &small(),
            41,
            0.35,
            &[2, 4, 8, 16, 24],
            Occlusion::Interferent { fraction: 0.4 },
        )
        .unwrap();
        assert_eq!(curve.len(), 5);
        assert_eq!(curve[0].learned, 2);
        assert_eq!(curve[4].learned, 24);
        let first: &CapacityPoint = &curve[0];
        let last: &CapacityPoint = &curve[4];
        assert!(
            last.mean_best_other > first.mean_best_other * 2.0,
            "interference did not rise: {curve:?}"
        );
        assert!(
            last.mean_margin < first.mean_margin * 0.5,
            "margin did not collapse: {curve:?}"
        );
        for c in &curve {
            assert!((0.0..=1.0).contains(&c.accuracy));
            assert!(c.mean_self > 0.5, "recall itself fell apart: {c:?}");
            assert!((c.mean_margin - (c.mean_self - c.mean_best_other)).abs() < 1e-12);
        }
        // The refusals.
        assert!(capacity_curve(&small(), 1, 0.35, &[], Occlusion::Noise { amplitude: 0.1 }).is_err());
        assert!(
            capacity_curve(&small(), 1, 0.35, &[4, 2], Occlusion::Noise { amplitude: 0.1 }).is_err()
        );
        assert!(matches!(
            capacity_curve(&small(), 1, 0.35, &[1, 99], Occlusion::Noise { amplitude: 0.1 }),
            Err(OlfactionError::Full { max: 24 })
        ));
    }

    // ---- determinism and the bill ----

    /// Same parameters, same odours, same spikes — and no hidden state carried between calls, which
    /// is the failure that makes a result depend on the order the tests happened to run in.
    #[test]
    fn the_same_parameters_give_the_same_spike_train() {
        let build = || {
            let mut src = OdourGenerator::new(55, 32, 0.35).unwrap();
            let mut epl = Epl::new(small()).unwrap();
            let library = src.library(3).unwrap();
            for o in &library {
                epl.learn(o).unwrap();
            }
            let probe = src.next_odour().unwrap();
            let shown = epl.present(&probe, 5).unwrap();
            (shown, probe, epl)
        };
        let (a, probe_a, mut epl_a) = build();
        let (b, probe_b, _) = build();
        assert_eq!(probe_a, probe_b);
        assert_eq!(a, b, "two identical builds diverged");
        assert!(!a.train.spikes().is_empty());
        // Re-presenting the same odour to the same circuit gives the same answer: `present` carries
        // no state across calls beyond the learned ensembles.
        let again = epl_a.present(&probe_a, 5).unwrap();
        assert_eq!(a.mitral, again.mitral);
        assert_eq!(a.train, again.train);
    }

    /// The loop's traffic is counted, and the ledger still refuses to price it — which is this
    /// crate's position, applied to the one application where a spiking network has a published win.
    #[test]
    fn the_ledger_counts_the_loop_and_still_refuses_to_price_it() {
        let mut src = OdourGenerator::new(63, 32, 0.35).unwrap();
        let mut epl = Epl::new(small()).unwrap();
        for o in &src.library(4).unwrap() {
            epl.learn(o).unwrap();
        }
        epl.recall(&src.next_odour().unwrap()).unwrap();
        assert!(epl.ledger.syn_ops > 0);
        // This implementation fetches a weight per delivery, so the two counts are equal BY
        // CONSTRUCTION, and a design that cached would make them differ.
        assert_eq!(epl.ledger.syn_ops, epl.ledger.syn_fetches);
        assert!(epl.ledger.spikes_out > 0);
        let idle = epl.ledger.idle_fraction().unwrap();
        assert!(idle > 0.0 && idle < 1.0, "idle fraction {idle}");

        // ⛔ THE COUNTS, RECOMPUTED FROM THE TRAIN. `syn_ops > 0` and `syn_ops == syn_fetches` (a
        // tautology: both lines add the same expression) let the whole granule→mitral direction
        // of the loop be deleted — 17.4% of the traffic — and the mitral→granule direction be
        // halved, with 24 tests green. AGENTS.md invariant 9 names that `/ 2` as the failure that
        // flatters every energy figure. Every mitral spike reaches every granule cell and every
        // granule spike reaches every mitral cell, so from one presentation's train:
        //   syn_ops = mitral_spikes * (broad + ensembles * per) + granule_spikes * mitral.
        let p = small();
        epl.ledger = Ledger::default();
        let odour = src.next_odour().unwrap();
        let cycles = 6;
        let shown = epl.present(&odour, cycles).unwrap();
        let n = p.mitral as u32;
        let (mut mitral_spikes, mut granule_spikes) = (0u64, 0u64);
        for sp in shown.train.spikes() {
            if sp.source < n {
                mitral_spikes += 1;
            } else {
                granule_spikes += 1;
            }
        }
        assert!(mitral_spikes > 0 && granule_spikes > 0, "{mitral_spikes} {granule_spikes}");
        let n_granule = (p.broad_granule + epl.ensembles().len() * p.granule_per_odour) as u64;
        assert_eq!(epl.ensembles().len(), 4);
        assert_eq!(epl.ledger.syn_ops, mitral_spikes * n_granule + granule_spikes * u64::from(n));
        assert_eq!(epl.ledger.syn_fetches, epl.ledger.syn_ops);
        assert_eq!(epl.ledger.spikes_out, mitral_spikes + granule_spikes);
        // Every cell is updated on every tick of its phase, driven or idle.
        let per_cycle = epl.ticks_per_cycle() as u64;
        let mitral_ticks = (per_cycle as f64 * p.mitral_duty).round() as u64;
        let granule_ticks = per_cycle - mitral_ticks;
        assert_eq!(
            epl.ledger.neuron_updates(),
            cycles as u64 * (mitral_ticks * u64::from(n) + granule_ticks * n_granule)
        );
        // Mitral cells integrate a non-zero current on every tick (i_bias > 0), so every mitral
        // update is driven; the idle updates are granule cells below threshold.
        assert!(p.i_bias > 0.0);
        assert!(epl.ledger.neuron_updates_driven >= cycles as u64 * mitral_ticks * u64::from(n));
        assert!(epl.ledger.neuron_updates_idle > 0);
        let bill = epl.ledger.bill(&LOIHI_2018);
        assert!(bill.total.is_none(), "the ledger priced a workload it cannot price");
        assert!(bill.unpriced.contains(&"synapse memory fetch"), "{:?}", bill.unpriced);
    }

    // ---- the generator and the occlusion models ----

    #[test]
    fn the_overlap_generator_hits_its_target_or_says_why_not() {
        let mut src = OdourGenerator::new(77, 32, 0.35).unwrap();
        for target in [0.4f64, 0.55, 0.7, 0.9, 0.99] {
            let (x, y, got) = src.pair_with_overlap(target).unwrap();
            assert!((got - target).abs() < 1e-6, "target {target}, got {got}");
            // And the returned pair really has that overlap when measured from scratch.
            let check = cosine_similarity(x.channels(), y.channels()).unwrap();
            assert!((check - got).abs() < 1e-12);
            assert_eq!(x.len(), 32);
        }
        // A sparse generator CAN draw two odours with disjoint supports, whose cosine is exactly
        // zero, so zero overlap is reachable there — and the retry loop finds it.
        let (_, _, zero) = src.pair_with_overlap(0.0).unwrap();
        assert!(zero < 1e-9, "sparse draws should reach near-zero overlap, got {zero}");

        // A dense generator cannot: with every channel active and every value in [0.2, 1], two
        // independent draws already overlap by about 0.95, and asking for less has no answer.
        let mut dense = OdourGenerator::new(78, 32, 1.0).unwrap();
        match dense.pair_with_overlap(0.5) {
            Err(OlfactionError::Unreachable { requested, floor }) => {
                assert_eq!(requested, 0.5);
                assert!(floor > 0.5 && floor < 1.0, "reported floor {floor}");
            }
            other => panic!("expected a refusal, got {other:?}"),
        }
        // And a target above that floor still works on the same generator.
        let (_, _, got) = dense.pair_with_overlap(0.99).unwrap();
        assert!((got - 0.99).abs() < 1e-6, "{got}");
        assert!(src.pair_with_overlap(1.5).is_err());
        assert!(src.pair_with_overlap(f64::NAN).is_err());
        assert!(OdourGenerator::new(1, 0, 0.5).is_err());
        assert!(OdourGenerator::new(1, 4, 0.0).is_err());
        assert!(OdourGenerator::new(1, 4, 1.5).is_err());
    }

    #[test]
    fn occlusion_refuses_what_it_cannot_do_and_does_what_it_can() {
        let mut src = OdourGenerator::new(4, 32, 0.35).unwrap();
        let a = src.next_odour().unwrap();
        let b = src.next_odour().unwrap();
        let mut rng = Rng::new(9);
        // A fraction outside [0, 1], and a NaN amplitude.
        assert!(Occlusion::Dropout { fraction: 1.5 }.apply(&a, &b, &mut rng).is_err());
        assert!(Occlusion::Dropout { fraction: f64::NAN }.apply(&a, &b, &mut rng).is_err());
        assert!(Occlusion::Noise { amplitude: f64::NAN }.apply(&a, &b, &mut rng).is_err());
        // Width mismatch names both numbers.
        let narrow = Odour::new(vec![0.4; 8]).unwrap();
        assert_eq!(
            Occlusion::Interferent { fraction: 0.5 }.apply(&a, &narrow, &mut rng),
            Err(OlfactionError::Width { expected: 32, found: 8 })
        );
        // Total dropout leaves nothing, and that refuses rather than returning a zero odour a
        // cosine would then have to invent an answer for.
        assert_eq!(
            Occlusion::Dropout { fraction: 1.0 }.apply(&a, &b, &mut rng),
            Err(OlfactionError::ZeroOdour)
        );
        // Zero corruption is the identity for both masking models.
        assert_eq!(Occlusion::Dropout { fraction: 0.0 }.apply(&a, &b, &mut rng).unwrap(), a);
        assert_eq!(Occlusion::Interferent { fraction: 0.0 }.apply(&a, &b, &mut rng).unwrap(), a);
        // Full interferent replacement is the background, exactly.
        assert_eq!(Occlusion::Interferent { fraction: 1.0 }.apply(&a, &b, &mut rng).unwrap(), b);
        // Noise stays non-negative and moves something.
        let noisy = Occlusion::Noise { amplitude: 0.3 }.apply(&a, &b, &mut rng).unwrap();
        assert!(noisy.channels().iter().all(|v| *v >= 0.0));
        assert_ne!(noisy, a);
        assert_eq!(Occlusion::Noise { amplitude: 0.0 }.apply(&a, &b, &mut rng).unwrap(), a);
    }

    /// ⛔ THE ODOUR-SPECIFIC LIMB WITH MORE THAN ONE ENSEMBLE. The occlusion test above stores
    /// exactly one odour, and with one ensemble `ensembles[0]`, `ensembles[n - 1 - k]` and a
    /// probe-independent mean of all masks are the same expression — so recall could deliver the
    /// WRONG odour's mask, and 24 tests stayed green while the circuit measurably changed.
    ///
    /// Spike counts cannot see it: the suppression is mostly "kill the resting spike on every
    /// inactive channel", which any mask does. The delivered inhibition can. With the broad limb
    /// off it is `i_specific` times a convex combination of the stored masks, and the weight `w`
    /// on the probed odour's own mask is read off by projection onto `mask_a - mask_b`. Probing A
    /// must put most of the weight on A's mask; probing B, on B's. Measured on the unmutated code:
    /// 0.74 and 0.23 — not 1 and 0, because the other odour's ensemble is partly recruited (its
    /// drive is 0.6, above the lower ensemble thresholds), which is what the doc means by "a blur
    /// of several masks". The band is 0.6 / 0.4: `ensembles[0]` for every probe gives 1.0 twice and
    /// fails B; the reversed pairing gives 0.26 and 0.74 and fails both; the unweighted mean gives
    /// 0.5 twice and fails both.
    #[test]
    fn recall_delivers_the_probed_odours_mask_and_not_anothers() {
        let p = EplParams { i_broad: 0.0, ..small() };
        assert!(p.i_specific > 0.0);
        let mut src = OdourGenerator::new(91, p.mitral, 0.35).unwrap();
        let (a, b) = (src.next_odour().unwrap(), src.next_odour().unwrap());
        let mut epl = Epl::new(p.clone()).unwrap();
        epl.learn(&a).unwrap();
        epl.learn(&b).unwrap();
        let mask_a = epl.ensembles()[0].mask.clone();
        let mask_b = epl.ensembles()[1].mask.clone();
        let diff: Vec<f64> = (0..p.mitral).map(|m| mask_a[m] - mask_b[m]).collect();
        let diff_sq: f64 = diff.iter().map(|d| d * d).sum();
        assert!(diff_sq > 0.5, "the fixture's masks nearly coincide: |diff|^2 = {diff_sq}");
        let weight_on_a = |epl: &mut Epl, odour: &Odour| -> f64 {
            let shown = epl.present(odour, 6).unwrap();
            let total: f64 = shown.inhibition.iter().sum();
            assert!(total > 0.0, "no specific inhibition was delivered at all");
            // inh / i_specific = w * mask_a + (1 - w) * mask_b  =>  project onto (mask_a - mask_b).
            let num: f64 = (0..p.mitral)
                .map(|m| (shown.inhibition[m] / p.i_specific - mask_b[m]) * diff[m])
                .sum();
            num / diff_sq
        };
        let w_a = weight_on_a(&mut epl, &a);
        let w_b = weight_on_a(&mut epl, &b);
        println!("weight on A's mask: probing A {w_a:.3}, probing B {w_b:.3}");
        assert!(w_a > 0.6, "probing A put weight {w_a} on A's mask");
        assert!(w_b < 0.4, "probing B put weight {w_b} on A's mask");
        // And the readout is a real convex combination, not a projection artefact: both weights
        // sit inside [0, 1] to rounding.
        assert!((-1e-9..=1.0 + 1e-9).contains(&w_a) && (-1e-9..=1.0 + 1e-9).contains(&w_b), "{w_a} {w_b}");
    }

    /// ⛔ `recall` scores beside the circuit, and needs no ensemble spike to do it. With every
    /// ensemble threshold set to 50 — cosines never exceed 1 — no ensemble cell can fire, the
    /// specific limb is off, and clean one-shot recall still identifies every stored odour. The
    /// module doc used to say the score "is not computed beside the circuit".
    #[test]
    fn recall_scores_beside_the_circuit_and_does_not_need_the_ensembles_to_fire() {
        let p = EplParams { ensemble_theta_lo: 50.0, ensemble_theta_hi: 50.0, ..small() };
        let mut src = OdourGenerator::new(5, p.mitral, 0.35).unwrap();
        let mut epl = Epl::new(p.clone()).unwrap();
        let lib = src.library(4).unwrap();
        for o in &lib {
            epl.learn(o).unwrap();
        }
        for (i, o) in lib.iter().enumerate() {
            assert_eq!(epl.recall(o).unwrap().identified, Some(i));
        }
        let shown = epl.present(&lib[0], 3).unwrap();
        let first_ensemble_source = (p.mitral + p.broad_granule) as u32;
        let ens_spikes = shown.train.spikes().iter().filter(|s| s.source >= first_ensemble_source).count();
        assert_eq!(ens_spikes, 0, "an ensemble cell fired through a threshold of 50");
        assert!(shown.granule_drive.iter().all(|d| *d < 1.0 + 1e-12));
    }

    /// ⛔ A mitral cell with its threshold below rest used to be accepted, and identified every
    /// odour in the world as odour 0 at a cosine of 0.9999999999999996. Now it is refused by name,
    /// as the granule cell always was.
    #[test]
    fn a_mitral_cell_that_cannot_fire_properly_is_refused_by_name() {
        let bad = |cell: Lif| Epl::new(EplParams { mitral_cell: cell, ..small() });
        for cell in [
            Lif { v_th: -80e-3, t_ref: 0.5e-3, ..Lif::default() },
            Lif { tau_m: f64::NAN, t_ref: 0.5e-3, ..Lif::default() },
            Lif { tau_m: -4e-3, t_ref: 0.5e-3, ..Lif::default() },
            Lif { r_m: 0.0, t_ref: 0.5e-3, ..Lif::default() },
        ] {
            assert!(
                matches!(bad(cell), Err(OlfactionError::Parameter { field: "mitral_cell", .. })),
                "{cell:?} was accepted"
            );
        }
        assert!(Epl::new(small()).is_ok());
    }

    /// The generator refuses instead of hanging: at a sparsity of 1e-18 the first version never
    /// returned. And `pair_with_overlap`'s bounded-retry promise now holds one call deep.
    #[test]
    fn an_impossible_sparsity_is_a_refusal_not_a_hang() {
        let mut g = OdourGenerator::new(1, 64, 1e-18).unwrap();
        assert_eq!(
            g.next_odour().unwrap_err(),
            OlfactionError::NoActiveChannel { tries: OdourGenerator::MAX_TRIES }
        );
        assert!(g.library(3).is_err());
        assert!(g.pair_with_overlap(0.5).is_err());
        let e = OlfactionError::NoActiveChannel { tries: 4096 };
        assert!(e.to_string().contains("4096"));
        // A sensible sparsity never comes near the bound.
        let mut g = OdourGenerator::new(1, 4, 0.05).unwrap();
        assert!(g.library(200).is_ok());
    }

    /// ⛔ `capacity_curve` is a function of `(params, seed, k)` and not of which probe points were
    /// asked for. The first version drew the interfering backgrounds from the same generator as
    /// the odours to learn, so an extra probe point changed every odour learned after it.
    #[test]
    fn the_capacity_curve_does_not_depend_on_the_probe_schedule() {
        let p = small();
        let occ = Occlusion::Interferent { fraction: 0.6 };
        let many = capacity_curve(&p, 3, 0.35, &[2, 4, 8, 16, 24], occ).unwrap();
        let one = capacity_curve(&p, 3, 0.35, &[24], occ).unwrap();
        let (a, b) = (&many[many.len() - 1], &one[0]);
        assert_eq!((a.learned, b.learned), (24, 24));
        assert_eq!(a.accuracy, b.accuracy);
        assert_eq!(a.mean_self, b.mean_self);
        assert_eq!(a.mean_best_other, b.mean_best_other);
    }

    /// ⛔ `granule_ceiling()` at its CALL SITE. The accessor is pinned exactly (12), but `+ 1.0` at
    /// the one place it is used rescaled every delivered inhibition by 12/13 with 24 tests green.
    /// A broad pool driven flat out — thresholds at zero, a gain that saturates every cell at its
    /// refractory ceiling — has `a_broad = 1` exactly, so the inhibition every mitral cell
    /// integrates is `i_broad` to the last bit. Under the rescaling it is `12/13 · i_broad`.
    #[test]
    fn a_saturated_broad_pool_delivers_exactly_i_broad() {
        let p = EplParams {
            broad_theta_lo: 0.0,
            broad_theta_hi: 0.0,
            granule_gain: 1e-6,
            i_specific: 0.0,
            ..small()
        };
        assert!(p.i_broad > 0.0);
        let mut epl = Epl::new(p.clone()).unwrap();
        let odour = OdourGenerator::new(9, p.mitral, 0.35).unwrap().next_odour().unwrap();
        let cycles = 3;
        let shown = epl.present(&odour, cycles).unwrap();
        // `a_broad` recomputed from the train: the broad pool's spikes in the cycle BEFORE the
        // final one (that is what the final cycle integrates), over `n_broad · granule_ceiling()`
        // with the accessor's value typed in here. A saturated cell fires every `t_ref + dt`
        // rather than every `t_ref` — the tick after the refractory period ends is spent
        // reaching threshold — so `a_broad` is 9/12 here, not 1, and the point is the DENOMINATOR:
        // under `granule_ceiling() + 1.0` at the call site the code divides by 13 while this test
        // divides by 12.
        let per_cycle = epl.ticks_per_cycle() as u64;
        let (from, to) = (per_cycle * (cycles as u64 - 2), per_cycle * (cycles as u64 - 1));
        let broad_lo = p.mitral as u32;
        let broad_hi = (p.mitral + p.broad_granule) as u32;
        let broad_spikes = shown
            .train
            .spikes()
            .iter()
            .filter(|s| s.t >= from && s.t < to && s.source >= broad_lo && s.source < broad_hi)
            .count();
        let a_broad = broad_spikes as f64 / (p.broad_granule as f64 * epl.granule_ceiling());
        assert!(a_broad > 0.5, "the pool was not driven hard: a_broad {a_broad}");
        assert!(a_broad <= 1.0, "more spikes than the ceiling allows: {a_broad}");
        for (m, inh) in shown.inhibition.iter().enumerate() {
            let want = p.i_broad * a_broad;
            assert!(
                (inh - want).abs() <= 1e-15 * want,
                "cell {m}: integrated {inh:e} against i_broad * a_broad = {want:e}"
            );
        }
    }
}
