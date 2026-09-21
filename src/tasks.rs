//! Teaching tasks: benchmark problems **generated**, never downloaded.
//!
//! A benchmark is a claim about what a model can do, and a claim is only worth as much as the
//! reader's ability to re-run it. The standard spiking-network datasets — `N-MNIST`, `DVS-Gesture`,
//! `SHD` — are real recordings, which is their strength and also the reason they cannot live in a
//! zero-dependency library: they are hundreds of megabytes, they are under licences that forbid
//! redistribution, and fetching them needs a network stack this crate does not have. A student on a
//! train, a CI runner with no egress, and a browser tab compiled to `wasm32` all fail the same way.
//!
//! So this module **generates** the equivalents. Every task here is a closed-form recipe plus a
//! seed: the same seed gives the same spikes on every platform and every release, the whole corpus
//! is a few kilobytes of state, and — the part that matters for teaching — each task isolates
//! **one** mechanism, so a failure names the mechanism that failed.
//!
//! # What that buys and what it costs
//!
//! It buys reproducibility, size, and diagnosis. A synthetic task has a *known* structure, so when
//! a network scores 0.62 you can ask which part of the structure it found. Real data cannot be
//! interrogated that way, because nobody knows what is in it.
//!
//! It costs realism, and the cost is not small. Each task below carries a `stands_in_for` field
//! naming the real dataset it substitutes for and a `not_captured` field naming, plainly, what it
//! throws away. A result on [`MovingBar`] is not a result on `DVS-Gesture`. Nothing in this module
//! should appear in a paper's results table as though it were. These are **teaching instruments**,
//! and the honest use of one is to show that a mechanism works at all before spending a week of
//! compute finding out that it does not.
//!
//! # The baseline travels with the benchmark
//!
//! Every [`Dataset`] carries [`Dataset::chance`] and [`Dataset::majority_baseline`] **as data**.
//! This is not decoration. The single most common way a spiking-network result gets misread is a
//! number reported without the floor it has to clear: 78% on a 2-class task with a 74% majority
//! class is almost nothing, and 78% on a 20-class balanced task is a great deal. A benchmark whose
//! baseline is not printed beside it is a benchmark that will be misread, so here the baseline is a
//! field on the struct and cannot be left out of the table.
//!
//! [`Dataset::majority_baseline`] is an `Option` for one case only: an empty test split, where
//! there is no majority class to measure and a `0.0` in that column would read as a floor *below*
//! chance. Every task in this module at its defaults reports `Some`.
//!
//! # Leakage is assumed until it is measured
//!
//! Train and test splits here are disjoint **by construction**: every sample is generated, keyed by
//! its *input* (not its label), and rejected if that exact input has been produced before.
//! [`Dataset::overlap`] recomputes the intersection so a test can assert it is zero, and every task
//! in this module has such a test. The identity is the input alone on purpose — the same input
//! appearing under two labels is a contradiction in the task, not leakage, and conflating the two
//! hides both.
//!
//! When a task's parameters cannot produce that many distinct inputs — a temporal `XOR` with zero
//! jitter has exactly four — generation returns [`TaskError::Exhausted`] naming how many distinct
//! inputs it actually found. It does not quietly hand back duplicates.
//!
//! # The one task to read first
//!
//! [`TemporalXor`]. It is the canonical demonstration that a **rate code throws away the answer**:
//! the two classes are built to carry *identical spike counts on every channel*, so any readout
//! that counts spikes — however many layers deep — is reading a quantity that is the same for both
//! classes. The information is in the *coincidence* of two channels, and a coincidence detector is
//! a neuron with a short membrane time constant. That is the entire argument for spiking
//! computation compressed into one task with four conditions, and its doc works through it.
//!
//! ```
//! use ferromorphic::tasks::{Split, TemporalXor};
//!
//! let d = TemporalXor::default().generate()?;
//! assert_eq!(d.chance, 0.5);
//! // The rate code is blind: per-channel spike totals are EXACTLY equal across the two classes.
//! assert_eq!(d.channel_counts(Split::Test, 0), d.channel_counts(Split::Test, 1));
//! assert_eq!(d.overlap(), 0);
//! # Ok::<(), ferromorphic::tasks::TaskError>(())
//! ```
//!
//! # Provenance
//!
//! The task designs are standard and old. Temporal `XOR` as the separator between rate and timing
//! codes is the framing in Maass, *Networks of Spiking Neurons: The Third Generation of Neural
//! Network Models*, Neural Networks 10(9):1659-1671, 1997, and is the standard demonstration
//! problem in the surrogate-gradient literature. Coincidence detection as the elementary spiking
//! primitive is Abeles, *Corticonics*, Cambridge University Press, 1991. Delayed match-to-sample is
//! Fuster and Alexander, Science 173:652-654, 1971. Rate discrimination and its Poisson optimum are
//! the psychophysical staple; the optimal-observer derivation reproduced in
//! [`RateDiscrimination::poisson_optimal_accuracy`] is elementary and is worked in Dayan and
//! Abbott, *Theoretical Neuroscience*, MIT Press, 2001, chapter 3. The real datasets these stand in
//! for are Orchard et al., Frontiers in Neuroscience 9:437, 2015 (`N-MNIST`), Amir et al., CVPR
//! 2017 (`DVS-Gesture`), and Cramer et al., IEEE Trans. Neural Netw. Learn. Syst. 33(7), 2022
//! (`SHD`). This review did not locate a generated, dependency-free stand-in corpus of this kind in
//! `SpikingJelly`, `snnTorch` or `NeuroBench`; those projects download the recordings.

use crate::rng::Rng;
use crate::spike::{Event, Polarity, Spike, Train};

// ---------------------------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------------------------

/// Why a task could not be generated.
///
/// Every variant names the offending quantity. A task that could not be built is never silently
/// replaced by a smaller or easier one: a benchmark that quietly shrank is worse than no benchmark,
/// because the number it produces still looks like a number.
#[derive(Debug, Clone, PartialEq)]
pub enum TaskError {
    /// A parameter was not a finite number.
    ///
    /// Rejected at the boundary rather than allowed through, because a `NaN` rate or a `NaN` speed
    /// produces a task that generates zero spikes and a classifier that reports chance — a result
    /// that looks like a modelling failure rather than a typo.
    NotFinite {
        /// Name of the parameter, as it is spelled on the config struct.
        what: &'static str,
        /// The value that was rejected.
        value: f64,
    },
    /// A count that must be at least one was zero.
    Empty {
        /// Name of the count, as it is spelled on the config struct.
        what: &'static str,
    },
    /// A parameter was finite but outside the range the task is defined on.
    OutOfRange {
        /// Name of the parameter, as it is spelled on the config struct. It is the field to
        /// change, which is not always the field the number below came from: a range check on
        /// `max_gap_ticks - threshold_ticks` names `max_gap_ticks`, because that is the one a
        /// caller can move.
        what: &'static str,
        /// The offending value: the parameter itself where the constraint is on it directly, and
        /// the derived quantity the constraint is actually on — a difference, a sum, a jittered
        /// extreme — where it is not. The `# Errors` section of each generator says which.
        value: f64,
        /// Smallest acceptable value, inclusive.
        low: f64,
        /// Largest acceptable value, inclusive.
        high: f64,
    },
    /// Generation ran out of draws before it found enough **distinct** inputs.
    ///
    /// The usual cause is a jitter or a window too small for the requested sample count: a task
    /// with `k` reachable input patterns cannot fill a disjoint split of more than `k` samples,
    /// however long it draws. The fix is more jitter, a longer window, or fewer samples — and the
    /// numbers here say which.
    Exhausted {
        /// Distinct samples this class needed.
        wanted: usize,
        /// Distinct samples actually found before the draw budget ran out.
        distinct: usize,
        /// Draws spent.
        draws: usize,
    },
}

impl core::fmt::Display for TaskError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotFinite { what, value } => write!(f, "{what} is not finite ({value})"),
            Self::Empty { what } => write!(f, "{what} must be at least 1, was 0"),
            Self::OutOfRange { what, value, low, high } => {
                write!(f, "{what} = {value} is outside [{low}, {high}]")
            }
            Self::Exhausted { wanted, distinct, draws } => write!(
                f,
                "wanted {wanted} distinct inputs per class, found {distinct} in {draws} draws; \
                 increase the jitter or the window, or ask for fewer samples"
            ),
        }
    }
}

/// So that `?` works in a caller whose error type is `Box<dyn Error>`, which is what every example
/// and doctest in this crate uses.
impl std::error::Error for TaskError {}

fn finite(what: &'static str, value: f64) -> Result<(), TaskError> {
    // `!is_finite()` rather than a comparison: the comparison forms clippy suggests here accept NaN.
    if !value.is_finite() {
        return Err(TaskError::NotFinite { what, value });
    }
    Ok(())
}

fn in_range(what: &'static str, value: f64, low: f64, high: f64) -> Result<(), TaskError> {
    finite(what, value)?;
    if value < low || value > high {
        return Err(TaskError::OutOfRange { what, value, low, high });
    }
    Ok(())
}

fn nonzero(what: &'static str, n: u64) -> Result<(), TaskError> {
    if n == 0 {
        return Err(TaskError::Empty { what });
    }
    Ok(())
}

/// Largest number of distinct values any uniform draw in this module may span.
///
/// Not `u32::MAX`, and the difference is measured rather than stylistic. [`Rng::below`] debiases by
/// rejection against `u32::MAX - (u32::MAX % n) - (n - 1)`, which is `(q - 1) * n + 2` acceptable
/// values out of `2^32` for `q = floor((2^32 - 1) / n)`. At `n = 2^31` and above, `q` is 1 and the
/// acceptance zone collapses to **two values in 2^32**: one draw of `n = 3_000_000_000` took
/// **14.1 seconds** on the machine this was written on, against 10 nanoseconds at `n = 2^31 - 1`,
/// where `q` is 2 and half of all draws are accepted. A task configured past this line does not
/// come back, so it is refused instead — a hang is not a result.
pub const MAX_DRAW_SPAN: u64 = i32::MAX as u64;

/// Every uniform draw in this module goes through [`Rng::below`], which takes a `u32`, while every
/// quantity a task is configured with is a `u64` of ticks. An `as u32` between the two is a silent
/// narrowing: a legal, fully validated config asking for gaps in `0..5_000_000_000` would draw them
/// from `0..705_032_696` and every assertion downstream would still pass, because a narrower range
/// is still inside the wider one. Worse, a span that lands on a multiple of `2^32` narrows to zero
/// and `below(0)` *panics* — out of a `Result`-returning constructor.
///
/// So the narrowing is done once, here, at the boundary, against [`MAX_DRAW_SPAN`], and it reports
/// [`TaskError::OutOfRange`] instead of truncating. `what` is the config field a user would have to
/// change.
fn draw_span(what: &'static str, span: u64) -> Result<u32, TaskError> {
    if span > MAX_DRAW_SPAN {
        return Err(TaskError::OutOfRange {
            what,
            value: span as f64,
            low: 1.0,
            high: MAX_DRAW_SPAN as f64,
        });
    }
    Ok(span as u32)
}

/// Largest jitter half-width a task may ask for, in ticks.
///
/// A half-width `h` draws from `2h + 1` values, so this is the largest `h` whose span still fits
/// inside [`MAX_DRAW_SPAN`].
pub const MAX_JITTER_HALF: u64 = (MAX_DRAW_SPAN - 1) / 2;

/// A jitter half-width that has been **checked** to admit a `2 * half + 1` uniform draw.
///
/// A newtype rather than a plain `u32` so that the check cannot be skipped: [`jitter`] takes only
/// this, and the only way to build one is [`half_width`], which refuses anything larger.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Half(u32);

fn half_width(what: &'static str, half: u64) -> Result<Half, TaskError> {
    if half > MAX_JITTER_HALF {
        return Err(TaskError::OutOfRange {
            what,
            value: half as f64,
            low: 0.0,
            high: MAX_JITTER_HALF as f64,
        });
    }
    Ok(Half(half as u32))
}

// ---------------------------------------------------------------------------------------------
// Samples and datasets
// ---------------------------------------------------------------------------------------------

/// Which half of a [`Dataset`] a query is about.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Split {
    /// The samples a model is allowed to fit.
    Train,
    /// The samples a model is scored on, and which it must never have seen.
    Test,
}

/// A labelled spike-train sample.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sample {
    /// The input spikes, sorted by `(t, source)`. `source` is the input channel index.
    pub train: Train,
    /// Class index in `0..n_classes`.
    pub label: u32,
}

/// A labelled event-stream sample, for the tasks whose inputs carry a polarity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EventSample {
    /// The input events, sorted by `(t, address, polarity)`. `address` is `y * width + x`.
    pub events: Vec<Event>,
    /// Class index in `0..n_classes`.
    pub label: u32,
}

/// Anything a [`Dataset`] can count classes over.
pub trait Labelled {
    /// Class index of this sample, in `0..n_classes`.
    fn label(&self) -> u32;
}

/// A canonical identity for a sample's **input**, used to prove train and test are disjoint.
///
/// The label is deliberately excluded. Leakage is a statement about inputs: the same input under
/// two labels is a contradiction in the task definition, and including the label in the key would
/// let such a pair sit on both sides of the split while the check reported a clean separation.
pub trait Keyed {
    /// A total, order-comparable encoding of the input. Two samples with equal keys are the same
    /// input; two with different keys are different inputs. No hashing, so there are no collisions.
    fn key(&self) -> Vec<u64>;
}

impl Labelled for Sample {
    fn label(&self) -> u32 {
        self.label
    }
}

impl Keyed for Sample {
    fn key(&self) -> Vec<u64> {
        let mut k = Vec::with_capacity(self.train.len() * 2);
        for s in self.train.spikes() {
            k.push(s.t);
            k.push(u64::from(s.source));
        }
        k
    }
}

impl Labelled for EventSample {
    fn label(&self) -> u32 {
        self.label
    }
}

impl Keyed for EventSample {
    fn key(&self) -> Vec<u64> {
        let mut k = Vec::with_capacity(self.events.len() * 2);
        for e in &self.events {
            k.push(e.t);
            // The polarity is part of the input and goes in the key: two streams that agree on
            // every `(t, address)` but disagree on a sign are different stimuli, and a key that
            // dropped the bit would call them the same input and reject one as a duplicate.
            //
            // What this bit does NOT do, stated because the comment here used to claim it: it is
            // not what keeps a leftward sweep from colliding with a rightward one. Those two
            // already differ in `(t, address)` — their leading edges move in opposite directions —
            // so on the streams this module generates the bit changes no comparison at all. It is
            // part of the key because [`EventSample`] is a public type that any caller can fill,
            // and the identity of an event stream includes its signs whether or not the generator
            // here can produce a pair that needs it.
            k.push((u64::from(e.address) << 1) | u64::from(e.polarity == Polarity::On));
        }
        k
    }
}

/// A generated benchmark: two disjoint splits and everything needed to read a score off them.
#[derive(Debug, Clone, PartialEq)]
pub struct Dataset<S> {
    /// Short identifier, matching the [`TaskCard::name`] in [`CATALOGUE`].
    pub name: &'static str,
    /// Samples a model may fit. Shuffled, so a class-ordered stream cannot be mistaken for
    /// curriculum learning.
    pub train: Vec<S>,
    /// Samples a model is scored on. Disjoint from `train` by construction; see [`Dataset::overlap`].
    pub test: Vec<S>,
    /// Number of classes; labels run `0..n_classes`.
    pub n_classes: u32,
    /// Number of input channels (for an event task, `width * height` pixels).
    pub n_inputs: u32,
    /// Length of every sample in ticks. All samples in a dataset share it.
    pub ticks: u64,
    /// Tick length in **seconds**. `ticks * dt` is the sample duration; feed this same `dt` to
    /// [`crate::sim::Sim`] or the spike times mean something else.
    pub dt: f64,
    /// Accuracy of uniform guessing, `1 / n_classes`. The floor a result must clear to mean
    /// anything at all.
    pub chance: f64,
    /// Accuracy of always answering the most common class in the **test** split.
    ///
    /// Equal to [`Dataset::chance`] only when the test split is exactly balanced, which every task
    /// in this module arranges. It is stored separately anyway, because the moment a user changes
    /// the per-class counts the two diverge and the larger of them is the real floor.
    ///
    /// `None` when the test split is **empty**, for the same reason [`Dataset::balance`] returns an
    /// empty vector there: the majority class of no samples is not a quantity. It was previously an
    /// `f64` that reported `0.0` for that case — a baseline *below chance*, printed in the column a
    /// score is read against, for a split nobody measured. `Some` for every task in this module at
    /// its defaults, so the table the module doc promises is still a table.
    pub majority_baseline: Option<f64>,
    /// The seed that produced this dataset. Recorded so a figure can be regenerated from its
    /// caption alone.
    pub seed: u64,
    /// The real recorded dataset this substitutes for, named so a reader is never confused about
    /// which one a number came from.
    pub stands_in_for: &'static str,
    /// What this generator throws away relative to that recording. Read it before quoting a score.
    pub not_captured: &'static str,
}

impl<S> Dataset<S> {
    /// The samples of one split.
    #[must_use]
    pub fn split(&self, split: Split) -> &[S] {
        match split {
            Split::Train => &self.train,
            Split::Test => &self.test,
        }
    }

    /// Sample duration in **seconds**, `ticks * dt`.
    #[must_use]
    pub fn duration_seconds(&self) -> f64 {
        self.ticks as f64 * self.dt
    }
}

impl<S: Labelled> Dataset<S> {
    /// Number of samples of each class in one split, indexed by class.
    ///
    /// The vector has one bin per class and **no bin for anything else**, so a sample whose label
    /// is outside `0..n_classes` — which no generator here produces, but which a caller filling
    /// [`Dataset::train`] by hand can create — is counted nowhere and the counts sum to less than
    /// the split. [`Dataset::balance`] divides by the split length rather than by the counted
    /// total, so that shortfall shows up as shares that sum to under one instead of being
    /// normalised away.
    #[must_use]
    pub fn counts(&self, split: Split) -> Vec<usize> {
        let mut c = vec![0usize; self.n_classes as usize];
        for s in self.split(split) {
            let l = s.label() as usize;
            if l < c.len() {
                c[l] += 1;
            }
        }
        c
    }

    /// Fraction of one split belonging to each class, indexed by class.
    ///
    /// Empty when the split is empty: a share of an empty set is a division by zero, and returning
    /// zeros would report "perfectly imbalanced" for "nothing measured". The denominator is the
    /// split's length, so the shares sum to one exactly when every label is in range — see
    /// [`Dataset::counts`].
    #[must_use]
    pub fn balance(&self, split: Split) -> Vec<f64> {
        let n = self.split(split).len();
        if n == 0 {
            return Vec::new();
        }
        self.counts(split).into_iter().map(|c| c as f64 / n as f64).collect()
    }
}

impl<S: Keyed> Dataset<S> {
    /// How many test samples have an input that also appears in the train split.
    ///
    /// **Zero for every task in this module**, asserted by a test on each one. Exposed because a
    /// user who changes a generator needs to be able to re-check it, and because the number is the
    /// first thing to look at when a model scores implausibly well.
    ///
    /// Cost is `O(n log n)` in the split sizes, with exact key comparison rather than hashing, so
    /// a reported zero is a proof rather than an absence of collisions.
    #[must_use]
    pub fn overlap(&self) -> usize {
        let mut keys: Vec<Vec<u64>> = self.train.iter().map(Keyed::key).collect();
        keys.sort_unstable();
        self.test.iter().filter(|s| keys.binary_search(&s.key()).is_ok()).count()
    }
}

impl Dataset<Sample> {
    /// Total spikes on each input channel, summed over every sample of one class in one split.
    ///
    /// The instrument that makes [`TemporalXor`]'s central claim checkable: if this vector is
    /// identical for two classes, then **no readout that is a function of spike counts alone can
    /// separate them**, because it is being handed the same numbers either way.
    #[must_use]
    pub fn channel_counts(&self, split: Split, class: u32) -> Vec<u64> {
        let mut c = vec![0u64; self.n_inputs as usize];
        for s in self.split(split).iter().filter(|s| s.label == class) {
            for sp in s.train.spikes() {
                let i = sp.source as usize;
                if i < c.len() {
                    c[i] += 1;
                }
            }
        }
        c
    }

    /// Tick of the first spike on `channel` in `sample`, or `None` if that channel is silent.
    ///
    /// `None` rather than a sentinel tick: a silent channel has no latency, and a caller that
    /// substituted `ticks` for it would be feeding a classifier a measurement nobody made.
    #[must_use]
    pub fn first_spike(sample: &Sample, channel: u32) -> Option<u64> {
        sample.train.spikes().iter().find(|s| s.source == channel).map(|s| s.t)
    }
}

/// Total events of each polarity in an event sample. Order is `(on, off)`.
#[must_use]
pub fn polarity_counts(sample: &EventSample) -> (usize, usize) {
    let on = sample.events.iter().filter(|e| e.polarity == Polarity::On).count();
    (on, sample.events.len() - on)
}

// ---------------------------------------------------------------------------------------------
// Generation machinery
// ---------------------------------------------------------------------------------------------

/// Deterministic Fisher-Yates. Order matters for online learners, and a class-ordered stream is a
/// silent confound rather than a visible one.
fn shuffle<T>(v: &mut [T], rng: &mut Rng) {
    if v.len() < 2 {
        return;
    }
    for i in (1..v.len()).rev() {
        let j = rng.below((i + 1) as u32) as usize;
        v.swap(i, j);
    }
}

/// Fill an exactly-balanced, globally-distinct split.
///
/// `make(class, accepted)` is called until `per_class_train + per_class_test` samples with keys
/// never seen before — in **any** class — have been accepted. Class counts are therefore exact, and
/// the train/test split is disjoint by construction rather than by inspection.
///
/// The second argument is the number of samples of this class **already accepted**, not the number
/// of draws spent. The difference matters for a generator that cycles a condition on it — cycling
/// on the draw index makes the condition counts depend on how many duplicates rejection happened to
/// eat, which is a label-correlated imbalance nobody asked for; cycling on the accepted count makes
/// them exact. [`TemporalXor`] is the generator that does this, and its blindness claim rests on it.
fn collect_split<S, F>(
    n_classes: u32,
    per_class_train: usize,
    per_class_test: usize,
    shuffle_seed: u64,
    mut make: F,
) -> Result<(Vec<S>, Vec<S>), TaskError>
where
    S: Keyed,
    F: FnMut(u32, usize) -> S,
{
    nonzero("n_classes", u64::from(n_classes))?;
    let per_class = per_class_train.saturating_add(per_class_test);
    nonzero("samples per class", per_class as u64)?;
    // `shuffle` draws an index with `Rng::below`, which takes a `u32`, so a longer split would
    // shuffle only its first `MAX_DRAW_SPAN` entries. Refused rather than narrowed; such a split
    // would need hundreds of gigabytes of samples to exist in the first place, so this costs
    // nothing and closes the last `as u32` in the module. It also keeps the `with_capacity`
    // products below inside `usize`.
    draw_span("samples per class", (per_class as u64).saturating_mul(u64::from(n_classes)))?;

    // 64 draws per accepted sample plus a floor. Generous enough that a well-parameterised task
    // never trips it and tight enough that a degenerate one fails in milliseconds with a number.
    let budget = per_class.saturating_mul(64).saturating_add(1024);
    let mut seen: Vec<Vec<u64>> = Vec::new();
    let mut train = Vec::with_capacity(per_class_train * n_classes as usize);
    let mut test = Vec::with_capacity(per_class_test * n_classes as usize);

    for c in 0..n_classes {
        let mut kept: Vec<S> = Vec::with_capacity(per_class);
        let mut draws = 0usize;
        while kept.len() < per_class {
            if draws >= budget {
                return Err(TaskError::Exhausted { wanted: per_class, distinct: kept.len(), draws });
            }
            let s = make(c, kept.len());
            draws += 1;
            let k = s.key();
            if let Err(pos) = seen.binary_search(&k) {
                seen.insert(pos, k);
                kept.push(s);
            }
        }
        for (i, s) in kept.into_iter().enumerate() {
            if i < per_class_train {
                train.push(s);
            } else {
                test.push(s);
            }
        }
    }

    let mut rng = Rng::new(shuffle_seed);
    shuffle(&mut train, &mut rng);
    shuffle(&mut test, &mut rng);
    Ok((train, test))
}

/// Assemble the dataset and compute the two baselines from the data rather than from the intent.
fn finish<S: Labelled>(
    name: &'static str,
    train: Vec<S>,
    test: Vec<S>,
    n_classes: u32,
    n_inputs: u32,
    ticks: u64,
    dt: f64,
    seed: u64,
    stands_in_for: &'static str,
    not_captured: &'static str,
) -> Dataset<S> {
    let mut d = Dataset {
        name,
        train,
        test,
        n_classes,
        n_inputs,
        ticks,
        dt,
        chance: 1.0 / f64::from(n_classes),
        majority_baseline: None,
        seed,
        stands_in_for,
        not_captured,
    };
    // MEASURED off the test split, not assumed from the per-class counts requested. If a future
    // change to a generator unbalances a split, this number moves and the task's own balance test
    // fails, which is the outcome we want.
    //
    // `balance` returns an empty vector for an empty split, and folding that with `max` from a 0.0
    // seed would report a baseline of zero — a number below chance, for a measurement that was
    // never made. `None` instead, which a caller has to handle rather than print.
    let bal = d.balance(Split::Test);
    d.majority_baseline =
        if bal.is_empty() { None } else { Some(bal.into_iter().fold(0.0f64, f64::max)) };
    d
}

/// A uniform integer draw in `[-half, half]`, used for every jitter in this module.
///
/// Takes a [`Half`], which is a half-width that [`half_width`] has already proved is drawable:
/// `half.0 <= (MAX_DRAW_SPAN - 1) / 2`, so `2 * half.0 + 1 <= MAX_DRAW_SPAN` and the arithmetic
/// below can neither overflow nor narrow. That is why there is no `as` cast in it.
fn jitter(rng: &mut Rng, half: Half) -> i64 {
    if half.0 == 0 {
        return 0;
    }
    let span = 2 * half.0 + 1;
    i64::from(rng.below(span)) - i64::from(half.0)
}

// ---------------------------------------------------------------------------------------------
// Counting distributions, for the one task with an analytic optimum
// ---------------------------------------------------------------------------------------------

/// Largest count support either optimal-observer bound will build, in spikes.
///
/// Both bounds sum over an explicit probability-mass vector, so the support is also the allocation:
/// three `Vec<f64>` of this length is 240 MB at the ceiling, and a caller asking for a billion-tick
/// window would otherwise be asking for 24 GB and getting an abort instead of a `None`. The figure
/// is also where the summation is still trustworthy — see
/// [`RateDiscrimination::optimal_accuracy`], which measures what happens past it.
pub const MAX_COUNT_SUPPORT: u64 = 10_000_000;

/// `ln(k!)` for `k` in `0..=n`, by summation. Exact to floating-point noise and monotone, which is
/// what the probability-mass recursions below need; a `lgamma` approximation is neither.
fn ln_factorials(n: usize) -> Vec<f64> {
    let mut v = Vec::with_capacity(n + 1);
    let mut acc = 0.0f64;
    v.push(0.0);
    for k in 1..=n {
        acc += (k as f64).ln();
        v.push(acc);
    }
    v
}

/// Binomial probability mass for `k` in `0..=n`, computed in log space so that `(1-p)^n` cannot
/// underflow to zero for a long window and silently zero out the whole distribution.
fn binomial_pmf(n: usize, p: f64, lnf: &[f64]) -> Vec<f64> {
    let mut v = vec![0.0f64; n + 1];
    if p <= 0.0 {
        v[0] = 1.0;
        return v;
    }
    if p >= 1.0 {
        v[n] = 1.0;
        return v;
    }
    let (lp, lq) = (p.ln(), (1.0 - p).ln());
    for k in 0..=n {
        let l = lnf[n] - lnf[k] - lnf[n - k] + k as f64 * lp + (n - k) as f64 * lq;
        v[k] = l.exp();
    }
    v
}

/// Poisson probability mass for `k` in `0..=k_max`, in log space for the same reason.
fn poisson_pmf(mu: f64, k_max: usize, lnf: &[f64]) -> Vec<f64> {
    let mut v = vec![0.0f64; k_max + 1];
    if mu <= 0.0 {
        v[0] = 1.0;
        return v;
    }
    let lmu = mu.ln();
    for k in 0..=k_max {
        v[k] = (-mu + k as f64 * lmu - lnf[k]).exp();
    }
    v
}

/// `P(H > L) + 0.5 * P(H = L)` for two independent count distributions.
///
/// The accuracy of the optimal observer that answers "whichever channel counted more" and flips a
/// fair coin on a tie. The half-weight on ties is not a convention: a tie carries no evidence, so
/// half of them are right whatever the rule, and a rule that always broke ties toward one class
/// would score exactly this on a balanced set anyway.
fn win_probability(hi: &[f64], lo: &[f64]) -> f64 {
    let mut suffix = vec![0.0f64; hi.len() + 1];
    for k in (0..hi.len()).rev() {
        suffix[k] = suffix[k + 1] + hi[k];
    }
    let mut win = 0.0f64;
    let mut tie = 0.0f64;
    for (j, &pj) in lo.iter().enumerate() {
        if pj == 0.0 {
            continue;
        }
        win += pj * suffix.get(j + 1).copied().unwrap_or(0.0);
        tie += pj * hi.get(j).copied().unwrap_or(0.0);
    }
    win + 0.5 * tie
}

// ---------------------------------------------------------------------------------------------
// 1. Temporal XOR
// ---------------------------------------------------------------------------------------------

/// **Temporal `XOR`**: two channels, each carrying one bit in *when* it fires, and the answer is
/// their exclusive-or. The task a rate code provably cannot solve.
///
/// # The construction
///
/// Channel 0 and channel 1 each emit a volley of exactly `spikes_per_channel` spikes. Each volley
/// starts either **early** (logical 0, tick `early_tick`) or **late** (logical 1, tick
/// `late_tick`), plus a common-mode jitter drawn per volley. The label is the exclusive-or:
///
/// | channel 0 | channel 1 | label | `abs(t0 - t1)` |
/// |---|---|---|---|
/// | early | early | 0 | 0 |
/// | late  | late  | 0 | 0 |
/// | early | late  | 1 | `late - early` |
/// | late  | early | 1 | `late - early` |
///
/// # Why a rate code cannot solve it, exactly
///
/// Every condition emits the **same number of spikes on the same channels**: `spikes_per_channel`
/// on channel 0 and `spikes_per_channel` on channel 1, in all four rows. A rate readout — a spike
/// count over the window, per channel, however it is weighted or nonlinearly transformed
/// afterwards — receives the vector `(k, k)` for class 0 and the vector `(k, k)` for class 1. Those
/// are the same vector. No function of it can tell the classes apart, and the depth of the network
/// behind the readout changes nothing, because the information was destroyed at the readout.
///
/// The claim is stronger than "counts are equal". With the four conditions balanced, the
/// **per-channel early/late counts** are identical between classes: channel 0 is early in exactly
/// half the samples of class 0 and exactly half the samples of class 1, and so is channel 1. So a
/// per-channel *latency* readout — mean first-spike time, one number per channel — is reading a
/// quantity with the same expectation either way. [`TemporalXor::condition_sample`] builds the four
/// jitter-free conditions so a test can assert the exact version of that, and
/// `temporal_xor_is_blind_to_every_per_channel_readout` does;
/// `the_early_late_counts_per_channel_are_exactly_equal_in_both_classes` asserts it on the
/// generated splits, where the conditions are cycled on accepted samples so the counts are exact
/// rather than approximate.
///
/// What is **not** claimed, because it is not true: that the realised spike times of the two
/// classes are identical sample for sample. Each volley carries an independent jitter draw, so a
/// finite split's per-channel mean first-spike times differ between classes by that sampling noise.
/// Measured on the default configuration and seed: the two class means on channel 0 differ by 0.02
/// ticks on the train split and 0.83 on the test split, against a jitter half-width of 6, and the
/// best single threshold on channel 0 alone — chosen with hindsight over every tick — scores 0.55
/// rather than 0.50, the way a coin flipped 128 times lands a little off 64. The blindness is a
/// property of the distribution the labels were drawn from, and it is exact there; it is not a
/// property of one realisation, and no generator that jitters can make it one.
///
/// What is left is the **joint** quantity `abs(t0 - t1)`: zero for class 0, `late - early` for
/// class 1. A neuron whose membrane constant is short compared to `late - early` computes it — it
/// reaches threshold only when both volleys land inside one membrane window. That is a coincidence
/// detector, and the whole of [`crate::neuron::Lif`]'s `tau_m` is the knob that sets its tolerance.
///
/// # The separability guarantee
///
/// With jitter `j` ticks, class 0 gives `abs(t0 - t1) <= 2j` and class 1 gives
/// `abs(t0 - t1) >= (late - early) - 2j`. The two are separable by a threshold exactly when
/// `late - early > 4j`, and [`TemporalXor::coincidence_separable`] reports whether the
/// configuration satisfies it. [`TemporalXor::classify_by_coincidence`] is the resulting perfect
/// classifier — included so that "solvable by timing" is a runnable statement rather than an
/// assertion.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TemporalXor {
    /// Tick at which an "early" volley starts. Logical 0.
    pub early_tick: u64,
    /// Tick at which a "late" volley starts. Logical 1. Must exceed `early_tick`.
    pub late_tick: u64,
    /// Spikes per volley, identical for every condition — which is what makes the count uninformative.
    pub spikes_per_channel: u32,
    /// Ticks between successive spikes within one volley. Must be at least 1.
    pub burst_gap: u64,
    /// Half-width of the uniform common-mode jitter applied to a whole volley, in ticks.
    ///
    /// Applied to the volley rather than to each spike so that within-volley spacing is exact and
    /// two spikes of one volley can never land on the same tick. Zero leaves four distinct inputs
    /// in total, which is fewer than any split needs — [`TemporalXor::generate`] then returns
    /// [`TaskError::Exhausted`] rather than handing back duplicates.
    pub jitter_ticks: u64,
    /// Sample length in ticks.
    pub ticks: u64,
    /// Tick length in seconds.
    pub dt: f64,
    /// Train samples per class.
    pub per_class_train: usize,
    /// Test samples per class.
    pub per_class_test: usize,
    /// Seed. Same seed, same dataset, every platform.
    pub seed: u64,
}

impl Default for TemporalXor {
    /// 1 ms ticks over a 100 ms window; volleys of three spikes 2 ms apart, starting at 15 ms or
    /// 55 ms, jittered by up to 6 ms.
    ///
    /// Two numbers are load-bearing. The 40 ms gap against a 6 ms jitter satisfies
    /// `late - early > 4 * jitter`, so the coincidence classifier is exact and the task is solvable
    /// rather than merely hard. And the jitter sets how many distinct inputs exist at all — two
    /// conditions per class times `(2 * 6 + 1)^2 = 169` jitter pairs is 338, comfortably above the
    /// 192 samples per class asked for here. Shrink the jitter and `generate` will refuse rather
    /// than repeat itself.
    fn default() -> Self {
        Self {
            early_tick: 15,
            late_tick: 55,
            spikes_per_channel: 3,
            burst_gap: 2,
            jitter_ticks: 6,
            ticks: 100,
            dt: 1e-3,
            per_class_train: 128,
            per_class_test: 64,
            seed: 0x5EED_0001,
        }
    }
}

impl TemporalXor {
    /// The four input conditions and their labels, as `(channel 0 late, channel 1 late, label)`.
    #[must_use]
    pub fn conditions() -> [(bool, bool, u32); 4] {
        [(false, false, 0), (true, true, 0), (false, true, 1), (true, false, 1)]
    }

    /// Whether `late - early > 4 * jitter`, the condition under which a coincidence threshold
    /// separates the classes with no errors at all.
    #[must_use]
    pub fn coincidence_separable(&self) -> bool {
        self.late_tick > self.early_tick
            && self.late_tick - self.early_tick > 4 * self.jitter_ticks
    }

    /// One jitter-free sample for a chosen condition, for inspection and for teaching.
    ///
    /// `a` and `b` are the logical bits of channel 0 and channel 1: `false` is early, `true` is
    /// late. The label is `a != b`.
    #[must_use]
    pub fn condition_sample(&self, a: bool, b: bool) -> Sample {
        let mut sp = Vec::new();
        for (ch, late) in [(0u32, a), (1u32, b)] {
            let base = if late { self.late_tick } else { self.early_tick };
            for k in 0..u64::from(self.spikes_per_channel) {
                sp.push(Spike { t: base + k * self.burst_gap, source: ch });
            }
        }
        Sample { train: Train::from_spikes(sp), label: u32::from(a != b) }
    }

    /// The reference solver: threshold the absolute difference of the two channels' first spikes.
    ///
    /// `None` when either channel is silent, which this generator never produces but a corrupted
    /// or filtered stream might; guessing a class for an input with a missing channel would hide
    /// the corruption inside an accuracy figure.
    #[must_use]
    pub fn classify_by_coincidence(&self, s: &Sample) -> Option<u32> {
        let t0 = Dataset::first_spike(s, 0)?;
        let t1 = Dataset::first_spike(s, 1)?;
        let gap = t0.abs_diff(t1);
        let half = (self.late_tick - self.early_tick) / 2;
        Some(u32::from(gap > half))
    }

    /// Build the dataset.
    ///
    /// # Errors
    ///
    /// [`TaskError::Empty`] for a zero volley size, zero `burst_gap` or zero sample count;
    /// [`TaskError::OutOfRange`] when `late_tick <= early_tick`, when the jitter could push a
    /// volley outside `0..ticks`, when `jitter_ticks` exceeds [`MAX_JITTER_HALF`] (the largest
    /// half-width a uniform draw can cover in bounded time), or when `dt` is not positive and
    /// finite; [`TaskError::Exhausted`] when the jitter admits fewer distinct inputs than the split
    /// needs — which is always the case for `jitter_ticks == 0`, where only four inputs exist.
    pub fn generate(&self) -> Result<Dataset<Sample>, TaskError> {
        in_range("dt", self.dt, f64::MIN_POSITIVE, f64::MAX)?;
        nonzero("spikes_per_channel", u64::from(self.spikes_per_channel))?;
        nonzero("burst_gap", self.burst_gap)?;
        let jit = half_width("jitter_ticks", self.jitter_ticks)?;
        if self.late_tick <= self.early_tick {
            return Err(TaskError::OutOfRange {
                what: "late_tick",
                value: self.late_tick as f64,
                low: (self.early_tick + 1) as f64,
                high: self.ticks as f64,
            });
        }
        // Saturating, so that a volley long enough to overflow a `u64` reports the window error it
        // deserves instead of wrapping to a small number that passes the check below.
        let span = u64::from(self.spikes_per_channel - 1).saturating_mul(self.burst_gap);
        if self.early_tick < self.jitter_ticks {
            return Err(TaskError::OutOfRange {
                what: "early_tick",
                value: self.early_tick as f64,
                low: self.jitter_ticks as f64,
                high: self.late_tick as f64,
            });
        }
        let last = self.late_tick.saturating_add(span).saturating_add(self.jitter_ticks);
        if last >= self.ticks {
            return Err(TaskError::OutOfRange {
                what: "ticks",
                value: self.ticks as f64,
                low: (last + 1) as f64,
                high: f64::MAX,
            });
        }

        let mut rng = Rng::new(self.seed);
        let cfg = *self;
        let (train, test) = collect_split(
            2,
            self.per_class_train,
            self.per_class_test,
            self.seed ^ 0xA5A5_0001,
            |class, accepted| {
                // Conditions alternate on the ACCEPTED count rather than on a coin or on the draw
                // index. On the draw index the two rows of a class came out unequal by however many
                // duplicates rejection happened to eat — and since a row is `early` or `late`, that
                // imbalance is a per-channel mean first-spike time that differs between the
                // classes, which is precisely the readout this task's doc promises is blind. On the
                // accepted count the two rows are exactly `per_class / 2` each, in both classes,
                // and the promise holds on the generated data and not only on the four conditions.
                let second = accepted % 2 == 1;
                let (a, b) = match (class, second) {
                    (0, false) => (false, false),
                    (0, true) => (true, true),
                    (_, false) => (false, true),
                    (_, true) => (true, false),
                };
                let ja = jitter(&mut rng, jit);
                let jb = jitter(&mut rng, jit);
                let mut sp = Vec::with_capacity(2 * cfg.spikes_per_channel as usize);
                for (ch, late, off) in [(0u32, a, ja), (1u32, b, jb)] {
                    let base = if late { cfg.late_tick } else { cfg.early_tick };
                    let start = (base as i64 + off).max(0) as u64;
                    for k in 0..u64::from(cfg.spikes_per_channel) {
                        sp.push(Spike { t: start + k * cfg.burst_gap, source: ch });
                    }
                }
                Sample { train: Train::from_spikes(sp), label: class }
            },
        )?;

        Ok(finish(
            "temporal_xor",
            train,
            test,
            2,
            2,
            self.ticks,
            self.dt,
            self.seed,
            "the timing-code demonstration in the surrogate-gradient literature; there is no \
             recorded dataset for this, it is a constructed proof",
            "everything about real data: two channels, no noise spikes, no dropped spikes, and a \
             gap that is either 0 or one fixed value rather than a distribution",
        ))
    }
}

// ---------------------------------------------------------------------------------------------
// 2. Coincidence detection
// ---------------------------------------------------------------------------------------------

/// **Coincidence detection**: two spikes, one question — did they arrive within `threshold_ticks`
/// of each other?
///
/// The elementary spiking primitive (Abeles, *Corticonics*, 1991). A leaky integrate-and-fire
/// neuron computes it directly: two inputs each worth less than threshold sum to more than
/// threshold only if the first has not yet leaked away, so the tolerance window *is* the membrane
/// time constant. Set `tau_m` around `threshold_ticks * dt` and a single neuron solves this task.
///
/// Like [`TemporalXor`], both classes carry exactly one spike per channel, so the spike **count**
/// is identical across classes and the rate code is again blind. Unlike [`TemporalXor`] the answer
/// is a *graded* function of one interval, which makes this the right task to look at first when a
/// timing model is not learning at all: it has one degree of freedom and a known solution.
///
/// Chance is 0.5 and the splits are exactly balanced, so `majority_baseline` is 0.5 too.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Coincidence {
    /// Largest gap that still counts as coincident, in ticks. Label 1 is `gap <= threshold_ticks`.
    pub threshold_ticks: u64,
    /// Largest gap drawn at all. Must exceed `threshold_ticks` or class 0 is unreachable.
    pub max_gap_ticks: u64,
    /// Tick of the reference spike on channel 0, before jitter.
    pub first_tick: u64,
    /// Half-width of the uniform jitter on the reference spike, in ticks. It shifts **both**
    /// spikes, so it changes the absolute times without changing the gap — which is exactly what a
    /// task about intervals should be invariant to, and what a network reading absolute time will
    /// fail on.
    pub jitter_ticks: u64,
    /// Sample length in ticks.
    pub ticks: u64,
    /// Tick length in seconds.
    pub dt: f64,
    /// Train samples per class.
    pub per_class_train: usize,
    /// Test samples per class.
    pub per_class_test: usize,
    /// Seed.
    pub seed: u64,
}

impl Default for Coincidence {
    /// 1 ms ticks: coincident means within 8 ms, gaps run out to 40 ms, and the reference spike
    /// sits at 20 ms with a 15 ms common-mode jitter.
    ///
    /// The jitter is large on purpose. It is what makes the coincident class carry
    /// `9 * 31 = 279` distinct inputs — above the 192 per class asked for — and it is also what
    /// forces a solver to read the *interval* rather than the absolute arrival time, which a
    /// smaller jitter would let it get away with.
    fn default() -> Self {
        Self {
            threshold_ticks: 8,
            max_gap_ticks: 40,
            first_tick: 20,
            jitter_ticks: 15,
            ticks: 100,
            dt: 1e-3,
            per_class_train: 128,
            per_class_test: 64,
            seed: 0x5EED_0002,
        }
    }
}

impl Coincidence {
    /// The membrane time constant a single [`crate::neuron::Lif`] would need, in **seconds**, for
    /// its coincidence window to match this task's threshold.
    ///
    /// A rule of thumb, not a derivation: two equal inputs each `w` volts below threshold sum past
    /// it while `w * exp(-gap / tau)` still covers the deficit, so the usable window is a small
    /// multiple of `tau`. Reported as `threshold_ticks * dt` so a student has a starting value and
    /// knows it is a starting value.
    #[must_use]
    pub fn suggested_tau_m(&self) -> f64 {
        self.threshold_ticks as f64 * self.dt
    }

    /// The reference solver: the interval between the two channels' first spikes, thresholded.
    ///
    /// `None` when a channel is silent.
    #[must_use]
    pub fn classify_by_interval(&self, s: &Sample) -> Option<u32> {
        let t0 = Dataset::first_spike(s, 0)?;
        let t1 = Dataset::first_spike(s, 1)?;
        Some(u32::from(t0.abs_diff(t1) <= self.threshold_ticks))
    }

    /// Build the dataset.
    ///
    /// # Errors
    ///
    /// [`TaskError::OutOfRange`] when `max_gap_ticks <= threshold_ticks` (class 0 would be
    /// unreachable), when the jitter and the largest gap do not fit inside `0..ticks`, when either
    /// gap range or the jitter is too wide for a single uniform draw (`threshold_ticks + 1` and
    /// `max_gap_ticks - threshold_ticks` must not exceed [`MAX_DRAW_SPAN`], nor `jitter_ticks`
    /// [`MAX_JITTER_HALF`]), or when `dt` is not positive and finite; [`TaskError::Exhausted`]
    /// when the reachable `(gap, jitter)` combinations are fewer than the split needs.
    pub fn generate(&self) -> Result<Dataset<Sample>, TaskError> {
        in_range("dt", self.dt, f64::MIN_POSITIVE, f64::MAX)?;
        let jit = half_width("jitter_ticks", self.jitter_ticks)?;
        if self.max_gap_ticks <= self.threshold_ticks {
            return Err(TaskError::OutOfRange {
                what: "max_gap_ticks",
                value: self.max_gap_ticks as f64,
                low: (self.threshold_ticks + 1) as f64,
                high: f64::MAX,
            });
        }
        if self.first_tick < self.jitter_ticks {
            return Err(TaskError::OutOfRange {
                what: "first_tick",
                value: self.first_tick as f64,
                low: self.jitter_ticks as f64,
                high: self.ticks as f64,
            });
        }
        let last =
            self.first_tick.saturating_add(self.jitter_ticks).saturating_add(self.max_gap_ticks);
        if last >= self.ticks {
            return Err(TaskError::OutOfRange {
                what: "ticks",
                value: self.ticks as f64,
                low: last.saturating_add(1) as f64,
                high: f64::MAX,
            });
        }
        // The two gap ranges, narrowed once and checked, instead of an `as u32` per draw inside the
        // closure. `max_gap_ticks > threshold_ticks` is already established, so both are at least 1.
        let coincident_span = draw_span("threshold_ticks", self.threshold_ticks.saturating_add(1))?;
        let distant_span = draw_span("max_gap_ticks", self.max_gap_ticks - self.threshold_ticks)?;

        let mut rng = Rng::new(self.seed);
        let cfg = *self;
        let (train, test) = collect_split(
            2,
            self.per_class_train,
            self.per_class_test,
            self.seed ^ 0xA5A5_0002,
            |class, _| {
                let gap = if class == 1 {
                    // Coincident: 0 ..= threshold.
                    u64::from(rng.below(coincident_span))
                } else {
                    // Not coincident: threshold+1 ..= max_gap.
                    cfg.threshold_ticks + 1 + u64::from(rng.below(distant_span))
                };
                let shift = jitter(&mut rng, jit);
                let t0 = (cfg.first_tick as i64 + shift).max(0) as u64;
                let sp =
                    vec![Spike { t: t0, source: 0 }, Spike { t: t0 + gap, source: 1 }];
                Sample { train: Train::from_spikes(sp), label: class }
            },
        )?;

        Ok(finish(
            "coincidence",
            train,
            test,
            2,
            2,
            self.ticks,
            self.dt,
            self.seed,
            "the coincidence-detection unit test used throughout the spiking literature; no \
             recorded dataset",
            "a real coincidence problem has jitter on each spike independently, more than two \
             afferents, and a gap distribution with no hard boundary at the threshold",
        ))
    }
}

// ---------------------------------------------------------------------------------------------
// 3. Delayed match-to-sample
// ---------------------------------------------------------------------------------------------

/// **Delayed match-to-sample**: a symbol, a silent delay, a second symbol — were they the same?
///
/// Fuster and Alexander, Science 173:652-654, 1971, is the original; it is the standard assay for
/// working memory because the answer is not a function of anything present at the moment it is
/// asked. During `delay_ticks` there is **no input at all** (with `distractor_cues` at zero), so
/// any network that solves it is holding the sample symbol in its state. The invariant is asserted:
/// `delayed_match_holds_an_empty_delay_window` checks that not one spike lands in the delay.
///
/// # What the delay demands, quantitatively
///
/// [`DelayedMatch::required_memory_seconds`] is `delay_ticks * dt`. A [`crate::neuron::Lif`] with
/// `tau_m = 20 ms` still holds `exp(-0.1 / 0.02) = 0.67%` of an input 100 ms later — it has lost
/// 99.3% of it — so a feedforward network of default `Lif` cells cannot carry a 100 ms delay in
/// its membranes. The mechanisms that
/// can are recurrence, the adapting threshold of [`crate::neuron::AdaptiveLif`] (Bellec et al.,
/// 2018, whose whole point is exactly this), or a synapse with a long time constant. That is the
/// lesson; the task is the instrument for it.
///
/// # An honest caveat, stated beside the task rather than beneath it
///
/// In the `distractor_cues == 0` form this task **has a rate-code shortcut**. A match puts
/// `2 * cue_spikes` spikes on one channel; a non-match puts `cue_spikes` on each of two. So the
/// per-channel spike count over the whole window separates the classes perfectly, with no memory
/// at all — [`DelayedMatch::count_readout`] is that shortcut, and it scores exactly 1.0. It is
/// provided rather than hidden because a baseline you cannot see is a baseline you will not beat.
/// Setting `distractor_cues` above zero puts cues on random channels during the delay, which breaks
/// the shortcut and leaves the memory requirement intact.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DelayedMatch {
    /// Number of distinct symbols, one input channel each. At least 2, or every trial is a match.
    pub n_symbols: u32,
    /// Length of each cue window in ticks.
    pub cue_ticks: u64,
    /// Spikes in one cue, evenly spaced across the cue window.
    pub cue_spikes: u32,
    /// Silent interval between the two cues, in ticks. The memory requirement, in ticks.
    pub delay_ticks: u64,
    /// Cues placed on random channels **during** the delay, to break the spike-count shortcut.
    /// Zero keeps the delay window provably empty.
    ///
    /// Distractors are drawn independently and may land on the same channel, or on top of each
    /// other. That is deliberate — a distractor that collided with another is still exactly
    /// `cue_spikes` more spikes inside the delay, which is the invariant the tests assert, and
    /// forbidding collisions would make the distractor distribution depend on the ones before it.
    pub distractor_cues: u32,
    /// Half-width of the uniform common-mode jitter on each cue's start, in ticks.
    pub jitter_ticks: u64,
    /// Tick length in seconds.
    pub dt: f64,
    /// Train samples per class.
    pub per_class_train: usize,
    /// Test samples per class.
    pub per_class_test: usize,
    /// Seed.
    pub seed: u64,
}

impl Default for DelayedMatch {
    /// Four symbols, 1 ms ticks, 30 ms cues of four spikes, and a 100 ms delay — five membrane
    /// constants of a default `Lif`, which is chosen so that the membrane cannot be the answer.
    ///
    /// The 4 ms jitter gives `4 * 9 * 9 = 324` distinct matching inputs, above the 192 per class
    /// asked for; the cue window is 30 ms rather than 20 because the spikes of a cue are spaced
    /// `cue_ticks / cue_spikes` apart, so at `cue_ticks = 20` four spikes span `3 * 5 = 15` ticks
    /// and `15 + 2 * 4` jitter ticks does not fit in 20. At 30 the spacing is 7, the span is 21,
    /// and `21 + 8 < 30` fits. `a_cue_window_too_short_for_its_spikes_and_jitter_is_refused` runs
    /// both halves of that sentence.
    fn default() -> Self {
        Self {
            n_symbols: 4,
            cue_ticks: 30,
            cue_spikes: 4,
            delay_ticks: 100,
            distractor_cues: 0,
            jitter_ticks: 4,
            dt: 1e-3,
            per_class_train: 128,
            per_class_test: 64,
            seed: 0x5EED_0003,
        }
    }
}

impl DelayedMatch {
    /// Total sample length in ticks: two cue windows and the delay between them.
    ///
    /// Saturates at `u64::MAX` rather than wrapping, so a configuration whose windows do not fit in
    /// a `u64` reports an impossibly long sample instead of a short one. [`DelayedMatch::generate`]
    /// refuses such a configuration; nothing in this module can produce a window that long.
    #[must_use]
    pub fn ticks(&self) -> u64 {
        self.cue_ticks.saturating_mul(2).saturating_add(self.delay_ticks)
    }

    /// How long the network must hold the sample symbol, in **seconds**.
    #[must_use]
    pub fn required_memory_seconds(&self) -> f64 {
        self.delay_ticks as f64 * self.dt
    }

    /// First tick of the delay window.
    #[must_use]
    pub fn delay_start(&self) -> u64 {
        self.cue_ticks
    }

    /// First tick after the delay window, where the test cue begins.
    ///
    /// Saturates at `u64::MAX` rather than wrapping, for the reason [`DelayedMatch::ticks`] gives.
    #[must_use]
    pub fn delay_end(&self) -> u64 {
        self.cue_ticks.saturating_add(self.delay_ticks)
    }

    /// The memory-free shortcut: answer "match" when some channel carries `2 * cue_spikes` spikes.
    ///
    /// Exactly correct when `distractor_cues == 0`, and degraded when it is not. Its purpose is to
    /// be the number a real model has to beat.
    #[must_use]
    pub fn count_readout(&self, s: &Sample) -> u32 {
        let mut counts = vec![0u32; self.n_symbols as usize];
        for sp in s.train.spikes() {
            let i = sp.source as usize;
            if i < counts.len() {
                counts[i] += 1;
            }
        }
        u32::from(counts.iter().copied().max().unwrap_or(0) >= 2 * self.cue_spikes)
    }

    fn spacing(&self) -> u64 {
        (self.cue_ticks / u64::from(self.cue_spikes)).max(1)
    }

    /// Build the dataset.
    ///
    /// # Errors
    ///
    /// [`TaskError::Empty`] for zero cue spikes or zero cue ticks; [`TaskError::OutOfRange`] when
    /// `n_symbols < 2`, when a jittered cue would not fit inside its window, when distractors are
    /// requested but do not fit inside the delay, when `jitter_ticks` exceeds [`MAX_JITTER_HALF`]
    /// or the distractor's room inside the delay exceeds [`MAX_DRAW_SPAN`] (either would narrow a
    /// uniform draw), or when `dt` is not positive and finite; [`TaskError::Exhausted`] when the
    /// reachable symbol-and-jitter combinations are fewer than the split needs.
    pub fn generate(&self) -> Result<Dataset<Sample>, TaskError> {
        in_range("dt", self.dt, f64::MIN_POSITIVE, f64::MAX)?;
        nonzero("cue_spikes", u64::from(self.cue_spikes))?;
        nonzero("cue_ticks", self.cue_ticks)?;
        let jit = half_width("jitter_ticks", self.jitter_ticks)?;
        if self.n_symbols < 2 {
            return Err(TaskError::OutOfRange {
                what: "n_symbols",
                value: f64::from(self.n_symbols),
                low: 2.0,
                high: f64::from(u32::MAX),
            });
        }
        let span = u64::from(self.cue_spikes - 1).saturating_mul(self.spacing());
        let need = span.saturating_add(2 * self.jitter_ticks);
        if need >= self.cue_ticks {
            return Err(TaskError::OutOfRange {
                what: "cue_ticks",
                value: self.cue_ticks as f64,
                low: need.saturating_add(1) as f64,
                high: f64::MAX,
            });
        }
        if self.distractor_cues > 0 && span.saturating_add(1) >= self.delay_ticks {
            return Err(TaskError::OutOfRange {
                what: "delay_ticks",
                value: self.delay_ticks as f64,
                low: span.saturating_add(2) as f64,
                high: f64::MAX,
            });
        }
        // Room for a distractor inside the delay, narrowed once and checked. Only reachable when
        // distractors are asked for, and the check above has already made it at least 1 there.
        let room = if self.distractor_cues > 0 {
            draw_span("delay_ticks", self.delay_ticks - span)?
        } else {
            1
        };

        let ticks = self.ticks();
        if ticks == u64::MAX {
            return Err(TaskError::OutOfRange {
                what: "delay_ticks",
                value: self.delay_ticks as f64,
                low: 0.0,
                high: (u64::MAX - 1) as f64,
            });
        }
        let spacing = self.spacing();
        let mut rng = Rng::new(self.seed);
        let cfg = *self;
        let (train, test) = collect_split(
            2,
            self.per_class_train,
            self.per_class_test,
            self.seed ^ 0xA5A5_0003,
            |class, _| {
                let sample_sym = rng.below(cfg.n_symbols);
                let test_sym = if class == 1 {
                    sample_sym
                } else {
                    // Uniform over the n-1 symbols that are not the sample.
                    let k = rng.below(cfg.n_symbols - 1);
                    if k >= sample_sym { k + 1 } else { k }
                };
                let mut sp = Vec::new();
                let cue = |sp: &mut Vec<Spike>, sym: u32, base: u64, off: i64| {
                    let start = (base as i64 + off).max(0) as u64;
                    for k in 0..u64::from(cfg.cue_spikes) {
                        sp.push(Spike { t: start + k * spacing, source: sym });
                    }
                };
                let j1 = jitter(&mut rng, jit);
                let j2 = jitter(&mut rng, jit);
                cue(&mut sp, sample_sym, cfg.jitter_ticks, j1);
                cue(&mut sp, test_sym, cfg.delay_end() + cfg.jitter_ticks, j2);
                for _ in 0..cfg.distractor_cues {
                    let sym = rng.below(cfg.n_symbols);
                    let at = cfg.delay_start() + u64::from(rng.below(room));
                    cue(&mut sp, sym, at, 0);
                }
                Sample { train: Train::from_spikes(sp), label: class }
            },
        )?;

        Ok(finish(
            "delayed_match_to_sample",
            train,
            test,
            2,
            self.n_symbols,
            ticks,
            self.dt,
            self.seed,
            "the working-memory assay of Fuster and Alexander, Science 173:652-654, 1971; no \
             public spiking dataset of it",
            "a real delayed-match trial has sensory variability within a symbol, a variable delay, \
             and an animal that can look away; here a symbol is one clean channel",
        ))
    }
}

// ---------------------------------------------------------------------------------------------
// 4. Poisson rate discrimination
// ---------------------------------------------------------------------------------------------

/// **Rate discrimination**: two noisy channels, which one is firing faster?
///
/// The task where a rate code is the *right* code, included so that the contrast with
/// [`TemporalXor`] is made by the same library rather than by assertion. It is also the only task
/// here whose best achievable accuracy is **computable in closed form**, which makes it the one
/// place a student can ask "is my model good, or is the task hard?" and get a number.
///
/// # The generator
///
/// Two channels over `ticks` ticks of `dt` seconds. One channel — chosen by the label — fires at
/// `rate_hi` hertz, the other at `rate_lo`. Each tick draws a Bernoulli with
/// `p = 1 - exp(-rate * dt)`, matching [`crate::encode::RateEncoder`] exactly rather than the
/// `rate * dt` approximation, which exceeds one and silently saturates at high rates.
///
/// # The optimal observer, derived
///
/// The log-likelihood ratio between "channel 0 is fast" and "channel 1 is fast" is
///
/// ```text
/// log LR = (N0 - N1) * log( p_hi (1 - p_lo) / (p_lo (1 - p_hi)) )
/// ```
///
/// and the bracket is positive whenever `rate_hi > rate_lo`. So the optimal rule is **whichever
/// channel counted more**, and no amount of architecture can beat it. Its accuracy is
///
/// ```text
/// P(correct) = P(N_hi > N_lo) + 0.5 * P(N_hi = N_lo)
/// ```
///
/// over independent `Binomial(ticks, p)` counts. That is [`RateDiscrimination::optimal_accuracy`],
/// summed exactly. Two special cases pin it: equal rates give exactly 0.5, and `rate_lo = 0` gives
/// exactly `1 - 0.5 * (1 - p_hi)^ticks`, both asserted in the tests.
///
/// [`RateDiscrimination::poisson_optimal_accuracy`] is the same quantity for the continuous-time
/// Poisson limit the literature quotes, with `mu = rate * ticks * dt`. The two converge as `dt`
/// shrinks at fixed `mu`, and `the_binomial_bound_converges_to_the_poisson_bound` asserts it. The
/// binomial one is the honest bound for this generator; the Poisson one is what you compare against
/// a paper.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RateDiscrimination {
    /// Firing rate of the fast channel, hertz. Must be at least `rate_lo`.
    pub rate_hi: f64,
    /// Firing rate of the slow channel, hertz. Equal rates make the task impossible and the bound
    /// reports exactly 0.5, which is the correct answer rather than an error.
    pub rate_lo: f64,
    /// Counting window in ticks. Difficulty falls roughly as `1 / sqrt(ticks)`.
    pub ticks: u64,
    /// Tick length in seconds.
    pub dt: f64,
    /// Train samples per class.
    pub per_class_train: usize,
    /// Test samples per class.
    pub per_class_test: usize,
    /// Seed.
    pub seed: u64,
}

impl Default for RateDiscrimination {
    /// 60 Hz against 40 Hz over 100 ms at 1 ms ticks: 5.82 and 3.92 expected spikes, which puts
    /// [`RateDiscrimination::optimal_accuracy`] at **0.7324** — hard enough that a model has
    /// somewhere to go, and easy enough that a working model is visibly above the 0.5 chance line.
    ///
    /// 0.7324 bounds the **expected** accuracy, not the accuracy of one measurement of it. On the
    /// default 800-sample test split a score has a binomial standard error of
    /// `sqrt(0.7324 * 0.2676 / 800) = 0.0156`, so a model reporting 0.80 is 4.3 standard errors
    /// above a ceiling no rule can pass in expectation: that is very strong evidence of a leak,
    /// which is a different sentence from "impossible", and a benchmark that blurs the two teaches
    /// the wrong reflex. A single score of 0.76 on this split is not evidence of anything.
    fn default() -> Self {
        Self {
            rate_hi: 60.0,
            rate_lo: 40.0,
            ticks: 100,
            dt: 1e-3,
            per_class_train: 400,
            per_class_test: 400,
            seed: 0x5EED_0004,
        }
    }
}

impl RateDiscrimination {
    /// Per-tick spike probability for a rate, `1 - exp(-rate * dt)`.
    ///
    /// `None` for a non-finite or negative rate, or a non-positive `dt`.
    #[must_use]
    pub fn p_tick(&self, rate: f64) -> Option<f64> {
        if !rate.is_finite() || rate < 0.0 || !self.dt.is_finite() || self.dt <= 0.0 {
            return None;
        }
        Some(1.0 - (-rate * self.dt).exp())
    }

    /// Best accuracy any observer can reach on **this** generator, exactly.
    ///
    /// `None` when a parameter is not finite, when `rate_lo > rate_hi` (the labels would then be
    /// upside down and a "bound" below chance would be reported as a bound), when `ticks` is
    /// zero — a window of no length carries no counts and has no discrimination at all, which is a
    /// different statement from "chance" — or when `ticks` exceeds [`MAX_COUNT_SUPPORT`].
    ///
    /// That last condition is the same ceiling [`RateDiscrimination::poisson_optimal_accuracy`]
    /// puts on its truncation, and it exists for two measured reasons. The sum builds three
    /// `Vec<f64>` of `ticks + 1` entries, so `ticks = 1e9` asks for 24 GB and aborts the process —
    /// an abort is not a `None`. And the summation error grows with the support: at `ticks = 1e7`
    /// the returned figure is within about `1e-6` of the truth, while at `ticks = 5e7` it comes
    /// back as `1.00002636954903035`, which is not a probability at all. A bound above one is
    /// worse than no bound, because it is still a number somebody will print.
    #[must_use]
    pub fn optimal_accuracy(&self) -> Option<f64> {
        if self.ticks == 0 || self.ticks > MAX_COUNT_SUPPORT || self.rate_lo > self.rate_hi {
            return None;
        }
        let p_hi = self.p_tick(self.rate_hi)?;
        let p_lo = self.p_tick(self.rate_lo)?;
        let n = usize::try_from(self.ticks).ok()?;
        let lnf = ln_factorials(n);
        let hi = binomial_pmf(n, p_hi, &lnf);
        let lo = binomial_pmf(n, p_lo, &lnf);
        Some(win_probability(&hi, &lo))
    }

    /// The same bound in the continuous-time Poisson limit, which is the form papers quote.
    ///
    /// `None` under the same conditions as [`RateDiscrimination::optimal_accuracy`]. The truncation
    /// is at `mu + 12 sqrt(mu) + 60` counts, where the neglected tail is below `1e-15` for every
    /// `mu` this crate's tasks reach; for `mu` beyond about `1e6` that margin has not been checked
    /// and the figure should not be trusted without re-deriving it.
    #[must_use]
    pub fn poisson_optimal_accuracy(&self) -> Option<f64> {
        if self.ticks == 0 || self.rate_lo > self.rate_hi {
            return None;
        }
        if !self.rate_hi.is_finite() || !self.rate_lo.is_finite() || !self.dt.is_finite() {
            return None;
        }
        if self.rate_lo < 0.0 || self.dt <= 0.0 {
            return None;
        }
        let t = self.ticks as f64 * self.dt;
        let mu_hi = self.rate_hi * t;
        let mu_lo = self.rate_lo * t;
        let k_max = (mu_hi + 12.0 * mu_hi.sqrt() + 60.0).ceil();
        if !k_max.is_finite() || k_max > MAX_COUNT_SUPPORT as f64 {
            return None;
        }
        let k_max = k_max as usize;
        let lnf = ln_factorials(k_max);
        let hi = poisson_pmf(mu_hi, k_max, &lnf);
        let lo = poisson_pmf(mu_lo, k_max, &lnf);
        Some(win_probability(&hi, &lo))
    }

    /// The optimal rule: answer with the channel that counted more, breaking ties toward class 0.
    ///
    /// Deterministic tie-breaking rather than a coin, because on an exactly balanced test set the
    /// two score the same in expectation — ties are equally likely in both classes — and a
    /// deterministic classifier is reproducible.
    #[must_use]
    pub fn classify_by_count(s: &Sample) -> u32 {
        let n0 = s.train.spikes().iter().filter(|x| x.source == 0).count();
        let n1 = s.train.spikes().iter().filter(|x| x.source == 1).count();
        u32::from(n1 > n0)
    }

    /// Build the dataset.
    ///
    /// # Errors
    ///
    /// [`TaskError::NotFinite`] for a non-finite rate or `dt`; [`TaskError::Empty`] for a
    /// zero-length window; [`TaskError::OutOfRange`] for a negative rate, a non-positive `dt`, or
    /// `rate_lo > rate_hi`; [`TaskError::Exhausted`] when the window is too short to produce
    /// enough distinct spike patterns.
    pub fn generate(&self) -> Result<Dataset<Sample>, TaskError> {
        in_range("dt", self.dt, f64::MIN_POSITIVE, f64::MAX)?;
        in_range("rate_hi", self.rate_hi, 0.0, f64::MAX)?;
        in_range("rate_lo", self.rate_lo, 0.0, self.rate_hi)?;
        nonzero("ticks", self.ticks)?;

        let p_hi = self.p_tick(self.rate_hi).ok_or(TaskError::NotFinite {
            what: "rate_hi",
            value: self.rate_hi,
        })?;
        let p_lo = self.p_tick(self.rate_lo).ok_or(TaskError::NotFinite {
            what: "rate_lo",
            value: self.rate_lo,
        })?;

        let mut rng = Rng::new(self.seed);
        let ticks = self.ticks;
        let (train, test) = collect_split(
            2,
            self.per_class_train,
            self.per_class_test,
            self.seed ^ 0xA5A5_0004,
            |class, _| {
                // class 0: channel 0 is fast. class 1: channel 1 is fast.
                let p = if class == 0 { [p_hi, p_lo] } else { [p_lo, p_hi] };
                let mut sp = Vec::new();
                for t in 0..ticks {
                    for ch in 0..2usize {
                        if rng.next_f64() < p[ch] {
                            sp.push(Spike { t, source: ch as u32 });
                        }
                    }
                }
                Sample { train: Train::from_spikes(sp), label: class }
            },
        )?;

        Ok(finish(
            "rate_discrimination",
            train,
            test,
            2,
            2,
            self.ticks,
            self.dt,
            self.seed,
            "two-alternative rate discrimination, the psychophysical staple; the closest recorded \
             analogue is a two-whisker or two-interval frequency task, which this does not \
             reproduce",
            "independent Bernoulli ticks, no refractoriness, no burstiness, no correlation between \
             channels — a real cortical spike train has all four and a lower information rate",
        ))
    }
}

// ---------------------------------------------------------------------------------------------
// 5. Latency pattern classification
// ---------------------------------------------------------------------------------------------

/// **Latency patterns**: `n_classes` prototypes, each a vector of one spike time per channel,
/// recovered through jitter.
///
/// Time-to-first-spike coding at its most explicit (see [`crate::encode::LatencyEncoder`]): every
/// channel fires exactly once, and the whole message is in the pattern of delays. One spike per
/// channel per sample makes this the cheapest task in the module by synaptic operations, and the
/// most fragile — a single-tick shift is a different number, where in a rate code it is noise.
///
/// # A guarantee, not a hope
///
/// Templates are drawn so that any two differ by more than `min_separation_ticks` on at least one
/// channel — that is, their `L-infinity` distance exceeds it. If that separation is more than twice
/// the jitter, nearest-template classification is **exactly perfect**, by the triangle inequality:
/// a sample lies within `j` of its own template, and within at least `separation - j > j` of every
/// other. [`LatencyPatterns::guaranteed_separable`] reports whether the configuration satisfies it,
/// and the test asserts an accuracy of exactly 1.0 when it does. When it does not, the task is
/// genuinely ambiguous and no model can be perfect — which is worth knowing before blaming a model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LatencyPatterns {
    /// Number of classes, one prototype each.
    pub n_classes: u32,
    /// Number of input channels; each fires exactly once per sample.
    pub n_inputs: u32,
    /// Sample length in ticks. Template latencies live in `jitter_ticks ..= ticks - 1 - jitter_ticks`.
    pub ticks: u64,
    /// Half-width of the uniform per-channel jitter, in ticks. Independent per channel, unlike the
    /// common-mode jitters elsewhere in this module — a latency code has to survive both, and this
    /// is the harder one.
    pub jitter_ticks: u64,
    /// Required `L-infinity` separation between any two templates, in ticks.
    pub min_separation_ticks: u64,
    /// Tick length in seconds.
    pub dt: f64,
    /// Train samples per class.
    pub per_class_train: usize,
    /// Test samples per class.
    pub per_class_test: usize,
    /// Seed.
    pub seed: u64,
}

impl Default for LatencyPatterns {
    /// Five classes over eight channels in a 50 ms window at 1 ms ticks, with 2 ms jitter and a
    /// 10 ms minimum separation — comfortably more than `2 * jitter`, so the task is exactly
    /// solvable and a model that is not perfect has something wrong with it.
    fn default() -> Self {
        Self {
            n_classes: 5,
            n_inputs: 8,
            ticks: 50,
            jitter_ticks: 2,
            min_separation_ticks: 10,
            dt: 1e-3,
            per_class_train: 100,
            per_class_test: 50,
            seed: 0x5EED_0005,
        }
    }
}

impl LatencyPatterns {
    /// Whether `min_separation_ticks > 2 * jitter_ticks`, the condition for exact separability.
    #[must_use]
    pub fn guaranteed_separable(&self) -> bool {
        self.min_separation_ticks > 2 * self.jitter_ticks
    }

    /// The class prototypes, indexed `[class][channel]`, in ticks.
    ///
    /// Deterministic from `seed` alone, so a caller can recompute them without holding the dataset.
    ///
    /// # Errors
    ///
    /// [`TaskError::Empty`] for zero classes or channels; [`TaskError::OutOfRange`] when the window
    /// leaves no room for a latency after the jitter margin, or when that room is too wide for a
    /// single uniform draw (`ticks - 2 * jitter_ticks` must not exceed [`MAX_DRAW_SPAN`]);
    /// [`TaskError::Exhausted`] when rejection sampling could not place that many templates at the
    /// requested separation, which happens when `min_separation_ticks` approaches the usable
    /// window.
    pub fn templates(&self) -> Result<Vec<Vec<u64>>, TaskError> {
        nonzero("n_classes", u64::from(self.n_classes))?;
        nonzero("n_inputs", u64::from(self.n_inputs))?;
        let margin = self.jitter_ticks.saturating_mul(2);
        if self.ticks <= margin {
            return Err(TaskError::OutOfRange {
                what: "ticks",
                value: self.ticks as f64,
                low: margin.saturating_add(1) as f64,
                high: f64::MAX,
            });
        }
        let lo = self.jitter_ticks;
        // Narrowed once, here, rather than with an `as u32` inside the rejection loop.
        let span = draw_span("ticks", self.ticks - margin)?;

        let mut rng = Rng::new(self.seed ^ 0x7E17_0005);
        let mut out: Vec<Vec<u64>> = Vec::with_capacity(self.n_classes as usize);
        let budget = 4096usize * self.n_classes as usize;
        let mut draws = 0usize;
        while out.len() < self.n_classes as usize {
            if draws >= budget {
                return Err(TaskError::Exhausted {
                    wanted: self.n_classes as usize,
                    distinct: out.len(),
                    draws,
                });
            }
            draws += 1;
            let cand: Vec<u64> =
                (0..self.n_inputs).map(|_| lo + u64::from(rng.below(span))).collect();
            let ok = out.iter().all(|t| linf(t, &cand) > self.min_separation_ticks);
            if ok {
                out.push(cand);
            }
        }
        Ok(out)
    }

    /// Nearest template in `L-infinity`, the norm the separability guarantee is stated in.
    ///
    /// `None` when a channel is silent, which this generator never produces; a caller filtering a
    /// stream can hit it, and an imputed latency would enter the accuracy figure as though it had
    /// been measured.
    #[must_use]
    pub fn nearest_template(&self, templates: &[Vec<u64>], s: &Sample) -> Option<u32> {
        let mut times = vec![u64::MAX; self.n_inputs as usize];
        for sp in s.train.spikes() {
            let i = sp.source as usize;
            if i < times.len() && times[i] == u64::MAX {
                times[i] = sp.t;
            }
        }
        if times.contains(&u64::MAX) {
            return None;
        }
        let mut best = (u64::MAX, 0u32);
        for (c, t) in templates.iter().enumerate() {
            let d = linf(t, &times);
            if d < best.0 {
                best = (d, c as u32);
            }
        }
        Some(best.1)
    }

    /// Build the dataset.
    ///
    /// # Errors
    ///
    /// As [`LatencyPatterns::templates`], plus [`TaskError::Exhausted`] when the jitter admits
    /// fewer distinct patterns than the split needs, and [`TaskError::OutOfRange`] for a `dt` that
    /// is not positive and finite or a `jitter_ticks` above [`MAX_JITTER_HALF`].
    pub fn generate(&self) -> Result<Dataset<Sample>, TaskError> {
        in_range("dt", self.dt, f64::MIN_POSITIVE, f64::MAX)?;
        let jit = half_width("jitter_ticks", self.jitter_ticks)?;
        let templates = self.templates()?;
        let mut rng = Rng::new(self.seed);
        let cfg = *self;
        let (train, test) = collect_split(
            self.n_classes,
            self.per_class_train,
            self.per_class_test,
            self.seed ^ 0xA5A5_0005,
            |class, _| {
                let t = &templates[class as usize];
                let mut sp = Vec::with_capacity(cfg.n_inputs as usize);
                for (ch, &base) in t.iter().enumerate() {
                    let off = jitter(&mut rng, jit);
                    let at = (base as i64 + off).clamp(0, cfg.ticks as i64 - 1) as u64;
                    sp.push(Spike { t: at, source: ch as u32 });
                }
                Sample { train: Train::from_spikes(sp), label: class }
            },
        )?;

        Ok(finish(
            "latency_patterns",
            train,
            test,
            self.n_classes,
            self.n_inputs,
            self.ticks,
            self.dt,
            self.seed,
            "the time-to-first-spike classification setup used for converted image networks; no \
             recorded dataset",
            "every channel fires exactly once and always fires; a real latency code has silent \
             channels, which carry the information that there was no evidence",
        ))
    }
}

fn linf(a: &[u64], b: &[u64]) -> u64 {
    a.iter().zip(b).map(|(&x, &y)| x.abs_diff(y)).max().unwrap_or(0)
}

// ---------------------------------------------------------------------------------------------
// 6. Moving bar: a synthetic event stream
// ---------------------------------------------------------------------------------------------

/// Which way the bar travels across the pixel grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Direction {
    /// Increasing `x`: a vertical bar sweeping left to right.
    Right,
    /// Decreasing `x`: a vertical bar sweeping right to left.
    Left,
    /// Increasing `y`: a horizontal bar sweeping top to bottom.
    Down,
    /// Decreasing `y`: a horizontal bar sweeping bottom to top.
    Up,
}

impl Direction {
    /// The four directions in class order, so a class index and a direction are interconvertible.
    #[must_use]
    pub fn all() -> [Self; 4] {
        [Self::Right, Self::Left, Self::Down, Self::Up]
    }

    /// Whether motion is along `x` (and the bar is therefore vertical, spanning every row).
    #[must_use]
    pub fn is_horizontal(self) -> bool {
        matches!(self, Self::Right | Self::Left)
    }
}

/// **Moving bar**: a bar sweeping a pixel grid, emitted as address-events with polarity — an
/// `N-MNIST` and `DVS-Gesture` stand-in for teaching event vision.
///
/// A real event camera reports per-pixel brightness *changes*, not frames. This generator does
/// exactly that for the simplest moving stimulus: as the bar's leading edge enters a pixel the
/// pixel emits [`Polarity::On`]; as the trailing edge leaves it emits [`Polarity::Off`]. Addresses
/// are `y * width + x`, which is the convention [`crate::spike::Event::address`] states.
///
/// # The lesson this task exists for
///
/// **Over a full traverse the event count does not depend on the speed.** Every pixel turns on once
/// and off once, so the total is exactly `2 * width * height`, whether the bar crosses in ten ticks
/// or a thousand. What speed changes is the *rate*: the same events arrive in proportionally less
/// time. This is the property that makes event sensing interesting and is almost always stated
/// backwards — doubling the speed of the world does not double the data, it halves the latency. The
/// exact count is [`MovingBar::events_per_traverse`], and the tests assert both the constancy and
/// the inverse-proportional duration.
///
/// The polarity split is exact too: `width * height` on-events and `width * height` off-events.
/// An implementation that dropped the sign, or folded both into one channel, would leave the total
/// unchanged — which is why the test checks the split rather than the total alone.
///
/// # Classes
///
/// Four, one per [`Direction`], with per-sample jitter in the start time and in the speed. Chance
/// is 0.25. Notice that a readout over the *whole* stream cannot separate `Right` from `Left`
/// either by count or by polarity balance — both are exactly `2 * width * height` and exactly half
/// on. Direction lives in the **order** the addresses arrive in, which is the same point
/// [`TemporalXor`] makes, in two dimensions.
///
/// # Noise
///
/// `noise_prob_per_pixel_per_tick` adds independent background events, as a real sensor does. At
/// zero the counts above are exact; above zero the expected total is
/// `2 * width * height + width * height * ticks * p`, which the tests check empirically.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MovingBar {
    /// Grid width in pixels.
    pub width: u32,
    /// Grid height in pixels.
    pub height: u32,
    /// Bar thickness in pixels, along the direction of travel. Must be at least
    /// `ceil(max speed)`, or the bar skips pixels and the exact event count no longer holds.
    pub bar_width: u32,
    /// Nominal speed in pixels per tick. Must be positive.
    pub speed_px_per_tick: f64,
    /// Fractional half-width of the uniform per-sample speed jitter, in `[0, 0.9]`. The traverse
    /// event count is invariant to it, which is the point.
    pub speed_jitter_frac: f64,
    /// Largest per-sample start delay in ticks. The bar sits fully outside the frame until then,
    /// so the delay adds no events.
    pub max_start_delay: u64,
    /// Independent background-event probability, per pixel per tick, in `[0, 1)`.
    pub noise_prob_per_pixel_per_tick: f64,
    /// Sample length in ticks. Must be at least [`MovingBar::min_ticks`].
    pub ticks: u64,
    /// Tick length in seconds.
    pub dt: f64,
    /// Train samples per class.
    pub per_class_train: usize,
    /// Test samples per class.
    pub per_class_test: usize,
    /// Seed.
    pub seed: u64,
}

impl Default for MovingBar {
    /// A 16x16 grid, a 3-pixel bar at 1 pixel per tick with 20% speed jitter, 1 ms ticks and a
    /// 40-tick window — enough for the slowest jittered traverse plus the largest start delay.
    fn default() -> Self {
        Self {
            width: 16,
            height: 16,
            bar_width: 3,
            speed_px_per_tick: 1.0,
            speed_jitter_frac: 0.2,
            max_start_delay: 4,
            noise_prob_per_pixel_per_tick: 0.0,
            ticks: 40,
            dt: 1e-3,
            per_class_train: 48,
            per_class_test: 24,
            seed: 0x5EED_0006,
        }
    }
}

impl MovingBar {
    /// Exact event count of one noiseless full traverse: `2 * width * height`.
    ///
    /// Independent of speed, of direction and of the start delay. Every pixel is covered by exactly
    /// one contiguous interval of ticks, so it emits exactly one on-event and one off-event.
    #[must_use]
    pub fn events_per_traverse(&self) -> u64 {
        2 * u64::from(self.width) * u64::from(self.height)
    }

    /// Slowest speed a sample can be drawn at, in pixels per tick.
    #[must_use]
    pub fn min_speed(&self) -> f64 {
        self.speed_px_per_tick * (1.0 - self.speed_jitter_frac)
    }

    /// Fastest speed a sample can be drawn at, in pixels per tick.
    #[must_use]
    pub fn max_speed(&self) -> f64 {
        self.speed_px_per_tick * (1.0 + self.speed_jitter_frac)
    }

    /// Shortest window in which the slowest jittered traverse completes after the longest start
    /// delay, in ticks.
    ///
    /// `None` when the speed parameters are not usable, in which case there is no such window
    /// rather than a very long one.
    #[must_use]
    pub fn min_ticks(&self) -> Option<u64> {
        let v = self.min_speed();
        if !v.is_finite() || v <= 0.0 {
            return None;
        }
        // Summed in `f64`, because `width.max(height) + bar_width` overflows a `u32` for a grid
        // near the top of the range and would wrap to a tiny extent — a window that looks ample.
        let extent = f64::from(self.width.max(self.height)) + f64::from(self.bar_width);
        let t = (extent / v).ceil();
        if !t.is_finite() || t > 1e12 {
            return None;
        }
        Some(self.max_start_delay.saturating_add(t as u64).saturating_add(2))
    }

    /// One noiseless traverse at a chosen direction and speed, starting immediately.
    ///
    /// Exposed because the speed-invariance lesson is a two-line experiment with it, and because a
    /// student should be able to look at an event stream without generating a dataset.
    ///
    /// # Errors
    ///
    /// [`TaskError::Empty`] for a zero grid dimension or zero bar width; [`TaskError::OutOfRange`]
    /// for a non-positive speed, a speed whose per-tick advance exceeds `bar_width` (which would
    /// skip pixels), or a window shorter than the traverse needs.
    pub fn traverse(&self, direction: Direction, speed: f64) -> Result<Vec<Event>, TaskError> {
        nonzero("width", u64::from(self.width))?;
        nonzero("height", u64::from(self.height))?;
        nonzero("bar_width", u64::from(self.bar_width))?;
        in_range("speed", speed, f64::MIN_POSITIVE, f64::from(self.bar_width))?;
        // The advance per tick is at most `ceil(speed)`; if that exceeds the bar's own thickness a
        // pixel can be stepped over entirely and never reported, and the exact count silently
        // becomes an inequality.
        if speed.ceil() > f64::from(self.bar_width) {
            return Err(TaskError::OutOfRange {
                what: "speed_px_per_tick",
                value: speed,
                low: f64::MIN_POSITIVE,
                high: f64::from(self.bar_width),
            });
        }
        let extent = if direction.is_horizontal() { self.width } else { self.height };
        let need = ((f64::from(extent) + f64::from(self.bar_width)) / speed).ceil() as u64 + 2;
        if self.ticks < need {
            return Err(TaskError::OutOfRange {
                what: "ticks",
                value: self.ticks as f64,
                low: need as f64,
                high: f64::MAX,
            });
        }
        Ok(self.bar_events(direction, 0, speed))
    }

    /// Leading-edge interval of the bar at tick `t`, clipped to the frame, as `(lo, hi)` with `hi`
    /// exclusive. An empty interval is returned as `(0, 0)`.
    fn occupancy(&self, direction: Direction, delay: u64, speed: f64, t: u64) -> (i64, i64) {
        if t < delay {
            return (0, 0);
        }
        let extent = i64::from(if direction.is_horizontal() { self.width } else { self.height });
        let bw = i64::from(self.bar_width);
        let travelled = (speed * (t - delay) as f64).floor() as i64;
        let lead = match direction {
            Direction::Right | Direction::Down => travelled - bw,
            Direction::Left | Direction::Up => extent - travelled,
        };
        let lo = lead.max(0);
        let hi = (lead + bw).min(extent);
        if hi <= lo { (0, 0) } else { (lo, hi) }
    }

    /// First tick at which the bar is certainly gone, for this delay and speed.
    ///
    /// `occupancy` is empty once `floor(speed * (t - delay)) >= extent + bar_width`, in every
    /// direction, and an off-event trails its tick by one. Iterating past that point emits nothing,
    /// so the loop in [`MovingBar::bar_events`] stops there instead of walking the whole window: a
    /// legal `ticks = 1e9` config — the window is not otherwise bounded above — took 20 seconds for
    /// eight samples before, and a `1e12` one never finished at all. Clipped to `ticks`, so the
    /// events are exactly the ones the full walk produced.
    fn last_active_tick(&self, direction: Direction, delay: u64, speed: f64) -> u64 {
        let extent = if direction.is_horizontal() { self.width } else { self.height };
        let reach = (f64::from(extent) + f64::from(self.bar_width)) / speed;
        if !reach.is_finite() || reach < 0.0 {
            return self.ticks;
        }
        let over = reach.ceil();
        // `as u64` saturates at the top of the range in Rust, and the `min` puts it back inside the
        // window, so neither a huge speed nor a tiny one can produce an out-of-range bound here.
        self.ticks.min(delay.saturating_add(over as u64).saturating_add(2))
    }

    fn bar_events(&self, direction: Direction, delay: u64, speed: f64) -> Vec<Event> {
        let mut out = Vec::new();
        let cross = if direction.is_horizontal() { self.height } else { self.width };
        let mut prev = self.occupancy(direction, delay, speed, 0);
        let end = self.last_active_tick(direction, delay, speed);
        for t in 1..end {
            let cur = self.occupancy(direction, delay, speed, t);
            let in_prev = |c: i64| c >= prev.0 && c < prev.1;
            let in_cur = |c: i64| c >= cur.0 && c < cur.1;
            for pol in [Polarity::On, Polarity::Off] {
                let (lo, hi) = if pol == Polarity::On { cur } else { prev };
                for c in lo..hi {
                    let fresh = if pol == Polarity::On { !in_prev(c) } else { !in_cur(c) };
                    if !fresh {
                        continue;
                    }
                    for k in 0..i64::from(cross) {
                        let (x, y) = if direction.is_horizontal() { (c, k) } else { (k, c) };
                        let address = (y * i64::from(self.width) + x) as u32;
                        out.push(Event { t, address, polarity: pol });
                    }
                }
            }
            prev = cur;
        }
        out
    }

    /// Build the dataset.
    ///
    /// # Errors
    ///
    /// [`TaskError::Empty`] for a zero grid dimension, zero bar width or zero window;
    /// [`TaskError::OutOfRange`] for a non-positive or pixel-skipping speed, a speed jitter outside
    /// `[0, 0.9]`, a noise probability outside `[0, 1)` — the top of that range is **exclusive**,
    /// as the field's own doc says, because `p = 1` is a sensor that emits every pixel on every
    /// tick and carries no stimulus at all — a `max_start_delay` too large for a single uniform
    /// draw, a `dt` that is not positive and finite, or a window shorter than
    /// [`MovingBar::min_ticks`]; [`TaskError::Exhausted`] when the jitters admit fewer distinct
    /// streams than the split needs.
    pub fn generate(&self) -> Result<Dataset<EventSample>, TaskError> {
        in_range("dt", self.dt, f64::MIN_POSITIVE, f64::MAX)?;
        in_range("speed_jitter_frac", self.speed_jitter_frac, 0.0, 0.9)?;
        in_range("noise_prob_per_pixel_per_tick", self.noise_prob_per_pixel_per_tick, 0.0, 1.0)?;
        // `in_range` is inclusive at both ends and the field is documented on `[0, 1)`, so the top
        // needs its own line rather than a second reading of the same doc sentence.
        if self.noise_prob_per_pixel_per_tick >= 1.0 {
            return Err(TaskError::OutOfRange {
                what: "noise_prob_per_pixel_per_tick",
                value: self.noise_prob_per_pixel_per_tick,
                low: 0.0,
                high: 1.0,
            });
        }
        let delay_span = draw_span("max_start_delay", self.max_start_delay.saturating_add(1))?;
        nonzero("width", u64::from(self.width))?;
        nonzero("height", u64::from(self.height))?;
        nonzero("bar_width", u64::from(self.bar_width))?;
        nonzero("ticks", self.ticks)?;
        in_range("speed_px_per_tick", self.speed_px_per_tick, f64::MIN_POSITIVE, f64::MAX)?;
        if self.max_speed().ceil() > f64::from(self.bar_width) {
            return Err(TaskError::OutOfRange {
                what: "speed_px_per_tick",
                value: self.max_speed(),
                low: f64::MIN_POSITIVE,
                high: f64::from(self.bar_width),
            });
        }
        // `n_inputs` is the pixel count and is a `u32`, so a grid whose product does not fit one
        // has no address space to be reported in; wrapping it would hand every consumer a channel
        // count smaller than the addresses the events actually carry.
        let pixels = self.width.checked_mul(self.height).ok_or(TaskError::OutOfRange {
            what: "width",
            value: f64::from(self.width),
            low: 1.0,
            high: f64::from(u32::MAX) / f64::from(self.height),
        })?;
        let need = self.min_ticks().ok_or(TaskError::OutOfRange {
            what: "speed_px_per_tick",
            value: self.speed_px_per_tick,
            low: f64::MIN_POSITIVE,
            high: f64::MAX,
        })?;
        if self.ticks < need {
            return Err(TaskError::OutOfRange {
                what: "ticks",
                value: self.ticks as f64,
                low: need as f64,
                high: f64::MAX,
            });
        }

        let mut rng = Rng::new(self.seed);
        let cfg = *self;
        let (train, test) = collect_split(
            4,
            self.per_class_train,
            self.per_class_test,
            self.seed ^ 0xA5A5_0006,
            |class, _| {
                let dir = Direction::all()[class as usize];
                let delay = u64::from(rng.below(delay_span));
                let u = rng.next_f64() * 2.0 - 1.0;
                let speed = cfg.speed_px_per_tick * (1.0 + cfg.speed_jitter_frac * u);
                let mut ev = cfg.bar_events(dir, delay, speed);
                if cfg.noise_prob_per_pixel_per_tick > 0.0 {
                    let pixels = u64::from(cfg.width) * u64::from(cfg.height);
                    for t in 0..cfg.ticks {
                        for a in 0..pixels {
                            if rng.next_f64() < cfg.noise_prob_per_pixel_per_tick {
                                let pol =
                                    if rng.next_f64() < 0.5 { Polarity::On } else { Polarity::Off };
                                ev.push(Event { t, address: a as u32, polarity: pol });
                            }
                        }
                    }
                }
                ev.sort_unstable();
                EventSample { events: ev, label: class }
            },
        )?;

        Ok(finish(
            "moving_bar",
            train,
            test,
            4,
            pixels,
            self.ticks,
            self.dt,
            self.seed,
            "Orchard et al., Frontiers in Neuroscience 9:437, 2015 (`N-MNIST`) and Amir et al., \
             CVPR 2017 (`DVS-Gesture`), for teaching the event representation only",
            "one rigid high-contrast object, no texture, no occlusion, no refractory period per \
             pixel, no latency spread across the array, and noise that is independent per pixel \
             rather than clustered on hot pixels",
        ))
    }
}

// ---------------------------------------------------------------------------------------------
// 7. Spoken digits
// ---------------------------------------------------------------------------------------------

/// **Spoken digits**: several frequency channels with class-specific temporal envelopes — an `SHD`
/// stand-in.
///
/// The Spiking Heidelberg Digits (Cramer et al., IEEE Trans. Neural Netw. Learn. Syst. 33(7), 2022)
/// pass spoken digits through a cochlear model and emit spikes on 700 frequency channels. What a
/// network must find there is a **spectro-temporal trajectory**: energy moving across channels over
/// time. This generator reproduces that one structural feature and nothing else.
///
/// Each class `c` is a straight sweep in the (time, channel) plane: channel `k` has a Gaussian
/// firing-rate envelope peaking at tick `centre + slope(c) * (k - mid)`, where `slope(c)` runs over
/// an integer grid from a down-sweep through flat to an up-sweep. Spikes are drawn per tick with
/// `p = 1 - exp(-rate * dt)`, the same exact form as [`crate::encode::RateEncoder`].
///
/// # What is checkable here
///
/// The expected spike count of a sample is a sum of Bernoulli probabilities and therefore exact:
/// [`SpokenDigits::expected_spikes_per_sample`] sums it, [`SpokenDigits::spike_count_variance`]
/// gives the matching Poisson-binomial variance, and the test compares the empirical mean against
/// them at four standard errors. The expected spike *centroid* of channel `k` is the envelope's
/// peak tick exactly, because the envelope is symmetric about it and `1 - exp(-x)` is monotone —
/// so the sweep's slope is recoverable in closed form and the test asserts it to under a tick.
///
/// Both hold only for `time_jitter_ticks == 0`, because a shifted envelope is truncated differently
/// by the window; the methods return `None` otherwise rather than reporting the unjittered figure.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpokenDigits {
    /// Number of digit classes.
    pub n_classes: u32,
    /// Number of frequency channels. `SHD` has 700; a teaching version wants tens.
    pub n_channels: u32,
    /// Sample length in ticks.
    pub ticks: u64,
    /// Peak instantaneous firing rate of an envelope, hertz.
    pub peak_hz: f64,
    /// Envelope standard deviation in ticks. Sets how long a channel is active.
    pub sigma_ticks: f64,
    /// Sweep slope step, in ticks per channel. Class `c` uses
    /// `slope_step_ticks * (c - (n_classes - 1) / 2)`, an integer, so envelope peaks land on ticks
    /// and the centroid identity is exact.
    pub slope_step_ticks: i64,
    /// Half-width of the uniform common-mode time jitter, in ticks. Non-zero makes the analytic
    /// count and centroid figures inapplicable, and they then return `None`.
    pub time_jitter_ticks: u64,
    /// Tick length in seconds.
    pub dt: f64,
    /// Train samples per class.
    pub per_class_train: usize,
    /// Test samples per class.
    pub per_class_test: usize,
    /// Seed.
    pub seed: u64,
}

impl Default for SpokenDigits {
    /// Five classes over sixteen channels in a 200 ms window at 1 ms ticks: a 200 Hz peak, a 6 ms
    /// envelope, and sweeps of -6, -3, 0, +3, +6 ticks per channel. About 45 spikes per sample
    /// (44.9 for the flat class, from `expected_spikes_per_sample`).
    ///
    /// The window has to hold the steepest sweep plus a four-sigma margin at both ends: the
    /// extreme peak sits `6 * 8 = 48` ticks off centre and needs `4 * 6 = 24` more on each side, so
    /// the floor is `ticks/2 + 48 + 24 < ticks` with `ticks/2 - 48 >= 24`, which first holds at
    /// **145** ticks. 200 is the round number chosen above that floor, with room left to widen
    /// `sigma_ticks` or the sweep without re-deriving it. That floor is not a claim in prose:
    /// `the_window_floor_is_the_one_the_doc_derives` generates at 145 and asserts the refusal at
    /// 144. A shorter window clips an envelope, which changes both the expected count and the
    /// centroid, and `generate` refuses rather than quietly clipping.
    fn default() -> Self {
        Self {
            n_classes: 5,
            n_channels: 16,
            ticks: 200,
            peak_hz: 200.0,
            sigma_ticks: 6.0,
            slope_step_ticks: 3,
            time_jitter_ticks: 0,
            dt: 1e-3,
            per_class_train: 100,
            per_class_test: 50,
            seed: 0x5EED_0007,
        }
    }
}

impl SpokenDigits {
    /// Sweep slope of a class, in ticks per channel. Negative is a down-sweep.
    ///
    /// Saturating rather than wrapping, so an absurd `slope_step_ticks` produces a peak far outside
    /// the window — which [`SpokenDigits::generate`] refuses — instead of wrapping to one inside it.
    #[must_use]
    pub fn slope(&self, class: u32) -> i64 {
        let mid = i64::from(self.n_classes.saturating_sub(1)) / 2;
        self.slope_step_ticks.saturating_mul(i64::from(class) - mid)
    }

    /// Tick at which channel `k` of `class` peaks, before jitter.
    ///
    /// `ticks / 2 + slope(class) * (channel - n_channels / 2)`, saturating for the same reason
    /// [`SpokenDigits::slope`] does.
    #[must_use]
    pub fn peak_tick(&self, class: u32, channel: u32) -> i64 {
        let mid = i64::from(self.n_channels) / 2;
        let centre = (self.ticks / 2) as i64;
        centre.saturating_add(self.slope(class).saturating_mul(i64::from(channel) - mid))
    }

    /// Per-tick spike probability of channel `k` of `class` at tick `t`, with an optional shift.
    fn p_at(&self, class: u32, channel: u32, t: u64, shift: i64) -> f64 {
        let peak = self.peak_tick(class, channel) + shift;
        let d = (t as i64 - peak) as f64 / self.sigma_ticks;
        let rate = self.peak_hz * (-0.5 * d * d).exp();
        1.0 - (-rate * self.dt).exp()
    }

    /// Expected number of spikes in one sample of `class`, exactly.
    ///
    /// The sum of every per-tick Bernoulli probability. `None` when `time_jitter_ticks != 0`, when
    /// `class` is out of range, or when a parameter is not finite — a shifted envelope is clipped
    /// differently by the window, and reporting the unjittered figure would be a number that is
    /// nearly right, which is worse than none.
    #[must_use]
    pub fn expected_spikes_per_sample(&self, class: u32) -> Option<f64> {
        self.moments(class).map(|(m, _)| m)
    }

    /// Variance of the spike count of one sample of `class`, exactly.
    ///
    /// The Poisson-binomial variance `sum p (1 - p)`, which is **smaller** than the mean — the
    /// counts are less variable than a Poisson process of the same rate because a tick can carry at
    /// most one spike. `None` under the same conditions as
    /// [`SpokenDigits::expected_spikes_per_sample`].
    #[must_use]
    pub fn spike_count_variance(&self, class: u32) -> Option<f64> {
        self.moments(class).map(|(_, v)| v)
    }

    fn moments(&self, class: u32) -> Option<(f64, f64)> {
        if class >= self.n_classes || self.time_jitter_ticks != 0 {
            return None;
        }
        if !self.peak_hz.is_finite() || !self.sigma_ticks.is_finite() || !self.dt.is_finite() {
            return None;
        }
        if self.sigma_ticks <= 0.0 || self.dt <= 0.0 {
            return None;
        }
        let mut mean = 0.0f64;
        let mut var = 0.0f64;
        for k in 0..self.n_channels {
            for t in 0..self.ticks {
                let p = self.p_at(class, k, t, 0);
                mean += p;
                var += p * (1.0 - p);
            }
        }
        Some((mean, var))
    }

    /// Build the dataset.
    ///
    /// # Errors
    ///
    /// [`TaskError::Empty`] for zero classes, channels or ticks; [`TaskError::OutOfRange`] when
    /// `sigma_ticks`, `peak_hz` or `dt` is not positive and finite, when `time_jitter_ticks`
    /// exceeds [`MAX_JITTER_HALF`] (which would narrow its uniform draw), or when a class's
    /// envelope peak plus the jitter would fall outside `4 * sigma` of the window edge — which
    /// would clip the envelope and quietly change both the count and the centroid;
    /// [`TaskError::Exhausted`] when the window admits fewer distinct spike patterns than the split
    /// needs.
    pub fn generate(&self) -> Result<Dataset<Sample>, TaskError> {
        in_range("dt", self.dt, f64::MIN_POSITIVE, f64::MAX)?;
        in_range("sigma_ticks", self.sigma_ticks, f64::MIN_POSITIVE, f64::MAX)?;
        in_range("peak_hz", self.peak_hz, f64::MIN_POSITIVE, f64::MAX)?;
        nonzero("n_classes", u64::from(self.n_classes))?;
        nonzero("n_channels", u64::from(self.n_channels))?;
        nonzero("ticks", self.ticks)?;
        // Checked before the margin below, which would otherwise cast the same quantity to `i64`.
        let jit = half_width("time_jitter_ticks", self.time_jitter_ticks)?;

        let margin = (4.0 * self.sigma_ticks).ceil() as i64 + i64::from(jit.0);
        // The floor `ticks` has to reach is set by the peak that sits FURTHEST FROM THE WINDOW
        // CENTRE, and that distance does not move with `ticks`: `peak_tick` is
        // `ticks / 2 + slope * (channel - mid)`, so widening the window carries the centre and
        // every peak along with it. A window of `2 * (furthest + margin) + 1` leaves the extreme
        // peak exactly `margin` inside each edge, which is the arithmetic the `SpokenDigits`
        // default doc works through to reach 145 — so that is the number the refusal reports.
        //
        // It used to report `2 * (|peak| + margin) + 1`, off the peak's ABSOLUTE tick rather than
        // its distance from the centre, and that is a different quantity by the whole half-window.
        // Measured: the default at `ticks = 144` asked for 289 when its own doc, and the test
        // beside this one, put the floor at 145. Worse, a peak clipped at the BOTTOM has a small
        // absolute tick, so a down-sweep of `slope_step_ticks = -7` at 200 ticks reported
        // `low = 97` — a bound the refused value of 200 already satisfies, printed as the reason
        // 200 was refused. Following it to 97 refused again and asked for 177, which refuses again
        // and asks for 97. The true floor for that configuration is 273, and it is what the line
        // below now reports.
        let halfway = (self.ticks / 2) as i64;
        let mut furthest = 0i64;
        let mut clipped = false;
        for c in 0..self.n_classes {
            for k in 0..self.n_channels {
                let p = self.peak_tick(c, k);
                furthest = furthest.max(p.saturating_sub(halfway).saturating_abs());
                if p.saturating_sub(margin) < 0 || p.saturating_add(margin) >= self.ticks as i64 {
                    clipped = true;
                }
            }
        }
        if clipped {
            return Err(TaskError::OutOfRange {
                what: "ticks",
                value: self.ticks as f64,
                low: furthest.saturating_add(margin).saturating_mul(2).saturating_add(1) as f64,
                high: f64::MAX,
            });
        }

        let mut rng = Rng::new(self.seed);
        let cfg = *self;
        let (train, test) = collect_split(
            self.n_classes,
            self.per_class_train,
            self.per_class_test,
            self.seed ^ 0xA5A5_0007,
            |class, _| {
                let shift = jitter(&mut rng, jit);
                let mut sp = Vec::new();
                for t in 0..cfg.ticks {
                    for k in 0..cfg.n_channels {
                        if rng.next_f64() < cfg.p_at(class, k, t, shift) {
                            sp.push(Spike { t, source: k });
                        }
                    }
                }
                Sample { train: Train::from_spikes(sp), label: class }
            },
        )?;

        Ok(finish(
            "spoken_digits",
            train,
            test,
            self.n_classes,
            self.n_channels,
            self.ticks,
            self.dt,
            self.seed,
            "Cramer et al., IEEE Trans. Neural Netw. Learn. Syst. 33(7), 2022 (`SHD`), for the \
             spectro-temporal sweep structure only",
            "a real cochleagram has formants, harmonics, speaker variability, variable duration \
             and a channel count of 700; this has one straight sweep per class and no speaker at \
             all",
        ))
    }
}

// ---------------------------------------------------------------------------------------------
// The catalogue
// ---------------------------------------------------------------------------------------------

/// One row of [`CATALOGUE`]: what a task is for, at a glance.
///
/// Kept as data rather than as prose so that a course page, a report header or a results table can
/// be generated from it, and so that the chance level printed beside a score comes from the same
/// place the task does.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TaskCard {
    /// Identifier, matching [`Dataset::name`] of the generated dataset.
    pub name: &'static str,
    /// Number of classes at the task's default settings.
    pub classes: u32,
    /// Uniform-guessing accuracy at the default settings, `1 / classes`.
    pub chance: f64,
    /// The recorded dataset this substitutes for, or a note that there is none.
    pub stands_in_for: &'static str,
    /// The single mechanism the task isolates. If a model fails here, this is what failed.
    pub isolates: &'static str,
}

/// Every task in this module, with its chance level and the mechanism it isolates.
///
/// `every_catalogue_row_matches_the_task_it_names` asserts that each row's name, class count and
/// chance level agree with the dataset the corresponding generator actually produces — so the table
/// cannot drift away from the code while continuing to look authoritative.
pub const CATALOGUE: [TaskCard; 7] = [
    TaskCard {
        name: "temporal_xor",
        classes: 2,
        chance: 0.5,
        stands_in_for: "no recorded dataset; a constructed proof",
        isolates: "coincidence of two channels, with spike counts held identical across classes",
    },
    TaskCard {
        name: "coincidence",
        classes: 2,
        chance: 0.5,
        stands_in_for: "no recorded dataset",
        isolates: "one inter-spike interval against a threshold — the membrane time constant",
    },
    TaskCard {
        name: "delayed_match_to_sample",
        classes: 2,
        chance: 0.5,
        stands_in_for: "Fuster and Alexander, Science 173:652-654, 1971",
        isolates: "holding a symbol across a silent delay",
    },
    TaskCard {
        name: "rate_discrimination",
        classes: 2,
        chance: 0.5,
        stands_in_for: "two-alternative rate discrimination psychophysics",
        isolates: "counting under Poisson-like noise, with a computable optimum",
    },
    TaskCard {
        name: "latency_patterns",
        classes: 5,
        chance: 0.2,
        stands_in_for: "no recorded dataset",
        isolates: "time-to-first-spike patterns under per-channel jitter",
    },
    TaskCard {
        name: "moving_bar",
        classes: 4,
        chance: 0.25,
        stands_in_for: "`N-MNIST` (Orchard 2015) and `DVS-Gesture` (Amir 2017)",
        isolates: "direction of motion in an address-event stream, with polarity",
    },
    TaskCard {
        name: "spoken_digits",
        classes: 5,
        chance: 0.2,
        stands_in_for: "`SHD` (Cramer 2022)",
        isolates: "a spectro-temporal sweep across frequency channels",
    },
];

// ---------------------------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::{
        CATALOGUE, Coincidence, Dataset, DelayedMatch, Direction, LatencyPatterns, MovingBar,
        RateDiscrimination, Split, SpokenDigits, TaskError, TemporalXor, polarity_counts,
    };
    use crate::rng::Rng;
    use crate::spike::{Polarity, Spike, Train};

    // -- (a) determinism -----------------------------------------------------------------------

    /// The property the whole module rests on: a seed names a dataset, on every platform.
    #[test]
    fn every_task_is_reproducible_from_its_seed_alone() {
        assert_eq!(TemporalXor::default().generate(), TemporalXor::default().generate());
        assert_eq!(Coincidence::default().generate(), Coincidence::default().generate());
        assert_eq!(DelayedMatch::default().generate(), DelayedMatch::default().generate());
        assert_eq!(
            RateDiscrimination::default().generate(),
            RateDiscrimination::default().generate()
        );
        assert_eq!(LatencyPatterns::default().generate(), LatencyPatterns::default().generate());
        assert_eq!(MovingBar::default().generate(), MovingBar::default().generate());
        assert_eq!(SpokenDigits::default().generate(), SpokenDigits::default().generate());
    }

    #[test]
    fn a_different_seed_gives_a_different_dataset() {
        let a = TemporalXor::default().generate().unwrap();
        let b = TemporalXor { seed: 99, ..TemporalXor::default() }.generate().unwrap();
        assert_ne!(a.train, b.train, "two seeds produced the same training set");
    }

    // -- (b) disjointness ----------------------------------------------------------------------

    /// Leakage is the single most common defect in a benchmark harness, so every task asserts it.
    #[test]
    fn no_task_leaks_a_test_input_into_its_training_set() {
        assert_eq!(TemporalXor::default().generate().unwrap().overlap(), 0, "temporal_xor");
        assert_eq!(Coincidence::default().generate().unwrap().overlap(), 0, "coincidence");
        assert_eq!(DelayedMatch::default().generate().unwrap().overlap(), 0, "dms");
        assert_eq!(RateDiscrimination::default().generate().unwrap().overlap(), 0, "rate");
        assert_eq!(LatencyPatterns::default().generate().unwrap().overlap(), 0, "latency");
        assert_eq!(MovingBar::default().generate().unwrap().overlap(), 0, "moving_bar");
        assert_eq!(SpokenDigits::default().generate().unwrap().overlap(), 0, "spoken_digits");
    }

    /// `overlap` has to be able to SEE a leak, or the zeros above mean nothing. Plant one.
    #[test]
    fn overlap_detects_a_planted_leak() {
        let mut d = Coincidence::default().generate().unwrap();
        let stolen = d.train[0].clone();
        d.test.push(stolen);
        assert_eq!(d.overlap(), 1, "a copied training sample was not detected in the test split");
    }

    // -- (c) class balance ---------------------------------------------------------------------

    #[test]
    fn every_task_has_the_class_balance_it_advertises() {
        let check = |counts: Vec<usize>, want: usize, name: &str| {
            for (c, n) in counts.iter().enumerate() {
                assert_eq!(*n, want, "{name}: class {c} has {n} samples, expected {want}");
            }
        };
        let x = TemporalXor::default().generate().unwrap();
        check(x.counts(Split::Train), 128, "temporal_xor train");
        check(x.counts(Split::Test), 64, "temporal_xor test");
        assert!((x.chance - 0.5).abs() < 1e-15);
        assert_eq!(x.majority_baseline, Some(0.5));

        let l = LatencyPatterns::default().generate().unwrap();
        check(l.counts(Split::Test), 50, "latency test");
        assert!((l.chance - 0.2).abs() < 1e-15);
        assert_eq!(l.majority_baseline, Some(0.2));

        let m = MovingBar::default().generate().unwrap();
        check(m.counts(Split::Test), 24, "moving_bar test");
        assert!((m.chance - 0.25).abs() < 1e-15);

        let s = SpokenDigits::default().generate().unwrap();
        check(s.counts(Split::Test), 50, "spoken_digits test");

        let c = Coincidence::default().generate().unwrap();
        check(c.counts(Split::Train), 128, "coincidence train");
        check(c.counts(Split::Test), 64, "coincidence test");

        let m2 = DelayedMatch::default().generate().unwrap();
        check(m2.counts(Split::Test), 64, "dms test");
        assert_eq!(m2.majority_baseline, Some(0.5));

        let r = RateDiscrimination::default().generate().unwrap();
        check(r.counts(Split::Test), 400, "rate test");
        assert_eq!(r.majority_baseline, Some(0.5));

        // And the balance vector agrees with the counts it was derived from.
        for d in [&x.balance(Split::Test), &l.balance(Split::Test)] {
            let total: f64 = d.iter().sum();
            assert!((total - 1.0).abs() < 1e-12, "class shares summed to {total}");
        }
    }

    /// The majority baseline is MEASURED, not assumed. Unbalance a split and it must move, or the
    /// field is decorative and the number beside every score is wrong.
    #[test]
    fn the_majority_baseline_follows_an_unbalanced_split() {
        let d = TemporalXor::default().generate().unwrap();
        assert_eq!(d.majority_baseline, Some(0.5));
        let mut e = d.clone();
        // Drop three quarters of class 1, deterministically by position.
        let mut seen = 0usize;
        e.test.retain(|s| {
            if s.label == 0 {
                return true;
            }
            seen += 1;
            seen.is_multiple_of(4)
        });
        let bal = e.balance(Split::Test);
        let want = bal.iter().copied().fold(0.0f64, f64::max);
        let re = super::finish(
            e.name, e.train, e.test, e.n_classes, e.n_inputs, e.ticks, e.dt, e.seed,
            e.stands_in_for, e.not_captured,
        );
        let got = re.majority_baseline.expect("a non-empty test split has a majority class");
        assert!((got - want).abs() < 1e-12);
        assert!(got > 0.5, "baseline {got} did not move");
    }

    // -- (e) temporal XOR is blind to every per-channel readout ---------------------------------

    /// THE claim of this module. The two classes hand a counting readout the identical vector.
    #[test]
    fn temporal_xor_gives_the_two_classes_identical_spike_counts() {
        let d = TemporalXor::default().generate().unwrap();
        for split in [Split::Train, Split::Test] {
            let c0 = d.channel_counts(split, 0);
            let c1 = d.channel_counts(split, 1);
            assert_eq!(c0, c1, "{split:?}: class counts {c0:?} vs {c1:?}");
            // And it is not vacuously zero.
            assert!(c0.iter().sum::<u64>() > 0);
        }
    }

    /// Stronger, and the reason the four conditions are balanced: even a PER-CHANNEL LATENCY
    /// readout — one number per channel — sees the same multiset for both classes. The information
    /// exists only in the joint timing.
    #[test]
    fn temporal_xor_is_blind_to_every_per_channel_readout() {
        let t = TemporalXor::default();
        let mut by_class: [Vec<Vec<u64>>; 2] = [Vec::new(), Vec::new()];
        for (a, b, label) in TemporalXor::conditions() {
            let s = t.condition_sample(a, b);
            let times: Vec<u64> = (0..2)
                .map(|ch| Dataset::first_spike(&s, ch).expect("both channels fire"))
                .collect();
            assert_eq!(s.label, label, "the truth table and condition_sample disagree");
            by_class[label as usize].push(times);
        }
        // Per channel, sort the two classes' first-spike times and compare. Identical multisets.
        for ch in 0..2usize {
            let mut a: Vec<u64> = by_class[0].iter().map(|v| v[ch]).collect();
            let mut b: Vec<u64> = by_class[1].iter().map(|v| v[ch]).collect();
            a.sort_unstable();
            b.sort_unstable();
            assert_eq!(a, b, "channel {ch}: class 0 latencies {a:?} vs class 1 {b:?}");
        }
    }

    /// The generator's own contract, which the blindness argument rests on: **every** channel
    /// carries exactly `spikes_per_channel` spikes, evenly spaced, in every sample of both classes.
    ///
    /// Comparing the two classes' totals is not enough. A generator that put four spikes on
    /// channel 1 and three on channel 0 would still hand both classes the identical vector `(3, 4)`
    /// and pass that comparison, while the task it described — one volley per channel — would no
    /// longer be the task it generated.
    #[test]
    fn every_temporal_xor_sample_carries_exactly_one_even_volley_per_channel() {
        let t = TemporalXor::default();
        let d = t.generate().unwrap();
        for s in d.train.iter().chain(d.test.iter()) {
            for ch in 0..2u32 {
                let ts: Vec<u64> =
                    s.train.spikes().iter().filter(|sp| sp.source == ch).map(|sp| sp.t).collect();
                assert_eq!(
                    ts.len() as u32,
                    t.spikes_per_channel,
                    "channel {ch} carried {} spikes, not {}",
                    ts.len(),
                    t.spikes_per_channel
                );
                for w in ts.windows(2) {
                    assert_eq!(w[1] - w[0], t.burst_gap, "volley spacing on channel {ch}");
                }
            }
        }
    }

    /// Global distinctness, not merely train-versus-test distinctness. The dedup key set spans
    /// every class, so an input cannot recur anywhere — which is also what stops one input from
    /// carrying two labels, a contradiction in the task rather than a leak.
    #[test]
    fn no_input_appears_twice_anywhere_in_a_dataset() {
        let mut cases: Vec<(&str, Dataset<super::Sample>)> = vec![
            ("temporal_xor", TemporalXor::default().generate().unwrap()),
            ("coincidence", Coincidence::default().generate().unwrap()),
            ("delayed_match", DelayedMatch::default().generate().unwrap()),
            ("latency_patterns", LatencyPatterns::default().generate().unwrap()),
            ("spoken_digits", SpokenDigits::default().generate().unwrap()),
        ];
        for (name, d) in &mut cases {
            let mut keys: Vec<Vec<u64>> =
                d.train.iter().chain(d.test.iter()).map(super::Keyed::key).collect();
            let n = keys.len();
            keys.sort();
            keys.dedup();
            assert_eq!(n, keys.len(), "{name}: {} of {n} inputs were repeats", n - keys.len());
        }
        let m = MovingBar::default().generate().unwrap();
        let mut keys: Vec<Vec<u64>> =
            m.train.iter().chain(m.test.iter()).map(super::Keyed::key).collect();
        let n = keys.len();
        keys.sort();
        keys.dedup();
        assert_eq!(n, keys.len(), "moving_bar: {} of {n} streams were repeats", n - keys.len());
    }

    /// And the timing code DOES solve it — exactly, when the separability condition holds.
    #[test]
    fn the_coincidence_rule_solves_temporal_xor_perfectly() {
        let t = TemporalXor::default();
        assert!(t.coincidence_separable(), "the default config is not separable");
        let d = t.generate().unwrap();
        for s in d.test.iter().chain(d.train.iter()) {
            let got = t.classify_by_coincidence(s).expect("both channels fire");
            assert_eq!(got, s.label, "coincidence rule missed a sample");
        }
    }

    #[test]
    fn a_jitter_free_temporal_xor_refuses_rather_than_repeating_four_inputs() {
        let t = TemporalXor { jitter_ticks: 0, ..TemporalXor::default() };
        match t.generate() {
            Err(TaskError::Exhausted { distinct, .. }) => {
                assert!(distinct <= 4, "found {distinct} distinct inputs where only 4 exist");
            }
            other => panic!("expected Exhausted, got {other:?}"),
        }
    }

    // -- coincidence ---------------------------------------------------------------------------

    #[test]
    fn the_interval_rule_solves_coincidence_perfectly_and_counts_do_not() {
        let c = Coincidence::default();
        let d = c.generate().unwrap();
        for s in d.test.iter().chain(d.train.iter()) {
            assert_eq!(c.classify_by_interval(s).unwrap(), s.label);
        }
        // Rate-blind for the same reason as temporal XOR: one spike per channel, both classes.
        assert_eq!(d.channel_counts(Split::Test, 0), d.channel_counts(Split::Test, 1));
    }

    /// The gap distribution has to match the label, or the task's threshold is decorative.
    #[test]
    fn every_coincidence_sample_falls_on_the_side_of_the_threshold_its_label_claims() {
        let c = Coincidence::default();
        let d = c.generate().unwrap();
        for s in d.train.iter().chain(d.test.iter()) {
            let t0 = Dataset::first_spike(s, 0).unwrap();
            let t1 = Dataset::first_spike(s, 1).unwrap();
            let gap = t0.abs_diff(t1);
            if s.label == 1 {
                assert!(gap <= c.threshold_ticks, "coincident sample with gap {gap}");
            } else {
                assert!(gap > c.threshold_ticks, "non-coincident sample with gap {gap}");
                assert!(gap <= c.max_gap_ticks, "gap {gap} exceeded max_gap_ticks");
            }
        }
    }

    #[test]
    fn a_coincidence_task_with_no_room_for_a_non_match_refuses() {
        let c = Coincidence { max_gap_ticks: 5, threshold_ticks: 5, ..Coincidence::default() };
        assert!(matches!(c.generate(), Err(TaskError::OutOfRange { what: "max_gap_ticks", .. })));
    }

    // -- delayed match-to-sample ---------------------------------------------------------------

    /// The invariant that makes it a memory task: nothing arrives during the delay.
    #[test]
    fn delayed_match_holds_an_empty_delay_window() {
        let t = DelayedMatch::default();
        assert_eq!(t.distractor_cues, 0);
        let d = t.generate().unwrap();
        for s in d.train.iter().chain(d.test.iter()) {
            for sp in s.train.spikes() {
                assert!(
                    sp.t < t.delay_start() || sp.t >= t.delay_end(),
                    "a spike at tick {} landed inside the delay [{}, {})",
                    sp.t,
                    t.delay_start(),
                    t.delay_end()
                );
            }
            assert_eq!(s.train.len(), 2 * t.cue_spikes as usize);
        }
        // Five membrane constants of a default Lif, so the membrane cannot be the answer.
        assert!((t.required_memory_seconds() - 0.1).abs() < 1e-12);
    }

    /// The shortcut is real, is exactly 1.0, and distractors break it. Stated as a measurement
    /// because a baseline that is not measured gets quoted wrong.
    #[test]
    fn the_count_shortcut_is_perfect_without_distractors_and_degrades_with_them() {
        let clean = DelayedMatch::default();
        let d = clean.generate().unwrap();
        let acc = |t: &DelayedMatch, d: &Dataset<super::Sample>| {
            let n = d.test.len();
            let ok = d.test.iter().filter(|s| t.count_readout(s) == s.label).count();
            ok as f64 / n as f64
        };
        let a0 = acc(&clean, &d);
        assert!((a0 - 1.0).abs() < 1e-15, "the shortcut scored {a0}, expected exactly 1.0");

        let noisy = DelayedMatch { distractor_cues: 3, ..DelayedMatch::default() };
        let dn = noisy.generate().unwrap();
        let a1 = acc(&noisy, &dn);
        assert!(a1 < 0.95, "distractors left the count shortcut at {a1}");
    }

    #[test]
    fn distractors_put_exactly_the_promised_number_of_spikes_in_the_delay() {
        let t = DelayedMatch { distractor_cues: 3, ..DelayedMatch::default() };
        let d = t.generate().unwrap();
        let want = (t.distractor_cues * t.cue_spikes) as usize;
        for s in d.test.iter() {
            let inside = s
                .train
                .spikes()
                .iter()
                .filter(|sp| sp.t >= t.delay_start() && sp.t < t.delay_end())
                .count();
            assert_eq!(inside, want, "delay carried {inside} spikes, expected {want}");
        }
    }

    // -- (d) rate discrimination against its analytic optimum -----------------------------------

    /// Equal rates: the task is impossible and the bound must say exactly 0.5, not approximately.
    #[test]
    fn equal_rates_have_an_optimal_accuracy_of_exactly_one_half() {
        let t = RateDiscrimination { rate_hi: 50.0, rate_lo: 50.0, ..RateDiscrimination::default() };
        let a = t.optimal_accuracy().unwrap();
        assert!((a - 0.5).abs() < 1e-12, "binomial bound {a}");
        let p = t.poisson_optimal_accuracy().unwrap();
        assert!((p - 0.5).abs() < 1e-12, "poisson bound {p}");
    }

    /// A silent slow channel gives the bound in ELEMENTARY closed form. Binomial:
    /// `1 - 0.5 (1 - p)^n`. Poisson: `1 - 0.5 exp(-mu)`. Both checked against the summation.
    #[test]
    fn a_silent_channel_reproduces_the_elementary_closed_form() {
        let t = RateDiscrimination {
            rate_hi: 30.0,
            rate_lo: 0.0,
            ticks: 100,
            dt: 1e-3,
            ..RateDiscrimination::default()
        };
        let p = t.p_tick(30.0).unwrap();
        let want_binom = 1.0 - 0.5 * (1.0 - p).powi(100);
        let got_binom = t.optimal_accuracy().unwrap();
        assert!(
            (got_binom - want_binom).abs() < 1e-12,
            "summed {got_binom} vs closed form {want_binom}"
        );

        let mu: f64 = 30.0 * 100.0 * 1e-3;
        let want_pois = 1.0 - 0.5 * (-mu).exp();
        let got_pois = t.poisson_optimal_accuracy().unwrap();
        assert!(
            (got_pois - want_pois).abs() < 1e-12,
            "summed {got_pois} vs closed form {want_pois}"
        );
    }

    /// The discrete-time bound must converge to the continuous-time one the literature quotes, as
    /// the tick shrinks at fixed expected counts. If it does not, one of the two is wrong.
    #[test]
    fn the_binomial_bound_converges_to_the_poisson_bound() {
        let mut prev = f64::INFINITY;
        for k in [1u32, 4, 16, 64] {
            let t = RateDiscrimination {
                rate_hi: 60.0,
                rate_lo: 40.0,
                ticks: u64::from(k) * 100,
                dt: 1e-3 / f64::from(k),
                ..RateDiscrimination::default()
            };
            let b = t.optimal_accuracy().unwrap();
            let p = t.poisson_optimal_accuracy().unwrap();
            let gap = (b - p).abs();
            assert!(gap < prev, "gap {gap} did not shrink below {prev} at k = {k}");
            prev = gap;
        }
        assert!(prev < 2e-3, "the finest tick still differed by {prev}");
    }

    /// `p_tick` feeds BOTH the generator and the analytic bound, so a check that compared them to
    /// each other would validate whatever formula is written there — the two would move together.
    /// These two properties are independent of the formula and the linear approximation
    /// `rate * dt` fails both.
    #[test]
    fn the_per_tick_probability_is_the_exact_poisson_form_not_its_linear_approximation() {
        // 1. A probability is never above one. `rate * dt` here is 5.0, which is not one.
        let t = RateDiscrimination { dt: 1e-3, ..RateDiscrimination::default() };
        let p = t.p_tick(5_000.0).unwrap();
        assert!((0.0..=1.0).contains(&p), "p_tick returned {p}, which is not a probability");
        assert!(p > 0.99, "a 5 kHz channel at 1 ms ticks should almost always fire, got {p}");

        // 2. Surviving two ticks IS surviving one tick twice — the functional equation that
        //    defines a Poisson process. `(1 - r dt)^2` misses `1 - 2 r dt` by exactly `(r dt)^2`.
        for &r in &[10.0f64, 60.0, 400.0, 2_000.0] {
            let fine = RateDiscrimination { dt: 1e-3, ..RateDiscrimination::default() };
            let coarse = RateDiscrimination { dt: 2e-3, ..RateDiscrimination::default() };
            let s1 = 1.0 - fine.p_tick(r).unwrap();
            let s2 = 1.0 - coarse.p_tick(r).unwrap();
            assert!(
                (s1 * s1 - s2).abs() < 1e-15,
                "rate {r}: one tick survived twice is {} but two ticks is {s2}",
                s1 * s1
            );
        }
    }

    #[test]
    fn a_wider_rate_gap_is_easier_and_a_longer_window_is_easier() {
        let base = RateDiscrimination::default();
        let wide = RateDiscrimination { rate_lo: 20.0, ..base };
        let long = RateDiscrimination { ticks: 400, ..base };
        let a = base.optimal_accuracy().unwrap();
        assert!(wide.optimal_accuracy().unwrap() > a);
        assert!(long.optimal_accuracy().unwrap() > a);
        assert!(a > 0.5 && a < 1.0, "the default bound {a} is degenerate");
    }

    /// THE closed-form check: the optimal counting rule on real generated data must land on the
    /// analytically computed optimum, within sampling error.
    #[test]
    fn the_counting_classifier_reaches_the_analytic_optimum() {
        let t = RateDiscrimination {
            per_class_train: 4,
            per_class_test: 2_000,
            ..RateDiscrimination::default()
        };
        let d = t.generate().unwrap();
        let want = t.optimal_accuracy().unwrap();
        let n = d.test.len();
        let ok = d
            .test
            .iter()
            .filter(|s| RateDiscrimination::classify_by_count(s) == s.label)
            .count();
        let got = ok as f64 / n as f64;
        // Binomial standard error of the measured accuracy at this n, times 3.5. At n = 4000 and
        // p = 0.7324 that is 0.0245 — loose enough never to flake, tight enough that a classifier
        // 5% off the optimum fails.
        let se = (want * (1.0 - want) / n as f64).sqrt();
        assert!(
            (got - want).abs() < 3.5 * se,
            "measured {got} vs analytic optimum {want} (3.5 SE = {})",
            3.5 * se
        );
        assert!(want > 0.6, "the default task is too easy to be a test: bound {want}");
    }

    #[test]
    fn an_inverted_rate_pair_has_no_bound_rather_than_one_below_chance() {
        let t = RateDiscrimination { rate_hi: 10.0, rate_lo: 50.0, ..RateDiscrimination::default() };
        assert!(t.optimal_accuracy().is_none());
        assert!(t.poisson_optimal_accuracy().is_none());
        assert!(matches!(t.generate(), Err(TaskError::OutOfRange { what: "rate_lo", .. })));
    }

    #[test]
    fn a_window_of_no_length_has_no_discrimination_rather_than_chance() {
        let t = RateDiscrimination { ticks: 0, ..RateDiscrimination::default() };
        assert!(t.optimal_accuracy().is_none());
        assert!(t.poisson_optimal_accuracy().is_none());
    }

    /// A NaN must be named, not propagated. A NaN rate gives a NaN probability, every comparison
    /// against it is false, the task generates no spikes at all, and the model reports chance —
    /// which reads as a modelling failure rather than as a typo three call sites away.
    #[test]
    fn a_non_finite_rate_is_refused_and_names_itself() {
        let t = RateDiscrimination { rate_hi: f64::NAN, ..RateDiscrimination::default() };
        match t.generate() {
            Err(TaskError::NotFinite { what, value }) => {
                assert_eq!(what, "rate_hi");
                assert!(value.is_nan());
            }
            other => panic!("a NaN rate produced {other:?}"),
        }
        assert!(t.p_tick(f64::NAN).is_none());
        assert!(t.optimal_accuracy().is_none());

        let d = MovingBar { speed_px_per_tick: f64::INFINITY, ..MovingBar::default() };
        assert!(matches!(d.generate(), Err(TaskError::NotFinite { .. })));
    }

    // -- latency patterns ----------------------------------------------------------------------

    /// The separability theorem, run: separation above twice the jitter means nearest-template is
    /// EXACTLY perfect. Not "high" — perfect, by the triangle inequality.
    #[test]
    fn nearest_template_is_exactly_perfect_when_separation_beats_twice_the_jitter() {
        let t = LatencyPatterns::default();
        assert!(t.guaranteed_separable());
        let templates = t.templates().unwrap();

        // Premise 1, with the ACTUAL separation rather than the requested one: the configured
        // minimum is a lower bound the sampler enforces, and a test that checked the config would
        // be checking a number nobody measured.
        let mut actual_sep = u64::MAX;
        for i in 0..templates.len() {
            for j in (i + 1)..templates.len() {
                actual_sep = actual_sep.min(super::linf(&templates[i], &templates[j]));
            }
        }
        assert!(
            actual_sep > t.min_separation_ticks,
            "templates are {actual_sep} apart, below the requested {}",
            t.min_separation_ticks
        );

        let d = t.generate().unwrap();

        // Premise 2, which is the generator's side of the contract: every sample lies within
        // `jitter_ticks` of its own template in L-infinity. Without this the theorem below is
        // proved about a jitter the generator does not actually respect — a larger jitter leaves
        // the perfect score standing on luck rather than on the triangle inequality.
        for s in d.test.iter().chain(d.train.iter()) {
            let times: Vec<u64> = (0..t.n_inputs)
                .map(|c| Dataset::first_spike(s, c).expect("every channel fires"))
                .collect();
            let r = super::linf(&templates[s.label as usize], &times);
            assert!(r <= t.jitter_ticks, "a sample sat {r} from its template, jitter is {}", t.jitter_ticks);
        }
        assert!(
            actual_sep > 2 * t.jitter_ticks,
            "separation {actual_sep} does not beat twice the jitter"
        );

        // The conclusion.
        for s in d.test.iter().chain(d.train.iter()) {
            assert_eq!(t.nearest_template(&templates, s).unwrap(), s.label);
        }
    }

    #[test]
    fn every_latency_sample_fires_every_channel_exactly_once() {
        let t = LatencyPatterns::default();
        let d = t.generate().unwrap();
        for s in d.train.iter().chain(d.test.iter()) {
            assert_eq!(s.train.len(), t.n_inputs as usize);
            let mut seen = vec![false; t.n_inputs as usize];
            for sp in s.train.spikes() {
                assert!(!seen[sp.source as usize], "channel {} fired twice", sp.source);
                seen[sp.source as usize] = true;
                assert!(sp.t < t.ticks);
            }
        }
    }

    #[test]
    fn an_impossible_separation_refuses_rather_than_placing_overlapping_templates() {
        let t = LatencyPatterns { min_separation_ticks: 1_000, ..LatencyPatterns::default() };
        assert!(matches!(t.templates(), Err(TaskError::Exhausted { .. })));
    }

    // -- (f) the moving bar and its speed scaling ----------------------------------------------

    /// THE event-vision lesson, as an exact count: a full traverse emits `2 * w * h` events at ANY
    /// speed, and exactly half of them are on-events.
    #[test]
    fn a_full_traverse_emits_exactly_two_events_per_pixel_at_every_speed() {
        let b = MovingBar { bar_width: 4, ticks: 400, ..MovingBar::default() };
        let want = b.events_per_traverse() as usize;
        let pixels = (b.width * b.height) as usize;
        for &v in &[0.25f64, 0.5, 1.0, 2.0, 4.0] {
            for dir in Direction::all() {
                let ev = b.traverse(dir, v).unwrap();
                assert_eq!(ev.len(), want, "{dir:?} at {v} px/tick emitted {}", ev.len());
                let on = ev.iter().filter(|e| e.polarity == Polarity::On).count();
                assert_eq!(on, pixels, "{dir:?} at {v}: {on} on-events, expected {pixels}");
                // Every pixel exactly once in each polarity — the count alone would pass if one
                // pixel fired twice and another never did.
                for pol in [Polarity::On, Polarity::Off] {
                    let mut seen = vec![0u8; pixels];
                    for e in ev.iter().filter(|e| e.polarity == pol) {
                        seen[e.address as usize] += 1;
                    }
                    assert!(
                        seen.iter().all(|&c| c == 1),
                        "{dir:?} at {v}, {pol:?}: uneven pixel coverage"
                    );
                }
            }
        }
    }

    /// Speed does not change the data volume, it changes the LATENCY. Doubling the speed halves
    /// the time the events take to arrive, so the event rate doubles.
    ///
    /// The count is taken from each returned stream rather than from `events_per_traverse()`. With
    /// the analytic count on both sides the rate ratio was `(n/fast)/(n/slow) = slow/fast` — the
    /// span assertion's own number a second time, so "the event rate doubles" was checked zero
    /// times rather than twice. Measured, the independent content is the first assertion: the two
    /// speeds emit the SAME number of events. The rate claim is that fact divided by the span, and
    /// it is kept because it is the sentence the module doc makes.
    #[test]
    fn doubling_the_speed_halves_the_traverse_and_doubles_the_event_rate() {
        let b = MovingBar { bar_width: 4, ticks: 400, ..MovingBar::default() };
        let measure = |v: f64| {
            let ev = b.traverse(Direction::Right, v).unwrap();
            let first = ev.first().unwrap().t;
            let last = ev.last().unwrap().t;
            (ev.len() as f64, (last - first + 1) as f64)
        };
        let (n_slow, slow) = measure(0.5);
        let (n_fast, fast) = measure(1.0);
        assert!(
            (n_slow - n_fast).abs() < f64::EPSILON,
            "the event count moved with the speed: {n_slow} at 0.5 px/tick, {n_fast} at 1.0"
        );
        let ratio = slow / fast;
        assert!((ratio - 2.0).abs() < 0.1, "traverse span ratio {ratio}, expected 2");
        let rate_slow = n_slow / slow;
        let rate_fast = n_fast / fast;
        assert!(
            ((rate_fast / rate_slow) - 2.0).abs() < 0.1,
            "event rate ratio {}",
            rate_fast / rate_slow
        );
    }

    /// Exact `(time, x)` and `(time, y)` covariances of a stream's on-events, in INTEGERS.
    ///
    /// `n * sum(t * x) - sum(t) * sum(x)`, which is `n^2` times the covariance and has its sign.
    /// Integer arithmetic on purpose: a bar moving along `x` covers a whole column at every tick,
    /// so the `y` sum per tick is the same constant and the `(t, y)` covariance is **exactly**
    /// zero — a fact about the geometry that a floating-point accumulation would turn into "small".
    fn motion_covariances(b: &MovingBar, ev: &[super::Event]) -> (i128, i128) {
        let (mut st, mut sx, mut sy, mut stx, mut sty, mut n) = (0i128, 0i128, 0i128, 0i128, 0i128, 0i128);
        for e in ev.iter().filter(|e| e.polarity == Polarity::On) {
            let t = i128::from(e.t);
            let x = i128::from(e.address % b.width);
            let y = i128::from(e.address / b.width);
            st += t;
            sx += x;
            sy += y;
            stx += t * x;
            sty += t * y;
            n += 1;
        }
        (n * stx - st * sx, n * sty - st * sy)
    }

    /// The direction a stream actually travels, read off the events with no help from
    /// [`Direction::all`] or from `occupancy`: `Right` is increasing `x`, `Down` is increasing `y`,
    /// and an address is `y * width + x`. Exactly one covariance is zero; the other one's sign is
    /// the answer.
    fn decode_direction(b: &MovingBar, ev: &[super::Event]) -> Direction {
        let (cx, cy) = motion_covariances(b, ev);
        assert!(
            (cx == 0) != (cy == 0),
            "neither axis was stationary: cov(t,x) = {cx}, cov(t,y) = {cy}"
        );
        if cy == 0 {
            if cx > 0 { Direction::Right } else { Direction::Left }
        } else if cy > 0 {
            Direction::Down
        } else {
            Direction::Up
        }
    }

    /// THE thing that stops this benchmark from silently becoming a three-class problem printing
    /// `chance = 0.25`.
    ///
    /// Nothing else in the module pins the class-to-direction map: `Direction::all()` could be
    /// `[Right, Right, Down, Up]` — two classes drawing the identical stimulus, a real ceiling of
    /// 0.75 — and every other test would pass, because global dedup only forbids identical streams
    /// and the catalogue check reads only the name, the class count and the chance level. So the
    /// expected order is written out here as a literal, not fetched from `all()`, and each class's
    /// stimulus is decoded from its own events and compared against it.
    #[test]
    fn every_moving_bar_class_travels_the_way_its_direction_names() {
        // The documented order, spelled out: a class index and a direction are interconvertible.
        let expected = [Direction::Right, Direction::Left, Direction::Down, Direction::Up];
        assert_eq!(Direction::all(), expected, "the class order drifted from the documented one");
        let mut distinct = expected.to_vec();
        distinct.dedup();
        assert_eq!(distinct.len(), 4, "two classes name the same direction");

        let b = MovingBar { per_class_train: 6, per_class_test: 6, ..MovingBar::default() };
        let d = b.generate().unwrap();
        for s in d.train.iter().chain(d.test.iter()) {
            let got = decode_direction(&b, &s.events);
            assert_eq!(
                got, expected[s.label as usize],
                "a sample labelled {} travels {got:?}",
                s.label
            );
        }
        // And on a bare traverse, where there is no label to hide behind.
        for dir in expected {
            assert_eq!(decode_direction(&b, &b.traverse(dir, 1.0).unwrap()), dir);
        }
    }

    /// The polarity convention, which the module doc states in words and nothing held in place:
    /// swapping the two emissions passes every other test, because each pixel still appears once
    /// in each polarity and the totals are unchanged.
    ///
    /// The invariant that is not symmetric under the swap: a pixel is turned ON when the leading
    /// edge arrives and OFF when the trailing edge leaves, and arrival precedes departure.
    #[test]
    fn a_pixel_turns_on_when_the_bar_arrives_and_off_when_it_leaves() {
        let b = MovingBar { bar_width: 4, ticks: 400, ..MovingBar::default() };
        let pixels = (b.width * b.height) as usize;
        for &v in &[0.25f64, 1.0, 4.0] {
            for dir in Direction::all() {
                let ev = b.traverse(dir, v).unwrap();
                let mut on = vec![None; pixels];
                let mut off = vec![None; pixels];
                for e in &ev {
                    let slot = if e.polarity == Polarity::On { &mut on } else { &mut off };
                    assert!(slot[e.address as usize].is_none(), "pixel reported twice");
                    slot[e.address as usize] = Some(e.t);
                }
                for a in 0..pixels {
                    let (i, o) = (on[a].expect("every pixel turns on"), off[a].expect("and off"));
                    assert!(i < o, "{dir:?} at {v}: pixel {a} went off at {o} before on at {i}");
                }
                // The first tick that carries anything carries only on-events: nothing has left
                // the frame yet when the leading edge first enters it.
                let t0 = ev.first().unwrap().t;
                assert!(
                    ev.iter().filter(|e| e.t == t0).all(|e| e.polarity == Polarity::On),
                    "{dir:?} at {v}: the first tick already carried an off-event"
                );
            }
        }
    }

    /// The event-stream key keeps the polarity bit, and this is what makes that checkable. Two
    /// streams that agree on every `(t, address)` and disagree on one sign are different inputs.
    #[test]
    fn two_streams_that_differ_only_in_a_sign_have_different_keys() {
        use super::Keyed;
        let base = super::EventSample {
            events: vec![
                super::Event { t: 3, address: 7, polarity: Polarity::On },
                super::Event { t: 9, address: 2, polarity: Polarity::Off },
            ],
            label: 0,
        };
        let mut flipped = base.clone();
        flipped.events[1].polarity = Polarity::On;
        assert_ne!(
            base.key(),
            flipped.key(),
            "a sign flip left the input key untouched; dedup would reject one as a duplicate"
        );
        // The label is NOT in the key, which is the other half of the contract.
        let relabelled = super::EventSample { label: 1, ..base.clone() };
        assert_eq!(base.key(), relabelled.key());
    }

    /// A window is not bounded above, and the generator used to walk every tick of it whether or
    /// not the bar was still in the frame: eight samples of a legal `ticks = 1e9` config took 20
    /// seconds, and `1e12` never finished. The events are the same either way — the bar is gone —
    /// so the loop stops when it is gone.
    #[test]
    fn a_window_far_longer_than_the_traverse_costs_nothing_to_generate() {
        let b = MovingBar {
            ticks: 1_000_000_000_000,
            per_class_train: 1,
            per_class_test: 1,
            ..MovingBar::default()
        };
        let d = b.generate().unwrap();
        assert_eq!(d.ticks, 1_000_000_000_000);
        for s in d.train.iter().chain(d.test.iter()) {
            assert_eq!(s.events.len(), b.events_per_traverse() as usize);
        }
    }

    /// The premise the early exit rests on, asserted rather than assumed: from `last_active_tick`
    /// onwards the bar covers nothing, in every direction and at every speed, so no on-event and
    /// no off-event can be lost by stopping there. (That no event IS lost is the exact-count test
    /// above, which runs the same directions and speeds.)
    #[test]
    fn nothing_is_left_to_emit_after_the_last_active_tick() {
        let b = MovingBar { bar_width: 4, ticks: 4_000, ..MovingBar::default() };
        for &v in &[0.25f64, 0.5, 1.0, 2.0, 4.0] {
            for dir in Direction::all() {
                for delay in [0u64, 1, 37] {
                    let end = b.last_active_tick(dir, delay, v);
                    assert!(end <= b.ticks);
                    for t in (end - 1)..(end + 64).min(b.ticks) {
                        assert_eq!(
                            b.occupancy(dir, delay, v, t),
                            (0, 0),
                            "{dir:?} at {v} px/tick, delay {delay}: tick {t} is still covered, \
                             but the walk stops at {end}"
                        );
                    }
                }
            }
        }
    }

    /// The field doc says `[0, 1)` and the `# Errors` section repeats it; `in_range` is inclusive
    /// at the top and used to accept 1.0 — a sensor that emits every pixel on every tick, which
    /// carries no stimulus at all.
    #[test]
    fn a_saturated_noise_probability_is_refused_because_the_range_is_half_open() {
        let saturated = MovingBar { noise_prob_per_pixel_per_tick: 1.0, ..MovingBar::default() };
        assert!(matches!(
            saturated.generate(),
            Err(TaskError::OutOfRange { what: "noise_prob_per_pixel_per_tick", .. })
        ));
        // Just inside the range still generates, so the check is a boundary and not a ban.
        let loud = MovingBar {
            noise_prob_per_pixel_per_tick: 0.999,
            per_class_train: 1,
            per_class_test: 1,
            ..MovingBar::default()
        };
        assert!(loud.generate().is_ok());
    }

    #[test]
    fn a_generated_bar_sample_carries_the_exact_traverse_count_and_polarity_split() {
        let b = MovingBar::default();
        let d = b.generate().unwrap();
        let want = b.events_per_traverse() as usize;
        let pixels = (b.width * b.height) as usize;
        for s in d.test.iter().chain(d.train.iter()) {
            assert_eq!(s.events.len(), want, "speed or delay jitter changed the event count");
            assert_eq!(polarity_counts(s), (pixels, pixels));
            assert!(s.events.windows(2).all(|w| w[0] <= w[1]), "events were not sorted");
        }
    }

    /// Noise has an exactly computable expectation too: `w * h * ticks * p` extra events.
    #[test]
    fn background_noise_adds_the_number_of_events_it_promises() {
        let p = 0.01;
        let b = MovingBar {
            noise_prob_per_pixel_per_tick: p,
            per_class_train: 4,
            per_class_test: 40,
            ..MovingBar::default()
        };
        let d = b.generate().unwrap();
        let pixels = f64::from(b.width * b.height);
        let want = b.events_per_traverse() as f64 + pixels * b.ticks as f64 * p;
        let n = d.test.len() as f64;
        let mean = d.test.iter().map(|s| s.events.len() as f64).sum::<f64>() / n;
        // Binomial variance of the noise count, divided by the sample size.
        let se = (pixels * b.ticks as f64 * p * (1.0 - p) / n).sqrt();
        assert!((mean - want).abs() < 4.0 * se, "mean {mean} vs expected {want}, 4 SE = {}", 4.0 * se);
    }

    #[test]
    fn a_bar_thinner_than_its_per_tick_advance_is_refused_rather_than_skipping_pixels() {
        let b = MovingBar { bar_width: 1, speed_px_per_tick: 3.0, ..MovingBar::default() };
        assert!(matches!(
            b.generate(),
            Err(TaskError::OutOfRange { what: "speed_px_per_tick", .. })
        ));
        assert!(matches!(
            b.traverse(Direction::Right, 3.0),
            Err(TaskError::OutOfRange { what: "speed", .. })
                | Err(TaskError::OutOfRange { what: "speed_px_per_tick", .. })
        ));
    }

    #[test]
    fn a_window_too_short_for_the_traverse_is_refused() {
        let b = MovingBar { ticks: 5, ..MovingBar::default() };
        assert!(matches!(b.generate(), Err(TaskError::OutOfRange { what: "ticks", .. })));
    }

    /// Direction is in the ORDER, not in the totals — the same point temporal XOR makes, in 2-D.
    #[test]
    fn opposite_directions_are_indistinguishable_by_count_or_polarity() {
        let b = MovingBar { bar_width: 4, ticks: 400, ..MovingBar::default() };
        let r = b.traverse(Direction::Right, 1.0).unwrap();
        let l = b.traverse(Direction::Left, 1.0).unwrap();
        assert_eq!(r.len(), l.len());
        let on = |v: &[super::Event]| v.iter().filter(|e| e.polarity == Polarity::On).count();
        assert_eq!(on(&r), on(&l));
        assert_ne!(r, l, "the two directions produced identical streams");
    }

    // -- spoken digits -------------------------------------------------------------------------

    /// The expected spike count is a sum of Bernoulli probabilities, so it is exact. Compare the
    /// empirical mean against it at four standard errors of the Poisson-binomial.
    #[test]
    fn the_spoken_digit_spike_count_matches_its_analytic_mean() {
        let t = SpokenDigits { per_class_train: 4, per_class_test: 120, ..SpokenDigits::default() };
        let d = t.generate().unwrap();
        for c in 0..t.n_classes {
            let want = t.expected_spikes_per_sample(c).unwrap();
            let var = t.spike_count_variance(c).unwrap();
            let samples: Vec<f64> =
                d.test.iter().filter(|s| s.label == c).map(|s| s.train.len() as f64).collect();
            let n = samples.len() as f64;
            assert!(n > 50.0, "class {c} had only {n} test samples");
            let mean = samples.iter().sum::<f64>() / n;
            let se = (var / n).sqrt();
            assert!(
                (mean - want).abs() < 4.0 * se,
                "class {c}: mean {mean} vs analytic {want}, 4 SE = {}",
                4.0 * se
            );
            // The Poisson-binomial variance is strictly below the mean, because a tick carries at
            // most one spike. If this ever inverts, `p_at` has left [0, 1].
            assert!(var < want, "class {c}: variance {var} was not below the mean {want}");
        }
    }

    /// The envelope is symmetric about its peak, so the expected spike centroid of a channel IS
    /// the peak tick. That makes the sweep slope recoverable in closed form.
    #[test]
    fn the_spike_centroid_of_each_channel_lands_on_its_analytic_peak() {
        let t = SpokenDigits { per_class_train: 4, per_class_test: 200, ..SpokenDigits::default() };
        let d = t.generate().unwrap();
        for c in [0u32, 2, 4] {
            for k in [0u32, 7, 15] {
                let mut sum = 0.0f64;
                let mut n = 0.0f64;
                for s in d.test.iter().filter(|s| s.label == c) {
                    for sp in s.train.spikes().iter().filter(|sp| sp.source == k) {
                        sum += sp.t as f64;
                        n += 1.0;
                    }
                }
                assert!(n > 200.0, "class {c} channel {k} produced only {n} spikes");
                let centroid = sum / n;
                let want = t.peak_tick(c, k) as f64;
                // Standard error of the centroid: the spike times have spread sigma, so the mean
                // of n of them has spread sigma / sqrt(n). Four of those.
                let se = t.sigma_ticks / n.sqrt();
                assert!(
                    (centroid - want).abs() < 4.0 * se,
                    "class {c} channel {k}: centroid {centroid} vs peak {want}, 4 SE = {}",
                    4.0 * se
                );
            }
        }
    }

    /// The slope recovered FROM THE SPIKES, against the integer grid the class index declares —
    /// not against `peak_tick`, which is the function under test. A model of this task has to find
    /// this number, so the generator has to have put it there.
    #[test]
    fn the_sweep_slope_is_recoverable_from_the_spikes_alone() {
        let t = SpokenDigits { per_class_train: 4, per_class_test: 200, ..SpokenDigits::default() };
        let d = t.generate().unwrap();
        let centroid = |class: u32, ch: u32| {
            let mut sum = 0.0f64;
            let mut n = 0.0f64;
            for s in d.test.iter().filter(|s| s.label == class) {
                for sp in s.train.spikes().iter().filter(|sp| sp.source == ch) {
                    sum += sp.t as f64;
                    n += 1.0;
                }
            }
            assert!(n > 200.0, "class {class} channel {ch} produced only {n} spikes");
            (sum / n, n)
        };
        let last = t.n_channels - 1;
        for c in 0..t.n_classes {
            let (c0, n0) = centroid(c, 0);
            let (c1, n1) = centroid(c, last);
            let measured = (c1 - c0) / f64::from(last);
            let declared = t.slope_step_ticks as f64
                * (f64::from(c) - f64::from(t.n_classes - 1).div_euclid(2.0));
            let se = t.sigma_ticks * (1.0 / n0 + 1.0 / n1).sqrt() / f64::from(last);
            assert!(
                (measured - declared).abs() < 4.0 * se + 1e-9,
                "class {c}: slope measured {measured} ticks/channel vs declared {declared}, \
                 4 SE = {}",
                4.0 * se
            );
        }
    }

    #[test]
    fn a_jittered_spoken_digit_task_has_no_analytic_count_rather_than_a_stale_one() {
        let t = SpokenDigits { time_jitter_ticks: 5, ..SpokenDigits::default() };
        assert!(t.expected_spikes_per_sample(0).is_none());
        assert!(t.spike_count_variance(0).is_none());
        // ... and it still generates.
        assert!(t.generate().is_ok());
    }

    #[test]
    fn a_sweep_that_would_run_off_the_window_is_refused() {
        let t = SpokenDigits { slope_step_ticks: 40, ..SpokenDigits::default() };
        assert!(matches!(t.generate(), Err(TaskError::OutOfRange { what: "ticks", .. })));
    }

    #[test]
    fn the_sweep_slopes_are_the_integer_grid_the_doc_describes() {
        let t = SpokenDigits::default();
        let slopes: Vec<i64> = (0..t.n_classes).map(|c| t.slope(c)).collect();
        assert_eq!(slopes, vec![-6, -3, 0, 3, 6]);
    }

    // -- the catalogue -------------------------------------------------------------------------

    /// A table that can drift away from the code is worse than no table: it still looks
    /// authoritative. Every row is checked against the generator it names.
    #[test]
    fn every_catalogue_row_matches_the_task_it_names() {
        let names: Vec<(&str, u32, f64)> = vec![
            {
                let d = TemporalXor::default().generate().unwrap();
                (d.name, d.n_classes, d.chance)
            },
            {
                let d = Coincidence::default().generate().unwrap();
                (d.name, d.n_classes, d.chance)
            },
            {
                let d = DelayedMatch::default().generate().unwrap();
                (d.name, d.n_classes, d.chance)
            },
            {
                let d = RateDiscrimination::default().generate().unwrap();
                (d.name, d.n_classes, d.chance)
            },
            {
                let d = LatencyPatterns::default().generate().unwrap();
                (d.name, d.n_classes, d.chance)
            },
            {
                let d = MovingBar::default().generate().unwrap();
                (d.name, d.n_classes, d.chance)
            },
            {
                let d = SpokenDigits::default().generate().unwrap();
                (d.name, d.n_classes, d.chance)
            },
        ];
        assert_eq!(names.len(), CATALOGUE.len());
        for (card, (name, classes, chance)) in CATALOGUE.iter().zip(names) {
            assert_eq!(card.name, name, "catalogue name drifted");
            assert_eq!(card.classes, classes, "{name}: catalogue class count drifted");
            assert!((card.chance - chance).abs() < 1e-15, "{name}: catalogue chance drifted");
            // The two prose columns are the reason this table is worth printing beside a score,
            // and nothing checked that they had been written at all.
            assert!(card.isolates.len() > 20, "{name}: `isolates` says nothing");
            assert!(card.stands_in_for.len() > 10, "{name}: `stands_in_for` says nothing");
        }
        // And each row names its own mechanism rather than repeating its neighbour's.
        let mut isolates: Vec<&str> = CATALOGUE.iter().map(|c| c.isolates).collect();
        isolates.sort_unstable();
        isolates.dedup();
        assert_eq!(isolates.len(), CATALOGUE.len(), "two catalogue rows isolate the same thing");
    }

    #[test]
    fn a_dataset_reports_its_duration_in_seconds() {
        let d = DelayedMatch::default().generate().unwrap();
        // 30 + 100 + 30 ticks of 1 ms.
        assert!((d.duration_seconds() - 0.16).abs() < 1e-12, "{}", d.duration_seconds());
    }

    #[test]
    fn an_empty_split_has_no_balance_rather_than_a_vector_of_zeros() {
        let mut d = TemporalXor::default().generate().unwrap();
        d.test.clear();
        assert!(d.balance(Split::Test).is_empty());
    }

    /// `balance` refuses to report zeros for a split nobody measured, and `finish` then folded that
    /// empty vector with `max` from a `0.0` seed — storing a majority baseline of **zero**, below
    /// chance, in the field the module doc says a score must be read against. `None` instead.
    /// A label outside `0..n_classes` is dropped by `counts` — it has no bin — and the module says
    /// so rather than leaving it to be discovered. `balance` divides by the split length, so the
    /// loss surfaces as shares that do not sum to one instead of being normalised out of sight.
    #[test]
    fn a_label_outside_the_class_range_is_dropped_visibly_not_silently() {
        let mut d = TemporalXor::default().generate().unwrap();
        let n = d.test.len();
        assert_eq!(d.counts(Split::Test).iter().sum::<usize>(), n);
        d.test[0].label = 7;
        let counts = d.counts(Split::Test);
        assert_eq!(counts.len(), 2, "the vector is indexed by class and has no overflow bin");
        assert_eq!(counts.iter().sum::<usize>(), n - 1, "the stray sample was counted somewhere");
        let total: f64 = d.balance(Split::Test).iter().sum();
        assert!(total < 1.0, "the shares summed to {total}, hiding the dropped sample");
        assert!((total - (n - 1) as f64 / n as f64).abs() < 1e-12);
    }

    #[test]
    fn a_dataset_with_no_test_split_has_no_majority_baseline_rather_than_zero() {
        let t = DelayedMatch { per_class_train: 8, per_class_test: 0, ..DelayedMatch::default() };
        let d = t.generate().unwrap();
        assert_eq!(d.train.len(), 16, "the train split still filled");
        assert!(d.test.is_empty());
        assert!(d.balance(Split::Test).is_empty());
        assert_eq!(
            d.majority_baseline, None,
            "a split of no samples reported a majority class anyway"
        );
        // And the ordinary case still carries the number, so this is not a silent opt-out.
        assert_eq!(DelayedMatch::default().generate().unwrap().majority_baseline, Some(0.5));
    }

    /// `Dataset::train` is documented as shuffled, and a no-op `shuffle` passed every other test in
    /// this file: `collect_split` appends class 0, then class 1, so an unshuffled split is
    /// perfectly class-ordered — the silent confound that breaks an online learner while moving no
    /// count, no balance and no baseline.
    #[test]
    fn the_training_split_is_shuffled_rather_than_class_ordered() {
        let d = LatencyPatterns::default().generate().unwrap();
        let ordered = usize::try_from(d.n_classes).unwrap() - 1;
        let flips = d.train.windows(2).filter(|w| w[0].label != w[1].label).count();
        // A class-ordered stream has exactly `n_classes - 1` label changes; a random permutation of
        // five balanced classes has about `(n - 1) * 4 / 5 = 399` of them.
        assert!(
            flips > 300,
            "{flips} label changes in {} training samples, and a class-ordered split has {ordered}",
            d.train.len()
        );
        // The other way to see it: every class appears in the first tenth of the stream.
        let head = &d.train[..d.train.len() / 10];
        for c in 0..d.n_classes {
            assert!(head.iter().any(|s| s.label == c), "class {c} is absent from the first tenth");
        }
        // The test split is shuffled too.
        let flips = d.test.windows(2).filter(|w| w[0].label != w[1].label).count();
        assert!(flips > 150, "{flips} label changes in the test split");
    }

    /// And `shuffle` itself, against the three things it has to be: a permutation, not the
    /// identity, and the same one for the same seed.
    #[test]
    fn shuffle_is_a_deterministic_permutation_that_actually_moves_things() {
        let ident: Vec<u32> = (0..64).collect();
        let mut a = ident.clone();
        super::shuffle(&mut a, &mut Rng::new(7));
        let mut sorted = a.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, ident, "shuffle lost or duplicated an element");
        assert_ne!(a, ident, "shuffle left the order exactly as it found it");
        let fixed = a.iter().enumerate().filter(|&(i, &v)| i as u32 == v).count();
        assert!(fixed < 8, "{fixed} of 64 entries stayed put, which is not a shuffle");

        let mut b = ident.clone();
        super::shuffle(&mut b, &mut Rng::new(7));
        assert_eq!(a, b, "the same seed gave a different permutation");
        let mut c = ident.clone();
        super::shuffle(&mut c, &mut Rng::new(8));
        assert_ne!(a, c, "two seeds gave the same permutation");

        // A slice of one draws nothing, which is why `below(0)` is unreachable from here.
        let mut one = [5u32];
        super::shuffle(&mut one, &mut Rng::new(7));
        assert_eq!(one, [5]);
    }

    /// The condition cycling is on ACCEPTED samples, so the per-channel early/late counts of the
    /// two classes are exactly equal — the claim the task doc makes about per-channel readouts.
    ///
    /// Cycling on the draw index instead made them differ by however many duplicates rejection ate:
    /// measured on this configuration, channel 0 of the test split was early in 31 samples of class
    /// 0 and 34 of class 1, which is a 2-tick difference in the class means of a quantity the doc
    /// says carries nothing.
    #[test]
    fn the_early_late_counts_per_channel_are_exactly_equal_in_both_classes() {
        let t = TemporalXor::default();
        // Premise: a jittered early volley can never be mistaken for a late one.
        assert!(
            2 * t.jitter_ticks < t.late_tick - t.early_tick,
            "the jitter overlaps the two volley positions, so 'early' is not recoverable"
        );
        let mid = (t.early_tick + t.late_tick) / 2;
        let d = t.generate().unwrap();
        for split in [Split::Train, Split::Test] {
            let n = d.split(split).len();
            for ch in 0..2u32 {
                let mut early = [0usize; 2];
                for s in d.split(split) {
                    let ft = Dataset::first_spike(s, ch).expect("both channels fire");
                    // Premise again, per sample: the first spike sits on one volley or the other.
                    assert!(
                        ft.abs_diff(t.early_tick) <= t.jitter_ticks
                            || ft.abs_diff(t.late_tick) <= t.jitter_ticks,
                        "a first spike at {ft} is on neither volley"
                    );
                    early[s.label as usize] += usize::from(ft < mid);
                }
                assert_eq!(
                    early[0], early[1],
                    "{split:?} channel {ch}: class 0 was early {} times, class 1 {} times",
                    early[0], early[1]
                );
                assert_eq!(4 * early[0], n, "{split:?} channel {ch}: {early:?} of {n} samples");
            }
        }
    }

    /// `peak_tick` had no independent check at all: the centroid test used it as the expected
    /// value and the generator computed the envelope from it, so both sides moved together —
    /// changing `n_channels / 2` to `n_channels / 4` passed everything. Here the expectation is
    /// written from the STRUCT DOC's formula, and the measurement comes from the spikes.
    #[test]
    fn the_channel_centroids_match_the_closed_form_in_the_doc_not_the_function() {
        let t = SpokenDigits { per_class_train: 4, per_class_test: 200, ..SpokenDigits::default() };
        let d = t.generate().unwrap();
        // "channel `k` has a Gaussian firing-rate envelope peaking at tick
        //  `centre + slope(c) * (k - mid)`", with `centre = ticks / 2`, `mid = n_channels / 2`,
        //  and `slope(c) = slope_step_ticks * (c - (n_classes - 1) / 2)` — the integer grid
        //  `the_sweep_slopes_are_the_integer_grid_the_doc_describes` pins to [-6, -3, 0, 3, 6].
        let centre = (t.ticks / 2) as f64;
        let mid = f64::from(t.n_channels / 2);
        for (c, slope) in [-6.0f64, -3.0, 0.0, 3.0, 6.0].into_iter().enumerate() {
            for k in [0u32, 4, 15] {
                let want = centre + slope * (f64::from(k) - mid);
                let mut sum = 0.0f64;
                let mut n = 0.0f64;
                for s in d.test.iter().filter(|s| s.label == c as u32) {
                    for sp in s.train.spikes().iter().filter(|sp| sp.source == k) {
                        sum += sp.t as f64;
                        n += 1.0;
                    }
                }
                assert!(n > 200.0, "class {c} channel {k} produced only {n} spikes");
                let centroid = sum / n;
                let se = t.sigma_ticks / n.sqrt();
                assert!(
                    (centroid - want).abs() < 4.0 * se,
                    "class {c} channel {k}: centroid {centroid} vs the doc's {want}, 4 SE = {}",
                    4.0 * se
                );
            }
        }
    }

    /// The window floor the `SpokenDigits::default` doc derives, run rather than asserted in prose.
    #[test]
    fn the_window_floor_is_the_one_the_doc_derives() {
        let fits = SpokenDigits {
            ticks: 145,
            per_class_train: 1,
            per_class_test: 1,
            ..SpokenDigits::default()
        };
        assert!(fits.generate().is_ok(), "145 ticks should hold the steepest sweep plus 4 sigma");
        let clipped = SpokenDigits { ticks: 144, ..fits };
        assert!(
            matches!(clipped.generate(), Err(TaskError::OutOfRange { what: "ticks", .. })),
            "144 ticks clips the extreme envelope and must be refused"
        );
    }

    /// The cue-window arithmetic the `DelayedMatch::default` doc argues from, with the numbers.
    #[test]
    fn a_cue_window_too_short_for_its_spikes_and_jitter_is_refused() {
        // At `cue_ticks = 20` the spacing is `20 / 4 = 5`, so four spikes span `3 * 5 = 15`, and
        // `15 + 2 * 4` jitter ticks is 23, which does not fit in 20.
        let short = DelayedMatch { cue_ticks: 20, ..DelayedMatch::default() };
        match short.generate() {
            Err(TaskError::OutOfRange { what: "cue_ticks", value, low, .. }) => {
                assert!((value - 20.0).abs() < f64::EPSILON);
                assert!((low - 24.0).abs() < f64::EPSILON, "the refusal asks for {low}, not 24");
            }
            other => panic!("a 20-tick cue window produced {other:?}"),
        }
        // At 30 the spacing is 7, the span is 21, and 21 + 8 fits.
        assert!(DelayedMatch::default().generate().is_ok());
    }

    /// Every dataset carries its own caveat. A blank one is a benchmark that will be over-quoted.
    ///
    /// Every task's caveat, not three of the seven. The old test named `TemporalXor`, `MovingBar`
    /// and `SpokenDigits` only, while the disclosure said "every `Dataset`".
    #[test]
    fn every_dataset_says_what_it_does_not_capture() {
        let checks: Vec<(&str, &str, &str)> = vec![
            {
                let d = TemporalXor::default().generate().unwrap();
                (d.name, d.stands_in_for, d.not_captured)
            },
            {
                let d = Coincidence::default().generate().unwrap();
                (d.name, d.stands_in_for, d.not_captured)
            },
            {
                let d = DelayedMatch::default().generate().unwrap();
                (d.name, d.stands_in_for, d.not_captured)
            },
            {
                let d = RateDiscrimination::default().generate().unwrap();
                (d.name, d.stands_in_for, d.not_captured)
            },
            {
                let d = LatencyPatterns::default().generate().unwrap();
                (d.name, d.stands_in_for, d.not_captured)
            },
            {
                let d = MovingBar::default().generate().unwrap();
                (d.name, d.stands_in_for, d.not_captured)
            },
            {
                let d = SpokenDigits::default().generate().unwrap();
                (d.name, d.stands_in_for, d.not_captured)
            },
        ];
        assert_eq!(checks.len(), CATALOGUE.len(), "a task has no caveat row here");
        for (name, stands, not) in &checks {
            assert!(stands.len() > 20, "{name}: stands_in_for is too short to mean anything");
            assert!(not.len() > 20, "{name}: not_captured is too short to mean anything");
        }
        // And each one is its own sentence rather than a copy of its neighbour's.
        let mut caveats: Vec<&str> = checks.iter().map(|c| c.2).collect();
        caveats.sort_unstable();
        caveats.dedup();
        assert_eq!(caveats.len(), checks.len(), "two tasks share a not_captured string");
    }

    // -- the u32 boundary every draw in this module crosses ------------------------------------

    /// `Rng::below` takes a `u32` and every span here is a `u64` of ticks, so an `as u32` between
    /// them draws from a range nobody asked for — and no assertion downstream notices, because a
    /// narrower range sits inside the wider one. Measured before the fix: this config drew its
    /// "up to `5_000_000_000`" gaps from `0..705_032_696`, and the largest gap in the dataset
    /// was `688_949_765`.
    #[test]
    fn a_gap_range_too_wide_for_a_uniform_draw_is_refused_rather_than_narrowed() {
        let wide = Coincidence {
            threshold_ticks: 8,
            max_gap_ticks: 5_000_000_000,
            first_tick: 20,
            jitter_ticks: 15,
            ticks: 6_000_000_000,
            per_class_train: 4,
            per_class_test: 4,
            ..Coincidence::default()
        };
        assert!(
            matches!(wide.generate(), Err(TaskError::OutOfRange { what: "max_gap_ticks", .. })),
            "a 5e9 gap range was narrowed instead of refused"
        );
        // A span landing on a multiple of 2^32 narrowed to ZERO, and `below(0)` PANICS — out of a
        // constructor whose `# Errors` section promises a `TaskError` and which carries no
        // `# Panics` section.
        let exact = Coincidence { max_gap_ticks: 8 + 4_294_967_296, ..wide };
        assert!(
            matches!(exact.generate(), Err(TaskError::OutOfRange { .. })),
            "a span of exactly 2^32 must not reach below(0)"
        );
        let coincident = Coincidence {
            threshold_ticks: 4_294_967_295,
            max_gap_ticks: 8_000_000_000,
            ticks: 20_000_000_000,
            ..wide
        };
        assert!(matches!(
            coincident.generate(),
            Err(TaskError::OutOfRange { what: "threshold_ticks", .. })
        ));

        // The ceiling is `MAX_DRAW_SPAN`, not `u32::MAX`, and this is the config that says why:
        // `Rng::below(4_000_000_000)` accepts two values in 2^32 and takes 14.1 SECONDS per draw.
        // A dataset built on it does not come back, so it is refused rather than started.
        let slow = Coincidence { max_gap_ticks: 4_000_000_000, ..wide };
        assert!(matches!(
            slow.generate(),
            Err(TaskError::OutOfRange { what: "max_gap_ticks", high, .. })
                if (high - super::MAX_DRAW_SPAN as f64).abs() < f64::EPSILON
        ));

        // The positive control, without which this is a test that the module refuses everything:
        // a range that DOES fit still generates, and the gaps still span what was asked for.
        let ok = Coincidence {
            max_gap_ticks: 2_000_000_000,
            ticks: 5_000_000_000,
            per_class_train: 16,
            per_class_test: 16,
            ..wide
        };
        let d = ok.generate().unwrap();
        let largest = d
            .train
            .iter()
            .chain(d.test.iter())
            .filter(|s| s.label == 0)
            .map(|s| s.train.spikes()[0].t.abs_diff(s.train.spikes()[1].t))
            .max()
            .unwrap();
        assert!(
            largest > 1_500_000_000,
            "largest gap {largest} of a requested 2_000_000_000 — the range was narrowed"
        );
        // The two ceilings are what they claim to be, and a jitter spans twice its half-width.
        assert_eq!(super::MAX_DRAW_SPAN, 2_147_483_647);
        assert_eq!(2 * super::MAX_JITTER_HALF + 1, super::MAX_DRAW_SPAN);
    }

    /// The same boundary on every jitter in the module. Before the fix a 3e9 half-width on
    /// `TemporalXor` drew from `(2 * 3e9 + 1) as u32 = 1_705_032_705`, and the volley starts
    /// spanned `64_380_331 ..= 5_669_951_247` instead of the requested `0 ..= 1e10`.
    #[test]
    fn a_jitter_too_wide_for_a_uniform_draw_is_refused_in_every_task() {
        let x = TemporalXor {
            early_tick: 3_000_000_000,
            late_tick: 7_000_000_000,
            jitter_ticks: 3_000_000_000,
            ticks: 20_000_000_000,
            per_class_train: 4,
            per_class_test: 4,
            ..TemporalXor::default()
        };
        assert!(matches!(
            x.generate(),
            Err(TaskError::OutOfRange { what: "jitter_ticks", .. })
        ));

        let c = Coincidence {
            jitter_ticks: 3_000_000_000,
            first_tick: 4_000_000_000,
            ticks: 9_000_000_000,
            ..Coincidence::default()
        };
        assert!(matches!(
            c.generate(),
            Err(TaskError::OutOfRange { what: "jitter_ticks", .. })
        ));

        let m = DelayedMatch {
            jitter_ticks: 3_000_000_000,
            cue_ticks: 8_000_000_000,
            ..DelayedMatch::default()
        };
        assert!(matches!(
            m.generate(),
            Err(TaskError::OutOfRange { what: "jitter_ticks", .. })
        ));

        let l = LatencyPatterns { jitter_ticks: 3_000_000_000, ticks: 9_000_000_000, ..LatencyPatterns::default() };
        assert!(matches!(
            l.generate(),
            Err(TaskError::OutOfRange { what: "jitter_ticks", .. })
        ));

        let s = SpokenDigits {
            time_jitter_ticks: 3_000_000_000,
            ticks: 9_000_000_000,
            ..SpokenDigits::default()
        };
        assert!(matches!(
            s.generate(),
            Err(TaskError::OutOfRange { what: "time_jitter_ticks", .. })
        ));

        let b = MovingBar { max_start_delay: 5_000_000_000, ticks: 6_000_000_000, ..MovingBar::default() };
        assert!(matches!(
            b.generate(),
            Err(TaskError::OutOfRange { what: "max_start_delay", .. })
        ));

        // The latency window itself is a draw span too.
        let t = LatencyPatterns { ticks: 9_000_000_000, ..LatencyPatterns::default() };
        assert!(matches!(t.templates(), Err(TaskError::OutOfRange { what: "ticks", .. })));

        // And the shuffle index: a split longer than `u32::MAX` would shuffle only its first four
        // billion entries. Refused before a byte is allocated.
        let big = TemporalXor { per_class_train: 3_000_000_000, per_class_test: 0, ..TemporalXor::default() };
        assert!(matches!(
            big.generate(),
            Err(TaskError::OutOfRange { what: "samples per class", .. })
        ));
    }

    /// Two public items nothing exercised at all: the `tau_m` rule of thumb, and the error
    /// messages — which are the whole of what a user sees when a configuration is refused, and
    /// which the `TaskError` doc promises will "name the offending quantity".
    #[test]
    fn the_tau_m_hint_is_in_seconds_and_every_refusal_names_its_quantity() {
        // `threshold_ticks * dt`, in SECONDS: 8 ticks of 1 ms.
        let c = Coincidence::default();
        assert!((c.suggested_tau_m() - 0.008).abs() < 1e-15, "{}", c.suggested_tau_m());
        let slower = Coincidence { dt: 2e-3, threshold_ticks: 10, ..c };
        assert!((slower.suggested_tau_m() - 0.02).abs() < 1e-15, "{}", slower.suggested_tau_m());

        let msgs = [
            TaskError::NotFinite { what: "rate_hi", value: f64::NAN }.to_string(),
            TaskError::Empty { what: "cue_spikes" }.to_string(),
            TaskError::OutOfRange { what: "ticks", value: 5.0, low: 30.0, high: 40.0 }.to_string(),
            TaskError::Exhausted { wanted: 192, distinct: 4, draws: 13312 }.to_string(),
        ];
        for (m, needle) in msgs.iter().zip(["rate_hi", "cue_spikes", "ticks", "192"]) {
            assert!(m.contains(needle), "{m:?} does not name {needle}");
        }
        assert!(msgs[0].contains("NaN"), "{:?}", msgs[0]);
        assert!(msgs[2].contains("30") && msgs[2].contains("40"), "{:?}", msgs[2]);
        assert!(msgs[3].contains("13312"), "{:?}", msgs[3]);
        // It is an `Error`, which is what lets `?` work in every example in this crate.
        let boxed: Box<dyn std::error::Error> = Box::new(TaskError::Empty { what: "n_classes" });
        assert!(boxed.to_string().contains("n_classes"));
    }

    /// The speed bounds the window floor is derived from, which nothing read back.
    #[test]
    fn the_speed_bounds_and_the_window_floor_are_the_ones_the_config_implies() {
        let b = MovingBar::default();
        assert!((b.min_speed() - 0.8).abs() < 1e-12, "{}", b.min_speed());
        assert!((b.max_speed() - 1.2).abs() < 1e-12, "{}", b.max_speed());
        // The slowest traverse of a 16-wide grid behind a 3-pixel bar is `ceil(19 / 0.8) = 24`
        // ticks, after a start delay of up to 4, plus the two ticks the edges need.
        assert_eq!(b.min_ticks(), Some(30));
        assert!(b.ticks >= b.min_ticks().unwrap(), "the default window is below its own floor");
        // A speed that cannot move has no window at all, rather than a very long one.
        let stuck = MovingBar { speed_px_per_tick: 0.0, ..b };
        assert_eq!(stuck.min_ticks(), None);
        // The floor rounds UP and never down, including through the floating point: at a 90%
        // jitter `1.0 * (1.0 - 0.9)` is 0.09999999999999998, so `19 / v` is 190.00000000000003 and
        // the traverse is charged 191 ticks rather than 190. A window one tick too long clips
        // nothing; one tick too short clips an event.
        let backwards = MovingBar { speed_jitter_frac: 0.9, ..b };
        assert!((backwards.min_speed() - 0.1).abs() < 1e-12);
        assert_eq!(backwards.min_ticks(), Some(4 + 191 + 2));
        let long = MovingBar { ticks: 197, ..backwards };
        assert!(long.generate().is_ok(), "a window at the floor must generate");
        let short = MovingBar { ticks: 196, ..backwards };
        assert!(matches!(short.generate(), Err(TaskError::OutOfRange { what: "ticks", .. })));
    }

    /// `poisson_optimal_accuracy` refuses a support above [`super::MAX_COUNT_SUPPORT`]; its exact
    /// twin had no guard at all, so `ticks = 1e9` asked for three `Vec<f64>` of 24 GB and aborted
    /// the process. Worse, the summation error grows with the support: at `ticks = 5e7` the
    /// function returned **1.00002636954903035**, which is not a probability, and printed it as a
    /// bound.
    #[test]
    fn an_unsummable_window_has_no_bound_rather_than_an_abort_or_a_number_above_one() {
        let over = RateDiscrimination {
            ticks: super::MAX_COUNT_SUPPORT + 1,
            ..RateDiscrimination::default()
        };
        assert!(
            over.optimal_accuracy().is_none(),
            "a window too long to sum exactly reported a bound anyway"
        );
        // The Poisson twin truncates on COUNTS, not on ticks, so it still answers here — the two
        // ceilings are different quantities and only one of them is an allocation.
        let p = over.poisson_optimal_accuracy().expect("the count support is still small");
        assert!((0.0..=1.0).contains(&p), "the Poisson bound {p} is not a probability");

        // Inside the ceiling the sum is still a probability, and still a bound above chance.
        let big = RateDiscrimination { ticks: 1_000_000, ..RateDiscrimination::default() };
        let a = big.optimal_accuracy().expect("a million ticks is inside the ceiling");
        assert!((0.5..=1.0).contains(&a), "the bound at a million ticks was {a}");
    }

    // -- (m) the survivors of the mutation audit ------------------------------------------------

    /// `nonzero` is the module's only guard against a count of zero, and it is called at
    /// nineteen places. Nothing asserted the refusal at any of them: the suite only ever built
    /// tasks with counts it had chosen to be sensible, so emptying `nonzero`'s body left every
    /// test passing.
    /// Zero is not a small task, it is a different one — a volley of no spikes, a grid of no
    /// pixels, a split of no samples — and each of those reaches different arithmetic downstream
    /// (a `spikes_per_channel` of 0 underflows `spikes_per_channel - 1`, a `per_class` of 0 skips
    /// the collection loop entirely and hands back an empty dataset). So the refusal is pinned
    /// here by the name it reports, at every call site, rather than by the fact that something
    /// went wrong somewhere.
    #[test]
    fn a_zero_count_is_refused_by_name_at_every_place_the_module_counts_something() {
        macro_rules! empty {
            ($what:literal, $e:expr) => {{
                let got = $e;
                assert!(
                    matches!(got, Err(TaskError::Empty { what: $what })),
                    concat!("a zero ", $what, " was accepted: {:?}"),
                    got
                );
            }};
        }
        let x = TemporalXor::default();
        empty!("spikes_per_channel", TemporalXor { spikes_per_channel: 0, ..x }.generate());
        empty!("burst_gap", TemporalXor { burst_gap: 0, ..x }.generate());
        // `collect_split`'s own two, reached through the generator that calls it.
        empty!("samples per class", TemporalXor { per_class_train: 0, per_class_test: 0, ..x }.generate());

        let m = DelayedMatch::default();
        empty!("cue_spikes", DelayedMatch { cue_spikes: 0, ..m }.generate());
        empty!("cue_ticks", DelayedMatch { cue_ticks: 0, ..m }.generate());

        let l = LatencyPatterns::default();
        empty!("n_classes", LatencyPatterns { n_classes: 0, ..l }.templates());
        empty!("n_inputs", LatencyPatterns { n_inputs: 0, ..l }.templates());

        let b = MovingBar::default();
        empty!("width", MovingBar { width: 0, ..b }.generate());
        empty!("height", MovingBar { height: 0, ..b }.generate());
        empty!("bar_width", MovingBar { bar_width: 0, ..b }.generate());
        empty!("ticks", MovingBar { ticks: 0, ..b }.generate());
        // `traverse` re-checks the grid itself, because it is public and takes no `Dataset`.
        empty!("width", MovingBar { width: 0, ..b }.traverse(Direction::Right, 1.0));
        empty!("height", MovingBar { height: 0, ..b }.traverse(Direction::Right, 1.0));
        empty!("bar_width", MovingBar { bar_width: 0, ..b }.traverse(Direction::Right, 1.0));

        empty!("ticks", RateDiscrimination { ticks: 0, ..RateDiscrimination::default() }.generate());

        let s = SpokenDigits::default();
        empty!("n_classes", SpokenDigits { n_classes: 0, ..s }.generate());
        empty!("n_channels", SpokenDigits { n_channels: 0, ..s }.generate());
        empty!("ticks", SpokenDigits { ticks: 0, ..s }.generate());

        // And `collect_split`'s class guard directly, which no generator can reach: every one of
        // them passes a class count that is a literal or already checked.
        let none = super::collect_split(0, 1, 1, 7, |c, _| super::Sample {
            train: Train::from_spikes(vec![Spike { t: 0, source: 0 }]),
            label: c,
        });
        empty!("n_classes", none.map(|_| ()));
    }

    /// The two ceilings are different numbers — `MAX_JITTER_HALF` is `(MAX_DRAW_SPAN - 1) / 2`,
    /// because a half-width of `h` spans `2h + 1` values — and every test that exercised them used
    /// a jitter of three billion, which is over *both*. A half-width between the two ceilings is
    /// therefore the only witness, and nothing generated one: `1_073_741_824` is refused by the
    /// real check and accepted by a check written against `MAX_DRAW_SPAN`, whose `Half` would then
    /// ask `Rng::below` for `2_147_483_649` values — past the point the `MAX_DRAW_SPAN` doc
    /// measures at 14.1 seconds a draw.
    #[test]
    fn a_jitter_half_width_is_measured_against_its_own_ceiling_not_the_draw_span() {
        // The arithmetic the ceiling is derived from, restated as the equality it is.
        assert_eq!(2 * super::MAX_JITTER_HALF + 1, super::MAX_DRAW_SPAN);
        assert_eq!(super::MAX_JITTER_HALF, 1_073_741_823);

        // At the ceiling: accepted, and the `Half` carries the value unchanged.
        assert_eq!(
            super::half_width("jitter_ticks", super::MAX_JITTER_HALF),
            Ok(super::Half(1_073_741_823))
        );
        // One past it: refused, and the refusal reports the JITTER ceiling as its upper bound
        // rather than the draw span, so the number a user is told to come under is the right one.
        assert!(
            matches!(
                super::half_width("jitter_ticks", super::MAX_JITTER_HALF + 1),
                Err(TaskError::OutOfRange { what: "jitter_ticks", high, .. })
                    if (high - super::MAX_JITTER_HALF as f64).abs() < f64::EPSILON
            ),
            "a half-width of {} was admitted; its draw spans {} values",
            super::MAX_JITTER_HALF + 1,
            2 * (super::MAX_JITTER_HALF + 1) + 1
        );
        // And a half-width equal to the DRAW span is a half-width of more than twice the ceiling.
        assert!(matches!(
            super::half_width("jitter_ticks", super::MAX_DRAW_SPAN),
            Err(TaskError::OutOfRange { what: "jitter_ticks", .. })
        ));
    }

    /// The `Keyed` doc says the label is excluded from a sample's key, and the module doc rests the
    /// whole disjointness argument on it: "the same input appearing under two labels is a
    /// contradiction in the task, not leakage". `EventSample` had that pinned; `Sample`, which is
    /// six of the seven tasks, did not. Every existing assertion reads a `Sample` key through
    /// `Keyed::key` on both sides of the comparison, so a key that smuggled the label in was being
    /// compared against itself and agreed.
    #[test]
    fn a_sample_key_is_its_input_alone_so_one_input_under_two_labels_is_one_key() {
        use super::Keyed;
        let spikes = vec![Spike { t: 4, source: 1 }, Spike { t: 9, source: 0 }];
        let a = super::Sample { train: Train::from_spikes(spikes.clone()), label: 0 };
        let b = super::Sample { train: Train::from_spikes(spikes), label: 3 };
        assert_eq!(
            a.key(),
            b.key(),
            "the label reached the key; one input under two labels would sit on both sides of a \
             split while `overlap` reported zero"
        );
        // The key is the `(t, source)` pairs in train order, and nothing else — stated as the
        // exact vector so a future field cannot be added to it unnoticed.
        assert_eq!(a.key(), vec![4, 1, 9, 0]);
        // And it still separates two inputs that differ only in which channel fired.
        let moved = super::Sample {
            train: Train::from_spikes(vec![Spike { t: 4, source: 0 }, Spike { t: 9, source: 0 }]),
            label: 0,
        };
        assert_ne!(a.key(), moved.key());
    }

    /// An event key packs `(address, polarity)` into one `u64` as `address << 1 | on`. The shift is
    /// load-bearing and nothing tested it: `two_streams_that_differ_only_in_a_sign_have_different_keys`
    /// flips a polarity while holding the address fixed, which an unshifted `address | on` also
    /// separates — for an even address. The pair that collides without the shift is an *even*
    /// address that is on against the *next odd* address that is off, and no fixture held one.
    #[test]
    fn an_event_key_keeps_the_address_and_the_polarity_in_separate_bits() {
        use super::Keyed;
        let even_on = super::EventSample {
            events: vec![super::Event { t: 1, address: 2, polarity: Polarity::On }],
            label: 0,
        };
        let odd_off = super::EventSample {
            events: vec![super::Event { t: 1, address: 3, polarity: Polarity::Off }],
            label: 0,
        };
        assert_ne!(
            even_on.key(),
            odd_off.key(),
            "pixel 2 turning on and pixel 3 turning off have the same key; one of the two streams \
             would be rejected as a duplicate of the other"
        );
        // The exact encoding, so the two fields cannot start overlapping again: `2 << 1 | 1` and
        // `3 << 1 | 0`.
        assert_eq!(even_on.key(), vec![1, 5]);
        assert_eq!(odd_off.key(), vec![1, 6]);
    }

    /// `channel_counts` is the instrument [`TemporalXor`]'s central claim is read through, and
    /// every assertion on it compares class 0 against class 1 — at a point where the two are
    /// exactly equal, on purpose. A counter that recorded *presence* instead of accumulating a
    /// total reports `[1, 1]` for both classes, and every one of those comparisons still holds.
    /// So this one asserts the absolute totals, which come from the configuration by
    /// multiplication: 64 test samples of a class, three spikes per channel, is 192 per channel.
    #[test]
    fn the_channel_counts_are_totals_over_the_split_not_a_record_of_which_channels_fired() {
        let x = TemporalXor::default();
        let d = x.generate().unwrap();
        let per_sample = u64::from(x.spikes_per_channel);
        let test_total = x.per_class_test as u64 * per_sample;
        let train_total = x.per_class_train as u64 * per_sample;
        assert_eq!(test_total, 192);
        assert_eq!(train_total, 384);
        for class in 0..2 {
            assert_eq!(
                d.channel_counts(Split::Test, class),
                vec![test_total; 2],
                "class {class} test totals are not {test_total} spikes per channel"
            );
            assert_eq!(d.channel_counts(Split::Train, class), vec![train_total; 2]);
        }
        // A class nobody generated totals zero on every channel rather than one, which is the
        // same arithmetic read at its empty end: nothing summed is 0, nothing PRESENT is 1.
        assert_eq!(d.channel_counts(Split::Test, 7), vec![0, 0]);
    }

    /// `polarity_counts` is documented to return `(on, off)` in that order, and the whole suite
    /// read it from one line — `assert_eq!(polarity_counts(s), (pixels, pixels))` on a full bar
    /// traverse, which emits exactly `width * height` of each. A pair whose two entries are equal
    /// cannot say which is which: vacuous-test mechanism #156 in the register, a balanced fixture
    /// cancelling the defect out. A stream with three on-events and one off-event is the smallest
    /// thing that can tell them apart.
    #[test]
    fn the_polarity_counts_are_on_first_then_off_on_a_stream_that_is_not_balanced() {
        let lopsided = super::EventSample {
            events: vec![
                super::Event { t: 1, address: 0, polarity: Polarity::On },
                super::Event { t: 2, address: 0, polarity: Polarity::On },
                super::Event { t: 3, address: 0, polarity: Polarity::On },
                super::Event { t: 4, address: 0, polarity: Polarity::Off },
            ],
            label: 0,
        };
        assert_eq!(polarity_counts(&lopsided), (3, 1), "the pair is (on, off), not (off, on)");
        // The two always sum to the stream length, whichever way round they are, which is why the
        // sum could never have caught this.
        let (on, off) = polarity_counts(&lopsided);
        assert_eq!(on + off, lopsided.events.len());
    }

    /// `shuffle` is documented as Fisher-Yates, which draws `j` from `0..=i` and therefore leaves
    /// an element where it found it about once per shuffle whatever the length. Drawing from
    /// `0..i` instead is Sattolo's algorithm, which is still deterministic, still a permutation,
    /// still not the identity and still the same one for the same seed — every property
    /// `shuffle_is_a_deterministic_permutation_that_actually_moves_things` asserts. Its one
    /// assertion about fixed points is `fixed < 8`, an **upper** bound, and a cycle satisfies an
    /// upper bound by having none at all; nothing anywhere put a floor under the count. The
    /// smallest case that does is two elements, where Fisher-Yates leaves the pair alone about half
    /// the time and Sattolo reverses it every single time.
    #[test]
    fn the_shuffle_sometimes_leaves_an_element_where_it_found_it() {
        let mut rng = Rng::new(0x1234);
        let mut stayed = 0usize;
        for _ in 0..64 {
            let mut pair = [0u32, 1];
            super::shuffle(&mut pair, &mut rng);
            if pair == [0, 1] {
                stayed += 1;
            }
        }
        // This implementation measures 38 of 64 — a fair coin would give 32, and 38 is 1.5
        // standard deviations off it. A cycle would give 0 and an identity would give 64.
        assert_eq!(stayed, 38, "a two-element shuffle left the pair alone {stayed} times in 64");

        // The same property on a longer array, where the expected number of fixed points in a
        // uniform permutation is exactly 1 per shuffle whatever the length. This implementation
        // measures 60 over 64 shuffles of ten elements; a cycle would measure 0.
        let mut rng = Rng::new(0x99);
        let mut fixed = 0usize;
        for _ in 0..64 {
            let mut v: Vec<usize> = (0..10).collect();
            super::shuffle(&mut v, &mut rng);
            fixed += v.iter().enumerate().filter(|(i, x)| i == *x).count();
        }
        assert_eq!(fixed, 60, "ten-element shuffles produced {fixed} fixed points in 64 tries");
    }

    /// `collect_split`'s doc says a key is rejected if it has been seen "in **any** class", and the
    /// module doc turns that into the disjointness guarantee. No generator in this module can
    /// produce a cross-class collision — each one puts the class into the spike times — so moving
    /// the `seen` list inside the per-class loop changes nothing any fixture can reach. Calling
    /// `collect_split` directly with a generator that offers *the same inputs to both classes* is
    /// the only way to ask the question.
    #[test]
    fn an_input_already_used_by_one_class_cannot_be_reused_by_another() {
        // Class 0 and class 1 are handed the identical pair of inputs. Globally-distinct keys mean
        // class 1 can never fill, and the budget runs out with nothing accepted.
        let shared = super::collect_split(2, 1, 1, 7, |class, accepted| super::Sample {
            train: Train::from_spikes(vec![Spike { t: accepted as u64, source: 0 }]),
            label: class,
        });
        assert!(
            matches!(shared, Err(TaskError::Exhausted { wanted: 2, distinct: 0, .. })),
            "class 1 re-used class 0's inputs: {shared:?}"
        );

        // The positive control, without which the above is a test that `collect_split` refuses
        // everything: the same generator with the classes offset by ten ticks fills both.
        let (train, test) = super::collect_split(2, 1, 1, 7, |class, accepted| super::Sample {
            train: Train::from_spikes(vec![Spike {
                t: accepted as u64 + 10 * u64::from(class),
                source: 0,
            }]),
            label: class,
        })
        .expect("distinct inputs per class must fill");
        assert_eq!((train.len(), test.len()), (2, 2));
    }

    /// `condition_sample`'s doc says `false` is early and `true` is late, per channel. Its only
    /// test checks the label — which is the exclusive-or of the two bits and is unchanged by
    /// exchanging early for late — and the per-channel spike multisets, which over the four
    /// conditions are also unchanged by it. The asymmetric condition is the witness: with bits
    /// `(false, true)` channel 0 must fire at `early_tick` and channel 1 at `late_tick`, and a
    /// reading that swapped them would put an identical label on the mirror-image stimulus.
    #[test]
    fn a_condition_sample_puts_the_early_volley_on_the_channel_whose_bit_is_false() {
        let x = TemporalXor::default();
        let s = x.condition_sample(false, true);
        assert_eq!(Dataset::first_spike(&s, 0), Some(x.early_tick), "bit false is not early");
        assert_eq!(Dataset::first_spike(&s, 1), Some(x.late_tick), "bit true is not late");
        assert_eq!(s.label, 1);

        let mirror = x.condition_sample(true, false);
        assert_eq!(Dataset::first_spike(&mirror, 0), Some(x.late_tick));
        assert_eq!(Dataset::first_spike(&mirror, 1), Some(x.early_tick));
        assert_eq!(mirror.label, 1);
        // The two class-1 rows carry the SAME label and OPPOSITE channel orders, which is what
        // makes the label alone unable to see this.
        assert_ne!(s.train, mirror.train);

        // The symmetric rows, for completeness: both channels on the same side.
        for (bit, tick) in [(false, x.early_tick), (true, x.late_tick)] {
            let flat = x.condition_sample(bit, bit);
            assert_eq!(Dataset::first_spike(&flat, 0), Some(tick));
            assert_eq!(Dataset::first_spike(&flat, 1), Some(tick));
            assert_eq!(flat.label, 0);
        }
    }

    /// `TemporalXor::generate`'s `# Errors` section promises a refusal "when the jitter could push
    /// a volley outside `0..ticks`", and no fixture ever sat near that edge — every configuration
    /// in the suite has a window far longer than it needs, so the check could be deleted outright
    /// and nothing would notice. The floor is arithmetic: the last spike of a late volley lands at
    /// `late_tick + (spikes_per_channel - 1) * burst_gap + jitter_ticks`, which for the default is
    /// `55 + 4 + 6 = 65`, so 66 ticks is the shortest window that holds it and 65 clips it.
    #[test]
    fn a_temporal_xor_window_too_short_for_a_jittered_volley_is_refused() {
        let x = TemporalXor::default();
        let span = u64::from(x.spikes_per_channel - 1) * x.burst_gap;
        let last = x.late_tick + span + x.jitter_ticks;
        assert_eq!(last, 65);

        let clipped = TemporalXor { ticks: last, ..x };
        assert!(
            matches!(
                clipped.generate(),
                Err(TaskError::OutOfRange { what: "ticks", low, .. })
                    if (low - (last + 1) as f64).abs() < f64::EPSILON
            ),
            "a window of {last} ticks cannot hold a volley whose last spike is at tick {last}"
        );
        // One tick more and it generates, so the check is a boundary rather than a ban — and the
        // spikes really do reach the end of it.
        let exact = TemporalXor { ticks: last + 1, per_class_train: 16, per_class_test: 8, ..x };
        let d = exact.generate().unwrap();
        let latest = d
            .train
            .iter()
            .chain(d.test.iter())
            .flat_map(|s| s.train.spikes().iter().map(|sp| sp.t))
            .max()
            .unwrap();
        assert!(
            latest < exact.ticks,
            "a spike at {latest} is outside a {}-tick window",
            exact.ticks
        );
        assert_eq!(latest, last, "no sample reached the tick the floor was derived from");
    }

    /// `Coincidence`'s doc says the coincident class draws its gap from `0 ..= threshold_ticks`,
    /// inclusive, and its `classify_by_interval` puts the boundary at `gap <= threshold_ticks` —
    /// so a gap of exactly the threshold is the one input that distinguishes an inclusive rule from
    /// an exclusive one. A span of `threshold_ticks` rather than `threshold_ticks + 1` still leaves
    /// 8 reachable gaps times 31 jitters, which is 248 distinct inputs against the 192 a class
    /// needs — so generation still succeeds, the classifier is still perfect, and the class is
    /// still balanced. The only visible trace is the missing value.
    #[test]
    fn the_coincident_class_realises_every_gap_from_zero_to_the_threshold_inclusive() {
        let c = Coincidence::default();
        let d = c.generate().unwrap();
        let realised = |label: u32| {
            let mut g: Vec<u64> = d
                .train
                .iter()
                .chain(d.test.iter())
                .filter(|s| s.label == label)
                .map(|s| s.train.spikes()[0].t.abs_diff(s.train.spikes()[1].t))
                .collect();
            g.sort_unstable();
            g.dedup();
            g
        };
        let coincident: Vec<u64> = (0..=c.threshold_ticks).collect();
        assert_eq!(
            realised(1),
            coincident,
            "the coincident class did not realise every gap in 0..={}",
            c.threshold_ticks
        );
        // The other class starts one past the threshold and stops at the largest gap asked for.
        let distant = realised(0);
        assert_eq!(distant.first(), Some(&(c.threshold_ticks + 1)));
        assert_eq!(distant.last(), Some(&c.max_gap_ticks));
        // The two ranges tile the whole gap space with no value in both and none missing.
        assert_eq!(realised(1).len() + distant.len(), (c.max_gap_ticks + 1) as usize);
    }

    /// `Coincidence::generate`'s `# Errors` section promises a refusal when the jitter does not fit
    /// inside `0..ticks`, and only the top end of that had a fixture. The bottom end is
    /// `first_tick < jitter_ticks`: the jitter is drawn on `[-jitter_ticks, jitter_ticks]` and the
    /// reference spike is placed with `.max(0)`, so without the check a negative shift is silently
    /// clamped to tick 0 — which is not a refusal but a pile-up, and it makes the common-mode
    /// jitter the task is built on stop being uniform exactly where a model would look for it.
    #[test]
    fn a_coincidence_jitter_that_would_push_the_reference_spike_below_tick_zero_is_refused() {
        let c = Coincidence::default();
        assert!(c.first_tick > c.jitter_ticks, "the default must not sit on the boundary");
        let too_low = Coincidence { first_tick: c.jitter_ticks - 1, ..c };
        assert!(
            matches!(
                too_low.generate(),
                Err(TaskError::OutOfRange { what: "first_tick", low, .. })
                    if (low - c.jitter_ticks as f64).abs() < f64::EPSILON
            ),
            "a reference spike {} ticks above zero was accepted with a jitter of {}",
            c.jitter_ticks - 1,
            c.jitter_ticks
        );
        // Exactly at the boundary is legal, and the earliest reference spike it produces is tick 0
        // reached by the draw rather than by the clamp.
        let edge = Coincidence {
            first_tick: c.jitter_ticks,
            per_class_train: 64,
            per_class_test: 32,
            ..c
        };
        let d = edge.generate().unwrap();
        let earliest = d
            .train
            .iter()
            .chain(d.test.iter())
            .map(|s| s.train.spikes()[0].t)
            .min()
            .unwrap();
        assert_eq!(earliest, 0);
    }

    /// `LatencyPatterns`'s field doc says template latencies live in
    /// `jitter_ticks ..= ticks - 1 - jitter_ticks`, and the lower margin is what stops a per-channel
    /// jitter from being clamped at tick 0 — a clamp that would compress the class prototypes
    /// together at exactly the point the `L-infinity` separability guarantee is stated about. At
    /// the defaults the margin is 2 ticks out of a 46-tick band, so random templates clear it
    /// anyway about five times in six and the clamp hides the rest. A window of exactly
    /// `2 * jitter_ticks + 1` collapses the band to a single value and makes the margin the only
    /// thing left to measure.
    #[test]
    fn every_template_latency_leaves_a_jitter_half_width_of_room_below_it() {
        // One value in the band, so every latency is the bottom of it and nothing is random.
        let pinned = LatencyPatterns {
            n_classes: 1,
            n_inputs: 4,
            ticks: 21,
            jitter_ticks: 10,
            min_separation_ticks: 0,
            ..LatencyPatterns::default()
        };
        assert_eq!(pinned.ticks - 2 * pinned.jitter_ticks, 1, "the band must hold one value");
        assert_eq!(pinned.templates().unwrap(), vec![vec![10, 10, 10, 10]]);

        // And the band at the defaults, stated as the closed interval the field doc names.
        let l = LatencyPatterns::default();
        let t = l.templates().unwrap();
        let lo = l.jitter_ticks;
        let hi = l.ticks - 1 - l.jitter_ticks;
        assert_eq!((lo, hi), (2, 47));
        for (c, row) in t.iter().enumerate() {
            for (k, &at) in row.iter().enumerate() {
                assert!(
                    (lo..=hi).contains(&at),
                    "class {c} channel {k} peaks at {at}, outside {lo}..={hi}"
                );
            }
        }
        // The margin is what keeps the clamp in `generate` from ever firing: the earliest spike a
        // jittered sample can carry is `lo - jitter_ticks`, which is 0 and not below it.
        let d = l.generate().unwrap();
        let earliest =
            d.train.iter().flat_map(|s| s.train.spikes().iter().map(|sp| sp.t)).min().unwrap();
        assert!(earliest >= lo - l.jitter_ticks);
    }

    /// `bar_events` spreads each leading-edge column across the axis the bar does **not** travel
    /// along: a bar moving right is vertical and spans every row, so the cross extent is `height`.
    /// Every `MovingBar` fixture in the suite is a 16x16 square, where `width` and `height` are the
    /// same number — vacuous-test mechanism #160 in the register, a parameter that is identical in
    /// every fixture makes the choice between it and its twin invisible. On an 8x4 grid the two
    /// readings differ by a factor of four in the event count, in opposite directions for the two
    /// orientations.
    #[test]
    fn a_bar_spans_the_axis_it_does_not_travel_along_on_a_grid_that_is_not_square() {
        let b = MovingBar { width: 8, height: 4, ticks: 60, ..MovingBar::default() };
        assert_ne!(b.width, b.height, "the point of this fixture is a grid that is not square");
        let want = b.events_per_traverse();
        assert_eq!(want, 64);
        for dir in Direction::all() {
            let ev = b.traverse(dir, 1.0).unwrap();
            assert_eq!(
                ev.len() as u64,
                want,
                "{dir:?} emitted {} events over an 8x4 grid, not {want}",
                ev.len()
            );
            // Every pixel of the grid, exactly once on and once off — which is the claim the count
            // is shorthand for, and which an out-of-range cross extent breaks by addressing rows
            // that do not exist.
            let pixels = b.width * b.height;
            for pol in [Polarity::On, Polarity::Off] {
                let mut seen: Vec<u32> =
                    ev.iter().filter(|e| e.polarity == pol).map(|e| e.address).collect();
                seen.sort_unstable();
                assert_eq!(seen, (0..pixels).collect::<Vec<u32>>(), "{dir:?} {pol:?}");
            }
        }
    }

    /// The speed jitter is documented as a fractional half-width and `min_speed` is derived from
    /// it, so a sample must be able to come out *slower* than the nominal speed as well as faster.
    /// Drawing `u` from `[0, 1)` instead of `[-1, 1)` makes `min_speed` decorative — and nothing
    /// noticed, because the traverse event count is invariant to speed on purpose, the polarity
    /// split is invariant too, and the window floor derived from `min_speed` is only ever asserted
    /// against configurations with room to spare. What moves with the speed is the *duration*: with
    /// the start delay held at zero, the last event of a rightward traverse lands at tick 24 at
    /// `min_speed` and tick 16 at `max_speed`, against 19 at the nominal speed.
    #[test]
    fn the_speed_jitter_reaches_below_the_nominal_speed_as_well_as_above_it() {
        let b = MovingBar {
            max_start_delay: 0,
            per_class_train: 24,
            per_class_test: 12,
            ..MovingBar::default()
        };
        let end_at =
            |v: f64| b.bar_events(Direction::Right, 0, v).iter().map(|e| e.t).max().unwrap();
        let (slowest, nominal, fastest) =
            (end_at(b.min_speed()), end_at(b.speed_px_per_tick), end_at(b.max_speed()));
        assert_eq!((slowest, nominal, fastest), (24, 19, 16));

        let d = b.generate().unwrap();
        let ends: Vec<u64> = d
            .train
            .iter()
            .chain(d.test.iter())
            .filter(|s| s.label == 0)
            .map(|s| s.events.iter().map(|e| e.t).max().unwrap())
            .collect();
        let (lo, hi) = (*ends.iter().min().unwrap(), *ends.iter().max().unwrap());
        assert!(
            hi > nominal,
            "no rightward sample was slower than the nominal speed: the slowest ended at {hi}, \
             the nominal traverse ends at {nominal}"
        );
        assert!(lo < nominal, "no rightward sample was faster than the nominal speed");
        // The realised extremes are the ones the speed bounds imply, which is the other half of
        // the claim: the jitter reaches both ends of its own interval.
        assert_eq!((lo, hi), (fastest, slowest));
    }

    /// `SpokenDigits`'s doc says spikes are drawn with `p = 1 - exp(-rate * dt)`, "the same exact
    /// form as `RateEncoder`". `RateDiscrimination::p_tick` has a test for exactly that; `p_at` had
    /// none, and could not get one from the tests it already had, because `p_at` feeds **both** the
    /// generator and `moments` — so `the_spoken_digit_spike_count_matches_its_analytic_mean`
    /// compares the formula against itself and agrees whichever formula it is. The reference here
    /// is recomputed in the test instead. At the 200 Hz peak with 1 ms ticks the two forms are
    /// `1 - exp(-0.2) = 0.18127` and `0.2`, a 10.3% error, and it compounds to 44.90 expected
    /// spikes against 48.13 over a whole sample.
    #[test]
    fn the_envelope_per_tick_probability_is_the_exact_poisson_form_not_its_linear_approximation() {
        let s = SpokenDigits::default();
        // At the peak of the flat class the rate is `peak_hz` exactly, so the arithmetic is closed.
        let at_peak = s.p_at(2, s.n_channels / 2, s.ticks / 2, 0);
        let exact = 1.0 - (-s.peak_hz * s.dt).exp();
        assert!((at_peak - exact).abs() < 1e-15, "{at_peak} is not 1 - exp(-0.2)");
        assert!(
            (at_peak - 0.18126924692201818).abs() < 1e-15,
            "the peak probability moved from the transcribed 1 - exp(-0.2)"
        );
        // The linear approximation differs in the third decimal place — far outside any tolerance
        // a probability is compared at here.
        assert!((at_peak - s.peak_hz * s.dt).abs() > 0.018);

        // Recomputed independently of `p_at`, so the analytic mean is compared against arithmetic
        // and not against the function that produced it.
        let mut reference = 0.0f64;
        for k in 0..s.n_channels {
            for t in 0..s.ticks {
                let d = (t as i64 - s.peak_tick(2, k)) as f64 / s.sigma_ticks;
                let rate = s.peak_hz * (-0.5 * d * d).exp();
                reference += 1.0 - (-rate * s.dt).exp();
            }
        }
        let reported = s.expected_spikes_per_sample(2).unwrap();
        // Same operations in the same order, so equality rather than a tolerance.
        assert_eq!(reported, reference);
        assert!((reported - 44.901_650_658_725_39).abs() < 1e-9, "{reported}");
        // The linear sum this implementation measures for the same class is 48.127, which the
        // existing mean-against-analytic test would have accepted because both sides would move.
        assert!(reference < 45.0);
    }

    /// The per-sample time jitter is drawn and then handed to `p_at` as the envelope's shift. At
    /// the default `time_jitter_ticks == 0` dropping it is literally the identity, and the one
    /// jittered fixture in the suite asserts only that `generate` returns `Ok` — so nothing in the
    /// module could see a jitter that was drawn, charged against the window margin, and then
    /// thrown away. What it moves is the sample's spike centroid, one for one: at a half-width of
    /// 20 ticks this implementation measures a centroid range of 38.2 ticks across the flat class,
    /// against 2.7 ticks for the same configuration with no jitter at all.
    #[test]
    fn the_sample_time_jitter_moves_the_whole_envelope_it_is_drawn_for() {
        let centroid_range = |cfg: &SpokenDigits| {
            let d = cfg.generate().unwrap();
            let c: Vec<f64> = d
                .train
                .iter()
                .chain(d.test.iter())
                .filter(|s| s.label == 2)
                .map(|s| {
                    let sp = s.train.spikes();
                    sp.iter().map(|x| x.t as f64).sum::<f64>() / sp.len() as f64
                })
                .collect();
            c.iter().copied().fold(f64::NEG_INFINITY, f64::max)
                - c.iter().copied().fold(f64::INFINITY, f64::min)
        };
        let base =
            SpokenDigits { per_class_train: 8, per_class_test: 4, ..SpokenDigits::default() };
        let half = 20u64;
        let still = centroid_range(&base);
        let shaken = centroid_range(&SpokenDigits { time_jitter_ticks: half, ..base });
        // Both bounds come from the half-width rather than from the measurement. Class 2 is the
        // flat sweep, so every channel peaks at `ticks / 2` and an unjittered sample's centroid is
        // that tick plus counting noise: a 6-tick envelope over ~45 spikes is well under a tick of
        // standard error, so a quarter of the half-width is an enormous allowance for it. This
        // implementation measures 2.7.
        assert!(
            still < half as f64 / 4.0,
            "the unjittered centroid range is {still} ticks, which is not counting noise"
        );
        // The shift is uniform on `[-half, half]`, so twelve draws of it span most of `2 * half`.
        // Requiring half of that interval is the weakest statement that a shift happened at all;
        // this implementation measures 38.2 of a possible 40.
        assert!(
            shaken > half as f64,
            "a jitter half-width of {half} ticks moved the centroid by only {shaken} ticks — the \
             envelope was built at the unshifted peak"
        );
    }

    /// `SpokenDigits::generate`'s `# Errors` section says a class's "envelope peak **plus the
    /// jitter**" must fit inside four sigma of the window edge, and the test named
    /// `the_window_floor_is_the_one_the_doc_derives` pins that floor at 145/144 — with
    /// `time_jitter_ticks` at its default of **zero**, where the jitter term of the margin is zero
    /// and therefore invisible. The margin is
    /// `ceil(4 * sigma) + time_jitter_ticks`, so raising the half-width to 20 ticks tightens the
    /// window at both ends and moves the floor by exactly `2 * 20`: from 145 to 185.
    #[test]
    fn the_window_margin_grows_with_the_jitter_half_width() {
        let shaken = SpokenDigits {
            time_jitter_ticks: 20,
            per_class_train: 4,
            per_class_test: 2,
            ..SpokenDigits::default()
        };
        // The steepest sweep puts a peak `6 * 8 = 48` ticks off centre, and the margin is
        // `ceil(4 * 6) + 20 = 44`, so the window needs `2 * (48 + 44) + 1 = 185` ticks.
        assert_eq!(shaken.peak_tick(4, 0).abs_diff((shaken.ticks / 2) as i64), 48);
        let margin = (4.0 * shaken.sigma_ticks).ceil() as u64 + shaken.time_jitter_ticks;
        let floor = 2 * (48 + margin) + 1;
        assert_eq!(floor, 185);

        assert!(
            SpokenDigits { ticks: floor, ..shaken }.generate().is_ok(),
            "a window at its own floor must generate"
        );
        assert!(
            matches!(
                SpokenDigits { ticks: floor - 1, ..shaken }.generate(),
                Err(TaskError::OutOfRange { what: "ticks", .. })
            ),
            "a {}-tick window clips a 20-tick jitter off a four-sigma envelope",
            floor - 1
        );
        // The zero-jitter floor is 40 ticks lower, which is the jitter half-width counted once at
        // each end — the term the margin was missing.
        let still = SpokenDigits { time_jitter_ticks: 0, ..shaken };
        assert!(SpokenDigits { ticks: 145, ..still }.generate().is_ok());
        assert!(matches!(
            SpokenDigits { ticks: 144, ..still }.generate(),
            Err(TaskError::OutOfRange { what: "ticks", .. })
        ));
        assert_eq!(floor - 145, 2 * shaken.time_jitter_ticks);
    }

    /// A refusal's `low` is the number a user changes the config to, and the test named
    /// `a_cue_window_too_short_for_its_spikes_and_jitter_is_refused` already holds `DelayedMatch`
    /// to reporting the real one. `SpokenDigits` was reporting a
    /// different quantity: `2 * (|peak| + margin) + 1` off the peak's **absolute** tick, where the
    /// floor is set by the peak's **distance from the window centre** — the two differ by the whole
    /// half-window, because `peak_tick` is `ticks / 2 + slope * (channel - mid)` and widening the
    /// window carries the centre and the peaks together. Nothing saw it because the only test that
    /// generated a refusal here matched on `what: "ticks"` and read no further, and the module's
    /// own derivation of the floor lives in a doc comment rather than in an assertion.
    ///
    /// Two separate failures, both measured on this implementation. At the default config and
    /// `ticks = 144` it asked for **289** where the `SpokenDigits::default` doc, and the test
    /// beside this one, put the floor at **145**. And a peak clipped at the *bottom* has a small
    /// absolute tick, so a down-sweep of `slope_step_ticks = -7` at 200 ticks reported
    /// `low = 97` — a bound the refused value of 200 already satisfies, printed as the reason it
    /// was refused; following it to 97 refused again and asked for 177, which refuses again and
    /// asks for 97. The true floor there is 273.
    #[test]
    fn the_window_refusal_names_the_floor_the_window_actually_has() {
        let floor_of = |cfg: &SpokenDigits, at: u64| {
            match (SpokenDigits { ticks: at, ..*cfg }).generate() {
                Err(TaskError::OutOfRange { what: "ticks", value, low, .. }) => {
                    assert!((value - at as f64).abs() < f64::EPSILON);
                    assert!(
                        low > value,
                        "a window of {value} ticks was refused for being under a floor of {low}"
                    );
                    low as u64
                }
                other => panic!("{at} ticks produced {other:?}"),
            }
        };
        // The bound is the smallest window that generates, found by search rather than asserted:
        // a `low` that is merely sufficient is still the wrong number to print.
        let smallest = |cfg: &SpokenDigits| {
            (1u64..800)
                .find(|&t| (SpokenDigits { ticks: t, ..*cfg }).generate().is_ok())
                .expect("some window in 1..800 must work")
        };

        let one = SpokenDigits { per_class_train: 1, per_class_test: 1, ..SpokenDigits::default() };
        for cfg in [
            one,
            SpokenDigits { time_jitter_ticks: 20, ..one },
            SpokenDigits { slope_step_ticks: -7, ..one },
            SpokenDigits { slope_step_ticks: 7, ..one },
        ] {
            let real = smallest(&cfg);
            assert_eq!(
                floor_of(&cfg, real - 1),
                real,
                "the refusal at {} names the wrong floor",
                real - 1
            );
            // Same floor whatever window it is asked from, because the floor is a property of the
            // sweep and the envelope and not of the window that failed to hold them.
            assert_eq!(floor_of(&cfg, 1), real);
        }

        // The four floors, as numbers, so they cannot drift: `2 * (furthest + margin) + 1` with a
        // margin of `ceil(4 * sigma) + time_jitter_ticks` and a furthest peak offset of
        // `|slope_step| * (n_classes - 1) / 2 * n_channels / 2`.
        assert_eq!(smallest(&one), 145);
        assert_eq!(smallest(&SpokenDigits { time_jitter_ticks: 20, ..one }), 185);
        assert_eq!(smallest(&SpokenDigits { slope_step_ticks: -7, ..one }), 273);
        assert_eq!(smallest(&SpokenDigits { slope_step_ticks: 7, ..one }), 273);
        // 2 * (48 + 24) + 1, 2 * (48 + 44) + 1, 2 * (112 + 24) + 1.
        assert_eq!(2 * (48 + 24) + 1, 145);
        assert_eq!(2 * (48 + 44) + 1, 185);
        assert_eq!(2 * (112 + 24) + 1, 273);
    }
}
