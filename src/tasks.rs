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
        /// Name of the parameter, as it is spelled on the config struct.
        what: &'static str,
        /// The value supplied.
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
            // The polarity is part of the input and goes in the key. An event stream that differs
            // only in sign is a different stimulus — dropping the bit here would let a leftward
            // sweep and a rightward one collide and be rejected as duplicates.
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
    pub majority_baseline: f64,
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
    /// zeros would report "perfectly imbalanced" for "nothing measured".
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
/// `make(class, draw_index)` is called until `per_class_train + per_class_test` samples with keys
/// never seen before — in **any** class — have been accepted. Class counts are therefore exact, and
/// the train/test split is disjoint by construction rather than by inspection.
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
    let per_class = per_class_train + per_class_test;
    nonzero("samples per class", per_class as u64)?;

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
            let s = make(c, draws);
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
        majority_baseline: 0.0,
        seed,
        stands_in_for,
        not_captured,
    };
    // MEASURED off the test split, not assumed from the per-class counts requested. If a future
    // change to a generator unbalances a split, this number moves and the task's own balance test
    // fails, which is the outcome we want.
    let bal = d.balance(Split::Test);
    d.majority_baseline = bal.into_iter().fold(0.0f64, f64::max);
    d
}

/// A uniform integer draw in `[-half, half]`, used for every jitter in this module.
fn jitter(rng: &mut Rng, half: u64) -> i64 {
    if half == 0 {
        return 0;
    }
    let span = 2 * half + 1;
    i64::from(rng.below(span as u32)) - half as i64
}

// ---------------------------------------------------------------------------------------------
// Counting distributions, for the one task with an analytic optimum
// ---------------------------------------------------------------------------------------------

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
/// **per-channel spike-time histogram** is also identical between classes: channel 0 is early half
/// the time in both classes, and so is channel 1. So even a per-channel *latency* readout — mean
/// first-spike time, one number per channel — is blind. [`TemporalXor::condition_sample`] builds
/// the four jitter-free conditions so a test can assert exactly that, and
/// `temporal_xor_is_blind_to_every_per_channel_readout` does.
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
    /// volley outside `0..ticks`, or when `dt` is not positive and finite;
    /// [`TaskError::Exhausted`] when the jitter admits fewer distinct inputs than the split needs
    /// — which is always the case for `jitter_ticks == 0`, where only four inputs exist.
    pub fn generate(&self) -> Result<Dataset<Sample>, TaskError> {
        in_range("dt", self.dt, f64::MIN_POSITIVE, f64::MAX)?;
        nonzero("spikes_per_channel", u64::from(self.spikes_per_channel))?;
        nonzero("burst_gap", self.burst_gap)?;
        if self.late_tick <= self.early_tick {
            return Err(TaskError::OutOfRange {
                what: "late_tick",
                value: self.late_tick as f64,
                low: (self.early_tick + 1) as f64,
                high: self.ticks as f64,
            });
        }
        let span = u64::from(self.spikes_per_channel - 1) * self.burst_gap;
        if self.early_tick < self.jitter_ticks {
            return Err(TaskError::OutOfRange {
                what: "early_tick",
                value: self.early_tick as f64,
                low: self.jitter_ticks as f64,
                high: self.late_tick as f64,
            });
        }
        let last = self.late_tick + span + self.jitter_ticks;
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
            |class, draw| {
                // Conditions alternate on the draw index rather than on a coin, so the two rows of
                // each class appear in equal numbers up to the few draws duplicate-rejection eats.
                let second = draw % 2 == 1;
                let (a, b) = match (class, second) {
                    (0, false) => (false, false),
                    (0, true) => (true, true),
                    (_, false) => (false, true),
                    (_, true) => (true, false),
                };
                let ja = jitter(&mut rng, cfg.jitter_ticks);
                let jb = jitter(&mut rng, cfg.jitter_ticks);
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
    /// unreachable), when the jitter and the largest gap do not fit inside `0..ticks`, or when
    /// `dt` is not positive and finite; [`TaskError::Exhausted`] when the reachable
    /// `(gap, jitter)` combinations are fewer than the split needs.
    pub fn generate(&self) -> Result<Dataset<Sample>, TaskError> {
        in_range("dt", self.dt, f64::MIN_POSITIVE, f64::MAX)?;
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
        let last = self.first_tick + self.jitter_ticks + self.max_gap_ticks;
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
            self.seed ^ 0xA5A5_0002,
            |class, _| {
                let gap = if class == 1 {
                    // Coincident: 0 ..= threshold.
                    u64::from(rng.below((cfg.threshold_ticks + 1) as u32))
                } else {
                    // Not coincident: threshold+1 ..= max_gap.
                    let span = cfg.max_gap_ticks - cfg.threshold_ticks;
                    cfg.threshold_ticks + 1 + u64::from(rng.below(span as u32))
                };
                let shift = jitter(&mut rng, cfg.jitter_ticks);
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
    /// asked for; the cue window is 30 ms rather than 20 because a four-spike cue spanning 21 ms
    /// plus jitter on both sides does not fit in 20.
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
    #[must_use]
    pub fn ticks(&self) -> u64 {
        2 * self.cue_ticks + self.delay_ticks
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
    #[must_use]
    pub fn delay_end(&self) -> u64 {
        self.cue_ticks + self.delay_ticks
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
    /// requested but do not fit inside the delay, or when `dt` is not positive and finite;
    /// [`TaskError::Exhausted`] when the reachable symbol-and-jitter combinations are fewer than
    /// the split needs.
    pub fn generate(&self) -> Result<Dataset<Sample>, TaskError> {
        in_range("dt", self.dt, f64::MIN_POSITIVE, f64::MAX)?;
        nonzero("cue_spikes", u64::from(self.cue_spikes))?;
        nonzero("cue_ticks", self.cue_ticks)?;
        if self.n_symbols < 2 {
            return Err(TaskError::OutOfRange {
                what: "n_symbols",
                value: f64::from(self.n_symbols),
                low: 2.0,
                high: f64::from(u32::MAX),
            });
        }
        let span = u64::from(self.cue_spikes - 1) * self.spacing();
        if span + 2 * self.jitter_ticks >= self.cue_ticks {
            return Err(TaskError::OutOfRange {
                what: "cue_ticks",
                value: self.cue_ticks as f64,
                low: (span + 2 * self.jitter_ticks + 1) as f64,
                high: f64::MAX,
            });
        }
        if self.distractor_cues > 0 && span + 1 >= self.delay_ticks {
            return Err(TaskError::OutOfRange {
                what: "delay_ticks",
                value: self.delay_ticks as f64,
                low: (span + 2) as f64,
                high: f64::MAX,
            });
        }

        let ticks = self.ticks();
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
                let j1 = jitter(&mut rng, cfg.jitter_ticks);
                let j2 = jitter(&mut rng, cfg.jitter_ticks);
                cue(&mut sp, sample_sym, cfg.jitter_ticks, j1);
                cue(&mut sp, test_sym, cfg.delay_end() + cfg.jitter_ticks, j2);
                for _ in 0..cfg.distractor_cues {
                    let sym = rng.below(cfg.n_symbols);
                    let room = cfg.delay_ticks - span;
                    let at = cfg.delay_start() + u64::from(rng.below(room as u32));
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
    /// No model on this configuration can do better than 0.7324, and one reporting 0.80 has a
    /// leak.
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
    /// upside down and a "bound" below chance would be reported as a bound), or when `ticks` is
    /// zero — a window of no length carries no counts and has no discrimination at all, which is a
    /// different statement from "chance".
    #[must_use]
    pub fn optimal_accuracy(&self) -> Option<f64> {
        if self.ticks == 0 || self.rate_lo > self.rate_hi {
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
        if !k_max.is_finite() || k_max > 1e7 {
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
    /// leaves no room for a latency after the jitter margin; [`TaskError::Exhausted`] when
    /// rejection sampling could not place that many templates at the requested separation, which
    /// happens when `min_separation_ticks` approaches the usable window.
    pub fn templates(&self) -> Result<Vec<Vec<u64>>, TaskError> {
        nonzero("n_classes", u64::from(self.n_classes))?;
        nonzero("n_inputs", u64::from(self.n_inputs))?;
        if self.ticks <= 2 * self.jitter_ticks {
            return Err(TaskError::OutOfRange {
                what: "ticks",
                value: self.ticks as f64,
                low: (2 * self.jitter_ticks + 1) as f64,
                high: f64::MAX,
            });
        }
        let lo = self.jitter_ticks;
        let span = self.ticks - 2 * self.jitter_ticks;

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
                (0..self.n_inputs).map(|_| lo + u64::from(rng.below(span as u32))).collect();
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
    /// is not positive and finite.
    pub fn generate(&self) -> Result<Dataset<Sample>, TaskError> {
        in_range("dt", self.dt, f64::MIN_POSITIVE, f64::MAX)?;
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
                    let off = jitter(&mut rng, cfg.jitter_ticks);
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
        let extent = f64::from(self.width.max(self.height) + self.bar_width);
        let t = (extent / v).ceil();
        if !t.is_finite() || t > 1e12 {
            return None;
        }
        Some(self.max_start_delay + t as u64 + 2)
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
        let need = ((f64::from(extent + self.bar_width)) / speed).ceil() as u64 + 2;
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

    fn bar_events(&self, direction: Direction, delay: u64, speed: f64) -> Vec<Event> {
        let mut out = Vec::new();
        let cross = if direction.is_horizontal() { self.height } else { self.width };
        let mut prev = self.occupancy(direction, delay, speed, 0);
        for t in 1..self.ticks {
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
    /// `[0, 0.9]`, a noise probability outside `[0, 1)`, a `dt` that is not positive and finite, or
    /// a window shorter than [`MovingBar::min_ticks`]; [`TaskError::Exhausted`] when the jitters
    /// admit fewer distinct streams than the split needs.
    pub fn generate(&self) -> Result<Dataset<EventSample>, TaskError> {
        in_range("dt", self.dt, f64::MIN_POSITIVE, f64::MAX)?;
        in_range("speed_jitter_frac", self.speed_jitter_frac, 0.0, 0.9)?;
        in_range("noise_prob_per_pixel_per_tick", self.noise_prob_per_pixel_per_tick, 0.0, 1.0)?;
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
                let delay = u64::from(rng.below((cfg.max_start_delay + 1) as u32));
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
            self.width * self.height,
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
    /// extreme peak sits `6 * 8 = 48` ticks off centre and needs 24 more on each side, so 200 is
    /// the first round number that fits. A shorter window clips an envelope, which changes both
    /// the expected count and the centroid, and `generate` refuses rather than quietly clipping.
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
    #[must_use]
    pub fn slope(&self, class: u32) -> i64 {
        let mid = i64::from(self.n_classes.saturating_sub(1)) / 2;
        self.slope_step_ticks * (i64::from(class) - mid)
    }

    /// Tick at which channel `k` of `class` peaks, before jitter.
    #[must_use]
    pub fn peak_tick(&self, class: u32, channel: u32) -> i64 {
        let mid = i64::from(self.n_channels) / 2;
        let centre = (self.ticks / 2) as i64;
        centre + self.slope(class) * (i64::from(channel) - mid)
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
    /// `sigma_ticks`, `peak_hz` or `dt` is not positive and finite, or when a class's envelope
    /// peak plus the jitter would fall outside `4 * sigma` of the window edge — which would clip
    /// the envelope and quietly change both the count and the centroid;
    /// [`TaskError::Exhausted`] when the window admits fewer distinct spike patterns than the split
    /// needs.
    pub fn generate(&self) -> Result<Dataset<Sample>, TaskError> {
        in_range("dt", self.dt, f64::MIN_POSITIVE, f64::MAX)?;
        in_range("sigma_ticks", self.sigma_ticks, f64::MIN_POSITIVE, f64::MAX)?;
        in_range("peak_hz", self.peak_hz, f64::MIN_POSITIVE, f64::MAX)?;
        nonzero("n_classes", u64::from(self.n_classes))?;
        nonzero("n_channels", u64::from(self.n_channels))?;
        nonzero("ticks", self.ticks)?;

        let margin = (4.0 * self.sigma_ticks).ceil() as i64 + self.time_jitter_ticks as i64;
        for c in 0..self.n_classes {
            for k in 0..self.n_channels {
                let p = self.peak_tick(c, k);
                if p - margin < 0 || p + margin >= self.ticks as i64 {
                    return Err(TaskError::OutOfRange {
                        what: "ticks",
                        value: self.ticks as f64,
                        low: (2 * (p.abs().max(margin) + margin) + 1) as f64,
                        high: f64::MAX,
                    });
                }
            }
        }

        let mut rng = Rng::new(self.seed);
        let cfg = *self;
        let (train, test) = collect_split(
            self.n_classes,
            self.per_class_train,
            self.per_class_test,
            self.seed ^ 0xA5A5_0007,
            |class, _| {
                let shift = jitter(&mut rng, cfg.time_jitter_ticks);
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
    use crate::spike::Polarity;

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
        assert!((x.majority_baseline - 0.5).abs() < 1e-15);

        let l = LatencyPatterns::default().generate().unwrap();
        check(l.counts(Split::Test), 50, "latency test");
        assert!((l.chance - 0.2).abs() < 1e-15);
        assert!((l.majority_baseline - 0.2).abs() < 1e-15);

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
        assert!((m2.majority_baseline - 0.5).abs() < 1e-15);

        let r = RateDiscrimination::default().generate().unwrap();
        check(r.counts(Split::Test), 400, "rate test");
        assert!((r.majority_baseline - 0.5).abs() < 1e-15);

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
        assert!((d.majority_baseline - 0.5).abs() < 1e-15);
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
        assert!((re.majority_baseline - want).abs() < 1e-12);
        assert!(re.majority_baseline > 0.5, "baseline {} did not move", re.majority_baseline);
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
    #[test]
    fn doubling_the_speed_halves_the_traverse_and_doubles_the_event_rate() {
        let b = MovingBar { bar_width: 4, ticks: 400, ..MovingBar::default() };
        let span = |v: f64| {
            let ev = b.traverse(Direction::Right, v).unwrap();
            let first = ev.first().unwrap().t;
            let last = ev.last().unwrap().t;
            (last - first + 1) as f64
        };
        let slow = span(0.5);
        let fast = span(1.0);
        let ratio = slow / fast;
        assert!((ratio - 2.0).abs() < 0.1, "traverse span ratio {ratio}, expected 2");
        let n = b.events_per_traverse() as f64;
        let rate_slow = n / slow;
        let rate_fast = n / fast;
        assert!(
            ((rate_fast / rate_slow) - 2.0).abs() < 0.1,
            "event rate ratio {}",
            rate_fast / rate_slow
        );
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
        }
    }

    /// Every dataset carries its own caveat. A blank one is a benchmark that will be over-quoted.
    #[test]
    fn every_dataset_says_what_it_does_not_capture() {
        let checks: Vec<(&str, &str, &str)> = vec![
            {
                let d = TemporalXor::default().generate().unwrap();
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
        for (name, stands, not) in checks {
            assert!(stands.len() > 20, "{name}: stands_in_for is too short to mean anything");
            assert!(not.len() > 20, "{name}: not_captured is too short to mean anything");
        }
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
}
