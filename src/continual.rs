//! Learning without forgetting: the on-chip case, with the forgetting measured first.
//!
//! # The lesson
//!
//! On-chip learning exists so that a deployed machine can adapt after it leaves the factory. The
//! reason it usually cannot is **catastrophic forgetting**: train the same parameters on task B and
//! what they encoded about task A is gone, not degraded — gone, back to near chance, in a few
//! thousand gradient steps. `McCloskey` and Cohen named it (*Catastrophic Interference in
//! Connectionist Networks*, Psychology of Learning and Motivation 24:109, 1989) and Ratcliff
//! (Psychological Review 97:285, 1990) showed it is not a bug in one architecture but a consequence
//! of storing two distributions in one set of weights.
//!
//! This is the binding constraint on every "learns at the edge" claim. A chip that can run a
//! learning rule is not a chip that can keep what it learned. It is also **measurable**, which is
//! the only reason this module exists: the field has a standard protocol — train on tasks in
//! sequence, evaluate on all of them after each — and standard numbers read off it (Lopez-Paz &
//! Ranzato, *Gradient Episodic Memory for Continual Learning*, `NeurIPS` 2017).
//!
//! ## The order of business, which is not negotiable
//!
//! **The collapse is demonstrated before anything is offered to fix it.** Without it every
//! mitigation below is unfalsifiable: it is being credited for preventing something that was not
//! happening. On the default two-task curriculum, [`run`] with [`Protocol::plain`] learns task A to
//! **1.000** and, after one task B, retains **0.244** of it against a chance floor of 0.200. Every
//! row below is a change to that number, with its bill beside it.
//!
//! | Protocol | Task A retained | Task B retained | Extra floats | Gradient steps |
//! |---|---|---|---|---|
//! | [`Protocol::plain`] | 0.244 | 1.000 | 0 | 12 000 |
//! | [`Consolidation::Si`], `c = 0.02` | 0.508 | 0.928 | 180 | 12 000 |
//! | [`Consolidation::Ewc`], `lambda = 150` | 0.812 | 0.800 | 90 | 12 000 |
//! | [`Replay::Experience`], budget 100 | 0.984 | 1.000 | 900 | 13 200 |
//! | [`Replay::Generative`], 50 per task | 1.000 | 1.000 | **160** | 15 000 |
//!
//! Two things in that table are worth more than the ranking. The first is that **generative replay
//! matches real examples at a fifth of the memory** — on a three-task run it reaches 0.715 average
//! accuracy on 240 floats where an experience buffer needs 18 000 floats for 0.713. The second is
//! that consolidation is not free in the direction nobody reports: pushing `lambda` up to hold task
//! A costs task B, monotonically, and this module asserts both halves of that trade rather than the
//! flattering half.
//!
//! ## What actually erases task A, decomposed
//!
//! Three different things can take task A away and a benchmark that does not separate them is
//! measuring its own optimiser. Holding the learner, the seed and the task fixed and changing one
//! thing at a time:
//!
//! | Curriculum | `weight_decay` | Task A retained | What took it |
//! |---|---|---|---|
//! | permutation leaves the mean intact | 0 | **0.980** | nothing |
//! | mean removed ([`Curriculum::permuted`]) | 0 | 0.568 | interference |
//! | permutation leaves the mean intact | `1e-3` | 0.400 | the optimiser |
//! | mean removed ([`Curriculum::permuted`]) | `1e-3` | 0.244 | both |
//!
//! Read rows one and three together, because that is the trap. The un-centred benchmark with weight
//! decay reports a collapse from 1.000 to 0.400, which looks exactly like catastrophic forgetting —
//! and **none of it is interference**, because the same curriculum without weight decay forgets
//! 0.02. Somebody reading only that row would credit a mitigation for undoing their own optimiser.
//!
//! The cause is that a permuted-input benchmark is only as unrelated as the statistics the
//! permutation destroys, and **the per-sample mean survives every permutation**. On
//! [`crate::tasks::LatencyPatterns`] at its defaults the mean latency alone classifies at 0.968, so
//! the un-centred tasks are barely different tasks. [`Curriculum::permuted`] therefore removes the
//! mean, [`remove_sample_mean`] says so out loud, and this module's tests assert all four numbers.
//! This implementation did not locate the same caveat stated for permuted `MNIST`, where mean pixel
//! intensity is invariant in exactly the same way.
//!
//! ## And at five tasks it is not forgetting any more
//!
//! Push the same curriculum to five tasks and the plain learner reaches 0.390 average accuracy —
//! but so does an experience buffer large enough to hold **every example of every task**, which
//! cannot forget anything by construction and still reaches only 0.549. The binding constraint has
//! changed from forgetting to **capacity**, and a continual-learning result that does not report
//! the joint-training ceiling beside it is not separating the two. Reporting that ceiling costs one
//! extra run.
//!
//! ## What each mechanism buys and what it costs
//!
//! | Mechanism | What it stores | Cost for the default curriculum |
//! |---|---|---|
//! | [`Consolidation::Ewc`] | an anchor and a diagonal importance | `2 * parameters`, plus one pass over the task at each boundary |
//! | [`Consolidation::Si`] | anchor, importance, path integral, task-start copy | `4 * parameters`, no boundary pass |
//! | [`Replay::Experience`] | real examples | `budget * (features + 1)` |
//! | [`Replay::Generative`] | a class-conditional density per task | `2 * tasks * classes * features` |
//! | [`Homeostat`] | nothing | nothing, and it buys nothing here — see its doc |
//! | [`Cascade`] | `log2(depth)` extra **bits** per synapse | 3 bits at depth 8, against 64 for one `f64` |
//!
//! The last row is the one native to spiking hardware. Everything above it stores extra **floats**
//! per parameter. A weight in the cascade model of Fusi, Drew and Abbott (*Cascade Models of
//! Synaptically Stored Memories*, Neuron 45:599–611, 2005) is a **single bit** plus a metaplastic
//! depth, and depth is what buys retention. That model's whole claim is a trade, and [`Cascade`]
//! measures both sides: [`Cascade::learning_probability`] falls as `2 / (depth + 1)` — exactly,
//! derived in its doc — while [`Cascade::lifetime`] rises. It also measures the part nobody quotes:
//! **deeper is not unconditionally better.** For a detection floor of 0.05 the best depth is 4; at
//! 0.01 it is 7; at 0.005 it is 8. Past the optimum the memory starts below the floor it would have
//! had to decay to.
//!
//! # Units, and where the dimensionless constants live
//!
//! The learner is a linear softmax readout over **spike-derived features**, and SI leaves the
//! pipeline exactly once, at [`latency_features`]: a first-spike time in ticks becomes a fraction
//! of the sample window centred on its midpoint, so features lie in `-0.5..=0.5`. Everything
//! downstream — learning rates, importance weights, `lambda`, `c` — is dimensionless and carries
//! its source paper's own convention, stated on the item. The spike times that go in are `u64`
//! ticks and a tick is [`crate::tasks::Dataset::dt`] seconds; multiply to get seconds.
//!
//! **One warning about `lambda` and `c`.** They are applied **per update**, at batch size one. The
//! published values are tied to their authors' batch sizes, so a `lambda` from Kirkpatrick et al.
//! is not comparable to a `lambda` here, and this module does not pretend otherwise. Worse, a
//! penalty stiff enough to break `lr * stiffness < 2` makes gradient descent oscillate to infinity;
//! that used to be reported as chance accuracy, which is the same number an honest failure to learn
//! gives, so it is now [`ContinualError::Diverged`].
//!
//! # What this module is not
//!
//! It is not a deep network. The learner is linear because a linear model has one decision function
//! per weight matrix, so "task A's solution was overwritten" is a statement about a vector rather
//! than about an optimisation trajectory, and because the whole module then runs in under a second
//! with no dependencies. **Catastrophic forgetting is not a deep-learning phenomenon** — it is a
//! consequence of shared capacity and it is fully visible here. What a linear model does *not* show
//! is the literature on representation drift in hidden layers, and that absence is stated rather
//! than papered over: this implementation did not locate a way to exhibit hidden-layer drift
//! without a hidden layer.
//!
//! ```
//! use ferromorphic::continual::{Consolidation, Curriculum, Protocol, Schedule, run};
//! use ferromorphic::tasks::LatencyPatterns;
//!
//! let base = LatencyPatterns::default().generate()?;
//! let course = Curriculum::permuted(&base, 2, 0xC0FF_EE01)?;
//! let sched = Schedule::default();
//!
//! let plain = run(&course, &Protocol::plain(sched))?;
//! let ewc = run(
//!     &course,
//!     &Protocol { consolidation: Consolidation::Ewc { lambda: 150.0 }, ..Protocol::plain(sched) },
//! )?;
//!
//! // Task A is learned perfectly, and then it is lost.
//! assert_eq!(plain.learned(0), Some(1.0));
//! assert!(plain.retained(0).unwrap() < 0.30, "{:?}", plain.retained(0));
//!
//! // Consolidation keeps most of it — and says what that cost, in both currencies.
//! assert!(ewc.retained(0).unwrap() > 0.80);
//! assert!(ewc.retained(1).unwrap() < plain.retained(1).unwrap()); // task B paid for it
//! assert_eq!(ewc.cost.extra_floats, 2 * ewc.cost.model_parameters);
//! assert_eq!(plain.cost.extra_floats, 0);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use crate::metrics::{self, MetricError};
use crate::rng::Rng;
use crate::tasks::{Dataset, Sample, Split, TaskError};

// ---------------------------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------------------------

/// What went wrong, named, rather than a number that happens to be finite.
#[derive(Debug, Clone, PartialEq)]
pub enum ContinualError {
    /// A collection that must hold at least one element held none.
    Empty {
        /// Name of the collection, as it is spelled in the API.
        what: &'static str,
    },
    /// A parameter was `NaN` or infinite. Rejected at the boundary: a non-finite learning rate
    /// turns every weight into `NaN` on the first step and the run then reports chance accuracy,
    /// which reads as a modelling result rather than as a typo.
    NotFinite {
        /// Name of the parameter.
        what: &'static str,
        /// The value rejected.
        value: f64,
    },
    /// A parameter was finite but outside the range it is defined on.
    OutOfRange {
        /// Name of the parameter.
        what: &'static str,
        /// The offending value.
        value: f64,
        /// Smallest acceptable value, inclusive.
        low: f64,
        /// Largest acceptable value, inclusive.
        high: f64,
    },
    /// Two shapes that must agree did not — a feature vector against the model's input width, or
    /// one task in a curriculum against another.
    Mismatch {
        /// What was being compared.
        what: &'static str,
        /// The length required.
        expected: usize,
        /// The length supplied.
        found: usize,
    },
    /// A class label was at or past the declared class count, so it names no output unit.
    Label {
        /// The label found.
        label: u32,
        /// The class count it exceeded.
        n_classes: usize,
    },
    /// A channel carried no spike, so it has no first-spike latency.
    ///
    /// Refused rather than imputed. Substituting the window length for "never fired" feeds the
    /// classifier a measurement nobody made, and the resulting accuracy is a number about the
    /// imputation.
    SilentChannel {
        /// Index of the silent input channel.
        channel: u32,
    },
    /// Rejection sampling could not find enough pairwise-deranged permutations within its budget.
    ///
    /// The usual cause is too few features for the number of tasks asked for; with `f` features
    /// there are only `f!` permutations and the derangement constraint prunes most of them.
    Exhausted {
        /// Permutations wanted.
        wanted: usize,
        /// Permutations accepted before the budget ran out.
        found: usize,
        /// Draws spent.
        draws: usize,
    },
    /// A parameter left the finite numbers during training.
    ///
    /// The cause here is always the same and it is worth naming: gradient descent on a quadratic
    /// penalty of stiffness `k` is stable only while `lr * k < 2`, and a consolidation strength
    /// large enough to break that makes the parameter oscillate with growing amplitude until it
    /// overflows. The run is then refused. It used to be reported as **chance accuracy**, which is
    /// the same number an honest failure to learn produces and is therefore the worst possible way
    /// to report a numerical blow-up.
    Diverged {
        /// Flat index of the parameter, weights first then biases.
        parameter: usize,
        /// The value it reached.
        value: f64,
    },
    /// Power iteration did not reach a stationary distribution within its iteration budget.
    ///
    /// A number computed from a distribution that has not settled is a number about the iteration
    /// count, so it is refused and the residual is reported instead.
    NoStationary {
        /// The `L1` change between the last two iterates when the budget ran out.
        residual: f64,
    },
    /// The underlying task generator refused.
    Task(TaskError),
    /// A metric in [`crate::metrics`] refused.
    Metric(MetricError),
}

impl core::fmt::Display for ContinualError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Empty { what } => write!(f, "{what} is empty"),
            Self::NotFinite { what, value } => write!(f, "{what} is not finite ({value})"),
            Self::OutOfRange { what, value, low, high } => {
                write!(f, "{what} = {value} is outside [{low}, {high}]")
            }
            Self::Mismatch { what, expected, found } => {
                write!(f, "{what}: expected {expected}, found {found}")
            }
            Self::Label { label, n_classes } => {
                write!(f, "label {label} is past the {n_classes} declared classes")
            }
            Self::SilentChannel { channel } => {
                write!(f, "channel {channel} never fired, so it has no latency")
            }
            Self::Exhausted { wanted, found, draws } => write!(
                f,
                "wanted {wanted} pairwise-deranged permutations, found {found} in {draws} draws"
            ),
            Self::Diverged { parameter, value } => write!(
                f,
                "parameter {parameter} diverged to {value}; the penalty is stiffer than this \
                 learning rate can integrate (stability needs lr * stiffness < 2)"
            ),
            Self::NoStationary { residual } => {
                write!(f, "power iteration did not settle; residual {residual}")
            }
            Self::Task(e) => write!(f, "task generation failed: {e}"),
            Self::Metric(e) => write!(f, "metric refused: {e}"),
        }
    }
}

impl std::error::Error for ContinualError {}

impl From<TaskError> for ContinualError {
    fn from(e: TaskError) -> Self {
        Self::Task(e)
    }
}

impl From<MetricError> for ContinualError {
    fn from(e: MetricError) -> Self {
        Self::Metric(e)
    }
}

fn finite(what: &'static str, value: f64) -> Result<f64, ContinualError> {
    if !value.is_finite() {
        return Err(ContinualError::NotFinite { what, value });
    }
    Ok(value)
}

fn in_range(what: &'static str, value: f64, low: f64, high: f64) -> Result<f64, ContinualError> {
    let value = finite(what, value)?;
    if value < low || value > high {
        return Err(ContinualError::OutOfRange { what, value, low, high });
    }
    Ok(value)
}

// ---------------------------------------------------------------------------------------------
// Examples, tasks and the curriculum
// ---------------------------------------------------------------------------------------------

/// Subtract each example's own mean feature from every one of its features, in place.
///
/// **Why this exists, and it is the most useful thing in this module for anyone designing a
/// benchmark.** A permuted-input curriculum is only as unrelated as the statistics the permutation
/// destroys, and the per-sample mean survives *every* permutation. On [`crate::tasks::LatencyPatterns`]
/// at its defaults the mean latency alone classifies at **0.968**, so a model trained on task A
/// walks into task B already holding most of a solution — which shows up as forward transfer of
/// about +0.42 and as a retention floor well above chance, neither of which has anything to do with
/// continual learning. Removing the mean removes the shortcut, at the cost of one degree of freedom
/// per sample.
///
/// This implementation did not locate that caveat stated for permuted `MNIST`, where the same
/// invariance holds for mean pixel intensity.
pub fn remove_sample_mean(examples: &mut [Example]) {
    for e in examples {
        if e.x.is_empty() {
            continue;
        }
        let m = e.x.iter().sum::<f64>() / e.x.len() as f64;
        for v in &mut e.x {
            *v -= m;
        }
    }
}

/// One feature vector and its class.
///
/// Features are dimensionless by the time they get here; see [`latency_features`] for the
/// conversion from ticks.
#[derive(Debug, Clone, PartialEq)]
pub struct Example {
    /// The feature vector. Every example in a [`Task`] has the same length.
    pub x: Vec<f64>,
    /// Class index in `0..n_classes`.
    pub label: u32,
}

/// First-spike latency on each channel, as a fraction of the sample window centred on its midpoint.
///
/// The output is dimensionless and lies in `-0.5..=0.5`: channel `c`'s value is
/// `t_c / (ticks - 1) - 0.5`, where `t_c` is its first spike in ticks. **This is the one place SI
/// leaves the pipeline.** A tick is [`crate::tasks::Dataset::dt`] seconds; multiply if you need the
/// latency in seconds. Centring rather than leaving the range at `0..=1` is a conditioning choice,
/// not a modelling one: it keeps the bias term from having to carry the mean of every feature.
///
/// # Errors
///
/// [`ContinualError::Empty`] when `n_inputs` is zero or `ticks` is below 2 (a one-tick window has
/// no latency to report), [`ContinualError::SilentChannel`] naming the first channel that carried
/// no spike, and [`ContinualError::Mismatch`] naming `"spike source"` for a spike whose source is
/// at or past `n_inputs` — a sample wider than the feature vector it was asked for is refused
/// rather than truncated, for the same reason a channel with no spike is refused rather than
/// imputed.
pub fn latency_features(
    sample: &Sample,
    n_inputs: u32,
    ticks: u64,
) -> Result<Vec<f64>, ContinualError> {
    if n_inputs == 0 {
        return Err(ContinualError::Empty { what: "n_inputs" });
    }
    if ticks < 2 {
        return Err(ContinualError::Empty { what: "ticks" });
    }
    let span = (ticks - 1) as f64;
    let mut out = vec![f64::NAN; n_inputs as usize];
    for sp in sample.train.spikes() {
        let i = sp.source as usize;
        if i < out.len() && out[i].is_nan() {
            out[i] = sp.t as f64 / span - 0.5;
        } else if i >= out.len() {
            // ⛔ A SPIKE FROM A CHANNEL THIS VECTOR DOES NOT HAVE IS REFUSED, NOT DROPPED. The
            // loop below refuses the mirror-image case — a declared channel that carried no spike
            // is a `SilentChannel` rather than an imputed number — and quietly discarding a spike
            // the caller handed in is the same failure in the other direction: they get a feature
            // vector that is missing a measurement and nothing says so. `n_inputs` is the width of
            // the sample, not a window onto the first `n_inputs` channels of a wider one.
            return Err(ContinualError::Mismatch {
                what: "spike source",
                expected: out.len(),
                found: i + 1,
            });
        }
    }
    for (i, v) in out.iter().enumerate() {
        if v.is_nan() {
            return Err(ContinualError::SilentChannel { channel: i as u32 });
        }
    }
    Ok(out)
}

/// One task in a sequence: a train split, a test split, and the input permutation that made it.
#[derive(Debug, Clone, PartialEq)]
pub struct Task {
    /// Short identifier, `"<base name>/perm<k>"`, carrying the position in the sequence.
    pub name: String,
    /// Examples the learner may fit.
    pub train: Vec<Example>,
    /// Examples it is scored on; disjoint from `train` because the underlying
    /// [`crate::tasks::Dataset`] splits are.
    pub test: Vec<Example>,
    /// Feature `k` of this task is feature `permutation[k]` of the base dataset. Task 0 is the
    /// identity, so it *is* the base task.
    pub permutation: Vec<usize>,
    /// Number of classes; labels run `0..n_classes`.
    pub n_classes: usize,
    /// Accuracy of uniform guessing, `1 / n_classes`. The floor a retention number is read against.
    pub chance: f64,
}

impl Task {
    /// Length of every feature vector in this task.
    #[must_use]
    pub fn n_features(&self) -> usize {
        self.permutation.len()
    }
}

/// A sequence of tasks presented one after another, sharing one set of parameters.
///
/// The construction is **permuted inputs**, from Goodfellow, Mirza, Xiao, Courville and Bengio
/// (*An Empirical Investigation of Catastrophic Forgetting in Gradient-Based Neural Networks*,
/// `arXiv`:1312.6211, 2013) and used as the headline benchmark by Kirkpatrick et al. (2017). Every
/// task has the same classes and the same marginal statistics; only the assignment of channels to
/// feature slots changes. That isolates interference from every other difference between tasks,
/// which is what makes the forgetting number attributable.
#[derive(Debug, Clone, PartialEq)]
pub struct Curriculum {
    /// The tasks, in presentation order.
    pub tasks: Vec<Task>,
}

impl Curriculum {
    /// Build a curriculum directly from feature vectors somebody else produced.
    ///
    /// [`Curriculum::permuted`] is this with [`latency_features`] in front of it. It is public
    /// because the featureisation is where a permuted-input benchmark can be made easier or harder
    /// without changing anything that shows up in its name — see [`remove_sample_mean`] — and a
    /// reader who wants to check that claim needs to be able to build the other version.
    ///
    /// # Errors
    ///
    /// [`ContinualError::Empty`] for `n_tasks == 0`, an empty split or zero features;
    /// [`ContinualError::Mismatch`] for an example whose feature vector is the wrong length;
    /// [`ContinualError::Label`] for a label past `n_classes`; [`ContinualError::Exhausted`] as
    /// [`Curriculum::permuted`].
    pub fn from_examples(
        name: &str,
        train: Vec<Example>,
        test: Vec<Example>,
        n_classes: usize,
        n_tasks: usize,
        seed: u64,
    ) -> Result<Self, ContinualError> {
        if n_tasks == 0 {
            return Err(ContinualError::Empty { what: "n_tasks" });
        }
        if train.is_empty() {
            return Err(ContinualError::Empty { what: "train split" });
        }
        if test.is_empty() {
            return Err(ContinualError::Empty { what: "test split" });
        }
        let f = train[0].x.len();
        if f == 0 {
            return Err(ContinualError::Empty { what: "n_features" });
        }
        for e in train.iter().chain(&test) {
            if e.x.len() != f {
                return Err(ContinualError::Mismatch {
                    what: "feature vector",
                    expected: f,
                    found: e.x.len(),
                });
            }
            if e.label as usize >= n_classes {
                return Err(ContinualError::Label { label: e.label, n_classes });
            }
        }
        let perms = deranged_permutations(f, n_tasks, seed)?;
        let chance = 1.0 / n_classes as f64;
        let tasks = perms
            .into_iter()
            .enumerate()
            .map(|(k, p)| Task {
                name: format!("{name}/perm{k}"),
                train: apply_permutation(&train, &p),
                test: apply_permutation(&test, &p),
                permutation: p,
                n_classes,
                chance,
            })
            .collect();
        Ok(Self { tasks })
    }

    /// Build `n_tasks` permuted versions of `base`, the first being `base` itself.
    ///
    /// Features are [`latency_features`] **with each sample's own mean removed**; see
    /// [`remove_sample_mean`] for why, because it is the difference between a benchmark and a
    /// number. Permutations after the first are drawn to be **pairwise deranged**: no feature slot
    /// keeps the same channel across any two tasks. A shared slot is a slot whose weight does not
    /// have to change, which reduces exactly the interference the benchmark is measuring, so the
    /// generator refuses to produce one rather than reporting a milder collapse for an accidental
    /// reason.
    ///
    /// # Errors
    ///
    /// [`ContinualError::Empty`] for `n_tasks == 0` or an empty split;
    /// [`ContinualError::SilentChannel`] from [`latency_features`];
    /// [`ContinualError::Exhausted`] when rejection sampling cannot find that many pairwise-deranged
    /// permutations, which for fewer than two features is immediate because no derangement exists.
    pub fn permuted(
        base: &Dataset<Sample>,
        n_tasks: usize,
        seed: u64,
    ) -> Result<Self, ContinualError> {
        let feat = |split: Split| -> Result<Vec<Example>, ContinualError> {
            let mut out: Vec<Example> = base
                .split(split)
                .iter()
                .map(|s| {
                    let x = latency_features(s, base.n_inputs, base.ticks)?;
                    Ok(Example { x, label: s.label })
                })
                .collect::<Result<_, ContinualError>>()?;
            remove_sample_mean(&mut out);
            Ok(out)
        };
        Self::from_examples(
            base.name,
            feat(Split::Train)?,
            feat(Split::Test)?,
            base.n_classes as usize,
            n_tasks,
            seed,
        )
    }

    /// Number of tasks in the sequence.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tasks.len()
    }

    /// Whether the sequence holds no tasks at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tasks.is_empty()
    }
}

fn apply_permutation(examples: &[Example], p: &[usize]) -> Vec<Example> {
    examples
        .iter()
        .map(|e| Example { x: p.iter().map(|&k| e.x[k]).collect(), label: e.label })
        .collect()
}

fn deranged_permutations(
    f: usize,
    n: usize,
    seed: u64,
) -> Result<Vec<Vec<usize>>, ContinualError> {
    let identity: Vec<usize> = (0..f).collect();
    let mut out = vec![identity.clone()];
    if n == 1 {
        return Ok(out);
    }
    if f < 2 {
        return Err(ContinualError::Exhausted { wanted: n, found: 1, draws: 0 });
    }
    let mut rng = Rng::new(seed ^ 0xC047_1D0A_5EED_0001);
    let budget = 4096usize * n;
    let mut draws = 0usize;
    while out.len() < n {
        if draws >= budget {
            return Err(ContinualError::Exhausted { wanted: n, found: out.len(), draws });
        }
        draws += 1;
        let mut cand = identity.clone();
        shuffle(&mut cand, &mut rng);
        if out.iter().all(|q| cand.iter().zip(q).all(|(a, b)| a != b)) {
            out.push(cand);
        }
    }
    Ok(out)
}

/// Fisher-Yates, drawing from `Rng::below`, which is rejection-debiased.
fn shuffle<T>(v: &mut [T], rng: &mut Rng) {
    for i in (1..v.len()).rev() {
        let j = rng.below(i as u32 + 1) as usize;
        v.swap(i, j);
    }
}

// ---------------------------------------------------------------------------------------------
// The learner
// ---------------------------------------------------------------------------------------------

/// A linear softmax readout: `n_classes` units over `n_features` inputs, plus a bias each.
///
/// Trained by stochastic gradient descent on the cross-entropy, which for this model is convex, so
/// "the solution for task A" is a well-defined point and "task B moved the parameters away from it"
/// is a statement about a distance rather than about an optimisation path.
#[derive(Debug, Clone, PartialEq)]
pub struct Linear {
    /// Weights, row-major: class `c`'s weight on feature `k` is `w[c * n_features + k]`.
    pub w: Vec<f64>,
    /// Bias per class, one entry per output unit.
    pub b: Vec<f64>,
    n_features: usize,
    n_classes: usize,
}

impl Linear {
    /// A zero-initialised readout.
    ///
    /// Zeros rather than random draws, deliberately: a zero-initialised softmax gives every class
    /// the same logit, so the random-initialisation baseline `b_j` of Lopez-Paz & Ranzato is
    /// **exactly chance** on a balanced split and carries no seed. The gradient at zero is not
    /// zero, so nothing is stuck.
    ///
    /// # Errors
    ///
    /// [`ContinualError::Empty`] for zero features or fewer than two classes — a one-class
    /// classifier has no decision to make and its accuracy is 1.0 by construction.
    pub fn new(n_features: usize, n_classes: usize) -> Result<Self, ContinualError> {
        if n_features == 0 {
            return Err(ContinualError::Empty { what: "n_features" });
        }
        if n_classes < 2 {
            return Err(ContinualError::Empty { what: "n_classes (needs at least 2)" });
        }
        Ok(Self {
            w: vec![0.0; n_features * n_classes],
            b: vec![0.0; n_classes],
            n_features,
            n_classes,
        })
    }

    /// Input width.
    #[must_use]
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// Output width.
    #[must_use]
    pub fn n_classes(&self) -> usize {
        self.n_classes
    }

    /// Total learnable parameters, `n_classes * (n_features + 1)`.
    #[must_use]
    pub fn parameters(&self) -> usize {
        self.w.len() + self.b.len()
    }

    /// Check that `w` and `b` still have the shape this readout was built with.
    ///
    /// Both are public — an importance vector and a weight vector have to be the same shape, and
    /// hiding them behind accessors would not change that — so a caller can resize them. Every
    /// read path calls this first, because the alternative is a **panic** on a slice index, and a
    /// library that panics on a value its own API handed out is the defect this crate's tests are
    /// written to catch.
    ///
    /// # Errors
    ///
    /// [`ContinualError::Mismatch`] naming the array whose length is wrong.
    pub fn check_shape(&self) -> Result<(), ContinualError> {
        if self.w.len() != self.n_classes * self.n_features {
            return Err(ContinualError::Mismatch {
                what: "weight matrix",
                expected: self.n_classes * self.n_features,
                found: self.w.len(),
            });
        }
        if self.b.len() != self.n_classes {
            return Err(ContinualError::Mismatch {
                what: "bias vector",
                expected: self.n_classes,
                found: self.b.len(),
            });
        }
        Ok(())
    }

    /// Class posteriors under the softmax, summing to 1.
    ///
    /// Computed by subtracting the maximum logit before exponentiating, which is exact in the sense
    /// that matters: it changes no probability and it removes the overflow that a logit above ~710
    /// would otherwise produce.
    ///
    /// # Errors
    ///
    /// [`ContinualError::Mismatch`] when `x` is not `n_features` long, and
    /// [`ContinualError::NotFinite`] for a non-finite feature.
    pub fn probabilities(&self, x: &[f64]) -> Result<Vec<f64>, ContinualError> {
        let mut p = self.logits(x)?;
        let max = p.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut sum = 0.0;
        for v in &mut p {
            *v = (*v - max).exp();
            sum += *v;
        }
        for v in &mut p {
            *v /= sum;
        }
        Ok(p)
    }

    /// Unnormalised scores, one per class.
    ///
    /// # Errors
    ///
    /// [`ContinualError::Mismatch`] when `x` is not `n_features` long or when `w` or `b` has been
    /// resized (see [`Linear::check_shape`]), and [`ContinualError::NotFinite`] for a non-finite
    /// feature.
    pub fn logits(&self, x: &[f64]) -> Result<Vec<f64>, ContinualError> {
        self.check_shape()?;
        if x.len() != self.n_features {
            return Err(ContinualError::Mismatch {
                what: "feature vector",
                expected: self.n_features,
                found: x.len(),
            });
        }
        for &v in x {
            if !v.is_finite() {
                return Err(ContinualError::NotFinite { what: "feature", value: v });
            }
        }
        let mut z = self.b.clone();
        for c in 0..self.n_classes {
            let row = &self.w[c * self.n_features..(c + 1) * self.n_features];
            let mut acc = z[c];
            for k in 0..self.n_features {
                acc += row[k] * x[k];
            }
            z[c] = acc;
        }
        Ok(z)
    }

    /// Highest-scoring class. **Ties go to the lowest class index**, which is what makes the
    /// zero-initialised baseline deterministic rather than a coin flip per sample.
    ///
    /// # Errors
    ///
    /// As [`Linear::logits`].
    pub fn predict(&self, x: &[f64]) -> Result<usize, ContinualError> {
        let z = self.logits(x)?;
        let mut best = 0usize;
        for c in 1..z.len() {
            if z[c] > z[best] {
                best = c;
            }
        }
        Ok(best)
    }

    /// Fraction of `examples` classified correctly, computed by [`crate::metrics::accuracy`].
    ///
    /// # Errors
    ///
    /// [`ContinualError::Metric`] wrapping [`crate::metrics::MetricError::Empty`] on an empty set —
    /// the accuracy of nothing is not 1.0 — and as [`Linear::logits`] otherwise.
    pub fn accuracy(&self, examples: &[Example]) -> Result<f64, ContinualError> {
        let mut pred = Vec::with_capacity(examples.len());
        let mut truth = Vec::with_capacity(examples.len());
        for e in examples {
            pred.push(self.predict(&e.x)?);
            truth.push(e.label as usize);
        }
        Ok(metrics::accuracy(&pred, &truth)?)
    }

    /// Mean cross-entropy over `examples`, in nats.
    ///
    /// # Errors
    ///
    /// [`ContinualError::Empty`] on an empty set, [`ContinualError::Label`] for a label past the
    /// class count, and as [`Linear::logits`] otherwise.
    pub fn loss(&self, examples: &[Example]) -> Result<f64, ContinualError> {
        if examples.is_empty() {
            return Err(ContinualError::Empty { what: "examples" });
        }
        let mut acc = 0.0;
        for e in examples {
            let y = e.label as usize;
            if y >= self.n_classes {
                return Err(ContinualError::Label {
                    label: e.label,
                    n_classes: self.n_classes,
                });
            }
            let p = self.probabilities(&e.x)?;
            acc -= p[y].ln();
        }
        Ok(acc / examples.len() as f64)
    }
}

// ---------------------------------------------------------------------------------------------
// Consolidation: importance-weighted regularisation
// ---------------------------------------------------------------------------------------------

/// Which importance-weighted penalty is applied while training a later task.
///
/// Both members implement the same shape — a quadratic pull back toward the previous task's
/// parameters, weighted per parameter — and differ in how the weight is estimated. Each keeps its
/// own paper's constant convention, stated on the variant, so the code can be compared against the
/// source line by line.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Consolidation {
    /// No penalty at all. The control the collapse is measured on.
    None,
    /// Elastic weight consolidation: Kirkpatrick et al., *Overcoming catastrophic forgetting in
    /// neural networks*, PNAS 114(13):3521–3526 (2017).
    ///
    /// Penalty `lambda / 2 * sum_k F_k * (theta_k - theta*_k)^2`, with `F` the **diagonal empirical
    /// Fisher** accumulated at each task boundary and `theta*` the parameters at the end of the
    /// most recent task.
    ///
    /// Two honest deviations from the paper, both of which change numbers:
    /// - The paper's Fisher is an expectation over the model's own predictive distribution; this
    ///   uses the observed labels (the "empirical Fisher"), which is the common implementation and
    ///   is **not** the same estimator — see Martens, *New insights and perspectives on the natural
    ///   gradient method*, `arXiv`:1412.1193.
    /// - Importances are **summed** across tasks against a single anchor, which is the online form
    ///   of Schwarz et al. (ICML 2018), not the paper's separate penalty per task.
    Ewc {
        /// Penalty strength, dimensionless. Zero reproduces the plain learner exactly; a test in
        /// this module asserts that equality on every parameter.
        lambda: f64,
    },
    /// Synaptic intelligence: Zenke, Poole & Ganguli, *Continual Learning Through Synaptic
    /// Intelligence*, ICML 2017.
    ///
    /// Importance is the **path integral** of each parameter's contribution to the drop in the task
    /// loss, `omega_k = sum_t -g_k(t) * delta_theta_k(t)`, normalised at the task boundary by the
    /// squared total displacement plus a damping term. Penalty `c * sum_k Omega_k *
    /// (theta_k - theta*_k)^2`, which is the paper's convention, so its gradient carries a factor 2
    /// where [`Consolidation::Ewc`]'s does not.
    ///
    /// It is the cheaper of the two at the boundary — no extra pass over the data — and the more
    /// expensive during training, because it needs the unregularised gradient and two more vectors
    /// the width of the parameters.
    Si {
        /// Penalty strength, dimensionless; `c` in the paper.
        c: f64,
        /// Damping in the denominator, `xi` in the paper. Keeps a parameter that barely moved from
        /// acquiring an unbounded importance; the paper uses 0.1 and so does [`Protocol`].
        xi: f64,
    },
}

/// The diagonal **empirical Fisher** of a readout over a set of examples, flat: weights then biases.
///
/// `F_k = mean_n (d log p(y_n | x_n) / d theta_k)^2`, which is what [`Consolidation::Ewc`] weights
/// its penalty by. Public because "which parameters does this method think matter?" is the first
/// question anyone asks of elastic weight consolidation, and because it is checkable in closed
/// form: on a **zero-initialised two-class** readout every posterior is `0.5`, so every `d` is
/// `±0.5` and the Fisher is exactly `0.25 * mean(x_k^2)` for each weight and exactly `0.25` for
/// each bias. That identity is asserted in this module's tests.
///
/// # The failure mode this makes visible
///
/// **On a perfectly fitted, separable problem the empirical Fisher goes to zero**, because the
/// posteriors saturate and the gradients with them. Elastic weight consolidation then has nothing
/// to weight and silently becomes the plain learner however large `lambda` is. That is the entire
/// reason [`Schedule::weight_decay`] defaults to a non-zero value, and the test
/// `the_fisher_collapses_when_the_model_saturates` measures the collapse rather than asserting the
/// doctrine.
///
/// It is the **empirical** Fisher — the expectation is taken over the observed labels rather than
/// over the model's own predictive distribution, as Kirkpatrick et al. write it. The two differ,
/// and Martens (`arXiv`:1412.1193) is the reference for how.
///
/// # Errors
///
/// [`ContinualError::Empty`] for an empty example set, [`ContinualError::Label`] for a label past
/// the class count, and as [`Linear::logits`] for a bad feature vector.
pub fn fisher_diagonal(
    model: &Linear,
    examples: &[Example],
) -> Result<Vec<f64>, ContinualError> {
    if examples.is_empty() {
        return Err(ContinualError::Empty { what: "examples" });
    }
    model.check_shape()?;
    let (f, cn) = (model.n_features(), model.n_classes());
    let mut fisher = vec![0.0; model.parameters()];
    for e in examples {
        let y = e.label as usize;
        if y >= cn {
            return Err(ContinualError::Label { label: e.label, n_classes: cn });
        }
        let p = model.probabilities(&e.x)?;
        for c in 0..cn {
            let d = p[c] - if c == y { 1.0 } else { 0.0 };
            for k in 0..f {
                let gk = d * e.x[k];
                fisher[c * f + k] += gk * gk;
            }
            fisher[cn * f + c] += d * d;
        }
    }
    let inv = 1.0 / examples.len() as f64;
    for v in &mut fisher {
        *v *= inv;
    }
    Ok(fisher)
}

/// One parameter's synaptic-intelligence importance: `max(0, path / (displacement^2 + xi))`.
///
/// Zenke, Poole & Ganguli (ICML 2017), equation 5. `path` is that parameter's accumulated
/// contribution to the fall in the task loss, `sum_t -g_k(t) * delta_theta_k(t)`; `displacement` is
/// how far it moved over the whole task; `xi` damps the denominator so a parameter that barely
/// moved cannot acquire an unbounded importance.
///
/// **The clamp is not cosmetic.** The path integral comes out negative for a parameter whose steps
/// and gradients disagreed in sign — which happens whenever the loss went up locally — and a
/// negative importance is a penalty with the wrong sign, one that actively **pushes** the parameter
/// away from where the last task left it. The paper's own implementation clamps, and so does this.
///
/// `None` for a non-finite argument or a `xi` that is not positive, rather than an infinity.
#[must_use]
pub fn path_importance(path: f64, displacement: f64, xi: f64) -> Option<f64> {
    if !path.is_finite() || !displacement.is_finite() || !xi.is_finite() || xi <= 0.0 {
        return None;
    }
    Some((path / (displacement * displacement + xi)).max(0.0))
}

/// The state a [`Consolidation`] carries between tasks, and the bill for it.
#[derive(Debug, Clone, PartialEq)]
struct Consolidator {
    method: Consolidation,
    anchor: Vec<f64>,
    omega: Vec<f64>,
    path: Vec<f64>,
    start: Vec<f64>,
    n: usize,
    passes: u64,
}

impl Consolidator {
    fn new(method: Consolidation, n: usize) -> Self {
        let (omega, path, start) = match method {
            Consolidation::None => (Vec::new(), Vec::new(), Vec::new()),
            Consolidation::Ewc { .. } => (vec![0.0; n], Vec::new(), Vec::new()),
            Consolidation::Si { .. } => (vec![0.0; n], vec![0.0; n], vec![0.0; n]),
        };
        let anchor = match method {
            Consolidation::None => Vec::new(),
            _ => vec![0.0; n],
        };
        Self { method, anchor, omega, path, start, n, passes: 0 }
    }

    /// Floats held beyond the model's own parameters.
    fn extra_floats(&self) -> usize {
        self.anchor.len() + self.omega.len() + self.path.len() + self.start.len()
    }

    fn begin_task(&mut self, flat: &[f64]) {
        if let Consolidation::Si { .. } = self.method {
            self.start.copy_from_slice(flat);
            self.path.fill(0.0);
        }
    }

    /// Add the penalty's gradient to `g`, which already holds the data gradient.
    fn add_penalty(&self, flat: &[f64], g: &mut [f64]) {
        match self.method {
            Consolidation::None => {}
            Consolidation::Ewc { lambda } => {
                // d/dtheta of (lambda/2) * F * (theta - theta*)^2.
                for k in 0..self.n {
                    g[k] += lambda * self.omega[k] * (flat[k] - self.anchor[k]);
                }
            }
            Consolidation::Si { c, .. } => {
                // d/dtheta of c * Omega * (theta - theta*)^2, hence the 2.
                for k in 0..self.n {
                    g[k] += 2.0 * c * self.omega[k] * (flat[k] - self.anchor[k]);
                }
            }
        }
    }

    /// Fold one completed SGD step into the path integral.
    fn observe_step(&mut self, data_grad: &[f64], delta: &[f64]) {
        if let Consolidation::Si { .. } = self.method {
            for k in 0..self.n {
                self.path[k] -= data_grad[k] * delta[k];
            }
        }
    }

    fn end_task(&mut self, model: &Linear, train: &[Example]) -> Result<(), ContinualError> {
        match self.method {
            Consolidation::None => {}
            Consolidation::Ewc { .. } => {
                let fisher = fisher_diagonal(model, train)?;
                for k in 0..self.n {
                    self.omega[k] += fisher[k];
                }
                self.passes += train.len() as u64;
                flatten_into(model, &mut self.anchor);
            }
            Consolidation::Si { xi, .. } => {
                let mut flat = vec![0.0; self.n];
                flatten_into(model, &mut flat);
                for k in 0..self.n {
                    let d = flat[k] - self.start[k];
                    // `None` only for a non-finite path, which `sgd_step` has already refused, or a
                    // non-positive `xi`. Both mean there is no importance to add rather than an
                    // importance of zero, so the parameter keeps whatever it had.
                    if let Some(contrib) = path_importance(self.path[k], d, xi) {
                        self.omega[k] += contrib;
                    }
                }
                self.anchor.copy_from_slice(&flat);
            }
        }
        Ok(())
    }
}

fn flatten_into(model: &Linear, out: &mut [f64]) {
    out[..model.w.len()].copy_from_slice(&model.w);
    out[model.w.len()..].copy_from_slice(&model.b);
}

fn unflatten_from(flat: &[f64], model: &mut Linear) {
    let n = model.w.len();
    model.w.copy_from_slice(&flat[..n]);
    model.b.copy_from_slice(&flat[n..]);
}

// ---------------------------------------------------------------------------------------------
// Replay
// ---------------------------------------------------------------------------------------------

/// Rehearsal: what, if anything, is shown again from earlier tasks.
///
/// Rehearsal is the oldest answer to catastrophic forgetting (Robins, *Catastrophic Forgetting,
/// Rehearsal and Pseudorehearsal*, Connection Science 7:123, 1995) and still the strongest. The
/// question is never whether it works — it works — but what it costs, so both members carry an
/// explicit budget and [`Cost::extra_floats`] reports what that budget occupies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Replay {
    /// Nothing is kept and nothing is shown again.
    None,
    /// Experience replay: a fixed-size buffer of **real** examples, maintained by reservoir
    /// sampling (Vitter, *Random Sampling with a Reservoir*, ACM TOMS 11(1):37–57, 1985) so that
    /// after `n` examples have streamed past, every one of them is in the buffer with probability
    /// exactly `budget / n` regardless of when it arrived.
    Experience {
        /// Examples retained, across all tasks. A **reservation**: [`Cost::extra_floats`] charges
        /// `held * (n_features + 1)`, which is at most `budget * (n_features + 1)` and is less
        /// while fewer examples than that have gone past.
        budget: usize,
    },
    /// Generative replay: instead of examples, a **model** of each past task's classes, sampled to
    /// produce rehearsal data (Shin, Lee, Kim & Kim, *Continual Learning with Deep Generative
    /// Replay*, `NeurIPS` 2017).
    ///
    /// The generator here is the simplest one that is still a generator: a diagonal Gaussian per
    /// (task, class), fitted by moments. That is a deliberate floor, not a claim — a real system
    /// uses a variational autoencoder or a diffusion model, and its samples are better. What the
    /// floor buys is that the memory cost is exactly `2 * tasks * classes * features` floats and
    /// does not grow with how much rehearsal you do, which is the property the method is chosen
    /// for; and what it costs shows up as the gap between this row and experience replay in the
    /// retention table.
    Generative {
        /// Synthetic examples drawn per past task, per training run. Costs compute, not memory.
        per_task: usize,
    },
}

/// A standard normal draw by the Box-Muller transform, from a seeded [`Rng`].
///
/// Exposed because the generative replay path needs it and because a caller reproducing a figure
/// needs the same stream. `Rng::next_f64` returns `[0, 1)`, so `1 - u` lies in `(0, 1]` and its
/// logarithm is finite; the cosine branch is used and the sine branch discarded, which halves the
/// throughput and keeps the stream position a simple function of the number of draws.
#[must_use]
pub fn standard_normal(rng: &mut Rng) -> f64 {
    let u1 = 1.0 - rng.next_f64();
    let u2 = rng.next_f64();
    (-2.0 * u1.ln()).sqrt() * (core::f64::consts::TAU * u2).cos()
}

#[derive(Debug, Clone, PartialEq)]
struct ClassModel {
    label: u32,
    mean: Vec<f64>,
    sd: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
struct ReplayStore {
    policy: Replay,
    buffer: Vec<Example>,
    seen: u64,
    models: Vec<ClassModel>,
    n_features: usize,
}

impl ReplayStore {
    fn new(policy: Replay, n_features: usize) -> Self {
        Self { policy, buffer: Vec::new(), seen: 0, models: Vec::new(), n_features }
    }

    fn extra_floats(&self) -> usize {
        match self.policy {
            Replay::None => 0,
            // What is HELD, not what was reserved: a budget of 5000 over 2500 examples occupies
            // 2500 slots, and charging the reservation would report a memory cost nobody paid.
            // The bound `budget * (n_features + 1)` still holds, because the buffer never exceeds
            // its budget. The label is one more number per retained example and it is counted.
            Replay::Experience { .. } => self.buffer.len() * (self.n_features + 1),
            Replay::Generative { .. } => self.models.len() * 2 * self.n_features,
        }
    }

    /// Absorb a finished task's training set.
    fn absorb(&mut self, train: &[Example], n_classes: usize, rng: &mut Rng) {
        match self.policy {
            Replay::None => {}
            Replay::Experience { budget } => {
                for e in train {
                    self.seen += 1;
                    if self.buffer.len() < budget {
                        self.buffer.push(e.clone());
                    } else if budget > 0 {
                        // Vitter's Algorithm R: keep with probability budget/seen.
                        let j = (rng.next_f64() * self.seen as f64) as u64;
                        if j < budget as u64 {
                            self.buffer[j as usize] = e.clone();
                        }
                    }
                }
            }
            Replay::Generative { .. } => {
                for c in 0..n_classes {
                    let mut mean = vec![0.0; self.n_features];
                    let mut sq = vec![0.0; self.n_features];
                    let mut n = 0u64;
                    for e in train.iter().filter(|e| e.label as usize == c) {
                        n += 1;
                        for k in 0..self.n_features {
                            mean[k] += e.x[k];
                            sq[k] += e.x[k] * e.x[k];
                        }
                    }
                    if n == 0 {
                        continue;
                    }
                    let inv = 1.0 / n as f64;
                    let mut sd = vec![0.0; self.n_features];
                    for k in 0..self.n_features {
                        mean[k] *= inv;
                        // Population variance; clamped at zero because a class whose feature is
                        // constant produces a tiny negative from cancellation, and `sqrt` of that
                        // is `NaN` in every sample it would then generate.
                        sd[k] = (sq[k] * inv - mean[k] * mean[k]).max(0.0).sqrt();
                    }
                    self.models.push(ClassModel { label: c as u32, mean, sd });
                }
            }
        }
    }

    /// What to interleave with the current task.
    fn rehearsal(&self, rng: &mut Rng) -> Vec<Example> {
        match self.policy {
            Replay::None => Vec::new(),
            Replay::Experience { .. } => self.buffer.clone(),
            Replay::Generative { per_task } => {
                let mut out = Vec::with_capacity(self.models.len() * per_task);
                for m in &self.models {
                    for _ in 0..per_task {
                        let x = (0..self.n_features)
                            .map(|k| m.mean[k] + m.sd[k] * standard_normal(rng))
                            .collect();
                        out.push(Example { x, label: m.label });
                    }
                }
                out
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Homeostasis
// ---------------------------------------------------------------------------------------------

/// Multiplicative synaptic scaling applied to a readout's weights, in the continual-learning role.
///
/// The biological mechanism is Turrigiano, Leslie, Desai, Rutherford and Nelson (*Activity-dependent
/// scaling of quantal amplitude in neocortical neurons*, Nature 391:892–896, 1998): a cell that
/// drifts from its target rate multiplies **all** of its afferents by a common factor, which changes
/// its gain without changing which pattern it prefers. [`crate::plasticity::SynapticScaling`] is
/// that mechanism on the spiking side of this crate, driven by a firing-rate estimate; it proves
/// the property that makes the multiplicative form the right one, namely that scaling preserves
/// every pairwise weight ratio exactly.
///
/// Here the drive is a **norm** rather than a rate, because a linear readout has no firing rate to
/// estimate. Each class row, bias included, is multiplied so that its Euclidean norm moves toward
/// `target_norm`.
///
/// # The honest finding
///
/// **It does not rescue task A.** Forgetting is a change of *direction* in weight space and scaling
/// changes only *magnitude*, so a homeostat cannot undo it in principle, and the measurement in
/// this module's tests agrees: on the five-task curriculum it moves backward transfer from −0.763
/// to −0.800, a change of 0.037, in the **wrong** direction. It is included because it is what neuromorphic hardware actually implements,
/// and because the result — a real mechanism that costs almost nothing and buys almost nothing here
/// — is the kind of thing a benchmark exists to find out.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Homeostat {
    /// Euclidean norm each class row is scaled toward, including its bias. Dimensionless, in the
    /// same units as the weights.
    pub target_norm: f64,
    /// Fraction of the way to the target one application moves, in `0.0..=1.0`. The factor applied
    /// is `(target / norm)^rate`, so `1.0` lands on the target exactly and `0.0` does nothing.
    pub rate: f64,
}

impl Homeostat {
    /// Scale every class row toward the target norm, returning the factor applied to each.
    ///
    /// A row whose norm is exactly zero is left alone and reports a factor of `1.0`: there is no
    /// direction to preserve, and any factor would be arbitrary.
    ///
    /// # Errors
    ///
    /// [`ContinualError::OutOfRange`] for a non-positive or non-finite `target_norm`, or a `rate`
    /// outside `0.0..=1.0`; [`ContinualError::Mismatch`] if the model's arrays have been resized.
    pub fn apply(&self, model: &mut Linear) -> Result<Vec<f64>, ContinualError> {
        let target = in_range("target_norm", self.target_norm, f64::MIN_POSITIVE, f64::MAX)?;
        let rate = in_range("rate", self.rate, 0.0, 1.0)?;
        model.check_shape()?;
        let f = model.n_features();
        let mut factors = Vec::with_capacity(model.n_classes());
        for c in 0..model.n_classes() {
            let row = &model.w[c * f..(c + 1) * f];
            let norm = (row.iter().map(|v| v * v).sum::<f64>() + model.b[c] * model.b[c]).sqrt();
            if norm == 0.0 {
                factors.push(1.0);
                continue;
            }
            let g = (target / norm).powf(rate);
            for k in 0..f {
                model.w[c * f + k] *= g;
            }
            model.b[c] *= g;
            factors.push(g);
        }
        Ok(factors)
    }
}

// ---------------------------------------------------------------------------------------------
// Protocol, cost and results
// ---------------------------------------------------------------------------------------------

/// The optimiser settings a task is trained under.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Schedule {
    /// Passes over the task's training set, at least 1.
    pub epochs: usize,
    /// Learning rate, dimensionless, positive.
    pub lr: f64,
    /// Coefficient of the `L2` penalty on the weights, dimensionless and non-negative.
    ///
    /// Not cosmetic here, and the size of the effect is measured rather than asserted. Without it
    /// the cross-entropy solution on separable data runs off to infinity, the posteriors saturate
    /// and the **empirical Fisher goes to zero** — at which point [`Consolidation::Ewc`] has
    /// nothing to weight and silently becomes the plain learner however large `lambda` is. On this
    /// module's default task, forty epochs at `1e-3` leave a total [`fisher_diagonal`] of 0.025
    /// against 0.00042 at zero decay, a factor of sixty. A bounded optimum is what makes an
    /// importance estimate mean anything.
    ///
    /// It also erases task A on its own, which is a different thing from forgetting it and is
    /// separated from it in this module's opening table. Set it to zero to see interference alone.
    pub weight_decay: f64,
    /// Seed for the example ordering and for generative replay. Same seed, same run, every
    /// platform.
    pub seed: u64,
}

impl Default for Schedule {
    /// Twelve epochs at a learning rate of 0.2 with `1e-3` weight decay. These are round numbers
    /// chosen so that the default curriculum's first task is learned to better than 0.99 without
    /// the posteriors saturating, not a tuned optimum; a different feature set will want different
    /// ones.
    fn default() -> Self {
        Self { epochs: 12, lr: 0.2, weight_decay: 1e-3, seed: 0xC047_0001 }
    }
}

impl Schedule {
    /// Reject a schedule that cannot produce a run.
    ///
    /// # Errors
    ///
    /// [`ContinualError::Empty`] for zero epochs, [`ContinualError::OutOfRange`] for a `lr` that is
    /// not positive and finite or a negative or non-finite `weight_decay`.
    pub fn validate(&self) -> Result<(), ContinualError> {
        if self.epochs == 0 {
            return Err(ContinualError::Empty { what: "epochs" });
        }
        in_range("lr", self.lr, f64::MIN_POSITIVE, f64::MAX)?;
        in_range("weight_decay", self.weight_decay, 0.0, f64::MAX)?;
        Ok(())
    }
}

/// One complete recipe: how to train, what to consolidate, what to rehearse, what to rescale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Protocol {
    /// Optimiser settings, shared by every task in the sequence.
    pub schedule: Schedule,
    /// The importance-weighted penalty, if any.
    pub consolidation: Consolidation,
    /// The rehearsal policy, if any.
    pub replay: Replay,
    /// A homeostat applied at each task boundary, if any.
    pub homeostasis: Option<Homeostat>,
}

impl Protocol {
    /// The control: train, and do nothing else. This is the learner whose collapse everything else
    /// is measured against.
    #[must_use]
    pub fn plain(schedule: Schedule) -> Self {
        Self {
            schedule,
            consolidation: Consolidation::None,
            replay: Replay::None,
            homeostasis: None,
        }
    }
}

/// What a protocol spent, reported beside what it bought.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Cost {
    /// Parameters the model itself holds, `n_classes * (n_features + 1)`.
    pub model_parameters: usize,
    /// Floats of extra state held beyond those parameters: consolidation vectors plus whatever the
    /// replay policy retains. Zero for [`Protocol::plain`].
    pub extra_floats: usize,
    /// Gradient steps over the whole curriculum, replayed examples included.
    pub gradient_steps: u64,
    /// How many of those steps were on rehearsal rather than on current-task data.
    pub replayed_steps: u64,
    /// Forward and backward passes spent estimating importance at task boundaries.
    /// [`Consolidation::Si`] needs none, which is the trade it makes against
    /// [`Consolidation::Ewc`].
    pub importance_passes: u64,
}

impl Cost {
    /// Extra state in bytes, at 8 bytes per `f64`.
    #[must_use]
    pub fn extra_bytes(&self) -> usize {
        self.extra_floats * 8
    }
}

/// The accuracy matrix of a finished run, and the metrics read off it.
///
/// `r[i][j]` is the accuracy on task `j`'s test split after training has finished on task `i`. This
/// is the matrix `R` of Lopez-Paz & Ranzato (`NeurIPS` 2017); every metric below is their
/// definition, with their indices shifted to zero-based.
#[derive(Debug, Clone, PartialEq)]
pub struct Results {
    /// The accuracy matrix, `tasks` by `tasks`.
    pub r: Vec<Vec<f64>>,
    /// Accuracy of the untrained model on each task, `b_j` in the source. With a zero-initialised
    /// softmax and a balanced split this is exactly chance.
    pub baseline: Vec<f64>,
    /// What the run spent.
    pub cost: Cost,
}

impl Results {
    /// Number of tasks in the sequence.
    #[must_use]
    pub fn tasks(&self) -> usize {
        self.r.len()
    }

    /// `ACC`: mean accuracy over every task after the last one has been trained.
    ///
    /// `None` for an empty run.
    #[must_use]
    pub fn average_accuracy(&self) -> Option<f64> {
        let last = self.r.last()?;
        if last.is_empty() {
            return None;
        }
        Some(last.iter().sum::<f64>() / last.len() as f64)
    }

    /// `BWT`: mean change in each earlier task's accuracy between the moment it was learned and the
    /// end of the sequence. Negative is forgetting.
    ///
    /// `None` for fewer than two tasks. The mean of an empty set of differences is **not** zero, and
    /// returning zero there would report "no forgetting" for a run in which nothing could have been
    /// forgotten — the one case where the number is guaranteed to look good. Also `None` for a
    /// ragged matrix, which [`run`] cannot produce but a caller filling [`Results::r`] by hand can;
    /// a refusal is better than the slice index that used to be there.
    #[must_use]
    pub fn backward_transfer(&self) -> Option<f64> {
        let t = self.r.len();
        if t < 2 {
            return None;
        }
        let last = self.r.last()?;
        let mut acc = 0.0;
        for j in 0..t - 1 {
            acc += last.get(j)? - self.r[j].get(j)?;
        }
        Some(acc / (t - 1) as f64)
    }

    /// `FWT`: mean accuracy on each task *before* it was trained, above the untrained baseline.
    /// Positive means earlier tasks helped.
    ///
    /// `None` for fewer than two tasks or for a short row or baseline, for the same reasons as
    /// [`Results::backward_transfer`].
    #[must_use]
    pub fn forward_transfer(&self) -> Option<f64> {
        let t = self.r.len();
        if t < 2 {
            return None;
        }
        let mut acc = 0.0;
        for j in 1..t {
            acc += self.r[j - 1].get(j)? - self.baseline.get(j)?;
        }
        Some(acc / (t - 1) as f64)
    }

    /// Accuracy on task `j` at the moment it finished training, `r[j][j]`.
    #[must_use]
    pub fn learned(&self, j: usize) -> Option<f64> {
        self.r.get(j)?.get(j).copied()
    }

    /// Accuracy on task `j` at the end of the whole sequence, `r[last][j]`.
    #[must_use]
    pub fn retained(&self, j: usize) -> Option<f64> {
        self.r.last()?.get(j).copied()
    }

    /// How much of task `j` was lost: `learned - retained`, positive when something was forgotten
    /// (Chaudhry, Dokania, Ajanthan & Torr, ECCV 2018).
    #[must_use]
    pub fn forgetting(&self, j: usize) -> Option<f64> {
        Some(self.learned(j)? - self.retained(j)?)
    }
}

/// Accuracy of one model on every task of a curriculum: one row of the accuracy matrix.
///
/// Public so that a caller running their own training loop can still read the standard metrics off
/// it — [`Results`] has public fields for the same reason. `run` uses this for every row it
/// records.
///
/// # Errors
///
/// [`ContinualError::Mismatch`] when a task's feature width does not match the model's, and
/// [`ContinualError::Metric`] for an empty test split.
pub fn evaluate(model: &Linear, curriculum: &Curriculum) -> Result<Vec<f64>, ContinualError> {
    curriculum.tasks.iter().map(|t| model.accuracy(&t.test)).collect()
}

/// Train `model` on one task with **no** consolidation and no rehearsal, returning the gradient
/// steps taken.
///
/// The plain learner, exposed on its own so a caller can drive a sequence by hand — and so that
/// "a learner that was trained on task 0 and then simply not trained again" is a thing this module
/// can construct, which is what makes its backward transfer an exact zero rather than a small one.
/// [`run`] uses the same step function, so this is not a second implementation.
///
/// # Errors
///
/// As [`Schedule::validate`], plus [`ContinualError::Label`] for a label past the class count,
/// [`ContinualError::Mismatch`] for a feature width that is not the model's, and
/// [`ContinualError::Diverged`] if a parameter leaves the finite numbers.
pub fn train(model: &mut Linear, task: &Task, schedule: &Schedule) -> Result<u64, ContinualError> {
    schedule.validate()?;
    if task.train.is_empty() {
        return Err(ContinualError::Empty { what: "task train split" });
    }
    let n = model.parameters();
    let consolidator = Consolidator::new(Consolidation::None, n);
    let mut rng = Rng::new(schedule.seed);
    let (mut flat, mut data_grad) = (vec![0.0; n], vec![0.0; n]);
    let (mut grad, mut delta) = (vec![0.0; n], vec![0.0; n]);
    let mut order: Vec<usize> = (0..task.train.len()).collect();
    let mut steps = 0u64;
    for _ in 0..schedule.epochs {
        shuffle(&mut order, &mut rng);
        for &i in &order {
            sgd_step(
                model,
                &task.train[i],
                schedule,
                &consolidator,
                &mut flat,
                &mut data_grad,
                &mut grad,
                &mut delta,
            )?;
            steps += 1;
        }
    }
    Ok(steps)
}

/// Train one learner through the whole curriculum, evaluating every task after every task.
///
/// # Errors
///
/// [`ContinualError::Empty`] for an empty curriculum or an empty split;
/// [`ContinualError::Mismatch`] when two tasks disagree about feature or class count;
/// [`ContinualError::OutOfRange`] from [`Schedule::validate`] or [`Homeostat::apply`];
/// [`ContinualError::Label`] for a label past the class count; and [`ContinualError::Diverged`]
/// when a consolidation strength is too large for the learning rate, which is refused rather than
/// reported as a chance-accuracy result.
pub fn run(curriculum: &Curriculum, protocol: &Protocol) -> Result<Results, ContinualError> {
    protocol.schedule.validate()?;
    let tasks = &curriculum.tasks;
    if tasks.is_empty() {
        return Err(ContinualError::Empty { what: "curriculum" });
    }
    let nf = tasks[0].n_features();
    let nc = tasks[0].n_classes;
    for t in tasks {
        if t.n_features() != nf {
            return Err(ContinualError::Mismatch {
                what: "task feature width",
                expected: nf,
                found: t.n_features(),
            });
        }
        if t.n_classes != nc {
            return Err(ContinualError::Mismatch {
                what: "task class count",
                expected: nc,
                found: t.n_classes,
            });
        }
        if t.train.is_empty() {
            return Err(ContinualError::Empty { what: "task train split" });
        }
        if t.test.is_empty() {
            return Err(ContinualError::Empty { what: "task test split" });
        }
    }

    let mut model = Linear::new(nf, nc)?;
    let n = model.parameters();
    let baseline: Vec<f64> = evaluate(&model, curriculum)?;

    let mut consolidator = Consolidator::new(protocol.consolidation, n);
    let mut store = ReplayStore::new(protocol.replay, nf);
    let mut rng = Rng::new(protocol.schedule.seed);

    let mut flat = vec![0.0; n];
    let mut data_grad = vec![0.0; n];
    let mut grad = vec![0.0; n];
    let mut delta = vec![0.0; n];
    let mut cost = Cost { model_parameters: n, ..Cost::default() };
    let mut r: Vec<Vec<f64>> = Vec::with_capacity(tasks.len());

    for task in tasks {
        flatten_into(&model, &mut flat);
        consolidator.begin_task(&flat);

        let rehearsal = store.rehearsal(&mut rng);
        let mut order: Vec<(bool, usize)> = (0..task.train.len())
            .map(|i| (false, i))
            .chain((0..rehearsal.len()).map(|i| (true, i)))
            .collect();

        for _ in 0..protocol.schedule.epochs {
            shuffle(&mut order, &mut rng);
            for &(replayed, i) in &order {
                let e = if replayed { &rehearsal[i] } else { &task.train[i] };
                sgd_step(
                    &mut model,
                    e,
                    &protocol.schedule,
                    &consolidator,
                    &mut flat,
                    &mut data_grad,
                    &mut grad,
                    &mut delta,
                )?;
                consolidator.observe_step(&data_grad, &delta);
                cost.gradient_steps += 1;
                if replayed {
                    cost.replayed_steps += 1;
                }
            }
        }

        if let Some(h) = protocol.homeostasis {
            h.apply(&mut model)?;
        }
        consolidator.end_task(&model, &task.train)?;
        store.absorb(&task.train, nc, &mut rng);

        r.push(evaluate(&model, curriculum)?);
    }

    cost.extra_floats = consolidator.extra_floats() + store.extra_floats();
    cost.importance_passes = consolidator.passes;
    Ok(Results { r, baseline, cost })
}

fn sgd_step(
    model: &mut Linear,
    e: &Example,
    schedule: &Schedule,
    consolidator: &Consolidator,
    flat: &mut [f64],
    data_grad: &mut [f64],
    grad: &mut [f64],
    delta: &mut [f64],
) -> Result<(), ContinualError> {
    let f = model.n_features();
    let cn = model.n_classes();
    let y = e.label as usize;
    if y >= cn {
        return Err(ContinualError::Label { label: e.label, n_classes: cn });
    }
    let p = model.probabilities(&e.x)?;
    // The task objective's gradient, weight decay included. The consolidation penalty is NOT
    // included: synaptic intelligence integrates the task loss along the path, and folding its own
    // penalty back in would let it certify its own importance estimate.
    for c in 0..cn {
        let d = p[c] - if c == y { 1.0 } else { 0.0 };
        for k in 0..f {
            data_grad[c * f + k] = d * e.x[k] + schedule.weight_decay * model.w[c * f + k];
        }
        data_grad[cn * f + c] = d;
    }
    flatten_into(model, flat);
    grad.copy_from_slice(data_grad);
    consolidator.add_penalty(flat, grad);
    for k in 0..flat.len() {
        delta[k] = -schedule.lr * grad[k];
        flat[k] += delta[k];
        if !flat[k].is_finite() {
            return Err(ContinualError::Diverged { parameter: k, value: flat[k] });
        }
    }
    unflatten_from(flat, model);
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// The cascade synapse
// ---------------------------------------------------------------------------------------------

/// A cascade synapse: one bit of weight, `depth` levels of metaplastic state.
///
/// Fusi, Drew and Abbott, *Cascade Models of Synaptically Stored Memories*, Neuron 45:599–611
/// (2005). The problem they set out is the one this whole module is about, stated for a single
/// synapse: a binary synapse that is always plastic forgets exponentially, and one that is rarely
/// plastic learns nothing. The cascade's answer is to give the synapse a **depth**. Its weight is
/// still one bit — `+` or `−`, which is what a memristive or a single-bit `SRAM` synapse actually
/// stores — but behind that bit sits a level `0..depth`. A stimulus of the opposite sign flips the
/// bit with a probability that **falls geometrically with level**, and a stimulus of the same sign
/// pushes the synapse one level deeper.
///
/// A fresh synapse is shallow and learns instantly. A synapse that has seen the same evidence
/// repeatedly is deep and ignores contradiction. That is the whole model, and its whole claim is a
/// **trade**: [`Cascade::learning_probability`] falls with depth while [`Cascade::lifetime`] rises.
/// Both sides are measured here, because a claim with only the favourable side measured is an
/// advertisement.
///
/// # Why this belongs in a neuromorphic library rather than in a note
///
/// Every mitigation above this line stores extra **floats** per parameter. The cascade stores extra
/// **bits**, and `log2(depth)` of them: depth 8 costs three bits on top of the weight bit, against
/// the 64 that one `f64` importance value costs. On hardware whose synapse is a single device, that
/// is the difference between a mechanism you can build and one you cannot.
///
/// # What is verified, and what is transcribed
///
/// The exact transition probabilities in the paper are stated for its own figures and this
/// implementation uses a two-parameter geometric family, `flip_k = q0 * x^k` and
/// `deepen_k = p0 * x^k`, with `x` the same ratio for both. **This is a simplification and is
/// stated as one.** What is verified is not the paper's figure but the internal mathematics: the
/// exact Markov chain in [`Cascade::signal_after`] reduces at `depth == 1` to `(1 - q0)^t` in closed
/// form, and the sampled ensemble in [`CascadeEnsemble`] reproduces the chain.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cascade {
    /// Number of metaplastic levels per polarity, at least 1. Depth 1 is an ordinary stochastic
    /// binary synapse.
    pub depth: usize,
    /// Probability that a level-0 synapse flips its bit under an opposing stimulus, in
    /// `0.0..=1.0`.
    pub q0: f64,
    /// Probability that a level-0 synapse moves one level deeper under a reinforcing stimulus, in
    /// `0.0..=1.0`.
    pub p0: f64,
    /// Geometric ratio between consecutive levels, in `0.0..=1.0`. Level `k`'s probabilities are
    /// `q0 * x^k` and `p0 * x^k`; `x == 1.0` removes the cascade and makes every level identical.
    pub x: f64,
}

/// TEST-ONLY independent oracle: solve `π P = π`, `Σπ = 1` by Gaussian elimination on the exact
/// generator `g = P − I`, verified against `m = P`. Kept so the flux-balance recursion that
/// replaced it can be checked against a second algorithm rather than against itself; it is not
/// used at runtime because it loses precision past depth ~48 (see `stationary_by_flux_balance`).
///
/// Takes `A = Gᵀ`, replaces the last row with ones (the normalisation) and solves `A π = e_n`
/// by Gaussian elimination with partial pivoting. For an irreducible chain `Pᵀ − I` has rank
/// `n − 1` and every row is minus the sum of the others, so replacing any one row with the
/// normalisation gives a nonsingular system; the last row is chosen for no deeper reason.
///
/// Returns `None` — rather than a number — when the system is singular, which is what a reducible
/// chain produces (an absorbing state is a structurally zero column), or when the solution fails
/// its own check: every entry non-negative to roundoff and the stationarity residual
/// `‖πP − π‖₁` below `1e-9`. The second guard is the one that matters: a solve that returned a
/// non-stationary vector for any reason at all is caught here regardless of why.
/// Solve `A y = rhs` for `A = Gᵀ`, `G = P − I` supplied exactly, with its last row replaced by ones, by Gauss-Jordan with
/// partial pivoting. `None` on an exactly-zero pivot, which is the structurally singular
/// (reducible) case. Factored out of [`stationary_direct`] so that iterative refinement can call it
/// again on a residual right-hand side.
#[cfg(test)]
fn solve_normalised(g: &[f64], n: usize, rhs: &[f64]) -> Option<Vec<f64>> {
    let w = n + 1;
    let mut a = vec![0.0f64; n * w];
    for i in 0..n {
        for j in 0..n {
            // A = Gᵀ where G = P − I is already exact; no subtraction from 1 happens here.
            a[i * w + j] = g[j * n + i];
        }
        a[i * w + n] = rhs[i];
    }
    for j in 0..n {
        a[(n - 1) * w + j] = 1.0;
    }
    a[(n - 1) * w + n] = rhs[n - 1];

    // A structurally zero column has NO nonzero candidate and yields an exactly-zero pivot, which
    // is the reducible case; the threshold sits far below any genuine entry (the deepest fusi
    // transition at depth 60 is ~1e-18) so it can only fire on that structural zero.
    for c in 0..n {
        let mut p = c;
        for r in c + 1..n {
            if a[r * w + c].abs() > a[p * w + c].abs() {
                p = r;
            }
        }
        if a[p * w + c].abs() < 1e-200 {
            return None;
        }
        if p != c {
            for k in 0..w {
                a.swap(c * w + k, p * w + k);
            }
        }
        let piv = a[c * w + c];
        for k in 0..w {
            a[c * w + k] /= piv;
        }
        for r in 0..n {
            if r != c {
                let f = a[r * w + c];
                if f != 0.0 {
                    for k in 0..w {
                        a[r * w + k] -= f * a[c * w + k];
                    }
                }
            }
        }
    }
    Some((0..n).map(|i| a[i * w + n]).collect())
}

/// `‖πP − π‖₁`, the stationarity residual of a candidate distribution.
fn stationarity_residual(m: &[f64], n: usize, pi: &[f64]) -> f64 {
    let mut residual = 0.0;
    for j in 0..n {
        let mut acc = 0.0;
        for i in 0..n {
            acc += pi[i] * m[i * n + j];
        }
        residual += (acc - pi[j]).abs();
    }
    residual
}

/// Solve `π P = π`, `Σπ = 1` directly for a row-stochastic `m` (row-major, `n × n`).
///
/// Builds `A = Pᵀ − I`, replaces the last row with ones (the normalisation) and solves `A π = e_n`.
/// For an irreducible chain `Pᵀ − I` has rank `n − 1` and every row is minus the sum of the
/// others, so replacing any one row with the normalisation gives a nonsingular system; the last
/// row is chosen for no deeper reason.
///
/// # Iterative refinement, and why
///
/// One elimination gives `π` to roundoff **amplified by the matrix's conditioning**, and the fusi
/// chain's entries span from `1` down to `2^−(d−1)`. At depth 23 a single solve left `‖πP − π‖₁`
/// near `1e-12`, which is invisible in `learning_probability` (a sum of large entries) but shows
/// up in `signal_after(0)` — that is `mass₊ − mass₋`, two halves near `0.5` cancelling to `0.083`,
/// which multiplies the absolute error by about six. So the solution is polished: compute the
/// residual `r = e_n − Aπ`, solve `Aδ = r`, add `δ`, and repeat while the stationarity residual
/// keeps falling. Two rounds are typically enough to reach the `1e-15` the old iteration reached
/// where it worked, at every depth where it did not.
///
/// Returns `None` — rather than a number — when the system is singular, which is what a reducible
/// chain produces, or when the solution fails its own check: every entry non-negative to roundoff
/// and the stationarity residual below `1e-9`. The second guard is the one that matters: a solve
/// that returned a non-stationary vector for any reason at all is caught here regardless of why.
#[cfg(test)]
fn stationary_direct(g: &[f64], m: &[f64], n: usize) -> Option<Vec<f64>> {
    if n == 0 {
        return None;
    }
    let mut rhs = vec![0.0f64; n];
    rhs[n - 1] = 1.0;
    let mut pi = solve_normalised(g, n, &rhs)?;
    let mut best = stationarity_residual(m, n, &pi);

    // Refine: residual of the NORMALISED system, i.e. rows 0..n-1 of Gᵀπ and Σπ − 1 last.
    for _ in 0..4 {
        let mut r = vec![0.0f64; n];
        for i in 0..n - 1 {
            let mut acc = 0.0;
            for j in 0..n {
                acc += g[j * n + i] * pi[j];
            }
            r[i] = -acc;
        }
        r[n - 1] = 1.0 - pi.iter().sum::<f64>();
        let Some(delta) = solve_normalised(g, n, &r) else { break };
        let cand: Vec<f64> = pi.iter().zip(&delta).map(|(a, b)| a + b).collect();
        let res = stationarity_residual(m, n, &cand);
        if res < best {
            pi = cand;
            best = res;
        } else {
            break;
        }
    }

    // ⛔ PROJECT ONTO THE SYMMETRIC SUBSPACE. `transition_matrix` gives the two stimulus signs
    // equal probability, so the chain is invariant under swapping polarity and its stationary
    // distribution is EXACTLY polarity-symmetric: `π[k] == π[d + k]`. The slowest eigenmode — the
    // deepest level's flip — is antisymmetric, and it is precisely along that direction that f64
    // cannot resolve stationarity: at depth 23 the spectral gap is 2.4e-7, so an error of 1e-9
    // along it changes the residual by less than machine epsilon and refinement converged to an
    // exactly-stationary vector with a 2.9e-11 asymmetry. Averaging the two halves removes the
    // ill-conditioned component by construction rather than by tolerance.
    let d = n / 2;
    if n.is_multiple_of(2) {
        for k in 0..d {
            let avg = 0.5 * (pi[k] + pi[d + k]);
            pi[k] = avg;
            pi[d + k] = avg;
        }
    }

    // Verify before trusting: non-negative to roundoff, and actually stationary.
    if pi.iter().any(|&x| !x.is_finite() || x < -1e-9) {
        return None;
    }
    for x in &mut pi {
        if *x < 0.0 {
            *x = 0.0;
        }
    }
    let sum: f64 = pi.iter().sum();
    if !(0.5..1.5).contains(&sum) {
        return None;
    }
    for x in &mut pi {
        *x /= sum;
    }
    if stationarity_residual(m, n, &pi) > 1e-9 {
        return None;
    }
    Some(pi)
}

impl Cascade {
    /// The parameterisation this module uses for its figures: `q0 = p0 = 1.0`, `x = 0.5`.
    ///
    /// The halving ratio is the paper's; the unit level-0 probabilities are this implementation's
    /// choice, so that a fresh synapse is maximally plastic and depth is the only thing that slows
    /// it down. **Whether these reproduce any particular figure in the paper has not been checked
    /// here**, and the tests check the model's internal mathematics instead.
    #[must_use]
    pub fn fusi_2005(depth: usize) -> Self {
        Self { depth, q0: 1.0, p0: 1.0, x: 0.5 }
    }

    /// Reject a parameterisation that is not a probability.
    ///
    /// # Errors
    ///
    /// [`ContinualError::Empty`] for `depth == 0`, [`ContinualError::OutOfRange`] for a `q0`, `p0`
    /// or `x` outside `0.0..=1.0` or not finite.
    pub fn validate(&self) -> Result<(), ContinualError> {
        if self.depth == 0 {
            return Err(ContinualError::Empty { what: "depth" });
        }
        in_range("q0", self.q0, 0.0, 1.0)?;
        in_range("p0", self.p0, 0.0, 1.0)?;
        in_range("x", self.x, 0.0, 1.0)?;
        Ok(())
    }

    /// Probability that a synapse at `level` flips its bit under an opposing stimulus.
    ///
    /// `None` for a level at or past [`Cascade::depth`].
    #[must_use]
    pub fn flip_probability(&self, level: usize) -> Option<f64> {
        (level < self.depth).then(|| self.q0 * self.x.powi(level as i32))
    }

    /// Probability that a synapse at `level` moves one level deeper under a reinforcing stimulus.
    ///
    /// `None` for a level at or past [`Cascade::depth`]; the deepest level returns `Some(0.0)`
    /// because there is nowhere deeper to go, which is a different statement from "not a level".
    #[must_use]
    pub fn deepen_probability(&self, level: usize) -> Option<f64> {
        if level >= self.depth {
            return None;
        }
        if level + 1 == self.depth {
            return Some(0.0);
        }
        Some(self.p0 * self.x.powi(level as i32))
    }

    /// The `2 * depth` by `2 * depth` transition matrix under a stimulus of random sign.
    ///
    /// State `k < depth` is polarity `+` at level `k`; state `depth + k` is polarity `−` at level
    /// `k`. Each stimulus is potentiating or depressing with probability one half, which is the
    /// unstructured-background assumption the paper's memory curves are drawn under.
    ///
    /// # Errors
    ///
    /// As [`Cascade::validate`].
    pub fn transition_matrix(&self) -> Result<Vec<f64>, ContinualError> {
        self.validate()?;
        let d = self.depth;
        let n = 2 * d;
        let mut t = vec![0.0; n * n];
        // ⛔ A RATE THIS MATRIX CANNOT HOLD IS REFUSED, NOT ROUNDED AWAY. Each row's diagonal is
        // `0.5·(1 − flip) + 0.5·(1 − deepen)`, and `1 − rate` is computed first, just below 1.0
        // where the spacing is 2^−53. Once a nonzero rate is at or below 2^−54 that subtraction
        // returns exactly 1.0 — depth 55 for the fusi parameters, whose deepest rate is 2^−(d−1)
        // — and the transition ceases to exist in the returned matrix. The stationary
        // distribution of THAT matrix is then wrong by whole percent (5.5e-4 in the learning
        // probability at depth 60) while every solver reports it as exactly stationary, because it
        // is: of the wrong chain. A rate that is zero because `q0`, `p0` or `x` is zero is a
        // different, legitimate thing (a reducible chain) and passes through.
        for k in 0..d {
            let flip = self.q0 * self.x.powi(k as i32);
            let deep = if k + 1 == d { 0.0 } else { self.p0 * self.x.powi(k as i32) };
            for rate in [flip, deep] {
                // This is the subtraction the matrix performs to store the rate, tested on the
                // rate itself: doubles just below 1.0 are spaced 2^−53, so `1 − rate` rounds
                // to 1.0 once `rate ≤ 2^−54` (ties-to-even) and the transition has left this
                // matrix. `generator_matrix` keeps it; only the dynamics that multiply by `P`
                // are affected.
                if rate > 0.0 && 1.0 - rate == 1.0 {
                    return Err(ContinualError::OutOfRange {
                        what: "depth (the deepest transition rate has rounded to zero in f64)",
                        value: d as f64,
                        low: 1.0,
                        high: k as f64,
                    });
                }
            }
            // Potentiating stimulus, probability 1/2.
            // (+, k): deepen, or stay.
            t[k * n + k] += 0.5 * (1.0 - deep);
            if k + 1 < d {
                t[k * n + k + 1] += 0.5 * deep;
            }
            // (−, k): flip to (+, 0), or stay.
            t[(d + k) * n] += 0.5 * flip;
            t[(d + k) * n + (d + k)] += 0.5 * (1.0 - flip);
            // Depressing stimulus, probability 1/2: the mirror image.
            t[(d + k) * n + (d + k)] += 0.5 * (1.0 - deep);
            if k + 1 < d {
                t[(d + k) * n + (d + k + 1)] += 0.5 * deep;
            }
            t[k * n + d] += 0.5 * flip;
            t[k * n + k] += 0.5 * (1.0 - flip);
        }
        Ok(t)
    }

    /// `P − I`, row-major, built **exactly** from the rates rather than by subtracting `I` from
    /// [`Cascade::transition_matrix`].
    ///
    /// The difference is not cosmetic. `transition_matrix` stores each diagonal as
    /// `0.5·(1 − flip) + 0.5·(1 − deepen)`, a number near `1.0` whose ulp is `2^−52`; a deep
    /// rate of `2^−51` survives in it with one bit of precision and `2^−52` does not survive at
    /// all. Here the off-diagonals are `0.5·rate` — exact dyadic fractions for the fusi
    /// parameters — and each diagonal is `−(flip + deepen)/2`, also tiny and exact. Every entry is
    /// representable to full precision at any depth where the rates themselves are, which is
    /// depth ~1,000 before `2^−(d−1)` underflows. The stationary solve uses this; the dynamics
    /// still multiply by `P` and are protected by `transition_matrix`'s refusal.
    ///
    /// # Errors
    ///
    /// As [`Cascade::validate`].
    pub fn generator_matrix(&self) -> Result<Vec<f64>, ContinualError> {
        self.validate()?;
        let d = self.depth;
        let n = 2 * d;
        let mut g = vec![0.0; n * n];
        for k in 0..d {
            let flip = self.q0 * self.x.powi(k as i32);
            let deep = if k + 1 == d { 0.0 } else { self.p0 * self.x.powi(k as i32) };
            // Off-diagonals exactly as `transition_matrix` places them; the diagonal is minus the
            // row's outgoing mass, computed from the SAME rates rather than from `1 − rate`.
            if k + 1 < d {
                g[k * n + k + 1] += 0.5 * deep;
                g[(d + k) * n + (d + k + 1)] += 0.5 * deep;
            }
            g[(d + k) * n] += 0.5 * flip;
            g[k * n + d] += 0.5 * flip;
            g[k * n + k] = -0.5 * (deep + flip);
            g[(d + k) * n + (d + k)] = -0.5 * (deep + flip);
        }
        Ok(g)
    }

    /// The stationary distribution over the `2 * depth` states under stimuli of random sign.
    ///
    /// Solved **exactly**, by flux balance across the cuts between levels — see
    /// `Cascade::stationary_by_flux_balance` — in `O(d)`, with no cancellation and therefore no
    /// loss of precision at any representable depth. Verified against the transition matrix before
    /// it is returned.
    ///
    /// # ⛔ Why it is not power iteration any more
    ///
    /// It was, from the uniform distribution with a fixed absolute residual threshold of `1e-15`
    /// and a million-iteration cap, and that was wrong in two different ways at once. The chain's
    /// slowest mode is the deepest level's flip rate, `q0·x^(d−1)` — `2^−(d−1)` for
    /// [`Cascade::fusi_2005`] — so iterations-to-converge grows like `2^d`:
    ///
    /// | depth | what the iteration returned |
    /// |---|---|
    /// | 1..=15 | correct, `2/(d+1)` |
    /// | 16..=45 | `Err(NoStationary)` after 0.7–6.2 s of CPU, for a chain with an exactly computable answer |
    /// | ≥ 46 | **`Ok` with the wrong distribution, silently**: `2/d` instead of `2/(d+1)` |
    ///
    /// Past `d ≈ 46` the slowest mode's rate `2^−(d−1)` is itself below `1e-15`, so successive
    /// iterates differed by less than the threshold while the distribution was still far from
    /// stationary. The guard was quiet exactly where the answer was wrong and loud where it was
    /// nearly right. `Cascade::fusi_2005(46).lifetime(0.01, 1000)` returned `Some(7)` against
    /// `Some(40)` at depth 15 — nonsense, from the model whose entire claim is that depth buys
    /// retention. There is no single threshold that fixes this; `1e-9` breaks the shallow depths
    /// instead. The direct solve has no threshold.
    ///
    /// The exact answer at depth 46 is `2/47 = 0.042553191489362…`, confirmed by solving the same
    /// system in exact rational arithmetic outside this crate.
    ///
    /// # The reducible case
    ///
    /// For `x == 0` every level past the first is absorbing, the chain is **reducible**, and the
    /// stationary distribution is not unique: the linear system is singular (an absorbing state
    /// contributes a structurally zero column to `Pᵀ − I`). That case falls back to the limit of
    /// the iteration from the uniform start — a well-defined number, and not the only stationary
    /// distribution — which converges quickly there because nothing slow is left in the chain.
    ///
    /// # Errors
    ///
    /// As [`Cascade::validate`], plus [`ContinualError::NoStationary`] only on the reducible
    /// fallback path, if its residual has not fallen below `1e-15` within a million iterations.
    pub fn stationary(&self) -> Result<Vec<f64>, ContinualError> {
        let m = self.transition_matrix()?;
        let n = 2 * self.depth;
        if let Some(pi) = self.stationary_by_flux_balance() {
            // Verified against the matrix the dynamics actually use before it is trusted.
            if stationarity_residual(&m, n, &pi) <= 1e-9 {
                return Ok(pi);
            }
        }
        Self::stationary_by_iteration(&m, n)
    }

    /// The stationary distribution by flux balance across each cut between levels — exact, `O(d)`,
    /// and free of cancellation.
    ///
    /// Stimuli are equiprobable in sign, so the chain is symmetric under polarity and reduces to a
    /// quotient on the `d` levels: from level `k` a synapse deepens to `k + 1` with probability
    /// `deepen_k / 2` or flips to level `0` with probability `flip_k / 2`. Only the deepening step
    /// crosses the cut between `k` and `k + 1` upward, and every flip from a level above `k`
    /// crosses it downward on its way to `0`, so at equilibrium
    ///
    /// ```text
    /// π_k · deepen_k  =  Σ_{j > k} π_j · flip_j
    /// ```
    ///
    /// Set `π_{d−1} = 1` and walk down: each `π_k` is a sum of positive terms divided by a positive
    /// rate. Nothing is subtracted, so nothing cancels, and the result is exact to roundoff at any
    /// depth where the rates are representable — which is what Gaussian elimination on the same
    /// system could not deliver past depth ~48, where the symmetric subspace's own slow mode
    /// `2^−(d−2)` reaches machine epsilon and the deep-level balance equations become numerically
    /// empty (learning probability wrong by `3.9e-10` at depth 50 and `1.6e-5` at depth 52).
    ///
    /// `None` when some `deepen_k` for `k < d − 1` is zero — `p0 == 0` or `x == 0` — because the
    /// division is then `0/0`: the chain is reducible and has no unique stationary distribution.
    fn stationary_by_flux_balance(&self) -> Option<Vec<f64>> {
        let d = self.depth;
        if d == 0 {
            return None;
        }
        let flip = |k: usize| self.q0 * self.x.powi(k as i32);
        let deep = |k: usize| self.p0 * self.x.powi(k as i32);
        for k in 0..d.saturating_sub(1) {
            if !(deep(k) > 0.0) {
                return None;
            }
        }
        let mut level = vec![0.0f64; d];
        level[d - 1] = 1.0;
        let mut above = level[d - 1] * flip(d - 1);
        for k in (0..d.saturating_sub(1)).rev() {
            level[k] = above / deep(k);
            above += level[k] * flip(k);
        }
        let total: f64 = 2.0 * level.iter().sum::<f64>();
        if !(total.is_finite() && total > 0.0) {
            return None;
        }
        let mut pi = vec![0.0f64; 2 * d];
        for k in 0..d {
            pi[k] = level[k] / total;
            pi[d + k] = level[k] / total;
        }
        Some(pi)
    }

    /// The former implementation, kept for the reducible chain and nothing else. See
    /// [`Cascade::stationary`] for why it is not the primary path.
    fn stationary_by_iteration(m: &[f64], n: usize) -> Result<Vec<f64>, ContinualError> {
        let mut v = vec![1.0 / n as f64; n];
        let mut next = vec![0.0; n];
        let mut residual = f64::INFINITY;
        for _ in 0..1_000_000 {
            for j in 0..n {
                let mut acc = 0.0;
                for i in 0..n {
                    acc += v[i] * m[i * n + j];
                }
                next[j] = acc;
            }
            residual = v.iter().zip(&next).map(|(a, b)| (a - b).abs()).sum();
            v.copy_from_slice(&next);
            if residual < 1e-15 {
                return Ok(v);
            }
        }
        Err(ContinualError::NoStationary { residual })
    }

    /// The memory signal `t` stimuli after one memory was stored, from the equilibrium the synapse
    /// population was already in.
    ///
    /// **This is the paper's protocol and the reason the numbers mean anything.** The population is
    /// first brought to its stationary distribution under unstructured stimuli, which is where a
    /// deployed synapse actually lives; then one coordinated potentiating stimulus stores the
    /// memory; then `t` further random stimuli wash over it. The signal is the ensemble's mean
    /// polarity, which is `+1` only if every synapse flipped and decays toward `0`. Starting every
    /// synapse at `(+, 0)` instead — the obvious thing to do — gives a signal of exactly 1 at
    /// `t = 0` for **every** depth and hides the entire trade, because it hands the deep cascade
    /// the one initial condition it can never reach.
    ///
    /// So `signal_after(0)` is not 1; it is [`Cascade::learning_probability`], and the two are
    /// asserted equal in this module's tests.
    ///
    /// # The closed form this is checked against
    ///
    /// At `depth == 1` there is no metaplastic state. Equilibrium is half plus and half minus; one
    /// potentiating stimulus flips a fraction `q0` of the minus half, giving a signal of exactly
    /// `q0`; and thereafter each random stimulus multiplies the deviation from equal odds by
    /// `1 - q0`. So
    ///
    /// ```text
    /// signal_after(t) = q0 * (1 - q0)^t
    /// ```
    ///
    /// which is the whole trade in one line: the factor that sets how much is learned is one minus
    /// the factor that sets how long it lasts. The cascade exists to break that tie, and it cannot
    /// be broken with a single scalar weight.
    ///
    /// # Errors
    ///
    /// As [`Cascade::stationary`].
    pub fn signal_after(&self, t: usize) -> Result<f64, ContinualError> {
        let m = self.transition_matrix()?;
        let d = self.depth;
        let n = 2 * d;
        let mut v = self.stationary()?;
        self.store_into(&mut v);
        let mut next = vec![0.0; n];
        for _ in 0..t {
            for j in 0..n {
                let mut acc = 0.0;
                for i in 0..n {
                    acc += v[i] * m[i * n + j];
                }
                next[j] = acc;
            }
            v.copy_from_slice(&next);
        }
        Ok(v[..d].iter().sum::<f64>() - v[d..].iter().sum::<f64>())
    }

    /// One coordinated potentiating stimulus, applied to a distribution in place.
    fn store_into(&self, v: &mut [f64]) {
        let d = self.depth;
        let mut out = vec![0.0; 2 * d];
        for k in 0..d {
            let flip = self.q0 * self.x.powi(k as i32);
            let deep = if k + 1 == d { 0.0 } else { self.p0 * self.x.powi(k as i32) };
            // A plus synapse is reinforced: it deepens or stays.
            out[k] += v[k] * (1.0 - deep);
            if k + 1 < d {
                out[k + 1] += v[k] * deep;
            }
            // A minus synapse is contradicted: it flips to (+, 0) or stays.
            out[0] += v[d + k] * flip;
            out[d + k] += v[d + k] * (1.0 - flip);
        }
        v.copy_from_slice(&out);
    }

    /// Smallest number of stimuli after which [`Cascade::signal_after`] drops below `floor`.
    ///
    /// The floor is **absolute**, not a fraction of the initial signal: a memory whose signal has
    /// fallen under some fixed detectability threshold is gone whatever it started at, and
    /// normalising by the initial signal would credit a cascade for the very weakness — a small
    /// initial signal — that the learning-rate column is there to charge it for.
    ///
    /// `None` when the signal has not dropped below `floor` within `cap` stimuli, and `Some(0)`
    /// when it never rose above it in the first place.
    ///
    /// # Errors
    ///
    /// As [`Cascade::stationary`], plus [`ContinualError::OutOfRange`] for a `floor` outside
    /// `0.0..=1.0`.
    pub fn lifetime(&self, floor: f64, cap: usize) -> Result<Option<usize>, ContinualError> {
        let floor = in_range("floor", floor, 0.0, 1.0)?;
        let m = self.transition_matrix()?;
        let d = self.depth;
        let n = 2 * d;
        let mut v = self.stationary()?;
        self.store_into(&mut v);
        let mut next = vec![0.0; n];
        for t in 0..=cap {
            let signal = v[..d].iter().sum::<f64>() - v[d..].iter().sum::<f64>();
            if signal < floor {
                return Ok(Some(t));
            }
            for j in 0..n {
                let mut acc = 0.0;
                for i in 0..n {
                    acc += v[i] * m[i * n + j];
                }
                next[j] = acc;
            }
            v.copy_from_slice(&next);
        }
        Ok(None)
    }

    /// Probability that one correctly-signed stimulus flips a synapse drawn from the stationary
    /// distribution of the *wrong* polarity. The learning rate of the model, dimensionless.
    ///
    /// This is the quantity that **falls** as depth rises, and it is the price of the retention
    /// that [`Cascade::lifetime`] reports. It is also exactly [`Cascade::signal_after`] at `t = 0`.
    ///
    /// # The closed form this is checked against
    ///
    /// For [`Cascade::fusi_2005`] — `q0 = p0 = 1`, `x = 1/2` — it is exactly `2 / (depth + 1)`.
    /// Writing `a_k` for the stationary mass of level `k` at one polarity, the level-to-level
    /// balance is `a_{k-1} * p_{k-1} = a_k * (p_k + q_k)`, and with `p_k = q_k = 2^-k` for every
    /// level but the deepest (where `p = 0`) that gives `a_0 = ... = a_{d-2}` and
    /// `a_{d-1} = 2 * a_{d-2}`. The masses sum to one half, so `a = 1 / (2 * (d + 1))`, and the
    /// flip probability averages to `4a = 2 / (d + 1)`.
    ///
    /// # Errors
    ///
    /// As [`Cascade::stationary`], plus [`ContinualError::Empty`] if the stationary distribution
    /// puts no mass on the opposing polarity at all, so the conditional probability has no
    /// denominator.
    pub fn learning_probability(&self) -> Result<f64, ContinualError> {
        let pi = self.stationary()?;
        let d = self.depth;
        let mass: f64 = pi[d..].iter().sum();
        if mass == 0.0 {
            return Err(ContinualError::Empty { what: "opposing-polarity mass" });
        }
        let mut acc = 0.0;
        for k in 0..d {
            acc += pi[d + k] * self.q0 * self.x.powi(k as i32);
        }
        Ok(acc / mass)
    }

    /// Bits of state per synapse: one for the weight, `ceil(log2(depth))` for the level.
    ///
    /// The number that makes this mechanism different in kind from the `f64` importance vectors
    /// above: depth 8 costs 4 bits per synapse in total.
    #[must_use]
    pub fn bits_per_synapse(&self) -> u32 {
        1 + self.depth.next_power_of_two().trailing_zeros()
    }
}

/// A sampled population of cascade synapses, for checking the chain against an actual simulation.
///
/// Deterministic from its seed. Every synapse starts at `(+, 0)`, which is the same initial
/// condition [`Cascade::signal_after`] propagates, so the two are directly comparable.
#[derive(Debug, Clone, PartialEq)]
pub struct CascadeEnsemble {
    cascade: Cascade,
    polarity: Vec<bool>,
    level: Vec<u32>,
    rng: Rng,
}

impl CascadeEnsemble {
    /// Build `n` synapses, all at `(+, 0)`.
    ///
    /// # Errors
    ///
    /// [`ContinualError::Empty`] for `n == 0`, and as [`Cascade::validate`].
    pub fn new(cascade: Cascade, n: usize, seed: u64) -> Result<Self, ContinualError> {
        cascade.validate()?;
        if n == 0 {
            return Err(ContinualError::Empty { what: "n" });
        }
        Ok(Self {
            cascade,
            polarity: vec![true; n],
            level: vec![0; n],
            rng: Rng::new(seed),
        })
    }

    /// Build `n` synapses drawn from the chain's stationary distribution.
    ///
    /// This is the initial condition [`Cascade::signal_after`] propagates, so an ensemble built
    /// this way is directly comparable to the exact chain at every step with no equilibration
    /// transient to wait out — which is what makes the agreement between them a check on the
    /// dynamics rather than on how long somebody ran a burn-in.
    ///
    /// # Errors
    ///
    /// [`ContinualError::Empty`] for `n == 0`, and as [`Cascade::stationary`].
    pub fn from_stationary(cascade: Cascade, n: usize, seed: u64) -> Result<Self, ContinualError> {
        let pi = cascade.stationary()?;
        if n == 0 {
            return Err(ContinualError::Empty { what: "n" });
        }
        let d = cascade.depth;
        let mut rng = Rng::new(seed);
        let mut polarity = Vec::with_capacity(n);
        let mut level = Vec::with_capacity(n);
        for _ in 0..n {
            // Inverse-CDF sampling over the 2*depth states; the final state absorbs the rounding
            // slack so a draw can never fall off the end.
            let u = rng.next_f64();
            let mut acc = 0.0;
            let mut s = 2 * d - 1;
            for (i, &p) in pi.iter().enumerate() {
                acc += p;
                if u < acc {
                    s = i;
                    break;
                }
            }
            polarity.push(s < d);
            level.push((s % d) as u32);
        }
        Ok(Self { cascade, polarity, level, rng })
    }

    /// Number of synapses.
    #[must_use]
    pub fn len(&self) -> usize {
        self.polarity.len()
    }

    /// Whether the population is empty; it never is, because [`CascadeEnsemble::new`] refuses zero.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.polarity.is_empty()
    }

    /// Mean polarity, `+1` for all-plus and `-1` for all-minus.
    #[must_use]
    pub fn signal(&self) -> f64 {
        let plus = self.polarity.iter().filter(|&&p| p).count();
        (2 * plus) as f64 / self.len() as f64 - 1.0
    }

    /// Apply one stimulus of the given sign to every synapse.
    pub fn stimulate(&mut self, potentiating: bool) {
        for i in 0..self.polarity.len() {
            self.apply_one(i, potentiating);
        }
    }

    /// Apply `steps` stimuli whose signs are drawn independently per synapse, which is the
    /// unstructured background the memory curve is measured against.
    pub fn random_stimuli(&mut self, steps: usize) {
        for _ in 0..steps {
            for i in 0..self.polarity.len() {
                let up = self.rng.next_f64() < 0.5;
                self.apply_one(i, up);
            }
        }
    }

    fn apply_one(&mut self, i: usize, potentiating: bool) {
        let k = self.level[i] as usize;
        let reinforcing = self.polarity[i] == potentiating;
        if reinforcing {
            let deep = if k + 1 == self.cascade.depth {
                0.0
            } else {
                self.cascade.p0 * self.cascade.x.powi(k as i32)
            };
            if self.rng.next_f64() < deep {
                self.level[i] += 1;
            }
        } else {
            let flip = self.cascade.q0 * self.cascade.x.powi(k as i32);
            if self.rng.next_f64() < flip {
                self.polarity[i] = potentiating;
                self.level[i] = 0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Cascade, CascadeEnsemble, Consolidation, ContinualError, Cost, Curriculum, Example,
        Homeostat, Linear, Protocol, Replay, Results, Schedule, evaluate, fisher_diagonal,
        latency_features, path_importance,
        remove_sample_mean, run, standard_normal, train,
    };
    use crate::rng::Rng;
    use crate::tasks::{Dataset, LatencyPatterns, Sample, Split};

    const SEED: u64 = 0xC0FF_EE01;

    fn base() -> Dataset<Sample> {
        LatencyPatterns::default().generate().unwrap()
    }

    fn course(n: usize) -> Curriculum {
        Curriculum::permuted(&base(), n, SEED).unwrap()
    }

    /// The same curriculum with the permutation-invariant mean left in, which is what a naive
    /// permuted-input benchmark measures.
    fn uncentred(n: usize) -> Curriculum {
        let d = base();
        let feat = |split: Split| -> Vec<Example> {
            d.split(split)
                .iter()
                .map(|s| Example {
                    x: latency_features(s, d.n_inputs, d.ticks).unwrap(),
                    label: s.label,
                })
                .collect()
        };
        Curriculum::from_examples(
            "uncentred",
            feat(Split::Train),
            feat(Split::Test),
            d.n_classes as usize,
            n,
            SEED,
        )
        .unwrap()
    }

    // -----------------------------------------------------------------------------------------
    // (a) The collapse, first, as a number.
    // -----------------------------------------------------------------------------------------

    /// THE CENTRAL MEASUREMENT OF THE MODULE. Everything below is a change to these two numbers,
    /// and if this one stops holding, nothing below means anything.
    #[test]
    fn a_plain_learner_learns_task_a_and_then_loses_it() {
        let c = course(2);
        let res = run(&c, &Protocol::plain(Schedule::default())).unwrap();
        let chance = c.tasks[0].chance;

        assert_eq!(res.r.len(), 2);
        assert_eq!(res.r[0].len(), 2);
        assert_eq!(res.learned(0), Some(1.0), "task A was never learned");
        assert_eq!(res.learned(1), Some(1.0), "task B was never learned");
        assert_eq!(res.retained(1), res.learned(1), "the newest task is always fine");

        let retained = res.retained(0).unwrap();
        assert!(retained < 0.30, "task A survived, so there is nothing to mitigate: {retained}");
        assert!(res.forgetting(0).unwrap() > 0.70, "{:?}", res.forgetting(0));
        // The collapse lands on chance, not below it: the readout is not anti-correlated with task
        // A, it simply no longer encodes it. A retention far under chance would mean something else
        // was wrong.
        assert!(retained >= chance - 0.05, "retention fell below chance ({retained})");
        assert!((res.backward_transfer().unwrap() + 0.756).abs() < 0.02);
        assert_eq!(res.cost.extra_floats, 0);
        assert_eq!(res.cost.replayed_steps, 0);
        assert_eq!(res.cost.importance_passes, 0);
        assert_eq!(res.cost.model_parameters, 5 * (8 + 1));
    }

    /// THE CONTROL THAT MAKES THE ONE ABOVE MEAN SOMETHING. Three things can erase task A and this
    /// separates them by changing exactly one at a time. The first row is the important one: with
    /// the permutation-invariant mean left in and no weight decay, **almost nothing is forgotten**,
    /// so a benchmark built that way would have reported a mitigation's success for free.
    #[test]
    fn what_erases_task_a_is_measured_in_three_pieces() {
        let no_decay = Schedule { weight_decay: 0.0, ..Schedule::default() };
        let decay = Schedule::default();

        let shortcut_no_decay =
            run(&uncentred(2), &Protocol::plain(no_decay)).unwrap().retained(0).unwrap();
        let shortcut_decay =
            run(&uncentred(2), &Protocol::plain(decay)).unwrap().retained(0).unwrap();
        let interference =
            run(&course(2), &Protocol::plain(no_decay)).unwrap().retained(0).unwrap();
        let both = run(&course(2), &Protocol::plain(decay)).unwrap().retained(0).unwrap();

        assert!(shortcut_no_decay > 0.95, "a shortcut benchmark forgot something: {shortcut_no_decay}");
        assert!((0.50..0.62).contains(&interference), "pure interference: {interference}");
        assert!((0.35..0.45).contains(&shortcut_decay), "decay alone: {shortcut_decay}");
        assert!(both < 0.30, "interference plus decay: {both}");
        // Strictly ordered, and the order is the finding: on this curriculum the OPTIMISER erases
        // more of task A than the interference does. A benchmark that reported only
        // `shortcut_decay` would be showing a collapse of 0.6 with no interference in it at all.
        assert!(shortcut_no_decay > interference);
        assert!(interference > shortcut_decay);
        assert!(shortcut_decay > both);
    }

    /// The shortcut itself, measured rather than asserted. The per-sample mean is invariant under
    /// **any** permutation — exactly, to the last bit that summation order allows — and on this task
    /// it alone classifies at better than 0.95.
    #[test]
    fn the_permutation_invariant_shortcut_is_real_and_measured() {
        let raw = uncentred(2);
        let (a, b) = (&raw.tasks[0], &raw.tasks[1]);
        let mut worst = 0.0f64;
        for i in 0..a.train.len() {
            let sa: f64 = a.train[i].x.iter().sum();
            let sb: f64 = b.train[i].x.iter().sum();
            worst = worst.max((sa - sb).abs());
        }
        assert!(worst < 1e-14, "the mean was not permutation-invariant: {worst}");

        // A classifier that sees only that one invariant number.
        let mut mean = [0.0f64; 5];
        let mut n = [0.0f64; 5];
        for e in &a.train {
            mean[e.label as usize] += e.x.iter().sum::<f64>();
            n[e.label as usize] += 1.0;
        }
        for c in 0..5 {
            mean[c] /= n[c];
        }
        let hits = a
            .test
            .iter()
            .filter(|e| {
                let v: f64 = e.x.iter().sum();
                let mut best = 0;
                for c in 1..5 {
                    if (v - mean[c]).abs() < (v - mean[best]).abs() {
                        best = c;
                    }
                }
                best == e.label as usize
            })
            .count();
        let acc = hits as f64 / a.test.len() as f64;
        assert!(acc > 0.95, "mean-only accuracy {acc}");

        // And the consequence: forward transfer is large on the shortcut curriculum and gone on
        // the centred one, which is the whole reason `permuted` centres.
        let raw4 = run(&uncentred(4), &Protocol::plain(Schedule::default())).unwrap();
        let cen4 = run(&course(4), &Protocol::plain(Schedule::default())).unwrap();
        assert!(raw4.forward_transfer().unwrap() > 0.10, "{:?}", raw4.forward_transfer());
        assert!(cen4.forward_transfer().unwrap().abs() < 0.05, "{:?}", cen4.forward_transfer());
    }

    /// Removing the mean has to actually remove it, exactly.
    #[test]
    fn removing_the_sample_mean_leaves_a_mean_of_zero() {
        let mut e = vec![
            Example { x: vec![0.25, -0.5, 0.125, 0.0], label: 0 },
            Example { x: vec![1.0; 3], label: 1 },
            Example { x: Vec::new(), label: 0 },
        ];
        remove_sample_mean(&mut e);
        assert!(e[0].x.iter().sum::<f64>().abs() < 1e-16);
        assert_eq!(e[1].x, vec![0.0; 3]);
        assert!(e[2].x.is_empty());
    }

    // -----------------------------------------------------------------------------------------
    // (b) Each mitigation, with its own number and its own bill.
    // -----------------------------------------------------------------------------------------

    fn two_task(p: &Protocol) -> (f64, f64, Cost) {
        let res = run(&course(2), p).unwrap();
        (res.retained(0).unwrap(), res.retained(1).unwrap(), res.cost)
    }

    #[test]
    fn every_mitigation_reduces_the_collapse_and_says_what_it_cost() {
        let sched = Schedule::default();
        let (plain, plain_b, plain_cost) = two_task(&Protocol::plain(sched));

        let (ewc, ewc_b, ewc_cost) = two_task(&Protocol {
            consolidation: Consolidation::Ewc { lambda: 150.0 },
            ..Protocol::plain(sched)
        });
        let (si, si_b, si_cost) = two_task(&Protocol {
            consolidation: Consolidation::Si { c: 0.02, xi: 0.1 },
            ..Protocol::plain(sched)
        });
        let (exp, _, exp_cost) = two_task(&Protocol {
            replay: Replay::Experience { budget: 100 },
            ..Protocol::plain(sched)
        });
        let (gnr, _, gen_cost) = two_task(&Protocol {
            replay: Replay::Generative { per_task: 50 },
            ..Protocol::plain(sched)
        });

        // Every one beats the control by a margin far wider than the gap between chance and the
        // control itself, so none of them is being credited for noise.
        for (name, got) in [("ewc", ewc), ("si", si), ("experience", exp), ("generative", gnr)] {
            assert!(got > plain + 0.20, "{name} retained {got} against a plain {plain}");
        }
        // Rehearsal on real data is the strongest, which is what the literature reports; the
        // interesting question is therefore its bill, not whether it works.
        assert!(exp > ewc && exp > si, "replay {exp} did not beat consolidation {ewc}/{si}");
        // Consolidation is paid for out of the NEW task. Plain and replay keep task B at 1.0.
        assert_eq!(plain_b, 1.0);
        assert!(ewc_b < 0.95, "EWC kept task A for free, which it cannot: {ewc_b}");
        assert!(si_b < 1.0, "SI kept task A for free: {si_b}");

        // The bills, each in its own currency.
        assert_eq!(plain_cost.extra_floats, 0);
        assert_eq!(ewc_cost.extra_floats, 2 * ewc_cost.model_parameters);
        assert_eq!(si_cost.extra_floats, 4 * si_cost.model_parameters);
        assert_eq!(exp_cost.extra_floats, 100 * (8 + 1));
        // Two tasks absorbed, five classes each, a mean and a standard deviation per feature.
        assert_eq!(gen_cost.extra_floats, 2 * 5 * 2 * 8);
        // THE HEADLINE TRADE: generative replay matches real examples at a fifth of the memory,
        // and pays for it in gradient steps instead.
        assert!(gnr >= exp);
        assert!(gen_cost.extra_floats * 5 < exp_cost.extra_floats);
        assert!(gen_cost.gradient_steps > exp_cost.gradient_steps);

        // EWC pays a pass over the training set at each boundary; SI pays none.
        assert_eq!(ewc_cost.importance_passes, 2 * 500);
        assert_eq!(si_cost.importance_passes, 0);
        assert_eq!(plain_cost.gradient_steps, 2 * 12 * 500);
        assert_eq!(exp_cost.gradient_steps, plain_cost.gradient_steps + 12 * 100);
        assert_eq!(exp_cost.replayed_steps, 12 * 100);
        assert_eq!(gen_cost.replayed_steps, 12 * 5 * 50);
        assert_eq!(ewc_cost.extra_bytes(), ewc_cost.extra_floats * 8);
    }

    /// THE HALF NOBODY REPORTS. Consolidation strength does not simply buy retention; it buys it
    /// from the new task, monotonically in both directions across the whole usable range.
    #[test]
    fn consolidation_strength_trades_the_old_task_against_the_new_one() {
        let sched = Schedule::default();
        let mut old = Vec::new();
        let mut new = Vec::new();
        for lambda in [30.0, 70.0, 150.0, 300.0] {
            let r = run(
                &course(2),
                &Protocol { consolidation: Consolidation::Ewc { lambda }, ..Protocol::plain(sched) },
            )
            .unwrap();
            old.push(r.retained(0).unwrap());
            new.push(r.retained(1).unwrap());
        }
        for i in 1..old.len() {
            assert!(old[i] > old[i - 1], "retention not monotone in lambda: {old:?}");
            assert!(new[i] < new[i - 1], "the new task did not pay for it: {new:?}");
        }
        // The two ends of the sweep are qualitatively different outcomes, not a wobble.
        assert!(old[0] < 0.5 && old[3] > 0.95, "{old:?}");
        assert!(new[0] > 0.95 && new[3] < 0.8, "{new:?}");
    }

    /// Past that trade the penalty is stiffer than the learning rate can integrate and the run
    /// blows up. It is REFUSED, because the alternative is reporting chance accuracy — the same
    /// number an honest failure to learn produces.
    #[test]
    fn too_much_consolidation_is_refused_rather_than_reported_as_chance() {
        let sched = Schedule::default();
        assert!(matches!(
            run(
                &course(2),
                &Protocol {
                    consolidation: Consolidation::Ewc { lambda: 3000.0 },
                    ..Protocol::plain(sched)
                }
            ),
            Err(ContinualError::Diverged { .. })
        ));
        assert!(matches!(
            run(
                &course(2),
                &Protocol {
                    consolidation: Consolidation::Si { c: 1.0, xi: 0.1 },
                    ..Protocol::plain(sched)
                }
            ),
            Err(ContinualError::Diverged { .. })
        ));
    }

    // -----------------------------------------------------------------------------------------
    // (c) The equality that pins the implementation.
    // -----------------------------------------------------------------------------------------

    /// With no importance weighting there is no penalty, so EWC must be the plain learner — not
    /// approximately, and not only in accuracy, but on **every entry of the accuracy matrix**. This
    /// catches a consolidation path that perturbs the example ordering, consumes a different number
    /// of random draws, or applies weight decay twice.
    #[test]
    fn ewc_with_zero_lambda_is_exactly_the_plain_learner() {
        let c = course(3);
        let sched = Schedule::default();
        let plain = run(&c, &Protocol::plain(sched)).unwrap();
        let zero = run(
            &c,
            &Protocol {
                consolidation: Consolidation::Ewc { lambda: 0.0 },
                ..Protocol::plain(sched)
            },
        )
        .unwrap();
        assert_eq!(plain.r, zero.r, "zero-lambda EWC diverged from the plain learner");
        assert_eq!(plain.baseline, zero.baseline);
        assert_eq!(plain.cost.gradient_steps, zero.cost.gradient_steps);
        // The cost is still charged, because the vectors are still allocated and the boundary pass
        // is still run. Free retention would be the giveaway that nothing happened.
        assert_eq!(zero.cost.extra_floats, 2 * zero.cost.model_parameters);
        assert_eq!(zero.cost.importance_passes, 3 * 500);
    }

    /// The same pin for synaptic intelligence, which reaches the penalty by a different route.
    #[test]
    fn si_with_zero_c_is_exactly_the_plain_learner() {
        let c = course(3);
        let sched = Schedule::default();
        let plain = run(&c, &Protocol::plain(sched)).unwrap();
        let zero = run(
            &c,
            &Protocol {
                consolidation: Consolidation::Si { c: 0.0, xi: 0.1 },
                ..Protocol::plain(sched)
            },
        )
        .unwrap();
        assert_eq!(plain.r, zero.r);
        assert_eq!(zero.cost.extra_floats, 4 * zero.cost.model_parameters);
    }

    // -----------------------------------------------------------------------------------------
    // Forgetting is not the only constraint, and saying so needs one more run.
    // -----------------------------------------------------------------------------------------

    /// At two tasks a buffer holding every example reaches 1.000 on both, so the two tasks ARE
    /// jointly solvable by one linear model and the collapse above is purely an ordering effect.
    /// At five tasks the same buffer — which cannot forget anything — reaches only 0.549, so the
    /// binding constraint has become capacity. Reporting one without the other misattributes it.
    #[test]
    fn at_five_tasks_the_binding_constraint_is_capacity_not_forgetting() {
        let sched = Schedule::default();
        let joint2 = run(
            &course(2),
            &Protocol { replay: Replay::Experience { budget: 5000 }, ..Protocol::plain(sched) },
        )
        .unwrap();
        assert_eq!(joint2.average_accuracy(), Some(1.0), "two tasks are jointly solvable");

        let plain5 = run(&course(5), &Protocol::plain(sched)).unwrap();
        let joint5 = run(
            &course(5),
            &Protocol { replay: Replay::Experience { budget: 5000 }, ..Protocol::plain(sched) },
        )
        .unwrap();
        // The buffer is charged for what it holds, not for what it reserved: 2500 examples of 8
        // features and a label, against a budget with room for twice that.
        assert_eq!(joint5.cost.extra_floats, 5 * 500 * (8 + 1));
        assert!(joint5.cost.extra_floats < 5000 * (8 + 1));
        let ceiling = joint5.average_accuracy().unwrap();
        assert!(ceiling < 0.60, "five tasks were jointly solvable after all: {ceiling}");
        assert!(ceiling > plain5.average_accuracy().unwrap(), "replay did nothing at all");
        // Less than a quarter of the gap from the plain learner to a perfect score is reachable by
        // removing forgetting entirely; the rest of it is capacity.
        let recovered = ceiling - plain5.average_accuracy().unwrap();
        assert!(recovered < 0.25, "recovered {recovered} of the gap");
    }

    // -----------------------------------------------------------------------------------------
    // (e) Backward transfer of a learner that trained on one task only.
    // -----------------------------------------------------------------------------------------

    /// A learner that trained on task 0 and was then simply **not trained again** cannot have
    /// forgotten anything, so its backward transfer is exactly zero: the two rows of its accuracy
    /// matrix are the same evaluation of the same parameters.
    ///
    /// The tempting shortcut — run the second task at a learning rate small enough to be a no-op —
    /// does not work and is worth knowing about: the softmax argmax is **scale invariant**, so a
    /// model whose weights are subnormal still classifies, and a run at `f64::MIN_POSITIVE` learns
    /// a perfectly good decision boundary out of numbers near `1e-308`. So the learner is frozen
    /// by not calling [`train`] rather than by making [`train`] small.
    #[test]
    fn backward_transfer_of_a_single_trained_task_is_exactly_zero() {
        let c = course(2);
        let mut model = Linear::new(8, 5).unwrap();
        let steps = train(&mut model, &c.tasks[0], &Schedule::default()).unwrap();
        assert_eq!(steps, 12 * 500);

        let row = evaluate(&model, &c).unwrap();
        let res = Results {
            // Task 1 arrives; the learner is not trained on it, so the row is unchanged.
            r: vec![row.clone(), row],
            baseline: c.tasks.iter().map(|t| t.chance).collect(),
            cost: Cost::default(),
        };
        assert_eq!(res.backward_transfer().unwrap(), 0.0);
        assert_eq!(res.forgetting(0).unwrap(), 0.0);
        // And it is not vacuous: the model really did learn task 0 and really is poor on task 1.
        assert_eq!(res.r[0][0], 1.0);
        assert!(res.r[0][1] < 0.35, "the two tasks were not different enough: {}", res.r[0][1]);
        // Which is negative forward transfer, close to zero because the tasks are unrelated.
        assert!(res.forward_transfer().unwrap().abs() < 0.2);
    }

    /// The scale-invariance trap above, stated as a test so nobody re-introduces the shortcut: a
    /// run at the smallest positive learning rate is NOT a frozen run.
    #[test]
    fn a_subnormal_learning_rate_still_learns_because_argmax_is_scale_invariant() {
        let c = course(2);
        let sched = Schedule { lr: f64::MIN_POSITIVE, epochs: 1, ..Schedule::default() };
        let res = run(&c, &Protocol::plain(sched)).unwrap();
        assert!(res.r[0][0] > 0.6, "subnormal weights classified at {}", res.r[0][0]);
        assert_ne!(res.r[1][0], res.r[0][0]);
    }

    /// With only one task there is no difference to average, which is not the same as a zero.
    #[test]
    fn backward_and_forward_transfer_refuse_a_single_task_sequence() {
        let res = run(&course(1), &Protocol::plain(Schedule::default())).unwrap();
        assert_eq!(res.backward_transfer(), None);
        assert_eq!(res.forward_transfer(), None);
        assert_eq!(res.average_accuracy(), Some(1.0));
    }

    /// The three metrics against a matrix worked out by hand. This is what catches a transposed
    /// index or a flipped sign, which the exact zero above cannot.
    #[test]
    fn the_metrics_match_a_hand_computed_matrix() {
        let res = Results {
            r: vec![vec![0.9, 0.3, 0.2], vec![0.5, 0.8, 0.25], vec![0.4, 0.45, 0.85]],
            baseline: vec![0.2, 0.2, 0.2],
            cost: Cost::default(),
        };
        // ACC = (0.4 + 0.45 + 0.85) / 3
        assert!((res.average_accuracy().unwrap() - 1.70 / 3.0).abs() < 1e-12);
        // BWT = ((0.4 - 0.9) + (0.45 - 0.8)) / 2
        assert!((res.backward_transfer().unwrap() + 0.425).abs() < 1e-12);
        // FWT = ((0.3 - 0.2) + (0.25 - 0.2)) / 2
        assert!((res.forward_transfer().unwrap() - 0.075).abs() < 1e-12);
        assert!((res.forgetting(0).unwrap() - 0.5).abs() < 1e-12);
        assert!((res.forgetting(2).unwrap() - 0.0).abs() < 1e-12);
        assert_eq!(res.learned(1), Some(0.8));
        assert_eq!(res.retained(1), Some(0.45));
        assert_eq!(res.tasks(), 3);
        assert_eq!(res.learned(3), None);
        assert_eq!(res.retained(3), None);
    }

    // -----------------------------------------------------------------------------------------
    // (d) The cascade synapse's trade, both sides measured.
    // -----------------------------------------------------------------------------------------

    /// THE CLOSED FORM. At depth 1 there is no cascade: equilibrium is half plus and half minus,
    /// one stimulus flips a fraction `q0` of the wrong half, and each later stimulus multiplies
    /// what is left by `1 - q0`. The trade is the whole line — the factor that sets how much is
    /// learned is one minus the factor that sets how long it lasts.
    #[test]
    fn depth_one_reproduces_the_two_state_closed_form() {
        for &q0 in &[0.05, 0.1, 0.25, 0.5, 0.9, 1.0] {
            let c = Cascade { depth: 1, q0, p0: 1.0, x: 0.5 };
            for t in 0..12 {
                let want = q0 * (1.0 - q0).powi(t);
                let got = c.signal_after(t as usize).unwrap();
                assert!((got - want).abs() < 1e-12, "q0={q0} t={t}: {got} vs {want}");
            }
            assert!((c.learning_probability().unwrap() - q0).abs() < 1e-12);
        }
    }

    /// A ratio of one makes every level identical, which IS a synapse with no cascade however deep
    /// it is — so the same two-state closed form must come back at every depth. This is the test
    /// that would catch a level index used where a probability belongs.
    #[test]
    fn a_ratio_of_one_collapses_every_depth_to_the_two_state_chain() {
        for depth in [1usize, 2, 5, 9] {
            for &q0 in &[0.25, 0.5] {
                let c = Cascade { depth, q0, p0: 1.0, x: 1.0 };
                assert!((c.learning_probability().unwrap() - q0).abs() < 1e-12, "depth {depth}");
                for t in 0..8 {
                    let want = q0 * (1.0 - q0).powi(t);
                    let got = c.signal_after(t as usize).unwrap();
                    assert!((got - want).abs() < 1e-12, "depth={depth} q0={q0} t={t}: {got}");
                }
            }
        }
    }

    /// THE EXACT IDENTITY. For the halving cascade the stationary learning probability is
    /// `2 / (depth + 1)`, derived from the level-to-level balance in
    /// [`Cascade::learning_probability`]'s doc. Any error in a transition probability, in the
    /// stationary solve or in the conditioning on the wrong polarity moves this off the integer
    /// sequence immediately.
    #[test]
    fn the_stationary_learning_probability_is_two_over_depth_plus_one() {
        // ⛔ 1..=10 ONCE. The power iteration this replaced was correct to depth 15, returned
        // `Err` from 16 to 45 (after seconds of CPU), and returned the WRONG answer silently from
        // 46 — `2/d` instead of `2/(d+1)` — and a sweep that stopped at 10 could not see any of
        // it. This runs to 54 — every depth `transition_matrix` can represent for these
        // parameters — and at depth 46 the exact value `2/47` was independently confirmed in
        // rational arithmetic. Depth 55 is where the deepest rate rounds away in `1 − rate` and
        // `transition_matrix` refuses; that boundary has its own test below.
        // 1e-14, not the 1e-11 this carried under the power iteration: the flux-balance solve
        // measures at 4e-17 or better across the whole range, and a tolerance three orders
        // looser than the solver's own error is one a regression can hide inside.
        for depth in 1..=54 {
            let got = Cascade::fusi_2005(depth).learning_probability().unwrap();
            let want = 2.0 / (depth as f64 + 1.0);
            assert!((got - want).abs() < 1e-14, "depth {depth}: {got} vs {want}");
            // And it is the signal at t = 0, which is what makes the two columns of the trade
            // commensurable.
            let s0 = Cascade::fusi_2005(depth).signal_after(0).unwrap();
            assert!((s0 - got).abs() < 1e-14, "depth {depth}: signal(0) {s0} vs lp {got}");
        }
    }

    /// The flux-balance recursion against an INDEPENDENT algorithm. Gaussian elimination on the
    /// exact generator is a different computation with different roundoff, kept under `cfg(test)`
    /// for exactly this purpose; agreement to 1e-12 is the check that neither is verifying itself.
    ///
    /// The fusi depths stop at 15 because the ORACLE is what degrades: its worst-entry error was
    /// measured at 1.3e-11 at depth 23 (the symmetric subspace's slow mode reaching machine
    /// epsilon), while the recursion sits at 2.8e-17 there against the closed form. Past 15 the
    /// recursion is pinned by `2/(d+1)` instead, at 1e-14. The two non-fusi chains are the point
    /// of this test: they have no closed form, so a second algorithm is the only independent check
    /// they get.
    #[test]
    fn the_flux_balance_solution_matches_gaussian_elimination() {
        for depth in [1usize, 2, 3, 7, 10, 15] {
            let c = Cascade::fusi_2005(depth);
            let n = 2 * depth;
            let m = c.transition_matrix().unwrap();
            let g = c.generator_matrix().unwrap();
            let recursion = c.stationary().unwrap();
            let elimination = super::stationary_direct(&g, &m, n).expect("irreducible");
            let worst = recursion.iter().zip(&elimination).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
            assert!(worst < 1e-12, "depth {depth}: the two solves differ by {worst:.2e}");
        }
        // Non-fusi parameterisations, so the agreement is not a property of q0 = p0 = 1 and
        // x = 1/2 — these have no closed form, and this is their only independent check.
        for c in [
            Cascade { depth: 12, q0: 0.3, p0: 0.8, x: 0.6 },
            Cascade { depth: 9, q0: 0.9, p0: 0.2, x: 0.85 },
        ] {
            let n = 2 * c.depth;
            let m = c.transition_matrix().unwrap();
            let g = c.generator_matrix().unwrap();
            let a = c.stationary().unwrap();
            let b = super::stationary_direct(&g, &m, n).unwrap();
            let worst = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max);
            assert!(worst < 1e-12, "{c:?}: the two solves differ by {worst:.2e}");
        }
    }

    /// The representable limit is a REFUSAL, not a wrong number. For the fusi parameters the
    /// deepest flip rate is `2^−(d−1)`; at depth 55 that is `2^−54`, exactly half the spacing of
    /// doubles just below 1.0, and `1 − 2^−54` rounds to 1.0 under ties-to-even. Before this guard,
    /// depth 60 returned a learning probability wrong by 5.5e-4 (1.7%) while reporting an exactly
    /// stationary distribution, because it was: of the chain the matrix could actually hold.
    #[test]
    fn a_depth_the_matrix_cannot_represent_is_refused_by_name() {
        assert!(Cascade::fusi_2005(54).transition_matrix().is_ok(), "54 is representable");
        let err = Cascade::fusi_2005(55).transition_matrix().unwrap_err();
        match err {
            ContinualError::OutOfRange { what, value, .. } => {
                assert!(what.contains("depth"), "{what}");
                assert!(what.contains("rounded"), "{what}");
                assert_eq!(value, 55.0);
            }
            other => panic!("wrong refusal: {other:?}"),
        }
        // And everything that routes through the matrix refuses the same way rather than
        // computing on the wrong chain.
        assert!(Cascade::fusi_2005(60).stationary().is_err());
        assert!(Cascade::fusi_2005(60).learning_probability().is_err());
        // A legitimately zero rate — a reducible chain — is not this error and still passes.
        assert!(Cascade { x: 0.0, ..Cascade::fusi_2005(5) }.transition_matrix().is_ok());
    }

    /// THE CLOSED FORM FOR THE IMPORTANCE ESTIMATOR. A zero-initialised two-class readout gives
    /// every example a posterior of exactly `0.5`, so every gradient factor is `±0.5` and the
    /// diagonal Fisher is exactly `0.25 * mean(x_k^2)` per weight and exactly `0.25` per bias.
    #[test]
    fn the_fisher_of_a_zero_initialised_binary_readout_is_a_quarter_of_the_second_moment() {
        let m = Linear::new(3, 2).unwrap();
        let ex = vec![
            Example { x: vec![0.5, -0.25, 0.0], label: 0 },
            Example { x: vec![-0.5, 0.75, 0.125], label: 1 },
            Example { x: vec![0.25, 0.25, -0.5], label: 0 },
        ];
        let f = fisher_diagonal(&m, &ex).unwrap();
        assert_eq!(f.len(), 8);
        for k in 0..3 {
            let want =
                0.25 * ex.iter().map(|e| e.x[k] * e.x[k]).sum::<f64>() / ex.len() as f64;
            for c in 0..2 {
                assert!((f[c * 3 + k] - want).abs() < 1e-15, "class {c} feature {k}: {}", f[c * 3 + k]);
            }
        }
        assert!((f[6] - 0.25).abs() < 1e-15);
        assert!((f[7] - 0.25).abs() < 1e-15);
        assert!(matches!(fisher_diagonal(&m, &[]), Err(ContinualError::Empty { .. })));
        assert!(matches!(
            fisher_diagonal(&m, &[Example { x: vec![0.0; 3], label: 7 }]),
            Err(ContinualError::Label { label: 7, n_classes: 2 })
        ));
    }

    /// THE REASON `weight_decay` IS NOT ZERO, measured instead of asserted. Train the same task to
    /// convergence with and without it: without, the posteriors saturate and the Fisher collapses
    /// by orders of magnitude, at which point elastic weight consolidation has nothing to weight.
    #[test]
    fn the_fisher_collapses_when_the_model_saturates() {
        let c = course(1);
        let total = |wd: f64| -> f64 {
            let mut m = Linear::new(8, 5).unwrap();
            let sched = Schedule { weight_decay: wd, epochs: 40, ..Schedule::default() };
            train(&mut m, &c.tasks[0], &sched).unwrap();
            fisher_diagonal(&m, &c.tasks[0].train).unwrap().iter().sum()
        };
        let bounded = total(1e-3);
        let saturated = total(0.0);
        // Measured at 0.0254 against 0.00042 — a factor of sixty — after forty epochs; the
        // threshold is set well under that so the test is about the collapse and not about the
        // exact epoch count.
        assert!(bounded > 30.0 * saturated, "bounded {bounded}, saturated {saturated}");
        assert!(saturated > 0.0, "the saturated Fisher hit exactly zero, which is a different bug");
    }

    /// The synaptic-intelligence normalisation against its own formula, including the clamp —
    /// which is the one thing a sequence-level test cannot reach, because a negative path integral
    /// has to be constructed rather than waited for.
    #[test]
    fn path_importance_is_its_formula_and_refuses_a_negative_importance() {
        // The formula, exactly.
        assert_eq!(path_importance(0.8, 0.5, 0.1), Some(0.8 / (0.25 + 0.1)));
        assert_eq!(path_importance(0.0, 3.0, 0.1), Some(0.0));
        // A parameter that barely moved is damped rather than made infinite.
        assert_eq!(path_importance(1.0, 0.0, 0.1), Some(10.0));
        // THE CLAMP. A negative path integral would be a penalty pushing the parameter AWAY from
        // the anchor, which is worse than no penalty at all.
        assert_eq!(path_importance(-5.0, 0.5, 0.1), Some(0.0));
        // And the refusals, because an infinity here poisons every later step silently.
        assert_eq!(path_importance(1.0, 0.5, 0.0), None);
        assert_eq!(path_importance(1.0, 0.5, -0.1), None);
        assert_eq!(path_importance(f64::NAN, 0.5, 0.1), None);
        assert_eq!(path_importance(1.0, f64::INFINITY, 0.1), None);
        assert_eq!(path_importance(1.0, 0.5, f64::NAN), None);
    }

    /// THE MUTATION THAT SURVIVED, turned into an invariant. Conditioning
    /// [`Cascade::learning_probability`] on the *plus* states instead of the *minus* ones changes
    /// no number this module can produce, because the stationary distribution is exactly symmetric
    /// under polarity — `q0`, `p0` and `x` are shared by both signs, so there is nothing to break
    /// the symmetry. That is asserted here rather than left as an accident, so that anyone adding
    /// per-polarity probabilities later finds out immediately which of the two halves the
    /// conditioning meant.
    #[test]
    fn the_stationary_distribution_is_exactly_symmetric_under_polarity() {
        for depth in 1..=6 {
            for &(q0, p0, x) in &[(1.0, 1.0, 0.5), (0.4, 0.8, 0.7)] {
                let pi = Cascade { depth, q0, p0, x }.stationary().unwrap();
                for k in 0..depth {
                    assert!(
                        (pi[k] - pi[depth + k]).abs() < 1e-15,
                        "depth {depth} level {k}: {} vs {}",
                        pi[k],
                        pi[depth + k]
                    );
                }
            }
        }
    }

    /// Rows of a transition matrix are probability distributions. A cascade that leaked probability
    /// would decay for the wrong reason and every curve above it would still look plausible.
    #[test]
    fn transition_rows_sum_to_one() {
        for depth in 1..=6 {
            for &(q0, p0, x) in &[(1.0, 1.0, 0.5), (0.3, 0.7, 0.9), (1.0, 1.0, 1.0)] {
                let c = Cascade { depth, q0, p0, x };
                let m = c.transition_matrix().unwrap();
                let n = 2 * depth;
                for i in 0..n {
                    let s: f64 = m[i * n..(i + 1) * n].iter().sum();
                    assert!((s - 1.0).abs() < 1e-15, "row {i} of depth {depth} sums to {s}");
                    assert!(m[i * n..(i + 1) * n].iter().all(|&v| v >= 0.0));
                }
                // A stationary distribution is one too.
                let pi = c.stationary().unwrap();
                assert!((pi.iter().sum::<f64>() - 1.0).abs() < 1e-12);
                assert!(pi.iter().all(|&v| v >= 0.0));
            }
        }
    }

    /// THE MODEL'S ENTIRE CLAIM, both sides of it, measured on the same object: retention rises
    /// with depth while the learning rate falls, strictly, at every step.
    #[test]
    fn the_cascade_trades_learning_rate_for_retention() {
        let mut lifetimes = Vec::new();
        let mut rates = Vec::new();
        for depth in 1..=6 {
            let c = Cascade::fusi_2005(depth);
            lifetimes.push(c.lifetime(0.01, 100_000).unwrap().unwrap());
            rates.push(c.learning_probability().unwrap());
        }
        for i in 1..lifetimes.len() {
            assert!(lifetimes[i] > lifetimes[i - 1], "depth {} did not outlast {i}: {lifetimes:?}", i + 1);
            assert!(rates[i] < rates[i - 1], "depth {} did not learn slower: {rates:?}", i + 1);
        }
        // And the trade is worth taking in this range: six levels buy eighty times the lifetime for
        // a factor of three and a half in learning rate.
        assert!(lifetimes[5] > 50 * lifetimes[0], "{lifetimes:?}");
        assert!(rates[5] > rates[0] / 4.0, "{rates:?}");
        // Four bits of state at depth 8, against the 64 one f64 importance value costs.
        assert_eq!(Cascade::fusi_2005(8).bits_per_synapse(), 4);
        assert_eq!(Cascade::fusi_2005(5).bits_per_synapse(), 4);
        assert_eq!(Cascade::fusi_2005(1).bits_per_synapse(), 1);
    }

    /// THE PART NOBODY QUOTES. Depth is not free retention: past an optimum the memory starts below
    /// the floor it would have had to decay to, and the lifetime falls again. The optimal depth is a
    /// function of the detection floor, and it moves the way it must.
    #[test]
    fn deeper_is_not_unconditionally_better() {
        let best = |floor: f64| -> usize {
            (1..=12)
                .max_by_key(|&d| Cascade::fusi_2005(d).lifetime(floor, 1_000_000).unwrap().unwrap())
                .unwrap()
        };
        let (coarse, middle, fine) = (best(0.05), best(0.01), best(0.005));
        assert!(coarse < middle, "optimal depth at 0.05 is {coarse}, at 0.01 is {middle}");
        assert!(middle < fine, "optimal depth at 0.01 is {middle}, at 0.005 is {fine}");
        assert!(fine < 12, "the optimum ran off the end of the sweep at {fine}");
        // Explicitly: a depth-12 cascade is WORSE than a depth-4 one at a floor of 0.05.
        let deep = Cascade::fusi_2005(12).lifetime(0.05, 100_000).unwrap().unwrap();
        let shallow = Cascade::fusi_2005(4).lifetime(0.05, 100_000).unwrap().unwrap();
        assert!(deep < shallow, "depth 12 lasted {deep}, depth 4 lasted {shallow}");
    }

    /// What depth buys, at the tail where it is supposed to buy it: 128 stimuli after storage a
    /// depth-8 cascade still carries a signal five hundred times a depth-4 one's, and a depth-2
    /// cascade has none left at all.
    #[test]
    fn the_cascade_tail_is_what_depth_buys() {
        let s2 = Cascade::fusi_2005(2).signal_after(128).unwrap();
        let s4 = Cascade::fusi_2005(4).signal_after(128).unwrap();
        let s8 = Cascade::fusi_2005(8).signal_after(128).unwrap();
        assert!(s2 < 1e-12, "depth 2 at t=128: {s2}");
        assert!(s4 > 1e-6 && s4 < 1e-4, "depth 4 at t=128: {s4}");
        assert!(s8 > 500.0 * s4, "depth 8 {s8} against depth 4 {s4}");
        // Monotone in t it is NOT: with q0 = 1 a freshly flipped synapse is flipped straight back by
        // the next opposing stimulus, so the signal dips at t = 1 and recovers at t = 2 as the
        // survivors deepen. Stated because it makes `lifetime` a first crossing rather than a
        // threshold on a monotone curve.
        let c = Cascade::fusi_2005(4);
        assert!(c.signal_after(1).unwrap() < c.signal_after(2).unwrap());
    }

    /// The sampled ensemble must reproduce the exact chain. Both start from the same stationary
    /// distribution, so this compares dynamics rather than burn-in, and it is what would catch a
    /// transition applied to the wrong polarity.
    #[test]
    fn the_sampled_ensemble_matches_the_exact_chain() {
        let c = Cascade::fusi_2005(4);
        let n = 40_000;
        for &t in &[0usize, 1, 3, 10, 40] {
            let mut ens = CascadeEnsemble::from_stationary(c, n, 0xCA5C_ADE0 ^ t as u64).unwrap();
            ens.stimulate(true);
            ens.random_stimuli(t);
            let want = c.signal_after(t).unwrap();
            let got = ens.signal();
            // About four standard errors of a mean polarity over 40,000 synapses.
            assert!((got - want).abs() < 0.02, "t={t}: sampled {got} against exact {want}");
        }
        assert_eq!(CascadeEnsemble::new(c, 7, 1).unwrap().len(), 7);
        assert!(!CascadeEnsemble::new(c, 7, 1).unwrap().is_empty());
        // A fresh ensemble is all-plus at level zero, which is the initial condition the chain does
        // NOT use — see the doc on `signal_after`.
        assert_eq!(CascadeEnsemble::new(c, 100, 1).unwrap().signal(), 1.0);
    }

    #[test]
    fn a_cascade_refuses_impossible_parameters() {
        assert!(matches!(
            Cascade { depth: 0, ..Cascade::fusi_2005(1) }.validate(),
            Err(ContinualError::Empty { .. })
        ));
        assert!(matches!(
            Cascade { x: 1.5, ..Cascade::fusi_2005(3) }.signal_after(1),
            Err(ContinualError::OutOfRange { what: "x", .. })
        ));
        assert!(matches!(
            Cascade { q0: f64::NAN, ..Cascade::fusi_2005(3) }.stationary(),
            Err(ContinualError::NotFinite { what: "q0", .. })
        ));
        assert!(matches!(
            Cascade::fusi_2005(3).lifetime(2.0, 10),
            Err(ContinualError::OutOfRange { what: "floor", .. })
        ));
        assert_eq!(Cascade::fusi_2005(3).flip_probability(3), None);
        assert_eq!(Cascade::fusi_2005(3).flip_probability(1), Some(0.5));
        assert_eq!(Cascade::fusi_2005(3).deepen_probability(2), Some(0.0));
        assert_eq!(Cascade::fusi_2005(3).deepen_probability(3), None);
        // A memory that has not decayed inside the observation window has a lifetime nobody
        // measured, and `None` says so. With deepening certain and flipping at 1e-9, nearly all
        // mass sits at the deepest level; the potentiating stimulus flips its `1.25e-10` share to
        // level 0, and deepening preserves polarity, so the signal starts at ~1.25e-10 and stays
        // there — far above a 1e-12 floor after five stimuli.
        //
        // ⛔ THIS ASSERTION WAS RIGHT ALL ALONG, AND WAS BRIEFLY CHANGED TO `Some(0)` ON THE BASIS
        // OF A WRONG NUMBER. An intermediate stationary solver (Gaussian elimination, refined but
        // not yet symmetrised) carried an antisymmetric error of ~1e-11, which swamped a level-0
        // mass of order 1e-10 and produced a signal below the floor. The exact flux-balance
        // solution restores the original answer, and the magnitude is now pinned so that the
        // difference between 1e-10 and 1e-18 can never again read as a passing test.
        let slow = Cascade { depth: 4, q0: 1e-9, p0: 1.0, x: 0.5 };
        let s0 = slow.signal_after(0).unwrap();
        assert!(s0 > 1e-12 && s0 < 1e-9, "signal(0) = {s0}; expected order 1e-10");
        assert_eq!(slow.lifetime(1e-12, 5).unwrap(), None);
        assert_eq!(Cascade { depth: 4, q0: 0.5, p0: 0.5, x: 0.5 }.lifetime(1e-12, 5).unwrap(), None);
        assert!(matches!(
            CascadeEnsemble::new(Cascade::fusi_2005(2), 0, 1),
            Err(ContinualError::Empty { .. })
        ));
    }

    // -----------------------------------------------------------------------------------------
    // Homeostasis: a real mechanism that buys nothing here, which is a result.
    // -----------------------------------------------------------------------------------------

    /// The property [`crate::plasticity::SynapticScaling`] proves on the spiking side: scaling
    /// changes gain, not preference.
    #[test]
    fn scaling_preserves_every_within_row_ratio() {
        let mut m = Linear::new(4, 3).unwrap();
        for (i, v) in m.w.iter_mut().enumerate() {
            *v = 0.3 * (i as f64 + 1.0) - 1.1;
        }
        m.b = vec![0.7, -0.2, 0.4];
        let before = m.clone();
        let g = Homeostat { target_norm: 1.0, rate: 1.0 }.apply(&mut m).unwrap();
        assert!(g.iter().any(|&f| (f - 1.0).abs() > 0.1), "the homeostat did nothing: {g:?}");
        for c in 0..3 {
            for k in 0..4 {
                assert!((m.w[c * 4 + k] - before.w[c * 4 + k] * g[c]).abs() < 1e-15);
            }
            let norm = (m.w[c * 4..(c + 1) * 4].iter().map(|v| v * v).sum::<f64>()
                + m.b[c] * m.b[c])
                .sqrt();
            assert!((norm - 1.0).abs() < 1e-12, "row {c} norm {norm}");
        }
    }

    /// When every row is already the same size the homeostat applies one common factor, and the
    /// decision function is untouched on every sample.
    #[test]
    fn a_uniform_rescaling_changes_no_prediction() {
        let mut m = Linear::new(2, 2).unwrap();
        m.w = vec![3.0, 4.0, -4.0, 3.0];
        m.b = vec![0.0, 0.0];
        let xs = [[0.4, -0.25], [-0.1, 0.9], [0.0, 0.0]];
        let before: Vec<usize> = xs.iter().map(|x| m.predict(x).unwrap()).collect();
        let g = Homeostat { target_norm: 2.5, rate: 1.0 }.apply(&mut m).unwrap();
        assert!((g[0] - g[1]).abs() < 1e-15, "rows of equal norm got different factors");
        assert!((g[0] - 0.5).abs() < 1e-15, "factor {g:?} against a norm of 5 and a target of 2.5");
        let after: Vec<usize> = xs.iter().map(|x| m.predict(x).unwrap()).collect();
        assert_eq!(before, after);
    }

    /// THE HONEST NEGATIVE. Forgetting is a change of direction in weight space and scaling changes
    /// only magnitude, so a homeostat cannot undo it in principle — and does not. Asserted as a
    /// two-sided band, and paired with a check that it is not simply a no-op, which is the mutation
    /// a one-sided "it does not help" assertion would happily survive.
    #[test]
    fn homeostasis_is_not_a_cure_and_is_not_a_no_op() {
        let sched = Schedule::default();
        let plain = run(&course(5), &Protocol::plain(sched)).unwrap();
        let homeo = run(
            &course(5),
            &Protocol {
                homeostasis: Some(Homeostat { target_norm: 6.0, rate: 1.0 }),
                ..Protocol::plain(sched)
            },
        )
        .unwrap();
        let delta = homeo.backward_transfer().unwrap() - plain.backward_transfer().unwrap();
        assert!(delta.abs() < 0.10, "homeostasis moved backward transfer by {delta}");
        assert_ne!(homeo.r, plain.r, "the homeostat never touched the model");
        assert_eq!(homeo.cost.extra_floats, 0);
    }

    #[test]
    fn a_homeostat_refuses_a_target_that_is_not_a_norm() {
        let mut m = Linear::new(2, 2).unwrap();
        assert!(matches!(
            Homeostat { target_norm: 0.0, rate: 1.0 }.apply(&mut m),
            Err(ContinualError::OutOfRange { what: "target_norm", .. })
        ));
        assert!(matches!(
            Homeostat { target_norm: 1.0, rate: 2.0 }.apply(&mut m),
            Err(ContinualError::OutOfRange { what: "rate", .. })
        ));
        // A zero row has no direction to preserve and is left alone rather than given an arbitrary
        // one.
        let g = Homeostat { target_norm: 1.0, rate: 1.0 }.apply(&mut m).unwrap();
        assert_eq!(g, vec![1.0, 1.0]);
        assert_eq!(m.w, vec![0.0; 4]);
    }

    // -----------------------------------------------------------------------------------------
    // Replay machinery, checked on its own.
    // -----------------------------------------------------------------------------------------

    /// Vitter's reservoir guarantee is `k / n` for every item regardless of arrival order. A buffer
    /// that kept the first `k` or the last `k` would still look fine in the retention table and
    /// would be a different method.
    #[test]
    fn the_replay_buffer_is_a_uniform_sample_of_everything_it_saw() {
        let stream: Vec<Example> =
            (0..50).map(|i| Example { x: vec![f64::from(i)], label: 0 }).collect();
        let mut counts = [0u32; 50];
        let trials = 4000u32;
        for trial in 0..trials {
            let mut store = super::ReplayStore::new(Replay::Experience { budget: 10 }, 1);
            let mut rng = Rng::new(0x5EED_0000 + u64::from(trial));
            store.absorb(&stream, 1, &mut rng);
            assert_eq!(store.buffer.len(), 10);
            for e in &store.buffer {
                counts[e.x[0] as usize] += 1;
            }
        }
        let expect = f64::from(trials) * 10.0 / 50.0;
        for (i, &c) in counts.iter().enumerate() {
            let rel = (f64::from(c) - expect).abs() / expect;
            assert!(rel < 0.12, "item {i} kept {c} times against {expect}");
        }
    }

    /// Box-Muller against the closed form: mean 0, variance 1, and the 68.27% of a standard normal
    /// inside one standard deviation.
    #[test]
    fn standard_normal_matches_the_gaussian_it_claims_to_be() {
        let mut rng = Rng::new(0xB0C5_0001);
        let n = 200_000u32;
        let (mut s, mut s2, mut inside) = (0.0f64, 0.0f64, 0u32);
        for _ in 0..n {
            let z = standard_normal(&mut rng);
            assert!(z.is_finite());
            s += z;
            s2 += z * z;
            if z.abs() < 1.0 {
                inside += 1;
            }
        }
        let mean = s / f64::from(n);
        let var = s2 / f64::from(n) - mean * mean;
        assert!(mean.abs() < 0.01, "mean {mean}");
        assert!((var - 1.0).abs() < 0.02, "variance {var}");
        let frac = f64::from(inside) / f64::from(n);
        assert!((frac - 0.682_689_5).abs() < 0.005, "one-sigma mass {frac}");
    }

    // -----------------------------------------------------------------------------------------
    // Determinism, boundaries and refusals.
    // -----------------------------------------------------------------------------------------

    #[test]
    fn the_same_seed_gives_the_same_run() {
        let c = course(3);
        let p = Protocol {
            consolidation: Consolidation::Si { c: 0.003, xi: 0.1 },
            replay: Replay::Experience { budget: 60 },
            homeostasis: Some(Homeostat { target_norm: 6.0, rate: 0.5 }),
            schedule: Schedule::default(),
        };
        assert_eq!(run(&c, &p).unwrap(), run(&c, &p).unwrap());
    }

    #[test]
    fn a_different_seed_gives_a_different_run() {
        let c = course(2);
        let a = run(&c, &Protocol::plain(Schedule { seed: 1, ..Schedule::default() })).unwrap();
        let b = run(&c, &Protocol::plain(Schedule { seed: 2, ..Schedule::default() })).unwrap();
        assert_ne!(a.r, b.r);
    }

    /// The baseline row is exactly chance, which is what makes forward transfer a difference from a
    /// known number rather than from a seeded artefact.
    #[test]
    fn the_untrained_baseline_is_exactly_chance() {
        let c = course(3);
        let res = run(&c, &Protocol::plain(Schedule::default())).unwrap();
        for (j, &b) in res.baseline.iter().enumerate() {
            assert_eq!(b, c.tasks[j].chance, "baseline on task {j}");
        }
        assert_eq!(c.tasks[0].chance, 0.2);
    }

    /// Permutations must be pairwise deranged; a shared feature slot is a slot that does not have
    /// to change, and it would soften the collapse for a reason that has nothing to do with the
    /// model.
    #[test]
    fn permutations_share_no_feature_slot() {
        let c = course(5);
        assert_eq!(c.len(), 5);
        assert!(!c.is_empty());
        assert_eq!(c.tasks[0].permutation, (0..8).collect::<Vec<_>>());
        assert_eq!(c.tasks[0].n_features(), 8);
        for i in 0..5 {
            let mut sorted = c.tasks[i].permutation.clone();
            sorted.sort_unstable();
            assert_eq!(sorted, (0..8).collect::<Vec<_>>(), "task {i} is not a permutation");
            for j in i + 1..5 {
                for k in 0..8 {
                    assert_ne!(
                        c.tasks[i].permutation[k], c.tasks[j].permutation[k],
                        "tasks {i} and {j} share slot {k}"
                    );
                }
            }
        }
    }

    /// The permutation has to reach the features, not merely be recorded beside them.
    #[test]
    fn the_permutation_is_applied_to_the_features() {
        let c = uncentred(2);
        let d = base();
        let raw = latency_features(&d.split(Split::Train)[0], d.n_inputs, d.ticks).unwrap();
        let p = &c.tasks[1].permutation;
        for k in 0..8 {
            assert_eq!(c.tasks[1].train[0].x[k], raw[p[k]]);
        }
        assert_eq!(c.tasks[0].train[0].x, raw);
    }

    /// With two features there are two permutations and only one of them deranges the identity, so
    /// a third task cannot exist and the generator says so rather than repeating one.
    #[test]
    fn the_curriculum_refuses_more_tasks_than_derangements() {
        let d = LatencyPatterns {
            n_inputs: 2,
            n_classes: 2,
            min_separation_ticks: 6,
            per_class_train: 4,
            per_class_test: 4,
            ..LatencyPatterns::default()
        }
        .generate()
        .unwrap();
        assert!(Curriculum::permuted(&d, 2, 7).is_ok());
        assert!(matches!(
            Curriculum::permuted(&d, 3, 7),
            Err(ContinualError::Exhausted { wanted: 3, found: 2, .. })
        ));
        assert!(matches!(
            Curriculum::permuted(&base(), 0, 1),
            Err(ContinualError::Empty { what: "n_tasks" })
        ));
        assert!(matches!(
            Curriculum::from_examples("x", Vec::new(), Vec::new(), 2, 1, 1),
            Err(ContinualError::Empty { what: "train split" })
        ));
        let one = vec![Example { x: vec![0.0, 1.0], label: 0 }];
        assert!(matches!(
            Curriculum::from_examples("x", one.clone(), vec![Example { x: vec![0.0], label: 0 }], 2, 1, 1),
            Err(ContinualError::Mismatch { what: "feature vector", .. })
        ));
        assert!(matches!(
            Curriculum::from_examples("x", one.clone(), vec![Example { x: vec![0.0, 0.0], label: 5 }], 2, 1, 1),
            Err(ContinualError::Label { label: 5, n_classes: 2 })
        ));
    }

    #[test]
    fn features_refuse_a_silent_channel_rather_than_imputing_one() {
        use crate::spike::{Spike, Train};
        let s = Sample { train: Train::from_spikes(vec![Spike { t: 3, source: 0 }]), label: 0 };
        assert_eq!(latency_features(&s, 1, 50).unwrap(), vec![3.0 / 49.0 - 0.5]);
        assert!(matches!(
            latency_features(&s, 2, 50),
            Err(ContinualError::SilentChannel { channel: 1 })
        ));
        assert!(matches!(latency_features(&s, 0, 50), Err(ContinualError::Empty { .. })));
        assert!(matches!(latency_features(&s, 1, 1), Err(ContinualError::Empty { .. })));
        // FIRST spike, not last. Every task in `crate::tasks` that this module reads fires each
        // channel exactly once, so a `last`-spike implementation would agree on all of them and
        // this is the only place the difference shows.
        let twice = Sample {
            train: Train::from_spikes(vec![
                Spike { t: 3, source: 0 },
                Spike { t: 40, source: 0 },
            ]),
            label: 0,
        };
        assert_eq!(latency_features(&twice, 1, 50).unwrap(), vec![3.0 / 49.0 - 0.5]);
        // The window ends are the ends of the range, exactly.
        let ends = Sample {
            train: Train::from_spikes(vec![Spike { t: 0, source: 0 }, Spike { t: 49, source: 1 }]),
            label: 0,
        };
        assert_eq!(latency_features(&ends, 2, 50).unwrap(), vec![-0.5, 0.5]);
    }

    #[test]
    fn the_learner_refuses_shapes_and_values_it_cannot_use() {
        let m = Linear::new(3, 2).unwrap();
        assert_eq!(m.parameters(), 8);
        assert_eq!(m.n_features(), 3);
        assert_eq!(m.n_classes(), 2);
        assert!(matches!(
            m.logits(&[0.0, 1.0]),
            Err(ContinualError::Mismatch { expected: 3, found: 2, .. })
        ));
        assert!(matches!(
            m.logits(&[0.0, f64::NAN, 1.0]),
            Err(ContinualError::NotFinite { what: "feature", .. })
        ));
        assert!(matches!(Linear::new(0, 2), Err(ContinualError::Empty { .. })));
        assert!(matches!(Linear::new(3, 1), Err(ContinualError::Empty { .. })));
        assert!(matches!(m.accuracy(&[]), Err(ContinualError::Metric(_))));
        assert!(matches!(m.loss(&[]), Err(ContinualError::Empty { .. })));
        assert!(matches!(
            m.loss(&[Example { x: vec![0.0; 3], label: 9 }]),
            Err(ContinualError::Label { label: 9, n_classes: 2 })
        ));
        // A zero-initialised softmax over k classes has loss ln(k) exactly, and ties go to the
        // lowest index.
        let l = m.loss(&[Example { x: vec![0.1, 0.2, 0.3], label: 1 }]).unwrap();
        assert!((l - 2.0f64.ln()).abs() < 1e-15, "{l}");
        assert_eq!(m.predict(&[0.1, 0.2, 0.3]).unwrap(), 0);
        let p = m.probabilities(&[0.1, 0.2, 0.3]).unwrap();
        assert_eq!(p, vec![0.5, 0.5]);
    }

    /// `w` and `b` are public, so they can be resized. Every read path has to REFUSE that rather
    /// than index off the end of a slice — a library that panics on a value its own API handed out
    /// is the exact defect this crate audits for.
    #[test]
    fn a_resized_weight_vector_is_refused_rather_than_indexed_off_the_end() {
        let mut m = Linear::new(3, 2).unwrap();
        m.w.pop();
        assert!(matches!(
            m.check_shape(),
            Err(ContinualError::Mismatch { what: "weight matrix", expected: 6, found: 5 })
        ));
        assert!(m.logits(&[0.0, 0.0, 0.0]).is_err());
        assert!(m.predict(&[0.0, 0.0, 0.0]).is_err());
        assert!(Homeostat { target_norm: 1.0, rate: 1.0 }.apply(&mut m).is_err());
        let mut m = Linear::new(3, 2).unwrap();
        m.b.push(0.0);
        assert!(matches!(
            m.check_shape(),
            Err(ContinualError::Mismatch { what: "bias vector", expected: 2, found: 3 })
        ));
    }

    /// And a ragged accuracy matrix, which `run` cannot produce but a caller filling the public
    /// fields can.
    #[test]
    fn a_ragged_accuracy_matrix_refuses_rather_than_panicking() {
        let missing_diagonal = Results {
            r: vec![Vec::new(), vec![0.5, 0.8]],
            baseline: vec![0.2, 0.2],
            cost: Cost::default(),
        };
        assert_eq!(missing_diagonal.backward_transfer(), None);
        let short_last_row = Results {
            r: vec![vec![0.9, 0.3], Vec::new()],
            baseline: vec![0.2, 0.2],
            cost: Cost::default(),
        };
        assert_eq!(short_last_row.backward_transfer(), None);
        let short_earlier_row = Results {
            r: vec![vec![0.9], vec![0.5, 0.8]],
            baseline: vec![0.2, 0.2],
            cost: Cost::default(),
        };
        assert_eq!(short_earlier_row.forward_transfer(), None);
        let short_baseline = Results {
            r: vec![vec![0.9, 0.3], vec![0.5, 0.8]],
            baseline: vec![0.2],
            cost: Cost::default(),
        };
        assert_eq!(short_baseline.forward_transfer(), None);
    }

    /// Softmax on logits large enough to overflow `exp`. The max-subtraction has to be there, and
    /// without it this is `NaN` rather than a probability.
    #[test]
    fn the_softmax_survives_logits_that_would_overflow() {
        let mut m = Linear::new(2, 3).unwrap();
        m.w = vec![1e6, 0.0, 0.0, 1e6, -1e6, -1e6];
        let p = m.probabilities(&[1.0, 1.0]).unwrap();
        assert!(p.iter().all(|v| v.is_finite()));
        assert_eq!(p, vec![0.5, 0.5, 0.0]);
        assert_eq!(m.predict(&[1.0, 1.0]).unwrap(), 0);
    }

    #[test]
    fn a_run_refuses_a_schedule_or_a_curriculum_it_cannot_use() {
        let c = course(2);
        assert!(matches!(
            run(&c, &Protocol::plain(Schedule { epochs: 0, ..Schedule::default() })),
            Err(ContinualError::Empty { what: "epochs" })
        ));
        assert!(matches!(
            run(&c, &Protocol::plain(Schedule { lr: -1.0, ..Schedule::default() })),
            Err(ContinualError::OutOfRange { what: "lr", .. })
        ));
        assert!(matches!(
            run(&c, &Protocol::plain(Schedule { weight_decay: f64::NAN, ..Schedule::default() })),
            Err(ContinualError::NotFinite { what: "weight_decay", .. })
        ));
        assert!(matches!(
            run(&Curriculum { tasks: Vec::new() }, &Protocol::plain(Schedule::default())),
            Err(ContinualError::Empty { what: "curriculum" })
        ));
        let mut ragged = c.clone();
        ragged.tasks[1].permutation.pop();
        assert!(matches!(
            run(&ragged, &Protocol::plain(Schedule::default())),
            Err(ContinualError::Mismatch { what: "task feature width", .. })
        ));
        let mut classes = c.clone();
        classes.tasks[1].n_classes = 4;
        assert!(matches!(
            run(&classes, &Protocol::plain(Schedule::default())),
            Err(ContinualError::Mismatch { what: "task class count", .. })
        ));
        let mut empty = c.clone();
        empty.tasks[1].test.clear();
        assert!(matches!(
            run(&empty, &Protocol::plain(Schedule::default())),
            Err(ContinualError::Empty { what: "task test split" })
        ));
        let mut notrain = c.clone();
        notrain.tasks[0].train.clear();
        assert!(matches!(
            run(&notrain, &Protocol::plain(Schedule::default())),
            Err(ContinualError::Empty { what: "task train split" })
        ));
    }

    #[test]
    fn errors_say_what_was_wrong() {
        let e = ContinualError::SilentChannel { channel: 3 };
        assert!(e.to_string().contains("channel 3"));
        let e = ContinualError::Mismatch { what: "x", expected: 2, found: 5 };
        assert!(e.to_string().contains("expected 2, found 5"));
        let e = ContinualError::Exhausted { wanted: 3, found: 2, draws: 9 };
        assert!(e.to_string().contains("deranged"));
        let e = ContinualError::Label { label: 4, n_classes: 2 };
        assert!(e.to_string().contains("past the 2"));
        let e = ContinualError::NotFinite { what: "lr", value: f64::INFINITY };
        assert!(e.to_string().contains("lr is not finite"));
        let e = ContinualError::OutOfRange { what: "x", value: 2.0, low: 0.0, high: 1.0 };
        assert!(e.to_string().contains("[0, 1]"));
        let e = ContinualError::Empty { what: "epochs" };
        assert!(e.to_string().contains("epochs is empty"));
        let e = ContinualError::Diverged { parameter: 40, value: f64::INFINITY };
        assert!(e.to_string().contains("parameter 40 diverged"));
        let e = ContinualError::NoStationary { residual: 1e-3 };
        assert!(e.to_string().contains("did not settle"));
    }

    // -----------------------------------------------------------------------------------------
    // (f) The claims this module states and the suite above could not see.
    //
    // Every test in this section was written against a SURVIVING mutation: an edit that changed a
    // documented claim and that no test then failed on. Each one names the shape of the hole it
    // fills, because that sentence is the part of it worth reading.
    // -----------------------------------------------------------------------------------------

    /// A spike from a channel the caller did not declare is REFUSED rather than dropped.
    ///
    /// [`latency_features`] refuses the mirror-image case — a declared channel that carried no
    /// spike is a [`ContinualError::SilentChannel`] rather than an imputed number — and until this
    /// test a spike past the declared width was silently discarded instead, which is the same
    /// failure in the other direction. No fixture could see it: every dataset in [`crate::tasks`]
    /// declares its own channel count and [`Curriculum::permuted`] passes that same number, so no
    /// sample in this module ever carried a source at or past `n_inputs`.
    #[test]
    fn a_spike_from_an_undeclared_channel_is_refused_rather_than_dropped() {
        use crate::spike::{Spike, Train};
        let s = Sample {
            train: Train::from_spikes(vec![
                Spike { t: 3, source: 0 },
                Spike { t: 7, source: 1 },
                Spike { t: 11, source: 2 },
            ]),
            label: 0,
        };
        // Three channels declared, three carried: a span of 50 ticks and the centring, exactly.
        assert_eq!(
            latency_features(&s, 3, 51).unwrap(),
            vec![3.0 / 50.0 - 0.5, 7.0 / 50.0 - 0.5, 11.0 / 50.0 - 0.5]
        );
        // Two declared, three carried: the third is not a channel of this feature vector.
        assert!(matches!(
            latency_features(&s, 2, 51),
            Err(ContinualError::Mismatch { what: "spike source", expected: 2, found: 3 })
        ));
    }

    /// The three guards [`Curriculum::from_examples`] lists in its own `# Errors` section that no
    /// fixture reached: an empty TEST split, a zero-width feature vector, and a label exactly AT
    /// the class count.
    ///
    /// The existing refusal fixture hands in an empty TRAIN split, which returns from the first
    /// guard and so answers for the two behind it; and its bad label is 5 against two classes,
    /// three past the boundary, so a bound written `>` instead of `>=` still refuses it.
    #[test]
    fn from_examples_reaches_every_guard_it_documents() {
        let train = vec![Example { x: vec![0.0, 1.0], label: 0 }];
        let test = vec![Example { x: vec![1.0, 0.0], label: 1 }];
        assert!(matches!(
            Curriculum::from_examples("g", train.clone(), Vec::new(), 2, 1, 1),
            Err(ContinualError::Empty { what: "test split" })
        ));
        // Both splits present, every vector the same length — and that length is zero.
        let widthless = vec![Example { x: Vec::new(), label: 0 }];
        assert!(matches!(
            Curriculum::from_examples("g", widthless.clone(), widthless, 2, 1, 1),
            Err(ContinualError::Empty { what: "n_features" })
        ));
        // AT the class count, not past it: label 2 names no output unit of a two-class readout.
        assert!(matches!(
            Curriculum::from_examples(
                "g",
                train.clone(),
                vec![Example { x: vec![0.0, 0.0], label: 2 }],
                2,
                1,
                1
            ),
            Err(ContinualError::Label { label: 2, n_classes: 2 })
        ));
        // And the same bound on the train side.
        assert!(matches!(
            Curriculum::from_examples(
                "g",
                vec![Example { x: vec![0.0, 0.0], label: 2 }],
                test.clone(),
                2,
                1,
                1
            ),
            Err(ContinualError::Label { label: 2, n_classes: 2 })
        ));
        // Not a blanket refusal: the same call with all three fixed builds a curriculum.
        assert!(Curriculum::from_examples("g", train, test, 2, 1, 1).is_ok());
    }

    /// [`Task::name`] carries the task's position in the sequence, which is the only thing that
    /// makes a row of an accuracy matrix attributable to a task. Nothing in this module reads a
    /// name, so every task could have carried the same one and every number would be unchanged.
    #[test]
    fn every_task_name_carries_its_position_in_the_sequence() {
        let c = course(3);
        let names: Vec<&str> = c.tasks.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(
            names,
            vec!["latency_patterns/perm0", "latency_patterns/perm1", "latency_patterns/perm2"]
        );
        let built = Curriculum::from_examples(
            "hand",
            vec![Example { x: vec![0.0, 1.0], label: 0 }],
            vec![Example { x: vec![1.0, 0.0], label: 1 }],
            2,
            2,
            9,
        )
        .unwrap();
        assert_eq!(built.tasks[0].name, "hand/perm0");
        assert_eq!(built.tasks[1].name, "hand/perm1");
    }

    /// The test split is featureised from the dataset's TEST split.
    ///
    /// Every task in this module is learned to 1.000 on both splits, so a run scored on the
    /// training data reports the same retention numbers and the same collapse; the two splits
    /// separate only on their own contents and their own sizes.
    #[test]
    fn the_curriculum_featureises_the_test_split_from_the_test_split() {
        let d = base();
        let c = course(1);
        // A hundred train and fifty test samples per class, five classes.
        assert_eq!(c.tasks[0].train.len(), 5 * 100);
        assert_eq!(c.tasks[0].test.len(), 5 * 50);
        let first = &d.split(Split::Test)[0];
        let mut want = vec![Example {
            x: latency_features(first, d.n_inputs, d.ticks).unwrap(),
            label: first.label,
        }];
        // `remove_sample_mean` works one example at a time, so a one-element slice gives the same
        // bits as the whole split does.
        remove_sample_mean(&mut want);
        assert_eq!(c.tasks[0].test[0], want[0]);
        // The two splits are disjoint, so this is not the same example arriving twice.
        assert_ne!(c.tasks[0].test[0].x, c.tasks[0].train[0].x);
    }

    /// The cross-entropy is a MEAN over the examples, as its doc says in nats.
    ///
    /// [`Linear::loss`] is reached with exactly one example everywhere else in this suite, and a
    /// sum of one term divided by one is that term: the normaliser was invisible.
    #[test]
    fn the_cross_entropy_is_a_mean_over_the_examples_and_not_a_sum() {
        let mut m = Linear::new(2, 2).unwrap();
        m.w = vec![0.5, -0.25, -1.0, 0.75];
        m.b = vec![0.125, -0.5];
        let a = Example { x: vec![0.6, -0.2], label: 0 };
        let b = Example { x: vec![-0.4, 0.9], label: 1 };
        let la = m.loss(std::slice::from_ref(&a)).unwrap();
        let lb = m.loss(std::slice::from_ref(&b)).unwrap();
        assert_ne!(la, lb, "the two examples must differ for a mean to differ from a sum");
        // `acc` accumulates `-ln p` in the same order in both calls and a divisor of 2 is exact,
        // so this is an equality rather than a tolerance.
        assert_eq!(m.loss(&[a, b]).unwrap(), 0.5 * (la + lb));
    }

    /// Synaptic intelligence's two per-task quantities, at the boundary where they are both
    /// finished: the path integral starts again from zero when a task begins, and the
    /// displacement is measured from where that task started rather than from the origin.
    ///
    /// Neither is visible end to end. Every penalised run in this suite has two tasks, the path
    /// integral of the second one is never read by a third, and the first task starts at the
    /// origin — where a displacement from the origin and a displacement from the task start are
    /// the same number.
    #[test]
    fn the_path_integral_and_the_displacement_are_both_measured_per_task() {
        let mut m = Linear::new(1, 2).unwrap();
        let one = [Example { x: vec![0.0], label: 0 }];
        let mut cons = super::Consolidator::new(Consolidation::Si { c: 1.0, xi: 0.1 }, 4);

        // Task one starts at the origin and takes one step of +0.5 against a gradient of −1.
        cons.begin_task(&[0.0; 4]);
        cons.observe_step(&[-1.0, 0.0, 0.0, 0.0], &[0.5, 0.0, 0.0, 0.0]);
        assert_eq!(cons.path[0], 0.5);
        m.w[0] = 0.5;
        cons.end_task(&m, &one).unwrap();
        let first = 0.5 / (0.5 * 0.5 + 0.1);
        assert_eq!(cons.omega[0], first);
        assert_eq!(cons.anchor, vec![0.5, 0.0, 0.0, 0.0]);

        // Task two begins where task one ended.
        cons.begin_task(&[0.5, 0.0, 0.0, 0.0]);
        assert_eq!(cons.path, vec![0.0; 4], "the path integral carried task one into task two");
        cons.observe_step(&[-2.0, 0.0, 0.0, 0.0], &[0.25, 0.0, 0.0, 0.0]);
        assert_eq!(cons.path[0], 0.5);
        m.w[0] = 0.75;
        cons.end_task(&m, &one).unwrap();
        // The displacement is 0.75 − 0.5, not 0.75, and the importances add.
        let second = 0.5 / (0.25 * 0.25 + 0.1);
        assert_eq!(cons.omega[0], first + second);
        assert_eq!(cons.anchor, vec![0.75, 0.0, 0.0, 0.0]);
        // And it paid no pass over the data, which is the trade it makes against EWC.
        assert_eq!(cons.passes, 0);
    }

    /// Elastic weight consolidation SUMS importances across task boundaries against a single
    /// anchor — the online form of Schwarz et al. (ICML 2018) that [`Consolidation::Ewc`] states
    /// as its second deviation from Kirkpatrick et al.
    ///
    /// Every penalised EWC run in this suite has exactly two tasks, and with two tasks the only
    /// penalised task reads one boundary's Fisher whether that boundary added to what was already
    /// there or replaced it.
    #[test]
    fn elastic_importances_are_summed_across_task_boundaries() {
        let m = Linear::new(2, 2).unwrap();
        let a = [Example { x: vec![0.5, -0.25], label: 0 }];
        let b = [Example { x: vec![0.0, 0.75], label: 1 }];
        let fa = fisher_diagonal(&m, &a).unwrap();
        let fb = fisher_diagonal(&m, &b).unwrap();
        assert_ne!(fa, fb, "the two boundaries must differ for a sum to differ from a replacement");
        let mut cons = super::Consolidator::new(Consolidation::Ewc { lambda: 1.0 }, 6);
        cons.end_task(&m, &a).unwrap();
        assert_eq!(cons.omega, fa);
        // The model does not move between the two boundaries, so each Fisher is computable on its
        // own and the accumulated one is their sum, entry for entry and in the same order.
        cons.end_task(&m, &b).unwrap();
        for k in 0..6 {
            assert_eq!(cons.omega[k], fa[k] + fb[k], "parameter {k}");
        }
        // One forward pass per example per boundary, which is what EWC pays and SI does not.
        assert_eq!(cons.passes, 2);
    }

    /// [`standard_normal`] is exactly the Box-Muller transform of its two draws, with the radius
    /// taken from `1 - u` so that it lies in `(0, 1]` and its logarithm is finite.
    ///
    /// Drawing the radius from `[0, 1)` instead changes no moment of the distribution — the two
    /// are the same law — and differs only on a draw of exactly `0.0`, which `Rng::next_f64`
    /// produces with probability `2^-53`. A distributional test cannot wait that long, so the
    /// mean, the variance and the one-sigma mass all passed; the transform itself is pinnable in
    /// sixty-four draws.
    #[test]
    fn standard_normal_is_the_box_muller_transform_of_a_radius_drawn_from_one_minus_u() {
        let mut rng = Rng::new(0xB0C5_0002);
        for i in 0..64 {
            let mut echo = rng; // `Rng` is `Copy`, so this starts at the same stream position.
            let u1 = echo.next_f64();
            let u2 = echo.next_f64();
            let want = (-2.0 * (1.0 - u1).ln()).sqrt() * (core::f64::consts::TAU * u2).cos();
            assert_eq!(standard_normal(&mut rng), want, "draw {i}");
            // And it consumes exactly two draws, which is what keeps the stream position a
            // function of how many samples have been taken.
            assert_eq!(rng, echo, "draw {i}");
        }
    }

    /// The generative replay density is fitted by moments, and both moments are checked here
    /// against hand arithmetic — including the clamp, which no end-to-end run can reach.
    ///
    /// A class whose feature is constant produces a NEGATIVE variance out of the cancellation in
    /// `E[x^2] - E[x]^2`: five copies of 0.7 measure -1.11e-16 in this implementation, and the
    /// square root of that is `NaN` in every sample the generator would then draw. The retention
    /// table never sees it, because no class of [`crate::tasks::LatencyPatterns`] has a constant
    /// feature, and it never sees a spread that is a variance either, because a narrower Gaussian
    /// still separates five well-spaced classes.
    #[test]
    fn the_class_density_is_the_population_mean_and_standard_deviation_of_its_examples() {
        let mut store = super::ReplayStore::new(Replay::Generative { per_task: 2 }, 2);
        let train = vec![
            // Class 0: feature 0 is {1, 0, 1, 0} — mean 1/2, second moment 1/2, variance 1/4 —
            // and feature 1 is {1/4, 1/4, -1/4, -1/4}, mean 0 and variance 1/16. Every step is
            // exact in binary, so these are equalities and not tolerances.
            Example { x: vec![1.0, 0.25], label: 0 },
            Example { x: vec![0.0, 0.25], label: 0 },
            Example { x: vec![1.0, -0.25], label: 0 },
            Example { x: vec![0.0, -0.25], label: 0 },
            // Class 1: constant, which is the case the clamp exists for.
            Example { x: vec![0.7, 0.7], label: 1 },
            Example { x: vec![0.7, 0.7], label: 1 },
            Example { x: vec![0.7, 0.7], label: 1 },
            Example { x: vec![0.7, 0.7], label: 1 },
            Example { x: vec![0.7, 0.7], label: 1 },
        ];
        let mut rng = Rng::new(0x5EED_0007);
        store.absorb(&train, 2, &mut rng);
        assert_eq!(store.models.len(), 2);
        assert_eq!(store.models[0].label, 0);
        assert_eq!(store.models[0].mean, vec![0.5, 0.0]);
        // sqrt(1/4) and sqrt(1/16): a spread is a standard deviation, not a variance.
        assert_eq!(store.models[0].sd, vec![0.5, 0.25]);
        assert_eq!(store.models[1].label, 1);
        assert_eq!(store.models[1].sd, vec![0.0, 0.0], "a negative variance reached the sqrt");
        // A density with no spread generates its own mean, exactly, rather than a `NaN`.
        let mean1 = store.models[1].mean.clone();
        let drawn = store.rehearsal(&mut rng);
        assert_eq!(drawn.len(), 2 * 2);
        for e in drawn.iter().filter(|e| e.label == 1) {
            assert_eq!(e.x, mean1);
        }
    }

    /// [`Homeostat::rate`] is the EXPONENT on the ratio, `(target / norm)^rate`, so 1.0 lands on
    /// the target and 0.0 does nothing at all.
    ///
    /// Every homeostat fixture in this suite passes `rate: 1.0`, and an exponent of one is the
    /// identity on its base; the only other value anywhere is the 0.5 in a test that compares a
    /// run against itself.
    #[test]
    fn the_homeostat_rate_is_the_exponent_on_the_ratio_it_applies() {
        let mut m = Linear::new(3, 2).unwrap();
        // Row 0 is the 3-4-5 triangle with the bias as the 3: its norm is exactly 5. Row 1 is
        // zero, has no direction to preserve, and is left alone.
        m.w = vec![0.0, 4.0, 0.0, 0.0, 0.0, 0.0];
        m.b = vec![3.0, 0.0];
        let g = Homeostat { target_norm: 1.25, rate: 0.5 }.apply(&mut m).unwrap();
        // (1.25 / 5)^0.5 = 0.25^0.5 = 0.5, which is exactly representable and exactly returned.
        assert_eq!(g, vec![0.5, 1.0]);
        assert_eq!(m.w, vec![0.0, 2.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(m.b, vec![1.5, 0.0]);
        // A rate of zero does nothing whatever the target is, which is the other end of the same
        // statement.
        let before = m.clone();
        let g0 = Homeostat { target_norm: 100.0, rate: 0.0 }.apply(&mut m).unwrap();
        assert_eq!(g0, vec![1.0, 1.0]);
        assert_eq!(m, before);
    }

    /// [`evaluate`] scores the TEST split.
    ///
    /// Every task this module builds is learned to 1.000 on both splits, so an accuracy matrix
    /// scored on the training data carries the same numbers; the two separate only on a model
    /// that has not fitted the training set.
    #[test]
    fn evaluate_scores_the_test_split_and_not_the_training_split() {
        // A zero-initialised readout gives every class the same logit and ties go to the lowest
        // index, so it answers class 0 for every input and its accuracy on a split is exactly the
        // fraction of that split labelled 0.
        let c = Curriculum::from_examples(
            "splits",
            vec![
                Example { x: vec![0.0, 1.0], label: 0 },
                Example { x: vec![1.0, 0.0], label: 0 },
            ],
            vec![
                Example { x: vec![0.0, 1.0], label: 1 },
                Example { x: vec![1.0, 0.0], label: 0 },
            ],
            2,
            1,
            3,
        )
        .unwrap();
        let m = Linear::new(2, 2).unwrap();
        assert_eq!(m.accuracy(&c.tasks[0].train).unwrap(), 1.0);
        assert_eq!(evaluate(&m, &c).unwrap(), vec![0.5]);
    }

    /// [`train`] reshuffles the presentation order at every epoch, so the schedule's seed reaches
    /// the weights. The suite's one direct call to it reads only the step count, and a step count
    /// is the same number whatever order the steps arrived in.
    #[test]
    fn the_plain_trainer_reorders_its_examples_and_the_seed_reaches_the_weights() {
        let c = course(1);
        let fit = |seed: u64| {
            let mut m = Linear::new(8, 5).unwrap();
            let steps =
                train(&mut m, &c.tasks[0], &Schedule { seed, epochs: 2, ..Schedule::default() })
                    .unwrap();
            (steps, m)
        };
        let (steps_one, one) = fit(1);
        let (steps_two, two) = fit(2);
        let (_, again) = fit(1);
        assert_eq!(steps_one, 2 * 500);
        assert_eq!(steps_two, 2 * 500);
        assert_eq!(one.w, again.w, "the same seed did not reproduce the run");
        assert_ne!(one.w, two.w, "the ordering never reached the weights");
    }

    /// The two refusals [`train`] documents and no fixture reached: a task with nothing to train
    /// on, and a label that names no output unit of the model being stepped.
    #[test]
    fn the_plain_trainer_refuses_an_empty_task_and_a_label_past_the_class_count() {
        let c = course(1);
        let mut m = Linear::new(8, 5).unwrap();
        let mut empty = c.tasks[0].clone();
        empty.train.clear();
        assert!(matches!(
            train(&mut m, &empty, &Schedule::default()),
            Err(ContinualError::Empty { what: "task train split" })
        ));
        assert_eq!(m.w, vec![0.0; 40], "a task with no examples still moved the model");
        let mut mislabelled = c.tasks[0].clone();
        mislabelled.train[7].label = 5;
        assert!(matches!(
            train(&mut m, &mislabelled, &Schedule::default()),
            Err(ContinualError::Label { label: 5, n_classes: 5 })
        ));
    }

    /// `sgd_step` hands the consolidator the gradient of the TASK objective — weight decay
    /// included, consolidation penalty excluded — because synaptic intelligence integrates that
    /// gradient along the path, and a penalty folded into it would let the method certify its own
    /// importance estimate.
    ///
    /// The routing is invisible from outside: the penalty reaches the step either way, and only
    /// the path integral of a THIRD task would read the difference. Here the step is taken on its
    /// own, so both vectors are visible at once.
    #[test]
    fn the_step_keeps_the_consolidation_penalty_out_of_the_gradient_it_reports() {
        let mut m = Linear::new(1, 2).unwrap();
        m.w = vec![0.25, -0.75];
        let mut cons = super::Consolidator::new(Consolidation::Ewc { lambda: 2.0 }, 4);
        cons.omega = vec![1.0, 0.5, 0.25, 0.125];
        let sched = Schedule { epochs: 1, lr: 0.1, weight_decay: 0.01, seed: 0 };
        let e = Example { x: vec![1.0], label: 0 };
        let p = m.probabilities(&e.x).unwrap();
        let before = m.clone();
        let mut flat = [0.0f64; 4];
        let mut data_grad = [0.0f64; 4];
        let mut grad = [0.0f64; 4];
        let mut delta = [0.0f64; 4];
        super::sgd_step(
            &mut m,
            &e,
            &sched,
            &cons,
            &mut flat,
            &mut data_grad,
            &mut grad,
            &mut delta,
        )
        .unwrap();
        // The data gradient by hand: residual times feature, plus the decay on the weight.
        let (d0, d1) = (p[0] - 1.0, p[1]);
        assert_eq!(
            data_grad,
            [d0 * 1.0 + 0.01 * before.w[0], d1 * 1.0 + 0.01 * before.w[1], d0, d1]
        );
        // The penalised gradient is that plus `lambda * omega * (theta - anchor)`, and the anchor
        // is still the origin.
        assert_eq!(grad[0], data_grad[0] + 2.0 * 1.0 * (before.w[0] - 0.0));
        assert_eq!(grad[1], data_grad[1] + 2.0 * 0.5 * (before.w[1] - 0.0));
        // And the step descends the penalised gradient, not the reported one.
        assert_eq!(delta[0], -0.1 * grad[0]);
        assert_eq!(m.w[0], before.w[0] + delta[0]);
    }

    /// The divergence guard is `is_finite`, not `is_nan`.
    ///
    /// An infinity survives the arithmetic that produced it and only becomes `NaN` one step
    /// later, when it meets the softmax — so a guard watching only `NaN` still returns
    /// [`ContinualError::Diverged`], one step late, naming a different parameter and a different
    /// value. Both divergence tests in this suite read only the variant, so the late refusal
    /// matched them exactly. Here the step is taken on its own and the value is visible.
    #[test]
    fn an_infinite_parameter_is_refused_at_the_step_that_produced_it() {
        let mut m = Linear::new(1, 2).unwrap();
        let cons = super::Consolidator::new(Consolidation::None, 4);
        let sched = Schedule { epochs: 1, lr: 10.0, weight_decay: 0.0, seed: 0 };
        // A finite feature whose gradient overflows once the learning rate multiplies it: the
        // residual is exactly -0.5 on a zero-initialised readout, 0.5e308 is finite, and ten
        // times that is not.
        let e = Example { x: vec![1e308], label: 0 };
        let mut flat = [0.0f64; 4];
        let mut data_grad = [0.0f64; 4];
        let mut grad = [0.0f64; 4];
        let mut delta = [0.0f64; 4];
        let err = super::sgd_step(
            &mut m,
            &e,
            &sched,
            &cons,
            &mut flat,
            &mut data_grad,
            &mut grad,
            &mut delta,
        )
        .unwrap_err();
        match err {
            ContinualError::Diverged { parameter, value } => {
                assert_eq!(parameter, 0, "the guard fired on a later parameter");
                assert!(value.is_infinite(), "the guard waited for a NaN: {value}");
            }
            other => panic!("wrong refusal: {other:?}"),
        }
        // And the model was not written back, so a caller that ignores the error still holds
        // finite parameters.
        assert_eq!(m.w, vec![0.0, 0.0]);
        assert_eq!(m.b, vec![0.0, 0.0]);
    }

    /// The two rates are scaled by their own level-0 probability: flipping by `q0`, deepening by
    /// `p0`. Every cascade this suite asserts a probability on is [`Cascade::fusi_2005`], where
    /// `q0 == p0 == 1.0` and the two are the same number at every level.
    #[test]
    fn the_flip_rate_is_scaled_by_q0_and_the_deepening_rate_by_p0() {
        let c = Cascade { depth: 3, q0: 0.25, p0: 0.75, x: 0.5 };
        assert_eq!(c.flip_probability(0), Some(0.25));
        assert_eq!(c.flip_probability(1), Some(0.125));
        assert_eq!(c.flip_probability(2), Some(0.0625));
        assert_eq!(c.flip_probability(3), None);
        assert_eq!(c.deepen_probability(0), Some(0.75));
        assert_eq!(c.deepen_probability(1), Some(0.375));
        // The deepest level has nowhere deeper to go, which is not the same as "not a level".
        assert_eq!(c.deepen_probability(2), Some(0.0));
        assert_eq!(c.deepen_probability(3), None);
    }

    /// The reducible chain, which is the only thing that reaches `stationary_by_iteration`.
    ///
    /// Flux balance succeeds for every other cascade in this module, so the fallback ran in no
    /// test at all — and the one `x == 0` fixture that exists calls only `transition_matrix`.
    /// With `x == 0` every level past the first is absorbing: from `(+, 0)` a potentiating
    /// stimulus deepens with certainty and a depressing one flips with certainty, so the level-0
    /// mass halves every step and all of it lands on level 1, while levels 2 and up keep whatever
    /// the start vector gave them. From the uniform start that limit is exactly
    /// `[0, 1/3, 1/6, 0, 1/3, 1/6]` — a distribution, which is what a start vector that is not
    /// normalised would not produce.
    #[test]
    fn the_reducible_chain_falls_back_to_a_normalised_limit_from_the_uniform_start() {
        let c = Cascade { depth: 3, q0: 1.0, p0: 1.0, x: 0.0 };
        let pi = c.stationary().unwrap();
        assert_eq!(pi.len(), 6);
        assert!(pi.iter().all(|&v| v >= 0.0), "{pi:?}");
        // The iteration stops at an absolute residual of 1e-15 and the level-0 mass halves every
        // step, so every entry is within that of its limit; 1e-14 is ten times the stopping
        // residual and a thousandth of the smallest entry.
        assert!((pi.iter().sum::<f64>() - 1.0).abs() < 1e-14, "sums to {}", pi.iter().sum::<f64>());
        let third = 1.0 / 3.0;
        let sixth = 1.0 / 6.0;
        for (k, want) in [(0, 0.0), (1, third), (2, sixth), (3, 0.0), (4, third), (5, sixth)] {
            assert!((pi[k] - want).abs() < 1e-14, "state {k}: {} against {want}", pi[k]);
        }
    }

    /// [`CascadeEnsemble::new`] starts every synapse at `(+, 0)` — at the SHALLOWEST level, which
    /// is the initial condition the exact chain deliberately does not use.
    ///
    /// The suite reads a fresh ensemble's `len`, its `is_empty` and its signal, and all three are
    /// the same at any level: the polarity is what the signal is made of. The level shows up one
    /// stimulus later, because [`Cascade::fusi_2005`] flips a level-0 synapse with probability
    /// exactly 1.0 and `Rng::next_f64` lives in `[0, 1)`, so a single depressing stimulus flips
    /// every synapse in the population. At level 1 it would flip half of them.
    #[test]
    fn a_fresh_ensemble_starts_at_the_shallowest_level_where_one_stimulus_flips_it() {
        let c = Cascade::fusi_2005(4);
        let mut e = CascadeEnsemble::new(c, 256, 0xFE51).unwrap();
        assert_eq!(e.signal(), 1.0);
        e.stimulate(false);
        assert_eq!(e.signal(), -1.0, "a fresh synapse was not at the level that always flips");
    }

    /// The test-only elimination oracle projects its solution onto the polarity-symmetric
    /// subspace, and that projection is load-bearing rather than tidy.
    ///
    /// `transition_matrix` gives the two stimulus signs equal probability and shares `q0`, `p0`
    /// and `x` between them, so the chain's stationary distribution is EXACTLY symmetric under
    /// polarity. The oracle's own solve is not: the antisymmetric direction is the slowest
    /// eigenmode and it is precisely where `f64` cannot resolve stationarity. Without the
    /// projection this implementation measures an asymmetry of 2.2e-16 at depth 2 and 2.9e-11 at
    /// depth 23 — and the test that compares the oracle against the flux-balance recursion runs
    /// only to depth 15, at 1e-12, where that error is four orders under the tolerance. An exact
    /// equality between the two halves sees it at every depth.
    #[test]
    fn the_elimination_oracle_returns_an_exactly_polarity_symmetric_vector() {
        for depth in [2usize, 3, 7, 10, 15, 18, 20, 23] {
            let c = Cascade::fusi_2005(depth);
            let n = 2 * depth;
            let m = c.transition_matrix().unwrap();
            let g = c.generator_matrix().unwrap();
            let pi = super::stationary_direct(&g, &m, n).expect("irreducible");
            for k in 0..depth {
                assert_eq!(pi[k], pi[depth + k], "depth {depth} level {k}");
            }
        }
        // And for parameterisations that are not fusi's, so the symmetry is a property of the
        // equiprobable stimulus signs rather than of `q0 == p0 == 1` and `x == 1/2`.
        for c in [
            Cascade { depth: 12, q0: 0.3, p0: 0.8, x: 0.6 },
            Cascade { depth: 9, q0: 0.9, p0: 0.2, x: 0.85 },
        ] {
            let n = 2 * c.depth;
            let m = c.transition_matrix().unwrap();
            let g = c.generator_matrix().unwrap();
            let pi = super::stationary_direct(&g, &m, n).expect("irreducible");
            for k in 0..c.depth {
                assert_eq!(pi[k], pi[c.depth + k], "{c:?} level {k}");
            }
        }
    }

    /// The importance and the anchor are taken from the model the NEXT task starts from — after
    /// the homeostat has rescaled it, not before.
    ///
    /// No test in this suite combined a homeostat with a consolidation method: every homeostat
    /// fixture runs [`Consolidation::None`] and every consolidation fixture runs no homeostat, so
    /// the order of the two statements at the task boundary was unobservable. Taking the anchor
    /// first leaves the next task starting at a point the penalty is already pulling it away
    /// from, and estimates the importance at a scale the learner never visits.
    ///
    /// It is measurable because the empirical Fisher is scale-dependent. Shrinking every class
    /// row to a norm of 2 un-saturates the posteriors and multiplies the Fisher, so a `lambda` of
    /// 150 — comfortable at this model's own scale, where it retains 0.812 — is then stiffer than
    /// the learning rate can integrate and the run is refused. With the anchor taken first the
    /// same run completes, and completes at the same retention of 0.532 for every target norm,
    /// because nothing the homeostat did reached the importance at all.
    #[test]
    fn the_importance_is_estimated_after_the_homeostat_has_rescaled_the_model() {
        let sched = Schedule::default();
        let scaled_to = |target_norm: f64| {
            run(
                &course(2),
                &Protocol {
                    consolidation: Consolidation::Ewc { lambda: 150.0 },
                    homeostasis: Some(Homeostat { target_norm, rate: 1.0 }),
                    ..Protocol::plain(sched)
                },
            )
        };
        assert!(matches!(scaled_to(2.0), Err(ContinualError::Diverged { .. })));
        // And where it does complete, what is retained depends on the target norm — which it
        // cannot if the importance was estimated before the rescaling. Measured at 1.000 for a
        // target of 4 and 0.436 for a target of 10.
        assert_eq!(scaled_to(4.0).unwrap().retained(0), Some(1.0));
        let wide = scaled_to(10.0).unwrap().retained(0).unwrap();
        assert!(wide < 0.50, "retention at a target norm of 10: {wide}");
    }


    /// Backward transfer reads the last accuracy row no further than column `t - 2`, so a row that
    /// is exactly one entry short still produces the mean, and a non-finite diagonal entry never
    /// enters the sum.
    ///
    /// `a_ragged_accuracy_matrix_refuses_rather_than_panicking` cannot see the loop bound at all:
    /// both of its short rows are `Vec::new()`, so `last.get(j)?` fails at `j = 0` for any bound
    /// whatsoever. The hole was a missing LENGTH rather than a missing case — the only row that
    /// separates `0..t - 1` from `0..t` is one of length exactly `t - 1`. The second fixture pins
    /// the other half: the recorded argument that the extra term is `x - x == 0.0` holds only for
    /// finite `x`, and `Results::r` is a public field that this module's own doc says a caller
    /// fills by hand, so nothing enforces that precondition.
    #[test]
    fn backward_transfer_reads_no_further_than_the_second_to_last_column() {
        // t = 2, last row of length exactly t - 1. The mean runs over j = 0 alone and every
        // quantity in it is a dyadic fraction, so the value is exact: (0.25 - 0.5) / 1.
        let short_by_one = Results {
            r: vec![vec![0.5, 0.3], vec![0.25]],
            baseline: vec![0.2, 0.2],
            cost: Cost::default(),
        };
        assert_eq!(short_by_one.backward_transfer(), Some(-0.25));
        // Full width, but the diagonal term the wider loop would add is `inf - inf`, which is NaN
        // rather than the zero the argument assumes.
        let infinite_diagonal = Results {
            r: vec![vec![0.5, 0.5], vec![0.25, f64::INFINITY]],
            baseline: vec![0.2, 0.2],
            cost: Cost::default(),
        };
        assert_eq!(infinite_diagonal.backward_transfer(), Some(-0.25));
    }

    /// The untrained baseline row is **measured** on each task's own test split, not read off
    /// `Task::chance`, and the two differ the moment the split is not balanced.
    ///
    /// `the_untrained_baseline_is_exactly_chance` cannot see the difference, and worse: it compares
    /// `res.baseline[j]` against `c.tasks[j].chance`, so replacing the measurement by the constant
    /// turns its own assertion into `chance[j] == chance[j]`, a tautology. Every curriculum that
    /// suite reaches comes from `crate::tasks`, whose test splits hold the same number of examples
    /// per class, so measurement and constant coincide there by construction. `Curriculum` is built
    /// here through the public `from_examples`, which validates widths and labels and says nothing
    /// about class balance — the hole was that no fixture ever presented an unbalanced split.
    ///
    /// Measured: on a two-class test split of four examples, one labelled 0 and three labelled 1,
    /// the zero-initialised readout answers class 0 everywhere (every logit is `+0.0`, and
    /// `Linear::predict` breaks that tie to the lowest index), so its accuracy is exactly 1/4 while
    /// `Task::chance` is 1/2.
    #[test]
    fn the_untrained_baseline_is_measured_rather_than_declared() {
        let train = vec![
            Example { x: vec![1.0, 0.0], label: 0 },
            Example { x: vec![0.0, 1.0], label: 1 },
        ];
        let test = vec![
            Example { x: vec![1.0, 0.0], label: 0 },
            Example { x: vec![0.0, 1.0], label: 1 },
            Example { x: vec![0.0, 1.0], label: 1 },
            Example { x: vec![0.0, 1.0], label: 1 },
        ];
        let c = Curriculum::from_examples("unbalanced", train, test, 2, 1, 7).unwrap();
        assert_eq!(c.tasks[0].chance, 0.5);
        // Both sides of the mutation, before the training loop is involved at all.
        assert_eq!(evaluate(&Linear::new(2, 2).unwrap(), &c).unwrap(), vec![0.25]);
        let res = run(&c, &Protocol::plain(Schedule::default())).unwrap();
        assert_eq!(res.baseline, vec![0.25]);
    }

    /// The learning probability is conditioned on the mass of the **opposing** polarity, and on a
    /// reducible cascade the two halves of the stationary distribution are not bit-identical, so
    /// the choice of half is observable — and conditioning on the wrong one returns a number above
    /// one.
    ///
    /// `the_stationary_distribution_is_exactly_symmetric_under_polarity` cannot see it: both of its
    /// parameter sets have `p0 > 0` and `x > 0`, which is exactly the condition under which
    /// `Cascade::stationary_by_flux_balance` succeeds and writes `pi[k]` and `pi[d + k]` from the
    /// same `f64`. The hole was the OTHER path — `p0 == 0` makes every `deepen_k` zero, flux
    /// balance returns `None`, and `Cascade::stationary` falls back to the power iteration, whose
    /// column sums are accumulated in index order so that the polarity-image column lands one ulp
    /// away. That test also asserts a `1e-15` tolerance rather than the bit equality its name
    /// claims, so it could not have pinned this even on the flux-balance path.
    ///
    /// Measured on `Cascade { depth: 3, q0: 1.0, p0: 0.0, x: 1.0 }`, whose power iteration
    /// converges in 51 steps: `pi[3..].iter().sum()` is `0.499_999_999_999_999_94` and
    /// `pi[..3].iter().sum()` is `0.499_999_999_999_999_9`. The numerator is accumulated from
    /// `pi[d + k]` in the same order as the first of those, so the quotient is exactly 1.0;
    /// against the other half it is `1.000_000_000_000_000_2`, a probability above one.
    #[test]
    fn the_learning_probability_is_conditioned_on_the_opposing_polarity_half() {
        for &(depth, x) in &[(3usize, 1.0f64), (4, 0.5), (8, 0.5)] {
            let c = Cascade { depth, q0: 1.0, p0: 0.0, x };
            // The reducible parameterisation: flux balance refuses it and the iteration answers.
            let p = c.learning_probability().unwrap();
            assert!(p <= 1.0, "depth {depth}: a probability exceeded one: {p}");
        }
        // The exact witness at depth 3: numerator and denominator are the same three f64s summed
        // in the same order, so the quotient is 1.0 and not merely near it.
        assert_eq!(Cascade { depth: 3, q0: 1.0, p0: 0.0, x: 1.0 }.learning_probability().unwrap(), 1.0);
    }

}
